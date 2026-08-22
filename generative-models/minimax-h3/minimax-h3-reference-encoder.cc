#include "generative-models/minimax-h3/minimax-h3-reference-encoder.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "generative-models/minimax-h3/metal-minimax-h3-audio-vae.h"
#include "generative-models/minimax-h3/metal-minimax-h3-video-vae.h"
#include "generative-models/minimax-h3/minimax-h3-references.h"
#include "generative-models/minimax-h3/minimax-h3-text-encoder.h"
#include "generative-models/qwen3/metal-qwen-vision.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace vpipe {
namespace genai {
namespace minimax_h3 {

namespace {

// The video VAE's pixel convention is IMAGENET-normalized, not [-1, 1]:
// the reference encoder's first act is `x/255` then `(. - mean)/std`.
// Feeding it [-1, 1] makes the input ~4.4x too wide and the conditioning
// latent silently wrong. vae-decode undoes exactly this on the way out.
constexpr float kImagenetMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kImagenetStd[3]  = {0.229f, 0.224f, 0.225f};

std::uint16_t
f32_to_bf16_(float v)
{
  std::uint32_t u;
  std::memcpy(&u, &v, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

float
bf16_to_f32_(std::uint16_t h)
{
  const std::uint32_t u = (std::uint32_t)h << 16;
  float v;
  std::memcpy(&v, &u, 4);
  return v;
}

// `num_frames` planar [3,H,W] u8 planes -> the bf16, ImageNet-normalized
// [3, T, H, W] buffer the video VAE encoder takes.
metal_compute::SharedBuffer
to_vae_input_(metal_compute::MetalCompute* mc, const std::uint8_t* rgb,
              int num_frames, int H, int W)
{
  const std::size_t plane = (std::size_t)H * W;
  const std::size_t n = (std::size_t)3 * num_frames * plane;
  metal_compute::SharedBuffer buf = mc->make_shared_buffer(n * 2);
  if (buf.empty()) { return buf; }
  auto* d = static_cast<std::uint16_t*>(buf.contents());
  for (int c = 0; c < 3; ++c) {
    const float pm = kImagenetMean[c], ps = kImagenetStd[c];
    for (int t = 0; t < num_frames; ++t) {
      // The source is frame-major and the VAE input is channel-major,
      // so this is a transpose, not a copy.
      const std::uint8_t* src =
          rgb + ((std::size_t)t * 3 + c) * plane;
      std::uint16_t* dst =
          d + ((std::size_t)c * num_frames + t) * plane;
      for (std::size_t i = 0; i < plane; ++i) {
        dst[i] = f32_to_bf16_(((float)src[i] / 255.0f - pm) / ps);
      }
    }
  }
  return buf;
}

// The MEAN half of the VAE's moments, whitened and packed into DiT rows.
//
// Two deliberate departures from the reference, both shared with the
// `fl2va` keyframe path this mirrors (stages/vae-encode-stage.cc):
// the posterior is not SAMPLED (the reference draws it under a fixed
// seed 42; an anchor whose whole job is to be exact does not want noise
// in it), and the latent is not rounded through float16 first.
void
pack_condition_rows_(const metal_compute::SharedBuffer& moments, int z,
                     int lt, int lh, int lw, int patch_h, int patch_w,
                     const std::vector<float>& mean,
                     const std::vector<float>& std_,
                     std::vector<float>* rows)
{
  const auto* mp = static_cast<const std::uint16_t*>(moments.contents());
  const std::size_t vox = (std::size_t)lt * lh * lw;
  const int gh = lh / patch_h, gw = lw / patch_w;
  const int PE = z * patch_h * patch_w;
  const bool whiten =
      (int)mean.size() == z && (int)std_.size() == z;
  const std::size_t base = rows->size();
  rows->resize(base + (std::size_t)lt * gh * gw * PE);
  float* out = rows->data() + base;
  for (int t = 0; t < lt; ++t) {
    for (int cell = 0; cell < gh * gw; ++cell) {
      float* row = out + ((std::size_t)t * gh * gw + cell) * PE;
      const int by = (cell / gw) * patch_h;
      const int bx = (cell % gw) * patch_w;
      for (int c = 0; c < z; ++c) {
        const float mu = whiten ? mean[(std::size_t)c] : 0.0f;
        const float sd = whiten ? std_[(std::size_t)c] : 1.0f;
        for (int y = 0; y < patch_h; ++y) {
          for (int x = 0; x < patch_w; ++x) {
            const std::size_t k = (std::size_t)c * vox +
                                  ((std::size_t)t * lh + (by + y)) * lw +
                                  (bx + x);
            const float v = bf16_to_f32_(mp[k]);
            row[((std::size_t)c * patch_h + y) * patch_w + x] =
                sd != 0.0f ? (v - mu) / sd : (v - mu);
          }
        }
      }
    }
  }
}

}  // namespace

bool
condition_frame_indices(int num_frames, double fps, double sample_fps,
                        int temporal_patch, std::vector<int>* indices,
                        std::vector<float>* block_seconds, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (indices == nullptr || block_seconds == nullptr) {
    return fail("null argument");
  }
  if (num_frames <= 0 || !(fps > 0.0) || !(sample_fps > 0.0) ||
      temporal_patch <= 0) {
    return fail("the conditioner's frame sampling needs positive rates");
  }
  indices->clear();
  block_seconds->clear();

  const double stride = fps / sample_fps;
  // `std::nearbyint` under the default rounding mode is half-to-even,
  // which is Python's `round` -- the two agree on the .5 the stride
  // lands on when `fps / sample_fps` is a half-integer.
  auto rnd = [](double v) { return (long long)std::nearbyint(v); };
  double cursor = 0.0;
  while (rnd(cursor) < (long long)num_frames) {
    const long long i = rnd(cursor);
    if (indices->empty() || i > (long long)indices->back()) {
      indices->push_back((int)i);
    }
    cursor += stride;
  }
  if ((int)indices->size() < temporal_patch) {
    const long long minimum = rnd((double)(temporal_patch - 1) * stride) + 1;
    return fail("a reference video is read at " +
                std::to_string(sample_fps) + " fps and its sampled frames "
                "are merged in groups of " + std::to_string(temporal_patch) +
                ", so it must run at least " + std::to_string(minimum) +
                " frames, got " + std::to_string(num_frames));
  }

  // One timestamp per sampled frame, the LAST repeated until the count
  // fills a whole number of merged groups, then a group's label is the
  // mean of its first and last.
  std::vector<double> ts;
  ts.reserve(indices->size() + (std::size_t)temporal_patch);
  for (std::size_t i = 0; i < indices->size(); ++i) {
    ts.push_back((double)i / sample_fps);
  }
  while (ts.size() % (std::size_t)temporal_patch != 0) {
    ts.push_back(ts.back());
  }
  for (std::size_t i = 0; i < ts.size(); i += (std::size_t)temporal_patch) {
    block_seconds->push_back(
        (float)((ts[i] + ts[i + (std::size_t)temporal_patch - 1]) / 2.0));
  }
  return true;
}

bool
validate_reference_request(const std::vector<MediaReference>& refs,
                           const ReferenceLimits& limits, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (refs.empty()) {
    return fail("a ref2va request needs at least one reference; a "
                "text-only request is the t2va workflow");
  }
  int n_img = 0, n_vid = 0, n_aud = 0;
  for (const MediaReference& r : refs) {
    switch (r.kind) {
      case MediaReference::Kind::kImage: ++n_img; break;
      case MediaReference::Kind::kVideo: ++n_vid; break;
      case MediaReference::Kind::kAudio: ++n_aud; break;
    }
  }
  if (n_img > limits.max_images) {
    return fail("MiniMax-H3 accepts at most " +
                std::to_string(limits.max_images) + " image references, got " +
                std::to_string(n_img));
  }
  if (n_vid > limits.max_videos) {
    return fail("MiniMax-H3 accepts at most " +
                std::to_string(limits.max_videos) + " video references, got " +
                std::to_string(n_vid));
  }
  if (n_aud > limits.max_audios) {
    return fail("MiniMax-H3 accepts at most " +
                std::to_string(limits.max_audios) + " audio references, got " +
                std::to_string(n_aud));
  }
  if ((int)refs.size() > limits.max_references) {
    return fail("MiniMax-H3 accepts at most " +
                std::to_string(limits.max_references) +
                " references in total, got " + std::to_string(refs.size()));
  }
  if (n_img == 0 && n_vid == 0) {
    return fail("an audio reference has to be paired with at least one "
                "image or video reference and cannot be used on its own");
  }
  return true;
}

bool
encode_references(const std::vector<MediaReference>& refs,
                  std::string_view prompt, const ReferencePlan& plan,
                  const ReferenceEncoders& models, EncodedReferences* out,
                  std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (out == nullptr) { return fail("null argument"); }
  if (plan.target_frames <= 0) {
    return fail("the generated frame count has to be positive");
  }
  if (plan.patch_h <= 0 || plan.patch_w <= 0) {
    return fail("the DiT patch has to be positive");
  }

  EncodedReferences r;
  // The tower results are held for the whole traversal: the
  // presentation points INTO them (one buffer per reference, indexed by
  // temporal cell), so they have to outlive the conditioner call.
  std::vector<MetalQwenVisionEncoder::Result> vision;
  std::vector<std::vector<float>> block_seconds;
  std::vector<int> vision_of;    // reference -> index into `vision`, or -1
  vision.reserve(refs.size());
  block_seconds.reserve(refs.size());

  metal_compute::MetalCompute* mc = models.mc;

  const double max_seconds = (double)plan.target_frames / kFps;

  for (std::size_t i = 0; i < refs.size(); ++i) {
    const MediaReference& m = refs[i];
    const std::string where = "reference " + std::to_string(i + 1);
    Reference L;
    vision_of.push_back(-1);

    // ---- the soundtrack, FIRST ---------------------------------------
    // Its rows are packed immediately before this reference's video
    // rows, so it is encoded in that order too -- one traversal, one
    // order, nothing to keep in step afterwards.
    if (m.has_audio()) {
      if (models.audio_vae == nullptr) {
        return fail(where + " carries a soundtrack but no audio VAE is "
                            "loaded");
      }
      const int n_samp = (int)(m.pcm.size() / (std::size_t)m.channels);
      std::vector<float> pcm;
      int kept = 0;
      std::string nerr;
      if (!normalize_audio_reference(m.pcm.data(), m.channels, n_samp,
                                     m.sample_rate, max_seconds, &pcm, &kept,
                                     &nerr)) {
        return fail(where + ": " + nerr);
      }
      std::vector<float> lat;
      int frames = 0;
      std::string aerr;
      if (!models.audio_vae->encode(pcm.data(), 2, kept, &lat, &frames,
                                    &aerr)) {
        return fail(where + ": the audio VAE encode failed (" +
                    (aerr.empty() ? "unknown error" : aerr) + ")");
      }
      const auto& ac = models.audio_vae->config();
      const int AC = ac.latent_channels;
      r.audio_row_elems = AC;
      const bool whiten = (int)ac.latents_mean.size() == AC &&
                          (int)ac.latents_std.size() == AC;
      const std::size_t base = r.audio_rows.size();
      r.audio_rows.resize(base + lat.size());
      for (std::size_t k = 0; k < lat.size(); ++k) {
        const std::size_t c = k % (std::size_t)AC;
        const float mu = whiten ? ac.latents_mean[c] : 0.0f;
        const float sd = whiten ? ac.latents_std[c] : 1.0f;
        r.audio_rows[base + k] =
            sd != 0.0f ? (lat[k] - mu) / sd : (lat[k] - mu);
      }
      L.num_audio_latents = frames;
    }

    // ---- the pixels ---------------------------------------------------
    if (m.kind != MediaReference::Kind::kAudio) {
      if (m.num_frames <= 0 || m.height <= 0 || m.width <= 0) {
        return fail(where + " has no pixels");
      }
      const std::size_t want =
          (std::size_t)m.num_frames * 3 * m.height * m.width;
      if (m.rgb.size() != want) {
        return fail(where + " carries " + std::to_string(m.rgb.size()) +
                    " bytes of pixels, expected " + std::to_string(want));
      }
      std::vector<std::uint8_t> px;
      int nf = 0, th = 0, tw = 0;
      std::string nerr;
      if (m.kind == MediaReference::Kind::kImage) {
        if (!normalize_image_reference(m.rgb.data(), m.height, m.width,
                                       plan.reference_image_short_edge,
                                       plan.canvas_multiple, &px, &th, &tw,
                                       &nerr)) {
          return fail(where + ": " + nerr);
        }
        nf = 1;
      } else {
        if (!normalize_video_reference(m.rgb.data(), m.num_frames, m.height,
                                       m.width, m.fps, plan.target_frames,
                                       plan.canvas_multiple,
                                       plan.canvas_short_edge,
                                       plan.canvas_max_pixels, &px, &nf, &th,
                                       &tw, kFps, &nerr)) {
          return fail(where + ": " + nerr);
        }
      }

      // The CONDITIONER's read: 2 fps for a video, the whole (single)
      // frame for an image. Its canvas is the tower's own smart-resize,
      // not this one -- the two models genuinely see different pixels.
      if (models.vision != nullptr) {
        MetalQwenVisionEncoder::Result res;
        std::vector<float> secs;
        if (m.kind == MediaReference::Kind::kImage) {
          res = models.vision->encode(px.data(), th, tw);
        } else {
          std::vector<int> idx;
          if (!condition_frame_indices(nf, kFps, plan.video_sample_fps,
                                       plan.temporal_patch, &idx, &secs,
                                       &nerr)) {
            return fail(where + ": " + nerr);
          }
          const std::size_t frame_bytes = (std::size_t)3 * th * tw;
          std::vector<std::uint8_t> sampled(frame_bytes * idx.size());
          for (std::size_t k = 0; k < idx.size(); ++k) {
            std::memcpy(sampled.data() + k * frame_bytes,
                        px.data() + (std::size_t)idx[k] * frame_bytes,
                        frame_bytes);
          }
          res = models.vision->encode_video(sampled.data(), (int)idx.size(),
                                            th, tw);
        }
        if (res.embeddings.empty()) {
          return fail(where + ": the vision tower produced nothing");
        }
        vision_of.back() = (int)vision.size();
        vision.push_back(std::move(res));
        block_seconds.push_back(std::move(secs));
      }

      // The VAE's read: the full 24 fps clip at MiniMax-H3's canvas.
      if (models.video_vae != nullptr) {
        if (mc == nullptr) { return fail("the video VAE has no compute"); }
        const auto& vc = models.video_vae->config();
        int use = nf;
        if (m.kind == MediaReference::Kind::kVideo) {
          // Snap DOWN to `17n + 5` so the VAE encodes without padding.
          // Only bites when the reference is SHORTER than the target,
          // whose own count already has that form. The clamp is what
          // Python's slice does for a clip shorter than one chunk.
          const int fpc = vc.clip_length;
          const int lpc = vc.tokens_per_chunk();
          if (fpc > 0 && lpc > 0) {
            use = std::max(1, (nf - lpc) / fpc) * fpc + lpc;
            use = std::min(use, nf);
          }
        }
        metal_compute::SharedBuffer in =
            to_vae_input_(mc, px.data(), use, th, tw);
        if (in.empty()) { return fail(where + ": VAE input alloc failed"); }
        int lf = 0;
        std::string eerr;
        metal_compute::SharedBuffer mom =
            models.video_vae->encode_video(in, use, th, tw, &lf, &eerr);
        if (mom.empty() || lf <= 0) {
          return fail(where + ": the video VAE encode failed (" +
                      (eerr.empty() ? "unknown error" : eerr) + ")");
        }
        const int lh = th / vc.patch, lw = tw / vc.patch;
        if ((lh % plan.patch_h) != 0 || (lw % plan.patch_w) != 0) {
          return fail(where + " encodes to a " + std::to_string(lh) + "x" +
                      std::to_string(lw) + " latent, which the DiT's " +
                      std::to_string(plan.patch_h) + "x" +
                      std::to_string(plan.patch_w) + " patch does not "
                      "divide");
        }
        r.video_row_elems = vc.z_channels * plan.patch_h * plan.patch_w;
        pack_condition_rows_(mom, vc.z_channels, lf, lh, lw, plan.patch_h,
                             plan.patch_w, vc.latents_mean, vc.latents_std,
                             &r.video_rows);
        L.num_latent_frames = lf;
        L.latent_height     = lh;
        L.latent_width      = lw;
      }
    }

    L.kind = m.kind == MediaReference::Kind::kImage ? Reference::Kind::kImage
             : m.kind == MediaReference::Kind::kVideo
                 ? Reference::Kind::kVideo
                 : Reference::Kind::kAudio;
    r.layout.push_back(L);
    if (models.progress) {
      // +1: the presentation below is a step of its own. See the
      // declaration of ReferenceEncoders::progress.
      models.progress((int)i + 1, (int)refs.size() + 1);
    }
  }

  // ---- the presentation ------------------------------------------------
  // One conditioner call over the whole request: the labels are numbered
  // across references, so this cannot be done a reference at a time.
  if (models.text != nullptr) {
    std::vector<MiniMaxH3TextEncoder::Reference> pres;
    pres.reserve(refs.size());
    for (std::size_t i = 0; i < refs.size(); ++i) {
      MiniMaxH3TextEncoder::Reference p;
      p.kind = refs[i].kind == MediaReference::Kind::kImage
                   ? MiniMaxH3TextEncoder::Reference::Kind::kImage
               : refs[i].kind == MediaReference::Kind::kVideo
                   ? MiniMaxH3TextEncoder::Reference::Kind::kVideo
                   : MiniMaxH3TextEncoder::Reference::Kind::kAudio;
      p.has_audio = refs[i].has_audio();
      const int vi = vision_of[i];
      if (vi >= 0) {
        p.vision = &vision[(std::size_t)vi];
        p.block_seconds = block_seconds[(std::size_t)vi];
      }
      pres.push_back(std::move(p));
    }
    std::string cerr;
    r.conditioning = models.text->encode_references(pres, prompt,
                                                    &r.token_tags,
                                                    &r.n_tokens, &cerr);
    if (r.conditioning.empty()) {
      return fail("the conditioner failed (" +
                  (cerr.empty() ? "unknown error" : cerr) + ")");
    }
  }

  // The presentation is done -- reported here rather than inside the
  // `models.text` branch above, because reaching this line is what makes
  // the request finished whether or not a conditioner was attached.
  if (models.progress) {
    models.progress((int)refs.size() + 1, (int)refs.size() + 1);
  }

  *out = std::move(r);
  return true;
}

}  // namespace minimax_h3
}  // namespace genai
}  // namespace vpipe

#include "generative-models/minimax-h3/minimax-h3-reference-encoder.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "generative-models/minimax-h3/metal-minimax-h3-audio-vae.h"
#include "generative-models/minimax-h3/metal-minimax-h3-video-vae.h"
#include "generative-models/minimax-h3/minimax-h3-references.h"
#include "generative-models/minimax-h3/minimax-h3-text-encoder.h"
#include "generative-models/qwen3/metal-qwen-vision.h"

#include <algorithm>
#include <chrono>
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

void
pack_condition_rows_raw_(const std::uint16_t* mp, int z, int lt, int lh,
                         int lw, int patch_h, int patch_w,
                         const std::vector<float>& mean,
                         const std::vector<float>& std_,
                         std::vector<float>* rows);

void
pack_condition_rows_(const metal_compute::SharedBuffer& moments, int z,
                     int lt, int lh, int lw, int patch_h, int patch_w,
                     const std::vector<float>& mean,
                     const std::vector<float>& std_,
                     std::vector<float>* rows)
{
  pack_condition_rows_raw_(
      static_cast<const std::uint16_t*>(moments.contents()), z, lt, lh, lw,
      patch_h, patch_w, mean, std_, rows);
}

// The CONDITION NOISE AUGMENTATION the released checkpoint was trained
// with: `0.999*z + 0.001*noise` on every visual conditioning row.
//
// It is not a denoising step and not a regulariser -- it is the
// distribution the anchors were trained in, so rows handed over exactly
// clean are off-distribution. ComfyUI applies it in `_cond_video_rows`
// (`VISUAL_COND_TIMESTEP = 0.999`) and diffusers documents the same
// number as `keyframe_noise_aug`, saying so outright: "the released
// model was trained with its anchors very slightly noised, so
// conditioning on exactly t = 1.0 is off-distribution".
//
// A deliberately NON variance-preserving lerp, matching the reference:
// the magnitude lands at 0.999x rather than being sqrt-corrected. The
// stream restarts per condition, as ComfyUI's does, so a reference
// packs the same rows wherever it sits in the request.
//
// AUDIO rows are left alone: the reference holds them at 1.0.
void
augment_condition_rows_(std::vector<float>* rows, std::size_t from,
                        double aug, std::uint64_t seed)
{
  if (rows == nullptr || !(aug < 1.0) || aug <= 0.0) { return; }
  std::uint64_t s = seed + 0x9e3779b97f4a7c15ull;
  auto next_normal = [&]() {
    auto u = [&]() {
      std::uint64_t z = (s += 0x9e3779b97f4a7c15ull);
      z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
      z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
      z ^= (z >> 31);
      return ((double)(z >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    };
    const double u1 = u(), u2 = u();
    return std::sqrt(-2.0 * std::log(u1)) *
           std::cos(6.283185307179586 * u2);
  };
  const float a = (float)aug, b = (float)(1.0 - aug);
  for (std::size_t i = from; i < rows->size(); ++i) {
    (*rows)[i] = a * (*rows)[i] + b * (float)next_normal();
  }
}

// The same gather as pack_condition_rows_, over an f32 latent that is
// ALREADY in the DiT's whitened space -- so there is no moments buffer
// to take the mean half of, and no whitening left to apply.
void
pack_latent_rows_(const float* z_data, int z, int lt, int lh, int lw,
                  int patch_h, int patch_w, std::vector<float>* rows)
{
  const std::size_t vox = (std::size_t)lt * lh * lw;
  const int gh = lh / patch_h, gw = lw / patch_w;
  const int PE = z * patch_h * patch_w;
  const std::size_t base = rows->size();
  rows->resize(base + (std::size_t)lt * gh * gw * PE);
  float* out = rows->data() + base;
  for (int t = 0; t < lt; ++t) {
    for (int cell = 0; cell < gh * gw; ++cell) {
      float* row = out + ((std::size_t)t * gh * gw + cell) * PE;
      const int by = (cell / gw) * patch_h;
      const int bx = (cell % gw) * patch_w;
      for (int c = 0; c < z; ++c) {
        for (int y = 0; y < patch_h; ++y) {
          for (int x = 0; x < patch_w; ++x) {
            const std::size_t k = (std::size_t)c * vox +
                                  ((std::size_t)t * lh + (by + y)) * lw +
                                  (bx + x);
            row[((std::size_t)c * patch_h + y) * patch_w + x] = z_data[k];
          }
        }
      }
    }
  }
}

// The MEAN half of the VAE's moments, whitened and packed into DiT rows.
//
// Two deliberate departures from the reference, both shared with the
// `fl2va` keyframe path this mirrors (stages/vae-encode-stage.cc):
// the posterior is not SAMPLED (the reference draws it under a fixed
// seed 42; an anchor whose whole job is to be exact does not want noise
// in it), and the latent is not rounded through float16 first.
void
pack_condition_rows_raw_(const std::uint16_t* mp, int z,
                         int lt, int lh, int lw, int patch_h, int patch_w,
                         const std::vector<float>& mean,
                         const std::vector<float>& std_,
                         std::vector<float>* rows)
{
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

// How many of a normalized reference's `nf` frames the video VAE
// encodes. A clip is snapped DOWN to `17n + 5` so it encodes without
// padding -- which only bites when the reference is SHORTER than the
// target, whose own count already has that form. The clamp is what
// Python's slice does for a clip shorter than one chunk. A still is its
// one frame.
int
vae_frames_(bool video, int nf, const MetalMiniMaxH3VideoVae::Config& vc)
{
  if (!video) { return nf; }
  int use = nf;
  const int fpc = vc.clip_length;
  const int lpc = vc.tokens_per_chunk();
  if (fpc > 0 && lpc > 0) {
    use = std::max(1, (nf - lpc) / fpc) * fpc + lpc;
    use = std::min(use, nf);
  }
  return use;
}

// What one reference will cost, in the unit ReferenceEncoders::progress
// counts: pixel-frames through the video VAE, padded to whole clips as
// encode_video pads them (`vae`), plus one frame of the canvas for the
// resize, the tower and a soundtrack. Planned from the same geometry the
// resize then uses, so the weights cannot drift from the work. A
// reference with no VAE work -- a soundtrack, a carried latent -- or
// whose geometry does not resolve weighs 1: it still moves the bar, and
// a geometry that fails does so in the loop, where the message belongs.
std::uint64_t
reference_work_(const MediaReference& m, const ReferencePlan& plan,
                const MetalMiniMaxH3VideoVae* video_vae, std::uint64_t* vae)
{
  *vae = 0;
  if (m.kind == MediaReference::Kind::kAudio) { return 1; }
  if (m.has_video_latent() && m.rgb.empty()) { return 1; }
  int nf = 0, th = 0, tw = 0;
  if (m.kind == MediaReference::Kind::kImage) {
    const int edge = m.short_edge >= 0 ? m.short_edge
                                       : plan.reference_image_short_edge;
    if (!resolve_reference_image_size(m.height, m.width, edge,
                                      plan.canvas_multiple,
                                      plan.reference_image_max_pixels, &th,
                                      &tw)) {
      return 1;
    }
    nf = 1;
  } else {
    const int edge =
        m.short_edge >= 0 ? m.short_edge : plan.canvas_short_edge;
    if (!video_reference_geometry(m.num_frames, m.height, m.width, m.fps,
                                  plan.target_frames, plan.canvas_multiple,
                                  edge, plan.canvas_max_pixels, &nf, &th,
                                  &tw, kFps)) {
      return 1;
    }
  }
  const std::uint64_t frame =
      std::max<std::uint64_t>(1, (std::uint64_t)th * (std::uint64_t)tw);
  if (video_vae != nullptr && !m.has_video_latent()) {
    const auto& vc = video_vae->config();
    const int use =
        vae_frames_(m.kind == MediaReference::Kind::kVideo, nf, vc);
    const int cl = std::max(1, vc.clip_length);
    const int padded = use <= 1 ? 1 : (use + cl - 1) / cl * cl;
    *vae = (std::uint64_t)padded * frame;
  }
  return frame + *vae;
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
  // An EMPTY list is Ref2VA's prompt-only form: the presentation is the
  // prompt alone and the layout packs no reference block, which is the
  // t2va sequence exactly. It is only reachable by saying so -- see the
  // stage, which refuses a request that merely has none.
  if (refs.empty()) { return true; }
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
  // The row WIDTHS belong to the models, not to the request, so they are
  // set before any reference is read. A modality with no rows still has
  // a width: a request of stills packs no audio, and its empty audio
  // beat has to say [0, 32] rather than [0, 1] for the DiT to take it.
  if (models.video_vae != nullptr) {
    r.video_row_elems = models.video_vae->config().z_channels *
                        plan.patch_h * plan.patch_w;
  }
  if (models.audio_vae != nullptr) {
    r.audio_row_elems = models.audio_vae->config().latent_channels;
  }
  // The tower results are held for the whole traversal: the
  // presentation points INTO them (one buffer per reference, indexed by
  // temporal cell), so they have to outlive the conditioner call.
  std::vector<MetalQwenVisionEncoder::Result> vision;
  std::vector<std::vector<float>> block_seconds;
  std::vector<int> vision_of;    // reference -> index into `vision`, or -1
  vision.reserve(refs.size());
  block_seconds.reserve(refs.size());

  metal_compute::MetalCompute* mc = models.mc;

  // Phase timing. `tick` restarts the clock and returns the ms since
  // the last call, so each phase is measured by bracketing it -- no
  // cost at all when `models.log` is unset.
  const bool timing = (bool)models.log;
  auto now_ = [] {
    return std::chrono::steady_clock::now();
  };
  auto mark = now_();
  auto tick = [&]() -> long long {
    const auto t = now_();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        t - mark).count();
    mark = t;
    return (long long)ms;
  };

  const double max_seconds = (double)plan.target_frames / kFps;

  // ---- progress, planned before anything runs --------------------------
  // See ReferenceEncoders::progress for the unit. The whole request is
  // weighed up front so the total never moves: a bar that rescales
  // itself mid-run reads as progress going backwards.
  const bool reporting = (bool)models.progress;
  std::vector<std::uint64_t> work(refs.size(), 1), vae_work(refs.size(), 0);
  std::uint64_t total_work = 0;
  if (reporting) {
    for (std::size_t i = 0; i < refs.size(); ++i) {
      work[i] = reference_work_(refs[i], plan, models.video_vae,
                                &vae_work[i]);
      total_work += work[i];
    }
  }
  std::uint64_t done_work = 0;      // the references already finished
  auto report = [&](std::uint64_t done, const std::string& detail) {
    if (reporting) {
      models.progress(std::min(done, total_work), total_work, detail);
    }
  };
  auto stopped = [&]() {
    return models.stopping && models.stopping();
  };
  // A STOP IS NOT A FAILURE: false, and `err` left exactly as it was.
  // An empty reason is what the caller matches on, so nothing has to
  // recognise a message.
  auto bail_stop = [&]() { return false; };

  // REACH INTO THE VIDEO VAE, which is where the time is -- 96% of a
  // reference encode, and one long clip can spend minutes in a single
  // phase. Bounding a stop by "the current reference" would leave the
  // request people actually wait through no more interruptible than
  // before. The VAE checks between tiles, each of which committed and
  // waited, so the bound is one tile.
  //
  // CLEARED ON THE WAY OUT. The VAE outlives this call -- the stage
  // holds it across requests -- so a predicate left behind would be
  // consulted by the next encode, whose RuntimeContext may be gone.
  struct VaeStopGuard {
    MetalMiniMaxH3VideoVae* vae;
    ~VaeStopGuard() { if (vae != nullptr) { vae->set_stop(nullptr); } }
  } vae_stop_guard{models.video_vae};
  if (models.video_vae != nullptr && models.stopping) {
    models.video_vae->set_stop(models.stopping);
  }

  for (std::size_t i = 0; i < refs.size(); ++i) {
    if (stopped()) { return bail_stop(); }
    const MediaReference& m = refs[i];
    const std::string where = "reference " + std::to_string(i + 1);
    const char* kind =
        m.kind == MediaReference::Kind::kImage   ? "image"
        : m.kind == MediaReference::Kind::kVideo ? "video"
                                                 : "audio";
    const std::string label = "reference " + std::to_string(i + 1) + "/" +
                              std::to_string(refs.size()) + " (" + kind +
                              ")";
    report(done_work, label);
    Reference L;
    ReferenceFit fit;
    long long ms_audio = 0, ms_norm = 0, ms_vision = 0, ms_vae = 0;
    int vision_frames = 0;
    fit.kind       = m.kind;
    fit.src_h      = m.height;
    fit.src_w      = m.width;
    fit.src_fps    = m.fps;
    fit.src_frames = m.num_frames;
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
      if (m.sample_rate > 0) {
        fit.audio_src_seconds = (double)n_samp / (double)m.sample_rate;
      }
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
      report(done_work, label + ": soundtrack");
      if (timing) { (void)tick(); }
      if (!models.audio_vae->encode(pcm.data(), 2, kept, &lat, &frames,
                                    &aerr)) {
        return fail(where + ": the audio VAE encode failed (" +
                    (aerr.empty() ? "unknown error" : aerr) + ")");
      }
      if (timing) { ms_audio = tick(); }
      const auto& ac = models.audio_vae->config();
      const int AC = ac.latent_channels;
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
      if (m.sample_rate > 0) {
        fit.audio_kept_seconds = (double)kept / (double)m.sample_rate;
      }
      L.num_audio_latents = frames;
    }

    // ---- a reference that carries its own latent ----------------------
    // Nothing to normalize, nothing for the VAE, and -- with no pixels
    // -- nothing for the VISION TOWER either, so this reference
    // contributes DiT rows but no conditioning rows. That is a real
    // difference from the pixel route and not a detail: the tower's
    // reading of the clip is part of the prompt. A caller that wants
    // both wires the pixels AS WELL, and the branch further down uses
    // the latent for the rows while the tower still reads the picture.
    const bool latent_only = m.kind != MediaReference::Kind::kAudio
                             && m.has_video_latent() && m.rgb.empty();
    if (latent_only) {
      const int lh = m.latent_height, lw = m.latent_width;
      if ((lh % plan.patch_h) != 0 || (lw % plan.patch_w) != 0) {
        return fail(where + ": a carried latent is " + std::to_string(lh) +
                    "x" + std::to_string(lw) + ", which the DiT's " +
                    std::to_string(plan.patch_h) + "x" +
                    std::to_string(plan.patch_w) + " patch does not divide");
      }
      const std::size_t need =
          (std::size_t)m.latent_z * m.latent_frames * lh * lw;
      if (m.video_latent.size() < need) {
        return fail(where + ": a carried latent is shorter than its own "
                            "shape");
      }
      const std::size_t before = r.video_rows.size();
      pack_latent_rows_(m.video_latent.data(), m.latent_z, m.latent_frames,
                        lh, lw, plan.patch_h, plan.patch_w, &r.video_rows);
      augment_condition_rows_(&r.video_rows, before,
                              plan.condition_noise_aug,
                              plan.condition_noise_seed);
      L.num_latent_frames = m.latent_frames;
      L.latent_height     = lh;
      L.latent_width      = lw;
      fit.vae_frames      = 0;
      fit.rate_frames     = m.latent_frames;
    }

    // ---- the pixels ---------------------------------------------------
    if (m.kind != MediaReference::Kind::kAudio && !latent_only) {
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
      report(done_work, label + ": resize");
      if (timing) { (void)tick(); }
      // A reference's own short edge wins over the plan's. Negative
      // means it did not state one.
      if (m.kind == MediaReference::Kind::kImage) {
        const int edge = m.short_edge >= 0 ? m.short_edge
                                           : plan.reference_image_short_edge;
        if (!normalize_image_reference(m.rgb.data(), m.height, m.width, edge,
                                       plan.canvas_multiple,
                                       plan.reference_image_max_pixels, &px,
                                       &th, &tw, &nerr)) {
          return fail(where + ": " + nerr);
        }
        nf = 1;
        fit.rate_frames = 1;
      } else {
        const int edge =
            m.short_edge >= 0 ? m.short_edge : plan.canvas_short_edge;
        if (!normalize_video_reference(m.rgb.data(), m.num_frames, m.height,
                                       m.width, m.fps, plan.target_frames,
                                       plan.canvas_multiple, edge,
                                       plan.canvas_max_pixels, &px, &nf, &th,
                                       &tw, kFps, &nerr)) {
          return fail(where + ": " + nerr);
        }
        // What the RATE resample produced, before the truncation. The
        // two losses have different remedies, so they are counted apart
        // rather than as one shortfall against the source.
        const std::vector<int> counts =
            frame_resample_counts(m.num_frames, m.fps, kFps);
        int total = 0;
        for (int c : counts) { total += c; }
        fit.rate_frames = total;
      }
      if (timing) { ms_norm = tick(); }
      fit.canvas_h    = th;
      fit.canvas_w    = tw;
      fit.used_frames = nf;

      // The CONDITIONER's read: 2 fps for a video, the whole (single)
      // frame for an image. The tower smart-resizes to a canvas of its
      // own, so the two models genuinely see different pixels -- but it
      // resizes FROM `px`, at the canvas resolved above, so that canvas
      // still sets how much the tower is given to look at.
      //
      // Which makes the canvas a CONDITIONING knob and not only a latent
      // one, and it is the half that is easy to miss: the reference rows
      // and the VAE cost are visible in the layout, while the token count
      // just quietly moves. MEASURED on the two-reference ref2va example,
      // changing only the video canvas 1344x768 -> 960x544: 3115 -> 2119
      // conditioning rows, all of it the clip's vision blocks (the still
      // is encoded at its own short edge and did not move).
      if (models.vision != nullptr) {
        report(done_work, label + ": vision tower");
        if (timing) { (void)tick(); }
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
          vision_frames = (int)idx.size();
          res = models.vision->encode_video(sampled.data(), (int)idx.size(),
                                            th, tw);
        }
        if (timing) { ms_vision = tick(); }
        if (res.embeddings.empty()) {
          return fail(where + ": the vision tower produced nothing");
        }
        vision_of.back() = (int)vision.size();
        vision.push_back(std::move(res));
        block_seconds.push_back(std::move(secs));
      }

      // A CARRIED latent needs no VAE at all -- it is already the
      // thing the VAE would have produced, without the round trip that
      // attenuates it.
      if (m.has_video_latent()) {
        const int lh = m.latent_height, lw = m.latent_width;
        if ((lh % plan.patch_h) != 0 || (lw % plan.patch_w) != 0) {
          return fail(where + ": a carried latent is " +
                      std::to_string(lh) + "x" + std::to_string(lw) +
                      ", which the DiT's " + std::to_string(plan.patch_h) +
                      "x" + std::to_string(plan.patch_w) + " patch does "
                      "not divide");
        }
        const std::size_t need = (std::size_t)m.latent_z * m.latent_frames
                                 * lh * lw;
        if (m.video_latent.size() < need) {
          return fail(where + ": a carried latent is shorter than its "
                      "own shape");
        }
        const std::size_t before2 = r.video_rows.size();
        pack_latent_rows_(m.video_latent.data(), m.latent_z,
                          m.latent_frames, lh, lw, plan.patch_h,
                          plan.patch_w, &r.video_rows);
        augment_condition_rows_(&r.video_rows, before2,
                                plan.condition_noise_aug,
                                plan.condition_noise_seed);
        L.num_latent_frames = m.latent_frames;
        L.latent_height     = lh;
        L.latent_width      = lw;
        fit.vae_frames      = 0;
      } else if (models.video_vae != nullptr) {
        if (mc == nullptr) { return fail("the video VAE has no compute"); }
        const auto& vc = models.video_vae->config();
        const int use =
            vae_frames_(m.kind == MediaReference::Kind::kVideo, nf, vc);
        fit.vae_frames = use;
        if (timing) { (void)tick(); }
        metal_compute::SharedBuffer in =
            to_vae_input_(mc, px.data(), use, th, tw);
        if (in.empty()) { return fail(where + ": VAE input alloc failed"); }
        int lf = 0;
        std::string eerr;
        // THE LONG STRETCH, so the bar moves per tile inside it: the
        // reference's VAE share, in proportion to the tiles the VAE has
        // finished. Cleared on every way out -- the VAE is the stage's,
        // and a callback left on it would point at this frame.
        struct TileProgressReset {
          MetalMiniMaxH3VideoVae* vae;
          ~TileProgressReset() { vae->set_tile_progress(nullptr); }
        } tile_reset{models.video_vae};
        if (reporting) {
          const std::uint64_t at = done_work + (work[i] - vae_work[i]);
          const std::uint64_t span = vae_work[i];
          report(at, label + ": video VAE");
          models.video_vae->set_tile_progress(
              [&report, &label, at, span](int d, int t) {
                const std::uint64_t part =
                    t > 0 ? span * (std::uint64_t)std::max(d, 0) /
                                (std::uint64_t)t
                          : 0;
                report(at + part, label + ": video VAE tile " +
                                      std::to_string(d) + "/" +
                                      std::to_string(t));
              });
        }
        metal_compute::SharedBuffer mom =
            models.video_vae->encode_video(in, use, th, tw, &lf, &eerr);
        if (mom.empty() || lf <= 0) {
          // A STOP COMES BACK THROUGH HERE, as an empty buffer with no
          // reason -- that is the VAE's contract for one, because a
          // stop is not a failure. Asking before reporting is what
          // keeps "the video VAE encode failed (unknown error)" off the
          // screen of somebody who pressed stop. MEASURED before this
          // check existed: a timer-fired stop 14 s into a 2048x2048
          // reference was served in 7.6 s and then filed as a failure.
          if (stopped()) { return bail_stop(); }
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
        const std::size_t before = r.video_rows.size();
        pack_condition_rows_(mom, vc.z_channels, lf, lh, lw, plan.patch_h,
                             plan.patch_w, vc.latents_mean, vc.latents_std,
                             &r.video_rows);
        augment_condition_rows_(&r.video_rows, before,
                                plan.condition_noise_aug,
                                plan.condition_noise_seed);
        L.num_latent_frames = lf;
        L.latent_height     = lh;
        L.latent_width      = lw;
        if (timing) { ms_vae = tick(); }
      }
    }

    if (timing) {
      models.log(
          "[h3-refenc] " + where + " (" + kind + " " +
          std::to_string(m.num_frames) + "f " + std::to_string(m.width) + "x" +
          std::to_string(m.height) + " -> " + std::to_string(fit.used_frames) +
          "f " + std::to_string(fit.canvas_w) + "x" +
          std::to_string(fit.canvas_h) + "): resize " +
          std::to_string(ms_norm) + " ms, vision tower " +
          std::to_string(vision_frames) + " frames " +
          std::to_string(ms_vision) + " ms, video VAE " +
          std::to_string(fit.vae_frames) + "f -> " +
          std::to_string(L.num_latent_frames) + " latent " +
          std::to_string(ms_vae) + " ms, audio VAE " +
          std::to_string(L.num_audio_latents) + " latents " +
          std::to_string(ms_audio) + " ms");
    }

    L.kind = m.kind == MediaReference::Kind::kImage ? Reference::Kind::kImage
             : m.kind == MediaReference::Kind::kVideo
                 ? Reference::Kind::kVideo
                 : Reference::Kind::kAudio;
    r.layout.push_back(L);
    r.fits.push_back(fit);
    done_work += work[i];
    report(done_work, label + ": done");
  }

  // ---- the presentation ------------------------------------------------
  // One conditioner call over the whole request: the labels are numbered
  // across references, so this cannot be done a reference at a time.
  //
  // THE LAST CHANCE TO STOP CHEAPLY. What follows is a single call with
  // no hook to offer, so a stop arriving inside it is served when it
  // returns; checking here is what keeps a stop from paying for a
  // conditioner run whose output is about to be thrown away.
  if (stopped()) { return bail_stop(); }
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
    // Indeterminate: see ReferenceEncoders::progress.
    if (reporting) {
      models.progress(0, 0, refs.empty()
                                ? std::string("conditioner, prompt only")
                                : "conditioner, " +
                                      std::to_string(refs.size()) +
                                      " reference(s)");
    }
    if (timing) { (void)tick(); }
    r.conditioning = models.text->encode_references(pres, prompt,
                                                    &r.token_tags,
                                                    &r.n_tokens, &cerr);
    if (r.conditioning.empty()) {
      return fail("the conditioner failed (" +
                  (cerr.empty() ? "unknown error" : cerr) + ")");
    }
    if (timing) {
      models.log("[h3-refenc] presentation: " + std::to_string(r.n_tokens) +
                 " tokens, conditioner " + std::to_string(tick()) + " ms");
    }
  }

  // The request is done -- reported here rather than inside the
  // `models.text` branch above, because reaching this line is what makes
  // it finished whether or not a conditioner was attached.
  report(total_work, "done");

  *out = std::move(r);
  return true;
}

void
pack_condition_rows_for_test(const std::uint16_t* moments, int z, int frames,
                             int h, int w, int patch_h, int patch_w,
                             const std::vector<float>& latents_mean,
                             const std::vector<float>& latents_std,
                             std::vector<float>* rows)
{
  pack_condition_rows_raw_(moments, z, frames, h, w, patch_h, patch_w,
                           latents_mean, latents_std, rows);
}

void
pack_latent_rows_for_test(const float* latent, int z, int frames, int h,
                          int w, int patch_h, int patch_w,
                          std::vector<float>* rows)
{
  pack_latent_rows_(latent, z, frames, h, w, patch_h, patch_w, rows);
}

}  // namespace minimax_h3
}  // namespace genai
}  // namespace vpipe

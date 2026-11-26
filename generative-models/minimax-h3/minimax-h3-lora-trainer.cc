#include "generative-models/minimax-h3/minimax-h3-lora-trainer.h"

#include "common/flex-data.h"
#include "generative-models/gen-input.h"
#include "generative-models/minimax-h3/minimax-h3-denoise.h"
#include "generative-models/minimax-h3/minimax-h3-image.h"
#include "generative-models/minimax-h3/p-h3-image.h"
#include "generative-models/minimax-h3/minimax-h3-reference-encoder.h"
#include "generative-models/weight-set.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

namespace vpipe {
namespace genai {

using metal_compute::SharedBuffer;
namespace h3 = minimax_h3;

namespace {

std::uint16_t
bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (std::uint16_t)(u >> 16);
}

}  // namespace

std::unique_ptr<MiniMaxH3LoraTrainer>
MiniMaxH3LoraTrainer::load(const LoadArgs& a, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return std::unique_ptr<MiniMaxH3LoraTrainer>();
  };
  if (a.ws == nullptr || a.mc == nullptr) {
    return fail("minimax-h3 trainer: no weights or metal context");
  }
  if (!a.cfg.audio && a.variant != Variant::kH3i) {
    return fail("minimax-h3 trainer: this DiT has no audio branch; only "
                "MiniMax-H3-Image's trains without one");
  }
  std::unique_ptr<MiniMaxH3LoraTrainer> t(new MiniMaxH3LoraTrainer());
  t->_mc = a.mc;
  t->_partition = a.partition;
  t->_variant = a.variant;
  for (const auto& l : a.base_loras) { t->_base_loras.push_back(l.path); }
  // The shifts the variant samples at: the release's 12 / 3 for the
  // video DiT; P-H3-Image's grid is run as written (shifts 1).
  if (a.variant == Variant::kPh3) {
    t->_vshift = 1.0;
    t->_ashift = 1.0;
    t->_ph3_sigmas = a.ph3_sigmas;
    if (t->_ph3_sigmas.size() < 2) {
      return fail("minimax-h3 trainer: P-H3-Image's grid is missing");
    }
  }
  if (a.variant == Variant::kH3i) {
    t->_h3i_run = h3i::grid_as_run(a.h3i_sigmas);
    if (t->_h3i_run.size() < 2 || a.heads_ws == nullptr) {
      return fail("minimax-h3 trainer: MiniMax-H3-Image's grid or per-step "
                  "heads are missing");
    }
  }
  t->_m = MetalMiniMaxH3Transformer::load(a.ws, a.mc, a.cfg,
                                          a.stream_blocks, a.base_loras);
  if (t->_m == nullptr) {
    return fail("minimax-h3 trainer: the DiT did not load");
  }
  if (!a.base_loras.empty() &&
      t->_m->lora_modules() == 0) {
    return fail("minimax-h3 trainer: no base adapter bound on this DiT ('" +
                a.base_loras.front().path + "')");
  }
  // The training blocks read the DiT's own weights; an adapter that
  // overwrites some whole (pi-H3) would train against the wrong model.
  if (t->_m->lora_overwrites()) {
    return fail("minimax-h3 trainer: a base adapter overwrites weights whole "
                "(pi-H3's kind), which the training blocks do not apply");
  }
  if (a.variant == Variant::kH3i) {
    std::string herr;
    if (!t->_m->set_step_heads(a.heads_ws, &herr)) {
      return fail("minimax-h3 trainer: MiniMax-H3-Image's per-step heads: " +
                  herr);
    }
  }
  if (a.floor_seq > 0) {
    t->_m->set_scratch_floor(a.floor_seq, a.floor_text);
    // A rank's exchange buffers at the widest sample from the first: the
    // samples' lengths vary, and each regrowth is a new registration.
    // Its eight band buffers at the WIDEST BAND ANY SAMPLE CUTS, which is
    // not the widest sample's: the bands are balanced, so a shorter
    // sequence can cut wider ones (a 2,561-row sample 1,408-row bands
    // where a 4,388-row one cuts 1,152). The most over every length up
    // to the widest -- band_rows without the prompt's rule, which only
    // ever runs a block whole.
    if (a.cfg.tp.active()) {
      const std::size_t row = (std::size_t)a.cfg.hidden * 2;
      int band = 0;
      for (int s = 1; s <= a.floor_seq; ++s) {
        band = std::max(band, MetalMiniMaxH3Transformer::band_rows(a.cfg, s));
      }
      const std::size_t whole = (std::size_t)a.floor_seq * row;
      const std::size_t banded = (std::size_t)band * row;
      t->_m->set_tp_exchange_floor(whole, banded);
      // The bands' eight and the bulk's two chunks (TpBulk's default).
      // Send and others are the plan's already: plan_arena_bytes counts
      // scratch_bytes' rank term, 2 x [seq, hidden] at the widest sample.
      t->_split_bytes = 8 * banded + 2 * ((std::size_t)32 << 20);
    }
  }
  std::string e;
  t->_t = MiniMaxH3Trainer::create(t->_m.get(), &e);
  if (t->_t == nullptr) { return fail(e); }
  return t;
}

double
MiniMaxH3LoraTrainer::audio_sigma(double sv, double video_shift,
                                  double audio_shift)
{
  // s = k u / (1 + (k - 1) u), inverted at the video's shift, then out
  // through the audio's.
  sv = std::clamp(sv, 0.0, 1.0);
  const double u = sv / (video_shift - (video_shift - 1.0) * sv);
  return audio_shift * u / (1.0 + (audio_shift - 1.0) * u);
}

int
MiniMaxH3LoraTrainer::h3i_step_(std::uint64_t seed) const
{
  const int n = (int)_h3i_run.size() - 1;
  std::mt19937_64 rng(seed ^ 0x6833692d73746570ull);
  return n > 0 ? (int)(rng() % (std::uint64_t)n) : 0;
}

int
MiniMaxH3LoraTrainer::pixel_frames(int latent_frames)
{
  if (latent_frames == 1) { return 1; }
  if (latent_frames < 2 || (latent_frames - 2) % 5 != 0) { return 0; }
  return 17 * ((latent_frames - 2) / 5) + 5;
}

int
MiniMaxH3LoraTrainer::sequence_rows(
    const MetalMiniMaxH3Transformer::Config& c, int frames, int gh, int gw,
    int text_rows, int anchors, int cond_rows)
{
  const int pf = pixel_frames(frames);
  const int vrows = (frames + anchors) * (gh / c.patch_h) * (gw / c.patch_w);
  const int arows =
      c.audio ? h3::kAudioChannels * h3::audio_latent_num_frames(pf) : 0;
  return text_rows + vrows + arows + cond_rows;
}

std::size_t
MiniMaxH3LoraTrainer::plan_arena_bytes(
    const MetalMiniMaxH3Transformer::Config& c, int frames, int gh, int gw,
    int text_rows, int rank, std::size_t widen, int anchors, int cond_rows,
    bool keep_all)
{
  const int seq =
      sequence_rows(c, frames, gh, gw, text_rows, anchors, cond_rows);
  // A T2VA sample's rows sit at two timesteps (video, audio); three
  // bounds a keyframe's or a reference's level besides.
  constexpr int kNt = 3;
  // The family's own sample buffers: a noised copy of the conditioning
  // rows (MiniMax-H3-Image's edit pictures, an augmented keyframe) and a
  // missing soundtrack's zeros and noise.
  const std::size_t own =
      (std::size_t)seq * ((std::size_t)c.video_patch_elems() +
                          2 * (std::size_t)c.audio_channels) * 2;
  return own +
         MiniMaxH3Trainer::arena_bytes(c, seq, text_rows, kNt, rank,
                                       keep_all) +
         MetalMiniMaxH3Transformer::scratch_bytes(c, seq, text_rows, kNt,
                                                  true, false, 0, 0, widen);
}

std::vector<train::ModuleShape>
MiniMaxH3LoraTrainer::modules(const std::string& targets) const
{
  return _t->modules(targets, true);
}

int
MiniMaxH3LoraTrainer::bind(train::LoraParams* params, std::string* err)
{
  return _t->bind(params, err);
}

int
MiniMaxH3LoraTrainer::latent_channels() const noexcept
{
  return _m->config().video_channels;
}

int
MiniMaxH3LoraTrainer::audio_latents(int frames) const
{
  const int pf = pixel_frames(frames);
  return _m->config().audio && pf > 0 ? h3::audio_latent_num_frames(pf) : 0;
}

std::vector<std::int64_t>
MiniMaxH3LoraTrainer::latent_shape(int t, int h, int w) const
{
  return {_m->config().video_channels, t, h, w};
}

SharedBuffer
MiniMaxH3LoraTrainer::pack_latent(const float* cthw, int c, int t, int h,
                                  int w, std::string* err)
{
  const MetalMiniMaxH3Transformer::Config& cfg = _m->config();
  const int Z = cfg.video_channels;
  if (c != Z || t < 1 || pixel_frames(t) == 0 || (h % cfg.patch_h) != 0 ||
      (w % cfg.patch_w) != 0) {
    if (err != nullptr) {
      *err = "minimax-h3 trainer: a latent must be [" + std::to_string(Z) +
             ", t, h, w] with 1 or 5n + 2 frames and an EVEN grid (pictures "
             "in multiples of 32); got [" + std::to_string(c) + ", " +
             std::to_string(t) + ", " + std::to_string(h) + ", " +
             std::to_string(w) + "]";
    }
    return {};
  }
  const int rows = t * (h / cfg.patch_h) * (w / cfg.patch_w);
  const int PE = cfg.video_patch_elems();
  std::vector<float> f((std::size_t)rows * PE);
  // The packing generate-video reads its rows back through: frame-major,
  // a frame's cells row-major, each cell [z, patch_h, patch_w].
  h3::patchify_latent_frames(cthw, Z, t, 0, t, h, w, cfg.patch_h,
                             cfg.patch_w, f.data());
  SharedBuffer b = _mc->make_shared_buffer(f.size() * 2);
  if (b.empty()) { return b; }
  auto* d = static_cast<std::uint16_t*>(b.contents());
  for (std::size_t i = 0; i < f.size(); ++i) { d[i] = bf16_(f[i]); }
  return b;
}

SharedBuffer
MiniMaxH3LoraTrainer::pack_audio(const float* audio, int stereo,
                                 int channels, int stride, int latents,
                                 std::string* err)
{
  const int AC = _m->config().audio_channels;
  if (stereo != h3::kAudioChannels || channels != AC || latents < 1 ||
      stride < latents) {
    if (err != nullptr) {
      *err = "minimax-h3 trainer: a soundtrack latent must be [" +
             std::to_string(h3::kAudioChannels) + ", " + std::to_string(AC) +
             ", n]; got [" + std::to_string(stereo) + ", " +
             std::to_string(channels) + ", " + std::to_string(stride) + "]";
    }
    return {};
  }
  std::vector<float> f((std::size_t)h3::kAudioChannels * latents * AC);
  h3::pack_audio_latent_rows(audio, AC, stride, latents, f.data());
  SharedBuffer b = _mc->make_shared_buffer(f.size() * 2);
  if (b.empty()) { return b; }
  auto* d = static_cast<std::uint16_t*>(b.contents());
  for (std::size_t i = 0; i < f.size(); ++i) { d[i] = bf16_(f[i]); }
  return b;
}

bool
MiniMaxH3LoraTrainer::layout_(const FlexData* sideband, int text_rows,
                              int frames, int gh, int gw, int audio_latents,
                              int anchors, h3::PackedLayout* L,
                              std::string* err, bool* has_refs) const
{
  if (has_refs != nullptr) { *has_refs = false; }
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = "minimax-h3 trainer: " + m; }
    return false;
  };
  const MetalMiniMaxH3Transformer::Config& c = _m->config();
  // The rows' modality tags: text, except a presentation's vision block
  // (FL2VA's keyframes shown to Qwen3-VL), which the conditioner tags.
  std::vector<int> tags((std::size_t)text_rows, h3::kTextTag);
  std::vector<h3::Reference> refs;
  if (sideband != nullptr && sideband->is_object()) {
    FlexData sb = *sideband;
    auto o = sb.as_object();
    // The request's references, as video-ref-encoder planned them (the
    // same reading as generate-video's): an empty list is a prompt-only
    // request, the T2VA layout under the reference builder.
    if (o.contains(cond_sideband::kReferences)) {
      const FlexData rl = o.at(cond_sideband::kReferences);
      if (rl.is_array()) {
        auto ra = rl.as_array();
        for (std::size_t i = 0; i < ra.size(); ++i) {
          const FlexData e = ra.at(i);
          if (!e.is_object()) { return fail("a reference is not an object"); }
          auto eo = e.as_object();
          auto num = [&](std::string_view k) -> int {
            return eo.contains(k) ? (int)eo.at(k).as_int(0) : 0;
          };
          const FlexData kd = eo.contains(cond_sideband::kKind)
                                  ? eo.at(cond_sideband::kKind)
                                  : FlexData{};
          const std::string kind =
              kd.is_string() ? std::string(kd.as_string("")) : "";
          h3::Reference r;
          using RK = h3::Reference::Kind;
          r.kind = kind == cond_sideband::kVideo   ? RK::kVideo
                   : kind == cond_sideband::kAudio ? RK::kAudio
                                                   : RK::kImage;
          r.num_latent_frames = num(cond_sideband::kLatentFrames);
          r.latent_height = num(cond_sideband::kLatentHeight);
          r.latent_width = num(cond_sideband::kLatentWidth);
          r.num_audio_latents = num(cond_sideband::kAudioLatents);
          refs.push_back(r);
        }
      }
      // References are Ref2VA's; on the two image models they are an
      // EDIT's pictures (the first the one edited).
      const bool image_model = _variant != Variant::kVideo;
      if (!refs.empty() && !image_model && _partition != "ref2va") {
        return fail("a caption encoded with references trains MiniMax-H3 "
                    "Ref2VA; this is the " +
                    (_partition.empty() ? std::string("t2va") : _partition) +
                    " partition");
      }
      if (image_model) {
        for (const h3::Reference& r : refs) {
          if (r.kind != h3::Reference::Kind::kImage) {
            return fail("an edit's references are pictures; this caption "
                        "carries a clip or a soundtrack");
          }
        }
      }
    }
    if (o.contains(cond_sideband::kTokenTags)) {
      const FlexData tf = o.at(cond_sideband::kTokenTags);
      std::vector<int> t;
      bool vision = false;
      if (tf.is_array()) {
        auto ta = tf.as_array();
        for (std::size_t i = 0; i < ta.size(); ++i) {
          t.push_back((int)ta.at(i).as_int(h3::kTextTag));
          vision = vision || t.back() != h3::kTextTag;
        }
      }
      // A vision block is a keyframe's -- or a reference's, presented.
      if (vision && anchors == 0 && refs.empty()) {
        return fail("a caption presented with keyframes trains a keyframe "
                    "task; this sample pins none");
      }
      if ((int)t.size() == text_rows) { tags = std::move(t); }
    }
  }
  if (anchors != 0 && _partition == "ref2va") {
    return fail("keyframe anchors are FL2VA's tasks; this is the Ref2VA "
                "partition");
  }
  std::vector<h3::Anchor> ends;
  if (anchors & 1) { ends.push_back(h3::Anchor::kFirst); }
  if (anchors & 2) { ends.push_back(h3::Anchor::kLast); }
  const int pf = pixel_frames(frames);
  if (pf == 0) {
    return fail("a clip of " + std::to_string(frames) + " latent frames is "
                "not one the video VAE makes (1, or 5n + 2)");
  }
  const int alat = audio_latents >= 0 ? audio_latents
                   : c.audio          ? h3::audio_latent_num_frames(pf)
                                      : 0;
  // generate-video's builder for the partition: Ref2VA's prompt-only form
  // packs the T2VA layout row for row, marked as the reference builder's;
  // FL2VA's keyframes are anchors of the keyframe layout.
  if (_variant != Variant::kVideo && (frames != 1 || anchors != 0)) {
    return fail(std::string(_variant == Variant::kPh3 ? "P-H3-Image"
                                                      : "MiniMax-H3-Image") +
                " trains on pictures: one latent frame, no keyframes");
  }
  // MiniMax-H3-Image packs [text | pictures | the picture] whether or not
  // it edits -- the reference builder's layout, no soundtrack; P-H3-Image
  // packs an edit's pictures the same way (its text-to-image is T2VA's).
  const bool ok =
      _partition == "ref2va" || _variant == Variant::kH3i ||
              (_variant == Variant::kPh3 && !refs.empty())
          ? h3::build_ref2va_packed_sequence(tags, refs, frames, gh, gw, alat,
                                             c.patch_h, c.patch_w,
                                             h3::kAudioChannels, L)
          : h3::build_packed_sequence(tags, frames, gh, gw, alat, c.patch_h,
                                      c.patch_w, h3::kAudioChannels, ends,
                                      L);
  if (!ok) {
    return fail("could not pack a " + std::to_string(frames) + "x" +
                std::to_string(gh) + "x" + std::to_string(gw) +
                " latent with " + std::to_string(alat) + " audio latents" +
                (refs.empty() ? std::string()
                              : " and " + std::to_string(refs.size()) +
                                    " reference(s)"));
  }
  if (has_refs != nullptr) { *has_refs = !refs.empty(); }
  if (!refs.empty() && anchors != 0) {
    return fail("a sample cannot pin keyframes and carry references");
  }
  return true;
}

bool
MiniMaxH3LoraTrainer::sample_(const train::TrainSample& s,
                              h3::PackedLayout* L,
                              MiniMaxH3Trainer::Sample* q, std::string* err)
{
  const MetalMiniMaxH3Transformer::Config& c = _m->config();
  const int AC = c.audio_channels;
  int alat = -1;
  if (s.audio_x0 != nullptr) {
    alat = (int)(s.audio_x0->byte_size() /
                 ((std::size_t)h3::kAudioChannels * AC * 2));
  }
  bool refs = false;
  if (!layout_(s.sideband, s.text_rows, s.frames, s.grid_h, s.grid_w, alat,
               s.anchors, L, err, &refs)) {
    return false;
  }
  // References: their rows as video-ref-encoder emitted them, already at
  // their level (its condition_noise_aug), the request's conditioning.
  if (refs) {
    const std::size_t vb =
        (std::size_t)L->num_condition_video_rows * c.video_patch_elems() * 2;
    const std::size_t ab = (std::size_t)L->num_condition_audio_rows * AC * 2;
    if ((vb > 0 && (s.cond_x0 == nullptr || s.cond_x0->byte_size() < vb)) ||
        (ab > 0 &&
         (s.cond_audio == nullptr || s.cond_audio->byte_size() < ab))) {
      if (err != nullptr) {
        *err = "minimax-h3 trainer: the caption's references pack " +
               std::to_string(L->num_condition_video_rows) + " video and " +
               std::to_string(L->num_condition_audio_rows) + " audio rows "
               "and fewer arrived (are video-ref-encoder's ref_video / "
               "ref_audio wired to train-lora?)";
      }
      return false;
    }
    const MetalMiniMaxH3Transformer::GenerationParams gp;
    q->video_cond = vb > 0 ? s.cond_x0 : nullptr;
    q->audio_cond = ab > 0 ? s.cond_audio : nullptr;
    q->cond_video_t = (float)gp.condition_timestep;
    q->cond_audio_t = (float)gp.condition_audio_timestep;
    if (_variant == Variant::kPh3) {
      // P-H3-Image's pictures come noised by the encoder, to the level
      // generate-video holds them at.
      q->cond_video_t = (float)ph3::kConditionNoiseAug;
    }
    if (_variant == Variant::kH3i && vb > 0) {
      // MiniMax-H3-Image's come CLEAN (their posteriors sampled) and are
      // noised here, as generate-video does: t z + (1 - t) n at t 0.999,
      // the noise the sample's own.
      const float t = h3i::kReferenceNoiseAug;
      const std::size_t n = vb / 2;
      if (_cond.byte_size() < vb) { _cond = _mc->make_shared_buffer(vb); }
      if (_cond.empty()) {
        if (err != nullptr) { *err = "minimax-h3 trainer: allocation"; }
        return false;
      }
      const auto* src =
          static_cast<const std::uint16_t*>(s.cond_x0->contents());
      auto* d = static_cast<std::uint16_t*>(_cond.contents());
      std::mt19937_64 rng(s.seed ^ 0x6833692d70696373ull);
      std::normal_distribution<float> nd(0.0f, 1.0f);
      for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t u = (std::uint32_t)src[i] << 16;
        float z;
        std::memcpy(&z, &u, 4);
        d[i] = bf16_(t * z + (1.0f - t) * nd(rng));
      }
      q->video_cond = &_cond;
      q->cond_video_t = t;
    }
  }
  // The keyframes: their stills' rows, at the keyframe level (generate-
  // video's condition_timestep: 1, clean, on the release).
  if (!refs && L->num_condition_video_rows > 0) {
    const std::size_t cb =
        (std::size_t)L->num_condition_video_rows * c.video_patch_elems() * 2;
    if (s.cond_x0 == nullptr || s.cond_x0->byte_size() < cb) {
      if (err != nullptr) {
        *err = "minimax-h3 trainer: the sample pins keyframes and carries "
               "fewer keyframe rows than its layout packs";
      }
      return false;
    }
    const double ct =
        MetalMiniMaxH3Transformer::GenerationParams{}.condition_timestep;
    q->video_cond = s.cond_x0;
    q->cond_video_t = (float)ct;
    if (ct < 1.0) {
      // The augmentation generate-video applies to its anchors.
      std::vector<float> v(cb / 2);
      const auto* src =
          static_cast<const std::uint16_t*>(s.cond_x0->contents());
      for (std::size_t i = 0; i < v.size(); ++i) {
        const std::uint32_t u = (std::uint32_t)src[i] << 16;
        std::memcpy(&v[i], &u, 4);
      }
      h3::augment_condition_values(v.data(), v.size(), ct, s.seed);
      if (_cond.byte_size() < cb) { _cond = _mc->make_shared_buffer(cb); }
      auto* d = static_cast<std::uint16_t*>(_cond.contents());
      for (std::size_t i = 0; i < v.size(); ++i) { d[i] = bf16_(v[i]); }
      q->video_cond = &_cond;
    }
  }
  const std::size_t vbytes =
      (std::size_t)L->num_video_rows * c.video_patch_elems() * 2;
  if (s.x0 == nullptr || s.noise == nullptr || s.x0->byte_size() < vbytes ||
      s.noise->byte_size() < vbytes) {
    if (err != nullptr) {
      *err = "minimax-h3 trainer: the latent holds fewer rows than its "
             "geometry packs";
    }
    return false;
  }
  q->layout = L;
  q->text = s.text;
  q->video_x0 = s.x0;
  q->video_noise = s.noise;
  q->sigma_video = s.sigma;
  q->sigma_audio = (float)audio_sigma(s.sigma, _vshift, _ashift);
  q->weight = s.weight;
  if (_variant == Variant::kH3i) {
    // A release step, by the sample's own seed: its sigma as the release
    // steps it, through its own output head (the heads were each
    // distilled for one step of that grid).
    const int i = h3i_step_(s.seed);
    q->sigma_video = _h3i_run[(std::size_t)i];
    q->out_head = i;
  }
  const std::size_t an = (std::size_t)L->num_audio_rows * AC;
  // The soundtrack's noise, or the sample's own: a stream of its own seed.
  auto draw = [&](SharedBuffer& dst) {
    if (dst.byte_size() < an * 2) {
      dst = _mc->make_shared_buffer(std::max<std::size_t>(an, 1) * 2);
    }
    if (dst.empty()) { return false; }
    std::mt19937_64 rng(s.seed ^ 0x61756469306e7365ull);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto* d = static_cast<std::uint16_t*>(dst.contents());
    for (std::size_t i = 0; i < an; ++i) { d[i] = bf16_(nd(rng)); }
    return true;
  };
  if (an == 0) {
    q->audio_weight = 0.0f;
    return true;
  }
  if (s.audio_x0 != nullptr) {
    q->audio_x0 = s.audio_x0;
    if (s.audio_noise != nullptr && s.audio_noise->byte_size() >= an * 2) {
      q->audio_noise = s.audio_noise;
    } else {
      if (!draw(_anoise)) { return false; }
      q->audio_noise = &_anoise;
    }
    q->audio_weight = s.audio_weight;
    return true;
  }
  // NO SOUNDTRACK: the rows inference would carry, at the audio's level
  // over a zero latent (the whitened mean), and no loss on them.
  if (_azero.byte_size() < an * 2) {
    _azero = _mc->make_shared_buffer(an * 2);
    if (_azero.empty()) { return false; }
    std::memset(_azero.contents(), 0, _azero.byte_size());
  }
  if (!draw(_anoise)) { return false; }
  q->audio_x0 = &_azero;
  q->audio_noise = &_anoise;
  q->audio_weight = 0.0f;
  return true;
}

bool
MiniMaxH3LoraTrainer::step(const train::TrainSample& s,
                           train::StepStats* out, std::string* err)
{
  h3::PackedLayout L;
  MiniMaxH3Trainer::Sample q;
  if (!sample_(s, &L, &q, err)) { return false; }
  MiniMaxH3Trainer::Result r;
  if (!_t->step(q, &r, err)) { return false; }
  _lv = r.loss_video;
  _la = r.loss_audio;
  if (out != nullptr) {
    out->loss = r.loss;
    out->forward_ms = r.forward_ms;
    out->backward_ms = r.backward_ms;
    out->loss_video = r.loss_video;
    // No soundtrack, no audio loss: absent rather than a perfect 0.
    out->loss_audio = q.audio_weight > 0.0f ? r.loss_audio : -1.0;
  }
  return true;
}

bool
MiniMaxH3LoraTrainer::eval_loss(const train::TrainSample& s, double* loss,
                                std::string* err)
{
  h3::PackedLayout L;
  MiniMaxH3Trainer::Sample q;
  if (!sample_(s, &L, &q, err)) { return false; }
  MiniMaxH3Trainer::Result r;
  if (!_t->loss(q, &r, err)) { return false; }
  *loss = r.loss;
  return true;
}

double
MiniMaxH3LoraTrainer::schedule_mu(int, int) const noexcept
{
  // H3's linear shift k is the exponential one at mu = ln k: 12 for the
  // video DiT, none for P-H3-Image's grid (MiniMax-H3-Image draws its
  // release steps itself).
  return std::log(_vshift);
}

bool
MiniMaxH3LoraTrainer::preview(const train::PreviewRequest& r,
                              std::vector<float>* chw, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (r.text == nullptr) { return fail("preview: no conditioning"); }
  const MetalMiniMaxH3Transformer::Config& c = _m->config();
  h3::PackedLayout L;
  bool refs = false;
  if (!layout_(r.sideband, r.text_rows, r.frames, r.grid_h, r.grid_w, -1, 0,
               &L, err, &refs)) {
    return false;
  }
  if (refs) {
    return fail("preview: a preview prompt is prompt-only; this one carries "
                "references");
  }
  const int PE = c.video_patch_elems(), AC = c.audio_channels;
  const int nvc = L.num_condition_video_rows, nvg = L.num_video_rows;
  std::vector<float> vid((std::size_t)(nvc + nvg) * PE, 0.0f);
  std::vector<float> aud((std::size_t)L.num_audio_rows * AC, 0.0f);
  {
    std::mt19937_64 rng(r.seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (std::size_t k = (std::size_t)nvc * PE; k < vid.size(); ++k) {
      vid[k] = nd(rng);
    }
    for (float& v : aud) { v = nd(rng); }
  }
  DenoiseRequest req;
  req.dit = _m.get();
  req.layout = &L;
  req.text = r.text;
  req.video = vid.data();
  req.audio = aud.empty() ? nullptr : aud.data();
  req.num_steps = std::max(r.steps, 1);
  req.video_shift = _vshift;
  req.audio_shift = _ashift;
  req.seed = r.seed;
  if (_variant == Variant::kPh3) {
    // P-H3-Image's grid as run, whatever `steps` says.
    req.sigma_grid = _ph3_sigmas;
    req.num_steps = (int)_ph3_sigmas.size() - 1;
  }
  if (_variant == Variant::kH3i) {
    // The release's grid, a head a step, the velocity form.
    req.video_sigmas.assign(_h3i_run.begin(), _h3i_run.end());
    req.num_steps = (int)_h3i_run.size() - 1;
    req.step_heads = true;
    req.update = StepUpdate::kVelocity;
  }
  req.progress = [this](int, int) { return !(_stop && _stop()); };
  std::string derr;
  if (!denoise(req, &derr)) {
    if (_stop && _stop()) {
      if (err != nullptr) { err->clear(); }
      return false;
    }
    return fail("preview: " + derr);
  }
  h3::unpatchify_latent_rows(vid.data() + (std::size_t)nvc * PE, nvg,
                             c.video_channels, r.frames, r.grid_h, r.grid_w,
                             c.patch_h, c.patch_w, chw);
  // The soundtrack the same denoise made, [2, C, n] as generate-video's
  // audio latent oport has it.
  if (r.audio != nullptr && L.num_audio_rows > 0) {
    const int alat = L.num_audio_rows / h3::kAudioChannels;
    h3::unpack_audio_latent_rows(aud.data(), AC, alat, &r.audio->latent);
    r.audio->shape = {h3::kAudioChannels, AC, alat};
    r.audio->latents_per_second = (double)h3::kAudioLatentsPerSecond;
  }
  return true;
}

void
MiniMaxH3LoraTrainer::set_stop(std::function<bool()> stop)
{
  _stop = stop;
  _t->set_stop(stop);
  _m->set_stream_stop(std::move(stop));
}

void
MiniMaxH3LoraTrainer::set_options(const FlexData& o)
{
  if (!o.is_object()) { return; }
  FlexData c = o;
  auto ob = c.as_object();
  // recompute: never (keep_all) keeps every main block's activations;
  // the ANE backward is not this family's yet.
  if (ob.contains(train::opt::kKeepAll)) {
    _t->set_keep_all(ob.at(train::opt::kKeepAll).as_bool(false));
  }
  if (_m->streaming() && ob.contains(train::opt::kResidencyReserve)) {
    _m->set_residency_reserve(
        (std::size_t)ob.at(train::opt::kResidencyReserve).as_int(0));
  }
}

std::size_t
MiniMaxH3LoraTrainer::arena_bytes(int frames, int grid_h, int grid_w,
                                  int text_rows, int rank,
                                  bool keep_all) const
{
  return plan_arena_bytes(_m->config(), frames, grid_h, grid_w, text_rows,
                          rank, _m->fp8_widen_bytes(), 0, 0, keep_all);
}

std::uint64_t
MiniMaxH3LoraTrainer::resident_bytes() const
{
  return _m->held_bytes();
}

std::map<std::string, std::string>
MiniMaxH3LoraTrainer::metadata() const
{
  std::map<std::string, std::string> m;
  if (!_partition.empty()) { m["vpipe.partition"] = _partition; }
  m["vpipe.adaln"] = _m->adaln_pruned_at_load() ? "pruned at load"
                     : _m->config().pruned()   ? "pruned checkpoint"
                                               : "released";
  m["vpipe.variant"] = _variant == Variant::kPh3   ? "p-h3-image"
                       : _variant == Variant::kH3i ? "minimax-h3-image"
                                                   : "video";
  // What inference must stack beneath it, in its order.
  if (!_base_loras.empty()) {
    std::string b;
    for (const std::string& p : _base_loras) {
      b += (b.empty() ? "" : ";") + p;
    }
    m["vpipe.base_adapter"] = b;
  }
  return m;
}

std::string
MiniMaxH3LoraTrainer::parent_model_type() const
{
  if (_variant == Variant::kPh3) { return "p-h3-image"; }
  if (_variant == Variant::kH3i) { return "minimax-h3-image"; }
  return _partition.empty() ? std::string("minimax-h3")
                            : "minimax-h3-" + _partition;
}

std::string
MiniMaxH3LoraTrainer::layout_type_of(Variant v, const std::string& partition)
{
  if (v == Variant::kPh3) { return "p-h3-image"; }
  if (v == Variant::kH3i) { return "minimax-h3-image"; }
  return partition == "ref2va" ? "minimax-h3-ref2va" : "minimax-h3";
}

}  // namespace genai
}  // namespace vpipe

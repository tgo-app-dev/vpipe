#include "generative-models/magi2/magi2-family.h"

#include "common/beat-keys.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "common/flex-bag.h"
#include "generative-models/magi2/magi2-accel.h"
#include "generative-models/magi2/magi2-audio.h"
#include "generative-models/magi2/magi2-layout.h"
#include "generative-models/magi2/magi2-preview.h"
#include "generative-models/magi2/magi2-refine.h"
#include "generative-models/magi2/magi2-schedule.h"
#include "generative-models/magi2/magi2-text.h"
#include "generative-models/magi2/metal-magi2-turbo.h"
#include "generative-models/shared/accel-settings.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/video-host-ext.h"
#include "generative-models/video-model-registry.h"
#include "generative-models/weight-set.h"
#include "generative-models/yue2/metal-oobleck-decoder.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "stages/model-memory-host.h"
#include "stages/model-memory.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {
namespace magi2 {

Accel
accel_of(const FlexData* bag, bool matrix_cores)
{
  Accel a;
  a.i8_tier = effective_i8_tier(asked_i8_tier(bag), matrix_cores);
  a.autotune = accel::flag(bag, accel::kAutotune, true);
  a.sage = sage::config_from_flex(bag);
  a.sage.enabled = sage_runs(a.sage, matrix_cores);
  return a;
}

int
align_frames(int frames)
{
  if (frames <= 1) { return 1; }
  return 4 * ((frames - 1 + 3) / 4) + 1;
}

int
latent_frames(int frames)
{
  return frames <= 0 ? 0 : (align_frames(frames) - 1) / 4 + 1;
}

int
audio_latents_of(int frames)
{
  // 25 audio latents a second at 12.5 frames: round(frames / 12.5 * 25).
  return frames <= 0 ? 0 : 2 * align_frames(frames);
}

bool
dit_plan(const std::string& preview_dir, DitPlan* out)
{
  static std::mutex mu;
  static std::map<std::string, DitPlan> memo;
  {
    std::lock_guard<std::mutex> lk(mu);
    auto it = memo.find(preview_dir);
    if (it != memo.end()) {
      if (out != nullptr) { *out = it->second; }
      return true;
    }
  }
  // The headers alone: a private set, dropped once read.
  auto ws = WeightSet::open(preview_dir, nullptr);
  if (ws == nullptr) { return false; }
  DitPlan p;
  if (!MetalMagi2Dit::config_from_weights(*ws, &p.cfg, nullptr)) {
    return false;
  }
  for (int t = -1; t <= 3; ++t) {
    MetalMagi2Dit::Config c = p.cfg;
    c.i8_tier = t;
    p.trunk_at[t + 1] = MetalMagi2Dit::trunk_bytes(*ws, c);
  }
  for (std::size_t i = 0; i < p.cfg.layers.size(); ++i) {
    if (!p.cfg.moe[i]) { continue; }
    ++p.moe_layers;
    if (p.expert_layer != 0) { continue; }
    for (const char* w : {"W_gate", "W_up", "W_down"}) {
      const auto* ti = ws->src().info(fmt("block.layers.{}.mlp.moe_mlp.{}",
                                          p.cfg.layers[i], w)());
      if (ti == nullptr) { return false; }
      p.expert_layer += ti->nbytes;
    }
  }
  std::lock_guard<std::mutex> lk(mu);
  memo[preview_dir] = p;
  if (out != nullptr) { *out = p; }
  return true;
}

bool
dit_plan(const std::string& preview_dir, const TpShare& share,
         DitPlan* out, std::string* err)
{
  DitPlan whole;
  if (!dit_plan(preview_dir, &whole)) {
    if (err != nullptr) {
      *err = "cannot read the preview DiT's shapes under " + preview_dir;
    }
    return false;
  }
  if (!share.active()) {
    if (out != nullptr) { *out = whole; }
    return true;
  }
  MetalMagi2Dit::Config c = whole.cfg;
  c.tp = share;
  MetalMagi2Dit::Share sh;
  if (!MetalMagi2Dit::share_of(c, &sh, err)) { return false; }
  static std::mutex mu;
  static std::map<std::string, DitPlan> memo;
  const std::string key =
      fmt("{}|{}/{}|{}-{}|{}-{}", preview_dir, share.rank, share.ranks,
          share.head0, share.head1, share.ffn0, share.ffn1)();
  {
    std::lock_guard<std::mutex> lk(mu);
    auto it = memo.find(key);
    if (it != memo.end()) {
      if (out != nullptr) { *out = it->second; }
      return true;
    }
  }
  auto ws = WeightSet::open(preview_dir, nullptr);
  if (ws == nullptr) { return false; }
  DitPlan p = whole;
  p.cfg = c;
  for (int t = -1; t <= 3; ++t) {
    MetalMagi2Dit::Config ct = c;
    ct.i8_tier = t;
    p.trunk_at[t + 1] = MetalMagi2Dit::trunk_bytes(*ws, ct);
  }
  // A layer's three expert tensors at the share's run of every head's.
  const std::size_t n = (std::size_t)c.moe_heads * c.moe_experts;
  p.expert_layer = n == 0 ? 0
                          : whole.expert_layer / n *
                                (std::size_t)c.moe_heads * sh.experts();
  std::lock_guard<std::mutex> lk(mu);
  memo[key] = p;
  if (out != nullptr) { *out = p; }
  return true;
}

std::size_t
dit_floor_bytes(const DitPlan& p, int tier)
{
  return p.trunk(tier) + (p.moe_layers > 0 ? 2 * p.expert_layer : 0);
}

std::size_t
dit_preload_bytes(const DitPlan& p, int tier)
{
  return p.trunk(tier) + (std::size_t)p.moe_layers * p.expert_layer;
}

std::size_t
denoise_scratch_bytes(const DitPlan& p, int width, int height, int frames,
                      const Accel& a)
{
  const int h = height / kGrid, w = width / kGrid;
  const int F = align_frames(frames);
  const int T = latent_frames(F), L = audio_latents_of(F);
  if (h <= 0 || w <= 0 || T <= 0) { return 0; }
  MetalMagi2Dit::Config c = p.cfg;
  c.i8_tier = a.i8_tier;
  c.autotune = a.autotune;
  c.sage = a.sage;
  const int text = kMaxTextTokens + 1;
  // Image-to-video's picture: at most the clip's long edge on either side
  // (generate() refuses a larger one), so its tokens at that square.
  const int side = std::max(h, w);
  const int img = side * side;
  const int n = T * h * w + img + L + text;
  const int inputs = c.video_c * (T * h * w + img) + c.audio_c * L +
                     c.text_c * text;
  // The loop's host state beside it: the two latents, the samplers'
  // history (two x0 a modality and the last sample) and four velocities.
  const std::size_t lat =
      ((std::size_t)c.video_c * T * h * w + (std::size_t)c.audio_c * L) * 4;
  return MetalMagi2Dit::scratch_bytes(c, n, 2, inputs) + 8 * lat;
}

int
resident_moe_for(const DitPlan& p, std::size_t room)
{
  if (p.moe_layers <= 0 || p.expert_layer == 0) { return -1; }
  const std::size_t all = (std::size_t)p.moe_layers * p.expert_layer;
  if (room >= all) { return -1; }
  // The two slots come out of the room first.
  if (room <= 2 * p.expert_layer) { return 0; }
  const int n = (int)((room - 2 * p.expert_layer) / p.expert_layer);
  return std::min(n, p.moe_layers - 1);
}

std::size_t
retention_room(
    const DitPlan& p, int tier, std::size_t ram,
    const std::vector<std::pair<std::string, std::size_t>>& phase_peaks,
    std::size_t own_beside)
{
  const std::size_t own = dit_floor_bytes(p, tier);
  std::size_t beside = own_beside;
  for (const auto& [phase, bytes] : phase_peaks) {
    (void)phase;
    beside = std::max(beside, bytes > own ? bytes - own : 0);
  }
  const std::size_t used =
      model_memory::kStreamHeadroom + beside + p.trunk(tier);
  return ram > used ? ram - used : 0;
}

RefineAsk
refine_ask(const FlexData* mc)
{
  RefineAsk a;
  if (mc == nullptr || !mc->is_object()) { return a; }
  FlexData cfg = *mc;               // as_object() is a view: keep it
  auto o = cfg.as_object();
  auto num = [&](const char* k, double def) {
    if (!o.contains(k)) { return def; }
    const FlexData v = o.at(k);
    return v.is_int() ? (double)v.as_int(0) : v.as_real(def);
  };
  a.width = (int)num("refine_width", 0.0);
  a.height = (int)num("refine_height", 0.0);
  a.steps = (int)num("refine_steps", (double)kRefineSteps);
  a.shift = num("refine_shift", kRefineShift);
  a.guidance = num("refine_guidance", kRefineGuidance);
  if (a.steps <= 0) { a.steps = kRefineSteps; }
  return a;
}

int
refined_frames(int frames)
{
  const int T = latent_frames(frames);
  return T <= 0 ? 0 : 4 * (refine_latent_frames(T) - 1) + 1;
}

bool
refiner_plan(const std::string& refiner_dir, RefinerPlan* out)
{
  static std::mutex mu;
  static std::map<std::string, RefinerPlan> memo;
  {
    std::lock_guard<std::mutex> lk(mu);
    auto it = memo.find(refiner_dir);
    if (it != memo.end()) {
      if (out != nullptr) { *out = it->second; }
      return true;
    }
  }
  auto ws = WeightSet::open(refiner_dir, nullptr);
  if (ws == nullptr) { return false; }
  RefinerPlan p;
  if (!MetalMagi2Refiner::config_from_weights(*ws, &p.cfg, nullptr)) {
    return false;
  }
  for (int t = -1; t <= 3; ++t) {
    MetalMagi2Refiner::Config c = p.cfg;
    c.i8_tier = t;
    p.weights_at[t + 1] = MetalMagi2Refiner::weight_bytes(*ws, c);
  }
  std::lock_guard<std::mutex> lk(mu);
  memo[refiner_dir] = p;
  if (out != nullptr) { *out = p; }
  return true;
}

bool
refiner_plan(const std::string& refiner_dir, const TpShare& preview_share,
             RefinerPlan* out)
{
  RefinerPlan whole;
  if (!refiner_plan(refiner_dir, &whole)) { return false; }
  if (!preview_share.active()) {
    if (out != nullptr) { *out = whole; }
    return true;
  }
  MetalMagi2Refiner::Config c = whole.cfg;
  c.tp = MetalMagi2Refiner::share_at(whole.cfg, preview_share);
  MetalMagi2Refiner::Share sh;
  if (!MetalMagi2Refiner::share_of(c, &sh, nullptr)) { return false; }
  static std::mutex mu;
  static std::map<std::string, RefinerPlan> memo;
  const std::string key =
      fmt("{}|{}/{}|{}-{}|{}-{}", refiner_dir, c.tp.rank, c.tp.ranks,
          c.tp.head0, c.tp.head1, c.tp.ffn0, c.tp.ffn1)();
  {
    std::lock_guard<std::mutex> lk(mu);
    auto it = memo.find(key);
    if (it != memo.end()) {
      if (out != nullptr) { *out = it->second; }
      return true;
    }
  }
  auto ws = WeightSet::open(refiner_dir, nullptr);
  if (ws == nullptr) { return false; }
  RefinerPlan p = whole;
  p.cfg = c;
  for (int t = -1; t <= 3; ++t) {
    MetalMagi2Refiner::Config ct = c;
    ct.i8_tier = t;
    p.weights_at[t + 1] = MetalMagi2Refiner::weight_bytes(*ws, ct);
  }
  std::lock_guard<std::mutex> lk(mu);
  memo[key] = p;
  if (out != nullptr) { *out = p; }
  return true;
}

std::size_t
refine_scratch_bytes(const RefinerPlan& p, const RefineAsk& ask, int frames,
                     const Accel& a)
{
  if (!ask.on()) { return 0; }
  const int To = refine_latent_frames(latent_frames(frames));
  const int H = ask.grid_h(), W = ask.grid_w();
  if (To <= 0) { return 0; }
  // The loop's host state: the latent, its noise while drawn, the two
  // velocities and the sampler's history (two x0 and the last sample).
  const std::size_t lat =
      (std::size_t)p.cfg.video_c * To * H * W * sizeof(float);
  MetalMagi2Refiner::Config c = p.cfg;
  c.i8_tier = a.i8_tier;
  c.autotune = a.autotune;
  c.sage = a.sage;
  return MetalMagi2Refiner::scratch_bytes(c, To, H, W, kMaxTextTokens + 1,
                                          true) +
         7 * lat;
}

namespace {

float
bf16_(std::uint16_t b)
{
  const std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

double
cfg_real_(const FlexData* mc, const char* key, double def)
{
  if (mc == nullptr || !mc->is_object()) { return def; }
  auto o = mc->as_object();
  if (!o.contains(key)) { return def; }
  const FlexData v = o.at(key);
  return v.is_int() ? (double)v.as_int(0) : v.as_real(def);
}

class Magi2Generator final : public VideoGenerator,
                             public VideoHeldBySource,
                             public VideoTpClip {
 public:
  Magi2Generator(std::unique_ptr<MetalMagi2Dit> dit,
                 const SessionContextIntf* session,
                 metal_compute::MetalCompute* mc, const Layout& layout,
                 const Accel& accel, const TpShare& tp = {})
      : _dit(std::move(dit)), _session(session), _mc(mc), _layout(layout),
        _accel(accel), _tp(tp)
  {}

  ~Magi2Generator() override { drop_refiner_(); }

  int latent_channels() const override { return kVideoChannels; }
  int spatial_compression() const override { return kSpatialStride; }

  bool
  generate(const VideoGenRequest& req, VideoGenResult* out) override
  {
    auto refuse = [&](const std::string& why) {
      if (_session != nullptr) { _session->warn(fmt("magi2: {}", why)); }
      return false;
    };
    if (_dit == nullptr || out == nullptr) { return refuse("not loaded"); }
    const MetalMagi2Dit::Config& c = _dit->config();
    if (req.cond == nullptr || req.cond_rows <= 0) {
      return refuse("no conditioning; wire a diffusion-conditioner on the "
                    "same MAGI-2 root to iport0 (and its negative, oport1, "
                    "to iport1)");
    }
    if (req.cond_dim != c.text_c) {
      return refuse(fmt("the conditioning is {} wide where this DiT reads "
                        "{} -- the conditioner resolved another model",
                        req.cond_dim, c.text_c)());
    }
    if (req.height % kGrid != 0 || req.width % kGrid != 0 ||
        req.height <= 0 || req.width <= 0) {
      return refuse(fmt("{}x{} is not on the {}-pixel grid", req.width,
                        req.height, kGrid)());
    }
    const int h = req.height / kGrid, w = req.width / kGrid;
    const int F = align_frames(req.frames);
    const int T = latent_frames(F), L = audio_latents_of(F);
    // The pictures' special tokens: the conditioning's last rows.
    int figs = 0;
    if (req.cond_sideband != nullptr && req.cond_sideband->is_object()) {
      auto o = req.cond_sideband->as_object();
      if (o.contains(kFigureRows)) {
        figs = (int)o.at(kFigureRows).as_int(0);
      }
    }
    if (figs < 0 || figs >= req.cond_rows) {
      return refuse(fmt("the conditioning says {} of its {} rows are the "
                        "pictures' tokens", figs, req.cond_rows)());
    }
    if (figs > 1) {
      return refuse("one picture is what image-to-video takes here");
    }
    if (figs > 0 && req.ref == nullptr) {
      return refuse("the prompt was conditioned for image-to-video (the "
                    "picture was wired to the conditioner), but no latent of "
                    "it reached generate-video's ref port -- wire a "
                    "vae-encode of the same picture there");
    }
    if (figs == 0 && req.ref != nullptr) {
      return refuse("a picture reached the ref port, but the prompt was "
                    "conditioned without it -- wire the same picture to the "
                    "conditioner's ref_image, which addresses it from the "
                    "prompt (\"<Figure 1>\")");
    }
    int hi = 0, wi = 0;
    if (req.ref != nullptr) {
      const std::vector<int>& s = req.ref_shape;
      const bool four = s.size() == 4 && s[0] == c.video_c && s[1] == 1;
      const bool three = s.size() == 3 && s[0] == c.video_c;
      if (!four && !three) {
        return refuse("the picture's latent is not a [48, 1, h, w] Wan2.2 "
                      "latent");
      }
      hi = four ? s[2] : s[1];
      wi = four ? s[3] : s[2];
      // The plan books the picture's tokens at the clip's long edge
      // squared (denoise_scratch_bytes).
      if (std::max(hi, wi) > std::max(h, w)) {
        return refuse(fmt("the picture's latent is {}x{} cells, past the "
                          "clip's long edge ({} cells); resize it so its "
                          "long edge is the clip's", wi, hi,
                          std::max(h, w))());
      }
    }
    const double steps_cfg = cfg_real_(req.model_config, "steps", 0.0);
    const int steps = steps_cfg > 0 ? (int)steps_cfg
                      : req.steps > 0 ? req.steps
                                      : kPreviewSteps;
    const RefineAsk ra = refine_ask(req.model_config);
    if ((ra.width > 0 || ra.height > 0) && !ra.on()) {
      return refuse(fmt("refine_width / refine_height {}x{}: give both, "
                        "each at least {} pixels", ra.width, ra.height,
                        kGrid)());
    }
    if (ra.on() && _layout.refiner.empty()) {
      return refuse("a refine was asked for, but this root has no "
                    "refiner/");
    }

    const int total_steps = steps + (ra.on() ? ra.steps : 0);
    if (F != kTrainedFrames && !_said_length) {
      _said_length = true;
      if (_session != nullptr) {
        _session->info(fmt("magi2: a {}-frame clip ({:.2f} s); the model "
                           "was trained on {} frames (10 s) and its release "
                           "supports no other length", F, F / kFps,
                           kTrainedFrames));
      }
    }

    // The rows, bf16 to f32.
    auto rows_f32 = [&](const void* src, int rows) {
      std::vector<float> v((std::size_t)rows * c.text_c);
      const auto* p = (const std::uint16_t*)src;
      for (std::size_t i = 0; i < v.size(); ++i) { v[i] = bf16_(p[i]); }
      return v;
    };
    const std::vector<float> cond = rows_f32(req.cond, req.cond_rows);
    std::vector<float> neg;
    if (req.neg != nullptr && req.neg_rows > 0) {
      neg = rows_f32(req.neg, req.neg_rows);
    }

    PreviewRequest r;
    r.T = T;
    r.h = h;
    r.w = w;
    r.audio_len = L;
    r.steps = steps;
    r.shift = cfg_real_(req.model_config, "shift", kPreviewShift);
    r.video_guidance =
        cfg_real_(req.model_config, "video_guidance", kVideoGuidance);
    r.audio_guidance =
        cfg_real_(req.model_config, "audio_guidance", kAudioGuidance);
    r.seed = req.seed;
    r.cond = cond.data();
    r.cond_rows = req.cond_rows - figs;
    r.neg = neg.empty() ? nullptr : neg.data();
    r.neg_rows = neg.empty() ? 0 : req.neg_rows;
    if (figs > 0) {
      MetalMagi2Dit::Image im;
      im.latent = req.ref;
      im.h = hi;
      im.w = wi;
      im.special = cond.data() + (std::size_t)r.cond_rows * c.text_c;
      r.images.push_back(im);
    }
    if (req.progress) {
      r.progress = [&req, total_steps](int s, int) {
        return req.progress(s, total_steps);
      };
    }
    _dit->set_block_progress(req.block_progress);

    std::vector<float> video, audio;
    std::string err;
    const bool ok = run_preview(*_dit, r, &video, &audio, &err);
    _dit->set_block_progress({});
    if (!ok) {
      return err.empty() ? false : refuse(err);
    }
    if (ra.on()) {
      std::vector<float> refined;
      if (!refine_(req, ra, video, T, h, w, cond, req.cond_rows - figs, neg,
                   steps, total_steps, &refined, &err)) {
        return err.empty() ? false : refuse(err);
      }
      const int To = refine_latent_frames(T);
      out->video = std::move(refined);
      out->video_shape = {c.video_c, To, ra.grid_h(), ra.grid_w()};
      // The clip at twice the preview's rate: what the stage stamps.
      out->sideband = FlexData::make_object();
      out->sideband.as_object().insert_or_assign(
          sideband::kFps, FlexData::make_real(kRefineFps));
      out->sideband.as_object().insert_or_assign(
          sideband::kFrames, FlexData::make_int(4 * (To - 1) + 1));
    } else {
      out->video = std::move(video);
      out->video_shape = {c.video_c, T, h, w};
    }
    // [L, 64] token-major to the decoder's [1, 64, L].
    out->audio.assign((std::size_t)c.audio_c * L, 0.0f);
    for (int t = 0; t < L; ++t) {
      for (int ch = 0; ch < c.audio_c; ++ch) {
        out->audio[(std::size_t)ch * L + t] =
            audio[(std::size_t)t * c.audio_c + ch];
      }
    }
    out->audio_shape = {1, c.audio_c, L};
    out->latents_per_second = (double)kAudioLatentRate;
    return true;
  }

  void release_idle() override
  {
    if (_dit) { _dit->release_scratch(); }
    if (_refiner) { _refiner->release_scratch(); }
  }

  std::uint64_t resident_bytes() const override
  {
    return (_dit ? _dit->resident_bytes() : 0) +
           (_refiner ? _refiner->resident_bytes() : 0);
  }

  // ---- VideoHeldBySource: the preview DiT and the refiner apart -------
  std::size_t held_bytes(const std::string& source) const override
  {
    if (source == _layout.preview) {
      return _dit ? _dit->resident_bytes() : 0;
    }
    if (!_layout.refiner.empty() && source == _layout.refiner) {
      return _refiner ? _refiner->resident_bytes() : 0;
    }
    return 0;
  }

  void* query_extension(std::string_view id) noexcept override
  {
    if (id == kVideoHeldBySourceExt) {
      return static_cast<VideoHeldBySource*>(this);
    }
    if (id == kVideoTpClipExt && _tp.active()) {
      return static_cast<VideoTpClip*>(this);
    }
    return nullptr;
  }

  // ---- VideoTpClip: a rank's channel for one clip ----------------------
  // The preview DiT takes the clip's channel first; a refine takes it
  // over at the exchange count the preview reached (refine_).
  bool begin_tp_generation(std::shared_ptr<TpChannel> ch,
                           std::function<bool()> stop,
                           std::string* err) override
  {
    if (_dit == nullptr) {
      if (err != nullptr) { *err = "tensor parallel: not loaded"; }
      return false;
    }
    _tp_ch = ch;
    _tp_stop = stop;
    return _dit->begin_tp_generation(std::move(ch), std::move(stop), err);
  }
  void end_tp_generation() override
  {
    if (_dit != nullptr) { _dit->end_tp_generation(); }
    if (_refiner != nullptr) { _refiner->end_tp_generation(); }
    _tp_ch.reset();
    _tp_stop = {};
  }

 private:
  // The second stage: the preview's latent refined on the asked grid.
  // The preview DiT's scratch goes first; the refiner loads at the first
  // refine (through the manager) and stays with the generator.
  bool
  refine_(const VideoGenRequest& req, const RefineAsk& ra,
          const std::vector<float>& video, int T, int h, int w,
          const std::vector<float>& cond, int cond_rows,
          const std::vector<float>& neg, int steps_done, int total_steps,
          std::vector<float>* out, std::string* err)
  {
    _dit->release_scratch();
    if (!_refiner) {
      RefinerPlan rp;
      if (!refiner_plan(_layout.refiner, &rp)) {
        *err = "cannot read the refiner's shapes under " + _layout.refiner;
        return false;
      }
      auto ws = open_weight_set(_layout.refiner, _session);
      if (ws == nullptr) {
        *err = "cannot open " + _layout.refiner;
        return false;
      }
      MetalMagi2Refiner::Config rc = rp.cfg;
      rc.i8_tier = _accel.i8_tier;
      rc.autotune = _accel.autotune;
      rc.sage = _accel.sage;
      // A rank holds the refiner's share at the preview's fractions.
      rc.tp = MetalMagi2Refiner::share_at(rp.cfg, _tp);
      _refiner = MetalMagi2Refiner::load(ws, _mc, rc, err);
      if (!_refiner) { return false; }
      if (_session != nullptr && rc.tp.active()) {
        _session->log_normal(fmt(
            "magi2: the refiner's share on rank {}: query heads [{}, {}) "
            "of {}, MLP rows [{}, {}) of {}", rc.tp.rank, rc.tp.head0,
            rc.tp.head1, rp.cfg.n_q, rc.tp.ffn0, rc.tp.ffn1,
            rp.cfg.mlp_inter));
      }
      if (_session != nullptr) {
        _session->log_normal(fmt(
            "magi2: refiner, {} layers at {} wide ({} q / {} kv heads), "
            "{} MB held; {}, {}; int8 GEMMs {}; SageAttention {}",
            rp.cfg.layers.size(), rp.cfg.hidden, rp.cfg.n_q, rp.cfg.n_kv,
            _refiner->resident_bytes() >> 20,
            _refiner->uses_mma() ? "matmul2d GEMMs" : "steel GEMMs",
            _refiner->uses_nax_attention() ? "matrix-core window attention"
                                           : "steel window attention",
            tier_name(_refiner->i8_tier()),
            !_refiner->sage_on() ? "off"
            : rc.sage.pv8        ? "int8 QK and P*V"
                                 : "int8 QK"));
      }
    }
    if (_session != nullptr) {
      _session->log_normal(fmt(
          "magi2: refining to {}x{} ({}x{} cells, {} latent frames at {} "
          "fps): {} steps at shift {}, guidance {}",
          ra.grid_w() * kGrid, ra.grid_h() * kGrid, ra.grid_w(),
          ra.grid_h(), refine_latent_frames(T), kRefineFps, ra.steps,
          ra.shift, ra.guidance));
    }
    RefineRequest rr;
    rr.T = T;
    rr.h = h;
    rr.w = w;
    rr.latent = video.data();
    rr.H = ra.grid_h();
    rr.W = ra.grid_w();
    rr.steps = ra.steps;
    rr.shift = ra.shift;
    rr.guidance = ra.guidance;
    // The re-noise is drawn from a generator of its own: the release's
    // continues CUDA's global one, which no CPU draw reproduces.
    rr.seed = req.seed + 1;
    rr.cond = cond.data();
    rr.cond_rows = cond_rows;
    rr.neg = neg.empty() ? nullptr : neg.data();
    rr.neg_rows = neg.empty() ? 0 : (int)(neg.size() / _refiner->config()
                                                        .text_c);
    if (req.progress) {
      rr.progress = [&req, steps_done, total_steps](int s, int) {
        return req.progress(steps_done + s, total_steps);
      };
    }
    // Its layers on the preview's scale, so the stage's bar -- steps x
    // the preview's layers -- keeps one denominator through both stages.
    if (req.block_progress) {
      const int dl = (int)_dit->config().layers.size();
      _refiner->set_block_progress([&req, dl](int done, int total) {
        const int d = total > 0 ? (int)((long)done * dl / total) : done;
        return req.block_progress(d, dl);
      });
    }
    // A RANK: the clip's channel from the preview DiT to the refiner, at
    // the exchange count the preview reached on every rank alike.
    if (_tp.active()) {
      const std::uint64_t seq = _dit->tp_seq();
      _dit->end_tp_generation();
      if (!_tp_ch ||
          !_refiner->begin_tp_generation(_tp_ch, _tp_stop, seq, err)) {
        if (err != nullptr && err->empty()) {
          *err = "tensor parallel: no channel for the refine";
        }
        return false;
      }
    }
    const bool ok = run_refine(*_refiner, rr, out, err);
    _refiner->set_block_progress({});
    _refiner->release_scratch();
    return ok;
  }

  void
  drop_refiner_()
  {
    if (!_refiner) { return; }
    _refiner.reset();
    if (_session == nullptr) { return; }
    if (auto* sv = _session->services()) {
      if (auto* mgr = sv->generative_model_manager()) {
        mgr->drop_weights(_layout.refiner);
      }
    }
  }

  std::unique_ptr<MetalMagi2Dit> _dit;
  const SessionContextIntf* _session = nullptr;
  metal_compute::MetalCompute* _mc = nullptr;
  Layout _layout;
  Accel _accel;
  // A rank's share (tensor parallel); inactive, the whole model. The
  // clip's channel and stop while one is open, for the refine.
  TpShare _tp;
  std::shared_ptr<TpChannel> _tp_ch;
  std::function<bool()> _tp_stop;
  std::unique_ptr<MetalMagi2Refiner> _refiner;
  bool _said_length = false;
};

class Magi2VideoFamily final : public VideoModelFamily,
                               public VideoConfigPlan,
                               public VideoTensorParallel {
 public:
  void* query_extension(std::string_view id) noexcept override
  {
    if (id == kVideoConfigPlanExt) {
      return static_cast<VideoConfigPlan*>(this);
    }
    if (id == kVideoTensorParallelExt) {
      return static_cast<VideoTensorParallel*>(this);
    }
    return nullptr;
  }

  // ---- VideoTensorParallel: the preview DiT split across hosts ---------
  // A share's heads are the attention's and its feed-forward rows count
  // EXPERTS -- of each MoE head's (MetalMagi2Dit::Config::tp).
  bool tp_widths(const std::string& root, int* n_heads, int* n_ffn,
                 std::string* err) const override
  {
    Layout l;
    DitPlan p;
    if (!resolve_layout(root, &l) || !dit_plan(l.preview, &p)) {
      if (err != nullptr) {
        *err = "cannot read the preview DiT's shapes under " + root;
      }
      return false;
    }
    *n_heads = p.cfg.n_q;
    *n_ffn = p.cfg.moe_experts;
    return true;
  }
  std::unique_ptr<VideoGenerator>
  load_rank(const VideoModelCreateArgs& args, const TpShare& share) override
  {
    return load_(args, share);
  }

  // ---- VideoConfigPlan: the refiner, when the model config asks, and
  //      both DiTs at the int8 tier the acceleration bag asks for ---------
  std::vector<ResourceClaim>
  config_resources(const std::string& root,
                   const VideoPlanContext& ctx) const override
  {
    const Accel a = accel_of(ctx.accel, ctx.matrix_cores);
    std::vector<ResourceClaim> out = dit_claims_(root, a.i8_tier, ctx.tp);
    Layout l;
    RefinerPlan rp;
    if (refine_ask(ctx.model_config).on() && resolve_layout(root, &l) &&
        !l.refiner.empty() &&
        refiner_plan(l.refiner, ctx.tp != nullptr ? *ctx.tp : TpShare{},
                     &rp)) {
      // Booked at what the load holds: the tensors it reads, the
      // projections at the tier's layout.
      for (auto& c : model_memory::weight_claims({l.refiner})) {
        bag::set_integer(c.extra, model_memory::kClaimHeldBytes,
                         (std::int64_t)rp.weights(a.i8_tier));
        out.push_back(std::move(c));
      }
    }
    return out;
  }

  std::vector<StageHolding>
  config_holdings(const std::string& root,
                  const VideoPlanContext& ctx) const override
  {
    const Accel a = accel_of(ctx.accel, ctx.matrix_cores);
    std::vector<StageHolding> out = dit_holdings_(root, a.i8_tier, ctx.tp);
    Layout l;
    RefinerPlan rp;
    if (refine_ask(ctx.model_config).on() && resolve_layout(root, &l) &&
        !l.refiner.empty() &&
        refiner_plan(l.refiner, ctx.tp != nullptr ? *ctx.tp : TpShare{},
                     &rp)) {
      StageHolding h;
      h.source = l.refiner;
      h.preload = rp.weights(a.i8_tier);
      h.floor = rp.weights(a.i8_tier);
      out.push_back(h);
    }
    return out;
  }

  std::size_t
  config_denoise_scratch_bytes(const std::string& root, int width,
                               int height, int frames,
                               const VideoPlanContext& ctx) const override
  {
    return denoise_scratch_(root, width, height, frames, ctx.model_config,
                            accel_of(ctx.accel, ctx.matrix_cores), ctx.tp);
  }

  bool output_clip(const std::string&, const FlexData* model_config,
                   int, int, int frames, int* ow, int* oh, int* of,
                   double* ofps) const override
  {
    const RefineAsk ra = refine_ask(model_config);
    if (!ra.on()) { return false; }
    *ow = ra.grid_w() * kGrid;
    *oh = ra.grid_h() * kGrid;
    *of = refined_frames(frames);
    *ofps = kRefineFps;
    return true;
  }

  std::size_t decode_working_bytes(const std::string& root, int width,
                                   int height, int frames) const override
  {
    Layout l;
    std::size_t scratch = 0;
    if (!resolve_layout(root, &l) ||
        !MetalMagi2Turbo::plan(l.turbo, latent_frames(frames),
                               height / kGrid, width / kGrid, nullptr,
                               &scratch)) {
      return 0;
    }
    return scratch;
  }

  std::string_view tag() const noexcept override { return "magi2"; }

  bool claims(const std::string& root,
              const std::string& model_type) const override
  {
    (void)model_type;   // the files decide: no record makes a root
    return is_root(root);
  }

  int align_frames(const std::string&, int frames) const override
  {
    return magi2::align_frames(frames);
  }

  void size_grid(const std::string&, int* gh, int* gw) const override
  {
    if (gh != nullptr) { *gh = kGrid; }
    if (gw != nullptr) { *gw = kGrid; }
  }

  std::vector<ResourceClaim>
  declare_resources(const std::string& root) const override
  {
    return dit_claims_(root, -1);
  }

  std::vector<StageHolding>
  declare_holdings(const std::string& root) const override
  {
    return dit_holdings_(root, -1);
  }

  std::size_t latent_bytes(const std::string&, int width, int height,
                           int frames) const override
  {
    const int T = latent_frames(frames);
    return (std::size_t)kVideoChannels * T * (height / kGrid) *
           (width / kGrid) * sizeof(float);
  }

  std::size_t
  denoise_scratch_bytes(const std::string& root, int width, int height,
                        int frames, const FlexData* model_config) const
      override
  {
    return denoise_scratch_(root, width, height, frames, model_config,
                            Accel{});
  }

  bool audio_cost(const std::string& root, int frames, double,
                  std::size_t* latent, std::size_t* pcm,
                  std::size_t* arena) const override
  {
    Layout l;
    MetalOobleckDecoder::Config ac;
    if (!resolve_layout(root, &l) ||
        !MetalOobleckDecoder::config_from_dir(l.audio, &ac, nullptr)) {
      return false;
    }
    const int L = audio_latents_of(frames);
    const long raw = (long)L * ac.hop();
    const long num = stretched_length(raw);
    if (latent != nullptr) {
      *latent = (std::size_t)kAudioChannels * L * sizeof(float);
    }
    if (pcm != nullptr) {
      *pcm = (std::size_t)ac.out_channels * num * sizeof(float);
    }
    if (arena != nullptr) {
      // The whole soundtrack in one tile (as the release decodes it), its
      // raw PCM beside the stretched one, and the resample's double
      // buffers and spectra.
      std::size_t raw_pcm = 0, dec = 0;
      MetalOobleckDecoder::decode_cost(ac, L, L, &raw_pcm, &dec);
      *arena = dec + raw_pcm +
               (std::size_t)(raw + num) * 8 +
               (std::size_t)(raw / 2 + num / 2 + 2) * 16;
    }
    return true;
  }

  // The preview DiT's claim and holding at the int8 tier `tier`. It
  // streams its experts on any box short of ~215 GB, so its claim carries
  // the floor it streams at: the trunk and two slots; a tier that holds
  // the projections in its layout holds less than the checkpoint weighs.
  // A rank's (`tp`): its share's figures (dit_plan's share overload).
  static bool plan_of_(const std::string& root, const TpShare* tp,
                       Layout* l, DitPlan* p)
  {
    return resolve_layout(root, l) &&
           (tp != nullptr && tp->active() ? dit_plan(l->preview, *tp, p)
                                          : dit_plan(l->preview, p));
  }
  std::vector<ResourceClaim> dit_claims_(const std::string& root, int tier,
                                         const TpShare* tp = nullptr) const
  {
    Layout l;
    DitPlan p;
    if (!plan_of_(root, tp, &l, &p)) { return {}; }
    const bool rank = tp != nullptr && tp->active();
    ResourceClaim c = model_memory::weight_claim_streamable(
        l.preview, dit_floor_bytes(p, tier));
    // A rank holds its share at most, never the checkpoint.
    if (tier >= 1 || rank) {
      bag::set_integer(c.extra, model_memory::kClaimHeldBytes,
                       (std::int64_t)dit_preload_bytes(p, tier));
    }
    return {std::move(c)};
  }

  std::vector<StageHolding> dit_holdings_(const std::string& root,
                                          int tier,
                                          const TpShare* tp = nullptr) const
  {
    Layout l;
    DitPlan p;
    if (!plan_of_(root, tp, &l, &p)) { return {}; }
    StageHolding h;
    h.source = l.preview;
    h.preload = dit_preload_bytes(p, tier);
    h.floor = dit_floor_bytes(p, tier);
    return {h};
  }

  std::size_t denoise_scratch_(const std::string& root, int width,
                               int height, int frames,
                               const FlexData* model_config,
                               const Accel& a,
                               const TpShare* tp = nullptr) const
  {
    Layout l;
    DitPlan p;
    if (!plan_of_(root, tp, &l, &p)) { return 0; }
    std::size_t s = magi2::denoise_scratch_bytes(p, width, height, frames,
                                                 a);
    // A refine runs after the preview has let its scratch go: the larger
    // of the two, with the preview's latent carried across.
    const RefineAsk ra = refine_ask(model_config);
    RefinerPlan rp;
    if (ra.on() && !l.refiner.empty() &&
        refiner_plan(l.refiner, tp != nullptr ? *tp : TpShare{}, &rp)) {
      s = std::max(s, refine_scratch_bytes(rp, ra, frames, a) +
                          latent_bytes(root, width, height, frames));
    }
    return s;
  }

  // THE RETAINED EXPERTS' ROOM (magi2::retention_room): the plan's widest
  // phase beside them, and this request's own forward and refiner. Not
  // plan_streaming's `others`: that counts this DiT too, at what it
  // DECLARED (its whole checkpoint), which left no room on any box.
  std::size_t
  retention_room_(const VideoModelCreateArgs& args, const DitPlan& p,
                  const Accel& acc, int W, int H, int Fr, std::size_t ram,
                  const TpShare* tp) const
  {
    std::size_t own =
        denoise_scratch_(args.root, W, H, Fr, args.model_config, acc, tp);
    Layout l;
    RefinerPlan rp;
    if (refine_ask(args.model_config).on() &&
        resolve_layout(args.root, &l) && !l.refiner.empty() &&
        refiner_plan(l.refiner, tp != nullptr ? *tp : TpShare{}, &rp)) {
      own += rp.weights(acc.i8_tier);
    }
    std::vector<std::pair<std::string, std::size_t>> by;
    if (args.session != nullptr && args.session->services() != nullptr) {
      if (auto* mgr = args.session->services()->generative_model_manager()) {
        mgr->phase_peak(&by);
      }
    }
    return magi2::retention_room(p, acc.i8_tier, ram, by, own);
  }

  std::unique_ptr<VideoGenerator>
  load(const VideoModelCreateArgs& args) override
  {
    return load_(args, TpShare{});
  }

  // The whole model, or a RANK's share (`share` active).
  std::unique_ptr<VideoGenerator>
  load_(const VideoModelCreateArgs& args, const TpShare& share)
  {
    auto fail = [&](const std::string& why)
        -> std::unique_ptr<VideoGenerator> {
      if (args.session != nullptr) {
        args.session->error(fmt("magi2: {}", why));
      }
      return nullptr;
    };
    if (args.metal == nullptr) { return fail("no metal-compute backend"); }
    Layout l;
    if (!resolve_layout(args.root, &l)) {
      return fail(fmt("'{}' is not a MAGI-2-preview root (preview/, "
                      "text_encoder/, turbo_vae/ and stable-audio-open-1.0/ "
                      "beside each other)", args.root)());
    }
    DitPlan p;
    std::string perr;
    if (!dit_plan(l.preview, share, &p, &perr)) {
      return fail(perr.empty()
                      ? fmt("cannot read the preview DiT's shapes under "
                            "'{}'", l.preview)()
                      : perr);
    }
    // THE ACCELERATION: the int8 tier on both DiTs' block projections and
    // SageAttention on their matrix-core attention, as the box runs them.
    const Accel acc =
        accel_of(args.accel, args.metal->supports_matrix_cores());
    const int W = args.width > 0 ? args.width : 896;
    const int H = args.height > 0 ? args.height : 512;
    const int Fr = args.frames > 0 ? args.frames : kTrainedFrames;
    const std::size_t ram = model_memory::phys_ram();
    MetalMagi2Dit::Config cfg = p.cfg;
    cfg.i8_tier = acc.i8_tier;
    cfg.autotune = acc.autotune;
    cfg.sage = acc.sage;
    cfg.resident_moe =
        ram == 0 ? -1
                 : resident_moe_for(
                       p, retention_room_(args, p, acc, W, H, Fr, ram,
                                          &share));
    if (const char* e = std::getenv("VPIPE_MAGI2_RESIDENT_MOE")) {
      if (*e != '\0') { cfg.resident_moe = std::atoi(e); }
    }
    auto ws = open_weight_set(l.preview, args.session);
    if (ws == nullptr) { return fail("cannot open " + l.preview); }
    std::string err;
    auto dit = MetalMagi2Dit::load(ws, args.metal, cfg, &err);
    if (dit == nullptr) { return fail(err.empty() ? "load failed" : err); }
    if (args.session != nullptr) {
      args.session->log_normal(fmt(
          "magi2: preview DiT, {} layers at {} wide; {} of {} MoE layers' "
          "experts held, {} streamed ({} MB a layer, read once a step for "
          "both guidance passes); {} MB held; {}, {}, {}",
          cfg.layers.size(), cfg.hidden, dit->moe_held(), p.moe_layers,
          dit->moe_streamed(), p.expert_layer >> 20,
          dit->resident_bytes() >> 20,
          dit->uses_mma() ? "matmul2d GEMMs" : "steel GEMMs",
          dit->uses_nax_attention() ? "matrix-core attention"
                                    : "steel attention",
          dit->uses_nax_experts()      ? "matrix-core experts"
          : dit->uses_staged_experts() ? "staged simdgroup experts"
                                       : "simdgroup experts"));
      // The tiers asked for, as they run -- and the ones not taken here.
      if (accel::flag(args.accel, accel::kI8Gemm, false)) {
        args.session->log_normal(fmt(
            "magi2: int8 GEMMs {} on the block projections of both DiTs "
            "(the routed experts keep their bf16 kernel){}",
            tier_name(dit->i8_tier()),
            dit->i8_tier() < 0 ? " -- this GPU has no matrix cores" : ""));
      }
      if (accel::flag(args.accel, accel::kSageAttn, false)) {
        args.session->log_normal(fmt(
            "magi2: SageAttention {}",
            dit->sage_on() ? (acc.sage.pv8 ? "on, int8 QK and P*V"
                                           : "on, int8 QK")
                           : "off -- it runs on the matrix-core attention "
                             "alone"));
      }
      for (std::string_view t : accel::tiers_on(args.accel)) {
        if (t == accel::kI8Gemm || t == accel::kSageAttn) { continue; }
        args.session->log_normal(
            fmt("magi2: {} is not implemented here", t));
      }
      if (share.active()) {
        MetalMagi2Dit::Share sh;
        MetalMagi2Dit::share_of(cfg, &sh, nullptr);
        args.session->log_normal(fmt(
            "magi2: rank {} of {}: query heads [{}, {}) of {}, experts [{}, "
            "{}) of each MoE head's {}, the dense MLPs at the same share; "
            "two partial sums a layer exchanged with the other ranks",
            share.rank, share.ranks, sh.h0, sh.h1, cfg.n_q, sh.e0, sh.e1,
            cfg.moe_experts));
      }
    }
    return std::make_unique<Magi2Generator>(std::move(dit), args.session,
                                            args.metal, l, acc, share);
  }
};

const int _magi2_family_reg = [] {
  VideoModelRegistry::get().add(std::make_unique<Magi2VideoFamily>());
  return 0;
}();

}  // namespace

}  // namespace magi2
}  // namespace genai
}  // namespace vpipe

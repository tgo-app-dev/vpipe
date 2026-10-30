#include "stages/latent-upscale-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-keys.h"
#include "common/beat-payload-intf.h"
#include "common/flex-bag.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/minimax-h3/metal-minimax-h3-latent-upscaler.h"
#include "generative-models/weight-set.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "stages/model-memory-host.h"
#include "stages/model-memory.h"
#include "stages/model-registry.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vpipe {

namespace {

namespace fs = std::filesystem;
using Upscaler = genai::MetalMiniMaxH3LatentUpscaler;

constexpr unsigned kModelPort = 1;

constexpr SpecExtra kNormChoices[] = {
  {spec_key::kChoices, "release,once"},
};
constexpr SpecExtra kUnloadChoices[] = {
  {spec_key::kChoices, "auto,destroy,keep"},
};
constexpr SpecExtra kI8TierChoices[] = {
  {spec_key::kChoices, "accurate,fast,token,token_plain"},
};

const ConfigKey kAttrs[] = {
  {.key = "hf_dir", .type = ConfigType::String, .required = false,
   .doc = "the latent upscaler: a models-DB key, the Open-H3-Regenerate-2K "
          "release (Efficient-Large-Model/Open-H3-Regenerate-2K: its "
          "latent_upsampler/), the upscaler's own repo, or the "
          ".safetensors itself. OPTIONAL: a model-select source on the "
          "model iport overrides it",
   .suggest_db = kModelRegistryDb,
   .suggest_db_type = "minimax-h3-latent-upscaler"},
  {.key = "width", .type = ConfigType::Int, .required = false,
   .doc = "target width in PIXELS (with `height`). Rounded as the "
          "upscaler's node rounds it: to `align`, then to whole 16-px "
          "latent cells", .def_int = 2560},
  {.key = "height", .type = ConfigType::Int, .required = false,
   .doc = "target height in pixels. A height that is not whole 32-px "
          "cells computes at the next (1080 -> 1088, 2160 -> 2176) and "
          "the latent carries a centred crop box vae-decode cuts it back "
          "to", .def_int = 1440},
  {.key = "scale", .type = ConfigType::Real, .required = false,
   .doc = "the target as a factor of the input's own pixel size, in place "
          "of width / height. 0 (default) uses width / height",
   .def_real = 0.0},
  {.key = "align", .type = ConfigType::Int, .required = false,
   .doc = "pixel alignment of the target; 32, the node's, keeps both latent "
          "sides even for the DiT's 2x2 patch", .def_int = 32},
  {.key = "two_pass_above", .type = ConfigType::Int, .required = false,
   .doc = "a target TALLER than this many rows is reached in two passes, "
          "this height first, each within the range the network was "
          "trained on -- the release's 4K. 0 never splits",
   .def_int = 1440},
  {.key = "normalization", .type = ConfigType::String, .required = false,
   .doc = "'release' (default): the network sees the whitened latent "
          "normalized AGAIN by H3's statistics, as Open-H3-Regenerate-2K "
          "calls it (its node expects ComfyUI's raw latents) -- what its "
          "results were made with. 'once': the whitened latent as the "
          "network's input, the node's own intent for a raw one",
   .def_str = "release", .extra = kNormChoices},
  {.key = "frames", .type = ConfigType::Int, .required = false,
   .doc = "PLANNING ONLY: the clip length the memory plan books this "
          "stage's activations at, before a latent exists. The latent "
          "that arrives says the truth", .def_int = 124},
  {.key = "i8_gemm", .type = ConfigType::Bool, .required = false,
   .doc = "accelerated mode (LOSSY): the convs' GEMMs on dynamic int8, M5 "
          "matrix cores only (elsewhere this is ignored). Off by default "
          "-- the upscaler is a few percent of a refine -- and settable so "
          "the trade can be measured. Env VPIPE_I8_GEMM overrides",
   .def_bool = false},
  {.key = "i8_gemm_tier", .type = ConfigType::String, .required = false,
   .doc = "which int8 i8_gemm runs, in generate-image's words: "
          "'accurate' (default) quantizes each conv's weight per call; "
          "'fast', 'token' and 'token_plain' hold the 3x3x3 convs' weights "
          "in the int8 layout (half their bytes)",
   .def_str = "accurate", .extra = kI8TierChoices},
  {.key = "unload_when_idle", .type = ConfigType::String, .required = false,
   .doc = "'auto' (default) and 'destroy' let the network go after each "
          "latent -- it is small to reload and the DiT that follows wants "
          "the room; 'keep' holds it for the run",
   .def_str = "auto", .extra = kUnloadChoices},
};

const PortSpec kIports[] = {
  {.name = "latent",
   .doc = "f32 [24, T, h, w] whitened MiniMax-H3 video latent (vae-encode, "
          "generate-video)",
   .type = &typeid(TensorBeatPayload), .tags = "latent", .clock_group = 0},
  {.name = "model",
   .doc = "OPTIONAL shared model reference from a model-select source; "
          "overrides hf_dir",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "latent",
   .doc = "f32 [24, T, H, W] whitened, the input's sideband carried, plus a "
          "`crop_box` when the asked size is not whole cells",
   .type = &typeid(TensorBeatPayload), .tags = "latent", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "latent-upscale",
  .doc       = "Enlarges a MiniMax-H3 video latent without decoding it, with "
               "the H3 latent upscaler (a 3D-conv network) -- the step "
               "Open-H3-Regenerate-2K runs between encoding a draft and "
               "refining it at the larger size (768P -> 2K).",
  .display_name = "Latent Upscale",
  .category  = StageCategory::Generative,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

// Latent frames a clip of `frames` pixel frames encodes to (H3's 17-frame
// chunks of 5, less the 3 dropped).
int
h3_latent_frames_(int frames)
{
  if (frames <= 1) { return 1; }
  const int n = (frames + 16) / 17;
  return std::max(1, n * 5 - 3);
}

}  // namespace

LatentUpscaleStage::LatentUpscaleStage(const SessionContextIntf* s,
                                       std::string id,
                                       std::vector<InEdge> ip, FlexData cfg)
  : TypedStage<LatentUpscaleStage>(s, std::move(id), std::move(ip),
                                   std::move(cfg))
{
  allocate_oports(spec().oports.size());
  _hf_dir = attr_str("hf_dir");
  _width = (int)attr_int("width");
  _height = (int)attr_int("height");
  _scale = attr_real("scale");
  _align = (int)attr_int("align");
  _two_pass_above = (int)attr_int("two_pass_above");
  _plan_frames = (int)attr_int("frames");
  const std::string nm = attr_str("normalization");
  _norm_once = nm == "once";
  _unload = attr_str("unload_when_idle");
  {
    const std::string t = attr_str("i8_gemm_tier");
    _i8_tier = !attr_bool("i8_gemm") ? -1
               : t == "token_plain"  ? 3
               : t == "token"        ? 2
               : t == "fast"         ? 1
                                     : 0;
    if (!t.empty() && t != "accurate" && t != "fast" && t != "token" &&
        t != "token_plain") {
      fail_config(fmt("LatentUpscaleStage('{}'): i8_gemm_tier must be "
                      "accurate, fast, token or token_plain (got '{}')",
                      this->id(), t));
    }
  }
  if (nm != "release" && nm != "once" && !nm.empty()) {
    fail_config(fmt("LatentUpscaleStage('{}'): normalization must be "
                    "release or once (got '{}')", this->id(), nm));
  }
  if (_unload != "auto" && _unload != "destroy" && _unload != "keep") {
    fail_config(fmt("LatentUpscaleStage('{}'): unload_when_idle must be "
                    "auto, destroy or keep (got '{}')", this->id(),
                    _unload));
  }
  if (_scale < 0.0 || (_scale == 0.0 && (_width <= 0 || _height <= 0)) ||
      _align <= 0 || _two_pass_above < 0) {
    fail_config(fmt("LatentUpscaleStage('{}'): give width and height (> 0) "
                    "or a scale > 0, and align > 0 (got {}x{}, scale {}, "
                    "align {})", this->id(), _width, _height, _scale,
                    _align));
  }
}

LatentUpscaleStage::~LatentUpscaleStage() = default;

const StageSpec&
LatentUpscaleStage::spec() const noexcept
{
  return kSpec;
}

void
LatentUpscaleStage::apply_constant(unsigned iport, const FlexData& beat)
{
  if (iport != kModelPort) { return; }
  apply_model_select_beat(beat, _hf_dir);
}

void
LatentUpscaleStage::reset_run_state()
{
  _model_latched = false;
  _emitted = 0;
  _scratch_stated = 0;
}

std::string
LatentUpscaleStage::weights_file_(std::string* err) const
{
  if (_hf_dir.empty()) {
    if (err != nullptr) { *err = "no upscaler (hf_dir or the model iport)"; }
    return {};
  }
  const ResolvedModel rm = resolve_model(session(), _hf_dir);
  // A record that pins the upscaler's file says which one it is.
  for (const std::string& f : rm.files) {
    if (f.find("latent_upsampler/") == 0 &&
        fs::path(f).extension() == ".safetensors") {
      return (fs::path(rm.dir) / f).string();
    }
  }
  return Upscaler::resolve_weights(rm.dir, err);
}

bool
LatentUpscaleStage::target_(int h, int w, int* H, int* W, double* scale,
                            int* want_w, int* want_h) const
{
  Upscaler::Target t;
  if (_scale > 0.0) {
    t = Upscaler::target_for_scale(h, w, _scale, _align);
    *want_w = t.w * 16;
    *want_h = t.h * 16;
  } else {
    t = Upscaler::target_for(h, w, _width, _height, _align);
    *want_w = _width;
    *want_h = _height;
  }
  *H = t.h;
  *W = t.w;
  *scale = t.scale;
  return t.h >= h && t.w >= w;
}

std::vector<ResourceClaim>
LatentUpscaleStage::declare_resources() const
{
  std::vector<ResourceClaim> out;
  const std::string file = weights_file_();
  if (file.empty()) { return out; }
  // It runs before the denoise and lets go after its latent (unless
  // kept), so it is the conditioning phase's.
  const bool keep = _unload == "keep";
  // At what the load holds: the file as stored, or its 3x3x3 convs in
  // the int8 layout under a tier that holds them.
  const int tier = effective_i8_tier_();
  const std::size_t held = Upscaler::weight_bytes_for(file, tier);
  for (auto& c : keep ? model_memory::weight_claims({file})
                      : model_memory::weight_claims_in_phase(
                            {file}, model_memory::kPhaseCondition)) {
    if (held > 0) {
      bag::set_integer(c.extra, model_memory::kClaimHeldBytes,
                       (std::int64_t)held);
    }
    out.push_back(std::move(c));
  }
  for (auto& c : model_memory::scratch_claims(
           scratch_label_(), planned_scratch_(),
           model_memory::kPhaseCondition)) {
    out.push_back(std::move(c));
  }
  return out;
}

std::string
LatentUpscaleStage::scratch_label_() const
{
  return fmt("latent-upscale:{}", this->id())();
}

// The output the config names, from a draft at half its size -- past an
// enlargement of ~1.22x the output stack's three planes are the high
// water whatever the input, and a two-pass 4K's second pass is one too.
// At `scale`, a 768P draft (H3's 1344 x 768) enlarged by it. The latent
// that arrives revises it (revise_scratch_).
std::size_t
LatentUpscaleStage::planned_scratch_() const
{
  const int T = h3_latent_frames_(_plan_frames);
  const int tier = effective_i8_tier_();
  if (_scale > 0.0) {
    constexpr int kDraftH = 48, kDraftW = 84;
    const auto t =
        Upscaler::target_for_scale(kDraftH, kDraftW, _scale, _align);
    return Upscaler::scratch_bytes(T, kDraftH, kDraftW, std::max(1, t.h),
                                   std::max(1, t.w), tier);
  }
  const auto t = Upscaler::target_for(1, 1, _width, _height, _align);
  return Upscaler::scratch_bytes(T, std::max(1, t.h / 2),
                                 std::max(1, t.w / 2), t.h, t.w, tier);
}

void
LatentUpscaleStage::revise_scratch_(std::size_t bytes)
{
  if (bytes == 0 || bytes == _scratch_stated || session() == nullptr ||
      session()->services() == nullptr) {
    return;
  }
  _scratch_stated = bytes;
  if (auto* mgr = session()->services()->generative_model_manager()) {
    mgr->revise_scratch(scratch_label_(), bytes);
  }
  StageMemory m = declare_memory();
  m.scratch = bytes;
  revise_memory(m);
}

StageMemory
LatentUpscaleStage::declare_memory() const
{
  StageMemory m;
  const std::string file = weights_file_();
  if (file.empty()) { return m; }
  const std::size_t held =
      Upscaler::weight_bytes_for(file, effective_i8_tier_());
  m.hold(file, held > 0 ? held : model_memory::dir_weights_bytes(file));
  m.scratch = planned_scratch_();
  return m;
}

int
LatentUpscaleStage::effective_i8_tier_() const
{
  auto* mc = session() != nullptr && session()->services() != nullptr
                 ? session()->services()->metal_compute()
                 : nullptr;
  return Upscaler::effective_i8_tier(
      _i8_tier, mc != nullptr && mc->supports_matrix_cores());
}

bool
LatentUpscaleStage::ensure_loaded_()
{
  if (_model) { return true; }
  std::string err;
  const std::string file = weights_file_(&err);
  if (file.empty()) {
    session()->error(fmt("LatentUpscaleStage('{}'): {}", this->id(), err));
    return false;
  }
  auto* mc = session()->services()->metal_compute();
  if (mc == nullptr) {
    session()->error(fmt("LatentUpscaleStage('{}'): no metal-compute "
                         "backend", this->id()));
    return false;
  }
  const auto t0 = std::chrono::steady_clock::now();
  _ws = genai::open_weight_set(file, session());
  if (_ws) { _model = Upscaler::load(_ws, mc, &err, _i8_tier); }
  if (!_model) {
    _ws.reset();
    session()->error(fmt("LatentUpscaleStage('{}'): '{}': {}", this->id(),
                         file, err.empty() ? "cannot open" : err));
    return false;
  }
  _loaded_file = file;
  const auto& c = _model->config();
  session()->info(fmt(
      "LatentUpscaleStage('{}'): MiniMax-H3 latent upscaler loaded in "
      "{:.2f} s -- {} channels, {} + {} stack entries, {} MB held, GEMMs on "
      "{}{}", this->id(),
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count(),
      c.channels, c.in_entries.size(), c.out_entries.size(),
      _model->weight_bytes() >> 20,
      _model->matrix_core_gemm() ? "matmul2d" : "steel",
      _model->i8_tier() < 0 ? std::string()
      : _model->i8_tier() == 0 ? std::string(" + int8 (accurate)")
      : _model->i8_tier() == 1 ? std::string(" + int8 (fast)")
      : _model->i8_tier() == 2 ? std::string(" + int8 (token)")
                               : std::string(" + int8 (token_plain)")));
  return true;
}

void
LatentUpscaleStage::release_()
{
  if (!_model) { return; }
  _model.reset();
  _ws.reset();
  if (auto* mgr = session()->services()->generative_model_manager()) {
    mgr->drop_weights(_loaded_file);
    mgr->note_phase_released(_loaded_file);
  }
}

Job
LatentUpscaleStage::initialize(RuntimeContext& ctx)
{
  (void)ctx;
  co_return;
}

Job
LatentUpscaleStage::process(RuntimeContext& ctx)
{
  if (!_model_latched && ctx.num_iports() > kModelPort &&
      ctx.iport_connected(kModelPort)) {
    auto mb = co_await ctx.read(kModelPort);
    _model_latched = true;
    if (const auto* mfd =
            mb ? dynamic_cast<const FlexDataPayload*>(mb.get()) : nullptr) {
      apply_model_select_beat(mfd->data, _hf_dir);
    }
  }
  auto in = co_await ctx.read(0);
  if (!in) {
    release_();
    ctx.signal_done();
    co_return;
  }
  const auto* tb = dynamic_cast<const TensorBeatPayload*>(in.get());
  if (tb == nullptr || tb->dtype != TensorBeat::DType::F32 ||
      tb->shape.size() != 4 || tb->shape[0] != 24) {
    session()->warn(fmt(
        "LatentUpscaleStage('{}'): expected an f32 [24, T, h, w] MiniMax-H3 "
        "latent, got {}; skipping", this->id(),
        in ? in->describe() : std::string("nothing")));
    co_return;
  }
  const int T = (int)tb->shape[1], h = (int)tb->shape[2];
  const int w = (int)tb->shape[3];
  int H = 0, W = 0, want_w = 0, want_h = 0;
  double scale = 1.0;
  if (!target_(h, w, &H, &W, &scale, &want_w, &want_h)) {
    session()->warn(fmt(
        "LatentUpscaleStage('{}'): the target {}x{} px is smaller than the "
        "latent's {}x{}; the network only enlarges. Skipping", this->id(),
        want_w, want_h, w * 16, h * 16));
    co_return;
  }

  auto out = std::make_unique<TensorBeatPayload>();
  out->dtype = TensorBeat::DType::F32;
  FlexData sb = tb->sideband.is_object() ? tb->sideband
                                         : FlexData::make_object();
  if (H == h && W == w) {
    // The node's rule too: nothing to enlarge, the latent goes through.
    out->shape = tb->shape;
    out->resize_contiguous(tb->element_count());
    std::memcpy(out->as_f32(), tb->as_f32(),
                tb->element_count() * sizeof(float));
  } else {
    if (!ensure_loaded_()) { co_return; }
    // Above `two_pass_above` rows, to that height first.
    std::vector<float> cur(tb->as_f32(),
                           tb->as_f32() + (std::size_t)24 * T * h * w);
    int ch = h, cw = w;
    std::vector<std::pair<int, int>> passes;
    std::vector<double> scales;
    if (_two_pass_above > 0 && H * 16 > _two_pass_above &&
        h * 16 < _two_pass_above) {
      const int mid_w = (int)std::lround((double)want_w * _two_pass_above /
                                         (double)want_h);
      const auto t = Upscaler::target_for(h, w, mid_w, _two_pass_above,
                                          _align);
      if (t.h > h && t.h < H) {
        passes.push_back({t.h, t.w});
        scales.push_back(t.scale);
      }
    }
    if (!passes.empty()) {
      const auto t2 = _scale > 0.0
          ? Upscaler::target_for_scale(passes.back().first,
                                       passes.back().second,
                                       _scale * h / passes.back().first,
                                       _align)
          : Upscaler::target_for(passes.back().first, passes.back().second,
                                 want_w, want_h, _align);
      passes.push_back({H, W});
      scales.push_back(t2.scale);
    } else {
      passes.push_back({H, W});
      scales.push_back(scale);
    }
    // What these passes allocate, for the ledger, before they do.
    {
      std::size_t need = 0;
      int ph = h, pw = w;
      for (const auto& [nH, nW] : passes) {
        need = std::max(need, Upscaler::scratch_bytes(
                                  _model->config(), T, ph, pw, nH, nW,
                                  _model->i8_tier()));
        ph = nH;
        pw = nW;
      }
      revise_scratch_(need);
    }
    UiProgress bar = session()->open_progress("latent upscale");
    const auto t0 = std::chrono::steady_clock::now();
    std::uint64_t flops = 0;
    for (std::size_t pi = 0; pi < passes.size(); ++pi) {
      std::vector<float> next;
      std::string err;
      const int nH = passes[pi].first, nW = passes[pi].second;
      const std::size_t npass = passes.size();
      const bool ok = _model->upscale(
          cur.data(), T, ch, cw, nH, nW, scales[pi],
          _norm_once ? Upscaler::Normalize::kOnce
                     : Upscaler::Normalize::kRelease,
          &next, &err, [&](int done, int total) {
            bar.update((std::uint64_t)(pi * total + done),
                       (std::uint64_t)(npass * total));
          });
      if (!ok) {
        bar.finish();
        session()->warn(fmt("LatentUpscaleStage('{}'): {}; skipping",
                            this->id(), err));
        if (_unload != "keep") { release_(); }
        co_return;
      }
      flops += _model->last_flops();
      session()->info(fmt(
          "LatentUpscaleStage('{}'): pass {} of {}: [24, {}, {}, {}] -> "
          "[24, {}, {}, {}] at scale {:.4f}", this->id(), pi + 1, npass, T,
          ch, cw, T, nH, nW, scales[pi]));
      cur = std::move(next);
      ch = nH;
      cw = nW;
    }
    bar.finish();
    const double s = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t0).count();
    session()->info(fmt(
        "LatentUpscaleStage('{}'): {}x{} px -> {}x{} px in {:.1f} s ({:.1f} "
        "TFLOP, {:.2f} TFLOP/s)", this->id(), w * 16, h * 16, W * 16,
        H * 16, s, (double)flops * 1e-12,
        s > 0.0 ? (double)flops * 1e-12 / s : 0.0));
    out->shape = {24, T, H, W};
    out->resize_contiguous(cur.size());
    std::memcpy(out->as_f32(), cur.data(), cur.size() * sizeof(float));
    if (_unload != "keep") { release_(); }
  }
  // The pixels asked for, centred in the computed ones: vae-decode cuts
  // the picture back to them.
  const int cw_px = W * 16, ch_px = H * 16;
  if (want_w > 0 && want_h > 0 && want_w <= cw_px && want_h <= ch_px &&
      (want_w != cw_px || want_h != ch_px)) {
    const int x0 = (cw_px - want_w) / 2, y0 = (ch_px - want_h) / 2;
    FlexData box = FlexData::make_array();
    for (int v : {x0, y0, x0 + want_w, y0 + want_h}) {
      box.as_array().push_back(FlexData::make_int((std::int64_t)v));
    }
    sb.as_object().insert_or_assign(sideband::kCropBox, std::move(box));
  }
  out->sideband = std::move(sb);
  ++_emitted;
  co_await ctx.write(0, std::move(out));
}

VPIPE_REGISTER_STAGE(LatentUpscaleStage)
VPIPE_REGISTER_SPEC(LatentUpscaleStage, kSpec)

}  // namespace vpipe

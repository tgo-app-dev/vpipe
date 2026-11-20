#include "stages/audio-video/relayvsr-upscale-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/host-report.h"
#include "common/flex-bag.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/relayvsr/metal-relayvsr-decoder.h"
#include "generative-models/relayvsr/metal-relayvsr-wan.h"
#include "generative-models/relayvsr/relayvsr-ops.h"
#include "generative-models/relayvsr/relayvsr-stream.h"
#include "generative-models/weight-set.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "interfaces/ui-delegate-intf.h"
#include "stages/model-memory-host.h"
#include "stages/model-memory.h"
#include "stages/model-registry.h"

#include <sys/sysctl.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace vpipe {

namespace {

// Booked until the first frame says, where no output size is given
// (output_width / output_height): a 3840 x 2176 OUTPUT -- a 960 x 544
// source at 4x, 1920 x 1088 at 2x.
constexpr int kPlanOutH = 2176, kPlanOutW = 3840;

constexpr SpecExtra kDtypeChoices[] = {
  {spec_key::kChoices, "auto,u8,f32"},
};
constexpr SpecExtra kI8TierChoices[] = {
  {spec_key::kChoices, "accurate,fast,token,token_plain"},
};
constexpr ConfigKey kAttrs[] = {
  {.key = "hf_dir", .type = ConfigType::String, .required = true,
   .doc = "the RelayVSR model: a models-DB key -- kopper/RelayVSR (the "
          "release, 4x), Comfy-Org/RelayVSR-x2 or Comfy-Org/RelayVSR-x4 "
          "(ComfyUI's int8 repack) -- or a path: a directory holding a "
          "keyframe DiT and its per-frame decoder (wan.safetensors + "
          "flash.safetensors as kopper/RelayVSR publishes them, "
          "wan_x2 / flash_x2 as lightx2v/RelayVSR_X2 does, or ComfyUI's "
          "diffusion_models/ + vae/), or either checkpoint FILE. Each file "
          "states its scale; the output is that scale times the input",
   .suggest_db = kModelRegistryDb, .suggest_db_type = "relayvsr"},
  {.key = "scale", .type = ConfigType::Int, .required = false,
   .doc = "the native scale, 2 or 4, where `hf_dir` holds both "
          "(Comfy-Org/RelayVSR's root); 0 (default) takes the one the "
          "model names. A 2x model is not a 4x one run smaller: it is its "
          "own pair of checkpoints (lightx2v/RelayVSR_X2), retrained for "
          "2x -- its keyframe DiT runs a quarter of the 4x one's tokens "
          "for the same input",
   .def_int = 0},
  {.key = "window", .type = ConfigType::Int, .required = false,
   .doc = "the release's --window-size: a keyframe every `window - 1` "
          "frames, each window of frames decoded between the two around "
          "it. 16 (default) is the release's. Larger is faster (fewer "
          "runs of the 5B keyframe model) and leans more on the small "
          "decoder; a frame leaves the stage up to `window - 1` frames "
          "after it arrived. 2 .. 64",
   .def_int = 16},
  {.key = "output_width", .type = ConfigType::Int, .required = false,
   .doc = "optional: the width of the frames this run will make (4x or 2x "
          "the input's, the model's scale), where it is known before the "
          "first frame. Only the resource plan reads it -- the working set "
          "(the keyframes' ring of keys and values, what grows with the "
          "frame) is booked at the input it implies -- so a small input "
          "is planned small, and a run that cannot fit the GPU's working "
          "set even alone is refused before its weights are read. The "
          "stage still makes exactly the scale times what it is given, "
          "and the run books its own frame at the first one. 0 (default) "
          "books 3840 x 2176. Give both or neither; a multiple of the "
          "scale, 32 .. 32768",
   .def_int = 0},
  {.key = "output_height", .type = ConfigType::Int, .required = false,
   .doc = "optional: the height of the output frames (see output_width); "
          "a multiple of the scale, 32 .. 32768",
   .def_int = 0},
  {.key = "seed", .type = ConfigType::Int, .required = false,
   .doc = "the keyframe noise: torch's CPU normal draw from one generator "
          "seeded with this at the start of every run, as the release "
          "draws it (it uses CUDA's generator; this is the CPU one). 42 "
          "(default) is the release's",
   .def_int = 42},
  {.key = "output_dtype", .type = ConfigType::String, .required = false,
   .doc = "\"auto\" (default): the input's -- u8 frames out for u8 in, "
          "f32 for f32 or f16. \"u8\": as the release writes its frames, "
          "(v + 1) * 127.5 clamped and truncated. \"f32\": planar f32 in "
          "[0, 1], unclamped -- what an f32 chain (avf-save-video, a "
          "further resample) keeps",
   .def_str = "auto", .extra = kDtypeChoices},
  {.key = "i8_gemm", .type = ConfigType::Bool, .required = false,
   .doc = "accelerated mode (LOSSY): both models' block projections on the "
          "int8 GEMM tiers -- the keyframe DiT's eight a block (q, k, v, "
          "out, the cross-attention's q and out, the feed-forward's two) "
          "and the decoder's (q, k, v, out, gate|up, down) -- while the "
          "frame encoder, the patch embedding and the heads stay "
          "f16 / bf16. Matrix cores only (M5 and later): elsewhere it does "
          "nothing. MEASURED on an M5 Max, 480x270 in: a keyframe 431 -> "
          "285 ms at 'token'. The decoder's windowed attention, the most "
          "of a frame there, is not a GEMM, so a window's frames move "
          "less (0.063 -> 0.051 s a frame). Env VPIPE_I8_GEMM overrides",
   .def_bool = false},
  {.key = "i8_gemm_tier", .type = ConfigType::String, .required = false,
   .doc = "which int8 i8_gemm runs, in generate-image's words: 'accurate' "
          "(default) quantizes each bf16 / f16 weight per call at groups "
          "of 64 (projections at least 1024 deep: every keyframe one, the "
          "decoder's down alone); 'fast' and 'token' HOLD the projections "
          "int8 -- the two models' 8.4 GB of weights in 4.3 -- at a "
          "rotated group or per channel; 'token_plain' the same unrotated. "
          "MEASURED against the fp32 reference (its own bf16 the bar), the "
          "worst keyframe / frame: accurate 1.31x / 1.03x the bar, fast "
          "1.36x / 1.16x, token 1.51x / 1.21x, token_plain 3.91x / 3.10x",
   .def_str = "accurate", .extra = kI8TierChoices},
  {.key = "ane_ffn", .type = ConfigType::Bool, .required = false,
   .doc = "split both models' feed-forward by rows between the GPU and the "
          "Apple Neural Engine, concurrently (fp16 on the ANE): one module "
          "a model, claimed from the resource plan, staged per block under "
          "its attention. The tier MEASURES both and keeps the GPU alone "
          "where that is faster. For an M4-class Mac: MEASURED on an M4 "
          "Pro at 480x270 in, a keyframe 3.35 -> 2.51 s (the ANE runs its "
          "half of the 5B model's feed-forward as fast as the GPU its "
          "own); the decoder's small feed-forward stays on the GPU there, "
          "and on an M5 Max both do. A block whose feed-forward an int8 "
          "tier holds ('fast' and up) keeps the GPU",
   .def_bool = false},
  {.key = "ane_rows", .type = ConfigType::Real, .required = false,
   .doc = "the ANE's share of each feed-forward's rows, rounded down to "
          "whole chunks; 0 (default) balances from the two engines' "
          "measured times",
   .def_real = 0.0},
  {.key = "sage_attn", .type = ConfigType::Bool, .required = false,
   .doc = "accelerated mode (LOSSY): SageAttention's int8 QK on the matrix "
          "cores -- both models' windowed self-attention (Q and K per "
          "token, K smoothed by one mean over every frame it attends) and "
          "the keyframe DiT's cross-attention onto its context (from 4096 "
          "query rows, where it pays). Matrix cores only (M5 and later): "
          "elsewhere it does nothing. MEASURED on an M5 Max, 960x540 in, "
          "with sage_pv8: a keyframe's windowed attention 277 -> 162 ms "
          "and its cross-attention 52 -> 39 (the keyframe 6% faster); a "
          "frame's attention 2-4% (the decoder's int8 prologue costs about "
          "what its kernel saves)",
   .def_bool = false},
  {.key = "sage_pv8", .type = ConfigType::Bool, .required = false,
   .doc = "with sage_attn: the P*V in int8 too (P against each row's "
          "running max, V per channel over every frame attended, the sum "
          "held in int32). MEASURED against the fp32 reference (its own "
          "bf16 the bar), the worst keyframe / frame: sage_attn alone "
          "1.19x / 1.10x the bar, with sage_pv8 1.73x / 1.31x",
   .def_bool = false},
  {.key = "autotune", .type = ConfigType::Bool,
   .doc = "whether the int8 accurate tier's split-K tuner MEASURES: false "
          "takes the choice believed fastest on this GPU, timing nothing "
          "and reading no kept answer, so a run repeats (accel-keys.h "
          "kAutotune). Default true",
   .def_bool = true},
};

const PortSpec kIports[] = {
  {.name = "frames",
   .doc = "planar RGB TensorBeat [3, H, W] per beat -- u8, or f32 / f16 in "
          "[0, 1], video-to-rgb's two -- every frame of a run one size",
   .type = &typeid(TensorBeatPayload),
   .tags = "rgb-frames", .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "frames",
   .doc = "the same frames at the model's scale (4x or 2x), planar "
          "[3, sH, sW], one per frame in "
          "and in order, each carrying its input's sideband. On a clock "
          "of its own: a frame leaves once its window is complete, so "
          "the beats come out in bursts behind the ones going in",
   .type = &typeid(TensorBeatPayload),
   .tags = "rgb-frames", .clock_group = 1},
};
const StageSpec kSpec = {
  .type_name = "relayvsr-upscale",
  .doc       = "4x or 2x video super-resolution by RelayVSR: a one-step "
               "5B DiT writes a keyframe every few frames and a small "
               "decoder turns every frame into its upscaled frame between "
               "the two around it. Frames in, upscaled frames out, in "
               "order.",
  .display_name = "RelayVSR Upscale",
  .category  = StageCategory::Visual,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

struct RelayVsrUpscaleStage::Configs {
  std::string wan_file, flash_file;
  genai::MetalRelayVsrWan::Config wan;
  genai::MetalRelayVsrDecoder::Config dec;
};

struct RelayVsrUpscaleStage::Model {
  std::string wan_file, flash_file;
  genai::MetalRelayVsrWan::Config wan_cfg;
  genai::MetalRelayVsrDecoder::Config dec_cfg;
  std::shared_ptr<genai::WeightSet> ws_wan, ws_flash;
  std::unique_ptr<genai::MetalRelayVsrWan> wan;
  std::unique_ptr<genai::MetalRelayVsrDecoder> dec;
  std::unique_ptr<genai::RelayVsrStream> stream;
  UiProgress bar;
  double t_start = 0.0;
};

RelayVsrUpscaleStage::RelayVsrUpscaleStage(const SessionContextIntf* s,
                                           std::string id,
                                           std::vector<InEdge> ip,
                                           FlexData cfg)
  : TypedStage<RelayVsrUpscaleStage>(s, std::move(id), std::move(ip),
                                     std::move(cfg))
{
  allocate_oports(spec().oports.size());
  _hf_dir = attr_str("hf_dir");
  _window = (int)attr_int("window");
  _seed = (std::uint64_t)attr_int("seed");
  _out_dtype = attr_str("output_dtype");
  const int out_w = (int)attr_int("output_width");
  const int out_h = (int)attr_int("output_height");
  _scale = (int)attr_int("scale");
  if (_hf_dir.empty()) {
    fail_config(fmt("RelayVsrUpscaleStage('{}'): `hf_dir` must name the "
                    "RelayVSR model (kopper/RelayVSR)", this->id()));
    return;
  }
  if (_window < 2 || _window > 64) {
    fail_config(fmt("RelayVsrUpscaleStage('{}'): window {} is outside "
                    "[2, 64]", this->id(), _window));
    return;
  }
  if (_scale != 0 && _scale != 2 && _scale != 4) {
    fail_config(fmt("RelayVsrUpscaleStage('{}'): scale {} is not 0, 2 or "
                    "4", this->id(), _scale));
    return;
  }
  // The stage makes exactly its scale times the input, so no other size
  // is one it can be asked for. The scale is the model's -- read at the
  // plan (plan_in_) -- so here a side need only be even; a 4x model
  // refuses one not a multiple of 4 there.
  const int mult = _scale != 0 ? _scale : 2;
  auto bad_side = [mult](int v) {
    return v < 32 || v > 32768 || v % mult != 0;
  };
  if ((out_w == 0) != (out_h == 0) ||
      (out_w != 0 && (bad_side(out_w) || bad_side(out_h)))) {
    fail_config(fmt("RelayVsrUpscaleStage('{}'): output_width x "
                    "output_height {} x {} must be both 0, or both "
                    "multiples of {} in 32 .. 32768 (the frames are the "
                    "scale times the input's)", this->id(), out_w, out_h,
                    mult));
    return;
  }
  _plan_given = out_w != 0;
  _out_w = out_w;
  _out_h = out_h;
  if (_out_dtype != "auto" && _out_dtype != "u8" && _out_dtype != "f32") {
    fail_config(fmt("RelayVsrUpscaleStage('{}'): output_dtype '{}' is not "
                    "one of auto|u8|f32", this->id(), _out_dtype));
    return;
  }
  if (attr_bool("i8_gemm")) {
    const std::string t = attr_str("i8_gemm_tier");
    _i8_tier = t == "accurate" || t.empty() ? 0
               : t == "fast"               ? 1
               : t == "token"              ? 2
               : t == "token_plain"        ? 3
                                           : -2;
    if (_i8_tier == -2) {
      fail_config(fmt("RelayVsrUpscaleStage('{}'): i8_gemm_tier must be "
                      "'accurate', 'fast', 'token' or 'token_plain' (got "
                      "'{}')", this->id(), t));
      return;
    }
  }
  _ane_ffn = attr_bool("ane_ffn");
  _ane_rows = (float)std::clamp(attr_real("ane_rows"), 0.0, 1.0);
  _sage_attn = attr_bool("sage_attn");
  _sage_pv8 = _sage_attn && attr_bool("sage_pv8");
  _autotune = attr_bool("autotune");
}

RelayVsrUpscaleStage::~RelayVsrUpscaleStage() = default;

const StageSpec&
RelayVsrUpscaleStage::spec() const noexcept
{
  return kSpec;
}

bool
RelayVsrUpscaleStage::files_(std::string* wan, std::string* flash,
                             std::string* err) const
{
  const ResolvedModel rm = resolve_model(session(), _hf_dir);
  genai::RelayVsrFiles f;
  if (!genai::resolve_relayvsr_files(rm.dir.empty() ? _hf_dir : rm.dir, &f,
                                     err, _scale)) {
    return false;
  }
  *wan = f.wan;
  *flash = f.flash;
  return true;
}

bool
RelayVsrUpscaleStage::configs_(Configs* out, std::string* err) const
{
  std::string e;
  if (!files_(&out->wan_file, &out->flash_file, &e) ||
      !genai::MetalRelayVsrWan::config_from_checkpoint(out->wan_file,
                                                       &out->wan, &e) ||
      !genai::MetalRelayVsrDecoder::config_from_checkpoint(
          out->flash_file, &out->dec, &e)) {
    if (err != nullptr) { *err = e; }
    return false;
  }
  auto* mc = session() != nullptr && session()->services() != nullptr
                 ? session()->services()->metal_compute()
                 : nullptr;
  const int tier = genai::relayvsr::effective_i8_tier(
      _i8_tier, mc != nullptr && mc->supports_matrix_cores());
  out->wan.i8_tier = out->dec.i8_tier = tier;
  out->wan.autotune = out->dec.autotune = _autotune;
  out->wan.ane_ffn = out->dec.ane_ffn = _ane_ffn;
  out->wan.ane_rows = out->dec.ane_rows = _ane_rows;
  // SageAttention lives in the matrix cores' attention kernels alone.
  const bool mx = mc != nullptr && mc->supports_matrix_cores();
  out->wan.sage_attn = out->dec.sage_attn = _sage_attn && mx;
  out->wan.sage_pv8 = out->dec.sage_pv8 = _sage_pv8 && mx;
  return true;
}

std::string
RelayVsrUpscaleStage::ane_label_(const char* model) const
{
  return fmt("ane-ffn/{}/{}", this->id(), model)();
}

std::string
RelayVsrUpscaleStage::scratch_label_() const
{
  return fmt("relayvsr-upscale:{}", this->id())();
}

std::uint64_t
RelayVsrUpscaleStage::run_bytes_(int h, int w) const
{
  Configs c;
  if (_m) {
    c.wan = _m->wan_cfg;
    c.dec = _m->dec_cfg;
  } else if (!configs_(&c)) {
    return 0;
  }
  return genai::RelayVsrStream::run_bytes(c.wan, c.dec, h, w, _window);
}

bool
RelayVsrUpscaleStage::plan_in_(const Configs& c, int* h, int* w,
                               std::string* why) const
{
  const int s = c.wan.upscale;
  const int ow = _plan_given ? _out_w : kPlanOutW;
  const int oh = _plan_given ? _out_h : kPlanOutH;
  if (ow % s != 0 || oh % s != 0) {
    if (why != nullptr) {
      *why = fmt("RelayVsrUpscaleStage('{}'): output_width x output_height "
                 "{} x {} is not a {}x frame (the model's scale): give "
                 "multiples of {}", this->id(), ow, oh, s, s)();
    }
    *h = oh / s;
    *w = ow / s;
    return false;
  }
  *h = oh / s;
  *w = ow / s;
  return true;
}

std::vector<ResourceClaim>
RelayVsrUpscaleStage::declare_resources() const
{
  std::vector<ResourceClaim> out;
  Configs cf;
  if (!configs_(&cf)) { return out; }
  const auto& wc = cf.wan;
  const auto& dc = cf.dec;
  // Held as loaded: the keyframe DiT less the cross-attention K / V
  // projections its load folds away, plus what it folds; the decoder
  // in f16 (its file is F32); the block projections of both at the int8
  // tier's layout where one holds it.
  const std::pair<std::string, std::uint64_t> held[] = {
      {cf.wan_file, genai::MetalRelayVsrWan::weight_bytes(wc)},
      {cf.flash_file, genai::MetalRelayVsrDecoder::weight_bytes(dc)}};
  for (const auto& [file, bytes] : held) {
    for (auto& c : model_memory::weight_claims({file})) {
      bag::set_integer(c.extra, model_memory::kClaimHeldBytes,
                       (std::int64_t)bytes);
      out.push_back(std::move(c));
    }
  }
  // At a launch the frame size is the output size given (output_width x
  // output_height) over the model's scale, else not known: booked at a
  // 3840 x 2176 output. Revised at the first frame either way. Never at
  // the last run's size -- the plan is collected before reset_run_state(),
  // so a member set by that run would still be here, and the same graph
  // would plan differently depending on what ran before it.
  int ph = 0, pw = 0;
  (void)plan_in_(cf, &ph, &pw);
  for (auto& c : model_memory::scratch_claims(
           scratch_label_(),
           genai::RelayVsrStream::run_bytes(wc, dc, ph, pw, _window),
           {})) {
    out.push_back(std::move(c));
  }
  // The ANE modules, one a model, in UNITS: CoreML holds their bytes where
  // no other ledger here can see them. Granted at load. None for a model
  // whose feed-forward the int8 tier holds: no module is built there.
  if (_ane_ffn && genai::MetalRelayVsrWan::ane_stages(wc)) {
    for (auto& c : model_memory::coreml_claims(
             ane_label_("keyframe"),
             genai::MetalRelayVsrWan::ane_runtime_bytes(wc), 1)) {
      out.push_back(std::move(c));
    }
  }
  if (_ane_ffn && genai::MetalRelayVsrDecoder::ane_stages(dc)) {
    for (auto& c : model_memory::coreml_claims(
             ane_label_("decoder"),
             genai::MetalRelayVsrDecoder::ane_runtime_bytes(dc), 1)) {
      out.push_back(std::move(c));
    }
  }
  return out;
}

StageMemory
RelayVsrUpscaleStage::declare_memory() const
{
  StageMemory m;
  Configs cf;
  if (!configs_(&cf)) { return m; }
  const auto& wc = cf.wan;
  const auto& dc = cf.dec;
  // What the loads hold, once they have; the plan's figure before.
  m.hold(cf.wan_file, _m && _m->wan
                          ? _m->wan->held_bytes()
                          : genai::MetalRelayVsrWan::weight_bytes(wc));
  m.hold(cf.flash_file, _m && _m->dec
                            ? _m->dec->held_bytes()
                            : genai::MetalRelayVsrDecoder::weight_bytes(dc));
  // The launch's figure (see declare_resources); revise_scratch_ states
  // the run's.
  int ph = 0, pw = 0;
  (void)plan_in_(cf, &ph, &pw);
  m.scratch = genai::RelayVsrStream::run_bytes(wc, dc, ph, pw, _window);
  return m;
}

void
RelayVsrUpscaleStage::revise_scratch_(int h, int w)
{
  const std::uint64_t bytes = run_bytes_(h, w);
  if (bytes == 0 || bytes == _scratch_stated) { return; }
  _scratch_stated = bytes;
  if (auto* mgr = session()->services()->generative_model_manager()) {
    mgr->revise_scratch(scratch_label_(), bytes);
  }
  StageMemory m = declare_memory();
  m.scratch = bytes;
  revise_memory(m);
}

std::size_t
RelayVsrUpscaleStage::gpu_working_set() const
{
  auto* mc = session() != nullptr && session()->services() != nullptr
                 ? session()->services()->metal_compute()
                 : nullptr;
  if (mc == nullptr) { return 0; }
  const auto mb = mc->memory_budget();
  std::size_t ws = mb.recommended;
  // Scaled to the believed RAM (total_physical honours VPIPE_RAM_LIMIT_MB),
  // so a smaller box is simulated wherever it is asked.
  std::uint64_t real = 0;
  std::size_t n = sizeof(real);
  if (mb.total_physical > 0 &&
      ::sysctlbyname("hw.memsize", &real, &n, nullptr, 0) == 0 &&
      mb.total_physical < real) {
    ws = (std::size_t)((double)ws * (double)mb.total_physical /
                       (double)real);
  }
  return ws;
}

// WHETHER A RUN FITS, before it allocates: `weights` the models still to
// load (0 once held), `run` the stream's working set at the frame read --
// the keyframes' ring of keys and values, which grows with the frame and
// is what does not fit first. Nothing here streams, so a run past the box
// cannot shrink to fit it: it pages every window. Refused instead, as
// generate-video refuses a forward, with the numbers in a "memory" report
// (common/host-report.h) here and the sentence in *why, for the caller to
// raise once it has given back what it holds: session()->error() THROWS,
// so nothing after it runs, and the runtime ends the stage (its outputs
// closed, its producer let go).
//
// TWO MOMENTS, and they ask different questions.
//
// AT THE LAUNCH (an output size given): does the run fit the GPU working
// set even ALONE -- the whole of it, nothing subtracted. Every stage
// initializes at once, so what is allocated then depends on which peer
// happened to load first, and a peer may give its memory back before
// this stage's first frame (an upstream generator's model, phased to its
// denoise). A refusal taken on that reading would come and go with the
// load order. Alone it is a property of the config and the device, the
// same at every launch; peers are the plan's to weigh (it books this run
// at the size given, and lets a warm model go where the launch does not
// fit beside it).
//
// AT THE FIRST FRAME, after every stage has loaded: what is LEFT of the
// working set (Metal's recommended size less what is allocated) and
// RECLAIMABLE RAM, 5% kept each, as generate-video's forward is judged.
//
// The working set is scaled to the RAM the session believes
// (gpu_working_set), so VPIPE_RAM_LIMIT_MB simulates a smaller box.
bool
RelayVsrUpscaleStage::fits_(std::uint64_t weights, std::uint64_t run, int h,
                            int w, Gate gate, std::string* why)
{
  auto* mc = session()->services()->metal_compute();
  if (mc == nullptr || run == 0) { return true; }
  const auto mb = mc->memory_budget();
  if (mb.recommended == 0) { return true; }      // the query unavailable
  const bool launch = gate == Gate::kLaunch;
  const std::size_t ws = gpu_working_set();
  const std::size_t have =
      launch ? ws : ws > mb.allocated ? ws - mb.allocated : 0;
  constexpr double kMargin = 0.05;
  auto grossed = [](std::uint64_t b) {
    return (std::size_t)((double)b / (1.0 - kMargin));
  };
  const std::size_t need = (std::size_t)(weights + run);
  const std::size_t want = grossed(need);
  const bool ws_ok = want <= have;
  const bool ph_ok =
      launch || mb.available_physical == 0 || want <= mb.available_physical;
  if (ws_ok && ph_ok) { return true; }
  {
    auto u = [](std::size_t b) { return FlexData::make_uint(b); };
    FlexData parts = FlexData::make_array();
    auto part = [&](const char* name, std::size_t b) {
      if (b == 0) { return; }
      FlexData e = FlexData::make_object();
      e.as_object().insert("name", FlexData::make_string(name));
      e.as_object().insert("bytes", u(b));
      parts.as_array().push_back(std::move(e));
    };
    part("weights", (std::size_t)weights);
    part("frame_ring", (std::size_t)run);
    FlexData gates = FlexData::make_array();
    auto gate_doc = [&](const char* name, bool ok, std::size_t had) {
      FlexData e = FlexData::make_object();
      auto eo = e.as_object();
      eo.insert("name", FlexData::make_string(name));
      eo.insert("need", u(want));
      eo.insert("have", u(had));
      eo.insert("ok", FlexData::make_bool(ok));
      gates.as_array().push_back(std::move(e));
    };
    gate_doc("gpu_working_set", ws_ok, have);
    if (!launch) {
      gate_doc("reclaimable_ram", ph_ok, mb.available_physical);
    }
    FlexData d = FlexData::make_object();
    auto o = d.as_object();
    o.insert("stage", FlexData::make_string(this->id()));
    o.insert("step", FlexData::make_string("upscale"));
    o.insert("at", FlexData::make_string(launch ? "launch" : "first_frame"));
    o.insert("width", FlexData::make_uint((std::uint64_t)w));
    o.insert("height", FlexData::make_uint((std::uint64_t)h));
    o.insert("need", u(need));
    o.insert("parts", std::move(parts));
    o.insert("gates", std::move(gates));
    host_report(session(), "memory", std::move(d));
  }
  // WHAT WOULD HELP, and only that: smaller frames do nothing for weights
  // that do not fit alone, and the int8 layout is offered only where the
  // weights are not held in it already and the GPU has the matrix cores.
  int tier = -1;
  if (_m) {
    tier = _m->wan_cfg.i8_tier;
  } else if (Configs cf; configs_(&cf)) {
    tier = cf.wan.i8_tier;
  }
  std::vector<std::string> helps;
  if (weights == 0 || grossed(weights) <= have) {
    helps.emplace_back("feed smaller frames (the ring grows with them)");
  }
  if (tier < 1 && mc->supports_matrix_cores()) {
    helps.emplace_back("hold the weights in int8 (`i8_gemm` with "
                       "`i8_gemm_tier` 'fast' or 'token': about half the "
                       "bytes)");
  }
  if (!launch) { helps.emplace_back("free what else is holding memory"); }
  std::string help;
  for (std::size_t i = 0; i < helps.size(); ++i) {
    help += i == 0 ? "" : i + 1 == helps.size() ? ", or " : ", ";
    help += helps[i];
  }
  if (!help.empty()) {
    help[0] = (char)std::toupper((unsigned char)help[0]);
    help = " " + help + ".";
  }
  const std::string against =
      launch ? fmt("the ~{} MB of GPU working set this Mac gives a process "
                   "(SHORT): the run cannot fit even alone, so its weights "
                   "are not read", have >> 20)()
             : fmt("~{} MB of GPU working set left{} and ~{} MB of "
                   "reclaimable RAM{}", have >> 20, ws_ok ? "" : " (SHORT)",
                   mb.available_physical >> 20, ph_ok ? "" : " (SHORT)")();
  *why = fmt(
      "RelayVsrUpscaleStage('{}'): not enough memory to upscale {}x{} "
      "frames ({}): it wants ~{} MB -- {} MB of weights and {} MB for the "
      "keyframes' ring at that size -- {} MB with the 5% margin, against "
      "{}. Refusing rather than paging every window: nothing here "
      "streams.{}",
      this->id(), w, h,
      launch ? "as output_width x output_height says" : "as fed",
      need >> 20, weights >> 20, run >> 20, want >> 20, against,
      help.empty() ? std::string(" This Mac cannot hold the models.")
                   : help)();
  return false;
}

bool
RelayVsrUpscaleStage::load_()
{
  if (_m && _m->wan && _m->dec) { return true; }
  auto m = std::make_unique<Model>();
  std::string err;
  Configs cf;
  if (!configs_(&cf, &err)) {
    session()->error(fmt("RelayVsrUpscaleStage('{}'): {}", this->id(), err));
    return false;
  }
  m->wan_file = cf.wan_file;
  m->flash_file = cf.flash_file;
  m->wan_cfg = cf.wan;
  m->dec_cfg = cf.dec;
  // THE ANE, as the plan granted it: each model's module in the room it
  // left (the decision taken here, after every stage has claimed).
  if (_ane_ffn) {
    struct Mod {
      const char* which;
      std::uint64_t bytes;
      bool stages;
      bool* on;
    };
    const Mod mods[] = {
        {"keyframe", genai::MetalRelayVsrWan::ane_runtime_bytes(cf.wan),
         genai::MetalRelayVsrWan::ane_stages(cf.wan), &m->wan_cfg.ane_ffn},
        {"decoder", genai::MetalRelayVsrDecoder::ane_runtime_bytes(cf.dec),
         genai::MetalRelayVsrDecoder::ane_stages(cf.dec),
         &m->dec_cfg.ane_ffn}};
    for (const auto& [which, bytes, stages, on] : mods) {
      if (!stages) {
        *on = false;
        session()->info(fmt("RelayVsrUpscaleStage('{}'): the int8 tier "
                            "holds the {}'s feed-forward, which the ANE "
                            "cannot stage; it keeps the GPU", this->id(),
                            which));
        continue;
      }
      const bool granted =
          model_memory::coreml_grant(session(), ane_label_(which),
                                     (std::size_t)bytes, 1) > 0;
      *on = granted;
      if (!granted) {
        session()->info(fmt("RelayVsrUpscaleStage('{}'): the ANE "
                            "feed-forward was asked for, but the plan left "
                            "the {}'s module ({} MB) no room; it keeps the "
                            "GPU", this->id(), which, bytes >> 20));
      }
    }
  }
  auto* mc = session()->services()->metal_compute();
  if (mc == nullptr) {
    session()->error(fmt("RelayVsrUpscaleStage('{}'): no metal-compute "
                         "backend", this->id()));
    return false;
  }
  const auto t0 = std::chrono::steady_clock::now();
  m->ws_wan = genai::open_weight_set(m->wan_file, session());
  m->ws_flash = genai::open_weight_set(m->flash_file, session());
  if (m->ws_wan && m->ws_flash) {
    m->wan = genai::MetalRelayVsrWan::load(m->ws_wan, mc, m->wan_cfg, &err);
    if (m->wan) {
      m->dec = genai::MetalRelayVsrDecoder::load(m->ws_flash, mc, m->dec_cfg,
                                                 &err);
    }
  }
  if (!m->wan || !m->dec) {
    session()->error(fmt("RelayVsrUpscaleStage('{}'): '{}': {}", this->id(),
                         _hf_dir, err.empty() ? "cannot open" : err));
    return false;
  }
  // What the loads settled on (the int8 tier a GPU without matrix cores
  // declines) is what the run books.
  m->wan_cfg = m->wan->config();
  m->dec_cfg = m->dec->config();
  const int tier = m->wan_cfg.i8_tier;
  // ...and so do the weights: the plan booked them at the layout the tier
  // predicts, and a context that declined it holds more.
  if (auto* mgr = session()->services()->generative_model_manager()) {
    mgr->revise_declaration(m->wan_file, m->wan->held_bytes());
    mgr->revise_declaration(m->flash_file, m->dec->held_bytes());
  }
  session()->info(fmt(
      "RelayVsrUpscaleStage('{}'): RelayVSR loaded in {:.1f} s -- keyframe "
      "DiT {} blocks x {} ({} MB held), decoder {} blocks x {} (f16), "
      "GEMMs on {}{}{}{}; a keyframe every {} frames",
      this->id(),
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count(),
      m->wan_cfg.layers, m->wan_cfg.hidden,
      m->wan->held_bytes() >> 20,
      m->dec_cfg.depth, m->dec_cfg.dim,
      m->wan->uses_matrix_cores() ? "the matrix cores" : "the steel tiles",
      tier < 0 ? std::string()
               : fmt(", int8 {}", tier == 0   ? "accurate"
                                  : tier == 1 ? "fast"
                                  : tier == 2 ? "token"
                                              : "token_plain")(),
      m->wan_cfg.ane_ffn || m->dec_cfg.ane_ffn
          ? ", the feed-forward split with the Neural Engine"
          : "",
      m->wan_cfg.sage_pv8   ? ", attention int8 (QK and P*V)"
      : m->wan_cfg.sage_attn ? ", attention int8 (QK)"
                             : "",
      _window - 1));
  if (_i8_tier >= 0 && tier < 0) {
    session()->info(fmt("RelayVsrUpscaleStage('{}'): i8_gemm asks for the "
                        "matrix cores' int8 pipe, which this GPU does not "
                        "have; the GEMMs stay f16 / bf16", this->id()));
  }
  if (_sage_attn && !m->wan_cfg.sage_attn) {
    session()->info(fmt("RelayVsrUpscaleStage('{}'): sage_attn asks for the "
                        "matrix cores' attention, which this GPU does not "
                        "have; attention stays f16 / bf16", this->id()));
  }
  _m = std::move(m);
  ++_loads;
  return true;
}

void
RelayVsrUpscaleStage::book_ane_()
{
  if (_ane_booked || !_ane_ffn || !_m || !_m->wan || !_m->dec) { return; }
  // A model the plan granted tries its tier at its first forward; until
  // both have, the booking stands as claimed.
  if ((_m->wan_cfg.ane_ffn && !_m->wan->ane_attempted()) ||
      (_m->dec_cfg.ane_ffn && !_m->dec->ane_attempted())) {
    return;
  }
  _ane_booked = true;
  auto* mgr = session()->services()->generative_model_manager();
  if (mgr == nullptr) { return; }
  // What each module holds at the chunk its first forward built it at
  // (smaller rows, a smaller chunk), 0 where none was built.
  mgr->revise_scratch("coreml:" + ane_label_("keyframe"),
                      _m->wan->ane_held_bytes());
  mgr->revise_scratch("coreml:" + ane_label_("decoder"),
                      _m->dec->ane_held_bytes());
}

void
RelayVsrUpscaleStage::release_()
{
  if (!_m) { return; }
  const std::string wan = _m->wan_file, flash = _m->flash_file;
  _m->bar.finish();
  _m.reset();
  withdraw_(wan, flash);
}

void
RelayVsrUpscaleStage::withdraw_(const std::string& wan,
                                const std::string& flash)
{
  // Everything this stage booked is given back: the weights'
  // declarations and the run's arena withdrawn (0 withdraws), so a peer
  // sizing after the end of this stream -- or after a refusal, when
  // nothing was loaded -- does not count them.
  if (auto* mgr = session()->services()->generative_model_manager()) {
    mgr->drop_weights(wan);
    mgr->drop_weights(flash);
    mgr->revise_declaration(wan, 0);
    mgr->revise_declaration(flash, 0);
    mgr->revise_scratch(scratch_label_(), 0);
    if (_ane_ffn) {
      mgr->revise_scratch("coreml:" + ane_label_("keyframe"), 0);
      mgr->revise_scratch("coreml:" + ane_label_("decoder"), 0);
    }
  }
  _scratch_stated = 0;
  revise_memory(StageMemory{});
}

Job
RelayVsrUpscaleStage::initialize(RuntimeContext& ctx)
{
  (void)ctx;
  // An output size given (output_width): a run that cannot fit even
  // alone is refused before 8 GB of weights are read for it, and what the
  // plan booked for it withdrawn -- nothing will hold it.
  if (_plan_given && !(_m && _m->wan)) {
    Configs cf;
    std::string why;
    int ph = 0, pw = 0;
    if (configs_(&cf) &&
        (!plan_in_(cf, &ph, &pw, &why) ||
         !fits_(genai::MetalRelayVsrWan::weight_bytes(cf.wan) +
                    genai::MetalRelayVsrDecoder::weight_bytes(cf.dec),
                run_bytes_(ph, pw), ph, pw, Gate::kLaunch, &why))) {
      withdraw_(cf.wan_file, cf.flash_file);
      session()->error(fmt("{}", why));    // throws: the stage ends here
      co_return;
    }
  }
  load_();
  co_return;
}

void
RelayVsrUpscaleStage::reset_run_state()
{
  if (_m && _m->stream) { _m->stream->reset(); }
  if (_m) { _m->stream.reset(); }
  _sb.clear();
  _emitted = 0;
  // The ledger was re-declared at this launch's plan: the first frame
  // states the run's figure whatever the last run stated.
  _scratch_stated = 0;
  _f32_out = false;
  _warned_shape = false;
  _ane_booked = false;
}

FlexData
RelayVsrUpscaleStage::out_sideband_(std::int64_t index)
{
  for (auto it = _sb.begin(); it != _sb.end(); ++it) {
    if (it->first == index) {
      FlexData sb = std::move(it->second);
      _sb.erase(it);
      return sb.is_object() ? sb : FlexData::make_object();
    }
  }
  return FlexData::make_object();
}

std::unique_ptr<TensorBeatPayload>
RelayVsrUpscaleStage::beat_(std::int64_t index, int h, int w, const void* px,
                            std::size_t bytes)
{
  auto t = std::make_unique<TensorBeatPayload>();
  t->dtype = _f32_out ? TensorBeat::DType::F32 : TensorBeat::DType::U8;
  t->shape = {3, h, w};
  const auto* p = (const std::uint8_t*)px;
  t->data.assign(p, p + bytes);
  t->sideband = out_sideband_(index);
  return t;
}

Job
RelayVsrUpscaleStage::process(RuntimeContext& ctx)
{
  auto stop = [&ctx]() { return ctx.stop_requested(); };
  auto p = co_await ctx.read(0);
  const bool eos = !p;
  if (eos) {
    if (_m && _m->stream) {
      std::string err;
      if (!_m->stream->finish(&err, stop) && !err.empty()) {
        session()->warn(fmt("RelayVsrUpscaleStage('{}'): the last window: "
                            "{}", this->id(), err));
      }
    }
  } else {
    if (!_m || !_m->wan) { co_return; }     // its error was at launch
    const auto* tb = dynamic_cast<const TensorBeatPayload*>(p.get());
    if (tb == nullptr || tb->shape.size() != 3 || tb->shape[0] != 3 ||
        tb->shape[1] <= 0 || tb->shape[2] <= 0 ||
        (tb->dtype != TensorBeat::DType::U8 &&
         tb->dtype != TensorBeat::DType::F32 &&
         tb->dtype != TensorBeat::DType::F16)) {
      if (!_warned_shape) {
        _warned_shape = true;
        session()->warn(fmt(
            "RelayVsrUpscaleStage('{}'): expected one planar RGB frame a "
            "beat -- [3, H, W], u8, f32 or f16; got {}; skipping",
            this->id(),
            tb != nullptr ? std::string(TensorBeat::name_of(tb->dtype))
                          : std::string(p->describe())));
      }
      co_return;
    }
    const int h = (int)tb->shape[1], w = (int)tb->shape[2];
    const auto kind = tb->dtype == TensorBeat::DType::U8
                          ? genai::RelayVsrStream::In::kU8
                          : tb->dtype == TensorBeat::DType::F32
                                ? genai::RelayVsrStream::In::kF32
                                : genai::RelayVsrStream::In::kF16;
    std::string why;
    if (!_m->stream &&
        !fits_(0, run_bytes_(h, w), h, w, Gate::kFirstFrame, &why)) {
      // The models let go and everything booked withdrawn BEFORE the
      // error, which throws: the runtime then ends this stage, closing its
      // output and letting its producer go, so the pipeline ends.
      release_();
      session()->error(fmt("{}", why));
      co_return;
    }
    if (!_m->stream) {
      _f32_out = _out_dtype == "f32" ||
                 (_out_dtype == "auto" &&
                  kind != genai::RelayVsrStream::In::kU8);
      genai::RelayVsrStream::Options o;
      o.window = _window;
      o.seed = _seed;
      o.f32_out = _f32_out;
      auto* mc = session()->services()->metal_compute();
      _m->stream = std::make_unique<genai::RelayVsrStream>(
          _m->wan.get(), _m->dec.get(), mc, o);
      revise_scratch_(h, w);
      _m->bar = session()->open_progress("relayvsr upscale");
      _m->t_start = std::chrono::duration<double>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
      session()->info(fmt(
          "RelayVsrUpscaleStage('{}'): {}x{} -> {}x{} ({}x), {} out, a "
          "keyframe every {} frames", this->id(), w, h,
          _m->stream->scale() * w, _m->stream->scale() * h,
          _m->stream->scale(),
          _f32_out ? "f32" : "u8", _window - 1));
    }
    _sb.emplace_back(_m->stream->frames_in(), tb->sideband);
    std::string err;
    const bool ok = _m->stream->push(tb->as_u8(), kind, h, w, &err, stop);
    p.reset();                              // the source frame is copied
    if (!ok) {
      if (!err.empty()) {
        session()->warn(fmt("RelayVsrUpscaleStage('{}'): {}; the stream "
                            "restarts at the next frame", this->id(), err));
      }
      _m->stream->reset();
      _sb.clear();
      co_return;
    }
  }
  // A complete window, ONE FRAME AT A TIME: each decoded, copied into its
  // beat and handed on before the next is decoded, so this stage holds
  // one output frame on the host, not a window of them.
  if (_m && _m->stream) {
    genai::RelayVsrStream::Frame f;
    std::string err;
    while (_m->stream->next(&f, &err, stop)) {
      auto t = beat_(f.index, f.h, f.w, f.px, f.bytes);
      ++_emitted;
      co_await ctx.write(0, std::move(t));
      _m->bar.update(_emitted, 0,
                     fmt("frame {}, {} keyframe(s)", _emitted,
                         _m->stream->keyframes())());
      book_ane_();
    }
    if (!err.empty() || _m->stream->ready()) {
      // A failure, or a stop inside the window: the stream starts again.
      if (!err.empty()) {
        session()->warn(fmt("RelayVsrUpscaleStage('{}'): {}; the stream "
                            "restarts at the next frame", this->id(), err));
      }
      _m->stream->reset();
      _sb.clear();
    }
  }
  if (!eos) { co_return; }
  if (_m && _m->t_start > 0.0) {
    const double s =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count() - _m->t_start;
    session()->info(fmt(
        "RelayVsrUpscaleStage('{}'): {} frame(s) upscaled in {:.1f} s "
        "({:.2f} s a frame)", this->id(), _emitted, s,
        _emitted > 0 ? s / (double)_emitted : 0.0));
  }
  release_();
  ctx.signal_done();
}

VPIPE_REGISTER_STAGE(RelayVsrUpscaleStage)
VPIPE_REGISTER_SPEC(RelayVsrUpscaleStage, kSpec)

}  // namespace vpipe

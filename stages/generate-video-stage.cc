#include "stages/generate-video-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/perf-scope.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "stages/model-registry.h"
#include "stages/denoise-progress.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/wan/metal-wan-vae.h"
#include "generative-models/weight-set.h"
#endif

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace vpipe {

namespace {

const ConfigKey kAttrs[] = {
  {.key = "hf_dir", .type = ConfigType::String, .required = false,
   .doc = "video model root -- Wan (transformer/, transformer_2/, vae/, "
          "text_encoder/) or a MiniMax-H3 repack (diffusion_models/, vae/, "
          "audio_vae/, text_encoders/). OPTIONAL: a model-select source on "
          "the model iport overrides it",
   .suggest_db = kModelRegistryDb,
   .suggest_db_type = "wan-i2v,wan-t2v,minimax-h3-fl2va"},
  {.key = "height", .type = ConfigType::Int, .required = false,
   .doc = "video height in pixels; a multiple of 16 (the VAE's 8x times the "
          "DiT's 2x patch)", .def_int = 480},
  {.key = "width", .type = ConfigType::Int, .required = false,
   .doc = "video width in pixels; a multiple of 16", .def_int = 832},
  {.key = "frames", .type = ConfigType::Int, .required = false,
   .doc = "video frames. Rounded UP to the nearest count the resident "
          "model's VAE can chunk, which differs per family, so any positive "
          "number is accepted here", .def_int = 81},
  {.key = "fps", .type = ConfigType::Real, .required = false,
   .doc = "frame rate stamped on the emitted latent, for the decoder and the "
          "encoder downstream of it", .def_real = 16.0},
  {.key = "steps", .type = ConfigType::Int, .required = false,
   .doc = "denoising steps", .def_int = 40},
  {.key = "seed", .type = ConfigType::Int, .required = false,
   .doc = "initial-noise RNG seed (default 0)"},
  {.key = "guidance_scale", .type = ConfigType::Real, .required = false,
   .doc = "classifier-free guidance for the HIGH-noise expert. Needs a "
          "negative conditioning on iport1; without one, guidance is forced "
          "to 1", .def_real = 3.5},
  {.key = "guidance_scale_2", .type = ConfigType::Real, .required = false,
   .doc = "classifier-free guidance for the LOW-noise expert (A14B only)",
   .def_real = 3.5},
  {.key = "boundary_ratio", .type = ConfigType::Real, .required = false,
   .doc = "sigma at which the low-noise expert takes over from the high-noise "
          "one. Read from the checkpoint's model_index.json when present; 0 "
          "means a single expert", .def_real = 0.9},
  {.key = "video_shift", .type = ConfigType::Real, .required = false,
   .doc = "sigma shift for the VIDEO schedule (minimax-h3; inert on wan, "
          "which takes its shift from the scheduler spec)",
   .def_real = 12.0},
  {.key = "audio_shift", .type = ConfigType::Real, .required = false,
   .doc = "sigma shift for the AUDIO schedule (minimax-h3)", .def_real = 3.0},
  {.key = "condition_timestep", .type = ConfigType::Real, .required = false,
   .doc = "the timestep the pinned keyframe rows are conditioned on "
          "(minimax-h3); 1.0 is CLEAN in this model's t = 1 - sigma "
          "convention", .def_real = 1.0},
  {.key = "audio_seconds", .type = ConfigType::Real, .required = false,
   .doc = "audio duration for minimax-h3; 0 derives it from frames / fps",
   .def_real = 0.0},
  {.key = "unload_when_idle", .type = ConfigType::String, .required = false,
   .doc = "drop the expert's weights after each clip and reload on the next "
          "one. \"auto\" (default) decides from physical RAM vs the "
          "pipeline's weight bytes; \"always\" / \"never\" force it",
   .def_str = "auto"},
};
const PortSpec kIports[] = {
  {.name = "conditioning",
   .doc = "umT5-XXL hidden states from a diffusion-conditioner: bf16 "
          "[text_seq, 4096]",
   .type = &typeid(TensorBeatPayload),
   .tags = "conditioning", .clock_group = 0},
  {.name = "neg_conditioning",
   .doc = "OPTIONAL negative conditioning (the conditioner's oport1) for "
          "classifier-free guidance. Wan is not distilled, so without this "
          "guidance is forced to 1",
   .type = &typeid(TensorBeatPayload),
   .tags = "conditioning", .clock_group = 0},
  {.name = "model", .doc = "OPTIONAL shared model reference from a model-select "
                           "source; overrides the hf_dir config",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
  {.name = "sampler",
   .doc = "OPTIONAL sampler spec FlexData (diffusion-sampler-select); the "
          "default is UniPC multistep, which is what Wan ships",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
  {.name = "scheduler",
   .doc = "OPTIONAL scheduler spec FlexData (scheduler-select)",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
  {.name = "ref_latent0",
   .doc = "OPTIONAL image-to-video conditioning latent from vae-encode: f32 "
          "[16, T, H/8, W/8] for wan (the conditioning image followed by "
          "blank frames), f32 [24, 1, H/16, W/16] for minimax-h3 (the "
          "FIRST-frame keyframe anchor). Present => image-to-video",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
  {.name = "ref_latent1",
   .doc = "OPTIONAL LAST-frame keyframe anchor from a second vae-encode "
          "(minimax-h3 only; wan's i2v latent is one clip-shaped tensor). "
          "With ref_latent0 this is the FL2VA first-and-last mode",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
};
[[maybe_unused]] constexpr unsigned kModelPort   = 2;
[[maybe_unused]] constexpr unsigned kSamplerPort = 3;
[[maybe_unused]] constexpr unsigned kSchedPort   = 4;
[[maybe_unused]] constexpr unsigned kRefPort     = 5;
[[maybe_unused]] constexpr unsigned kRefPort1    = 6;

const PortSpec kOports[] = {
  {.name = "latent",
   .doc = "f32 latent VIDEO [z, T, H/r, W/r] (whitened), sideband "
          "{fps, frames}. z/r are the family's VAE geometry: 16/8 for wan, "
          "24/16 for minimax-h3",
   .type = &typeid(TensorBeatPayload),
   .tags = "latent", .clock_group = 0},
  {.name = "audio_latent",
   .doc = "f32 latent AUDIO [audio_channels, audio_latents], minimax-h3 "
          "only -- the families that generate no audio never write it",
   .type = &typeid(TensorBeatPayload),
   .tags = "latent", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "generate-video",
  .doc       = "Wan video DiT denoiser: conditioning (+ an optional image "
               "latent) -> UniPC over a two-expert 14B transformer -> a latent "
               "video, on the metal-compute backend. Feed vae-decode.",
  .display_name = "Generate Video",
  .category  = StageCategory::Generative,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

#ifdef VPIPE_BUILD_APPLE_SILICON
// boundary_ratio lives in the PIPELINE config (model_index.json), not in
// either expert's own config, because it is a property of the pair.
double
boundary_from_index_(const std::string& root, double fallback)
{
  std::ifstream in(std::filesystem::path(root) / "model_index.json");
  if (!in) { return fallback; }
  FlexData fd;
  try {
    fd = FlexData::from_json(in);
  } catch (...) {
    return fallback;
  }
  if (!fd.is_object()) { return fallback; }
  auto o = fd.as_object();
  if (!o.contains("boundary_ratio")) { return fallback; }
  FlexData br = o.at("boundary_ratio");
  if (br.is_null()) { return 0.0; }   // explicitly a single-expert pipeline
  return br.as_real(fallback);
}
#endif

}  // namespace

GenerateVideoStage::GenerateVideoStage(const SessionContextIntf* s,
                                       std::string               id,
                                       std::vector<InEdge>       iports,
                                       FlexData                  config)
  : TypedStage<GenerateVideoStage>(s, std::move(id), std::move(iports),
                                   std::move(config))
{
  _hf_dir = attr_str("hf_dir");
  _height = (int)attr_int("height");
  _width  = (int)attr_int("width");
  _frames = (int)attr_int("frames");
  _fps    = attr_real("fps");
  _steps  = (int)attr_int("steps");
  _seed   = (std::uint64_t)attr_int("seed");
  _guidance   = attr_real("guidance_scale");
  _guidance_2 = attr_real("guidance_scale_2");
  _boundary   = attr_real("boundary_ratio");
  _video_shift   = attr_real("video_shift");
  _audio_shift   = attr_real("audio_shift");
  _cond_timestep = attr_real("condition_timestep");
  _audio_seconds = attr_real("audio_seconds");
  if (_height <= 0) { _height = 480; }
  if (_width  <= 0) { _width  = 832; }
  if (_frames <= 0) { _frames = 81; }
  if (_steps  <= 0) { _steps  = 40; }
  if (!(_fps > 0.0)) { _fps = 16.0; }
  // Deferred validation: never throw from a constructor, so a geometry the
  // model cannot represent is reported and the runtime skips the stage.
  if ((_height % 16) != 0 || (_width % 16) != 0) {
    fail_config(fmt("height {} and width {} must both be multiples of 16 "
                    "(the VAE's 8x downsample times the DiT's 2x patch)",
                    _height, _width));
  }
  // The frame count is NOT validated here, deliberately. A count the VAE
  // cannot chunk has no latent representation -- but the rule is PER
  // FAMILY (wan compresses in 4-frame chunks after a 1-frame first chunk,
  // 4k+1; MiniMax-H3 takes 17-frame clips keeping 5 latents, 17n+5) and
  // the family is only known once the checkpoint is read, which happens
  // after this constructor. So resolve_config_ rounds UP to the resident
  // family's rule instead, and any positive count is accepted.
  //
  // Rounding rather than rejecting because the two rules share almost no
  // legal counts -- 81 is fine for wan and impossible for H3, 56 the
  // reverse -- so a graph that names one family's number could not change
  // checkpoints without also being re-authored, which is exactly what
  // being family-generic is supposed to buy.
#ifdef VPIPE_BUILD_APPLE_SILICON
  {
    bool bad = false;
    _unload_cfg = model_memory::parse_unload_policy(
        attr_str("unload_when_idle"), &bad);
    if (bad) {
      session()->warn(fmt(
          "GenerateVideoStage('{}'): unload_when_idle '{}' is not "
          "auto|always|never; using auto", this->id(),
          attr_str("unload_when_idle")));
    }
  }
  // UniPC multistep on the checkpoint's flow schedule -- the sampler every
  // Wan checkpoint ships with. A sampler-select source overrides it.
  _sampler_spec.method = "unipc";
  _sampler_spec.order = 2;
  _sampler_spec.solver_bh2 = true;
  _scheduler_spec.type = "unipc_flow";
  _scheduler_spec.steps = _steps;
  _scheduler_spec.shift = 3.0;
  _scheduler_spec.num_train = 1000;
#endif
  allocate_oports(spec().oports.size());
}

GenerateVideoStage::~GenerateVideoStage() = default;

const StageSpec&
GenerateVideoStage::spec() const noexcept
{
  return kSpec;
}

int
GenerateVideoStage::latent_frames() const noexcept
{
  return _frames > 0 ? 1 + (_frames - 1) / 4 : 0;
}

void
GenerateVideoStage::align_frames_(int aligned)
{
  if (aligned <= 0 || aligned == _frames) { return; }
  // SAY SO. The clip that comes back is longer than the one that was
  // asked for, its duration at `fps` is longer to match, and a graph
  // downstream that assumed the configured count would otherwise find out
  // by way of a shape it did not expect. One line naming both numbers is
  // cheaper than that, and it is info rather than a warning because
  // rounding is the designed behaviour, not a problem being tolerated.
  session()->info(fmt(
      "GenerateVideoStage('{}'): frames {} -> {}, the nearest count at or "
      "above it that the {} VAE can chunk", this->id(), _frames, aligned,
      _family));
  _frames = aligned;
}

std::vector<ResourceClaim>
GenerateVideoStage::declare_resources() const
{
#ifdef VPIPE_BUILD_APPLE_SILICON
  if (_hf_dir.empty()) { return {}; }
  namespace fs = std::filesystem;
  const std::string root = resolve_model_dir(session(), _hf_dir);
  // ONE expert, not both. The stage holds exactly one at a time, so
  // claiming the pair would size every peer against a peak that never
  // occurs and push them all into streaming for nothing.
  //
  // Through the DiT resolver, because this runs BEFORE resolve_config_ and
  // so before the family is known -- and a repack spells its DiT
  // `diffusion_models/`, not `transformer/`. Claiming a path that does not
  // exist is not a harmless miss: this is the declaration every peer sizes
  // itself against, so a 24 GB DiT reported as 0 bytes is exactly the
  // silent under-count the resource-planning phase exists to prevent.
  // Falls back to the diffusers spelling when nothing resolves.
  std::string dit = genai::MetalMiniMaxH3Transformer::resolve_dit_dir(root);
  if (dit == root) { dit = (fs::path(root) / "transformer").string(); }
  return model_memory::weight_claims({dit});
#else
  return {};
#endif
}

void
GenerateVideoStage::reset_run_state()
{
  _emitted = 0;
  _model_latched = false;
  _sampler_latched = false;
  _scheduler_latched = false;
}

#ifdef VPIPE_BUILD_APPLE_SILICON

using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

void
GenerateVideoStage::resolve_config_()
{
  if (_have_cfg || _hf_dir.empty()) { return; }
  namespace fs = std::filesystem;
  _root = resolve_model_dir(session(), _hf_dir);
  const std::string t1 = (fs::path(_root) / "transformer").string();
  // The DiT's own `_class_name` picks the family. Reading it here rather
  // than taking it from config keeps a graph from having to be rewired
  // to change checkpoints -- the same stage, ports and keys serve both.
  //
  // H3 is asked FIRST, and through its own reader rather than by opening
  // `transformer/config.json`, because that spelling is a diffusers one
  // and this family also ships as a Comfy-Org repack -- one .safetensors
  // under `diffusion_models/` with the config in its `__metadata__`, or a
  // directory checkpoint there once model-quantize has processed it.
  // config_from_json resolves all three and REFUSES a checkpoint that is
  // not H3, which is what makes trying it first safe: a success is the
  // detection and the config at once, and a failure costs one JSON parse.
  // (Reading `transformer/config.json` directly is what left every stage
  // in this graph inert on a repack root.)
  std::string h3err;
  if (genai::MetalMiniMaxH3Transformer::config_from_json(_root, _h3_cfg,
                                                         &h3err)) {
    _family = "minimax-h3";
  }
  if (_family == "minimax-h3") {
    // Now that the family is known, round the frame count UP to H3's own
    // rule (see the constructor): the video VAE takes 17-frame clips and
    // keeps 5 latents from each, so only 17n+5 has a latent form.
    align_frames_(genai::minimax_h3::align_num_frames(_frames, 17, 5));
    // _h3_cfg is already filled -- the detection above IS the read.
    _two_experts = false;
    _boundary    = 0.0;      // one stack; nothing to switch at
    _have_cfg    = true;
    // NOTE: the idle-unload policy is NOT decided here. It needs the
    // streaming verdict, which only exists after the load -- see
    // resolve_unload_policy_h3_().
    session()->info(fmt(
        "GenerateVideoStage('{}'): MiniMax-H3 (video+audio) at {}x{}x{} "
        "frames, {} steps, shifts {:.1f}/{:.1f}", this->id(), _width, _height,
        _frames, _steps, _video_shift, _audio_shift));
    return;
  }
  std::string cerr;
  if (!genai::MetalWanTransformer::config_from_json(t1, _cfg, &cerr)) {
    session()->error(fmt(
        "GenerateVideoStage('{}'): {}; inert", this->id(), cerr));
    return;
  }
  align_frames_(genai::MetalWanVae::align_num_frames(_frames));
  _two_experts = fs::exists(fs::path(_root) / "transformer_2" / "config.json");
  _boundary = boundary_from_index_(_root, _boundary);
  if (!_two_experts) { _boundary = 0.0; }
  _have_cfg = true;
  if (!_unload_resolved) {
    _unload_resolved = true;
    const std::vector<std::string> peers = {
        (fs::path(_root) / "text_encoder").string(),
        (fs::path(_root) / "vae").string()};
    switch (_unload_cfg) {
      case model_memory::UnloadPolicy::kAlways: _unload_idle = true;  break;
      case model_memory::UnloadPolicy::kNever:  _unload_idle = false; break;
      default:
        _unload_idle = model_memory::bounded(session(), peers,
                                             model_memory::kHeadroom);
        break;
    }
  }
  session()->info(fmt(
      "GenerateVideoStage('{}'): Wan {} at {}x{}x{} frames, {} steps, "
      "in_channels {}{}", this->id(),
      _two_experts ? "A14B (two experts)" : "single-expert",
      _width, _height, _frames, _steps, _cfg.in_channels,
      _two_experts ? fmt(", boundary {:.2f}", _boundary)() : std::string()));
}

bool
GenerateVideoStage::ensure_expert_(int which)
{
  if (_family == "minimax-h3") {
    resolve_config_();
    if (!_have_cfg) { return false; }
    if (_h3_dit) { return true; }
    auto* h3mc = session()->services()->metal_compute();
    if (h3mc == nullptr) { return false; }
    // The 33B DiT is the largest here: ~33 GB at w8, which does NOT fit
    // resident beside the 10 GB video VAE on a 64 GB box -- and it is the
    // SEQUENCE that decides, since the model's own default is 124 frames
    // (37 latent frames, ~19k rows at 960x544). Preloading it works only
    // for the short, low-resolution sequences that are outside the
    // model's supported range anyway, so this asks the same question the
    // other DiT families ask rather than assuming the stack fits.
    namespace fs = std::filesystem;
    const std::string dit_dir =
        genai::MetalMiniMaxH3Transformer::resolve_dit_dir(_root);
    // Through the encoder's own resolver, not by spelling a sibling of
    // the DiT: on a Comfy-Org repack the DiT's parent IS the root and the
    // encoder lives under `text_encoders/`, so building the path by hand
    // produced a directory that does not exist and silently dropped the
    // encoder out of the streaming decision.
    std::string enc_dir =
        genai::MiniMaxH3TextEncoder::resolve_encoder_dir(_root);
    if (enc_dir == _root || !fs::exists(enc_dir)) { enc_dir.clear(); }
    const auto plan = model_memory::plan_streaming(
        session(), dit_dir, enc_dir, model_memory::kStreamHeadroom);
    bool   stream_blocks = plan.stream;
    double pin_frac      = plan.pin_frac;
    if (const char* e = std::getenv("VPIPE_H3_STREAM")) {
      stream_blocks = (std::atoi(e) != 0);
      if (!stream_blocks) { pin_frac = 0.0; }
    }
    if (const char* e = std::getenv("VPIPE_H3_PIN_FRAC")) {
      pin_frac = std::atof(e);
    }
    session()->log_debug(fmt(
        "GenerateVideoStage('{}'): MiniMax-H3 footprint {} GB (others {} GB) "
        "+ {} GB headroom -> {}", this->id(), plan.footprint >> 30,
        plan.others >> 30, model_memory::kStreamHeadroom >> 30,
        stream_blocks ? "STREAM blocks" : "PRELOAD"));
    _h3_dit = genai::MetalMiniMaxH3Transformer::load(
        genai::open_weight_set(dit_dir, session()), h3mc, _h3_cfg,
        stream_blocks, pin_frac);
    resolve_unload_policy_h3_(stream_blocks);
    if (_h3_dit && stream_blocks) {
      // A streamed DiT holds ~one block, not the checkpoint, so the
      // load-time claim would keep sizing every peer against 33 GB. What
      // the SET holds is the right number: the retained top-level
      // tensors are cached there, the streamed blocks deliberately are
      // not.
      auto* mgr = session()->services()->generative_model_manager();
      auto ws = mgr != nullptr ? mgr->weight_set(dit_dir) : nullptr;
      if (ws) {
        const std::size_t held = ws->stats().bytes;
        mgr->revise_declaration(dit_dir, held);
        session()->log_debug(fmt(
            "GenerateVideoStage('{}'): streaming DiT keeps {} MB resident, "
            "revised down from {} MB on disk", this->id(), held >> 20,
            model_memory::dir_weights_bytes(dit_dir) >> 20));
      }
    }
    // Say how many blocks are PINNED, the way the conditioner says it for
    // its layers. Worth stating even when it is zero, which is the common
    // case here and is not obvious: plan_streaming only pins when
    // RAM > others + 5 GB, and with the text encoder counted in `others`
    // at ~15 GB on a 16 GB box that test fails outright. Zero pinned is
    // why the resident set below has to earn its room at runtime.
    if (_h3_dit) {
      session()->info(fmt(
          "GenerateVideoStage('{}'): MiniMax-H3 DiT {} of {} blocks pinned "
          "at load{}", this->id(), _h3_dit->pinned_blocks(), _h3_cfg.n_layers,
          _h3_dit->pinned_blocks() == 0
              ? " (none fit beside the other models; the resident set grows "
                "into free RAM as the denoise runs)"
              : ""));
    }
    if (!_h3_dit) {
      session()->error(fmt(
          "GenerateVideoStage('{}'): MiniMax-H3 DiT load failed from '{}'",
          this->id(), _root));
      return false;
    }
    return true;
  }
  if (_dit && _expert == which) { return true; }
  resolve_config_();
  if (!_have_cfg) { return false; }
  auto* mc = session()->services()->metal_compute();
  if (mc == nullptr) { return false; }
  namespace fs = std::filesystem;
  const std::string dir =
      (fs::path(_root) / (which == 0 ? "transformer" : "transformer_2"))
          .string();
  // Drop first, THEN load. At bf16 an expert is ~28 GB, so holding both
  // across the swap is not a transient peak, it is an out-of-memory kill.
  if (_dit) {
    _dit.reset();
    _expert = -1;
  }
  session()->info(fmt(
      "GenerateVideoStage('{}'): loading the {}-noise expert from '{}'",
      this->id(), which == 0 ? "high" : "low", dir));
  std::shared_ptr<genai::WeightSet> ws = genai::open_weight_set(dir, session());
  if (!ws) {
    session()->error(fmt(
        "GenerateVideoStage('{}'): no readable checkpoint under '{}'",
        this->id(), dir));
    return false;
  }
  _dit = genai::MetalWanTransformer::load(ws, mc, _cfg);
  if (!_dit) {
    session()->error(fmt(
        "GenerateVideoStage('{}'): failed to load the DiT from '{}'",
        this->id(), dir));
    return false;
  }
  _expert = which;
  return true;
}

Job
GenerateVideoStage::initialize(RuntimeContext& ctx)
{
  // Defer everything when a model-select source feeds the model iport (its
  // beat only arrives after the init barrier). The expert itself is loaded
  // lazily at the first denoise step regardless -- it is 28 GB, and a graph
  // that never receives a prompt should not pay for it.
  if (!(ctx.num_iports() > kModelPort && ctx.iport_connected(kModelPort))) {
    resolve_config_();
  }
  co_return;
}

namespace h3 = genai::minimax_h3;

// Should the DiT be dropped after each clip? Decided AFTER the load, and
// that timing is the whole point.
//
// `auto` asks model_memory::bounded(peers), i.e. "does RAM cover the
// PEERS' weights plus headroom". Asked before those peers have loaded it
// measures a footprint of ~0 and always answers "roomy" -- and for this
// stage the peers are the text encoder and the video VAE, both of which
// are now loaded lazily by OTHER stages, so at any moment this stage
// could ask, the honest answer is zero. The rule was never wrong; it was
// being handed nothing.
//
// So the streaming verdict decides instead, and it is a better signal
// anyway: plan_streaming has ALREADY compared this checkpoint against the
// box and concluded the stack does not fit resident. A model that has to
// stream its own blocks is not one to keep alive across a decode that
// needs the working set -- which on MiniMax-H3 is exactly what happened:
// the DiT stayed resident and a 251 MB video decode was refused with zero
// headroom and 5.3 GB physically free.
void
GenerateVideoStage::resolve_unload_policy_h3_(bool streamed)
{
  if (_unload_resolved) { return; }
  _unload_resolved = true;
  std::vector<std::string> peers;
  for (const std::string& p :
       {genai::MiniMaxH3TextEncoder::resolve_encoder_dir(_root),
        genai::MetalMiniMaxH3VideoVae::resolve_vae_dir(_root)}) {
    if (!p.empty() && p != _root) { peers.push_back(p); }
  }
  switch (_unload_cfg) {
    case model_memory::UnloadPolicy::kAlways: _unload_idle = true;  break;
    case model_memory::UnloadPolicy::kNever:  _unload_idle = false; break;
    default:
      _unload_idle = streamed ||
                     model_memory::bounded(session(), peers,
                                           model_memory::kHeadroom);
      break;
  }
  if (_unload_idle) {
    session()->info(fmt(
        "GenerateVideoStage('{}'): memory-bounded -- the DiT is dropped "
        "after each clip so the VAE decode has working set, and reloads "
        "on the next prompt{}", this->id(),
        streamed ? " (its blocks stream, so the box is already tight)" : ""));
  }
}

// Is there room for this forward's SCRATCH, and if not, can we make it?
//
// The image stages never needed this: a 1024x1024 DiT's activations are
// tens of MB. A video forward's are not -- MiniMax-H3's scratch is ~200 KB
// per row, so the 9382-row production layout wants ~1.9 GB and a 19k-row
// one ~3.9 GB, none of which the weight accounting can see (it is
// allocated per forward, not per checkpoint).
//
// Getting this wrong on a 16 GB box is not a failed allocation. Metal
// SharedBuffers are mlock-WIRED, so they cannot be paged out or
// compressed: the VM drains the file cache, fills the compressor, stops
// making progress, userspace stops being scheduled, and the kernel
// watchdog panics the machine. Two panic logs on this box say exactly
// that ("no checkins from watchdogd in 94 seconds", free 14 MB, file
// cache ~0, 4-5 GB wired). So refusing loudly is the SAFE outcome here,
// and proceeding hopefully is the dangerous one.
bool
GenerateVideoStage::preflight_h3_scratch_(int seq, int text_rows)
{
  auto* mc = session()->services()->metal_compute();
  if (mc == nullptr || seq <= 0) { return true; }
  auto mb = mc->memory_budget();
  if (mb.recommended == 0) { return true; }   // budget query unavailable

  // The distinct-timestep count is decided inside denoise(), but its
  // buffers are sub-MB at any plausible value (mod is n_t x adaln_out),
  // so an upper bound costs nothing and keeps this callable before the
  // schedule exists.
  constexpr int kTimestepsUpperBound = 4;
  const bool dq = _h3_dit && _h3_dit->uses_matrix_cores();
  const std::size_t need = genai::MetalMiniMaxH3Transformer::scratch_bytes(
      _h3_cfg, seq, text_rows, kTimestepsUpperBound, dq);

  // Tell the DiT how much room to leave clear when it decides whether to
  // keep a streamed block resident. Its growth must not eat the scratch
  // the very next forward needs -- that would trade a disk read for an
  // allocation failure. A further margin on top keeps room for the peers
  // that run after the denoise (the VAE decode above all).
  if (_h3_dit) {
    _h3_dit->set_residency_reserve(need + (1ull << 30));
  }

  // Both budgets, for the reason generate-image gives: fits() is our Metal
  // working set and misses other processes' resident memory; fits_physical
  // is host-wide reclaimable RAM and catches it.
  if (mb.fits(need) && mb.fits_physical(need)) { return true; }

  // Make room. Parking hands weight pages to the kernel as purgeable --
  // reversible, and free unless something actually takes them.
  std::size_t parked = 0;
  if (auto* gm = session()->services()->generative_model_manager()) {
    parked = gm->reclaim_at_least(need);
  }
  mb = mc->memory_budget();
  if (mb.fits(need) && mb.fits_physical(need)) {
    session()->info(fmt(
        "GenerateVideoStage('{}'): parked ~{} MB to fit the {}-row forward's "
        "~{} MB of scratch", this->id(), parked >> 20, seq, need >> 20));
    return true;
  }
  session()->error(fmt(
      "GenerateVideoStage('{}'): not enough memory for a {}-row forward -- "
      "it needs ~{} MB of scratch and there is ~{} MB of GPU working set / "
      "~{} MB reclaimable{}. Refusing rather than thrashing: wired Metal "
      "buffers cannot be paged out, so overcommitting here takes the whole "
      "machine down rather than failing this stage. Use a smaller "
      "height/width/frames, or free another model first",
      this->id(), seq, need >> 20, mb.headroom >> 20,
      mb.available_physical >> 20,
      parked > 0 ? fmt(" after parking ~{} MB", parked >> 20)()
                 : std::string()));
  return false;
}

// The minimax-h3 denoise: one packed sequence carrying both modalities,
// so this produces two latents where the Wan path produces one.
bool
GenerateVideoStage::run_h3_(const void* cond, int text_rows, const float* ref,
                            int ref_frames,
                            std::vector<float>* video_out,
                            std::vector<int>* video_shape,
                            std::vector<float>* audio_out,
                            std::vector<int>* audio_shape)
{
  auto* mc = session()->services()->metal_compute();
  if (mc == nullptr || !_h3_dit) { return false; }
  const auto& c = _h3_cfg;

  // Latent geometry. The video VAE compresses 16x spatially and 4x in
  // time with the causal round-UP the encoder does, so the frame count
  // is not a plain division.
  const int lh = _height / 16, lw = _width / 16;
  // The video VAE encodes in 17-frame clips that yield 5 latent frames
  // each and then drops 3, so a request has to be snapped UP to a length
  // the chunking can actually produce. These two are the released
  // checkpoint's `vae_clip_length` / tokens-per-chunk; a checkpoint that
  // changed them would want them read from video_vae/config.json rather
  // than assumed here.
  constexpr int kFramesPerChunk = 17, kLatentsPerChunk = 5;
  const int aligned =
      h3::align_num_frames(_frames, kFramesPerChunk, kLatentsPerChunk);
  const int lt =
      h3::video_latent_num_frames(aligned, kFramesPerChunk, kLatentsPerChunk);
  if (lh <= 0 || lw <= 0 || lt <= 0) {
    session()->warn(fmt(
        "GenerateVideoStage('{}'): {}x{}x{} does not give a usable "
        "MiniMax-H3 latent grid", this->id(), _width, _height, _frames));
    return false;
  }
  // Audio latents are counted from the VIDEO frames and fps, so the two
  // modalities stay the same duration by construction. `audio_seconds`
  // overrides that for a clip whose audio is deliberately a different
  // length.
  const int alat =
      _audio_seconds > 0.0
          ? (int)(_audio_seconds * (double)h3::kAudioLatentsPerSecond + 0.5)
          : h3::audio_latent_num_frames(aligned, _fps);

  h3::PackedLayout L;
  const std::vector<int> tags((std::size_t)text_rows, h3::kTextTag);
  // One latent frame per anchor. No ref => text-to-video-and-audio; one
  // => a first-frame anchor; two => first AND last, which is what the
  // FL2VA partition is named for.
  std::vector<h3::Anchor> anchors;
  if (ref != nullptr && ref_frames > 0) {
    anchors.push_back(h3::Anchor::kFirst);
    if (ref_frames >= 2) { anchors.push_back(h3::Anchor::kLast); }
  }
  // h3::kAudioChannels is the STEREO count (2) -- how many soundtrack
  // channels the audio rows are packed channel-major over. It is NOT
  // `c.audio_channels`, which is the 32-wide audio LATENT that each of
  // those rows carries. Passing the latter here packs 16x the audio rows
  // the model was trained with; every one of them is a real row that the
  // 33B forward pays for, and the result still decodes, just as a
  // soundtrack whose two "channels" are slices of one 32-way split.
  if (!h3::build_packed_sequence(tags, lt, lh, lw, alat, c.patch_h, c.patch_w,
                                 h3::kAudioChannels, anchors, &L)) {
    session()->warn(fmt(
        "GenerateVideoStage('{}'): could not pack a {}x{}x{} latent with {} "
        "audio latents", this->id(), lt, lh, lw, alat));
    return false;
  }
  // The sequence length is known now and nothing large has been allocated
  // yet, which is the only moment a preflight is worth anything.
  if (!preflight_h3_scratch_(L.seq_len, text_rows)) { return false; }

  const int PE = c.video_patch_elems();
  const int AC = c.audio_channels;
  const int vrows = (int)L.video_indices.size();
  std::vector<float> vid((std::size_t)vrows * PE);
  std::vector<float> aud((std::size_t)L.num_audio_rows * AC);
  {
    std::mt19937_64 rng(_seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : vid) { v = nd(rng); }
    for (auto& v : aud) { v = nd(rng); }
  }
  // Diagnostic: take the INITIAL NOISE from files instead of the RNG.
  // A reference implementation draws from torch's generator, not from
  // mt19937_64, so two runs of the same prompt are different SAMPLES and
  // can only be compared by eye. Feeding both the same noise makes the
  // comparison elementwise. Files are raw f32 already in ROW layout:
  // video [num_video_indices, PE], audio [num_audio_rows, AC].
  {
    auto slurp = [&](const char* env, std::vector<float>& dst,
                     const char* what) {
      const char* p = std::getenv(env);
      if (p == nullptr) { return; }
      std::FILE* f = std::fopen(p, "rb");
      if (f == nullptr) {
        session()->warn(fmt("GenerateVideoStage('{}'): cannot open {} noise "
                            "'{}'", this->id(), what, p));
        return;
      }
      const std::size_t n = std::fread(dst.data(), 4, dst.size(), f);
      std::fclose(f);
      if (n != dst.size()) {
        session()->warn(fmt("GenerateVideoStage('{}'): {} noise '{}' has {} "
                            "floats, expected {}", this->id(), what, p, n,
                            dst.size()));
        return;
      }
      session()->info(fmt("GenerateVideoStage('{}'): loaded {} initial noise "
                          "({} floats) from {}", this->id(), what, n, p));
    };
    slurp("VPIPE_H3_NOISE_VID", vid, "video");
    slurp("VPIPE_H3_NOISE_AUD", aud, "audio");
  }

  // Patchify the anchors into the CONDITIONING rows, which lead the
  // video block one whole latent frame per anchor and in the same cell
  // order the generated frames use. The denoise loop never writes these
  // rows, so what is put here is what the model sees at every step.
  if (!anchors.empty()) {
    const int gh0 = lh / c.patch_h, gw0 = lw / c.patch_w;
    const int rows_per_frame = gh0 * gw0;
    const std::size_t rplane = (std::size_t)lh * lw;
    const int ZCr = c.video_channels;
    for (std::size_t k = 0; k < anchors.size(); ++k) {
      for (int cell = 0; cell < rows_per_frame; ++cell) {
        float* row = vid.data() +
                     ((std::size_t)k * rows_per_frame + cell) * PE;
        const int by = (cell / gw0) * c.patch_h;
        const int bx = (cell % gw0) * c.patch_w;
        for (int ch = 0; ch < ZCr; ++ch) {
          for (int y = 0; y < c.patch_h; ++y) {
            for (int x = 0; x < c.patch_w; ++x) {
              row[((std::size_t)ch * c.patch_h + y) * c.patch_w + x] =
                  ref[((std::size_t)ch * ref_frames + (int)k) * rplane +
                      (std::size_t)(by + y) * lw + (bx + x)];
            }
          }
        }
      }
    }
  }

  // Diagnostic: write the text conditioning the DiT is about to read to
  // a raw f32 file, so the REFERENCE pipeline can be driven with the
  // exact same rows. It is the one DiT input with no golden of its own
  // (the encoder goldens stop at tap 3), and feeding it to the reference
  // separates "our conditioning is wrong" from "everything downstream
  // is wrong" without needing a 32B reference encoder to fit in RAM.
  if (const char* dp = std::getenv("VPIPE_H3_COND_DUMP")) {
    std::FILE* f = std::fopen(dp, "wb");
    if (f != nullptr) {
      const auto* d = reinterpret_cast<const std::uint16_t*>(cond);
      const std::size_t n = (std::size_t)text_rows * c.text_dim;
      std::vector<float> v(n);
      for (std::size_t k = 0; k < n; ++k) {
        const std::uint32_t u = (std::uint32_t)d[k] << 16;
        std::memcpy(&v[k], &u, 4);
      }
      std::fwrite(v.data(), 4, n, f);
      std::fclose(f);
      session()->info(fmt("GenerateVideoStage('{}'): dumped {} x {} text "
                          "conditioning rows to {}", this->id(), text_rows,
                          c.text_dim, dp));
    }
  }
  metal_compute::SharedBuffer tb =
      mc->make_shared_buffer((std::size_t)text_rows * c.text_dim * 2);
  if (tb.empty()) { return false; }
  std::memcpy(tb.contents(), cond,
              (std::size_t)text_rows * c.text_dim * 2);

  genai::DenoiseRequest req;
  req.dit    = _h3_dit.get();
  req.layout = &L;
  req.text   = &tb;
  req.video  = vid.data();
  req.audio  = aud.data();
  req.num_steps   = _steps;
  req.video_shift = _video_shift;
  req.audio_shift = _audio_shift;
  req.condition_timestep = (float)_cond_timestep;
  UiProgress bar = session()->open_progress("denoise");
  // Block-granular, like the image DiTs. A step here is one forward of a
  // 33B stack over a ~19k-row sequence -- tens of seconds at the model's
  // own geometry -- so a step-granular bar sits still for the entire time
  // anything is happening. H3 is guidance-distilled, so exactly ONE
  // forward per step.
  DenoiseProgress prog(&bar, _steps, /*forwards_per_step=*/1);
  ScopedBlockProgress<genai::MetalMiniMaxH3Transformer> hook(_h3_dit.get(),
                                                             prog);
  req.progress = [&](int step, int total) {
    // `total` is the count the SCHEDULER settled on, which is not
    // `_steps`: that is a sigma grid including the terminal zero, and the
    // shift can collapse duplicates on top. Adopting it is what makes the
    // bar finish at 100% instead of at (steps-1)/steps.
    prog.set_steps(total);
    // `step` is 1-based on entry here; end_step takes the 0-based index
    // it just finished, and re-syncs the bar to the exact boundary.
    prog.end_step(step - 1);
    return true;
  };
  std::string derr;
  const bool ok = genai::denoise(req, &derr);
  bar.finish();
  if (!ok) {
    session()->warn(fmt("GenerateVideoStage('{}'): {}", this->id(), derr));
    return false;
  }

  // Unpatchify the GENERATED video rows back to a [z, T, lh, lw] grid.
  // Each row is one (1, patch_h, patch_w) cell of z_channels, and the
  // conditioning rows lead the block, so the generated ones start at
  // num_condition_rows.
  const int ZC = c.video_channels;
  const int ph = c.patch_h, pw = c.patch_w;
  const int gh = lh / ph, gw = lw / pw;
  video_out->assign((std::size_t)ZC * lt * lh * lw, 0.0f);
  *video_shape = {ZC, lt, lh, lw};
  const std::size_t plane = (std::size_t)lh * lw;
  for (int r = 0; r < L.num_video_rows; ++r) {
    const float* row = vid.data() +
                       ((std::size_t)L.num_condition_rows + r) * PE;
    const int cell = r % (gh * gw);
    const int t    = r / (gh * gw);
    const int by   = (cell / gw) * ph, bx = (cell % gw) * pw;
    for (int ch = 0; ch < ZC; ++ch) {
      for (int y = 0; y < ph; ++y) {
        for (int x = 0; x < pw; ++x) {
          (*video_out)[(std::size_t)ch * lt * plane + (std::size_t)t * plane +
                       (std::size_t)(by + y) * lw + (bx + x)] =
              row[((std::size_t)ch * ph + y) * pw + x];
        }
      }
    }
  }
  // Diagnostic: the FINAL video latent, [z, T, lh, lw] f32, so it can be
  // compared cell-by-cell against the reference's. Spatial coherence is
  // a property of the LATENT, so this settles "is the tile grid already
  // in the latent" without involving the VAE at all.
  if (const char* lp = std::getenv("VPIPE_H3_LATENT_DUMP")) {
    std::FILE* f = std::fopen(lp, "wb");
    if (f != nullptr) {
      std::fwrite(video_out->data(), 4, video_out->size(), f);
      std::fclose(f);
      session()->info(fmt("GenerateVideoStage('{}'): dumped latent "
                          "[{}, {}, {}, {}] to {}", this->id(), ZC, lt, lh,
                          lw, lp));
    }
  }

  // The audio rows come out of the packed sequence as [stereo * alat, AC]
  // -- channel-major in time, each row a 32-wide latent. Transpose to the
  // [stereo, AC, alat] the audio VAE decodes, so the packed-sequence
  // layout stops at this stage's boundary the same way the video
  // unpatchify above does.
  audio_out->assign((std::size_t)h3::kAudioChannels * AC * alat, 0.0f);
  for (int ch = 0; ch < h3::kAudioChannels; ++ch) {
    for (int i = 0; i < alat; ++i) {
      const float* row = aud.data() + (std::size_t)(ch * alat + i) * AC;
      for (int k = 0; k < AC; ++k) {
        (*audio_out)[((std::size_t)ch * AC + k) * alat + i] = row[k];
      }
    }
  }
  *audio_shape = {h3::kAudioChannels, AC, alat};
  return true;
}

Job
GenerateVideoStage::process(RuntimeContext& ctx)
{
  if (!_model_latched && ctx.num_iports() > kModelPort &&
      ctx.iport_connected(kModelPort)) {
    auto mb = co_await ctx.read(kModelPort);
    _model_latched = true;
    if (const auto* mfd =
            mb ? dynamic_cast<const FlexDataPayload*>(mb.get()) : nullptr) {
      if (apply_model_select_beat(mfd->data, _hf_dir)) { resolve_config_(); }
    }
  }
  if (!_sampler_latched && ctx.num_iports() > kSamplerPort &&
      ctx.iport_connected(kSamplerPort)) {
    auto sb = co_await ctx.read(kSamplerPort);
    _sampler_latched = true;
    if (const auto* sfd =
            sb ? dynamic_cast<const FlexDataPayload*>(sb.get()) : nullptr) {
      std::string serr;
      _sampler_spec = genai::FlowSamplerSpec::from_flex(sfd->data, &serr);
      if (!serr.empty()) {
        session()->warn(fmt("GenerateVideoStage('{}'): sampler spec: {}",
                            this->id(), serr));
      }
    }
  }
  if (!_scheduler_latched && ctx.num_iports() > kSchedPort &&
      ctx.iport_connected(kSchedPort)) {
    auto cb = co_await ctx.read(kSchedPort);
    _scheduler_latched = true;
    if (const auto* cfd =
            cb ? dynamic_cast<const FlexDataPayload*>(cb.get()) : nullptr) {
      std::string cerr;
      _scheduler_spec = genai::FlowSchedulerSpec::from_flex(cfd->data, &cerr);
      if (!cerr.empty()) {
        session()->warn(fmt("GenerateVideoStage('{}'): scheduler spec: {}",
                            this->id(), cerr));
      }
    }
  }

  auto in = co_await ctx.read(0);
  if (!in) { ctx.signal_done(); co_return; }
  const auto* cond = dynamic_cast<const TensorBeatPayload*>(in.get());
  if (cond == nullptr || cond->shape.size() != 2) {
    session()->warn(fmt(
        "GenerateVideoStage('{}'): expected a [text_seq, {}] conditioning "
        "TensorBeat, got {}; skipping", this->id(), _cfg.text_dim,
        in->describe()));
    co_return;
  }
  // The negative is emitted BEFORE the positive by the conditioner, so by
  // the time the positive lands its pair is already queued -- a non-blocking
  // poll rather than a read that could deadlock when there is no negative.
  std::unique_ptr<BeatPayloadIntf> negb;
  if (ctx.num_iports() > 1 && ctx.iport_connected(1) && ctx.backlog(1) > 0) {
    negb = co_await ctx.read(1);
  }
  const auto* neg =
      negb ? dynamic_cast<const TensorBeatPayload*>(negb.get()) : nullptr;
  std::unique_ptr<BeatPayloadIntf> refb;
  if (ctx.num_iports() > kRefPort && ctx.iport_connected(kRefPort)) {
    refb = co_await ctx.read(kRefPort);
  }
  const auto* ref =
      refb ? dynamic_cast<const TensorBeatPayload*>(refb.get()) : nullptr;
  std::unique_ptr<BeatPayloadIntf> refb1;
  if (ctx.num_iports() > kRefPort1 && ctx.iport_connected(kRefPort1)) {
    refb1 = co_await ctx.read(kRefPort1);
  }
  const auto* ref1 =
      refb1 ? dynamic_cast<const TensorBeatPayload*>(refb1.get()) : nullptr;

  if (!ensure_expert_(0)) {
    session()->warn(fmt(
        "GenerateVideoStage('{}'): no DiT; skipping", this->id()));
    co_return;
  }

  // ---- minimax-h3: one packed sequence, two latents out ---------------
  if (_family == "minimax-h3") {
    if ((int)cond->shape[1] != _h3_cfg.text_dim) {
      session()->warn(fmt(
          "GenerateVideoStage('{}'): MiniMax-H3 wants [text_seq, {}] "
          "conditioning, got [{}, {}]; skipping", this->id(),
          _h3_cfg.text_dim, cond->shape[0], cond->shape[1]));
      co_return;
    }
    if (neg != nullptr) {
      // Not an error -- a graph wired for Wan should still run -- but
      // silently dropping it would misrepresent what ran.
      session()->log_debug(fmt(
          "GenerateVideoStage('{}'): MiniMax-H3 is guidance-distilled; the "
          "negative conditioning on iport1 is ignored", this->id()));
    }
    std::vector<float> vlat, alat_out;
    std::vector<int>   vshape, ashape;
    // The keyframe anchor arrives on the SAME port Wan's i2v latent
    // does, from a vae-encode over the keyframe image -- so a graph
    // changes checkpoints without being rewired.
    const int lh0 = _height / 16, lw0 = _width / 16;
    auto anchor_ok = [&](const TensorBeatPayload* t) {
      return t != nullptr && t->shape.size() == 4 &&
             (int)t->shape[0] == _h3_cfg.video_channels &&
             (int)t->shape[2] == lh0 && (int)t->shape[3] == lw0;
    };
    const float* refp = nullptr;
    int ref_frames = 0;
    if (ref != nullptr) {
      if (anchor_ok(ref)) {
        refp = ref->as_f32();
        ref_frames = (int)ref->shape[1];
      } else {
        session()->warn(fmt(
            "GenerateVideoStage('{}'): keyframe latent does not match "
            "[{}, n, {}, {}]; generating without an anchor", this->id(),
            _h3_cfg.video_channels, lh0, lw0));
      }
    }
    // The LAST-frame anchor arrives as its own beat, because the stage
    // that makes one encodes ONE image -- so first-and-last is two
    // vae-encodes, not one that somehow emits two latent frames. They
    // are concatenated here, on the frame axis, into the [z, k, lh, lw]
    // block run_h3_ patchifies (anchor k -> conditioning frame k).
    std::vector<float> stacked;
    if (ref1 != nullptr) {
      if (refp == nullptr) {
        // Dropping it silently would run a plain t2v while the graph
        // says otherwise. L2V is not a partition this model was
        // trained for, so the honest move is to name what was ignored.
        session()->warn(fmt(
            "GenerateVideoStage('{}'): a LAST-frame anchor on iport{} needs "
            "a FIRST-frame anchor on iport{} too; ignoring it", this->id(),
            kRefPort1, kRefPort));
      } else if (!anchor_ok(ref1)) {
        session()->warn(fmt(
            "GenerateVideoStage('{}'): last-frame keyframe latent does not "
            "match [{}, n, {}, {}]; generating with the first frame only",
            this->id(), _h3_cfg.video_channels, lh0, lw0));
      } else {
        const int ZC = _h3_cfg.video_channels;
        const int n1 = (int)ref1->shape[1];
        const std::size_t plane = (std::size_t)lh0 * lw0;
        const int total = ref_frames + n1;
        stacked.resize((std::size_t)ZC * total * plane);
        const float* a = refp;
        const float* b = ref1->as_f32();
        for (int c = 0; c < ZC; ++c) {
          float* dst = stacked.data() + (std::size_t)c * total * plane;
          std::memcpy(dst, a + (std::size_t)c * ref_frames * plane,
                      (std::size_t)ref_frames * plane * sizeof(float));
          std::memcpy(dst + (std::size_t)ref_frames * plane,
                      b + (std::size_t)c * n1 * plane,
                      (std::size_t)n1 * plane * sizeof(float));
        }
        refp = stacked.data();
        ref_frames = total;
      }
    }
    // Cooperative stop, polled per BLOCK inside the DiT. A step is one
    // pass over 50 blocks of a 33B stack, so stopping only between steps
    // leaves a stop request hanging for tens of seconds; this makes it
    // land in about a block. Cleared afterwards so a stale `ctx` is never
    // captured past this generation -- the lambda holds a reference.
    auto stopping = [&ctx]() { return ctx.stop_requested(); };
    if (_h3_dit) { _h3_dit->set_stream_stop(stopping); }
    const bool ok_h3 =
        run_h3_(cond->data.data(), (int)cond->shape[0], refp, ref_frames,
                &vlat, &vshape, &alat_out, &ashape);
    if (_h3_dit) { _h3_dit->set_stream_stop({}); }
    // What the adaptive residency actually reached. This is the number
    // that says whether the machine got used: with 0 pinned at load, every
    // resident block here was earned at runtime out of free headroom, and
    // the count is what a "why was this slow" question needs first.
    if (_h3_dit && _h3_dit->streaming()) {
      session()->info(fmt(
          "GenerateVideoStage('{}'): DiT residency ended at {} of {} blocks "
          "({} MB), {} pinned at load", this->id(),
          _h3_dit->resident_block_count() + _h3_dit->pinned_blocks(),
          _h3_cfg.n_layers,
          (_h3_dit->resident_block_bytes()) >> 20,
          _h3_dit->pinned_blocks()));
    }
    if (!ok_h3) {
      if (ctx.stop_requested()) {
        session()->info(fmt(
            "GenerateVideoStage('{}'): stopped mid-denoise; dropping the "
            "partial clip", this->id()));
      }
      co_return;
    }
    if (_unload_idle) { _h3_dit.reset(); }

    auto vout = std::make_unique<TensorBeatPayload>();
    vout->dtype = TensorBeat::DType::F32;
    vout->shape = {vshape[0], vshape[1], vshape[2], vshape[3]};
    vout->resize_contiguous(vlat.size());
    std::memcpy(vout->as_f32(), vlat.data(), vlat.size() * sizeof(float));
    {
      FlexData sb = FlexData::make_object();
      sb.as_object().insert_or_assign("fps", FlexData::make_real(_fps));
      sb.as_object().insert_or_assign(
          "frames", FlexData::make_int((std::int64_t)_frames));
      vout->sideband = std::move(sb);
    }
    ++_emitted;
    session()->info(fmt(
        "GenerateVideoStage('{}'): emitted MiniMax-H3 latents #{} -- video "
        "[{}, {}, {}, {}] + audio [{}, {}, {}]", this->id(), _emitted,
        vshape[0], vshape[1], vshape[2], vshape[3], ashape[0], ashape[1],
        ashape[2]));
    co_await ctx.write(0, std::move(vout));

    // The audio half. Written only when the port is connected: the audio
    // VAE is a separate decode stage, and a graph that only wants video
    // should not have to wire a sink for a beat it will not read.
    if (ctx.num_oports() > 1 && !alat_out.empty()) {
      auto aout = std::make_unique<TensorBeatPayload>();
      aout->dtype = TensorBeat::DType::F32;
      aout->shape = {ashape[0], ashape[1], ashape[2]};
      aout->resize_contiguous(alat_out.size());
      std::memcpy(aout->as_f32(), alat_out.data(),
                  alat_out.size() * sizeof(float));
      FlexData sb = FlexData::make_object();
      sb.as_object().insert_or_assign(
          "latents_per_second",
          FlexData::make_int((std::int64_t)h3::kAudioLatentsPerSecond));
      aout->sideband = std::move(sb);
      co_await ctx.write(1, std::move(aout));
    }
    co_return;
  }

  if (ref1 != nullptr) {
    // Wan's i2v conditioning is ONE clip-shaped tensor that already spans
    // every frame, so there is no second anchor to take. Debug rather than
    // warn: a graph built for h3 should still run here.
    session()->log_debug(fmt(
        "GenerateVideoStage('{}'): iport{} (last-frame anchor) is minimax-h3 "
        "only; ignoring it", this->id(), kRefPort1));
  }

  auto* mc = session()->services()->metal_compute();
  const int ZC = 16;                       // the VAE's latent channels
  const int T  = latent_frames();
  const int h8 = _height / 8, w8 = _width / 8;
  const int text_seq = (int)cond->shape[0];
  const std::size_t nlat = (std::size_t)ZC * T * h8 * w8;
  const std::size_t plane = (std::size_t)h8 * w8;

  // ---- image-to-video conditioning channels -------------------------
  // in_channels 36 = 16 noise + 4 first-frame mask + 16 image latent. The
  // mask is 4 channels rather than 1 because the VAE's 4x temporal
  // compression folds four video frames into one latent frame: latent
  // frame 0 covers video frame 0 alone (the conditioning image) plus three
  // repeats of it, and every later latent frame covers four blank ones.
  const bool i2v = _cfg.in_channels > ZC;
  std::vector<float> cond_ch;              // [20, T, h8, w8] = mask + image
  if (i2v) {
    if (ref == nullptr || ref->shape.size() != 4 ||
        (int)ref->shape[0] != ZC || (int)ref->shape[1] != T ||
        (int)ref->shape[2] != h8 || (int)ref->shape[3] != w8) {
      session()->warn(fmt(
          "GenerateVideoStage('{}'): this checkpoint has in_channels {}, so it "
          "is image-to-video and needs a [{}, {}, {}, {}] conditioning latent "
          "on ref_latent0 (from a vae-encode over the conditioning image); "
          "skipping", this->id(), _cfg.in_channels, ZC, T, h8, w8));
      co_return;
    }
    cond_ch.assign((std::size_t)20 * T * plane, 0.0f);
    // Mask channel c of latent frame t is 1 exactly where the video frame
    // it stands for is the conditioning image: (t=0, c=0..3) covers video
    // frame 0 four times over, and nothing else does.
    for (int c = 0; c < 4; ++c) {
      float* m = cond_ch.data() + ((std::size_t)c * T) * plane;
      for (std::size_t i = 0; i < plane; ++i) { m[i] = 1.0f; }
    }
    std::memcpy(cond_ch.data() + (std::size_t)4 * T * plane, ref->as_f32(),
                nlat * sizeof(float));
  }

  // ---- schedule + initial noise --------------------------------------
  genai::FlowSchedulerSpec sched = _scheduler_spec;
  if (sched.steps <= 0) { sched.steps = _steps; }
  genai::FlowSampler sampler(_sampler_spec, sched);
  sampler.reset();
  const std::vector<double>& sigmas = sampler.sigmas();

  std::vector<float> x(nlat);
  {
    std::mt19937_64 rng(_seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : x) { v = nd(rng); }
  }

  // Text conditioning is projected by the EXPERT's own text_embedder, so it
  // is re-projected on a swap rather than computed once for the clip.
  SharedBuffer tpos, tneg;
  int expert_for_text = -1;
  auto upload_bf16 = [&](const TensorBeatPayload& b) {
    const std::size_t n = (std::size_t)b.shape[0] * b.shape[1];
    SharedBuffer s = mc->make_shared_buffer(n * 2);
    if (!s.empty()) { std::memcpy(s.contents(), b.as_bf16(), n * 2); }
    return s;
  };
  const SharedBuffer cond_raw = upload_bf16(*cond);
  const SharedBuffer neg_raw = neg != nullptr ? upload_bf16(*neg)
                                              : SharedBuffer{};
  const bool cfg_on = neg != nullptr;

  // The DiT input buffer, rebuilt per evaluation from the sampler's
  // candidate latent plus the constant conditioning channels.
  SharedBuffer dit_in =
      mc->make_shared_buffer((std::size_t)_cfg.in_channels * T * plane * 2);
  if (dit_in.empty()) {
    session()->warn(fmt("GenerateVideoStage('{}'): latent allocation failed",
                        this->id()));
    co_return;
  }

  bool failed = false;
  // Block-granular denoise progress, as the image DiTs report it. A 14B
  // step at 720p is seconds of silence, and this loop previously reported
  // nothing at all -- only a debug log.
  //
  // Guidance runs the DiT a second time per step, so the bar is sized for
  // two forwards when a negative prompt is present. When an expert's
  // guidance is exactly 1.0 that second forward is skipped and the step
  // fills only part way; end_step re-syncs at the boundary, so it is a
  // pacing detail rather than a wrong number.
  UiProgress bar = session()->open_progress("denoise");
  DenoiseProgress prog(&bar, sched.steps, cfg_on ? 2 : 1);
  ScopedBlockProgress<genai::MetalWanTransformer> hook(_dit.get(), prog);
  auto denoise = [&](const std::vector<float>& xin,
                     double sigma) -> std::vector<float> {
    std::vector<float> out(nlat, 0.0f);
    if (failed) { return out; }
    // Two experts, one boundary, one crossing: the schedule descends, so
    // this swaps at most once per clip.
    const int want = (_boundary > 0.0 && sigma < _boundary) ? 1 : 0;
    if (!ensure_expert_(want)) { failed = true; return out; }
    // Crossing the boundary rebuilds _dit, which takes the old hook with
    // it; re-arm on whatever is loaded now. No-op when nothing swapped.
    hook.rearm(_dit.get());
    if (_expert != expert_for_text) {
      std::string terr;
      tpos = _dit->encode_text(cond_raw, text_seq, &terr);
      tneg = cfg_on ? _dit->encode_text(neg_raw, (int)neg->shape[0], &terr)
                    : SharedBuffer{};
      if (tpos.empty()) {
        session()->warn(fmt("GenerateVideoStage('{}'): text projection: {}",
                            this->id(), terr));
        failed = true;
        return out;
      }
      expert_for_text = _expert;
    }
    auto* dp = static_cast<std::uint16_t*>(dit_in.contents());
    auto put = [&](std::size_t i, float v) {
      std::uint32_t u;
      std::memcpy(&u, &v, 4);
      dp[i] = (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    };
    for (std::size_t i = 0; i < nlat; ++i) { put(i, xin[i]); }
    for (std::size_t i = 0; i < cond_ch.size(); ++i) {
      put(nlat + i, cond_ch[i]);
    }
    // The scheduler's timestep is sigma on its native 0..num_train scale,
    // which is what the DiT's sinusoidal embedding was trained against.
    const float ts = (float)(sigma * (double)sched.num_train);
    const double g = (_expert == 1) ? _guidance_2 : _guidance;
    std::string derr;
    SharedBuffer vp =
        _dit->forward(dit_in, T, h8, w8, tpos, text_seq, ts, &derr);
    if (vp.empty()) {
      session()->warn(fmt("GenerateVideoStage('{}'): DiT forward: {}",
                          this->id(), derr));
      failed = true;
      return out;
    }
    auto bf16 = [](std::uint16_t b) {
      const std::uint32_t u = (std::uint32_t)b << 16;
      float f;
      std::memcpy(&f, &u, 4);
      return f;
    };
    const auto* pp = static_cast<const std::uint16_t*>(vp.contents());
    if (cfg_on && g != 1.0) {
      SharedBuffer vn = _dit->forward(dit_in, T, h8, w8, tneg,
                                      (int)neg->shape[0], ts, &derr);
      if (vn.empty()) {
        session()->warn(fmt("GenerateVideoStage('{}'): negative forward: {}",
                            this->id(), derr));
        failed = true;
        return out;
      }
      const auto* np = static_cast<const std::uint16_t*>(vn.contents());
      for (std::size_t i = 0; i < nlat; ++i) {
        const float u = bf16(np[i]);
        out[i] = u + (float)g * (bf16(pp[i]) - u);
      }
    } else {
      for (std::size_t i = 0; i < nlat; ++i) { out[i] = bf16(pp[i]); }
    }
    return out;
  };

  {
    PerfAuxScope _perf(session(), kPerfLaneLLM, kGvidLlmDit, kPerfLlmDitBegin,
                       (std::uint64_t)T * (h8 / 2) * (w8 / 2));
    for (int i = 0; i < sched.steps && !failed; ++i) {
      session()->log_debug(fmt(
          "GenerateVideoStage('{}'): step {}/{} sigma {:.4f} ({}-noise "
          "expert)", this->id(), i + 1, sched.steps, sigmas[(std::size_t)i],
          (_boundary > 0.0 && sigmas[(std::size_t)i] < _boundary) ? "low"
                                                                  : "high"));
      sampler.step(i, x, denoise);
      prog.end_step(i);
    }
  }
  bar.finish();
  if (failed) { co_return; }

  if (_unload_idle) {
    _dit.reset();
    _expert = -1;
    expert_for_text = -1;
  }

  auto out = std::make_unique<TensorBeatPayload>();
  out->dtype = TensorBeat::DType::F32;
  out->shape = {ZC, T, h8, w8};
  out->resize_contiguous(nlat);
  std::memcpy(out->as_f32(), x.data(), nlat * sizeof(float));
  FlexData sb = FlexData::make_object();
  sb.as_object().insert_or_assign("fps", FlexData::make_real(_fps));
  sb.as_object().insert_or_assign("frames",
                                  FlexData::make_int((std::int64_t)_frames));
  out->sideband = std::move(sb);
  ++_emitted;
  session()->info(fmt(
      "GenerateVideoStage('{}'): emitted latent video #{} [{}, {}, {}, {}] "
      "({} frames at {}x{}, {:.3f} fps)", this->id(), _emitted, ZC, T, h8, w8,
      _frames, _width, _height, _fps));
  co_await ctx.write(0, std::move(out));
}

#else   // !VPIPE_BUILD_APPLE_SILICON

Job
GenerateVideoStage::initialize(RuntimeContext& ctx)
{
  (void)ctx;
  if (session()) {
    session()->error(fmt(
        "GenerateVideoStage('{}'): built without VPIPE_BUILD_APPLE_SILICON; "
        "the metal Wan DiT is unavailable, the stage is inert", this->id()));
  }
  co_return;
}

Job
GenerateVideoStage::process(RuntimeContext& ctx)
{
  auto in = co_await ctx.read(0);
  (void)in;
  ctx.signal_done();
  co_return;
}

#endif  // VPIPE_BUILD_APPLE_SILICON

VPIPE_REGISTER_STAGE(GenerateVideoStage)
VPIPE_REGISTER_SPEC(GenerateVideoStage, kSpec)

}  // namespace vpipe

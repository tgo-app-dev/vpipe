#include "stages/generate-image-stage.h"

#include "stages/model-memory.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "generative-models/shared/dit-block-progress.h"
#include "stages/denoise-progress.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/weight-set.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "stages/model-config-source.h"
#include "stages/model-registry.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#endif

#include <type_traits>
#include <sys/sysctl.h>

#include <chrono>
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

// The Qwen-Image-Edit DiT runs bf16 (its residual stream exceeds f16's 65504);
// its packed-latent + velocity buffers are raw bf16, not _Float16.
inline std::uint16_t
f32_to_bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
inline float
bf16_to_f32_(std::uint16_t b)
{
  std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

const ConfigKey kAttrs[] = {
  {.key = "hf_dir", .type = ConfigType::String, .required = false,
   .doc = "Krea-2-Turbo / FLUX.2 model dir (text_encoder/, transformer/, "
          "tokenizer/); an original or model-quantize'd (self-contained) "
          "pipeline. OPTIONAL: a model-select source on the model iport "
          "overrides it",
   .suggest_db = kModelRegistryDb, .suggest_db_type =
       "krea2,flux2,qwen-image-edit,mage-flow,mage-flow-edit,"
       "boogu-image,boogu-image-edit"},
  {.key = "dit_dir", .type = ConfigType::String, .required = false,
   .doc = "override DiT dir (e.g. a quantized 4/8-bit DiT); else <hf_dir>/transformer",
   .suggest_db = kModelRegistryDb,
   .suggest_db_type =
       "krea2-dit,flux2-dit,qwen-image-edit-dit,mage-flow-dit,"
       "boogu-image-dit"},
  {.key = "strength", .type = ConfigType::Real, .required = false,
   .doc = "img2img strength in [0,1]; 0 (default) = text-to-image from noise "
          "(the init latent arrives on the `latent` iport from vae-encode)"},
  {.key = "height", .type = ConfigType::Int, .required = false,
   .doc = "output height, multiple of 16. Unset TOGETHER with width = infer "
          "from ref_latent0 (iport5) at its family's VAE scale; 256 if there "
          "is no reference either"},
  {.key = "width", .type = ConfigType::Int, .required = false,
   .doc = "output width, multiple of 16. Unset TOGETHER with height = infer "
          "from ref_latent0 (iport5); 256 if there is no reference either"},
  {.key = "steps", .type = ConfigType::Int, .required = false,
   .doc = "turbo sampler steps (default 8)"},
  {.key = "seed", .type = ConfigType::Int, .required = false,
   .doc = "initial-noise RNG seed (default 0)"},
  {.key = "guidance_scale", .type = ConfigType::Real, .required = false,
   .doc = "classifier-free guidance scale; 1 (default) disables CFG. >1 with "
          "a negative prompt on iport1 runs a 2nd DiT pass per step"},
  {.key = "init_latents", .type = ConfigType::String, .required = false,
   .doc = "debug: raw f32 packed initial latents [img_seq, 64] (repro/golden)"},
  {.key = "i8_gemm", .type = ConfigType::Bool, .required = false,
   .doc = "accelerated mode (LOSSY): dynamic-int8 GEMMs for the DiT's big "
          "block matmuls, ~2x their f16 rate at int8 quality; IGNORED "
          "without NAX matmul2d (matrix-core GPU + kernels). Default "
          "false; env VPIPE_I8_GEMM overrides"},
};

// The keys that MOVED to the per-family config stages. Named so a
// pipeline written against the old union says what to do instead of
// silently running the defaults -- an unknown key is not an error
// anywhere in this runtime, which is what makes a quiet removal the
// wrong way to do this. `klein_kv` is the one that matters most: it
// selects a RECIPE, so ignoring it does not degrade the image, it makes
// a different one.
struct MovedKey { const char* key; const char* to; };
constexpr MovedKey kMovedKeys[] = {
  {"klein_kv",      "flux2-model-config"},
  {"no_watermark",  "mage-flow-model-config"},
  {"watermark_key", "mage-flow-model-config"},
};
const PortSpec kIports[] = {
  {.name = "conditioning", .doc = "conditioning tensor from a diffusion-"
                                  "conditioner stage (family-shaped + typed)",
   .type = &typeid(TensorBeatPayload),
   .tags = "conditioning", .clock_group = 0},
  {.name = "neg_conditioning", .doc = "OPTIONAL negative conditioning (the "
                                      "conditioner's oport1) for classifier-free "
                                      "guidance",
   .type = &typeid(TensorBeatPayload),
   .tags = "conditioning", .clock_group = 0},
  {.name = "model", .doc = "OPTIONAL shared model reference from a model-select "
                           "source; overrides the hf_dir config",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
  {.name = "sampler",
   .doc = "OPTIONAL sampler spec FlexData (diffusion-sampler-select)",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
  {.name = "scheduler", .doc = "OPTIONAL scheduler spec FlexData (scheduler-select)",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
  {.name = "ref_latent0", .doc = "OPTIONAL reference latent 0 (channel-first "
                                 "f32 from vae-encode); FLUX.2/QIE conditioning "
                                 "/ Krea-2 img2img init",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
  {.name = "ref_latent1", .doc = "OPTIONAL reference latent 1 (channel-first "
                                 "f32 from vae-encode); FLUX.2/QIE 2nd reference "
                                 "(distinct RoPE position); ignored by Krea-2",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
  {.name = "model_config",
   .doc = "OPTIONAL model-specific parameters from the resident family's own "
          "config source -- flux2-model-config (the klein-kv recipe) or "
          "mage-flow-model-config (the provenance watermark). Passed to that "
          "family's own params struct UNREAD, so this stage carries none of "
          "the knobs. Wiring it DEFERS the DiT load to the first beat, "
          "because a recipe like klein_kv has to be known before the weights "
          "are read",
   .type = &typeid(FlexDataPayload),
   .tags = "model-config", .clock_group = 0},
};

// The model iport (a model-select source) overrides hf_dir. Inserted after the
// conditioning + neg_conditioning inputs, so it is iport2; sampler / scheduler /
// ref_latent0 / ref_latent1 follow at iports 3 / 4 / 5 / 6. (Referenced only
// from the Apple-gated code; maybe_unused for the inert non-Apple build.)
[[maybe_unused]] constexpr unsigned kModelPort = 2;
[[maybe_unused]] constexpr unsigned kModelCfgPort = 7;
const PortSpec kOports[] = {
  {.name = "latent",
   .doc = "f32 latent [z_dim, H/8, W/8] (unpacked, whitened)",
   .type = &typeid(TensorBeatPayload),
   .tags = "latent", .clock_group = 0},
  {.name = "step_latent",
   .doc = "OPTIONAL per-denoise-step latent (one beat per sampler step, SAME "
          "format as `latent`) -- connect a vae-decode here to visualize the "
          "denoising progression for debugging. Only emitted when connected.",
   .type = &typeid(TensorBeatPayload),
   .tags = "latent", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "generate-image",
  .doc       = "Diffusion DiT denoiser: conditioning (from a diffusion-"
               "conditioner stage) -> family MMDiT -> FlowMatchEuler -> latent, "
               "on the metal-compute backend. The denoiser half of the split "
               "(feed vae-decode).",
  .display_name = "Generate Image",
  .category  = StageCategory::Generative,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

GenerateImageStage::GenerateImageStage(const SessionContextIntf* s,
                                   std::string               id,
                                   std::vector<InEdge>       iports,
                                   FlexData                  config)
  : TypedStage<GenerateImageStage>(s, std::move(id), std::move(iports),
                                 std::move(config))
{
  _hf_dir    = attr_str("hf_dir");
  _dit_dir   = attr_str("dit_dir");
  _init_latents = attr_str("init_latents");
  _height = (int)attr_int("height");
  _width  = (int)attr_int("width");
  _steps  = (int)attr_int("steps");
  _strength = attr_real("strength");
  _guidance_scale = attr_real("guidance_scale");
  // Accelerated mode (LOSSY, opt-in): dynamic-int8 GEMMs for the DiT's
  // big block matmuls (~2x their f16 rate on matrix-core GPUs at int8
  // quality). Env VPIPE_I8_GEMM=0|1 overrides.
  //
  // This one STAYS here, unlike klein_kv beside it, and the line
  // between them is worth stating: i8_gemm chooses a KERNEL, so it is
  // a property of the machine and the caller's quality budget, and a
  // family without int8 kernels simply runs f16 -- no worse, just not
  // faster. klein_kv chooses a RECIPE, so on the wrong checkpoint it
  // does not run slower, it runs wrong.
  _i8_gemm = attr_bool("i8_gemm");
  _seed   = (std::uint64_t)attr_int("seed");
  // The family-specific keys are gone from this stage; a pipeline still
  // carrying one gets told where it went. Warning rather than failing
  // because the key is now inert, not wrong -- the graph runs, it just
  // runs on defaults.
  {
    auto o = this->config().as_object();
    for (const auto& mk : kMovedKeys) {
      if (o.contains(mk.key)) {
        session()->warn(fmt(
            "GenerateImageStage('{}'): config key '{}' moved to the '{}' "
            "stage and is IGNORED here; wire one to the model_config iport "
            "or the family runs its defaults", this->id(), mk.key, mk.to));
      }
    }
  }
  // Neither axis configured => infer the size from ref_latent0 when a
  // reference is wired (process()); both stay 0 so the check below is vacuous
  // and the log reads "auto". A PARTIALLY specified size is not half-inferred
  // -- the given axis stands and the other keeps the historical 256 default.
  _infer_size = (_height <= 0 && _width <= 0);
  if (!_infer_size) {
    if (_height <= 0) { _height = 256; }
    if (_width  <= 0) { _width  = 256; }
  }
  if (_steps  <= 0) { _steps  = 8; }
  if (_guidance_scale <= 0.0) { _guidance_scale = 1.0; }   // <=0 => CFG off
  if (_strength < 0.0 || _strength > 1.0) {
    fail_config(fmt(
        "GenerateImageStage('{}'): strength must be in [0,1] (got {})",
        this->id(), _strength));
  }
  // hf_dir is OPTIONAL: a model-select source on the model iport can supply it
  // instead, so "no model at all" is reported at initialize()/process() time
  // (when iport connectivity is known), not here.
  if (_height % 16 != 0 || _width % 16 != 0) {
    fail_config(fmt(
        "GenerateImageStage('{}'): height/width must be multiples of 16 (got "
        "{}x{})", this->id(), _height, _width));
  }
  allocate_oports(spec().oports.size());
#ifdef VPIPE_BUILD_APPLE_SILICON
  _scheduler_spec.steps = _steps;   // config default; port beats override
#endif
}

GenerateImageStage::~GenerateImageStage() = default;

const StageSpec&
GenerateImageStage::spec() const noexcept
{
  return kSpec;
}

void
GenerateImageStage::apply_constant(unsigned iport, const FlexData& beat)
{
  // Pre-launch twin of the runtime latch in process(): the same
  // beat and the same parse, early enough that declare_resources()
  // sees the model. Bookkeeping only -- nothing loads here; the
  // pipeline is not assembled yet (see Stage::apply_constant).
  if (iport != kModelPort) { return; }
  apply_model_select_beat(beat, _hf_dir);
}

#ifdef VPIPE_BUILD_APPLE_SILICON

namespace {

// Output pixels per ref_latent cell, i.e. the factor the paired vae-encode
// applied when it produced ref_latent0 for this family:
//   krea2 / qwen-image-edit   [16, H/8,  W/8 ]   -> 8
//   boogu-image               [16, H/8,  W/8 ]   -> 8
//   flux2 / mage-flow         [C,  H/16, W/16]   -> 16
// (FLUX.2's VAE is 8x but vae-encode emits its 2x2-packed DiT latent, so the
// pixels-per-cell figure is 16 there too.)
int
latent_scale_(const std::string& family)
{
  return (family == "flux2" || family == "mage-flow") ? 16 : 8;
}

// FLUX.2 empirical flow-shift mu (diffusers Flux2Pipeline.compute_empirical_mu):
// the sigma time-shift is resolution- AND step-dependent (a fixed shift washes
// out at other resolutions -> grey). `image_seq_len` is the packed image token
// count (grid_h*grid_w); `num_steps` the sampler steps. vpipe's FlowScheduler
// `shift` field IS this mu (time_shift(sigma, mu, exponential)).
double
flux2_empirical_mu_(int image_seq_len, int num_steps)
{
  const double a1 = 8.73809524e-05, b1 = 1.89833333;
  const double a2 = 0.00016927,     b2 = 0.45666666;
  const double n = (double)image_seq_len;
  if (n > 4300.0) { return a2 * n + b2; }
  const double m_200 = a2 * n + b2;
  const double m_10  = a1 * n + b1;
  const double a = (m_200 - m_10) / 190.0;
  const double b = m_200 - 200.0 * a;
  return a * (double)num_steps + b;
}

// The transformer family from <root>/transformer/config.json `_class_name`:
// "Flux2Transformer2DModel" -> "flux2"; "QwenImageTransformer2DModel" ->
// "qwen-image-edit"; else "krea2".
std::string
t2i_family_(const std::string& transformer_dir)
{
  namespace fs = std::filesystem;
  std::ifstream in(fs::path(transformer_dir) / "config.json");
  if (in) {
    FlexData fd = FlexData::from_json(in);
    if (fd.is_object()) {
      auto obj = fd.as_object();
      if (obj.contains("_class_name")) {
        const std::string cls(obj.at("_class_name").as_string(""));
        if (cls == "Flux2Transformer2DModel") { return "flux2"; }
        if (cls == "QwenImageTransformer2DModel") { return "qwen-image-edit"; }
        if (cls == "MageFlow") { return "mage-flow"; }
        if (cls == "BooguImageTransformer2DModel") { return "boogu-image"; }
      }
    }
  }
  return "krea2";
}

// Physical RAM + component weight bytes now live in stages/model-memory.h so
// the conditioner and the VAE stages size the box the same way this one does.
using model_memory::phys_ram;

// FLUX.2 VAE decode-peak channel count from <root>/vae/config.json. The top
// up-block's FIRST resnet reads block_out_channels[1] (256) at FULL res (the
// upsampled level-2 output) before it reduces to block_out_channels[0] (128),
// so the largest full-res im2col scales with max(block_out[0], block_out[1]) --
// mirror MetalFlux2Vae::decode_peak_bytes, which was widened to that max in the
// 2K uint32-overflow fix (using block_out[0] alone under-modelled the peak 2x,
// so the DiT-reclaim decision could skip freeing and then OOM the vae-decode).
// 128 when unreadable -- the stock AutoencoderKLFlux2 block_out[0] default.
int
flux2_vae_base_(const std::string& root)
{
  namespace fs = std::filesystem;
  std::ifstream in(fs::path(root) / "vae" / "config.json");
  if (in) {
    FlexData fd = FlexData::from_json(in);
    if (fd.is_object()) {
      auto o = fd.as_object();
      if (o.contains("block_out_channels")) {
        FlexData bo = o.at("block_out_channels");
        auto arr = bo.as_array();       // owner `bo` outlives this view
        // block_out_channels parses as a JSON INTEGER array (mode Int), so read
        // each element with as_real() -- as_real_span() only exposes native-real
        // arrays and returns EMPTY here (which had silently pinned this to 128).
        if (arr.size() >= 2) {
          const int b0 = (int)arr[0].as_real(0.0);
          const int b1 = (int)arr[1].as_real(0.0);
          if (b0 > 0 && b1 > 0) { return std::max(b0, b1); }
        }
        if (arr.size() >= 1) {
          const int b0 = (int)arr[0].as_real(0.0);
          if (b0 > 0) { return b0; }
        }
      }
    }
  }
  return 128;
}

// Mage-Flow's static flow-matching shift from <root>/scheduler/
// scheduler_config.json (FlowMatchEulerDiscreteScheduler, shift 6.0,
// use_dynamic_shifting false). 6.0 when unreadable -- the published value and
// the reference's own fallback (pipeline.py _get_scheduler).
double
mage_static_shift_(const std::string& root)
{
  namespace fs = std::filesystem;
  std::ifstream in(fs::path(root) / "scheduler" / "scheduler_config.json");
  if (in) {
    FlexData fd = FlexData::from_json(in);
    if (fd.is_object()) {
      auto o = fd.as_object();
      if (o.contains("shift")) {
        const double s = o.at("shift").as_real(6.0);
        if (s > 0.0) { return s; }
      }
    }
  }
  return 6.0;
}

// The Krea-2 VAE (WanVAE-style, shared with Qwen-Image-Edit) sizes its conv
// stack from `base_dim`, not diffusers' block_out_channels. Read it for the
// decode-peak estimate; fall back to the checkpoint default (96).
int
krea2_vae_base_(const std::string& root)
{
  namespace fs = std::filesystem;
  std::ifstream in(fs::path(root) / "vae" / "config.json");
  if (in) {
    FlexData fd = FlexData::from_json(in);
    if (fd.is_object()) {
      auto o = fd.as_object();
      if (o.contains("base_dim")) {
        const int b = (int)o.at("base_dim").as_real(0.0);
        if (b > 0) { return b; }
      }
    }
  }
  return 96;
}

}  // namespace

void
GenerateImageStage::reset_run_state()
{
  // If a previous run left the weights UNLOADED (the idle-unload
  // policy drops them between beats), let this launch load them again:
  // ensure_loaded_'s once-only guard is per-Stage, not per-launch, so
  // without this the stage stays inert for the whole run. When the
  // weights are still held we deliberately leave the guard set --
  // reloading on top of a resident copy is exactly what doubles peak
  // memory.
  if (!_dit && !_flux2_dit && !_qie_dit && !_mage_dit && !_boogu_dit) {
    _load_attempted = false;
    _dit_unloaded   = false;
  }

  // Per-launch reset: the stage survives a stop/relaunch, and the
  // sources upstream re-emit on every launch. Without this the
  // re-emitted beat is never latched and this stage keeps the previous
  // run's selection.
  _model_latched     = false;
  _sampler_latched   = false;
  _scheduler_latched = false;
  _cfg_latched       = false;
  _model_cfg         = FlexData{};
  // The reference latents latch on FIRST arrival and are cached for
  // every later prompt (`_ref[]`), which is right within a run and
  // wrong across one: a non-empty cache makes the iport5/iport6 read
  // conditional, so on the next launch the stage never consumes the
  // reference beat the vae-encode upstream just produced. That beat
  // then sits on the wire forever (the edge shows a permanent backlog)
  // and the run silently re-uses the PREVIOUS image's reference.
  _ref[0] = RefLatent{};
  _ref[1] = RefLatent{};

}

void
GenerateImageStage::revise_dit_declaration_(const std::string& dit_dir) const
{
  auto* mgr = session() ? session()->services()->generative_model_manager() : nullptr;
  if (mgr == nullptr || dit_dir.empty()) { return; }
  // What the set holds IS the right number for a DiT: unlike the LMs,
  // its retained tensors go through tensor()/derived() and are cached
  // here, while the streamed blocks deliberately are not.
  auto ws = mgr->weight_set(dit_dir);
  if (!ws) { return; }
  const std::size_t held = ws->stats().bytes;
  mgr->revise_declaration(dit_dir, held);
  session()->log_debug(fmt(
      "GenerateImageStage('{}'): streaming DiT keeps {} MB resident; revised "
      "down from its {} MB on disk so peers do not size against weights "
      "that are never there", this->id(), held >> 20,
      model_memory::dir_weights_bytes(dit_dir) >> 20));
}

std::vector<ResourceClaim>
GenerateImageStage::declare_resources() const
{
  if (_hf_dir.empty()) { return {}; }
  namespace fs = std::filesystem;
  const std::string root = resolve_model_dir(session(), _hf_dir);
  const std::string mllm = (fs::path(root) / "mllm").string();
  std::error_code ec;
  const std::string enc = fs::exists(mllm, ec)
                              ? mllm
                              : (fs::path(root) / "text_encoder").string();
  std::vector<ResourceClaim> out = model_memory::weight_claims(
      {(fs::path(root) / "transformer").string(), enc});

  // The DECODE ARENA this stage's output implies.
  //
  // Declared HERE, by the stage that has the geometry, because nothing
  // else in the graph does: vae-decode takes its size from whatever
  // latent arrives, so at planning time it cannot name a number, and
  // sizing itself against peer WEIGHTS is the proxy it has been using.
  // The arena is the larger quantity for every image family -- FLUX.2's
  // VAE weighs 160 MB and its decode peaks near 2.8 GB at 1024^2.
  //
  // kPhaseDecode, so it is not charged to the denoise: the arena does
  // not exist while the DiT runs. It still lands in the box-level peak,
  // which is where it belongs -- generate-image frees its DiT for a
  // decode that will not fit (free_flux2_dit_for_decode_ and its
  // siblings), and the peak is what says whether that will be needed.
  //
  // An over-estimate here costs caution; the geometry is config, so it
  // is settled before anything loads.
  //
  // With NO geometry -- an edit graph takes its size from the reference
  // image, so height/width are absent -- this is kUnknownArena: a
  // presence marker, negligible by construction, that gives the runtime
  // figure something to revise. Declaring nothing would leave
  // publish_decode_arena_ with no entry to correct.
  std::size_t arena =
      model_memory::vae_decode_scratch_bytes(root, _width, _height);
  if (arena == 0) { arena = model_memory::kUnknownArena; }
  for (auto& c : model_memory::scratch_claims("vae-decode", arena,
                                              model_memory::kPhaseDecode)) {
    out.push_back(std::move(c));
  }
  return out;
}

// Both directions of one question: what did the caller ask for, and
// which family is resident. It runs on every change to either, because a
// config beat and a model reference arrive on different ports and either
// can be first -- parsing at only one of those moments loses whichever
// came earlier. False when the beat is for a DIFFERENT family, which is
// reported and then ignored.
bool
GenerateImageStage::apply_model_config_()
{
  const std::string want = model_config::family_of(_model_cfg);
  if (!want.empty() && want != _family) {
    session()->warn(fmt(
        "GenerateImageStage('{}'): the model_config beat is for the '{}' "
        "family but the resident checkpoint is '{}'; IGNORING it and using "
        "the defaults. Wire the config stage that matches the checkpoint",
        this->id(), want, _family));
    return false;
  }
  std::string perr;
  if (_family == "flux2") {
    _flux2_params =
        genai::MetalFlux2Transformer::GenerationParams::from_flex(_model_cfg,
                                                                  &perr);
  } else if (_family == "mage-flow") {
    _wm_params = genai::mage_wm::Params::from_flex(_model_cfg, &perr);
  }
  if (!perr.empty() && !_model_cfg.is_null()) {
    session()->warn(fmt("GenerateImageStage('{}'): model_config: {}",
                        this->id(), perr));
  }
  return true;
}

Job
GenerateImageStage::initialize(RuntimeContext& ctx)
{
  // Defer the (heavy) DiT load when a model-select source feeds the model iport
  // (its beat only arrives after the init barrier, in process()). Otherwise
  // load now from the config hf_dir, as before.
  const bool model_from_iport =
      ctx.num_iports() > kModelPort && ctx.iport_connected(kModelPort);
  // The SAME deferral for a wired model_config, and for a sharper
  // reason: `klein_kv` selects the attention recipe, so load() has to
  // know it. Loading here and reading the config afterwards would build
  // the DiT under the wrong recipe and then never rebuild it -- and
  // FLUX.2's two variants both load cleanly either way, so the mistake
  // would surface as wrong images rather than as an error.
  const bool cfg_from_iport =
      ctx.num_iports() > kModelCfgPort && ctx.iport_connected(kModelCfgPort);
  if (!model_from_iport && !cfg_from_iport) { ensure_loaded_(); }
  co_return;
}

void
GenerateImageStage::ensure_loaded_()
{
  if (_load_attempted) { return; }   // idempotent: load at most once
  _load_attempted = true;
  if (_hf_dir.empty()) {
    session()->error(fmt(
        "GenerateImageStage('{}'): no model -- set config.hf_dir or wire a "
        "model-select source to the model iport; inert", this->id()));
    return;
  }
  auto* mc = session() ? session()->services()->metal_compute() : nullptr;
  if (mc == nullptr) {
    session()->error(fmt(
        "GenerateImageStage('{}'): no metal-compute backend; the stage is inert",
        this->id()));
    return;
  }
  namespace fs = std::filesystem;
  const std::string root = resolve_model_dir(session(), _hf_dir);
  // The text encoder lives in the paired diffusion-conditioner stage, but its
  // weights are resident in the SAME process while the DiT runs, so the
  // encoder-coexistence streaming heuristics below still budget for it.
  // Boogu-Image calls its encoder `mllm/` (a full multimodal LLM, not a
  // diffusers text_encoder): sizing it as text_encoder/ silently scored the
  // encoder at ZERO bytes, so the streaming decision under-counted by the
  // largest peer in the pipeline and a bounded box would preload instead.
  std::string enc_dir = (fs::path(root) / "mllm").string();
  {
    std::error_code eec;
    if (!fs::is_directory(enc_dir, eec)) {
      enc_dir = (fs::path(root) / "text_encoder").string();
    }
  }
  const std::string dit_dir = _dit_dir.empty()
      ? (fs::path(root) / "transformer").string()
      : resolve_model_dir(session(), _dit_dir);
  _family = t2i_family_(dit_dir);
  // The family is only now known, so this is the first moment a config
  // beat that already arrived can be parsed by anything -- and it has to
  // happen BEFORE the load below, since klein_kv is an argument to it.
  apply_model_config_();
  const std::string size_desc =
      _infer_size ? std::string("auto (from ref_latent0)")
                  : std::to_string(_width) + "x" + std::to_string(_height);
  session()->log_debug(fmt(
      "GenerateImageStage('{}'): init root='{}' dit='{}' family={} "
      "(default {}, {} steps, seed {}, strength {})", this->id(), root,
      dit_dir, _family, size_desc, _steps, _seed, _strength));

  session()->info(fmt(
      "GenerateImageStage('{}'): loading {} DiT from '{}'", this->id(),
      _family == "flux2" ? "FLUX.2"
      : _family == "qwen-image-edit" ? "Qwen-Image-Edit MMDiT"
      : _family == "mage-flow" ? "Mage-Flow NR-MMDiT"
      : _family == "boogu-image" ? "Boogu-Image NextDiT"
      : "Krea2 MMDiT", dit_dir));
  if (_family == "flux2") {
    // Stream the DiT blocks when the box can't hold encoder + DiT together
    // (e.g. the 18 GB 9B DiT + a 16 GB encoder on a 16/32 GB box); ~2-3x slower
    // per step but bounds peak RAM to ~one block. VPIPE_FLUX2_STREAM forces it.
    bool stream_blocks;
    double pin_frac = 0.0;
    {
      const auto plan = model_memory::plan_streaming(
          session(), dit_dir, enc_dir, model_memory::kStreamHeadroom);
      stream_blocks = plan.stream;
      pin_frac      = plan.pin_frac;
      if (const char* e = std::getenv("VPIPE_FLUX2_STREAM")) {
        stream_blocks = (std::atoi(e) != 0);
        if (!stream_blocks) { pin_frac = 0.0; }
      }
      session()->log_debug(fmt(
          "GenerateImageStage('{}'): FLUX.2 footprint {} GB (others {} GB) + 6 "
          "GB vs {} GB RAM -> {}", this->id(), plan.footprint >> 30,
          plan.others >> 30, phys_ram() >> 30,
          stream_blocks ? "STREAM blocks" : "PRELOAD"));
    }
    if (const char* e = std::getenv("VPIPE_FLUX2_PIN_FRAC")) {
      pin_frac = std::atof(e);
    }
    genai::MetalFlux2Transformer::Config fcfg;
    fcfg.i8_gemm = _i8_gemm;
    _flux2_params.apply_to(fcfg);
    // Nothing in the checkpoint says which recipe it wants, and getting it
    // backwards costs plausible-looking wrong images rather than an error --
    // so when the path LOOKS like the other variant, say so. A hint, never an
    // override: the config key stays authoritative.
    {
      std::string dl = dit_dir;
      for (auto& ch : dl) { ch = (char)std::tolower((unsigned char)ch); }
      const bool looks_kv = dl.find("-kv") != std::string::npos
                            || dl.find("_kv") != std::string::npos;
      if (looks_kv != _flux2_params.klein_kv) {
        session()->warn(fmt(
            "GenerateImageStage('{}'): '{}' looks like {} but klein_kv={}. The "
            "-kv checkpoint isolates reference tokens; running either variant "
            "under the other's attention produces wrong images silently",
            this->id(), dit_dir,
            looks_kv ? "FLUX.2-klein-9b-kv" : "plain FLUX.2-klein",
            _flux2_params.klein_kv ? "true" : "false"));
      }
    }
    _flux2_dit = genai::MetalFlux2Transformer::load(
        weight_set_(dit_dir), mc, fcfg, stream_blocks, pin_frac);
    if (!_flux2_dit) {
      session()->error(fmt(
          "GenerateImageStage('{}'): failed to load the FLUX.2 DiT from '{}'; "
          "inert", this->id(), dit_dir));
      return;
    }
    // When the DiT had to stream (box can't hold encoder + DiT together), drop
    // the DiT's per-forward scratch after each generation so it doesn't crowd
    // out a large downstream VAE decode.
    _release_scratch = stream_blocks;
    if (stream_blocks) { revise_dit_declaration_(dit_dir); }
    // Cache the load params + VAE base so a memory-driven free after a
    // generation (free_flux2_dit_for_decode_) can reload identically.
    _flux2_dit_dir  = dit_dir;
    _flux2_stream   = stream_blocks;
    _flux2_pin_frac = pin_frac;
    _vae_base       = flux2_vae_base_(root);
  } else if (_family == "qwen-image-edit") {
    // Dual-stream Qwen-Image-Edit DiT (20B). Stream the blocks when the box
    // can't hold encoder + DiT together (else it can't run on a 16 GB box), and
    // PIN as many leading blocks resident as fit ALONGSIDE the per-prompt
    // encoder: pinned + encoder + headroom <= RAM. The encoder is reloaded for
    // each prompt's conditioning and stays resident while the DiT weights do,
    // so -- unlike the calibration collector, which frees the encoder before
    // loading the DiT and can pin 60% -- the stage must budget around it. On a
    // 16 GB box with the ~14 GB bf16 encoder that leaves no room (pin 0 => pure
    // streaming, ~one block resident); a roomier box pins more. Pinned blocks
    // are read once + reused, only the tail streams. VPIPE_QIE_STREAM /
    // VPIPE_QIE_PIN_FRAC override.
    const auto plan = model_memory::plan_streaming(
        session(), dit_dir, enc_dir, model_memory::kStreamHeadroom);
    bool   stream_blocks = plan.stream;
    double pin_frac      = plan.pin_frac;
    if (const char* e = std::getenv("VPIPE_QIE_STREAM")) {
      stream_blocks = (std::atoi(e) != 0);
      if (!stream_blocks) { pin_frac = 0.0; }
    }
    if (const char* e = std::getenv("VPIPE_QIE_PIN_FRAC")) {
      pin_frac = std::atof(e);
    }
    session()->log_debug(fmt(
        "GenerateImageStage('{}'): Qwen-Image-Edit footprint {} GB (others {} "
        "GB) + {} GB headroom vs {} GB RAM -> {}", this->id(),
        plan.footprint >> 30, model_memory::kStreamHeadroom >> 30,
        plan.others >> 30, phys_ram() >> 30,
        stream_blocks ? "STREAM blocks" : "PRELOAD"));
    genai::MetalQwenImageTransformer::Config qcfg;
    _qie_dit = genai::MetalQwenImageTransformer::load(
        weight_set_(dit_dir), mc, qcfg, stream_blocks, pin_frac);
    if (!_qie_dit) {
      session()->error(fmt(
          "GenerateImageStage('{}'): failed to load the Qwen-Image-Edit DiT from "
          "'{}'; inert", this->id(), dit_dir));
      return;
    }
    if (stream_blocks) {
      session()->info(fmt(
          "GenerateImageStage('{}'): Qwen-Image-Edit DiT streaming, pinned {} of "
          "{} blocks resident", this->id(), _qie_dit->pinned_blocks(),
          qcfg.n_layers));
    }
    _release_scratch = stream_blocks;
    if (stream_blocks) { revise_dit_declaration_(dit_dir); }
  } else if (_family == "boogu-image") {
    // Boogu's 10B NextDiT. At ~20 GB bf16 it streams on any box that cannot
    // hold it beside the resident Qwen3-VL mllm (~16 GB), which is every box
    // short of ~48 GB; the three refiner stacks stay resident either way.
    // VPIPE_BOOGU_STREAM / VPIPE_BOOGU_PIN_FRAC override.
    const auto plan = model_memory::plan_streaming(
        session(), dit_dir, enc_dir, model_memory::kStreamHeadroom);
    bool   stream_blocks = plan.stream;
    double pin_frac      = plan.pin_frac;
    if (const char* e = std::getenv("VPIPE_BOOGU_STREAM")) {
      stream_blocks = (std::atoi(e) != 0);
      if (!stream_blocks) { pin_frac = 0.0; }
    }
    if (const char* e = std::getenv("VPIPE_BOOGU_PIN_FRAC")) {
      pin_frac = std::atof(e);
    }
    session()->log_debug(fmt(
        "GenerateImageStage('{}'): Boogu footprint {} GB (others {} GB) + {} "
        "GB headroom vs {} GB RAM -> {}", this->id(), plan.footprint >> 30,
        plan.others >> 30, phys_ram() >> 30,
        stream_blocks ? "STREAM blocks" : "PRELOAD"));
    // Cache the load params so the stage can free the DiT for a big decode and
    // reload it on the next prompt (free_boogu_dit_for_decode_).
    _boogu_dit_dir = dit_dir;
    _boogu_stream = stream_blocks;
    _boogu_pin_frac = pin_frac;
    _vae_base = flux2_vae_base_(root);   // Boogu's VAE is a plain AutoencoderKL
    _release_scratch = stream_blocks;
    if (stream_blocks) { revise_dit_declaration_(dit_dir); }
    if (stream_blocks) {
      // DEFER the load on a bounded box. Streaming and per-prompt reload only
      // help if the DiT is not resident WHILE the conditioner's encoder is:
      // both stages load in initialize(), so a preloaded DiT plus a resident
      // Qwen3-VL mllm is the peak, and it is the peak that decides whether the
      // box copes (measured: 17.7 GB resident for a 4-bit Boogu when both load
      // up front, 8.7 GB when they take turns). The DiT loads on its first
      // conditioning beat -- by which time the conditioner has dropped the
      // encoder -- and is freed again for the vae-decode.
      _dit_unloaded = true;
      session()->info(fmt(
          "GenerateImageStage('{}'): memory-bounded -- the Boogu-Image DiT loads "
          "on the first conditioning beat (block streaming on) and is freed for "
          "the vae-decode, so it never shares the box with the text encoder",
          this->id()));
    } else if (!load_boogu_dit_()) {
      session()->error(fmt(
          "GenerateImageStage('{}'): failed to load the Boogu-Image DiT from "
          "'{}'; inert", this->id(), dit_dir));
      return;
    }
  } else if (_family == "mage-flow") {
    // Mage-Flow's 4B NR-MMDiT is MetalQwenImageTransformer under a different
    // Config (12 dual-stream blocks, in_channels 128 @ patch_size 1, txt_dim
    // 2560, text stream unrotated, bf16 timestep frequencies) -- see
    // mage_flow_dit_config(). At ~8 GB bf16 it streams on a box that can't
    // hold it beside the resident Qwen3-VL encoder, same rule as the others.
    const auto plan = model_memory::plan_streaming(
        session(), dit_dir, enc_dir, model_memory::kStreamHeadroom);
    bool   stream_blocks = plan.stream;
    double pin_frac      = plan.pin_frac;
    if (const char* e = std::getenv("VPIPE_MAGE_STREAM")) {
      stream_blocks = (std::atoi(e) != 0);
      if (!stream_blocks) { pin_frac = 0.0; }
    }
    if (const char* e = std::getenv("VPIPE_MAGE_PIN_FRAC")) {
      pin_frac = std::atof(e);
    }
    session()->log_debug(fmt(
        "GenerateImageStage('{}'): Mage-Flow footprint {} GB (others {} GB) + 6 "
        "GB vs {} GB RAM -> {}", this->id(), plan.footprint >> 30,
        plan.others >> 30, phys_ram() >> 30,
        stream_blocks ? "STREAM blocks" : "PRELOAD"));
    auto mcfg = genai::mage_flow_dit_config();
    _mage_dit = genai::MetalMageFlowTransformer::load(
        weight_set_(dit_dir), mc, mcfg, stream_blocks, pin_frac);
    if (!_mage_dit) {
      session()->error(fmt(
          "GenerateImageStage('{}'): failed to load the Mage-Flow DiT from '{}'; "
          "inert", this->id(), dit_dir));
      return;
    }
    if (stream_blocks) {
      session()->info(fmt(
          "GenerateImageStage('{}'): Mage-Flow DiT streaming, pinned {} of {} "
          "blocks resident", this->id(), _mage_dit->pinned_blocks(),
          mcfg.n_layers));
    }
    _release_scratch = stream_blocks;
    if (stream_blocks) { revise_dit_declaration_(dit_dir); }
    _mage_shift = mage_static_shift_(root);
  } else {
    genai::MetalKrea2Transformer::Config kcfg;
    kcfg.i8_gemm = _i8_gemm;
    _dit = genai::MetalKrea2Transformer::load(weight_set_(dit_dir), mc, kcfg);
    if (!_dit) {
      session()->error(fmt(
          "GenerateImageStage('{}'): failed to load the DiT from '{}'; inert",
          this->id(), dit_dir));
      return;
    }
    // Cache the resolved dir (reload after a decode-driven free) + the VAE base
    // channels for the decode-peak estimate (shared MetalKrea2Vae -> base_dim).
    _krea2_dit_dir = dit_dir;
    _vae_base = krea2_vae_base_(root);
    // The Krea-2 DiT mmaps its quantized weights (evictable under GPU memory
    // pressure), but the conditioner's dense text encoder is copied into dirty,
    // non-reclaimable buffers resident in this process. When the box can't hold
    // encoder + DiT + a large VAE decode at once, drop the DiT's per-forward
    // scratch after each generation so its working set doesn't crowd out the
    // downstream vae-decode stage.
    const std::vector<std::string> peers = {dit_dir, enc_dir};
    _release_scratch =
        model_memory::bounded(session(), peers, model_memory::kHeadroom);
    session()->log_debug(fmt(
        "GenerateImageStage('{}'): Krea2 footprint {} GB + {} GB headroom vs {} "
        "GB RAM -> {} DiT scratch", this->id(),
        model_memory::weight_footprint(session(), peers) >> 30,
        phys_ram() >> 30, _release_scratch ? "RELEASE" : "keep"));
  }
  session()->log_debug(fmt(
      "GenerateImageStage('{}'): {} DiT ready{}",
      this->id(), _family,
      _strength > 0.0 ? "; img2img init latent expected on a ref_latent iport"
                      : ""));
}

bool
GenerateImageStage::load_flux2_dit_()
{
  auto* mc = session() ? session()->services()->metal_compute() : nullptr;
  if (mc == nullptr || _flux2_dit_dir.empty()) { return false; }
  genai::MetalFlux2Transformer::Config fcfg;
  fcfg.i8_gemm = _i8_gemm;
  _flux2_params.apply_to(fcfg);
  _flux2_dit = genai::MetalFlux2Transformer::load(
      weight_set_(_flux2_dit_dir), mc, fcfg, _flux2_stream, _flux2_pin_frac);
  return (bool)_flux2_dit;
}

// State the decode arena this image implies, before deciding anything
// about it.
//
// The plan could only bound this, and for an EDIT graph it could not
// even do that: the output geometry comes from the reference image, so
// there is no height/width in any config to estimate from and
// declare_resources() correctly declares nothing. This is where the
// number becomes known -- the same expression the reclaim check below
// uses, so the ledger and the decision cannot drift apart.
//
// Published BEFORE the budget guards, and unconditionally: whether THIS
// stage needs to free its DiT is a different question from how much
// vae-decode is about to allocate, and a roomy box must still put the
// arena on the books for whoever sizes next.
//
// A REVISION, so declare_resources must have claimed the label -- with
// kUnknownArena when it had no geometry to size from. That is what
// keeps the plan authoritative about what exists.
void
GenerateImageStage::publish_decode_arena_(std::size_t peak) const
{
  if (peak == 0 || session() == nullptr ||
      session()->services() == nullptr) {
    return;
  }
  if (auto* mgr = session()->services()->generative_model_manager()) {
    mgr->revise_scratch("vae-decode", peak);
  }
}

void
GenerateImageStage::free_flux2_dit_for_decode_(int gen_w, int gen_h)
{
  if (!_flux2_dit || gen_w <= 0 || gen_h <= 0) { return; }
  publish_decode_arena_((std::size_t)gen_h * gen_w *
                        (std::size_t)_vae_base * 2 * 7);
  auto* mc = session() ? session()->services()->metal_compute() : nullptr;
  if (mc == nullptr) { return; }
  const auto mb = mc->memory_budget();
  if (mb.recommended == 0) { return; }   // budget query unavailable
  // FLUX.2 VAE decode peak, mirroring MetalFlux2Vae::decode_peak_bytes on the
  // DEFAULT hw-conv + split path: the big convs run the NAX hardware conv and
  // never materialize the 9*base im2col scratch, so the resident peak is the
  // per-level activation pool -- MEASURED ~7x one full-res base-ch buffer at
  // 1024^2 (~2.8 GB decode delta), NOT the 9x-im2col figure (a ~2.3x phantom
  // that over-estimated the decode ~2x). _vae_base = max(block_out[0..1]).
  const std::size_t peak =
      (std::size_t)gen_h * gen_w * (std::size_t)_vae_base * 2 * 7;
  // Free if EITHER budget is short: fits() (our Metal working set) misses
  // pressure from OTHER apps' resident memory; fits_physical() (host-wide
  // reclaimable RAM) catches it, so external pressure triggers the reclaim
  // instead of the decode later failing its own physical backstop.
  if (mb.fits(peak) && mb.fits_physical(peak)) { return; }
  // The DiT is idle now (its latent is read back). Free the ~9 GB of mmap'd
  // weights so the downstream vae-decode stage's working set has room; the DiT
  // reloads lazily when the next prompt arrives. Done BEFORE the latent is
  // published, so the decode sees the freed working set.
  session()->info(fmt(
      "GenerateImageStage('{}'): freeing the FLUX.2 DiT ({} MB working-set "
      "headroom / ~{} MB reclaimable < ~{} MB for the {}x{} vae-decode); "
      "reloads on the next prompt", this->id(), mb.headroom >> 20,
      mb.available_physical >> 20, peak >> 20, gen_w, gen_h));
  _flux2_dit.reset();
  _dit_unloaded = true;
}

// The manager's shared, reference-counted view of a checkpoint. Two
// pipelines running the same model get the same set and share its
// weights; the DiT holds the ONLY reference from this stage, deliberately
// -- free_*_dit_for_decode_ has to actually free, and it would not if the
// stage kept a handle of its own.
std::shared_ptr<genai::WeightSet>
GenerateImageStage::weight_set_(const std::string& dir) const
{
  if (dir.empty()) { return nullptr; }
  // Falls back to a private set when the session has no manager; see
  // open_weight_set().
  return genai::open_weight_set(dir, session());
}

bool
GenerateImageStage::load_boogu_dit_()
{
  auto* mc = session() ? session()->services()->metal_compute() : nullptr;
  if (mc == nullptr || _boogu_dit_dir.empty()) { return false; }
  genai::MetalBooguTransformer::Config bcfg;
  bcfg.i8_gemm = _i8_gemm;
  _boogu_dit = genai::MetalBooguTransformer::load(weight_set_(_boogu_dit_dir),
                                                 mc, bcfg, _boogu_stream,
                                                 _boogu_pin_frac);
  if (_boogu_dit && _boogu_stream) {
    session()->info(fmt(
        "GenerateImageStage('{}'): Boogu DiT streaming, pinned {} of {} blocks "
        "resident", this->id(), _boogu_dit->pinned_blocks(),
        _boogu_dit->config().n_double + _boogu_dit->config().n_single));
  }
  return (bool)_boogu_dit;
}

void
GenerateImageStage::free_boogu_dit_for_decode_(int gen_w, int gen_h)
{
  if (!_boogu_dit || gen_w <= 0 || gen_h <= 0) { return; }
  publish_decode_arena_((std::size_t)gen_h * gen_w *
                        (std::size_t)_vae_base * 2 * 7);
  auto* mc = session() ? session()->services()->metal_compute() : nullptr;
  if (mc == nullptr) { return; }
  const auto mb = mc->memory_budget();
  if (mb.recommended == 0) { return; }   // budget query unavailable
  // Boogu's VAE is the plain AutoencoderKL through MetalFlux2Vae, so the decode
  // peak is the FLUX.2 estimate: ~7x one full-res base-channel buffer.
  const std::size_t peak =
      (std::size_t)gen_h * gen_w * (std::size_t)_vae_base * 2 * 7;
  if (mb.fits(peak) && mb.fits_physical(peak)) { return; }
  session()->info(fmt(
      "GenerateImageStage('{}'): freeing the Boogu-Image DiT ({} MB working-set "
      "headroom / ~{} MB reclaimable < ~{} MB for the {}x{} vae-decode); "
      "reloads on the next prompt", this->id(), mb.headroom >> 20,
      mb.available_physical >> 20, peak >> 20, gen_w, gen_h));
  _boogu_dit.reset();
  _dit_unloaded = true;
}

bool
GenerateImageStage::load_krea2_dit_()
{
  auto* mc = session() ? session()->services()->metal_compute() : nullptr;
  if (mc == nullptr || _krea2_dit_dir.empty()) { return false; }
  genai::MetalKrea2Transformer::Config kcfg;
  kcfg.i8_gemm = _i8_gemm;
  _dit = genai::MetalKrea2Transformer::load(weight_set_(_krea2_dit_dir), mc,
                                            kcfg);
  return (bool)_dit;
}

void
GenerateImageStage::free_krea2_dit_for_decode_(int gen_w, int gen_h)
{
  if (!_dit || gen_w <= 0 || gen_h <= 0) { return; }
  {
    const std::size_t im2col =
        (std::size_t)gen_h * gen_w * 9 * (std::size_t)_vae_base * 2;
    publish_decode_arena_(im2col + im2col / 2);
  }
  auto* mc = session() ? session()->services()->metal_compute() : nullptr;
  if (mc == nullptr) { return; }
  const auto mb = mc->memory_budget();
  if (mb.recommended == 0) { return; }   // budget query unavailable
  // Krea-2 VAE (shared MetalKrea2Vae) decode peak, mirroring its
  // decode_peak_bytes on the default split-on path: one up-level's top-res
  // im2col scratch Hout*Wout*9*base*2 (= gen_h*gen_w*9*base*2) + ~50% for the
  // level's I/O activations. Checked AFTER release_forward_scratch() so we only
  // free the (reloadable) weights when dropping the scratch wasn't enough.
  const std::size_t im2col =
      (std::size_t)gen_h * gen_w * 9 * (std::size_t)_vae_base * 2;
  const std::size_t peak = im2col + im2col / 2;
  // Free if EITHER budget is short. fits() (our Metal working set) misses
  // pressure from OTHER apps' resident memory; fits_physical() (host-wide
  // reclaimable RAM) catches it -- otherwise external pressure leaves the DiT
  // resident and the decode just fails its own physical backstop instead.
  if (mb.fits(peak) && mb.fits_physical(peak)) { return; }
  session()->info(fmt(
      "GenerateImageStage('{}'): freeing the Krea-2 DiT ({} MB working-set "
      "headroom / ~{} MB reclaimable < ~{} MB for the {}x{} vae-decode); "
      "reloads on the next prompt", this->id(), mb.headroom >> 20,
      mb.available_physical >> 20, peak >> 20, gen_w, gen_h));
  _dit.reset();
  _dit_unloaded = true;
}

namespace {

// Copy a conditioning TensorBeat (2 bytes/elt, f16 or bf16) into a metal
// SharedBuffer for the DiT. The element type is family-fixed and interpreted by
// the DiT (f16 for krea2/flux2, bf16 for qwen-image-edit); here it is opaque
// bytes. Empty on allocation failure.
metal_compute::SharedBuffer
cond_to_shared_(metal_compute::MetalCompute* mc, const TensorBeatPayload& tb)
{
  const auto bytes = tb.materialize_contiguous();
  metal_compute::SharedBuffer b = mc->make_shared_buffer(bytes.size());
  if (!b.empty()) { std::memcpy(b.contents(), bytes.data(), bytes.size()); }
  return b;
}

}  // namespace

std::vector<float>
GenerateImageStage::generate_(const metal_compute::SharedBuffer& cond, int n_real,
                            const metal_compute::SharedBuffer& cond_neg,
                            int n_real_neg, int gen_h, int gen_w,
                            const std::vector<float>* init_packed,
                            const std::vector<float>* init_latent,
                            const std::vector<RefLatent>& refs,
                            const std::function<void(const std::vector<float>&)>&
                                emit_step) const
{
  auto* mc = session()->services()->metal_compute();
  using metal_compute::SharedBuffer;
  const int H = gen_h, W = gen_w;
  const int lh = H / 8, lw = W / 8;          // latent H/W
  const int gh = H / 16, gw = W / 16;        // 2x2-patch grid
  const int img_seq = gh * gw;
  const int IC = 64;                         // z_dim(16) * patch(2) * patch(2)

  // How much room the streamed DiT must leave clear to keep a block
  // resident. Krea-2 has carried the residency machinery -- measurement,
  // eviction, ratchet -- since it was written, and never grew a single
  // block, because nothing ever set this: a reserve left at its default
  // means "this model was never taught what its activations cost", and
  // growth stays off. Wired here so the policy is live rather than
  // decorative.
  //
  // Same rule as FLUX.2, against Krea-2's own decode peak
  // (free_krea2_dit_for_decode_): reserve the VAE decode only when it
  // would actually run beside us. When it would not, that free drops the
  // whole DiT first, which is strictly more than a reserve could hold
  // clear -- so reserving it there protects a coexistence that never
  // happens and costs every block of every step.
  if (_vae_base > 0 && _dit) {
    const std::size_t im2col =
        (std::size_t)H * W * 9 * (std::size_t)_vae_base * 2;
    const std::size_t peak = im2col + im2col / 2;
    const auto mb = mc->memory_budget();
    // No budget to read -> the free bails out too, so the DiT survives.
    const bool decode_runs_beside_us =
        mb.recommended == 0 || (mb.fits(peak) && mb.fits_physical(peak));
    _dit->set_residency_reserve(decode_runs_beside_us ? peak : 0);
    session()->log_debug(fmt(
        "GenerateImageStage('{}'): Krea-2 residency reserve {} MB -- the "
        "{}x{} vae-decode wants {} MB and {}",
        this->id(), (decode_runs_beside_us ? peak : 0) >> 20, W, H,
        peak >> 20,
        decode_runs_beside_us ? "fits beside the DiT"
                              : "does not, so the DiT is freed before it"));
  }

  // The conditioner emits the 12-tap f16 conditioning [n_real, 12, EH]; the
  // DiT's text-fusion tower fuses it into the DiT-facing text [n_real, hidden].
  SharedBuffer fused = _dit->forward_text(cond, n_real);
  if (fused.empty()) { return {}; }
  session()->log_debug(fmt(
      "GenerateImageStage('{}'): fused conditioning [{}, hidden]; "
      "{}x{} grid {}x{} img_seq {}", this->id(), n_real, W, H, gh, gw,
      img_seq));

  // Classifier-free guidance: fuse the negative conditioning too and, at each
  // denoise step, push the velocity away from it. Skipped when no negative
  // conditioning is wired or the scale is a no-op (1.0), so the single-pass
  // turbo default stays token-exact.
  bool cfg = !cond_neg.empty() && n_real_neg > 0 && _guidance_scale != 1.0;
  SharedBuffer fused_neg;
  if (cfg) {
    fused_neg = _dit->forward_text(cond_neg, n_real_neg);
    if (fused_neg.empty()) {
      session()->warn(fmt(
          "GenerateImageStage('{}'): negative-conditioning fuse failed; running "
          "without classifier-free guidance", this->id()));
      cfg = false;
    } else {
      session()->info(fmt(
          "GenerateImageStage('{}'): CFG on, scale {}, negative [{} rows]",
          this->id(), (float)_guidance_scale, n_real_neg));
    }
  }

  // 6. build the sampler (integrator) + scheduler (sigma schedule) from the
  //    active specs (config defaults or latched select-stage beats). The
  //    distilled turbo default (euler + simple / steps / shift 1.15 exp) is the
  //    token-exact schedule.
  genai::FlowSampler sampler(_sampler_spec, _scheduler_spec);
  const int S = sampler.steps();
  const std::vector<double>& sig = sampler.sigmas();

  // img2img start step: run only the tail (t_start .. S). t2i => start 0.
  int start = 0;
  if (_strength > 0.0) {
    const double init_ts = std::min((double)S * _strength, (double)S);
    start = (int)std::max((double)S - init_ts, 0.0);
    if (start >= S) { start = S - 1; }
  }
  const double sig0 = sig[(std::size_t)start];
  session()->log_debug(fmt(
      "GenerateImageStage('{}'): sampler={} scheduler={} steps={} start={} "
      "sig0={} (strength {})", this->id(), _sampler_spec.method,
      _scheduler_spec.type, S, start, (float)sig0, _strength));

  // 7. initial packed latents [img_seq, 64].
  std::vector<float> packed((std::size_t)img_seq * IC);
  if (init_packed != nullptr && init_packed->size() == packed.size()) {
    packed = *init_packed;                       // supplied (repro / golden)
  } else if (init_latent != nullptr && _strength > 0.0 &&
             init_latent->size() == (std::size_t)16 * lh * lw) {
    // img2img: the whitened latent [16, lh, lw] arrives from the vae-encode
    // stage on the `latent` iport. Mix it with noise at sig0 (scale_noise),
    // then pack.
    const float* lp = init_latent->data();
    std::mt19937_64 rng(_seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> mixed((std::size_t)16 * lh * lw);
    for (std::size_t i = 0; i < mixed.size(); ++i) {
      const float n = nd(rng);
      mixed[i] = (float)sig0 * n + (float)(1.0 - sig0) * lp[i];
    }
    // pack [16, lh, lw] -> [img_seq, 64] (inverse of the unpack below).
    for (int c = 0; c < 16; ++c) {
      for (int y = 0; y < lh; ++y) {
        for (int xx = 0; xx < lw; ++xx) {
          const int a = y / 2, ph = y % 2, b = xx / 2, pw = xx % 2;
          const std::size_t t = (std::size_t)a * gw + b;
          packed[t * IC + (std::size_t)c * 4 + ph * 2 + pw] =
              mixed[((std::size_t)c * lh + y) * lw + xx];
        }
      }
    }
  } else {
    std::mt19937_64 rng(_seed);                  // text-to-image: pure noise
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : packed) { v = nd(rng); }
  }

  // 8. sampler denoising loop (from `start` for img2img). The interchangeable
  //    integrator drives the DiT via this denoise callback: upload the
  //    candidate latent, run the DiT at the given sigma, read back the velocity.
  SharedBuffer latbuf = mc->make_shared_buffer((std::size_t)img_seq * IC * 2);
  if (latbuf.empty()) { return {}; }
  // Reference-image conditioning (built once, constant across denoise steps).
  // A ref latent arrives channel-first [16, rlh, rlw] (vae-encode output); pack
  // it 2x2 into token-major [rseq, 64] (the same patchify the generated latent
  // uses) so the DiT embeds it via img_in. Each reference lands in its own RoPE
  // frame band inside the DiT. Odd-dim or non-16-channel refs are skipped.
  std::vector<genai::MetalKrea2Transformer::RefImage> ri;
  for (const auto& r : refs) {
    if (r.empty() || r.c != 16 || (r.h % 2) != 0 || (r.w % 2) != 0) {
      if (!r.empty()) {
        session()->warn(fmt(
            "GenerateImageStage('{}'): krea2 reference latent [{}, {}, {}] must be "
            "16-channel with even H/W; ignoring", this->id(), r.c, r.h, r.w));
      }
      continue;
    }
    const int rgh = r.h / 2, rgw = r.w / 2, rseq = rgh * rgw;
    SharedBuffer rb = mc->make_shared_buffer((std::size_t)rseq * IC * 2);
    if (rb.empty()) { continue; }
    auto* d = static_cast<_Float16*>(rb.contents());
    std::memset(d, 0, rb.byte_size());
    for (int cc = 0; cc < 16; ++cc) {
      for (int y = 0; y < r.h; ++y) {
        for (int x = 0; x < r.w; ++x) {
          const int a = y / 2, ph = y % 2, bcol = x / 2, pw = x % 2;
          const std::size_t t = (std::size_t)a * rgw + bcol;
          d[t * IC + (std::size_t)cc * 4 + ph * 2 + pw] =
              (_Float16)r.chw[((std::size_t)cc * r.h + y) * r.w + x];
        }
      }
    }
    genai::MetalKrea2Transformer::RefImage img;
    img.latents = std::move(rb);
    img.seq = rseq; img.grid_h = rgh; img.grid_w = rgw;
    ri.push_back(std::move(img));
  }
  if (!ri.empty()) {
    session()->info(fmt(
        "GenerateImageStage('{}'): Krea-2 conditioning on {} reference image(s)",
        this->id(), ri.size()));
  }
  bool dit_ok = true;
  const float gscale = (float)_guidance_scale;
  // Opened before the denoise callable so the per-block hook is
  // live for the very first forward.
  UiProgress bar = session()->open_progress("denoise");
  DenoiseProgress prog(&bar, S - start, cfg ? 2 : 1);
  ScopedBlockProgress<std::remove_reference_t<decltype(*_dit)>>
      prog_guard(_dit.get(), prog);
  auto denoise = [&](const std::vector<float>& cand,
                     double sigma) -> std::vector<float> {
    auto* lb = static_cast<_Float16*>(latbuf.contents());
    for (std::size_t k = 0; k < cand.size(); ++k) { lb[k] = (_Float16)cand[k]; }
    SharedBuffer vel = _dit->forward_dit(fused, n_real, latbuf, img_seq, gh, gw,
                                         (float)sigma, -1, ri);
    prog.end_forward();
    if (vel.empty()) { dit_ok = false; return {}; }
    const auto* vp = static_cast<const _Float16*>(vel.contents());
    std::vector<float> v(cand.size());
    for (std::size_t k = 0; k < v.size(); ++k) { v[k] = (float)vp[k]; }
    if (cfg) {
      // Second DiT pass on the SAME candidate (latbuf unchanged), conditioned
      // on the negative prompt: v = v_neg + scale*(v_pos - v_neg).
      SharedBuffer veln = _dit->forward_dit(fused_neg, n_real_neg, latbuf,
                                            img_seq, gh, gw, (float)sigma, -1,
                                            ri);
      prog.end_forward();
      if (veln.empty()) { dit_ok = false; return {}; }
      const auto* np = static_cast<const _Float16*>(veln.contents());
      for (std::size_t k = 0; k < v.size(); ++k) {
        const float vneg = (float)np[k];
        v[k] = vneg + gscale * (v[k] - vneg);
      }
    }
    return v;
  };
  sampler.reset();   // clear multistep history / reseed the SDE RNG for this run
  const bool prof = std::getenv("VPIPE_KREA2_PROFILE") != nullptr;
  double dit_ms = 0.0;
  int dit_calls = 0;
  auto denoise_p = [&](const std::vector<float>& cand, double sigma) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<float> v = denoise(cand, sigma);
    dit_ms += std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t0).count();
    ++dit_calls;
    return v;
  };
  // Denoising is the slow part (several seconds of DiT forwards). Paint an
  // in-place progress bar on the user-facing stream (model-quantize style)
  // and time the whole loop so the operator sees steady progress instead
  // of a silent stall, plus a wall-clock summary when the latent is done.
  // unpack packed [img_seq, 64] -> channel-first latent [16, lh, lw] (2x2). Used
  // per-step (debug step_latents) and for the final return.
  auto unpack = [&](const std::vector<float>& pk) {
    std::vector<float> latent((std::size_t)16 * lh * lw);
    for (int c = 0; c < 16; ++c) {
      for (int y = 0; y < lh; ++y) {
        for (int xx = 0; xx < lw; ++xx) {
          const int a = y / 2, ph = y % 2, b = xx / 2, pw = xx % 2;
          const std::size_t t = (std::size_t)a * gw + b;
          latent[((std::size_t)c * lh + y) * lw + xx] =
              pk[t * IC + (std::size_t)c * 4 + ph * 2 + pw];
        }
      }
    }
    return latent;
  };
  const int nsteps = S - start;
  const auto gen_t0 = std::chrono::steady_clock::now();
  for (int i = start; i < S; ++i) {
    session()->log_debug(fmt(
        "GenerateImageStage('{}'): denoise step {}/{} sigma {}", this->id(),
        i + 1, S, (float)sig[(std::size_t)i]));
    sampler.step(i, packed,
                 prof ? genai::FlowSampler::DenoiseFn(denoise_p) : denoise);
    if (!dit_ok) { return {}; }
    if (emit_step) { emit_step(unpack(packed)); }
    prog.end_step(i - start);
  }
  const double gen_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - gen_t0).count();
  session()->info(fmt(
      "GenerateImageStage('{}'): latent generated in {:.2f}s ({} denoise "
      "steps, {} ms/step)", this->id(), gen_s, nsteps,
      nsteps ? (long)(gen_s * 1000.0 / nsteps) : 0));
  if (prof) {
    session()->log_normal(fmt(
        "GenerateImageStage('{}'): DiT {} forward_dit calls, {} ms total, {} "
        "ms/call (seq {}+{}={})", this->id(), dit_calls, (long)dit_ms,
        dit_calls ? (long)(dit_ms / dit_calls) : 0, n_real, img_seq,
        n_real + img_seq));
  }

  // 9. unpack [img_seq, 64] -> channel-first latent [16, lh, lw].
  return unpack(packed);
}

std::vector<float>
GenerateImageStage::generate_flux2_(const metal_compute::SharedBuffer& context,
                                  int n_real, int gen_h, int gen_w,
                                  const std::vector<RefLatent>& refs,
                                  const std::vector<float>* init_packed,
                                  const std::function<void(
                                      const std::vector<float>&)>& emit_step)
    const
{
  auto* mc = session()->services()->metal_compute();
  using metal_compute::SharedBuffer;
  const int H = gen_h, W = gen_w;
  const int gh = H / 16, gw = W / 16;        // VAE latent grid (128ch @ H/16)
  const int img_seq = gh * gw;
  const int IC = _flux2_dit->config().in_channels;   // 128
  // How much room the DiT must leave clear when it decides whether to keep
  // a streamed block resident. Its own activations are allocated before
  // the budget it grows against is read, so what is left to protect is the
  // VAE DECODE that runs after the denoise -- but ONLY where the decode
  // runs beside us.
  //
  // When it does not fit beside us, free_flux2_dit_for_decode_ drops the
  // whole ~9 GB DiT before the latent is published. It re-asks with the
  // grown set resident, and it frees strictly more than any reserve could
  // have held clear, so the decode cannot be starved by growth. Reserving
  // its peak on that path protects a coexistence that never happens, and
  // pays for it by streaming every block of every step.
  //
  // So reserve it only where it is real -- where the decode WOULD fit
  // beside us, and growth is the one thing that could tip us into a free
  // we did not need (which costs a reload on the next prompt, not a
  // failure). Same predicate as the free, measured rather than assumed;
  // the post-denoise check stays the authority on a borderline box.
  if (_vae_base > 0) {
    const std::size_t peak =
        (std::size_t)H * W * (std::size_t)_vae_base * 2 * 7;
    const auto mb = mc->memory_budget();
    // No budget to read -> the free bails out too, so the DiT survives.
    const bool decode_runs_beside_us =
        mb.recommended == 0 || (mb.fits(peak) && mb.fits_physical(peak));
    _flux2_dit->set_residency_reserve(decode_runs_beside_us ? peak : 0);
    session()->log_debug(fmt(
        "GenerateImageStage('{}'): FLUX.2 residency reserve {} MB -- the "
        "{}x{} vae-decode wants {} MB and {}",
        this->id(), (decode_runs_beside_us ? peak : 0) >> 20, W, H,
        peak >> 20,
        decode_runs_beside_us ? "fits beside the DiT"
                              : "does not, so the DiT is freed before it"));
  }
  // `context` is the diffusion-conditioner's flux2 conditioning: the {9,18,27}
  // encoder taps concatenated per token -> f16 [n_real, joint_dim=3*enc_hidden].

  // Sampler (FlowMatchEuler; klein distilled -> no CFG). FLUX.2's flow-shift
  // (mu) is resolution- AND step-dependent (compute_empirical_mu): a fixed
  // shift is only right near the base resolution and washes the latent to grey
  // elsewhere (e.g. 1024). Recompute mu from the image token count + steps and
  // override the scheduler shift (unless a scheduler-select beat explicitly set
  // a non-default shift, which the operator then owns).
  genai::FlowSchedulerSpec sched = _scheduler_spec;
  if (!_scheduler_latched) {
    sched.shift = flux2_empirical_mu_(img_seq, sched.steps);
    session()->log_debug(fmt(
        "GenerateImageStage('{}'): FLUX.2 flow-shift mu = {} (img_seq {}, {} "
        "steps)", this->id(), (float)sched.shift, img_seq, sched.steps));
  }
  genai::FlowSampler sampler(_sampler_spec, sched);
  const int S = sampler.steps();

  // Initial packed latents [img_seq, IC] (pure noise; the VAE already
  // patchified, so no extra 2x2 packing here).
  std::vector<float> packed((std::size_t)img_seq * IC);
  {
    std::mt19937_64 rng(_seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : packed) { v = nd(rng); }
  }
  if (init_packed != nullptr && init_packed->size() == packed.size()) {
    packed = *init_packed;                       // supplied (repro / golden)
  }
  SharedBuffer latbuf = mc->make_shared_buffer((std::size_t)img_seq * IC * 2);
  if (latbuf.empty()) { return {}; }
  // Embedded guidance (guidance-distilled variants, e.g. base FLUX.2 dev/pro):
  // when the DiT has a guidance_embedder AND the user raises guidance_scale
  // above 1, feed guidance = guidance_scale so it is embedded into the timestep
  // modulation (single forward pass, the base pipeline's semantics). The klein
  // reference passes guidance=None, so the default (guidance_scale 1.0) -> -1
  // reproduces the distilled golden exactly.
  const float guid = (_flux2_dit->config().guidance_embeds &&
                      _guidance_scale > 1.0)
                         ? (float)_guidance_scale
                         : -1.0f;
  if (guid >= 0.0f) {
    session()->info(fmt(
        "GenerateImageStage('{}'): FLUX.2 embedded guidance scale {}",
        this->id(), guid));
  }
  // Reference-image conditioning (built once, constant across denoise steps):
  // patchify-pack each vae-encode latent [IC, rh, rw] into a token-major [rh*rw,
  // IC] f16 buffer. The DiT appends these tokens to the image stream and gives
  // each its own RoPE T offset. Non-IC-channel or empty refs are skipped.
  std::vector<genai::MetalFlux2Transformer::RefImage> ri;
  for (const auto& r : refs) {
    if (r.empty()) { continue; }
    if (r.c != IC) {
      session()->warn(fmt(
          "GenerateImageStage('{}'): reference latent has {} channels, expected "
          "{} (DiT in_channels); ignoring", this->id(), r.c, IC));
      continue;
    }
    const int rh = r.h, rw = r.w, rseq = rh * rw;
    SharedBuffer rb = mc->make_shared_buffer((std::size_t)rseq * IC * 2);
    if (rb.empty()) { continue; }
    auto* d = static_cast<_Float16*>(rb.contents());
    for (int y = 0; y < rh; ++y) {
      for (int x = 0; x < rw; ++x) {
        const std::size_t t = (std::size_t)y * rw + x;
        for (int cc = 0; cc < IC; ++cc) {
          d[t * IC + cc] =
              (_Float16)r.chw[((std::size_t)cc * rh + y) * rw + x];
        }
      }
    }
    genai::MetalFlux2Transformer::RefImage img;
    img.latents = std::move(rb);
    img.seq = rseq; img.grid_h = rh; img.grid_w = rw;
    ri.push_back(std::move(img));
  }
  if (!ri.empty()) {
    session()->info(fmt(
        "GenerateImageStage('{}'): FLUX.2 conditioning on {} reference image(s)",
        this->id(), ri.size()));
  }
  bool dit_ok = true;
  // Reference K/V cache for the -kv checkpoint, scoped to THIS generation: the
  // first denoise step fills it, the rest reuse it. A local means a new run
  // (or new reference set) always starts from an empty one.
  genai::MetalFlux2Transformer::KvCache kvc;
  // VPIPE_FLUX2_NO_KV_CACHE recomputes the reference band every step while
  // keeping the recipe itself intact. That is the only honest way to price the
  // cache: turning `klein_kv` off instead would change the ATTENTION, so the
  // two arms would not be computing the same thing.
  genai::MetalFlux2Transformer::KvCache* kvp =
      (_flux2_params.klein_kv && !ri.empty() &&
       std::getenv("VPIPE_FLUX2_NO_KV_CACHE") == nullptr) ? &kvc : nullptr;
  // Opened before the denoise callable so the per-block hook is
  // live for the very first forward.
  UiProgress bar = session()->open_progress("denoise");
  DenoiseProgress prog(&bar, S, 1);
  ScopedBlockProgress<std::remove_reference_t<decltype(*_flux2_dit)>>
      prog_guard(_flux2_dit.get(), prog);
  auto denoise = [&](const std::vector<float>& cand,
                     double sigma) -> std::vector<float> {
    auto* lb = static_cast<_Float16*>(latbuf.contents());
    for (std::size_t k = 0; k < cand.size(); ++k) { lb[k] = (_Float16)cand[k]; }
    SharedBuffer vel = _flux2_dit->forward_dit(context, n_real, latbuf, img_seq,
                                               gh, gw, (float)sigma, guid, ri,
                                               kvp);
    prog.end_forward();
    if (vel.empty()) { dit_ok = false; return {}; }
    const auto* vp = static_cast<const _Float16*>(vel.contents());
    std::vector<float> v(cand.size());
    for (std::size_t k = 0; k < v.size(); ++k) { v[k] = (float)vp[k]; }
    return v;
  };
  sampler.reset();
  // Per-step DiT timing (VPIPE_FLUX2_PROFILE), mirroring the Krea-2 path's
  // VPIPE_KREA2_PROFILE: time each forward_dit + log a ms/call summary.
  const bool prof = std::getenv("VPIPE_FLUX2_PROFILE") != nullptr;
  double dit_ms = 0.0;
  int dit_calls = 0;
  auto denoise_p = [&](const std::vector<float>& cand, double sigma) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<float> v = denoise(cand, sigma);
    dit_ms += std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t0).count();
    ++dit_calls;
    return v;
  };
  // Unpack [img_seq, IC] (token-major) -> channel-first [IC, gh, gw]. Used
  // per-step (debug step_latents) and for the final return.
  auto unpack = [&](const std::vector<float>& pk) {
    std::vector<float> latent((std::size_t)IC * gh * gw);
    for (int i = 0; i < gh; ++i) {
      for (int j = 0; j < gw; ++j) {
        const std::size_t t = (std::size_t)i * gw + j;
        for (int cc = 0; cc < IC; ++cc) {
          latent[((std::size_t)cc * gh + i) * gw + j] = pk[t * IC + cc];
        }
      }
    }
    return latent;
  };
  const auto gen_t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < S; ++i) {
    sampler.step(i, packed,
                 prof ? genai::FlowSampler::DenoiseFn(denoise_p) : denoise);
    if (!dit_ok) { return {}; }
    if (emit_step) { emit_step(unpack(packed)); }
    prog.end_step(i);
  }
  const double gen_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - gen_t0).count();
  session()->info(fmt(
      "GenerateImageStage('{}'): FLUX.2 latent generated in {:.2f}s ({} steps)",
      this->id(), gen_s, S));
  if (prof) {
    session()->log_normal(fmt(
        "GenerateImageStage('{}'): FLUX.2 DiT {} forward_dit calls, {} ms total, "
        "{} ms/call (seq {}+{}={})", this->id(), dit_calls, (long)dit_ms,
        dit_calls ? (long)(dit_ms / dit_calls) : 0, n_real, img_seq,
        n_real + img_seq));
  }
  return unpack(packed);
}

// Boogu-Image: NextDiT forward + the checkpoint's own sampler.
//
// Two integrators live here because Boogu ships two kinds of checkpoint and
// they do NOT share a loop:
//   * "dmd" (Base/Edit-Turbo, the distilled default) -- at each ASCENDING sigma
//     jump straight to the x0 prediction, x <- x + (1 - s) * v, then RE-NOISE
//     down to the next sigma, x <- (1 - s') * eps + s' * x. No CFG.
//   * anything else (Base/Edit) -- Euler along the v1 logistic time-shift
//     schedule, x <- x + (s_next - s) * v, optionally with classifier-free
//     guidance from the conditioner's negative beat.
// Both run on Boogu's INVERTED sigma convention: 0 is pure noise and 1 is
// clean, so the schedule ASCENDS. That is why neither uses genai::FlowSampler
// (whose integrators all assume a descending sigma with a terminal 0).
std::vector<float>
GenerateImageStage::generate_boogu_(const metal_compute::SharedBuffer& txt_pos,
                                  int n_real,
                                  const metal_compute::SharedBuffer& txt_neg,
                                  int n_real_neg, int gen_h, int gen_w,
                                  const std::vector<float>* init_packed,
                                  const std::vector<RefLatent>& refs,
                                  const std::function<void(
                                      const std::vector<float>&)>& emit_step)
    const
{
  auto* mc = session()->services()->metal_compute();
  using metal_compute::SharedBuffer;
  const int H = gen_h, W = gen_w;
  const int Z = _boogu_dit->config().in_channels;   // 16 latent channels
  const int P = _boogu_dit->config().patch;         // 2
  const int lh = H / 8, lw = W / 8;                 // latent grid (VAE 8x)
  const int gh = lh / P, gw = lw / P;               // patch-token grid
  const int img_seq = gh * gw;
  const int IC = _boogu_dit->config().x_in();       // 64 = 16 * 2 * 2
  const int OC = _boogu_dit->config().out_channels;
  if (img_seq <= 0 || (lh % P) != 0 || (lw % P) != 0) {
    session()->warn(fmt(
        "GenerateImageStage('{}'): {}x{} gives a {}x{} latent, which is not a "
        "whole number of {}x{} patches -- use multiples of 16",
        this->id(), W, H, lw, lh, P, P));
    return {};
  }

  session()->log_debug(fmt(
      "GenerateImageStage('{}'): Boogu conditioning [{}, txt]; {}x{} latent {}x{} "
      "grid {}x{} img_seq {}", this->id(), n_real, W, H, lw, lh, gw, gh,
      img_seq));

  // Schedule. The sampler/scheduler beats win when latched; otherwise the
  // checkpoint default is the distilled 4-step DMD student.
  genai::FlowSamplerSpec samp = _sampler_spec;
  genai::FlowSchedulerSpec sched = _scheduler_spec;
  if (!_sampler_latched) { samp.method = "dmd"; samp.conditioning_sigma = 0.0; }
  if (!_scheduler_latched) {
    sched.type = "boogu_v1";
    sched.steps = _steps > 0 ? _steps : 4;
    sched.base_shift = 0.5; sched.max_shift = 1.15;
    sched.base_seq = 256; sched.max_seq = 4096;
    sched.seq_len = 4096;
  }
  sched.img_seq_len = img_seq;
  const bool dmd = (samp.method == "dmd");
  const int S = sched.steps < 1 ? 1 : sched.steps;
  // DMD builds its own linspace(conditioning_sigma, 1, S+1)[:-1]; the Euler
  // path takes the shifted schedule (ascending, terminal 1.0).
  std::vector<double> sig;
  if (dmd) {
    sig.resize((std::size_t)S);
    const double c0 = samp.conditioning_sigma;
    for (int i = 0; i < S; ++i) {
      sig[(std::size_t)i] = c0 + (1.0 - c0) * (double)i / (double)S;
    }
  } else {
    if (sched.type != "boogu_v1") {
      session()->warn(fmt(
          "GenerateImageStage('{}'): scheduler '{}' has a DESCENDING sigma "
          "convention, which Boogu's does not share; using boogu_v1",
          this->id(), sched.type));
      sched.type = "boogu_v1";
    }
    sig = sched.sigmas();
  }

  // Packed latents [img_seq, IC]: a supplied init (repro / golden) or noise.
  std::vector<float> packed((std::size_t)img_seq * IC);
  if (init_packed != nullptr && init_packed->size() == packed.size()) {
    packed = *init_packed;
  } else {
    std::mt19937_64 rng(_seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : packed) { v = nd(rng); }
  }
  SharedBuffer latbuf = mc->make_shared_buffer((std::size_t)img_seq * IC * 2);
  if (latbuf.empty()) { return {}; }

  // Reference conditioning: vae-encode hands us [16, rlh, rlw] channel-first;
  // pack it 2x2 into token-major [rseq, IC] bf16, the same patchify the target
  // uses. Boogu-Edit takes ONE reference (the model card says so outright) but
  // the DiT carries image_index_embedding rows for up to max_ref_images, so
  // extra references are passed through rather than dropped.
  std::vector<genai::MetalBooguTransformer::RefImage> bri;
  for (const auto& r : refs) {
    if (r.empty()) { continue; }
    if (r.c != Z || (r.h % P) != 0 || (r.w % P) != 0) {
      session()->warn(fmt(
          "GenerateImageStage('{}'): reference latent [{}, {}, {}] must be "
          "{}-channel with H/W divisible by {}; ignoring", this->id(), r.c,
          r.h, r.w, Z, P));
      continue;
    }
    if ((int)bri.size() >= _boogu_dit->config().max_ref_images) {
      session()->warn(fmt(
          "GenerateImageStage('{}'): more than {} reference images; ignoring the "
          "rest", this->id(), _boogu_dit->config().max_ref_images));
      break;
    }
    const int rgh = r.h / P, rgw = r.w / P, rseq = rgh * rgw;
    SharedBuffer rb = mc->make_shared_buffer((std::size_t)rseq * IC * 2);
    if (rb.empty()) { continue; }
    auto* d = static_cast<std::uint16_t*>(rb.contents());
    std::memset(d, 0, rb.byte_size());
    for (int cc = 0; cc < Z; ++cc) {
      for (int y = 0; y < r.h; ++y) {
        for (int x = 0; x < r.w; ++x) {
          const int a = y / P, ph = y % P, bcol = x / P, pw = x % P;
          const std::size_t t = (std::size_t)a * rgw + bcol;
          // Boogu packs "c (h p1) (w p2) -> (h w) (p1 p2 c)": inside a token
          // the CHANNEL is the fastest axis and the patch offsets the slower
          // ones. (Getting this transposed still round-trips -- pack and
          // unpack cancel -- but feeds x_embedder a permuted 64-vector, which
          // shows up as a 2x2 lattice over an otherwise correct image.)
          d[t * IC + ((std::size_t)ph * P + pw) * Z + cc] =
              f32_to_bf16_(r.chw[((std::size_t)cc * r.h + y) * r.w + x]);
        }
      }
    }
    genai::MetalBooguTransformer::RefImage img;
    img.latents = std::move(rb);
    img.seq = rseq; img.grid_h = r.h; img.grid_w = r.w;
    bri.push_back(std::move(img));
  }
  if (!bri.empty()) {
    session()->info(fmt(
        "GenerateImageStage('{}'): Boogu-Image editing on {} reference image(s)",
        this->id(), bri.size()));
  }

  // Classifier-free guidance (Base/Edit only -- the Turbo students are
  // distilled to guidance 1 and the DMD path forbids it outright). Boogu's
  // reference pipeline actually runs a THREE-way guidance (text, image and
  // empty-instruction scales); this is the single-negative reduction the other
  // families here use, i.e. v = v_neg + scale * (v_pos - v_neg).
  const bool cfg = !dmd && !txt_neg.empty() && n_real_neg > 0 &&
                   _guidance_scale != 1.0;
  if (dmd && _guidance_scale != 1.0) {
    session()->warn(fmt(
        "GenerateImageStage('{}'): the DMD student is distilled to guidance 1; "
        "ignoring guidance_scale {}", this->id(), _guidance_scale));
  }
  const float gscale = (float)_guidance_scale;

  bool dit_ok = true;
  // Opened before the denoise callable so the per-block hook is
  // live for the very first forward.
  UiProgress bar = session()->open_progress("denoise");
  DenoiseProgress prog(&bar, S, cfg ? 2 : 1);
  ScopedBlockProgress<std::remove_reference_t<decltype(*_boogu_dit)>>
      prog_guard(_boogu_dit.get(), prog);
  auto velocity = [&](const std::vector<float>& cand,
                      double sigma) -> std::vector<float> {
    auto* lb = static_cast<std::uint16_t*>(latbuf.contents());
    for (std::size_t k = 0; k < cand.size(); ++k) {
      lb[k] = f32_to_bf16_(cand[k]);
    }
    SharedBuffer vel = _boogu_dit->forward_dit(txt_pos, n_real, latbuf, img_seq,
                                               lh, lw, (float)sigma, bri);
    prog.end_forward();
    if (vel.empty()) { dit_ok = false; return {}; }
    const auto* vp = static_cast<const std::uint16_t*>(vel.contents());
    std::vector<float> v((std::size_t)img_seq * OC);
    for (std::size_t k = 0; k < v.size(); ++k) { v[k] = bf16_to_f32_(vp[k]); }
    if (cfg) {
      SharedBuffer vn = _boogu_dit->forward_dit(txt_neg, n_real_neg, latbuf,
                                                img_seq, lh, lw, (float)sigma,
                                                bri);
      prog.end_forward();
      if (vn.empty()) { dit_ok = false; return {}; }
      const auto* np = static_cast<const std::uint16_t*>(vn.contents());
      for (std::size_t k = 0; k < v.size(); ++k) {
        const float neg = bf16_to_f32_(np[k]);
        v[k] = neg + gscale * (v[k] - neg);
      }
    }
    return v;
  };

  // Unpack [img_seq, IC] -> channel-first latent [Z, lh, lw], the inverse of
  // the "(h w) (p1 p2 c)" pack above (channel fastest inside a token).
  auto unpack = [&](const std::vector<float>& pk) {
    std::vector<float> latent((std::size_t)Z * lh * lw);
    for (int c = 0; c < Z; ++c) {
      for (int y = 0; y < lh; ++y) {
        for (int xx = 0; xx < lw; ++xx) {
          const int a = y / P, ph = y % P, b = xx / P, pw = xx % P;
          const std::size_t t = (std::size_t)a * gw + b;
          latent[((std::size_t)c * lh + y) * lw + xx] =
              pk[t * IC + ((std::size_t)ph * P + pw) * Z + c];
        }
      }
    }
    return latent;
  };

  const bool prof = std::getenv("VPIPE_BOOGU_PROFILE") != nullptr;
  double dit_ms = 0.0;
  int dit_calls = 0;
  std::mt19937_64 renoise_rng(samp.seed != 0 ? samp.seed : _seed + 1);
  std::normal_distribution<float> rnd(0.0f, 1.0f);
  const auto gen_t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < S; ++i) {
    const double s = sig[(std::size_t)i];
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<float> v = velocity(packed, s);
    if (prof) {
      dit_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
      ++dit_calls;
    }
    if (!dit_ok) { return {}; }
    if (dmd) {
      // Jump to the x0 prediction, then re-noise back down to the next sigma.
      for (std::size_t k = 0; k < packed.size() && k < v.size(); ++k) {
        packed[k] += (float)((1.0 - s) * (double)v[k]);
      }
      if (i + 1 < S) {
        const double s1 = sig[(std::size_t)i + 1];
        for (auto& x : packed) {
          x = (float)((1.0 - s1) * (double)rnd(renoise_rng) + s1 * (double)x);
        }
      }
    } else {
      const double dt = sig[(std::size_t)i + 1] - s;
      for (std::size_t k = 0; k < packed.size() && k < v.size(); ++k) {
        packed[k] += (float)(dt * (double)v[k]);
      }
    }
    if (emit_step) { emit_step(unpack(packed)); }
    prog.end_step(i);
  }
  const double gen_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - gen_t0).count();
  session()->info(fmt(
      "GenerateImageStage('{}'): latent generated in {:.2f}s ({} {} steps, "
      "{} ms/step)", this->id(), gen_s, S, dmd ? "DMD" : "Euler",
      S ? (long)(gen_s * 1000.0 / S) : 0));
  if (prof) {
    session()->log_normal(fmt(
        "GenerateImageStage('{}'): Boogu DiT {} forward calls, {} ms total, {} "
        "ms/call (txt {} + img {} = {})", this->id(), dit_calls, (long)dit_ms,
        dit_calls ? (long)(dit_ms / dit_calls) : 0, n_real, img_seq,
        n_real + img_seq));
  }
  return unpack(packed);
}

std::vector<float>
GenerateImageStage::generate_qie_(const metal_compute::SharedBuffer& txt_pos,
                               int n_real,
                               const metal_compute::SharedBuffer& txt_neg,
                               int n_real_neg, int gen_h, int gen_w,
                               const std::vector<float>* init_packed,
                               const std::vector<RefLatent>& refs,
                               const std::function<void(
                                   const std::vector<float>&)>& emit_step)
    const
{
  auto* mc = session()->services()->metal_compute();
  using metal_compute::SharedBuffer;
  const int H = gen_h, W = gen_w;
  const int lh = H / 8, lw = W / 8;          // latent H/W (z_dim 16)
  const int gh = H / 16, gw = W / 16;        // 2x2-patch grid
  const int img_seq = gh * gw;
  const int IC = _qie_dit->config().in_channels;   // 64 = 16 * 2 * 2

  // `txt_pos` is the diffusion-conditioner's qwen-image-edit conditioning: the
  // image-aware last-hidden [n_real, 3584] bf16, already POST encoder final-norm
  // (the conditioner ran the vision tower + splice + drop-64 + final-RMSNorm).
  session()->log_debug(fmt(
      "GenerateImageStage('{}'): QIE conditioning [{}, txt]; {}x{} grid {}x{} "
      "img_seq {}", this->id(), n_real, W, H, gh, gw, img_seq));

  // norm-preserving true-CFG: use the negative conditioning too and, per step,
  // comb = neg + scale*(pos-neg), then rescale comb to pos's per-token norm.
  const bool cfg = !txt_neg.empty() && n_real_neg > 0 && _guidance_scale != 1.0;

  // Sampler: FlowMatchEuler with the QIE dynamic shift (mu from img_seq, terminal
  // stretch 0.02). Operator-supplied scheduler beats win (skip the override).
  genai::FlowSchedulerSpec sched = _scheduler_spec;
  if (!_scheduler_latched) {
    sched.dynamic_shift  = true;
    sched.base_shift     = 0.5;
    sched.max_shift      = 0.9;
    sched.shift_terminal = 0.02;
    sched.base_seq       = 256;
    sched.max_seq        = 8192;
    sched.num_train      = 1000;
    sched.shift_type     = "exponential";
    sched.img_seq_len    = img_seq;
  } else {
    sched.img_seq_len    = img_seq;
  }
  genai::FlowSampler sampler(_sampler_spec, sched);
  const int S = sampler.steps();

  // Packed latents [img_seq, IC]: a supplied init (repro / golden) or pure noise.
  std::vector<float> packed((std::size_t)img_seq * IC);
  if (init_packed != nullptr && init_packed->size() == packed.size()) {
    packed = *init_packed;
  } else {
    std::mt19937_64 rng(_seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : packed) { v = nd(rng); }
  }
  SharedBuffer latbuf = mc->make_shared_buffer((std::size_t)img_seq * IC * 2);
  if (latbuf.empty()) { return {}; }

  // Reference conditioning: each ref latent arrives channel-first [16, rlh, rlw]
  // (vae-encode output); pack it 2x2 into token-major [rseq, IC] bf16 (the same
  // patchify the generated latent uses) so the DiT embeds it via img_in in its
  // own RoPE frame band. Odd-dim / non-16-channel refs are skipped.
  //
  // The DiT's scale_rope centers each reference's h/w positions at 0 (matching
  // diffusers), so a reference grid SMALLER than the generated grid (gh x gw)
  // covers only the centered sub-region and leaves a visible rectangular "box"
  // at its boundary. The fix is to encode the reference at the OUTPUT resolution
  // (so ref grid == gen grid == full overlap, as the diffusers pipeline does) --
  // we keep the reference at native resolution (re-gridding the latent here just
  // blurs it) and WARN when the grids differ so the user can match them.
  std::vector<genai::MetalQwenImageTransformer::RefImage> ri;
  for (const auto& r : refs) {
    if (r.empty() || r.c != 16 || (r.h % 2) != 0 || (r.w % 2) != 0) {
      if (!r.empty()) {
        session()->warn(fmt(
            "GenerateImageStage('{}'): reference latent [{}, {}, {}] must be "
            "16-channel with even H/W; ignoring", this->id(), r.c, r.h, r.w));
      }
      continue;
    }
    const int rgh = r.h / 2, rgw = r.w / 2, rseq = rgh * rgw;
    if (rgh != gh || rgw != gw) {
      session()->warn(fmt(
          "GenerateImageStage('{}'): reference grid {}x{} != output grid {}x{} -- "
          "the centered reference will cover only part of the output and leave a "
          "rectangular artifact. Encode the reference at the output resolution "
          "(set vae-encode target to {}x{}) to avoid it.",
          this->id(), rgw, rgh, gw, gh, W, H));
    }
    SharedBuffer rb = mc->make_shared_buffer((std::size_t)rseq * IC * 2);
    if (rb.empty()) { continue; }
    auto* d = static_cast<std::uint16_t*>(rb.contents());
    std::memset(d, 0, rb.byte_size());
    for (int cc = 0; cc < 16; ++cc) {
      for (int y = 0; y < r.h; ++y) {
        for (int x = 0; x < r.w; ++x) {
          const int a = y / 2, ph = y % 2, bcol = x / 2, pw = x % 2;
          const std::size_t t = (std::size_t)a * rgw + bcol;
          d[t * IC + (std::size_t)cc * 4 + ph * 2 + pw] =
              f32_to_bf16_(r.chw[((std::size_t)cc * r.h + y) * r.w + x]);
        }
      }
    }
    genai::MetalQwenImageTransformer::RefImage img;
    img.latents = std::move(rb);
    img.seq = rseq; img.grid_h = rgh; img.grid_w = rgw;
    ri.push_back(std::move(img));
  }
  if (!ri.empty()) {
    session()->info(fmt(
        "GenerateImageStage('{}'): Qwen-Image-Edit conditioning on {} reference "
        "image(s)", this->id(), ri.size()));
  }

  // Denoise callback: upload the candidate, run the dual-stream DiT at `sigma`,
  // read the velocity back; apply norm-preserving true-CFG when enabled.
  bool dit_ok = true;
  const float gscale = (float)_guidance_scale;
  // Opened before the denoise callable so the per-block hook is
  // live for the very first forward.
  UiProgress bar = session()->open_progress("denoise");
  DenoiseProgress prog(&bar, S, cfg ? 2 : 1);
  ScopedBlockProgress<std::remove_reference_t<decltype(*_qie_dit)>>
      prog_guard(_qie_dit.get(), prog);
  auto denoise = [&](const std::vector<float>& cand,
                     double sigma) -> std::vector<float> {
    auto* lb = static_cast<std::uint16_t*>(latbuf.contents());
    for (std::size_t k = 0; k < cand.size(); ++k) {
      lb[k] = f32_to_bf16_(cand[k]);
    }
    SharedBuffer vel = _qie_dit->forward(latbuf, img_seq, txt_pos, n_real, gh,
                                         gw, (float)sigma, ri);
    prog.end_forward();
    if (vel.empty()) { dit_ok = false; return {}; }
    const auto* vp = static_cast<const std::uint16_t*>(vel.contents());
    std::vector<float> v(cand.size());
    for (std::size_t k = 0; k < v.size(); ++k) { v[k] = bf16_to_f32_(vp[k]); }
    if (cfg) {
      SharedBuffer veln = _qie_dit->forward(latbuf, img_seq, txt_neg, n_real_neg,
                                            gh, gw, (float)sigma, ri);
      prog.end_forward();
      if (veln.empty()) { dit_ok = false; return {}; }
      const auto* np = static_cast<const std::uint16_t*>(veln.contents());
      // Per-token (over IC channels): comb = neg + scale*(pos-neg), then
      // rescale comb to preserve pos's L2 norm (diffusers true-CFG).
      for (int t = 0; t < img_seq; ++t) {
        double npos = 0.0, ncomb = 0.0;
        std::vector<float> comb((std::size_t)IC);
        for (int c = 0; c < IC; ++c) {
          const std::size_t k = (std::size_t)t * IC + c;
          const float vneg = bf16_to_f32_(np[k]);
          const float cb = vneg + gscale * (v[k] - vneg);
          comb[(std::size_t)c] = cb;
          npos += (double)v[k] * v[k];
          ncomb += (double)cb * cb;
        }
        const double s = ncomb > 0.0 ? std::sqrt(npos / ncomb) : 1.0;
        for (int c = 0; c < IC; ++c) {
          v[(std::size_t)t * IC + c] = (float)(comb[(std::size_t)c] * s);
        }
      }
    }
    return v;
  };

  sampler.reset();
  // Per-step DiT timing (VPIPE_QIE_PROFILE), mirroring the Krea-2 / FLUX.2
  // paths: time each forward + log a ms/call summary. Pair with
  // VPIPE_QIE_DIT_PROFILE for the per-section breakdown inside a step.
  const bool prof = std::getenv("VPIPE_QIE_PROFILE") != nullptr;
  double dit_ms = 0.0;
  int dit_calls = 0;
  auto denoise_p = [&](const std::vector<float>& cand, double sigma) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<float> v = denoise(cand, sigma);
    dit_ms += std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t0).count();
    ++dit_calls;
    return v;
  };
  // Unpack [img_seq, IC] -> channel-first latent [16, lh, lw] (2x2). Used
  // per-step (debug step_latents) and for the final return.
  auto unpack = [&](const std::vector<float>& pk) {
    std::vector<float> latent((std::size_t)16 * lh * lw);
    for (int c = 0; c < 16; ++c) {
      for (int y = 0; y < lh; ++y) {
        for (int xx = 0; xx < lw; ++xx) {
          const int a = y / 2, ph = y % 2, b = xx / 2, pw = xx % 2;
          const std::size_t t = (std::size_t)a * gw + b;
          latent[((std::size_t)c * lh + y) * lw + xx] =
              pk[t * IC + (std::size_t)c * 4 + ph * 2 + pw];
        }
      }
    }
    return latent;
  };
  // Debug: VPIPE_QIE_DUMP_LATENT=<prefix> writes the raw PACKED latent
  // [img_seq, IC] f32 after each step as <prefix>NN.f32. Unlike the oport1
  // step stream this skips the VAE and the 8-bit PNG, so a golden comparison
  // sees the sampler state exactly -- needed because the early-step state is
  // dominated by the shared init noise, which masks the velocity error ~31x.
  const char* dump_pre = std::getenv("VPIPE_QIE_DUMP_LATENT");
  const auto gen_t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < S; ++i) {
    sampler.step(i, packed,
                 prof ? genai::FlowSampler::DenoiseFn(denoise_p) : denoise);
    if (!dit_ok) { return {}; }
    if (dump_pre != nullptr && *dump_pre != '\0') {
      char nm[32];
      std::snprintf(nm, sizeof nm, "%02d.f32", i + 1);
      std::ofstream df(std::string(dump_pre) + nm, std::ios::binary);
      if (df) {
        df.write(reinterpret_cast<const char*>(packed.data()),
                 (std::streamsize)(packed.size() * sizeof(float)));
      }
    }
    if (emit_step) { emit_step(unpack(packed)); }
    prog.end_step(i);
  }
  const double gen_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - gen_t0).count();
  session()->info(fmt(
      "GenerateImageStage('{}'): latent generated in {:.2f}s ({} denoise steps, "
      "{} ms/step)", this->id(), gen_s, S,
      S ? (long)(gen_s * 1000.0 / S) : 0));
  if (prof) {
    session()->log_normal(fmt(
        "GenerateImageStage('{}'): QIE DiT {} forward calls, {} ms total, {} "
        "ms/call (txt {} + img {} = {})", this->id(), dit_calls, (long)dit_ms,
        dit_calls ? (long)(dit_ms / dit_calls) : 0, n_real, img_seq,
        n_real + img_seq));
  }
  return unpack(packed);
}

std::vector<float>
GenerateImageStage::generate_mage_(const metal_compute::SharedBuffer& txt_pos,
                                 int n_real,
                                 const metal_compute::SharedBuffer& txt_neg,
                                 int n_real_neg, int gen_h, int gen_w,
                                 const std::vector<float>* init_packed,
                                 const std::vector<RefLatent>& refs,
                                 const std::function<void(
                                     const std::vector<float>&)>& emit_step)
    const
{
  auto* mc = session()->services()->metal_compute();
  using metal_compute::SharedBuffer;
  const int IC = _mage_dit->config().in_channels;   // 128
  // MageVAE downsamples 16x and the DiT patch_size is 1, so the latent grid IS
  // the token grid -- there is no 2x2 pack/unpack anywhere on this path (the
  // reason the Qwen-Image transformer could be reused verbatim is that the
  // packing lived in the caller, not the DiT).
  const int gh = gen_h / 16, gw = gen_w / 16;
  const int img_seq = gh * gw;
  if (img_seq <= 0) { return {}; }

  session()->log_debug(fmt(
      "GenerateImageStage('{}'): Mage-Flow conditioning [{} rows]; {}x{} grid "
      "{}x{} img_seq {}", this->id(), n_real, gen_w, gen_h, gh, gw, img_seq));

  const bool cfg = !txt_neg.empty() && n_real_neg > 0 && _guidance_scale != 1.0;

  // Sampler: FlowMatchEuler over linspace(1, 1/S, S) with the checkpoint's
  // STATIC shift (scheduler_config.json shift 6.0, use_dynamic_shifting
  // false). vpipe's "linear" time-shift IS the diffusers static-shift curve
  // mu*s/(1 + (mu-1)*s), so shift_type must be "linear" here, NOT the
  // exponential default. Turbo is 4 steps at cfg 1.0 (no negative branch).
  // An operator-supplied scheduler beat still wins.
  genai::FlowSchedulerSpec sched = _scheduler_spec;
  if (!_scheduler_latched) {
    sched.dynamic_shift = false;
    sched.type          = "simple";
    sched.shift         = _mage_shift;
    sched.shift_type    = "linear";
  }
  genai::FlowSampler sampler(_sampler_spec, sched);
  const int S = sampler.steps();

  // Packed target latents [img_seq, IC]: a supplied init (repro / golden),
  // else Gaussian-Shading WATERMARKED noise (the reference's encode_noise),
  // else plain noise when the watermark is disabled. The watermark forces each
  // entry into a key-chosen half-plane and randomizes the magnitude within it,
  // so the sample is still exactly ~N(0,1) -- no quality cost -- and a
  // detector that inverts the flow ODE back to the noise can read the signs.
  std::vector<float> packed((std::size_t)img_seq * IC);
  if (init_packed != nullptr && init_packed->size() == packed.size()) {
    packed = *init_packed;   // pinned noise: never watermarked (repro/golden)
  } else if (_wm_params.enabled) {
    // encode_noise lays the mark out CHANNEL-first (the order the detector
    // reshapes into) and returns it token-major, which is what the DiT wants.
    packed = genai::mage_wm::encode_noise(
        IC, gh, gw, genai::mage_wm::resolve_key(_wm_params.key), _seed);
    if (packed.size() != (std::size_t)img_seq * IC) {
      session()->warn(fmt(
          "GenerateImageStage('{}'): watermark noise generation failed; falling "
          "back to plain noise (NO provenance mark)", this->id()));
      packed.assign((std::size_t)img_seq * IC, 0.0f);
      std::mt19937_64 rng(_seed);
      std::normal_distribution<float> nd(0.0f, 1.0f);
      for (auto& v : packed) { v = nd(rng); }
    }
  } else {
    session()->log_debug(fmt(
        "GenerateImageStage('{}'): watermark DISABLED -- this image carries no "
        "provenance mark", this->id()));
    std::mt19937_64 rng(_seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : packed) { v = nd(rng); }
  }
  SharedBuffer latbuf = mc->make_shared_buffer((std::size_t)img_seq * IC * 2);
  if (latbuf.empty()) { return {}; }

  // Reference conditioning: each MageVAE reference latent arrives channel-first
  // [128, rh, rw]; transpose to token-major [rh*rw, 128] bf16 so the DiT embeds
  // it via img_in in its own RoPE frame band (frame = index + 1). References
  // stay CLEAN (never noised) and the sampler steps only the target tokens.
  std::vector<genai::MetalMageFlowTransformer::RefImage> ri;
  for (const auto& r : refs) {
    if (r.empty()) { continue; }
    if (r.c != IC || r.h <= 0 || r.w <= 0) {
      session()->warn(fmt(
          "GenerateImageStage('{}'): reference latent [{}, {}, {}] must be "
          "{}-channel; ignoring", this->id(), r.c, r.h, r.w, IC));
      continue;
    }
    // scale_rope centers each segment's h/w positions at 0, so a reference
    // grid smaller than the target's covers only a centered sub-region (the
    // same trap the Qwen-Image-Edit path warns about).
    if (r.h != gh || r.w != gw) {
      session()->warn(fmt(
          "GenerateImageStage('{}'): reference grid {}x{} != output grid {}x{} -- "
          "the centered reference will cover only part of the output. Encode "
          "the reference at the output resolution (set the vae-encode target "
          "to {}x{}).", this->id(), r.w, r.h, gw, gh, gen_w, gen_h));
    }
    const int rseq = r.h * r.w;
    SharedBuffer rb = mc->make_shared_buffer((std::size_t)rseq * IC * 2);
    if (rb.empty()) { continue; }
    auto* d = static_cast<std::uint16_t*>(rb.contents());
    for (int cc = 0; cc < IC; ++cc) {
      for (int t = 0; t < rseq; ++t) {
        d[(std::size_t)t * IC + cc] =
            f32_to_bf16_(r.chw[(std::size_t)cc * rseq + t]);
      }
    }
    genai::MetalMageFlowTransformer::RefImage img;
    img.latents = std::move(rb);
    img.seq = rseq; img.grid_h = r.h; img.grid_w = r.w;
    ri.push_back(std::move(img));
  }
  if (!ri.empty()) {
    session()->info(fmt(
        "GenerateImageStage('{}'): Mage-Flow edit conditioning on {} reference "
        "image(s)", this->id(), ri.size()));
  }

  bool dit_ok = true;
  const float gscale = (float)_guidance_scale;
  // Opened before the denoise callable so the per-block hook is
  // live for the very first forward.
  UiProgress bar = session()->open_progress("denoise");
  DenoiseProgress prog(&bar, S, cfg ? 2 : 1);
  ScopedBlockProgress<std::remove_reference_t<decltype(*_mage_dit)>>
      prog_guard(_mage_dit.get(), prog);
  auto denoise = [&](const std::vector<float>& cand,
                     double sigma) -> std::vector<float> {
    auto* lb = static_cast<std::uint16_t*>(latbuf.contents());
    for (std::size_t k = 0; k < cand.size(); ++k) {
      lb[k] = f32_to_bf16_(cand[k]);
    }
    SharedBuffer vel = _mage_dit->forward(latbuf, img_seq, txt_pos, n_real, gh,
                                          gw, (float)sigma, ri);
    prog.end_forward();
    if (vel.empty()) { dit_ok = false; return {}; }
    const auto* vp = static_cast<const std::uint16_t*>(vel.contents());
    std::vector<float> v(cand.size());
    for (std::size_t k = 0; k < v.size(); ++k) { v[k] = bf16_to_f32_(vp[k]); }
    if (cfg) {
      SharedBuffer veln = _mage_dit->forward(latbuf, img_seq, txt_neg,
                                             n_real_neg, gh, gw, (float)sigma,
                                             ri);
      prog.end_forward();
      if (veln.empty()) { dit_ok = false; return {}; }
      const auto* np = static_cast<const std::uint16_t*>(veln.contents());
      // Plain CFG: unc + scale*(cond - unc). The reference's CFG-renorm
      // (pipeline.py `renormalization`) defaults OFF -- unlike Qwen-Image-Edit,
      // whose path always norm-preserves.
      for (std::size_t k = 0; k < v.size(); ++k) {
        const float vneg = bf16_to_f32_(np[k]);
        v[k] = vneg + gscale * (v[k] - vneg);
      }
    }
    return v;
  };

  const bool prof = std::getenv("VPIPE_MAGE_PROFILE") != nullptr;
  double dit_ms = 0.0;
  int dit_calls = 0;
  auto denoise_p = [&](const std::vector<float>& cand, double sigma) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<float> v = denoise(cand, sigma);
    dit_ms += std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t0).count();
    ++dit_calls;
    return v;
  };
  // [img_seq, IC] token-major -> [IC, gh, gw] channel-first: a transpose, with
  // no patch to undo.
  auto unpack = [&](const std::vector<float>& pk) {
    std::vector<float> latent((std::size_t)IC * img_seq);
    for (int c = 0; c < IC; ++c) {
      for (int t = 0; t < img_seq; ++t) {
        latent[(std::size_t)c * img_seq + t] = pk[(std::size_t)t * IC + c];
      }
    }
    return latent;
  };
  const auto gen_t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < S; ++i) {
    sampler.step(i, packed,
                 prof ? genai::FlowSampler::DenoiseFn(denoise_p) : denoise);
    if (!dit_ok) { return {}; }
    if (emit_step) { emit_step(unpack(packed)); }
    prog.end_step(i);
  }
  const double gen_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - gen_t0).count();
  session()->info(fmt(
      "GenerateImageStage('{}'): latent generated in {:.2f}s ({} denoise steps, "
      "{} ms/step)", this->id(), gen_s, S,
      S ? (long)(gen_s * 1000.0 / S) : 0));
  if (prof) {
    session()->log_normal(fmt(
        "GenerateImageStage('{}'): Mage-Flow DiT {} forward calls, {} ms total, "
        "{} ms/call (txt {} + img {} = {})", this->id(), dit_calls,
        (long)dit_ms, dit_calls ? (long)(dit_ms / dit_calls) : 0, n_real,
        img_seq, n_real + img_seq));
  }
  return unpack(packed);
}


void
GenerateImageStage::tag_model_(TensorBeat& tb) const
{
  // `_hf_dir` is the reference the user actually named (a registry key like
  // "local/Mage-Flow-Edit-Turbo-8bit", or a path), which is the meaningful
  // identity of the generator -- not the resolved directory.
  if (_hf_dir.empty()) { return; }
  FlexData o = tb.sideband.is_object() ? tb.sideband : FlexData::make_object();
  o.as_object().insert_or_assign("model_name",
                                 FlexData::make_string(_hf_dir));
  tb.sideband = std::move(o);
}

Job
GenerateImageStage::process(RuntimeContext& ctx)
{
  auto* mc = session()->services()->metal_compute();
  // The model config FIRST, before anything can load: `klein_kv` is an
  // argument to the DiT's construction, not a per-step knob, so a beat
  // read after ensure_loaded_ would arrive too late to matter and would
  // do so silently.
  //
  // Latched, but RE-READ whenever another beat is waiting: a config
  // source with no trigger emits once for the whole run, while a
  // trigger-driven one emits per request. Blocking on the first and
  // polling after serves both. (A later beat cannot change `klein_kv` on
  // an already-built DiT -- see the note where that is reported.)
  if (ctx.num_iports() > kModelCfgPort && ctx.iport_connected(kModelCfgPort) &&
      (!_cfg_latched || ctx.backlog(kModelCfgPort) > 0)) {
    auto gb = co_await ctx.read(kModelCfgPort);
    const bool first = !_cfg_latched;
    _cfg_latched = true;
    if (const auto* gfd =
            gb ? dynamic_cast<const FlexDataPayload*>(gb.get()) : nullptr) {
      const bool kv_before = _flux2_params.klein_kv;
      _model_cfg = gfd->data;
      // Only if the family is already known; otherwise ensure_loaded_
      // applies it the moment it is.
      if (_load_attempted) { apply_model_config_(); }
      if (!first && _family == "flux2" &&
          _flux2_params.klein_kv != kv_before) {
        session()->warn(fmt(
            "GenerateImageStage('{}'): klein_kv changed after the DiT was "
            "built; it selects the attention recipe at LOAD time, so this "
            "run keeps the old one. Restart the pipeline to change it",
            this->id()));
        _flux2_params.klein_kv = kv_before;
      }
    }
  }
  // Latch the shared model (iport2) once -- a model-select source overrides the
  // hf_dir config -- then lazily load the DiT before the first generation.
  if (!_model_latched && ctx.num_iports() > kModelPort &&
      ctx.iport_connected(kModelPort)) {
    auto mb = co_await ctx.read(kModelPort);
    _model_latched = true;
    if (const auto* mfd =
            mb ? dynamic_cast<const FlexDataPayload*>(mb.get()) : nullptr) {
      if (apply_model_select_beat(mfd->data, _hf_dir)) {
        ensure_loaded_();
      }
    }
  }
  // A config wired but no model-select source: initialize() deferred the
  // load so the config could be read first, so nothing has loaded yet.
  ensure_loaded_();
  auto in = co_await ctx.read(0);
  if (!in) { ctx.signal_done(); co_return; }
  // Did the conditioner REFUSE this prompt? Read the flag off the beat up
  // front; the refusal itself is emitted further down, once the output size
  // is known.
  bool cond_blocked = false;
  if (const auto* stb = dynamic_cast<const TensorBeatPayload*>(in.get())) {
    if (stb->sideband.is_object()) {
      FlexData sb = stb->sideband;        // as_object() is a view: keep it
      auto o = sb.as_object();
      cond_blocked = o.contains("content_blocked")
                     && o.at("content_blocked").as_bool(false);
    }
  }
  // A prior generation may have freed the FLUX.2 DiT to make room for the
  // vae-decode on a memory-bounded box; reload it now a new prompt arrived.
  if (_family == "flux2" && _dit_unloaded && !_flux2_dit) {
    session()->info(fmt(
        "GenerateImageStage('{}'): reloading the FLUX.2 DiT for the next prompt",
        this->id()));
    if (load_flux2_dit_()) { _dit_unloaded = false; }
  }
  if (_family == "krea2" && _dit_unloaded && !_dit) {
    session()->info(fmt(
        "GenerateImageStage('{}'): reloading the Krea-2 DiT for the next prompt",
        this->id()));
    if (load_krea2_dit_()) { _dit_unloaded = false; }
  }
  if (_family == "boogu-image" && _dit_unloaded && !_boogu_dit) {
    session()->info(fmt(
        "GenerateImageStage('{}'): reloading the Boogu-Image DiT for the next "
        "prompt", this->id()));
    if (load_boogu_dit_()) { _dit_unloaded = false; }
  }
  const bool have_dit = _family == "flux2" ? (bool)_flux2_dit
      : _family == "qwen-image-edit" ? (bool)_qie_dit
      : _family == "mage-flow" ? (bool)_mage_dit
      : _family == "boogu-image" ? (bool)_boogu_dit
      : (bool)_dit;
  if (!have_dit) {
    session()->warn(fmt(
        "GenerateImageStage('{}'): models not loaded; dropping beat", this->id()));
    co_return;
  }
  // iport0: the conditioning tensor from a diffusion-conditioner stage
  // (family-shaped + typed; rows = shape[0]). Copy it into a metal buffer.
  const auto* ctb = dynamic_cast<const TensorBeatPayload*>(in.get());
  if (ctb == nullptr || ctb->shape.empty() || ctb->shape[0] <= 0) {
    session()->warn(fmt(
        "GenerateImageStage('{}'): expected a conditioning TensorBeat, got {}; "
        "dropping beat", this->id(), in->describe()));
    co_return;
  }
  metal_compute::SharedBuffer cond = cond_to_shared_(mc, *ctb);
  if (cond.empty() && !cond_blocked) {
    session()->warn(fmt(
        "GenerateImageStage('{}'): conditioning upload failed; dropping beat",
        this->id()));
    co_return;
  }
  const int n_real = (int)ctb->shape[0];
  session()->log_debug(fmt(
      "GenerateImageStage('{}'): conditioning beat [{} rows, {}]", this->id(),
      n_real, ctb->dtype == TensorBeat::DType::Bf16 ? "bf16" : "f16"));

  // iport1: OPTIONAL negative conditioning (the conditioner's oport1) for
  // classifier-free guidance. The conditioner enqueues oport1 BEFORE oport0, so
  // when iport0 arrives its paired negative is already in this port's FIFO --
  // the non-blocking backlog gate reads it reliably (and never stalls when no
  // negative is wired).
  metal_compute::SharedBuffer cond_neg;
  int n_real_neg = 0;
  if (ctx.num_iports() >= 2 && ctx.iport_connected(1) && ctx.backlog(1) > 0) {
    auto nb = co_await ctx.read(1);
    const auto* ntb = nb ? dynamic_cast<const TensorBeatPayload*>(nb.get())
                         : nullptr;
    if (ntb != nullptr && !ntb->shape.empty() && ntb->shape[0] > 0) {
      cond_neg = cond_to_shared_(mc, *ntb);
      if (!cond_neg.empty()) { n_real_neg = (int)ntb->shape[0]; }
      session()->log_debug(fmt(
          "GenerateImageStage('{}'): negative conditioning [{} rows]", this->id(),
          n_real_neg));
    }
  }

  // Optional debug/repro: packed initial latents from a raw f32 file.
  std::vector<float> init;
  const std::vector<float>* init_ptr = nullptr;
  if (!_init_latents.empty()) {
    std::ifstream f(_init_latents, std::ios::binary);
    if (f) {
      f.seekg(0, std::ios::end);
      const std::streamoff nb = f.tellg();
      f.seekg(0, std::ios::beg);
      init.resize((std::size_t)nb / 4);
      f.read(reinterpret_cast<char*>(init.data()), nb);
      init_ptr = &init;
    }
  }

  // Reference latents on iport5 / iport6: latch the FIRST reference on each
  // connected port (blocking, like the sampler/scheduler specs) and cache it
  // (`_ref[]`), so the reference reliably pairs with the prompt (a non-blocking
  // poll would race the producer) and a fixed reference supplied once is reused
  // for every later prompt. FLUX.2 threads them as multi-reference conditioning
  // tokens (below); Krea-2 uses ref latent 0 as the img2img init and ignores
  // ref latent 1.
  for (int r = 0; r < 2; ++r) {
    const int port = 5 + r;
    if ((int)ctx.num_iports() > port && ctx.iport_connected(port) &&
        _ref[r].empty()) {
      auto rb = co_await ctx.read(port);
      const auto* tb = rb ? dynamic_cast<const TensorBeatPayload*>(rb.get())
                          : nullptr;
      if (tb != nullptr && tb->dtype == TensorBeat::DType::F32 &&
          tb->shape.size() == 3 && tb->shape[0] > 0 && tb->shape[1] > 0 &&
          tb->shape[2] > 0) {
        const auto bytes = tb->materialize_contiguous();
        const std::size_t n =
            (std::size_t)tb->shape[0] * tb->shape[1] * tb->shape[2];
        const float* fp = reinterpret_cast<const float*>(bytes.data());
        _ref[r].chw.assign(fp, fp + n);
        _ref[r].c = (int)tb->shape[0];
        _ref[r].h = (int)tb->shape[1];
        _ref[r].w = (int)tb->shape[2];
        session()->log_debug(fmt(
            "GenerateImageStage('{}'): reference latent {} = [{}, {}, {}]",
            this->id(), r, _ref[r].c, _ref[r].h, _ref[r].w));
      } else if (rb) {
        session()->warn(fmt(
            "GenerateImageStage('{}'): ref_latent{} must be an f32 [C,H,W] "
            "TensorBeat; got {}, ignoring", this->id(), r, rb->describe()));
      }
    }
  }

  // Output size. An explicit width/height ALWAYS wins; only a completely
  // unconfigured size infers, and then from ref_latent0 -- vae-encode's output
  // for the source image -- times the family's VAE scale, so an edit comes out
  // at the source resolution with no size config at all.
  //
  // The multiple-of-16 test is the same one initialize() applies to a
  // configured size. It is vacuous for the 16x families and is the real
  // constraint for the 8x ones, whose DiTs patchify 2x2 and so need an EVEN
  // latent grid (Krea-2 and Qwen-Image-Edit both). An odd reference warns and
  // falls back rather than silently truncating the grid.
  int gen_h = _height, gen_w = _width;
  if (_infer_size) {
    const RefLatent& r0 = _ref[0];
    if (!r0.empty()) {
      const int s = latent_scale_(_family);
      const int ih = r0.h * s, iw = r0.w * s;
      if (ih % 16 == 0 && iw % 16 == 0) {
        gen_h = ih; gen_w = iw;
      } else {
        session()->warn(fmt(
            "GenerateImageStage('{}'): ref_latent0 [{}, {}, {}] implies {}x{}, "
            "not a multiple of 16 ({}x latent needs an even grid); using "
            "256x256 -- set width/height", this->id(), r0.c, r0.h, r0.w, iw,
            ih, s));
      }
    }
    if (gen_h <= 0 || gen_w <= 0) { gen_h = 256; gen_w = 256; }
    session()->log_debug(fmt(
        "GenerateImageStage('{}'): output size {}x{} ({})", this->id(), gen_w,
        gen_h, r0.empty() ? "no reference, default" : "from ref_latent0"));
  }
  // ---- Content-policy refusal ------------------------------------------
  // The Mage-Flow conditioner screens every prompt against the model's own
  // policy classifier and tags a refused one `content_blocked`. Honour it
  // HERE, once the output size is known and before any DiT work: a refusal
  // costs no denoise. The beat carries the size explicitly so vae-decode can
  // paint the blank refusal image without interpreting a latent that was
  // never generated.
  //
  // This is not family-gated: a `content_blocked` beat is refused whatever
  // produced it. Dropping the beat instead would stall a pipeline waiting on
  // this stage's oport0, so a refusal still emits.
  if (cond_blocked) {
    auto out = std::make_unique<TensorBeatPayload>();
    out->dtype = TensorBeat::DType::F32;
    out->shape = {1, 1, 1};
    out->resize_contiguous(1);
    out->as_f32()[0] = 0.0f;
    FlexData sb = FlexData::make_object();
    auto o = sb.as_object();
    o.insert_or_assign("content_blocked", FlexData::make_bool(true));
    o.insert_or_assign("refusal_height", FlexData::make_int(gen_h));
    o.insert_or_assign("refusal_width",  FlexData::make_int(gen_w));
    out->sideband = std::move(sb);
    ++_latents_emitted;
    session()->info(fmt(
        "GenerateImageStage('{}'): conditioning refused by the content policy; "
        "skipping the denoise and emitting a {}x{} refusal", this->id(),
        gen_w, gen_h));
    co_await ctx.write(0, std::move(out));
    co_return;
  }

  int lh = gen_h / 8, lw = gen_w / 8;
  std::vector<float> latent;
  const std::vector<float>* latent_ptr = nullptr;
  // Krea-2 img2img: ref latent 0 (the whitened [z_dim, H/8, W/8] vae-encode
  // output) is mixed into the noise at the strength-selected sigma. FLUX.2
  // skips this -- refs are conditioning tokens, not an init, handled in
  // generate_flux2_. When the size was inferred above, rr.h/rr.w match lh/lw
  // by construction.
  if (_family == "krea2" && _strength > 0.0 && !_ref[0].empty()) {
    const RefLatent& rr = _ref[0];
    const bool ok_type = rr.c == 16 && rr.h > 0 && rr.w > 0;
    if (ok_type && rr.h == lh && rr.w == lw) {
      latent = rr.chw;
      latent_ptr = &latent;
    } else {
      session()->warn(fmt(
          "GenerateImageStage('{}'): img2img init must be an f32 latent [16, {}, "
          "{}]; got [{}, {}, {}], ignoring", this->id(), lh, lw, rr.c, rr.h,
          rr.w));
    }
  }

  // Latch the sampler / scheduler specs off iport3 / iport4 (once each): the
  // `diffusion-sampler-select` / `scheduler-select` sources emit a single spec
  // beat, which
  // we cache and reuse for every subsequent prompt.
  if (!_sampler_latched && ctx.num_iports() >= 4 && ctx.iport_connected(3)) {
    auto sb = co_await ctx.read(3);
    _sampler_latched = true;
    const auto* sfd = dynamic_cast<const FlexDataPayload*>(sb.get());
    if (sfd != nullptr) {
      std::string serr;
      _sampler_spec = genai::FlowSamplerSpec::from_flex(sfd->data, &serr);
      if (!serr.empty()) {
        session()->warn(fmt("GenerateImageStage('{}'): sampler spec: {}",
                            this->id(), serr));
      }
      session()->info(fmt(
          "GenerateImageStage('{}'): sampler = {} (eta {}, s_noise {})",
          this->id(), _sampler_spec.method, _sampler_spec.eta,
          _sampler_spec.s_noise));
    }
  }
  if (!_scheduler_latched && ctx.num_iports() >= 5 && ctx.iport_connected(4)) {
    auto cb = co_await ctx.read(4);
    _scheduler_latched = true;
    const auto* cfd = dynamic_cast<const FlexDataPayload*>(cb.get());
    if (cfd != nullptr) {
      std::string cerr;
      _scheduler_spec = genai::FlowSchedulerSpec::from_flex(cfd->data, &cerr);
      if (!cerr.empty()) {
        session()->warn(fmt("GenerateImageStage('{}'): scheduler spec: {}",
                            this->id(), cerr));
      }
      session()->info(fmt(
          "GenerateImageStage('{}'): scheduler = {} ({} steps, shift {} {})",
          this->id(), _scheduler_spec.type, _scheduler_spec.steps,
          _scheduler_spec.shift, _scheduler_spec.shift_type));
    }
  }

  // ---- FLUX.2: text-to-image from noise (+ optional reference-image
  // conditioning from iport5/iport6) -> latent [dit_channels, H/16, W/16].
  // (No negative-prompt CFG; klein is distilled.) ----
  // step_latent (oport1): stream each denoise step's latent LIVE so a downstream
  // vae-decode -> preview updates every step. Emitted only when a consumer is
  // wired. `step_emitter(shape)` returns a per-step callback the generate_ loops
  // call right after each sampler step; it write_sync's the unpacked latent from
  // the (synchronous) denoise loop -- the runtime wakes the consumer on the
  // thread pool, so it decodes concurrently with the next DiT step. write_sync
  // never blocks the loop (drops if the ring is full) and stops on consumer
  // close (step_alive). Empty callback (no consumer) => zero overhead.
  const bool want_steps = ctx.num_oports() > 1 && ctx.has_consumers(1);
  bool step_alive = true;
  auto step_emitter = [&](std::vector<std::int64_t> shape)
      -> std::function<void(const std::vector<float>&)> {
    if (!want_steps) { return {}; }
    return [&, shape](const std::vector<float>& lat) {
      if (!step_alive) { return; }
      auto so = std::make_unique<TensorBeatPayload>();
      so->dtype = TensorBeat::DType::F32;
      so->shape = shape;
      so->resize_contiguous(lat.size());
      tag_model_(*so);
      std::memcpy(so->as_f32(), lat.data(), lat.size() * sizeof(float));
      step_alive = ctx.write_sync(1, std::move(so));
    };
  };
  // Cooperative stop: feed the active DiT a hook reporting the pipeline stop
  // flag so it abandons the forward within ~one block instead of running the
  // whole (multi-second at high res) step. Set around each generate_ call and
  // cleared after -- the callback captures ctx, valid only for this process().
  auto stopping = [&ctx]() { return ctx.stop_requested(); };

  if (_family == "flux2") {
    std::vector<RefLatent> frefs;
    if (!_ref[0].empty()) { frefs.push_back(_ref[0]); }
    if (!_ref[1].empty()) { frefs.push_back(_ref[1]); }
    const int Cdit = _flux2_dit->config().in_channels;
    const int fgh = gen_h / 16, fgw = gen_w / 16;
    _flux2_dit->set_stream_stop(stopping);
    const std::vector<float> fl =
        generate_flux2_(cond, n_real, gen_h, gen_w, frefs, init_ptr,
                        step_emitter({Cdit, fgh, fgw}));
    _flux2_dit->set_stream_stop({});
    if (fl.empty()) {
      session()->info(fmt(
          "GenerateImageStage('{}'): FLUX.2 generation {}; dropping beat",
          this->id(), ctx.stop_requested() ? "stopped" : "failed"));
      co_return;
    }
    auto out = std::make_unique<TensorBeatPayload>();
    out->dtype = TensorBeat::DType::F32;
    out->shape = {Cdit, fgh, fgw};
    out->resize_contiguous(fl.size());
    std::memcpy(out->as_f32(), fl.data(), fl.size() * sizeof(float));
    tag_model_(*out);
    ++_latents_emitted;
    session()->info(fmt(
        "GenerateImageStage('{}'): FLUX.2 latent [{}, {}, {}] ({} steps @ {}x{})",
        this->id(), Cdit, fgh, fgw, _scheduler_spec.steps, gen_h, gen_w));
    // Free the DiT if the downstream vae-decode won't fit alongside it (before
    // publishing the latent, so the decode stage sees the freed working set).
    free_flux2_dit_for_decode_(gen_w, gen_h);
    co_await ctx.write(0, std::move(out));
    co_return;
  }

  // ---- Qwen-Image-Edit: image-aware conditioning (from the conditioner) +
  // dual-stream DiT (reference latents from iport5/iport6 as DiT conditioning
  // tokens) + norm-preserving true-CFG -> whitened latent [16, H/8, W/8]. ----
  if (_family == "qwen-image-edit") {
    std::vector<RefLatent> qrefs;
    if (!_ref[0].empty()) { qrefs.push_back(_ref[0]); }
    if (!_ref[1].empty()) { qrefs.push_back(_ref[1]); }
    _qie_dit->set_stream_stop(stopping);
    const std::vector<float> ql =
        generate_qie_(cond, n_real, cond_neg, n_real_neg, gen_h, gen_w,
                      init_ptr, qrefs, step_emitter({16, lh, lw}));
    _qie_dit->set_stream_stop({});
    if (ql.empty()) {
      session()->info(fmt(
          "GenerateImageStage('{}'): Qwen-Image-Edit generation {}; dropping "
          "beat", this->id(), ctx.stop_requested() ? "stopped" : "failed"));
      co_return;
    }
    auto out = std::make_unique<TensorBeatPayload>();
    out->dtype = TensorBeat::DType::F32;
    out->shape = {16, lh, lw};
    out->resize_contiguous(ql.size());
    std::memcpy(out->as_f32(), ql.data(), ql.size() * sizeof(float));
    tag_model_(*out);
    ++_latents_emitted;
    session()->info(fmt(
        "GenerateImageStage('{}'): Qwen-Image-Edit latent [16, {}, {}] "
        "({} steps @ {}x{})", this->id(), lh, lw, _scheduler_spec.steps,
        gen_h, gen_w));
    co_await ctx.write(0, std::move(out));
    co_return;
  }

  // ---- Boogu-Image: single last-hidden conditioning (the WHOLE templated
  // sequence) + the NextDiT, with the DMD student's ascending-sigma loop.
  // The VAE is 8x and the DiT patches 2x2 itself, so the emitted latent is
  // [16, H/8, W/8] -- the same shape Qwen-Image-Edit emits. ----
  if (_family == "boogu-image") {
    std::vector<RefLatent> brefs;
    if (!_ref[0].empty()) { brefs.push_back(_ref[0]); }
    if (!_ref[1].empty()) { brefs.push_back(_ref[1]); }
    _boogu_dit->set_stream_stop(stopping);
    const std::vector<float> bl =
        generate_boogu_(cond, n_real, cond_neg, n_real_neg, gen_h, gen_w,
                        init_ptr, brefs, step_emitter({16, lh, lw}));
    _boogu_dit->set_stream_stop({});
    if (bl.empty()) {
      session()->info(fmt(
          "GenerateImageStage('{}'): Boogu-Image generation {}; dropping beat",
          this->id(), ctx.stop_requested() ? "stopped" : "failed"));
      co_return;
    }
    auto out = std::make_unique<TensorBeatPayload>();
    out->dtype = TensorBeat::DType::F32;
    out->shape = {16, lh, lw};
    out->resize_contiguous(bl.size());
    std::memcpy(out->as_f32(), bl.data(), bl.size() * sizeof(float));
    tag_model_(*out);
    ++_latents_emitted;
    session()->info(fmt(
        "GenerateImageStage('{}'): Boogu-Image latent [16, {}, {}] ({} steps "
        "@ {}x{})", this->id(), lh, lw, _scheduler_spec.steps, gen_h, gen_w));
    // Before publishing: on a bounded box the resident DiT would leave the
    // downstream vae-decode no working set. Freed here, reloaded on the next
    // prompt (the latent is already read back, so the DiT is idle).
    free_boogu_dit_for_decode_(gen_w, gen_h);
    co_await ctx.write(0, std::move(out));
    co_return;
  }

  // ---- Mage-Flow: single last-hidden conditioning + the 12-block NR-MMDiT.
  // MageVAE is 16x, so the latent grid is H/16 x W/16 (128 channels) and the
  // reference latents on iport5/iport6 ride along as clean edit-conditioning
  // segments. -> latent [128, H/16, W/16]. ----
  if (_family == "mage-flow") {
    std::vector<RefLatent> mrefs;
    if (!_ref[0].empty()) { mrefs.push_back(_ref[0]); }
    if (!_ref[1].empty()) { mrefs.push_back(_ref[1]); }
    const int mgh = gen_h / 16, mgw = gen_w / 16;
    const int MC = _mage_dit->config().in_channels;
    _mage_dit->set_stream_stop(stopping);
    const std::vector<float> ml =
        generate_mage_(cond, n_real, cond_neg, n_real_neg, gen_h, gen_w,
                       init_ptr, mrefs, step_emitter({MC, mgh, mgw}));
    _mage_dit->set_stream_stop({});
    if (ml.empty()) {
      session()->info(fmt(
          "GenerateImageStage('{}'): Mage-Flow generation {}; dropping beat",
          this->id(), ctx.stop_requested() ? "stopped" : "failed"));
      co_return;
    }
    auto out = std::make_unique<TensorBeatPayload>();
    out->dtype = TensorBeat::DType::F32;
    out->shape = {MC, mgh, mgw};
    out->resize_contiguous(ml.size());
    std::memcpy(out->as_f32(), ml.data(), ml.size() * sizeof(float));
    tag_model_(*out);
    ++_latents_emitted;
    session()->info(fmt(
        "GenerateImageStage('{}'): Mage-Flow latent [{}, {}, {}] ({} steps @ "
        "{}x{})", this->id(), MC, mgh, mgw, _scheduler_spec.steps, gen_h,
        gen_w));
    co_await ctx.write(0, std::move(out));
    co_return;
  }

  // Krea-2 reference conditioning (Qwen-Image-Edit multi-reference): the ref
  // latents NOT consumed by img2img. strength>0 uses ref0 as the img2img init,
  // so only ref1 conditions; strength==0 (pure text-to-image) conditions on
  // both. (Reference conditioning only steers a reference-trained checkpoint.)
  std::vector<RefLatent> krefs;
  {
    const int first = (latent_ptr != nullptr) ? 1 : 0;
    for (int i = first; i < 2; ++i) {
      if (!_ref[i].empty()) { krefs.push_back(_ref[i]); }
    }
  }
  _dit->set_stream_stop(stopping);
  const std::vector<float> out_latent =
      generate_(cond, n_real, cond_neg, n_real_neg, gen_h, gen_w, init_ptr,
                latent_ptr, krefs, step_emitter({16, lh, lw}));
  _dit->set_stream_stop({});
  if (out_latent.empty()) {
    session()->info(fmt(
        "GenerateImageStage('{}'): generation {}; dropping beat", this->id(),
        ctx.stop_requested() ? "stopped" : "failed"));
    co_return;
  }
  // The DiT is idle now (latent read back): on a memory-bounded box first drop
  // its per-forward scratch (DitScratch activations + dequant/split-K + i8
  // accel buffers, ~1-2 GB at 1024px), which regrows on the next generation.
  if (_release_scratch && _dit) { _dit->release_forward_scratch(); }
  // The DiT's ~7 GB of quantized weights are resident (wired), not actually
  // evictable, so dropping the scratch alone leaves a 1024px vae-decode short
  // of working set. If it still won't fit, free the whole DiT here (reloaded
  // lazily on the next prompt) BEFORE publishing the latent, so the separate
  // vae-decode stage sees the freed room.
  free_krea2_dit_for_decode_(gen_w, gen_h);

  auto out = std::make_unique<TensorBeatPayload>();
  out->dtype = TensorBeat::DType::F32;
  out->shape = {16, lh, lw};
  out->resize_contiguous(out_latent.size());
  std::memcpy(out->as_f32(), out_latent.data(),
              out_latent.size() * sizeof(float));
  tag_model_(*out);
  ++_latents_emitted;
  session()->info(fmt(
      "GenerateImageStage('{}'): latent [16, {}, {}] ({}+{} {} steps @ "
      "{}x{})", this->id(), lh, lw, _sampler_spec.method,
      _scheduler_spec.type, _scheduler_spec.steps, gen_h, gen_w));
  co_await ctx.write(0, std::move(out));
}

#else   // !VPIPE_BUILD_APPLE_SILICON

Job
GenerateImageStage::initialize(RuntimeContext& ctx)
{
  (void)ctx;
  if (session()) {
    session()->error(fmt(
        "GenerateImageStage('{}'): built without VPIPE_BUILD_APPLE_SILICON; "
        "inert", this->id()));
  }
  co_return;
}

Job
GenerateImageStage::process(RuntimeContext& ctx)
{
  auto in = co_await ctx.read(0);
  (void)in;
  ctx.signal_done();
  co_return;
}

#endif  // VPIPE_BUILD_APPLE_SILICON

VPIPE_REGISTER_STAGE(GenerateImageStage)
VPIPE_REGISTER_SPEC(GenerateImageStage, kSpec)

}

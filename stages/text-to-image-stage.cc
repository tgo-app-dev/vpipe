#include "stages/text-to-image-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "stages/model-registry.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#endif

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
  {.key = "hf_dir", .type = ConfigType::String, .required = true,
   .doc = "Krea-2-Turbo / FLUX.2 model dir (text_encoder/, transformer/, "
          "tokenizer/); an original or model-quantize'd (self-contained) "
          "pipeline",
   .suggest_db = "models", .suggest_db_type = "krea2,flux2,qwen-image-edit"},
  {.key = "dit_dir", .type = ConfigType::String, .required = false,
   .doc = "override DiT dir (e.g. a quantized 4/8-bit DiT); else <hf_dir>/transformer",
   .suggest_db = "models",
   .suggest_db_type = "krea2-dit,flux2-dit,qwen-image-edit-dit"},
  {.key = "strength", .type = ConfigType::Real, .required = false,
   .doc = "img2img strength in [0,1]; 0 (default) = text-to-image from noise "
          "(the init latent arrives on the `latent` iport from vae-encode)"},
  {.key = "height", .type = ConfigType::Int, .required = false,
   .doc = "output height, multiple of 16 (default 256)"},
  {.key = "width", .type = ConfigType::Int, .required = false,
   .doc = "output width, multiple of 16 (default 256)"},
  {.key = "steps", .type = ConfigType::Int, .required = false,
   .doc = "turbo sampler steps (default 8)"},
  {.key = "seed", .type = ConfigType::Int, .required = false,
   .doc = "initial-noise RNG seed (default 0)"},
  {.key = "guidance_scale", .type = ConfigType::Real, .required = false,
   .doc = "classifier-free guidance scale; 1 (default) disables CFG. >1 with "
          "a negative prompt on iport1 runs a 2nd DiT pass per step"},
  {.key = "models_db", .type = ConfigType::String, .required = false,
   .doc = "model registry db for resolve_model_dir (default \"models\")"},
  {.key = "init_latents", .type = ConfigType::String, .required = false,
   .doc = "debug: raw f32 packed initial latents [img_seq, 64] (repro/golden)"},
  {.key = "adopt_latent_dims", .type = ConfigType::Bool, .required = false,
   .doc = "img2img: take output H/W from the incoming latent (shape[1]*8 x "
          "shape[2]*8) instead of width/height (default false)"},
  {.key = "i8_gemm", .type = ConfigType::Bool, .required = false,
   .doc = "accelerated mode (LOSSY): dynamic-int8 GEMMs for the DiT's big "
          "block matmuls, ~2x their f16 rate at int8 quality; IGNORED "
          "without NAX matmul2d (matrix-core GPU + kernels). Default "
          "false; env VPIPE_I8_GEMM overrides"},
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
  {.name = "sampler", .doc = "OPTIONAL sampler spec FlexData (sampler-select)",
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
};
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
  .type_name = "text-to-image",
  .doc       = "Diffusion DiT denoiser: conditioning (from a diffusion-"
               "conditioner stage) -> family MMDiT -> FlowMatchEuler -> latent, "
               "on the metal-compute backend. The denoiser half of the split "
               "(feed vae-decode).",
  .display_name = "Text to Image",
  .category  = StageCategory::Generative,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

TextToImageStage::TextToImageStage(const SessionContextIntf* s,
                                   std::string               id,
                                   std::vector<InEdge>       iports,
                                   FlexData                  config)
  : TypedStage<TextToImageStage>(s, std::move(id), std::move(iports),
                                 std::move(config))
{
  _hf_dir    = attr_str("hf_dir");
  _dit_dir   = attr_str("dit_dir");
  _models_db = attr_str("models_db");
  _init_latents = attr_str("init_latents");
  if (_models_db.empty()) { _models_db = "models"; }
  _height = (int)attr_int("height");
  _width  = (int)attr_int("width");
  _steps  = (int)attr_int("steps");
  _strength = attr_real("strength");
  _guidance_scale = attr_real("guidance_scale");
  _adopt_latent_dims = attr_bool("adopt_latent_dims");
  // Accelerated mode (LOSSY, opt-in): dynamic-int8 GEMMs for the DiT's
  // big block matmuls (~2x their f16 rate on matrix-core GPUs at int8
  // quality). Env VPIPE_I8_GEMM=0|1 overrides.
  _i8_gemm = attr_bool("i8_gemm");
  _seed   = (std::uint64_t)attr_int("seed");
  if (_height <= 0) { _height = 256; }
  if (_width  <= 0) { _width  = 256; }
  if (_steps  <= 0) { _steps  = 8; }
  if (_guidance_scale <= 0.0) { _guidance_scale = 1.0; }   // <=0 => CFG off
  if (_strength < 0.0 || _strength > 1.0) {
    fail_config(fmt(
        "TextToImageStage('{}'): strength must be in [0,1] (got {})",
        this->id(), _strength));
  }
  if (_hf_dir.empty()) {
    fail_config(fmt(
        "TextToImageStage('{}'): config.hf_dir is required (the Krea-2-Turbo "
        "model dir)", this->id()));
  }
  if (_height % 16 != 0 || _width % 16 != 0) {
    fail_config(fmt(
        "TextToImageStage('{}'): height/width must be multiples of 16 (got "
        "{}x{})", this->id(), _height, _width));
  }
  allocate_oports(spec().oports.size());
#ifdef VPIPE_BUILD_APPLE_SILICON
  _scheduler_spec.steps = _steps;   // config default; port beats override
#endif
}

TextToImageStage::~TextToImageStage() = default;

const StageSpec&
TextToImageStage::spec() const noexcept
{
  return kSpec;
}

#ifdef VPIPE_BUILD_APPLE_SILICON

namespace {

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
      }
    }
  }
  return "krea2";
}

// Total physical RAM (bytes), 0 if unknown.
std::size_t
phys_ram_()
{
  std::uint64_t mem = 0;
  std::size_t len = sizeof(mem);
  if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) != 0) { return 0; }
  return (std::size_t)mem;
}

// Sum of the .safetensors bytes in a dir (proxy for the wired f16 footprint).
std::size_t
dir_weights_bytes_(const std::string& dir)
{
  namespace fs = std::filesystem;
  std::size_t total = 0;
  std::error_code ec;
  for (fs::recursive_directory_iterator it(dir, ec), end; it != end;
       it.increment(ec)) {
    if (it->is_regular_file(ec) && it->path().extension() == ".safetensors") {
      total += (std::size_t)it->file_size(ec);
    }
  }
  return total;
}

}  // namespace

Job
TextToImageStage::initialize(RuntimeContext& ctx)
{
  (void)ctx;
  if (_hf_dir.empty()) { co_return; }
  auto* mc = session() ? session()->metal_compute() : nullptr;
  if (mc == nullptr) {
    session()->error(fmt(
        "TextToImageStage('{}'): no metal-compute backend; the stage is inert",
        this->id()));
    co_return;
  }
  namespace fs = std::filesystem;
  const std::string root = resolve_model_dir(session(), _models_db, _hf_dir);
  // The text encoder lives in the paired diffusion-conditioner stage, but its
  // weights are resident in the SAME process while the DiT runs, so the
  // encoder-coexistence streaming heuristics below still budget for it.
  const std::string enc_dir = (fs::path(root) / "text_encoder").string();
  const std::string dit_dir = _dit_dir.empty()
      ? (fs::path(root) / "transformer").string()
      : resolve_model_dir(session(), _models_db, _dit_dir);
  _family = t2i_family_(dit_dir);
  session()->log_debug(fmt(
      "TextToImageStage('{}'): init root='{}' dit='{}' family={} "
      "(default {}x{}, {} steps, seed {}, strength {})", this->id(), root,
      dit_dir, _family, _width, _height, _steps, _seed, _strength));

  session()->info(fmt(
      "TextToImageStage('{}'): loading {} DiT from '{}'", this->id(),
      _family == "flux2" ? "FLUX.2"
      : _family == "qwen-image-edit" ? "Qwen-Image-Edit MMDiT"
      : "Krea2 MMDiT", dit_dir));
  if (_family == "flux2") {
    // Stream the DiT blocks when the box can't hold encoder + DiT together
    // (e.g. the 18 GB 9B DiT + a 16 GB encoder on a 16/32 GB box); ~2-3x slower
    // per step but bounds peak RAM to ~one block. VPIPE_FLUX2_STREAM forces it.
    bool stream_blocks;
    double pin_frac = 0.0;
    {
      const std::size_t dit_b = dir_weights_bytes_(dit_dir);
      const std::size_t enc_b = dir_weights_bytes_(enc_dir);
      const std::size_t ram = phys_ram_();
      const std::size_t need = dit_b + enc_b + (6ull << 30);   // +6 GB headroom
      stream_blocks = (ram != 0) && (ram < need);
      if (const char* e = std::getenv("VPIPE_FLUX2_STREAM")) {
        stream_blocks = (std::atoi(e) != 0);
      }
      // Pin as many leading blocks as fit alongside the resident encoder (see
      // the Qwen-Image-Edit branch below for the encoder-coexistence rationale).
      if (stream_blocks && ram > enc_b + (5ull << 30)) {
        pin_frac = std::min(0.60,
            double(ram - enc_b - (5ull << 30)) / double(ram));
      }
      session()->log_debug(fmt(
          "TextToImageStage('{}'): FLUX.2 DiT {} GB + enc {} GB + 6 GB vs {} GB "
          "RAM -> {}", this->id(), dit_b >> 30, enc_b >> 30, ram >> 30,
          stream_blocks ? "STREAM blocks" : "PRELOAD"));
    }
    if (const char* e = std::getenv("VPIPE_FLUX2_PIN_FRAC")) {
      pin_frac = std::atof(e);
    }
    genai::MetalFlux2Transformer::Config fcfg;
    fcfg.i8_gemm = _i8_gemm;
    _flux2_dit = genai::MetalFlux2Transformer::load(
        dit_dir, mc, fcfg, stream_blocks, pin_frac);
    if (!_flux2_dit) {
      session()->error(fmt(
          "TextToImageStage('{}'): failed to load the FLUX.2 DiT from '{}'; "
          "inert", this->id(), dit_dir));
      co_return;
    }
    // When the DiT had to stream (box can't hold encoder + DiT together), drop
    // the DiT's per-forward scratch after each generation so it doesn't crowd
    // out a large downstream VAE decode.
    _release_scratch = stream_blocks;
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
    const std::size_t dit_b = dir_weights_bytes_(dit_dir);
    const std::size_t enc_b = dir_weights_bytes_(enc_dir);
    const std::size_t ram = phys_ram_();
    bool stream_blocks = (ram != 0) && (ram < dit_b + enc_b + (6ull << 30));
    if (const char* e = std::getenv("VPIPE_QIE_STREAM")) {
      stream_blocks = (std::atoi(e) != 0);
    }
    double pin_frac = 0.0;
    if (stream_blocks && ram > enc_b + (5ull << 30)) {
      pin_frac = std::min(0.60,
          double(ram - enc_b - (5ull << 30)) / double(ram));
    }
    if (const char* e = std::getenv("VPIPE_QIE_PIN_FRAC")) {
      pin_frac = std::atof(e);
    }
    session()->log_debug(fmt(
        "TextToImageStage('{}'): Qwen-Image-Edit DiT {} GB + enc {} GB + 6 GB "
        "vs {} GB RAM -> {}", this->id(), dit_b >> 30, enc_b >> 30, ram >> 30,
        stream_blocks ? "STREAM blocks" : "PRELOAD"));
    genai::MetalQwenImageTransformer::Config qcfg;
    _qie_dit = genai::MetalQwenImageTransformer::load(dit_dir, mc, qcfg,
                                                      stream_blocks, pin_frac);
    if (!_qie_dit) {
      session()->error(fmt(
          "TextToImageStage('{}'): failed to load the Qwen-Image-Edit DiT from "
          "'{}'; inert", this->id(), dit_dir));
      co_return;
    }
    if (stream_blocks) {
      session()->info(fmt(
          "TextToImageStage('{}'): Qwen-Image-Edit DiT streaming, pinned {} of "
          "{} blocks resident", this->id(), _qie_dit->pinned_blocks(),
          qcfg.n_layers));
    }
    _release_scratch = stream_blocks;
  } else {
    genai::MetalKrea2Transformer::Config kcfg;
    kcfg.i8_gemm = _i8_gemm;
    _dit = genai::MetalKrea2Transformer::load(dit_dir, mc, kcfg);
    if (!_dit) {
      session()->error(fmt(
          "TextToImageStage('{}'): failed to load the DiT from '{}'; inert",
          this->id(), dit_dir));
      co_return;
    }
    // The Krea-2 DiT mmaps its quantized weights (evictable under GPU memory
    // pressure), but the conditioner's dense text encoder is copied into dirty,
    // non-reclaimable buffers resident in this process. When the box can't hold
    // encoder + DiT + a large VAE decode at once, drop the DiT's per-forward
    // scratch after each generation so its working set doesn't crowd out the
    // downstream vae-decode stage.
    const std::size_t dit_b = dir_weights_bytes_(dit_dir);
    const std::size_t enc_b = dir_weights_bytes_(enc_dir);
    const std::size_t ram = phys_ram_();
    const std::size_t need = dit_b + enc_b + (6ull << 30);   // +6 GB headroom
    _release_scratch = (ram != 0) && (ram < need);
    session()->log_debug(fmt(
        "TextToImageStage('{}'): Krea2 DiT {} GB + enc {} GB + 6 GB vs {} GB "
        "RAM -> {} DiT scratch", this->id(), dit_b >> 30, enc_b >> 30,
        ram >> 30, _release_scratch ? "RELEASE" : "keep"));
  }
  session()->log_debug(fmt(
      "TextToImageStage('{}'): {} DiT ready{}",
      this->id(), _family,
      _strength > 0.0 ? "; img2img init latent expected on a ref_latent iport"
                      : ""));
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

// Throttled in-place denoise progress bar, mirroring the model-quantize
// stage's bar: painted to the user-facing text stream and redrawn on a
// carriage-return only when the integer percentage changes (the frame is
// space-padded so a shorter redraw fully overwrites a longer prior one).
void
denoise_progress_(UiTextStream* bar, int done, int total, int& last_pct)
{
  if (bar == nullptr || total <= 0) { return; }
  int pct = static_cast<int>(static_cast<long>(done) * 100 / total);
  if (pct < 0) { pct = 0; } else if (pct > 100) { pct = 100; }
  if (pct == last_pct) { return; }
  last_pct = pct;
  constexpr int W = 24;
  const int fill = pct * W / 100;
  std::string b(static_cast<std::size_t>(fill), '#');
  b += std::string(static_cast<std::size_t>(W - fill), '-');
  std::string line = fmt("\r[{}] {}% denoise ({}/{})", b, pct, done,
                         total)();
  while (line.size() < 48) { line += ' '; }   // wipe stale tail
  bar->write(line);
}

}  // namespace

std::vector<float>
TextToImageStage::generate_(const metal_compute::SharedBuffer& cond, int n_real,
                            const metal_compute::SharedBuffer& cond_neg,
                            int n_real_neg, int gen_h, int gen_w,
                            const std::vector<float>* init_packed,
                            const std::vector<float>* init_latent,
                            const std::vector<RefLatent>& refs,
                            const std::function<void(const std::vector<float>&)>&
                                emit_step) const
{
  auto* mc = session()->metal_compute();
  using metal_compute::SharedBuffer;
  const int H = gen_h, W = gen_w;
  const int lh = H / 8, lw = W / 8;          // latent H/W
  const int gh = H / 16, gw = W / 16;        // 2x2-patch grid
  const int img_seq = gh * gw;
  const int IC = 64;                         // z_dim(16) * patch(2) * patch(2)

  // The conditioner emits the 12-tap f16 conditioning [n_real, 12, EH]; the
  // DiT's text-fusion tower fuses it into the DiT-facing text [n_real, hidden].
  SharedBuffer fused = _dit->forward_text(cond, n_real);
  if (fused.empty()) { return {}; }
  session()->log_debug(fmt(
      "TextToImageStage('{}'): fused conditioning [{}, hidden]; "
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
          "TextToImageStage('{}'): negative-conditioning fuse failed; running "
          "without classifier-free guidance", this->id()));
      cfg = false;
    } else {
      session()->log_debug(fmt(
          "TextToImageStage('{}'): CFG on, scale {}, negative [{}, hidden]",
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
      "TextToImageStage('{}'): sampler={} scheduler={} steps={} start={} "
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
            "TextToImageStage('{}'): krea2 reference latent [{}, {}, {}] must be "
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
        "TextToImageStage('{}'): Krea-2 conditioning on {} reference image(s)",
        this->id(), ri.size()));
  }
  bool dit_ok = true;
  const float gscale = (float)_guidance_scale;
  auto denoise = [&](const std::vector<float>& cand,
                     double sigma) -> std::vector<float> {
    auto* lb = static_cast<_Float16*>(latbuf.contents());
    for (std::size_t k = 0; k < cand.size(); ++k) { lb[k] = (_Float16)cand[k]; }
    SharedBuffer vel = _dit->forward_dit(fused, n_real, latbuf, img_seq, gh, gw,
                                         (float)sigma, -1, ri);
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
  std::unique_ptr<UiTextStream> bar = session()->open_text_stream();
  const int nsteps = S - start;
  int last_pct = -1;
  const auto gen_t0 = std::chrono::steady_clock::now();
  denoise_progress_(bar.get(), 0, nsteps, last_pct);
  for (int i = start; i < S; ++i) {
    session()->log_debug(fmt(
        "TextToImageStage('{}'): denoise step {}/{} sigma {}", this->id(),
        i + 1, S, (float)sig[(std::size_t)i]));
    sampler.step(i, packed,
                 prof ? genai::FlowSampler::DenoiseFn(denoise_p) : denoise);
    if (!dit_ok) { bar->end(); return {}; }
    if (emit_step) { emit_step(unpack(packed)); }
    denoise_progress_(bar.get(), i - start + 1, nsteps, last_pct);
  }
  bar->end();
  const double gen_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - gen_t0).count();
  session()->info(fmt(
      "TextToImageStage('{}'): latent generated in {:.2f}s ({} denoise "
      "steps, {} ms/step)", this->id(), gen_s, nsteps,
      nsteps ? (long)(gen_s * 1000.0 / nsteps) : 0));
  if (prof) {
    session()->log_normal(fmt(
        "TextToImageStage('{}'): DiT {} forward_dit calls, {} ms total, {} "
        "ms/call (seq {}+{}={})", this->id(), dit_calls, (long)dit_ms,
        dit_calls ? (long)(dit_ms / dit_calls) : 0, n_real, img_seq,
        n_real + img_seq));
  }

  // 9. unpack [img_seq, 64] -> channel-first latent [16, lh, lw].
  return unpack(packed);
}

std::vector<float>
TextToImageStage::generate_flux2_(const metal_compute::SharedBuffer& context,
                                  int n_real, int gen_h, int gen_w,
                                  const std::vector<RefLatent>& refs,
                                  const std::function<void(
                                      const std::vector<float>&)>& emit_step)
    const
{
  auto* mc = session()->metal_compute();
  using metal_compute::SharedBuffer;
  const int H = gen_h, W = gen_w;
  const int gh = H / 16, gw = W / 16;        // VAE latent grid (128ch @ H/16)
  const int img_seq = gh * gw;
  const int IC = _flux2_dit->config().in_channels;   // 128
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
        "TextToImageStage('{}'): FLUX.2 flow-shift mu = {} (img_seq {}, {} "
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
        "TextToImageStage('{}'): FLUX.2 embedded guidance scale {}",
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
          "TextToImageStage('{}'): reference latent has {} channels, expected "
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
        "TextToImageStage('{}'): FLUX.2 conditioning on {} reference image(s)",
        this->id(), ri.size()));
  }
  bool dit_ok = true;
  auto denoise = [&](const std::vector<float>& cand,
                     double sigma) -> std::vector<float> {
    auto* lb = static_cast<_Float16*>(latbuf.contents());
    for (std::size_t k = 0; k < cand.size(); ++k) { lb[k] = (_Float16)cand[k]; }
    SharedBuffer vel = _flux2_dit->forward_dit(context, n_real, latbuf, img_seq,
                                               gh, gw, (float)sigma, guid, ri);
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
  std::unique_ptr<UiTextStream> bar = session()->open_text_stream();
  int last_pct = -1;
  const auto gen_t0 = std::chrono::steady_clock::now();
  denoise_progress_(bar.get(), 0, S, last_pct);
  for (int i = 0; i < S; ++i) {
    sampler.step(i, packed,
                 prof ? genai::FlowSampler::DenoiseFn(denoise_p) : denoise);
    if (!dit_ok) { bar->end(); return {}; }
    if (emit_step) { emit_step(unpack(packed)); }
    denoise_progress_(bar.get(), i + 1, S, last_pct);
  }
  bar->end();
  const double gen_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - gen_t0).count();
  session()->info(fmt(
      "TextToImageStage('{}'): FLUX.2 latent generated in {:.2f}s ({} steps)",
      this->id(), gen_s, S));
  if (prof) {
    session()->log_normal(fmt(
        "TextToImageStage('{}'): FLUX.2 DiT {} forward_dit calls, {} ms total, "
        "{} ms/call (seq {}+{}={})", this->id(), dit_calls, (long)dit_ms,
        dit_calls ? (long)(dit_ms / dit_calls) : 0, n_real, img_seq,
        n_real + img_seq));
  }
  return unpack(packed);
}

std::vector<float>
TextToImageStage::generate_qie_(const metal_compute::SharedBuffer& txt_pos,
                               int n_real,
                               const metal_compute::SharedBuffer& txt_neg,
                               int n_real_neg, int gen_h, int gen_w,
                               const std::vector<float>* init_packed,
                               const std::vector<RefLatent>& refs,
                               const std::function<void(
                                   const std::vector<float>&)>& emit_step)
    const
{
  auto* mc = session()->metal_compute();
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
      "TextToImageStage('{}'): QIE conditioning [{}, txt]; {}x{} grid {}x{} "
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
            "TextToImageStage('{}'): reference latent [{}, {}, {}] must be "
            "16-channel with even H/W; ignoring", this->id(), r.c, r.h, r.w));
      }
      continue;
    }
    const int rgh = r.h / 2, rgw = r.w / 2, rseq = rgh * rgw;
    if (rgh != gh || rgw != gw) {
      session()->warn(fmt(
          "TextToImageStage('{}'): reference grid {}x{} != output grid {}x{} -- "
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
        "TextToImageStage('{}'): Qwen-Image-Edit conditioning on {} reference "
        "image(s)", this->id(), ri.size()));
  }

  // Denoise callback: upload the candidate, run the dual-stream DiT at `sigma`,
  // read the velocity back; apply norm-preserving true-CFG when enabled.
  bool dit_ok = true;
  const float gscale = (float)_guidance_scale;
  auto denoise = [&](const std::vector<float>& cand,
                     double sigma) -> std::vector<float> {
    auto* lb = static_cast<std::uint16_t*>(latbuf.contents());
    for (std::size_t k = 0; k < cand.size(); ++k) {
      lb[k] = f32_to_bf16_(cand[k]);
    }
    SharedBuffer vel = _qie_dit->forward(latbuf, img_seq, txt_pos, n_real, gh,
                                         gw, (float)sigma, ri);
    if (vel.empty()) { dit_ok = false; return {}; }
    const auto* vp = static_cast<const std::uint16_t*>(vel.contents());
    std::vector<float> v(cand.size());
    for (std::size_t k = 0; k < v.size(); ++k) { v[k] = bf16_to_f32_(vp[k]); }
    if (cfg) {
      SharedBuffer veln = _qie_dit->forward(latbuf, img_seq, txt_neg, n_real_neg,
                                            gh, gw, (float)sigma, ri);
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
  std::unique_ptr<UiTextStream> bar = session()->open_text_stream();
  int last_pct = -1;
  const auto gen_t0 = std::chrono::steady_clock::now();
  denoise_progress_(bar.get(), 0, S, last_pct);
  for (int i = 0; i < S; ++i) {
    sampler.step(i, packed,
                 prof ? genai::FlowSampler::DenoiseFn(denoise_p) : denoise);
    if (!dit_ok) { bar->end(); return {}; }
    if (emit_step) { emit_step(unpack(packed)); }
    denoise_progress_(bar.get(), i + 1, S, last_pct);
  }
  bar->end();
  const double gen_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - gen_t0).count();
  session()->info(fmt(
      "TextToImageStage('{}'): latent generated in {:.2f}s ({} denoise steps, "
      "{} ms/step)", this->id(), gen_s, S,
      S ? (long)(gen_s * 1000.0 / S) : 0));
  if (prof) {
    session()->log_normal(fmt(
        "TextToImageStage('{}'): QIE DiT {} forward calls, {} ms total, {} "
        "ms/call (txt {} + img {} = {})", this->id(), dit_calls, (long)dit_ms,
        dit_calls ? (long)(dit_ms / dit_calls) : 0, n_real, img_seq,
        n_real + img_seq));
  }
  return unpack(packed);
}

Job
TextToImageStage::process(RuntimeContext& ctx)
{
  auto* mc = session()->metal_compute();
  auto in = co_await ctx.read(0);
  if (!in) { ctx.signal_done(); co_return; }
  const bool have_dit = _family == "flux2" ? (bool)_flux2_dit
      : _family == "qwen-image-edit" ? (bool)_qie_dit
      : (bool)_dit;
  if (!have_dit) {
    session()->warn(fmt(
        "TextToImageStage('{}'): models not loaded; dropping beat", this->id()));
    co_return;
  }
  // iport0: the conditioning tensor from a diffusion-conditioner stage
  // (family-shaped + typed; rows = shape[0]). Copy it into a metal buffer.
  const auto* ctb = dynamic_cast<const TensorBeatPayload*>(in.get());
  if (ctb == nullptr || ctb->shape.empty() || ctb->shape[0] <= 0) {
    session()->warn(fmt(
        "TextToImageStage('{}'): expected a conditioning TensorBeat, got {}; "
        "dropping beat", this->id(), in->describe()));
    co_return;
  }
  metal_compute::SharedBuffer cond = cond_to_shared_(mc, *ctb);
  if (cond.empty()) {
    session()->warn(fmt(
        "TextToImageStage('{}'): conditioning upload failed; dropping beat",
        this->id()));
    co_return;
  }
  const int n_real = (int)ctb->shape[0];
  session()->log_debug(fmt(
      "TextToImageStage('{}'): conditioning beat [{} rows, {}]", this->id(),
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
          "TextToImageStage('{}'): negative conditioning [{} rows]", this->id(),
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

  // Reference latents on iport4 / iport5: latch the FIRST reference on each
  // connected port (blocking, like the sampler/scheduler specs) and cache it
  // (`_ref[]`), so the reference reliably pairs with the prompt (a non-blocking
  // poll would race the producer) and a fixed reference supplied once is reused
  // for every later prompt. FLUX.2 threads them as multi-reference conditioning
  // tokens (below); Krea-2 uses ref latent 0 as the img2img init and ignores
  // ref latent 1.
  for (int r = 0; r < 2; ++r) {
    const int port = 4 + r;
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
            "TextToImageStage('{}'): reference latent {} = [{}, {}, {}]",
            this->id(), r, _ref[r].c, _ref[r].h, _ref[r].w));
      } else if (rb) {
        session()->warn(fmt(
            "TextToImageStage('{}'): ref_latent{} must be an f32 [C,H,W] "
            "TensorBeat; got {}, ignoring", this->id(), r, rb->describe()));
      }
    }
  }

  int gen_h = _height, gen_w = _width;
  int lh = gen_h / 8, lw = gen_w / 8;
  std::vector<float> latent;
  const std::vector<float>* latent_ptr = nullptr;
  // Krea-2 img2img: ref latent 0 (the whitened [z_dim, H/8, W/8] vae-encode
  // output) is mixed into the noise at the strength-selected sigma. With
  // adopt_latent_dims the output size is taken FROM the latent (the DiT
  // patchifies 2x2, so H/W must be even). FLUX.2 skips this -- refs are
  // conditioning tokens, not an init, handled in generate_flux2_.
  if (_family == "krea2" && _strength > 0.0 && !_ref[0].empty()) {
    const RefLatent& rr = _ref[0];
    const bool ok_type = rr.c == 16 && rr.h > 0 && rr.w > 0;
    if (ok_type && _adopt_latent_dims) {
      if ((rr.h % 2) == 0 && (rr.w % 2) == 0) {
        gen_h = rr.h * 8; gen_w = rr.w * 8;
        lh = rr.h; lw = rr.w;
      } else {
        session()->warn(fmt(
            "TextToImageStage('{}'): adopt_latent_dims: latent [16, {}, {}] "
            "needs even H and W; keeping {}x{}", this->id(), rr.h, rr.w,
            _width, _height));
      }
    }
    if (ok_type && rr.h == lh && rr.w == lw) {
      latent = rr.chw;
      latent_ptr = &latent;
    } else {
      session()->warn(fmt(
          "TextToImageStage('{}'): img2img init must be an f32 latent [16, {}, "
          "{}]; got [{}, {}, {}], ignoring", this->id(), lh, lw, rr.c, rr.h,
          rr.w));
    }
  }

  // Latch the sampler / scheduler specs off iport2 / iport3 (once each): the
  // `sampler-select` / `scheduler-select` sources emit a single spec beat, which
  // we cache and reuse for every subsequent prompt.
  if (!_sampler_latched && ctx.num_iports() >= 3 && ctx.iport_connected(2)) {
    auto sb = co_await ctx.read(2);
    _sampler_latched = true;
    const auto* sfd = dynamic_cast<const FlexDataPayload*>(sb.get());
    if (sfd != nullptr) {
      std::string serr;
      _sampler_spec = genai::FlowSamplerSpec::from_flex(sfd->data, &serr);
      if (!serr.empty()) {
        session()->warn(fmt("TextToImageStage('{}'): sampler spec: {}",
                            this->id(), serr));
      }
      session()->info(fmt(
          "TextToImageStage('{}'): sampler = {} (eta {}, s_noise {})",
          this->id(), _sampler_spec.method, _sampler_spec.eta,
          _sampler_spec.s_noise));
    }
  }
  if (!_scheduler_latched && ctx.num_iports() >= 4 && ctx.iport_connected(3)) {
    auto cb = co_await ctx.read(3);
    _scheduler_latched = true;
    const auto* cfd = dynamic_cast<const FlexDataPayload*>(cb.get());
    if (cfd != nullptr) {
      std::string cerr;
      _scheduler_spec = genai::FlowSchedulerSpec::from_flex(cfd->data, &cerr);
      if (!cerr.empty()) {
        session()->warn(fmt("TextToImageStage('{}'): scheduler spec: {}",
                            this->id(), cerr));
      }
      session()->info(fmt(
          "TextToImageStage('{}'): scheduler = {} ({} steps, shift {} {})",
          this->id(), _scheduler_spec.type, _scheduler_spec.steps,
          _scheduler_spec.shift, _scheduler_spec.shift_type));
    }
  }

  // ---- FLUX.2: text-to-image from noise (+ optional reference-image
  // conditioning from iport4/iport5) -> latent [dit_channels, H/16, W/16].
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
        generate_flux2_(cond, n_real, gen_h, gen_w, frefs,
                        step_emitter({Cdit, fgh, fgw}));
    _flux2_dit->set_stream_stop({});
    if (fl.empty()) {
      session()->info(fmt(
          "TextToImageStage('{}'): FLUX.2 generation {}; dropping beat",
          this->id(), ctx.stop_requested() ? "stopped" : "failed"));
      co_return;
    }
    auto out = std::make_unique<TensorBeatPayload>();
    out->dtype = TensorBeat::DType::F32;
    out->shape = {Cdit, fgh, fgw};
    out->resize_contiguous(fl.size());
    std::memcpy(out->as_f32(), fl.data(), fl.size() * sizeof(float));
    ++_latents_emitted;
    session()->info(fmt(
        "TextToImageStage('{}'): FLUX.2 latent [{}, {}, {}] ({} steps @ {}x{})",
        this->id(), Cdit, fgh, fgw, _scheduler_spec.steps, gen_h, gen_w));
    co_await ctx.write(0, std::move(out));
    co_return;
  }

  // ---- Qwen-Image-Edit: image-aware conditioning (from the conditioner) +
  // dual-stream DiT (reference latents from iport4/iport5 as DiT conditioning
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
          "TextToImageStage('{}'): Qwen-Image-Edit generation {}; dropping "
          "beat", this->id(), ctx.stop_requested() ? "stopped" : "failed"));
      co_return;
    }
    auto out = std::make_unique<TensorBeatPayload>();
    out->dtype = TensorBeat::DType::F32;
    out->shape = {16, lh, lw};
    out->resize_contiguous(ql.size());
    std::memcpy(out->as_f32(), ql.data(), ql.size() * sizeof(float));
    ++_latents_emitted;
    session()->info(fmt(
        "TextToImageStage('{}'): Qwen-Image-Edit latent [16, {}, {}] "
        "({} steps @ {}x{})", this->id(), lh, lw, _scheduler_spec.steps,
        gen_h, gen_w));
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
        "TextToImageStage('{}'): generation {}; dropping beat", this->id(),
        ctx.stop_requested() ? "stopped" : "failed"));
    co_return;
  }
  // The DiT is idle now (latent read back): on a memory-bounded box drop its
  // per-forward scratch (DitScratch activations + dequant/split-K + i8 accel
  // buffers, ~1-2 GB at 1024px), which regrows on the next generation. The DiT
  // weights stay mmap'd (evictable). This frees enough working set for the
  // separate vae-decode stage; without it a 1024px decode OOMs on a 16 GB box.
  if (_release_scratch && _dit) { _dit->release_forward_scratch(); }

  auto out = std::make_unique<TensorBeatPayload>();
  out->dtype = TensorBeat::DType::F32;
  out->shape = {16, lh, lw};
  out->resize_contiguous(out_latent.size());
  std::memcpy(out->as_f32(), out_latent.data(),
              out_latent.size() * sizeof(float));
  ++_latents_emitted;
  session()->info(fmt(
      "TextToImageStage('{}'): latent [16, {}, {}] ({}+{} {} steps @ "
      "{}x{})", this->id(), lh, lw, _sampler_spec.method,
      _scheduler_spec.type, _scheduler_spec.steps, gen_h, gen_w));
  co_await ctx.write(0, std::move(out));
}

#else   // !VPIPE_BUILD_APPLE_SILICON

Job
TextToImageStage::initialize(RuntimeContext& ctx)
{
  (void)ctx;
  if (session()) {
    session()->error(fmt(
        "TextToImageStage('{}'): built without VPIPE_BUILD_APPLE_SILICON; "
        "inert", this->id()));
  }
  co_return;
}

Job
TextToImageStage::process(RuntimeContext& ctx)
{
  auto in = co_await ctx.read(0);
  (void)in;
  ctx.signal_done();
  co_return;
}

#endif  // VPIPE_BUILD_APPLE_SILICON

VPIPE_REGISTER_STAGE(TextToImageStage)
VPIPE_REGISTER_SPEC(TextToImageStage, kSpec)

}

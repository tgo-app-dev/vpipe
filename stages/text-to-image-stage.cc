#include "stages/text-to-image-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "stages/model-registry.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "generative-models/context-manager.h"
#include "generative-models/llama3/metal-llama-weights.h"
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

// The Krea-2 prompt template (pipeline_krea2.py): the DiT is conditioned on
// [prefix | prompt | suffix], with the 34-token system prefix DROPPED after
// encoding. Verified compact-equivalent to the padded+masked HF path.
constexpr const char* kPrefix =
    "<|im_start|>system\nDescribe the image by detailing the color, shape, "
    "size, texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n";
constexpr const char* kSuffix = "<|im_end|>\n<|im_start|>assistant\n";
constexpr int kDropPrefix = 34;          // prompt_template_encode_start_idx
const int kSelectLayers[12] = {2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35};

// True when `dir` looks like a transformer-only Krea2 DiT (the quantize
// output) rather than a full Krea-2-Turbo pipeline: it has a diffusers
// config.json whose _class_name is the DiT model, and lacks the tokenizer
// sub-dir. Used only to sharpen the error message when hf_dir is misset.
bool
looks_like_bare_dit_(const std::string& dir)
{
  namespace fs = std::filesystem;
  if (fs::exists(fs::path(dir) / "tokenizer" / "tokenizer.json")) {
    return false;
  }
  std::ifstream in(fs::path(dir) / "config.json");
  if (!in) { return false; }
  FlexData fd = FlexData::from_json(in);
  if (!fd.is_object()) { return false; }
  auto obj = fd.as_object();
  if (!obj.contains("_class_name")) { return false; }
  const std::string cls(obj.at("_class_name").as_string(""));
  return cls.find("Transformer") != std::string::npos;
}

const ConfigKey kAttrs[] = {
  {.key = "hf_dir", .type = ConfigType::String, .required = true,
   .doc = "Krea-2-Turbo / FLUX.2 model dir (text_encoder/, transformer/, "
          "tokenizer/); an original or model-quantize'd (self-contained) "
          "pipeline",
   .suggest_db = "models", .suggest_db_type = "krea2,flux2"},
  {.key = "dit_dir", .type = ConfigType::String, .required = false,
   .doc = "override DiT dir (e.g. a quantized 4/8-bit DiT); else <hf_dir>/transformer",
   .suggest_db = "models", .suggest_db_type = "krea2-dit,flux2-dit"},
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
  {.name = "prompt", .doc = "prompt text (FlexData string or {text: ...})",
   .type = &typeid(FlexDataPayload),
   .tags = "text", .clock_group = 0},
  {.name = "negative", .doc = "OPTIONAL negative prompt (FlexData string or "
                              "{text: ...}) for classifier-free guidance",
   .type = &typeid(FlexDataPayload),
   .tags = "text", .clock_group = 0},
  {.name = "sampler", .doc = "OPTIONAL sampler spec FlexData (sampler-select)",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
  {.name = "scheduler", .doc = "OPTIONAL scheduler spec FlexData (scheduler-select)",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
  {.name = "ref_latent0", .doc = "OPTIONAL reference latent 0 (channel-first "
                                 "f32 from vae-encode); FLUX.2 conditioning / "
                                 "Krea-2 img2img init",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
  {.name = "ref_latent1", .doc = "OPTIONAL reference latent 1 (channel-first "
                                 "f32 from vae-encode); FLUX.2 2nd reference "
                                 "(distinct RoPE position); ignored by Krea-2",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "latent",
   .doc = "f32 latent [z_dim, H/8, W/8] (unpacked, whitened)",
   .type = &typeid(TensorBeatPayload),
   .tags = "latent", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "text-to-image",
  .doc       = "Krea-2-Turbo text-to-image: prompt -> Qwen3-VL 12-layer encode "
               "-> 12B MMDiT -> 8-step FlowMatchEuler -> latent, on the metal-"
               "compute backend. First half of the krea2 split (feed vae-decode).",
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

// The Krea-2 text encoder: a dense Qwen3-VL decoder (36L / 2560 / 32q-8kv /
// hd128 / ffn9728, q/k-norm, STANDARD RMSNorm, mRoPE theta 5e6), run on
// MetalQwenModel as raw bf16 with the per-layer tap.
genai::MetalQwenModel::Config
encoder_config_()
{
  genai::MetalQwenModel::Config c;
  c.n_layers           = 36;
  c.hidden             = 2560;
  c.n_heads            = 32;
  c.n_kv_heads         = 8;
  c.head_dim           = 128;
  c.ffn_inner          = 9728;
  c.vocab              = 151936;
  c.rope_theta         = 5.0e6f;
  c.rms_eps            = 1e-6f;
  c.rotary_dim         = 128;
  c.full_attn_interval = 1;
  c.tie_embeddings     = true;
  c.use_bf16           = true;
  c.dense              = true;
  c.zero_centered_norm = false;
  c.attn_output_gate   = false;
  c.backbone_only      = true;      // we host-gather the embeddings ourselves
  c.weight_prefix      = "language_model.";
  c.model_seg          = "";
  c.max_seq            = 1024;
  c.page_tokens        = 256;
  return c;
}

// FLUX.2 klein text encoder: a plain DENSE Qwen3ForCausalLM under the "model."
// prefix (not the Qwen3-VL "language_model."). Sized from the encoder's
// config.json so both klein sizes work off one path -- the 4B taps a ~4B Qwen3
// (36L / 2560), the 9B an 8B Qwen3 (larger depth/width). Absent keys keep the
// base (4B) default.
genai::MetalQwenModel::Config
encoder_config_flux2_(const std::string& enc_dir)
{
  genai::MetalQwenModel::Config c = encoder_config_();
  c.rope_theta    = 1.0e6f;
  c.weight_prefix = "model.";
  namespace fs = std::filesystem;
  std::ifstream in(fs::path(enc_dir) / "config.json");
  if (in) {
    FlexData fd = FlexData::from_json(in);
    if (fd.is_object()) {
      auto o = fd.as_object();
      auto geti = [&](const char* k, int cur) -> int {
        return o.contains(k) ? (int)o.at(k).as_int(cur) : cur;
      };
      auto getf = [&](const char* k, float cur) -> float {
        return o.contains(k) ? (float)o.at(k).as_real(cur) : cur;
      };
      c.n_layers    = geti("num_hidden_layers", c.n_layers);
      c.hidden      = geti("hidden_size", c.hidden);
      c.n_heads     = geti("num_attention_heads", c.n_heads);
      c.n_kv_heads  = geti("num_key_value_heads", c.n_kv_heads);
      c.head_dim    = geti("head_dim",
                           c.n_heads > 0 ? c.hidden / c.n_heads : c.head_dim);
      c.rotary_dim  = c.head_dim;
      c.ffn_inner   = geti("intermediate_size", c.ffn_inner);
      c.vocab       = geti("vocab_size", c.vocab);
      c.rope_theta  = getf("rope_theta", c.rope_theta);
      c.rms_eps     = getf("rms_norm_eps", c.rms_eps);
      if (o.contains("tie_word_embeddings")) {
        c.tie_embeddings = o.at("tie_word_embeddings").as_bool(c.tie_embeddings);
      }
    }
  }
  return c;
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
// "Flux2Transformer2DModel" -> "flux2"; else "krea2".
std::string
t2i_family_(const std::string& transformer_dir)
{
  namespace fs = std::filesystem;
  std::ifstream in(fs::path(transformer_dir) / "config.json");
  if (in) {
    FlexData fd = FlexData::from_json(in);
    if (fd.is_object()) {
      auto obj = fd.as_object();
      if (obj.contains("_class_name") &&
          std::string(obj.at("_class_name").as_string("")) ==
              "Flux2Transformer2DModel") {
        return "flux2";
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

bool
TextToImageStage::load_encoder_(metal_compute::MetalCompute* mc)
{
  namespace fs = std::filesystem;
  const std::string& enc_dir = _enc_dir;
  if (mc == nullptr || enc_dir.empty()) { return false; }
  // The encoder may be affine-quantized (model-quantize target=text_encoder).
  // The loader auto-detects quantized-vs-dense weights but needs the bit-width
  // to pick the w4g64 vs w8g64 kernel, so read it from the encoder's
  // config.json quantization block (absent => dense bf16, quant_bits unused).
  genai::MetalQwenModel::Config ecfg =
      _family == "flux2" ? encoder_config_flux2_(enc_dir) : encoder_config_();
  _enc_hidden = ecfg.hidden;
  {
    std::ifstream in(fs::path(enc_dir) / "config.json");
    if (in) {
      FlexData fd = FlexData::from_json(in);
      if (fd.is_object()) {
        auto o = fd.as_object();
        if (o.contains("quantization")) {
          FlexData q = o.at("quantization");
          if (q.is_object()) {
            auto qo = q.as_object();
            const int b = qo.contains("bits")
                ? (int)qo.at("bits").as_int(0) : 0;
            if (b == 4 || b == 8) {
              ecfg.quant_bits = b;
              session()->log_debug(fmt(
                  "TextToImageStage('{}'): text encoder is {}-bit quantized",
                  this->id(), b));
            }
          }
        }
      }
    }
  }
  session()->info(fmt(
      "TextToImageStage('{}'): loading {} text encoder from '{}'",
      this->id(), _family == "flux2" ? "Qwen3 (dense)" : "Qwen3-VL", enc_dir));
  _encoder = genai::MetalQwenModel::load(enc_dir, mc, ecfg);
  if (!_encoder) {
    session()->error(fmt(
        "TextToImageStage('{}'): failed to load text encoder from '{}'; inert",
        this->id(), enc_dir));
    return false;
  }
  // The embed table (backbone_only skips the model's embed muxer, so we gather
  // input rows ourselves) -- kept resident for per-prompt gathers.
  auto wts = genai::MetalLlamaWeights::open_model(enc_dir);
  if (wts.has_value()) {
    _embed = wts->load(_family == "flux2"
                           ? "model.embed_tokens.weight"
                           : "language_model.embed_tokens.weight", mc);
  }
  if (_embed.empty()) {
    session()->error(fmt(
        "TextToImageStage('{}'): failed to load the encoder embed table; inert",
        this->id()));
    _encoder.reset();
    return false;
  }
  return true;
}

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
  const std::string enc_dir = (fs::path(root) / "text_encoder").string();
  const std::string dit_dir = _dit_dir.empty()
      ? (fs::path(root) / "transformer").string()
      : resolve_model_dir(session(), _models_db, _dit_dir);
  const std::string tok_path =
      (fs::path(root) / "tokenizer" / "tokenizer.json").string();
  _family = t2i_family_(dit_dir);
  session()->log_debug(fmt(
      "TextToImageStage('{}'): init root='{}' enc='{}' dit='{}' "
      "(default {}x{}, {} steps, seed {}, strength {})", this->id(), root,
      enc_dir, dit_dir, _width, _height, _steps, _seed, _strength));

  _tokenizer = genai::Tokenizer::from_huggingface_json(tok_path, session());
  if (!_tokenizer) {
    session()->error(fmt(
        "TextToImageStage('{}'): failed to load tokenizer from '{}'; inert",
        this->id(), tok_path));
    // Common mistake: hf_dir points at a quantized DiT (transformer-only)
    // instead of the full Krea-2-Turbo pipeline. Detect it and steer the
    // user to the dit_dir override.
    if (looks_like_bare_dit_(root)) {
      session()->error(fmt(
          "TextToImageStage('{}'): '{}' looks like a transformer-only DiT "
          "(no tokenizer/text_encoder/vae). Set hf_dir to the full "
          "Krea-2-Turbo model and pass this dir as dit_dir instead.",
          this->id(), root));
    }
    co_return;
  }

  // Load the text encoder + embed table (see load_encoder_; reloaded lazily
  // after a _free_encoder free between generations).
  _enc_dir = enc_dir;
  if (!load_encoder_(mc)) { co_return; }

  session()->info(fmt(
      "TextToImageStage('{}'): loading {} DiT from '{}'", this->id(),
      _family == "flux2" ? "FLUX.2" : "Krea2 MMDiT", dit_dir));
  if (_family == "flux2") {
    // Stream the DiT blocks when the box can't hold encoder + DiT together
    // (e.g. the 18 GB 9B DiT + a 16 GB encoder on a 16/32 GB box); ~2-3x slower
    // per step but bounds peak RAM to ~one block. VPIPE_FLUX2_STREAM forces it.
    bool stream_blocks;
    {
      const std::size_t dit_b = dir_weights_bytes_(dit_dir);
      const std::size_t enc_b = dir_weights_bytes_(enc_dir);
      const std::size_t ram = phys_ram_();
      const std::size_t need = dit_b + enc_b + (6ull << 30);   // +6 GB headroom
      stream_blocks = (ram != 0) && (ram < need);
      if (const char* e = std::getenv("VPIPE_FLUX2_STREAM")) {
        stream_blocks = (std::atoi(e) != 0);
      }
      session()->log_debug(fmt(
          "TextToImageStage('{}'): FLUX.2 DiT {} GB + enc {} GB + 6 GB vs {} GB "
          "RAM -> {}", this->id(), dit_b >> 30, enc_b >> 30, ram >> 30,
          stream_blocks ? "STREAM blocks" : "PRELOAD"));
    }
    genai::MetalFlux2Transformer::Config fcfg;
    fcfg.i8_gemm = _i8_gemm;
    _flux2_dit = genai::MetalFlux2Transformer::load(
        dit_dir, mc, fcfg, stream_blocks);
    if (!_flux2_dit) {
      session()->error(fmt(
          "TextToImageStage('{}'): failed to load the FLUX.2 DiT from '{}'; "
          "inert", this->id(), dit_dir));
      _encoder.reset();
      co_return;
    }
    // When the DiT had to stream (box can't hold encoder + DiT together), also
    // free the idle encoder around each generation so it doesn't crowd out a
    // large VAE decode; reloaded lazily per prompt (see load_encoder_).
    _free_encoder = stream_blocks;
  } else {
    genai::MetalKrea2Transformer::Config kcfg;
    kcfg.i8_gemm = _i8_gemm;
    _dit = genai::MetalKrea2Transformer::load(dit_dir, mc, kcfg);
    if (!_dit) {
      session()->error(fmt(
          "TextToImageStage('{}'): failed to load the DiT from '{}'; inert",
          this->id(), dit_dir));
      _encoder.reset();
      co_return;
    }
    // The Krea-2 DiT mmaps its quantized weights (evictable under GPU memory
    // pressure), but the dense text encoder is copied into dirty, non-
    // reclaimable buffers. When the box can't hold encoder + DiT + a large VAE
    // decode at once, free the idle encoder around each generation (reloaded
    // lazily next prompt) so its multi-GB footprint doesn't crowd out the
    // downstream vae-decode stage's working set -- the same reclaim the FLUX.2
    // path does when it streams. VPIPE_FLUX2_FREE_ENCODER overrides below.
    const std::size_t dit_b = dir_weights_bytes_(dit_dir);
    const std::size_t enc_b = dir_weights_bytes_(enc_dir);
    const std::size_t ram = phys_ram_();
    const std::size_t need = dit_b + enc_b + (6ull << 30);   // +6 GB headroom
    _free_encoder = (ram != 0) && (ram < need);
    session()->log_debug(fmt(
        "TextToImageStage('{}'): Krea2 DiT {} GB + enc {} GB + 6 GB vs {} GB "
        "RAM -> {} idle encoder", this->id(), dit_b >> 30, enc_b >> 30,
        ram >> 30, _free_encoder ? "FREE" : "keep"));
  }
  if (const char* e = std::getenv("VPIPE_FLUX2_FREE_ENCODER")) {
    _free_encoder = (std::atoi(e) != 0);
  }
  session()->log_debug(fmt(
      "TextToImageStage('{}'): {} DiT + encoder ready (encoder hidden {}){}",
      this->id(), _family, _enc_hidden,
      _strength > 0.0 ? "; img2img init latent expected on the `latent` iport"
                      : ""));
}

namespace {

// Encode a template string that embeds ChatML special-token markers
// (<|im_start|> / <|im_end|>). genai::Tokenizer::encode() byte-encodes literal
// text and does NOT isolate special tokens, so split at the markers, encode
// each plain-text run, and splice the markers' ids via special_token_id() --
// matching the HF fast tokenizer (special tokens are isolated, each text run
// BPE-encoded independently).
std::vector<std::int32_t>
encode_with_specials_(const genai::Tokenizer& tok, const std::string& text)
{
  static const char* kMarkers[] = {"<|im_start|>", "<|im_end|>"};
  std::vector<std::int32_t> out;
  std::size_t pos = 0;
  while (pos < text.size()) {
    std::size_t best = std::string::npos;
    int which = -1;
    for (int mi = 0; mi < 2; ++mi) {
      const std::size_t f = text.find(kMarkers[mi], pos);
      if (f != std::string::npos && (best == std::string::npos || f < best)) {
        best = f; which = mi;
      }
    }
    if (which < 0) {
      const std::vector<std::int32_t> seg = tok.encode(text.substr(pos));
      out.insert(out.end(), seg.begin(), seg.end());
      break;
    }
    if (best > pos) {
      const std::vector<std::int32_t> seg =
          tok.encode(text.substr(pos, best - pos));
      out.insert(out.end(), seg.begin(), seg.end());
    }
    const std::int32_t sid = tok.special_token_id(kMarkers[which]);
    if (sid >= 0) { out.push_back(sid); }
    pos = best + std::strlen(kMarkers[which]);
  }
  return out;
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

std::vector<std::int32_t>
TextToImageStage::tokenize_prompt(const std::string& prompt) const
{
  if (!_tokenizer) { return {}; }
  return encode_with_specials_(*_tokenizer,
                               std::string(kPrefix) + prompt + kSuffix);
}

std::vector<float>
TextToImageStage::generate_(const std::string& prompt,
                            const std::string& negative, int gen_h, int gen_w,
                            const std::vector<float>* init_packed,
                            const std::vector<float>* init_latent,
                            const std::vector<RefLatent>& refs) const
{
  auto* mc = session()->metal_compute();
  using metal_compute::SharedBuffer;
  const int H = gen_h, W = gen_w;
  const int lh = H / 8, lw = W / 8;          // latent H/W
  const int gh = H / 16, gw = W / 16;        // 2x2-patch grid
  const int img_seq = gh * gw;
  const int IC = 64;                         // z_dim(16) * patch(2) * patch(2)
  const int EH = _enc_hidden;                // 2560
  const int NL = 12;

  // Steps 1-5: text -> fused DiT conditioning [n_real, hidden]. Factored so
  // it can run for both the positive prompt and (for classifier-free
  // guidance) the negative prompt. Returns the fused buffer (empty on
  // failure) and writes the post-prefix row count to `n_real_out`.
  auto encode_text = [&](const std::string& text, const char* which,
                         int& n_real_out) -> SharedBuffer {
    // 1. tokenize -> compact [prefix | text | suffix].
    const std::vector<std::int32_t> ids = tokenize_prompt(text);
    if ((int)ids.size() <= kDropPrefix) { return SharedBuffer{}; }
    const int n = (int)ids.size();
    const int n_real = n - kDropPrefix;
    session()->log_debug(fmt(
        "TextToImageStage('{}'): [{}] tokenized {} ids ({} rows after "
        "dropping {} prefix)", this->id(), which, n, n_real, kDropPrefix));

    // 2. gather encoder input rows (bf16) by id.
    SharedBuffer x = mc->make_shared_buffer((std::size_t)n * EH * 2);
    if (x.empty()) { return SharedBuffer{}; }
    {
      const auto* tbl = static_cast<const std::uint8_t*>(_embed.contents());
      auto* xb = static_cast<std::uint8_t*>(x.contents());
      const std::size_t vocab = _embed.byte_size() / ((std::size_t)EH * 2);
      for (int i = 0; i < n; ++i) {
        const std::uint32_t id = (std::uint32_t)ids[(std::size_t)i];
        if (id >= vocab) { return SharedBuffer{}; }
        std::memcpy(xb + (std::size_t)i * EH * 2,
                    tbl + (std::size_t)id * EH * 2, (std::size_t)EH * 2);
      }
    }

    // 3. tap the 12 selected layers.
    std::vector<int> tap_layers;
    for (int k : kSelectLayers) { tap_layers.push_back(k - 1); }
    genai::ContextManager* cm = _encoder->context_manager();
    const genai::ContextId cid = cm->acquire_root();
    SharedBuffer taps =
        _encoder->forward_embeddings_taps(cid, x, n, tap_layers);
    cm->release(cid);
    if (taps.empty()) { return SharedBuffer{}; }

    // 4. drop the prefix rows + reorder to the DiT's [n_real, 12, 2560] f16
    //    (taps slot j is [n][EH] at j*n*EH).
    SharedBuffer ehs =
        mc->make_shared_buffer((std::size_t)n_real * NL * EH * 2);
    if (ehs.empty()) { return SharedBuffer{}; }
    {
      const auto* tp = static_cast<const std::uint16_t*>(taps.contents());
      auto* ep = static_cast<_Float16*>(ehs.contents());
      for (int p = 0; p < n_real; ++p) {
        for (int j = 0; j < NL; ++j) {
          const std::size_t src =
              ((std::size_t)j * n + (kDropPrefix + p)) * EH;
          const std::size_t dst = ((std::size_t)p * NL + j) * EH;
          for (int h = 0; h < EH; ++h) {
            std::uint32_t u = (std::uint32_t)tp[src + h] << 16;
            float f; std::memcpy(&f, &u, 4);
            ep[dst + h] = (_Float16)f;
          }
        }
      }
    }

    // 5. text-fusion tower + txt_in -> fused text [n_real, hidden].
    SharedBuffer fused = _dit->forward_text(ehs, n_real);
    if (fused.empty()) { return SharedBuffer{}; }
    n_real_out = n_real;
    return fused;
  };

  int n_real = 0;
  SharedBuffer fused = encode_text(prompt, "prompt", n_real);
  if (fused.empty()) { return {}; }
  session()->log_debug(fmt(
      "TextToImageStage('{}'): encoded prompt -> fused text [{}, hidden]; "
      "{}x{} grid {}x{} img_seq {}", this->id(), n_real, W, H, gh, gw,
      img_seq));

  // Classifier-free guidance: encode the negative prompt too and, at each
  // denoise step, push the velocity away from it. Skipped when no negative
  // prompt is wired or the scale is a no-op (1.0), so the single-pass turbo
  // default stays token-exact.
  bool cfg = !negative.empty() && _guidance_scale != 1.0;
  int n_real_neg = 0;
  SharedBuffer fused_neg;
  if (cfg) {
    fused_neg = encode_text(negative, "negative", n_real_neg);
    if (fused_neg.empty()) {
      session()->warn(fmt(
          "TextToImageStage('{}'): negative-prompt encode failed; running "
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
  std::vector<float> latent((std::size_t)16 * lh * lw);
  for (int c = 0; c < 16; ++c) {
    for (int y = 0; y < lh; ++y) {
      for (int xx = 0; xx < lw; ++xx) {
        const int a = y / 2, ph = y % 2, b = xx / 2, pw = xx % 2;
        const std::size_t t = (std::size_t)a * gw + b;
        latent[((std::size_t)c * lh + y) * lw + xx] =
            packed[t * IC + (std::size_t)c * 4 + ph * 2 + pw];
      }
    }
  }
  return latent;
}

std::vector<float>
TextToImageStage::generate_flux2_(const std::string& prompt, int gen_h,
                                  int gen_w,
                                  const std::vector<RefLatent>& refs) const
{
  auto* mc = session()->metal_compute();
  using metal_compute::SharedBuffer;
  const int H = gen_h, W = gen_w;
  const int gh = H / 16, gw = W / 16;        // VAE latent grid (128ch @ H/16)
  const int img_seq = gh * gw;
  const int IC = _flux2_dit->config().in_channels;   // 128
  const int EH = _enc_hidden;                        // 2560
  const int JD = _flux2_dit->config().joint_dim;     // 7680 = 3 * EH
  // FLUX.2 taps output.hidden_states[{9,18,27}] (the pipeline default
  // text_encoder_out_layers) and concatenates them -> joint_dim. Tap index into
  // forward_embeddings_taps is (hidden_states index - 1), matching Krea-2.
  static const int kFluxTaps[3] = {9, 18, 27};

  // FLUX.2 uses the plain Qwen3 chat template (no Krea-2 system prompt, no
  // prefix drop -- ALL rows feed the DiT). NOTE: enable_thinking=False; the
  // exact thinking-block tokens + the reference's pad-to-512 are NOT replicated
  // here, so this is not yet token-exact. VERIFY vs the tokenizer chat_template.
  const std::string templated = std::string("<|im_start|>user\n") + prompt +
                                "<|im_end|>\n<|im_start|>assistant\n";
  const std::vector<std::int32_t> ids =
      encode_with_specials_(*_tokenizer, templated);
  if (ids.empty()) { return {}; }
  const int n = (int)ids.size();

  // Gather encoder input rows (bf16) by id.
  SharedBuffer x = mc->make_shared_buffer((std::size_t)n * EH * 2);
  if (x.empty()) { return {}; }
  {
    const auto* tbl = static_cast<const std::uint8_t*>(_embed.contents());
    auto* xb = static_cast<std::uint8_t*>(x.contents());
    const std::size_t vocab = _embed.byte_size() / ((std::size_t)EH * 2);
    for (int i = 0; i < n; ++i) {
      const std::uint32_t id = (std::uint32_t)ids[(std::size_t)i];
      if (id >= vocab) { return {}; }
      std::memcpy(xb + (std::size_t)i * EH * 2,
                  tbl + (std::size_t)id * EH * 2, (std::size_t)EH * 2);
    }
  }

  // Tap the 3 selected layers -> concat per token into context [n, JD=7680].
  std::vector<int> tap_layers;
  for (int k : kFluxTaps) { tap_layers.push_back(k - 1); }
  genai::ContextManager* cm = _encoder->context_manager();
  const genai::ContextId cid = cm->acquire_root();
  SharedBuffer taps = _encoder->forward_embeddings_taps(cid, x, n, tap_layers);
  cm->release(cid);
  if (taps.empty()) { return {}; }
  SharedBuffer context = mc->make_shared_buffer((std::size_t)n * JD * 2);
  if (context.empty()) { return {}; }
  {
    const auto* tp = static_cast<const std::uint16_t*>(taps.contents());
    auto* cp = static_cast<_Float16*>(context.contents());
    for (int p = 0; p < n; ++p) {
      for (int j = 0; j < 3; ++j) {          // taps slot j is [n][EH] at j*n*EH
        const std::size_t src = ((std::size_t)j * n + p) * EH;
        const std::size_t dst = (std::size_t)p * JD + (std::size_t)j * EH;
        for (int h = 0; h < EH; ++h) {
          std::uint32_t u = (std::uint32_t)tp[src + h] << 16;
          float f; std::memcpy(&f, &u, 4);
          cp[dst + h] = (_Float16)f;
        }
      }
    }
  }
  const int n_real = n;

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
  std::unique_ptr<UiTextStream> bar = session()->open_text_stream();
  int last_pct = -1;
  const auto gen_t0 = std::chrono::steady_clock::now();
  denoise_progress_(bar.get(), 0, S, last_pct);
  for (int i = 0; i < S; ++i) {
    sampler.step(i, packed,
                 prof ? genai::FlowSampler::DenoiseFn(denoise_p) : denoise);
    if (!dit_ok) { bar->end(); return {}; }
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

  // Unpack [img_seq, IC] (token-major) -> channel-first [IC, gh, gw].
  std::vector<float> latent((std::size_t)IC * gh * gw);
  for (int i = 0; i < gh; ++i) {
    for (int j = 0; j < gw; ++j) {
      const std::size_t t = (std::size_t)i * gw + j;
      for (int cc = 0; cc < IC; ++cc) {
        latent[((std::size_t)cc * gh + i) * gw + j] =
            packed[t * IC + cc];
      }
    }
  }
  return latent;
}

Job
TextToImageStage::process(RuntimeContext& ctx)
{
  auto in = co_await ctx.read(0);
  if (!in) { ctx.signal_done(); co_return; }
  const bool have_dit = _family == "flux2" ? (bool)_flux2_dit : (bool)_dit;
  if (!have_dit) {
    session()->warn(fmt(
        "TextToImageStage('{}'): models not loaded; dropping beat", this->id()));
    co_return;
  }
  // Lazily reload the text encoder if it was freed after the previous
  // generation (see _free_encoder); a no-op (encoder resident) otherwise.
  if (!_encoder && !load_encoder_(session()->metal_compute())) {
    session()->warn(fmt(
        "TextToImageStage('{}'): text encoder reload failed; dropping beat",
        this->id()));
    co_return;
  }
  const auto* fdp = dynamic_cast<const FlexDataPayload*>(in.get());
  if (fdp == nullptr) {
    session()->warn(fmt(
        "TextToImageStage('{}'): expected a FlexDataPayload prompt, got {}; "
        "dropping beat", this->id(), in->describe()));
    co_return;
  }
  std::string prompt(fdp->data.as_string(""));
  if (prompt.empty() && fdp->data.is_object()) {
    auto obj = fdp->data.as_object();
    if (obj.contains("text")) { prompt = std::string(obj.at("text").as_string("")); }
  }
  if (prompt.empty()) {
    session()->warn(fmt(
        "TextToImageStage('{}'): empty prompt; dropping beat", this->id()));
    co_return;
  }
  session()->log_debug(fmt(
      "TextToImageStage('{}'): beat received, prompt ({} chars): '{}'",
      this->id(), prompt.size(), prompt));

  // iport1: OPTIONAL negative prompt for classifier-free guidance. Read one
  // beat per generation when available and cache it (`_negative_prompt`), so a
  // fixed negative supplied once is reused for every later prompt. The
  // non-blocking backlog gate avoids stalling when none is wired/pending.
  if (ctx.num_iports() >= 2 && ctx.iport_connected(1) && ctx.backlog(1) > 0) {
    auto nb = co_await ctx.read(1);
    const auto* nfd = nb ? dynamic_cast<const FlexDataPayload*>(nb.get())
                         : nullptr;
    if (nfd != nullptr) {
      std::string np(nfd->data.as_string(""));
      if (np.empty() && nfd->data.is_object()) {
        auto o = nfd->data.as_object();
        if (o.contains("text")) {
          np = std::string(o.at("text").as_string(""));
        }
      }
      _negative_prompt = std::move(np);
      session()->log_debug(fmt(
          "TextToImageStage('{}'): negative prompt ({} chars): '{}'",
          this->id(), _negative_prompt.size(), _negative_prompt));
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
  if (_family != "flux2" && _strength > 0.0 && !_ref[0].empty()) {
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
  if (_family == "flux2") {
    std::vector<RefLatent> frefs;
    if (!_ref[0].empty()) { frefs.push_back(_ref[0]); }
    if (!_ref[1].empty()) { frefs.push_back(_ref[1]); }
    const std::vector<float> fl =
        generate_flux2_(prompt, gen_h, gen_w, frefs);
    if (fl.empty()) {
      session()->warn(fmt(
          "TextToImageStage('{}'): FLUX.2 generation failed; dropping beat",
          this->id()));
      co_return;
    }
    // Conditioning is built and denoising is done -- the encoder is now idle
    // through the downstream VAE decode. Free it (reloaded lazily next beat) so
    // its footprint doesn't crowd out a large decode on a memory-bounded box.
    if (_free_encoder) { _encoder.reset(); _embed = {}; }
    const int Cdit = _flux2_dit->config().in_channels;
    const int fgh = gen_h / 16, fgw = gen_w / 16;
    auto out = std::make_unique<TensorBeatPayload>();
    out->dtype = TensorBeat::DType::F32;
    out->shape = {Cdit, fgh, fgw};
    out->resize_contiguous(fl.size());
    std::memcpy(out->as_f32(), fl.data(), fl.size() * sizeof(float));
    ++_latents_emitted;
    session()->info(fmt(
        "TextToImageStage('{}'): '{}' -> FLUX.2 latent [{}, {}, {}] ({} steps "
        "@ {}x{})", this->id(), prompt, Cdit, fgh, fgw,
        _scheduler_spec.steps, gen_h, gen_w));
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
  const std::vector<float> out_latent =
      generate_(prompt, _negative_prompt, gen_h, gen_w, init_ptr, latent_ptr,
                krefs);
  if (out_latent.empty()) {
    session()->warn(fmt(
        "TextToImageStage('{}'): generation failed; dropping beat", this->id()));
    co_return;
  }
  // Encoder idle through the downstream VAE decode -- free it (lazily reloaded
  // next beat) so it doesn't crowd out a large decode on a memory-bounded box.
  // The DiT is idle too now (latent read back): drop its per-forward scratch
  // (DitScratch activations + dequant/split-K + i8 accel buffers, ~1-2 GB at
  // 1024px), which regrows on the next generation. The DiT weights themselves
  // stay mmap'd (evictable). Together these free enough working set for the
  // separate vae-decode stage; without it a 1024px decode OOMs on a 16 GB box.
  if (_free_encoder) {
    _encoder.reset();
    _embed = {};
    if (_dit) { _dit->release_forward_scratch(); }
  }

  auto out = std::make_unique<TensorBeatPayload>();
  out->dtype = TensorBeat::DType::F32;
  out->shape = {16, lh, lw};
  out->resize_contiguous(out_latent.size());
  std::memcpy(out->as_f32(), out_latent.data(),
              out_latent.size() * sizeof(float));
  ++_latents_emitted;
  session()->info(fmt(
      "TextToImageStage('{}'): '{}' -> latent [16, {}, {}] ({}+{} {} steps @ "
      "{}x{})", this->id(), prompt, lh, lw, _sampler_spec.method,
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

#include "stages/diffusion-conditioner-stage.h"
#include <chrono>

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/perf-scope.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "stages/model-registry.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/context-manager.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/model-loader.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#endif

#include <memory>
#include <string>
#include <vector>

namespace vpipe {

namespace {

// The model iport (a model-select source) overrides hf_dir. Inserted after the
// prompt + negative inputs, so it is iport2 (ref_image follows at iport3).
// (Referenced only from the Apple-gated code; maybe_unused for non-Apple.)
[[maybe_unused]] constexpr unsigned kModelPort = 2;

const ConfigKey kAttrs[] = {
  {.key = "hf_dir", .type = ConfigType::String, .required = false,
   .doc = "model dir (text_encoder/, transformer/, tokenizer/); the "
          "transformer's _class_name selects the family + encoder. OPTIONAL: a "
          "model-select source on the model iport overrides it",
   .suggest_db = "models",
   .suggest_db_type = "krea2,flux2,qwen-image-edit,mage-flow,mage-flow-edit"},
  {.key = "models_db", .type = ConfigType::String, .required = false,
   .doc = "model registry db for resolve_model_dir (default \"models\")"},
  {.key = "grounded_negative", .type = ConfigType::Bool, .required = false,
   .doc = "image-aware families only: always emit a negative conditioning on "
          "oport1 -- a GROUNDED encode of the (possibly empty) negative prompt "
          "-- so the DiT can run classifier-free guidance (CFG>1). Matches the "
          "Krea-2 edit deletion recipe (empty grounded negative). Default false "
          "(emit a negative only when a non-empty negative prompt is wired)"},
};
const PortSpec kIports[] = {
  {.name = "prompt", .doc = "prompt text (FlexData string or {text: ...})",
   .type = &typeid(FlexDataPayload), .tags = "text", .clock_group = 0},
  {.name = "negative", .doc = "OPTIONAL negative prompt (FlexData) for the DiT's "
                              "classifier-free guidance; its conditioning is "
                              "emitted on oport1",
   .type = &typeid(FlexDataPayload), .tags = "text", .clock_group = 0},
  {.name = "model", .doc = "OPTIONAL shared model reference from a model-select "
                           "source; overrides the hf_dir config",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
  {.name = "ref_image", .doc = "OPTIONAL raw reference image (planar U8 RGB "
                               "TensorBeat [3,H,W], load-image format). Image-"
                               "aware families (Qwen-Image-Edit) run it through "
                               "the Qwen2.5-VL vision tower; others ignore it.",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "conditioning",
   .doc = "conditioning tensor for the text-to-image DiT (family-shaped + typed: "
          "krea2 f16 [n,12,2560]; flux2 f16 [n,3*enc_hidden]; qwen-image-edit "
          "bf16 [n_real,3584] image-aware; mage-flow bf16 [n_real,2560] "
          "image-aware)",
   .type = &typeid(TensorBeatPayload), .tags = "conditioning", .clock_group = 0},
  {.name = "neg_conditioning",
   .doc = "conditioning for the negative prompt (same shape/type); emitted only "
          "when a negative prompt is wired",
   .type = &typeid(TensorBeatPayload), .tags = "conditioning", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "diffusion-conditioner",
  .doc       = "Prompt (+ optional reference image) -> conditioning embeddings "
               "for a diffusion DiT. Owns the tokenizer + text encoder + (for "
               "image-aware models) the Qwen2.5-VL vision tower. The encoder "
               "half of the text-to-image split; pair it with a text-to-image "
               "stage on the same hf_dir.",
  .display_name = "Diffusion Conditioner",
  .category  = StageCategory::Generative,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

#ifdef VPIPE_BUILD_APPLE_SILICON

using metal_compute::SharedBuffer;

inline std::uint16_t f32_to_bf16_(float f)
{
  std::uint32_t u; std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
inline float bf16_to_f32_(std::uint16_t b)
{
  std::uint32_t u = (std::uint32_t)b << 16;
  float f; std::memcpy(&f, &u, 4); return f;
}

// Downscale planar U8 RGB [3,H,W] so the longest side <= `cap`, preserving the
// aspect ratio with an area-average box filter (matching the ComfyUI grounded-
// encode node's "area" grounding resize). Writes into `out` + sets *oh/*ow when
// it resizes; leaves `out` empty (no-op, caller keeps the original) when the
// image already fits. The Qwen3-VL tower smart-resizes internally, but its cap
// (~1M px area) is looser than the node's 768 grounding_px default, so bound the
// longest side here to stay in the LoRA's 384-768px training distribution.
void cap_longest_side_(const std::uint8_t* rgb, int H, int W, int cap,
                       std::vector<std::uint8_t>& out, int* oh, int* ow)
{
  *oh = H; *ow = W;
  const int longest = std::max(H, W);
  if (longest <= cap || longest <= 0 || H <= 0 || W <= 0) { return; }
  const double s = (double)cap / (double)longest;
  const int nh = std::max(1, (int)std::lround(H * s));
  const int nw = std::max(1, (int)std::lround(W * s));
  out.assign((std::size_t)3 * nh * nw, 0);
  for (int c = 0; c < 3; ++c) {
    const std::uint8_t* src = rgb + (std::size_t)c * H * W;
    std::uint8_t* dst = out.data() + (std::size_t)c * nh * nw;
    for (int y = 0; y < nh; ++y) {
      const int y0 = (int)((double)y * H / nh);
      const int y1 = std::max(y0 + 1, (int)((double)(y + 1) * H / nh));
      for (int x = 0; x < nw; ++x) {
        const int x0 = (int)((double)x * W / nw);
        const int x1 = std::max(x0 + 1, (int)((double)(x + 1) * W / nw));
        std::uint32_t acc = 0, cnt = 0;
        for (int yy = y0; yy < y1 && yy < H; ++yy) {
          for (int xx = x0; xx < x1 && xx < W; ++xx) {
            acc += src[(std::size_t)yy * W + xx]; ++cnt;
          }
        }
        dst[(std::size_t)y * nw + x] = (std::uint8_t)(cnt ? acc / cnt : 0u);
      }
    }
  }
  *oh = nh; *ow = nw;
}

// ---- Prompt templates (shared with the text-to-image stage) ----
constexpr const char* kPrefix =
    "<|im_start|>system\nDescribe the image by detailing the color, shape, "
    "size, texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n";
constexpr const char* kSuffix = "<|im_end|>\n<|im_start|>assistant\n";
constexpr int kDropPrefix = 34;
const int kSelectLayers[12] = {2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35};
const int kFluxTaps[3] = {9, 18, 27};

constexpr const char* kQiePrefix =
    "<|im_start|>system\nDescribe the key features of the input image (color, "
    "shape, size, texture, objects, background), then explain how the user's "
    "text instruction should alter or modify the image. Generate a new image "
    "that meets the user's requirements while maintaining consistency with the "
    "original input where appropriate.<|im_end|>\n<|im_start|>user\n";
constexpr const char* kQieSuffix = "<|im_end|>\n<|im_start|>assistant\n";
constexpr int kQieDropPrefix = 64;

// Mage-Flow reuses BOTH templates verbatim -- its models/utils.py
// PROMPT_TEMPLATE["mage-flow"] is byte-identical to kPrefix + "{}" + kSuffix
// (start_idx 34 == kDropPrefix) and PROMPT_TEMPLATE["mage-flow-edit"] to
// kQiePrefix + "{}" + kQieSuffix (start_idx 64 == kQieDropPrefix); both
// descend from the same Qwen-Image conventions. Only the multi-reference body
// is its own: `Image {j}: <|vision_start|><|image_pad|><|vision_end|>` per
// reference (no separator), then the instruction -- pipeline.py
// _edit_prompt_body. Qwen-Image-Edit says "Picture 1: " instead.
constexpr const char* kMageRefPlaceholder =
    "<|vision_start|><|image_pad|><|vision_end|>";

// Special-token-aware encode: split at the markers, BPE each text run, splice
// the markers' ids (matches the HF fast tokenizer). Includes the Qwen2.5-VL
// vision markers so the QIE image-aware template isolates them.
std::vector<std::int32_t>
encode_with_specials_(const genai::Tokenizer& tok, const std::string& text)
{
  static const char* kMarkers[] = {"<|im_start|>", "<|im_end|>",
                                   "<|vision_start|>", "<|vision_end|>",
                                   "<|image_pad|>", "<think>", "</think>"};
  static const int kN = (int)(sizeof(kMarkers) / sizeof(kMarkers[0]));
  std::vector<std::int32_t> out;
  std::size_t pos = 0;
  while (pos < text.size()) {
    std::size_t best = std::string::npos;
    int which = -1;
    for (int mi = 0; mi < kN; ++mi) {
      const std::size_t f = text.find(kMarkers[mi], pos);
      if (f != std::string::npos && (best == std::string::npos || f < best)) {
        best = f; which = mi;
      }
    }
    if (which < 0) {
      const auto seg = tok.encode(text.substr(pos));
      out.insert(out.end(), seg.begin(), seg.end());
      break;
    }
    if (best > pos) {
      const auto seg = tok.encode(text.substr(pos, best - pos));
      out.insert(out.end(), seg.begin(), seg.end());
    }
    const std::int32_t sid = tok.special_token_id(kMarkers[which]);
    if (sid >= 0) { out.push_back(sid); }
    pos = best + std::strlen(kMarkers[which]);
  }
  return out;
}

// ---- Encoder configs (per family) ----
genai::MetalQwenModel::Config encoder_config_krea2_()
{
  genai::MetalQwenModel::Config c;
  c.n_layers = 36; c.hidden = 2560; c.n_heads = 32; c.n_kv_heads = 8;
  c.head_dim = 128; c.ffn_inner = 9728; c.vocab = 151936; c.rope_theta = 5.0e6f;
  c.rms_eps = 1e-6f; c.rotary_dim = 128; c.full_attn_interval = 1;
  c.tie_embeddings = true; c.use_bf16 = true; c.dense = true;
  c.zero_centered_norm = false; c.attn_output_gate = false;
  c.backbone_only = true; c.weight_prefix = "language_model."; c.model_seg = "";
  c.max_seq = 1024; c.page_tokens = 256;
  return c;
}
genai::MetalQwenModel::Config encoder_config_flux2_(const std::string& enc_dir)
{
  genai::MetalQwenModel::Config c = encoder_config_krea2_();
  c.rope_theta = 1.0e6f; c.weight_prefix = "model.";
  namespace fs = std::filesystem;
  std::ifstream in(fs::path(enc_dir) / "config.json");
  if (in) {
    FlexData fd = FlexData::from_json(in);
    if (fd.is_object()) {
      auto o = fd.as_object();
      auto geti = [&](const char* k, int cur) {
        return o.contains(k) ? (int)o.at(k).as_int(cur) : cur; };
      auto getf = [&](const char* k, float cur) {
        return o.contains(k) ? (float)o.at(k).as_real(cur) : cur; };
      c.n_layers = geti("num_hidden_layers", c.n_layers);
      c.hidden = geti("hidden_size", c.hidden);
      c.n_heads = geti("num_attention_heads", c.n_heads);
      c.n_kv_heads = geti("num_key_value_heads", c.n_kv_heads);
      c.head_dim = geti("head_dim",
                        c.n_heads > 0 ? c.hidden / c.n_heads : c.head_dim);
      c.rotary_dim = c.head_dim;
      c.ffn_inner = geti("intermediate_size", c.ffn_inner);
      c.vocab = geti("vocab_size", c.vocab);
      c.rope_theta = getf("rope_theta", c.rope_theta);
      c.rms_eps = getf("rms_norm_eps", c.rms_eps);
      if (o.contains("tie_word_embeddings")) {
        c.tie_embeddings = o.at("tie_word_embeddings").as_bool(c.tie_embeddings);
      }
    }
  }
  return c;
}
// Mage-Flow's text encoder IS the same Qwen3-VL 4B krea2 drives; the only
// difference is the checkpoint layout -- Mage-Flow wraps everything in
// `model.` (398 LM tensors under "model.language_model.", 315 tower tensors
// under "model.visual."), krea2 omits that wrapper.
genai::MetalQwenModel::Config encoder_config_mage_()
{
  genai::MetalQwenModel::Config c = encoder_config_krea2_();
  c.weight_prefix = "model.language_model.";
  return c;
}
genai::MetalQwenModel::Config encoder_config_qie_()
{
  genai::MetalQwenModel::Config c;
  c.n_layers = 28; c.hidden = 3584; c.n_heads = 28; c.n_kv_heads = 4;
  c.head_dim = 128; c.ffn_inner = 18944; c.vocab = 152064; c.rope_theta = 1.0e6f;
  c.rms_eps = 1e-6f; c.rotary_dim = 128; c.full_attn_interval = 1;
  c.tie_embeddings = false; c.use_bf16 = true; c.dense = true;
  c.zero_centered_norm = false; c.attn_output_gate = false;
  c.qk_norm = false; c.attention_bias = true; c.backbone_only = true;
  c.weight_prefix = ""; c.model_seg = "model."; c.max_seq = 1024;
  c.page_tokens = 256;
  return c;
}

// The transformer family from <root>/transformer/config.json `_class_name`.
std::string family_(const std::string& transformer_dir)
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
        // Mage-Flow (microsoft/Mage-Flow*). Named EXPLICITLY, never left to
        // the "krea2" default: its text encoder is the same Qwen3-VL 4B krea2
        // drives, so an unrecognized repo would LOAD and silently produce
        // krea2's 12-tap conditioning instead of Mage-Flow's single
        // last-hidden tap (and with the wrong weight prefix).
        if (cls == "MageFlow") { return "mage-flow"; }
      }
    }
  }
  return "krea2";
}

// Extract prompt text from a FlexData beat (string or {text: ...}).
std::string flex_text_(const FlexData& fd)
{
  if (fd.is_string()) { return std::string(fd.as_string("")); }
  if (fd.is_object()) {
    auto o = fd.as_object();
    if (o.contains("text")) { return std::string(o.at("text").as_string("")); }
  }
  return "";
}

#endif  // VPIPE_BUILD_APPLE_SILICON

}  // namespace

DiffusionConditionerStage::DiffusionConditionerStage(
    const SessionContextIntf* s, std::string id, std::vector<InEdge> iports,
    FlexData config)
  : TypedStage<DiffusionConditionerStage>(s, std::move(id), std::move(iports),
                                          std::move(config))
{
  // hf_dir is OPTIONAL: a model-select source on the model iport can supply it
  // instead, so "no model at all" is reported at initialize()/process() time
  // (when iport connectivity is known), not at construction.
  _hf_dir    = attr_str("hf_dir");
  _models_db = attr_str("models_db");
  if (_models_db.empty()) { _models_db = "models"; }
  _grounded_negative = attr_bool("grounded_negative");
  allocate_oports(spec().oports.size());
}

DiffusionConditionerStage::~DiffusionConditionerStage() = default;

const StageSpec&
DiffusionConditionerStage::spec() const noexcept
{
  return kSpec;
}

#ifndef VPIPE_BUILD_APPLE_SILICON
Job DiffusionConditionerStage::initialize(RuntimeContext&) { co_return; }
Job DiffusionConditionerStage::process(RuntimeContext&) { co_return; }
#else

bool
DiffusionConditionerStage::load_encoder_(metal_compute::MetalCompute* mc)
{
  genai::MetalQwenModel::Config ecfg =
      _family == "flux2" ? encoder_config_flux2_(_enc_dir)
      : _family == "qwen-image-edit" ? encoder_config_qie_()
      : _family == "mage-flow" ? encoder_config_mage_()
      : encoder_config_krea2_();
  _enc_hidden = ecfg.hidden;
  // The encoder may be affine-quantized (model-quantize target=text_encoder).
  // The loader auto-detects quantized-vs-dense weights but needs the bit-width
  // to pick the w4g64 vs w8g64 kernel, so read it from the encoder's
  // config.json quantization block (absent => dense bf16, quant_bits unused).
  {
    std::ifstream in(std::filesystem::path(_enc_dir) / "config.json");
    if (in) {
      FlexData fd = FlexData::from_json(in);
      if (fd.is_object()) {
        auto o = fd.as_object();
        if (o.contains("quantization")) {
          FlexData q = o.at("quantization");
          if (q.is_object()) {
            auto qo = q.as_object();
            const int b = qo.contains("bits") ? (int)qo.at("bits").as_int(0) : 0;
            if (b == 4 || b == 8) { ecfg.quant_bits = b; }
          }
        }
      }
    }
  }
  _encoder = genai::MetalQwenModel::load(_enc_dir, mc, ecfg);
  if (!_encoder) {
    session()->error(fmt("DiffusionConditionerStage('{}'): text encoder load "
                         "failed: {}", this->id(), _enc_dir));
    return false;
  }
  auto wts = genai::MetalLlamaWeights::open_model(_enc_dir);
  if (!wts.has_value()) { return false; }
  const std::string emb_name =
      (_family == "flux2" || _family == "qwen-image-edit")
          ? "model.embed_tokens.weight"
      : _family == "mage-flow" ? "model.language_model.embed_tokens.weight"
                               : "language_model.embed_tokens.weight";
  _embed = wts->load(emb_name, mc);
  return !_embed.empty();
}

Job
DiffusionConditionerStage::initialize(RuntimeContext& ctx)
{
  // Defer the encoder load when a model-select source feeds the model iport
  // (its beat only arrives after the init barrier, in process()). Otherwise
  // load now from the config hf_dir, as before.
  const bool model_from_iport =
      ctx.num_iports() > kModelPort && ctx.iport_connected(kModelPort);
  if (!model_from_iport) { ensure_loaded_(); }
  co_return;
}

void
DiffusionConditionerStage::ensure_loaded_()
{
  if (_load_attempted) { return; }   // idempotent: load at most once
  _load_attempted = true;
  if (_hf_dir.empty()) {
    session()->error(fmt("DiffusionConditionerStage('{}'): no model -- set "
        "config.hf_dir or wire a model-select source to the model iport; "
        "inert", this->id()));
    return;
  }
  auto* mc = session()->metal_compute();
  if (mc == nullptr) {
    session()->error(fmt(
        "DiffusionConditionerStage('{}'): no metal-compute backend; inert",
        this->id()));
    return;
  }
  const std::string root = resolve_model_dir(session(), _models_db, _hf_dir);
  _enc_dir = (std::filesystem::path(root) / "text_encoder").string();
  _family = family_((std::filesystem::path(root) / "transformer").string());

  namespace fs = std::filesystem;
  std::string tok_path = (fs::path(root) / "tokenizer" / "tokenizer.json").string();
  if (!fs::exists(tok_path)) {
    tok_path = (fs::path(root) / "processor" / "tokenizer.json").string();
  }
  if (!fs::exists(tok_path)) {
    // Mage-Flow ships no separate tokenizer/ or processor/ dir -- the
    // Qwen3-VL tokenizer + processor configs live beside the weights.
    tok_path = (fs::path(_enc_dir) / "tokenizer.json").string();
  }
  _tokenizer = genai::Tokenizer::from_huggingface_json(tok_path, session());
  if (!_tokenizer) {
    session()->error(fmt(
        "DiffusionConditionerStage('{}'): tokenizer load failed: {}; inert",
        this->id(), tok_path));
    return;
  }
  if (!load_encoder_(mc)) {
    session()->error(fmt(
        "DiffusionConditionerStage('{}'): encoder/embeds load failed; inert",
        this->id()));
    return;
  }
  session()->info(fmt(
      "DiffusionConditionerStage('{}'): family {} encoder ({}), hidden {}",
      this->id(), _family,
      _family == "flux2" ? "Qwen3 dense"
      : _family == "qwen-image-edit" ? "Qwen2.5-VL" : "Qwen3-VL", _enc_hidden));
}

// True for the families whose conditioning is a SINGLE post-final-norm
// last-hidden tap [n_real, hidden] (bf16), as opposed to krea2's 12-tap stack
// or flux2's 3-tap concat.
static bool
single_tap_(const std::string& family)
{
  return family == "qwen-image-edit" || family == "mage-flow";
}

SharedBuffer
DiffusionConditionerStage::vision_tokens_(metal_compute::MetalCompute* mc,
                                          int& n_img) const
{
  n_img = 0;
  if (_ref_rgb.empty()) { return {}; }

  // Qwen-Image-Edit: Qwen2.5-VL tower -> bf16 [n_img, 3584].
  if (_family == "qwen-image-edit") {
    if (!_vision) {
      genai::MetalQwen25Vision::Config vcfg;
      _vision = genai::MetalQwen25Vision::load(_enc_dir, mc, vcfg);
      if (!_vision) {
        session()->warn(fmt("DiffusionConditionerStage('{}'): vision tower load "
                            "failed; text-only conditioning", this->id()));
        return {};
      }
    }
    int vgh = 0, vgw = 0;
    SharedBuffer vt = _vision->encode_rgb(_ref_rgb.data(), _ref_rgb_h,
                                          _ref_rgb_w, 384 * 384, vgh, vgw);
    if (vt.empty()) { return {}; }
    const int mm = _vision->config().merge;
    n_img = (vgh / mm) * (vgw / mm);
    session()->info(fmt(
        "DiffusionConditionerStage('{}'): image-aware conditioning -> {} vision "
        "tokens (grid {}x{})", this->id(), n_img, vgh, vgw));
    return vt;
  }

  // Krea-2 edit (identity-edit LoRA): Qwen3-VL tower -> f16 [n_img, 2560]. The
  // instruction is encoded WITH the source image (training-matched grounded
  // encode); the raw source RGB comes through the ref_image iport. The 315
  // visual.* tower tensors ship inside text_encoder/, so it loads from _enc_dir.
  // Mage-Flow rides the SAME Qwen3-VL tower + deepstack path as krea2; only
  // the checkpoint prefix ("model.visual." vs "visual."), the conditioning
  // long-edge cap (384 vs 768) and the processor's min_pixels differ.
  if (_family == "krea2" || _family == "mage-flow") {
    const bool mage = (_family == "mage-flow");
    if (!_vision3) {
      genai::ModelLoader loader(session());
      const auto mcfg = loader.load_config(_enc_dir);
      if (!mcfg.has_value()) {
        session()->warn(fmt("DiffusionConditionerStage('{}'): Qwen3-VL config "
                            "parse failed; text-only conditioning", this->id()));
        return {};
      }
      auto vcfg = genai::MetalQwenVisionEncoder::config_from(*mcfg);
      vcfg.weight_prefix = mage ? "model.visual." : "visual.";
      if (mage) {
        // Mage-Flow's preprocessor_config.json sets size.shortest_edge 65536
        // (vs the Qwen processor default 3136), so a small or very wide
        // reference is UPSCALED before patching. Past ~2.25:1 aspect the
        // 384-capped image falls under 65536 px and the default bound would
        // silently skip that upscale.
        vcfg.min_pixels = 65536;
        vcfg.max_pixels = 16777216;
      }
      // Qwen3-VL normalizes images to [-1,1] (mean/std 0.5), NOT the OpenAI-CLIP
      // defaults the loader assumes when the repo ships no preprocessor_config.
      // With CLIP normalization the vision tokens -- hence the grounded image
      // conditioning -- are wrong, and the edit mis-targets (removes the
      // foreground subject instead of the background). Qwen3-VL's official
      // preprocessor_config uses 0.5/0.5.
      for (int i = 0; i < 3; ++i) {
        vcfg.image_mean[i] = 0.5f;
        vcfg.image_std[i] = 0.5f;
      }
      _vision3 = genai::MetalQwenVisionEncoder::load(_enc_dir, mc, vcfg);
      if (!_vision3) {
        session()->warn(fmt("DiffusionConditionerStage('{}'): Qwen3-VL vision "
                            "tower load failed; text-only conditioning",
                            this->id()));
        return {};
      }
      _vision3->set_session(session());
    }
    std::vector<std::uint8_t> capped;
    int rh = _ref_rgb_h, rw = _ref_rgb_w;
    // Mage-Flow caps the VL conditioning image's long edge at 384 (its
    // training preprocessing -- pipeline.py `vl_cond_long_edge`); the VAE
    // reference path keeps the full target resolution. krea2's grounding
    // node uses 768.
    cap_longest_side_(_ref_rgb.data(), _ref_rgb_h, _ref_rgb_w, mage ? 384 : 768,
                      capped, &rh, &rw);
    const std::uint8_t* rgb = capped.empty() ? _ref_rgb.data() : capped.data();
    auto r = _vision3->encode(rgb, rh, rw);
    if (r.embeddings.empty() || r.n_tokens <= 0) { return {}; }
    n_img = r.n_tokens;
    // Merged grid (mh, mw) for the image tokens' 2-D mROPE positions. The
    // encoder returns the PATCH grid; the LM tokens are the S x S-merged set
    // in merger (row-major mh x mw) order.
    const int S = _vision3->config().spatial_merge > 0
                      ? _vision3->config().spatial_merge : 2;
    _img_mh = r.grid_h / S;
    _img_mw = r.grid_w / S;
    // Deepstack features (f16 [n_img, EH]) -> bf16 (the encoder residual dtype),
    // for injection into the text encoder at layers 0.. (see encode_).
    _ds_feats.clear();
    for (auto& df : r.deepstack) {
      if (df.empty()) { continue; }
      const std::size_t ne = (std::size_t)n_img * _enc_hidden;
      SharedBuffer b = mc->make_shared_buffer(ne * 2);
      if (b.empty()) { _ds_feats.clear(); break; }
      const auto* s = static_cast<const _Float16*>(df.contents());
      auto* d = static_cast<std::uint16_t*>(b.contents());
      for (std::size_t i = 0; i < ne; ++i) { d[i] = f32_to_bf16_((float)s[i]); }
      _ds_feats.push_back(std::move(b));
    }
    session()->info(fmt(
        "DiffusionConditionerStage('{}'): image-grounded conditioning -> {} "
        "vision tokens (grid {}x{}), {} deepstack", this->id(), n_img,
        r.grid_h, r.grid_w, _ds_feats.size()));
    return std::move(r.embeddings);
  }

  return {};   // flux2 etc.: text-only
}

SharedBuffer
DiffusionConditionerStage::encode_(const std::string& text, const char* which,
                                   int& n_real_out,
                                   const SharedBuffer& vtok, int n_img) const
{
  auto* mc = session()->metal_compute();
  const int EH = _enc_hidden;

  if (_family == "flux2") {
    const int JD = 3 * EH;
    // Match diffusers Flux2Klein encode_prompt: apply_chat_template with
    // enable_thinking=False appends the Qwen3 empty thinking block after the
    // assistant turn. Omitting its 4 tokens (the causal encoder's most-
    // contextualized) drifts the conditioning -> degraded edits.
    const std::string tmpl = std::string("<|im_start|>user\n") + text +
                             "<|im_end|>\n<|im_start|>assistant\n"
                             "<think>\n\n</think>\n\n";
    auto ids = encode_with_specials_(*_tokenizer, tmpl);
    if (ids.empty()) { return {}; }
    const int real_n = (int)ids.size();   // real tokens, before padding
    // diffusers Flux2Klein tokenizes with padding="max_length", max_length=512
    // (pad id <|endoftext|>) and feeds ALL 512 Qwen3 hidden states to the DiT
    // (the DiT does NOT mask the conditioning). Under right-padding + causal
    // attention the 19 real tokens are unaffected; the padding positions carry
    // their own position-dependent hidden states that the DiT still attends to.
    // Emitting only the real tokens (the old behavior) fed the DiT a 19-token
    // conditioning where the reference feeds 512 -> a per-step attention bias
    // that compounds over the trajectory (color cast + composition drift).
    {
      const int kFluxMaxSeq = 512;
      std::int32_t pad = _tokenizer->special_token_id("<|endoftext|>");
      if (pad < 0) { pad = 151643; }
      if ((int)ids.size() > kFluxMaxSeq) { ids.resize(kFluxMaxSeq); }
      else { ids.resize(kFluxMaxSeq, pad); }
    }
    const int n = (int)ids.size();
    SharedBuffer x = mc->make_shared_buffer((std::size_t)n * EH * 2);
    if (x.empty()) { return {}; }
    const auto* tbl = static_cast<const std::uint8_t*>(_embed.contents());
    auto* xb = static_cast<std::uint8_t*>(x.contents());
    const std::size_t vocab = _embed.byte_size() / ((std::size_t)EH * 2);
    for (int i = 0; i < n; ++i) {
      const std::uint32_t id = (std::uint32_t)ids[(std::size_t)i];
      if (id >= vocab) { return {}; }
      std::memcpy(xb + (std::size_t)i * EH * 2, tbl + (std::size_t)id * EH * 2,
                  (std::size_t)EH * 2);
    }
    std::vector<int> taps_l; for (int k : kFluxTaps) { taps_l.push_back(k - 1); }
    genai::ContextManager* cm = _encoder->context_manager();
    const genai::ContextId cid = cm->acquire_root();
    // key_valid_len = real_n: the padding tokens attend ONLY to the real
    // prefix (HF attention_mask semantics), matching diffusers' 512-token
    // conditioning exactly (vs plain causal, where padding attends to padding).
    const auto _enc_t0 = std::chrono::steady_clock::now();
    // LLM-lane perf event (perf-visualizer): the DiT text-conditioning encoder
    // prefill; value = the padded conditioning length.
    SharedBuffer taps;
    {
      PerfAuxScope _perf(session(), kPerfLaneLLM, kGvidLlmDitText,
                         kPerfLlmDitTextBegin, (std::uint64_t)n);
      taps = _encoder->forward_embeddings_taps(cid, x, n, taps_l, real_n);
    }
    if (std::getenv("VPIPE_COND_PROFILE")) {
      const double ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - _enc_t0).count();
      session()->log_normal(fmt("DiffusionConditionerStage: encoder prefill "
                                "n={} real={} -> {:.1f} ms", n, real_n, ms));
    }
    cm->release(cid);
    if (taps.empty()) { return {}; }
    // The FLUX.2 DiT consumes f16 context ([n, 3*EH]); convert bf16 taps to
    // _Float16 here (byte-identical to the old inline encode path).
    SharedBuffer ctxb = mc->make_shared_buffer((std::size_t)n * JD * 2);
    const auto* tp = static_cast<const std::uint16_t*>(taps.contents());
    auto* cp = static_cast<_Float16*>(ctxb.contents());
    for (int p = 0; p < n; ++p) {
      for (int j = 0; j < 3; ++j) {
        const std::size_t src = ((std::size_t)j * n + p) * EH;
        const std::size_t dst = (std::size_t)p * JD + (std::size_t)j * EH;
        for (int h = 0; h < EH; ++h) {
          cp[dst + h] = (_Float16)bf16_to_f32_(tp[src + h]);
        }
      }
    }
    n_real_out = n;
    session()->log_debug(fmt("DiffusionConditionerStage('{}'): [{}] flux2 -> "
                             "[{}, {}] f16", this->id(), which, n, JD));
    return ctxb;
  }

  if (_family == "qwen-image-edit") {
    const int TD = EH;   // 3584
    const int NL = 28;
    const bool img_aware = (n_img > 0) && !vtok.empty();
    const std::int32_t pad_id =
        img_aware ? _tokenizer->special_token_id("<|image_pad|>") : -1;
    std::string tmpl =
        std::string(kQiePrefix) +
        (img_aware && pad_id >= 0
             ? "Picture 1: <|vision_start|><|image_pad|><|vision_end|>" : "") +
        text + std::string(kQieSuffix);
    std::vector<std::int32_t> ids = encode_with_specials_(*_tokenizer, tmpl);
    if (img_aware && pad_id >= 0) {
      std::vector<std::int32_t> ex; ex.reserve(ids.size() + (std::size_t)n_img);
      for (const std::int32_t id : ids) {
        if (id == pad_id) { for (int j = 0; j < n_img; ++j) ex.push_back(pad_id); }
        else { ex.push_back(id); }
      }
      ids.swap(ex);
    }
    if ((int)ids.size() <= kQieDropPrefix) { return {}; }
    const int n = (int)ids.size();
    const int n_real = n - kQieDropPrefix;
    SharedBuffer x = mc->make_shared_buffer((std::size_t)n * EH * 2);
    if (x.empty()) { return {}; }
    {
      const auto* tbl = static_cast<const std::uint8_t*>(_embed.contents());
      auto* xb = static_cast<std::uint8_t*>(x.contents());
      const std::size_t vocab = _embed.byte_size() / ((std::size_t)EH * 2);
      for (int i = 0; i < n; ++i) {
        const std::uint32_t id = (std::uint32_t)ids[(std::size_t)i];
        if (id >= vocab) { return {}; }
        std::memcpy(xb + (std::size_t)i * EH * 2, tbl + (std::size_t)id * EH * 2,
                    (std::size_t)EH * 2);
      }
    }
    if (img_aware && pad_id >= 0) {
      const auto* vt = static_cast<const std::uint16_t*>(vtok.contents());
      auto* xh = static_cast<std::uint16_t*>(x.contents());
      int j = 0;
      for (int i = 0; i < n && j < n_img; ++i) {
        if (ids[(std::size_t)i] == pad_id) {
          std::memcpy(xh + (std::size_t)i * EH, vt + (std::size_t)j * EH,
                      (std::size_t)EH * 2);
          ++j;
        }
      }
    }
    genai::ContextManager* cm = _encoder->context_manager();
    const genai::ContextId cid = cm->acquire_root();
    SharedBuffer taps;
    {   // LLM-lane perf event: DiT text-conditioning encoder prefill.
      PerfAuxScope _perf(session(), kPerfLaneLLM, kGvidLlmDitText,
                         kPerfLlmDitTextBegin, (std::uint64_t)n);
      taps =
          _encoder->forward_embeddings_taps(cid, x, n, std::vector<int>{NL - 1});
    }
    cm->release(cid);
    if (taps.empty()) { return {}; }
    // final-RMSNorm weight (host-applied): load once per call (cheap).
    std::vector<float> fnorm(EH, 1.0f);
    {
      auto wts = genai::MetalLlamaWeights::open_model(_enc_dir);
      if (wts.has_value()) {
        SharedBuffer nw = wts->load("model.norm.weight", mc);
        if (!nw.empty()) {
          const auto* p = static_cast<const std::uint16_t*>(nw.contents());
          for (int h = 0; h < EH; ++h) { fnorm[h] = bf16_to_f32_(p[h]); }
        }
      }
    }
    SharedBuffer txt = mc->make_shared_buffer((std::size_t)n_real * TD * 2);
    const auto* tp = static_cast<const std::uint16_t*>(taps.contents());
    auto* op = static_cast<std::uint16_t*>(txt.contents());
    for (int p = 0; p < n_real; ++p) {
      const auto* row = tp + (std::size_t)(p + kQieDropPrefix) * EH;
      double ss = 0.0;
      for (int h = 0; h < EH; ++h) { const double v = bf16_to_f32_(row[h]);
                                     ss += v * v; }
      const double inv = 1.0 / std::sqrt(ss / (double)EH + 1e-6);
      for (int h = 0; h < TD; ++h) {
        op[(std::size_t)p * TD + h] =
            f32_to_bf16_((float)(bf16_to_f32_(row[h]) * inv * fnorm[h]));
      }
    }
    n_real_out = n_real;
    session()->log_debug(fmt("DiffusionConditionerStage('{}'): [{}] qie -> "
                             "[{}, {}]{}", this->id(), which, n_real, TD,
                             img_aware ? ", image-aware" : ""));
    return txt;
  }

  if (_family == "mage-flow") {
    const int NL = 36;   // Qwen3-VL 4B layers
    const bool img_aware = (n_img > 0) && !vtok.empty();
    const std::int32_t pad_id =
        img_aware ? _tokenizer->special_token_id("<|image_pad|>") : -1;
    const bool grounded = img_aware && pad_id >= 0;
    // Edit template (drop 64) when a reference rides along, t2i template
    // (drop 34) otherwise -- the reference picks the template from the CALL
    // (generate_edits vs generate_images), which is the same distinction.
    const int drop = grounded ? kQieDropPrefix : kDropPrefix;
    const std::string tmpl =
        (grounded ? std::string(kQiePrefix) + "Image 1: " + kMageRefPlaceholder
                  : std::string(kPrefix)) +
        text + (grounded ? std::string(kQieSuffix) : std::string(kSuffix));
    std::vector<std::int32_t> ids = encode_with_specials_(*_tokenizer, tmpl);
    if (grounded) {   // expand the single pad id to n_img copies
      std::vector<std::int32_t> ex; ex.reserve(ids.size() + (std::size_t)n_img);
      for (const std::int32_t id : ids) {
        if (id == pad_id) { for (int j = 0; j < n_img; ++j) ex.push_back(pad_id); }
        else { ex.push_back(id); }
      }
      ids.swap(ex);
    }
    if ((int)ids.size() <= drop) { return {}; }
    const int n = (int)ids.size();
    const int n_real = n - drop;
    SharedBuffer x = mc->make_shared_buffer((std::size_t)n * EH * 2);
    if (x.empty()) { return {}; }
    {
      const auto* tbl = static_cast<const std::uint8_t*>(_embed.contents());
      auto* xb = static_cast<std::uint8_t*>(x.contents());
      const std::size_t vocab = _embed.byte_size() / ((std::size_t)EH * 2);
      for (int i = 0; i < n; ++i) {
        const std::uint32_t id = (std::uint32_t)ids[(std::size_t)i];
        if (id >= vocab) { return {}; }
        std::memcpy(xb + (std::size_t)i * EH * 2, tbl + (std::size_t)id * EH * 2,
                    (std::size_t)EH * 2);
      }
    }
    // Splice the tower rows over the image_pad embeddings (f16 tower -> bf16
    // encoder input, as in the krea2 path below).
    int first_pad = -1;
    if (grounded) {
      const auto* vt = static_cast<const _Float16*>(vtok.contents());
      auto* xh = static_cast<std::uint16_t*>(x.contents());
      int j = 0;
      for (int i = 0; i < n && j < n_img; ++i) {
        if (ids[(std::size_t)i] == pad_id) {
          if (first_pad < 0) { first_pad = i; }
          for (int h = 0; h < EH; ++h) {
            xh[(std::size_t)i * EH + h] =
                f32_to_bf16_((float)vt[(std::size_t)j * EH + h]);
          }
          ++j;
        }
      }
    }
    genai::MetalQwenModel::DeepstackInject ds;
    const bool use_ds = grounded && first_pad >= 0 && !_ds_feats.empty();
    if (use_ds) {
      for (int i = 0; i < (int)_ds_feats.size(); ++i) {
        ds.feats.push_back(&_ds_feats[(std::size_t)i]);
        ds.layers.push_back(i);
      }
      ds.row0 = first_pad;
      ds.rows = n_img;
    }
    // NOTE: SEQUENTIAL positions, NOT the 2-D mROPE grid krea2 uses. Mage-Flow
    // overrides position_ids with a per-sequence `torch.arange(length)`
    // (text_encoder.py TextEncoder.forward), which the patched Qwen3-VL text
    // model expands to three IDENTICAL axes -- so the image rows carry plain
    // sequential positions even though the tower is multimodal.
    genai::ContextManager* cm = _encoder->context_manager();
    const genai::ContextId cid = cm->acquire_root();
    SharedBuffer taps;
    {   // LLM-lane perf event: DiT text-conditioning encoder prefill.
      PerfAuxScope _perf(session(), kPerfLaneLLM, kGvidLlmDitText,
                         kPerfLlmDitTextBegin, (std::uint64_t)n);
      taps = _encoder->forward_embeddings_taps(
          cid, x, n, std::vector<int>{NL - 1}, /*key_valid_len=*/0,
          use_ds ? &ds : nullptr);
    }
    cm->release(cid);
    if (taps.empty()) { return {}; }
    // Final RMSNorm on the host (the tap is the last layer's pre-norm output).
    std::vector<float> fnorm(EH, 1.0f);
    {
      auto wts = genai::MetalLlamaWeights::open_model(_enc_dir);
      if (wts.has_value()) {
        SharedBuffer nw = wts->load("model.language_model.norm.weight", mc);
        if (!nw.empty()) {
          const auto* p = static_cast<const std::uint16_t*>(nw.contents());
          for (int h = 0; h < EH; ++h) { fnorm[h] = bf16_to_f32_(p[h]); }
        }
      }
    }
    SharedBuffer txt = mc->make_shared_buffer((std::size_t)n_real * EH * 2);
    if (txt.empty()) { return {}; }
    const auto* tp = static_cast<const std::uint16_t*>(taps.contents());
    auto* op = static_cast<std::uint16_t*>(txt.contents());
    for (int p = 0; p < n_real; ++p) {
      const auto* row = tp + (std::size_t)(p + drop) * EH;
      double ss = 0.0;
      for (int h = 0; h < EH; ++h) {
        const double v = bf16_to_f32_(row[h]); ss += v * v;
      }
      const double inv = 1.0 / std::sqrt(ss / (double)EH + 1e-6);
      for (int h = 0; h < EH; ++h) {
        op[(std::size_t)p * EH + h] =
            f32_to_bf16_((float)(bf16_to_f32_(row[h]) * inv * fnorm[h]));
      }
    }
    n_real_out = n_real;
    session()->log_debug(fmt("DiffusionConditionerStage('{}'): [{}] mage-flow "
                             "-> [{}, {}] bf16{}", this->id(), which, n_real,
                             EH, grounded ? ", image-grounded (edit)" : ""));
    return txt;
  }

  // krea2: 12-tap -> [n_real, 12, EH] (the DiT fuses via forward_text). With a
  // reference image (edit / identity-edit LoRA) the instruction is encoded
  // GROUNDED on the source: insert <|vision_start|><|image_pad|><|vision_end|>
  // in the user turn and overwrite the pad positions with the Qwen3-VL vision
  // tokens (training-matched). kPrefix -- hence kDropPrefix -- is unchanged: the
  // vision block sits after the (identical) system prefix, so it survives the
  // prefix drop and joins the retained conditioning.
  const bool img_aware = (n_img > 0) && !vtok.empty();
  const std::int32_t pad_id =
      img_aware ? _tokenizer->special_token_id("<|image_pad|>") : -1;
  const std::string tmpl =
      std::string(kPrefix) +
      (img_aware && pad_id >= 0
           ? "<|vision_start|><|image_pad|><|vision_end|>" : "") +
      text + std::string(kSuffix);
  std::vector<std::int32_t> ids = encode_with_specials_(*_tokenizer, tmpl);
  if (img_aware && pad_id >= 0) {   // expand the single pad id to n_img copies
    std::vector<std::int32_t> ex; ex.reserve(ids.size() + (std::size_t)n_img);
    for (const std::int32_t id : ids) {
      if (id == pad_id) { for (int j = 0; j < n_img; ++j) ex.push_back(pad_id); }
      else { ex.push_back(id); }
    }
    ids.swap(ex);
  }
  if ((int)ids.size() <= kDropPrefix) { return {}; }
  const int n = (int)ids.size();
  const int n_real = n - kDropPrefix;
  const int NL = 12;
  SharedBuffer x = mc->make_shared_buffer((std::size_t)n * EH * 2);
  if (x.empty()) { return {}; }
  {
    const auto* tbl = static_cast<const std::uint8_t*>(_embed.contents());
    auto* xb = static_cast<std::uint8_t*>(x.contents());
    const std::size_t vocab = _embed.byte_size() / ((std::size_t)EH * 2);
    for (int i = 0; i < n; ++i) {
      const std::uint32_t id = (std::uint32_t)ids[(std::size_t)i];
      if (id >= vocab) { return {}; }
      std::memcpy(xb + (std::size_t)i * EH * 2, tbl + (std::size_t)id * EH * 2,
                  (std::size_t)EH * 2);
    }
  }
  // Splice the vision-tower rows over the image_pad embeddings. The Qwen3-VL
  // tower emits f16; the encoder input is bf16, so convert per element (the QIE
  // tower already emits bf16 and raw-copies -- this is the krea2 difference).
  int first_pad = -1;   // row0 of the contiguous image block (for deepstack)
  if (img_aware && pad_id >= 0) {
    const auto* vt = static_cast<const _Float16*>(vtok.contents());
    auto* xh = static_cast<std::uint16_t*>(x.contents());
    int j = 0;
    for (int i = 0; i < n && j < n_img; ++i) {
      if (ids[(std::size_t)i] == pad_id) {
        if (first_pad < 0) { first_pad = i; }
        for (int h = 0; h < EH; ++h) {
          xh[(std::size_t)i * EH + h] =
              f32_to_bf16_((float)vt[(std::size_t)j * EH + h]);
        }
        ++j;
      }
    }
  }
  std::vector<int> taps_l; for (int k : kSelectLayers) { taps_l.push_back(k - 1); }
  // Qwen3-VL deepstack: after LM layers 0..N-1, ADD the vision deepstack
  // features to the image rows (contiguous from first_pad). Only when grounded
  // and the tower produced deepstack features.
  genai::MetalQwenModel::DeepstackInject ds;
  const bool use_ds = img_aware && first_pad >= 0 && !_ds_feats.empty();
  if (use_ds) {
    for (int i = 0; i < (int)_ds_feats.size(); ++i) {
      ds.feats.push_back(&_ds_feats[(std::size_t)i]);
      ds.layers.push_back(i);
    }
    ds.row0 = first_pad;
    ds.rows = n_img;
  }
  // Qwen3-VL image tokens require 3-axis mROPE: text tokens sequential, the
  // n_img image tokens (contiguous from first_pad, merger row-major order) get
  // 2-D grid positions (t=base, h=base+row, w=base+col); the next text token
  // resumes at base + max(mh, mw). Without this the image rows carry sequential
  // positions and the grounded conditioning collapses (image tokens diverge
  // from the reference). Sequential path kept for the text-only case.
  const bool use_mrope = img_aware && first_pad >= 0
                         && _img_mh > 0 && _img_mw > 0;
  std::vector<std::int32_t> pos;
  if (use_mrope) {
    pos.assign((std::size_t)3 * n, 0);
    int cur = 0;
    for (int i = 0; i < n;) {
      if (i == first_pad) {
        const int base = cur;
        for (int j = 0; j < n_img && i < n; ++j, ++i) {
          pos[(std::size_t)i] = base;                          // T
          pos[(std::size_t)n + i] = base + j / _img_mw;        // H
          pos[(std::size_t)2 * n + i] = base + j % _img_mw;    // W
        }
        cur = base + std::max(_img_mh, _img_mw);
      } else {
        pos[(std::size_t)i] = cur;
        pos[(std::size_t)n + i] = cur;
        pos[(std::size_t)2 * n + i] = cur;
        ++cur; ++i;
      }
    }
  }
  genai::ContextManager* cm = _encoder->context_manager();
  const genai::ContextId cid = cm->acquire_root();
  SharedBuffer taps;
  {   // LLM-lane perf event: DiT text-conditioning encoder prefill.
    PerfAuxScope _perf(session(), kPerfLaneLLM, kGvidLlmDitText,
                       kPerfLlmDitTextBegin, (std::uint64_t)n);
    taps = use_mrope
               ? _encoder->forward_embeddings_taps_mrope(
                     cid, x, n, pos, taps_l, /*key_valid_len=*/0,
                     use_ds ? &ds : nullptr)
               : _encoder->forward_embeddings_taps(
                     cid, x, n, taps_l, /*key_valid_len=*/0,
                     use_ds ? &ds : nullptr);
  }
  cm->release(cid);
  if (taps.empty()) { return {}; }
  // The Krea-2 DiT's forward_text consumes f16 ([n_real, 12, EH]); convert the
  // bf16 taps to _Float16 here (byte-identical to the old inline encode path).
  SharedBuffer ehs = mc->make_shared_buffer((std::size_t)n_real * NL * EH * 2);
  const auto* tp = static_cast<const std::uint16_t*>(taps.contents());
  auto* ep = static_cast<_Float16*>(ehs.contents());
  for (int p = 0; p < n_real; ++p) {
    for (int j = 0; j < NL; ++j) {
      const std::size_t src = ((std::size_t)j * n + (kDropPrefix + p)) * EH;
      const std::size_t dst = ((std::size_t)p * NL + j) * EH;
      for (int h = 0; h < EH; ++h) {
        ep[dst + h] = (_Float16)bf16_to_f32_(tp[src + h]);
      }
    }
  }
  n_real_out = n_real;
  session()->log_debug(fmt("DiffusionConditionerStage('{}'): [{}] krea2 -> "
                           "[{}, 12, {}] f16{}", this->id(), which, n_real, EH,
                           img_aware ? ", image-grounded" : ""));
  return ehs;
}

// Wrap a 2-byte/elt metal buffer [rows, ...] as a TensorBeatPayload of `shape`.
// `dt` records the element type the paired DiT consumes -- F16 for krea2/flux2
// (forward_text/forward_dit read _Float16), Bf16 for qwen-image-edit.
static std::unique_ptr<TensorBeatPayload>
to_beat_(const SharedBuffer& buf, std::vector<std::int64_t> shape,
         TensorBeat::DType dt)
{
  auto out = std::make_unique<TensorBeatPayload>();
  out->dtype = dt;
  out->shape = std::move(shape);
  std::size_t n = 1; for (auto d : out->shape) { n *= (std::size_t)d; }
  out->resize_contiguous(n);
  std::memcpy(out->as_u8(), buf.contents(), n * 2);
  return out;
}

Job
DiffusionConditionerStage::process(RuntimeContext& ctx)
{
  auto* mc = session()->metal_compute();
  if (mc == nullptr) { co_return; }

  // Latch the shared model (iport2) once -- a model-select source overrides the
  // hf_dir config -- then lazily load the encoder before we need it.
  if (!_model_latched && ctx.num_iports() > kModelPort &&
      ctx.iport_connected(kModelPort)) {
    auto mb = co_await ctx.read(kModelPort);
    _model_latched = true;
    if (const auto* mfd =
            mb ? dynamic_cast<const FlexDataPayload*>(mb.get()) : nullptr) {
      if (apply_model_select_beat(mfd->data, _hf_dir, _models_db)) {
        ensure_loaded_();
      }
    }
  }
  if (!_encoder) { co_return; }   // no model loaded -> inert

  // Latch the negative prompt (iport1) + reference image (iport3) once.
  if (!_negative_latched && (int)ctx.num_iports() > 1 &&
      ctx.iport_connected(1)) {
    auto nb = co_await ctx.read(1);
    if (const auto* fp = nb ? dynamic_cast<const FlexDataPayload*>(nb.get())
                            : nullptr) {
      _negative_prompt = flex_text_(fp->data);
    }
    _negative_latched = true;
  }
  if (_ref_rgb.empty() && (int)ctx.num_iports() > 3 && ctx.iport_connected(3)) {
    auto rb = co_await ctx.read(3);
    const auto* tb = rb ? dynamic_cast<const TensorBeatPayload*>(rb.get())
                        : nullptr;
    if (tb != nullptr && tb->dtype == TensorBeat::DType::U8 &&
        tb->shape.size() == 3 && tb->shape[0] == 3) {
      const auto bytes = tb->materialize_contiguous();
      _ref_rgb.assign(bytes.begin(), bytes.end());
      _ref_rgb_h = (int)tb->shape[1];
      _ref_rgb_w = (int)tb->shape[2];
    }
  }

  // Read the prompt (iport0). A null beat means the upstream source is
  // exhausted: signal done so the runtime tears the stage down instead of
  // re-invoking process() in a tight EOF-read loop.
  auto pb = co_await ctx.read(0);
  if (!pb) { ctx.signal_done(); co_return; }
  const auto* fp = dynamic_cast<const FlexDataPayload*>(pb.get());
  if (fp == nullptr) { co_return; }
  const std::string prompt = flex_text_(fp->data);
  if (prompt.empty()) { co_return; }

  // Vision tokens (QIE + ref image) -- shared by the positive + negative encode.
  int n_img = 0;
  SharedBuffer vtok = vision_tokens_(mc, n_img);

  // Shape helper for the family conditioning tensor.
  auto shape_for = [&](int rows) -> std::vector<std::int64_t> {
    if (_family == "krea2") { return {rows, 12, _enc_hidden}; }
    if (_family == "flux2") { return {rows, 3 * _enc_hidden}; }
    return {rows, _enc_hidden};   // qwen-image-edit / mage-flow
  };
  // Element type the paired DiT consumes: krea2/flux2 -> f16, the single-tap
  // families (qwen-image-edit / mage-flow) -> bf16.
  const TensorBeat::DType cdt = single_tap_(_family) ? TensorBeat::DType::Bf16
                                                     : TensorBeat::DType::F16;

  int n_real = 0;
  SharedBuffer cond = encode_(prompt, "prompt", n_real, vtok, n_img);
  if (cond.empty()) {
    session()->warn(fmt("DiffusionConditionerStage('{}'): prompt encode failed",
                        this->id()));
    co_return;
  }
  // Emit the negative conditioning (oport1) BEFORE the positive (oport0): the
  // text-to-image stage blocks on iport0, so enqueuing the negative first
  // guarantees its paired beat is already in iport1's FIFO when the positive
  // arrives (a race-free backlog poll on the consumer side).
  // Emit the negative conditioning (oport1) for the DiT's CFG. Normally only
  // when a non-empty negative prompt is wired; the Krea-2 edit deletion recipe
  // instead uses a GROUNDED encode of an EMPTY instruction as the negative
  // (README: "negative (leave empty) -- only used at CFG > 1"), enabled by
  // `grounded_negative` on an image-aware run. encode_("") still splices the
  // vision tokens, so the empty negative stays image-grounded.
  const bool emit_neg =
      !_negative_prompt.empty()
      || (_grounded_negative && n_img > 0 && !vtok.empty());
  if (emit_neg) {
    int n_neg = 0;
    SharedBuffer nc = encode_(_negative_prompt, "negative", n_neg, vtok, n_img);
    if (!nc.empty()) {
      co_await ctx.write(1, to_beat_(nc, shape_for(n_neg), cdt));
    }
  }
  co_await ctx.write(0, to_beat_(cond, shape_for(n_real), cdt));
  ++_emitted;
}

#endif  // VPIPE_BUILD_APPLE_SILICON

VPIPE_REGISTER_STAGE(DiffusionConditionerStage)
VPIPE_REGISTER_SPEC(DiffusionConditionerStage, kSpec)

}  // namespace vpipe

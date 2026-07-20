#include "stages/diffusion-conditioner-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "stages/model-registry.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/context-manager.h"
#include "generative-models/llama3/metal-llama-weights.h"

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

const ConfigKey kAttrs[] = {
  {.key = "hf_dir", .type = ConfigType::String, .required = true,
   .doc = "model dir (text_encoder/, transformer/, tokenizer/); the "
          "transformer's _class_name selects the family + encoder",
   .suggest_db = "models", .suggest_db_type = "krea2,flux2,qwen-image-edit"},
  {.key = "models_db", .type = ConfigType::String, .required = false,
   .doc = "model registry db for resolve_model_dir (default \"models\")"},
};
const PortSpec kIports[] = {
  {.name = "prompt", .doc = "prompt text (FlexData string or {text: ...})",
   .type = &typeid(FlexDataPayload), .tags = "text", .clock_group = 0},
  {.name = "negative", .doc = "OPTIONAL negative prompt (FlexData) for the DiT's "
                              "classifier-free guidance; its conditioning is "
                              "emitted on oport1",
   .type = &typeid(FlexDataPayload), .tags = "text", .clock_group = 0},
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
          "bf16 [n_real,3584] image-aware)",
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

// Special-token-aware encode: split at the markers, BPE each text run, splice
// the markers' ids (matches the HF fast tokenizer). Includes the Qwen2.5-VL
// vision markers so the QIE image-aware template isolates them.
std::vector<std::int32_t>
encode_with_specials_(const genai::Tokenizer& tok, const std::string& text)
{
  static const char* kMarkers[] = {"<|im_start|>", "<|im_end|>",
                                   "<|vision_start|>", "<|vision_end|>",
                                   "<|image_pad|>"};
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
  _hf_dir    = attr_str("hf_dir");
  _models_db = attr_str("models_db");
  if (_models_db.empty()) { _models_db = "models"; }
  if (_hf_dir.empty()) {
    fail_config(fmt("DiffusionConditionerStage('{}'): config.hf_dir is required",
                    this->id()));
  }
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
          : "language_model.embed_tokens.weight";
  _embed = wts->load(emb_name, mc);
  return !_embed.empty();
}

Job
DiffusionConditionerStage::initialize(RuntimeContext& ctx)
{
  (void)ctx;
  auto* mc = session()->metal_compute();
  if (mc == nullptr) {
    fail_config(fmt("DiffusionConditionerStage('{}'): no metal-compute backend",
                    this->id()));
    co_return;
  }
  const std::string root = resolve_model_dir(session(), _models_db, _hf_dir);
  _enc_dir = (std::filesystem::path(root) / "text_encoder").string();
  _family = family_((std::filesystem::path(root) / "transformer").string());

  namespace fs = std::filesystem;
  std::string tok_path = (fs::path(root) / "tokenizer" / "tokenizer.json").string();
  if (!fs::exists(tok_path)) {
    tok_path = (fs::path(root) / "processor" / "tokenizer.json").string();
  }
  _tokenizer = genai::Tokenizer::from_huggingface_json(tok_path, session());
  if (!_tokenizer) {
    fail_config(fmt("DiffusionConditionerStage('{}'): tokenizer load failed: {}",
                    this->id(), tok_path));
    co_return;
  }
  if (!load_encoder_(mc)) {
    fail_config(fmt("DiffusionConditionerStage('{}'): encoder/embeds load failed",
                    this->id()));
    co_return;
  }
  session()->info(fmt(
      "DiffusionConditionerStage('{}'): family {} encoder ({}), hidden {}",
      this->id(), _family,
      _family == "flux2" ? "Qwen3 dense"
      : _family == "qwen-image-edit" ? "Qwen2.5-VL" : "Qwen3-VL", _enc_hidden));
  co_return;
}

SharedBuffer
DiffusionConditionerStage::vision_tokens_(metal_compute::MetalCompute* mc,
                                          int& n_img) const
{
  n_img = 0;
  if (_family != "qwen-image-edit" || _ref_rgb.empty()) { return {}; }
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
  SharedBuffer vt = _vision->encode_rgb(_ref_rgb.data(), _ref_rgb_h, _ref_rgb_w,
                                        384 * 384, vgh, vgw);
  if (vt.empty()) { return {}; }
  const int mm = _vision->config().merge;
  n_img = (vgh / mm) * (vgw / mm);
  session()->info(fmt(
      "DiffusionConditionerStage('{}'): image-aware conditioning -> {} vision "
      "tokens (grid {}x{})", this->id(), n_img, vgh, vgw));
  return vt;
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
    const std::string tmpl = std::string("<|im_start|>user\n") + text +
                             "<|im_end|>\n<|im_start|>assistant\n";
    const auto ids = encode_with_specials_(*_tokenizer, tmpl);
    if (ids.empty()) { return {}; }
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
    SharedBuffer taps = _encoder->forward_embeddings_taps(cid, x, n, taps_l);
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
    SharedBuffer taps =
        _encoder->forward_embeddings_taps(cid, x, n, std::vector<int>{NL - 1});
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

  // krea2: 12-tap -> [n_real, 12, EH] (the DiT fuses via forward_text).
  const std::string tmpl =
      std::string(kPrefix) + text + std::string(kSuffix);
  const auto ids = encode_with_specials_(*_tokenizer, tmpl);
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
  std::vector<int> taps_l; for (int k : kSelectLayers) { taps_l.push_back(k - 1); }
  genai::ContextManager* cm = _encoder->context_manager();
  const genai::ContextId cid = cm->acquire_root();
  SharedBuffer taps = _encoder->forward_embeddings_taps(cid, x, n, taps_l);
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
                           "[{}, 12, {}] f16", this->id(), which, n_real, EH));
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
  if (mc == nullptr || !_encoder) { co_return; }

  // Latch the negative prompt (iport1) + reference image (iport2) once.
  if (!_negative_latched && (int)ctx.num_iports() > 1 &&
      ctx.iport_connected(1)) {
    auto nb = co_await ctx.read(1);
    if (const auto* fp = nb ? dynamic_cast<const FlexDataPayload*>(nb.get())
                            : nullptr) {
      _negative_prompt = flex_text_(fp->data);
    }
    _negative_latched = true;
  }
  if (_ref_rgb.empty() && (int)ctx.num_iports() > 2 && ctx.iport_connected(2)) {
    auto rb = co_await ctx.read(2);
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
    return {rows, _enc_hidden};   // qwen-image-edit
  };
  // Element type the paired DiT consumes: krea2/flux2 -> f16, QIE -> bf16.
  const TensorBeat::DType cdt = (_family == "qwen-image-edit")
                                    ? TensorBeat::DType::Bf16
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
  if (!_negative_prompt.empty()) {
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

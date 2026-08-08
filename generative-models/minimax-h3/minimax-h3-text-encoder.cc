#include "generative-models/minimax-h3/minimax-h3-text-encoder.h"

#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "generative-models/context-manager.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/shared/comfy-checkpoint.h"
#include "generative-models/tokenizer.h"
#include "generative-models/weight-set.h"
#include "interfaces/session-context-intf.h"

#include <cstring>
#include <filesystem>
#include <fstream>

namespace vpipe {
namespace genai {

using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

// The Comfy-Org single-file encoder's `__metadata__` key. Unlike the DiT
// and the two VAEs, what it holds is NOT a config -- only
// `{"num_hidden_layers": 50, "output": "unnormalized_hidden_after_layer_50"}`,
// i.e. the tap and nothing else. Every architecture field has to come
// from the TENSOR SHAPES instead (see comfy_config_).
constexpr const char* kComfyKey = "minimax_h3_te";

// Size a Config from the Comfy-Org encoder's TENSOR SHAPES.
//
// Every other Comfy-Org component here embeds a real config; this one
// embeds only the tap, so the architecture has to be measured. That is
// sound for the fields below -- each is a matrix dimension, so reading
// it wrong would not load -- but NOT for the two constants at the
// bottom, which no shape encodes. They keep the released Qwen3-VL-32B
// values, and a checkpoint that changed either would load and produce
// subtly wrong conditioning rather than fail, so they are named here
// instead of being silently inherited from a default.
//
// The file is also PRUNED to the tap: 50 layers, not 64. So the layer
// count read here is both `total_layers` and `tap` -- there is nothing
// past it to tap at.
bool
comfy_config_(const std::string& file, MiniMaxH3TextEncoder::Config& out,
              std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  auto wts = MetalLlamaWeights::open(file);
  if (!wts) { return fail("cannot open " + file); }
  auto dim = [&](const char* nm, int axis) -> int {
    const auto* i = wts->info(nm);
    if (i == nullptr || (int)i->shape.size() <= axis) { return 0; }
    return (int)i->shape[axis];
  };
  // The weight prefix is "model." here, not the diffusers
  // "model.language_model." -- Comfy-Org's conversion drops the
  // language_model segment. MetalQwenModel re-derives this from the
  // tensor names on its own; it is set explicitly so the probes below
  // and the embedding bind in load() agree with it.
  MetalQwenModel::Config& c = out.lm;
  c.weight_prefix = "model.";
  c.model_seg     = "";

  int layers = 0;
  while (dim(("model.layers." + std::to_string(layers) +
              ".input_layernorm.weight").c_str(), 0) > 0) {
    ++layers;
  }
  if (layers <= 0) { return fail(file + ": no model.layers.N.* tensors"); }
  out.total_layers = layers;
  out.tap          = layers;

  c.vocab     = dim("model.embed_tokens.weight", 0);
  c.hidden    = dim("model.embed_tokens.weight", 1);
  c.head_dim  = dim("model.layers.0.self_attn.q_norm.weight", 0);
  const int q  = dim("model.layers.0.self_attn.q_proj.weight", 0);
  const int kv = dim("model.layers.0.self_attn.k_proj.weight", 0);
  c.ffn_inner = dim("model.layers.0.mlp.gate_proj.weight", 0);
  if (c.hidden <= 0 || c.head_dim <= 0 || q <= 0 || kv <= 0 ||
      c.ffn_inner <= 0 || c.vocab <= 0) {
    return fail(file + ": cannot measure the encoder geometry from its "
                       "tensor shapes");
  }
  c.n_heads    = q / c.head_dim;
  c.n_kv_heads = kv / c.head_dim;
  c.rotary_dim = c.head_dim;
  out.text_dim = c.hidden;
  if (c.n_heads <= 0 || c.n_kv_heads <= 0 ||
      c.n_heads * c.head_dim != q || c.n_kv_heads * c.head_dim != kv) {
    return fail(file + ": q/k projection widths are not a whole number of "
                       "head_dim heads");
  }
  // NOT derivable from any shape. These are the released Qwen3-VL-32B
  // values, which is what this checkpoint is a repack of.
  c.rope_theta = 5.0e6f;
  c.rms_eps    = 1e-6f;
  return true;
}

}  // namespace

std::string
MiniMaxH3TextEncoder::resolve_encoder_dir(const std::string& path)
{
  namespace fs = std::filesystem;
  fs::path p(path);
  {
    const std::string f = comfy::resolve_component(path, "text_encoders",
                                                   kComfyKey, {"qwen3vl"});
    if (!f.empty()) { return f; }
  }
  if (!fs::is_directory(p)) { return path; }
  if (fs::exists(p / "config.json") &&
      (fs::exists(p / "model.safetensors.index.json") ||
       fs::exists(p / "model.safetensors"))) {
    return p.string();                                  // already text_encoder
  }
  if (fs::exists(p / "text_encoder" / "config.json")) {
    return (p / "text_encoder").string();               // a partition root
  }
  if (fs::exists(p / "FL2VA" / "text_encoder" / "config.json")) {
    return (p / "FL2VA" / "text_encoder").string();
  }
  return path;
}

bool
MiniMaxH3TextEncoder::config_from_json(const std::string& enc_dir, Config& out,
                                       std::string* err)
{
  namespace fs = std::filesystem;
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  const fs::path dir(resolve_encoder_dir(enc_dir));
  // The Comfy-Org single file: no config.json, and its `__metadata__`
  // carries only the tap -- so the geometry is measured off the tensor
  // shapes instead. See comfy_config_.
  if (comfy::is_component(dir.string(), kComfyKey)) {
    MiniMaxH3TextEncoder::Config c;
    if (!comfy_config_(dir.string(), c, err)) { return false; }
    // The invariant fields, shared with the diffusers path below: a
    // stock Qwen3-VL text stack, backbone only, read for an
    // intermediate hidden state rather than for tokens.
    MetalQwenModel::Config& lm = c.lm;
    lm.full_attn_interval = 1;
    lm.dense = true;
    lm.use_bf16 = true;
    lm.qk_norm = true;
    lm.attention_bias = false;
    lm.attn_output_gate = false;
    lm.zero_centered_norm = false;
    lm.tie_embeddings = false;
    lm.backbone_only = true;
    lm.max_seq = 4096;
    lm.page_tokens = 256;
    lm.n_layers = c.tap;
    out = std::move(c);
    return true;
  }
  std::ifstream f(dir / "config.json");
  if (!f) { return fail("cannot open " + (dir / "config.json").string()); }
  FlexData cfg;
  try {
    cfg = FlexData::from_json(f);
  } catch (...) {
    return fail("cannot parse " + (dir / "config.json").string());
  }
  if (!cfg.is_object()) { return fail("encoder config is not a JSON object"); }
  auto root = cfg.as_object();

  MetalQwenModel::Config& c = out.lm;
  // A stock Qwen3-VL text stack: dense SwiGLU, GQA with per-head q/k
  // RMS, no attention bias and no output gate.
  c.full_attn_interval = 1;
  c.dense = true;
  c.use_bf16 = true;
  c.qk_norm = true;
  c.attention_bias = false;
  c.attn_output_gate = false;
  c.zero_centered_norm = false;
  c.tie_embeddings = false;
  // Nothing past the tap runs, so there is no head to load. This is also
  // what lets the layer count below be the TAP rather than the
  // checkpoint's depth.
  c.backbone_only = true;
  c.weight_prefix = "model.language_model.";
  c.model_seg = "";
  c.max_seq = 4096;
  c.page_tokens = 256;
  c.rms_eps = 1e-6f;

  if (!root.contains("text_config")) {
    return fail("no text_config in the encoder config: this does not look "
                "like a Qwen3-VL checkpoint");
  }
  FlexData tcd = root.at("text_config");
  if (!tcd.is_object()) { return fail("text_config is not an object"); }
  auto o = tcd.as_object();
  auto geti = [&](const char* k, int d) {
    return o.contains(k) ? (int)o.at(k).as_int(d) : d;
  };
  auto getf = [&](const char* k, float d) {
    return o.contains(k) ? (float)o.at(k).as_real(d) : d;
  };
  out.total_layers = geti("num_hidden_layers", 64);
  c.hidden      = geti("hidden_size", 5120);
  c.n_heads     = geti("num_attention_heads", 64);
  c.n_kv_heads  = geti("num_key_value_heads", 8);
  c.head_dim    = geti("head_dim", c.n_heads > 0 ? c.hidden / c.n_heads : 128);
  c.rotary_dim  = c.head_dim;
  c.ffn_inner   = geti("intermediate_size", 25600);
  c.vocab       = geti("vocab_size", 151936);
  c.rope_theta  = getf("rope_theta", 5.0e6f);
  c.rms_eps     = getf("rms_norm_eps", c.rms_eps);
  out.text_dim  = c.hidden;
  if (root.contains("tie_word_embeddings")) {
    c.tie_embeddings = root.at("tie_word_embeddings").as_bool(false);
  }
  // A quantized encoder (model-quantize target=text_encoder). The loader
  // auto-detects quantized-vs-dense weights, but it needs the bit width
  // to pick the w4g64 vs w8g64 kernel -- left at the default a 8-bit
  // checkpoint decodes with the 4-bit kernel and returns garbage rather
  // than failing, so read it rather than assuming.
  if (root.contains("quantization")) {
    FlexData q = root.at("quantization");
    if (q.is_object()) {
      auto qo = q.as_object();
      const int b = qo.contains("bits") ? (int)qo.at("bits").as_int(0) : 0;
      if (b == 4 || b == 8) { c.quant_bits = b; out.quantized = true; }
    }
  }
  if (out.tap <= 0 || out.tap > out.total_layers) {
    return fail("tap layer " + std::to_string(out.tap) + " is outside the "
                "checkpoint's " + std::to_string(out.total_layers) +
                " layers");
  }
  // Load exactly as deep as the tap. Layers past it never run.
  c.n_layers = out.tap;
  return true;
}

MiniMaxH3TextEncoder::~MiniMaxH3TextEncoder() = default;

std::unique_ptr<MiniMaxH3TextEncoder>
MiniMaxH3TextEncoder::load(const std::string& enc_dir, MetalCompute* mc,
                           SessionContextIntf* session, const Config& cfg)
{
  if (mc == nullptr) { return nullptr; }
  const std::string dir = resolve_encoder_dir(enc_dir);
  auto m = std::unique_ptr<MiniMaxH3TextEncoder>(new MiniMaxH3TextEncoder());
  m->_mc = mc;
  m->_cfg = cfg;
  if (m->_cfg.tap <= 0 || m->_cfg.tap > m->_cfg.total_layers) {
    return nullptr;
  }
  m->_cfg.lm.n_layers = m->_cfg.tap;

  m->_ws = open_weight_set(dir, session);
  if (!m->_ws) { return nullptr; }
  m->_lm = MetalQwenModel::load(m->_ws, mc, m->_cfg.lm);
  if (!m->_lm) { return nullptr; }

  // The embedding table is not part of a backbone-only load, so bind it
  // here. It comes from the same set, so a peer naming this checkpoint
  // shares the bytes rather than mapping a second copy. Two spellings:
  // the diffusers checkpoint nests the text stack under
  // `model.language_model.`, Comfy-Org's conversion drops that segment.
  for (const char* nm : {"model.language_model.embed_tokens.weight",
                         "model.embed_tokens.weight"}) {
    m->_embed = m->_ws->tensor(nm, mc, WeightSet::Residency::Copied);
    if (!m->_embed.empty()) { break; }
  }
  if (m->_embed.empty()) { return nullptr; }

  // The tokenizer. A diffusers text_encoder/ ships its own; the
  // Comfy-Org repack ships NONE -- it is weights only -- so the search
  // also covers the repo root and a tokenizer/ beside it, which is
  // where a user pairing the repack with MiniMaxAI's tokenizer would
  // put one. Missing, this fails LOUDLY rather than returning a nullptr
  // the caller has to guess the meaning of: encode(prompt) genuinely
  // cannot work, and "no tokenizer" is a fixable setup problem where a
  // bare load failure reads as an unsupported checkpoint.
  namespace fs = std::filesystem;
  const fs::path d(dir);
  const fs::path base = fs::is_directory(d) ? d : d.parent_path();
  const fs::path root = base.parent_path();
  for (const fs::path& p : {base / "tokenizer.json",
                            base / "tokenizer" / "tokenizer.json",
                            root / "tokenizer.json",
                            root / "tokenizer" / "tokenizer.json"}) {
    if (fs::exists(p)) {
      m->_tok = Tokenizer::from_huggingface_json(p.string(), session);
      if (m->_tok) { break; }
    }
  }
  if (!m->_tok) {
    if (session != nullptr) {
      session->warn(fmt(
          "MiniMaxH3TextEncoder: no tokenizer.json for '{}' -- looked in "
          "'{}' and '{}' (and their tokenizer/ subdirs). The Comfy-Org "
          "repack ships weights only; copy MiniMaxAI/MiniMax-H3's "
          "FL2VA/text_encoder/tokenizer.json next to it",
          dir, base.string(), root.string()));
    }
    return nullptr;
  }

  if (session != nullptr) {
    session->log_debug(
        fmt("MiniMaxH3TextEncoder: Qwen3-VL hidden {} x {} heads ({} kv), "
            "tapped at layer {} of {} -- {} layers loaded",
            m->_cfg.lm.hidden, m->_cfg.lm.n_heads, m->_cfg.lm.n_kv_heads,
            m->_cfg.tap, m->_cfg.total_layers, m->_cfg.lm.n_layers));
  }
  return m;
}

std::vector<std::int32_t>
MiniMaxH3TextEncoder::tokenize(std::string_view prompt) const
{
  if (!_tok) { return {}; }
  // Tokenizer::encode adds no BOS/EOS of its own, which IS the
  // reference's `add_special_tokens=False`.
  return _tok->encode(prompt);
}

SharedBuffer
MiniMaxH3TextEncoder::encode_ids(const std::vector<std::int32_t>& ids,
                                 std::string* err)
{
  auto fail = [&](std::string m) -> SharedBuffer {
    if (err != nullptr) { *err = std::move(m); }
    return {};
  };
  if (!_lm || _embed.empty()) { return fail("encoder is not loaded"); }
  const int n = (int)ids.size();
  if (n <= 0) { return fail("empty prompt"); }
  if (n > _cfg.lm.max_seq) {
    return fail("prompt is " + std::to_string(n) + " tokens, past the " +
                std::to_string(_cfg.lm.max_seq) + "-token pool");
  }
  const int H = _cfg.lm.hidden;
  const std::size_t vocab = _embed.byte_size() / ((std::size_t)H * 2);

  SharedBuffer x = _mc->make_shared_buffer((std::size_t)n * H * 2);
  if (x.empty()) { return fail("embedding allocation failed"); }
  {
    const auto* tbl = static_cast<const std::uint8_t*>(_embed.contents());
    auto* xb = static_cast<std::uint8_t*>(x.contents());
    for (int i = 0; i < n; ++i) {
      const std::int32_t id = ids[(std::size_t)i];
      if (id < 0 || (std::size_t)id >= vocab) {
        return fail("token id " + std::to_string(id) + " is outside the "
                    "vocabulary");
      }
      std::memcpy(xb + (std::size_t)i * H * 2,
                  tbl + (std::size_t)id * H * 2, (std::size_t)H * 2);
    }
  }

  // The tap is 0-INDEXED over layers; HF's output_hidden_states[k] is
  // the residual after layer k-1. So conditioning on hidden_states[50]
  // means tapping after 0-indexed layer 49.
  const std::vector<int> taps{_cfg.tap - 1};
  ContextManager* cm = _lm->context_manager();
  if (cm == nullptr) { return fail("encoder has no context manager"); }
  const ContextId cid = cm->acquire_root();
  SharedBuffer h = _lm->forward_embeddings_taps(cid, x, n, taps);
  cm->release(cid);
  if (h.empty()) { return fail("encoder forward failed"); }
  return h;
}

SharedBuffer
MiniMaxH3TextEncoder::encode(std::string_view prompt, int* n_tokens,
                             std::string* err)
{
  const std::vector<std::int32_t> ids = tokenize(prompt);
  if (ids.empty()) {
    if (err != nullptr) { *err = "prompt tokenized to nothing"; }
    return {};
  }
  SharedBuffer h = encode_ids(ids, err);
  if (!h.empty() && n_tokens != nullptr) { *n_tokens = (int)ids.size(); }
  return h;
}

}  // namespace genai
}  // namespace vpipe

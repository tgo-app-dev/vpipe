#include "generative-models/shared/gguf-convert.h"

#include "generative-models/shared/gguf-file.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>

namespace vpipe::genai {

namespace {

inline float
f16_to_f32_(std::uint16_t h)
{
  _Float16 v;
  std::memcpy(&v, &h, 2);
  return static_cast<float>(v);
}
inline std::uint16_t
f32_to_f16_(float f)
{
  _Float16 h = static_cast<_Float16>(f);
  std::uint16_t b;
  std::memcpy(&b, &h, 2);
  return b;
}

constexpr int kQ4Group = 32;   // q4_0 block size
constexpr int kEmbedGroup = 32;
constexpr int kEmbedBits = 8;

}  // namespace

std::string
find_gguf_in_dir(const std::string& dir)
{
  namespace fs = std::filesystem;
  std::error_code ec;
  if (!fs::is_directory(dir, ec)) {
    // Allow passing a `.gguf` file directly.
    if (fs::is_regular_file(dir, ec) &&
        fs::path(dir).extension() == ".gguf") {
      return dir;
    }
    return {};
  }
  std::string best;
  for (const auto& e : fs::directory_iterator(dir, ec)) {
    if (ec) { break; }
    if (!e.is_regular_file()) { continue; }
    const fs::path p = e.path();
    if (p.extension() != ".gguf") { continue; }
    const std::string fn = p.filename().string();
    // Skip the multimodal projector; pick the main model shard.
    if (fn.rfind("mmproj", 0) == 0) { continue; }
    if (best.empty() || fn < fs::path(best).filename().string()) {
      best = p.string();
    }
  }
  return best;
}

// Populate a ModelConfig from a `qwen35` (Qwen3.5 hybrid GDN) GGUF's
// metadata. Mirrors the safetensors Qwen3.5 config so MetalQwenModel::
// config_from() self-configures identically off either source. The tensor
// payloads are k-quant (q4_K/q5_K/q6_K/q8_0) -- see GgufQwen35Converter.
static bool
gguf_to_qwen35_config_(const GgufFile& g, ModelConfig* out)
{
  auto uget = [&](const char* k, std::int64_t dflt) -> std::int64_t {
    auto v = g.get_int(k);
    return v ? *v : dflt;
  };
  auto fget = [&](const char* k, double dflt) -> double {
    auto v = g.get_float(k);
    return v ? *v : dflt;
  };

  *out = ModelConfig{};
  out->architecture = "Qwen3_5ForConditionalGeneration";
  // block_count includes any bundled NextN/MTP draft blocks; the MAIN model
  // is block_count - nextn_predict_layers. Keep n_layers as the main count and
  // carry the MTP count separately (the converter emits those blocks under the
  // mtp.* head names; absent on non-MTP GGUFs -> nextn 0, n_layers unchanged).
  out->num_nextn_layers =
      static_cast<int>(uget("qwen35.nextn_predict_layers", 0));
  out->n_layers   = static_cast<int>(uget("qwen35.block_count", 0)) -
                    out->num_nextn_layers;
  out->hidden     = static_cast<int>(uget("qwen35.embedding_length", 0));
  out->ffn_inner  = static_cast<int>(uget("qwen35.feed_forward_length", 0));
  out->n_heads    = static_cast<int>(uget("qwen35.attention.head_count", 0));
  out->n_kv_heads =
      static_cast<int>(uget("qwen35.attention.head_count_kv", 0));
  out->head_dim   = static_cast<int>(uget("qwen35.attention.key_length", 0));
  out->rms_eps    = static_cast<float>(
      fget("qwen35.attention.layer_norm_rms_epsilon", 1e-6));
  out->rope_theta = static_cast<float>(fget("qwen35.rope.freq_base", 1e7));
  out->tie_word_embeddings = true;          // 2B ties lm_head to token_embd
  out->max_position_embeddings =
      static_cast<int>(uget("qwen35.context_length", 0));
  out->attn_output_gate = true;             // Qwen3.5 gated attention
  out->full_attention_interval =
      static_cast<int>(uget("qwen35.full_attention_interval", 4));

  // Partial rotary: only rope.dimension_count of head_dim dims rotate.
  const int rope_dim = static_cast<int>(uget(
      "qwen35.rope.dimension_count",
      out->head_dim > 0 ? out->head_dim / 4 : 0));
  out->partial_rotary_factor =
      out->head_dim > 0
          ? static_cast<float>(rope_dim) / static_cast<float>(out->head_dim)
          : 1.0f;

  // Gated-DeltaNet dims. ssm.state_size = linear_key_head_dim;
  // group_count = linear_num_key_heads; time_step_rank = linear_num_value
  // _heads; inner_size = value_dim -> v_head_dim = inner_size / n_v_heads.
  out->linear_conv_kernel = static_cast<int>(uget("qwen35.ssm.conv_kernel", 0));
  out->linear_num_k_heads = static_cast<int>(uget("qwen35.ssm.group_count", 0));
  out->linear_num_v_heads =
      static_cast<int>(uget("qwen35.ssm.time_step_rank", 0));
  out->linear_k_head_dim = static_cast<int>(uget("qwen35.ssm.state_size", 0));
  const int ssm_inner = static_cast<int>(uget("qwen35.ssm.inner_size", 0));
  out->linear_v_head_dim =
      out->linear_num_v_heads > 0 ? ssm_inner / out->linear_num_v_heads : 0;

  // mROPE section split (T/H/W[/0]) over the rotary pairs.
  std::vector<std::int64_t> sec =
      g.get_int_array("qwen35.rope.dimension_sections");
  for (auto v : sec) {
    if (v > 0) { out->mrope_section.push_back(static_cast<int>(v)); }
  }

  // Layer types: full-attention iff (L+1) % full_attention_interval == 0;
  // every other layer is linear-attention (GDN).
  out->is_linear_layer.assign(static_cast<std::size_t>(out->n_layers), false);
  const int interval = out->full_attention_interval > 0
                           ? out->full_attention_interval : 4;
  for (int L = 0; L < out->n_layers; ++L) {
    out->is_linear_layer[static_cast<std::size_t>(L)] =
        ((L + 1) % interval) != 0;
  }

  if (const GgufFile::Tensor* emb = g.tensor("token_embd.weight");
      emb != nullptr && emb->dims.size() == 2) {
    out->vocab_size = static_cast<int>(emb->dims[1]);
  } else {
    out->vocab_size =
        static_cast<int>(g.array_len("tokenizer.ggml.tokens"));
  }

  // Native k-quant payloads -- not a single uniform affine bit-width. The
  // metal Qwen GGUF path keys off per-tensor ggml types, not this field;
  // left at the affine default so non-k-quant code paths stay sane.
  out->quantization.bits = 4;
  out->quantization.group_size = 0;
  return out->n_layers > 0 && out->hidden > 0;
}

bool
gguf_to_model_config(const GgufFile& g, ModelConfig* out)
{
  auto arch = g.get_string("general.architecture");
  if (arch && *arch == "qwen35") {
    return gguf_to_qwen35_config_(g, out);
  }
  if (!arch || (*arch != "gemma4" && *arch != "gemma4_unified")) {
    return false;
  }
  auto uget = [&](const char* k, std::int64_t dflt) -> std::int64_t {
    auto v = g.get_int(k);
    return v ? *v : dflt;
  };
  auto fget = [&](const char* k, double dflt) -> double {
    auto v = g.get_float(k);
    return v ? *v : dflt;
  };

  *out = ModelConfig{};
  out->architecture = "Gemma4UnifiedForConditionalGeneration";
  out->n_layers  = static_cast<int>(uget("gemma4.block_count", 0));
  out->hidden    = static_cast<int>(uget("gemma4.embedding_length", 0));
  out->ffn_inner = static_cast<int>(uget("gemma4.feed_forward_length", 0));
  out->n_heads   = static_cast<int>(uget("gemma4.attention.head_count", 0));
  out->rms_eps   =
      static_cast<float>(fget("gemma4.attention.layer_norm_rms_epsilon",
                              1e-6));
  out->rope_theta = static_cast<float>(fget("gemma4.rope.freq_base", 1e6));
  out->tie_word_embeddings = true;
  out->max_position_embeddings =
      static_cast<int>(uget("gemma4.context_length", 0));

  // Per-layer K/V head counts (sliding=8, global=1 for 12B). Base
  // n_kv_heads is the max -- it sizes fallback buffers; the executor and
  // ContextManager read the per-layer vector.
  std::vector<std::int64_t> kv =
      g.get_int_array("gemma4.attention.head_count_kv");
  int max_kv = 0;
  for (auto v : kv) { max_kv = std::max(max_kv, static_cast<int>(v)); }
  if (max_kv == 0) { max_kv = out->n_heads; }
  out->n_kv_heads = max_kv;

  if (const GgufFile::Tensor* emb = g.tensor("token_embd.weight");
      emb != nullptr && emb->dims.size() == 2) {
    out->vocab_size = static_cast<int>(emb->dims[1]);
  } else {
    out->vocab_size =
        static_cast<int>(g.array_len("tokenizer.ggml.tokens"));
  }

  auto& gm = out->gemma4;
  gm.present = true;
  gm.head_dim_sliding =
      static_cast<int>(uget("gemma4.attention.key_length_swa", 256));
  gm.head_dim_full =
      static_cast<int>(uget("gemma4.attention.key_length", 512));
  out->head_dim = gm.head_dim_sliding;
  gm.num_kv_shared_layers =
      static_cast<int>(uget("gemma4.attention.shared_kv_layers", 0));
  gm.hidden_per_layer_input =
      static_cast<int>(uget("gemma4.embedding_length_per_layer_input", 0));
  gm.sliding_window =
      static_cast<int>(uget("gemma4.attention.sliding_window", 0));
  gm.final_logit_softcapping =
      static_cast<float>(fget("gemma4.final_logit_softcapping", 0.0));
  gm.rope_theta_sliding =
      static_cast<float>(fget("gemma4.rope.freq_base_swa", 1e4));
  gm.rope_theta_full = out->rope_theta;
  const int dim_full =
      static_cast<int>(uget("gemma4.rope.dimension_count",
                            gm.head_dim_full));
  gm.full_partial_rotary_factor =
      gm.head_dim_full > 0
          ? static_cast<float>(dim_full) / static_cast<float>(gm.head_dim_full)
          : 1.0f;

  // is_full_layer: sliding_window_pattern[L]==true means a SLIDING (local)
  // layer; false means a FULL (global) layer.
  std::vector<std::int64_t> pat =
      g.get_int_array("gemma4.attention.sliding_window_pattern");
  gm.is_full_layer.assign(static_cast<std::size_t>(out->n_layers), false);
  for (int L = 0; L < out->n_layers; ++L) {
    if (L < static_cast<int>(pat.size())) {
      gm.is_full_layer[static_cast<std::size_t>(L)] = (pat[L] == 0);
    }
  }
  // Per-layer K/V head count, straight from the GGUF array (sliding=8,
  // global=1 for 12B).
  gm.layer_n_kv_heads.assign(static_cast<std::size_t>(out->n_layers),
                             out->n_kv_heads);
  for (int L = 0; L < out->n_layers && L < static_cast<int>(kv.size()); ++L) {
    gm.layer_n_kv_heads[static_cast<std::size_t>(L)] =
        static_cast<int>(kv[static_cast<std::size_t>(L)]);
  }
  // The GGUF ships NO v_proj on the global (full-attention) layers -- they
  // reuse K as V (k_eq_v), same as the safetensors gemma4_unified. Their
  // K/V head count is the head_count_kv on a full layer (1 for 12B).
  gm.attention_k_eq_v = true;
  gm.num_global_kv_heads = 0;
  for (int L = 0; L < out->n_layers; ++L) {
    if (gm.is_full_layer[static_cast<std::size_t>(L)]) {
      gm.num_global_kv_heads = gm.layer_n_kv_heads[static_cast<std::size_t>(L)];
      break;
    }
  }

  out->quantization.bits = 4;
  out->quantization.group_size = kQ4Group;
  // Q4_0 is symmetric: the repacked affine bias is exactly -8*scale, so the
  // decode GEMVs read scale-only (skip the redundant bias buffer).
  out->quantization.symmetric = true;

  return out->n_layers > 0 && out->hidden > 0 && out->vocab_size > 0;
}

// ---------------------------------------------------------------------
// GgufGemma4Converter
// ---------------------------------------------------------------------
GgufGemma4Converter::GgufGemma4Converter(const GgufFile* gguf,
                                         const ModelConfig& cfg)
  : _g(gguf), _cfg(cfg)
{
  build_specs_();
}

GgufGemma4Converter::~GgufGemma4Converter() = default;

void
GgufGemma4Converter::build_specs_()
{
  auto add_q4 = [&](const std::string& gname, const std::string& hf) {
    const GgufFile::Tensor* t = _g->tensor(gname);
    if (t == nullptr || t->dims.size() != 2) { return; }
    const std::int64_t in = t->dims[0], out = t->dims[1];
    _specs.push_back({hf + ".weight", "U32", {out, in / 8},
                      static_cast<std::uint64_t>(out) * (in / 8) * 4, gname,
                      ConvertedTensorSpec::Op::kQ4Weight});
    _specs.push_back({hf + ".scales", "F16", {out, in / kQ4Group},
                      static_cast<std::uint64_t>(out) * (in / kQ4Group) * 2,
                      gname, ConvertedTensorSpec::Op::kQ4Scales});
    _specs.push_back({hf + ".biases", "F16", {out, in / kQ4Group},
                      static_cast<std::uint64_t>(out) * (in / kQ4Group) * 2,
                      gname, ConvertedTensorSpec::Op::kQ4Biases});
  };
  auto add_float = [&](const std::string& gname, const std::string& hf) {
    const GgufFile::Tensor* t = _g->tensor(gname);
    if (t == nullptr) { return; }
    _specs.push_back({hf, "F32", t->dims,
                      static_cast<std::uint64_t>(t->numel()) * 4, gname,
                      ConvertedTensorSpec::Op::kFloatPass});
  };

  const std::string P = "language_model.model.";

  // token_embd (tied lm_head) -> affine 8-bit g32.
  if (const GgufFile::Tensor* emb = _g->tensor("token_embd.weight");
      emb != nullptr && emb->dims.size() == 2) {
    const std::int64_t hidden = emb->dims[0], vocab = emb->dims[1];
    const std::int64_t wpr = hidden / (32 / kEmbedBits);   // u32 per row
    const std::int64_t gpr = hidden / kEmbedGroup;         // groups per row
    _specs.push_back({P + "embed_tokens.weight", "U32", {vocab, wpr},
                      static_cast<std::uint64_t>(vocab) * wpr * 4,
                      "token_embd.weight",
                      ConvertedTensorSpec::Op::kEmbedWeight});
    _specs.push_back({P + "embed_tokens.scales", "F16", {vocab, gpr},
                      static_cast<std::uint64_t>(vocab) * gpr * 2,
                      "token_embd.weight",
                      ConvertedTensorSpec::Op::kEmbedScales});
    _specs.push_back({P + "embed_tokens.biases", "F16", {vocab, gpr},
                      static_cast<std::uint64_t>(vocab) * gpr * 2,
                      "token_embd.weight",
                      ConvertedTensorSpec::Op::kEmbedBiases});
    // Raw Q6_K pass-through (metal-only): the lossless 6.5625-bit table,
    // ~25% smaller than the affine8 requant above. The metal model reads it
    // directly via q6k_value(); MLX has no native Q6_K so it stays on the
    // affine8 specs and skips this one (Op::kEmbedQ6KRaw).
    if (emb->type == GgufFile::kQ6_K) {
      _specs.push_back({P + "embed_tokens.q6k", "Q6K", {vocab, hidden},
                        emb->nbytes, "token_embd.weight",
                        ConvertedTensorSpec::Op::kEmbedQ6KRaw});
    }
  }

  add_float("output_norm.weight", P + "norm.weight");

  for (int L = 0; L < _cfg.n_layers; ++L) {
    const std::string bp = "blk." + std::to_string(L) + ".";
    const std::string hp = P + "layers." + std::to_string(L) + ".";
    add_q4(bp + "attn_q.weight",      hp + "self_attn.q_proj");
    add_q4(bp + "attn_k.weight",      hp + "self_attn.k_proj");
    add_q4(bp + "attn_v.weight",      hp + "self_attn.v_proj");
    add_q4(bp + "attn_output.weight", hp + "self_attn.o_proj");
    add_q4(bp + "ffn_gate.weight",    hp + "mlp.gate_proj");
    add_q4(bp + "ffn_up.weight",      hp + "mlp.up_proj");
    add_q4(bp + "ffn_down.weight",    hp + "mlp.down_proj");
    add_float(bp + "attn_q_norm.weight", hp + "self_attn.q_norm.weight");
    add_float(bp + "attn_k_norm.weight", hp + "self_attn.k_norm.weight");
    add_float(bp + "attn_norm.weight",   hp + "input_layernorm.weight");
    add_float(bp + "post_attention_norm.weight",
              hp + "post_attention_layernorm.weight");
    add_float(bp + "ffn_norm.weight", hp + "pre_feedforward_layernorm.weight");
    add_float(bp + "post_ffw_norm.weight",
              hp + "post_feedforward_layernorm.weight");
    add_float(bp + "layer_output_scale.weight", hp + "layer_scalar");
  }
}

bool
GgufGemma4Converter::ensure_embed_()
{
  if (_embed_ready) { return true; }
  const GgufFile::Tensor* t = _g->tensor("token_embd.weight");
  if (t == nullptr || t->dims.size() != 2) { return false; }
  const std::int64_t hidden = t->dims[0], vocab = t->dims[1];
  const std::int64_t wpr = hidden / 4;           // 8-bit: 4 values per u32
  const std::int64_t gpr = hidden / kEmbedGroup;
  _embed_w.assign(static_cast<std::size_t>(vocab) * wpr, 0);
  _embed_s.resize(static_cast<std::size_t>(vocab) * gpr);
  _embed_b.resize(static_cast<std::size_t>(vocab) * gpr);
  std::vector<float> row(static_cast<std::size_t>(hidden));
  for (std::int64_t r = 0; r < vocab; ++r) {
    if (!_g->dequant_row_f32(*t, r, row.data())) { return false; }
    std::uint32_t* wrow = _embed_w.data() + r * wpr;
    std::uint16_t* srow = _embed_s.data() + r * gpr;
    std::uint16_t* brow = _embed_b.data() + r * gpr;
    for (std::int64_t gi = 0; gi < gpr; ++gi) {
      const float* seg = row.data() + gi * kEmbedGroup;
      float mn = seg[0], mx = seg[0];
      for (int k = 1; k < kEmbedGroup; ++k) {
        mn = std::min(mn, seg[k]);
        mx = std::max(mx, seg[k]);
      }
      float scale = (mx > mn) ? (mx - mn) / 255.0f : 1.0f;
      const std::uint16_t s16 = f32_to_f16_(scale);
      const std::uint16_t b16 = f32_to_f16_(mn);
      const float sf = f16_to_f32_(s16);
      const float bf = f16_to_f32_(b16);
      const float inv = (sf != 0.0f) ? 1.0f / sf : 0.0f;
      srow[gi] = s16;
      brow[gi] = b16;
      for (int k = 0; k < kEmbedGroup; ++k) {
        int q = static_cast<int>(std::lround((seg[k] - bf) * inv));
        q = std::min(255, std::max(0, q));
        const std::int64_t c = gi * kEmbedGroup + k;
        wrow[c / 4] |= static_cast<std::uint32_t>(q) << (8 * (c % 4));
      }
    }
  }
  _embed_ready = true;
  return true;
}

void
GgufGemma4Converter::maybe_release_embed_()
{
  if (_embed_served == 0x7u) {
    _embed_w.clear(); _embed_w.shrink_to_fit();
    _embed_s.clear(); _embed_s.shrink_to_fit();
    _embed_b.clear(); _embed_b.shrink_to_fit();
    _embed_ready = false;
  }
}

bool
GgufGemma4Converter::convert(const ConvertedTensorSpec& spec,
                             std::uint8_t* dst)
{
  using Op = ConvertedTensorSpec::Op;
  const GgufFile::Tensor* t = _g->tensor(spec.gguf_name);
  if (t == nullptr) { return false; }

  switch (spec.op) {
    case Op::kQ4Weight: {
      if (t->dims.size() != 2 || t->type != GgufFile::kQ4_0) { return false; }
      const std::int64_t in = t->dims[0], out = t->dims[1];
      const std::int64_t nblk = in / kQ4Group;
      const std::int64_t row_bytes = nblk * 18;
      auto* d = reinterpret_cast<std::uint32_t*>(dst);
      std::memset(dst, 0, spec.nbytes);
      for (std::int64_t r = 0; r < out; ++r) {
        const std::uint8_t* rp = t->data + r * row_bytes;
        std::uint32_t* drow = d + r * (in / 8);
        for (std::int64_t b = 0; b < nblk; ++b) {
          const std::uint8_t* qs = rp + b * 18 + 2;
          for (int j = 0; j < 16; ++j) {
            const std::int64_t clo = b * kQ4Group + j;
            const std::int64_t chi = clo + 16;
            drow[clo / 8] |= static_cast<std::uint32_t>(qs[j] & 0x0F)
                             << (4 * (clo % 8));
            drow[chi / 8] |= static_cast<std::uint32_t>(qs[j] >> 4)
                             << (4 * (chi % 8));
          }
        }
      }
      return true;
    }
    case Op::kQ4Scales:
    case Op::kQ4Biases: {
      if (t->dims.size() != 2 || t->type != GgufFile::kQ4_0) { return false; }
      const std::int64_t in = t->dims[0], out = t->dims[1];
      const std::int64_t nblk = in / kQ4Group;
      auto* d = reinterpret_cast<std::uint16_t*>(dst);
      const bool biases = (spec.op == Op::kQ4Biases);
      for (std::int64_t r = 0; r < out; ++r) {
        const std::uint8_t* rp = t->data + r * nblk * 18;
        std::uint16_t* drow = d + r * nblk;
        for (std::int64_t b = 0; b < nblk; ++b) {
          std::uint16_t d16;
          std::memcpy(&d16, rp + b * 18, 2);
          drow[b] = biases ? f32_to_f16_(-8.0f * f16_to_f32_(d16)) : d16;
        }
      }
      return true;
    }
    case Op::kEmbedWeight:
    case Op::kEmbedScales:
    case Op::kEmbedBiases: {
      if (!ensure_embed_()) { return false; }
      if (spec.op == Op::kEmbedWeight) {
        std::memcpy(dst, _embed_w.data(), spec.nbytes);
        _embed_served |= 0x1u;
      } else if (spec.op == Op::kEmbedScales) {
        std::memcpy(dst, _embed_s.data(), spec.nbytes);
        _embed_served |= 0x2u;
      } else {
        std::memcpy(dst, _embed_b.data(), spec.nbytes);
        _embed_served |= 0x4u;
      }
      maybe_release_embed_();
      return true;
    }
    case Op::kEmbedQ6KRaw: {
      // Lossless raw copy of the Q6_K super-blocks (metal reads them with
      // q6k_value()). No dequant/requant -- cheap, exact, ~25% smaller.
      if (t->type != GgufFile::kQ6_K || t->nbytes != spec.nbytes) {
        return false;
      }
      std::memcpy(dst, t->data, spec.nbytes);
      return true;
    }
    case Op::kFloatPass: {
      return _g->dequant_all_f32(*t, reinterpret_cast<float*>(dst));
    }
    case Op::kKQuantRaw:
    case Op::kSsmALog:
      return false;   // qwen35-only ops; never produced by this converter
  }
  return false;
}

// ---------------------------------------------------------------------
// GgufQwen35Converter -- native k-quant (no requant) for the metal Qwen.
// ---------------------------------------------------------------------
namespace {
// ggml type -> the short dtype tag the metal Qwen loader keys off.
const char*
kquant_dtype_(std::uint32_t t)
{
  switch (t) {
    case GgufFile::kQ4_K: return "Q4K";
    case GgufFile::kQ5_K: return "Q5K";
    case GgufFile::kQ6_K: return "Q6K";
    default:              return nullptr;   // not a supported k-quant
  }
}
}  // namespace

GgufQwen35Converter::GgufQwen35Converter(const GgufFile* gguf,
                                         const ModelConfig& cfg)
  : _g(gguf), _cfg(cfg)
{
  build_specs_();
}

void
GgufQwen35Converter::build_specs_()
{
  using Op = ConvertedTensorSpec::Op;
  // A raw k-quant linear weight: GGUF dims are [ne0=in(K), ne1=out(N)];
  // the metal qmv reads it row-major as [N, K-blocked] -- byte-identical
  // to the GGUF payload, so this is a pure passthrough (dtype tags the
  // k-quant family for the loader's per-tensor dispatch).
  auto add_kq = [&](const std::string& gname, const std::string& hf) {
    const GgufFile::Tensor* t = _g->tensor(gname);
    if (t == nullptr || t->dims.size() != 2) { return; }
    const char* dt = kquant_dtype_(t->type);
    if (dt == nullptr) { return; }
    const std::int64_t in = t->dims[0], out = t->dims[1];
    _specs.push_back({hf, dt, {out, in}, t->nbytes, gname, Op::kKQuantRaw});
  };
  // An f32 side-tensor (norm / conv1d / dt_bias / dequant of a tiny Q8_0
  // alpha/beta projection): GgufFile::dequant_all_f32 handles f32 + q8_0.
  auto add_f32 = [&](const std::string& gname, const std::string& hf,
                     Op op = Op::kFloatPass) {
    const GgufFile::Tensor* t = _g->tensor(gname);
    if (t == nullptr) { return; }
    _specs.push_back({hf, "F32", t->dims,
                      static_cast<std::uint64_t>(t->numel()) * 4, gname, op});
  };

  const std::string P = "language_model.model.";

  // token_embd (tied lm_head): raw Q6_K table, gathered + matvec'd natively.
  if (const GgufFile::Tensor* emb = _g->tensor("token_embd.weight");
      emb != nullptr && emb->dims.size() == 2 &&
      emb->type == GgufFile::kQ6_K) {
    const std::int64_t hidden = emb->dims[0], vocab = emb->dims[1];
    _specs.push_back({P + "embed_tokens.q6k", "Q6K", {vocab, hidden},
                      emb->nbytes, "token_embd.weight", Op::kEmbedQ6KRaw});
  }
  add_f32("output_norm.weight", P + "norm.weight");

  for (int L = 0; L < _cfg.n_layers; ++L) {
    const std::string bp = "blk." + std::to_string(L) + ".";
    const std::string hp = P + "layers." + std::to_string(L) + ".";
    const bool linear = L < (int)_cfg.is_linear_layer.size() &&
                        _cfg.is_linear_layer[(std::size_t)L];
    add_f32(bp + "attn_norm.weight",            hp + "input_layernorm.weight");
    add_f32(bp + "post_attention_norm.weight",
            hp + "post_attention_layernorm.weight");
    if (linear) {
      add_kq(bp + "attn_qkv.weight",  hp + "linear_attn.in_proj_qkv.weight");
      add_kq(bp + "attn_gate.weight", hp + "linear_attn.in_proj_z.weight");
      add_kq(bp + "ssm_out.weight",   hp + "linear_attn.out_proj.weight");
      // The two tiny Q8_0 alpha/beta projections -> f32 (loader -> f16).
      add_f32(bp + "ssm_alpha.weight", hp + "linear_attn.in_proj_a.weight");
      add_f32(bp + "ssm_beta.weight",  hp + "linear_attn.in_proj_b.weight");
      add_f32(bp + "ssm_conv1d.weight", hp + "linear_attn.conv1d.weight");
      add_f32(bp + "ssm_dt.bias",       hp + "linear_attn.dt_bias");
      add_f32(bp + "ssm_norm.weight",   hp + "linear_attn.norm.weight");
      add_f32(bp + "ssm_a", hp + "linear_attn.A_log", Op::kSsmALog);
    } else {
      add_kq(bp + "attn_q.weight",      hp + "self_attn.q_proj.weight");
      add_kq(bp + "attn_k.weight",      hp + "self_attn.k_proj.weight");
      add_kq(bp + "attn_v.weight",      hp + "self_attn.v_proj.weight");
      add_kq(bp + "attn_output.weight", hp + "self_attn.o_proj.weight");
      add_f32(bp + "attn_q_norm.weight", hp + "self_attn.q_norm.weight");
      add_f32(bp + "attn_k_norm.weight", hp + "self_attn.k_norm.weight");
    }
    add_kq(bp + "ffn_gate.weight", hp + "mlp.gate_proj.weight");
    add_kq(bp + "ffn_up.weight",   hp + "mlp.up_proj.weight");
    add_kq(bp + "ffn_down.weight", hp + "mlp.down_proj.weight");
  }

  // NextN / MTP draft block(s): the bundled speculative head. One full-attn
  // decoder layer (same shapes as a main full layer) plus the eh-fusion
  // projection and its three norms, exposed under the mtp.* HF names the metal
  // MTP loader reads (mirrors the sidecar mtp.safetensors layout, but the
  // payloads are native k-quant / q8_0 here). Only depth-1 (nextn==1) is wired.
  if (_cfg.num_nextn_layers > 0) {
    const std::string bp = "blk." + std::to_string(_cfg.n_layers) + ".";
    const std::string mp = "mtp.";
    // eh_proj: q8_0 [in=2H, out=H] -> f32 [H, 2H]; the loader narrows to f16
    // and splits into its embedding / hidden halves.
    add_f32(bp + "nextn.eh_proj.weight",         mp + "fc.weight");
    add_f32(bp + "nextn.enorm.weight",           mp + "pre_fc_norm_embedding.weight");
    add_f32(bp + "nextn.hnorm.weight",           mp + "pre_fc_norm_hidden.weight");
    add_f32(bp + "nextn.shared_head_norm.weight", mp + "norm.weight");
    add_f32(bp + "attn_norm.weight",   mp + "layers.0.input_layernorm.weight");
    add_f32(bp + "post_attention_norm.weight",
            mp + "layers.0.post_attention_layernorm.weight");
    add_f32(bp + "attn_q_norm.weight", mp + "layers.0.self_attn.q_norm.weight");
    add_f32(bp + "attn_k_norm.weight", mp + "layers.0.self_attn.k_norm.weight");
    add_kq(bp + "attn_q.weight",      mp + "layers.0.self_attn.q_proj.weight");
    add_kq(bp + "attn_k.weight",      mp + "layers.0.self_attn.k_proj.weight");
    add_kq(bp + "attn_v.weight",      mp + "layers.0.self_attn.v_proj.weight");
    add_kq(bp + "attn_output.weight", mp + "layers.0.self_attn.o_proj.weight");
    add_kq(bp + "ffn_gate.weight",    mp + "layers.0.mlp.gate_proj.weight");
    add_kq(bp + "ffn_up.weight",      mp + "layers.0.mlp.up_proj.weight");
    add_kq(bp + "ffn_down.weight",    mp + "layers.0.mlp.down_proj.weight");
  }
}

bool
GgufQwen35Converter::convert(const ConvertedTensorSpec& spec,
                             std::uint8_t* dst)
{
  using Op = ConvertedTensorSpec::Op;
  const GgufFile::Tensor* t = _g->tensor(spec.gguf_name);
  if (t == nullptr) { return false; }
  switch (spec.op) {
    case Op::kKQuantRaw:
    case Op::kEmbedQ6KRaw: {
      // Lossless raw copy of the k-quant super-blocks; the metal forward
      // reads them with q4k/q5k/q6k_value().
      if (t->nbytes != spec.nbytes) { return false; }
      std::memcpy(dst, t->data, spec.nbytes);
      return true;
    }
    case Op::kFloatPass:
      return _g->dequant_all_f32(*t, reinterpret_cast<float*>(dst));
    case Op::kSsmALog: {
      // ssm_a = -exp(A_log); recover A_log = log(-ssm_a). The metal GDN
      // step re-derives -exp(A_log) inside the kernel (matching HF).
      const std::size_t n = (std::size_t)t->numel();
      std::vector<float> tmp(n);
      if (!_g->dequant_all_f32(*t, tmp.data())) { return false; }
      auto* out = reinterpret_cast<float*>(dst);
      for (std::size_t i = 0; i < n; ++i) {
        const float a = tmp[i];
        out[i] = (a < 0.0f) ? std::log(-a) : 0.0f;
      }
      return true;
    }
    default:
      return false;   // gemma-only ops
  }
}

}  // namespace vpipe::genai

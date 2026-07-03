#include "generative-models/quantize/arch-detect.h"

#include "common/flex-data.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/model-loader.h"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <string>

namespace vpipe::genai {

namespace {

// Split a tensor name ".../layers.<N>.<suffix>" into the prefix up to and
// including "layers." and the integer layer index. Returns false when the
// name doesn't end with `suffix` or carries no ".layers.<digits>." segment.
bool
split_layer_(const std::string& name, const std::string& suffix,
             std::string& prefix, int& idx)
{
  if (name.size() < suffix.size() ||
      name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) {
    return false;
  }
  static const std::string mark = ".layers.";
  const std::size_t p = name.rfind(mark);
  if (p == std::string::npos) { return false; }
  const std::size_t num = p + mark.size();
  std::size_t e = num;
  while (e < name.size() &&
         std::isdigit(static_cast<unsigned char>(name[e]))) {
    ++e;
  }
  if (e == num) { return false; }
  idx = std::atoi(name.c_str() + num);
  prefix = name.substr(0, num);   // includes the trailing "layers."
  return true;
}

// Map an HF config.json model_type to a vpipe family tag.
std::string
arch_tag_(const std::string& model_type)
{
  auto starts = [&](const char* p) {
    return model_type.rfind(p, 0) == 0;
  };
  if (model_type == "moss_tts_local") { return "moss-tts-local"; }
  if (starts("qwen3_5"))              { return "qwen3.5"; }
  if (model_type == "qwen3")          { return "qwen3"; }
  if (model_type == "qwen2")          { return "qwen2"; }
  if (model_type == "llama")          { return "llama3"; }
  if (starts("gemma"))               { return "gemma4"; }
  if (model_type.empty())            { return "unknown"; }
  return model_type;
}

}  // namespace

QuantArchInfo
detect_quant_arch(const SessionContextIntf* session, const std::string& src_dir)
{
  QuantArchInfo m;

  // ---- 1. config.json: model_type + text-config dims --------------------
  std::string model_type;
  int   n_layers_cfg = 0, hidden = 0, n_heads = 0, kv = 0, head_dim = 0;
  int   ffn = 0, vocab = 0;
  float rope = 1.0e6f, eps = 1e-6f;
  {
    std::ifstream in(src_dir + "/config.json");
    if (in) {
      try {
        FlexData root = FlexData::from_json(in);
        if (root.is_object()) {
          auto ro = root.as_object();
          if (ro.contains("model_type")) {
            model_type = std::string(ro.at("model_type").as_string(""));
          }
          // HF wraps the LM-side parameters under one of these for the
          // multimodal / wrapped architectures; else they sit at the root.
          static const char* const kSubs[] = {
              "text_config", "language_config", "qwen3_config",
              "thinker_config"};
          FlexData sub;
          bool have_sub = false;
          for (const char* s : kSubs) {
            if (ro.contains(s)) {
              FlexData t = ro.at(s);
              if (t.is_object()) { sub = t; have_sub = true; break; }
            }
          }
          auto read_dims = [&](const FlexData& objfd) {
            auto o = objfd.as_object();
            auto gi = [&](const char* k, int d) {
              return o.contains(k) ? (int)o.at(k).as_int(d) : d;
            };
            n_layers_cfg = gi("num_hidden_layers", n_layers_cfg);
            hidden       = gi("hidden_size", hidden);
            n_heads      = gi("num_attention_heads", n_heads);
            kv           = gi("num_key_value_heads", kv);
            head_dim     = gi("head_dim", head_dim);
            ffn          = gi("intermediate_size", ffn);
            vocab        = gi("vocab_size", vocab);
            if (o.contains("rope_theta")) {
              rope = (float)o.at("rope_theta").as_real(rope);
            }
            if (o.contains("rms_norm_eps")) {
              eps = (float)o.at("rms_norm_eps").as_real(eps);
            }
          };
          if (have_sub) { read_dims(sub); } else { read_dims(root); }
        }
      } catch (...) {}
    }
  }
  m.arch = arch_tag_(model_type);

  // ---- 2. probe the safetensors layer layout ----------------------------
  // Anchor on self_attn.q_proj-bearing layers; pick the prefix with the most
  // such layers (the LM stack, vs a vision/audio tower that also uses
  // ".layers."). Then count ALL layer indices under that prefix so hybrid
  // (gated-DeltaNet) layers still contribute to the total.
  auto wts = MetalLlamaWeights::open_model(src_dir);
  if (wts.has_value()) {
    const std::string qsfx = ".self_attn.q_proj.weight";
    std::map<std::string, std::set<int>> attn_by_prefix;
    const std::vector<std::string> names = wts->tensor_names();
    for (const std::string& n : names) {
      std::string pfx; int idx = 0;
      if (split_layer_(n, qsfx, pfx, idx)) {
        attn_by_prefix[pfx].insert(idx);
      }
    }
    if (!attn_by_prefix.empty()) {
      const std::string* best = nullptr;
      std::size_t best_n = 0;
      for (const auto& kvp : attn_by_prefix) {
        if (kvp.second.size() > best_n) {
          best_n = kvp.second.size();
          best = &kvp.first;
        }
      }
      m.layer_prefix   = *best;
      m.n_attn_layers  = (int)best_n;
      // Total layers = max index + 1 over every tensor under this prefix.
      int maxidx = -1;
      for (const std::string& n : names) {
        if (n.rfind(m.layer_prefix, 0) != 0) { continue; }
        const std::size_t s = m.layer_prefix.size();
        std::size_t e = s;
        while (e < n.size() &&
               std::isdigit(static_cast<unsigned char>(n[e]))) {
          ++e;
        }
        if (e > s) {
          const int L = std::atoi(n.c_str() + s);
          if (L > maxidx) { maxidx = L; }
        }
      }
      m.n_layers = maxidx + 1;
      m.detected = true;

      // Count gated-DeltaNet layers (Qwen3.5 hybrid): they carry
      // linear_attn.in_proj_qkv instead of self_attn.q_proj, but also start
      // with input_layernorm, so their in-projection group folds the same way.
      int n_gdn = 0;
      for (int L = 0; L < m.n_layers; ++L) {
        if (wts->info(m.layer_prefix + std::to_string(L) +
                      ".linear_attn.in_proj_qkv.weight") != nullptr) {
          ++n_gdn;
        }
      }

      // Structural AWQ/SmoothQuant compatibility: every layer is a foldable
      // block (full attention OR gated-DeltaNet, both rooted at
      // input_layernorm) + the standard input/post-attention layernorm pair +
      // a plain dense MLP (gate/up/down -- not MoE). Excludes the Gemma FFN-norm
      // layout (gate/up input is pre_feedforward_layernorm, so the fold target
      // would be wrong).
      const std::string l0 = m.layer_prefix + "0.";
      const bool std_ln =
          wts->info(l0 + "input_layernorm.weight") != nullptr &&
          wts->info(l0 + "post_attention_layernorm.weight") != nullptr;
      const bool gemma_ffn_ln =
          wts->info(l0 + "pre_feedforward_layernorm.weight") != nullptr;
      const bool dense_mlp =
          wts->info(l0 + "mlp.gate_proj.weight") != nullptr;
      // MoE stacks (3D expert slabs) are also AWQ-foldable: the attention
      // in-projection group folds the same way; the experts/shared/router are
      // plain-quantized (per-expert AWQ is a later milestone).
      const bool moe_mlp =
          wts->info(l0 + "mlp.experts.gate_up_proj") != nullptr;
      m.awq_ok = (m.n_attn_layers + n_gdn == m.n_layers) && m.n_layers > 0 &&
                 std_ln && (dense_mlp || moe_mlp) && !gemma_ffn_ln;
    }
  }

  // config.json layer count is the fallback when the probe found none.
  if (m.n_layers <= 0) { m.n_layers = n_layers_cfg; }

  // weight_prefix = the layer prefix without the trailing "layers." (used by
  // the manually-built backbone configs below).
  std::string wp = m.layer_prefix;
  {
    static const std::string suf = "layers.";
    if (wp.size() >= suf.size() &&
        wp.compare(wp.size() - suf.size(), suf.size(), suf) == 0) {
      wp.resize(wp.size() - suf.size());
    }
  }

  // ---- 3. on-device calibration capability + backbone config ------------
  // MetalQwenModel runs the Qwen3 family: dense full-attention backbones AND
  // the Qwen3.5 full-attn + gated-DeltaNet hybrid. For the Qwen3.5 family we
  // reuse MetalQwenModel::config_from(ModelLoader::load_config) so the GDN
  // dims / full-attn interval / mROPE / MoE flags are exactly right; for the
  // MOSS dense backbone (under "transformer.") we build the config directly.
  const bool qwen35_family =
      model_type == "qwen3_5" || model_type.rfind("qwen3_5", 0) == 0;
  if (m.awq_ok && model_type == "moss_tts_local") {
    MetalQwenModel::Config& c = m.backbone;
    c.n_layers   = m.n_layers;
    c.hidden     = hidden;
    c.n_heads    = n_heads;
    c.n_kv_heads = kv;
    c.head_dim   = head_dim > 0 ? head_dim
                                : (n_heads > 0 ? hidden / n_heads : 0);
    c.ffn_inner  = ffn;
    c.vocab      = vocab;
    c.rope_theta = rope;
    c.rms_eps    = eps;
    c.rotary_dim = c.head_dim;
    c.full_attn_interval = 1;
    c.tie_embeddings  = false;
    c.use_bf16        = true;
    c.quant_bits      = 8;        // calibration runs over the 8-bit base
    c.dense           = true;
    c.attn_output_gate = false;
    c.backbone_only   = true;
    c.weight_prefix   = wp;
    c.model_seg       = "";
    c.max_seq         = 2048;
    c.page_tokens     = 256;
    m.calib_ok = true;
  } else if (m.awq_ok && (model_type == "qwen3" || qwen35_family)) {
    ModelLoader loader(session);
    std::optional<ModelConfig> mcfg = loader.load_config(src_dir);
    if (mcfg.has_value()) {
      MetalQwenModel::Config c = MetalQwenModel::config_from(*mcfg);
      // Override for a calibration backbone run over the 8-bit base.
      c.backbone_only = true;
      c.use_bf16      = true;
      c.quant_bits    = 8;
      // config_from hardcodes the LM weight root ("language_model." +
      // "model."); pin it to the PROBED layer prefix instead so it tracks the
      // checkpoint's actual naming -- the 4B uses "language_model.model." but
      // the 27B/"3.6" uses the reversed "model.language_model.".
      if (!wp.empty()) { c.weight_prefix = wp; c.model_seg = ""; }
      m.backbone = c;
      // On-device auto-calibration runs the backbone forward through
      // MetalQwenModel. Dense backbones use the full-load
      // collect_backbone_calibration; MoE uses the memory-safe layer-by-layer
      // collect_backbone_calibration_streaming (never resides the whole expert
      // stack), which also taps the PER-EXPERT gate/up + down stats. Either way
      // the stage runs it when AWQ is on and no calib_dir was supplied.
      m.calib_ok = true;
    }
  }
  return m;
}

}  // namespace vpipe::genai

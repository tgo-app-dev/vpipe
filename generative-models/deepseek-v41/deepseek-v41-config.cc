#include "generative-models/deepseek-v41/deepseek-v41-config.h"

#include "common/flex-data.h"

#include <algorithm>
#include <filesystem>
#include <format>
#include <fstream>
#include <sstream>

namespace vpipe::genai {

int
DeepseekV41Config::engram_slot(int L) const
{
  for (std::size_t i = 0; i < engram_layers.size(); ++i) {
    if (engram_layers[i] == L) { return (int)i; }
  }
  return -1;
}

int
DeepseekV41Config::ced_mid() const
{
  const int NL = (int)plan.size();
  for (int L = 0; L < NL; ++L) {
    const Plan& p = plan[(std::size_t)L];
    if (!p.kv_source || p.ratio != 1) { continue; }
    for (int l = L + 1; l < NL; ++l) {
      const Plan& q = plan[(std::size_t)l];
      if (q.ratio != 1 || q.kv_source || q.kv_from != L ||
          q.keys_from != L || q.index_from < L) {
        return -1;
      }
    }
    if (candidate_source >= 0 && candidate_source < L) { return -1; }
    for (int e : engram_layers) {
      if (e >= L) { return -1; }
    }
    return L;
  }
  return -1;
}

bool
DeepseekV41Config::from_dir(const std::string& dir, DeepseekV41Config* out,
                            std::string* err)
{
  std::ifstream in(std::filesystem::path(dir) / "config.json");
  if (!in) {
    if (err != nullptr) { *err = "no config.json under " + dir; }
    return false;
  }
  std::stringstream ss;
  ss << in.rdbuf();
  return from_json_text(ss.str(), out, err);
}

bool
DeepseekV41Config::from_json_text(const std::string& text,
                                  DeepseekV41Config* out, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = "DeepSeek-V4.1 config: " + std::move(m); }
    return false;
  };
  FlexData fd;
  try {
    fd = FlexData::from_json(text);
  } catch (const std::exception& e) {
    return fail(e.what());
  }
  if (!fd.is_object()) { return fail("config.json is not an object"); }
  auto root = fd.as_object();
  auto str = [](const auto& o, const char* k) -> std::string {
    return o.contains(k) && o.at(k).is_string()
               ? std::string(o.at(k).as_string()) : std::string();
  };
  auto geti = [](const auto& o, const char* k, int d) {
    return o.contains(k) && !o.at(k).is_null() ? (int)o.at(k).as_int(d) : d;
  };
  auto getf = [](const auto& o, const char* k, float d) {
    return o.contains(k) && !o.at(k).is_null() ? (float)o.at(k).as_real(d)
                                               : d;
  };
  auto getb = [](const auto& o, const char* k, bool d) {
    return o.contains(k) && o.at(k).is_bool() ? o.at(k).as_bool(d) : d;
  };
  auto ints = [](const auto& o, const char* k) {
    std::vector<int> v;
    if (o.contains(k) && o.at(k).is_array()) {
      const FlexData a = o.at(k);
      for (auto e : a.as_array()) { v.push_back((int)e.as_int()); }
    }
    return v;
  };
  if (str(root, "model_type") != "deepseek_v41") {
    return fail("model_type is not deepseek_v41");
  }
  if (!root.contains("text_config") || !root.at("text_config").is_object()) {
    return fail("no text_config");
  }
  DeepseekV41Config c;
  c.image_token = geti(root, "image_token_id", c.image_token);
  c.bos = geti(root, "bos_token_id", c.bos);
  c.eos = geti(root, "eos_token_id", c.eos);
  if (root.contains("quantization_config") &&
      root.at("quantization_config").is_object()) {
    const FlexData qfd = root.at("quantization_config");
    auto q = qfd.as_object();
    const std::string m = str(q, "quant_method");
    c.fp8 = m == "fp8";
    if (c.fp8) {
      std::vector<int> blk = ints(q, "weight_block_size");
      if (blk != std::vector<int>{32, 32} ||
          str(q, "scale_fmt") != "ue8m0") {
        return fail("fp8 weights are not 32x32 blocks with ue8m0 scales");
      }
    }
    c.fp4_experts = str(q, "expert_dtype") == "fp4";
  } else if (root.contains("omlx_deepseek_v41") &&
             root.at("omlx_deepseek_v41").is_object()) {
    // An oMLX pack (MLX layout): its own record of every quantized module.
    // oQ4e keeps the release's numbers -- the routed experts MXFP4, every
    // other quantized projection MXFP8 (both a scale per 32); the packs
    // below it (oQ2e, oQ2.5e, oQ3e) requantize the backbone's experts and
    // projections to MLX's affine codes (2-8 bits, a bf16 scale and bias
    // per group), the width chosen per module. The n-gram tables are 4-bit
    // affine in groups of 32 (or the release's FP8). Anything else is
    // refused by name. Their `quantize_input` (oMLX rounding a module's
    // input to the release's FP8 activations) is not followed: the port
    // keeps activations in the compute dtype, as for the release's own.
    const FlexData ofd = root.at("omlx_deepseek_v41");
    auto o = ofd.as_object();
    if (o.contains("quantized_modules") &&
        o.at("quantized_modules").is_object()) {
      const FlexData mfd = o.at("quantized_modules");
      for (const auto& [name, m] : mfd.as_object()) {
        if (!m.is_object()) { continue; }
        auto mo = m.as_object();
        const bool expert =
            std::string_view(name).find(".ffn.experts.") !=
            std::string_view::npos;
        const std::string mode = str(mo, "mode");
        const int bits = geti(mo, "bits", 0);
        const int gs = geti(mo, "group_size", 0);
        const bool affine = mode == "affine" &&
                            (bits >= 2 && bits <= 8 && bits != 7) &&
                            (gs == 32 || gs == 64 || gs == 128);
        if (affine) {
          c.affine = true;
          continue;
        }
        if (expert ? mode != "mxfp4" || bits != 4
                   : mode != "mxfp8" || bits != 8) {
          return fail(std::string("oMLX module ") + std::string(name) +
                      " is " + mode + " at " + std::to_string(bits) +
                      " bits; read are MXFP4 experts, MXFP8 projections "
                      "and affine codes of 2-6 or 8 bits in groups of 32, "
                      "64 or 128");
        }
      }
    }
    if (o.contains("engram_tables") && o.at("engram_tables").is_object()) {
      const FlexData efd = o.at("engram_tables");
      for (const auto& [name, m] : efd.as_object()) {
        if (!m.is_object()) { continue; }
        auto mo = m.as_object();
        // oQ's 4-bit affine, or (its -ngram packs) the release's FP8
        const std::string mode = str(mo, "mode");
        const bool fp8 = mode == "mxfp8" && geti(mo, "group_size", 0) == 32;
        if (!fp8 && (mode != "affine" || geti(mo, "bits", 0) != 4 ||
                     geti(mo, "group_size", 0) != 32)) {
          return fail(std::string("oMLX n-gram table ") + std::string(name) +
                      " is neither 4-bit affine nor MXFP8 in groups of 32");
        }
      }
    }
    c.fp8 = true;
    c.fp4_experts = true;
  } else {
    c.fp8 = false;
    c.fp4_experts = false;
  }
  const FlexData tfd = root.at("text_config");
  auto t = tfd.as_object();
  c.vocab = geti(t, "vocab_size", c.vocab);
  c.dim = geti(t, "hidden_size", c.dim);
  c.n_layers = geti(t, "num_hidden_layers", c.n_layers);
  c.n_mtp = geti(t, "num_nextn_predict_layers", 0);
  c.n_heads = geti(t, "num_attention_heads", c.n_heads);
  c.head_dim = geti(t, "head_dim", c.head_dim);
  c.rope_dim = geti(t, "qk_rope_head_dim", c.rope_dim);
  c.q_lora = geti(t, "q_lora_rank", c.q_lora);
  c.o_lora = geti(t, "o_lora_rank", c.o_lora);
  c.o_groups = geti(t, "o_groups", c.o_groups);
  c.window = geti(t, "sliding_window", c.window);
  c.eps = getf(t, "rms_norm_eps", c.eps);
  c.moe_inter = geti(t, "moe_intermediate_size", c.moe_inter);
  c.n_experts = geti(t, "n_routed_experts", c.n_experts);
  c.top_k = geti(t, "num_experts_per_tok", c.top_k);
  c.n_shared = geti(t, "n_shared_experts", c.n_shared);
  c.route_scale = getf(t, "routed_scaling_factor", c.route_scale);
  c.swiglu_limit = getf(t, "swiglu_limit", 0.0f);
  c.norm_topk = getb(t, "norm_topk_prob", true);
  c.score_func = str(t, "scoring_func");
  c.rope_theta = getf(t, "rope_theta", c.rope_theta);
  c.compress_rope_theta =
      getf(t, "compress_rope_theta", c.compress_rope_theta);
  c.orig_seq = 0;
  if (t.contains("rope_scaling") && t.at("rope_scaling").is_object()) {
    const FlexData rfd = t.at("rope_scaling");
    auto r = rfd.as_object();
    const std::string kind = str(r, "rope_type");
    if (kind == "yarn") {
      c.rope_factor = getf(r, "factor", c.rope_factor);
      c.beta_fast = geti(r, "beta_fast", c.beta_fast);
      c.beta_slow = geti(r, "beta_slow", c.beta_slow);
      c.orig_seq = geti(r, "original_max_position_embeddings", 0);
    } else if (!kind.empty() && kind != "default") {
      return fail("rope_type \"" + kind + "\" is not yarn");
    }
  }
  c.compress_ratios = ints(t, "compress_ratios");
  c.kv_sources = ints(t, "kv_source_layer_ids");
  c.index_sources = ints(t, "index_source_layer_ids");
  c.candidate_source = geti(t, "candidate_source_layer_id", -1);
  c.candidate_blocks = geti(t, "candidate_topk_blocks", 0);
  c.candidate_block = geti(t, "candidate_block_size", 0);
  c.index_heads = geti(t, "index_n_heads", c.index_heads);
  c.index_dim = geti(t, "index_head_dim", c.index_dim);
  c.index_topk = geti(t, "index_topk", c.index_topk);
  c.hc = geti(t, "hc_mult", c.hc);
  c.hc_iters = geti(t, "hc_sinkhorn_iters", c.hc_iters);
  c.hc_eps = getf(t, "hc_eps", c.hc_eps);
  c.engram_layers = ints(t, "engram_layer_ids");
  if (t.contains("engram_num_embeddings") &&
      t.at("engram_num_embeddings").is_array()) {
    const FlexData a = t.at("engram_num_embeddings");
    for (auto e : a.as_array()) { c.engram_rows.push_back(e.as_int()); }
  }
  c.engram_ngram = geti(t, "engram_max_ngram_size", c.engram_ngram);
  c.engram_vocab = geti(t, "engram_vocab_size", c.engram_vocab);
  c.engram_heads = geti(t, "engram_n_heads", c.engram_heads);
  c.engram_dim = geti(t, "engram_head_dim", c.engram_dim);
  c.engram_pad = geti(t, "engram_pad_token_id", c.engram_pad);
  c.engram_cvocab = geti(t, "engram_compressed_vocab_size", 0);
  c.dspark_block = geti(t, "dspark_block_size", 0);
  c.dspark_noise = geti(t, "dspark_noise_token_id", 0);
  c.dspark_targets = ints(t, "dspark_target_layer_ids");
  c.dspark_markov = geti(t, "dspark_markov_rank", c.dspark_markov);
  c.dspark_experts = geti(t, "dspark_n_routed_experts", 0);
  c.dspark_top_k = geti(t, "dspark_num_experts_per_tok", 0);
  if (root.contains("vision_config") && root.at("vision_config").is_object()) {
    const FlexData vfd = root.at("vision_config");
    auto v = vfd.as_object();
    c.vis_layers = geti(v, "num_hidden_layers", 0);
    c.vis_dim = geti(v, "hidden_size", c.vis_dim);
    c.vis_heads = geti(v, "num_attention_heads", c.vis_heads);
    c.vis_inter = geti(v, "intermediate_size", c.vis_inter);
    c.vis_patch = geti(v, "patch_size", c.vis_patch);
    c.vis_rope_theta = getf(v, "rope_theta", c.vis_rope_theta);
    c.vis_downsample = geti(v, "downsample_ratio", c.vis_downsample);
    c.vis_max_tokens = geti(v, "max_image_tokens", c.vis_max_tokens);
    c.vis_min_pixels = geti(v, "min_pixels", c.vis_min_pixels);
    c.vis_max_wh_ratio = geti(v, "max_wh_ratio", 0);
  }

  // ---- what the port runs ---------------------------------------------
  const int NL = c.n_layers;
  if (NL <= 0 || c.dim <= 0 || c.dim % 256 != 0) {
    return fail(std::format("hidden size {} is not a multiple of 256",
                            c.dim));
  }
  if ((int)c.compress_ratios.size() != NL + c.n_mtp) {
    return fail(std::format("compress_ratios has {} entries, the layers "
                            "and DSpark stages are {}",
                            c.compress_ratios.size(), NL + c.n_mtp));
  }
  if (c.head_dim != 512 || c.rope_dim != 64) {
    return fail(std::format("head_dim {} / rope dim {}: the attention "
                            "kernels run 512 / 64", c.head_dim, c.rope_dim));
  }
  if (c.index_dim != 128 || c.index_heads > 32 || c.index_heads <= 0) {
    return fail("the indexer runs head dim 128, at most 32 heads");
  }
  if (c.n_heads % c.o_groups != 0 || c.hc != 4) {
    return fail("heads must split into o_groups; hc_mult must be 4");
  }
  if (c.score_func != "sqrtsoftplus" || c.n_shared != 1) {
    return fail("the router is sqrtsoftplus over one shared expert");
  }
  if (c.window <= 0 || c.window > 1024) {
    return fail("sliding window outside 1..1024");
  }
  if (c.fp8 != c.fp4_experts) {
    return fail("the projections and experts must be FP8 / FP4 together");
  }
  if (c.moe_inter % 256 != 0 || c.q_lora % 32 != 0 || c.o_lora % 32 != 0) {
    return fail("an inner width is not a multiple of 256 / 32");
  }
  if (!c.engram_layers.empty()) {
    if (c.engram_rows.size() != c.engram_layers.size() ||
        c.engram_ngram < 2 || c.engram_cvocab <= 0 ||
        c.engram_dim % 32 != 0) {
      return fail("engram layout is incomplete");
    }
  }
  auto has = [](const std::vector<int>& v, int L) {
    return std::find(v.begin(), v.end(), L) != v.end();
  };
  c.plan.assign((std::size_t)NL, Plan{});
  int kv = -1, idx = -1, keys = -1;
  for (int L = 0; L < NL; ++L) {
    Plan& p = c.plan[(std::size_t)L];
    p.ratio = c.compress_ratios[(std::size_t)L];
    if (p.ratio < 0 || p.ratio > 16) {
      return fail(std::format("layer {} compress ratio {}", L, p.ratio));
    }
    p.kv_source = has(c.kv_sources, L);
    p.index_source = has(c.index_sources, L);
    if ((p.kv_source || p.index_source) && p.ratio == 0) {
      return fail(std::format("layer {} is a source with no compression",
                              L));
    }
    p.owns_index_keys = p.index_source && p.kv_source;
    p.candidate_source = p.index_source && L == c.candidate_source;
    p.uses_candidates = p.index_source && c.candidate_source >= 0 &&
                        c.candidate_source < L;
    if (p.kv_source) { kv = L; }
    if (p.owns_index_keys) { keys = L; }
    if (p.index_source) { idx = L; }
    if (p.ratio > 0) {
      p.kv_from = kv;
      p.index_from = idx;
      p.keys_from = keys;
      const auto& P = c.plan;
      if (kv < 0 || idx < 0 || P[(std::size_t)kv].ratio != p.ratio ||
          P[(std::size_t)idx].ratio != p.ratio) {
        return fail(std::format("layer {} (ratio {}) has no kv / index "
                                "source of its ratio before it", L,
                                p.ratio));
      }
      if (p.index_source &&
          (keys < 0 || P[(std::size_t)keys].ratio != p.ratio)) {
        return fail(std::format("index source {} has no index keys of its "
                                "ratio", L));
      }
    }
  }
  if (c.candidate_source >= 0) {
    if (c.candidate_source >= NL ||
        !c.plan[(std::size_t)c.candidate_source].index_source ||
        c.candidate_blocks <= 0 || c.candidate_block <= 0) {
      return fail("the candidate source is not an index source");
    }
    for (int L = c.candidate_source + 1; L < NL; ++L) {
      const Plan& p = c.plan[(std::size_t)L];
      if (p.index_source &&
          p.ratio != c.plan[(std::size_t)c.candidate_source].ratio) {
        return fail("a candidate-restricted indexer at another ratio");
      }
    }
  }
  for (int m = 0; m < c.n_mtp; ++m) {
    if (c.compress_ratios[(std::size_t)(NL + m)] != 0) {
      return fail("a DSpark stage with compressed attention");
    }
  }
  *out = std::move(c);
  return true;
}

}  // namespace vpipe::genai

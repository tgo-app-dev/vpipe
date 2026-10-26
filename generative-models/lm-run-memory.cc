// lm-run-memory.cc -- see lm-run-memory.h.

#include "generative-models/lm-run-memory.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-bag.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "generative-models/deepseek-v41/deepseek-v41-config.h"
#include "generative-models/deepseek-v41/metal-deepseek-v41-model.h"
#include "generative-models/gemma4/metal-gemma-model.h"
#include "generative-models/llama3/metal-llama-model.h"
#include "generative-models/model-loader.h"
#include "generative-models/qwen3/metal-qwen-model.h"

#include <algorithm>
#include <filesystem>
#include <system_error>

namespace vpipe::genai {

namespace {

// A checkpoint whose every quantized module shares the section's one
// width: MLX affine with no per-module entry. A dense checkpoint (no
// section), a mixed pack (an entry of its own, or `false`, per departing
// module) and a GGUF file run the MLP's second buffer and the dequant
// scratch.
bool
uniform_affine_(const std::string& dir, const ModelConfig& c)
{
  if (dir.size() > 5 && dir.compare(dir.size() - 5, 5, ".gguf") == 0) {
    return false;
  }
  const FlexData q = bag::value(c.extra, model_config::kQuantization);
  if (!q.is_object()) { return false; }
  for (const auto& [k, v] : q.as_object()) {
    (void)k;
    if (v.is_object() || v.is_bool()) { return false; }
  }
  return true;
}

// The K/V `shape` can hold on a model whose contexts stop at `cap` tokens.
std::int64_t
kv_bound_(const LmRunShape& shape, std::int64_t cap)
{
  const std::int64_t ctxs = std::max(shape.contexts, 1);
  cap = std::max<std::int64_t>(cap, 1);
  std::int64_t kv = std::max<std::int64_t>(shape.kv_tokens, 1);
  if (shape.branch_tokens > 0) {
    kv = std::min(kv, cap) + (ctxs - 1) * shape.branch_tokens;
  } else {
    kv = std::min(kv, ctxs * cap);
  }
  const std::int64_t pool =
      (std::int64_t)std::max(shape.page_tokens, 1) *
      (std::int64_t)std::max<std::uint32_t>(shape.max_pages, 1);
  return std::max<std::int64_t>(std::min(kv, pool), 1);
}

}  // namespace

LmFamily
lm_family(const std::string& a)
{
  if (a == "LlamaForCausalLM") { return LmFamily::kLlama; }
  if (a == "Qwen3_5ForConditionalGeneration" ||
      a == "Qwen3_5MoeForConditionalGeneration" ||
      a == "Qwen4ExpForConditionalGeneration" ||
      a == "Qwen3ASRForConditionalGeneration" || a == "MossTTSLocalModel" ||
      a == "MossTTSRealtime" || a == "Mistral3ForConditionalGeneration") {
    return LmFamily::kQwen;
  }
  if (a == "Gemma4ForConditionalGeneration" ||
      a == "Gemma4UnifiedForConditionalGeneration") {
    return LmFamily::kGemma;
  }
  if (a == "DeepseekV41ForCausalLM") { return LmFamily::kDeepseekV41; }
  return LmFamily::kNone;
}

std::size_t
lm_run_bytes(const SessionContextIntf* session, const std::string& dir,
             const LmRunShape& shape)
{
  if (dir.empty()) { return 0; }
  ModelLoader loader(session);
  const auto mc = loader.load_config(dir);
  if (!mc.has_value() || mc->hidden <= 0) { return 0; }
  const metal_compute::MetalCompute* gpu =
      session != nullptr && session->services() != nullptr
          ? session->services()->metal_compute()
          : nullptr;
  const bool cores = gpu != nullptr && gpu->supports_matrix_cores();
  const int ctxs = std::max(shape.contexts, 1);
  switch (lm_family(mc->architecture)) {
    case LmFamily::kQwen: {
      MetalQwenModel::Config c = MetalQwenModel::config_from(*mc);
      c.page_tokens = shape.page_tokens;
      c.max_pages = (int)shape.max_pages;
      const std::int64_t kv = kv_bound_(shape, c.max_seq);
      return MetalQwenModel::run_bytes(c, kv, ctxs, shape.one_pass,
                                       !uniform_affine_(dir, *mc), cores);
    }
    case LmFamily::kGemma: {
      // as LoadedLanguageModel loads it: the pool capped at 128 pages, the
      // context at the pool
      MetalGemmaModel::Config c = MetalGemmaModel::config_from(*mc);
      const std::int64_t pages =
          std::min<std::int64_t>(shape.max_pages > 0 ? shape.max_pages : 128,
                                 128);
      c.max_seq = (int)std::min<std::int64_t>(
          (std::int64_t)shape.page_tokens * pages, 1 << 20);
      const FlexData tc = bag::value(mc->extra, model_config::kTextConfig);
      const bool wide = bag::flag(tc, "use_double_wide_mlp", false);
      const std::int64_t kv = kv_bound_(shape, c.max_seq);
      return MetalGemmaModel::run_bytes(c, kv, ctxs,
                                        !uniform_affine_(dir, *mc), wide,
                                        cores);
    }
    case LmFamily::kLlama: {
      // as LoadedLanguageModel loads it (make_metal_cfg_)
      MetalLlamaModel::Config c;
      c.n_layers = mc->n_layers;
      c.hidden = mc->hidden;
      c.n_heads = mc->n_heads;
      c.n_kv_heads = mc->n_kv_heads;
      c.head_dim = mc->head_dim;
      c.ffn_inner = mc->ffn_inner;
      c.vocab = mc->vocab_size;
      c.max_seq = 2048;
      c.page_tokens = shape.page_tokens;
      c.max_pages = (int)shape.max_pages;
      return MetalLlamaModel::run_bytes(c, kv_bound_(shape, c.max_seq));
    }
    case LmFamily::kDeepseekV41: {
      DeepseekV41Config dc;
      std::string err;
      if (!DeepseekV41Config::from_dir(dir, &dc, &err)) { return 0; }
      const MetalDeepseekV41Model::Options o;
      const int ctx = (int)kv_bound_(shape, 1 << 20);
      return MetalDeepseekV41Model::run_bytes(dc, o.max_rows, ctx,
                                              shape.draft_head) +
             (std::size_t)(ctxs - 1) *
                 MetalDeepseekV41Model::context_bytes(dc, 0);
    }
    case LmFamily::kNone:
      break;
  }
  return 0;
}

std::string
lm_run_label(const std::string& dir)
{
  std::error_code ec;
  const std::filesystem::path c =
      std::filesystem::weakly_canonical(std::filesystem::path(dir), ec);
  return "lm:" + (ec ? dir : c.string());
}

}  // namespace vpipe::genai

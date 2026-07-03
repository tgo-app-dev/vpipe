#ifndef VPIPE_GENERATIVE_MODELS_METAL_LLAMA_MODEL_EXEC_H
#define VPIPE_GENERATIVE_MODELS_METAL_LLAMA_MODEL_EXEC_H

// MetalLlamaModelExec -- ModelExec adapter over the procedural
// metal-compute MetalLlamaModel, so the metal Llama plugs into the
// LoadedLanguageModel paradigm. Step 3 of the backend-native LLM core.
//
// Key design: the metal path is token-id-native, so this adapter uses
// ModelExec's TOKEN-ID methods (prefill(tokens) / decode_one(id))
// directly and leaves the MLX-embedding forward_chunk path unused
// (returns -1). The only MLX-typed surface is last_logits(), which
// just wraps the host logits vector in an mlx::core::array. No neutral
// tensor threads through the shared interface, so the MLX path is
// untouched. Single context for v1 (matches MetalLlamaModel).

#include "generative-models/llama3/metal-llama-model.h"
#include "generative-models/model-exec.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace vpipe { class SessionContextIntf; }
namespace vpipe::metal_compute { class MetalCompute; }

namespace vpipe::genai {

class MetalLlamaModelExec : public ModelExec {
public:
  MetalLlamaModelExec(const std::string&             model_dir,
                      metal_compute::MetalCompute*   mc,
                      const MetalLlamaModel::Config& cfg,
                      const SessionContextIntf*      session);

  bool valid() const noexcept override { return _model != nullptr; }

  // The metal exec owns its own KV-page pool keyed by ContextId.
  bool owns_kv() const noexcept override { return true; }

  std::int32_t
  prefill(ContextId ctx, std::span<const std::int32_t> tokens) override;
  std::int32_t decode_one(ContextId ctx, std::int32_t id) override;
  bool pdecode_begin(ContextId ctx, std::int32_t first_token,
                     std::span<const std::int32_t> prompt,
                     const GpuSamplerParams& sp, int max_tokens,
                     int rope_first) override;
  bool pdecode_commit(ContextId ctx) override;
  std::int32_t pdecode_next(ContextId ctx) override;
  void pdecode_end(ContextId ctx) override;
  const std::vector<float>& last_logits_host() const override { return _logits; }

  // Fork the metal KV from parent to child (copy-on-branch).
  bool branch_context(ContextId parent, ContextId child) override;
  // Return this context's KV pages to the pool (no-op leak otherwise).
  void release_context(ContextId ctx) override;

  // Read-only KV length of a context (0 = never touched).
  int context_seq_len(ContextId ctx) const override;

  void set_eval_per_layer(bool) noexcept override {}

private:
  struct CtxState {
    ContextId metal_cid;
  };
  // Lazily get-or-create the metal context for a LoadedLanguageModel
  // ContextId (keyed by .v); returns nullptr if a metal context can't
  // be acquired.
  CtxState* state_for_(ContextId ctx);

  std::unique_ptr<MetalLlamaModel>             _model;
  std::unordered_map<std::uint32_t, CtxState>  _ctxmap;
  std::vector<float>                           _logits;
};

}  // namespace vpipe::genai

#endif

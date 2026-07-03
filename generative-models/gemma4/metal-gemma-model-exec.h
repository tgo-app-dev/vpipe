#ifndef VPIPE_GENERATIVE_MODELS_METAL_GEMMA_MODEL_EXEC_H
#define VPIPE_GENERATIVE_MODELS_METAL_GEMMA_MODEL_EXEC_H

// MetalGemmaModelExec -- ModelExec adapter over the procedural
// metal-compute MetalGemmaModel (no MLX in the forward). Thinner than
// MetalQwenModelExec: MetalGemmaModel owns its per-context KV keyed by
// ContextId directly (no separate metal ContextManager), so the exec
// passes the LoadedLanguageModel ContextId straight through. owns_kv()
// is true: LoadedLanguageModel drives prefill/decode via the token-id
// path (Gemma needs the raw ids for per-layer-input embeddings) and
// forks/frees this exec's KV on branch/release.

#include "generative-models/gemma4/metal-gemma-model.h"
#include "generative-models/model-exec.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace vpipe { class SessionContextIntf; }
namespace vpipe::metal_compute { class MetalCompute; }

namespace vpipe::genai {

class MetalGemmaModelExec : public ModelExec {
public:
  MetalGemmaModelExec(const std::string&             model_dir,
                      metal_compute::MetalCompute*   mc,
                      const MetalGemmaModel::Config& cfg,
                      const SessionContextIntf*      session);

  bool valid() const noexcept override { return _model != nullptr; }
  bool owns_kv() const noexcept override { return true; }

  std::int32_t
  prefill(ContextId ctx, std::span<const std::int32_t> tokens) override;

  // owns_kv multimodal prefill: `ids` is the mixed sequence (placeholder
  // ids at the mm positions), `mm_rows` is [n_mm*hidden] host f32
  // (encoder rows, prompt order), `positions[k]` the index in `ids` of mm
  // row k. The metal forward splices those rows + zeros the PLE ids.
  std::int32_t
  prefill_multimodal_ids(ContextId ctx,
                         const std::vector<std::int32_t>& ids,
                         const std::vector<float>& mm_rows, int n_mm,
                         const std::vector<int>& positions);

  std::int32_t decode_one(ContextId ctx, std::int32_t id) override;
  std::int32_t decode_one_greedy(ContextId ctx, std::int32_t id) override;

  // GPU-resident pipelined decode: forwards straight to MetalGemmaModel
  // (which maps the LM ContextId via cm_for_). Greedy + sampled.
  bool pdecode_begin(ContextId ctx, std::int32_t first_token,
                     std::span<const std::int32_t> prompt,
                     const GpuSamplerParams& sp, int max_tokens,
                     int rope_first) override;
  bool pdecode_commit(ContextId ctx) override;
  std::int32_t pdecode_next(ContextId ctx) override;
  void pdecode_end(ContextId ctx) override;
  bool pdecode_supports_runahead() const override;
  bool supports_batched_decode() const override { return _model != nullptr; }
  bool batched_decode_logits(
      std::span<const ContextId>    cids,
      std::span<const std::int32_t> in_tokens,
      std::span<const std::int32_t> rope_pos,
      std::vector<float>&           out_logits) override
  {
    return _model != nullptr
        && _model->decode_batched_step(cids, in_tokens, rope_pos, out_logits);
  }
  const std::vector<float>& last_logits_host() const override { return _logits; }

  bool branch_context(ContextId parent, ContextId child) override;
  void release_context(ContextId ctx) override;

  // Read-only KV length of a context (0 = never touched). The gemma
  // model maps the LM ContextId internally, so this forwards to it.
  int context_seq_len(ContextId ctx) const override
  {
    return _model != nullptr ? _model->context_seq_len(ctx) : 0;
  }

  void set_eval_per_layer(bool) noexcept override {}

  void set_suppressed_tokens(std::span<const std::int32_t> ids) override
  {
    if (_model != nullptr) { _model->set_suppressed_tokens(ids); }
  }

private:
  std::unique_ptr<MetalGemmaModel> _model;
  std::vector<float>               _logits;
};

}  // namespace vpipe::genai

#endif

#ifndef VPIPE_GENERATIVE_MODELS_LFM3_METAL_LFM3_MODEL_EXEC_H
#define VPIPE_GENERATIVE_MODELS_LFM3_METAL_LFM3_MODEL_EXEC_H

// The ModelExec for LFM3 (MetalLfm3Model): LoadedLanguageModel's contexts
// mapped onto the model's own ContextManager -- prefill, decode, the
// embedding splice a vision tower's rows ride in on (1-D positions), and
// the batched step realtime-vqa decodes its question branches with (one
// forward, the weights read once for every branch).

#include "generative-models/lfm3/metal-lfm3-model.h"
#include "generative-models/model-exec.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace vpipe { class SessionContextIntf; }
namespace vpipe::metal_compute { class MetalCompute; }

namespace vpipe::genai {

class MetalLfm3ModelExec : public ModelExec {
public:
  // `page_tokens` / `max_pages` size the paged K/V (LoadSpec).
  MetalLfm3ModelExec(const std::string&           model_dir,
                     metal_compute::MetalCompute* mc,
                     int page_tokens, std::uint32_t max_pages,
                     const SessionContextIntf*    session);

  bool valid() const noexcept override { return _model != nullptr; }
  bool owns_kv() const noexcept override { return true; }
  std::size_t kv_bytes() const override
  {
    return _model != nullptr ? _model->kv_bytes() : 0;
  }
  int context_seq_len(ContextId ctx) const override;
  bool branch_context(ContextId parent, ContextId child) override;
  void release_context(ContextId ctx) override;

  std::int32_t
  prefill(ContextId ctx, std::span<const std::int32_t> tokens) override;
  std::int32_t decode_one(ContextId ctx, std::int32_t id) override;
  const std::vector<float>& last_logits_host() const override
  {
    return _logits;
  }
  void set_eval_per_layer(bool) noexcept override {}
  // The int8 prefill GEMMs (the stages' i8_prefill; MetalLfm3Model).
  void set_i8_gemm(bool on) override
  {
    if (_model) { _model->set_i8_gemm(on); }
  }
  void set_suppressed_tokens(std::span<const std::int32_t> ids) override
  {
    _suppress.assign(ids.begin(), ids.end());
  }

  // The multimodal splice: text rows from the table, an image's rows
  // from its tower, 1-D positions.
  std::vector<float>
  embed_text_rows(std::span<const std::int32_t> ids) override;
  std::int32_t prefill_embeddings(ContextId ctx,
                                  const std::vector<float>& rows,
                                  int n) override;

  // The reusable branch pool (realtime-vqa's question branches): a
  // pooled child keeps its conv state buffers across scenes and takes
  // its pages from the pool as it appends.
  bool supports_branch_pool() const noexcept override { return true; }
  bool reserve_branch_context(ContextId child, int max_tokens) override;
  bool rebranch_context(ContextId parent, ContextId child) override;
  bool detach_branch_context(ContextId child) override;

  bool supports_batched_decode() const override { return true; }
  bool batched_decode_logits(std::span<const ContextId>    cids,
                             std::span<const std::int32_t> in_tokens,
                             std::span<const std::int32_t> rope_pos,
                             std::vector<float>& out_logits) override;

  MetalLfm3Model* model() { return _model.get(); }
  // The model's context for an exec context (tests), invalid if unknown.
  ContextId model_context(ContextId ctx) const;

private:
  ContextId state_for_(ContextId ctx);
  std::int32_t pick_();            // suppress, then argmax _logits

  std::unique_ptr<MetalLfm3Model> _model;
  std::unordered_map<std::uint32_t, ContextId> _map;
  std::vector<float> _logits;
  std::vector<std::int32_t> _suppress;
};

}  // namespace vpipe::genai

#endif

#ifndef VPIPE_GENERATIVE_MODELS_METAL_QWEN_MODEL_EXEC_H
#define VPIPE_GENERATIVE_MODELS_METAL_QWEN_MODEL_EXEC_H

// MetalQwenModelExec -- ModelExec adapter over the procedural
// metal-compute MetalQwenModel (hybrid Qwen3.5), the Qwen counterpart of
// MetalLlamaModelExec. Token-id-native: drives generation through
// prefill(tokens) / decode_one(id); the MLX-embedding forward_chunk path
// is unused (returns -1). last_logits() wraps the host logits vector.
// Each LoadedLanguageModel ContextId maps to one metal context.

#include "generative-models/qwen3/metal-qwen-model.h"
#include "generative-models/context-io.h"
#include "generative-models/model-exec.h"
#include "generative-models/speculative-decode.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace vpipe { class SessionContextIntf; }
namespace vpipe::metal_compute { class MetalCompute; }

namespace vpipe::genai {

class MetalQwenModelExec : public ModelExec, private SpecDecodeExt,
                           private ContextIoExt {
public:
  MetalQwenModelExec(const std::string&            model_dir,
                     metal_compute::MetalCompute*  mc,
                     const MetalQwenModel::Config& cfg,
                     const SessionContextIntf*     session);

  bool valid() const noexcept override { return _model != nullptr; }
  // A hyper-connection model (Qwen3.8-Flash-Next): its spliced prefills
  // carry the token ids (prefill_hc_spliced) -- the n-gram PLE hashes them.
  bool hc() const noexcept
  {
    return _model != nullptr && _model->config().hc();
  }
  std::int32_t prefill_hc_spliced(ContextId ctx,
                                  const std::vector<float>& emb,
                                  const std::vector<std::int32_t>& ids,
                                  const std::vector<std::int32_t>& pos, int n);

  // The metal exec owns its own KV-page pool keyed by ContextId.
  bool owns_kv() const noexcept override { return true; }

  // This exec keeps its own paged KV; report what it has grown to.
  std::size_t kv_bytes() const override
  {
    auto* cm = _model ? _model->context_manager() : nullptr;
    return cm != nullptr ? cm->resident_bytes() : 0;
  }

  std::int32_t
  prefill(ContextId ctx, std::span<const std::int32_t> tokens) override;
  std::int32_t decode_one(ContextId ctx, std::int32_t id) override;
  const std::vector<float>& last_logits_host() const override { return _logits; }

  // Metal multimodal splice path (Qwen3-ASR audio): host f32 embedding
  // rows + host text-embed gather.
  std::int32_t
  prefill_embeddings(ContextId ctx, const std::vector<float>& rows,
                     int n) override;
  std::int32_t
  prefill_embeddings_mrope(ContextId ctx, const std::vector<float>& rows,
                           const std::vector<std::int32_t>& position_ids,
                           int n) override;
  std::int32_t
  decode_one_at(ContextId ctx, std::int32_t id, int rope_pos) override;
  std::int32_t decode_one_greedy(ContextId ctx, std::int32_t id) override;
  std::int32_t
  decode_one_greedy_at(ContextId ctx, std::int32_t id, int rope_pos) override;
  bool pdecode_begin(ContextId ctx, std::int32_t first_token,
                     std::span<const std::int32_t> prompt,
                     const GpuSamplerParams& sp, int max_tokens,
                     int rope_first) override;
  bool pdecode_commit(ContextId ctx) override;
  std::int32_t pdecode_next(ContextId ctx) override;
  void pdecode_end(ContextId ctx) override;
  bool pdecode_supports_runahead() const override;
  // Batched decode runs the weight-bound matmuls batched over the N branches
  // (like prefill) with per-branch attention/GDN (like decode), so it supports
  // every quant the single-decode forward does: uniform-4bit AND mixed-precision
  // affine (OptiQ) -- encode_batched_step_ de-fuses the mixed projections
  // per-tensor (vqmm_), exactly as the verify/single decode. (Native k-quant is
  // excluded: its q6_K embed gather isn't wired into the affine-embed batched
  // step yet -- a separate follow-on.)
  bool supports_batched_decode() const override
  { return _model != nullptr && !_model->uses_kquant(); }
  bool batched_decode_logits(
      std::span<const ContextId>    cids,
      std::span<const std::int32_t> in_tokens,
      std::span<const std::int32_t> rope_pos,
      std::vector<float>&           out_logits) override;
  bool supports_batched_pipelined_decode() const override
  { return _model != nullptr && !_model->uses_kquant(); }
  bool bdecode_begin(std::span<const ContextId>    cids,
                     std::span<const std::int32_t> first_tokens,
                     const GpuSamplerParams& sp, int max_tokens,
                     std::span<const std::int32_t> rope_pos) override;
  bool bdecode_commit() override;
  bool bdecode_next(std::vector<std::int32_t>& out_tokens) override;
  void bdecode_end() override;
  // Constant-N bdecode never rolls back (a speculative uncollected tail is
  // discarded like any stopped branch's over-advanced KV/GDN), so run-ahead
  // is always safe here.
  bool bdecode_supports_runahead() const override
  { return _model != nullptr; }

  bool supports_mtp() const override
  { return _model != nullptr && _model->has_mtp(); }
  void set_mtp_prefix_seed(bool on) override
  { if (_model != nullptr) { _model->set_mtp_prefix_seed(on); } }
  bool mtp_generate(
      ContextId ctx, std::int32_t first_token, int max_tokens, int rope_first,
      const GpuSamplerParams& sp,
      const std::function<bool(std::int32_t)>&                  is_stop,
      const std::function<bool(std::span<const std::int32_t>)>& on_tokens,
      int* produced, bool* hit_stop) override;
  std::vector<float>
  embed_text_rows(std::span<const std::int32_t> ids) override;

  // Native-f16 zero-copy splice: pre-assembled [n,hidden] f16 buffer.
  std::int32_t
  prefill_embeddings_buf(ContextId ctx, metal_compute::SharedBuffer&& rows,
                         int n) override;
  std::int32_t
  prefill_embeddings_mrope_buf(
      ContextId ctx, metal_compute::SharedBuffer&& rows,
      const std::vector<std::int32_t>& position_ids, int n) override;
  metal_compute::SharedBuffer
  embed_text_buf(std::span<const std::int32_t> ids) override;

  bool branch_context(ContextId parent, ContextId child) override;
  void release_context(ContextId ctx) override;

  // Read-only KV length of a context (0 = never touched).
  int context_seq_len(ContextId ctx) const override;

  bool supports_branch_pool() const noexcept override { return true; }
  bool reserve_branch_context(ContextId child, int max_tokens) override;
  bool rebranch_context(ContextId parent, ContextId child) override;
  bool detach_branch_context(ContextId child) override;

  void set_eval_per_layer(bool) noexcept override {}

  // Dynamic-int8 accelerated prefill GEMMs (LOSSY, opt-in) -- forwards
  // to the metal model (see MetalQwenModel::set_i8_gemm).
  void set_i8_gemm(bool on) override;

  // spec_decode::kExtensionId -> the speculative decode below.
  void* query_extension(std::string_view id) noexcept override;

private:
  // ---- SpecDecodeExt: DFlash when a drafter is attached, else MTP ----
  std::string_view drafter() const noexcept override;
  bool attach_drafter(const std::string& dir, int bits,
                      std::string* err) override;
  bool generate(const SpecDecodeRequest& rq, SpecDecodeResult* out) override;

  // ---- ContextIoExt: a context written out and read back --------------
  bool export_context(ContextId ctx,
                      const ContextManager::ByteSink& out) override;
  bool import_context(ContextId ctx,
                      const ContextManager::ByteSource& in) override;

  struct CtxState { ContextId metal_cid; };
  CtxState* state_for_(ContextId ctx);
  // The metal context behind `ctx`, or an invalid id: looked up under
  // _ctx_mu, for a caller that only needs the id.
  ContextId metal_cid_(ContextId ctx) const;

  std::unique_ptr<MetalQwenModel>             _model;
  // _ctxmap is reached from more than the thread generating on this
  // model: a context parked with a warm model is saved, read back and
  // released on the model manager's threads (its reaper, a launch's
  // planner, a host dropping a conversation) while another context of
  // the same model decodes. Every access holds _ctx_mu. A CtxState is
  // fixed once inserted and the map's nodes never move, so a pointer
  // state_for_ hands out stays valid with the lock dropped; the
  // ContextManager behind it locks its own state.
  mutable std::mutex                          _ctx_mu;
  std::unordered_map<std::uint32_t, CtxState> _ctxmap;
  std::vector<float>                          _logits;
  // For attach_drafter: the drafter's weights go through the session's
  // model manager like the model's own.
  metal_compute::MetalCompute*                _mc = nullptr;
  const SessionContextIntf*                   _session = nullptr;
};

}  // namespace vpipe::genai

#endif

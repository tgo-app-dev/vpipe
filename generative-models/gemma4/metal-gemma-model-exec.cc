#include "generative-models/gemma4/metal-gemma-model-exec.h"

#include <limits>


namespace vpipe::genai {

namespace {
std::int32_t
argmax_(const std::vector<float>& v)
{
  std::int32_t best = 0;
  float bv = v.empty() ? 0.0f : v[0];
  for (std::size_t i = 1; i < v.size(); ++i) {
    if (v[i] > bv) { bv = v[i]; best = (std::int32_t)i; }
  }
  return best;
}
}  // namespace

MetalGemmaModelExec::MetalGemmaModelExec(
    const std::string&             model_dir,
    metal_compute::MetalCompute*   mc,
    const MetalGemmaModel::Config& cfg,
    const SessionContextIntf*      session)
{
  (void)session;
  _model = MetalGemmaModel::load(model_dir, mc, cfg);
}

std::int32_t
MetalGemmaModelExec::prefill(ContextId ctx,
                             std::span<const std::int32_t> tokens)
{
  if (_model == nullptr || tokens.empty()) { return -1; }
  std::vector<std::int32_t> ids(tokens.begin(), tokens.end());
  _logits = _model->prefill(ctx, ids);
  if (_logits.empty()) { return -1; }
  return argmax_(_logits);
}

std::int32_t
MetalGemmaModelExec::prefill_multimodal_ids(
    ContextId ctx, const std::vector<std::int32_t>& ids,
    const std::vector<float>& mm_rows, int n_mm,
    const std::vector<int>& positions)
{
  if (_model == nullptr || ids.empty()) { return -1; }
  _logits = _model->prefill_mm(ctx, ids, mm_rows, n_mm, positions);
  if (_logits.empty()) { return -1; }
  return argmax_(_logits);
}

std::int32_t
MetalGemmaModelExec::decode_one(ContextId ctx, std::int32_t id)
{
  if (_model == nullptr) { return -1; }
  _logits = _model->forward(ctx, id);
  if (_logits.empty()) { return -1; }
  return argmax_(_logits);
}

std::int32_t
MetalGemmaModelExec::decode_one_greedy(ContextId ctx, std::int32_t id)
{
  if (_model == nullptr) { return -1; }
  // On-GPU argmax fast path (no full [vocab] host pull). INT32_MIN means
  // the kernel is unavailable -> fall back to the logit-producing path.
  std::int32_t r = _model->decode_step_fast(ctx, id);
  if (r == std::numeric_limits<std::int32_t>::min()) {
    return decode_one(ctx, id);
  }
  // last_logits_host() is stale after a greedy step (no logits pulled);
  // greedy callers don't read it.
  _logits.clear();
  return r;
}

bool
MetalGemmaModelExec::pdecode_begin(ContextId ctx, std::int32_t first_token,
                                   std::span<const std::int32_t> prompt,
                                   const GpuSamplerParams& sp, int max_tokens,
                                   int rope_first)
{
  return _model != nullptr
      && _model->pdecode_begin(ctx, first_token, prompt, sp, max_tokens,
                               rope_first);
}

bool
MetalGemmaModelExec::pdecode_commit(ContextId ctx)
{
  return _model != nullptr && _model->pdecode_commit(ctx);
}

std::int32_t
MetalGemmaModelExec::pdecode_next(ContextId ctx)
{
  return _model != nullptr ? _model->pdecode_next(ctx) : -1;
}

void
MetalGemmaModelExec::pdecode_end(ContextId ctx)
{
  if (_model != nullptr) { _model->pdecode_end(ctx); }
}

bool
MetalGemmaModelExec::pdecode_supports_runahead() const
{
  // MetalGemmaModel's contiguous KV supports kv_rollback, so pdecode_end
  // discards a run-ahead speculative tail. Paged backends (qwen/llama) do
  // not yet, and inherit the false default.
  return _model != nullptr;
}

bool
MetalGemmaModelExec::branch_context(ContextId parent, ContextId child)
{
  if (_model == nullptr) { return false; }
  return _model->branch_kv(parent, child);
}

void
MetalGemmaModelExec::release_context(ContextId ctx)
{
  if (_model == nullptr) { return; }
  _model->release_kv(ctx);
}


}  // namespace vpipe::genai

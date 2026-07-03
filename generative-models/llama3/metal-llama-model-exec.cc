#include "generative-models/llama3/metal-llama-model-exec.h"


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

MetalLlamaModelExec::MetalLlamaModelExec(
    const std::string&             model_dir,
    metal_compute::MetalCompute*   mc,
    const MetalLlamaModel::Config& cfg,
    const SessionContextIntf*      session)
{
  (void)session;
  _model = MetalLlamaModel::load(model_dir, mc, cfg);
}

MetalLlamaModelExec::CtxState*
MetalLlamaModelExec::state_for_(ContextId ctx)
{
  if (_model == nullptr) {
    return nullptr;
  }
  auto it = _ctxmap.find(ctx.v);
  if (it != _ctxmap.end()) {
    return &it->second;
  }
  ContextId cid = _model->context_manager()->acquire_root();
  if (!cid.valid()) {
    return nullptr;
  }
  CtxState st;
  st.metal_cid = cid;
  return &_ctxmap.emplace(ctx.v, st).first->second;
}

int
MetalLlamaModelExec::context_seq_len(ContextId ctx) const
{
  if (!_model) {
    return 0;
  }
  // Read-only: an unmapped context has no KV yet -- do NOT lazily
  // materialize one just to report its length.
  auto it = _ctxmap.find(ctx.v);
  if (it == _ctxmap.end()) {
    return 0;
  }
  return _model->context_manager()->seq_len_of(it->second.metal_cid);
}

std::int32_t
MetalLlamaModelExec::prefill(ContextId ctx, std::span<const std::int32_t> tokens)
{
  if (tokens.empty()) {
    return -1;
  }
  CtxState* st = state_for_(ctx);
  if (st == nullptr) {
    return -1;
  }
  std::vector<std::int32_t> ids(tokens.begin(), tokens.end());
  _logits = _model->prefill(st->metal_cid, ids);
  if (_logits.empty()) {
    return -1;
  }
  return argmax_(_logits);
}

std::int32_t
MetalLlamaModelExec::decode_one(ContextId ctx, std::int32_t id)
{
  CtxState* st = state_for_(ctx);
  if (st == nullptr) {
    return -1;
  }
  _logits = _model->forward(st->metal_cid, id);
  if (_logits.empty()) {
    return -1;
  }
  return argmax_(_logits);
}

bool
MetalLlamaModelExec::pdecode_begin(ContextId ctx, std::int32_t first_token,
                                   std::span<const std::int32_t> prompt,
                                   const GpuSamplerParams& sp, int max_tokens,
                                   int rope_first)
{
  CtxState* st = state_for_(ctx);
  if (st == nullptr) { return false; }
  return _model->pdecode_begin(st->metal_cid, first_token, prompt, sp,
                               max_tokens, rope_first);
}

bool
MetalLlamaModelExec::pdecode_commit(ContextId ctx)
{
  CtxState* st = state_for_(ctx);
  if (st == nullptr) { return false; }
  return _model->pdecode_commit(st->metal_cid);
}

std::int32_t
MetalLlamaModelExec::pdecode_next(ContextId ctx)
{
  CtxState* st = state_for_(ctx);
  if (st == nullptr) { return -1; }
  return _model->pdecode_next(st->metal_cid);
}

void
MetalLlamaModelExec::pdecode_end(ContextId ctx)
{
  CtxState* st = state_for_(ctx);
  if (st == nullptr) { return; }
  _model->pdecode_end(st->metal_cid);
}

bool
MetalLlamaModelExec::branch_context(ContextId parent, ContextId child)
{
  if (_model == nullptr) {
    return false;
  }
  auto it = _ctxmap.find(parent.v);
  if (it == _ctxmap.end()) {
    return false;
  }
  ContextId metal_child =
      _model->context_manager()->branch(it->second.metal_cid);
  if (!metal_child.valid()) {
    return false;
  }
  CtxState st;
  st.metal_cid = metal_child;
  _ctxmap[child.v] = st;
  return true;
}

void
MetalLlamaModelExec::release_context(ContextId ctx)
{
  if (_model == nullptr) { return; }
  auto it = _ctxmap.find(ctx.v);
  if (it == _ctxmap.end()) { return; }
  // Return this context's KV pages to the pool (refcount drop; shared
  // prefix pages survive until the last brancher releases) and drop the
  // bookkeeping entry so a recycled ContextId re-acquires a fresh root.
  _model->context_manager()->release(it->second.metal_cid);
  _ctxmap.erase(it);
}


}  // namespace vpipe::genai

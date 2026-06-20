#include "generative-models/qwen3/metal-qwen-model-exec.h"

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

MetalQwenModelExec::MetalQwenModelExec(
    const std::string&            model_dir,
    metal_compute::MetalCompute*  mc,
    const MetalQwenModel::Config& cfg,
    const SessionContextIntf*     session)
{
  (void)session;
  _model = MetalQwenModel::load(model_dir, mc, cfg);
}

MetalQwenModelExec::CtxState*
MetalQwenModelExec::state_for_(ContextId ctx)
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

std::int32_t
MetalQwenModelExec::prefill(ContextId ctx, std::span<const std::int32_t> tokens)
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
MetalQwenModelExec::decode_one(ContextId ctx, std::int32_t id)
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

std::int32_t
MetalQwenModelExec::prefill_embeddings(ContextId ctx,
                                       const std::vector<float>& rows, int n)
{
  if (n <= 0 || rows.empty()) {
    return -1;
  }
  CtxState* st = state_for_(ctx);
  if (st == nullptr) {
    return -1;
  }
  _logits = _model->prefill_embeddings(st->metal_cid, rows, n);
  if (_logits.empty()) {
    return -1;
  }
  return argmax_(_logits);
}

std::int32_t
MetalQwenModelExec::prefill_embeddings_mrope(
    ContextId ctx, const std::vector<float>& rows,
    const std::vector<std::int32_t>& position_ids, int n)
{
  if (n <= 0 || rows.empty()) {
    return -1;
  }
  CtxState* st = state_for_(ctx);
  if (st == nullptr) {
    return -1;
  }
  _logits = _model->prefill_multimodal(st->metal_cid, rows, position_ids);
  if (_logits.empty()) {
    return -1;
  }
  return argmax_(_logits);
}

std::int32_t
MetalQwenModelExec::decode_one_at(ContextId ctx, std::int32_t id, int rope_pos)
{
  CtxState* st = state_for_(ctx);
  if (st == nullptr) {
    return -1;
  }
  _logits = _model->forward(st->metal_cid, id, rope_pos);
  if (_logits.empty()) {
    return -1;
  }
  return argmax_(_logits);
}

std::int32_t
MetalQwenModelExec::decode_one_greedy(ContextId ctx, std::int32_t id)
{
  return decode_one_greedy_at(ctx, id, -1);
}

std::int32_t
MetalQwenModelExec::decode_one_greedy_at(ContextId ctx, std::int32_t id,
                                         int rope_pos)
{
  CtxState* st = state_for_(ctx);
  if (st == nullptr) {
    return -1;
  }
  // On-GPU embed + argmax: returns the token id without a host logit pull.
  // INT32_MIN means the fast kernels are unavailable (no state touched) ->
  // fall back to the logit-producing path.
  const std::int32_t r = _model->decode_step_fast(st->metal_cid, id, rope_pos);
  if (r != std::numeric_limits<std::int32_t>::min()) {
    return r;
  }
  _logits = _model->forward(st->metal_cid, id, rope_pos);
  if (_logits.empty()) {
    return -1;
  }
  return argmax_(_logits);
}

bool
MetalQwenModelExec::pdecode_begin(ContextId ctx, std::int32_t first_token,
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
MetalQwenModelExec::batched_decode_logits(
    std::span<const ContextId>    cids,
    std::span<const std::int32_t> in_tokens,
    std::span<const std::int32_t> rope_pos,
    std::vector<float>&           out_logits)
{
  if (_model == nullptr || cids.empty()) { return false; }
  std::vector<ContextId> mcids(cids.size());
  for (std::size_t i = 0; i < cids.size(); ++i) {
    CtxState* st = state_for_(cids[i]);
    if (st == nullptr) { return false; }
    mcids[i] = st->metal_cid;
  }
  return _model->decode_batched_step(
      std::span<const ContextId>(mcids.data(), mcids.size()),
      in_tokens, rope_pos, out_logits);
}

bool
MetalQwenModelExec::bdecode_begin(
    std::span<const ContextId>    cids,
    std::span<const std::int32_t> first_tokens,
    const GpuSamplerParams& sp, int max_tokens,
    std::span<const std::int32_t> rope_pos)
{
  if (_model == nullptr || cids.empty()) { return false; }
  std::vector<ContextId> mcids(cids.size());
  for (std::size_t i = 0; i < cids.size(); ++i) {
    CtxState* st = state_for_(cids[i]);
    if (st == nullptr) { return false; }
    mcids[i] = st->metal_cid;
  }
  return _model->bdecode_begin(
      std::span<const ContextId>(mcids.data(), mcids.size()),
      first_tokens, sp, max_tokens, rope_pos);
}

bool
MetalQwenModelExec::bdecode_commit() { return _model && _model->bdecode_commit(); }

bool
MetalQwenModelExec::bdecode_next(std::vector<std::int32_t>& out_tokens)
{
  return _model && _model->bdecode_next(out_tokens);
}

void
MetalQwenModelExec::bdecode_end() { if (_model) { _model->bdecode_end(); } }

bool
MetalQwenModelExec::mtp_generate(
    ContextId ctx, std::int32_t first_token, int max_tokens, int rope_first,
    const GpuSamplerParams& sp,
    const std::function<bool(std::int32_t)>&                  is_stop,
    const std::function<bool(std::span<const std::int32_t>)>& on_tokens,
    int* produced, bool* hit_stop)
{
  if (produced) { *produced = 0; }
  if (hit_stop) { *hit_stop = false; }
  if (_model == nullptr || !_model->has_mtp()) { return false; }
  CtxState* st = state_for_(ctx);
  if (st == nullptr) { return false; }
  MtpDecodeCtl c;
  c.rope_first = rope_first;
  c.is_stop    = is_stop;
  c.on_round   = on_tokens;
  c.hit_stop   = hit_stop;
  c.sampler    = sp;
  std::vector<std::int32_t> out;
  const bool ok = _model->mtp_decode(st->metal_cid, first_token, max_tokens,
                                     out, /*draft_len=*/1, /*accepted_out=*/
                                     nullptr, /*rounds_out=*/nullptr, c);
  if (produced) { *produced = static_cast<int>(out.size()); }
  return ok;
}

bool
MetalQwenModelExec::pdecode_commit(ContextId ctx)
{
  CtxState* st = state_for_(ctx);
  if (st == nullptr) { return false; }
  return _model->pdecode_commit(st->metal_cid);
}

std::int32_t
MetalQwenModelExec::pdecode_next(ContextId ctx)
{
  CtxState* st = state_for_(ctx);
  if (st == nullptr) { return -1; }
  return _model->pdecode_next(st->metal_cid);
}

void
MetalQwenModelExec::pdecode_end(ContextId ctx)
{
  CtxState* st = state_for_(ctx);
  if (st == nullptr) { return; }
  _model->pdecode_end(st->metal_cid);
}

bool
MetalQwenModelExec::pdecode_supports_runahead() const
{
  // pdecode_end rolls back the speculative tail on BOTH state planes the
  // qwen forward advances: the paged KV (ContextManager::kv_rollback) and the
  // GDN recurrent ssm/conv ring (gdn_ring_rollback + gdn_ring_end). Dense
  // backends (Qwen3-ASR) carry no GDN state -> the ring is inert and the KV
  // rollback alone suffices. Either way run-ahead is safe.
  return _model != nullptr;
}

std::vector<float>
MetalQwenModelExec::embed_text_rows(std::span<const std::int32_t> ids)
{
  if (_model == nullptr || ids.empty()) {
    return {};
  }
  return _model->embed_text(
      std::vector<std::int32_t>(ids.begin(), ids.end()));
}

std::int32_t
MetalQwenModelExec::prefill_embeddings_buf(
    ContextId ctx, metal_compute::SharedBuffer&& rows, int n)
{
  if (_model == nullptr || n <= 0 || rows.empty()) { return -1; }
  CtxState* st = state_for_(ctx);
  if (st == nullptr) { return -1; }
  _logits = _model->prefill_embeddings_buf(st->metal_cid, std::move(rows), n);
  if (_logits.empty()) { return -1; }
  return argmax_(_logits);
}

std::int32_t
MetalQwenModelExec::prefill_embeddings_mrope_buf(
    ContextId ctx, metal_compute::SharedBuffer&& rows,
    const std::vector<std::int32_t>& position_ids, int n)
{
  if (_model == nullptr || n <= 0 || rows.empty()) { return -1; }
  CtxState* st = state_for_(ctx);
  if (st == nullptr) { return -1; }
  _logits = _model->prefill_multimodal_buf(
      st->metal_cid, std::move(rows), position_ids, n);
  if (_logits.empty()) { return -1; }
  return argmax_(_logits);
}

metal_compute::SharedBuffer
MetalQwenModelExec::embed_text_buf(std::span<const std::int32_t> ids)
{
  if (_model == nullptr || ids.empty()) { return {}; }
  return _model->embed_text_buf(
      std::vector<std::int32_t>(ids.begin(), ids.end()));
}

bool
MetalQwenModelExec::branch_context(ContextId parent, ContextId child)
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
MetalQwenModelExec::release_context(ContextId ctx)
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

bool
MetalQwenModelExec::reserve_branch_context(ContextId child, int max_tokens)
{
  if (_model == nullptr) { return false; }
  (void)max_tokens;
  auto* cm = _model->context_manager();
  // Reserve the per-branch SSM/conv + contiguous-KV BUFFERS only (the
  // expensive, churn-to-reallocate part). Do NOT pre-reserve KV pages:
  // pinning ~max_tokens/page_tokens pages PER pooled branch held a large
  // slice of the page pool idle across scenes and starved the scene's
  // base prefill (a big multi-frame describe-prefix would then fail to
  // allocate and the prefill returned -1). Branches now allocate their
  // question/answer pages lazily from the pool and release them each
  // scene (detach_branch / rebranch), so the full pool is available to
  // the base prefill.
  auto ids = cm->reserve_branches(1, /*pages_each=*/0);
  if (ids.empty() || !ids[0].valid()) { return false; }
  CtxState st;
  st.metal_cid = ids[0];
  _ctxmap[child.v] = st;
  return true;
}

bool
MetalQwenModelExec::rebranch_context(ContextId parent, ContextId child)
{
  if (_model == nullptr) { return false; }
  auto pit = _ctxmap.find(parent.v);
  auto cit = _ctxmap.find(child.v);
  if (pit == _ctxmap.end() || cit == _ctxmap.end()) { return false; }
  return _model->context_manager()->rebranch(cit->second.metal_cid,
                                             pit->second.metal_cid);
}

bool
MetalQwenModelExec::detach_branch_context(ContextId child)
{
  if (_model == nullptr) { return false; }
  auto cit = _ctxmap.find(child.v);
  if (cit == _ctxmap.end()) { return false; }
  _model->context_manager()->detach_branch(cit->second.metal_cid);
  return true;
}


}  // namespace vpipe::genai

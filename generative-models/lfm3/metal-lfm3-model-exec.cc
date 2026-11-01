#include "generative-models/lfm3/metal-lfm3-model-exec.h"

#include "generative-models/weight-set.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace vpipe::genai {

MetalLfm3ModelExec::MetalLfm3ModelExec(const std::string&           model_dir,
                                       metal_compute::MetalCompute* mc,
                                       int page_tokens,
                                       std::uint32_t max_pages,
                                       const SessionContextIntf*    session)
{
  MetalLfm3Model::Config cfg;
  std::string err;
  if (MetalLfm3Model::Config::from_dir(model_dir, &cfg, &err)) {
    // The steel prefill reads K/V in 16-key blocks inside a page.
    cfg.page_tokens = page_tokens > 0 ? (page_tokens + 15) / 16 * 16 : 256;
    cfg.max_pages = (int)max_pages;
    _model = MetalLfm3Model::load(open_weight_set(model_dir, session), mc,
                                  cfg, &err);
  }
  if (_model == nullptr && session != nullptr) {
    session->warn(fmt("MetalLfm3ModelExec('{}'): {}", model_dir, err));
  }
}

ContextId
MetalLfm3ModelExec::state_for_(ContextId ctx)
{
  if (_model == nullptr) { return {}; }
  auto it = _map.find(ctx.v);
  if (it != _map.end()) { return it->second; }
  const ContextId cid = _model->context_manager()->acquire_root();
  if (!cid.valid()) { return {}; }
  _map.emplace(ctx.v, cid);
  return cid;
}

ContextId
MetalLfm3ModelExec::model_context(ContextId ctx) const
{
  auto it = _map.find(ctx.v);
  return it != _map.end() ? it->second : ContextId{};
}

int
MetalLfm3ModelExec::context_seq_len(ContextId ctx) const
{
  if (_model == nullptr) { return 0; }
  auto it = _map.find(ctx.v);
  if (it == _map.end()) { return 0; }
  return _model->context_manager()->seq_len_of(it->second);
}

bool
MetalLfm3ModelExec::branch_context(ContextId parent, ContextId child)
{
  if (_model == nullptr) { return false; }
  auto it = _map.find(parent.v);
  if (it == _map.end()) { return false; }
  const ContextId c = _model->context_manager()->branch(it->second);
  if (!c.valid()) { return false; }
  _map[child.v] = c;
  return true;
}

void
MetalLfm3ModelExec::release_context(ContextId ctx)
{
  if (_model == nullptr) { return; }
  auto it = _map.find(ctx.v);
  if (it == _map.end()) { return; }
  // The context's pages back to the pool (a shared prefix survives its
  // last holder), its conv state with it.
  _model->context_manager()->release(it->second);
  _map.erase(it);
}

bool
MetalLfm3ModelExec::reserve_branch_context(ContextId child, int max_tokens)
{
  (void)max_tokens;
  if (_model == nullptr) { return false; }
  // Buffers only, no pages held (they would sit idle across scenes and
  // starve the next scene's prefill): a branch's pages come from the
  // pool as it appends and go back at the detach.
  auto ids = _model->context_manager()->reserve_branches(1, 0);
  if (ids.empty() || !ids[0].valid()) { return false; }
  _map[child.v] = ids[0];
  return true;
}

bool
MetalLfm3ModelExec::rebranch_context(ContextId parent, ContextId child)
{
  if (_model == nullptr) { return false; }
  const ContextId p = model_context(parent), c = model_context(child);
  if (!p.valid() || !c.valid()) { return false; }
  return _model->context_manager()->rebranch(c, p);
}

bool
MetalLfm3ModelExec::detach_branch_context(ContextId child)
{
  if (_model == nullptr) { return false; }
  const ContextId c = model_context(child);
  if (!c.valid()) { return false; }
  _model->context_manager()->detach_branch(c);
  return true;
}

std::int32_t
MetalLfm3ModelExec::pick_()
{
  const int V = _model->config().vocab;
  if ((int)_logits.size() < V) { return -1; }
  for (std::int32_t id : _suppress) {
    if (id >= 0 && id < V) {
      _logits[(std::size_t)id] = -std::numeric_limits<float>::infinity();
    }
  }
  std::int32_t best = 0;
  float bv = _logits[0];
  for (int i = 1; i < V; ++i) {
    if (_logits[(std::size_t)i] > bv) {
      bv = _logits[(std::size_t)i];
      best = i;
    }
  }
  return best;
}

std::int32_t
MetalLfm3ModelExec::prefill(ContextId ctx,
                            std::span<const std::int32_t> tokens)
{
  if (tokens.empty()) { return -1; }
  const ContextId cid = state_for_(ctx);
  if (!cid.valid() || !_model->prefill(cid, tokens, nullptr, &_logits)) {
    return -1;
  }
  return pick_();
}

std::int32_t
MetalLfm3ModelExec::decode_one(ContextId ctx, std::int32_t id)
{
  const ContextId cid = state_for_(ctx);
  if (!cid.valid()) { return -1; }
  const MetalLfm3Model::Segment s{cid, 0, 1};
  if (!_model->forward(std::span<const MetalLfm3Model::Segment>(&s, 1),
                       std::span<const std::int32_t>(&id, 1), nullptr, 0,
                       &_logits)) {
    return -1;
  }
  return pick_();
}

std::vector<float>
MetalLfm3ModelExec::embed_text_rows(std::span<const std::int32_t> ids)
{
  return _model != nullptr ? _model->embed_rows(ids) : std::vector<float>{};
}

std::int32_t
MetalLfm3ModelExec::prefill_embeddings(ContextId ctx,
                                       const std::vector<float>& rows,
                                       int n)
{
  if (_model == nullptr || n <= 0
      || rows.size() < (std::size_t)n * _model->config().hidden) {
    return -1;
  }
  const ContextId cid = state_for_(ctx);
  if (!cid.valid()) { return -1; }
  bool ok;
  if (rows.size() == (std::size_t)n * _model->config().hidden) {
    ok = _model->prefill(cid, {}, &rows, &_logits);
  } else {
    const std::vector<float> r(
        rows.begin(), rows.begin() + (long)n * _model->config().hidden);
    ok = _model->prefill(cid, {}, &r, &_logits);
  }
  return ok ? pick_() : -1;
}

bool
MetalLfm3ModelExec::batched_decode_logits(
    std::span<const ContextId>    cids,
    std::span<const std::int32_t> in_tokens,
    std::span<const std::int32_t> rope_pos,
    std::vector<float>&           out_logits)
{
  (void)rope_pos;   // 1-D positions: a context's next slot
  if (_model == nullptr || cids.empty()
      || cids.size() != in_tokens.size()) {
    return false;
  }
  // Eight branches a forward (the few-row GEMV's rows); more in turns.
  const int V = _model->config().vocab;
  out_logits.resize(cids.size() * (std::size_t)V);
  std::vector<float> part;
  for (std::size_t at = 0; at < cids.size(); at += 8) {
    const std::size_t k = std::min<std::size_t>(8, cids.size() - at);
    std::vector<MetalLfm3Model::Segment> segs;
    for (std::size_t i = 0; i < k; ++i) {
      const ContextId c = state_for_(cids[at + i]);
      if (!c.valid()) { return false; }
      segs.push_back({c, (int)i, 1});
    }
    if (!_model->forward(segs, in_tokens.subspan(at, k), nullptr, 0,
                         &part)) {
      return false;
    }
    std::copy(part.begin(), part.end(),
              out_logits.begin() + (long)(at * (std::size_t)V));
  }
  for (std::size_t i = 0; i < cids.size(); ++i) {
    for (std::int32_t id : _suppress) {
      if (id >= 0 && id < V) {
        out_logits[i * (std::size_t)V + (std::size_t)id] =
            -std::numeric_limits<float>::infinity();
      }
    }
  }
  return true;
}

}  // namespace vpipe::genai

#include "generative-models/shared/metal-sol-attention.h"

#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace vpipe {
namespace genai {

using metal_compute::ComputeEncoder;
using metal_compute::FunctionConstants;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

// The two flash kernels' tiles for head_dim 128, from their own
// instantiations: the ALU steel entry is bq 32 / bk 16, the matrix-core
// one bq 64 / bk 32. The CSR is in the exact kernel's key blocks and
// the routing is decided at its query block, so choosing the kernel
// chooses both.
constexpr int kSteelBQ = 32;
constexpr int kSteelBK = 16;
constexpr int kNaxBQ   = 64;
constexpr int kNaxBK   = 32;
constexpr int kHeadDim = 128;
// The approximate half tiles 32 rows whichever kernel takes the exact
// one; see sol_approx_mma's BQR.
constexpr int kApproxBQ = 32;
// sol_proxy_mma's tile and simdgroup count; see its header for why they
// are not the projection GEMM's 128.
// One carve's alignment: a page, so a buffer keeps the slack a private
// allocation used to give it.
constexpr std::size_t kCarveAlign = 16384;
constexpr int kProxyTile = 64;
constexpr unsigned kProxySg = 4;

// The layout AttnParams has, restated because the vendored header is a
// Metal one. Kept in one place so the field order cannot drift.
struct AttnP {
  int B, H, D, qL, kL, gqa;
  float scale;
  int NQ, NK, NQa, NKa, qL_rem, kL_rem, qL_off;
  std::int64_t Qs[3], Ks[3], Vs[3], Os[3];
};

}  // namespace

std::unique_ptr<MetalSolAttention>
MetalSolAttention::load(MetalCompute* mc, bool bf16, std::string* err)
{
  auto fail = [&](const char* m) {
    if (err != nullptr) { *err = m; }
    return std::unique_ptr<MetalSolAttention>();
  };
  if (mc == nullptr) { return fail("no metal-compute backend"); }
  auto s = std::unique_ptr<MetalSolAttention>(new MetalSolAttention());
  s->_mc = mc;
  s->_bf16 = bf16;
  s->_lib = mc->load_library(bf16 ? "sol_attn_mma_bf16" : "sol_attn_mma");
  // ONE library, two entry points: attn_steel carries both the f16
  // and the bf16 instantiation, so the dtype picks the FUNCTION, not
  // the library.
  s->_lib_steel = mc->load_library("attn_steel");
  // THE MATRIX-CORE FLASH KERNEL FOR THE EXACT HALF, where there are
  // matrix cores. It is the same kernel with the same span + export_ml
  // contract, on the hardware units -- which is where the routed cost
  // lives, so it is the one part of this method whose port is worth
  // more than all the others together.
  //
  // VPIPE_SOL_NO_NAX takes the ALU kernel instead. Not only a
  // stopwatch: the two route at different query blocks, so it is also
  // how the coarser decision is compared against the finer one.
  //
  // READ PER LOAD, not once per process: a static would latch whichever
  // arm ran first, and the two are compared inside one test binary.
  // load() happens once per model, so there is nothing to save.
  //
  // AND THE APPROXIMATE HALF GOES TO THE FLASH KERNEL ON BOTH ARMS.
  // That half is an attention over the summary sequence with the
  // routing's flags as a per-(query block, key) mask, and there is
  // nothing matrix-core about saying so -- the mask is in the SHARED
  // steel header, so the ALU kernel carries it too.
  // VPIPE_SOL_NO_MASKED_APPROX keeps sol_approx_mma, which is what the
  // two are compared against.
  s->_masked = std::getenv("VPIPE_SOL_NO_MASKED_APPROX") == nullptr;
  const bool no_nax = std::getenv("VPIPE_SOL_NO_NAX") != nullptr;
  if (mc->supports_matrix_cores() && !no_nax) {
    s->_lib_nax = mc->load_library("attn_steel_nax");
    s->_nax = s->_lib_nax.valid();
  }
  s->_bq = s->_nax ? kNaxBQ : kSteelBQ;
  s->_bk = s->_nax ? kNaxBK : kSteelBK;
  // AND THE ROUTING PROXY AS A GEMM, on the same hardware condition.
  // Optional in the strong sense: if the library or the entry point is
  // missing the fused routing kernel still runs, and the answer is the
  // same one.
  //
  // VPIPE_SOL_NO_PROXY_GEMM keeps the fused routing kernel on a
  // matrix-core box, which is the A/B: the two must keep the SAME
  // blocks, not merely produce answers of similar quality.
  const bool no_proxy = std::getenv("VPIPE_SOL_NO_PROXY_GEMM") != nullptr;
  if (s->_nax && !no_proxy) {
    s->_lib_proxy =
        mc->load_library(bf16 ? "sol_proxy_mma_bf16" : "sol_proxy_mma");
    if (s->_lib_proxy.valid()) {
      s->_fn_proxy = s->_lib_proxy.function("sol_proxy_mma");
      s->_fn_route_p = s->_lib.function("sol_route_p_mma");
    }
  }
  s->_fn_sum    = s->_lib.function("sol_summaries_mma");
  s->_fn_stats  = s->_lib.function("sol_kc_stats_mma");
  s->_fn_route  = s->_lib.function("sol_route_mma");
  s->_fn_scan   = s->_lib.function("sol_scan_mma");
  s->_fn_emit   = s->_lib.function("sol_emit_mma");
  s->_fn_approx = s->_lib.function("sol_approx_mma");
  s->_fn_merge  = s->_lib.function("sol_merge_mma");
  s->_fn_merge_ml = s->_lib.function("sol_merge_ml_mma");
  // REFUSED, not warned. An unvalidated ComputeFunction dispatches as a
  // no-op, so a missing kernel here is attention that returns whatever
  // the output buffer held -- downstream, a plausible frame.
  if (!s->_fn_sum.valid() || !s->_fn_stats.valid() || !s->_fn_route.valid() ||
      !s->_fn_scan.valid() || !s->_fn_emit.valid() ||
      !s->_fn_approx.valid() || !s->_fn_merge.valid() ||
      !s->_fn_merge_ml.valid()) {
    return fail("sol_attn_mma kernels did not validate");
  }
  if (!s->_lib_steel.valid()) {
    return fail("sol_attn needs the steel attention library");
  }

  return s;
}

// THE BUFFER LIST AS SIZES, in ensure_scratch_'s own order.
//
// One list with two readers -- the total and the carve simulation --
// rather than two arithmetics that have to agree. The pairing with
// ensure_scratch_ itself is still a mirror, and is held honest by a
// test that reserves and compares EXACTLY; see
// sol_attention_mma.the_scratch_estimate_matches_the_allocation. Before
// that test existed this arithmetic was a bound rather than a figure,
// and it was also uncalled, so nothing noticed that it counted an fp32
// approximate output the matrix-core path does not allocate and missed
// the two buffers that path does.
namespace {

void
sol_scratch_plan_(std::size_t H, std::size_t KVH, std::size_t T,
                  std::size_t D, std::size_t nq, std::size_t nk,
                  std::size_t tp, std::size_t bk, bool nax,
                  std::vector<std::size_t>* out)
{
  out->clear();
  // WHICH HEAD COUNT EACH BUFFER CARRIES, and it is not one of them.
  // The routing is decided per QUERY head -- flags, kept, the CSR, both
  // partial outputs -- while the key centroids and their statistics
  // summarise K and V, which under GQA have FEWER heads. Sizing the
  // latter by H is what a caller gets away with at GQA 1 and what
  // silently reads past the end of K/V at anything else.
  out->push_back(H * nq * D * 2);                    // qc   (query heads)
  out->push_back(KVH * nk * D * 2);                  // kc   (kv heads)
  out->push_back(KVH * nk * D * 2);                  // vc   (kv heads)
  out->push_back(KVH * D * 4);                       // mean (kv heads)
  out->push_back(KVH * D * 4);                       // var  (kv heads)
  out->push_back(H * nq * nk);                       // flags
  if (nax) { out->push_back(H * nq * nk * 4); }      // proxy
  out->push_back(H * nq * 4);                        // kept
  out->push_back(H * (nq + 1) * 4);                  // qb_off
  // The block LIST is worst-cased: one entry per key block of the exact
  // kernel per query block per head, because nothing bounds what the
  // routing keeps until it has run.
  out->push_back(H * nq * (T / bk + 1) * 4);         // qb_blk
  out->push_back(H * T * 4);                         // m_a
  out->push_back(H * T * 4);                         // l_a
  out->push_back(H * T * D * 2);                     // o_e
  // The approximate half's output, which BOTH arms now take from the
  // flash kernel: the tensor dtype and no padded tail. The fp32
  // [H, TPAD, D] one sol_approx_mma wants is twice this and is
  // allocated only under VPIPE_SOL_NO_MASKED_APPROX -- a bench arm, so
  // the estimate describes the DEFAULT and says so, rather than
  // carrying an env read into a static. `tp` is what that buffer would
  // have been sized by, and is why the parameter is still here.
  out->push_back(H * T * D * 2);                     // o_an
  (void)tp;
  out->push_back(H * T * 4);                         // m_e
  out->push_back(H * T * 4);                         // l_e
}

// The geometry the plan is written against, or false when the key block
// does not divide the exact kernel's.
bool
sol_geometry_(int heads, int kv_heads, int tokens, int d, int key_block,
              bool nax, std::size_t* H, std::size_t* KVH, std::size_t* T,
              std::size_t* D, std::size_t* nq, std::size_t* nk,
              std::size_t* tp, std::size_t* bk)
{
  if (heads <= 0 || tokens <= 0 || d <= 0) { return false; }
  // GQA, refused here rather than mis-indexed later. The group factor
  // has to divide exactly: every query head belongs to one KV head, and
  // a ragged last group would have no key centroids to read.
  if (kv_heads <= 0 || kv_heads > heads || heads % kv_heads != 0) {
    return false;
  }
  const int bq = nax ? kNaxBQ : kSteelBQ;
  const int kb = nax ? kNaxBK : kSteelBK;
  const int blk = key_block > 0 ? key_block : sol::kBlock;
  if (blk % kb != 0) { return false; }
  *H = (std::size_t)heads;
  *KVH = (std::size_t)kv_heads;
  *T = (std::size_t)tokens;
  *D = (std::size_t)d;
  *nq = (std::size_t)((tokens + bq - 1) / bq);
  *nk = (std::size_t)((tokens + blk - 1) / blk);
  *tp = (std::size_t)(((tokens + kApproxBQ - 1) / kApproxBQ) * kApproxBQ);
  *bk = (std::size_t)kb;
  return true;
}

}  // namespace

std::size_t
MetalSolAttention::scratch_bytes(int heads, int tokens, int d, int key_block,
                                 bool nax)
{
  return scratch_bytes(heads, heads, tokens, d, key_block, nax);
}

std::size_t
MetalSolAttention::scratch_bytes(int heads, int kv_heads, int tokens, int d,
                                 int key_block, bool nax)
{
  std::size_t H, KVH, T, D, nq, nk, tp, bk;
  if (!sol_geometry_(heads, kv_heads, tokens, d, key_block, nax, &H, &KVH,
                     &T, &D, &nq, &nk, &tp, &bk)) {
    return 0;
  }
  std::vector<std::size_t> plan;
  sol_scratch_plan_(H, KVH, T, D, nq, nk, tp, bk, nax, &plan);
  std::size_t n = 0;
  for (std::size_t b : plan) { n += b; }
  return n;
}

std::size_t
MetalSolAttention::private_bytes(int heads, int tokens, int d, int key_block,
                                 bool nax, std::size_t lend_a,
                                 std::size_t lend_b)
{
  return private_bytes(heads, heads, tokens, d, key_block, nax, lend_a,
                       lend_b);
}

std::size_t
MetalSolAttention::private_bytes(int heads, int kv_heads, int tokens, int d,
                                 int key_block, bool nax, std::size_t lend_a,
                                 std::size_t lend_b)
{
  std::size_t H, KVH, T, D, nq, nk, tp, bk;
  if (!sol_geometry_(heads, kv_heads, tokens, d, key_block, nax, &H, &KVH,
                     &T, &D, &nq, &nk, &tp, &bk)) {
    return 0;
  }
  std::vector<std::size_t> plan;
  sol_scratch_plan_(H, KVH, T, D, nq, nk, tp, bk, nax, &plan);
  // THE SAME GREEDY CARVE ensure_scratch_ runs, alignment included --
  // a subtraction would be a lower bound, and a planner wants the other
  // one. Two cursors, first fit, fall through to private.
  const std::size_t cap[2] = {lend_a, lend_b};
  std::size_t used[2] = {0, 0};
  std::size_t priv = pinned_bytes();
  for (std::size_t b : plan) {
    bool placed = false;
    for (int i = 0; i < 2 && !placed; ++i) {
      if (cap[i] == 0) { continue; }
      const std::size_t off = (used[i] + kCarveAlign - 1) & ~(kCarveAlign - 1);
      if (off < cap[i] && b <= cap[i] - off) {
        used[i] = off + b;
        placed = true;
      }
    }
    if (!placed) { priv += b; }
  }
  return priv;
}

std::vector<SharedBuffer*>
MetalSolAttention::scratch_buffers_()
{
  // Everything ensure_scratch_ hands out, in ONE list, so counting them
  // and giving them back cannot disagree about which they are. The
  // pinned four are NOT here: they are not carved and not released.
  return {&_qc, &_kc, &_vc, &_mean, &_var, &_flags, &_proxy, &_kept,
          &_qb_off, &_qb_blk, &_o_a, &_m_a, &_l_a, &_o_e, &_o_an, &_m_e,
          &_l_e};
}

MetalSolAttention::~MetalSolAttention()
{
  drop_residency_();
}

void
MetalSolAttention::drop_residency_()
{
  if (!_res_added || _mc == nullptr) { return; }
  _res_added = false;
  // The SAME list the add walked, pinned four included -- they are this
  // object's own and go back with the rest. Removing an allocation the
  // set does not hold is a no-op, which is what makes it safe to walk a
  // list whose carved entries may share one parent: the first remove
  // takes the parent out and the rest find nothing.
  for (SharedBuffer* b : scratch_buffers_()) {
    if (!b->empty()) { _mc->residency_remove(*b); }
  }
  for (const SharedBuffer* b : {&_counts, &_params, &_sp_params,
                                &_sp_bounds, &_params_a}) {
    if (!b->empty()) { _mc->residency_remove(*b); }
  }
  // Adds and removes both take effect at the commit, not at the call.
  _mc->residency_commit();
}

std::size_t
MetalSolAttention::pinned_bytes()
{
  return 2 * sizeof(AttnP) + 5 * sizeof(int) + 2 * sizeof(int) + 8;
}

void
MetalSolAttention::set_sage(const MetalSageAttention* sage)
{
  // MATRIX CORES ONLY, and silently so is not an option: a caller that
  // asked for both and got the f16 exact half would report a Sage run.
  // The ALU steel kernel has no int8 branch -- there is no fragment MMA
  // under it to buy -- so the honest answer on that arm is `false`, and
  // sage_engaged() is what the caller reports instead of its own config.
  _sage = (sage != nullptr && _nax) ? sage : nullptr;
}

void
MetalSolAttention::set_arena(const SharedBuffer& a, const SharedBuffer& b)
{
  const bool same = a.contents() == _arena_a.contents() &&
                    a.byte_size() == _arena_a.byte_size() &&
                    b.contents() == _arena_b.contents() &&
                    b.byte_size() == _arena_b.byte_size();
  if (same) { return; }
  // A SharedBuffer is move-only, so what is kept is a window over the
  // whole of each -- which also makes the lender's lifetime explicit:
  // this object holds a reference, and a lender about to replace what
  // it lent has to lend nothing first or it will be holding two.
  auto hold = [](const SharedBuffer& x) {
    return x.byte_size() > 0 ? x.subview(0, x.byte_size()) : SharedBuffer{};
  };
  _arena_a = hold(a);
  _arena_b = hold(b);
  // AND DROP EVERY WINDOW THE LAST RESERVATION HANDED OUT. Not only the
  // geometry flag: each carved buffer is a window holding the lender's
  // allocation alive, so keeping them would keep WORKING -- off in
  // memory the lender has already replaced -- and would pay for it
  // twice until the next call rebuilt them.
  // BEFORE the windows go, because removing needs the handles: what the
  // set is holding is the OLD arena, and a lender calls this precisely
  // when it is about to replace it.
  drop_residency_();
  for (SharedBuffer* p : scratch_buffers_()) { *p = SharedBuffer{}; }
  _heads = _tokens = _d = _nq = _nk = 0;
  _arena_used[0] = 0;
  _arena_used[1] = 0;
}

// Fill the two attention param blocks for THIS call's band. Separated
// from ensure_scratch_ because the scratch is cached on the extents and
// the band is not: same shape, different offset, different answer.
void
MetalSolAttention::set_band_params_(int heads, int kv_heads,
                                    const Band& band, int d)
{
  const int q_tokens = band.q_tokens, k_tokens = band.k_tokens;
  const int q_rows = band.q_rows, k_rows = band.k_rows;
  const int q_off = band.q_off;
  const int nq = _nq, nk = _nk;
  if (_params.empty()) { return; }
  auto* p = static_cast<AttnP*>(_params.contents());
  p->B = 1; p->H = heads; p->D = d; p->qL = q_tokens; p->kL = k_tokens;
  // The exact half is steel, which walks K/V with this factor. It was
  // pinned at 1 because every caller was MHA.
  p->gqa = heads / kv_heads;
  p->NQ = nq; p->NK = (k_tokens + _bk - 1) / _bk;
  p->NQa = q_tokens / _bq; p->NKa = k_tokens / _bk;
  p->qL_rem = q_tokens - p->NQa * _bq;
  p->kL_rem = k_tokens - p->NKa * _bk;
  // THE QUERY'S POSITION IN THE SEQUENCE, which the causal predicate
  // measures a row against. Zero for a whole-sequence call.
  p->qL_off = q_off;
  // Head-major, but with the BAND's stride: q and the output step by
  // the rows their buffer holds, k and v by theirs, and the two are
  // only the same number when the band is the whole sequence.
  const std::int64_t qm[3] = {(std::int64_t)heads * q_rows * d,
                              (std::int64_t)q_rows * d, d};
  const std::int64_t km[3] = {(std::int64_t)kv_heads * k_rows * d,
                              (std::int64_t)k_rows * d, d};
  // BOTH FLASH HALVES WRITE SOL'S OWN partials, which are dense
  // [H, q_tokens, D] whatever the band is. Only the merge touches the
  // caller's buffer, and it takes that stride separately. Giving the
  // output the band's stride here would scatter the partials.
  const std::int64_t om[3] = {(std::int64_t)heads * q_tokens * d,
                              (std::int64_t)q_tokens * d, d};
  for (int i = 0; i < 3; ++i) {
    p->Qs[i] = qm[i]; p->Ks[i] = km[i]; p->Vs[i] = km[i]; p->Os[i] = om[i];
  }
  if (_masked) {
    // The APPROXIMATE half's geometry: the same queries against the
    // SUMMARY sequence, so qL is the rows and kL is the block count.
    // `NQ` is what the block mask divides by -- it is the routing's
    // query-block count and not a key-block one, which is exactly what
    // the exact half already carries in the same field.
    auto* pa = static_cast<AttnP*>(_params_a.contents());
    *pa = *p;
    // The summary sequence has no position of its own, so the routing's
    // flags are the only mask and the causal offset does not apply.
    pa->qL_off = 0;
    pa->kL = nk;
    pa->NK = (nk + _bk - 1) / _bk;
    pa->NKa = nk / _bk;
    pa->kL_rem = nk - pa->NKa * _bk;
    // The summary sequence is the approximate half's K/V, so its strides
    // are the KV head count's -- and `pa->gqa` (copied from `p`) is what
    // makes that kernel read the right group.
    const std::int64_t sm[3] = {(std::int64_t)kv_heads * nk * d,
                                (std::int64_t)nk * d, d};
    for (int i = 0; i < 3; ++i) { pa->Ks[i] = sm[i]; pa->Vs[i] = sm[i]; }

  }
}

bool
MetalSolAttention::ensure_scratch_(int heads, int kv_heads, int q_tokens,
                                   int k_tokens, int d, std::string* err)
{
  const int blk = _blk;
  // QUERY blocks from the query extent and KEY blocks from the key one.
  // They were one number because every caller was square.
  const int nq = (q_tokens + _bq - 1) / _bq;
  const int nk = (k_tokens + blk - 1) / blk;
  // Everything below that follows the QUERIES -- the partial softmaxes,
  // the two partial outputs, the CSR's row count -- is sized by this,
  // and everything that follows the KEYS by `k_tokens`.
  const int tokens = q_tokens;
  // `_kv_heads` joins the geometry tag: the key summaries are sized by
  // it, so a caller that changed only the group factor would otherwise
  // keep buffers shaped for the previous one.
  if (_heads == heads && _kv_heads == kv_heads && _tokens == tokens &&
      _k_tokens == k_tokens && _d == d && _nk == nk && _nq == nq &&
      !_qc.empty()) {
    return true;
  }
  // A REBUILD, so the previous geometry's buffers are about to be
  // overwritten. Hand them back first; the assignments below are the
  // last references to them.
  drop_residency_();
  const std::size_t H = (std::size_t)heads, D = (std::size_t)d;
  const std::size_t KVH = (std::size_t)kv_heads;
  // WHERE THE SCRATCH COMES FROM. Every buffer below is written and read
  // inside ONE encode -- the summaries, the routing, the CSR and the two
  // partial outputs all die at the merge -- so any memory the caller is
  // not using for the length of the call will do, and this carves from
  // what it lent (set_arena) before allocating its own.
  //
  // The exceptions are the ones the HOST writes and the GPU reads on
  // every encode -- the AttnParams pair, the span params and their
  // bounds -- plus the routing counter the host reads back. Those have
  // to survive between calls, so they stay this object's own. A few
  // hundred bytes; see pinned_bytes().
  //
  // TWO REGIONS, EACH WITH ITS OWN CURSOR, because what a caller has
  // spare is not usually one run: the DiT lends its fused qkv
  // projection AND the attention window past the one it writes into,
  // which are separate allocations. A buffer that fits neither falls
  // back to its own allocation, per buffer -- so a short lend strands
  // the tail rather than failing.
  //
  // Page-aligned, for the reason the VDN branch's carve is: a private
  // allocation is page-rounded, so each of these used to have slack
  // behind it, and a kernel writing a padded tile past its
  // destination's exact size landed there rather than in a neighbour.
  _arena_used[0] = 0;
  _arena_used[1] = 0;
  auto mk = [&](std::size_t n) -> SharedBuffer {
    SharedBuffer* reg[2] = {&_arena_a, &_arena_b};
    for (int i = 0; i < 2; ++i) {
      if (reg[i]->empty()) { continue; }
      const std::size_t off =
          (_arena_used[i] + kCarveAlign - 1) & ~(kCarveAlign - 1);
      if (off < reg[i]->byte_size() && n <= reg[i]->byte_size() - off) {
        _arena_used[i] = off + n;
        return reg[i]->subview(off, n);
      }
    }
    return _mc->make_shared_buffer(n);
  };
  // The approximate half's own tiling, which is 32 rows whatever the
  // exact half's block is -- so o_a's padded height follows THAT and
  // not the routing block.
  _nqa = (tokens + kApproxBQ - 1) / kApproxBQ;
  _tpad = _nqa * kApproxBQ;
  _qc = mk(H * (std::size_t)nq * D * 2);
  // PER KV HEAD, and the mirror of sol_scratch_plan_ above. K and V have
  // kv_heads of them; sizing these by the query count is what read past
  // the end of the summaries on the first GQA caller.
  _kc = mk(KVH * (std::size_t)nk * D * 2);
  _vc = mk(KVH * (std::size_t)nk * D * 2);
  _mean = mk(KVH * D * 4);
  _var  = mk(KVH * D * 4);
  _flags = mk(H * (std::size_t)nq * (std::size_t)nk);
  if (_fn_proxy.valid() && _fn_route_p.valid()) {
    _proxy = mk(H * (std::size_t)nq * (std::size_t)nk * 4);
    if (_proxy.empty()) {
      if (err != nullptr) { *err = "sol_attn scratch allocation failed"; }
      return false;
    }
  }
  _kept  = mk(H * (std::size_t)nq * 4);
  _qb_off = mk(H * (std::size_t)(nq + 1) * 4);
  _qb_blk = mk(H * (std::size_t)nq *
               (std::size_t)(k_tokens / _bk + 1) * 4);
  // THE fp32 [H, TPAD, D] APPROXIMATE OUTPUT IS sol_approx_mma'S ALONE.
  // The flash approximate half -- which is both arms now -- stores
  // normalised and in the tensor dtype into `_o_an`, half the bytes and
  // no padded tail, and never touches this. Allocating it anyway cost
  // 427 MB of the 861 MB Sol held at 56 heads x 14861 rows, for a
  // buffer nothing on that path reads.
  if (!_masked) { _o_a = mk(H * (std::size_t)_tpad * D * 4); }
  else          { _o_a = SharedBuffer{}; }
  _m_a = mk(H * (std::size_t)tokens * 4);
  _l_a = mk(H * (std::size_t)tokens * 4);
  _o_e = mk(H * (std::size_t)tokens * D * 2);
  if (_masked) {
    // The flash approximate half stores in the tensor dtype and needs no
    // padded tail, so it is HALF the bytes of the fp32 [H, TPAD, D] the
    // simdgroup kernel wants.
    _o_an = mk(H * (std::size_t)tokens * D * 2);
    _params_a = _mc->make_shared_buffer(sizeof(AttnP));   // pinned
    if (_o_an.empty() || _params_a.empty()) {
      if (err != nullptr) { *err = "sol_attn scratch allocation failed"; }
      return false;
    }
  }
  _m_e = mk(H * (std::size_t)tokens * 4);
  _l_e = mk(H * (std::size_t)tokens * 4);
  // ZEROED AT BIRTH. The counter is an atomic the route kernel ADDS to,
  // and it is read at the start of the NEXT forward -- so whatever the
  // allocator handed over is reported as the first forward's routing.
  // In the model that read as "kept 1037318338 of 57960 key blocks
  // exact (1789714.2%)", which is loud; a plausible-looking wrong
  // percentage would have been worse.
  // PRIVATE, not carved. It is the one buffer here that must SURVIVE
  // between calls -- the host reads the routing back from it -- so it is
  // deliberately not rebuilt, and a window into the caller's arena is
  // therefore exactly the wrong home for it: it would still be pointing
  // into an arena the lender had since replaced, and it would hold that
  // arena alive for the life of this object. pinned_bytes() has always
  // counted these 8 bytes as our own; this is what makes that true.
  if (_counts.empty()) {
    _counts = _mc->make_shared_buffer(8);
    if (!_counts.empty()) {
      std::memset(_counts.contents(), 0, _counts.byte_size());
    }
  }
  // PINNED: written by the host here (and `scale` per encode) and read
  // by the GPU on every one, so they cannot live in memory the caller
  // reuses between calls.
  _params = _mc->make_shared_buffer(sizeof(AttnP));
  _sp_params = _mc->make_shared_buffer(5 * sizeof(int));
  _sp_bounds = _mc->make_shared_buffer(2 * sizeof(int));
  // The approximate half's output is ONE of the two, never both: which
  // one is the arm's, so checking the wrong one refuses a scratch that
  // is complete.
  const bool approx_out_ok = _masked ? !_o_an.empty() : !_o_a.empty();
  if (_qc.empty() || _kc.empty() || _vc.empty() || _flags.empty() ||
      _qb_blk.empty() || !approx_out_ok || _o_e.empty() ||
      _params.empty() || _sp_params.empty()) {
    if (err != nullptr) { *err = "sol_attn scratch allocation failed"; }
    return false;
  }

  // THE PARAMS ARE PER CALL, not per scratch. The allocation above is
  // keyed on the extents, but a band with the same extents can sit at a
  // different offset -- so a cache hit must still refresh them. See
  // set_band_params_, which encode() calls after this returns.
  int* sp = static_cast<int*>(_sp_params.contents());
  // tokens_per_frame 0 switches steel's span-EDGE predicate off. The Sol
  // list is exact at BK granularity -- one routing block is a whole
  // number of steel blocks -- so there are no partial blocks to mask,
  // and that predicate is VDN's, not ours.
  sp[0] = 0; sp[1] = 0; sp[2] = 0; sp[3] = 0;
  sp[4] = nq + 1;            // qb_off is [H][NQ + 1]: routing is per HEAD
  std::memset(_sp_bounds.contents(), 0, _sp_bounds.byte_size());

  // INTO THE RESIDENCY SET, and this is not an optimisation.
  //
  // ~1 GB of scratch at production geometry, created long after the
  // model committed its own residency. A caller that commits MANY
  // command buffers per forward -- which the DiT does, between its
  // profile splits and the encoder's own auto-split every 50 dispatches
  // -- then pays to make these buffers resident on each one. MEASURED on
  // MiniMax-H3: 36 s of attention a block against 0.49 s for the same
  // geometry, dtype and routing in a single-commit harness. Nothing
  // about the kernels differed; only how often the command buffer
  // turned over.
  //
  // ...AND EVERY ADD IS OWED A REMOVE. The set retains what it holds and
  // outlives this object, so the walk below is the same one
  // drop_residency_() runs in reverse -- written as the same two lists
  // rather than a third hand-copied spelling, because the failure of a
  // list that drifts is a multi-gigabyte arena that never comes back and
  // nothing that reports it.
  for (const SharedBuffer* b : scratch_buffers_()) {
    if (!b->empty()) { _mc->residency_add(*b); }
  }
  for (const SharedBuffer* b : {&_counts, &_params, &_sp_params,
                                &_sp_bounds, &_params_a}) {
    if (!b->empty()) { _mc->residency_add(*b); }
  }
  _res_added = true;
  _mc->residency_commit();

  // Allocation is announced, because it is ~1 GB at production geometry
  // and a cache MISS per block would be the whole cost of the method.
  if (_mc->session() != nullptr) {
    _mc->session()->log_normal(fmt(
        "MetalSolAttention: scratch (re)allocated for {} heads x {} rows, "
        "block {} -> nq {} nk {}, {} MB; both halves on {} (bq {} bk {}), "
        "approximate half {}",
        heads, tokens, blk, nq, nk, resident_bytes() >> 20,
        _nax ? "attn_steel_nax" : "attn_steel", _bq, _bk,
        _masked ? "block-masked" : "sol_approx_mma"));
  }
  _heads = heads; _kv_heads = kv_heads; _tokens = tokens;
  _k_tokens = k_tokens; _d = d;
  _nq = nq; _nk = nk;
  // In STEEL key blocks: that is what the route kernel counts, so a
  // fraction against routing blocks would read above 100%.
  _total_blocks = (long long)heads * nq *
                  ((tokens + _bk - 1) / _bk);
  return true;
}

metal_compute::ComputeFunction
MetalSolAttention::_lib_for_width_(
    const metal_compute::FunctionConstants& fc) const
{
  // ONE place that spells the entry point, because there are now four of
  // them (two kernels x two widths) and both halves need the same one.
  const bool d64 = (_d == 64);
  if (_nax) {
    return _lib_nax.function(
        d64 ? (_bf16 ? "attn_steel_nax_h_bd64_bf16" : "attn_steel_nax_h_bd64")
            : (_bf16 ? "attn_steel_nax_h_bd128_bf16"
                     : "attn_steel_nax_h_bd128"),
        fc);
  }
  return _lib_steel.function(
      d64 ? (_bf16 ? "attn_steel_h_bd64_bf16" : "attn_steel_h_bd64")
          : (_bf16 ? "attn_steel_h_bd128_bf16" : "attn_steel_h_bd128"),
      fc);
}

bool
MetalSolAttention::ensure_steel_(int tokens, std::string* err)
{
  // KEYED ON THE SUMMARY LENGTH TOO. align_K for the approximate half is
  // a property of `nk`, which moves with the key block at a fixed
  // sequence -- so a bench sweeping the block at one length would keep
  // the first specialisation for every later one.
  // THE SAGE FLAG IS PART OF THE KEY. It changes the exact function and
  // nothing else about the geometry does, so without it an A/B in one
  // process keeps whichever arm specialised first.
  const int want_sage = (_sage != nullptr) ? 1 : 0;
  // ...AND ON THE HEAD WIDTH, which now selects the entry point. Two
  // widths at one sequence length is not a shape anything runs today,
  // but the cache would hand the second one the first one's kernel and
  // the failure is a silent read past the shorter rows.
  if (_steel_seq == tokens && _steel_nk == _nk && _steel_sage == want_sage
      && _steel_d == _d && _fn_steel.valid()) {
    return true;
  }
  FunctionConstants fc;
  fc.set_bool(200, (tokens % 64) == 0).set_bool(201, (tokens % 32) == 0)
      .set_bool(300, false).set_bool(301, false).set_bool(302, false)
      .set_bool(303, true).set_bool(304, true)
      .set_bool(sage::kQkInt8Constant, want_sage != 0);
  _fn_steel = _lib_for_width_(fc);
  if (!_fn_steel.valid()) {
    if (err != nullptr) {
      *err = "the block-sparse steel attention did not specialise";
    }
    return false;
  }
  if (_masked) {
    // The approximate half on the same kernel: no spans, the routing's
    // flag array as a per-(query block, key) mask, and its own softmax
    // statistics out for the merge.
    //
    // align_K is asked of the SUMMARY length and at the coarser of the
    // two key blocks, which is conservative on the ALU kernel (16, not
    // 32) and therefore right: an unnecessary length mask costs the
    // ragged block a predicate, where a missing one reads past kL.
    FunctionConstants fa;
    fa.set_bool(200, (tokens % 64) == 0).set_bool(201, (_nk % 32) == 0)
        .set_bool(300, false).set_bool(301, false).set_bool(302, false)
        .set_bool(303, false).set_bool(304, true).set_bool(305, true);
    _fn_approx_masked = _lib_for_width_(fa);
    if (!_fn_approx_masked.valid()) {
      if (err != nullptr) {
        *err = "the block-masked flash attention did not specialise";
      }
      return false;
    }
  }
  _steel_seq = tokens;
  _steel_nk = _nk;
  _steel_sage = want_sage;
  _steel_d = _d;
  return true;
}

bool
MetalSolAttention::encode(ComputeEncoder& enc, const SharedBuffer& q,
                          const SharedBuffer& k, const SharedBuffer& v,
                          SharedBuffer& out, int heads, int tokens, int d,
                          float scale, const sol::Config& cfg,
                          std::string* err)
{
  return encode(enc, q, k, v, out, heads, heads, tokens, d, scale, cfg, err);
}

bool
MetalSolAttention::encode(ComputeEncoder& enc, const SharedBuffer& q,
                          const SharedBuffer& k, const SharedBuffer& v,
                          SharedBuffer& out, int heads, int kv_heads,
                          int tokens, int d, float scale,
                          const sol::Config& cfg, std::string* err)
{
  // The square case IS the banded one with the band spanning the whole
  // sequence, so it is expressed that way rather than duplicated.
  const Band b{tokens, tokens, tokens, tokens, 0, 0};
  return encode(enc, q, k, v, out, heads, kv_heads, b, d, scale, cfg, err);
}

// The banded form. `q`/`out` cover `b.q_rows` rows per head and this
// call attends `b.q_tokens` of them starting at `b.q_off`; `k`/`v` cover
// `b.k_rows` and this call reads `b.k_tokens` from the start.
//
// WHY IT EXISTS. A block-causal layout does not have one attention --
// it has one dispatch per segment, each a row band of a buffer holding
// the whole joint sequence, and each RECTANGULAR (a target image block
// attends itself plus everything before it, so qL < kL). Sol's method
// does not care: the routing is a proxy score of query-block centroids
// against key-block centroids and neither has to be the same sequence.
// What cared was the plumbing -- three kernels reused the token count
// as a head stride, and the local band assumed the query's row index
// equalled the key's.
bool
MetalSolAttention::encode(ComputeEncoder& enc, const SharedBuffer& q,
                          const SharedBuffer& k, const SharedBuffer& v,
                          SharedBuffer& out, int heads, int kv_heads,
                          const Band& band, int d, float scale,
                          const sol::Config& cfg, std::string* err)
{
  auto fail = [&](const char* m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  const int q_tokens = band.q_tokens, k_tokens = band.k_tokens;
  const int q_rows = band.q_rows, k_rows = band.k_rows;
  const int q_off = band.q_off;
  // `tokens` still names the KEY extent, which is what the sink window,
  // the steel key-block count and the scratch that follows the keys are
  // all measured in.
  const int tokens = k_tokens;
  if (heads <= 0 || q_tokens <= 0 || k_tokens <= 0 || d <= 0) {
    return fail("empty geometry");
  }
  if (q_rows < q_tokens || k_rows < k_tokens || q_off < 0) {
    return fail("sol_attn: a band must fit inside its buffer");
  }
  // GQA, REFUSED RATHER THAN MIS-INDEXED. Every query head has to belong
  // to exactly one KV head: a ragged last group would read key centroids
  // that were never summarised, which is a wrong answer and not a crash.
  if (kv_heads <= 0 || kv_heads > heads || heads % kv_heads != 0) {
    return fail("sol_attn: kv_heads must be positive and divide heads");
  }
  const int gqa = heads / kv_heads;
  // The simdgroup approximate kernel indexes the summary sequence by the
  // QUERY head, so under GQA it would read another group's centroids.
  // It is the bench A/B rather than the default path, so it is refused
  // here instead of being taught the group factor.
  if (gqa > 1 && !_masked) {
    return fail("sol_attn: VPIPE_SOL_NO_MASKED_APPROX is MHA only -- "
                "sol_approx_mma indexes the summaries by the query head");
  }
  // HEAD DIM 64 OR 128, which is every width the steel entries are
  // instantiated at. Both flash kernels tile these IDENTICALLY -- ALU
  // 32/16, NAX 64/32 -- so nothing about the routing, the CSR or the
  // block statistics moves with the width; only the entry point's name
  // and the summaries' row length do, and those are already parameters.
  //
  // It used to assert 128 because 128 was all there was. The one piece
  // that really is specialised is sol_approx_mma, whose register-tiled
  // output is built around SOL_D -- and that kernel is no longer on the
  // default path (the block-masked flash half replaced it on both arms),
  // so a width it cannot serve is refused only when a caller asks for it
  // back with VPIPE_SOL_NO_MASKED_APPROX.
  if (d != 64 && d != kHeadDim) {
    return fail("sol_attn needs head_dim 64 or 128");
  }
  if (!_masked && d != kHeadDim) {
    return fail("VPIPE_SOL_NO_MASKED_APPROX is head_dim 128 only -- "
                "sol_approx_mma's output tile is built around it");
  }
  if (cfg.thresh != sol::Threshold::kDiag) {
    return fail("sol_attn on metal implements the 'diag' threshold only");
  }
  _blk = cfg.key_block > 0 ? cfg.key_block : sol::kBlock;
  if (_blk % _bk != 0 || _blk < _bk) {
    return fail("the Sol key block must be a whole number of the exact "
                "kernel's key blocks");
  }
  if (!ensure_scratch_(heads, kv_heads, q_tokens, k_tokens, d, err)) {
    return false;
  }
  set_band_params_(heads, kv_heads, band, d);
  if (!ensure_steel_(tokens, err)) { return false; }
  // WHAT ACTUALLY HAPPENED, not what was asked: set_sage already
  // dropped the driver on a box with no matrix cores, so this is the
  // one place that knows, and sage_engaged() is what a caller should
  // report and a bench should key its arms on.
  _sage_on = (_sage != nullptr) && _steel_sage == 1;
  // THE SCALE IS PER CALL AND THE SCRATCH IS NOT, so it is written here
  // rather than where the rest of AttnParams is filled.
  //
  // It was written NOWHERE, which is what a zero-initialised buffer
  // makes of a missing assignment: the exact half ran every routed
  // block at scale 0, i.e. a uniform average of the values it kept. The
  // model's own tests could not see it -- a uniform attention is finite,
  // reproducible, and certainly "changes the answer" -- and the kernel
  // tests could not either, their rig filling the field itself. Caught
  // by driving THIS class at tau = -inf against dense; see
  // sol_attention_mma.the_class_reproduces_dense.
  static_cast<AttnP*>(_params.contents())->scale = scale;
  if (!_params_a.empty()) {
    static_cast<AttnP*>(_params_a.contents())->scale = scale;
  }

  const int nq = _nq, nk = _nk, per = _blk / _bk;
  const int nks = (tokens + _bk - 1) / _bk;
  // VPIPE_SOL_SKIP is a bitmask of passes to leave out, for bisecting a
  // cost that only appears at one call site. 1 summaries, 2 stats,
  // 4 route, 8 scan, 16 emit, 32 approx, 64 steel, 128 merge. The result
  // is nonsense with anything set; this is a stopwatch, not a mode.
  static const int skip = []() {
    const char* e = std::getenv("VPIPE_SOL_SKIP");
    return (e != nullptr) ? std::atoi(e) : 0;
  }();
  int sink_lo = 0, sink_hi = 0;
  if (cfg.sink_tokens > 0) {
    const int s0 = std::max(0, cfg.sink_start);
    const int s1 = std::min(tokens, s0 + cfg.sink_tokens);
    if (s1 > s0) {
      sink_lo = s0 / _blk;
      sink_hi = (s1 + _blk - 1) / _blk;
    }
  }

  // 1. summaries. Twice, because q is summarised at the QUERY block size
  //    and k/v at the KEY block size: routing has to be decided at
  //    steel's own query block or two of them would share one CSR entry.
  //    EACH CALL ASKS FOR ONLY WHAT IT WANTS -- 2|4 then 1 -- because
  //    the block size is per call and the destination count with it.
  const int kSumQ = 1, kSumKV = 6;
  // BYTE offset of the band's first query row. Q's head stride is the
  // buffer's row count, so basing it here places head h row r at
  // q_off + h*q_rows + r -- the band, read in place.
  const std::size_t q_boff =
      (std::size_t)band.q_row0 * (std::size_t)d * 2;
  if ((skip & 1) == 0) {
  enc.set_function(_fn_sum);
  enc.set_buffer(0, q); enc.set_buffer(1, k); enc.set_buffer(2, v);
  enc.set_buffer(3, _qc); enc.set_buffer(4, _kc); enc.set_buffer(5, _vc);
  enc.set_constant(6, k_tokens); enc.set_constant(7, d);
  enc.set_constant(8, nk); enc.set_constant(9, _blk);
  enc.set_constant(10, kSumKV);
  enc.set_constant(11, k_rows);   // head stride: k/v may be a band
  // OVER KV HEADS. The kernel takes its head from the grid's y, so the
  // k/v pass simply runs `kv_heads` of them and writes [KVH, nk, D] --
  // no kernel change, and the q pass below is untouched at `heads`.
  enc.dispatch({(unsigned)d, (unsigned)kv_heads, (unsigned)nk},
               {(unsigned)d, 1, 1});
  enc.set_function(_fn_sum);
  enc.set_buffer(0, q, q_boff); enc.set_buffer(1, q, q_boff);
  enc.set_buffer(2, q, q_boff);
  enc.set_buffer(3, _qc); enc.set_buffer(4, _qc); enc.set_buffer(5, _qc);
  enc.set_constant(6, q_tokens); enc.set_constant(7, d);
  enc.set_constant(8, nq); enc.set_constant(9, _bq);
  enc.set_constant(10, kSumQ);
  enc.set_constant(11, q_rows);   // head stride: q may be a band
  enc.dispatch({(unsigned)d, (unsigned)heads, (unsigned)nq},
               {(unsigned)d, 1, 1});
  }

  // 2. the key centroids' spread, per head.
  if ((skip & 2) == 0) {
  enc.set_function(_fn_stats);
  enc.set_buffer(0, _kc); enc.set_buffer(1, _mean); enc.set_buffer(2, _var);
  enc.set_constant(3, d); enc.set_constant(4, nk);
  // The spread of the KEY centroids, so per KV head like the centroids.
  enc.dispatch({(unsigned)d, (unsigned)kv_heads, 1}, {(unsigned)d, 1, 1});
  }

  // 3-5. route, scan, emit -- the CSR the steel kernel walks.
  //
  // THE PROXY IS A GEMM where there are matrix cores for it: one
  // [NQ, D] x [D, NK] per head instead of NQ * NK simdgroup dot
  // products, each of which re-read a key centroid. MEASURED at 8 heads
  // x 20036 rows: 2.7 ms of the routing became 0.6.
  const bool proxy_gemm = !_proxy.empty() && _fn_proxy.valid() &&
                          _fn_route_p.valid();
  if ((skip & 4) == 0 && proxy_gemm) {
  enc.set_function(_fn_proxy);
  enc.set_buffer(0, _qc); enc.set_buffer(1, _kc); enc.set_buffer(2, _proxy);
  enc.set_constant(3, nq); enc.set_constant(4, nk); enc.set_constant(5, d);
  enc.set_constant(6, gqa);
  const unsigned mt = (unsigned)((nq + kProxyTile - 1) / kProxyTile);
  const unsigned nt = (unsigned)((nk + kProxyTile - 1) / kProxyTile);
  enc.dispatch({32u * kProxySg * nt, mt, (unsigned)heads},
               {32u * kProxySg, 1, 1});
  enc.set_function(_fn_route_p);
  enc.set_buffer(0, _proxy); enc.set_buffer(1, _qc);
  enc.set_buffer(2, _mean); enc.set_buffer(3, _var);
  enc.set_buffer(4, _flags); enc.set_buffer(5, _kept);
  enc.set_buffer(6, _counts);
  enc.set_constant(7, scale); enc.set_constant(8, cfg.tau);
  enc.set_constant(9, d); enc.set_constant(10, nk); enc.set_constant(11, nq);
  enc.set_constant(12, _blk); enc.set_constant(13, _bq);
  enc.set_constant(14, cfg.local_radius);
  enc.set_constant(15, sink_lo); enc.set_constant(16, sink_hi);
  enc.set_constant(17, per);
  enc.set_constant(18, nks);
  enc.set_constant(19, gqa);
  enc.set_constant(20, q_off);
  enc.dispatch({32, (unsigned)heads, (unsigned)nq}, {32, 1, 1});
  } else if ((skip & 4) == 0) {
  enc.set_function(_fn_route);
  enc.set_buffer(0, _qc); enc.set_buffer(1, _kc); enc.set_buffer(2, _mean);
  enc.set_buffer(3, _var); enc.set_buffer(4, _flags); enc.set_buffer(5, _kept);
  enc.set_buffer(6, _counts);
  enc.set_constant(7, scale); enc.set_constant(8, cfg.tau);
  enc.set_constant(9, d); enc.set_constant(10, nk); enc.set_constant(11, nq);
  enc.set_constant(12, _blk); enc.set_constant(13, _bq);
  enc.set_constant(14, cfg.local_radius);
  enc.set_constant(15, sink_lo); enc.set_constant(16, sink_hi);
  enc.set_constant(17, per);
  enc.set_constant(18, nks);
  enc.set_constant(19, gqa);
  enc.set_constant(20, q_off);
  enc.dispatch({32, (unsigned)heads, (unsigned)nq}, {32, 1, 1});
  }

  if ((skip & 8) == 0) {
  enc.set_function(_fn_scan);
  enc.set_buffer(0, _kept); enc.set_buffer(1, _qb_off);
  enc.set_constant(2, nq); enc.set_constant(3, heads);
  enc.set_constant(4, per);
  enc.dispatch({256, 1, 1}, {256, 1, 1});
  }

  if ((skip & 16) == 0) {
  enc.set_function(_fn_emit);
  enc.set_buffer(0, _flags); enc.set_buffer(1, _qb_off);
  enc.set_buffer(2, _qb_blk);
  enc.set_constant(3, nq); enc.set_constant(4, nk); enc.set_constant(5, per);
  enc.set_constant(6, nks);
  enc.dispatch({32, (unsigned)heads, (unsigned)nq}, {32, 1, 1});
  }

  // 6. the approximate half, over the summary sequence.
  //
  // ON THE FLASH KERNEL where there is a matrix-core one: this half IS
  // an attention over the summaries, and the only thing that made it
  // special was excluding the blocks the exact half already took --
  // which the flag array says directly, as a per-(query block, key)
  // mask. MEASURED at 8 heads x 20036 rows: 17.7 ms of a 52.5 ms call
  // on the simdgroup kernel, at 1/64 of the dense work.
  if ((skip & 32) == 0 && _masked) {
  enc.set_function(_fn_approx_masked);
  enc.set_buffer(0, q, q_boff); enc.set_buffer(1, _kc); enc.set_buffer(2, _vc);
  enc.set_buffer(3, _o_an); enc.set_buffer(4, _params_a);
  enc.set_buffer(12, _m_a); enc.set_buffer(13, _l_a);
  enc.set_buffer(14, _flags);
  enc.dispatch({32u * (unsigned)nq, 4u * (unsigned)heads, 1}, {32, 4, 1});
  } else if ((skip & 32) == 0) {
  enc.set_function(_fn_approx);
  enc.set_buffer(0, q, q_boff); enc.set_buffer(1, _kc); enc.set_buffer(2, _vc);
  enc.set_buffer(3, _flags); enc.set_buffer(4, _o_a); enc.set_buffer(5, _m_a);
  enc.set_buffer(6, _l_a);
  enc.set_constant(7, scale); enc.set_constant(8, tokens);
  enc.set_constant(9, d); enc.set_constant(10, nk); enc.set_constant(11, nq);
  enc.set_constant(12, _blk); enc.set_constant(13, _tpad);
  enc.set_constant(14, _bq);
  // y is 4 * heads because the threadgroup is 2-D and tid.y must come
  // out as the head; the grid is in THREADS. z is this kernel's OWN
  // 32-row tiles, which is not the routing block when the exact half
  // runs on the matrix cores -- see sol_approx_mma's BQR.
  enc.dispatch({32, 4u * (unsigned)heads, (unsigned)_nqa}, {32, 4, 1});
  }

  // 7. the exact half, on steel, walking only the listed blocks.
  if ((skip & 64) == 0) {
  enc.set_function(_fn_steel);
  enc.set_buffer(0, q, q_boff); enc.set_buffer(1, k); enc.set_buffer(2, v);
  enc.set_buffer(3, _o_e); enc.set_buffer(4, _params);
  enc.set_buffer(8, _qb_off); enc.set_buffer(9, _qb_blk);
  enc.set_buffer(10, _sp_params); enc.set_buffer(11, _sp_bounds);
  enc.set_buffer(12, _m_e); enc.set_buffer(13, _l_e);
  // The int8 operands the caller's prologue filled, at 15..18. The
  // function was built with constant 306 above or these slots are
  // unused, so the two cannot disagree: both follow `_sage`.
  if (_sage != nullptr) { _sage->bind(enc); }
  enc.dispatch({32u * (unsigned)nq, 4u * (unsigned)heads, 1}, {32, 4, 1});
  }

  // 8. merge. Which one follows from which kernel produced the
  //    approximate half: the flash one stores normalised and in the
  //    tensor dtype, and its +log2(BLK) is applied here.
  if ((skip & 128) == 0 && _masked) {
  const float bonus = std::log2((float)_blk);
  enc.set_function(_fn_merge_ml);
  enc.set_buffer(0, _o_e); enc.set_buffer(1, _m_e); enc.set_buffer(2, _l_e);
  enc.set_buffer(3, _o_an); enc.set_buffer(4, _m_a); enc.set_buffer(5, _l_a);
  enc.set_buffer(6, out, q_boff);
  enc.set_constant(7, q_tokens); enc.set_constant(8, d);
  enc.set_constant(9, bonus);
  enc.set_constant(10, q_rows);   // destination stride: out may be a band
  enc.dispatch({(unsigned)d, (unsigned)heads, (unsigned)q_tokens},
               {(unsigned)d, 1, 1});
  } else if ((skip & 128) == 0) {
  enc.set_function(_fn_merge);
  enc.set_buffer(0, _o_e); enc.set_buffer(1, _m_e); enc.set_buffer(2, _l_e);
  enc.set_buffer(3, _o_a); enc.set_buffer(4, _m_a); enc.set_buffer(5, _l_a);
  enc.set_buffer(6, out, q_boff);
  enc.set_constant(7, q_tokens); enc.set_constant(8, d);
  enc.set_constant(9, _tpad);
  enc.set_constant(10, q_rows);   // destination stride: out may be a band
  enc.dispatch({(unsigned)d, (unsigned)heads, (unsigned)q_tokens},
               {(unsigned)d, 1, 1});
  }
  return true;
}

long long
MetalSolAttention::exact_blocks() const
{
  if (_counts.empty()) { return 0; }
  const auto* c = static_cast<const std::uint32_t*>(_counts.contents());
  return (long long)c[0];
}

void
MetalSolAttention::reset_counts()
{
  if (_counts.empty()) { return; }
  std::memset(_counts.contents(), 0, _counts.byte_size());
}

std::size_t
MetalSolAttention::resident_bytes() const
{
  // OWNED ONLY. A carved buffer reports its own byte_size, so counting
  // one would book an allocation the lender made -- and once everything
  // is lent the honest answer here is pinned_bytes().
  std::size_t n = 0;
  for (SharedBuffer* b :
       const_cast<MetalSolAttention*>(this)->scratch_buffers_()) {
    if (b->is_owned()) { n += b->byte_size(); }
  }
  return n;
}

}  // namespace genai
}  // namespace vpipe

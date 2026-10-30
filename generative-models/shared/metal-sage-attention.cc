#include "generative-models/shared/metal-sage-attention.h"

#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <cstdlib>
#include <vector>

namespace vpipe {
namespace genai {

using metal_compute::ComputeEncoder;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

// The steel NAX flash kernel's tiles at head_dim 128. Not this file's
// choice: they come from attn_steel_nax.metal, and the scale arrays are
// indexed by the kernel with them.
constexpr int kNaxBQ = 64;
constexpr int kNaxBK = 32;

// What sage_quant_block's threadgroup staging tile bounds. Asserted
// rather than assumed, because exceeding either is a write past a
// threadgroup array -- which is not a crash, it is a wrong answer.
constexpr int kMaxBlock = 64;
constexpr int kMaxD     = 128;

// Page-aligned carving, for the reason MetalSolAttention's is: a
// private allocation is page-rounded, so each buffer used to have slack
// behind it, and a kernel writing a padded tile past its destination's
// exact size landed there rather than in a neighbour.
constexpr std::size_t kCarveAlign = 16384;

}  // namespace

int MetalSageAttention::nax_query_block() { return kNaxBQ; }
int MetalSageAttention::nax_key_block() { return kNaxBK; }

bool
MetalSageAttention::available(const MetalCompute* mc)
{
  if (mc == nullptr || !mc->valid()) { return false; }
  // MATRIX CORES ARE THE WHOLE CONDITION. The int8 fragment MMA is the
  // instruction being bought; on a box without one there is no int8
  // attention to fall back to -- the ALU steel kernel has no such
  // branch -- so this is a hardware question, not a preference.
  if (!mc->supports_matrix_cores()) { return false; }
  return std::getenv("VPIPE_NO_SAGE_ATTN") == nullptr;
}

std::unique_ptr<MetalSageAttention>
MetalSageAttention::load(MetalCompute* mc, bool bf16, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return std::unique_ptr<MetalSageAttention>{};
  };
  if (mc == nullptr) { return fail("no metal-compute backend"); }
  if (!available(mc)) {
    return fail("sage_attn needs matrix cores (M5 or later)");
  }
  auto s = std::unique_ptr<MetalSageAttention>(new MetalSageAttention());
  s->_mc = mc;
  s->_bf16 = bf16;
  s->_lib = mc->load_library(bf16 ? "sage_quant_bf16" : "sage_quant");
  if (!s->_lib.valid()) { return fail("sage_quant library unavailable"); }
  s->_fn_sum    = s->_lib.function("sage_k_sum_chunk");
  s->_fn_reduce = s->_lib.function("sage_k_mean_reduce");
  s->_fn_quant  = s->_lib.function("sage_quant_block");
  // REFUSED, not warned. An unvalidated ComputeFunction dispatches as a
  // no-op, so a missing quantize kernel here is an attention reading
  // whatever the int8 buffers happened to hold -- which is not a
  // degraded picture, it is noise, and the kernel would report nothing.
  if (!s->_fn_sum.valid() || !s->_fn_reduce.valid() ||
      !s->_fn_quant.valid()) {
    return fail("sage_quant kernels did not validate");
  }
  // The int8 P*V's four are OPTIONAL (the contract lists them so): a
  // plugin's copy of this class may run on a host older than they are,
  // and there the int8 QK alone is the answer, not a refused model.
  s->_fn_qtok   = s->_lib.function("sage_quant_token");
  s->_fn_vam    = s->_lib.function("sage_v_absmax_chunk");
  s->_fn_var    = s->_lib.function("sage_v_absmax_reduce");
  s->_fn_vq8    = s->_lib.function("sage_quant_v_channel_u8");
  s->_pv8_kernels = s->_fn_qtok.valid() && s->_fn_vam.valid() &&
                    s->_fn_var.valid() && s->_fn_vq8.valid();
  return s;
}

std::unique_ptr<MetalSageAttention>
MetalSageAttention::load_for_model(MetalCompute* mc, bool bf16,
                                   const sage::Config& cfg, const char* who,
                                   bool* fatal)
{
  if (fatal != nullptr) { *fatal = false; }
  if (!cfg.enabled) { return {}; }
  auto* sess = mc != nullptr ? mc->session() : nullptr;
  if (!available(mc)) {
    if (sess != nullptr) {
      sess->log_normal(fmt(
          "{}: sage_attn asked for but this GPU has no matrix cores -- "
          "attention stays dense", who));
    }
    return {};
  }
  std::string err;
  std::unique_ptr<MetalSageAttention> s = load(mc, bf16, &err);
  if (!s) {
    if (fatal != nullptr) { *fatal = true; }
    if (sess != nullptr) {
      sess->log_normal(fmt("{}: sage_attn requested but {}", who, err));
    }
    return {};
  }
  if (sess != nullptr) {
    sess->log_normal(fmt(
        "{}: SageAttention ON -- INT8 QK^T{}, layers {}+, key smoothing {}",
        who, s->pv8_for(cfg) ? " and P*V (held, uint8 P)" : "",
        cfg.dense_layers, cfg.smooth_k ? "on" : "OFF (A/B only)"));
    // Said, because the config asked: a host whose library predates the
    // int8 P*V runs the int8 QK alone, and the run must not read as one
    // with it.
    if (cfg.pv8 && !s->pv8_available()) {
      sess->log_normal(fmt(
          "{}: sage_pv8 asked for but this host's kernels predate it -- "
          "the int8 QK alone", who));
    }
  }
  return s;
}

MetalSageAttention::~MetalSageAttention()
{
  drop_residency_();
}

void
MetalSageAttention::drop_residency_()
{
  if (!_res_added || _mc == nullptr) { return; }
  _res_added = false;
  for (SharedBuffer* b : scratch_buffers_()) {
    if (!b->empty()) { _mc->residency_remove(*b); }
  }
  _mc->residency_commit();
}

std::vector<SharedBuffer*>
MetalSageAttention::scratch_buffers_()
{
  // Everything ensure_scratch_ hands out, in ONE list, so counting them,
  // announcing them and giving them back cannot disagree about which
  // they are.
  return {&_km, &_kmp, &_q8, &_k8, &_qs, &_ks, &_v8, &_vs, &_vm, &_vmp};
}

namespace {

// THE BUFFER LIST AS SIZES, in ensure_scratch_'s own order, padded the
// way the carve pads. One list with two readers -- the total and the
// carve simulation -- rather than two arithmetics that have to agree.
bool
sage_plan_(int heads, int kv_heads, int q_tokens, int k_tokens, int d,
           int bq, int bk, bool pv8, std::vector<std::size_t>* out)
{
  out->clear();
  if (heads <= 0 || kv_heads <= 0 || q_tokens <= 0 || k_tokens <= 0 ||
      d <= 0 || bq <= 0 || bk <= 0) {
    return false;
  }
  const std::size_t H = (std::size_t)heads, D = (std::size_t)d;
  const std::size_t KH = (std::size_t)kv_heads;
  const std::size_t nq = (std::size_t)sage::query_blocks(q_tokens, bq);
  const std::size_t nk = (std::size_t)sage::key_blocks(k_tokens, bk);
  // Padded per buffer, because the carve aligns each one: a bare sum is
  // short by up to one alignment apiece, which strands the tail in
  // private allocations and reads as the lend not working.
  auto pad = [](std::size_t n) {
    return (n + kCarveAlign - 1) & ~(kCarveAlign - 1);
  };
  out->push_back(pad(KH * D * 4));                                    // km
  out->push_back(pad(KH * (std::size_t)sage::kMeanChunks * D * 4));   // kmp
  out->push_back(pad(H * (std::size_t)q_tokens * D));                 // q8
  out->push_back(pad(KH * (std::size_t)k_tokens * D));                // k8
  out->push_back(pad(H * (pv8 ? (std::size_t)q_tokens : nq) * 4));  // qs
  out->push_back(pad(KH * nk * 4));                                   // ks
  if (pv8) {
    out->push_back(pad(KH * (std::size_t)k_tokens * D));              // v8
    out->push_back(pad(KH * D * 4));                                  // vs
    out->push_back(pad(KH * D * 4));                                  // vm
    out->push_back(pad(KH * (std::size_t)sage::kMeanChunks * D * 4)); // vmp
  }
  return true;
}

}  // namespace

std::size_t
MetalSageAttention::scratch_bytes(int heads, int kv_heads, int q_tokens,
                                  int k_tokens, int d, int bq, int bk,
                                  bool pv8)
{
  std::vector<std::size_t> plan;
  if (!sage_plan_(heads, kv_heads, q_tokens, k_tokens, d, bq, bk, pv8,
                  &plan)) {
    return 0;
  }
  std::size_t n = 0;
  for (std::size_t b : plan) { n += b; }
  return n;
}

std::size_t
MetalSageAttention::private_bytes(int heads, int kv_heads, int q_tokens,
                                  int k_tokens, int d, int bq, int bk,
                                  std::size_t lend_a, std::size_t lend_b,
                                  bool pv8)
{
  std::vector<std::size_t> plan;
  if (!sage_plan_(heads, kv_heads, q_tokens, k_tokens, d, bq, bk, pv8,
                  &plan)) {
    return 0;
  }
  // THE SAME GREEDY CARVE ensure_scratch_ runs: first region that has
  // room takes it, and anything that fits neither is the caller's own.
  std::size_t used[2] = {0, 0};
  const std::size_t cap[2] = {lend_a, lend_b};
  std::size_t own = 0;
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
    if (!placed) { own += b; }
  }
  return own;
}

void
MetalSageAttention::set_arena(const SharedBuffer& a, const SharedBuffer& b)
{
  const bool same = a.contents() == _arena_a.contents() &&
                    a.byte_size() == _arena_a.byte_size() &&
                    b.contents() == _arena_b.contents() &&
                    b.byte_size() == _arena_b.byte_size();
  if (same) { return; }
  auto hold = [](const SharedBuffer& x) {
    return x.byte_size() > 0 ? x.subview(0, x.byte_size()) : SharedBuffer{};
  };
  _arena_a = hold(a);
  _arena_b = hold(b);
  // BEFORE the windows go, because removing needs the handles, and what
  // the set is holding is the OLD arena.
  drop_residency_();
  for (SharedBuffer* p : scratch_buffers_()) { *p = SharedBuffer{}; }
  _heads = _qt = _kt = _d = 0;
  _pv8 = false;
  _reserved = false;
  _own = 0;
  _ready = false;
  _pv8_ready = false;
  _arena_used[0] = 0;
  _arena_used[1] = 0;
}

std::size_t
MetalSageAttention::resident_bytes() const
{
  std::size_t n = 0;
  for (const SharedBuffer* b : {&_km, &_kmp, &_q8, &_k8, &_qs, &_ks, &_v8,
                                 &_vs, &_vm, &_vmp}) {
    n += b->byte_size();
  }
  return n;
}

std::string
MetalSageAttention::check_geometry_(int heads, int kv_heads, int q_tokens,
                                    int k_tokens, int d, int bq, int bk)
{
  if (heads <= 0 || kv_heads <= 0 || q_tokens <= 0 || k_tokens <= 0 ||
      d <= 0) {
    return "sage_attn: degenerate geometry";
  }
  // The kernel derives the KV head from gqa_factor = heads / kv_heads,
  // so a ratio that is not whole would put it on a head this never
  // quantized. Refused rather than rounded.
  if (kv_heads > heads || (heads % kv_heads) != 0) {
    return fmt("sage_attn: {} query heads is not a whole multiple of "
               "{} kv heads", heads, kv_heads)();
  }
  // REFUSED rather than clamped. Both bounds are threadgroup array
  // sizes inside sage_quant_block, so exceeding one writes past a
  // threadgroup allocation -- which produces a plausible wrong answer
  // rather than a fault, and would be found downstream as a bad frame.
  if (d > kMaxD) {
    return fmt("sage_attn: head_dim {} exceeds {}", d, kMaxD)();
  }
  if (bq > kMaxBlock || bk > kMaxBlock || bq <= 0 || bk <= 0) {
    return fmt("sage_attn: block ({}, {}) outside 1..{}", bq, bk,
               kMaxBlock)();
  }
  return {};
}

bool
MetalSageAttention::reserve(int heads, int kv_heads, int q_tokens,
                            int k_tokens, int d, int bq, int bk, bool pv8,
                            std::string* err)
{
  if (_mc == nullptr) {
    if (err != nullptr) { *err = "sage_attn not loaded"; }
    return false;
  }
  const std::string bad =
      check_geometry_(heads, kv_heads, q_tokens, k_tokens, d, bq, bk);
  if (!bad.empty()) {
    if (err != nullptr) { *err = bad; }
    return false;
  }
  return ensure_scratch_(heads, kv_heads, q_tokens, k_tokens, d, bq, bk,
                         pv8 && _pv8_kernels, /*exact=*/true, err);
}

bool
MetalSageAttention::ensure_scratch_(int heads, int kv_heads, int q_tokens,
                                    int k_tokens, int d, int bq, int bk,
                                    bool pv8, bool exact, std::string* err)
{
  // A scratch with the int8 P*V's buffers serves a call without them
  // (its Q scales are per token, more than per block needs), so a run
  // that toggles pv8 -- MiniMax-H3's early steps -- rebuilds once at
  // most, and never if the caller reserved them (reserve_pv8).
  pv8 = pv8 || (_reserve_pv8 && _pv8_kernels);
  // Every buffer's size grows with the token counts alone once heads,
  // head dim and tiles agree, and prepare() writes (and the kernel
  // reads) each call at its own geometry -- so a scratch for more
  // tokens serves fewer. Taken only from a reservation: a caller that
  // made none gets a scratch the size of its last call, as before,
  // rather than one that only ever grows.
  if (_heads == heads && _kvh == kv_heads && _d == d && _bq == bq &&
      _bk == bk && (_pv8 || !pv8) && !_q8.empty()) {
    if (_qt == q_tokens && _kt == k_tokens) {
      if (exact) { _reserved = true; }
      return true;
    }
    if (!exact && _reserved && q_tokens <= _qt && k_tokens <= _kt) {
      return true;
    }
  }
  // A REBUILD: the previous geometry's buffers are about to be
  // overwritten, and the assignments below are the last references to
  // them.
  drop_residency_();
  _ready = false;
  _keys_ready = false;

  const std::size_t H = (std::size_t)heads, D = (std::size_t)d;
  const std::size_t KH = (std::size_t)kv_heads;
  const int nq = sage::query_blocks(q_tokens, bq);
  const int nk = sage::key_blocks(k_tokens, bk);

  _arena_used[0] = 0;
  _arena_used[1] = 0;
  _own = 0;
  auto mk = [&](std::size_t n) -> SharedBuffer {
    // PADDED THE WAY THE PLAN PADS, and asked for at that size: the
    // allocator rounds a private buffer to a page anyway, so charging
    // the plan one figure and the carve another is how an estimate and
    // an allocation come to differ by an alignment apiece.
    const std::size_t need = (n + kCarveAlign - 1) & ~(kCarveAlign - 1);
    SharedBuffer* reg[2] = {&_arena_a, &_arena_b};
    for (int i = 0; i < 2; ++i) {
      if (reg[i]->empty()) { continue; }
      const std::size_t off =
          (_arena_used[i] + kCarveAlign - 1) & ~(kCarveAlign - 1);
      if (off < reg[i]->byte_size() && need <= reg[i]->byte_size() - off) {
        _arena_used[i] = off + need;
        return reg[i]->subview(off, n);
      }
    }
    _own += need;
    return _mc->make_shared_buffer(n);
  };

  _km  = mk(KH * D * 4);
  _kmp = mk(KH * (std::size_t)sage::kMeanChunks * D * 4);
  _q8  = mk(H * (std::size_t)q_tokens * D);
  _k8  = mk(KH * (std::size_t)k_tokens * D);
  _qs  = mk(H * (pv8 ? (std::size_t)q_tokens : (std::size_t)nq) * 4);
  _ks  = mk(KH * (std::size_t)nk * 4);
  if (pv8) {
    _v8  = mk(KH * (std::size_t)k_tokens * D);
    _vs  = mk(KH * D * 4);
    _vm  = mk(KH * D * 4);
    _vmp = mk(KH * (std::size_t)sage::kMeanChunks * D * 4);
  } else {
    _v8 = SharedBuffer{};
    _vs = SharedBuffer{};
    _vm = SharedBuffer{};
    _vmp = SharedBuffer{};
  }
  const bool v_ok = !pv8 || (!_v8.empty() && !_vs.empty() && !_vm.empty() &&
                             !_vmp.empty());
  if (_km.empty() || _kmp.empty() || _q8.empty() || _k8.empty() ||
      _qs.empty() || _ks.empty() || !v_ok) {
    _own = 0;
    _reserved = false;
    for (SharedBuffer* p : scratch_buffers_()) { *p = SharedBuffer{}; }
    if (err != nullptr) { *err = "sage_attn scratch allocation failed"; }
    return false;
  }

  for (const SharedBuffer* b : scratch_buffers_()) {
    if (!b->empty()) { _mc->residency_add(*b); }
  }
  _res_added = true;
  _mc->residency_commit();

  _heads = heads; _kvh = kv_heads; _qt = q_tokens; _kt = k_tokens; _d = d;
  _bq = bq; _bk = bk; _pv8 = pv8;
  _reserved = exact;
  ++_builds;
  return true;
}

bool
MetalSageAttention::prepare_keys(ComputeEncoder& e, const Operand& k,
                                 int heads, int kv_heads, int q_tokens,
                                 int k_tokens, int d, int bq, int bk,
                                 const sage::Config& cfg, std::string* err,
                                 const Operand* v)
{
  _ready = false;
  _pv8_ready = false;
  _keys_ready = false;
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (_mc == nullptr) { return fail("sage_attn not loaded"); }
  const std::string bad =
      check_geometry_(heads, kv_heads, q_tokens, k_tokens, d, bq, bk);
  if (!bad.empty()) { return fail(bad); }
  if (k.buf == nullptr || k.buf->empty()) {
    return fail("sage_attn: no k");
  }
  if (k.row <= 0 || k.head <= 0) {
    return fail("sage_attn: k strides must be positive");
  }
  const bool pv8 = pv8_for(cfg) && v != nullptr && v->buf != nullptr &&
                   !v->buf->empty();
  if (pv8 && (v->row <= 0 || v->head <= 0)) {
    return fail("sage_attn: v strides must be positive");
  }
  if (!ensure_scratch_(heads, kv_heads, q_tokens, k_tokens, d, bq, bk, pv8,
                       /*exact=*/false, err)) {
    return false;
  }
  const unsigned D = (unsigned)d;
  const unsigned KH = (unsigned)kv_heads;
  const int chunks = sage::kMeanChunks;
  const int nk = sage::key_blocks(k_tokens, bk);
  if (cfg.smooth_k) {
    e.set_function(_fn_sum);
    e.set_buffer(0, *k.buf);
    e.set_buffer(1, _kmp);
    e.set_constant(2, k_tokens);
    e.set_constant(3, d);
    e.set_constant(4, k.row);
    e.set_constant(5, (unsigned)k.off);
    e.set_constant(6, chunks);
    e.set_constant(7, k.head);
    e.dispatch({D, KH, (unsigned)chunks}, {D, 1, 1});

    e.set_function(_fn_reduce);
    e.set_buffer(0, _kmp);
    e.set_buffer(1, _km);
    e.set_constant(2, k_tokens);
    e.set_constant(3, d);
    e.set_constant(4, chunks);
    e.dispatch({D, KH, 1}, {D, 1, 1});
  }
  e.set_function(_fn_quant);
  e.set_buffer(0, *k.buf);
  e.set_buffer(1, _km);
  e.set_buffer(2, _k8);
  e.set_buffer(3, _ks);
  e.set_constant(4, k_tokens);
  e.set_constant(5, d);
  e.set_constant(6, k.row);
  e.set_constant(7, (unsigned)k.off);
  e.set_constant(8, bk);
  e.set_constant(9, cfg.smooth_k ? 1 : 0);
  e.set_constant(10, k.head);
  e.dispatch({256u, KH, (unsigned)nk}, {256, 1, 1});
  if (pv8) {
    e.set_function(_fn_sum);
    e.set_buffer(0, *v->buf);
    e.set_buffer(1, _vmp);
    e.set_constant(2, k_tokens);
    e.set_constant(3, d);
    e.set_constant(4, v->row);
    e.set_constant(5, (unsigned)v->off);
    e.set_constant(6, chunks);
    e.set_constant(7, v->head);
    e.dispatch({D, KH, (unsigned)chunks}, {D, 1, 1});

    e.set_function(_fn_reduce);
    e.set_buffer(0, _vmp);
    e.set_buffer(1, _vm);
    e.set_constant(2, k_tokens);
    e.set_constant(3, d);
    e.set_constant(4, chunks);
    e.dispatch({D, KH, 1}, {D, 1, 1});

    e.set_function(_fn_vam);
    e.set_buffer(0, *v->buf);
    e.set_buffer(1, _vm);
    e.set_buffer(2, _vmp);
    e.set_constant(3, k_tokens);
    e.set_constant(4, d);
    e.set_constant(5, v->row);
    e.set_constant(6, (unsigned)v->off);
    e.set_constant(7, chunks);
    e.set_constant(8, v->head);
    e.dispatch({D, KH, (unsigned)chunks}, {D, 1, 1});

    e.set_function(_fn_var);
    e.set_buffer(0, _vmp);
    e.set_buffer(1, _vs);
    e.set_constant(2, d);
    e.set_constant(3, chunks);
    e.dispatch({D, KH, 1}, {D, 1, 1});

    e.set_function(_fn_vq8);
    e.set_buffer(0, *v->buf);
    e.set_buffer(1, _vm);
    e.set_buffer(2, _vs);
    e.set_buffer(3, _v8);
    e.set_constant(4, k_tokens);
    e.set_constant(5, d);
    e.set_constant(6, v->row);
    e.set_constant(7, (unsigned)v->off);
    e.set_constant(8, v->head);
    e.dispatch({D, KH, (unsigned)((k_tokens + 7) / 8)}, {D, 1, 1});
  }
  _keys_pv8 = pv8;
  _keys_ready = true;
  return true;
}

bool
MetalSageAttention::prepare_queries(ComputeEncoder& e, const Operand& q,
                                    int q_tokens, std::string* err)
{
  _ready = false;
  _pv8_ready = false;
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (!_keys_ready) { return fail("sage_attn: queries before keys"); }
  if (q_tokens <= 0 || q_tokens > _qt) {
    return fail("sage_attn: a band wider than the keys' prologue sized");
  }
  if (q.buf == nullptr || q.buf->empty() || q.row <= 0 || q.head <= 0) {
    return fail("sage_attn: no q, or q strides not positive");
  }
  const unsigned H = (unsigned)_heads;
  if (_keys_pv8) {
    e.set_function(_fn_qtok);
    e.set_buffer(0, *q.buf);
    e.set_buffer(1, _q8);
    e.set_buffer(2, _qs);
    e.set_constant(3, q_tokens);
    e.set_constant(4, _d);
    e.set_constant(5, q.row);
    e.set_constant(6, (unsigned)q.off);
    e.set_constant(7, q.head);
    e.set_buffer(8, _km);
    e.set_constant(9, 0);
    e.dispatch({256u, H, (unsigned)((q_tokens + 63) / 64)}, {256, 1, 1});
  } else {
    e.set_function(_fn_quant);
    e.set_buffer(0, *q.buf);
    e.set_buffer(1, _km);
    e.set_buffer(2, _q8);
    e.set_buffer(3, _qs);
    e.set_constant(4, q_tokens);
    e.set_constant(5, _d);
    e.set_constant(6, q.row);
    e.set_constant(7, (unsigned)q.off);
    e.set_constant(8, _bq);
    e.set_constant(9, 0);
    e.set_constant(10, q.head);
    e.dispatch({256u, H, (unsigned)sage::query_blocks(q_tokens, _bq)},
               {256, 1, 1});
  }
  _ready = true;
  _pv8_ready = _keys_pv8;
  return true;
}

bool
MetalSageAttention::prepare(ComputeEncoder& e, const Operand& q,
                            const Operand& k, int heads, int kv_heads,
                            int q_tokens, int k_tokens, int d, int bq, int bk,
                            const sage::Config& cfg, std::string* err,
                            const Operand* v)
{
  _ready = false;
  _pv8_ready = false;
  _keys_ready = false;
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (_mc == nullptr) { return fail("sage_attn not loaded"); }
  const std::string bad =
      check_geometry_(heads, kv_heads, q_tokens, k_tokens, d, bq, bk);
  if (!bad.empty()) { return fail(bad); }
  if (q.buf == nullptr || k.buf == nullptr || q.buf->empty() ||
      k.buf->empty()) {
    return fail("sage_attn: no q/k");
  }
  if (q.row <= 0 || k.row <= 0 || q.head <= 0 || k.head <= 0) {
    return fail("sage_attn: q/k strides must be positive");
  }
  // The int8 P*V wants V too, and its kernels; asked for without
  // either, the int8 QK alone (pv8_for() says which to a caller).
  const bool pv8 = pv8_for(cfg) && v != nullptr && v->buf != nullptr &&
                   !v->buf->empty();
  if (pv8 && (v->row <= 0 || v->head <= 0)) {
    return fail("sage_attn: v strides must be positive");
  }
  if (!ensure_scratch_(heads, kv_heads, q_tokens, k_tokens, d, bq, bk, pv8,
                       /*exact=*/false, err)) {
    return false;
  }

  const unsigned H = (unsigned)heads, D = (unsigned)d;
  const unsigned KH = (unsigned)kv_heads;
  const int chunks = sage::kMeanChunks;
  const int nq = sage::query_blocks(q_tokens, bq);
  const int nk = sage::key_blocks(k_tokens, bk);

  // 1. The key mean, in two stages. See sage::kMeanChunks for why the
  //    obvious one-pass shape is not used.
  if (cfg.smooth_k) {
    e.set_function(_fn_sum);
    e.set_buffer(0, *k.buf);
    e.set_buffer(1, _kmp);
    e.set_constant(2, k_tokens);
    e.set_constant(3, d);
    e.set_constant(4, k.row);
    e.set_constant(5, (unsigned)k.off);
    e.set_constant(6, chunks);
    e.set_constant(7, k.head);
    e.dispatch({D, KH, (unsigned)chunks}, {D, 1, 1});

    e.set_function(_fn_reduce);
    e.set_buffer(0, _kmp);
    e.set_buffer(1, _km);
    e.set_constant(2, k_tokens);
    e.set_constant(3, d);
    e.set_constant(4, chunks);
    e.dispatch({D, KH, 1}, {D, 1, 1});
  }

  // 2. Q, unsmoothed, at the QUERY block -- or, for the int8 P*V, per
  //    TOKEN (its kernel reads constant 310's layout). See
  //    sage-attention.h for why only the key side carries a mean.
  if (pv8) {
    e.set_function(_fn_qtok);
    e.set_buffer(0, *q.buf);
    e.set_buffer(1, _q8);
    e.set_buffer(2, _qs);
    e.set_constant(3, q_tokens);
    e.set_constant(4, d);
    e.set_constant(5, q.row);
    e.set_constant(6, (unsigned)q.off);
    e.set_constant(7, q.head);
    e.set_buffer(8, _km);
    e.set_constant(9, 0);
    e.dispatch({256u, H, (unsigned)((q_tokens + 63) / 64)}, {256, 1, 1});
  } else {
  e.set_function(_fn_quant);
  e.set_buffer(0, *q.buf);
  e.set_buffer(1, _km);
  e.set_buffer(2, _q8);
  e.set_buffer(3, _qs);
  e.set_constant(4, q_tokens);
  e.set_constant(5, d);
  e.set_constant(6, q.row);
  e.set_constant(7, (unsigned)q.off);
  e.set_constant(8, bq);
  e.set_constant(9, 0);
  e.set_constant(10, q.head);
  e.dispatch({256u, H, (unsigned)nq}, {256, 1, 1});
  }

  // 3. K, smoothed, at the KEY block.
  e.set_function(_fn_quant);
  e.set_buffer(0, *k.buf);
  e.set_buffer(1, _km);
  e.set_buffer(2, _k8);
  e.set_buffer(3, _ks);
  e.set_constant(4, k_tokens);
  e.set_constant(5, d);
  e.set_constant(6, k.row);
  e.set_constant(7, (unsigned)k.off);
  e.set_constant(8, bk);
  e.set_constant(9, cfg.smooth_k ? 1 : 0);
  e.set_constant(10, k.head);
  e.dispatch({256u, KH, (unsigned)nk}, {256, 1, 1});

  // 4. The int8 P*V's V: its mean over tokens (added back exactly after
  //    the kernel's normalize), its absmax per channel over the whole
  //    sequence -- one scale a channel, as the held int32 sum needs --
  //    and V8 + 128 as uint8.
  if (pv8) {
    e.set_function(_fn_sum);
    e.set_buffer(0, *v->buf);
    e.set_buffer(1, _vmp);
    e.set_constant(2, k_tokens);
    e.set_constant(3, d);
    e.set_constant(4, v->row);
    e.set_constant(5, (unsigned)v->off);
    e.set_constant(6, chunks);
    e.set_constant(7, v->head);
    e.dispatch({D, KH, (unsigned)chunks}, {D, 1, 1});

    e.set_function(_fn_reduce);
    e.set_buffer(0, _vmp);
    e.set_buffer(1, _vm);
    e.set_constant(2, k_tokens);
    e.set_constant(3, d);
    e.set_constant(4, chunks);
    e.dispatch({D, KH, 1}, {D, 1, 1});

    e.set_function(_fn_vam);
    e.set_buffer(0, *v->buf);
    e.set_buffer(1, _vm);
    e.set_buffer(2, _vmp);
    e.set_constant(3, k_tokens);
    e.set_constant(4, d);
    e.set_constant(5, v->row);
    e.set_constant(6, (unsigned)v->off);
    e.set_constant(7, chunks);
    e.set_constant(8, v->head);
    e.dispatch({D, KH, (unsigned)chunks}, {D, 1, 1});

    e.set_function(_fn_var);
    e.set_buffer(0, _vmp);
    e.set_buffer(1, _vs);
    e.set_constant(2, d);
    e.set_constant(3, chunks);
    e.dispatch({D, KH, 1}, {D, 1, 1});

    e.set_function(_fn_vq8);
    e.set_buffer(0, *v->buf);
    e.set_buffer(1, _vm);
    e.set_buffer(2, _vs);
    e.set_buffer(3, _v8);
    e.set_constant(4, k_tokens);
    e.set_constant(5, d);
    e.set_constant(6, v->row);
    e.set_constant(7, (unsigned)v->off);
    e.set_constant(8, v->head);
    e.dispatch({D, KH, (unsigned)((k_tokens + 7) / 8)}, {D, 1, 1});
    _pv8_ready = true;
  }

  _ready = true;
  return true;
}

void
MetalSageAttention::bind(ComputeEncoder& e) const
{
  (void)bind(e, 0);
}

bool
MetalSageAttention::bind(ComputeEncoder& e, int q_row0) const
{
  if (!_ready || q_row0 < 0) { return false; }
  if (q_row0 > 0) {
    // Per token only: a per-block scale would be indexed by the
    // dispatch's own block numbers, which no longer count from 0. What
    // decides is what the LAST prepare() wrote, not the scratch's size:
    // a pv8 scratch serves a call that wrote per-block scales.
    if (!_pv8_ready) { return false; }
    e.set_buffer(15, _q8, (std::size_t)q_row0 * (std::size_t)_d);
    e.set_buffer(16, _k8);
    e.set_buffer(17, _qs, (std::size_t)q_row0 * 4);
    e.set_buffer(18, _ks);
    if (_pv8_ready) {
      e.set_buffer(19, _v8);
      e.set_buffer(20, _vs);
      e.set_buffer(21, _vm);
    }
    return true;
  }
  e.set_buffer(15, _q8);
  e.set_buffer(16, _k8);
  e.set_buffer(17, _qs);
  e.set_buffer(18, _ks);
  if (_pv8_ready) {
    e.set_buffer(19, _v8);
    e.set_buffer(20, _vs);
    e.set_buffer(21, _vm);
  }
  return true;
}

void
MetalSageAttention::pv8_constants(metal_compute::FunctionConstants& fc,
                                  bool on)
{
  // 308 the int8 PV (its held uint8 form), 310 Q per token.
  fc.set_bool(sage::kPvInt8Constant, on)
      .set_bool(sage::kQTokenScaleConstant, on);
}

}  // namespace genai
}  // namespace vpipe

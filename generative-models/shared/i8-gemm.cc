#include "generative-models/shared/i8-gemm.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace vpipe {
namespace genai {

using metal_compute::ComputeEncoder;
using metal_compute::ComputeFunction;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

I8GemmContext::I8GemmContext(MetalCompute* mc, bool want, bool bf16) : _mc(mc)
{
  bool on = want;
  if (const char* e = std::getenv("VPIPE_I8_GEMM")) {
    on = std::atoi(e) != 0;                    // env overrides either way
  }
  if (!on || mc == nullptr || !mc->supports_matrix_cores()) { return; }
  // The kernels' float I/O is VPIPE_ELT; load the _bf16 twins for a bf16
  // caller (its bf16 activations/weights would be garbage through the f16
  // kernels). Entry-point names keep the historical "_f16" label.
  _lib_q = mc->load_library(bf16 ? "affine_dequant_bf16" : "affine_dequant");
  _lib_g = mc->load_library(bf16 ? "dense_gemm_mma_bf16" : "dense_gemm_mma");
  // The group, and with it the kernel pair. 64 is the mode; 512 is the
  // drained pair it replaced, kept for the A/B (see the class note).
  if (const char* e = std::getenv("VPIPE_I8_GROUP")) {
    const int v = std::atoi(e);
    _group = (v == 32 || v == 128 || v == 512) ? v : 64;
  }
  const bool g64 = _group == 64;
  // The register pairs at 32 / 64 / 128, the drained pair at 512.
  struct Names { int g; const char *q, *qp, *gemm, *sk; };
  static const Names kNames[] = {
      {32, "quant_f16_i8_row_g32", "quant_f16_i8_row_g32_pad",
       "gemm_i8i8_sc_f16_n64_g32rs", "gemm_i8i8_sc_f16_n64_g32rs_sk"},
      {64, "quant_f16_i8_row_g64", "quant_f16_i8_row_g64_pad",
       "gemm_i8i8_sc_f16_n64_g64rs", "gemm_i8i8_sc_f16_n64_g64rs_sk"},
      {128, "quant_f16_i8_row_g128", "quant_f16_i8_row_g128_pad",
       "gemm_i8i8_sc_f16_n64_g128rs", "gemm_i8i8_sc_f16_n64_g128rs_sk"},
      {512, "quant_f16_i8_row_g512", "quant_f16_i8_row_g512_pad",
       "gemm_i8i8_sc_f16_n64_g512", "gemm_i8i8_sc_f16_n64_g512_sk"},
  };
  const Names* nm = &kNames[1];
  for (const Names& n : kNames) {
    if (n.g == _group) { nm = &n; }
  }
  const char* q_name = nm->q;
  const char* qp_name = nm->qp;
  const char* g_name = nm->gemm;
  const char* sk_name = nm->sk;
  _fn_quant = _lib_q.function(q_name);
  _fn_quant_pad = _lib_q.function(qp_name);
  _fn_gemm = _lib_g.function(g_name);
  _on = _fn_quant.valid() && _fn_gemm.valid();
  // The split-K twin and the fold it drains through. Both optional: a
  // missing one leaves the single-op path exactly as it was, which is
  // why nothing below is guarded on `_on` alone.
  _fn_gemm_sk = _lib_g.function(sk_name);
  _lib_elt = mc->load_library(bf16 ? "llm_elementwise_bf16"
                                   : "llm_elementwise");
  _fn_fold = _lib_elt.function("splitk_fold_f32_f16");
  _fn_fold_bias = _lib_elt.function("splitk_fold_bias_f32_f16");
  // The native affine pair, group 64 only.
  if (const char* e = std::getenv("VPIPE_I8_NATIVE")) {
    const int v = std::atoi(e);
    _native = v == 0 ? 0 : (v == 1 ? 12 : (v & 12));
  }
  if (g64) {
    _fn_quant_u8 = _lib_q.function("quant_f16_u8_row_g64");
    _fn_wsum8 = _lib_q.function("affine_group_sums_w8g64");
    _fn_wsum4 = _lib_q.function("affine_group_sums_w4g64");
    _fn_u8q_w8 = _lib_g.function("gemm_u8q_w8g64");
    _fn_u8q_w4 = _lib_g.function("gemm_u8q_w4g64");
    _fn_u8q_w8_sk = _lib_g.function("gemm_u8q_w8g64_sk");
    _fn_u8q_w4_sk = _lib_g.function("gemm_u8q_w4g64_sk");
    _fn_u8q_w8_cm = _lib_g.function("gemm_u8q_w8g64_cm");
    _fn_u8q_w4_cm = _lib_g.function("gemm_u8q_w4g64_cm");
    _fn_u8q_w8_cm_sk = _lib_g.function("gemm_u8q_w8g64_cm_sk");
    _fn_u8q_w4_cm_sk = _lib_g.function("gemm_u8q_w4g64_cm_sk");
  }
  if (const char* e = std::getenv("VPIPE_I8_NATIVE_CMODE")) {
    _cmode = std::atoi(e) == 0 ? 0 : 1;
  }
  {
    long mb = 512;
    if (const char* e = std::getenv("VPIPE_I8_SPLITK_MAX_MB")) {
      const long v = std::atol(e);
      if (v > 0) { mb = v; }
    }
    _plane_budget = (std::size_t)mb * 1024u * 1024u;
  }
  if (const char* e = std::getenv("VPIPE_I8_SPLITK_MAX_S")) {
    const int v = std::atoi(e);
    _max_splits = v >= 0 ? v : 16;
  }
  if (const char* e = std::getenv("VPIPE_I8_SPLIT_MIN_K")) {
    const int v = std::atoi(e);
    if (v >= 0) { _split_min_k = v; }
  }
  if (const char* e = std::getenv("VPIPE_I8_GEMM_MIN_M")) {
    _min_m = std::atoi(e);
  }
  if (const char* e = std::getenv("VPIPE_I8_MAX_PAD_PCT")) {
    const int v = std::atoi(e);
    if (v >= 0) { _max_pad_pct = v; }
  }
}

std::vector<int>
I8GemmContext::split_candidates(int M, int N, int K) const
{
  std::vector<int> out;
  if (!split_available()) { return out; }
  const int G = kpad_(K) / _group;
  // BALANCED PLANES ONLY. The kernel takes a ragged last plane happily,
  // but an S that leaves one plane with a third of the groups is a
  // straggler the fold waits on, and it never won a round here.
  for (int S = 2; S <= _max_splits && S <= G; ++S) {
    if (G % S != 0) { continue; }
    if (rows_per_block_(S, N) < 64) { continue; }   // budget too tight
    out.push_back(S);
  }
  (void)M;
  return out;
}

std::vector<int>
I8GemmContext::tune_candidates(int M, int N, int K) const
{
  std::vector<int> out{0};
  for (int S : split_candidates(M, N, K)) {
    if (K >= _split_min_k || S == 2) { out.push_back(S); }
  }
  return out;
}

int
I8GemmContext::plan_(int M, int N, int K) const
{
  if (!split_available() || bypass_split) { return 0; }
  if (force_splits > 0) { return force_splits; }
  const int key = m_key_(M);
  for (const Tuned& t : _tuned) {
    if (t.N == N && t.K == K && t.M == key) { return t.splits; }
  }
  // UNTUNED, short of the deep-K cliff: the single op. There is no
  // occupancy deficit there for a split to cure, and S=2 measured 0.67-
  // 0.92x of it at the shapes tried (see split_min_k()).
  if (K < _split_min_k) { return 0; }
  // UNTUNED, past it: 2, when the shape admits it. Not 0, because a
  // caller that never tunes should still get the part of this that is
  // free, and not more, because 2 is the only width that won at every row
  // count measured at H3's fc2 (1.23-1.44x) -- the wider ones win big at
  // 256 rows and LOSE at 1024+. An untuned guess that can lose is worse
  // than a small certain gain.
  const std::vector<int> c = split_candidates(M, N, K);
  for (int S : c) {
    if (S == 2) { return 2; }
  }
  return 0;
}

bool
I8GemmContext::gemm(ComputeEncoder& enc, const SharedBuffer& x,
                    std::size_t xe, const SharedBuffer& w,
                    const SharedBuffer& b, const SharedBuffer& y,
                    std::size_t ye, int M, int N, int K)
{
  // A bias with no fold to add it in would be silently dropped on the
  // split path, so the split is refused rather than the bias.
  const bool bias = !b.empty();
  if (bias && !_fn_fold_bias.valid() && plan_(M, N, K) > 0) {
    return false;
  }
  if (!accepts(M, N, K)) { return false; }
  // Everything below the quantizers works in the PADDED contraction: the
  // scratches, the group count and the GEMM's K. Only the two quantize
  // dispatches see the real K, and only to know where to stop reading.
  const int KP = kpad_(K);
  const int G = KP / _group;
  auto need = [&](SharedBuffer& b, std::size_t bytes) {
    if (b.empty() || b.byte_size() < bytes) {
      b = _mc->make_shared_buffer(bytes);
    }
    return !b.empty();
  };
  if (!need(_xq, (std::size_t)M * KP) ||
      !need(_as, (std::size_t)M * G * 2) ||
      !need(_wq, (std::size_t)N * KP) ||
      !need(_ws, (std::size_t)N * G * 2)) {
    return false;
  }
  const bool pad = (KP != K);
  // Activation rows -> i8 + per-(row, group) scales.
  enc.set_function(pad ? _fn_quant_pad : _fn_quant);
  enc.set_buffer(0, x, xe * 2);
  enc.set_buffer(1, _xq);
  enc.set_buffer(2, _as);
  enc.set_constant(3, K);
  if (pad) { enc.set_constant(4, KP); }
  enc.dispatch({256u, (unsigned)M, 1}, {256, 1, 1});
  // Weight rows (the [N,K] f16 dense weight) -> i8 + per-(chan, group)
  // scales. Re-quantized per call: one extra weight pass, no persistent
  // copy. Serial command streams + Metal's hazard tracking make the
  // shared scratches safe across back-to-back GEMMs.
  enc.set_function(pad ? _fn_quant_pad : _fn_quant);
  enc.set_buffer(0, w);
  enc.set_buffer(1, _wq);
  enc.set_buffer(2, _ws);
  enc.set_constant(3, K);
  if (pad) { enc.set_constant(4, KP); }
  enc.dispatch({256u, (unsigned)N, 1}, {256, 1, 1});
  // RECORD, so a caller with no closure can tune this shape later. Only
  // when a split is possible and this one is not settled: a shape the
  // tuner has answered is never pending again.
  if (split_available() && !bypass_split && force_splits == 0) {
    const int key = m_key_(M);
    bool known = false;
    for (const Tuned& t : _tuned) {
      if (t.N == N && t.K == K && t.M == key) { known = true; break; }
    }
    for (const Tuned& t : _pending) {
      if (t.N == N && t.K == K && t.M == key) { known = true; break; }
    }
    if (!known && _pending.size() < 16) {
      _pending.push_back(Tuned{N, K, key, 0});
    }
  }
  const unsigned nx = (unsigned)(((N + 63) / 64) * 128);
  const int S = plan_(M, N, K);
  if (S <= 0) {
    // i8 x i8 GEMM, per-group f32 accumulate, f16 store.
    enc.set_function(_fn_gemm);
    enc.set_buffer(0, _xq);
    enc.set_buffer(1, _wq);
    enc.set_buffer(2, _as);
    enc.set_buffer(3, _ws);
    enc.set_buffer(4, y, ye * 2);
    enc.set_constant(5, KP);
    enc.set_constant(6, N);
    enc.set_constant(7, M);
    enc.set_buffer(8, bias ? b : _ws);
    enc.set_constant(9, bias ? 1 : 0);
    enc.dispatch({nx, (unsigned)((M + 63) / 64), 1}, {128, 1, 1});
    return true;
  }
  // SPLIT, in row blocks the plane budget admits. The planes are S f32
  // copies of the block's output, so a deep split costs more DISPATCHES
  // rather than more memory -- the same trade MmaSplitK makes, and the
  // reason the budget constrains the search instead of vetoing after it.
  const int rows = rows_per_block_(S, N);
  const std::size_t plane_bytes =
      (std::size_t)S * (std::size_t)rows * (std::size_t)N * 4u;
  if (rows < 64 || !need(_planes, plane_bytes)) {
    // Fall back rather than fail: the caller has no dense path left at
    // this point (the quantize dispatches are already encoded), so the
    // single op is the only correct answer.
    enc.set_function(_fn_gemm);
    enc.set_buffer(0, _xq);
    enc.set_buffer(1, _wq);
    enc.set_buffer(2, _as);
    enc.set_buffer(3, _ws);
    enc.set_buffer(4, y, ye * 2);
    enc.set_constant(5, KP);
    enc.set_constant(6, N);
    enc.set_constant(7, M);
    enc.set_buffer(8, bias ? b : _ws);
    enc.set_constant(9, bias ? 1 : 0);
    enc.dispatch({nx, (unsigned)((M + 63) / 64), 1}, {128, 1, 1});
    return true;
  }
  const int gpp = (G + S - 1) / S;
  for (int m0 = 0; m0 < M; m0 += rows) {
    const int mb = (M - m0) < rows ? (M - m0) : rows;
    enc.set_function(_fn_gemm_sk);
    // Row-banded by OFFSET: xq is [M, KP] i8 and as is [M, G] scales, so
    // a band is a plain stride in each. The weight side is not banded.
    enc.set_buffer(0, _xq, (std::size_t)m0 * KP);
    enc.set_buffer(1, _wq);
    enc.set_buffer(2, _as, (std::size_t)m0 * G * 2);
    enc.set_buffer(3, _ws);
    enc.set_buffer(4, _planes);
    enc.set_constant(5, KP);
    enc.set_constant(6, N);
    enc.set_constant(7, mb);
    enc.set_constant(8, gpp);
    enc.dispatch({nx, (unsigned)((mb + 63) / 64), (unsigned)S},
                 {128, 1, 1});
    enc.set_function(bias ? _fn_fold_bias : _fn_fold);
    enc.set_buffer(0, _planes);
    enc.set_buffer(1, y, (ye + (std::size_t)m0 * N) * 2);
    const int n_el = mb * N;
    enc.set_constant(2, n_el);
    enc.set_constant(3, S);
    if (bias) { enc.set_buffer(4, b); enc.set_constant(5, N); }
    enc.dispatch({(unsigned)n_el, 1, 1}, {256, 1, 1});
  }
  return true;
}

bool
I8GemmContext::gemm_affine(ComputeEncoder& enc, const SharedBuffer& x,
                           std::size_t xe, const SharedBuffer& codes,
                           const SharedBuffer& scales,
                           const SharedBuffer& qbias, int bits,
                           const SharedBuffer& b, const SharedBuffer& y,
                           std::size_t ye, int M, int N, int K)
{
  if (!native_available(bits)) { return false; }
  if (M < _min_m || K < 1024 || N < 16 || (K % 64) != 0) { return false; }
  const bool bias = !b.empty();
  const int S = plan_(M, N, K);
  if (bias && !_fn_fold_bias.valid() && S > 0) { return false; }
  const bool cm = _cmode == 1 &&
      (bits == 8 ? _fn_u8q_w8_cm.valid() : _fn_u8q_w4_cm.valid());
  const ComputeFunction& fg =
      cm ? (bits == 8 ? _fn_u8q_w8_cm : _fn_u8q_w4_cm)
         : (bits == 8 ? _fn_u8q_w8 : _fn_u8q_w4);
  const ComputeFunction& fsk =
      cm ? (bits == 8 ? _fn_u8q_w8_cm_sk : _fn_u8q_w4_cm_sk)
         : (bits == 8 ? _fn_u8q_w8_sk : _fn_u8q_w4_sk);
  const ComputeFunction& fsum = bits == 8 ? _fn_wsum8 : _fn_wsum4;
  const int G = K / 64;
  auto need = [&](SharedBuffer& sb, std::size_t bytes) {
    if (sb.empty() || sb.byte_size() < bytes) {
      sb = _mc->make_shared_buffer(bytes);
    }
    return !sb.empty();
  };
  if (!need(_xu, (std::size_t)M * K) ||
      !need(_as, (std::size_t)M * G * 2) ||
      !need(_xs, (std::size_t)M * G * 2) ||
      !need(_uf, (std::size_t)M * G * 2) ||
      !need(_wsum, (std::size_t)N * G * 2) ||
      !need(_chi, (std::size_t)N * G * 2) ||
      !need(_clo, (std::size_t)N * G * 2)) {
    return false;
  }
  // Activation rows -> u8 codes + per-(row, group) scale, code sum and
  // the scaled sum the zero-point matmul reads.
  enc.set_function(_fn_quant_u8);
  enc.set_buffer(0, x, xe * 2);
  enc.set_buffer(1, _xu);
  enc.set_buffer(2, _as);
  enc.set_buffer(3, _xs);
  enc.set_constant(4, K);
  enc.set_buffer(5, _uf);
  enc.dispatch({256u, (unsigned)M, 1}, {256, 1, 1});
  // The weight's per-(channel, group) code sums: one read of the codes,
  // against the M/64 reads the GEMM makes.
  enc.set_function(fsum);
  enc.set_buffer(0, codes);
  enc.set_buffer(1, _wsum);
  enc.set_constant(2, K);
  enc.set_constant(3, N);
  enc.set_buffer(4, scales);
  enc.set_buffer(5, _chi);
  enc.set_buffer(6, _clo);
  enc.dispatch({(unsigned)G, (unsigned)N, 1}, {64, 1, 1});

  const unsigned nx = (unsigned)(((N + 63) / 64) * 128);
  auto single = [&]() {
    enc.set_function(fg);
    enc.set_buffer(0, _xu);
    enc.set_buffer(1, codes);
    enc.set_buffer(2, _as);
    enc.set_buffer(3, _xs);
    enc.set_buffer(4, scales);
    enc.set_buffer(5, qbias);
    enc.set_buffer(6, _wsum);
    enc.set_buffer(7, y, ye * 2);
    enc.set_constant(8, K);
    enc.set_constant(9, N);
    enc.set_constant(10, M);
    enc.set_buffer(11, bias ? b : scales);
    enc.set_constant(12, bias ? 1 : 0);
    enc.set_buffer(13, _uf);
    enc.set_buffer(14, _chi);
    enc.set_buffer(15, _clo);
    enc.dispatch({nx, (unsigned)((M + 63) / 64), 1}, {128, 1, 1});
  };
  if (S <= 0 || !fsk.valid()) {
    single();
    return true;
  }
  const int rows = rows_per_block_(S, N);
  const std::size_t plane_bytes =
      (std::size_t)S * (std::size_t)rows * (std::size_t)N * 4u;
  if (rows < 64 || !need(_planes, plane_bytes)) {
    single();
    return true;
  }
  const int gpp = (G + S - 1) / S;
  for (int m0 = 0; m0 < M; m0 += rows) {
    const int mb = (M - m0) < rows ? (M - m0) : rows;
    enc.set_function(fsk);
    enc.set_buffer(0, _xu, (std::size_t)m0 * K);
    enc.set_buffer(1, codes);
    enc.set_buffer(2, _as, (std::size_t)m0 * G * 2);
    enc.set_buffer(3, _xs, (std::size_t)m0 * G * 2);
    enc.set_buffer(4, scales);
    enc.set_buffer(5, qbias);
    enc.set_buffer(6, _wsum);
    enc.set_buffer(7, _planes);
    enc.set_constant(8, K);
    enc.set_constant(9, N);
    enc.set_constant(10, mb);
    enc.set_constant(11, gpp);
    enc.set_buffer(13, _uf, (std::size_t)m0 * G * 2);
    enc.set_buffer(14, _chi);
    enc.set_buffer(15, _clo);
    enc.dispatch({nx, (unsigned)((mb + 63) / 64), (unsigned)S},
                 {128, 1, 1});
    enc.set_function(bias ? _fn_fold_bias : _fn_fold);
    enc.set_buffer(0, _planes);
    enc.set_buffer(1, y, (ye + (std::size_t)m0 * N) * 2);
    const int n_el = mb * N;
    enc.set_constant(2, n_el);
    enc.set_constant(3, S);
    if (bias) { enc.set_buffer(4, b); enc.set_constant(5, N); }
    enc.dispatch({(unsigned)n_el, 1, 1}, {256, 1, 1});
  }
  return true;
}

void
I8GemmContext::encode_tuned_(ComputeEncoder& enc, int M, int N, int KP,
                             int S)
{
  const unsigned nx = (unsigned)(((N + 63) / 64) * 128);
  const int G = KP / _group;
  if (S <= 0) {
    enc.set_function(_fn_gemm);
    enc.set_buffer(0, _xq); enc.set_buffer(1, _wq);
    enc.set_buffer(2, _as); enc.set_buffer(3, _ws);
    enc.set_buffer(4, _tune_y);
    enc.set_constant(5, KP); enc.set_constant(6, N); enc.set_constant(7, M);
    // No bias while TIMING: it is one add per output, identical across
    // candidates, so including it would only add noise to the vote.
    enc.set_buffer(8, _ws); enc.set_constant(9, 0);
    enc.dispatch({nx, (unsigned)((M + 63) / 64), 1}, {128, 1, 1});
    return;
  }
  const int rows = rows_per_block_(S, N);
  if (rows < 64) { return; }
  const int gpp = (G + S - 1) / S;
  for (int m0 = 0; m0 < M; m0 += rows) {
    const int mb = (M - m0) < rows ? (M - m0) : rows;
    enc.set_function(_fn_gemm_sk);
    enc.set_buffer(0, _xq, (std::size_t)m0 * KP);
    enc.set_buffer(1, _wq);
    enc.set_buffer(2, _as, (std::size_t)m0 * G * 2);
    enc.set_buffer(3, _ws);
    enc.set_buffer(4, _planes);
    enc.set_constant(5, KP); enc.set_constant(6, N); enc.set_constant(7, mb);
    enc.set_constant(8, gpp);
    enc.dispatch({nx, (unsigned)((mb + 63) / 64), (unsigned)S}, {128, 1, 1});
    enc.set_function(_fn_fold);
    enc.set_buffer(0, _planes);
    enc.set_buffer(1, _tune_y, (std::size_t)m0 * N * 2);
    const int n_el = mb * N;
    enc.set_constant(2, n_el); enc.set_constant(3, S);
    enc.dispatch({(unsigned)n_el, 1, 1}, {256, 1, 1});
  }
}

void
I8GemmContext::tune_pending(MetalCompute* mc)
{
  if (_pending.empty() || mc == nullptr || !split_available()) { return; }
  std::vector<Tuned> todo;
  todo.swap(_pending);
  for (const Tuned& p : todo) {
    const int KP = kpad_(p.K);
    const int M = p.M < kTuneRows ? p.M : kTuneRows;
    const std::vector<int> cands = tune_candidates(M, p.N, p.K);
    // The scratches must still cover the shape. They are grow-only and
    // this shape ran, so they do -- unless release_scratch() intervened,
    // in which case the honest answer is to drop the shape rather than
    // measure a buffer that is not there.
    if (cands.size() < 2 || _xq.empty() || _wq.empty() ||
        _xq.byte_size() < (std::size_t)M * KP ||
        _wq.byte_size() < (std::size_t)p.N * KP) {
      _tuned.push_back(Tuned{p.N, p.K, p.M, 0});
      continue;
    }
    const std::size_t ybytes = (std::size_t)M * p.N * 2;
    if (_tune_y.empty() || _tune_y.byte_size() < ybytes) {
      _tune_y = mc->make_shared_buffer(ybytes);
    }
    int smax = 0;
    for (int S : cands) { smax = S > smax ? S : smax; }
    const std::size_t pbytes = (std::size_t)smax *
        (std::size_t)rows_per_block_(smax, p.N) * (std::size_t)p.N * 4u;
    if (_planes.empty() || _planes.byte_size() < pbytes) {
      _planes = mc->make_shared_buffer(pbytes);
    }
    if (_tune_y.empty() || _planes.empty()) {
      _tuned.push_back(Tuned{p.N, p.K, p.M, 0});
      continue;
    }
    // A SAMPLE OF ~4 ms, not one dispatch. At the tune rows a short-K
    // GEMM is 0.5-3 ms, the same order as a commit-and-wait's jitter and
    // as the gap between the two candidates below the cliff, and
    // one-dispatch samples voted S=2 in 4 of 35 trials there (M5 Pro)
    // against a single op 10-30% faster. Sized off the single op, timed
    // once (which also warms it).
    auto sample = [&](int S, int reps) {
      return autotune_time(mc, reps, [&](ComputeEncoder& e) {
        encode_tuned_(e, M, p.N, KP, S);
      });
    };
    const double one_ms = 1e3 * sample(0, 1);
    const int reps = one_ms <= 0.0 ? 1
        : std::clamp((int)std::ceil(kTuneSampleMs / one_ms), 1, 8);
    const int w = autotune_vote((int)cands.size(), /*rounds=*/3, reps,
        [&](int i) { return sample(cands[(std::size_t)i], reps); });
    _tuned.push_back(Tuned{p.N, p.K, p.M, cands[(std::size_t)w]});
  }
  // Held only for the measurement: it is as large as an output and the
  // forward that follows wants that room back.
  _tune_y = {};
}

void
I8GemmContext::release_scratch()
{
  _xq = {};
  _as = {};
  _wq = {};
  _ws = {};
  _xu = {};
  _xs = {};
  _wsum = {};
  _uf = {};
  _chi = {};
  _clo = {};
  _planes = {};
  _tune_y = {};
}

}  // namespace genai
}  // namespace vpipe

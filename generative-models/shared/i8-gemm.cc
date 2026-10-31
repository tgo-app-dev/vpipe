#include "generative-models/shared/i8-gemm.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace vpipe {
namespace genai {

using metal_compute::ComputeEncoder;
using metal_compute::ComputeFunction;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

// A refcount-sharing handle on the whole of `b` (SharedBuffer is move-
// only): what a pending tune record keeps of the caller's weight.
SharedBuffer
alias_(const SharedBuffer& b)
{
  return b.empty() ? SharedBuffer{} : b.subview(0, b.byte_size());
}

// A route's name in a session key (split_key_), by the Route's value:
// kRequant, kNative, kSigned.
const char*
route_name_(int route)
{
  switch (route) {
    case 1: return "native";
    case 2: return "signed";
    default: return "requant";
  }
}

}  // namespace

I8GemmContext::I8GemmContext(MetalCompute* mc, bool want, bool bf16)
  : _mc(mc), _bf16(bf16)
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
  if (g64) {   // the rotated pair (set_g64_rotation); optional
    _fn_quant_r = _lib_q.function("quant_f16_i8_row_g64_r");
    _fn_quant_r_pad = _lib_q.function("quant_f16_i8_row_g64_r_pad");
  }
  if (const char* e = std::getenv("VPIPE_I8_ACC_ROT")) {
    _g64_rot_env = std::atoi(e) != 0 ? 1 : 0;
  }
  // ON by default, where the host has the kernels (see the header).
  (void)set_g64_rotation(true);
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
  // The signed layout, group 64 only.
  if (const char* e = std::getenv("VPIPE_I8_SIGNED")) {
    const int v = std::atoi(e);
    _signed = v == 0 ? 0 : (v == 1 ? 12 : (v & 12));
  }
  if (g64) {
    _fn_quant_s8 = _lib_q.function("quant_f16_i8s_row_g64");
    _fn_quant_swiglu_s8 = _lib_q.function("swiglu_quant_f16_i8s_row_g64");
    _fn_to_s8 = _lib_q.function("affine_to_signed_w8g64");
    _fn_to_s4 = _lib_q.function("affine_to_signed_w4g64");
    _fn_dq_s8 = _lib_q.function("affine_dequant_w8g64s");
    _fn_dq_s4 = _lib_q.function("affine_dequant_w4g64s");
    _fn_s8_w8 = _lib_g.function("gemm_i8s_w8g64");
    _fn_s8_w4 = _lib_g.function("gemm_i8s_w4g64");
    _fn_s8_w8_sk = _lib_g.function("gemm_i8s_w8g64_sk");
    _fn_s8_w4_sk = _lib_g.function("gemm_i8s_w4g64_sk");
  }
  // THE FAST TIER's kernels, per group: the rotated quantizers and the
  // coarse GEMM at its tile -- 128x64 over 8 simdgroups, which beat
  // 64x128 in both forward A/Bs on the M5 Ultra (1.085x against 1.067x
  // of the accurate tier); VPIPE_I8_FAST_TILE=64x128 for the A/B.
  {
    const bool t128 = [] {
      const char* e = std::getenv("VPIPE_I8_FAST_TILE");
      return e == nullptr || std::strcmp(e, "64x128") != 0;
    }();
    _fast_bm = t128 ? 128 : 64;
    _fast_bn = t128 ? 64 : 128;
    const int gcs[3] = {256, 512, 1024};
    for (int i = 0; i < 3; ++i) {
      const std::string g = std::to_string(gcs[i]);
      _fast[i].quant = _lib_q.function(("quant_i8_row_gc" + g + "_r").c_str());
      _fast[i].swiglu =
          _lib_q.function(("swiglu_quant_i8_row_gc" + g + "_r").c_str());
      _fast[i].to_fast =
          _lib_q.function(("affine_to_i8c_w8_gc" + g + "_r").c_str());
      _fast[i].to_fast4 =
          _lib_q.function(("affine_to_i8c_w4_gc" + g + "_r").c_str());
      _fast[i].dequant =
          _lib_q.function(("i8c_dequant_gc" + g + "_r").c_str());
      _fast[i].gemm = _lib_g.function(
          ("gemm_i8c_g" + g + (t128 ? "_t128x64" : "_t64x128")).c_str());
    }
  }
  // THE TOKEN TIER's: the per-row quantizers and the register-fragment
  // GEMM (set_token_kernel: MLX's blocked loop at 128x64 as 4x2
  // simdgroups of 32x32 unless the caller asks otherwise).
  // VPIPE_I8_TOKEN_TILE=64x128 / VPIPE_I8_TOKEN_KERNEL=<entry suffix>
  // (m128x64, d64x128, ...) override the caller.
  {
    _fast[3].quant = _lib_q.function("quant_i8_tok");
    _fast[3].swiglu = _lib_q.function("swiglu_quant_i8_tok");
    _fast[3].to_fast = _lib_q.function("affine_to_i8pt_w8");
    _fast[3].to_fast4 = _lib_q.function("affine_to_i8pt_w4");
    _fast[3].dequant = _lib_q.function("i8pt_dequant");
    _fast[4].quant = _lib_q.function("quant_i8_tok_r");
    _fast[4].swiglu = _lib_q.function("swiglu_quant_i8_tok_r");
    _fast[4].to_fast = _lib_q.function("affine_to_i8pt_w8_r");
    _fast[4].to_fast4 = _lib_q.function("affine_to_i8pt_w4_r");
    _fast[4].dequant = _lib_q.function("i8pt_dequant_r");
    _fn_tok_cr = _lib_q.function("quant_i8_tok_cr");
    set_token_kernel(false, false);
    bool db = false, wide = false, env = false;
    if (const char* e = std::getenv("VPIPE_I8_TOKEN_TILE")) {
      wide = std::strcmp(e, "64x128") == 0;
      env = true;
    }
    if (const char* e = std::getenv("VPIPE_I8_TOKEN_KERNEL")) {
      int bm = 0, bn = 0;
      if ((e[0] == 'm' || e[0] == 'd') &&
          std::sscanf(e + 1, "%dx%d", &bm, &bn) == 2 &&
          ((bm == 128 && bn == 64) || (bm == 64 && bn == 128))) {
        db = e[0] == 'd';
        wide = bm == 64;
        env = true;
      }
    }
    if (env) {
      set_token_kernel(db, wide);
      _tok_kernel_env = true;
    }
    if (const char* e = std::getenv("VPIPE_I8_TOKEN_ROT")) {
      _tok_rot_env = std::atoi(e) != 0 ? 1 : 0;
      _tok_rot = _tok_rot_env != 0;
    }
  }
  if (const char* e = std::getenv("VPIPE_I8_TIER")) {
    if (std::strcmp(e, "token") == 0 || std::strcmp(e, "2") == 0) {
      _tier_env = (int)Tier::kToken;
    } else if (std::strcmp(e, "fast") == 0 || std::strcmp(e, "1") == 0) {
      _tier_env = (int)Tier::kFast;
    } else if (std::strcmp(e, "accurate") == 0 || std::strcmp(e, "0") == 0) {
      _tier_env = (int)Tier::kAccurate;
    }
    if (_tier_env >= 0) { _tier = (Tier)_tier_env; }
  }
  if (const char* e = std::getenv("VPIPE_I8_FAST_GROUP")) {
    const int v = std::atoi(e);
    if (v == 256 || v == 512 || v == 1024) { _fast_group = v; }
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
  if (const char* e = std::getenv("VPIPE_I8_UNTUNED_SPLIT")) {
    const int v = std::atoi(e);
    _untuned_split = v >= 2 ? v : 0;
  }
  if (const char* e = std::getenv("VPIPE_I8_GEMM_MIN_M")) {
    _min_m = std::atoi(e);
  }
  if (const char* e = std::getenv("VPIPE_I8_SIGNED_MIN_M")) {
    _signed_min_m = std::max(1, std::atoi(e));
  }
  if (const char* e = std::getenv("VPIPE_I8_MAX_PAD_PCT")) {
    const int v = std::atoi(e);
    if (v >= 0) { _max_pad_pct = v; }
  }
  // What this session's contexts have measured before -- a model loaded
  // again, or another with the same shapes (tuned-answers.h).
  if (_on) { _answers = tuning_cache(mc); }
}

bool
I8GemmContext::set_g64_rotation(bool r)
{
  if (_g64_rot_env >= 0) { r = _g64_rot_env != 0; }
  _g64_rot = r && _group == 64 && _fn_quant_r.valid() &&
             _fn_quant_r_pad.valid();
  return _g64_rot;
}

void
I8GemmContext::set_token_kernel(bool double_buffered, bool wide_n)
{
  if (_tok_kernel_env || !_lib_g.valid()) { return; }
  const char* name = double_buffered
      ? (wide_n ? "gemm_i8pt_d64x128" : "gemm_i8pt_d128x64")
      : (wide_n ? "gemm_i8pt_m64x128" : "gemm_i8pt_m128x64");
  _fast[3].gemm = _lib_g.function(name);
  _fast[4].gemm = _lib_g.function(name);
  _tok_db = double_buffered;
  _tok_wm = wide_n ? 2 : 4;
  _tok_wn = wide_n ? 4 : 2;
}

std::string
I8GemmContext::token_kernel() const
{
  return std::string(_tok_db ? "double-buffered " : "blocked ") +
         (_tok_wm == 4 ? "128x64" : "64x128");
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
  // UNTUNED: the single op, at every K. On the g64 register kernels it
  // won at every deep-K shape measured from the i8 floor up (see
  // split_min_k()); the S=2 this returned past K 9216 cost 7-8% there.
  // VPIPE_I8_UNTUNED_SPLIT brings that rule back for an A/B.
  if (_untuned_split > 0 && K >= _split_min_k) {
    for (int S : split_candidates(M, N, K)) {
      if (S == _untuned_split) { return S; }
    }
  }
  return 0;
}

bool
I8GemmContext::gemm(ComputeEncoder& enc, const SharedBuffer& x,
                    std::size_t xe, const SharedBuffer& w,
                    const SharedBuffer& b, const SharedBuffer& y,
                    std::size_t ye, int M, int N, int K)
{
  return gemm_(enc, x, xe, w, b, y, ye, M, N, K, _g64_rot);
}

bool
I8GemmContext::gemm_(ComputeEncoder& enc, const SharedBuffer& x,
                     std::size_t xe, const SharedBuffer& w,
                     const SharedBuffer& b, const SharedBuffer& y,
                     std::size_t ye, int M, int N, int K, bool rot)
{
  const bool bias = !b.empty();
  if (!accepts_(M, N, K, rot)) { return false; }
  settle_(Route::kRequant, 8, N, K, M);
  // A bias with no fold to add it in would be silently dropped on the
  // split path, so the split is refused rather than the bias.
  if (bias && !_fn_fold_bias.valid() && plan_(M, N, K) > 0) {
    return false;
  }
  _fq_m = _fq_k = _fq_gc = 0;   // the scratch is about to be rewritten
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
  // The plain pair, or the rotated one: the GEMM below is the same.
  const ComputeFunction& fq =
      rot ? (pad ? _fn_quant_r_pad : _fn_quant_r)
          : (pad ? _fn_quant_pad : _fn_quant);
  // Activation rows -> i8 + per-(row, group) scales.
  enc.set_function(fq);
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
  enc.set_function(fq);
  enc.set_buffer(0, w);
  enc.set_buffer(1, _wq);
  enc.set_buffer(2, _ws);
  enc.set_constant(3, K);
  if (pad) { enc.set_constant(4, KP); }
  enc.dispatch({256u, (unsigned)N, 1}, {256, 1, 1});
  // RECORD, so a caller with no closure can tune this shape later. Only
  // when a split is possible and this one is not settled: a shape the
  // tuner has answered is never pending again.
  record_(Pending{N, K, M});
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
  const int rows = rows_per_block_(S, N, M);
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

void
I8GemmContext::settle_(Route route, int bits, int N, int K, int M)
{
  if (_answers == nullptr || !split_available() || bypass_split ||
      force_splits != 0) {
    return;
  }
  const int key = m_key_(M);
  for (const Tuned& t : _tuned) {
    if (t.N == N && t.K == K && t.M == key) { return; }
  }
  for (const Asked& a : _asked) {
    if (a.N == N && a.K == K && a.M == key) { return; }
  }
  int kept = 0;
  if (find_tuned(_answers,
                 split_key_(route_name_((int)route), bits, N, K, M), kept)) {
    _tuned.push_back(Tuned{N, K, key, kept});
    return;
  }
  // Bounded as _pending is: a caller sweeping sequence lengths asks
  // again past it, which costs a lookup, never a wrong answer.
  if (_asked.size() < 64) { _asked.push_back(Asked{N, K, key}); }
}

void
I8GemmContext::record_(Pending p)
{
  if (!split_available() || bypass_split || force_splits != 0) { return; }
  const int key = m_key_(p.M);
  for (const Tuned& t : _tuned) {
    if (t.N == p.N && t.K == p.K && t.M == key) { return; }
  }
  for (const Pending& t : _pending) {
    if (t.N == p.N && t.K == p.K && m_key_(t.M) == key) { return; }
  }
  if (_pending.size() < 16) { _pending.push_back(std::move(p)); }
}

void
I8GemmContext::encode_native_gemm_(ComputeEncoder& enc,
                                   const SharedBuffer& codes,
                                   const SharedBuffer& scales,
                                   const SharedBuffer& qbias, int bits,
                                   const SharedBuffer& b,
                                   const SharedBuffer& y, std::size_t ye,
                                   int M, int N, int K, int S)
{
  const bool bias = !b.empty();
  const bool cm = _cmode == 1 &&
      (bits == 8 ? _fn_u8q_w8_cm.valid() : _fn_u8q_w4_cm.valid());
  const ComputeFunction& fg =
      cm ? (bits == 8 ? _fn_u8q_w8_cm : _fn_u8q_w4_cm)
         : (bits == 8 ? _fn_u8q_w8 : _fn_u8q_w4);
  const ComputeFunction& fsk =
      cm ? (bits == 8 ? _fn_u8q_w8_cm_sk : _fn_u8q_w4_cm_sk)
         : (bits == 8 ? _fn_u8q_w8_sk : _fn_u8q_w4_sk);
  const int G = K / 64;
  const unsigned nx = (unsigned)(((N + 63) / 64) * 128);
  const int rows = S > 0 ? rows_per_block_(S, N, M) : 0;
  const std::size_t plane_bytes =
      (std::size_t)S * (std::size_t)rows * (std::size_t)N * 4u;
  if (S <= 0 || !fsk.valid() || rows < 64 || _planes.empty() ||
      _planes.byte_size() < plane_bytes) {
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
    return;
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
}

void
I8GemmContext::encode_signed_gemm_(ComputeEncoder& enc,
                                   const SharedBuffer& codes,
                                   const SharedBuffer& scales,
                                   const SharedBuffer& zb, int bits,
                                   const SharedBuffer& b,
                                   const SharedBuffer& y, std::size_t ye,
                                   int M, int N, int K, int S)
{
  const bool bias = !b.empty();
  const ComputeFunction& fg = bits == 8 ? _fn_s8_w8 : _fn_s8_w4;
  const ComputeFunction& fsk = bits == 8 ? _fn_s8_w8_sk : _fn_s8_w4_sk;
  const int G = K / 64;
  const unsigned nx = (unsigned)(((N + 63) / 64) * 128);
  const int rows = S > 0 ? rows_per_block_(S, N, M) : 0;
  const std::size_t plane_bytes =
      (std::size_t)S * (std::size_t)rows * (std::size_t)N * 4u;
  if (S <= 0 || !fsk.valid() || rows < 64 || _planes.empty() ||
      _planes.byte_size() < plane_bytes) {
    enc.set_function(fg);
    enc.set_buffer(0, _xq);
    enc.set_buffer(1, codes);
    enc.set_buffer(2, _as);
    enc.set_buffer(3, scales);
    enc.set_buffer(4, zb);
    enc.set_buffer(5, _uf);
    enc.set_buffer(6, y, ye * 2);
    enc.set_constant(7, K);
    enc.set_constant(8, N);
    enc.set_constant(9, M);
    enc.set_buffer(10, bias ? b : scales);
    enc.set_constant(11, bias ? 1 : 0);
    enc.dispatch({nx, (unsigned)((M + 63) / 64), 1}, {128, 1, 1});
    return;
  }
  const int gpp = (G + S - 1) / S;
  for (int m0 = 0; m0 < M; m0 += rows) {
    const int mb = (M - m0) < rows ? (M - m0) : rows;
    enc.set_function(fsk);
    enc.set_buffer(0, _xq, (std::size_t)m0 * K);
    enc.set_buffer(1, codes);
    enc.set_buffer(2, _as, (std::size_t)m0 * G * 2);
    enc.set_buffer(3, scales);
    enc.set_buffer(4, zb);
    enc.set_buffer(5, _uf, (std::size_t)m0 * G * 2);
    enc.set_buffer(6, _planes);
    enc.set_constant(7, K);
    enc.set_constant(8, N);
    enc.set_constant(9, mb);
    enc.set_constant(10, gpp);
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
  if (K < 1024 || N < 16 || (K % 64) != 0) { return false; }
  _fq_m = _fq_k = _fq_gc = 0;   // the scratch is about to be rewritten
  const bool bias = !b.empty();
  const int G = K / 64;
  auto need = [&](SharedBuffer& sb, std::size_t bytes) {
    if (sb.empty() || sb.byte_size() < bytes) {
      sb = _mc->make_shared_buffer(bytes);
    }
    return !sb.empty();
  };
  // IN THE SIGNED LAYOUT, PER CALL, where the width allows it: convert
  // the codes into a scratch (one pass over them, as the unsigned route's
  // code sums are) and take the signed kernel. MEASURED on the M5 Pro
  // (gemm_i8.w8_against_bf16_routes): 0.98-1.05x of a bf16 weight under
  // i8 where the unsigned kernel ran 0.90-1.02x. A family that holds the
  // layout itself skips the pass and calls gemm_signed (1.01-1.09x).
  // At the requant route's floor, whatever VPIPE_I8_SIGNED_MIN_M says:
  // which shapes go int8 is the caller's contract with this mode, and a
  // caller of the native route (an LM's prefill among them) was promised
  // 1024 rows.
  if (M >= _min_m && signed_accepts(M, N, K, bits) &&
      need(_sc, (std::size_t)N * (std::size_t)K * (std::size_t)bits / 8) &&
      need(_szb, (std::size_t)N * (std::size_t)G * 2) &&
      encode_to_signed(enc, codes, scales, qbias, _sc, _szb, bits, N, K)) {
    if (gemm_signed(enc, x, xe, _sc, scales, _szb, bits, b, y, ye, M, N,
                    K)) {
      return true;
    }
  }
  if (M < _min_m) { return false; }
  settle_(Route::kNative, bits, N, K, M);
  int S = plan_(M, N, K);
  if (bias && !_fn_fold_bias.valid() && S > 0) { return false; }
  const ComputeFunction& fsum = bits == 8 ? _fn_wsum8 : _fn_wsum4;
  if (!need(_xu, (std::size_t)M * K) ||
      !need(_as, (std::size_t)M * G * 2) ||
      !need(_xs, (std::size_t)M * G * 2) ||
      !need(_uf, (std::size_t)M * G * 2) ||
      !need(_wsum, (std::size_t)N * G * 2) ||
      !need(_chi, (std::size_t)N * G * 2) ||
      !need(_clo, (std::size_t)N * G * 2)) {
    return false;
  }
  // The split's planes, or the single op when they cannot be had.
  if (S > 0) {
    const std::size_t pb = (std::size_t)S *
        (std::size_t)rows_per_block_(S, N, M) * (std::size_t)N * 4u;
    if (rows_per_block_(S, N, M) < 64 || !need(_planes, pb)) { S = 0; }
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
  // RECORD, so tune_pending can settle this shape's split. The native
  // route never did, and kept the untuned default at every shape.
  record_(Pending{N, K, M, Route::kNative, bits, alias_(codes),
                  alias_(scales), alias_(qbias)});
  encode_native_gemm_(enc, codes, scales, qbias, bits, b, y, ye, M, N, K,
                      S);
  return true;
}

bool
I8GemmContext::gemm_signed(ComputeEncoder& enc, const SharedBuffer& x,
                           std::size_t xe, const SharedBuffer& codes,
                           const SharedBuffer& scales,
                           const SharedBuffer& zb, int bits,
                           const SharedBuffer& b, const SharedBuffer& y,
                           std::size_t ye, int M, int N, int K)
{
  if (!signed_accepts(M, N, K, bits)) { return false; }
  _fq_m = _fq_k = _fq_gc = 0;   // the scratch is about to be rewritten
  const bool bias = !b.empty();
  settle_(Route::kSigned, bits, N, K, M);
  int S = plan_(M, N, K);
  if (bias && !_fn_fold_bias.valid() && S > 0) { return false; }
  const int G = K / 64;
  auto need = [&](SharedBuffer& sb, std::size_t bytes) {
    if (sb.empty() || sb.byte_size() < bytes) {
      sb = _mc->make_shared_buffer(bytes);
    }
    return !sb.empty();
  };
  // The activation scratches are the requant route's own (i8 codes and
  // scales) plus the native route's u: the routes run one at a time on a
  // serial stream, so sharing them costs nothing.
  if (!need(_xq, (std::size_t)M * K) ||
      !need(_as, (std::size_t)M * G * 2) ||
      !need(_uf, (std::size_t)M * G * 2)) {
    return false;
  }
  if (S > 0) {
    const std::size_t pb = (std::size_t)S *
        (std::size_t)rows_per_block_(S, N, M) * (std::size_t)N * 4u;
    if (rows_per_block_(S, N, M) < 64 || !need(_planes, pb)) { S = 0; }
  }
  // Activation rows -> signed codes, per-(row, group) scales, and u =
  // scale * sum(codes) for the zero-point matmul.
  enc.set_function(_fn_quant_s8);
  enc.set_buffer(0, x, xe * 2);
  enc.set_buffer(1, _xq);
  enc.set_buffer(2, _as);
  enc.set_buffer(3, _uf);
  enc.set_constant(4, K);
  enc.dispatch({256u, (unsigned)M, 1}, {256, 1, 1});
  record_(Pending{N, K, M, Route::kSigned, bits, alias_(codes),
                  alias_(scales), alias_(zb)});
  encode_signed_gemm_(enc, codes, scales, zb, bits, b, y, ye, M, N, K, S);
  return true;
}

const I8GemmContext::FastFns*
I8GemmContext::fast_fns_(int gc) const
{
  const int i = gc == 256 ? 0 : gc == 512 ? 1 : gc == 1024 ? 2
                : gc == kTokenGroup ? (_tok_rot ? 4 : 3) : -1;
  if (i < 0) { return nullptr; }
  const FastFns& f = _fast[i];
  return f.quant.valid() && f.swiglu.valid() && f.to_fast.valid() &&
                 f.dequant.valid() && f.gemm.valid()
             ? &f : nullptr;
}

int
I8GemmContext::fast_group(int K) const
{
  if (!_on || _tier == Tier::kAccurate || K <= 0) { return 0; }
  if (_tier == Tier::kToken) {
    return (K % (_tok_rot ? 256 : 64)) == 0 && K <= 16384 &&
                   fast_fns_(kTokenGroup) != nullptr
               ? kTokenGroup : 0;
  }
  for (int gc : {1024, 512, 256}) {
    if (gc <= _fast_group && (K % gc) == 0 && fast_fns_(gc) != nullptr) {
      return gc;
    }
  }
  return 0;
}

namespace {

// Whether K splits into whole groups of gc: any K the token tier's
// kernels take (a multiple of their 64-deep step, of 256 rotated) at
// kTokenGroup.
bool
group_fits_(int K, int gc, bool rot)
{
  return gc == I8GemmContext::kTokenGroup
             ? (K % (rot ? 256 : 64)) == 0 && K <= 16384
             : gc > 0 && (K % gc) == 0;
}

// The coarse tier's row passes (quant_i8_row_gc* and friends): one
// simdgroup per (row, group), four groups a threadgroup. The token
// tier's: a threadgroup of K/16 threads a row (i8pt_dequant: 256);
// rotated, a simdgroup per TWO 256-blocks (the kernels take one or
// two) -- 1.05-1.18x one each (gemm_i8.fast_conversion_rate, M5 Ultra
// and base M5).
void
encode_rows_gc_(ComputeEncoder& enc, int rows, int K, int gc,
                bool dequant = false, bool rot = false)
{
  if (gc == I8GemmContext::kTokenGroup) {
    const unsigned T =
        rot       ? 32u * (unsigned)std::max(1, (K / 256 + 1) / 2)
        : dequant ? 256u
                  : (unsigned)std::max(32, (K / 16 + 31) / 32 * 32);
    enc.dispatch({T, (unsigned)rows, 1}, {T, 1, 1});
    return;
  }
  const unsigned tgs = (unsigned)((K / gc + 3) / 4);
  enc.dispatch({tgs * 128u, (unsigned)rows, 1}, {128, 1, 1});
}

// The coarse GEMM's dispatch (gemm_i8c_* in dense_gemm_mma); wm > 0 is
// the token tier's register kernel, wm x wn simdgroups of 32x32.
void
encode_fast_gemm_(ComputeEncoder& enc, const ComputeFunction& f,
                  const SharedBuffer& xq, const SharedBuffer& codes,
                  const SharedBuffer& as, const SharedBuffer& ws,
                  const SharedBuffer& b, const SharedBuffer& y,
                  std::size_t ye, int M, int N, int K, int bm, int bn,
                  int wm = 0, int wn = 0)
{
  enc.set_function(f);
  enc.set_buffer(0, xq);
  enc.set_buffer(1, codes);
  enc.set_buffer(2, as);
  enc.set_buffer(3, ws);
  enc.set_buffer(4, y, ye * 2);
  enc.set_constant(5, K);
  enc.set_constant(6, N);
  enc.set_constant(7, M);
  enc.set_buffer(8, b.empty() ? ws : b);
  enc.set_constant(9, b.empty() ? 0 : 1);
  if (wm > 0) {
    const unsigned BM = 32u * wm, BN = 32u * wn;
    enc.dispatch({(unsigned)((N + BN - 1) / BN) * 32u,
                  (unsigned)((M + BM - 1) / BM) * wn, (unsigned)wm},
                 {32u, (unsigned)wn, (unsigned)wm});
    return;
  }
  enc.dispatch({(unsigned)((N + bn - 1) / bn) * 256u,
                (unsigned)((M + bm - 1) / bm), 1}, {256, 1, 1});
}

}  // namespace

bool
I8GemmContext::gemm_fast(ComputeEncoder& enc, const SharedBuffer& x,
                         std::size_t xe, const SharedBuffer& codes,
                         const SharedBuffer& scales, int gc,
                         const SharedBuffer& b, const SharedBuffer& y,
                         std::size_t ye, int M, int N, int K)
{
  const FastFns* f = fast_fns_(gc);
  if (f == nullptr || !fast_accepts(M, N, K) || !group_fits_(K, gc, _tok_rot)) {
    return false;
  }
  auto need = [&](SharedBuffer& sb, std::size_t bytes) {
    if (sb.empty() || sb.byte_size() < bytes) {
      sb = _mc->make_shared_buffer(bytes);
    }
    return !sb.empty();
  };
  if (!need(_xq, (std::size_t)M * K) ||
      !need(_as, fast_scale_bytes(M, K, gc))) {
    return false;
  }
  enc.set_function(f->quant);
  enc.set_buffer(0, x, xe * 2);
  enc.set_buffer(1, _xq);
  enc.set_buffer(2, _as);
  enc.set_constant(3, K);
  encode_rows_gc_(enc, M, K, gc, false, _tok_rot);
  _fq_m = _fq_k = _fq_gc = 0;
  encode_fast_gemm_(enc, f->gemm, _xq, codes, _as, scales, b, y, ye, M, N,
                    K, _fast_bm, _fast_bn, tok_wm_(gc), _tok_wn);
  return true;
}

bool
I8GemmContext::quantize_fast(ComputeEncoder& enc, const SharedBuffer& x,
                             std::size_t xe, int M, int K, int gc)
{
  _fq_m = _fq_k = _fq_gc = 0;
  const FastFns* f = fast_fns_(gc);
  if (f == nullptr || M <= 0 || K <= 0 || !group_fits_(K, gc, _tok_rot)) {
    return false;
  }
  auto need = [&](SharedBuffer& sb, std::size_t bytes) {
    if (sb.empty() || sb.byte_size() < bytes) {
      sb = _mc->make_shared_buffer(bytes);
    }
    return !sb.empty();
  };
  if (!need(_xq, (std::size_t)M * K) ||
      !need(_as, fast_scale_bytes(M, K, gc))) {
    return false;
  }
  enc.set_function(f->quant);
  enc.set_buffer(0, x, xe * 2);
  enc.set_buffer(1, _xq);
  enc.set_buffer(2, _as);
  enc.set_constant(3, K);
  encode_rows_gc_(enc, M, K, gc, false, _tok_rot);
  _fq_m = M;
  _fq_k = K;
  _fq_gc = gc;
  return true;
}

bool
I8GemmContext::quantize_token_convrot(ComputeEncoder& enc,
                                      const SharedBuffer& x, std::size_t xe,
                                      int M, int K)
{
  _fq_m = _fq_k = _fq_gc = 0;
  if (!_fn_tok_cr.valid() || fast_fns_(kTokenGroup) == nullptr || M <= 0 ||
      K <= 0 || K % 256 != 0 || fast_group(K) != kTokenGroup) {
    return false;
  }
  auto need = [&](SharedBuffer& sb, std::size_t bytes) {
    if (sb.empty() || sb.byte_size() < bytes) {
      sb = _mc->make_shared_buffer(bytes);
    }
    return !sb.empty();
  };
  if (!need(_xq, (std::size_t)M * K) ||
      !need(_as, fast_scale_bytes(M, K, kTokenGroup))) {
    return false;
  }
  enc.set_function(_fn_tok_cr);
  enc.set_buffer(0, x, xe * 2);
  enc.set_buffer(1, _xq);
  enc.set_buffer(2, _as);
  enc.set_constant(3, K);
  // The rotated quantizers' geometry: a simdgroup a 256-block or two.
  encode_rows_gc_(enc, M, K, kTokenGroup, false, /*rot=*/true);
  _fq_m = M;
  _fq_k = K;
  _fq_gc = kTokenGroup;
  return true;
}

bool
I8GemmContext::gemm_fast_quantized(ComputeEncoder& enc,
                                   const SharedBuffer& codes,
                                   const SharedBuffer& scales, int gc,
                                   const SharedBuffer& b,
                                   const SharedBuffer& y, std::size_t ye,
                                   int M, int N, int K)
{
  const FastFns* f = fast_fns_(gc);
  if (f == nullptr || M != _fq_m || K != _fq_k || gc != _fq_gc ||
      !fast_accepts(M, N, K)) {
    return false;
  }
  encode_fast_gemm_(enc, f->gemm, _xq, codes, _as, scales, b, y, ye, M, N,
                    K, _fast_bm, _fast_bn, tok_wm_(gc), _tok_wn);
  return true;
}

bool
I8GemmContext::gemm_fast_swiglu(ComputeEncoder& enc, const SharedBuffer& gate,
                                std::size_t ge, const SharedBuffer& up,
                                std::size_t ue, const SharedBuffer& codes,
                                const SharedBuffer& scales, int gc,
                                const SharedBuffer& y, std::size_t ye, int M,
                                int N, int K)
{
  const FastFns* f = fast_fns_(gc);
  if (f == nullptr || !fast_accepts(M, N, K) || !group_fits_(K, gc, _tok_rot)) {
    return false;
  }
  auto need = [&](SharedBuffer& sb, std::size_t bytes) {
    if (sb.empty() || sb.byte_size() < bytes) {
      sb = _mc->make_shared_buffer(bytes);
    }
    return !sb.empty();
  };
  if (!need(_xq, (std::size_t)M * K) ||
      !need(_as, fast_scale_bytes(M, K, gc))) {
    return false;
  }
  _fq_m = _fq_k = _fq_gc = 0;   // the scratch is about to be rewritten
  enc.set_function(f->swiglu);
  enc.set_buffer(0, gate, ge * 2);
  enc.set_buffer(1, up, ue * 2);
  enc.set_buffer(2, _xq);
  enc.set_buffer(3, _as);
  enc.set_constant(4, K);
  encode_rows_gc_(enc, M, K, gc, false, _tok_rot);
  encode_fast_gemm_(enc, f->gemm, _xq, codes, _as, scales, SharedBuffer{}, y,
                    ye, M, N, K, _fast_bm, _fast_bn, tok_wm_(gc), _tok_wn);
  return true;
}

bool
I8GemmContext::fast_converts(int bits, int gc) const
{
  const FastFns* f = fast_fns_(gc);
  return f != nullptr && (bits == 8 || (bits == 4 && f->to_fast4.valid()));
}

bool
I8GemmContext::encode_to_fast(ComputeEncoder& enc, const SharedBuffer& codes_in,
                              const SharedBuffer& scales,
                              const SharedBuffer& biases,
                              const SharedBuffer& codes_out,
                              const SharedBuffer& out_scales, int gc, int N,
                              int K, int bits) const
{
  const FastFns* f = fast_fns_(gc);
  if (f == nullptr || N <= 0 || K <= 0 || !group_fits_(K, gc, _tok_rot) ||
      !fast_converts(bits, gc) ||
      codes_in.byte_size() < (std::size_t)N * K * bits / 8 ||
      codes_out.byte_size() < (std::size_t)N * K ||
      scales.byte_size() < (std::size_t)N * (K / 64) * 2 ||
      biases.byte_size() < (std::size_t)N * (K / 64) * 2 ||
      out_scales.byte_size() < fast_scale_bytes(N, K, gc)) {
    return false;
  }
  if (bits == 4) {
    // Each output row is twice its input row: in place, one row's
    // writes would land on rows other threadgroups are still reading.
    const auto* ci = static_cast<const char*>(codes_in.contents());
    const auto* co = static_cast<const char*>(codes_out.contents());
    if (ci == nullptr || co == nullptr ||
        (co < ci + (std::size_t)N * K / 2 && ci < co + (std::size_t)N * K)) {
      return false;
    }
  }
  enc.set_function(bits == 4 ? f->to_fast4 : f->to_fast);
  enc.set_buffer(0, codes_in);
  enc.set_buffer(1, scales);
  enc.set_buffer(2, biases);
  enc.set_buffer(3, codes_out);
  enc.set_buffer(4, out_scales);
  enc.set_constant(5, K);
  encode_rows_gc_(enc, N, K, gc, false, _tok_rot);
  return true;
}

bool
I8GemmContext::encode_to_fast_scratch(ComputeEncoder& enc,
                                      const SharedBuffer& codes,
                                      const SharedBuffer& scales,
                                      const SharedBuffer& biases, int gc,
                                      int N, int K)
{
  if (!fast_converts(4, gc) || N <= 0 || K <= 0) { return false; }
  auto need = [&](SharedBuffer& sb, std::size_t bytes) {
    if (sb.empty() || sb.byte_size() < bytes) {
      sb = _mc->make_shared_buffer(bytes);
    }
    return !sb.empty();
  };
  if (!need(_wq, (std::size_t)N * K) ||
      !need(_ws, fast_scale_bytes(N, K, gc))) {
    return false;
  }
  return encode_to_fast(enc, codes, scales, biases, _wq, _ws, gc, N, K, 4);
}

bool
I8GemmContext::encode_quant_fast(ComputeEncoder& enc, const SharedBuffer& w,
                                 const SharedBuffer& codes_out,
                                 const SharedBuffer& out_scales, int gc,
                                 int N, int K) const
{
  const FastFns* f = fast_fns_(gc);
  if (f == nullptr || N <= 0 || K <= 0 || !group_fits_(K, gc, _tok_rot) ||
      w.byte_size() < (std::size_t)N * K * 2 ||
      codes_out.byte_size() < (std::size_t)N * K ||
      out_scales.byte_size() < fast_scale_bytes(N, K, gc)) {
    return false;
  }
  enc.set_function(f->quant);
  enc.set_buffer(0, w);
  enc.set_buffer(1, codes_out);
  enc.set_buffer(2, out_scales);
  enc.set_constant(3, K);
  encode_rows_gc_(enc, N, K, gc, false, _tok_rot);
  return true;
}

bool
I8GemmContext::encode_dequant_fast(ComputeEncoder& enc,
                                   const SharedBuffer& codes,
                                   const SharedBuffer& scales, int gc,
                                   const SharedBuffer& out, int N,
                                   int K) const
{
  const FastFns* f = fast_fns_(gc);
  if (f == nullptr || N <= 0 || K <= 0 || !group_fits_(K, gc, _tok_rot) ||
      out.byte_size() < (std::size_t)N * K * 2) {
    return false;
  }
  enc.set_function(f->dequant);
  enc.set_buffer(0, codes);
  enc.set_buffer(1, scales);
  enc.set_buffer(2, out);
  enc.set_constant(3, K);
  encode_rows_gc_(enc, N, K, gc, /*dequant=*/true, _tok_rot);
  return true;
}

bool
I8GemmContext::gemm_signed_swiglu(ComputeEncoder& enc,
                                  const SharedBuffer& gate, std::size_t ge,
                                  const SharedBuffer& up, std::size_t ue,
                                  const SharedBuffer& codes,
                                  const SharedBuffer& scales,
                                  const SharedBuffer& zb, int bits,
                                  const SharedBuffer& y, std::size_t ye,
                                  int M, int N, int K)
{
  if (!_fn_quant_swiglu_s8.valid() || !signed_accepts(M, N, K, bits)) {
    return false;
  }
  _fq_m = _fq_k = _fq_gc = 0;   // the scratch is about to be rewritten
  int S = plan_(M, N, K);
  const int G = K / 64;
  auto need = [&](SharedBuffer& sb, std::size_t bytes) {
    if (sb.empty() || sb.byte_size() < bytes) {
      sb = _mc->make_shared_buffer(bytes);
    }
    return !sb.empty();
  };
  if (!need(_xq, (std::size_t)M * K) ||
      !need(_as, (std::size_t)M * G * 2) ||
      !need(_uf, (std::size_t)M * G * 2)) {
    return false;
  }
  if (S > 0) {
    const std::size_t pb = (std::size_t)S *
        (std::size_t)rows_per_block_(S, N, M) * (std::size_t)N * 4u;
    if (rows_per_block_(S, N, M) < 64 || !need(_planes, pb)) { S = 0; }
  }
  enc.set_function(_fn_quant_swiglu_s8);
  enc.set_buffer(0, gate, ge * 2);
  enc.set_buffer(1, up, ue * 2);
  enc.set_buffer(2, _xq);
  enc.set_buffer(3, _as);
  enc.set_buffer(4, _uf);
  enc.set_constant(5, K);
  enc.dispatch({256u, (unsigned)M, 1}, {256, 1, 1});
  record_(Pending{N, K, M, Route::kSigned, bits, alias_(codes),
                  alias_(scales), alias_(zb)});
  encode_signed_gemm_(enc, codes, scales, zb, bits, SharedBuffer{}, y, ye, M,
                      N, K, S);
  return true;
}

bool
I8GemmContext::encode_to_signed(ComputeEncoder& enc,
                                const SharedBuffer& codes_in,
                                const SharedBuffer& scales,
                                const SharedBuffer& biases,
                                const SharedBuffer& codes_out,
                                const SharedBuffer& zb, int bits, int N,
                                int K) const
{
  if (!signed_available(bits) || N <= 0 || K <= 0 || (K % 64) != 0) {
    return false;
  }
  const std::size_t code_bytes = (std::size_t)N * (std::size_t)K * bits / 8;
  const std::size_t g_bytes = (std::size_t)N * (std::size_t)(K / 64) * 2;
  if (codes_in.byte_size() < code_bytes ||
      codes_out.byte_size() < code_bytes || scales.byte_size() < g_bytes ||
      biases.byte_size() < g_bytes || zb.byte_size() < g_bytes) {
    return false;
  }
  enc.set_function(bits == 8 ? _fn_to_s8 : _fn_to_s4);
  enc.set_buffer(0, codes_in);
  enc.set_buffer(1, scales);
  enc.set_buffer(2, biases);
  enc.set_buffer(3, codes_out);
  enc.set_buffer(4, zb);
  enc.set_constant(5, K);
  enc.set_constant(6, N);
  enc.dispatch({(unsigned)(K / 64), (unsigned)N, 1}, {64, 1, 1});
  return true;
}

bool
I8GemmContext::encode_dequant_signed(ComputeEncoder& enc,
                                     const SharedBuffer& codes,
                                     const SharedBuffer& scales,
                                     const SharedBuffer& zb, int bits,
                                     const SharedBuffer& out, int N,
                                     int K) const
{
  if (!signed_available(bits) || N <= 0 || (K % 64) != 0 ||
      out.byte_size() < (std::size_t)N * (std::size_t)K * 2) {
    return false;
  }
  enc.set_function(bits == 8 ? _fn_dq_s8 : _fn_dq_s4);
  enc.set_buffer(0, codes);
  enc.set_buffer(1, scales);
  enc.set_buffer(2, zb);
  enc.set_buffer(3, out);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  enc.dispatch({(unsigned)(bits == 8 ? K / 4 : K / 8), (unsigned)N, 1},
               {64, 1, 1});
  return true;
}

void
I8GemmContext::encode_tuned_(ComputeEncoder& enc, const Pending& p, int M,
                             int KP, int S)
{
  const int N = p.N;
  // No bias while TIMING: it is one add per output, identical across
  // candidates, so including it would only add noise to the vote.
  if (p.route == Route::kNative) {
    encode_native_gemm_(enc, p.codes, p.scales, p.zb, p.bits, SharedBuffer{},
                        _tune_y, 0, M, N, KP, S);
    return;
  }
  if (p.route == Route::kSigned) {
    encode_signed_gemm_(enc, p.codes, p.scales, p.zb, p.bits, SharedBuffer{},
                        _tune_y, 0, M, N, KP, S);
    return;
  }
  const unsigned nx = (unsigned)(((N + 63) / 64) * 128);
  const int G = KP / _group;
  if (S <= 0) {
    enc.set_function(_fn_gemm);
    enc.set_buffer(0, _xq); enc.set_buffer(1, _wq);
    enc.set_buffer(2, _as); enc.set_buffer(3, _ws);
    enc.set_buffer(4, _tune_y);
    enc.set_constant(5, KP); enc.set_constant(6, N); enc.set_constant(7, M);
    enc.set_buffer(8, _ws); enc.set_constant(9, 0);
    enc.dispatch({nx, (unsigned)((M + 63) / 64), 1}, {128, 1, 1});
    return;
  }
  const int rows = rows_per_block_(S, N, M);
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
  std::vector<Pending> todo;
  todo.swap(_pending);
  for (const Pending& p : todo) {
    const int KP = p.route == Route::kRequant ? kpad_(p.K) : p.K;
    const int G = KP / 64;
    const int M = p.M < kTuneRows ? p.M : kTuneRows;
    const std::vector<int> cands = tune_candidates(M, p.N, p.K);
    // The scratches must still cover the shape. They are grow-only and
    // this shape ran, so they do -- unless release_scratch() intervened,
    // in which case the honest answer is to drop the shape rather than
    // measure a buffer that is not there.
    auto covers = [](const SharedBuffer& b, std::size_t bytes) {
      return !b.empty() && b.byte_size() >= bytes;
    };
    const std::size_t mk = (std::size_t)M * KP;
    const std::size_t mg = (std::size_t)M * G * 2;
    const std::size_t ng = (std::size_t)p.N * G * 2;
    bool have = false;
    switch (p.route) {
      case Route::kRequant:
        have = covers(_xq, mk) && covers(_wq, (std::size_t)p.N * KP);
        break;
      case Route::kNative:
        have = covers(_xu, mk) && covers(_as, mg) && covers(_xs, mg) &&
               covers(_uf, mg) && covers(_wsum, ng) && covers(_chi, ng) &&
               covers(_clo, ng) && !p.codes.empty();
        break;
      case Route::kSigned:
        have = covers(_xq, mk) && covers(_as, mg) && covers(_uf, mg) &&
               !p.codes.empty();
        break;
    }
    // The session's key for this shape's answer. Filed only when it IS
    // an answer: one candidate decides itself, but a shape dropped for
    // want of its scratch was never measured.
    const TuneKey skey = split_key_(route_name_((int)p.route), p.bits, p.N,
                                    p.K, p.M);
    if (cands.size() < 2 || !have) {
      _tuned.push_back(Tuned{p.N, p.K, m_key_(p.M), 0});
      if (cands.size() < 2) { store_tuned(_answers, skey, 0); }
      continue;
    }
    const std::size_t ybytes = (std::size_t)M * p.N * 2;
    if (_tune_y.empty() || _tune_y.byte_size() < ybytes) {
      _tune_y = mc->make_shared_buffer(ybytes);
    }
    int smax = 0;
    for (int S : cands) { smax = S > smax ? S : smax; }
    const std::size_t pbytes = (std::size_t)smax *
        (std::size_t)rows_per_block_(smax, p.N, M) * (std::size_t)p.N * 4u;
    if (_planes.empty() || _planes.byte_size() < pbytes) {
      _planes = mc->make_shared_buffer(pbytes);
    }
    if (_tune_y.empty() || _planes.empty()) {
      _tuned.push_back(Tuned{p.N, p.K, m_key_(p.M), 0});
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
        encode_tuned_(e, p, M, KP, S);
      });
    };
    const double one_ms = 1e3 * sample(0, 1);
    const int reps = one_ms <= 0.0 ? 1
        : std::clamp((int)std::ceil(kTuneSampleMs / one_ms), 1, 8);
    const int w = autotune_vote((int)cands.size(), /*rounds=*/3, reps,
        [&](int i) { return sample(cands[(std::size_t)i], reps); });
    _tuned.push_back(Tuned{p.N, p.K, m_key_(p.M), cands[(std::size_t)w]});
    store_tuned(_answers, skey, cands[(std::size_t)w]);
    ++_measured;
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
  _sc = {};
  _szb = {};
  // Pending records hold references to the weights they were recorded
  // over; a caller that never replays them (one tuning through tune())
  // must not pin a model's buffers past the generation that left them.
  _pending.clear();
}

}  // namespace genai
}  // namespace vpipe

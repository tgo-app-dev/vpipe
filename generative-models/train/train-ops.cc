#include "generative-models/train/train-ops.h"

#include <algorithm>
#include <cstdlib>

namespace vpipe {
namespace genai {
namespace train {

using metal_compute::ComputeEncoder;
using metal_compute::ComputeFunction;
using metal_compute::SharedBuffer;

namespace {

constexpr int kTg = 256;          // TRAIN_TG in train_ops.metal
constexpr int kReducePer = 16;    // elements a thread folds per partial

unsigned
ceil_div(std::size_t a, std::size_t b)
{
  return (unsigned)((a + b - 1) / b);
}

}  // namespace

bool
TrainOps::init(metal_compute::MetalCompute* mc, bool bf16, std::string* err)
{
  _mc = mc;
  _ok = false;
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (mc == nullptr) { return fail("train ops: no metal context"); }
  _lib_ops = mc->load_library(bf16 ? "train_ops_bf16" : "train_ops");
  _lib_gemm = mc->load_library(bf16 ? "train_gemm_bf16" : "train_gemm");
  _lib_attn = mc->load_library("attn_bwd");
  if (const char* e = std::getenv("VPIPE_TRAIN_ATTN_BWD")) {
    _attn_gemm = std::string(e) != "fused";
  }
  if (const char* e = std::getenv("VPIPE_TRAIN_ATTN_MMA")) {
    _attn_mma = std::atoi(e) != 0;
  }
  const char* tsuf = bf16 ? "bf16" : "f16";
  struct Want { ComputeFunction* fn; const metal_compute::ComputeLibrary* lib;
                std::string name; };
  const std::vector<Want> want = {
      {&_transpose, &_lib_ops, "train_transpose_2d"},
      {&_ln_mod_bwd, &_lib_ops, "train_ln_mod_bwd"},
      {&_gated_bwd, &_lib_ops, "train_gated_tanh_bwd"},
      {&_swiglu_bwd, &_lib_ops, "train_swiglu_bwd"},
      {&_rms_bwd, &_lib_ops, "train_rms_bwd"},
      {&_rope_bwd, &_lib_ops, "train_rope_bwd"},
      {&_hd_to_td, &_lib_ops, "train_hd_f32_to_td"},
      {&_rowdot, &_lib_ops, "train_attn_rowdot"},
      {&_lse, &_lib_ops, "train_attn_lse"},
      {&_flow_loss, &_lib_ops, "train_flow_loss"},
      {&_fill, &_lib_ops, "train_fill_f32"},
      {&_sqnorm, &_lib_ops, "train_sqnorm_partial"},
      {&_abs, &_lib_ops, "train_abs_partial"},
      {&_adamw, &_lib_ops, "train_adamw"},
      {&_lion, &_lib_ops, "train_lion"},
      {&_sgd, &_lib_ops, "train_sgd"},
      {&_prod_acc, &_lib_ops, "train_prodigy_accum"},
      {&_prod_step, &_lib_ops, "train_prodigy_step"},
      {&_ema, &_lib_ops, "train_ema"},
      {&_shadow, &_lib_ops, "train_shadow"},
      {&_tn_32x64, &_lib_gemm, "train_gemm_tn_f32_32x64"},
      {&_tn_64x32, &_lib_gemm, "train_gemm_tn_f32_64x32"},
      {&_nn_64x32, &_lib_gemm, "train_gemm_nn_64x32"},
      {&_nn_64x64, &_lib_gemm, "train_gemm_nn_64x64"},
      {&_nt_64x64, &_lib_gemm, "train_gemm_nt_64x64"},
      {&_dq, &_lib_attn, std::string("attn_bwd_dq_bd128_") + tsuf},
      {&_dkv, &_lib_attn, std::string("attn_bwd_dkv_bd128_") + tsuf},
      {&_pds, &_lib_attn, std::string("attn_bwd_pds_bd128_") + tsuf},
      {&_nn_f32_64x64, &_lib_gemm, "train_gemm_nn_f32_64x64"},
      {&_tn_64x64, &_lib_gemm, "train_gemm_tn_64x64"},
  };
  for (const Want& w : want) {
    *w.fn = w.lib->function(w.name);
    if (!w.fn->valid()) {
      // A kernel name that does not exist is a silent no-op dispatch
      // otherwise -- refuse at init, where the name is still in hand.
      return fail("train ops: kernel '" + w.name + "' did not load");
    }
  }
  // The matrix-core twins: optional, and only where there are matrix
  // cores -- a box without them keeps the steel GEMMs.
  _mma_nn_f32 = ComputeFunction();
  _mma_tn = ComputeFunction();
  _pds_mma = ComputeFunction();
  if (mc->supports_matrix_cores()) {
    _lib_gemm_mma = mc->load_library(bf16 ? "train_gemm_mma_bf16"
                                          : "train_gemm_mma");
    _mma_nn_f32 = _lib_gemm_mma.function("train_gemm_mma_nn_f32_64x128");
    _mma_tn = _lib_gemm_mma.function("train_gemm_mma_tn_64x128");
    _pds_mma = _lib_gemm_mma.function("attn_bwd_pds_mma_64x64");
  }
  _elt = 2;
  _ok = true;
  return true;
}

bool
TrainOps::attn_mma() const noexcept
{
  return _attn_mma && _mma_nn_f32.valid() && _mma_tn.valid();
}

namespace {

std::size_t
attn_budget_(std::size_t budget)
{
  if (budget > 0) { return budget; }
  std::size_t mb = 512;
  if (const char* e = std::getenv("VPIPE_TRAIN_ATTN_SCRATCH_MB")) {
    const long v = std::atol(e);
    if (v > 0) { mb = (std::size_t)v; }
  }
  return mb << 20;
}

// Heads per group: as many whole heads as the budget holds, at least one.
int
attn_group_(int H, int qL, int kL, std::size_t budget)
{
  const std::size_t per = (std::size_t)2 * qL * kL * 2;
  if (per == 0) { return H; }
  const std::size_t n = attn_budget_(budget) / per;
  return (int)std::max<std::size_t>(1, std::min<std::size_t>(n, H));
}

}  // namespace

std::size_t
TrainOps::attn_scratch_bytes(int H, int qL, int kL,
                             std::size_t budget) noexcept
{
  return (std::size_t)attn_group_(H, qL, kL, budget) * 2 * qL * kL * 2;
}

void
TrainOps::transpose(ComputeEncoder& enc, const SharedBuffer& in,
                    std::size_t ie, const SharedBuffer& out, std::size_t oe,
                    int R, int C, int ld_in, int ld_out, int col_off) const
{
  enc.set_function(_transpose);
  enc.set_buffer(0, in, ie * _elt);
  enc.set_buffer(1, out, oe * _elt);
  enc.set_constant(2, R); enc.set_constant(3, C);
  enc.set_constant(4, ld_in); enc.set_constant(5, ld_out);
  enc.set_constant(6, col_off);
  enc.dispatch({ceil_div((std::size_t)C, 32) * 32,
                ceil_div((std::size_t)R, 32) * 8, 1}, {32, 8, 1});
}

void
TrainOps::ln_mod_bwd(ComputeEncoder& enc, const SharedBuffer& x,
                     std::size_t xe, const SharedBuffer& dout, std::size_t de,
                     bool dout_f32, const SharedBuffer& scale, std::size_t se,
                     const SharedBuffer& dx, std::size_t dxe, int rows, int H,
                     float eps) const
{
  enc.set_function(_ln_mod_bwd);
  enc.set_buffer(0, x, xe * _elt);
  enc.set_buffer(1, dout, dout_f32 ? 0 : de * _elt);
  enc.set_buffer(2, dout, dout_f32 ? de * 4 : 0);
  enc.set_buffer(3, scale, se * _elt);
  enc.set_buffer(4, dx, dxe * 4);
  enc.set_constant(5, H);
  enc.set_constant(6, eps);
  enc.set_constant(7, (int)(dout_f32 ? 1 : 0));
  enc.dispatch({(unsigned)kTg, (unsigned)rows, 1}, {(unsigned)kTg, 1, 1});
}

void
TrainOps::gated_tanh_bwd(ComputeEncoder& enc, const SharedBuffer& dh,
                         std::size_t he, const SharedBuffer& gate,
                         std::size_t ge, const SharedBuffer& dsub,
                         std::size_t se, int rows, int N) const
{
  enc.set_function(_gated_bwd);
  enc.set_buffer(0, dh, he * 4);
  enc.set_buffer(1, gate, ge * _elt);
  enc.set_buffer(2, dsub, se * _elt);
  enc.set_constant(3, N);
  enc.set_constant(4, rows);
  enc.dispatch({(unsigned)N, (unsigned)rows, 1}, {256, 1, 1});
}

void
TrainOps::swiglu_bwd(ComputeEncoder& enc, const SharedBuffer& g,
                     const SharedBuffer& u, const SharedBuffer& ds,
                     const SharedBuffer& dg, std::size_t dge,
                     const SharedBuffer& du, std::size_t due, int rows,
                     int width, int ld) const
{
  const std::size_t n = (std::size_t)rows * width;
  enc.set_function(_swiglu_bwd);
  enc.set_buffer(0, g); enc.set_buffer(1, u); enc.set_buffer(2, ds);
  enc.set_buffer(3, dg, dge * _elt); enc.set_buffer(4, du, due * _elt);
  enc.set_constant(5, (int)n);
  enc.set_constant(6, width);
  enc.set_constant(7, ld);
  enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
}

void
TrainOps::rms_bwd(ComputeEncoder& enc, const SharedBuffer& x, std::size_t xe,
                  const SharedBuffer& dy, std::size_t de,
                  const SharedBuffer& w, const SharedBuffer& dx,
                  std::size_t dxe, int rows, int D, float eps, int heads,
                  int ld) const
{
  enc.set_function(_rms_bwd);
  enc.set_buffer(0, x, xe * _elt);
  enc.set_buffer(1, dy, de * _elt);
  enc.set_buffer(2, w);
  enc.set_buffer(3, dx, dxe * _elt);
  enc.set_constant(4, D);
  enc.set_constant(5, eps);
  enc.set_constant(6, rows);
  enc.set_constant(7, heads);
  enc.set_constant(8, ld);
  enc.dispatch({32, ceil_div((std::size_t)rows, 8) * 8, 1}, {32, 8, 1});
}

void
TrainOps::rope_bwd(ComputeEncoder& enc, const SharedBuffer& in, bool in_f32,
                   const SharedBuffer& out, const SharedBuffer& cosb,
                   const SharedBuffer& sinb, int tab_row, int H, int T,
                   int D) const
{
  enc.set_function(_rope_bwd);
  enc.set_buffer(0, in);
  enc.set_buffer(1, in);
  enc.set_buffer(2, out);
  const std::size_t tb = (std::size_t)tab_row * D * 4;
  enc.set_buffer(3, cosb, tb);
  enc.set_buffer(4, sinb, tb);
  enc.set_constant(5, H); enc.set_constant(6, T); enc.set_constant(7, D);
  enc.set_constant(8, (int)(in_f32 ? 1 : 0));
  enc.dispatch({(unsigned)(D / 2), (unsigned)T, (unsigned)H},
               {(unsigned)(D / 2), 1, 1});
}

void
TrainOps::hd_f32_to_td(ComputeEncoder& enc, const SharedBuffer& in,
                       const SharedBuffer& out, std::size_t oe, int H, int T,
                       int D, int ld) const
{
  enc.set_function(_hd_to_td);
  enc.set_buffer(0, in); enc.set_buffer(1, out, oe * _elt);
  enc.set_constant(2, H); enc.set_constant(3, T); enc.set_constant(4, D);
  enc.set_constant(5, ld);
  enc.dispatch({(unsigned)D, (unsigned)T, (unsigned)H},
               {(unsigned)D, 1, 1});
}

void
TrainOps::attn_backward(ComputeEncoder& enc, const SharedBuffer& q,
                        const SharedBuffer& k, const SharedBuffer& v,
                        const SharedBuffer& o, const SharedBuffer& d_o,
                        const SharedBuffer& ml_max,
                        const SharedBuffer& ml_sum, const SharedBuffer& lse2,
                        const SharedBuffer& dsum, const SharedBuffer& dq,
                        const SharedBuffer& dk32, const SharedBuffer& dv32,
                        int H, int T, int D, float scale,
                        const std::vector<AttnSegment>& segs) const
{
  // The per-row inputs: Dsum for every row, LSE2 per segment from the
  // statistics the forward exported for it.
  enc.set_function(_rowdot);
  enc.set_buffer(0, d_o); enc.set_buffer(1, o); enc.set_buffer(2, dsum);
  enc.set_constant(3, T); enc.set_constant(4, D);
  enc.dispatch({32, (unsigned)T, (unsigned)H}, {32, 1, 1});
  for (const AttnSegment& s : segs) {
    enc.set_function(_lse);
    const std::size_t mo = (std::size_t)H * s.q_row0 * 4;
    enc.set_buffer(0, ml_max, mo); enc.set_buffer(1, ml_sum, mo);
    enc.set_buffer(2, lse2);
    enc.set_constant(3, s.qL); enc.set_constant(4, T);
    enc.set_constant(5, s.q_row0);
    enc.dispatch({(unsigned)s.qL, (unsigned)H, 1}, {256, 1, 1});
  }
  const std::int64_t hs = (std::int64_t)T * D;
  if (_attn_gemm) {
    // Sized for the widest segment BEFORE anything is encoded: replacing
    // the scratch between segments would free a buffer the earlier
    // segments' dispatches are still queued against.
    std::size_t need = 0;
    for (const AttnSegment& s : segs) {
      need = std::max(need, attn_scratch_bytes(H, s.qL, s.kL, _attn_budget));
    }
    if (_attn_scratch.empty() || _attn_scratch.byte_size() < need) {
      _attn_scratch = _mc->make_shared_buffer(need);
    }
    for (const AttnSegment& s : segs) {
      const int hg = attn_group_(H, s.qL, s.kL, _attn_budget);
      const std::size_t plane = (std::size_t)s.kL * s.qL;
      const std::size_t ds_off = (std::size_t)hg * plane * _elt;   // bytes
      for (int h0 = 0; h0 < H; h0 += hg) {
        const int n = std::min(hg, H - h0);
        AttnBwdParams p;
        p.H = H; p.qL = s.qL; p.kL = s.kL; p.scale = scale;
        p.qL_off = s.qL_off; p.causal = s.causal ? 1 : 0;
        p.T = T; p.q_row0 = s.q_row0;
        p.q_hstride = hs; p.k_hstride = hs; p.acc_hstride = hs;
        p.h0 = h0; p.ldp = s.qL; p.p_hstride = (std::int64_t)plane;
        const std::size_t qo = (std::size_t)s.q_row0 * D * _elt;
        // The matrix cores' tile pass where there are some, else the
        // key-block pass on simdgroup matrices. Same buffers.
        const bool mma = attn_mma() && _pds_mma.valid();
        enc.set_function(mma ? _pds_mma : _pds);
        enc.set_buffer(0, q, qo); enc.set_buffer(1, k); enc.set_buffer(2, v);
        enc.set_buffer(3, d_o, qo);
        enc.set_buffer(4, _attn_scratch);
        enc.set_buffer(5, _attn_scratch, ds_off);
        enc.set_buffer(6, lse2); enc.set_buffer(7, dsum);
        enc.set_constant(8, p);
        if (mma) {
          enc.dispatch({ceil_div((std::size_t)s.qL, 64) * 128,
                        ceil_div((std::size_t)s.kL, 64), (unsigned)n},
                       {128, 1, 1});
        } else {
          enc.dispatch({32u * ceil_div((std::size_t)s.kL, 32),
                        4u * (unsigned)n, 1}, {32, 4, 1});
        }
        // Each head's rows of the segment, and of the whole sequence.
        const std::size_t hq = ((std::size_t)h0 * T + s.q_row0) * D;
        const std::size_t hk = (std::size_t)h0 * T * D;
        // dV += P^T dO and dK += dS^T Q: [kL, qL] x [qL, D].
        attn_gemm_(enc, _attn_scratch, 0, d_o, hq * _elt, dv32, hk * 4,
                   /*nn_f32=*/true, s.kL, D, s.qL, s.qL, D, D,
                   (std::int64_t)plane, hs, hs, n);
        attn_gemm_(enc, _attn_scratch, ds_off, q, hq * _elt, dk32, hk * 4,
                   true, s.kL, D, s.qL, s.qL, D, D, (std::int64_t)plane, hs,
                   hs, n);
        // dQ = dS K, from dS^T stored [kL][qL]: [qL, kL] x [kL, D].
        attn_gemm_(enc, _attn_scratch, ds_off, k, hk * _elt, dq, hq * _elt,
                   /*nn_f32=*/false, s.qL, D, s.kL, s.qL, D, D,
                   (std::int64_t)plane, hs, hs, n);
      }
    }
    return;
  }
  for (const AttnSegment& s : segs) {
    AttnBwdParams p;
    p.H = H; p.qL = s.qL; p.kL = s.kL; p.scale = scale;
    p.qL_off = s.qL_off; p.causal = s.causal ? 1 : 0;
    p.T = T; p.q_row0 = s.q_row0;
    p.q_hstride = hs; p.k_hstride = hs; p.acc_hstride = hs;
    const std::size_t qo = (std::size_t)s.q_row0 * D * _elt;
    enc.set_function(_dq);
    enc.set_buffer(0, q, qo); enc.set_buffer(1, k); enc.set_buffer(2, v);
    enc.set_buffer(3, d_o, qo); enc.set_buffer(4, dq, qo);
    enc.set_buffer(5, lse2); enc.set_buffer(6, dsum);
    enc.set_constant(7, p);
    enc.dispatch({32u * ceil_div((std::size_t)s.qL, 32), 4u * (unsigned)H, 1},
                 {32, 4, 1});
    enc.set_function(_dkv);
    enc.set_buffer(0, q, qo); enc.set_buffer(1, k); enc.set_buffer(2, v);
    enc.set_buffer(3, d_o, qo);
    enc.set_buffer(4, dk32); enc.set_buffer(5, dv32);
    enc.set_buffer(6, lse2); enc.set_buffer(7, dsum);
    enc.set_constant(8, p);
    enc.dispatch({32u * ceil_div((std::size_t)s.kL, 32), 4u * (unsigned)H, 1},
                 {32, 4, 1});
  }
}

namespace {

void
gemm_(ComputeEncoder& enc, const ComputeFunction& fn, int bm, int bn,
      const SharedBuffer& a, std::size_t ab, const SharedBuffer& b,
      std::size_t bb, const SharedBuffer* ct, std::size_t cb,
      const SharedBuffer* cf, const GemmParams& p)
{
  enc.set_function(fn);
  enc.set_buffer(0, a, ab);
  enc.set_buffer(1, b, bb);
  // The kernel binds both outputs; the unused one only needs to be a
  // valid buffer.
  enc.set_buffer(2, ct != nullptr ? *ct : *cf, ct != nullptr ? cb : 0);
  enc.set_buffer(3, cf != nullptr ? *cf : *ct, cf != nullptr ? cb : 0);
  enc.set_constant(4, p);
  enc.dispatch({ceil_div((std::size_t)p.N, (std::size_t)bn) * 128,
                ceil_div((std::size_t)p.M, (std::size_t)bm), 1},
               {128, 1, 1});
}

}  // namespace

void
TrainOps::attn_gemm_(ComputeEncoder& enc, const SharedBuffer& a,
                     std::size_t ab, const SharedBuffer& b, std::size_t bb,
                     const SharedBuffer& c, std::size_t cb, bool f32_acc,
                     int M, int N, int K, int lda, int ldb, int ldc,
                     std::int64_t a_bs, std::int64_t b_bs, std::int64_t c_bs,
                     int batch) const
{
  // Two shapes only: NN added into an f32 accumulator (dK, dV), and TN
  // into the storage type (dQ).
  GemmParams p;
  p.M = M; p.N = N; p.K = K;
  p.lda = lda; p.ldb = ldb; p.ldc = ldc;
  p.alpha = 1.0f; p.accumulate = f32_acc ? 1 : 0;
  p.a_bstride = a_bs; p.b_bstride = b_bs; p.c_bstride = c_bs;
  const bool mma = attn_mma();
  const ComputeFunction& fn =
      mma ? (f32_acc ? _mma_nn_f32 : _mma_tn)
          : (f32_acc ? _nn_f32_64x64 : _tn_64x64);
  enc.set_function(fn);
  enc.set_buffer(0, a, ab);
  enc.set_buffer(1, b, bb);
  // Both outputs are bound; the unused one only needs to be valid.
  enc.set_buffer(2, c, f32_acc ? 0 : cb);
  enc.set_buffer(3, c, f32_acc ? cb : 0);
  enc.set_constant(4, p);
  if (mma) {
    // 64 x 128 tiles, four simdgroups each.
    enc.dispatch({ceil_div((std::size_t)N, 128) * 128,
                  ceil_div((std::size_t)M, 64), (unsigned)batch},
                 {128, 1, 1});
  } else {
    enc.dispatch({ceil_div((std::size_t)N, 64) * 128,
                  ceil_div((std::size_t)M, 64), (unsigned)batch},
                 {128, 1, 1});
  }
}

void
TrainOps::gemm_tn_f32(ComputeEncoder& enc, const SharedBuffer& a,
                      std::size_t ae, const SharedBuffer& b, std::size_t be,
                      const SharedBuffer& c, std::size_t ce, int M, int P,
                      int Q, int lda, int ldb, int ldc, float alpha,
                      bool accumulate) const
{
  GemmParams p;
  p.M = P; p.N = Q; p.K = M;
  p.lda = lda; p.ldb = ldb; p.ldc = ldc;
  p.alpha = alpha; p.accumulate = accumulate ? 1 : 0;
  // Few output rows (a rank-wide A^T) want the short, wide tile.
  const bool wide = P <= Q;
  gemm_(enc, wide ? _tn_32x64 : _tn_64x32, wide ? 32 : 64, wide ? 64 : 32,
        a, ae * _elt, b, be * _elt, nullptr, ce * 4, &c, p);
}

void
TrainOps::gemm_nn(ComputeEncoder& enc, const SharedBuffer& a, std::size_t ae,
                  const SharedBuffer& b, std::size_t be, const SharedBuffer& c,
                  std::size_t ce, int M, int N, int K, int lda, int ldb,
                  int ldc, float alpha, bool accumulate) const
{
  GemmParams p;
  p.M = M; p.N = N; p.K = K;
  p.lda = lda; p.ldb = ldb; p.ldc = ldc;
  p.alpha = alpha; p.accumulate = accumulate ? 1 : 0;
  const bool narrow = N <= 32;
  gemm_(enc, narrow ? _nn_64x32 : _nn_64x64, 64, narrow ? 32 : 64, a,
        ae * _elt, b, be * _elt, &c, ce * _elt, nullptr, p);
}

void
TrainOps::gemm_nt(ComputeEncoder& enc, const SharedBuffer& a, std::size_t ae,
                  const SharedBuffer& b, std::size_t be, const SharedBuffer& c,
                  std::size_t ce, int M, int N, int K, int lda, int ldb,
                  int ldc, float alpha, bool accumulate) const
{
  GemmParams p;
  p.M = M; p.N = N; p.K = K;
  p.lda = lda; p.ldb = ldb; p.ldc = ldc;
  p.alpha = alpha; p.accumulate = accumulate ? 1 : 0;
  gemm_(enc, _nt_64x64, 64, 64, a, ae * _elt, b, be * _elt, &c, ce * _elt,
        nullptr, p);
}

int
TrainOps::flow_loss_groups(int n) noexcept
{
  return (int)ceil_div((std::size_t)n, kTg);
}

int
TrainOps::flow_loss(ComputeEncoder& enc, const SharedBuffer& pred,
                    const SharedBuffer& x0, const SharedBuffer& noise,
                    const SharedBuffer& dpred, const SharedBuffer& partial,
                    int n, float coef) const
{
  enc.set_function(_flow_loss);
  enc.set_buffer(0, pred); enc.set_buffer(1, x0); enc.set_buffer(2, noise);
  enc.set_buffer(3, dpred); enc.set_buffer(4, partial);
  enc.set_constant(5, n); enc.set_constant(6, coef);
  const int groups = flow_loss_groups(n);
  enc.dispatch({(unsigned)groups * kTg, 1, 1}, {(unsigned)kTg, 1, 1});
  return groups;
}

void
TrainOps::fill_f32(ComputeEncoder& enc, const SharedBuffer& buf,
                   std::size_t e, float value, std::size_t n) const
{
  if (n == 0) { return; }
  enc.set_function(_fill);
  enc.set_buffer(0, buf, e * 4);
  enc.set_constant(1, value);
  enc.set_constant(2, (int)n);
  enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
}

int
TrainOps::reduce_groups(std::size_t n) noexcept
{
  return (int)ceil_div(n, (std::size_t)kTg * kReducePer);
}

int
TrainOps::sqnorm_partial(ComputeEncoder& enc, const SharedBuffer& x,
                         std::size_t xe, std::size_t n,
                         const SharedBuffer& part, std::size_t pe) const
{
  const int groups = reduce_groups(n);
  if (groups == 0) { return 0; }
  enc.set_function(_sqnorm);
  enc.set_buffer(0, x, xe * 4);
  enc.set_buffer(1, part, pe * 4);
  enc.set_constant(2, (int)n);
  enc.set_constant(3, kReducePer);
  enc.dispatch({(unsigned)groups * kTg, 1, 1}, {(unsigned)kTg, 1, 1});
  return groups;
}

int
TrainOps::abs_partial(ComputeEncoder& enc, const SharedBuffer& x,
                      std::size_t xe, std::size_t n, const SharedBuffer& part,
                      std::size_t pe) const
{
  const int groups = reduce_groups(n);
  if (groups == 0) { return 0; }
  enc.set_function(_abs);
  enc.set_buffer(0, x, xe * 4);
  enc.set_buffer(1, part, pe * 4);
  enc.set_constant(2, (int)n);
  enc.set_constant(3, kReducePer);
  enc.dispatch({(unsigned)groups * kTg, 1, 1}, {(unsigned)kTg, 1, 1});
  return groups;
}

void
TrainOps::adamw(ComputeEncoder& enc, const SharedBuffer& p,
                const SharedBuffer& g, const SharedBuffer& m,
                const SharedBuffer& v, const SharedBuffer& shadow,
                const OptParams& o) const
{
  if (o.n <= 0) { return; }
  enc.set_function(_adamw);
  enc.set_buffer(0, p); enc.set_buffer(1, g); enc.set_buffer(2, m);
  enc.set_buffer(3, v); enc.set_buffer(4, shadow);
  enc.set_constant(5, o);
  enc.dispatch({(unsigned)o.n, 1, 1}, {256, 1, 1});
}

void
TrainOps::lion(ComputeEncoder& enc, const SharedBuffer& p,
               const SharedBuffer& g, const SharedBuffer& m,
               const SharedBuffer& shadow, const OptParams& o) const
{
  if (o.n <= 0) { return; }
  enc.set_function(_lion);
  enc.set_buffer(0, p); enc.set_buffer(1, g); enc.set_buffer(2, m);
  enc.set_buffer(3, shadow);
  enc.set_constant(4, o);
  enc.dispatch({(unsigned)o.n, 1, 1}, {256, 1, 1});
}

void
TrainOps::sgd(ComputeEncoder& enc, const SharedBuffer& p,
              const SharedBuffer& g, const SharedBuffer& m,
              const SharedBuffer& shadow, const OptParams& o) const
{
  if (o.n <= 0) { return; }
  enc.set_function(_sgd);
  enc.set_buffer(0, p); enc.set_buffer(1, g); enc.set_buffer(2, m);
  enc.set_buffer(3, shadow);
  enc.set_constant(4, o);
  enc.dispatch({(unsigned)o.n, 1, 1}, {256, 1, 1});
}

void
TrainOps::prodigy_accum(ComputeEncoder& enc, const SharedBuffer& p,
                        const SharedBuffer& g, const SharedBuffer& x0,
                        const SharedBuffer& m, const SharedBuffer& v,
                        const SharedBuffer& s, const SharedBuffer& num,
                        const OptParams& o) const
{
  if (o.n <= 0) { return; }
  enc.set_function(_prod_acc);
  enc.set_buffer(0, p); enc.set_buffer(1, g); enc.set_buffer(2, x0);
  enc.set_buffer(3, m); enc.set_buffer(4, v); enc.set_buffer(5, s);
  enc.set_buffer(6, num);
  enc.set_constant(7, o);
  enc.dispatch({(unsigned)o.n, 1, 1}, {256, 1, 1});
}

void
TrainOps::prodigy_step(ComputeEncoder& enc, const SharedBuffer& p,
                       const SharedBuffer& g, const SharedBuffer& m,
                       const SharedBuffer& v, const SharedBuffer& shadow,
                       const OptParams& o) const
{
  if (o.n <= 0) { return; }
  enc.set_function(_prod_step);
  enc.set_buffer(0, p); enc.set_buffer(1, g); enc.set_buffer(2, m);
  enc.set_buffer(3, v); enc.set_buffer(4, shadow);
  enc.set_constant(5, o);
  enc.dispatch({(unsigned)o.n, 1, 1}, {256, 1, 1});
}

void
TrainOps::ema(ComputeEncoder& enc, const SharedBuffer& p,
              const SharedBuffer& e, float decay, std::size_t n) const
{
  if (n == 0) { return; }
  enc.set_function(_ema);
  enc.set_buffer(0, p); enc.set_buffer(1, e);
  enc.set_constant(2, decay);
  enc.set_constant(3, (int)n);
  enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
}

void
TrainOps::shadow(ComputeEncoder& enc, const SharedBuffer& p,
                 const SharedBuffer& out, float mul, std::size_t n) const
{
  if (n == 0) { return; }
  enc.set_function(_shadow);
  enc.set_buffer(0, p); enc.set_buffer(1, out);
  enc.set_constant(2, mul);
  enc.set_constant(3, (int)n);
  enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
}

}  // namespace train
}  // namespace genai
}  // namespace vpipe

#ifndef VPIPE_GENERATIVE_MODELS_TRAIN_TRAIN_OPS_H
#define VPIPE_GENERATIVE_MODELS_TRAIN_TRAIN_OPS_H

// The GPU operations LoRA training adds to a DiT's forward: the backward
// of each op a block runs, the GEMM shapes the forward has no kernel for,
// the flash attention's backward, the flow-matching loss, the optimizer
// updates and the reductions they need.
//
// FROZEN BASE. A LoRA trains only its own factors, so every backward here
// produces the gradient with respect to an op's INPUT and nothing else:
// no base-weight gradient is ever formed. The large backward products
// (dX = dY W) are NOT here either -- a family hands its forward's own GEMM
// route the transposed weight (transpose()), which keeps the steel tiles,
// the matrix cores, int8 and the ANE for the backward too.
//
// Encoders only: nothing here commits. Offsets are in ELEMENTS of the
// buffer's own type -- the storage type (2 bytes) for activations, f32
// for gradients' accumulators and statistics.
//
// Deterministic: no atomics; every reduction writes one partial per
// threadgroup, folded by the caller in index order.

#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {
namespace train {

// Mirrors of the kernels' parameter blocks (train_ops.metal,
// train_gemm.metal, attn_bwd.metal). Layout must match exactly.
struct OptParams {
  float lr = 0.0f;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float beta3 = 0.0f;
  float eps = 1e-8f;
  float weight_decay = 0.0f;
  float bc1 = 1.0f;
  float bc2 = 1.0f;
  float grad_scale = 1.0f;
  float mul = 1.0f;
  float d = 1.0f;
  float d0 = 1.0f;
  float dlr = 0.0f;
  std::int32_t n = 0;
};

struct GemmParams {
  std::int32_t M = 0, N = 0, K = 0;
  std::int32_t lda = 0, ldb = 0, ldc = 0;
  float alpha = 1.0f;
  std::int32_t accumulate = 0;
  std::int64_t a_bstride = 0, b_bstride = 0, c_bstride = 0;
};

struct AttnBwdParams {
  std::int32_t H = 0;
  std::int32_t qL = 0;
  std::int32_t kL = 0;
  float scale = 0.0f;
  std::int32_t qL_off = 0;
  std::int32_t causal = 0;
  std::int32_t T = 0;
  std::int32_t q_row0 = 0;
  std::int64_t q_hstride = 0;
  std::int64_t k_hstride = 0;
  std::int64_t acc_hstride = 0;
  std::int32_t h0 = 0;
  std::int32_t ldp = 0;
  std::int64_t p_hstride = 0;
};

// One dispatch group of a block-causal attention, as the forward ran it:
// query rows [q_row0, q_row0 + qL) against keys [0, kL). A causal group
// lets its local query row i see key j only when i + qL_off >= j, with
// `qL_off` the offset the forward passed its causal kernel.
struct AttnSegment {
  int  q_row0 = 0;
  int  qL = 0;
  int  kL = 0;
  int  qL_off = 0;
  bool causal = false;
};

class TrainOps {
 public:
  // `bf16` picks the storage type of every activation: the bf16 libraries
  // for the DiTs that train today, the f16 ones otherwise.
  bool init(metal_compute::MetalCompute* mc, bool bf16, std::string* err);
  bool valid() const noexcept { return _ok; }
  metal_compute::MetalCompute* mc() const noexcept { return _mc; }

  // in [R, C] (row stride ld_in) -> out[c * ld_out + col_off + r].
  void transpose(metal_compute::ComputeEncoder& enc,
                 const metal_compute::SharedBuffer& in, std::size_t ie,
                 const metal_compute::SharedBuffer& out, std::size_t oe,
                 int R, int C, int ld_in, int ld_out, int col_off) const;

  // dx (f32) += the backward of LN(x) * (1 + scale) given dout, which is
  // the storage type or f32 (dout_f32). `rows` rows of width H.
  void ln_mod_bwd(metal_compute::ComputeEncoder& enc,
                  const metal_compute::SharedBuffer& x, std::size_t xe,
                  const metal_compute::SharedBuffer& dout, std::size_t de,
                  bool dout_f32,
                  const metal_compute::SharedBuffer& scale, std::size_t se,
                  const metal_compute::SharedBuffer& dx, std::size_t dxe,
                  int rows, int H, float eps) const;

  // dsub = tanh(gate[n]) * dh (f32 in, storage type out).
  void gated_tanh_bwd(metal_compute::ComputeEncoder& enc,
                      const metal_compute::SharedBuffer& dh, std::size_t he,
                      const metal_compute::SharedBuffer& gate, std::size_t ge,
                      const metal_compute::SharedBuffer& dsub,
                      std::size_t se, int rows, int N) const;

  // SwiGLU backward: g, u, ds [rows, width] contiguous; dg and du written
  // with row stride `ld` from their own offsets (dge / due), so they can
  // be the two column halves of one [rows, 2 * width] matrix.
  void swiglu_bwd(metal_compute::ComputeEncoder& enc,
                  const metal_compute::SharedBuffer& g,
                  const metal_compute::SharedBuffer& u,
                  const metal_compute::SharedBuffer& ds,
                  const metal_compute::SharedBuffer& dg, std::size_t dge,
                  const metal_compute::SharedBuffer& du, std::size_t due,
                  int rows, int width, int ld) const;

  // RMSNorm (frozen weight w[D]) backward over `rows` rows of D. Row r is
  // head r % heads of token r / heads, written at token * ld + head * D
  // from dxe (heads = 1, ld = D: contiguous).
  void rms_bwd(metal_compute::ComputeEncoder& enc,
               const metal_compute::SharedBuffer& x, std::size_t xe,
               const metal_compute::SharedBuffer& dy, std::size_t de,
               const metal_compute::SharedBuffer& w,
               const metal_compute::SharedBuffer& dx, std::size_t dxe,
               int rows, int D, float eps, int heads, int ld) const;

  // Interleaved-pair RoPE backward: head-major [H][T][D] in (storage type,
  // or f32 when in_f32) -> token-major [T][H][D] out, rotated by -theta.
  // The tables are the forward's [T, D] f32 cos/sin, bound at `tab_row`.
  void rope_bwd(metal_compute::ComputeEncoder& enc,
                const metal_compute::SharedBuffer& in, bool in_f32,
                const metal_compute::SharedBuffer& out,
                const metal_compute::SharedBuffer& cosb,
                const metal_compute::SharedBuffer& sinb, int tab_row,
                int H, int T, int D) const;

  // f32 head-major [H][T][D] -> storage-type token-major rows of stride
  // `ld` (H * D for a plain [T][H*D]) from element oe.
  void hd_f32_to_td(metal_compute::ComputeEncoder& enc,
                    const metal_compute::SharedBuffer& in,
                    const metal_compute::SharedBuffer& out, std::size_t oe,
                    int H, int T, int D, int ld) const;

  // ---- attention backward ----------------------------------------------
  //
  // Everything head-major [H][T][D] over the whole joint sequence of T
  // rows; the forward's exported statistics ml_max / ml_sum hold, for
  // each segment, an [H][qL] block starting at element H * q_row0.
  // dQ is written for every row; dK and dV (f32) must be ZEROED by the
  // caller -- they accumulate across segments.
  //
  // Two forms, the same maths. FUSED: two flash kernels, seven matmul
  // units per segment, no scratch. GEMM (the default): one kernel stores
  // P^T and dS^T and three batched GEMMs finish the job -- five units,
  // three of them on the box's fastest GEMM (matrix cores where there
  // are some) -- for two [kL][qL] planes per head of scratch, bounded by
  // running the heads in groups (attn_scratch_bytes). VPIPE_TRAIN_ATTN_BWD
  // = fused | gemm picks, VPIPE_TRAIN_ATTN_SCRATCH_MB bounds the scratch
  // (512), VPIPE_TRAIN_ATTN_MMA=0 keeps the GEMMs off the matrix cores.
  void attn_backward(metal_compute::ComputeEncoder& enc,
                     const metal_compute::SharedBuffer& q,
                     const metal_compute::SharedBuffer& k,
                     const metal_compute::SharedBuffer& v,
                     const metal_compute::SharedBuffer& o,
                     const metal_compute::SharedBuffer& d_o,
                     const metal_compute::SharedBuffer& ml_max,
                     const metal_compute::SharedBuffer& ml_sum,
                     const metal_compute::SharedBuffer& lse2,
                     const metal_compute::SharedBuffer& dsum,
                     const metal_compute::SharedBuffer& dq,
                     const metal_compute::SharedBuffer& dk32,
                     const metal_compute::SharedBuffer& dv32,
                     int H, int T, int D, float scale,
                     const std::vector<AttnSegment>& segs) const;

  void set_attn_gemm(bool on) noexcept { _attn_gemm = on; }
  bool attn_gemm() const noexcept { return _attn_gemm; }
  void set_attn_mma(bool on) noexcept { _attn_mma = on; }
  // Matrix-core GEMMs loaded and allowed.
  bool attn_mma() const noexcept;
  void set_attn_scratch_budget(std::size_t bytes) noexcept {
    _attn_budget = bytes;
  }
  // What the GEMM form holds for a segment of qL query rows against kL
  // keys, over H heads, within `budget` bytes (0: the default).
  static std::size_t attn_scratch_bytes(int H, int qL, int kL,
                                        std::size_t budget = 0) noexcept;

  // ---- GEMMs --------------------------------------------------------------
  //
  // C[P, Q] (f32, ldc) (+)= alpha * A[M, P]^T B[M, Q]: reduction over the
  // rows. lda / ldb are the row strides of A and B.
  void gemm_tn_f32(metal_compute::ComputeEncoder& enc,
                   const metal_compute::SharedBuffer& a, std::size_t ae,
                   const metal_compute::SharedBuffer& b, std::size_t be,
                   const metal_compute::SharedBuffer& c, std::size_t ce,
                   int M, int P, int Q, int lda, int ldb, int ldc,
                   float alpha, bool accumulate) const;
  // C[M, N] (storage type) (+)= alpha * A[M, K] B[K, N].
  void gemm_nn(metal_compute::ComputeEncoder& enc,
               const metal_compute::SharedBuffer& a, std::size_t ae,
               const metal_compute::SharedBuffer& b, std::size_t be,
               const metal_compute::SharedBuffer& c, std::size_t ce,
               int M, int N, int K, int lda, int ldb, int ldc,
               float alpha, bool accumulate) const;
  // C[M, N] (storage type) (+)= alpha * A[M, K] B[N, K]^T.
  void gemm_nt(metal_compute::ComputeEncoder& enc,
               const metal_compute::SharedBuffer& a, std::size_t ae,
               const metal_compute::SharedBuffer& b, std::size_t be,
               const metal_compute::SharedBuffer& c, std::size_t ce,
               int M, int N, int K, int lda, int ldb, int ldc,
               float alpha, bool accumulate) const;

  // ---- loss and reductions ------------------------------------------------

  // dpred = coef * (pred - (noise - x0)); partial[g] = sum of the squared
  // error over group g. Returns the group count the caller folds.
  int flow_loss(metal_compute::ComputeEncoder& enc,
                const metal_compute::SharedBuffer& pred,
                const metal_compute::SharedBuffer& x0,
                const metal_compute::SharedBuffer& noise,
                const metal_compute::SharedBuffer& dpred,
                const metal_compute::SharedBuffer& partial,
                int n, float coef) const;
  static int flow_loss_groups(int n) noexcept;

  void fill_f32(metal_compute::ComputeEncoder& enc,
                const metal_compute::SharedBuffer& buf, std::size_t e,
                float value, std::size_t n) const;
  // Partial sums of x^2 (or |x|) over n f32 elements into partial[pe..].
  // Returns the group count.
  int sqnorm_partial(metal_compute::ComputeEncoder& enc,
                     const metal_compute::SharedBuffer& x, std::size_t xe,
                     std::size_t n, const metal_compute::SharedBuffer& part,
                     std::size_t pe) const;
  int abs_partial(metal_compute::ComputeEncoder& enc,
                  const metal_compute::SharedBuffer& x, std::size_t xe,
                  std::size_t n, const metal_compute::SharedBuffer& part,
                  std::size_t pe) const;
  static int reduce_groups(std::size_t n) noexcept;

  // ---- optimizer updates, over one flat arena of o.n elements -------------
  //
  // Each zeroes the gradient it consumed and rewrites the storage-type
  // shadow as ELT(ELT(p) * o.mul).
  void adamw(metal_compute::ComputeEncoder& enc,
             const metal_compute::SharedBuffer& p,
             const metal_compute::SharedBuffer& g,
             const metal_compute::SharedBuffer& m,
             const metal_compute::SharedBuffer& v,
             const metal_compute::SharedBuffer& shadow,
             const OptParams& o) const;
  void lion(metal_compute::ComputeEncoder& enc,
            const metal_compute::SharedBuffer& p,
            const metal_compute::SharedBuffer& g,
            const metal_compute::SharedBuffer& m,
            const metal_compute::SharedBuffer& shadow,
            const OptParams& o) const;
  void sgd(metal_compute::ComputeEncoder& enc,
           const metal_compute::SharedBuffer& p,
           const metal_compute::SharedBuffer& g,
           const metal_compute::SharedBuffer& m,
           const metal_compute::SharedBuffer& shadow,
           const OptParams& o) const;
  void prodigy_accum(metal_compute::ComputeEncoder& enc,
                     const metal_compute::SharedBuffer& p,
                     const metal_compute::SharedBuffer& g,
                     const metal_compute::SharedBuffer& x0,
                     const metal_compute::SharedBuffer& m,
                     const metal_compute::SharedBuffer& v,
                     const metal_compute::SharedBuffer& s,
                     const metal_compute::SharedBuffer& num,
                     const OptParams& o) const;
  void prodigy_step(metal_compute::ComputeEncoder& enc,
                    const metal_compute::SharedBuffer& p,
                    const metal_compute::SharedBuffer& g,
                    const metal_compute::SharedBuffer& m,
                    const metal_compute::SharedBuffer& v,
                    const metal_compute::SharedBuffer& shadow,
                    const OptParams& o) const;
  void ema(metal_compute::ComputeEncoder& enc,
           const metal_compute::SharedBuffer& p,
           const metal_compute::SharedBuffer& e, float decay,
           std::size_t n) const;
  void shadow(metal_compute::ComputeEncoder& enc,
              const metal_compute::SharedBuffer& p,
              const metal_compute::SharedBuffer& out, float mul,
              std::size_t n) const;

 private:
  metal_compute::MetalCompute* _mc = nullptr;
  bool _ok = false;
  std::size_t _elt = 2;
  metal_compute::ComputeLibrary _lib_ops, _lib_gemm, _lib_attn,
      _lib_gemm_mma;
  metal_compute::ComputeFunction _transpose, _ln_mod_bwd, _gated_bwd,
      _swiglu_bwd, _rms_bwd, _rope_bwd, _hd_to_td, _rowdot, _lse,
      _flow_loss, _fill, _sqnorm, _abs, _adamw, _lion, _sgd, _prod_acc,
      _prod_step, _ema, _shadow, _tn_32x64, _tn_64x32, _nn_64x32,
      _nn_64x64, _nt_64x64, _dq, _dkv, _pds, _nn_f32_64x64, _tn_64x64,
      _mma_nn_f32, _mma_tn, _pds_mma;
  bool _attn_gemm = true;
  bool _attn_mma = true;
  std::size_t _attn_budget = 0;
  mutable metal_compute::SharedBuffer _attn_scratch;

  void attn_gemm_(metal_compute::ComputeEncoder& enc,
                  const metal_compute::SharedBuffer& a, std::size_t ab,
                  const metal_compute::SharedBuffer& b, std::size_t bb,
                  const metal_compute::SharedBuffer& c, std::size_t cb,
                  bool f32_acc, int M, int N, int K, int lda, int ldb,
                  int ldc, std::int64_t a_bs, std::int64_t b_bs,
                  std::int64_t c_bs, int batch) const;
};

}  // namespace train
}  // namespace genai
}  // namespace vpipe

#endif

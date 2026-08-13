#ifndef VPIPE_GENERATIVE_MODELS_SHARED_I8_GEMM_H
#define VPIPE_GENERATIVE_MODELS_SHARED_I8_GEMM_H

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cstddef>

namespace vpipe {
namespace genai {

// Dynamic-int8 accelerated GEMM ("accelerated mode" for large matmuls --
// DiT blocks, LLM prefill): the f16/bf16 activation is quantized ON THE FLY
// to i8 with per-(row, 512-group) scales (quant_f16_i8_row_g512), the
// f16/bf16 weight (dense, or a dequant scratch) is quantized per-(out-
// channel, 512-group) into a reusable i8 scratch, and the product runs on
// the matrix units' int8 pipe -- ~2x the matmul2d rate at qualifying shapes
// -- accumulating each 512-deep group in f32 with its own scales
// (gemm_i8i8_sc_f16_n64_g512) and storing the element type back.
//
// The activation/weight/scale/output element type is chosen by `bf16`: the
// f16 caller loads the `dense_gemm_mma`/`affine_dequant` metallibs, the bf16
// caller (e.g. the FLUX.2 klein DiT, whose residual stream overflows f16) the
// `_bf16` twins -- the same kernels compiled with VPIPE_ELT=bfloat. The int8
// scratch is format-independent and the scale scratch is 2 bytes either way,
// so only the library selection changes. Reading a bf16 weight/activation
// through the f16 kernels (or vice versa) reinterprets the bits -> garbage;
// the caller MUST match `bf16` to its buffers' element type.
//
// LOSSY: int8 quantization, rel-L2 ~1e-2 per GEMM. Strictly OPT-IN and
// never part of a token-exact path. The per-call weight re-quant costs
// one extra pass over the weight (~1-2% of a qualifying GEMM), so the
// mode has NO persistent memory cost and composes with every checkpoint
// format that already produces an f16/bf16 weight for the matmul.
//
// The stage/model switch arrives via `want`; env VPIPE_I8_GEMM=0|1
// overrides either way (A/B), VPIPE_I8_GEMM_MIN_M tunes the M gate.
class I8GemmContext {
 public:
  I8GemmContext(metal_compute::MetalCompute* mc, bool want, bool bf16 = false);

  bool enabled() const { return _on; }

  // Shape gate: the win regime is big-M compute-bound GEMMs (measured
  // crossover ~1k rows on M5). K must split into whole 512-groups, OR the
  // padding quantizer must be available to make it so -- see kpad_() -- and
  // the padding must be CHEAP, which is the last clause.
  //
  // Padding is exact but not free: it is (KP-K)/K extra int8 MACs on every
  // call, and the int8 rate has to beat bf16 by more than that for the mode
  // to still pay. Capped at _max_pad_pct (10 by default).
  //
  // Where the cap bites: the pad is at most 511, so any K >= 10*512 = 5120
  // passes whatever its remainder (511/5120 = 9.98%). Below that it depends
  // entirely on K % 512 -- K=2816 pads 256 (9.1%, taken), K=1600 pads 448
  // (28%, refused). So the rule reads as "shallow contractions must be
  // nearly aligned already; deep ones need not care".
  bool accepts(int M, int N, int K) const
  {
    if (!_on || M < _min_m || K < 1024 || N < 16) { return false; }
    const int KP = kpad_(K);
    if (KP == K) { return true; }
    if (!_fn_quant_pad.valid()) { return false; }
    return (KP - K) * 100 <= K * _max_pad_pct;
  }

  // Encode act-quant + weight-quant + the i8 GEMM (element type = the ctor's
  // `bf16`; x/w/y must all be that type):
  //   y[M,N] (elem offset ye) = x[M,K] (elem offset xe) @ w[N,K]^T (dense)
  // Returns false -- with nothing encoded -- when the shape does not
  // qualify or a scratch allocation fails (caller keeps its dense path).
  bool gemm(metal_compute::ComputeEncoder& enc,
            const metal_compute::SharedBuffer& x, std::size_t xe,
            const metal_compute::SharedBuffer& w,
            const metal_compute::SharedBuffer& y, std::size_t ye,
            int M, int N, int K);

  // Drop the grow-only act/weight scratches (they re-grow on demand at the
  // next gemm()). Call between generations on a memory-bounded box so the
  // idle scratch doesn't crowd out a large downstream allocation.
  void release_scratch();

 private:
  metal_compute::MetalCompute* _mc = nullptr;
  bool _on = false;
  int _min_m = 1024;
  int _max_pad_pct = 10;   // VPIPE_I8_MAX_PAD_PCT
  metal_compute::ComputeLibrary _lib_q, _lib_g;
  metal_compute::ComputeFunction _fn_quant, _fn_quant_pad, _fn_gemm;

  // K rounded up to a whole number of 512-groups. The int8 GEMM contracts
  // in 512-wide chunks, so a K that is not a multiple of 512 has no chunk
  // to sit in; the quantizer zero-fills up to here instead, which is exact
  // (zeros add nothing to the dot product and cannot move an absmax scale)
  // and costs (kpad-K)/K extra int8 MACs -- 4.8% at H3's hidden 5376.
  static int kpad_(int K) { return ((K + 511) / 512) * 512; }
  // Grow-only scratches: xq[M,K] i8 + as[M,G] scales (activations),
  // wq[N,K] i8 + ws[N,G] scales (per-call weight re-quant). Scales are the
  // element type (2 bytes either way), so the byte sizes are format-agnostic.
  metal_compute::SharedBuffer _xq, _as, _wq, _ws;
};

}  // namespace genai
}  // namespace vpipe

#endif

#ifndef VPIPE_GENERATIVE_MODELS_SHARED_I8_GEMM_H
#define VPIPE_GENERATIVE_MODELS_SHARED_I8_GEMM_H

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cstddef>

namespace vpipe {
namespace genai {

// Dynamic-int8 accelerated GEMM ("accelerated mode" for large matmuls --
// DiT blocks, LLM prefill): the f16 activation is quantized ON THE FLY to
// i8 with per-(row, 512-group) scales (quant_f16_i8_row_g512), the f16
// weight (dense, or a dequant scratch) is quantized per-(out-channel,
// 512-group) into a reusable i8 scratch, and the product runs on the
// matrix units' int8 pipe -- ~2x the f16 matmul2d rate at qualifying
// shapes -- accumulating each 512-deep group in f32 with its own scales
// (gemm_i8i8_sc_f16_n64_g512) and storing f16.
//
// LOSSY: int8 quantization, rel-L2 ~1e-2 per GEMM. Strictly OPT-IN and
// never part of a token-exact path. The per-call weight re-quant costs
// one extra pass over the weight (~1-2% of a qualifying GEMM), so the
// mode has NO persistent memory cost and composes with every checkpoint
// format that already produces an f16 weight for the matmul.
//
// The stage/model switch arrives via `want`; env VPIPE_I8_GEMM=0|1
// overrides either way (A/B), VPIPE_I8_GEMM_MIN_M tunes the M gate.
class I8GemmContext {
 public:
  I8GemmContext(metal_compute::MetalCompute* mc, bool want);

  bool enabled() const { return _on; }

  // Shape gate: the win regime is big-M compute-bound GEMMs (measured
  // crossover ~1k rows on M5); K must split into whole 512-groups.
  bool accepts(int M, int N, int K) const
  {
    return _on && M >= _min_m && (K % 512) == 0 && K >= 1024 && N >= 16;
  }

  // Encode act-quant + weight-quant + the i8 GEMM:
  //   y[M,N] (f16, elem offset ye) = x[M,K] (f16, elem offset xe)
  //                                  @ w[N,K]^T (f16 dense)
  // Returns false -- with nothing encoded -- when the shape does not
  // qualify or a scratch allocation fails (caller keeps its f16 path).
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
  metal_compute::ComputeLibrary _lib_q, _lib_g;
  metal_compute::ComputeFunction _fn_quant, _fn_gemm;
  // Grow-only scratches: xq[M,K] i8 + as[M,G] f16 (activations),
  // wq[N,K] i8 + ws[N,G] f16 (per-call weight re-quant).
  metal_compute::SharedBuffer _xq, _as, _wq, _ws;
};

}  // namespace genai
}  // namespace vpipe

#endif

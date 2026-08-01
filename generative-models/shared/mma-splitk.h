#ifndef VPIPE_GENERATIVE_MODELS_SHARED_MMA_SPLITK_H
#define VPIPE_GENERATIVE_MODELS_SHARED_MMA_SPLITK_H

// Split-K deep-reduction dense matmul2d for the LM prefill GEMMs (M5+).
//
// A single-op matmul2d reduction underutilizes the matrix units once K gets
// very deep: only N/BN threadgroups exist, each walking one long serial
// contraction, so the pipeline stalls with nothing else in flight. At the two
// large dense OptiQ checkpoints' down_proj depths -- K = 17408 (Qwen3.6-27B,
// ffn) and 21504 (gemma-4-31B) -- that costs more than half the rate: the
// single-op kernel manages 4.7-5.4 TFLOP/s where the SAME weights contracted
// in the gate|up direction reach 10-11. A transpose control at identical
// FLOPs and identical weight bytes runs 10.9-11.6, which is what establishes
// this as contraction depth rather than the narrow N
// (optiq_blocks.mlp_tile_probe).
//
// Splitting K into S chunks gives each chunk its own threadgroup plane
// (grid.z), multiplying the threadgroup count by S and shortening each
// reduction back into the fast regime. A fold then sums the planes.
//
// NUMERICS -- why this uses the f32-plane kernels and not the f16-plane ones
// the diffusion models use. Folding f16 planes costs one rounding per plane
// against the single-op path's one in total; measured at these shapes that
// perturbs 37.5% of outputs by up to ~5 ulp. That is fine for a rel-L2-
// verified DiT and is exactly the kind of drift that flips a near-tie argmax
// in a decoder held to greedy token-exact. With f32 planes + an f32 fold the
// output rounds to f16 exactly once, and the only remaining difference from
// the single-op path is where the f32 contraction was split -- a
// reassociation. That leaves 0.5-0.6% of outputs differing by a single f16
// ulp, smaller than the steel-vs-matmul2d difference this codebase already
// accepts between prefill paths.
//
// The one thing NOT verified here: neither checkpoint this fires on fits in
// 16 GB, so the end-to-end greedy token-exact run against the single-op path
// is owed on the 64 GB box. VPIPE_LM_NO_SPLITK=1 disables the path for that
// A/B and as the kill switch. Nothing else is affected -- the KCs below
// divide only these two depths, so every other model's numerics are
// untouched by construction.

#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cstdlib>
#include <string>

namespace vpipe::genai {

// Loaded kernels + the partial-plane scratch. One per model instance.
struct MmaSplitK {
  // 4-way splits of the two shipped depths. 4-way beats 2-way at both, on
  // both shapes and at every M measured -- see the dense_gemm_mma.metal note;
  // the 2-way twins exist for the re-probe and are not loaded here.
  static constexpr int kSplitS = 4;
  static constexpr int kDepthA = 17408;   // Qwen3.6-27B ffn
  static constexpr int kDepthB = 21504;   // gemma-4-31B ffn

  // VPIPE_LM_SPLITK_TEST=1 additionally routes ffn = 9216 (Qwen3.5-4B) here.
  // That depth does not want splitting; it is the only one available on a
  // 16 GB box, and it exists so the wiring, the fold, and the resulting token
  // stream can be verified against a real decoder at all.
  static constexpr int kDepthTest = 9216;

  metal_compute::ComputeFunction fn_a, fn_b, fn_t, fn_fold;
  metal_compute::SharedBuffer planes;
  bool enabled = false;
  bool test_depth = false;

  void load(metal_compute::MetalCompute* mc,
            metal_compute::ComputeLibrary& lib_dense,
            metal_compute::ComputeLibrary& lib_elt)
  {
    if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
    fn_a = lib_dense.function(
        "dense_gemm_mma_splitk32_n128x256_k4352_f16");
    fn_b = lib_dense.function(
        "dense_gemm_mma_splitk32_n128x256_k5376_f16");
    fn_fold = lib_elt.function("splitk_fold_f32_f16");
    enabled = fn_fold.valid() && (fn_a.valid() || fn_b.valid());
    if (const char* e = std::getenv("VPIPE_LM_NO_SPLITK")) {
      if (std::atoi(e) != 0) { enabled = false; }
    }
    if (const char* e = std::getenv("VPIPE_LM_SPLITK_TEST")) {
      test_depth = std::atoi(e) != 0;
      if (test_depth) {
        fn_t = lib_dense.function(
            "dense_gemm_mma_splitk32_n128x256_k2304_f16");
      }
    }
  }

  // Max partial-plane scratch, MB, via VPIPE_LM_SPLITK_MAX_MB.
  static std::size_t budget_bytes()
  {
    static const std::size_t b = []() {
      const char* e = std::getenv("VPIPE_LM_SPLITK_MAX_MB");
      const long v = (e != nullptr) ? std::atol(e) : 0;
      return (std::size_t)(v > 0 ? v : 512) * 1024u * 1024u;
    }();
    return b;
  }

  const metal_compute::ComputeFunction* pick(int K) const
  {
    if (!enabled) { return nullptr; }
    if (K == kDepthA && fn_a.valid()) { return &fn_a; }
    if (K == kDepthB && fn_b.valid()) { return &fn_b; }
    if (test_depth && K == kDepthTest && fn_t.valid()) { return &fn_t; }
    return nullptr;
  }

  // Encode y[M,N] = x[M,K] @ wdeq[N,K]^T through the split path, or return
  // false to let the caller run its single-op dispatch. `M` must already have
  // passed the caller's matrix-core row threshold.
  bool encode(metal_compute::MetalCompute* mc,
              metal_compute::ComputeEncoder& enc,
              const metal_compute::SharedBuffer& x,
              const metal_compute::SharedBuffer& wdeq,
              const metal_compute::SharedBuffer& y,
              int K, int N, int M)
  {
    const metal_compute::ComputeFunction* fn = pick(K);
    if (fn == nullptr) { return false; }
    const std::size_t plane = (std::size_t)M * (std::size_t)N;
    const std::size_t need = plane * (std::size_t)kSplitS * 4;   // f32
    // The planes are S f32 copies of the output, so the scratch grows with M:
    // 352 MB at M=4096 on the 31B, 1.4 GB by M=16384. These are the two models
    // that already peak ~39% above the reference on footprint, so past a
    // budget the single-op GEMM -- slower but allocation-free -- is the better
    // trade. Default 512 MB covers M up to ~6000, and prefill above that pays
    // the un-split rate rather than a GB of scratch.
    if (need > budget_bytes()) { return false; }
    if (planes.empty() || planes.byte_size() < need) {
      planes = mc->make_shared_buffer(need);
      if (planes.empty()) { return false; }   // fall back, do not fault
    }
    enc.set_function(*fn);
    enc.set_buffer(0, x);
    enc.set_buffer(1, wdeq);
    enc.set_buffer(2, planes);
    enc.set_constant(3, K);
    enc.set_constant(4, N);
    enc.set_constant(5, M);
    enc.dispatch({(unsigned)(((N + 255) / 256) * 256),
                  (unsigned)((M + 127) / 128), (unsigned)kSplitS},
                 {256, 1, 1});
    // Fold: out = (elt)(sum_s planes[s]). Metal's hazard tracking orders this
    // after the split writes (both touch `planes`).
    enc.set_function(fn_fold);
    enc.set_buffer(0, planes);
    enc.set_buffer(1, y);
    enc.set_constant(2, (int)plane);
    enc.set_constant(3, kSplitS);
    enc.dispatch({(unsigned)plane, 1, 1}, {256, 1, 1});
    return true;
  }
};

}  // namespace vpipe::genai

#endif

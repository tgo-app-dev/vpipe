#ifndef VPIPE_GENERATIVE_MODELS_MAGI2_MAGI2_ACCEL_H
#define VPIPE_GENERATIVE_MODELS_MAGI2_MAGI2_ACCEL_H

// INTERNAL to generative-models/magi2/: what the preview DiT and the
// refiner share of their acceleration -- the int8 GEMM tiers over their
// block projections, the GEMM routes those fall back to, and the
// SageAttention switch. Not installed, not exported.
//
// THE INT8 TIERS are the image DiTs' (shared/i8-gemm.h), matrix cores
// only: -1 off, 0 accurate (the bf16 weight quantized per call, groups of
// 64, K >= 1024 and >= 1024 rows), 1 fast (held int8 at a coarse rotated
// group), 2 token (held int8 a weight row x a token, K rotated in blocks of
// 256), 3 token_plain (the same unrotated). What they take is every block
// projection a model runs through Gemms::proj -- q|k|v, the attention's
// output projection, the MLPs and the preview's split / merge / shared
// experts. What they leave in bf16: the per-head gate (24 or 32 outputs a
// row, too narrow to gain), the f32 embedders, routers and heads, and the
// preview's ROUTED experts, which run their own grouped kernel.
//
// SAGEATTENTION (shared/metal-sage-attention.h) runs inside the matrix-core
// flash kernel both models dispatch on an M5: the int8 QK, and with
// `sage_pv8` the P*V too -- over the preview's sinks and the refiner's
// block-sparse window alike.

#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/shared/sage-attention.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vpipe {
class FlexData;
namespace genai {

class GpuFlopsTally;
class I8GemmContext;
class WeightSet;

namespace magi2 {

// ---- the int8 tiers ------------------------------------------------------

// The tier the graph asked for (the accel bag's i8_gemm and i8_gemm_tier):
// -1 when i8_gemm is off.
int asked_i8_tier(const FlexData* accel);
// ...as the engine runs it: -1 without matrix cores; VPIPE_I8_GEMM,
// VPIPE_I8_TIER and VPIPE_I8_TOKEN_ROT override, as everywhere.
int effective_i8_tier(int asked, bool matrix_cores);
// "off", "accurate", "fast", "token", "token_plain".
const char* tier_name(int tier);
// The group an [N, K] projection is HELD at on `tier` (fast or token), as
// I8GemmContext::fast_group picks it; 0 where the tier holds nothing or its
// layout does not take K -- that projection stays bf16.
int held_group(int K, int tier);
// What such a projection holds: its codes and scales, or its bf16 weight.
std::uint64_t proj_bytes(int N, int K, int tier);

// ---- SageAttention -------------------------------------------------------

// Whether a model asked for `cfg` runs SageAttention on a box with or
// without matrix cores: the matrix-core attention is its only kernel
// (VPIPE_MAGI2_NO_NAX_ATTN=1 keeps the steel one, and Sage with it off),
// and VPIPE_NO_SAGE_ATTN turns it off everywhere.
bool sage_runs(const sage::Config& cfg, bool matrix_cores);
// Whether its P*V runs in int8 at sampling step `step` (-1: unknown, every
// step): sage_pv8 from sage_pv8_from_step on (-1 there: every step).
bool sage_pv8_at(const sage::Config& cfg, int step);

// ---- one block projection -------------------------------------------------

// One [N, K] projection as a model holds it: a bf16 weight (a window into
// the checkpoint's tensor, which stacks a per-modality layer's three), or
// -- on a tier that holds the layout and a K it takes -- its int8 codes
// [N, K] and scales at group `gc`.
struct Proj {
  metal_compute::SharedBuffer w;
  metal_compute::SharedBuffer codes, scales;
  int gc = 0;                     // 0: the bf16 weight
  int n = 0, k = 0;
  bool held() const noexcept { return gc != 0; }
  bool valid() const noexcept
  {
    return n > 0 && k > 0 &&
           (held() ? !codes.empty() && !scales.empty() : !w.empty());
  }
  std::uint64_t bytes() const noexcept
  {
    return held() ? codes.byte_size() + scales.byte_size() : w.byte_size();
  }
};

// The GEMM routes of one bf16 model.
class Gemms {
 public:
  Gemms();
  ~Gemms();
  Gemms(const Gemms&) = delete;
  Gemms& operator=(const Gemms&) = delete;

  // The dense kernels: matmul2d on matrix cores (VPIPE_MAGI2_NO_MMA2=1
  // keeps the steel tiles), the steel tiles otherwise. False with `err`
  // when a kernel this port cannot run without is missing.
  bool load(metal_compute::MetalCompute* mc, std::string* err);
  bool matrix_cores() const noexcept { return _mma; }

  // The model's tally for the achieved-FLOP/s meter (shared/dit-flops.h):
  // every dense() and proj() adds its 2MNK to it as it is encoded --
  // once, whichever route runs it and however many row bands. Null: none.
  void set_flops(GpuFlopsTally* t) noexcept { _flops = t; }

  // The int8 tier, settled BEFORE the first load_stacked: a tier that
  // holds the layout converts each projection as it loads. `tier` is
  // effective_i8_tier's answer; false (tier -1 here) where the context
  // declines -- no matrix cores, or its kernels missing.
  bool set_i8(int tier, bool autotune);
  int i8_tier() const noexcept { return _i8_tier; }

  // A [m * N, K] bf16 tensor -- a per-modality layer's three stacked, or
  // one -- as its m projections: windows into the cached tensor, or on a
  // holding tier the layout, converted once into a derived entry of the
  // set (the bf16 read uncached and dropped) under `key`. False with `err`.
  bool load_stacked(WeightSet& ws, const std::string& name, int m, int N,
                    int K, const std::string& key, std::vector<Proj>* out,
                    std::string* err);
  // A RANK'S SHARE of the same tensor (tensor parallel): of each slab, its
  // `rows` [first, end) of N in order -- an output cut -- and of every
  // row its columns [c0, c1) of K (c1 0: all) -- an input cut. Read
  // UNCACHED into buffers of the model's own (only the rows of a row cut
  // are read; a column cut reads the tensor and drops it), never into the
  // set's cache: the whole tensor is what a rank must not hold. On a
  // holding tier the share's own layout, converted from the cut. False
  // with `err`.
  bool load_cut(WeightSet& ws, const std::string& name, int m, int N, int K,
                const std::vector<std::pair<int, int>>& rows, int c0, int c1,
                std::vector<Proj>* out, std::string* err);

  // y[M, N] = x[M, K] W[N, K]^T, bf16, byte offsets -- matmul2d from 64
  // rows on matrix cores, else the steel tiles, in row bands that keep each
  // operand under 2^30 elements (matmul2d's 2 GB line; the steel kernel's
  // int indices).
  void dense(metal_compute::ComputeEncoder& enc,
             const metal_compute::SharedBuffer& x, std::size_t x_off,
             const metal_compute::SharedBuffer& w, std::size_t w_off,
             const metal_compute::SharedBuffer& y, std::size_t y_off, int M,
             int N, int K) const;
  // ...through the projection's route: a held layout's tier GEMM, the
  // accurate tier's per-call quantization where it takes the shape, else
  // dense -- in the same row bands.
  void proj(metal_compute::ComputeEncoder& enc,
            const metal_compute::SharedBuffer& x, std::size_t x_off,
            const Proj& p, const metal_compute::SharedBuffer& y,
            std::size_t y_off, int M) const;

  // At the top of a forward, with no encoder open: the accurate tier's
  // split-K tuning of the shapes it met (a no-op otherwise).
  void tune_pending();
  // ...and the int8 context's scratch at the forward's largest shapes (see
  // i8_scratch_bytes), made before anything is encoded: grown shape by
  // shape inside one command buffer, every smaller copy would stay alive
  // until it completes.
  struct Shape;
  bool reserve(int M, const std::vector<Shape>& projs);
  // The int8 context's grow-only scratch given back (between generations).
  void release_scratch();
  // GEMMs on the int8 route / dense, since load (engagement checks).
  long i8_gemms() const noexcept { return _n_i8; }
  long dense_gemms() const noexcept { return _n_dense; }

  // The rows a band of an [N, K] GEMM takes (see dense()).
  static int band_rows(int N, int K);
  // A projection's shape, and `rows` where it runs over fewer than M (an
  // MLP's chunk).
  struct Shape {
    int n = 0, k = 0;
    int rows = 0;               // 0: M
  };
  // What the int8 context holds at most in a forward whose projections
  // (`projs`, each [n, k]) run over up to M rows, at `tier`. A projection
  // runs the int8 route where the tier holds its layout or, held or not,
  // where the ACCURATE route takes its depth (k >= 1024). Held: the
  // activation's codes and scales at the widest k, over a band. Grown only
  // where some projection runs the accurate route: the widest such
  // weight's per-call codes, and -- where its tuner runs -- the tuner's
  // rows and the split-K planes (two f32 [rows, n] at the widest n, or
  // the context's whole 512 MB budget where one is 9216 deep or more).
  // Each buffer at the largest of the shapes, as reserve() makes it.
  static std::uint64_t i8_scratch_bytes(int tier, bool autotune, int M,
                                        const std::vector<Shape>& projs);

 private:
  // dense() without the count: proj()'s fallbacks, which proj() counted.
  void dense_(metal_compute::ComputeEncoder& enc,
              const metal_compute::SharedBuffer& x, std::size_t x_off,
              const metal_compute::SharedBuffer& w, std::size_t w_off,
              const metal_compute::SharedBuffer& y, std::size_t y_off, int M,
              int N, int K) const;

  metal_compute::MetalCompute* _mc = nullptr;
  GpuFlopsTally* _flops = nullptr;
  bool _mma = false;
  std::unique_ptr<I8GemmContext> _i8;
  int _i8_tier = -1;
  metal_compute::ComputeLibrary _lib_gemm, _lib_mma;
  metal_compute::ComputeFunction _fn_dgemm, _fn_dgemm_bn64, _fn_mma_n128,
      _fn_mma_deep, _fn_mma_tn2;
  // A held layout widened back to bf16, for a GEMM its tier declines
  // (none at this model's shapes: grown only if one does).
  mutable metal_compute::SharedBuffer _wdeq;
  mutable long _n_i8 = 0, _n_dense = 0;
};

}  // namespace magi2
}  // namespace genai
}  // namespace vpipe

#endif  // VPIPE_GENERATIVE_MODELS_MAGI2_MAGI2_ACCEL_H

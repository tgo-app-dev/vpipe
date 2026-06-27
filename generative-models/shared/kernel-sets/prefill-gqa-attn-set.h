#pragma once

// The PREFILL GQA attention kernel set (paged KV, head_dim 256). A family of
// interchangeable full-attention kernels that all speak one protocol, autotune
// WITHIN the set per machine, and expose a unified dispatch(). Same idea as
// DecodeGqaAttnSet, but keyed by the query count `n` (the prefill chunk size)
// instead of a decode position -- so the steel/flash/qtile crossover threshold
// is discovered per GPU rather than hard-coded (VPIPE_QWEN_STEEL_MIN).
//
// Members (capability-gated): steel (MLX register-softmax MMA flash, f16-only,
// from the attn_steel lib), mma (M5 matrix-core flash), flash (M4 simdgroup_-
// matrix key-split), qtile (scalar query-tiled), scalar (per-query). All consume
// the same buffer contract; only the grid differs. The model keeps the Q/K/V
// projection, rope, transpose, gate + o_proj around the set's dispatch(); the
// set just runs the SDPA (qt -> out).

#include "apple-silicon/metal-compute/compute-library.h"  // ComputeFunction
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/shared/kernel-autotune.h"

#include <string>

namespace vpipe::metal_compute {
class MetalCompute;
class ComputeEncoder;
class ComputeLibrary;
}  // namespace vpipe::metal_compute

namespace vpipe::genai {

class PrefillGqaAttnSet {
 public:
  struct Dims {
    int D = 256;
    int Hq = 16;
    int Hkv = 2;
  };

  // The paged-prefill protocol: `n` roped query rows qt[Hq,n,D] at global
  // q_offset attend the paged KV causally; result [Hq,n,D] -> out. Buffers are
  // the active dtype.
  struct Attn {
    const metal_compute::SharedBuffer* qt = nullptr;
    const metal_compute::SharedBuffer* kpool = nullptr;
    const metal_compute::SharedBuffer* vpool = nullptr;
    const metal_compute::SharedBuffer* out = nullptr;
    const metal_compute::SharedBuffer* page_table = nullptr;
    int n = 0;
    int q_offset = 0;
    int page_tokens = 0;
    int n_pages = 0;
    float scale = 0.0f;
  };

  // Load the member kernels. lib_sdpa supplies scalar/qtile/flash; lib_attn (may
  // be null / invalid for bf16) supplies the f16 steel kernel; lib_mma (may be
  // null / invalid off M5) supplies the matrix-core flash. use_mma enables the
  // mma member. Returns true if usable.
  bool load(metal_compute::ComputeLibrary& lib_sdpa,
            metal_compute::ComputeLibrary* lib_attn,
            metal_compute::ComputeLibrary* lib_mma, bool use_mma);

  // Autotune the kernel per n-regime for this GPU (the crossover thresholds fall
  // out of the per-regime winners). Appends timing to `rep`.
  void prepare(metal_compute::MetalCompute* mc, Dims dims, TuningReport& rep);

  bool ready() const { return _ready; }

  // THE UNIFIED ENTRANCE. Runs the winning member for a.n's regime.
  void dispatch(metal_compute::ComputeEncoder& enc, const Attn& a) const;

  // Resolved member name for `n` (for the model's load-time debug log).
  const char* kernel_name(int n) const;

 private:
  enum Member { kScalar = 0, kQtile = 1, kFlash = 2, kMma = 3, kSteel = 4 };
  metal_compute::ComputeFunction _fn[5];        // indexed by Member
  bool _have[5] = {false, false, false, false, false};
  Dims _dims;
  bool _ready = false;

  // Per n-regime winning member.
  static constexpr int kRegimes = 3;            // short / mid / long n
  int _member[kRegimes] = {kFlash, kSteel, kSteel};

  int regime_of(int n) const;
  void encode_member(metal_compute::ComputeEncoder& enc, const Attn& a,
                     int member) const;
};

}  // namespace vpipe::genai

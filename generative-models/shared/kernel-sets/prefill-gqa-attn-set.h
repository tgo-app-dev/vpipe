#pragma once

// The PREFILL GQA attention kernel set (paged KV, head_dim 256 or 128). A
// family of interchangeable full-attention kernels that all speak one protocol, autotune
// WITHIN the set per machine, and expose a unified dispatch(). Same idea as
// DecodeGqaAttnSet, but keyed by the query count `n` (the prefill chunk size)
// instead of a decode position -- so the steel/flash/qtile crossover threshold
// is discovered per GPU rather than hard-coded (VPIPE_QWEN_STEEL_MIN).
//
// Members (capability-gated): steel (MLX register-softmax MMA flash, f16-only,
// from the attn_steel lib), mma (M5 matrix-core flash), flash (M4 simdgroup_-
// matrix key-split), qtile (scalar query-tiled), scalar (per-query), and the two
// M5 NAX attentions from attn_steel_nax -- nax (the plain register-softmax
// kernel at bd 256) and nax-split (MLX's head-dim split variant). The paged
// members consume the pool directly; the NAX members read strided per-head
// planes, so the set first GATHERS the pool into contiguous [Hkv, kL, D] K/V it
// owns (one dispatch, kv_gather_paged) and then runs the kernel over those.
// MEASURED on the M5 Pro at Qwen3.5 shapes (16q/4kv, bd 256, f16, one page):
// nax-split 1.5x the best paged member at 1k and 3.6-3.8x at 4k-16k. At head
// dim 128 (Ministral-3, Qwen3-ASR) the members are the bd128 entry points;
// qtile and nax-split have none and are not offered. The model
// keeps the Q/K/V projection, rope, transpose, gate + o_proj around the set's
// dispatch(); the set just runs the SDPA (qt -> out).

#include "apple-silicon/metal-compute/compute-library.h"  // ComputeFunction
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/shared/kernel-autotune.h"

#include <string>
#include <string_view>

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

  // Load the member kernels. lib_sdpa supplies scalar/qtile/flash and the pool
  // gather; lib_attn (may be null / invalid for bf16) supplies the f16 steel
  // kernel; lib_mma (may be null / invalid off M5) supplies the matrix-core
  // flash; lib_nax (may be null / invalid off M5) the NAX attentions, whose
  // bf16 twins are chosen by `bf16`. use_mma enables the M5 members. Returns
  // true if usable.
  // `head_dim` (256 or 128) picks the members' entry points; prepare()'s
  // Dims::D must match it.
  bool load(metal_compute::ComputeLibrary& lib_sdpa,
            metal_compute::ComputeLibrary* lib_attn,
            metal_compute::ComputeLibrary* lib_mma, bool use_mma,
            metal_compute::ComputeLibrary* lib_nax = nullptr,
            bool bf16 = false, int head_dim = 256);

  // Autotune the kernel per n-regime for this GPU (the crossover thresholds fall
  // out of the per-regime winners). Appends timing to `rep`.
  // VPIPE_QWEN_PREFILL_ATTN=<member> pins every regime to that member when it
  // is available (A/B).
  //
  // `measure` false (the accel key `autotune`, accel-keys.h): nothing is
  // timed or read from the session's answers, and every regime takes the
  // member believed fastest here -- the NAX attention where the GPU has it
  // (nax-split 1.5x the best paged member at 1k and 3.6-3.8x at 4k-16k on
  // the M5 Pro at bd 256, and the tuner's pick there down to 768 rows; nax
  // 3.8 ms against steel's 10.8 at 2048 rows, bd 128) -- else the defaults:
  // flash (qtile) short, steel (mma) past it. The same members every load,
  // so the same rounding.
  void prepare(metal_compute::MetalCompute* mc, Dims dims, TuningReport& rep,
               bool measure = true);

  bool ready() const { return _ready; }

  // THE UNIFIED ENTRANCE. Runs the winning member for a.n's regime.
  void dispatch(metal_compute::ComputeEncoder& enc, const Attn& a) const;

  // Run a NAMED member ("flash", "steel", "mma", "nax", "nax-split", ...),
  // for tests and diagnostics. False when that member is not available.
  bool dispatch_member(metal_compute::ComputeEncoder& enc, const Attn& a,
                       std::string_view member) const;

  // The shortest chunk the set serves: every one at head dim 256; at 128
  // the regimes from 1536 rows, the ones whose tuner had something to add
  // (on the M5 the NAX attention), so a short prompt keeps the kernels it
  // always had. prepare() tunes only the regimes served.
  int min_n() const;

  // Resolved member name for `n` (for the model's load-time debug log).
  const char* kernel_name(int n) const;
  // Whether a named member loaded here ("nax", "steel", ...).
  bool has(std::string_view member) const;

  // The steel member's shape (attn_steel_paged): the key block a page must
  // be a multiple of, and how many simdgroups split the head dim -- its
  // threadgroup is (32, 4, split). Head dim 256 splits with 32-key blocks;
  // 128 keeps 16 and no split, which measured faster there. The model's
  // own steel path reads the same two.
  static constexpr int steel_block(int head_dim) {
    return head_dim == 256 ? 32 : 16;
  }
  static constexpr int steel_split(int head_dim) {
    return head_dim == 256 ? 2 : 1;
  }

 private:
  enum Member {
    kScalar = 0, kQtile = 1, kFlash = 2, kMma = 3, kSteel = 4,
    kNax = 5, kNaxSplit = 6, kMembers = 7
  };
  metal_compute::ComputeFunction _fn[kMembers];  // indexed by Member
  bool _have[kMembers] = {false, false, false, false, false, false, false};
  Dims _dims;
  bool _ready = false;

  // The NAX members' strided K/V: the library, the pool gather, and the
  // pipelines per (split, align_Q, align_K) -- function constants, so built on
  // first use and kept.
  metal_compute::ComputeLibrary* _lib_nax = nullptr;
  bool _nax_bf16 = false;
  int _head_dim = 256;                            // load()'s, 256 or 128
  metal_compute::ComputeFunction _fn_gather;
  metal_compute::MetalCompute* _mc = nullptr;
  mutable metal_compute::ComputeFunction _nax_fn[2][2][2];
  mutable metal_compute::SharedBuffer _kc, _vc;   // [Hkv, kL, D], grown

  // Per n-regime winning member.
  static constexpr int kRegimes = 4;            // short / mid / long / very long
  int _member[kRegimes] = {kFlash, kSteel, kSteel, kSteel};

  int regime_of(int n) const;
  void encode_member(metal_compute::ComputeEncoder& enc, const Attn& a,
                     int member) const;
  void encode_nax(metal_compute::ComputeEncoder& enc, const Attn& a,
                  bool split) const;
  const metal_compute::ComputeFunction& nax_fn(bool split, bool align_q,
                                               bool align_k) const;
};

}  // namespace vpipe::genai

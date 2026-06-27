#pragma once

// A "kernel set": a family of INTERCHANGEABLE GPU kernels that all speak ONE
// protocol + dtype contract, autotune WITHIN the set per machine, and expose a
// single unified dispatch() that callers "tap into". Models hold the set and
// call dispatch() -- they don't know which member kernel runs or how it was
// chosen. New models reuse the set instead of re-implementing kernel selection.
//
// This is the DECODE GQA ATTENTION set (paged KV, head_dim 256, f16 OR bf16).
// Members today: vec2 (MLX sdpa_vector 2-pass + coalesced merge2) and kbl
// (key-across-lanes block flash + merge); both consume the same paged-context
// protocol below and the same f16/bf16 KV pool. autotune() picks the kernel +
// #position-splits per context-regime (see kernel-autotune.h); dispatch() runs
// the winner for the query's position. Companion sets (prefill attention, the
// q4-g64 / q8-g64 / q8-g32 qmm families) follow the same shape.

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

class DecodeGqaAttnSet {
 public:
  // Fixed shape of the attention this set serves.
  struct Dims {
    int D = 256;
    int Hq = 16;
    int Hkv = 2;
  };

  // The paged-decode protocol every member kernel obeys: ONE query row [Hq,D]
  // at global position `pos` attends the paged KV pool [n_pages, Hkv,
  // page_tokens, D]; the [Hq,D] result lands in `out`. Buffers are the active
  // dtype (f16 or bf16); offsets are in BYTES. page_table is {pid,nvalid,gstart}
  // triples. This is the ONLY thing a caller needs to know.
  struct Attn {
    const metal_compute::SharedBuffer* q = nullptr;
    std::size_t q_off = 0;
    const metal_compute::SharedBuffer* kpool = nullptr;
    const metal_compute::SharedBuffer* vpool = nullptr;
    const metal_compute::SharedBuffer* out = nullptr;
    std::size_t out_off = 0;
    const metal_compute::SharedBuffer* page_table = nullptr;
    std::size_t pgt_off = 0;
    int pos = 0;
    int page_tokens = 0;
    int n_pages = 0;
    float scale = 0.0f;
  };

  // Load the member kernels for the active dtype from the compute library.
  // Returns true if the set is usable (>=1 member + its merge available).
  bool load(metal_compute::ComputeLibrary& lib, bool use_bf16);

  // Allocate the set's own partial scratch (O/m/l) and autotune kernel+split per
  // regime for this GPU. Idempotent. Appends timing to `rep`.
  void prepare(metal_compute::MetalCompute* mc, Dims dims, TuningReport& rep);

  bool ready() const { return _ready; }

  // THE UNIFIED ENTRANCE. Encodes the winning member kernel (+ its tuned split)
  // for a.pos's regime into a.out. No-op if !ready().
  void dispatch(metal_compute::ComputeEncoder& enc, const Attn& a) const;

  // Resolved choice for `pos` (for the model's load-time debug log).
  const char* kernel_name(int pos) const;
  bool uses_vec2(int pos) const;
  int split_for(int pos) const;

 private:
  // ---- members + dtype-active handles ---------------------------------------
  metal_compute::ComputeFunction _vec2, _merge2, _kbl, _merge;
  bool _use_bf16 = false;
  Dims _dims;

  // ---- self-owned partial scratch (O[Hq,split,D], m/l[Hq,split]) -------------
  metal_compute::SharedBuffer _oacc, _m, _l;

  // ---- per-regime selection (autotuned) -------------------------------------
  static constexpr int kRegimes = 3;
  int _kernel[kRegimes] = {1, 1, 1};         // 1=vec2, 0=kbl
  int _split[kRegimes] = {128, 256, 256};
  bool _ready = false;

  int regime_of(int pos) const;
  void encode_member(metal_compute::ComputeEncoder& enc, const Attn& a,
                     bool vec2, int split) const;
};

}  // namespace vpipe::genai

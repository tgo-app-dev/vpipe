#ifndef VPIPE_GENERATIVE_MODELS_SHARED_STREAMED_REFILL_H
#define VPIPE_GENERATIVE_MODELS_SHARED_STREAMED_REFILL_H

// Refill ONE streamed tensor into a buffer the caller already owns.
//
// A block-streamed DiT re-reads its whole stack every forward. Doing that
// by allocating a block's worth of SharedBuffers per block and filling
// them out of the shard's mmap costs both the allocation and, far more,
// the demand faulting -- MEASURED on an M5 over a 206 MB block, arms
// interleaved and their order rotated: 0.86-1.48 GB/s that way against
// 6.7-6.9 GB/s for a pread into a buffer that already exists. The mapped
// path's rate also moves by 2x between rounds where pread's does not,
// which is what turns it into GPU occupancy that will not sit still.
//
// So a streaming model wants two reusable destinations and a way to put
// the next block's bytes into one of them. WeightSet::stream_into is that
// read; this is the part above it that every such model needs and would
// otherwise write again: which tensors a raw read can serve, and what to
// do about the ones it cannot.
//
// WHY THREE OUTCOMES AND NOT TWO. The first version of this lived inside
// MiniMax-H3 and answered a per-BLOCK question -- any tensor it could not
// place turned the whole mechanism off for the run. That is right for a
// checkpoint whose blocks are uniform and wrong for one that carries a
// handful of odd tensors among many: LTX-2.5's blocks hold 140 tensors of
// which 134 (99.9% of the bytes) are raw-copyable and 6 are f32
// modulation tables, so an all-or-nothing rule gives up 411 MB of fast
// reads to avoid 0.4 MB of awkward ones. `kUnservable` is therefore a
// statement about ONE tensor, and says "build this one the way you always
// did", not "give up".
//
// It is kept distinct from `kFailed` because the two mean opposite things
// to a caller. Unservable is expected and costs nothing but the old path;
// failed means the checkpoint and the destination disagree, or the read
// was short, and the buffer is now PARTLY WRITTEN -- so that one has to
// be rebuilt from scratch rather than topped up.
//
// THE DTYPE RULE, AND WHY THE CALLER HAS TO STATE IT. What a destination
// can accept depends on what the caller will read out of it, and the two
// answers differ:
//
//   kRaw   the buffer holds the checkpoint's OWN bytes. BF16 and the U32
//          words a quantized pack stores its codes in are placed; F16 is
//          NOT, because a caller expecting f16 must get f16.
//   kBf16  the buffer is read as bf16. The same raw dtypes are placed,
//          and F16 is additionally converted IN PLACE -- legitimate only
//          because f16 and bf16 are the same width, which is the whole
//          reason a checkpoint's f16 scales can land in a bf16
//          destination.
//
// There is no default, deliberately. The two differ only for F16, and a
// wrong guess there is silent: the buffer is the right size and full of
// plausible numbers that are off by an exponent bias. A caller that
// converts f16 elsewhere (LTX-2.5 binds its quantized scales through its
// own as-bf16 path) must ask for kRaw, or this would convert them a
// second time.
//
// f32 is unservable under both: the destination is half the source, so
// there is nowhere to put the bytes.

#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/weight-set.h"

#include <cstdint>
#include <cstring>
#include <string>

namespace vpipe::genai {

// What the caller will read out of the destination. See the dtype rule
// above -- this is not a hint, it decides whether F16 is converted.
enum class RefillDst {
  kRaw,
  kBf16,
};

enum class Refill {
  // The bytes are in `dst`, in the layout the forward reads.
  kFilled,
  // No raw read can place this tensor. Expected, and not an error: the
  // caller builds this ONE tensor however it did before.
  kUnservable,
  // It should have worked and did not. `dst` may be partly written, so
  // the caller must rebuild it by another route rather than continue.
  kFailed,
};

// Round-to-nearest-even, and deliberately the same expression the
// in-tree converters use -- a refill has to be bit-identical to the
// allocate-and-convert path it replaces, or the two cannot be compared.
inline std::uint16_t
bf16_from_f32(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// f16 -> bf16 where the bytes already lie.
//
// Through memcpy, not a _Float16* aliasing the same memory. Two pointers
// of different types over one buffer is exactly the pattern the optimiser
// is allowed to assume cannot happen, and it would be free to hoist the
// reads above the writes -- at -O2, on a loop this simple, that is a
// vectorisation away from reading values this pass has already
// overwritten.
inline void
bf16_from_f16_in_place(const metal_compute::SharedBuffer& buf)
{
  const std::size_t n = buf.byte_size() / 2;
  auto* p = static_cast<std::uint16_t*>(buf.contents());
  for (std::size_t i = 0; i < n; ++i) {
    _Float16 h;
    std::memcpy(&h, &p[i], sizeof(h));
    p[i] = bf16_from_f32((float)h);
  }
}

// Whether a refill would serve this tensor, WITHOUT allocating anything.
//
// A caller that owns its destinations already (a slot pair) never needs
// this -- it asks once and keeps the answer. A caller that still
// allocates per tensor does: asking first is the difference between
// allocating only what the fast path will use and allocating for every
// tensor and dropping the ones it cannot fill. On an LTX-2.5 block that
// is 62 buffers of 24.5 MB per block per step that never get read.
//
// The rule lives here rather than at the call site so the two cannot
// drift apart.
inline bool
refill_serves(WeightSet& ws, const std::string& name, RefillDst as)
{
  const auto* ti = ws.src().info(name);
  if (ti == nullptr) { return false; }
  return ti->dtype == "BF16" || ti->dtype == "U32" ||
         (ti->dtype == "F16" && as == RefillDst::kBf16);
}

// Put `name`'s bytes into `dst`, which the caller owns and which must
// already be the tensor's size.
//
// The size check is against the CHECKPOINT rather than an expectation:
// a destination sized for one block only fits another if the shapes
// genuinely agree, and where they do not this would otherwise write the
// wrong number of bytes and run.
inline Refill
refill_streamed_tensor(WeightSet& ws, const std::string& name,
                       const metal_compute::SharedBuffer& dst, RefillDst as)
{
  if (dst.empty()) { return Refill::kUnservable; }
  const auto* ti = ws.src().info(name);
  if (ti == nullptr) { return Refill::kFailed; }
  const bool f16 = ti->dtype == "F16" && as == RefillDst::kBf16;
  if (!refill_serves(ws, name, as)) { return Refill::kUnservable; }
  if (ti->nbytes != dst.byte_size()) { return Refill::kFailed; }
  if (!ws.stream_into(name, dst.contents(), dst.byte_size())) {
    return Refill::kFailed;
  }
  if (f16) { bf16_from_f16_in_place(dst); }
  return Refill::kFilled;
}

}  // namespace vpipe::genai

#endif

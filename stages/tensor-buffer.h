#ifndef VPIPE_STAGES_TENSOR_BUFFER_H
#define VPIPE_STAGES_TENSOR_BUFFER_H

// TensorBeat <-> DataBuffer, for stages that move tensors through a
// command channel (pipeline/stage-command.h).
//
// The two directions are deliberately asymmetric. A TensorBeat can be
// VIEWED as a DataBuffer in place -- same bytes, its strides turned into
// byte strides -- which is how a stage exposes a beat to a caller
// without a copy. The other way COPIES: a TensorBeat owns its bytes
// outright, so a caller's buffer becomes a beat by being gathered into
// one. A caller that wants no copy at all asks the stage for a buffer
// to fill instead (external-input's `lease`).

#include "apple-silicon/tensor-beat.h"
#include "vpipe/stage-command.h"

#include <memory>
#include <string>

namespace vpipe {

// The element type a TensorBeat dtype is, and back. The TensorBeat side
// has five types, so the reverse fails for the other eight.
ElementType element_type_of(TensorBeat::DType d) noexcept;
bool tensor_dtype_of(ElementType t, TensorBeat::DType* out) noexcept;

// The accepted-type list a BufferSpec names for "anything a TensorBeat
// can carry".
inline constexpr const char* kTensorElementTypes = "u8,i8,f16,bf16,f32";

// `tb` as a view named `name`, kept alive by `owner` (which must keep
// `tb` itself alive). Mutable exactly when `writable`.
DataBuffer tensor_view(const TensorBeat& tb, std::string name,
                       std::shared_ptr<void> owner, bool writable);

// A contiguous TensorBeat holding a copy of `b`'s elements, gathered
// through any strides. False + *err when `b`'s type has no TensorBeat
// dtype or its layout does not fit its size.
bool tensor_from_buffer(const DataBuffer& b, TensorBeat* out,
                        std::string* err);

}

#endif

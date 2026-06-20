// vpipe_tensor_view.metal -- MSL-side wire format for the
// BufferView metadata that ComputeEncoder::set_buffer_view packs
// into a constant buffer. The C++ side (compute-encoder.cc) defines
// a struct with the same layout and writes it via setBytes; kernels
// pull it back via a `constant vpipe_tensor_view&` argument.
//
// Strides and offset are in ELEMENTS (matching vpipe TensorBeat
// conventions), not bytes. Maximum rank is fixed at 8.

#pragma once

#include <metal_stdlib>

struct vpipe_tensor_view {
    uint rank;
    uint dtype;      // matches vpipe::metal_compute::DType ordinal
    int  shape[8];
    int  strides[8]; // elements
    int  offset;     // elements from buffer base
};

// Logical -> element-offset index for ranks 1..4. Higher ranks can
// add more args following the same pattern; kernels that only use
// rank-1 access just call vpipe_tv_index(v, i).
inline int vpipe_tv_index(constant vpipe_tensor_view& v,
                          int i0,
                          int i1 = 0,
                          int i2 = 0,
                          int i3 = 0)
{
    int idx = v.offset;
    if (v.rank > 0) { idx += i0 * v.strides[0]; }
    if (v.rank > 1) { idx += i1 * v.strides[1]; }
    if (v.rank > 2) { idx += i2 * v.strides[2]; }
    if (v.rank > 3) { idx += i3 * v.strides[3]; }
    return idx;
}

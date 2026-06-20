// saxpy.metal -- y = a*x + y, 1-D strided. First real kernel using
// the vpipe_tensor_view helper. Doubles as the step-6 end-to-end
// smoke for ComputeEncoder::set_buffer_view + the embedded-
// metallib pipeline.
//
// Argument layout (MSL buffer slots):
//   0: x   (device const float*)
//   1: xv  (constant vpipe_tensor_view&)   -- view metadata for x
//   2: y   (device float*)
//   3: yv  (constant vpipe_tensor_view&)   -- view metadata for y
//   4: params (constant saxpy_params&)     -- scalar a

#include <metal_stdlib>
#include "vpipe_tensor_view.metal"

using namespace metal;

struct saxpy_params {
    float a;
};

kernel void saxpy(
    device const float*         x      [[buffer(0)]],
    constant vpipe_tensor_view& xv     [[buffer(1)]],
    device float*               y      [[buffer(2)]],
    constant vpipe_tensor_view& yv     [[buffer(3)]],
    constant saxpy_params&      params [[buffer(4)]],
    uint                        tid    [[thread_position_in_grid]])
{
    // Single-rank views; dispatch grid bounds the work but we
    // still guard against tail threads on partial threadgroups.
    if (xv.rank < 1 || yv.rank < 1) { return; }
    int i = int(tid);
    if (i >= xv.shape[0]) { return; }

    int xi = vpipe_tv_index(xv, i);
    int yi = vpipe_tv_index(yv, i);
    y[yi] = params.a * x[xi] + y[yi];
}

// fence_demo.metal -- two kernels used by the fence test:
//   fill_with_tid: out[tid] = tid
//   copy_uint:    dst[tid] = src[tid]
// Together they exercise an encoder pair where fill must retire
// before copy reads its output.

#include <metal_stdlib>

using namespace metal;

kernel void fill_with_tid(
    device uint*   out [[buffer(0)]],
    constant uint& n   [[buffer(1)]],
    uint           tid [[thread_position_in_grid]])
{
    if (tid >= n) { return; }
    out[tid] = tid;
}

kernel void copy_uint(
    device const uint* src [[buffer(0)]],
    device uint*       dst [[buffer(1)]],
    constant uint&     n   [[buffer(2)]],
    uint               tid [[thread_position_in_grid]])
{
    if (tid >= n) { return; }
    dst[tid] = src[tid];
}

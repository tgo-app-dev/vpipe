// concurrent_demo.metal -- two kernels for the concurrent-encoder
// test:
//   write_constant: out[tid] = value
//   read_then_write: dst[tid] = src[tid] + value
// Two `write_constant` dispatches to DIFFERENT buffers are
// concurrent-safe; a `read_then_write` chained behind a
// `write_constant` to the SAME buffer is NOT (the read must see
// the prior write) and requires an explicit memory_barrier.

#include <metal_stdlib>

using namespace metal;

kernel void write_constant(
    device uint*   out   [[buffer(0)]],
    constant uint& n     [[buffer(1)]],
    constant uint& value [[buffer(2)]],
    uint           tid   [[thread_position_in_grid]])
{
    if (tid >= n) { return; }
    out[tid] = value;
}

kernel void read_then_write(
    device const uint* src   [[buffer(0)]],
    device uint*       dst   [[buffer(1)]],
    constant uint&     n     [[buffer(2)]],
    constant uint&     value [[buffer(3)]],
    uint               tid   [[thread_position_in_grid]])
{
    if (tid >= n) { return; }
    dst[tid] = src[tid] + value;
}

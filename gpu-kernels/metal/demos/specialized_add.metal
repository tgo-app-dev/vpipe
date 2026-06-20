// specialized_add.metal -- y[tid] = kMultiplier * x[tid] + kBias
// where (kBias, kMultiplier) are pinned at JIT-compile time via
// MTL::FunctionConstantValues. Exercises the
// ComputeLibrary::function(name, FunctionConstants&) path.

#include <metal_stdlib>

using namespace metal;

constant int  kBias       [[function_constant(0)]];
constant uint kMultiplier [[function_constant(1)]];

kernel void specialized_add(
    device int*       y      [[buffer(0)]],
    device const int* x      [[buffer(1)]],
    constant uint&    n      [[buffer(2)]],
    uint              tid    [[thread_position_in_grid]])
{
    if (tid >= n) { return; }
    y[tid] = int(kMultiplier) * x[tid] + kBias;
}

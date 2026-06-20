// Trivial Metal compute kernel. Exists only to exercise the
// xcrun-metal/metallib + embedded-registry build path; the real
// kernels (saxpy and the strided helpers) land in later steps.

#include <metal_stdlib>
using namespace metal;

kernel void noop(uint tid [[thread_position_in_grid]])
{
  (void)tid;
}

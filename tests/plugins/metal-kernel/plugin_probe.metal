// A trivial Metal compute kernel used by the plugin-metal-library test.
// It is compiled OFFLINE to a standalone .metallib (not embedded into
// libvpipe) and registered at RUNTIME via
// MetalCompute::register_metal_library, mimicking how a plugin ships its
// own shaders. The kernel just writes each thread's id, enough to prove
// the library loads and its pipeline state builds.

#include <metal_stdlib>
using namespace metal;

kernel void plugin_probe(device uint* out [[buffer(0)]],
                         uint tid [[thread_position_in_grid]])
{
  out[tid] = tid;
}

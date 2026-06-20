// texture_read_pixel.metal -- single-thread kernel that reads
// pixel (px, py) from `tex` and writes it as float4 into out[0].
// Used by the metal_compute_texture suite to verify the texture
// binding path end-to-end.

#include <metal_stdlib>

using namespace metal;

struct read_params {
    uint x;
    uint y;
};

kernel void texture_read_pixel_rgba8(
    texture2d<float, access::read> tex    [[texture(0)]],
    device float4*                 out    [[buffer(0)]],
    constant read_params&          params [[buffer(1)]])
{
    out[0] = tex.read(uint2(params.x, params.y));
}

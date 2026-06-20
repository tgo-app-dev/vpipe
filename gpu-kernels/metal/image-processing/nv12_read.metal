// nv12_read.metal -- single-thread kernel that samples one pixel
// from a bi-planar NV12 source and writes the (y, cb, cr) triple
// as float3 into out[0]. Exercises the
// nv12_textures_from_cv_pixel_buffer bridge.
//
// Note: chroma is half-resolution; for pixel (x,y) the chroma
// sample comes from (x/2, y/2). Range conversion (video range
// 16..235 vs full range 0..255) and YCbCr->RGB are caller
// concerns; this kernel just reads the raw normalized values.

#include <metal_stdlib>

using namespace metal;

struct read_params {
    uint x;
    uint y;
};

kernel void nv12_read(
    texture2d<float, access::read>  luma   [[texture(0)]],
    texture2d<float, access::read>  chroma [[texture(1)]],
    device float4*                  out    [[buffer(0)]],
    constant read_params&           params [[buffer(1)]])
{
    const float y_val = luma.read(uint2(params.x, params.y)).r;
    const float2 cbcr = chroma.read(uint2(params.x / 2,
                                          params.y / 2)).rg;
    out[0] = float4(y_val, cbcr.x, cbcr.y, 0.0);
}

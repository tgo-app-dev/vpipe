// letterbox_planar_u8_to_rgb_f16.metal -- twin of
// letterbox_planar_u8_to_bgra_u8, but emits an f16 RGB tensor for direct
// binding as a CoreML MLMultiArray input (zero-copy: the GPU writes the
// model's input buffer; no CVPixelBuffer + lock + host memcpy).
//
// Aspect-preserving letterbox (bilinear resample + 114/255 grey pad), same
// geometry as the BGRA twin. Differences:
//   * Output is `half` (IEEE f16) regardless of the f16/bf16 build, so the
//     buffer matches MLMultiArrayDataTypeFloat16 exactly.
//   * RGB channel order is PRESERVED (no B/R swap): the model reads R,G,B
//     directly from the tensor.
//   * Layout: chw != 0 -> NCHW planar [3,out_h,out_w]; else NHWC interleaved
//     [out_h,out_w,3].
//   * Values are RAW pixels in [0,255], rounded as the u8 path does, stored
//     as half. The CoreML model bakes its own normalization (just as the
//     ImageType export's model baked pixel/255 then 2*(x-0.5)).
//
// 0:src(u8 planar RGB [3,in_h,in_w]) 1:dst(half) 2:params(in_w,in_h,out_w,
// out_h) 3:params2(new_w,new_h,pad_x,pad_y) 4:params3(inv,..) 5:chw(uint).

#include <metal_stdlib>
using namespace metal;

kernel void letterbox_planar_u8_to_rgb_f16(
    device const uchar* src     [[buffer(0)]],
    device half*        dst     [[buffer(1)]],
    constant uint4&     params  [[buffer(2)]],
    constant uint4&     params2 [[buffer(3)]],
    constant float4&    params3 [[buffer(4)]],
    constant uint&      chw     [[buffer(5)]],
    uint2               gid     [[thread_position_in_grid]])
{
  uint in_w  = params.x;
  uint in_h  = params.y;
  uint out_w = params.z;
  uint out_h = params.w;
  uint new_w = params2.x;
  uint new_h = params2.y;
  uint pad_x = params2.z;
  uint pad_y = params2.w;
  float inv  = params3.x;

  if (gid.x >= out_w || gid.y >= out_h) { return; }

  float px[3];
  bool outside =
      (gid.x < pad_x) || (gid.x >= pad_x + new_w) ||
      (gid.y < pad_y) || (gid.y >= pad_y + new_h);
  if (outside) {
    px[0] = 114.0; px[1] = 114.0; px[2] = 114.0;
  } else {
    float sxf = (float(gid.x - pad_x) + 0.5) * inv - 0.5;
    float syf = (float(gid.y - pad_y) + 0.5) * inv - 0.5;
    int ix0 = int(floor(sxf));
    int iy0 = int(floor(syf));
    int ix1 = ix0 + 1;
    int iy1 = iy0 + 1;
    float fx = sxf - float(ix0);
    float fy = syf - float(iy0);
    ix0 = clamp(ix0, 0, int(in_w) - 1);
    ix1 = clamp(ix1, 0, int(in_w) - 1);
    iy0 = clamp(iy0, 0, int(in_h) - 1);
    iy1 = clamp(iy1, 0, int(in_h) - 1);
    uint plane = in_w * in_h;
    for (uint c = 0; c < 3u; ++c) {
      uint base = c * plane;
      float v00 = float(src[base + uint(iy0) * in_w + uint(ix0)]);
      float v01 = float(src[base + uint(iy0) * in_w + uint(ix1)]);
      float v10 = float(src[base + uint(iy1) * in_w + uint(ix0)]);
      float v11 = float(src[base + uint(iy1) * in_w + uint(ix1)]);
      float v0  = mix(v00, v01, fx);
      float v1  = mix(v10, v11, fx);
      px[c]     = mix(v0, v1, fy);
    }
  }
  // Round to the same integer pixel values the u8/BGRA path yields, then
  // store as half (the model normalizes downstream).
  half r = half(clamp(floor(px[0] + 0.5), 0.0, 255.0));
  half g = half(clamp(floor(px[1] + 0.5), 0.0, 255.0));
  half b = half(clamp(floor(px[2] + 0.5), 0.0, 255.0));
  if (chw != 0u) {
    uint plane = out_w * out_h;
    uint off   = gid.y * out_w + gid.x;
    dst[0u * plane + off] = r;
    dst[1u * plane + off] = g;
    dst[2u * plane + off] = b;
  } else {
    uint off = (gid.y * out_w + gid.x) * 3u;
    dst[off + 0u] = r;
    dst[off + 1u] = g;
    dst[off + 2u] = b;
  }
}

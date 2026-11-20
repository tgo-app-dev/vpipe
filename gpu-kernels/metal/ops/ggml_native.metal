// ggml_native.metal -- ops/ggml_native.h's readers as kernels of their
// own: a row stack widened to f32 (the formats' test against ggml-blocks'
// decode on the CPU, and a reference for every kernel built on them).

#include "ops/ggml_native.h"

// rows [N, K] of type t at src -> out [N, K] f32. grid {K / 32, N}.
kernel void ggml_native_dequant(const device uchar* src [[buffer(0)]],
                                const device ulong* tab [[buffer(1)]],
                                device float* out [[buffer(2)]],
                                constant uint& t [[buffer(3)]],
                                constant int& K [[buffer(4)]],
                                constant int& N [[buffer(5)]],
                                uint2 gid [[thread_position_in_grid]])
{
  const int sub = (int)gid.x, r = (int)gid.y;
  if (sub >= K / 32 || r >= N) { return; }
  float v[32];
  ggml_native::w32(t, src + (long)r * ggml_native::row_bytes(t, K), sub,
                   tab, v);
  device float* o = out + (long)r * K + sub * 32;
  for (int j = 0; j < 32; ++j) { o[j] = v[j]; }
}

// The same through w8: grid {K / 8, N}.
kernel void ggml_native_dequant8(const device uchar* src [[buffer(0)]],
                                 const device ulong* tab [[buffer(1)]],
                                 device float* out [[buffer(2)]],
                                 constant uint& t [[buffer(3)]],
                                 constant int& K [[buffer(4)]],
                                 constant int& N [[buffer(5)]],
                                 uint2 gid [[thread_position_in_grid]])
{
  const int e = (int)gid.x, r = (int)gid.y;
  if (e >= K / 8 || r >= N) { return; }
  float v[8];
  ggml_native::w8(t, src + (long)r * ggml_native::row_bytes(t, K), e / 4,
                  e % 4, tab, v);
  device float* o = out + (long)r * K + e * 8;
  for (int j = 0; j < 8; ++j) { o[j] = v[j]; }
}

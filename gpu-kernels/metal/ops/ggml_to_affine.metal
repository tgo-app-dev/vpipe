// ggml_to_affine.metal -- a GGUF's routed-expert rows (IQ2_XXS, IQ2_XS,
// IQ1_M) into 8-bit affine codes at group 32 with bf16 scales and biases,
// ON THE GPU: what generative-models/shared/ggml-blocks.cc's row_affine32
// does on the CPU (at 8 bits, bf16), the same bits. A DeepSeek-V4.1 GGUF's
// expert miss is 8 MB read and ~38 MB written; the CPU's conversion was
// the miss's cost.
//
// One thread a 32-value group. IQ2_XXS is EXACT (to_affine32's form:
// code = +-grid + 128, the scale its block's db rounded to bf16, the bias
// -128 times that); IQ2_XS and IQ1_M are DECODED as ggml decodes them --
// each value one product chain in f32 -- and rounded to the nearest of
// 255 steps over the group's range (rtn_row_: the bias the minimum
// rounded, the scale the next bf16 at or above range / 255, each code the
// nearest against the two as stored, ties to even).
//
// Built without fast math (CMake: FLAGS), so every operation rounds as the
// CPU's does: no fused multiply-adds, IEEE division.

#include <metal_stdlib>
using namespace metal;

// The tables, as one buffer of ulongs: the IQ2_XXS grid (256), the IQ2_XS
// grid (512), the IQ1_S/M grid (2048), then ksigns_iq2xs's 128 bytes.
constant uint kGridXXS = 0;
constant uint kGridXS = 256;
constant uint kGrid1S = 768;
constant uint kSigns = 2816;   // in ulongs: its bytes from here

enum : uint { kIQ2_XXS = 16, kIQ2_XS = 17, kIQ1_M = 29 };

inline float
f16_at(device const uchar* p)
{
  return float(as_type<half>(ushort(p[0] | (ushort(p[1]) << 8))));
}

inline uint
u32_at(device const uchar* p)
{
  return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) |
         (uint(p[3]) << 24);
}

inline ushort
u16_at(device const uchar* p)
{
  return ushort(p[0] | (ushort(p[1]) << 8));
}

inline uint
grid_byte(device const ulong* t, uint at, uint idx, uint j)
{
  return uint((t[at + idx] >> (8 * j)) & 0xff);
}

inline uint
sign_byte(device const ulong* t, uint i)
{
  return uint((t[kSigns + i / 8] >> (8 * (i % 8))) & 0xff);
}

// Round-to-nearest-even bf16 bits, and back.
inline ushort
bf16_bits(float f)
{
  uint u = as_type<uint>(f);
  if ((u & 0x7f800000u) == 0x7f800000u) { return ushort(u >> 16); }
  u += 0x7fffu + ((u >> 16) & 1u);
  return ushort(u >> 16);
}

inline float
bf16_f32(ushort b)
{
  return as_type<float>(uint(b) << 16);
}

// rtn_row_ for one group of decoded values.
inline void
rtn_group(thread const float* v, device uchar* c, device ushort* s,
          device ushort* b)
{
  float mn = v[0], mx = v[0];
  for (int j = 1; j < 32; ++j) {
    mn = v[j] < mn ? v[j] : mn;
    mx = v[j] > mx ? v[j] : mx;
  }
  const ushort bb = bf16_bits(mn);
  const float bf = bf16_f32(bb);
  const float range = mx - bf > 0.0f ? mx - bf : 0.0f;
  ushort sb = 0;
  if (range > 0.0f) {
    const float r = range / 255.0f;
    sb = bf16_bits(r);
    if (bf16_f32(sb) < r) { ++sb; }
  }
  const float sf = bf16_f32(sb);
  for (int j = 0; j < 32; ++j) {
    float q = sf > 0.0f ? rint((v[j] - bf) / sf) : 0.0f;
    q = q < 0.0f ? 0.0f : (q > 255.0f ? 255.0f : q);
    c[j] = uchar(q);
  }
  *s = sb;
  *b = bb;
}

// `groups` groups: row r's group g at src + src_off + r * rowb (its block g / 8,
// sub-block g % 8); codes [rows, K] bytes, scales / biases [rows, K / 32].
kernel void
ggml_rows_to_affine8(device const uchar* src [[buffer(0)]],
                     device const ulong* tab [[buffer(1)]],
                     device uchar* codes [[buffer(2)]],
                     device ushort* scales [[buffer(3)]],
                     device ushort* biases [[buffer(4)]],
                     constant uint& type [[buffer(5)]],
                     constant uint& K [[buffer(6)]],
                     constant uint& rowb [[buffer(7)]],
                     constant uint& groups [[buffer(8)]],
                     constant uint& src_off [[buffer(9)]],
                     uint gid [[thread_position_in_grid]])
{
  if (gid >= groups) { return; }
  src += src_off;
  const uint gpr = K / 32;
  const uint r = gid / gpr, g = gid % gpr;
  const uint ib = g % 8;
  device uchar* c = codes + (ulong)gid * 32;
  device ushort* s = scales + gid;
  device ushort* b = biases + gid;
  (void)r;
  if (type == kIQ2_XXS) {
    device const uchar* blk = src + (ulong)r * rowb + (g / 8) * 66;
    const float d = f16_at(blk);
    device const uchar* q = blk + 2 + 8 * ib;
    const uint a1 = u32_at(q + 4);
    const float db = d * (0.5f + float(a1 >> 28)) * 0.25f;
    for (uint l = 0; l < 4; ++l) {
      const uint sg = sign_byte(tab, (a1 >> (7 * l)) & 127);
      for (uint j = 0; j < 8; ++j) {
        const int gv = int(grid_byte(tab, kGridXXS, q[l], j));
        const int a = (sg & (1u << j)) != 0 ? -gv : gv;
        c[8 * l + j] = uchar(a + 128);
      }
    }
    // fit_group_ with off 128: the scale rounded, the bias -128 times it
    const ushort sb = bf16_bits(db);
    *s = sb;
    *b = bf16_bits(-128.0f * bf16_f32(sb));
    return;
  }
  float v[32];
  if (type == kIQ2_XS) {
    device const uchar* blk = src + (ulong)r * rowb + (g / 8) * 74;
    const float d = f16_at(blk);
    const uint sc = blk[66 + ib];
    const float db0 = d * (0.5f + float(sc & 0xf)) * 0.25f;
    const float db1 = d * (0.5f + float(sc >> 4)) * 0.25f;
    for (uint l = 0; l < 4; ++l) {
      const uint q = u16_at(blk + 2 + 2 * (4 * ib + l));
      const uint sg = sign_byte(tab, q >> 9);
      const float db = l / 2 == 0 ? db0 : db1;
      for (uint j = 0; j < 8; ++j) {
        v[8 * l + j] = db * float(grid_byte(tab, kGridXS, q & 511, j)) *
                       ((sg & (1u << j)) != 0 ? -1.0f : 1.0f);
      }
    }
  } else if (type == kIQ1_M) {
    device const uchar* blk = src + (ulong)r * rowb + (g / 8) * 56;
    device const uchar* qs = blk + 4 * ib;
    device const uchar* qh = blk + 32 + 2 * ib;
    ushort sc[4];
    for (uint i = 0; i < 4; ++i) { sc[i] = u16_at(blk + 48 + 2 * i); }
    const ushort su = ushort((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                             ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
    const float d = float(as_type<half>(su));
    const float dl1 =
        d * float(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1);
    const float dl2 =
        d * float(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);
    const uint idx[4] = {qs[0] | ((uint(qh[0]) << 8) & 0x700),
                         qs[1] | ((uint(qh[0]) << 4) & 0x700),
                         qs[2] | ((uint(qh[1]) << 8) & 0x700),
                         qs[3] | ((uint(qh[1]) << 4) & 0x700)};
    const float delta[4] = {(qh[0] & 0x08) != 0 ? -0.125f : 0.125f,
                            (qh[0] & 0x80) != 0 ? -0.125f : 0.125f,
                            (qh[1] & 0x08) != 0 ? -0.125f : 0.125f,
                            (qh[1] & 0x80) != 0 ? -0.125f : 0.125f};
    for (uint l = 0; l < 4; ++l) {
      const float dl = l < 2 ? dl1 : dl2;
      for (uint j = 0; j < 8; ++j) {
        const int gv = int(char(grid_byte(tab, kGrid1S, idx[l], j)));
        v[8 * l + j] = dl * (float(gv) + delta[l]);
      }
    }
  } else {
    return;
  }
  rtn_group(v, c, s, b);
}

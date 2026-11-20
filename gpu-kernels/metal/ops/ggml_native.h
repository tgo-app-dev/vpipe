// ggml_native.h -- llama.cpp's block formats read on the GPU as they are
// stored: a row's 32-value SUB-BLOCK widened into registers, the values
// ggml-blocks.cc's decode_block_ makes (each one product chain in f32, in
// its order). What a GGUF's routed experts are run from where vpipe keeps
// them in their own blocks (deepseek_v41.metal, the Qwen MoE's), never
// converted to another form.
//
// Formats: Q8_0, IQ4_NL (32-value blocks); Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS,
// IQ3_XXS, IQ3_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ1_M (256-value blocks).
// `tab` is gpu_iq_tables_native() (ggml-blocks.h): the grids, the IQ1_M
// grid's nibbles and IQ4_NL's values at the offsets below, in u64s.

#ifndef VPIPE_GGML_NATIVE_H
#define VPIPE_GGML_NATIVE_H

#include <metal_stdlib>
using namespace metal;

namespace ggml_native {

enum : uint {
  kQ8_0 = 8, kQ3_K = 11, kQ4_K = 12, kQ5_K = 13, kQ6_K = 14,
  kIQ2_XXS = 16, kIQ2_XS = 17, kIQ3_XXS = 18, kIQ4_NL = 20, kIQ3_S = 21,
  kIQ2_S = 22, kIQ4_XS = 23, kIQ1_M = 29,
};

// gpu_iq_tables_native's offsets, in u64s
constant constexpr uint kGridXXS = 0, kGridXS = 256, kGrid1S = 768,
                        kSigns = 2816, kGrid1N = 2832, kGrid2S = 3856,
                        kGrid3XXS = 4880, kGrid3S = 5008, kValues4NL = 5264;

// A block's values and bytes.
inline int block_elems(const uint t)
{
  return t == kQ8_0 || t == kIQ4_NL ? 32 : 256;
}
inline int block_bytes(const uint t)
{
  switch (t) {
    case kQ8_0: return 34;
    case kIQ4_NL: return 18;
    case kQ3_K: return 110;
    case kQ4_K: return 144;
    case kQ5_K: return 176;
    case kQ6_K: return 210;
    case kIQ2_XXS: return 66;
    case kIQ2_XS: return 74;
    case kIQ2_S: return 82;
    case kIQ3_XXS: return 98;
    case kIQ3_S: return 110;
    case kIQ4_XS: return 136;
    case kIQ1_M: return 56;
    default: return 0;
  }
}
inline bool supported(const uint t) { return block_bytes(t) != 0; }
// A row of K values' bytes.
inline long row_bytes(const uint t, const int K)
{
  return (long)(K / block_elems(t)) * block_bytes(t);
}

inline float f16_at(const device uchar* p)
{
  return float(as_type<half>(ushort(p[0] | (ushort(p[1]) << 8))));
}
inline uint u32_at(const device uchar* p)
{
  return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) |
         (uint(p[3]) << 24);
}
inline uint ksign(const uint i) { return i | ((popcount(i) & 1u) << 7); }
inline uint grid_u32(const device ulong* tab, const uint at, const uint i)
{
  return ((const device uint*)(tab + at))[i];
}
inline uint grid_byte8(const device ulong* tab, const uint at, const uint i,
                       const int j)
{
  return uint((tab[at + i] >> (8 * j)) & 0xff);
}

// Q4_K / Q5_K's scale and min of sub-block j (scale_min_k4_).
inline void scale_min_k4(const int j, const device uchar* q, thread float& d,
                         thread float& m)
{
  if (j < 4) {
    d = float(q[j] & 63);
    m = float(q[j + 4] & 63);
  } else {
    d = float((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
    m = float((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
  }
}

// Q3_K's scale i (0..15), unbiased (q3k_scales_).
inline int q3k_scale(const device uchar* s12, const int i)
{
  const uint a0 = u32_at(s12), a1 = u32_at(s12 + 4), t = u32_at(s12 + 8);
  constexpr uint m1 = 0x03030303u, m2 = 0x0f0f0f0fu;
  uint w;
  switch (i / 4) {
    case 0: w = (a0 & m2) | (((t >> 0) & m1) << 4); break;
    case 1: w = (a1 & m2) | (((t >> 2) & m1) << 4); break;
    case 2: w = ((a0 >> 4) & m2) | (((t >> 4) & m1) << 4); break;
    default: w = ((a1 >> 4) & m2) | (((t >> 6) & m1) << 4); break;
  }
  return int(char((w >> (8 * (i % 4))) & 0xff)) - 32;
}

// Sub-block `sub` (values 32 * sub ..) of a row of type t starting at
// `row`, into v[32].
inline void w32(const uint t, const device uchar* row, const int sub,
                const device ulong* tab, thread float* v)
{
  const int elems = block_elems(t);
  const device uchar* b =
      row + (long)(sub * 32 / elems) * block_bytes(t);
  const int ib = (sub * 32 % elems) / 32;   // the sub-block in its block
  switch (t) {
    case kQ8_0: {
      const float d = f16_at(b);
      for (int j = 0; j < 32; ++j) { v[j] = float(char(b[2 + j])) * d; }
      return;
    }
    case kIQ4_NL: {
      const float d = f16_at(b);
      const device char* kv = (const device char*)(tab + kValues4NL);
      for (int j = 0; j < 32; ++j) {
        const uint n = j < 16 ? (b[2 + j] & 0xf) : (b[2 + j - 16] >> 4);
        v[j] = d * float(kv[n]);
      }
      return;
    }
    case kIQ4_XS: {
      const float d = f16_at(b);
      const uint sh = uint(b[2]) | (uint(b[3]) << 8);
      const int ls = int(((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) |
                         (((sh >> (2 * ib)) & 3) << 4));
      const float dl = d * float(ls - 32);
      const device uchar* qs = b + 8 + 16 * ib;
      const device char* kv = (const device char*)(tab + kValues4NL);
      for (int j = 0; j < 32; ++j) {
        const uint n = j < 16 ? (qs[j] & 0xf) : (qs[j - 16] >> 4);
        v[j] = dl * float(kv[n]);
      }
      return;
    }
    case kQ3_K: {
      const float d = f16_at(b + 108);
      const int n = ib / 4, sh = 2 * (ib % 4);
      const device uchar* q = b + 32 + 32 * n;
      const uint m = 1u << ib;
      for (int l = 0; l < 32; ++l) {
        const float dl = d * float(q3k_scale(b + 96, 2 * ib + l / 16));
        const int q4 = int((q[l] >> sh) & 3) - ((b[l] & m) != 0 ? 0 : 4);
        v[l] = dl * float(q4);
      }
      return;
    }
    case kQ4_K: {
      const float d = f16_at(b), mn = f16_at(b + 2);
      float s, m;
      scale_min_k4(ib, b + 4, s, m);
      const float d1 = d * s, m1 = mn * m;
      const device uchar* q = b + 16 + 32 * (ib / 2);
      for (int l = 0; l < 32; ++l) {
        const uint c = ib % 2 == 0 ? (q[l] & 0xF) : (q[l] >> 4);
        v[l] = d1 * float(c) - m1;
      }
      return;
    }
    case kQ5_K: {
      const float d = f16_at(b), mn = f16_at(b + 2);
      float s, m;
      scale_min_k4(ib, b + 4, s, m);
      const float d1 = d * s, m1 = mn * m;
      const device uchar* qh = b + 16;
      const device uchar* ql = b + 48 + 32 * (ib / 2);
      const uint u = 1u << ib;
      for (int l = 0; l < 32; ++l) {
        const uint c = (ib % 2 == 0 ? (ql[l] & 0xF) : (ql[l] >> 4)) +
                       ((qh[l] & u) != 0 ? 16u : 0u);
        v[l] = d1 * float(c) - m1;
      }
      return;
    }
    case kQ6_K: {
      const float d = f16_at(b + 208);
      const int n = ib / 4, q4 = ib % 4;
      const device uchar* ql = b + 64 * n + (q4 % 2 == 1 ? 32 : 0);
      const device uchar* qh = b + 128 + 32 * n;
      const device char* sc = (const device char*)(b + 192 + 8 * n);
      for (int l = 0; l < 32; ++l) {
        const uint lo = q4 < 2 ? (ql[l] & 0xF) : (ql[l] >> 4);
        const int q = int(lo | (((qh[l] >> (2 * q4)) & 3) << 4)) - 32;
        v[l] = d * float(sc[l / 16 + 2 * q4]) * float(q);
      }
      return;
    }
    case kIQ2_XXS: {
      const float d = f16_at(b);
      const device uchar* q = b + 2 + 8 * ib;
      const uint a1 = u32_at(q + 4);
      const float db = d * (0.5f + float(a1 >> 28)) * 0.25f;
      for (int l = 0; l < 4; ++l) {
        const uint sg = ksign((a1 >> (7 * l)) & 127);
        for (int j = 0; j < 8; ++j) {
          v[8 * l + j] = db * float(grid_byte8(tab, kGridXXS, q[l], j)) *
                         ((sg & (1u << j)) != 0 ? -1.f : 1.f);
        }
      }
      return;
    }
    case kIQ2_XS: {
      const float d = f16_at(b);
      const uint sc = b[66 + ib];
      for (int l = 0; l < 4; ++l) {
        const float db = d * (0.5f + float(l < 2 ? sc & 0xf : sc >> 4)) *
                         0.25f;
        const uint q = uint(b[2 + 2 * (4 * ib + l)]) |
                       (uint(b[3 + 2 * (4 * ib + l)]) << 8);
        const uint sg = ksign(q >> 9);
        for (int j = 0; j < 8; ++j) {
          v[8 * l + j] = db * float(grid_byte8(tab, kGridXS, q & 511, j)) *
                         ((sg & (1u << j)) != 0 ? -1.f : 1.f);
        }
      }
      return;
    }
    case kIQ2_S: {
      const float d = f16_at(b);
      const uint sc = b[74 + ib];
      for (int l = 0; l < 4; ++l) {
        const float db = d * (0.5f + float(l < 2 ? sc & 0xf : sc >> 4)) *
                         0.25f;
        const uint gi = uint(b[2 + 4 * ib + l]) |
                        ((uint(b[66 + ib]) << (8 - 2 * l)) & 0x300);
        const uint sg = b[2 + 32 + 4 * ib + l];
        for (int j = 0; j < 8; ++j) {
          v[8 * l + j] = db * float(grid_byte8(tab, kGrid2S, gi, j)) *
                         ((sg & (1u << j)) != 0 ? -1.f : 1.f);
        }
      }
      return;
    }
    case kIQ3_XXS: {
      const float d = f16_at(b);
      const device uchar* qs = b + 2 + 8 * ib;
      const uint a = u32_at(b + 66 + 4 * ib);
      const float db = d * (0.5f + float(a >> 28)) * 0.5f;
      for (int l = 0; l < 4; ++l) {
        const uint sg = ksign((a >> (7 * l)) & 127);
        const uint g1 = grid_u32(tab, kGrid3XXS, qs[2 * l]);
        const uint g2 = grid_u32(tab, kGrid3XXS, qs[2 * l + 1]);
        for (int j = 0; j < 8; ++j) {
          const float a8 = float(((j < 4 ? g1 : g2) >> (8 * (j % 4))) & 0xff);
          v[8 * l + j] = db * a8 * ((sg & (1u << j)) != 0 ? -1.f : 1.f);
        }
      }
      return;
    }
    case kIQ3_S: {
      const float d = f16_at(b);
      const device uchar* qs = b + 2 + 8 * ib;
      const uint qh = b[66 + ib];
      const device uchar* sg = b + 74 + 4 * ib;
      const uint nib = ib % 2 == 0 ? (b[106 + ib / 2] & 0xf)
                                   : (b[106 + ib / 2] >> 4);
      const float db = d * float(1 + 2 * nib);
      for (int l = 0; l < 4; ++l) {
        const uint i1 = qs[2 * l] | ((qh << (8 - 2 * l)) & 256);
        const uint i2 = qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256);
        const uint g1 = grid_u32(tab, kGrid3S, i1);
        const uint g2 = grid_u32(tab, kGrid3S, i2);
        for (int j = 0; j < 8; ++j) {
          const float a8 = float(((j < 4 ? g1 : g2) >> (8 * (j % 4))) & 0xff);
          v[8 * l + j] = db * a8 * ((sg[l] & (1u << j)) != 0 ? -1.f : 1.f);
        }
      }
      return;
    }
    case kIQ1_M: {
      const device uchar* qs = b + 4 * ib;
      const device uchar* qh = b + 32 + 2 * ib;
      uint sc[4];
      for (int i = 0; i < 4; ++i) {
        sc[i] = uint(b[48 + 2 * i]) | (uint(b[49 + 2 * i]) << 8);
      }
      const ushort su = ushort((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                               ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
      const float d = float(as_type<half>(su));
      for (int l = 0; l < 4; ++l) {
        const int sh = 6 * (ib % 2) + (l < 2 ? 0 : 3);
        const float dl = d * float(2 * ((sc[ib / 2] >> sh) & 0x7) + 1);
        const uint h = qh[l / 2];
        const uint idx =
            uint(qs[l]) | ((l % 2 == 0 ? (h << 8) : (h << 4)) & 0x700);
        const float delta =
            (h & (l % 2 == 0 ? 0x08u : 0x80u)) != 0 ? -0.125f : 0.125f;
        for (int j = 0; j < 8; ++j) {
          const float gv = float(char(grid_byte8(tab, kGrid1S, idx, j)));
          v[8 * l + j] = dl * (gv + delta);
        }
      }
      return;
    }
    default:
      for (int j = 0; j < 32; ++j) { v[j] = 0.0f; }
      return;
  }
}

// The 8 values 8 * l .. 8 * l + 7 of sub-block `sub` (l 0..3): w32's,
// value for value, only those computed -- a tile loader's share, four
// threads a sub-block.
inline void w8(const uint t, const device uchar* row, const int sub,
               const int l, const device ulong* tab, thread float* v)
{
  const int elems = block_elems(t);
  const device uchar* b =
      row + (long)(sub * 32 / elems) * block_bytes(t);
  const int ib = (sub * 32 % elems) / 32;
  const int j0 = 8 * l;
  switch (t) {
    case kQ8_0: {
      const float d = f16_at(b);
      for (int j = 0; j < 8; ++j) { v[j] = float(char(b[2 + j0 + j])) * d; }
      return;
    }
    case kIQ4_NL: {
      const float d = f16_at(b);
      const device char* kv = (const device char*)(tab + kValues4NL);
      for (int j = 0; j < 8; ++j) {
        const int k = j0 + j;
        const uint n = k < 16 ? (b[2 + k] & 0xf) : (b[2 + k - 16] >> 4);
        v[j] = d * float(kv[n]);
      }
      return;
    }
    case kIQ4_XS: {
      const float d = f16_at(b);
      const uint sh = uint(b[2]) | (uint(b[3]) << 8);
      const int ls = int(((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) |
                         (((sh >> (2 * ib)) & 3) << 4));
      const float dl = d * float(ls - 32);
      const device uchar* qs = b + 8 + 16 * ib;
      const device char* kv = (const device char*)(tab + kValues4NL);
      for (int j = 0; j < 8; ++j) {
        const int k = j0 + j;
        const uint n = k < 16 ? (qs[k] & 0xf) : (qs[k - 16] >> 4);
        v[j] = dl * float(kv[n]);
      }
      return;
    }
    case kQ3_K: {
      const float d = f16_at(b + 108);
      const int n = ib / 4, sh = 2 * (ib % 4);
      const device uchar* q = b + 32 + 32 * n;
      const uint m = 1u << ib;
      // a run of 8 lies inside one 16: one scale
      const float dl = d * float(q3k_scale(b + 96, 2 * ib + j0 / 16));
      for (int j = 0; j < 8; ++j) {
        const int k = j0 + j;
        const int q4 = int((q[k] >> sh) & 3) - ((b[k] & m) != 0 ? 0 : 4);
        v[j] = dl * float(q4);
      }
      return;
    }
    case kQ4_K: {
      const float d = f16_at(b), mn = f16_at(b + 2);
      float s, m;
      scale_min_k4(ib, b + 4, s, m);
      const float d1 = d * s, m1 = mn * m;
      const device uchar* q = b + 16 + 32 * (ib / 2);
      for (int j = 0; j < 8; ++j) {
        const int k = j0 + j;
        const uint c = ib % 2 == 0 ? (q[k] & 0xF) : (q[k] >> 4);
        v[j] = d1 * float(c) - m1;
      }
      return;
    }
    case kQ5_K: {
      const float d = f16_at(b), mn = f16_at(b + 2);
      float s, m;
      scale_min_k4(ib, b + 4, s, m);
      const float d1 = d * s, m1 = mn * m;
      const device uchar* qh = b + 16;
      const device uchar* ql = b + 48 + 32 * (ib / 2);
      const uint u = 1u << ib;
      for (int j = 0; j < 8; ++j) {
        const int k = j0 + j;
        const uint c = (ib % 2 == 0 ? (ql[k] & 0xF) : (ql[k] >> 4)) +
                       ((qh[k] & u) != 0 ? 16u : 0u);
        v[j] = d1 * float(c) - m1;
      }
      return;
    }
    case kQ6_K: {
      const float d = f16_at(b + 208);
      const int n = ib / 4, q4 = ib % 4;
      const device uchar* ql = b + 64 * n + (q4 % 2 == 1 ? 32 : 0);
      const device uchar* qh = b + 128 + 32 * n;
      const device char* sc = (const device char*)(b + 192 + 8 * n);
      const float ds = d * float(sc[j0 / 16 + 2 * q4]);
      for (int j = 0; j < 8; ++j) {
        const int k = j0 + j;
        const uint lo = q4 < 2 ? (ql[k] & 0xF) : (ql[k] >> 4);
        const int q = int(lo | (((qh[k] >> (2 * q4)) & 3) << 4)) - 32;
        v[j] = ds * float(q);
      }
      return;
    }
    case kIQ2_XXS: {
      const float d = f16_at(b);
      const device uchar* q = b + 2 + 8 * ib;
      const uint a1 = u32_at(q + 4);
      const float db = d * (0.5f + float(a1 >> 28)) * 0.25f;
      const uint sg = ksign((a1 >> (7 * l)) & 127);
      for (int j = 0; j < 8; ++j) {
        v[j] = db * float(grid_byte8(tab, kGridXXS, q[l], j)) *
               ((sg & (1u << j)) != 0 ? -1.f : 1.f);
      }
      return;
    }
    case kIQ2_XS: {
      const float d = f16_at(b);
      const uint sc = b[66 + ib];
      const float db =
          d * (0.5f + float(l < 2 ? sc & 0xf : sc >> 4)) * 0.25f;
      const uint q = uint(b[2 + 2 * (4 * ib + l)]) |
                     (uint(b[3 + 2 * (4 * ib + l)]) << 8);
      const uint sg = ksign(q >> 9);
      for (int j = 0; j < 8; ++j) {
        v[j] = db * float(grid_byte8(tab, kGridXS, q & 511, j)) *
               ((sg & (1u << j)) != 0 ? -1.f : 1.f);
      }
      return;
    }
    case kIQ2_S: {
      const float d = f16_at(b);
      const uint sc = b[74 + ib];
      const float db =
          d * (0.5f + float(l < 2 ? sc & 0xf : sc >> 4)) * 0.25f;
      const uint gi = uint(b[2 + 4 * ib + l]) |
                      ((uint(b[66 + ib]) << (8 - 2 * l)) & 0x300);
      const uint sg = b[2 + 32 + 4 * ib + l];
      for (int j = 0; j < 8; ++j) {
        v[j] = db * float(grid_byte8(tab, kGrid2S, gi, j)) *
               ((sg & (1u << j)) != 0 ? -1.f : 1.f);
      }
      return;
    }
    case kIQ3_XXS: {
      const float d = f16_at(b);
      const device uchar* qs = b + 2 + 8 * ib;
      const uint a = u32_at(b + 66 + 4 * ib);
      const float db = d * (0.5f + float(a >> 28)) * 0.5f;
      const uint sg = ksign((a >> (7 * l)) & 127);
      const uint g1 = grid_u32(tab, kGrid3XXS, qs[2 * l]);
      const uint g2 = grid_u32(tab, kGrid3XXS, qs[2 * l + 1]);
      for (int j = 0; j < 8; ++j) {
        const float a8 = float(((j < 4 ? g1 : g2) >> (8 * (j % 4))) & 0xff);
        v[j] = db * a8 * ((sg & (1u << j)) != 0 ? -1.f : 1.f);
      }
      return;
    }
    case kIQ3_S: {
      const float d = f16_at(b);
      const device uchar* qs = b + 2 + 8 * ib;
      const uint qh = b[66 + ib];
      const uint sg = b[74 + 4 * ib + l];
      const uint nib = ib % 2 == 0 ? (b[106 + ib / 2] & 0xf)
                                   : (b[106 + ib / 2] >> 4);
      const float db = d * float(1 + 2 * nib);
      const uint i1 = qs[2 * l] | ((qh << (8 - 2 * l)) & 256);
      const uint i2 = qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256);
      const uint g1 = grid_u32(tab, kGrid3S, i1);
      const uint g2 = grid_u32(tab, kGrid3S, i2);
      for (int j = 0; j < 8; ++j) {
        const float a8 = float(((j < 4 ? g1 : g2) >> (8 * (j % 4))) & 0xff);
        v[j] = db * a8 * ((sg & (1u << j)) != 0 ? -1.f : 1.f);
      }
      return;
    }
    case kIQ1_M: {
      const device uchar* qs = b + 4 * ib;
      const device uchar* qh = b + 32 + 2 * ib;
      uint sc[4];
      for (int i = 0; i < 4; ++i) {
        sc[i] = uint(b[48 + 2 * i]) | (uint(b[49 + 2 * i]) << 8);
      }
      const ushort su = ushort((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                               ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
      const float d = float(as_type<half>(su));
      const int sh = 6 * (ib % 2) + (l < 2 ? 0 : 3);
      const float dl = d * float(2 * ((sc[ib / 2] >> sh) & 0x7) + 1);
      const uint h = qh[l / 2];
      const uint idx =
          uint(qs[l]) | ((l % 2 == 0 ? (h << 8) : (h << 4)) & 0x700);
      const float delta =
          (h & (l % 2 == 0 ? 0x08u : 0x80u)) != 0 ? -0.125f : 0.125f;
      for (int j = 0; j < 8; ++j) {
        const float gv = float(char(grid_byte8(tab, kGrid1S, idx, j)));
        v[j] = dl * (gv + delta);
      }
      return;
    }
    default:
      for (int j = 0; j < 8; ++j) { v[j] = 0.0f; }
      return;
  }
}

}  // namespace ggml_native

#endif

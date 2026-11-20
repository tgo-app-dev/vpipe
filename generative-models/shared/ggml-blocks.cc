#include "ggml-blocks.h"

#include <cstring>
#include <iterator>
#include <vector>

// ggml's own decoders evaluate each value as one product chain in f32;
// keep clang from fusing them, so a value here is the value there.
#pragma clang fp contract(off)

namespace vpipe::genai::ggml {

namespace {

#include "ggml-iq-tables.inc"

inline float
f16_(const std::uint8_t* p)
{
  _Float16 h;
  std::memcpy(&h, p, 2);
  return (float)h;
}

inline float
bf16_(const std::uint8_t* p)
{
  std::uint32_t u = (std::uint32_t)p[0] << 16 | (std::uint32_t)p[1] << 24;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

inline std::uint32_t
u32_(const std::uint8_t* p)
{
  std::uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}

inline std::uint16_t
u16_(const std::uint8_t* p)
{
  std::uint16_t v;
  std::memcpy(&v, p, 2);
  return v;
}

inline std::uint8_t
grid_byte_(const std::uint32_t* g, int idx, int j)
{
  return (std::uint8_t)(g[idx] >> (8 * j));
}

inline std::uint8_t
grid_byte_(const std::uint64_t* g, int idx, int j)
{
  return (std::uint8_t)(g[idx] >> (8 * j));
}

// Q4_K / Q5_K: sub-block j's 6-bit scale and min.
inline void
scale_min_k4_(int j, const std::uint8_t* q, std::uint8_t* d,
              std::uint8_t* m)
{
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}

// Q3_K: the sixteen 6-bit scales, unbiased (s - 32).
inline void
q3k_scales_(const std::uint8_t* s12, int* out)
{
  std::uint32_t aux[4];
  std::memcpy(aux, s12, 12);
  constexpr std::uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
  const std::uint32_t tmp = aux[2];
  aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
  aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
  aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
  aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
  const auto* sc = reinterpret_cast<const std::int8_t*>(aux);
  for (int i = 0; i < 16; ++i) { out[i] = sc[i] - 32; }
}

// Q3_K: value k (0..255) of the block as (q - 4), with the scale index.
// Fills q4[256] (in -4..3) and returns nothing; the scale of value k is
// sc[k / 16].
inline void
q3k_values_(const std::uint8_t* blk, int* q4)
{
  const std::uint8_t* hm = blk;
  const std::uint8_t* q = blk + 32;
  int k = 0;
  std::uint8_t m = 1;
  for (int n = 0; n < 256; n += 128) {
    int shift = 0;
    for (int j = 0; j < 4; ++j) {
      for (int l = 0; l < 16; ++l) {
        q4[k++] = (int)((q[l + 0] >> shift) & 3) -
                  ((hm[l + 0] & m) ? 0 : 4);
      }
      for (int l = 0; l < 16; ++l) {
        q4[k++] = (int)((q[l + 16] >> shift) & 3) -
                  ((hm[l + 16] & m) ? 0 : 4);
      }
      shift += 2;
      m <<= 1;
    }
    q += 32;
  }
}

// Q5_0 / Q5_1: the 5-bit code j (0..31) of a block.
inline int
q5_code_(const std::uint8_t* qh4, const std::uint8_t* qs, int j)
{
  const std::uint32_t qh = u32_(qh4);
  if (j < 16) {
    return (qs[j] & 0x0F) | ((((qh >> (j + 0)) << 4)) & 0x10);
  }
  const int i = j - 16;
  return (qs[i] >> 4) | (((qh >> (i + 12))) & 0x10);
}

// IQ3_XXS: value k of the block as a signed grid value, and db's sub-
// block scale nibble.
inline void
iq3xxs_values_(const std::uint8_t* blk, int* v, int* s4)
{
  const std::uint8_t* qs = blk + 2;
  const std::uint8_t* ss = qs + 64;
  int k = 0;
  for (int ib = 0; ib < 8; ++ib) {
    const std::uint32_t aux = u32_(ss + 4 * ib);
    s4[ib] = (int)(aux >> 28);
    for (int l = 0; l < 4; ++l) {
      const std::uint8_t signs = k_ksigns_iq2xs[(aux >> 7 * l) & 127];
      const int g1 = qs[2 * l + 0], g2 = qs[2 * l + 1];
      for (int j = 0; j < 4; ++j) {
        const int a = grid_byte_(k_iq3xxs_grid, g1, j);
        v[k + j] = (signs & k_kmask_iq2xs[j]) ? -a : a;
      }
      for (int j = 0; j < 4; ++j) {
        const int a = grid_byte_(k_iq3xxs_grid, g2, j);
        v[k + 4 + j] = (signs & k_kmask_iq2xs[j + 4]) ? -a : a;
      }
      k += 8;
    }
    qs += 8;
  }
}

// IQ3_S: value k as a signed grid value, and each sub-block's nibble.
inline void
iq3s_values_(const std::uint8_t* blk, int* v, int* s4)
{
  const std::uint8_t* qs = blk + 2;
  const std::uint8_t* qh = blk + 66;
  const std::uint8_t* signs = blk + 74;
  const std::uint8_t* scales = blk + 106;
  int k = 0;
  for (int ib = 0; ib < 8; ib += 2) {
    s4[ib] = scales[ib / 2] & 0xf;
    s4[ib + 1] = scales[ib / 2] >> 4;
    for (int h = 0; h < 2; ++h) {
      for (int l = 0; l < 4; ++l) {
        const int i1 = qs[2 * l + 0] | ((qh[h] << (8 - 2 * l)) & 256);
        const int i2 = qs[2 * l + 1] | ((qh[h] << (7 - 2 * l)) & 256);
        for (int j = 0; j < 4; ++j) {
          const int a = grid_byte_(k_iq3s_grid, i1, j);
          v[k + j] = (signs[l] & k_kmask_iq2xs[j]) ? -a : a;
        }
        for (int j = 0; j < 4; ++j) {
          const int a = grid_byte_(k_iq3s_grid, i2, j);
          v[k + 4 + j] = (signs[l] & k_kmask_iq2xs[j + 4]) ? -a : a;
        }
        k += 8;
      }
      qs += 8;
      signs += 4;
    }
    qh += 2;
  }
}

// IQ4_XS: sub-block ib's 6-bit scale, unbiased.
inline int
iq4xs_scale_(const std::uint8_t* blk, int ib)
{
  const std::uint16_t sh = (std::uint16_t)(blk[2] | blk[3] << 8);
  const std::uint8_t* sl = blk + 4;
  const int ls = ((sl[ib / 2] >> 4 * (ib % 2)) & 0xf) |
                 (((sh >> 2 * ib) & 3) << 4);
  return ls - 32;
}

// A block's 32-value nibble run (IQ4_NL, Q4_0, and an IQ4_XS sub-block):
// low nibbles first, then high.
inline int
nib_(const std::uint8_t* qs, int j)
{
  return j < 16 ? (qs[j] & 0xf) : (qs[j - 16] >> 4);
}

bool
decode_block_(std::uint32_t t, const std::uint8_t* b, float* y)
{
  switch (t) {
    case kQ4_0: {
      const float d = f16_(b);
      for (int j = 0; j < 32; ++j) { y[j] = (nib_(b + 2, j) - 8) * d; }
      return true;
    }
    case kQ5_0: {
      const float d = f16_(b);
      for (int j = 0; j < 32; ++j) {
        y[j] = (q5_code_(b + 2, b + 6, j) - 16) * d;
      }
      return true;
    }
    case kQ5_1: {
      const float d = f16_(b), m = f16_(b + 2);
      for (int j = 0; j < 32; ++j) {
        y[j] = q5_code_(b + 4, b + 8, j) * d + m;
      }
      return true;
    }
    case kQ8_0: {
      const float d = f16_(b);
      const auto* q = reinterpret_cast<const std::int8_t*>(b + 2);
      for (int j = 0; j < 32; ++j) { y[j] = q[j] * d; }
      return true;
    }
    case kIQ4_NL: {
      const float d = f16_(b);
      for (int j = 0; j < 32; ++j) {
        y[j] = d * k_kvalues_iq4nl[nib_(b + 2, j)];
      }
      return true;
    }
    case kQ3_K: {
      const float d = f16_(b + 108);
      int sc[16], q4[256];
      q3k_scales_(b + 96, sc);
      q3k_values_(b, q4);
      for (int k = 0; k < 256; ++k) {
        const float dl = d * sc[k / 16];
        y[k] = dl * q4[k];
      }
      return true;
    }
    case kQ4_K: {
      const float d = f16_(b), mn = f16_(b + 2);
      const std::uint8_t* sc = b + 4;
      const std::uint8_t* q = b + 16;
      int is = 0;
      for (int j = 0; j < 256; j += 64) {
        std::uint8_t s, m;
        scale_min_k4_(is + 0, sc, &s, &m);
        const float d1 = d * s, m1 = mn * m;
        scale_min_k4_(is + 1, sc, &s, &m);
        const float d2 = d * s, m2 = mn * m;
        for (int l = 0; l < 32; ++l) { *y++ = d1 * (q[l] & 0xF) - m1; }
        for (int l = 0; l < 32; ++l) { *y++ = d2 * (q[l] >> 4) - m2; }
        q += 32;
        is += 2;
      }
      return true;
    }
    case kQ5_K: {
      const float d = f16_(b), mn = f16_(b + 2);
      const std::uint8_t* sc = b + 4;
      const std::uint8_t* qh = b + 16;
      const std::uint8_t* ql = b + 48;
      int is = 0;
      std::uint8_t u1 = 1, u2 = 2;
      for (int j = 0; j < 256; j += 64) {
        std::uint8_t s, m;
        scale_min_k4_(is + 0, sc, &s, &m);
        const float d1 = d * s, m1 = mn * m;
        scale_min_k4_(is + 1, sc, &s, &m);
        const float d2 = d * s, m2 = mn * m;
        for (int l = 0; l < 32; ++l) {
          *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
        }
        for (int l = 0; l < 32; ++l) {
          *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
        }
        ql += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
      }
      return true;
    }
    case kQ6_K: {
      const float d = f16_(b + 208);
      const std::uint8_t* ql = b;
      const std::uint8_t* qh = b + 128;
      const auto* sc = reinterpret_cast<const std::int8_t*>(b + 192);
      for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
          const int is = l / 16;
          const int q1 = (int)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) -
                         32;
          const int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) -
                         32;
          const int q3 = (int)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) -
                         32;
          const int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) -
                         32;
          y[l + 0] = d * sc[is + 0] * q1;
          y[l + 32] = d * sc[is + 2] * q2;
          y[l + 64] = d * sc[is + 4] * q3;
          y[l + 96] = d * sc[is + 6] * q4;
        }
        y += 128;
        ql += 64;
        qh += 32;
        sc += 8;
      }
      return true;
    }
    case kIQ2_S: {
      const float d = f16_(b);
      const std::uint8_t* qs = b + 2;
      const std::uint8_t* qh = b + 66;
      const std::uint8_t* scales = b + 74;
      const std::uint8_t* signs = qs + 32;
      for (int ib = 0; ib < 8; ++ib) {
        const float db[2] = {d * (0.5f + (scales[ib] & 0xf)) * 0.25f,
                             d * (0.5f + (scales[ib] >> 4)) * 0.25f};
        for (int l = 0; l < 4; ++l) {
          const float dl = db[l / 2];
          const int gi = qs[l] | ((qh[ib] << (8 - 2 * l)) & 0x300);
          for (int j = 0; j < 8; ++j) {
            y[j] = dl * grid_byte_(k_iq2s_grid, gi, j) *
                   ((signs[l] & k_kmask_iq2xs[j]) ? -1.f : 1.f);
          }
          y += 8;
        }
        qs += 4;
        signs += 4;
      }
      return true;
    }
    case kIQ2_XXS: {
      const float d = f16_(b);
      for (int ib = 0; ib < 8; ++ib) {
        const std::uint8_t* q = b + 2 + 8 * ib;
        const std::uint32_t a1 = u32_(q + 4);
        const float db = d * (0.5f + (a1 >> 28)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
          const std::uint8_t sg = k_ksigns_iq2xs[(a1 >> (7 * l)) & 127];
          for (int j = 0; j < 8; ++j) {
            y[j] = db * grid_byte_(k_iq2xxs_grid, q[l], j) *
                   ((sg & k_kmask_iq2xs[j]) ? -1.f : 1.f);
          }
          y += 8;
        }
      }
      return true;
    }
    case kIQ2_XS: {
      const float d = f16_(b);
      const std::uint8_t* scales = b + 66;
      for (int ib = 0; ib < 8; ++ib) {
        const float db[2] = {d * (0.5f + (scales[ib] & 0xf)) * 0.25f,
                             d * (0.5f + (scales[ib] >> 4)) * 0.25f};
        for (int l = 0; l < 4; ++l) {
          const std::uint16_t q = u16_(b + 2 + 2 * (4 * ib + l));
          const std::uint8_t sg = k_ksigns_iq2xs[q >> 9];
          for (int j = 0; j < 8; ++j) {
            y[j] = db[l / 2] * grid_byte_(k_iq2xs_grid, q & 511, j) *
                   ((sg & k_kmask_iq2xs[j]) ? -1.f : 1.f);
          }
          y += 8;
        }
      }
      return true;
    }
    case kIQ1_M: {
      const std::uint8_t* qs = b;
      const std::uint8_t* qh = b + 32;
      std::uint16_t sc[4];
      for (int i = 0; i < 4; ++i) { sc[i] = u16_(b + 48 + 2 * i); }
      const std::uint16_t su = (std::uint16_t)(
          (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) |
          (sc[3] & 0xf000));
      std::uint8_t sb[2] = {(std::uint8_t)su, (std::uint8_t)(su >> 8)};
      const float d = f16_(sb);
      for (int ib = 0; ib < 8; ++ib) {
        const float dl1 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7)
                               + 1);
        const float dl2 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7)
                               + 1);
        const int idx[4] = {qs[0] | ((qh[0] << 8) & 0x700),
                            qs[1] | ((qh[0] << 4) & 0x700),
                            qs[2] | ((qh[1] << 8) & 0x700),
                            qs[3] | ((qh[1] << 4) & 0x700)};
        const float delta[4] = {(qh[0] & 0x08) ? -0.125f : 0.125f,
                                (qh[0] & 0x80) ? -0.125f : 0.125f,
                                (qh[1] & 0x08) ? -0.125f : 0.125f,
                                (qh[1] & 0x80) ? -0.125f : 0.125f};
        for (int l = 0; l < 4; ++l) {
          const float dl = l < 2 ? dl1 : dl2;
          for (int j = 0; j < 8; ++j) {
            const auto gv = (std::int8_t)grid_byte_(k_iq1s_grid, idx[l], j);
            y[j] = dl * (gv + delta[l]);
          }
          y += 8;
        }
        qs += 4;
        qh += 2;
      }
      return true;
    }
    case kIQ3_XXS: {
      const float d = f16_(b);
      int v[256], s4[8];
      iq3xxs_values_(b, v, s4);
      for (int ib = 0; ib < 8; ++ib) {
        const float db = d * (0.5f + s4[ib]) * 0.5f;
        for (int j = 0; j < 32; ++j) {
          const int a = v[32 * ib + j];
          y[32 * ib + j] = db * (a < 0 ? -a : a) * (a < 0 ? -1.f : 1.f);
        }
      }
      return true;
    }
    case kIQ3_S: {
      const float d = f16_(b);
      int v[256], s4[8];
      iq3s_values_(b, v, s4);
      for (int ib = 0; ib < 8; ++ib) {
        const float db = d * (1 + 2 * s4[ib]);
        for (int j = 0; j < 32; ++j) {
          const int a = v[32 * ib + j];
          y[32 * ib + j] = db * (a < 0 ? -a : a) * (a < 0 ? -1.f : 1.f);
        }
      }
      return true;
    }
    case kIQ4_XS: {
      const float d = f16_(b);
      const std::uint8_t* qs = b + 8;
      for (int ib = 0; ib < 8; ++ib) {
        const float dl = d * iq4xs_scale_(b, ib);
        for (int j = 0; j < 32; ++j) {
          y[32 * ib + j] = dl * k_kvalues_iq4nl[nib_(qs + 16 * ib, j)];
        }
      }
      return true;
    }
    default:
      return false;
  }
}

// One block as codes of `bits` and its groups' forms (32 values a group).
bool
affine_block_(std::uint32_t t, const std::uint8_t* b, int bits,
              std::uint8_t* c, Affine32* g)
{
  const int top = (1 << bits) - 1;
  auto put = [&](int k, int code) {
    if (code < 0 || code > top) { return false; }
    c[k] = (std::uint8_t)code;
    return true;
  };
  switch (t) {
    case kQ4_0:
    case kQ5_0:
    case kQ8_0:
    case kIQ4_NL: {
      // Symmetric: value = d * (code - off).
      const float d = f16_(b);
      int off = 0;
      for (int j = 0; j < 32; ++j) {
        int code = 0;
        if (t == kQ4_0) {
          off = 8;
          code = nib_(b + 2, j);
        } else if (t == kQ5_0) {
          off = 16;
          code = q5_code_(b + 2, b + 6, j);
        } else if (t == kQ8_0) {
          off = 128;
          code = reinterpret_cast<const std::int8_t*>(b + 2)[j] + 128;
        } else {
          off = 128;
          code = k_kvalues_iq4nl[nib_(b + 2, j)] + 128;
        }
        // A narrower format widened: the same values about a larger
        // offset (8 -> 128 is the code + 120), still exact.
        if (bits == 8 && off != 128) { code += 128 - off; }
        if (!put(j, code)) { return false; }
      }
      const float o = bits == 8 ? 128.0f : (float)off;
      g[0] = {d, -o * d, o};
      return true;
    }
    case kQ5_1: {
      const float d = f16_(b), m = f16_(b + 2);
      for (int j = 0; j < 32; ++j) {
        if (!put(j, q5_code_(b + 4, b + 8, j))) { return false; }
      }
      g[0] = {d, m, 0.0f};
      return true;
    }
    case kQ4_K:
    case kQ5_K: {
      const float d = f16_(b), mn = f16_(b + 2);
      const std::uint8_t* sc = b + 4;
      const bool five = t == kQ5_K;
      const std::uint8_t* qh = b + 16;
      const std::uint8_t* q = b + (five ? 48 : 16);
      std::uint8_t u1 = 1, u2 = 2;
      for (int j = 0; j < 4; ++j) {
        for (int h = 0; h < 2; ++h) {
          std::uint8_t s, m;
          scale_min_k4_(2 * j + h, sc, &s, &m);
          const float dl = d * s, ml = mn * m;
          g[2 * j + h] = {dl, -ml, 0.0f};
          for (int l = 0; l < 32; ++l) {
            int code = h == 0 ? (q[l] & 0xF) : (q[l] >> 4);
            if (five && (qh[l] & (h == 0 ? u1 : u2))) { code += 16; }
            if (!put(64 * j + 32 * h + l, code)) { return false; }
          }
        }
        q += 32;
        u1 <<= 2;
        u2 <<= 2;
      }
      return true;
    }
    case kQ3_K: {
      // d * sc * (q - 4) with the integer product p in [-124, 128]: as
      // code 128 - p about a NEGATIVE scale, so the code always fits.
      if (bits != 8) { return false; }
      const float d = f16_(b + 108);
      int sc[16], q4[256];
      q3k_scales_(b + 96, sc);
      q3k_values_(b, q4);
      for (int k = 0; k < 256; ++k) {
        if (!put(k, 128 - sc[k / 16] * q4[k])) { return false; }
      }
      for (int i = 0; i < 8; ++i) { g[i] = {-d, 128.0f * d, 128.0f}; }
      return true;
    }
    case kIQ4_XS: {
      if (bits != 8) { return false; }
      const float d = f16_(b);
      const std::uint8_t* qs = b + 8;
      for (int ib = 0; ib < 8; ++ib) {
        const float dl = d * iq4xs_scale_(b, ib);
        g[ib] = {dl, -128.0f * dl, 128.0f};
        for (int j = 0; j < 32; ++j) {
          const int kv = k_kvalues_iq4nl[nib_(qs + 16 * ib, j)];
          if (!put(32 * ib + j, kv + 128)) { return false; }
        }
      }
      return true;
    }
    case kIQ2_XXS: {
      // value = db * (+-g), g in {8, 25, 43}: code +-g + off, off a power
      // of two (bias -off * db exact against the stored scale); db has at
      // most 16 significant bits and a code 8, so db * code is exact in
      // f32, as ggml's db * g is.
      if (bits < 7) { return false; }
      const float d = f16_(b);
      const int off = 1 << (bits - 1);
      for (int ib = 0; ib < 8; ++ib) {
        const std::uint8_t* q = b + 2 + 8 * ib;
        const std::uint32_t a1 = u32_(q + 4);
        const float db = d * (0.5f + (a1 >> 28)) * 0.25f;
        g[ib] = {db, -(float)off * db, (float)off};
        for (int l = 0; l < 4; ++l) {
          const std::uint8_t sg = k_ksigns_iq2xs[(a1 >> (7 * l)) & 127];
          for (int j = 0; j < 8; ++j) {
            const int gv = grid_byte_(k_iq2xxs_grid, q[l], j);
            const int a = (sg & k_kmask_iq2xs[j]) ? -gv : gv;
            if (!put(32 * ib + 8 * l + j, a + off)) { return false; }
          }
        }
      }
      return true;
    }
    case kIQ3_XXS: {
      // Grid values are even: value = 2 db * (v / 2), v / 2 in [-31, 31].
      const float d = f16_(b);
      int v[256], s4[8];
      iq3xxs_values_(b, v, s4);
      const int off = 1 << (bits - 1);
      for (int ib = 0; ib < 8; ++ib) {
        const float db2 = d * (0.5f + s4[ib]);   // 2 * db, exactly
        g[ib] = {db2, -(float)off * db2, (float)off};
        for (int j = 0; j < 32; ++j) {
          if (!put(32 * ib + j, v[32 * ib + j] / 2 + off)) { return false; }
        }
      }
      return true;
    }
    case kIQ3_S: {
      // Grid values are odd: at 4 bits value = 2 db * (code - 7.5); wider,
      // value = db * (v + off) with off a power of two.
      const float d = f16_(b);
      int v[256], s4[8];
      iq3s_values_(b, v, s4);
      for (int ib = 0; ib < 8; ++ib) {
        const float db = d * (1 + 2 * s4[ib]);
        if (bits == 4) {
          g[ib] = {2.0f * db, -15.0f * db, 7.5f};
        } else {
          const int off = 1 << (bits - 1);
          g[ib] = {db, -(float)off * db, (float)off};
        }
        for (int j = 0; j < 32; ++j) {
          const int a = v[32 * ib + j];
          const int code =
              bits == 4 ? (a + 15) / 2 : a + (1 << (bits - 1));
          if (!put(32 * ib + j, code)) { return false; }
        }
      }
      return true;
    }
    default:
      return false;
  }
}

// Round-to-nearest-even bf16 / f16 bits of an f32, and back.
inline std::uint16_t
bf16_bits_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7f800000u) == 0x7f800000u) { return (std::uint16_t)(u >> 16); }
  u += 0x7fffu + ((u >> 16) & 1u);
  return (std::uint16_t)(u >> 16);
}
inline float
bf16_f32_(std::uint16_t b)
{
  const std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
inline std::uint16_t
elt_bits_(float f, bool bf16)
{
  if (bf16) { return bf16_bits_(f); }
  const _Float16 h = (_Float16)f;
  std::uint16_t b;
  std::memcpy(&b, &h, 2);
  return b;
}
inline float
elt_f32_(std::uint16_t b, bool bf16)
{
  if (bf16) { return bf16_f32_(b); }
  _Float16 h;
  std::memcpy(&h, &b, 2);
  return (float)h;
}
// The next stored value at or above |s| (so 255 steps cover a range).
inline std::uint16_t
elt_up_(float s, bool bf16)
{
  std::uint16_t b = elt_bits_(s, bf16);
  if (elt_f32_(b, bf16) < s) { ++b; }
  return b;
}

// A group's exact form (codes q, value = scale * q + bias in f32) with
// its scale and bias stored in the compute dtype. The scale rounds to the
// nearest. A symmetric format's bias is -off * s' (exact against the
// stored scale where off is a power of two): every value then moves by
// the scale's relative rounding alone, bf16's own error. A free bias (the
// K-quants, Q5_1) is FITTED to the stored scale -- mean(value - s' q),
// rounded -- which leaves (q - mean q) times the scale's rounding plus
// the bias's own; storing the exact bias instead also carried mean(q)
// times the scale's (MEASURED on Q5_K: 6.0e-3 from the file, 4.5e-3
// fitted, against bf16's 1.6e-3).
inline void
fit_group_(const std::uint8_t* q, int n, const Affine32& f, bool bf16,
           std::uint16_t* s_out, std::uint16_t* b_out)
{
  const std::uint16_t sb = elt_bits_(f.scale, bf16);
  *s_out = sb;
  const float s1 = elt_f32_(sb, bf16);
  if (f.off != 0.0f) {
    *b_out = elt_bits_(-f.off * s1, bf16);
    return;
  }
  int sq = 0;
  for (int j = 0; j < n; ++j) { sq += q[j]; }
  const double mq = (double)sq / n;
  *b_out = elt_bits_((float)(((double)f.scale - s1) * mq + f.bias), bf16);
}

// Decoded values' 32-value groups as 8-bit codes: the bias the group's
// minimum, the scale the next stored value up from its range / 255 (so
// the top of the range is reachable), each code the nearest against those
// two AS STORED.
void
rtn_row_(const float* x, int K, bool bf16, std::uint8_t* c,
         std::uint16_t* s, std::uint16_t* b)
{
  for (int g = 0; g < K / 32; ++g) {
    const float* v = x + g * 32;
    float mn = v[0], mx = v[0];
    for (int j = 1; j < 32; ++j) {
      mn = v[j] < mn ? v[j] : mn;
      mx = v[j] > mx ? v[j] : mx;
    }
    const std::uint16_t bb = elt_bits_(mn, bf16);
    const float bf = elt_f32_(bb, bf16);
    const float range = mx - bf > 0.0f ? mx - bf : 0.0f;
    const std::uint16_t sb = range > 0.0f ? elt_up_(range / 255.0f, bf16)
                                          : 0;
    const float sf = elt_f32_(sb, bf16);
    for (int j = 0; j < 32; ++j) {
      float q = sf > 0.0f ? __builtin_nearbyintf((v[j] - bf) / sf) : 0.0f;
      q = q < 0.0f ? 0.0f : (q > 255.0f ? 255.0f : q);
      c[g * 32 + j] = (std::uint8_t)q;
    }
    s[g] = sb;
    b[g] = bb;
  }
}

}  // namespace

Geom
geom(std::uint32_t type)
{
  switch (type) {
    case kF32: return {1, 4};
    case kF16: return {1, 2};
    case kBF16: return {1, 2};
    case kQ4_0: return {32, 18};
    case kQ5_0: return {32, 22};
    case kQ5_1: return {32, 24};
    case kQ8_0: return {32, 34};
    case kIQ4_NL: return {32, 18};
    case kQ3_K: return {256, 110};
    case kQ4_K: return {256, 144};
    case kQ5_K: return {256, 176};
    case kQ6_K: return {256, 210};
    case kIQ2_XXS: return {256, 66};
    case kIQ2_XS: return {256, 74};
    case kIQ1_M: return {256, 56};
    case kIQ2_S: return {256, 82};
    case kIQ3_XXS: return {256, 98};
    case kIQ3_S: return {256, 110};
    case kIQ4_XS: return {256, 136};
    default: return {};
  }
}

const char*
name(std::uint32_t type)
{
  switch (type) {
    case kF32: return "F32";
    case kF16: return "F16";
    case kBF16: return "BF16";
    case kQ4_0: return "Q4_0";
    case kQ5_0: return "Q5_0";
    case kQ5_1: return "Q5_1";
    case kQ8_0: return "Q8_0";
    case kIQ4_NL: return "IQ4_NL";
    case kQ3_K: return "Q3_K";
    case kQ4_K: return "Q4_K";
    case kQ5_K: return "Q5_K";
    case kQ6_K: return "Q6_K";
    case kIQ2_XXS: return "IQ2_XXS";
    case kIQ2_XS: return "IQ2_XS";
    case kIQ1_M: return "IQ1_M";
    case kIQ2_S: return "IQ2_S";
    case kIQ3_XXS: return "IQ3_XXS";
    case kIQ3_S: return "IQ3_S";
    case kIQ4_XS: return "IQ4_XS";
    default: return "?";
  }
}

bool
decode(std::uint32_t type, const void* src, std::int64_t n, float* out)
{
  const auto* p = static_cast<const std::uint8_t*>(src);
  if (type == kF32) {
    std::memcpy(out, p, (std::size_t)n * 4);
    return true;
  }
  if (type == kF16 || type == kBF16) {
    for (std::int64_t i = 0; i < n; ++i) {
      out[i] = type == kF16 ? f16_(p + 2 * i) : bf16_(p + 2 * i);
    }
    return true;
  }
  const Geom g = geom(type);
  if (g.elems == 0 || n % g.elems != 0) { return false; }
  for (std::int64_t i = 0; i < n / g.elems; ++i) {
    if (!decode_block_(type, p + i * g.bytes, out + i * g.elems)) {
      return false;
    }
  }
  return true;
}

int
affine_bits(std::uint32_t type)
{
  switch (type) {
    case kQ4_0: return 4;
    case kQ4_K: return 4;
    case kIQ3_S: return 4;
    case kQ5_0: return 5;
    case kQ5_1: return 5;
    case kQ5_K: return 5;
    case kIQ3_XXS: return 6;
    case kQ8_0: return 8;
    case kIQ4_NL: return 8;
    case kIQ4_XS: return 8;
    case kQ3_K: return 8;
    case kIQ2_XXS: return 8;
    default: return 0;
  }
}

bool
to_affine32(std::uint32_t type, const void* src, std::int64_t n, int bits,
            std::uint8_t* codes, Affine32* groups)
{
  const int need = affine_bits(type);
  // A wider width keeps the codes (or recentres them about a wider power
  // of two), so it is exact too; a code that does not fit fails `put`.
  if (need == 0 || bits < need || bits > 8) { return false; }
  const Geom g = geom(type);
  if (g.elems == 0 || n % g.elems != 0) { return false; }
  const auto* p = static_cast<const std::uint8_t*>(src);
  for (std::int64_t i = 0; i < n / g.elems; ++i) {
    if (!affine_block_(type, p + i * g.bytes, bits, codes + i * g.elems,
                       groups + i * (g.elems / 32))) {
      return false;
    }
  }
  return true;
}

void
pack_codes(const std::uint8_t* codes, std::int64_t n, int bits,
           std::uint8_t* out)
{
  if (bits == 8) {
    std::memcpy(out, codes, (std::size_t)n);
    return;
  }
  if (bits == 4) {
    for (std::int64_t i = 0; i < n / 2; ++i) {
      out[i] = (std::uint8_t)(codes[2 * i] | codes[2 * i + 1] << 4);
    }
    return;
  }
  std::memset(out, 0, (std::size_t)(n * bits / 8));
  for (std::int64_t j = 0; j < n; ++j) {
    const std::int64_t bit = j * bits;
    const unsigned v = (unsigned)codes[j] << (bit & 7);
    out[bit >> 3] |= (std::uint8_t)v;
    if ((bit & 7) + bits > 8) { out[(bit >> 3) + 1] |= (std::uint8_t)(v >> 8); }
  }
}

bool
row_affine32(std::uint32_t type, const void* src, int K, int bits,
             bool bf16, std::uint8_t* codes, std::uint16_t* scales,
             std::uint16_t* biases)
{
  if (K <= 0 || K % 32 != 0) { return false; }
  const int need = affine_bits(type);
  if (need > 0 && bits >= need) {
    thread_local std::vector<Affine32> a;
    a.resize((std::size_t)K / 32);
    if (!to_affine32(type, src, K, bits, codes, a.data())) { return false; }
    for (int g = 0; g < K / 32; ++g) {
      fit_group_(codes + g * 32, 32, a[(std::size_t)g], bf16, scales + g,
                 biases + g);
    }
    return true;
  }
  if (bits != 8) { return false; }
  thread_local std::vector<float> x;
  x.resize((std::size_t)K);
  if (!decode(type, src, K, x.data())) { return false; }
  rtn_row_(x.data(), K, bf16, codes, scales, biases);
  return true;
}

std::vector<std::uint64_t>
gpu_iq_tables()
{
  std::vector<std::uint64_t> t;
  t.reserve(2832);
  t.insert(t.end(), std::begin(k_iq2xxs_grid), std::end(k_iq2xxs_grid));
  t.insert(t.end(), std::begin(k_iq2xs_grid), std::end(k_iq2xs_grid));
  t.insert(t.end(), std::begin(k_iq1s_grid), std::end(k_iq1s_grid));
  std::uint64_t signs[16] = {};
  std::memcpy(signs, k_ksigns_iq2xs, sizeof k_ksigns_iq2xs);
  t.insert(t.end(), std::begin(signs), std::end(signs));
  return t;
}

std::vector<std::uint64_t>
gpu_iq_tables_native()
{
  std::vector<std::uint64_t> t = gpu_iq_tables();
  // the IQ1_S / IQ1_M grid's 2048 entries as nibbles (value + 1: 0, 1, 2),
  // a u32 each -- byte j holds element j low, element j + 4 high -- two to
  // a u64
  std::vector<std::uint32_t> n(2048);
  for (int i = 0; i < 2048; ++i) {
    std::uint32_t u = 0;
    for (int j = 0; j < 8; ++j) {
      const int v = (int)(std::int8_t)((k_iq1s_grid[i] >> (8 * j)) & 0xff);
      const std::uint32_t nib = (std::uint32_t)(v + 1) & 0xf;
      u |= nib << (j < 4 ? 8 * j : 8 * (j - 4) + 4);
    }
    n[(std::size_t)i] = u;
  }
  const std::size_t at = t.size();
  t.resize(at + 1024);
  std::memcpy(t.data() + at, n.data(), 2048 * 4);
  // then IQ2_S's grid (1024 u64, at 3856) and IQ3_XXS's (256 u32, two to a
  // u64, at 4880)
  t.insert(t.end(), std::begin(k_iq2s_grid), std::end(k_iq2s_grid));
  const std::size_t a3 = t.size();
  t.resize(a3 + 128);
  std::memcpy(t.data() + a3, k_iq3xxs_grid, 256 * 4);
  // then IQ3_S's grid (512 u32, at 5008) and IQ4_NL's 16 values (int8, at
  // 5264)
  const std::size_t a4 = t.size();
  t.resize(a4 + 256 + 2);
  std::memcpy(t.data() + a4, k_iq3s_grid, 512 * 4);
  std::memcpy(t.data() + a4 + 256, k_kvalues_iq4nl, 16);
  return t;
}

bool
gpu_affine8(std::uint32_t type)
{
  return type == kIQ2_XXS || type == kIQ2_XS || type == kIQ1_M;
}

int
affine_width(std::uint32_t type)
{
  const int need = affine_bits(type);
  if (need > 0) { return need; }
  return geom(type).elems > 0 ? 8 : 0;
}

}  // namespace vpipe::genai::ggml

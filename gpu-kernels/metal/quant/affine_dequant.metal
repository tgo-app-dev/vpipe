// affine_dequant.metal -- expand a 4-bit affine-quantized weight matrix
// W[N,K] (row-major, group 64) into a dense VPIPE_ELT matrix
// W_deq[n,k] = scale[n,k/64] * q[n,k] + bias[n,k/64].
//
// Used by the matrix-core prefill path: dequant a projection weight ONCE
// into a reusable device scratch, then run the dense matmul2d GEMM
// (dense_gemm_mma) which reads dense weights directly and drives the
// hardware matrix units far more efficiently than a fused dequant-in-
// threadgroup loop. The dequant is pure bandwidth: one thread expands a
// whole 32-bit weight word (8 nibbles -> 8 outputs) with a SINGLE scale/
// bias load (8 consecutive k share a group, since group 64 % 8 == 0) and
// a single 8-wide vector store, so it runs near peak DRAM bandwidth.
//
//   0:w(uint32) 1:scales 2:biases 3:y_deq(VPIPE_ELT)  4:K 5:N
// One thread per weight word: grid {K/8, N, 1}.

#include <metal_stdlib>
using namespace metal;

#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif
#define DQ_GROUP 64

kernel void affine_dequant_w4g64(
    const device uint32_t*  w      [[buffer(0)]],
    const device VPIPE_ELT* scales [[buffer(1)]],
    const device VPIPE_ELT* biases [[buffer(2)]],
    device VPIPE_ELT*       ydeq   [[buffer(3)]],
    const constant int& K [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int word = (int)gid.x;        // 0 .. K/8-1  (8 nibbles per u32)
  const int n    = (int)gid.y;        // 0 .. N-1
  if (n >= N || word >= (K >> 3)) { return; }
  const int Kw = K >> 3;              // u32 words per weight row
  const int Kg = K / DQ_GROUP;        // scale/bias entries per row
  const int k0 = word << 3;           // first k of this word
  // 8 consecutive k -> same group (group 64 is a multiple of 8): one load.
  const int g = n * Kg + (k0 / DQ_GROUP);
  const float s = (float)scales[g];
  const float b = (float)biases[g];

  const uint32_t packed = w[(int64_t)n * Kw + word];
  // Nibble order matches the steel loader: byte b holds k=2b (low) then
  // k=2b+1 (high); within a u32 the low byte is k0,k0+1.
  VPIPE_ELT out[8];
  for (int i = 0; i < 8; ++i) {
    const uint q = (packed >> (4 * i)) & 0xf;
    out[i] = (VPIPE_ELT)(s * (float)q + b);
  }
  device VPIPE_ELT* dst = ydeq + (int64_t)n * K + k0;
  *reinterpret_cast<device vec<VPIPE_ELT, 4>*>(dst) =
      vec<VPIPE_ELT, 4>(out[0], out[1], out[2], out[3]);
  *reinterpret_cast<device vec<VPIPE_ELT, 4>*>(dst + 4) =
      vec<VPIPE_ELT, 4>(out[4], out[5], out[6], out[7]);
}

// 8-bit twin of affine_dequant_w4g64 (group 64). 8-bit affine weights store
// one BYTE per value (no nibble packing), 4 values per u32 word, so a row is
// K/4 words and a thread expands one word into 4 outputs with a single
// scale/bias load (group 64 % 4 == 0). Byte order is little-endian within the
// word (byte i -> k0+i), matching qmv_fast_impl<,64,8> which reads w as raw
// bytes and computes scale*byte + bias.
//   0:w(uint32) 1:scales 2:biases 3:y_deq(VPIPE_ELT)  4:K 5:N
// One thread per weight word: grid {K/4, N, 1}.
kernel void affine_dequant_w8g64(
    const device uint32_t*  w      [[buffer(0)]],
    const device VPIPE_ELT* scales [[buffer(1)]],
    const device VPIPE_ELT* biases [[buffer(2)]],
    device VPIPE_ELT*       ydeq   [[buffer(3)]],
    const constant int& K [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int word = (int)gid.x;        // 0 .. K/4-1  (4 bytes per u32)
  const int n    = (int)gid.y;        // 0 .. N-1
  if (n >= N || word >= (K >> 2)) { return; }
  const int Kw = K >> 2;              // u32 words per weight row
  const int Kg = K / DQ_GROUP;        // scale/bias entries per row
  const int k0 = word << 2;           // first k of this word
  // 4 consecutive k -> same group (group 64 is a multiple of 4): one load.
  const int g = n * Kg + (k0 / DQ_GROUP);
  const float s = (float)scales[g];
  const float b = (float)biases[g];

  const uint32_t packed = w[(int64_t)n * Kw + word];
  VPIPE_ELT out[4];
  for (int i = 0; i < 4; ++i) {
    const uint q = (packed >> (8 * i)) & 0xff;
    out[i] = (VPIPE_ELT)(s * (float)q + b);
  }
  device VPIPE_ELT* dst = ydeq + (int64_t)n * K + k0;
  *reinterpret_cast<device vec<VPIPE_ELT, 4>*>(dst) =
      vec<VPIPE_ELT, 4>(out[0], out[1], out[2], out[3]);
}

// group_size=32 twin of affine_dequant_w8g64 (the MOSS codec int8 weights).
// Only the scale/bias group stride changes; 32 is a multiple of 4 so the
// 4-consecutive-k-per-u32-word share one group entry, same as the g64 path.
// Used by the codec int8 decode GEMM (dequant-once -> the f16 dense_gemm_mma,
// which beats the fused dequant-in-matmul2d at the codec's prefill-like M).
//   0:w(uint32) 1:scales 2:biases 3:ydeq(f16)  4:K 5:N   grid {K/4, N, 1}.
kernel void affine_dequant_w8g32(
    const device uint32_t*  w      [[buffer(0)]],
    const device VPIPE_ELT* scales [[buffer(1)]],
    const device VPIPE_ELT* biases [[buffer(2)]],
    device VPIPE_ELT*       ydeq   [[buffer(3)]],
    const constant int& K [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int word = (int)gid.x;        // 0 .. K/4-1  (4 bytes per u32)
  const int n    = (int)gid.y;        // 0 .. N-1
  if (n >= N || word >= (K >> 2)) { return; }
  const int Kw = K >> 2;              // u32 words per weight row
  const int Kg = K / 32;              // scale/bias entries per row (group 32)
  const int k0 = word << 2;           // first k of this word
  const int g = n * Kg + (k0 / 32);   // 4 consecutive k -> same group (32%4==0)
  const float s = (float)scales[g];
  const float b = (float)biases[g];

  const uint32_t packed = w[(int64_t)n * Kw + word];
  VPIPE_ELT out[4];
  for (int i = 0; i < 4; ++i) {
    const uint q = (packed >> (8 * i)) & 0xff;
    out[i] = (VPIPE_ELT)(s * (float)q + b);
  }
  device VPIPE_ELT* dst = ydeq + (int64_t)n * K + k0;
  *reinterpret_cast<device vec<VPIPE_ELT, 4>*>(dst) =
      vec<VPIPE_ELT, 4>(out[0], out[1], out[2], out[3]);
}

// Fused 8-bit -> 4-bit affine REQUANT (group 64), no f16 temp. Builds a cheaper
// half-bandwidth DRAFT lm_head from an 8-bit affine weight: per group of 64,
// dequant the 8 packed bytes' values (s8*q8 + b8), take their min/max, and
// re-quantize to 4-bit (scale4 = (max-min)/15, bias4 = min, q4 = round((v-min)/
// scale4)). The MTP draft head reads this 4-bit copy (verifier keeps the 8-bit
// weight), halving its vocab-matrix read -- the verify-corrected output is
// unchanged so the lossy draft quant only costs a little acceptance.
//   0:w8(uint32) 1:s8 2:b8 3:w4(uint32) 4:s4 5:b4  6:K 7:N
// One thread per group of 64: grid {K/64, N, 1}.
kernel void affine_requant_w8_to_w4_g64(
    const device uint32_t*  w8 [[buffer(0)]],
    const device VPIPE_ELT* s8 [[buffer(1)]],
    const device VPIPE_ELT* b8 [[buffer(2)]],
    device uint32_t*        w4 [[buffer(3)]],
    device VPIPE_ELT*       s4 [[buffer(4)]],
    device VPIPE_ELT*       b4 [[buffer(5)]],
    const constant int& K [[buffer(6)]],
    const constant int& N [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int grp = (int)gid.x;         // 0 .. K/64-1
  const int n   = (int)gid.y;         // 0 .. N-1
  const int Kg = K / DQ_GROUP;
  if (n >= N || grp >= Kg) { return; }
  const float s = (float)s8[(int64_t)n * Kg + grp];
  const float b = (float)b8[(int64_t)n * Kg + grp];
  const int Kw8 = K >> 2;             // 8-bit: 4 bytes per word
  const device uint32_t* wp8 = w8 + (int64_t)n * Kw8 + (int64_t)grp * 16;
  float dq[64];
  float mn = INFINITY, mx = -INFINITY;
  for (int wi = 0; wi < 16; ++wi) {
    const uint32_t packed = wp8[wi];
    for (int i = 0; i < 4; ++i) {
      const uint q = (packed >> (8 * i)) & 0xff;
      const float v = s * (float)q + b;
      dq[wi * 4 + i] = v;
      mn = min(mn, v); mx = max(mx, v);
    }
  }
  const float scale = (mx - mn) / 15.0f;
  const float inv = (scale > 0.0f) ? 1.0f / scale : 0.0f;
  s4[(int64_t)n * Kg + grp] = (VPIPE_ELT)scale;
  b4[(int64_t)n * Kg + grp] = (VPIPE_ELT)mn;
  const int Kw4 = K >> 3;             // 4-bit: 8 nibbles per word
  device uint32_t* wp4 = w4 + (int64_t)n * Kw4 + (int64_t)grp * 8;
  for (int wi = 0; wi < 8; ++wi) {
    uint32_t packed = 0;
    for (int i = 0; i < 8; ++i) {
      int q = (int)round((dq[wi * 8 + i] - mn) * inv);
      q = clamp(q, 0, 15);
      packed |= ((uint32_t)q) << (4 * i);
    }
    wp4[wi] = packed;
  }
}

// group_size=32 twin of affine_dequant_w4g64 (GGUF q4_0). Identical to the
// g64 kernel but the scale/bias group stride is 32, so DQ_GROUP_32 is used
// in place of DQ_GROUP. 8 consecutive k still share a group (group 32 is a
// multiple of 8), so the single scale/bias load per word holds.
#define DQ_GROUP_32 32
kernel void affine_dequant_w4g32(
    const device uint32_t*  w      [[buffer(0)]],
    const device VPIPE_ELT* scales [[buffer(1)]],
    const device VPIPE_ELT* biases [[buffer(2)]],
    device VPIPE_ELT*       ydeq   [[buffer(3)]],
    const constant int& K [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int word = (int)gid.x;        // 0 .. K/8-1  (8 nibbles per u32)
  const int n    = (int)gid.y;        // 0 .. N-1
  if (n >= N || word >= (K >> 3)) { return; }
  const int Kw = K >> 3;              // u32 words per weight row
  const int Kg = K / DQ_GROUP_32;     // scale/bias entries per row
  const int k0 = word << 3;           // first k of this word
  // 8 consecutive k -> same group (group 32 is a multiple of 8): one load.
  const int g = n * Kg + (k0 / DQ_GROUP_32);
  const float s = (float)scales[g];
  const float b = (float)biases[g];

  const uint32_t packed = w[(int64_t)n * Kw + word];
  // Nibble order matches the steel loader: byte b holds k=2b (low) then
  // k=2b+1 (high); within a u32 the low byte is k0,k0+1.
  VPIPE_ELT out[8];
  for (int i = 0; i < 8; ++i) {
    const uint q = (packed >> (4 * i)) & 0xf;
    out[i] = (VPIPE_ELT)(s * (float)q + b);
  }
  device VPIPE_ELT* dst = ydeq + (int64_t)n * K + k0;
  *reinterpret_cast<device vec<VPIPE_ELT, 4>*>(dst) =
      vec<VPIPE_ELT, 4>(out[0], out[1], out[2], out[3]);
  *reinterpret_cast<device vec<VPIPE_ELT, 4>*>(dst + 4) =
      vec<VPIPE_ELT, 4>(out[4], out[5], out[6], out[7]);
}

// ---------------------------------------------------------------------------
// FORWARD affine quantization: dense W[N,K] (VPIPE_ELT, row-major) -> packed
// U32 codes + F16 scales + F16 biases, asymmetric per group:
//   scale = (max - min) / ((1<<BITS)-1),  bias = min,
//   q = clamp(round((w - min) / scale), 0, (1<<BITS)-1).
// The exact inverse of affine_dequant_* above; the byte (8-bit) / nibble
// (4-bit) pack order is identical, so a subsequent dequant (or the qmv /
// steel loaders) reproduces W_q. scales/biases are emitted as `half`
// REGARDLESS of VPIPE_ELT so the on-disk format stays F16 (MLX-affine).
// One thread per group: grid {K/GROUP, N, 1}.  `clip` (1.0 = plain min/max)
// shrinks the captured range toward the group midpoint for an outlier-robust
// (unsloth-style) fit; the AWQ clip search drives it below 1.0.
template <int BITS, int GROUP>
static inline void
affine_quant_impl(const device VPIPE_ELT* x, device uint32_t* w,
                  device half* s, device half* b, int K, int N, float clip,
                  uint2 gid)
{
  const int grp = (int)gid.x;          // 0 .. K/GROUP-1
  const int n   = (int)gid.y;          // 0 .. N-1
  const int KG  = K / GROUP;           // groups per row
  if (n >= N || grp >= KG) { return; }
  constexpr int VPW  = 32 / BITS;      // values per u32 word (8b:4, 4b:8)
  constexpr int WPG  = GROUP / VPW;    // u32 words per group
  constexpr int QMAX = (1 << BITS) - 1;
  const int WPR = K * BITS / 32;       // u32 words per row

  const device VPIPE_ELT* xp = x + (int64_t)n * K + (int64_t)grp * GROUP;
  float mn = INFINITY, mx = -INFINITY;
  for (int i = 0; i < GROUP; ++i) {
    const float v = (float)xp[i];
    mn = min(mn, v); mx = max(mx, v);
  }
  if (clip < 1.0f) {
    const float mid = 0.5f * (mn + mx);
    mn = mid + (mn - mid) * clip;
    mx = mid + (mx - mid) * clip;
  }
  const float scale = (mx - mn) / (float)QMAX;
  const float inv   = (scale > 0.0f) ? 1.0f / scale : 0.0f;
  s[(int64_t)n * KG + grp] = (half)scale;
  b[(int64_t)n * KG + grp] = (half)mn;

  device uint32_t* wp = w + (int64_t)n * WPR + (int64_t)grp * WPG;
  for (int wi = 0; wi < WPG; ++wi) {
    uint32_t packed = 0;
    for (int i = 0; i < VPW; ++i) {
      const float v = (float)xp[wi * VPW + i];
      int q = (int)round((v - mn) * inv);
      q = clamp(q, 0, QMAX);
      packed |= ((uint32_t)q) << (BITS * i);
    }
    wp[wi] = packed;
  }
}

#define VPIPE_AFFINE_QUANT(NAME, BITS, GROUP)                              \
  kernel void NAME(const device VPIPE_ELT* x [[buffer(0)]],               \
                   device uint32_t*  w    [[buffer(1)]],                   \
                   device half*      s    [[buffer(2)]],                   \
                   device half*      b    [[buffer(3)]],                   \
                   const constant int&   K    [[buffer(4)]],              \
                   const constant int&   N    [[buffer(5)]],              \
                   const constant float& clip [[buffer(6)]],              \
                   uint2 gid [[thread_position_in_grid]])                  \
  { affine_quant_impl<BITS, GROUP>(x, w, s, b, K, N, clip, gid); }

VPIPE_AFFINE_QUANT(affine_quant_w8_g64, 8, 64)
VPIPE_AFFINE_QUANT(affine_quant_w4_g64, 4, 64)
VPIPE_AFFINE_QUANT(affine_quant_w8_g32, 8, 32)
VPIPE_AFFINE_QUANT(affine_quant_w4_g32, 4, 32)

// FIT TO THE STORED PAIR (model-quantize's run_dtype): the same affine
// quantization as affine_quant_impl, but the scale and bias are rounded
// to ST -- the type the model that runs the pack holds them in, half or
// bfloat -- FIRST, written in it, and the codes fit to the rounded pair,
// as affine_quant_rows_w8g64 does for the in-memory build. Fit to the f32
// scale and stored rounded (affine_quant_impl), the codes sit up to ~half
// a step off the scale the dequant uses (255 * 2^-9 relative for bfloat).
// `clip` as there. One thread per group: grid {K/GROUP, N, 1}.
template <int BITS, int GROUP, typename ST>
static inline void
affine_quant_fit_impl(const device VPIPE_ELT* x, device uint32_t* w,
                      device ST* s, device ST* b, int K, int N, float clip,
                      uint2 gid)
{
  const int grp = (int)gid.x;
  const int n   = (int)gid.y;
  const int KG  = K / GROUP;
  if (n >= N || grp >= KG) { return; }
  constexpr int VPW  = 32 / BITS;
  constexpr int WPG  = GROUP / VPW;
  constexpr int QMAX = (1 << BITS) - 1;
  const int WPR = K * BITS / 32;

  const device VPIPE_ELT* xp = x + (int64_t)n * K + (int64_t)grp * GROUP;
  float mn = INFINITY, mx = -INFINITY;
  for (int i = 0; i < GROUP; ++i) {
    const float v = (float)xp[i];
    mn = min(mn, v); mx = max(mx, v);
  }
  if (clip < 1.0f) {
    const float mid = 0.5f * (mn + mx);
    mn = mid + (mn - mid) * clip;
    mx = mid + (mx - mid) * clip;
  }
  const ST se = (ST)((mx - mn) / (float)QMAX);
  const ST be = (ST)mn;
  s[(int64_t)n * KG + grp] = se;
  b[(int64_t)n * KG + grp] = be;
  const float sf  = (float)se;
  const float bf  = (float)be;
  const float inv = (sf > 0.0f) ? 1.0f / sf : 0.0f;

  device uint32_t* wp = w + (int64_t)n * WPR + (int64_t)grp * WPG;
  for (int wi = 0; wi < WPG; ++wi) {
    uint32_t packed = 0;
    for (int i = 0; i < VPW; ++i) {
      const float v = (float)xp[wi * VPW + i];
      const int q = clamp((int)round((v - bf) * inv), 0, QMAX);
      packed |= ((uint32_t)q) << (BITS * i);
    }
    wp[wi] = packed;
  }
}

#define VPIPE_AFFINE_QUANT_FIT(NAME, BITS, GROUP, ST)                      \
  kernel void NAME(const device VPIPE_ELT* x [[buffer(0)]],               \
                   device uint32_t*  w    [[buffer(1)]],                   \
                   device ST*        s    [[buffer(2)]],                   \
                   device ST*        b    [[buffer(3)]],                   \
                   const constant int&   K    [[buffer(4)]],              \
                   const constant int&   N    [[buffer(5)]],              \
                   const constant float& clip [[buffer(6)]],              \
                   uint2 gid [[thread_position_in_grid]])                  \
  { affine_quant_fit_impl<BITS, GROUP, ST>(x, w, s, b, K, N, clip, gid); }

VPIPE_AFFINE_QUANT_FIT(affine_quant_fit_w8_g64_f16s, 8, 64, half)
VPIPE_AFFINE_QUANT_FIT(affine_quant_fit_w4_g64_f16s, 4, 64, half)
VPIPE_AFFINE_QUANT_FIT(affine_quant_fit_w8_g32_f16s, 8, 32, half)
VPIPE_AFFINE_QUANT_FIT(affine_quant_fit_w4_g32_f16s, 4, 32, half)
VPIPE_AFFINE_QUANT_FIT(affine_quant_fit_w8_g64_bf16s, 8, 64, bfloat)
VPIPE_AFFINE_QUANT_FIT(affine_quant_fit_w4_g64_bf16s, 4, 64, bfloat)
VPIPE_AFFINE_QUANT_FIT(affine_quant_fit_w8_g32_bf16s, 8, 32, bfloat)
VPIPE_AFFINE_QUANT_FIT(affine_quant_fit_w4_g32_bf16s, 4, 32, bfloat)

// ---------------------------------------------------------------------------
// IN-MEMORY affine w8 group-64 quantization, for a model that quantizes its
// raw checkpoint at load (and re-quantizes a streamed layer per use) rather
// than reading a pack. Two differences from affine_quant_impl above:
//
//   * scales / biases are written in VPIPE_ELT -- the element type the qmv /
//     steel kernels read -- not F16, so nothing narrows them afterwards;
//   * the codes are computed against the ROUNDED scale and bias. A bf16
//     scale carries 8 bits, so quantizing against the f32 scale and then
//     storing it rounded moves the top code by up to ~half a step
//     (255 * 2^-9); fitting the codes to the stored pair keeps every value
//     within half a step of the scale the dequant actually uses.
//
// Output row r of the [N, K] source lands at row r * row_mul + row_add of
// the destination, so one dispatch per projection writes straight into a
// fused q|k|v matrix (row_add = rows before it) or an interleaved gate|up
// one (row_mul 2, row_add 0 / 1). Byte order per u32 word as
// affine_dequant_w8g64.
//   0:x(VPIPE_ELT [N,K]) 1:w(u32) 2:scales 3:biases (VPIPE_ELT)
//   4:K 5:N 6:row_mul 7:row_add       grid {K/64, N, 1}
kernel void affine_quant_rows_w8g64(
    const device VPIPE_ELT* x      [[buffer(0)]],
    device uint32_t*        w      [[buffer(1)]],
    device VPIPE_ELT*       s      [[buffer(2)]],
    device VPIPE_ELT*       b      [[buffer(3)]],
    const constant int&     K      [[buffer(4)]],
    const constant int&     N      [[buffer(5)]],
    const constant int&     row_mul [[buffer(6)]],
    const constant int&     row_add [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int grp = (int)gid.x;           // 0 .. K/64-1
  const int n   = (int)gid.y;           // 0 .. N-1
  const int KG  = K / 64;
  if (n >= N || grp >= KG) { return; }
  const int64_t dn = (int64_t)n * row_mul + row_add;
  const device VPIPE_ELT* xp = x + (int64_t)n * K + (int64_t)grp * 64;
  float mn = INFINITY, mx = -INFINITY;
  for (int i = 0; i < 64; ++i) {
    const float v = (float)xp[i];
    mn = min(mn, v); mx = max(mx, v);
  }
  const VPIPE_ELT se = (VPIPE_ELT)((mx - mn) / 255.0f);
  const VPIPE_ELT be = (VPIPE_ELT)mn;
  s[dn * KG + grp] = se;
  b[dn * KG + grp] = be;
  const float sf  = (float)se;
  const float bf  = (float)be;
  const float inv = (sf > 0.0f) ? 1.0f / sf : 0.0f;
  device uint32_t* wp = w + dn * (K / 4) + (int64_t)grp * 16;
  for (int wi = 0; wi < 16; ++wi) {
    uint32_t packed = 0;
    for (int i = 0; i < 4; ++i) {
      const float v = (float)xp[wi * 4 + i];
      const int q = clamp((int)round((v - bf) * inv), 0, 255);
      packed |= ((uint32_t)q) << (8 * i);
    }
    wp[wi] = packed;
  }
}

// The 4-bit twin of affine_quant_rows_w8g64: same contract (VPIPE_ELT
// scales / biases, codes fit to the ROUNDED pair, rows placed at
// r * row_mul + row_add), 15 steps a group, eight codes a word at nibble
// i << 4i -- the order affine_qmv / affine_qmm read. Used for a model that
// holds a raw checkpoint at 4 bits in memory (the DFlash drafter's w4).
//   0:x(VPIPE_ELT [N,K]) 1:w(u32) 2:scales 3:biases (VPIPE_ELT)
//   4:K 5:N 6:row_mul 7:row_add       grid {K/64, N, 1}
kernel void affine_quant_rows_w4g64(
    const device VPIPE_ELT* x      [[buffer(0)]],
    device uint32_t*        w      [[buffer(1)]],
    device VPIPE_ELT*       s      [[buffer(2)]],
    device VPIPE_ELT*       b      [[buffer(3)]],
    const constant int&     K      [[buffer(4)]],
    const constant int&     N      [[buffer(5)]],
    const constant int&     row_mul [[buffer(6)]],
    const constant int&     row_add [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int grp = (int)gid.x;           // 0 .. K/64-1
  const int n   = (int)gid.y;           // 0 .. N-1
  const int KG  = K / 64;
  if (n >= N || grp >= KG) { return; }
  const int64_t dn = (int64_t)n * row_mul + row_add;
  const device VPIPE_ELT* xp = x + (int64_t)n * K + (int64_t)grp * 64;
  float mn = INFINITY, mx = -INFINITY;
  for (int i = 0; i < 64; ++i) {
    const float v = (float)xp[i];
    mn = min(mn, v); mx = max(mx, v);
  }
  const VPIPE_ELT se = (VPIPE_ELT)((mx - mn) / 15.0f);
  const VPIPE_ELT be = (VPIPE_ELT)mn;
  s[dn * KG + grp] = se;
  b[dn * KG + grp] = be;
  const float sf  = (float)se;
  const float bf  = (float)be;
  const float inv = (sf > 0.0f) ? 1.0f / sf : 0.0f;
  device uint32_t* wp = w + dn * (K / 8) + (int64_t)grp * 8;
  for (int wi = 0; wi < 8; ++wi) {
    uint32_t packed = 0;
    for (int i = 0; i < 8; ++i) {
      const float v = (float)xp[wi * 8 + i];
      const int q = clamp((int)round((v - bf) * inv), 0, 15);
      packed |= ((uint32_t)q) << (4 * i);
    }
    wp[wi] = packed;
  }
}

// ---------------------------------------------------------------------------
// ROW PLACEMENT for a model that STREAMS an affine pack: the pack stores q,
// k, v and gate, up as separate [N, K] tensors, and the forward reads them
// fused (q|k|v by rows) and interleaved (gate|up, row_mul 2). Source row r
// lands at destination row r * row_mul + row_add, the same map as
// affine_quant_rows_w8g64 above, so a streamed pack layer ends up in the
// layout a pack loaded resident has. Type-blind: one call each for the
// codes, the scales and the biases, `row_words` 32-bit words per row.
//   0:src 1:dst 2:row_words 3:row_mul 4:row_add   grid {row_words, N, 1}
kernel void copy_rows_u32(
    device const uint*  src       [[buffer(0)]],
    device uint*        dst       [[buffer(1)]],
    const constant int& row_words [[buffer(2)]],
    const constant int& row_mul   [[buffer(3)]],
    const constant int& row_add   [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
  if ((int)gid.x >= row_words) { return; }
  const ulong r = gid.y;
  dst[(r * (ulong)row_mul + (ulong)row_add) * (ulong)row_words + gid.x] =
      src[r * (ulong)row_words + gid.x];
}

// ---------------------------------------------------------------------
// Block-floating-point f16 -> i8 quantization (group 64), feeding the
// int8 convolution2d/matmul paths: one simdgroup per 64-element block
// (each lane a half2), PURE INTEGER pipeline on the f16 bit patterns --
// no fdiv, no float rounding:
//   1. block max EXPONENT Emax = simd_max over the raw 5-bit exponents;
//   2. per element recover the 11-bit magnitude (10-bit mantissa with
//      the implicit leading one), right-shift by 4 + (Emax - e) (4 fits
//      the 11-bit magnitude into 7 bits at e == Emax) with round-to-
//      nearest (add half the shifted-out range), clamp 127, apply sign;
//   3. scale = 2^(Emax - 21) per block, EXACT in f16 (normal for
//      Emax >= 7, denormal below), so dequant q * scale reproduces the
//      top 7 magnitude bits exactly.
// Power-of-2 scales give up <= 1 bit of precision vs an amax scale (the
// amax variant below is the float baseline: simd_max(|x|), q =
// rint(x * 127/amax), scale = amax/127). Denormals/zeros (e == 0)
// quantize to 0; NaN/Inf are not handled (activations are finite).
// f16-format-specific (explicit half): the bf16 metallib twin compiles
// but is meaningless -- load these from the f16 lib only.
//   0:x[n] f16  1:q[n] i8  2:scales[n/64] f16  3:n (n % 64 == 0)
//   dispatch (threads): {n/2, 1, 1}, tg {128, 1, 1}
kernel void quant_f16_i8_g64_bfp(
    const device half* x      [[buffer(0)]],
    device char*       q      [[buffer(1)]],
    device half*       scales [[buffer(2)]],
    const constant int& n     [[buffer(3)]],
    uint tid  [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
  const uint base = tid * 2;                  // == simdgroup*64 + lane*2
  if (base + 1 >= (uint)n) { return; }
  const half2 h = *reinterpret_cast<const device half2*>(x + base);
  const ushort2 u = as_type<ushort2>(h);
  const ushort e0 = (u.x >> 10) & 0x1F, e1 = (u.y >> 10) & 0x1F;
  const ushort emax = simd_max(max(e0, e1));

  char2 o;
  {
    // Element 0: 11-bit magnitude, shift by the exponent gap (+4), round.
    const ushort m = (u.x & 0x3FF) | 0x400;
    const ushort s = 4 + (emax - e0);
    int v = (e0 == 0 || s > 14) ? 0 : (int)((m + (1 << (s - 1))) >> s);
    v = min(v, 127);
    o.x = (char)((u.x & 0x8000) ? -v : v);
  }
  {
    const ushort m = (u.y & 0x3FF) | 0x400;
    const ushort s = 4 + (emax - e1);
    int v = (e1 == 0 || s > 14) ? 0 : (int)((m + (1 << (s - 1))) >> s);
    v = min(v, 127);
    o.y = (char)((u.y & 0x8000) ? -v : v);
  }
  *reinterpret_cast<device char2*>(q + base) = o;

  if (lane == 0) {
    // scale = 2^(emax - 21): f16-normal for emax >= 7 (exponent field
    // emax - 6), denormal 1 << (emax + 3) below. Exact either way.
    const ushort sbits = (emax >= 7) ? (ushort)((emax - 6) << 10)
                                     : (ushort)(1 << (emax + 3));
    scales[base / 64] = as_type<half>(sbits);
  }
}

// EXPONENT-emitting twin of quant_f16_i8_g64_bfp, for the shift-aligned
// integer-accumulate GEMMs (gemm_i8i8_sc_f16_n64_g64ri): the same
// integer pipeline and the same q, but buffer 2 is the raw exponent
// Emax - 21 as int8 (range [-21, 9]) instead of the f16 scale 2^(Emax-21).
// Over a row-major [M, K] matrix with K % 64 == 0 the exponents land as
// [M, K/64] row-major, which is the GEMM's sa layout.
//   0:x[n] f16  1:q[n] i8  2:eexp[n/64] i8  3:n (n % 64 == 0)
//   dispatch (threads): {n/2, 1, 1}, tg {128, 1, 1}
kernel void quant_f16_i8_g64_bfpe(
    const device half* x      [[buffer(0)]],
    device char*       q      [[buffer(1)]],
    device char*       eexp   [[buffer(2)]],
    const constant int& n     [[buffer(3)]],
    uint tid  [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
  const uint base = tid * 2;
  if (base + 1 >= (uint)n) { return; }
  const half2 h = *reinterpret_cast<const device half2*>(x + base);
  const ushort2 u = as_type<ushort2>(h);
  const ushort e0 = (u.x >> 10) & 0x1F, e1 = (u.y >> 10) & 0x1F;
  const ushort emax = simd_max(max(e0, e1));

  char2 o;
  {
    const ushort m = (u.x & 0x3FF) | 0x400;
    const ushort s = 4 + (emax - e0);
    int v = (e0 == 0 || s > 14) ? 0 : (int)((m + (1 << (s - 1))) >> s);
    v = min(v, 127);
    o.x = (char)((u.x & 0x8000) ? -v : v);
  }
  {
    const ushort m = (u.y & 0x3FF) | 0x400;
    const ushort s = 4 + (emax - e1);
    int v = (e1 == 0 || s > 14) ? 0 : (int)((m + (1 << (s - 1))) >> s);
    v = min(v, 127);
    o.y = (char)((u.y & 0x8000) ? -v : v);
  }
  *reinterpret_cast<device char2*>(q + base) = o;
  if (lane == 0) { eexp[base / 64] = (char)((int)emax - 21); }
}

// Float amax baseline for the A/B: same block/grid contract.
kernel void quant_f16_i8_g64_amax(
    const device half* x      [[buffer(0)]],
    device char*       q      [[buffer(1)]],
    device half*       scales [[buffer(2)]],
    const constant int& n     [[buffer(3)]],
    uint tid  [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
  const uint base = tid * 2;
  if (base + 1 >= (uint)n) { return; }
  const half2 h = *reinterpret_cast<const device half2*>(x + base);
  const float a0 = fabs((float)h.x), a1 = fabs((float)h.y);
  const float amax = simd_max(max(a0, a1));
  const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
  char2 o;
  o.x = (char)clamp((int)rint((float)h.x * inv), -127, 127);
  o.y = (char)clamp((int)rint((float)h.y * inv), -127, 127);
  *reinterpret_cast<device char2*>(q + base) = o;
  if (lane == 0) { scales[base / 64] = (half)(amax * (1.0f / 127.0f)); }
}

// bfp2: SAME block-floating-point quantization (block max exponent,
// power-of-2 scale) but the mantissa shift runs on the FLOAT pipe: x *
// 2^(21 - Emax) is EXACT (power-of-2 multiply), so rint() computes the
// identical shifted-and-rounded magnitude the integer version derives by
// bit surgery -- at amax-kernel speed (the pure-integer path's variable
// shifts run ~27% below the bandwidth roofline). The inverse scale is
// built in f32 (2^(21-Emax) overflows f16 for Emax < 6); the stored
// per-block scale is the same exact f16 2^(Emax-21).
kernel void quant_f16_i8_g64_bfp2(
    const device half* x      [[buffer(0)]],
    device char*       q      [[buffer(1)]],
    device half*       scales [[buffer(2)]],
    const constant int& n     [[buffer(3)]],
    uint tid  [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
  const uint base = tid * 2;
  if (base + 1 >= (uint)n) { return; }
  const half2 h = *reinterpret_cast<const device half2*>(x + base);
  const ushort2 u = as_type<ushort2>(h);
  const ushort emax =
      simd_max(max((u.x >> 10) & 0x1F, (u.y >> 10) & 0x1F));
  // 2^(21 - emax) exact in f32; multiply cannot round, rint() then does
  // the round-to-nearest the integer path adds by hand.
  const float inv = as_type<float>((uint)(21 - (int)emax + 127) << 23);
  char2 o;
  o.x = (char)clamp((int)rint((float)h.x * inv), -127, 127);
  o.y = (char)clamp((int)rint((float)h.y * inv), -127, 127);
  *reinterpret_cast<device char2*>(q + base) = o;
  if (lane == 0) {
    const ushort sbits = (emax >= 7) ? (ushort)((emax - 6) << 10)
                                     : (ushort)(1 << (emax + 3));
    scales[base / 64] = as_type<half>(sbits);
  }
}

// Per-ROW f16 -> i8 quantization for the int8 GEMM path: the row IS the
// GEMM's K contraction, so ONE scale rides the whole integer dot product
// (the hw matmul accumulates int32 over full K -- scales cannot vary
// along K). One threadgroup (256 threads) per row: amax reduction
// (simd_max + tgmem cross-simdgroup fold), scale = amax/127, then a
// strided quantize pass. K must be even.
//   0:x[M,K] f16  1:q[M,K] i8  2:scales[M] f16  3:K
//   dispatch (threads): {256, M, 1}, tg {256, 1, 1}
kernel void quant_f16_i8_row(
    const device half* x      [[buffer(0)]],
    device char*       q      [[buffer(1)]],
    device half*       scales [[buffer(2)]],
    const constant int& K     [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
  const int row = (int)tgid.y;
  const device half* xr = x + (int64_t)row * K;
  device char* qr = q + (int64_t)row * K;

  float am = 0.0f;
  for (int i = (int)lid * 2; i < K; i += 512) {
    const half2 h = *reinterpret_cast<const device half2*>(xr + i);
    am = max(am, max(fabs((float)h.x), fabs((float)h.y)));
  }
  am = simd_max(am);
  threadgroup float sm[8];
  if (lane == 0) { sm[sgid] = am; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  am = max(max(sm[0], sm[1]), max(sm[2], sm[3]));
  am = max(am, max(max(sm[4], sm[5]), max(sm[6], sm[7])));

  const float inv = am > 0.0f ? 127.0f / am : 0.0f;
  for (int i = (int)lid * 2; i < K; i += 512) {
    const half2 h = *reinterpret_cast<const device half2*>(xr + i);
    char2 o;
    o.x = (char)clamp((int)rint((float)h.x * inv), -127, 127);
    o.y = (char)clamp((int)rint((float)h.y * inv), -127, 127);
    *reinterpret_cast<device char2*>(qr + i) = o;
  }
  if (lid == 0) { scales[row] = (half)(am * (1.0f / 127.0f)); }
}

// Group-512 per-row VPIPE_ELT -> i8 quant for the K-chunked int8 GEMM: scales
// vary along K in groups of 512 (finer than per-row = better outlier
// isolation), one SIMDGROUP per group (32 lanes x 16 elems = 512; pure
// simd_max, no tgmem). scales layout [M, K/512]. K % 512 == 0. The element
// type is VPIPE_ELT (half or bfloat -- the FLUX.2 bf16 DiT feeds this its
// bf16 activations/weights; a bf16 group's absmax can exceed f16's 65504, so
// the derived scale is stored VPIPE_ELT, not half, to avoid overflow).
//   0:x[M,K] 1:q[M,K] 2:scales[M,K/512] 3:K
//   dispatch (threads): {256, M, 1}, tg {256, 1, 1}
kernel void quant_f16_i8_row_g512(
    const device VPIPE_ELT* x      [[buffer(0)]],
    device char*            q      [[buffer(1)]],
    device VPIPE_ELT*       scales [[buffer(2)]],
    const constant int& K     [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
  const int row = (int)tgid.y;
  const int G = K / 512;
  for (int g = (int)sgid; g < G; g += 8) {
    const int64_t base = (int64_t)row * K + g * 512 + (int)lane * 16;
    const device VPIPE_ELT* xp = x + base;
    float am = 0.0f;
    vec<VPIPE_ELT, 2> h[8];
    for (int i = 0; i < 8; ++i) {
      h[i] = *reinterpret_cast<const device vec<VPIPE_ELT, 2>*>(xp + i * 2);
      am = max(am, max(fabs((float)h[i].x), fabs((float)h[i].y)));
    }
    am = simd_max(am);
    const float inv = am > 0.0f ? 127.0f / am : 0.0f;
    device char* qp = q + base;
    for (int i = 0; i < 8; ++i) {
      char2 o;
      o.x = (char)clamp((int)rint((float)h[i].x * inv), -127, 127);
      o.y = (char)clamp((int)rint((float)h[i].y * inv), -127, 127);
      *reinterpret_cast<device char2*>(qp + i * 2) = o;
    }
    if (lane == 0) {
      scales[(int64_t)row * G + g] = (VPIPE_ELT)(am * (1.0f / 127.0f));
    }
  }
}

// K-PADDING twin of quant_f16_i8_row_g512: reads a row of Ksrc elements and
// writes Kpad of them, zero-filling the tail, where Kpad is Ksrc rounded up
// to 512.
//
// This exists so a projection whose contraction is NOT a multiple of 512 can
// still take the int8 GEMM. MiniMax-H3 is the case: its hidden is 5376, so
// qkv and fc1 (K = hidden) missed the i8 path entirely while fc2 (K = ffn =
// 14336) took it -- half the block's GEMM FLOPs excluded by an alignment
// rule rather than by anything about the math.
//
// Zero padding is EXACT here, which is why it is padding and not a
// resampling: the added elements contribute nothing to the dot product, and
// they cannot move the group's scale either, since the scale is an absmax
// and |0| never wins. The tail group therefore quantizes exactly as it would
// have over its real elements alone.
//
// The bounds test is per element pair and not per group on purpose: the row
// stride differs between source and destination, so the last group of EVERY
// row straddles the boundary, and reading past Ksrc would pull in the next
// row (or run off the buffer on the last one).
//   0:x[M,Ksrc] 1:q[M,Kpad] 2:scales[M,Kpad/512] 3:Ksrc 4:Kpad
//   dispatch (threads): {256, M, 1}, tg {256, 1, 1}
kernel void quant_f16_i8_row_g512_pad(
    const device VPIPE_ELT* x      [[buffer(0)]],
    device char*            q      [[buffer(1)]],
    device VPIPE_ELT*       scales [[buffer(2)]],
    const constant int& Ksrc  [[buffer(3)]],
    const constant int& Kpad  [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
  const int row = (int)tgid.y;
  const int G = Kpad / 512;
  for (int g = (int)sgid; g < G; g += 8) {
    const int col0 = g * 512 + (int)lane * 16;
    const int64_t src = (int64_t)row * Ksrc + col0;
    const int64_t dst = (int64_t)row * Kpad + col0;
    float am = 0.0f;
    vec<VPIPE_ELT, 2> h[8];
    for (int i = 0; i < 8; ++i) {
      const int c = col0 + i * 2;
      if (c + 1 < Ksrc) {
        h[i] = *reinterpret_cast<const device vec<VPIPE_ELT, 2>*>(
            x + src + i * 2);
      } else if (c < Ksrc) {
        h[i] = vec<VPIPE_ELT, 2>((VPIPE_ELT)x[src + i * 2], (VPIPE_ELT)0);
      } else {
        h[i] = vec<VPIPE_ELT, 2>((VPIPE_ELT)0, (VPIPE_ELT)0);
      }
      am = max(am, max(fabs((float)h[i].x), fabs((float)h[i].y)));
    }
    am = simd_max(am);
    const float inv = am > 0.0f ? 127.0f / am : 0.0f;
    device char* qp = q + dst;
    for (int i = 0; i < 8; ++i) {
      char2 o;
      o.x = (char)clamp((int)rint((float)h[i].x * inv), -127, 127);
      o.y = (char)clamp((int)rint((float)h[i].y * inv), -127, 127);
      *reinterpret_cast<device char2*>(qp + i * 2) = o;
    }
    if (lane == 0) {
      scales[(int64_t)row * G + g] = (VPIPE_ELT)(am * (1.0f / 127.0f));
    }
  }
}

// GROUP-32 / 64 / 128 twins of quant_f16_i8_row_g512 / _pad, for the
// register-resident int8 GEMM (gemm_i8i8_sc_f16_n64_g{32,64,128}rs),
// which contracts in KC-wide groups on both operands. Same row dispatch
// ({256, M, 1}, eight simdgroups per row), one simdgroup per group: 32
// lanes x KC/32 elements. Scales layout [M, Kpad/KC]. The padding twin
// zero-fills to Kpad, which is exact for the reason given at
// quant_f16_i8_row_g512_pad.
//   0:x[M,Ksrc] 1:q[M,Kpad] 2:scales[M,Kpad/KC] 3:Ksrc (4:Kpad, _pad)
template <int KC, bool PAD>
static inline void quant_row_gk_impl(
    const device VPIPE_ELT* x, device char* q, device VPIPE_ELT* scales,
    int Ksrc, int Kpad, uint2 tgid, uint sgid, uint lane)
{
  static_assert(KC == 32 || KC == 64 || KC == 128,
                "one, two or four elements per lane");
  constexpr int EPL = KC / 32;
  const int row = (int)tgid.y;
  const int G = Kpad / KC;
  for (int g = (int)sgid; g < G; g += 8) {
    const int col = g * KC + (int)lane * EPL;
    const int64_t src = (int64_t)row * Ksrc + col;
    const int64_t dst = (int64_t)row * Kpad + col;
    float h[EPL];
    if (!PAD || col + EPL - 1 < Ksrc) {
      if constexpr (EPL == 1) {
        h[0] = (float)x[src];
      } else {
        const vec<VPIPE_ELT, EPL> v =
            *reinterpret_cast<const device vec<VPIPE_ELT, EPL>*>(x + src);
        for (int i = 0; i < EPL; ++i) { h[i] = (float)v[i]; }
      }
    } else {
      // The last group of a padded row straddles the source's end: per
      // element, and zero past it.
      for (int i = 0; i < EPL; ++i) {
        h[i] = (col + i < Ksrc) ? (float)x[src + i] : 0.0f;
      }
    }
    float am = 0.0f;
    for (int i = 0; i < EPL; ++i) { am = max(am, fabs(h[i])); }
    am = simd_max(am);
    const float inv = am > 0.0f ? 127.0f / am : 0.0f;
    if constexpr (EPL == 1) {
      q[dst] = (char)clamp((int)rint(h[0] * inv), -127, 127);
    } else {
      vec<char, EPL> o;
      for (int i = 0; i < EPL; ++i) {
        o[i] = (char)clamp((int)rint(h[i] * inv), -127, 127);
      }
      *reinterpret_cast<device vec<char, EPL>*>(q + dst) = o;
    }
    if (lane == 0) {
      scales[(int64_t)row * G + g] = (VPIPE_ELT)(am * (1.0f / 127.0f));
    }
  }
}

#define VPIPE_QUANT_ROW_GK(NAME, KC)                                     \
  kernel void NAME(                                                      \
      const device VPIPE_ELT* x [[buffer(0)]],                           \
      device char* q [[buffer(1)]],                                      \
      device VPIPE_ELT* scales [[buffer(2)]],                            \
      const constant int& K [[buffer(3)]],                              \
      uint2 tgid [[threadgroup_position_in_grid]],                       \
      uint sgid [[simdgroup_index_in_threadgroup]],                      \
      uint lane [[thread_index_in_simdgroup]]) {                         \
    quant_row_gk_impl<KC, false>(x, q, scales, K, K, tgid, sgid, lane);  \
  }                                                                      \
  kernel void NAME##_pad(                                                \
      const device VPIPE_ELT* x [[buffer(0)]],                           \
      device char* q [[buffer(1)]],                                      \
      device VPIPE_ELT* scales [[buffer(2)]],                            \
      const constant int& Ksrc [[buffer(3)]],                           \
      const constant int& Kpad [[buffer(4)]],                           \
      uint2 tgid [[threadgroup_position_in_grid]],                       \
      uint sgid [[simdgroup_index_in_threadgroup]],                      \
      uint lane [[thread_index_in_simdgroup]]) {                         \
    quant_row_gk_impl<KC, true>(x, q, scales, Ksrc, Kpad, tgid, sgid,    \
                                lane);                                   \
  }

VPIPE_QUANT_ROW_GK(quant_f16_i8_row_g64, 64)
VPIPE_QUANT_ROW_GK(quant_f16_i8_row_g32, 32)
VPIPE_QUANT_ROW_GK(quant_f16_i8_row_g128, 128)

// UNSIGNED twin of quant_f16_i8_row_g64 for the native affine GEMM
// (gemm_u8q_*): the same symmetric int8 code stored as q + 128 -- the
// matrix units' unsigned x unsigned form -- plus, per (row, group), the
// SUM of the signed codes the epilogue needs to take the offset back
// out (|sum| <= 64 * 127, so int16). K % 64 == 0.
// Buffer 5 is the same sum pre-scaled, u = scale * sum, in the element
// type: the operand of the zero-point matmul (see gemm_u8q_impl).
//   0:x[M,K] 1:q[M,K] u8 2:scales[M,K/64] 3:xsum[M,K/64] i16 4:K
//   5:u[M,K/64]   dispatch (threads): {256, M, 1}, tg {256, 1, 1}
kernel void quant_f16_u8_row_g64(
    const device VPIPE_ELT* x      [[buffer(0)]],
    device uchar*           q      [[buffer(1)]],
    device VPIPE_ELT*       scales [[buffer(2)]],
    device short*           xsum   [[buffer(3)]],
    const constant int& K     [[buffer(4)]],
    device VPIPE_ELT*       u      [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
  const int row = (int)tgid.y;
  const int G = K / 64;
  for (int g = (int)sgid; g < G; g += 8) {
    const int64_t at = (int64_t)row * K + g * 64 + (int)lane * 2;
    const vec<VPIPE_ELT, 2> h =
        *reinterpret_cast<const device vec<VPIPE_ELT, 2>*>(x + at);
    float am = max(fabs((float)h.x), fabs((float)h.y));
    am = simd_max(am);
    const float inv = am > 0.0f ? 127.0f / am : 0.0f;
    const int qx = clamp((int)rint((float)h.x * inv), -127, 127);
    const int qy = clamp((int)rint((float)h.y * inv), -127, 127);
    *reinterpret_cast<device uchar2*>(q + at) =
        uchar2((uchar)(qx + 128), (uchar)(qy + 128));
    const int sum = simd_sum(qx + qy);
    if (lane == 0) {
      const VPIPE_ELT sc = (VPIPE_ELT)(am * (1.0f / 127.0f));
      scales[(int64_t)row * G + g] = sc;
      xsum[(int64_t)row * G + g] = (short)sum;
      u[(int64_t)row * G + g] = (VPIPE_ELT)((float)sc * (float)sum);
    }
  }
}

// THE SIGNED-CODE LAYOUT of an affine weight: q ^ 0x80 per byte (w8) or
// q ^ 0x8 per nibble (w4), which two's complement reads as q - 128 /
// q - 8, and the zero point moved to match, b' = b + 128 s / 8 s. Both
// halves are exact in real arithmetic and the codes are exact in fact;
// only b' is rounded, once, to the element type. The int8 GEMM over this
// layout (gemm_i8s_*) runs on the signed matrix pipe with no -128*Q
// term, which is what makes a w8 pack at least as fast as bf16 under
// i8_gemm (see I8GemmContext::gemm_signed).
//
// One thread per (group, channel): its 64 codes (16 words of w8, 8 of
// w4) and its zero point. `codes_out` may BE `codes_in` and `zb` may be
// `biases` -- each thread reads its own words and entry before writing
// them -- so a caller-owned buffer converts in place.
//   0:codes_in(u32) 1:scales 2:biases 3:codes_out(u32) 4:zb 5:K 6:N
//   grid (threads) {K/64, N, 1}
template <int BITS>
static inline void affine_to_signed_impl(
    const device uint32_t* cin, const device VPIPE_ELT* scales,
    const device VPIPE_ELT* biases, device uint32_t* cout,
    device VPIPE_ELT* zb, int K, int N, uint2 gid)
{
  const int g = (int)gid.x, n = (int)gid.y;
  const int G = K / 64;
  if (n >= N || g >= G) { return; }
  constexpr int W = BITS == 8 ? 16 : 8;            // words per group
  constexpr uint32_t FLIP = BITS == 8 ? 0x80808080u : 0x88888888u;
  constexpr float OFF = BITS == 8 ? 128.0f : 8.0f;
  const int64_t at = (int64_t)n * (K / (32 / BITS)) + g * W;
  for (int i = 0; i < W; ++i) { cout[at + i] = cin[at + i] ^ FLIP; }
  const int64_t gi = (int64_t)n * G + g;
  zb[gi] = (VPIPE_ELT)((float)biases[gi] + OFF * (float)scales[gi]);
}

kernel void affine_to_signed_w8g64(
    const device uint32_t*  cin    [[buffer(0)]],
    const device VPIPE_ELT* scales [[buffer(1)]],
    const device VPIPE_ELT* biases [[buffer(2)]],
    device uint32_t*        cout   [[buffer(3)]],
    device VPIPE_ELT*       zb     [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
  affine_to_signed_impl<8>(cin, scales, biases, cout, zb, K, N, gid);
}

kernel void affine_to_signed_w4g64(
    const device uint32_t*  cin    [[buffer(0)]],
    const device VPIPE_ELT* scales [[buffer(1)]],
    const device VPIPE_ELT* biases [[buffer(2)]],
    device uint32_t*        cout   [[buffer(3)]],
    device VPIPE_ELT*       zb     [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
  affine_to_signed_impl<4>(cin, scales, biases, cout, zb, K, N, gid);
}

// Dequant of the signed layout: w = s * code + b', code read as int8
// (w8) or as a sign-extended nibble (w4). Same contract and grid as
// affine_dequant_w8g64 / _w4g64, with buffer 2 the zero point b'.
kernel void affine_dequant_w8g64s(
    const device uint32_t*  w      [[buffer(0)]],
    const device VPIPE_ELT* scales [[buffer(1)]],
    const device VPIPE_ELT* zb     [[buffer(2)]],
    device VPIPE_ELT*       ydeq   [[buffer(3)]],
    const constant int& K [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int word = (int)gid.x;
  const int n    = (int)gid.y;
  if (n >= N || word >= (K >> 2)) { return; }
  const int k0 = word << 2;
  const int g = n * (K / DQ_GROUP) + (k0 / DQ_GROUP);
  const float s = (float)scales[g];
  const float b = (float)zb[g];
  const uint32_t packed = w[(int64_t)n * (K >> 2) + word];
  VPIPE_ELT out[4];
  for (int i = 0; i < 4; ++i) {
    const int q = (int)(char)((packed >> (8 * i)) & 0xffu);
    out[i] = (VPIPE_ELT)(s * (float)q + b);
  }
  *reinterpret_cast<device vec<VPIPE_ELT, 4>*>(ydeq + (int64_t)n * K + k0) =
      vec<VPIPE_ELT, 4>(out[0], out[1], out[2], out[3]);
}

kernel void affine_dequant_w4g64s(
    const device uint32_t*  w      [[buffer(0)]],
    const device VPIPE_ELT* scales [[buffer(1)]],
    const device VPIPE_ELT* zb     [[buffer(2)]],
    device VPIPE_ELT*       ydeq   [[buffer(3)]],
    const constant int& K [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int word = (int)gid.x;
  const int n    = (int)gid.y;
  if (n >= N || word >= (K >> 3)) { return; }
  const int k0 = word << 3;
  const int g = n * (K / DQ_GROUP) + (k0 / DQ_GROUP);
  const float s = (float)scales[g];
  const float b = (float)zb[g];
  const uint32_t packed = w[(int64_t)n * (K >> 3) + word];
  VPIPE_ELT out[8];
  for (int i = 0; i < 8; ++i) {
    const int q = (int)((packed >> (4 * i)) & 0xfu);
    out[i] = (VPIPE_ELT)(s * (float)(q >= 8 ? q - 16 : q) + b);
  }
  device VPIPE_ELT* dst = ydeq + (int64_t)n * K + k0;
  *reinterpret_cast<device vec<VPIPE_ELT, 4>*>(dst) =
      vec<VPIPE_ELT, 4>(out[0], out[1], out[2], out[3]);
  *reinterpret_cast<device vec<VPIPE_ELT, 4>*>(dst + 4) =
      vec<VPIPE_ELT, 4>(out[4], out[5], out[6], out[7]);
}

// SIGNED twin of quant_f16_u8_row_g64, for the signed-code GEMM
// (gemm_i8q_*): the symmetric int8 code as it is, its per-(row, group)
// scale, and u = scale * sum(codes) in the element type -- the operand
// of that GEMM's zero-point matmul. K % 64 == 0.
//   0:x[M,K] 1:q[M,K] i8 2:scales[M,K/64] 3:u[M,K/64] 4:K
//   dispatch (threads): {256, M, 1}, tg {256, 1, 1}
kernel void quant_f16_i8s_row_g64(
    const device VPIPE_ELT* x      [[buffer(0)]],
    device char*            q      [[buffer(1)]],
    device VPIPE_ELT*       scales [[buffer(2)]],
    device VPIPE_ELT*       u      [[buffer(3)]],
    const constant int& K     [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
  const int row = (int)tgid.y;
  const int G = K / 64;
  for (int g = (int)sgid; g < G; g += 8) {
    const int64_t at = (int64_t)row * K + g * 64 + (int)lane * 2;
    const vec<VPIPE_ELT, 2> h =
        *reinterpret_cast<const device vec<VPIPE_ELT, 2>*>(x + at);
    float am = max(fabs((float)h.x), fabs((float)h.y));
    am = simd_max(am);
    const float inv = am > 0.0f ? 127.0f / am : 0.0f;
    const int qx = clamp((int)rint((float)h.x * inv), -127, 127);
    const int qy = clamp((int)rint((float)h.y * inv), -127, 127);
    *reinterpret_cast<device char2*>(q + at) = char2((char)qx, (char)qy);
    const int sum = simd_sum(qx + qy);
    if (lane == 0) {
      const VPIPE_ELT sc = (VPIPE_ELT)(am * (1.0f / 127.0f));
      scales[(int64_t)row * G + g] = sc;
      u[(int64_t)row * G + g] = (VPIPE_ELT)((float)sc * (float)sum);
    }
  }
}

// quant_f16_i8s_row_g64 over silu(gate) * up, never storing the product:
// the down projection's input is quantized straight from the two halves
// the up projection wrote. The product is rounded to the element type
// first, exactly as swiglu_v4_f16 stores it, so the codes, scales and u
// are the ones that kernel followed by quant_f16_i8s_row_g64 produce.
//   0:gate[M,K] 1:up[M,K] 2:q[M,K] i8 3:scales[M,K/64] 4:u[M,K/64] 5:K
//   dispatch (threads): {256, M, 1}, tg {256, 1, 1}
kernel void swiglu_quant_f16_i8s_row_g64(
    const device VPIPE_ELT* gate   [[buffer(0)]],
    const device VPIPE_ELT* up     [[buffer(1)]],
    device char*            q      [[buffer(2)]],
    device VPIPE_ELT*       scales [[buffer(3)]],
    device VPIPE_ELT*       u      [[buffer(4)]],
    const constant int& K     [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
  const int row = (int)tgid.y;
  const int G = K / 64;
  for (int g = (int)sgid; g < G; g += 8) {
    const int64_t at = (int64_t)row * K + g * 64 + (int)lane * 2;
    const vec<VPIPE_ELT, 2> gv =
        *reinterpret_cast<const device vec<VPIPE_ELT, 2>*>(gate + at);
    const vec<VPIPE_ELT, 2> uv =
        *reinterpret_cast<const device vec<VPIPE_ELT, 2>*>(up + at);
    const float2 gf = float2(gv), uf2 = float2(uv);
    const float2 sg = gf / (1.0f + exp(-gf));
    const vec<VPIPE_ELT, 2> h = vec<VPIPE_ELT, 2>(sg * uf2);
    float am = max(fabs((float)h.x), fabs((float)h.y));
    am = simd_max(am);
    const float inv = am > 0.0f ? 127.0f / am : 0.0f;
    const int qx = clamp((int)rint((float)h.x * inv), -127, 127);
    const int qy = clamp((int)rint((float)h.y * inv), -127, 127);
    *reinterpret_cast<device char2*>(q + at) = char2((char)qx, (char)qy);
    const int sum = simd_sum(qx + qy);
    if (lane == 0) {
      const VPIPE_ELT sc = (VPIPE_ELT)(am * (1.0f / 127.0f));
      scales[(int64_t)row * G + g] = sc;
      u[(int64_t)row * G + g] = (VPIPE_ELT)((float)sc * (float)sum);
    }
  }
}

// THE COARSE TIER's quantizers (gemm_i8c_* in dense_gemm_mma): symmetric
// int8 per (row, group of GC) with an amax scale in the element type, one
// simdgroup per group, each lane GC/32 CONTIGUOUS elements. Rows are
// activation rows or weight channels alike -- a bf16 weight takes the
// same pass once, at load. K % GC == 0.
//   quant_i8_row_gc*:        0:x[M,K] 1:q[M,K] i8 2:scales[M,K/GC] 3:K
//   swiglu_quant_i8_row_gc*: 0:gate 1:up 2:q 3:scales 4:K -- the product
//     rounded to the element type first, as swiglu_v4_f16 stores it
//   affine_to_i8c_w8_gc*:    0:codes[N,K] u8 (g64 affine) 1:scales
//     2:biases [N,K/64] 3:q[N,K] i8 (may be 0) 4:cscales[N,K/GC] 5:K --
//     each element dequantized, then requantized at GC
//   affine_to_i8c_w4_gc*:    the same from a w4 g64 pack (codes [N,K/2],
//     byte b holding k = 2b low and 2b + 1 high); q is twice the codes'
//     size, so it is a buffer of its own
//   i8c_dequant_gc*:         0:q[N,K] i8 1:cscales 2:out[N,K] 3:K
//   all: ONE SIMDGROUP PER (row, group), four groups a threadgroup:
//   dispatch (threads) {ceil(K/GC / 4) * 128, rows, 1}, tg {128, 1, 1}.
//   (A simdgroup walking its row's groups in turn left too few in
//   flight to hide the loads once the rotation's registers cut
//   occupancy.)
// THE ROTATION (ROT): an orthonormal Walsh-Hadamard transform of each
// group, applied to BOTH operands, so the product is unchanged
// (x W^T = (x H)(W H)^T, H H^T = I) while an outlier channel's energy is
// spread over its whole group instead of setting the group's scale --
// QuaRot's trick, block-diagonal at GC. Sylvester order, so H is
// symmetric and its own inverse: the dequant rotates back with the same
// transform. A lane holds GC/32 contiguous elements: the strides inside
// a lane are register butterflies, the five across lanes simd shuffles.
// The group's values are passed by ARRAY REFERENCE and the helpers are
// forced inline: through a `thread float*` the array stayed in thread
// memory, and every butterfly stage was a stack round trip -- the rotated
// quantizer ran 3x the plain one's time whatever the rotation's size
// (gemm_i8.coarse_tier: blocks of 16, no shuffles at all, cost the same
// as 512). And EVERY INDEX IS A TEMPLATE CONSTANT (the recursions
// below): written as loops -- nested, the inner bounds off the outer
// variable -- `unroll(full)` or not, v[] still went to thread memory.
// MEASURED on the M5 Ultra, a w4 [12288, 4096] into the token layout
// (gemm_i8.fast_conversion_rate): the three in-lane stages 0.16 ms as
// loops, 0.005 written out, the whole conversion 0.06 without them.
template <int PL, int H, int J = 0>
__attribute__((always_inline)) static inline void
i8c_bfly_(thread float (&v)[PL])
{
  if constexpr (J < PL) {
    if constexpr ((J & H) == 0) {
      const float a = v[J], b = v[J + H];
      v[J] = a + b;
      v[J + H] = a - b;
    }
    i8c_bfly_<PL, H, J + 1>(v);
  }
}

template <int PL, int HL, int H = 1>
__attribute__((always_inline)) static inline void
i8c_bfly_stages_(thread float (&v)[PL])
{
  if constexpr (H < HL) {
    i8c_bfly_<PL, H>(v);
    i8c_bfly_stages_<PL, HL, 2 * H>(v);
  }
}

// A cross-lane butterfly is o + sg v with sg = -1 on the high lane: one
// fma, exact in the product, so the same bits as `hi ? o - v : v + o`.
template <int PL, int J = 0>
__attribute__((always_inline)) static inline void
i8c_xbfly_(thread float (&v)[PL], float sg, ushort m)
{
  if constexpr (J < PL) {
    v[J] = fma(sg, v[J], simd_shuffle_xor(v[J], m));
    i8c_xbfly_<PL, J + 1>(v, sg, m);
  }
}

template <int PL, int ML, int M = 1>
__attribute__((always_inline)) static inline void
i8c_xstages_(thread float (&v)[PL], uint lane)
{
  if constexpr (M < ML) {
    i8c_xbfly_<PL>(v, (lane & (uint)M) != 0 ? -1.0f : 1.0f, (ushort)M);
    i8c_xstages_<PL, ML, 2 * M>(v, lane);
  }
}

template <int PL, int J = 0>
__attribute__((always_inline)) static inline void
i8c_scale_(thread float (&v)[PL], float n)
{
  if constexpr (J < PL) {
    v[J] *= n;
    i8c_scale_<PL, J + 1>(v, n);
  }
}

template <int GC, int RB = GC, bool NORM = true>
__attribute__((always_inline)) static inline void
i8c_fwht(thread float (&v)[GC / 32], uint lane)
{
  constexpr int PL = GC / 32;
  // RB < GC rotates in independent blocks of RB (block-diagonal): the
  // strides at and past RB are skipped -- in-lane ones past PL, the
  // cross-lane ones past RB / PL lanes.
  constexpr int HL = RB < PL ? RB : PL;
  constexpr int ML = RB > PL ? RB / PL : 1;
  i8c_bfly_stages_<PL, HL>(v);
  i8c_xstages_<PL, ML>(v, lane);
  if constexpr (NORM) { i8c_scale_<PL>(v, rsqrt((float)RB)); }
}

template <int GC, bool ROT = false, int RB = GC>
__attribute__((always_inline)) static inline void
i8c_quant_group(thread float (&v)[GC / 32], device char* qp,
                device VPIPE_ELT* sp, uint lane)
{
  constexpr int PL = GC / 32;
  if constexpr (ROT) { i8c_fwht<GC, RB>(v, lane); }
  float am = 0.0f;
#pragma clang loop unroll(full)
  for (int i = 0; i < PL; ++i) { am = max(am, fabs(v[i])); }
  am = simd_max(am);
  const float inv = am > 0.0f ? 127.0f / am : 0.0f;
#pragma clang loop unroll(full)
  for (int i = 0; i < PL; i += 4) {
    const int4 r = clamp(int4(rint(float4(v[i], v[i + 1], v[i + 2],
                                          v[i + 3]) * inv)),
                         int4(-127), int4(127));
    *reinterpret_cast<device char4*>(qp + i) = char4(r);
  }
  if (lane == 0) { *sp = (VPIPE_ELT)(am * (1.0f / 127.0f)); }
}

template <int GC, bool ROT = false, int RB = GC>
static inline void quant_i8_row_gc_impl(
    const device VPIPE_ELT* x, device char* q, device VPIPE_ELT* scales,
    int K, uint row, uint gx, uint sgid,
    uint lane)
{
  constexpr int PL = GC / 32;
  const int G = K / GC;
  {
    const int g = (int)gx * 4 + (int)sgid;
    if (g >= G) { return; }
    const int64_t at = (int64_t)row * K + g * GC + (int)lane * PL;
    float v[PL];
#pragma clang loop unroll(full)
    for (int i = 0; i < PL; i += 4) {
      const vec<VPIPE_ELT, 4> h =
          *reinterpret_cast<const device vec<VPIPE_ELT, 4>*>(x + at + i);
      v[i] = (float)h.x; v[i + 1] = (float)h.y;
      v[i + 2] = (float)h.z; v[i + 3] = (float)h.w;
    }
    i8c_quant_group<GC, ROT, RB>(v, q + at, scales + (int64_t)row * G + g,
                                  lane);
  }
}

template <int GC, bool ROT = false>
static inline void swiglu_quant_i8_row_gc_impl(
    const device VPIPE_ELT* gate, const device VPIPE_ELT* up, device char* q,
    device VPIPE_ELT* scales, int K, uint row, uint gx, uint sgid,
    uint lane)
{
  constexpr int PL = GC / 32;
  const int G = K / GC;
  {
    const int g = (int)gx * 4 + (int)sgid;
    if (g >= G) { return; }
    const int64_t at = (int64_t)row * K + g * GC + (int)lane * PL;
    float v[PL];
#pragma clang loop unroll(full)
    for (int i = 0; i < PL; i += 4) {
      const float4 gv = float4(
          *reinterpret_cast<const device vec<VPIPE_ELT, 4>*>(gate + at + i));
      const float4 uv = float4(
          *reinterpret_cast<const device vec<VPIPE_ELT, 4>*>(up + at + i));
      const vec<VPIPE_ELT, 4> h =
          vec<VPIPE_ELT, 4>(gv / (1.0f + exp(-gv)) * uv);
      v[i] = (float)h.x; v[i + 1] = (float)h.y;
      v[i + 2] = (float)h.z; v[i + 3] = (float)h.w;
    }
    i8c_quant_group<GC, ROT>(v, q + at, scales + (int64_t)row * G + g,
                              lane);
  }
}

// An affine pack's code for element `at` of its [N, K] matrix: a byte at
// w8, a nibble at w4 (byte at / 2, low nibble for an even at).
template <int BITS>
__attribute__((always_inline)) static inline uint
affine_code(const device uchar* codes, int64_t at)
{
  if constexpr (BITS == 8) {
    return codes[at];
  } else {
    const uint c = codes[at >> 1];
    return (at & 1) != 0 ? (c >> 4) : (c & 0xfu);
  }
}

// Four consecutive codes of an affine pack from element `at` (a multiple
// of 4): a uchar4 at w8, two bytes of nibbles at w4.
template <int BITS>
__attribute__((always_inline)) static inline float4
affine_codes4(const device uchar* codes, int64_t at)
{
  if constexpr (BITS == 8) {
    return float4(*reinterpret_cast<const device uchar4*>(codes + at));
  } else {
    const uint p =
        *reinterpret_cast<const device ushort*>(codes + (at >> 1));
    return float4(p & 0xfu, (p >> 4) & 0xfu, (p >> 8) & 0xfu, p >> 12);
  }
}

template <int GC, bool ROT = false, int BITS = 8>
static inline void affine_to_i8c_impl(
    const device uchar* codes, const device VPIPE_ELT* scales,
    const device VPIPE_ELT* biases, device char* q,
    device VPIPE_ELT* cscales, int K, uint row, uint gx, uint sgid,
    uint lane)
{
  constexpr int PL = GC / 32;
  const int G = K / GC, G64 = K / 64;
  {
    const int g = (int)gx * 4 + (int)sgid;
    if (g >= G) { return; }
    const int k0 = g * GC + (int)lane * PL;
    const int64_t at = (int64_t)row * K + k0;
    // A lane's PL elements are contiguous and inside ONE g64 group (PL
    // divides 64): one scale, one bias, and the codes as whole words --
    // what a per-call conversion (encode_to_fast_scratch) runs at.
    const int64_t sg = (int64_t)row * G64 + k0 / 64;
    const float s = (float)scales[sg], b = (float)biases[sg];
    // Four at a time (affine_codes4): a per-element shift and mask over
    // whole words ran at half the rate (gemm_i8.fast_conversion_rate).
    float v[PL];
#pragma clang loop unroll(full)
    for (int i = 0; i < PL; i += 4) {
      const float4 a = affine_codes4<BITS>(codes, at + i) * s + b;
      v[i] = a.x; v[i + 1] = a.y; v[i + 2] = a.z; v[i + 3] = a.w;
    }
    // Every lane has read its elements before any writes them, and no
    // lane touches another's: at w8, in place (q == codes) is safe.
    simdgroup_barrier(mem_flags::mem_none);
    i8c_quant_group<GC, ROT>(v, q + at, cscales + (int64_t)row * G + g,
                             lane);
  }
}

template <int GC, bool ROT = false>
static inline void i8c_dequant_impl(
    const device char* q, const device VPIPE_ELT* cscales,
    device VPIPE_ELT* out, int K, uint row, uint gx, uint sgid,
    uint lane)
{
  constexpr int PL = GC / 32;
  const int G = K / GC;
  {
    const int g = (int)gx * 4 + (int)sgid;
    if (g >= G) { return; }
    const int64_t at = (int64_t)row * K + g * GC + (int)lane * PL;
    const float s = (float)cscales[(int64_t)row * G + g];
    float v[PL];
#pragma clang loop unroll(full)
    for (int i = 0; i < PL; ++i) { v[i] = (float)q[at + i] * s; }
    if constexpr (ROT) { i8c_fwht<GC>(v, lane); }
#pragma clang loop unroll(full)
    for (int i = 0; i < PL; ++i) { out[at + i] = (VPIPE_ELT)v[i]; }
  }
}

#define I8C_QUANT_ENTRIES(GC, ROT, SFX)                                        \
  kernel void quant_i8_row_gc##GC##SFX(                                        \
      const device VPIPE_ELT* x [[buffer(0)]],                                 \
      device char* q [[buffer(1)]],                                            \
      device VPIPE_ELT* scales [[buffer(2)]],                                  \
      const constant int& K [[buffer(3)]],                                     \
      uint2 tgid [[threadgroup_position_in_grid]],                             \
      uint sgid [[simdgroup_index_in_threadgroup]],                            \
      uint lane [[thread_index_in_simdgroup]]) {                               \
    quant_i8_row_gc_impl<GC, ROT>(x, q, scales, K, tgid.y, tgid.x, sgid,     \
                                  lane);                                       \
  }                                                                            \
  kernel void swiglu_quant_i8_row_gc##GC##SFX(                                 \
      const device VPIPE_ELT* gate [[buffer(0)]],                              \
      const device VPIPE_ELT* up [[buffer(1)]],                                \
      device char* q [[buffer(2)]],                                            \
      device VPIPE_ELT* scales [[buffer(3)]],                                  \
      const constant int& K [[buffer(4)]],                                     \
      uint2 tgid [[threadgroup_position_in_grid]],                             \
      uint sgid [[simdgroup_index_in_threadgroup]],                            \
      uint lane [[thread_index_in_simdgroup]]) {                               \
    swiglu_quant_i8_row_gc_impl<GC, ROT>(gate, up, q, scales, K, tgid.y,       \
                                    tgid.x, sgid, lane);                       \
  }                                                                            \
  kernel void affine_to_i8c_w8_gc##GC##SFX(                                    \
      const device uchar* codes [[buffer(0)]],                                 \
      const device VPIPE_ELT* scales [[buffer(1)]],                            \
      const device VPIPE_ELT* biases [[buffer(2)]],                            \
      device char* q [[buffer(3)]],                                            \
      device VPIPE_ELT* cscales [[buffer(4)]],                                 \
      const constant int& K [[buffer(5)]],                                     \
      uint2 tgid [[threadgroup_position_in_grid]],                             \
      uint sgid [[simdgroup_index_in_threadgroup]],                            \
      uint lane [[thread_index_in_simdgroup]]) {                               \
    affine_to_i8c_impl<GC, ROT, 8>(codes, scales, biases, q, cscales, K,       \
                                   tgid.y, tgid.x, sgid, lane);                \
  }                                                                            \
  kernel void affine_to_i8c_w4_gc##GC##SFX(                                    \
      const device uchar* codes [[buffer(0)]],                                 \
      const device VPIPE_ELT* scales [[buffer(1)]],                            \
      const device VPIPE_ELT* biases [[buffer(2)]],                            \
      device char* q [[buffer(3)]],                                            \
      device VPIPE_ELT* cscales [[buffer(4)]],                                 \
      const constant int& K [[buffer(5)]],                                     \
      uint2 tgid [[threadgroup_position_in_grid]],                             \
      uint sgid [[simdgroup_index_in_threadgroup]],                            \
      uint lane [[thread_index_in_simdgroup]]) {                               \
    affine_to_i8c_impl<GC, ROT, 4>(codes, scales, biases, q, cscales, K,       \
                                   tgid.y, tgid.x, sgid, lane);                \
  }                                                                            \
  kernel void i8c_dequant_gc##GC##SFX(                                         \
      const device char* q [[buffer(0)]],                                      \
      const device VPIPE_ELT* cscales [[buffer(1)]],                           \
      device VPIPE_ELT* out [[buffer(2)]],                                     \
      const constant int& K [[buffer(3)]],                                     \
      uint2 tgid [[threadgroup_position_in_grid]],                             \
      uint sgid [[simdgroup_index_in_threadgroup]],                            \
      uint lane [[thread_index_in_simdgroup]]) {                               \
    i8c_dequant_impl<GC, ROT>(q, cscales, out, K, tgid.y, tgid.x, sgid, lane); \
  }
I8C_QUANT_ENTRIES(256, false, )
I8C_QUANT_ENTRIES(512, false, )
I8C_QUANT_ENTRIES(1024, false, )
// ...and ROTATED (suffix _r): the group Walsh-Hadamard before the
// quantizer, and after the dequant.
I8C_QUANT_ENTRIES(256, true, _r)
I8C_QUANT_ENTRIES(512, true, _r)
I8C_QUANT_ENTRIES(1024, true, _r)
// THE ROTATION BLOCK, for its A/B (gemm_i8.coarse_tier): the g512
// activation quantizer rotating in blocks of RB < 512.
#define I8C_QUANT_RB(RB)                                                       \
  kernel void quant_i8_row_gc512_rb##RB(                                       \
      const device VPIPE_ELT* x [[buffer(0)]],                                 \
      device char* q [[buffer(1)]],                                            \
      device VPIPE_ELT* scales [[buffer(2)]],                                  \
      const constant int& K [[buffer(3)]],                                     \
      uint2 tgid [[threadgroup_position_in_grid]],                             \
      uint sgid [[simdgroup_index_in_threadgroup]],                            \
      uint lane [[thread_index_in_simdgroup]]) {                               \
    quant_i8_row_gc_impl<512, true, RB>(x, q, scales, K, tgid.y, tgid.x,       \
                                        sgid, lane);                           \
  }
I8C_QUANT_RB(16)
I8C_QUANT_RB(64)
I8C_QUANT_RB(128)
I8C_QUANT_RB(256)

// THE ACCURATE TIER ROTATED (suffix _r; I8GemmContext::set_g64_rotation):
// quant_f16_i8_row_g64 / _pad with each group of 64 Walsh-Hadamard
// transformed before its amax, on the activation and the weight alike.
// The GEMM (gemm_i8i8_sc_f16_n64_g64rs) is the plain tier's: a group's
// dot product is the same in the rotated basis. Same contract and
// dispatch as the plain pair -- {256, M, 1} threads, eight simdgroups a
// row, one a group -- and a lane holds two contiguous elements, so the
// stride-1 butterfly runs in the lane and the other five are shuffles.
// A padded group's zeros are rotated in as the other operand's are.
template <bool PAD>
static inline void quant_row_g64r_impl(
    const device VPIPE_ELT* x, device char* q, device VPIPE_ELT* scales,
    int Ksrc, int Kpad, uint2 tgid, uint sgid, uint lane)
{
  const int row = (int)tgid.y;
  const int G = Kpad / 64;
  for (int g = (int)sgid; g < G; g += 8) {
    const int col = g * 64 + (int)lane * 2;
    const int64_t src = (int64_t)row * Ksrc + col;
    const int64_t dst = (int64_t)row * Kpad + col;
    float h[2];
    if (!PAD || col + 1 < Ksrc) {
      const vec<VPIPE_ELT, 2> v =
          *reinterpret_cast<const device vec<VPIPE_ELT, 2>*>(x + src);
      h[0] = (float)v[0];
      h[1] = (float)v[1];
    } else {
      h[0] = col < Ksrc ? (float)x[src] : 0.0f;
      h[1] = col + 1 < Ksrc ? (float)x[src + 1] : 0.0f;
    }
    i8c_fwht<64>(h, lane);
    const float am = simd_max(max(fabs(h[0]), fabs(h[1])));
    const float inv = am > 0.0f ? 127.0f / am : 0.0f;
    vec<char, 2> o;
    o[0] = (char)clamp((int)rint(h[0] * inv), -127, 127);
    o[1] = (char)clamp((int)rint(h[1] * inv), -127, 127);
    *reinterpret_cast<device vec<char, 2>*>(q + dst) = o;
    if (lane == 0) {
      scales[(int64_t)row * G + g] = (VPIPE_ELT)(am * (1.0f / 127.0f));
    }
  }
}

kernel void quant_f16_i8_row_g64_r(
    const device VPIPE_ELT* x [[buffer(0)]],
    device char* q [[buffer(1)]],
    device VPIPE_ELT* scales [[buffer(2)]],
    const constant int& K [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
  quant_row_g64r_impl<false>(x, q, scales, K, K, tgid, sgid, lane);
}

kernel void quant_f16_i8_row_g64_r_pad(
    const device VPIPE_ELT* x [[buffer(0)]],
    device char* q [[buffer(1)]],
    device VPIPE_ELT* scales [[buffer(2)]],
    const constant int& Ksrc [[buffer(3)]],
    const constant int& Kpad [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
  quant_row_g64r_impl<true>(x, q, scales, Ksrc, Kpad, tgid, sgid, lane);
}

// THE PER-TOKEN TIER's quantizers (gemm_i8pt_* in dense_gemm_mma): one
// symmetric int8 scale per ROW over all of K, stored f32 -- the i8x
// layout. One threadgroup a row of T = K/16 threads (rounded up to a
// simdgroup), each holding 16 of the row's values in registers: read
// once, amax, quantize from registers. K % 4 == 0, K <= 16384. Sizing
// the threadgroup to the row rather than the registers to the largest
// row is what keeps the occupancy up: a fixed 256-thread group holding
// 64 values a thread ran 1.6x the rotated g512 quantizer's time.
// Rows are activation tokens or weight channels alike.
//   quant_i8_tok:        0:x[M,K] 1:q[M,K] i8 2:scales[M] f32 3:K
//   swiglu_quant_i8_tok: 0:gate[M,K] 1:up[M,K] 2:q 3:scales 4:K -- the
//     product rounded to the element type first, as swiglu_v4_f16
//     stores it
//   affine_to_i8pt_w8:   0:codes[N,K] u8 (g64 affine) 1:scales 2:biases
//     [N,K/64] 3:q[N,K] i8 4:cscales[N] f32 5:K -- dequantized, then
//     requantized per channel; q may alias codes
//   affine_to_i8pt_w4:   the same from a w4 g64 pack (codes [N,K/2]); q
//     is a buffer of its own
//   i8pt_dequant:        0:q[N,K] i8 1:cscales[N] f32 2:w[N,K] 3:K
//   dispatch (threads): {T, rows, 1}, tg {T, 1, 1} (i8pt_dequant: 256)
constant constexpr int kI8ptNV = 4;  // float4s a thread

static inline float i8pt_row_amax(float am, uint sgid, uint lane,
                                  uint nsg, threadgroup float* sm)
{
  am = simd_max(am);
  if (lane == 0) { sm[sgid] = am; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  am = lane < nsg ? sm[lane] : 0.0f;
  return simd_max(am);
}

static inline void i8pt_row_store(thread const float4 (&v)[kI8ptNV],
                                  float am, device char* q, int64_t base,
                                  int K, uint lid, uint T,
                                  device float* scale)
{
  const float inv = am > 0.0f ? 127.0f / am : 0.0f;
#pragma clang loop unroll(full)
  for (int j = 0; j < kI8ptNV; ++j) {
    const int k = (j * (int)T + (int)lid) * 4;
    if (k < K) {
      const int4 r = clamp(int4(rint(v[j] * inv)), int4(-127), int4(127));
      *reinterpret_cast<device char4*>(q + base + k) = char4(r);
    }
  }
  if (lid == 0) { *scale = am * (1.0f / 127.0f); }
}

#define I8PT_ARGS                                                              \
  uint2 tgid [[threadgroup_position_in_grid]],                                 \
      uint lid [[thread_index_in_threadgroup]],                                \
      uint2 T2 [[threads_per_threadgroup]],                                    \
      uint sgid [[simdgroup_index_in_threadgroup]],                            \
      uint lane [[thread_index_in_simdgroup]]

kernel void quant_i8_tok(
    const device VPIPE_ELT* x [[buffer(0)]],
    device char* q [[buffer(1)]],
    device float* scales [[buffer(2)]],
    const constant int& K [[buffer(3)]],
    I8PT_ARGS)
{
  const uint T = T2.x;
  threadgroup float sm[32];
  const int64_t base = (int64_t)tgid.y * K;
  float4 v[kI8ptNV];
  float am = 0.0f;
#pragma clang loop unroll(full)
  for (int j = 0; j < kI8ptNV; ++j) {
    const int k = (j * (int)T + (int)lid) * 4;
    v[j] = float4(0.0f);
    if (k < K) {
      v[j] = float4(
          *reinterpret_cast<const device vec<VPIPE_ELT, 4>*>(x + base + k));
      const float4 a = fabs(v[j]);
      am = max(am, max(max(a.x, a.y), max(a.z, a.w)));
    }
  }
  am = i8pt_row_amax(am, sgid, lane, (T + 31) / 32, sm);
  i8pt_row_store(v, am, q, base, K, lid, T, scales + tgid.y);
}

kernel void swiglu_quant_i8_tok(
    const device VPIPE_ELT* gate [[buffer(0)]],
    const device VPIPE_ELT* up [[buffer(1)]],
    device char* q [[buffer(2)]],
    device float* scales [[buffer(3)]],
    const constant int& K [[buffer(4)]],
    I8PT_ARGS)
{
  const uint T = T2.x;
  threadgroup float sm[32];
  const int64_t base = (int64_t)tgid.y * K;
  float4 v[kI8ptNV];
  float am = 0.0f;
#pragma clang loop unroll(full)
  for (int j = 0; j < kI8ptNV; ++j) {
    const int k = (j * (int)T + (int)lid) * 4;
    v[j] = float4(0.0f);
    if (k < K) {
      const float4 g = float4(
          *reinterpret_cast<const device vec<VPIPE_ELT, 4>*>(gate + base + k));
      const float4 u = float4(
          *reinterpret_cast<const device vec<VPIPE_ELT, 4>*>(up + base + k));
      v[j] = float4(vec<VPIPE_ELT, 4>(g / (1.0f + exp(-g)) * u));
      const float4 a = fabs(v[j]);
      am = max(am, max(max(a.x, a.y), max(a.z, a.w)));
    }
  }
  am = i8pt_row_amax(am, sgid, lane, (T + 31) / 32, sm);
  i8pt_row_store(v, am, q, base, K, lid, T, scales + tgid.y);
}


template <int BITS>
static inline void affine_to_i8pt_impl(
    const device uchar* codes, const device VPIPE_ELT* scales,
    const device VPIPE_ELT* biases, device char* q, device float* cscales,
    int K, uint row, uint lid, uint T, uint sgid, uint lane,
    threadgroup float* sm)
{
  const int64_t base = (int64_t)row * K;
  const int64_t gbase = (int64_t)row * (K / 64);
  float4 v[kI8ptNV];
  float am = 0.0f;
#pragma clang loop unroll(full)
  for (int j = 0; j < kI8ptNV; ++j) {
    const int k = (j * (int)T + (int)lid) * 4;
    v[j] = float4(0.0f);
    if (k < K) {
      const float4 c = affine_codes4<BITS>(codes, base + k);
      const float s = (float)scales[gbase + k / 64];
      const float b = (float)biases[gbase + k / 64];
      // The pack's own element-type dequant, as affine_dequant_w8g64.
      v[j] = float4(vec<VPIPE_ELT, 4>(c * s + b));
      const float4 a = fabs(v[j]);
      am = max(am, max(max(a.x, a.y), max(a.z, a.w)));
    }
  }
  am = i8pt_row_amax(am, sgid, lane, (T + 31) / 32, sm);
  // In place (w8) is safe: every read of the row is done by the barrier
  // the reduction took.
  i8pt_row_store(v, am, q, base, K, lid, T, cscales + row);
}

kernel void affine_to_i8pt_w8(
    const device uchar* codes [[buffer(0)]],
    const device VPIPE_ELT* scales [[buffer(1)]],
    const device VPIPE_ELT* biases [[buffer(2)]],
    device char* q [[buffer(3)]],
    device float* cscales [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    I8PT_ARGS)
{
  threadgroup float sm[32];
  affine_to_i8pt_impl<8>(codes, scales, biases, q, cscales, K, tgid.y, lid,
                         T2.x, sgid, lane, sm);
}

kernel void affine_to_i8pt_w4(
    const device uchar* codes [[buffer(0)]],
    const device VPIPE_ELT* scales [[buffer(1)]],
    const device VPIPE_ELT* biases [[buffer(2)]],
    device char* q [[buffer(3)]],
    device float* cscales [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    I8PT_ARGS)
{
  threadgroup float sm[32];
  affine_to_i8pt_impl<4>(codes, scales, biases, q, cscales, K, tgid.y, lid,
                         T2.x, sgid, lane, sm);
}


// THE ROTATED TOKEN QUANTIZERS (_r): the same per-row scale, taken
// after an orthonormal Walsh-Hadamard transform of every 256-wide block
// of K (i8c_fwht<256>), on the activation and on the weight alike, so
// the product is unchanged and an outlier channel is spread over its
// block instead of setting the whole row's scale. The GEMM does not
// change. A lane eight contiguous values of a 256-block, a simdgroup
// one block or two: S simdgroups a row, ceil(K/512) <= S <= 32 (the
// context dispatches ceil(K/512) -- two blocks each ran 1.05-1.18x one,
// gemm_i8.fast_conversion_rate). K % 256 == 0, K <= 16384.
//   quant_i8_tok_r / swiglu_quant_i8_tok_r / affine_to_i8pt_w{8,4}_r:
//     the buffers of their unrotated twins; dispatch {32 S, rows}, tg
//     {32 S}
//   i8pt_dequant_r: 0:q 1:cscales 2:w 3:K -- the transform is its own
//     inverse; the same dispatch
// MODE 0 plain, 1 swiglu, 2 an affine g64 pack at BITS.
template <int MODE, int BITS = 8>
static inline void quant_i8_tok_r_impl(
    const device VPIPE_ELT* x, const device VPIPE_ELT* up,
    const device uchar* codes, const device VPIPE_ELT* ascales,
    const device VPIPE_ELT* abiases, device char* q, device float* scale,
    int K, uint row, uint T, uint sgid, uint lane, threadgroup float* sm)
{
  const int G = K / 256, S = (int)T / 32;
  const int64_t base = (int64_t)row * K;
  float v[2][8];
  float am = 0.0f;
#pragma clang loop unroll(full)
  for (int nb = 0; nb < 2; ++nb) {
    const int b = (int)sgid + nb * S;
    const int64_t at = base + b * 256 + (int)lane * 8;
#pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) { v[nb][i] = 0.0f; }
    if (b < G) {
#pragma clang loop unroll(full)
      for (int i = 0; i < 8; i += 4) {
        float4 a;
        if (MODE == 0) {
          a = float4(*reinterpret_cast<const device vec<VPIPE_ELT, 4>*>(
              x + at + i));
        } else if (MODE == 1) {
          const float4 g = float4(
              *reinterpret_cast<const device vec<VPIPE_ELT, 4>*>(x + at + i));
          const float4 u = float4(
              *reinterpret_cast<const device vec<VPIPE_ELT, 4>*>(up + at + i));
          a = float4(vec<VPIPE_ELT, 4>(g / (1.0f + exp(-g)) * u));
        } else {
          // A lane's eight elements share one g64 group.
          const float4 c = affine_codes4<BITS>(codes, at + i);
          const int64_t gi =
              (int64_t)row * (K / 64) + (b * 256 + (int)lane * 8) / 64;
          a = float4(vec<VPIPE_ELT, 4>(c * (float)ascales[gi] +
                                       (float)abiases[gi]));
        }
        v[nb][i] = a.x; v[nb][i + 1] = a.y;
        v[nb][i + 2] = a.z; v[nb][i + 3] = a.w;
      }
      i8c_fwht<256>(v[nb], lane);
#pragma clang loop unroll(full)
      for (int i = 0; i < 8; ++i) { am = max(am, fabs(v[nb][i])); }
    }
  }
  am = i8pt_row_amax(am, sgid, lane, (T + 31) / 32, sm);
  const float inv = am > 0.0f ? 127.0f / am : 0.0f;
#pragma clang loop unroll(full)
  for (int nb = 0; nb < 2; ++nb) {
    const int b = (int)sgid + nb * S;
    if (b < G) {
      const int64_t at = base + b * 256 + (int)lane * 8;
#pragma clang loop unroll(full)
      for (int i = 0; i < 8; i += 4) {
        const int4 r = clamp(int4(rint(float4(v[nb][i], v[nb][i + 1],
                                              v[nb][i + 2], v[nb][i + 3]) *
                                       inv)),
                             int4(-127), int4(127));
        *reinterpret_cast<device char4*>(q + at + i) = char4(r);
      }
    }
  }
  if (sgid == 0 && lane == 0) { *scale = am * (1.0f / 127.0f); }
}

kernel void quant_i8_tok_r(
    const device VPIPE_ELT* x [[buffer(0)]],
    device char* q [[buffer(1)]],
    device float* scales [[buffer(2)]],
    const constant int& K [[buffer(3)]],
    I8PT_ARGS)
{
  threadgroup float sm[32];
  quant_i8_tok_r_impl<0>(x, x, nullptr, nullptr, nullptr, q,
                         scales + tgid.y, K, tgid.y, T2.x, sgid, lane, sm);
}

kernel void swiglu_quant_i8_tok_r(
    const device VPIPE_ELT* gate [[buffer(0)]],
    const device VPIPE_ELT* up [[buffer(1)]],
    device char* q [[buffer(2)]],
    device float* scales [[buffer(3)]],
    const constant int& K [[buffer(4)]],
    I8PT_ARGS)
{
  threadgroup float sm[32];
  quant_i8_tok_r_impl<1>(gate, up, nullptr, nullptr, nullptr, q,
                         scales + tgid.y, K, tgid.y, T2.x, sgid, lane, sm);
}

kernel void affine_to_i8pt_w8_r(
    const device uchar* codes [[buffer(0)]],
    const device VPIPE_ELT* scales [[buffer(1)]],
    const device VPIPE_ELT* biases [[buffer(2)]],
    device char* q [[buffer(3)]],
    device float* cscales [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    I8PT_ARGS)
{
  threadgroup float sm[32];
  // In place is safe: the row's every read precedes the reduction's
  // barrier.
  quant_i8_tok_r_impl<2>(nullptr, nullptr, codes, scales, biases, q,
                         cscales + tgid.y, K, tgid.y, T2.x, sgid, lane, sm);
}

kernel void affine_to_i8pt_w4_r(
    const device uchar* codes [[buffer(0)]],
    const device VPIPE_ELT* scales [[buffer(1)]],
    const device VPIPE_ELT* biases [[buffer(2)]],
    device char* q [[buffer(3)]],
    device float* cscales [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    I8PT_ARGS)
{
  threadgroup float sm[32];
  quant_i8_tok_r_impl<2, 4>(nullptr, nullptr, codes, scales, biases, q,
                            cscales + tgid.y, K, tgid.y, T2.x, sgid, lane,
                            sm);
}


kernel void i8pt_dequant_r(
    const device char* q [[buffer(0)]],
    const device float* cscales [[buffer(1)]],
    device VPIPE_ELT* w [[buffer(2)]],
    const constant int& K [[buffer(3)]],
    I8PT_ARGS)
{
  const int G = K / 256, S = (int)T2.x / 32;
  const float s = cscales[tgid.y];
#pragma clang loop unroll(full)
  for (int nb = 0; nb < 2; ++nb) {
    const int b = (int)sgid + nb * S;
    if (b < G) {
      const int64_t at = (int64_t)tgid.y * K + b * 256 + (int)lane * 8;
      float v[8];
#pragma clang loop unroll(full)
      for (int i = 0; i < 8; i += 4) {
        const float4 c =
            float4(*reinterpret_cast<const device char4*>(q + at + i)) * s;
        v[i] = c.x; v[i + 1] = c.y; v[i + 2] = c.z; v[i + 3] = c.w;
      }
      i8c_fwht<256>(v, lane);
#pragma clang loop unroll(full)
      for (int i = 0; i < 8; i += 4) {
        *reinterpret_cast<device vec<VPIPE_ELT, 4>*>(w + at + i) =
            vec<VPIPE_ELT, 4>(float4(v[i], v[i + 1], v[i + 2], v[i + 3]));
      }
    }
  }
}
kernel void i8pt_dequant(
    const device char* q [[buffer(0)]],
    const device float* cscales [[buffer(1)]],
    device VPIPE_ELT* w [[buffer(2)]],
    const constant int& K [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
  const int64_t base = (int64_t)tgid.y * K;
  const float s = cscales[tgid.y];
  for (int k = (int)lid * 4; k < K; k += 1024) {
    const char4 c = *reinterpret_cast<const device char4*>(q + base + k);
    *reinterpret_cast<device vec<VPIPE_ELT, 4>*>(w + base + k) =
        vec<VPIPE_ELT, 4>(float4(c) * s);
  }
}

// Per-(channel, 64-group) sums of an affine weight's codes, for the
// native GEMM's offset correction (see gemm_u8q_impl). One thread per
// group: 16 words of 4 bytes (w8) or 8 words of 8 nibbles (w4). Sums
// are <= 64 * 255, so int16. Buffers 4-6 add the scaled form the
// CMODE=1 kernel reads instead.
//   0:w(uint32) 1:wsum[N,K/64] i16 2:K 3:N 4:scales 5:c_hi 6:c_lo
//   grid (threads) {K/64, N, 1}
kernel void affine_group_sums_w8g64(
    const device uint32_t* w    [[buffer(0)]],
    device short*          wsum [[buffer(1)]],
    const constant int& K [[buffer(2)]],
    const constant int& N [[buffer(3)]],
    const device VPIPE_ELT* scales [[buffer(4)]],
    device VPIPE_ELT*       c_hi   [[buffer(5)]],
    device VPIPE_ELT*       c_lo   [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int g = (int)gid.x, n = (int)gid.y;
  const int G = K / 64;
  if (n >= N || g >= G) { return; }
  const device uint32_t* p = w + (int64_t)n * (K >> 2) + g * 16;
  int sum = 0;
  for (int i = 0; i < 16; ++i) {
    const uint32_t v = p[i];
    sum += (int)(v & 0xffu) + (int)((v >> 8) & 0xffu) +
           (int)((v >> 16) & 0xffu) + (int)(v >> 24);
  }
  wsum[(int64_t)n * G + g] = (short)sum;
  // -128 * s * Q as a two-term element-type expansion, for the CMODE=1
  // kernel's post-loop matmul.
  const float c = -128.0f * (float)scales[(int64_t)n * G + g] * (float)sum;
  const VPIPE_ELT hi = (VPIPE_ELT)c;
  c_hi[(int64_t)n * G + g] = hi;
  c_lo[(int64_t)n * G + g] = (VPIPE_ELT)(c - (float)hi);
}

kernel void affine_group_sums_w4g64(
    const device uint32_t* w    [[buffer(0)]],
    device short*          wsum [[buffer(1)]],
    const constant int& K [[buffer(2)]],
    const constant int& N [[buffer(3)]],
    const device VPIPE_ELT* scales [[buffer(4)]],
    device VPIPE_ELT*       c_hi   [[buffer(5)]],
    device VPIPE_ELT*       c_lo   [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int g = (int)gid.x, n = (int)gid.y;
  const int G = K / 64;
  if (n >= N || g >= G) { return; }
  const device uint32_t* p = w + (int64_t)n * (K >> 3) + g * 8;
  int sum = 0;
  for (int i = 0; i < 8; ++i) {
    uint32_t v = p[i];
    for (int j = 0; j < 8; ++j) { sum += (int)(v & 0xfu); v >>= 4; }
  }
  wsum[(int64_t)n * G + g] = (short)sum;
  // -128 * s * Q as a two-term element-type expansion, for the CMODE=1
  // kernel's post-loop matmul.
  const float c = -128.0f * (float)scales[(int64_t)n * G + g] * (float)sum;
  const VPIPE_ELT hi = (VPIPE_ELT)c;
  c_hi[(int64_t)n * G + g] = hi;
  c_lo[(int64_t)n * G + g] = (VPIPE_ELT)(c - (float)hi);
}

// POW2-scale (block-floating-point) twin of quant_f16_i8_row_g512, for
// the SHIFT-ALIGNED integer accumulation GEMM: per-(row, 512-group)
// scale = 2^(Emax - 21) (Emax = the group's max f16 exponent field),
// stored as the raw EXPONENT int8 (Emax - 21, range [-21, 9]) so the
// GEMM can align group partials by integer shifts. The mantissa shift
// runs on the float pipe (exact 2^(21-Emax) multiply + rint -- see
// quant_f16_i8_g64_bfp2). Same contract as quant_f16_i8_row_g512 but
// buffer 2 is int8 exponents [M, K/512].
kernel void quant_f16_i8_row_g512_bfp(
    const device half* x     [[buffer(0)]],
    device char*       q     [[buffer(1)]],
    device char*       eexp  [[buffer(2)]],
    const constant int& K    [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
  const int row = (int)tgid.y;
  const int G = K / 512;
  for (int g = (int)sgid; g < G; g += 8) {
    const int64_t base = (int64_t)row * K + g * 512 + (int)lane * 16;
    const device half* xp = x + base;
    ushort em = 0;
    half2 h[8];
    for (int i = 0; i < 8; ++i) {
      h[i] = *reinterpret_cast<const device half2*>(xp + i * 2);
      const ushort2 u = as_type<ushort2>(h[i]);
      em = max(em, max((ushort)((u.x >> 10) & 0x1F),
                       (ushort)((u.y >> 10) & 0x1F)));
    }
    em = simd_max(em);
    const float inv = as_type<float>((uint)(21 - (int)em + 127) << 23);
    device char* qp = q + base;
    for (int i = 0; i < 8; ++i) {
      char2 o;
      o.x = (char)clamp((int)rint((float)h[i].x * inv), -127, 127);
      o.y = (char)clamp((int)rint((float)h[i].y * inv), -127, 127);
      *reinterpret_cast<device char2*>(qp + i * 2) = o;
    }
    if (lane == 0) { eexp[(int64_t)row * G + g] = (char)((int)em - 21); }
  }
}

// ---- FP8 weights --------------------------------------------------------
//
// A checkpoint stored as FP8 (safetensors F8_E4M3 / F8_E5M2) keeps its
// codes resident and has each weight expanded into a dense VPIPE_ELT
// scratch just before its GEMM -- the same dequant-then-dense shape the
// matrix-core path takes for affine weights, and on a box without matrix
// cores the dense steel tiles then read the scratch. No scales: plain FP8
// is its own value.
//
// E4M3 (OCP E4M3FN: bias 7, subnormals, no infinities). Its seven
// magnitude bits placed at half's bit 7 ARE a half -- same mantissa, the
// exponent field read against bias 15 instead of 7, and the subnormals
// line up because both formats put them at exponent field 0 -- worth
// exactly 2^-8 of the FP8 value. So one shift and one multiply decode
// every finite code EXACTLY (the product is a power-of-two scaling of a
// value with a 3-bit mantissa). The one NaN code, S.1111.111, comes out as
// +-480 rather than NaN; a weight is never NaN, and the host decoder
// (shared/fp8.h) is the reference that says otherwise.
//
// E5M2 is the top byte of a half, so it needs no arithmetic at all.
//
//   0:codes(uchar) 1:y(VPIPE_ELT) 2:n (elements, the whole [N,K])
// Flat, one thread per 4 codes: grid {ceil(n/4), 1, 1}.
inline float
fp8_e4m3_f_(uint c)
{
  const ushort h = (ushort)(((c & 0x80u) << 8) | ((c & 0x7Fu) << 7));
  return (float)as_type<half>(h) * 256.0f;
}

inline float
fp8_e5m2_f_(uint c)
{
  return (float)as_type<half>((ushort)(c << 8));
}

#define VPIPE_FP8_DEQUANT(NAME, DECODE)                                     \
  kernel void NAME(const device uchar* w [[buffer(0)]],                     \
                   device VPIPE_ELT*   y [[buffer(1)]],                     \
                   const constant uint& n [[buffer(2)]],                    \
                   uint gid [[thread_position_in_grid]])                    \
  {                                                                         \
    const uint e = gid * 4u;                                                \
    if (e + 4u <= n) {                                                      \
      const uchar4 c = *reinterpret_cast<const device uchar4*>(w + e);     \
      *reinterpret_cast<device vec<VPIPE_ELT, 4>*>(y + e) =                 \
          vec<VPIPE_ELT, 4>((VPIPE_ELT)DECODE(c.x), (VPIPE_ELT)DECODE(c.y), \
                            (VPIPE_ELT)DECODE(c.z), (VPIPE_ELT)DECODE(c.w));\
      return;                                                               \
    }                                                                       \
    for (uint i = e; i < n; ++i) { y[i] = (VPIPE_ELT)DECODE(w[i]); }        \
  }

VPIPE_FP8_DEQUANT(fp8_e4m3_dequant, fp8_e4m3_f_)
VPIPE_FP8_DEQUANT(fp8_e5m2_dequant, fp8_e5m2_f_)

// SCALED FP8 (ComfyUI's `scaled_fp8` / comfy_quant float8): value =
// code * scale, the product taken in f32 and rounded ONCE into
// VPIPE_ELT -- what the host decoder (shared/fp8-layout.h) and ComfyUI
// compute, so a runtime expansion and a dense conversion agree. Element e
// takes scale[e / div]: div = K gives one scale per output row, div = n
// the single per-tensor scale to every element.
//
//   0:codes(uchar) 1:y(VPIPE_ELT) 2:n 3:scale(float) 4:div
#define VPIPE_FP8_DEQUANT_SCALED(NAME, DECODE)                              \
  kernel void NAME(const device uchar* w     [[buffer(0)]],                 \
                   device VPIPE_ELT*   y     [[buffer(1)]],                 \
                   const constant uint& n    [[buffer(2)]],                 \
                   const device float* scale [[buffer(3)]],                 \
                   const constant uint& div  [[buffer(4)]],                 \
                   uint gid [[thread_position_in_grid]])                    \
  {                                                                         \
    const uint e = gid * 4u;                                                \
    if (e + 4u <= n) {                                                      \
      const uchar4 c = *reinterpret_cast<const device uchar4*>(w + e);     \
      *reinterpret_cast<device vec<VPIPE_ELT, 4>*>(y + e) =                 \
          vec<VPIPE_ELT, 4>((VPIPE_ELT)(DECODE(c.x) * scale[e / div]),      \
                            (VPIPE_ELT)(DECODE(c.y) * scale[(e + 1u) / div]),\
                            (VPIPE_ELT)(DECODE(c.z) * scale[(e + 2u) / div]),\
                            (VPIPE_ELT)(DECODE(c.w) * scale[(e + 3u) / div]));\
      return;                                                               \
    }                                                                       \
    for (uint i = e; i < n; ++i) {                                          \
      y[i] = (VPIPE_ELT)(DECODE(w[i]) * scale[i / div]);                    \
    }                                                                       \
  }

VPIPE_FP8_DEQUANT_SCALED(fp8_e4m3_dequant_scaled, fp8_e4m3_f_)
VPIPE_FP8_DEQUANT_SCALED(fp8_e5m2_dequant_scaled, fp8_e5m2_f_)

// INT8 (ComfyUI's int8_tensorwise): a signed code, value = code * scale,
// the same shape as scaled FP8 (shared/fp8.h's kI8).
inline float
i8_f_(uint c)
{
  return (float)as_type<char>((uchar)c);
}

VPIPE_FP8_DEQUANT_SCALED(i8_dequant_scaled, i8_f_)

// INT8 CONVROT: the codes quantize W * H^T, H the normalized REGULAR
// Hadamard -- kron powers of [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],
// [-1,1,1,1]] / 2 -- over each `group` (a power of four, 4..1024) of a
// row's inputs. H is symmetric and orthogonal, so the weight is
// (code * scale) * H again, group by group. The scale is a row's, common
// to its groups, so the CODES are un-rotated first, in integers: every
// value stays under 128 * 1024 in magnitude, so the stages are exact,
// and the weight is v * scale / sqrt(group), one product rounded ONCE
// into VPIPE_ELT -- shared/fp8.h's convrot_unrotate and fp8::decode_bf16
// to the bit. A threadgroup of 256 takes 1024 consecutive elements:
// whole groups, since a group divides 1024 and each row. Each radix-4
// stage is a quad a thread, the four values that differ in one base-4
// digit of the index: with s their sum, the d-th becomes s - 2 * the
// (3 - d)-th.
//
//   0:codes(uchar) 1:y(VPIPE_ELT) 2:n 3:scale(float) 4:div 5:group
// grid {ceil(n / 1024) * 256, 1, 1}, tg {256, 1, 1}.
kernel void i8_convrot_dequant(const device uchar* w      [[buffer(0)]],
                               device VPIPE_ELT*   y      [[buffer(1)]],
                               const constant uint& n     [[buffer(2)]],
                               const device float* scale  [[buffer(3)]],
                               const constant uint& div   [[buffer(4)]],
                               const constant uint& group [[buffer(5)]],
                               uint tg [[threadgroup_position_in_grid]],
                               uint t  [[thread_index_in_threadgroup]])
{
  threadgroup int v[1024];
  const uint e0 = tg * 1024u;
  for (uint j = 0; j < 4u; ++j) {
    const uint i = t * 4u + j, e = e0 + i;
    v[i] = e < n ? (int)as_type<char>(w[e]) : 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint st = 1u; st < group; st *= 4u) {
    const uint base = (t / st) * st * 4u + t % st;
    const int a0 = v[base], a1 = v[base + st];
    const int a2 = v[base + 2u * st], a3 = v[base + 3u * st];
    const int s = a0 + a1 + a2 + a3;
    v[base] = s - 2 * a3;
    v[base + st] = s - 2 * a2;
    v[base + 2u * st] = s - 2 * a1;
    v[base + 3u * st] = s - 2 * a0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // 1 / sqrt(group), exact: a power of two.
  const float inv = 1.0f / (float)(1u << (ctz(group) / 2u));
  for (uint j = 0; j < 4u; ++j) {
    const uint i = t * 4u + j, e = e0 + i;
    if (e < n) { y[e] = (VPIPE_ELT)((float)v[i] * scale[e / div] * inv); }
  }
}

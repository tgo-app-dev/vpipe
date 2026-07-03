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

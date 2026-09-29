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

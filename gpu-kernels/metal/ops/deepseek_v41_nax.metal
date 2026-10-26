// deepseek_v41_nax.metal -- DeepSeek-V4.1's routed experts as grouped GEMMs
// on the matrix cores (NAX): the twins of deepseek_v41.metal's
// dsv41_moe_{gu,down}_q<B>_grouped and dsv41_moe_{gu,down}_fp4_grouped,
// with their buffers, constants and sorted 32-row tiles (dsv41_moe_sort:
// tile tid.y reads ONE expert's slot, tile2e < 0 an unused tile; srow /
// sdst a pad's -1).
//
// The shape is affine_qmm_grouped_nax's (after MLX's affine_gather_qmm_rhs
// _nax): a BN x BK weight tile dequantized into threadgroup memory, 2 x 2
// simdgroups each 16 x 32 of a 32 x 64 output tile, the K step run as
// NAXTile fragments through tile_matmad_nax (f32 accumulators), each
// thread reading its two activation rows straight from device memory --
// the gate|up GEMM through srow, the down GEMM its own sorted rows. What
// differs is the weight side: an expert's codes sit in the cache's slot
// (`cw` bytes / `cs` scales apart, affine) or tight per expert (MXFP4),
// the affine scale and bias are bf16 in either build and their group is a
// runtime 32 / 64 / 128 (a thread's 32 values of a 64-wide K step lie in
// one group), and gate and up are two tiles of two weights.
//
// The steel twins widen weights and activations to f32; these stage both
// in the element type (as MLX's quantized GEMMs do), with the dequant
// s * q + b in f32 rounded once. N % 64 == 0 and K % 64 == 0 (the host
// routes other shapes to the steel twins). Built only for the tensor-
// capable target (-std=metal4.0); bound only on a GPU with matrix cores.
// dispatch (threads): {N / 64 * 32, tiles * 2, 2}, tg {32, 2, 2}.

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif
typedef VPIPE_ELT T;

#if defined(__HAVE_TENSOR__)

#include "mlx/backend/metal/kernels/steel/attn/nax.h"

namespace {

using mlx::steel::BaseNAXFrag;
using mlx::steel::NAXTile;
using mlx::steel::tile_matmad_nax;

constexpr constant int kBM = 32;     // = dsv41_moe_sort's tile
constexpr constant int kBN = 64;
constexpr constant int kBK = 64;
constexpr constant int kBKp = kBK + 16 / sizeof(T);   // padded row
constexpr constant int kWN = 2;

constant constexpr float kFp4v[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

// A 64 x 64 tile of an expert's affine codes (B bits, the little-endian bit
// stream of a row; a bf16 scale and bias per gs) into Ws as T: thread t
// takes row t / 2, the 32 values from column 32 (t % 2) -- B words of the
// stream, one group.
template <int B>
struct AffTile {
  const device uint* src;     // this thread's row, at its first value
  const device bfloat* sc;
  const device bfloat* bs;
  threadgroup T* dst;
  const int gs;
  int k;                      // the K step's first column
  METAL_FUNC AffTile(const device uchar* codes, const device bfloat* scales,
                     const device bfloat* biases, const int K, const int gs_,
                     const int row0, threadgroup T* Ws, uint t)
      : gs(gs_), k(32 * (int)(t & 1u)) {
    const int r = (int)(t >> 1);
    src = (const device uint*)(codes + (long)(row0 + r) * (K / 8) * B) +
          B * (int)(t & 1u);
    sc = scales + (long)(row0 + r) * (K / gs_);
    bs = biases + (long)(row0 + r) * (K / gs_);
    dst = Ws + r * kBKp + 32 * (int)(t & 1u);
  }
  METAL_FUNC void load() const {
    uint w[B];
    for (int i = 0; i < B; ++i) { w[i] = src[i]; }
    const float a = (float)sc[k / gs], z = (float)bs[k / gs];
    constexpr uint mask = (1u << B) - 1u;
#pragma unroll
    for (int j = 0; j < 32; j += 4) {
      vec<T, 4> v;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int bit = B * (j + c);
        const int wi = bit >> 5, off = bit & 31;
        uint q = w[wi] >> off;
        if (off + B > 32) { q |= w[wi + 1] << (32 - off); }
        v[c] = (T)fma(a, (float)(q & mask), z);
      }
      *(threadgroup vec<T, 4>*)(dst + j) = v;
    }
  }
  METAL_FUNC void next() {
    src += 2 * B;             // 64 values of B bits = 2B words
    k += kBK;
  }
};

// The same over E2M1 codes [N, K / 2] with an E8M0 scale per 32 of a row:
// thread t's 32 values are 4 words and one scale.
struct Fp4Tile {
  const device uint* src;
  const device uchar* sc;
  threadgroup T* dst;
  int k;
  METAL_FUNC Fp4Tile(const device uchar* codes, const device uchar* scales,
                     const int K, const int row0, threadgroup T* Ws, uint t)
      : k(32 * (int)(t & 1u)) {
    const int r = (int)(t >> 1);
    src = (const device uint*)(codes + (long)(row0 + r) * (K / 2)) +
          4 * (int)(t & 1u);
    sc = scales + (long)(row0 + r) * (K / 32);
    dst = Ws + r * kBKp + 32 * (int)(t & 1u);
  }
  METAL_FUNC void load() const {
    const float s = as_type<float>((uint)sc[k / 32] << 23);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint u = src[i];
#pragma unroll
      for (int j = 0; j < 8; j += 4) {
        vec<T, 4> v;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          v[c] = (T)(kFp4v[(u >> (4 * (j + c))) & 0xFu] * s);
        }
        *(threadgroup vec<T, 4>*)(dst + 8 * i + j) = v;
      }
    }
  }
  METAL_FUNC void next() {
    src += 8;
    k += kBK;
  }
};

constexpr constant short kSM = kBM / 2, kSN = kBN / kWN, kSK = 32;
constexpr constant short kTN = kSN / 16, kTK = kSK / 16;

// This thread's two activation rows' K step into a fragment tile.
METAL_FUNC void
load_rows(thread NAXTile<T, 1, kTK>& At, const device T* x0,
          const device T* x1, const bool on0, const bool on1, const int k)
{
  STEEL_PRAGMA_UNROLL
  for (short j = 0; j < kTK; ++j) {
    thread auto& f = At.frag_at(0, j);
    const int kj = k + j * 16;
    const vec<T, 4> a0 = on0 ? *(const device vec<T, 4>*)(x0 + kj)
                             : vec<T, 4>(0);
    const vec<T, 4> a1 = on1 ? *(const device vec<T, 4>*)(x1 + kj)
                             : vec<T, 4>(0);
    STEEL_PRAGMA_UNROLL
    for (short c = 0; c < 4; ++c) {
      f[c] = a0[c];
      f[4 + c] = a1[c];
    }
  }
}

// gate|up: act[row] = swt[row] * silu(min(g, lim)) * clamp(u, +-lim) for
// the tile's 32 sorted rows (hidden rows srow[row], < 0 a zero pad).
template <typename L>
METAL_FUNC void
gu_impl(thread L& lg, thread L& lu, const device T* x, device T* act,
        const device int* srow, const device float* swt, const int K,
        const int I, const float lim, threadgroup T* Wg, threadgroup T* Wu,
        uint3 tid, uint sgid)
{
  const int y_row = (int)tid.y * kBM, y_col = (int)tid.x * kBN;
  const short tm = kSM * (short)(sgid / kWN);
  const short tn = kSN * (short)(sgid % kWN);
  const short2 sc = BaseNAXFrag::get_coord();
  const int s0 = y_row + tm + sc.y, s1 = s0 + 8;
  const int r0 = srow[s0], r1 = srow[s1];
  const device T* x0 = x + (long)max(r0, 0) * K + sc.x;
  const device T* x1 = x + (long)max(r1, 0) * K + sc.x;
  const bool on0 = r0 >= 0, on1 = r1 >= 0;
  NAXTile<float, 1, kTN> Dg, Du;
  Dg.clear();
  Du.clear();
  for (int k0 = 0; k0 < K; k0 += kBK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    lg.load();
    lu.load();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_NO_UNROLL
    for (short kk = 0; kk < kBK; kk += kSK) {
      NAXTile<T, 1, kTK> At;
      NAXTile<T, kTN, kTK> Bg, Bu;
      volatile int compiler_barrier;
      load_rows(At, x0, x1, on0, on1, k0 + kk);
      Bg.template load<T, kBKp, 1>(Wg + tn * kBKp + kk);
      Bu.template load<T, kBKp, 1>(Wu + tn * kBKp + kk);
      tile_matmad_nax(Dg, At, metal::bool_constant<false>{}, Bg,
                      metal::bool_constant<true>{});
      tile_matmad_nax(Du, At, metal::bool_constant<false>{}, Bu,
                      metal::bool_constant<true>{});
      (void)compiler_barrier;
    }
    lg.next();
    lu.next();
  }
  // fragment element i: row (i / 4) * 8, column i % 4 of its fragment
  STEEL_PRAGMA_UNROLL
  for (short j = 0; j < kTN; ++j) {
    const thread auto& fg = Dg.frag_at(0, j);
    const thread auto& fu = Du.frag_at(0, j);
    const int col = y_col + tn + j * 16 + sc.x;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; ++i) {
      const int row = i == 0 ? s0 : s1;
      const float w = swt[row];
      vec<T, 4> o;
      STEEL_PRAGMA_UNROLL
      for (short c = 0; c < 4; ++c) {
        float gg = fg[i * 4 + c], uu = fu[i * 4 + c];
        if (lim > 0.0f) {
          uu = clamp(uu, -lim, lim);
          gg = min(gg, lim);
        }
        o[c] = (T)(w * (gg / (1.0f + exp(-gg)) * uu));
      }
      *(device vec<T, 4>*)(act + (long)row * I + col) = o;
    }
  }
}

// down: part[sdst[row]] = w2 . act[row], rounded to T; pads store nothing.
template <typename L>
METAL_FUNC void
down_impl(thread L& lw, const device T* act, device T* part,
          const device int* sdst, const int I, const int H,
          threadgroup T* Ws, uint3 tid, uint sgid)
{
  const int y_row = (int)tid.y * kBM, y_col = (int)tid.x * kBN;
  const short tm = kSM * (short)(sgid / kWN);
  const short tn = kSN * (short)(sgid % kWN);
  const short2 sc = BaseNAXFrag::get_coord();
  const int s0 = y_row + tm + sc.y, s1 = s0 + 8;
  const device T* x0 = act + (long)s0 * I + sc.x;
  const device T* x1 = act + (long)s1 * I + sc.x;
  NAXTile<float, 1, kTN> D;
  D.clear();
  for (int k0 = 0; k0 < I; k0 += kBK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    lw.load();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_NO_UNROLL
    for (short kk = 0; kk < kBK; kk += kSK) {
      NAXTile<T, 1, kTK> At;
      NAXTile<T, kTN, kTK> Bt;
      volatile int compiler_barrier;
      load_rows(At, x0, x1, true, true, k0 + kk);
      Bt.template load<T, kBKp, 1>(Ws + tn * kBKp + kk);
      tile_matmad_nax(D, At, metal::bool_constant<false>{}, Bt,
                      metal::bool_constant<true>{});
      (void)compiler_barrier;
    }
    lw.next();
  }
  STEEL_PRAGMA_UNROLL
  for (short j = 0; j < kTN; ++j) {
    const thread auto& f = D.frag_at(0, j);
    const int col = y_col + tn + j * 16 + sc.x;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; ++i) {
      const int dr = sdst[i == 0 ? s0 : s1];
      if (dr < 0) { continue; }
      vec<T, 4> o;
      STEEL_PRAGMA_UNROLL
      for (short c = 0; c < 4; ++c) { o[c] = (T)f[i * 4 + c]; }
      *(device vec<T, 4>*)(part + (long)dr * H + col) = o;
    }
  }
}

}  // namespace

#define DSV41_NAX_Q(B)                                                       \
  kernel void dsv41_moe_gu_nax_q##B(                                         \
      const device uchar* w1 [[buffer(0)]],                                  \
      const device bfloat* s1 [[buffer(1)]],                                 \
      const device bfloat* b1 [[buffer(2)]],                                 \
      const device uchar* w3 [[buffer(3)]],                                  \
      const device bfloat* s3 [[buffer(4)]],                                 \
      const device bfloat* b3 [[buffer(5)]],                                 \
      const device T* x [[buffer(6)]], device T* act [[buffer(7)]],          \
      const device int* srow [[buffer(8)]],                                  \
      const device float* swt [[buffer(9)]],                                 \
      const device int* tile2e [[buffer(10)]],                               \
      constant int& K [[buffer(11)]], constant int& I [[buffer(12)]],        \
      constant float& lim [[buffer(13)]], constant int& gs [[buffer(14)]],   \
      constant long& cw [[buffer(15)]], constant long& cs [[buffer(16)]],    \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint sgid [[simdgroup_index_in_threadgroup]],                          \
      uint t [[thread_index_in_threadgroup]])                                \
  {                                                                          \
    const int e = tile2e[tid.y];                                             \
    if (e < 0) { return; }                                                   \
    threadgroup T Wg[kBN * kBKp], Wu[kBN * kBKp];                            \
    const int y_col = (int)tid.x * kBN;                                      \
    AffTile<B> lg(w1 + (long)e * cw, s1 + (long)e * cs, b1 + (long)e * cs,   \
                  K, gs, y_col, Wg, t);                                      \
    AffTile<B> lu(w3 + (long)e * cw, s3 + (long)e * cs, b3 + (long)e * cs,   \
                  K, gs, y_col, Wu, t);                                      \
    gu_impl(lg, lu, x, act, srow, swt, K, I, lim, Wg, Wu, tid, sgid);        \
  }                                                                          \
  kernel void dsv41_moe_down_nax_q##B(                                       \
      const device uchar* w2 [[buffer(0)]],                                  \
      const device bfloat* s2 [[buffer(1)]],                                 \
      const device bfloat* b2 [[buffer(2)]],                                 \
      const device T* act [[buffer(3)]], device T* part [[buffer(4)]],       \
      const device int* sdst [[buffer(5)]],                                  \
      const device int* tile2e [[buffer(6)]],                                \
      constant int& I [[buffer(7)]], constant int& H [[buffer(8)]],          \
      constant int& gs [[buffer(9)]], constant long& cw [[buffer(10)]],      \
      constant long& cs [[buffer(11)]],                                      \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint sgid [[simdgroup_index_in_threadgroup]],                          \
      uint t [[thread_index_in_threadgroup]])                                \
  {                                                                          \
    const int e = tile2e[tid.y];                                             \
    if (e < 0) { return; }                                                   \
    threadgroup T Ws[kBN * kBKp];                                            \
    AffTile<B> lw(w2 + (long)e * cw, s2 + (long)e * cs, b2 + (long)e * cs,   \
                  I, gs, (int)tid.x * kBN, Ws, t);                           \
    down_impl(lw, act, part, sdst, I, H, Ws, tid, sgid);                     \
  }

DSV41_NAX_Q(2)
DSV41_NAX_Q(3)
DSV41_NAX_Q(4)
DSV41_NAX_Q(5)
DSV41_NAX_Q(6)
DSV41_NAX_Q(8)

kernel void dsv41_moe_gu_nax_fp4(
    const device uchar* w1     [[buffer(0)]],
    const device uchar* s1     [[buffer(1)]],
    const device uchar* w3     [[buffer(2)]],
    const device uchar* s3     [[buffer(3)]],
    const device T*     x      [[buffer(4)]],
    device T*           act    [[buffer(5)]],
    const device int*   srow   [[buffer(6)]],
    const device float* swt    [[buffer(7)]],
    const device int*   tile2e [[buffer(8)]],
    constant int&       K      [[buffer(9)]],
    constant int&       I      [[buffer(10)]],
    constant float&     lim    [[buffer(11)]],
    uint3 tid  [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  t    [[thread_index_in_threadgroup]])
{
  const int e = tile2e[tid.y];
  if (e < 0) { return; }
  threadgroup T Wg[kBN * kBKp], Wu[kBN * kBKp];
  const long wo = (long)e * I * (K / 2), so = (long)e * I * (K / 32);
  const int y_col = (int)tid.x * kBN;
  Fp4Tile lg(w1 + wo, s1 + so, K, y_col, Wg, t);
  Fp4Tile lu(w3 + wo, s3 + so, K, y_col, Wu, t);
  gu_impl(lg, lu, x, act, srow, swt, K, I, lim, Wg, Wu, tid, sgid);
}

kernel void dsv41_moe_down_nax_fp4(
    const device uchar* w2     [[buffer(0)]],
    const device uchar* s2     [[buffer(1)]],
    const device T*     act    [[buffer(2)]],
    device T*           part   [[buffer(3)]],
    const device int*   sdst   [[buffer(4)]],
    const device int*   tile2e [[buffer(5)]],
    constant int&       I      [[buffer(6)]],
    constant int&       H      [[buffer(7)]],
    uint3 tid  [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  t    [[thread_index_in_threadgroup]])
{
  const int e = tile2e[tid.y];
  if (e < 0) { return; }
  threadgroup T Ws[kBN * kBKp];
  Fp4Tile lw(w2 + (long)e * H * (I / 2), s2 + (long)e * H * (I / 32), I,
             (int)tid.x * kBN, Ws, t);
  down_impl(lw, act, part, sdst, I, H, Ws, tid, sgid);
}

// ---- the sparse attention -------------------------------------------------
//
// dsv41_attn on the matrix cores, FlashMLA's way: the heads of one query
// share its keys (the window, then the picks; one latent as K and V), so
// they are the rows of the products. A threadgroup takes 32 heads of a
// query -- 2 row groups of 16 by 4 slices of the 512 dims (MLX's head-dim
// split attention, #4487) -- through the query's keys 32 at a time: S =
// Q K^T on NAX fragments in f32, each slice's partial scores summed in a
// fixed order through threadgroup memory, the online softmax in base 2
// with the head's sink, O += P V for the slice (P in two parts of the
// element type unless psplit is 0). A key nothing wrote (a
// pick of -1 or past what the query sees) loads zeros and scores -inf.
// The output's rotary tail rotated back, stored group-major as dsv41_attn
// stores it. Rows past NH (a model of fewer heads) load zeros and store
// nothing. D == 512. grid {256 * ceil(NH / 32), n}, tg {256}.

namespace {

constexpr constant short kAU = 16;
constexpr constant int kAD = 512, kAWN = 4, kADh = kAD / kAWN;
constexpr constant short kATK = 2, kATD = kADh / kAU;   // 32 keys, 8 frags
constexpr constant float kAMin = -3.0e38f;

// key `code` (row << 2 | buffer: 0 this chunk's, 1 the ring, 2 the
// compressed), or null for none
METAL_FUNC const device T*
attn_key_(const int code, const device T* kvn, const device T* ring,
          const device T* comp)
{
  if (code < 0) { return nullptr; }
  const int kind = code & 3;
  const device T* b = kind == 0 ? kvn : (kind == 1 ? ring : comp);
  return b + (long)(code >> 2) * kAD;
}

METAL_FUNC vec<T, 4>
attn_ld4_(const device T* row, const int col)
{
  return row != nullptr ? *(const device vec<T, 4>*)(row + col)
                        : vec<T, 4>(0);
}

}  // namespace

kernel void dsv41_attn_nax(
    const device T*      q     [[buffer(0)]],   // [n, NH, 512]
    const device T*      kvn   [[buffer(1)]],   // [n, 512]
    const device T*      ring  [[buffer(2)]],   // [R, 512]
    const device T*      comp  [[buffer(3)]],   // [C, 512]
    const device int*    idx   [[buffer(4)]],   // [n, KC]
    const device float*  sink  [[buffer(5)]],   // [NH]
    device T*            o     [[buffer(6)]],
    const device float2* tab   [[buffer(7)]],
    constant int&        n     [[buffer(8)]],
    constant int&        NH    [[buffer(9)]],
    constant int&        W     [[buffer(10)]],
    constant int&        p0    [[buffer(11)]],
    constant int&        KC    [[buffer(12)]],
    constant int&        ratio [[buffer(13)]],
    constant float&      scale [[buffer(14)]],
    constant int&        G     [[buffer(15)]],
    constant int&        rd    [[buffer(16)]],
    constant int&        R     [[buffer(17)]],
    constant int&        lo    [[buffer(18)]],
    constant int&        psplit [[buffer(19)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  t    [[thread_index_in_threadgroup]])
{
  using stile_t = NAXTile<float, 1, kATK>;
  using otile_t = NAXTile<float, 1, kATD>;
  constexpr short kEPF = stile_t::NAXFrag_t::kElemsPerFrag;
  constexpr short kRPT = otile_t::kRowsPerThread;     // 2
  threadgroup float xchg[2][kAWN][kATK * kEPF * 32];
  threadgroup int kc[2][32];

  const int i = (int)tg.y;
  const int pos = p0 + i;
  const short rg = (short)(sgid / kAWN), dg = (short)(sgid % kAWN);
  const int hbase = (int)tg.x * 32 + rg * kAU;      // this group's heads
  const short2 sc = BaseNAXFrag::get_coord();
  const short sm = sc.y, sn = sc.x;
  // the query's keys: the window from wlo (a CED decoder's from its tail),
  // then the picks
  const int wlo = max(max(0, lo), pos - W + 1);
  const int nw = max(pos - wlo + 1, 0);
  const int vis = (pos + 1) / max(ratio, 1);
  const int total = nw + KC;
  const int nb = (total + 31) / 32;
  auto code = [&](int j) -> int {
    if (j < nw) {
      const int p = wlo + j;
      return p >= p0 ? ((p - p0) << 2) : (((p % R) << 2) | 1);
    }
    if (j >= total) { return -1; }
    const int tt = idx[(long)i * KC + (j - nw)];
    return tt >= 0 && tt < vis ? ((tt << 2) | 2) : -1;
  };
  if (t < 32u) { kc[0][t] = code((int)t); }

  // Q's slice of this group's 16 heads, resident
  const int lim = NH - hbase;                       // rows that are heads
  const device T* qp = q + ((long)i * NH + max(min(hbase, NH - 1), 0)) * kAD +
                       dg * kADh;
  NAXTile<T, 1, 1> Qt[kATD];
  STEEL_PRAGMA_UNROLL
  for (short id = 0; id < kATD; ++id) {
    if (lim >= kAU) {
      Qt[id].load(qp + id * kAU, kAD);
    } else {
      Qt[id].load_rows(qp + id * kAU, kAD, (short)max(lim, 0));
    }
  }
  otile_t Ot;
  Ot.clear();
  metal::vec<float, kRPT> mx, sum;
  STEEL_PRAGMA_UNROLL
  for (short r = 0; r < kRPT; ++r) {
    const int h = hbase + sm + r * 8;
    mx[r] = h < NH ? M_LOG2E_F * sink[h] : 0.0f;
    sum[r] = 1.0f;
  }
  const float scale2 = scale * M_LOG2E_F;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int kb = 0; kb < nb; ++kb) {
    threadgroup const int* kcb = kc[kb & 1];
    stile_t St;
    St.clear();
    // S = Q K^T over this slice's 128 dims
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < kATD; ++id) {
      const int c0 = dg * kADh + id * kAU + sn;
      NAXTile<T, 2, 1> Kt;
      STEEL_PRAGMA_UNROLL
      for (short f = 0; f < 2; ++f) {
        thread auto& fr = Kt.frag_at(f, 0);
        STEEL_PRAGMA_UNROLL
        for (short r = 0; r < 2; ++r) {
          const vec<T, 4> v = attn_ld4_(
              attn_key_(kcb[f * 16 + sm + r * 8], kvn, ring, comp), c0);
          STEEL_PRAGMA_UNROLL
          for (short c = 0; c < 4; ++c) { fr[r * 4 + c] = v[c]; }
        }
      }
      stile_t::NAXFrag_t::mma(St.frag_at(0, 0), St.frag_at(0, 1),
                              Qt[id].frag_at(0, 0), metal::false_type{},
                              Kt.frag_at(0, 0), Kt.frag_at(1, 0),
                              metal::true_type{});
    }
    // the slices' partial scores, summed in slice order
    threadgroup float* slot = xchg[rg][dg];
    const short base = (short)lane * (kATK * kEPF);
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < kATK; ++ik) {
      thread auto& sf = St.frag_at(0, ik);
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < kEPF; ++e) { slot[base + ik * kEPF + e] = sf[e]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // the next block's keys (every group is past this block's start, so
    // past the block before's last read of that half)
    if (t < 32u && kb + 1 < nb) {
      kc[(kb + 1) & 1][t] = code((kb + 1) * 32 + (int)t);
    }
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < kATK; ++ik) {
      thread auto& sf = St.frag_at(0, ik);
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < kEPF; ++e) {
        float v = xchg[rg][0][base + ik * kEPF + e];
        STEEL_PRAGMA_UNROLL
        for (short g = 1; g < kAWN; ++g) {
          v += xchg[rg][g][base + ik * kEPF + e];
        }
        sf[e] = v * scale2;
      }
      // a column (key) nothing wrote: -inf
      STEEL_PRAGMA_UNROLL
      for (short c = 0; c < 4; ++c) {
        if (kcb[ik * 16 + sn + c] < 0) {
          sf[c] = kAMin;
          sf[4 + c] = kAMin;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // the online softmax, each slice for itself (the same rows)
    metal::vec<float, kRPT> nmx = mx, fac;
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < kATK; ++ik) {
      const thread auto& sf = St.frag_at(0, ik);
      STEEL_PRAGMA_UNROLL
      for (short r = 0; r < kRPT; ++r) {
        float m = max(max(sf[r * 4], sf[r * 4 + 1]),
                      max(sf[r * 4 + 2], sf[r * 4 + 3]));
        m = max(m, simd_shuffle_xor(m, ushort(1)));
        m = max(m, simd_shuffle_xor(m, ushort(8)));
        nmx[r] = max(nmx[r], m);
      }
    }
    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < kRPT; ++r) {
      fac[r] = fast::exp2(mx[r] - nmx[r]);
      mx[r] = nmx[r];
      sum[r] *= fac[r];
    }
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < kATK; ++ik) {
      thread auto& sf = St.frag_at(0, ik);
      STEEL_PRAGMA_UNROLL
      for (short r = 0; r < kRPT; ++r) {
        float ps = 0.0f;
        STEEL_PRAGMA_UNROLL
        for (short c = 0; c < 4; ++c) {
          sf[r * 4 + c] = fast::exp2(sf[r * 4 + c] - nmx[r]);
          ps += sf[r * 4 + c];
        }
        ps += simd_shuffle_xor(ps, ushort(1));
        ps += simd_shuffle_xor(ps, ushort(8));
        sum[r] += ps;
      }
    }
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < kATD; ++id) {
      thread auto& of = Ot.frag_at(0, id);
      STEEL_PRAGMA_UNROLL
      for (short r = 0; r < kRPT; ++r) {
        STEEL_PRAGMA_UNROLL
        for (short c = 0; c < 4; ++c) { of[r * 4 + c] *= fac[r]; }
      }
    }
    // O += P V over this slice
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < kATD; id += 2) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < kATK; ++ik) {
        NAXTile<T, 1, 2> Vt;
        STEEL_PRAGMA_UNROLL
        for (short r = 0; r < 2; ++r) {
          const device T* kr =
              attn_key_(kcb[ik * 16 + sm + r * 8], kvn, ring, comp);
          STEEL_PRAGMA_UNROLL
          for (short f = 0; f < 2; ++f) {
            const vec<T, 4> v =
                attn_ld4_(kr, dg * kADh + (id + f) * kAU + sn);
            thread auto& fr = Vt.frag_at(0, f);
            STEEL_PRAGMA_UNROLL
            for (short c = 0; c < 4; ++c) { fr[r * 4 + c] = v[c]; }
          }
        }
        if (psplit != 0) {
          // P in two parts of V's type, P = hi + lo: the product rounds its
          // f32 operand to V's type (MEASURED: f16 twice dsv41_attn's
          // error against f64 without the split, the same with it)
          NAXTile<float, 1, 1> Ph, Pl;
          const thread auto& sf = St.frag_at(0, ik);
          STEEL_PRAGMA_UNROLL
          for (short e = 0; e < kEPF; ++e) {
            const float hi = (float)(T)sf[e];
            Ph.frag_at(0, 0)[e] = hi;
            Pl.frag_at(0, 0)[e] = sf[e] - hi;
          }
          otile_t::NAXFrag_t::mma(Ot.frag_at(0, id), Ot.frag_at(0, id + 1),
                                  Ph.frag_at(0, 0), metal::false_type{},
                                  Vt.frag_at(0, 0), Vt.frag_at(0, 1),
                                  metal::false_type{});
          otile_t::NAXFrag_t::mma(Ot.frag_at(0, id), Ot.frag_at(0, id + 1),
                                  Pl.frag_at(0, 0), metal::false_type{},
                                  Vt.frag_at(0, 0), Vt.frag_at(0, 1),
                                  metal::false_type{});
        } else {
        otile_t::NAXFrag_t::mma(Ot.frag_at(0, id), Ot.frag_at(0, id + 1),
                                St.frag_at(0, ik), metal::false_type{},
                                Vt.frag_at(0, 0), Vt.frag_at(0, 1),
                                metal::false_type{});
        }
      }
    }
  }

  // normalize, rotate the rotary tail back, store group-major
  const int D = kAD;
  const int hg = NH / G;
  STEEL_PRAGMA_UNROLL
  for (short r = 0; r < kRPT; ++r) {
    const int h = hbase + sm + r * 8;
    if (h >= NH) { continue; }
    const float inv = 1.0f / sum[r];
    const int g = h / hg, hh = h % hg;
    device T* orow = o + ((long)g * n + i) * (long)(hg * D) + hh * D;
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < kATD; ++id) {
      const thread auto& of = Ot.frag_at(0, id);
      const int d = dg * kADh + id * kAU + sn;
      float v[4];
      STEEL_PRAGMA_UNROLL
      for (short c = 0; c < 4; ++c) { v[c] = of[r * 4 + c] * inv; }
      if (d >= D - rd) {
        STEEL_PRAGMA_UNROLL
        for (short c = 0; c < 4; c += 2) {
          const int j = (d + c - (D - rd)) / 2;
          const float2 cs = tab[(long)pos * (rd / 2) + j];
          const float re = v[c], im = v[c + 1];
          v[c] = re * cs.x + im * cs.y;
          v[c + 1] = -re * cs.y + im * cs.x;
        }
      }
      *(device vec<T, 4>*)(orow + d) =
          vec<T, 4>((T)v[0], (T)v[1], (T)v[2], (T)v[3]);
    }
  }
}

// ---- the indexer's scores --------------------------------------------------
//
// dsv41_index_score_mma on the matrix cores: S[i, t] = sum_h w[i, h] *
// relu(q[i, h] . k[t]) -- -inf where the query cannot see t, or (under a
// candidate source) where t's block was not kept. A threadgroup takes 4
// query rows by 64 keys: the keys' tile [64, 128] staged once in
// threadgroup memory for the 4 rows; a simdgroup a row's 32 heads by 32
// keys on NAX fragments (f32 accumulators), then w * relu over the heads
// in a fixed tree order. A tile no row sees only writes -inf. IH <= 32
// (padded to 32). grid {256 * ceil(C / 64), ceil(n / 4)}, tg {256}.

namespace {

constexpr constant int kIQ = 4, kIK = 64, kID = 128, kIDp = kID + 8;

}  // namespace

kernel void dsv41_index_score_nax(
    const device T*     q        [[buffer(0)]],   // [n, IH, 128]
    const device T*     keys     [[buffer(1)]],   // [C, 128]
    const device T*     w        [[buffer(2)]],   // [n, IH]
    const device uchar* cand     [[buffer(3)]],   // [n, nblk]
    device float*       S        [[buffer(4)]],   // [n, C]
    constant int&       IH       [[buffer(5)]],
    constant int&       C        [[buffer(6)]],
    constant int&       p0       [[buffer(7)]],
    constant int&       ratio    [[buffer(8)]],
    constant float&     wscale   [[buffer(9)]],
    constant int&       use_cand [[buffer(10)]],
    constant int&       blk      [[buffer(11)]],
    constant int&       nblk     [[buffer(12)]],
    constant int&       n        [[buffer(13)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  t    [[thread_index_in_threadgroup]])
{
  threadgroup T ks[kIK * kIDp];
  threadgroup int live;
  const int t0 = (int)tg.x * kIK;
  const int i0 = (int)tg.y * kIQ;
  const int nr = min(kIQ, n - i0);
  // the last row sees the most; a block's keys share their fate
  auto ok = [&](int i, int tt) {
    return tt < C && tt < (p0 + i + 1) / ratio &&
           (use_cand == 0 || cand[(long)i * nblk + tt / blk] != 0);
  };
  if (t == 0) {
    int any = 0;
    for (int r = 0; r < nr && any == 0; ++r) {
      for (int tt = t0; tt < min(C, t0 + kIK) && any == 0; tt += blk) {
        any = ok(i0 + r, tt) ? 1 : 0;
      }
    }
    live = any;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (live == 0) {
    for (int k = (int)t; k < nr * kIK; k += 256) {
      const int r = k / kIK, j = k % kIK;
      if (t0 + j < C) { S[(long)(i0 + r) * C + t0 + j] = -INFINITY; }
    }
    return;
  }
  // the keys' tile, 8 values a load (zeros past C)
  for (int k = (int)t; k < kIK * kID / 8; k += 256) {
    const int j = k / (kID / 8), d = (k % (kID / 8)) * 8;
    const int tt = t0 + j;
    vec<T, 8> v = tt < C ? *(const device vec<T, 8>*)(keys + (long)tt * kID + d)
                         : vec<T, 8>(0);
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < 8; ++e) { ks[j * kIDp + d + e] = v[e]; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int r = (int)sgid / 2, kh = (int)sgid % 2;
  if (r >= nr) { return; }
  const int i = i0 + r;
  const short2 sc = BaseNAXFrag::get_coord();
  const short sm = sc.y, sn = sc.x;
  // S [32 heads, 32 keys]: 2 row fragments (heads 0-15, 16-31) by 2 key
  // fragments, the K steps of 16 dims
  NAXTile<float, 2, 2> St;
  St.clear();
  const device T* qr = q + (long)i * IH * kID;
  const threadgroup T* kt = ks + kh * 32 * kIDp;
  STEEL_PRAGMA_UNROLL
  for (short id = 0; id < kID / kAU; ++id) {
    NAXTile<T, 2, 1> Qt;
    if (IH >= 32) {
      Qt.load(qr + id * kAU, kID);
    } else {
      Qt.load_rows(qr + id * kAU, kID, (short)IH);
    }
    NAXTile<T, 2, 1> Kt;
    Kt.template load<T, kIDp, 1>(kt + id * kAU);
    STEEL_PRAGMA_UNROLL
    for (short rf = 0; rf < 2; ++rf) {
      BaseNAXFrag::mma(St.frag_at(rf, 0), St.frag_at(rf, 1),
                       Qt.frag_at(rf, 0), metal::false_type{},
                       Kt.frag_at(0, 0), Kt.frag_at(1, 0),
                       metal::true_type{});
    }
  }
  // w * relu summed over the heads: a thread's 4 rows, then its column's
  // other lanes (lane bits 1, 2, 4)
  float wr[2][2];
  STEEL_PRAGMA_UNROLL
  for (short rf = 0; rf < 2; ++rf) {
    STEEL_PRAGMA_UNROLL
    for (short rr = 0; rr < 2; ++rr) {
      const int h = rf * 16 + sm + rr * 8;
      wr[rf][rr] = h < IH ? (float)w[(long)i * IH + h] * wscale : 0.0f;
    }
  }
  STEEL_PRAGMA_UNROLL
  for (short cf = 0; cf < 2; ++cf) {
    float acc[4];
    STEEL_PRAGMA_UNROLL
    for (short c = 0; c < 4; ++c) {
      float v = 0.0f;
      STEEL_PRAGMA_UNROLL
      for (short rf = 0; rf < 2; ++rf) {
        const thread auto& f = St.frag_at(rf, cf);
        STEEL_PRAGMA_UNROLL
        for (short rr = 0; rr < 2; ++rr) {
          v += wr[rf][rr] * max(f[rr * 4 + c], 0.0f);
        }
      }
      v += simd_shuffle_xor(v, ushort(2));
      v += simd_shuffle_xor(v, ushort(4));
      v += simd_shuffle_xor(v, ushort(16));
      acc[c] = v;
    }
    if ((lane & 22u) == 0u) {
      STEEL_PRAGMA_UNROLL
      for (short c = 0; c < 4; ++c) {
        const int tt = t0 + kh * 32 + cf * 16 + sn + c;
        if (tt < C) {
          S[(long)i * C + tt] = ok(i, tt) ? acc[c] : -INFINITY;
        }
      }
    }
  }
}

#endif  // __HAVE_TENSOR__

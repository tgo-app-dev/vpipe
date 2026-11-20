// qwen_moe_native.metal -- a Qwen MoE's routed experts (Qwen3.8-Flash-
// Next's llama.cpp GGUFs) run from their own blocks, as the expert cache
// holds them: a slot's gate, up and down slabs as the file has them, each
// row's 32-value sub-blocks read by ops/ggml_native.h. The affine gathers'
// and grouped GEMMs' contracts (affine_qmv.metal, the steel grouped twins):
// act[p, I] = silu(gate . x) * (up . x), part[p, H] = down . act, unweighted
// (the combine weighs them), f32 sums rounded to the element type once.
//
// Element type: half by default, -DVPIPE_ELT=bfloat for the bf16 library.

#include <metal_stdlib>
using namespace metal;

#include "ops/ggml_native.h"
#ifndef METAL_FUNC
#define METAL_FUNC inline
#endif
#define MLX_MTL_CONST static constant constexpr const
#define SIMD_SIZE 32
typedef bfloat bfloat16_t;
#include "mlx/backend/metal/kernels/complex.h"
#include "mlx/backend/metal/kernels/steel/gemm/mma.h"

#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif
typedef VPIPE_ELT T;

using namespace ggml_native;

inline float qn_dot32(thread const float* w, const device T* x)
{
  const device vec<T, 4>* xp = (const device vec<T, 4>*)x;
  float s = 0.0f;
  for (int q = 0; q < 8; ++q) {
    const float4 y = float4(xp[q]);
    s += dot(float4(w[4 * q], w[4 * q + 1], w[4 * q + 2], w[4 * q + 3]), y);
  }
  return s;
}

// ---- a decode's or a verify's pairs: one pair a threadgroup column, a lane
// a sub-block, 2 rows a simdgroup, 4 simdgroups.
// grid {32, ceil(I / 8) * 4, pairs}, tg {32, 4, 1}.
kernel void qwen_native_gather_swiglu(
    const device uchar* wg [[buffer(0)]],    // the pools: gate slabs
    const device uchar* wu [[buffer(1)]],    // up slabs
    const device ulong* tab [[buffer(2)]],
    const device T* x [[buffer(3)]],         // [M, H]
    device T* act [[buffer(4)]],             // [np, I]
    const device int* slot [[buffer(5)]],    // [np]
    constant int& H [[buffer(6)]], constant int& I [[buffer(7)]],
    constant int& topk [[buffer(8)]], constant uint& tg_ [[buffer(9)]],
    constant uint& tu [[buffer(10)]], constant long& cwg [[buffer(11)]],
    constant long& cwu [[buffer(12)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
  const int p = (int)tg.z;
  const long e = slot[p];
  const int f0 = ((int)tg.y * 4 + (int)sg) * 2;
  const long rg = row_bytes(tg_, H), ru = row_bytes(tu, H);
  const device T* xr = x + (long)(p / topk) * H;
  float g[2] = {0.0f, 0.0f}, u[2] = {0.0f, 0.0f};
  for (int sb = (int)lane; sb < H / 32; sb += 32) {
    for (int r = 0; r < 2; ++r) {
      const int f = f0 + r;
      if (f >= I) { break; }
      float w[32];
      w32(tg_, wg + e * cwg + f * rg, sb, tab, w);
      g[r] += qn_dot32(w, xr + sb * 32);
      w32(tu, wu + e * cwu + f * ru, sb, tab, w);
      u[r] += qn_dot32(w, xr + sb * 32);
    }
  }
  for (int r = 0; r < 2; ++r) {
    const float gg = simd_sum(g[r]), uu = simd_sum(u[r]);
    const int f = f0 + r;
    if (lane == 0 && f < I) {
      act[(long)p * I + f] = (T)((gg / (1.0f + exp(-gg))) * uu);
    }
  }
}

// grid {32, ceil(H / 8) * 4, pairs}, tg {32, 4, 1}.
kernel void qwen_native_gather_down(
    const device uchar* wd [[buffer(0)]],
    const device ulong* tab [[buffer(1)]],
    const device T* act [[buffer(2)]],       // [np, I]
    device T* part [[buffer(3)]],            // [np, H]
    const device int* slot [[buffer(4)]],
    constant int& I [[buffer(5)]], constant int& H [[buffer(6)]],
    constant uint& td [[buffer(7)]], constant long& cwd [[buffer(8)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
  const int p = (int)tg.z;
  const long e = slot[p];
  const int o0 = ((int)tg.y * 4 + (int)sg) * 2;
  const long rd = row_bytes(td, I);
  const device T* ar = act + (long)p * I;
  float a[2] = {0.0f, 0.0f};
  for (int sb = (int)lane; sb < I / 32; sb += 32) {
    for (int r = 0; r < 2; ++r) {
      const int o = o0 + r;
      if (o >= H) { break; }
      float w[32];
      w32(td, wd + e * cwd + o * rd, sb, tab, w);
      a[r] += qn_dot32(w, ar + sb * 32);
    }
  }
  for (int r = 0; r < 2; ++r) {
    const float s = simd_sum(a[r]);
    const int o = o0 + r;
    if (lane == 0 && o < H) { part[(long)p * H + o] = (T)s; }
  }
}

// ---- a prefill: the pairs sorted into 32-row tiles of one expert (t2e:
// the tile's slot, -1 past the last; srow / sdst: a row's input row and
// pair, -1 a pad), each tile's weights read once for its rows, on the
// steel BlockMMA (f32). A thread widens its own 8 of a row's 32 tile
// columns (w8): a sub-block's four threads each read a quarter of it.

// A 32 x 32 weight tile of rows y_col.. of a slab, in f32.
template <bool W8>
struct QnLoader {
  const short bi, bj;
  threadgroup float* dst;
  const device uchar* row;
  const uint t;
  const device ulong* tab;
  int k_off;
  METAL_FUNC QnLoader(const device uchar* slab, const int K, const uint t_,
                      const device ulong* tab_, const int y_col,
                      threadgroup float* dst_, ushort sg, ushort lane)
      : bi((short)((sg * 32 + lane) / 4)),
        bj((short)(8 * ((sg * 32 + lane) % 4))),
        dst(dst_ + bi * 36 + bj),
        row(slab + (long)(y_col + bi) * row_bytes(t_, K)), t(t_), tab(tab_),
        k_off(0) {}
  METAL_FUNC void load() const {
    if (W8) {
      float v[8];
      w8(t, row, k_off / 32, bj / 8, tab, v);
      for (short j = 0; j < 8; ++j) { dst[j] = v[j]; }
    } else {
      // (the bench's reference: the whole sub-block widened, 8 kept)
      float v[32];
      w32(t, row, k_off / 32, tab, v);
      for (short j = 0; j < 8; ++j) { dst[j] = v[bj + j]; }
    }
  }
  METAL_FUNC void next() { k_off += 32; }
};

// A 32 x 32 tile of input rows in f32: srow[base + r] (< 0 a zero row), or
// contiguous (srow null).
struct QnRowLoader {
  const short bi, bj;
  threadgroup float* dst;
  const device T* src;
  const int ld;
  int r;
  int k_off;
  METAL_FUNC QnRowLoader(const device T* src_, const int ld_,
                         const device int* srow, const int base,
                         threadgroup float* dst_, ushort sg, ushort lane)
      : bi((short)((sg * 32 + lane) / 4)),
        bj((short)(8 * ((sg * 32 + lane) % 4))),
        dst(dst_ + bi * 36 + bj), src(src_), ld(ld_), k_off(0) {
    r = srow != nullptr ? srow[base + bi] : base + bi;
  }
  METAL_FUNC void load() const {
    if (r < 0) {
      for (short j = 0; j < 8; ++j) { dst[j] = 0.0f; }
      return;
    }
    const device T* p = src + (long)r * ld + k_off + bj;
    for (short j = 0; j < 8; ++j) { dst[j] = (float)p[j]; }
  }
  METAL_FUNC void next() { k_off += 32; }
};

using qn_mma_t = mlx::steel::BlockMMA<float, float, 32, 32, 32, 2, 2, false,
                                      true, 36, 36, float>;

// gact[row, I] = silu(gate . x[srow]) * (up . x[srow]) over the tile's
// slot. threadgroups {I / 32, tiles}, tg {32, 2, 2}.
template <bool W8>
kernel void qwen_native_grouped_swiglu_(
    const device uchar* wg [[buffer(0)]],
    const device uchar* wu [[buffer(1)]],
    const device ulong* tab [[buffer(2)]],
    const device T* x [[buffer(3)]],         // [rows, H]
    device T* gact [[buffer(4)]],            // [npad, I]
    const device int* srow [[buffer(5)]],
    const device int* t2e [[buffer(6)]],
    constant int& H [[buffer(7)]], constant int& I [[buffer(8)]],
    constant uint& tg_ [[buffer(9)]], constant uint& tu [[buffer(10)]],
    constant long& cwg [[buffer(11)]], constant long& cwu [[buffer(12)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
  threadgroup float Xs[32 * 36], Gs[32 * 36], Us[32 * 36];
  const int e = t2e[tid.y];
  if (e < 0) { return; }
  const int y_row = (int)tid.y * 32, y_col = (int)tid.x * 32;
  QnRowLoader lx(x, H, srow, y_row, Xs, sg, lane);
  QnLoader<W8> lg(wg + (long)e * cwg, H, tg_, tab, y_col, Gs, sg, lane);
  QnLoader<W8> lu(wu + (long)e * cwu, H, tu, tab, y_col, Us, sg, lane);
  qn_mma_t mg(sg, lane), mu(sg, lane);
  for (int k = 0; k < H; k += 32) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    lx.load();
    lg.load();
    lu.load();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mg.mma(Xs, Gs);
    mu.mma(Xs, Us);
    lx.next();
    lg.next();
    lu.next();
  }
  for (short ti = 0; ti < 2; ++ti) {
    const int row = y_row + mg.sm + ti * 16;
    for (short tj = 0; tj < 2; ++tj) {
      const int col = y_col + mg.sn + tj * 16;
      const thread auto& fg = mg.Ctile.frag_at(ti, tj);
      const thread auto& fu = mu.Ctile.frag_at(ti, tj);
      for (short c = 0; c < 2; ++c) {
        const float gg = fg[c], uu = fu[c];
        gact[(long)row * I + col + c] = (T)((gg / (1.0f + exp(-gg))) * uu);
      }
    }
  }
}

// part[sdst[row], H] = down . gact[row] over the tile's slot.
// threadgroups {H / 32, tiles}, tg {32, 2, 2}.
template <bool W8>
kernel void qwen_native_grouped_down_(
    const device uchar* wd [[buffer(0)]],
    const device ulong* tab [[buffer(1)]],
    const device T* gact [[buffer(2)]],      // [npad, I]
    device T* part [[buffer(3)]],            // [np, H]
    const device int* sdst [[buffer(4)]],
    const device int* t2e [[buffer(5)]],
    constant int& I [[buffer(6)]], constant int& H [[buffer(7)]],
    constant uint& td [[buffer(8)]], constant long& cwd [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
  threadgroup float Xs[32 * 36], Ws[32 * 36];
  const int e = t2e[tid.y];
  if (e < 0) { return; }
  const int y_row = (int)tid.y * 32, y_col = (int)tid.x * 32;
  QnRowLoader lx(gact, I, nullptr, y_row, Xs, sg, lane);
  QnLoader<W8> lw(wd + (long)e * cwd, I, td, tab, y_col, Ws, sg, lane);
  qn_mma_t mm(sg, lane);
  for (int k = 0; k < I; k += 32) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    lx.load();
    lw.load();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mm.mma(Xs, Ws);
    lx.next();
    lw.next();
  }
  for (short ti = 0; ti < 2; ++ti) {
    const int row = y_row + mm.sm + ti * 16;
    const int dr = sdst[row];
    if (dr < 0) { continue; }
    for (short tj = 0; tj < 2; ++tj) {
      const int col = y_col + mm.sn + tj * 16;
      const thread auto& f = mm.Ctile.frag_at(ti, tj);
      part[(long)dr * H + col] = (T)f[0];
      part[(long)dr * H + col + 1] = (T)f[1];
    }
  }
}

#define QN_GROUPED(SUF, W)                                                   \
  template [[host_name("qwen_native_grouped_swiglu" SUF)]] kernel void       \
  qwen_native_grouped_swiglu_<W>(                                            \
      const device uchar*, const device uchar*, const device ulong*,         \
      const device T*, device T*, const device int*, const device int*,      \
      constant int&, constant int&, constant uint&, constant uint&,          \
      constant long&, constant long&, uint3, uint, uint);                    \
  template [[host_name("qwen_native_grouped_down" SUF)]] kernel void         \
  qwen_native_grouped_down_<W>(                                              \
      const device uchar*, const device ulong*, const device T*, device T*,  \
      const device int*, const device int*, constant int&, constant int&,    \
      constant uint&, constant long&, uint3, uint, uint);
QN_GROUPED("", true)
QN_GROUPED("_w32", false)   // the bench's reference only

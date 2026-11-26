// magi2_nax.metal -- MAGI-2-preview's ROUTED EXPERTS on the M5's matrix
// cores (NAX): the twin of magi2.metal's magi2_moe_expert, same buffers,
// same sorted-tile contract, same rounding points.
//
// A WORK ITEM (moe_tile_items_i32, two of the sort's 32-row tiles at a
// time, or an expert's odd last one) is up to 64 sorted slots of ONE expert,
// their rows of 256 gathered from X through srow, through
//   g = x Wg[e], u = x Wu[e]         ([256, I] each, f32 accumulation)
//   h = swiglu7(g, u)                (rounded to the element type)
//   y += h Wd[e]                     ([I, 256], f32 accumulation)
// over the expert's width in chunks of DE = 64, h never leaving the
// threadgroup; y times the pair's prob, written f32 [npad, 256] at its
// slot (pads, srow / sdst < 0, are neither read nor written).
//
// 8 simdgroups: WM = 4 down the item's rows (16 each), WN = 2 across. For a
// chunk, simdgroup (wm, wn) computes g and u for its 16 rows x 32 of the
// chunk's 64 columns (K = 256, A straight from X at the gathered rows, B
// straight from the weights), applies swiglu7 into the threadgroup's
// [64, DE] h; then, h whole, accumulates its 16 rows x 128 columns of y
// (K = DE, A from h, B from Wd). The fragments are NAX 16 x 16 tiles
// (nax.h's BaseNAXFrag) through matmul2d 16 x 32 x 16 -- a simdgroup's two
// adjacent column fragments a call. A simdgroup whose first row is a pad
// (a segment fills its slots from the first) or past the item computes
// nothing but keeps the barriers.
//
// Weights as the release stores them: Wg / Wu [E, 256, I], Wd [E, I, 256],
// row-major. Requires I % 64 == 0 (the host keeps the simdgroup kernel
// otherwise). Built only for the tensor-capable target (-std=metal4.0).
//   0:X 1:srow 2:sdst 3:prob 4:tile2e 5:Wg 6:Wu 7:Wd 8:Y 9:I 10:items
//   grid (256 * maxitems) threads, threadgroup 256

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif
typedef VPIPE_ELT elt;

#if defined(__HAVE_TENSOR__)

#include "mlx/backend/metal/kernels/steel/attn/nax.h"

namespace {

using mlx::steel::BaseNAXFrag;
using mlx::steel::Int;

constexpr constant int kTm = 32;     // the sort's tile (kMaxm)
constexpr constant int kDh = 256;    // an expert head's width
constexpr constant int kDe = 64;     // the hidden chunk
constexpr constant int kWn = 2;

// magi2.metal's swiglu7: gate clamped above at 7, up to [-7, 7].
inline float
swiglu7_(float g, float u)
{
  g = min(g, 7.0f);
  u = clamp(u, -7.0f, 7.0f);
  return g / (1.0f + exp(-1.702f * g)) * (u + 1.0f);
}

}  // namespace

kernel void magi2_moe_expert_nax(
    const device elt*   X      [[buffer(0)]],
    const device int*   srow   [[buffer(1)]],
    const device int*   sdst   [[buffer(2)]],
    const device float* prob   [[buffer(3)]],
    const device int*   tile2e [[buffer(4)]],
    const device elt*   Wg     [[buffer(5)]],
    const device elt*   Wu     [[buffer(6)]],
    const device elt*   Wd     [[buffer(7)]],
    device float*       Y      [[buffer(8)]],
    constant int&       I      [[buffer(9)]],
    const device int*   items  [[buffer(10)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]])
{
  using Frag = BaseNAXFrag;
  using fe = Frag::dtype_frag_t<elt>;
  using ff = Frag::dtype_frag_t<float>;
  const int it = items[tg];
  if (it < 0) { return; }                  // the whole threadgroup alike
  const int t = it >> 2, rows_in = ((it & 3) + 1) * kTm;
  const int e = tile2e[t];
  const int y_row = t * kTm;
  const short wm = (short)(sgid / kWn), wn = (short)(sgid % kWn);
  const int r16 = 16 * wm;                 // this simdgroup's rows
  const bool act = r16 < rows_in && srow[y_row + r16] >= 0;
  const short2 sc = Frag::get_coord();     // (fn, fm)
  // This thread's two rows of the item (fm, fm + 8) and their sources.
  const int s0 = y_row + r16 + sc.y, s1 = s0 + 8;
  const int xr0 = act ? srow[s0] : -1, xr1 = act ? srow[s1] : -1;
  const device elt* x0 = X + (long)max(xr0, 0) * kDh + sc.x;
  const device elt* x1 = X + (long)max(xr1, 0) * kDh + sc.x;
  const device elt* wg = Wg + (long)e * kDh * I;
  const device elt* wu = Wu + (long)e * kDh * I;
  const device elt* wd = Wd + (long)e * I * kDh;

  threadgroup elt Hs[64 * kDe];

  ff y[8];
  STEEL_PRAGMA_UNROLL
  for (short j = 0; j < 8; ++j) { y[j] = ff(0.0f); }

  for (int c0 = 0; c0 < I; c0 += kDe) {
    if (act) {
      ff g0 = ff(0.0f), g1 = ff(0.0f), u0 = ff(0.0f), u1 = ff(0.0f);
      const int nc = c0 + 32 * wn;         // this simdgroup's 32 columns
      STEEL_PRAGMA_NO_UNROLL
      for (int k = 0; k < kDh; k += 16) {
        fe a;
        const vec<elt, 4> a0 = xr0 >= 0 ? *(const device vec<elt, 4>*)(x0 + k)
                                        : vec<elt, 4>(0);
        const vec<elt, 4> a1 = xr1 >= 0 ? *(const device vec<elt, 4>*)(x1 + k)
                                        : vec<elt, 4>(0);
        STEEL_PRAGMA_UNROLL
        for (short c = 0; c < 4; ++c) {
          a[c] = a0[c];
          a[4 + c] = a1[c];
        }
        fe bg0, bg1, bu0, bu1;
        const long wo = (long)k * I + nc;
        Frag::load(bg0, wg + wo, I, Int<1>{});
        Frag::load(bg1, wg + wo + 16, I, Int<1>{});
        Frag::load(bu0, wu + wo, I, Int<1>{});
        Frag::load(bu1, wu + wo + 16, I, Int<1>{});
        Frag::template mma<float, elt, elt>(g0, g1, a, metal::false_type{},
                                            bg0, bg1, metal::false_type{});
        Frag::template mma<float, elt, elt>(u0, u1, a, metal::false_type{},
                                            bu0, bu1, metal::false_type{});
      }
      // swiglu7 into h: element i * 4 + c of a fragment sits at row
      // fm + i * 8, column fn + c.
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < 2; ++i) {
        threadgroup elt* hr =
            Hs + (r16 + sc.y + i * 8) * kDe + 32 * wn + sc.x;
        STEEL_PRAGMA_UNROLL
        for (short c = 0; c < 4; ++c) {
          hr[c] = (elt)swiglu7_(g0[i * 4 + c], u0[i * 4 + c]);
          hr[16 + c] = (elt)swiglu7_(g1[i * 4 + c], u1[i * 4 + c]);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (act) {
      STEEL_PRAGMA_NO_UNROLL
      for (int k = 0; k < kDe; k += 16) {
        fe a;
        Frag::load(a, (const threadgroup elt*)Hs + r16 * kDe + k, kDe,
                   Int<1>{});
        const device elt* wk = wd + (long)(c0 + k) * kDh + 128 * wn;
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < 8; j += 2) {
          fe b0, b1;
          Frag::load(b0, wk + 16 * j, kDh, Int<1>{});
          Frag::load(b1, wk + 16 * j + 16, kDh, Int<1>{});
          Frag::template mma<float, elt, elt>(y[j], y[j + 1], a,
                                              metal::false_type{}, b0, b1,
                                              metal::false_type{});
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!act) { return; }

  // Out: y times the pair's prob at the slot's row, f32.
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < 2; ++i) {
    const int slot = i == 0 ? s0 : s1;
    const int p = sdst[slot];
    if (p < 0) { continue; }
    const float pr = prob[p];
    device float* o = Y + (long)slot * kDh + 128 * wn + sc.x;
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < 8; ++j) {
      STEEL_PRAGMA_UNROLL
      for (short c = 0; c < 4; ++c) {
        o[16 * j + c] = y[j][i * 4 + c] * pr;
      }
    }
  }
}

#else

// Tensor ops unavailable for this target: a stub so the metallib builds.
// The loader binds it only on a GPU with matrix cores.
kernel void magi2_moe_expert_nax(device float* Y [[buffer(8)]],
                                 uint tid [[thread_position_in_grid]])
{
  (void)Y;
  (void)tid;
}

#endif

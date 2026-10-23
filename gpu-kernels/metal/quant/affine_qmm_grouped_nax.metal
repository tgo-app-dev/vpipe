// affine_qmm_grouped_nax.metal -- the MoE grouped expert GEMMs on the M5
// matrix cores (NAX): the matrix-core twins of affine_qmm_steel's
// affine_qmm_grouped_swiglu_w{4,5,8}g64 / affine_qmm_grouped_w{4,5,8}g64,
// with the SAME buffers, constants and sorted-tile contract:
//
//   0:w(uint32) 1:scales 2:biases 3:x 4:y 5:K 6:N 7:M(= npad)
//   8:tile2e   9:srow (gate|up: sorted slot -> hidden row, < 0 a pad)
//                or sdst (down: sorted slot -> pair row, < 0 a pad)
//
// The (token, expert) pairs are counting-sorted into per-expert segments
// padded to whole BM-row tiles; tile tid.y belongs to ONE expert
// (tile2e[tid.y], < 0 an unused tile) and reads that expert's slab. The
// gate|up GEMM gathers its rows from the hidden through srow and stores
// silu(gate) * up (gate|up interleaved by feature: row 2g gate g, 2g+1 up
// g); the down GEMM reads the sorted activations and scatters its rows to
// the pairs' partials through sdst.
//
// The shape is MLX's affine_gather_qmm_rhs_nax (quantized_nax.h, MIT): a
// BN x BK weight tile dequantized into threadgroup memory by the shared
// QuantizedBlockLoader, 2 x 2 simdgroups each SM x SN of the output, the
// K step SK = 32 run as NAXTile fragments through tile_matmad_nax (f32
// accumulators). What differs is the activation side: MLX gathers x
// before the GEMM; here each thread reads its two fragment rows straight
// from device memory at the rows srow names (the steel twin's gather,
// fused, with no threadgroup staging), and pads read zeros.
//
// Requires N % BN == 0 and K % BK == 0 (the host routes other shapes to
// the steel twin) and group 64. Built only for the tensor-capable target
// (-std=metal4.0); the loader binds it only on a GPU with matrix cores.
// dispatch (threads): {N/BN * 32, npad/BM * 2, 2}, tg {32, 2, 2}.

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

#ifndef VPIPE_ELT
#define VPIPE_ELT half          // unsuffixed = f16; _bf16 build sets bfloat
#endif

#if defined(__HAVE_TENSOR__)

#include "mlx/backend/metal/kernels/steel/attn/nax.h"
#include "quant/affine_qmm_loaders.h"

namespace {

using mlx::steel::BaseNAXFrag;
using mlx::steel::NAXTile;
using mlx::steel::tile_matmad_nax;

constexpr constant int kBM = 32;     // = the sort's segment padding (MAXM)
constexpr constant int kBN = 64;
constexpr constant int kBK = 64;     // one affine group per K step
constexpr constant int kWM = 2;
constexpr constant int kWN = 2;

template <typename T, int group_size, int bits, bool kSwiglu>
METAL_FUNC void
grouped_nax_impl(const device uint32_t* w, const device T* scales,
                 const device T* biases, const device T* x, device T* y,
                 threadgroup T* Ws, const int K, const int N,
                 const device int* tile2e, const device int* rows,
                 uint3 tid, uint sgid, uint slid)
{
  constexpr int BM = kBM, BN = kBN, BK = kBK, WM = kWM, WN = kWN;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = BK + 16 / sizeof(T);
  constexpr short SM = BM / WM, SN = BN / WN, SK = 32;
  constexpr short TM = SM / 16, TN = SN / 16, TK = SK / 16;
  static_assert(TM == 1 && TN % 2 == 0, "tile_matmad_nax wants even TN");

  const int e = tile2e[tid.y];
  if (e < 0) { return; }
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = (int)tid.y * BM;
  const int y_col = (int)tid.x * BN;
  const int64_t wrow0 = (int64_t)e * N + y_col;   // the tile's first row
  auto wl = (const device uint8_t*)w + wrow0 * K_w;
  scales += wrow0 * K_g;
  biases += wrow0 * K_g;

  using loader_w_t = QuantizedBlockLoader<
      T, BN, BK, BK_padded, 1, WM * WN * 32, group_size, bits>;
  loader_w_t loader_w(wl, scales, biases, K, Ws, (ushort)sgid, (ushort)slid);

  const short tm = SM * (short)(sgid / WN);
  const short tn = SN * (short)(sgid % WN);
  // This thread's two fragment rows (BaseNAXFrag: rows fm and fm + 8,
  // columns fn .. fn + 3 of each 16-wide fragment) and where they come
  // from: a hidden row through srow (gate|up) or the slot itself (down).
  const short2 sc = BaseNAXFrag::get_coord();
  const int s0 = y_row + tm + sc.y, s1 = s0 + 8;
  const int r0 = kSwiglu ? rows[s0] : s0;
  const int r1 = kSwiglu ? rows[s1] : s1;
  const device T* x0 = x + (int64_t)max(r0, 0) * K + sc.x;
  const device T* x1 = x + (int64_t)max(r1, 0) * K + sc.x;
  const bool on0 = r0 >= 0, on1 = r1 >= 0;

  NAXTile<float, TM, TN> Dt;
  Dt.clear();
  for (int k0 = 0; k0 < K; k0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_w.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_NO_UNROLL
    for (short kk = 0; kk < BK; kk += SK) {
      NAXTile<T, TM, TK> At;
      NAXTile<T, TN, TK> Bt;
      volatile int compiler_barrier;
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TK; ++j) {
        thread auto& f = At.frag_at(0, j);
        const int k = k0 + kk + j * 16;
        const vec<T, 4> a0 = on0 ? *(const device vec<T, 4>*)(x0 + k)
                                 : vec<T, 4>(0);
        const vec<T, 4> a1 = on1 ? *(const device vec<T, 4>*)(x1 + k)
                                 : vec<T, 4>(0);
        STEEL_PRAGMA_UNROLL
        for (short c = 0; c < 4; ++c) {
          f[c] = a0[c];
          f[4 + c] = a1[c];
        }
      }
      Bt.template load<T, BK_padded, 1>(Ws + tn * BK_padded + kk);
      tile_matmad_nax(Dt, At, metal::bool_constant<false>{}, Bt,
                      metal::bool_constant<true>{});
      (void)compiler_barrier;
    }
    loader_w.next();
  }

  // Store: fragment element i sits at row fm + (i / 4) * 8, column
  // fn + i % 4 of its fragment, so a thread holds whole (gate, up) pairs.
  STEEL_PRAGMA_UNROLL
  for (short j = 0; j < TN; ++j) {
    const thread auto& f = Dt.frag_at(0, j);
    const int col = y_col + tn + j * 16 + sc.x;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; ++i) {
      const int slot = i == 0 ? s0 : s1;
      if (kSwiglu) {
        const int outN = N / 2;
        device T* o = y + (int64_t)slot * outN + (col >> 1);
        STEEL_PRAGMA_UNROLL
        for (short c = 0; c < 4; c += 2) {
          const float ga = f[i * 4 + c], up = f[i * 4 + c + 1];
          o[c >> 1] = (T)(ga / (1.0f + metal::exp(-ga)) * up);
        }
      } else {
        const int dr = rows[slot];        // pair-ordered destination
        if (dr < 0) { continue; }
        device T* o = y + (int64_t)dr * N + col;
        STEEL_PRAGMA_UNROLL
        for (short c = 0; c < 4; ++c) { o[c] = (T)f[i * 4 + c]; }
      }
    }
  }
}

}  // namespace

#define VPIPE_GROUPED_NAX(name, bits, swiglu)                               \
  kernel void name(                                                         \
      const device uint32_t* w       [[buffer(0)]],                         \
      const device VPIPE_ELT* scales [[buffer(1)]],                         \
      const device VPIPE_ELT* biases [[buffer(2)]],                         \
      const device VPIPE_ELT* x      [[buffer(3)]],                         \
      device VPIPE_ELT*       y      [[buffer(4)]],                         \
      const constant int& K          [[buffer(5)]],                         \
      const constant int& N          [[buffer(6)]],                         \
      const constant int& M          [[buffer(7)]],                         \
      const device int* tile2e       [[buffer(8)]],                         \
      const device int* rows         [[buffer(9)]],                         \
      uint3 tid  [[threadgroup_position_in_grid]],                          \
      uint  sgid [[simdgroup_index_in_threadgroup]],                        \
      uint  slid [[thread_index_in_simdgroup]])                             \
  {                                                                         \
    (void)M;                                                                \
    threadgroup VPIPE_ELT Ws[kBN * (kBK + 16 / sizeof(VPIPE_ELT))];         \
    grouped_nax_impl<VPIPE_ELT, 64, bits, swiglu>(                          \
        w, scales, biases, x, y, Ws, K, N, tile2e, rows, tid, sgid, slid);  \
  }

VPIPE_GROUPED_NAX(affine_qmm_grouped_swiglu_nax_w4g64, 4, true)
VPIPE_GROUPED_NAX(affine_qmm_grouped_swiglu_nax_w5g64, 5, true)
VPIPE_GROUPED_NAX(affine_qmm_grouped_swiglu_nax_w8g64, 8, true)
VPIPE_GROUPED_NAX(affine_qmm_grouped_nax_w4g64, 4, false)
VPIPE_GROUPED_NAX(affine_qmm_grouped_nax_w5g64, 5, false)
VPIPE_GROUPED_NAX(affine_qmm_grouped_nax_w8g64, 8, false)
// 6-bit experts (the oQ6e packs); group 32 has no twin here (kBK 64 is one
// group a K step), so a group-32 pack takes the steel GEMMs.
VPIPE_GROUPED_NAX(affine_qmm_grouped_swiglu_nax_w6g64, 6, true)
VPIPE_GROUPED_NAX(affine_qmm_grouped_nax_w6g64, 6, false)

#endif  // __HAVE_TENSOR__

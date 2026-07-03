// affine_qmm_steel.metal -- 4-bit affine quantized GEMM built on
// MLX's vendored steel BlockMMA + BlockLoader machinery, to reach
// MLX-parity prefill throughput. Our hand-rolled affine_qmm plateaued
// at ~1.17x MLX because the remaining gap is steel's hand-tuned
// register-tiled MMA / load scheduling at an identical tile shape;
// this kernel uses that machinery directly.
//
// Vendored under msl/mlx/backend/metal/kernels/steel/ (resolved via
// the kernel build's -I msl): defines.h, utils.h, gemm/transforms.h,
// gemm/mma.h, gemm/loader.h, utils/{type_traits,integral_constant}.h.
// The quantized pieces (dequantize, QuantizedBlockLoader, qmm_t_impl)
// are copied verbatim from MLX's quantized.h so the math + packed
// layout match exactly.
//
//   T = half     bits = 4     group_size = 64     tile 32x32, WM=WN=2
//
// Non-batched only. Launch (dispatch_threadgroups-equivalent under
// metal-compute's dispatchThreads): threadgroup (32, WN, WM), grid
// threadgroups (ceil(N/BN), ceil(M/BM), 1). M/N tails handled inside
// qmm_t_impl, so y needs no padding.
//
//   0:w(uint32) 1:scales(VPIPE_ELT) 2:biases(VPIPE_ELT) 3:x(VPIPE_ELT) 4:y(VPIPE_ELT)
//   5:K 6:N 7:M

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

#ifndef METAL_FUNC
#define METAL_FUNC inline
#endif
#define MLX_MTL_CONST static constant constexpr const
#define SIMD_SIZE 32

using namespace metal;

#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif

// steel mma.h has a complex64_t BlockMMA specialization (unused here
// but must parse), so complex64_t needs to be in scope first. MLX's
// complex.h references their bfloat16_t wrapper for an unused ctor;
// alias it to the native type (we never touch bf16 or complex).
typedef bfloat16 bfloat16_t;
#include "mlx/backend/metal/kernels/complex.h"
#include "mlx/backend/metal/kernels/steel/gemm/mma.h"
#include "mlx/backend/metal/kernels/steel/gemm/loader.h"

// ===================================================================
// Quantized helpers -- verbatim from MLX quantized.h (affine path).
// ===================================================================

template <int bits, int wsize = 8>
inline constexpr short get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}

template <int bits, int wsize = 8>
inline constexpr short get_bytes_per_pack() {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}

// `Compute` is the arithmetic type for the dequant scale*w+bias. It defaults
// to the storage type U (so the half path is byte-identical to MLX), but the
// bf16 loader passes Compute=float: Apple GPUs emulate scalar bfloat math by
// widening to fp32 per op, so dequantizing in bfloat is markedly slower than
// computing in fp32 and rounding to bfloat once on store.
template <typename U, int N, int bits, typename Compute = U>
inline void
dequantize(const device uint8_t* w, U scale, U bias, threadgroup U* w_local) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  if (bits == 4) {
    const Compute cs = static_cast<Compute>(scale);
    const Compute cb = static_cast<Compute>(bias);
    const Compute s[2] = {cs, cs / static_cast<Compute>(16.0f)};
    for (int i = 0; i < (N / 2); i++) {
      w_local[2 * i] =
          static_cast<U>(s[0] * static_cast<Compute>(w[i] & 0x0f) + cb);
      w_local[2 * i + 1] =
          static_cast<U>(s[1] * static_cast<Compute>(w[i] & 0xf0) + cb);
    }
  }

  else if (bits == 2) {
    U s[4] = {
        scale,
        scale / static_cast<U>(4.0f),
        scale / static_cast<U>(16.0f),
        scale / static_cast<U>(64.0f)};
    for (int i = 0; i < (N / 4); i++) {
      w_local[4 * i] = s[0] * (w[i] & 0x03) + bias;
      w_local[4 * i + 1] = s[1] * (w[i] & 0x0c) + bias;
      w_local[4 * i + 2] = s[2] * (w[i] & 0x30) + bias;
      w_local[4 * i + 3] = s[3] * (w[i] & 0xc0) + bias;
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (N / 8); i++) {
      w_local += 8 * i;
      w += 3 * i;

      w_local[0] = (w[0] & 0x7) * scale + bias;
      w_local[1] = ((w[0] & 0x38) >> 3) * scale + bias;
      w_local[2] = (((w[0] & 0xc0) >> 6) + ((w[1] & 0x1) << 2)) * scale + bias;
      w_local[3] = ((w[1] & 0xe) >> 1) * scale + bias;
      w_local[4] = ((w[1] & 0x70) >> 4) * scale + bias;
      w_local[5] = (((w[1] & 0x80) >> 7) + ((w[2] & 0x3) << 1)) * scale + bias;
      w_local[6] = ((w[2] & 0x1c) >> 2) * scale + bias;
      w_local[7] = ((w[2] & 0xe0) >> 5) * scale + bias;
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (N / 8); i++) {
      w_local += 8 * i;
      w += 5 * i;

      w_local[0] = (w[0] & 0x1f) * scale + bias;
      w_local[1] = (((w[0] & 0xe0) >> 5) + ((w[1] & 0x3) << 3)) * scale + bias;
      w_local[2] = ((w[1] & 0x7c) >> 2) * scale + bias;
      w_local[3] = (((w[1] & 0x80) >> 7) + ((w[2] & 0xf) << 1)) * scale + bias;
      w_local[4] = (((w[2] & 0xf0) >> 4) + ((w[3] & 0x1) << 4)) * scale + bias;
      w_local[5] = ((w[3] & 0x3e) >> 1) * scale + bias;
      w_local[6] = (((w[3] & 0xc0) >> 6) + ((w[4] & 0x7) << 2)) * scale + bias;
      w_local[7] = ((w[4] & 0xf8) >> 3) * scale + bias;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (N / 4); i++) {
      w_local += 4 * i;
      w += 3 * i;
      w_local[0] = (w[0] & 0x3f) * scale + bias;
      w_local[1] = (((w[0] >> 6) & 0x03) + ((w[1] & 0x0f) << 2)) * scale + bias;
      w_local[2] = (((w[1] >> 4) & 0x0f) + ((w[2] & 0x03) << 4)) * scale + bias;
      w_local[3] = ((w[2] >> 2) & 0x3f) * scale + bias;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      w_local[i] = scale * w[i] + bias;
    }
  }
}

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short group_size,
    short bits>
struct QuantizedBlockLoader {
  static_assert(
      BCOLS <= group_size,
      "The group size should be larger than the columns");
  static_assert(
      group_size % BCOLS == 0,
      "The group size should be divisible by the columns");
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  MLX_MTL_CONST short pack_factor = get_pack_factor<bits, 8>();
  MLX_MTL_CONST short bytes_per_pack = get_bytes_per_pack<bits>();
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  // Dequant arithmetic type: fp32 for bf16 storage (scalar bf16 math is
  // emulated on Apple GPUs), native otherwise. Half path stays byte-identical.
  typedef typename metal::conditional<
      metal::is_same<T, bfloat>::value, float, T>::type DeqCompute;
  MLX_MTL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  MLX_MTL_CONST short group_steps = group_size / BCOLS;

  const int src_ld;
  const int tile_stride;
  short group_step_cnt;
  const int group_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;
  const device T* scales;
  const device T* biases;

  QuantizedBlockLoader(
      const device uint8_t* src_,
      const device T* scales_,
      const device T* biases_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        tile_stride(
            reduction_dim ? BCOLS_PACKED * bytes_per_pack
                          : BROWS * src_ld * bytes_per_pack / pack_factor),
        group_step_cnt(0),
        group_stride(BROWS * src_ld / group_size),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads * thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor +
            bj * bytes_per_pack),
        scales(scales_ + bi * src_ld / group_size),
        biases(biases_ + bi * src_ld / group_size) {}

  void load_unsafe() const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    T scale = *scales;
    T bias = *biases;
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits, DeqCompute>(
          src + i * bytes_per_pack, scale, bias, dst + i * pack_factor);
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    if (reduction_dim == 1 && bi >= src_tile_dim.x) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    if (reduction_dim == 0 && bi >= src_tile_dim.y) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    T scale = *scales;
    T bias = *biases;
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits, DeqCompute>(
          (device uint8_t*)(src + i * bytes_per_pack),
          scale,
          bias,
          dst + i * pack_factor);
    }
  }

  void next() {
    src += tile_stride;
    if (reduction_dim == 1) {
      if (group_steps > 1) {
        group_step_cnt++;
        if (group_step_cnt == group_steps) {
          group_step_cnt = 0;
          scales++;
          biases++;
        }
      } else {
        scales++;
        biases++;
      }
    } else {
      scales += group_stride;
      biases += group_stride;
    }
  }
};

// ===================================================================
// qmm_t_impl -- verbatim from MLX quantized.h.
// ===================================================================

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
METAL_FUNC void qmm_t_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& K_eff,
    const device int* tile2e,   // MoE grouped: per-tile expert (or nullptr)
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;

  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  // Instantiate the appropriate BlockMMA and Loader
  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      1,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  // Set the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;

  // MoE grouped GEMM: each BM-row M-tile is one expert's (sorted, padded)
  // rows -> offset the weight slab to that expert; skip empty padding tiles.
  if (tile2e) {
    const int e = tile2e[tid.y];
    if (e < 0) { return; }
    wl += static_cast<int64_t>(e) * N * K_w;
    scales += static_cast<int64_t>(e) * N * K_g;
    biases += static_cast<int64_t>(e) * N * K_g;
  }

  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  y += y_row * static_cast<int64_t>(N) + y_col;

  // Make the x loader and mma operation
  const short num_els = min(BM, M - y_row);
  const short num_outs = min(BN, N - y_col);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  if (num_els < BM) {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  } else {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);

        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  }

  // Store results to device memory
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM || num_outs < BN) {
    mma_op.store_result_safe(y, N, short2(num_outs, num_els));
  } else {
    mma_op.store_result(y, N);
  }
}

// ===================================================================
// Kernel entry: non-batched, half / 4-bit / group 64, 32x32 tile.
// ===================================================================

kernel void affine_qmm_steel_w4g64(
    const device uint32_t* w      [[buffer(0)]],
    const device VPIPE_ELT*     scales [[buffer(1)]],
    const device VPIPE_ELT*     biases [[buffer(2)]],
    const device VPIPE_ELT*     x      [[buffer(3)]],
    device VPIPE_ELT*           y      [[buffer(4)]],
    const constant int&    K      [[buffer(5)]],
    const constant int&    N      [[buffer(6)]],
    const constant int&    M      [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  lid      [[thread_index_in_threadgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_impl<VPIPE_ELT, 64, 4, /*aligned_N=*/true, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, K,
      /*tile2e=*/nullptr, tid, lid, simd_gid, simd_lid);
}

// group_size=32 twin of affine_qmm_steel_w4g64 (GGUF q4_0). Only the
// template group_size differs.
kernel void affine_qmm_steel_w4g32(
    const device uint32_t* w      [[buffer(0)]],
    const device VPIPE_ELT*     scales [[buffer(1)]],
    const device VPIPE_ELT*     biases [[buffer(2)]],
    const device VPIPE_ELT*     x      [[buffer(3)]],
    device VPIPE_ELT*           y      [[buffer(4)]],
    const constant int&    K      [[buffer(5)]],
    const constant int&    N      [[buffer(6)]],
    const constant int&    M      [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  lid      [[thread_index_in_threadgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_impl<VPIPE_ELT, 32, 4, /*aligned_N=*/true, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, K,
      /*tile2e=*/nullptr, tid, lid, simd_gid, simd_lid);
}

// ===================================================================
// Fused SwiGLU MLP GEMM: y = silu(gate) * up in one pass.
//
// The gate and up projection weights are pre-INTERLEAVED by output
// feature into a single [N=2*ffn, K] quantized matrix: fused row 2g is
// gate feature g, row 2g+1 is up feature g. So the GEMM result column
// 2g is x.gate_g and column 2g+1 is x.up_g. With BN even and the 8x8
// simdgroup-matrix frag layout, those two adjacent columns are exactly
// the two accumulator elements THIS thread already holds per frag (the
// frag's column base fn is always even -> elem 0 = gate, elem 1 = up),
// so the activation+multiply is purely register-local -- no cross-lane
// shuffle, and the written output is HALVED to [M, ffn] (no separate
// gate/up buffers, no separate swiglu pass).
//
//   0:w(uint32) 1:scales 2:biases 3:x 4:y([M, N/2]) 5:K 6:N(=2*ffn) 7:M
//
// qmm_t_swiglu_impl mirrors qmm_t_impl's MMA loop verbatim; only the
// epilogue (store) differs. N must be even and the matmul N tail is
// assumed aligned (2*ffn % BN == 0), so only the M tail needs guarding.
template <
    typename T,
    const int group_size,
    const int bits,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
METAL_FUNC void qmm_t_swiglu_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, bits>;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;
  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;

  const short num_els = min(BM, M - y_row);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  if (num_els < BM) {
    for (int k = 0; k < K; k += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(BK, num_els));
      loader_w.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
      loader_x.next();
      loader_w.next();
    }
  } else {
    for (int k = 0; k < K; k += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_unsafe();
      loader_w.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
      loader_x.next();
      loader_w.next();
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Fused SwiGLU store. Each accumulator frag holds, for this thread, a
  // (gate, up) pair in elems [0] (even column) and [1] (odd column);
  // emit silu(gate)*up to y[row, col/2]. Output width is N/2.
  constexpr int FS = 8;
  constexpr int TM = BM / (FS * WM);   // simdgroup tiles along M (2)
  constexpr int TN = BN / (FS * WN);   // simdgroup tiles along N (2)
  const int outN = N / 2;
  const short sm = mma_op.sm;
  const short sn = mma_op.sn;
  for (short ti = 0; ti < TM; ti++) {
    const int row = y_row + sm + ti * FS * WM;
    if (row >= M) { continue; }
    for (short tj = 0; tj < TN; tj++) {
      const int col = y_col + sn + tj * FS * WN;   // even fused column
      const thread auto& fr = mma_op.Ctile.frag_at(ti, tj);
      const float ga = (float)fr[0];               // gate (even col)
      const float up = (float)fr[1];               // up   (odd col)
      const float s = ga / (1.0f + metal::exp(-ga));
      y[(int64_t)row * outN + (col >> 1)] = (T)(s * up);
    }
  }
}

kernel void affine_qmm_swiglu_w4g64(
    const device uint32_t* w      [[buffer(0)]],
    const device VPIPE_ELT*     scales [[buffer(1)]],
    const device VPIPE_ELT*     biases [[buffer(2)]],
    const device VPIPE_ELT*     x      [[buffer(3)]],
    device VPIPE_ELT*           y      [[buffer(4)]],
    const constant int&    K      [[buffer(5)]],
    const constant int&    N      [[buffer(6)]],   // fused width = 2*ffn
    const constant int&    M      [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_swiglu_impl<VPIPE_ELT, 64, 4, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M,
      tid, simd_gid, simd_lid);
}

// Row-gathering block loader: a drop-in for mlx::steel::BlockLoader whose BM
// rows are GATHERED from a base matrix via srow[] (row r of the tile reads
// src[srow[base_slot + r], :]; srow < 0 -> zero-filled padding row). Lets the
// grouped gate|up GEMM read its sorted rows straight from the hidden, fusing
// away the explicit moe_gather_x copy (the ablation's #1 prefill overhead).
// K is a multiple of BK for these GEMMs so only load (column-aligned) is
// needed; load_safe falls back to it (padding rows masked by srow).
template <typename T, short BROWS, short BCOLS, short dst_ld, short tgp_size,
          short n_reads = (BCOLS * BROWS) / tgp_size,
          short TCOLS = BCOLS / n_reads, short TROWS = tgp_size / TCOLS>
struct GatherBlockLoader {
  STEEL_CONST short vec_size = n_reads;
  const int src_ld;
  const short bi;
  const short bj;
  threadgroup T* dst;
  const device T* src;        // base hidden [M, K]
  const device int* srow;     // [npad] gathered row per sorted slot (<0 = pad)
  const int base_slot;        // tile's first sorted slot (= y_row)
  int k_off;
  struct alignas(sizeof(T) * vec_size) ReadVector {
    uint8_t v[sizeof(T) * vec_size];
  };
  METAL_FUNC GatherBlockLoader(
      const device T* src_, const int src_ld_, const device int* srow_,
      const int base_slot_, threadgroup T* dst_, ushort simd_group_id,
      ushort simd_lane_id)
      : src_ld(src_ld_),
        bi((short)((simd_group_id * 32 + simd_lane_id) / TCOLS)),
        bj((short)(vec_size * ((simd_group_id * 32 + simd_lane_id) % TCOLS))),
        dst(dst_ + bi * dst_ld + bj),
        src(src_), srow(srow_), base_slot(base_slot_), k_off(0) {}
  METAL_FUNC void load_unsafe() const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      const int r = srow[base_slot + bi + i];
      if (r >= 0) {
        *((threadgroup ReadVector*)(&dst[i * dst_ld])) =
            *((const device ReadVector*)(&src[(int64_t)r * src_ld + bj + k_off]));
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) { dst[i * dst_ld + j] = T(0); }
      }
    }
  }
  METAL_FUNC void load_safe(short2) const { load_unsafe(); }
  METAL_FUNC void next() { k_off += BCOLS; }
};

// ===================================================================
// MoE GROUPED steel GEMM (Qwen3.5-MoE prefill -- the matrix-tiled expert
// GEMM that closes the prefill gap vs MLX gather_qmm). The (token,expert)
// pairs are counting-sorted into per-expert blocks padded to BM rows; each
// M-tile (tid.y) belongs to ONE expert (tile2e[tid.y]) and the weight slab is
// offset to it -- so the steel MMA runs per expert reading the weight once and
// at full matrix-core/SIMD efficiency. Token-exact with the per-pair/GEMV
// paths. The gate|up GEMM gathers its rows from the hidden via srow (fused into
// the load); the down GEMM scatters its rows to partials via sdst (fused into
// the store) -- no explicit gather/scatter copies.
// ===================================================================
// Forward decls (impls below): the down GEMM's scatter store + gate|up gather.
template <typename T, const int group_size, const int bits,
          const int BM, const int BK, const int BN>
METAL_FUNC void qmm_t_grouped_down_impl(
    const device uint32_t*, const device T*, const device T*, const device T*,
    device T*, threadgroup T*, threadgroup T*, const constant int&,
    const constant int&, const constant int&, const device int*,
    const device int*, uint3, uint, uint);

kernel void affine_qmm_grouped_w4g64(
    const device uint32_t* w       [[buffer(0)]],
    const device VPIPE_ELT* scales [[buffer(1)]],
    const device VPIPE_ELT* biases [[buffer(2)]],
    const device VPIPE_ELT* x      [[buffer(3)]],   // sorted acts [npad,K]
    device VPIPE_ELT*       y      [[buffer(4)]],   // partials [np,N] scattered
    const constant int& K          [[buffer(5)]],
    const constant int& N          [[buffer(6)]],
    const constant int& M          [[buffer(7)]],
    const device int* tile2e       [[buffer(8)]],
    const device int* sdst         [[buffer(9)]],   // sorted-slot -> pair index
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_grouped_down_impl<VPIPE_ELT, 64, 4, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, tile2e, sdst,
      tid, simd_gid, simd_lid);
}

// 8-bit twin of affine_qmm_grouped_w4g64 (prefill steel down GEMM). The impl
// (QuantizedBlockLoader + K_w = K*bytes_per_pack/pack_factor) is fully bit-
// parameterized, so bits=8 just instantiates the w8 slab stride + loader.
kernel void affine_qmm_grouped_w8g64(
    const device uint32_t* w       [[buffer(0)]],
    const device VPIPE_ELT* scales [[buffer(1)]],
    const device VPIPE_ELT* biases [[buffer(2)]],
    const device VPIPE_ELT* x      [[buffer(3)]],   // sorted acts [npad,K]
    device VPIPE_ELT*       y      [[buffer(4)]],   // partials [np,N] scattered
    const constant int& K          [[buffer(5)]],
    const constant int& N          [[buffer(6)]],
    const constant int& M          [[buffer(7)]],
    const device int* tile2e       [[buffer(8)]],
    const device int* sdst         [[buffer(9)]],   // sorted-slot -> pair index
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_grouped_down_impl<VPIPE_ELT, 64, 8, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, tile2e, sdst,
      tid, simd_gid, simd_lid);
}

// Grouped gate|up + SwiGLU GEMM (copy of qmm_t_swiglu_impl + the per-M-tile
// expert weight offset). Interleaved [E, 2*inner, K] weight; output halved to
// sorted [npad, inner] (silu(gate)*up).
template <typename T, const int group_size, const int bits,
          const int BM = 32, const int BK = 32, const int BN = 32>
METAL_FUNC void qmm_t_grouped_swiglu_impl(
    const device uint32_t* w, const device T* scales, const device T* biases,
    const device T* x, device T* y, threadgroup T* Xs, threadgroup T* Ws,
    const constant int& K, const constant int& N, const constant int& M,
    const device int* tile2e, const device int* srow,
    uint3 tid, uint simd_gid, uint simd_lid) {
  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  // x is GATHERED from the hidden via srow (the gather fused into the load).
  using loader_x_t =
      GatherBlockLoader<T, BM, BK, BK_padded, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, bits>;
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  auto wl = (const device uint8_t*)w;
  if (tile2e) {
    const int e = tile2e[tid.y];
    if (e < 0) { return; }
    wl += static_cast<int64_t>(e) * N * K_w;
    scales += static_cast<int64_t>(e) * N * K_g;
    biases += static_cast<int64_t>(e) * N * K_g;
  }
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  const short num_els = min(BM, M - y_row);
  loader_x_t loader_x(x, K, srow, y_row, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);
  if (num_els < BM) {
    for (int k = 0; k < K; k += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(BK, num_els));
      loader_w.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
      loader_x.next();
      loader_w.next();
    }
  } else {
    for (int k = 0; k < K; k += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_unsafe();
      loader_w.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
      loader_x.next();
      loader_w.next();
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  constexpr int FS = 8;
  constexpr int TM = BM / (FS * WM);
  constexpr int TN = BN / (FS * WN);
  const int outN = N / 2;
  const short sm = mma_op.sm;
  const short sn = mma_op.sn;
  for (short ti = 0; ti < TM; ti++) {
    const int row = y_row + sm + ti * FS * WM;
    if (row >= M) { continue; }
    for (short tj = 0; tj < TN; tj++) {
      const int col = y_col + sn + tj * FS * WN;
      const thread auto& fr = mma_op.Ctile.frag_at(ti, tj);
      const float ga = (float)fr[0];
      const float up = (float)fr[1];
      const float s = ga / (1.0f + metal::exp(-ga));
      y[(int64_t)row * outN + (col >> 1)] = (T)(s * up);
    }
  }
}

kernel void affine_qmm_grouped_swiglu_w4g64(
    const device uint32_t* w       [[buffer(0)]],
    const device VPIPE_ELT* scales [[buffer(1)]],
    const device VPIPE_ELT* biases [[buffer(2)]],
    const device VPIPE_ELT* x      [[buffer(3)]],   // hidden [M, K] (gathered)
    device VPIPE_ELT*       y      [[buffer(4)]],
    const constant int& K          [[buffer(5)]],
    const constant int& N          [[buffer(6)]],   // fused width = 2*inner
    const constant int& M          [[buffer(7)]],
    const device int* tile2e       [[buffer(8)]],
    const device int* srow         [[buffer(9)]],   // sorted-slot -> hidden row
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_grouped_swiglu_impl<VPIPE_ELT, 64, 4, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, tile2e, srow,
      tid, simd_gid, simd_lid);
}

// 8-bit twin of affine_qmm_grouped_swiglu_w4g64 (prefill steel gate|up GEMM).
// Fully bit-parameterized impl; bits=8 instantiates the w8 slab stride +
// QuantizedBlockLoader.
kernel void affine_qmm_grouped_swiglu_w8g64(
    const device uint32_t* w       [[buffer(0)]],
    const device VPIPE_ELT* scales [[buffer(1)]],
    const device VPIPE_ELT* biases [[buffer(2)]],
    const device VPIPE_ELT* x      [[buffer(3)]],   // hidden [M, K] (gathered)
    device VPIPE_ELT*       y      [[buffer(4)]],
    const constant int& K          [[buffer(5)]],
    const constant int& N          [[buffer(6)]],   // fused width = 2*inner
    const constant int& M          [[buffer(7)]],
    const device int* tile2e       [[buffer(8)]],
    const device int* srow         [[buffer(9)]],   // sorted-slot -> hidden row
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_grouped_swiglu_impl<VPIPE_ELT, 64, 8, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, tile2e, srow,
      tid, simd_gid, simd_lid);
}

// Grouped down GEMM with a SCATTER store: each sorted output row is written to
// partials[sdst[slot], :] (fuses moe_scatter_back into the store). x = sorted
// activations (contiguous); w = per-tile expert down slab. Mirrors qmm_t_impl's
// MMA loop; only the store differs (manual per-row scatter, no store_result).
template <typename T, const int group_size, const int bits,
          const int BM = 32, const int BK = 32, const int BN = 32>
METAL_FUNC void qmm_t_grouped_down_impl(
    const device uint32_t* w, const device T* scales, const device T* biases,
    const device T* x, device T* y, threadgroup T* Xs, threadgroup T* Ws,
    const constant int& K, const constant int& N, const constant int& M,
    const device int* tile2e, const device int* sdst,
    uint3 tid, uint simd_gid, uint simd_lid) {
  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, bits>;
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  auto wl = (const device uint8_t*)w;
  if (tile2e) {
    const int e = tile2e[tid.y];
    if (e < 0) { return; }
    wl += static_cast<int64_t>(e) * N * K_w;
    scales += static_cast<int64_t>(e) * N * K_g;
    biases += static_cast<int64_t>(e) * N * K_g;
  }
  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  const short num_els = min(BM, M - y_row);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);
  if (num_els < BM) {
    for (int k = 0; k < K; k += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(BK, num_els));
      loader_w.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
      loader_x.next();
      loader_w.next();
    }
  } else {
    for (int k = 0; k < K; k += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_unsafe();
      loader_w.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
      loader_x.next();
      loader_w.next();
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  constexpr int FS = 8;
  constexpr int TM = BM / (FS * WM);
  constexpr int TN = BN / (FS * WN);
  const short sm = mma_op.sm;
  const short sn = mma_op.sn;
  for (short ti = 0; ti < TM; ti++) {
    const int row = y_row + sm + ti * FS * WM;
    if (row >= M) { continue; }
    const int dr = sdst[row];              // pair-ordered destination (or -1)
    if (dr < 0) { continue; }
    for (short tj = 0; tj < TN; tj++) {
      const int col = y_col + sn + tj * FS * WN;
      const thread auto& fr = mma_op.Ctile.frag_at(ti, tj);
      y[(int64_t)dr * N + col] = (T)fr[0];
      y[(int64_t)dr * N + col + 1] = (T)fr[1];
    }
  }
}

// Fused GeGLU MLP GEMM (Gemma-4 prefill): mirrors qmm_t_swiglu_impl
// verbatim; only the activation in the store is gelu_pytorch_tanh (matches
// geglu_f16) instead of silu. Interleaved [N=2*ffn, K] weight, output y
// halved to [M, ffn].  0:w 1:scales 2:biases 3:x 4:y 5:K 6:N(=2*ffn) 7:M
template <
    typename T,
    const int group_size,
    const int bits,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
METAL_FUNC void qmm_t_geglu_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, bits>;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;
  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;

  const short num_els = min(BM, M - y_row);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  if (num_els < BM) {
    for (int k = 0; k < K; k += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(BK, num_els));
      loader_w.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
      loader_x.next();
      loader_w.next();
    }
  } else {
    for (int k = 0; k < K; k += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_unsafe();
      loader_w.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
      loader_x.next();
      loader_w.next();
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Fused GeGLU store: frag elem[0]=gate (even col), elem[1]=up (odd col);
  // emit gelu_tanh(gate)*up to y[row, col/2]. Output width N/2.
  constexpr int FS = 8;
  constexpr int TM = BM / (FS * WM);
  constexpr int TN = BN / (FS * WN);
  const int outN = N / 2;
  const short sm = mma_op.sm;
  const short sn = mma_op.sn;
  for (short ti = 0; ti < TM; ti++) {
    const int row = y_row + sm + ti * FS * WM;
    if (row >= M) { continue; }
    for (short tj = 0; tj < TN; tj++) {
      const int col = y_col + sn + tj * FS * WN;   // even fused column
      const thread auto& fr = mma_op.Ctile.frag_at(ti, tj);
      const float ga = (float)fr[0];               // gate (even col)
      const float up = (float)fr[1];               // up   (odd col)
      const float k0 = 0.7978845608028654f;
      const float t = metal::precise::tanh(k0 * (ga + 0.044715f * ga * ga * ga));
      y[(int64_t)row * outN + (col >> 1)] = (T)(0.5f * ga * (1.0f + t) * up);
    }
  }
}

kernel void affine_qmm_geglu_w4g64(
    const device uint32_t* w      [[buffer(0)]],
    const device VPIPE_ELT*     scales [[buffer(1)]],
    const device VPIPE_ELT*     biases [[buffer(2)]],
    const device VPIPE_ELT*     x      [[buffer(3)]],
    device VPIPE_ELT*           y      [[buffer(4)]],
    const constant int&    K      [[buffer(5)]],
    const constant int&    N      [[buffer(6)]],   // fused width = 2*ffn
    const constant int&    M      [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_geglu_impl<VPIPE_ELT, 64, 4, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M,
      tid, simd_gid, simd_lid);
}

// group_size=32 twin of affine_qmm_geglu_w4g64 (GGUF q4_0). Only the
// template group_size differs.
kernel void affine_qmm_geglu_w4g32(
    const device uint32_t* w      [[buffer(0)]],
    const device VPIPE_ELT*     scales [[buffer(1)]],
    const device VPIPE_ELT*     biases [[buffer(2)]],
    const device VPIPE_ELT*     x      [[buffer(3)]],
    device VPIPE_ELT*           y      [[buffer(4)]],
    const constant int&    K      [[buffer(5)]],
    const constant int&    N      [[buffer(6)]],   // fused width = 2*ffn
    const constant int&    M      [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_geglu_impl<VPIPE_ELT, 32, 4, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M,
      tid, simd_gid, simd_lid);
}

// ===================================================================
// 8-bit (group 64) steel GEMM entry points. The MMA loop, dequantize,
// and QuantizedBlockLoader are all `bits`-templated (dequantize handles
// bits==8), so these mirror the 4-bit kernels with bits=8. One byte per
// weight value (pack_factor=bytes_per_pack=1 for bits=8).
// ===================================================================
kernel void affine_qmm_steel_w8g64(
    const device uint32_t* w      [[buffer(0)]],
    const device VPIPE_ELT*     scales [[buffer(1)]],
    const device VPIPE_ELT*     biases [[buffer(2)]],
    const device VPIPE_ELT*     x      [[buffer(3)]],
    device VPIPE_ELT*           y      [[buffer(4)]],
    const constant int&    K      [[buffer(5)]],
    const constant int&    N      [[buffer(6)]],
    const constant int&    M      [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  lid      [[thread_index_in_threadgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_impl<VPIPE_ELT, 64, 8, /*aligned_N=*/true, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, K,
      /*tile2e=*/nullptr, tid, lid, simd_gid, simd_lid);
}

// group_size=32 twin of affine_qmm_steel_w8g64 (the MOSS codec int8 path):
// only the template group_size differs. Requires N % 32 == 0 (aligned_N).
kernel void affine_qmm_steel_w8g32(
    const device uint32_t* w      [[buffer(0)]],
    const device VPIPE_ELT*     scales [[buffer(1)]],
    const device VPIPE_ELT*     biases [[buffer(2)]],
    const device VPIPE_ELT*     x      [[buffer(3)]],
    device VPIPE_ELT*           y      [[buffer(4)]],
    const constant int&    K      [[buffer(5)]],
    const constant int&    N      [[buffer(6)]],
    const constant int&    M      [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  lid      [[thread_index_in_threadgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_impl<VPIPE_ELT, 32, 8, /*aligned_N=*/true, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, K,
      /*tile2e=*/nullptr, tid, lid, simd_gid, simd_lid);
}

kernel void affine_qmm_swiglu_w8g64(
    const device uint32_t* w      [[buffer(0)]],
    const device VPIPE_ELT*     scales [[buffer(1)]],
    const device VPIPE_ELT*     biases [[buffer(2)]],
    const device VPIPE_ELT*     x      [[buffer(3)]],
    device VPIPE_ELT*           y      [[buffer(4)]],
    const constant int&    K      [[buffer(5)]],
    const constant int&    N      [[buffer(6)]],   // fused width = 2*ffn
    const constant int&    M      [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_swiglu_impl<VPIPE_ELT, 64, 8, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M,
      tid, simd_gid, simd_lid);
}

// Fused GeGLU MLP steel GEMM, 8-bit weights (Gemma-4 12B gate/up). Mirrors
// affine_qmm_geglu_w4g64 with bits=8.
kernel void affine_qmm_geglu_w8g64(
    const device uint32_t* w      [[buffer(0)]],
    const device VPIPE_ELT*     scales [[buffer(1)]],
    const device VPIPE_ELT*     biases [[buffer(2)]],
    const device VPIPE_ELT*     x      [[buffer(3)]],
    device VPIPE_ELT*           y      [[buffer(4)]],
    const constant int&    K      [[buffer(5)]],
    const constant int&    N      [[buffer(6)]],   // fused width = 2*ffn
    const constant int&    M      [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_geglu_impl<VPIPE_ELT, 64, 8, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M,
      tid, simd_gid, simd_lid);
}

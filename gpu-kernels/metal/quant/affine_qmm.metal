// affine_qmm.metal -- 4-bit affine quantized matrix * matrix (GEMM),
// y[M,N] = x[M,K] @ dequant(w)[N,K]^T. The prefill counterpart to
// affine_qmv: when M>1 we must load each weight tile ONCE and reuse
// it across all M rows, so this is a real tiled GEMM (running the
// GEMV M times would re-stream the weights M times).
//
// Self-contained simdgroup_matrix tiling -- no MLX/steel deps. The
// weight tile is dequantized into threadgroup memory inline (affine:
// w_deq = scale*q + bias, 4-bit nibbles 8/uint32 low-first, one
// scale/bias per 64-wide K group), then a half-in / float-accumulate
// MMA runs over it.
//
// This GPU is OCCUPANCY-bound (120 GB/s, base tier), so the kernel:
//   * keeps threadgroup memory small -- one BK-chunk of X and W, no
//     double buffering (that regressed), no float spill tile;
//   * register-blocks each simdgroup over a 32x16 output sub-tile
//     (4x2 = 8 accumulator fragments) so each shared-memory barrier
//     amortizes more MMAs (GEMM here is compute-bound, intensity
//     ~2000 FLOP/byte);
//   * stores fragments straight to device memory (float->half in
//     registers), which needs the output row count padded up to BM.
//
//   T          = half        bits = 4        group_size = 64
//
// Tile: BM=64, BN=32, BK=32; 4 simdgroups (WM=2 x WN=2). K is a
// multiple of 512 and N a multiple of 32 for every Llama matmul; the
// caller pads y's rows up to a multiple of BM (only first M valid).
//
// Argument layout (MSL buffer slots):
//   0: w      (device const uint32*)  packed weights  [N, K/8]
//   1: scales (device const half*)    per-group scale [N, K/64]
//   2: biases (device const half*)    per-group bias  [N, K/64]
//   3: x      (device const half*)    activations     [M, K]
//   4: y      (device half*)          output          [padM, N]
//   5: K      (constant int&)
//   6: N      (constant int&)
//   7: M      (constant int&)

#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

#define BM 64
#define BN 64
#define BK 32              // K-chunk
#define BK_PAD 40          // BK + 16/sizeof(half), bank-conflict pad
#define GROUP_SIZE 64
#define WM 4               // simdgroup rows
#define WN 2               // simdgroup columns
// Keep fragments-per-simdgroup small: each simdgroup_matrix is a
// scarce register resource, and >~16 live matrices per thread
// silently corrupts results. MFRAG*NFRAG + a[MFRAG] + b[NFRAG] must
// stay modest. Here 2*4 accumulators + 2 + 4 = 14.
#define NTHREADS (WM * WN * 32)   // 256
#define MFRAG (BM / WM / 8)       // row fragments per simdgroup = 2
#define NFRAG (BN / WN / 8)       // col fragments per simdgroup = 4
#define WORDS_PER_ROW (BK / 8)
#define BN_PAD (BN + 8)    // bank-conflict pad for the [k][n] W tile
// As is [BM][BK_PAD]; Bs is the W tile stored TRANSPOSED as [BK][BN_PAD]
// so the MMA can plain-load B fragments (no per-read transpose).
#define TILE_HALVES (BM * BK_PAD + BK * BN_PAD)

kernel void affine_qmm_w4g64(
    const device uint32_t* w      [[buffer(0)]],
    const device half*     scales [[buffer(1)]],
    const device half*     biases [[buffer(2)]],
    const device half*     x      [[buffer(3)]],
    device half*           y      [[buffer(4)]],
    constant int&          K      [[buffer(5)]],
    constant int&          N      [[buffer(6)]],
    constant int&          M      [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  lid      [[thread_index_in_threadgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]])
{
  threadgroup half tile[TILE_HALVES];
  threadgroup half* As = tile;
  threadgroup half* Bs = tile + BM * BK_PAD;

  const int y_row0 = tid.y * BM;
  const int y_col0 = tid.x * BN;
  const int K_g = K / GROUP_SIZE;
  const int K_w = K / 8;

  const int wm = simd_gid / WN;        // 0..WM-1
  const int wn = simd_gid % WN;        // 0..WN-1
  const int row_base = wm * (BM / WM); // first output row of simdgroup
  const int col_base = wn * (BN / WN); // first output col of simdgroup

  simdgroup_matrix<float, 8, 8> C[MFRAG][NFRAG];
  for (int i = 0; i < MFRAG; i++) {
    for (int j = 0; j < NFRAG; j++) {
      C[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
    }
  }

  for (int k0 = 0; k0 < K; k0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage X tile [BM, BK] -> As (zero-fill rows past M). K is
    // contiguous in x and BK_PAD is even, so load/store half4.
    for (int e = (int)lid; e < BM * (BK / 4); e += NTHREADS) {
      const int r = e / (BK / 4);
      const int c = (e % (BK / 4)) * 4;
      const int gr = y_row0 + r;
      const half4 v = (gr < M)
          ? *(const device half4*)(x + (uint)gr * K + k0 + c)
          : half4(0);
      *(threadgroup half4*)(As + r * BK_PAD + c) = v;
    }

    // Dequantize W tile [BN, BK] -> Bs.
    const int kg = k0 / GROUP_SIZE;
    for (int wi = (int)lid; wi < BN * WORDS_PER_ROW; wi += NTHREADS) {
      const int row = wi / WORDS_PER_ROW;
      const int wcol = wi % WORDS_PER_ROW;
      const int n = y_col0 + row;
      const float s = scales[(uint)n * K_g + kg];
      const float b = biases[(uint)n * K_g + kg];
      const uint32_t packed = w[(uint)n * K_w + k0 / 8 + wcol];
      // Write into the TRANSPOSED tile Bs[k][n]: the 8 nibbles span
      // k = wcol*8 .. wcol*8+7 for this output col `row`, so the
      // stores are strided by BN_PAD (scattered). That's fine -- the
      // dequant runs once per tile while B is read ~16x.
      const float s1 = s * (1.0f / 16.0f);
      const uint b0 = packed & 0xff, b1 = (packed >> 8) & 0xff;
      const uint b2 = (packed >> 16) & 0xff, b3 = (packed >> 24) & 0xff;
      threadgroup half* col = Bs + wcol * 8 * BN_PAD + row;
      col[0 * BN_PAD] = half(s  * float(b0 & 0x0f) + b);
      col[1 * BN_PAD] = half(s1 * float(b0 & 0xf0) + b);
      col[2 * BN_PAD] = half(s  * float(b1 & 0x0f) + b);
      col[3 * BN_PAD] = half(s1 * float(b1 & 0xf0) + b);
      col[4 * BN_PAD] = half(s  * float(b2 & 0x0f) + b);
      col[5 * BN_PAD] = half(s1 * float(b2 & 0xf0) + b);
      col[6 * BN_PAD] = half(s  * float(b3 & 0x0f) + b);
      col[7 * BN_PAD] = half(s1 * float(b3 & 0xf0) + b);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int kk = 0; kk < BK; kk += 8) {
      simdgroup_matrix<half, 8, 8> a[MFRAG];
      simdgroup_matrix<half, 8, 8> b[NFRAG];
      for (int mi = 0; mi < MFRAG; mi++) {
        simdgroup_load(a[mi], As + (row_base + mi * 8) * BK_PAD + kk, BK_PAD);
      }
      for (int ni = 0; ni < NFRAG; ni++) {
        // Bs is already [k][n]; plain load gives the [k][n] fragment.
        simdgroup_load(
            b[ni], Bs + kk * BN_PAD + (col_base + ni * 8), BN_PAD);
      }
      for (int mi = 0; mi < MFRAG; mi++) {
        for (int ni = 0; ni < NFRAG; ni++) {
          simdgroup_multiply_accumulate(C[mi][ni], a[mi], b[ni], C[mi][ni]);
        }
      }
    }
  }

  // Store fragments straight to device memory, float->half in
  // registers. y must be allocated with rows padded up to BM.
  for (int mi = 0; mi < MFRAG; mi++) {
    for (int ni = 0; ni < NFRAG; ni++) {
      const int rb = y_row0 + row_base + mi * 8;
      const int cb = y_col0 + col_base + ni * 8;
      simdgroup_matrix<half, 8, 8> hc;
      for (int t = 0; t < 2; t++) {
        hc.thread_elements()[t] = half(C[mi][ni].thread_elements()[t]);
      }
      simdgroup_store(hc, y + (uint)rb * N + cb, N);
    }
  }
}

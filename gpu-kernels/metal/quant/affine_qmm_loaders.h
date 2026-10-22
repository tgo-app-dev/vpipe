// affine_qmm_loaders.h -- the affine dequantizing tile loaders the
// quantized GEMMs share: MLX's get_pack_factor / get_bytes_per_pack /
// dequantize / QuantizedBlockLoader (affine path, with this tree's fixes:
// fp32 dequant arithmetic for bf16 storage, the 5-/6-bit bit stream, the
// ragged-tile row test) and the row-GATHERING activation loader the MoE
// grouped GEMMs read their sorted rows through. Included by the steel
// kernels (affine_qmm_steel.metal) and their matrix-core twins
// (affine_qmm_grouped_nax.metal), so both stage a weight tile identically.
//
// The includer brings <metal_stdlib> and `using namespace metal;` first.

#pragma once

#include "mlx/backend/metal/kernels/steel/defines.h"

#ifndef METAL_FUNC
#define METAL_FUNC inline
#endif
#ifndef MLX_MTL_CONST
#define MLX_MTL_CONST static constant constexpr const
#endif

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

  else if (bits == 5 || bits == 6) {
    // Codes from the little-endian bit stream (8 in 5 bytes / 4 in 3), the
    // dequant in Compute as the 4-bit arm does -- bf16 scalar math is
    // emulated, and rounding once on the store is the more accurate. (MLX's
    // own loop advanced its pointers cumulatively, right only for N <= 16.)
    const Compute cs = static_cast<Compute>(scale);
    const Compute cb = static_cast<Compute>(bias);
    constexpr int per = bits == 5 ? 8 : 4;
    constexpr int nb = bits == 5 ? 5 : 3;
    for (int i = 0; i < (N / per); i++) {
      const device uint8_t* b = w + nb * i;
      ulong v = 0;
      for (int j = 0; j < nb; j++) { v |= (ulong)b[j] << (8 * j); }
      for (int j = 0; j < per; j++) {
        const uint q = (uint)(v >> (bits * j)) & ((1u << bits) - 1u);
        w_local[per * i + j] =
            static_cast<U>(cs * static_cast<Compute>(q) + cb);
      }
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

    // bi is a tile ROW in either orientation and src_tile_dim.y is the rows
    // present, so one test serves both (MLX #3912). The reduction_dim == 1
    // arm compared bi against .x -- the BK columns, which bi never reaches
    // -- so a ragged last N tile still read every row past num_outs.
    if (bi >= src_tile_dim.y) {
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

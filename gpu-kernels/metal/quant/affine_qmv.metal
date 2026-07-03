// affine_qmv.metal -- 4-bit affine quantized matrix * vector (GEMV),
// y = x @ dequant(w)^T. This is the decode-critical "quantized
// matmul" keystone for running LLMs on the metal-compute framework
// without MLX. The math is a faithful port of MLX's qmv_fast_impl
// (extern/mlx/.../kernels/quantized.h) specialized to:
//
//   T          = half        (f16 activations / scales / output)
//   bits       = 4
//   group_size = 64
//
// Affine dequant: w_deq = scale * q + bias, where q is a 4-bit value
// and (scale, bias) are shared across each contiguous group of 64
// values along the contracted (K) dimension. Weights are packed
// 8 nibbles per uint32, low nibble first.
//
// Constraints (hold for every Llama-3.1 matmul): in_vec_size (K) is
// a multiple of 512, out_vec_size (N) is a multiple of 8.
//
// Argument layout (MSL buffer slots):
//   0: w            (device const uint32*)  packed weights  [N, K/8]
//   1: scales       (device const half*)    per-group scale [N, K/64]
//   2: biases       (device const half*)    per-group bias  [N, K/64]
//   3: x            (device const half*)    activations     [M, K]
//   4: y            (device VPIPE_ELT*)          output          [M, N]
//   5: in_vec_size  (constant int&)         K
//   6: out_vec_size (constant int&)         N

#include <metal_stdlib>

using namespace metal;

// Element (storage) type: half by default; -DVPIPE_ELT=bfloat compiles
// the bf16 variant. Quantized weights stay uint32; math stays f32.
#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif

#define SIMD_SIZE 32
#ifndef METAL_FUNC   // <metal_stdlib> already defines it (always-inline)
#define METAL_FUNC inline
#endif

template <int bits, int wsize = 8>
inline constexpr short get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}

template <int bits, int wsize = 8>
inline constexpr short get_bytes_per_pack() {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}

// Load values_per_thread activations into registers, pre-divided by
// powers of 16 so the masked-bit multiply in qdot reconstructs the
// nibble's magnitude. Returns sum(x) for the affine bias term.
template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector(const device T* x, thread U* x_thread) {
  U sum = 0;
  if (bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i]     = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  } else if (bits == 8) {
    // 8-bit weights are stored one byte per value (no nibble packing),
    // so no power-of-16 pre-division is needed; qdot multiplies the raw
    // byte directly.
    for (int i = 0; i < values_per_thread; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }
  return sum;
}

// Bounds-safe load for the LAST (partial) block when in_vec_size is not a
// multiple of block_size (= values_per_thread*32). `valid` is how many of
// this thread's values_per_thread lanes fall inside in_vec_size; the rest
// are zeroed so qdot contributes 0 (x*w) and the bias sum excludes them.
// Without this the tail lanes read past the row -> garbage dot. (vpipe's
// only ported GEMV is MLX's qmv_FAST, which assumes K % block_size == 0;
// Gemma-12B hidden 3840 = 7.5*512 is the first model to break the
// assumption -- Llama 4096 / Qwen 2048 / e4b 2048 are all multiples.)
template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector_safe(const device T* x, thread U* x_thread, int valid) {
  U sum = 0;
  if (bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      const U v0 = (i + 0 < valid) ? (U)x[i + 0] : (U)0;
      const U v1 = (i + 1 < valid) ? (U)x[i + 1] : (U)0;
      const U v2 = (i + 2 < valid) ? (U)x[i + 2] : (U)0;
      const U v3 = (i + 3 < valid) ? (U)x[i + 3] : (U)0;
      sum += v0 + v1 + v2 + v3;
      x_thread[i]     = v0;
      x_thread[i + 1] = v1 / 16.0f;
      x_thread[i + 2] = v2 / 256.0f;
      x_thread[i + 3] = v3 / 4096.0f;
    }
  } else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      const U v = (i < valid) ? (U)x[i] : (U)0;
      sum += v;
      x_thread[i] = v;
    }
  }
  return sum;
}

// Threadgroup-memory counterpart of load_vector (for the fused rmsnorm+qmv
// kernel, where the normalized x lives in threadgroup memory). Same 4-bit
// power-of-16 pre-division so qdot's nibble-shift trick is unchanged.
template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector_tg(const threadgroup T* x, thread U* x_thread) {
  U sum = 0;
  if (bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i]     = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  } else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }
  return sum;
}

// Dot product of x with one dequantized weight row chunk:
//   scale * sum_i(x_i * q_i) + bias * sum(x_i)
// The masked bits carry the nibble shifted left, which cancels the
// power-of-16 pre-division applied in load_vector.
//
// NOTE: this is the MLX qmv_fast form -- it deliberately MATCHES MLX bit-for-
// bit (the metal-vs-MLX token-exact bar), including MLX's affine-4bit
// cancellation (scale*sum(x*q) and bias*sum(x) are two large nearly-cancelling
// terms when bias ~ -8*scale and the nibbles cluster near 8). That cancellation
// is benign for LM inference (RMS-normed activations bound sum(x); MLX carries
// the same error and stays coherent). A cancellation-free CENTERED variant
// (accumulate sum x_i*(q_i-8)) was prototyped + verified but it makes qmv MORE
// accurate than MLX -> diverges from MLX -> breaks matches_cpu_and_mlx; if a
// q4_0 model ever needs it, add it as a symmetric-q4-only kernel variant so the
// MLX-native (g64) path keeps matching MLX.
template <typename U, int values_per_thread, int bits>
inline U qdot(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum) {
  U accum = 0;
  if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i]     * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  } else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * w[i];
    }
  }
  return scale * accum + sum * bias;
}

// RPS/PPT/NSG (results_per_simdgroup / packs_per_thread / num_simdgroups)
// are the bandwidth-tuning knobs: more rows-per-simdgroup and packs-per-
// thread issue more in-flight weight loads, hiding DRAM latency (the
// decode GEMV is weight-read bound). Defaults = MLX's qmv_fast (4/2/2);
// override via the *_rNpM[_sK] entry points and pick the best per GPU.
// SCALE_ONLY: for symmetric quants (GGUF Q4_0, repacked to affine with the
// REDUNDANT bias = -8*scale), skip the per-group bias buffer read entirely and
// compute b = -8*scale in-register. Bit-identical (-8*d is exact in fp16), and
// drops ~11% of the weight-read bytes (2 of every 20 B/block) -> ~that much
// decode bandwidth back, matching llama.cpp's native scale-only Q4_0 mat-vec.
template <typename T, int group_size, int bits, bool ADD,
          int RPS = 4, int PPT = 2, int NSG = 2, bool SCALE_ONLY = false,
          bool GELU_MUL = false>
METAL_FUNC void qmv_fast_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const device T* residual,   // ADD: += residual[row]. GELU_MUL: *= it (the
                                // per-layer-input gate vector). [out_vec_size]
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int packs_per_thread = PPT;
  constexpr int num_simdgroups = NSG;
  constexpr int results_per_simdgroup = RPS;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  // Adjust positions
  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  if (!SCALE_ONLY) {
    biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  }
  x += tid.x * in_vec_size + simd_lid * values_per_thread;
  y += tid.x * out_vec_size + out_row;
  if (ADD || GELU_MUL) {
    residual += tid.x * out_vec_size + out_row;
  }

  for (int k = 0; k < in_vec_size; k += block_size) {
    // Last block may be partial (in_vec_size % block_size != 0): mask the
    // tail lanes that fall past in_vec_size so they contribute 0.
    const int krem = in_vec_size - k;
    U sum;
    if (krem >= block_size) {
      sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);
    } else {
      const int v = in_vec_size - (k + (int)simd_lid * values_per_thread);
      sum = load_vector_safe<T, U, values_per_thread, bits>(
          x, x_thread, v < 0 ? 0 : v);
    }

    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device T* sl = scales + row * in_vec_size_g;

      U s = sl[0];
      // Q4_0 symmetric: bias is exactly -8*scale, computed in-register so the
      // bias buffer is never read (SCALE_ONLY). Else read the stored bias.
      U b = SCALE_ONLY ? (U)(-8.0f) * s
                       : (U)(biases + row * in_vec_size_g)[0];
      result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    if (!SCALE_ONLY) { biases += block_size / group_size; }
    x += block_size;
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      T v = static_cast<T>(result[row]);
      if (ADD) { v += residual[row]; }
      if (GELU_MUL) {
        // gelu_pytorch_tanh(v) * residual[row] -- the gemma per-layer-input
        // gate fused into the GEMV write (drops the standalone geglu dispatch).
        // gelu's input is the f16-rounded GEMV result (== the standalone path,
        // which reads y back), so the output is bit-identical.
        const float xv = (float)v;
        const float k0 = 0.7978845608028654f;            // sqrt(2/pi)
        const float t = precise::tanh(k0 * (xv + 0.044715f * xv * xv * xv));
        v = (T)(0.5f * xv * (1.0f + t) * (float)residual[row]);
      }
      y[row] = v;
    }
  }
}

kernel void affine_qmv_w4g64(
    const device uint32_t* w            [[buffer(0)]],
    const device VPIPE_ELT*     scales       [[buffer(1)]],
    const device VPIPE_ELT*     biases       [[buffer(2)]],
    const device VPIPE_ELT*     x            [[buffer(3)]],
    device VPIPE_ELT*           y            [[buffer(4)]],
    constant int&          in_vec_size  [[buffer(5)]],
    constant int&          out_vec_size [[buffer(6)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  qmv_fast_impl<VPIPE_ELT, 64, 4, false>(
      w, scales, biases, x, y, /*residual=*/nullptr,
      in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// group_size=32 twin of affine_qmv_w4g64 (GGUF q4_0). Same dispatch
// geometry; only the template group_size differs.
kernel void affine_qmv_w4g32(
    const device uint32_t* w            [[buffer(0)]],
    const device VPIPE_ELT*     scales       [[buffer(1)]],
    const device VPIPE_ELT*     biases       [[buffer(2)]],
    const device VPIPE_ELT*     x            [[buffer(3)]],
    device VPIPE_ELT*           y            [[buffer(4)]],
    constant int&          in_vec_size  [[buffer(5)]],
    constant int&          out_vec_size [[buffer(6)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  qmv_fast_impl<VPIPE_ELT, 32, 4, false>(
      w, scales, biases, x, y, /*residual=*/nullptr,
      in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// Scale-only Q4_0 GEMV: same as affine_qmv_w4g32 but the bias buffer is NOT
// read -- bias = -8*scale (exact in fp16) is computed in-register. Bit-
// identical output, ~11% fewer weight-read bytes (matches llama.cpp's native
// scale-only Q4_0 mat-vec). buffer(2) is ignored (pass anything). The decode-
// critical FFN/proj GEMVs for GGUF Q4_0 models should use this.
kernel void affine_qmv_w4g32_q40(
    const device uint32_t* w            [[buffer(0)]],
    const device VPIPE_ELT*     scales       [[buffer(1)]],
    const device VPIPE_ELT*     biases       [[buffer(2)]],   // ignored
    const device VPIPE_ELT*     x            [[buffer(3)]],
    device VPIPE_ELT*           y            [[buffer(4)]],
    constant int&          in_vec_size  [[buffer(5)]],
    constant int&          out_vec_size [[buffer(6)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  qmv_fast_impl<VPIPE_ELT, 32, 4, false, 4, 2, 2, /*SCALE_ONLY=*/true>(
      w, scales, biases, x, y, /*residual=*/nullptr,
      in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// Fused variant: y = x @ dequant(w)^T + residual. Folds the residual
// add into the matmul write so the forward pass drops a dispatch +
// barrier per residual (o_proj, down_proj). buffer(7) = residual.
kernel void affine_qmv_w4g64_add(
    const device uint32_t* w            [[buffer(0)]],
    const device VPIPE_ELT*     scales       [[buffer(1)]],
    const device VPIPE_ELT*     biases       [[buffer(2)]],
    const device VPIPE_ELT*     x            [[buffer(3)]],
    device VPIPE_ELT*           y            [[buffer(4)]],
    constant int&          in_vec_size  [[buffer(5)]],
    constant int&          out_vec_size [[buffer(6)]],
    const device VPIPE_ELT*     residual     [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  qmv_fast_impl<VPIPE_ELT, 64, 4, true>(
      w, scales, biases, x, y, residual,
      in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// group_size=32 twin of affine_qmv_w4g64_add (GGUF q4_0). buffer(7) =
// residual; only the template group_size differs.
kernel void affine_qmv_w4g32_add(
    const device uint32_t* w            [[buffer(0)]],
    const device VPIPE_ELT*     scales       [[buffer(1)]],
    const device VPIPE_ELT*     biases       [[buffer(2)]],
    const device VPIPE_ELT*     x            [[buffer(3)]],
    device VPIPE_ELT*           y            [[buffer(4)]],
    constant int&          in_vec_size  [[buffer(5)]],
    constant int&          out_vec_size [[buffer(6)]],
    const device VPIPE_ELT*     residual     [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  qmv_fast_impl<VPIPE_ELT, 32, 4, true>(
      w, scales, biases, x, y, residual,
      in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// Scale-only Q4_0 twin of affine_qmv_w4g32_add (down/o-proj residual-fused
// GEMV): bias = -8*scale in-register, bias buffer (2) ignored. Bit-identical.
kernel void affine_qmv_w4g32_add_q40(
    const device uint32_t* w            [[buffer(0)]],
    const device VPIPE_ELT*     scales       [[buffer(1)]],
    const device VPIPE_ELT*     biases       [[buffer(2)]],   // ignored
    const device VPIPE_ELT*     x            [[buffer(3)]],
    device VPIPE_ELT*           y            [[buffer(4)]],
    constant int&          in_vec_size  [[buffer(5)]],
    constant int&          out_vec_size [[buffer(6)]],
    const device VPIPE_ELT*     residual     [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  qmv_fast_impl<VPIPE_ELT, 32, 4, true, 4, 2, 2, /*SCALE_ONLY=*/true>(
      w, scales, biases, x, y, residual,
      in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// ---- QKV-fused decode GEMV ---------------------------------------------
// One M=1 GEMV over the row-concatenated q|k|v weight [q_rows+k_rows+v_rows,
// K], writing each row to its own output buffer (q_out/k_out/v_out). Decode
// q/k/v share the same normed input and are independent, but issued as 2-3
// SEPARATE small GEMVs the k/v halves (e.g. 512 rows = 64 threadgroups)
// under-occupy the GPU and miss peak bandwidth. Fusing into ONE dispatch over
// the concatenated rows keeps the GPU saturated (e.g. 3072 rows = 384 TGs)
// while still landing q/k/v in their existing separate buffers (downstream
// norm/rope/kv-write unchanged). q_rows/k_rows are multiples of the 8-row
// threadgroup block, so each 8-row block lands entirely in one output buffer.
// v_rows==0 => q|k fusion (k_eq_v layers; V is derived from K separately).
template <typename T, int group_size, int bits>
METAL_FUNC void qmv_qkv_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* q_out,
    device T* k_out,
    device T* v_out,
    const constant int& in_vec_size,
    const constant int& q_rows,
    const constant int& k_rows,
    const constant int&,            // v_rows: unused (v-range = grid - q - k);
                                    // kept positional for the host buffer ABI.
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int packs_per_thread = 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;
  typedef float U;
  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += simd_lid * values_per_thread;

  for (int k = 0; k < in_vec_size; k += block_size) {
    const int krem = in_vec_size - k;
    U sum;
    if (krem >= block_size) {
      sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);
    } else {
      const int v = in_vec_size - (k + (int)simd_lid * values_per_thread);
      sum = load_vector_safe<T, U, values_per_thread, bits>(
          x, x_thread, v < 0 ? 0 : v);
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device T* sl = scales + row * in_vec_size_g;
      U s = sl[0];
      U b = (U)(biases + row * in_vec_size_g)[0];
      result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
    }
    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }

  const int qk = q_rows + k_rows;
  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      const int grow = out_row + row;
      const T val = static_cast<T>(result[row]);
      if (grow < q_rows) {
        q_out[grow] = val;
      } else if (grow < qk) {
        k_out[grow - q_rows] = val;
      } else {
        v_out[grow - qk] = val;
      }
    }
  }
}

kernel void affine_qmv_qkv_w4g64(
    const device uint32_t* w        [[buffer(0)]],
    const device VPIPE_ELT* scales  [[buffer(1)]],
    const device VPIPE_ELT* biases  [[buffer(2)]],
    const device VPIPE_ELT* x       [[buffer(3)]],
    device VPIPE_ELT*       q_out   [[buffer(4)]],
    device VPIPE_ELT*       k_out   [[buffer(5)]],
    device VPIPE_ELT*       v_out   [[buffer(6)]],
    constant int&           in_vec_size [[buffer(7)]],
    constant int&           q_rows  [[buffer(8)]],
    constant int&           k_rows  [[buffer(9)]],
    constant int&           v_rows  [[buffer(10)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  qmv_qkv_impl<VPIPE_ELT, 64, 4>(
      w, scales, biases, x, q_out, k_out, v_out,
      in_vec_size, q_rows, k_rows, v_rows, tid, simd_gid, simd_lid);
}

kernel void affine_qmv_qkv_w4g32(
    const device uint32_t* w        [[buffer(0)]],
    const device VPIPE_ELT* scales  [[buffer(1)]],
    const device VPIPE_ELT* biases  [[buffer(2)]],
    const device VPIPE_ELT* x       [[buffer(3)]],
    device VPIPE_ELT*       q_out   [[buffer(4)]],
    device VPIPE_ELT*       k_out   [[buffer(5)]],
    device VPIPE_ELT*       v_out   [[buffer(6)]],
    constant int&           in_vec_size [[buffer(7)]],
    constant int&           q_rows  [[buffer(8)]],
    constant int&           k_rows  [[buffer(9)]],
    constant int&           v_rows  [[buffer(10)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  qmv_qkv_impl<VPIPE_ELT, 32, 4>(
      w, scales, biases, x, q_out, k_out, v_out,
      in_vec_size, q_rows, k_rows, v_rows, tid, simd_gid, simd_lid);
}

// 8-bit twin of affine_qmv_qkv_w4g64 (uniform-w8 checkpoints: q|k|v all w8).
kernel void affine_qmv_qkv_w8g64(
    const device uint32_t* w        [[buffer(0)]],
    const device VPIPE_ELT* scales  [[buffer(1)]],
    const device VPIPE_ELT* biases  [[buffer(2)]],
    const device VPIPE_ELT* x       [[buffer(3)]],
    device VPIPE_ELT*       q_out   [[buffer(4)]],
    device VPIPE_ELT*       k_out   [[buffer(5)]],
    device VPIPE_ELT*       v_out   [[buffer(6)]],
    constant int&           in_vec_size [[buffer(7)]],
    constant int&           q_rows  [[buffer(8)]],
    constant int&           k_rows  [[buffer(9)]],
    constant int&           v_rows  [[buffer(10)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  qmv_qkv_impl<VPIPE_ELT, 64, 8>(
      w, scales, biases, x, q_out, k_out, v_out,
      in_vec_size, q_rows, k_rows, v_rows, tid, simd_gid, simd_lid);
}

// Fused RMSNorm(input_layernorm) + QKV qmv (Gemma-4 decode). Stages x into
// threadgroup memory, computes the row RMS there, normalizes by norm_w, then
// runs the QKV GEMV from threadgroup memory -- removing the standalone rms
// dispatch AND the normed-x device round-trip (one fewer dispatch + less
// intermediate bandwidth per layer). norm_w is Gemma's (1+w): the +1 is folded
// into the stored weight at LOAD time (the mlx checkpoint already bakes it), so
// this is a plain multiply, no runtime +1. tg_x is dynamic threadgroup memory
// sized by the host to ceil(in_vec_size/block_size)*block_size elements; the
// pad lanes are zeroed so the (non-safe) block loader matches the safe tail.
template <typename T, int group_size, int bits>
METAL_FUNC void qmv_qkv_rmsnorm_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    const device T* norm_w,
    device T* q_out,
    device T* k_out,
    device T* v_out,
    const constant int& in_vec_size,
    const constant int& q_rows,
    const constant int& k_rows,
    const constant int&,            // v_rows: unused (v-range = grid - q - k);
                                    // kept positional for the host buffer ABI.
    const constant float& eps,
    threadgroup T* tg_x,
    threadgroup float* partial,    // [num_simdgroups] reduction scratch
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int packs_per_thread = 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  // ---- RMSNorm prologue: stage + normalize x in threadgroup memory --------
  const int lid = (int)simd_gid * SIMD_SIZE + (int)simd_lid;   // 0..63
  const int nthreads = num_simdgroups * SIMD_SIZE;             // 64
  float sq = 0.0f;
  for (int k = lid; k < in_vec_size; k += nthreads) {
    const float v = float(x[k]);
    sq += v * v;
  }
  sq = simd_sum(sq);
  if (simd_lid == 0) { partial[simd_gid] = sq; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (int s = 0; s < num_simdgroups; ++s) { total += partial[s]; }
  const float inv = rsqrt(total / float(in_vec_size) + eps);
  // ceil to block_size so the GEMV's non-safe loader never reads past the pad.
  const int aligned = ((in_vec_size + block_size - 1) / block_size) * block_size;
  for (int k = lid; k < aligned; k += nthreads) {
    tg_x[k] = (k < in_vec_size)
        ? T(float(x[k]) * inv * float(norm_w[k]))
        : T(0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---- QKV GEMV from threadgroup memory (mirror of qmv_qkv_impl) ----------
  const device uint8_t* ws = (const device uint8_t*)w;
  typedef float U;
  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  const threadgroup T* xt = tg_x + simd_lid * values_per_thread;

  for (int k = 0; k < in_vec_size; k += block_size) {
    U sum = load_vector_tg<T, U, values_per_thread, bits>(xt, x_thread);
    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device T* sl = scales + row * in_vec_size_g;
      U s = sl[0];
      U b = (U)(biases + row * in_vec_size_g)[0];
      result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
    }
    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    xt += block_size;
  }

  const int qk = q_rows + k_rows;
  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      const int grow = out_row + row;
      const T val = static_cast<T>(result[row]);
      if (grow < q_rows) {
        q_out[grow] = val;
      } else if (grow < qk) {
        k_out[grow - q_rows] = val;
      } else {
        v_out[grow - qk] = val;
      }
    }
  }
}

kernel void affine_qmv_qkv_rms_w4g64(
    const device uint32_t* w        [[buffer(0)]],
    const device VPIPE_ELT* scales  [[buffer(1)]],
    const device VPIPE_ELT* biases  [[buffer(2)]],
    const device VPIPE_ELT* x       [[buffer(3)]],
    const device VPIPE_ELT* norm_w  [[buffer(4)]],
    device VPIPE_ELT*       q_out   [[buffer(5)]],
    device VPIPE_ELT*       k_out   [[buffer(6)]],
    device VPIPE_ELT*       v_out   [[buffer(7)]],
    constant int&           in_vec_size [[buffer(8)]],
    constant int&           q_rows  [[buffer(9)]],
    constant int&           k_rows  [[buffer(10)]],
    constant int&           v_rows  [[buffer(11)]],
    constant float&         eps     [[buffer(12)]],
    threadgroup VPIPE_ELT*  tg_x    [[threadgroup(0)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  threadgroup float partial[2];
  qmv_qkv_rmsnorm_impl<VPIPE_ELT, 64, 4>(
      w, scales, biases, x, norm_w, q_out, k_out, v_out, in_vec_size,
      q_rows, k_rows, v_rows, eps, tg_x, partial, tid, simd_gid, simd_lid);
}

kernel void affine_qmv_qkv_rms_w4g32(
    const device uint32_t* w        [[buffer(0)]],
    const device VPIPE_ELT* scales  [[buffer(1)]],
    const device VPIPE_ELT* biases  [[buffer(2)]],
    const device VPIPE_ELT* x       [[buffer(3)]],
    const device VPIPE_ELT* norm_w  [[buffer(4)]],
    device VPIPE_ELT*       q_out   [[buffer(5)]],
    device VPIPE_ELT*       k_out   [[buffer(6)]],
    device VPIPE_ELT*       v_out   [[buffer(7)]],
    constant int&           in_vec_size [[buffer(8)]],
    constant int&           q_rows  [[buffer(9)]],
    constant int&           k_rows  [[buffer(10)]],
    constant int&           v_rows  [[buffer(11)]],
    constant float&         eps     [[buffer(12)]],
    threadgroup VPIPE_ELT*  tg_x    [[threadgroup(0)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  threadgroup float partial[2];
  qmv_qkv_rmsnorm_impl<VPIPE_ELT, 32, 4>(
      w, scales, biases, x, norm_w, q_out, k_out, v_out, in_vec_size,
      q_rows, k_rows, v_rows, eps, tg_x, partial, tid, simd_gid, simd_lid);
}

// 8-bit twin of affine_qmv_qkv_rms_w4g64 (uniform-w8; opt-in RMS+QKV fuse).
kernel void affine_qmv_qkv_rms_w8g64(
    const device uint32_t* w        [[buffer(0)]],
    const device VPIPE_ELT* scales  [[buffer(1)]],
    const device VPIPE_ELT* biases  [[buffer(2)]],
    const device VPIPE_ELT* x       [[buffer(3)]],
    const device VPIPE_ELT* norm_w  [[buffer(4)]],
    device VPIPE_ELT*       q_out   [[buffer(5)]],
    device VPIPE_ELT*       k_out   [[buffer(6)]],
    device VPIPE_ELT*       v_out   [[buffer(7)]],
    constant int&           in_vec_size [[buffer(8)]],
    constant int&           q_rows  [[buffer(9)]],
    constant int&           k_rows  [[buffer(10)]],
    constant int&           v_rows  [[buffer(11)]],
    constant float&         eps     [[buffer(12)]],
    threadgroup VPIPE_ELT*  tg_x    [[threadgroup(0)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  threadgroup float partial[2];
  qmv_qkv_rmsnorm_impl<VPIPE_ELT, 64, 8>(
      w, scales, biases, x, norm_w, q_out, k_out, v_out, in_vec_size,
      q_rows, k_rows, v_rows, eps, tg_x, partial, tid, simd_gid, simd_lid);
}

// Fused per-layer-input GATE GEMV (Gemma-4 e4b decode): y = gelu_pytorch_tanh(
// dequant(w) @ x) * mul, where mul (buffer 7) is the per-layer-input vector
// pli[L] (out_vec_size == hpli). Folds the standalone geglu dispatch into the
// GEMV write -- one fewer dispatch per layer (42/token). buffer(2)=biases.
kernel void affine_qmv_gelu_mul_w4g64(
    const device uint32_t* w            [[buffer(0)]],
    const device VPIPE_ELT*     scales       [[buffer(1)]],
    const device VPIPE_ELT*     biases       [[buffer(2)]],
    const device VPIPE_ELT*     x            [[buffer(3)]],
    device VPIPE_ELT*           y            [[buffer(4)]],
    constant int&          in_vec_size  [[buffer(5)]],
    constant int&          out_vec_size [[buffer(6)]],
    const device VPIPE_ELT*     mul          [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  qmv_fast_impl<VPIPE_ELT, 64, 4, false, 4, 2, 2, false, /*GELU_MUL=*/true>(
      w, scales, biases, x, y, mul,
      in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// group_size=32 twin of affine_qmv_gelu_mul_w4g64.
kernel void affine_qmv_gelu_mul_w4g32(
    const device uint32_t* w            [[buffer(0)]],
    const device VPIPE_ELT*     scales       [[buffer(1)]],
    const device VPIPE_ELT*     biases       [[buffer(2)]],
    const device VPIPE_ELT*     x            [[buffer(3)]],
    device VPIPE_ELT*           y            [[buffer(4)]],
    constant int&          in_vec_size  [[buffer(5)]],
    constant int&          out_vec_size [[buffer(6)]],
    const device VPIPE_ELT*     mul          [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  qmv_fast_impl<VPIPE_ELT, 32, 4, false, 4, 2, 2, false, /*GELU_MUL=*/true>(
      w, scales, biases, x, y, mul,
      in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// 8-bit twin of affine_qmv_gelu_mul_w4g64 (Gemma PLE per-layer-input gate).
kernel void affine_qmv_gelu_mul_w8g64(
    const device uint32_t* w            [[buffer(0)]],
    const device VPIPE_ELT*     scales       [[buffer(1)]],
    const device VPIPE_ELT*     biases       [[buffer(2)]],
    const device VPIPE_ELT*     x            [[buffer(3)]],
    device VPIPE_ELT*           y            [[buffer(4)]],
    constant int&          in_vec_size  [[buffer(5)]],
    constant int&          out_vec_size [[buffer(6)]],
    const device VPIPE_ELT*     mul          [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  qmv_fast_impl<VPIPE_ELT, 64, 8, false, 4, 2, 2, false, /*GELU_MUL=*/true>(
      w, scales, biases, x, y, mul,
      in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// Fused SwiGLU MLP GEMV (decode path): y = silu(gate) * up. The gate
// and up weights are pre-interleaved by output feature into one
// [N=2*ffn, K] matrix (row 2g=gate g, row 2g+1=up g), matching the
// prefill fused GEMM. Each simdgroup computes results_per_simdgroup(=4)
// consecutive fused rows; out_row is a multiple of 4 (hence even), so
// the 4 results are (gate_g, up_g, gate_{g+1}, up_{g+1}) for g=out_row/2
// -- both (gate,up) pairs are already in this thread's registers. Writes
// the HALVED output [M, ffn]; no separate gate/up buffers, no swiglu
// pass. buffer layout matches affine_qmv_w4g64 with N = 2*ffn.
template <typename T, int group_size, int bits>
METAL_FUNC void qmv_swiglu_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,                        // [M, out_vec_size/2]
    const constant int& in_vec_size,    // K
    const constant int& out_vec_size,   // N = 2*ffn
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;
  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += tid.x * in_vec_size + simd_lid * values_per_thread;

  for (int k = 0; k < in_vec_size; k += block_size) {
    // Last block may be partial (in_vec_size % block_size != 0): mask the
    // tail lanes that fall past in_vec_size so they contribute 0.
    const int krem = in_vec_size - k;
    U sum;
    if (krem >= block_size) {
      sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);
    } else {
      const int v = in_vec_size - (k + (int)simd_lid * values_per_thread);
      sum = load_vector_safe<T, U, values_per_thread, bits>(
          x, x_thread, v < 0 ? 0 : v);
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device T* sl = scales + row * in_vec_size_g;
      const device T* bl = biases + row * in_vec_size_g;
      U s = sl[0];
      U b = bl[0];
      result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
    }
    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
  }
  if (simd_lid == 0) {
    // out_row even -> (g0,u0,g1,u1) for features out_row/2 and +1.
    device T* yo = y + tid.x * (out_vec_size / 2) + (out_row >> 1);
    const U g0 = result[0], u0 = result[1], g1 = result[2], u1 = result[3];
    yo[0] = static_cast<T>((g0 / (1.0f + metal::exp(-g0))) * u0);
    yo[1] = static_cast<T>((g1 / (1.0f + metal::exp(-g1))) * u1);
  }
}

kernel void affine_qmv_swiglu_w4g64(
    const device uint32_t* w            [[buffer(0)]],
    const device VPIPE_ELT*     scales       [[buffer(1)]],
    const device VPIPE_ELT*     biases       [[buffer(2)]],
    const device VPIPE_ELT*     x            [[buffer(3)]],
    device VPIPE_ELT*           y            [[buffer(4)]],
    constant int&          in_vec_size  [[buffer(5)]],
    constant int&          out_vec_size [[buffer(6)]],   // N = 2*ffn
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  qmv_swiglu_impl<VPIPE_ELT, 64, 4>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      tid, simd_gid, simd_lid);
}

// ====================================================================
// MoE (Qwen3.5-MoE) gathered expert GEMVs (decode, M=1).
//
// The router selects top_k of n_experts experts per token, producing a list
// of (token, expert) PAIRS -- npair = M*top_k (decode M=1 -> npair=top_k).
// Each expert's gate/up/down weights live in a BATCHED 3D affine-quantized
// tensor [n_experts, N, K-packed]; the gather kernels index the routed
// expert (pair_eid[p]) and run the standard qmv against the selected slab.
// grid.z = pair p; the weight/scale/bias/out/in base pointers are advanced
// to the expert's slab + the pair's rows before delegating to the shared
// qmv_fast_impl / qmv_swiglu_impl above -- so a gathered expert is token-
// EXACT with the same expert run through the plain kernels. tid.x is 0
// (grid.x == one threadgroup); tid.y tiles the output rows. This single pair-
// indexed form serves BOTH decode and prefill (the "pair-batched" path).
// ====================================================================

// Gate|up gather + SwiGLU: for pair p, x = X[p/top_k, :K] @ the routed
// expert's interleaved gate|up [2*inner, K] -> silu(gate)*up -> y[p, inner].
// Pairs are ordered [token*top_k + slot], so the input row is p/top_k (decode:
// top_k pairs, all row 0; prefill: M*top_k pairs over M token rows).
kernel void affine_gather_qmv_swiglu_w4g64(
    const device uint32_t* w        [[buffer(0)]],   // [E, 2*inner, K/8]
    const device VPIPE_ELT* scales  [[buffer(1)]],   // [E, 2*inner, K/64]
    const device VPIPE_ELT* biases  [[buffer(2)]],   // [E, 2*inner, K/64]
    const device VPIPE_ELT* x       [[buffer(3)]],   // [M, K] input rows
    device VPIPE_ELT*       y       [[buffer(4)]],   // [npair, inner]
    constant int& in_vec_size       [[buffer(5)]],   // K
    constant int& out_vec_size      [[buffer(6)]],   // N = 2*inner
    const device int* pair_eid      [[buffer(7)]],   // [npair] expert per pair
    constant int& top_k             [[buffer(8)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  const int p = (int)tid.z;
  const uint e = (uint)pair_eid[p];
  const uint wstride = (uint)out_vec_size * ((uint)in_vec_size / 8u);
  const uint gstride = (uint)out_vec_size * ((uint)in_vec_size / 64u);
  w      += e * wstride;
  scales += e * gstride;
  biases += e * gstride;
  x      += (uint)(p / top_k) * (uint)in_vec_size;
  y      += (uint)p * ((uint)out_vec_size / 2u);
  const uint3 t0 = uint3(0u, tid.y, 0u);
  qmv_swiglu_impl<VPIPE_ELT, 64, 4>(
      w, scales, biases, x, y, in_vec_size, out_vec_size, t0,
      simd_gid, simd_lid);
}

// 8-bit twin of affine_gather_qmv_swiglu_w4g64: the expert weight is w8
// (K/4 u32 words per row, so the slab stride uses /4u instead of /8u) and
// the impl is instantiated with bits=8. Per-row arithmetic (qmv_swiglu_impl
// -> qdot) is bit-parameterized, so this is token-exact with the plain w8
// swiglu kernel on the same expert.
kernel void affine_gather_qmv_swiglu_w8g64(
    const device uint32_t* w        [[buffer(0)]],   // [E, 2*inner, K/4]
    const device VPIPE_ELT* scales  [[buffer(1)]],   // [E, 2*inner, K/64]
    const device VPIPE_ELT* biases  [[buffer(2)]],   // [E, 2*inner, K/64]
    const device VPIPE_ELT* x       [[buffer(3)]],   // [M, K] input rows
    device VPIPE_ELT*       y       [[buffer(4)]],   // [npair, inner]
    constant int& in_vec_size       [[buffer(5)]],   // K
    constant int& out_vec_size      [[buffer(6)]],   // N = 2*inner
    const device int* pair_eid      [[buffer(7)]],   // [npair] expert per pair
    constant int& top_k             [[buffer(8)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  const int p = (int)tid.z;
  const uint e = (uint)pair_eid[p];
  const uint wstride = (uint)out_vec_size * ((uint)in_vec_size / 4u);
  const uint gstride = (uint)out_vec_size * ((uint)in_vec_size / 64u);
  w      += e * wstride;
  scales += e * gstride;
  biases += e * gstride;
  x      += (uint)(p / top_k) * (uint)in_vec_size;
  y      += (uint)p * ((uint)out_vec_size / 2u);
  const uint3 t0 = uint3(0u, tid.y, 0u);
  qmv_swiglu_impl<VPIPE_ELT, 64, 8>(
      w, scales, biases, x, y, in_vec_size, out_vec_size, t0,
      simd_gid, simd_lid);
}

// Down gather: x[p, inner] (per-pair expert activation, contiguous) @ the
// routed expert's down [H, inner] -> partials[p, H] (UN-weighted; moe_combine
// applies the routing weight and sums over the token's pairs).
kernel void affine_gather_down_qmv_w4g64(
    const device uint32_t* w        [[buffer(0)]],   // [E, H, inner/8]
    const device VPIPE_ELT* scales  [[buffer(1)]],   // [E, H, inner/64]
    const device VPIPE_ELT* biases  [[buffer(2)]],   // [E, H, inner/64]
    const device VPIPE_ELT* x       [[buffer(3)]],   // [npair, inner]
    device VPIPE_ELT*       y       [[buffer(4)]],   // [npair, H]
    constant int& in_vec_size       [[buffer(5)]],   // inner
    constant int& out_vec_size      [[buffer(6)]],   // H
    const device int* pair_eid      [[buffer(7)]],   // [npair] expert per pair
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  const int p = (int)tid.z;
  const uint e = (uint)pair_eid[p];
  const uint wstride = (uint)out_vec_size * ((uint)in_vec_size / 8u);
  const uint gstride = (uint)out_vec_size * ((uint)in_vec_size / 64u);
  w      += e * wstride;
  scales += e * gstride;
  biases += e * gstride;
  x      += (uint)p * (uint)in_vec_size;
  y      += (uint)p * (uint)out_vec_size;
  const uint3 t0 = uint3(0u, tid.y, 0u);
  qmv_fast_impl<VPIPE_ELT, 64, 4, false>(
      w, scales, biases, x, y, /*residual=*/nullptr, in_vec_size,
      out_vec_size, t0, simd_gid, simd_lid);
}

// 8-bit twin of affine_gather_down_qmv_w4g64: w8 slab stride uses /4u and
// the impl is instantiated with bits=8 (qmv_fast_impl -> qdot is bit-aware).
kernel void affine_gather_down_qmv_w8g64(
    const device uint32_t* w        [[buffer(0)]],   // [E, H, inner/4]
    const device VPIPE_ELT* scales  [[buffer(1)]],   // [E, H, inner/64]
    const device VPIPE_ELT* biases  [[buffer(2)]],   // [E, H, inner/64]
    const device VPIPE_ELT* x       [[buffer(3)]],   // [npair, inner]
    device VPIPE_ELT*       y       [[buffer(4)]],   // [npair, H]
    constant int& in_vec_size       [[buffer(5)]],   // inner
    constant int& out_vec_size      [[buffer(6)]],   // H
    const device int* pair_eid      [[buffer(7)]],   // [npair] expert per pair
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  const int p = (int)tid.z;
  const uint e = (uint)pair_eid[p];
  const uint wstride = (uint)out_vec_size * ((uint)in_vec_size / 4u);
  const uint gstride = (uint)out_vec_size * ((uint)in_vec_size / 64u);
  w      += e * wstride;
  scales += e * gstride;
  biases += e * gstride;
  x      += (uint)p * (uint)in_vec_size;
  y      += (uint)p * (uint)out_vec_size;
  const uint3 t0 = uint3(0u, tid.y, 0u);
  qmv_fast_impl<VPIPE_ELT, 64, 8, false>(
      w, scales, biases, x, y, /*residual=*/nullptr, in_vec_size,
      out_vec_size, t0, simd_gid, simd_lid);
}

// Shared-expert sigmoid gate (Qwen3.5-MoE): g[t] = sigmoid(dequant(w[1,K] w8)
// . X[t]). One simdgroup per input row (tid.y = row); lane 0 writes the gate.
// grid {32, M, 1}, tg {32,1,1}; decode is the M=1 case. The router gate and
// this shared gate are the only w8 tensors in the MoE block.
kernel void affine_moe_gate_w8g64(
    const device uint32_t* w       [[buffer(0)]],   // [1, K/4] (w8 packed)
    const device VPIPE_ELT* scales [[buffer(1)]],   // [1, K/64]
    const device VPIPE_ELT* biases [[buffer(2)]],   // [1, K/64]
    const device VPIPE_ELT* x      [[buffer(3)]],   // [M, K]
    device VPIPE_ELT*       outg   [[buffer(4)]],   // [M] = sigmoid(logit)
    constant int&  in_vec_size     [[buffer(5)]],   // K
    uint3 tid     [[threadgroup_position_in_grid]],
    uint simd_lid [[thread_index_in_simdgroup]])
{
  const device uint8_t* wb = (const device uint8_t*)w;   // K bytes (q in 0..255)
  const int K = in_vec_size;
  const device VPIPE_ELT* xr = x + (uint)tid.y * (uint)K;
  float acc = 0.0f;
  for (int k = (int)simd_lid; k < K; k += SIMD_SIZE) {
    const int g = k / 64;
    const float s = (float)scales[g];
    const float b = (float)biases[g];
    acc += ((float)wb[k] * s + b) * (float)xr[k];
  }
  acc = simd_sum(acc);
  if (simd_lid == 0) { outg[tid.y] = (VPIPE_ELT)(1.0f / (1.0f + metal::exp(-acc))); }
}

// Fused GeGLU MLP GEMV (Gemma-4): identical to qmv_swiglu_impl but the
// activation is gelu_pytorch_tanh (matches geglu_f16) instead of silu.
// SCALE_ONLY skips the redundant Q4_0 bias read (bias = -8*scale); see
// qmv_fast_impl.
template <typename T, int group_size, int bits, bool SCALE_ONLY = false>
METAL_FUNC void qmv_geglu_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,                        // [M, out_vec_size/2]
    const constant int& in_vec_size,    // K
    const constant int& out_vec_size,   // N = 2*ffn
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;
  typedef float U;
  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  if (!SCALE_ONLY) {
    biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  }
  x += tid.x * in_vec_size + simd_lid * values_per_thread;

  for (int k = 0; k < in_vec_size; k += block_size) {
    // Last block may be partial (in_vec_size % block_size != 0): mask the
    // tail lanes that fall past in_vec_size so they contribute 0.
    const int krem = in_vec_size - k;
    U sum;
    if (krem >= block_size) {
      sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);
    } else {
      const int v = in_vec_size - (k + (int)simd_lid * values_per_thread);
      sum = load_vector_safe<T, U, values_per_thread, bits>(
          x, x_thread, v < 0 ? 0 : v);
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const U s = scales[row * in_vec_size_g];
      // Q4_0 symmetric: bias = -8*scale (exact in fp16), no bias read.
      const U b = SCALE_ONLY ? (U)(-8.0f) * s
                             : (U)biases[row * in_vec_size_g];
      result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
    }
    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    if (!SCALE_ONLY) { biases += block_size / group_size; }
    x += block_size;
  }
  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
  }
  if (simd_lid == 0) {
    device T* yo = y + tid.x * (out_vec_size / 2) + (out_row >> 1);
    const U g0 = result[0], u0 = result[1], g1 = result[2], u1 = result[3];
    const U k0 = 0.7978845608028654f;
    const U t0 = metal::precise::tanh(k0 * (g0 + 0.044715f * g0 * g0 * g0));
    const U t1 = metal::precise::tanh(k0 * (g1 + 0.044715f * g1 * g1 * g1));
    yo[0] = static_cast<T>(0.5f * g0 * (1.0f + t0) * u0);
    yo[1] = static_cast<T>(0.5f * g1 * (1.0f + t1) * u1);
  }
}

kernel void affine_qmv_geglu_w4g64(
    const device uint32_t* w            [[buffer(0)]],
    const device VPIPE_ELT*     scales       [[buffer(1)]],
    const device VPIPE_ELT*     biases       [[buffer(2)]],
    const device VPIPE_ELT*     x            [[buffer(3)]],
    device VPIPE_ELT*           y            [[buffer(4)]],
    constant int&          in_vec_size  [[buffer(5)]],
    constant int&          out_vec_size [[buffer(6)]],   // N = 2*ffn
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  qmv_geglu_impl<VPIPE_ELT, 64, 4>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      tid, simd_gid, simd_lid);
}

// group_size=32 twin of affine_qmv_geglu_w4g64 (GGUF q4_0). Only the
// template group_size differs.
kernel void affine_qmv_geglu_w4g32(
    const device uint32_t* w            [[buffer(0)]],
    const device VPIPE_ELT*     scales       [[buffer(1)]],
    const device VPIPE_ELT*     biases       [[buffer(2)]],
    const device VPIPE_ELT*     x            [[buffer(3)]],
    device VPIPE_ELT*           y            [[buffer(4)]],
    constant int&          in_vec_size  [[buffer(5)]],
    constant int&          out_vec_size [[buffer(6)]],   // N = 2*ffn
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  qmv_geglu_impl<VPIPE_ELT, 32, 4>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      tid, simd_gid, simd_lid);
}

// Scale-only Q4_0 twin of affine_qmv_geglu_w4g32 (gate/up GEMV): bias =
// -8*scale in-register, bias buffer (2) ignored. Bit-identical, ~10% fewer
// weight-read bytes (the dominant FFN decode cost for GGUF Q4_0 models).
kernel void affine_qmv_geglu_w4g32_q40(
    const device uint32_t* w            [[buffer(0)]],
    const device VPIPE_ELT*     scales       [[buffer(1)]],
    const device VPIPE_ELT*     biases       [[buffer(2)]],   // ignored
    const device VPIPE_ELT*     x            [[buffer(3)]],
    device VPIPE_ELT*           y            [[buffer(4)]],
    constant int&          in_vec_size  [[buffer(5)]],
    constant int&          out_vec_size [[buffer(6)]],   // N = 2*ffn
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  qmv_geglu_impl<VPIPE_ELT, 32, 4, /*SCALE_ONLY=*/true>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      tid, simd_gid, simd_lid);
}

// Gemma-4 MoE gathered expert gate|up GEMV: the geglu (gelu_pytorch_tanh)
// twin of affine_gather_qmv_swiglu_w4g64. Identical routed-slab gather (the
// routed expert's interleaved gate|up), but delegates to qmv_geglu_impl so the
// activation is gelu(gate)*up. Token-exact with the plain qmv_geglu kernel on
// the same expert. Defined here (after qmv_geglu_impl) so the template is in
// scope.
kernel void affine_gather_qmv_geglu_w4g64(
    const device uint32_t* w        [[buffer(0)]],   // [E, 2*inner, K/8]
    const device VPIPE_ELT* scales  [[buffer(1)]],   // [E, 2*inner, K/64]
    const device VPIPE_ELT* biases  [[buffer(2)]],   // [E, 2*inner, K/64]
    const device VPIPE_ELT* x       [[buffer(3)]],   // [M, K] input rows
    device VPIPE_ELT*       y       [[buffer(4)]],   // [npair, inner]
    constant int& in_vec_size       [[buffer(5)]],   // K
    constant int& out_vec_size      [[buffer(6)]],   // N = 2*inner
    const device int* pair_eid      [[buffer(7)]],   // [npair] expert per pair
    constant int& top_k             [[buffer(8)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  const int p = (int)tid.z;
  const uint e = (uint)pair_eid[p];
  const uint wstride = (uint)out_vec_size * ((uint)in_vec_size / 8u);
  const uint gstride = (uint)out_vec_size * ((uint)in_vec_size / 64u);
  w      += e * wstride;
  scales += e * gstride;
  biases += e * gstride;
  x      += (uint)(p / top_k) * (uint)in_vec_size;
  y      += (uint)p * ((uint)out_vec_size / 2u);
  const uint3 t0 = uint3(0u, tid.y, 0u);
  qmv_geglu_impl<VPIPE_ELT, 64, 4>(
      w, scales, biases, x, y, in_vec_size, out_vec_size, t0,
      simd_gid, simd_lid);
}

// 8-bit twin of affine_gather_qmv_geglu_w4g64 (w8 slab stride /4u, bits=8).
kernel void affine_gather_qmv_geglu_w8g64(
    const device uint32_t* w        [[buffer(0)]],   // [E, 2*inner, K/4]
    const device VPIPE_ELT* scales  [[buffer(1)]],   // [E, 2*inner, K/64]
    const device VPIPE_ELT* biases  [[buffer(2)]],   // [E, 2*inner, K/64]
    const device VPIPE_ELT* x       [[buffer(3)]],   // [M, K] input rows
    device VPIPE_ELT*       y       [[buffer(4)]],   // [npair, inner]
    constant int& in_vec_size       [[buffer(5)]],   // K
    constant int& out_vec_size      [[buffer(6)]],   // N = 2*inner
    const device int* pair_eid      [[buffer(7)]],   // [npair] expert per pair
    constant int& top_k             [[buffer(8)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  const int p = (int)tid.z;
  const uint e = (uint)pair_eid[p];
  const uint wstride = (uint)out_vec_size * ((uint)in_vec_size / 4u);
  const uint gstride = (uint)out_vec_size * ((uint)in_vec_size / 64u);
  w      += e * wstride;
  scales += e * gstride;
  biases += e * gstride;
  x      += (uint)(p / top_k) * (uint)in_vec_size;
  y      += (uint)p * ((uint)out_vec_size / 2u);
  const uint3 t0 = uint3(0u, tid.y, 0u);
  qmv_geglu_impl<VPIPE_ELT, 64, 8>(
      w, scales, biases, x, y, in_vec_size, out_vec_size, t0,
      simd_gid, simd_lid);
}

// ===================================================================
// Batched GEMV (decode of N branches): y[N,Nout] = x[N,K] @ dequant(w)^T.
//
// Decode is DRAM-bound on the weight read. The steel GEMM (affine_qmm)
// reuses one weight read across its M rows but runs at a fraction of
// peak bandwidth for small M (it is tuned for prefill, M>=256), so at
// the decode batch (N=1..8) it is ~3x slower per weight-read than the
// qmv matvec. These kernels are the M-row generalization of qmv_fast:
// each thread loads one weight chunk ONCE and dots it against ALL N
// activation rows held in registers -- qmv's bandwidth, weights read
// once. The per-row arithmetic is bit-identical to affine_qmv_w4g64,
// so batched decode stays token-exact vs the serial qmv path.
//
// Rows are tiled by MAXM along grid.z: tile t = tid.z covers rows
// [t*MAXM, t*MAXM+MAXM); the last tile is partial (m_rows<MAXM). Pad
// rows alias the last valid row (no OOB read) and are never written.
// For N<=MAXM the whole batch is one tile -> weights read exactly once.
// Dispatch grid = {32, Nout/4, ceil(N/MAXM)}, tg = {32, 2, 1}; buffer(7)
// carries the total row count N (m_total).
// ===================================================================
template <typename T, int group_size, int bits, int MAXM>
METAL_FUNC void qmv_batch_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,                  // [m_total, K]
    device T* y,                        // [m_total, N]
    const constant int& in_vec_size,    // K
    const constant int& out_vec_size,   // N
    const constant int& m_total,        // total rows in the batch
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int packs_per_thread = 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const int base_row = (int)tid.z * MAXM;
  if (base_row >= m_total) { return; }
  const int m_rows = metal::min(MAXM, m_total - base_row);

  typedef float U;
  thread U x_thread[MAXM][values_per_thread];
  thread U result[results_per_simdgroup][MAXM];
  for (int r = 0; r < results_per_simdgroup; r++) {
    for (int m = 0; m < MAXM; m++) { result[r][m] = 0; }
  }

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  const device uint8_t* ws = (const device uint8_t*)w +
      out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;

  const device T* xptr[MAXM];
  for (int m = 0; m < MAXM; m++) {
    const int mm = (m < m_rows) ? m : (m_rows - 1);   // pad -> last valid row
    xptr[m] = x + (base_row + mm) * in_vec_size + simd_lid * values_per_thread;
  }

  for (int k = 0; k < in_vec_size; k += block_size) {
    // Last block may be partial (in_vec_size % block_size != 0): mask the
    // tail lanes past in_vec_size so they contribute 0 (see load_vector_safe).
    const int krem = in_vec_size - k;
    const int kvalid = in_vec_size - (k + (int)simd_lid * values_per_thread);
    U sum[MAXM];
    for (int m = 0; m < MAXM; m++) {
      sum[m] = (krem >= block_size)
          ? load_vector<T, U, values_per_thread, bits>(xptr[m], x_thread[m])
          : load_vector_safe<T, U, values_per_thread, bits>(
                xptr[m], x_thread[m], kvalid < 0 ? 0 : kvalid);
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
      const device uint8_t* wl = ws + row * in_vec_size_w;
      const U s = scales[row * in_vec_size_g];
      const U b = biases[row * in_vec_size_g];
      U acc[MAXM] = {0};
      if (bits == 4) {
        const device uint16_t* w16 = (const device uint16_t*)wl;
        for (int i = 0; i < (values_per_thread / 4); i++) {
          const uint16_t p = w16[i];
          const U w0 = (U)(p & 0x000f), w1 = (U)(p & 0x00f0);
          const U w2 = (U)(p & 0x0f00), w3 = (U)(p & 0xf000);
          for (int m = 0; m < MAXM; m++) {
            acc[m] += x_thread[m][4 * i] * w0 + x_thread[m][4 * i + 1] * w1 +
                      x_thread[m][4 * i + 2] * w2 + x_thread[m][4 * i + 3] * w3;
          }
        }
      } else if (bits == 8) {
        for (int i = 0; i < values_per_thread; i++) {
          const U wv = (U)wl[i];
          for (int m = 0; m < MAXM; m++) { acc[m] += x_thread[m][i] * wv; }
        }
      }
      for (int m = 0; m < MAXM; m++) { result[row][m] += s * acc[m] + sum[m] * b; }
    }
    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    for (int m = 0; m < MAXM; m++) { xptr[m] += block_size; }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    for (int m = 0; m < MAXM; m++) {
      const U v = simd_sum(result[row][m]);
      if (simd_lid == 0 && m < m_rows) {
        y[(base_row + m) * out_vec_size + out_row + row] = static_cast<T>(v);
      }
    }
  }
}

#define VPIPE_QMV_BATCH(NAME, MAXM)                                          \
  kernel void NAME(                                                          \
      const device uint32_t* w      [[buffer(0)]],                           \
      const device VPIPE_ELT* scales [[buffer(1)]],                          \
      const device VPIPE_ELT* biases [[buffer(2)]],                          \
      const device VPIPE_ELT* x      [[buffer(3)]],                          \
      device VPIPE_ELT*       y      [[buffer(4)]],                          \
      constant int& in_vec_size  [[buffer(5)]],                              \
      constant int& out_vec_size [[buffer(6)]],                              \
      constant int& m_total      [[buffer(7)]],                              \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lid [[thread_index_in_simdgroup]]) {                         \
    qmv_batch_impl<VPIPE_ELT, 64, 4, MAXM>(                                  \
        w, scales, biases, x, y, in_vec_size, out_vec_size, m_total,         \
        tid, simd_gid, simd_lid);                                            \
  }
// MAXM=2 is the sweet spot: at 2 rows the per-row 4-bit unpack/FMA is
// fully hidden behind the (once-read) weight load (bandwidth-bound), so
// 2 branches cost ~1 qmv step. Larger MAXM goes compute-bound; instead
// N>2 tiles the rows along grid.z (ceil(N/2) tiles) -- the extra weight
// reads are largely served from L2, which beats a compute-bound MAXM=4
// (measured 68.9 vs 58.3 tok/s aggregate at N=4). N==1 stays on plain
// affine_qmv.
VPIPE_QMV_BATCH(affine_qmv_batch_w4g64, 2)
// MAXM=4 twin for the MTP speculative VERIFY at draft depth>=2 (n=3..4): the
// MAXM=2 form tiles 3-4 rows into ceil(n/2)=2 grid.z passes, re-reading EVERY
// weight matrix from DRAM (the depth-2 cliff -- measured to ~2x the depth-1
// verify, esp. the >L2 lm_head). MAXM=4 reads weights ONCE for 3-4 rows. Goes
// more compute-bound per the MAXM=2 note above, so it's gated to the verify
// path and A/B'd; per-row math is bit-identical (token-exact).
VPIPE_QMV_BATCH(affine_qmv_batch4_w4g64, 4)
// MAXM>4 is intentionally absent: an 8-row tile blows the register budget and
// occupancy collapses (measured ~4x slower at m=5..8), a net loss for the
// L2-resident decode weights vs MAXM=2's L2-served re-reads. Batched decode
// with >=5 rows stays on MAXM=2; the MTP verify never exceeds m=4.

// 8-bit twin: qmv_batch_impl already carries the bits==8 dot (one raw byte
// per value, no nibble unpack), so this is the same macro with the impl's
// `bits` template arg set to 8. Lets the mixed-precision (OptiQ) decode/verify
// batch its 8-bit tensors at qmv bandwidth (weight read once across MAXM=2
// rows) instead of falling to the steel GEMM (~3x slower at small M) -- so a
// 2-token verify costs ~1 decode step, the goal for MTP speculative decode.
// Per-row arithmetic is bit-identical to affine_qmv_w8g64 (token-exact).
#define VPIPE_QMV_BATCH_W8(NAME, MAXM)                                       \
  kernel void NAME(                                                          \
      const device uint32_t* w      [[buffer(0)]],                           \
      const device VPIPE_ELT* scales [[buffer(1)]],                          \
      const device VPIPE_ELT* biases [[buffer(2)]],                          \
      const device VPIPE_ELT* x      [[buffer(3)]],                          \
      device VPIPE_ELT*       y      [[buffer(4)]],                          \
      constant int& in_vec_size  [[buffer(5)]],                              \
      constant int& out_vec_size [[buffer(6)]],                              \
      constant int& m_total      [[buffer(7)]],                              \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lid [[thread_index_in_simdgroup]]) {                         \
    qmv_batch_impl<VPIPE_ELT, 64, 8, MAXM>(                                  \
        w, scales, biases, x, y, in_vec_size, out_vec_size, m_total,         \
        tid, simd_gid, simd_lid);                                            \
  }
VPIPE_QMV_BATCH_W8(affine_qmv_batch_w8g64, 2)
VPIPE_QMV_BATCH_W8(affine_qmv_batch4_w8g64, 4)   // MTP verify n=3..4 (see w4)

// ===================================================================
// Batched GEMV, TALL tile with THREADGROUP-STAGED activations
// (the small-M / large-K / large-N decode-batch shape).
//
// qmv_batch_impl above holds every row's activation slice in per-thread
// registers (x_thread[MAXM][16] f32) -- at MAXM=8 that alone is 128
// registers/thread and occupancy collapses (the measured ~4x loss that
// capped the MAXM ladder at 4). Here the k-block of ALL MAXM rows is
// staged ONCE per threadgroup in threadgroup memory (MAXM*512 halfs =
// 8 KB at MAXM=8) by a cooperative copy, and each lane re-reads its
// 16-value slices from tgmem per output row -- so the register footprint
// is ~MAXM-independent (one extracted weight chunk + the result
// accumulators) and a tall tile keeps qmv-like occupancy while reading
// each weight byte ONCE for all MAXM rows (m<=MAXM -> grid.z=1).
//
// Bit-exactness with qmv_batch_impl / affine_qmv: the weight nibbles are
// extracted UNSHIFTED ((p>>4j)&0xf) instead of the shifted-mask +
// power-of-16-pre-divided-x trick; every product pairs the same two
// exactly-representable values (x/16^j is exact, nib<<4j is exact), so
// the f32 rounding of each product -- and the per-quad accumulation
// grouping, mirrored verbatim incl. load_vector's half-precision quad
// sums on full blocks vs load_vector_safe's float sums on the K tail --
// is identical. Verified byte-identical in
// metal_lm_smoke.qmv_batch_tg_matches_batch.
//
// NSG simdgroups per threadgroup (NSG*4 output rows each): more
// simdgroups amortize the cooperative stage and raise per-core occupancy
// under the 32 KB threadgroup-memory budget (8 KB/tg at MAXM=8).
// Dispatch grid = {32, Nout/4, ceil(N/MAXM)}, tg = {32, NSG, 1};
// buffer(7) = total row count (m_total).
// ===================================================================
template <typename T, int group_size, int bits, int MAXM, int NSG>
METAL_FUNC void qmv_batch_tg_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,                  // [m_total, K]
    device T* y,                        // [m_total, N]
    threadgroup T* Xs,                  // [MAXM * block_size]
    const constant int& in_vec_size,    // K
    const constant int& out_vec_size,   // N
    const constant int& m_total,        // total rows in the batch
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int packs_per_thread = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  constexpr int tg_threads = NSG * SIMD_SIZE;

  const int base_row = (int)tid.z * MAXM;
  if (base_row >= m_total) { return; }
  const int m_rows = metal::min(MAXM, m_total - base_row);

  typedef float U;
  U result[results_per_simdgroup][MAXM];
  for (int r = 0; r < results_per_simdgroup; r++) {
    for (int m = 0; m < MAXM; m++) { result[r][m] = 0; }
  }

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (NSG * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  const device uint8_t* ws = (const device uint8_t*)w +
      out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;

  const int tidx = (int)(simd_gid * SIMD_SIZE + simd_lid);

  for (int k = 0; k < in_vec_size; k += block_size) {
    // Cooperative stage of the k-block for all MAXM rows: pad rows alias
    // the last valid row (never written back), the K tail zero-fills.
    // The leading barrier orders the copy after the previous block's
    // readers (single-buffered).
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = tidx; e < MAXM * block_size; e += tg_threads) {
      const int mrow = e / block_size;
      const int col = e % block_size;
      const int mm = (mrow < m_rows) ? mrow : (m_rows - 1);
      Xs[e] = (k + col < in_vec_size)
          ? x[(size_t)(base_row + mm) * (size_t)in_vec_size + k + col]
          : (T)0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int krem = in_vec_size - k;
    // Per-row bias-term sums, mirroring load_vector's arithmetic exactly:
    // full blocks accumulate each quad in T (half) precision, the K-tail
    // block in float (load_vector_safe's form). Zeros staged past the
    // tail contribute 0 either way.
    U sum[MAXM];
    for (int m = 0; m < MAXM; m++) {
      const threadgroup T* xm =
          Xs + m * block_size + (int)simd_lid * values_per_thread;
      U s = 0;
      if (bits == 4) {
        if (krem >= block_size) {
          for (int i = 0; i < values_per_thread; i += 4) {
            s += xm[i] + xm[i + 1] + xm[i + 2] + xm[i + 3];
          }
        } else {
          for (int i = 0; i < values_per_thread; i += 4) {
            s += (U)xm[i] + (U)xm[i + 1] + (U)xm[i + 2] + (U)xm[i + 3];
          }
        }
      } else {
        for (int i = 0; i < values_per_thread; i++) { s += xm[i]; }
      }
      sum[m] = s;
    }

    // Output rows: extract this lane's weight chunk ONCE per row into
    // f32 registers (unshifted nibbles), then sweep the MAXM rows from
    // tgmem. Registers stay MAXM-independent.
    for (int row = 0; row < results_per_simdgroup; row++) {
      const device uint8_t* wl = ws + row * in_vec_size_w;
      const U s = scales[row * in_vec_size_g];
      const U b = biases[row * in_vec_size_g];
      U wv[values_per_thread];
      if (bits == 4) {
        const device uint16_t* w16 = (const device uint16_t*)wl;
        for (int i = 0; i < (values_per_thread / 4); i++) {
          const uint16_t p = w16[i];
          wv[4 * i]     = (U)(p & 0x000f);
          wv[4 * i + 1] = (U)((p >> 4) & 0x000f);
          wv[4 * i + 2] = (U)((p >> 8) & 0x000f);
          wv[4 * i + 3] = (U)(p >> 12);
        }
      } else if (bits == 8) {
        for (int i = 0; i < values_per_thread; i++) { wv[i] = (U)wl[i]; }
      }
      for (int m = 0; m < MAXM; m++) {
        const threadgroup T* xm =
            Xs + m * block_size + (int)simd_lid * values_per_thread;
        U acc = 0;
        if (bits == 4) {
          for (int i = 0; i < (values_per_thread / 4); i++) {
            acc += ((U)xm[4 * i] * wv[4 * i] +
                    (U)xm[4 * i + 1] * wv[4 * i + 1] +
                    (U)xm[4 * i + 2] * wv[4 * i + 2] +
                    (U)xm[4 * i + 3] * wv[4 * i + 3]);
          }
        } else {
          for (int i = 0; i < values_per_thread; i++) {
            acc += (U)xm[i] * wv[i];
          }
        }
        result[row][m] += s * acc + sum[m] * b;
      }
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    for (int m = 0; m < MAXM; m++) {
      const U v = simd_sum(result[row][m]);
      if (simd_lid == 0 && m < m_rows) {
        y[(base_row + m) * out_vec_size + out_row + row] = static_cast<T>(v);
      }
    }
  }
}

#define VPIPE_QMV_BATCH_TG(NAME, BITS, MAXM, NSG, VPT)                       \
  kernel void NAME(                                                          \
      const device uint32_t* w      [[buffer(0)]],                           \
      const device VPIPE_ELT* scales [[buffer(1)]],                          \
      const device VPIPE_ELT* biases [[buffer(2)]],                          \
      const device VPIPE_ELT* x      [[buffer(3)]],                          \
      device VPIPE_ELT*       y      [[buffer(4)]],                          \
      constant int& in_vec_size  [[buffer(5)]],                              \
      constant int& out_vec_size [[buffer(6)]],                              \
      constant int& m_total      [[buffer(7)]],                              \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lid [[thread_index_in_simdgroup]]) {                         \
    threadgroup VPIPE_ELT Xs[MAXM * VPT * 32];                               \
    qmv_batch_tg_impl<VPIPE_ELT, 64, BITS, MAXM, NSG>(                       \
        w, scales, biases, x, y, Xs, in_vec_size, out_vec_size, m_total,     \
        tid, simd_gid, simd_lid);                                            \
  }
// The audit candidates: MAXM=8 tall tile, 2 vs 4 simdgroups/threadgroup
// (NSG=4 halves both the per-tg stage overhead and the tgmem footprint
// per resident simdgroup). w8 twin at NSG=4 for the OptiQ mixed path.
// MEASURED (M5, qmv_batch_bandwidth_sweep): the barrier-staged pipeline
// throttles the weight stream to ~22 (NSG=2) / ~30 (NSG=4) GB/s vs the
// register kernel's ~129 GB/s per pass -- a net loss at every m. Kept
// for the A/B record; the xd variant below is the winning shape.
VPIPE_QMV_BATCH_TG(affine_qmv_batch8_tg_w4g64, 4, 8, 2, 16)
VPIPE_QMV_BATCH_TG(affine_qmv_batch8_tg4_w4g64, 4, 8, 4, 16)
VPIPE_QMV_BATCH_TG(affine_qmv_batch8_tg4_w8g64, 8, 8, 4, 8)

// ===================================================================
// Batched GEMV, TALL tile, DEVICE-reread activations ("xd"): MAXM=8
// with NO tgmem, NO barriers, and a register footprint ~equal to the
// MAXM=2 kernel's.
//
// The sweep showed each kernel has an intrinsic weight-stream rate set
// by LATENCY HIDING (resident simdgroups): ~129 GB/s at MAXM=2's ~70
// regs/thread, ~70 GB/s at MAXM=4's ~120 (the register x_thread[MAXM][16]
// is the hog), and the barrier-staged tgmem form chops the stream to
// ~22-30 GB/s regardless. This variant keeps qmv_batch_impl's exact
// streaming structure but drops the per-thread activation residency:
// per output row it extracts the weight chunk ONCE into 16 f32 registers
// (unshifted nibbles) and re-reads each row's 16-value x slice straight
// from device memory inside the m loop. x is tiny (MAXM*K*2B, ~40-155 KB)
// and every re-read hits L1/L2; the weight bytes -- the DRAM stream --
// are read ONCE for all MAXM rows with no synchronization anywhere.
// Registers: wv[16] + result[4][MAXM] + sums ~= the MAXM=2 footprint.
//
// Bit-exact with qmv_batch_impl by the same argument as the tg variant
// (exact-product pairing, per-quad grouping, half-precision full-block
// sums / float tail sums); covered by qmv_batch_tg_matches_batch.
// Dispatch grid = {32, Nout/4, ceil(m/MAXM)}, tg = {32, 2, 1}.
// ===================================================================
template <typename T, int group_size, int bits, int MAXM>
METAL_FUNC void qmv_batch_xd_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,                  // [m_total, K]
    device T* y,                        // [m_total, N]
    const constant int& in_vec_size,    // K
    const constant int& out_vec_size,   // N
    const constant int& m_total,        // total rows in the batch
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int packs_per_thread = 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const int base_row = (int)tid.z * MAXM;
  if (base_row >= m_total) { return; }
  const int m_rows = metal::min(MAXM, m_total - base_row);

  typedef float U;
  U result[results_per_simdgroup][MAXM];
  for (int r = 0; r < results_per_simdgroup; r++) {
    for (int m = 0; m < MAXM; m++) { result[r][m] = 0; }
  }

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  const device uint8_t* ws = (const device uint8_t*)w +
      out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;

  const device T* xptr[MAXM];
  for (int m = 0; m < MAXM; m++) {
    const int mm = (m < m_rows) ? m : (m_rows - 1);   // pad -> last valid row
    xptr[m] = x + (size_t)(base_row + mm) * (size_t)in_vec_size +
        (int)simd_lid * values_per_thread;
  }

  for (int k = 0; k < in_vec_size; k += block_size) {
    const int krem = in_vec_size - k;
    const bool full = (krem >= block_size);
    const int kvalid = full ? values_per_thread
        : metal::max(0, metal::min(values_per_thread,
              in_vec_size - (k + (int)simd_lid * values_per_thread)));

    // Per-row bias-term sums (load_vector's half-precision quad sums on
    // full blocks; load_vector_safe's clamped float sums on the tail).
    U sum[MAXM];
    for (int m = 0; m < MAXM; m++) {
      const device T* xm = xptr[m];
      U s = 0;
      if (bits == 4) {
        if (full) {
          for (int i = 0; i < values_per_thread; i += 4) {
            s += xm[i] + xm[i + 1] + xm[i + 2] + xm[i + 3];
          }
        } else {
          for (int i = 0; i < values_per_thread; i += 4) {
            const U v0 = (i + 0 < kvalid) ? (U)xm[i + 0] : (U)0;
            const U v1 = (i + 1 < kvalid) ? (U)xm[i + 1] : (U)0;
            const U v2 = (i + 2 < kvalid) ? (U)xm[i + 2] : (U)0;
            const U v3 = (i + 3 < kvalid) ? (U)xm[i + 3] : (U)0;
            s += v0 + v1 + v2 + v3;
          }
        }
      } else {
        for (int i = 0; i < values_per_thread; i++) {
          s += (i < kvalid) ? (U)xm[i] : (U)0;
        }
      }
      sum[m] = s;
    }

    for (int row = 0; row < results_per_simdgroup; row++) {
      const device uint8_t* wl = ws + row * in_vec_size_w;
      const U s = scales[row * in_vec_size_g];
      const U b = biases[row * in_vec_size_g];
      U wv[values_per_thread];
      if (bits == 4) {
        const device uint16_t* w16 = (const device uint16_t*)wl;
        for (int i = 0; i < (values_per_thread / 4); i++) {
          const uint16_t p = w16[i];
          wv[4 * i]     = (U)(p & 0x000f);
          wv[4 * i + 1] = (U)((p >> 4) & 0x000f);
          wv[4 * i + 2] = (U)((p >> 8) & 0x000f);
          wv[4 * i + 3] = (U)(p >> 12);
        }
      } else if (bits == 8) {
        for (int i = 0; i < values_per_thread; i++) { wv[i] = (U)wl[i]; }
      }
      for (int m = 0; m < MAXM; m++) {
        const device T* xm = xptr[m];
        U acc = 0;
        if (bits == 4) {
          if (full) {
            for (int i = 0; i < (values_per_thread / 4); i++) {
              acc += ((U)xm[4 * i] * wv[4 * i] +
                      (U)xm[4 * i + 1] * wv[4 * i + 1] +
                      (U)xm[4 * i + 2] * wv[4 * i + 2] +
                      (U)xm[4 * i + 3] * wv[4 * i + 3]);
            }
          } else {
            for (int i = 0; i < (values_per_thread / 4); i++) {
              const U v0 = (4 * i + 0 < kvalid) ? (U)xm[4 * i + 0] : (U)0;
              const U v1 = (4 * i + 1 < kvalid) ? (U)xm[4 * i + 1] : (U)0;
              const U v2 = (4 * i + 2 < kvalid) ? (U)xm[4 * i + 2] : (U)0;
              const U v3 = (4 * i + 3 < kvalid) ? (U)xm[4 * i + 3] : (U)0;
              acc += (v0 * wv[4 * i] + v1 * wv[4 * i + 1] +
                      v2 * wv[4 * i + 2] + v3 * wv[4 * i + 3]);
            }
          }
        } else {
          for (int i = 0; i < values_per_thread; i++) {
            const U v = (i < kvalid) ? (U)xm[i] : (U)0;
            acc += v * wv[i];
          }
        }
        result[row][m] += s * acc + sum[m] * b;
      }
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    for (int m = 0; m < MAXM; m++) { xptr[m] += block_size; }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    for (int m = 0; m < MAXM; m++) {
      const U v = simd_sum(result[row][m]);
      if (simd_lid == 0 && m < m_rows) {
        y[(base_row + m) * out_vec_size + out_row + row] = static_cast<T>(v);
      }
    }
  }
}

#define VPIPE_QMV_BATCH_XD(NAME, BITS, MAXM)                                 \
  kernel void NAME(                                                          \
      const device uint32_t* w      [[buffer(0)]],                           \
      const device VPIPE_ELT* scales [[buffer(1)]],                          \
      const device VPIPE_ELT* biases [[buffer(2)]],                          \
      const device VPIPE_ELT* x      [[buffer(3)]],                          \
      device VPIPE_ELT*       y      [[buffer(4)]],                          \
      constant int& in_vec_size  [[buffer(5)]],                              \
      constant int& out_vec_size [[buffer(6)]],                              \
      constant int& m_total      [[buffer(7)]],                              \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lid [[thread_index_in_simdgroup]]) {                         \
    qmv_batch_xd_impl<VPIPE_ELT, 64, BITS, MAXM>(                            \
        w, scales, biases, x, y, in_vec_size, out_vec_size, m_total,         \
        tid, simd_gid, simd_lid);                                            \
  }
VPIPE_QMV_BATCH_XD(affine_qmv_batch8_xd_w4g64, 4, 8)
VPIPE_QMV_BATCH_XD(affine_qmv_batch8_xd_w8g64, 8, 8)

// ===================================================================
// Batched GEMV, TALL tile, GROUPED-row x registers + weight packs in
// registers ("xp"): the third audit candidate.
//
// xd (above) showed the failure mode isn't only register pressure: at
// MAXM=8 the per-row device re-reads of x cost ~16 load-issues per
// weight byte and the kernel goes LOAD-ISSUE bound (~22 GB/s). Here the
// block's weight packs are loaded from device ONCE into registers
// (4 rows x 4 uint16 per lane) and x is register-resident for GROUPM
// rows at a time (load_vector, exactly as qmv_batch_impl) -- the m loop
// steps in GROUPM-row groups, re-masking the register-held packs per
// group. Per-row load issue matches MAXM=2 exactly (x loaded once per
// row per block, weights once per block); the register peak is
// x_thread[GROUPM][16] + result[4][MAXM] + packs, ~84 (GROUPM=1) /
// ~100 (GROUPM=2) instead of the register form's ~220 at MAXM=8.
// Extraction ALU grows (re-mask per group), traded for full occupancy.
//
// Bit-exact with qmv_batch_impl: reuses load_vector/load_vector_safe
// verbatim and the identical shifted-mask product/quad grouping; only
// the (associativity-free) loop nesting differs.
// Dispatch grid = {32, Nout/4, ceil(m/MAXM)}, tg = {32, 2, 1}.
// ===================================================================
// SWIGLU: fused gate|up epilogue -- the weights are the interleaved
// [2*ffn, K] gate/up (row 2g = gate g, 2g+1 = up g) and y is the halved
// [m, ffn] silu(gate)*up, exactly as qmv_batch_swiglu_impl (bit-exact).
template <typename T, int group_size, int bits, int MAXM, int GROUPM,
          bool SWIGLU = false>
METAL_FUNC void qmv_batch_xp_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,                  // [m_total, K]
    device T* y,                        // [m_total, N]
    const constant int& in_vec_size,    // K
    const constant int& out_vec_size,   // N
    const constant int& m_total,        // total rows in the batch
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int packs_per_thread = 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  // uint16 packs per lane per row (bits==4: 2 packs = 4 uint16).
  constexpr int u16_per_row = packs_per_thread * bytes_per_pack / 2;

  const int base_row = (int)tid.z * MAXM;
  if (base_row >= m_total) { return; }
  const int m_rows = metal::min(MAXM, m_total - base_row);

  typedef float U;
  U result[results_per_simdgroup][MAXM];
  for (int r = 0; r < results_per_simdgroup; r++) {
    for (int m = 0; m < MAXM; m++) { result[r][m] = 0; }
  }

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  const device uint8_t* ws = (const device uint8_t*)w +
      out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;

  const device T* xbase = x +
      (size_t)base_row * (size_t)in_vec_size +
      (int)simd_lid * values_per_thread;

  for (int k = 0; k < in_vec_size; k += block_size) {
    const int krem = in_vec_size - k;
    // Hoist the block's weight packs + scales/biases for all 4 output
    // rows into registers: ONE device read per weight byte per block,
    // re-masked per row group below.
    uint16_t wp[results_per_simdgroup][u16_per_row];
    U sv[results_per_simdgroup], bv[results_per_simdgroup];
    for (int row = 0; row < results_per_simdgroup; row++) {
      if (bits == 4) {
        const device uint16_t* w16 =
            (const device uint16_t*)(ws + row * in_vec_size_w);
        for (int j = 0; j < u16_per_row; j++) { wp[row][j] = w16[j]; }
      } else {
        const device uint16_t* w16 =
            (const device uint16_t*)(ws + row * in_vec_size_w);
        for (int j = 0; j < u16_per_row; j++) { wp[row][j] = w16[j]; }
      }
      sv[row] = scales[row * in_vec_size_g];
      bv[row] = biases[row * in_vec_size_g];
    }

    for (int g = 0; g < MAXM; g += GROUPM) {
      // x for GROUPM rows, register-resident via load_vector (identical
      // pre-division + sum arithmetic to qmv_batch_impl).
      U x_thread[GROUPM][values_per_thread];
      U sum[GROUPM];
      for (int gm = 0; gm < GROUPM; gm++) {
        const int m = g + gm;
        const int mm = (m < m_rows) ? m : (m_rows - 1);
        const device T* xm = xbase + (size_t)mm * (size_t)in_vec_size + k;
        if (krem >= block_size) {
          sum[gm] = load_vector<T, U, values_per_thread, bits>(
              xm, x_thread[gm]);
        } else {
          const int kvalid =
              in_vec_size - (k + (int)simd_lid * values_per_thread);
          sum[gm] = load_vector_safe<T, U, values_per_thread, bits>(
              xm, x_thread[gm], kvalid < 0 ? 0 : kvalid);
        }
      }
      for (int row = 0; row < results_per_simdgroup; row++) {
        for (int gm = 0; gm < GROUPM; gm++) {
          U acc = 0;
          if (bits == 4) {
            for (int i = 0; i < u16_per_row; i++) {
              const uint16_t p = wp[row][i];
              acc += (x_thread[gm][4 * i]     * (U)(p & 0x000f) +
                      x_thread[gm][4 * i + 1] * (U)(p & 0x00f0) +
                      x_thread[gm][4 * i + 2] * (U)(p & 0x0f00) +
                      x_thread[gm][4 * i + 3] * (U)(p & 0xf000));
            }
          } else if (bits == 8) {
            for (int i = 0; i < u16_per_row; i++) {
              const uint16_t p = wp[row][i];
              acc += (x_thread[gm][2 * i]     * (U)(p & 0x00ff) +
                      x_thread[gm][2 * i + 1] * (U)(p >> 8));
            }
          }
          result[row][g + gm] += sv[row] * acc + sum[gm] * bv[row];
        }
      }
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
  }

  if (SWIGLU) {
    for (int row = 0; row < results_per_simdgroup; row++) {
      for (int m = 0; m < MAXM; m++) {
        result[row][m] = simd_sum(result[row][m]);
      }
    }
    if (simd_lid == 0) {
      const int half_out = out_vec_size / 2;
      for (int m = 0; m < MAXM; m++) {
        if (m >= m_rows) { continue; }
        device T* yo = y + (base_row + m) * half_out + (out_row >> 1);
        const U g0 = result[0][m], u0 = result[1][m];
        const U g1 = result[2][m], u1 = result[3][m];
        yo[0] = static_cast<T>((g0 / (1.0f + metal::exp(-g0))) * u0);
        yo[1] = static_cast<T>((g1 / (1.0f + metal::exp(-g1))) * u1);
      }
    }
    return;
  }
  for (int row = 0; row < results_per_simdgroup; row++) {
    for (int m = 0; m < MAXM; m++) {
      const U v = simd_sum(result[row][m]);
      if (simd_lid == 0 && m < m_rows) {
        y[(base_row + m) * out_vec_size + out_row + row] = static_cast<T>(v);
      }
    }
  }
}

#define VPIPE_QMV_BATCH_XP(NAME, BITS, MAXM, GROUPM)                         \
  kernel void NAME(                                                          \
      const device uint32_t* w      [[buffer(0)]],                           \
      const device VPIPE_ELT* scales [[buffer(1)]],                          \
      const device VPIPE_ELT* biases [[buffer(2)]],                          \
      const device VPIPE_ELT* x      [[buffer(3)]],                          \
      device VPIPE_ELT*       y      [[buffer(4)]],                          \
      constant int& in_vec_size  [[buffer(5)]],                              \
      constant int& out_vec_size [[buffer(6)]],                              \
      constant int& m_total      [[buffer(7)]],                              \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lid [[thread_index_in_simdgroup]]) {                         \
    qmv_batch_xp_impl<VPIPE_ELT, 64, BITS, MAXM, GROUPM>(                    \
        w, scales, biases, x, y, in_vec_size, out_vec_size, m_total,         \
        tid, simd_gid, simd_lid);                                            \
  }
VPIPE_QMV_BATCH_XP(affine_qmv_batch8_xp1_w4g64, 4, 8, 1)
VPIPE_QMV_BATCH_XP(affine_qmv_batch8_xp2_w4g64, 4, 8, 2)
// MAXM=4 twin: same ALU/byte as the register-resident batch4 kernel but
// ~70 regs instead of ~120 (x_thread[2][16] + result[4][4] + packs) --
// the occupancy that caps batch4 at ~70 GB/s/pass comes back. Candidate
// replacement for the m=3..4 tier (MTP depth-2 verify's >SLC lm_head).
VPIPE_QMV_BATCH_XP(affine_qmv_batch4_xp_w4g64, 4, 4, 2)
// 8-bit twins (the OptiQ mixed path's per-tensor w8 verify/decode): the
// impl's bits==8 dot is the raw-byte form, bit-identical to
// affine_qmv_batch_w8g64 per row. Same tile shapes as the w4 ladder.
VPIPE_QMV_BATCH_XP(affine_qmv_batch4_xp_w8g64, 8, 4, 2)
VPIPE_QMV_BATCH_XP(affine_qmv_batch8_xp2_w8g64, 8, 8, 2)

// Fused-SwiGLU xp twins (the gate|up MLP GEMV -- the LARGEST decode
// weight stream): interleaved gate/up weights, halved y, silu epilogue;
// per-row arithmetic identical to the plain xp form, epilogue identical
// to qmv_batch_swiglu_impl -> bit-exact vs affine_qmv_batch_swiglu_*.
#define VPIPE_QMV_BATCH_XP_SWIGLU(NAME, BITS, MAXM, GROUPM)                  \
  kernel void NAME(                                                          \
      const device uint32_t* w      [[buffer(0)]],                           \
      const device VPIPE_ELT* scales [[buffer(1)]],                          \
      const device VPIPE_ELT* biases [[buffer(2)]],                          \
      const device VPIPE_ELT* x      [[buffer(3)]],                          \
      device VPIPE_ELT*       y      [[buffer(4)]],                          \
      constant int& in_vec_size  [[buffer(5)]],                              \
      constant int& out_vec_size [[buffer(6)]],   /* N = 2*ffn */            \
      constant int& m_total      [[buffer(7)]],                              \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lid [[thread_index_in_simdgroup]]) {                         \
    qmv_batch_xp_impl<VPIPE_ELT, 64, BITS, MAXM, GROUPM, true>(              \
        w, scales, biases, x, y, in_vec_size, out_vec_size, m_total,         \
        tid, simd_gid, simd_lid);                                            \
  }
VPIPE_QMV_BATCH_XP_SWIGLU(affine_qmv_batch4_xp_swiglu_w4g64, 4, 4, 2)
VPIPE_QMV_BATCH_XP_SWIGLU(affine_qmv_batch8_xp2_swiglu_w4g64, 4, 8, 2)

// ===================================================================
// Batched GEMV, TALL tile, HOISTED dequant ("xh" / "xh16"): the last
// two rungs of the m=7..8 one-pass ladder.
//
// xp2 (above) re-extracts the register-held weight packs once per
// 2-row group (4x per block) and pays load_vector's power-of-16
// pre-division per row; its measured ~43 GB/s pass sits near the
// scalar-f32 ALU ceiling (8 FMAs per weight nibble at MAXM=8). xh
// extracts each block's weights ONCE into registers as UNSHIFTED
// nibbles (no pre-division anywhere: raw-x * true-nibble pairs the
// same exactly-representable values, so f32 products round identically
// -- still BIT-IDENTICAL to qmv_batch_impl) and streams x one row at a
// time (transient 16 registers).
//
// xh16 (COMPUTE_HALF) additionally does the products + in-quad sums in
// HALF precision (double-rate on Apple GPUs, and the hoisted weight
// registers halve), accumulating quads into the f32 block accumulator.
// All values are exact in f16 (nibbles 0..15, raw f16 x, products
// bounded ~1e3); only the 4-term quad sums round in half, so logits
// differ from the reference in the last bits: NOT bit-identical --
// gate it on the greedy token-exact bar (qwen_batched_decode_token_
// exact), like the matmul2d prefill path.
// Dispatch grid = {32, Nout/4, ceil(m/MAXM)}, tg = {32, 2, 1}.
// ===================================================================
template <typename T, int group_size, int bits, int MAXM, bool COMPUTE_HALF>
METAL_FUNC void qmv_batch_xh_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,                  // [m_total, K]
    device T* y,                        // [m_total, N]
    const constant int& in_vec_size,    // K
    const constant int& out_vec_size,   // N
    const constant int& m_total,        // total rows in the batch
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int packs_per_thread = 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  constexpr int u16_per_row = packs_per_thread * bytes_per_pack / 2;

  const int base_row = (int)tid.z * MAXM;
  if (base_row >= m_total) { return; }
  const int m_rows = metal::min(MAXM, m_total - base_row);

  typedef float U;
  // Weight element type for the dot: half (double-rate) or float.
  typedef typename metal::conditional<COMPUTE_HALF, half, float>::type W;
  U result[results_per_simdgroup][MAXM];
  for (int r = 0; r < results_per_simdgroup; r++) {
    for (int m = 0; m < MAXM; m++) { result[r][m] = 0; }
  }

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  const device uint8_t* ws = (const device uint8_t*)w +
      out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;

  const device T* xbase = x +
      (size_t)base_row * (size_t)in_vec_size +
      (int)simd_lid * values_per_thread;

  for (int k = 0; k < in_vec_size; k += block_size) {
    const int krem = in_vec_size - k;
    const bool full = (krem >= block_size);
    const int kvalid = full ? values_per_thread
        : metal::max(0, metal::min(values_per_thread,
              in_vec_size - (k + (int)simd_lid * values_per_thread)));

    // Hoist + dequantize the block's weights ONCE: unshifted nibbles in
    // W registers (exact in both half and float), plus scales/biases.
    W wv[results_per_simdgroup][values_per_thread];
    U sv[results_per_simdgroup], bv[results_per_simdgroup];
    for (int row = 0; row < results_per_simdgroup; row++) {
      if (bits == 4) {
        const device uint16_t* w16 =
            (const device uint16_t*)(ws + row * in_vec_size_w);
        for (int i = 0; i < u16_per_row; i++) {
          const uint16_t p = w16[i];
          wv[row][4 * i]     = (W)(p & 0x000f);
          wv[row][4 * i + 1] = (W)((p >> 4) & 0x000f);
          wv[row][4 * i + 2] = (W)((p >> 8) & 0x000f);
          wv[row][4 * i + 3] = (W)(p >> 12);
        }
      } else if (bits == 8) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        for (int i = 0; i < values_per_thread; i++) {
          wv[row][i] = (W)wl[i];
        }
      }
      sv[row] = scales[row * in_vec_size_g];
      bv[row] = biases[row * in_vec_size_g];
    }

    for (int m = 0; m < MAXM; m++) {
      const int mm = (m < m_rows) ? m : (m_rows - 1);
      const device T* xm = xbase + (size_t)mm * (size_t)in_vec_size + k;
      // Raw x, transient registers; sum mirrors load_vector /
      // load_vector_safe exactly (half quad sums on full blocks).
      W xt[values_per_thread];
      U sum = 0;
      if (full) {
        if (bits == 4) {
          for (int i = 0; i < values_per_thread; i += 4) {
            sum += xm[i] + xm[i + 1] + xm[i + 2] + xm[i + 3];
            xt[i]     = (W)xm[i];
            xt[i + 1] = (W)xm[i + 1];
            xt[i + 2] = (W)xm[i + 2];
            xt[i + 3] = (W)xm[i + 3];
          }
        } else {
          for (int i = 0; i < values_per_thread; i++) {
            sum += xm[i];
            xt[i] = (W)xm[i];
          }
        }
      } else if (bits == 4) {
        // load_vector_safe's tail: clamped values, QUAD-grouped float sums.
        for (int i = 0; i < values_per_thread; i += 4) {
          const U v0 = (i + 0 < kvalid) ? (U)xm[i + 0] : (U)0;
          const U v1 = (i + 1 < kvalid) ? (U)xm[i + 1] : (U)0;
          const U v2 = (i + 2 < kvalid) ? (U)xm[i + 2] : (U)0;
          const U v3 = (i + 3 < kvalid) ? (U)xm[i + 3] : (U)0;
          sum += v0 + v1 + v2 + v3;
          xt[i]     = (W)v0;
          xt[i + 1] = (W)v1;
          xt[i + 2] = (W)v2;
          xt[i + 3] = (W)v3;
        }
      } else {
        for (int i = 0; i < values_per_thread; i++) {
          const U v = (i < kvalid) ? (U)xm[i] : (U)0;
          sum += v;
          xt[i] = (W)v;
        }
      }
      for (int row = 0; row < results_per_simdgroup; row++) {
        U acc = 0;
        for (int i = 0; i < (values_per_thread / 4); i++) {
          const W q = xt[4 * i]     * wv[row][4 * i] +
                      xt[4 * i + 1] * wv[row][4 * i + 1] +
                      xt[4 * i + 2] * wv[row][4 * i + 2] +
                      xt[4 * i + 3] * wv[row][4 * i + 3];
          acc += (U)q;
        }
        result[row][m] += sv[row] * acc + sum * bv[row];
      }
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    for (int m = 0; m < MAXM; m++) {
      const U v = simd_sum(result[row][m]);
      if (simd_lid == 0 && m < m_rows) {
        y[(base_row + m) * out_vec_size + out_row + row] = static_cast<T>(v);
      }
    }
  }
}

#define VPIPE_QMV_BATCH_XH(NAME, BITS, MAXM, CHALF)                          \
  kernel void NAME(                                                          \
      const device uint32_t* w      [[buffer(0)]],                           \
      const device VPIPE_ELT* scales [[buffer(1)]],                          \
      const device VPIPE_ELT* biases [[buffer(2)]],                          \
      const device VPIPE_ELT* x      [[buffer(3)]],                          \
      device VPIPE_ELT*       y      [[buffer(4)]],                          \
      constant int& in_vec_size  [[buffer(5)]],                              \
      constant int& out_vec_size [[buffer(6)]],                              \
      constant int& m_total      [[buffer(7)]],                              \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lid [[thread_index_in_simdgroup]]) {                         \
    qmv_batch_xh_impl<VPIPE_ELT, 64, BITS, MAXM, CHALF>(                     \
        w, scales, biases, x, y, in_vec_size, out_vec_size, m_total,         \
        tid, simd_gid, simd_lid);                                            \
  }
VPIPE_QMV_BATCH_XH(affine_qmv_batch8_xh_w4g64, 4, 8, false)
VPIPE_QMV_BATCH_XH(affine_qmv_batch8_xh16_w4g64, 4, 8, true)

// ====================================================================
// MoE GROUPED expert GEMV (Qwen3.5-MoE prefill optimization).
//
// The pair-batched gather kernels above re-read an expert's full weight
// matrix once PER routed token (npair = M*top_k GEMVs). For large M (prefill)
// that is the dominant cost. Here the (token,expert) pairs are first sorted
// into per-expert blocks padded to MAXM rows (on-GPU counting sort); this
// grouped GEMV then processes one tile (MAXM rows of ONE expert) per grid.z,
// reading that expert's weight slab ONCE across the tile -> ~MAXM-fold fewer
// weight reads. Mirrors qmv_batch_impl (weight read once across MAXM rows)
// plus: (1) the expert slab offset from tile2e[tile], (2) input rows gathered
// via srow[] (GATHER, gate|up reads the shared hidden) or the slot itself
// (!GATHER, down reads the sorted activations), (3) srow<0 marks padding.
// Per-row arithmetic is identical to the plain kernels -> token-exact.
template <typename T, int group_size, int bits, int MAXM, bool SWIGLU,
          bool GATHER>
METAL_FUNC void qmv_grouped_impl(
    const device uint32_t* w,           // [E, N, K/8]
    const device T* scales,             // [E, N, K/64]
    const device T* biases,             // [E, N, K/64]
    const device T* x,                  // [*, K] input rows
    device T* y,                        // [npad, N]  (SWIGLU: [npad, N/2])
    const device int* srow,             // [npad] input row per slot (<0 = pad)
    const device int* tile2e,           // [ntiles] expert per tile (<0 = none)
    const constant int& in_vec_size,    // K
    const constant int& out_vec_size,   // N
    uint3 tid, uint simd_gid, uint simd_lid) {
  constexpr int packs_per_thread = 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const int tile = (int)tid.z;
  const int e = tile2e[tile];
  if (e < 0) { return; }                 // unused padding tile
  const int base_slot = tile * MAXM;

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  // expert slab strides (rows N * per-row entries)
  w      += (uint)e * (uint)out_vec_size * (uint)(in_vec_size_w / bytes_per_pack);
  scales += (uint)e * (uint)out_vec_size * (uint)in_vec_size_g;
  biases += (uint)e * (uint)out_vec_size * (uint)in_vec_size_g;

  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  const device uint8_t* ws = (const device uint8_t*)w +
      out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;

  typedef float U;
  thread U x_thread[MAXM][values_per_thread];
  thread U result[results_per_simdgroup][MAXM];
  for (int r = 0; r < results_per_simdgroup; r++) {
    for (int m = 0; m < MAXM; m++) { result[r][m] = 0; }
  }

  bool valid[MAXM];
  const device T* xptr[MAXM];
  for (int m = 0; m < MAXM; m++) {
    const int rr = srow[base_slot + m];
    valid[m] = rr >= 0;
    const int row = GATHER ? (valid[m] ? rr : 0) : (base_slot + m);
    xptr[m] = x + row * in_vec_size + simd_lid * values_per_thread;
  }

  for (int k = 0; k < in_vec_size; k += block_size) {
    const int krem = in_vec_size - k;
    const int kvalid = in_vec_size - (k + (int)simd_lid * values_per_thread);
    U sum[MAXM];
    for (int m = 0; m < MAXM; m++) {
      sum[m] = (krem >= block_size)
          ? load_vector<T, U, values_per_thread, bits>(xptr[m], x_thread[m])
          : load_vector_safe<T, U, values_per_thread, bits>(
                xptr[m], x_thread[m], kvalid < 0 ? 0 : kvalid);
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
      const device uint8_t* wl = ws + row * in_vec_size_w;
      const U s = scales[row * in_vec_size_g];
      const U b = biases[row * in_vec_size_g];
      U acc[MAXM] = {0};
      if (bits == 4) {
        // 4-bit: x_thread is power-of-16 pre-divided (load_vector) so the
        // masked nibble (shifted) reconstructs the magnitude -- weight read
        // once across MAXM rows.
        const device uint16_t* w16 = (const device uint16_t*)wl;
        for (int i = 0; i < (values_per_thread / 4); i++) {
          const uint16_t p = w16[i];
          const U w0 = (U)(p & 0x000f), w1 = (U)(p & 0x00f0);
          const U w2 = (U)(p & 0x0f00), w3 = (U)(p & 0xf000);
          for (int m = 0; m < MAXM; m++) {
            acc[m] += x_thread[m][4 * i] * w0 + x_thread[m][4 * i + 1] * w1 +
                      x_thread[m][4 * i + 2] * w2 + x_thread[m][4 * i + 3] * w3;
          }
        }
      } else {   // bits == 8: weights are raw bytes; x_thread is raw x.
        for (int i = 0; i < values_per_thread; i++) {
          const U wv = (U)wl[i];
          for (int m = 0; m < MAXM; m++) { acc[m] += x_thread[m][i] * wv; }
        }
      }
      for (int m = 0; m < MAXM; m++) {
        result[row][m] += s * acc[m] + sum[m] * b;
      }
    }
    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    for (int m = 0; m < MAXM; m++) { xptr[m] += block_size; }
  }

  if (!SWIGLU) {
    for (int row = 0; row < results_per_simdgroup; row++) {
      for (int m = 0; m < MAXM; m++) {
        const U v = simd_sum(result[row][m]);
        if (simd_lid == 0 && valid[m]) {
          y[(base_slot + m) * out_vec_size + out_row + row] = static_cast<T>(v);
        }
      }
    }
  } else {
    for (int m = 0; m < MAXM; m++) {
      const U g0 = simd_sum(result[0][m]);
      const U u0 = simd_sum(result[1][m]);
      const U g1 = simd_sum(result[2][m]);
      const U u1 = simd_sum(result[3][m]);
      if (simd_lid == 0 && valid[m]) {
        device T* yo = y + (base_slot + m) * (out_vec_size / 2) + (out_row >> 1);
        yo[0] = static_cast<T>((g0 / (1.0f + metal::exp(-g0))) * u0);
        yo[1] = static_cast<T>((g1 / (1.0f + metal::exp(-g1))) * u1);
      }
    }
  }
}

#define VPIPE_GROUPED(NAME, BITS, MAXM, SW, GA)                              \
  kernel void NAME(                                                          \
      const device uint32_t* w       [[buffer(0)]],                          \
      const device VPIPE_ELT* scales [[buffer(1)]],                          \
      const device VPIPE_ELT* biases [[buffer(2)]],                          \
      const device VPIPE_ELT* x      [[buffer(3)]],                          \
      device VPIPE_ELT*       y      [[buffer(4)]],                          \
      constant int& in_vec_size  [[buffer(5)]],                              \
      constant int& out_vec_size [[buffer(6)]],                              \
      const device int* srow     [[buffer(7)]],                              \
      const device int* tile2e   [[buffer(8)]],                              \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lid [[thread_index_in_simdgroup]]) {                         \
    qmv_grouped_impl<VPIPE_ELT, 64, BITS, MAXM, SW, GA>(                     \
        w, scales, biases, x, y, srow, tile2e, in_vec_size, out_vec_size,    \
        tid, simd_gid, simd_lid);                                            \
  }
// MAXM=2: weight read once per 2 sorted same-expert rows (vs per token in the
// pair path) -- the bandwidth-bound sweet spot (MAXM>=4 goes compute-bound on
// the 4-bit unpack, losing the weight-read win; see the batch note above).
// gate|up gathers the hidden via srow (GATHER); down reads the sorted
// activations by slot (!GATHER). w8 twins: same layout, bits=8 (the impl's
// inner unpack + slab stride are bit-parameterized).
VPIPE_GROUPED(affine_grouped_swiglu_w4g64, 4, 2, true, true)
VPIPE_GROUPED(affine_grouped_down_w4g64, 4, 2, false, false)
VPIPE_GROUPED(affine_grouped_swiglu_w8g64, 8, 2, true, true)
VPIPE_GROUPED(affine_grouped_down_w8g64, 8, 2, false, false)

// group_size=32 twin of VPIPE_QMV_BATCH (GGUF q4_0): identical to the
// macro above but the impl group_size template arg is 32. The g64 macro
// hardcodes group 64, so this parallel macro sets group 32 without
// touching its behavior.
#define VPIPE_QMV_BATCH_G32(NAME, MAXM)                                      \
  kernel void NAME(                                                          \
      const device uint32_t* w      [[buffer(0)]],                           \
      const device VPIPE_ELT* scales [[buffer(1)]],                          \
      const device VPIPE_ELT* biases [[buffer(2)]],                          \
      const device VPIPE_ELT* x      [[buffer(3)]],                          \
      device VPIPE_ELT*       y      [[buffer(4)]],                          \
      constant int& in_vec_size  [[buffer(5)]],                              \
      constant int& out_vec_size [[buffer(6)]],                              \
      constant int& m_total      [[buffer(7)]],                              \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lid [[thread_index_in_simdgroup]]) {                         \
    qmv_batch_impl<VPIPE_ELT, 32, 4, MAXM>(                                  \
        w, scales, biases, x, y, in_vec_size, out_vec_size, m_total,         \
        tid, simd_gid, simd_lid);                                            \
  }
VPIPE_QMV_BATCH_G32(affine_qmv_batch_w4g32, 2)

// Batched fused SwiGLU MLP GEMV (Qwen): y[N,ffn] = silu(gate)*up, with the
// gate/up weights interleaved by output feature ([2*ffn, K], row 2g=gate g,
// 2g+1=up g) exactly like affine_qmv_swiglu_w4g64. M-row generalization.
template <typename T, int group_size, int bits, int MAXM>
METAL_FUNC void qmv_batch_swiglu_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,                        // [m_total, out_vec_size/2]
    const constant int& in_vec_size,    // K
    const constant int& out_vec_size,   // N = 2*ffn
    const constant int& m_total,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int packs_per_thread = 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const int base_row = (int)tid.z * MAXM;
  if (base_row >= m_total) { return; }
  const int m_rows = metal::min(MAXM, m_total - base_row);

  typedef float U;
  thread U x_thread[MAXM][values_per_thread];
  thread U result[results_per_simdgroup][MAXM];
  for (int r = 0; r < results_per_simdgroup; r++) {
    for (int m = 0; m < MAXM; m++) { result[r][m] = 0; }
  }

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  const device uint8_t* ws = (const device uint8_t*)w +
      out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;

  const device T* xptr[MAXM];
  for (int m = 0; m < MAXM; m++) {
    const int mm = (m < m_rows) ? m : (m_rows - 1);
    xptr[m] = x + (base_row + mm) * in_vec_size + simd_lid * values_per_thread;
  }

  for (int k = 0; k < in_vec_size; k += block_size) {
    // Last block may be partial (in_vec_size % block_size != 0): mask the
    // tail lanes past in_vec_size so they contribute 0 (see load_vector_safe).
    const int krem = in_vec_size - k;
    const int kvalid = in_vec_size - (k + (int)simd_lid * values_per_thread);
    U sum[MAXM];
    for (int m = 0; m < MAXM; m++) {
      sum[m] = (krem >= block_size)
          ? load_vector<T, U, values_per_thread, bits>(xptr[m], x_thread[m])
          : load_vector_safe<T, U, values_per_thread, bits>(
                xptr[m], x_thread[m], kvalid < 0 ? 0 : kvalid);
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
      const device uint8_t* wl = ws + row * in_vec_size_w;
      const U s = scales[row * in_vec_size_g];
      const U b = biases[row * in_vec_size_g];
      U acc[MAXM] = {0};
      if (bits == 4) {
        const device uint16_t* w16 = (const device uint16_t*)wl;
        for (int i = 0; i < (values_per_thread / 4); i++) {
          const uint16_t p = w16[i];
          const U w0 = (U)(p & 0x000f), w1 = (U)(p & 0x00f0);
          const U w2 = (U)(p & 0x0f00), w3 = (U)(p & 0xf000);
          for (int m = 0; m < MAXM; m++) {
            acc[m] += x_thread[m][4 * i] * w0 + x_thread[m][4 * i + 1] * w1 +
                      x_thread[m][4 * i + 2] * w2 + x_thread[m][4 * i + 3] * w3;
          }
        }
      } else if (bits == 8) {
        for (int i = 0; i < values_per_thread; i++) {
          const U wv = (U)wl[i];
          for (int m = 0; m < MAXM; m++) { acc[m] += x_thread[m][i] * wv; }
        }
      }
      for (int m = 0; m < MAXM; m++) { result[row][m] += s * acc[m] + sum[m] * b; }
    }
    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    for (int m = 0; m < MAXM; m++) { xptr[m] += block_size; }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    for (int m = 0; m < MAXM; m++) { result[row][m] = simd_sum(result[row][m]); }
  }
  if (simd_lid == 0) {
    const int half_out = out_vec_size / 2;
    for (int m = 0; m < MAXM; m++) {
      if (m >= m_rows) { continue; }
      device T* yo = y + (base_row + m) * half_out + (out_row >> 1);
      const U g0 = result[0][m], u0 = result[1][m];
      const U g1 = result[2][m], u1 = result[3][m];
      yo[0] = static_cast<T>((g0 / (1.0f + metal::exp(-g0))) * u0);
      yo[1] = static_cast<T>((g1 / (1.0f + metal::exp(-g1))) * u1);
    }
  }
}

#define VPIPE_QMV_BATCH_SWIGLU(NAME, MAXM)                                   \
  kernel void NAME(                                                          \
      const device uint32_t* w      [[buffer(0)]],                           \
      const device VPIPE_ELT* scales [[buffer(1)]],                          \
      const device VPIPE_ELT* biases [[buffer(2)]],                          \
      const device VPIPE_ELT* x      [[buffer(3)]],                          \
      device VPIPE_ELT*       y      [[buffer(4)]],                          \
      constant int& in_vec_size  [[buffer(5)]],                              \
      constant int& out_vec_size [[buffer(6)]],   /* N = 2*ffn */            \
      constant int& m_total      [[buffer(7)]],                              \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lid [[thread_index_in_simdgroup]]) {                         \
    qmv_batch_swiglu_impl<VPIPE_ELT, 64, 4, MAXM>(                           \
        w, scales, biases, x, y, in_vec_size, out_vec_size, m_total,         \
        tid, simd_gid, simd_lid);                                            \
  }
VPIPE_QMV_BATCH_SWIGLU(affine_qmv_batch_swiglu_w4g64, 2)
VPIPE_QMV_BATCH_SWIGLU(affine_qmv_batch4_swiglu_w4g64, 4)   // MTP verify n=3..4

// Batched fused GeGLU MLP GEMV (Gemma-4): identical to the swiglu batch but
// the activation is gelu_pytorch_tanh (matches affine_qmv_geglu_w4g64).
template <typename T, int group_size, int bits, int MAXM>
METAL_FUNC void qmv_batch_geglu_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,                        // [m_total, out_vec_size/2]
    const constant int& in_vec_size,    // K
    const constant int& out_vec_size,   // N = 2*ffn
    const constant int& m_total,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int packs_per_thread = 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const int base_row = (int)tid.z * MAXM;
  if (base_row >= m_total) { return; }
  const int m_rows = metal::min(MAXM, m_total - base_row);

  typedef float U;
  thread U x_thread[MAXM][values_per_thread];
  thread U result[results_per_simdgroup][MAXM];
  for (int r = 0; r < results_per_simdgroup; r++) {
    for (int m = 0; m < MAXM; m++) { result[r][m] = 0; }
  }

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  const device uint8_t* ws = (const device uint8_t*)w +
      out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;

  const device T* xptr[MAXM];
  for (int m = 0; m < MAXM; m++) {
    const int mm = (m < m_rows) ? m : (m_rows - 1);
    xptr[m] = x + (base_row + mm) * in_vec_size + simd_lid * values_per_thread;
  }

  for (int k = 0; k < in_vec_size; k += block_size) {
    // Last block may be partial (in_vec_size % block_size != 0): mask the
    // tail lanes past in_vec_size so they contribute 0 (see load_vector_safe).
    const int krem = in_vec_size - k;
    const int kvalid = in_vec_size - (k + (int)simd_lid * values_per_thread);
    U sum[MAXM];
    for (int m = 0; m < MAXM; m++) {
      sum[m] = (krem >= block_size)
          ? load_vector<T, U, values_per_thread, bits>(xptr[m], x_thread[m])
          : load_vector_safe<T, U, values_per_thread, bits>(
                xptr[m], x_thread[m], kvalid < 0 ? 0 : kvalid);
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
      const device uint8_t* wl = ws + row * in_vec_size_w;
      const U s = scales[row * in_vec_size_g];
      const U b = biases[row * in_vec_size_g];
      U acc[MAXM] = {0};
      if (bits == 4) {
        const device uint16_t* w16 = (const device uint16_t*)wl;
        for (int i = 0; i < (values_per_thread / 4); i++) {
          const uint16_t p = w16[i];
          const U w0 = (U)(p & 0x000f), w1 = (U)(p & 0x00f0);
          const U w2 = (U)(p & 0x0f00), w3 = (U)(p & 0xf000);
          for (int m = 0; m < MAXM; m++) {
            acc[m] += x_thread[m][4 * i] * w0 + x_thread[m][4 * i + 1] * w1 +
                      x_thread[m][4 * i + 2] * w2 + x_thread[m][4 * i + 3] * w3;
          }
        }
      } else if (bits == 8) {
        for (int i = 0; i < values_per_thread; i++) {
          const U wv = (U)wl[i];
          for (int m = 0; m < MAXM; m++) { acc[m] += x_thread[m][i] * wv; }
        }
      }
      for (int m = 0; m < MAXM; m++) { result[row][m] += s * acc[m] + sum[m] * b; }
    }
    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    for (int m = 0; m < MAXM; m++) { xptr[m] += block_size; }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    for (int m = 0; m < MAXM; m++) { result[row][m] = simd_sum(result[row][m]); }
  }
  if (simd_lid == 0) {
    const int half_out = out_vec_size / 2;
    const U k0 = 0.7978845608028654f;
    for (int m = 0; m < MAXM; m++) {
      if (m >= m_rows) { continue; }
      device T* yo = y + (base_row + m) * half_out + (out_row >> 1);
      const U g0 = result[0][m], u0 = result[1][m];
      const U g1 = result[2][m], u1 = result[3][m];
      const U t0 = metal::precise::tanh(k0 * (g0 + 0.044715f * g0 * g0 * g0));
      const U t1 = metal::precise::tanh(k0 * (g1 + 0.044715f * g1 * g1 * g1));
      yo[0] = static_cast<T>(0.5f * g0 * (1.0f + t0) * u0);
      yo[1] = static_cast<T>(0.5f * g1 * (1.0f + t1) * u1);
    }
  }
}

#define VPIPE_QMV_BATCH_GEGLU(NAME, MAXM)                                    \
  kernel void NAME(                                                          \
      const device uint32_t* w      [[buffer(0)]],                           \
      const device VPIPE_ELT* scales [[buffer(1)]],                          \
      const device VPIPE_ELT* biases [[buffer(2)]],                          \
      const device VPIPE_ELT* x      [[buffer(3)]],                          \
      device VPIPE_ELT*       y      [[buffer(4)]],                          \
      constant int& in_vec_size  [[buffer(5)]],                              \
      constant int& out_vec_size [[buffer(6)]],   /* N = 2*ffn */            \
      constant int& m_total      [[buffer(7)]],                              \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lid [[thread_index_in_simdgroup]]) {                         \
    qmv_batch_geglu_impl<VPIPE_ELT, 64, 4, MAXM>(                            \
        w, scales, biases, x, y, in_vec_size, out_vec_size, m_total,         \
        tid, simd_gid, simd_lid);                                            \
  }
VPIPE_QMV_BATCH_GEGLU(affine_qmv_batch_geglu_w4g64, 2)

// group_size=32 twin of VPIPE_QMV_BATCH_GEGLU (GGUF q4_0): identical to
// the macro above but the impl group_size template arg is 32. The g64
// macro hardcodes group 64, so this parallel macro sets group 32 without
// touching its behavior.
#define VPIPE_QMV_BATCH_GEGLU_G32(NAME, MAXM)                                \
  kernel void NAME(                                                          \
      const device uint32_t* w      [[buffer(0)]],                           \
      const device VPIPE_ELT* scales [[buffer(1)]],                          \
      const device VPIPE_ELT* biases [[buffer(2)]],                          \
      const device VPIPE_ELT* x      [[buffer(3)]],                          \
      device VPIPE_ELT*       y      [[buffer(4)]],                          \
      constant int& in_vec_size  [[buffer(5)]],                              \
      constant int& out_vec_size [[buffer(6)]],   /* N = 2*ffn */            \
      constant int& m_total      [[buffer(7)]],                              \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lid [[thread_index_in_simdgroup]]) {                         \
    qmv_batch_geglu_impl<VPIPE_ELT, 32, 4, MAXM>(                            \
        w, scales, biases, x, y, in_vec_size, out_vec_size, m_total,         \
        tid, simd_gid, simd_lid);                                            \
  }
VPIPE_QMV_BATCH_GEGLU_G32(affine_qmv_batch_geglu_w4g32, 2)

// ===================================================================
// 8-bit (group 64) entry points -- identical dispatch geometry to the
// 4-bit variants above; only the `bits` template arg differs (it drives
// pack_factor / bytes_per_pack and the load_vector/qdot byte path). The
// 8-bit checkpoint (e.g. Qwen3-ASR) stores one byte per weight value.
// ===================================================================
kernel void affine_qmv_w8g64(
    const device uint32_t* w            [[buffer(0)]],
    const device VPIPE_ELT*     scales       [[buffer(1)]],
    const device VPIPE_ELT*     biases       [[buffer(2)]],
    const device VPIPE_ELT*     x            [[buffer(3)]],
    device VPIPE_ELT*           y            [[buffer(4)]],
    constant int&          in_vec_size  [[buffer(5)]],
    constant int&          out_vec_size [[buffer(6)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  qmv_fast_impl<VPIPE_ELT, 64, 8, false>(
      w, scales, biases, x, y, /*residual=*/nullptr,
      in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// group_size=32 twin of affine_qmv_w8g64 (8-bit, GGUF q8 group 32). Only
// the template group_size differs.
kernel void affine_qmv_w8g32(
    const device uint32_t* w            [[buffer(0)]],
    const device VPIPE_ELT*     scales       [[buffer(1)]],
    const device VPIPE_ELT*     biases       [[buffer(2)]],
    const device VPIPE_ELT*     x            [[buffer(3)]],
    device VPIPE_ELT*           y            [[buffer(4)]],
    constant int&          in_vec_size  [[buffer(5)]],
    constant int&          out_vec_size [[buffer(6)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  qmv_fast_impl<VPIPE_ELT, 32, 8, false>(
      w, scales, biases, x, y, /*residual=*/nullptr,
      in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// ---- Bandwidth-tuning variants of affine_qmv_w8g64 (RPS/PPT/NSG) -------
// Same args + dispatch convention, but grid.y = N/RPS and tg = (32,NSG,1)
// (rows per threadgroup = RPS*NSG). For sweeping the best GEMV bandwidth
// per GPU; the winner is wired into the decode (see metal-qwen-model.cc).
#define VPIPE_QMV8_VARIANT(NAME, RPS, PPT, NSG)                              \
  kernel void affine_qmv_w8g64_##NAME(                                      \
      const device uint32_t* w      [[buffer(0)]],                          \
      const device VPIPE_ELT* scales [[buffer(1)]],                         \
      const device VPIPE_ELT* biases [[buffer(2)]],                         \
      const device VPIPE_ELT* x      [[buffer(3)]],                         \
      device VPIPE_ELT*       y      [[buffer(4)]],                         \
      constant int& in_vec_size  [[buffer(5)]],                             \
      constant int& out_vec_size [[buffer(6)]],                             \
      uint3 tid [[threadgroup_position_in_grid]],                           \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                     \
      uint simd_lid [[thread_index_in_simdgroup]]) {                        \
    qmv_fast_impl<VPIPE_ELT, 64, 8, false, RPS, PPT, NSG>(                  \
        w, scales, biases, x, y, nullptr, in_vec_size, out_vec_size, tid,   \
        simd_gid, simd_lid);                                                \
  }
// r8p2 (more ILP) kept as the bench's reference alt -- it does NOT beat
// the 4/2/2 baseline on M4 (the GEMV is bandwidth-bound, not latency-
// bound), confirming MLX's qmv_fast tiling is already optimal here. The
// r4p4/r8p4/_s4/r16p2 variants were swept and dropped (equal-or-slower).
VPIPE_QMV8_VARIANT(r8p2, 8, 2, 2)

// RPS/PPT/NSG tuning variant for the 4-bit GEMV (the e4b decode FFN + proj
// path). MLX uses 4/2/2 everywhere; metal_compute_qmv.w4_bandwidth_sweep
// retested the tile on M5 and CONFIRMED 4/2/2 optimal: the GEMV is bandwidth-
// bound at M5's ~128 GB/s DRAM ceiling (the 26 MB gate/up weight; smaller
// weights read ~145 GB/s only because they fit M5's SLC and get cached across
// the bench's reps -- an artifact, not the real per-token DRAM read). r6p2/
// r4p4/r8p4/r4p2s4 were all equal-or-slower and dropped; r8p2 kept as the
// bench's reference alt (mirrors the w8 path).
#define VPIPE_QMV4_VARIANT(NAME, RPS, PPT, NSG)                              \
  kernel void affine_qmv_w4g64_##NAME(                                      \
      const device uint32_t* w      [[buffer(0)]],                          \
      const device VPIPE_ELT* scales [[buffer(1)]],                         \
      const device VPIPE_ELT* biases [[buffer(2)]],                         \
      const device VPIPE_ELT* x      [[buffer(3)]],                         \
      device VPIPE_ELT*       y      [[buffer(4)]],                         \
      constant int& in_vec_size  [[buffer(5)]],                             \
      constant int& out_vec_size [[buffer(6)]],                             \
      uint3 tid [[threadgroup_position_in_grid]],                           \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                     \
      uint simd_lid [[thread_index_in_simdgroup]]) {                        \
    qmv_fast_impl<VPIPE_ELT, 64, 4, false, RPS, PPT, NSG>(                  \
        w, scales, biases, x, y, nullptr, in_vec_size, out_vec_size, tid,   \
        simd_gid, simd_lid);                                                \
  }
VPIPE_QMV4_VARIANT(r8p2, 8, 2, 2)

kernel void affine_qmv_w8g64_add(
    const device uint32_t* w            [[buffer(0)]],
    const device VPIPE_ELT*     scales       [[buffer(1)]],
    const device VPIPE_ELT*     biases       [[buffer(2)]],
    const device VPIPE_ELT*     x            [[buffer(3)]],
    device VPIPE_ELT*           y            [[buffer(4)]],
    constant int&          in_vec_size  [[buffer(5)]],
    constant int&          out_vec_size [[buffer(6)]],
    const device VPIPE_ELT*     residual     [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  qmv_fast_impl<VPIPE_ELT, 64, 8, true>(
      w, scales, biases, x, y, residual,
      in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

kernel void affine_qmv_swiglu_w8g64(
    const device uint32_t* w            [[buffer(0)]],
    const device VPIPE_ELT*     scales       [[buffer(1)]],
    const device VPIPE_ELT*     biases       [[buffer(2)]],
    const device VPIPE_ELT*     x            [[buffer(3)]],
    device VPIPE_ELT*           y            [[buffer(4)]],
    constant int&          in_vec_size  [[buffer(5)]],
    constant int&          out_vec_size [[buffer(6)]],   // N = 2*ffn
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  qmv_swiglu_impl<VPIPE_ELT, 64, 8>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      tid, simd_gid, simd_lid);
}

// Fused GeGLU MLP GEMV, 8-bit weights (Gemma-4 12B: gate/up are 8-bit). Same
// dispatch geometry as affine_qmv_geglu_w4g64; only the `bits` arg differs.
kernel void affine_qmv_geglu_w8g64(
    const device uint32_t* w            [[buffer(0)]],
    const device VPIPE_ELT*     scales       [[buffer(1)]],
    const device VPIPE_ELT*     biases       [[buffer(2)]],
    const device VPIPE_ELT*     x            [[buffer(3)]],
    device VPIPE_ELT*           y            [[buffer(4)]],
    constant int&          in_vec_size  [[buffer(5)]],
    constant int&          out_vec_size [[buffer(6)]],   // N = 2*ffn
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  qmv_geglu_impl<VPIPE_ELT, 64, 8>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      tid, simd_gid, simd_lid);
}

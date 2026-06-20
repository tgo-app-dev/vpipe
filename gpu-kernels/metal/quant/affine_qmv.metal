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

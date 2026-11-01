// sage_quant.metal -- the prologue an INT8 QK^T needs: per-block int8
// quantization of Q and K, and the key mean that makes it accurate.
//
// SageAttention (arXiv:2410.02367) runs QK^T in int8 with ONE scale per
// block -- per query block for Q, per key block for K -- which is the
// granularity a flash kernel already tiles at, so the dequant of a score
// tile is a single scalar. These kernels produce that.
//
// THE MEAN IS NOT AN OPTIMISATION, IT IS WHAT MAKES IT WORK. K is
// quantized as K - mean(K) over TOKENS: a key's outlier is not variation
// between tokens but a large bias shared by all of them, and subtracting
// it spends the int8 range on the signal. It is exact rather than
// approximate -- subtracting a per-channel mean shifts every score in a
// row by the same q.mean, and softmax does not see a per-row shift, so
// nothing is added back. MEASURED on the method
// (sage_attention.per_block_int8_needs_the_smoothing): with smoothing
// the cosine against an f64 reference holds 0.99991 however large the
// bias; without it, it falls away as the bias grows.
//
// Q IS NOT SMOOTHED. Only the key side carries the shared bias the
// argument is about, and a shift of Q would NOT cancel -- it would move
// each score by q_shift.k_j, which varies along the row.
//
// Quantized WHERE THE KERNEL READS: [B*H, L, D] tightly packed, which is
// head-major and not necessarily the layout the f16 tensors have (those
// may be a run inside a fused projection). So the source carries a row
// stride and the destination does not.

#include <metal_stdlib>

using namespace metal;

#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif

// Per-channel sum of K over a CHUNK of tokens: partial[bh, ch, c].
//
// TWO STAGES, because one is a reduction with no parallelism in it. A
// thread per channel walking the whole sequence is D * BH threads --
// 1024 at video geometry -- each doing 20036 dependent adds, and it
// MEASURED 4.33 ms of a 6.0 ms prologue. Split over chunks it is 65536
// threads and the second stage sums 64 numbers per channel.
//
//   0:k (VPIPE_ELT) 1:partial (float [BH, CH, D]) 2:L 3:D 4:row_stride
//   5:base_off 6:chunks 7:head_stride
// grid (D, BH, CH) THREADS; threadgroup (D, 1, 1). One thread per
// channel, so a threadgroup reads one whole row at a time.
//
// 64-BIT OFFSETS throughout: a long sequence's keys -- a MiniMax-H3
// refine's [56, rows, 128] past 599,186 rows, or its fused [rows, 21504]
// projection past 199,728 -- pass 2^32 elements, and in 32 bits the
// later heads were read from, and quantized into, the earlier ones.
//
// HEAD STRIDE IS ITS OWN NUMBER, not L * row_stride. That identity
// holds for a head-major [H, L, D] tensor and fails for every DiT in
// this tree, because they read q/k IN PLACE inside a fused [rows, 3*I]
// projection: there a row is 3*I apart and a head is one head_dim
// apart, so the two strides are unrelated. Deriving one from the other
// would read head h at row h*L -- inside the buffer, wrong, and silent.
kernel void sage_k_sum_chunk(
    const device VPIPE_ELT* k           [[buffer(0)]],
    device float*           partial     [[buffer(1)]],
    constant int&           L           [[buffer(2)]],
    constant int&           D           [[buffer(3)]],
    constant int&           row_stride  [[buffer(4)]],
    constant uint&          base_off    [[buffer(5)]],
    constant int&           chunks      [[buffer(6)]],
    constant int&           head_stride [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint  c   [[thread_index_in_threadgroup]])
{
  if ((int)c >= D) { return; }
  const int bh = (int)tid.y;
  const int ch = (int)tid.z;
  const int per = (L + chunks - 1) / chunks;
  const int t0 = ch * per, t1 = min(L, t0 + per);
  const device VPIPE_ELT* src =
      k + (ulong)base_off + (ulong)bh * (ulong)head_stride;
  // fp32 accumulation: the sum of 100k values around 1 is ~1e5, where
  // f16 would have stopped resolving the addend long before.
  float acc = 0.0f;
  for (int t = t0; t < t1; ++t) {
    acc += (float)src[(ulong)t * (ulong)row_stride + c];
  }
  partial[((uint)bh * chunks + ch) * D + c] = acc;
}

// ...and the chunk partials into the mean.
//
//   0:partial 1:mean (float [BH, D]) 2:L 3:D 4:chunks
// grid (D, BH, 1) THREADS; threadgroup (D, 1, 1).
kernel void sage_k_mean_reduce(
    const device float* partial [[buffer(0)]],
    device float*       mean    [[buffer(1)]],
    constant int&       L       [[buffer(2)]],
    constant int&       D       [[buffer(3)]],
    constant int&       chunks  [[buffer(4)]],
    uint2 tid [[threadgroup_position_in_grid]],
    uint  c   [[thread_index_in_threadgroup]])
{
  if ((int)c >= D) { return; }
  const int bh = (int)tid.y;
  float acc = 0.0f;
  for (int ch = 0; ch < chunks; ++ch) {
    acc += partial[((uint)bh * chunks + ch) * D + c];
  }
  mean[(uint)bh * D + c] = acc / (float)max(L, 1);
}

// Per-BLOCK int8 quantization, optionally over x - mean.
//
//   0:x (VPIPE_ELT) 1:mean (float [BH, D] or unused) 2:q (int8 [BH, L, D])
//   3:scale (float [BH, NB]) 4:L 5:D 6:row_stride 7:base_off 8:block
//   9:sub_mean (int) 10:head_stride
// grid (256, BH, NB) THREADS; threadgroup (256, 1, 1). One threadgroup
// per (head, block): absmax over the whole [block, D] tile, then the
// tile written back quantized.
//
// The SOURCE carries a row stride, a head stride and a base offset
// because it is usually a run inside a fused projection; the
// DESTINATION is always tightly packed [BH, L, D], because that is what
// the flash kernel indexes. See sage_k_sum_chunk on why the head stride
// is passed rather than derived.
kernel void sage_quant_block(
    const device VPIPE_ELT* x           [[buffer(0)]],
    const device float*     mean        [[buffer(1)]],
    device char*            q           [[buffer(2)]],
    device float*           scale       [[buffer(3)]],
    constant int&           L           [[buffer(4)]],
    constant int&           D           [[buffer(5)]],
    constant int&           row_stride  [[buffer(6)]],
    constant uint&          base_off    [[buffer(7)]],
    constant int&           block       [[buffer(8)]],
    constant int&           sub_mean    [[buffer(9)]],
    constant int&           head_stride [[buffer(10)]],
    uint3 tid  [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]])
{
  constexpr int kThreads = 256;
  // The staging tile's bound: the largest block either operand uses (a
  // query block, 64) by the largest head dim (128). 16 KB as VPIPE_ELT.
  constexpr int kMaxBlock = 64, kMaxD = 128;
  threadgroup float red[kThreads];

  const int bh = (int)tid.y;
  const int b  = (int)tid.z;
  const int r0 = b * block;
  const int r1 = min(L, r0 + block);
  if (r0 >= L) { return; }

  const device VPIPE_ELT* src =
      x + (ulong)base_off + (ulong)bh * (ulong)head_stride;
  const device float* mn = mean + (uint)bh * D;
  device char* dst = q + (ulong)bh * (ulong)L * (ulong)D;

  // ONE DEVICE READ. The absmax has to see the whole block before a
  // single element can be written, so the naive shape reads the source
  // twice; staged here it is read once and the write pass comes out of
  // threadgroup memory. Staged as VPIPE_ELT and not float because the
  // value is on its way to INT8 -- f16 is already far finer than the
  // quantizer that follows, and float would be 32 KB at block 64.
  threadgroup VPIPE_ELT stage[kMaxBlock * kMaxD];
  const int n = (r1 - r0) * D;
  float am = 0.0f;
  for (int i = (int)lid; i < n; i += kThreads) {
    const int r = r0 + i / D, c = i - (i / D) * D;
    float v = (float)src[(ulong)r * (ulong)row_stride + c];
    if (sub_mean != 0) { v -= mn[c]; }
    stage[i] = (VPIPE_ELT)v;
    am = max(am, fabs(v));
  }
  red[lid] = am;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int s = kThreads / 2; s > 0; s >>= 1) {
    if ((int)lid < s) { red[lid] = max(red[lid], red[lid + s]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float amax = red[0];
  // A zero block quantizes to zeros with a zero scale, which dequantizes
  // back to zero -- the same answer, and no division by nothing.
  const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
  if (lid == 0) { scale[(uint)bh * ((L + block - 1) / block) + b] =
                      amax / 127.0f; }

  for (int i = (int)lid; i < n; i += kThreads) {
    const int r = r0 + i / D, c = i - (i / D) * D;
    const float qv = rint((float)stage[i] * inv);
    dst[(uint)r * D + c] = (char)clamp(qv, -127.0f, 127.0f);
  }
  // NOTHING IS WRITTEN PAST L, and the tail needs nothing: the flash
  // kernel reads a ragged last block through load_rows, which stops at
  // the row limit. Zeroing the remainder here would be a write past the
  // [BH, L, D] destination for the sake of rows nobody reads.
}

// Q quantized PER TOKEN: one scale per row, its absmax over the head dim
// -- the int8 GEMMs' Per Token tier's activation side, and what the
// flash kernel's int8 P*V reads (function constant 310). A row is the
// unit a query's score is made in, so a row with small values keeps its
// own resolution instead of its block's largest row's.
//
// A source over its mean (`sub_mean`: K) as the per-block kernel takes.
//
//   0:x (VPIPE_ELT) 1:q (int8 [BH, L, D]) 2:scale (float [BH, L])
//   3:L 4:D 5:row_stride 6:base_off 7:head_stride 8:mean (float [BH, D])
//   9:sub_mean
// grid (256, BH, ceil(L/64)) THREADS; threadgroup (256, 1, 1): 64 rows a
// threadgroup, four threads a row, each ceil(D/4) channels of it (the
// last thread's run cut at D). D <= 128 -- prepare() refuses more.
kernel void sage_quant_token(
    const device VPIPE_ELT* x           [[buffer(0)]],
    device char*            q           [[buffer(1)]],
    device float*           scale       [[buffer(2)]],
    constant int&           L           [[buffer(3)]],
    constant int&           D           [[buffer(4)]],
    constant int&           row_stride  [[buffer(5)]],
    constant uint&          base_off    [[buffer(6)]],
    constant int&           head_stride [[buffer(7)]],
    const device float*     mean        [[buffer(8)]],
    constant int&           sub_mean    [[buffer(9)]],
    uint3 tid  [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]])
{
  threadgroup float red[256];
  const int bh = (int)tid.y;
  const int r = (int)tid.z * 64 + (int)lid / 4;
  const int part = (int)lid % 4;
  const int per = (D + 3) / 4;
  const int c0 = part * per;
  const int n = max(0, min(per, D - c0));
  const device VPIPE_ELT* src =
      x + (ulong)base_off + (ulong)bh * (ulong)head_stride;
  float am = 0.0f;
  float v[32];
  if (r < L) {
    for (int c = 0; c < n; ++c) {
      v[c] = (float)src[(ulong)r * (ulong)row_stride + (ulong)(c0 + c)];
      if (sub_mean != 0) { v[c] -= mean[(uint)bh * D + c0 + c]; }
      am = max(am, fabs(v[c]));
    }
  }
  red[lid] = am;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int b4 = (int)lid & ~3;
  am = max(max(red[b4], red[b4 + 1]), max(red[b4 + 2], red[b4 + 3]));
  if (r >= L) { return; }
  const float inv = am > 0.0f ? 127.0f / am : 0.0f;
  if (part == 0) { scale[(uint)bh * L + r] = am / 127.0f; }
  device char* dst =
      q + ((ulong)bh * (ulong)L + (ulong)r) * (ulong)D + (ulong)c0;
  for (int c = 0; c < n; ++c) {
    dst[c] = (char)clamp(rint(v[c] * inv), -127.0f, 127.0f);
  }
}

// V's SCALE for the INT8 P*V: one per channel over the WHOLE sequence,
// its absmax over V minus its mean (the mean is sage_k_sum_chunk /
// sage_k_mean_reduce run over V). One a channel and not one a block: the
// kernel holds the product in int32 across key blocks, so V's scale may
// not change along them. Two stages, as the mean: chunks, then a reduce.
//
// sage_v_absmax_chunk: 0:v 1:mean (float [BH, D]) 2:partial (float
//   [BH, CH, D]) 3:L 4:D 5:row_stride 6:base_off 7:chunks 8:head_stride
//   grid (D, BH, CH) THREADS; threadgroup (D, 1, 1)
// sage_v_absmax_reduce: 0:partial 1:scale (float [BH, D]: absmax / 127)
//   2:D 3:chunks   grid (D, BH, 1); threadgroup (D, 1, 1)
kernel void sage_v_absmax_chunk(
    const device VPIPE_ELT* v           [[buffer(0)]],
    const device float*     mean        [[buffer(1)]],
    device float*           partial     [[buffer(2)]],
    constant int&           L           [[buffer(3)]],
    constant int&           D           [[buffer(4)]],
    constant int&           row_stride  [[buffer(5)]],
    constant uint&          base_off    [[buffer(6)]],
    constant int&           chunks      [[buffer(7)]],
    constant int&           head_stride [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint  c   [[thread_index_in_threadgroup]])
{
  if ((int)c >= D) { return; }
  const int bh = (int)tid.y, ch = (int)tid.z;
  const int per = (L + chunks - 1) / chunks;
  const int t0 = ch * per, t1 = min(L, t0 + per);
  const device VPIPE_ELT* src =
      v + (ulong)base_off + (ulong)bh * (ulong)head_stride;
  const float mu = mean[(uint)bh * D + c];
  float am = 0.0f;
  for (int t = t0; t < t1; ++t) {
    am = max(am, fabs((float)src[(ulong)t * (ulong)row_stride + c] - mu));
  }
  partial[((uint)bh * chunks + ch) * D + c] = am;
}

kernel void sage_v_absmax_reduce(
    const device float* partial [[buffer(0)]],
    device float*       scale   [[buffer(1)]],
    constant int&       D       [[buffer(2)]],
    constant int&       chunks  [[buffer(3)]],
    uint2 tid [[threadgroup_position_in_grid]],
    uint  c   [[thread_index_in_threadgroup]])
{
  if ((int)c >= D) { return; }
  const int bh = (int)tid.y;
  float am = 0.0f;
  for (int ch = 0; ch < chunks; ++ch) {
    am = max(am, partial[((uint)bh * chunks + ch) * D + c]);
  }
  scale[(uint)bh * D + c] = am / 127.0f;
}

// V quantized for the INT8 P*V (the flash kernel's function constant 308):
// V minus its mean, at the scale sage_v_absmax_reduce made, as int8 plus
// 128 -- uint8, which the kernel multiplies against P's uint8 -- eight
// tokens a threadgroup.
//   0:v (VPIPE_ELT) 1:mean (float [BH, D]) 2:scale (float [BH, D])
//   3:q (uint8 [BH, L, D]) 4:L 5:D 6:row_stride 7:base_off 8:head_stride
//   grid (D, BH, ceil(L / 8)) THREADS; threadgroup (D, 1, 1)
kernel void sage_quant_v_channel_u8(
    const device VPIPE_ELT* v           [[buffer(0)]],
    const device float*     mean        [[buffer(1)]],
    const device float*     scale       [[buffer(2)]],
    device uchar*           q           [[buffer(3)]],
    constant int&           L           [[buffer(4)]],
    constant int&           D           [[buffer(5)]],
    constant int&           row_stride  [[buffer(6)]],
    constant uint&          base_off    [[buffer(7)]],
    constant int&           head_stride [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint  c   [[thread_index_in_threadgroup]])
{
  if ((int)c >= D) { return; }
  const int bh = (int)tid.y;
  const int t0 = (int)tid.z * 8, t1 = min(L, t0 + 8);
  const device VPIPE_ELT* src =
      v + (ulong)base_off + (ulong)bh * (ulong)head_stride;
  const float s = scale[(uint)bh * D + c];
  const float inv = s > 0.0f ? 1.0f / s : 0.0f;
  const float mu = mean[(uint)bh * D + c];
  for (int t = t0; t < t1; ++t) {
    const float x = (float)src[(ulong)t * (ulong)row_stride + c] - mu;
    const int qi = (int)clamp(rint(x * inv), -127.0f, 127.0f);
    q[((ulong)bh * (ulong)L + (ulong)t) * (ulong)D + c] =
        (uchar)(qi + 128);
  }
}

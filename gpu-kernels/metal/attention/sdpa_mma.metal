// sdpa_mma.metal -- matrix-core (M5+) flash attention for head_dim 256.
// QK^T and P*V run on the hardware matrix units (Metal 4 matmul2d); the
// online-softmax state (running max m, sum l, and the BQ x D output O) is
// kept in THREADGROUP memory so the per-query-row rescale-by-correction is
// a trivial indexed op -- the cooperative_tensor layout matmul2d returns is
// opaque and ill-suited to flash's row-wise softmax, so only the two
// matmuls use it. K/V live in the PAGED pool; within a page a BK-key block
// is contiguous, so matmul2d reads it straight from device (no tg staging,
// which keeps the D=256 threadgroup budget under 32 KB).
//
// Same buffer/param contract as sdpa_paged_qtile_f16 (drop-in):
//   0:q[Hq,n_q,D] 1:kpool 2:vpool 3:out[Hq,n_q,D]
//   4:scale 5:D 6:Hq 7:Hkv 8:n_q 9:q_offset 10:page_tokens
//   11:n_pages 12:page_table(int[n_pages*3] = {pid,nvalid,gstart})
// grid (threads): {SA_SG*32, Hq, ceil(n_q/BQ)}, tg {SA_SG*32,1,1}.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif
#define SA_BQ 16
#define SA_BK 16
#define SA_SG 4
#define SA_THREADS (SA_SG * 32)

#if defined(__HAVE_TENSOR__)

// Head-dim-parameterized (SA_D template arg): instantiated for 256 (Qwen3.5)
// and 128 (Llama-3 / the Krea-2 Qwen3-VL text encoder). The body is fully
// SA_D-driven; only the Of[SA_BQ*SA_D] tg budget scales (8 KB @128, 16 KB
// @256, both under 32 KB). matmul2d descriptors take SA_D as a constexpr
// (a non-type template parameter), so each width JITs its own tile shapes.
template <int SA_D>
[[kernel]] void sdpa_mma_tmpl(
    const device VPIPE_ELT* q          [[buffer(0)]],
    const device VPIPE_ELT* kpool      [[buffer(1)]],
    const device VPIPE_ELT* vpool      [[buffer(2)]],
    device VPIPE_ELT*       out        [[buffer(3)]],
    constant float& scale       [[buffer(4)]],
    constant int&   D           [[buffer(5)]],
    constant int&   Hq          [[buffer(6)]],
    constant int&   Hkv         [[buffer(7)]],
    constant int&   n_q         [[buffer(8)]],
    constant int&   q_offset    [[buffer(9)]],
    constant int&   page_tokens [[buffer(10)]],
    constant int&   n_pages     [[buffer(11)]],
    const device int* page_table[[buffer(12)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]])
{
  (void)D;
  const int h    = (int)tgid.y;
  const int tile = (int)tgid.z;
  const int q0   = tile * SA_BQ;
  const int kvh  = h / (Hq / Hkv);
  const uint page_stride = (uint)Hkv * page_tokens * SA_D;
  const uint head_off    = (uint)kvh * page_tokens * SA_D;

  threadgroup VPIPE_ELT Ss[SA_BQ * SA_BK];   // S / P tile
  threadgroup float Of[SA_BQ * SA_D];        // running output (f32)
  threadgroup float mrow[SA_BQ], lrow[SA_BQ];

  for (int e = (int)lid; e < SA_BQ * SA_D; e += SA_THREADS) { Of[e] = 0.0f; }
  for (int e = (int)lid; e < SA_BQ; e += SA_THREADS) {
    mrow[e] = -INFINITY; lrow[e] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const device VPIPE_ELT* qh = q + ((int64_t)h * n_q + q0) * SA_D;
  using TQ = tensor<device VPIPE_ELT, dextents<int32_t,2>, tensor_inline>;
  using TP = tensor<threadgroup VPIPE_ELT, dextents<int32_t,2>, tensor_inline>;
  using TO = tensor<threadgroup float, dextents<int32_t,2>, tensor_inline>;
  constexpr auto qk = matmul2d_descriptor(
      SA_BQ, SA_BK, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<qk, execution_simdgroups<SA_SG>> opQK;
  constexpr auto pv = matmul2d_descriptor(
      SA_BQ, SA_D, SA_BK, false, false, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<pv, execution_simdgroups<SA_SG>> opPV;
  TQ tQ(const_cast<device VPIPE_ELT*>(qh), dextents<int32_t,2>(SA_D, SA_BQ));

  const int tile_qpos_max = q_offset + min(q0 + SA_BQ - 1, n_q - 1);

  for (int pg = 0; pg < n_pages; ++pg) {
    const int pid    = page_table[pg * 3 + 0];
    const int nvalid = page_table[pg * 3 + 1];
    const int gstart = page_table[pg * 3 + 2];
    if (gstart > tile_qpos_max) { break; }
    const device VPIPE_ELT* kbase = kpool + (uint)pid * page_stride + head_off;
    const device VPIPE_ELT* vbase = vpool + (uint)pid * page_stride + head_off;
    for (int bs = 0; bs < nvalid; bs += SA_BK) {
      if (gstart + bs > tile_qpos_max) { break; }
      const int bk = min(SA_BK, nvalid - bs);
      // S = Q @ K_block^T  (matmul reads SA_BK rows; rows >= bk are masked
      // below, so reading slightly past nvalid within the page is harmless
      // -- it is in-bounds page memory, never used).
      TQ tK(const_cast<device VPIPE_ELT*>(kbase + (int64_t)bs * SA_D),
            dextents<int32_t,2>(SA_D, SA_BK));
      auto cS = opQK.get_destination_cooperative_tensor<TQ, TQ, VPIPE_ELT>();
      opQK.run(tQ, tK, cS);
      TP tS(Ss, dextents<int32_t,2>(SA_BK, SA_BQ));
      cS.store(tS);
      threadgroup_barrier(mem_flags::mem_threadgroup);

      for (int r = (int)lid; r < SA_BQ; r += SA_THREADS) {
        const int qpos = q_offset + q0 + r;
        const bool qok = (q0 + r) < n_q;
        float mloc = mrow[r];
        for (int j = 0; j < SA_BK; ++j) {
          const int kpos = gstart + bs + j;
          const bool ok = qok && j < bk && kpos <= qpos;
          const float s = ok ? float(Ss[r * SA_BK + j]) * scale : -INFINITY;
          Ss[r * SA_BK + j] = (VPIPE_ELT)s;
          mloc = max(mloc, s);
        }
        const float corr = exp(mrow[r] - mloc);
        float lloc = lrow[r] * corr;
        for (int j = 0; j < SA_BK; ++j) {
          const float s = float(Ss[r * SA_BK + j]);
          const float p = (s == -INFINITY) ? 0.0f : exp(s - mloc);
          Ss[r * SA_BK + j] = (VPIPE_ELT)p;
          lloc += p;
        }
        for (int dd = 0; dd < SA_D; ++dd) { Of[r * SA_D + dd] *= corr; }
        mrow[r] = mloc; lrow[r] = lloc;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      TP tP(Ss, dextents<int32_t,2>(SA_BK, SA_BQ));
      TQ tV(const_cast<device VPIPE_ELT*>(vbase + (int64_t)bs * SA_D),
            dextents<int32_t,2>(SA_D, SA_BK));
      TO tO(Of, dextents<int32_t,2>(SA_D, SA_BQ));
      opPV.run(tP, tV, tO);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  for (int e = (int)lid; e < SA_BQ * SA_D; e += SA_THREADS) {
    const int r = e / SA_D, dd = e % SA_D;
    if (q0 + r < n_q) {
      const float inv = (lrow[r] > 0.0f) ? 1.0f / lrow[r] : 0.0f;
      out[((int64_t)h * n_q + (q0 + r)) * SA_D + dd] = (VPIPE_ELT)(Of[e] * inv);
    }
  }
}

// Concrete entry points (the MLX-steel instantiate pattern): head_dim 256
// (Qwen3.5) + head_dim 128 (Llama-3 / Krea-2 Qwen3-VL text encoder).
template [[host_name("sdpa_mma_f16")]] [[kernel]]
decltype(sdpa_mma_tmpl<256>) sdpa_mma_tmpl<256>;
template [[host_name("sdpa_mma_d128_f16")]] [[kernel]]
decltype(sdpa_mma_tmpl<128>) sdpa_mma_tmpl<128>;

// Contiguous-KV matrix-core flash attention for Gemma-4 PREFILL, head_dim 512
// (the GLOBAL/full-attention layers -- the O(n^2) term that dominates 12B
// prefill, and the ONLY part still on the ALU simdgroup_matrix path). Same
// drop-in contract as the simdgroup_matrix sdpa_causal_flash_f16 it replaces on
// M5: K/V are CONTIGUOUS [Hkv, kv_stride, D] (no paging); QK^T and P*V run on
// the hardware matrix units (matmul2d), the online softmax stays in tg memory.
// BQ=8 so Of[BQ*512] f32 == 16 KB leaves room under the 32 KB tg budget (BQ=16
// would be 32 KB with no slack for matmul2d's scratch). Full-causal only
// (window==0); the over-read of the last BK block past T_kv is masked (caller
// pads the KV cap, like the flash kernel).
//   0:q[Hq,n_q,D] 1:k 2:v[Hkv,kv_stride,D] 3:out[Hq,n_q,D] 4:scale 5:T_kv 6:D
//   7:Hq 8:Hkv 9:n_q 10:q_offset 11:kv_stride 12:window 13:ring_cap
// grid {SA2_SG*32, Hq, ceil(n_q/SA2_BQ)}, tg {SA2_SG*32,1,1}.
#define SA2_BQ 8
#define SA2_BK 16
#define SA2_D  512
#define SA2_SG 4
#define SA2_THREADS (SA2_SG * 32)
kernel void sdpa_causal_mma2_d512_f16(
    const device VPIPE_ELT* q     [[buffer(0)]],
    const device VPIPE_ELT* k     [[buffer(1)]],
    const device VPIPE_ELT* v     [[buffer(2)]],
    device VPIPE_ELT*       out   [[buffer(3)]],
    constant float& scale     [[buffer(4)]],
    constant int&   T_kv      [[buffer(5)]],
    constant int&   D         [[buffer(6)]],
    constant int&   Hq        [[buffer(7)]],
    constant int&   Hkv       [[buffer(8)]],
    constant int&   n_q       [[buffer(9)]],
    constant int&   q_offset  [[buffer(10)]],
    constant int&   kv_stride [[buffer(11)]],
    constant int&   window    [[buffer(12)]],
    constant int&   ring_cap  [[buffer(13)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]])
{
  (void)D; (void)window; (void)ring_cap;
  const int h    = (int)tgid.y;
  const int q0   = (int)tgid.z * SA2_BQ;
  const int kvh  = h / (Hq / Hkv);
  const device VPIPE_ELT* kbase = k + (int64_t)kvh * kv_stride * SA2_D;
  const device VPIPE_ELT* vbase = v + (int64_t)kvh * kv_stride * SA2_D;

  threadgroup VPIPE_ELT Ss[SA2_BQ * SA2_BK];
  threadgroup float Of[SA2_BQ * SA2_D];
  threadgroup float mrow[SA2_BQ], lrow[SA2_BQ], corr_s[SA2_BQ];

  for (int e = (int)lid; e < SA2_BQ * SA2_D; e += SA2_THREADS) { Of[e] = 0.0f; }
  for (int e = (int)lid; e < SA2_BQ; e += SA2_THREADS) {
    mrow[e] = -INFINITY; lrow[e] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const device VPIPE_ELT* qh = q + ((int64_t)h * n_q + q0) * SA2_D;
  using TQ = tensor<device VPIPE_ELT, dextents<int32_t,2>, tensor_inline>;
  using TP = tensor<threadgroup VPIPE_ELT, dextents<int32_t,2>, tensor_inline>;
  using TO = tensor<threadgroup float, dextents<int32_t,2>, tensor_inline>;
  constexpr auto qk = matmul2d_descriptor(
      SA2_BQ, SA2_BK, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<qk, execution_simdgroups<SA2_SG>> opQK;
  constexpr auto pv = matmul2d_descriptor(
      SA2_BQ, SA2_D, SA2_BK, false, false, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<pv, execution_simdgroups<SA2_SG>> opPV;
  TQ tQ(const_cast<device VPIPE_ELT*>(qh), dextents<int32_t,2>(SA2_D, SA2_BQ));

  const int tile_qpos_max = q_offset + min(q0 + SA2_BQ - 1, n_q - 1);
  const int last = min(T_kv - 1, tile_qpos_max);

  for (int bs = 0; bs <= last; bs += SA2_BK) {
    const int bk = min(SA2_BK, T_kv - bs);
    // S = Q @ K_block^T (reads SA2_BK rows; rows >= bk masked below).
    TQ tK(const_cast<device VPIPE_ELT*>(kbase + (int64_t)bs * SA2_D),
          dextents<int32_t,2>(SA2_D, SA2_BK));
    auto cS = opQK.get_destination_cooperative_tensor<TQ, TQ, VPIPE_ELT>();
    opQK.run(tQ, tK, cS);
    TP tS(Ss, dextents<int32_t,2>(SA2_BK, SA2_BQ));
    cS.store(tS);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int r = (int)lid; r < SA2_BQ; r += SA2_THREADS) {
      const int qpos = q_offset + q0 + r;
      const bool qok = (q0 + r) < n_q;
      float mloc = mrow[r];
      for (int j = 0; j < SA2_BK; ++j) {
        const int kpos = bs + j;
        const bool ok = qok && j < bk && kpos <= qpos;
        const float s = ok ? float(Ss[r * SA2_BK + j]) * scale : -INFINITY;
        Ss[r * SA2_BK + j] = (VPIPE_ELT)s;
        mloc = max(mloc, s);
      }
      const float corr = exp(mrow[r] - mloc);
      float lloc = lrow[r] * corr;
      for (int j = 0; j < SA2_BK; ++j) {
        const float s = float(Ss[r * SA2_BK + j]);
        const float p = (s == -INFINITY) ? 0.0f : exp(s - mloc);
        Ss[r * SA2_BK + j] = (VPIPE_ELT)p;
        lloc += p;
      }
      corr_s[r] = corr;
      mrow[r] = mloc; lrow[r] = lloc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Rescale the running PV accumulator across ALL threads (was BQ threads
    // each looping D -- the serial hot spot: 8 threads x 512 mults/block).
    // SA2_D is a power of two so /,% fold to shifts.
    for (int e = (int)lid; e < SA2_BQ * SA2_D; e += SA2_THREADS) {
      Of[e] *= corr_s[e / SA2_D];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    TP tP(Ss, dextents<int32_t,2>(SA2_BK, SA2_BQ));
    TQ tV(const_cast<device VPIPE_ELT*>(vbase + (int64_t)bs * SA2_D),
          dextents<int32_t,2>(SA2_D, SA2_BK));
    TO tO(Of, dextents<int32_t,2>(SA2_D, SA2_BQ));
    opPV.run(tP, tV, tO);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (int e = (int)lid; e < SA2_BQ * SA2_D; e += SA2_THREADS) {
    const int r = e / SA2_D, dd = e % SA2_D;
    if (q0 + r < n_q) {
      const float inv = (lrow[r] > 0.0f) ? 1.0f / lrow[r] : 0.0f;
      out[((int64_t)h * n_q + (q0 + r)) * SA2_D + dd] = (VPIPE_ELT)(Of[e] * inv);
    }
  }
}

// head_dim 256 + sliding-WINDOW sibling for Gemma-4's SLIDING prefill layers in
// the no-wrap ring regime. Same matrix-core matmul2d QK/PV; BQ=16 (Of[16*256]
// f32 == 16 KB). Adds the window lower bound kpos >= qpos-window+1 (the scan
// starts at the window base) on top of the causal mask; a BK block that a row's
// window excludes entirely is a no-op (corr==1). Linear KV addressing -- the
// caller gates on no-wrap so logical key == physical slot.
#define SA3_BQ 16
#define SA3_BK 16
#define SA3_D  256
#define SA3_SG 4
#define SA3_THREADS (SA3_SG * 32)
kernel void sdpa_causal_mma2_d256_f16(
    const device VPIPE_ELT* q     [[buffer(0)]],
    const device VPIPE_ELT* k     [[buffer(1)]],
    const device VPIPE_ELT* v     [[buffer(2)]],
    device VPIPE_ELT*       out   [[buffer(3)]],
    constant float& scale     [[buffer(4)]],
    constant int&   T_kv      [[buffer(5)]],
    constant int&   D         [[buffer(6)]],
    constant int&   Hq        [[buffer(7)]],
    constant int&   Hkv       [[buffer(8)]],
    constant int&   n_q       [[buffer(9)]],
    constant int&   q_offset  [[buffer(10)]],
    constant int&   kv_stride [[buffer(11)]],
    constant int&   window    [[buffer(12)]],
    constant int&   ring_cap  [[buffer(13)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]])
{
  (void)D; (void)ring_cap;
  const int h    = (int)tgid.y;
  const int q0   = (int)tgid.z * SA3_BQ;
  const int kvh  = h / (Hq / Hkv);
  const device VPIPE_ELT* kbase = k + (int64_t)kvh * kv_stride * SA3_D;
  const device VPIPE_ELT* vbase = v + (int64_t)kvh * kv_stride * SA3_D;

  threadgroup VPIPE_ELT Ss[SA3_BQ * SA3_BK];
  threadgroup float Of[SA3_BQ * SA3_D];
  threadgroup float mrow[SA3_BQ], lrow[SA3_BQ], corr_s[SA3_BQ];

  for (int e = (int)lid; e < SA3_BQ * SA3_D; e += SA3_THREADS) { Of[e] = 0.0f; }
  for (int e = (int)lid; e < SA3_BQ; e += SA3_THREADS) {
    mrow[e] = -INFINITY; lrow[e] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const device VPIPE_ELT* qh = q + ((int64_t)h * n_q + q0) * SA3_D;
  using TQ = tensor<device VPIPE_ELT, dextents<int32_t,2>, tensor_inline>;
  using TP = tensor<threadgroup VPIPE_ELT, dextents<int32_t,2>, tensor_inline>;
  using TO = tensor<threadgroup float, dextents<int32_t,2>, tensor_inline>;
  constexpr auto qk = matmul2d_descriptor(
      SA3_BQ, SA3_BK, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<qk, execution_simdgroups<SA3_SG>> opQK;
  constexpr auto pv = matmul2d_descriptor(
      SA3_BQ, SA3_D, SA3_BK, false, false, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<pv, execution_simdgroups<SA3_SG>> opPV;
  TQ tQ(const_cast<device VPIPE_ELT*>(qh), dextents<int32_t,2>(SA3_D, SA3_BQ));

  const int tile_qpos_max = q_offset + min(q0 + SA3_BQ - 1, n_q - 1);
  const int last = min(T_kv - 1, tile_qpos_max);
  const int tile_qmin = q_offset + q0;
  const int first = (window > 0) ? max(0, tile_qmin - window + 1) : 0;

  for (int bs = (first / SA3_BK) * SA3_BK; bs <= last; bs += SA3_BK) {
    const int bk = min(SA3_BK, T_kv - bs);
    TQ tK(const_cast<device VPIPE_ELT*>(kbase + (int64_t)bs * SA3_D),
          dextents<int32_t,2>(SA3_D, SA3_BK));
    auto cS = opQK.get_destination_cooperative_tensor<TQ, TQ, VPIPE_ELT>();
    opQK.run(tQ, tK, cS);
    TP tS(Ss, dextents<int32_t,2>(SA3_BK, SA3_BQ));
    cS.store(tS);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int r = (int)lid; r < SA3_BQ; r += SA3_THREADS) {
      const int qpos = q_offset + q0 + r;
      const bool qok = (q0 + r) < n_q;
      const int wlo = (window > 0) ? (qpos - window + 1) : 0;
      float mloc = mrow[r];
      for (int j = 0; j < SA3_BK; ++j) {
        const int kpos = bs + j;
        const bool ok = qok && j < bk && kpos <= qpos && kpos >= wlo;
        const float s = ok ? float(Ss[r * SA3_BK + j]) * scale : -INFINITY;
        Ss[r * SA3_BK + j] = (VPIPE_ELT)s;
        mloc = max(mloc, s);
      }
      const float corr = (mloc == -INFINITY) ? 1.0f : exp(mrow[r] - mloc);
      float lloc = lrow[r] * corr;
      for (int j = 0; j < SA3_BK; ++j) {
        const float s = float(Ss[r * SA3_BK + j]);
        const float p = (s == -INFINITY) ? 0.0f : exp(s - mloc);
        Ss[r * SA3_BK + j] = (VPIPE_ELT)p;
        lloc += p;
      }
      corr_s[r] = corr;
      mrow[r] = mloc; lrow[r] = lloc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Rescale the running PV accumulator across ALL threads (was BQ threads
    // each looping D). SA3_D is a power of two so /,% fold to shifts.
    for (int e = (int)lid; e < SA3_BQ * SA3_D; e += SA3_THREADS) {
      Of[e] *= corr_s[e / SA3_D];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    TP tP(Ss, dextents<int32_t,2>(SA3_BK, SA3_BQ));
    TQ tV(const_cast<device VPIPE_ELT*>(vbase + (int64_t)bs * SA3_D),
          dextents<int32_t,2>(SA3_D, SA3_BK));
    TO tO(Of, dextents<int32_t,2>(SA3_D, SA3_BQ));
    opPV.run(tP, tV, tO);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (int e = (int)lid; e < SA3_BQ * SA3_D; e += SA3_THREADS) {
    const int r = e / SA3_D, dd = e % SA3_D;
    if (q0 + r < n_q) {
      const float inv = (lrow[r] > 0.0f) ? 1.0f / lrow[r] : 0.0f;
      out[((int64_t)h * n_q + (q0 + r)) * SA3_D + dd] = (VPIPE_ELT)(Of[e] * inv);
    }
  }
}
// head_dim 64 sliding-WINDOW causal sibling of the d256 kernel above -- the same
// matrix-core matmul2d QK^T/PV flash body (online softmax in threadgroup, window
// lower bound kpos >= qpos-window+1 on top of the causal mask), sized for D=64
// (Of[16*64] f32 == 4 KB). This is the MOSS-codec-v2 decoder attention path (12
// heads, head_dim 64, sliding window); linear KV addressing (ring_cap ignored,
// logical key == physical slot). Replaces the scalar per-query sdpa_causal_
// window_f16 on M5 (that kernel wasted 32x on redundant per-key simd_sum + exp).
#define SA4_BQ 16
#define SA4_BK 16
#define SA4_D  64
#define SA4_SG 4
#define SA4_THREADS (SA4_SG * 32)
kernel void sdpa_causal_mma2_d64_f16(
    const device VPIPE_ELT* q     [[buffer(0)]],
    const device VPIPE_ELT* k     [[buffer(1)]],
    const device VPIPE_ELT* v     [[buffer(2)]],
    device VPIPE_ELT*       out   [[buffer(3)]],
    constant float& scale     [[buffer(4)]],
    constant int&   T_kv      [[buffer(5)]],
    constant int&   D         [[buffer(6)]],
    constant int&   Hq        [[buffer(7)]],
    constant int&   Hkv       [[buffer(8)]],
    constant int&   n_q       [[buffer(9)]],
    constant int&   q_offset  [[buffer(10)]],
    constant int&   kv_stride [[buffer(11)]],
    constant int&   window    [[buffer(12)]],
    constant int&   ring_cap  [[buffer(13)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]])
{
  (void)D; (void)ring_cap;
  const int h    = (int)tgid.y;
  const int q0   = (int)tgid.z * SA4_BQ;
  const int kvh  = h / (Hq / Hkv);
  const device VPIPE_ELT* kbase = k + (int64_t)kvh * kv_stride * SA4_D;
  const device VPIPE_ELT* vbase = v + (int64_t)kvh * kv_stride * SA4_D;

  threadgroup VPIPE_ELT Ss[SA4_BQ * SA4_BK];
  threadgroup float Of[SA4_BQ * SA4_D];
  threadgroup float mrow[SA4_BQ], lrow[SA4_BQ], corr_s[SA4_BQ];

  for (int e = (int)lid; e < SA4_BQ * SA4_D; e += SA4_THREADS) { Of[e] = 0.0f; }
  for (int e = (int)lid; e < SA4_BQ; e += SA4_THREADS) {
    mrow[e] = -INFINITY; lrow[e] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const device VPIPE_ELT* qh = q + ((int64_t)h * n_q + q0) * SA4_D;
  using TQ = tensor<device VPIPE_ELT, dextents<int32_t,2>, tensor_inline>;
  using TP = tensor<threadgroup VPIPE_ELT, dextents<int32_t,2>, tensor_inline>;
  using TO = tensor<threadgroup float, dextents<int32_t,2>, tensor_inline>;
  constexpr auto qk = matmul2d_descriptor(
      SA4_BQ, SA4_BK, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<qk, execution_simdgroups<SA4_SG>> opQK;
  constexpr auto pv = matmul2d_descriptor(
      SA4_BQ, SA4_D, SA4_BK, false, false, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<pv, execution_simdgroups<SA4_SG>> opPV;
  TQ tQ(const_cast<device VPIPE_ELT*>(qh), dextents<int32_t,2>(SA4_D, SA4_BQ));

  const int tile_qpos_max = q_offset + min(q0 + SA4_BQ - 1, n_q - 1);
  const int last = min(T_kv - 1, tile_qpos_max);
  const int tile_qmin = q_offset + q0;
  const int first = (window > 0) ? max(0, tile_qmin - window + 1) : 0;

  for (int bs = (first / SA4_BK) * SA4_BK; bs <= last; bs += SA4_BK) {
    const int bk = min(SA4_BK, T_kv - bs);
    TQ tK(const_cast<device VPIPE_ELT*>(kbase + (int64_t)bs * SA4_D),
          dextents<int32_t,2>(SA4_D, SA4_BK));
    auto cS = opQK.get_destination_cooperative_tensor<TQ, TQ, VPIPE_ELT>();
    opQK.run(tQ, tK, cS);
    TP tS(Ss, dextents<int32_t,2>(SA4_BK, SA4_BQ));
    cS.store(tS);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int r = (int)lid; r < SA4_BQ; r += SA4_THREADS) {
      const int qpos = q_offset + q0 + r;
      const bool qok = (q0 + r) < n_q;
      const int wlo = (window > 0) ? (qpos - window + 1) : 0;
      float mloc = mrow[r];
      for (int j = 0; j < SA4_BK; ++j) {
        const int kpos = bs + j;
        const bool ok = qok && j < bk && kpos <= qpos && kpos >= wlo;
        const float s = ok ? float(Ss[r * SA4_BK + j]) * scale : -INFINITY;
        Ss[r * SA4_BK + j] = (VPIPE_ELT)s;
        mloc = max(mloc, s);
      }
      const float corr = (mloc == -INFINITY) ? 1.0f : exp(mrow[r] - mloc);
      float lloc = lrow[r] * corr;
      for (int j = 0; j < SA4_BK; ++j) {
        const float s = float(Ss[r * SA4_BK + j]);
        const float p = (s == -INFINITY) ? 0.0f : exp(s - mloc);
        Ss[r * SA4_BK + j] = (VPIPE_ELT)p;
        lloc += p;
      }
      corr_s[r] = corr;
      mrow[r] = mloc; lrow[r] = lloc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = (int)lid; e < SA4_BQ * SA4_D; e += SA4_THREADS) {
      Of[e] *= corr_s[e / SA4_D];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    TP tP(Ss, dextents<int32_t,2>(SA4_BK, SA4_BQ));
    TQ tV(const_cast<device VPIPE_ELT*>(vbase + (int64_t)bs * SA4_D),
          dextents<int32_t,2>(SA4_D, SA4_BK));
    TO tO(Of, dextents<int32_t,2>(SA4_D, SA4_BQ));
    opPV.run(tP, tV, tO);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (int e = (int)lid; e < SA4_BQ * SA4_D; e += SA4_THREADS) {
    const int r = e / SA4_D, dd = e % SA4_D;
    if (q0 + r < n_q) {
      const float inv = (lrow[r] > 0.0f) ? 1.0f / lrow[r] : 0.0f;
      out[((int64_t)h * n_q + (q0 + r)) * SA4_D + dd] = (VPIPE_ELT)(Of[e] * inv);
    }
  }
}
// PAGED sibling of sdpa_causal_mma2_d512_f16: Gemma-4's GLOBAL (full-attention,
// head_dim 512) layers, K/V in the PAGED pool instead of a contiguous cache.
// Same D=512 matrix-core flash body (matmul2d QK^T/PV, online softmax in tg,
// parallelized Of rescale) as the contiguous kernel; only the K/V addressing
// changes -- it walks the page table (the page-by-page loop + causal-by-global-
// position masking of sdpa_mma_f16). This is what makes no-copy branch work for
// the global layers: children share the parent's frozen prefix pages by
// refcount and own only their divergent tail pages; attention streams straight
// across shared + private pages with zero gather/copy. Full-causal only (global
// layers, window==0). page_tokens % SA2_BK == 0 so a BK block never straddles a
// page (the over-read of the last block past nvalid is masked, in-page memory).
//   0:q[Hq,n_q,D] 1:kpool 2:vpool 3:out[Hq,n_q,D] 4:scale 5:D 6:Hq 7:Hkv
//   8:n_q 9:q_offset 10:page_tokens 11:n_pages 12:page_table(int[n_pages*3])
// grid {SA2_SG*32, Hq, ceil(n_q/SA2_BQ)}, tg {SA2_SG*32,1,1}.
kernel void sdpa_paged_mma2_d512_f16(
    const device VPIPE_ELT* q          [[buffer(0)]],
    const device VPIPE_ELT* kpool      [[buffer(1)]],
    const device VPIPE_ELT* vpool      [[buffer(2)]],
    device VPIPE_ELT*       out        [[buffer(3)]],
    constant float& scale       [[buffer(4)]],
    constant int&   D           [[buffer(5)]],
    constant int&   Hq          [[buffer(6)]],
    constant int&   Hkv         [[buffer(7)]],
    constant int&   n_q         [[buffer(8)]],
    constant int&   q_offset    [[buffer(9)]],
    constant int&   page_tokens [[buffer(10)]],
    constant int&   n_pages     [[buffer(11)]],
    const device int* page_table[[buffer(12)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]])
{
  (void)D;
  const int h    = (int)tgid.y;
  const int q0   = (int)tgid.z * SA2_BQ;
  const int kvh  = h / (Hq / Hkv);
  const uint page_stride = (uint)Hkv * page_tokens * SA2_D;
  const uint head_off    = (uint)kvh * page_tokens * SA2_D;

  threadgroup VPIPE_ELT Ss[SA2_BQ * SA2_BK];
  threadgroup float Of[SA2_BQ * SA2_D];
  threadgroup float mrow[SA2_BQ], lrow[SA2_BQ], corr_s[SA2_BQ];

  for (int e = (int)lid; e < SA2_BQ * SA2_D; e += SA2_THREADS) { Of[e] = 0.0f; }
  for (int e = (int)lid; e < SA2_BQ; e += SA2_THREADS) {
    mrow[e] = -INFINITY; lrow[e] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const device VPIPE_ELT* qh = q + ((int64_t)h * n_q + q0) * SA2_D;
  using TQ = tensor<device VPIPE_ELT, dextents<int32_t,2>, tensor_inline>;
  using TP = tensor<threadgroup VPIPE_ELT, dextents<int32_t,2>, tensor_inline>;
  using TO = tensor<threadgroup float, dextents<int32_t,2>, tensor_inline>;
  constexpr auto qk = matmul2d_descriptor(
      SA2_BQ, SA2_BK, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<qk, execution_simdgroups<SA2_SG>> opQK;
  constexpr auto pv = matmul2d_descriptor(
      SA2_BQ, SA2_D, SA2_BK, false, false, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<pv, execution_simdgroups<SA2_SG>> opPV;
  TQ tQ(const_cast<device VPIPE_ELT*>(qh), dextents<int32_t,2>(SA2_D, SA2_BQ));

  const int tile_qpos_max = q_offset + min(q0 + SA2_BQ - 1, n_q - 1);

  for (int pg = 0; pg < n_pages; ++pg) {
    const int pid    = page_table[pg * 3 + 0];
    const int nvalid = page_table[pg * 3 + 1];
    const int gstart = page_table[pg * 3 + 2];
    if (gstart > tile_qpos_max) { break; }
    const device VPIPE_ELT* kbase = kpool + (uint)pid * page_stride + head_off;
    const device VPIPE_ELT* vbase = vpool + (uint)pid * page_stride + head_off;
    for (int bs = 0; bs < nvalid; bs += SA2_BK) {
      if (gstart + bs > tile_qpos_max) { break; }
      const int bk = min(SA2_BK, nvalid - bs);
      TQ tK(const_cast<device VPIPE_ELT*>(kbase + (int64_t)bs * SA2_D),
            dextents<int32_t,2>(SA2_D, SA2_BK));
      auto cS = opQK.get_destination_cooperative_tensor<TQ, TQ, VPIPE_ELT>();
      opQK.run(tQ, tK, cS);
      TP tS(Ss, dextents<int32_t,2>(SA2_BK, SA2_BQ));
      cS.store(tS);
      threadgroup_barrier(mem_flags::mem_threadgroup);

      for (int r = (int)lid; r < SA2_BQ; r += SA2_THREADS) {
        const int qpos = q_offset + q0 + r;
        const bool qok = (q0 + r) < n_q;
        float mloc = mrow[r];
        for (int j = 0; j < SA2_BK; ++j) {
          const int kpos = gstart + bs + j;
          const bool ok = qok && j < bk && kpos <= qpos;
          const float s = ok ? float(Ss[r * SA2_BK + j]) * scale : -INFINITY;
          Ss[r * SA2_BK + j] = (VPIPE_ELT)s;
          mloc = max(mloc, s);
        }
        const float corr = (mloc == -INFINITY) ? 1.0f : exp(mrow[r] - mloc);
        float lloc = lrow[r] * corr;
        for (int j = 0; j < SA2_BK; ++j) {
          const float s = float(Ss[r * SA2_BK + j]);
          const float p = (s == -INFINITY) ? 0.0f : exp(s - mloc);
          Ss[r * SA2_BK + j] = (VPIPE_ELT)p;
          lloc += p;
        }
        corr_s[r] = corr;
        mrow[r] = mloc; lrow[r] = lloc;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int e = (int)lid; e < SA2_BQ * SA2_D; e += SA2_THREADS) {
        Of[e] *= corr_s[e / SA2_D];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      TP tP(Ss, dextents<int32_t,2>(SA2_BK, SA2_BQ));
      TQ tV(const_cast<device VPIPE_ELT*>(vbase + (int64_t)bs * SA2_D),
            dextents<int32_t,2>(SA2_D, SA2_BK));
      TO tO(Of, dextents<int32_t,2>(SA2_D, SA2_BQ));
      opPV.run(tP, tV, tO);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  for (int e = (int)lid; e < SA2_BQ * SA2_D; e += SA2_THREADS) {
    const int r = e / SA2_D, dd = e % SA2_D;
    if (q0 + r < n_q) {
      const float inv = (lrow[r] > 0.0f) ? 1.0f / lrow[r] : 0.0f;
      out[((int64_t)h * n_q + (q0 + r)) * SA2_D + dd] = (VPIPE_ELT)(Of[e] * inv);
    }
  }
}

// NON-causal (bidirectional) matrix-core flash attention for ViT towers,
// head_dim 64, full MHA. Reads Q/K/V DIRECTLY from the [n, Hq*D] interleaved
// layout (each patch row holds all heads contiguously; row stride Hq*D) and
// writes OUT in the same [n, Hq*D] layout -- so the encoder needs NO q/k/v/o
// transposes. Per head h, static_slice<D,BQ>(h*D, q0) carves the strided
// [BQ, D] tile: inner extent D fixes the matmul K-reduction to this head's D
// columns, while the parent row stride Hq*D is preserved. The last K/V block
// past T_kv is matmul-read but masked (j>=bk -> -inf -> p=0); the caller pads
// q/k/v by SAVF_BQ rows so the row over-read stays in bounds.
//   0:q[n,Hq*D] 1:k[T_kv,Hq*D] 2:v 3:out[n,Hq*D] 4:scale 5:T_kv 6:D 7:Hq
//   8:Hkv 9:n_q 10:q_offset 11:(unused)
// grid {SAVF_SG*32, Hq, ceil(n_q/SAVF_BQ)}, tg {SAVF_SG*32,1,1}.
#define SAVF_BQ 32
#define SAVF_BK 32
#define SAVF_D  64
#define SAVF_SG 4
#define SAVF_THREADS (SAVF_SG * 32)
kernel void sdpa_full_mma2_d64_f16(
    const device VPIPE_ELT* q     [[buffer(0)]],
    const device VPIPE_ELT* k     [[buffer(1)]],
    const device VPIPE_ELT* v     [[buffer(2)]],
    device VPIPE_ELT*       out   [[buffer(3)]],
    constant float& scale     [[buffer(4)]],
    constant int&   T_kv      [[buffer(5)]],
    constant int&   D         [[buffer(6)]],
    constant int&   Hq        [[buffer(7)]],
    constant int&   Hkv       [[buffer(8)]],
    constant int&   n_q       [[buffer(9)]],
    constant int&   q_offset  [[buffer(10)]],
    constant int&   kv_stride [[buffer(11)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]])
{
  (void)D; (void)Hkv; (void)q_offset; (void)kv_stride;
  const int h   = (int)tgid.y;
  const int q0  = (int)tgid.z * SAVF_BQ;
  const int rs  = Hq * SAVF_D;          // row stride (= hidden)
  const int hc  = h * SAVF_D;           // this head's column origin

  // Strided per-head Q/K/V tiles staged into PACKED threadgroup memory: the
  // [n, hidden] layout's head-h columns (stride hidden) are gathered into
  // contiguous [BQ/BK, D] tiles that the matmul reads directly -- so no q/k/v
  // transposes upstream. Bounds-checked (zero past n_q/T_kv) => no padding.
  threadgroup VPIPE_ELT Qs[SAVF_BQ * SAVF_D];
  threadgroup VPIPE_ELT Ks[SAVF_BK * SAVF_D];
  threadgroup VPIPE_ELT Vs[SAVF_BK * SAVF_D];
  threadgroup VPIPE_ELT Ss[SAVF_BQ * SAVF_BK];
  threadgroup float Of[SAVF_BQ * SAVF_D];
  threadgroup float mrow[SAVF_BQ], lrow[SAVF_BQ], corr_s[SAVF_BQ];

  for (int e = (int)lid; e < SAVF_BQ * SAVF_D; e += SAVF_THREADS) { Of[e] = 0.0f; }
  for (int e = (int)lid; e < SAVF_BQ; e += SAVF_THREADS) {
    mrow[e] = -INFINITY; lrow[e] = 0.0f;
  }
  for (int e = (int)lid; e < SAVF_BQ * SAVF_D; e += SAVF_THREADS) {
    const int r = e / SAVF_D, d = e % SAVF_D, qp = q0 + r;
    Qs[e] = (qp < n_q) ? q[(int64_t)qp * rs + hc + d] : (VPIPE_ELT)0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  using TT = tensor<threadgroup VPIPE_ELT, dextents<int32_t,2>, tensor_inline>;
  using TO = tensor<threadgroup float, dextents<int32_t,2>, tensor_inline>;
  constexpr auto qk = matmul2d_descriptor(
      SAVF_BQ, SAVF_BK, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<qk, execution_simdgroups<SAVF_SG>> opQK;
  constexpr auto pv = matmul2d_descriptor(
      SAVF_BQ, SAVF_D, SAVF_BK, false, false, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<pv, execution_simdgroups<SAVF_SG>> opPV;
  TT tQ(Qs, dextents<int32_t,2>(SAVF_D, SAVF_BQ));

  for (int bs = 0; bs < T_kv; bs += SAVF_BK) {
    const int bk = min(SAVF_BK, T_kv - bs);
    for (int e = (int)lid; e < SAVF_BK * SAVF_D; e += SAVF_THREADS) {
      const int j = e / SAVF_D, d = e % SAVF_D, kp = bs + j;
      const bool ok = kp < T_kv;
      Ks[e] = ok ? k[(int64_t)kp * rs + hc + d] : (VPIPE_ELT)0;
      Vs[e] = ok ? v[(int64_t)kp * rs + hc + d] : (VPIPE_ELT)0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    TT tK(Ks, dextents<int32_t,2>(SAVF_D, SAVF_BK));
    auto cS = opQK.get_destination_cooperative_tensor<TT, TT, VPIPE_ELT>();
    opQK.run(tQ, tK, cS);
    TT tS(Ss, dextents<int32_t,2>(SAVF_BK, SAVF_BQ));
    cS.store(tS);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int r = (int)lid; r < SAVF_BQ; r += SAVF_THREADS) {
      const bool qok = (q0 + r) < n_q;
      float mloc = mrow[r];
      for (int j = 0; j < SAVF_BK; ++j) {
        const bool ok = qok && j < bk;
        const float s = ok ? float(Ss[r * SAVF_BK + j]) * scale : -INFINITY;
        Ss[r * SAVF_BK + j] = (VPIPE_ELT)s;
        mloc = max(mloc, s);
      }
      const float corr = exp(mrow[r] - mloc);
      float lloc = lrow[r] * corr;
      for (int j = 0; j < SAVF_BK; ++j) {
        const float s = float(Ss[r * SAVF_BK + j]);
        const float p = (s == -INFINITY) ? 0.0f : exp(s - mloc);
        Ss[r * SAVF_BK + j] = (VPIPE_ELT)p;
        lloc += p;
      }
      corr_s[r] = corr;
      mrow[r] = mloc; lrow[r] = lloc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = (int)lid; e < SAVF_BQ * SAVF_D; e += SAVF_THREADS) {
      Of[e] *= corr_s[e / SAVF_D];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    TT tP(Ss, dextents<int32_t,2>(SAVF_BK, SAVF_BQ));
    TT tV(Vs, dextents<int32_t,2>(SAVF_D, SAVF_BK));
    TO tO(Of, dextents<int32_t,2>(SAVF_D, SAVF_BQ));
    opPV.run(tP, tV, tO);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (int e = (int)lid; e < SAVF_BQ * SAVF_D; e += SAVF_THREADS) {
    const int r = e / SAVF_D, dd = e % SAVF_D;
    if (q0 + r < n_q) {
      const float inv = (lrow[r] > 0.0f) ? 1.0f / lrow[r] : 0.0f;
      out[(int64_t)(q0 + r) * rs + hc + dd] = (VPIPE_ELT)(Of[e] * inv);
    }
  }
}

// Contiguous-KV matrix-core FULL (non-causal) flash attention, head_dim a
// template width. For the VAE spatial self-attention: SINGLE head (Hq=Hkv=1),
// head_dim = the channel dim (Krea-2 384, FLUX.2 512), every one of the N=H*W
// latent tokens attends every other. The scalar sdpa_full_f16 (grid one tg per
// query, O(N^2) simd-reduce) DOMINATES VAE decode at high res (~60-75% @1024);
// this runs QK^T + P*V on the matrix units (matmul2d) with the online softmax
// in tg memory. Q/K/V are read DIRECTLY from device memory as tensors (no tg
// staging), so only Ss[BQ,BK] + Of[BQ,D] f32 sit in tg memory: at D=512, BQ=8
// that is 16 KB, under the 32 KB budget with slack for matmul2d scratch. The
// over-read of the last BK block past T_kv is masked. Drop-in for sdpa_full_f16
// (identical buffers 0..10, no q_offset -- full attention).
//
// The QK^T accumulator is FLOAT, unlike the LM attention kernels above (which
// accumulate in VPIPE_ELT). It has to be: this is the one call site whose Q/K
// are raw conv1x1 outputs rather than RMSNorm'd projections, and the dot runs
// over D=384/512 rather than a head_dim <= 256. A VAE decoder's mid-block
// scores leave f16's range, and the failure is SILENT at the magnitudes that
// actually occur: the output stays FINITE (only a further overflow reaches inf
// and NaN, which is what the synthetic range arm of the unit test sees) and is
// merely wrong -- nothing traps. Measured on Boogu's plain
// AutoencoderKL -- which, unlike FLUX.2's, whitens its latent with a scalar
// shift/scale rather than a BatchNorm, so its decoder activations are much
// larger -- rel-L2 2.23 against the simdgroup-matrix flash it replaces, at 256,
// 1024 and 4096 mid tokens ALIKE. Shape-independence is what distinguishes a
// range failure from a tiling bug; f32 here takes it to bit-identical. FLUX.2's
// own decode sat at 5e-4 before and after, which is why an M5 bring-up that
// only exercised FLUX.2 could not see this. The extra tg cost is 512 B.
//   0:q[Hq,n_q,D] 1:k 2:v[Hkv,kv_stride,D] 3:out[Hq,n_q,D] 4:scale 5:T_kv 6:D
//   7:Hq 8:Hkv 9:n_q 10:kv_stride
// grid {SAF_SG*32, Hq, ceil(n_q/SAF_BQ)}, tg {SAF_SG*32,1,1}.
#define SAF_BQ 8
#define SAF_BK 16
#define SAF_SG 4
#define SAF_THREADS (SAF_SG * 32)
template <int SAF_D>
[[kernel]] void sdpa_full_mma2_tmpl(
    const device VPIPE_ELT* q     [[buffer(0)]],
    const device VPIPE_ELT* k     [[buffer(1)]],
    const device VPIPE_ELT* v     [[buffer(2)]],
    device VPIPE_ELT*       out   [[buffer(3)]],
    constant float& scale     [[buffer(4)]],
    constant int&   T_kv      [[buffer(5)]],
    constant int&   D         [[buffer(6)]],
    constant int&   Hq        [[buffer(7)]],
    constant int&   Hkv       [[buffer(8)]],
    constant int&   n_q       [[buffer(9)]],
    constant int&   kv_stride [[buffer(10)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]])
{
  (void)D;
  const int h    = (int)tgid.y;
  const int q0   = (int)tgid.z * SAF_BQ;
  const int kvh  = h / (Hq / Hkv);
  const device VPIPE_ELT* kbase = k + (int64_t)kvh * kv_stride * SAF_D;
  const device VPIPE_ELT* vbase = v + (int64_t)kvh * kv_stride * SAF_D;

  threadgroup VPIPE_ELT Ss[SAF_BQ * SAF_BK];   // P (softmax output, in [0,1])
  threadgroup float Sf[SAF_BQ * SAF_BK];       // QK^T scores (see above)
  threadgroup float Of[SAF_BQ * SAF_D];
  threadgroup float mrow[SAF_BQ], lrow[SAF_BQ], corr_s[SAF_BQ];

  for (int e = (int)lid; e < SAF_BQ * SAF_D; e += SAF_THREADS) { Of[e] = 0.0f; }
  for (int e = (int)lid; e < SAF_BQ; e += SAF_THREADS) {
    mrow[e] = -INFINITY; lrow[e] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const device VPIPE_ELT* qh = q + ((int64_t)h * n_q + q0) * SAF_D;
  using TQ = tensor<device VPIPE_ELT, dextents<int32_t,2>, tensor_inline>;
  using TP = tensor<threadgroup VPIPE_ELT, dextents<int32_t,2>, tensor_inline>;
  using TO = tensor<threadgroup float, dextents<int32_t,2>, tensor_inline>;
  constexpr auto qk = matmul2d_descriptor(
      SAF_BQ, SAF_BK, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<qk, execution_simdgroups<SAF_SG>> opQK;
  constexpr auto pv = matmul2d_descriptor(
      SAF_BQ, SAF_D, SAF_BK, false, false, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<pv, execution_simdgroups<SAF_SG>> opPV;
  TQ tQ(const_cast<device VPIPE_ELT*>(qh), dextents<int32_t,2>(SAF_D, SAF_BQ));

  for (int bs = 0; bs < T_kv; bs += SAF_BK) {
    const int bk = min(SAF_BK, T_kv - bs);
    TQ tK(const_cast<device VPIPE_ELT*>(kbase + (int64_t)bs * SAF_D),
          dextents<int32_t,2>(SAF_D, SAF_BK));
    auto cS = opQK.get_destination_cooperative_tensor<TQ, TQ, float>();
    opQK.run(tQ, tK, cS);
    TO tSf(Sf, dextents<int32_t,2>(SAF_BK, SAF_BQ));
    cS.store(tSf);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int r = (int)lid; r < SAF_BQ; r += SAF_THREADS) {
      const bool qok = (q0 + r) < n_q;
      float mloc = mrow[r];
      for (int j = 0; j < SAF_BK; ++j) {
        const bool ok = qok && j < bk;
        const float s = ok ? Sf[r * SAF_BK + j] * scale : -INFINITY;
        Sf[r * SAF_BK + j] = s;
        mloc = max(mloc, s);
      }
      const float corr = exp(mrow[r] - mloc);
      float lloc = lrow[r] * corr;
      for (int j = 0; j < SAF_BK; ++j) {
        const float s = Sf[r * SAF_BK + j];
        const float p = (s == -INFINITY) ? 0.0f : exp(s - mloc);
        Ss[r * SAF_BK + j] = (VPIPE_ELT)p;
        lloc += p;
      }
      corr_s[r] = corr;
      mrow[r] = mloc; lrow[r] = lloc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = (int)lid; e < SAF_BQ * SAF_D; e += SAF_THREADS) {
      Of[e] *= corr_s[e / SAF_D];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    TP tP(Ss, dextents<int32_t,2>(SAF_BK, SAF_BQ));
    TQ tV(const_cast<device VPIPE_ELT*>(vbase + (int64_t)bs * SAF_D),
          dextents<int32_t,2>(SAF_D, SAF_BK));
    TO tO(Of, dextents<int32_t,2>(SAF_D, SAF_BQ));
    opPV.run(tP, tV, tO);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (int e = (int)lid; e < SAF_BQ * SAF_D; e += SAF_THREADS) {
    const int r = e / SAF_D, dd = e % SAF_D;
    if (q0 + r < n_q) {
      const float inv = (lrow[r] > 0.0f) ? 1.0f / lrow[r] : 0.0f;
      out[((int64_t)h * n_q + (q0 + r)) * SAF_D + dd] = (VPIPE_ELT)(Of[e] * inv);
    }
  }
}
template [[host_name("sdpa_full_mma2_d384_f16")]] [[kernel]]
decltype(sdpa_full_mma2_tmpl<384>) sdpa_full_mma2_tmpl<384>;
template [[host_name("sdpa_full_mma2_d512_f16")]] [[kernel]]
decltype(sdpa_full_mma2_tmpl<512>) sdpa_full_mma2_tmpl<512>;

// Wide-query-tile twin of sdpa_full_mma2_tmpl. Identical math, identical
// buffers, identical f32 accumulators -- the ONE difference is where the
// BQ x D output accumulator lives, and that is what sets the runtime.
//
// This kernel is bandwidth-bound, not compute-bound, and BQ is the only knob
// that touches the bandwidth. Every threadgroup streams the WHOLE of K and V
// (that is what flash attention does), so total K/V traffic is
// ceil(n_q/BQ) * 2 * n_q * D * sizeof(elt) -- inversely proportional to BQ and
// to nothing else. MEASURED on the FLUX.2 VAE mid-block at 1024x768
// (n_q = 12288, D = 512, one head): BQ=8 puts 1536 threadgroups x 24 MB of K/V
// = 38.7 GB across the fabric in 161 ms, i.e. 238 GB/s, which is the roof. The
// same 309 GFLOP at the matrix units' ~9 TFLOP/s would be 34 ms, so the
// arithmetic intensity (8 FLOP/byte against an M5 balance point near 45) is
// the whole story. BK cannot substitute: it sets how many blocks each
// threadgroup walks its K/V in, not how many bytes that walk costs.
//
// BQ above 8 is impossible with the accumulator in threadgroup memory: Of is
// BQ*D floats, so D=512 hits the 32 KB threadgroup budget at BQ=16 with
// nothing left for Ss/Sf. Hence the register accumulator -- a cooperative
// tensor held across the whole K/V loop in mode::multiply_accumulate, spread
// over the SG simdgroups' threads. The online-softmax rescale that the base
// kernel does as an indexed threadgroup op becomes a walk over this thread's
// own elements, each asking the (implementation-defined) layout which row it
// belongs to via get_multidimensional_index. The tile is chosen so that
// BQ*D/(SG*32) stays near 64 floats per thread at both widths.
//
// STILL f32 for both accumulators, deliberately. The QK^T f32 requirement
// documented on the base kernel is a RANGE argument about this call site's
// unnormalized conv1x1 Q/K -- it does not weaken because the tile got wider,
// and the win here is bandwidth, which f32 accumulators do not cost (they are
// registers and threadgroup scratch, never traffic). O accumulates in f32 for
// the same reason it did in threadgroup memory: a 12288-key online sum in f16
// would drift regardless of tiling.
//   0:q[Hq,n_q,D] 1:k 2:v[Hkv,kv_stride,D] 3:out[Hq,n_q,D] 4:scale 5:T_kv 6:D
//   7:Hq 8:Hkv 9:n_q 10:kv_stride
// grid {SG*32, Hq, ceil(n_q/BQ)}, tg {SG*32,1,1}.
template <int SAW_D, int SAW_BQ, int SAW_BK, int SAW_SG>
[[kernel]] void sdpa_full_mma2_wide_tmpl(
    const device VPIPE_ELT* q     [[buffer(0)]],
    const device VPIPE_ELT* k     [[buffer(1)]],
    const device VPIPE_ELT* v     [[buffer(2)]],
    device VPIPE_ELT*       out   [[buffer(3)]],
    constant float& scale     [[buffer(4)]],
    constant int&   T_kv      [[buffer(5)]],
    constant int&   D         [[buffer(6)]],
    constant int&   Hq        [[buffer(7)]],
    constant int&   Hkv       [[buffer(8)]],
    constant int&   n_q       [[buffer(9)]],
    constant int&   kv_stride [[buffer(10)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]])
{
  (void)D;
  constexpr int THREADS = SAW_SG * 32;
  const int h    = (int)tgid.y;
  const int q0   = (int)tgid.z * SAW_BQ;
  const int kvh  = h / (Hq / Hkv);
  const device VPIPE_ELT* kbase = k + (int64_t)kvh * kv_stride * SAW_D;
  const device VPIPE_ELT* vbase = v + (int64_t)kvh * kv_stride * SAW_D;

  threadgroup VPIPE_ELT Ss[SAW_BQ * SAW_BK];   // P (softmax output, in [0,1])
  threadgroup float Sf[SAW_BQ * SAW_BK];       // QK^T scores (f32, see above)
  threadgroup float mrow[SAW_BQ], lrow[SAW_BQ], corr_s[SAW_BQ];

  for (int e = (int)lid; e < SAW_BQ; e += THREADS) {
    mrow[e] = -INFINITY; lrow[e] = 0.0f;
  }

  const device VPIPE_ELT* qh = q + ((int64_t)h * n_q + q0) * SAW_D;
  using TD = tensor<device VPIPE_ELT, dextents<int32_t,2>, tensor_inline>;
  using TP = tensor<threadgroup VPIPE_ELT, dextents<int32_t,2>, tensor_inline>;
  using TS = tensor<threadgroup float, dextents<int32_t,2>, tensor_inline>;
  constexpr auto qk = matmul2d_descriptor(
      SAW_BQ, SAW_BK, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<qk, execution_simdgroups<SAW_SG>> opQK;
  constexpr auto pv = matmul2d_descriptor(
      SAW_BQ, SAW_D, SAW_BK, false, false, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<pv, execution_simdgroups<SAW_SG>> opPV;
  // Every device tile carries its TRUE extent, not the tile constant, so a
  // ragged tail reads nothing past the buffer -- matmul2d clamps a partially
  // out-of-range tile. The base kernel gets away with over-reading because
  // BQ=8/BK=16 keeps the overshoot inside the page; at BQ=64 the same over-read
  // would be 63 rows * D past the end, and V garbage that decodes to NaN
  // survives the P=0 mask (0 * NaN = NaN).
  const int qrows = min(SAW_BQ, n_q - q0);
  TD tQ(const_cast<device VPIPE_ELT*>(qh),
        dextents<int32_t,2>(SAW_D, qrows));

  // The O accumulator, register-resident for the whole K/V walk.
  auto cO = opPV.template get_destination_cooperative_tensor<TP, TD, float>();
  for (auto it = cO.begin(); it != cO.end(); ++it) { *it = 0.0f; }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int bs = 0; bs < T_kv; bs += SAW_BK) {
    const int bk = min(SAW_BK, T_kv - bs);
    TD tK(const_cast<device VPIPE_ELT*>(kbase + (int64_t)bs * SAW_D),
          dextents<int32_t,2>(SAW_D, bk));
    auto cS = opQK.template get_destination_cooperative_tensor<TD, TD, float>();
    opQK.run(tQ, tK, cS);
    TS tSf(Sf, dextents<int32_t,2>(SAW_BK, SAW_BQ));
    cS.store(tSf);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int r = (int)lid; r < SAW_BQ; r += THREADS) {
      const bool qok = (q0 + r) < n_q;
      float mloc = mrow[r];
      for (int j = 0; j < SAW_BK; ++j) {
        const bool ok = qok && j < bk;
        const float s = ok ? Sf[r * SAW_BK + j] * scale : -INFINITY;
        Sf[r * SAW_BK + j] = s;
        mloc = max(mloc, s);
      }
      const float corr = exp(mrow[r] - mloc);
      float lloc = lrow[r] * corr;
      for (int j = 0; j < SAW_BK; ++j) {
        const float s = Sf[r * SAW_BK + j];
        const float p = (s == -INFINITY) ? 0.0f : exp(s - mloc);
        Ss[r * SAW_BK + j] = (VPIPE_ELT)p;
        lloc += p;
      }
      corr_s[r] = corr;
      mrow[r] = mloc; lrow[r] = lloc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Rescale this thread's slice of O. index [1] is the m (query-row) axis:
    // matmul2d's destination coordinates follow the tensor extent order
    // (n, m), the same convention the tensors above are built with. The
    // validity check keeps a padded capacity from indexing corr_s out of
    // range; every shipped tile divides evenly (BQ*D/(SG*32) is exact at both
    // widths), so on those it folds away at compile time.
    #pragma clang loop unroll(full)
    for (auto it = cO.begin(); it != cO.end(); ++it) {
      if (!it.is_valid_element()) { continue; }
      *it *= corr_s[it.get_multidimensional_index()[1]];
    }

    TP tP(Ss, dextents<int32_t,2>(SAW_BK, SAW_BQ));
    TD tV(const_cast<device VPIPE_ELT*>(vbase + (int64_t)bs * SAW_D),
          dextents<int32_t,2>(SAW_D, bk));
    opPV.run(tP, tV, cO);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  #pragma clang loop unroll(full)
  for (auto it = cO.begin(); it != cO.end(); ++it) {
    if (!it.is_valid_element()) { continue; }
    const auto ij = it.get_multidimensional_index();
    const int r = (int)ij[1], dd = (int)ij[0];
    if (q0 + r < n_q) {
      const float inv = (lrow[r] > 0.0f) ? 1.0f / lrow[r] : 0.0f;
      out[((int64_t)h * n_q + (q0 + r)) * SAW_D + dd] = (VPIPE_ELT)(*it * inv);
    }
  }
}
#define SAW_INST(NAME, D, BQ, BK, SG)                                    \
  template [[host_name(NAME)]] [[kernel]]                                \
  decltype(sdpa_full_mma2_wide_tmpl<D, BQ, BK, SG>)                      \
      sdpa_full_mma2_wide_tmpl<D, BQ, BK, SG>;
// (BK, SG) per query tile are TUNED, not free parameters -- only BQ is a knob
// the host passes. M5, D=512, n=12288, interleaved arms, best of 5:
//   BQ=8 (the base kernel)            161 ms   1.00x   38.7 GB of K/V
//   BQ=16 BK=32  SG=4                  70 ms   2.31x   19.3 GB
//   BQ=32 BK=32  SG=8                  65 ms   2.48x    9.7 GB
//   BQ=32 BK=64  SG=8                  55 ms   2.91x
//   BQ=32 BK=128 SG=8                  51 ms   3.25x   <-- shipped
//   BQ=64 BK=32  SG=8                 109 ms   1.48x    4.8 GB  (spills)
//   BQ=64 BK=64  SG=16                 59 ms   2.81x
// Two effects cross here. Halving the threadgroup count halves the K/V bytes,
// which is worth ~2.3x by itself; past BQ=32 the register accumulator
// (BQ*D/(SG*32) floats per thread) starts to cost more occupancy than the
// traffic it saves -- BQ=64 at SG=8 is 128 floats a thread and LOSES to the
// base kernel's bandwidth. Raising SG instead of BQ does not help either (the
// same tile over 512 threads measured 74 ms against 55): fewer elements per
// thread, but the matmul tile is then split too finely to keep the matrix
// units fed. Deeper BK is nearly free -- it does not change the bytes, it
// amortizes the per-block barriers -- and 128 is where the f32 score tile
// (BQ*BK*4 = 16 KB, plus 8 KB of P) runs out of the 32 KB threadgroup budget.
SAW_INST("sdpa_full_mma2_d384_q16_f16", 384, 16, 32, 4)
SAW_INST("sdpa_full_mma2_d384_q32_f16", 384, 32, 128, 8)
SAW_INST("sdpa_full_mma2_d384_q64_f16", 384, 64, 64, 16)
SAW_INST("sdpa_full_mma2_d512_q16_f16", 512, 16, 32, 4)
SAW_INST("sdpa_full_mma2_d512_q32_f16", 512, 32, 128, 8)
SAW_INST("sdpa_full_mma2_d512_q64_f16", 512, 64, 64, 16)
#undef SAW_INST

#else
// Tensor ops unavailable for this target: stubs so the metallib still builds.
kernel void sdpa_mma_f16(device VPIPE_ELT* out [[buffer(3)]],
                         uint t [[thread_position_in_grid]])
{ if (t == 0) { out[0] = (VPIPE_ELT)0; } }
kernel void sdpa_full_mma2_d384_f16(device VPIPE_ELT* out [[buffer(3)]],
                                    uint t [[thread_position_in_grid]])
{ if (t == 0) { out[0] = (VPIPE_ELT)0; } }
kernel void sdpa_full_mma2_d512_f16(device VPIPE_ELT* out [[buffer(3)]],
                                    uint t [[thread_position_in_grid]])
{ if (t == 0) { out[0] = (VPIPE_ELT)0; } }
#define SAW_STUB(NAME)                                                   \
  kernel void NAME(device VPIPE_ELT* out [[buffer(3)]],                  \
                   uint t [[thread_position_in_grid]])                   \
  { if (t == 0) { out[0] = (VPIPE_ELT)0; } }
SAW_STUB(sdpa_full_mma2_d384_q16_f16)
SAW_STUB(sdpa_full_mma2_d384_q32_f16)
SAW_STUB(sdpa_full_mma2_d384_q64_f16)
SAW_STUB(sdpa_full_mma2_d512_q16_f16)
SAW_STUB(sdpa_full_mma2_d512_q32_f16)
SAW_STUB(sdpa_full_mma2_d512_q64_f16)
#undef SAW_STUB
kernel void sdpa_mma_d128_f16(device VPIPE_ELT* out [[buffer(3)]],
                              uint t [[thread_position_in_grid]])
{ if (t == 0) { out[0] = (VPIPE_ELT)0; } }
kernel void sdpa_causal_mma2_d512_f16(device VPIPE_ELT* out [[buffer(3)]],
                                      uint t [[thread_position_in_grid]])
{ if (t == 0) { out[0] = (VPIPE_ELT)0; } }
kernel void sdpa_paged_mma2_d512_f16(device VPIPE_ELT* out [[buffer(3)]],
                                     uint t [[thread_position_in_grid]])
{ if (t == 0) { out[0] = (VPIPE_ELT)0; } }
kernel void sdpa_causal_mma2_d256_f16(device VPIPE_ELT* out [[buffer(3)]],
                                      uint t [[thread_position_in_grid]])
{ if (t == 0) { out[0] = (VPIPE_ELT)0; } }
kernel void sdpa_full_mma2_d64_f16(device VPIPE_ELT* out [[buffer(3)]],
                                   uint t [[thread_position_in_grid]])
{ if (t == 0) { out[0] = (VPIPE_ELT)0; } }
kernel void sdpa_causal_mma2_d64_f16(device VPIPE_ELT* out [[buffer(3)]],
                                     uint t [[thread_position_in_grid]])
{ if (t == 0) { out[0] = (VPIPE_ELT)0; } }
#endif

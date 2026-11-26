// Copyright © 2024-25 Apple Inc.

#include "mlx/backend/metal/kernels/steel/attn/nax.h"
#include "mlx/backend/metal/kernels/steel/attn/params.h"
#include "mlx/backend/metal/kernels/steel/attn/transforms.h"
#include "mlx/backend/metal/kernels/steel/utils.h"

using namespace mlx::steel;

///////////////////////////////////////////////////////////////////////////////
// GEMM kernels
///////////////////////////////////////////////////////////////////////////////

constant bool align_Q [[function_constant(200)]];
constant bool align_K [[function_constant(201)]];

constant bool has_mask [[function_constant(300)]];
constant bool do_causal [[function_constant(301)]];
constant bool has_sinks [[function_constant(302)]];
// vpipe: BLOCK-SPARSE attention, the matrix-core twin of the same
// constant in steel_attention.h. Same slot, same buffers, same closed-
// form edge mask -- the two kernels are one predicate on two tile sizes,
// which is what lets a windowed forward keep the matrix cores instead of
// falling back to the ALU kernel for the whole block.
//
// Read through is_function_constant_defined for the reason the ALU one
// is: every other caller of this entry point sets 200..302 and stops, so
// a slot they do not know about must mean false to them.
constant bool has_spans_set [[function_constant(303)]];
constant bool has_spans =
    is_function_constant_defined(has_spans_set) ? has_spans_set : false;

// vpipe: export this pass's online-softmax statistics, so a SECOND
// partial attention over a disjoint key set can be merged with it.
//
// The matrix-core twin of the ALU kernel's constant 304, and it exists
// for the same one caller: Sol-Attn runs the blocks its routing KEEPS
// here at full flash throughput and summarises the rest elsewhere, and
// the two halves combine only if both report their running maximum and
// denominator. O is stored ALREADY divided by sum_score, so the merge
// multiplies it back and nothing about the store changes.
//
// Read through is_function_constant_defined, like has_spans: the four
// callers that predate this set 200/201/300..303 and stop there, and
// unset must mean false EXPLICITLY rather than by luck.
constant bool export_ml_set [[function_constant(304)]];
constant bool export_ml =
    is_function_constant_defined(export_ml_set) ? export_ml_set : false;

// vpipe: a per-(query block, KEY) byte mask, which is what lets this
// kernel also run Sol-Attn's APPROXIMATE half.
//
// That half is an attention over the summary sequence -- one key per
// routing block -- from which the blocks the routing kept exact must be
// excluded, or they would be counted twice. The exclusion is decided
// per query block and per key, which is exactly the routing's own flag
// array, so nothing is materialised: `block_mask` IS that array, read
// as [H][params->NQ][params->kL] with a nonzero byte meaning EXCLUDE.
//
// A per-element mask (has_mask) would be [qL, kL] -- 50 million bytes a
// head at video geometry against the flag array's 100 thousand -- and a
// span list cannot say it at all, the CSR's unit being 32 keys where
// this decision is per key.
//
// A ROW WITH EVERY KEY EXCLUDED IS LEGITIMATE (tau low enough keeps
// every block) and produces max_score == finite_min, since that is both
// the initial value and what the mask writes. The caller reads that
// sentinel and drops the half; see sol_merge_ml_mma.
constant bool has_block_mask_set [[function_constant(305)]];
constant bool has_block_mask =
    is_function_constant_defined(has_block_mask_set) ? has_block_mask_set
                                                     : false;

// vpipe: run the QK^T product in INT8, SageAttention-style
// (arXiv:2410.02367). Q and K arrive pre-quantized with ONE scale per
// block -- per query block for Q, per key block for K -- so the dequant
// of a score tile is a single scalar, and it folds into the softmax's
// log2 scale that the f16 path applies here anyway.
//
// P*V IS NOT QUANTIZED and is not affected: the probabilities are a
// softmax output and V carries the outliers, which is the same division
// the paper draws.
//
// WHY IT IS WORTH DOING AT ALL, on a part where a standalone K=128 GEMM
// shows int8 only 1.16x ahead: this kernel does not issue a 128-deep
// product per key block. It builds it from eight 16-deep fragment MMAs
// chained into the same register fragments, and at THAT shape the
// matrix pipe runs int8 at exactly 2.00x f16 (32.2 against 16.1
// TFLOP/s, attn_qk_i8.frag_rate). The GEMM figure was measuring a
// per-tile store this kernel does not pay.
//
// The caller must also pass the SMOOTHED K -- K minus its mean over
// tokens. That is free and exact rather than an approximation:
// subtracting a per-channel mean shifts every score in a row by the same
// q.mean, and softmax does not see a per-row shift. It is what makes the
// int8 range hold the signal instead of a shared bias, and it is worth
// most of the accuracy (sage_attention.per_block_int8_needs_the_smoothing).
constant bool qk_int8_set [[function_constant(306)]];
constant bool qk_int8 =
    is_function_constant_defined(qk_int8_set) ? qk_int8_set : false;

// vpipe: Q quantized PER TOKEN (q_scale [B, H, qL], one per query row)
// instead of per query block -- what the int8 P*V's prologue makes. A
// score's dequant is then the row's scale by the key block's, which the
// tile applies per row as it is anyway. MEASURED as free, and a quarter
// less error than the block's scale.
constant bool qk_q_token_set [[function_constant(310)]];
constant bool qk_q_token =
    is_function_constant_defined(qk_q_token_set) ? qk_q_token_set : false;

// vpipe: P*V in INT8 too (Sage's pv8), with the int8 QK and Q per token.
// The product is HELD in int32 across key blocks -- no f32 O tile until
// the end: P as uint8 against the row's running max (0..255), V minus its
// mean over tokens with one scale per channel over the whole sequence,
// offset by 128 to uint8 as well (MPP multiplies two uint8 into int32).
// When a row's max grows the held sum is rescaled by the online softmax's
// own factor -- in float, rounded back: exact to 2^24, and the sum is far
// under 2^31 -- after the offset's 128 a key is taken out with P's exact
// integer row sum, which also happens every 64 visited blocks and at the
// end. A row whose held sum passes 2^30 has P's grid stepped down (255,
// 127, 63, ...) as if its max had grown -- by lvl/2 over lvl, so that a
// P of exactly the max stays on the grid -- and int32 is never overrun.
// V's scale and mean come back after the normalize.
//
// WHY HELD: dequantizing each block's product into an f32 O cost more
// ALU than the int8 product saved -- 61 ms against SageAttention's 50 at
// 8 heads x 20036 x 128 on the M5 Pro. Held, 43.8 (bf16: 44.6 against
// 49.3). The price is P's grid, 8 bits against the running max: its error
// grows on flat rows over long sequences (cosine 0.99966 against the
// dense kernel at 20k rows of random keys, SageAttention 0.99994;
// sage_attention.int8_pv_held).
constant bool pv_int8_set [[function_constant(308)]];
constant bool pv_int8 =
    is_function_constant_defined(pv_int8_set) ? pv_int8_set : false;

// vpipe: SINKS UNDER THE SMOOTHED INT8 QK. K is quantized as K - mean(K)
// over tokens, which moves every key's score in a row by -q.mean -- exact
// for a softmax over the keys alone, but a sink is a logit of its own and
// must move with them, or its share of the row's mass is off by
// exp(q.mean). With this (and has_sinks, qk_int8) each row's sink is
// taken as sink - scale * q.mean, q the row's query as bound at buffer 0
// and the mean K's as the prologue subtracted it ([KVH, BD] f32 at buffer
// 22, MetalSageAttention::bind). MEASURED before it (sage_attention.
// int8_pv_held, sinks taking a large share): cosine 0.76 against f64.
constant bool qk_sink_shift_set [[function_constant(311)]];
constant bool qk_sink_shift =
    is_function_constant_defined(qk_sink_shift_set) ? qk_sink_shift_set
                                                    : false;

template <typename T>
struct TransformScale {
  T scale;
  METAL_FUNC TransformScale(T scale_) thread : scale(scale_) {}

  METAL_FUNC T apply(T x) const thread {
    return scale * x;
  }
};

struct MaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};

struct SumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct MulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct SubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x - y;
  }
};

struct ExpSubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return fast::exp2(x - y);
  }
};

struct DivOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x / y;
  }
};

// clang-format off
template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int WM,
    int WN,
    typename MaskType = float,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attention_nax(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant AttnParams* params [[buffer(4)]],
    const constant AttnMaskParams* mask_params [[buffer(5), function_constant(has_mask)]],
    const device MaskType* mask [[buffer(6), function_constant(has_mask)]],
    const device T* sinks [[buffer(7), function_constant(has_sinks)]],
    // vpipe block-sparse: qb_off[NQ + 1] indexes qb_blocks, which lists
    // the key BLOCK indices each query block visits, ascending. Built
    // for THIS kernel's BQ/BK -- a list built for the ALU kernel's 32x16
    // read against 64x32 blocks is not a slower answer, it is a
    // different window.
    const device int* qb_off [[buffer(8), function_constant(has_spans)]],
    const device int* qb_blocks [[buffer(9), function_constant(has_spans)]],
    const constant AttnSpanParams* span_params
        [[buffer(10), function_constant(has_spans)]],
    // [num_frames] window bounds, inclusive, UNCLAMPED -- lo < 0 and
    // hi >= num_frames are how "nothing outside on that side" is said.
    const device int2* span_bounds
        [[buffer(11), function_constant(has_spans)]],
    // vpipe: [B, H, qL] each, this pass's per-row max and denominator.
    device float* ml_max [[buffer(12), function_constant(export_ml)]],
    device float* ml_sum [[buffer(13), function_constant(export_ml)]],
    // vpipe: [H, NQ, kL] bytes; nonzero excludes that key from this
    // query block.
    const device uchar* block_mask
        [[buffer(14), function_constant(has_block_mask)]],
    // vpipe: the int8 QK operands and their per-block scales. Q8/K8 are
    // [B, H, qL|kL, BD] int8, tightly packed; the scales are [B, H, NQ]
    // and [B, H, NK] fp32, one per block of the kernel's own tiling (Q's
    // [B, H, qL] with qk_q_token).
    const device int8_t* Q8 [[buffer(15), function_constant(qk_int8)]],
    const device int8_t* K8 [[buffer(16), function_constant(qk_int8)]],
    const device float* q_scale [[buffer(17), function_constant(qk_int8)]],
    const device float* k_scale [[buffer(18), function_constant(qk_int8)]],
    // vpipe: the int8 P*V's: V8 [B, KVH, kL, BD] uint8 -- V minus its mean
    // over tokens, quantized, plus 128 -- tightly packed; its scale and
    // mean [B, KVH, BD] fp32.
    const device uint8_t* V8 [[buffer(19), function_constant(pv_int8)]],
    const device float* v_scale [[buffer(20), function_constant(pv_int8)]],
    const device float* v_mean [[buffer(21), function_constant(pv_int8)]],
    // vpipe: K's mean as the smoothing subtracted it (qk_sink_shift).
    const device float* k_mean [[buffer(22),
                                 function_constant(qk_sink_shift)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on

  // Pacifying compiler
  (void)lid;
  (void)simd_lane_id;

  // Move to correct block
  ulong3 tidl{tid.x, tid.y, tid.z};

  Q += tidl.z * params->Q_strides[0] + // Batch
      tidl.y * params->Q_strides[1] + // Head
      tidl.x * BQ * params->Q_strides[2]; // Sequence

  ulong kv_head_idx = int(tid.y) / params->gqa_factor;
  K += tidl.z * params->K_strides[0] + // Batch
      kv_head_idx * params->K_strides[1]; // Head

  V += tidl.z * params->V_strides[0] + // Batch
      kv_head_idx * params->V_strides[1]; // Head

  O += tidl.z * params->O_strides[0] + // Batch
      tidl.y * params->O_strides[1] + // Head
      tidl.x * BQ * params->O_strides[2]; // Sequence

  // vpipe: the int8 twins, tightly packed [B, H, L, BD] -- their row
  // stride is BD and not the f16 tensors', which may be a fused
  // projection's. Advanced in step with K below.
  const device int8_t* Q8b = nullptr;
  const device int8_t* K8b = nullptr;
  const device float* qsc = nullptr;
  const device float* ksc = nullptr;
  if (qk_int8) {
    // Q is indexed by the QUERY head and K by the KV head, exactly as
    // the f16 tensors above are. Reducing K8 to the query head would be
    // invisible at gqa_factor 1 -- which is what H3 and Qwen-Image run --
    // and would read a neighbouring head's keys on every GQA family
    // (FLUX.2 and Krea-2 are both HED/KVH > 1). So the KV base is its
    // own arithmetic, over KVH = H / gqa_factor heads.
    const size_t hbase =
        (size_t(tid.z) * size_t(params->H) + size_t(tid.y));
    const size_t kvh =
        size_t(params->H) / size_t(max(1, params->gqa_factor));
    const size_t kvbase = size_t(tid.z) * kvh + size_t(kv_head_idx);
    Q8b = Q8 + (hbase * size_t(params->qL) + size_t(tid.x) * BQ) * BD;
    K8b = K8 + kvbase * size_t(params->kL) * BD;
    qsc = qk_q_token
        ? q_scale + hbase * size_t(params->qL) + size_t(tid.x) * BQ
        : q_scale + hbase * size_t(params->NQ) + size_t(tid.x);
    ksc = k_scale + kvbase * size_t(params->NK);
  }
  // vpipe: the int8 P*V's V, by the KV head as K8 is.
  const device uint8_t* V8b = nullptr;
  const device float* vsc = nullptr;
  const device float* vmu = nullptr;
  if (pv_int8) {
    const size_t kvh =
        size_t(params->H) / size_t(max(1, params->gqa_factor));
    const size_t kvbase = size_t(tid.z) * kvh + size_t(kv_head_idx);
    V8b = V8 + kvbase * size_t(params->kL) * BD;
    vsc = v_scale + kvbase * BD;
    vmu = v_mean + kvbase * BD;
  }

  if (has_mask) {
    mask += tidl.z * mask_params->M_strides[0] + // Batch
        tidl.y * mask_params->M_strides[1]; // Head
  }

  const metal::uniform<float> scale2 =
      make_uniform(params->scale) * make_uniform(1.44269504089f);

  // Prepare MMA tiles
  constexpr short kU = 16;

  constexpr int kNWarps = WM * WN;
  static_assert(
      BQ >= (kNWarps * kU) && BQ % (kNWarps * kU) == 0,
      "Each simdgroup must host atleast 1 simdgroup matrix along Q sequence.");

  // Q seq frags per warp
  constexpr int TQ = BQ / (kNWarps * kU);
  // HeadDim frags (all warps load the same frags)
  constexpr int TD = BD / kU;
  // KV seq frags per warp
  constexpr short TK = BK / kU;

  static_assert(TQ == 1, "Check TQ");
  using otile_t = NAXTile<AccumType, TQ, TD>;
  otile_t Otile;

  Otile.clear();

  // Prepare mma tile offsets
  const short tm = kU * TQ * simd_group_id;
  Q += tm * int(params->Q_strides[2]);
  if (qk_int8) { Q8b += tm * BD; }

  const short2 simd_coord = otile_t::NAXFrag_t::get_coord();
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;

  // Init row reduction variables
  constexpr short kRowsPT = otile_t::kRowsPerThread;

  metal::vec<AccumType, kRowsPT> max_score;
  metal::vec<AccumType, kRowsPT> sum_score{0};

  // Init to -Inf
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<AccumType>::finite_min;
  }

  // vpipe: each row's sink as the scores see it -- shifted by the smoothed
  // int8 QK's q.mean where qk_sink_shift says (see it above).
  metal::vec<AccumType, kRowsPT> sink_row{0};
  if (has_sinks) {
    const AccumType s0 = M_LOG2E_F * static_cast<AccumType>(sinks[tidl.y]);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      AccumType s2 = s0;
      if (qk_int8 && qk_sink_shift) {
        const int r = sm + i * otile_t::kFragRowsJump;
        if (int(tid.x) * BQ + tm + r < params->qL) {
          const size_t kvh =
              size_t(params->H) / size_t(max(1, params->gqa_factor));
          const device float* mu =
              k_mean + (size_t(tid.z) * kvh + size_t(kv_head_idx)) * BD;
          const device T* qr = Q + int64_t(r) * params->Q_strides[2];
          float dot = 0.0f;
          for (short c = 0; c < BD; ++c) {
            dot = fma(float(qr[c]), mu[c], dot);
          }
          s2 -= AccumType(float(scale2) * dot);
        }
      }
      sink_row[i] = s2;
      // Under the int8 P*V the sink stays OUT of the running max: P is
      // quantized to 8 bits against that max, and a sink above the keys'
      // scores would squeeze them into the bottom of the grid while
      // holding no V itself. Its mass is added to the sum after the loop
      // -- the same softmax.
      if (!pv_int8) {
        max_score[i] = s2;
        sum_score[i] = 1;
      }
    }
  }

  int kb_lim = params->NK;
  int kb_min_causal = params->NK;

  if (do_causal) {
    int q_max = (tid.x + 1) * BQ + params->qL_off;
    kb_lim = (q_max + BK - 1) / BK;
    kb_lim = min(params->NK, kb_lim);

    int q_min = tid.x * BQ + params->qL_off;
    q_min = max(0, q_min);
    kb_min_causal = (q_min / BK);
  }

  const bool is_last_bq = int(tid.x) == (params->NQ_aligned);
  // const bool is_last_tq = int(simd_group_id) >= (params->qL_rem / UQ);
  const bool is_last_q = is_last_bq;

  const short lim_rows_q = params->qL_rem - tm;
  const short lim_rows_k = params->kL_rem;

  // vpipe block-sparse: walk the query block's OWN list of key blocks
  // instead of all of them. `nvis` is how many it visits and `kb` is
  // which, so every position computed from `kb` below -- the length
  // mask, the causal bound, the edge mask -- stays the real column.
  int nvis = kb_lim;
  int vis_base = 0;
  int kb_prev = 0;
  if (has_spans) {
    // qb_stride == 0 is one list shared by every head; anything else
    // lays qb_off out as [H][NQ + 1]. See AttnSpanParams -- a geometric
    // window does not depend on the head and Sol's routing does.
    const int qbi = int(tid.y) * span_params->qb_stride + int(tid.x);
    vis_base = qb_off[qbi];
    nvis = qb_off[qbi + 1] - vis_base;
  }

  // vpipe: per-token Q -- this thread's rows' scales, read once.
  metal::vec<float, kRowsPT> q_row;
  if (qk_int8 && qk_q_token) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      const int r = tm + sm + i * otile_t::kFragRowsJump;
      q_row[i] = (r < (is_last_q ? lim_rows_q + tm : BQ)) ? qsc[r] : 0.0f;
    }
  }

  // vpipe: the int8 P*V's held product; P's row sums since the offset was
  // last taken out, this thread's columns only; and the level a row's
  // largest P lands on (255 until a row is stepped down for room).
  NAXTile<int32_t, TQ, TD> Oacc;
  int r_part[kRowsPT];
  int p_lvl[kRowsPT];
  if (pv_int8) {
    Oacc.clear();
  }
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    r_part[i] = 0;
    p_lvl[i] = 255;
  }
// The offset taken out: each row's P sum across its four lanes (as
// NAXFrag's row_reduce pairs them), 128 a key.
#define VPIPE_PV8_FIX()                                                   \
  {                                                                       \
    STEEL_PRAGMA_UNROLL                                                   \
    for (short ii = 0; ii < kRowsPT; ii++) {                              \
      int r = r_part[ii];                                                 \
      r += simd_shuffle_xor(r, ushort(1));                                \
      r += simd_shuffle_xor(r, ushort(8));                                \
      r_part[ii] = 0;                                                     \
      const int off = 128 * r;                                            \
      STEEL_PRAGMA_UNROLL                                                 \
      for (short id = 0; id < TD; id++) {                                 \
        thread auto& oi = Oacc.frag_at(0, id);                            \
        STEEL_PRAGMA_UNROLL                                               \
        for (short jj = 0; jj < otile_t::kFragThrCols; jj++) {            \
          oi[ii * otile_t::kFragThrCols + jj] -= off;                     \
        }                                                                 \
      }                                                                   \
    }                                                                     \
  }

  // Loop over KV seq length
  for (int vi = 0; vi < nvis; vi++) {
    const int kb = has_spans ? qb_blocks[vis_base + vi] : vi;
    if (has_spans) {
      // K and V only ever move FORWARD, so the step is the gap since the
      // last visited block -- kb itself on the first iteration, which is
      // how a query block whose window starts late skips straight to it.
      // int64 ON PURPOSE. The dense loop steps one block at a time and
      // accumulates through the POINTER, so its increment is small
      // whatever the sequence; a sparse jump is the GAP, which for the
      // first visited block is the block index itself. At video geometry
      // K_strides[2] is 3*inner = 21504, so a 32-bit product wraps at
      // 3120 key blocks -- 99840 rows, which is a 27-second clip and not
      // a hypothetical. It wraps silently: the loads come back from
      // somewhere else in the same buffer.
      K += (int64_t)(kb - kb_prev) * BK * params->K_strides[2];
      V += (int64_t)(kb - kb_prev) * BK * params->V_strides[2];
      kb_prev = kb;
    }
    const int is_last_k = (kb == (params->NK_aligned));

    // Do S = Q @ K.T
    using stile_t = NAXTile<AccumType, TQ, TK>;
    stile_t Stile;

    Stile.clear();

    // vpipe: the INT8 product, when the caller supplied one.
    //
    // Same loop, same fragment shapes, same accumulation order -- only
    // the operand type and the accumulator change, and the dequant
    // scalar replaces the scale multiply the f16 path does below. The
    // int8 operands are indexed by the ABSOLUTE key block rather than by
    // an advancing pointer, so the sparse jump above needs no twin.
    if (qk_int8) {
      using itile_t = NAXTile<int32_t, TQ, TK>;
      itile_t Si;
      Si.clear();
      const device int8_t* K8t = K8b + (size_t)kb * BK * BD;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik += 2) {
          STEEL_PRAGMA_UNROLL
          for (short id = 0; id < TD; id++) {
            NAXTile<int8_t, 1, 1> Qt;
            NAXTile<int8_t, 2, 1> Kt;
            const int qoff = iq * kU * BD + id * kU;
            const int koff = ik * kU * BD + id * kU;
            if (!align_Q && is_last_q) {
              Qt.load_rows(Q8b + qoff, BD, lim_rows_q - iq * kU);
            } else {
              Qt.load(Q8b + qoff, BD);
            }
            if (!align_K && is_last_k) {
              Kt.load_rows(K8t + koff, BD, lim_rows_k - ik * kU);
            } else {
              Kt.load(K8t + koff, BD);
            }
            itile_t::NAXFrag_t::template mma<int32_t, int8_t, int8_t>(
                Si.frag_at(iq, ik),
                Si.frag_at(iq, ik + 1),
                Qt.frag_at(0, 0),
                metal::false_type{},
                Kt.frag_at(0, 0),
                Kt.frag_at(1, 0),
                metal::true_type{});
          }
        }
      }
      // ONE SCALAR for the whole tile: the two block scales and the
      // softmax's log2 factor, which is what makes per-block
      // quantization the right granularity for a flash kernel.
      if (qk_q_token) {
        // Per token: the row's Q scale by the key block's.
        const float kd = ksc[kb] * float(scale2);
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          thread auto& si = Si.frag_at(0, ik);
          thread auto& sf = Stile.frag_at(0, ik);
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            const float d = q_row[ii] * kd;
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto loc = ii * stile_t::kFragThrCols + jj;
              sf[loc] = (AccumType)((float)si[loc] * d);
            }
          }
        }
      } else {
        const float dq = qsc[0] * ksc[kb] * float(scale2);
        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {
          Stile.elems()[ii] = (AccumType)((float)Si.elems()[ii] * dq);
        }
      }
    } else {

    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik += 2) {
        // Unrolling the head-dim loop by 4, rather than fully, potentially
        // lets the compiler interleave the next K-tile loads with the running
        // mma chain instead of hoisting all TD loads up front (MLX #3843).
        // vpipe MEASURED on the M5 Pro, 56x128 bf16: +2.6 to +3.8% at 2.3k-19k
        // rows, A-B-A within 0.5%.
#pragma clang loop unroll_count(4)
        for (short id = 0; id < TD; id++) {
          NAXTile<T, 1, 1> Qtile;
          NAXTile<T, 2, 1> Ktile;

          const int Q_load_off = iq * kU * int(params->Q_strides[2]) + id * kU;
          const int K_load_off = ik * kU * int(params->K_strides[2]) + id * kU;

          if (!align_Q && is_last_q) {
            Qtile.load_rows(
                Q + Q_load_off,
                int(params->Q_strides[2]),
                lim_rows_q - iq * kU);
          } else {
            Qtile.load(Q + Q_load_off, int(params->Q_strides[2]));
          }

          if (!align_K && is_last_k) {
            Ktile.load_rows(
                K + K_load_off,
                int(params->K_strides[2]),
                lim_rows_k - ik * kU);
          } else {
            Ktile.load(K + K_load_off, int(params->K_strides[2]));
          }

          stile_t::NAXFrag_t::mma(
              Stile.frag_at(iq, ik),
              Stile.frag_at(iq, ik + 1),
              Qtile.frag_at(0, 0),
              metal::false_type{},
              Ktile.frag_at(0, 0),
              Ktile.frag_at(1, 0),
              metal::true_type{});
        }
      }
    }

    // Scale S
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= float(scale2);
    }
    }   // !qk_int8

    // Mask out length sequence
    if (!align_K && is_last_k) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const short col_pos = ik * kU + sn;

          thread auto& fg = Stile.frag_at(iq, ik);

          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto loc = ii * stile_t::kFragThrCols + jj;
              fg[loc] = ((col_pos + jj) < params->kL_rem) ? fg[loc] : neg_inf;
            }
          }
        }
      }
    }

    // Mask out if causal
    if (do_causal && kb >= kb_min_causal) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      const int base_row = tid.x * BQ + params->qL_off + tm;
      const int base_col = kb * BK;

      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          thread auto& fg = Stile.frag_at(iq, ik);

          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto r =
                  base_row + iq * kU + ii * stile_t::kFragRowsJump + sm;
              const auto c = base_col + ik * kU + jj + sn;
              const auto loc = ii * stile_t::kFragThrCols + jj;
              fg[loc] = (r < c) ? neg_inf : fg[loc];
            }
          }
        }
      }
    }

    // vpipe block-sparse: the block list is rounded OUTWARD to whole key
    // blocks, so a block at a span's edge carries keys the mask forbids.
    // This is WindowMask::allows, element by element and in closed form
    // -- nothing is read but the frame's own [lo, hi]. Byte for byte the
    // predicate the ALU kernel applies; only the fragment indexing
    // differs, which is why the two agree to the accumulation floor.
    if (has_spans) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      const int vs = span_params->video_start;
      const int tpf = span_params->tokens_per_frame;
      const int nf = span_params->num_frames;
      const int anc = span_params->anchors;
      const int ve = vs + nf * tpf;
      const int base_row = tid.x * BQ + params->qL_off + tm;
      const int base_col = kb * BK;

      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
          const int row_pos =
              base_row + iq * kU + ii * stile_t::kFragRowsJump + sm;
          // A query row outside the video block is dense both ways, and
          // so is every row past the sequence -- those are already -inf
          // from the length mask and must not be revived here.
          if (!(row_pos >= vs && row_pos < ve && tpf > 0)) { continue; }
          int qf = (row_pos - vs) / tpf;
          qf = qf < 0 ? 0 : (qf > nf - 1 ? nf - 1 : qf);
          const int2 b = span_bounds[qf];
          const bool q_anchor = (qf == 0) || (qf == nf - 1);
          // Anchor ROWS see everything, so the row is already right.
          if (((anc & 2) != 0) && q_anchor) { continue; }

          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            thread auto& fg = Stile.frag_at(iq, ik);
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const int c = base_col + ik * kU + jj + sn;
              if (c < vs || c >= ve) { continue; }   // a global key: dense
              const int kf = (c - vs) / tpf;
              bool ok = (kf >= b.x && kf <= b.y);
              if (!ok && ((anc & 1) != 0)) {
                ok = (kf == 0) || (kf == nf - 1);
              }
              if (!ok) { fg[ii * stile_t::kFragThrCols + jj] = neg_inf; }
            }
          }
        }
      }
    }

    // vpipe: the routing's own exclusion, per query block and per key.
    // Column-only -- every row of this query block obeys the same flag
    // row -- so the row index never enters it.
    if (has_block_mask) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;
      const device uchar* brow =
          block_mask + (size_t(tid.y) * size_t(params->NQ) + size_t(tid.x)) *
                           size_t(params->kL);
      const int base_col = kb * BK;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          thread auto& fg = Stile.frag_at(iq, ik);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
            const int c = base_col + ik * kU + jj + sn;
            const bool drop = (c >= params->kL) || (brow[c] != 0);
            if (!drop) { continue; }
            STEEL_PRAGMA_UNROLL
            for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
              fg[ii * stile_t::kFragThrCols + jj] = neg_inf;
            }
          }
        }
      }
    }

    // Other masking as needed
    if (has_mask) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      const int base_row = tid.x * BQ + tm;
      const int base_col = kb * BK;

      constexpr bool is_bool = is_same_v<MaskType, bool>;
      using melem_t = typename metal::conditional_t<is_bool, bool, AccumType>;
      using mtile_t = NAXTile<melem_t, TQ, TK>;
      using mfrag_t = typename mtile_t::frag_type;

      if (base_row + BQ <= params->qL && base_col + BK <= params->kL) {
        for (short iq = 0; iq < TQ; iq++) {
          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            const int row_pos = base_row + iq * kU;
            const int col_pos = base_col + ik * kU;

            mfrag_t mfrag;
            mtile_t::NAXFrag_t::load(
                mfrag,
                mask,
                int64_t(mask_params->M_strides[2]),
                Int<1>{},
                row_pos,
                col_pos);

            thread auto& fg = Stile.frag_at(iq, ik);

            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < mtile_t::kElemsPerFrag; jj++) {
              if constexpr (is_bool) {
                fg[jj] = mfrag[jj] ? fg[jj] : neg_inf;
              } else {
                fg[jj] += M_LOG2E_F * AccumType(mfrag[jj]);
              }
            }
          }
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < TQ; iq++) {
          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            const int row_pos = base_row + iq * kU;
            const int col_pos = base_col + ik * kU;

            mfrag_t mfrag;
            mtile_t::NAXFrag_t::load_safe(
                mfrag,
                mask,
                int64_t(mask_params->M_strides[2]),
                Int<1>{},
                params->qL,
                params->kL,
                row_pos,
                col_pos);

            thread auto& fg = Stile.frag_at(iq, ik);

            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < mtile_t::kElemsPerFrag; jj++) {
              if constexpr (is_bool) {
                fg[jj] = mfrag[jj] ? fg[jj] : neg_inf;
              } else {
                fg[jj] += M_LOG2E_F * AccumType(mfrag[jj]);
              }
            }
          }
        }
      }
    }

    // Do softmax

    // Temp variables
    metal::vec<AccumType, kRowsPT> new_max;
    metal::vec<AccumType, kRowsPT> factor;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }

    // Row max
    Stile.template row_reduce<MaxOp>(new_max);

    // exp(Si - rowmax(Si))
    Stile.template row_bin_op<ExpSubOp>(new_max);

    // vpipe: whether any row of the tile has a new max -- the int8 P*V's
    // cue to rescale its held sum.
    bool grew = false;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      grew = grew || (new_max[i] > max_score[i]);
    }

    // Factor exp(rowmax(Si) - rowmax(Si-1))
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }

    // Row Sum
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i];
    }

    Stile.template row_reduce<SumOp>(sum_score);

    // Update O
    if (pv_int8) {
      // The held sum rescaled -- the offset out first, being in the OLD
      // units -- and a grown row's grid back to 255.
      if (simd_any(grew)) {
        VPIPE_PV8_FIX();
        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < kRowsPT; ii++) {
          if (factor[ii] < 1.0f) { p_lvl[ii] = 255; }
        }
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < TD; id++) {
          thread auto& oi = Oacc.frag_at(0, id);
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < otile_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < otile_t::kFragThrCols; jj++) {
              const auto loc = ii * otile_t::kFragThrCols + jj;
              oi[loc] = (int32_t)metal::rint((float)oi[loc] * factor[ii]);
            }
          }
        }
      }
    } else {
      Otile.template row_bin_op<MulOp>(factor);
    }

    simdgroup_barrier(mem_flags::mem_none);

    if (pv_int8) {
      // vpipe: Oacc += Pu @ Vu, P on 0..255, V + 128 -- uint8 both.
      NAXTile<uint8_t, TQ, TK> Pu;
      STEEL_PRAGMA_UNROLL
      for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {
        Pu.elems()[ii] = (uint8_t)metal::clamp(
            metal::rint(Stile.elems()[ii] * 255.0f), 0.0f, 255.0f);
      }
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        thread auto& pf = Pu.frag_at(0, ik);
        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
            r_part[ii] += (int)pf[ii * stile_t::kFragThrCols + jj];
          }
        }
      }
      const device uint8_t* Vut = V8b + (size_t)kb * BK * BD;
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TD; id += 2) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          NAXTile<uint8_t, 1, 2> Vt;
          const int voff = ik * kU * BD + id * kU;
          if (!align_K && is_last_k) {
            Vt.load_rows(Vut + voff, BD, lim_rows_k - ik * kU);
          } else {
            Vt.load(Vut + voff, BD);
          }
          NAXTile<int32_t, 1, 2>::NAXFrag_t::template
              mma<int32_t, uint8_t, uint8_t>(
              Oacc.frag_at(0, id),
              Oacc.frag_at(0, id + 1),
              Pu.frag_at(0, ik),
              metal::false_type{},
              Vt.frag_at(0, 0),
              Vt.frag_at(0, 1),
              metal::false_type{});
        }
      }
      // Every 64 VISITED blocks (a span list jumps kb): the offset out,
      // and a row whose held sum has passed 2^30 stepped down -- as if
      // its max had grown: the sum, its softmax sum and its max moved
      // together by lvl/2 over lvl (127/255, 63/127, ...), not by 1/2,
      // which would put a P of exactly the max at 127.5 and round every
      // later key of a uniform row the same way. The row's ACTUAL largest
      // |sum|, not a bound. 64 blocks add at most 2^28, so 2^31 is never
      // reached; only near-uniform attention onto a long run of
      // same-signed V gets there.
      if ((vi & 63) == 63) {
        VPIPE_PV8_FIX();
        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < kRowsPT; ii++) {
          int big = 0;
          STEEL_PRAGMA_UNROLL
          for (short id = 0; id < TD; id++) {
            thread auto& oi = Oacc.frag_at(0, id);
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < otile_t::kFragThrCols; jj++) {
              big = metal::max(
                  big, metal::abs(oi[ii * otile_t::kFragThrCols + jj]));
            }
          }
          big = metal::max(big, simd_shuffle_xor(big, ushort(1)));
          big = metal::max(big, simd_shuffle_xor(big, ushort(8)));
          if (big > (1 << 30) && p_lvl[ii] > 1) {
            const int nl = p_lvl[ii] >> 1;
            const float f = (float)nl / (float)p_lvl[ii];
            p_lvl[ii] = nl;
            sum_score[ii] *= f;
            max_score[ii] -= metal::precise::log2(f);
            STEEL_PRAGMA_UNROLL
            for (short id = 0; id < TD; id++) {
              thread auto& oi = Oacc.frag_at(0, id);
              STEEL_PRAGMA_UNROLL
              for (short jj = 0; jj < otile_t::kFragThrCols; jj++) {
                const auto loc = ii * otile_t::kFragThrCols + jj;
                oi[loc] = (int32_t)metal::rint((float)oi[loc] * f);
              }
            }
          }
        }
      }
    } else {

    // Do O = P @ V
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TD; id += 2) {
        if constexpr (BD == 128) {
          if (id == 4) {
            threadgroup_barrier(mem_flags::mem_none);
          }
        }

        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          NAXTile<T, 1, 2> Vtile;

          const int V_load_off = ik * kU * int(params->V_strides[2]) + id * kU;

          if (!align_K && is_last_k) {
            Vtile.load_rows(
                V + V_load_off,
                int(params->V_strides[2]),
                lim_rows_k - ik * kU);
          } else {
            Vtile.load(V + V_load_off, int(params->V_strides[2]));
          }

          otile_t::NAXFrag_t::mma(
              Otile.frag_at(iq, id),
              Otile.frag_at(iq, id + 1),
              Stile.frag_at(iq, ik),
              metal::false_type{},
              Vtile.frag_at(0, 0),
              Vtile.frag_at(0, 1),
              metal::false_type{});
        }
      }
    }

    }   // !pv_int8

    // Prepare for next iteration. The sparse path steps at the TOP
    // instead, because the gap is a property of the block it is about to
    // read and not of the one it has just finished.
    if (!has_spans) {
      K += BK * int(params->K_strides[2]);
      V += BK * int(params->V_strides[2]);
    }
  }

  // vpipe: the sink's mass, kept out of the int8 P*V's running max (see
  // its init), now in the sum: exp2(sink - max), the max as rescaled and
  // stepped down. A row that attended no key reads +inf here, and is zero
  // below as an empty row is.
  if (pv_int8 && has_sinks) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] += fast::exp2(sink_row[i] - max_score[i]);
    }
  }

  // vpipe: the int8 P*V's held product into O, its offset out first.
  if (pv_int8) {
    VPIPE_PV8_FIX();
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < otile_t::kElemsPerTile; ii++) {
      Otile.elems()[ii] =
          (AccumType)Oacc.elems()[ii] * AccumType(1.0f / 255.0f);
    }
  }
#undef VPIPE_PV8_FIX

  // vpipe: the statistics, BEFORE the normalize consumes them.
  //
  // row_reduce leaves every lane of a row holding the same value -- it
  // finishes with two simd_shuffle_xor steps across exactly the lanes
  // that share the row -- so one of the four writes. `sn` is the
  // fragment COLUMN and is 0 for exactly one of them.
  //
  // The row is the O tile's own: TQ == 1 here (asserted above), so the
  // vec index i is the fragment's element row and lands at
  // sm + i * kFragRowsJump, under the same tid.x * BQ + tm the store
  // below uses.
  if (export_ml) {
    if (sn == 0) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kRowsPT; ++i) {
        const int row =
            int(tid.x) * BQ + tm + sm + i * otile_t::kFragRowsJump;
        if (row < params->qL) {
          const size_t o =
              (size_t(tid.z) * params->H + size_t(tid.y)) * params->qL + row;
          ml_max[o] = float(max_score[i]);
          ml_sum[o] = float(sum_score[i]);
        }
      }
    }
  }

  // Normalize output

  threadgroup_barrier(mem_flags::mem_none);

  // vpipe: A ROW THAT ATTENDED NOTHING IS ZERO. A span list can route a
  // query to no key block at all, and then sum_score is still its initial
  // 0 and 1/0 turns the all-zero tile into NaN -- the FlashAttention
  // answer for a query with no keys is 0, and a NaN row poisons every
  // later layer. The per-key mask's all-excluded row is the same row
  // spelled differently: max_score never left the sentinel the mask
  // writes, and without this it would come out a uniform average of keys
  // it excluded. The statistics export above is untouched, so Sol still
  // reads its sentinel.
  metal::vec<AccumType, kRowsPT> rcp;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    const bool empty =
        sum_score[i] == 0 ||
        (has_block_mask && max_score[i] == Limits<AccumType>::finite_min);
    rcp[i] = empty ? AccumType(0) : 1.f / sum_score[i];
  }

  Otile.template row_bin_op<MulOp>(rcp);

  // vpipe: the int8 P*V's V scale and mean back, after the normalize, so a
  // row that attended nothing stays zero. The mean comes back over the
  // KEYS' share of the softmax mass: a sink's share holds no V. Its weight
  // at the end is exp2(sink - max) -- it entered as 1 at max = sink and
  // has been rescaled with the sum ever since, step-downs included -- so
  // the keys hold 1 - that over the sum. Without a sink they hold it all.
  if (pv_int8) {
    metal::vec<AccumType, kRowsPT> keys_share;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      keys_share[i] =
          has_sinks
              ? AccumType(1) - fast::exp2(sink_row[i] - max_score[i]) * rcp[i]
              : AccumType(1);
    }
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TD; id++) {
      thread auto& of = Otile.frag_at(0, id);
      STEEL_PRAGMA_UNROLL
      for (short jj = 0; jj < otile_t::kFragThrCols; jj++) {
        const float mu = vmu[id * kU + sn + jj];
        const float vc = vsc[id * kU + sn + jj];
        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < otile_t::kFragThrRows; ii++) {
          if (rcp[ii] != AccumType(0)) {
            const auto loc = ii * otile_t::kFragThrCols + jj;
            // Without a sink, the expression as it always was (the same
            // bits for every caller before sinks were taken here).
            if (has_sinks) {
              of[loc] = of[loc] * (AccumType)vc +
                        (AccumType)mu * keys_share[ii];
            } else {
              of[loc] = of[loc] * (AccumType)vc + (AccumType)mu;
            }
          }
        }
      }
    }
  }

  // Store results
  O += tm * int(params->O_strides[2]);

  if (!align_Q && is_last_q) {
    if (lim_rows_q <= 0)
      return;

    Otile.store_rows(O, int(params->O_strides[2]), lim_rows_q);
  } else {
    Otile.store(O, int(params->O_strides[2]));
  }
}

// vpipe: MLX #3842, and #4487's generalization to WN = 4 (head dim 512),
// from upstream 388f227f without its save_lse output: function constant
// 303 and buffer 8 are has_spans / qb_off here, and export_ml already
// carries the row statistics.
///////////////////////////////////////////////////////////////////////////////
// Head-dim split attention kernel
///////////////////////////////////////////////////////////////////////////////

// Split wide heads across WN simdgroups to reduce the accumulator working set.
// Each group owns one D slice. The groups exchange their partial Q@K.T scores
// through threadgroup memory and reduce them in the same order. Each group then
// runs softmax and accumulates its P@V slice.

// clang-format off
template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int WM,
    int WN,
    typename MaskType = float,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attention_nax_dsplit(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant AttnParams* params [[buffer(4)]],
    const constant AttnMaskParams* mask_params [[buffer(5), function_constant(has_mask)]],
    const device MaskType* mask [[buffer(6), function_constant(has_mask)]],
    const device T* sinks [[buffer(7), function_constant(has_sinks)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on

  // Pacifying compiler
  (void)lid;

  // Move to correct block
  ulong3 tidl{tid.x, tid.y, tid.z};

  Q += tidl.z * params->Q_strides[0] + // Batch
      tidl.y * params->Q_strides[1] + // Head
      tidl.x * BQ * params->Q_strides[2]; // Sequence

  ulong kv_head_idx = int(tid.y) / params->gqa_factor;
  K += tidl.z * params->K_strides[0] + // Batch
      kv_head_idx * params->K_strides[1]; // Head

  V += tidl.z * params->V_strides[0] + // Batch
      kv_head_idx * params->V_strides[1]; // Head

  O += tidl.z * params->O_strides[0] + // Batch
      tidl.y * params->O_strides[1] + // Head
      tidl.x * BQ * params->O_strides[2]; // Sequence

  if (has_mask) {
    mask += tidl.z * mask_params->M_strides[0] + // Batch
        tidl.y * mask_params->M_strides[1]; // Head
  }

  const metal::uniform<float> scale2 =
      make_uniform(params->scale) * make_uniform(1.44269504089f);

  // Prepare MMA tiles
  constexpr short kU = 16;

  // The WM simdgroups along the first warp dimension split the Q sequence;
  // the WN simdgroups along the second split the head dim.
  static_assert(
      WN == 2 || WN == 4, "The head-dim split kernel needs WN == 2 or WN == 4");
  constexpr int kNWarps = WM;
  static_assert(
      BQ >= (kNWarps * kU) && BQ % (kNWarps * kU) == 0,
      "Each simdgroup must host atleast 1 simdgroup matrix along Q sequence.");

  // Q seq frags per warp
  constexpr int TQ = BQ / (kNWarps * kU);
  // HeadDim frags over the full head dim
  constexpr int TD = BD / kU;
  // KV seq frags per warp
  constexpr short TK = BK / kU;

  static_assert(TQ == 1, "Check TQ");
  static_assert(TD % WN == 0, "The head dim must split evenly across WN");

  // HeadDim frags / columns owned by each of the WN simdgroups of a row group
  constexpr int TDh = TD / WN;
  constexpr int BDh = BD / WN;

  static_assert(TDh % 2 == 0, "P@V accumulates output fragments in pairs");
  static_assert(TK % 2 == 0, "S fragments are computed pair by pair");

  const short row_group = simd_group_id / WN;
  const short d_group = simd_group_id % WN;

  using otile_t = NAXTile<AccumType, TQ, TDh>;
  otile_t Otile;
  Otile.clear();

  const short tm = kU * TQ * row_group;
  Q += tm * int(params->Q_strides[2]) + d_group * BDh;
  K += d_group * BDh;
  V += d_group * BDh;
  O += tm * int(params->O_strides[2]) + d_group * BDh;

  constexpr short kRowsPT = otile_t::kRowsPerThread;

  metal::vec<AccumType, kRowsPT> max_score;
  metal::vec<AccumType, kRowsPT> sum_score{0};

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<AccumType>::finite_min;
  }

  if (has_sinks) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = M_LOG2E_F * static_cast<AccumType>(sinks[tidl.y]);
      sum_score[i] = 1;
    }
  }

  int kb_lim = params->NK;
  int kb_min_causal = params->NK;

  if (do_causal) {
    int q_max = (tid.x + 1) * BQ + params->qL_off;
    kb_lim = (q_max + BK - 1) / BK;
    kb_lim = min(params->NK, kb_lim);

    int q_min = tid.x * BQ + params->qL_off;
    q_min = max(0, q_min);
    kb_min_causal = (q_min / BK);
  }

  const bool is_last_q = int(tid.x) == (params->NQ_aligned);
  const short lim_rows_q = params->qL_rem - tm;
  const short lim_rows_k = params->kL_rem;

  using stile_t = NAXTile<AccumType, TQ, TK>;
  constexpr short kEPF = stile_t::NAXFrag_t::kElemsPerFrag;

  // One slot per (row group, D group) in per-lane-linear layout. All groups
  // share the fragment-to-lane mapping.
  threadgroup AccumType s_xchg[WM][WN][TK * kEPF * 32];

  // Keep the simdgroup's Q slice resident in registers for the whole KV
  // loop: TDh fragments of T are cheap next to the accumulators.
  NAXTile<T, 1, 1> Qtiles[TDh];
  STEEL_PRAGMA_UNROLL
  for (short id = 0; id < TDh; id++) {
    const int Q_load_off = id * kU;
    if (!align_Q && is_last_q) {
      Qtiles[id].load_rows(
          Q + Q_load_off, int(params->Q_strides[2]), lim_rows_q);
    } else {
      Qtiles[id].load(Q + Q_load_off, int(params->Q_strides[2]));
    }
  }

  const short2 simd_coord = otile_t::NAXFrag_t::get_coord();
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;

  // Loop over KV seq length
  for (int kb = 0; kb < kb_lim; kb++) {
    const int is_last_k = (kb == (params->NK_aligned));

    stile_t Stile;
    Stile.clear();

    // S = Q @ K.T for this D slice.
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ik += 2) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TDh; id++) {
        NAXTile<T, 2, 1> Ktile;
        const int K_load_off = ik * kU * int(params->K_strides[2]) + id * kU;

        if (!align_K && is_last_k) {
          Ktile.load_rows(
              K + K_load_off, int(params->K_strides[2]), lim_rows_k - ik * kU);
        } else {
          Ktile.load(K + K_load_off, int(params->K_strides[2]));
        }

        stile_t::NAXFrag_t::mma(
            Stile.frag_at(0, ik),
            Stile.frag_at(0, ik + 1),
            Qtiles[id].frag_at(0, 0),
            metal::false_type{},
            Ktile.frag_at(0, 0),
            Ktile.frag_at(1, 0),
            metal::true_type{});
      }
    }

    // Exchange all partial scores and reduce them in a fixed order.
    threadgroup AccumType* slot = s_xchg[row_group][d_group];
    const short base = short(simd_lane_id) * (TK * kEPF);
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ik++) {
      thread auto& s = Stile.frag_at(0, ik);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kEPF; i++) {
        slot[base + ik * kEPF + i] = s[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ik++) {
      thread auto& s = Stile.frag_at(0, ik);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kEPF; i++) {
        s[i] = s_xchg[row_group][0][base + ik * kEPF + i];
      }
      STEEL_PRAGMA_UNROLL
      for (short peer_group = 1; peer_group < WN; peer_group++) {
        const threadgroup AccumType* peer = s_xchg[row_group][peer_group];
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kEPF; i++) {
          s[i] += peer[base + ik * kEPF + i];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Scale S
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= float(scale2);
    }

    // Mask out length sequence
    if (!align_K && is_last_k) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        const short col_pos = ik * kU + sn;
        thread auto& fg = Stile.frag_at(0, ik);

        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
            const auto loc = ii * stile_t::kFragThrCols + jj;
            fg[loc] = ((col_pos + jj) < params->kL_rem) ? fg[loc] : neg_inf;
          }
        }
      }
    }

    // Mask out if causal
    if (do_causal && kb >= kb_min_causal) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      const int base_row = tid.x * BQ + params->qL_off + tm;
      const int base_col = kb * BK;

      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        thread auto& fg = Stile.frag_at(0, ik);

        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
            const auto r = base_row + ii * stile_t::kFragRowsJump + sm;
            const auto c = base_col + ik * kU + jj + sn;
            const auto loc = ii * stile_t::kFragThrCols + jj;
            fg[loc] = (r < c) ? neg_inf : fg[loc];
          }
        }
      }
    }

    // Other masking as needed
    if (has_mask) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      const int base_row = tid.x * BQ + tm;
      const int base_col = kb * BK;

      constexpr bool is_bool = is_same_v<MaskType, bool>;
      using melem_t = typename metal::conditional_t<is_bool, bool, AccumType>;
      using mtile_t = NAXTile<melem_t, TQ, TK>;
      using mfrag_t = typename mtile_t::frag_type;

      if (base_row + kU <= params->qL && base_col + BK <= params->kL) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const int row_pos = base_row;
          const int col_pos = base_col + ik * kU;

          mfrag_t mfrag;
          mtile_t::NAXFrag_t::load(
              mfrag,
              mask,
              int64_t(mask_params->M_strides[2]),
              Int<1>{},
              row_pos,
              col_pos);

          thread auto& fg = Stile.frag_at(0, ik);

          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < mtile_t::kElemsPerFrag; jj++) {
            if constexpr (is_bool) {
              fg[jj] = mfrag[jj] ? fg[jj] : neg_inf;
            } else {
              fg[jj] += M_LOG2E_F * AccumType(mfrag[jj]);
            }
          }
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const int row_pos = base_row;
          const int col_pos = base_col + ik * kU;

          mfrag_t mfrag;
          mtile_t::NAXFrag_t::load_safe(
              mfrag,
              mask,
              int64_t(mask_params->M_strides[2]),
              Int<1>{},
              params->qL,
              params->kL,
              row_pos,
              col_pos);

          thread auto& fg = Stile.frag_at(0, ik);

          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < mtile_t::kElemsPerFrag; jj++) {
            if constexpr (is_bool) {
              fg[jj] = mfrag[jj] ? fg[jj] : neg_inf;
            } else {
              fg[jj] += M_LOG2E_F * AccumType(mfrag[jj]);
            }
          }
        }
      }
    }

    // Do softmax in each D group; the row statistics are cheap.
    metal::vec<AccumType, kRowsPT> new_max;
    metal::vec<AccumType, kRowsPT> factor;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }

    Stile.template row_reduce<MaxOp>(new_max);
    Stile.template row_bin_op<ExpSubOp>(new_max);

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i];
    }

    Stile.template row_reduce<SumOp>(sum_score);

    Otile.template row_bin_op<MulOp>(factor);

    simdgroup_barrier(mem_flags::mem_none);

    // O = P @ V for this D slice.
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TDh; id += 2) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        NAXTile<T, 1, 2> Vtile;

        const int V_load_off = ik * kU * int(params->V_strides[2]) + id * kU;

        if (!align_K && is_last_k) {
          Vtile.load_rows(
              V + V_load_off, int(params->V_strides[2]), lim_rows_k - ik * kU);
        } else {
          Vtile.load(V + V_load_off, int(params->V_strides[2]));
        }

        otile_t::NAXFrag_t::mma(
            Otile.frag_at(0, id),
            Otile.frag_at(0, id + 1),
            Stile.frag_at(0, ik),
            metal::false_type{},
            Vtile.frag_at(0, 0),
            Vtile.frag_at(0, 1),
            metal::false_type{});
      }
    }

    // Next block
    K += BK * int(params->K_strides[2]);
    V += BK * int(params->V_strides[2]);
  }

  // Normalize output
  threadgroup_barrier(mem_flags::mem_none);

  metal::vec<AccumType, kRowsPT> rcp;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    rcp[i] = 1.f / sum_score[i];
  }

  Otile.template row_bin_op<MulOp>(rcp);

  if (!align_Q && is_last_q) {
    if (lim_rows_q <= 0)
      return;
    Otile.store_rows(O, int(params->O_strides[2]), lim_rows_q);
  } else {
    Otile.store(O, int(params->O_strides[2]));
  }
}

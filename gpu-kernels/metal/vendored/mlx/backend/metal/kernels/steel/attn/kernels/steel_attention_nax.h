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

template <typename T>
struct TransformScale {
  T scale;
  METAL_FUNC TransformScale(T scale_) : scale(scale_) {}

  METAL_FUNC T apply(T x) const {
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
    vis_base = qb_off[tid.x];
    nvis = qb_off[tid.x + 1] - vis_base;
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

    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik += 2) {
        STEEL_PRAGMA_UNROLL
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
    Otile.template row_bin_op<MulOp>(factor);

    simdgroup_barrier(mem_flags::mem_none);

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

    // Prepare for next iteration. The sparse path steps at the TOP
    // instead, because the gap is a property of the block it is about to
    // read and not of the one it has just finished.
    if (!has_spans) {
      K += BK * int(params->K_strides[2]);
      V += BK * int(params->V_strides[2]);
    }
  }

  // Normalize output

  threadgroup_barrier(mem_flags::mem_none);

  metal::vec<AccumType, kRowsPT> rcp;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    rcp[i] = 1.f / sum_score[i];
  }

  Otile.template row_bin_op<MulOp>(rcp);

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

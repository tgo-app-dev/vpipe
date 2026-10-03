// attn_bwd.metal -- the backward of the steel flash attention, head_dim
// 128, in the forward's own BAND form.
//
// A block-causal DiT runs its attention as one dispatch per SEGMENT: a
// band of query rows against keys [0, kL), causal for a text run and
// dense for an image block. The backward mirrors that decomposition
// exactly, dispatch for dispatch, so no mask buffer exists here either.
//
// Two kernels, FlashAttention-2's split, and NO ATOMICS -- the toolchain
// has no float atomic add, and a fixed order is what makes a training run
// reproducible bit for bit:
//
//   attn_bwd_dq   one threadgroup per (query block, head). Loops over the
//                 segment's key blocks, recomputing P from the forward's
//                 exported row statistics, and writes dQ for its rows --
//                 complete, since every query row belongs to exactly one
//                 segment.
//   attn_bwd_dkv  one threadgroup per (key block, head). Loops over the
//                 segment's query rows and ADDS its dK and dV into f32
//                 accumulators. Segments are dispatched in order on one
//                 stream, so the read-modify-write across them is serial.
//
// THE GEMM FORM, attn_bwd_pds: the same pass as attn_bwd_dkv, but instead
// of multiplying P^T and dS^T into dV and dK itself it STORES them (bf16,
// dS already scaled), and the three products that remain -- dV = P^T dO,
// dK = dS^T Q, dQ = dS K -- run as ordinary batched GEMMs on whatever the
// box multiplies fastest (matrix cores on an M5). The fused pair spends
// seven matmul units per segment, two of them recomputing S and dP a
// second time; this spends five, and three of those at GEMM rate. The
// price is the two [kL][qL] planes per head, which the host bounds by
// running the heads in groups. FlashAttention-2 rounds dS to the storage
// type before its dQ and dK products too.
//
// Inputs per row (train_ops.metal): LSE2 = ml_max + log2(ml_sum) from the
// forward's export_ml, in the kernel's scaled log-2 domain, and
// Dsum = rowsum(dO * O). With S2 = scale * log2(e) * Q K^T:
//   P  = exp2(S2 - LSE2)
//   dP = dO V^T,  dS = P * (dP - Dsum)
//   dQ = scale * dS K,  dK = scale * dS^T Q,  dV = P^T dO
// Everything is accumulated in f32 simdgroup matrices.
//
// The attn/* steel headers are self-contained; do NOT also include the
// gemm/* ones in this translation unit.

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

#ifndef METAL_FUNC
#define METAL_FUNC inline
#endif

using namespace metal;

#ifndef M_LOG2E_F
#define M_LOG2E_F 1.44269504088896340736f
#endif

template <typename U>
struct Limits {
  static constexpr constant U max = metal::numeric_limits<U>::max();
  static constexpr constant U min = metal::numeric_limits<U>::min();
  static constexpr constant U finite_max = metal::numeric_limits<U>::max();
  static constexpr constant U finite_min = metal::numeric_limits<U>::min();
};
template <>
struct Limits<float> {
  static constexpr constant float max =
      metal::numeric_limits<float>::infinity();
  static constexpr constant float min =
      -metal::numeric_limits<float>::infinity();
  static constexpr constant float finite_max =
      metal::numeric_limits<float>::max();
  static constexpr constant float finite_min =
      -metal::numeric_limits<float>::max();
};

#include "mlx/backend/metal/kernels/steel/attn/attn.h"

using namespace mlx::steel;

// One segment's geometry. Q, dO and dQ are bound at the segment's first
// query row; K and V at key 0. LSE2 and Dsum are [H][T] over the whole
// joint sequence, indexed by q_row0 + the segment-local row.
struct AttnBwdParams {
  int   H;
  int   qL;          // the segment's query rows
  int   kL;          // keys [0, kL)
  float scale;       // 1 / sqrt(D), NOT folded with log2(e)
  int   qL_off;      // causal: the global position of query row 0
  int   causal;      // text runs; image blocks are dense
  int   T;           // row count of LSE2 / Dsum per head
  int   q_row0;      // global row of the segment's query row 0
  long  q_hstride;   // elements between heads of Q / dO / dQ
  long  k_hstride;   // elements between heads of K / V
  long  acc_hstride; // elements between heads of the f32 dK / dV
  // attn_bwd_pds only: the head group starts at head h0, and P^T / dS^T
  // are [kL][ldp] planes p_hstride apart, one per head of the group.
  int   h0;
  int   ldp;
  long  p_hstride;
};

// ---- dQ ---------------------------------------------------------------------

template <typename T, int BQ, int BK, int BD, int WM>
[[kernel, max_total_threads_per_threadgroup(WM * 32)]] void attn_bwd_dq(
    const device T* Q      [[buffer(0)]],
    const device T* K      [[buffer(1)]],
    const device T* V      [[buffer(2)]],
    const device T* dO     [[buffer(3)]],
    device T*       dQ     [[buffer(4)]],
    const device float* LSE2 [[buffer(5)]],
    const device float* Dsum [[buffer(6)]],
    const constant AttnBwdParams* p [[buffer(7)]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]])
{
  const int h = (int)tid.y;
  const int qb = (int)tid.x;
  const long qoff = (long)h * p->q_hstride + (long)qb * BQ * BD;
  Q += qoff;
  dO += qoff;
  dQ += qoff;
  K += (long)h * p->k_hstride;
  V += (long)h * p->k_hstride;

  constexpr short pad = 16 / sizeof(T);
  constexpr short LDQ = BD + pad;
  constexpr short LDK = BK + pad;

  threadgroup T Qs[BQ * LDQ];
  threadgroup T dOs[BQ * LDQ];
  threadgroup T Ks[BD * LDK];
  threadgroup T Vs[BD * LDK];

  constexpr short tgp = WM * 32;
  using QLoader = BlockLoaderT<T, BQ, BD, LDQ, 1, 1, tgp>;
  // K and V are staged TRANSPOSED ([BD][BK]), the forward's K layout.
  using KLoader = BlockLoaderT<T, BK, BD, 1, LDK, 0, tgp>;

  QLoader lq(Q, BD, Qs, simd_group_id, simd_lane_id);
  QLoader ldo(dO, BD, dOs, simd_group_id, simd_lane_id);
  KLoader lk(K, BD, Ks, simd_group_id, simd_lane_id);
  KLoader lv(V, BD, Vs, simd_group_id, simd_lane_id);

  constexpr short kFrag = 8;
  using Frag = BaseMMAFrag<float, kFrag, kFrag>;
  constexpr int TK = BK / kFrag;
  constexpr int TD = BD / kFrag;

  MMATile<float, 1, 1, Frag> Qt;
  MMATile<float, 1, TK, Frag> Kt;
  MMATile<float, 1, TK, Frag> St;
  MMATile<float, 1, TK, Frag> dPt;
  MMATile<float, 1, 1, Frag> Bt;
  MMATile<float, 1, TD, Frag> dQt;
  dQt.clear();

  const short2 sc = Frag::get_coord(simd_lane_id);
  const short sm = sc.y;
  const short sn = sc.x;
  const short tm = kFrag * simd_group_id;
  const short Qs_off = (tm + sm) * LDQ + sn;
  const short Ks_off = sm * LDK + sn;

  const int q_rem = p->qL - qb * BQ;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (q_rem < BQ) {
    lq.load_safe(short2(BD, q_rem));
    ldo.load_safe(short2(BD, q_rem));
  } else {
    lq.load_unsafe();
    ldo.load_unsafe();
  }

  const int row = qb * BQ + tm + sm;           // segment-local query row
  const bool row_ok = row < p->qL;
  const long rix = (long)h * p->T + p->q_row0 + row;
  const float lse = row_ok ? LSE2[rix] : 0.0f;
  const float dsum = row_ok ? Dsum[rix] : 0.0f;
  const float scale2 = p->scale * M_LOG2E_F;

  const int NK = (p->kL + BK - 1) / BK;
  int kb_lim = NK;
  if (p->causal != 0) {
    const int q_max = (qb + 1) * BQ + p->qL_off;
    kb_lim = min(NK, (q_max + BK - 1) / BK);
  }

  for (int kb = 0; kb < kb_lim; ++kb) {
    const int k_rem = p->kL - kb * BK;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k_rem < BK) {
      lk.load_safe(short2(BD, k_rem));
      lv.load_safe(short2(BD, k_rem));
    } else {
      lk.load_unsafe();
      lv.load_unsafe();
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    St.clear();
    dPt.clear();
    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      simdgroup_barrier(mem_flags::mem_none);
      Qt.template load<T, 1, 1, LDQ, 1>(&Qs[Qs_off + dd * kFrag]);
      Kt.template load<T, 1, 1, LDK, 1>(&Ks[Ks_off + dd * kFrag * LDK]);
      simdgroup_barrier(mem_flags::mem_none);
      tile_matmad(St, Qt, Kt, St);
      Qt.template load<T, 1, 1, LDQ, 1>(&dOs[Qs_off + dd * kFrag]);
      Kt.template load<T, 1, 1, LDK, 1>(&Vs[Ks_off + dd * kFrag * LDK]);
      simdgroup_barrier(mem_flags::mem_none);
      tile_matmad(dPt, Qt, Kt, dPt);
    }

    // P and dS, in place of S.
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < TK; j++) {
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < 2; e++) {
        const int col = kb * BK + j * kFrag + sn + e;
        const bool ok = row_ok && col < p->kL &&
                        (p->causal == 0 || row + p->qL_off >= col);
        const float pv =
            ok ? fast::exp2(St.frag_at(0, j)[e] * scale2 - lse) : 0.0f;
        St.frag_at(0, j)[e] = pv * (dPt.frag_at(0, j)[e] - dsum);
      }
    }

    // dQ += dS K. K sits transposed in Ks, so a fragment of K (rows = keys,
    // cols = d) strides by LDK along its columns.
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TD; id++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        simdgroup_barrier(mem_flags::mem_none);
        Bt.template load<T, 1, 1, 1, LDK>(
            &Ks[(id * kFrag + sn) * LDK + ik * kFrag + sm]);
        simdgroup_barrier(mem_flags::mem_none);
        Frag::mma(dQt.frag_at(0, id), St.frag_at(0, ik), Bt.frag_at(0, 0),
                  dQt.frag_at(0, id));
      }
    }

    lk.next();
    lv.next();
  }

  STEEL_PRAGMA_UNROLL
  for (short ii = 0; ii < decltype(dQt)::kElemsPerTile; ii++) {
    dQt.elems()[ii] *= p->scale;
  }
  threadgroup_barrier(mem_flags::mem_none);
  device T* o = dQ + (tm + sm) * BD + sn;
  if (q_rem < BQ) {
    const short2 dims = short2(BD - sn, q_rem - (tm + sm));
    if (dims.x > 0 && dims.y > 0) {
      dQt.template store_safe<T, 1, 1>(o, BD, dims);
    }
  } else {
    dQt.template store<T, 1, 1>(o, BD);
  }
}

// ---- dK, dV -----------------------------------------------------------------

template <typename T, int BKR, int BQC, int BD, int WM>
[[kernel, max_total_threads_per_threadgroup(WM * 32)]] void attn_bwd_dkv(
    const device T* Q      [[buffer(0)]],
    const device T* K      [[buffer(1)]],
    const device T* V      [[buffer(2)]],
    const device T* dO     [[buffer(3)]],
    device float*   dK     [[buffer(4)]],
    device float*   dV     [[buffer(5)]],
    const device float* LSE2 [[buffer(6)]],
    const device float* Dsum [[buffer(7)]],
    const constant AttnBwdParams* p [[buffer(8)]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
  const int h = (int)tgid.y;
  const int kbk = (int)tgid.x;
  const uint lid = tid.y * 32 + tid.x;
  const long koff = (long)h * p->k_hstride + (long)kbk * BKR * BD;
  K += koff;
  V += koff;
  Q += (long)h * p->q_hstride;
  dO += (long)h * p->q_hstride;

  constexpr short pad = 16 / sizeof(T);
  constexpr short LDK = BD + pad;
  constexpr short LDQ = BD + pad;

  threadgroup T Ks[BKR * LDK];
  threadgroup T Vs[BKR * LDK];
  threadgroup T Qs[BQC * LDQ];
  threadgroup T dOs[BQC * LDQ];
  threadgroup float lse_s[BQC];
  threadgroup float d_s[BQC];

  constexpr short tgp = WM * 32;
  using KLoader = BlockLoaderT<T, BKR, BD, LDK, 1, 0, tgp>;
  using QLoader = BlockLoaderT<T, BQC, BD, LDQ, 1, 0, tgp>;

  KLoader lk(K, BD, Ks, simd_group_id, simd_lane_id);
  KLoader lv(V, BD, Vs, simd_group_id, simd_lane_id);
  QLoader lq(Q, BD, Qs, simd_group_id, simd_lane_id);
  QLoader ldo(dO, BD, dOs, simd_group_id, simd_lane_id);

  constexpr short kFrag = 8;
  using Frag = BaseMMAFrag<float, kFrag, kFrag>;
  constexpr int TQ = BQC / kFrag;
  constexpr int TD = BD / kFrag;

  MMATile<float, 1, 1, Frag> At;
  MMATile<float, 1, TQ, Frag> Bt;
  MMATile<float, 1, TQ, Frag> St;     // S^T, then P^T
  MMATile<float, 1, TQ, Frag> dPt;    // dP^T, then dS^T
  MMATile<float, 1, 1, Frag> Ft;
  MMATile<float, 1, TD, Frag> dKt;
  MMATile<float, 1, TD, Frag> dVt;
  dKt.clear();
  dVt.clear();

  const short2 sc = Frag::get_coord(simd_lane_id);
  const short sm = sc.y;
  const short sn = sc.x;
  const short tm = kFrag * simd_group_id;

  const int k_rem = p->kL - kbk * BKR;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (k_rem < BKR) {
    lk.load_safe(short2(BD, k_rem));
    lv.load_safe(short2(BD, k_rem));
  } else {
    lk.load_unsafe();
    lv.load_unsafe();
  }

  const int krow = kbk * BKR + tm + sm;      // key index
  const bool krow_ok = krow < p->kL;
  const float scale2 = p->scale * M_LOG2E_F;

  const int NQ = (p->qL + BQC - 1) / BQC;
  int qb0 = 0;
  if (p->causal != 0) {
    // The first query row that can see this block's first key.
    const int first = max(0, kbk * BKR - p->qL_off);
    qb0 = first / BQC;
  }
  if (qb0 > 0) {
    lq.jump(qb0);
    ldo.jump(qb0);
  }

  for (int qb = qb0; qb < NQ; ++qb) {
    const int q_rem = p->qL - qb * BQC;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (q_rem < BQC) {
      lq.load_safe(short2(BD, q_rem));
      ldo.load_safe(short2(BD, q_rem));
    } else {
      lq.load_unsafe();
      ldo.load_unsafe();
    }
    if (lid < (uint)BQC) {
      const int r = qb * BQC + (int)lid;
      const long rix = (long)h * p->T + p->q_row0 + r;
      lse_s[lid] = r < p->qL ? LSE2[rix] : 0.0f;
      d_s[lid] = r < p->qL ? Dsum[rix] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // S^T = K Q^T and dP^T = V dO^T, rows = this warp's keys.
    St.clear();
    dPt.clear();
    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      simdgroup_barrier(mem_flags::mem_none);
      At.template load<T, 1, 1, LDK, 1>(&Ks[(tm + sm) * LDK + dd * kFrag + sn]);
      Bt.template load<T, 1, 1, 1, LDQ>(&Qs[sn * LDQ + dd * kFrag + sm]);
      simdgroup_barrier(mem_flags::mem_none);
      tile_matmad(St, At, Bt, St);
      At.template load<T, 1, 1, LDK, 1>(&Vs[(tm + sm) * LDK + dd * kFrag + sn]);
      Bt.template load<T, 1, 1, 1, LDQ>(&dOs[sn * LDQ + dd * kFrag + sm]);
      simdgroup_barrier(mem_flags::mem_none);
      tile_matmad(dPt, At, Bt, dPt);
    }

    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < TQ; j++) {
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < 2; e++) {
        const int qc = j * kFrag + sn + e;         // column in the block
        const int q = qb * BQC + qc;               // segment-local row
        const bool ok = krow_ok && q < p->qL &&
                        (p->causal == 0 || q + p->qL_off >= krow);
        const float pv =
            ok ? fast::exp2(St.frag_at(0, j)[e] * scale2 - lse_s[qc]) : 0.0f;
        St.frag_at(0, j)[e] = pv;
        dPt.frag_at(0, j)[e] = pv * (dPt.frag_at(0, j)[e] - d_s[qc]);
      }
    }

    // dV += P^T dO, dK += dS^T Q.
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TD; id++) {
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        simdgroup_barrier(mem_flags::mem_none);
        Ft.template load<T, 1, 1, LDQ, 1>(
            &dOs[(iq * kFrag + sm) * LDQ + id * kFrag + sn]);
        simdgroup_barrier(mem_flags::mem_none);
        Frag::mma(dVt.frag_at(0, id), St.frag_at(0, iq), Ft.frag_at(0, 0),
                  dVt.frag_at(0, id));
        Ft.template load<T, 1, 1, LDQ, 1>(
            &Qs[(iq * kFrag + sm) * LDQ + id * kFrag + sn]);
        simdgroup_barrier(mem_flags::mem_none);
        Frag::mma(dKt.frag_at(0, id), dPt.frag_at(0, iq), Ft.frag_at(0, 0),
                  dKt.frag_at(0, id));
      }
    }

    lq.next();
    ldo.next();
  }

  // Accumulate into the f32 planes. Each (key, column) belongs to this
  // threadgroup alone within a dispatch; across segments the dispatches
  // are ordered, so the read-modify-write is safe without atomics.
  if (krow_ok) {
    device float* dk = dK + (long)h * p->acc_hstride + (long)krow * BD;
    device float* dv = dV + (long)h * p->acc_hstride + (long)krow * BD;
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TD; id++) {
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < 2; e++) {
        const int c = id * kFrag + sn + e;
        dk[c] += dKt.frag_at(0, id)[e] * p->scale;
        dv[c] += dVt.frag_at(0, id)[e];
      }
    }
  }
}

// ---- P^T, dS^T (the GEMM form) ---------------------------------------------

// One threadgroup per (key block, head of the group); every query row of
// the segment, so the planes are whole. A causal segment starts at query
// block 0 too: its masked entries must be written as ZERO, since the GEMMs
// that follow read the whole plane.
template <typename T, int BKR, int BQC, int BD, int WM>
[[kernel, max_total_threads_per_threadgroup(WM * 32)]] void attn_bwd_pds(
    const device T* Q      [[buffer(0)]],
    const device T* K      [[buffer(1)]],
    const device T* V      [[buffer(2)]],
    const device T* dO     [[buffer(3)]],
    device T*       Pt     [[buffer(4)]],
    device T*       dSt    [[buffer(5)]],
    const device float* LSE2 [[buffer(6)]],
    const device float* Dsum [[buffer(7)]],
    const constant AttnBwdParams* p [[buffer(8)]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
  const int hl = (int)tgid.y;                  // head within the group
  const int h = p->h0 + hl;
  const int kbk = (int)tgid.x;
  const uint lid = tid.y * 32 + tid.x;
  const long koff = (long)h * p->k_hstride + (long)kbk * BKR * BD;
  K += koff;
  V += koff;
  Q += (long)h * p->q_hstride;
  dO += (long)h * p->q_hstride;
  Pt += (long)hl * p->p_hstride;
  dSt += (long)hl * p->p_hstride;

  constexpr short pad = 16 / sizeof(T);
  constexpr short LDK = BD + pad;
  constexpr short LDQ = BD + pad;

  threadgroup T Ks[BKR * LDK];
  threadgroup T Vs[BKR * LDK];
  threadgroup T Qs[BQC * LDQ];
  threadgroup T dOs[BQC * LDQ];
  threadgroup float lse_s[BQC];
  threadgroup float d_s[BQC];

  constexpr short tgp = WM * 32;
  using KLoader = BlockLoaderT<T, BKR, BD, LDK, 1, 0, tgp>;
  using QLoader = BlockLoaderT<T, BQC, BD, LDQ, 1, 0, tgp>;

  KLoader lk(K, BD, Ks, simd_group_id, simd_lane_id);
  KLoader lv(V, BD, Vs, simd_group_id, simd_lane_id);
  QLoader lq(Q, BD, Qs, simd_group_id, simd_lane_id);
  QLoader ldo(dO, BD, dOs, simd_group_id, simd_lane_id);

  constexpr short kFrag = 8;
  using Frag = BaseMMAFrag<float, kFrag, kFrag>;
  constexpr int TQ = BQC / kFrag;
  constexpr int TD = BD / kFrag;

  MMATile<float, 1, 1, Frag> At;
  MMATile<float, 1, TQ, Frag> Bt;
  MMATile<float, 1, TQ, Frag> St;     // S^T, then P^T
  MMATile<float, 1, TQ, Frag> dPt;    // dP^T, then dS^T

  const short2 sc = Frag::get_coord(simd_lane_id);
  const short sm = sc.y;
  const short sn = sc.x;
  const short tm = kFrag * simd_group_id;

  const int k_rem = p->kL - kbk * BKR;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (k_rem < BKR) {
    lk.load_safe(short2(BD, k_rem));
    lv.load_safe(short2(BD, k_rem));
  } else {
    lk.load_unsafe();
    lv.load_unsafe();
  }

  const int krow = kbk * BKR + tm + sm;      // key index
  const bool krow_ok = krow < p->kL;
  const float scale2 = p->scale * M_LOG2E_F;
  device T* prow = Pt + (long)krow * p->ldp;
  device T* drow = dSt + (long)krow * p->ldp;

  const int NQ = (p->qL + BQC - 1) / BQC;
  for (int qb = 0; qb < NQ; ++qb) {
    const int q_rem = p->qL - qb * BQC;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (q_rem < BQC) {
      lq.load_safe(short2(BD, q_rem));
      ldo.load_safe(short2(BD, q_rem));
    } else {
      lq.load_unsafe();
      ldo.load_unsafe();
    }
    if (lid < (uint)BQC) {
      const int r = qb * BQC + (int)lid;
      const long rix = (long)h * p->T + p->q_row0 + r;
      lse_s[lid] = r < p->qL ? LSE2[rix] : 0.0f;
      d_s[lid] = r < p->qL ? Dsum[rix] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    St.clear();
    dPt.clear();
    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      simdgroup_barrier(mem_flags::mem_none);
      At.template load<T, 1, 1, LDK, 1>(&Ks[(tm + sm) * LDK + dd * kFrag + sn]);
      Bt.template load<T, 1, 1, 1, LDQ>(&Qs[sn * LDQ + dd * kFrag + sm]);
      simdgroup_barrier(mem_flags::mem_none);
      tile_matmad(St, At, Bt, St);
      At.template load<T, 1, 1, LDK, 1>(&Vs[(tm + sm) * LDK + dd * kFrag + sn]);
      Bt.template load<T, 1, 1, 1, LDQ>(&dOs[sn * LDQ + dd * kFrag + sm]);
      simdgroup_barrier(mem_flags::mem_none);
      tile_matmad(dPt, At, Bt, dPt);
    }

    // P^T and scale * dS^T, straight to the planes: each lane holds two
    // adjacent query columns of its key row.
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < TQ; j++) {
      const int qc = j * kFrag + sn;               // column in the block
      const int q = qb * BQC + qc;                 // segment-local row
      vec<T, 2> pv2, ds2;
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < 2; e++) {
        const bool ok = krow_ok && q + e < p->qL &&
                        (p->causal == 0 || q + e + p->qL_off >= krow);
        const float pv = ok ? fast::exp2(St.frag_at(0, j)[e] * scale2 -
                                         lse_s[qc + e])
                            : 0.0f;
        pv2[e] = T(pv);
        ds2[e] = T(pv * (dPt.frag_at(0, j)[e] - d_s[qc + e]) * p->scale);
      }
      if (krow_ok) {
        if (q + 1 < p->qL && (p->ldp % 2) == 0) {
          *reinterpret_cast<device vec<T, 2>*>(prow + q) = pv2;
          *reinterpret_cast<device vec<T, 2>*>(drow + q) = ds2;
        } else {
          if (q < p->qL) { prow[q] = pv2[0]; drow[q] = ds2[0]; }
          if (q + 1 < p->qL) { prow[q + 1] = pv2[1]; drow[q + 1] = ds2[1]; }
        }
      }
    }

    lq.next();
    ldo.next();
  }
}

#define VPIPE_ATTN_BWD(TNAME, T)                                             \
  template [[host_name("attn_bwd_dq_bd128_" TNAME)]] [[kernel]]              \
  decltype(attn_bwd_dq<T, 32, 16, 128, 4>) attn_bwd_dq<T, 32, 16, 128, 4>;   \
  template [[host_name("attn_bwd_dkv_bd128_" TNAME)]] [[kernel]]             \
  decltype(attn_bwd_dkv<T, 32, 16, 128, 4>) attn_bwd_dkv<T, 32, 16, 128, 4>; \
  template [[host_name("attn_bwd_pds_bd128_" TNAME)]] [[kernel]]             \
  decltype(attn_bwd_pds<T, 32, 16, 128, 4>) attn_bwd_pds<T, 32, 16, 128, 4>;

VPIPE_ATTN_BWD("bf16", bfloat)
VPIPE_ATTN_BWD("f16", half)

// attn_steel.metal -- MLX's steel flash-attention kernel, vendored and
// instantiated for vpipe's metal-compute prefill path (T=half,
// head_dim=128, GQA). This is the register-resident MMA flash kernel MLX
// itself ships (steel/attn): QK^T + PV as simdgroup_matrix matmuls, the
// online softmax done entirely on the register MMATile via the
// fragment-aware row_reduce / row_bin_op (no threadgroup round-trip),
// with double-staged Q/K/V BlockLoaders. It is the "optimal" reference;
// vpipe's own sdpa_paged_mma_f16 already overtakes MLX end-to-end, but in
// ISOLATION trails this kernel ~1.8x, so for fresh prefill (contiguous
// K/V, q_offset == 0) we hand attention straight to it.
//
// The attn/* steel headers are SELF-CONTAINED (define their own
// BlockLoader/BlockMMA/BaseMMAFrag/MMATile) -- do NOT also pull in the
// gemm/* mma+loader (they would redefine those symbols). Only gemm/params.h
// (GEMMParams, referenced by a leftover GEMMKernel in attn.h) is shared.
//
// Buffer/param contract (steel `attention`):
//   0:Q 1:K 2:V 3:O (half)  4:AttnParams*  (5:AttnMaskParams 6:mask
//   7:sinks are function-constant-gated off: has_mask=has_sinks=false).
//   Function constants: align_Q(200), align_K(201), do_causal(301).
//   scale in AttnParams is the plain 1/sqrt(D) (the kernel folds in
//   M_LOG2E_F and uses exp2). Grid: threadgroups (NQ, H, B), tg (32,wm,wn)
//   = (32,4,1) for BD=128.
// Config for head_dim 128: BQ=32, BK=16, BD=128, WM=4, WN=1 (MLX default).

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

#ifndef METAL_FUNC
#define METAL_FUNC inline
#endif

using namespace metal;

// steel_attention.h needs Limits<> and M_LOG2E_F, normally supplied by the
// heavy mlx/backend/metal/kernels/utils.h. The attn/* headers themselves do
// NOT use them, so provide just these two here and skip vendoring utils.h.
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
template <>
struct Limits<half> {
  static constexpr constant half max = metal::numeric_limits<half>::infinity();
  static constexpr constant half min =
      -metal::numeric_limits<half>::infinity();
  static constexpr constant half finite_max =
      metal::numeric_limits<half>::max();
  static constexpr constant half finite_min =
      -metal::numeric_limits<half>::max();
};

#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"

// Concrete kernel entry point (the MLX instantiate_kernel expansion).
template [[host_name("attn_steel_h_bd128")]] [[kernel]] decltype(attention<
                                                                 half,
                                                                 32,
                                                                 16,
                                                                 128,
                                                                 4,
                                                                 1,
                                                                 half,
                                                                 float>)
attention<half, 32, 16, 128, 4, 1, half, float>;

// bf16 variant of the head_dim-128 steel flash-attention, for the
// Qwen-Image-Edit DiT: its residual stream reaches ~1e7 (beyond f16 range), so
// the whole DiT -- and hence its joint attention Q/K/V -- runs bf16. Same steel
// kernel, T=bfloat (accumulation stays f32). Lets the dual-stream MMDiT use the
// register-resident flash path instead of the scalar O(seq^2) sdpa at high res.
template [[host_name("attn_steel_h_bd128_bf16")]] [[kernel]] decltype(attention<
                                                                 bfloat,
                                                                 32,
                                                                 16,
                                                                 128,
                                                                 4,
                                                                 1,
                                                                 bfloat,
                                                                 float>)
attention<bfloat, 32, 16, 128, 4, 1, bfloat, float>;

// head_dim=120 variant for the Boogu-Image NextDiT (28 heads x 120). 120 is not
// a power of two, but every tiling constant still divides exactly, so the
// register-resident flash path needs no padding:
//   * MMA fragments:   TD = 120 / 8 = 15
//   * Q BlockLoader:   n_reads = 120*32/128 = 30, TCOLS = 4,  TROWS = 32
//   * K/V BlockLoader: n_reads = 120*16/128 = 15, TCOLS = 8,  TROWS = 16
//   * threadgroup mem: 32*(120+8) + max(24*120, 16*128) elts = ~14 KB of 32 KB
// The loaders read with scalar loops bounded by vec_size (no vector types), so a
// non-multiple-of-4 width is fine.
// MEASURED SLOWER than zero-padding to the bd128 kernel, and therefore OPT-IN
// (VPIPE_BOOGU_ATTN_NATIVE): 0.985x at seq 2271, 0.965x at seq 8415, even
// though it does 6.7% less arithmetic and skips a pad/unpad transpose pair.
// vec_size 30 at 60-byte thread strides cannot widen into aligned 16-byte
// loads the way bd128's 32 at 64-byte strides does, and the BD==128 simdgroup
// scheduling hints in the V-load loop do not apply here. Kept because it is
// numerically exact and makes the retest one env var on a new GPU.
template [[host_name("attn_steel_h_bd120_bf16")]] [[kernel]] decltype(attention<
                                                                 bfloat,
                                                                 32,
                                                                 16,
                                                                 120,
                                                                 4,
                                                                 1,
                                                                 bfloat,
                                                                 float>)
attention<bfloat, 32, 16, 120, 4, 1, bfloat, float>;

// (Qwen3.5 head_dim 256 full-attention prefill uses the paged-KV variant
// attn_steel_paged_bd256 below -- it serves both fresh and mid-context prefill
// straight from the paged pool, so the contiguous bd256 entry point is not
// instantiated.)

// head_dim=64 variant for the Qwen3-VL vision tower / Qwen3-ASR audio
// encoder (both d_model/n_heads == 64). Same BQ/BK/WM/WN as bd128; only
// BD changes. Driven non-causal (do_causal function constant = false) for
// the bidirectional ViT attention, but the entry point itself is
// causal-agnostic (the constant is bound at pipeline creation).
template [[host_name("attn_steel_h_bd64")]] [[kernel]] decltype(attention<
                                                                half,
                                                                32,
                                                                16,
                                                                64,
                                                                4,
                                                                1,
                                                                half,
                                                                float>)
attention<half, 32, 16, 64, 4, 1, half, float>;

// bf16 head_dim-64 twin, for the MiniMax-H3 video VAE's ViT decoder. Its
// 36 blocks run bf16 for the same reason the DiTs do -- the reference
// ships fp32 and f16's range is the thing that bites first -- so the f16
// entry point above cannot serve it.
template [[host_name("attn_steel_h_bd64_bf16")]] [[kernel]] decltype(attention<
                                                                bfloat,
                                                                32,
                                                                16,
                                                                64,
                                                                4,
                                                                1,
                                                                bfloat,
                                                                float>)
attention<bfloat, 32, 16, 64, 4, 1, bfloat, float>;

// attn_steel_paged -- the PAGED-KV port of MLX's steel register-resident
// flash. Keeps the register MMATile core (Stile/Otile in registers, online
// softmax via row_reduce/row_bin_op, tile_matmad QK, exp2 -- NO threadgroup
// score/output round-trip) but STAGES K/V STRAIGHT FROM THE PAGED KV POOL
// into threadgroup memory instead of MLX's contiguous BlockLoader. So
// mid-context prefill (q_offset>0) needs NO de-paged kfull/vfull device
// scratch -- the prefix K/V are read in place from the pool. Q is the new
// chunk (contiguous), causal by GLOBAL position, which subsumes the fresh
// case (q_offset==0).
//
// THE PAGE TABLE IS WALKED, not indexed. A page holds `nvalid` keys from
// global position `gstart`, and a page left partial by a branch (the
// parent's frozen tail) is followed by one that starts mid-sequence -- so
// position / page_tokens is NOT its page. Each page is read in BK-key
// blocks over [0, nvalid), keys past nvalid masked like keys past t_kv.
// MEASURED (Qwen3.5-4B): a 2200-token chunk prefilled onto a branched
// context matched the unbranched run on 1 of 12 greedy tokens when this
// kernel computed the page from the position; the flash kernels always
// walked the table.
//
// HEAD-DIM SPLIT AT BD 256 (MLX #4505). WN = 2 simdgroups share a query
// row group and split the head dim: each computes Q.K^T over its half of D,
// the pair exchanges partial scores through threadgroup memory (the K
// buffer, done with by then), both run the softmax on the full scores, and
// each accumulates P.V into its half of D -- half the O registers. Q stays
// in registers and the MMA inputs in half. With 32-key blocks: MEASURED on
// the M4 Pro (16 q / 4 kv heads, causal) 380 ms at n = 16384 against 523
// for the unsplit 16-key kernel (1.38x), and 26.0 against 35.4 ms at 4096.
// At BD 128 the split LOSES (0.87-0.91x, DiT and LM shapes) and so does a
// 32-key block alone (0.92-0.95x), so bd128 keeps the original shape.
//
//   0:q[Hq,n_q,D] 1:kpool 2:vpool 3:out[Hq,n_q,D] 4:scale 5:D 6:Hq 7:Hkv
//   8:n_q 9:q_offset 10:page_tokens 11:n_pages 12:page_table{pid,nvalid,gstart}
//   13:window (WIN only)
// page_tokens % BK == 0, so a block never leaves its page's allocation.
// grid (32*NQ, 4*Hq, WN); threadgroup (32,4,WN): 128 threads at bd128,
// 256 at bd256 (attn_steel_paged_bk / _wn in prefill-gqa-attn-set.h).
//
// WIN: a TRAILING SLIDING WINDOW (LFM3's sliding_attention layers) -- a
// query at position p reads keys (p - window, p]. Blocks below the lowest
// row's window are skipped (uniform per tile) and the rest masked per
// key. A row's first blocks can be wholly outside its own window while
// its running max is still finite_min; they add exp2(0) terms that the
// first in-window key's rescale (exp2(finite_min - max) = 0) erases, so
// every row ends on its own window alone (its own key is always in it).
// WIN false is the kernel it was, buffer 13 unread.
template <int BD, int BK, int WN, bool WIN = false>
[[kernel]] void attn_steel_paged(
    const device half*  q          [[buffer(0)]],
    const device half*  kpool      [[buffer(1)]],
    const device half*  vpool      [[buffer(2)]],
    device half*        out        [[buffer(3)]],
    constant float&     scale      [[buffer(4)]],
    constant int&       D          [[buffer(5)]],
    constant int&       Hq         [[buffer(6)]],
    constant int&       Hkv        [[buffer(7)]],
    constant int&       n_q        [[buffer(8)]],
    constant int&       q_offset   [[buffer(9)]],
    constant int&       page_tokens[[buffer(10)]],
    constant int&       n_pages    [[buffer(11)]],
    const device int*   page_table [[buffer(12)]],
    constant int&       window     [[buffer(13)]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]],
    uint  lid           [[thread_index_in_threadgroup]],
    uint3 tid           [[threadgroup_position_in_grid]])
{
  static_assert(WN == 1 || (WN == 2 && BK == 32),
                "the split keeps Q in registers, which wants 32-key blocks");
  constexpr int BQ = 32, WM = 4, FS = 8;
  constexpr int TQ = BQ / (WM * FS);          // 1
  constexpr int TK = BK / FS;
  constexpr int TD = BD / FS;
  constexpr int TDh = TD / WN;                // this simdgroup's D frags
  constexpr int BDh = BD / WN;
  constexpr int NTHREADS = WM * WN * 32;
  constexpr bool q_regs = WN == 2;
  constexpr short padH = 16 / sizeof(half);   // 8
  constexpr short LDQ = BD + padH;
  constexpr short LDK = BK + padH;            // K transposed: [BD, BK]
  constexpr short LDV = BD + padH;

  const int h      = (int)tid.y;
  const int qb     = (int)tid.x;
  const int kvh    = h / (Hq / Hkv);
  const int q_row0 = qb * BQ;
  if (q_row0 >= n_q) { return; }
  const int t_kv = q_offset + n_q;            // full context length
  const uint page_stride = (uint)Hkv * page_tokens * D;
  const uint head_off    = (uint)kvh * page_tokens * D;

  // K and V TIME-SHARE one tg buffer (K is consumed by QK^T before V is
  // loaded for PV), exactly like MLX's KV_smem, which keeps two threadgroups
  // resident per core at bd256 WN 1. Ks layout [d*LDK + key] (transposed),
  // Vs [key*LDV + d]. With the split it also carries the score exchange.
  threadgroup half Qs[q_regs ? 1 : BQ * LDQ];
  constexpr int KVSZ = (BD * LDK > BK * LDV) ? (BD * LDK) : (BK * LDV);
  threadgroup half KV_smem[KVSZ];
  threadgroup half* Ks = KV_smem;
  threadgroup half* Vs = KV_smem;

  // The split's MMA inputs are half (accumulation stays float); unsplit
  // keeps the original float fragments.
  using AccFrag = BaseMMAFrag<float, FS, FS>;
  using InT = metal::conditional_t<q_regs, half, float>;
  using InFrag = BaseMMAFrag<InT, FS, FS>;
  MMATile<InT, TQ, 1, InFrag>     Qtile;
  MMATile<InT, 1, TK, InFrag>     Ktile;
  MMATile<float, TQ, TK, AccFrag> Stile;
  MMATile<InT, 1, 1, InFrag>      Vtile;
  MMATile<float, TQ, TDh, AccFrag> Otile;
  MMATile<InT, 1, 1, InFrag>      Qtiles[q_regs ? TDh : 1];
  using stile_t = decltype(Stile);
  // MLX's vectorized BlockLoaders, re-pointed per key block at the page's
  // contiguous K/V. K loaded transposed (reduction_dim=0, kDstStrRow=1) ->
  // Ks[d*LDK + key]; V -> Vs[key*LDV + d].
  using KBL = BlockLoaderT<half, BK, BD, 1, LDK, 0, NTHREADS>;
  using VBL = BlockLoaderT<half, BK, BD, LDV, 1, 0, NTHREADS>;
  Otile.clear();

  const short2 sc = AccFrag::get_coord(simd_lane_id);
  const short sm = sc.y, sn = sc.x;
  const short row_group = (short)simd_group_id / WN;
  const short d_half = (short)simd_group_id % WN;
  const short tm = FS * TQ * row_group;
  const short Qs_off = (tm + sm) * LDQ + sn;
  const short Ks_off = d_half * BDh * LDK + sm * LDK + sn;
  const short Vs_off = sm * LDV + d_half * BDh + sn;
  constexpr short Qs_tstride = FS;
  constexpr short Ks_tstride = FS * LDK;

  const float lscale = scale * M_LOG2E_F;     // fold into exp2
  constexpr short kRowsPT = stile_t::kRowsPerThread;   // 1
  float max_score[kRowsPT], sum_score[kRowsPT];
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<float>::finite_min;
    sum_score[i] = 0.0f;
  }

  if constexpr (!q_regs) {
    // Stage Q [BQ, BD] (the new chunk) into Qs (rows past n_q zero-padded).
    for (uint e = lid; e < (uint)(BQ * BD); e += NTHREADS) {
      const int r = (int)e / BD, d = (int)e % BD;
      const int row = q_row0 + r;
      Qs[r * LDQ + d] =
          (row < n_q) ? q[((uint)h * n_q + row) * D + d] : (half)0;
    }
  } else {
    // This simdgroup's half of its Q rows, held for the whole key loop.
    const int row = q_row0 + tm + sm;
    const device half* qrow =
        q + ((uint)h * n_q + (uint)min(row, n_q - 1)) * D + d_half * BDh + sn;
    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TDh; ++dd) {
      if (q_row0 + BQ > n_q) {
        InFrag::load_safe(Qtiles[dd].frag_at(0, 0), qrow + dd * FS, D,
                          Int<1>{}, n_q - row, BDh - (dd * FS + sn),
                          Int<0>{}, Int<0>{});
      } else {
        InFrag::load(Qtiles[dd].frag_at(0, 0), qrow + dd * FS, D, Int<1>{});
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Causal: the highest global query pos in this tile bounds the key scan.
  const int q_tile_max = q_offset + min(q_row0 + BQ - 1, n_q - 1);
  // Window: the lowest query pos in this tile bounds it from below.
  const int win_lo = WIN ? q_offset + q_row0 - window + 1 : 0;

  for (int pg = 0; pg < n_pages; ++pg) {
    const int gstart = page_table[pg * 3 + 2];
    if (gstart > q_tile_max) { break; }       // past the tile (uniform)
    const int pid    = page_table[pg * 3 + 0];
    const int nvalid = page_table[pg * 3 + 1];
    if (WIN && gstart + nvalid <= win_lo) { continue; }   // uniform
    const device half* kpage = kpool + (uint)pid * page_stride + head_off;
    const device half* vpage = vpool + (uint)pid * page_stride + head_off;
  for (int local0 = 0; local0 < nvalid; local0 += BK) {
    const int gbs = gstart + local0;          // global key block start
    if (gbs > q_tile_max) { break; }          // uniform
    if (WIN && gbs + BK <= win_lo) { continue; }          // uniform
    // page_tokens % BK == 0 -> the block lies inside the page's allocation,
    // so load_unsafe never faults. Slots past nvalid (and keys past t_kv)
    // are read stale and masked to -inf below.
    KBL loader_k(kpage + (uint)local0 * D, D, Ks, simd_group_id,
                 simd_lane_id);
    VBL loader_v(vpage + (uint)local0 * D, D, Vs, simd_group_id,
                 simd_lane_id);
    threadgroup_barrier(mem_flags::mem_threadgroup);   // prev block's PV done
    loader_k.load_unsafe();                            // K -> shared KV_smem
    Stile.clear();
    threadgroup_barrier(mem_flags::mem_threadgroup);   // K ready

    // S = Q @ K^T over this simdgroup's part of the head dim.
    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TDh; ++dd) {
      simdgroup_barrier(mem_flags::mem_none);
      if constexpr (!q_regs) {
        Qtile.template load<half, 1, 1, LDQ, 1>(&Qs[Qs_off + dd * Qs_tstride]);
      }
      Ktile.template load<half, 1, 1, LDK, 1>(&Ks[Ks_off + dd * Ks_tstride]);
      simdgroup_barrier(mem_flags::mem_none);
      if constexpr (!q_regs) {
        tile_matmad(Stile, Qtile, Ktile, Stile);
      } else {
        tile_matmad(Stile, Qtiles[dd], Ktile, Stile);
      }
    }
    if constexpr (WN == 2) {
      // Exchange the partial scores with the other half of the pair through
      // KV_smem (K is consumed) and add. a + b == b + a, so both halves hold
      // bit-identical scores and run identical softmaxes.
      constexpr int xs = stile_t::kElemsPerTile * 32;
      static_assert(WM * WN * xs * sizeof(float) <= KVSZ * sizeof(half),
                    "the score exchange must fit in KV_smem");
      threadgroup_barrier(mem_flags::mem_threadgroup);
      threadgroup float* sx = reinterpret_cast<threadgroup float*>(KV_smem);
      threadgroup float* mine = sx + (row_group * WN + d_half) * xs;
      const short base = (short)simd_lane_id * stile_t::kElemsPerTile;
      STEEL_PRAGMA_UNROLL
      for (short ii = 0; ii < stile_t::kElemsPerTile; ++ii) {
        mine[base + ii] = Stile.elems()[ii];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const threadgroup float* peer =
          sx + (row_group * WN + 1 - d_half) * xs;
      STEEL_PRAGMA_UNROLL
      for (short ii = 0; ii < stile_t::kElemsPerTile; ++ii) {
        Stile.elems()[ii] += peer[base + ii];
      }
    }
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < stile_t::kElemsPerTile; ++ii) {
      Stile.elems()[ii] *= lscale;
    }
    // Causal + page + tail mask by GLOBAL position (each lane owns frag
    // elems).
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < stile_t::kTileRows; ++i) {
      const int row_pos =
          q_offset + q_row0 + tm + sm + i * stile_t::kFragRows;
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < stile_t::kTileCols; ++j) {
        const int lc0 = local0 + sn + j * stile_t::kFragCols;
        STEEL_PRAGMA_UNROLL
        for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; ++jj) {
          const int lc = lc0 + jj;
          const int gk = gstart + lc;
          if (lc >= nvalid || gk >= t_kv || row_pos < gk ||
              (WIN && gk <= row_pos - window)) {
            Stile.frag_at(i, j)[jj] = Limits<float>::finite_min;
          }
        }
      }
    }
    // K (and the exchange) fully consumed -> overwrite KV_smem with V (load
    // overlaps the register-only softmax that follows; PV waits on the
    // barrier below).
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_v.load_unsafe();
    // Online softmax on the register tile (no tg traffic).
    float new_max[kRowsPT];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) { new_max[i] = max_score[i]; }
    Stile.template row_reduce<MaxOp>(new_max);
    Stile.template row_bin_op<ExpSubOp>(new_max);
    float factor[kRowsPT];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }
    float sum_tmp[kRowsPT] = {0};
    Stile.template row_reduce<SumOp>(sum_tmp);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i] + sum_tmp[i];
    }
    Otile.template row_bin_op<MulOp>(factor);
    // P @ V into this simdgroup's part of the register O tile.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; ++iq) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TDh; ++id) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ++ik) {
          Vtile.template load<half, 1, 1, LDV, 1>(
              &Vs[Vs_off + ik * FS * LDV + id * FS]);
          AccFrag::mma(Otile.frag_at(iq, id), Stile.frag_at(iq, ik),
                       Vtile.frag_at(0, 0), Otile.frag_at(iq, id));
        }
      }
    }
  }
  }

  // Normalize + store this simdgroup's part of O [Hq, n_q, D].
  Otile.template row_bin_op<DivOp>(sum_score);
  threadgroup_barrier(mem_flags::mem_none);
  const int qL_rem = n_q - (n_q / BQ) * BQ;
  device half* O = out + ((uint)h * n_q + q_row0) * D + (tm + sm) * D
                   + d_half * BDh + sn;
  if (qL_rem != 0 && qb == (n_q / BQ)) {
    const short2 dims = short2(BDh - sn, (short)(qL_rem - (tm + sm)));
    if (dims.x > 0 && dims.y > 0) {
      Otile.template store_safe<half, 1, 1>(O, D, dims);
    }
  } else {
    Otile.template store<half, 1, 1>(O, D);
  }
}

template [[host_name("attn_steel_paged_bd256")]] [[kernel]]
decltype(attn_steel_paged<256, 32, 2>) attn_steel_paged<256, 32, 2>;
template [[host_name("attn_steel_paged_bd128")]] [[kernel]]
decltype(attn_steel_paged<128, 16, 1>) attn_steel_paged<128, 16, 1>;
template [[host_name("attn_steel_paged_win_bd128")]] [[kernel]]
decltype(attn_steel_paged<128, 16, 1, true>)
    attn_steel_paged<128, 16, 1, true>;

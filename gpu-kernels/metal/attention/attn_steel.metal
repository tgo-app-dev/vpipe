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

// attn_steel_paged_bd256 -- the PAGED-KV port of MLX's steel register-resident
// flash (head_dim 256). Keeps the register MMATile core verbatim (Stile/Otile
// in registers, online softmax via row_reduce/row_bin_op, tile_matmad QK, exp2
// -- NO threadgroup score/output round-trip) but STAGES K/V STRAIGHT FROM THE
// PAGED KV POOL into threadgroup memory (a page-walking cooperative load)
// instead of MLX's contiguous BlockLoader. So mid-context prefill (q_offset>0)
// needs NO de-paged kfull/vfull device scratch -- the prefix K/V are read in
// place from the pool. Q is the new chunk (contiguous), causal by GLOBAL
// position, which subsumes the fresh case (q_offset==0). page_tokens % BK(16)
// == 0 so a key block never straddles a page (one page lookup per block).
//   0:q[Hq,n_q,D] 1:kpool 2:vpool 3:out[Hq,n_q,D] 4:scale 5:D 6:Hq 7:Hkv
//   8:n_q 9:q_offset 10:page_tokens 11:n_pages 12:page_table{pid,nvalid,gstart}
// grid (32*NQ, 4*Hq, 1); threadgroup (32,4,1) = 128 threads.
kernel void attn_steel_paged_bd256(
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
    uint  simd_group_id [[simdgroup_index_in_threadgroup]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]],
    uint  lid           [[thread_index_in_threadgroup]],
    uint3 tid           [[threadgroup_position_in_grid]])
{
  constexpr int BQ = 32, BK = 16, BD = 256, WM = 4, WN = 1, FS = 8;
  constexpr int kNWarps = WM * WN;            // 4
  constexpr int TQ = BQ / (kNWarps * FS);     // 1
  constexpr int TK = BK / FS;                 // 2
  constexpr int TD = BD / FS;                 // 32
  constexpr int NTHREADS = WM * WN * 32;      // 128
  constexpr short padH = 16 / sizeof(half);   // 8
  constexpr short LDQ = BD + padH;            // 264
  constexpr short LDK = BK + padH;            // 24  (K transposed: [BD, BK])
  constexpr short LDV = BD + padH;            // 264

  const int h      = (int)tid.y;
  const int qb     = (int)tid.x;
  const int kvh    = h / (Hq / Hkv);
  const int q_row0 = qb * BQ;
  (void)n_pages;                  // walked via page_table; kept for the host ABI
  if (q_row0 >= n_q) { return; }
  const int t_kv = q_offset + n_q;            // full context length
  const uint page_stride = (uint)Hkv * page_tokens * D;
  const uint head_off    = (uint)kvh * page_tokens * D;

  threadgroup half Qs[BQ * LDQ];
  // K and V TIME-SHARE one tg buffer (K is consumed by QK^T before V is loaded
  // for PV), exactly like MLX's KV_smem -> tg stays ~28KB so 2 threadgroups
  // stay resident per core. Separate Ks+Vs would be ~37.6KB (1 group/core,
  // ~20% slower). Ks layout [d*LDK + key] (transposed), Vs [key*LDV + d].
  constexpr int KVSZ = (BD * LDK > BK * LDV) ? (BD * LDK) : (BK * LDV);
  threadgroup half KV_smem[KVSZ];
  threadgroup half* Ks = KV_smem;
  threadgroup half* Vs = KV_smem;

  using Frag = BaseMMAFrag<float, FS, FS>;
  MMATile<float, TQ, 1>  Qtile;
  MMATile<float, 1, TK>  Ktile;
  MMATile<float, TQ, TK> Stile;
  MMATile<float, 1, 1>   Vtile;
  MMATile<float, TQ, TD> Otile;
  using stile_t = decltype(Stile);
  // MLX's vectorized BlockLoaders, re-pointed per key block at the page's
  // contiguous K/V (a BK-block never straddles a page). K loaded transposed
  // (reduction_dim=0, kDstStrRow=1) -> Ks[d*LDK + key]; V -> Vs[key*LDV + d].
  using KBL = BlockLoaderT<half, BK, BD, 1, LDK, 0, NTHREADS>;
  using VBL = BlockLoaderT<half, BK, BD, LDV, 1, 0, NTHREADS>;
  Otile.clear();

  const short2 sc = Frag::get_coord(simd_lane_id);
  const short sm = sc.y, sn = sc.x;
  const short tm = FS * TQ * (short)simd_group_id;
  const short Qs_off = (tm + sm) * LDQ + sn;
  const short Ks_off = sm * LDK + sn;
  const short Vs_off = sm * LDV + sn;
  constexpr short Qs_tstride = FS;
  constexpr short Ks_tstride = FS * LDK;

  const float lscale = scale * M_LOG2E_F;     // fold into exp2
  constexpr short kRowsPT = TQ * Frag::kElemRows;   // 1
  float max_score[kRowsPT], sum_score[kRowsPT];
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<float>::finite_min;
    sum_score[i] = 0.0f;
  }

  // Stage Q [BQ, BD] (the new chunk) into Qs (rows past n_q zero-padded).
  for (uint e = lid; e < (uint)(BQ * BD); e += NTHREADS) {
    const int r = (int)e / BD, d = (int)e % BD;
    const int row = q_row0 + r;
    Qs[r * LDQ + d] = (row < n_q) ? q[((uint)h * n_q + row) * D + d] : (half)0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Causal: the highest global query pos in this tile bounds the key scan.
  const int q_tile_max = q_offset + min(q_row0 + BQ - 1, n_q - 1);
  const int kb_lim = min(t_kv - 1, q_tile_max) / BK + 1;

  for (int kb = 0; kb < kb_lim; ++kb) {
    const int gbs = kb * BK;                  // global key block start
    // gbs -> (page, local); page_tokens % BK == 0 -> block within one page, and
    // the page always has BK slots allocated -> load_unsafe never faults.
    // Keys past t_kv are read (stale) but masked to -inf in Stile below, so
    // their softmax/PV contribution is zero -- no explicit tail guard needed.
    const int pg     = gbs / page_tokens;
    const int pid    = page_table[pg * 3 + 0];
    const int local0 = gbs - pg * page_tokens;
    const device half* kbp = kpool + (uint)pid * page_stride + head_off
                             + (uint)local0 * D;
    const device half* vbp = vpool + (uint)pid * page_stride + head_off
                             + (uint)local0 * D;
    KBL loader_k(kbp, D, Ks, simd_group_id, simd_lane_id);
    VBL loader_v(vbp, D, Vs, simd_group_id, simd_lane_id);
    threadgroup_barrier(mem_flags::mem_threadgroup);   // prev block's V/PV done
    loader_k.load_unsafe();                            // K -> shared KV_smem
    Stile.clear();
    threadgroup_barrier(mem_flags::mem_threadgroup);   // K ready

    // S = Q @ K^T (register MMATile, contraction over the head dim).
    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; ++dd) {
      simdgroup_barrier(mem_flags::mem_none);
      Qtile.load<half, 1, 1, LDQ, 1>(&Qs[Qs_off + dd * Qs_tstride]);
      Ktile.load<half, 1, 1, LDK, 1>(&Ks[Ks_off + dd * Ks_tstride]);
      simdgroup_barrier(mem_flags::mem_none);
      tile_matmad(Stile, Qtile, Ktile, Stile);
    }
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < stile_t::kElemsPerTile; ++ii) {
      Stile.elems()[ii] *= lscale;
    }
    // Causal + tail mask by GLOBAL position (each lane owns frag elems).
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < stile_t::kTileRows; ++i) {
      const int row_pos =
          q_offset + q_row0 + tm + sm + i * stile_t::kFragRows;
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < stile_t::kTileCols; ++j) {
        const int col_pos = gbs + sn + j * stile_t::kFragCols;
        STEEL_PRAGMA_UNROLL
        for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; ++jj) {
          const int gk = col_pos + jj;
          if (gk >= t_kv || row_pos < gk) {
            Stile.frag_at(i, j)[jj] = Limits<float>::finite_min;
          }
        }
      }
    }
    // K is fully consumed -> overwrite KV_smem with V (load overlaps the
    // register-only softmax that follows; PV waits on the barrier below).
    threadgroup_barrier(mem_flags::mem_threadgroup);   // QK done reading K
    loader_v.load_unsafe();
    // Online softmax on the register tile (no tg traffic).
    float new_max[kRowsPT];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) { new_max[i] = max_score[i]; }
    Stile.row_reduce<MaxOp>(new_max);
    Stile.row_bin_op<ExpSubOp>(new_max);
    float factor[kRowsPT];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }
    float sum_tmp[kRowsPT] = {0};
    Stile.row_reduce<SumOp>(sum_tmp);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i] + sum_tmp[i];
    }
    Otile.row_bin_op<MulOp>(factor);
    // P @ V accumulate into the register O tile.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; ++iq) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TD; ++id) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ++ik) {
          Vtile.load<half, 1, 1, LDV, 1>(
              &Vs[Vs_off + ik * FS * LDV + id * FS]);
          Frag::mma(Otile.frag_at(iq, id), Stile.frag_at(iq, ik),
                    Vtile.frag_at(0, 0), Otile.frag_at(iq, id));
        }
      }
    }
  }

  // Normalize + store O [Hq, n_q, D].
  Otile.row_bin_op<DivOp>(sum_score);
  threadgroup_barrier(mem_flags::mem_none);
  const int qL_rem = n_q - (n_q / BQ) * BQ;
  device half* O = out + ((uint)h * n_q + q_row0) * D + (tm + sm) * D + sn;
  if (qL_rem != 0 && qb == (n_q / BQ)) {
    const short2 dims = short2(BD - sn, (short)(qL_rem - (tm + sm)));
    if (dims.x > 0 && dims.y > 0) {
      Otile.store_safe<half, 1, 1>(O, D, dims);
    }
  } else {
    Otile.store<half, 1, 1>(O, D);
  }
}

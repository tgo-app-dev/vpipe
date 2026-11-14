// local_window_attn.metal -- attention through a per-query SPATIAL WINDOW
// over a stack of key planes that share one token grid (RelayVSR's two
// networks, generative-models/relayvsr/).
//
// Every query (y, x) of an H x W grid attends, in each of up to four key
// planes, only the keys inside a rectangle of its own:
//
//   shifted == 1   a wh x ww window centred on the query and SHIFTED
//                  INWARD at the borders, so it is always whole (wh, ww
//                  clipped to the grid first) -- the keyframe DiT's
//                  spatial attention: top = clamp(y - wh/2, 0, H - wh).
//   shifted == 0   centred, HALF-OPEN and TRUNCATED at the borders: keys
//                  with dy in [-wh/2, wh/2) and dx in [-ww/2, ww/2) that
//                  lie on the grid -- the decoder's (FlashVSR's local GQA).
//
// One softmax runs over every visible key of every plane, which is what
// the references compute: one SDPA over the planes' concatenation under a
// boolean mask. Rows are not empty in either mode (a query sees its own
// position), so nothing here guards a 0/0.
//
// THE KERNEL is attn_steel_paged's register-resident flash core (MLX's
// steel MMATile, online softmax on the register tile, exp2) with its K/V
// staged by GATHER. A threadgroup takes a 4 x 8 tile of queries -- four
// simdgroups, one tile row of 8 consecutive tokens each, so a simdgroup's
// query rows and output rows are a contiguous run and every query in it
// shares a row -- and, per plane, walks the UNION of its queries' windows
// (a rectangle: both window rules are monotone in the query's position)
// in 16-key blocks taken in row-major order across the rectangle's rows,
// each key's address computed from its (row, column). A key outside a
// given query's own window is masked per element. The union costs
// (wh + 3)(ww + 7) keys a tile against wh * ww a query: 1.2x for a 48 x 48
// window, 1.45x for 24 x 24, 1.35x for 22 x 40.
//
// LAYOUT. Q, K, V and O are TOKEN-MAJOR with per-tensor strides (in
// elements): row r of head h at base + r * tok + h * head -- the
// projections' own [tokens, heads * D] output can be read as is, a fused
// q|k|v row too, and O lands where the output projection reads it. Each
// key plane has its own buffer (bind the same buffer at another offset
// for a ring's slots) and its own token stride. GQA: query head h reads
// key/value head h / (Hq / Hkv).
//
// RoPE is NOT applied here: Q and K arrive rotated.
//
//   0:q  1..4:k planes  5..8:v planes  9:out  10:params
//   grid threadgroups (tiles_x * tiles_y, Hq); tg (32, 4, 1)

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

// The host mirrors this struct (local-window-attn.h); keep the two in step.
struct LocalWindowAttnParams {
  int H, W;            // the token grid every plane shares
  int Hq, Hkv;
  int planes;          // 1..4
  int tiles_x;         // ceil(W / 8)
  int q_tok, q_head;   // element strides
  int o_tok, o_head;
  int k_head, v_head;  // head strides, shared by the planes
  float scale;         // 1 / sqrt(D)
  int k_tok[4];        // per plane token strides
  int v_tok[4];
  int win_h[4];
  int win_w[4];
  int shifted[4];
};

template <typename T, int BD>
[[kernel]] void local_window_attn(
    const device T* q   [[buffer(0)]],
    const device T* k0  [[buffer(1)]],
    const device T* k1  [[buffer(2)]],
    const device T* k2  [[buffer(3)]],
    const device T* k3  [[buffer(4)]],
    const device T* v0  [[buffer(5)]],
    const device T* v1  [[buffer(6)]],
    const device T* v2  [[buffer(7)]],
    const device T* v3  [[buffer(8)]],
    device T*       out [[buffer(9)]],
    constant LocalWindowAttnParams& p [[buffer(10)]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]],
    uint  lid           [[thread_index_in_threadgroup]],
    uint3 tid           [[threadgroup_position_in_grid]])
{
  constexpr int BQ = 32, BK = 16, WM = 4, FS = 8, TW = 8, TH = 4;
  constexpr int TK = BK / FS;                 // 2
  constexpr int TD = BD / FS;
  constexpr int NTHREADS = WM * 32;
  constexpr short padH = 16 / sizeof(T);      // 8
  constexpr short LDQ = BD + padH;
  constexpr short LDK = BK + padH;            // K transposed: [BD, BK]
  constexpr short LDV = BD + padH;
  constexpr int V4 = BD / 4;                  // vec4 loads per key row

  const int h   = (int)tid.y;
  const int kvh = h / (p.Hq / p.Hkv);
  const int ty0 = ((int)tid.x / p.tiles_x) * TH;
  const int tx0 = ((int)tid.x % p.tiles_x) * TW;
  if (ty0 >= p.H) { return; }
  const int ty1 = min(ty0 + TH, p.H), tx1 = min(tx0 + TW, p.W);

  threadgroup T Qs[BQ * LDQ];
  constexpr int KVSZ = (BD * LDK > BK * LDV) ? (BD * LDK) : (BK * LDV);
  threadgroup T KV_smem[KVSZ];
  threadgroup T* Ks = KV_smem;
  threadgroup T* Vs = KV_smem;

  using AccFrag = BaseMMAFrag<float, FS, FS>;
  MMATile<float, 1, 1, AccFrag>  Qtile;
  MMATile<float, 1, TK, AccFrag> Ktile;
  MMATile<float, 1, TK, AccFrag> Stile;
  MMATile<float, 1, 1, AccFrag>  Vtile;
  MMATile<float, 1, TD, AccFrag> Otile;
  using stile_t = decltype(Stile);
  Otile.clear();

  const short2 sc = AccFrag::get_coord(simd_lane_id);
  const short sm = sc.y, sn = sc.x;
  const short tm = FS * (short)simd_group_id;
  const short Qs_off = (tm + sm) * LDQ + sn;
  const short Ks_off = sm * LDK + sn;
  const short Vs_off = sm * LDV + sn;

  // This lane's query: row ty0 + simdgroup, column tx0 + sm.
  const int qy = ty0 + (int)simd_group_id;
  const int qx = tx0 + (int)sm;

  // Stage Q [32, BD]: row r is the token at (ty0 + r / 8, tx0 + r % 8),
  // zero past the grid.
  for (int e = (int)lid; e < BQ * V4; e += NTHREADS) {
    const int r = e / V4, d = (e % V4) * 4;
    const int y = ty0 + r / TW, x = tx0 + r % TW;
    vec<T, 4> v = vec<T, 4>(T(0));
    if (y < ty1 && x < tx1) {
      v = *(const device vec<T, 4>*)(q + (long)(y * p.W + x) * p.q_tok +
                                     (long)h * p.q_head + d);
    }
    *(threadgroup vec<T, 4>*)(&Qs[r * LDQ + d]) = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float lscale = p.scale * M_LOG2E_F;
  float max_score = Limits<float>::finite_min;
  float sum_score = 0.0f;

  for (int pl = 0; pl < p.planes; ++pl) {
    const device T* kp = pl == 0 ? k0 : pl == 1 ? k1 : pl == 2 ? k2 : k3;
    const device T* vp = pl == 0 ? v0 : pl == 1 ? v1 : pl == 2 ? v2 : v3;
    const int ktok = p.k_tok[pl], vtok = p.v_tok[pl];
    kp += (long)kvh * p.k_head;
    vp += (long)kvh * p.v_head;
    const bool sh = p.shifted[pl] != 0;
    const int wh = sh ? min(p.win_h[pl], p.H) : p.win_h[pl];
    const int ww = sh ? min(p.win_w[pl], p.W) : p.win_w[pl];
    // The union of the tile's windows, and this lane's own window.
    int r0, r1, c0, c1, qt, qb, ql, qr;
    if (sh) {
      r0 = clamp(ty0 - wh / 2, 0, p.H - wh);
      r1 = clamp(ty1 - 1 - wh / 2, 0, p.H - wh) + wh;
      c0 = clamp(tx0 - ww / 2, 0, p.W - ww);
      c1 = clamp(tx1 - 1 - ww / 2, 0, p.W - ww) + ww;
      qt = clamp(qy - wh / 2, 0, p.H - wh);
      ql = clamp(qx - ww / 2, 0, p.W - ww);
      qb = qt + wh;
      qr = ql + ww;
    } else {
      r0 = max(0, ty0 - wh / 2);
      r1 = min(p.H, ty1 - 1 + wh / 2);
      c0 = max(0, tx0 - ww / 2);
      c1 = min(p.W, tx1 - 1 + ww / 2);
      qt = qy - wh / 2;
      qb = qy + wh / 2;
      ql = qx - ww / 2;
      qr = qx + ww / 2;
    }
    const int rw = c1 - c0;
    const int cnt = (r1 - r0) * rw;

    for (int kb = 0; kb < cnt; kb += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);   // prev PV done
      // K transposed into Ks[d * LDK + key]; keys past the union zero.
      for (int e = (int)lid; e < BK * V4; e += NTHREADS) {
        const int key = e / V4, d = (e % V4) * 4;
        const int j = kb + key;
        vec<T, 4> v = vec<T, 4>(T(0));
        if (j < cnt) {
          const int ky = r0 + j / rw, kx = c0 + j % rw;
          v = *(const device vec<T, 4>*)(kp + (long)(ky * p.W + kx) * ktok +
                                         d);
        }
        Ks[(d + 0) * LDK + key] = v[0];
        Ks[(d + 1) * LDK + key] = v[1];
        Ks[(d + 2) * LDK + key] = v[2];
        Ks[(d + 3) * LDK + key] = v[3];
      }
      Stile.clear();
      threadgroup_barrier(mem_flags::mem_threadgroup);   // K ready

      STEEL_PRAGMA_UNROLL
      for (short dd = 0; dd < TD; ++dd) {
        simdgroup_barrier(mem_flags::mem_none);
        Qtile.template load<T, 1, 1, LDQ, 1>(&Qs[Qs_off + dd * FS]);
        Ktile.template load<T, 1, 1, LDK, 1>(&Ks[Ks_off + dd * FS * LDK]);
        simdgroup_barrier(mem_flags::mem_none);
        tile_matmad(Stile, Qtile, Ktile, Stile);
      }
      STEEL_PRAGMA_UNROLL
      for (short ii = 0; ii < stile_t::kElemsPerTile; ++ii) {
        Stile.elems()[ii] *= lscale;
      }
      // Each lane's two scores per fragment: its query against keys
      // sn, sn + 1 of the fragment.
      STEEL_PRAGMA_UNROLL
      for (short jf = 0; jf < TK; ++jf) {
        STEEL_PRAGMA_UNROLL
        for (short jj = 0; jj < 2; ++jj) {
          const int j = kb + sn + jf * FS + jj;
          bool vis = j < cnt;
          if (vis) {
            const int ky = r0 + j / rw, kx = c0 + j % rw;
            vis = ky >= qt && ky < qb && kx >= ql && kx < qr;
          }
          if (!vis) {
            Stile.frag_at(0, jf)[jj] = Limits<float>::finite_min;
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);   // K consumed
      for (int e = (int)lid; e < BK * V4; e += NTHREADS) {
        const int key = e / V4, d = (e % V4) * 4;
        const int j = kb + key;
        vec<T, 4> v = vec<T, 4>(T(0));
        if (j < cnt) {
          const int ky = r0 + j / rw, kx = c0 + j % rw;
          v = *(const device vec<T, 4>*)(vp + (long)(ky * p.W + kx) * vtok +
                                         d);
        }
        *(threadgroup vec<T, 4>*)(&Vs[key * LDV + d]) = v;
      }
      // Online softmax on the register tile.
      float new_max = max_score;
      Stile.template row_reduce<MaxOp>(&new_max);
      Stile.template row_bin_op<ExpSubOp>(&new_max);
      float factor = fast::exp2(max_score - new_max);
      max_score = new_max;
      float sum_tmp = 0.0f;
      Stile.template row_reduce<SumOp>(&sum_tmp);
      sum_score = sum_score * factor + sum_tmp;
      Otile.template row_bin_op<MulOp>(&factor);
      threadgroup_barrier(mem_flags::mem_threadgroup);   // V ready
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TD; ++id) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ++ik) {
          Vtile.template load<T, 1, 1, LDV, 1>(
              &Vs[Vs_off + ik * FS * LDV + id * FS]);
          AccFrag::mma(Otile.frag_at(0, id), Stile.frag_at(0, ik),
                       Vtile.frag_at(0, 0), Otile.frag_at(0, id));
        }
      }
    }
  }

  // Normalize and store this lane's query row, its two columns per frag.
  Otile.template row_bin_op<DivOp>(&sum_score);
  if (qy < ty1 && qx < tx1) {
    device T* o = out + (long)(qy * p.W + qx) * p.o_tok + (long)h * p.o_head +
                  sn;
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TD; ++id) {
      o[id * FS + 0] = static_cast<T>(Otile.frag_at(0, id)[0]);
      o[id * FS + 1] = static_cast<T>(Otile.frag_at(0, id)[1]);
    }
  }
}

template [[host_name("local_window_attn_bd64_f16")]] [[kernel]]
decltype(local_window_attn<half, 64>) local_window_attn<half, 64>;
template [[host_name("local_window_attn_bd64_bf16")]] [[kernel]]
decltype(local_window_attn<bfloat, 64>) local_window_attn<bfloat, 64>;
template [[host_name("local_window_attn_bd128_f16")]] [[kernel]]
decltype(local_window_attn<half, 128>) local_window_attn<half, 128>;
template [[host_name("local_window_attn_bd128_bf16")]] [[kernel]]
decltype(local_window_attn<bfloat, 128>) local_window_attn<bfloat, 128>;

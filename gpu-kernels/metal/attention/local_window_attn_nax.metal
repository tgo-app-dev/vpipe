// local_window_attn_nax.metal -- local_window_attn on the MATRIX CORES
// (M5 and later), and with SageAttention's int8 QK and int8 P*V.
//
// The same attention as local_window_attn.metal: every query (y, x) of an
// H x W grid attends, in each of up to four key planes sharing that grid,
// only the keys in a rectangle of its own (`shifted` 1: a wh x ww window
// shifted inward at the borders; 0: centred, half-open, truncated), one
// softmax over every visible key of every plane. Q, K, V and O are
// token-major with per-tensor strides; RoPE is not applied here.
//
// THE KERNEL is attention_nax's core (steel_attention_nax.h): MPP
// matmul2d fragments, 16 x 32 x 16, the online softmax on the register
// tile. What differs is the walk. A SIMDGROUP takes a 4 x 4 block of
// queries -- the 16 rows of one fragment -- and walks, per plane, the
// union of their windows (a rectangle) 32 keys at a time in row-major
// order across its rows, each key GATHERED into the fragment registers by
// its own address: a fragment is plain registers loaded element by
// element, so no threadgroup memory is staged and no barrier is taken. A
// key outside a query's own window is masked per element. Four
// simdgroups -- an 8 x 8 block -- a threadgroup, so the unions they read
// overlap in the caches. The union costs (wh + 3)(ww + 3) keys a block
// against wh * ww a query: 1.13x for 48 x 48, 1.27x for 24 x 24, 1.22x
// for 22 x 40.
//
// INT8 (function constant 0, `lwa_qk8`): Q and K as int8, ONE SCALE A
// TOKEN each (Q per query row, K per key -- a key block here is a run of
// the union's rows, not a block of memory, so a per-block scale has no
// block to belong to), K over a CENTRE shared by every plane (each
// channel's midrange): subtracting one vector from every key a row sees
// shifts its scores by one q.centre, which softmax does not see -- the
// reason it must be ONE vector across the planes (one per plane would
// shift the planes' scores apart).
// A score is the int32 product by the row's scale by the key's.
//
// INT8 P*V (function constant 1, `lwa_pv8`, with lwa_qk8): attention_nax's
// held product, as it is -- P on a uint8 grid against the row's running
// max, V minus its centre with one step a channel, plus 128, uint8; the sum
// held in int32 across key blocks, rescaled when a row's max grows, the
// offset taken out with P's integer row sums, a row past 2^30 stepped
// down. V's centre and step are, again, ONE across the planes: the held
// sum mixes every plane's keys.
//
//   0:q  1..4:k planes  5..8:v planes  9:out  10:params
//   11:q8 (int8 [Hq, n, BD])  12:q_scale (float [Hq, n])
//   13:k8 (int8 [planes, Hkv, n, BD])  14:k_scale (float [planes, Hkv, n])
//   15:v8 (uint8 [planes, Hkv, n, BD])  16:v_scale, 17:v_mean (float
//   [Hkv, BD])
//   grid threadgroups (ceil(W / 8) * ceil(H / 8), Hq); tg (32, 4, 1)

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

// The host mirrors this struct (relayvsr-ops.h); keep the two in step.
struct LwaNaxParams {
  int H, W;            // the token grid every plane shares; n = H * W
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

#if defined(__HAVE_TENSOR__)

#include "mlx/backend/metal/kernels/steel/attn/nax.h"

using namespace mlx::steel;

constant bool lwa_qk8_set [[function_constant(0)]];
constant bool lwa_qk8 =
    is_function_constant_defined(lwa_qk8_set) ? lwa_qk8_set : false;
constant bool lwa_pv8_set [[function_constant(1)]];
constant bool lwa_pv8 =
    is_function_constant_defined(lwa_pv8_set) ? lwa_pv8_set : false;

struct LwaMax {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) { return metal::max(x, y); }
};
struct LwaSum {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) { return x + y; }
};
struct LwaMul {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) { return x * y; }
};
struct LwaExpSub {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) { return fast::exp2(x - y); }
};
// The same, rounded to half, for the f16 / bf16 P.V: the matrix units read
// an f32 P at 16 bits (MEASURED: twice the steel kernel's error against
// f64 in f16, and keyframes 1.17x the reference's bf16 distance, until the
// row sum added the values the product reads). Rounded once here, the sum
// and the product agree. The int8 QK mode keeps P as is: its error is the
// int8 scores' (the same to three digits), and the rounding cost its hd128
// kernel a third.
struct LwaExpSubH {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return (T)(half)fast::exp2(x - y);
  }
};

// Union index j into its (row, column), j / rw by a float reciprocal and
// one correction each way (exact for any j a frame has).
METAL_FUNC int2
lwa_rc_(int j, int rw, float inv)
{
  int r = (int)((float)j * inv);
  r -= (r * rw > j) ? 1 : 0;
  r += ((r + 1) * rw <= j) ? 1 : 0;
  return int2(r, j - r * rw);
}

template <typename T, int BD>
[[kernel, max_total_threads_per_threadgroup(128)]] void local_window_attn_nax(
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
    constant LwaNaxParams& p [[buffer(10)]],
    const device int8_t* q8  [[buffer(11), function_constant(lwa_qk8)]],
    const device float* qsc  [[buffer(12), function_constant(lwa_qk8)]],
    const device int8_t* k8  [[buffer(13), function_constant(lwa_qk8)]],
    const device float* ksc  [[buffer(14), function_constant(lwa_qk8)]],
    const device uint8_t* v8 [[buffer(15), function_constant(lwa_pv8)]],
    const device float* vsc  [[buffer(16), function_constant(lwa_pv8)]],
    const device float* vmu  [[buffer(17), function_constant(lwa_pv8)]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid           [[threadgroup_position_in_grid]])
{
  constexpr short kU = 16;
  constexpr short TD = BD / kU;
  constexpr int BK = 32;
  constexpr short TK = BK / kU;                // 2
  using otile_t = NAXTile<float, 1, TD>;
  using stile_t = NAXTile<float, 1, TK>;
  constexpr short kRowsPT = otile_t::kRowsPerThread;   // 2
  constexpr short kCols = otile_t::kFragThrCols;       // 4

  const int n = p.H * p.W;
  const int h = (int)tid.y;
  const int kvh = h / (p.Hq / p.Hkv);
  const int ty0 = ((int)tid.x / p.tiles_x) * 8 + ((int)simd_group_id >> 1) * 4;
  const int tx0 = ((int)tid.x % p.tiles_x) * 8 + ((int)simd_group_id & 1) * 4;
  if (ty0 >= p.H || tx0 >= p.W) { return; }
  const int ty1 = min(ty0 + 4, p.H), tx1 = min(tx0 + 4, p.W);

  const short2 sc = otile_t::NAXFrag_t::get_coord();
  const short sm = sc.y, sn = sc.x;

  // This thread's two query rows, block indices sm and sm + 8 (a 4 x 4
  // block, row-major). A row past the grid computes a valid neighbour's
  // window -- its scores stay finite -- and is not stored.
  int qy[kRowsPT], qx[kRowsPT], qtok[kRowsPT];
  bool qok[kRowsPT];
  STEEL_PRAGMA_UNROLL
  for (short e = 0; e < kRowsPT; ++e) {
    const int i = sm + 8 * e;
    const int y = ty0 + i / 4, x = tx0 + i % 4;
    qok[e] = y < ty1 && x < tx1;
    qy[e] = min(y, ty1 - 1);
    qx[e] = min(x, tx1 - 1);
    qtok[e] = qy[e] * p.W + qx[e];
  }

  // Q, held in registers for the whole walk.
  NAXTile<T, 1, TD> Qt;
  NAXTile<int8_t, 1, TD> Q8t;
  float q_row[kRowsPT] = {0.0f, 0.0f};
  STEEL_PRAGMA_UNROLL
  for (short id = 0; id < TD; ++id) {
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < kRowsPT; ++e) {
      if (lwa_qk8) {
        char4 v = char4(0);
        if (qok[e]) {
          v = *(const device char4*)(q8 + ((long)h * n + qtok[e]) * BD +
                                     id * kU + sn);
        }
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kCols; ++j) {
          Q8t.frag_at(0, id)[e * kCols + j] = (int8_t)v[j];
        }
      } else {
        vec<T, 4> v = vec<T, 4>(T(0));
        if (qok[e]) {
          v = *(const device vec<T, 4>*)(q + (long)qtok[e] * p.q_tok +
                                         (long)h * p.q_head + id * kU + sn);
        }
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kCols; ++j) {
          Qt.frag_at(0, id)[e * kCols + j] = v[j];
        }
      }
    }
  }
  if (lwa_qk8) {
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < kRowsPT; ++e) {
      q_row[e] = qok[e] ? qsc[(long)h * n + qtok[e]] : 0.0f;
    }
  }

  // A MASKED SCORE IS THE LOWEST FLOAT under a running max that starts at
  // a floor far above it (both finite: fast::exp2 is not asked about an
  // infinity), so a key outside a row's window adds exp2(-3e38) = 0 --
  // also before the row has seen its first visible key, where a mask
  // equal to the starting max would add exp2(0) = 1 a key and leave it to
  // the next rescale to take back. That matters to the int8 P*V: the walk
  // reaches each of a block's rows' windows at a different key, so rows
  // START at different blocks, and a row that has not started needs no
  // rescale of its held sum (it is zero).
  const float kFloor = -1.0e30f;
  const float kMasked = metal::numeric_limits<float>::lowest();
  const float lscale = p.scale * M_LOG2E_F;
  metal::vec<float, kRowsPT> max_score, sum_score;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = kFloor;
    sum_score[i] = 0.0f;
  }
  otile_t Otile;
  Otile.clear();
  // The int8 P*V's held product, P's row sums since the offset was last
  // taken out (this thread's columns), and each row's P grid.
  NAXTile<int32_t, 1, TD> Oacc;
  int r_part[kRowsPT] = {0, 0};
  int p_lvl[kRowsPT] = {255, 255};
  if (lwa_pv8) { Oacc.clear(); }

#define LWA_PV8_FIX()                                                     \
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
        for (short jj = 0; jj < kCols; jj++) {                            \
          oi[ii * kCols + jj] -= off;                                     \
        }                                                                 \
      }                                                                   \
    }                                                                     \
  }

  int visited = 0;
  for (int pl = 0; pl < p.planes; ++pl) {
    const device T* kp = pl == 0 ? k0 : pl == 1 ? k1 : pl == 2 ? k2 : k3;
    const device T* vp = pl == 0 ? v0 : pl == 1 ? v1 : pl == 2 ? v2 : v3;
    const long ktok = p.k_tok[pl], vtok = p.v_tok[pl];
    kp += (long)kvh * p.k_head;
    vp += (long)kvh * p.v_head;
    // The int8 operands of this plane and KV head, [n, BD] each.
    const long pbase = ((long)pl * p.Hkv + kvh) * n;
    const bool sh = p.shifted[pl] != 0;
    const int wh = sh ? min(p.win_h[pl], p.H) : p.win_h[pl];
    const int ww = sh ? min(p.win_w[pl], p.W) : p.win_w[pl];
    int r0, r1, c0, c1;
    int qt[kRowsPT], qb[kRowsPT], ql[kRowsPT], qr[kRowsPT];
    if (sh) {
      r0 = clamp(ty0 - wh / 2, 0, p.H - wh);
      r1 = clamp(ty1 - 1 - wh / 2, 0, p.H - wh) + wh;
      c0 = clamp(tx0 - ww / 2, 0, p.W - ww);
      c1 = clamp(tx1 - 1 - ww / 2, 0, p.W - ww) + ww;
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < kRowsPT; ++e) {
        qt[e] = clamp(qy[e] - wh / 2, 0, p.H - wh);
        ql[e] = clamp(qx[e] - ww / 2, 0, p.W - ww);
        qb[e] = qt[e] + wh;
        qr[e] = ql[e] + ww;
      }
    } else {
      r0 = max(0, ty0 - wh / 2);
      r1 = min(p.H, ty1 - 1 + wh / 2);
      c0 = max(0, tx0 - ww / 2);
      c1 = min(p.W, tx1 - 1 + ww / 2);
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < kRowsPT; ++e) {
        qt[e] = qy[e] - wh / 2;
        qb[e] = qy[e] + wh / 2;
        ql[e] = qx[e] - ww / 2;
        qr[e] = qx[e] + ww / 2;
      }
    }
    const int rw = c1 - c0;
    const int cnt = (r1 - r0) * rw;
    const float inv_rw = 1.0f / (float)rw;

    for (int kb = 0; kb < cnt; kb += BK, ++visited) {
      // The four keys this thread LOADS -- K's and V's fragment rows,
      // kb + f * 16 + sm + 8 e -- and the eight its SCORES are against,
      // kb + f * 16 + sn + jj.
      // 32-bit: a frame's tokens are far under 2^31, and these are live
      // across the block's MMAs.
      int ltok[TK][kRowsPT];
      bool lok[TK][kRowsPT];
      STEEL_PRAGMA_UNROLL
      for (short f = 0; f < TK; ++f) {
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < kRowsPT; ++e) {
          const int j = kb + f * kU + sm + 8 * e;
          lok[f][e] = j < cnt;
          const int2 rc = lwa_rc_(j < cnt ? j : 0, rw, inv_rw);
          ltok[f][e] = (r0 + rc.x) * p.W + (c0 + rc.y);
        }
      }
      bool vis[TK][kRowsPT][kCols];
      int stok[TK][kCols];
      STEEL_PRAGMA_UNROLL
      for (short f = 0; f < TK; ++f) {
        STEEL_PRAGMA_UNROLL
        for (short jj = 0; jj < kCols; ++jj) {
          const int j = kb + f * kU + sn + jj;
          const bool in = j < cnt;
          const int2 rc = lwa_rc_(in ? j : 0, rw, inv_rw);
          const int ky = r0 + rc.x, kx = c0 + rc.y;
          stok[f][jj] = ky * p.W + kx;
          STEEL_PRAGMA_UNROLL
          for (short e = 0; e < kRowsPT; ++e) {
            vis[f][e][jj] = in && ky >= qt[e] && ky < qb[e] &&
                            kx >= ql[e] && kx < qr[e];
          }
        }
      }

      stile_t Stile;
      if (lwa_qk8) {
        NAXTile<int32_t, 1, TK> Si;
        Si.clear();
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < TD; ++id) {
          NAXTile<int8_t, 2, 1> Kt;
          STEEL_PRAGMA_UNROLL
          for (short f = 0; f < TK; ++f) {
            STEEL_PRAGMA_UNROLL
            for (short e = 0; e < kRowsPT; ++e) {
              char4 v = char4(0);
              if (lok[f][e]) {
                v = *(const device char4*)(k8 + (pbase + ltok[f][e]) * BD +
                                           id * kU + sn);
              }
              STEEL_PRAGMA_UNROLL
              for (short j = 0; j < kCols; ++j) {
                Kt.frag_at(f, 0)[e * kCols + j] = (int8_t)v[j];
              }
            }
          }
          NAXTile<int32_t, 1, TK>::NAXFrag_t::template
              mma<int32_t, int8_t, int8_t>(
              Si.frag_at(0, 0), Si.frag_at(0, 1), Q8t.frag_at(0, id),
              metal::false_type{}, Kt.frag_at(0, 0), Kt.frag_at(1, 0),
              metal::true_type{});
        }
        STEEL_PRAGMA_UNROLL
        for (short f = 0; f < TK; ++f) {
          float kcol[kCols];
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < kCols; ++jj) {
            kcol[jj] = ksc[pbase + stok[f][jj]] * lscale;
          }
          STEEL_PRAGMA_UNROLL
          for (short e = 0; e < kRowsPT; ++e) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < kCols; ++jj) {
              Stile.frag_at(0, f)[e * kCols + jj] =
                  (float)Si.frag_at(0, f)[e * kCols + jj] * q_row[e] *
                  kcol[jj];
            }
          }
        }
      } else {
        Stile.clear();
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < TD; ++id) {
          NAXTile<T, 2, 1> Kt;
          STEEL_PRAGMA_UNROLL
          for (short f = 0; f < TK; ++f) {
            STEEL_PRAGMA_UNROLL
            for (short e = 0; e < kRowsPT; ++e) {
              vec<T, 4> v = vec<T, 4>(T(0));
              if (lok[f][e]) {
                v = *(const device vec<T, 4>*)(kp + (long)ltok[f][e] * ktok +
                                               id * kU + sn);
              }
              STEEL_PRAGMA_UNROLL
              for (short j = 0; j < kCols; ++j) {
                Kt.frag_at(f, 0)[e * kCols + j] = v[j];
              }
            }
          }
          stile_t::NAXFrag_t::mma(
              Stile.frag_at(0, 0), Stile.frag_at(0, 1), Qt.frag_at(0, id),
              metal::false_type{}, Kt.frag_at(0, 0), Kt.frag_at(1, 0),
              metal::true_type{});
        }
        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < stile_t::kElemsPerTile; ++ii) {
          Stile.elems()[ii] *= lscale;
        }
      }
      // Each query sees only its own window.
      STEEL_PRAGMA_UNROLL
      for (short f = 0; f < TK; ++f) {
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < kRowsPT; ++e) {
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < kCols; ++jj) {
            if (!vis[f][e][jj]) {
              Stile.frag_at(0, f)[e * kCols + jj] = kMasked;
            }
          }
        }
      }

      // The online softmax.
      metal::vec<float, kRowsPT> new_max, factor;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kRowsPT; ++i) { new_max[i] = max_score[i]; }
      Stile.template row_reduce<LwaMax>(new_max);
      if (lwa_pv8 || lwa_qk8) {
        Stile.template row_bin_op<LwaExpSub>(new_max);
      } else {
        Stile.template row_bin_op<LwaExpSubH>(new_max);
      }
      // A row whose max grew AFTER it started (it held something) is the
      // one whose held int8 sum must be rescaled.
      bool grew = false;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kRowsPT; ++i) {
        grew = grew || (new_max[i] > max_score[i] && max_score[i] > kFloor);
        factor[i] = fast::exp2(max_score[i] - new_max[i]);
        max_score[i] = new_max[i];
        sum_score[i] = sum_score[i] * factor[i];
      }
      Stile.template row_reduce<LwaSum>(sum_score);

      if (lwa_pv8) {
        if (simd_any(grew)) {
          LWA_PV8_FIX();
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < kRowsPT; ii++) {
            if (factor[ii] < 1.0f) { p_lvl[ii] = 255; }
          }
          STEEL_PRAGMA_UNROLL
          for (short id = 0; id < TD; id++) {
            thread auto& oi = Oacc.frag_at(0, id);
            STEEL_PRAGMA_UNROLL
            for (short ii = 0; ii < kRowsPT; ii++) {
              STEEL_PRAGMA_UNROLL
              for (short jj = 0; jj < kCols; jj++) {
                const short loc = ii * kCols + jj;
                oi[loc] = (int32_t)metal::rint((float)oi[loc] * factor[ii]);
              }
            }
          }
        }
        NAXTile<uint8_t, 1, TK> Pu;
        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {
          Pu.elems()[ii] = (uint8_t)metal::clamp(
              metal::rint(Stile.elems()[ii] * 255.0f), 0.0f, 255.0f);
        }
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < kRowsPT; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < kCols; jj++) {
              r_part[ii] += (int)Pu.frag_at(0, ik)[ii * kCols + jj];
            }
          }
        }
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < TD; id += 2) {
          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            NAXTile<uint8_t, 1, 2> Vt;
            STEEL_PRAGMA_UNROLL
            for (short e = 0; e < kRowsPT; ++e) {
              uchar4 a = uchar4(0), b = uchar4(0);
              if (lok[ik][e]) {
                const device uint8_t* vr =
                    v8 + (pbase + ltok[ik][e]) * BD + id * kU + sn;
                a = *(const device uchar4*)(vr);
                b = *(const device uchar4*)(vr + kU);
              }
              STEEL_PRAGMA_UNROLL
              for (short j = 0; j < kCols; ++j) {
                Vt.frag_at(0, 0)[e * kCols + j] = a[j];
                Vt.frag_at(0, 1)[e * kCols + j] = b[j];
              }
            }
            NAXTile<int32_t, 1, 2>::NAXFrag_t::template
                mma<int32_t, uint8_t, uint8_t>(
                Oacc.frag_at(0, id), Oacc.frag_at(0, id + 1),
                Pu.frag_at(0, ik), metal::false_type{}, Vt.frag_at(0, 0),
                Vt.frag_at(0, 1), metal::false_type{});
          }
        }
        // Every 64 visited blocks: the offset out, and a row whose held
        // sum passed 2^30 stepped down (attention_nax's rule, as it is).
        if ((visited & 63) == 63) {
          LWA_PV8_FIX();
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < kRowsPT; ii++) {
            int big = 0;
            STEEL_PRAGMA_UNROLL
            for (short id = 0; id < TD; id++) {
              STEEL_PRAGMA_UNROLL
              for (short jj = 0; jj < kCols; jj++) {
                big = metal::max(
                    big, metal::abs(Oacc.frag_at(0, id)[ii * kCols + jj]));
              }
            }
            big = metal::max(big, simd_shuffle_xor(big, ushort(1)));
            big = metal::max(big, simd_shuffle_xor(big, ushort(8)));
            if (big > (1 << 30) && p_lvl[ii] > 1) {
              const int nl = p_lvl[ii] >> 1;
              const float fct = (float)nl / (float)p_lvl[ii];
              p_lvl[ii] = nl;
              sum_score[ii] *= fct;
              max_score[ii] -= metal::precise::log2(fct);
              STEEL_PRAGMA_UNROLL
              for (short id = 0; id < TD; id++) {
                thread auto& oi = Oacc.frag_at(0, id);
                STEEL_PRAGMA_UNROLL
                for (short jj = 0; jj < kCols; jj++) {
                  const short loc = ii * kCols + jj;
                  oi[loc] = (int32_t)metal::rint((float)oi[loc] * fct);
                }
              }
            }
          }
        }
      } else {
        Otile.template row_bin_op<LwaMul>(factor);
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < TD; id += 2) {
          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            NAXTile<T, 1, 2> Vt;
            STEEL_PRAGMA_UNROLL
            for (short e = 0; e < kRowsPT; ++e) {
              vec<T, 4> a = vec<T, 4>(T(0)), b = vec<T, 4>(T(0));
              if (lok[ik][e]) {
                const device T* vr =
                    vp + (long)ltok[ik][e] * vtok + id * kU + sn;
                a = *(const device vec<T, 4>*)(vr);
                b = *(const device vec<T, 4>*)(vr + kU);
              }
              STEEL_PRAGMA_UNROLL
              for (short j = 0; j < kCols; ++j) {
                Vt.frag_at(0, 0)[e * kCols + j] = a[j];
                Vt.frag_at(0, 1)[e * kCols + j] = b[j];
              }
            }
            otile_t::NAXFrag_t::mma(
                Otile.frag_at(0, id), Otile.frag_at(0, id + 1),
                Stile.frag_at(0, ik), metal::false_type{}, Vt.frag_at(0, 0),
                Vt.frag_at(0, 1), metal::false_type{});
          }
        }
      }
    }
  }

  if (lwa_pv8) {
    LWA_PV8_FIX();
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < otile_t::kElemsPerTile; ii++) {
      Otile.elems()[ii] = (float)Oacc.elems()[ii] * (1.0f / 255.0f);
    }
  }
#undef LWA_PV8_FIX

  metal::vec<float, kRowsPT> rcp;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    rcp[i] = sum_score[i] > 0.0f ? 1.0f / sum_score[i] : 0.0f;
  }
  Otile.template row_bin_op<LwaMul>(rcp);

  STEEL_PRAGMA_UNROLL
  for (short id = 0; id < TD; ++id) {
    float mu[kCols], vs[kCols];
    STEEL_PRAGMA_UNROLL
    for (short jj = 0; jj < kCols; ++jj) {
      mu[jj] = lwa_pv8 ? vmu[(long)kvh * BD + id * kU + sn + jj] : 0.0f;
      vs[jj] = lwa_pv8 ? vsc[(long)kvh * BD + id * kU + sn + jj] : 1.0f;
    }
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < kRowsPT; ++e) {
      if (!qok[e]) { continue; }
      vec<T, 4> o;
      STEEL_PRAGMA_UNROLL
      for (short jj = 0; jj < kCols; ++jj) {
        float x = Otile.frag_at(0, id)[e * kCols + jj];
        if (lwa_pv8) { x = x * vs[jj] + mu[jj]; }
        o[jj] = static_cast<T>(x);
      }
      *(device vec<T, 4>*)(out + (long)qtok[e] * p.o_tok +
                           (long)h * p.o_head + id * kU + sn) = o;
    }
  }
}

// ---- STAGED: one union a threadgroup ----------------------------------------
//
// The kernel above is bound by its GATHERS, not its matrix work: MEASURED on
// an M5 Max at RelayVSR's 480x270 geometry, the same kernel with its K / V
// loads replaced by constants ran 1.18 ms against 2.12 (decoder, head dim
// 64) and 1.43 against 4.39 (keyframe, 128). Each simdgroup re-reads its own
// union for its 16 rows, (wh + 3)(ww + 3) keys -- 163 a row for the
// decoder's 48 x 48 planes. Here the four simdgroups of an 8 x 8 tile share
// ONE union, (wh + 7)(ww + 7) keys for 64 rows (47 a row): each block of 32
// keys is loaded once into threadgroup memory -- four threads a key, its
// row in 16-byte pieces, with its coordinates and (int8) its scale -- and
// every simdgroup takes its fragments from there. A block none of a
// simdgroup's rows sees is skipped by that simdgroup, so its matrix work
// stays its own union's. Same buffers, params and constants as above.
template <typename T, int BD>
[[kernel, max_total_threads_per_threadgroup(128)]]
void local_window_attn_nax_tg(
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
    constant LwaNaxParams& p [[buffer(10)]],
    const device int8_t* q8  [[buffer(11), function_constant(lwa_qk8)]],
    const device float* qsc  [[buffer(12), function_constant(lwa_qk8)]],
    const device int8_t* k8  [[buffer(13), function_constant(lwa_qk8)]],
    const device float* ksc  [[buffer(14), function_constant(lwa_qk8)]],
    const device uint8_t* v8 [[buffer(15), function_constant(lwa_pv8)]],
    const device float* vsc  [[buffer(16), function_constant(lwa_pv8)]],
    const device float* vmu  [[buffer(17), function_constant(lwa_pv8)]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]],
    uint  lid           [[thread_index_in_threadgroup]],
    uint3 tid           [[threadgroup_position_in_grid]])
{
  constexpr short kU = 16;
  constexpr short TD = BD / kU;
  constexpr int BK = 32;
  constexpr short TK = BK / kU;
  using otile_t = NAXTile<float, 1, TD>;
  using stile_t = NAXTile<float, 1, TK>;
  constexpr short kRowsPT = otile_t::kRowsPerThread;   // 2
  constexpr short kCols = otile_t::kFragThrCols;       // 4
  // Threadgroup rows: T padded by 8 elements, int8 by 16 bytes -- both a
  // multiple of 16 bytes, so a 16-byte piece stays aligned.
  constexpr int LDT = BD + 8;
  constexpr int LD8 = BD + 16;
  constexpr int kPieces = 4;                           // threads a key
  constexpr int kPieceT = BD / kPieces;                // elements a piece
  // K's block then V's, each in its element type or int8 by the
  // constants: one buffer, two views of each half.
  constexpr int kHalf = BK * LDT * (int)sizeof(T);     // >= BK * LD8
  static_assert(BK * LD8 <= kHalf, "the int8 view must fit the T one");
  threadgroup uchar smem[2 * kHalf];
  threadgroup T* Ks = (threadgroup T*)smem;
  threadgroup T* Vs = (threadgroup T*)(smem + kHalf);
  threadgroup int8_t* K8s = (threadgroup int8_t*)smem;
  threadgroup uint8_t* V8s = (threadgroup uint8_t*)(smem + kHalf);
  threadgroup int2 kyx[BK];
  threadgroup float kss[BK];

  const int n = p.H * p.W;
  const int h = (int)tid.y;
  const int kvh = h / (p.Hq / p.Hkv);
  const int gy0 = ((int)tid.x / p.tiles_x) * 8;
  const int gx0 = ((int)tid.x % p.tiles_x) * 8;
  if (gy0 >= p.H || gx0 >= p.W) { return; }    // the whole threadgroup
  const int gy1 = min(gy0 + 8, p.H), gx1 = min(gx0 + 8, p.W);
  // This simdgroup's 4 x 4 block; past the grid it loads, and waits at the
  // barriers, but computes nothing.
  const int ty0 = gy0 + ((int)simd_group_id >> 1) * 4;
  const int tx0 = gx0 + ((int)simd_group_id & 1) * 4;
  const bool active = ty0 < p.H && tx0 < p.W;
  const int ty1 = min(ty0 + 4, p.H), tx1 = min(tx0 + 4, p.W);

  const short2 sc = otile_t::NAXFrag_t::get_coord();
  const short sm = sc.y, sn = sc.x;

  int qy[kRowsPT], qx[kRowsPT], qtok[kRowsPT];
  bool qok[kRowsPT];
  STEEL_PRAGMA_UNROLL
  for (short e = 0; e < kRowsPT; ++e) {
    const int i = sm + 8 * e;
    const int y = ty0 + i / 4, x = tx0 + i % 4;
    qok[e] = active && y < ty1 && x < tx1;
    qy[e] = active ? min(y, ty1 - 1) : gy0;
    qx[e] = active ? min(x, tx1 - 1) : gx0;
    qtok[e] = qy[e] * p.W + qx[e];
  }

  NAXTile<T, 1, TD> Qt;
  NAXTile<int8_t, 1, TD> Q8t;
  float q_row[kRowsPT] = {0.0f, 0.0f};
  STEEL_PRAGMA_UNROLL
  for (short id = 0; id < TD; ++id) {
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < kRowsPT; ++e) {
      if (lwa_qk8) {
        char4 v = char4(0);
        if (qok[e]) {
          v = *(const device char4*)(q8 + ((long)h * n + qtok[e]) * BD +
                                     id * kU + sn);
        }
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kCols; ++j) {
          Q8t.frag_at(0, id)[e * kCols + j] = (int8_t)v[j];
        }
      } else {
        vec<T, 4> v = vec<T, 4>(T(0));
        if (qok[e]) {
          v = *(const device vec<T, 4>*)(q + (long)qtok[e] * p.q_tok +
                                         (long)h * p.q_head + id * kU + sn);
        }
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kCols; ++j) {
          Qt.frag_at(0, id)[e * kCols + j] = v[j];
        }
      }
    }
  }
  if (lwa_qk8) {
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < kRowsPT; ++e) {
      q_row[e] = qok[e] ? qsc[(long)h * n + qtok[e]] : 0.0f;
    }
  }

  // See the kernel above: the lowest float under a finite floor.
  const float kFloor = -1.0e30f;
  const float kMasked = metal::numeric_limits<float>::lowest();
  const float lscale = p.scale * M_LOG2E_F;
  metal::vec<float, kRowsPT> max_score, sum_score;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = kFloor;
    sum_score[i] = 0.0f;
  }
  otile_t Otile;
  Otile.clear();
  NAXTile<int32_t, 1, TD> Oacc;
  int r_part[kRowsPT] = {0, 0};
  int p_lvl[kRowsPT] = {255, 255};
  if (lwa_pv8) { Oacc.clear(); }

#define LWA_PV8_FIX()                                                     \
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
        for (short jj = 0; jj < kCols; jj++) {                            \
          oi[ii * kCols + jj] -= off;                                     \
        }                                                                 \
      }                                                                   \
    }                                                                     \
  }

  // The loader's key and piece.
  const int lkey = (int)lid / kPieces, lpart = (int)lid % kPieces;
  int visited = 0;
  for (int pl = 0; pl < p.planes; ++pl) {
    const device T* kp = pl == 0 ? k0 : pl == 1 ? k1 : pl == 2 ? k2 : k3;
    const device T* vp = pl == 0 ? v0 : pl == 1 ? v1 : pl == 2 ? v2 : v3;
    const long ktok = p.k_tok[pl], vtok = p.v_tok[pl];
    kp += (long)kvh * p.k_head;
    vp += (long)kvh * p.v_head;
    const long pbase = ((long)pl * p.Hkv + kvh) * n;
    const bool sh = p.shifted[pl] != 0;
    const int wh = sh ? min(p.win_h[pl], p.H) : p.win_h[pl];
    const int ww = sh ? min(p.win_w[pl], p.W) : p.win_w[pl];
    // The TILE's union, which every thread walks; this thread's rows'
    // windows.
    int r0, r1, c0, c1;
    int qt[kRowsPT], qb[kRowsPT], ql[kRowsPT], qr[kRowsPT];
    if (sh) {
      r0 = clamp(gy0 - wh / 2, 0, p.H - wh);
      r1 = clamp(gy1 - 1 - wh / 2, 0, p.H - wh) + wh;
      c0 = clamp(gx0 - ww / 2, 0, p.W - ww);
      c1 = clamp(gx1 - 1 - ww / 2, 0, p.W - ww) + ww;
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < kRowsPT; ++e) {
        qt[e] = clamp(qy[e] - wh / 2, 0, p.H - wh);
        ql[e] = clamp(qx[e] - ww / 2, 0, p.W - ww);
        qb[e] = qt[e] + wh;
        qr[e] = ql[e] + ww;
      }
    } else {
      r0 = max(0, gy0 - wh / 2);
      r1 = min(p.H, gy1 - 1 + wh / 2);
      c0 = max(0, gx0 - ww / 2);
      c1 = min(p.W, gx1 - 1 + ww / 2);
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < kRowsPT; ++e) {
        qt[e] = qy[e] - wh / 2;
        qb[e] = qy[e] + wh / 2;
        ql[e] = qx[e] - ww / 2;
        qr[e] = qx[e] + ww / 2;
      }
    }
    const int rw = c1 - c0;
    const int cnt = (r1 - r0) * rw;
    const float inv_rw = 1.0f / (float)rw;

    for (int kb = 0; kb < cnt; kb += BK) {
      // ---- the block into threadgroup memory: four threads a key.
      {
        const int j = kb + lkey;
        const bool in = j < cnt;
        const int2 rc = lwa_rc_(in ? j : 0, rw, inv_rw);
        const int ky = r0 + rc.x, kx = c0 + rc.y;
        const int tok = ky * p.W + kx;
        if (lpart == 0) {
          kyx[lkey] = in ? int2(ky, kx) : int2(-(1 << 28), -(1 << 28));
        }
        if (lwa_qk8) {
          if (lpart == 1) {
            kss[lkey] = in ? ksc[pbase + tok] * lscale : 0.0f;
          }
          constexpr int kB = BD / kPieces;             // bytes a piece
          threadgroup int8_t* dst = K8s + lkey * LD8 + lpart * kB;
          const device int8_t* src = k8 + (pbase + tok) * BD + lpart * kB;
          STEEL_PRAGMA_UNROLL
          for (int c = 0; c < kB; c += 16) {
            *(threadgroup uint4*)(dst + c) =
                in ? *(const device uint4*)(src + c) : uint4(0);
          }
        } else {
          threadgroup T* dst = Ks + lkey * LDT + lpart * kPieceT;
          const device T* src = kp + (long)tok * ktok + lpart * kPieceT;
          STEEL_PRAGMA_UNROLL
          for (int c = 0; c < kPieceT; c += 8) {
            *(threadgroup uint4*)(dst + c) =
                in ? *(const device uint4*)(src + c) : uint4(0);
          }
        }
        if (lwa_pv8) {
          constexpr int kB = BD / kPieces;
          threadgroup uint8_t* dst = V8s + lkey * LD8 + lpart * kB;
          const device uint8_t* src = v8 + (pbase + tok) * BD + lpart * kB;
          STEEL_PRAGMA_UNROLL
          for (int c = 0; c < kB; c += 16) {
            *(threadgroup uint4*)(dst + c) =
                in ? *(const device uint4*)(src + c) : uint4(0);
          }
        } else {
          threadgroup T* dst = Vs + lkey * LDT + lpart * kPieceT;
          const device T* src = vp + (long)tok * vtok + lpart * kPieceT;
          STEEL_PRAGMA_UNROLL
          for (int c = 0; c < kPieceT; c += 8) {
            *(threadgroup uint4*)(dst + c) =
                in ? *(const device uint4*)(src + c) : uint4(0);
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // ---- this simdgroup's rows against it.
      bool vis[TK][kRowsPT][kCols];
      bool any = false;
      STEEL_PRAGMA_UNROLL
      for (short f = 0; f < TK; ++f) {
        STEEL_PRAGMA_UNROLL
        for (short jj = 0; jj < kCols; ++jj) {
          const int2 c = kyx[f * kU + sn + jj];
          STEEL_PRAGMA_UNROLL
          for (short e = 0; e < kRowsPT; ++e) {
            vis[f][e][jj] = c.x >= qt[e] && c.x < qb[e] && c.y >= ql[e] &&
                            c.y < qr[e];
            any = any || vis[f][e][jj];
          }
        }
      }
      if (active && simd_any(any)) {
        ++visited;
        stile_t Stile;
        if (lwa_qk8) {
          NAXTile<int32_t, 1, TK> Si;
          Si.clear();
          STEEL_PRAGMA_UNROLL
          for (short id = 0; id < TD; ++id) {
            NAXTile<int8_t, 2, 1> Kt;
            STEEL_PRAGMA_UNROLL
            for (short f = 0; f < TK; ++f) {
              STEEL_PRAGMA_UNROLL
              for (short e = 0; e < kRowsPT; ++e) {
                const char4 v = *(const threadgroup char4*)(
                    K8s + (f * kU + sm + 8 * e) * LD8 + id * kU + sn);
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < kCols; ++j) {
                  Kt.frag_at(f, 0)[e * kCols + j] = (int8_t)v[j];
                }
              }
            }
            NAXTile<int32_t, 1, TK>::NAXFrag_t::template
                mma<int32_t, int8_t, int8_t>(
                Si.frag_at(0, 0), Si.frag_at(0, 1), Q8t.frag_at(0, id),
                metal::false_type{}, Kt.frag_at(0, 0), Kt.frag_at(1, 0),
                metal::true_type{});
          }
          STEEL_PRAGMA_UNROLL
          for (short f = 0; f < TK; ++f) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < kCols; ++jj) {
              const float kc = kss[f * kU + sn + jj];
              STEEL_PRAGMA_UNROLL
              for (short e = 0; e < kRowsPT; ++e) {
                Stile.frag_at(0, f)[e * kCols + jj] =
                    (float)Si.frag_at(0, f)[e * kCols + jj] * q_row[e] * kc;
              }
            }
          }
        } else {
          Stile.clear();
          STEEL_PRAGMA_UNROLL
          for (short id = 0; id < TD; ++id) {
            NAXTile<T, 2, 1> Kt;
            STEEL_PRAGMA_UNROLL
            for (short f = 0; f < TK; ++f) {
              STEEL_PRAGMA_UNROLL
              for (short e = 0; e < kRowsPT; ++e) {
                const vec<T, 4> v = *(const threadgroup vec<T, 4>*)(
                    Ks + (f * kU + sm + 8 * e) * LDT + id * kU + sn);
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < kCols; ++j) {
                  Kt.frag_at(f, 0)[e * kCols + j] = v[j];
                }
              }
            }
            stile_t::NAXFrag_t::mma(
                Stile.frag_at(0, 0), Stile.frag_at(0, 1), Qt.frag_at(0, id),
                metal::false_type{}, Kt.frag_at(0, 0), Kt.frag_at(1, 0),
                metal::true_type{});
          }
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < stile_t::kElemsPerTile; ++ii) {
            Stile.elems()[ii] *= lscale;
          }
        }
        STEEL_PRAGMA_UNROLL
        for (short f = 0; f < TK; ++f) {
          STEEL_PRAGMA_UNROLL
          for (short e = 0; e < kRowsPT; ++e) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < kCols; ++jj) {
              if (!vis[f][e][jj]) {
                Stile.frag_at(0, f)[e * kCols + jj] = kMasked;
              }
            }
          }
        }

        metal::vec<float, kRowsPT> new_max, factor;
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kRowsPT; ++i) { new_max[i] = max_score[i]; }
        Stile.template row_reduce<LwaMax>(new_max);
        if (lwa_pv8 || lwa_qk8) {
          Stile.template row_bin_op<LwaExpSub>(new_max);
        } else {
          Stile.template row_bin_op<LwaExpSubH>(new_max);
        }
        bool grew = false;
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kRowsPT; ++i) {
          grew = grew ||
                 (new_max[i] > max_score[i] && max_score[i] > kFloor);
          factor[i] = fast::exp2(max_score[i] - new_max[i]);
          max_score[i] = new_max[i];
          sum_score[i] = sum_score[i] * factor[i];
        }
        Stile.template row_reduce<LwaSum>(sum_score);

        if (lwa_pv8) {
          if (simd_any(grew)) {
            LWA_PV8_FIX();
            STEEL_PRAGMA_UNROLL
            for (short ii = 0; ii < kRowsPT; ii++) {
              if (factor[ii] < 1.0f) { p_lvl[ii] = 255; }
            }
            STEEL_PRAGMA_UNROLL
            for (short id = 0; id < TD; id++) {
              thread auto& oi = Oacc.frag_at(0, id);
              STEEL_PRAGMA_UNROLL
              for (short ii = 0; ii < kRowsPT; ii++) {
                STEEL_PRAGMA_UNROLL
                for (short jj = 0; jj < kCols; jj++) {
                  const short loc = ii * kCols + jj;
                  oi[loc] =
                      (int32_t)metal::rint((float)oi[loc] * factor[ii]);
                }
              }
            }
          }
          NAXTile<uint8_t, 1, TK> Pu;
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {
            Pu.elems()[ii] = (uint8_t)metal::clamp(
                metal::rint(Stile.elems()[ii] * 255.0f), 0.0f, 255.0f);
          }
          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            STEEL_PRAGMA_UNROLL
            for (short ii = 0; ii < kRowsPT; ii++) {
              STEEL_PRAGMA_UNROLL
              for (short jj = 0; jj < kCols; jj++) {
                r_part[ii] += (int)Pu.frag_at(0, ik)[ii * kCols + jj];
              }
            }
          }
          STEEL_PRAGMA_UNROLL
          for (short id = 0; id < TD; id += 2) {
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < TK; ik++) {
              NAXTile<uint8_t, 1, 2> Vt;
              STEEL_PRAGMA_UNROLL
              for (short e = 0; e < kRowsPT; ++e) {
                const threadgroup uint8_t* vr =
                    V8s + (ik * kU + sm + 8 * e) * LD8 + id * kU + sn;
                const uchar4 a = *(const threadgroup uchar4*)(vr);
                const uchar4 b = *(const threadgroup uchar4*)(vr + kU);
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < kCols; ++j) {
                  Vt.frag_at(0, 0)[e * kCols + j] = a[j];
                  Vt.frag_at(0, 1)[e * kCols + j] = b[j];
                }
              }
              NAXTile<int32_t, 1, 2>::NAXFrag_t::template
                  mma<int32_t, uint8_t, uint8_t>(
                  Oacc.frag_at(0, id), Oacc.frag_at(0, id + 1),
                  Pu.frag_at(0, ik), metal::false_type{}, Vt.frag_at(0, 0),
                  Vt.frag_at(0, 1), metal::false_type{});
            }
          }
          if ((visited & 63) == 63) {
            LWA_PV8_FIX();
            STEEL_PRAGMA_UNROLL
            for (short ii = 0; ii < kRowsPT; ii++) {
              int big = 0;
              STEEL_PRAGMA_UNROLL
              for (short id = 0; id < TD; id++) {
                STEEL_PRAGMA_UNROLL
                for (short jj = 0; jj < kCols; jj++) {
                  big = metal::max(
                      big, metal::abs(Oacc.frag_at(0, id)[ii * kCols + jj]));
                }
              }
              big = metal::max(big, simd_shuffle_xor(big, ushort(1)));
              big = metal::max(big, simd_shuffle_xor(big, ushort(8)));
              if (big > (1 << 30) && p_lvl[ii] > 1) {
                const int nl = p_lvl[ii] >> 1;
                const float fct = (float)nl / (float)p_lvl[ii];
                p_lvl[ii] = nl;
                sum_score[ii] *= fct;
                max_score[ii] -= metal::precise::log2(fct);
                STEEL_PRAGMA_UNROLL
                for (short id = 0; id < TD; id++) {
                  thread auto& oi = Oacc.frag_at(0, id);
                  STEEL_PRAGMA_UNROLL
                  for (short jj = 0; jj < kCols; jj++) {
                    const short loc = ii * kCols + jj;
                    oi[loc] = (int32_t)metal::rint((float)oi[loc] * fct);
                  }
                }
              }
            }
          }
        } else {
          Otile.template row_bin_op<LwaMul>(factor);
          STEEL_PRAGMA_UNROLL
          for (short id = 0; id < TD; id += 2) {
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < TK; ik++) {
              NAXTile<T, 1, 2> Vt;
              STEEL_PRAGMA_UNROLL
              for (short e = 0; e < kRowsPT; ++e) {
                const threadgroup T* vr =
                    Vs + (ik * kU + sm + 8 * e) * LDT + id * kU + sn;
                const vec<T, 4> a = *(const threadgroup vec<T, 4>*)(vr);
                const vec<T, 4> b = *(const threadgroup vec<T, 4>*)(vr + kU);
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < kCols; ++j) {
                  Vt.frag_at(0, 0)[e * kCols + j] = a[j];
                  Vt.frag_at(0, 1)[e * kCols + j] = b[j];
                }
              }
              otile_t::NAXFrag_t::mma(
                  Otile.frag_at(0, id), Otile.frag_at(0, id + 1),
                  Stile.frag_at(0, ik), metal::false_type{},
                  Vt.frag_at(0, 0), Vt.frag_at(0, 1), metal::false_type{});
            }
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  if (!active) { return; }

  if (lwa_pv8) {
    LWA_PV8_FIX();
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < otile_t::kElemsPerTile; ii++) {
      Otile.elems()[ii] = (float)Oacc.elems()[ii] * (1.0f / 255.0f);
    }
  }
#undef LWA_PV8_FIX

  metal::vec<float, kRowsPT> rcp;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    rcp[i] = sum_score[i] > 0.0f ? 1.0f / sum_score[i] : 0.0f;
  }
  Otile.template row_bin_op<LwaMul>(rcp);

  STEEL_PRAGMA_UNROLL
  for (short id = 0; id < TD; ++id) {
    float mu[kCols], vs[kCols];
    STEEL_PRAGMA_UNROLL
    for (short jj = 0; jj < kCols; ++jj) {
      mu[jj] = lwa_pv8 ? vmu[(long)kvh * BD + id * kU + sn + jj] : 0.0f;
      vs[jj] = lwa_pv8 ? vsc[(long)kvh * BD + id * kU + sn + jj] : 1.0f;
    }
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < kRowsPT; ++e) {
      if (!qok[e]) { continue; }
      vec<T, 4> o;
      STEEL_PRAGMA_UNROLL
      for (short jj = 0; jj < kCols; ++jj) {
        float x = Otile.frag_at(0, id)[e * kCols + jj];
        if (lwa_pv8) { x = x * vs[jj] + mu[jj]; }
        o[jj] = static_cast<T>(x);
      }
      *(device vec<T, 4>*)(out + (long)qtok[e] * p.o_tok +
                           (long)h * p.o_head + id * kU + sn) = o;
    }
  }
}

template [[host_name("local_window_attn_nax_tg_bd64_f16")]] [[kernel]]
decltype(local_window_attn_nax_tg<half, 64>)
    local_window_attn_nax_tg<half, 64>;
template [[host_name("local_window_attn_nax_tg_bd64_bf16")]] [[kernel]]
decltype(local_window_attn_nax_tg<bfloat, 64>)
    local_window_attn_nax_tg<bfloat, 64>;
template [[host_name("local_window_attn_nax_tg_bd128_f16")]] [[kernel]]
decltype(local_window_attn_nax_tg<half, 128>)
    local_window_attn_nax_tg<half, 128>;
template [[host_name("local_window_attn_nax_tg_bd128_bf16")]] [[kernel]]
decltype(local_window_attn_nax_tg<bfloat, 128>)
    local_window_attn_nax_tg<bfloat, 128>;

template [[host_name("local_window_attn_nax_bd64_f16")]] [[kernel]]
decltype(local_window_attn_nax<half, 64>) local_window_attn_nax<half, 64>;
template [[host_name("local_window_attn_nax_bd64_bf16")]] [[kernel]]
decltype(local_window_attn_nax<bfloat, 64>)
    local_window_attn_nax<bfloat, 64>;
template [[host_name("local_window_attn_nax_bd128_f16")]] [[kernel]]
decltype(local_window_attn_nax<half, 128>) local_window_attn_nax<half, 128>;
template [[host_name("local_window_attn_nax_bd128_bf16")]] [[kernel]]
decltype(local_window_attn_nax<bfloat, 128>)
    local_window_attn_nax<bfloat, 128>;

#else
// Tensor ops unavailable for this target: stubs so the metallib builds.
// The loader binds these only where the GPU has matrix cores.
kernel void local_window_attn_nax_bd64_f16(device half* o [[buffer(9)]],
                                           uint t [[thread_position_in_grid]])
{ if (t == 0) { o[0] = (half)0; } }
kernel void local_window_attn_nax_bd64_bf16(
    device bfloat* o [[buffer(9)]], uint t [[thread_position_in_grid]])
{ if (t == 0) { o[0] = (bfloat)0; } }
kernel void local_window_attn_nax_bd128_f16(
    device half* o [[buffer(9)]], uint t [[thread_position_in_grid]])
{ if (t == 0) { o[0] = (half)0; } }
kernel void local_window_attn_nax_bd128_bf16(
    device bfloat* o [[buffer(9)]], uint t [[thread_position_in_grid]])
{ if (t == 0) { o[0] = (bfloat)0; } }
kernel void local_window_attn_nax_tg_bd64_f16(
    device half* o [[buffer(9)]], uint t [[thread_position_in_grid]])
{ if (t == 0) { o[0] = (half)0; } }
kernel void local_window_attn_nax_tg_bd64_bf16(
    device bfloat* o [[buffer(9)]], uint t [[thread_position_in_grid]])
{ if (t == 0) { o[0] = (bfloat)0; } }
kernel void local_window_attn_nax_tg_bd128_f16(
    device half* o [[buffer(9)]], uint t [[thread_position_in_grid]])
{ if (t == 0) { o[0] = (half)0; } }
kernel void local_window_attn_nax_tg_bd128_bf16(
    device bfloat* o [[buffer(9)]], uint t [[thread_position_in_grid]])
{ if (t == 0) { o[0] = (bfloat)0; } }
#endif

// ---- THE INT8 PROLOGUE ---------------------------------------------------
//
// Plain MSL, built everywhere. What the attention above reads with its
// int8 constants on: Q per token, K per token over a centre shared by
// every plane, V per channel over a centre and a step shared by every
// plane (each centre the channel's midrange).

// The planes, as the attention binds them: up to four sources, each with
// its own token stride, one head stride, n tokens a plane.
struct LwaPlanes {
  int planes;
  int n;
  int heads;          // the KV heads (Hq for Q)
  int D;
  int head_stride;
  int tok[4];
};

// Q (or every plane of K over its centre) per TOKEN: one simdgroup a
// (token, head), the row's absmax over D, int8 at amax / 127.
//   0..3:src planes  4:centre (float [heads, D]; read when `sub` is set)
//   5:q (int8 [planes, heads, n, D])  6:scale (float [planes, heads, n])
//   7:LwaPlanes  8:sub (int)
//   grid threadgroups (ceil(n / 4), heads, planes); tg (32, 4, 1)
template <typename T>
[[kernel]] void lwa_quant_token(
    const device T* s0 [[buffer(0)]],
    const device T* s1 [[buffer(1)]],
    const device T* s2 [[buffer(2)]],
    const device T* s3 [[buffer(3)]],
    const device float* mean [[buffer(4)]],
    device char*  q8    [[buffer(5)]],
    device float* scale [[buffer(6)]],
    constant LwaPlanes& g [[buffer(7)]],
    constant int& sub [[buffer(8)]],
    uint3 tid  [[threadgroup_position_in_grid]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
  const int tok = (int)tid.x * 4 + (int)sg;
  const int h = (int)tid.y, pl = (int)tid.z;
  if (tok >= g.n) { return; }
  const device T* s = pl == 0 ? s0 : pl == 1 ? s1 : pl == 2 ? s2 : s3;
  s += (long)tok * g.tok[pl] + (long)h * g.head_stride;
  const int per = g.D / 32;            // 2 or 4
  float v[4];
  float am = 0.0f;
  for (int i = 0; i < per; ++i) {
    const int d = (int)lane * per + i;
    v[i] = (float)s[d];
    if (sub != 0) { v[i] -= mean[h * g.D + d]; }
    am = max(am, fabs(v[i]));
  }
  am = simd_max(am);
  const float inv = am > 0.0f ? 127.0f / am : 0.0f;
  const long row = ((long)pl * g.heads + h) * g.n + tok;
  if (lane == 0) { scale[row] = am / 127.0f; }
  for (int i = 0; i < per; ++i) {
    q8[row * g.D + (int)lane * per + i] =
        (char)clamp(rint(v[i] * inv), -127.0f, 127.0f);
  }
}

// Each channel's RANGE over EVERY plane's tokens at once -- min and max,
// one pass, in chunks. The centre a channel is taken around need not be
// its mean: any one vector shared by every key a row sees leaves K's
// softmax exact, and for V the epilogue adds back whatever was taken
// out. The midrange spends int8's range symmetrically, in one pass.
//   0..3:src planes  4:partial (float2 [chunks, heads * D], (min, max))
//   5:LwaPlanes
//   grid threadgroups (ceil(heads * D / 32), chunks); tg (32, 8, 1)
template <typename T>
[[kernel]] void lwa_channel_range(
    const device T* s0 [[buffer(0)]],
    const device T* s1 [[buffer(1)]],
    const device T* s2 [[buffer(2)]],
    const device T* s3 [[buffer(3)]],
    device float2* partial [[buffer(4)]],
    constant LwaPlanes& g [[buffer(5)]],
    uint2 tid [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tgs [[threadgroups_per_grid]])
{
  threadgroup float2 red[8][32];
  const int C = g.heads * g.D;
  const int c = (int)tid.x * 32 + (int)lid.x;
  const int chunks = (int)tgs.y;
  const int total = g.planes * g.n;
  const int per = (total + chunks - 1) / chunks;
  const int t0 = (int)tid.y * per, t1 = min(total, t0 + per);
  float lo = INFINITY, hi = -INFINITY;
  if (c < C) {
    const int h = c / g.D, d = c % g.D;
    for (int t = t0 + (int)lid.y; t < t1; t += 8) {
      const int pl = t / g.n, tok = t % g.n;
      const device T* s = pl == 0 ? s0 : pl == 1 ? s1 : pl == 2 ? s2 : s3;
      const float x =
          (float)s[(long)tok * g.tok[pl] + (long)h * g.head_stride + d];
      lo = min(lo, x);
      hi = max(hi, x);
    }
  }
  red[lid.y][lid.x] = float2(lo, hi);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid.y == 0 && c < C) {
    float2 r = red[0][lid.x];
    for (int i = 1; i < 8; ++i) {
      r = float2(min(r.x, red[i][lid.x].x), max(r.y, red[i][lid.x].y));
    }
    partial[(long)tid.y * C + c] = r;
  }
}

// ...the chunks into each channel's centre (its midrange) and int8 step
// (half its range / 127).
//   0:partial 1:centre 2:step (float [C]) 3:C 4:chunks
//   grid (C) threads
kernel void lwa_channel_reduce(
    const device float2* partial [[buffer(0)]],
    device float*        centre  [[buffer(1)]],
    device float*        step    [[buffer(2)]],
    constant int&        C       [[buffer(3)]],
    constant int&        chunks  [[buffer(4)]],
    uint c [[thread_position_in_grid]])
{
  if ((int)c >= C) { return; }
  float lo = INFINITY, hi = -INFINITY;
  for (int i = 0; i < chunks; ++i) {
    const float2 x = partial[(long)i * C + c];
    lo = min(lo, x.x);
    hi = max(hi, x.y);
  }
  centre[c] = 0.5f * (lo + hi);
  step[c] = 0.5f * (hi - lo) / 127.0f;
}

// V for the int8 P*V: (v - centre) / step, as int8 plus 128.
//   0..3:src planes  4:centre  5:step (float [heads * D])
//   6:v8 (uint8 [planes, heads, n, D])  7:LwaPlanes
//   grid threadgroups (ceil(n / 8), heads, planes); tg (D, 1, 1)
template <typename T>
[[kernel]] void lwa_quant_v(
    const device T* s0 [[buffer(0)]],
    const device T* s1 [[buffer(1)]],
    const device T* s2 [[buffer(2)]],
    const device T* s3 [[buffer(3)]],
    const device float* mean [[buffer(4)]],
    const device float* step [[buffer(5)]],
    device uchar* v8 [[buffer(6)]],
    constant LwaPlanes& g [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint  d   [[thread_index_in_threadgroup]])
{
  if ((int)d >= g.D) { return; }
  const int h = (int)tid.y, pl = (int)tid.z;
  const device T* s = pl == 0 ? s0 : pl == 1 ? s1 : pl == 2 ? s2 : s3;
  const int c = h * g.D + (int)d;
  const float mu = mean[c];
  const float st = step[c];
  const float inv = st > 0.0f ? 1.0f / st : 0.0f;
  const int t0 = (int)tid.x * 8, t1 = min(g.n, t0 + 8);
  for (int tok = t0; tok < t1; ++tok) {
    const float x =
        (float)s[(long)tok * g.tok[pl] + (long)h * g.head_stride + d] - mu;
    const int qi = (int)clamp(rint(x * inv), -127.0f, 127.0f);
    v8[(((long)pl * g.heads + h) * g.n + tok) * g.D + d] = (uchar)(qi + 128);
  }
}

template [[host_name("lwa_quant_token_f16")]] [[kernel]]
decltype(lwa_quant_token<half>) lwa_quant_token<half>;
template [[host_name("lwa_quant_token_bf16")]] [[kernel]]
decltype(lwa_quant_token<bfloat>) lwa_quant_token<bfloat>;
template [[host_name("lwa_channel_range_f16")]] [[kernel]]
decltype(lwa_channel_range<half>) lwa_channel_range<half>;
template [[host_name("lwa_channel_range_bf16")]] [[kernel]]
decltype(lwa_channel_range<bfloat>) lwa_channel_range<bfloat>;
template [[host_name("lwa_quant_v_f16")]] [[kernel]]
decltype(lwa_quant_v<half>) lwa_quant_v<half>;
template [[host_name("lwa_quant_v_bf16")]] [[kernel]]
decltype(lwa_quant_v<bfloat>) lwa_quant_v<bfloat>;

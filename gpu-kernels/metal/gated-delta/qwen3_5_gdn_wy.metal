// qwen3_5_gdn_wy.metal -- Qwen3.5 gated-DeltaNet CHUNKWISE (WY/UT)
// representation kernels, ported from the MLX fast::metal_kernel trio in
// qwen3-5-model-exec.cc (qwen3_5_gdn_chunk_{tri_solve,build_L_solve,
// output_state}, themselves a port of FLA's gated_delta_rule chunkwise
// path). This is the alternative to the recurrent step kernel
// (qwen3_5_gated_delta_step); it processes a whole BT-chunk at once via a
// triangular solve + two batched products. Kept available but NOT the
// default (the recurrent kernel wins on M2 Max; this is preserved for an
// M5 matrix-core retest, mirroring VPIPE_GDN_WY in the MLX path).
//
// Compile-time buckets fixed to the Qwen3.5 GDN config:
//   BT_MAX 64, BLOCK_W 32, BN_dv 4, Dk = Dv = 128 (per_lane = Dk/32 = 4).
// bt (actual chunk height <= BT_MAX) and rhs_w are runtime scalars.

#include <metal_stdlib>
using namespace metal;

#define WY_BT_MAX  64
#define WY_BLOCK_W 32
#define WY_BN_DV   4
#define WY_DK      128
#define WY_DV      128
#define WY_PER_LANE (WY_DK / 32)

// Forward substitution (I + L_lower) X = B, L strictly lower (diag 1):
//   x[i,c] = b[i,c] - sum_{j<i} A[i,j] * x[j,c]
// One simdgroup (BLOCK_W lanes); each lane owns one RHS column. fp32
// accumulation regardless of the half storage. Batched over
// (col_block, batch).  0:A[batch,bt,bt] 1:B[batch,bt,rhs_w]
// 2:X(out) 3:bt 4:rhs_w
kernel void qwen3_5_gdn_chunk_tri_solve_f16(
    const device half* A_mat [[buffer(0)]],
    const device half* B     [[buffer(1)]],
    device half*       X     [[buffer(2)]],
    constant int&      bt    [[buffer(3)]],
    constant int&      rhs_w [[buffer(4)]],
    uint3 tpig [[thread_position_in_grid]],
    uint3 tpit [[thread_position_in_threadgroup]])
{
  const int batch_idx = (int)tpig.z;
  const int col_block = (int)tpig.y;
  const int lane = (int)tpit.x;
  const int col = col_block * WY_BLOCK_W + lane;
  const bool col_valid = (col < rhs_w);
  const int btl = bt;

  threadgroup float X_tile[WY_BT_MAX * WY_BLOCK_W];
  const device half* A_b = A_mat + (uint)batch_idx * btl * btl;
  const device half* B_b = B + (uint)batch_idx * btl * rhs_w;
  device half* X_b = X + (uint)batch_idx * btl * rhs_w;

  for (int i = 0; i < btl; ++i) {
    X_tile[i * WY_BLOCK_W + lane] =
        col_valid ? (float)B_b[i * rhs_w + col] : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int i = 0; i < btl; ++i) {
    float acc = X_tile[i * WY_BLOCK_W + lane];
    for (int j = 0; j < i; ++j) {
      acc -= (float)A_b[i * btl + j] * X_tile[j * WY_BLOCK_W + lane];
    }
    X_tile[i * WY_BLOCK_W + lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (col_valid) {
    for (int i = 0; i < btl; ++i) {
      X_b[i * rhs_w + col] = (half)X_tile[i * WY_BLOCK_W + lane];
    }
  }
}

// Fused build-L + forward-sub: L_ij = beta[i]*(G[i]/G[j])*kkT[i,j] built
// on the fly (no A_mat materialization). All f32.
//   0:kkT[batch,bt,bt] 1:G_cum[batch,bt] 2:beta[batch,bt]
//   3:B[batch,bt,rhs_w] 4:X(out) 5:bt 6:rhs_w
kernel void qwen3_5_gdn_chunk_build_L_solve_f32(
    const device float* kkT   [[buffer(0)]],
    const device float* G_cum [[buffer(1)]],
    const device float* beta  [[buffer(2)]],
    const device float* B     [[buffer(3)]],
    device float*       X     [[buffer(4)]],
    constant int&       bt    [[buffer(5)]],
    constant int&       rhs_w [[buffer(6)]],
    uint3 tpig [[thread_position_in_grid]],
    uint3 tpit [[thread_position_in_threadgroup]])
{
  const int batch_idx = (int)tpig.z;
  const int col_block = (int)tpig.y;
  const int lane = (int)tpit.x;
  const int col = col_block * WY_BLOCK_W + lane;
  const bool col_valid = (col < rhs_w);
  const int btl = bt;

  threadgroup float X_tile[WY_BT_MAX * WY_BLOCK_W];
  const device float* kkT_b = kkT + (uint)batch_idx * btl * btl;
  const device float* G_b = G_cum + (uint)batch_idx * btl;
  const device float* beta_b = beta + (uint)batch_idx * btl;
  const device float* B_b = B + (uint)batch_idx * btl * rhs_w;
  device float* X_b = X + (uint)batch_idx * btl * rhs_w;

  for (int i = 0; i < btl; ++i) {
    X_tile[i * WY_BLOCK_W + lane] =
        col_valid ? B_b[i * rhs_w + col] : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int i = 0; i < btl; ++i) {
    const float beta_i = beta_b[i];
    const float G_i = G_b[i];
    float acc = X_tile[i * WY_BLOCK_W + lane];
    for (int j = 0; j < i; ++j) {
      const float L_ij = beta_i * (G_i / G_b[j]) * kkT_b[i * btl + j];
      acc -= L_ij * X_tile[j * WY_BLOCK_W + lane];
    }
    X_tile[i * WY_BLOCK_W + lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (col_valid) {
    for (int i = 0; i < btl; ++i) {
      X_b[i * rhs_w + col] = X_tile[i * WY_BLOCK_W + lane];
    }
  }
}

// Per-chunk output + state update. One tg per (head, dv_block);
// 32 lanes x BN_dv simdgroups, each thread owns a (dv_row, dk_slots)
// state tile in registers. delta = u - w@state^T; y = G*(q@state^T) +
// tril((G_i/G_j)*qkT)@delta; state' = G_last*state + delta^T*(G_last/G*k).
//   0:u[Hv,bt,Dv] f32  1:w[Hv,bt,Dk] f32  2:q[Hv,bt,Dk] half
//   3:k[Hv,bt,Dk] half 4:qkT[Hv,bt,bt] half 5:G[Hv,bt] f32
//   6:state_in[Hv,Dv,Dk] f32  7:y(out)[Hv,bt,Dv] half
//   8:state_out(out)[Hv,Dv,Dk] f32  9:bt
kernel void qwen3_5_gdn_chunk_output_state_f16(
    const device float* u        [[buffer(0)]],
    const device float* w        [[buffer(1)]],
    const device half*  q_p      [[buffer(2)]],
    const device half*  k_p      [[buffer(3)]],
    const device half*  qkT      [[buffer(4)]],
    const device float* G_p      [[buffer(5)]],
    const device float* state_in [[buffer(6)]],
    device half*        y_chunk  [[buffer(7)]],
    device float*       state_out[[buffer(8)]],
    constant int&       bt       [[buffer(9)]],
    uint3 tpig [[thread_position_in_grid]],
    uint3 tpit [[thread_position_in_threadgroup]])
{
  const int h = (int)tpig.z;
  const int my_dv = (int)tpig.y;
  const int lane = (int)tpit.x;
  const int sg_dv = (int)tpit.y;
  const int btl = bt;

  const device float* u_h = u + (uint)h * btl * WY_DV;
  const device float* w_h = w + (uint)h * btl * WY_DK;
  const device half* q_h = q_p + (uint)h * btl * WY_DK;
  const device half* k_h = k_p + (uint)h * btl * WY_DK;
  const device half* qkT_h = qkT + (uint)h * btl * btl;
  const device float* G_h = G_p + (uint)h * btl;
  const device float* state_in_h = state_in + (uint)h * WY_DV * WY_DK;
  device float* state_out_h = state_out + (uint)h * WY_DV * WY_DK;
  device half* y_h = y_chunk + (uint)h * btl * WY_DV;

  float state_reg[WY_PER_LANE];
  for (int p = 0; p < WY_PER_LANE; ++p) {
    state_reg[p] = state_in_h[my_dv * WY_DK + lane * WY_PER_LANE + p];
  }

  threadgroup float delta_tile[WY_BT_MAX * WY_BN_DV];
  threadgroup float G_tile[WY_BT_MAX];
  const int tid = lane + sg_dv * 32;
  const int total_threads = 32 * WY_BN_DV;
  for (int idx = tid; idx < btl; idx += total_threads) {
    G_tile[idx] = G_h[idx];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Phase 1: delta[i, my_dv] = u[i] - sum_k w[i,k]*state[my_dv,k].
  for (int i = 0; i < btl; ++i) {
    float ws_partial = 0.0f;
    for (int p = 0; p < WY_PER_LANE; ++p) {
      ws_partial += w_h[i * WY_DK + lane * WY_PER_LANE + p] * state_reg[p];
    }
    const float ws_dot = simd_sum(ws_partial);
    if (lane == 0) {
      delta_tile[i * WY_BN_DV + sg_dv] = u_h[i * WY_DV + my_dv] - ws_dot;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Phase 2: y[i] = G[i]*(q[i].state) + sum_{j<=i}(G[i]/G[j])qkT[i,j]delta[j].
  for (int i = 0; i < btl; ++i) {
    float qs_partial = 0.0f;
    for (int p = 0; p < WY_PER_LANE; ++p) {
      qs_partial += (float)q_h[i * WY_DK + lane * WY_PER_LANE + p] * state_reg[p];
    }
    const float qs_dot = simd_sum(qs_partial);
    const float G_i = G_tile[i];
    const float y_inter = G_i * qs_dot;
    float intra_partial = 0.0f;
    for (int j = lane; j <= i; j += 32) {
      const float P_ij = (G_i / G_tile[j]) * (float)qkT_h[i * btl + j];
      intra_partial += P_ij * delta_tile[j * WY_BN_DV + sg_dv];
    }
    const float y_intra = simd_sum(intra_partial);
    if (lane == 0) {
      y_h[i * WY_DV + my_dv] = (half)(y_inter + y_intra);
    }
  }

  // Phase 3: state[my_dv, dk] = G_last*state + sum_i delta[i]*(G_last/G[i])*k[i,dk].
  const float G_last = G_tile[btl - 1];
  for (int p = 0; p < WY_PER_LANE; ++p) { state_reg[p] *= G_last; }
  for (int i = 0; i < btl; ++i) {
    const float sc = G_last / G_tile[i];
    const float d_i = delta_tile[i * WY_BN_DV + sg_dv];
    for (int p = 0; p < WY_PER_LANE; ++p) {
      state_reg[p] += d_i * sc * (float)k_h[i * WY_DK + lane * WY_PER_LANE + p];
    }
  }
  for (int p = 0; p < WY_PER_LANE; ++p) {
    state_out_h[my_dv * WY_DK + lane * WY_PER_LANE + p] = state_reg[p];
  }
}

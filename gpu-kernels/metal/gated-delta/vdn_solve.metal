// vdn_solve.metal -- VDN-H3's per-frame SPD solve.
//
// The linear branch's delta rule is a SOLVE over a sum,
//
//     S_out = (S_in Diag(alpha) + B) (I + A)^-1,
//     A = sum_s beta_s k_s k_s^T   [d, d], symmetric PSD,
//
// and A is not small: the branch L2-normalises its keys, so
// trace(A) = sum_s beta_s exactly -- at production geometry a trace in
// the hundreds over d = 128. The first-order form (I - A) is a different
// operator there, and the ordered-product (WY) form is the same
// contraction over a TRIANGULAR mask of the same S x S matrix, which is
// precisely why that one factorises over chunks and this one does not.
// So: an actual batched Cholesky.
//
// WHY IT IS PANELLED. A 128x128 fp32 matrix is 64 KB and this GPU's
// maxThreadgroupMemoryLength is 32 KB (measured, M4 Pro); even the packed
// lower triangle is exactly 32 KB, leaving nothing for anything else. A
// one-threadgroup-holds-the-matrix kernel is therefore not available at
// the head dim that matters, which is the same wall OpenVDN's own
// single-CTA Triton attempt hit before they fell back to cuSOLVER. The
// left-looking block column below holds a [D_MAX, NB] panel (16 KB) and
// one [NB, NB] diagonal block (4 KB) and streams the rest from device.
//
// fp32 THROUGHOUT, deliberately. A is formed as (k*beta)^T k, so its
// (i,j) and (j,i) entries multiply differently-rounded operands and the
// matrix is only symmetric to the working precision. In bf16 that
// asymmetry is enough to push the smallest eigenvalue of I + A below the
// 1 the maths guarantees, and a Cholesky then factorises an indefinite
// matrix -- a NaN many layers from where it was caused.

#include <metal_stdlib>
using namespace metal;

#define VDN_NB     32          // block column width
#define VDN_D_MAX  128         // largest head dim this kernel serves
#define VDN_THREADS 128

// Batched Cholesky of M = I + A, lower factor.
//
// One threadgroup per (frame, head). Reads only the LOWER triangle of A,
// which is what torch.linalg.cholesky does and therefore what the
// reference's numbers come from.
//
//   0:A [batch, d, d]  1:L out [batch, d, d]  2:d  3:fail (atomic flag)
kernel void vdn_cholesky_f32(
    const device float* A     [[buffer(0)]],
    device float*       L     [[buffer(1)]],
    constant int&       d     [[buffer(2)]],
    device atomic_uint* fail  [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpit [[thread_position_in_threadgroup]])
{
  // [row, col-in-block]. The DIAGONAL block is factored in place in rows
  // c0..c0+nb-1 of this same array rather than in a copy: a separate
  // [NB, NB] tile plus `lrow` below would put the kernel at exactly the
  // 32 KB limit, and "exactly the limit" is a launch failure waiting for
  // a device with a byte of overhead.
  threadgroup float panel[VDN_D_MAX * VDN_NB];                   // 16 KB
  // The already-factored part of the ROW block, staged once per panel.
  // Without it the left-looking update below reads BOTH of its operands
  // from device on every multiply-add -- two loads per flop, which is
  // what held the first version of this kernel to ~53 GFLOP/s.
  threadgroup float lrow[VDN_NB * (VDN_D_MAX - VDN_NB)];         // 12 KB
  threadgroup uint  bad;

  const uint batch = tgid.x;
  const int  tid   = (int)tpit.x;
  const device float* Ab = A + (uint)batch * d * d;
  device float* Lb = L + (uint)batch * d * d;

  if (tid == 0) { bad = 0u; }
  for (int i = tid; i < d * d; i += VDN_THREADS) { Lb[i] = 0.0f; }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

  for (int c0 = 0; c0 < d; c0 += VDN_NB) {
    const int nb = min(VDN_NB, d - c0);

    // --- load this block column, rows c0..d-1, and add I on the diagonal
    for (int r = c0 + tid; r < d; r += VDN_THREADS) {
      for (int c = 0; c < nb; ++c) {
        const int col = c0 + c;
        float v = (r >= col) ? Ab[(uint)r * d + col] : 0.0f;
        if (r == col) { v += 1.0f; }
        panel[(uint)r * VDN_NB + c] = v;
      }
    }
    // --- stage L[c0 .. c0+nb-1][0 .. c0-1], the right-hand operand of
    //     every update below
    for (int i = tid; i < nb * c0; i += VDN_THREADS) {
      lrow[i] = Lb[(uint)(c0 + i / c0) * d + (i % c0)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- left-looking update, one device load per (row, k) instead of
    //     two per multiply-add
    for (int r = c0 + tid; r < d; r += VDN_THREADS) {
      float acc[VDN_NB];
      for (int c = 0; c < VDN_NB; ++c) {
        acc[c] = (c < nb) ? panel[(uint)r * VDN_NB + c] : 0.0f;
      }
      for (int k = 0; k < c0; ++k) {
        const float lrk = Lb[(uint)r * d + k];
        for (int c = 0; c < VDN_NB; ++c) {
          acc[c] -= lrk * lrow[(uint)c * c0 + k];
        }
      }
      for (int c = 0; c < nb; ++c) { panel[(uint)r * VDN_NB + c] = acc[c]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- factor the nb x nb diagonal block, in place in `panel`
    for (int c = 0; c < nb; ++c) {
      if (tid == 0) {
        float sum = panel[(uint)(c0 + c) * VDN_NB + c];
        for (int k = 0; k < c; ++k) {
          const float v = panel[(uint)(c0 + c) * VDN_NB + k];
          sum -= v * v;
        }
        // I + A has eigenvalues >= 1 in exact arithmetic, so this is a
        // statement about the INPUT's precision, not about this kernel.
        if (!(sum > 0.0f)) { bad = 1u; sum = 1.0f; }
        panel[(uint)(c0 + c) * VDN_NB + c] = sqrt(sum);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const float dcc = panel[(uint)(c0 + c) * VDN_NB + c];
      // Every row below c within the block AND every row below the
      // block: one loop, so the trailing panel needs no second pass.
      for (int r = c0 + c + 1 + tid; r < d; r += VDN_THREADS) {
        float sum = panel[(uint)r * VDN_NB + c];
        for (int k = 0; k < c; ++k) {
          sum -= panel[(uint)r * VDN_NB + k]
                 * panel[(uint)(c0 + c) * VDN_NB + k];
        }
        panel[(uint)r * VDN_NB + c] = sum / dcc;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --- publish this block column
    for (int r = c0 + tid; r < d; r += VDN_THREADS) {
      for (int c = 0; c < nb; ++c) {
        if (r >= c0 + c) {
          Lb[(uint)r * d + (c0 + c)] = panel[(uint)r * VDN_NB + c];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  }

  if (tid == 0 && bad != 0u) {
    atomic_fetch_add_explicit(fail, 1u, memory_order_relaxed);
  }
}

// Batched solve of X M = C given M's lower factor L, i.e. M X^T = C^T.
//
// Row i of C and row i of X are contiguous, so one lane owning one ROW
// of C is one system: forward substitution against L, then back
// substitution against L^T. The RHS rows live in threadgroup memory
// (a [d, BLOCK_W] tile, 16 KB at d = 128) exactly as the GDN chunk
// solver does; L is streamed from device and hits in cache.
//
//   0:L [batch,d,d]  1:C [batch,rows,d]  2:X out [batch,rows,d]
//   3:d  4:rows
#define VDN_BLOCK_W 32

kernel void vdn_chol_solve_f32(
    const device float* L    [[buffer(0)]],
    const device float* C    [[buffer(1)]],
    device float*       X    [[buffer(2)]],
    constant int&       d    [[buffer(3)]],
    constant int&       rows [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpit [[thread_position_in_threadgroup]])
{
  threadgroup float tile[VDN_D_MAX * VDN_BLOCK_W];   // [d, lane]

  const uint batch = tgid.y;
  const int  lane  = (int)tpit.x;
  const int  row   = (int)tgid.x * VDN_BLOCK_W + lane;
  const bool live  = row < rows;

  const device float* Lb = L + (uint)batch * d * d;
  const device float* Cb = C + (uint)batch * rows * d;
  device float* Xb = X + (uint)batch * rows * d;

  for (int i = 0; i < d; ++i) {
    tile[(uint)i * VDN_BLOCK_W + lane] = live ? Cb[(uint)row * d + i] : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int i = 0; i < d; ++i) {              // L y = c
    float acc = tile[(uint)i * VDN_BLOCK_W + lane];
    for (int k = 0; k < i; ++k) {
      acc -= Lb[(uint)i * d + k] * tile[(uint)k * VDN_BLOCK_W + lane];
    }
    tile[(uint)i * VDN_BLOCK_W + lane] = acc / Lb[(uint)i * d + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (int i = d - 1; i >= 0; --i) {         // L^T x = y
    float acc = tile[(uint)i * VDN_BLOCK_W + lane];
    for (int k = i + 1; k < d; ++k) {
      acc -= Lb[(uint)k * d + i] * tile[(uint)k * VDN_BLOCK_W + lane];
    }
    tile[(uint)i * VDN_BLOCK_W + lane] = acc / Lb[(uint)i * d + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (live) {
    for (int i = 0; i < d; ++i) {
      Xb[(uint)row * d + i] = tile[(uint)i * VDN_BLOCK_W + lane];
    }
  }
}

// ---------------------------------------------------------------------
// WHY THE EXPLICIT INVERSE, and not two triangular solves per scan step.
//
// MEASURED on the M4 Pro at the production batch (F*H = 5712, d = 128):
// the Cholesky above costs 75.9 ms, and a 128-RHS forward+back
// substitution on top of it costs another 259 ms -- 16.7 s per 50-block
// denoise step, dominated by the solve rather than the factorisation.
// That is the same wall OpenVDN documents ("a batched trsm at 128x128
// runs an order of magnitude below a batched GEMM at the same shape"),
// and their answer is the right one: invert the FACTOR once, then do
// everything else with products.
//
// A triangular inverse blocks into GEMMs almost entirely -- one 32x32
// substitution per diagonal block and matrix products for the rest --
// so the serial part shrinks from 128 dependent rows to 32, four times
// over, and the rest runs at GEMM throughput.

#define VDN_TI_THREADS 128

// L^-1 for a lower-triangular L, blocked. One threadgroup per
// (matrix, block column): the columns of the inverse are INDEPENDENT,
// which is what lets this be wide instead of one long dependency chain.
//
//   0:L [batch,d,d]  1:X out [batch,d,d]  2:d
kernel void vdn_trinv_f32(
    const device float* L [[buffer(0)]],
    device float*       X [[buffer(1)]],
    constant int&       d [[buffer(2)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpit [[thread_position_in_threadgroup]])
{
  threadgroup float xcol[VDN_D_MAX * VDN_NB];   // [row, col-in-block]
  threadgroup float ltile[VDN_NB * VDN_NB];
  threadgroup float acc[VDN_NB * VDN_NB];

  const int tid = (int)tpit.x;
  const int j   = (int)tgid.x;                  // block column
  const uint batch = tgid.y;
  const int c0 = j * VDN_NB;
  if (c0 >= d) { return; }
  const int nb = min(VDN_NB, d - c0);

  const device float* Lb = L + (uint)batch * d * d;
  device float* Xb = X + (uint)batch * d * d;

  // Rows above the diagonal block are zero: L^-1 is lower triangular.
  for (int r = tid; r < c0; r += VDN_TI_THREADS) {
    for (int c = 0; c < nb; ++c) { Xb[(uint)r * d + (c0 + c)] = 0.0f; }
  }
  for (int i = tid; i < VDN_D_MAX * VDN_NB; i += VDN_TI_THREADS) {
    xcol[i] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // --- the diagonal block: solve L_jj Y = I by forward substitution
  for (int i = tid; i < nb * nb; i += VDN_TI_THREADS) {
    ltile[i] = Lb[(uint)(c0 + i / nb) * d + (c0 + (i % nb))];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = 0; i < nb; ++i) {
    if (tid < nb) {
      float s = (i == tid) ? 1.0f : 0.0f;
      for (int k = 0; k < i; ++k) {
        s -= ltile[(uint)i * VDN_NB + k] * xcol[(uint)(c0 + k) * VDN_NB + tid];
      }
      xcol[(uint)(c0 + i) * VDN_NB + tid] = s / ltile[(uint)i * VDN_NB + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // --- the rows below it: X_Ij = -L_II^-1 (sum_{K<I} L_IK X_Kj)
  for (int r0 = c0 + nb; r0 < d; r0 += VDN_NB) {
    const int mb = min(VDN_NB, d - r0);
    for (int i = tid; i < mb * nb; i += VDN_TI_THREADS) { acc[i] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k0 = c0; k0 < r0; k0 += VDN_NB) {
      const int kb = min(VDN_NB, d - k0);
      // Stage L[r0.., k0..] first. Read straight from device inside the
      // k loop it is one device load per multiply-add -- the same two-
      // loads-per-flop shape that held the Cholesky above to a third of
      // its throughput, and the same fix.
      for (int i = tid; i < mb * kb; i += VDN_TI_THREADS) {
        ltile[(uint)(i / kb) * VDN_NB + (i % kb)] =
            Lb[(uint)(r0 + i / kb) * d + (k0 + (i % kb))];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int i = tid; i < mb * nb; i += VDN_TI_THREADS) {
        const int r = i / nb, c = i % nb;
        float s = 0.0f;
        for (int k = 0; k < kb; ++k) {
          s += ltile[(uint)r * VDN_NB + k]
               * xcol[(uint)(k0 + k) * VDN_NB + c];
        }
        acc[i] += s;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = tid; i < mb * mb; i += VDN_TI_THREADS) {
      ltile[i] = Lb[(uint)(r0 + i / mb) * d + (r0 + (i % mb))];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < mb; ++i) {
      if (tid < nb) {
        float s = -acc[(uint)i * nb + tid];
        for (int k = 0; k < i; ++k) {
          s -= ltile[(uint)i * VDN_NB + k]
               * xcol[(uint)(r0 + k) * VDN_NB + tid];
        }
        xcol[(uint)(r0 + i) * VDN_NB + tid] = s / ltile[(uint)i * VDN_NB + i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  for (int r = c0 + tid; r < d; r += VDN_TI_THREADS) {
    for (int c = 0; c < nb; ++c) {
      Xb[(uint)r * d + (c0 + c)] = xcol[(uint)r * VDN_NB + c];
    }
  }
}

// inv = Linv^T Linv  ( = (I + A)^-1 ), and transition = Diag(alpha) inv.
//
// Both in one pass because the second is the first with a row scale, and
// writing inv only to read it straight back would double the traffic on
// the largest tensor in the chain. Linv is LOWER triangular, so
// inv[i,j] = sum_{k >= max(i,j)} Linv[k,i] Linv[k,j] -- the loop starts
// at the diagonal rather than at zero, which halves the work.
//
//   0:Linv [batch,d,d]  1:alpha [batch,d]  2:inv out  3:transition out
//   4:d
#define VDN_GT 16          // 16x16 threads, each owning a 2x2 output

kernel void vdn_inv_and_transition_f32(
    const device float* Linv  [[buffer(0)]],
    const device float* alpha [[buffer(1)]],
    device float*       inv   [[buffer(2)]],
    device float*       trans [[buffer(3)]],
    constant int&       d     [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpit [[thread_position_in_threadgroup]])
{
  const uint batch = tgid.z;
  const int i0 = (int)tgid.x * VDN_NB;
  const int j0 = (int)tgid.y * VDN_NB;
  if (i0 >= d || j0 >= d) { return; }

  const device float* Lb = Linv + (uint)batch * d * d;
  const device float* ab = alpha + (uint)batch * d;
  device float* ib = inv + (uint)batch * d * d;
  device float* tb = trans + (uint)batch * d * d;

  const int tx = (int)tpit.x, ty = (int)tpit.y;
  float sum[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
  const int ii[2] = {i0 + ty, i0 + ty + VDN_GT};
  const int jj[2] = {j0 + tx, j0 + tx + VDN_GT};

  // k runs from the larger of the two block origins: every term with
  // k < max(i,j) has a zero factor.
  const int kstart = max(i0, j0);
  for (int k = kstart; k < d; ++k) {
    const float la = (k >= ii[0]) ? Lb[(uint)k * d + ii[0]] : 0.0f;
    const float lb2 =
        (k >= ii[1] && ii[1] < d) ? Lb[(uint)k * d + ii[1]] : 0.0f;
    const float ra = (k >= jj[0]) ? Lb[(uint)k * d + jj[0]] : 0.0f;
    const float rb = (k >= jj[1] && jj[1] < d) ? Lb[(uint)k * d + jj[1]] : 0.0f;
    sum[0][0] += la * ra;
    sum[0][1] += la * rb;
    sum[1][0] += lb2 * ra;
    sum[1][1] += lb2 * rb;
  }
  for (int a = 0; a < 2; ++a) {
    if (ii[a] >= d) { continue; }
    const float scale = ab[ii[a]];
    for (int b = 0; b < 2; ++b) {
      if (jj[b] >= d) { continue; }
      const uint off = (uint)ii[a] * d + jj[b];
      ib[off] = sum[a][b];
      tb[off] = scale * sum[a][b];
    }
  }
}

// Batched C = A @ B, fp32, for injection = B_stat (I + A)^-1.
//   0:A [batch,m,k]  1:B [batch,k,n]  2:C out [batch,m,n]  3:m 4:n 5:k
kernel void vdn_gemm_nn_f32(
    const device float* A [[buffer(0)]],
    const device float* B [[buffer(1)]],
    device float*       C [[buffer(2)]],
    constant int&       m [[buffer(3)]],
    constant int&       n [[buffer(4)]],
    constant int&       k [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpit [[thread_position_in_threadgroup]])
{
  threadgroup float at[VDN_NB * VDN_NB];
  threadgroup float bt[VDN_NB * VDN_NB];

  const uint batch = tgid.z;
  const int i0 = (int)tgid.x * VDN_NB;
  const int j0 = (int)tgid.y * VDN_NB;
  const int tx = (int)tpit.x, ty = (int)tpit.y;

  const device float* Ab = A + (uint)batch * m * k;
  const device float* Bb = B + (uint)batch * k * n;
  device float* Cb = C + (uint)batch * m * n;

  float sum[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
  for (int k0 = 0; k0 < k; k0 += VDN_NB) {
    for (int t = ty; t < VDN_NB; t += VDN_GT) {
      const int ar = i0 + t, bc = j0 + tx;
      at[(uint)t * VDN_NB + tx] =
          (ar < m && k0 + tx < k) ? Ab[(uint)ar * k + (k0 + tx)] : 0.0f;
      at[(uint)t * VDN_NB + tx + VDN_GT] =
          (ar < m && k0 + tx + VDN_GT < k)
              ? Ab[(uint)ar * k + (k0 + tx + VDN_GT)] : 0.0f;
      bt[(uint)t * VDN_NB + tx] =
          (k0 + t < k && bc < n) ? Bb[(uint)(k0 + t) * n + bc] : 0.0f;
      bt[(uint)t * VDN_NB + tx + VDN_GT] =
          (k0 + t < k && bc + VDN_GT < n)
              ? Bb[(uint)(k0 + t) * n + bc + VDN_GT] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int kk = 0; kk < VDN_NB; ++kk) {
      const float a0 = at[(uint)ty * VDN_NB + kk];
      const float a1 = at[(uint)(ty + VDN_GT) * VDN_NB + kk];
      const float b0 = bt[(uint)kk * VDN_NB + tx];
      const float b1 = bt[(uint)kk * VDN_NB + tx + VDN_GT];
      sum[0][0] += a0 * b0;
      sum[0][1] += a0 * b1;
      sum[1][0] += a1 * b0;
      sum[1][1] += a1 * b1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const int ii[2] = {i0 + ty, i0 + ty + VDN_GT};
  const int jj[2] = {j0 + tx, j0 + tx + VDN_GT};
  for (int a = 0; a < 2; ++a) {
    if (ii[a] >= m) { continue; }
    for (int b = 0; b < 2; ++b) {
      if (jj[b] >= n) { continue; }
      Cb[(uint)ii[a] * n + jj[b]] = sum[a][b];
    }
  }
}

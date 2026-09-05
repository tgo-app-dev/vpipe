// vdn_branch.metal -- VDN-H3's linear branch on the GPU.
//
// The branch collapses each frame's S tokens into two d x d matrices and
// then runs a bidirectional delta-rule recurrence over FRAMES. At
// production geometry S ~ 1008 and d = 128, so the collapse is the
// branch's dominant term: two [d, S] x [S, d] products per (frame, head)
// -- 2 x 33 MFLOP x 5712 pairs = 376 GFLOP per block per forward, which
// is more than everything downstream of it put together.
//
// fp32, and not as a default. A is the matrix the scan INVERTS, and it
// is only symmetric to the working precision because its (i,j) and (j,i)
// entries come from different reductions. In bf16 that asymmetry is
// enough to push the smallest eigenvalue of I + A below the 1 the maths
// guarantees, and the Cholesky -- which reads one triangle -- then
// factorises a matrix nobody meant. B is a plain readout that is never
// inverted and would be fine narrower; it shares this kernel because it
// shares the operand and the beta scaling, and reading k twice would
// cost more than the extra width does.

#include <metal_stdlib>
using namespace metal;

#define VB_TILE 32       // d x d output tile per threadgroup
#define VB_GT   16       // 16x16 threads, each owning a 2x2 of each output
#define VB_SB   16       // tokens staged per pass

// THE INPUT SIDE IS STRIDED, and that is the whole reason this branch
// costs no memory inside the transformer.
//
// The CPU reference hands q/k/v over as three tight fp32 [rows, H, d]
// tensors, and the goldens do the same. H3 has no such thing: its
// projection is ONE fused bf16 [rows, 3*inner] buffer, grouped per head
// on the released checkpoint, whose video rows are a run inside a packed
// sequence that also holds text and audio. Materialising three fp32
// copies of that would be 8.67 GB per block at production geometry --
// larger than everything the tiling and the bank aliasing took out.
//
// So every kernel that reads an EXTERNAL tensor takes both pointers, an
// element-type flag and the two strides. Tight fp32 is row_stride = H*d,
// head_stride = d, elt = 0, which is exactly what the reference path
// passes -- so that path is not a special case here, it is the default
// arguments of the general one.
inline float
vb_ld(const device float* f, const device ushort* b, int elt, uint idx)
{
  if (elt != 0) { return as_type<float>((uint)b[idx] << 16); }
  return f[idx];
}

// bf16 with round-to-nearest-even. The READ side truncates because it
// is undoing a store somebody else made; a store of our own has no
// excuse to lose the half-ulp.
inline ushort
vb_bf16(float v)
{
  const uint u = as_type<uint>(v);
  if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0u) {
    return (ushort)((u >> 16) | 0x0040u);          // keep NaN a NaN
  }
  return (ushort)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// The store side of the same choice. `elt` picks which pointer is
// written; both are always bound, to the same bytes.
//
// THE FEATURE TENSORS ARE bf16 IN THE REFERENCE and fp32 here, which is
// the one place this port is MORE precise than the model it implements.
// scan.py is explicit -- "Only the small [F,H,d,d] results are promoted
// to fp32" -- because an fp32 contraction over [F,H,S,d] both doubles
// the traffic and misses the tensor cores. On a matrix-core GPU that
// second half is the bigger one, which is why this is a switch rather
// than a constant: the goldens were taken against an fp32 reference and
// still check against it exactly, while a real forward runs narrow.
inline void
vb_st(device float* f, device ushort* b, int elt, uint idx, float v)
{
  if (elt != 0) { b[idx] = vb_bf16(v); }
  else          { f[idx] = v; }
}

// Source index of (row, head, chan) under those strides.
inline uint
vb_at(uint row, int h, int i, int row_stride, int head_stride)
{
  return row * (uint)row_stride + (uint)(h * head_stride + i);
}

// A[f,h] = sum_s beta k k^T   and   B[f,h] = sum_s beta v k^T.
//
// One threadgroup per (frame, head, output tile). Both outputs share the
// right operand k and the beta weighting, so the tokens are read once.
//
//   0:key [F*S, H, d]  1:value [F*S, H, d]  2:beta [F*S, H]
//   3:A out [F, H, d, d]  4:B out  5:tokens_per_frame  6:heads  7:d
kernel void vdn_frame_stats_f32(
    const device float* key   [[buffer(0)]],
    const device float* value [[buffer(1)]],
    const device float* beta  [[buffer(2)]],
    device float*       A     [[buffer(3)]],
    device float*       B     [[buffer(4)]],
    constant int&       S     [[buffer(5)]],
    constant int&       H     [[buffer(6)]],
    constant int&       d     [[buffer(7)]],
    const device ushort* key_b   [[buffer(8)]],
    const device ushort* value_b [[buffer(9)]],
    const device ushort* beta_b  [[buffer(10)]],
    constant int&        felt    [[buffer(11)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpit [[thread_position_in_threadgroup]])
{
  // The FEATURES are narrow; A and B are not. That is the reference's
  // own line -- "Only the small [F,H,d,d] results are promoted to fp32"
  // -- and the reason is that A is what the Cholesky inverts.
  threadgroup float ki[VB_TILE * VB_SB];   // k[i0 + i][s] * beta[s]
  threadgroup float kj[VB_TILE * VB_SB];   // k[j0 + j][s]
  threadgroup float vi[VB_TILE * VB_SB];   // v[i0 + i][s] * beta[s]

  const int i0 = (int)tgid.x * VB_TILE;
  const int j0 = (int)tgid.y * VB_TILE;
  const int fh = (int)tgid.z;              // frame * heads + head
  if (i0 >= d || j0 >= d) { return; }
  const int f = fh / H, h = fh - (fh / H) * H;

  const int tx = (int)tpit.x, ty = (int)tpit.y;
  const int lin = ty * VB_GT + tx;
  float sa[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
  float sb[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

  for (int s0 = 0; s0 < S; s0 += VB_SB) {
    const int sb_n = min(VB_SB, S - s0);
    // Stage three [VB_TILE, sb_n] panels. 256 threads, 512 elements each.
    for (int e = lin; e < VB_TILE * VB_SB; e += VB_GT * VB_GT) {
      const int r = e / VB_SB, s = e - (e / VB_SB) * VB_SB;
      float a = 0.0f, b = 0.0f, c = 0.0f;
      if (s < sb_n) {
        const uint tok = (uint)((f * S + s0 + s) * H + h);
        const float bt = vb_ld(beta, beta_b, felt, tok);
        if (i0 + r < d) {
          const uint e0 = (uint)tok * d + (i0 + r);
          a = vb_ld(key, key_b, felt, e0) * bt;
          c = vb_ld(value, value_b, felt, e0) * bt;
        }
        if (j0 + r < d) {
          b = vb_ld(key, key_b, felt, (uint)tok * d + (j0 + r));
        }
      }
      ki[e] = a;
      kj[e] = b;
      vi[e] = c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int s = 0; s < sb_n; ++s) {
      const float a0 = ki[(uint)ty * VB_SB + s];
      const float a1 = ki[(uint)(ty + VB_GT) * VB_SB + s];
      const float c0 = vi[(uint)ty * VB_SB + s];
      const float c1 = vi[(uint)(ty + VB_GT) * VB_SB + s];
      const float b0 = kj[(uint)tx * VB_SB + s];
      const float b1 = kj[(uint)(tx + VB_GT) * VB_SB + s];
      sa[0][0] += a0 * b0; sa[0][1] += a0 * b1;
      sa[1][0] += a1 * b0; sa[1][1] += a1 * b1;
      sb[0][0] += c0 * b0; sb[0][1] += c0 * b1;
      sb[1][0] += c1 * b0; sb[1][1] += c1 * b1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  device float* Ab = A + (uint)fh * d * d;
  device float* Bb = B + (uint)fh * d * d;
  const int ii[2] = {i0 + ty, i0 + ty + VB_GT};
  const int jj[2] = {j0 + tx, j0 + tx + VB_GT};
  for (int a = 0; a < 2; ++a) {
    if (ii[a] >= d) { continue; }
    for (int b = 0; b < 2; ++b) {
      if (jj[b] >= d) { continue; }
      const uint off = (uint)ii[a] * d + jj[b];
      Ab[off] = sa[a][b];
      Bb[off] = sb[a][b];
    }
  }
}

// A <- (A + A^T) / 2, in place, batched.
//
// Separate from the kernel above because no threadgroup there owns both
// (i,j) and (j,i): the tiles are independent by construction, which is
// what makes the product fast. Cheap -- one read and one write per
// element -- and NOT optional: the Cholesky downstream reads a single
// triangle, so an unsymmetrised A silently factorises the wrong matrix.
//
//   0:A [batch, d, d]  1:d
kernel void vdn_symmetrise_f32(
    device float* A [[buffer(0)]],
    constant int& d [[buffer(1)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpit [[thread_position_in_threadgroup]])
{
  const uint batch = tgid.y;
  device float* Ab = A + (uint)batch * d * d;
  const int i = (int)tgid.x * 32 + (int)tpit.x;
  if (i >= d) { return; }
  for (int j = i + 1; j < d; ++j) {
    const uint a = (uint)i * d + j, b = (uint)j * d + i;
    const float m = 0.5f * (Ab[a] + Ab[b]);
    Ab[a] = m;
    Ab[b] = m;
  }
}

// ---------------------------------------------------------------------
// The branch's feature stage: [ShortConv ->] SiLU [-> L2Norm].
//
// TWO KERNELS, not one fused 5^3 gather. The stencil is separable by
// construction -- the effective 3D kernel is the outer product
// w_tm x w_sp, 30 parameters per channel rather than 125 -- and the
// dense form has no fast shape anywhere: 125 taps per output against
// 25 + 5. The two halves COMMUTE (they act on disjoint axes: the 5x5 is
// identical for every frame, the 5-tap identical for every spatial
// position), which is what licenses putting the SiLU and the L2 norm on
// the temporal half rather than writing the conv output to memory and
// reading it straight back.

#define VB_ACT_THREADS 128     // one per channel within a head

// Depthwise 5x5 within each frame, zero-padded. Channels are
// (head, head_dim) flattened.
//   0:in [F*S, C]  1:w [C, K, K]  2:out  3:frames 4:grid_h 5:grid_w
//   6:channels 7:kernel
// The short conv's SPATIAL half: a depthwise 5x5 over the (y, x) grid.
//
// EACH THREAD COMPUTES A RUN OF `VB_CONV_NX` NEIGHBOURING x, and that is
// the whole optimisation. One output needs 25 input loads and 25 weight
// loads; the first version did exactly that and moved 40.6 GB for 13.5
// GFLOP -- 82 GF/s, 1.3% of what this box's GEMMs reach, at 90% of its
// DRAM bandwidth. Neither operand was reused, though both working sets
// (0.72 MB of weights, a 2.15 MB stencil window) fit in cache: the
// traversal is channel-fastest, so a full sweep of the weights happens
// per CELL, and in this layout one step in x moves the input address by
// `rst` elements and one step in y by gw * rst -- the stencil has no
// spatial locality at all, only coalescing across adjacent channels.
//
// A run of NX outputs shares its 25 weights (loaded once into registers)
// and overlaps its input: the union of columns for NX outputs is NX + 4
// per row, so 5 * (NX + 4) loads instead of 25 * NX.
//
// SPECIALISED ON K = 5, which is the only value this port has: the host
// hard-codes it and the released config carries no kernel size. `wl` has
// to be indexable at compile time to stay in registers, which a runtime
// K forbids -- so a mismatch falls back to the general scalar loop
// rather than reading garbage. Same for a caller whose `nx` disagrees
// with VB_CONV_NX, which makes the two definitions self-correcting
// instead of merely commented.

#define VB_CONV_K  5
// MEASURED at generation geometry (37 frames of 17x30): 4 -> 57.5 ms,
// 8 -> 41.8, 12 -> 76.0, 16 -> 75.0. Eight is the knee -- below it the
// 25 weights are not amortised over enough outputs, above it `col`,
// `acc` and `wl` stop fitting in registers and the spill costs more
// than the reuse buys.
#define VB_CONV_NX 8

kernel void vdn_conv_spatial_f32(
    const device float* in  [[buffer(0)]],
    const device float* w   [[buffer(1)]],
    device float*       out [[buffer(2)]],
    constant int&       F   [[buffer(3)]],
    constant int&       gh  [[buffer(4)]],
    constant int&       gw  [[buffer(5)]],
    constant int&       C   [[buffer(6)]],
    constant int&       K   [[buffer(7)]],
    const device ushort* in_b [[buffer(8)]],
    constant int&       elt [[buffer(9)]],
    constant int&       rst [[buffer(10)]],
    constant int&       hst [[buffer(11)]],
    constant int&       hd  [[buffer(12)]],
    constant int&       nx  [[buffer(13)]],
    device ushort*      out_b [[buffer(14)]],
    constant int&       oelt  [[buffer(15)]],
    uint tid [[thread_position_in_grid]])
{
  const int pad = K / 2;
  const bool fast = (K == VB_CONV_K) && (nx == VB_CONV_NX);
  const int run = fast ? VB_CONV_NX : 1;
  const int nxg = (gw + run - 1) / run;

  const uint total = (uint)F * (uint)gh * (uint)nxg * (uint)C;
  if (tid >= total) { return; }

  const int c = (int)(tid % (uint)C);
  const uint blk = tid / (uint)C;
  const int xg = (int)(blk % (uint)nxg);
  const int y  = (int)((blk / (uint)nxg) % (uint)gh);
  const int f  = (int)(blk / ((uint)nxg * (uint)gh));
  const int x0 = xg * run;

  // Loop-invariant: the channel's head and lane in the source's layout,
  // and the base of its own 5x5 kernel.
  const int chh = c / hd, chl = c - (c / hd) * hd;
  const device float* wc = w + (uint)c * (uint)K * (uint)K;
  const uint plane = (uint)f * (uint)gh * (uint)gw;

  if (!fast) {
    // The general path, for a kernel size this file is not specialised
    // for. One output, 25 loads, exactly as before.
    float acc = 0.0f;
    for (int ky = 0; ky < K; ++ky) {
      const int sy = y + ky - pad;
      if (sy < 0 || sy >= gh) { continue; }
      for (int kx = 0; kx < K; ++kx) {
        const int sx = x0 + kx - pad;
        if (sx < 0 || sx >= gw) { continue; }
        const uint row = plane + (uint)sy * (uint)gw + (uint)sx;
        acc += vb_ld(in, in_b, elt, vb_at(row, chh, chl, rst, hst))
               * wc[ky * K + kx];
      }
    }
    if (x0 < gw) {
      vb_st(out, out_b, oelt,
            (plane + (uint)y * (uint)gw + (uint)x0) * (uint)C + (uint)c, acc);
    }
    return;
  }

  float wl[VB_CONV_K * VB_CONV_K];
  for (int j = 0; j < VB_CONV_K * VB_CONV_K; ++j) { wl[j] = wc[j]; }

  float acc[VB_CONV_NX];
  for (int n = 0; n < VB_CONV_NX; ++n) { acc[n] = 0.0f; }

  // One row of the stencil at a time: NX + K - 1 columns cover every
  // output in the run, and each is read ONCE instead of up to K times.
  float col[VB_CONV_NX + VB_CONV_K - 1];
  for (int ky = 0; ky < VB_CONV_K; ++ky) {
    const int sy = y + ky - pad;
    if (sy < 0 || sy >= gh) { continue; }        // zero padding
    const uint row = plane + (uint)sy * (uint)gw;
    for (int t = 0; t < VB_CONV_NX + VB_CONV_K - 1; ++t) {
      const int sx = x0 + t - pad;
      col[t] = (sx >= 0 && sx < gw)
                   ? vb_ld(in, in_b, elt,
                           vb_at(row + (uint)sx, chh, chl, rst, hst))
                   : 0.0f;
    }
    for (int n = 0; n < VB_CONV_NX; ++n) {
      for (int kx = 0; kx < VB_CONV_K; ++kx) {
        acc[n] += col[n + kx] * wl[ky * VB_CONV_K + kx];
      }
    }
  }

  for (int n = 0; n < VB_CONV_NX; ++n) {
    const int x = x0 + n;
    if (x >= gw) { break; }
    vb_st(out, out_b, oelt,
          (plane + (uint)y * (uint)gw + (uint)x) * (uint)C + (uint)c, acc[n]);
  }
}

kernel void vdn_temporal_act_f32(
    const device float* in   [[buffer(0)]],
    const device float* w    [[buffer(1)]],
    device float*       out  [[buffer(2)]],
    constant int&       F    [[buffer(3)]],
    constant int&       S    [[buffer(4)]],
    constant int&       H    [[buffer(5)]],
    constant int&       d    [[buffer(6)]],
    constant int&       K    [[buffer(7)]],
    constant int&       use_temporal [[buffer(8)]],
    constant int&       l2norm       [[buffer(9)]],
    constant int&       frame_base   [[buffer(10)]],
    constant int&       total_frames [[buffer(11)]],
    constant int&       out_base     [[buffer(12)]],
    const device ushort* in_b [[buffer(13)]],
    constant int&       elt  [[buffer(14)]],
    constant int&       rst  [[buffer(15)]],
    constant int&       hst  [[buffer(16)]],
    device ushort*      out_b [[buffer(17)]],
    constant int&       oelt  [[buffer(18)]],
    // vdn: the per-(token, head) beta, folded in as its SQUARE ROOT.
    //
    // The statistics downstream are A = sum_s beta k k^T and
    // B = sum_s beta v k^T, and the fp32 kernel applies beta itself while
    // it reads. A matmul2d cannot: it contracts two tensors and there is
    // nowhere to put a per-row weight. Splitting it half and half --
    // sqrt(beta) on BOTH operands -- gives the same products with no
    // third tensor and no extra pass, since every operand of both
    // products is one of these two.
    //
    // It also makes A EXACTLY symmetric, being X^T X for one X: (i, j)
    // and (j, i) are then the same products summed in the same order.
    // The symmetrise pass still runs, because that is a property of the
    // matmul and not of this.
    //
    // beta is a sigmoid, so it is in (0, 1) and the root is real.
    const device float*  beta   [[buffer(19)]],
    const device ushort* beta_b [[buffer(20)]],
    constant int&        use_beta [[buffer(21)]],
    constant int&        belt   [[buffer(22)]],
    // vdn: the spatial conv's output is a RING over frames, so that a
    // frame it computed for one tile's halo is still there when the next
    // tile wants it as a body frame. `slot_base` is where this tile's
    // first input frame sits in that ring and `nslot` is its length;
    // nslot == 0 is the flat buffer, which is what the source-reading
    // path (no spatial conv) uses.
    //
    // The 5-tap stencil is the only reader that has to know: it walks
    // frames either side of its own, and near the ring's seam those are
    // at the other end of the buffer.
    constant int&        slot_base [[buffer(23)]],
    constant int&        nslot     [[buffer(24)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpit [[thread_position_in_threadgroup]])
{
  threadgroup float part[VB_ACT_THREADS];

  // The OUTPUT token, which may be a sub-range of the input when the
  // caller supplied a halo for the 5-tap stencil.
  const int token = (int)tgid.x + out_base * S;
  const int h = (int)tgid.y;
  const int i = (int)tpit.x;
  if (token >= F * S || h >= H || i >= d) { return; }
  const int f = token / S, s = token - (token / S) * S;
  const int c = h * d + i;
  const int pad = K / 2;

  float v = 0.0f;
  if (use_temporal != 0) {
    for (int dt = 0; dt < K; ++dt) {
      const int sf = f + dt - pad;
      // Zero padding at the CLIP's ends, not the tile's.
      const int global_f = frame_base + sf;
      if (sf < 0 || sf >= F || global_f < 0 || global_f >= total_frames) {
        continue;
      }
      // The conv path reads the SPATIAL kernel's own tight output, so
      // it is the strides' default case and never the fused source --
      // but it follows the same element type, which the caller sets.
      // slot_base + sf < 2 * nslot (the window is at most the ring), so
      // one wrap is all it can need.
      int sl = sf;
      if (nslot > 0) {
        sl = slot_base + sf;
        if (sl >= nslot) { sl -= nslot; }
      }
      const uint src = ((uint)(sl * S + s) * H + h) * d + i;
      v += vb_ld(in, in_b, elt, src) * w[(uint)c * K + dt];
    }
  } else {
    v = vb_ld(in, in_b, elt, vb_at((uint)token, h, i, rst, hst));
  }
  v = v / (1.0f + exp(-v));                      // SiLU

  if (l2norm != 0) {
    part[i] = v * v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int stride = VB_ACT_THREADS / 2; stride > 0; stride >>= 1) {
      if (i < stride && i + stride < d) { part[i] += part[i + stride]; }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // fla's L2Norm: the norm is CLAMPED at eps, not eps added under the
    // root -- a distinction that only shows on a row that is all zero,
    // which SiLU can produce.
    const float n = max(sqrt(part[0]), 1e-6f);
    v = v / n;
  }
  if (use_beta != 0) {
    // Indexed TILE-LOCALLY, exactly as the store below is: the caller
    // binds beta at the tile's own base, so token 0 here is the tile's
    // first token in both tensors.
    const uint bt = (uint)(token - out_base * S) * (uint)H + (uint)h;
    v *= sqrt(vb_ld(beta, beta_b, belt, bt));
  }
  vb_st(out, out_b, oelt,
        ((uint)(token - out_base * S) * H + h) * d + i, v);
}

// ---------------------------------------------------------------------
// The readout: q . state, RMSNorm, gate.
//
// Split into a GATE kernel and a READOUT kernel because the gate is a
// function of the hidden state alone -- one low-rank chain per token,
// shared by all 56 heads -- while the readout is per (token, head). Fused,
// the gate's 5376-wide reduction would be recomputed once per head.

#define VB_GATE_THREADS 128

// gate = sigmoid(up(down(x)) + b), per (token, head, channel).
//
// `down` may be null-shaped (bottleneck <= 0), which is the SOFTMAX
// branch's gate: direct, per head, one value broadcast over channels.
// The linear branch's is low rank (bottleneck = head_dim) and per
// channel -- a routing decision on a new pathway rather than a scale on
// a distribution, which is why the two differ in granularity.
//
// `bias` must ALWAYS be bound -- an unbound Metal buffer is not
// reliably null, so the comparison would read past the end of nothing.
// A caller with no bias binds zeros; beta_proj is exactly that case (no
// bias, which is why beta is centred on 0.5 whatever the hidden state
// does, hence trace(A) ~ S/2, hence the solve).
//
//   0:x [tokens, hidden]  1:down [bottleneck, hidden]  2:up [out, bneck]
//   3:bias [out] 4:gate out [tokens, out]  5:hidden 6:bottleneck 7:out
kernel void vdn_gate_f32(
    const device float* x    [[buffer(0)]],
    const device float* down [[buffer(1)]],
    const device float* up   [[buffer(2)]],
    const device float* bias [[buffer(3)]],
    device float*       gate [[buffer(4)]],
    constant int&       hidden [[buffer(5)]],
    constant int&       bneck  [[buffer(6)]],
    constant int&       outn   [[buffer(7)]],
    const device ushort* x_b [[buffer(8)]],
    constant int&       elt  [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpit [[thread_position_in_threadgroup]])
{
  threadgroup float lo[VB_GATE_THREADS];

  const uint token = tgid.x;
  const int t = (int)tpit.x;
  const uint xt0 = (uint)token * (uint)hidden;
  const device float*  xt  = x   + xt0;
  const device ushort* xtb = x_b + xt0;

  const int rank = bneck > 0 ? bneck : hidden;
  if (bneck > 0) {
    for (int j = t; j < bneck; j += VB_GATE_THREADS) {
      float acc = 0.0f;
      const device float* w = down + (uint)j * hidden;
      for (int i = 0; i < hidden; ++i) {
        acc += w[i] * vb_ld(xt, xtb, elt, (uint)i);
      }
      lo[j] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (int c = t; c < outn; c += VB_GATE_THREADS) {
    float acc = bias[c];
    const device float* w = up + (uint)c * rank;
    if (bneck > 0) {
      for (int j = 0; j < rank; ++j) { acc += w[j] * lo[j]; }
    } else {
      for (int i = 0; i < rank; ++i) {
        acc += w[i] * vb_ld(xt, xtb, elt, (uint)i);
      }
    }
    gate[(uint)token * outn + c] = 1.0f / (1.0f + exp(-acc));
  }
}

// readout[token, h, v] = RMSNorm_v( sum_k state[f, h, v, k] q[token, h, k] )
//                        * norm_w[v] * gate[token, h * d + v]
//
// One threadgroup per (token, head), one thread per output channel v:
// the state row v is contiguous and q is shared by every v, so q is
// staged once and the RMS reduction is a shuffle inside the head rather
// than a second pass.
//
//   0:q [tokens, H, d]  1:state [F, H, d, d]  2:gate [tokens, H*d]
//   3:norm_w [d]  4:out [tokens, H*d]  5:tokens_per_frame 6:H 7:d 8:eps
#define VB_RO_TM 8      // tokens a threadgroup carries
#define VB_RO_KC 32     // K staged per pass

kernel void vdn_readout_f32(
    const device float* q     [[buffer(0)]],
    const device float* state [[buffer(1)]],
    const device float* gate  [[buffer(2)]],
    const device float* normw [[buffer(3)]],
    device float*       out   [[buffer(4)]],
    constant int&       S     [[buffer(5)]],
    constant int&       H     [[buffer(6)]],
    constant int&       d     [[buffer(7)]],
    constant float&     eps   [[buffer(8)]],
    device ushort*      out_b [[buffer(9)]],
    constant int&       elt   [[buffer(10)]],
    const device ushort* q_b     [[buffer(11)]],
    const device ushort* state_b [[buffer(12)]],
    const device ushort* gate_b  [[buffer(13)]],
    constant int&        felt    [[buffer(14)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpit [[thread_position_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]])
{
  // A GEMM PER (FRAME, HEAD), not a dot product per (token, head).
  //
  // out[s, v] = sum_k state[f, h, v, k] q[s, h, k] is [S, d] x [d, d],
  // and every token of a frame shares that state -- 64 KB of it. The
  // first version put one threadgroup on one TOKEN and re-read the
  // whole state for each, which is why it ran at 5% of what this box's
  // own projections reach. Here a threadgroup owns VB_RO_TM tokens, so
  // each state element is read once for eight of them.
  //
  // TILES ARE FRAME-ALIGNED (grid.z is the frame) precisely so a tile
  // cannot straddle a frame: S is not a multiple of VB_RO_TM, and a
  // straddling tile would need two states and lose the reuse that is
  // the whole point.
  threadgroup float qs[VB_RO_TM][VB_RO_KC];
  threadgroup float red[VB_RO_TM][4];

  const int v = (int)tpit.x;
  const int h = (int)tgid.y;
  const int fz = (int)tgid.z;
  const int s0 = (int)tgid.x * VB_RO_TM;
  if (v >= d || h >= H) { return; }

  // The STATE is the branch's largest read here -- 8.19 of 9.00 GB,
  // because all S tokens of a frame share it and it comes back once per
  // token tile. The reference narrows exactly this tensor on its way
  // into the readout (`gather_linear_state(...).to(xv.dtype)`).
  const uint stb = ((uint)fz * H + h) * d * d;
  const device float*  st  = state   + stb;
  const device ushort* stn = state_b + stb;
  float acc[VB_RO_TM];
  for (int i = 0; i < VB_RO_TM; ++i) { acc[i] = 0.0f; }

  for (int k0 = 0; k0 < d; k0 += VB_RO_KC) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = v; idx < VB_RO_TM * VB_RO_KC; idx += d) {
      const int i = idx / VB_RO_KC, kk = idx - (idx / VB_RO_KC) * VB_RO_KC;
      const int s = s0 + i;
      const int k = k0 + kk;
      qs[i][kk] = (s < S && k < d)
                      ? vb_ld(q, q_b, felt,
                              ((uint)(fz * S + s) * H + h) * d + k)
                      : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int kk = 0; kk < VB_RO_KC && k0 + kk < d; ++kk) {
      const float sv = vb_ld(st, stn, felt, (uint)v * d + k0 + kk);
      for (int i = 0; i < VB_RO_TM; ++i) { acc[i] += sv * qs[i][kk]; }
    }
  }

  // RMSNorm over head_dim, ONE barrier for all VB_RO_TM rows. d = 128 is
  // four simdgroups, so a simd_sum plus a four-way fold replaces the
  // seven-deep barrier tree this used to walk per token.
  for (int i = 0; i < VB_RO_TM; ++i) {
    const float p = simd_sum(acc[i] * acc[i]);
    if (lane == 0) { red[i][sg] = p; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int i = 0; i < VB_RO_TM; ++i) {
    const int s = s0 + i;
    if (s >= S) { break; }
    const float tot = red[i][0] + red[i][1] + red[i][2] + red[i][3];
    const float inv = rsqrt(tot / (float)d + eps);
    const uint token = (uint)(fz * S + s);
    const uint o = token * (uint)H * (uint)d + (uint)h * (uint)d + (uint)v;
    const float r = acc[i] * inv * normw[v] * vb_ld(gate, gate_b, felt, o);
    // The transformer wants this bf16: it is about to be projected by a
    // bf16 GEMM and added to a bf16 residual stream, and holding it fp32
    // would put 1.45 GB back for one dispatch. The goldens read the fp32
    // spelling.
    if (elt != 0) { out_b[o] = vb_bf16(r); }
    else          { out[o] = r; }
  }
}

// ---------------------------------------------------------------------
// PROBE, not a production kernel: does the Metal compiler evaluate a
// 32-bit signed product in 32 bits when it feeds a 64-bit pointer?
//
// The block-sparse attentions advance their key pointer by
// (gap blocks) * (block size) * (sequence stride). All three are int,
// and at video geometry with a FUSED projection (stride 3 * inner =
// 21504) the product passes 2^31 on a clip of ~206 frames. Signed
// overflow is UB, so LLVM may assume it does not happen and widen the
// multiply into the sign-extension that follows -- in which case the
// 32-bit spelling gives the right answer anyway, on this toolchain, by
// an assumption the source does not state.
//
// This asks the question directly instead of inferring it from an
// end-to-end run, where a promoted multiply and a correct one are
// indistinguishable.
//
//   0:out int64[3] -- [0] the int spelling widened, [1] the int64
//   spelling, [2] the int spelling truncated to 32 bits
//   1:gap 2:blk 3:stride
kernel void vdn_index_probe(
    device long*  out    [[buffer(0)]],
    constant int& gap    [[buffer(1)]],
    constant int& blk    [[buffer(2)]],
    constant int& stride [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
  if (tid != 0) { return; }
  device const half* base = (device const half*)0;
  // Exactly the shape the kernels use: an int product added to a
  // pointer.
  device const half* p32 = base + gap * blk * stride;
  device const half* p64 = base + (long)gap * blk * stride;
  out[0] = (long)(size_t)p32;
  out[1] = (long)(size_t)p64;
  out[2] = (long)(int)(gap * blk * stride);
}

// ---------------------------------------------------------------------
// The short conv's spatial weights as a BLOCK-DIAGONAL dense kernel.
//
// MPP's convolution2d takes groups == 1 only, so the one way a depthwise
// conv reaches the matrix units is as a dense conv over a block of B
// channels whose weight is zero off the diagonal -- B times the
// arithmetic for the same answer. MEASURED on an M5 that is a 1.25x win
// at B = 16 and a LOSS at 8, 32 and 64 (conv2d_mma.hw_op_depthwise_perf):
// below 16 the matrix unit's issued rate collapses (8.0 -> 1.8 TFLOP/s),
// above it the waste grows faster than the rate improves.
//
// Built per forward into scratch rather than per layer into the block:
// it is B times the bytes of the weight it expands, which over 50 layers
// would be 573 MB against the 4.28 GB of the whole checkpoint, while the
// expansion itself is 2.9M stores -- under a tenth of a millisecond.
//
//   0:w [C, KS, KS] fp32  1:out [C/B, KS, KS, B, B] bf16, hwio
//   2:C 3:KS 4:B
kernel void vdn_dw_weight_blockdiag(
    const device float* w   [[buffer(0)]],
    device ushort*      out [[buffer(1)]],
    constant int&       C   [[buffer(2)]],
    constant int&       KS  [[buffer(3)]],
    constant int&       B   [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
  // One thread per OUTPUT element, so the zeros are written too -- the
  // buffer is reused across layers and a stale off-diagonal entry is a
  // cross-channel term that no shape check would catch.
  const uint total = (uint)(C / B) * (uint)(KS * KS) * (uint)(B * B);
  if (tid >= total) { return; }
  const uint bb = (uint)(B * B);
  const uint o  = tid % (uint)B;
  const uint i  = (tid / (uint)B) % (uint)B;
  const uint kk = (tid / bb) % (uint)(KS * KS);
  const uint cb = tid / (bb * (uint)(KS * KS));
  float v = 0.0f;
  if (i == o) { v = w[(cb * (uint)B + i) * (uint)(KS * KS) + kk]; }
  out[tid] = vb_bf16(v);
}

// ---------------------------------------------------------------------
// The tail of the readout: RMSNorm over head_dim, the norm weight, the
// gate.
//
// Split off for the matrix-core route, where the contraction in front of
// it is vdn_readout_mma. Everything here needs a WHOLE 128-wide row and
// a reduction across it, which is what a cooperative tensor will not
// hand over -- so the matmul stores its raw product and this reads it
// back, once, still in cache.
//
// The reduction is the same shape as the fused kernel's: d = 128 is four
// simdgroups, so a simd_sum and a four-way fold replace a barrier tree.
// One threadgroup per (token, head).
//
//   0:raw bf16 [rows, H*d]  1:gate  2:norm_w f32 [d]  3:out f32
//   4:out bf16  5:rows 6:H 7:d 8:eps 9:oelt 10:felt
kernel void vdn_readout_norm_f32(
    const device ushort* raw   [[buffer(0)]],
    const device float*  gate  [[buffer(1)]],
    const device float*  normw [[buffer(2)]],
    device float*        out   [[buffer(3)]],
    device ushort*       out_b [[buffer(4)]],
    constant int&        rows  [[buffer(5)]],
    constant int&        H     [[buffer(6)]],
    constant int&        d     [[buffer(7)]],
    constant float&      eps   [[buffer(8)]],
    constant int&        oelt  [[buffer(9)]],
    const device ushort* gate_b[[buffer(10)]],
    constant int&        felt  [[buffer(11)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpit [[thread_position_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]])
{
  threadgroup float red[4];
  const int v = (int)tpit.x;
  const int h = (int)tgid.y;
  const int token = (int)tgid.x;
  if (v >= d || h >= H || token >= rows) { return; }

  const uint o = (uint)token * (uint)H * (uint)d + (uint)h * (uint)d
                 + (uint)v;
  const float a = as_type<float>((uint)raw[o] << 16);
  const float p = simd_sum(a * a);
  if (lane == 0) { red[sg] = p; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float tot = red[0] + red[1] + red[2] + red[3];
  const float inv = rsqrt(tot / (float)d + eps);
  const float r = a * inv * normw[v] * vb_ld(gate, gate_b, felt, o);
  vb_st(out, out_b, oelt, o, r);
}

// ---------------------------------------------------------------------
// A TILED fp32 GEMM: C[M, N] = act(A[M, K] . B[N, K]^T + bias[N]).
//
// WHY THIS EXISTS. beta, the branch's output gate and the softmax gate
// were all one kernel that put ONE THREADGROUP ON ONE TOKEN and walked
// the weight matrix out of device memory -- so every token re-read all
// of it and nothing was reused. MEASURED at generation geometry: the
// output gate moved 550 GB/s of weights for 275 GF/s of arithmetic, one
// multiply-add per 4-byte load, which is DRAM rate on a 273 GB/s box.
// They were not slow kernels, they were matrix multiplies that had
// never been dispatched as one.
//
// B is [N, K] row-major, i.e. the checkpoint's own orientation for a
// linear layer -- so this is `y = x W^T`, and no weight is transposed
// to get here.
//
// A may be bf16 and strided, because the caller's `x` is the
// TRANSFORMER's normed hidden where it lies. B, the bias and C are
// always fp32 and tight.

#define VG_TM 32       // rows per threadgroup
#define VG_TN 32       // cols per threadgroup
#define VG_TK 16       // K staged per pass
#define VG_T  16       // 16x16 threads, each owning a 2x2 of the tile

kernel void vdn_gemm_act_f32(
    const device float*  A     [[buffer(0)]],
    const device float*  B     [[buffer(1)]],
    const device float*  bias  [[buffer(2)]],
    device float*        C     [[buffer(3)]],
    constant int&        M     [[buffer(4)]],
    constant int&        N     [[buffer(5)]],
    constant int&        K     [[buffer(6)]],
    constant int&        lda   [[buffer(7)]],   // elements per row of A
    constant int&        act   [[buffer(8)]],   // 0 none, 1 sigmoid
    constant int&        use_bias [[buffer(9)]],
    const device ushort* A_b   [[buffer(10)]],
    constant int&        a_elt [[buffer(11)]],  // 0 fp32, 1 bf16
    device ushort*       C_b   [[buffer(12)]],
    constant int&        c_elt [[buffer(13)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpit [[thread_position_in_threadgroup]])
{
  threadgroup float As[VG_TM][VG_TK];
  threadgroup float Bs[VG_TN][VG_TK];

  const int m0 = (int)tgid.x * VG_TM;
  const int n0 = (int)tgid.y * VG_TN;
  const int tx = (int)tpit.x, ty = (int)tpit.y;
  const int lin = ty * VG_T + tx;

  float acc[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

  for (int k0 = 0; k0 < K; k0 += VG_TK) {
    // 256 threads cooperatively stage a 32x16 tile of each operand.
    for (int i = lin; i < VG_TM * VG_TK; i += VG_T * VG_T) {
      const int r = i / VG_TK, c = i - r * VG_TK;
      const int gm = m0 + r, gk = k0 + c;
      As[r][c] = (gm < M && gk < K)
                     ? vb_ld(A, A_b, a_elt, (uint)gm * (uint)lda + (uint)gk)
                     : 0.0f;
    }
    for (int i = lin; i < VG_TN * VG_TK; i += VG_T * VG_T) {
      const int r = i / VG_TK, c = i - r * VG_TK;
      const int gn = n0 + r, gk = k0 + c;
      Bs[r][c] = (gn < N && gk < K) ? B[(uint)gn * (uint)K + (uint)gk]
                                    : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int kk = 0; kk < VG_TK; ++kk) {
      const float a0 = As[ty * 2 + 0][kk], a1 = As[ty * 2 + 1][kk];
      const float b0 = Bs[tx * 2 + 0][kk], b1 = Bs[tx * 2 + 1][kk];
      acc[0][0] += a0 * b0; acc[0][1] += a0 * b1;
      acc[1][0] += a1 * b0; acc[1][1] += a1 * b1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      const int gm = m0 + ty * 2 + i, gn = n0 + tx * 2 + j;
      if (gm >= M || gn >= N) { continue; }
      float v = acc[i][j];
      if (use_bias != 0) { v += bias[gn]; }
      if (act != 0) { v = 1.0f / (1.0f + exp(-v)); }
      vb_st(C, C_b, c_elt, (uint)gm * (uint)N + (uint)gn, v);
    }
  }
}

// ---------------------------------------------------------------------
// bias + sigmoid over a GEMM's raw output.
//
// The tail of vdn_gemm_act_f32, split off so the matmul in front of it
// can be somebody else's. On a matrix-core GPU the branch's three plain
// GEMMs go to dense_gemm_mma, which writes the product and nothing else
// -- it has no bias slot it reads and no activation -- so the bias and
// the sigmoid come back here, over a tensor that is already in cache.
//
// MEASURED at generation geometry the pass costs ~4 ms on the widest of
// them (the output gate, 18870 x 7168) against ~40 ms saved on the
// matmul, and nothing at all on the other two, which are 56 columns
// wide. Folding it into the readout -- which already reads the gate --
// would take that 4 ms back, at the price of a gate whose meaning
// depends on which route produced it.
//
//   0:src bf16 [M, N]  1:bias f32 [N]  2:dst f32  3:dst bf16
//   4:M 5:N 6:use_bias 7:act (0 none, 1 sigmoid)  8:c_elt
kernel void vdn_bias_act_f32(
    const device ushort* src  [[buffer(0)]],
    const device float*  bias [[buffer(1)]],
    device float*        dst  [[buffer(2)]],
    device ushort*       dst_b[[buffer(3)]],
    constant int&        M    [[buffer(4)]],
    constant int&        N    [[buffer(5)]],
    constant int&        use_bias [[buffer(6)]],
    constant int&        act  [[buffer(7)]],
    constant int&        c_elt[[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
  const uint total = (uint)M * (uint)N;
  if (tid >= total) { return; }
  const int n = (int)(tid % (uint)N);
  float v = as_type<float>((uint)src[tid] << 16);
  if (use_bias != 0) { v += bias[n]; }
  if (act != 0) { v = 1.0f / (1.0f + exp(-v)); }
  vb_st(dst, dst_b, c_elt, tid, v);
}

// ---------------------------------------------------------------------
// The SOFTMAX half's gate, applied in place to a bf16 attention output.
//
// Per HEAD and not per channel, and that is not an economy: a windowed
// softmax renormalises to 1 however little of the sequence it saw, so
// this scales it back toward the share it actually captured -- a scalar
// property of a distribution. The linear branch's gate is per channel
// because it routes a new pathway, which is a different question.
//
//   0:out [rows, H*d] bf16, in place  1:gate [rows, H]  2:H  3:d  4:rows
kernel void vdn_softmax_gate_apply(
    device ushort*      out  [[buffer(0)]],
    const device float* gate [[buffer(1)]],
    constant int&       H    [[buffer(2)]],
    constant int&       d    [[buffer(3)]],
    constant int&       rows [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
  const uint total = (uint)rows * (uint)H * (uint)d;
  if (tid >= total) { return; }
  const uint row = tid / (uint)(H * d);
  const uint h = (tid / (uint)d) % (uint)H;
  const uint b = (uint)out[tid] << 16;
  out[tid] = vb_bf16(as_type<float>(b) * gate[row * (uint)H + h]);
}

// ---------------------------------------------------------------------

// The per-frame mean of the hidden state, in fp32.
//
// Its own kernel rather than a line inside the alpha kernel because the
// precision is the point: the mean must be taken in fp32 BEFORE any
// downcast, since the scan multiplies alpha across every frame and a
// per-element error compounds over ~100 of them.
//
//   0:x [F*S, hidden]  1:mean out [F, hidden]  2:S  3:hidden
kernel void vdn_frame_mean_f32(
    const device float* x      [[buffer(0)]],
    device float*       mean   [[buffer(1)]],
    constant int&       S      [[buffer(2)]],
    constant int&       hidden [[buffer(3)]],
    const device ushort* x_b [[buffer(4)]],
    constant int&       elt  [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]])
{
  const int i = (int)tid.x;
  const uint f = tid.y;
  if (i >= hidden) { return; }
  float acc = 0.0f;
  for (int s = 0; s < S; ++s) {
    acc += vb_ld(x, x_b, elt, ((uint)f * S + s) * (uint)hidden + (uint)i);
  }
  mean[f * (uint)hidden + i] = acc / (float)S;
}

// alpha = exp(-exp(A_log) * softplus(up(down(mean)) + dt_bias)).
//
// KDA's DOUBLE exponential in fla's layout: A_log is [heads] -- PER HEAD
// -- and the per-channel freedom is dt_bias [heads*head_dim]. Reading
// A_log as per-channel type-checks and changes the retention spectrum;
// that mutation moves alpha by 1.4e-1 against a 1e-5 bar.
//
//   0:mean [F,hidden] 1:down [d,hidden] 2:up [H*d,d] 3:dt_bias [H*d]
//   4:A_log [H] 5:alpha out [F,H,d] 6:hidden 7:H 8:d
kernel void vdn_alpha_f32(
    const device float* mean   [[buffer(0)]],
    const device float* down   [[buffer(1)]],
    const device float* up     [[buffer(2)]],
    const device float* dtb    [[buffer(3)]],
    const device float* a_log  [[buffer(4)]],
    device float*       alpha  [[buffer(5)]],
    constant int&       hidden [[buffer(6)]],
    constant int&       H      [[buffer(7)]],
    constant int&       d      [[buffer(8)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpit [[thread_position_in_threadgroup]])
{
  threadgroup float lo[VB_GATE_THREADS];
  const uint f = tgid.x;
  const int t = (int)tpit.x;
  const device float* m = mean + f * (uint)hidden;

  for (int j = t; j < d; j += VB_GATE_THREADS) {
    float acc = 0.0f;
    const device float* w = down + (uint)j * hidden;
    for (int i = 0; i < hidden; ++i) { acc += w[i] * m[i]; }
    lo[j] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int c = t; c < H * d; c += VB_GATE_THREADS) {
    float acc = dtb[c];
    const device float* w = up + (uint)c * d;
    for (int j = 0; j < d; ++j) { acc += w[j] * lo[j]; }
    const float sp = acc > 20.0f ? acc : log(1.0f + exp(acc));
    alpha[f * (uint)(H * d) + c] = exp(-exp(a_log[c / d]) * sp);
  }
}

// The bridge factors: the product of alpha over the frames between a
// window boundary and the query frame.
//
// One thread per CHANNEL, walking frames: a prefix has only one order,
// and the H*d channels are what runs wide. The two index vectors are NOT
// the gather indices -- see vdn-geometry.h. A boundary row gathers a
// clamped state it then discards but must still decay the text state
// over the frames it really skipped, and clamping both the same way is
// off by one frame.
//
//   0:alpha [F,H,d] 1:bridge_before [F] 2:bridge_after [F]
//   3:from_before out 4:from_after out 5:F 6:H*d
kernel void vdn_alpha_bridge_f32(
    const device float* alpha   [[buffer(0)]],
    const device int*   bbefore [[buffer(1)]],
    const device int*   bafter  [[buffer(2)]],
    device float*       fbefore [[buffer(3)]],
    device float*       fafter  [[buffer(4)]],
    constant int&       F       [[buffer(5)]],
    constant int&       HD      [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
  const int c = (int)tid;
  if (c >= HD) { return; }
  // The EXCLUSIVE log-prefix, walked once and kept in registers as a
  // running sum: row r is the sum over frames [0, r). Every span the
  // gather needs is then one subtraction of two of them.
  float run = 0.0f;
  for (int t = 0; t < F; ++t) {
    const float lt = log(max(alpha[(uint)t * HD + c], 1e-12f));
    // prefix[t] = run (rows [0,t)), prefix[t+1] = run + lt.
    // from_before = exp(prefix[t+1] - prefix[bridge_before]).
    // from_after  = exp(prefix[bridge_after] - prefix[t]).
    // Both boundaries are re-walked because they are not monotone in t.
    float pb = 0.0f;
    for (int u = 0; u < bbefore[t]; ++u) {
      pb += log(max(alpha[(uint)u * HD + c], 1e-12f));
    }
    float pa = 0.0f;
    for (int u = 0; u < bafter[t]; ++u) {
      pa += log(max(alpha[(uint)u * HD + c], 1e-12f));
    }
    fbefore[(uint)t * HD + c] = exp(run + lt - pb);
    fafter[(uint)t * HD + c] = exp(pa - run);
    run += lt;
  }
}

// linear_state[f] = bridge(prefix[before]) + bridge(suffix[after]), with
// the text state standing in on a side that has nothing outside the
// window. [F, H, d_v, d_k]; alpha is per KEY channel and broadcasts over
// d_v.
//
//   0:prefix 1:suffix 2:fbefore 3:fafter 4:before_idx 5:after_idx
//   6:has_before 7:has_after 8:text_state (bound always) 9:out
//   10:H 11:d 12:use_text
kernel void vdn_gather_f32(
    const device float* prefix  [[buffer(0)]],
    const device float* suffix  [[buffer(1)]],
    const device float* fbefore [[buffer(2)]],
    const device float* fafter  [[buffer(3)]],
    const device int*   bidx    [[buffer(4)]],
    const device int*   aidx    [[buffer(5)]],
    const device int*   hasb    [[buffer(6)]],
    const device int*   hasa    [[buffer(7)]],
    const device float* text    [[buffer(8)]],
    device float*       out     [[buffer(9)]],
    constant int&       H       [[buffer(10)]],
    constant int&       d       [[buffer(11)]],
    constant int&       use_text [[buffer(12)]],
    device ushort*      out_b    [[buffer(13)]],
    constant int&       oelt     [[buffer(14)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpit [[thread_position_in_threadgroup]])
{
  // The BANKS this reads are fp32 -- alpha compounds over ~100 frames
  // and the scan is fp32 state maths -- but what it WRITES is read only
  // by the readout, which is where the reference narrows it too.
  const int f = (int)tgid.z;
  const int h = (int)tgid.y;
  const int v = (int)tgid.x;
  const int k = (int)tpit.x;
  if (k >= d) { return; }

  const uint mat = (uint)d * d;
  const uint e = (uint)h * mat + (uint)v * d + k;
  float before = prefix[(uint)bidx[f] * H * mat + e];
  float after = suffix[(uint)aidx[f] * H * mat + e];
  if (use_text != 0) {
    if (hasb[f] == 0) { before = text[e]; }
    if (hasa[f] == 0) { after = text[e]; }
  }
  const uint ai = (uint)f * H * d + (uint)h * d + k;
  before *= fbefore[ai];
  after *= fafter[ai];
  if (use_text == 0) {
    before *= (float)(hasb[f] != 0);
    after *= (float)(hasa[f] != 0);
  }
  vb_st(out, out_b, oelt, (uint)f * H * mat + e, before + after);
}

// ---------------------------------------------------------------------
// One scan step, batched over heads:
//
//     state_out[h] = state_in[h] @ transition[h] + injection[h]
//
// The frame loop stays on the HOST. It is a serial recurrence -- frame t
// needs frame t-1 -- and a whole head's [128, 128] fp32 state is 64 KB
// against a 32 KB threadgroup budget, so no threadgroup can own a head
// across frames and carry the chain in fast memory. What a dispatch per
// frame costs is launch latency on a kernel that is already small; what
// it buys is that the ONLY ordering constraint is expressed by the
// queue, rather than by a barrier inside a kernel that cannot honour it.
//
// The caller chains in place: state_in is the previous frame's row of
// the bank and state_out is this one's, so nothing is copied between
// steps.
//
//   0:state_in [H,d,d] 1:transition [H,d,d] 2:injection [H,d,d]
//   3:state_out [H,d,d] 4:d
kernel void vdn_state_step_f32(
    const device float* state_in [[buffer(0)]],
    const device float* trans    [[buffer(1)]],
    const device float* inject   [[buffer(2)]],
    device float*       state_out [[buffer(3)]],
    constant int&       d        [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpit [[thread_position_in_threadgroup]])
{
  threadgroup float at[VB_TILE * VB_TILE];
  threadgroup float bt[VB_TILE * VB_TILE];

  const uint h = tgid.z;
  const int i0 = (int)tgid.x * VB_TILE;
  const int j0 = (int)tgid.y * VB_TILE;
  const int tx = (int)tpit.x, ty = (int)tpit.y;
  const uint mat = (uint)d * d;
  const device float* A = state_in + h * mat;
  const device float* B = trans + h * mat;
  const device float* D = inject + h * mat;
  device float* C = state_out + h * mat;

  float sum[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
  for (int k0 = 0; k0 < d; k0 += VB_TILE) {
    for (int t = ty; t < VB_TILE; t += VB_GT) {
      const int ar = i0 + t, bc = j0 + tx;
      at[(uint)t * VB_TILE + tx] =
          (ar < d && k0 + tx < d) ? A[(uint)ar * d + (k0 + tx)] : 0.0f;
      at[(uint)t * VB_TILE + tx + VB_GT] =
          (ar < d && k0 + tx + VB_GT < d)
              ? A[(uint)ar * d + (k0 + tx + VB_GT)] : 0.0f;
      bt[(uint)t * VB_TILE + tx] =
          (k0 + t < d && bc < d) ? B[(uint)(k0 + t) * d + bc] : 0.0f;
      bt[(uint)t * VB_TILE + tx + VB_GT] =
          (k0 + t < d && bc + VB_GT < d)
              ? B[(uint)(k0 + t) * d + bc + VB_GT] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int kk = 0; kk < VB_TILE; ++kk) {
      const float a0 = at[(uint)ty * VB_TILE + kk];
      const float a1 = at[(uint)(ty + VB_GT) * VB_TILE + kk];
      const float b0 = bt[(uint)kk * VB_TILE + tx];
      const float b1 = bt[(uint)kk * VB_TILE + tx + VB_GT];
      sum[0][0] += a0 * b0; sum[0][1] += a0 * b1;
      sum[1][0] += a1 * b0; sum[1][1] += a1 * b1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const int ii[2] = {i0 + ty, i0 + ty + VB_GT};
  const int jj[2] = {j0 + tx, j0 + tx + VB_GT};
  for (int a = 0; a < 2; ++a) {
    if (ii[a] >= d) { continue; }
    for (int b = 0; b < 2; ++b) {
      if (jj[b] >= d) { continue; }
      const uint off = (uint)ii[a] * d + jj[b];
      C[off] = sum[a][b] + D[off];
    }
  }
}

// Elementwise scale, in place. Exists for exactly one caller: the text
// state is halved because BOTH scan directions start from it and the
// gather adds them, so half each keeps the sum at roughly one copy of
// the prompt while the per-frame video injections stay at full weight.
// Doing it on the GPU keeps the branch's chain unbroken; the tensor is
// [H, d, d] and read once, so the kernel is bandwidth and nothing else.
//
//   0:buf  1:n  2:scale
kernel void vdn_scale_f32(
    device float*   buf   [[buffer(0)]],
    constant int&   n     [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
  if ((int)tid >= n) { return; }
  buf[tid] *= scale;
}

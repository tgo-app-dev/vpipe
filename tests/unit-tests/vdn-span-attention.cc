// The WINDOW on both steel flash kernels, against the CPU reference.
//
// The hybrid's two halves PARTITION the keys: the softmax covers a
// window of whole frames and the linear branch carries exactly the
// complement. So a kernel that masks one frame differently from
// vdn::window_softmax does not produce a slightly worse video, it
// produces one where a frame is counted twice or not at all -- and it
// renders either way. That is what this pins, for both spellings:
//
//   attn_steel_h_bd128_bf16       32x16 blocks, simdgroup ALU (M4)
//   attn_steel_nax_h_bd128_bf16   64x32 blocks, matrix cores  (M5)
//
// SELF-CONTAINED ON PURPOSE. minimax_h3_vdn.block_sparse_steel_matches_
// the_span_reference drives the same kernels through a real H3 stack and
// needs 60 GB of checkpoint to say anything; this needs a GPU. It is the
// test that can run on the box where a kernel is being changed.
//
// The bar is the bf16 accumulation floor rather than equality: the
// reference sums in double over exact spans, steel accumulates on
// register tiles over blocks rounded outward and masked at the edges.
// A mask that differed by one FRAME would be nowhere near it -- a frame
// is 1/25th of this clip's keys.

#include "minitest.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"
#include "generative-models/minimax-h3/vdn-geometry.h"
#include "generative-models/minimax-h3/vdn-window-softmax.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::metal_compute;
namespace vdn = vpipe::genai::minimax_h3::vdn;

namespace {

// The geometry. Chunk 5 / radius 1 is the released config; 25 frames is
// past the (radius + 1) * chunk = 10 at which the window covers the clip
// and the whole hybrid turns itself off.
constexpr int kFrames = 25;
constexpr int kTpf    = 16;      // a 4x4 patch grid
constexpr int kText   = 8;       // global rows, dense in both directions
constexpr int kHeads  = 2;
constexpr int kDim    = 128;     // the only head_dim these entries carry
constexpr int kSeq    = kText + kFrames * kTpf;

float
bf16_(float v)
{
  std::uint32_t u;
  std::memcpy(&u, &v, 4);
  if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0u) {
    u = ((u >> 16) | 0x0040u) << 16;
  } else {
    u = ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16) << 16;
  }
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

std::uint16_t
to_bf16_(float v)
{
  std::uint32_t u;
  std::memcpy(&u, &v, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

float
from_bf16_(std::uint16_t b)
{
  const std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// C++ mirror of mlx::steel::AttnParams, as the transformer keeps one.
struct P {
  int B, H, D, qL, kL, gqa_factor;
  float scale;
  int NQ, NK, NQ_aligned, NK_aligned, qL_rem, kL_rem, qL_off;
  std::int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
};

double
rel_l2_(const std::vector<float>& got, const std::vector<float>& want)
{
  if (got.size() != want.size() || got.empty()) { return -1.0; }
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double d = (double)got[i] - (double)want[i];
    num += d * d;
    den += (double)want[i] * (double)want[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

struct Arm {
  bool ran = false;
  std::vector<float> out;
  std::size_t visited = 0, dense = 0;
};

// The bench's own geometry, which the correctness test's constants
// cannot serve: a production clip is 18887 rows of 56 heads and its
// buffers are a gigabyte.
struct Geom {
  int frames = 37, tpf = 510, text = 17, heads = 56;
  int seq() const { return text + frames * tpf; }
};

// One timed windowed forward, arms INTERLEAVED by the caller. Returns
// the best of `reps` -- the SoC's power state moves under a run and the
// minimum is the least contaminated statistic available here.
double
time_(MetalCompute* mc, ComputeLibrary& lib, const char* name, int bq,
      int bk, bool spans, const Geom& g, const vdn::WindowMask& mask,
      const SharedBuffer& qb, const SharedBuffer& kb_, const SharedBuffer& vb,
      SharedBuffer& ob, int reps)
{
  if (!lib.valid()) { return -1.0; }
  const int seq = g.seq();
  FunctionConstants fc;
  fc.set_bool(200, (seq % bq) == 0).set_bool(201, (seq % bk) == 0)
      .set_bool(300, false).set_bool(301, false).set_bool(302, false)
      .set_bool(303, spans);
  ComputeFunction fn = lib.function(name, fc);
  if (!fn.valid()) { return -1.0; }

  const vdn::BlockSpans bs = vdn::build_block_spans(mask, bq, bk);
  const int NQ = (seq + bq - 1) / bq, NK = (seq + bk - 1) / bk;
  SharedBuffer pb = mc->make_shared_buffer(sizeof(P));
  SharedBuffer off = mc->make_shared_buffer(bs.off.size() * 4);
  SharedBuffer blk = mc->make_shared_buffer(bs.blocks.size() * 4);
  SharedBuffer spp = mc->make_shared_buffer(4 * sizeof(int));
  SharedBuffer spb = mc->make_shared_buffer((std::size_t)g.frames * 2 * 4);
  if (pb.empty() || off.empty() || blk.empty() || spp.empty()
      || spb.empty()) {
    return -1.0;
  }
  std::memcpy(off.contents(), bs.off.data(), bs.off.size() * 4);
  std::memcpy(blk.contents(), bs.blocks.data(), bs.blocks.size() * 4);
  auto* p = static_cast<P*>(pb.contents());
  p->B = 1; p->H = g.heads; p->D = kDim;
  p->qL = seq; p->kL = seq;
  p->gqa_factor = 1;
  p->scale = 1.0f / std::sqrt((float)kDim);
  p->NQ = NQ; p->NK = NK;
  p->NQ_aligned = seq / bq; p->NK_aligned = seq / bk;
  p->qL_rem = seq - p->NQ_aligned * bq;
  p->kL_rem = seq - p->NK_aligned * bk;
  p->qL_off = 0;
  const std::int64_t hm[3] = {(std::int64_t)g.heads * seq * kDim,
                              (std::int64_t)seq * kDim, kDim};
  for (int i = 0; i < 3; ++i) {
    p->Q_strides[i] = hm[i];
    p->K_strides[i] = hm[i];
    p->V_strides[i] = hm[i];
    p->O_strides[i] = hm[i];
  }
  int* sp = static_cast<int*>(spp.contents());
  sp[0] = g.text; sp[1] = g.tpf; sp[2] = g.frames; sp[3] = 3;
  int* bn = static_cast<int*>(spb.contents());
  for (int f = 0; f < g.frames; ++f) {
    bn[2 * f] = mask.bounds[(std::size_t)f].lo;
    bn[2 * f + 1] = mask.bounds[(std::size_t)f].hi;
  }

  double best = -1.0;
  for (int r = 0; r < reps; ++r) {
    CommandStream cs = mc->make_command_stream();
    ComputeEncoder enc = cs.begin_compute();
    enc.set_function(fn);
    enc.set_buffer(0, qb); enc.set_buffer(1, kb_); enc.set_buffer(2, vb);
    enc.set_buffer(3, ob); enc.set_buffer(4, pb);
    if (spans) {
      enc.set_buffer(8, off); enc.set_buffer(9, blk);
      enc.set_buffer(10, spp); enc.set_buffer(11, spb);
    }
    enc.dispatch({32 * (unsigned)NQ, 4 * (unsigned)g.heads, 1}, {32, 4, 1});
    enc.end();
    const auto t0 = std::chrono::steady_clock::now();
    std::string cerr;
    if (!cs.commit().wait_ok(&cerr)) { return -1.0; }
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    if (best < 0.0 || ms < best) { best = ms; }
  }
  return best;
}

// One block-sparse forward on `lib`'s bd128 bf16 entry, at `bq` x `bk`.
Arm
run_(MetalCompute* mc, ComputeLibrary& lib, const char* name, int bq, int bk,
     const vdn::WindowMask& mask, const std::vector<float>& q,
     const std::vector<float>& k, const std::vector<float>& v,
     bool spans = true, int dim = kDim, int seq = kSeq,
     int heads = kHeads)
{
  Arm a;
  if (!lib.valid()) { return a; }
  // Everything else about the geometry comes from the MASK, so a caller
  // may hand in a clip of any length without a second set of constants
  // that has to agree with it.
  const int text = mask.video_start, tpf = mask.tokens_per_frame;
  const int nframes = mask.num_frames;
  FunctionConstants fc;
  fc.set_bool(200, (seq % bq) == 0).set_bool(201, (seq % bk) == 0)
      .set_bool(300, false).set_bool(301, false).set_bool(302, false)
      .set_bool(303, spans);
  ComputeFunction fn = lib.function(name, fc);
  // An UNVALIDATED function is a silent no-op, which would read here as
  // an output of zeros -- and a rel-L2 of 1.0 is a mask bug's answer too.
  if (!fn.valid()) { return a; }

  const vdn::BlockSpans bs = vdn::build_block_spans(mask, bq, bk);
  const int NQ = (seq + bq - 1) / bq, NK = (seq + bk - 1) / bk;
  a.visited = bs.blocks.size();
  a.dense   = (std::size_t)NQ * (std::size_t)NK;

  // Head-major [H, seq, D] -- the unfused layout, so the strides are the
  // plain ones and nothing here depends on the transformer's packing.
  const std::size_t n = (std::size_t)heads * seq * dim;
  SharedBuffer qb = mc->make_shared_buffer(n * 2);
  SharedBuffer kb_ = mc->make_shared_buffer(n * 2);
  SharedBuffer vb = mc->make_shared_buffer(n * 2);
  SharedBuffer ob = mc->make_shared_buffer(n * 2);
  SharedBuffer pb = mc->make_shared_buffer(sizeof(P));
  SharedBuffer off = mc->make_shared_buffer(bs.off.size() * 4);
  SharedBuffer blk = mc->make_shared_buffer(bs.blocks.size() * 4);
  SharedBuffer spp = mc->make_shared_buffer(4 * sizeof(int));
  SharedBuffer spb = mc->make_shared_buffer((std::size_t)nframes * 2 * 4);
  if (qb.empty() || kb_.empty() || vb.empty() || ob.empty() || pb.empty()
      || off.empty() || blk.empty() || spp.empty() || spb.empty()) {
    return a;
  }
  auto pack = [&](const std::vector<float>& src, SharedBuffer& dst) {
    auto* o = static_cast<std::uint16_t*>(dst.contents());
    for (int h = 0; h < heads; ++h) {
      for (int s = 0; s < seq; ++s) {
        for (int c = 0; c < dim; ++c) {
          o[((std::size_t)h * seq + s) * dim + c] =
              to_bf16_(src[((std::size_t)s * heads + h) * dim + c]);
        }
      }
    }
  };
  pack(q, qb);
  pack(k, kb_);
  pack(v, vb);
  std::memset(ob.contents(), 0, ob.byte_size());
  std::memcpy(off.contents(), bs.off.data(), bs.off.size() * 4);
  std::memcpy(blk.contents(), bs.blocks.data(), bs.blocks.size() * 4);

  auto* p = static_cast<P*>(pb.contents());
  p->B = 1; p->H = heads; p->D = dim;
  p->qL = seq; p->kL = seq;
  p->gqa_factor = 1;
  p->scale = 1.0f / std::sqrt((float)dim);
  p->NQ = NQ; p->NK = NK;
  p->NQ_aligned = seq / bq; p->NK_aligned = seq / bk;
  p->qL_rem = seq - p->NQ_aligned * bq;
  p->kL_rem = seq - p->NK_aligned * bk;
  p->qL_off = 0;
  const std::int64_t hm[3] = {(std::int64_t)heads * seq * dim,
                              (std::int64_t)seq * dim, dim};
  for (int i = 0; i < 3; ++i) {
    p->Q_strides[i] = hm[i];
    p->K_strides[i] = hm[i];
    p->V_strides[i] = hm[i];
    p->O_strides[i] = hm[i];
  }
  int* sp = static_cast<int*>(spp.contents());
  sp[0] = text; sp[1] = tpf; sp[2] = nframes;
  sp[3] = 3;                    // anchors BOTH: bit 0 columns, bit 1 rows
  int* bn = static_cast<int*>(spb.contents());
  for (int f = 0; f < nframes; ++f) {
    bn[2 * f] = mask.bounds[(std::size_t)f].lo;
    bn[2 * f + 1] = mask.bounds[(std::size_t)f].hi;
  }

  CommandStream cs = mc->make_command_stream();
  ComputeEncoder enc = cs.begin_compute();
  enc.set_function(fn);
  enc.set_buffer(0, qb); enc.set_buffer(1, kb_); enc.set_buffer(2, vb);
  enc.set_buffer(3, ob); enc.set_buffer(4, pb);
  if (spans) {
    enc.set_buffer(8, off); enc.set_buffer(9, blk);
    enc.set_buffer(10, spp); enc.set_buffer(11, spb);
  } else {
    a.visited = a.dense;               // it walks every key block
  }
  enc.dispatch({32 * (unsigned)NQ, 4 * (unsigned)heads, 1}, {32, 4, 1});
  enc.end();
  std::string cerr;
  if (!cs.commit().wait_ok(&cerr)) { return a; }

  const auto* o = static_cast<const std::uint16_t*>(ob.contents());
  a.out.resize((std::size_t)seq * heads * dim);
  for (int h = 0; h < heads; ++h) {
    for (int s = 0; s < seq; ++s) {
      for (int c = 0; c < dim; ++c) {
        a.out[((std::size_t)s * heads + h) * dim + c] =
            from_bf16_(o[((std::size_t)h * seq + s) * dim + c]);
      }
    }
  }
  a.ran = true;
  return a;
}


// The same forward, but reading a FUSED projection the way the
// transformer does: one [rows, 3 * inner] buffer with q, k and v as
// interleaved fields of a row.
//
// THIS IS THE LAYOUT THE OVERFLOW NEEDS, and the reason the head-major
// runner above cannot show it. A block jump costs
// (blocks) x (block size) x (SEQUENCE STRIDE), and head-major that
// stride is head_dim = 128 -- so the product stays inside 32 bits at any
// length a machine can hold. Fused it is 3 * inner = 21504, which is
// 168x larger, and the product passes 2^31 at a clip this model is
// actually asked for.
//
// The buffer is therefore ~5 GB of ADDRESS SPACE and that is
// irreducible: the jump that overflows is, in elements, the size of the
// region it jumps across. Only the q/k/v lanes are written, so the
// resident set is the pages those touch (one per row) and not the whole
// span.
Arm
run_fused_(MetalCompute* mc, ComputeLibrary& lib, const char* name, int bq,
           int bk, const vdn::WindowMask& mask, const std::vector<float>& q,
           const std::vector<float>& k, const std::vector<float>& v,
           int dim, int seq, int heads, int row_stride)
{
  Arm a;
  if (!lib.valid()) { return a; }
  const int text = mask.video_start, tpf = mask.tokens_per_frame;
  const int nframes = mask.num_frames;
  FunctionConstants fc;
  fc.set_bool(200, (seq % bq) == 0).set_bool(201, (seq % bk) == 0)
      .set_bool(300, false).set_bool(301, false).set_bool(302, false)
      .set_bool(303, true);
  ComputeFunction fn = lib.function(name, fc);
  if (!fn.valid()) { return a; }

  const vdn::BlockSpans bs = vdn::build_block_spans(mask, bq, bk);
  const int NQ = (seq + bq - 1) / bq, NK = (seq + bk - 1) / bk;
  a.visited = bs.blocks.size();
  a.dense   = (std::size_t)NQ * (std::size_t)NK;

  const std::size_t span = (std::size_t)seq * (std::size_t)row_stride;
  SharedBuffer fb = mc->make_shared_buffer(span * 2);
  SharedBuffer ob = mc->make_shared_buffer((std::size_t)heads * seq * dim
                                           * 2);
  SharedBuffer pb = mc->make_shared_buffer(sizeof(P));
  SharedBuffer off = mc->make_shared_buffer(bs.off.size() * 4);
  SharedBuffer blk = mc->make_shared_buffer(bs.blocks.size() * 4);
  SharedBuffer spp = mc->make_shared_buffer(4 * sizeof(int));
  SharedBuffer spb = mc->make_shared_buffer((std::size_t)nframes * 2 * 4);
  if (fb.empty() || ob.empty() || pb.empty() || off.empty() || blk.empty()
      || spp.empty() || spb.empty()) {
    return a;
  }
  auto* f = static_cast<std::uint16_t*>(fb.contents());
  for (int r = 0; r < seq; ++r) {
    const std::size_t base = (std::size_t)r * (std::size_t)row_stride;
    for (int h = 0; h < heads; ++h) {
      for (int c = 0; c < dim; ++c) {
        const std::size_t src = ((std::size_t)r * heads + h) * dim + c;
        const std::size_t lane = (std::size_t)h * dim + c;
        f[base + lane]                                = to_bf16_(q[src]);
        f[base + (std::size_t)heads * dim + lane]     = to_bf16_(k[src]);
        f[base + (std::size_t)2 * heads * dim + lane] = to_bf16_(v[src]);
      }
    }
  }
  std::memset(ob.contents(), 0, ob.byte_size());
  std::memcpy(off.contents(), bs.off.data(), bs.off.size() * 4);
  std::memcpy(blk.contents(), bs.blocks.data(), bs.blocks.size() * 4);

  auto* p = static_cast<P*>(pb.contents());
  p->B = 1; p->H = heads; p->D = dim;
  p->qL = seq; p->kL = seq;
  p->gqa_factor = 1;
  p->scale = 1.0f / std::sqrt((float)dim);
  p->NQ = NQ; p->NK = NK;
  p->NQ_aligned = seq / bq; p->NK_aligned = seq / bk;
  p->qL_rem = seq - p->NQ_aligned * bq;
  p->kL_rem = seq - p->NK_aligned * bk;
  p->qL_off = 0;
  for (int i = 0; i < 3; ++i) {
    p->Q_strides[0] = (std::int64_t)seq * row_stride;
    p->Q_strides[1] = dim;
    p->Q_strides[2] = row_stride;
    p->K_strides[i] = p->Q_strides[i];
    p->V_strides[i] = p->Q_strides[i];
  }
  const std::int64_t hm[3] = {(std::int64_t)heads * seq * dim,
                              (std::int64_t)seq * dim, dim};
  for (int i = 0; i < 3; ++i) { p->O_strides[i] = hm[i]; }
  int* sp = static_cast<int*>(spp.contents());
  sp[0] = text; sp[1] = tpf; sp[2] = nframes; sp[3] = 3;
  int* bn = static_cast<int*>(spb.contents());
  for (int fr = 0; fr < nframes; ++fr) {
    bn[2 * fr] = mask.bounds[(std::size_t)fr].lo;
    bn[2 * fr + 1] = mask.bounds[(std::size_t)fr].hi;
  }

  CommandStream cs = mc->make_command_stream();
  ComputeEncoder enc = cs.begin_compute();
  enc.set_function(fn);
  enc.set_buffer(0, fb);
  enc.set_buffer(1, fb, (std::size_t)heads * dim * 2);
  enc.set_buffer(2, fb, (std::size_t)2 * heads * dim * 2);
  enc.set_buffer(3, ob); enc.set_buffer(4, pb);
  enc.set_buffer(8, off); enc.set_buffer(9, blk);
  enc.set_buffer(10, spp); enc.set_buffer(11, spb);
  enc.dispatch({32 * (unsigned)NQ, 4 * (unsigned)heads, 1}, {32, 4, 1});
  enc.end();
  std::string cerr;
  if (!cs.commit().wait_ok(&cerr)) { return a; }

  const auto* o = static_cast<const std::uint16_t*>(ob.contents());
  a.out.resize((std::size_t)seq * heads * dim);
  for (int h = 0; h < heads; ++h) {
    for (int r = 0; r < seq; ++r) {
      for (int c = 0; c < dim; ++c) {
        a.out[((std::size_t)r * heads + h) * dim + c] =
            from_bf16_(o[((std::size_t)h * seq + r) * dim + c]);
      }
    }
  }
  a.ran = true;
  return a;
}

}  // namespace

TEST(vdn_span_attention, both_steel_kernels_mask_the_window_the_same_way)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  vdn::WindowMask mask;
  mask.seq_len          = kSeq;
  mask.video_start      = kText;
  mask.num_frames       = kFrames;
  mask.tokens_per_frame = kTpf;
  mask.anchors          = vdn::AnchorFrames::kBoth;
  mask.bounds           = vdn::window_bounds(kFrames, 1, 5);
  ASSERT_TRUE((int)mask.bounds.size() == kFrames);
  // The window must not cover the clip, or every arm agrees for the
  // uninteresting reason that nothing is masked.
  bool windowed = false;
  for (const vdn::Bound& b : mask.bounds) {
    if (b.lo > 0 || b.hi < kFrames - 1) { windowed = true; break; }
  }
  ASSERT_TRUE(windowed);

  // bf16-ROUNDED inputs, so the reference and the kernels see the same
  // numbers and the only difference left is the arithmetic.
  const std::size_t n = (std::size_t)kSeq * kHeads * kDim;
  std::vector<float> q(n), k(n), v(n);
  for (std::size_t i = 0; i < n; ++i) {
    q[i] = bf16_(std::sin((float)i * 0.013f) * 0.5f);
    k[i] = bf16_(std::cos((float)i * 0.017f) * 0.5f);
    v[i] = bf16_(std::sin((float)i * 0.007f + 1.3f) * 0.5f);
  }
  std::vector<float> want(n, 0.0f);
  vdn::window_softmax(q.data(), k.data(), v.data(), mask, kHeads, kDim,
                      1.0f / std::sqrt((float)kDim), want.data());

  ComputeLibrary alu = mc->load_library("attn_steel");
  ComputeLibrary nax = mc->load_library("attn_steel_nax");
  const Arm a = run_(mc, alu, "attn_steel_h_bd128_bf16", 32, 16, mask, q, k,
                     v);
  ASSERT_TRUE(a.ran);
  if (!a.ran) { return; }
  const double rel_a = rel_l2_(a.out, want);
  std::printf("[vdn_spans] ALU 32x16: rel-L2 %.3e, %zu of %zu key blocks "
              "(%.1f%%)\n", rel_a, a.visited, a.dense,
              100.0 * (double)a.visited / (double)a.dense);
  EXPECT_TRUE(rel_a >= 0.0 && rel_a < 0.01);
  // And the window must actually be a window: an arm that visited every
  // block would agree with a DENSE reference, not this one.
  EXPECT_TRUE(a.visited < a.dense);

  // The matrix-core twin. Absent before M5, where the entry point is a
  // stub the loader never binds -- so this half is a skip and not a
  // failure.
  if (!mc->supports_matrix_cores()) {
    std::printf("[vdn_spans] no matrix cores -- NAX arm SKIPPED\n");
    return;
  }
  const Arm b = run_(mc, nax, "attn_steel_nax_h_bd128_bf16", 64, 32, mask,
                     q, k, v);
  ASSERT_TRUE(b.ran);
  if (!b.ran) { return; }
  const double rel_b = rel_l2_(b.out, want);
  const double rel_ab = rel_l2_(b.out, a.out);
  std::printf("[vdn_spans] NAX 64x32: rel-L2 %.3e vs reference, %.3e vs "
              "ALU, %zu of %zu key blocks (%.1f%%)\n", rel_b, rel_ab,
              b.visited, b.dense,
              100.0 * (double)b.visited / (double)b.dense);
  EXPECT_TRUE(rel_b >= 0.0 && rel_b < 0.01);
  EXPECT_TRUE(b.visited < b.dense);
  // The two kernels are one predicate on two tile sizes, so they must
  // agree with each OTHER at least as well as either agrees with the
  // reference -- that is the statement that the port did not change the
  // window, as opposed to landing inside a loose tolerance by luck.
  EXPECT_TRUE(rel_ab >= 0.0 && rel_ab < 0.01);
}

TEST(vdn_span_attention, the_coarser_matrix_core_tiling_costs_little)
{
  // WHAT THE 64x32 TILING COSTS, at the geometry a generation uses.
  //
  // The block list is rounded OUTWARD to whole key blocks, so bigger
  // blocks admit more keys the mask then throws away -- and a query
  // block of 64 rows takes the UNION of two frames' windows whenever it
  // straddles a frame. Both are real, and both are a property of the
  // TOKENS PER FRAME: at 510 a frame is 16 key blocks of 32, so the
  // rounding is a sixteenth of a frame at each edge and 1 query block in
  // 8 straddles. At the 16-token geometry of the test above the same two
  // effects are enormous (74.6% -> 84.6% of the blocks), which is why
  // that number must not be read as the cost of the port.
  //
  // CPU only: this asks what the window ADMITS, which is the host's
  // answer and does not need a GPU to compute.
  const int frames = 37, tpf = 510, text = 17;
  const int seq = text + frames * tpf;
  vdn::WindowMask mask;
  mask.seq_len          = seq;
  mask.video_start      = text;
  mask.num_frames       = frames;
  mask.tokens_per_frame = tpf;
  mask.anchors          = vdn::AnchorFrames::kBoth;
  mask.bounds           = vdn::window_bounds(frames, 1, 5);
  ASSERT_TRUE((int)mask.bounds.size() == frames);

  auto share = [&](int bq, int bk) {
    const vdn::BlockSpans bs = vdn::build_block_spans(mask, bq, bk);
    const double dense = (double)((seq + bq - 1) / bq)
                         * (double)((seq + bk - 1) / bk);
    const double got = (double)bs.blocks.size();
    std::printf("[vdn_spans] %dx%d: %.0f of %.0f key blocks (%.1f%% of "
                "dense)\n", bq, bk, got, dense, 100.0 * got / dense);
    return got / dense;
  };
  const double alu = share(32, 16);
  const double nax = share(64, 32);
  ASSERT_TRUE(alu > 0.0);
  if (alu <= 0.0) { return; }
  std::printf("[vdn_spans] the 64x32 tiling admits %.2fx the work\n",
              nax / alu);
  // Both must still be a WINDOW -- a tiling that admitted everything
  // would make the hybrid double count, since the linear branch is
  // carrying the complement either way.
  EXPECT_TRUE(alu < 0.5 && nax < 0.5);
  // And the coarser one must stay close: this is the whole reason the
  // window may ride the matrix cores at all. 1.1x is far above the
  // measured 1.02x and far below the 1.13x of the 16-token geometry.
  EXPECT_TRUE(nax < alu * 1.1);
}

TEST(vdn_span_attention, the_dense_specialisation_is_untouched)
{
  // THE OTHER ELEVEN CALLERS. has_spans is a new function-constant slot
  // on an entry point six DiTs, two encoders and an LM already dispatch,
  // and every one of them sets 200..302 and stops. Read through
  // is_function_constant_defined an unset slot means false, so their
  // instantiations should be byte-for-byte what they were -- this is
  // the test that says so rather than the audit that assumes it.
  //
  // A window whose bounds reach every frame IS dense attention, so the
  // same CPU reference serves as the oracle and no second one is needed.
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  vdn::WindowMask mask;
  mask.seq_len          = kSeq;
  mask.video_start      = kText;
  mask.num_frames       = kFrames;
  mask.tokens_per_frame = kTpf;
  mask.anchors          = vdn::AnchorFrames::kNone;
  mask.bounds.assign((std::size_t)kFrames, vdn::Bound{0, kFrames - 1});

  const std::size_t n = (std::size_t)kSeq * kHeads * kDim;
  std::vector<float> q(n), k(n), v(n);
  for (std::size_t i = 0; i < n; ++i) {
    q[i] = bf16_(std::sin((float)i * 0.013f) * 0.5f);
    k[i] = bf16_(std::cos((float)i * 0.017f) * 0.5f);
    v[i] = bf16_(std::sin((float)i * 0.007f + 1.3f) * 0.5f);
  }
  std::vector<float> want(n, 0.0f);
  vdn::window_softmax(q.data(), k.data(), v.data(), mask, kHeads, kDim,
                      1.0f / std::sqrt((float)kDim), want.data());

  ComputeLibrary alu = mc->load_library("attn_steel");
  ComputeLibrary nax = mc->load_library("attn_steel_nax");
  const Arm a = run_(mc, alu, "attn_steel_h_bd128_bf16", 32, 16, mask, q, k,
                     v, false);
  ASSERT_TRUE(a.ran);
  if (a.ran) {
    const double rel = rel_l2_(a.out, want);
    std::printf("[vdn_spans] ALU dense: rel-L2 %.3e\n", rel);
    EXPECT_TRUE(rel >= 0.0 && rel < 0.01);
  }
  if (!mc->supports_matrix_cores()) { return; }
  const Arm b = run_(mc, nax, "attn_steel_nax_h_bd128_bf16", 64, 32, mask,
                     q, k, v, false);
  ASSERT_TRUE(b.ran);
  if (b.ran) {
    const double rel = rel_l2_(b.out, want);
    std::printf("[vdn_spans] NAX dense: rel-L2 %.3e\n", rel);
    EXPECT_TRUE(rel >= 0.0 && rel < 0.01);
  }

  // The OTHER head_dim. bd64 is what the Qwen3-VL vision tower, the
  // Qwen3-ASR audio encoder and the H3 video VAE dispatch, so it is a
  // different instantiation of the header this change edited -- and the
  // one whose callers are furthest from anything VDN.
  const int d64 = 64;
  std::vector<float> q6(n), k6(n), v6(n), w6(n, 0.0f);
  for (std::size_t i = 0; i < n; ++i) {
    q6[i] = q[i];
    k6[i] = k[i];
    v6[i] = v[i];
  }
  q6.resize((std::size_t)kSeq * kHeads * d64);
  k6.resize(q6.size());
  v6.resize(q6.size());
  w6.resize(q6.size());
  vdn::window_softmax(q6.data(), k6.data(), v6.data(), mask, kHeads, d64,
                      1.0f / std::sqrt((float)d64), w6.data());
  const Arm c = run_(mc, nax, "attn_steel_nax_h_bd64_bf16", 64, 32, mask,
                     q6, k6, v6, false, d64);
  ASSERT_TRUE(c.ran);
  if (c.ran) {
    const double rel = rel_l2_(c.out, w6);
    std::printf("[vdn_spans] NAX dense bd64: rel-L2 %.3e\n", rel);
    EXPECT_TRUE(rel >= 0.0 && rel < 0.01);
  }
}

TEST(vdn_span_attention, span_bench)
{
  // THE WINDOW ON THE MATRIX CORES, at generation geometry. Opt-in
  // (VPIPE_VDN_SPAN_BENCH) because it allocates a gigabyte of q/k/v and
  // asserts nothing about speed.
  //
  // Three arms, INTERLEAVED, because a run of one arm followed by a run
  // of the other measures the SoC power state as much as the kernel --
  // the mistake recorded in mma-tile.h and repeated since:
  //
  //   dense on the ALU kernel     what an M4 runs without a branch
  //   windowed on the ALU kernel  what an M5 ran WITH one, before this
  //   windowed on the NAX kernel  what an M5 runs now
  //
  // Defaults are the docs pipeline at 960x544, 120 frames: 37 latent
  // frames of 510 tokens, 56 heads. VPIPE_VDN_SPAN_BENCH_{F,TPF,REPS}.
  if (std::getenv("VPIPE_VDN_SPAN_BENCH") == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto envi = [](const char* k, int d) {
    const char* v = std::getenv(k);
    return (v != nullptr && *v != '\0') ? std::atoi(v) : d;
  };
  Geom g;
  g.frames = envi("VPIPE_VDN_SPAN_BENCH_F", 37);
  g.tpf    = envi("VPIPE_VDN_SPAN_BENCH_TPF", 510);
  const int reps = envi("VPIPE_VDN_SPAN_BENCH_REPS", 3);
  const int seq = g.seq();

  vdn::WindowMask mask;
  mask.seq_len          = seq;
  mask.video_start      = g.text;
  mask.num_frames       = g.frames;
  mask.tokens_per_frame = g.tpf;
  mask.anchors          = vdn::AnchorFrames::kBoth;
  mask.bounds           = vdn::window_bounds(g.frames, 1, 5);

  const std::size_t n = (std::size_t)g.heads * seq * kDim;
  SharedBuffer qb = mc->make_shared_buffer(n * 2);
  SharedBuffer kb_ = mc->make_shared_buffer(n * 2);
  SharedBuffer vb = mc->make_shared_buffer(n * 2);
  SharedBuffer ob = mc->make_shared_buffer(n * 2);
  if (qb.empty() || kb_.empty() || vb.empty() || ob.empty()) {
    std::printf("[vdn_spans] cannot allocate %.2f GB\n",
                (double)(4 * n * 2) / 1073741824.0);
    return;
  }
  // Not denormal and not NaN: the arithmetic's rate depends on both,
  // the answer does not matter here.
  auto fill = [&](SharedBuffer& b, float kk) {
    auto* o = static_cast<std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) {
      o[i] = to_bf16_(std::sin((float)(i % 9973) * kk) * 0.5f);
    }
  };
  fill(qb, 0.013f);
  fill(kb_, 0.017f);
  fill(vb, 0.007f);
  std::memset(ob.contents(), 0, ob.byte_size());

  ComputeLibrary alu = mc->load_library("attn_steel");
  ComputeLibrary nax = mc->load_library("attn_steel_nax");
  const bool have_nax = mc->supports_matrix_cores() && nax.valid();
  std::printf("[vdn_spans] %d frames x %d tokens = %d rows, %d heads, "
              "%.2f GB of q/k/v/o%s\n", g.frames, g.tpf, seq, g.heads,
              (double)(4 * n * 2) / 1073741824.0,
              have_nax ? "" : " -- NO MATRIX CORES");

  double d_alu = -1, w_alu = -1, d_nax = -1, w_nax = -1;
  for (int r = 0; r < reps; ++r) {
    const double a = time_(mc, alu, "attn_steel_h_bd128_bf16", 32, 16,
                           false, g, mask, qb, kb_, vb, ob, 1);
    const double b = time_(mc, alu, "attn_steel_h_bd128_bf16", 32, 16,
                           true, g, mask, qb, kb_, vb, ob, 1);
    double c = -1, d = -1;
    if (have_nax) {
      c = time_(mc, nax, "attn_steel_nax_h_bd128_bf16", 64, 32, false, g,
                mask, qb, kb_, vb, ob, 1);
      d = time_(mc, nax, "attn_steel_nax_h_bd128_bf16", 64, 32, true, g,
                mask, qb, kb_, vb, ob, 1);
    }
    if (r == 0) { continue; }              // pipelines, not the timing
    auto keep = [](double& best, double got) {
      if (got >= 0.0 && (best < 0.0 || got < best)) { best = got; }
    };
    keep(d_alu, a);
    keep(w_alu, b);
    keep(d_nax, c);
    keep(w_nax, d);
  }
  auto line = [&](const char* what, double ms, double ref) {
    if (ms < 0.0) { return; }
    std::printf("[vdn_spans]   %-26s %8.1f ms", what, ms);
    if (ref > 0.0 && ms > 0.0) { std::printf("   %.2fx", ref / ms); }
    std::printf("\n");
  };
  line("dense, ALU 32x16", d_alu, -1.0);
  line("windowed, ALU 32x16", w_alu, d_alu);
  line("dense, NAX 64x32", d_nax, d_alu);
  line("windowed, NAX 64x32", w_nax, d_alu);
  if (w_alu > 0.0 && w_nax > 0.0) {
    std::printf("[vdn_spans] the window on the matrix cores: %.2fx the "
                "window on the ALU kernel\n", w_alu / w_nax);
  }
}

TEST(vdn_span_attention, long_sequences_do_not_wrap_a_32_bit_index)
{
  // A CLIP LONG ENOUGH TO OVERFLOW THE BLOCK JUMPS, at exactly the
  // rows an overflow would land on.
  //
  // The dense flash loop steps ONE key block at a time and accumulates
  // through the pointer, so its increment is small however long the
  // clip. A block-sparse one jumps by the GAP between visited blocks,
  // which for a query block's FIRST visited block is the block index
  // itself -- and that product is stride x block-size x index. At video
  // geometry (a fused projection, so the row stride is 3 * inner =
  // 21504) it passes 2^31 at 3120 key blocks on the matrix-core kernel
  // and 6241 on the ALU one: 99840 and 99856 rows, both inside a clip
  // this model is asked for. It wraps SILENTLY -- the loads come back
  // from elsewhere in the same buffer, finite and plausible.
  //
  // So the rows this checks are the LAST frames', whose windows start
  // latest and whose first jump is therefore largest, and it checks
  // them against an exact per-row reference rather than against the
  // other kernel: two kernels that wrap at different lengths would
  // disagree, but two that wrap the same way would not.
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  auto envi = [](const char* k, int d) {
    const char* v = std::getenv(k);
    return (v != nullptr && *v != '\0') ? std::atoi(v) : d;
  };
  // 510 tokens a frame is the docs pipeline's 960x544, so 200 frames is
  // 102017 rows -- PAST BOTH WRAP POINTS (3188 key blocks of 32 against
  // the matrix-core kernel's 3120, 6377 of 16 against the ALU's 6241),
  // which is the only length at which this test is evidence. At 160
  // frames / 81617 rows both kernels are still inside 32 bits and pass
  // whether the arithmetic is widened or not.
  const int frames = envi("VPIPE_VDN_LONG_FRAMES", 230);
  // The row stride the transformer's FUSED projection has: 3 * inner,
  // inner = 56 heads x 128. It is what makes the jump product large, so
  // the arm that uses it is the only one that can fail.
  const int fused_rs = envi("VPIPE_VDN_LONG_STRIDE", 3 * 56 * 128);
  const int tpf = 510, text = 17, heads = 2, dim = 128;
  const int seq = text + frames * tpf;

  vdn::WindowMask mask;
  mask.seq_len          = seq;
  mask.video_start      = text;
  mask.num_frames       = frames;
  mask.tokens_per_frame = tpf;
  mask.anchors          = vdn::AnchorFrames::kBoth;
  mask.bounds           = vdn::window_bounds(frames, 1, 5);
  ASSERT_TRUE((int)mask.bounds.size() == frames);

  const std::size_t n = (std::size_t)seq * heads * dim;
  std::vector<float> q(n), k(n), v(n);
  std::uint32_t st = 7u;
  for (std::size_t i = 0; i < n; ++i) {
    auto nxt = [&]() {
      st = st * 1664525u + 1013904223u;
      return (float)(st >> 8) / 8388608.0f - 1.0f;
    };
    q[i] = bf16_(nxt() * 0.5f);
    k[i] = bf16_(nxt() * 0.5f);
    v[i] = bf16_(nxt() * 0.5f);
  }
  std::printf("[vdn_spans] %d frames x %d = %d rows, %d heads, %.2f GB "
              "of q/k/v/o\n", frames, tpf, seq, heads,
              (double)(4 * n * 2) / 1073741824.0);

  // The rows to check: the last frame's (largest first jump), the
  // middle, and the first video row. Exact, one row at a time, so the
  // reference is O(seq) per row rather than O(seq^2).
  // THE LAST FRAME IS AN ANCHOR ROW and sees everything, so its query
  // blocks visit every key block and jump nowhere. The rows that make
  // the largest jump are the last NON-anchor frame's: their window
  // starts latest of any windowed row. Sampling the very last rows
  // instead -- the obvious choice -- measures the one case that cannot
  // fail, which is how a 1.12 x 2^31 jump went unnoticed here once.
  const int last_windowed = frames - 2;
  std::vector<int> rows;
  const int lw0 = text + last_windowed * tpf;
  for (int r = lw0; r < lw0 + 3; ++r) { rows.push_back(r); }
  rows.push_back(lw0 + tpf - 1);
  rows.push_back(seq - 1);                     // an anchor row: dense
  rows.push_back(text + (frames / 2) * tpf);
  rows.push_back(text);
  rows.push_back(text - 1);                    // a global row: dense
  auto ref_row = [&](int r, int h, std::vector<float>* out) {
    const float scale = 1.0f / std::sqrt((float)dim);
    double mx = -1e30;
    std::vector<double> w((std::size_t)seq, 0.0);
    for (int c = 0; c < seq; ++c) {
      if (!mask.allows(r, c)) { w[(std::size_t)c] = -1e30; continue; }
      double dot = 0.0;
      for (int e = 0; e < dim; ++e) {
        dot += (double)q[((std::size_t)r * heads + h) * dim + e]
               * (double)k[((std::size_t)c * heads + h) * dim + e];
      }
      dot *= (double)scale;
      w[(std::size_t)c] = dot;
      if (dot > mx) { mx = dot; }
    }
    double sum = 0.0;
    for (int c = 0; c < seq; ++c) {
      w[(std::size_t)c] = w[(std::size_t)c] <= -1e29
                              ? 0.0
                              : std::exp(w[(std::size_t)c] - mx);
      sum += w[(std::size_t)c];
    }
    out->assign((std::size_t)dim, 0.0f);
    if (sum <= 0.0) { return; }
    for (int c = 0; c < seq; ++c) {
      const double p = w[(std::size_t)c] / sum;
      if (p == 0.0) { continue; }
      for (int e = 0; e < dim; ++e) {
        (*out)[(std::size_t)e] +=
            (float)(p * (double)v[((std::size_t)c * heads + h) * dim + e]);
      }
    }
  };

  ComputeLibrary alu = mc->load_library("attn_steel");
  ComputeLibrary nax = mc->load_library("attn_steel_nax");
  struct Arm2 { const char* what; ComputeLibrary* lib; const char* fn;
                int bq; int bk; };
  const Arm2 arms[] = {
      {"ALU 32x16", &alu, "attn_steel_h_bd128_bf16", 32, 16},
      {"NAX 64x32", &nax, "attn_steel_nax_h_bd128_bf16", 64, 32},
  };
  bool ran_any = false;
  for (const Arm2& a : arms) {
    if (a.bq == 64 && !mc->supports_matrix_cores()) {
      std::printf("[vdn_spans] no matrix cores -- %s SKIPPED\n", a.what);
      continue;
    }
    // BOTH LAYOUTS. Head-major is the cheap one and checks the mask at
    // length; fused is the one whose sequence stride can overflow a
    // 32-bit jump, and the only one that is evidence about it.
    for (int fused = 0; fused < 2; ++fused) {
    const Arm got = fused == 0
        ? run_(mc, *a.lib, a.fn, a.bq, a.bk, mask, q, k, v, true, dim, seq,
               heads)
        : run_fused_(mc, *a.lib, a.fn, a.bq, a.bk, mask, q, k, v, dim, seq,
                     heads, fused_rs);
    if (fused == 1 && !got.ran) {
      std::printf("[vdn_spans]   %s fused: could not allocate %.2f GB -- "
                  "SKIPPED\n", a.what,
                  (double)((std::size_t)seq * fused_rs * 2) / 1073741824.0);
      continue;
    }
    ASSERT_TRUE(got.ran);
    if (!got.ran) { continue; }
    ran_any = true;
    double worst = 0.0;
    int worst_row = -1;
    bool finite = true;
    for (const int r : rows) {
      for (int h = 0; h < heads; ++h) {
        std::vector<float> want;
        ref_row(r, h, &want);
        double num = 0.0, den = 0.0;
        for (int e = 0; e < dim; ++e) {
          const double g =
              (double)got.out[((std::size_t)r * heads + h) * dim + e];
          if (!std::isfinite(g)) { finite = false; }
          const double d2 = g - (double)want[(std::size_t)e];
          num += d2 * d2;
          den += (double)want[(std::size_t)e] * (double)want[(std::size_t)e];
        }
        const double rel = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
        if (rel > worst) { worst = rel; worst_row = r; }
      }
    }
    // The largest jump any query block makes, which is what overflows:
    // the gap from the global rows in block 0 to the first block of its
    // own window.
    const std::size_t jump =
        (std::size_t)(mask.video_start
                      + mask.bounds[(std::size_t)last_windowed].lo * tpf)
        / (std::size_t)a.bk * (std::size_t)a.bk
        * (std::size_t)(fused == 1 ? fused_rs : dim);
    std::printf("[vdn_spans]   %-9s %-10s worst rel-L2 %.3e over %zu rows "
                "(row %d), %zu/%zu blocks, largest jump %.2f x 2^31\n",
                a.what, fused == 1 ? "FUSED:" : "head-major:", worst,
                rows.size(), worst_row, got.visited, got.dense,
                (double)jump / 2147483648.0);
    EXPECT_TRUE(finite);
    EXPECT_TRUE(worst < 0.02);
    }
  }
  EXPECT_TRUE(ran_any);
}

TEST(vdn_span_attention, a_32_bit_jump_product_wraps_at_a_reachable_clip)
{
  // WHAT THE 32-BIT SPELLING ACTUALLY DID on this toolchain.
  //
  // The block-sparse kernels advance by (gap) * (block) * (stride), all
  // int, and at a long clip with a fused projection that product passes
  // 2^31. Widening it is unambiguously right. But the end-to-end test
  // above passes with OR without the widening, and that is worth
  // pinning down rather than leaving as a coincidence: signed overflow
  // is UB, so LLVM may assume it away and widen the multiply itself,
  // making the narrow spelling correct BY AN ASSUMPTION THE SOURCE DOES
  // NOT STATE -- on this compiler, this version.
  //
  // If this test starts failing, the compiler stopped doing that and
  // the widening in steel_attention_nax.h / loader.h is the only thing
  // standing between a long clip and a pointer 3.8 GB before its
  // buffer.
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  ComputeLibrary lib = mc->load_library("vdn_branch");
  ComputeFunction fn = lib.function("vdn_index_probe");
  if (!fn.valid()) {
    std::printf("[vdn_spans] vdn_index_probe did not validate -- SKIP\n");
    return;
  }
  SharedBuffer ob = mc->make_shared_buffer(3 * sizeof(std::int64_t));
  ASSERT_TRUE(!ob.empty());
  if (ob.empty()) { return; }
  // 3490 blocks of 32 at a stride of 3 * 56 * 128: the largest jump a
  // 230-frame clip makes, and 1.12 x 2^31.
  const int gap = 3490, blk = 32, stride = 3 * 56 * 128;
  std::memset(ob.contents(), 0, ob.byte_size());
  CommandStream cs = mc->make_command_stream();
  ComputeEncoder enc = cs.begin_compute();
  enc.set_function(fn);
  enc.set_buffer(0, ob);
  enc.set_constant(1, gap);
  enc.set_constant(2, blk);
  enc.set_constant(3, stride);
  enc.dispatch({1, 1, 1}, {1, 1, 1});
  enc.end();
  std::string cerr;
  ASSERT_TRUE(cs.commit().wait_ok(&cerr));
  const auto* o = static_cast<const std::int64_t*>(ob.contents());
  const std::int64_t want = (std::int64_t)gap * blk * stride;
  std::printf("[vdn_spans] jump %lld elements (%.2f x 2^31): int spelling "
              "-> %lld, int64 -> %lld, truncated to 32 bits -> %lld\n",
              (long long)want, (double)want / 2147483648.0,
              (long long)o[0], (long long)o[1], (long long)o[2]);
  // The int64 spelling is right by construction; that is the control.
  EXPECT_TRUE(o[1] == want * 2);          // half is 2 bytes
  // AND THE NARROW ONE REALLY DOES WRAP -- it is not saved by the
  // compiler widening the multiply under the no-signed-overflow
  // assumption, which was the first guess and is wrong. The pointer
  // lands 3.8 GB BEFORE its buffer, and the loads that follow read
  // whatever is mapped there.
  EXPECT_TRUE(o[0] != o[1]);
  EXPECT_TRUE(o[0] == (std::int64_t)o[2] * 2);
}

// SageAttention's recipe, measured before any kernel is written for it.
//
// Sage-Attn (Zhang et al., "SageAttention: Accurate 8-Bit Attention for
// Plug-and-play Inference Acceleration", arXiv:2410.02367, thu-ml,
// Apache-2.0) runs the QK^T product of a flash attention in INT8 and
// leaves P*V in FP16. Two things make the int8 half accurate enough to
// be worth doing, and this file measures both on the CPU, because they
// decide whether a kernel is worth writing at all:
//
//   SMOOTHING. K is quantized as K - mean(K), the mean taken over
//   TOKENS. The paper's observation is that a key's outlier is not
//   variation across tokens but a large bias SHARED by all of them, so
//   subtracting it spends the int8 range on the signal instead of on
//   the bias. It is free and exact: subtracting m from every key shifts
//   every score in a row by the same q.m, and softmax is invariant to a
//   per-row shift. Nothing is added back.
//
//   PER-BLOCK SCALES. One scale per query block and one per key block,
//   which is exactly the granularity a flash kernel already tiles at --
//   the score tile's dequant is a single scalar, sq[i] * sk[j], applied
//   where the int32 accumulator becomes float.
//
// WHAT THIS FILE IS NOT. It is not a port -- it is the accuracy half of
// the case for writing one. The speed half is gemm_i8.k512_chunked's
// chunk-depth sweep: at attention's 128-deep contraction, issued back to
// back into a live accumulator the way a flash kernel does, int8 holds
// 20.8 TFLOP/s against f16's 11.0. A standalone K=128 GEMM shows only
// 1.16x and is the wrong shape to ask -- it pays a threadgroup's store
// for every 128 of contraction where a flash loop pays one for the whole
// query block.
//
// So the arithmetic below is what says the 1.9x is SPENDABLE: an int8 QK
// is worth having only if it is still attention afterwards.

#include "minitest.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/session.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kD = 128;      // every DiT attention in this tree

// q/k/v for one head, [T, D], with K carrying a per-channel bias of
// `bias` times the per-token signal's own scale.
//
// THE BIAS IS THE POINT. Sage's smoothing is worth exactly as much as
// this ratio is large, so it is the sweep variable rather than a
// constant: at 0 the method has nothing to remove and the two int8 arms
// must agree, and at 8 it is the regime the paper describes.
struct Head {
  std::vector<float> q, k, v;
  int t = 0;
};

Head
make_head_(int T, float bias, std::uint32_t seed)
{
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  Head h;
  h.t = T;
  h.q.resize((std::size_t)T * kD);
  h.k.resize((std::size_t)T * kD);
  h.v.resize((std::size_t)T * kD);
  std::vector<float> chan_bias(kD);
  for (int c = 0; c < kD; ++c) { chan_bias[c] = nd(rng) * bias; }
  for (int t = 0; t < T; ++t) {
    for (int c = 0; c < kD; ++c) {
      h.q[(std::size_t)t * kD + c] = nd(rng);
      h.k[(std::size_t)t * kD + c] = nd(rng) + chan_bias[c];
      h.v[(std::size_t)t * kD + c] = nd(rng);
    }
  }
  return h;
}

// Attention in double precision: the reference every arm is scored
// against.
// `sink`: a per-head logit in the softmax's denominator alone (no value),
// as attention sinks are; -inf for none.
std::vector<double>
attend_ref_(const Head& h, double scale, double sink = -HUGE_VAL)
{
  const int T = h.t;
  std::vector<double> out((std::size_t)T * kD, 0.0);
  std::vector<double> s((std::size_t)T);
  for (int i = 0; i < T; ++i) {
    double m = sink;
    for (int j = 0; j < T; ++j) {
      double d = 0.0;
      for (int c = 0; c < kD; ++c) {
        d += (double)h.q[(std::size_t)i * kD + c] *
             (double)h.k[(std::size_t)j * kD + c];
      }
      s[(std::size_t)j] = d * scale;
      m = std::max(m, s[(std::size_t)j]);
    }
    double z = std::exp(sink - m);
    for (int j = 0; j < T; ++j) {
      s[(std::size_t)j] = std::exp(s[(std::size_t)j] - m);
      z += s[(std::size_t)j];
    }
    for (int j = 0; j < T; ++j) {
      const double p = s[(std::size_t)j] / z;
      for (int c = 0; c < kD; ++c) {
        out[(std::size_t)i * kD + c] +=
            p * (double)h.v[(std::size_t)j * kD + c];
      }
    }
  }
  return out;
}

// One block's int8 quantization: a single scale for the whole [rows, D]
// tile, which is Sage's per-block granularity.
struct Q8 {
  std::vector<std::int8_t> q;
  std::vector<float>       scale;   // one per block
};

Q8
quant_blocks_(const std::vector<float>& x, int T, int block)
{
  Q8 out;
  out.q.resize(x.size());
  const int nb = (T + block - 1) / block;
  out.scale.assign((std::size_t)nb, 0.0f);
  for (int b = 0; b < nb; ++b) {
    const int r0 = b * block, r1 = std::min(T, r0 + block);
    float am = 0.0f;
    for (int r = r0; r < r1; ++r) {
      for (int c = 0; c < kD; ++c) {
        am = std::max(am, std::fabs(x[(std::size_t)r * kD + c]));
      }
    }
    const float inv = am > 0.0f ? 127.0f / am : 0.0f;
    out.scale[(std::size_t)b] = am / 127.0f;
    for (int r = r0; r < r1; ++r) {
      for (int c = 0; c < kD; ++c) {
        const float qv = std::rint(x[(std::size_t)r * kD + c] * inv);
        out.q[(std::size_t)r * kD + c] =
            (std::int8_t)std::max(-127.0f, std::min(127.0f, qv));
      }
    }
  }
  return out;
}

// The Sage arm: QK^T from int8 with per-block scales (optionally over a
// smoothed K), softmax in float, P*V in float from f16-rounded values.
std::vector<double>
attend_sage_(const Head& h, double scale, int bq, int bk, bool smooth)
{
  const int T = h.t;
  std::vector<float> kk = h.k;
  if (smooth) {
    // K - mean(K) over TOKENS, per channel. Exact under softmax: it
    // shifts every score in a row by the same q.mean.
    std::vector<double> m((std::size_t)kD, 0.0);
    for (int t = 0; t < T; ++t) {
      for (int c = 0; c < kD; ++c) {
        m[(std::size_t)c] += (double)h.k[(std::size_t)t * kD + c];
      }
    }
    for (int c = 0; c < kD; ++c) { m[(std::size_t)c] /= (double)T; }
    for (int t = 0; t < T; ++t) {
      for (int c = 0; c < kD; ++c) {
        kk[(std::size_t)t * kD + c] -= (float)m[(std::size_t)c];
      }
    }
  }
  const Q8 qq = quant_blocks_(h.q, T, bq);
  const Q8 qk = quant_blocks_(kk, T, bk);

  std::vector<double> out((std::size_t)T * kD, 0.0);
  std::vector<double> s((std::size_t)T);
  for (int i = 0; i < T; ++i) {
    const float sq = qq.scale[(std::size_t)(i / bq)];
    double m = -1e300;
    for (int j = 0; j < T; ++j) {
      // int32 accumulation, exactly as the matrix unit would do it.
      std::int32_t acc = 0;
      for (int c = 0; c < kD; ++c) {
        acc += (std::int32_t)qq.q[(std::size_t)i * kD + c] *
               (std::int32_t)qk.q[(std::size_t)j * kD + c];
      }
      const float sk = qk.scale[(std::size_t)(j / bk)];
      s[(std::size_t)j] = (double)acc * (double)sq * (double)sk * scale;
      m = std::max(m, s[(std::size_t)j]);
    }
    double z = 0.0;
    for (int j = 0; j < T; ++j) {
      s[(std::size_t)j] = std::exp(s[(std::size_t)j] - m);
      z += s[(std::size_t)j];
    }
    for (int j = 0; j < T; ++j) {
      // P*V stays FP16, which is the half Sage does not quantize.
      const float p = (_Float16)(float)(s[(std::size_t)j] / z);
      for (int c = 0; c < kD; ++c) {
        out[(std::size_t)i * kD + c] +=
            (double)p * (double)(_Float16)h.v[(std::size_t)j * kD + c];
      }
    }
  }
  return out;
}

// The f16 baseline's OWN error: what the shipped kernel already costs
// against a double-precision reference. An int8 arm is only interesting
// beside this.
std::vector<double>
attend_f16_(const Head& h, double scale)
{
  Head r = h;
  for (auto* v : {&r.q, &r.k, &r.v}) {
    for (float& x : *v) { x = (float)(_Float16)x; }
  }
  return attend_ref_(r, scale);
}

double
cos_sim_(const std::vector<double>& a, const std::vector<double>& b)
{
  double d = 0.0, na = 0.0, nb = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    d += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return (na > 0 && nb > 0) ? d / (std::sqrt(na) * std::sqrt(nb)) : 0.0;
}

double
rel_l1_(const std::vector<double>& a, const std::vector<double>& b)
{
  double n = 0.0, d = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    n += std::fabs(a[i] - b[i]);
    d += std::fabs(b[i]);
  }
  return d > 0 ? n / d : 0.0;
}

}  // namespace

// SMOOTHING IS FREE, and this is why: subtracting a per-channel mean
// from every key shifts every score in a row by the same q.mean, and
// softmax does not see a per-row shift. So there is nothing to add back
// after the product -- the identity is exact, not approximate, and it
// is what lets the quantizer spend its range on the signal.
TEST(sage_attention, smoothing_leaves_the_softmax_unchanged)
{
  const Head h = make_head_(96, /*bias=*/6.0f, 0x5a9eu);
  const double scale = 1.0 / std::sqrt((double)kD);
  const std::vector<double> plain = attend_ref_(h, scale);

  Head sm = h;
  std::vector<double> m((std::size_t)kD, 0.0);
  for (int t = 0; t < h.t; ++t) {
    for (int c = 0; c < kD; ++c) {
      m[(std::size_t)c] += (double)h.k[(std::size_t)t * kD + c];
    }
  }
  for (int c = 0; c < kD; ++c) { m[(std::size_t)c] /= (double)h.t; }
  for (int t = 0; t < h.t; ++t) {
    for (int c = 0; c < kD; ++c) {
      sm.k[(std::size_t)t * kD + c] -= (float)m[(std::size_t)c];
    }
  }
  const std::vector<double> shifted = attend_ref_(sm, scale);
  const double r = rel_l1_(shifted, plain);
  std::printf("[sage] smoothed vs plain attention, both in f64: rel-L1 "
              "%.3e\n", r);
  // Exact but for the subtraction's own rounding into f32.
  EXPECT_TRUE(r < 1e-6);
}

// WHAT THE INT8 HALF COSTS, against the f16 baseline the kernel already
// pays -- and how much of that is the smoothing.
//
// Swept over the bias ratio rather than asserted at one, because the
// method's whole claim is that keys carry a large SHARED bias: at ratio
// 0 there is nothing to smooth and the two int8 arms must agree, and the
// gap has to open as the bias grows. A single number could not tell the
// method working from the fixture being kind.
TEST(sage_attention, per_block_int8_needs_the_smoothing)
{
  const int T = 256, bq = 64, bk = 32;   // the matrix-core kernel's tiles
  const double scale = 1.0 / std::sqrt((double)kD);
  std::printf("[sage] T=%d, blocks %dx%d, head_dim %d\n", T, bq, bk, kD);
  std::printf("[sage] %6s %12s %12s %12s %12s\n", "bias", "f16 cos",
              "i8 raw cos", "i8 smooth", "smooth L1");
  double worst_smoothed = 1.0;
  for (const float bias : {0.0f, 1.0f, 4.0f, 8.0f}) {
    const Head h = make_head_(T, bias, 0x1234u + (unsigned)(bias * 10));
    const std::vector<double> ref = attend_ref_(h, scale);
    const double c_f16 = cos_sim_(attend_f16_(h, scale), ref);
    const std::vector<double> raw = attend_sage_(h, scale, bq, bk, false);
    const std::vector<double> smo = attend_sage_(h, scale, bq, bk, true);
    const double c_raw = cos_sim_(raw, ref), c_smo = cos_sim_(smo, ref);
    const double l_smo = rel_l1_(smo, ref);
    std::printf("[sage] %6.1f %12.6f %12.6f %12.6f %12.4f\n", bias, c_f16,
                c_raw, c_smo, l_smo);
    worst_smoothed = std::min(worst_smoothed, c_smo);
    if (bias == 0.0f) {
      // Nothing to remove: the two int8 arms are the same computation.
      EXPECT_TRUE(std::fabs(c_raw - c_smo) < 5e-4);
    }
    if (bias >= 4.0f) {
      // ...and where there IS a bias, removing it is most of the
      // accuracy. This is the paper's claim, and it is the reason the
      // method is not simply "quantize and hope".
      EXPECT_TRUE(c_smo > c_raw);
    }
  }
  // The bar the method has to clear to be worth a kernel at all: within
  // reach of the f16 path it would replace.
  std::printf("[sage] worst smoothed cosine over the sweep: %.6f\n",
              worst_smoothed);
  EXPECT_TRUE(worst_smoothed > 0.99);
}

namespace {

using vpipe::Session;
using vpipe::metal_compute::CommandStream;
using vpipe::metal_compute::ComputeEncoder;
using vpipe::metal_compute::ComputeFunction;
using vpipe::metal_compute::ComputeLibrary;
using vpipe::metal_compute::FunctionConstants;
using vpipe::metal_compute::MetalCompute;
using vpipe::metal_compute::SharedBuffer;

// AttnParams as the vendored kernel reads it.
struct AttnP {
  int B, H, D, qL, kL, gqa;
  float scale;
  int NQ, NK, NQa, NKa, qL_rem, kL_rem, qL_off;
  std::int64_t Qs[3], Ks[3], Vs[3], Os[3];
};

constexpr int kBQ = 64, kBK = 32;   // attn_steel_nax_h_bd128's tiles
// Token chunks the key mean is reduced over. One thread per channel
// walking the whole sequence has D * BH threads and no parallelism in
// the reduction; this makes it D * BH * kChunks.
constexpr int kChunks = 64;

float
h2f_(std::uint16_t h)
{
  _Float16 v;
  std::memcpy(&v, &h, 2);
  return (float)v;
}

std::uint16_t
f2h_(float f)
{
  const _Float16 v = (_Float16)f;
  std::uint16_t o;
  std::memcpy(&o, &v, 2);
  return o;
}

}  // namespace

// THE INT8 QK PATH IN THE FLASH KERNEL, against the f16 one it replaces.
//
// Same kernel, same inputs, same everything but function constant 306
// and the four buffers it gates. What the comparison has to show is that
// the int8 arm is an ATTENTION and not merely a finite array: it is
// scored against a double-precision reference, and the f16 arm is scored
// against the same reference so the two errors are comparable.
//
// Two sequence lengths, because the tail is where a quantized operand
// can go wrong differently from an f16 one: 1024 divides both tiles and
// 1000 divides neither, so the second runs the kernel's ragged
// load_rows path over int8 rows.
TEST(sage_attention, the_int8_qk_path_matches_the_f16_one)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  if (!mc->supports_matrix_cores()) {
    std::printf("[sage] no matrix cores -- SKIPPED\n");
    return;
  }
  ComputeLibrary lib = mc->load_library("attn_steel_nax");
  ComputeLibrary lq = mc->load_library("sage_quant");
  ComputeFunction f_sum = lq.function("sage_k_sum_chunk");
  ComputeFunction f_red = lq.function("sage_k_mean_reduce");
  ComputeFunction f_quant = lq.function("sage_quant_block");
  if (!lib.valid() || !f_sum.valid() || !f_red.valid() ||
      !f_quant.valid()) {
    std::printf("[sage] kernels unavailable -- skip\n");
    return;
  }

  for (const int T : {1024, 1000}) {
    const int H = 4;
    const double scale = 1.0 / std::sqrt((double)kD);
    const int NQ = (T + kBQ - 1) / kBQ, NK = (T + kBK - 1) / kBK;

    std::vector<Head> heads;
    for (int h = 0; h < H; ++h) {
      heads.push_back(make_head_(T, /*bias=*/5.0f, 0x5a6e0000u + h));
    }
    const std::size_t n = (std::size_t)H * T * kD;
    SharedBuffer qb = mc->make_shared_buffer(n * 2);
    SharedBuffer kb = mc->make_shared_buffer(n * 2);
    SharedBuffer vb = mc->make_shared_buffer(n * 2);
    SharedBuffer ob = mc->make_shared_buffer(n * 2);
    SharedBuffer ob8 = mc->make_shared_buffer(n * 2);
    SharedBuffer q8 = mc->make_shared_buffer(n);
    SharedBuffer k8 = mc->make_shared_buffer(n);
    SharedBuffer qs = mc->make_shared_buffer((std::size_t)H * NQ * 4);
    SharedBuffer ks = mc->make_shared_buffer((std::size_t)H * NK * 4);
    SharedBuffer km = mc->make_shared_buffer((std::size_t)H * kD * 4);
    SharedBuffer kmp =
        mc->make_shared_buffer((std::size_t)H * kChunks * kD * 4);
    SharedBuffer pb = mc->make_shared_buffer(sizeof(AttnP));
    ASSERT_TRUE(!qb.empty() && !ob8.empty() && !q8.empty() && !pb.empty());
    if (pb.empty()) { return; }

    auto* qp = static_cast<std::uint16_t*>(qb.contents());
    auto* kp = static_cast<std::uint16_t*>(kb.contents());
    auto* vp = static_cast<std::uint16_t*>(vb.contents());
    for (int h = 0; h < H; ++h) {
      for (std::size_t i = 0; i < (std::size_t)T * kD; ++i) {
        const std::size_t o = (std::size_t)h * T * kD + i;
        qp[o] = f2h_(heads[(std::size_t)h].q[i]);
        kp[o] = f2h_(heads[(std::size_t)h].k[i]);
        vp[o] = f2h_(heads[(std::size_t)h].v[i]);
      }
    }
    auto* p = static_cast<AttnP*>(pb.contents());
    p->B = 1; p->H = H; p->D = kD; p->qL = T; p->kL = T; p->gqa = 1;
    p->scale = (float)scale;
    p->NQ = NQ; p->NK = NK;
    p->NQa = T / kBQ; p->NKa = T / kBK;
    p->qL_rem = T - p->NQa * kBQ;
    p->kL_rem = T - p->NKa * kBK;
    p->qL_off = 0;
    const std::int64_t hm[3] = {(std::int64_t)H * T * kD,
                                (std::int64_t)T * kD, kD};
    for (int i = 0; i < 3; ++i) {
      p->Qs[i] = hm[i]; p->Ks[i] = hm[i]; p->Vs[i] = hm[i]; p->Os[i] = hm[i];
    }

    auto attn_fn = [&](bool i8) {
      FunctionConstants fc;
      fc.set_bool(200, (T % kBQ) == 0).set_bool(201, (T % kBK) == 0)
          .set_bool(300, false).set_bool(301, false).set_bool(302, false)
          .set_bool(303, false).set_bool(304, false).set_bool(305, false)
          .set_bool(306, i8);
      return lib.function("attn_steel_nax_h_bd128", fc);
    };
    ComputeFunction f_f16 = attn_fn(false);
    ComputeFunction f_i8 = attn_fn(true);
    if (!f_f16.valid() || !f_i8.valid()) {
      std::printf("[sage] attention did not specialise -- skip\n");
      return;
    }

    std::string err;
    {
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder e = st.begin_compute();
        // The f16 arm.
        e.set_function(f_f16);
        e.set_buffer(0, qb); e.set_buffer(1, kb); e.set_buffer(2, vb);
        e.set_buffer(3, ob); e.set_buffer(4, pb);
        e.dispatch({32u * (unsigned)NQ, 4u * (unsigned)H, 1}, {32, 4, 1});
        // The prologue: the key mean, then Q and K quantized per block.
        const unsigned zero = 0;
        e.set_function(f_sum);
        e.set_buffer(0, kb); e.set_buffer(1, kmp);
        e.set_constant(2, T); e.set_constant(3, kD);
        e.set_constant(4, kD); e.set_constant(5, zero);
        e.set_constant(6, kChunks);
        e.set_constant(7, T * kD);        // head stride, head-major
        e.dispatch({(unsigned)kD, (unsigned)H, (unsigned)kChunks},
                   {(unsigned)kD, 1, 1});
        e.set_function(f_red);
        e.set_buffer(0, kmp); e.set_buffer(1, km);
        e.set_constant(2, T); e.set_constant(3, kD);
        e.set_constant(4, kChunks);
        e.dispatch({(unsigned)kD, (unsigned)H, 1}, {(unsigned)kD, 1, 1});
        e.set_function(f_quant);
        e.set_buffer(0, qb); e.set_buffer(1, km); e.set_buffer(2, q8);
        e.set_buffer(3, qs);
        e.set_constant(4, T); e.set_constant(5, kD); e.set_constant(6, kD);
        e.set_constant(7, zero); e.set_constant(8, kBQ);
        e.set_constant(9, 0);
        e.set_constant(10, T * kD);
        e.dispatch({256, (unsigned)H, (unsigned)NQ}, {256, 1, 1});
        e.set_function(f_quant);
        e.set_buffer(0, kb); e.set_buffer(1, km); e.set_buffer(2, k8);
        e.set_buffer(3, ks);
        e.set_constant(4, T); e.set_constant(5, kD); e.set_constant(6, kD);
        e.set_constant(7, zero); e.set_constant(8, kBK);
        e.set_constant(9, 1);          // K is smoothed, Q is not
        e.set_constant(10, T * kD);
        e.dispatch({256, (unsigned)H, (unsigned)NK}, {256, 1, 1});
        // The int8 arm.
        e.set_function(f_i8);
        e.set_buffer(0, qb); e.set_buffer(1, kb); e.set_buffer(2, vb);
        e.set_buffer(3, ob8); e.set_buffer(4, pb);
        e.set_buffer(15, q8); e.set_buffer(16, k8);
        e.set_buffer(17, qs); e.set_buffer(18, ks);
        e.dispatch({32u * (unsigned)NQ, 4u * (unsigned)H, 1}, {32, 4, 1});
      }
      ASSERT_TRUE(st.commit().wait_ok(&err));
    }

    // Scored against a double-precision reference, per head.
    const auto* o16 = static_cast<const std::uint16_t*>(ob.contents());
    const auto* o8 = static_cast<const std::uint16_t*>(ob8.contents());
    double c_f16 = 0.0, c_i8 = 0.0, l_i8 = 0.0;
    for (int h = 0; h < H; ++h) {
      const std::vector<double> ref = attend_ref_(heads[(std::size_t)h], scale);
      std::vector<double> a(ref.size()), b(ref.size());
      for (std::size_t i = 0; i < ref.size(); ++i) {
        a[i] = (double)h2f_(o16[(std::size_t)h * T * kD + i]);
        b[i] = (double)h2f_(o8[(std::size_t)h * T * kD + i]);
      }
      c_f16 += cos_sim_(a, ref) / H;
      c_i8 += cos_sim_(b, ref) / H;
      l_i8 += rel_l1_(b, ref) / H;
    }
    std::printf("[sage] T=%4d: kernel f16 cos %.6f | int8 QK cos %.6f, "
                "rel-L1 %.4f\n", T, c_f16, c_i8, l_i8);
    // The f16 arm is the bar: the int8 one has to land beside it, not
    // merely be finite.
    EXPECT_TRUE(c_f16 > 0.999);
    EXPECT_TRUE(c_i8 > 0.995);
    // ...and it must actually be the int8 path, not the f16 one under a
    // different name.
    EXPECT_TRUE(c_i8 < c_f16);
  }
}

// WHAT THE INT8 QK IS WORTH, at the geometry a video DiT actually runs.
//
// MEASURED on an M5, 8 heads x 20036 rows x 128: f16 156.6 ms against
// 130.6 for the int8 arm INCLUDING its prologue -- 1.20x -- of which the
// prologue is 2.2, so the kernel itself is 128.4 and 1.22x. The ceiling
// is 1.33x (int8 is 2.00x on the pipe and QK is half a flash kernel's
// matrix work), and attention runs at about two thirds of the pipe's
// peak, so this is most of what was there to take.
//
// THE PROLOGUE WAS 8.9 ms AND IS NOW 2.2, and neither number was about
// fusion. 4.33 of the original was the key mean: one thread per channel
// walking the whole sequence, which is D * BH threads -- 1024 -- each
// doing 20036 dependent adds. Chunked over tokens it is 1.4. The
// quantize passes then read their source TWICE, once for the block
// absmax and once to write, which staging the block in threadgroup
// memory folds into one.
//
// So the prologue is now 1.7% of the call, and fusing it into a
// family's transpose would save at most that -- against restructuring a
// kernel that currently owns one row per threadgroup into one that owns
// a whole quantization block. Worth revisiting only if a caller's
// transpose is being rewritten anyway.
//
// The quantization prologue is INSIDE the timed region for the int8 arm,
// because it is work the f16 arm does not do: the key mean is a pass
// over K, and each of Q and K is read once more and written as int8.
// Reported apart as well, since it is amortised differently -- one
// prologue serves every query block, and in a model it could serve
// every step if K did not change.
TEST(sage_attention, int8_qk_bench)
{
  if (std::getenv("VPIPE_SAGE_BENCH") == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid() || !mc->supports_matrix_cores()) {
    return;
  }
  ComputeLibrary lib = mc->load_library("attn_steel_nax");
  ComputeLibrary lq = mc->load_library("sage_quant");
  ComputeFunction f_sum = lq.function("sage_k_sum_chunk");
  ComputeFunction f_red = lq.function("sage_k_mean_reduce");
  ComputeFunction f_quant = lq.function("sage_quant_block");
  if (!lib.valid() || !f_sum.valid() || !f_red.valid() ||
      !f_quant.valid()) {
    return;
  }

  int H = 8, T = 20036;
  if (const char* e = std::getenv("VPIPE_SAGE_HEADS")) { H = std::atoi(e); }
  if (const char* e = std::getenv("VPIPE_SAGE_ROWS")) { T = std::atoi(e); }
  const int NQ = (T + kBQ - 1) / kBQ, NK = (T + kBK - 1) / kBK;
  const std::size_t n = (std::size_t)H * T * kD;

  SharedBuffer qb = mc->make_shared_buffer(n * 2);
  SharedBuffer kb = mc->make_shared_buffer(n * 2);
  SharedBuffer vb = mc->make_shared_buffer(n * 2);
  SharedBuffer ob = mc->make_shared_buffer(n * 2);
  SharedBuffer q8 = mc->make_shared_buffer(n);
  SharedBuffer k8 = mc->make_shared_buffer(n);
  SharedBuffer qs = mc->make_shared_buffer((std::size_t)H * NQ * 4);
  SharedBuffer ks = mc->make_shared_buffer((std::size_t)H * NK * 4);
  SharedBuffer km = mc->make_shared_buffer((std::size_t)H * kD * 4);
  SharedBuffer kmp =
      mc->make_shared_buffer((std::size_t)H * kChunks * kD * 4);
  SharedBuffer pb = mc->make_shared_buffer(sizeof(AttnP));
  if (pb.empty() || ob.empty() || k8.empty()) {
    std::printf("[sage] alloc failed at %d x %d -- skip\n", H, T);
    return;
  }
  std::mt19937 rng(7u);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  for (SharedBuffer* b : {&qb, &kb, &vb}) {
    auto* pp = static_cast<std::uint16_t*>(b->contents());
    for (std::size_t i = 0; i < n; ++i) { pp[i] = f2h_(nd(rng)); }
  }
  auto* p = static_cast<AttnP*>(pb.contents());
  p->B = 1; p->H = H; p->D = kD; p->qL = T; p->kL = T; p->gqa = 1;
  p->scale = 1.0f / std::sqrt((float)kD);
  p->NQ = NQ; p->NK = NK;
  p->NQa = T / kBQ; p->NKa = T / kBK;
  p->qL_rem = T - p->NQa * kBQ;
  p->kL_rem = T - p->NKa * kBK;
  p->qL_off = 0;
  const std::int64_t hm[3] = {(std::int64_t)H * T * kD,
                              (std::int64_t)T * kD, kD};
  for (int i = 0; i < 3; ++i) {
    p->Qs[i] = hm[i]; p->Ks[i] = hm[i]; p->Vs[i] = hm[i]; p->Os[i] = hm[i];
  }

  auto attn_fn = [&](bool i8) {
    FunctionConstants fc;
    fc.set_bool(200, (T % kBQ) == 0).set_bool(201, (T % kBK) == 0)
        .set_bool(300, false).set_bool(301, false).set_bool(302, false)
        .set_bool(303, false).set_bool(304, false).set_bool(305, false)
        .set_bool(306, i8);
    return lib.function("attn_steel_nax_h_bd128", fc);
  };
  ComputeFunction f_f16 = attn_fn(false), f_i8 = attn_fn(true);
  if (!f_f16.valid() || !f_i8.valid()) { return; }

  auto encode_prologue = [&](ComputeEncoder& e) {
    const unsigned zero = 0;
    e.set_function(f_sum);
    e.set_buffer(0, kb); e.set_buffer(1, kmp);
    e.set_constant(2, T); e.set_constant(3, kD);
    e.set_constant(4, kD); e.set_constant(5, zero);
    e.set_constant(6, kChunks);
    e.set_constant(7, T * kD);            // head stride, head-major
    e.dispatch({(unsigned)kD, (unsigned)H, (unsigned)kChunks},
               {(unsigned)kD, 1, 1});
    e.set_function(f_red);
    e.set_buffer(0, kmp); e.set_buffer(1, km);
    e.set_constant(2, T); e.set_constant(3, kD);
    e.set_constant(4, kChunks);
    e.dispatch({(unsigned)kD, (unsigned)H, 1}, {(unsigned)kD, 1, 1});
    e.set_function(f_quant);
    e.set_buffer(0, qb); e.set_buffer(1, km); e.set_buffer(2, q8);
    e.set_buffer(3, qs);
    e.set_constant(4, T); e.set_constant(5, kD); e.set_constant(6, kD);
    e.set_constant(7, zero); e.set_constant(8, kBQ); e.set_constant(9, 0);
    e.set_constant(10, T * kD);
    e.dispatch({256, (unsigned)H, (unsigned)NQ}, {256, 1, 1});
    e.set_function(f_quant);
    e.set_buffer(0, kb); e.set_buffer(1, km); e.set_buffer(2, k8);
    e.set_buffer(3, ks);
    e.set_constant(4, T); e.set_constant(5, kD); e.set_constant(6, kD);
    e.set_constant(7, zero); e.set_constant(8, kBK); e.set_constant(9, 1);
    e.set_constant(10, T * kD);
    e.dispatch({256, (unsigned)H, (unsigned)NK}, {256, 1, 1});
  };
  // The prologue's pieces apart, because "the prologue costs 8.9 ms" is
  // not actionable until it says WHICH.
  auto time_piece = [&](int piece) {          // 0 mean, 1 quant Q, 2 K
    const auto t0 = std::chrono::steady_clock::now();
    CommandStream st = mc->make_command_stream();
    {
      ComputeEncoder e = st.begin_compute();
      const unsigned zero = 0;
      if (piece == 0) {
        e.set_function(f_sum);
        e.set_buffer(0, kb); e.set_buffer(1, kmp);
        e.set_constant(2, T); e.set_constant(3, kD);
        e.set_constant(4, kD); e.set_constant(5, zero);
        e.set_constant(6, kChunks);
        e.set_constant(7, T * kD);        // head stride, head-major
        e.dispatch({(unsigned)kD, (unsigned)H, (unsigned)kChunks},
                   {(unsigned)kD, 1, 1});
        e.set_function(f_red);
        e.set_buffer(0, kmp); e.set_buffer(1, km);
        e.set_constant(2, T); e.set_constant(3, kD);
        e.set_constant(4, kChunks);
        e.dispatch({(unsigned)kD, (unsigned)H, 1}, {(unsigned)kD, 1, 1});
      } else {
        e.set_function(f_quant);
        e.set_buffer(0, piece == 1 ? qb : kb); e.set_buffer(1, km);
        e.set_buffer(2, piece == 1 ? q8 : k8);
        e.set_buffer(3, piece == 1 ? qs : ks);
        e.set_constant(4, T); e.set_constant(5, kD); e.set_constant(6, kD);
        e.set_constant(7, zero);
        e.set_constant(8, piece == 1 ? kBQ : kBK);
        e.set_constant(9, piece == 1 ? 0 : 1);
        e.set_constant(10, T * kD);
        e.dispatch({256, (unsigned)H,
                    (unsigned)(piece == 1 ? NQ : NK)}, {256, 1, 1});
      }
    }
    std::string err;
    if (!st.commit().wait_ok(&err)) { return 0.0; }
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
  };
  auto time = [&](int mode) {                 // 0 f16, 1 int8, 2 prologue
    const auto t0 = std::chrono::steady_clock::now();
    CommandStream st = mc->make_command_stream();
    {
      ComputeEncoder e = st.begin_compute();
      if (mode != 0) { encode_prologue(e); }
      if (mode != 2) {
        e.set_function(mode == 0 ? f_f16 : f_i8);
        e.set_buffer(0, qb); e.set_buffer(1, kb); e.set_buffer(2, vb);
        e.set_buffer(3, ob); e.set_buffer(4, pb);
        if (mode == 1) {
          e.set_buffer(15, q8); e.set_buffer(16, k8);
          e.set_buffer(17, qs); e.set_buffer(18, ks);
        }
        e.dispatch({32u * (unsigned)NQ, 4u * (unsigned)H, 1}, {32, 4, 1});
      }
    }
    std::string err;
    if (!st.commit().wait_ok(&err)) { return 0.0; }
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
  };
  time(0); time(1);                            // warm
  const double ms_f16 = time(0);
  const double ms_i8 = time(1);
  const double ms_pro = time(2);
  const double gflop = 4.0 * H * (double)T * T * kD / 1e9;
  for (int i = 0; i < 3; ++i) { time_piece(i); }        // warm
  std::printf("[sage]   prologue: k-mean %.2f ms, quant Q %.2f, quant K "
              "%.2f\n", time_piece(0), time_piece(1), time_piece(2));
  std::printf("[sage] %d heads x %d rows x 128: f16 %.1f ms (%.0f GF/s)  "
              "int8 QK %.1f ms (%.0f GF/s, prologue %.1f)  speedup %.2fx\n",
              H, T, ms_f16, gflop / (ms_f16 / 1e3), ms_i8,
              gflop / (ms_i8 / 1e3), ms_pro,
              ms_i8 > 0.0 ? ms_f16 / ms_i8 : 0.0);
  EXPECT_TRUE(ms_f16 > 0.0 && ms_i8 > 0.0);
}

namespace {

// bf16 as the bf16 kernels read it: the top half of an f32, rounded to
// nearest even.
std::uint16_t
f2bf_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (std::uint16_t)(u >> 16);
}

float
bf2f_(std::uint16_t b)
{
  const std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

}  // namespace

// P*V IN INT8 TOO (function constants 308 and 310): the product held in
// int32 across key blocks, P as uint8 against the row's running max, V
// with one scale per channel over the sequence as uint8 + 128, Q per
// token -- beside the f16 kernel and SageAttention's int8 QK alone, each
// with the prologue it needs, dispatched by hand from head-major buffers.
//
// Scored against f64 at three sharpnesses (random keys with a shared
// bias, then queries 4x and 8x as large), and on the case the held sum's
// headroom exists for: uniform attention (Q = 0) over 140k keys onto V
// that is +8 for the first half and -8 for the second, whose running sum
// passes 2^30 and has its grid stepped down -- an overrun of int32 there
// is garbage, a grid stepped by 1/2 a bias of every later key.
//
// VPIPE_SAGE_BENCH=1 times the three at a video DiT's geometry (8 heads
// x 20036 rows x 128 unless VPIPE_SAGE_HEADS / _ROWS say otherwise; _BF16
// the bf16 kernels; _SHARP queries scaled; _REPS repeats), each against
// the f16 kernel's output at that length.
TEST(sage_attention, int8_pv_held)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  if (!mc->supports_matrix_cores()) {
    std::printf("[sage-pv8] no matrix cores -- SKIPPED\n");
    return;
  }
  ComputeLibrary lib = mc->load_library("attn_steel_nax");
  ASSERT_TRUE(lib.valid());
  struct Quant {
    ComputeLibrary lib;
    ComputeFunction sum, red, block, token, vam, var, vq8;
  };
  auto load_quant = [&](Quant& f, const char* name) {
    f.lib = mc->load_library(name);
    f.sum = f.lib.function("sage_k_sum_chunk");
    f.red = f.lib.function("sage_k_mean_reduce");
    f.block = f.lib.function("sage_quant_block");
    f.token = f.lib.function("sage_quant_token");
    f.vam = f.lib.function("sage_v_absmax_chunk");
    f.var = f.lib.function("sage_v_absmax_reduce");
    f.vq8 = f.lib.function("sage_quant_v_channel_u8");
  };
  Quant qf;
  load_quant(qf, "sage_quant");
  ASSERT_TRUE(qf.sum.valid() && qf.red.valid() && qf.block.valid() &&
              qf.token.valid() && qf.vam.valid() && qf.var.valid() &&
              qf.vq8.valid());

  // {name, 306 the int8 QK, 308 + 310 the int8 P*V and Q per token}.
  struct Arm {
    const char* name;
    bool qk8, pv8;
  };
  const Arm arms[] = {{"f16", false, false},
                      {"QK8 (Sage)", true, false},
                      {"QK8+PV8", true, true}};

  struct Bufs {
    int H = 0, T = 0, NQ = 0, NK = 0;
    SharedBuffer q, k, v, o, q8, k8, v8, qs, ks, vs, km, vm, part, p;
    SharedBuffer sk;      // per-head sinks (f16), bound where an arm has them
  };
  auto make = [&](int H, int T) {
    Bufs b;
    b.H = H; b.T = T;
    b.NQ = (T + kBQ - 1) / kBQ; b.NK = (T + kBK - 1) / kBK;
    const std::size_t n = (std::size_t)H * T * kD;
    b.q = mc->make_shared_buffer(n * 2); b.k = mc->make_shared_buffer(n * 2);
    b.v = mc->make_shared_buffer(n * 2); b.o = mc->make_shared_buffer(n * 2);
    b.q8 = mc->make_shared_buffer(n); b.k8 = mc->make_shared_buffer(n);
    b.v8 = mc->make_shared_buffer(n);
    b.qs = mc->make_shared_buffer((std::size_t)H * T * 4);
    b.ks = mc->make_shared_buffer((std::size_t)H * b.NK * 4);
    b.vs = mc->make_shared_buffer((std::size_t)H * kD * 4);
    b.km = mc->make_shared_buffer((std::size_t)H * kD * 4);
    b.vm = mc->make_shared_buffer((std::size_t)H * kD * 4);
    b.part = mc->make_shared_buffer((std::size_t)H * kChunks * kD * 4);
    b.p = mc->make_shared_buffer(sizeof(AttnP));
    if (b.p.empty()) { return b; }
    auto* p = static_cast<AttnP*>(b.p.contents());
    p->B = 1; p->H = H; p->D = kD; p->qL = T; p->kL = T; p->gqa = 1;
    p->scale = 1.0f / std::sqrt((float)kD);
    p->NQ = b.NQ; p->NK = b.NK;
    p->NQa = T / kBQ; p->NKa = T / kBK;
    p->qL_rem = T - p->NQa * kBQ;
    p->kL_rem = T - p->NKa * kBK;
    p->qL_off = 0;
    const std::int64_t hm[3] = {(std::int64_t)H * T * kD,
                                (std::int64_t)T * kD, kD};
    for (int i = 0; i < 3; ++i) {
      p->Qs[i] = hm[i]; p->Ks[i] = hm[i]; p->Vs[i] = hm[i]; p->Os[i] = hm[i];
    }
    return b;
  };
  auto fn = [&](const Arm& a, int T, bool bf16, bool sinks = false) {
    FunctionConstants fc;
    fc.set_bool(200, (T % kBQ) == 0).set_bool(201, (T % kBK) == 0)
        .set_bool(300, false).set_bool(301, false).set_bool(302, sinks)
        .set_bool(303, false).set_bool(304, false).set_bool(305, false)
        .set_bool(306, a.qk8).set_bool(308, a.pv8).set_bool(310, a.pv8)
        .set_bool(311, sinks && a.qk8);
    return lib.function(bf16 ? "attn_steel_nax_h_bd128_bf16"
                             : "attn_steel_nax_h_bd128", fc);
  };
  // The prologue an arm needs, then the arm.
  auto encode = [&](ComputeEncoder& e, Bufs& b, const Arm& a,
                    ComputeFunction& f, const Quant& qq) {
    const unsigned zero = 0;
    const unsigned H = (unsigned)b.H, D = (unsigned)kD;
    auto mean = [&](SharedBuffer& src, SharedBuffer& out) {
      e.set_function(qq.sum);
      e.set_buffer(0, src); e.set_buffer(1, b.part);
      e.set_constant(2, b.T); e.set_constant(3, kD);
      e.set_constant(4, kD); e.set_constant(5, zero);
      e.set_constant(6, kChunks); e.set_constant(7, b.T * kD);
      e.dispatch({D, H, (unsigned)kChunks}, {D, 1, 1});
      e.set_function(qq.red);
      e.set_buffer(0, b.part); e.set_buffer(1, out);
      e.set_constant(2, b.T); e.set_constant(3, kD);
      e.set_constant(4, kChunks);
      e.dispatch({D, H, 1}, {D, 1, 1});
    };
    if (a.qk8) {
      mean(b.k, b.km);
      if (a.pv8) {
        e.set_function(qq.token);
        e.set_buffer(0, b.q); e.set_buffer(1, b.q8); e.set_buffer(2, b.qs);
        e.set_constant(3, b.T); e.set_constant(4, kD);
        e.set_constant(5, kD); e.set_constant(6, zero);
        e.set_constant(7, b.T * kD);
        e.set_buffer(8, b.km); e.set_constant(9, 0);
        e.dispatch({256, H, (unsigned)((b.T + 63) / 64)}, {256, 1, 1});
      }
      for (int w = a.pv8 ? 1 : 0; w < 2; ++w) {
        e.set_function(qq.block);
        e.set_buffer(0, w == 0 ? b.q : b.k); e.set_buffer(1, b.km);
        e.set_buffer(2, w == 0 ? b.q8 : b.k8);
        e.set_buffer(3, w == 0 ? b.qs : b.ks);
        e.set_constant(4, b.T); e.set_constant(5, kD);
        e.set_constant(6, kD); e.set_constant(7, zero);
        e.set_constant(8, w == 0 ? kBQ : kBK);
        e.set_constant(9, w == 0 ? 0 : 1);
        e.set_constant(10, b.T * kD);
        e.dispatch({256, H, (unsigned)(w == 0 ? b.NQ : b.NK)}, {256, 1, 1});
      }
    }
    if (a.pv8) {
      mean(b.v, b.vm);
      e.set_function(qq.vam);
      e.set_buffer(0, b.v); e.set_buffer(1, b.vm); e.set_buffer(2, b.part);
      e.set_constant(3, b.T); e.set_constant(4, kD); e.set_constant(5, kD);
      e.set_constant(6, zero); e.set_constant(7, kChunks);
      e.set_constant(8, b.T * kD);
      e.dispatch({D, H, (unsigned)kChunks}, {D, 1, 1});
      e.set_function(qq.var);
      e.set_buffer(0, b.part); e.set_buffer(1, b.vs);
      e.set_constant(2, kD); e.set_constant(3, kChunks);
      e.dispatch({D, H, 1}, {D, 1, 1});
      e.set_function(qq.vq8);
      e.set_buffer(0, b.v); e.set_buffer(1, b.vm); e.set_buffer(2, b.vs);
      e.set_buffer(3, b.v8);
      e.set_constant(4, b.T); e.set_constant(5, kD); e.set_constant(6, kD);
      e.set_constant(7, zero); e.set_constant(8, b.T * kD);
      e.dispatch({D, H, (unsigned)((b.T + 7) / 8)}, {D, 1, 1});
    }
    e.set_function(f);
    e.set_buffer(0, b.q); e.set_buffer(1, b.k); e.set_buffer(2, b.v);
    e.set_buffer(3, b.o); e.set_buffer(4, b.p);
    if (!b.sk.empty()) {
      e.set_buffer(7, b.sk);
      e.set_buffer(22, b.km);   // the sink's shift, under 311
    }
    if (a.qk8) {
      e.set_buffer(15, b.q8); e.set_buffer(16, b.k8);
      e.set_buffer(17, b.qs); e.set_buffer(18, b.ks);
    }
    if (a.pv8) {
      e.set_buffer(19, b.v8); e.set_buffer(20, b.vs); e.set_buffer(21, b.vm);
    }
    e.dispatch({32u * (unsigned)b.NQ, 4u * H, 1}, {32, 4, 1});
  };
  auto run = [&](Bufs& b, const Arm& a, ComputeFunction& f, const Quant& qq) {
    const auto t0 = std::chrono::steady_clock::now();
    CommandStream st = mc->make_command_stream();
    {
      ComputeEncoder e = st.begin_compute();
      encode(e, b, a, f, qq);
    }
    std::string err;
    if (!st.commit().wait_ok(&err)) { return -1.0; }
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
  };
  // Cosine of an output against another, in the tensor dtype's values.
  auto cos_of = [](const Bufs& b, const std::vector<float>& ref, bool bf16) {
    const auto* o = static_cast<const std::uint16_t*>(b.o.contents());
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
      const double x = bf16 ? bf2f_(o[i]) : h2f_(o[i]), y = ref[i];
      if (!std::isfinite(x)) { return -1.0; }
      dot += x * y; na += x * x; nb += y * y;
    }
    return dot / std::sqrt(std::max(1e-30, na * nb));
  };

  // ---- accuracy, against f64, at three sharpnesses
  for (const float sharp : {1.0f, 4.0f, 8.0f}) {
    for (const int T : {1024, 1000}) {
      const int H = 2;
      std::vector<Head> heads;
      for (int h = 0; h < H; ++h) {
        heads.push_back(make_head_(T, /*bias=*/5.0f, 0x5a6e1000u + h));
        for (float& x : heads.back().q) { x *= sharp; }
      }
      const double scale = 1.0 / std::sqrt((double)kD);
      std::vector<std::vector<double>> refs;
      for (const auto& h : heads) { refs.push_back(attend_ref_(h, scale)); }
      Bufs b = make(H, T);
      ASSERT_TRUE(!b.p.empty() && !b.part.empty());
      auto* qp = static_cast<std::uint16_t*>(b.q.contents());
      auto* kp = static_cast<std::uint16_t*>(b.k.contents());
      auto* vp = static_cast<std::uint16_t*>(b.v.contents());
      for (int h = 0; h < H; ++h) {
        for (std::size_t i = 0; i < (std::size_t)T * kD; ++i) {
          const std::size_t o = (std::size_t)h * T * kD + i;
          qp[o] = f2h_(heads[(std::size_t)h].q[i]);
          kp[o] = f2h_(heads[(std::size_t)h].k[i]);
          vp[o] = f2h_(heads[(std::size_t)h].v[i]);
        }
      }
      std::printf("[sage-pv8] x%.0f T=%d:", sharp, T);
      double c_qk8 = 0.0;
      for (const Arm& a : arms) {
        ComputeFunction f = fn(a, T, false);
        ASSERT_TRUE(f.valid());
        if (!f.valid()) { return; }
        ASSERT_TRUE(run(b, a, f, qf) >= 0.0);
        const auto* o = static_cast<const std::uint16_t*>(b.o.contents());
        double c = 0.0;
        for (int h = 0; h < H; ++h) {
          std::vector<double> got(refs[(std::size_t)h].size());
          for (std::size_t i = 0; i < got.size(); ++i) {
            got[i] = (double)h2f_(o[(std::size_t)h * T * kD + i]);
          }
          c += cos_sim_(got, refs[(std::size_t)h]) / H;
        }
        std::printf("  %s cos %.6f |", a.name, c);
        EXPECT_TRUE(c > (a.qk8 ? 0.999 : 0.99999));
        if (a.qk8 && !a.pv8) { c_qk8 = c; }
        // The int8 P*V costs less than the int8 QK does already.
        if (a.pv8) { EXPECT_TRUE(c > c_qk8 - 0.0005); }
      }
      std::printf("\n");
    }
  }

  // ---- the held sum's headroom: uniform attention onto a long run of
  //      same-signed V, 140k keys
  {
    const int H = 1, T = 140000;
    Bufs b = make(H, T);
    ASSERT_TRUE(!b.p.empty() && !b.part.empty());
    if (b.p.empty()) { return; }
    std::mt19937 rng(11u);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const std::size_t n = (std::size_t)H * T * kD;
    auto* qp = static_cast<std::uint16_t*>(b.q.contents());
    auto* kp = static_cast<std::uint16_t*>(b.k.contents());
    auto* vp = static_cast<std::uint16_t*>(b.v.contents());
    for (std::size_t i = 0; i < n; ++i) {
      qp[i] = f2h_(0.0f);
      kp[i] = f2h_(nd(rng));
      const bool first = (int)((i / kD) % (std::size_t)T) < T / 2;
      vp[i] = f2h_(nd(rng) + (first ? 8.0f : -8.0f));
    }
    ComputeFunction f16 = fn(arms[0], T, false);
    ComputeFunction pv8 = fn(arms[2], T, false);
    ASSERT_TRUE(f16.valid() && pv8.valid());
    ASSERT_TRUE(run(b, arms[0], f16, qf) >= 0.0);
    std::vector<float> ref(n);
    const auto* o = static_cast<const std::uint16_t*>(b.o.contents());
    for (std::size_t i = 0; i < n; ++i) { ref[i] = h2f_(o[i]); }
    ASSERT_TRUE(run(b, arms[2], pv8, qf) >= 0.0);
    const double c = cos_of(b, ref, false);
    std::printf("[sage-pv8] uniform over %d keys, V +-8: QK8+PV8 cos %.6f "
                "against f16\n", T, c);
    // V's int8 grid is what is left here (the output, V's mean, is near
    // zero); an overrun of int32 would be noise.
    EXPECT_TRUE(c > 0.99);
  }

  // ---- SINKS (constant 302: a logit per head in the denominator alone --
  //      MAGI-2's preview DiT). The int8 P*V adds V's mean back after the
  //      normalize, and a sink's share of the mass holds no V, so the mean
  //      must come back over the keys' share alone. V carries a large
  //      common offset (+4 a channel) so a missed share shows: the sinks
  //      below take a large part of each row's mass.
  for (const int T : {1024, 1000}) {
    const int H = 2;
    const float sink[2] = {7.0f, 8.0f};
    std::vector<Head> heads;
    for (int h = 0; h < H; ++h) {
      heads.push_back(make_head_(T, /*bias=*/5.0f, 0x51c0000u + h));
      for (float& x : heads.back().v) { x += 4.0f; }
    }
    const double scale = 1.0 / std::sqrt((double)kD);
    std::vector<std::vector<double>> refs;
    for (int h = 0; h < H; ++h) {
      refs.push_back(attend_ref_(heads[(std::size_t)h], scale,
                                 (double)sink[h]));
    }
    Bufs b = make(H, T);
    b.sk = mc->make_shared_buffer((std::size_t)H * 2);
    ASSERT_TRUE(!b.p.empty() && !b.part.empty() && !b.sk.empty());
    if (b.sk.empty()) { return; }
    auto* sp = static_cast<std::uint16_t*>(b.sk.contents());
    for (int h = 0; h < H; ++h) { sp[h] = f2h_(sink[h]); }
    auto* qp = static_cast<std::uint16_t*>(b.q.contents());
    auto* kp = static_cast<std::uint16_t*>(b.k.contents());
    auto* vp = static_cast<std::uint16_t*>(b.v.contents());
    for (int h = 0; h < H; ++h) {
      for (std::size_t i = 0; i < (std::size_t)T * kD; ++i) {
        const std::size_t o = (std::size_t)h * T * kD + i;
        qp[o] = f2h_(heads[(std::size_t)h].q[i]);
        kp[o] = f2h_(heads[(std::size_t)h].k[i]);
        vp[o] = f2h_(heads[(std::size_t)h].v[i]);
      }
    }
    std::printf("[sage-pv8] sinks T=%d:", T);
    double c_qk8 = 0.0;
    for (const Arm& a : arms) {
      ComputeFunction f = fn(a, T, false, /*sinks=*/true);
      ASSERT_TRUE(f.valid());
      if (!f.valid()) { return; }
      ASSERT_TRUE(run(b, a, f, qf) >= 0.0);
      const auto* o = static_cast<const std::uint16_t*>(b.o.contents());
      double c = 0.0, rl = 0.0;
      for (int h = 0; h < H; ++h) {
        std::vector<double> got(refs[(std::size_t)h].size());
        double num = 0.0, den = 0.0;
        for (std::size_t i = 0; i < got.size(); ++i) {
          got[i] = (double)h2f_(o[(std::size_t)h * T * kD + i]);
          const double d = got[i] - refs[(std::size_t)h][i];
          num += d * d;
          den += refs[(std::size_t)h][i] * refs[(std::size_t)h][i];
        }
        c += cos_sim_(got, refs[(std::size_t)h]) / H;
        rl += std::sqrt(num / std::max(den, 1e-30)) / H;
      }
      std::printf("  %s cos %.6f rel %.2e |", a.name, c, rl);
      EXPECT_TRUE(c > (a.qk8 ? 0.999 : 0.99999));
      if (a.qk8 && !a.pv8) { c_qk8 = c; }
      if (a.pv8) {
        EXPECT_TRUE(c > c_qk8 - 0.0005);
        EXPECT_TRUE(rl < 2e-2);
      }
    }
    std::printf("\n");
  }

  // ---- speed
  if (std::getenv("VPIPE_SAGE_BENCH") == nullptr) { return; }
  const bool bf16 = std::getenv("VPIPE_SAGE_BF16") != nullptr;
  Quant qbf;
  if (bf16) { load_quant(qbf, "sage_quant_bf16"); }
  const Quant& qb = bf16 ? qbf : qf;
  ASSERT_TRUE(qb.token.valid() && qb.vq8.valid());
  int H = 8, T = 20036;
  if (const char* e = std::getenv("VPIPE_SAGE_HEADS")) { H = std::atoi(e); }
  if (const char* e = std::getenv("VPIPE_SAGE_ROWS")) { T = std::atoi(e); }
  const float sharp = std::getenv("VPIPE_SAGE_SHARP")
      ? (float)std::atof(std::getenv("VPIPE_SAGE_SHARP")) : 1.0f;
  const int reps = std::getenv("VPIPE_SAGE_REPS")
      ? std::max(1, std::atoi(std::getenv("VPIPE_SAGE_REPS"))) : 3;
  Bufs b = make(H, T);
  if (b.p.empty() || b.part.empty() || b.v8.empty()) {
    std::printf("[sage-pv8] alloc failed -- skip\n");
    return;
  }
  const std::size_t n = (std::size_t)H * T * kD;
  {
    std::mt19937 rng(7u);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (SharedBuffer* s : {&b.q, &b.k, &b.v}) {
      auto* pp = static_cast<std::uint16_t*>(s->contents());
      const float m = s == &b.q ? sharp : 1.0f;
      for (std::size_t i = 0; i < n; ++i) {
        const float x = nd(rng) * m;
        pp[i] = bf16 ? f2bf_(x) : f2h_(x);
      }
    }
  }
  const double gflop = 4.0 * H * (double)T * T * kD / 1e9;
  double base = 0.0;
  std::vector<float> ref;
  for (const Arm& a : arms) {
    ComputeFunction f = fn(a, T, bf16);
    if (!f.valid()) { continue; }
    run(b, a, f, qb);                              // warm
    double best = 1e30;
    for (int i = 0; i < reps; ++i) { best = std::min(best, run(b, a, f, qb)); }
    if (base == 0.0) { base = best; }
    if (ref.empty()) {
      ref.resize(n);
      const auto* o = static_cast<const std::uint16_t*>(b.o.contents());
      for (std::size_t i = 0; i < n; ++i) {
        ref[i] = bf16 ? bf2f_(o[i]) : h2f_(o[i]);
      }
    }
    std::printf("[sage-pv8] %d x %d x 128 %s %-11s %7.1f ms  %6.1f TF/s  "
                "%.2fx f16  cos(f16) %.6f\n", H, T, bf16 ? "bf16" : "f16 ",
                a.name, best, gflop / best, base / best,
                cos_of(b, ref, bf16));
  }
}

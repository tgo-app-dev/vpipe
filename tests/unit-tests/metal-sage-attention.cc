// MetalSageAttention: the DRIVER, as a model actually calls it.
//
// sage-attention.cc next door tests the METHOD and the kernels, dispatched
// by hand from head-major buffers. This file tests the class the four DiT
// families use, and specifically the one thing that separates it from
// those hand dispatches: the source of q and k is NOT head-major. Every
// DiT in this tree reads them in place inside a fused [rows, 3*I]
// projection, where a row is 3*I apart and a head one head_dim apart --
// two strides that are unrelated, where head-major has head = L * row.
//
// That is why the interesting test here is a FUSED source against the
// head-major one, required BIT-IDENTICAL. A stride the driver derived
// instead of carrying would still read inside the buffer, still quantize,
// still produce a picture -- of the wrong head.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"
#include "generative-models/shared/metal-sage-attention.h"
#include "generative-models/shared/sage-attention.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using namespace vpipe::metal_compute;

namespace {

// AttnParams as the vendored kernel reads it.
struct AttnP {
  int B, H, D, qL, kL, gqa;
  float scale;
  int NQ, NK, NQa, NKa, qL_rem, kL_rem, qL_off;
  std::int64_t Qs[3], Ks[3], Vs[3], Os[3];
};

constexpr int kD = 128;

float h2f_(std::uint16_t h)
{
  _Float16 v;
  std::memcpy(&v, &h, 2);
  return (float)v;
}

std::uint16_t f2h_(float f)
{
  const _Float16 v = (_Float16)f;
  std::uint16_t o;
  std::memcpy(&o, &v, 2);
  return o;
}

// A head whose keys carry a large per-channel bias, which is the
// condition the smoothing exists for -- so a driver that dropped the
// mean pass would show up here rather than in the easy case.
void fill_(std::vector<float>* q, std::vector<float>* k,
           std::vector<float>* v, int heads, int tokens, std::uint32_t seed)
{
  const std::size_t n = (std::size_t)heads * tokens * kD;
  q->resize(n); k->resize(n); v->resize(n);
  std::uint32_t s = seed;
  auto rnd = [&]() {
    s = s * 1664525u + 1013904223u;
    return (float)((s >> 8) & 0xffff) / 32768.0f - 1.0f;
  };
  std::vector<float> bias((std::size_t)kD);
  for (int c = 0; c < kD; ++c) { bias[(std::size_t)c] = 4.0f * rnd(); }
  for (int h = 0; h < heads; ++h) {
    for (int t = 0; t < tokens; ++t) {
      for (int c = 0; c < kD; ++c) {
        const std::size_t i =
            ((std::size_t)h * tokens + (std::size_t)t) * kD + (std::size_t)c;
        (*q)[i] = 0.5f * rnd();
        (*k)[i] = 0.5f * rnd() + bias[(std::size_t)c];
        (*v)[i] = rnd();
      }
    }
  }
}

double cos_(const std::vector<double>& a, const std::vector<double>& b)
{
  double d = 0.0, na = 0.0, nb = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i];
  }
  return (na > 0.0 && nb > 0.0) ? d / (std::sqrt(na) * std::sqrt(nb)) : 0.0;
}

// Attention in double, per head, from head-major fp32 sources.
std::vector<double> ref_(const std::vector<float>& q,
                         const std::vector<float>& k,
                         const std::vector<float>& v, int h, int tokens,
                         double scale)
{
  std::vector<double> out((std::size_t)tokens * kD, 0.0);
  std::vector<double> s((std::size_t)tokens);
  const std::size_t base = (std::size_t)h * tokens * kD;
  for (int i = 0; i < tokens; ++i) {
    double m = -1e300;
    for (int j = 0; j < tokens; ++j) {
      double d = 0.0;
      for (int c = 0; c < kD; ++c) {
        d += (double)q[base + (std::size_t)i * kD + c] *
             (double)k[base + (std::size_t)j * kD + c];
      }
      s[(std::size_t)j] = d * scale;
      m = m > s[(std::size_t)j] ? m : s[(std::size_t)j];
    }
    double z = 0.0;
    for (int j = 0; j < tokens; ++j) {
      s[(std::size_t)j] = std::exp(s[(std::size_t)j] - m);
      z += s[(std::size_t)j];
    }
    for (int j = 0; j < tokens; ++j) {
      const double w = s[(std::size_t)j] / z;
      for (int c = 0; c < kD; ++c) {
        out[(std::size_t)i * kD + c] +=
            w * (double)v[base + (std::size_t)j * kD + c];
      }
    }
  }
  return out;
}

// Everything the two arms share: params, the attention function, and the
// dispatch. `sage` null runs the dense arm.
struct Rig {
  MetalCompute* mc = nullptr;
  ComputeLibrary lib;
  int H = 0, T = 0;
  double scale = 0.0;
  SharedBuffer pb;

  bool init(MetalCompute* m, int heads, int tokens)
  {
    mc = m; H = heads; T = tokens;
    scale = 1.0 / std::sqrt((double)kD);
    lib = mc->load_library("attn_steel_nax");
    if (!lib.valid()) { return false; }
    pb = mc->make_shared_buffer(sizeof(AttnP));
    if (pb.empty()) { return false; }
    auto* p = static_cast<AttnP*>(pb.contents());
    const int bq = MetalSageAttention::nax_query_block();
    const int bk = MetalSageAttention::nax_key_block();
    p->B = 1; p->H = H; p->D = kD; p->qL = T; p->kL = T; p->gqa = 1;
    p->scale = (float)scale;
    p->NQ = (T + bq - 1) / bq; p->NK = (T + bk - 1) / bk;
    p->NQa = T / bq; p->NKa = T / bk;
    p->qL_rem = T - p->NQa * bq; p->kL_rem = T - p->NKa * bk;
    p->qL_off = 0;
    return true;
  }

  // Strides are set per arm, because that is what differs between them.
  void set_strides(std::int64_t b, std::int64_t head, std::int64_t row)
  {
    auto* p = static_cast<AttnP*>(pb.contents());
    const std::int64_t s[3] = {b, head, row};
    for (int i = 0; i < 3; ++i) {
      p->Qs[i] = s[i]; p->Ks[i] = s[i]; p->Vs[i] = s[i];
    }
    const std::int64_t hm[3] = {(std::int64_t)H * T * kD,
                                (std::int64_t)T * kD, kD};
    for (int i = 0; i < 3; ++i) { p->Os[i] = hm[i]; }
  }

  // Query blocks the dispatch covers -- the kernel's grid, not sage's.
  int fn_blocks() const
  {
    const int bq = MetalSageAttention::nax_query_block();
    return (T + bq - 1) / bq;
  }

  ComputeFunction fn(bool i8) const
  {
    const int bq = MetalSageAttention::nax_query_block();
    const int bk = MetalSageAttention::nax_key_block();
    FunctionConstants fc;
    fc.set_bool(200, (T % bq) == 0).set_bool(201, (T % bk) == 0)
        .set_bool(300, false).set_bool(301, false).set_bool(302, false)
        .set_bool(303, false).set_bool(304, false).set_bool(305, false)
        .set_bool(sage::kQkInt8Constant, i8);
    return lib.function("attn_steel_nax_h_bd128", fc);
  }
};

}  // namespace

// The class drives the kernels to the same answer the hand dispatch
// does: an attention, scored against a double reference, beside the
// dense arm scored against the same one.
TEST(metal_sage_attention, the_driver_reproduces_the_attention)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  if (!MetalSageAttention::available(mc)) {
    std::printf("[sage] no matrix cores -- SKIPPED\n");
    return;
  }
  std::string err;
  std::unique_ptr<MetalSageAttention> sage =
      MetalSageAttention::load(mc, /*bf16=*/false, &err);
  ASSERT_TRUE(sage != nullptr);
  if (!sage) { return; }

  // 1024 divides both tiles; 1000 divides neither, so the second runs
  // the kernel's ragged load_rows path over int8 rows.
  for (const int T : {1024, 1000}) {
    const int H = 4;
    Rig r;
    if (!r.init(mc, H, T)) { return; }
    ComputeFunction f16 = r.fn(false), f_i8 = r.fn(true);
    if (!f16.valid() || !f_i8.valid()) { return; }

    std::vector<float> q, k, v;
    fill_(&q, &k, &v, H, T, 0x51a6e001u);
    const std::size_t n = (std::size_t)H * T * kD;
    SharedBuffer qb = mc->make_shared_buffer(n * 2);
    SharedBuffer kb = mc->make_shared_buffer(n * 2);
    SharedBuffer vb = mc->make_shared_buffer(n * 2);
    SharedBuffer o0 = mc->make_shared_buffer(n * 2);
    SharedBuffer o1 = mc->make_shared_buffer(n * 2);
    ASSERT_TRUE(!qb.empty() && !o1.empty());
    if (o1.empty()) { return; }
    auto* qp = static_cast<std::uint16_t*>(qb.contents());
    auto* kp = static_cast<std::uint16_t*>(kb.contents());
    auto* vp = static_cast<std::uint16_t*>(vb.contents());
    for (std::size_t i = 0; i < n; ++i) {
      qp[i] = f2h_(q[i]); kp[i] = f2h_(k[i]); vp[i] = f2h_(v[i]);
    }
    r.set_strides((std::int64_t)H * T * kD, (std::int64_t)T * kD, kD);

    MetalSageAttention::Operand qo{&qb, 0, kD, T * kD};
    MetalSageAttention::Operand ko{&kb, 0, kD, T * kD};
    sage::Config cfg;
    cfg.enabled = true;
    {
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder e = st.begin_compute();
        e.set_function(f16);
        e.set_buffer(0, qb); e.set_buffer(1, kb); e.set_buffer(2, vb);
        e.set_buffer(3, o0); e.set_buffer(4, r.pb);
        e.dispatch({32u * (unsigned)r.fn_blocks(), 4u * (unsigned)H, 1},
                   {32, 4, 1});
        ASSERT_TRUE(sage->prepare(
            e, qo, ko, H, H, T, T, kD,
            MetalSageAttention::nax_query_block(),
            MetalSageAttention::nax_key_block(), cfg, &err));
        e.set_function(f_i8);
        e.set_buffer(0, qb); e.set_buffer(1, kb); e.set_buffer(2, vb);
        e.set_buffer(3, o1); e.set_buffer(4, r.pb);
        sage->bind(e);
        e.dispatch({32u * (unsigned)r.fn_blocks(), 4u * (unsigned)H, 1},
                   {32, 4, 1});
      }
      ASSERT_TRUE(st.commit().wait_ok(&err));
    }

    const auto* a = static_cast<const std::uint16_t*>(o0.contents());
    const auto* b = static_cast<const std::uint16_t*>(o1.contents());
    double c0 = 0.0, c1 = 0.0;
    for (int h = 0; h < H; ++h) {
      const std::vector<double> rf = ref_(q, k, v, h, T, r.scale);
      std::vector<double> x(rf.size()), y(rf.size());
      for (std::size_t i = 0; i < rf.size(); ++i) {
        x[i] = (double)h2f_(a[(std::size_t)h * T * kD + i]);
        y[i] = (double)h2f_(b[(std::size_t)h * T * kD + i]);
      }
      c0 += cos_(x, rf) / H;
      c1 += cos_(y, rf) / H;
    }
    // BITWISE, not by cosine, for the question "is this really the int8
    // path". Two cosines that both round to 1.000000 cannot answer it,
    // and a function constant that silently failed to apply would give
    // exactly that -- the dense kernel twice, agreeing perfectly.
    std::size_t differ = 0;
    for (std::size_t i = 0; i < n; ++i) { differ += (a[i] != b[i]) ? 1 : 0; }
    std::printf("[sage-drv] T=%4d: dense cos %.8f | driver int8 cos %.8f, "
                "%zu of %zu elements differ\n", T, c0, c1, differ, n);
    EXPECT_TRUE(c0 > 0.999);
    EXPECT_TRUE(c1 > 0.995);
    EXPECT_TRUE(differ > n / 2);
  }
}

// THE FUSED LAYOUT, which is the one every DiT actually runs.
//
// The same q/k/v written twice -- once head-major, once interleaved into
// a [rows, 3*I] projection with heads side by side -- and the two int8
// outputs required BIT-IDENTICAL. Not "close": the quantizer sees the
// same values in both arms, so any difference is an addressing bug, and
// a tolerance would hide exactly the one this test exists for.
TEST(metal_sage_attention, a_fused_qkv_source_is_the_same_source)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  if (!MetalSageAttention::available(mc)) { return; }
  std::string err;
  std::unique_ptr<MetalSageAttention> sage =
      MetalSageAttention::load(mc, /*bf16=*/false, &err);
  ASSERT_TRUE(sage != nullptr);
  if (!sage) { return; }

  const int H = 4, T = 512;
  const int I = H * kD;                  // the projection's hidden width
  Rig r;
  if (!r.init(mc, H, T)) { return; }
  ComputeFunction f_i8 = r.fn(true);
  if (!f_i8.valid()) { return; }

  std::vector<float> q, k, v;
  fill_(&q, &k, &v, H, T, 0x7c0ffee1u);
  const std::size_t n = (std::size_t)H * T * kD;

  SharedBuffer hm_q = mc->make_shared_buffer(n * 2);
  SharedBuffer hm_k = mc->make_shared_buffer(n * 2);
  SharedBuffer hm_v = mc->make_shared_buffer(n * 2);
  // [rows, 3*I]: q | k | v side by side, heads contiguous within each.
  SharedBuffer fused = mc->make_shared_buffer((std::size_t)T * 3 * I * 2);
  SharedBuffer o_hm = mc->make_shared_buffer(n * 2);
  SharedBuffer o_fu = mc->make_shared_buffer(n * 2);
  ASSERT_TRUE(!fused.empty() && !o_fu.empty());
  if (o_fu.empty()) { return; }

  auto* hq = static_cast<std::uint16_t*>(hm_q.contents());
  auto* hk = static_cast<std::uint16_t*>(hm_k.contents());
  auto* hv = static_cast<std::uint16_t*>(hm_v.contents());
  auto* fu = static_cast<std::uint16_t*>(fused.contents());
  for (int h = 0; h < H; ++h) {
    for (int t = 0; t < T; ++t) {
      for (int c = 0; c < kD; ++c) {
        const std::size_t i =
            ((std::size_t)h * T + (std::size_t)t) * kD + (std::size_t)c;
        hq[i] = f2h_(q[i]); hk[i] = f2h_(k[i]); hv[i] = f2h_(v[i]);
        const std::size_t f = (std::size_t)t * 3 * I +
                              (std::size_t)h * kD + (std::size_t)c;
        fu[f]                     = hq[i];
        fu[f + (std::size_t)I]    = hk[i];
        fu[f + (std::size_t)2 * I] = hv[i];
      }
    }
  }

  sage::Config cfg;
  cfg.enabled = true;
  const int bq = MetalSageAttention::nax_query_block();
  const int bk = MetalSageAttention::nax_key_block();
  auto run = [&](bool fused_src, SharedBuffer& out) {
    MetalSageAttention::Operand qo, ko;
    if (fused_src) {
      // Row stride 3*I, head stride head_dim, base offset the q/k run.
      qo = {&fused, 0, 3 * I, kD};
      ko = {&fused, (std::size_t)I, 3 * I, kD};
      r.set_strides((std::int64_t)T * 3 * I, kD, 3 * I);
    } else {
      qo = {&hm_q, 0, kD, T * kD};
      ko = {&hm_k, 0, kD, T * kD};
      r.set_strides((std::int64_t)H * T * kD, (std::int64_t)T * kD, kD);
    }
    CommandStream st = mc->make_command_stream();
    {
      ComputeEncoder e = st.begin_compute();
      if (!sage->prepare(e, qo, ko, H, H, T, T, kD, bq, bk, cfg, &err)) {
        return false;
      }
      e.set_function(f_i8);
      if (fused_src) {
        e.set_buffer(0, fused, 0);
        e.set_buffer(1, fused, (std::size_t)I * 2);
        e.set_buffer(2, fused, (std::size_t)2 * I * 2);
      } else {
        e.set_buffer(0, hm_q); e.set_buffer(1, hm_k); e.set_buffer(2, hm_v);
      }
      e.set_buffer(3, out);
      e.set_buffer(4, r.pb);
      sage->bind(e);
      e.dispatch({32u * (unsigned)r.fn_blocks(), 4u * (unsigned)H, 1},
                 {32, 4, 1});
    }
    return st.commit().wait_ok(&err);
  };
  // The FUSED arm first, so a driver that cached scratch from the
  // head-major arm cannot pass by reusing it.
  ASSERT_TRUE(run(true, o_fu));
  ASSERT_TRUE(run(false, o_hm));

  const bool same = std::memcmp(o_hm.contents(), o_fu.contents(), n * 2) == 0;
  if (!same) {
    // Report HOW far off, because the two failures look different: a
    // wrong head stride is a large mismatch over most of the tensor, a
    // wrong base offset a total one.
    const auto* a = static_cast<const std::uint16_t*>(o_hm.contents());
    const auto* b = static_cast<const std::uint16_t*>(o_fu.contents());
    std::size_t bad = 0;
    for (std::size_t i = 0; i < n; ++i) { bad += (a[i] != b[i]) ? 1 : 0; }
    std::printf("[sage-drv] fused vs head-major: %zu of %zu differ\n", bad, n);
  }
  EXPECT_TRUE(same);
}

// The scratch estimate is what a caller sizes the box with, so it has to
// be the figure and not a bound: lend exactly it and nothing may fall
// back to a private allocation.
TEST(metal_sage_attention, the_scratch_estimate_matches_the_allocation)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  if (!MetalSageAttention::available(mc)) { return; }
  std::string err;
  std::unique_ptr<MetalSageAttention> sage =
      MetalSageAttention::load(mc, /*bf16=*/false, &err);
  ASSERT_TRUE(sage != nullptr);
  if (!sage) { return; }

  const int H = 8, T = 2048;
  const int bq = MetalSageAttention::nax_query_block();
  const int bk = MetalSageAttention::nax_key_block();
  const std::size_t want =
      MetalSageAttention::scratch_bytes(H, H, T, T, kD, bq, bk);
  EXPECT_TRUE(want > 0);

  SharedBuffer arena = mc->make_shared_buffer(want);
  ASSERT_TRUE(!arena.empty());
  if (arena.empty()) { return; }
  sage->set_arena(arena, SharedBuffer{});

  const std::size_t before = mc->memory_budget().allocated;
  SharedBuffer q = mc->make_shared_buffer((std::size_t)H * T * kD * 2);
  SharedBuffer k = mc->make_shared_buffer((std::size_t)H * T * kD * 2);
  ASSERT_TRUE(!q.empty() && !k.empty());
  if (k.empty()) { return; }
  const std::size_t qk = q.byte_size() + k.byte_size();

  MetalSageAttention::Operand qo{&q, 0, kD, T * kD};
  MetalSageAttention::Operand ko{&k, 0, kD, T * kD};
  sage::Config cfg;
  cfg.enabled = true;
  {
    CommandStream st = mc->make_command_stream();
    {
      ComputeEncoder e = st.begin_compute();
      EXPECT_TRUE(sage->prepare(e, qo, ko, H, H, T, T, kD, bq, bk, cfg, &err));
    }
    ASSERT_TRUE(st.commit().wait_ok(&err));
  }
  // Everything came out of the lend: what the device gained is the two
  // operands this test allocated and nothing else.
  const std::size_t after = mc->memory_budget().allocated;
  std::printf("[sage-drv] estimate %zu KB, private growth %lld B\n",
              want >> 10, (long long)after - (long long)before - (long long)qk);
  EXPECT_TRUE(after <= before + qk);
  EXPECT_TRUE(sage->resident_bytes() <= want);
}

// A lender gets its arena back, for the reason the Sol pair next door
// does: the scratch is carved windows, and a subview added to the
// residency set pins its whole parent.
TEST(metal_sage_attention, destroying_the_driver_gives_the_arena_back)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  if (!MetalSageAttention::available(mc)) { return; }
  if (!mc->residency_set_supported()) { return; }

  const int H = 8, T = 2048;
  const int bq = MetalSageAttention::nax_query_block();
  const int bk = MetalSageAttention::nax_key_block();
  const std::size_t lent =
      MetalSageAttention::scratch_bytes(H, H, T, T, kD, bq, bk);
  const std::size_t base = mc->memory_budget().allocated;
  {
    std::string err;
    std::unique_ptr<MetalSageAttention> sage =
        MetalSageAttention::load(mc, /*bf16=*/false, &err);
    ASSERT_TRUE(sage != nullptr);
    if (!sage) { return; }
    SharedBuffer arena = mc->make_shared_buffer(lent);
    SharedBuffer q = mc->make_shared_buffer((std::size_t)H * T * kD * 2);
    SharedBuffer k = mc->make_shared_buffer((std::size_t)H * T * kD * 2);
    ASSERT_TRUE(!arena.empty() && !k.empty());
    if (k.empty()) { return; }
    sage->set_arena(arena, SharedBuffer{});
    MetalSageAttention::Operand qo{&q, 0, kD, T * kD};
    MetalSageAttention::Operand ko{&k, 0, kD, T * kD};
    sage::Config cfg;
    cfg.enabled = true;
    CommandStream st = mc->make_command_stream();
    {
      ComputeEncoder e = st.begin_compute();
      EXPECT_TRUE(sage->prepare(e, qo, ko, H, H, T, T, kD, bq, bk, cfg, &err));
    }
    ASSERT_TRUE(st.commit().wait_ok(&err));
  }
  EXPECT_TRUE(mc->memory_budget().allocated < base + lent / 2);
}

// GROUPED KV, which is what FLUX.2 and Krea-2 run (both HED/KVH > 1).
//
// The int8 operands are indexed by head, and the two sides are indexed
// by DIFFERENT heads: Q by the query head, K by the query head divided
// by gqa_factor. Collapsing that -- reading K8 at the query head, as the
// first version of the kernel did -- is invisible at gqa_factor 1, which
// is exactly what H3 and Qwen-Image run and what every earlier test
// here used. So this is the case that was passing while wrong.
//
// The reference attends head h against kv head h / gqa, in double.
TEST(metal_sage_attention, grouped_kv_reads_the_kv_head)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  if (!MetalSageAttention::available(mc)) { return; }
  std::string err;
  std::unique_ptr<MetalSageAttention> sage =
      MetalSageAttention::load(mc, /*bf16=*/false, &err);
  ASSERT_TRUE(sage != nullptr);
  if (!sage) { return; }

  const int H = 8, KVH = 2, T = 512;
  const int gqa = H / KVH;
  Rig r;
  if (!r.init(mc, H, T)) { return; }
  {
    auto* p = static_cast<AttnP*>(r.pb.contents());
    p->gqa = gqa;
  }
  ComputeFunction f_i8 = r.fn(true);
  if (!f_i8.valid()) { return; }

  // Q over H heads, K/V over KVH -- and the kv heads are made DELIBERATELY
  // unlike each other, so reading the wrong one is a large error rather
  // than a subtle one.
  std::vector<float> q, k, v;
  fill_(&q, &k, &v, H, T, 0x9a5e0011u);
  k.resize((std::size_t)KVH * T * kD);
  v.resize((std::size_t)KVH * T * kD);
  for (int h = 0; h < KVH; ++h) {
    for (std::size_t i = 0; i < (std::size_t)T * kD; ++i) {
      k[(std::size_t)h * T * kD + i] += 3.0f * (float)(h + 1);
      v[(std::size_t)h * T * kD + i] *= (float)(h + 1);
    }
  }

  const std::size_t nq = (std::size_t)H * T * kD;
  const std::size_t nk = (std::size_t)KVH * T * kD;
  SharedBuffer qb = mc->make_shared_buffer(nq * 2);
  SharedBuffer kb = mc->make_shared_buffer(nk * 2);
  SharedBuffer vb = mc->make_shared_buffer(nk * 2);
  SharedBuffer ob = mc->make_shared_buffer(nq * 2);
  ASSERT_TRUE(!qb.empty() && !ob.empty());
  if (ob.empty()) { return; }
  auto* qp = static_cast<std::uint16_t*>(qb.contents());
  auto* kp = static_cast<std::uint16_t*>(kb.contents());
  auto* vp = static_cast<std::uint16_t*>(vb.contents());
  for (std::size_t i = 0; i < nq; ++i) { qp[i] = f2h_(q[i]); }
  for (std::size_t i = 0; i < nk; ++i) {
    kp[i] = f2h_(k[i]); vp[i] = f2h_(v[i]);
  }

  {
    auto* p = static_cast<AttnP*>(r.pb.contents());
    p->Qs[0] = (std::int64_t)H * T * kD;
    p->Qs[1] = (std::int64_t)T * kD; p->Qs[2] = kD;
    p->Ks[0] = (std::int64_t)KVH * T * kD;
    p->Ks[1] = (std::int64_t)T * kD; p->Ks[2] = kD;
    for (int i = 0; i < 3; ++i) { p->Vs[i] = p->Ks[i]; }
    p->Os[0] = p->Qs[0]; p->Os[1] = p->Qs[1]; p->Os[2] = kD;
  }

  MetalSageAttention::Operand qo{&qb, 0, kD, T * kD};
  MetalSageAttention::Operand ko{&kb, 0, kD, T * kD};
  sage::Config cfg;
  cfg.enabled = true;
  {
    CommandStream st = mc->make_command_stream();
    {
      ComputeEncoder e = st.begin_compute();
      ASSERT_TRUE(sage->prepare(
          e, qo, ko, H, KVH, T, T, kD,
          MetalSageAttention::nax_query_block(),
          MetalSageAttention::nax_key_block(), cfg, &err));
      e.set_function(f_i8);
      e.set_buffer(0, qb); e.set_buffer(1, kb); e.set_buffer(2, vb);
      e.set_buffer(3, ob); e.set_buffer(4, r.pb);
      sage->bind(e);
      e.dispatch({32u * (unsigned)r.fn_blocks(), 4u * (unsigned)H, 1},
                 {32, 4, 1});
    }
    ASSERT_TRUE(st.commit().wait_ok(&err));
  }

  const auto* o = static_cast<const std::uint16_t*>(ob.contents());
  double worst = 1.0;
  for (int h = 0; h < H; ++h) {
    // The reference for query head h, against KV head h / gqa. Built by
    // gathering that head's k/v into head-major position 0 so ref_ can
    // be reused unchanged.
    std::vector<float> qh((std::size_t)T * kD), kh, vh;
    std::memcpy(qh.data(), q.data() + (std::size_t)h * T * kD,
                (std::size_t)T * kD * sizeof(float));
    const std::size_t kvo = (std::size_t)(h / gqa) * T * kD;
    kh.assign(k.begin() + (long)kvo, k.begin() + (long)(kvo + (std::size_t)T * kD));
    vh.assign(v.begin() + (long)kvo, v.begin() + (long)(kvo + (std::size_t)T * kD));
    const std::vector<double> rf = ref_(qh, kh, vh, 0, T, r.scale);
    std::vector<double> got(rf.size());
    for (std::size_t i = 0; i < rf.size(); ++i) {
      got[i] = (double)h2f_(o[(std::size_t)h * T * kD + i]);
    }
    const double c = cos_(got, rf);
    worst = c < worst ? c : worst;
  }
  std::printf("[sage-drv] gqa %d: worst head cosine %.8f\n", gqa, worst);
  // A head reading its NEIGHBOUR's keys lands far below this; the kv
  // heads were separated above precisely so that it would.
  EXPECT_TRUE(worst > 0.995);
}

// THE TWO FLASH KERNELS AT QWEN-IMAGE'S SHAPE.
//
// Qwen-Image was the last DiT here still on the ALU steel attention everywhere; it
// now takes the matrix-core entry by default, as krea2, flux2 and boogu
// already did. The model-level guard for that lives in
// qwen_image_edit_dit.forward_nax_matches_alu -- and it needs a 20B
// checkpoint, so on a box without one it says nothing.
//
// This says something without one. It is the same question with the
// model taken out: at 24 heads over a joint sequence of text + image
// tokens at head_dim 128, do the two kernels compute the same attention?
// They tile differently (64x32 against 32x16) and accumulate the online
// softmax in a different order, so they are not bit-identical and are not
// meant to be. Both are scored against the SAME double-precision
// reference, which is what makes the two errors comparable rather than
// merely small.
TEST(metal_sage_attention, the_alu_and_nax_kernels_agree_at_qwen_image_shape)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  if (!mc->supports_matrix_cores()) {
    std::printf("[sage-drv] no matrix cores -- the NAX arm cannot run, "
                "SKIPPED\n");
    return;
  }
  ComputeLibrary alu = mc->load_library("attn_steel");
  ComputeLibrary nax = mc->load_library("attn_steel_nax");
  if (!alu.valid() || !nax.valid()) { return; }

  // Qwen-Image's own numbers: 24 heads, head_dim 128, and a joint
  // sequence of 96 text rows over a 16x16 image grid -- the shape its
  // own DiT A/B uses.
  const int H = 24, T = 96 + 16 * 16;
  const double scale = 1.0 / std::sqrt((double)kD);

  std::vector<float> q, k, v;
  fill_(&q, &k, &v, H, T, 0x2c0ffee7u);
  const std::size_t n = (std::size_t)H * T * kD;
  SharedBuffer qb = mc->make_shared_buffer(n * 2);
  SharedBuffer kb = mc->make_shared_buffer(n * 2);
  SharedBuffer vb = mc->make_shared_buffer(n * 2);
  SharedBuffer o_alu = mc->make_shared_buffer(n * 2);
  SharedBuffer o_nax = mc->make_shared_buffer(n * 2);
  SharedBuffer pb = mc->make_shared_buffer(sizeof(AttnP));
  ASSERT_TRUE(!qb.empty() && !o_nax.empty() && !pb.empty());
  if (pb.empty()) { return; }
  auto* qp = static_cast<std::uint16_t*>(qb.contents());
  auto* kp = static_cast<std::uint16_t*>(kb.contents());
  auto* vp = static_cast<std::uint16_t*>(vb.contents());
  for (std::size_t i = 0; i < n; ++i) {
    qp[i] = f2h_(q[i]); kp[i] = f2h_(k[i]); vp[i] = f2h_(v[i]);
  }

  // Each arm fills the params with ITS OWN tiles -- which is the whole
  // plumbing question, since every NQ/NK the kernel reads comes from
  // them. A params block filled for the other kernel's tiles is exactly
  // the bug this guards.
  auto run = [&](bool is_nax, SharedBuffer& out) {
    const int bq = is_nax ? 64 : 32;
    const int bk = is_nax ? 32 : 16;
    auto* p = static_cast<AttnP*>(pb.contents());
    p->B = 1; p->H = H; p->D = kD; p->qL = T; p->kL = T; p->gqa = 1;
    p->scale = (float)scale;
    p->NQ = (T + bq - 1) / bq; p->NK = (T + bk - 1) / bk;
    p->NQa = T / bq; p->NKa = T / bk;
    p->qL_rem = T - p->NQa * bq; p->kL_rem = T - p->NKa * bk;
    p->qL_off = 0;
    const std::int64_t hm[3] = {(std::int64_t)H * T * kD,
                                (std::int64_t)T * kD, kD};
    for (int i = 0; i < 3; ++i) {
      p->Qs[i] = hm[i]; p->Ks[i] = hm[i]; p->Vs[i] = hm[i]; p->Os[i] = hm[i];
    }
    FunctionConstants fc;
    fc.set_bool(200, (T % bq) == 0).set_bool(201, (T % bk) == 0)
        .set_bool(300, false).set_bool(301, false).set_bool(302, false)
        .set_bool(303, false).set_bool(304, false).set_bool(305, false)
        .set_bool(sage::kQkInt8Constant, false);
    ComputeFunction fn = is_nax
        ? nax.function("attn_steel_nax_h_bd128", fc)
        : alu.function("attn_steel_h_bd128", fc);
    if (!fn.valid()) { return false; }
    std::string err;
    CommandStream st = mc->make_command_stream();
    {
      ComputeEncoder e = st.begin_compute();
      e.set_function(fn);
      e.set_buffer(0, qb); e.set_buffer(1, kb); e.set_buffer(2, vb);
      e.set_buffer(3, out); e.set_buffer(4, pb);
      e.dispatch({32u * (unsigned)((T + bq - 1) / bq), 4u * (unsigned)H, 1},
                 {32, 4, 1});
    }
    return st.commit().wait_ok(&err);
  };
  // The params buffer is rewritten per arm, so the arms must not overlap.
  ASSERT_TRUE(run(false, o_alu));
  ASSERT_TRUE(run(true, o_nax));

  const auto* a = static_cast<const std::uint16_t*>(o_alu.contents());
  const auto* b = static_cast<const std::uint16_t*>(o_nax.contents());
  double c_alu = 0.0, c_nax = 0.0;
  for (int h = 0; h < H; ++h) {
    const std::vector<double> rf = ref_(q, k, v, h, T, scale);
    std::vector<double> x(rf.size()), y(rf.size());
    for (std::size_t i = 0; i < rf.size(); ++i) {
      x[i] = (double)h2f_(a[(std::size_t)h * T * kD + i]);
      y[i] = (double)h2f_(b[(std::size_t)h * T * kD + i]);
    }
    c_alu += cos_(x, rf) / H;
    c_nax += cos_(y, rf) / H;
  }
  std::printf("[sage-drv] qwen-image shape (%d heads x %d rows): ALU cos "
              "%.8f | NAX cos %.8f\n", H, T, c_alu, c_nax);
  EXPECT_TRUE(c_alu > 0.999);
  // The bar the change rests on: the new default is no further from the
  // truth than the old one.
  EXPECT_TRUE(c_nax > 0.999);
  EXPECT_TRUE(c_nax > c_alu - 1e-4);
}

// A PARTIAL LEND, which is the case private_bytes() exists for and the
// one H3's preflight actually hits: Sol takes the forward arena whenever
// it is on, so what is left for Sage is either the whole buffer or
// nothing -- and the estimate has to be right in the middle too, or the
// first geometry that half-fits reports a number nobody can trust.
//
// Lend a region big enough for some of the plan and not all of it, and
// require the DEVICE's growth to be exactly what private_bytes()
// predicted. A bound would pass an estimate that was merely the right
// order of magnitude.
TEST(metal_sage_attention, the_private_estimate_matches_a_partial_lend)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  if (!MetalSageAttention::available(mc)) { return; }
  std::string err;
  std::unique_ptr<MetalSageAttention> sage =
      MetalSageAttention::load(mc, /*bf16=*/false, &err);
  ASSERT_TRUE(sage != nullptr);
  if (!sage) { return; }

  const int H = 8, T = 2048;
  const int bq = MetalSageAttention::nax_query_block();
  const int bk = MetalSageAttention::nax_key_block();
  const std::size_t all =
      MetalSageAttention::scratch_bytes(H, H, T, T, kD, bq, bk);
  ASSERT_TRUE(all > 0);
  if (all == 0) { return; }

  // Three lends: nothing, a third, and everything. The middle one is
  // the point; the outer two pin the ends the other tests already cover
  // so a regression cannot pass by getting only those right.
  for (const int frac : {0, 3, 1}) {
    const std::size_t lend = frac == 0 ? 0 : all / (std::size_t)frac;
    const std::size_t want = MetalSageAttention::private_bytes(
        H, H, T, T, kD, bq, bk, lend, 0);
    SharedBuffer arena;
    if (lend > 0) {
      arena = mc->make_shared_buffer(lend);
      ASSERT_TRUE(!arena.empty());
      if (arena.empty()) { return; }
    }
    // Cleared first: a stale lend would keep the previous arm's windows.
    sage->set_arena(SharedBuffer{}, SharedBuffer{});
    sage->set_arena(arena, SharedBuffer{});

    SharedBuffer q = mc->make_shared_buffer((std::size_t)H * T * kD * 2);
    SharedBuffer k = mc->make_shared_buffer((std::size_t)H * T * kD * 2);
    ASSERT_TRUE(!q.empty() && !k.empty());
    if (k.empty()) { return; }
    MetalSageAttention::Operand qo{&q, 0, kD, T * kD};
    MetalSageAttention::Operand ko{&k, 0, kD, T * kD};
    sage::Config cfg;
    cfg.enabled = true;
    {
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder e = st.begin_compute();
        EXPECT_TRUE(sage->prepare(e, qo, ko, H, H, T, T, kD, bq, bk, cfg,
                                  &err));
      }
      ASSERT_TRUE(st.commit().wait_ok(&err));
    }
    // ASKED OF THE OBJECT, not of the device: currentAllocatedSize() is
    // process-wide and moves with every other Metal object, so it can
    // bound this but cannot equal it.
    const std::size_t got = sage->private_held();
    std::printf("[sage-drv] lend %6zu KB: private predicted %6zu KB, "
                "held %6zu KB%s\n", lend >> 10, want >> 10, got >> 10,
                got == want ? "" : "   MISMATCH");
    EXPECT_TRUE(got == want);
  }
}

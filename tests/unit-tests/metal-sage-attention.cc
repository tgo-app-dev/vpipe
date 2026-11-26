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
#include <cstdlib>
#include <cstring>
#include <memory>
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
// The reference attends head h against kv head h / gqa, in double. Both
// arms: the int8 QK alone, and with its P*V (sage_pv8), whose uint8 V,
// scale and mean are indexed by the KV head too -- the kv heads' V are
// scaled apart below, so V8 read at the query head is a large error.
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
  ComputeFunction f_pv;
  {
    const int bq = MetalSageAttention::nax_query_block();
    const int bk = MetalSageAttention::nax_key_block();
    FunctionConstants fc;
    fc.set_bool(200, (T % bq) == 0).set_bool(201, (T % bk) == 0)
        .set_bool(300, false).set_bool(301, false).set_bool(302, false)
        .set_bool(303, false).set_bool(304, false).set_bool(305, false)
        .set_bool(sage::kQkInt8Constant, true);
    MetalSageAttention::pv8_constants(fc, true);
    f_pv = r.lib.function("attn_steel_nax_h_bd128", fc);
  }
  ASSERT_TRUE(f_pv.valid() && sage->pv8_available());
  if (!f_pv.valid()) { return; }

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
  MetalSageAttention::Operand vo{&vb, 0, kD, T * kD};
  for (const bool pv8 : {false, true}) {
    sage::Config cfg;
    cfg.enabled = true;
    cfg.pv8 = pv8;
    {
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder e = st.begin_compute();
        ASSERT_TRUE(sage->prepare(
            e, qo, ko, H, KVH, T, T, kD,
            MetalSageAttention::nax_query_block(),
            MetalSageAttention::nax_key_block(), cfg, &err, &vo));
        ASSERT_TRUE(sage->pv8_ready() == pv8);
        e.set_function(pv8 ? f_pv : f_i8);
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
      kh.assign(k.begin() + (long)kvo,
                k.begin() + (long)(kvo + (std::size_t)T * kD));
      vh.assign(v.begin() + (long)kvo,
                v.begin() + (long)(kvo + (std::size_t)T * kD));
      const std::vector<double> rf = ref_(qh, kh, vh, 0, T, r.scale);
      std::vector<double> got(rf.size());
      for (std::size_t i = 0; i < rf.size(); ++i) {
        got[i] = (double)h2f_(o[(std::size_t)h * T * kD + i]);
      }
      const double c = cos_(got, rf);
      worst = c < worst ? c : worst;
    }
    std::printf("[sage-drv] gqa %d%s: worst head cosine %.8f\n", gqa,
                pv8 ? " + int8 PV" : "", worst);
    // A head reading its NEIGHBOUR's keys (or values) lands far below
    // this; the kv heads were separated above precisely so that it would.
    EXPECT_TRUE(worst > 0.995);
  }
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

namespace {

std::uint16_t f2bf_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (std::uint16_t)(u >> 16);
}

float bf2f_(std::uint16_t b)
{
  const std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

}  // namespace

// THE INT8 P*V THROUGH THE DRIVER (sage::Config::pv8): V's operand handed
// to prepare(), Q per token, V's mean, scale and uint8 copy made by it,
// the function built with pv8_constants(). Scored against f64 beside the
// int8 QK alone, in f16 and bf16 -- the two the families run -- and then
// from a FUSED [rows, 3*I] source against the head-major one, required
// bit-identical, which is what says V's operand is addressed as given.
TEST(metal_sage_attention, the_driver_runs_the_int8_pv)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  if (!MetalSageAttention::available(mc)) { return; }
  const int bq = MetalSageAttention::nax_query_block();
  const int bk = MetalSageAttention::nax_key_block();

  for (const bool bf16 : {false, true}) {
    std::string err;
    std::unique_ptr<MetalSageAttention> sage =
        MetalSageAttention::load(mc, bf16, &err);
    ASSERT_TRUE(sage != nullptr);
    if (!sage) { return; }
    auto enc = [&](float x) { return bf16 ? f2bf_(x) : f2h_(x); };
    auto dec = [&](std::uint16_t x) { return bf16 ? bf2f_(x) : h2f_(x); };
    for (const int T : {1024, 1000}) {
      const int H = 4, I = H * kD;
      Rig r;
      if (!r.init(mc, H, T)) { return; }
      auto build = [&](bool pv8) {
        FunctionConstants fc;
        fc.set_bool(200, (T % bq) == 0).set_bool(201, (T % bk) == 0)
            .set_bool(300, false).set_bool(301, false).set_bool(302, false)
            .set_bool(303, false).set_bool(304, false).set_bool(305, false)
            .set_bool(sage::kQkInt8Constant, true);
        MetalSageAttention::pv8_constants(fc, pv8);
        return r.lib.function(bf16 ? "attn_steel_nax_h_bd128_bf16"
                                   : "attn_steel_nax_h_bd128", fc);
      };
      ComputeFunction f_qk = build(false), f_pv = build(true);
      ASSERT_TRUE(f_qk.valid() && f_pv.valid());
      if (!f_pv.valid()) { return; }

      std::vector<float> q, k, v;
      fill_(&q, &k, &v, H, T, 0x0b8f0001u + (unsigned)T);
      const std::size_t n = (std::size_t)H * T * kD;
      SharedBuffer qb = mc->make_shared_buffer(n * 2);
      SharedBuffer kb = mc->make_shared_buffer(n * 2);
      SharedBuffer vb = mc->make_shared_buffer(n * 2);
      SharedBuffer fused = mc->make_shared_buffer((std::size_t)T * 3 * I * 2);
      SharedBuffer o_qk = mc->make_shared_buffer(n * 2);
      SharedBuffer o_hm = mc->make_shared_buffer(n * 2);
      SharedBuffer o_fu = mc->make_shared_buffer(n * 2);
      ASSERT_TRUE(!fused.empty() && !o_fu.empty());
      if (o_fu.empty()) { return; }
      auto* qp = static_cast<std::uint16_t*>(qb.contents());
      auto* kp = static_cast<std::uint16_t*>(kb.contents());
      auto* vp = static_cast<std::uint16_t*>(vb.contents());
      auto* fu = static_cast<std::uint16_t*>(fused.contents());
      for (int h = 0; h < H; ++h) {
        for (int t = 0; t < T; ++t) {
          for (int c = 0; c < kD; ++c) {
            const std::size_t i =
                ((std::size_t)h * T + (std::size_t)t) * kD + (std::size_t)c;
            qp[i] = enc(q[i]); kp[i] = enc(k[i]); vp[i] = enc(v[i]);
            const std::size_t f = (std::size_t)t * 3 * I +
                                  (std::size_t)h * kD + (std::size_t)c;
            fu[f] = qp[i];
            fu[f + (std::size_t)I] = kp[i];
            fu[f + (std::size_t)2 * I] = vp[i];
          }
        }
      }
      sage::Config cfg;
      cfg.enabled = true;
      auto run = [&](bool fused_src, bool pv8, SharedBuffer& out) {
        MetalSageAttention::Operand qo, ko, vo;
        if (fused_src) {
          qo = {&fused, 0, 3 * I, kD};
          ko = {&fused, (std::size_t)I, 3 * I, kD};
          vo = {&fused, (std::size_t)2 * I, 3 * I, kD};
          r.set_strides((std::int64_t)T * 3 * I, kD, 3 * I);
        } else {
          qo = {&qb, 0, kD, T * kD};
          ko = {&kb, 0, kD, T * kD};
          vo = {&vb, 0, kD, T * kD};
          r.set_strides((std::int64_t)H * T * kD, (std::int64_t)T * kD, kD);
        }
        cfg.pv8 = pv8;
        CommandStream st = mc->make_command_stream();
        {
          ComputeEncoder e = st.begin_compute();
          if (!sage->prepare(e, qo, ko, H, H, T, T, kD, bq, bk, cfg, &err,
                             &vo)) {
            return false;
          }
          if (sage->pv8_ready() != pv8) { return false; }
          e.set_function(pv8 ? f_pv : f_qk);
          if (fused_src) {
            e.set_buffer(0, fused, 0);
            e.set_buffer(1, fused, (std::size_t)I * 2);
            e.set_buffer(2, fused, (std::size_t)2 * I * 2);
          } else {
            e.set_buffer(0, qb); e.set_buffer(1, kb); e.set_buffer(2, vb);
          }
          e.set_buffer(3, out);
          e.set_buffer(4, r.pb);
          sage->bind(e);
          e.dispatch({32u * (unsigned)r.fn_blocks(), 4u * (unsigned)H, 1},
                     {32, 4, 1});
        }
        return st.commit().wait_ok(&err);
      };
      ASSERT_TRUE(run(true, true, o_fu));
      ASSERT_TRUE(run(false, true, o_hm));
      ASSERT_TRUE(run(false, false, o_qk));

      const auto* a = static_cast<const std::uint16_t*>(o_qk.contents());
      const auto* b = static_cast<const std::uint16_t*>(o_hm.contents());
      double c_qk = 0.0, c_pv = 0.0;
      for (int h = 0; h < H; ++h) {
        const std::vector<double> rf = ref_(q, k, v, h, T, r.scale);
        std::vector<double> x(rf.size()), y(rf.size());
        for (std::size_t i = 0; i < rf.size(); ++i) {
          x[i] = (double)dec(a[(std::size_t)h * T * kD + i]);
          y[i] = (double)dec(b[(std::size_t)h * T * kD + i]);
        }
        c_qk += cos_(x, rf) / H;
        c_pv += cos_(y, rf) / H;
      }
      std::size_t bad = 0;
      const auto* c = static_cast<const std::uint16_t*>(o_fu.contents());
      for (std::size_t i = 0; i < n; ++i) { bad += (b[i] != c[i]) ? 1 : 0; }
      std::printf("[sage-drv] %s T=%4d: int8 QK cos %.6f | + int8 PV cos "
                  "%.6f | fused vs head-major: %zu of %zu differ\n",
                  bf16 ? "bf16" : "f16 ", T, c_qk, c_pv, bad, n);
      EXPECT_TRUE(c_pv > 0.995);
      EXPECT_TRUE(c_pv > c_qk - 0.002);
      EXPECT_TRUE(bad == 0);
    }
  }
}

// A DISPATCH FROM A QUERY ROW ON (bind(e, q_row0)), as a family running
// the int8 P*V on some rows only does it: the first `nb` query blocks on
// one function, the rest on another, Q and O bound from row nb * BQ and
// the params counting blocks from there. Each part must be BIT-IDENTICAL
// to that function run over every row -- a row offset that missed the
// int8 Q, its per-token scale or the ragged last block would still give
// a plausible answer, of other rows. Fused and head-major sources both.
TEST(metal_sage_attention, a_dispatch_from_a_query_row_is_the_same_rows)
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
  const int bq = MetalSageAttention::nax_query_block();
  const int bk = MetalSageAttention::nax_key_block();
  const int H = 4, T = 1000, I = H * kD, nb = 3;
  const int nq = (T + bq - 1) / bq;
  Rig r;
  if (!r.init(mc, H, T)) { return; }
  auto build = [&](bool pv8) {
    FunctionConstants fc;
    fc.set_bool(200, (T % bq) == 0).set_bool(201, (T % bk) == 0)
        .set_bool(300, false).set_bool(301, false).set_bool(302, false)
        .set_bool(303, false).set_bool(304, false).set_bool(305, false)
        .set_bool(sage::kQkInt8Constant, true);
    MetalSageAttention::pv8_constants(fc, pv8);
    fc.set_bool(310, true);   // Q per token, as a pv8 prologue makes it
    return r.lib.function("attn_steel_nax_h_bd128", fc);
  };
  ComputeFunction f_qk = build(false), f_pv = build(true);
  ASSERT_TRUE(f_qk.valid() && f_pv.valid());
  if (!f_pv.valid()) { return; }

  std::vector<float> q, k, v;
  fill_(&q, &k, &v, H, T, 0x0b8f0b0du);
  const std::size_t n = (std::size_t)H * T * kD;
  SharedBuffer qb = mc->make_shared_buffer(n * 2);
  SharedBuffer kb = mc->make_shared_buffer(n * 2);
  SharedBuffer vb = mc->make_shared_buffer(n * 2);
  SharedBuffer fused = mc->make_shared_buffer((std::size_t)T * 3 * I * 2);
  SharedBuffer o_qk = mc->make_shared_buffer(n * 2);
  SharedBuffer o_pv = mc->make_shared_buffer(n * 2);
  SharedBuffer o_sp = mc->make_shared_buffer(n * 2);
  SharedBuffer pband = mc->make_shared_buffer(sizeof(AttnP));
  ASSERT_TRUE(!fused.empty() && !o_sp.empty() && !pband.empty());
  if (pband.empty()) { return; }
  auto* qp = static_cast<std::uint16_t*>(qb.contents());
  auto* kp = static_cast<std::uint16_t*>(kb.contents());
  auto* vp = static_cast<std::uint16_t*>(vb.contents());
  auto* fu = static_cast<std::uint16_t*>(fused.contents());
  for (int h = 0; h < H; ++h) {
    for (int t = 0; t < T; ++t) {
      for (int c = 0; c < kD; ++c) {
        const std::size_t i =
            ((std::size_t)h * T + (std::size_t)t) * kD + (std::size_t)c;
        qp[i] = f2h_(q[i]); kp[i] = f2h_(k[i]); vp[i] = f2h_(v[i]);
        const std::size_t f = (std::size_t)t * 3 * I +
                              (std::size_t)h * kD + (std::size_t)c;
        fu[f] = qp[i];
        fu[f + (std::size_t)I] = kp[i];
        fu[f + (std::size_t)2 * I] = vp[i];
      }
    }
  }
  sage::Config cfg;
  cfg.enabled = true;
  cfg.pv8 = true;
  for (const bool fused_src : {false, true}) {
    MetalSageAttention::Operand qo, ko, vo;
    std::size_t qoff = 0, koff = 0, voff = 0, qrow = kD;
    const SharedBuffer* src_q = &qb;
    const SharedBuffer* src_k = &kb;
    const SharedBuffer* src_v = &vb;
    if (fused_src) {
      qo = {&fused, 0, 3 * I, kD};
      ko = {&fused, (std::size_t)I, 3 * I, kD};
      vo = {&fused, (std::size_t)2 * I, 3 * I, kD};
      r.set_strides((std::int64_t)T * 3 * I, kD, 3 * I);
      src_q = src_k = src_v = &fused;
      koff = (std::size_t)I * 2;
      voff = (std::size_t)2 * I * 2;
      qrow = (std::size_t)3 * I;
    } else {
      qo = {&qb, 0, kD, T * kD};
      ko = {&kb, 0, kD, T * kD};
      vo = {&vb, 0, kD, T * kD};
      r.set_strides((std::int64_t)H * T * kD, (std::int64_t)T * kD, kD);
    }
    std::memcpy(pband.contents(), r.pb.contents(), sizeof(AttnP));
    static_cast<AttnP*>(pband.contents())->NQa -= nb;
    const std::size_t r0 = (std::size_t)nb * bq;
    CommandStream st = mc->make_command_stream();
    {
      ComputeEncoder e = st.begin_compute();
      ASSERT_TRUE(sage->prepare(e, qo, ko, H, H, T, T, kD, bq, bk, cfg,
                                &err, &vo));
      auto whole = [&](ComputeFunction& f, SharedBuffer& out) {
        e.set_function(f);
        e.set_buffer(0, *src_q, qoff); e.set_buffer(1, *src_k, koff);
        e.set_buffer(2, *src_v, voff);
        e.set_buffer(3, out); e.set_buffer(4, r.pb);
        sage->bind(e);
        e.dispatch({32u * (unsigned)nq, 4u * (unsigned)H, 1}, {32, 4, 1});
      };
      whole(f_qk, o_qk);
      whole(f_pv, o_pv);
      // The split: blocks [0, nb) on the int8 QK, the rest on the P*V.
      e.set_function(f_qk);
      e.set_buffer(0, *src_q, qoff); e.set_buffer(1, *src_k, koff);
      e.set_buffer(2, *src_v, voff);
      e.set_buffer(3, o_sp); e.set_buffer(4, r.pb);
      sage->bind(e);
      e.dispatch({32u * (unsigned)nb, 4u * (unsigned)H, 1}, {32, 4, 1});
      e.set_function(f_pv);
      e.set_buffer(0, *src_q, qoff + r0 * qrow * 2);
      e.set_buffer(3, o_sp, r0 * kD * 2);
      e.set_buffer(4, pband);
      sage->bind(e, (int)r0);
      e.dispatch({32u * (unsigned)(nq - nb), 4u * (unsigned)H, 1},
                 {32, 4, 1});
    }
    ASSERT_TRUE(st.commit().wait_ok(&err));
    const auto* a = static_cast<const std::uint16_t*>(o_qk.contents());
    const auto* b = static_cast<const std::uint16_t*>(o_pv.contents());
    const auto* c = static_cast<const std::uint16_t*>(o_sp.contents());
    std::size_t bad_head = 0, bad_tail = 0, pv_differs = 0;
    for (int h = 0; h < H; ++h) {
      for (int t = 0; t < T; ++t) {
        for (int ch = 0; ch < kD; ++ch) {
          const std::size_t i =
              ((std::size_t)h * T + (std::size_t)t) * kD + (std::size_t)ch;
          if ((std::size_t)t < r0) {
            bad_head += c[i] != a[i] ? 1 : 0;
          } else {
            bad_tail += c[i] != b[i] ? 1 : 0;
          }
          pv_differs += a[i] != b[i] ? 1 : 0;
        }
      }
    }
    std::printf("[sage-drv] %s rows split at %zu: prefix %zu differ, rest "
                "%zu differ (the two functions differ at %zu of %zu)\n",
                fused_src ? "fused     " : "head-major", r0, bad_head,
                bad_tail, pv_differs, n);
    EXPECT_TRUE(bad_head == 0 && bad_tail == 0);
    EXPECT_TRUE(pv_differs > n / 2);
  }
}

// ONE SCRATCH FOR BOTH ARMS. MiniMax-H3 runs its early steps on the int8
// QK alone and the rest with the int8 P*V, and the scratch used to be
// rebuilt at every toggle -- twice a clip, a private free and allocation
// of ~3 bytes a q/k/v element where nothing is lent. A scratch with the
// P*V's buffers now serves a call without them: no rebuild (its size
// does not move), and the int8 QK's output BIT-IDENTICAL to a driver
// that never had them -- its per-block Q scales written into a buffer
// sized per token. reserve_pv8() builds them from the first call. And
// bind(e, q_row0 > 0) refuses after a call that wrote per-block scales,
// pv8-sized scratch or not: no row offset is right for those.
TEST(metal_sage_attention, a_pv8_scratch_serves_the_int8_qk_alone)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  if (!MetalSageAttention::available(mc)) { return; }
  const int bq = MetalSageAttention::nax_query_block();
  const int bk = MetalSageAttention::nax_key_block();
  const int H = 4, T = 1000;
  Rig r;
  if (!r.init(mc, H, T)) { return; }
  r.set_strides((std::int64_t)H * T * kD, (std::int64_t)T * kD, kD);
  ComputeFunction f_qk = r.fn(true);
  ASSERT_TRUE(f_qk.valid());
  if (!f_qk.valid()) { return; }

  std::vector<float> q, k, v;
  fill_(&q, &k, &v, H, T, 0x5ca7c001u);
  const std::size_t n = (std::size_t)H * T * kD;
  SharedBuffer qb = mc->make_shared_buffer(n * 2);
  SharedBuffer kb = mc->make_shared_buffer(n * 2);
  SharedBuffer vb = mc->make_shared_buffer(n * 2);
  ASSERT_TRUE(!qb.empty() && !kb.empty() && !vb.empty());
  if (vb.empty()) { return; }
  auto* qp = static_cast<std::uint16_t*>(qb.contents());
  auto* kp = static_cast<std::uint16_t*>(kb.contents());
  auto* vp = static_cast<std::uint16_t*>(vb.contents());
  for (std::size_t i = 0; i < n; ++i) {
    qp[i] = f2h_(q[i]); kp[i] = f2h_(k[i]); vp[i] = f2h_(v[i]);
  }
  const MetalSageAttention::Operand qo{&qb, 0, kD, T * kD};
  const MetalSageAttention::Operand ko{&kb, 0, kD, T * kD};
  const MetalSageAttention::Operand vo{&vb, 0, kD, T * kD};
  const std::size_t plain =
      MetalSageAttention::scratch_bytes(H, H, T, T, kD, bq, bk, false);
  const std::size_t full =
      MetalSageAttention::scratch_bytes(H, H, T, T, kD, bq, bk, true);
  ASSERT_TRUE(full > plain);

  // A prologue alone (pv8 asked or not) and, for the int8 QK arm, its
  // attention into `out`; whether bind(e, 64) bound.
  std::string err;
  auto call = [&](MetalSageAttention& s, bool pv8, SharedBuffer* out,
                  bool* row_bind) {
    sage::Config cfg;
    cfg.enabled = true;
    cfg.pv8 = pv8;
    CommandStream st = mc->make_command_stream();
    {
      ComputeEncoder e = st.begin_compute();
      if (!s.prepare(e, qo, ko, H, H, T, T, kD, bq, bk, cfg, &err, &vo)) {
        return false;
      }
      if (row_bind != nullptr) { *row_bind = s.bind(e, 64); }
      if (out != nullptr) {
        e.set_function(f_qk);
        e.set_buffer(0, qb); e.set_buffer(1, kb); e.set_buffer(2, vb);
        e.set_buffer(3, *out);
        e.set_buffer(4, r.pb);
        s.bind(e);
        e.dispatch({32u * (unsigned)r.fn_blocks(), 4u * (unsigned)H, 1},
                   {32, 4, 1});
      }
    }
    return st.commit().wait_ok(&err);
  };

  std::string lerr;
  std::unique_ptr<MetalSageAttention> a =
      MetalSageAttention::load(mc, /*bf16=*/false, &lerr);
  std::unique_ptr<MetalSageAttention> b =
      MetalSageAttention::load(mc, /*bf16=*/false, &lerr);
  std::unique_ptr<MetalSageAttention> c =
      MetalSageAttention::load(mc, /*bf16=*/false, &lerr);
  ASSERT_TRUE(a && b && c && a->pv8_available());
  if (!a || !b || !c) { return; }
  SharedBuffer o_fresh = mc->make_shared_buffer(n * 2);
  SharedBuffer o_served = mc->make_shared_buffer(n * 2);
  ASSERT_TRUE(!o_fresh.empty() && !o_served.empty());
  if (o_served.empty()) { return; }

  // Sizes by private_held(): nothing is lent, so it is the whole carve,
  // padded as scratch_bytes() pads it. `a` never sees pv8: the int8 QK
  // on a scratch of its own size.
  ASSERT_TRUE(call(*a, false, &o_fresh, nullptr));
  const std::size_t a_held = a->private_held();
  // `b` is built with it, then asked without: kept, not rebuilt.
  bool rb_pv = false, rb_qk = true;
  ASSERT_TRUE(call(*b, true, nullptr, &rb_pv));
  const std::size_t b_pv = b->private_held();
  ASSERT_TRUE(call(*b, false, &o_served, &rb_qk));
  const std::size_t b_qk = b->private_held();
  const bool b_ready = b->pv8_ready();
  // `c` reserves them, and its FIRST call has none.
  c->reserve_pv8(true);
  ASSERT_TRUE(call(*c, false, nullptr, nullptr));
  const std::size_t c_held = c->private_held();

  const bool same = std::memcmp(o_fresh.contents(), o_served.contents(),
                                n * 2) == 0;
  std::printf("[sage-drv] scratch %zu / %zu KB (int8 QK / + P*V): fresh "
              "%zu, pv8 then not %zu -> %zu, reserved %zu; int8 QK on the "
              "pv8 scratch: %s; bind at row 64: %d after pv8, %d after "
              "not\n", plain >> 10, full >> 10, a_held >> 10, b_pv >> 10,
              b_qk >> 10, c_held >> 10, same ? "bit-identical" : "DIFFERS",
              (int)rb_pv, (int)rb_qk);
  EXPECT_TRUE(a_held == plain);
  EXPECT_TRUE(b_pv == full && b_qk == full);
  EXPECT_TRUE(c_held == full);
  EXPECT_TRUE(!b_ready);
  EXPECT_TRUE(same);
  EXPECT_TRUE(rb_pv && !rb_qk);
}

// A RESERVATION COVERS THE SMALLER CALLS. A family whose attention runs
// at several row counts in one forward (FLUX.3: each mode's stack over
// its own rows, then the joint stack) rebuilt the scratch at every
// change -- a private free and allocation, two or three a forward. With
// reserve() at the largest, a prepare() it covers reuses it, and the
// answer is the SAME BITS as a scratch built for that call alone (each
// call is written and read at its own geometry), int8 QK alone and with
// the P*V. A caller that reserves nothing keeps the old behaviour: its
// scratch follows its last call, down as well as up. And reserve() is
// exact: a smaller reservation shrinks it.
TEST(metal_sage_attention, a_reservation_covers_smaller_calls)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  if (!MetalSageAttention::available(mc)) { return; }
  const int bq = MetalSageAttention::nax_query_block();
  const int bk = MetalSageAttention::nax_key_block();
  const int H = 4, big = 1000, small = 600;
  // Two rigs: each call's params and functions are its own geometry's.
  Rig rs;
  if (!rs.init(mc, H, small)) { return; }
  rs.set_strides((std::int64_t)H * small * kD, (std::int64_t)small * kD,
                 kD);
  ComputeFunction f_qk = rs.fn(true), f_pv;
  {
    FunctionConstants fc;
    fc.set_bool(200, (small % bq) == 0).set_bool(201, (small % bk) == 0)
        .set_bool(300, false).set_bool(301, false).set_bool(302, false)
        .set_bool(303, false).set_bool(304, false).set_bool(305, false)
        .set_bool(sage::kQkInt8Constant, true);
    MetalSageAttention::pv8_constants(fc, true);
    f_pv = rs.lib.function("attn_steel_nax_h_bd128", fc);
  }
  ASSERT_TRUE(f_qk.valid() && f_pv.valid());
  if (!f_pv.valid()) { return; }

  std::vector<float> q, k, v;
  fill_(&q, &k, &v, H, small, 0x7e5e7001u);
  const std::size_t n = (std::size_t)H * small * kD;
  SharedBuffer qb = mc->make_shared_buffer(n * 2);
  SharedBuffer kb = mc->make_shared_buffer(n * 2);
  SharedBuffer vb = mc->make_shared_buffer(n * 2);
  ASSERT_TRUE(!qb.empty() && !kb.empty() && !vb.empty());
  if (vb.empty()) { return; }
  auto* qp = static_cast<std::uint16_t*>(qb.contents());
  auto* kp = static_cast<std::uint16_t*>(kb.contents());
  auto* vp = static_cast<std::uint16_t*>(vb.contents());
  for (std::size_t i = 0; i < n; ++i) {
    qp[i] = f2h_(q[i]); kp[i] = f2h_(k[i]); vp[i] = f2h_(v[i]);
  }
  const MetalSageAttention::Operand qo{&qb, 0, kD, small * kD};
  const MetalSageAttention::Operand ko{&kb, 0, kD, small * kD};
  const MetalSageAttention::Operand vo{&vb, 0, kD, small * kD};

  // The `small` attention through `s` into `out`, int8 QK or with P*V.
  std::string err;
  auto run = [&](MetalSageAttention& s, bool pv8, SharedBuffer& out) {
    sage::Config cfg;
    cfg.enabled = true;
    cfg.pv8 = pv8;
    CommandStream st = mc->make_command_stream();
    {
      ComputeEncoder e = st.begin_compute();
      if (!s.prepare(e, qo, ko, H, H, small, small, kD, bq, bk, cfg, &err,
                     &vo) ||
          s.pv8_ready() != pv8) {
        return false;
      }
      e.set_function(pv8 ? f_pv : f_qk);
      e.set_buffer(0, qb); e.set_buffer(1, kb); e.set_buffer(2, vb);
      e.set_buffer(3, out);
      e.set_buffer(4, rs.pb);
      s.bind(e);
      e.dispatch({32u * (unsigned)rs.fn_blocks(), 4u * (unsigned)H, 1},
                 {32, 4, 1});
    }
    return st.commit().wait_ok(&err);
  };
  auto size_of = [&](int t, bool pv8) {
    return MetalSageAttention::scratch_bytes(H, H, t, t, kD, bq, bk, pv8);
  };

  std::string lerr;
  std::unique_ptr<MetalSageAttention> fresh =
      MetalSageAttention::load(mc, /*bf16=*/false, &lerr);
  std::unique_ptr<MetalSageAttention> held =
      MetalSageAttention::load(mc, /*bf16=*/false, &lerr);
  std::unique_ptr<MetalSageAttention> plain =
      MetalSageAttention::load(mc, /*bf16=*/false, &lerr);
  ASSERT_TRUE(fresh && held && plain && held->pv8_available());
  if (!fresh || !held || !plain) { return; }

  bool same[2] = {false, false};
  int builds_after[2] = {0, 0};
  for (const bool pv8 : {false, true}) {
    SharedBuffer o_fresh = mc->make_shared_buffer(n * 2);
    SharedBuffer o_held = mc->make_shared_buffer(n * 2);
    ASSERT_TRUE(!o_fresh.empty() && !o_held.empty());
    if (o_held.empty()) { return; }
    ASSERT_TRUE(run(*fresh, pv8, o_fresh));
    // The capacity first, then the call it covers -- twice.
    ASSERT_TRUE(held->reserve(H, H, big, big, kD, bq, bk, true, &err));
    ASSERT_TRUE(run(*held, pv8, o_held));
    ASSERT_TRUE(held->reserve(H, H, big, big, kD, bq, bk, true, &err));
    ASSERT_TRUE(run(*held, pv8, o_held));
    same[pv8 ? 1 : 0] =
        std::memcmp(o_fresh.contents(), o_held.contents(), n * 2) == 0;
    builds_after[pv8 ? 1 : 0] = held->scratch_builds();
  }
  const std::size_t held_big = held->private_held();
  // ...and an exact smaller reservation shrinks it.
  ASSERT_TRUE(held->reserve(H, H, small, small, kD, bq, bk, true, &err));
  const std::size_t held_small = held->private_held();

  // No reservation: the scratch follows the last call, down as well --
  // a prologue at `big` (zeros will do), then the `small` attention.
  std::size_t plain_big = 0, plain_small = 0;
  {
    const std::size_t nb = (std::size_t)H * big * kD * 2;
    SharedBuffer qB = mc->make_shared_buffer(nb);
    SharedBuffer kB = mc->make_shared_buffer(nb);
    SharedBuffer vB = mc->make_shared_buffer(nb);
    ASSERT_TRUE(!qB.empty() && !kB.empty() && !vB.empty());
    if (vB.empty()) { return; }
    for (SharedBuffer* b : {&qB, &kB, &vB}) {
      std::memset(b->contents(), 0, nb);
    }
    const MetalSageAttention::Operand qoB{&qB, 0, kD, big * kD};
    const MetalSageAttention::Operand koB{&kB, 0, kD, big * kD};
    const MetalSageAttention::Operand voB{&vB, 0, kD, big * kD};
    sage::Config cfg;
    cfg.enabled = true;
    CommandStream st = mc->make_command_stream();
    {
      ComputeEncoder e = st.begin_compute();
      ASSERT_TRUE(plain->prepare(e, qoB, koB, H, H, big, big, kD, bq, bk,
                                 cfg, &err, &voB));
    }
    ASSERT_TRUE(st.commit().wait_ok(&err));
    plain_big = plain->private_held();
    SharedBuffer o = mc->make_shared_buffer(n * 2);
    ASSERT_TRUE(run(*plain, false, o));
    plain_small = plain->private_held();
  }

  std::printf("[sage-drv] reserved %d rows, called at %d: int8 QK %s, + "
              "P*V %s against a scratch of the call's own; builds %d then "
              "%d; held %zu KB (reserved), %zu KB after a %d-row "
              "reservation; unreserved %zu -> %zu KB, %d builds\n", big,
              small, same[0] ? "bit-identical" : "DIFFERS",
              same[1] ? "bit-identical" : "DIFFERS", builds_after[0],
              builds_after[1], held_big >> 10, held_small >> 10, small,
              plain_big >> 10, plain_small >> 10, plain->scratch_builds());
  EXPECT_TRUE(same[0] && same[1]);
  EXPECT_TRUE(builds_after[0] == 1 && builds_after[1] == 1);
  EXPECT_TRUE(held_big == size_of(big, true));
  EXPECT_TRUE(held_small == size_of(small, true));
  EXPECT_TRUE(held->scratch_builds() == 2);
  EXPECT_TRUE(plain_big == size_of(big, false));
  EXPECT_TRUE(plain_small == size_of(small, false));
  EXPECT_TRUE(plain->scratch_builds() == 2);
}

namespace {

// Attention in double with a SINK: a logit of its own in the denominator,
// holding no V (MAGI-2's preview DiT, constant 302).
std::vector<double> ref_sink_(const std::vector<float>& q,
                              const std::vector<float>& k,
                              const std::vector<float>& v, int h, int tokens,
                              double scale, double sink)
{
  std::vector<double> out((std::size_t)tokens * kD, 0.0);
  std::vector<double> s((std::size_t)tokens);
  const std::size_t base = (std::size_t)h * tokens * kD;
  for (int i = 0; i < tokens; ++i) {
    double m = sink;
    for (int j = 0; j < tokens; ++j) {
      double d = 0.0;
      for (int c = 0; c < kD; ++c) {
        d += (double)q[base + (std::size_t)i * kD + c] *
             (double)k[base + (std::size_t)j * kD + c];
      }
      s[(std::size_t)j] = d * scale;
      m = m > s[(std::size_t)j] ? m : s[(std::size_t)j];
    }
    double z = std::exp(sink - m);
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

}  // namespace

// AN ATTENTION WITH SINKS (302) through the driver, and the gate in front
// of it. sinks_available() is this host's answer (the contract's optional
// sage_sinks_marker); under VPIPE_SAGE_NO_SINKS=1 it reads as an older
// host's, where load_for_model(..., with_sinks) declines -- null, not
// fatal -- and without with_sinks loads as ever. Then the attention:
// sink_constants() sets 311 EXACTLY when the keys were smoothed, scored
// against f64 for smoothing on and off, the int8 QK alone and with its
// P*V. The smoothing-on arms run first, so K's mean is left in the
// driver's scratch: a function built with 311 regardless (the mutation
// arm) then shifts the sinks by a mean the unsmoothed keys never lost,
// and must read worse than the arm built as sink_constants() says.
TEST(metal_sage_attention, an_attention_with_sinks_passes_the_gate)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  if (!MetalSageAttention::available(mc)) { return; }
  const int bq = MetalSageAttention::nax_query_block();
  const int bk = MetalSageAttention::nax_key_block();
  std::string err;

  // ---- the gate
  sage::Config cfg;
  cfg.enabled = true;
  {
    std::unique_ptr<MetalSageAttention> s =
        MetalSageAttention::load(mc, /*bf16=*/false, &err);
    ASSERT_TRUE(s != nullptr && s->sinks_available());
    bool fatal = true;
    EXPECT_TRUE(MetalSageAttention::load_for_model(mc, false, cfg, "test",
                                                   &fatal, true) != nullptr);
    EXPECT_TRUE(!fatal);
    ::setenv("VPIPE_SAGE_NO_SINKS", "1", 1);
    std::unique_ptr<MetalSageAttention> old =
        MetalSageAttention::load(mc, /*bf16=*/false, &err);
    const bool old_sinks = old != nullptr && old->sinks_available();
    const bool old_pv8 = old != nullptr && old->pv8_available();
    fatal = true;
    const bool declined = MetalSageAttention::load_for_model(
                              mc, false, cfg, "test", &fatal, true) ==
                          nullptr;
    const bool declined_fatal = fatal;
    const bool plain = MetalSageAttention::load_for_model(
                           mc, false, cfg, "test", &fatal, false) != nullptr;
    ::unsetenv("VPIPE_SAGE_NO_SINKS");
    std::printf("[sage-sinks] gate: here %d, an older host's %d (its P*V "
                "%d); with sinks there declined %d (fatal %d), without "
                "loaded %d\n", (int)s->sinks_available(), (int)old_sinks,
                (int)old_pv8, (int)declined, (int)declined_fatal,
                (int)plain);
    EXPECT_TRUE(old != nullptr && !old_sinks && old_pv8);
    EXPECT_TRUE(declined && !declined_fatal && plain);
    for (const bool smooth : {true, false}) {
      sage::Config c2 = cfg;
      c2.smooth_k = smooth;
      FunctionConstants a, b;
      MetalSageAttention::sink_constants(a, c2);
      b.set_bool(sage::kSinkShiftConstant, smooth);
      EXPECT_TRUE(a.signature() == b.signature());
    }
  }

  // ---- the attention
  std::unique_ptr<MetalSageAttention> sage =
      MetalSageAttention::load(mc, /*bf16=*/false, &err);
  ASSERT_TRUE(sage != nullptr && sage->pv8_available());
  if (!sage || !sage->pv8_available()) { return; }
  const int H = 2, T = 1000;
  const float sink[2] = {7.0f, 8.0f};
  Rig r;
  if (!r.init(mc, H, T)) { return; }
  r.set_strides((std::int64_t)H * T * kD, (std::int64_t)T * kD, kD);
  // force: -1 as sink_constants says, 0 / 1 that value regardless.
  auto build = [&](const sage::Config& c, bool pv8, int force) {
    FunctionConstants fc;
    fc.set_bool(200, (T % bq) == 0).set_bool(201, (T % bk) == 0)
        .set_bool(300, false).set_bool(301, false).set_bool(302, true)
        .set_bool(303, false).set_bool(304, false).set_bool(305, false)
        .set_bool(sage::kQkInt8Constant, true);
    MetalSageAttention::sink_constants(fc, c);
    if (force >= 0) { fc.set_bool(sage::kSinkShiftConstant, force == 1); }
    MetalSageAttention::pv8_constants(fc, pv8);
    return r.lib.function("attn_steel_nax_h_bd128", fc);
  };
  std::vector<float> q, k, v;
  fill_(&q, &k, &v, H, T, 0x5157a001u);
  const std::size_t n = (std::size_t)H * T * kD;
  SharedBuffer qb = mc->make_shared_buffer(n * 2);
  SharedBuffer kb = mc->make_shared_buffer(n * 2);
  SharedBuffer vb = mc->make_shared_buffer(n * 2);
  SharedBuffer ob = mc->make_shared_buffer(n * 2);
  SharedBuffer sb = mc->make_shared_buffer((std::size_t)H * 2);
  ASSERT_TRUE(!ob.empty() && !sb.empty());
  if (ob.empty() || sb.empty()) { return; }
  auto* qp = static_cast<std::uint16_t*>(qb.contents());
  auto* kp = static_cast<std::uint16_t*>(kb.contents());
  auto* vp = static_cast<std::uint16_t*>(vb.contents());
  for (std::size_t i = 0; i < n; ++i) {
    qp[i] = f2h_(q[i]); kp[i] = f2h_(k[i]); vp[i] = f2h_(v[i]);
  }
  auto* sp = static_cast<std::uint16_t*>(sb.contents());
  for (int h = 0; h < H; ++h) { sp[h] = f2h_(sink[h]); }
  std::vector<std::vector<double>> refs;
  for (int h = 0; h < H; ++h) {
    refs.push_back(ref_sink_(q, k, v, h, T, r.scale, (double)sink[h]));
  }
  // One arm: cosine against f64 over every head.
  auto arm = [&](const sage::Config& c, bool pv8, int force) {
    ComputeFunction f = build(c, pv8, force);
    if (!f.valid()) { return -1.0; }
    sage::Config sc = c;
    sc.pv8 = pv8;
    const MetalSageAttention::Operand qo{&qb, 0, kD, T * kD};
    const MetalSageAttention::Operand ko{&kb, 0, kD, T * kD};
    const MetalSageAttention::Operand vo{&vb, 0, kD, T * kD};
    CommandStream st = mc->make_command_stream();
    {
      ComputeEncoder e = st.begin_compute();
      if (!sage->prepare(e, qo, ko, H, H, T, T, kD, bq, bk, sc, &err,
                         pv8 ? &vo : nullptr)) {
        return -1.0;
      }
      e.set_function(f);
      e.set_buffer(0, qb); e.set_buffer(1, kb); e.set_buffer(2, vb);
      e.set_buffer(3, ob);
      e.set_buffer(4, r.pb);
      e.set_buffer(7, sb);
      sage->bind(e);
      e.dispatch({32u * (unsigned)r.fn_blocks(), 4u * (unsigned)H, 1},
                 {32, 4, 1});
    }
    if (!st.commit().wait_ok(&err)) { return -1.0; }
    const auto* o = static_cast<const std::uint16_t*>(ob.contents());
    double cs = 0.0;
    for (int h = 0; h < H; ++h) {
      std::vector<double> got(refs[(std::size_t)h].size());
      for (std::size_t i = 0; i < got.size(); ++i) {
        got[i] = (double)h2f_(o[(std::size_t)h * T * kD + i]);
      }
      cs += cos_(got, refs[(std::size_t)h]) / H;
    }
    return cs;
  };
  double c[2][2] = {};  // [smooth][pv8]
  for (const bool smooth : {true, false}) {
    sage::Config sc = cfg;
    sc.smooth_k = smooth;
    for (const bool pv8 : {false, true}) {
      c[smooth ? 1 : 0][pv8 ? 1 : 0] = arm(sc, pv8, -1);
    }
  }
  sage::Config off = cfg;
  off.smooth_k = false;
  const double mut = arm(off, false, 1);
  std::printf("[sage-sinks] cos vs f64, smoothed: QK8 %.6f + PV8 %.6f | "
              "unsmoothed: QK8 %.6f + PV8 %.6f | unsmoothed with 311 set "
              "anyway (a stale mean): %.6f\n", c[1][0], c[1][1], c[0][0],
              c[0][1], mut);
  EXPECT_TRUE(c[1][0] > 0.999 && c[1][1] > 0.999);
  EXPECT_TRUE(c[0][0] > 0.999 && c[0][1] > 0.999);
  EXPECT_TRUE(mut < c[0][0] - 0.01);
}

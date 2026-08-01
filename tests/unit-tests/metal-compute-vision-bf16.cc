// The Qwen3-VL vision-tower pointwise kernels (qwen3_5_vision.metal), run in
// BOTH element types against a double-precision CPU oracle.
//
// The tower gained a bf16 storage mode so it can reproduce a reference that
// runs bf16 (Mage-Flow casts its whole text encoder). That mode is a second
// compile of the same source with -DVPIPE_ELT=bfloat, and a silently-wrong
// twin is exactly the kind of thing that shows up only as a slightly-off
// end-to-end number -- so pin the kernels themselves here, where a failure
// says WHICH kernel rather than "the tower drifted".

#include "minitest.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::metal_compute;

namespace {

std::uint16_t
to_bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
float
from_bf16_(std::uint16_t b)
{
  std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
std::uint16_t
enc_(float v, bool bf16)
{
  if (bf16) { return to_bf16_(v); }
  const _Float16 h = (_Float16)v;
  std::uint16_t b;
  std::memcpy(&b, &h, 2);
  return b;
}
float
dec_(std::uint16_t b, bool bf16)
{
  if (bf16) { return from_bf16_(b); }
  _Float16 h;
  std::memcpy(&h, &b, 2);
  return (float)h;
}

// Deterministic pseudo-random in [-1,1) -- no RNG dependency.
float
val_(int i, int salt)
{
  const int x = (i * 1103515245 + salt * 12345 + 7) & 0x7fffffff;
  return (float)((x % 20000) - 10000) / 10000.0f;
}

SharedBuffer
up_(MetalCompute* mc, const std::vector<float>& v, bool bf16)
{
  SharedBuffer b = mc->make_shared_buffer(v.size() * 2);
  auto* p = static_cast<std::uint16_t*>(b.contents());
  for (std::size_t i = 0; i < v.size(); ++i) { p[i] = enc_(v[i], bf16); }
  return b;
}

double
rel_l2_(const SharedBuffer& got, const std::vector<double>& want, bool bf16)
{
  const auto* p = static_cast<const std::uint16_t*>(got.contents());
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    const double d = (double)dec_(p[i], bf16) - want[i];
    num += d * d;
    den += want[i] * want[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

MetalCompute*
mc_(Session& s)
{
  return s.metal_compute();
}

// Both dtypes are checked against the SAME oracle with the SAME tolerance
// band per dtype: f16 and bf16 differ in mantissa bits (11 vs 8), so bf16 is
// allowed ~8x the error -- but nowhere near the O(1) a broken twin produces.
struct Arm {
  const char* lib;
  bool        bf16;
  double      tol;
};
const Arm kArms[] = {
  {"qwen3_5_vision",      false, 2e-3},
  {"qwen3_5_vision_bf16", true,  2e-2},
};

}  // namespace

TEST(metal_compute_vision_bf16, layer_norm_matches_the_cpu_oracle)
{
  Session sess;
  MetalCompute* mc = mc_(sess);
  if (mc == nullptr || !mc->valid()) { return; }

  const int R = 7, H = 128;
  const float eps = 1e-6f;
  std::vector<float> x((std::size_t)R * H), w(H), b(H);
  for (int i = 0; i < R * H; ++i) { x[(std::size_t)i] = val_(i, 1) * 4.0f; }
  for (int i = 0; i < H; ++i) {
    w[(std::size_t)i] = 1.0f + 0.25f * val_(i, 2);
    b[(std::size_t)i] = 0.5f * val_(i, 3);
  }
  std::vector<double> want((std::size_t)R * H);
  for (int r = 0; r < R; ++r) {
    double m = 0.0, s2 = 0.0;
    for (int i = 0; i < H; ++i) { m += x[(std::size_t)r * H + i]; }
    m /= H;
    for (int i = 0; i < H; ++i) {
      const double d = x[(std::size_t)r * H + i] - m;
      s2 += d * d;
    }
    const double inv = 1.0 / std::sqrt(s2 / H + eps);
    for (int i = 0; i < H; ++i) {
      want[(std::size_t)r * H + i] =
          (x[(std::size_t)r * H + i] - m) * inv * w[(std::size_t)i] +
          b[(std::size_t)i];
    }
  }

  for (const Arm& a : kArms) {
    ComputeLibrary lib = mc->load_library(a.lib);
    ComputeFunction fn = lib.function("layer_norm_bias_f16");
    ASSERT_TRUE(fn.valid());
    SharedBuffer xb = up_(mc, x, a.bf16), wb = up_(mc, w, a.bf16),
                 bb = up_(mc, b, a.bf16);
    SharedBuffer ob = mc->make_shared_buffer((std::size_t)R * H * 2);
    CommandStream st = mc->make_command_stream();
    {
      ComputeEncoder e = st.begin_compute();
      e.set_function(fn);
      e.set_buffer(0, xb);
      e.set_buffer(1, wb);
      e.set_buffer(2, bb);
      e.set_buffer(3, ob);
      e.set_constant(4, H);
      e.set_constant(5, eps);
      e.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
    }
    st.commit().wait();
    const double r = rel_l2_(ob, want, a.bf16);
    std::printf("[vision-bf16] %-22s layer_norm rel-L2 = %.6f\n", a.lib, r);
    EXPECT_TRUE(r < a.tol);
  }
}

TEST(metal_compute_vision_bf16, gelu_matches_the_cpu_oracle)
{
  Session sess;
  MetalCompute* mc = mc_(sess);
  if (mc == nullptr || !mc->valid()) { return; }

  const int N = 1024;
  std::vector<float> x((std::size_t)N);
  for (int i = 0; i < N; ++i) { x[(std::size_t)i] = val_(i, 5) * 5.0f; }
  std::vector<double> want_tanh(N), want_erf(N);
  for (int i = 0; i < N; ++i) {
    const double v = x[(std::size_t)i];
    const double inner = 0.7978845608028654 * (v + 0.044715 * v * v * v);
    want_tanh[(std::size_t)i] = 0.5 * v * (1.0 + std::tanh(inner));
    want_erf[(std::size_t)i] =
        0.5 * v * (1.0 + std::erf(v * 0.7071067811865476));
  }

  for (const Arm& a : kArms) {
    ComputeLibrary lib = mc->load_library(a.lib);
    for (int which = 0; which < 2; ++which) {
      ComputeFunction fn =
          lib.function(which == 0 ? "gelu_tanh_f16" : "gelu_erf_f16");
      ASSERT_TRUE(fn.valid());
      SharedBuffer xb = up_(mc, x, a.bf16);
      SharedBuffer ob = mc->make_shared_buffer((std::size_t)N * 2);
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder e = st.begin_compute();
        e.set_function(fn);
        e.set_buffer(0, xb);
        e.set_buffer(1, ob);
        e.set_constant(2, N);
        e.dispatch({(unsigned)N, 1, 1}, {256, 1, 1});
      }
      st.commit().wait();
      const double r =
          rel_l2_(ob, which == 0 ? want_tanh : want_erf, a.bf16);
      std::printf("[vision-bf16] %-22s %-10s rel-L2 = %.6f\n", a.lib,
                  which == 0 ? "gelu_tanh" : "gelu_erf", r);
      EXPECT_TRUE(r < a.tol);
    }
  }
}

TEST(metal_compute_vision_bf16, vision_rope_matches_the_cpu_oracle)
{
  Session sess;
  MetalCompute* mc = mc_(sess);
  if (mc == nullptr || !mc->valid()) { return; }

  const int n = 12, Hh = 4, D = 64;
  const std::size_t total = (std::size_t)n * Hh * D;
  std::vector<float> q(total), cs((std::size_t)n * D), sn((std::size_t)n * D);
  for (std::size_t i = 0; i < total; ++i) { q[i] = val_((int)i, 7) * 2.0f; }
  for (std::size_t i = 0; i < (std::size_t)n * D; ++i) {
    cs[i] = std::cos(0.01f * (float)i);
    sn[i] = std::sin(0.01f * (float)i);
  }
  std::vector<double> want(total);
  const int hd2 = D / 2;
  for (int p = 0; p < n; ++p) {
    for (int h = 0; h < Hh; ++h) {
      for (int d = 0; d < D; ++d) {
        const std::size_t base = ((std::size_t)p * Hh + h) * D;
        const double rot = (d < hd2) ? -(double)q[base + d + hd2]
                                     : (double)q[base + d - hd2];
        want[base + d] = (double)q[base + d] * cs[(std::size_t)p * D + d] +
                         rot * sn[(std::size_t)p * D + d];
      }
    }
  }

  for (const Arm& a : kArms) {
    ComputeLibrary lib = mc->load_library(a.lib);
    ComputeFunction fn = lib.function("vision_rope_f16");
    ASSERT_TRUE(fn.valid());
    SharedBuffer qb = up_(mc, q, a.bf16), cb = up_(mc, cs, a.bf16),
                 sb = up_(mc, sn, a.bf16);
    SharedBuffer ob = mc->make_shared_buffer(total * 2);
    CommandStream st = mc->make_command_stream();
    {
      ComputeEncoder e = st.begin_compute();
      e.set_function(fn);
      e.set_buffer(0, qb);
      e.set_buffer(1, cb);
      e.set_buffer(2, sb);
      e.set_buffer(3, ob);
      e.set_constant(4, Hh);
      e.set_constant(5, D);
      e.dispatch({(unsigned)total, 1, 1}, {256, 1, 1});
    }
    st.commit().wait();
    const double r = rel_l2_(ob, want, a.bf16);
    std::printf("[vision-bf16] %-22s vision_rope rel-L2 = %.6f\n", a.lib, r);
    EXPECT_TRUE(r < a.tol);
  }
}

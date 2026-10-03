// train-ops.cc -- every GPU operation LoRA training adds, against a CPU
// reference in double precision.
//
// Two layers of checking, because a backward can be wrong in two ways:
//
//  * the DERIVATION -- each backward formula is checked against central
//    differences of its own forward, on the CPU, before any GPU is
//    involved (the *_formula tests);
//  * the KERNEL -- the GPU result against that formula on the same
//    bf16-rounded inputs.
//
// The attention backward is driven the way a DiT drives it: the forward
// is the real steel kernel with its statistics exported, over a
// block-causal layout of a causal text run and a dense image block, at
// lengths that leave partial tiles on both sides.
//
//   vpipe_test --filter 'train_ops*'

#include "minitest.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "common/session.h"
#include "generative-models/train/train-ops.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace vpipe;
using metal_compute::SharedBuffer;

namespace {

std::uint16_t
bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (std::uint16_t)(u >> 16);
}

float
from_bf16_(std::uint16_t h)
{
  const std::uint32_t u = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

float
round_bf16_(float f)
{
  return from_bf16_(bf16_(f));
}

std::vector<float>
randn_(std::size_t n, float sd, std::uint64_t seed)
{
  std::mt19937_64 g(seed);
  std::normal_distribution<float> d(0.0f, sd);
  std::vector<float> v(n);
  for (float& x : v) { x = round_bf16_(d(g)); }
  return v;
}

struct Gpu {
  Session sess;
  metal_compute::MetalCompute* mc = nullptr;
  genai::train::TrainOps ops;
  bool ok = false;
  Gpu()
  {
    mc = sess.metal_compute();
    if (mc == nullptr) { return; }
    std::string err;
    ok = ops.init(mc, /*bf16=*/true, &err);
    if (!ok) { std::printf("[train_ops] init: %s\n", err.c_str()); }
  }
  SharedBuffer
  bf(const std::vector<float>& v)
  {
    SharedBuffer b =
        mc->make_shared_buffer(std::max<std::size_t>(v.size(), 1) * 2);
    auto* d = static_cast<std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < v.size(); ++i) { d[i] = bf16_(v[i]); }
    return b;
  }
  SharedBuffer
  f32(const std::vector<float>& v)
  {
    SharedBuffer b =
        mc->make_shared_buffer(std::max<std::size_t>(v.size(), 1) * 4);
    std::memcpy(b.contents(), v.data(), v.size() * 4);
    return b;
  }
  SharedBuffer
  zeros(std::size_t bytes)
  {
    SharedBuffer b = mc->make_shared_buffer(std::max<std::size_t>(bytes, 4));
    std::memset(b.contents(), 0, b.byte_size());
    return b;
  }
  bool
  run(const std::function<void(metal_compute::ComputeEncoder&)>& fn)
  {
    metal_compute::CommandStream st = mc->make_command_stream();
    metal_compute::ComputeEncoder enc = st.begin_compute();
    fn(enc);
    enc.end();
    std::string e;
    const bool r = st.commit().wait_ok(&e);
    if (!r) { std::printf("[train_ops] gpu: %s\n", e.c_str()); }
    return r;
  }
};

std::vector<float>
read_bf_(const SharedBuffer& b, std::size_t n, std::size_t off = 0)
{
  std::vector<float> v(n);
  const auto* d = static_cast<const std::uint16_t*>(b.contents()) + off;
  for (std::size_t i = 0; i < n; ++i) { v[i] = from_bf16_(d[i]); }
  return v;
}

std::vector<float>
read_f32_(const SharedBuffer& b, std::size_t n, std::size_t off = 0)
{
  std::vector<float> v(n);
  std::memcpy(v.data(), static_cast<const float*>(b.contents()) + off, n * 4);
  return v;
}

double
rel_l2_(const std::vector<float>& got, const std::vector<double>& want)
{
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    const double d = (double)got[i] - want[i];
    num += d * d;
    den += want[i] * want[i];
  }
  return std::sqrt(num / std::max(den, 1e-30));
}

// ---- CPU forwards and their analytic backwards --------------------------

// y = LN(x) * (1 + s), one row.
std::vector<double>
ln_mod_fwd_(const std::vector<double>& x, const std::vector<double>& s,
            double eps)
{
  const std::size_t H = x.size();
  double m = 0.0, q = 0.0;
  for (double v : x) { m += v; q += v * v; }
  m /= (double)H;
  const double r = 1.0 / std::sqrt(q / (double)H - m * m + eps);
  std::vector<double> y(H);
  for (std::size_t i = 0; i < H; ++i) { y[i] = (1.0 + s[i]) * (x[i] - m) * r; }
  return y;
}

std::vector<double>
ln_mod_bwd_(const std::vector<double>& x, const std::vector<double>& s,
            const std::vector<double>& dy, double eps)
{
  const std::size_t H = x.size();
  double m = 0.0, q = 0.0;
  for (double v : x) { m += v; q += v * v; }
  m /= (double)H;
  const double r = 1.0 / std::sqrt(q / (double)H - m * m + eps);
  double mg = 0.0, mgx = 0.0;
  for (std::size_t i = 0; i < H; ++i) {
    const double g = dy[i] * (1.0 + s[i]);
    mg += g;
    mgx += g * (x[i] - m) * r;
  }
  mg /= (double)H;
  mgx /= (double)H;
  std::vector<double> dx(H);
  for (std::size_t i = 0; i < H; ++i) {
    const double g = dy[i] * (1.0 + s[i]);
    dx[i] = r * (g - mg - (x[i] - m) * r * mgx);
  }
  return dx;
}

std::vector<double>
rms_fwd_(const std::vector<double>& x, const std::vector<double>& w,
         double eps)
{
  double q = 0.0;
  for (double v : x) { q += v * v; }
  const double r = 1.0 / std::sqrt(q / (double)x.size() + eps);
  std::vector<double> y(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) { y[i] = x[i] * r * w[i]; }
  return y;
}

std::vector<double>
rms_bwd_(const std::vector<double>& x, const std::vector<double>& w,
         const std::vector<double>& dy, double eps)
{
  const std::size_t D = x.size();
  double q = 0.0;
  for (double v : x) { q += v * v; }
  const double r = 1.0 / std::sqrt(q / (double)D + eps);
  double dot = 0.0;
  for (std::size_t i = 0; i < D; ++i) { dot += dy[i] * w[i] * x[i] * r; }
  dot /= (double)D;
  std::vector<double> dx(D);
  for (std::size_t i = 0; i < D; ++i) {
    dx[i] = r * (dy[i] * w[i] - x[i] * r * dot);
  }
  return dx;
}

// Numerical gradient of sum(dy * f(x)) by central differences.
std::vector<double>
num_grad_(const std::function<std::vector<double>(const std::vector<double>&)>&
              f,
          std::vector<double> x, const std::vector<double>& dy)
{
  std::vector<double> g(x.size());
  const double h = 1e-5;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const double x0 = x[i];
    x[i] = x0 + h;
    const std::vector<double> yp = f(x);
    x[i] = x0 - h;
    const std::vector<double> ym = f(x);
    x[i] = x0;
    double s = 0.0;
    for (std::size_t j = 0; j < yp.size(); ++j) {
      s += dy[j] * (yp[j] - ym[j]);
    }
    g[i] = s / (2.0 * h);
  }
  return g;
}

double
max_rel_(const std::vector<double>& a, const std::vector<double>& b)
{
  double w = 0.0, scale = 0.0;
  for (double v : b) { scale = std::max(scale, std::fabs(v)); }
  for (std::size_t i = 0; i < a.size(); ++i) {
    w = std::max(w, std::fabs(a[i] - b[i]) / std::max(scale, 1e-12));
  }
  return w;
}

std::vector<double>
to_d_(const std::vector<float>& v)
{
  return std::vector<double>(v.begin(), v.end());
}

}  // namespace

// ---- the derivations -----------------------------------------------------

TEST(train_ops, ln_mod_and_rms_formulas_match_central_differences)
{
  const std::size_t H = 24;
  const auto x = to_d_(randn_(H, 1.3f, 1));
  const auto s = to_d_(randn_(H, 0.4f, 2));
  const auto dy = to_d_(randn_(H, 1.0f, 3));
  const double eps = 1e-6;
  const auto an = ln_mod_bwd_(x, s, dy, eps);
  const auto nu = num_grad_(
      [&](const std::vector<double>& xx) { return ln_mod_fwd_(xx, s, eps); },
      x, dy);
  EXPECT_TRUE(max_rel_(an, nu) < 1e-6);

  const auto w = to_d_(randn_(H, 0.8f, 4));
  const auto ar = rms_bwd_(x, w, dy, eps);
  const auto nr = num_grad_(
      [&](const std::vector<double>& xx) { return rms_fwd_(xx, w, eps); }, x,
      dy);
  EXPECT_TRUE(max_rel_(ar, nr) < 1e-6);
}

// ---- the kernels ---------------------------------------------------------

TEST(train_ops, transpose_lands_in_a_column_band)
{
  Gpu g;
  if (!g.ok) { return; }
  const int R = 70, C = 45, ldo = 200, off = 37;
  const auto in = randn_((std::size_t)R * C, 1.0f, 10);
  SharedBuffer bi = g.bf(in);
  SharedBuffer bo = g.zeros((std::size_t)C * ldo * 2);
  ASSERT_TRUE(g.run([&](metal_compute::ComputeEncoder& e) {
    g.ops.transpose(e, bi, 0, bo, 0, R, C, C, ldo, off);
  }));
  const auto out = read_bf_(bo, (std::size_t)C * ldo);
  int wrong = 0;
  for (int r = 0; r < R; ++r) {
    for (int c = 0; c < C; ++c) {
      if (out[(std::size_t)c * ldo + off + r] != in[(std::size_t)r * C + c]) {
        ++wrong;
      }
    }
  }
  EXPECT_TRUE(wrong == 0);
}

TEST(train_ops, gemm_tn_nn_nt_match_the_cpu)
{
  Gpu g;
  if (!g.ok) { return; }
  // TN over the rows, both tile shapes, accumulating onto what is there.
  for (int shape = 0; shape < 2; ++shape) {
    const int M = 301;
    const int P = shape == 0 ? 16 : 130;
    const int Q = shape == 0 ? 100 : 16;
    const auto a = randn_((std::size_t)M * P, 1.0f, 20 + shape);
    const auto b = randn_((std::size_t)M * Q, 1.0f, 30 + shape);
    const auto c0 = randn_((std::size_t)P * Q, 1.0f, 40 + shape);
    SharedBuffer ba = g.bf(a), bb = g.bf(b), bc = g.f32(c0);
    ASSERT_TRUE(g.run([&](metal_compute::ComputeEncoder& e) {
      g.ops.gemm_tn_f32(e, ba, 0, bb, 0, bc, 0, M, P, Q, P, Q, Q, 0.5f,
                        true);
    }));
    std::vector<double> want((std::size_t)P * Q);
    for (int p = 0; p < P; ++p) {
      for (int q = 0; q < Q; ++q) {
        double s = 0.0;
        for (int m = 0; m < M; ++m) {
          s += (double)a[(std::size_t)m * P + p] * b[(std::size_t)m * Q + q];
        }
        want[(std::size_t)p * Q + q] = c0[(std::size_t)p * Q + q] + 0.5 * s;
      }
    }
    const double e = rel_l2_(read_f32_(bc, want.size()), want);
    std::printf("[train_ops] tn %dx%d rel-L2 %.2e\n", P, Q, e);
    EXPECT_TRUE(e < 1e-5);
  }
  // NN into bf16, with and without accumulate; NT likewise.
  for (int nt = 0; nt < 2; ++nt) {
    for (int acc = 0; acc < 2; ++acc) {
      const int M = 77, N = nt == 0 ? 20 : 90, K = 33;
      const auto a = randn_((std::size_t)M * K, 1.0f, 50);
      const auto b = randn_((std::size_t)K * N, 1.0f, 51);
      const auto c0 = randn_((std::size_t)M * N, 1.0f, 52);
      SharedBuffer ba = g.bf(a), bb = g.bf(b), bc = g.bf(c0);
      ASSERT_TRUE(g.run([&](metal_compute::ComputeEncoder& e) {
        if (nt == 0) {
          g.ops.gemm_nn(e, ba, 0, bb, 0, bc, 0, M, N, K, K, N, N, 2.0f,
                        acc != 0);
        } else {
          // b read as [N, K]
          g.ops.gemm_nt(e, ba, 0, bb, 0, bc, 0, M, N, K, K, K, N, 2.0f,
                        acc != 0);
        }
      }));
      std::vector<double> want((std::size_t)M * N);
      for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
          double s = 0.0;
          for (int k = 0; k < K; ++k) {
            const float bv = nt == 0 ? b[(std::size_t)k * N + n]
                                     : b[(std::size_t)n * K + k];
            s += (double)a[(std::size_t)m * K + k] * bv;
          }
          want[(std::size_t)m * N + n] =
              (acc != 0 ? c0[(std::size_t)m * N + n] : 0.0) + 2.0 * s;
        }
      }
      const double e = rel_l2_(read_bf_(bc, want.size()), want);
      std::printf("[train_ops] %s acc=%d rel-L2 %.2e\n",
                  nt == 0 ? "nn" : "nt", acc, e);
      EXPECT_TRUE(e < 5e-3);
    }
  }
}

TEST(train_ops, elementwise_backwards_match_the_cpu)
{
  Gpu g;
  if (!g.ok) { return; }
  const double eps = 1e-6;
  // LN * (1 + s): bf16 dout, and f32 dout; accumulates into dx.
  {
    const int rows = 5, H = 1000;
    const auto x = randn_((std::size_t)rows * H, 1.5f, 60);
    const auto s = randn_(H, 0.3f, 61);
    const auto dy = randn_((std::size_t)rows * H, 1.0f, 62);
    const auto dx0 = randn_((std::size_t)rows * H, 0.1f, 63);
    for (int f32 = 0; f32 < 2; ++f32) {
      SharedBuffer bx = g.bf(x), bs = g.bf(s);
      SharedBuffer bdy = f32 != 0 ? g.f32(dy) : g.bf(dy);
      SharedBuffer bdx = g.f32(dx0);
      ASSERT_TRUE(g.run([&](metal_compute::ComputeEncoder& e) {
        g.ops.ln_mod_bwd(e, bx, 0, bdy, 0, f32 != 0, bs, 0, bdx, 0, rows, H,
                         (float)eps);
      }));
      std::vector<double> want;
      for (int r = 0; r < rows; ++r) {
        std::vector<double> xr(x.begin() + (long)r * H,
                               x.begin() + (long)(r + 1) * H);
        std::vector<double> dr(dy.begin() + (long)r * H,
                               dy.begin() + (long)(r + 1) * H);
        const auto d = ln_mod_bwd_(xr, to_d_(s), dr, eps);
        for (int i = 0; i < H; ++i) {
          want.push_back(d[(std::size_t)i] + dx0[(std::size_t)r * H + i]);
        }
      }
      const double e = rel_l2_(read_f32_(bdx, want.size()), want);
      std::printf("[train_ops] ln_mod_bwd f32=%d rel-L2 %.2e\n", f32, e);
      EXPECT_TRUE(e < 1e-5);
    }
  }
  // gated tanh.
  {
    const int rows = 7, N = 300;
    const auto dh = randn_((std::size_t)rows * N, 1.0f, 70);
    const auto gt = randn_(N, 1.0f, 71);
    SharedBuffer bdh = g.f32(dh), bg = g.bf(gt);
    SharedBuffer bo = g.zeros((std::size_t)rows * N * 2);
    ASSERT_TRUE(g.run([&](metal_compute::ComputeEncoder& e) {
      g.ops.gated_tanh_bwd(e, bdh, 0, bg, 0, bo, 0, rows, N);
    }));
    std::vector<double> want((std::size_t)rows * N);
    for (int r = 0; r < rows; ++r) {
      for (int n = 0; n < N; ++n) {
        want[(std::size_t)r * N + n] =
            std::tanh((double)gt[(std::size_t)n]) * dh[(std::size_t)r * N + n];
      }
    }
    EXPECT_TRUE(rel_l2_(read_bf_(bo, want.size()), want) < 5e-3);
  }
  // SwiGLU.
  {
    const std::size_t n = 1111;
    const auto gv = randn_(n, 2.0f, 80);
    const auto uv = randn_(n, 1.0f, 81);
    const auto ds = randn_(n, 1.0f, 82);
    SharedBuffer bg = g.bf(gv), bu = g.bf(uv), bds = g.bf(ds);
    SharedBuffer bdg = g.zeros(n * 2), bdu = g.zeros(n * 2);
    ASSERT_TRUE(g.run([&](metal_compute::ComputeEncoder& e) {
      g.ops.swiglu_bwd(e, bg, bu, bds, bdg, 0, bdu, 0, 1, (int)n, (int)n);
    }));
    std::vector<double> wg(n), wu(n);
    for (std::size_t i = 0; i < n; ++i) {
      const double x = gv[i];
      const double sg = 1.0 / (1.0 + std::exp(-x));
      wu[i] = ds[i] * x * sg;
      wg[i] = ds[i] * uv[i] * sg * (1.0 + x * (1.0 - sg));
      // The derivative itself, by central differences of silu(x) * u.
      const double h = 1e-6;
      auto f = [&](double z) { return z / (1.0 + std::exp(-z)) * uv[i]; };
      const double fd = ds[i] * (f(x + h) - f(x - h)) / (2 * h);
      if (i < 20) {
        EXPECT_TRUE(std::fabs(fd - wg[i]) <=
                    1e-6 * std::max(1.0, std::fabs(fd)));
      }
    }
    EXPECT_TRUE(rel_l2_(read_bf_(bdg, n), wg) < 5e-3);
    EXPECT_TRUE(rel_l2_(read_bf_(bdu, n), wu) < 5e-3);
  }
  // RMS with a frozen weight, D = 128 as the q/k norms have it.
  {
    const int rows = 37, D = 128;
    const auto x = randn_((std::size_t)rows * D, 2.0f, 90);
    const auto w = randn_(D, 0.5f, 91);
    const auto dy = randn_((std::size_t)rows * D, 1.0f, 92);
    SharedBuffer bx = g.bf(x), bw = g.bf(w), bdy = g.bf(dy);
    SharedBuffer bdx = g.zeros((std::size_t)rows * D * 2);
    ASSERT_TRUE(g.run([&](metal_compute::ComputeEncoder& e) {
      g.ops.rms_bwd(e, bx, 0, bdy, 0, bw, bdx, 0, rows, D, (float)eps, 1, D);
    }));
    std::vector<double> want;
    for (int r = 0; r < rows; ++r) {
      std::vector<double> xr(x.begin() + (long)r * D,
                             x.begin() + (long)(r + 1) * D);
      std::vector<double> dr(dy.begin() + (long)r * D,
                             dy.begin() + (long)(r + 1) * D);
      const auto d = rms_bwd_(xr, to_d_(w), dr, eps);
      want.insert(want.end(), d.begin(), d.end());
    }
    EXPECT_TRUE(rel_l2_(read_bf_(bdx, want.size()), want) < 5e-3);
  }
  // RoPE: the backward is the inverse rotation, and both input types.
  {
    const int H = 3, T = 11, D = 8;
    std::vector<float> cs((std::size_t)(T + 4) * D), sn(cs.size());
    for (int t = 0; t < T + 4; ++t) {
      for (int i = 0; i < D / 2; ++i) {
        const double th = 0.37 * t / std::pow(3.0, i);
        cs[(std::size_t)t * D + 2 * i] = cs[(std::size_t)t * D + 2 * i + 1] =
            (float)std::cos(th);
        sn[(std::size_t)t * D + 2 * i] = sn[(std::size_t)t * D + 2 * i + 1] =
            (float)std::sin(th);
      }
    }
    const int row = 4;   // the table bound at the step's first row
    const auto dy = randn_((std::size_t)H * T * D, 1.0f, 100);
    std::vector<double> want((std::size_t)T * H * D);
    for (int h = 0; h < H; ++h) {
      for (int t = 0; t < T; ++t) {
        for (int i = 0; i < D / 2; ++i) {
          const std::size_t cb = (std::size_t)(t + row) * D + 2 * i;
          const double c = cs[cb], s = sn[cb];
          const std::size_t ib = ((std::size_t)h * T + t) * D + 2 * i;
          const std::size_t ob = ((std::size_t)t * H + h) * D + 2 * i;
          // forward: o1 = x1 c - x2 s; o2 = x1 s + x2 c  => transpose
          want[ob] = dy[ib] * c + dy[ib + 1] * s;
          want[ob + 1] = -dy[ib] * s + dy[ib + 1] * c;
        }
      }
    }
    SharedBuffer bc = g.f32(cs), bs = g.f32(sn);
    for (int f32 = 0; f32 < 2; ++f32) {
      SharedBuffer bin = f32 != 0 ? g.f32(dy) : g.bf(dy);
      SharedBuffer bo = g.zeros(want.size() * 2);
      ASSERT_TRUE(g.run([&](metal_compute::ComputeEncoder& e) {
        g.ops.rope_bwd(e, bin, f32 != 0, bo, bc, bs, row, H, T, D);
      }));
      EXPECT_TRUE(rel_l2_(read_bf_(bo, want.size()), want) < 5e-3);
    }
  }
}

// The flash attention's backward, driven as a DiT drives it.
// ATTENTION BACKWARD, every form against one double-precision reference:
// the fused flash pair, and the GEMM form on the steel GEMMs and (where
// the GPU has them) on the matrix cores -- the latter also with a scratch
// budget of one head, so the head-group loop runs. Two geometries: the
// small one, and one whose segments span several tiles of every kernel
// in every dimension and end ragged (kL 340 against 32-key blocks, 64- and
// 128-wide GEMM tiles).
TEST(train_ops, attention_backward_matches_the_cpu_over_block_causal_bands)
{
  Gpu g;
  if (!g.ok) { return; }
  const int D = 128;
  const float scale = 1.0f / std::sqrt((float)D);
  struct Geo { int H, TXT, IMG; };
  for (const Geo geo : {Geo{2, 21, 75}, Geo{3, 40, 300}}) {
    const int H = geo.H, TXT = geo.TXT, IMG = geo.IMG, T = TXT + IMG;
    // Head-major [H][T][D].
    const auto q = randn_((std::size_t)H * T * D, 1.0f, 200);
    const auto k = randn_((std::size_t)H * T * D, 1.0f, 201);
    const auto v = randn_((std::size_t)H * T * D, 1.0f, 202);
    const auto dO = randn_((std::size_t)H * T * D, 1.0f, 203);
    SharedBuffer bq = g.bf(q), bk = g.bf(k), bv = g.bf(v), bdo = g.bf(dO);
    SharedBuffer bo = g.zeros((std::size_t)H * T * D * 2);
    SharedBuffer mlm = g.zeros((std::size_t)H * T * 4);
    SharedBuffer mls = g.zeros((std::size_t)H * T * 4);

    // The segments, as the forward's block-causal decomposition has them.
    std::vector<genai::train::AttnSegment> segs = {
        {0, TXT, TXT, 0, true},
        {TXT, IMG, T, 0, false},
    };

    // Forward: the real steel kernel, statistics exported per segment.
    struct SteelAttnParams {
      int B, H, D, qL, kL, gqa_factor;
      float scale;
      int NQ, NK, NQ_aligned, NK_aligned, qL_rem, kL_rem, qL_off;
      std::int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
    };
    metal_compute::ComputeLibrary lib = g.mc->load_library("attn_steel");
    ASSERT_TRUE(g.run([&](metal_compute::ComputeEncoder& e) {
      for (const auto& s : segs) {
        SteelAttnParams p{};
        p.B = 1; p.H = H; p.D = D; p.qL = s.qL; p.kL = s.kL;
        p.gqa_factor = 1; p.scale = scale;
        p.NQ = (s.qL + 31) / 32; p.NK = (s.kL + 15) / 16;
        p.NQ_aligned = s.qL / 32; p.NK_aligned = s.kL / 16;
        p.qL_rem = s.qL - p.NQ_aligned * 32;
        p.kL_rem = s.kL - p.NK_aligned * 16;
        p.qL_off = s.qL_off;
        const std::int64_t hs = (std::int64_t)T * D;
        p.Q_strides[0] = H * hs; p.Q_strides[1] = hs; p.Q_strides[2] = D;
        p.K_strides[0] = H * hs; p.K_strides[1] = hs; p.K_strides[2] = D;
        p.V_strides[0] = H * hs; p.V_strides[1] = hs; p.V_strides[2] = D;
        p.O_strides[0] = H * hs; p.O_strides[1] = hs; p.O_strides[2] = D;
        metal_compute::FunctionConstants fc;
        fc.set_bool(200, (s.qL % 32) == 0).set_bool(201, (s.kL % 16) == 0)
            .set_bool(300, false).set_bool(301, s.causal)
            .set_bool(302, false).set_bool(304, true);
        metal_compute::ComputeFunction fn =
            lib.function("attn_steel_h_bd128_bf16", fc);
        e.set_function(fn);
        const std::size_t qo = (std::size_t)s.q_row0 * D * 2;
        e.set_buffer(0, bq, qo); e.set_buffer(1, bk); e.set_buffer(2, bv);
        e.set_buffer(3, bo, qo);
        e.set_constant(4, p);
        const std::size_t mo = (std::size_t)H * s.q_row0 * 4;
        e.set_buffer(12, mlm, mo); e.set_buffer(13, mls, mo);
        e.dispatch({(unsigned)(32 * p.NQ), (unsigned)(4 * H), 1},
                   {32, 4, 1});
      }
    }));

    // Reference: the dense attention under the block-causal rule, in
    // double.
    auto allowed = [&](int qi, int kj) {
      if (qi < TXT) { return kj <= qi; }     // the causal text run
      return true;                           // the image block sees all
    };
    std::vector<double> rdq((std::size_t)H * T * D, 0.0),
        rdk((std::size_t)H * T * D, 0.0), rdv((std::size_t)H * T * D, 0.0),
        ro((std::size_t)H * T * D, 0.0);
    for (int h = 0; h < H; ++h) {
      auto at = [&](const std::vector<float>& a, int t, int d) {
        return (double)a[((std::size_t)h * T + t) * D + d];
      };
      for (int i = 0; i < T; ++i) {
        std::vector<double> p(T, 0.0), dp(T, 0.0);
        double mx = -1e300;
        for (int j = 0; j < T; ++j) {
          if (!allowed(i, j)) { continue; }
          double sv = 0.0;
          for (int d = 0; d < D; ++d) { sv += at(q, i, d) * at(k, j, d); }
          p[j] = sv * scale;
          mx = std::max(mx, p[j]);
        }
        double den = 0.0;
        for (int j = 0; j < T; ++j) {
          p[j] = allowed(i, j) ? std::exp(p[j] - mx) : 0.0;
          den += p[j];
        }
        for (int j = 0; j < T; ++j) { p[j] /= den; }
        double dsm = 0.0;
        for (int j = 0; j < T; ++j) {
          if (!allowed(i, j)) { continue; }
          double sv = 0.0;
          for (int d = 0; d < D; ++d) { sv += at(dO, i, d) * at(v, j, d); }
          dp[j] = sv;
          dsm += p[j] * sv;
        }
        for (int j = 0; j < T; ++j) {
          if (p[j] == 0.0) { continue; }
          const double ds = p[j] * (dp[j] - dsm);
          for (int d = 0; d < D; ++d) {
            const std::size_t io = ((std::size_t)h * T + i) * D + d;
            const std::size_t jo = ((std::size_t)h * T + j) * D + d;
            rdq[io] += scale * ds * at(k, j, d);
            rdk[jo] += scale * ds * at(q, i, d);
            rdv[jo] += p[j] * at(dO, i, d);
            ro[io] += p[j] * at(v, j, d);
          }
        }
      }
    }
    const double eo = rel_l2_(read_bf_(bo, ro.size()), ro);
    EXPECT_TRUE(eo < 1e-2);

    // 0 fused; 1 GEMM form, steel; 2 the same with one head per group;
    // 3, 4 the GEMM form on the matrix cores, likewise.
    for (int form = 0; form < 5; ++form) {
      const bool mma = form >= 3;
      if (mma && !g.mc->supports_matrix_cores()) { break; }
      g.ops.set_attn_gemm(form > 0);
      g.ops.set_attn_mma(mma);
      // One head's planes for the image segment, the widest.
      g.ops.set_attn_scratch_budget(
          form == 2 || form == 4 ? (std::size_t)2 * IMG * T * 2 : 0);
      if (mma) { ASSERT_TRUE(g.ops.attn_mma()); }
      SharedBuffer lse2 = g.zeros((std::size_t)H * T * 4);
      SharedBuffer dsum = g.zeros((std::size_t)H * T * 4);
      SharedBuffer bdq = g.zeros((std::size_t)H * T * D * 2);
      SharedBuffer bdk = g.zeros((std::size_t)H * T * D * 4);
      SharedBuffer bdv = g.zeros((std::size_t)H * T * D * 4);
      ASSERT_TRUE(g.run([&](metal_compute::ComputeEncoder& e) {
        g.ops.attn_backward(e, bq, bk, bv, bo, bdo, mlm, mls, lse2, dsum,
                            bdq, bdk, bdv, H, T, D, scale, segs);
      }));
      const double eq = rel_l2_(read_bf_(bdq, rdq.size()), rdq);
      const double ek = rel_l2_(read_f32_(bdk, rdk.size()), rdk);
      const double ev = rel_l2_(read_f32_(bdv, rdv.size()), rdv);
      const char* names[5] = {"fused", "gemm/steel", "gemm/steel 1-head",
                              "gemm/mma", "gemm/mma 1-head"};
      std::printf("[train_ops] attention H%d T%d %-15s rel-L2: O %.2e "
                  "dQ %.2e dK %.2e dV %.2e\n", H, T, names[form], eo, eq,
                  ek, ev);
      // dQ is stored bf16; dK / dV are the f32 accumulators, but their
      // inputs (O, the statistics -- and in the GEMM form P and dS) are
      // bf16.
      EXPECT_TRUE(eq < 1e-2);
      EXPECT_TRUE(ek < 1e-2);
      EXPECT_TRUE(ev < 1e-2);
    }
    g.ops.set_attn_gemm(true);
    g.ops.set_attn_mma(true);
    g.ops.set_attn_scratch_budget(0);
  }
}

TEST(train_ops, flow_loss_and_reductions_match_the_cpu)
{
  Gpu g;
  if (!g.ok) { return; }
  const int n = 1000;
  const auto pred = randn_(n, 1.0f, 300);
  const auto x0 = randn_(n, 1.0f, 301);
  const auto ns = randn_(n, 1.0f, 302);
  SharedBuffer bp = g.bf(pred), bx = g.bf(x0), bn = g.bf(ns);
  SharedBuffer bd = g.zeros((std::size_t)n * 2);
  const int groups = genai::train::TrainOps::flow_loss_groups(n);
  SharedBuffer part = g.zeros((std::size_t)groups * 4);
  const float coef = 2.0f / (float)n;
  ASSERT_TRUE(g.run([&](metal_compute::ComputeEncoder& e) {
    (void)g.ops.flow_loss(e, bp, bx, bn, bd, part, n, coef);
  }));
  double want = 0.0;
  std::vector<double> wd(n);
  for (int i = 0; i < n; ++i) {
    const double e = (double)pred[i] - ((double)ns[i] - x0[i]);
    want += e * e;
    wd[i] = coef * e;
  }
  double got = 0.0;
  for (float v : read_f32_(part, (std::size_t)groups)) { got += v; }
  EXPECT_TRUE(std::fabs(got - want) / want < 1e-5);
  EXPECT_TRUE(rel_l2_(read_bf_(bd, (std::size_t)n), wd) < 5e-3);

  const std::size_t m = 70001;
  const auto x = randn_(m, 1.0f, 303);
  SharedBuffer bxx = g.f32(x);
  const int rg = genai::train::TrainOps::reduce_groups(m);
  SharedBuffer p2 = g.zeros((std::size_t)rg * 8);
  ASSERT_TRUE(g.run([&](metal_compute::ComputeEncoder& e) {
    (void)g.ops.sqnorm_partial(e, bxx, 0, m, p2, 0);
    (void)g.ops.abs_partial(e, bxx, 0, m, p2, (std::size_t)rg);
  }));
  double sq = 0.0, ab = 0.0, gsq = 0.0, gab = 0.0;
  for (float v : x) { sq += (double)v * v; ab += std::fabs(v); }
  const auto pp = read_f32_(p2, (std::size_t)rg * 2);
  for (int i = 0; i < rg; ++i) { gsq += pp[i]; gab += pp[rg + i]; }
  EXPECT_TRUE(std::fabs(gsq - sq) / sq < 1e-5);
  EXPECT_TRUE(std::fabs(gab - ab) / ab < 1e-5);
}

TEST(train_ops, optimizers_step_and_write_the_loaders_shadow)
{
  Gpu g;
  if (!g.ok) { return; }
  const int n = 513;
  const auto p0 = randn_(n, 0.1f, 400);
  const auto gr = randn_(n, 1.0f, 401);
  genai::train::OptParams o;
  o.lr = 1e-2f; o.beta1 = 0.9f; o.beta2 = 0.99f; o.eps = 1e-8f;
  o.weight_decay = 0.01f; o.bc1 = 1.0f - 0.9f; o.bc2 = 1.0f - 0.99f;
  o.grad_scale = 0.5f; o.mul = 0.125f; o.n = n;
  SharedBuffer bp = g.f32(p0), bg = g.f32(gr);
  SharedBuffer bm = g.zeros((std::size_t)n * 4);
  SharedBuffer bv = g.zeros((std::size_t)n * 4);
  SharedBuffer bs = g.zeros((std::size_t)n * 2);
  ASSERT_TRUE(g.run([&](metal_compute::ComputeEncoder& e) {
    g.ops.adamw(e, bp, bg, bm, bv, bs, o);
  }));
  const auto pn = read_f32_(bp, (std::size_t)n);
  const auto gz = read_f32_(bg, (std::size_t)n);
  const auto sh = static_cast<const std::uint16_t*>(bs.contents());
  int bad = 0, shadow_bad = 0, zero_bad = 0;
  for (int i = 0; i < n; ++i) {
    const double gi = (double)gr[i] * 0.5;
    const double m = 0.1 * gi, v = 0.01 * gi * gi;
    double p = p0[i];
    p -= 1e-2 * 0.01 * p;
    p -= 1e-2 * (m / 0.1) / (std::sqrt(v / 0.01) + 1e-8);
    if (std::fabs(p - pn[i]) > 1e-6 * std::max(1.0, std::fabs(p))) { ++bad; }
    // Bit for bit what the runtime-LoRA loader makes of a bf16 file.
    if (sh[i] != bf16_(from_bf16_(bf16_(pn[i])) * 0.125f)) { ++shadow_bad; }
    if (gz[i] != 0.0f) { ++zero_bad; }
  }
  EXPECT_TRUE(bad == 0);
  EXPECT_TRUE(shadow_bad == 0);
  EXPECT_TRUE(zero_bad == 0);
}

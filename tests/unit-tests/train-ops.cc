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
#include "generative-models/train/lora-params.h"
#include "generative-models/train/optimizer.h"
#include "generative-models/train/train-ops.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <random>
#include <string>
#include <vector>

using namespace vpipe;
using metal_compute::SharedBuffer;

namespace {
std::string
fs_tmp_()
{
  const char* t = std::getenv("TMPDIR");
  return t != nullptr ? std::string(t) : std::string("/tmp");
}
}  // namespace

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

// ---- MiniMax-H3's block -------------------------------------------------
//
// Each H3 backward against central differences of its own forward in
// double (f(x) = sum dy . y(x)), on the GPU's bf16 inputs: the RMSNorm +
// per-row modulation (rows reading table rows by index), the plain gate,
// SwiGLU off a fused [gate | up], the partial rotate-half RoPE written
// back to a strided fused projection, the per-head RMSNorm in place, and
// the banded adapter B's expand / extract.
TEST(train_ops, h3_block_backwards_match_central_differences)
{
  Gpu g;
  if (!g.ok) { return; }
  const int rows = 5, H = 96, stride = 6 * H, n_mod = 4;
  const float eps = 1e-5f;
  const std::vector<float> x = randn_((std::size_t)rows * H, 1.0f, 11);
  const std::vector<float> dy = randn_((std::size_t)rows * H, 1.0f, 12);
  const std::vector<float> w = randn_((std::size_t)H, 0.5f, 13);
  const std::vector<float> mod = randn_((std::size_t)n_mod * stride, 0.3f, 14);
  const std::vector<int> idx = {2, 0, 3, 3, 1};
  SharedBuffer bx = g.bf(x), bdy = g.bf(dy), bw = g.bf(w), bm = g.bf(mod);
  SharedBuffer bi = g.mc->make_shared_buffer(idx.size() * 4);
  std::memcpy(bi.contents(), idx.data(), idx.size() * 4);
  // dx starts at 1: the kernel ADDS.
  SharedBuffer bdx = g.f32(std::vector<float>((std::size_t)rows * H, 1.0f));
  const int scale_off = 4 * H;
  ASSERT_TRUE(g.run([&](metal_compute::ComputeEncoder& enc) {
    g.ops.rms_mod_bwd(enc, bx, 0, bdy, 0, false, bw, bm, 0, bi, 0, bdx, 0,
                      rows, H, eps, stride, scale_off, true);
  }));
  {
    const std::vector<float> got = read_f32_(bdx, (std::size_t)rows * H);
    std::vector<double> want((std::size_t)rows * H);
    for (int r = 0; r < rows; ++r) {
      auto f = [&](const std::vector<double>& xr) {
        double ss = 0.0;
        for (double v : xr) { ss += v * v; }
        const double inv = 1.0 / std::sqrt(ss / H + eps);
        double acc = 0.0;
        for (int i = 0; i < H; ++i) {
          const double sc = mod[(std::size_t)idx[r] * stride + scale_off + i];
          acc += dy[(std::size_t)r * H + i] * (1.0 + sc) * xr[i] * inv * w[i];
        }
        return acc;
      };
      std::vector<double> xr(x.begin() + r * H, x.begin() + (r + 1) * H);
      for (int i = 0; i < H; ++i) {
        const double h = 1e-4, o = xr[i];
        xr[i] = o + h; const double p = f(xr);
        xr[i] = o - h; const double m = f(xr);
        xr[i] = o;
        want[(std::size_t)r * H + i] = 1.0 + (p - m) / (2.0 * h);
      }
    }
    const double e = rel_l2_(got, want);
    std::printf("[train_ops] h3 rms + per-row modulation bwd rel-L2 %.2e\n", e);
    EXPECT_TRUE(e < 2e-3);
  }
  // The plain gate: dsub = gate[idx] * dh.
  SharedBuffer bdh = g.f32(dy);
  SharedBuffer bds = g.zeros((std::size_t)rows * H * 2);
  ASSERT_TRUE(g.run([&](metal_compute::ComputeEncoder& enc) {
    g.ops.gated_idx_bwd(enc, bdh, 0, bm, 0, bi, 0, bds, 0, rows, H, stride,
                        5 * H, true);
  }));
  {
    const std::vector<float> got = read_bf_(bds, (std::size_t)rows * H);
    std::vector<double> want((std::size_t)rows * H);
    for (int r = 0; r < rows; ++r) {
      for (int i = 0; i < H; ++i) {
        want[(std::size_t)r * H + i] =
            (double)mod[(std::size_t)idx[r] * stride + 5 * H + i] *
            dy[(std::size_t)r * H + i];
      }
    }
    const double e = rel_l2_(got, want);
    std::printf("[train_ops] h3 gated bwd rel-L2 %.2e\n", e);
    EXPECT_TRUE(e < 5e-3);
  }
  // SwiGLU off [gate | up].
  {
    const int W = 40;
    const std::vector<float> gu = randn_((std::size_t)rows * 2 * W, 1.0f, 21);
    const std::vector<float> ds = randn_((std::size_t)rows * W, 1.0f, 22);
    SharedBuffer bgu = g.bf(gu), bdsw = g.bf(ds);
    SharedBuffer bdgu = g.zeros((std::size_t)rows * 2 * W * 2);
    ASSERT_TRUE(g.run([&](metal_compute::ComputeEncoder& enc) {
      g.ops.swiglu_gf_bwd(enc, bgu, bdsw, bdgu, rows, W);
    }));
    const std::vector<float> got = read_bf_(bdgu, (std::size_t)rows * 2 * W);
    std::vector<double> want((std::size_t)rows * 2 * W);
    for (int r = 0; r < rows; ++r) {
      for (int d = 0; d < W; ++d) {
        auto s = [](double gg, double uu) {
          return uu * gg / (1.0 + std::exp(-gg));
        };
        const double gg = gu[(std::size_t)r * 2 * W + d];
        const double uu = gu[(std::size_t)r * 2 * W + W + d];
        const double dd = ds[(std::size_t)r * W + d];
        const double h = 1e-5;
        want[(std::size_t)r * 2 * W + d] =
            dd * (s(gg + h, uu) - s(gg - h, uu)) / (2 * h);
        want[(std::size_t)r * 2 * W + W + d] =
            dd * (s(gg, uu + h) - s(gg, uu - h)) / (2 * h);
      }
    }
    const double e = rel_l2_(got, want);
    std::printf("[train_ops] h3 swiglu [gate | up] bwd rel-L2 %.2e\n", e);
    EXPECT_TRUE(e < 5e-3);
  }
  // Partial rotate-half RoPE, back into a strided per-head fused layout,
  // then the per-head RMSNorm in place.
  {
    const int T = 3, NH = 2, D = 128, rot = 96, hr = rot / 2;
    const int ld = 3 * NH * D, hs = 3 * D, off = D;   // k of per-head qkv
    const std::vector<float> in = randn_((std::size_t)NH * T * D, 1.0f, 31);
    std::vector<float> cs((std::size_t)T * hr), sn((std::size_t)T * hr);
    for (int t = 0; t < T; ++t) {
      for (int i = 0; i < hr; ++i) {
        const float a = 0.37f * (float)(t + 1) * (float)(i + 1);
        cs[(std::size_t)t * hr + i] = std::cos(a);
        sn[(std::size_t)t * hr + i] = std::sin(a);
      }
    }
    SharedBuffer bin = g.f32(in), bc = g.f32(cs), bs = g.f32(sn);
    SharedBuffer bq = g.zeros((std::size_t)T * ld * 2);
    ASSERT_TRUE(g.run([&](metal_compute::ComputeEncoder& enc) {
      g.ops.rope_half_bwd(enc, bin, true, bq, 0, bc, bs, NH, T, D, rot, ld,
                          off, hs);
    }));
    // Want: the transpose of the forward rotation y = R x (per t, h).
    std::vector<double> want((std::size_t)T * ld, 0.0);
    for (int t = 0; t < T; ++t) {
      for (int h = 0; h < NH; ++h) {
        const float* y = in.data() + ((std::size_t)h * T + t) * D;
        double* o = want.data() + (std::size_t)t * ld + off + h * hs;
        for (int d = 0; d < D; ++d) { o[d] = y[d]; }
        for (int i = 0; i < hr; ++i) {
          const double c = cs[(std::size_t)t * hr + i];
          const double s = sn[(std::size_t)t * hr + i];
          // forward: y1 = x1 c - x2 s, y2 = x2 c + x1 s
          o[i] = y[i] * c + y[i + hr] * s;
          o[i + hr] = -y[i] * s + y[i + hr] * c;
        }
      }
    }
    std::vector<float> got = read_bf_(bq, (std::size_t)T * ld);
    std::vector<double> wk;
    std::vector<float> gk;
    for (int t = 0; t < T; ++t) {
      for (int h = 0; h < NH; ++h) {
        for (int d = 0; d < D; ++d) {
          const std::size_t p = (std::size_t)t * ld + off + h * hs + d;
          wk.push_back(want[p]);
          gk.push_back(got[p]);
        }
      }
    }
    const double e = rel_l2_(gk, wk);
    std::printf("[train_ops] h3 partial rope bwd rel-L2 %.2e\n", e);
    EXPECT_TRUE(e < 5e-3);
    // The per-head norm's backward over the same place, in place.
    const std::vector<float> xq = randn_((std::size_t)T * ld, 1.0f, 41);
    const std::vector<float> wq = randn_((std::size_t)D, 0.5f, 42);
    SharedBuffer bxq = g.bf(xq), bwq = g.bf(wq);
    const std::vector<float> dq0 = read_bf_(bq, (std::size_t)T * ld);
    ASSERT_TRUE(g.run([&](metal_compute::ComputeEncoder& enc) {
      g.ops.rms_heads_bwd(enc, bxq, bq, bwq, T, NH, D, ld, off, hs, eps);
    }));
    const std::vector<float> dq1 = read_bf_(bq, (std::size_t)T * ld);
    std::vector<double> wn;
    std::vector<float> gn;
    for (int t = 0; t < T; ++t) {
      for (int h = 0; h < NH; ++h) {
        const std::size_t b = (std::size_t)t * ld + off + h * hs;
        auto f = [&](const std::vector<double>& xr) {
          double ss = 0.0;
          for (double v : xr) { ss += v * v; }
          const double inv = 1.0 / std::sqrt(ss / D + eps);
          double acc = 0.0;
          for (int i = 0; i < D; ++i) {
            acc += dq0[b + i] * xr[i] * inv * wq[i];
          }
          return acc;
        };
        std::vector<double> xr(xq.begin() + b, xq.begin() + b + D);
        for (int i = 0; i < D; ++i) {
          const double hh = 1e-4, o = xr[i];
          xr[i] = o + hh; const double p = f(xr);
          xr[i] = o - hh; const double m = f(xr);
          xr[i] = o;
          wn.push_back((p - m) / (2.0 * hh));
          gn.push_back(dq1[b + i]);
        }
      }
    }
    const double en = rel_l2_(gn, wn);
    std::printf("[train_ops] h3 per-head rms bwd (in place) rel-L2 %.2e\n",
                en);
    EXPECT_TRUE(en < 5e-3);
  }
  // A banded B as block-diagonal, and a gradient back on its band.
  {
    const int n = 12, r = 2, parts = 3, group = 2;
    const std::vector<float> b = randn_((std::size_t)n * r, 1.0f, 51);
    SharedBuffer bb = g.bf(b);
    SharedBuffer bd = g.zeros((std::size_t)n * parts * r * 2);
    const std::vector<float> src =
        randn_((std::size_t)n * parts * r, 1.0f, 52);
    SharedBuffer bsrc = g.f32(src);
    SharedBuffer bdst = g.f32(std::vector<float>((std::size_t)n * r, 0.5f));
    ASSERT_TRUE(g.run([&](metal_compute::ComputeEncoder& enc) {
      g.ops.band_expand(enc, bb, 0, bd, n, r, parts, group);
      g.ops.band_extract_acc(enc, bsrc, bdst, 0, n, r, parts, group, 2.0f);
    }));
    const std::vector<float> e = read_bf_(bd, (std::size_t)n * parts * r);
    const std::vector<float> x = read_f32_(bdst, (std::size_t)n * r);
    int bad = 0;
    for (int j = 0; j < n; ++j) {
      const int p = (j / group) % parts;
      for (int c = 0; c < parts * r; ++c) {
        const float want = c / r == p ? b[(std::size_t)j * r + c % r] : 0.0f;
        bad += e[(std::size_t)j * parts * r + c] != want;
      }
      for (int c = 0; c < r; ++c) {
        const float want =
            0.5f + 2.0f * src[(std::size_t)j * parts * r + p * r + c];
        bad += std::fabs(x[(std::size_t)j * r + c] - want) > 1e-6f;
      }
    }
    std::printf("[train_ops] banded expand / extract: %d wrong\n", bad);
    EXPECT_TRUE(bad == 0);
  }
}

// 8-BIT AdamW and LoRA+, through the Optimizer over a real LoraParams set.
// A quadratic per parameter (loss = sum (w - w*)^2 / 2, so the gradient
// is w - w*, a target spread over five decades so the second moments of
// a block span several of the map's decades) run 300 steps three ways:
// AdamW in f32, AdamW with 8-bit moments, and the 8-bit run saved at 150
// and resumed into a fresh optimizer. The 8-bit trajectory stays close to
// f32's and converges as far; the resumed run is the uninterrupted one
// to the bit. The dynamic map is 256 ascending codes a sign kind, 0 and 1
// among them. LoRA+ on SGD without momentum: B's step is exactly the
// ratio times A's for the same gradient.
TEST(train_ops, adamw8bit_tracks_adamw_and_loraplus_scales_b)
{
  using namespace genai::train;
  Gpu g;
  if (!g.ok) { return; }
  const std::vector<float> codes = TrainOps::adamw8_codes();
  ASSERT_TRUE(codes.size() == 512);
  bool sorted = true, has01 = true;
  for (int h = 0; h < 2; ++h) {
    const float* c = codes.data() + h * 256;
    for (int i = 1; i < 256; ++i) { sorted = sorted && c[i] >= c[i - 1]; }
    has01 = has01 && std::count(c, c + 256, 0.0f) == 1 &&
            std::count(c, c + 256, 1.0f) == 1;
  }
  EXPECT_TRUE(sorted && has01 && codes[256] == 0.0f && codes[0] < -0.99f);
  const std::vector<ModuleShape> mods = {{"m.0", 96, 160, {}},
                                         {"m.1", 200, 64, {}}};
  const std::size_t S = 300;
  auto run = [&](const std::string& opt, int stop_at,
                 const std::string* load, const std::string* save,
                 std::vector<float>* out, double* loss_end) {
    LoraParams p;
    std::string err;
    if (!p.create(g.mc, mods, 8, 8.0f, &err)) { return false; }
    p.init_random(3);
    // B starts away from zero too, so both arenas move.
    auto* b = static_cast<float*>(p.b_master().contents());
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (std::size_t i = 0; i < p.b_elems(); ++i) { b[i] = 0.05f * nd(rng); }
    p.refresh_shadows_cpu();
    const std::size_t na = p.a_elems(), nb = p.b_elems();
    std::vector<float> ta(na), tb(nb);
    for (std::size_t i = 0; i < na; ++i) {
      ta[i] = 0.1f * nd(rng) * std::pow(10.0f, -(float)(i % 5));
    }
    for (std::size_t i = 0; i < nb; ++i) {
      tb[i] = 0.1f * nd(rng) * std::pow(10.0f, -(float)(i % 5));
    }
    TrainOps ops;
    if (!ops.init(g.mc, true, &err)) { return false; }
    OptimizerSpec spec;
    spec.optimizer = opt;
    spec.lr = 5e-3;
    spec.warmup_steps = 0;
    spec.grad_clip = 0.0;
    spec.weight_decay = 0.0;
    Optimizer o;
    if (!o.init(&ops, &p, spec, (int)S, &err)) { return false; }
    if (load != nullptr && !o.load_state(*load, &err)) {
      std::printf("  load: %s\n", err.c_str());
      return false;
    }
    for (int t = o.steps_taken(); t < stop_at; ++t) {
      const auto* a = static_cast<const float*>(p.a_master().contents());
      const auto* bb = static_cast<const float*>(p.b_master().contents());
      auto* ga = static_cast<float*>(p.a_grad().contents());
      auto* gb = static_cast<float*>(p.b_grad().contents());
      for (std::size_t i = 0; i < na; ++i) { ga[i] = a[i] - ta[i]; }
      for (std::size_t i = 0; i < nb; ++i) { gb[i] = bb[i] - tb[i]; }
      double gn = 0.0, lr = 0.0;
      if (!o.step(1, &gn, &lr, &err)) { return false; }
    }
    if (save != nullptr && !o.save_state(*save, &err)) { return false; }
    const auto* a = static_cast<const float*>(p.a_master().contents());
    const auto* bb = static_cast<const float*>(p.b_master().contents());
    out->assign(a, a + na);
    out->insert(out->end(), bb, bb + nb);
    double l = 0.0;
    for (std::size_t i = 0; i < na; ++i) {
      l += 0.5 * std::pow(a[i] - ta[i], 2);
    }
    for (std::size_t i = 0; i < nb; ++i) {
      l += 0.5 * std::pow(bb[i] - tb[i], 2);
    }
    *loss_end = l;
    return true;
  };
  std::vector<float> w32, w8, w8r, wh;
  double l32 = 0.0, l8 = 0.0, l8r = 0.0, lh = 0.0, l0 = 0.0;
  std::vector<float> w0;
  ASSERT_TRUE(run("adamw", 0, nullptr, nullptr, &w0, &l0));
  ASSERT_TRUE(run("adamw", (int)S, nullptr, nullptr, &w32, &l32));
  ASSERT_TRUE(run("adamw8bit", (int)S, nullptr, nullptr, &w8, &l8));
  const std::string st = (fs_tmp_() + "/vpipe-opt8-state.bin");
  ASSERT_TRUE(run("adamw8bit", 150, nullptr, &st, &wh, &lh));
  ASSERT_TRUE(run("adamw8bit", (int)S, &st, nullptr, &w8r, &l8r));
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < w32.size(); ++i) {
    num += std::pow((double)w8[i] - w32[i], 2);
    den += std::pow((double)w32[i] - w0[i], 2);
  }
  const double rel = std::sqrt(num / den);
  std::size_t resumed_off = 0;
  for (std::size_t i = 0; i < w8.size(); ++i) {
    resumed_off += w8[i] != w8r[i];
  }
  // Mid-flight, where the two can differ: 30 steps.
  std::vector<float> m32, m8;
  double lm32 = 0.0, lm8 = 0.0;
  ASSERT_TRUE(run("adamw", 30, nullptr, nullptr, &m32, &lm32));
  ASSERT_TRUE(run("adamw8bit", 30, nullptr, nullptr, &m8, &lm8));
  double mn = 0.0, md = 0.0;
  std::size_t mdiff = 0;
  for (std::size_t i = 0; i < m32.size(); ++i) {
    mn += std::pow((double)m8[i] - m32[i], 2);
    md += std::pow((double)m32[i] - w0[i], 2);
    mdiff += m8[i] != m32[i];
  }
  const double mrel = std::sqrt(mn / md);
  std::printf("[train_ops] loss %.6f -> adamw %.3e, adamw8bit %.3e at 300 "
              "(%.4f apart, of the distance travelled); at 30 %.3e / %.3e, "
              "%.4f apart, %zu of %zu weights differ; resumed at 150: %zu "
              "differ\n", l0, l32, l8, rel, lm32, lm8, mrel, mdiff,
              m32.size(), resumed_off);
  EXPECT_TRUE(l8 < 1e-2 * l0 && l8 < 2.0 * l32 + 1e-6 * l0);
  EXPECT_TRUE(rel < 0.05);
  EXPECT_TRUE(mrel < 0.05 && mdiff > m32.size() / 2);
  EXPECT_TRUE(resumed_off == 0);
  std::remove(st.c_str());

  // LoRA+.
  {
    LoraParams p;
    std::string err;
    ASSERT_TRUE(p.create(g.mc, {{"m.0", 16, 32, {}}}, 4, 4.0f, &err));
    p.init_random(1);
    TrainOps ops;
    ASSERT_TRUE(ops.init(g.mc, true, &err));
    OptimizerSpec spec;
    spec.optimizer = "sgd";
    spec.lr = 1e-2;
    spec.beta1 = 0.0;
    spec.warmup_steps = 0;
    spec.grad_clip = 0.0;
    spec.weight_decay = 0.0;
    spec.lora_plus_ratio = 16.0;
    Optimizer o;
    ASSERT_TRUE(o.init(&ops, &p, spec, 10, &err));
    const std::size_t na = p.a_elems(), nb = p.b_elems();
    std::vector<float> a0(static_cast<const float*>(p.a_master().contents()),
                          static_cast<const float*>(p.a_master().contents()) +
                              na);
    std::vector<float> b0(static_cast<const float*>(p.b_master().contents()),
                          static_cast<const float*>(p.b_master().contents()) +
                              nb);
    auto* ga = static_cast<float*>(p.a_grad().contents());
    auto* gb = static_cast<float*>(p.b_grad().contents());
    for (std::size_t i = 0; i < na; ++i) { ga[i] = 1.0f; }
    for (std::size_t i = 0; i < nb; ++i) { gb[i] = 1.0f; }
    double gn = 0.0, lr = 0.0;
    ASSERT_TRUE(o.step(1, &gn, &lr, &err));
    const auto* a1 = static_cast<const float*>(p.a_master().contents());
    const auto* b1 = static_cast<const float*>(p.b_master().contents());
    const double da = a0[0] - a1[0], db = b0[0] - b1[0];
    std::printf("[train_ops] LoRA+ 16: A stepped %.6g, B %.6g (%.4fx)\n", da,
                db, db / da);
    EXPECT_TRUE(std::fabs(db / da - 16.0) < 1e-4);
  }
}

// ADAFACTOR, through the Optimizer, against a restatement in double of
// transformers' Adafactor (relative_step / scale_parameter / first moment
// off; decay -0.8, eps1 1e-30, clip 1) on each factor as a matrix -- A
// [rank, k] per module, B [n, rank] per module, and a FUSED group whose
// B rows interleave (two modules' b_rows alternating) as one [2n, rank]
// matrix, as H3's q|k|v rows are held. Four steps of random gradients
// with weight decay: every weight within 2e-5 of the restatement; the
// state saved at 2 and resumed lands on the uninterrupted run to the
// bit; LoRA+ scales B's step.
TEST(train_ops, adafactor_matches_its_restatement)
{
  using namespace genai::train;
  Gpu g;
  if (!g.ok) { return; }
  const int R = 4;
  std::vector<ModuleShape> mods = {{"a.0", 24, 40, {}},
                                   {"q", 6, 16, {0, 2, 4, 6, 8, 10}},
                                   {"k", 6, 16, {-5, -3, -1, 1, 3, 5}}};
  const int steps = 4;
  auto grads = [&](std::size_t n, int step) {
    std::mt19937 rng(100 + step);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> v(n);
    for (float& x : v) { x = nd(rng) * (0.1f + (float)(x != 0)); }
    return v;
  };
  struct Run {
    std::vector<float> a, b;
  };
  auto gpu_run = [&](int upto, const std::string* load,
                     const std::string* save, double ratio, Run* out) {
    LoraParams p;
    std::string err;
    if (!p.create(g.mc, mods, R, 4.0f, &err)) {
      std::printf("  %s\n", err.c_str());
      return false;
    }
    p.init_random(9);
    auto* b = static_cast<float*>(p.b_master().contents());
    for (std::size_t i = 0; i < p.b_elems(); ++i) {
      b[i] = 0.01f * (float)((i * 37) % 23) - 0.1f;
    }
    p.refresh_shadows_cpu();
    TrainOps ops;
    if (!ops.init(g.mc, true, &err)) { return false; }
    OptimizerSpec spec;
    spec.optimizer = "adafactor";
    spec.lr = 1e-2;
    spec.warmup_steps = 0;
    spec.grad_clip = 0.0;
    spec.weight_decay = 0.1;
    spec.lora_plus_ratio = ratio;
    Optimizer o;
    if (!o.init(&ops, &p, spec, steps, &err)) {
      std::printf("  %s\n", err.c_str());
      return false;
    }
    if (load != nullptr && !o.load_state(*load, &err)) {
      std::printf("  %s\n", err.c_str());
      return false;
    }
    for (int t = o.steps_taken(); t < upto; ++t) {
      const auto ga = grads(p.a_elems(), 2 * t);
      const auto gb = grads(p.b_elems(), 2 * t + 1);
      std::memcpy(p.a_grad().contents(), ga.data(), ga.size() * 4);
      std::memcpy(p.b_grad().contents(), gb.data(), gb.size() * 4);
      double gn = 0.0, lr = 0.0;
      if (!o.step(1, &gn, &lr, &err)) {
        std::printf("  %s\n", err.c_str());
        return false;
      }
    }
    if (save != nullptr && !o.save_state(*save, &err)) { return false; }
    const auto* a = static_cast<const float*>(p.a_master().contents());
    const auto* bb = static_cast<const float*>(p.b_master().contents());
    out->a.assign(a, a + p.a_elems());
    out->b.assign(bb, bb + p.b_elems());
    return true;
  };
  Run gpu, start;
  ASSERT_TRUE(gpu_run(0, nullptr, nullptr, 1.0, &start));
  ASSERT_TRUE(gpu_run(steps, nullptr, nullptr, 1.0, &gpu));
  // The restatement. Matrices: A per module; B module 0 alone (rows
  // [0, 24)), then q and k's interleaved rows as one [12, R] matrix.
  struct Mat { bool isa; std::size_t off; int rows, cols; };
  std::vector<Mat> mats = {{true, 0, R, 40}, {true, (std::size_t)R * 40, R,
                                               16},
                           {true, (std::size_t)R * 56, R, 16},
                           {false, 0, 24, R}, {false, (std::size_t)24 * R, 12,
                                               R}};
  std::vector<double> pa(start.a.begin(), start.a.end()),
      pb(start.b.begin(), start.b.end());
  std::vector<std::vector<double>> rs(mats.size()), cs(mats.size());
  for (std::size_t j = 0; j < mats.size(); ++j) {
    rs[j].assign((std::size_t)mats[j].rows, 0.0);
    cs[j].assign((std::size_t)mats[j].cols, 0.0);
  }
  for (int t = 0; t < steps; ++t) {
    const auto ga = grads(pa.size(), 2 * t);
    const auto gb = grads(pb.size(), 2 * t + 1);
    const double b2t = 1.0 - std::pow((double)(t + 1), -0.8);
    for (std::size_t j = 0; j < mats.size(); ++j) {
      const Mat& m = mats[j];
      const auto& gg = m.isa ? ga : gb;
      std::vector<double>& pp = m.isa ? pa : pb;
      auto G = [&](int r, int c) {
        return (double)gg[m.off + (std::size_t)r * m.cols + c];
      };
      for (int r = 0; r < m.rows; ++r) {
        double s = 0.0;
        for (int c = 0; c < m.cols; ++c) { s += G(r, c) * G(r, c) + 1e-30; }
        rs[j][(std::size_t)r] = b2t * rs[j][(std::size_t)r] +
                                (1.0 - b2t) * s / m.cols;
      }
      for (int c = 0; c < m.cols; ++c) {
        double s = 0.0;
        for (int r = 0; r < m.rows; ++r) { s += G(r, c) * G(r, c) + 1e-30; }
        cs[j][(std::size_t)c] = b2t * cs[j][(std::size_t)c] +
                                (1.0 - b2t) * s / m.rows;
      }
      double rm = 0.0;
      for (double v : rs[j]) { rm += v; }
      rm /= m.rows;
      std::vector<double> u((std::size_t)m.rows * m.cols);
      double usq = 0.0;
      for (int r = 0; r < m.rows; ++r) {
        for (int c = 0; c < m.cols; ++c) {
          const double v = G(r, c) / std::sqrt(rs[j][(std::size_t)r] / rm) /
                           std::sqrt(cs[j][(std::size_t)c]);
          u[(std::size_t)r * m.cols + c] = v;
          usq += v * v;
        }
      }
      const double div = std::max(std::sqrt(usq / u.size()), 1.0);
      for (std::size_t e = 0; e < u.size(); ++e) {
        double& w = pp[m.off + e];
        w -= 1e-2 * 0.1 * w;
        w -= 1e-2 * u[e] / div;
      }
    }
  }
  double worst = 0.0, moved = 0.0;
  for (std::size_t i = 0; i < pa.size(); ++i) {
    worst = std::max(worst, std::fabs(pa[i] - gpu.a[i]));
    moved = std::max(moved, std::fabs(pa[i] - start.a[i]));
  }
  for (std::size_t i = 0; i < pb.size(); ++i) {
    worst = std::max(worst, std::fabs(pb[i] - gpu.b[i]));
    moved = std::max(moved, std::fabs(pb[i] - start.b[i]));
  }
  // Resumed at 2.
  const std::string st = fs_tmp_() + "/vpipe-adafactor-state.bin";
  Run half, resumed;
  ASSERT_TRUE(gpu_run(2, nullptr, &st, 1.0, &half));
  ASSERT_TRUE(gpu_run(steps, &st, nullptr, 1.0, &resumed));
  std::size_t off = 0;
  for (std::size_t i = 0; i < gpu.a.size(); ++i) {
    off += gpu.a[i] != resumed.a[i];
  }
  for (std::size_t i = 0; i < gpu.b.size(); ++i) {
    off += gpu.b[i] != resumed.b[i];
  }
  std::remove(st.c_str());
  // LoRA+ at 4: B moves further, A alike.
  Run lp;
  ASSERT_TRUE(gpu_run(1, nullptr, nullptr, 4.0, &lp));
  Run one;
  ASSERT_TRUE(gpu_run(1, nullptr, nullptr, 1.0, &one));
  double da = 0.0, db1 = 0.0, db4 = 0.0;
  for (std::size_t i = 0; i < one.a.size(); ++i) {
    da = std::max(da, std::fabs((double)one.a[i] - lp.a[i]));
  }
  for (std::size_t i = 0; i < one.b.size(); ++i) {
    db1 = std::max(db1, std::fabs((double)one.b[i] - start.b[i]));
    db4 = std::max(db4, std::fabs((double)lp.b[i] - start.b[i]));
  }
  std::printf("[train_ops] adafactor: %d steps moved weights up to %.4g, "
              "worst %.3g from the restatement; resumed at 2: %zu differ; "
              "LoRA+ 4: B's largest step %.4g against %.4g, A's %.3g "
              "apart\n", steps, moved, worst, off, db4, db1, da);
  EXPECT_TRUE(moved > 1e-3 && worst < 2e-5);
  EXPECT_TRUE(off == 0);
  EXPECT_TRUE(da == 0.0 && db4 > 3.0 * db1);
}

// THE RESUME FILE IS VERSIONED, and a version-1 file (written before the
// 8-bit and Adafactor states) still loads for every optimizer that has
// no such state: a v2 AdamW file with its trailing section cut and its
// version set back to 1 resumes the same weights; for adamw8bit it is
// refused, with a reason, rather than read as zeros.
TEST(train_ops, a_version_1_optimizer_state_still_loads)
{
  using namespace genai::train;
  Gpu g;
  if (!g.ok) { return; }
  const std::string dir = fs_tmp_();
  auto make = [&](const std::string& opt, LoraParams* p, TrainOps* ops,
                  Optimizer* o, std::string* err) {
    if (!p->create(g.mc, {{"m.0", 16, 32, {}}}, 4, 4.0f, err)) {
      return false;
    }
    p->init_random(2);
    if (!ops->init(g.mc, true, err)) { return false; }
    OptimizerSpec spec;
    spec.optimizer = opt;
    spec.lr = 1e-2;
    spec.warmup_steps = 0;
    return o->init(ops, p, spec, 4, err);
  };
  std::string err;
  LoraParams p;
  TrainOps ops;
  Optimizer o;
  ASSERT_TRUE(make("adamw", &p, &ops, &o, &err));
  auto* ga = static_cast<float*>(p.a_grad().contents());
  for (std::size_t i = 0; i < p.a_elems(); ++i) { ga[i] = 0.5f; }
  double gn = 0.0, lr = 0.0;
  ASSERT_TRUE(o.step(1, &gn, &lr, &err));
  const std::string v2 = dir + "/vpipe-opt-v2.bin";
  ASSERT_TRUE(o.save_state(v2, &err));
  // Back to version 1: the trailing section is 8 bytes a raw buffer, all
  // empty for AdamW -- 8 of them.
  std::vector<char> bytes;
  {
    std::ifstream in(v2, std::ios::binary);
    bytes.assign(std::istreambuf_iterator<char>(in),
                 std::istreambuf_iterator<char>());
  }
  ASSERT_TRUE(bytes.size() > 64);
  bytes.resize(bytes.size() - 8 * 8);
  const std::uint32_t one = 1;
  std::memcpy(bytes.data() + 4, &one, 4);
  const std::string v1 = dir + "/vpipe-opt-v1.bin";
  {
    std::ofstream out(v1, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), (std::streamsize)bytes.size());
  }
  LoraParams p2;
  TrainOps ops2;
  Optimizer o2;
  ASSERT_TRUE(make("adamw", &p2, &ops2, &o2, &err));
  const bool loaded = o2.load_state(v1, &err);
  std::printf("[train_ops] a version-1 AdamW state: %s%s\n",
              loaded ? "loaded" : "REFUSED: ", loaded ? "" : err.c_str());
  EXPECT_TRUE(loaded && o2.steps_taken() == 1);
  EXPECT_TRUE(loaded && std::memcmp(p.a_master().contents(),
                                    p2.a_master().contents(),
                                    p.a_elems() * 4) == 0);
  LoraParams p3;
  TrainOps ops3;
  Optimizer o3;
  ASSERT_TRUE(make("adamw8bit", &p3, &ops3, &o3, &err));
  err.clear();
  EXPECT_TRUE(!o3.load_state(v1, &err) && !err.empty());
  std::printf("[train_ops] the same file for adamw8bit: %s\n", err.c_str());
  std::remove(v1.c_str());
  std::remove(v2.c_str());
}

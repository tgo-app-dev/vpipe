// Ternary Bonsai 27B (prism-ml) tests.
//
//   VPIPE_BONSAI1_TEST_MODEL_PATH  prism-ml/Ternary-Bonsai-27B-mlx-2bit
//
// bonsai_kernels.*: no checkpoint -- the 2-bit group-128 affine kernels
// the ternary packs run on (MLX affine 2-bit g128: Bonsai 1's MLX pack as
// stored, Bonsai 2's GGUF PQ2_0 / PTQ1_0 repacked to it) against an f64
// reference over random codes, every route a projection takes; and the
// Prism Hadamard rotation Bonsai 2's activations take, against f64.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/session.h"
#include "generative-models/chat-template.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/loaded-language-model.h"
#include "generative-models/model-loader.h"
#include "generative-models/qwen3/metal-qwen-model.h"
#include "generative-models/qwen3/metal-qwen-vision.h"
#include "generative-models/shared/gguf-file.h"
#include "generative-models/tokenizer.h"
#include "stages/model-detect.h"
#include "stages/model-memory.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <vector>

using namespace vpipe;
using metal_compute::ComputeEncoder;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

std::uint16_t
to_bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  const std::uint32_t r = ((u >> 16) & 1u) + 0x7FFFu;
  return (std::uint16_t)((u + r) >> 16);
}

double
from_bf16_(std::uint16_t h)
{
  const std::uint32_t u = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

std::uint16_t
to_f16_(double d)
{
  const _Float16 h = (_Float16)d;
  std::uint16_t u;
  std::memcpy(&u, &h, 2);
  return u;
}

double
from_f16_(std::uint16_t u)
{
  _Float16 h;
  std::memcpy(&h, &u, 2);
  return (double)h;
}

double
rel_l2_(const std::vector<double>& ref, const std::vector<double>& got)
{
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double d = got[i] - ref[i];
    num += d * d;
    den += ref[i] * ref[i];
  }
  return std::sqrt(num / std::max(den, 1e-30));
}

struct Rng {
  std::uint32_t s = 7u;
  std::uint32_t next()
  {
    s = s * 1664525u + 1013904223u;
    return s;
  }
  double unif() { return (next() >> 8) * (1.0 / 16777216.0); }
};

}  // namespace

// Every 2-bit g128 route a projection takes -- the decode GEMV (with the
// fused residual, and the fused SwiGLU over an interleaved gate|up), the
// MAXM=2 batched GEMVs (a realtime-vqa step's branches; 3 rows, so a
// partial tile), the steel GEMM and its SwiGLU (prefill), the dequant
// (the matrix-core prefill) -- against f64 over the same rounded codes,
// scales, biases and inputs, at f16 and bf16. Widths are the 27B's own
// (K 5120, 17408) and K 640, a partial last 512-block.
TEST(bonsai_kernels, w2g128_routes_match_f64)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  Rng rng;
  for (const bool bf16 : {false, true}) {
    const std::string sfx = bf16 ? "_bf16" : "";
    auto lv = mc->load_library("affine_qmv" + sfx);
    auto lq = mc->load_library("affine_qmm_steel" + sfx);
    auto ld = mc->load_library("affine_dequant" + sfx);
    auto f_qmv = lv.function("affine_qmv_w2g128");
    auto f_add = lv.function("affine_qmv_w2g128_add");
    auto f_sw = lv.function("affine_qmv_swiglu_w2g128");
    auto f_b = lv.function("affine_qmv_batch_w2g128");
    auto f_bsw = lv.function("affine_qmv_batch_swiglu_w2g128");
    auto f_qmm = lq.function("affine_qmm_steel_w2g128");
    auto f_qsw = lq.function("affine_qmm_swiglu_w2g128");
    auto f_dq = ld.function("affine_dequant_w2g128");
    ASSERT_TRUE(f_qmv.valid() && f_add.valid() && f_sw.valid() &&
                f_b.valid() && f_bsw.valid() && f_qmm.valid() &&
                f_qsw.valid() && f_dq.valid());
    if (!f_dq.valid() || !f_qsw.valid()) { return; }
    auto el = [&](double d) { return bf16 ? to_bf16_((float)d) : to_f16_(d); };
    auto un = [&](std::uint16_t u) {
      return bf16 ? from_bf16_(u) : from_f16_(u);
    };
    // 352 rows: a multiple of 32 but not 64, as an MLX pack's fused GDN
    // in_proj (q|k|v|z|a|b, 16480 rows) is.
    for (const auto [K, N] : {std::pair{5120, 256}, std::pair{17408, 128},
                              std::pair{640, 192}, std::pair{5120, 352}}) {
      constexpr int M = 37;      // the steel GEMM's rows: a partial tile
      constexpr int MB = 3;      // the batched GEMV's rows
      const int G = K / 128;
      std::vector<std::uint8_t> q((std::size_t)N * K);
      for (auto& v : q) { v = (std::uint8_t)(rng.next() >> 30); }
      std::vector<std::uint32_t> packed((std::size_t)N * K / 16, 0u);
      for (std::size_t i = 0; i < q.size(); ++i) {
        packed[i / 16] |= (std::uint32_t)q[i] << (2 * (i % 16));
      }
      std::vector<std::uint16_t> sc((std::size_t)N * G), bi(sc.size()),
          x((std::size_t)M * K), res((std::size_t)N);
      // The ternary form: scale 2s, bias -s.
      for (std::size_t i = 0; i < sc.size(); ++i) {
        const double s = 0.002 + 0.02 * rng.unif();
        sc[i] = el(2.0 * s);
        bi[i] = el(-s);
      }
      for (auto& v : x) { v = el(2.0 * rng.unif() - 1.0); }
      for (auto& v : res) { v = el(rng.unif() - 0.5); }
      // f64 references.
      auto wv = [&](int n, int k) {
        const std::size_t g = (std::size_t)n * G + (std::size_t)(k / 128);
        return un(sc[g]) * q[(std::size_t)n * K + k] + un(bi[g]);
      };
      std::vector<double> ref((std::size_t)M * N);
      for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
          double a = 0.0;
          for (int k = 0; k < K; ++k) {
            a += un(x[(std::size_t)m * K + k]) * wv(n, k);
          }
          ref[(std::size_t)m * N + n] = a;
        }
      }
      auto swi = [&](int m, int f) {   // rows 2f gate, 2f+1 up
        const double g = ref[(std::size_t)m * N + 2 * f];
        const double u = ref[(std::size_t)m * N + 2 * f + 1];
        return g / (1.0 + std::exp(-g)) * u;
      };
      auto up = [&](const void* p, std::size_t n) {
        SharedBuffer b = mc->make_shared_buffer(n);
        std::memcpy(b.contents(), p, n);
        return b;
      };
      SharedBuffer bw = up(packed.data(), packed.size() * 4);
      SharedBuffer bs = up(sc.data(), sc.size() * 2);
      SharedBuffer bb = up(bi.data(), bi.size() * 2);
      SharedBuffer bx = up(x.data(), x.size() * 2);
      SharedBuffer br = up(res.data(), res.size() * 2);
      SharedBuffer y_qmv = mc->make_shared_buffer((std::size_t)N * 2);
      SharedBuffer y_add = mc->make_shared_buffer((std::size_t)N * 2);
      SharedBuffer y_sw = mc->make_shared_buffer((std::size_t)N);
      SharedBuffer y_b = mc->make_shared_buffer((std::size_t)MB * N * 2);
      SharedBuffer y_bsw = mc->make_shared_buffer((std::size_t)MB * N);
      SharedBuffer y_qmm = mc->make_shared_buffer((std::size_t)M * N * 2);
      SharedBuffer y_qsw = mc->make_shared_buffer((std::size_t)M * N);
      SharedBuffer y_dq = mc->make_shared_buffer((std::size_t)N * K * 2);
      SharedBuffer y_rows = mc->make_shared_buffer((std::size_t)MB * N * 2);
      SharedBuffer y_swr = mc->make_shared_buffer((std::size_t)MB * N);
      auto st = mc->make_command_stream();
      {
        ComputeEncoder enc = st.begin_compute();
        auto common = [&](const SharedBuffer& y) {
          enc.set_buffer(0, bw); enc.set_buffer(1, bs); enc.set_buffer(2, bb);
          enc.set_buffer(3, bx); enc.set_buffer(4, y);
          enc.set_constant(5, K); enc.set_constant(6, N);
        };
        const unsigned gy = (unsigned)((N + 3) / 4);
        enc.set_function(f_qmv); common(y_qmv);
        enc.dispatch({32, gy, 1}, {32, 2, 1});
        enc.set_function(f_add); common(y_add); enc.set_buffer(7, br);
        enc.dispatch({32, gy, 1}, {32, 2, 1});
        enc.set_function(f_sw); common(y_sw);
        enc.dispatch({32, gy, 1}, {32, 2, 1});
        enc.set_function(f_b); common(y_b); enc.set_constant(7, MB);
        enc.dispatch({32, gy, (unsigned)((MB + 1) / 2)}, {32, 2, 1});
        enc.set_function(f_bsw); common(y_bsw); enc.set_constant(7, MB);
        enc.dispatch({32, gy, (unsigned)((MB + 1) / 2)}, {32, 2, 1});
        enc.set_function(f_qmm); common(y_qmm); enc.set_constant(7, M);
        enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                      (unsigned)(((M + 31) / 32) * 2), 2}, {32, 2, 2});
        enc.set_function(f_qsw); common(y_qsw); enc.set_constant(7, M);
        enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                      (unsigned)(((M + 31) / 32) * 2), 2}, {32, 2, 2});
        // The batched rows' twins: the one-row GEMVs, a row at a time.
        for (int r = 0; r < MB; ++r) {
          for (const bool sw : {false, true}) {
            enc.set_function(sw ? f_sw : f_qmv);
            enc.set_buffer(0, bw); enc.set_buffer(1, bs);
            enc.set_buffer(2, bb);
            enc.set_buffer(3, bx, (std::size_t)r * K * 2);
            enc.set_buffer(4, sw ? y_swr : y_rows,
                           (std::size_t)r * (sw ? N / 2 : N) * 2);
            enc.set_constant(5, K); enc.set_constant(6, N);
            enc.dispatch({32, gy, 1}, {32, 2, 1});
          }
        }
        enc.set_function(f_dq);
        enc.set_buffer(0, bw); enc.set_buffer(1, bs); enc.set_buffer(2, bb);
        enc.set_buffer(3, y_dq);
        enc.set_constant(4, K); enc.set_constant(5, N);
        enc.dispatch({(unsigned)(K / 16), (unsigned)N, 1}, {64, 1, 1});
      }
      ASSERT_TRUE(st.commit().wait_ok());
      auto got = [&](const SharedBuffer& b, std::size_t n) {
        std::vector<double> v(n);
        const auto* p = static_cast<const std::uint16_t*>(b.contents());
        for (std::size_t i = 0; i < n; ++i) { v[i] = un(p[i]); }
        return v;
      };
      auto rows = [&](int m0, int nm) {
        return std::vector<double>(ref.begin() + (std::ptrdiff_t)m0 * N,
                                   ref.begin() + (std::ptrdiff_t)(m0 + nm) *
                                                     N);
      };
      std::vector<double> r_add = rows(0, 1), r_sw, r_bsw, r_qsw;
      for (int n = 0; n < N; ++n) {
        r_add[(std::size_t)n] += un(res[(std::size_t)n]);
      }
      for (int f = 0; f < N / 2; ++f) { r_sw.push_back(swi(0, f)); }
      for (int m = 0; m < MB; ++m) {
        for (int f = 0; f < N / 2; ++f) { r_bsw.push_back(swi(m, f)); }
      }
      for (int m = 0; m < M; ++m) {
        for (int f = 0; f < N / 2; ++f) { r_qsw.push_back(swi(m, f)); }
      }
      struct Arm { const char* name; double d; };
      const Arm arms[] = {
          {"qmv", rel_l2_(rows(0, 1), got(y_qmv, (std::size_t)N))},
          {"qmv+add", rel_l2_(r_add, got(y_add, (std::size_t)N))},
          {"qmv swiglu", rel_l2_(r_sw, got(y_sw, (std::size_t)N / 2))},
          {"batch", rel_l2_(rows(0, MB), got(y_b, (std::size_t)MB * N))},
          {"batch swiglu",
           rel_l2_(r_bsw, got(y_bsw, (std::size_t)MB * N / 2))},
          {"steel", rel_l2_(ref, got(y_qmm, (std::size_t)M * N))},
          {"steel swiglu",
           rel_l2_(r_qsw, got(y_qsw, (std::size_t)M * N / 2))}};
      for (const Arm& a : arms) {
        std::printf("[bonsai_kernels] %s K %5d N %3d %-12s %.3e from f64\n",
                    bf16 ? "bf16" : "f16 ", K, N, a.name, a.d);
        // An output's own rounding: f16 ~3e-4, bf16 ~2.5e-3.
        EXPECT_TRUE(a.d < (bf16 ? 6e-3 : 1e-3));
      }
      // A batched row is the one-row GEMV's, bit for bit (a realtime-vqa
      // branch decodes as a serial context does).
      EXPECT_TRUE(std::memcmp(y_b.contents(), y_rows.contents(),
                              (std::size_t)MB * N * 2) == 0);
      EXPECT_TRUE(std::memcmp(y_bsw.contents(), y_swr.contents(),
                              (std::size_t)MB * N) == 0);
      // The dequant is exact: s * q + b in f32, rounded once.
      const auto* dq = static_cast<const std::uint16_t*>(y_dq.contents());
      int bad = 0;
      for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
          const std::size_t g = (std::size_t)n * G + (std::size_t)(k / 128);
          const float w = (float)un(sc[g]) *
                              (float)q[(std::size_t)n * K + k] +
                          (float)un(bi[g]);
          bad += dq[(std::size_t)n * K + k] != el(w);
        }
      }
      EXPECT_TRUE(bad == 0);
    }
  }
}

// The Prism rotation: rows of [rows, K] f16 / bf16 activations through
// prism_hadamard_rows_1024 -- forward H(s * x) and inverse s * (H x), per
// 1024-block, H the normalized Sylvester Walsh-Hadamard -- against f64;
// forward then inverse is the identity to the element type's rounding, and
// in place equals out of place.
TEST(bonsai_kernels, hadamard_rows_match_f64)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  Rng rng;
  for (const bool bf16 : {false, true}) {
    auto lib = mc->load_library(bf16 ? "llm_elementwise_bf16"
                                     : "llm_elementwise");
    auto fn = lib.function("prism_hadamard_rows_1024");
    ASSERT_TRUE(fn.valid());
    if (!fn.valid()) { return; }
    auto el = [&](double d) { return bf16 ? to_bf16_((float)d) : to_f16_(d); };
    auto un = [&](std::uint16_t u) {
      return bf16 ? from_bf16_(u) : from_f16_(u);
    };
    for (const int K : {5120, 6144, 17408}) {
      constexpr int R = 3;
      std::vector<std::uint16_t> x((std::size_t)R * K);
      std::vector<float> s((std::size_t)K);
      for (auto& v : x) { v = el(4.0 * rng.unif() - 2.0); }
      for (auto& v : s) { v = (rng.next() >> 31) ? 1.0f : -1.0f; }
      // f64: forward H(s x) per block.
      auto hadamard = [](const double* in, double* out) {
        for (int r = 0; r < 1024; ++r) {
          double a = 0.0;
          for (int c = 0; c < 1024; ++c) {
            a += (__builtin_popcount((unsigned)(r & c)) & 1 ? -in[c]
                                                            : in[c]);
          }
          out[r] = a / 32.0;
        }
      };
      std::vector<double> fwd((std::size_t)R * K), inv(fwd.size());
      for (int r = 0; r < R; ++r) {
        for (int b = 0; b < K / 1024; ++b) {
          double in[1024], out[1024];
          for (int j = 0; j < 1024; ++j) {
            const std::size_t i = (std::size_t)r * K + b * 1024 + j;
            in[j] = s[(std::size_t)(b * 1024 + j)] * un(x[i]);
          }
          hadamard(in, out);
          for (int j = 0; j < 1024; ++j) {
            fwd[(std::size_t)r * K + b * 1024 + j] = out[j];
          }
          for (int j = 0; j < 1024; ++j) {
            in[j] = un(x[(std::size_t)r * K + b * 1024 + j]);
          }
          hadamard(in, out);
          for (int j = 0; j < 1024; ++j) {
            inv[(std::size_t)r * K + b * 1024 + j] =
                s[(std::size_t)(b * 1024 + j)] * out[j];
          }
        }
      }
      SharedBuffer bx = mc->make_shared_buffer(x.size() * 2);
      std::memcpy(bx.contents(), x.data(), x.size() * 2);
      SharedBuffer bs = mc->make_shared_buffer(s.size() * 4);
      std::memcpy(bs.contents(), s.data(), s.size() * 4);
      SharedBuffer yf = mc->make_shared_buffer(x.size() * 2);
      SharedBuffer yi = mc->make_shared_buffer(x.size() * 2);
      SharedBuffer yb = mc->make_shared_buffer(x.size() * 2);
      SharedBuffer ip = mc->make_shared_buffer(x.size() * 2);
      std::memcpy(ip.contents(), x.data(), x.size() * 2);
      auto st = mc->make_command_stream();
      {
        ComputeEncoder enc = st.begin_compute();
        auto run = [&](const SharedBuffer& in, const SharedBuffer& out,
                       int inverse) {
          enc.set_function(fn);
          enc.set_buffer(0, in); enc.set_buffer(1, out);
          enc.set_buffer(2, bs);
          enc.set_constant(3, K); enc.set_constant(4, inverse);
          enc.set_constant(5, K); enc.set_constant(6, K);
          const int one = 1;
          enc.set_constant(7, one); enc.set_constant(8, one);
          enc.set_constant(9, one);
          enc.dispatch({(unsigned)(K / 1024 * 256), (unsigned)R, 1},
                       {256, 1, 1});
        };
        run(bx, yf, 0);
        run(bx, yi, 1);
        run(yf, yb, 1);    // forward then inverse
        run(ip, ip, 0);    // in place
      }
      ASSERT_TRUE(st.commit().wait_ok());
      auto got = [&](const SharedBuffer& b) {
        std::vector<double> v(x.size());
        const auto* p = static_cast<const std::uint16_t*>(b.contents());
        for (std::size_t i = 0; i < v.size(); ++i) { v[i] = un(p[i]); }
        return v;
      };
      std::vector<double> xs(x.size());
      for (std::size_t i = 0; i < xs.size(); ++i) { xs[i] = un(x[i]); }
      const double df = rel_l2_(fwd, got(yf));
      const double di = rel_l2_(inv, got(yi));
      const double db = rel_l2_(xs, got(yb));
      const bool same = std::memcmp(ip.contents(), yf.contents(),
                                    x.size() * 2) == 0;
      std::printf("[bonsai_kernels] %s K %5d hadamard: forward %.3e, "
                  "inverse %.3e from f64, round trip %.3e, in place %s\n",
                  bf16 ? "bf16" : "f16 ", K, df, di, db,
                  same ? "identical" : "DIFFERS");
      const double bar = bf16 ? 4e-3 : 5e-4;
      EXPECT_TRUE(df < bar && di < bar && db < 2 * bar && same);
      if (K != 6144) { continue; }
      // GROUPED: a GDN output in the file's tiled v-head order (head j = r
      // nk + k, hd 128, nk 16, rep 3) read as the grouped order the
      // rotated out_proj was folded in (head k rep + r), then rotated.
      const int hd = 128, nk = 16, rep = 3;
      std::vector<double> gref((std::size_t)R * K);
      for (int r = 0; r < R; ++r) {
        for (int b = 0; b < K / 1024; ++b) {
          double in[1024], out[1024];
          for (int j = 0; j < 1024; ++j) {
            const int p = b * 1024 + j;
            const int src = (p % hd) +
                            hd * ((p / (hd * rep)) + nk * ((p / hd) % rep));
            in[j] = s[(std::size_t)p] * un(x[(std::size_t)r * K + src]);
          }
          hadamard(in, out);
          for (int j = 0; j < 1024; ++j) {
            gref[(std::size_t)r * K + b * 1024 + j] = out[j];
          }
        }
      }
      SharedBuffer yg = mc->make_shared_buffer(x.size() * 2);
      auto sg = mc->make_command_stream();
      {
        ComputeEncoder enc = sg.begin_compute();
        enc.set_function(fn);
        enc.set_buffer(0, bx); enc.set_buffer(1, yg);
        enc.set_buffer(2, bs);
        const int zero = 0;
        enc.set_constant(3, K); enc.set_constant(4, zero);
        enc.set_constant(5, K); enc.set_constant(6, K);
        enc.set_constant(7, hd); enc.set_constant(8, nk);
        enc.set_constant(9, rep);
        enc.dispatch({(unsigned)(K / 1024 * 256), (unsigned)R, 1},
                     {256, 1, 1});
      }
      ASSERT_TRUE(sg.commit().wait_ok());
      const double dg = rel_l2_(gref, got(yg));
      // MUTATION guard: the unpermuted forward must be far from it.
      const double dn = rel_l2_(gref, got(yf));
      std::printf("[bonsai_kernels] %s K %5d grouped: %.3e from f64 "
                  "(unpermuted %.3f)\n", bf16 ? "bf16" : "f16 ", K, dg, dn);
      EXPECT_TRUE(dg < bar && dn > 0.5);
    }
  }
}

namespace {

// A Bonsai checkpoint and its tokenizer: Bonsai 1's MLX directory, or a
// Bonsai 2 GGUF FILE (PQ2_0 or PTQ1_0; the tokenizer from its metadata).
struct Bonsai {
  std::unique_ptr<genai::MetalQwenModel> model;
  std::unique_ptr<genai::Tokenizer> tok;
};

Bonsai
load_bonsai_(const std::string& path, Session& sess, MetalCompute* mc,
             int max_pages, bool bf16 = false)
{
  Bonsai b;
  genai::ModelLoader loader(&sess);
  auto cfg = loader.load_config(path);
  if (!cfg) { return b; }
  auto mcfg = genai::MetalQwenModel::config_from(*cfg);
  mcfg.use_bf16 = bf16;
  mcfg.page_tokens = 256;
  mcfg.max_pages = max_pages;
  std::printf("[bonsai] config: %d-bit group %d, %d layers\n",
              mcfg.quant_bits, mcfg.quant_group, mcfg.n_layers);
  const auto t0 = std::chrono::steady_clock::now();
  b.model = genai::MetalQwenModel::load(path, mc, mcfg);
  if (!b.model) { return b; }
  std::printf("[bonsai] loaded in %.1f s\n",
              std::chrono::duration<double>(
                  std::chrono::steady_clock::now() - t0).count());
  const bool gguf = path.size() > 5 &&
                    path.compare(path.size() - 5, 5, ".gguf") == 0;
  if (gguf) {
    if (auto g = genai::GgufFile::open(path)) {
      b.tok = genai::Tokenizer::from_gguf(*g, &sess);
    }
  } else {
    b.tok = genai::Tokenizer::from_huggingface_json(path + "/tokenizer.json",
                                                    &sess);
  }
  return b;
}

std::int32_t
argmax_(const std::vector<float>& lg)
{
  std::int32_t b = 0;
  for (std::size_t v = 1; v < lg.size(); ++v) {
    if (lg[v] > lg[(std::size_t)b]) { b = (std::int32_t)v; }
  }
  return b;
}

// minitest's checks live on the test case; a helper reports and returns.
#define BONSAI_CHECK(cond)                                                    \
  if (!(cond)) {                                                              \
    std::printf("[bonsai] check failed: %s (line %d)\n", #cond, __LINE__);   \
    return false;                                                             \
  }

// Load, prefill a plain prompt, decode greedily: the answer must be Paris.
bool
greedy_smoke_(const char* env)
{
  const char* path = std::getenv(env);
  if (path == nullptr || *path == 0) { return true; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return true; }
  Bonsai b = load_bonsai_(path, sess, mc, 16);
  BONSAI_CHECK(b.model != nullptr && b.tok != nullptr);
    auto* cm = b.model->context_manager();
  auto root = cm->acquire_root();
  auto logits = b.model->prefill(root, b.tok->encode(
                                           "The capital of France is"));
  BONSAI_CHECK(!logits.empty());
  std::int32_t next = argmax_(logits);
  std::vector<std::int32_t> out{next};
  const auto t1 = std::chrono::steady_clock::now();
  for (int i = 0; i < 23 && next >= 0; ++i) {
    next = b.model->decode_step_fast(root, next);
    out.push_back(next);
  }
  const double s = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - t1).count();
  const std::string text = b.tok->decode(out);
  std::printf("[bonsai] greedy: '%s' (%.1f tok/s)\n", text.c_str(),
              out.size() / s);
  const bool ok = text.find("Paris") != std::string::npos;
  cm->release(root);
  // A 512-token prompt's prefill (llama-bench's pp512), twice: the first
  // pays the kernels' first use.
  std::vector<std::int32_t> pp;
  while (pp.size() < 512) {
    for (const auto t : b.tok->encode(" The quick brown fox jumps over the "
                                      "lazy dog by the river bank.")) {
      if (pp.size() < 512) { pp.push_back(t); }
    }
  }
  for (int rep = 0; rep < 2; ++rep) {
    auto c = cm->acquire_root();
    const auto t2 = std::chrono::steady_clock::now();
    BONSAI_CHECK(!b.model->prefill(c, pp).empty());
    const double ps = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - t2).count();
    std::printf("[bonsai] prefill 512 tokens: %.2f s (%.0f tok/s)\n", ps,
                512 / ps);
    cm->release(c);
  }
  return ok;
}

// Batched decode (realtime-vqa's branches) against serial: branches off
// one prefix, each fed a different-length suffix, so they sit at
// different positions; 3 and 5 rows tile the 2-row kernels. Every
// batched row is the one-row GEMV's bit for bit, so TOKEN-EXACT.
bool
batched_matches_serial_(const char* env)
{
  const char* path = std::getenv(env);
  if (path == nullptr || *path == 0) { return true; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return true; }
  Bonsai b = load_bonsai_(path, sess, mc, 40);
  BONSAI_CHECK(b.model != nullptr && b.tok != nullptr);
    auto& model = b.model;
  auto* cm = model->context_manager();
  auto root = cm->acquire_root();
  BONSAI_CHECK(!model->prefill(root, b.tok->encode("The weather today is"))
                   .empty());
  constexpr int kSteps = 12;
  bool ok = true;
  for (const int N : {3, 5}) {
    auto batched = cm->branch(root, N);
    auto serial = cm->branch(root, N);
    BONSAI_CHECK((int)batched.size() == N && (int)serial.size() == N);
    std::vector<std::int32_t> first((std::size_t)N);
    for (int i = 0; i < N; ++i) {
      const std::vector<std::int32_t> suf((std::size_t)(i + 1),
                                          (std::int32_t)(100 + i));
      auto lb = model->prefill(batched[(std::size_t)i], suf);
      auto ls = model->prefill(serial[(std::size_t)i], suf);
      BONSAI_CHECK(!lb.empty() && !ls.empty());
      first[(std::size_t)i] = argmax_(lb);
    }
    std::vector<std::vector<std::int32_t>> ref((std::size_t)N);
    for (int i = 0; i < N; ++i) {
      std::int32_t cur = first[(std::size_t)i];
      for (int k = 0; k < kSteps; ++k) {
        cur = model->decode_step_fast(serial[(std::size_t)i], cur);
        if (cur < 0) { break; }
        ref[(std::size_t)i].push_back(cur);
      }
    }
    auto got = model->decode_batched_argmax(
        std::span<const genai::ContextId>(batched.data(), batched.size()),
        std::span<const std::int32_t>(first.data(), first.size()), kSteps);
    int same = 0, total = 0;
    for (int i = 0; i < N; ++i) {
      for (std::size_t k = 0; k < ref[(std::size_t)i].size() &&
                              k < got[(std::size_t)i].size(); ++k) {
        ++total;
        same += got[(std::size_t)i][k] == ref[(std::size_t)i][k];
      }
    }
    std::printf("[bonsai] batched N=%d: %d of %d tokens as serial\n", N,
                same, total);
    ok = ok && total == N * kSteps && same == total;
    for (auto c : batched) { cm->release(c); }
    for (auto c : serial) { cm->release(c); }
  }
  cm->release(root);
  return ok;
}

// Against the reference's fp32 forward (tools/dump_bonsai_golden.py: ids,
// prompt_len, logits [n, vocab] and the reference's own bf16 logits, the
// bar). The prompt prefills (its last row), the continuation is teacher-
// forced a token at a time through the decode step, and the whole
// sequence prefills once more (its last row). rel-L2 over every row
// compared, and the argmax picks.
std::vector<float>
read_f32_(const std::string& p)
{
  std::vector<float> v;
  std::FILE* f = std::fopen(p.c_str(), "rb");
  if (f == nullptr) { return v; }
  std::fseek(f, 0, SEEK_END);
  const long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  v.resize((std::size_t)n / 4);
  if (std::fread(v.data(), 4, v.size(), f) != v.size()) { v.clear(); }
  std::fclose(f);
  return v;
}

bool
golden_(const char* env, const char* genv, bool bf16)
{
  const char* path = std::getenv(env);
  const char* gold = std::getenv(genv);
  if (path == nullptr || *path == 0 || gold == nullptr || *gold == 0) {
    return true;
  }
  const std::string g(gold);
  const std::vector<float> idf = read_f32_(g + "/ids.i32");
  const std::vector<float> plf = read_f32_(g + "/prompt_len.i32");
  const std::vector<float> ref = read_f32_(g + "/logits.f32");
  const std::vector<float> rbf = read_f32_(g + "/logits_bf16.f32");
  BONSAI_CHECK(!idf.empty() && plf.size() == 1 && !ref.empty() &&
               ref.size() == rbf.size());
  std::vector<std::int32_t> ids(idf.size());
  std::memcpy(ids.data(), idf.data(), ids.size() * 4);
  int pl = 0;
  std::memcpy(&pl, plf.data(), 4);
  const int n = (int)ids.size();
  const std::size_t V = ref.size() / (std::size_t)n;
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return true; }
  Bonsai b = load_bonsai_(path, sess, mc, 16, bf16);
  BONSAI_CHECK(b.model != nullptr);
  auto* cm = b.model->context_manager();
  // Row r of ours, from the prefill / the decode step.
  std::vector<std::vector<float>> ours((std::size_t)n);
  auto root = cm->acquire_root();
  ours[(std::size_t)(pl - 1)] = b.model->prefill(
      root, std::vector<std::int32_t>(ids.begin(), ids.begin() + pl));
  for (int r = pl; r < n; ++r) {
    ours[(std::size_t)r] = b.model->forward(root, ids[(std::size_t)r]);
  }
  cm->release(root);
  auto whole = cm->acquire_root();
  const std::vector<float> last = b.model->prefill(whole, ids);
  cm->release(whole);
  {
    // Where the distance builds: the residual after layers 0, 3, 31, 63
    // (the reference's hidden_L) over the whole sequence.
    const std::vector<int> taps = {0, 3, 31, 63};
    auto tc = cm->acquire_root();
    SharedBuffer e = b.model->embed_text_buf(ids);
    SharedBuffer t = e.empty() ? SharedBuffer{}
                               : b.model->forward_embeddings_taps(tc, e, n,
                                                                  taps);
    cm->release(tc);
    const std::size_t H = 5120;
    for (std::size_t j = 0; j < taps.size() && !t.empty(); ++j) {
      const std::vector<float> hr = read_f32_(
          g + "/hidden_" + std::to_string(taps[j]) + ".f32");
      if (hr.size() != (std::size_t)n * H) { continue; }
      const auto* q = static_cast<const std::uint16_t*>(t.contents()) +
                      j * (std::size_t)n * H;
      double a = 0, c = 0;
      for (std::size_t i = 0; i < hr.size(); ++i) {
        const double o = bf16 ? from_bf16_(q[i]) : from_f16_(q[i]);
        a += (o - hr[i]) * (o - hr[i]);
        c += (double)hr[i] * hr[i];
      }
      std::printf("[bonsai]   hidden after layer %2d: %.4f from fp32\n",
                  taps[j], std::sqrt(a / c));
    }
  }
  BONSAI_CHECK(last.size() == V);
  double e2 = 0, r2 = 0, b2 = 0;
  int picks = 0, ref_picks = 0, rows = 0;
  auto amax = [&](const float* p) {
    std::size_t k = 0;
    for (std::size_t v = 1; v < V; ++v) { if (p[v] > p[k]) { k = v; } }
    return k;
  };
  for (int r = pl - 1; r < n; ++r) {
    const auto& o = ours[(std::size_t)r];
    BONSAI_CHECK(o.size() == V);
    const float* f = ref.data() + (std::size_t)r * V;
    const float* h = rbf.data() + (std::size_t)r * V;
    for (std::size_t v = 0; v < V; ++v) {
      e2 += (double)(o[v] - f[v]) * (o[v] - f[v]);
      b2 += (double)(h[v] - f[v]) * (h[v] - f[v]);
      r2 += (double)f[v] * f[v];
    }
    picks += amax(o.data()) == amax(f);
    ref_picks += amax(h) == amax(f);
    ++rows;
  }
  const float* fl = ref.data() + (std::size_t)(n - 1) * V;
  double l2 = 0, lr = 0;
  for (std::size_t v = 0; v < V; ++v) {
    l2 += (double)(last[v] - fl[v]) * (last[v] - fl[v]);
    lr += (double)fl[v] * fl[v];
  }
  const double d = std::sqrt(e2 / r2), dref = std::sqrt(b2 / r2);
  const double dl = std::sqrt(l2 / lr);
  std::printf("[bonsai] %s vs fp32, %d rows (prefill + decode): rel-L2 "
              "%.4f (the reference's bf16 %.4f), picks %d / %d (its bf16 "
              "%d); the whole prefill's last row %.4f\n",
              bf16 ? "bf16" : "f16 ", rows, d, dref, picks, rows, ref_picks,
              dl);
  // f16 is the compute type these packs want (their scales are F16):
  // MEASURED 0.0006 / 0.0007 (Bonsai 1 / 2) against the reference bf16's
  // 0.0047 / 0.0055, so its bar is half the latter -- MLX's form of the
  // GDN's q / k eps (_gdn_qk_eps) read 0.0071 on Bonsai 1, and fails it.
  // bf16 is the reference's own order: 0.0052 / 0.0114 (Bonsai 2's
  // keys of near-zero norm normalized from bf16 rounding; MLX's eps form
  // reads 0.0060 there and 0.0088 on Bonsai 1), so 2.5x.
  const double bar = (bf16 ? 2.5 : 0.5) * dref;
  return d < bar && dl < bar && picks >= ref_picks - 1;
}

// Through the model manager, as text-chat and realtime-vqa load it: the
// checkpoint's own tokenizer and chat template (a GGUF's from its
// metadata), at f16.
bool
chat_(const char* env, bool no_think = false)
{
  const char* path = std::getenv(env);
  if (path == nullptr || *path == 0) { return true; }
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (mgr == nullptr) { return true; }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "f16";
  spec.page_tokens = 256;
  spec.max_pages = 8;
  auto lm = mgr->load(spec);
  BONSAI_CHECK(lm != nullptr && lm->valid());
  // Thinking off: the template realtime-vqa builds (disable_thinking,
  // its default) -- an empty think block, then the answer.
  std::unique_ptr<genai::ChatTemplate> own;
  if (no_think) {
    own = genai::make_chat_template(lm->config(), lm->tokenizer(), true);
  }
  const genai::ChatTemplate* tpl = no_think ? own.get()
                                            : lm->chat_template();
  BONSAI_CHECK(tpl != nullptr);
  std::vector<std::int32_t> ids;
  tpl->render_user_turn("What is the capital of France? Reply with the "
                        "city name only.", /*is_first_turn=*/true, &ids);
  BONSAI_CHECK(!ids.empty());
  auto ctx = lm->make_context();
  BONSAI_CHECK(ctx.valid());
  std::vector<std::int32_t> gen;
  const std::int32_t first = lm->prefill(ctx, ids);
  BONSAI_CHECK(first >= 0);
  gen.push_back(first);
  for (int i = 0; i < 400 && !tpl->is_stop_token(gen.back()); ++i) {
    const std::int32_t t = lm->next_token(ctx);
    if (t < 0) { break; }
    gen.push_back(t);
  }
  const std::string text = lm->tokenizer().decode(
      std::span<const std::int32_t>(gen.data(), gen.size()));
  std::printf("[bonsai] chat%s (%zu tokens): '%s'\n",
              no_think ? ", thinking off" : "", gen.size(), text.c_str());
  return text.find("Paris") != std::string::npos;
}

// An image turn through the manager's tower (Bonsai 1: the pack's own
// vision_tower; Bonsai 2: the mmproj GGUF beside the file): a red disc
// on white, asked its colour.
bool
picture_(const char* env)
{
  const char* path = std::getenv(env);
  if (path == nullptr || *path == 0) { return true; }
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (mgr == nullptr) { return true; }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "f16";
  spec.page_tokens = 256;
  spec.max_pages = 16;
  auto lm = mgr->load(spec);
  BONSAI_CHECK(lm != nullptr && lm->valid());
  auto* vis = lm->metal_vision_encoder();
  BONSAI_CHECK(vis != nullptr);
  const int S = vis->config().spatial_merge;
  const int H = 384, W = 384;
  std::vector<std::uint8_t> img((std::size_t)3 * H * W, 255);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const int dx = x - W / 2, dy = y - H / 2;
      if (dx * dx + dy * dy > 120 * 120) { continue; }
      img[((std::size_t)1 * H + y) * W + x] = 30;    // G
      img[((std::size_t)2 * H + y) * W + x] = 30;    // B
    }
  }
  auto enc = vis->encode(img.data(), H, W);
  BONSAI_CHECK(enc.n_tokens > 0 && !enc.embeddings.empty());
  const int n_im = enc.n_tokens;
  const std::pair<int, int> grid{enc.grid_h / S, enc.grid_w / S};
  const auto* tpl = lm->chat_template();
  BONSAI_CHECK(tpl != nullptr);
  std::vector<std::int32_t> ids;
  tpl->render_user_turn_vlm("What colour is the shape in the picture? "
                            "Answer with one word.",
                            std::span<const int>(&n_im, 1), true, &ids);
  const std::int32_t pad = tpl->image_pad_token_id();
  std::vector<genai::TokenRef> refs;
  int off = 0;
  for (const std::int32_t id : ids) {
    genai::TokenRef r;
    if (id == pad && off < n_im) {
      r.kind = genai::TokenRef::Kind::ImageTokens;
      r.embeddings_buf = &enc.embeddings;
      r.image_token_offset = off++;
    } else {
      r.kind = genai::TokenRef::Kind::Text;
      r.text_id = id;
    }
    refs.push_back(r);
  }
  BONSAI_CHECK(off == n_im);
  auto ctx = lm->make_context();
  BONSAI_CHECK(ctx.valid());
  std::vector<std::int32_t> gen{lm->prefill_multimodal_metal(
      ctx, std::span<const genai::TokenRef>(refs),
      std::span<const std::pair<int, int>>(&grid, 1))};
  BONSAI_CHECK(gen[0] >= 0);
  for (int i = 0; i < 400 && !tpl->is_stop_token(gen.back()); ++i) {
    const std::int32_t t = lm->next_token(ctx);
    if (t < 0) { break; }
    gen.push_back(t);
  }
  std::string text = lm->tokenizer().decode(
      std::span<const std::int32_t>(gen.data(), gen.size()));
  std::printf("[bonsai] picture (%d image tokens, %zu out): '%s'\n", n_im,
              gen.size(), text.c_str());
  for (char& c : text) { c = (char)std::tolower((unsigned char)c); }
  const std::size_t end = text.rfind("think_end");
  return text.find("red", end == std::string::npos ? 0 : end) !=
         std::string::npos;
}

}  // namespace

// Bonsai 1: VPIPE_BONSAI1_TEST_MODEL_PATH = the MLX pack's directory.
// Bonsai 2: VPIPE_BONSAI2_TEST_MODEL_PATH = a GGUF FILE (PQ2_0), and
// VPIPE_BONSAI2_PTQ_TEST_MODEL_PATH = the PTQ1_0 file.
TEST(bonsai, mlx_pack_greedy_smoke)
{
  EXPECT_TRUE(greedy_smoke_("VPIPE_BONSAI1_TEST_MODEL_PATH"));
}

TEST(bonsai, mlx_batched_decode_matches_serial)
{
  EXPECT_TRUE(batched_matches_serial_("VPIPE_BONSAI1_TEST_MODEL_PATH"));
}

TEST(bonsai, gguf_greedy_smoke)
{
  EXPECT_TRUE(greedy_smoke_("VPIPE_BONSAI2_TEST_MODEL_PATH"));
  EXPECT_TRUE(greedy_smoke_("VPIPE_BONSAI2_PTQ_TEST_MODEL_PATH"));
}

TEST(bonsai, gguf_batched_decode_matches_serial)
{
  EXPECT_TRUE(batched_matches_serial_("VPIPE_BONSAI2_TEST_MODEL_PATH"));
}

TEST(bonsai, mlx_pack_matches_reference)
{
  for (const bool bf16 : {false, true}) {
    EXPECT_TRUE(golden_("VPIPE_BONSAI1_TEST_MODEL_PATH",
                        "VPIPE_BONSAI1_GOLDEN", bf16));
  }
}

// PQ2_0 and PTQ1_0 hold the same values (the dumper checked every tensor),
// so one golden serves both files.
TEST(bonsai, gguf_matches_reference)
{
  for (const bool bf16 : {false, true}) {
    EXPECT_TRUE(golden_("VPIPE_BONSAI2_TEST_MODEL_PATH",
                        "VPIPE_BONSAI2_GOLDEN", bf16));
  }
  EXPECT_TRUE(golden_("VPIPE_BONSAI2_PTQ_TEST_MODEL_PATH",
                      "VPIPE_BONSAI2_GOLDEN", false));
}

TEST(bonsai, chat_through_the_manager)
{
  for (const bool no_think : {false, true}) {
    EXPECT_TRUE(chat_("VPIPE_BONSAI1_TEST_MODEL_PATH", no_think));
    EXPECT_TRUE(chat_("VPIPE_BONSAI2_TEST_MODEL_PATH", no_think));
  }
}

TEST(bonsai, picture_through_the_manager)
{
  EXPECT_TRUE(picture_("VPIPE_BONSAI1_TEST_MODEL_PATH"));
  EXPECT_TRUE(picture_("VPIPE_BONSAI2_TEST_MODEL_PATH"));
}

// The vision towers against the reference's: transformers'
// Qwen3_5VisionModel in fp32 (tools/dump_bonsai_golden.py `tower`) over
// Bonsai 1's pack and over Bonsai 2's BF16 projector -- mapped from
// llama.cpp's names by the dumper, not by our loader -- on one 448 x 640
// photo, 280 merged rows. Bonsai 2 ships its projector twice, BF16 and
// Q8_0 (read through the GGUF's own dequantizer): both are held to the
// BF16 weights' fp32 run -- Q8_0 at the reference's bf16 distance, since
// its own rounding of the weights is in the figure; a wrong read is O(1)
// -- and the two must differ (it really was the other file). On an M5 the
// tower takes NAX attention (head dim 72, padded to 128) and the matmul2d
// GEMMs by default; the arms with each off say what each costs.
TEST(bonsai, tower_matches_reference)
{
  namespace fs = std::filesystem;
  struct Case {
    const char* name;
    const char* model_env;
    const char* gold_env;
    const char* mmproj;  // a file name part beside the GGUF, or none
    double bar;          // x the reference's own bf16 distance
  };
  const Case cases[] = {
      {"Bonsai 1", "VPIPE_BONSAI1_TEST_MODEL_PATH",
       "VPIPE_BONSAI1_TOWER_GOLDEN", nullptr, 0.5},
      {"Bonsai 2 BF16", "VPIPE_BONSAI2_TEST_MODEL_PATH",
       "VPIPE_BONSAI2_TOWER_GOLDEN", "mmproj-BF16", 0.5},
      {"Bonsai 2 Q8_0", "VPIPE_BONSAI2_TEST_MODEL_PATH",
       "VPIPE_BONSAI2_TOWER_GOLDEN", "mmproj-Q8_0", 1.0},
  };
  struct Arm {
    const char* name;
    const char* env;
  };
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  std::vector<Arm> arms = {{"default", nullptr}};
  if (mc->supports_matrix_cores()) {
    arms.push_back({"steel attention", "VPIPE_QWEN_VISION_NO_NAX_ATTN"});
    arms.push_back({"no matrix cores", "VPIPE_QWEN_VISION_NO_MMA2"});
  }
  const int H = 448, W = 640, rows = 280, D = 5120;
  std::vector<std::vector<double>> bonsai2;  // BF16, Q8_0 (default arm)
  for (const Case& c : cases) {
    const char* path = std::getenv(c.model_env);
    const char* gold = std::getenv(c.gold_env);
    if (path == nullptr || *path == 0 || gold == nullptr || *gold == 0) {
      continue;
    }
    const std::string g(gold);
    const std::vector<float> ref = read_f32_(g + "/tower.f32");
    const std::vector<float> rbf = read_f32_(g + "/tower_bf16.f32");
    std::vector<std::uint8_t> img((std::size_t)3 * H * W);
    std::FILE* f = std::fopen((g + "/image_rgb.u8").c_str(), "rb");
    const bool read_ok = f != nullptr &&
                         std::fread(img.data(), 1, img.size(), f) ==
                             img.size();
    if (f != nullptr) { std::fclose(f); }
    const bool sized = ref.size() == (std::size_t)rows * D &&
                       rbf.size() == ref.size();
    ASSERT_TRUE(read_ok && sized);
    if (!read_ok || !sized) { continue; }
    genai::ModelLoader loader(&sess);
    auto w = loader.load(path);  // load() reads the mmproj beside a GGUF
    ASSERT_TRUE(w.has_value() && w->config.vision.present);
    if (!w || !w->config.vision.present) { continue; }
    auto cfg = genai::MetalQwenVisionEncoder::config_from(w->config);
    if (c.mmproj != nullptr) {
      cfg.gguf_mmproj.clear();
      const fs::path dir = fs::path(path).parent_path();
      for (const auto& e : fs::directory_iterator(dir)) {
        const std::string n = e.path().filename().string();
        if (n.find(c.mmproj) != std::string::npos) {
          cfg.gguf_mmproj = e.path().string();
        }
      }
      if (cfg.gguf_mmproj.empty()) { continue; }
    }
    const std::vector<double> rd(ref.begin(), ref.end());
    const double ref_bf16 =
        rel_l2_(rd, std::vector<double>(rbf.begin(), rbf.end()));
    for (const Arm& arm : arms) {
      if (arm.env != nullptr) { ::setenv(arm.env, "1", 1); }
      auto mv = genai::MetalQwenVisionEncoder::load(path, mc, cfg);
      if (arm.env != nullptr) { ::unsetenv(arm.env); }
      ASSERT_TRUE(mv != nullptr);
      if (!mv) { continue; }
      auto r = mv->encode(img.data(), H, W);
      const bool f16 = !mv->is_bf16() &&
                       r.embeddings.byte_size() == ref.size() * 2;
      ASSERT_TRUE(f16);
      if (!f16) { continue; }
      const auto* q = static_cast<const _Float16*>(r.embeddings.contents());
      std::vector<double> got(ref.size());
      for (std::size_t i = 0; i < got.size(); ++i) { got[i] = (double)q[i]; }
      const double d = rel_l2_(rd, got);
      std::printf("[bonsai] %s tower (%s) vs fp32: %.4f (the reference's "
                  "bf16 %.4f)\n", c.name, arm.name, d, ref_bf16);
      EXPECT_TRUE(d < c.bar * ref_bf16);
      if (arm.env == nullptr && c.mmproj != nullptr) {
        bonsai2.push_back(std::move(got));
      }
    }
  }
  if (bonsai2.size() == 2) {
    const double d = rel_l2_(bonsai2[0], bonsai2[1]);
    std::printf("[bonsai] Bonsai 2 tower, Q8_0 vs BF16 projector: %.4f\n", d);
    EXPECT_TRUE(d > 0.0);
  }
}

// Either checkpoint reads as the Qwen3.5 family, text + picture in.
TEST(bonsai, detected_as_qwen35)
{
  for (const char* env : {"VPIPE_BONSAI1_TEST_MODEL_PATH",
                          "VPIPE_BONSAI2_TEST_MODEL_PATH"}) {
    const char* path = std::getenv(env);
    if (path == nullptr || *path == 0) { continue; }
    // The directory, as model-fetch / model-register name it, and a
    // bare file (the GGUF's own architecture).
    std::string dir(path);
    if (dir.size() > 5 && dir.compare(dir.size() - 5, 5, ".gguf") == 0) {
      const DetectedModel f = detect_model_dir(dir);
      std::printf("[bonsai] %s: model_type '%s' (by %s)\n", path,
                  f.model_type.c_str(), f.detected_by.c_str());
      EXPECT_TRUE(f.model_type == "qwen3.5");
      dir = std::filesystem::path(dir).parent_path().string();
    }
    const DetectedModel d = detect_model_dir(dir);
    std::printf("[bonsai] %s: model_type '%s' (by %s), inputs %zu\n",
                dir.c_str(), d.model_type.c_str(), d.detected_by.c_str(),
                d.inputs.size());
    EXPECT_TRUE(d.model_type == "qwen3.5");
    EXPECT_TRUE(std::find(d.inputs.begin(), d.inputs.end(), "image") !=
                d.inputs.end());
  }
}

// The plan books what the load holds: the weights claim (dir_weights_bytes;
// a GGUF at its served layout, its mmproj at two bytes an element) against
// the device's allocation over the model's load and its tower's. PTQ1_0
// holds ~1.3x its file. The page pool is two pages, so K/V is noise.
TEST(bonsai, the_plan_books_what_the_load_holds)
{
  for (const char* env : {"VPIPE_BONSAI1_TEST_MODEL_PATH",
                          "VPIPE_BONSAI2_TEST_MODEL_PATH",
                          "VPIPE_BONSAI2_PTQ_TEST_MODEL_PATH"}) {
    const char* path = std::getenv(env);
    if (path == nullptr || *path == 0) { continue; }
    const std::size_t booked = model_memory::dir_weights_bytes(path);
    std::error_code ec;
    const std::uintmax_t file =
        std::filesystem::is_regular_file(path, ec)
            ? std::filesystem::file_size(path, ec) : 0;
    Session sess;
    MetalCompute* mc = sess.metal_compute();
    if (mc == nullptr || !mc->valid()) { return; }
    genai::ModelLoader loader(&sess);
    auto w = loader.load(path);
    ASSERT_TRUE(w.has_value());
    if (!w) { continue; }
    auto mcfg = genai::MetalQwenModel::config_from(w->config);
    mcfg.use_bf16 = false;
    mcfg.page_tokens = 256;
    mcfg.max_pages = 2;
    const std::size_t a0 = mc->memory_budget_settled().allocated;
    auto m = genai::MetalQwenModel::load(path, mc, mcfg);
    ASSERT_TRUE(m != nullptr);
    if (!m) { continue; }
    const std::size_t a1 = mc->memory_budget_settled().allocated;
    std::unique_ptr<genai::MetalQwenVisionEncoder> vis;
    if (w->config.vision.present) {
      auto vc = genai::MetalQwenVisionEncoder::config_from(w->config);
      vc.gguf_mmproj = w->config.vision.mmproj_path;
      vis = genai::MetalQwenVisionEncoder::load(path, mc, vc);
    }
    const std::size_t a2 = mc->memory_budget_settled().allocated;
    const std::size_t held = a2 - a0;
    std::printf("[bonsai] %s: booked %.0f MB (file %.0f), held %.0f MB "
                "(model %.0f + tower %.0f)\n", path, booked / 1e6,
                file / 1e6, held / 1e6, (a1 - a0) / 1e6, (a2 - a1) / 1e6);
    EXPECT_TRUE(vis != nullptr || !w->config.vision.present);
    EXPECT_TRUE((double)held < 1.05 * (double)booked &&
                (double)held > 0.90 * (double)booked);
  }
}

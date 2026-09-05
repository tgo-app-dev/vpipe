// VDN-H3's vdn_solve delta rule against the reference's own VdnDelta.
//
// The rule is (S Diag(alpha) + B)(I + A)^-1, and the checkpoint was
// trained under the EXACT inverse. Two cheaper-looking spellings are
// available and neither is a substitute:
//
//   the first-order truncation (I - c^2 A)  -- a different operator once
//       A is unscaled, and trace(A) = sum(beta) exactly here because the
//       branch L2-normalises its keys, so at production geometry the
//       trace is in the hundreds;
//   the ordered product / WY form            -- the same contraction over
//       a TRIANGULAR mask of the same S x S matrix, which is exactly why
//       it factorises over chunks and this does not.
//
// So the test's job is to pin the solve, and to pin that the two ways of
// applying it -- an explicit inverse and a factor-plus-solve -- agree,
// because a Metal port wants the second and the reference stores the
// first.

#include "minitest.h"

#include "common/flex-data.h"
#include "generative-models/minimax-h3/vdn-delta-solve.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai::minimax_h3;

namespace {

std::string
golden_path_()
{
  if (const char* e = std::getenv("VPIPE_VDN_SOLVE_GOLDEN")) { return e; }
  const char* home = std::getenv("HOME");
  if (home == nullptr) { return ""; }
  return std::string(home) + "/dock/dump/vpipe-test/vdn/vdn-solve.json";
}

std::vector<float>
floats_(const FlexData& v)
{
  std::vector<float> out;
  FlexData owner = v;
  if (!owner.is_array()) { return out; }
  auto a = owner.as_array();
  out.reserve(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    out.push_back((float)a.at(i).as_real(0.0));
  }
  return out;
}

// Relative L2, the bar the rest of the tree states encoder accuracy in.
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

}  // namespace

TEST(vdn_delta_solve, matches_the_reference)
{
  std::ifstream in(golden_path_());
  if (!in) { return; }        // dock-side artifact; see gen_goldens.py
  FlexData doc = FlexData::from_json(in);
  ASSERT_TRUE(doc.is_object());
  if (!doc.is_object()) { return; }
  auto root = doc.as_object();
  ASSERT_TRUE(root.contains("cases"));
  if (!root.contains("cases")) { return; }
  FlexData cases_owner = root.at("cases");
  auto cases = cases_owner.as_array();
  EXPECT_TRUE(cases.size() >= 4);

  int ran = 0, at_128 = 0;
  for (std::size_t ci = 0; ci < cases.size(); ++ci) {
    FlexData c_owner = cases.at(ci);
    auto c = c_owner.as_object();
    const int F  = (int)c.at("num_frames").as_int(0);
    const int H  = (int)c.at("heads").as_int(0);
    const int d  = (int)c.at("d").as_int(0);
    const std::vector<float> A = floats_(c.at("A"));
    const std::vector<float> B = floats_(c.at("B"));
    const std::vector<float> alpha = floats_(c.at("alpha"));
    const std::vector<float> want_tr = floats_(c.at("transition"));
    const std::vector<float> want_in = floats_(c.at("injection"));

    std::vector<float> tr(want_tr.size()), inj(want_in.size());
    const bool ok = vdn::factor_apply(A.data(), B.data(), alpha.data(), F, H,
                                      d, d, tr.data(), inj.data());
    EXPECT_TRUE(ok);
    if (!ok) { continue; }
    ++ran;
    if (d == 128) { ++at_128; }

    const double e_tr = rel_l2_(tr, want_tr);
    const double e_in = rel_l2_(inj, want_in);
    // The reference factors in fp32 and forms L^-T L^-1; this accumulates
    // in double and associates the same way. The gap is the reference's
    // own fp32 rounding through a matrix whose condition number grows
    // with trace(A), so the bar is loose enough to allow that and tight
    // enough to catch a different operator (the truncation and the
    // ordered product both sit at 1e-1 .. 1e0 here).
    const bool close = e_tr >= 0.0 && e_tr < 2e-4 && e_in < 2e-4;
    EXPECT_TRUE(close);
    if (!close) {
      std::printf("[vdn_solve] d=%d corr=%.2f trace(A)~%.1f: transition "
                  "rel-L2 %.3e injection %.3e\n", d,
                  c.at("corr").as_real(0.0),
                  floats_(c.at("trace_A")).empty()
                      ? 0.0 : (double)floats_(c.at("trace_A"))[0],
                  e_tr, e_in);
    }
  }
  EXPECT_TRUE(ran >= 4);
  EXPECT_TRUE(at_128 >= 1);   // the real head dim, not just the readable one
}

TEST(vdn_delta_solve, the_two_spellings_agree)
{
  // The reference stores an explicit inverse and multiplies; a Metal port
  // would rather factor once and solve twice (each transition is read
  // exactly twice, once per scan direction). If those two disagree the
  // port is not the model, so pin them against each other on a matrix
  // with a production-scale trace.
  const int d = 48;
  std::vector<float> A((std::size_t)d * d, 0.0f);
  std::vector<float> alpha((std::size_t)d), state((std::size_t)d * d),
      B((std::size_t)d * d);
  // A = K^T diag(beta) K with L2-normed, correlated keys: PSD by
  // construction and with trace(A) = sum(beta), the real regime.
  const int S = 200;
  std::vector<float> k((std::size_t)d);
  unsigned seed = 12345u;
  auto rnd = [&seed]() {
    seed = seed * 1664525u + 1013904223u;
    return (float)((seed >> 8) & 0xFFFF) / 32768.0f - 1.0f;
  };
  double trace_a = 0.0;
  for (int s = 0; s < S; ++s) {
    double n = 0.0;
    for (int i = 0; i < d; ++i) { k[i] = rnd() + 0.8f; n += k[i] * k[i]; }
    n = std::sqrt(n);
    for (int i = 0; i < d; ++i) { k[i] = (float)(k[i] / n); }
    const float beta = 0.5f + 0.25f * rnd();
    trace_a += beta;
    for (int i = 0; i < d; ++i) {
      for (int j = 0; j < d; ++j) {
        A[(std::size_t)i * d + j] += beta * k[i] * k[j];
      }
    }
  }
  EXPECT_TRUE(trace_a > 80.0);        // the regime the header describes
  for (int i = 0; i < d; ++i) { alpha[i] = 0.5f + 0.2f * rnd(); }
  for (std::size_t i = 0; i < B.size(); ++i) { B[i] = rnd(); }
  for (std::size_t i = 0; i < state.size(); ++i) { state[i] = rnd(); }

  std::vector<float> tr((std::size_t)d * d), inj((std::size_t)d * d);
  ASSERT_TRUE(vdn::factor_apply(A.data(), B.data(), alpha.data(), 1, 1, d, d,
                                tr.data(), inj.data()));
  // state @ transition + injection
  std::vector<float> via_inv((std::size_t)d * d, 0.0f);
  for (int i = 0; i < d; ++i) {
    for (int j = 0; j < d; ++j) {
      double sum = inj[(std::size_t)i * d + j];
      for (int kk = 0; kk < d; ++kk) {
        sum += (double)state[(std::size_t)i * d + kk]
               * (double)tr[(std::size_t)kk * d + j];
      }
      via_inv[(std::size_t)i * d + j] = (float)sum;
    }
  }
  std::vector<float> via_solve((std::size_t)d * d, 0.0f);
  ASSERT_TRUE(vdn::solve_state(A.data(), alpha.data(), state.data(),
                               B.data(), d, d, via_solve.data()));
  EXPECT_TRUE(rel_l2_(via_solve, via_inv) < 1e-5);
}

TEST(vdn_delta_solve, refuses_a_matrix_that_is_not_positive_definite)
{
  // I + A has eigenvalues >= 1 in exact arithmetic, so a failure here
  // means the statistics upstream were computed in a precision that lost
  // the property -- which the reference documents happening in bf16,
  // where A's asymmetry pushes the smallest eigenvalue below zero. A
  // silent NaN would surface as a black frame many layers later.
  const int d = 4;
  std::vector<float> a((std::size_t)d * d, 0.0f), l((std::size_t)d * d);
  for (int i = 0; i < d; ++i) { a[(std::size_t)i * d + i] = -2.0f; }
  EXPECT_FALSE(vdn::cholesky_lower(a.data(), d, l.data()));
  // ... and accepts the identity, whose factor is the identity.
  std::vector<float> zero((std::size_t)d * d, 0.0f);
  ASSERT_TRUE(vdn::cholesky_lower(zero.data(), d, l.data()));
  bool eye = true;
  for (int i = 0; i < d; ++i) {
    for (int j = 0; j < d; ++j) {
      const float want = i == j ? 1.0f : 0.0f;
      eye = eye && std::fabs(l[(std::size_t)i * d + j] - want) < 1e-6f;
    }
  }
  EXPECT_TRUE(eye);
}

// ---- the Metal port ---------------------------------------------------
//
// A 128x128 fp32 matrix is 64 KB against this GPU's 32 KB threadgroup
// limit, so the factorisation cannot be a one-threadgroup-holds-the-
// matrix kernel and is panelled instead. That makes it the kind of code
// where an off-by-one in a block boundary is a wrong answer only in the
// trailing panel -- which a d = 32 test would never reach. So every case
// below runs at d = 128, the real head dim, and the batch is large
// enough that a threadgroup indexing bug cannot hide either.

#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"

using namespace vpipe::metal_compute;

namespace {

// A batch of realistic I + A: PSD by construction from L2-normed,
// correlated keys, with trace(A) = sum(beta) in the production range.
void
make_batch_(int batch, int d, int tokens, std::vector<float>* a,
            std::vector<float>* trace)
{
  a->assign((std::size_t)batch * d * d, 0.0f);
  trace->assign((std::size_t)batch, 0.0f);
  unsigned seed = 987u;
  auto rnd = [&seed]() {
    seed = seed * 1664525u + 1013904223u;
    return (float)((seed >> 8) & 0xFFFF) / 32768.0f - 1.0f;
  };
  std::vector<float> k((std::size_t)d);
  for (int b = 0; b < batch; ++b) {
    float* ab = a->data() + (std::size_t)b * d * d;
    for (int s = 0; s < tokens; ++s) {
      double n = 0.0;
      for (int i = 0; i < d; ++i) { k[i] = rnd() + 0.7f; n += k[i] * k[i]; }
      n = std::sqrt(n);
      for (int i = 0; i < d; ++i) { k[i] = (float)(k[i] / n); }
      const float beta = 0.5f + 0.25f * rnd();
      (*trace)[(std::size_t)b] += beta;
      for (int i = 0; i < d; ++i) {
        for (int j = 0; j <= i; ++j) {
          ab[(std::size_t)i * d + j] += beta * k[i] * k[j];
        }
      }
    }
    // Mirror into the upper triangle: the kernel reads only the lower
    // one, but the CPU oracle should be handed the same matrix a real
    // (symmetrised) producer would emit.
    for (int i = 0; i < d; ++i) {
      for (int j = 0; j < i; ++j) {
        ab[(std::size_t)j * d + i] = ab[(std::size_t)i * d + j];
      }
    }
  }
}

}  // namespace

TEST(vdn_delta_solve, metal_cholesky_matches_the_cpu_oracle)
{
  Session s;
  MetalCompute* mc = s.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  ComputeLibrary lib = mc->load_library("vdn_solve");
  ComputeFunction chol = lib.function("vdn_cholesky_f32");
  // An unvalidated ComputeFunction is a silent no-op, so say so rather
  // than passing a test that dispatched nothing.
  ASSERT_TRUE(chol.valid());
  if (!chol.valid()) { return; }

  const int d = 128, batch = 37, tokens = 900;   // trace(A) ~ 450
  std::vector<float> a;
  std::vector<float> trace;
  make_batch_(batch, d, tokens, &a, &trace);
  EXPECT_TRUE(trace[0] > 300.0f);        // the production regime

  const std::size_t n = (std::size_t)batch * d * d;
  SharedBuffer a_buf = mc->make_shared_buffer(n * sizeof(float));
  SharedBuffer l_buf = mc->make_shared_buffer(n * sizeof(float));
  SharedBuffer f_buf = mc->make_shared_buffer(sizeof(unsigned));
  ASSERT_TRUE(!a_buf.empty() && !l_buf.empty() && !f_buf.empty());
  if (a_buf.empty() || l_buf.empty() || f_buf.empty()) { return; }
  std::memcpy(a_buf.contents(), a.data(), n * sizeof(float));
  std::memset(f_buf.contents(), 0, sizeof(unsigned));

  {
    CommandStream stream = mc->make_command_stream();
    ComputeEncoder enc = stream.begin_compute();
    enc.set_function(chol);
    enc.set_buffer(0, a_buf);
    enc.set_buffer(1, l_buf);
    enc.set_constant(2, d);
    enc.set_buffer(3, f_buf);
    enc.dispatch({(unsigned)(batch * 128), 1, 1}, {128, 1, 1});
    enc.end();
    std::string err;
    ASSERT_TRUE(stream.commit().wait_ok(&err));
  }
  EXPECT_TRUE(*(const unsigned*)f_buf.contents() == 0u);

  const float* got = (const float*)l_buf.contents();
  std::vector<float> want((std::size_t)d * d);
  double worst_l = 0.0, worst_recon = 0.0;
  for (int b = 0; b < batch; ++b) {
    const float* ab = a.data() + (std::size_t)b * d * d;
    const float* gb = got + (std::size_t)b * d * d;
    ASSERT_TRUE(vdn::cholesky_lower(ab, d, want.data()));
    for (int i = 0; i < d; ++i) {
      for (int j = 0; j <= i; ++j) {
        worst_l = std::max(worst_l,
                           std::fabs((double)gb[(std::size_t)i * d + j]
                                     - (double)want[(std::size_t)i * d + j]));
      }
      // The strict upper triangle must be ZERO, not stale: the solve
      // reads L^T out of it and a leftover would be silently wrong.
      for (int j = i + 1; j < d; ++j) {
        worst_l = std::max(worst_l,
                           std::fabs((double)gb[(std::size_t)i * d + j]));
      }
    }
    // L L^T == I + A is the property that actually matters, and it does
    // not depend on the oracle being right.
    for (int i = 0; i < d; i += 17) {
      for (int j = 0; j <= i; j += 13) {
        double sum = 0.0;
        for (int k = 0; k <= j; ++k) {
          sum += (double)gb[(std::size_t)i * d + k]
                 * (double)gb[(std::size_t)j * d + k];
        }
        const double m = (double)ab[(std::size_t)i * d + j]
                         + (i == j ? 1.0 : 0.0);
        worst_recon = std::max(worst_recon,
                               std::fabs(sum - m) / (1.0 + std::fabs(m)));
      }
    }
  }
  const bool ok = worst_l < 2e-3 && worst_recon < 1e-5;
  EXPECT_TRUE(ok);
  if (!ok) {
    std::printf("[vdn_solve] metal chol: worst |dL| %.3e, worst L L^T-M "
                "rel %.3e\n", worst_l, worst_recon);
  }
}

TEST(vdn_delta_solve, metal_solve_matches_the_cpu_oracle)
{
  Session s;
  MetalCompute* mc = s.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  ComputeLibrary lib = mc->load_library("vdn_solve");
  ComputeFunction chol = lib.function("vdn_cholesky_f32");
  ComputeFunction solve = lib.function("vdn_chol_solve_f32");
  ASSERT_TRUE(chol.valid() && solve.valid());
  if (!chol.valid() || !solve.valid()) { return; }

  const int d = 128, batch = 5, rows = 128, tokens = 900;
  std::vector<float> a, trace;
  make_batch_(batch, d, tokens, &a, &trace);
  std::vector<float> c((std::size_t)batch * rows * d);
  unsigned seed = 42u;
  for (std::size_t i = 0; i < c.size(); ++i) {
    seed = seed * 1664525u + 1013904223u;
    c[i] = (float)((seed >> 8) & 0xFFFF) / 32768.0f - 1.0f;
  }

  const std::size_t na = (std::size_t)batch * d * d;
  const std::size_t nc = (std::size_t)batch * rows * d;
  SharedBuffer a_buf = mc->make_shared_buffer(na * sizeof(float));
  SharedBuffer l_buf = mc->make_shared_buffer(na * sizeof(float));
  SharedBuffer c_buf = mc->make_shared_buffer(nc * sizeof(float));
  SharedBuffer x_buf = mc->make_shared_buffer(nc * sizeof(float));
  SharedBuffer f_buf = mc->make_shared_buffer(sizeof(unsigned));
  if (a_buf.empty() || l_buf.empty() || c_buf.empty() || x_buf.empty()
      || f_buf.empty()) {
    EXPECT_TRUE(false);
    return;
  }
  std::memcpy(a_buf.contents(), a.data(), na * sizeof(float));
  std::memcpy(c_buf.contents(), c.data(), nc * sizeof(float));
  std::memset(f_buf.contents(), 0, sizeof(unsigned));

  {
    CommandStream stream = mc->make_command_stream();
    ComputeEncoder enc = stream.begin_compute();
    enc.set_function(chol);
    enc.set_buffer(0, a_buf);
    enc.set_buffer(1, l_buf);
    enc.set_constant(2, d);
    enc.set_buffer(3, f_buf);
    enc.dispatch({(unsigned)(batch * 128), 1, 1}, {128, 1, 1});
    enc.end();
    std::string err;
    ASSERT_TRUE(stream.commit().wait_ok(&err));
  }
  {
    CommandStream stream = mc->make_command_stream();
    ComputeEncoder enc = stream.begin_compute();
    enc.set_function(solve);
    enc.set_buffer(0, l_buf);
    enc.set_buffer(1, c_buf);
    enc.set_buffer(2, x_buf);
    enc.set_constant(3, d);
    enc.set_constant(4, rows);
    enc.dispatch({(unsigned)(((rows + 31) / 32) * 32), (unsigned)batch, 1},
                 {32, 1, 1});
    enc.end();
    std::string err;
    ASSERT_TRUE(stream.commit().wait_ok(&err));
  }

  const float* got = (const float*)x_buf.contents();
  std::vector<float> l((std::size_t)d * d), ct((std::size_t)d * rows),
      xt((std::size_t)d * rows);
  double worst = 0.0, denom = 0.0;
  for (int b = 0; b < batch; ++b) {
    const float* ab = a.data() + (std::size_t)b * d * d;
    const float* cb = c.data() + (std::size_t)b * rows * d;
    const float* gb = got + (std::size_t)b * rows * d;
    ASSERT_TRUE(vdn::cholesky_lower(ab, d, l.data()));
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < d; ++j) {
        ct[(std::size_t)j * rows + i] = cb[(std::size_t)i * d + j];
      }
    }
    vdn::cholesky_solve(l.data(), d, ct.data(), rows, xt.data());
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < d; ++j) {
        const double w = xt[(std::size_t)j * rows + i];
        const double g = gb[(std::size_t)i * d + j];
        worst = std::max(worst, (g - w) * (g - w));
        denom = std::max(denom, w * w);
      }
    }
  }
  const double rel = denom > 0.0 ? std::sqrt(worst / denom) : 1.0;
  const bool ok = rel < 5e-3;
  EXPECT_TRUE(ok);
  if (!ok) { std::printf("[vdn_solve] metal solve: worst rel %.3e\n", rel); }
}

TEST(vdn_delta_solve, metal_cholesky_reports_a_non_pd_matrix)
{
  // The flag exists because the failure it reports is not this kernel's
  // fault: I + A has eigenvalues >= 1 in exact arithmetic, so a refusal
  // means the statistics upstream lost the property. Silently emitting
  // NaN would surface as a black frame many layers away.
  Session s;
  MetalCompute* mc = s.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  ComputeLibrary lib = mc->load_library("vdn_solve");
  ComputeFunction chol = lib.function("vdn_cholesky_f32");
  ASSERT_TRUE(chol.valid());
  if (!chol.valid()) { return; }

  const int d = 128, batch = 2;
  std::vector<float> a((std::size_t)batch * d * d, 0.0f);
  // batch 0 fine (A = 0 -> M = I); batch 1 has a diagonal below -1.
  for (int i = 0; i < d; ++i) {
    a[(std::size_t)d * d + (std::size_t)i * d + i] = -3.0f;
  }
  const std::size_t n = (std::size_t)batch * d * d;
  SharedBuffer a_buf = mc->make_shared_buffer(n * sizeof(float));
  SharedBuffer l_buf = mc->make_shared_buffer(n * sizeof(float));
  SharedBuffer f_buf = mc->make_shared_buffer(sizeof(unsigned));
  if (a_buf.empty() || l_buf.empty() || f_buf.empty()) {
    EXPECT_TRUE(false);
    return;
  }
  std::memcpy(a_buf.contents(), a.data(), n * sizeof(float));
  std::memset(f_buf.contents(), 0, sizeof(unsigned));
  CommandStream stream = mc->make_command_stream();
  ComputeEncoder enc = stream.begin_compute();
  enc.set_function(chol);
  enc.set_buffer(0, a_buf);
  enc.set_buffer(1, l_buf);
  enc.set_constant(2, d);
  enc.set_buffer(3, f_buf);
  enc.dispatch({(unsigned)(batch * 128), 1, 1}, {128, 1, 1});
  enc.end();
  std::string err;
  ASSERT_TRUE(stream.commit().wait_ok(&err));
  EXPECT_TRUE(*(const unsigned*)f_buf.contents() > 0u);
}

TEST(vdn_delta_solve, metal_solve_throughput)
{
  // VPIPE_VDN_SOLVE_BENCH=1. At production geometry one DiT block needs
  // F * H = 102 * 56 = 5712 of these per denoise step, and there are 50
  // blocks -- so the question this answers is whether the solve is a
  // term worth optimising or a rounding error against a 33B DiT step.
  if (std::getenv("VPIPE_VDN_SOLVE_BENCH") == nullptr) { return; }
  Session s;
  MetalCompute* mc = s.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  ComputeLibrary lib = mc->load_library("vdn_solve");
  ComputeFunction chol = lib.function("vdn_cholesky_f32");
  ComputeFunction solve = lib.function("vdn_chol_solve_f32");
  if (!chol.valid() || !solve.valid()) { return; }

  const int d = 128, batch = 102 * 56, rows = 128;
  std::vector<float> a, trace;
  make_batch_(batch < 400 ? batch : 400, d, 900, &a, &trace);
  // Replicate the 400 distinct matrices across the full batch: the
  // kernel's cost is per matrix and does not depend on the values.
  const std::size_t per = (std::size_t)d * d;
  std::vector<float> full((std::size_t)batch * per);
  for (int b = 0; b < batch; ++b) {
    std::memcpy(full.data() + (std::size_t)b * per,
                a.data() + (std::size_t)(b % 400) * per,
                per * sizeof(float));
  }
  SharedBuffer a_buf = mc->make_shared_buffer(full.size() * sizeof(float));
  SharedBuffer l_buf = mc->make_shared_buffer(full.size() * sizeof(float));
  SharedBuffer c_buf =
      mc->make_shared_buffer((std::size_t)batch * rows * d * sizeof(float));
  SharedBuffer x_buf =
      mc->make_shared_buffer((std::size_t)batch * rows * d * sizeof(float));
  SharedBuffer f_buf = mc->make_shared_buffer(sizeof(unsigned));
  if (a_buf.empty() || l_buf.empty() || c_buf.empty() || x_buf.empty()) {
    return;
  }
  std::memcpy(a_buf.contents(), full.data(), full.size() * sizeof(float));
  std::memset(f_buf.contents(), 0, sizeof(unsigned));

  auto run = [&](bool with_solve) {
    CommandStream stream = mc->make_command_stream();
    ComputeEncoder enc = stream.begin_compute();
    enc.set_function(chol);
    enc.set_buffer(0, a_buf);
    enc.set_buffer(1, l_buf);
    enc.set_constant(2, d);
    enc.set_buffer(3, f_buf);
    enc.dispatch({(unsigned)(batch * 128), 1, 1}, {128, 1, 1});
    if (with_solve) {
      enc.set_function(solve);
      enc.set_buffer(0, l_buf);
      enc.set_buffer(1, c_buf);
      enc.set_buffer(2, x_buf);
      enc.set_constant(3, d);
      enc.set_constant(4, rows);
      enc.dispatch({(unsigned)(((rows + 31) / 32) * 32), (unsigned)batch, 1},
                   {32, 1, 1});
    }
    enc.end();
    std::string err;
    return stream.commit().wait_ok(&err);
  };

  ASSERT_TRUE(run(true));            // warm
  for (int with = 0; with < 2; ++with) {
    const auto t0 = std::chrono::steady_clock::now();
    const int reps = 3;
    for (int i = 0; i < reps; ++i) { run(with != 0); }
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count() / reps;
    std::printf("[vdn_solve] %d matrices d=%d %s: %.1f ms/block  "
                "-> %.2f s per 50-block step\n", batch, d,
                with != 0 ? "chol+solve" : "chol only ", ms, ms * 50 / 1000.0);
  }
}

namespace {

// chol -> trinv -> inv/transition -> injection, the shape the scan wants.
struct SolvePipeline {
  ComputeLibrary lib;
  ComputeFunction chol, trinv, inv_tr, gemm;

  bool load(MetalCompute* mc)
  {
    lib = mc->load_library("vdn_solve");
    chol   = lib.function("vdn_cholesky_f32");
    trinv  = lib.function("vdn_trinv_f32");
    inv_tr = lib.function("vdn_inv_and_transition_f32");
    gemm   = lib.function("vdn_gemm_nn_f32");
    return chol.valid() && trinv.valid() && inv_tr.valid() && gemm.valid();
  }

  // Encodes the whole chain; `stat_b` may be empty to skip the injection.
  // `stages`: 1 = chol, 2 = +trinv, 3 = +inv/transition, 4 = +injection.
  void encode(ComputeEncoder& enc, int d, int batch, const SharedBuffer& a,
              const SharedBuffer& l, const SharedBuffer& linv,
              const SharedBuffer& alpha, const SharedBuffer& inv,
              const SharedBuffer& trans, const SharedBuffer& stat_b,
              const SharedBuffer& injection, const SharedBuffer& fail,
              int stages = 4)
  {
    const unsigned nblk = (unsigned)((d + 31) / 32);
    enc.set_function(chol);
    enc.set_buffer(0, a);
    enc.set_buffer(1, l);
    enc.set_constant(2, d);
    enc.set_buffer(3, fail);
    enc.dispatch({(unsigned)batch * 128, 1, 1}, {128, 1, 1});

    if (stages < 2) { return; }
    enc.set_function(trinv);
    enc.set_buffer(0, l);
    enc.set_buffer(1, linv);
    enc.set_constant(2, d);
    enc.dispatch({nblk * 128, (unsigned)batch, 1}, {128, 1, 1});

    if (stages < 3) { return; }
    enc.set_function(inv_tr);
    enc.set_buffer(0, linv);
    enc.set_buffer(1, alpha);
    enc.set_buffer(2, inv);
    enc.set_buffer(3, trans);
    enc.set_constant(4, d);
    enc.dispatch({nblk * 16, nblk * 16, (unsigned)batch}, {16, 16, 1});

    if (stages >= 4 && !stat_b.empty()) {
      enc.set_function(gemm);
      enc.set_buffer(0, stat_b);
      enc.set_buffer(1, inv);
      enc.set_buffer(2, injection);
      enc.set_constant(3, d);
      enc.set_constant(4, d);
      enc.set_constant(5, d);
      enc.dispatch({nblk * 16, nblk * 16, (unsigned)batch}, {16, 16, 1});
    }
  }
};

}  // namespace

TEST(vdn_delta_solve, metal_factor_apply_matches_the_cpu_oracle)
{
  // The whole chain the scan consumes: transition = Diag(alpha)(I+A)^-1
  // and injection = B (I+A)^-1, via the FACTOR's inverse rather than a
  // per-step triangular solve. Checked against the CPU factor_apply,
  // which is itself checked against the reference.
  Session s;
  MetalCompute* mc = s.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  SolvePipeline p;
  ASSERT_TRUE(p.load(mc));
  if (!p.load(mc)) { return; }

  const int d = 128, batch = 11;
  std::vector<float> a, trace;
  make_batch_(batch, d, 900, &a, &trace);
  std::vector<float> alpha((std::size_t)batch * d);
  std::vector<float> bstat((std::size_t)batch * d * d);
  unsigned seed = 7u;
  auto rnd = [&seed]() {
    seed = seed * 1664525u + 1013904223u;
    return (float)((seed >> 8) & 0xFFFF) / 32768.0f - 1.0f;
  };
  for (std::size_t i = 0; i < alpha.size(); ++i) { alpha[i] = 0.5f + 0.2f * rnd(); }
  for (std::size_t i = 0; i < bstat.size(); ++i) { bstat[i] = rnd(); }

  const std::size_t nm = (std::size_t)batch * d * d;
  SharedBuffer a_b = mc->make_shared_buffer(nm * sizeof(float));
  SharedBuffer l_b = mc->make_shared_buffer(nm * sizeof(float));
  SharedBuffer li_b = mc->make_shared_buffer(nm * sizeof(float));
  SharedBuffer inv_b = mc->make_shared_buffer(nm * sizeof(float));
  SharedBuffer tr_b = mc->make_shared_buffer(nm * sizeof(float));
  SharedBuffer bs_b = mc->make_shared_buffer(nm * sizeof(float));
  SharedBuffer inj_b = mc->make_shared_buffer(nm * sizeof(float));
  SharedBuffer al_b =
      mc->make_shared_buffer(alpha.size() * sizeof(float));
  SharedBuffer f_b = mc->make_shared_buffer(sizeof(unsigned));
  if (a_b.empty() || tr_b.empty() || inj_b.empty()) {
    EXPECT_TRUE(false);
    return;
  }
  std::memcpy(a_b.contents(), a.data(), nm * sizeof(float));
  std::memcpy(bs_b.contents(), bstat.data(), nm * sizeof(float));
  std::memcpy(al_b.contents(), alpha.data(), alpha.size() * sizeof(float));
  std::memset(f_b.contents(), 0, sizeof(unsigned));

  {
    CommandStream stream = mc->make_command_stream();
    ComputeEncoder enc = stream.begin_compute();
    p.encode(enc, d, batch, a_b, l_b, li_b, al_b, inv_b, tr_b, bs_b, inj_b,
             f_b);
    enc.end();
    std::string err;
    ASSERT_TRUE(stream.commit().wait_ok(&err));
  }
  EXPECT_TRUE(*(const unsigned*)f_b.contents() == 0u);

  std::vector<float> want_tr(nm), want_inj(nm);
  ASSERT_TRUE(vdn::factor_apply(a.data(), bstat.data(), alpha.data(), batch,
                                1, d, d, want_tr.data(), want_inj.data()));
  std::vector<float> got_tr(nm), got_inj(nm);
  std::memcpy(got_tr.data(), tr_b.contents(), nm * sizeof(float));
  std::memcpy(got_inj.data(), inj_b.contents(), nm * sizeof(float));
  const double e_tr = rel_l2_(got_tr, want_tr);
  const double e_in = rel_l2_(got_inj, want_inj);
  // fp32 on the GPU against a double-accumulating CPU oracle, through a
  // matrix whose condition number grows with trace(A) ~ 450.
  const bool ok = e_tr >= 0.0 && e_tr < 5e-4 && e_in < 5e-4;
  EXPECT_TRUE(ok);
  if (!ok) {
    std::printf("[vdn_solve] metal factor_apply: transition rel-L2 %.3e, "
                "injection %.3e\n", e_tr, e_in);
  }
}

TEST(vdn_delta_solve, metal_factor_apply_throughput)
{
  // VPIPE_VDN_SOLVE_BENCH=1. The comparison that matters is against the
  // 334.6 ms/block the two-triangular-solve spelling measured.
  if (std::getenv("VPIPE_VDN_SOLVE_BENCH") == nullptr) { return; }
  Session s;
  MetalCompute* mc = s.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  SolvePipeline p;
  if (!p.load(mc)) { return; }

  const int d = 128, batch = 102 * 56;
  std::vector<float> a, trace;
  make_batch_(400, d, 900, &a, &trace);
  const std::size_t per = (std::size_t)d * d;
  const std::size_t nm = (std::size_t)batch * per;
  SharedBuffer a_b = mc->make_shared_buffer(nm * sizeof(float));
  SharedBuffer l_b = mc->make_shared_buffer(nm * sizeof(float));
  SharedBuffer li_b = mc->make_shared_buffer(nm * sizeof(float));
  SharedBuffer inv_b = mc->make_shared_buffer(nm * sizeof(float));
  SharedBuffer tr_b = mc->make_shared_buffer(nm * sizeof(float));
  SharedBuffer bs_b = mc->make_shared_buffer(nm * sizeof(float));
  SharedBuffer inj_b = mc->make_shared_buffer(nm * sizeof(float));
  SharedBuffer al_b =
      mc->make_shared_buffer((std::size_t)batch * d * sizeof(float));
  SharedBuffer f_b = mc->make_shared_buffer(sizeof(unsigned));
  if (a_b.empty() || inj_b.empty() || bs_b.empty()) { return; }
  for (int b = 0; b < batch; ++b) {
    std::memcpy((float*)a_b.contents() + (std::size_t)b * per,
                a.data() + (std::size_t)(b % 400) * per,
                per * sizeof(float));
  }
  std::memset(f_b.contents(), 0, sizeof(unsigned));

  auto run = [&](int stages) {
    CommandStream stream = mc->make_command_stream();
    ComputeEncoder enc = stream.begin_compute();
    p.encode(enc, d, batch, a_b, l_b, li_b, al_b, inv_b, tr_b, bs_b, inj_b,
             f_b, stages);
    enc.end();
    std::string err;
    return stream.commit().wait_ok(&err);
  };
  ASSERT_TRUE(run(4));
  const char* names[4] = {"chol            ", "+trinv          ",
                          "+inv/transition ", "+injection      "};
  for (int st = 1; st <= 4; ++st) {
    const auto t0 = std::chrono::steady_clock::now();
    const int reps = 3;
    for (int i = 0; i < reps; ++i) { run(st); }
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count() / reps;
    std::printf("[vdn_solve] %d matrices d=%d %s: %6.1f ms/block "
                "-> %5.2f s per 50-block step\n", batch, d, names[st - 1],
                ms, ms * 50 / 1000.0);
  }
}

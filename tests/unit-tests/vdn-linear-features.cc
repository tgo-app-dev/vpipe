// VDN-H3's linear-branch features against the REAL checkpoint's conv.
//
// Goldened with the released stage-dmd-step-250 short-conv weights, not
// synthetic ones: the four ways this stage goes wrong (which projections
// are convolved, the order of the separable halves, the temporal
// correlation's zero padding, and V not being L2-normalised) are all
// invisible to a shape check and most of them are invisible to a
// scale-invariant one too, because the branch norm downstream hides
// magnitude. So the bar is the reference's own numbers.
//
// Regenerate: ~/dock/dump/vpipe-test/vdn/gen_goldens.py features

#include "minitest.h"

#include "common/flex-data.h"
#include "generative-models/minimax-h3/vdn-linear-features.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai::minimax_h3;

namespace {

std::string
dir_()
{
  if (const char* e = std::getenv("VPIPE_VDN_FEATURES_GOLDEN")) { return e; }
  const char* home = std::getenv("HOME");
  if (home == nullptr) { return ""; }
  return std::string(home) + "/dock/dump/vpipe-test/vdn/features";
}

bool
read_bin_(const std::string& path, std::vector<float>* out)
{
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) { return false; }
  const std::streamsize n = in.tellg();
  if (n <= 0 || (n % (std::streamsize)sizeof(float)) != 0) { return false; }
  in.seekg(0);
  out->resize((std::size_t)n / sizeof(float));
  return (bool)in.read((char*)out->data(), n);
}

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

TEST(vdn_linear_features, matches_the_real_checkpoint)
{
  const std::string g = dir_();
  std::ifstream meta_in(g + "/meta.json");
  if (!meta_in) { return; }        // dock-side artifact; see gen_goldens.py
  FlexData meta = FlexData::from_json(meta_in);
  ASSERT_TRUE(meta.is_object());
  if (!meta.is_object()) { return; }
  auto m = meta.as_object();
  const int F  = (int)m.at("frames").as_int(0);
  const int gh = (int)m.at("grid_h").as_int(0);
  const int gw = (int)m.at("grid_w").as_int(0);
  const int H  = (int)m.at("heads").as_int(0);
  const int dh = (int)m.at("head_dim").as_int(0);
  const int K  = (int)m.at("kernel").as_int(0);
  ASSERT_TRUE(F > 0 && gh > 0 && gw > 0 && H > 0 && dh > 0 && K == 5);
  if (K != 5) { return; }
  const int channels = H * dh;

  std::vector<float> w_k_sp, w_k_tm, w_v_sp, w_v_tm;
  ASSERT_TRUE(read_bin_(g + "/w_k_sp.bin", &w_k_sp));
  ASSERT_TRUE(read_bin_(g + "/w_k_tm.bin", &w_k_tm));
  ASSERT_TRUE(read_bin_(g + "/w_v_sp.bin", &w_v_sp));
  ASSERT_TRUE(read_bin_(g + "/w_v_tm.bin", &w_v_tm));
  if (w_k_sp.size() != (std::size_t)channels * K * K) {
    EXPECT_TRUE(false);
    return;
  }

  struct Case {
    const char* name;
    bool has_conv;
    bool l2norm;
    const std::vector<float>* sp;
    const std::vector<float>* tm;
  };
  const Case cases[] = {
      {"q", false, true,  nullptr, nullptr},   // NOT convolved
      {"k", true,  true,  &w_k_sp, &w_k_tm},
      {"v", true,  false, &w_v_sp, &w_v_tm},   // NOT L2-normalised
  };

  int ran = 0;
  for (const Case& c : cases) {
    std::vector<float> in, want;
    if (!read_bin_(g + "/in_" + c.name + ".bin", &in)
        || !read_bin_(g + "/out_" + c.name + ".bin", &want)) {
      EXPECT_TRUE(false);
      continue;
    }
    vdn::ShortConv conv;
    conv.kernel = K;
    if (c.has_conv) {
      conv.spatial = c.sp->data();
      conv.temporal = c.tm->data();
    }
    std::vector<float> got(in.size());
    vdn::linear_features(in.data(), F, gh, gw, H, dh,
                         c.has_conv ? &conv : nullptr, c.l2norm, got.data());
    const double e = rel_l2_(got, want);
    // fp32 against the reference's fp32 through a 25-tap then 5-tap
    // chain; this is rounding, not method.
    const bool ok = e >= 0.0 && e < 2e-6;
    EXPECT_TRUE(ok);
    if (!ok) { std::printf("[vdn_features] %s rel-L2 %.3e\n", c.name, e); }
    ++ran;
  }
  EXPECT_TRUE(ran == 3);
}

TEST(vdn_linear_features, the_two_conv_halves_commute)
{
  // They act on DISJOINT axes -- the 5x5 is per frame and identical for
  // every frame, the 5-tap is per spatial position and identical for
  // every position -- so as linear operators they commute, and the zero
  // padding does not change that: padding the frame axis is invisible to
  // an operator that never touches it.
  //
  // Worth pinning rather than assuming, because it is what licenses the
  // inference path to fuse the temporal half with the SiLU/L2 epilogue
  // and leave the spatial half on its own kernel. (I wrote this test the
  // other way round first, expecting the padding to make the order
  // observable. It does not.)
  const int F = 4, gh = 3, gw = 3, C = 2, K = 5;
  const int S = gh * gw;
  std::vector<float> sp((std::size_t)C * K * K), tm((std::size_t)C * K);
  unsigned seed = 3u;
  auto rnd = [&seed]() {
    seed = seed * 1664525u + 1013904223u;
    return (float)((seed >> 8) & 0xFFFF) / 32768.0f - 1.0f;
  };
  for (std::size_t i = 0; i < sp.size(); ++i) { sp[i] = rnd(); }
  for (std::size_t i = 0; i < tm.size(); ++i) { tm[i] = rnd(); }
  std::vector<float> x((std::size_t)F * S * C);
  for (std::size_t i = 0; i < x.size(); ++i) { x[i] = rnd(); }

  vdn::ShortConv conv;
  conv.kernel = K;
  conv.spatial = sp.data();
  conv.temporal = tm.data();

  std::vector<float> a(x.size()), b(x.size()), t1(x.size()), t2(x.size());
  vdn::short_conv_spatial(x.data(), F, gh, gw, C, conv, t1.data());
  vdn::short_conv_temporal(t1.data(), F, S, C, conv, a.data());
  vdn::short_conv_temporal(x.data(), F, S, C, conv, t2.data());
  vdn::short_conv_spatial(t2.data(), F, gh, gw, C, conv, b.data());

  double worst = 0.0, scale = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    worst = std::max(worst, std::fabs((double)a[i] - (double)b[i]));
    scale = std::max(scale, std::fabs((double)a[i]));
  }
  const bool ok = scale > 0.0 && worst < 1e-5 * scale;
  EXPECT_TRUE(ok);
  if (!ok) {
    std::printf("[vdn_features] order worst |d| %.3e against scale %.3e\n",
                worst, scale);
  }
}

TEST(vdn_linear_features, v_keeps_its_magnitude)
{
  // V is not L2-normalised, and the branch norm downstream would hide a
  // wrongly-normalised V from any scale-invariant check. Pin it here,
  // where the magnitude still means something.
  const int F = 2, gh = 2, gw = 2, H = 1, dh = 8;
  std::vector<float> x((std::size_t)F * gh * gw * H * dh, 3.0f);
  std::vector<float> out(x.size());
  vdn::linear_features(x.data(), F, gh, gw, H, dh, nullptr, /*l2norm=*/false,
                       out.data());
  // SiLU(3) = 3 * sigmoid(3) = 2.8577...
  const float want = 3.0f / (1.0f + std::exp(-3.0f));
  bool ok = true;
  for (float v : out) { ok = ok && std::fabs(v - want) < 1e-6f; }
  EXPECT_TRUE(ok);

  vdn::linear_features(x.data(), F, gh, gw, H, dh, nullptr, /*l2norm=*/true,
                       out.data());
  // Normalised, every row of 8 equal entries becomes 1/sqrt(8).
  const float unit = 1.0f / std::sqrt(8.0f);
  ok = true;
  for (float v : out) { ok = ok && std::fabs(v - unit) < 1e-6f; }
  EXPECT_TRUE(ok);
}

// ---- the feature stage on Metal --------------------------------------

#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"

#include <cstring>

using namespace vpipe::metal_compute;

TEST(vdn_linear_features, metal_matches_the_cpu_reference)
{
  // Same goldens, same real conv weights, all three projections -- so
  // this checks the GPU against the reference and not merely against the
  // CPU port's reading of it. Q exercises the no-conv path, V the
  // no-L2-norm one; running only K would leave both flags untested.
  const std::string g = dir_();
  std::ifstream meta_in(g + "/meta.json");
  if (!meta_in) { return; }
  FlexData meta = FlexData::from_json(meta_in);
  if (!meta.is_object()) { return; }
  auto m = meta.as_object();
  const int F  = (int)m.at("frames").as_int(0);
  const int gh = (int)m.at("grid_h").as_int(0);
  const int gw = (int)m.at("grid_w").as_int(0);
  const int H  = (int)m.at("heads").as_int(0);
  const int dh = (int)m.at("head_dim").as_int(0);
  const int K  = (int)m.at("kernel").as_int(0);
  const int C  = H * dh;
  const int S  = gh * gw;

  Session s;
  MetalCompute* mc = s.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  ComputeLibrary lib = mc->load_library("vdn_branch");
  ComputeFunction sp = lib.function("vdn_conv_spatial_f32");
  ComputeFunction ta = lib.function("vdn_temporal_act_f32");
  ASSERT_TRUE(sp.valid() && ta.valid());
  if (!sp.valid() || !ta.valid()) { return; }

  std::vector<float> w_k_sp, w_k_tm, w_v_sp, w_v_tm;
  if (!read_bin_(g + "/w_k_sp.bin", &w_k_sp)
      || !read_bin_(g + "/w_k_tm.bin", &w_k_tm)
      || !read_bin_(g + "/w_v_sp.bin", &w_v_sp)
      || !read_bin_(g + "/w_v_tm.bin", &w_v_tm)) {
    EXPECT_TRUE(false);
    return;
  }

  struct Case {
    const char* name;
    bool conv;
    bool l2;
    const std::vector<float>* wsp;
    const std::vector<float>* wtm;
  };
  const Case cases[] = {{"q", false, true, nullptr, nullptr},
                        {"k", true, true, &w_k_sp, &w_k_tm},
                        {"v", true, false, &w_v_sp, &w_v_tm}};

  int ran = 0;
  for (const Case& c : cases) {
    std::vector<float> in, want;
    if (!read_bin_(g + "/in_" + c.name + ".bin", &in)
        || !read_bin_(g + "/out_" + c.name + ".bin", &want)) {
      EXPECT_TRUE(false);
      continue;
    }
    const std::size_t n = in.size();
    SharedBuffer ib = mc->make_shared_buffer(n * sizeof(float));
    SharedBuffer mb = mc->make_shared_buffer(n * sizeof(float));
    SharedBuffer ob = mc->make_shared_buffer(n * sizeof(float));
    SharedBuffer wsp = mc->make_shared_buffer(
        (std::size_t)C * K * K * sizeof(float));
    SharedBuffer wtm =
        mc->make_shared_buffer((std::size_t)C * K * sizeof(float));
    if (ib.empty() || ob.empty() || wsp.empty() || wtm.empty()) {
      EXPECT_TRUE(false);
      continue;
    }
    std::memcpy(ib.contents(), in.data(), n * sizeof(float));
    if (c.conv) {
      std::memcpy(wsp.contents(), c.wsp->data(),
                  (std::size_t)C * K * K * sizeof(float));
      std::memcpy(wtm.contents(), c.wtm->data(),
                  (std::size_t)C * K * sizeof(float));
    }
    {
      CommandStream stream = mc->make_command_stream();
      ComputeEncoder enc = stream.begin_compute();
      if (c.conv) {
        enc.set_function(sp);
        enc.set_buffer(0, ib);
        enc.set_buffer(1, wsp);
        enc.set_buffer(2, mb);
        enc.set_constant(3, F);
        enc.set_constant(4, gh);
        enc.set_constant(5, gw);
        enc.set_constant(6, C);
        enc.set_constant(7, K);
        // The strided-source slots, at their TIGHT defaults. Both
        // pointer slots are always bound -- an unbound Metal buffer is
        // not reliably null and an unbound constant is not reliably
        // zero, so a caller that leaves them out reads garbage strides
        // rather than falling back to anything.
        enc.set_buffer(8, ib);
        enc.set_constant(9, 0);
        enc.set_constant(10, C);
        enc.set_constant(11, dh);
        enc.set_constant(12, dh);
        // Cells per thread -- see VB_CONV_NX. The kernel checks it, so a
        // stale value here is slow and not wrong.
        const int crun = 8;
        enc.set_constant(13, crun);
        const int cnxg = (gw + crun - 1) / crun;
        enc.set_buffer(14, mb);        // fp32 out: the tight default
        enc.set_constant(15, 0);
        enc.dispatch({(unsigned)(F * gh * cnxg * C), 1, 1}, {256, 1, 1});
      }
      const int use_t = c.conv ? 1 : 0;
      const int l2 = c.l2 ? 1 : 0;
      enc.set_function(ta);
      enc.set_buffer(0, c.conv ? mb : ib);
      enc.set_buffer(1, wtm);
      enc.set_buffer(2, ob);
      enc.set_constant(3, F);
      enc.set_constant(4, S);
      enc.set_constant(5, H);
      enc.set_constant(6, dh);
      enc.set_constant(7, K);
      enc.set_constant(8, use_t);
      enc.set_constant(9, l2);
      const int fbase = 0, ftot = F, obase = 0;
      enc.set_constant(10, fbase);
      enc.set_constant(11, ftot);
      enc.set_constant(12, obase);
      enc.set_buffer(13, c.conv ? mb : ib);
      enc.set_constant(14, 0);
      enc.set_constant(15, C);
      enc.set_constant(16, dh);
      enc.set_buffer(17, ob);
      enc.set_constant(18, 0);
      enc.dispatch({(unsigned)(F * S) * 128, (unsigned)H, 1}, {128, 1, 1});
      enc.end();
      std::string err;
      ASSERT_TRUE(stream.commit().wait_ok(&err));
    }
    std::vector<float> got(n);
    std::memcpy(got.data(), ob.contents(), n * sizeof(float));
    const double e = rel_l2_(got, want);
    const bool ok = e >= 0.0 && e < 5e-6;
    EXPECT_TRUE(ok);
    if (!ok) {
      std::printf("[vdn_features] metal %s rel-L2 %.3e\n", c.name, e);
    }
    ++ran;
  }
  EXPECT_TRUE(ran == 3);
}

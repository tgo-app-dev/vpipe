// The branch as a COMPONENT, driven from the real checkpoint.
//
// Everything below this has been checked with tensors handed in from a
// golden. This test is the first that makes the component read the
// released `linear_branch/model.safetensors` itself -- so it is what
// catches the two things a golden cannot: a wrong tensor NAME (the
// loader would report a missing weight, or worse, find a different one)
// and a wrong bf16 -> f32 conversion.
//
// The golden's own weights are used as the ORACLE for the loaded ones,
// which is the only comparison that separates "loaded the right bytes"
// from "computed the right answer with the wrong bytes".

#include "minitest.h"

#include "common/flex-data.h"
#include "common/session.h"
#include "generative-models/minimax-h3/metal-vdn-branch.h"
#include "generative-models/minimax-h3/vdn-config.h"
#include "generative-models/weight-set.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using namespace vpipe::genai::minimax_h3;
using namespace vpipe::metal_compute;

namespace {

std::string
stage_()
{
  if (const char* e = std::getenv("VPIPE_VDN_MODEL_PATH")) {
    return std::string(e) + "/stage-dmd-step-250";
  }
  const char* home = std::getenv("HOME");
  if (home == nullptr) { return ""; }
  return std::string(home)
         + "/dock/dump/vpipe-test/models/OpenVDN/vdn-minimax-h3/"
           "stage-dmd-step-250";
}

std::string
golden_()
{
  if (const char* e = std::getenv("VPIPE_VDN_BRANCH_GOLDEN")) { return e; }
  const char* home = std::getenv("HOME");
  if (home == nullptr) { return ""; }
  return std::string(home) + "/dock/dump/vpipe-test/vdn/branch";
}

bool
read_bin_(const std::string& p, std::vector<float>* out)
{
  std::ifstream in(p, std::ios::binary | std::ios::ate);
  if (!in) { return false; }
  const std::streamsize n = in.tellg();
  if (n <= 0) { return false; }
  in.seekg(0);
  out->resize((std::size_t)n / sizeof(float));
  return (bool)in.read((char*)out->data(), n);
}

double
rel_l2_(const float* got, const float* want, std::size_t n)
{
  if (got == nullptr || want == nullptr || n == 0) { return -1.0; }
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = (double)got[i] - (double)want[i];
    num += d * d;
    den += (double)want[i] * (double)want[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

}  // namespace

TEST(metal_vdn_branch, loads_the_real_checkpoint_and_reproduces_the_golden)
{
  const std::string stage = stage_(), gdir = golden_();
  std::ifstream mi(gdir + "/meta.json");
  if (!mi) { return; }        // dock-side artifacts; see gen_goldens.py
  FlexData meta = FlexData::from_json(mi);
  if (!meta.is_object()) { return; }
  auto m = meta.as_object();
  const int F = (int)m.at("frames").as_int(0);
  const int gh = (int)m.at("grid_h").as_int(0);
  const int gw = (int)m.at("grid_w").as_int(0);
  const int H = (int)m.at("heads").as_int(0);
  const int d = (int)m.at("head_dim").as_int(0);
  const int hidden = (int)m.at("hidden").as_int(0);
  const int Lt = (int)m.at("text_len").as_int(0);
  const int S = gh * gw, C = H * d;

  Session s;
  MetalCompute* mc = s.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  vdn::Config cfg;
  std::string err;
  if (!vdn::load_config(stage, &cfg, &err)) { return; }   // not downloaded
  ASSERT_TRUE(cfg.supported(&err));

  std::shared_ptr<WeightSet> ws =
      WeightSet::open(stage + "/linear_branch", nullptr);
  ASSERT_TRUE((bool)ws);
  if (!ws) { return; }

  MetalVdnBranch::Dims dims;
  dims.heads = H;
  dims.head_dim = d;
  dims.hidden = hidden;
  dims.n_layers = 50;
  // The GOLDEN is an fp32 reference, so this arm stays wide and keeps
  // its exact 1e-5 bar. What bf16 costs is measured against this, in
  // narrow_features_track_the_wide_path.
  dims.bf16_features = false;
  std::unique_ptr<MetalVdnBranch> br =
      MetalVdnBranch::load(ws, mc, cfg, dims, &err);
  ASSERT_TRUE((bool)br);
  if (!br) {
    std::printf("[vdn_component] load: %s\n", err.c_str());
    return;
  }
  // Block 0, because that is the block the goldens were taken from.
  ASSERT_TRUE(br->ensure_block(0, &err));
  if (!br->block_ready(0)) {
    std::printf("[vdn_component] ensure_block: %s\n", err.c_str());
    return;
  }

  auto load = [&](const char* name, std::vector<float>* v) {
    return read_bin_(gdir + "/" + name + ".bin", v);
  };
  std::vector<float> xv, qr, kr, vr, tx, tk, tv, want;
  if (!load("xv", &xv) || !load("q_raw", &qr) || !load("k_raw", &kr)
      || !load("v_raw", &vr) || !load("text_x", &tx)
      || !load("text_k_raw", &tk) || !load("text_v_raw", &tv)
      || !load("readout", &want)) {
    EXPECT_TRUE(false);
    return;
  }

  auto buf = [&](const std::vector<float>& v) {
    SharedBuffer b = mc->make_shared_buffer(v.size() * 4);
    if (!b.empty()) { std::memcpy(b.contents(), v.data(), v.size() * 4); }
    return b;
  };
  SharedBuffer xb = buf(xv), qb = buf(qr), kb = buf(kr), vb = buf(vr);
  SharedBuffer txb = buf(tx), tkb = buf(tk), tvb = buf(tv);
  SharedBuffer ob = mc->make_shared_buffer((std::size_t)F * S * C * 4);
  ASSERT_TRUE(!ob.empty());
  if (ob.empty()) { return; }
  // The anchor rows are the caller's to leave alone: the branch writes
  // only the frames it owns, so the buffer starts zeroed and the two
  // anchor frames stay that way.
  std::memset(ob.contents(), 0, (std::size_t)F * S * C * 4);

  MetalVdnBranch::Geometry geo;
  geo.frames = F;
  geo.grid_h = gh;
  geo.grid_w = gw;
  geo.text_len = Lt;
  const std::vector<vdn::Bound> bounds =
      vdn::window_bounds(F, (int)m.at("radius").as_int(0),
                         (int)m.at("chunk").as_int(0));

  MetalVdnBranch::Inputs in;
  in.x = &xb; in.q_raw = &qb; in.k_raw = &kb; in.v_raw = &vb;
  in.text_x = &txb; in.text_k = &tkb; in.text_v = &tvb;

  br->clear_solve_failures();
  {
    CommandStream stream = mc->make_command_stream();
    ComputeEncoder enc = stream.begin_compute();
    const bool ok = br->encode(enc, 0, geo, bounds, in, ob, &err);
    EXPECT_TRUE(ok);
    if (!ok) { std::printf("[vdn_component] encode: %s\n", err.c_str()); }
    enc.end();
    std::string gerr;
    ASSERT_TRUE(stream.commit().wait_ok(&gerr));
  }
  // I + A has eigenvalues >= 1 in exact arithmetic, so a refusal is a
  // statement about the statistics' precision, not about the solve.
  EXPECT_TRUE(br->solve_failures() == 0u);

  const double e = rel_l2_((const float*)ob.contents(), want.data(),
                           want.size());
  const bool ok = e >= 0.0 && e < 1e-5;
  EXPECT_TRUE(ok);
  if (!ok) { std::printf("[vdn_component] readout rel-L2 %.3e\n", e); }

  // The anchor frames must be EXACTLY zero, not merely small: the
  // softmax side makes them exact in both directions, so anything the
  // branch added there would be counted twice.
  const float* out = (const float*)ob.contents();
  bool zero = true;
  for (std::size_t i = 0; i < (std::size_t)S * C; ++i) {
    zero = zero && out[i] == 0.0f
           && out[((std::size_t)(F - 1) * S) * C + i] == 0.0f;
  }
  EXPECT_TRUE(zero);

  // Re-encoding must not re-allocate or drift: a second forward at the
  // same geometry reuses the scratch, and a component that got that
  // wrong would show up as a changed answer rather than as a crash.
  std::vector<float> first((std::size_t)F * S * C);
  std::memcpy(first.data(), out, first.size() * 4);
  {
    CommandStream stream = mc->make_command_stream();
    ComputeEncoder enc = stream.begin_compute();
    EXPECT_TRUE(br->encode(enc, 0, geo, bounds, in, ob, &err));
    enc.end();
    std::string gerr;
    ASSERT_TRUE(stream.commit().wait_ok(&gerr));
  }
  EXPECT_TRUE(std::memcmp(first.data(), ob.contents(), first.size() * 4) == 0);
}

TEST(metal_vdn_branch, refuses_a_rule_it_cannot_run)
{
  // A checkpoint asking for a delta rule this port has not implemented
  // must be refused at LOAD, with the rule named. Running `sana_scaled`
  // where the weights were trained under `vdn_solve` produces an
  // entirely normal-looking video.
  Session s;
  MetalCompute* mc = s.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  const std::string stage = stage_();
  std::shared_ptr<WeightSet> ws =
      WeightSet::open(stage + "/linear_branch", nullptr);
  if (!ws) { return; }

  vdn::Config cfg;
  std::string err;
  if (!vdn::load_config(stage, &cfg, &err)) { return; }
  cfg.delta_rule = "sana_scaled";
  MetalVdnBranch::Dims dims;
  EXPECT_TRUE(MetalVdnBranch::load(ws, mc, cfg, dims, &err) == nullptr);
  EXPECT_TRUE(err.find("sana_scaled") != std::string::npos);

  // ...and a branch head dim the kernels are not built for.
  vdn::Config narrow;
  if (vdn::load_config(stage, &narrow, &err)) {
    narrow.linear_head_dim = 64;
    EXPECT_TRUE(MetalVdnBranch::load(ws, mc, narrow, dims, &err) == nullptr);
  }
}

TEST(metal_vdn_branch, tile_size_does_not_change_the_answer)
{
  // The per-token half walks the clip a few frames at a time, and the
  // short conv's temporal stencil reaches two frames either side -- so a
  // tile boundary that is INTERIOR to the clip must read its neighbour
  // while one at the clip's end must read a zero. Get that backwards and
  // the error lives only on the tile seams, which a single-tile run (the
  // golden's clip is exactly one tile) never visits.
  //
  // So the property is invariance: the tile is a memory knob, and the
  // answer may not depend on it.
  const std::string stage = stage_(), gdir = golden_();
  std::ifstream mi(gdir + "/meta.json");
  if (!mi) { return; }
  FlexData meta = FlexData::from_json(mi);
  if (!meta.is_object()) { return; }
  auto m = meta.as_object();
  const int F = (int)m.at("frames").as_int(0);
  const int gh = (int)m.at("grid_h").as_int(0);
  const int gw = (int)m.at("grid_w").as_int(0);
  const int H = (int)m.at("heads").as_int(0);
  const int d = (int)m.at("head_dim").as_int(0);
  const int hidden = (int)m.at("hidden").as_int(0);
  const int Lt = (int)m.at("text_len").as_int(0);
  const int S = gh * gw, C = H * d;

  Session s;
  MetalCompute* mc = s.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  vdn::Config cfg;
  std::string err;
  if (!vdn::load_config(stage, &cfg, &err)) { return; }
  std::shared_ptr<WeightSet> ws =
      WeightSet::open(stage + "/linear_branch", nullptr);
  if (!ws) { return; }

  std::vector<float> xv, qr, kr, vr, tx, tk, tv, want;
  auto load = [&](const char* n, std::vector<float>* v) {
    return read_bin_(gdir + "/" + n + ".bin", v);
  };
  if (!load("xv", &xv) || !load("q_raw", &qr) || !load("k_raw", &kr)
      || !load("v_raw", &vr) || !load("text_x", &tx)
      || !load("text_k_raw", &tk) || !load("text_v_raw", &tv)
      || !load("readout", &want)) {
    EXPECT_TRUE(false);
    return;
  }
  auto buf = [&](const std::vector<float>& v) {
    SharedBuffer b = mc->make_shared_buffer(v.size() * 4);
    if (!b.empty()) { std::memcpy(b.contents(), v.data(), v.size() * 4); }
    return b;
  };
  SharedBuffer xb = buf(xv), qb = buf(qr), kb = buf(kr), vb = buf(vr);
  SharedBuffer txb = buf(tx), tkb = buf(tk), tvb = buf(tv);

  MetalVdnBranch::Geometry geo;
  geo.frames = F; geo.grid_h = gh; geo.grid_w = gw; geo.text_len = Lt;
  const std::vector<vdn::Bound> bounds =
      vdn::window_bounds(F, (int)m.at("radius").as_int(0),
                         (int)m.at("chunk").as_int(0));
  MetalVdnBranch::Inputs in;
  in.x = &xb; in.q_raw = &qb; in.k_raw = &kb; in.v_raw = &vb;
  in.text_x = &txb; in.text_k = &tkb; in.text_v = &tvb;

  // The clip the golden uses is 4 inner frames, so tiles of 1, 2 and 3
  // all cross a boundary and 4 does not.
  const int tiles[] = {1, 2, 3, 4};
  double worst = 0.0;
  int ran = 0;
  for (int t : tiles) {
    MetalVdnBranch::Dims dims;
    dims.heads = H; dims.head_dim = d; dims.hidden = hidden;
    dims.n_layers = 50; dims.frame_tile = t;
    dims.bf16_features = false;      // against an fp32 golden
    std::unique_ptr<MetalVdnBranch> br =
        MetalVdnBranch::load(ws, mc, cfg, dims, &err);
    if (!br || !br->ensure_block(0, &err)) { return; }
    SharedBuffer ob = mc->make_shared_buffer((std::size_t)F * S * C * 4);
    if (ob.empty()) { return; }
    std::memset(ob.contents(), 0, (std::size_t)F * S * C * 4);
    CommandStream stream = mc->make_command_stream();
    ComputeEncoder enc = stream.begin_compute();
    EXPECT_TRUE(br->encode(enc, 0, geo, bounds, in, ob, &err));
    enc.end();
    std::string gerr;
    ASSERT_TRUE(stream.commit().wait_ok(&gerr));
    EXPECT_TRUE(br->solve_failures() == 0u);
    const double e = rel_l2_((const float*)ob.contents(), want.data(),
                             want.size());
    worst = std::max(worst, e);
    const bool ok = e >= 0.0 && e < 1e-5;
    EXPECT_TRUE(ok);
    if (!ok) {
      std::printf("[vdn_component] tile %d: rel-L2 %.3e\n", t, e);
    }
    ++ran;
  }
  EXPECT_TRUE(ran == 4);
  if (ran == 4) {
    std::printf("[vdn_component] worst over tiles 1..4: %.3e\n", worst);
  }
}

TEST(metal_vdn_branch, reads_a_fused_per_head_projection_in_place)
{
  // THE TRANSFORMER HAS NO q_raw, k_raw OR v_raw, and this is the test
  // that says the branch can work without them.
  //
  // H3 projects one fused [rows, 3*inner] buffer whose released
  // checkpoint groups it PER HEAD -- h0(q,k,v) | h1(q,k,v) | ... -- and
  // whose video rows are a run inside a packed sequence that also
  // carries the prompt. Copying three tight fp32 tensors out of that
  // costs 8.67 GB a block at production geometry, which is more than
  // the tiling and the bank aliasing took off the branch in the first
  // place. So the kernels read it where it lies.
  //
  // TWO ARMS, and the first is the one that matters. Strided fp32 must
  // be BIT-IDENTICAL to the tight run: the arithmetic is unchanged and
  // only the addressing moved, so anything but equality is a wrong
  // index -- which is the silent failure here, since a transposed or
  // mis-grouped read still produces a perfectly plausible tensor. The
  // bf16 arm then measures only what narrowing the inputs costs.
  const std::string stage = stage_(), gdir = golden_();
  std::ifstream mi(gdir + "/meta.json");
  if (!mi) { return; }
  FlexData meta = FlexData::from_json(mi);
  if (!meta.is_object()) { return; }
  auto m = meta.as_object();
  const int F = (int)m.at("frames").as_int(0);
  const int gh = (int)m.at("grid_h").as_int(0);
  const int gw = (int)m.at("grid_w").as_int(0);
  const int H = (int)m.at("heads").as_int(0);
  const int d = (int)m.at("head_dim").as_int(0);
  const int hidden = (int)m.at("hidden").as_int(0);
  const int Lt = (int)m.at("text_len").as_int(0);
  const int S = gh * gw, C = H * d;

  Session s;
  MetalCompute* mc = s.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  vdn::Config cfg;
  std::string err;
  if (!vdn::load_config(stage, &cfg, &err)) { return; }
  std::shared_ptr<WeightSet> ws =
      WeightSet::open(stage + "/linear_branch", nullptr);
  if (!ws) { return; }

  std::vector<float> xv, qr, kr, vr, tx, tk, tv, want;
  auto load = [&](const char* n, std::vector<float>* v) {
    return read_bin_(gdir + "/" + n + ".bin", v);
  };
  if (!load("xv", &xv) || !load("q_raw", &qr) || !load("k_raw", &kr)
      || !load("v_raw", &vr) || !load("text_x", &tx)
      || !load("text_k_raw", &tk) || !load("text_v_raw", &tv)
      || !load("readout", &want)) {
    EXPECT_TRUE(false);
    return;
  }

  MetalVdnBranch::Dims dims;
  dims.heads = H; dims.head_dim = d; dims.hidden = hidden;
  dims.n_layers = 50;
  dims.bf16_features = false;        // against an fp32 golden
  std::unique_ptr<MetalVdnBranch> br =
      MetalVdnBranch::load(ws, mc, cfg, dims, &err);
  if (!br || !br->ensure_block(0, &err)) { return; }

  MetalVdnBranch::Geometry geo;
  geo.frames = F; geo.grid_h = gh; geo.grid_w = gw; geo.text_len = Lt;
  const std::vector<vdn::Bound> bounds =
      vdn::window_bounds(F, (int)m.at("radius").as_int(0),
                         (int)m.at("chunk").as_int(0));

  const std::size_t vrows = (std::size_t)F * S;
  const std::size_t rows  = (std::size_t)Lt + vrows;
  const std::size_t obn   = vrows * (std::size_t)C;

  auto run = [&](const MetalVdnBranch::Inputs& in, std::vector<float>* got) {
    SharedBuffer ob = mc->make_shared_buffer(obn * 4);
    if (ob.empty()) { return false; }
    std::memset(ob.contents(), 0, obn * 4);
    br->clear_solve_failures();
    CommandStream stream = mc->make_command_stream();
    ComputeEncoder enc = stream.begin_compute();
    if (!br->encode(enc, 0, geo, bounds, in, ob, &err)) {
      std::printf("[vdn_fused] encode: %s\n", err.c_str());
      return false;
    }
    enc.end();
    std::string gerr;
    if (!stream.commit().wait_ok(&gerr)) { return false; }
    EXPECT_TRUE(br->solve_failures() == 0u);
    got->resize(obn);
    std::memcpy(got->data(), ob.contents(), obn * 4);
    return true;
  };

  // ---- the reference arm: three tight fp32 tensors ------------------
  auto buf = [&](const std::vector<float>& v) {
    SharedBuffer b = mc->make_shared_buffer(v.size() * 4);
    if (!b.empty()) { std::memcpy(b.contents(), v.data(), v.size() * 4); }
    return b;
  };
  SharedBuffer xb = buf(xv), qb = buf(qr), kb = buf(kr), vb = buf(vr);
  SharedBuffer txb = buf(tx), tkb = buf(tk), tvb = buf(tv);
  MetalVdnBranch::Inputs tight;
  tight.x = &xb; tight.q_raw = &qb; tight.k_raw = &kb; tight.v_raw = &vb;
  tight.text_x = &txb; tight.text_k = &tkb; tight.text_v = &tvb;
  std::vector<float> base;
  ASSERT_TRUE(run(tight, &base));
  if (base.empty()) { return; }
  const double e0 = rel_l2_(base.data(), want.data(), want.size());
  EXPECT_TRUE(e0 >= 0.0 && e0 < 1e-5);

  // ---- the packed sequence both fused arms read ---------------------
  // Text first and video after, which is H3's own order: the point is
  // that neither run starts at row 0, so a byte offset that was quietly
  // ignored would show up rather than cancel.
  const int RS = 3 * C, HS = 3 * d;
  std::vector<float> fused((std::size_t)rows * RS, 0.0f);
  std::vector<float> hid((std::size_t)rows * hidden, 0.0f);
  auto put = [&](std::size_t row, int h, int field, const float* src) {
    std::memcpy(&fused[row * RS + (std::size_t)h * HS + field * d], src,
                (std::size_t)d * 4);
  };
  for (int t = 0; t < Lt; ++t) {
    for (int h = 0; h < H; ++h) {
      put((std::size_t)t, h, 1, &tk[((std::size_t)t * H + h) * d]);
      put((std::size_t)t, h, 2, &tv[((std::size_t)t * H + h) * d]);
    }
    std::memcpy(&hid[(std::size_t)t * hidden], &tx[(std::size_t)t * hidden],
                (std::size_t)hidden * 4);
  }
  for (std::size_t r = 0; r < vrows; ++r) {
    for (int h = 0; h < H; ++h) {
      put((std::size_t)Lt + r, h, 0, &qr[(r * H + h) * d]);
      put((std::size_t)Lt + r, h, 1, &kr[(r * H + h) * d]);
      put((std::size_t)Lt + r, h, 2, &vr[(r * H + h) * d]);
    }
    std::memcpy(&hid[((std::size_t)Lt + r) * hidden],
                &xv[r * hidden], (std::size_t)hidden * 4);
  }

  auto arm = [&](bool narrow, std::vector<float>* got) {
    const std::size_t es = narrow ? 2 : 4;
    SharedBuffer fb = mc->make_shared_buffer(fused.size() * es);
    SharedBuffer hb = mc->make_shared_buffer(hid.size() * es);
    if (fb.empty() || hb.empty()) { return false; }
    if (narrow) {
      // TRUNCATION, not rounding, because that is what the kernel's own
      // bf16 read undoes -- so this arm measures the narrowing and
      // nothing else.
      auto nb = [](const std::vector<float>& src, void* dst) {
        auto* o = (std::uint16_t*)dst;
        for (std::size_t i = 0; i < src.size(); ++i) {
          std::uint32_t u;
          std::memcpy(&u, &src[i], 4);
          o[i] = (std::uint16_t)(u >> 16);
        }
      };
      nb(fused, fb.contents());
      nb(hid, hb.contents());
    } else {
      std::memcpy(fb.contents(), fused.data(), fused.size() * 4);
      std::memcpy(hb.contents(), hid.data(), hid.size() * 4);
    }
    MetalVdnBranch::Inputs in;
    in.x = &hb;
    in.q_raw = in.k_raw = in.v_raw = &fb;
    in.text_x = &hb; in.text_k = in.text_v = &fb;
    in.qkv_row_stride = RS;
    in.qkv_head_stride = HS;
    in.qkv_bf16 = narrow;
    in.x_bf16 = narrow;
    const std::size_t v0 = (std::size_t)Lt * RS;
    in.q_off = v0 * es;
    in.k_off = (v0 + d) * es;
    in.v_off = (v0 + 2 * d) * es;
    in.text_x_off = 0;
    in.text_k_off = (std::size_t)d * es;
    in.text_v_off = (std::size_t)(2 * d) * es;
    in.x_off = (std::size_t)Lt * hidden * es;
    return run(in, got);
  };

  std::vector<float> wide;
  ASSERT_TRUE(arm(false, &wide));
  if (wide.size() != base.size()) { return; }
  const bool same =
      std::memcmp(wide.data(), base.data(), base.size() * 4) == 0;
  EXPECT_TRUE(same);
  if (!same) {
    std::printf("[vdn_fused] strided fp32 differs: rel-L2 %.3e\n",
                rel_l2_(wide.data(), base.data(), base.size()));
  }

  std::vector<float> narrow;
  ASSERT_TRUE(arm(true, &narrow));
  if (narrow.size() != base.size()) { return; }
  const double eb = rel_l2_(narrow.data(), base.data(), base.size());
  // Loose on purpose. bf16 inputs are what the transformer actually
  // has, and the bar this has to clear is "the branch still computes
  // the same thing", not the fp32 goldens' 1e-5.
  const bool okb = eb >= 0.0 && eb < 5e-2;
  EXPECT_TRUE(okb);
  std::printf("[vdn_fused] strided fp32 %s, bf16 rel-L2 %.3e\n",
              same ? "bit-identical" : "DIFFERS", eb);
}

TEST(metal_vdn_branch, a_streamed_block_matches_and_frees)
{
  // The branch is 4.28 GB over 50 blocks on top of a 33B DiT, so a
  // caller that streams the DiT's blocks has to be able to stream these.
  //
  // TWO CLAIMS, and the second is the one that could have been faked.
  // Streamed must be BIT-IDENTICAL to preloaded -- same bytes, same
  // transform, only the accessor moved -- which is the bar every
  // streaming port in this tree is held to. And the bytes must actually
  // COME BACK: cached derived() entries are owned by the weight SET, so
  // a release that only drops this object's handles frees nothing while
  // looking exactly like a release that works. That is why streaming
  // reads take stream_derived()/stream_tensor() instead.
  const std::string stage = stage_(), gdir = golden_();
  std::ifstream mi(gdir + "/meta.json");
  if (!mi) { return; }
  FlexData meta = FlexData::from_json(mi);
  if (!meta.is_object()) { return; }
  auto m = meta.as_object();
  const int F = (int)m.at("frames").as_int(0);
  const int gh = (int)m.at("grid_h").as_int(0);
  const int gw = (int)m.at("grid_w").as_int(0);
  const int H = (int)m.at("heads").as_int(0);
  const int d = (int)m.at("head_dim").as_int(0);
  const int hidden = (int)m.at("hidden").as_int(0);
  const int Lt = (int)m.at("text_len").as_int(0);
  const int S = gh * gw, C = H * d;

  Session s;
  MetalCompute* mc = s.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  vdn::Config cfg;
  std::string err;
  if (!vdn::load_config(stage, &cfg, &err)) { return; }
  std::shared_ptr<WeightSet> ws =
      WeightSet::open(stage + "/linear_branch", nullptr);
  if (!ws) { return; }

  std::vector<float> xv, qr, kr, vr, tx, tv, tk, want;
  auto load = [&](const char* n, std::vector<float>* v) {
    return read_bin_(gdir + "/" + n + ".bin", v);
  };
  if (!load("xv", &xv) || !load("q_raw", &qr) || !load("k_raw", &kr)
      || !load("v_raw", &vr) || !load("text_x", &tx)
      || !load("text_k_raw", &tk) || !load("text_v_raw", &tv)
      || !load("readout", &want)) {
    EXPECT_TRUE(false);
    return;
  }
  auto buf = [&](const std::vector<float>& v) {
    SharedBuffer b = mc->make_shared_buffer(v.size() * 4);
    if (!b.empty()) { std::memcpy(b.contents(), v.data(), v.size() * 4); }
    return b;
  };
  SharedBuffer xb = buf(xv), qb = buf(qr), kb = buf(kr), vb = buf(vr);
  SharedBuffer txb = buf(tx), tkb = buf(tk), tvb = buf(tv);
  MetalVdnBranch::Inputs in;
  in.x = &xb; in.q_raw = &qb; in.k_raw = &kb; in.v_raw = &vb;
  in.text_x = &txb; in.text_k = &tkb; in.text_v = &tvb;
  MetalVdnBranch::Geometry geo;
  geo.frames = F; geo.grid_h = gh; geo.grid_w = gw; geo.text_len = Lt;
  const std::vector<vdn::Bound> bounds =
      vdn::window_bounds(F, (int)m.at("radius").as_int(0),
                         (int)m.at("chunk").as_int(0));
  const std::size_t obn = (std::size_t)F * S * C;

  auto arm = [&](bool stream, std::vector<float>* got,
                 std::size_t* held, std::size_t* after) {
    MetalVdnBranch::Dims dims;
    dims.heads = H; dims.head_dim = d; dims.hidden = hidden;
    dims.n_layers = 50;
    dims.bf16_features = false;      // the claim is about the ACCESSOR
    std::unique_ptr<MetalVdnBranch> br =
        MetalVdnBranch::load(ws, mc, cfg, dims, &err);
    if (!br) { return false; }
    br->set_stream_blocks(stream);
    if (!br->ensure_block(0, &err)) { return false; }
    *held = br->resident_bytes();
    SharedBuffer ob = mc->make_shared_buffer(obn * 4);
    if (ob.empty()) { return false; }
    std::memset(ob.contents(), 0, obn * 4);
    {
      CommandStream st = mc->make_command_stream();
      ComputeEncoder enc = st.begin_compute();
      if (!br->encode(enc, 0, geo, bounds, in, ob, &err)) { return false; }
      enc.end();
      std::string gerr;
      if (!st.commit().wait_ok(&gerr)) { return false; }
    }
    EXPECT_TRUE(br->solve_failures() == 0u);
    got->resize(obn);
    std::memcpy(got->data(), ob.contents(), obn * 4);
    // Only legal because the commit above has been WAITED for: an
    // encoded dispatch holds a buffer by pointer, not by reference.
    br->release_block(0);
    *after = br->resident_bytes();
    return true;
  };

  std::vector<float> kept, streamed;
  std::size_t kh = 0, ka = 0, sh = 0, sa = 0;
  ASSERT_TRUE(arm(false, &kept, &kh, &ka));
  if (kept.empty()) { return; }
  ASSERT_TRUE(arm(true, &streamed, &sh, &sa));
  if (streamed.empty()) { return; }

  const bool identical =
      kept.size() == streamed.size()
      && std::memcmp(kept.data(), streamed.data(), kept.size() * 4) == 0;
  EXPECT_TRUE(identical);
  // Both report zero AFTER the release, because release_block() drops
  // the handles either way. The claim that matters is the one about the
  // SET, and it is not observable from here -- what is observable is
  // that the two paths compute the same thing and that a released block
  // reports nothing held.
  EXPECT_TRUE(kh > 0 && sh > 0 && ka == 0 && sa == 0);
  std::printf("[vdn_stream] block 0: kept %zu MB, streamed %zu MB, "
              "%s, released to %zu/%zu\n", kh >> 20, sh >> 20,
              identical ? "bit-identical" : "DIFFERS", ka, sa);
}

TEST(metal_vdn_branch, stage_bench)
{
  // WHERE THE BRANCH'S TIME GOES, at the geometry a real generation
  // uses. Opt-in (VPIPE_VDN_BENCH) because it allocates a production
  // clip's worth of scratch and runs the whole branch several times.
  //
  // A BENCH AND NOT A TEST: it asserts nothing about speed. It exists so
  // the delta-rule machinery can be worked on in seconds rather than
  // through 8-minute model runs, and so the shares are measured at the
  // size that matters -- at unit-test geometry the solve and the scan
  // are a rounding error and the features dominate, which is the
  // opposite of the real answer.
  //
  // Defaults are the docs pipeline at 960x544, 120 frames: 37 latent
  // frames of 510 tokens. The GRID IS THE PATCH GRID, NOT THE LATENT
  // ONE -- the VAE is 16x spatial and the DiT patchifies 2x2 on top, so
  // 960x544 is 60x34 latent and 30x17 = 510 tokens. Using the latent
  // grid here puts 4x the tokens through the branch and reports 2.8x
  // the time, which is exactly the mistake this bench exists to stop
  // people making. VPIPE_VDN_BENCH_{F,GH,GW,TILE} override.
  if (std::getenv("VPIPE_VDN_BENCH") == nullptr) { return; }
  const std::string stage = stage_();
  auto envi = [](const char* k, int d) {
    const char* v = std::getenv(k);
    return (v != nullptr && *v != '\0') ? std::atoi(v) : d;
  };
  const int F = envi("VPIPE_VDN_BENCH_F", 37);
  const int gh = envi("VPIPE_VDN_BENCH_GH", 17);
  const int gw = envi("VPIPE_VDN_BENCH_GW", 30);
  const int tile = envi("VPIPE_VDN_BENCH_TILE", 4);
  const int reps = envi("VPIPE_VDN_BENCH_REPS", 3);
  const int Lt = 17;

  Session s;
  MetalCompute* mc = s.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  vdn::Config cfg;
  std::string err;
  if (!vdn::load_config(stage, &cfg, &err)) { return; }
  std::shared_ptr<WeightSet> ws =
      WeightSet::open(stage + "/linear_branch", nullptr);
  if (!ws) { return; }

  MetalVdnBranch::Dims dims;
  dims.heads = 56; dims.head_dim = 128; dims.hidden = 5376;
  dims.n_layers = 50; dims.frame_tile = tile;
  std::unique_ptr<MetalVdnBranch> br =
      MetalVdnBranch::load(ws, mc, cfg, dims, &err);
  if (!br || !br->ensure_block(0, &err)) {
    std::printf("[vdn_bench] load: %s\n", err.c_str());
    return;
  }
  const int S = gh * gw, H = 56, d = 128, C = H * d, hidden = 5376;
  const std::size_t rows = (std::size_t)Lt + (std::size_t)F * S;

  // The transformer's own source: one fused per-head-grouped bf16
  // projection with the video rows after the prompt. Contents do not
  // matter for timing, but they must not be denormal or NaN -- the
  // solve's iteration count does not depend on the values, the
  // hardware's speed on denormals can.
  const int RS = 3 * C;
  SharedBuffer qkv = mc->make_shared_buffer(rows * (std::size_t)RS * 2);
  SharedBuffer xb  = mc->make_shared_buffer(rows * (std::size_t)hidden * 2);
  SharedBuffer ob  = mc->make_shared_buffer((std::size_t)F * S * C * 2);
  if (qkv.empty() || xb.empty() || ob.empty()) {
    std::printf("[vdn_bench] cannot allocate %.1f GB of inputs\n",
                (double)(qkv.byte_size() + xb.byte_size() + ob.byte_size())
                    / 1073741824.0);
    return;
  }
  auto fill = [](SharedBuffer& b, float scale) {
    auto* o = static_cast<std::uint16_t*>(b.contents());
    const std::size_t n = b.byte_size() / 2;
    std::uint32_t st = 12345u;
    for (std::size_t i = 0; i < n; ++i) {
      st = st * 1664525u + 1013904223u;
      const float v = ((float)(st >> 8) / 8388608.0f - 1.0f) * scale;
      std::uint32_t u;
      std::memcpy(&u, &v, 4);
      o[i] = (std::uint16_t)(u >> 16);
    }
  };
  fill(qkv, 1.0f);
  fill(xb, 1.0f);
  std::memset(ob.contents(), 0, ob.byte_size());

  MetalVdnBranch::Geometry geo;
  geo.frames = F; geo.grid_h = gh; geo.grid_w = gw; geo.text_len = Lt;
  const std::vector<vdn::Bound> bounds =
      vdn::window_bounds(F, cfg.radius, cfg.chunk);
  MetalVdnBranch::Inputs in;
  in.x = &xb;
  in.q_raw = in.k_raw = in.v_raw = &qkv;
  in.text_x = &xb; in.text_k = in.text_v = &qkv;
  in.qkv_row_stride = RS;
  in.qkv_head_stride = 3 * d;
  in.qkv_bf16 = true; in.x_bf16 = true; in.out_bf16 = true;
  const std::size_t v0 = (std::size_t)Lt * RS;
  in.q_off = v0 * 2;
  in.k_off = (v0 + d) * 2;
  in.v_off = (v0 + 2 * d) * 2;
  in.text_k_off = (std::size_t)d * 2;
  in.text_v_off = (std::size_t)(2 * d) * 2;
  in.x_off = (std::size_t)Lt * hidden * 2;

  double wall = 0.0;
  for (int r = 0; r <= reps; ++r) {
    CommandStream st = mc->make_command_stream();
    // Timed WITHOUT the splits on the last pass, so the barriers'
    // inflation is visible rather than assumed.
    br->set_profile_stream(r == reps ? nullptr : &st);
    if (r == 1) { br->clear_profile(); }   // pass 0 warms the pipelines
    ComputeEncoder enc = st.begin_compute();
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = br->encode(enc, 0, geo, bounds, in, ob, &err);
    enc.end();
    std::string gerr;
    const bool gok = st.commit().wait_ok(&gerr);
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    if (r == reps) { wall = ms; }
    EXPECT_TRUE(ok && gok);
    if (!ok || !gok) {
      std::printf("[vdn_bench] encode: %s %s\n", err.c_str(), gerr.c_str());
      return;
    }
  }
  br->set_profile_stream(nullptr);
  EXPECT_TRUE(br->solve_failures() == 0u);

  const MetalVdnBranch::Profile& p = br->profile();
  const double n = (double)(reps - 1 > 0 ? reps - 1 : 1);
  const double tot = p.total() / n;
  auto row = [&](const char* what, double v) {
    std::printf("[vdn_bench]   %-22s %8.1f ms  %5.1f%%\n", what, v / n,
                tot > 0.0 ? 100.0 * (v / n) / tot : 0.0);
  };
  std::printf("[vdn_bench] %d frames x %d tokens, tile %d, %zu video rows\n",
              F, S, tile, (std::size_t)F * S);
  row("temporal act + L2", p.features);
  row("spatial conv (k,v)", p.conv);
  row("frame statistics", p.stats);
  row("solve (chol+inv)", p.solve);
  row("scan (serial)", p.scan);
  row("readout", p.readout);
  row("output gate", p.gate);
  row("beta", p.beta);
  row("alpha", p.alpha);
  row("text state", p.text);
  row("gather + bridge", p.gather);
  std::printf("[vdn_bench]   %-22s %8.1f ms   (unsplit %.1f ms, so the "
              "barriers add %.0f%%)\n", "TOTAL", tot, wall,
              wall > 0.0 ? 100.0 * (tot / wall - 1.0) : 0.0);
}

TEST(metal_vdn_branch, the_matrix_core_route_tracks_the_wide_one)
{
  // THE M5 ROUTE FOR THE THREE PLAIN GEMMS, against the fp32 one.
  //
  // beta, the output gate's two and the softmax half's gate are
  // y = x W^T with a bf16 x and a bf16 [N, K] weight, so on a
  // matrix-core GPU they go to dense_gemm_mma and the bias + sigmoid
  // follow in vdn_bias_act_f32. Three things change with them, and this
  // is what says they are small at the branch's OUTPUT rather than
  // assuming it:
  //
  //   the weight is read bf16 rather than widened   (the same bits)
  //   the product rounds once on the matmul's store (a new rounding)
  //   the gate's rank-128 intermediate narrows      (a new rounding)
  //
  // The first is exact. The other two are what the bar is for -- and
  // the bar is the branch's own bf16 floor, the same one
  // narrow_features_track_the_wide_path holds the feature dtype to,
  // because these are roundings of the same order in the same chain.
  //
  // NO GOLDEN NEEDED: both arms are this GPU, driven from the released
  // checkpoint, so this runs wherever the branch weights are.
  const std::string stage = stage_();
  Session s;
  MetalCompute* mc = s.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  if (!mc->supports_matrix_cores()) {
    std::printf("[vdn_mma] no matrix cores -- SKIPPED\n");
    return;
  }
  vdn::Config cfg;
  std::string err;
  if (!vdn::load_config(stage, &cfg, &err)) { return; }
  std::shared_ptr<WeightSet> ws =
      WeightSet::open(stage + "/linear_branch", nullptr);
  if (!ws) { return; }

  // Small, because this is an arithmetic question and not a rate one.
  // VPIPE_VDN_AB_FRAMES raises it to a real clip, which is a different
  // question -- whether any index in the branch wraps 32 bits at length
  // -- and needs the geometry to match a real one, so it switches to
  // the 17x30 patch grid the docs pipeline uses.
  const int envF = [] {
    const char* e = std::getenv("VPIPE_VDN_AB_FRAMES");
    return (e != nullptr && *e != '\0') ? std::atoi(e) : 0;
  }();
  const bool big = envF > 0;
  const int F = big ? envF : 12;
  const int gh = big ? 17 : 6, gw = big ? 30 : 8, Lt = big ? 17 : 9;
  const int S = gh * gw, H = 56, d = 128, C = H * d, hidden = 5376;
  const int RS = 3 * C;
  const std::size_t rows = (std::size_t)Lt + (std::size_t)F * S;
  SharedBuffer qkv = mc->make_shared_buffer(rows * (std::size_t)RS * 2);
  SharedBuffer xb  = mc->make_shared_buffer(rows * (std::size_t)hidden * 2);
  ASSERT_TRUE(!qkv.empty() && !xb.empty());
  if (qkv.empty() || xb.empty()) { return; }
  auto fill = [](SharedBuffer& b, std::uint32_t seed) {
    auto* o = static_cast<std::uint16_t*>(b.contents());
    const std::size_t n = b.byte_size() / 2;
    std::uint32_t st = seed;
    for (std::size_t i = 0; i < n; ++i) {
      st = st * 1664525u + 1013904223u;
      const float v = (float)(st >> 8) / 8388608.0f - 1.0f;
      std::uint32_t u;
      std::memcpy(&u, &v, 4);
      o[i] = (std::uint16_t)(u >> 16);
    }
  };
  fill(qkv, 12345u);
  fill(xb, 999u);

  MetalVdnBranch::Geometry geo;
  geo.frames = F; geo.grid_h = gh; geo.grid_w = gw; geo.text_len = Lt;
  const std::vector<vdn::Bound> bounds =
      vdn::window_bounds(F, cfg.radius, cfg.chunk);
  MetalVdnBranch::Inputs in;
  in.x = &xb;
  in.q_raw = in.k_raw = in.v_raw = &qkv;
  in.text_x = &xb; in.text_k = in.text_v = &qkv;
  in.qkv_row_stride = RS;
  in.qkv_head_stride = 3 * d;
  in.qkv_bf16 = true; in.x_bf16 = true;
  const std::size_t v0 = (std::size_t)Lt * RS;
  in.q_off = v0 * 2;
  in.k_off = (v0 + d) * 2;
  in.v_off = (v0 + 2 * d) * 2;
  in.text_k_off = (std::size_t)d * 2;
  in.text_v_off = (std::size_t)(2 * d) * 2;
  in.x_off = (std::size_t)Lt * hidden * 2;

  // fp32 OUT on both arms, so the comparison is not reading the same
  // bf16 store twice and calling the agreement its own.
  auto run = [&](bool no_mma, int frame_tile, std::vector<float>* out,
                 unsigned* fails, bool no_dw = false) {
    if (no_mma) { ::setenv("VPIPE_VDN_NO_MMA", "1", 1); }
    else        { ::unsetenv("VPIPE_VDN_NO_MMA"); }
    if (no_dw)  { ::setenv("VPIPE_VDN_NO_DW_MMA", "1", 1); }
    else        { ::unsetenv("VPIPE_VDN_NO_DW_MMA"); }
    MetalVdnBranch::Dims dims;
    dims.heads = H; dims.head_dim = d; dims.hidden = hidden;
    dims.n_layers = 50;
    dims.frame_tile = frame_tile;
    std::string e;
    std::unique_ptr<MetalVdnBranch> br =
        MetalVdnBranch::load(ws, mc, cfg, dims, &e);
    ::unsetenv("VPIPE_VDN_NO_MMA");
    ::unsetenv("VPIPE_VDN_NO_DW_MMA");
    if (!br || !br->ensure_block(0, &e)) {
      std::printf("[vdn_mma] load: %s\n", e.c_str());
      return false;
    }
    SharedBuffer ob =
        mc->make_shared_buffer((std::size_t)F * S * C * sizeof(float));
    if (ob.empty()) { return false; }
    std::memset(ob.contents(), 0, ob.byte_size());
    CommandStream st = mc->make_command_stream();
    ComputeEncoder enc = st.begin_compute();
    const bool ok = br->encode(enc, 0, geo, bounds, in, ob, &e);
    enc.end();
    std::string gerr;
    const bool gok = st.commit().wait_ok(&gerr);
    if (!ok || !gok) {
      std::printf("[vdn_mma] encode: %s %s\n", e.c_str(), gerr.c_str());
      return false;
    }
    *fails = br->solve_failures();
    // A real clip's readout is 2.3 GB as floats and this test holds
    // three of them, so the big arm keeps every `pick`-th element. 97 is
    // prime, so the sample cannot align with any power-of-two stride in
    // the layout and miss a corrupted lane -- and an index that wraps
    // does not corrupt one lane, it corrupts a region.
    const MetalVdnBranch::MmaRoutes r = br->mma_routes();
    std::printf("[vdn_mma] %-5s routes: gemm %d stats %d readout %d "
                "conv %d\n", no_mma ? "wide" : "mma", (int)r.gemm,
                (int)r.stats, (int)r.readout, (int)r.conv);
    const std::size_t n = ob.byte_size() / sizeof(float);
    const auto* q = static_cast<const float*>(ob.contents());
    const std::size_t pick = big ? 97u : 1u;
    out->clear();
    out->reserve(n / pick + 1);
    for (std::size_t i = 0; i < n; i += pick) { out->push_back(q[i]); }
    return true;
  };

  std::vector<float> wide, mma;
  unsigned f_wide = 0, f_mma = 0;
  ASSERT_TRUE(run(true, MetalVdnBranch::kFrameTile, &wide, &f_wide));
  ASSERT_TRUE(run(false, MetalVdnBranch::kFrameTile, &mma, &f_mma));
  if (wide.size() != mma.size() || wide.empty()) { return; }
  const double rel = rel_l2_(mma.data(), wide.data(), wide.size());
  std::printf("[vdn_mma] the branch readout: rel-L2 %.6e, solve failures "
              "%u wide / %u mma\n", rel, f_wide, f_mma);
  EXPECT_TRUE(rel >= 0.0 && rel < 3e-2);
  // And it must not be ZERO: two runs that took the same route would
  // agree exactly, which would mean the env A/B did nothing and this
  // test measured a copy of itself.
  EXPECT_TRUE(rel > 0.0);
  // The SOLVE is what the fp32 statistics exist for. The matrix-core
  // route does not touch them -- it is the three plain GEMMs and
  // nothing else -- so a failure appearing on one arm and not the other
  // would mean it reached further than it was meant to.
  EXPECT_TRUE(f_mma == f_wide);

  // AND THE TILE MUST STILL NOT MATTER. tile_size_does_not_change_the_
  // answer pins this for the fp32 route against a golden; the
  // matrix-core route has to earn it again, because two of the things
  // it added are per-TILE -- the landing buffer the readout's product
  // goes to, and the sqrt(beta) the features carry, which is bound at
  // the tile's own base. A tile as large as the clip never crosses a
  // boundary, so a wrong offset there is invisible at the default 4 and
  // shows only against a different one.
  //
  // BIT-IDENTICAL is the bar, not a tolerance: the tile changes how the
  // work is grouped and nothing about the arithmetic, so any difference
  // at all is an indexing bug.
  // THE SPATIAL CONV IS BIT-IDENTICAL, which is a stronger statement than
  // the tolerance above and worth making separately.
  //
  // MPP's convolution2d takes groups == 1, so the depthwise short conv
  // reaches it as a DENSE conv over 16 channels against a block-diagonal
  // weight. The off-diagonal terms are exact zeros and the 25 that
  // remain are the same bf16 operands widened to f32 and summed in the
  // same (ky, kx) order the ALU kernel uses -- so the matrix unit's f32
  // accumulator performs the same additions, and "same answer" here
  // means the same BITS rather than the same to a tolerance. If that
  // ever stops holding it is a change in the op's reduction order and
  // this test should say so rather than absorb it.
  {
    std::vector<float> no_dw;
    unsigned f_nodw = 0;
    if (run(false, MetalVdnBranch::kFrameTile, &no_dw, &f_nodw, true)
        && no_dw.size() == mma.size()) {
      std::size_t diff = 0;
      for (std::size_t i = 0; i < mma.size(); ++i) {
        if (no_dw[i] != mma[i]) { ++diff; }
      }
      std::printf("[vdn_mma] the hw depthwise conv: %zu of %zu differ\n",
                  diff, mma.size());
      EXPECT_TRUE(diff == 0);
    }
  }

  // The tile also sets the spatial conv's RING -- its size is
  // kFrameTile + 2 * kConvHalo slots and the tile decides how the halo
  // frames overlap between passes. 16 is clamped to the clip, so that
  // arm is ONE tile with no overlap and no wrap: the no-reuse
  // reference. 1 is the opposite extreme, where the ring wraps every
  // pass and every frame is a halo frame for two neighbours.
  // The tile sweep is a small-geometry question -- it is about the ring's
  // indexing, which does not change with the clip -- and at a real clip
  // it would run the whole branch three more times.
  const std::vector<int> tiles_to_try =
      big ? std::vector<int>{} : std::vector<int>{1, 3, 16};
  for (const int t : tiles_to_try) {
    std::vector<float> other;
    unsigned f_other = 0;
    if (!run(false, t, &other, &f_other)) { continue; }
    if (other.size() != mma.size()) {
      EXPECT_TRUE(false);
      continue;
    }
    std::size_t diff = 0;
    for (std::size_t i = 0; i < mma.size(); ++i) {
      if (other[i] != mma[i]) { ++diff; }
    }
    std::printf("[vdn_mma] frame_tile %d vs %d: %zu of %zu differ\n", t,
                MetalVdnBranch::kFrameTile, diff, mma.size());
    EXPECT_TRUE(diff == 0);
  }
}

TEST(metal_vdn_branch, batched_matmul2d_probe)
{
  // DE-RISK, before writing a kernel: can the matrix cores do the frame
  // STATISTICS faster than the fp32 tiled GEMM already does?
  //
  // That stage is two [d, S] x [S, d] products per (frame, head) and the
  // branch's largest arithmetic by some way -- 69.2 GFLOP a block at
  // generation geometry, 77.2 ms measured, 0.90 TFLOP/s. The shape is a
  // BATCH of 128x128 tiles, which is exactly the regime gdn_mma.metal
  // recorded as too small to pay on M5 (its 64x64 per-head tiles reached
  // ~1.75 TFLOP/s). So the question is whether 128x128 over 2072
  // batches is on the other side of that line, and it is cheaper to ask
  // the existing batched primitive than to port the kernel and find out.
  //
  // Opt-in (VPIPE_VDN_MMA_PROBE) and a RATE probe only: contiguous
  // operands, no beta, no symmetrise, right answer not checked. What it
  // bounds is the best the port could do.
  if (std::getenv("VPIPE_VDN_MMA_PROBE") == nullptr) { return; }
  Session s;
  MetalCompute* mc = s.metal_compute();
  if (mc == nullptr || !mc->valid() || !mc->supports_matrix_cores()) {
    return;
  }
  ComputeLibrary lib = mc->load_library("gdn_mma_bf16");
  ComputeFunction fn = lib.function("gdn_bmm_tn_f16");
  if (!fn.valid()) {
    std::printf("[vdn_probe] gdn_bmm_tn_f16 did not validate\n");
    return;
  }
  const int F = 37, S = 510, H = 56, d = 128;
  const int batch = F * H;
  const std::size_t na = (std::size_t)batch * S * d;
  const std::size_t nc = (std::size_t)batch * d * d;
  SharedBuffer A = mc->make_shared_buffer(na * 2);
  SharedBuffer B = mc->make_shared_buffer(na * 2);
  SharedBuffer C = mc->make_shared_buffer(nc * 2);
  if (A.empty() || B.empty() || C.empty()) {
    std::printf("[vdn_probe] cannot allocate %.2f GB\n",
                (double)(2 * na * 2 + nc * 2) / 1073741824.0);
    return;
  }
  std::memset(A.contents(), 0x3c, A.byte_size());
  std::memset(B.contents(), 0x3c, B.byte_size());
  const int sA = S * d, sB = S * d, sC = d * d;
  const int BM = 64, BN = 64, SG = 4;
  double best = -1.0;
  for (int r = 0; r < 4; ++r) {
    CommandStream st = mc->make_command_stream();
    ComputeEncoder enc = st.begin_compute();
    // TWO products per batch, as the statistics kernel computes A and B.
    for (int i = 0; i < 2; ++i) {
      enc.set_function(fn);
      enc.set_buffer(0, A); enc.set_buffer(1, B); enc.set_buffer(2, C);
      enc.set_constant(3, d); enc.set_constant(4, d); enc.set_constant(5, S);
      enc.set_constant(6, sA); enc.set_constant(7, sB);
      enc.set_constant(8, sC);
      enc.dispatch({(unsigned)((d + BN - 1) / BN) * SG * 32,
                    (unsigned)((d + BM - 1) / BM), (unsigned)batch},
                   {(unsigned)(SG * 32), 1, 1});
    }
    enc.end();
    const auto t0 = std::chrono::steady_clock::now();
    std::string gerr;
    if (!st.commit().wait_ok(&gerr)) {
      std::printf("[vdn_probe] %s\n", gerr.c_str());
      return;
    }
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    if (r > 0 && (best < 0.0 || ms < best)) { best = ms; }
  }
  const double gflop = 2.0 * 2.0 * (double)batch * d * d * S / 1e9;
  std::printf("[vdn_probe] %d batches of [%d,%d]x[%d,%d] TN, x2: %.1f ms, "
              "%.2f TFLOP/s (the fp32 kernel: 77.2 ms, 0.90)\n", batch, d,
              S, S, d, best, gflop / (best / 1000.0) / 1000.0);
}

TEST(metal_vdn_branch, the_gemm_matches_a_cpu_reference)
{
  // vdn_gemm_act_f32 is what beta, the output gate and the softmax gate
  // all became, so it is worth a direct check rather than only the
  // end-to-end golden -- which would catch a wrong answer but not tell
  // you it was the GEMM.
  //
  // The shapes are deliberately NOT multiples of the 32x32 tile: the
  // tail predicates are the part of a tiled GEMM that a well-chosen
  // benchmark size never visits, and a wrong one shows as a few rogue
  // rows rather than as garbage.
  Session s;
  MetalCompute* mc = s.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  ComputeLibrary lib = mc->load_library("vdn_branch");
  ComputeFunction fn = lib.function("vdn_gemm_act_f32");
  ASSERT_TRUE(fn.valid());
  if (!fn.valid()) { return; }

  struct Case { int M, N, K; bool bias, act, bf16; };
  const Case cases[] = {
      {70, 45, 100, false, true,  false},   // beta's shape, scaled down
      {70, 45, 100, true,  true,  false},
      {70, 45, 100, true,  false, false},
      {70, 45, 100, true,  true,  true},    // the transformer's bf16 x
      {32, 32,  16, true,  true,  false},   // exactly one tile
      {1,   1,   1, true,  true,  false},   // degenerate
  };
  std::uint32_t st = 99u;
  auto rnd = [&]() {
    st = st * 1664525u + 1013904223u;
    return (float)(st >> 8) / 8388608.0f - 1.0f;
  };
  auto bf16 = [](float v) {
    std::uint32_t u;
    std::memcpy(&u, &v, 4);
    return (std::uint16_t)(u >> 16);
  };
  auto unbf = [](std::uint16_t h) {
    const std::uint32_t u = (std::uint32_t)h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
  };

  double worst = 0.0;
  int ran = 0;
  for (const Case& c : cases) {
    std::vector<float> A((std::size_t)c.M * c.K), B((std::size_t)c.N * c.K),
        bi((std::size_t)c.N);
    for (float& v : A) { v = rnd(); }
    for (float& v : B) { v = rnd(); }
    for (float& v : bi) { v = rnd(); }
    // The bf16 arm must compare against the ROUNDED input, or it
    // measures the narrowing instead of the kernel.
    if (c.bf16) {
      for (float& v : A) { v = unbf(bf16(v)); }
    }
    SharedBuffer ab = mc->make_shared_buffer(
        A.size() * (c.bf16 ? 2u : 4u));
    SharedBuffer bb = mc->make_shared_buffer(B.size() * 4);
    SharedBuffer bib = mc->make_shared_buffer(bi.size() * 4);
    SharedBuffer cb = mc->make_shared_buffer(
        (std::size_t)c.M * c.N * sizeof(float));
    if (ab.empty() || bb.empty() || bib.empty() || cb.empty()) { return; }
    if (c.bf16) {
      auto* o = static_cast<std::uint16_t*>(ab.contents());
      for (std::size_t i = 0; i < A.size(); ++i) { o[i] = bf16(A[i]); }
    } else {
      std::memcpy(ab.contents(), A.data(), A.size() * 4);
    }
    std::memcpy(bb.contents(), B.data(), B.size() * 4);
    std::memcpy(bib.contents(), bi.data(), bi.size() * 4);

    const int use_bias = c.bias ? 1 : 0, act = c.act ? 1 : 0;
    const int aelt = c.bf16 ? 1 : 0;
    {
      CommandStream stream = mc->make_command_stream();
      ComputeEncoder enc = stream.begin_compute();
      enc.set_function(fn);
      enc.set_buffer(0, ab);  enc.set_buffer(1, bb);
      enc.set_buffer(2, bib); enc.set_buffer(3, cb);
      enc.set_constant(4, c.M); enc.set_constant(5, c.N);
      enc.set_constant(6, c.K); enc.set_constant(7, c.K);
      enc.set_constant(8, act); enc.set_constant(9, use_bias);
      enc.set_buffer(10, ab); enc.set_constant(11, aelt);
      enc.set_buffer(12, cb); enc.set_constant(13, 0);   // fp32 C
      enc.dispatch({(unsigned)((c.M + 31) / 32) * 16,
                    (unsigned)((c.N + 31) / 32) * 16, 1}, {16, 16, 1});
      enc.end();
      std::string gerr;
      ASSERT_TRUE(stream.commit().wait_ok(&gerr));
    }
    std::vector<float> want((std::size_t)c.M * c.N);
    for (int m = 0; m < c.M; ++m) {
      for (int n = 0; n < c.N; ++n) {
        double acc = c.bias ? (double)bi[(std::size_t)n] : 0.0;
        for (int k = 0; k < c.K; ++k) {
          acc += (double)A[(std::size_t)m * c.K + k]
                 * (double)B[(std::size_t)n * c.K + k];
        }
        if (c.act) { acc = 1.0 / (1.0 + std::exp(-acc)); }
        want[(std::size_t)m * c.N + n] = (float)acc;
      }
    }
    const double e = rel_l2_((const float*)cb.contents(), want.data(),
                             want.size());
    worst = std::max(worst, e);
    const bool ok = e >= 0.0 && e < 1e-5;
    EXPECT_TRUE(ok);
    if (!ok) {
      std::printf("[vdn_gemm] %dx%dx%d bias %d act %d bf16 %d: rel-L2 "
                  "%.3e\n", c.M, c.N, c.K, (int)c.bias, (int)c.act,
                  (int)c.bf16, e);
    }
    ++ran;
  }
  EXPECT_TRUE(ran == 6);
  std::printf("[vdn_gemm] %d shapes, worst rel-L2 %.3e\n", ran, worst);
}

TEST(metal_vdn_branch, narrow_features_track_the_wide_path)
{
  // WHAT bf16 COSTS, measured rather than assumed -- and this is the
  // only test that checks the path a real forward takes, because the
  // goldens are an fp32 reference and the other tests pin themselves to
  // it exactly.
  //
  // The reference is bf16 here: scan.py says "Only the small [F,H,d,d]
  // results are promoted to fp32", so the features, the conv's output,
  // beta, the gate and the state going into the readout are all narrow
  // in the model this port implements. A is NOT (`a_fp32`), nor is
  // alpha's frame mean, nor the Cholesky island. So the wide arm below
  // is the generous one and the narrow arm is the faithful one; the bar
  // is what bf16's 8 mantissa bits cost, not a correctness threshold.
  const std::string stage = stage_(), gdir = golden_();
  std::ifstream mi(gdir + "/meta.json");
  if (!mi) { return; }
  FlexData meta = FlexData::from_json(mi);
  if (!meta.is_object()) { return; }
  auto m = meta.as_object();
  const int F = (int)m.at("frames").as_int(0);
  const int gh = (int)m.at("grid_h").as_int(0);
  const int gw = (int)m.at("grid_w").as_int(0);
  const int H = (int)m.at("heads").as_int(0);
  const int d = (int)m.at("head_dim").as_int(0);
  const int hidden = (int)m.at("hidden").as_int(0);
  const int Lt = (int)m.at("text_len").as_int(0);
  const int S = gh * gw, C = H * d;

  Session s;
  MetalCompute* mc = s.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  vdn::Config cfg;
  std::string err;
  if (!vdn::load_config(stage, &cfg, &err)) { return; }
  std::shared_ptr<WeightSet> ws =
      WeightSet::open(stage + "/linear_branch", nullptr);
  if (!ws) { return; }

  std::vector<float> xv, qr, kr, vr, tx, tk, tv, want;
  auto load = [&](const char* n, std::vector<float>* v) {
    return read_bin_(gdir + "/" + n + ".bin", v);
  };
  if (!load("xv", &xv) || !load("q_raw", &qr) || !load("k_raw", &kr)
      || !load("v_raw", &vr) || !load("text_x", &tx)
      || !load("text_k_raw", &tk) || !load("text_v_raw", &tv)
      || !load("readout", &want)) {
    EXPECT_TRUE(false);
    return;
  }
  auto buf = [&](const std::vector<float>& v) {
    SharedBuffer b = mc->make_shared_buffer(v.size() * 4);
    if (!b.empty()) { std::memcpy(b.contents(), v.data(), v.size() * 4); }
    return b;
  };
  SharedBuffer xb = buf(xv), qb = buf(qr), kb = buf(kr), vb = buf(vr);
  SharedBuffer txb = buf(tx), tkb = buf(tk), tvb = buf(tv);
  MetalVdnBranch::Inputs in;
  in.x = &xb; in.q_raw = &qb; in.k_raw = &kb; in.v_raw = &vb;
  in.text_x = &txb; in.text_k = &tkb; in.text_v = &tvb;
  MetalVdnBranch::Geometry geo;
  geo.frames = F; geo.grid_h = gh; geo.grid_w = gw; geo.text_len = Lt;
  const std::vector<vdn::Bound> bounds =
      vdn::window_bounds(F, (int)m.at("radius").as_int(0),
                         (int)m.at("chunk").as_int(0));
  const std::size_t obn = (std::size_t)F * S * C;

  auto arm = [&](bool narrow, std::vector<float>* got) {
    MetalVdnBranch::Dims dims;
    dims.heads = H; dims.head_dim = d; dims.hidden = hidden;
    dims.n_layers = 50;
    dims.bf16_features = narrow;
    std::unique_ptr<MetalVdnBranch> br =
        MetalVdnBranch::load(ws, mc, cfg, dims, &err);
    if (!br || !br->ensure_block(0, &err)) { return false; }
    SharedBuffer ob = mc->make_shared_buffer(obn * 4);
    if (ob.empty()) { return false; }
    std::memset(ob.contents(), 0, obn * 4);
    CommandStream st = mc->make_command_stream();
    ComputeEncoder enc = st.begin_compute();
    if (!br->encode(enc, 0, geo, bounds, in, ob, &err)) { return false; }
    enc.end();
    std::string gerr;
    if (!st.commit().wait_ok(&gerr)) { return false; }
    EXPECT_TRUE(br->solve_failures() == 0u);
    got->resize(obn);
    std::memcpy(got->data(), ob.contents(), obn * 4);
    return true;
  };

  std::vector<float> wide, narrow;
  ASSERT_TRUE(arm(false, &wide));
  ASSERT_TRUE(arm(true, &narrow));
  if (wide.empty() || narrow.size() != wide.size()) { return; }

  const double ew = rel_l2_(wide.data(), want.data(), want.size());
  const double en = rel_l2_(narrow.data(), want.data(), want.size());
  const double ab = rel_l2_(narrow.data(), wide.data(), wide.size());
  std::printf("[vdn_narrow] vs fp32 golden: wide %.3e, narrow %.3e; "
              "narrow vs wide %.3e\n", ew, en, ab);
  // The wide arm is the goldens' own bar; the narrow arm is held to
  // bf16's 8 mantissa bits, which is where the DiT's whole residual
  // stream already lives.
  EXPECT_TRUE(ew >= 0.0 && ew < 1e-5);
  EXPECT_TRUE(en >= 0.0 && en < 2e-2);
  // And it must NOT be identical -- that would mean the switch did
  // nothing and this test is measuring one arm twice.
  EXPECT_TRUE(ab > 0.0);
  // The anchor rows stay exactly zero in both: the partition does not
  // depend on precision.
  bool zero = true;
  for (std::size_t i = 0; i < (std::size_t)S * C; ++i) {
    zero = zero && narrow[i] == 0.0f
           && narrow[((std::size_t)(F - 1) * S) * C + i] == 0.0f;
  }
  EXPECT_TRUE(zero);
}

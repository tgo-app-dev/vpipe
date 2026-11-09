// RIFE frame interpolation (generative-models/rife/) against an fp32
// restatement of Practical-RIFE's IFNet.
//
// Goldens come from tools/dump_rife_golden.py, which writes the network
// as the release's train_log/IFNet_HDv3.py defines it (read, not run) and
// runs it in fp32 on the CPU -- the truth -- and in fp16 the way the
// release's --fp16 does, which is the bar: a port in f16 is as good as it
// can be when it is no further from fp32 than that.
//
//   VPIPE_RIFE_GOLDEN           the dumper's --out ROOT: one directory a
//       model version (v4.25/, v4.22/, ...), one a case under it, each
//       with a meta.json naming the checkpoint it ran. Every case found is
//       checked, so the versions' shapes (4.22's 8-channel Head, 4.25.lite's
//       32 / 16 / 8 / 4 / 1 scale list) all run.
//   VPIPE_RIFE_TEST_MODEL_PATH  one release's train_log/ (flownet.pkl
//       beside RIFE_HDv3.py) for the tests that need a model and no
//       golden.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "common/vpipe-format.h"
#include "generative-models/rife/metal-rife.h"
#include "generative-models/weight-set.h"
#include "stages/model-detect.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

using namespace vpipe;
using metal_compute::MetalCompute;
using genai::MetalRife;

namespace fs = std::filesystem;

namespace {

const char*
env_(const char* k)
{
  const char* v = std::getenv(k);
  return v != nullptr && *v != '\0' ? v : nullptr;
}

template <typename T>
bool
read_bin_(const fs::path& p, std::size_t n, std::vector<T>* out)
{
  std::ifstream f(p, std::ios::binary);
  if (!f) { return false; }
  out->resize(n);
  f.read(reinterpret_cast<char*>(out->data()),
         (std::streamsize)(n * sizeof(T)));
  return (std::size_t)f.gcount() == n * sizeof(T);
}

FlexData
read_json_(const fs::path& p)
{
  std::ifstream f(p);
  if (!f) { return FlexData{}; }
  return FlexData::from_json(f);
}

double
num_(const FlexData& o, const char* k)
{
  FlexData c = o;
  if (!c.is_object() || !c.as_object().contains(k)) { return 0.0; }
  const FlexData v = c.as_object().at(k);
  return v.as_real(0.0);
}

std::string
str_(const FlexData& o, const char* k)
{
  FlexData c = o;
  if (!c.is_object() || !c.as_object().contains(k)) { return {}; }
  return std::string(c.as_object().at(k).as_string(""));
}

double
rel_l2_(const std::vector<float>& a, const std::vector<float>& b)
{
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = (double)a[i] - b[i];
    num += d * d;
    den += (double)b[i] * b[i];
  }
  return std::sqrt(num / std::max(den, 1e-30));
}

std::shared_ptr<genai::WeightSet>
open_(const std::string& file)
{
  return genai::WeightSet::open(file, nullptr);
}

// Every case directory under the golden root: <root>/<version>/<case>.
std::vector<fs::path>
golden_cases_(const char* root)
{
  std::vector<fs::path> out;
  std::error_code ec;
  for (const auto& v : fs::directory_iterator(root, ec)) {
    if (!v.is_directory()) { continue; }
    for (const auto& c : fs::directory_iterator(v.path(), ec)) {
      if (fs::exists(c.path() / "meta.json")) { out.push_back(c.path()); }
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

// The device's growth over a settled baseline while `fn` runs, sampled.
template <typename Fn>
std::size_t
device_peak_of_(MetalCompute* mc, Fn&& fn)
{
  const std::size_t base = mc->memory_budget_settled().allocated;
  std::atomic<std::size_t> peak{base};
  std::atomic<bool> done{false};
  std::thread sampler([&] {
    while (!done.load()) {
      const std::size_t a = mc->memory_budget().allocated;
      if (a > peak.load()) { peak.store(a); }
      std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
  });
  fn();
  done.store(true);
  sampler.join();
  const std::size_t top =
      std::max(peak.load(), mc->memory_budget().allocated);
  return top > base ? top - base : 0;
}

}  // namespace

// The network is read off the tensors, and the one thing that is not in
// them -- the scale list -- off the driver beside them. 4.25.lite is the
// release whose list is not 2^(n-1) .. 1, which is why the driver is read
// at all.
TEST(rife, reads_the_network_off_the_checkpoint)
{
  const char* golden = env_("VPIPE_RIFE_GOLDEN");
  if (golden == nullptr) { return; }
  int seen = 0;
  for (const fs::path& c : golden_cases_(golden)) {
    const FlexData meta = read_json_(c / "meta.json");
    const std::string model = str_(meta, "model");
    std::string err;
    const std::string file = MetalRife::resolve_weights(model, &err);
    ASSERT_TRUE(!file.empty());
    MetalRife::Info info;
    ASSERT_TRUE(MetalRife::read_info(file, &info, &err));
    EXPECT_TRUE(info.scales_from_driver);
    EXPECT_TRUE(info.blocks() == (int)num_(meta, "blocks"));
    // The golden's list is at its --scale; the Info's at 1.
    FlexData m = meta;
    const FlexData sl = m.as_object().at("scale_list");
    const double sc = num_(meta, "scale");
    bool same = sl.is_array() && (int)sl.as_array().size() == info.blocks();
    for (int i = 0; same && i < info.blocks(); ++i) {
      same = std::fabs(sl.as_array()[(std::size_t)i].as_real(0.0) -
                       info.scales[(std::size_t)i] / sc) < 1e-9;
    }
    EXPECT_TRUE(same);
    std::printf("[rife] %s: head %d -> %d, blocks", c.parent_path()
                .filename().c_str(), info.head_width, info.features);
    for (int i = 0; i < info.blocks(); ++i) {
      std::printf(" %d@%g", info.widths[(std::size_t)i],
                  info.scales[(std::size_t)i]);
    }
    std::printf(", %.2f M params\n", (double)info.params * 1e-6);
    // A train_log/ is offered where a stage asks for a `rife` model.
    const DetectedModel det = detect_model_dir(model);
    EXPECT_TRUE(det.model_type == "rife");
    EXPECT_TRUE(det.inputs == std::vector<std::string>{"video"} &&
                det.outputs == std::vector<std::string>{"video"});
    ++seen;
  }
  EXPECT_TRUE(seen > 0);
}

// The interpolation, against the fp32 restatement, held to the release's
// own fp16 distance from it -- on EVERY conv route the machine has: the
// hardware op where there are matrix cores (and it must be what ran),
// the gather conv always (VPIPE_RIFE_NO_HWCONV forces it).
TEST(rife, matches_the_reference)
{
  const char* golden = env_("VPIPE_RIFE_GOLDEN");
  if (golden == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const bool user_off = env_("VPIPE_RIFE_NO_HWCONV") != nullptr;
  std::vector<bool> routes = {false};
  if (MetalRife::hw_conv_wanted(mc)) { routes.insert(routes.begin(), true); }
  int cases = 0;
  for (const bool hw : routes) {
    if (!hw) { setenv("VPIPE_RIFE_NO_HWCONV", "1", 1); }
    for (const fs::path& c : golden_cases_(golden)) {
      const FlexData meta = read_json_(c / "meta.json");
      const int h = (int)num_(meta, "h"), w = (int)num_(meta, "w");
      const double scale = num_(meta, "scale");
      std::string err;
      const std::string file =
          MetalRife::resolve_weights(str_(meta, "model"), &err);
      auto ws = open_(file);
      ASSERT_TRUE(ws != nullptr);
      auto rife = MetalRife::load(ws, mc, file, &err);
      if (!rife) { std::printf("[rife] load: %s\n", err.c_str()); }
      ASSERT_TRUE(rife != nullptr);
      if (!rife) { continue; }
      EXPECT_TRUE(rife->hw_conv() == hw);
      const std::size_t n = (std::size_t)3 * h * w;
      std::vector<std::uint8_t> i0, i1;
      ASSERT_TRUE(read_bin_(c / "i0.u8", n, &i0));
      ASSERT_TRUE(read_bin_(c / "i1.u8", n, &i1));
      MetalRife::Frame a, b;
      const std::uint64_t ph0 = rife->hw_dispatches();
      const std::uint64_t pg0 = rife->gather_dispatches();
      ASSERT_TRUE(rife->prepare(i0.data(), false, h, w, scale, &a, &err));
      ASSERT_TRUE(rife->prepare(i1.data(), false, h, w, scale, &b, &err));
      // The Head's four convs a frame, its 4-wide first one included.
      EXPECT_TRUE(rife->hw_dispatches() - ph0 == (hw ? 8u : 0u));
      EXPECT_TRUE(rife->gather_dispatches() - pg0 == (hw ? 0u : 8u));
      FlexData m = meta;
      const FlexData tvs = m.as_object().at("timesteps");
      for (const FlexData& tv : tvs.as_array()) {
        const double t = tv.as_real(0.0);
        char tag[32];
        std::snprintf(tag, sizeof tag, "%g", t);
        std::vector<float> ref, ref16, ours(n);
        ASSERT_TRUE(read_bin_(c / fmt("out_{}_f32.bin", tag)(), n, &ref));
        ASSERT_TRUE(
            read_bin_(c / fmt("out_{}_f16.bin", tag)(), n, &ref16));
        const std::uint64_t hw0 = rife->hw_dispatches();
        const std::uint64_t g0 = rife->gather_dispatches();
        ASSERT_TRUE(rife->interpolate(a, b, (float)t, true, ours.data(),
                                      &err));
        const std::uint64_t nhw = rife->hw_dispatches() - hw0;
        const std::uint64_t ng = rife->gather_dispatches() - g0;
        const double d = rel_l2_(ours, ref);
        const double bar = rel_l2_(ref16, ref);
        {
          // The flow and mask the blend used, against the reference's.
          const int ph = (int)num_(meta, "padded_h");
          const int pw = (int)num_(meta, "padded_w");
          const std::size_t P = (std::size_t)ph * pw;
          std::vector<float> rf, rm, of(4 * P), om(P);
          if (read_bin_(c / fmt("flow_{}_f32.bin", tag)(), 4 * P, &rf) &&
              read_bin_(c / fmt("mask_{}_f32.bin", tag)(), P, &rm)) {
            const auto* fl =
                static_cast<const float*>(rife->last_flow().contents());
            const auto* ax =
                static_cast<const _Float16*>(rife->last_aux().contents());
            for (std::size_t k = 0; k < P; ++k) {
              for (int ch = 0; ch < 4; ++ch) {
                of[(std::size_t)ch * P + k] = fl[k * 4 + ch];
              }
              om[k] = (float)ax[k * 12];
            }
            std::printf("[rife]   flow %.5f, mask logit %.5f from fp32\n",
                        rel_l2_(of, rf), rel_l2_(om, rm));
          }
        }
        // u8 is the same blend, rounded.
        std::vector<std::uint8_t> u8(n);
        ASSERT_TRUE(rife->interpolate(a, b, (float)t, false, u8.data(),
                                      &err));
        int worst = 0;
        for (std::size_t k = 0; k < n; ++k) {
          const int want = (int)std::lround(
              std::clamp(ours[k], 0.0f, 1.0f) * 255.0f);
          worst = std::max(worst, std::abs(want - (int)u8[k]));
        }
        std::printf("[rife] %s %s/%s t=%s: rel-L2 %.5f from fp32 (the "
                    "release's fp16 %.5f), u8 %d level(s) from f32 "
                    "rounded; %llu convs on the hardware op, %llu gather\n",
                    hw ? "hw" : "gather",
                    c.parent_path().filename().c_str(),
                    c.filename().c_str(), tag, d, bar, worst,
                    (unsigned long long)nhw, (unsigned long long)ng);
        EXPECT_TRUE(d <= 1.25 * bar);
        EXPECT_TRUE(worst <= 1);
        // The route that was asked for is the one that ran: on the
        // hardware route EVERY conv takes the op.
        if (hw) {
          EXPECT_TRUE(nhw > 0 && ng == 0);
        } else {
          EXPECT_TRUE(nhw == 0 && ng > 0);
        }
        ++cases;
      }
    }
    if (!hw && !user_off) { unsetenv("VPIPE_RIFE_NO_HWCONV"); }
  }
  EXPECT_TRUE(cases > 0);
}

// The driver's scene test, ported: the same score off the same padded
// 32 x 32 resize.
TEST(rife, the_scene_score_is_the_releases)
{
  const char* golden = env_("VPIPE_RIFE_GOLDEN");
  if (golden == nullptr) { return; }
  int seen = 0;
  for (const fs::path& c : golden_cases_(golden)) {
    const FlexData meta = read_json_(c / "meta.json");
    const int h = (int)num_(meta, "h"), w = (int)num_(meta, "w");
    const std::size_t n = (std::size_t)3 * h * w;
    std::vector<std::uint8_t> i0, i1;
    ASSERT_TRUE(read_bin_(c / "i0.u8", n, &i0));
    ASSERT_TRUE(read_bin_(c / "i1.u8", n, &i1));
    const double ours = MetalRife::scene_ssim(i0.data(), i1.data(), false,
                                              h, w, num_(meta, "scale"));
    const double ref = num_(meta, "ssim");
    std::printf("[rife] %s/%s scene ssim %.6f (reference %.6f)\n",
                c.parent_path().filename().c_str(), c.filename().c_str(),
                ours, ref);
    EXPECT_TRUE(std::fabs(ours - ref) < 1e-5);
    // A frame against itself is 1.
    EXPECT_TRUE(std::fabs(MetalRife::scene_ssim(i0.data(), i0.data(), false,
                                                h, w, 1.0) - 1.0) < 1e-9);
    ++seen;
  }
  EXPECT_TRUE(seen > 0);
}

// What the model holds is what was booked: the static figures the stage
// books before anything loads, against the load and the device's own
// growth over a load, two prepares and an interpolation. Each geometry
// from a fresh load: a second geometry reallocates into room the first
// freed, and its growth would read as nothing.
TEST(rife, allocates_what_it_books)
{
  const char* dir = env_("VPIPE_RIFE_TEST_MODEL_PATH");
  if (dir == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  std::string err;
  const std::string file = MetalRife::resolve_weights(dir, &err);
  ASSERT_TRUE(!file.empty());
  MetalRife::Info info;
  ASSERT_TRUE(MetalRife::read_info(file, &info, &err));
  for (const auto& [h, w, scale] :
       {std::tuple<int, int, double>{720, 1280, 1.0},
        std::tuple<int, int, double>{540, 960, 0.5}}) {
    // A smooth picture, so the flow between a frame and itself is ~0
    // (on noise the network's flow is anything).
    std::vector<std::uint8_t> px((std::size_t)3 * h * w);
    for (int c = 0; c < 3; ++c) {
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          px[((std::size_t)c * h + y) * w + x] = (std::uint8_t)std::lround(
              128.0 + 90.0 * std::sin(x / (17.0 + 5 * c)) *
                          std::cos(y / 23.0));
        }
      }
    }
    std::vector<std::uint8_t> out(px.size());
    std::unique_ptr<MetalRife> rife;
    MetalRife::Frame a, b;
    const std::size_t peak = device_peak_of_(mc, [&] {
      rife = MetalRife::load(open_(file), mc, file, &err);
      ASSERT_TRUE(rife != nullptr);
      if (!rife) { return; }
      ASSERT_TRUE(rife->prepare(px.data(), false, h, w, scale, &a, &err));
      ASSERT_TRUE(rife->prepare(px.data(), false, h, w, scale, &b, &err));
      ASSERT_TRUE(rife->interpolate(a, b, 0.5f, false, out.data(), &err));
    });
    if (!rife) { continue; }
    EXPECT_TRUE(rife->held_weight_bytes() ==
                MetalRife::weight_bytes(info, rife->hw_conv()));
    EXPECT_TRUE(rife->resident_bytes() ==
                rife->held_weight_bytes() +
                    MetalRife::scratch_bytes(info, h, w, scale, false,
                                         rife->hw_conv()));
    const std::size_t booked =
        MetalRife::weight_bytes(info, rife->hw_conv()) +
        MetalRife::scratch_bytes(info, h, w, scale, false,
                                 rife->hw_conv()) +
        2 * MetalRife::frame_bytes(info, h, w, scale);
    std::printf("[rife] %dx%d at scale %g: device +%.1f MB, booked %.1f MB "
                "(weights %.1f, scratch %.1f, 2 frames of %.1f)\n", w, h,
                scale, peak / 1048576.0, booked / 1048576.0,
                MetalRife::weight_bytes(info, rife->hw_conv()) / 1048576.0,
                MetalRife::scratch_bytes(info, h, w, scale, false,
                                         rife->hw_conv()) /
                    1048576.0,
                MetalRife::frame_bytes(info, h, w, scale) / 1048576.0);
    EXPECT_TRUE(peak <= booked + (8u << 20));
    EXPECT_TRUE(peak + (8u << 20) >= booked);
    // A frame interpolated with itself is itself -- away from the edges.
    // At the bottom and right the network sees the driver's zero padding
    // (540 rows padded to 768 at scale 0.5), and the release itself moves
    // pixels there: 76 levels at the corner on this picture, which the
    // v4.25/self-pad golden holds this port to.
    constexpr int kMargin = 32;
    int worst = 0;
    for (int c = 0; c < 3; ++c) {
      for (int y = kMargin; y < h - kMargin; ++y) {
        for (int x = kMargin; x < w - kMargin; ++x) {
          const std::size_t k = ((std::size_t)c * h + y) * w + x;
          worst = std::max(worst, std::abs((int)out[k] - (int)px[k]));
        }
      }
    }
    std::printf("[rife] a frame with itself: worst %d level(s) inside a "
                "%d px margin\n", worst, kMargin);
    EXPECT_TRUE(worst <= 2);
    rife.reset();
    a = MetalRife::Frame{};
    b = MetalRife::Frame{};
  }
}

// How long an interpolation takes at the sizes a user runs. Opt-in:
// VPIPE_RIFE_BENCH=1.
TEST(rife, bench)
{
  const char* dir = env_("VPIPE_RIFE_TEST_MODEL_PATH");
  if (dir == nullptr || env_("VPIPE_RIFE_BENCH") == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  std::string err;
  const std::string file = MetalRife::resolve_weights(dir, &err);
  const bool user_off = env_("VPIPE_RIFE_NO_HWCONV") != nullptr;
  std::vector<bool> routes = {false};
  if (MetalRife::hw_conv_wanted(mc)) { routes.insert(routes.begin(), true); }
  for (const bool hw : routes) {
  if (!hw) { setenv("VPIPE_RIFE_NO_HWCONV", "1", 1); }
  auto rife = MetalRife::load(open_(file), mc, file, &err);
  ASSERT_TRUE(rife != nullptr);
  if (!rife) { return; }
  for (const auto& [h, w, scale] :
       {std::tuple<int, int, double>{540, 960, 1.0},
        std::tuple<int, int, double>{1080, 1920, 1.0},
        std::tuple<int, int, double>{2160, 3840, 0.5}}) {
    std::vector<std::uint8_t> px((std::size_t)3 * h * w, 128), out(px.size());
    MetalRife::Frame a, b;
    ASSERT_TRUE(rife->prepare(px.data(), false, h, w, scale, &a, &err));
    ASSERT_TRUE(rife->prepare(px.data(), false, h, w, scale, &b, &err));
    ASSERT_TRUE(rife->interpolate(a, b, 0.5f, false, out.data(), &err));
    constexpr int kN = 10;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kN; ++i) {
      ASSERT_TRUE(rife->prepare(px.data(), false, h, w, scale, &b, &err));
    }
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < kN; ++i) {
      ASSERT_TRUE(rife->interpolate(a, b, 0.5f, false, out.data(), &err));
    }
    auto t2 = std::chrono::steady_clock::now();
    const double prep =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / kN;
    const double interp =
        std::chrono::duration<double, std::milli>(t2 - t1).count() / kN;
    std::printf("[rife] %s %dx%d scale %g: prepare %.2f ms, interpolate "
                "%.2f ms\n", hw ? "hw" : "gather", w, h, scale, prep,
                interp);
    if (env_("VPIPE_RIFE_PROFILE") != nullptr) {
      rife->clear_profile();
      ASSERT_TRUE(rife->prepare(px.data(), false, h, w, scale, &b, &err));
      ASSERT_TRUE(rife->interpolate(a, b, 0.5f, false, out.data(), &err));
      for (const auto& [cat, row] : rife->profile()) {
        std::printf("[rife]   %-16s %8.2f ms over %d dispatches\n",
                    cat.c_str(), row.ms, row.dispatches);
      }
    }
  }
  if (!hw && !user_off) { unsetenv("VPIPE_RIFE_NO_HWCONV"); }
  }
}

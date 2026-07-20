// Qwen-Image-Edit-2511 AWQ calibration collector (M8). Runs
// collect_qwen_image_calibration with a FEW prompts/steps (light) and checks it
// produces the 8 per-group col-absmax files, each finite + non-degenerate (the
// DiT taps captured real activation magnitudes). Persists the calib to
// <golden>/awq-calib so a follow-up `model-quantize awq=true calib_dir=...` can
// reuse it without re-running the heavy denoise.
//
// Env: VPIPE_QWEN_IMAGE_EDIT_TEST_MODEL_PATH (the pipeline root),
// VPIPE_QWEN_IMAGE_EDIT_GOLDEN (calib output dir). Skips if unset. SLOW (loads
// the 20B DiT); env-gated.

#include "minitest.h"

#include "common/session.h"
#include "generative-models/qwen-image/metal-qwen-image-calibration.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace vpipe;

namespace {

std::vector<float>
read_f32_(const std::string& p)
{
  std::ifstream in(p, std::ios::binary);
  std::vector<float> o;
  if (!in) { return o; }
  in.seekg(0, std::ios::end);
  const std::streamoff n = in.tellg();
  in.seekg(0, std::ios::beg);
  o.resize((std::size_t)n / 4);
  in.read(reinterpret_cast<char*>(o.data()), n);
  return o;
}

}  // namespace

TEST(qwen_image_edit_awq, collector_produces_calibration)
{
  const char* root = std::getenv("VPIPE_QWEN_IMAGE_EDIT_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_QWEN_IMAGE_EDIT_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }

  const std::string calib = std::string(gd) + "/awq-calib";
  // A few diverse prompts, few steps -- enough to verify the taps, not a full
  // production calibration.
  const std::vector<std::string> prompts = {
      "a red fox in fresh snow, photorealistic",
      "a neon city street at night, reflections",
      "a still life of fruit on a wooden table"};
  std::string err;
  const bool ok = genai::collect_qwen_image_calibration(
      sess.metal_compute(), root, prompts, /*steps=*/4, 256, 256, /*seed=*/1,
      calib, &err);
  if (!ok) {
    std::printf("[qwen_image_edit_awq] collector failed: %s\n", err.c_str());
  }
  ASSERT_TRUE(ok);

  // The 8 dual-stream groups must exist, be finite, and have captured non-zero
  // activation magnitudes (some channels large -> AWQ has signal to clip on).
  const char* groups[] = {"img_qkv", "txt_qkv", "img_o", "txt_o",
                          "img_fc1", "txt_fc1", "img_fc2", "txt_fc2"};
  for (const char* g : groups) {
    const std::vector<float> v = read_f32_(calib + "/" + g + ".f32");
    ASSERT_TRUE(!v.empty());
    bool finite = true;
    double mx = 0.0;
    int nz = 0;
    for (float x : v) {
      if (!std::isfinite(x)) { finite = false; break; }
      if (x > mx) { mx = x; }
      if (x > 0.0f) { ++nz; }
    }
    std::printf("[qwen_image_edit_awq] %-8s n=%zu max=%.4f nonzero=%d\n", g,
                v.size(), mx, nz);
    ASSERT_TRUE(finite);
    EXPECT_TRUE(mx > 0.0);              // taps captured activation magnitude
    EXPECT_TRUE(nz > (int)v.size() / 4);  // most channels active
  }
}

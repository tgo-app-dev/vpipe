// Two real MiniMax-H3 DiT blocks through one forward, for the tests of
// what rides on the DiT's own blocks -- adapters, curve branches, the
// banded forward -- without loading 50 of them. Header-only, one copy
// per test file (an anonymous namespace).
//
// VPIPE_MINIMAX_H3_TEST_MODEL_PATH names the checkpoint (a MiniMax-H3
// root or a partition); `setup` takes the config and cuts it to `layers`.
#ifndef TESTS_UNIT_TESTS_MINIMAX_H3_BLOCK_RUN_H
#define TESTS_UNIT_TESTS_MINIMAX_H3_BLOCK_RUN_H

#include "apple-silicon/metal-compute/metal-compute.h"
#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/minimax-h3/minimax-h3-layout.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace {

using vpipe::metal_compute::MetalCompute;
using vpipe::genai::MetalMiniMaxH3Transformer;
namespace h3b = vpipe::genai::minimax_h3;

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
  const std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

vpipe::metal_compute::SharedBuffer
bf16_buf_(MetalCompute* mc, const std::vector<float>& v)
{
  vpipe::metal_compute::SharedBuffer b =
      mc->make_shared_buffer(v.size() * 2);
  if (b.empty()) { return b; }
  auto* d = static_cast<std::uint16_t*>(b.contents());
  for (std::size_t i = 0; i < v.size(); ++i) { d[i] = to_bf16_(v[i]); }
  return b;
}

double
rel_l2_(const std::vector<float>& a, const std::vector<float>& b)
{
  if (a.size() != b.size() || a.empty()) { return 1e9; }
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d;
    den += (double)b[i] * (double)b[i];
  }
  return std::sqrt(num / std::max(den, 1e-30));
}

std::string
env_(const char* k)
{
  const char* v = std::getenv(k);
  return v != nullptr ? std::string(v) : std::string();
}

// Two real blocks of the DiT through one forward, with the adapter specs
// given; the video velocity back as f32.
struct BlockRun {
  std::string root;
  MetalMiniMaxH3Transformer::Config cfg;
  h3b::PackedLayout L;
  std::vector<float> uniq;
  std::vector<int> row_idx;
  vpipe::metal_compute::SharedBuffer vb, ab, tb;
  int n_video = 0;

  bool
  setup(MetalCompute* mc, const std::string& r, int layers, int T = 2,
        int h = 12, int w = 20, int audio = 8)
  {
    root = r;
    std::string cerr;
    if (!MetalMiniMaxH3Transformer::config_from_json(root, cfg, &cerr)) {
      std::printf("  config: %s\n", cerr.c_str());
      return false;
    }
    cfg.n_layers = layers;
    const std::vector<int> tags(8, h3b::kTextTag);
    if (!h3b::build_packed_sequence(tags, T, h, w, audio, cfg.patch_h,
                                    cfg.patch_w, h3b::kAudioChannels, {},
                                    &L)) {
      return false;
    }
    // The refine's plan: the video at 1 - 0.909375, the audio clean.
    h3b::build_row_timesteps(L, 0.090625f, 1.0f, 1.0f, &uniq, &row_idx);
    n_video = (int)L.video_indices.size();
    auto ramp = [](std::size_t n, float k) {
      std::vector<float> v(n);
      for (std::size_t i = 0; i < n; ++i) {
        v[i] = std::sin((float)i * k) * 0.5f;
      }
      return v;
    };
    vb = bf16_buf_(mc, ramp((std::size_t)n_video * cfg.video_patch_elems(),
                            0.017f));
    ab = bf16_buf_(mc, ramp((std::size_t)L.num_audio_rows *
                                cfg.audio_channels, 0.031f));
    tb = bf16_buf_(mc, ramp(tags.size() * (std::size_t)cfg.text_dim, 0.005f));
    return !vb.empty() && !ab.empty() && !tb.empty();
  }

  // `with`: the config to load with (the tiers), else `cfg`; `routes`:
  // the engine's GEMM routes, else steel's 32-row tile pinned.
  bool
  run(MetalCompute* mc,
      const std::vector<MetalMiniMaxH3Transformer::LoraSpec>& specs,
      std::vector<float>* out, int* mods, float turn_to = -1.0f,
      bool baked = false,
      const MetalMiniMaxH3Transformer::Config* with = nullptr,
      bool routes = false,
      std::unique_ptr<MetalMiniMaxH3Transformer>* keep = nullptr)
  {
    auto m = MetalMiniMaxH3Transformer::load(
        root, mc, with != nullptr ? *with : cfg, false, specs);
    if (m == nullptr) { return false; }
    if (!routes) {
      m->set_gemm_route(MetalMiniMaxH3Transformer::GemmRoute::kSteelBm32);
    }
    if (mods != nullptr) { *mods = m->lora_modules(); }
    if (turn_to >= 0.0f) { m->set_lora_scale(0, turn_to); }
    MetalMiniMaxH3Transformer::Step step;
    step.video = &vb;  step.audio = &ab;  step.text = &tb;
    step.layout = &L;  step.timesteps = &uniq;
    step.row_timestep_index = &row_idx;
    // The denoise's route: the modulation baked for the schedule, the
    // curve added to the baked slice per forward.
    if (baked) {
      std::string berr;
      if (!m->bake_adaln({uniq}, &berr)) {
        std::printf("  bake: %s\n", berr.c_str());
        return false;
      }
      step.schedule_index = 0;
    }
    std::string ferr;
    const auto v = m->forward(step, &ferr);
    if (v.empty()) {
      std::printf("  forward: %s\n", ferr.c_str());
      return false;
    }
    const std::size_t n = (std::size_t)n_video * cfg.video_patch_elems();
    out->resize(n);
    const auto* p = static_cast<const std::uint16_t*>(v.video.contents());
    for (std::size_t i = 0; i < n; ++i) { (*out)[i] = from_bf16_(p[i]); }
    if (keep != nullptr) { *keep = std::move(m); }
    return true;
  }

  // One more forward on a model run() kept; its wall clock in `ms`.
  // `tweak` edits the Step first (a forward's own Sol routing, say).
  bool
  forward_on(MetalMiniMaxH3Transformer& m, std::vector<float>* out,
             double* ms,
             const std::function<void(MetalMiniMaxH3Transformer::Step&)>&
                 tweak = {})
  {
    MetalMiniMaxH3Transformer::Step step;
    step.video = &vb;  step.audio = &ab;  step.text = &tb;
    step.layout = &L;  step.timesteps = &uniq;
    step.row_timestep_index = &row_idx;
    if (tweak) { tweak(step); }
    std::string ferr;
    const auto t0 = std::chrono::steady_clock::now();
    const auto v = m.forward(step, &ferr);
    *ms = std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - t0).count();
    if (v.empty()) {
      std::printf("  forward: %s\n", ferr.c_str());
      return false;
    }
    const std::size_t n = (std::size_t)n_video * cfg.video_patch_elems();
    out->resize(n);
    const auto* p = static_cast<const std::uint16_t*>(v.video.contents());
    for (std::size_t i = 0; i < n; ++i) { (*out)[i] = from_bf16_(p[i]); }
    return true;
  }
};

}  // namespace

#endif

#ifndef VPIPE_TESTS_UNIT_TESTS_MINIMAX_H3_SYNTHETIC_H
#define VPIPE_TESTS_UNIT_TESTS_MINIMAX_H3_SYNTHETIC_H

// A synthetic MiniMax-H3 DiT as one ComfyUI-saved file, for tests that
// need the real code paths on a model small enough to train in
// milliseconds: hidden 256, two heads of 128, ffn 512, text 256, two main
// blocks and one refiner block, the released layout (gate-first fc1,
// time MLP, AdaLN), seeded.

#include "generative-models/train/lora-params.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace vpipe {
namespace h3_synthetic {

namespace fs = std::filesystem;

struct St {
  std::string name, dtype;
  std::vector<std::int64_t> shape;
  std::vector<std::uint8_t> bytes;
};

inline bool
write_st_(const fs::path& p, const std::vector<St>& ts)
{
  std::string hdr = "{";
  std::uint64_t off = 0;
  for (std::size_t i = 0; i < ts.size(); ++i) {
    std::string shp;
    for (std::size_t d = 0; d < ts[i].shape.size(); ++d) {
      shp += (d ? "," : "") + std::to_string(ts[i].shape[d]);
    }
    hdr += (i ? "," : "") + std::string("\"") + ts[i].name +
           "\":{\"dtype\":\"" + ts[i].dtype + "\",\"shape\":[" + shp +
           "],\"data_offsets\":[" + std::to_string(off) + "," +
           std::to_string(off + ts[i].bytes.size()) + "]}";
    off += ts[i].bytes.size();
  }
  hdr += "}";
  while (hdr.size() % 8 != 0) { hdr += ' '; }
  std::ofstream f(p, std::ios::binary);
  if (!f) { return false; }
  const std::uint64_t n = hdr.size();
  f.write(reinterpret_cast<const char*>(&n), 8);
  f.write(hdr.data(), (std::streamsize)hdr.size());
  for (const St& t : ts) {
    f.write(reinterpret_cast<const char*>(t.bytes.data()),
            (std::streamsize)t.bytes.size());
  }
  return f.good();
}

inline St
st_(const std::string& nm, std::vector<std::int64_t> shape,
    const std::vector<float>& v, bool f32)
{
  St t{nm, f32 ? "F32" : "BF16", std::move(shape), {}};
  t.bytes.resize(v.size() * (f32 ? 4 : 2));
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (f32) {
      std::memcpy(&t.bytes[i * 4], &v[i], 4);
    } else {
      const std::uint16_t b = genai::train::f32_to_bf16(v[i]);
      std::memcpy(&t.bytes[i * 2], &b, 2);
    }
  }
  return t;
}

// A released-layout H3 as one ComfyUI-saved file: hidden 256, two heads of
// 128, ffn 512, two main blocks and one refiner block.
inline bool
write_synthetic_(const fs::path& p)
{
  const int H = 256, A = 256, TXT = 256, FREQ = 256, TH = 64, M = 64,
            F = 512;
  const std::string P = "model.diffusion_model.";
  std::mt19937 rng(23);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  auto rnd = [&](std::size_t n, float sd, float mean = 0.0f) {
    std::vector<float> v(n);
    for (float& x : v) { x = mean + sd * nd(rng); }
    return v;
  };
  std::vector<St> ts;
  auto add = [&](const std::string& nm, std::vector<std::int64_t> shape,
                 const std::vector<float>& v, bool f32 = false) {
    ts.push_back(st_(P + nm, std::move(shape), v, f32));
  };
  add("video_patch_proj.weight", {H, 96}, rnd(H * 96, 0.05f), true);
  add("video_patch_proj.bias", {H}, rnd(H, 0.01f), true);
  add("audio_patch_proj.weight", {H, 32}, rnd(H * 32, 0.05f), true);
  add("audio_patch_proj.bias", {H}, rnd(H, 0.01f), true);
  add("condition_proj.weight", {H, TXT}, rnd((std::size_t)H * TXT, 0.03f));
  add("condition_proj.bias", {H}, rnd(H, 0.01f));
  std::vector<float> inv(16);
  for (int i = 0; i < 16; ++i) {
    inv[(std::size_t)i] = 1.0f / std::pow(10000.0f, (float)(2 * i) / 32.0f);
  }
  add("rope.inv_freq", {16}, inv, true);
  for (int L = 0; L < 3; ++L) {
    const std::string b = L < 2 ? "blocks." + std::to_string(L) + "."
                                : std::string("token_refiner.blocks.0.");
    add(b + "norm1.weight", {H}, rnd(H, 0.02f, 1.0f));
    add(b + "norm2.weight", {H}, rnd(H, 0.02f, 1.0f));
    add(b + "attn.q_norm.weight", {128}, rnd(128, 0.02f, 1.0f));
    add(b + "attn.k_norm.weight", {128}, rnd(128, 0.02f, 1.0f));
    add(b + "attn.qkv_proj.weight", {3 * A, H},
        rnd((std::size_t)3 * A * H, 0.04f));
    add(b + "attn.out_proj.weight", {H, A}, rnd((std::size_t)H * A, 0.04f));
    add(b + "mlp.fc1.weight", {2 * F, H}, rnd((std::size_t)2 * F * H, 0.04f));
    add(b + "mlp.fc2.weight", {H, F}, rnd((std::size_t)H * F, 0.01f));
    if (L < 2) {
      add(b + "adaln_proj.linear.weight", {18 * H, M},
          rnd((std::size_t)18 * H * M, 0.08f));
      add(b + "adaln_proj.linear.bias", {18 * H},
          rnd((std::size_t)18 * H, 0.05f));
    }
  }
  add("token_refiner.final_norm.weight", {H}, rnd(H, 0.02f, 1.0f));
  add("final_layer.norm.weight", {H}, rnd(H, 0.02f, 1.0f));
  add("final_layer.adaln_proj.linear.weight", {2 * H, M},
      rnd((std::size_t)2 * H * M, 0.08f));
  add("final_layer.adaln_proj.linear.bias", {2 * H},
      rnd((std::size_t)2 * H, 0.05f));
  add("final_layer.video_out.weight", {96, H}, rnd(96 * H, 0.05f), true);
  add("final_layer.video_out.bias", {96}, rnd(96, 0.01f), true);
  add("final_layer.audio_out.weight", {32, H}, rnd(32 * H, 0.05f), true);
  add("final_layer.audio_out.bias", {32}, rnd(32, 0.01f), true);
  add("time_embedder.proj_in.weight", {TH, FREQ},
      rnd((std::size_t)TH * FREQ, 0.15f), true);
  add("time_embedder.proj_in.bias", {TH}, rnd(TH, 0.2f), true);
  add("time_embedder.proj_out.weight", {M, TH}, rnd((std::size_t)M * TH, 0.3f),
      true);
  add("time_embedder.proj_out.bias", {M}, rnd(M, 0.2f), true);
  return write_st_(p, ts);
}

}  // namespace h3_synthetic
}  // namespace vpipe

#endif

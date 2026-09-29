// fp8-layouts.cc -- the checkpoint SPELLINGS FP8 arrives in beyond
// ComfyUI's, and the layouts those files use, on synthetic files small
// enough to write here:
//
//   torchao     `<layer>._weight_qdata` + `._weight_scale` (unsloth's
//               pre-quantized checkpoints) -> the comfy_quant spelling
//   refill      a streamed FP8 weight widened to bf16 WITH its scale
//   H3 names    MiniMax-H3 in the diffusers naming (lightx2v's FP8 turbo
//               DiT): renames, and the split qkv / swapped fc1 pieces
//   Qwen file   a Qwen-Image-2.1 DiT as one file with no config
//
// The real checkpoints are held to their own dense conversions, BIT FOR
// BIT, by minimax_h3_dit.layouts_agree and qwen_image_21_dit.
// checkpoints_agree; these are the parts that must not need a 34 GB file
// to be checked.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/session.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/minimax-h3/minimax-h3-diffusers-layout.h"
#include "generative-models/qwen-image/metal-qwen-image21-transformer.h"
#include "generative-models/shared/fp8-expand.h"
#include "generative-models/shared/fp8-layout.h"
#include "generative-models/shared/fp8.h"
#include "generative-models/shared/streamed-refill.h"
#include "generative-models/weight-set.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;

namespace {

namespace fs = std::filesystem;

struct T {
  std::string               name;
  std::string               dtype;
  std::vector<std::int64_t> shape;
  std::vector<std::uint8_t> bytes;
};

bool
write_st_(const fs::path& p, const std::vector<T>& ts)
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
  for (const T& t : ts) {
    f.write(reinterpret_cast<const char*>(t.bytes.data()),
            (std::streamsize)t.bytes.size());
  }
  return f.good();
}

fs::path
tmp_(const std::string& name)
{
  return fs::temp_directory_path() /
         ("vpipe-fp8-layouts-" + std::to_string(::getpid()) + "-" + name);
}

// A tensor of `n` elements of the given dtype width, filled with a
// cheap pattern that never produces an FP8 NaN code (0x7f / 0xff).
T
tensor_(const std::string& name, const std::string& dtype,
        std::vector<std::int64_t> shape, unsigned seed)
{
  std::size_t n = 1;
  for (auto d : shape) { n *= (std::size_t)d; }
  const std::size_t w = dtype == "F32" ? 4 : (dtype == "BF16" ? 2 : 1);
  T t{name, dtype, std::move(shape), std::vector<std::uint8_t>(n * w)};
  unsigned s = seed;
  for (std::size_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    if (w == 1) {
      t.bytes[i] = (std::uint8_t)(0x20 + (s >> 24) % 0x40);   // |v| < 2
    } else if (w == 2) {
      const std::uint16_t b = (std::uint16_t)(0x3c00 + (s >> 20) % 0x300);
      std::memcpy(&t.bytes[i * 2], &b, 2);
    } else {
      const float f = 0.5f + (float)((s >> 16) % 1000) / 1000.0f;
      std::memcpy(&t.bytes[i * 4], &f, 4);
    }
  }
  return t;
}

std::uint16_t
rne_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

}  // namespace

// torchao's serialized Float8Tensor, as unsloth ships it, reads as
// per-row scaled FP8 under the comfy_quant names, and decodes to
// code x scale rounded once.
TEST(fp8_layouts, torchao_spelling_reads_as_scaled_fp8)
{
  const fs::path p = tmp_("torchao.safetensors");
  const T q = tensor_("m.lin._weight_qdata", "F8_E4M3", {4, 8}, 1);
  const T sc = tensor_("m.lin._weight_scale", "F32", {4, 1}, 2);
  ASSERT_TRUE(write_st_(p, {q, sc, tensor_("m.norm.weight", "BF16", {8}, 3)}));
  auto w = MetalLlamaWeights::open_model(p.string());
  ASSERT_TRUE(w.has_value());
  if (!w.has_value()) { return; }
  EXPECT_TRUE(w->has("m.lin.weight") && w->has("m.lin.weight_scale"));
  EXPECT_TRUE(!w->has("m.lin._weight_qdata") && w->has("m.norm.weight"));
  fp8::Weight fw;
  std::string err;
  EXPECT_TRUE(fp8::resolve(*w, "m.lin.weight", &fw, &err));
  EXPECT_TRUE(fw.format == fp8::Format::kE4M3 && fw.per_row);
  std::vector<std::uint16_t> got(32);
  EXPECT_TRUE(fp8::decode_tensor_bf16(*w, "m.lin.weight", got.data(), 32,
                                      &err));
  const auto& t = fp8::f32_table(fp8::Format::kE4M3);
  std::size_t bad = 0;
  for (std::size_t r = 0; r < 4; ++r) {
    float s;
    std::memcpy(&s, &sc.bytes[r * 4], 4);
    for (std::size_t c = 0; c < 8; ++c) {
      bad += got[r * 8 + c] != rne_(t[q.bytes[r * 8 + c]] * s);
    }
  }
  EXPECT_TRUE(bad == 0);
  fs::remove(p);
}

// A STREAMED scaled weight widened to bf16 carries its scale -- the
// value a dense conversion holds -- and a plain one is the exact codes.
// The first version of the widening dropped the scale, which a model
// that refills in bf16 (MiniMax-H3's slots) would have run unnoticed.
TEST(fp8_layouts, refill_widens_with_the_scale)
{
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const fs::path p = tmp_("refill.safetensors");
  ASSERT_TRUE(write_st_(p, {tensor_("a.weight", "F8_E4M3", {4, 8}, 4),
                            tensor_("a.weight_scale", "F32", {4, 1}, 5),
                            tensor_("b.weight", "F8_E4M3", {4, 8}, 6)}));
  auto ws = WeightSet::open(p.string(), nullptr);
  ASSERT_TRUE(ws != nullptr);
  if (!ws) { return; }
  for (const char* nm : {"a.weight", "b.weight"}) {
    metal_compute::SharedBuffer dst = mc->make_shared_buffer(64);
    ASSERT_TRUE(refill_streamed_tensor(*ws, nm, dst, RefillDst::kBf16) ==
                Refill::kFilled);
    std::vector<std::uint16_t> want(32);
    std::string err;
    ASSERT_TRUE(fp8::decode_tensor_bf16(ws->src(), nm, want.data(), 32,
                                        &err));
    EXPECT_TRUE(std::memcmp(dst.contents(), want.data(), 64) == 0);
  }
  fs::remove(p);
}

// MiniMax-H3 in the diffusers naming opens under H3's own names, with
// the split projection and the swapped one left as the pieces the
// loader assembles -- in the order that makes them the native tensors.
TEST(fp8_layouts, h3_diffusers_names_translate)
{
  const fs::path p = tmp_("h3.safetensors");
  const std::string b = "transformer_blocks.0.";
  const std::string r = "token_refiner.refiner_blocks.0.";
  std::vector<T> ts = {
      tensor_("context_embedder.weight", "BF16", {8, 6}, 1),
      tensor_("proj_in.weight", "F32", {8, 4}, 2),
      tensor_("norm_out.linear.weight", "BF16", {16, 4}, 3),
      tensor_("time_embedder.linear_1.weight", "F32", {8, 2}, 4),
      tensor_(b + "adaln_proj.linear.weight", "F8_E4M3", {24, 4}, 5),
      tensor_(b + "adaln_proj.linear.weight_scale", "BF16", {24, 1}, 6),
      tensor_(b + "attn.to_q.weight", "F8_E4M3", {6, 8}, 7),
      tensor_(b + "attn.to_k.weight", "F8_E4M3", {6, 8}, 8),
      tensor_(b + "attn.to_v.weight", "F8_E4M3", {6, 8}, 9),
      tensor_(b + "attn.to_out.0.weight", "F8_E4M3", {8, 6}, 10),
      tensor_(b + "attn.norm_q.weight", "BF16", {4}, 11),
      tensor_(b + "ff.net.0.proj.weight", "F8_E4M3", {12, 8}, 12),
      tensor_(b + "ff.net.2.weight", "F8_E4M3", {8, 6}, 13),
      tensor_(r + "attn.to_q.weight", "BF16", {6, 8}, 14),
      tensor_(r + "attn.to_k.weight", "BF16", {6, 8}, 15),
      tensor_(r + "attn.to_v.weight", "BF16", {6, 8}, 16),
  };
  ASSERT_TRUE(write_st_(p, ts));
  auto w = MetalLlamaWeights::open_model(p.string());
  ASSERT_TRUE(w.has_value());
  if (!w.has_value()) { return; }
  for (const char* n :
       {"condition_proj.weight", "video_patch_proj.weight",
        "final_layer.adaln_proj.linear.weight",
        "time_embedder.proj_in.weight",
        "blocks.0.adaln_proj.linear.weight",
        "blocks.0.adaln_proj.linear.weight_scale",
        "blocks.0.attn.qkv_proj.q.weight", "blocks.0.attn.out_proj.weight",
        "blocks.0.attn.q_norm.weight", "blocks.0.mlp.fc1.ug.weight",
        "blocks.0.mlp.fc2.weight",
        "token_refiner.blocks.0.attn.qkv_proj.v.weight"}) {
    EXPECT_TRUE(w->has(n));
    if (!w->has(n)) { std::printf("[fp8_layouts] missing %s\n", n); }
  }
  EXPECT_TRUE(minimax_h3::is_diffusers_dit_file(p.string()));
  const auto qkv = minimax_h3::assembled_parts(*w, "blocks.0.attn.qkv_proj");
  ASSERT_TRUE(qkv.size() == 3u);
  if (qkv.size() == 3u) {
    EXPECT_TRUE(qkv[0].layer == "blocks.0.attn.qkv_proj.q" &&
                qkv[1].layer == "blocks.0.attn.qkv_proj.k" &&
                qkv[2].layer == "blocks.0.attn.qkv_proj.v" &&
                qkv[0].row0 == 0 && qkv[0].rows == 6);
  }
  // Gate first in the model, up first on disk: rows [6, 12) then [0, 6).
  const auto fc1 = minimax_h3::assembled_parts(*w, "blocks.0.mlp.fc1");
  ASSERT_TRUE(fc1.size() == 2u);
  if (fc1.size() == 2u) {
    EXPECT_TRUE(fc1[0].row0 == 6 && fc1[0].rows == 6 && fc1[1].row0 == 0 &&
                fc1[1].rows == 6);
  }
  EXPECT_TRUE(minimax_h3::assembled_parts(*w, "blocks.0.mlp.fc2").empty());
  fs::remove(p);

  // A tensor the layout does not place: the file keeps its own names
  // whole, and the reason is named.
  ts.push_back(tensor_("mystery.weight", "BF16", {2, 2}, 17));
  ASSERT_TRUE(write_st_(p, ts));
  auto u = MetalLlamaWeights::open_model(p.string());
  ASSERT_TRUE(u.has_value());
  if (u.has_value()) {
    EXPECT_TRUE(u->has("context_embedder.weight"));
    EXPECT_TRUE(minimax_h3::untranslatable_tensor(*u) == "mystery.weight");
  }
  fs::remove(p);
}

// A Qwen-Image-2.1 DiT as one file with no config -- recognised by its
// tensor names, its dimensions read off their shapes.
TEST(fp8_layouts, qwen_image21_single_file_reads_its_dims)
{
  const fs::path p = tmp_("qi21.safetensors");
  const std::string b = "transformer_blocks.";
  std::vector<T> ts = {
      tensor_("img_in.weight", "BF16", {16, 4}, 1),
      tensor_("proj_out.weight", "BF16", {4, 16}, 2),
      tensor_("modulation.1._weight_qdata", "F8_E4M3", {96, 16}, 3),
      tensor_("modulation.1._weight_scale", "F32", {96, 1}, 4),
      tensor_("txt_in.text_norm.weight", "BF16", {8}, 5),
      tensor_("txt_in.in_layer._weight_qdata", "F8_E4M3", {16, 8}, 6),
      tensor_("txt_in.in_layer._weight_scale", "F32", {16, 1}, 7),
  };
  for (int L = 0; L < 3; ++L) {
    const std::string pre = b + std::to_string(L) + ".";
    ts.push_back(tensor_(pre + "attn.to_q._weight_qdata", "F8_E4M3", {16, 16},
                         10 + L));
    ts.push_back(tensor_(pre + "attn.norm_q.weight", "BF16", {8}, 20 + L));
    ts.push_back(tensor_(pre + "img_mlp.gate_layer._weight_qdata", "F8_E4M3",
                         {48, 16}, 30 + L));
  }
  ASSERT_TRUE(write_st_(p, ts));
  EXPECT_TRUE(MetalQwenImage21Transformer::is_single_file_dit(p.string()));
  MetalQwenImage21Transformer::Config c;
  EXPECT_TRUE(MetalQwenImage21Transformer::Config::read_dims(p.string(), &c));
  EXPECT_TRUE(c.hidden == 16 && c.head_dim == 8 && c.n_heads == 2 &&
              c.n_layers == 3 && c.in_channels == 4 && c.out_channels == 4 &&
              c.txt_dim == 8 && c.mlp_ratio == 3);
  fs::remove(p);
}

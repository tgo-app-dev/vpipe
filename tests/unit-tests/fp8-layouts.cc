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
//   int8        ComfyUI's int8_tensorwise, plain and ConvRot-rotated, for
//               a model that opts in -- held to comfy_kitchen's arithmetic
//               on the host and the GPU
//   H3 saved    a MiniMax-H3 DiT ComfyUI saved: its prefix dropped, its
//               qkv read flat; and a small one in int8 forwarding as its
//               own dense twin in every hold
//
// The real checkpoints are held to their own dense conversions, BIT FOR
// BIT, by minimax_h3_dit.layouts_agree and qwen_image_21_dit.
// checkpoints_agree; these are the parts that must not need a 34 GB file
// to be checked.

#include "minitest.h"
#include "residency-reserve.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/session.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/minimax-h3/minimax-h3-diffusers-layout.h"
#include "generative-models/minimax-h3/minimax-h3-layout.h"
#include "generative-models/qwen-image/metal-qwen-image21-transformer.h"
#include "generative-models/shared/fp8-expand.h"
#include "generative-models/shared/fp8-layout.h"
#include "generative-models/shared/fp8.h"
#include "generative-models/shared/streamed-refill.h"
#include "generative-models/weight-set.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
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

// `qmeta`, when given: a `_quantization_metadata` JSON document for the
// header's __metadata__, the records' other home.
bool
write_st_(const fs::path& p, const std::vector<T>& ts,
          const std::string& qmeta = {})
{
  std::string hdr = "{";
  if (!qmeta.empty()) {
    std::string esc;
    for (char c : qmeta) {
      if (c == '"' || c == '\\') { esc += '\\'; }
      esc += c;
    }
    hdr += "\"__metadata__\":{\"_quantization_metadata\":\"" + esc +
           "\"}" + (ts.empty() ? "" : ",");
  }
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

namespace {

// comfy_kitchen's regular Hadamard as a MATRIX, built as int8_utils.py
// builds it -- kron(h, h4) from h4 up, scaled by 1/sqrt(g) -- so the
// radix-4 transform in shared/fp8.h is checked against the definition,
// not against itself.
std::vector<double>
regular_hadamard_(int g)
{
  static const int h4[4][4] = {
      {1, 1, 1, -1}, {1, 1, -1, 1}, {1, -1, 1, 1}, {-1, 1, 1, 1}};
  std::vector<double> h(16);
  for (int i = 0; i < 16; ++i) { h[(std::size_t)i] = h4[i / 4][i % 4]; }
  for (int n = 4; n < g; n *= 4) {
    std::vector<double> k((std::size_t)(n * 4) * (std::size_t)(n * 4));
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        for (int a = 0; a < 4; ++a) {
          for (int b = 0; b < 4; ++b) {
            k[(std::size_t)(i * 4 + a) * (std::size_t)(n * 4) +
              (std::size_t)(j * 4 + b)] =
                h[(std::size_t)i * (std::size_t)n + (std::size_t)j] *
                h4[a][b];
          }
        }
      }
    }
    h = std::move(k);
  }
  const double s = 1.0 / std::sqrt((double)g);
  for (double& v : h) { v *= s; }
  return h;
}

// `w` [n, k] times H^T group by group: comfy_kitchen's _rotate_weight,
// in double.
std::vector<double>
rotate_(const std::vector<double>& w, int n, int k, int g)
{
  const std::vector<double> h = regular_hadamard_(g);
  std::vector<double> out(w.size(), 0.0);
  for (int r = 0; r < n; ++r) {
    for (int g0 = 0; g0 < k; g0 += g) {
      for (int j = 0; j < g; ++j) {
        double acc = 0.0;
        for (int i = 0; i < g; ++i) {
          // (W H^T)[j] = sum_i W[i] H[j][i]
          acc += w[(std::size_t)r * k + g0 + i] *
                 h[(std::size_t)j * g + (std::size_t)i];
        }
        out[(std::size_t)r * k + g0 + j] = acc;
      }
    }
  }
  return out;
}

T
record_(const std::string& layer, const std::string& json)
{
  return T{layer + ".comfy_quant", "U8",
           {(std::int64_t)json.size()},
           std::vector<std::uint8_t>(json.begin(), json.end())};
}

// A layer as ComfyUI's int8_tensorwise writes it: `w` rotated by `g`
// (0: not), then a scale a row and codes rounded to it. Returns the
// tensors; `codes` and `scale` what was written.
std::vector<T>
int8_layer_(const std::string& layer, const std::vector<double>& w, int n,
            int k, int g, bool per_row, const std::string& rec,
            std::vector<std::int8_t>* codes, std::vector<float>* scale)
{
  const std::vector<double> q = g > 0 ? rotate_(w, n, k, g) : w;
  codes->assign((std::size_t)n * k, 0);
  scale->assign(per_row ? (std::size_t)n : 1, 0.0f);
  double whole = 0.0;
  for (double v : q) { whole = std::max(whole, std::abs(v)); }
  for (int r = 0; r < n; ++r) {
    double m = 0.0;
    for (int c = 0; c < k; ++c) {
      m = std::max(m, std::abs(q[(std::size_t)r * k + c]));
    }
    const float s = (float)((per_row ? m : whole) / 127.0);
    (*scale)[per_row ? (std::size_t)r : 0] = s;
    for (int c = 0; c < k; ++c) {
      const double v = std::round(q[(std::size_t)r * k + c] / s);
      (*codes)[(std::size_t)r * k + c] =
          (std::int8_t)std::clamp(v, -127.0, 127.0);
    }
  }
  T wt{layer + ".weight", "I8", {n, k},
       std::vector<std::uint8_t>((std::size_t)n * k)};
  std::memcpy(wt.bytes.data(), codes->data(), wt.bytes.size());
  T st{layer + ".weight_scale", "F32",
       per_row ? std::vector<std::int64_t>{n, 1} : std::vector<std::int64_t>{},
       std::vector<std::uint8_t>(scale->size() * 4)};
  std::memcpy(st.bytes.data(), scale->data(), st.bytes.size());
  return {wt, st, record_(layer, rec)};
}

std::vector<double>
random_(std::size_t n, unsigned seed)
{
  std::vector<double> v(n);
  unsigned s = seed;
  for (double& x : v) {
    s = s * 1664525u + 1013904223u;
    x = ((double)(s >> 8) / (double)(1u << 24) - 0.5) * 0.2;
  }
  return v;
}

// A layer as ComfyUI's asym_w4a8_int8 stores it, from random but VALID
// parts: nibble codes, E4M3 relative scales a group of 16 (16..120, so
// the int8 grid spans most of +-127), a symmetric codebook (or none) and a
// row scale. `g` the ConvRot group (0: none). What the tensors hold comes
// back for an independent decode.
struct W4a8Parts {
  std::vector<std::uint8_t> packed, s_rel;
  std::vector<float> codebook, s_channel;
};
std::vector<T>
w4a8_layer_(const std::string& layer, int n, int k, int g, bool codebook,
            unsigned seed, W4a8Parts* out)
{
  unsigned s = seed;
  auto next = [&]() {
    s = s * 1664525u + 1013904223u;
    return s >> 8;
  };
  W4a8Parts& p = *out;
  p.packed.resize((std::size_t)n * k / 2);
  for (auto& b : p.packed) { b = (std::uint8_t)(next() & 0xFFu); }
  p.s_rel.resize((std::size_t)n * k / 16);
  for (auto& b : p.s_rel) {
    // E4M3 exponents 11..13, any mantissa: 16 to 120.
    b = (std::uint8_t)(((11u + next() % 3u) << 3) | (next() & 7u));
  }
  p.codebook.clear();
  if (codebook) {
    for (int i = 0; i < 16; ++i) {
      p.codebook.push_back((float)((i - 7.5) / 7.5));
    }
  }
  p.s_channel.resize((std::size_t)n);
  for (auto& v : p.s_channel) {
    v = (float)(0.1 / 127.0 * (0.5 + (double)(next() % 1000) / 1000.0));
  }
  std::string rec = "{\"format\": \"asym_w4a8_int8\", \"group_size\": 16";
  if (g > 0) {
    rec += ", \"convrot\": true, \"convrot_groupsize\": " +
           std::to_string(g);
  }
  rec += "}";
  std::vector<T> ts;
  ts.push_back(T{layer + ".weight", "I8", {n, k / 2}, p.packed});
  ts.push_back(T{layer + ".weight_s_rel", "F8_E4M3", {n, k / 16}, p.s_rel});
  T sc{layer + ".weight_s_channel", "F32", {n},
       std::vector<std::uint8_t>((std::size_t)n * 4)};
  std::memcpy(sc.bytes.data(), p.s_channel.data(), sc.bytes.size());
  ts.push_back(std::move(sc));
  if (codebook) {
    T cb{layer + ".weight_codebook", "F32", {16},
         std::vector<std::uint8_t>(64)};
    std::memcpy(cb.bytes.data(), p.codebook.data(), 64);
    ts.push_back(std::move(cb));
  }
  ts.push_back(record_(layer, rec));
  return ts;
}

}  // namespace

// ComfyUI's int8 (comfy/ops.py's int8_tensorwise, comfy_kitchen's layout):
// codes and a scale a row -- or one for the tensor -- and, with ConvRot,
// the weight they stand for is (code * scale) * H^T a group at a time.
// Read only where a model asks for it; decoded on the host and widened on
// the GPU to the same bf16, which is comfy_kitchen's value to a rounding.
TEST(fp8_layouts, comfy_int8_and_convrot_read_as_their_weights)
{
  const fs::path p = tmp_("int8.safetensors");
  const int N = 6, K = 512;
  const std::vector<double> wa = random_((std::size_t)N * K, 1);
  const std::vector<double> wb = random_((std::size_t)N * 64, 2);
  const std::vector<double> wc = random_((std::size_t)N * 192, 3);
  std::vector<std::int8_t> ca, cb, cc;
  std::vector<float> sa, sb, sc;
  std::vector<T> ts;
  for (auto& t : int8_layer_(
           "a", wa, N, K, 256, true,
           "{\"format\": \"int8_tensorwise\", \"convrot\": true, "
           "\"convrot_groupsize\": 256}", &ca, &sa)) {
    ts.push_back(std::move(t));
  }
  for (auto& t : int8_layer_("b", wb, N, 64, 0, false,
                             "{\"format\": \"int8_tensorwise\"}", &cb,
                             &sb)) {
    ts.push_back(std::move(t));
  }
  // The group under `params`, as comfy/ops.py also reads it.
  for (auto& t : int8_layer_(
           "c", wc, N, 192, 64, true,
           "{\"format\": \"int8_tensorwise\", \"params\": {\"convrot\": "
           "true, \"convrot_groupsize\": 64}}", &cc, &sc)) {
    ts.push_back(std::move(t));
  }
  ASSERT_TRUE(write_st_(p, ts));
  auto w = MetalLlamaWeights::open(p.string());
  ASSERT_TRUE(w.has_value());
  if (!w.has_value()) { return; }

  // Refused by name where nobody asked -- every FP8 reader but H3.
  fp8::Layout l;
  std::string err;
  EXPECT_TRUE(!fp8::scan(*w, &l, &err));
  EXPECT_TRUE(err.find("int8_tensorwise") != std::string::npos);
  fp8::Weight one;
  EXPECT_TRUE(!fp8::resolve(*w, "a.weight", &one, &err));
  EXPECT_TRUE(!err.empty());

  // Read where asked.
  ASSERT_TRUE(fp8::scan(*w, &l, &err, /*int8=*/true));
  const fp8::Weight* a = l.find("a.weight");
  const fp8::Weight* b = l.find("b.weight");
  const fp8::Weight* c = l.find("c.weight");
  ASSERT_TRUE(a != nullptr && b != nullptr && c != nullptr);
  if (a == nullptr || b == nullptr || c == nullptr) { return; }
  EXPECT_TRUE(a->format == fp8::Format::kI8 && a->per_row &&
              a->rot_group == 256);
  EXPECT_TRUE(b->format == fp8::Format::kI8 && !b->per_row &&
              b->rot_group == 0);
  EXPECT_TRUE(c->rot_group == 64 && c->per_row);
  EXPECT_TRUE(l.is_aux("a.comfy_quant") && l.is_aux("a.weight_scale"));

  // The host decode against comfy_kitchen's dequantize_int8_convrot_
  // weight -- (q * s) @ H^T, H the matrix above -- in double, and then
  // against the weight the codes were made from.
  struct Case {
    const char* name;
    const std::vector<std::int8_t>* codes;
    const std::vector<float>* scale;
    const std::vector<double>* orig;
    int k, g;
  };
  for (const Case& cs : {Case{"a.weight", &ca, &sa, &wa, K, 256},
                         Case{"b.weight", &cb, &sb, &wb, 64, 0},
                         Case{"c.weight", &cc, &sc, &wc, 192, 64}}) {
    const std::size_t n = (std::size_t)N * cs.k;
    std::vector<double> deq(n);
    for (std::size_t i = 0; i < n; ++i) {
      const float s = (*cs.scale)[cs.scale->size() > 1 ? i / cs.k : 0];
      deq[i] = (double)(*cs.codes)[i] * (double)s;
    }
    const std::vector<double> want =
        cs.g > 0 ? rotate_(deq, N, cs.k, cs.g) : deq;
    std::vector<float> got(n);
    ASSERT_TRUE(fp8::decode_tensor_f32(*w, cs.name, got.data(), n, &err));
    double worst = 0.0, err2 = 0.0, ref2 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      // bf16's half ulp of the value, and a little for f32's sums.
      const double tol = std::abs(want[i]) * 0x1p-8 + 1e-7;
      worst = std::max(worst, std::abs(got[i] - want[i]) / tol);
      err2 += (got[i] - (*cs.orig)[i]) * (got[i] - (*cs.orig)[i]);
      ref2 += (*cs.orig)[i] * (*cs.orig)[i];
    }
    std::printf("[fp8_layouts] %s: off comfy_kitchen's value by %.2f of "
                "its bf16 rounding; %.4f from the weight\n", cs.name, worst,
                std::sqrt(err2 / ref2));
    EXPECT_TRUE(worst <= 1.0);
    EXPECT_TRUE(std::sqrt(err2 / ref2) < 0.01);
  }

  // The GPU widening: the host decode's bf16, bit for bit.
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { fs::remove(p); return; }
  fp8::Expander x;
  ASSERT_TRUE(x.load(mc));
  for (const Case& cs : {Case{"a.weight", &ca, &sa, &wa, K, 256},
                         Case{"b.weight", &cb, &sb, &wb, 64, 0},
                         Case{"c.weight", &cc, &sc, &wc, 192, 64}}) {
    const fp8::Weight* fw = l.find(cs.name);
    const std::size_t n = (std::size_t)N * cs.k;
    metal_compute::SharedBuffer codes = mc->make_shared_buffer(n);
    metal_compute::SharedBuffer dst = mc->make_shared_buffer(n * 2);
    ASSERT_TRUE(w->read_into(cs.name, codes.contents(), n));
    bool ok = true;
    metal_compute::SharedBuffer scale =
        fp8::scale_buffer(*w, *fw, (std::size_t)N, mc, &ok);
    ASSERT_TRUE(ok);
    fp8::Codes cd;
    cd.codes = &codes;
    cd.format = fw->format;
    cd.scale = &scale;
    cd.per_row = fw->per_row;
    cd.rot_group = fw->rot_group;
    metal_compute::CommandStream st = mc->make_command_stream();
    {
      metal_compute::ComputeEncoder enc = st.begin_compute();
      ASSERT_TRUE(x.encode(enc, cd, dst, (std::size_t)N, (std::size_t)cs.k));
    }
    ASSERT_TRUE(st.commit().wait_ok());
    std::vector<std::uint16_t> want(n);
    ASSERT_TRUE(fp8::decode_tensor_bf16(*w, cs.name, want.data(), n, &err));
    std::size_t diff = 0;
    for (std::size_t i = 0; i < n; ++i) {
      std::uint16_t g;
      std::memcpy(&g, static_cast<const std::uint8_t*>(dst.contents()) + 2 * i,
                  2);
      if (g != want[i]) { ++diff; }
    }
    std::printf("[fp8_layouts] %s: GPU and host bf16 differ in %zu of %zu\n",
                cs.name, diff, n);
    EXPECT_TRUE(diff == 0);
  }

  // A group the layout cannot hold is refused by name: 128 is no power
  // of four.
  std::vector<T> bad;
  std::vector<std::int8_t> cx;
  std::vector<float> sx;
  for (auto& t : int8_layer_(
           "d", wb, N, 64, 0, true,
           "{\"format\": \"int8_tensorwise\", \"convrot\": true, "
           "\"convrot_groupsize\": 128}", &cx, &sx)) {
    bad.push_back(std::move(t));
  }
  // A file of its own: `w` still maps the first.
  const fs::path pb = tmp_("int8-bad.safetensors");
  ASSERT_TRUE(write_st_(pb, bad));
  auto wd = MetalLlamaWeights::open(pb.string());
  ASSERT_TRUE(wd.has_value());
  if (wd.has_value()) {
    EXPECT_TRUE(!fp8::scan(*wd, &l, &err, /*int8=*/true));
    EXPECT_TRUE(err.find("ConvRot group 128") != std::string::npos);
  }
  fs::remove(pb);
  fs::remove(p);
}

// ComfyUI's W4A8 (comfy_kitchen's asym_w4a8_int8): nibble codes into a
// codebook (or code - 8), an E4M3 relative scale a group of 16, rounded to
// an int8 code -- the int8 grid ComfyUI's runtime multiplies -- then
// ConvRot and the row scale exactly as int8's. Read only where a model
// asks; decoded on the host against an independent restatement (the
// regular Hadamard as a matrix, in double), and widened on the GPU to the
// host's bf16 bit for bit. The asymmetric form and six-bit codes are
// refused by name.
TEST(fp8_layouts, comfy_w4a8_reads_as_its_weight)
{
  const fs::path p = tmp_("w4a8.safetensors");
  const int N = 6;
  struct Case {
    const char* name;
    int k, g;
    bool cb;
  };
  const Case cases[] = {{"a", 512, 256, true}, {"b", 192, 64, true},
                        {"c", 64, 0, false}};
  std::vector<W4a8Parts> parts(3);
  std::vector<T> ts;
  for (int i = 0; i < 3; ++i) {
    for (auto& t : w4a8_layer_(cases[i].name, N, cases[i].k, cases[i].g,
                               cases[i].cb, 11u + (unsigned)i,
                               &parts[(std::size_t)i])) {
      ts.push_back(std::move(t));
    }
  }
  ASSERT_TRUE(write_st_(p, ts));
  auto w = MetalLlamaWeights::open(p.string());
  ASSERT_TRUE(w.has_value());
  if (!w.has_value()) { return; }
  fp8::Layout l;
  std::string err;
  EXPECT_TRUE(!fp8::scan(*w, &l, &err));
  EXPECT_TRUE(err.find("asym_w4a8_int8") != std::string::npos);
  ASSERT_TRUE(fp8::scan(*w, &l, &err, /*int8=*/true));
  const fp8::Weight* a = l.find("a.weight");
  const fp8::Weight* c = l.find("c.weight");
  ASSERT_TRUE(a != nullptr && c != nullptr);
  if (a == nullptr || c == nullptr) { return; }
  EXPECT_TRUE(a->format == fp8::Format::kW4A8 && a->per_row &&
              a->rot_group == 256 && a->qgroup == 16 &&
              a->scale == "a.weight_s_channel" && !a->codebook.empty());
  EXPECT_TRUE(c->rot_group == 0 && c->codebook.empty());
  EXPECT_TRUE(l.is_aux("a.weight_s_rel") && l.is_aux("a.weight_codebook") &&
              l.is_aux("a.weight_s_channel"));

  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  fp8::Expander x;
  const bool gpu = mc != nullptr && x.load(mc) && x.w4a8_valid();
  for (int i = 0; i < 3; ++i) {
    const Case& cs = cases[i];
    const W4a8Parts& pp = parts[(std::size_t)i];
    const std::size_t n = (std::size_t)N * cs.k;
    // The restatement: each code's int8 value, scaled by its row, then
    // rotated back by H^T in double.
    std::vector<double> deq(n);
    for (std::size_t e = 0; e < n; ++e) {
      const std::uint8_t b = pp.packed[e / 2];
      const int code = (e & 1u) ? (b >> 4) : (b & 15);
      const double val = cs.cb ? pp.codebook[(std::size_t)code] : code - 8;
      const float prod = (float)val *
                         fp8::to_f32(pp.s_rel[e / 16], fp8::Format::kE4M3);
      const double q = std::clamp((double)std::nearbyint(prod), -127.0, 127.0);
      deq[e] = q * (double)pp.s_channel[e / (std::size_t)cs.k];
    }
    const std::vector<double> want =
        cs.g > 0 ? rotate_(deq, N, cs.k, cs.g) : deq;
    std::vector<float> got(n);
    const std::string nm = std::string(cs.name) + ".weight";
    ASSERT_TRUE(fp8::decode_tensor_f32(*w, nm, got.data(), n, &err));
    double worst = 0.0;
    for (std::size_t e = 0; e < n; ++e) {
      const double tol = std::abs(want[e]) * 0x1p-8 + 1e-9;
      worst = std::max(worst, std::abs(got[e] - want[e]) / tol);
    }
    std::size_t diff = 0;
    if (gpu) {
      const fp8::Weight* fw = l.find(nm);
      metal_compute::SharedBuffer codes = mc->make_shared_buffer(n / 2);
      metal_compute::SharedBuffer srel = mc->make_shared_buffer(n / 16);
      metal_compute::SharedBuffer dst = mc->make_shared_buffer(n * 2);
      metal_compute::SharedBuffer cbuf;
      std::memcpy(codes.contents(), pp.packed.data(), n / 2);
      std::memcpy(srel.contents(), pp.s_rel.data(), n / 16);
      if (cs.cb) {
        cbuf = mc->make_shared_buffer(64);
        std::memcpy(cbuf.contents(), pp.codebook.data(), 64);
      }
      bool ok = true;
      metal_compute::SharedBuffer scale =
          fp8::scale_buffer(*w, *fw, (std::size_t)N, mc, &ok);
      ASSERT_TRUE(ok);
      fp8::Codes cd;
      cd.codes = &codes;
      cd.format = fw->format;
      cd.scale = &scale;
      cd.per_row = true;
      cd.rot_group = fw->rot_group;
      cd.s_rel = &srel;
      cd.codebook = &cbuf;
      cd.qgroup = fw->qgroup;
      metal_compute::CommandStream st = mc->make_command_stream();
      {
        metal_compute::ComputeEncoder enc = st.begin_compute();
        ASSERT_TRUE(x.encode(enc, cd, dst, (std::size_t)N,
                             (std::size_t)cs.k));
      }
      ASSERT_TRUE(st.commit().wait_ok());
      std::vector<std::uint16_t> host(n);
      ASSERT_TRUE(fp8::decode_tensor_bf16(*w, nm, host.data(), n, &err));
      const auto* g16 = static_cast<const std::uint16_t*>(dst.contents());
      for (std::size_t e = 0; e < n; ++e) { diff += g16[e] != host[e]; }
    }
    std::printf("[fp8_layouts] W4A8 %s (K %d, ConvRot %d, %s): off the "
                "restatement by %.2f of its bf16 rounding; GPU and host "
                "differ in %zu of %zu%s\n", cs.name, cs.k, cs.g,
                cs.cb ? "codebook" : "code - 8", worst, diff, n,
                gpu ? "" : " (no GPU)");
    EXPECT_TRUE(worst <= 1.0);
    EXPECT_TRUE(diff == 0);
  }
  fs::remove(p);

  // Refused by name: the asymmetric correction, and six-bit codes.
  for (int bad = 0; bad < 2; ++bad) {
    W4a8Parts pp;
    std::vector<T> tb = w4a8_layer_("d", N, 64, 0, true, 5u, &pp);
    if (bad == 0) {
      tb.push_back(T{"d.weight_correction", "F32", {4, N},
                     std::vector<std::uint8_t>((std::size_t)4 * N * 4)});
    } else {
      tb.back() = record_("d", "{\"format\": \"w6a8_int8\", "
                               "\"group_size\": 16}");
    }
    const fs::path pb = tmp_("w4a8-bad.safetensors");
    ASSERT_TRUE(write_st_(pb, tb));
    auto wd = MetalLlamaWeights::open(pb.string());
    ASSERT_TRUE(wd.has_value());
    if (wd.has_value()) {
      EXPECT_TRUE(!fp8::scan(*wd, &l, &err, /*int8=*/true));
      EXPECT_TRUE(err.find(bad == 0 ? "weight_correction" : "w6a8_int8") !=
                  std::string::npos);
    }
    fs::remove(pb);
  }
}

// The token tier's CONVROT quantizer (quant_i8_tok_cr): an activation
// rotated by comfy_kitchen's regular Hadamard over each 256-block -- built
// here as the matrix, in double -- then quantized per row. The kernel
// takes it as a signed permutation of the Sylvester transform; every code
// must be the matrix's, but for a value within 1e-4 of a rounding tie.
TEST(fp8_layouts, convrot_token_quantizer_is_the_regular_hadamard)
{
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto lib = mc->load_library("affine_dequant_bf16");
  auto fn = lib.function("quant_i8_tok_cr");
  ASSERT_TRUE(fn.valid());
  if (!fn.valid()) { return; }
  const int M = 5, K = 768;
  const std::vector<double> r = random_((std::size_t)M * K, 9);
  metal_compute::SharedBuffer xb =
      mc->make_shared_buffer((std::size_t)M * K * 2);
  std::vector<double> x((std::size_t)M * K);
  auto* xp = static_cast<std::uint16_t*>(xb.contents());
  for (std::size_t i = 0; i < x.size(); ++i) {
    // An outlier channel a row, as an activation has.
    const float v = (float)(r[i] * ((i % (std::size_t)K) == 37 ? 40.0 : 5.0));
    xp[i] = rne_(v);
    const std::uint32_t u = (std::uint32_t)xp[i] << 16;
    float f;
    std::memcpy(&f, &u, 4);
    x[i] = f;
  }
  metal_compute::SharedBuffer qb = mc->make_shared_buffer((std::size_t)M * K);
  metal_compute::SharedBuffer sb = mc->make_shared_buffer((std::size_t)M * 4);
  {
    metal_compute::CommandStream st = mc->make_command_stream();
    {
      metal_compute::ComputeEncoder enc = st.begin_compute();
      enc.set_function(fn);
      enc.set_buffer(0, xb);
      enc.set_buffer(1, qb);
      enc.set_buffer(2, sb);
      enc.set_constant(3, K);
      // The rotated token quantizers' geometry: a simdgroup a 256-block
      // or two, one threadgroup a row.
      const unsigned T = 32u * (unsigned)std::max(1, (K / 256 + 1) / 2);
      enc.dispatch({T, (unsigned)M, 1}, {T, 1, 1});
    }
    ASSERT_TRUE(st.commit().wait_ok());
  }
  const std::vector<double> rot = rotate_(x, M, K, 256);
  const auto* q = static_cast<const std::int8_t*>(qb.contents());
  const auto* sc = static_cast<const float*>(sb.contents());
  int off = 0, ties = 0;
  double sworst = 0.0;
  for (int m = 0; m < M; ++m) {
    double am = 0.0;
    for (int k = 0; k < K; ++k) {
      am = std::max(am, std::abs(rot[(std::size_t)m * K + k]));
    }
    sworst = std::max(sworst, std::abs(sc[m] - am / 127.0) / (am / 127.0));
    for (int k = 0; k < K; ++k) {
      const double t = rot[(std::size_t)m * K + k] * 127.0 / am;
      const int want = (int)std::clamp(std::nearbyint(t), -127.0, 127.0);
      if (q[(std::size_t)m * K + k] != want) {
        const double frac = t - std::floor(t);
        if (std::abs(frac - 0.5) < 1e-4) { ++ties; } else { ++off; }
      }
    }
  }
  std::printf("[fp8_layouts] ConvRot token quantizer: %d of %d codes off "
              "the matrix's (%d at a rounding tie); row scale within %.2g\n",
              off, M * K, ties, sworst);
  EXPECT_TRUE(off == 0 && sworst < 1e-5);
}

// W4A8 into the token layout's int8 codes (w4a8_to_i8): each the int8
// code fp8::w4a8_code gives, exactly.
TEST(fp8_layouts, w4a8_to_i8_is_the_int8_code)
{
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto lib = mc->load_library("affine_dequant_bf16");
  auto fn = lib.function("w4a8_to_i8");
  ASSERT_TRUE(fn.valid());
  if (!fn.valid()) { return; }
  const int N = 8, K = 512;
  for (int cbk = 0; cbk < 2; ++cbk) {
    W4a8Parts pp;
    (void)w4a8_layer_("x", N, K, 256, cbk != 0, 21u + (unsigned)cbk, &pp);
    const std::size_t n = (std::size_t)N * K;
    metal_compute::SharedBuffer w = mc->make_shared_buffer(n / 2);
    metal_compute::SharedBuffer sr = mc->make_shared_buffer(n / 16);
    metal_compute::SharedBuffer cb = mc->make_shared_buffer(64);
    metal_compute::SharedBuffer q = mc->make_shared_buffer(n);
    std::memcpy(w.contents(), pp.packed.data(), n / 2);
    std::memcpy(sr.contents(), pp.s_rel.data(), n / 16);
    if (cbk != 0) { std::memcpy(cb.contents(), pp.codebook.data(), 64); }
    metal_compute::CommandStream st = mc->make_command_stream();
    {
      metal_compute::ComputeEncoder enc = st.begin_compute();
      enc.set_function(fn);
      enc.set_buffer(0, w);
      enc.set_buffer(1, q);
      enc.set_constant(2, (std::uint32_t)n);
      enc.set_buffer(3, sr);
      enc.set_buffer(4, cb);
      enc.set_constant(5, 16u);
      enc.set_constant(6, (std::uint32_t)cbk);
      enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
    }
    ASSERT_TRUE(st.commit().wait_ok());
    const auto* qp = static_cast<const std::int8_t*>(q.contents());
    std::size_t off = 0;
    for (std::size_t e = 0; e < n; ++e) {
      const std::uint8_t b = pp.packed[e / 2];
      const std::int32_t want = fp8::w4a8_code(
          (e & 1u) ? (std::uint8_t)(b >> 4) : b,
          fp8::to_f32(pp.s_rel[e / 16], fp8::Format::kE4M3),
          cbk != 0 ? pp.codebook.data() : nullptr);
      off += qp[e] != want;
    }
    EXPECT_TRUE(off == 0);
  }
}

// A MiniMax-H3 DiT as ComfyUI SAVES one -- a community fine-tune, here in
// int8 -- opens under H3's own names and reads its qkv FLAT, its
// dimensions off its tensors. The same file without the prefix is no
// longer known to be ComfyUI's, and a fused qkv of unknown grouping is
// refused rather than guessed.
TEST(fp8_layouts, h3_comfyui_saved_reads_flat)
{
  const fs::path p = tmp_("h3saved.safetensors");
  const int hidden = 256, heads = 2, ffn = 512;
  auto build = [&](const std::string& pre) {
    std::vector<T> ts = {
        tensor_(pre + "condition_proj.weight", "BF16", {hidden, 64}, 1),
        tensor_(pre + "video_patch_proj.weight", "BF16", {hidden, 96}, 2),
        tensor_(pre + "audio_patch_proj.weight", "BF16", {hidden, 32}, 3),
        tensor_(pre + "time_embedder.proj_in.weight", "BF16", {hidden, 256},
                4),
        tensor_(pre + "time_embedder.proj_out.weight", "BF16", {128, hidden},
                5),
        tensor_(pre + "rope.inv_freq", "F32", {16}, 6),
        tensor_(pre + "token_refiner.blocks.0.norm1.weight", "BF16",
                {hidden}, 7),
    };
    for (int L = 0; L < 2; ++L) {
      const std::string b = pre + "blocks." + std::to_string(L) + ".";
      ts.push_back(tensor_(b + "norm1.weight", "BF16", {hidden}, 10 + L));
      ts.push_back(tensor_(b + "mlp.fc2.weight", "BF16", {hidden, ffn},
                           20 + L));
      std::vector<std::int8_t> c;
      std::vector<float> s;
      for (auto& t : int8_layer_(
               b + "attn.qkv_proj",
               random_((std::size_t)3 * 128 * heads * hidden, 30 + L),
               3 * 128 * heads, hidden, 256, true,
               "{\"format\": \"int8_tensorwise\", \"convrot\": true, "
               "\"convrot_groupsize\": 256}", &c, &s)) {
        ts.push_back(std::move(t));
      }
    }
    return ts;
  };
  ASSERT_TRUE(write_st_(p, build("model.diffusion_model.")));
  EXPECT_TRUE(minimax_h3::is_comfy_saved_dit_file(p.string()));
  auto w = MetalLlamaWeights::open_model(p.string());
  ASSERT_TRUE(w.has_value());
  if (!w.has_value()) { return; }
  EXPECT_TRUE(w->has("blocks.0.attn.qkv_proj.weight") &&
              w->has("blocks.0.attn.qkv_proj.comfy_quant") &&
              w->has("rope.inv_freq"));
  EXPECT_TRUE(!w->has("model.diffusion_model.condition_proj.weight"));
  fp8::Layout l;
  std::string err;
  ASSERT_TRUE(fp8::scan(*w, &l, &err, /*int8=*/true));
  const fp8::Weight* q = l.find("blocks.1.attn.qkv_proj.weight");
  EXPECT_TRUE(q != nullptr && q->format == fp8::Format::kI8 &&
              q->rot_group == 256);

  MetalMiniMaxH3Transformer::Config cfg;
  const bool ok =
      MetalMiniMaxH3Transformer::config_from_json(p.string(), cfg, &err);
  if (!ok) { std::printf("[fp8_layouts] %s\n", err.c_str()); }
  ASSERT_TRUE(ok);
  EXPECT_TRUE(!cfg.qkv_per_head);
  EXPECT_TRUE(cfg.hidden == hidden && cfg.n_heads == heads &&
              cfg.n_layers == 2 && cfg.n_refiner == 1 && cfg.ffn == ffn &&
              cfg.video_channels == 24 && cfg.audio_channels == 32 &&
              cfg.text_dim == 64 && cfg.time_dim == 128);

  // The records in the HEADER instead (`_quantization_metadata`, keyed
  // by the file's own, prefixed, layer names): the prefix dropped from the
  // tensors at open, the records are still found -- and a capitalized
  // extension is the same file.
  const fs::path pm = tmp_("h3saved-md.SafeTensors");
  {
    std::vector<T> ts;
    for (T& t : build("model.diffusion_model.")) {
      if (!t.name.ends_with(".comfy_quant")) { ts.push_back(std::move(t)); }
    }
    std::string md = "{\"format_version\": \"1.0\", \"layers\": {";
    for (int L = 0; L < 2; ++L) {
      md += std::string(L > 0 ? ", " : "") +
            "\"model.diffusion_model.blocks." + std::to_string(L) +
            ".attn.qkv_proj\": {\"format\": \"int8_tensorwise\", "
            "\"convrot\": true, \"convrot_groupsize\": 256}";
    }
    md += "}}";
    ASSERT_TRUE(write_st_(pm, ts, md));
  }
  EXPECT_TRUE(minimax_h3::is_comfy_saved_dit_file(pm.string()));
  auto wm = MetalLlamaWeights::open_model(pm.string());
  ASSERT_TRUE(wm.has_value());
  if (wm.has_value()) {
    fp8::Layout lm;
    const bool sok = fp8::scan(*wm, &lm, &err, /*int8=*/true);
    if (!sok) { std::printf("[fp8_layouts] %s\n", err.c_str()); }
    EXPECT_TRUE(sok);
    const fp8::Weight* qm = lm.find("blocks.1.attn.qkv_proj.weight");
    EXPECT_TRUE(qm != nullptr && qm->format == fp8::Format::kI8 &&
                qm->rot_group == 256 && qm->per_row);
  }
  EXPECT_TRUE(
      MetalMiniMaxH3Transformer::config_from_json(pm.string(), cfg, &err) &&
      !cfg.qkv_per_head);
  fs::remove(pm);

  // One tensor outside the prefix: nothing is renamed.
  const fs::path px = tmp_("h3saved-stray.safetensors");
  std::vector<T> mixed = build("model.diffusion_model.");
  mixed.push_back(tensor_("stray.weight", "BF16", {2, 2}, 40));
  ASSERT_TRUE(write_st_(px, mixed));
  EXPECT_TRUE(!minimax_h3::is_comfy_saved_dit_file(px.string()));
  auto m = MetalLlamaWeights::open_model(px.string());
  ASSERT_TRUE(m.has_value());
  if (m.has_value()) {
    EXPECT_TRUE(m->has("model.diffusion_model.condition_proj.weight"));
  }
  fs::remove(px);

  // No prefix, no config: its grouping is unknown, so it is not read.
  const fs::path pn = tmp_("h3saved-bare.safetensors");
  ASSERT_TRUE(write_st_(pn, build("")));
  EXPECT_TRUE(!minimax_h3::is_comfy_saved_dit_file(pn.string()));
  EXPECT_TRUE(
      !MetalMiniMaxH3Transformer::config_from_json(pn.string(), cfg, &err));
  fs::remove(pn);
  fs::remove(p);
}

namespace {

// A MiniMax-H3 DiT ComfyUI saved IN INT8, end to end. Its projections
// are int8 records -- ConvRot at groups of 256 and, the AdaLN's, of 64;
// one plain per row and one plain per tensor -- and its twin holds the
// SAME weights decoded to bf16 (the host decode, which the GPU widening
// matches bit for bit). The int8 file must forward BYTE-IDENTICAL to the
// twin preloaded, streamed through the slot pair, streamed with no
// slots, streamed and promoted (residency clones a slot's block), and
// with its AdaLN baked, preloaded and streamed. Each hold reaches the
// rotation by its own path -- linear_, the slot refill, the block clone,
// the bake -- and a weight read without its rotation is noise, so any one
// path that drops it fails here. Tuners and the GEMM route pinned: this
// compares bytes.
//
// `pruned`: ComfyUI's PRUNED form instead (Config::curve_grid) -- no
// timestep MLP, a 1025 x 8 `adaln_t_table`, and every AdaLN projection
// [D, 8] F16 (as the pruned files store it) in BOTH files, held in the
// trunk, so the holds are the projections' alone; and there is no bake.
struct H3Twin {
  struct Out {
    std::vector<std::uint16_t> v, a, v0;   // v0: the first forward's video
    int kept = 0;
    std::uint64_t native = 0;   // GEMMs on ConvRot codes as stored
    std::size_t deq = 0;        // the dequant scratch the forwards grew
  };
  bool ran = false, finite = false, bake_refused = false;
  MetalMiniMaxH3Transformer::Config cfg;
  Out dense, pre, str, flat, grown, dense_baked, baked, str_baked;
  // Pruned: the model's curve points against curve_coords' own.
  bool points_match = false;
  int layers = 0;
  // The widening the plan books for the int8 file (fp8_widen_bytes off
  // its headers) and what the loaded model reports.
  std::size_t widen_booked = 0, widen_loaded = 0;
};

H3Twin
h3_int8_twin_(bool pruned, bool w4a8 = false, int tier = 0)
{
  H3Twin res;
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return res; }
  fp8::Expander probe;
  if (!probe.load(mc) || !probe.int8_valid()) { return res; }
  struct Env {
    const char* k;
    Env(const char* key, const char* v) : k(key) { ::setenv(key, v, 1); }
    ~Env() { ::unsetenv(k); }
  };
  Env noqmm("VPIPE_H3_NO_QMM_AUTOTUNE", "1");
  Env notile("VPIPE_H3_NO_ATTN_TILE_AUTOTUNE", "1");
  // An int8 tier (`tier`, Config::i8_tier) reaches the int8 GEMM at this
  // test's 72 rows only with its row floor lowered.
  std::unique_ptr<Env> minm;
  if (tier > 0) { minm = std::make_unique<Env>("VPIPE_I8_GEMM_MIN_M", "16"); }

  // hidden 256, 2 heads of 128, ffn 512, text 256, time 256 -> 128 (or
  // pruned, the 1025 x 8 curve); four main blocks (two reach the slot
  // refill) and two refiner blocks.
  const int H = 256, A = 256, F = 512, TXT = 256, FREQ = 256, TH = 256,
            TD = pruned ? 8 : 128, LAYERS = 4, REFINER = 2, GRID = 1025;
  res.layers = LAYERS;
  const std::string P = "model.diffusion_model.";
  std::vector<T> q, d;   // the int8 file, its dense twin
  unsigned seed = 100;
  auto both = [&](T t) {
    t.name = P + t.name;
    q.push_back(t);
    d.push_back(std::move(t));
  };
  auto f32 = [&](const std::string& nm, std::vector<std::int64_t> shape,
                 double center, double spread) {
    std::size_t n = 1;
    for (auto x : shape) { n *= (std::size_t)x; }
    const std::vector<double> r = random_(n, ++seed);
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) {
      v[i] = (float)(center + r[i] * spread * 10.0);
    }
    T t{nm, "F32", std::move(shape), std::vector<std::uint8_t>(n * 4)};
    std::memcpy(t.bytes.data(), v.data(), n * 4);
    return t;
  };
  auto bf16 = [&](const std::string& nm, std::vector<std::int64_t> shape,
                  double center, double spread) {
    T t = f32(nm, shape, center, spread);
    const std::size_t n = t.bytes.size() / 4;
    std::vector<std::uint8_t> b(n * 2);
    for (std::size_t i = 0; i < n; ++i) {
      float f;
      std::memcpy(&f, &t.bytes[i * 4], 4);
      const std::uint16_t h = rne_(f);
      std::memcpy(&b[i * 2], &h, 2);
    }
    t.dtype = "BF16";
    t.bytes = std::move(b);
    return t;
  };
  auto f16 = [&](const std::string& nm, std::vector<std::int64_t> shape,
                 double center, double spread) {
    T t = f32(nm, shape, center, spread);
    const std::size_t n = t.bytes.size() / 4;
    std::vector<std::uint8_t> b(n * 2);
    for (std::size_t i = 0; i < n; ++i) {
      float f;
      std::memcpy(&f, &t.bytes[i * 4], 4);
      const _Float16 h = (_Float16)f;
      std::memcpy(&b[i * 2], &h, 2);
    }
    t.dtype = "F16";
    t.bytes = std::move(b);
    return t;
  };
  // An int8 projection in the file, and its decode in the twin. g > 0:
  // ConvRot at that group.
  auto lin = [&](const std::string& nm, int n, int k, int g, bool per_row) {
    // W4A8 in place of the ConvRot-256 int8 projections; the twin holds
    // its host decode (the GPU widening's bf16, test comfy_w4a8_...).
    if (w4a8 && g == 256) {
      W4a8Parts pp;
      for (auto& t : w4a8_layer_(P + nm, n, k, g, true, ++seed, &pp)) {
        q.push_back(std::move(t));
      }
      std::vector<std::uint16_t> b((std::size_t)n * k);
      fp8::decode_w4a8_bf16(pp.packed.data(), pp.s_rel.data(),
                            pp.codebook.data(), pp.s_channel, (std::size_t)n,
                            (std::size_t)k, 16, g, b.data());
      T t{P + nm + ".weight", "BF16", {n, k},
          std::vector<std::uint8_t>(b.size() * 2)};
      std::memcpy(t.bytes.data(), b.data(), t.bytes.size());
      d.push_back(std::move(t));
      return;
    }
    const std::string rec =
        g > 0 ? "{\"format\": \"int8_tensorwise\", \"convrot\": true, "
                "\"convrot_groupsize\": " + std::to_string(g) + "}"
              : std::string("{\"format\": \"int8_tensorwise\"}");
    std::vector<std::int8_t> c;
    std::vector<float> sc;
    for (auto& t : int8_layer_(P + nm, random_((std::size_t)n * k, ++seed),
                               n, k, g, per_row, rec, &c, &sc)) {
      q.push_back(std::move(t));
    }
    std::vector<std::uint16_t> b((std::size_t)n * k);
    fp8::decode_bf16(reinterpret_cast<const std::uint8_t*>(c.data()),
                     b.data(), (std::size_t)n, (std::size_t)k,
                     fp8::Format::kI8, sc, g);
    T t{P + nm + ".weight", "BF16", {n, k},
        std::vector<std::uint8_t>(b.size() * 2)};
    std::memcpy(t.bytes.data(), b.data(), t.bytes.size());
    d.push_back(std::move(t));
  };

  both(f32("video_patch_proj.weight", {H, 96}, 0.0, 0.01));
  both(f32("video_patch_proj.bias", {H}, 0.0, 0.01));
  both(f32("audio_patch_proj.weight", {H, 32}, 0.0, 0.01));
  both(f32("audio_patch_proj.bias", {H}, 0.0, 0.01));
  lin("condition_proj", H, TXT, 256, true);
  both(bf16("condition_proj.bias", {H}, 0.0, 0.01));
  // The pruned curve: smooth in t, each column its own phase, so a point
  // read at the wrong t or column is a different number.
  std::vector<float> table;
  if (pruned) {
    table.resize((std::size_t)GRID * 8);
    for (int g = 0; g < GRID; ++g) {
      for (int k = 0; k < 8; ++k) {
        table[(std::size_t)g * 8 + k] = std::sin(
            0.004f * (float)g * (float)(k + 1) + 0.7f * (float)k);
      }
    }
    T t{"adaln_t_table", "F32", {GRID, 8},
        std::vector<std::uint8_t>(table.size() * 4)};
    std::memcpy(t.bytes.data(), table.data(), t.bytes.size());
    both(std::move(t));
  } else {
    both(f32("time_embedder.proj_in.weight", {TH, FREQ}, 0.0, 0.01));
    both(f32("time_embedder.proj_in.bias", {TH}, 0.0, 0.01));
    both(f32("time_embedder.proj_out.weight", {TD, TH}, 0.0, 0.01));
    both(f32("time_embedder.proj_out.bias", {TD}, 0.0, 0.01));
  }
  {
    T t{"rope.inv_freq", "F32", {16}, std::vector<std::uint8_t>(64)};
    for (int i = 0; i < 16; ++i) {
      const float f = 1.0f / std::pow(10000.0f, (float)(2 * i) / 32.0f);
      std::memcpy(&t.bytes[(std::size_t)i * 4], &f, 4);
    }
    both(std::move(t));
  }
  for (int L = 0; L < LAYERS; ++L) {
    const std::string b = "blocks." + std::to_string(L) + ".";
    both(bf16(b + "norm1.weight", {H}, 1.0, 0.01));
    both(bf16(b + "norm2.weight", {H}, 1.0, 0.01));
    both(bf16(b + "attn.q_norm.weight", {128}, 1.0, 0.01));
    both(bf16(b + "attn.k_norm.weight", {128}, 1.0, 0.01));
    lin(b + "attn.qkv_proj", 3 * A, H, 256, true);
    lin(b + "attn.out_proj", H, A, 256, true);
    lin(b + "mlp.fc1", 2 * F, H, 256, true);
    lin(b + "mlp.fc2", H, F, 256, true);
    if (pruned) {
      both(f16(b + "adaln_proj.linear.weight", {18 * H, TD}, 0.0, 0.01));
      both(f16(b + "adaln_proj.linear.bias", {18 * H}, 0.0, 0.01));
    } else {
      lin(b + "adaln_proj.linear", 18 * H, TD, 64, true);
      both(bf16(b + "adaln_proj.linear.bias", {18 * H}, 0.0, 0.01));
    }
  }
  for (int L = 0; L < REFINER; ++L) {
    const std::string b = "token_refiner.blocks." + std::to_string(L) + ".";
    both(bf16(b + "norm1.weight", {H}, 1.0, 0.01));
    both(bf16(b + "norm2.weight", {H}, 1.0, 0.01));
    both(bf16(b + "attn.q_norm.weight", {128}, 1.0, 0.01));
    both(bf16(b + "attn.k_norm.weight", {128}, 1.0, 0.01));
    lin(b + "attn.qkv_proj", 3 * A, H, 256, true);
    lin(b + "attn.out_proj", H, A, 0, false);   // plain, one scale
    lin(b + "mlp.fc1", 2 * F, H, 256, true);
    lin(b + "mlp.fc2", H, F, 0, true);          // plain, a scale a row
  }
  both(bf16("token_refiner.final_norm.weight", {H}, 1.0, 0.01));
  both(bf16("final_layer.norm.weight", {H}, 1.0, 0.01));
  if (pruned) {
    both(f16("final_layer.adaln_proj.linear.weight", {2 * H, TD}, 0.0,
             0.01));
    both(f16("final_layer.adaln_proj.linear.bias", {2 * H}, 0.0, 0.01));
  } else {
    lin("final_layer.adaln_proj.linear", 2 * H, TD, 64, true);
    both(bf16("final_layer.adaln_proj.linear.bias", {2 * H}, 0.0, 0.01));
  }
  both(f32("final_layer.video_out.weight", {96, H}, 0.0, 0.01));
  both(f32("final_layer.video_out.bias", {96}, 0.0, 0.01));
  both(f32("final_layer.audio_out.weight", {32, H}, 0.0, 0.01));
  both(f32("final_layer.audio_out.bias", {32}, 0.0, 0.01));

  const std::string tag =
      std::string(pruned ? "pruned-" : "") + (w4a8 ? "w4a8" : "int8");
  const fs::path pq = tmp_("h3-" + tag + ".safetensors");
  const fs::path pd = tmp_("h3-" + tag + "-bf16.safetensors");
  if (!write_st_(pq, q) || !write_st_(pd, d)) { return res; }
  MetalMiniMaxH3Transformer::Config& base = res.cfg;
  std::string err;
  if (!MetalMiniMaxH3Transformer::config_from_json(pq.string(), base,
                                                   &err)) {
    std::printf("[fp8_layouts] %s\n", err.c_str());
    return res;
  }
  res.widen_booked =
      MetalMiniMaxH3Transformer::fp8_widen_bytes(pq.string(), base);

  namespace h3 = vpipe::genai::minimax_h3;
  const int latf = 2, lath = 8, latw = 8, naud = 8, ntext = 8;
  h3::PackedLayout lay;
  const std::vector<int> tags((std::size_t)ntext, h3::kTextTag);
  if (!h3::build_packed_sequence(tags, latf, lath, latw, naud,
                                 base.patch_h, base.patch_w,
                                 h3::kAudioChannels, {h3::Anchor::kFirst},
                                 &lay)) {
    return res;
  }
  auto input = [&](std::size_t n, float k) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) {
      v[i] = std::sin((float)i * k) * 0.5f;
    }
    metal_compute::SharedBuffer b = mc->make_shared_buffer(n * 2);
    auto* p = static_cast<std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) { p[i] = rne_(v[i]); }
    return b;
  };
  const metal_compute::SharedBuffer vb = input(
      lay.video_indices.size() * (std::size_t)base.video_patch_elems(),
      0.017f);
  const metal_compute::SharedBuffer ab = input(
      (std::size_t)lay.num_audio_rows * (std::size_t)base.audio_channels,
      0.031f);
  const metal_compute::SharedBuffer tb =
      input((std::size_t)ntext * base.text_dim, 0.005f);
  const float kV[] = {0.9f, 0.5f};
  const float kA[] = {0.8f, 0.45f};
  std::vector<std::vector<float>> sched, uniqs(2);
  std::vector<std::vector<int>> ridx(2);
  for (int i = 0; i < 2; ++i) {
    h3::build_row_timesteps(lay, kV[i], kA[i], 1.0f,
                            &uniqs[(std::size_t)i], &ridx[(std::size_t)i]);
    sched.push_back(uniqs[(std::size_t)i]);
  }
  using Out = H3Twin::Out;
  // Two forwards; the second is compared, so a grown arm runs it on
  // promoted blocks.
  auto run = [&](const fs::path& file, bool stream, bool slots, bool grow,
                 bool bake) -> Out {
    Out o;
    std::unique_ptr<Env> ns;
    if (!slots) { ns = std::make_unique<Env>("VPIPE_H3_NO_SLOTS", "1"); }
    MetalMiniMaxH3Transformer::Config cfg = base;
    if (tier > 0) {
      cfg.i8_gemm = true;
      cfg.i8_tier = tier;
    }
    auto m = MetalMiniMaxH3Transformer::load(file.string(), mc, cfg, stream);
    if (m == nullptr) {
      std::printf("[fp8_layouts] %s: load failed\n", file.c_str());
      return o;
    }
    m->set_gemm_route(MetalMiniMaxH3Transformer::GemmRoute::kSteelBm32);
    if (grow) {
      m->set_residency_reserve(test_residency_reserve(2ull << 30));
    }
    std::string berr;
    if (bake && !m->bake_adaln(sched, &berr)) {
      std::printf("[fp8_layouts] bake: %s\n", berr.c_str());
      return o;
    }
    // Pruned: the bake is refused, and the model's curve points are the
    // ones curve_coords reads off the table.
    if (pruned && file == pd && !bake) {
      std::string perr;
      res.bake_refused = !m->bake_adaln(sched, &perr) && !perr.empty();
      std::vector<float> got, want((std::size_t)uniqs[0].size() * 8);
      MetalMiniMaxH3Transformer::curve_coords(table, GRID, uniqs[0],
                                              want.data());
      res.points_match = m->time_embedding(uniqs[0], nullptr, &got) &&
                         got == want;
    }
    MetalMiniMaxH3Transformer::Velocity out;
    for (int i = 0; i < 2; ++i) {
      MetalMiniMaxH3Transformer::Step st;
      st.video = &vb;
      st.audio = &ab;
      st.text = &tb;
      st.layout = &lay;
      st.timesteps = &uniqs[(std::size_t)i];
      st.row_timestep_index = &ridx[(std::size_t)i];
      if (bake) { st.schedule_index = i; }
      std::string ferr;
      out = m->forward(st, &ferr);
      if (out.empty()) {
        std::printf("[fp8_layouts] forward: %s\n", ferr.c_str());
        return o;
      }
      if (i == 0) {
        const auto* p = static_cast<const std::uint16_t*>(
            out.video.contents());
        o.v0.assign(p, p + out.video.byte_size() / 2);
      }
    }
    auto words = [](const metal_compute::SharedBuffer& b) {
      const auto* p = static_cast<const std::uint16_t*>(b.contents());
      return std::vector<std::uint16_t>(p, p + b.byte_size() / 2);
    };
    o.v = words(out.video);
    o.a = words(out.audio);
    o.kept = m->resident_block_count();
    o.native = m->convrot_native_gemms();
    o.deq = m->dequant_scratch_bytes();
    if (file == pq) { res.widen_loaded = m->fp8_widen_bytes(); }
    return o;
  };
  auto finite_moving = [](const Out& x) {
    bool any = false;
    for (std::uint16_t w : x.v) {
      if ((w & 0x7f80u) == 0x7f80u) { return false; }
      any = any || (w & 0x7fffu) != 0;
    }
    return any;
  };
  res.dense = run(pd, false, true, false, false);
  res.pre = run(pq, false, true, false, false);
  res.str = run(pq, true, true, false, false);
  res.flat = run(pq, true, false, false, false);
  res.grown = run(pq, true, true, true, false);
  if (!pruned) {
    res.dense_baked = run(pd, false, true, false, true);
    res.baked = run(pq, false, true, false, true);
    res.str_baked = run(pq, true, true, false, true);
  }
  res.finite = !res.dense.v.empty() && finite_moving(res.dense);
  res.ran = true;
  fs::remove(pq);
  fs::remove(pd);
  return res;
}

bool
same_out_(const H3Twin::Out& x, const H3Twin::Out& y)
{
  return !x.v.empty() && x.v == y.v && x.a == y.a;
}

const char*
say_(const H3Twin::Out& x, const H3Twin::Out& ref)
{
  return same_out_(x, ref) ? "IDENTICAL" : "DIFFERENT";
}

// The plan's widening figure is the scratch every int8 arm grew: off the
// headers == off the load == what the widening allocated, each hold.
bool
widen_as_booked_(const H3Twin& r)
{
  bool ok = r.widen_booked > 0 && r.widen_booked == r.widen_loaded;
  for (const H3Twin::Out* o :
       {&r.pre, &r.str, &r.flat, &r.grown, &r.baked, &r.str_baked}) {
    if (!o->v.empty()) { ok = ok && o->deq == r.widen_booked; }
  }
  std::printf("[fp8_layouts] widening: booked %zu B, loaded %zu B, grown "
              "%zu / %zu / %zu / %zu B (preloaded / streamed / no slots / "
              "promoted)\n", r.widen_booked, r.widen_loaded, r.pre.deq,
              r.str.deq, r.flat.deq, r.grown.deq);
  return ok;
}

}  // namespace

TEST(fp8_layouts, h3_comfy_int8_forwards_as_its_dense_twin)
{
  const H3Twin r = h3_int8_twin_(false);
  if (!r.ran) { return; }
  EXPECT_TRUE(!r.cfg.qkv_per_head && r.cfg.hidden == 256 &&
              r.cfg.n_layers == 4 && r.cfg.n_refiner == 2 &&
              r.cfg.time_dim == 128 && !r.cfg.pruned());
  ASSERT_TRUE(r.finite);
  std::printf("[fp8_layouts] H3 int8 against its bf16 twin: preloaded %s, "
              "streamed %s, no slots %s, promoted (%d of %d) %s; baked: "
              "preloaded %s, streamed %s\n",
              say_(r.pre, r.dense), say_(r.str, r.dense),
              say_(r.flat, r.dense), r.grown.kept, r.layers,
              say_(r.grown, r.dense), say_(r.baked, r.dense_baked),
              say_(r.str_baked, r.dense_baked));
  EXPECT_TRUE(same_out_(r.pre, r.dense));
  EXPECT_TRUE(same_out_(r.str, r.dense));
  EXPECT_TRUE(same_out_(r.flat, r.dense));
  EXPECT_TRUE(r.grown.kept > 0 && same_out_(r.grown, r.dense));
  EXPECT_TRUE(!r.dense_baked.v.empty() &&
              same_out_(r.baked, r.dense_baked));
  EXPECT_TRUE(same_out_(r.str_baked, r.dense_baked));
  // The released AdaLN (2,688 wide in the real model) is the widest it
  // widens -- past the four sequence projections the plan used to count.
  EXPECT_TRUE(widen_as_booked_(r));
  EXPECT_TRUE(r.widen_booked == (std::size_t)18 * 256 * 128 * 2);
  // And the forward's figure carries it, with or without matrix cores'
  // dequant term (narrower here: the four sequence projections).
  using DiT = MetalMiniMaxH3Transformer;
  for (const bool mx : {false, true}) {
    const std::size_t a =
        DiT::scratch_bytes(r.cfg, 4096, 64, 4, mx, false, 0, 0, 0);
    const std::size_t b = DiT::scratch_bytes(r.cfg, 4096, 64, 4, mx, false,
                                             0, 0, r.widen_booked);
    const std::size_t four = mx ? (std::size_t)2 * 512 * 256 * 2 : 0;
    EXPECT_TRUE(b - a == r.widen_booked - four);
  }
}

// The same in ComfyUI's PRUNED form (the community's "pruned" H3 files,
// int8 ConvRot or bf16): read off the file (curve_grid 1025, time_dim 8),
// the curve points those curve_coords reads, no bake, and the int8 file
// BYTE-IDENTICAL to its bf16 twin in every hold. The two forwards run at
// different timesteps on the SAME inputs, so their videos differ only if
// the modulation follows t -- which the pruned AdaLN is the whole of.
TEST(fp8_layouts, h3_pruned_int8_forwards_as_its_dense_twin)
{
  const H3Twin r = h3_int8_twin_(true);
  if (!r.ran) { return; }
  EXPECT_TRUE(r.cfg.pruned() && r.cfg.curve_grid == 1025 &&
              r.cfg.time_dim == 8 && !r.cfg.qkv_per_head);
  ASSERT_TRUE(r.finite);
  std::printf("[fp8_layouts] pruned H3 int8 against its bf16 twin: "
              "preloaded %s, streamed %s, no slots %s, promoted (%d of %d) "
              "%s; points %s, bake %s\n",
              say_(r.pre, r.dense), say_(r.str, r.dense),
              say_(r.flat, r.dense), r.grown.kept, r.layers,
              say_(r.grown, r.dense), r.points_match ? "match" : "DIFFER",
              r.bake_refused ? "refused" : "TAKEN");
  EXPECT_TRUE(same_out_(r.pre, r.dense));
  EXPECT_TRUE(same_out_(r.str, r.dense));
  EXPECT_TRUE(same_out_(r.flat, r.dense));
  EXPECT_TRUE(r.grown.kept > 0 && same_out_(r.grown, r.dense));
  EXPECT_TRUE(r.points_match && r.bake_refused);
  EXPECT_TRUE(!r.dense.v0.empty() && r.dense.v0 != r.dense.v);
  EXPECT_TRUE(widen_as_booked_(r));
}

// The same with W4A8 projections (comfy_kitchen's asym_w4a8_int8, ConvRot
// 256, a codebook) in ComfyUI's pruned form -- the community W4A8 files:
// the four-bit codes kept and widened per GEMM, BYTE-IDENTICAL to the
// twin holding their decode, preloaded, streamed through the slot pair
// (codes, relative scales and codebook refilled), with no slots and
// promoted.
TEST(fp8_layouts, h3_pruned_w4a8_forwards_as_its_dense_twin)
{
  const H3Twin r = h3_int8_twin_(true, true);
  if (!r.ran) { return; }
  ASSERT_TRUE(r.finite && r.cfg.pruned());
  std::printf("[fp8_layouts] pruned H3 W4A8 against its bf16 twin: "
              "preloaded %s, streamed %s, no slots %s, promoted (%d of %d) "
              "%s\n",
              say_(r.pre, r.dense), say_(r.str, r.dense),
              say_(r.flat, r.dense), r.grown.kept, r.layers,
              say_(r.grown, r.dense));
  EXPECT_TRUE(same_out_(r.pre, r.dense));
  EXPECT_TRUE(same_out_(r.str, r.dense));
  EXPECT_TRUE(same_out_(r.flat, r.dense));
  EXPECT_TRUE(r.grown.kept > 0 && same_out_(r.grown, r.dense));
  // At its unpacked width: a W4A8 weight widens to K, twice what it
  // stores.
  EXPECT_TRUE(widen_as_booked_(r));
}

// ComfyUI's int8 ConvRot and W4A8 RUN AS STORED on the token tier (M5:
// the int8 GEMM needs matrix cores): the activation rotated as the codes
// were (quant_i8_tok_cr), the token GEMM over the codes -- a W4A8 weight's
// decoded per GEMM -- and nothing converted. Engaged on every main-block
// projection; preloaded == streamed == no slots == promoted BYTE-
// IDENTICAL; and against the forward of the same weights with no int8
// GEMM, within twice what this tier costs the twin's bf16 weights
// quantized its own way.
TEST(fp8_layouts, h3_convrot_runs_as_stored_on_the_token_tier)
{
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) {
    std::printf("[fp8_layouts] no matrix cores: the token tier does not "
                "run here\n");
    return;
  }
  for (const bool w4 : {false, true}) {
    const H3Twin ref = h3_int8_twin_(true, w4, 0);
    const H3Twin tok = h3_int8_twin_(true, w4, 2);
    if (!ref.ran || !tok.ran) { return; }
    const auto rel = [](const H3Twin::Out& a, const H3Twin::Out& b) {
      double d = 0.0, n = 0.0;
      for (std::size_t i = 0; i < a.v.size() && i < b.v.size(); ++i) {
        const std::uint32_t ua = (std::uint32_t)a.v[i] << 16;
        const std::uint32_t ub = (std::uint32_t)b.v[i] << 16;
        float fa, fb;
        std::memcpy(&fa, &ua, 4);
        std::memcpy(&fb, &ub, 4);
        d += ((double)fa - fb) * ((double)fa - fb);
        n += (double)fb * fb;
      }
      return a.v.empty() || a.v.size() != b.v.size()
                 ? 1e9 : std::sqrt(d / std::max(n, 1e-30));
    };
    const double native = rel(tok.pre, ref.dense);
    const double own = rel(tok.dense, ref.dense);
    std::printf("[fp8_layouts] %s on the token tier, as stored: %llu GEMMs "
                "native; %.4g from the forward with no int8 GEMM (the "
                "twin's bf16 quantized by the tier itself: %.4g); streamed "
                "%s, no slots %s, promoted (%d of %d) %s\n",
                w4 ? "W4A8" : "int8 ConvRot",
                (unsigned long long)tok.pre.native, native, own,
                say_(tok.str, tok.pre), say_(tok.flat, tok.pre),
                tok.grown.kept, tok.layers, say_(tok.grown, tok.pre));
    EXPECT_TRUE(tok.pre.native > 0 && tok.str.native > 0);
    EXPECT_TRUE(native < 2.0 * std::max(own, 1e-3));
    EXPECT_TRUE(same_out_(tok.str, tok.pre) && same_out_(tok.flat, tok.pre));
    EXPECT_TRUE(tok.grown.kept > 0 && same_out_(tok.grown, tok.pre));
  }
}

// What running ConvRot codes as stored buys on the token tier, against
// widening them per GEMM (VPIPE_H3_CONVROT_NATIVE=0): a synthetic pruned
// H3 at a real block's widths -- hidden 3072, ffn 8192 -- from random
// int8 ConvRot codes, two main blocks, ~4k rows, forwards timed in
// alternation. Opt-in (VPIPE_CONVROT_BENCH=1, or =w4a8 for the W4A8
// layout) and M5 only.
TEST(fp8_layouts, h3_convrot_native_bench)
{
  if (std::getenv("VPIPE_CONVROT_BENCH") == nullptr) { return; }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  struct Env {
    const char* k;
    Env(const char* key, const char* v) : k(key) { ::setenv(key, v, 1); }
    ~Env() { ::unsetenv(k); }
  };
  Env noqmm("VPIPE_H3_NO_QMM_AUTOTUNE", "1");
  Env notile("VPIPE_H3_NO_ATTN_TILE_AUTOTUNE", "1");
  const int H = 3072, A = 3072, F = 8192, TXT = 256, LAYERS = 2;
  const std::string P = "model.diffusion_model.";
  std::vector<T> ts;
  unsigned sd = 7;
  auto rnd = [&]() {
    sd = sd * 1664525u + 1013904223u;
    return sd >> 8;
  };
  auto f32 = [&](const std::string& nm, std::vector<std::int64_t> shape,
                 float center, float spread) {
    std::size_t n = 1;
    for (auto x : shape) { n *= (std::size_t)x; }
    std::vector<float> v(n);
    for (float& x : v) {
      x = center + spread * ((float)(rnd() % 2001) / 1000.0f - 1.0f);
    }
    T t{P + nm, "F32", std::move(shape), std::vector<std::uint8_t>(n * 4)};
    std::memcpy(t.bytes.data(), v.data(), t.bytes.size());
    ts.push_back(std::move(t));
  };
  auto bf = [&](const std::string& nm, std::vector<std::int64_t> shape,
                float center, float spread) {
    std::size_t n = 1;
    for (auto x : shape) { n *= (std::size_t)x; }
    T t{P + nm, "BF16", std::move(shape), std::vector<std::uint8_t>(n * 2)};
    for (std::size_t i = 0; i < n; ++i) {
      const std::uint16_t b = rne_(
          center + spread * ((float)(rnd() % 2001) / 1000.0f - 1.0f));
      std::memcpy(&t.bytes[i * 2], &b, 2);
    }
    ts.push_back(std::move(t));
  };
  const bool w4 = std::string(std::getenv("VPIPE_CONVROT_BENCH")) == "w4a8";
  auto i8 = [&](const std::string& nm, int n, int k) {
    if (w4) {
      W4a8Parts pp;
      for (auto& t : w4a8_layer_(P + nm, n, k, 256, true, rnd(), &pp)) {
        ts.push_back(std::move(t));
      }
      return;
    }
    T w{P + nm + ".weight", "I8", {n, k},
        std::vector<std::uint8_t>((std::size_t)n * k)};
    for (auto& b : w.bytes) { b = (std::uint8_t)((int)(rnd() % 255) - 127); }
    ts.push_back(std::move(w));
    std::vector<float> sc((std::size_t)n, 0.02f / 127.0f);
    T st{P + nm + ".weight_scale", "F32", {n, 1},
         std::vector<std::uint8_t>((std::size_t)n * 4)};
    std::memcpy(st.bytes.data(), sc.data(), st.bytes.size());
    ts.push_back(std::move(st));
    ts.push_back(record_(P + nm, "{\"format\": \"int8_tensorwise\", "
                                 "\"convrot\": true, "
                                 "\"convrot_groupsize\": 256}"));
  };
  f32("video_patch_proj.weight", {H, 96}, 0.0f, 0.05f);
  f32("video_patch_proj.bias", {H}, 0.0f, 0.01f);
  f32("audio_patch_proj.weight", {H, 32}, 0.0f, 0.05f);
  f32("audio_patch_proj.bias", {H}, 0.0f, 0.01f);
  bf("condition_proj.weight", {H, TXT}, 0.0f, 0.03f);
  bf("condition_proj.bias", {H}, 0.0f, 0.01f);
  {
    std::vector<float> table((std::size_t)1025 * 8);
    for (int g = 0; g < 1025; ++g) {
      for (int k = 0; k < 8; ++k) {
        table[(std::size_t)g * 8 + k] =
            std::sin(0.004f * (float)g * (float)(k + 1));
      }
    }
    T t{P + "adaln_t_table", "F32", {1025, 8},
        std::vector<std::uint8_t>(table.size() * 4)};
    std::memcpy(t.bytes.data(), table.data(), t.bytes.size());
    ts.push_back(std::move(t));
    std::vector<float> inv(16);
    for (int i = 0; i < 16; ++i) {
      inv[(std::size_t)i] = 1.0f / std::pow(10000.0f, (float)(2 * i) / 32.0f);
    }
    T r{P + "rope.inv_freq", "F32", {16}, std::vector<std::uint8_t>(64)};
    std::memcpy(r.bytes.data(), inv.data(), 64);
    ts.push_back(std::move(r));
  }
  for (int L = 0; L < LAYERS + 1; ++L) {
    const bool main = L < LAYERS;
    const std::string b = main ? "blocks." + std::to_string(L) + "."
                               : std::string("token_refiner.blocks.0.");
    bf(b + "norm1.weight", {H}, 1.0f, 0.02f);
    bf(b + "norm2.weight", {H}, 1.0f, 0.02f);
    bf(b + "attn.q_norm.weight", {128}, 1.0f, 0.02f);
    bf(b + "attn.k_norm.weight", {128}, 1.0f, 0.02f);
    if (main) {
      i8(b + "attn.qkv_proj", 3 * A, H);
      i8(b + "attn.out_proj", H, A);
      i8(b + "mlp.fc1", 2 * F, H);
      i8(b + "mlp.fc2", H, F);
      f32(b + "adaln_proj.linear.weight", {18 * H, 8}, 0.0f, 0.02f);
      f32(b + "adaln_proj.linear.bias", {18 * H}, 0.0f, 0.02f);
    } else {
      bf(b + "attn.qkv_proj.weight", {3 * A, H}, 0.0f, 0.02f);
      bf(b + "attn.out_proj.weight", {H, A}, 0.0f, 0.02f);
      bf(b + "mlp.fc1.weight", {2 * F, H}, 0.0f, 0.02f);
      bf(b + "mlp.fc2.weight", {H, F}, 0.0f, 0.02f);
    }
  }
  bf("token_refiner.final_norm.weight", {H}, 1.0f, 0.02f);
  bf("final_layer.norm.weight", {H}, 1.0f, 0.02f);
  f32("final_layer.adaln_proj.linear.weight", {2 * H, 8}, 0.0f, 0.02f);
  f32("final_layer.adaln_proj.linear.bias", {2 * H}, 0.0f, 0.02f);
  f32("final_layer.video_out.weight", {96, H}, 0.0f, 0.05f);
  f32("final_layer.video_out.bias", {96}, 0.0f, 0.01f);
  f32("final_layer.audio_out.weight", {32, H}, 0.0f, 0.05f);
  f32("final_layer.audio_out.bias", {32}, 0.0f, 0.01f);
  const fs::path pf = tmp_("h3-convrot-bench.safetensors");
  ASSERT_TRUE(write_st_(pf, ts));
  ts.clear();
  MetalMiniMaxH3Transformer::Config base;
  std::string err;
  ASSERT_TRUE(MetalMiniMaxH3Transformer::config_from_json(pf.string(), base,
                                                          &err));
  namespace h3 = vpipe::genai::minimax_h3;
  h3::PackedLayout lay;
  const std::vector<int> tags(64, h3::kTextTag);
  ASSERT_TRUE(h3::build_packed_sequence(tags, 4, 64, 64, 32, base.patch_h,
                                        base.patch_w, h3::kAudioChannels,
                                        {h3::Anchor::kFirst}, &lay));
  auto input = [&](std::size_t n, float k) {
    metal_compute::SharedBuffer b = mc->make_shared_buffer(n * 2);
    auto* p = static_cast<std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) {
      p[i] = rne_(std::sin((float)i * k) * 0.5f);
    }
    return b;
  };
  const metal_compute::SharedBuffer vb = input(
      lay.video_indices.size() * (std::size_t)base.video_patch_elems(), 0.017f);
  const metal_compute::SharedBuffer ab = input(
      (std::size_t)lay.num_audio_rows * (std::size_t)base.audio_channels,
      0.03f);
  const metal_compute::SharedBuffer tb = input((std::size_t)64 * TXT, 0.005f);
  std::vector<float> uniq;
  std::vector<int> ridx;
  h3::build_row_timesteps(lay, 0.6f, 0.5f, 1.0f, &uniq, &ridx);
  auto arm = [&](bool native) {
    std::unique_ptr<Env> off;
    if (!native) {
      off = std::make_unique<Env>("VPIPE_H3_CONVROT_NATIVE", "0");
    }
    MetalMiniMaxH3Transformer::Config c = base;
    c.i8_gemm = true;
    c.i8_tier = 2;   // token
    return MetalMiniMaxH3Transformer::load(pf.string(), mc, c, false);
  };
  auto mn = arm(true), mw = arm(false);
  ASSERT_TRUE(mn != nullptr && mw != nullptr);
  if (mn == nullptr || mw == nullptr) { fs::remove(pf); return; }
  auto fwd = [&](MetalMiniMaxH3Transformer& m) {
    MetalMiniMaxH3Transformer::Step st;
    st.video = &vb;
    st.audio = &ab;
    st.text = &tb;
    st.layout = &lay;
    st.timesteps = &uniq;
    st.row_timestep_index = &ridx;
    const auto t0 = std::chrono::steady_clock::now();
    std::string ferr;
    const auto out = m.forward(st, &ferr);
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();
    return out.empty() ? -1.0 : ms;
  };
  (void)fwd(*mn);
  (void)fwd(*mw);
  double tn = 1e30, tw = 1e30;
  for (int r = 0; r < 4; ++r) {
    tn = std::min(tn, fwd(*mn));
    tw = std::min(tw, fwd(*mw));
  }
  std::printf("[fp8_layouts] ConvRot %s bench, %d rows, hidden %d, %d "
              "blocks: native %.1f ms (%llu GEMMs native), widened per GEMM "
              "%.1f ms -> %.2fx\n", w4 ? "W4A8" : "int8", lay.seq_len, H,
              LAYERS, tn,
              (unsigned long long)mn->convrot_native_gemms(), tw, tw / tn);
  EXPECT_TRUE(tn > 0.0 && tw > 0.0 && mn->convrot_native_gemms() > 0);
  fs::remove(pf);
}

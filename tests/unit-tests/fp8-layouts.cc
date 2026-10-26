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
TEST(fp8_layouts, h3_comfy_int8_forwards_as_its_dense_twin)
{
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  fp8::Expander probe;
  if (!probe.load(mc) || !probe.int8_valid()) { return; }
  struct Env {
    const char* k;
    Env(const char* key, const char* v) : k(key) { ::setenv(key, v, 1); }
    ~Env() { ::unsetenv(k); }
  };
  Env noqmm("VPIPE_H3_NO_QMM_AUTOTUNE", "1");
  Env notile("VPIPE_H3_NO_ATTN_TILE_AUTOTUNE", "1");

  // hidden 256, 2 heads of 128, ffn 512, text 256, time 256 -> 128; four
  // main blocks (two reach the slot refill) and two refiner blocks.
  const int H = 256, A = 256, F = 512, TXT = 256, FREQ = 256, TH = 256,
            TD = 128, LAYERS = 4, REFINER = 2;
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
  // An int8 projection in the file, and its decode in the twin. g > 0:
  // ConvRot at that group.
  auto lin = [&](const std::string& nm, int n, int k, int g, bool per_row) {
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
  both(f32("time_embedder.proj_in.weight", {TH, FREQ}, 0.0, 0.01));
  both(f32("time_embedder.proj_in.bias", {TH}, 0.0, 0.01));
  both(f32("time_embedder.proj_out.weight", {TD, TH}, 0.0, 0.01));
  both(f32("time_embedder.proj_out.bias", {TD}, 0.0, 0.01));
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
    lin(b + "adaln_proj.linear", 18 * H, TD, 64, true);
    both(bf16(b + "adaln_proj.linear.bias", {18 * H}, 0.0, 0.01));
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
  lin("final_layer.adaln_proj.linear", 2 * H, TD, 64, true);
  both(bf16("final_layer.adaln_proj.linear.bias", {2 * H}, 0.0, 0.01));
  both(f32("final_layer.video_out.weight", {96, H}, 0.0, 0.01));
  both(f32("final_layer.video_out.bias", {96}, 0.0, 0.01));
  both(f32("final_layer.audio_out.weight", {32, H}, 0.0, 0.01));
  both(f32("final_layer.audio_out.bias", {32}, 0.0, 0.01));

  const fs::path pq = tmp_("h3-int8.safetensors");
  const fs::path pd = tmp_("h3-bf16.safetensors");
  ASSERT_TRUE(write_st_(pq, q) && write_st_(pd, d));
  MetalMiniMaxH3Transformer::Config base;
  std::string err;
  ASSERT_TRUE(
      MetalMiniMaxH3Transformer::config_from_json(pq.string(), base, &err));
  if (!err.empty()) { std::printf("[fp8_layouts] %s\n", err.c_str()); }
  EXPECT_TRUE(!base.qkv_per_head && base.hidden == H &&
              base.n_layers == LAYERS && base.n_refiner == REFINER &&
              base.time_dim == TD);

  namespace h3 = vpipe::genai::minimax_h3;
  const int latf = 2, lath = 8, latw = 8, naud = 8, ntext = 8;
  h3::PackedLayout lay;
  const std::vector<int> tags((std::size_t)ntext, h3::kTextTag);
  ASSERT_TRUE(h3::build_packed_sequence(tags, latf, lath, latw, naud,
                                        base.patch_h, base.patch_w,
                                        h3::kAudioChannels,
                                        {h3::Anchor::kFirst}, &lay));
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
  struct Out {
    std::vector<std::uint16_t> v, a;
    int kept = 0;
  };
  // Two forwards; the second is compared, so a grown arm runs it on
  // promoted blocks.
  auto run = [&](const fs::path& file, bool stream, bool slots, bool grow,
                 bool bake) -> Out {
    Out o;
    std::unique_ptr<Env> ns;
    if (!slots) { ns = std::make_unique<Env>("VPIPE_H3_NO_SLOTS", "1"); }
    auto m = MetalMiniMaxH3Transformer::load(file.string(), mc, base, stream);
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
    }
    auto words = [](const metal_compute::SharedBuffer& b) {
      const auto* p = static_cast<const std::uint16_t*>(b.contents());
      return std::vector<std::uint16_t>(p, p + b.byte_size() / 2);
    };
    o.v = words(out.video);
    o.a = words(out.audio);
    o.kept = m->resident_block_count();
    return o;
  };
  auto same = [](const Out& x, const Out& y) {
    return !x.v.empty() && x.v == y.v && x.a == y.a;
  };
  auto finite_moving = [](const Out& x) {
    bool any = false;
    for (std::uint16_t w : x.v) {
      if ((w & 0x7f80u) == 0x7f80u) { return false; }
      any = any || (w & 0x7fffu) != 0;
    }
    return any;
  };
  const Out dense = run(pd, false, true, false, false);
  const Out pre = run(pq, false, true, false, false);
  const Out str = run(pq, true, true, false, false);
  const Out flat = run(pq, true, false, false, false);
  const Out grown = run(pq, true, true, true, false);
  const Out dense_baked = run(pd, false, true, false, true);
  const Out baked = run(pq, false, true, false, true);
  const Out str_baked = run(pq, true, true, false, true);
  ASSERT_TRUE(!dense.v.empty() && finite_moving(dense));
  auto say = [&](const Out& x, const Out& ref) {
    return same(x, ref) ? "IDENTICAL" : "DIFFERENT";
  };
  std::printf("[fp8_layouts] H3 int8 against its bf16 twin: preloaded %s, "
              "streamed %s, no slots %s, promoted (%d of %d) %s; baked: "
              "preloaded %s, streamed %s\n",
              say(pre, dense), say(str, dense), say(flat, dense), grown.kept,
              LAYERS, say(grown, dense), say(baked, dense_baked),
              say(str_baked, dense_baked));
  EXPECT_TRUE(same(pre, dense));
  EXPECT_TRUE(same(str, dense));
  EXPECT_TRUE(same(flat, dense));
  EXPECT_TRUE(grown.kept > 0 && same(grown, dense));
  EXPECT_TRUE(!dense_baked.v.empty() && same(baked, dense_baked));
  EXPECT_TRUE(same(str_baked, dense_baked));
  fs::remove(pq);
  fs::remove(pd);
}

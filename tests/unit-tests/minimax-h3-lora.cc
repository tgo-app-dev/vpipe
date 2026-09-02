// minimax-h3-lora.cc -- fusing a MiniMax-H3 Turbo LoRA into the DiT.
//
// Two things are being checked, and they fail in different ways.
//
// The first is the OUTPUT SHAPE of the pass. The H3 base that a LoRA is
// trained against is a Comfy-Org repack: one file per component, config
// inside the safetensors `__metadata__`, no config.json anywhere. The
// fuse writes a DIRECTORY of shards, so it has to synthesize that
// config -- and two of its fields are invisible in the tensors.
// `qkv_per_head` says whether the fused qkv projection is grouped per
// head or flat (same names, same shapes, different bytes), and the
// partition says whether these 33B of weights are the fl2va task or the
// ref2va one. Both lived only in the source FILENAME, which a directory
// output destroys. Lose either and the result loads and computes
// nonsense, so the synthetic arm below reads them back off the written
// config.json rather than trusting that the copy happened.
//
// The second is that the real adapter still MATCHES the real base. The
// Turbo LoRA is keyed on the model's own module names, so every module
// resolves through lora-fuse's first rule with no remap -- which is a
// property of somebody else's upload, not of this code. If it is
// re-published under different names, or against a base whose qkv
// grouping differs, the fuse would quietly fuse FEWER modules and still
// report success. That arm needs the files and skips without them.

#include "minitest.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "generative-models/lora-fusion.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/minimax-h3/minimax-h3-layout.h"
#include "generative-models/shared/runtime-lora.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
namespace fs = std::filesystem;
namespace h3 = vpipe::genai::minimax_h3;

namespace {

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

metal_compute::SharedBuffer
to_bf16_buf_(metal_compute::MetalCompute* mc, const std::vector<float>& v)
{
  metal_compute::SharedBuffer b = mc->make_shared_buffer(v.size() * 2);
  if (b.empty()) { return b; }
  auto* d = static_cast<std::uint16_t*>(b.contents());
  for (std::size_t i = 0; i < v.size(); ++i) { d[i] = to_bf16_(v[i]); }
  return b;
}

struct Tensor {
  std::string               name;
  std::vector<std::int64_t> shape;
  std::vector<float>        v;
};

// A safetensors file with an optional `__metadata__` block. Hand-rolled
// because the writer in the tree does not emit metadata -- and metadata
// is exactly what the Comfy-Org half of this test is about.
bool
write_st_(const fs::path& p, const std::vector<Tensor>& ts,
          const std::string& metadata_json)
{
  std::string hdr = "{";
  if (!metadata_json.empty()) {
    hdr += "\"__metadata__\":" + metadata_json + ",";
  }
  std::uint64_t off = 0;
  for (std::size_t i = 0; i < ts.size(); ++i) {
    const std::uint64_t n = ts[i].v.size() * 2;
    hdr += "\"" + ts[i].name + "\":{\"dtype\":\"BF16\",\"shape\":[";
    for (std::size_t d = 0; d < ts[i].shape.size(); ++d) {
      hdr += (d ? "," : "") + std::to_string(ts[i].shape[d]);
    }
    hdr += "],\"data_offsets\":[" + std::to_string(off) + "," +
           std::to_string(off + n) + "]}";
    hdr += (i + 1 < ts.size()) ? "," : "";
    off += n;
  }
  hdr += "}";
  std::ofstream f(p, std::ios::binary);
  if (!f) { return false; }
  const std::uint64_t hl = hdr.size();
  f.write(reinterpret_cast<const char*>(&hl), 8);
  f.write(hdr.data(), (std::streamsize)hdr.size());
  for (const Tensor& t : ts) {
    std::vector<std::uint16_t> b(t.v.size());
    for (std::size_t i = 0; i < t.v.size(); ++i) { b[i] = to_bf16_(t.v[i]); }
    f.write(reinterpret_cast<const char*>(b.data()),
            (std::streamsize)b.size() * 2);
  }
  return (bool)f;
}

std::vector<float>
ramp_(std::size_t n, float k, float amp)
{
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    v[i] = amp * std::sin((float)i * k);
  }
  return v;
}

}  // namespace

// The Comfy-Org single-file base -> fused directory round trip.
TEST(minimax_h3_lora, comfy_base_fuses_into_a_typed_directory)
{
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  const fs::path root =
      fs::temp_directory_path() / "vpipe-h3-lora-test";
  std::error_code ec;
  fs::remove_all(root, ec);
  fs::create_directories(root / "diffusion_models", ec);

  // The filename is load-bearing: the partition is not in the bytes.
  const fs::path base =
      root / "diffusion_models" / "minimax_h3_fl2va_bf16.safetensors";
  const fs::path lora = root / "turbo.safetensors";
  const fs::path out  = root / "fused";

  const int N = 8, K = 4, R = 2;
  std::vector<Tensor> bt = {
      {"blocks.0.attn.qkv_proj.weight", {N, K}, ramp_(N * K, 0.7f, 0.5f)},
      // Never adapted: it must survive byte-for-byte.
      {"blocks.0.norm1.weight", {K}, ramp_(K, 0.3f, 1.0f)},
  };
  // The embedded config, as Comfy-Org ships it: a JSON *string* under
  // the `config` key, with the DiT under `transformer`.
  const std::string cfg =
      "{\"config\":\"{\\\"transformer\\\": {\\\"image_model\\\": "
      "\\\"minimax_h3\\\", \\\"hidden_size\\\": 4}}\"}";
  ASSERT_TRUE(write_st_(base, bt, cfg));

  std::vector<Tensor> lt = {
      {"blocks.0.attn.qkv_proj.lora_A.weight", {R, K},
       ramp_(R * K, 1.1f, 0.25f)},
      {"blocks.0.attn.qkv_proj.lora_B.weight", {N, R},
       ramp_(N * R, 0.9f, 0.25f)},
  };
  ASSERT_TRUE(write_st_(lora, lt, ""));

  std::string err;
  const bool ok = fuse_lora(mc, base.string(), lora.string(), out.string(),
                            1.0f, &err);
  if (!ok) { std::printf("[minimax_h3_lora] fuse: %s\n", err.c_str()); }
  ASSERT_TRUE(ok);

  // ---- the config the output could not have inherited ---------------
  std::ifstream cf(out / "config.json");
  ASSERT_TRUE((bool)cf);
  const std::string ctxt((std::istreambuf_iterator<char>(cf)),
                         std::istreambuf_iterator<char>());
  FlexData written = FlexData::from_json(ctxt);
  ASSERT_TRUE(written.is_object());
  auto wo = written.as_object();
  ASSERT_TRUE(wo.contains("qkv_per_head"));
  // FLAT, because a Comfy-Org repack is what was fused. An output that
  // does not say so is read as the released per-head grouping.
  EXPECT_TRUE(wo.at("qkv_per_head").as_bool(true) == false);
  ASSERT_TRUE(wo.contains("_minimax_h3_partition"));
  EXPECT_TRUE(std::string(wo.at("_minimax_h3_partition").as_string("")) ==
              "fl2va");
  EXPECT_TRUE(std::string(wo.at("_class_name").as_string("")) ==
              "MiniMaxH3DiTModel");

  // ---- the arithmetic ------------------------------------------------
  auto fused = MetalLlamaWeights::open_model(out.string());
  ASSERT_TRUE(fused.has_value());
  const auto* fi = fused->info("blocks.0.attn.qkv_proj.weight");
  ASSERT_TRUE(fi != nullptr);
  metal_compute::SharedBuffer fb =
      fused->load("blocks.0.attn.qkv_proj.weight", mc);
  ASSERT_TRUE(!fb.empty());
  const auto* fp = static_cast<const std::uint16_t*>(fb.contents());
  double num = 0.0, den = 0.0;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < K; ++j) {
      double d = 0.0;
      for (int r = 0; r < R; ++r) {
        d += (double)lt[1].v[(std::size_t)i * R + r] *
             (double)lt[0].v[(std::size_t)r * K + j];
      }
      const double want = (double)bt[0].v[(std::size_t)i * K + j] + d;
      const double got  = (double)from_bf16_(fp[(std::size_t)i * K + j]);
      num += (got - want) * (got - want);
      den += want * want;
    }
  }
  const double rl2 = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
  std::printf("[minimax_h3_lora] fused vs base+B@A rel-L2 %.3e\n", rl2);
  EXPECT_TRUE(rl2 < 1e-2);   // one bf16 store

  // The unadapted tensor passes through.
  metal_compute::SharedBuffer nb = fused->load("blocks.0.norm1.weight", mc);
  ASSERT_TRUE(!nb.empty());
  const auto* np = static_cast<const std::uint16_t*>(nb.contents());
  bool same = true;
  for (int j = 0; j < K; ++j) {
    if (from_bf16_(np[j]) != from_bf16_(to_bf16_(bt[1].v[(std::size_t)j]))) {
      same = false;
    }
  }
  EXPECT_TRUE(same);
  fs::remove_all(root, ec);
}

// The REAL adapter against the REAL base: every module has to resolve.
//
// Env: VPIPE_MINIMAX_H3_TURBO_LORA (the .safetensors) and
// VPIPE_MINIMAX_H3_BASE_DIT (the FL2VA single file). Both absent =>
// skip; this cannot be synthesized, since what it checks is that
// somebody else's naming still lines up with ours.
TEST(minimax_h3_lora, turbo_lora_modules_resolve_against_the_base)
{
  const char* lp = std::getenv("VPIPE_MINIMAX_H3_TURBO_LORA");
  const char* bp = std::getenv("VPIPE_MINIMAX_H3_BASE_DIT");
  if (lp == nullptr || bp == nullptr || *lp == '\0' || *bp == '\0') {
    return;
  }
  auto lora = MetalLlamaWeights::open(lp);
  auto base = MetalLlamaWeights::open(bp);
  ASSERT_TRUE(lora.has_value() && base.has_value());

  int mods = 0, resolved = 0, shape_ok = 0, stripped = 0;
  for (const std::string& n : lora->tensor_names()) {
    const std::string suf = ".lora_A.weight";
    if (n.size() < suf.size() ||
        n.compare(n.size() - suf.size(), suf.size(), suf) != 0) {
      continue;
    }
    ++mods;
    const std::string mod = n.substr(0, n.size() - suf.size());
    // The same two rules lora-fuse's find_base applies, in its order:
    // the module's own name, then the name with its leading container
    // segment stripped. The second is what a ComfyUI-convention adapter
    // needs -- lightx2v's keys are `diffusion_model.blocks.N...` where
    // the model's own are unprefixed -- and an exact-match-only check
    // here would report a working adapter as unresolvable.
    const auto* bi = base->info(mod + ".weight");
    if (bi == nullptr && mod.find('.') != std::string::npos) {
      const std::string t = mod.substr(mod.find('.') + 1) + ".weight";
      bi = base->info(t);
      if (bi != nullptr) { ++stripped; }
    }
    if (bi == nullptr) {
      std::printf("[minimax_h3_lora] UNRESOLVED %s\n", mod.c_str());
      continue;
    }
    ++resolved;
    const auto* ai = lora->info(n);
    const auto* bl = lora->info(mod + ".lora_B.weight");
    if (ai == nullptr || bl == nullptr || bi->shape.size() != 2) { continue; }
    // base [N, K] against B [N, r] @ A [r, K].
    if (bl->shape[0] == bi->shape[0] && ai->shape[1] == bi->shape[1] &&
        bl->shape[1] == ai->shape[0]) {
      ++shape_ok;
    }
  }
  std::printf("[minimax_h3_lora] %d modules, %d resolved (%d by stripping a "
              "container prefix), %d shapes agree\n", mods, resolved,
              stripped, shape_ok);
  EXPECT_TRUE(mods > 0);
  // No remap, no misses: the adapter is keyed on the model's own names.
  EXPECT_TRUE(resolved == mods);
  EXPECT_TRUE(shape_ok == mods);
}

// The runtime adapter, through a real DiT.
//
// Three loads of the same checkpoint: no adapter, the adapter at scale
// 0, and at scale 1. Scale 0 has to reproduce the un-adapted velocity
// EXACTLY -- the strength is folded into A at bind, so a zero scale
// makes every delta identically zero, and any difference at all means
// the LoRA path is writing somewhere it should not. Scale 1 then has to
// move the output, or the adapter is being loaded and ignored.
//
// This is a PLUMBING test, not a quality one: it will happily use the
// Ref2VA checkpoint with the FL2VA-trained Turbo adapter, because the
// two partitions share every module name and shape and what is under
// test is the wiring. Judging whether the adapter helps needs the
// matching partition and a generation.
//
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH + VPIPE_MINIMAX_H3_TURBO_LORA.
TEST(minimax_h3_lora, runtime_adapter_is_off_at_zero_and_on_at_one)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  const char* lp   = std::getenv("VPIPE_MINIMAX_H3_TURBO_LORA");
  if (root == nullptr || lp == nullptr || *root == '\0' || *lp == '\0') {
    return;
  }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalMiniMaxH3Transformer::Config cfg;
  std::string cerr;
  if (!MetalMiniMaxH3Transformer::config_from_json(root, cfg, &cerr)) {
    std::printf("[minimax_h3_lora] config: %s\n", cerr.c_str());
    return;
  }
  cfg.n_layers = 4;

  h3::PackedLayout L;
  const std::vector<int> tags(8, h3::kTextTag);
  ASSERT_TRUE(h3::build_packed_sequence(tags, 2, 12, 20, 8, cfg.patch_h,
                                        cfg.patch_w, h3::kAudioChannels,
                                        {}, &L));
  std::vector<float> uniq;
  std::vector<int>   row_idx;
  h3::build_row_timesteps(L, 0.3125f, 0.5f, 1.0f, &uniq, &row_idx);
  const int n_video = (int)L.video_indices.size();
  auto ramp = [](std::size_t n, float k) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) { v[i] = std::sin((float)i * k) * 0.5f; }
    return v;
  };
  const std::vector<float> vin =
      ramp((std::size_t)n_video * cfg.video_patch_elems(), 0.017f);
  const std::vector<float> ain =
      ramp((std::size_t)L.num_audio_rows * cfg.audio_channels, 0.031f);
  const std::vector<float> tin =
      ramp((std::size_t)tags.size() * cfg.text_dim, 0.005f);
  const metal_compute::SharedBuffer vb = to_bf16_buf_(mc, vin);
  const metal_compute::SharedBuffer ab = to_bf16_buf_(mc, ain);
  const metal_compute::SharedBuffer tb = to_bf16_buf_(mc, tin);
  ASSERT_TRUE(!vb.empty() && !ab.empty() && !tb.empty());

  auto run = [&](MetalMiniMaxH3Transformer* m, std::vector<float>* out) {
    MetalMiniMaxH3Transformer::Step step;
    step.video = &vb;  step.audio = &ab;  step.text = &tb;
    step.layout = &L;  step.timesteps = &uniq;
    step.row_timestep_index = &row_idx;
    std::string ferr;
    const auto v = m->forward(step, &ferr);
    if (v.empty()) {
      std::printf("[minimax_h3_lora] forward: %s\n", ferr.c_str());
      return false;
    }
    const std::size_t n = (std::size_t)n_video * cfg.video_patch_elems();
    out->resize(n);
    const auto* p = static_cast<const std::uint16_t*>(v.video.contents());
    for (std::size_t i = 0; i < n; ++i) { (*out)[i] = from_bf16_(p[i]); }
    return true;
  };
  // Every arm pins the base GEMM route, and the bars below cannot hold
  // without it on a matrix-core box. They are EXACT-equality bars, which
  // is right -- scale 0 encodes no adapter kernels at all, so the two
  // runs should be the same run. But the base projections' route is
  // chosen by MEASUREMENT, and steel and matmul2d are a verified pair
  // that still differ in the last bits; this model turns that into
  // 3.7e-3 of velocity (runtime_adapter_routes_agree measures exactly
  // that). Two loads that voted differently then fail an exact bar with
  // nothing wrong, which is how this first read as an adapter defect.
  // kSteelBm32 because route_ok_ accepts it unconditionally, so the pin
  // holds on a box with no matrix cores too.
  const auto kPin = MetalMiniMaxH3Transformer::GemmRoute::kSteelBm32;
  using Specs = std::vector<MetalMiniMaxH3Transformer::LoraSpec>;
  auto arm_turned = [&](const Specs& specs,
                        float to, std::vector<float>* out, int* mods) {
    auto m = MetalMiniMaxH3Transformer::load(root, mc, cfg, false, specs);
    if (m == nullptr) { return false; }
    m->set_gemm_route(kPin);
    *mods = m->lora_modules();
    m->set_lora_scale(0, to);
    return run(m.get(), out);
  };
  auto arm = [&](const Specs& specs,
                 std::vector<float>* out, int* mods) {
    auto m = MetalMiniMaxH3Transformer::load(root, mc, cfg, false, specs);
    if (m == nullptr) { return false; }
    m->set_gemm_route(kPin);
    *mods = m->lora_modules();
    MetalMiniMaxH3Transformer::Step step;
    step.video = &vb;  step.audio = &ab;  step.text = &tb;
    step.layout = &L;  step.timesteps = &uniq;
    step.row_timestep_index = &row_idx;
    std::string ferr;
    const auto v = m->forward(step, &ferr);
    if (v.empty()) {
      std::printf("[minimax_h3_lora] forward: %s\n", ferr.c_str());
      return false;
    }
    const std::size_t n = (std::size_t)n_video * cfg.video_patch_elems();
    out->resize(n);
    const auto* p = static_cast<const std::uint16_t*>(v.video.contents());
    for (std::size_t i = 0; i < n; ++i) { (*out)[i] = from_bf16_(p[i]); }
    return true;
  };

  // Every arm on the SPLIT feed-forward. An adapter that touches mlp.fc1
  // disables the fused SwiGLU (its epilogue leaves nowhere for a
  // pre-activation delta), so an un-adapted baseline would run the FUSED
  // path and the difference measured below would be the two FF kernels
  // rounding differently -- 4.6e-3 here, which is larger than a real
  // adapter effect at low strength and looks exactly like one. Pinning
  // the path leaves the adapter as the only variable.
  ::setenv("VPIPE_H3_NO_FUSED_FF", "1", 1);
  MetalMiniMaxH3Transformer::LoraSpec zero{lp, 0.0f};
  MetalMiniMaxH3Transformer::LoraSpec one{lp, 1.0f};
  std::vector<float> v_off, v_zero, v_one, v_turned;
  int m_off = 0, m_zero = 0, m_one = 0, m_turned = 0;
  ASSERT_TRUE(arm({}, &v_off, &m_off));
  ASSERT_TRUE(arm({zero}, &v_zero, &m_zero));
  ASSERT_TRUE(arm({one}, &v_one, &m_one));
  // The strength is a per-forward constant, so an adapter loaded at 0
  // and TURNED UP has to land on the same velocity as one loaded at 1.
  // That is the claim that makes it a knob rather than a load argument,
  // and it is exactly what a scale folded into A at bind would fail.
  ASSERT_TRUE(arm_turned({zero}, 1.0f, &v_turned, &m_turned));
  EXPECT_TRUE(m_off == 0);
  EXPECT_TRUE(m_zero > 0 && m_one == m_zero);

  auto rel = [](const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      const double d = (double)a[i] - (double)b[i];
      num += d * d;
      den += (double)b[i] * (double)b[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
  };
  ::unsetenv("VPIPE_H3_NO_FUSED_FF");
  const double at_zero = rel(v_zero, v_off), at_one = rel(v_one, v_off);
  const double turned = rel(v_turned, v_one);
  std::printf("[minimax_h3_lora] %d modules over %d blocks | scale 0 moves "
              "%.3e, scale 1 moves %.3e | loaded-at-0-then-turned-up vs "
              "loaded-at-1: %.3e\n", m_one, cfg.n_layers, at_zero, at_one,
              turned);
  EXPECT_TRUE(at_zero == 0.0);
  EXPECT_TRUE(at_one > 1e-3);
  EXPECT_TRUE(turned == 0.0);
}

// The fused-SwiGLU gate has to ask in BOTH of fc1's spellings.
//
// An adapter on `mlp.fc1` and the fused SwiGLU are mutually exclusive --
// the fused epilogue writes silu(gate)*up straight out of the
// accumulator, leaving nowhere for a pre-activation delta -- and load()
// settles that before any block is built, by looking for the factor
// tensors in the FILE. It used to look only for `.mlp.fc1.lora_`, which
// is the ComfyUI spelling. bind_lora_ also reaches fc1 from the
// DIFFUSERS decomposition, where the same projection is `ff.net.0.proj`,
// so such a file left the fusion on and had its fc1 delta silently
// dropped.
//
// WHAT THIS PINS is the needle, not the consequence: reaching the
// consequence needs a box where the fusion is on at all, and a
// matrix-core GPU turns it off unconditionally (see _use_mma2 in
// load()), which is why the gap went unseen. The two EXPECTs below are
// the whole reason the second needle exists.
TEST(minimax_h3_lora, both_fc1_spellings_reach_the_fused_swiglu_gate)
{
  const fs::path dir = fs::temp_directory_path() / "vpipe-h3-fc1-needle";
  std::error_code ec;
  fs::remove_all(dir, ec);
  fs::create_directories(dir, ec);
  const fs::path comfy = dir / "comfy.safetensors";
  const fs::path diff  = dir / "diffusers.safetensors";
  const int R = 2, K = 4, N = 8;
  ASSERT_TRUE(write_st_(comfy, {
      {"blocks.0.mlp.fc1.lora_A.weight", {R, K}, ramp_(R * K, 1.1f, 0.25f)},
      {"blocks.0.mlp.fc1.lora_B.weight", {N, R}, ramp_(N * R, 0.9f, 0.25f)},
  }, ""));
  ASSERT_TRUE(write_st_(diff, {
      {"transformer_blocks.0.ff.net.0.proj.lora_A.default.weight", {R, K},
       ramp_(R * K, 1.1f, 0.25f)},
      {"transformer_blocks.0.ff.net.0.proj.lora_B.default.weight", {N, R},
       ramp_(N * R, 0.9f, 0.25f)},
  }, ""));
  using A = genai::lora::Adapter;
  EXPECT_TRUE(A::file_touches(comfy.string(), ".mlp.fc1.lora_"));
  // The gap: the diffusers copy adapts the same projection under
  // another name, and the old single needle could not see it.
  EXPECT_FALSE(A::file_touches(diff.string(), ".mlp.fc1.lora_"));
  EXPECT_TRUE(A::file_touches(diff.string(), ".ff.net.0.proj.lora_"));
  fs::remove_all(dir, ec);
}

// TWO adapters ride together, and their strengths are independent.
//
// The claim a second slot has to earn is LINEARITY: every adapted
// projection computes W x + s1 B1 (A1 x) + s2 B2 (A2 x), so with the
// same file in both slots the deltas must ADD. That is checkable with
// one adapter file, which is what a test box has -- half strength twice
// has to land where full strength once does, and two at full strength
// where one at double does.
//
// Independence is the other half, and it is the reason the slots are not
// merged into one set of factors at bind: concatenating them on the rank
// axis is exact arithmetic and free in the forward, but it folds both
// strengths into A and takes the live knob away. So the load-at-1-and-
// turn-one-down arm is not a refinement of the linearity check -- it is
// the property the design is built around.
//
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH + VPIPE_MINIMAX_H3_TURBO_LORA.
TEST(minimax_h3_lora, two_adapters_add_and_are_dialled_apart)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  const char* lp   = std::getenv("VPIPE_MINIMAX_H3_TURBO_LORA");
  if (root == nullptr || lp == nullptr || *root == '\0' || *lp == '\0') {
    return;
  }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalMiniMaxH3Transformer::Config cfg;
  std::string cerr;
  if (!MetalMiniMaxH3Transformer::config_from_json(root, cfg, &cerr)) {
    std::printf("[minimax_h3_lora] config: %s\n", cerr.c_str());
    return;
  }
  cfg.n_layers = 4;

  h3::PackedLayout L;
  const std::vector<int> tags(8, h3::kTextTag);
  ASSERT_TRUE(h3::build_packed_sequence(tags, 2, 12, 20, 8, cfg.patch_h,
                                        cfg.patch_w, h3::kAudioChannels,
                                        {}, &L));
  std::vector<float> uniq;
  std::vector<int>   row_idx;
  h3::build_row_timesteps(L, 0.3125f, 0.5f, 1.0f, &uniq, &row_idx);
  const int n_video = (int)L.video_indices.size();
  auto ramp = [](std::size_t n, float k) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) { v[i] = std::sin((float)i * k) * 0.5f; }
    return v;
  };
  const metal_compute::SharedBuffer vb = to_bf16_buf_(
      mc, ramp((std::size_t)n_video * cfg.video_patch_elems(), 0.017f));
  const metal_compute::SharedBuffer ab = to_bf16_buf_(
      mc, ramp((std::size_t)L.num_audio_rows * cfg.audio_channels, 0.031f));
  const metal_compute::SharedBuffer tb = to_bf16_buf_(
      mc, ramp((std::size_t)tags.size() * cfg.text_dim, 0.005f));
  ASSERT_TRUE(!vb.empty() && !ab.empty() && !tb.empty());

  // Same pin and same split-FF reasoning as
  // runtime_adapter_is_off_at_zero_and_on_at_one: with the route and the
  // FF path fixed, the adapters are the only variable left.
  const auto kPin = MetalMiniMaxH3Transformer::GemmRoute::kSteelBm32;
  ::setenv("VPIPE_H3_NO_FUSED_FF", "1", 1);
  using Specs = std::vector<MetalMiniMaxH3Transformer::LoraSpec>;
  int mods0 = 0, mods1 = 0, mods_all = 0, slots = 0;
  auto arm = [&](const Specs& specs, const std::vector<float>& turn,
                 std::vector<float>* out) {
    auto m = MetalMiniMaxH3Transformer::load(root, mc, cfg, false, specs);
    if (m == nullptr) { return false; }
    m->set_gemm_route(kPin);
    for (std::size_t i = 0; i < turn.size(); ++i) {
      m->set_lora_scale((int)i, turn[i]);
    }
    slots    = m->lora_slots();
    mods0    = m->lora_modules(0);
    mods1    = m->lora_modules(1);
    mods_all = m->lora_modules();
    MetalMiniMaxH3Transformer::Step step;
    step.video = &vb;  step.audio = &ab;  step.text = &tb;
    step.layout = &L;  step.timesteps = &uniq;
    step.row_timestep_index = &row_idx;
    std::string ferr;
    const auto v = m->forward(step, &ferr);
    if (v.empty()) {
      std::printf("[minimax_h3_lora] forward: %s\n", ferr.c_str());
      return false;
    }
    const std::size_t n = (std::size_t)n_video * cfg.video_patch_elems();
    out->resize(n);
    const auto* p = static_cast<const std::uint16_t*>(v.video.contents());
    for (std::size_t i = 0; i < n; ++i) { (*out)[i] = from_bf16_(p[i]); }
    return true;
  };
  auto rel = [](const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      const double d = (double)a[i] - (double)b[i];
      num += d * d;
      den += (double)b[i] * (double)b[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
  };

  // A SECOND, DIFFERENT adapter, synthesised against this DiT's own
  // dimensions. The linearity arms below use one file in both slots,
  // which cannot see whether the slots share a scratch region: with the
  // same A on both sides, one overwriting the other's [rows, rank]
  // intermediate is invisible. Two different files make the swap arm
  // below discriminating.
  const fs::path syn_dir = fs::temp_directory_path() / "vpipe-h3-lora2";
  std::error_code sec;
  fs::remove_all(syn_dir, sec);
  fs::create_directories(syn_dir, sec);
  const fs::path syn = syn_dir / "other.safetensors";
  {
    const int R = 8, HH = cfg.hidden, II = cfg.inner();
    std::vector<Tensor> ts;
    for (int i = 0; i < cfg.n_layers; ++i) {
      const std::string p = "blocks." + std::to_string(i) + ".attn.out_proj";
      ts.push_back({p + ".lora_A.weight", {R, II},
                    ramp_((std::size_t)R * II, 0.013f, 0.30f)});
      ts.push_back({p + ".lora_B.weight", {HH, R},
                    ramp_((std::size_t)HH * R, 0.021f, 0.30f)});
    }
    ASSERT_TRUE(write_st_(syn, ts, ""));
  }
  const MetalMiniMaxH3Transformer::LoraSpec other{syn.string(), 1.0f};

  const MetalMiniMaxH3Transformer::LoraSpec half{lp, 0.5f};
  const MetalMiniMaxH3Transformer::LoraSpec one{lp, 1.0f};
  const MetalMiniMaxH3Transformer::LoraSpec two{lp, 2.0f};
  const MetalMiniMaxH3Transformer::LoraSpec four{lp, 4.0f};
  const MetalMiniMaxH3Transformer::LoraSpec eight{lp, 8.0f};
  std::vector<float> v_four, v_eight, v_44;
  std::vector<float> v_off, v_one, v_two, v_half_half, v_one_one, v_dialled;
  ASSERT_TRUE(arm({}, {}, &v_off));
  EXPECT_TRUE(slots == 0 && mods_all == 0);
  ASSERT_TRUE(arm({one}, {}, &v_one));
  const int one_mods = mods_all;
  EXPECT_TRUE(slots == 1 && one_mods > 0);
  ASSERT_TRUE(arm({two}, {}, &v_two));

  // Both slots bind, and the totals are the sum rather than the last one
  // to write a member -- which is what the single-slot members did.
  ASSERT_TRUE(arm({half, half}, {}, &v_half_half));
  EXPECT_TRUE(slots == 2);
  EXPECT_TRUE(mods0 == one_mods && mods1 == one_mods);
  EXPECT_TRUE(mods_all == 2 * one_mods);
  ASSERT_TRUE(arm({one, one}, {}, &v_one_one));
  // Loaded at full in BOTH and the second turned to zero afterwards:
  // the first slot must still be at full. This is the arm that fails if
  // the two slots share a strength, or if slot 1's scratch region
  // overlaps slot 0's and its zeroed GEMM scribbles on the other's
  // intermediate.
  ASSERT_TRUE(arm({one, one}, {1.0f, 0.0f}, &v_dialled));
  ::unsetenv("VPIPE_H3_NO_FUSED_FF");

  // ORDER DOES NOT MATTER, and that is the arm that says the slots have
  // their own scratch. Addition commutes, so naming the two adapters
  // either way round is the same function -- but if both slots wrote
  // their [rows, rank] intermediate to the same place, the LAST one's
  // would be what both B factors multiply, and the two orders would
  // compute two different (wrong) things.
  std::vector<float> v_ab, v_ba;
  ASSERT_TRUE(arm({one, other}, {}, &v_ab));
  EXPECT_TRUE(slots == 2 && mods1 > 0);
  ASSERT_TRUE(arm({other, one}, {}, &v_ba));
  ASSERT_TRUE(arm({four}, {}, &v_four));
  ASSERT_TRUE(arm({eight}, {}, &v_eight));
  ASSERT_TRUE(arm({four, four}, {}, &v_44));
  ::unsetenv("VPIPE_H3_NO_FUSED_FF");

  const double d_one  = rel(v_one, v_off);
  const double d_two  = rel(v_two, v_off);
  const double add_hh = rel(v_half_half, v_one);
  const double add_11 = rel(v_one_one, v_two);
  const double add_44 = rel(v_44, v_eight);
  const double apart  = rel(v_four, v_eight);
  const double solo   = rel(v_dialled, v_one);
  const double swap   = rel(v_ab, v_ba);
  const double other_moves = rel(v_ab, v_one);
  std::printf("[minimax_h3_lora] two slots: %d+%d=%d modules | 1.0 moves "
              "%.3e, 2.0 moves %.3e | halves vs whole: 0.5+0.5 %.3e, 1+1 "
              "%.3e, 4+4 %.3e (4 vs 8 is %.3e away) | second dialled to 0 "
              "vs one alone %.3e\n",
              mods0, mods1, mods_all, d_one, d_two, add_hh, add_11, add_44,
              apart, solo);
  std::printf("[minimax_h3_lora] two DIFFERENT adapters: swapping the "
              "slots moves %.3e; the second one moves %.3e\n", swap,
              other_moves);
  // The adapter has to be DOING something, or the sums below are three
  // ways of comparing zero to zero.
  EXPECT_TRUE(d_one > 1e-3 && d_two > d_one);

  // TWO SLOTS ADD. `s` in one slot and `s` in the other is the same
  // arithmetic as `2s` in one -- W x + s BAx + s BAx against
  // W x + 2s BAx -- so the two runs differ only in rounding, and the
  // discriminating question is whether the pair lands on the SUM or on
  // one of its halves.
  //
  // At 4+4 the pair is 7.6e-3 from the sum where the single adapter is
  // 8.5e-2 away: eleven times closer, which no dropped or overwritten
  // slot could be.
  EXPECT_TRUE(add_44 < 0.1 * apart);
  // The residual is NOT an error that grows with strength. MEASURED at
  // three scales it is FLAT -- 5.7e-3 at 0.5+0.5, 6.6e-3 at 1+1, 7.6e-3
  // at 4+4, while the delta itself goes 1.9e-2 -> 3.8e-2 -> 1.6e-1 --
  // which is the signature of a fixed number of bf16 roundings of y and
  // not of a scale-dependent mistake. Splitting ONE adapter across two
  // slots costs about one and a half bf16 ULPs of the output (eps is
  // 2^-8 = 3.9e-3); two DIFFERENT adapters, which is the case this
  // exists for, pay the same and have nothing to be compared against.
  EXPECT_TRUE(add_hh < 0.02 && add_11 < 0.02 && add_44 < 0.02);
  // And turning the second slot off leaves the first exactly where it
  // was: no kernels are encoded for a slot at zero, so this is the same
  // run and not a near miss. The arm that fails if the slots share a
  // strength, or if slot 1's scratch region overlaps slot 0's.
  EXPECT_TRUE(solo == 0.0);
  // The second adapter has to be reaching the output, or the swap below
  // compares a run with itself.
  EXPECT_TRUE(other_moves > 1e-3);
  // Two accumulates onto y in the other order: bf16, so a ULP or two.
  // MEASURED 4.7e-3, which is one bf16 eps (2^-8 = 3.9e-3) of the
  // output and does NOT shrink as the adapters get stronger -- a
  // rounding floor, not a scale-dependent error. A shared scratch would
  // make the two orders compute different FUNCTIONS (the last slot's
  // intermediate is what both B factors would multiply), which lands an
  // order of magnitude above this.
  EXPECT_TRUE(swap < 0.01);
  EXPECT_TRUE(other_moves > 10.0 * swap);
  fs::remove_all(syn_dir, sec);
}

// The adapter's matrix-core route must agree with the steel pair.
//
// The runtime LoRA landed from a box with no matrix cores, so both of its
// GEMMs were steel unconditionally. On M5 the first one (t = x A^T, whose
// N is the RANK) belongs on matmul2d at every rank, and the second one
// (y += t B^T, whose K is the rank) belongs there from rank 128 up --
// measured in minimax_h3_blocks.lora_route_sweep. This is the arm that
// says the resulting four-way choice is WIRED right, and it is needed
// because the two routes do not carry the strength in the same place:
// steel's scalar epilogue applies it on the second GEMM, matmul2d has no
// room for a coefficient in its accumulate mode so the scaled tile
// applies it on the first. Getting that wrong applies it twice or not at
// all, and 0.625^2 vs 0.625 is a plausible-looking model either way.
//
// VPIPE_H3_NO_LORA_MMA moves ONLY the adapter back to steel -- the base
// projections stay on the matrix cores in both arms -- so what the
// comparison isolates is the adapter's route and nothing else.
//
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH + VPIPE_MINIMAX_H3_TURBO_LORA.
TEST(minimax_h3_lora, runtime_adapter_routes_agree)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  const char* lp   = std::getenv("VPIPE_MINIMAX_H3_TURBO_LORA");
  if (root == nullptr || lp == nullptr || *root == '\0' || *lp == '\0') {
    return;
  }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) {
    std::printf("[minimax_h3_lora] no matrix cores, route arm skipped\n");
    return;
  }
  MetalMiniMaxH3Transformer::Config cfg;
  std::string cerr;
  if (!MetalMiniMaxH3Transformer::config_from_json(root, cfg, &cerr)) {
    std::printf("[minimax_h3_lora] config: %s\n", cerr.c_str());
    return;
  }
  cfg.n_layers = 4;

  // Enough rows that the matrix-core route actually engages -- it is
  // gated on M >= 64, and a sequence under that would make both arms run
  // the same steel kernels and agree for the wrong reason. 16 latent
  // frames of a 40x24 grid is ~1900 rows, which is also past the point
  // where the tile choice stops being degenerate.
  h3::PackedLayout L;
  const std::vector<int> tags(8, h3::kTextTag);
  ASSERT_TRUE(h3::build_packed_sequence(tags, 16, 40, 24, 8, cfg.patch_h,
                                        cfg.patch_w, h3::kAudioChannels,
                                        {}, &L));
  std::vector<float> uniq;
  std::vector<int>   row_idx;
  h3::build_row_timesteps(L, 0.3125f, 0.5f, 1.0f, &uniq, &row_idx);
  const int n_video = (int)L.video_indices.size();
  auto ramp = [](std::size_t n, float k) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) {
      v[i] = std::sin((float)i * k) * 0.5f;
    }
    return v;
  };
  const metal_compute::SharedBuffer vb = to_bf16_buf_(
      mc, ramp((std::size_t)n_video * cfg.video_patch_elems(), 0.017f));
  const metal_compute::SharedBuffer ab = to_bf16_buf_(
      mc, ramp((std::size_t)L.num_audio_rows * cfg.audio_channels, 0.031f));
  const metal_compute::SharedBuffer tb = to_bf16_buf_(
      mc, ramp((std::size_t)tags.size() * cfg.text_dim, 0.005f));
  ASSERT_TRUE(!vb.empty() && !ab.empty() && !tb.empty());
  std::printf("[minimax_h3_lora] routes: %d rows packed (%d video)\n",
              L.seq_len, n_video);

  // A strength that is neither 0 nor 1 -- the two values that survive
  // applying the scale in the wrong place.
  const float kScale = 0.625f;
  ::setenv("VPIPE_H3_NO_FUSED_FF", "1", 1);
  // Every arm below PINS its base route (the `base` argument). Left
  // free, the tuner's own vote moves the velocity by the same 3.7e-3
  // this test is trying to measure, and every arm -- including the two
  // controls -- would read that instead of what it was asked. Pinned
  // through the API rather than VPIPE_H3_NO_QMM_AUTOTUNE because that
  // env is latched into a function-local static on first use, so in a
  // full-suite run an earlier H3 test decides it and a setenv here does
  // nothing.
  using Route = MetalMiniMaxH3Transformer::GemmRoute;
  auto arm = [&](bool steel_lora, std::vector<float>* out, int* mods,
                 Route base = Route::kAuto) {
    if (steel_lora) { ::setenv("VPIPE_H3_NO_LORA_MMA", "1", 1); }
    else            { ::unsetenv("VPIPE_H3_NO_LORA_MMA"); }
    MetalMiniMaxH3Transformer::LoraSpec spec{lp, kScale};
    auto m = MetalMiniMaxH3Transformer::load(root, mc, cfg, false, {spec});
    if (m == nullptr) { return false; }
    if (base != Route::kAuto) { m->set_gemm_route(base); }
    *mods = m->lora_modules();
    MetalMiniMaxH3Transformer::Step step;
    step.video = &vb;  step.audio = &ab;  step.text = &tb;
    step.layout = &L;  step.timesteps = &uniq;
    step.row_timestep_index = &row_idx;
    std::string ferr;
    const auto v = m->forward(step, &ferr);
    if (v.empty()) {
      std::printf("[minimax_h3_lora] forward: %s\n", ferr.c_str());
      return false;
    }
    const std::size_t n = (std::size_t)n_video * cfg.video_patch_elems();
    out->resize(n);
    const auto* p = static_cast<const std::uint16_t*>(v.video.contents());
    for (std::size_t i = 0; i < n; ++i) { (*out)[i] = from_bf16_(p[i]); }
    return true;
  };
  // The un-adapted velocity, so "they agree" can be weighed against how
  // much the adapter moved -- two routes that both dropped the delta
  // would agree perfectly.
  auto base_arm = [&](std::vector<float>* out) {
    ::unsetenv("VPIPE_H3_NO_LORA_MMA");
    auto m = MetalMiniMaxH3Transformer::load(root, mc, cfg, false, {});
    if (m == nullptr) { return false; }
    m->set_gemm_route(Route::kMma128);
    MetalMiniMaxH3Transformer::Step step;
    step.video = &vb;  step.audio = &ab;  step.text = &tb;
    step.layout = &L;  step.timesteps = &uniq;
    step.row_timestep_index = &row_idx;
    std::string ferr;
    const auto v = m->forward(step, &ferr);
    if (v.empty()) { return false; }
    const std::size_t n = (std::size_t)n_video * cfg.video_patch_elems();
    out->resize(n);
    const auto* p = static_cast<const std::uint16_t*>(v.video.contents());
    for (std::size_t i = 0; i < n; ++i) { (*out)[i] = from_bf16_(p[i]); }
    return true;
  };

  std::vector<float> v_mma, v_steel, v_steel2, v_base;
  int m_mma = 0, m_steel = 0, m_steel2 = 0;
  ASSERT_TRUE(arm(false, &v_mma, &m_mma, Route::kMma128));
  ASSERT_TRUE(arm(true, &v_steel, &m_steel, Route::kMma128));
  // The CONTROL: the same steel route loaded a second time. Two loads of
  // one checkpoint are not bit-identical on this box -- the base GEMM's
  // route is chosen by measurement, so a tuner that votes differently
  // between loads moves the velocity by more than the adapter does.
  // Without this floor, "mma vs steel" is a number with no scale, and
  // the first reading of it here (0.004 against an adapter effect of
  // 0.010) looked like a wiring bug when it was the tuner.
  ASSERT_TRUE(arm(true, &v_steel2, &m_steel2, Route::kMma128));
  std::vector<float> v_mma2;
  int m_mma2 = 0;
  ASSERT_TRUE(arm(false, &v_mma2, &m_mma2, Route::kMma128));
  // THE YARDSTICK. Run the BASE projections on steel and then on
  // matmul2d, with the adapter held on steel in both -- the SHIPPED
  // choice, made by a tuner that picks either one at run time. The two
  // are a verified pair (minimax_h3_blocks.mma_matches_steel) and differ
  // at the kernel level by about what the adapter's two routes do, so
  // this says what a difference of that size becomes at the velocity
  // after four blocks of residual stream.
  //
  // It is the only honest denominator. Two matmul2d TILES were tried
  // first and came out bit-identical, which calibrates nothing: the
  // question is not whether a benign swap is free but what an arithmetic
  // difference costs, and only an arm that changes the arithmetic
  // answers it.
  // And the FOLD's own A/B: same matrix-core adapter, but its second
  // GEMM encoded as its own dispatch instead of being absorbed into the
  // base tile. Isolated, the two are equally accurate -- both 2.398e-3
  // against an f64 reference (minimax_h3_blocks.lora_accumulate_matches_
  // reference), so the fold buys traffic and not precision. What comes
  // out here is therefore last-bit difference times this model's
  // amplification, which is what the yardstick measures.
  std::vector<float> v_nofuse;
  int m_nofuse = 0;
  ::setenv("VPIPE_H3_NO_LORA_FUSE", "1", 1);
  ASSERT_TRUE(arm(false, &v_nofuse, &m_nofuse, Route::kMma128));
  ::unsetenv("VPIPE_H3_NO_LORA_FUSE");

  std::vector<float> v_t1, v_t2;
  int mt1 = 0, mt2 = 0;
  ASSERT_TRUE(arm(true, &v_t1, &mt1, Route::kSteelBm64));
  ASSERT_TRUE(arm(true, &v_t2, &mt2, Route::kMma128));
  ASSERT_TRUE(base_arm(&v_base));
  ::unsetenv("VPIPE_H3_NO_LORA_MMA");
  ::unsetenv("VPIPE_H3_NO_FUSED_FF");
  EXPECT_TRUE(m_mma > 0 && m_mma == m_steel);

  auto rel = [](const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      const double d = (double)a[i] - (double)b[i];
      num += d * d;
      den += (double)b[i] * (double)b[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
  };
  const double routes = rel(v_mma, v_steel);
  const double floor_  = rel(v_steel2, v_steel);
  const double mfloor  = rel(v_mma2, v_mma);
  const double fold    = rel(v_nofuse, v_mma);
  const double tiles   = rel(v_t2, v_t1);
  const double moved  = rel(v_steel, v_base);
  std::printf("[minimax_h3_lora] %d modules, adapter at %.3f moves the "
              "velocity %.3e | mma vs steel adapter %.3e | steel vs steel "
              "(reload floor) %.3e | mma vs mma %.3e | BASE steel vs mma "
              "(yardstick) %.3e | folded vs separate %.3e\n", m_mma,
              (double)kScale, moved, routes, floor_, mfloor, tiles, fold);
  // Against the RELOAD FLOOR, not against zero: what has to be true is
  // that changing the adapter's route costs no more than reloading the
  // same model does. A scale applied in the wrong place shows up at the
  // 0.4-1.6 level -- two orders above either -- so the bar catches it
  // whichever way the floor lands. The 3e-3 slack covers the adapter's
  // own bf16 rounding, which the isolated arm
  // (minimax_h3_blocks.lora_mma_matches_steel) pins at 0.5% of the delta.
  EXPECT_TRUE(routes <= std::max(floor_, tiles) * 2.0 + 1e-3);
  // Bounded by the same yardstick, for the same reason: the fold is a
  // last-bit change to y, so it must not cost more than swapping one
  // verified base kernel for another does.
  EXPECT_TRUE(fold <= std::max(floor_, tiles) + 1e-3);
  // ...and the adapter has to be doing something, or the agreement is
  // between two paths that both computed the base.
  EXPECT_TRUE(moved > 1e-3);
}

// Which adapter route is RIGHT, on the adapter's own numbers.
//
// The two routes agree to 0.5% of the delta on synthetic factors
// (minimax_h3_blocks.lora_mma_matches_steel) and both land within 2.4e-3
// of an f64 reference there -- yet through a real DiT they move the
// velocity by 35% of what the adapter itself moves it. Synthetic operands
// are therefore not answering the question, and "the two routes agree" is
// not the claim that matters anyway. This runs the REAL factors and
// compares each route against f64, so the answer is about accuracy rather
// than about consensus.
//
// What makes this shape hard is the CONDITIONING of x A^T, not the
// kernel: A is a trained rank-64 factor and the contraction runs over
// thousands of terms that largely cancel, so the result is small next to
// the sum of the magnitudes and the ORDER of a f32 accumulation shows.
// steel walks K in its own tiling, matmul2d in the matrix unit's -- so
// they land differently, and neither is a bug. The number below says how
// far apart, and whether the matrix-core one is the worse of the two.
//
// Env: VPIPE_MINIMAX_H3_TURBO_LORA.
TEST(minimax_h3_lora, adapter_factors_condition_the_first_gemm)
{
  const char* lp = std::getenv("VPIPE_MINIMAX_H3_TURBO_LORA");
  if (lp == nullptr || *lp == '\0') { return; }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto w = MetalLlamaWeights::open(lp);
  if (!w.has_value()) { return; }
  metal_compute::ComputeLibrary lg = mc->load_library("dense_gemm_bf16");
  metal_compute::ComputeFunction steel = lg.function("dense_gemm_t_bm64_f16");
  metal_compute::ComputeLibrary lm = mc->load_library("dense_gemm_mma_bf16");
  metal_compute::ComputeFunction mma = lm.function("dense_gemm_mma_t_scaled_f16");
  if (!steel.valid()) { return; }
  const bool have_mma = mc->supports_matrix_cores() && mma.valid();

  // The first A factor in the file, whatever the publisher named it.
  std::string an;
  for (const std::string& n : w->tensor_names()) {
    if (n.size() > 14 && n.rfind(".lora_A.weight") == n.size() - 14) {
      an = n;
      break;
    }
  }
  if (an.empty()) { return; }
  const auto* ai = w->info(an);
  if (ai == nullptr || ai->shape.size() != 2) { return; }
  const int R = (int)ai->shape[0], K = (int)ai->shape[1];
  const int M = 512;
  metal_compute::SharedBuffer A = w->load(an, mc);
  if (A.empty()) { return; }
  // The factor may be stored at any width; the GEMMs want bf16.
  std::vector<double> Ad((std::size_t)R * K);
  {
    const auto* p16 = static_cast<const std::uint16_t*>(A.contents());
    const auto* p32 = static_cast<const float*>(A.contents());
    for (std::size_t i = 0; i < Ad.size(); ++i) {
      Ad[i] = ai->dtype == "F32"    ? (double)p32[i]
            : ai->dtype == "BF16"   ? (double)from_bf16_(p16[i])
                                    : (double)(float)((const _Float16*)
                                                      A.contents())[i];
    }
  }
  std::vector<float> Af(Ad.begin(), Ad.end());
  metal_compute::SharedBuffer Ab = to_bf16_buf_(mc, Af);

  // Rows with unit RMS and zero mean -- what a normalized activation
  // entering an adapted projection actually looks like. A uniform ramp
  // would understate the cancellation this test is about.
  std::vector<float> xv((std::size_t)M * K);
  std::uint32_t rs = 12345u;
  for (int m = 0; m < M; ++m) {
    double sum = 0.0, sq = 0.0;
    for (int k = 0; k < K; ++k) {
      rs = rs * 1664525u + 1013904223u;
      const double u1 = ((rs >> 8) & 0xffffff) / 16777216.0 + 1e-9;
      rs = rs * 1664525u + 1013904223u;
      const double u2 = ((rs >> 8) & 0xffffff) / 16777216.0;
      const double g = std::sqrt(-2.0 * std::log(u1)) *
                       std::cos(6.283185307179586 * u2);
      xv[(std::size_t)m * K + k] = (float)g;
      sum += g;
      sq  += g * g;
    }
    const double mean = sum / K;
    const double rms = std::sqrt(std::max(1e-12, sq / K - mean * mean));
    for (int k = 0; k < K; ++k) {
      xv[(std::size_t)m * K + k] =
          (float)((xv[(std::size_t)m * K + k] - mean) / rms);
    }
  }
  metal_compute::SharedBuffer xb = to_bf16_buf_(mc, xv);
  metal_compute::SharedBuffer ts = mc->make_shared_buffer(
      (std::size_t)M * R * 2);
  metal_compute::SharedBuffer tm = mc->make_shared_buffer(
      (std::size_t)M * R * 2);
  if (xb.empty() || Ab.empty() || ts.empty() || tm.empty()) { return; }

  {
    metal_compute::CommandStream st = mc->make_command_stream();
    {
      metal_compute::ComputeEncoder e = st.begin_compute();
      e.set_function(steel);
      e.set_buffer(0, xb); e.set_buffer(1, Ab); e.set_buffer(2, Ab);
      e.set_buffer(3, ts);
      e.set_constant(4, K); e.set_constant(5, R); e.set_constant(6, M);
      e.set_constant(7, 0);
      e.dispatch({(unsigned)(((R + 31) / 32) * 32),
                  (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
      if (have_mma) {
        e.set_function(mma);
        e.set_buffer(0, xb); e.set_buffer(1, Ab); e.set_buffer(2, Ab);
        e.set_buffer(3, tm);
        e.set_constant(4, K); e.set_constant(5, R); e.set_constant(6, M);
        e.set_constant(7, 0); e.set_constant(8, 1.0f);
        e.dispatch({(unsigned)(((R + 63) / 64) * 128),
                    (unsigned)((M + 63) / 64), 1}, {128, 1, 1});
      }
    }
    st.commit().wait();
  }

  // f64 truth off the SAME bf16 inputs, plus the conditioning number the
  // whole thing turns on: sum|terms| / |sum|.
  double ns = 0.0, nm = 0.0, den = 0.0, cond = 0.0;
  const auto* xp = static_cast<const std::uint16_t*>(xb.contents());
  const auto* ap = static_cast<const std::uint16_t*>(Ab.contents());
  const auto* sp = static_cast<const std::uint16_t*>(ts.contents());
  const auto* mp = static_cast<const std::uint16_t*>(tm.contents());
  for (int m = 0; m < M; ++m) {
    for (int r = 0; r < R; ++r) {
      double v = 0.0, mag = 0.0;
      for (int k = 0; k < K; ++k) {
        const double p = (double)from_bf16_(xp[(std::size_t)m * K + k]) *
                         (double)from_bf16_(ap[(std::size_t)r * K + k]);
        v += p;
        mag += std::fabs(p);
      }
      den += v * v;
      cond += (v != 0.0) ? mag / std::fabs(v) : 0.0;
      const double ds = (double)from_bf16_(sp[(std::size_t)m * R + r]) - v;
      ns += ds * ds;
      if (have_mma) {
        const double dm = (double)from_bf16_(mp[(std::size_t)m * R + r]) - v;
        nm += dm * dm;
      }
    }
  }
  const double rl_s = std::sqrt(ns / den);
  const double rl_m = have_mma ? std::sqrt(nm / den) : 0.0;
  std::printf("[minimax_h3_lora] %s [%d x %d], x unit-RMS, M=%d | "
              "cancellation sum|p|/|sum p| = %.1f | t vs f64: steel %.3e, "
              "mma %.3e\n", an.c_str(), R, K, M,
              cond / ((double)M * R), rl_s, rl_m);
  // Both routes have to be at the same order. What is NOT asserted is
  // that they equal each other: on an ill-conditioned contraction two f32
  // summation orders legitimately differ by more than either differs from
  // a single rounding, and that is exactly the regime here.
  EXPECT_TRUE(rl_s < 5e-2);
  if (have_mma) { EXPECT_TRUE(rl_m < 5e-2); }
}

// The adapter on top of the int8 accelerated GEMMs.
//
// These are the two opt-in accelerations a matrix-core box turns on for
// this model, and the shipped Turbo pipeline turns on BOTH -- so the
// combination is what actually runs, not a corner. They meet in one
// place: `i8_gemm` quantizes the BASE projection (activation and dequanted
// weight to int8, per call), the adapter's delta is added on top of that
// projection's output in bf16, and neither knows about the other.
//
// Independent is not the same as harmless, and the risk is one-sided.
// int8 is LOSSY, and a Turbo adapter is a small, precise correction --
// so the question is not whether the adapter still runs (it does; it is a
// separate pair of GEMMs) but whether what it contributes survives being
// added to a base that has been perturbed by more than the correction is
// worth. If i8's error swamped the adapter, both would still "work" and
// the adapter would be pointless.
//
// MEASURED over 4 blocks, and the answer is yes-with-a-caveat: the
// adapter moves the velocity 1.58e-2 on a bf16 base and 1.99e-2 on an
// int8 one -- it composes, and the 26% is the two perturbations not
// being orthogonal rather than the adapter being clipped. But int8
// ALONE moves it 1.43e-2, i.e. about as much as applying the adapter
// does. That is not a defect -- int8 is a systematic approximation of
// the same function, and this model is sensitive enough at the velocity
// that swapping one GEMM kernel for another verified one costs 3.7e-3
// (runtime_adapter_routes_agree) -- but it does mean the two must not be
// changed together when the question is whether the ADAPTER helps.
//
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH + VPIPE_MINIMAX_H3_TURBO_LORA.
TEST(minimax_h3_lora, runtime_adapter_survives_i8_gemm)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  const char* lp   = std::getenv("VPIPE_MINIMAX_H3_TURBO_LORA");
  if (root == nullptr || lp == nullptr || *root == '\0' || *lp == '\0') {
    return;
  }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) {
    std::printf("[minimax_h3_lora] no matrix cores, i8 arm skipped\n");
    return;
  }
  MetalMiniMaxH3Transformer::Config cfg;
  std::string cerr;
  if (!MetalMiniMaxH3Transformer::config_from_json(root, cfg, &cerr)) {
    return;
  }
  cfg.n_layers = 4;

  // I8GemmContext gates itself at M >= 1024, so a short clip silently
  // keeps the bf16 tiles and every arm below would be the same arm.
  h3::PackedLayout L;
  const std::vector<int> tags(8, h3::kTextTag);
  ASSERT_TRUE(h3::build_packed_sequence(tags, 16, 40, 24, 8, cfg.patch_h,
                                        cfg.patch_w, h3::kAudioChannels,
                                        {}, &L));
  std::vector<float> uniq;
  std::vector<int>   row_idx;
  h3::build_row_timesteps(L, 0.3125f, 0.5f, 1.0f, &uniq, &row_idx);
  const int n_video = (int)L.video_indices.size();
  auto ramp = [](std::size_t n, float k) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) { v[i] = std::sin((float)i * k) * 0.5f; }
    return v;
  };
  const metal_compute::SharedBuffer vb = to_bf16_buf_(
      mc, ramp((std::size_t)n_video * cfg.video_patch_elems(), 0.017f));
  const metal_compute::SharedBuffer ab = to_bf16_buf_(
      mc, ramp((std::size_t)L.num_audio_rows * cfg.audio_channels, 0.031f));
  const metal_compute::SharedBuffer tb = to_bf16_buf_(
      mc, ramp((std::size_t)tags.size() * cfg.text_dim, 0.005f));
  ASSERT_TRUE(!vb.empty() && !ab.empty() && !tb.empty());

  ::setenv("VPIPE_H3_NO_FUSED_FF", "1", 1);
  auto arm = [&](bool i8, bool lora, std::vector<float>* out) {
    MetalMiniMaxH3Transformer::Config c = cfg;
    c.i8_gemm = i8;
    MetalMiniMaxH3Transformer::LoraSpec spec{lp, 1.0f};
    auto m = MetalMiniMaxH3Transformer::load(
        root, mc, c, false,
        lora ? std::vector<MetalMiniMaxH3Transformer::LoraSpec>{spec}
             : std::vector<MetalMiniMaxH3Transformer::LoraSpec>{});
    if (m == nullptr) { return false; }
    // Pin the base tile: the route tuner's own vote moves the velocity by
    // about what i8 does, so an unpinned arm measures the wrong thing.
    m->set_gemm_route(MetalMiniMaxH3Transformer::GemmRoute::kMma128);
    MetalMiniMaxH3Transformer::Step step;
    step.video = &vb;  step.audio = &ab;  step.text = &tb;
    step.layout = &L;  step.timesteps = &uniq;
    step.row_timestep_index = &row_idx;
    std::string ferr;
    const auto v = m->forward(step, &ferr);
    if (v.empty()) {
      std::printf("[minimax_h3_lora] forward: %s\n", ferr.c_str());
      return false;
    }
    const std::size_t n = (std::size_t)n_video * cfg.video_patch_elems();
    out->resize(n);
    const auto* p = static_cast<const std::uint16_t*>(v.video.contents());
    for (std::size_t i = 0; i < n; ++i) { (*out)[i] = from_bf16_(p[i]); }
    return true;
  };
  std::vector<float> bf, bf_l, i8, i8_l;
  ASSERT_TRUE(arm(false, false, &bf));
  ASSERT_TRUE(arm(false, true,  &bf_l));
  ASSERT_TRUE(arm(true,  false, &i8));
  ASSERT_TRUE(arm(true,  true,  &i8_l));
  ::unsetenv("VPIPE_H3_NO_FUSED_FF");

  auto rel = [](const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      const double d = (double)a[i] - (double)b[i];
      num += d * d;
      den += (double)b[i] * (double)b[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
  };
  const double moved_bf = rel(bf_l, bf);       // adapter, bf16 base
  const double moved_i8 = rel(i8_l, i8);       // adapter, int8 base
  const double i8_cost  = rel(i8, bf);         // what int8 alone changes
  std::printf("[minimax_h3_lora] i8+lora | adapter moves the velocity: "
              "bf16 base %.3e, int8 base %.3e | int8 alone moves %.3e\n",
              moved_bf, moved_i8, i8_cost);
  // int8 has to be ENGAGED, or the three numbers below are one number
  // measured three times and the test proves nothing.
  EXPECT_TRUE(i8_cost > 1e-4);
  // The adapter contributes the same thing on either base -- it is a
  // separate pair of GEMMs reading the same activations, so this is the
  // claim that the two accelerations compose rather than interact.
  EXPECT_TRUE(moved_i8 > 0.5 * moved_bf && moved_i8 < 2.0 * moved_bf);
}

// A baked schedule and a runtime adapter, together.
//
// They meet on the AdaLN projection, which is the one module both want.
// bake_adaln precomputes its output for the whole schedule and drops the
// weights; the Turbo adapter has a rank-16 factor on that same module in
// every block. Baking the adapter IN would have been simpler and is
// deliberately not done -- it would turn `lora_scale` from a knob into a
// 33B reload, which is the property the runtime path exists for. So the
// baked table is the BASE, and the adapter is added to a copy of this
// step's slice, per forward.
//
// Two claims, and the second is the one that costs something to keep:
// the adapter must still move the velocity by what it moved before, and
// the strength must still be live -- an adapter loaded at 0 and turned
// up has to land exactly where one loaded at 1 does.
//
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH + VPIPE_MINIMAX_H3_TURBO_LORA.
TEST(minimax_h3_lora, baked_adaln_keeps_the_adapter_live)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  const char* lp   = std::getenv("VPIPE_MINIMAX_H3_TURBO_LORA");
  if (root == nullptr || lp == nullptr || *root == '\0' || *lp == '\0') {
    return;
  }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalMiniMaxH3Transformer::Config cfg;
  std::string cerr;
  if (!MetalMiniMaxH3Transformer::config_from_json(root, cfg, &cerr)) {
    return;
  }
  cfg.n_layers = 4;

  h3::PackedLayout L;
  const std::vector<int> tags(8, h3::kTextTag);
  ASSERT_TRUE(h3::build_packed_sequence(tags, 5, 20, 12, 8, cfg.patch_h,
                                        cfg.patch_w, h3::kAudioChannels,
                                        {}, &L));
  const float kV[] = {0.9f, 0.4f};
  const float kA[] = {0.8f, 0.35f};
  const int kSteps = 2;
  std::vector<std::vector<float>> sched, uniqs((std::size_t)kSteps);
  std::vector<std::vector<int>> ridx((std::size_t)kSteps);
  for (int i = 0; i < kSteps; ++i) {
    h3::build_row_timesteps(L, kV[i], kA[i], 1.0f, &uniqs[(std::size_t)i],
                            &ridx[(std::size_t)i]);
    sched.push_back(uniqs[(std::size_t)i]);
  }

  const int n_video = (int)L.video_indices.size();
  auto ramp = [](std::size_t n, float k) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) { v[i] = std::sin((float)i * k) * 0.5f; }
    return v;
  };
  const metal_compute::SharedBuffer vb = to_bf16_buf_(
      mc, ramp((std::size_t)n_video * cfg.video_patch_elems(), 0.017f));
  const metal_compute::SharedBuffer ab = to_bf16_buf_(
      mc, ramp((std::size_t)L.num_audio_rows * cfg.audio_channels, 0.031f));
  const metal_compute::SharedBuffer tb = to_bf16_buf_(
      mc, ramp((std::size_t)tags.size() * cfg.text_dim, 0.005f));
  ASSERT_TRUE(!vb.empty() && !ab.empty() && !tb.empty());

  ::setenv("VPIPE_H3_NO_FUSED_FF", "1", 1);
  const auto kPin = MetalMiniMaxH3Transformer::GemmRoute::kSteelBm32;
  // `turn_to` >= 0 loads at 0 and turns the strength up afterwards,
  // which is what a folded-in adapter could not do.
  auto run = [&](bool bake, bool lora, float scale, float turn_to,
                 std::vector<float>* out) {
    MetalMiniMaxH3Transformer::LoraSpec spec{lp, turn_to >= 0.0f ? 0.0f
                                                                : scale};
    auto m = MetalMiniMaxH3Transformer::load(
        root, mc, cfg, false,
        lora ? std::vector<MetalMiniMaxH3Transformer::LoraSpec>{spec}
             : std::vector<MetalMiniMaxH3Transformer::LoraSpec>{});
    if (m == nullptr) { return false; }
    m->set_gemm_route(kPin);
    if (bake) {
      std::string berr;
      if (!m->bake_adaln(sched, &berr)) { return false; }
    }
    if (turn_to >= 0.0f) { m->set_lora_scale(0, turn_to); }
    out->clear();
    for (int i = 0; i < kSteps; ++i) {
      MetalMiniMaxH3Transformer::Step st;
      st.video = &vb;  st.audio = &ab;  st.text = &tb;
      st.layout = &L;
      st.timesteps          = &uniqs[(std::size_t)i];
      st.row_timestep_index = &ridx[(std::size_t)i];
      st.schedule_index     = bake ? i : -1;
      std::string ferr;
      const auto v = m->forward(st, &ferr);
      if (v.empty()) { return false; }
      const std::size_t n = (std::size_t)n_video * cfg.video_patch_elems();
      const auto* p = static_cast<const std::uint16_t*>(v.video.contents());
      for (std::size_t k = 0; k < n; ++k) { out->push_back(from_bf16_(p[k])); }
    }
    return true;
  };

  std::vector<float> base_p, lora_p, base_b, lora_b, turned_b;
  ASSERT_TRUE(run(false, false, 1.0f, -1.0f, &base_p));
  ASSERT_TRUE(run(false, true,  1.0f, -1.0f, &lora_p));
  ASSERT_TRUE(run(true,  false, 1.0f, -1.0f, &base_b));
  ASSERT_TRUE(run(true,  true,  1.0f, -1.0f, &lora_b));
  ASSERT_TRUE(run(true,  true,  1.0f,  1.0f, &turned_b));
  ::unsetenv("VPIPE_H3_NO_FUSED_FF");

  auto rel = [](const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      const double d = (double)a[i] - (double)b[i];
      num += d * d;
      den += (double)b[i] * (double)b[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
  };
  const double moved_p = rel(lora_p, base_p);
  const double moved_b = rel(lora_b, base_b);
  const double agree   = rel(lora_b, lora_p);
  // The same comparison WITHOUT the adapter, which localizes any
  // difference: baking alone is bit-exact
  // (minimax_h3_dit.baked_adaln_matches_the_projections), so anything
  // here that is not zero belongs to the adapter path.
  const double agree0  = rel(base_b, base_p);
  const double live    = rel(turned_b, lora_b);
  std::printf("[minimax_h3_lora] baked+adapter | adapter moves: projections "
              "%.3e, baked %.3e | baked vs projections: with adapter %.3e, "
              "without %.3e | turned-up vs loaded-at-1 %.3e\n",
              moved_p, moved_b, agree, agree0, live);
  // The adapter has to be doing something, on both paths.
  EXPECT_TRUE(moved_p > 1e-3 && moved_b > 1e-3);
  // EXACT, both ways. Baking moves WHERE the base modulation is computed
  // and the copy is bf16-to-bf16, so nothing rounds differently -- and
  // an exact bar is what caught the adapter reading an uninitialized
  // temb, which a tolerance would have passed as "close enough".
  EXPECT_TRUE(agree == 0.0);
  EXPECT_TRUE(agree0 == 0.0);
  // And the strength is still a per-forward constant: exactly equal, not
  // approximately, because scale 0 encodes no adapter kernels at all.
  EXPECT_TRUE(live == 0.0);
}

// ---- the split (diffusers) decomposition ----------------------------

// Where a split adapter's q/k/v rows land in the fused qkv output.
//
// This is the piece of the conversion that can be wrong without
// anything failing: every shape stays right and the delta simply lands
// on the wrong channels. The flat case is checked against the exact
// permutation upstream's own ComfyUI conversion produces; the per-head
// case is checked against the offsets the FORWARD uses for the same
// buffer, which is the coupling that would otherwise be nothing but a
// comment.
TEST(minimax_h3_lora, split_qkv_rows_land_where_the_forward_reads_them)
{
  using T = MetalMiniMaxH3Transformer;
  const int HD = 128, HEADS = 56;
  const int I = HEADS * HD;                 // 7168
  const int N = 3 * I;

  for (int arm = 0; arm < 2; ++arm) {
    const bool per_head = arm == 1;
    // A bijection onto [0, 3*inner). Anything else either drops a
    // channel or writes two deltas onto one.
    std::vector<int> hit((std::size_t)N, 0);
    for (int pt = 0; pt < 3; ++pt) {
      for (int r = 0; r < I; ++r) {
        const int d = T::qkv_fused_row(pt, r, I, HD, per_head);
        ASSERT_TRUE(d >= 0 && d < N);
        if (d >= 0 && d < N) { ++hit[(std::size_t)d]; }
      }
    }
    int once = 0;
    for (int x : hit) { if (x == 1) { ++once; } }
    EXPECT_TRUE(once == N);
  }

  // FLAT is exactly [all q | all k | all v] -- the layout Comfy-Org's
  // repack uses and the one every published fusion is built for.
  EXPECT_TRUE(T::qkv_fused_row(0, 0, I, HD, false) == 0);
  EXPECT_TRUE(T::qkv_fused_row(1, 0, I, HD, false) == I);
  EXPECT_TRUE(T::qkv_fused_row(2, 0, I, HD, false) == 2 * I);
  EXPECT_TRUE(T::qkv_fused_row(0, I - 1, I, HD, false) == I - 1);

  // PER-HEAD must match the forward's own view of the same buffer:
  // head stride 3*head_dim, k at +head_dim, v at +2*head_dim. These are
  // the QKV_HSTRIDE / K_OFF / V_OFF of forward_dit, restated here so a
  // change to either side without the other fails.
  const int HSTRIDE = 3 * HD, K_OFF = HD, V_OFF = 2 * HD;
  bool ok = true;
  for (int h = 0; h < HEADS; ++h) {
    for (int d = 0; d < HD; ++d) {
      const int r = h * HD + d;
      ok = ok && T::qkv_fused_row(0, r, I, HD, true) == h * HSTRIDE + d;
      ok = ok && T::qkv_fused_row(1, r, I, HD, true)
                     == h * HSTRIDE + K_OFF + d;
      ok = ok && T::qkv_fused_row(2, r, I, HD, true)
                     == h * HSTRIDE + V_OFF + d;
    }
  }
  EXPECT_TRUE(ok);

  // The two groupings are NOT the same map -- which is the whole reason
  // a published (flat) fusion is wrong on per-head weights, and the
  // reason this tree fuses from the split file instead.
  EXPECT_TRUE(T::qkv_fused_row(1, 0, I, HD, false) !=
              T::qkv_fused_row(1, 0, I, HD, true));
}

// The flat -> per-head map, against the two published checkpoints.
//
// Comfy-Org's repack and MiniMaxAI's release hold the SAME qkv weights
// in different row orders, so the map between them is provable on the
// bytes rather than argued from a tag -- which is what this needs, the
// two arguments-from-tags in this area having both been wrong once.
//
// Env: VPIPE_H3_COMFY_DIT = the repack's diffusion_models/*.safetensors,
// VPIPE_H3_DIFFUSERS_DIT = MiniMaxAI's transformer/ directory. Skips
// without both -- they are 66 GB apiece.
TEST(minimax_h3_lora, the_flat_to_per_head_map_is_the_two_checkpoints)
{
  const char* cf = std::getenv("VPIPE_H3_COMFY_DIT");
  const char* dd = std::getenv("VPIPE_H3_DIFFUSERS_DIT");
  if (cf == nullptr || dd == nullptr || !*cf || !*dd) { return; }
  namespace fs = std::filesystem;
  auto comfy = MetalLlamaWeights::open(cf);
  auto diff  = MetalLlamaWeights::open_model(dd);
  ASSERT_TRUE(comfy.has_value() && diff.has_value());
  if (!comfy.has_value() || !diff.has_value()) { return; }

  const std::string key = "blocks.0.attn.qkv_proj.weight";
  const auto* ci = comfy->info(key);
  const auto* di = diff->info(key);
  ASSERT_TRUE(ci != nullptr && di != nullptr);
  if (ci == nullptr || di == nullptr) { return; }
  ASSERT_TRUE(ci->shape == di->shape && ci->shape.size() == 2);
  if (ci->shape != di->shape) { return; }

  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  metal_compute::SharedBuffer cb = comfy->load(key, mc);
  metal_compute::SharedBuffer db = diff->load(key, mc);
  ASSERT_TRUE(!cb.empty() && !db.empty());
  if (cb.empty() || db.empty()) { return; }

  const int rows = (int)ci->shape[0], cols = (int)ci->shape[1];
  const int HD = 128, I = rows / 3;
  const auto* c = static_cast<const std::uint16_t*>(cb.contents());
  const auto* d = static_cast<const std::uint16_t*>(db.contents());
  auto row_eq = [&](int cr, int dr) {
    return std::memcmp(c + (std::size_t)cr * cols,
                       d + (std::size_t)dr * cols,
                       (std::size_t)cols * 2) == 0;
  };
  int mapped = 0, identity = 0;
  const int kStride = rows / 200 + 1;
  int checked = 0;
  for (int r = 0; r < rows; r += kStride) {
    ++checked;
    const int p = MetalMiniMaxH3Transformer::qkv_fused_row(
        r / I, r % I, I, HD, /*per_head=*/true);
    if (row_eq(r, p)) { ++mapped; }
    if (row_eq(r, r)) { ++identity; }
  }
  std::printf("[minimax_h3_lora] %d rows: mapped %d, identity %d\n",
              checked, mapped, identity);
  // EVERY row matches under the map. The identity count is not asserted
  // to be zero -- the permutation has fixed points, and row 0 is one --
  // only that it is nowhere near the whole, or the two layouts would be
  // the same layout and there would be nothing to adapt.
  EXPECT_TRUE(mapped == checked);
  EXPECT_TRUE(identity * 4 < checked);
}

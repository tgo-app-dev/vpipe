// Krea-2 LoRA fusion: fuse the softwatercolor LoRA into the DiT and verify the
// fused weight equals base + B@A, and that the fused DiT reloads + runs.
//
// Env: VPIPE_KREA2_TEST_MODEL_PATH = the Krea-2-Turbo root. The LoRA is read
// from VPIPE_KREA2_LORA (a .safetensors) or the sibling
// <root>/../Krea-2-LoRA-softwatercolor/softwatercolor.safetensors. Skips if
// unset / not present.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"
#include "generative-models/krea2/metal-krea2-transformer.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/lora-fusion.h"

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
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

float
bf16f(std::uint16_t b)
{
  std::uint32_t u = (std::uint32_t)b << 16;
  float f; std::memcpy(&f, &u, 4); return f;
}

std::vector<float>
tensor_f32(const MetalLlamaWeights& w, const std::string& name, MetalCompute* mc)
{
  const auto* ti = w.info(name);
  std::vector<float> v;
  if (ti == nullptr) { return v; }
  std::size_t n = 1;
  for (auto d : ti->shape) { n *= (std::size_t)d; }
  SharedBuffer b = w.load(name, mc);
  if (b.empty()) { return v; }
  v.resize(n);
  if (ti->dtype == "F32") {
    std::memcpy(v.data(), b.contents(), n * 4);
  } else if (ti->dtype == "F16") {
    const auto* s = static_cast<const _Float16*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) { v[i] = (float)s[i]; }
  } else {   // BF16
    const auto* s = static_cast<const std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) { v[i] = bf16f(s[i]); }
  }
  return v;
}

double
rel_l2(const std::vector<float>& a, const std::vector<float>& b)
{
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d; den += (double)b[i] * (double)b[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

std::string
lora_path_(const std::string& root)
{
  const char* e = std::getenv("VPIPE_KREA2_LORA");
  if (e != nullptr && *e != '\0') { return e; }
  namespace fs = std::filesystem;
  return (fs::path(root).parent_path() / "Krea-2-LoRA-softwatercolor" /
          "softwatercolor.safetensors").string();
}

// Fuse the LoRA into the DiT -> a cached <root>-dit-lora-<stem> dir (keyed by
// the LoRA filename so different adapters don't collide).
std::string
ensure_fused_(MetalCompute* mc, const std::string& root, const std::string& lora)
{
  namespace fs = std::filesystem;
  const std::string stem = fs::path(lora).stem().string();
  const std::string dst = root + "-dit-lora-" + stem;
  if (fs::exists(fs::path(dst) / "config.json")) { return dst; }
  std::string err;
  if (!fuse_lora(mc, root + "/transformer", lora, dst, 1.0f, &err)) {
    std::printf("[krea2_lora] fuse failed: %s\n", err.c_str());
    return {};
  }
  return dst;
}

}  // namespace

TEST(krea2_lora, fusion_matches_base_plus_ba)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  const std::string lora = lora_path_(root);
  if (!std::filesystem::exists(lora)) {
    std::printf("[krea2_lora] no LoRA at %s; skipping\n", lora.c_str());
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  const std::string fused = ensure_fused_(mc, std::string(root), lora);
  ASSERT_TRUE(!fused.empty());

  auto baseopt = MetalLlamaWeights::open_model(std::string(root) + "/transformer");
  auto fuseopt = MetalLlamaWeights::open_model(fused);
  auto loraopt = MetalLlamaWeights::open(lora);
  ASSERT_TRUE(baseopt.has_value() && fuseopt.has_value() && loraopt.has_value());

  // One adapted weight must change vs base after fusion.
  const std::string nm = "transformer_blocks.0.attn.to_q";
  const std::vector<float> Wb = tensor_f32(*baseopt, nm + ".weight", mc);
  const std::vector<float> Wf = tensor_f32(*fuseopt, nm + ".weight", mc);
  ASSERT_TRUE(!Wb.empty() && !Wf.empty());
  const double rb = rel_l2(Wf, Wb);       // fused should DIFFER from base
  EXPECT_TRUE(rb > 1e-3);                  // the LoRA actually changed the weight

  // For diffusers-named low-rank LoRAs (softwatercolor) also verify the exact
  // base + B@A. ai-toolkit / LoKr adapters use other key names + math, verified
  // against a numpy reference outside this test.
  const std::vector<float> A =
      tensor_f32(*loraopt, "transformer." + nm + ".lora_A.weight", mc);
  const std::vector<float> B =
      tensor_f32(*loraopt, "transformer." + nm + ".lora_B.weight", mc);
  if (!A.empty() && !B.empty()) {
    const auto* bi = baseopt->info(nm + ".weight");
    const int N = (int)bi->shape[0], K = (int)bi->shape[1];
    const int rank =
        (int)loraopt->info("transformer." + nm + ".lora_A.weight")->shape[0];
    ASSERT_TRUE((int)A.size() == rank * K && (int)B.size() == N * rank);
    std::vector<float> ref = Wb;                   // ref = base + B@A
    for (int n = 0; n < N; ++n) {
      float* r = ref.data() + (std::size_t)n * K;
      const float* br = B.data() + (std::size_t)n * rank;
      for (int rr = 0; rr < rank; ++rr) {
        const float b = br[rr];
        const float* ar = A.data() + (std::size_t)rr * K;
        for (int k = 0; k < K; ++k) { r[k] += b * ar[k]; }
      }
    }
    const double rf = rel_l2(Wf, ref);
    std::printf("[krea2_lora] to_q fused vs base+BA rel-L2 = %.6f (vs base = "
                "%.4f)\n", rf, rb);
    EXPECT_TRUE(rf < 0.01);        // matches base + B@A (bf16 rounding)
  } else {
    std::printf("[krea2_lora] to_q fused vs base rel-L2 = %.4f (non-diffusers "
                "adapter; exact match checked in numpy)\n", rb);
  }

  // A non-adapted tensor (a norm) passes through unchanged.
  const std::vector<float> nb = tensor_f32(*baseopt, nm.substr(0, 21) + "norm1.weight", mc);
  const std::vector<float> nf = tensor_f32(*fuseopt, nm.substr(0, 21) + "norm1.weight", mc);
  if (!nb.empty() && !nf.empty()) {
    EXPECT_TRUE(rel_l2(nf, nb) < 1e-6);
  }
}

TEST(krea2_lora, fused_dit_loads_and_runs)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_KREA2_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  const std::string lora = lora_path_(root);
  if (!std::filesystem::exists(lora)) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string fused = ensure_fused_(mc, std::string(root), lora);
  ASSERT_TRUE(!fused.empty());

  auto rd = [&](const std::string& p) {
    std::ifstream in(std::string(gd) + "/" + p, std::ios::binary);
    std::vector<float> v;
    if (!in) { return v; }
    in.seekg(0, std::ios::end); const std::streamoff n = in.tellg();
    in.seekg(0, std::ios::beg); v.resize((std::size_t)n / 4);
    in.read(reinterpret_cast<char*>(v.data()), n);
    return v;
  };
  const std::vector<float> txt = rd("d_txtin.f32");
  const std::vector<float> latin = rd("a3_step0_latin.f32");
  const std::vector<float> tstep = rd("a3_step0_tstep.f32");
  const std::vector<float> noise = rd("a3_step0_noise.f32");
  ASSERT_TRUE(!txt.empty() && !latin.empty() && !tstep.empty() && !noise.empty());

  MetalKrea2Transformer::Config cfg;
  const int HID = cfg.hidden, IC = cfg.in_channels;
  const int text_seq = (int)(txt.size() / HID);
  const int img_seq = (int)(latin.size() / IC);
  int grid = 1;
  while (grid * grid < img_seq) { ++grid; }

  auto m = MetalKrea2Transformer::load(fused, mc, cfg);
  ASSERT_TRUE(m != nullptr);          // the fused DiT reloads through bf16
  auto buf = [&](const std::vector<float>& s) {
    SharedBuffer b = mc->make_shared_buffer(s.size() * 2);
    auto* d = static_cast<_Float16*>(b.contents());
    for (std::size_t i = 0; i < s.size(); ++i) { d[i] = (_Float16)s[i]; }
    return b;
  };
  SharedBuffer out = m->forward_dit(buf(txt), text_seq, buf(latin), img_seq,
                                    grid, grid, tstep[0], -1);
  ASSERT_TRUE(!out.empty());
  const auto* op = static_cast<const _Float16*>(out.contents());
  std::vector<float> got(noise.size());
  bool finite = true;
  for (std::size_t i = 0; i < got.size(); ++i) {
    got[i] = (float)op[i];
    if (!std::isfinite(got[i])) { finite = false; }
  }
  EXPECT_TRUE(finite);
  // The LoRA changes the velocity vs the base DiT (a3_step0_noise), but it
  // stays in the same ballpark (a style adapter, not a different model).
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double d = got[i] - noise[i]; num += d * d; den += noise[i] * noise[i];
  }
  const double r = den > 0 ? std::sqrt(num / den) : 0.0;
  std::printf("[krea2_lora] fused DiT velocity vs base rel-L2 = %.4f\n", r);
  EXPECT_TRUE(r > 1e-3 && r < 1.5);   // differs from base, but sane
}

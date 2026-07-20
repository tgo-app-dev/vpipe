// Qwen-Image-Edit-2511 text encoder (M3): the Qwen2.5-VL backbone on
// MetalQwenModel. Unlike Qwen3-VL (Krea-2's encoder) it has q/k/v attention
// BIAS and NO per-head q/k RMSNorm -- the two new dense-path knobs (Config
// attention_bias / qk_norm). GQA 28:4, SwiGLU, plain 1D RoPE (theta 1e6).
//
// Verified against the diffusers golden (text_golden.py, text-only prompt):
// feed the exact templated token ids, host-gather embeds, tap the LAST layer
// (forward_embeddings_taps at n_layers-1 == HF hidden_states[-1] PRE final-
// norm), and rel-L2 against g_txt_prenorm. A second check applies the final
// RMSNorm on the host and matches g_txt (POST-norm = what the DiT conditions
// on; the all-positions normed tap itself is M6).
//
// Env: VPIPE_QWEN_IMAGE_EDIT_TEST_MODEL_PATH (uses <root>/text_encoder),
// VPIPE_QWEN_IMAGE_EDIT_GOLDEN (golden dir). Skips if unset.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"
#include "generative-models/context-manager.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/qwen3/metal-qwen-model.h"

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
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

float bf16_to_f32_(std::uint16_t b)
{
  std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

std::vector<float>
read_f32_(const std::string& path)
{
  std::ifstream in(path, std::ios::binary);
  std::vector<float> out;
  if (!in) { return out; }
  in.seekg(0, std::ios::end);
  const std::streamoff n = in.tellg();
  in.seekg(0, std::ios::beg);
  out.resize((std::size_t)n / 4);
  in.read(reinterpret_cast<char*>(out.data()), n);
  return out;
}

std::vector<std::int32_t>
read_i32_(const std::string& path)
{
  std::ifstream in(path, std::ios::binary);
  std::vector<std::int32_t> out;
  if (!in) { return out; }
  in.seekg(0, std::ios::end);
  const std::streamoff n = in.tellg();
  in.seekg(0, std::ios::beg);
  out.resize((std::size_t)n / 4);
  in.read(reinterpret_cast<char*>(out.data()), n);
  return out;
}

double
rel_l2_(const float* a, const float* b, std::size_t n)
{
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d;
    den += (double)b[i] * (double)b[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

// Qwen2.5-VL-7B text backbone (Qwen-Image-Edit-2511 text_encoder). Dense bf16;
// the two new knobs are attention_bias=true + qk_norm=false.
MetalQwenModel::Config qwen25_encoder_config_()
{
  MetalQwenModel::Config c;
  c.n_layers           = 28;
  c.hidden             = 3584;
  c.n_heads            = 28;
  c.n_kv_heads         = 4;
  c.head_dim           = 128;
  c.ffn_inner          = 18944;
  c.vocab              = 152064;
  c.rope_theta         = 1.0e6f;
  c.rms_eps            = 1e-6f;
  c.rotary_dim         = 128;      // full rotary
  c.full_attn_interval = 1;
  c.tie_embeddings     = false;    // untied lm_head (backbone_only skips it)
  c.use_bf16           = true;
  c.dense              = true;
  c.zero_centered_norm = false;
  c.attn_output_gate   = false;    // q_proj = qd (no gate)
  c.qk_norm            = false;    // Qwen2.5-VL: NO per-head q/k RMSNorm
  c.attention_bias     = true;     // Qwen2.5-VL: q/k/v bias
  c.backbone_only      = true;
  c.weight_prefix      = "";
  c.model_seg          = "model.";  // names are model.layers.N
  c.max_seq            = 512;
  c.page_tokens        = 256;
  return c;
}

}  // namespace

TEST(qwen_image_edit_text, backbone_matches_golden)
{
  const char* root = std::getenv("VPIPE_QWEN_IMAGE_EDIT_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_QWEN_IMAGE_EDIT_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  const std::string enc_dir = std::string(root) + "/text_encoder";
  const std::string gdir    = gd;

  const std::vector<std::int32_t> ids = read_i32_(gdir + "/g_txt_ids.i32");
  const std::vector<float> gpre = read_f32_(gdir + "/g_txt_prenorm.f32");
  const std::vector<float> gpost = read_f32_(gdir + "/g_txt.f32");
  if (ids.empty() || gpre.empty()) { return; }

  const int H = 3584, NL = 28;
  const int n = (int)ids.size();
  ASSERT_TRUE(gpre.size() == (std::size_t)n * H);

  auto m = MetalQwenModel::load(enc_dir, mc, qwen25_encoder_config_());
  ASSERT_TRUE(m != nullptr);

  auto wts = MetalLlamaWeights::open_model(enc_dir);
  ASSERT_TRUE(wts.has_value());
  SharedBuffer emb = wts->load("model.embed_tokens.weight", mc);
  ASSERT_TRUE(!emb.empty());

  // Host-gather the n embedding rows (bf16) by id.
  SharedBuffer x = mc->make_shared_buffer((std::size_t)n * H * 2);
  ASSERT_TRUE(!x.empty());
  const auto* tbl = static_cast<const std::uint8_t*>(emb.contents());
  auto* xb = static_cast<std::uint8_t*>(x.contents());
  for (int i = 0; i < n; ++i) {
    std::memcpy(xb + (std::size_t)i * H * 2,
                tbl + (std::size_t)ids[(std::size_t)i] * H * 2,
                (std::size_t)H * 2);
  }

  // Tap the LAST layer (un-normed) == HF hidden_states[-1] pre final-norm.
  const std::vector<int> tap_layers = {NL - 1};
  ContextManager* cm = m->context_manager();
  const ContextId cid = cm->acquire_root();
  SharedBuffer taps = m->forward_embeddings_taps(cid, x, n, tap_layers);
  cm->release(cid);
  ASSERT_TRUE(!taps.empty());
  ASSERT_TRUE(taps.byte_size() >= (std::size_t)n * H * 2);

  const auto* tp = static_cast<const std::uint16_t*>(taps.contents());
  std::vector<float> got((std::size_t)n * H);
  for (std::size_t i = 0; i < got.size(); ++i) { got[i] = bf16_to_f32_(tp[i]); }

  const double rpre = rel_l2_(got.data(), gpre.data(), got.size());
  std::printf("[qwen_image_edit_text] backbone (pre-norm) rel-L2 = %.6f (n=%d)\n",
              rpre, n);
  EXPECT_TRUE(rpre < 0.03);

  // Apply the final RMSNorm on the host and match the POST-norm golden (what the
  // DiT conditions on). RMSNorm: y = x / sqrt(mean(x^2)+eps) * weight.
  if (!gpost.empty()) {
    SharedBuffer nw = wts->load("model.norm.weight", mc);
    ASSERT_TRUE(!nw.empty());
    std::vector<float> w(H);
    const auto* nwp = static_cast<const std::uint16_t*>(nw.contents());
    for (int h = 0; h < H; ++h) { w[h] = bf16_to_f32_(nwp[h]); }
    std::vector<float> normed((std::size_t)n * H);
    for (int i = 0; i < n; ++i) {
      double ss = 0.0;
      for (int h = 0; h < H; ++h) {
        const double v = got[(std::size_t)i * H + h];
        ss += v * v;
      }
      const double inv = 1.0 / std::sqrt(ss / (double)H + 1e-6);
      for (int h = 0; h < H; ++h) {
        normed[(std::size_t)i * H + h] =
            (float)((double)got[(std::size_t)i * H + h] * inv * (double)w[h]);
      }
    }
    const double rpost = rel_l2_(normed.data(), gpost.data(), normed.size());
    std::printf("[qwen_image_edit_text] post-norm rel-L2 = %.6f\n", rpost);
    EXPECT_TRUE(rpost < 0.03);
  }
}

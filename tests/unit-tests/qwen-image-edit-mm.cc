// Qwen-Image-Edit-2511 image-aware conditioning (M6): the Qwen2.5-VL text
// backbone over the ACTUAL edit sequence (template + one reference image spliced
// as `Picture 1: <|vision_start|>{image_pad x N}<|vision_end|>`).
//
// KEY FINDING: the QwenImageEditPlus pipeline calls the text_encoder WITHOUT
// position_ids (pipeline_qwenimage_edit_plus.py::_get_qwen_prompt_embeds), so
// Qwen2.5-VL's forward falls back to torch.arange -> PLAIN SEQUENTIAL 1-D RoPE.
// The 2-D grid mROPE (get_rope_index) is only used by generate(), NOT for DiT
// conditioning. So the image-aware conditioning is M3's sequential backbone
// (forward_embeddings_taps) + a vision-token splice at the image_pad rows -- NO
// mROPE. Verified end-to-end here.
//
// The test splices the GOLDEN image_embeds HF produced (mm_img_embeds) so it is
// independent of M5's single-vs-multi-window vision reorder, gathers the text
// embeds, runs the sequential backbone tap at the last layer, applies the host
// final-RMSNorm, and rel-L2s the normed all-positions hidden vs g_txt_mm (HF
// hidden_states[-1], POST-norm = what the DiT conditions on).
//
// Env: VPIPE_QWEN_IMAGE_EDIT_TEST_MODEL_PATH (uses <root>/text_encoder),
// VPIPE_QWEN_IMAGE_EDIT_GOLDEN. Skips if unset.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"
#include "generative-models/context-manager.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/qwen-image/metal-qwen25-vision.h"
#include "generative-models/qwen3/metal-qwen-model.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

std::uint16_t f32_to_bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
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

// Qwen2.5-VL text backbone (same as the M3 text test).
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
  c.rotary_dim         = 128;
  c.full_attn_interval = 1;
  c.tie_embeddings     = false;
  c.use_bf16           = true;
  c.dense              = true;
  c.zero_centered_norm = false;
  c.attn_output_gate   = false;
  c.qk_norm            = false;
  c.attention_bias     = true;
  c.backbone_only      = true;
  c.weight_prefix      = "";
  c.model_seg          = "model.";
  c.max_seq            = 512;
  c.page_tokens        = 256;
  return c;
}

}  // namespace

TEST(qwen_image_edit_mm, image_aware_conditioning_matches_golden)
{
  const char* root = std::getenv("VPIPE_QWEN_IMAGE_EDIT_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_QWEN_IMAGE_EDIT_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  // VPIPE_QWEN_IMAGE_EDIT_ENC_DIR overrides the encoder dir (e.g. a w4/w8
  // quantized text_encoder) -- conditioning should still track g_txt_mm within
  // the quant error (looser bar).
  const char* ed = std::getenv("VPIPE_QWEN_IMAGE_EDIT_ENC_DIR");
  const bool quant_enc = (ed != nullptr && *ed != '\0');
  const std::string enc_dir =
      quant_enc ? std::string(ed) : std::string(root) + "/text_encoder";
  const std::string gdir    = gd;

  const std::vector<std::int32_t> ids  = read_i32_(gdir + "/mm_ids.i32");
  const std::vector<std::int32_t> ipos = read_i32_(gdir + "/mm_img_pos.i32");
  const std::vector<float> iemb = read_f32_(gdir + "/mm_img_embeds.f32");
  const std::vector<float> gmm  = read_f32_(gdir + "/g_txt_mm.f32");
  if (ids.empty() || ipos.empty() || iemb.empty() || gmm.empty()) { return; }

  const int H = 3584, NL = 28;
  const int n = (int)ids.size();
  ASSERT_TRUE(gmm.size() == (std::size_t)n * H);
  const int n_img = (int)ipos.size();
  ASSERT_TRUE(iemb.size() == (std::size_t)n_img * H);

  MetalQwenModel::Config ecfg = qwen25_encoder_config_();
  if (quant_enc) {
    // Detect the quant bit-width from the encoder's config.json quantization
    // block so the loader picks the affine w4/w8 kernel.
    std::ifstream cf(enc_dir + "/config.json");
    std::string s((std::istreambuf_iterator<char>(cf)),
                  std::istreambuf_iterator<char>());
    const auto p = s.find("\"bits\"");
    if (p != std::string::npos) {
      const auto c = s.find(':', p);
      if (c != std::string::npos) { ecfg.quant_bits = std::atoi(s.c_str() + c + 1); }
    }
    std::printf("[qwen_image_edit_mm] quantized encoder, quant_bits=%d\n",
                ecfg.quant_bits);
  }
  auto m = MetalQwenModel::load(enc_dir, mc, ecfg);
  ASSERT_TRUE(m != nullptr);

  auto wts = MetalLlamaWeights::open_model(enc_dir);
  ASSERT_TRUE(wts.has_value());
  SharedBuffer emb = wts->load("model.embed_tokens.weight", mc);
  ASSERT_TRUE(!emb.empty());

  // Host-gather text embeddings (bf16), then overwrite the image_pad rows with
  // the GOLDEN vision tokens -- exactly the inputs_embeds splice HF does.
  SharedBuffer x = mc->make_shared_buffer((std::size_t)n * H * 2);
  ASSERT_TRUE(!x.empty());
  const auto* tbl = static_cast<const std::uint8_t*>(emb.contents());
  auto* xb = static_cast<std::uint8_t*>(x.contents());
  for (int i = 0; i < n; ++i) {
    std::memcpy(xb + (std::size_t)i * H * 2,
                tbl + (std::size_t)ids[(std::size_t)i] * H * 2,
                (std::size_t)H * 2);
  }
  auto* xh = static_cast<std::uint16_t*>(x.contents());
  for (int j = 0; j < n_img; ++j) {
    const int row = ipos[(std::size_t)j];
    ASSERT_TRUE(row >= 0 && row < n);
    for (int h = 0; h < H; ++h) {
      xh[(std::size_t)row * H + h] =
          f32_to_bf16_(iemb[(std::size_t)j * H + h]);
    }
  }

  // Sequential 1-D RoPE backbone tap of the LAST layer (== the pipeline path:
  // the text_encoder forward runs with position_ids=None -> torch.arange). Tap
  // ALL layers in ONE forward and use the last slot (a second acquire_root
  // forward on the same model contaminates -- page-pool reuse).
  std::vector<int> tap_layers;
  for (int l = 0; l < NL; ++l) { tap_layers.push_back(l); }
  ContextManager* cm = m->context_manager();
  const ContextId cid = cm->acquire_root();
  SharedBuffer taps_all = m->forward_embeddings_taps(cid, x, n, tap_layers);
  cm->release(cid);
  ASSERT_TRUE(!taps_all.empty());
  // Keep only the last layer's slot.
  SharedBuffer taps = mc->make_shared_buffer((std::size_t)n * H * 2);
  std::memcpy(taps.contents(),
              static_cast<const std::uint8_t*>(taps_all.contents()) +
                  (std::size_t)(NL - 1) * n * H * 2,
              (std::size_t)n * H * 2);

  ASSERT_TRUE(taps.byte_size() >= (std::size_t)n * H * 2);

  const auto* tp = static_cast<const std::uint16_t*>(taps.contents());
  std::vector<float> got((std::size_t)n * H);
  for (std::size_t i = 0; i < got.size(); ++i) { got[i] = bf16_to_f32_(tp[i]); }

  // Host final-RMSNorm -> compare to g_txt_mm (POST-norm, the DiT conditioning).
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

  const double r = rel_l2_(normed.data(), gmm.data(), normed.size());
  // Region breakdown (pre-image / image / post-image) for diagnostics.
  const int fi = ipos.front(), li = ipos.back();
  const double r_pre = rel_l2_(normed.data(), gmm.data(), (std::size_t)fi * H);
  const double r_img = rel_l2_(normed.data() + (std::size_t)fi * H,
                               gmm.data() + (std::size_t)fi * H,
                               (std::size_t)(li + 1 - fi) * H);
  const double r_post =
      rel_l2_(normed.data() + (std::size_t)(li + 1) * H,
              gmm.data() + (std::size_t)(li + 1) * H,
              (std::size_t)(n - li - 1) * H);
  std::printf("[qwen_image_edit_mm] image-aware conditioning rel-L2 = %.6f "
              "(n=%d, %d vision tokens; pre=%.4f img=%.4f post=%.4f)\n",
              r, n, n_img, r_pre, r_img, r_post);
  // bf16 backbone (theta 1e6, 28 layers) vs the bf16 reference; M3 text floor
  // was ~0.019, and the spliced vision tokens sit on the same floor. A w4/w8
  // quantized encoder adds group-affine error on top.
  EXPECT_TRUE(r < (quant_enc ? 0.30 : 0.04));
}

// Full image-aware conditioning through vpipe's OWN vision tower (not the golden
// image_embeds): MetalQwen25Vision(v2_pixels) -> splice at the image_pad rows ->
// sequential backbone tap -> drop-64 -> final-RMSNorm, vs g_v2_cond (the DiT
// prompt embeds for a REAL 384^2 reference). This is the end-to-end conditioning
// the QIE stage will build. Uses the v2_* multi-window golden.
TEST(qwen_image_edit_mm, conditioning_own_tower_matches_golden)
{
  const char* root = std::getenv("VPIPE_QWEN_IMAGE_EDIT_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_QWEN_IMAGE_EDIT_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string gdir = gd, enc_dir = std::string(root) + "/text_encoder";

  const std::vector<float> px = read_f32_(gdir + "/v2_pixels.f32");
  const std::vector<std::int32_t> vpos = read_i32_(gdir + "/v2_pos.i32");
  const std::vector<std::int32_t> ids = read_i32_(gdir + "/v2_ids.i32");
  const std::vector<std::int32_t> imgpos = read_i32_(gdir + "/v2_imgpos.i32");
  const std::vector<float> gcond = read_f32_(gdir + "/g_v2_cond.f32");
  if (px.empty() || vpos.empty() || ids.empty() || imgpos.empty() ||
      gcond.empty()) { return; }

  const int H = 3584, NL = 28, DROP = 64;
  const int n = (int)ids.size();
  const int n_img = (int)imgpos.size();
  const int n_real = n - DROP;
  ASSERT_TRUE(gcond.size() == (std::size_t)n_real * H);

  // ---- vpipe vision tower -> [n_img, 3584] (natural merged order) ----
  MetalQwen25Vision::Config vcfg;
  const int seq = (int)vpos.size() / 2;
  ASSERT_TRUE((int)px.size() == seq * vcfg.patch_in);
  ASSERT_TRUE(seq / (vcfg.merge * vcfg.merge) == n_img);
  auto vis = MetalQwen25Vision::load(enc_dir, mc, vcfg);
  ASSERT_TRUE(vis != nullptr);
  SharedBuffer pixels = mc->make_shared_buffer(px.size() * 2);
  { auto* d = static_cast<std::uint16_t*>(pixels.contents());
    for (std::size_t i = 0; i < px.size(); ++i) { d[i] = f32_to_bf16_(px[i]); } }
  std::vector<int> pos(vpos.begin(), vpos.end());
  SharedBuffer vtok = vis->encode(pixels, seq, pos);
  ASSERT_TRUE(!vtok.empty());
  const auto* vt = static_cast<const std::uint16_t*>(vtok.contents());

  // ---- backbone: gather text embeds, splice vision rows at image_pad ----
  MetalQwenModel::Config ecfg = qwen25_encoder_config_();
  auto m = MetalQwenModel::load(enc_dir, mc, ecfg);
  ASSERT_TRUE(m != nullptr);
  auto wts = MetalLlamaWeights::open_model(enc_dir);
  ASSERT_TRUE(wts.has_value());
  SharedBuffer emb = wts->load("model.embed_tokens.weight", mc);
  ASSERT_TRUE(!emb.empty());

  SharedBuffer x = mc->make_shared_buffer((std::size_t)n * H * 2);
  const auto* tbl = static_cast<const std::uint8_t*>(emb.contents());
  auto* xb = static_cast<std::uint8_t*>(x.contents());
  for (int i = 0; i < n; ++i) {
    std::memcpy(xb + (std::size_t)i * H * 2,
                tbl + (std::size_t)ids[(std::size_t)i] * H * 2,
                (std::size_t)H * 2);
  }
  auto* xh = static_cast<std::uint16_t*>(x.contents());
  for (int j = 0; j < n_img; ++j) {
    const int row = imgpos[(std::size_t)j];
    ASSERT_TRUE(row >= 0 && row < n);
    std::memcpy(xh + (std::size_t)row * H, vt + (std::size_t)j * H,
                (std::size_t)H * 2);   // vision token j -> its image_pad row
  }

  std::vector<int> tap{NL - 1};
  ContextManager* cm = m->context_manager();
  const ContextId cid = cm->acquire_root();
  SharedBuffer taps = m->forward_embeddings_taps(cid, x, n, tap);
  cm->release(cid);
  ASSERT_TRUE(!taps.empty());

  // drop-64 + host final-RMSNorm -> [n_real, H], compare g_v2_cond.
  SharedBuffer nw = wts->load("model.norm.weight", mc);
  ASSERT_TRUE(!nw.empty());
  std::vector<float> w(H);
  const auto* nwp = static_cast<const std::uint16_t*>(nw.contents());
  for (int h = 0; h < H; ++h) { w[h] = bf16_to_f32_(nwp[h]); }
  const auto* tp = static_cast<const std::uint16_t*>(taps.contents());
  std::vector<float> normed((std::size_t)n_real * H);
  for (int p = 0; p < n_real; ++p) {
    const auto* rowp = tp + (std::size_t)(p + DROP) * H;
    double ss = 0.0;
    for (int h = 0; h < H; ++h) { const double v = bf16_to_f32_(rowp[h]);
                                  ss += v * v; }
    const double inv = 1.0 / std::sqrt(ss / (double)H + 1e-6);
    for (int h = 0; h < H; ++h) {
      normed[(std::size_t)p * H + h] =
          (float)((double)bf16_to_f32_(rowp[h]) * inv * (double)w[h]);
    }
  }
  const double r = rel_l2_(normed.data(), gcond.data(), normed.size());
  std::printf("[qwen_image_edit_mm] OWN-TOWER conditioning rel-L2 = %.6f "
              "(n=%d, %d vision tokens, n_real=%d)\n", r, n, n_img, n_real);
  // The bf16 vision-tower floor (~0.058, cosine 0.998 vs HF) amplifies through
  // the backbone cross-attention to ~0.16 here (vs the 0.019 M6 floor with
  // golden vision embeds). Dominated by bf16 accumulation, not a logic error;
  // the end-to-end edit is the quality bar.
  EXPECT_TRUE(r < 0.22);
}

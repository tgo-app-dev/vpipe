// detect_quant_arch: architecture introspection for the model-quantize stage.
// Validates the config.json + safetensors-probe derivation of layer_prefix /
// n_layers and the AWQ / on-device-calibration capability flags across the
// model families vpipe supports. Env-gated on the per-family test model dirs;
// each case skips vacuously when its model is absent. The probe only reads
// tensor NAMES, so the (already-quantized) test checkpoints exercise it fine.

#include "minitest.h"

#include "common/session.h"
#include "generative-models/quantize/arch-detect.h"

#include <cstdlib>
#include <filesystem>
#include <string>

using namespace vpipe;

namespace {
bool has_model_(const char* d)
{
  return d != nullptr && *d != '\0' &&
         std::filesystem::exists(std::filesystem::path(d) / "config.json");
}
}  // namespace

// MOSS-TTS-Local-v1.5: dense Qwen3 backbone under "transformer." -> AWQ +
// on-device calibration both supported.
TEST(quant_arch_detect, moss_tts_local_dense_calibratable)
{
  const char* d = std::getenv("VPIPE_MOSS_TTS_LOCAL_MODEL");
  if (!has_model_(d)) { return; }
  Session sess;
  const auto m = genai::detect_quant_arch(&sess, d);
  EXPECT_TRUE(m.detected);
  EXPECT_TRUE(m.arch == "moss-tts-local");
  EXPECT_TRUE(m.layer_prefix == "transformer.layers.");
  EXPECT_TRUE(m.n_layers == 36);
  EXPECT_TRUE(m.n_attn_layers == 36);
  EXPECT_TRUE(m.awq_ok);
  EXPECT_TRUE(m.calib_ok);
  EXPECT_TRUE(m.backbone.weight_prefix == "transformer.");
  EXPECT_TRUE(m.backbone.model_seg.empty());
  EXPECT_TRUE(m.backbone.n_layers == 36);
  EXPECT_TRUE(m.backbone.hidden == 2560);
}

// Qwen3.5 hybrid: only the periodic full-attention layers carry self_attn.
// q_proj (the rest are gated-DeltaNet), but BOTH block types start with
// input_layernorm, so AWQ folds the in-projection group on every layer ->
// awq_ok true. On-device calibration runs the hybrid backbone through
// MetalQwenModel (config_from), so calib_ok true. The multimodal
// "language_model.model." wrap is probed.
TEST(quant_arch_detect, qwen35_hybrid_supported)
{
  const char* d = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!has_model_(d)) { return; }
  Session sess;
  const auto m = genai::detect_quant_arch(&sess, d);
  EXPECT_TRUE(m.detected);
  EXPECT_TRUE(m.arch == "qwen3.5");
  EXPECT_TRUE(m.layer_prefix == "language_model.model.layers.");
  EXPECT_TRUE(m.n_layers == 32);
  EXPECT_TRUE(m.n_attn_layers > 0 && m.n_attn_layers < m.n_layers);  // hybrid
  EXPECT_TRUE(m.awq_ok);      // GDN in-proj group folds into input_layernorm
  EXPECT_TRUE(m.calib_ok);    // hybrid backbone runs via MetalQwenModel
  // config_from produced a real hybrid backbone (not dense).
  EXPECT_FALSE(m.backbone.dense);
  EXPECT_TRUE(m.backbone.n_layers == 32);
  EXPECT_TRUE(m.backbone.full_attn_interval > 1);
  EXPECT_TRUE(m.backbone.gdn_v_heads > 0);
  EXPECT_TRUE(m.backbone.backbone_only);
  EXPECT_TRUE(m.backbone.quant_bits == 8);
}

// Gemma-4: dense attention, but the gate/up input is pre_feedforward_layernorm
// (not post_attention_layernorm), so the standard AWQ fold target is wrong ->
// awq_ok must be false even though every layer has self_attn.q_proj.
TEST(quant_arch_detect, gemma4_ffn_norm_blocks_awq)
{
  const char* d = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!has_model_(d)) { return; }
  Session sess;
  const auto m = genai::detect_quant_arch(&sess, d);
  EXPECT_TRUE(m.detected);
  EXPECT_TRUE(m.arch == "gemma4");
  EXPECT_TRUE(m.n_layers > 0);
  EXPECT_TRUE(m.n_attn_layers == m.n_layers);   // dense attention...
  EXPECT_FALSE(m.awq_ok);                         // ...but Gemma FFN norms
  EXPECT_FALSE(m.calib_ok);
}

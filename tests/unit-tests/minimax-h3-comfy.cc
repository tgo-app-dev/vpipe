// Reading the Comfy-Org repack of MiniMax-H3.
//
// Comfy-Org republishes upstream diffusers models as one .safetensors
// per component under diffusion_models/ | text_encoders/ | vae/, with
// each component's config inside the file's `__metadata__` instead of a
// config.json. Two things then have to work that do not for a diffusers
// directory: the config has to be lifted out of the safetensors header,
// and the DiT's fused qkv projection has to be read FLAT rather than
// per-head -- a difference with no signature in the tensor names or
// shapes, so a wrong guess loads cleanly and computes nonsense.
//
// These tests need no weights: a header carrying only `__metadata__` is
// enough to check the plumbing, and doing it that way is the point --
// the file-format contract is what a future Comfy-Org repo (Wan-Animate-2
// carries the same tag and the same tree) will meet first, long before
// anyone downloads 66 GB.
//
// The NUMERICAL claim -- that the flat read computes the same function
// as the per-head one -- is minimax_h3_dit.comfy_layout_matches_golden,
// which needs real weights.

#include "minitest.h"

#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/shared/comfy-checkpoint.h"
#include "common/flex-data.h"
#include "stages/model-detect.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

using namespace vpipe;
using namespace vpipe::genai;

namespace {

namespace fs = std::filesystem;

// The transformer config Comfy-Org embeds, verbatim from
// diffusion_models/minimax_h3_fl2va_bf16.safetensors. Kept here in full
// rather than trimmed to the fields under test: the released
// config.json is the other half of this comparison, and a subset would
// quietly stop checking the fields it dropped.
constexpr const char* kH3Config =
    "{\"transformer\": {\"hidden_size\": 5376, \"num_layers\": 50, "
    "\"token_refiner_num_layers\": 2, \"num_attention_heads\": 56, "
    "\"attention_head_dim\": 128, \"ffn_hidden_size\": 14336, "
    "\"latents_dim\": 24, \"audio_latents_dim\": 32, "
    "\"patch_size\": [1, 2, 2], \"text_dim\": 5120, "
    "\"timestep_input_dim\": 256, \"time_embed_hidden_size\": 5376, "
    "\"time_embed_dim\": 2688, \"adaln_out_features\": 96768, "
    "\"final_adaln_out_features\": 10752, \"rope_inv_freq_len\": 16, "
    "\"norm_eps\": 1e-05, \"qk_norm_eps\": 1e-05, "
    "\"final_norm_eps\": 1e-05, \"image_model\": \"minimax_h3\"}}";

// A safetensors file carrying `__metadata__` and one 4-byte tensor.
// That is a VALID safetensors -- the reader here only ever looks at the
// header, but writing a malformed one would test a parser that real
// files never meet.
void
write_component_(const fs::path& p, const std::string& meta_key,
                 const std::string& meta_val)
{
  FlexData hdr = FlexData::make_object();
  {
    auto o = hdr.as_object();
    FlexData md = FlexData::make_object();
    md.as_object().insert_or_assign(meta_key,
                                    FlexData::make_string(meta_val));
    o.insert_or_assign("__metadata__", std::move(md));
    FlexData t = FlexData::make_object();
    {
      auto to = t.as_object();
      to.insert_or_assign("dtype", FlexData::make_string("F32"));
      FlexData shape = FlexData::make_array();
      shape.as_array().push_back(FlexData::make_int(1));
      to.insert_or_assign("shape", std::move(shape));
      FlexData off = FlexData::make_array();
      {
        auto a = off.as_array();
        a.push_back(FlexData::make_int(0));
        a.push_back(FlexData::make_int(4));
      }
      to.insert_or_assign("data_offsets", std::move(off));
    }
    o.insert_or_assign("probe.weight", std::move(t));
  }
  const std::string json = hdr.to_json(false);
  fs::create_directories(p.parent_path());
  std::ofstream f(p, std::ios::binary);
  const std::uint64_t n = json.size();
  f.write(reinterpret_cast<const char*>(&n), 8);
  f.write(json.data(), (std::streamsize)json.size());
  const float zero = 0.0f;
  f.write(reinterpret_cast<const char*>(&zero), 4);
}

fs::path
scratch_()
{
  return fs::temp_directory_path() / "vpipe-h3-comfy-test";
}

}  // namespace

// A repo root, its diffusion_models/ subdir and the file itself must all
// resolve to the same file -- the catalogue registers the ROOT, so that
// is the only one a caller holding a registry path actually has.
TEST(minimax_h3_comfy, resolves_from_root_subdir_or_file)
{
  const fs::path root = scratch_() / "repo";
  std::error_code ec;
  fs::remove_all(root, ec);
  const fs::path dit =
      root / "diffusion_models" / "minimax_h3_fl2va_bf16.safetensors";
  write_component_(dit, "config", kH3Config);

  const std::string want = dit.string();
  EXPECT_TRUE(MetalMiniMaxH3Transformer::resolve_dit_dir(root.string())
              == want);
  EXPECT_TRUE(MetalMiniMaxH3Transformer::resolve_dit_dir(
                  (root / "diffusion_models").string()) == want);
  EXPECT_TRUE(MetalMiniMaxH3Transformer::resolve_dit_dir(want) == want);
  fs::remove_all(root, ec);
}

// Which file wins when the repo holds several. Comfy-Org ships five DiT
// variants; only the bf16 ones are readable here, and among those the
// plain conversion has to beat the `pruned` one -- a smaller checkpoint
// that is a DIFFERENT model, not a cheaper encoding of this one.
TEST(minimax_h3_comfy, prefers_the_plain_bf16_over_pruned_and_quantized)
{
  const fs::path root = scratch_() / "multi";
  std::error_code ec;
  fs::remove_all(root, ec);
  const fs::path d = root / "diffusion_models";
  for (const char* n : {"minimax_h3_fl2va_pruned_bf16.safetensors",
                        "minimax_h3_fl2va_bf16.safetensors",
                        "minimax_h3_fl2va_int8_convrot.safetensors",
                        "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"}) {
    write_component_(d / n, "config", kH3Config);
  }
  const std::string got =
      MetalMiniMaxH3Transformer::resolve_dit_dir(root.string());
  EXPECT_TRUE(got == (d / "minimax_h3_fl2va_bf16.safetensors").string());

  // The packings this build cannot read are rejected even when they are
  // the ONLY thing present -- silently loading Comfy-Org's int8_convrot
  // as bf16 would produce a model of the right shape and no meaning.
  const fs::path only = scratch_() / "quantonly";
  fs::remove_all(only, ec);
  write_component_(only / "diffusion_models" /
                       "minimax_h3_fl2va_int8_convrot.safetensors",
                   "config", kH3Config);
  EXPECT_TRUE(comfy::resolve_component(only.string(), "diffusion_models",
                                       "config", {"fl2va"}).empty());
  fs::remove_all(root, ec);
  fs::remove_all(only, ec);
}

// The config comes out of the header, and it arrives carrying the FLAT
// qkv grouping. That flag is the whole reason this format needs a branch
// rather than just a different path.
TEST(minimax_h3_comfy, config_comes_from_the_header_and_says_flat_qkv)
{
  const fs::path root = scratch_() / "cfg";
  std::error_code ec;
  fs::remove_all(root, ec);
  write_component_(root / "diffusion_models" /
                       "minimax_h3_fl2va_bf16.safetensors",
                   "config", kH3Config);

  MetalMiniMaxH3Transformer::Config cfg;
  std::string err;
  const bool ok =
      MetalMiniMaxH3Transformer::config_from_json(root.string(), cfg, &err);
  if (!ok) { std::printf("[minimax_h3_comfy] %s\n", err.c_str()); }
  ASSERT_TRUE(ok);
  EXPECT_TRUE(!cfg.qkv_per_head);
  EXPECT_TRUE(cfg.hidden == 5376 && cfg.n_heads == 56);
  EXPECT_TRUE(cfg.head_dim == 128 && cfg.inner() == 7168);
  EXPECT_TRUE(cfg.n_layers == 50 && cfg.n_refiner == 2);
  EXPECT_TRUE(cfg.ffn == 14336 && cfg.text_dim == 5120);
  EXPECT_TRUE(cfg.video_channels == 24 && cfg.audio_channels == 32);
  EXPECT_TRUE(cfg.patch_t == 1 && cfg.patch_h == 2 && cfg.patch_w == 2);
  EXPECT_TRUE(cfg.time_dim == 2688 && cfg.rope_freq_dim == 16);
  EXPECT_TRUE(cfg.adaln_out() == 96768);

  // A default-constructed Config is the RELEASED layout: the diffusers
  // checkpoint is the one this tree was written against, so per-head has
  // to be what a caller gets without asking.
  const MetalMiniMaxH3Transformer::Config dflt;
  EXPECT_TRUE(dflt.qkv_per_head);
  fs::remove_all(root, ec);
}

// A sibling Comfy-Org repo has the same tree, the same metadata key and
// the same field names -- so `image_model` is the only thing keeping a
// Wan file from parsing into an H3 config and loading into the wrong
// model. Refuse it rather than build a plausible config.
TEST(minimax_h3_comfy, another_architecture_is_refused)
{
  const fs::path root = scratch_() / "other";
  std::error_code ec;
  fs::remove_all(root, ec);
  write_component_(root / "diffusion_models" / "wan_animate_2_bf16.safetensors",
                   "config",
                   "{\"transformer\": {\"hidden_size\": 5120, "
                   "\"image_model\": \"wan2.2\"}}");
  MetalMiniMaxH3Transformer::Config cfg;
  std::string err;
  EXPECT_TRUE(!MetalMiniMaxH3Transformer::config_from_json(root.string(),
                                                           cfg, &err));
  EXPECT_TRUE(err.find("wan2.2") != std::string::npos);
  fs::remove_all(root, ec);
}

// Detection, which is what makes a hand-placed checkout usable: there is
// no transformer/config.json for the diffusers probe to find, so without
// this the directory registers as unrecognized.
TEST(minimax_h3_comfy, detects_as_fl2va_and_records_the_format)
{
  const fs::path root = scratch_() / "detect";
  std::error_code ec;
  fs::remove_all(root, ec);
  write_component_(root / "diffusion_models" /
                       "minimax_h3_fl2va_bf16.safetensors",
                   "config", kH3Config);
  const DetectedModel d = detect_model_dir(root.string(), "");
  EXPECT_TRUE(d.model_type == "minimax-h3-fl2va");
  EXPECT_TRUE(d.weight_format == "comfyui");
  EXPECT_TRUE(d.detected_by == "comfyui");
  EXPECT_TRUE(!d.outputs.empty());

  // Ref2VA is a different packed layout that this tree does not
  // implement. Tagging it would claim support we do not have, so it
  // stays untagged -- and untagged is not the same as broken: it still
  // registers as a directory.
  const fs::path r2 = scratch_() / "detect-ref2va";
  fs::remove_all(r2, ec);
  write_component_(r2 / "diffusion_models" /
                       "minimax_h3_ref2va_bf16.safetensors",
                   "config", kH3Config);
  EXPECT_TRUE(detect_model_dir(r2.string(), "").model_type.empty());
  fs::remove_all(root, ec);
  fs::remove_all(r2, ec);
}

// A checkpoint DERIVED from the repack -- model-quantize's output. It is
// an ordinary diffusers directory, so nothing about its path or its
// tensors says which qkv order it copied through; the producer writes
// `qkv_per_head` into config.json and the loader has to believe it.
// Without this the quantized form of a Comfy-Org DiT loads as the
// released layout and computes nonsense, which is the one way this
// feature could ship a silently-wrong model.
TEST(minimax_h3_comfy, an_explicit_flag_in_config_json_wins)
{
  const fs::path dir = scratch_() / "derived";
  std::error_code ec;
  fs::remove_all(dir, ec);
  fs::create_directories(dir, ec);
  auto write_cfg = [&](const char* extra) {
    std::ofstream f(dir / "config.json");
    f << "{\"_class_name\": \"MiniMaxH3DiTModel\", \"hidden_size\": 5376, "
         "\"num_attention_heads\": 56, \"attention_head_dim\": 128, "
         "\"adaln_out_features\": 96768, \"final_adaln_out_features\": 10752"
      << extra << "}";
  };
  MetalMiniMaxH3Transformer::Config cfg;
  std::string err;

  write_cfg(", \"qkv_per_head\": false");
  ASSERT_TRUE(
      MetalMiniMaxH3Transformer::config_from_json(dir.string(), cfg, &err));
  EXPECT_TRUE(!cfg.qkv_per_head);

  // Absent, a directory checkpoint is the RELEASED layout -- every
  // diffusers H3 in the wild is, and quantize only writes the key when
  // the source was not.
  write_cfg("");
  cfg = MetalMiniMaxH3Transformer::Config{};
  ASSERT_TRUE(
      MetalMiniMaxH3Transformer::config_from_json(dir.string(), cfg, &err));
  EXPECT_TRUE(cfg.qkv_per_head);
  fs::remove_all(dir, ec);
}

// WHOLE-REPO detection: what is in this checkout?
//
// The per-component resolvers answer "where is the DiT?"; this answers
// "what is here?". It matters because a repack checkout is routinely a
// SUBSET -- the catalogue pins a `files` whitelist, downloads are
// interrupted, and unreadable packings are skipped -- and which subset
// is present decides what can run. A caller that cannot ask ends up
// diagnosing a missing component as a broken model.
TEST(minimax_h3_comfy, scan_repo_lists_every_readable_component)
{
  const fs::path root = scratch_() / "whole";
  std::error_code ec;
  fs::remove_all(root, ec);
  write_component_(root / "diffusion_models" /
                       "minimax_h3_fl2va_bf16.safetensors",
                   "config", kH3Config);
  write_component_(root / "vae" / "minimax_h3_video_vae_fp16.safetensors",
                   "minimax_h3_video_vae", "{\"vae_clip_length\": 17}");
  write_component_(root / "vae" / "minimax_h3_audio_vae_fp32.safetensors",
                   "minimax_h3_audio_vae", "{\"sample_rate\": 32000}");
  write_component_(root / "text_encoders" /
                       "qwen3vl_32b_minimax_h3_bf16.safetensors",
                   "minimax_h3_te", "{\"num_hidden_layers\": 50}");
  // Skipped: this build cannot read Comfy-Org's own packings, so an
  // inventory that listed them would promise a component that will not
  // load.
  write_component_(root / "diffusion_models" /
                       "minimax_h3_fl2va_int8_convrot.safetensors",
                   "config", kH3Config);

  const std::vector<comfy::Component> got = comfy::scan_repo(root.string());
  EXPECT_TRUE(got.size() == 4u);
  // Stable order: by role in the fixed list, then filename.
  EXPECT_TRUE(got[0].role == "diffusion_models" && got[0].meta_key == "config");
  EXPECT_TRUE(got[1].role == "text_encoders" &&
              got[1].meta_key == "minimax_h3_te");
  EXPECT_TRUE(got[2].role == "vae" &&
              got[2].meta_key == "minimax_h3_audio_vae");
  EXPECT_TRUE(got[3].role == "vae" &&
              got[3].meta_key == "minimax_h3_video_vae");
  for (const auto& c : got) {
    EXPECT_TRUE(c.file.find("int8_convrot") == std::string::npos);
  }

  // A PARTIAL checkout is the normal case, not an error -- the
  // catalogue's whitelist alone makes one (no text encoder pinned).
  const fs::path part = scratch_() / "partial";
  fs::remove_all(part, ec);
  write_component_(part / "vae" / "minimax_h3_audio_vae_fp32.safetensors",
                   "minimax_h3_audio_vae", "{\"sample_rate\": 32000}");
  const std::vector<comfy::Component> p = comfy::scan_repo(part.string());
  EXPECT_TRUE(p.size() == 1u && p[0].role == "vae");

  // Not a Comfy-Org repo at all -> empty, which is how a caller tells.
  EXPECT_TRUE(comfy::scan_repo((scratch_() / "nope").string()).empty());
  fs::remove_all(root, ec);
  fs::remove_all(part, ec);
}

// The record has to SAY which components a checkout holds, so a later
// failure reads as "this copy has no text encoder" rather than as an
// unsupported model.
TEST(minimax_h3_comfy, detection_records_which_components_are_present)
{
  const fs::path root = scratch_() / "detect-parts";
  std::error_code ec;
  fs::remove_all(root, ec);
  write_component_(root / "diffusion_models" /
                       "minimax_h3_fl2va_bf16.safetensors",
                   "config", kH3Config);
  write_component_(root / "vae" / "minimax_h3_video_vae_fp16.safetensors",
                   "minimax_h3_video_vae", "{\"vae_clip_length\": 17}");
  DetectedModel d = detect_model_dir(root.string(), "");
  EXPECT_TRUE(d.model_type == "minimax-h3-fl2va");
  EXPECT_TRUE(d.variant.find("dit") != std::string::npos);
  EXPECT_TRUE(d.variant.find("vae") != std::string::npos);
  EXPECT_TRUE(d.variant.find("text_encoder") == std::string::npos);

  write_component_(root / "text_encoders" /
                       "qwen3vl_32b_minimax_h3_bf16.safetensors",
                   "minimax_h3_te", "{\"num_hidden_layers\": 50}");
  d = detect_model_dir(root.string(), "");
  EXPECT_TRUE(d.variant.find("text_encoder") != std::string::npos);
  fs::remove_all(root, ec);
}

// model-register has to recognize an H3 COMPONENT, not just the
// pipeline. The text encoder is a stock Qwen3-VL, so its own config.json
// says "qwen3_vl" -- its architecture, not its role -- and both the
// released text_encoder/ and any quantized copy of it registered as
// "type unknown", which means no picker offers them. Two independent
// signals fix that, and both are needed: a quantized copy has LEFT the
// pipeline that gave it meaning, and the released one predates any
// stamp this tree could have written.
TEST(minimax_h3_comfy, an_h3_text_encoder_is_recognized_either_way)
{
  const fs::path root = scratch_() / "components";
  std::error_code ec;
  fs::remove_all(root, ec);
  auto write_json = [&](const fs::path& p, const char* body) {
    fs::create_directories(p.parent_path(), ec);
    std::ofstream f(p);
    f << body;
  };

  // 1. The released layout: a text_encoder/ inside an H3 pipeline. The
  //    parent's model_index.json is the only thing that says so.
  write_json(root / "FL2VA" / "model_index.json",
             "{\"_minimax_h3\": {\"partition\": \"fl2va\"}}");
  write_json(root / "FL2VA" / "text_encoder" / "config.json",
             "{\"model_type\": \"qwen3_vl\", \"text_config\": "
             "{\"hidden_size\": 5120}}");
  DetectedModel d =
      detect_model_dir((root / "FL2VA" / "text_encoder").string(), "");
  EXPECT_TRUE(d.model_type == "minimax-h3-text-encoder");
  EXPECT_TRUE(d.detected_by == "pipeline");
  EXPECT_TRUE(d.category == "component");

  // 2. A quantized copy, anywhere on disk. Nothing around it says H3, so
  //    model-quantize stamped the role into the config it wrote.
  write_json(root / "te-w8" / "config.json",
             "{\"model_type\": \"qwen3_vl\", \"_vpipe_component\": "
             "\"minimax-h3-text-encoder\", \"quantization\": "
             "{\"bits\": 8, \"group_size\": 64}}");
  d = detect_model_dir((root / "te-w8").string(), "");
  EXPECT_TRUE(d.model_type == "minimax-h3-text-encoder");
  EXPECT_TRUE(d.detected_by == "component-tag");

  // 3. A bare Qwen3-VL with NEITHER signal must NOT be claimed as H3's
  //    encoder -- it is just a Qwen3-VL, and mislabeling it would offer
  //    it in a slot it cannot fill.
  write_json(root / "plain" / "config.json",
             "{\"model_type\": \"qwen3_vl\", \"text_config\": "
             "{\"hidden_size\": 5120}}");
  d = detect_model_dir((root / "plain").string(), "");
  EXPECT_TRUE(d.model_type != "minimax-h3-text-encoder");

  // 4. The DiT half, quantized from EITHER publisher: both write
  //    `_class_name`, so the existing component probe already covers it
  //    -- including the Comfy-Org-derived one, whose extra
  //    `qkv_per_head` must not change how it is TAGGED.
  write_json(root / "dit-w8" / "config.json",
             "{\"_class_name\": \"MiniMaxH3DiTModel\", \"qkv_per_head\": "
             "false, \"quantization\": {\"bits\": 8, \"group_size\": 64}}");
  d = detect_model_dir((root / "dit-w8").string(), "");
  EXPECT_TRUE(d.model_type == "minimax-h3-dit");
  EXPECT_TRUE(d.family == "MiniMax");
  fs::remove_all(root, ec);
}

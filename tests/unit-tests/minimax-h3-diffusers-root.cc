// A MiniMax-H3 graph pointed at MiniMaxAI's REPOSITORY ROOT, in each of
// the shapes a checkout of it takes.
//
// The repository carries the released task partitions (`FL2VA/`,
// `Ref2VA/`) AND, at its root, a diffusers pipeline over both of them:
// model_index.json (MiniMaxH3ModularPipeline), `transformer/` (FL2VA's
// DiT), `transformer_ref/` (Ref2VA's), and one text encoder, tokenizer,
// processor and pair of VAEs that are byte-identical to each partition's.
// Its README asks for `model_index.json` plus one partition, so three
// checkouts are real:
//
//   readme     model_index.json + FL2VA/          (the README's download)
//   full       everything
//   diffusers  the root pipeline alone
//
// What every stage of a graph resolves on each is held here: the DiT
// (per partition -- the root's `transformer/` is FL2VA's, so taking it as
// THE DiT handed a Ref2VA graph the FL2VA weights), the partition a probe
// reports, the text encoder, both VAEs, what detection registers, and
// what model-quantize takes as its source. A released partition, where
// present, is what answers: the same weights, and the copy the other
// layouts were verified against.
//
// No weights: configs and one-tensor placeholders carry the structure.

#include "minitest.h"

#include "common/flex-data.h"
#include "common/session.h"
#include "generative-models/minimax-h3/metal-minimax-h3-audio-vae.h"
#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/minimax-h3/metal-minimax-h3-video-vae.h"
#include "generative-models/minimax-h3/minimax-h3-diffusers-layout.h"
#include "generative-models/minimax-h3/minimax-h3-text-encoder.h"
#include "stages/diffusion-conditioner-stage.h"
#include "stages/generate-video-stage.h"
#include "stages/model-detect.h"
#include "stages/model-quantize-stage.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;

namespace {

namespace fs = std::filesystem;
using DiT = MetalMiniMaxH3Transformer;

fs::path
scratch_()
{
  return fs::temp_directory_path() / "vpipe-h3-diffusers-root-test";
}

void
text_(const fs::path& p, const std::string& s)
{
  fs::create_directories(p.parent_path());
  std::ofstream(p) << s;
}

// A valid safetensors with one tensor.
void
placeholder_(const fs::path& p)
{
  const std::string json =
      "{\"probe.weight\":{\"dtype\":\"F32\",\"shape\":[1],"
      "\"data_offsets\":[0,4]}}";
  fs::create_directories(p.parent_path());
  std::ofstream f(p, std::ios::binary);
  const std::uint64_t n = json.size();
  f.write(reinterpret_cast<const char*>(&n), 8);
  f.write(json.data(), (std::streamsize)json.size());
  const float zero = 0.0f;
  f.write(reinterpret_cast<const char*>(&zero), 4);
}

// The released DiT config, verbatim but for whitespace.
constexpr const char* kReleasedDit =
    "{\"_class_name\": \"MiniMaxH3DiTModel\", \"hidden_size\": 5376, "
    "\"num_layers\": 50, \"token_refiner_num_layers\": 2, "
    "\"num_attention_heads\": 56, \"attention_head_dim\": 128, "
    "\"ffn_hidden_size\": 14336, \"latents_dim\": 24, "
    "\"audio_latents_dim\": 32, \"patch_size\": [1, 2, 2], "
    "\"text_dim\": 5120, \"timestep_input_dim\": 256, "
    "\"time_embed_hidden_size\": 5376, \"time_embed_dim\": 2688, "
    "\"adaln_out_features\": 96768, \"final_adaln_out_features\": 10752, "
    "\"rope_inv_freq_len\": 16, \"norm_eps\": 1e-05, "
    "\"qk_norm_eps\": 1e-05, \"final_norm_eps\": 1e-05}";

// The root pipeline's, as published (both DiTs carry this one).
constexpr const char* kDiffusersDit =
    "{\"_class_name\": \"MiniMaxH3Transformer3DModel\", "
    "\"num_attention_heads\": 56, \"attention_head_dim\": 128, "
    "\"hidden_size\": 5376, \"num_layers\": 50, \"num_refiner_layers\": 2, "
    "\"ffn_dim\": 14336, \"in_channels\": 24, \"audio_in_channels\": 32, "
    "\"patch_size\": [1, 2, 2], \"text_dim\": 5120, \"freq_dim\": 256, "
    "\"time_embed_hidden_dim\": 5376, \"time_embed_dim\": 2688, "
    "\"rope_freq_dim\": 16, \"rope_theta\": 10000.0, \"norm_eps\": 1e-05, "
    "\"qk_norm_eps\": 1e-05, \"final_norm_eps\": 1e-05}";

void
released_partition_(const fs::path& root, const char* name, const char* part)
{
  const fs::path p = root / name;
  text_(p / "model_index.json",
        std::string("{\"_class_name\": \"MiniMaxH3Pipeline\", "
                    "\"_minimax_h3\": {\"schema_version\": 1, "
                    "\"partition\": \"") + part + "\"}}");
  text_(p / "transformer" / "config.json", kReleasedDit);
  placeholder_(p / "transformer" / "model.safetensors");
  text_(p / "text_encoder" / "config.json", "{\"model_type\": \"qwen3_vl\"}");
  placeholder_(p / "text_encoder" / "model.safetensors");
  text_(p / "video_vae" / "config.json",
        "{\"_class_name\": \"MiniMaxH3VideoVAE\"}");
  placeholder_(p / "video_vae" / "source" / "model.safetensors");
  text_(p / "audio_vae" / "config.json",
        "{\"_class_name\": \"MiniMaxH3AudioVAE\"}");
  placeholder_(p / "audio_vae" / "model.safetensors");
}

void
root_manifest_(const fs::path& root)
{
  text_(root / "model_index.json",
        "{\"_class_name\": \"MiniMaxH3ModularPipeline\", "
        "\"transformer\": [\"diffusers\", \"MiniMaxH3Transformer3DModel\", "
        "{\"subfolder\": \"transformer\"}]}");
}

void
root_pipeline_(const fs::path& root)
{
  root_manifest_(root);
  for (const char* d : {"transformer", "transformer_ref"}) {
    text_(root / d / "config.json", kDiffusersDit);
    placeholder_(root / d / "diffusion_pytorch_model.safetensors");
  }
  text_(root / "text_encoder" / "config.json",
        "{\"model_type\": \"qwen3_vl\"}");
  placeholder_(root / "text_encoder" / "model.safetensors");
  text_(root / "vae" / "config.json",
        std::string("{\"_class_name\": \"") +
            minimax_h3::kDiffusersVideoVaeClass + "\"}");
  placeholder_(root / "vae" / "diffusion_pytorch_model.safetensors");
  text_(root / "audio_vae" / "config.json",
        std::string("{\"_class_name\": \"") +
            minimax_h3::kDiffusersAudioVaeClass + "\"}");
  placeholder_(root / "audio_vae" / "diffusion_pytorch_model.safetensors");
  text_(root / "scheduler" / "scheduler_config.json",
        "{\"_class_name\": \"MiniMaxH3Scheduler\", \"shift\": 12.0}");
}

fs::path
fresh_(const char* name)
{
  const fs::path r = scratch_() / name;
  std::error_code ec;
  fs::remove_all(r, ec);
  fs::create_directories(r);
  return r;
}

// The keys a stage of type `St` pointed at `root` claims for the
// planning phase.
template <class St>
std::vector<std::string>
claims_of_(const std::string& root)
{
  Session sess;
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("hf_dir", FlexData::make_string(root));
  St st(&sess, "s", std::vector<InEdge>{}, std::move(cfg));
  std::vector<std::string> keys;
  for (const ResourceClaim& c : st.declare_resources()) {
    keys.push_back(c.key);
  }
  return keys;
}

std::vector<std::string>
conditioner_claims_(const std::string& root)
{
  return claims_of_<DiffusionConditionerStage>(root);
}

std::vector<std::string>
generator_claims_(const std::string& root)
{
  return claims_of_<GenerateVideoStage>(root);
}

bool
claims_(const std::vector<std::string>& keys, const fs::path& dir)
{
  for (const std::string& k : keys) {
    if (fs::path(k).lexically_normal() == dir.lexically_normal()) {
      return true;
    }
  }
  return false;
}

#define EXPECT_EQ_STR_(got, want)                                          \
  do {                                                                     \
    const std::string g_ = (got), w_ = (want);                             \
    if (g_ != w_) {                                                        \
      std::printf("[h3_root] %s:%d: got '%s', want '%s'\n", __FILE__,      \
                  __LINE__, g_.c_str(), w_.c_str());                       \
    }                                                                      \
    EXPECT_TRUE(g_ == w_);                                                 \
  } while (0)

}  // namespace

// The README's download: the root manifest above the released FL2VA
// partition. Nothing the root names is on disk, and every resolver
// answers with the partition it always did.
TEST(minimax_h3_diffusers_root, the_readme_download_resolves_its_partition)
{
  const fs::path r = fresh_("readme");
  root_manifest_(r);
  released_partition_(r, "FL2VA", "fl2va");
  const std::string root = r.string();
  EXPECT_EQ_STR_(DiT::resolve_dit_dir(root),
                 (r / "FL2VA/transformer").string());
  EXPECT_EQ_STR_(DiT::resolve_dit_dir(root, "fl2va"),
                 (r / "FL2VA/transformer").string());
  // Told the partition it does not hold: nothing, never the other one.
  EXPECT_EQ_STR_(DiT::resolve_dit_dir(root, "ref2va"), root);
  EXPECT_EQ_STR_(DiT::partition_of(root), "fl2va");
  EXPECT_EQ_STR_(MiniMaxH3TextEncoder::resolve_encoder_dir(root),
                 (r / "FL2VA/text_encoder").string());
  EXPECT_EQ_STR_(resolve_vae_dir(root), (r / "FL2VA/video_vae").string());
  EXPECT_EQ_STR_(MetalMiniMaxH3AudioVae::resolve_vae_dir(root),
                 (r / "FL2VA/audio_vae").string());
  std::string fam;
  EXPECT_EQ_STR_(ModelQuantizeStage::resolve_source_dit_dir(root, &fam),
                 (r / "FL2VA/transformer").string());
  EXPECT_EQ_STR_(fam, "minimax-h3");
}

// Everything: the released partitions answer, and the partition picks
// between them -- the root's `transformer/` (FL2VA's, in diffusers names)
// is NOT taken for a Ref2VA graph.
TEST(minimax_h3_diffusers_root, a_full_checkout_takes_the_released_partitions)
{
  const fs::path r = fresh_("full");
  root_pipeline_(r);
  released_partition_(r, "FL2VA", "fl2va");
  released_partition_(r, "Ref2VA", "ref2va");
  const std::string root = r.string();
  EXPECT_EQ_STR_(DiT::resolve_dit_dir(root),
                 (r / "FL2VA/transformer").string());
  EXPECT_EQ_STR_(DiT::resolve_dit_dir(root, "fl2va"),
                 (r / "FL2VA/transformer").string());
  EXPECT_EQ_STR_(DiT::resolve_dit_dir(root, "ref2va"),
                 (r / "Ref2VA/transformer").string());
  EXPECT_EQ_STR_(DiT::partition_of(root), "fl2va");
  EXPECT_EQ_STR_(DiT::partition_of(root, "ref2va"), "ref2va");
  EXPECT_EQ_STR_(resolve_vae_dir(root), (r / "FL2VA/video_vae").string());
  EXPECT_EQ_STR_(MetalMiniMaxH3AudioVae::resolve_vae_dir(root),
                 (r / "FL2VA/audio_vae").string());
  // The root's encoder is byte-identical to each partition's; whichever
  // answers is the same 64 layers.
  const std::string enc = MiniMaxH3TextEncoder::resolve_encoder_dir(root);
  EXPECT_TRUE(enc == (r / "text_encoder").string() ||
              enc == (r / "FL2VA/text_encoder").string());
  std::string fam;
  EXPECT_EQ_STR_(ModelQuantizeStage::resolve_source_dit_dir(root, &fam,
                                                            "ref2va"),
                 (r / "Ref2VA/transformer").string());
  // Two models in one directory: only a record can name one.
  EXPECT_TRUE(detect_model_dir(root, "local/h3-full").model_type.empty());
  // The plan carries the DiT generate-video loads -- the released
  // partition's -- and never the root's copy beside it, or it holds a
  // second 63 GB DiT nothing loads. The stage that loads a DiT is the one
  // that claims it; the conditioner claims its encoder and no DiT at all.
  {
    const auto gen = generator_claims_(root);
    EXPECT_TRUE(claims_(gen, r / "FL2VA/transformer"));
    EXPECT_FALSE(claims_(gen, r / "transformer"));
    EXPECT_FALSE(claims_(gen, r / "transformer_ref"));
    const auto cond = conditioner_claims_(root);
    EXPECT_FALSE(claims_(cond, r / "FL2VA/transformer"));
    EXPECT_FALSE(claims_(cond, r / "transformer"));
  }

  // Holding only FL2VA's partition, a Ref2VA graph takes the root's
  // Ref2VA DiT -- the right weights in the other naming -- not FL2VA's.
  fs::remove_all(r / "Ref2VA");
  EXPECT_EQ_STR_(DiT::resolve_dit_dir(root, "ref2va"),
                 (r / "transformer_ref").string());
  EXPECT_EQ_STR_(DiT::resolve_dit_dir(root, "fl2va"),
                 (r / "FL2VA/transformer").string());
}

// The root pipeline alone: its DiTs by partition, its encoder, its VAEs,
// and every one of them read as what it is.
TEST(minimax_h3_diffusers_root, the_pipeline_alone_answers_by_partition)
{
  const fs::path r = fresh_("diffusers");
  root_pipeline_(r);
  const std::string root = r.string();
  const std::string fl = (r / "transformer").string();
  const std::string rf = (r / "transformer_ref").string();
  EXPECT_TRUE(minimax_h3::is_diffusers_pipeline_root(root));
  EXPECT_EQ_STR_(DiT::resolve_dit_dir(root), fl);
  EXPECT_EQ_STR_(DiT::resolve_dit_dir(root, "fl2va"), fl);
  EXPECT_EQ_STR_(DiT::resolve_dit_dir(root, "ref2va"), rf);
  EXPECT_EQ_STR_(DiT::resolve_dit_dir(rf), rf);
  EXPECT_EQ_STR_(DiT::resolve_dit_dir(rf + "/"), rf + "/");
  // The probe reads the partition off the directory's place.
  EXPECT_EQ_STR_(DiT::partition_of(root), "fl2va");
  EXPECT_EQ_STR_(DiT::partition_of(rf), "ref2va");
  EXPECT_EQ_STR_(DiT::partition_of(rf + "/"), "ref2va");
  EXPECT_EQ_STR_(DiT::partition_of(fl), "fl2va");
  // And the config reads, in diffusers keys, for either.
  for (const char* part : {"fl2va", "ref2va"}) {
    DiT::Config c;
    std::string err;
    const bool ok = DiT::config_from_json(root, c, &err, part);
    if (!ok) { std::printf("[h3_root] %s: %s\n", part, err.c_str()); }
    EXPECT_TRUE(ok);
    EXPECT_TRUE(c.hidden == 5376 && c.n_layers == 50 && c.n_refiner == 2);
    EXPECT_FALSE(c.qkv_per_head);   // split q/k/v assemble FLAT
  }
  EXPECT_EQ_STR_(MiniMaxH3TextEncoder::resolve_encoder_dir(root),
                 (r / "text_encoder").string());
  EXPECT_EQ_STR_(resolve_vae_dir(root), (r / "vae").string());
  EXPECT_EQ_STR_(builtin_vae_family((r / "vae").string()), "minimax-h3");
  EXPECT_EQ_STR_(MetalMiniMaxH3VideoVae::resolve_vae_dir(root),
                 (r / "vae").string());
  EXPECT_EQ_STR_(MetalMiniMaxH3AudioVae::resolve_vae_dir(root),
                 (r / "audio_vae").string());

  // Detection: the root holds two models and is left untagged, as a
  // checkout of both released partitions is; each component registers
  // as what it is.
  EXPECT_TRUE(
      detect_model_dir(root, "local/h3-diffusers").model_type.empty());
  EXPECT_EQ_STR_(detect_model_dir(rf, "local/h3-ref").model_type,
                 "minimax-h3-dit");
  EXPECT_EQ_STR_(
      detect_model_dir((r / "text_encoder").string(), "local/h3-enc")
          .model_type,
      "minimax-h3-text-encoder");

  {
    EXPECT_TRUE(claims_(generator_claims_(root), r / "transformer"));
    const auto cond = conditioner_claims_(root);
    EXPECT_TRUE(claims_(cond, r / "text_encoder"));
    EXPECT_FALSE(claims_(cond, r / "transformer"));
  }

  // model-quantize does not take a DiT whose qkv it would read as one
  // tensor and find in three: no source, rather than a pack missing it.
  std::string fam = "x";
  EXPECT_TRUE(ModelQuantizeStage::resolve_source_dit_dir(root, &fam,
                                                         "fl2va").empty());
  EXPECT_TRUE(fam.empty());
  std::error_code ec;
  fs::remove_all(scratch_(), ec);
}

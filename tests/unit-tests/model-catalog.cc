#include "minitest.h"
#include "stages/model-catalog.h"
#include "stages/qwen-asr-tokenizer.h"
#include "common/flex-data.h"

#include <algorithm>
#include <fstream>
#include <filesystem>
#include <cstdio>
#include <string>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {
bool has_(const vector<string>& v, const string& x) {
  return std::find(v.begin(), v.end(), x) != v.end();
}
string flex_str_(const FlexData& obj, const char* key) {
  auto o = obj.as_object();
  return o.contains(key) ? string(o.at(key).as_string("")) : string();
}
vector<string> flex_arr_(const FlexData& obj, const char* key) {
  vector<string> out;
  auto o = obj.as_object();
  if (!o.contains(key)) { return out; }
  FlexData a = o.at(key);
  if (!a.is_array()) { return out; }
  auto arr = a.as_array();
  for (size_t i = 0; i < arr.size(); ++i) {
    out.push_back(string(arr.at(i).as_string("")));
  }
  return out;
}
}

TEST(model_catalog, families_present) {
  auto f = catalog_families();
  EXPECT_TRUE(has_(f, "Qwen"));
  EXPECT_TRUE(has_(f, "Gemma"));
  EXPECT_TRUE(has_(f, "MOSS"));
}

// family -> version -> param -> variant drill-down filters correctly.
TEST(model_catalog, drilldown_qwen) {
  auto v = catalog_versions("Qwen");
  EXPECT_TRUE(has_(v, "3.5"));
  EXPECT_TRUE(has_(v, "3-ASR"));

  auto p = catalog_param_classes("Qwen", "3.5");
  EXPECT_TRUE(has_(p, "9B"));
  EXPECT_TRUE(has_(p, "4B"));
  EXPECT_TRUE(has_(p, "2B"));
  EXPECT_FALSE(has_(p, "1.7B"));   // that's a 3-ASR class

  auto asr = catalog_param_classes("Qwen", "3-ASR");
  EXPECT_TRUE(has_(asr, "1.7B"));
  EXPECT_TRUE(has_(asr, "0.6B"));

  // 9B: two publishers' MLX 4-bit + the OptiQ 4-bit variant.
  auto vars = catalog_variants("Qwen", "3.5", "9B");
  EXPECT_TRUE(vars.size() == 3);

  // Qwen 3.6 is its own version with a 27B class.
  EXPECT_TRUE(has_(catalog_versions("Qwen"), "3.6"));
  EXPECT_TRUE(has_(catalog_param_classes("Qwen", "3.6"), "27B"));
}

TEST(model_catalog, find_resolves_entry) {
  auto vars = catalog_variants("Qwen", "3.5", "9B");
  EXPECT_TRUE(!vars.empty());
  const ModelCatalogEntry* e =
      catalog_find("Qwen", "3.5", "9B", vars[0]);
  EXPECT_TRUE(e != nullptr);
  EXPECT_TRUE(e->family == "Qwen");
  // Qwen3.5 MLX repos ship tokenizer.json -> no synthesis needed.
  EXPECT_FALSE(e->needs_tokenizer_json);
  EXPECT_TRUE(e->hf_path.find("Qwen3.5-9B-MLX-4bit") != string::npos);
}

// Only Qwen3-ASR is flagged for tokenizer.json synthesis.
TEST(model_catalog, asr_flagged_for_tokenizer_prep) {
  const ModelCatalogEntry* e =
      catalog_by_path("mlx-community/Qwen3-ASR-0.6B-8bit");
  EXPECT_TRUE(e != nullptr);
  EXPECT_TRUE(e->version == "3-ASR");
  EXPECT_TRUE(e->needs_tokenizer_json);
}

TEST(model_catalog, find_unknown_is_null) {
  EXPECT_TRUE(catalog_find("Qwen", "3.5", "9B", "nope") == nullptr);
}

TEST(model_catalog, gemma_needs_no_tokenizer_prep) {
  const ModelCatalogEntry* e =
      catalog_by_path("mlx-community/gemma-4-e4b-it-4bit");
  EXPECT_TRUE(e != nullptr);
  EXPECT_TRUE(e->family == "Gemma");
  EXPECT_FALSE(e->needs_tokenizer_json);
}

TEST(model_catalog, by_path_unknown_is_null) {
  EXPECT_TRUE(catalog_by_path("nobody/nothing") == nullptr);
}

// 8-bit MLX + GGUF variants were added for the 2B/4B classes (4-bit and
// 8-bit from both publishers, plus the unsloth GGUF).
TEST(model_catalog, qwen35_4b_has_8bit_and_gguf_variants) {
  auto vars = catalog_variants("Qwen", "3.5", "4B");
  EXPECT_TRUE(has_(vars, "MLX 4-bit (mlx-community)"));
  EXPECT_TRUE(has_(vars, "MLX 8-bit (lmstudio-community)"));
  EXPECT_TRUE(has_(vars, "MLX 8-bit (mlx-community)"));
  EXPECT_TRUE(has_(vars, "Q4_K_M GGUF +mmproj (unsloth)"));
  EXPECT_TRUE(has_(vars, "MTP Q4_K_M GGUF +mmproj +imatrix (unsloth)"));

  auto v2 = catalog_variants("Qwen", "3.5", "2B");
  EXPECT_TRUE(has_(v2, "MLX 8-bit (lmstudio-community)"));
  EXPECT_TRUE(has_(v2, "Q4_K_M GGUF +mmproj (unsloth)"));
}

// A Qwen3.5 GGUF entry pins the main quant + the BF16 mmproj (multimodal
// projector) out of the multi-quant repo; the MTP entry also pins the
// imatrix. MLX entries leave `files` empty (fetch the whole repo).
TEST(model_catalog, gguf_entry_pins_quant_and_companions) {
  const ModelCatalogEntry* g = catalog_by_path("unsloth/Qwen3.5-4B-GGUF");
  EXPECT_TRUE(g != nullptr);
  EXPECT_TRUE(g->model_type == "qwen3.5");
  EXPECT_TRUE(g->files.size() == 2);
  EXPECT_TRUE(g->files[0] == "Qwen3.5-4B-Q4_K_M.gguf");
  EXPECT_TRUE(g->files[1] == "mmproj-BF16.gguf");
  EXPECT_FALSE(g->needs_tokenizer_json);

  const ModelCatalogEntry* mtp =
      catalog_by_path("unsloth/Qwen3.5-4B-MTP-GGUF");
  EXPECT_TRUE(mtp != nullptr);
  EXPECT_TRUE(mtp->files.size() == 3);
  EXPECT_TRUE(mtp->files[0] == "Qwen3.5-4B-Q4_K_M.gguf");
  EXPECT_TRUE(mtp->files[1] == "mmproj-BF16.gguf");
  EXPECT_TRUE(mtp->files[2] == "imatrix_unsloth.gguf_file");

  const ModelCatalogEntry* b2 = catalog_by_path("unsloth/Qwen3.5-2B-GGUF");
  EXPECT_TRUE(b2 != nullptr);
  EXPECT_TRUE(b2->files.size() == 2);
  EXPECT_TRUE(b2->files[1] == "mmproj-BF16.gguf");

  // MLX 8-bit entry: no pinned files (whole-repo fetch).
  const ModelCatalogEntry* mlx =
      catalog_by_path("mlx-community/Qwen3.5-4B-MLX-8bit");
  EXPECT_TRUE(mlx != nullptr);
  EXPECT_TRUE(mlx->files.empty());
}

// The supplementary CoreML models (vpipe-supplement) share ONE hf_path, so
// each pins its .tar, sets extract_archive, and carries a distinct `name`
// (the registration key) plus a compatibility model_type. catalog_by_path
// can't disambiguate the shared repo, so resolve via catalog_find.
TEST(model_catalog, supplement_archive_entries) {
  struct Want {
    const char* family;
    const char* version;
    const char* param_class;
    const char* variant;
    const char* model_type;
    const char* name;
    const char* tar;
  };
  const Want wants[] = {
    {"Qwen", "3.5", "4B", "Vision tower CoreML 512x320 w8 (vpipe-supplement)",
     "qwen3.5-vision-encoder", "qwen3_5_mlx_4b_vision_vid_512x320",
     "qwen3_5_mlx_4b_vision_vid_512x320_w8.tar"},
    {"Qwen", "3.5", "4B", "Vision tower CoreML 768x480 w8 (vpipe-supplement)",
     "qwen3.5-vision-encoder", "qwen3_5_mlx_4b_vision_vid_768x480",
     "qwen3_5_mlx_4b_vision_vid_768x480_w8.tar"},
    {"Gemma", "4", "E4B", "Vision tower CoreML 768x480 w8 (vpipe-supplement)",
     "gemma4-vision-encoder", "gemma4_mlx_e4b_vision_768x480",
     "gemma4_mlx_e4b_vision_768x480_w8.tar"},
    {"YOLOX", "L", "1024x640", "CoreML w8 (vpipe-supplement)",
     "yolo", "yolox_l_1024x640", "yolox_l_1024x640_w8.tar"},
    {"Silero", "VAD v6", "unified", "CoreML (vpipe-supplement)",
     "silero-vad", "silero_vad_unified_v6", "silero-vad-unified-v6.tar"},
    {"BEATs", "iter3+", "AS2M",
     "Audio tagging CoreML 10s (vpipe-supplement)",
     "audio-tagging", "beats_as2m_10s", "beats_as2m_10s.tar"},
  };
  for (const Want& w : wants) {
    const ModelCatalogEntry* e =
        catalog_find(w.family, w.version, w.param_class, w.variant);
    EXPECT_TRUE(e != nullptr);
    EXPECT_TRUE(e->hf_path == "tgo-app-dev/vpipe-supplement");
    EXPECT_TRUE(e->model_type == w.model_type);
    EXPECT_TRUE(e->name == w.name);
    EXPECT_TRUE(e->extract_archive);
    EXPECT_TRUE(e->files.size() == 1u);
    EXPECT_TRUE(e->files[0] == w.tar);
    EXPECT_FALSE(e->needs_tokenizer_json);
  }
}

// The MOSS text-to-speech pair: the LM (hf_dir, model_type "moss-tts")
// and the audio codec (codec_dir, model_type "moss-codec"). Both are
// whole-repo fetches (no pinned files, no tokenizer.json synthesis) and
// live under their own "MOSS" family with distinct versions so the
// drill-down keeps them apart.
TEST(model_catalog, moss_tts_pair_present) {
  EXPECT_TRUE(has_(catalog_versions("MOSS"), "TTS"));
  EXPECT_TRUE(has_(catalog_versions("MOSS"), "Audio-Tokenizer"));

  const ModelCatalogEntry* lm =
      catalog_by_path("mlx-community/MOSS-TTS-8B-8bit");
  EXPECT_TRUE(lm != nullptr);
  EXPECT_TRUE(lm->family == "MOSS");
  EXPECT_TRUE(lm->model_type == "moss-tts");
  EXPECT_TRUE(lm->files.empty());
  EXPECT_FALSE(lm->needs_tokenizer_json);

  const ModelCatalogEntry* codec =
      catalog_by_path("OpenMOSS-Team/MOSS-Audio-Tokenizer");
  EXPECT_TRUE(codec != nullptr);
  EXPECT_TRUE(codec->family == "MOSS");
  EXPECT_TRUE(codec->model_type == "moss-codec");
  EXPECT_TRUE(codec->files.empty());
  EXPECT_FALSE(codec->needs_tokenizer_json);
}

// OptiQ 4-bit MLX variants for 2B/4B/9B; whole-repo fetch (no pin).
TEST(model_catalog, optiq_variants_present) {
  EXPECT_TRUE(has_(catalog_variants("Qwen", "3.5", "9B"),
                   "MLX OptiQ 4-bit (mlx-community)"));
  for (const char* p : {"mlx-community/Qwen3.5-2B-OptiQ-4bit",
                        "mlx-community/Qwen3.5-4B-OptiQ-4bit",
                        "mlx-community/Qwen3.5-9B-OptiQ-4bit"}) {
    const ModelCatalogEntry* e = catalog_by_path(p);
    EXPECT_TRUE(e != nullptr);
    EXPECT_TRUE(e->files.empty());
    EXPECT_FALSE(e->needs_tokenizer_json);
  }
}

// The Gemma-4 OptiQ 4-bit MLX variants: dense 12B/31B + the 26B-A4B MoE.
// All three are "gemma4_unified" (like their bf16 sources -- the E-series
// PLE models are the ones tagged "gemma4") and whole-repo fetches.
TEST(model_catalog, gemma4_optiq_variants_present) {
  // {OptiQ repo, param class, the gated bf16 source it sits beside}.
  struct Row { const char* path; const char* param; const char* src; };
  const Row rows[] = {
      {"mlx-community/gemma-4-12B-it-OptiQ-4bit", "12B",
       "bf16 (google, gated)"},
      {"mlx-community/gemma-4-31B-it-OptiQ-4bit", "31B",
       "bf16 (google, gated)"},
      {"mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit", "26B-A4B",
       "bf16 MoE (google, gated)"}};
  for (const Row& r : rows) {
    const ModelCatalogEntry* e = catalog_by_path(r.path);
    EXPECT_TRUE(e != nullptr);
    if (e == nullptr) { continue; }
    EXPECT_TRUE(e->family == "Gemma");
    EXPECT_TRUE(e->version == "4");
    EXPECT_TRUE(e->param_class == r.param);
    EXPECT_TRUE(e->variant == "MLX OptiQ 4-bit (mlx-community)");
    EXPECT_TRUE(e->model_type == "gemma4_unified");
    EXPECT_TRUE(e->files.empty());
    EXPECT_FALSE(e->needs_tokenizer_json);
    EXPECT_TRUE(catalog_category(*e) == "model");
    // Reachable through the drill-down, beside the gated bf16 source.
    auto vars = catalog_variants("Gemma", "4", r.param);
    EXPECT_TRUE(has_(vars, "MLX OptiQ 4-bit (mlx-community)"));
    EXPECT_TRUE(has_(vars, r.src));
  }
}

// The Qwen 3.6 OptiQ 4-bit MLX variants (27B dense-hybrid + 35B-A3B MoE).
// Both are qwen3_5-family configs -> model_type "qwen3.5" (like their bf16
// sources), and both are whole-repo fetches -- `files` must stay empty so
// the recursive tree walk also picks up the optiq/ companions (the MTP head
// and the vision tower).
TEST(model_catalog, qwen36_optiq_variants_present) {
  struct Row { const char* path; const char* param; };
  const Row rows[] = {
      {"mlx-community/Qwen3.6-27B-OptiQ-4bit", "27B"},
      {"mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit", "35B-A3B"}};
  for (const Row& r : rows) {
    const ModelCatalogEntry* e = catalog_by_path(r.path);
    EXPECT_TRUE(e != nullptr);
    if (e == nullptr) { continue; }
    EXPECT_TRUE(e->family == "Qwen");
    EXPECT_TRUE(e->version == "3.6");
    EXPECT_TRUE(e->param_class == r.param);
    EXPECT_TRUE(e->variant == "MLX OptiQ 4-bit (mlx-community)");
    EXPECT_TRUE(e->model_type == "qwen3.5");
    EXPECT_TRUE(e->files.empty());
    EXPECT_FALSE(e->needs_tokenizer_json);
    EXPECT_TRUE(catalog_category(*e) == "model");
    // Reachable through the drill-down, next to the bf16 source.
    auto vars = catalog_variants("Qwen", "3.6", r.param);
    EXPECT_TRUE(has_(vars, "MLX OptiQ 4-bit (mlx-community)"));
    EXPECT_TRUE(has_(vars, "bf16 (Qwen)"));
  }
  EXPECT_TRUE(has_(catalog_param_classes("Qwen", "3.6"), "35B-A3B"));
}

// Qwen 3.8 is its own DISPLAY version over the qwen3.5 runtime family --
// its config.json is Qwen3.6-27B's field for field bar the transformers
// version, so it runs the same path and must be labelled apart from it.
TEST(model_catalog, qwen38_27b_family) {
  EXPECT_TRUE(has_(catalog_versions("Qwen"), "3.8"));
  auto p = catalog_param_classes("Qwen", "3.8");
  EXPECT_TRUE(p.size() == 1);
  EXPECT_TRUE(has_(p, "27B"));
  EXPECT_TRUE(catalog_variants("Qwen", "3.8", "27B").size() == 4);

  struct Row { const char* path; const char* variant; };
  const Row rows[] = {
      {"Qwen/Qwen3.8-27B", "bf16 (Qwen)"},
      {"lmstudio-community/Qwen3.8-27B-MLX-4bit",
       "MLX 4-bit (lmstudio-community)"},
      {"unsloth/Qwen3.8-27B-GGUF", "Q4_K_M GGUF +mmproj (unsloth)"},
  };
  for (const Row& r : rows) {
    const ModelCatalogEntry* e = catalog_by_path(r.path);
    EXPECT_TRUE(e != nullptr);
    if (e == nullptr) { continue; }
    EXPECT_TRUE(e->family == "Qwen");
    EXPECT_TRUE(e->version == "3.8");
    EXPECT_TRUE(e->param_class == "27B");
    EXPECT_TRUE(e->variant == r.variant);
    // The RUNTIME tag is the 3.5 family's, deliberately: every consumer
    // of the tag treats 3.5 and 3.6 alike, and a third spelling would be
    // one more thing each of them has to learn for no behaviour change.
    EXPECT_TRUE(e->model_type == "qwen3.5");
    EXPECT_FALSE(e->needs_tokenizer_json);
  }

  // Only the GGUF entry pins files; the two safetensors repos fetch whole.
  EXPECT_TRUE(catalog_by_path("Qwen/Qwen3.8-27B")->files.empty());
  EXPECT_TRUE(
      catalog_by_path("lmstudio-community/Qwen3.8-27B-MLX-4bit")
          ->files.empty());
  // TWO GGUF rows share the repo -- one per quantization point -- so the
  // by-path lookup answers with the first (documented) and the caller
  // that must not take the wrong one asks for all of them.
  auto gs = catalog_all_by_path("unsloth/Qwen3.8-27B-GGUF");
  EXPECT_TRUE(gs.size() == 2);
  const ModelCatalogEntry* g = catalog_by_path("unsloth/Qwen3.8-27B-GGUF");
  EXPECT_TRUE(g == gs[0]);
  const char* quants[] = {"Qwen3.8-27B-Q4_K_M.gguf", "Qwen3.8-27B-Q8_0.gguf"};
  for (std::size_t i = 0; i < gs.size() && i < 2; ++i) {
    EXPECT_TRUE(gs[i]->files.size() == 2);
    EXPECT_TRUE(gs[i]->files[0] == quants[i]);
    // The F16 projector, NOT the BF16 one: they differ in how
    // v.patch_embd.weight is stored (F16 vs F32) and the tower reads both.
    EXPECT_TRUE(has_(gs[i]->files, "mmproj-F16.gguf"));
    EXPECT_FALSE(has_(gs[i]->files, "mmproj-BF16.gguf"));
  }
}

// It is a VLM, so the derived modalities take images and video in and
// text out -- the same defaulting the 3.5/3.6 entries rely on.
TEST(model_catalog, qwen38_io_modalities) {
  std::vector<std::string> in, out;
  catalog_default_io("qwen3.5", in, out);
  EXPECT_TRUE(has_(in, "text"));
  EXPECT_TRUE(has_(in, "image"));
  EXPECT_TRUE(has_(in, "video"));
  EXPECT_TRUE(out.size() == 1 && out[0] == "text");
}

// BOTH MiniMax-H3 partitions derive their modalities, not just FL2VA.
//
// The two ship from one repo and differ in what they CONSUME: FL2VA
// takes a prompt and up to two keyframes, Ref2VA also takes video and
// audio references. Only FL2VA was in the table, so a Ref2VA registered
// from disk -- which is every locally quantized one -- derived nothing,
// and a picker filtering on need_inputs hid it from the stages that
// would use it. A fetched one was unaffected, because its catalogue
// entry records the I/O explicitly, which is why the gap survived.
TEST(model_catalog, both_h3_partitions_derive_their_modalities) {
  std::vector<std::string> in, out;
  catalog_default_io("minimax-h3-fl2va", in, out);
  EXPECT_TRUE(has_(in, "text"));
  EXPECT_TRUE(has_(in, "image"));
  EXPECT_FALSE(has_(in, "video"));
  EXPECT_TRUE(has_(out, "video") && has_(out, "audio"));

  std::vector<std::string> rin, rout;
  catalog_default_io("minimax-h3-ref2va", rin, rout);
  EXPECT_TRUE(has_(rin, "text"));
  EXPECT_TRUE(has_(rin, "image"));
  EXPECT_TRUE(has_(rin, "video"));
  EXPECT_TRUE(has_(rin, "audio"));
  EXPECT_TRUE(has_(rout, "video") && has_(rout, "audio"));

  // And the derived list matches what the CATALOGUE entry for the same
  // partition records, so a registered model and a fetched one describe
  // themselves identically.
  const ModelCatalogEntry* e = nullptr;
  for (const ModelCatalogEntry& c : model_catalog()) {
    if (c.model_type == "minimax-h3-ref2va" && !c.inputs.empty()) {
      e = &c;
      break;
    }
  }
  EXPECT_TRUE(e != nullptr);
  if (e != nullptr) { EXPECT_TRUE(e->inputs == rin && e->outputs == rout); }
}

// No catalogue entry repeats a modality.
//
// catalog_default_io APPENDS, and a detection path that already filled
// its I/O and then ran it again produced ["text","image","text","image"]
// on a quantized MiniMax-H3. The compatibility filters use includes()
// and survived it; the picker drew one badge per entry and did not.
TEST(model_catalog, no_entry_repeats_a_modality) {
  for (const ModelCatalogEntry& c : model_catalog()) {
    for (const auto* list : {&c.inputs, &c.outputs}) {
      for (std::size_t i = 0; i < list->size(); ++i) {
        for (std::size_t j = i + 1; j < list->size(); ++j) {
          EXPECT_FALSE((*list)[i] == (*list)[j]);
        }
      }
    }
  }
}

// The Qwen3.6 27B GGUF entry pins the main quant + mmproj + imatrix.
TEST(model_catalog, qwen36_27b_pins_multimodal_files) {
  const ModelCatalogEntry* e =
      catalog_by_path("unsloth/Qwen3.6-27B-MTP-GGUF");
  EXPECT_TRUE(e != nullptr);
  EXPECT_TRUE(e->version == "3.6");
  EXPECT_TRUE(e->param_class == "27B");
  EXPECT_TRUE(e->model_type == "qwen3.6");
  EXPECT_TRUE(e->files.size() == 3);
  EXPECT_TRUE(has_(e->files, "Qwen3.6-27B-Q4_K_M.gguf"));
  EXPECT_TRUE(has_(e->files, "mmproj-BF16.gguf"));
  EXPECT_TRUE(has_(e->files, "imatrix_unsloth.gguf_file"));
}

// The Mage-Flow family: six 4B entries in two instantiations -- Gen
// (t2i, "mage-flow", text-in) and Edit ("mage-flow-edit", text+image-in),
// each in Base / RL-aligned / Turbo. All six pin the SAME 18-file
// non-asset layout.
TEST(model_catalog, boogu_image_family_present) {
  struct Row { const char* path; const char* ver; const char* mt; };
  const Row rows[] = {
      {"Boogu/Boogu-Image-0.1-Base", "0.1", "boogu-image"},
      {"Boogu/Boogu-Image-0.1-Turbo", "0.1", "boogu-image"},
      {"Boogu/Boogu-Image-0.1-Edit", "0.1-Edit", "boogu-image-edit"},
      {"Boogu/Boogu-Image-0.1-Edit-Turbo", "0.1-Edit", "boogu-image-edit"}};
  for (const Row& r : rows) {
    const ModelCatalogEntry* e = catalog_by_path(r.path);
    EXPECT_TRUE(e != nullptr);
    if (e == nullptr) { continue; }
    EXPECT_TRUE(e->family == "Boogu-Image");
    EXPECT_TRUE(e->version == r.ver);
    EXPECT_TRUE(e->param_class == "10B");
    EXPECT_TRUE(e->model_type == r.mt);
    EXPECT_FALSE(e->needs_tokenizer_json);
    // Pinned layout: the sharded DiT + the mllm text encoder + the VAE +
    // processor/scheduler configs, and NO assets/ or trust_remote_code shims.
    EXPECT_TRUE(has_(e->files, "model_index.json"));
    EXPECT_TRUE(has_(
        e->files, "transformer/diffusion_pytorch_model-00003-of-00003.safetensors"));
    EXPECT_TRUE(has_(e->files, "mllm/model-00004-of-00004.safetensors"));
    EXPECT_TRUE(has_(e->files, "vae/diffusion_pytorch_model.safetensors"));
    EXPECT_TRUE(has_(e->files, "processor/chat_template.jinja"));
    EXPECT_TRUE(has_(e->files, "scheduler/scheduler_config.json"));
    EXPECT_FALSE(has_(e->files, "transformer/transformer_boogu.py"));
  }
  // Both variants of each task drill down under one family.
  auto gen = catalog_variants("Boogu-Image", "0.1", "10B");
  EXPECT_TRUE(gen.size() == 2);
  EXPECT_TRUE(has_(gen, "Turbo 4-step distilled bf16 (Boogu)"));
  auto edit = catalog_variants("Boogu-Image", "0.1-Edit", "10B");
  EXPECT_TRUE(edit.size() == 2);
  EXPECT_TRUE(has_(edit, "Edit-Turbo 4-step distilled bf16 (Boogu)"));

  // Modalities: Edit is image-conditioned, the t2i pair is not.
  FlexData ef = catalog_entry_to_flex(
      *catalog_by_path("Boogu/Boogu-Image-0.1-Edit-Turbo"));
  auto ein = flex_arr_(ef, "inputs");
  EXPECT_TRUE(has_(ein, "text"));
  EXPECT_TRUE(has_(ein, "image"));
  EXPECT_TRUE(has_(flex_arr_(ef, "outputs"), "image"));
  FlexData tf = catalog_entry_to_flex(
      *catalog_by_path("Boogu/Boogu-Image-0.1-Turbo"));
  auto tin = flex_arr_(tf, "inputs");
  EXPECT_TRUE(has_(tin, "text"));
  EXPECT_FALSE(has_(tin, "image"));
}

TEST(model_catalog, mage_flow_family_present) {
  struct Row { const char* path; const char* ver; const char* mt; };
  const Row rows[] = {
      {"microsoft/Mage-Flow-Base", "Gen", "mage-flow"},
      {"microsoft/Mage-Flow", "Gen", "mage-flow"},
      {"microsoft/Mage-Flow-Turbo", "Gen", "mage-flow"},
      {"microsoft/Mage-Flow-Edit-Base", "Edit", "mage-flow-edit"},
      {"microsoft/Mage-Flow-Edit", "Edit", "mage-flow-edit"},
      {"microsoft/Mage-Flow-Edit-Turbo", "Edit", "mage-flow-edit"}};
  for (const Row& r : rows) {
    const ModelCatalogEntry* e = catalog_by_path(r.path);
    EXPECT_TRUE(e != nullptr);
    if (e == nullptr) { continue; }
    EXPECT_TRUE(e->family == "Mage-Flow");
    EXPECT_TRUE(e->version == r.ver);
    EXPECT_TRUE(e->param_class == "4B");
    EXPECT_TRUE(e->model_type == r.mt);
    EXPECT_FALSE(e->needs_tokenizer_json);
    // Pinned layout: the split-stage sub-models, no assets/.
    EXPECT_TRUE(e->files.size() == 18);
    EXPECT_TRUE(has_(e->files, "model_index.json"));
    EXPECT_TRUE(has_(e->files,
                     "transformer/diffusion_pytorch_model.safetensors"));
    EXPECT_TRUE(has_(e->files, "vae/diffusion_pytorch_model.safetensors"));
    EXPECT_TRUE(has_(e->files,
                     "text_encoder/model-00002-of-00002.safetensors"));
    EXPECT_TRUE(has_(e->files, "scheduler/scheduler_config.json"));
  }
  // Both variants of each instantiation drill down under one family.
  auto gen = catalog_variants("Mage-Flow", "Gen", "4B");
  EXPECT_TRUE(gen.size() == 3);
  EXPECT_TRUE(has_(gen, "Turbo 4-step distilled bf16 (microsoft)"));
  auto edit = catalog_variants("Mage-Flow", "Edit", "4B");
  EXPECT_TRUE(edit.size() == 3);
  EXPECT_TRUE(has_(edit, "Turbo 4-step distilled bf16 (microsoft)"));

  // Modalities: Edit is image-conditioned, Gen is not; both emit images.
  FlexData ef = catalog_entry_to_flex(
      *catalog_by_path("microsoft/Mage-Flow-Edit-Turbo"));
  auto ein = flex_arr_(ef, "inputs");
  EXPECT_TRUE(has_(ein, "text"));
  EXPECT_TRUE(has_(ein, "image"));
  EXPECT_TRUE(has_(flex_arr_(ef, "outputs"), "image"));
  FlexData tf = catalog_entry_to_flex(
      *catalog_by_path("microsoft/Mage-Flow-Turbo"));
  auto tin = flex_arr_(tf, "inputs");
  EXPECT_TRUE(tin.size() == 1);
  EXPECT_TRUE(has_(tin, "text"));
  EXPECT_TRUE(has_(flex_arr_(tf, "outputs"), "image"));
}

// Input/output modalities + derived category are exposed via
// catalog_entry_to_flex (gemma-4 e4b multimodal-in/text-out; flux2
// text+image-in/image-out).
TEST(model_catalog, io_types_and_category) {
  const ModelCatalogEntry* g =
      catalog_by_path("mlx-community/gemma-4-e4b-it-4bit");
  EXPECT_TRUE(g != nullptr);
  EXPECT_TRUE(catalog_category(*g) == "model");
  FlexData gf = catalog_entry_to_flex(*g);
  EXPECT_TRUE(flex_str_(gf, "category") == "model");
  auto gin = flex_arr_(gf, "inputs");
  EXPECT_TRUE(has_(gin, "text"));
  EXPECT_TRUE(has_(gin, "image"));
  EXPECT_TRUE(has_(gin, "audio"));
  EXPECT_TRUE(has_(gin, "video"));
  auto gout = flex_arr_(gf, "outputs");
  EXPECT_TRUE(gout.size() == 1);
  EXPECT_TRUE(has_(gout, "text"));

  // The unified 12B is the same multimodal Gemma-4 family (not text-only):
  // image/audio/video/text in, text out -- like the e4b.
  const ModelCatalogEntry* g12 =
      catalog_by_path("google/gemma-4-12B-it-qat-q4_0-gguf");
  EXPECT_TRUE(g12 != nullptr);
  FlexData g12f = catalog_entry_to_flex(*g12);
  auto g12in = flex_arr_(g12f, "inputs");
  EXPECT_TRUE(has_(g12in, "text"));
  EXPECT_TRUE(has_(g12in, "image"));
  EXPECT_TRUE(has_(g12in, "audio"));
  EXPECT_TRUE(has_(g12in, "video"));
  auto g12out = flex_arr_(g12f, "outputs");
  EXPECT_TRUE(g12out.size() == 1);
  EXPECT_TRUE(has_(g12out, "text"));

  const ModelCatalogEntry* fx =
      catalog_by_path("black-forest-labs/FLUX.2-klein-4B");
  EXPECT_TRUE(fx != nullptr);
  FlexData ff = catalog_entry_to_flex(*fx);
  auto fin = flex_arr_(ff, "inputs");
  EXPECT_TRUE(has_(fin, "text"));
  EXPECT_TRUE(has_(fin, "image"));
  auto fout = flex_arr_(ff, "outputs");
  EXPECT_TRUE(has_(fout, "image"));
  EXPECT_FALSE(has_(fout, "text"));
}

// Datasets are their own category.
TEST(model_catalog, dataset_category) {
  const ModelCatalogEntry* d =
      catalog_by_path("vpipe-eval-datasets/wikitext-2-raw-test");
  EXPECT_TRUE(d != nullptr);
  EXPECT_TRUE(catalog_category(*d) == "dataset");
  EXPECT_TRUE(flex_str_(catalog_entry_to_flex(*d), "category") == "dataset");
}

// Supplements (vision towers, LoRA) carry an explicit parent linkage, and
// catalog_by_name disambiguates the ones sharing the vpipe-supplement repo.
TEST(model_catalog, supplement_parent_linkage) {
  const ModelCatalogEntry* qt =
      catalog_by_name("qwen3_5_mlx_4b_vision_vid_512x320");
  EXPECT_TRUE(qt != nullptr);
  EXPECT_TRUE(qt->model_type == "qwen3.5-vision-encoder");
  EXPECT_TRUE(catalog_category(*qt) == "supplement");
  FlexData qf = catalog_entry_to_flex(*qt);
  EXPECT_TRUE(flex_str_(qf, "category") == "supplement");
  EXPECT_TRUE(flex_str_(qf, "parent_model_type") == "qwen3.5");
  EXPECT_TRUE(flex_str_(qf, "parent_param_class") == "4B");

  const ModelCatalogEntry* gt =
      catalog_by_name("gemma4_mlx_e4b_vision_768x480");
  EXPECT_TRUE(gt != nullptr);
  EXPECT_TRUE(gt->model_type == "gemma4-vision-encoder");
  EXPECT_TRUE(gt->parent_model_type == "gemma4");
  EXPECT_TRUE(gt->parent_param_class == "E4B");

  // The Krea LoRA attaches to any krea2 DiT (no size pin).
  const ModelCatalogEntry* lora =
      catalog_by_path("krea/Krea-2-LoRA-softwatercolor");
  EXPECT_TRUE(lora != nullptr);
  EXPECT_TRUE(catalog_category(*lora) == "supplement");
  EXPECT_TRUE(lora->parent_model_type == "krea2");
  EXPECT_TRUE(lora->parent_param_class.empty());

  // The community Krea-2 LoRAs (fusible via the ai-toolkit name remap + LoKr
  // support in lora-fuse) are catalogued as krea2 supplements too.
  for (const char* p : {"mgwr/M87", "RudySen/Krea2-realism-V2"}) {
    const ModelCatalogEntry* e = catalog_by_path(p);
    EXPECT_TRUE(e != nullptr);
    EXPECT_TRUE(catalog_category(*e) == "supplement");
    EXPECT_TRUE(e->parent_model_type == "krea2");
  }

  // A plain model has no parent linkage.
  const ModelCatalogEntry* lm =
      catalog_by_path("lmstudio-community/Qwen3.5-4B-MLX-4bit");
  EXPECT_TRUE(lm != nullptr);
  EXPECT_TRUE(lm->parent_model_type.empty());

  EXPECT_TRUE(catalog_by_name("nope") == nullptr);
  EXPECT_TRUE(catalog_by_name("") == nullptr);
}

TEST(model_catalog, normalize_paths) {
  EXPECT_TRUE(normalize_hf_path(
      "https://huggingface.co/mlx-community/Qwen3.5-9B-MLX-4bit")
      == "mlx-community/Qwen3.5-9B-MLX-4bit");
  EXPECT_TRUE(normalize_hf_path(
      "huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf/tree/main")
      == "google/gemma-4-12B-it-qat-q4_0-gguf");
  EXPECT_TRUE(normalize_hf_path("  owner/repo/  ") == "owner/repo");
  EXPECT_TRUE(normalize_hf_path(
      "https://huggingface.co/owner/repo?library=mlx") == "owner/repo");
  EXPECT_TRUE(normalize_hf_path("owner/repo/resolve/main/x.bin")
      == "owner/repo");
  EXPECT_TRUE(normalize_hf_path(
      "https://www.huggingface.co/a/b") == "a/b");
  EXPECT_TRUE(normalize_hf_path("notapath").empty());
  EXPECT_TRUE(normalize_hf_path("").empty());
}

TEST(model_catalog, hf_tree_files_parses_and_filters) {
  const char* json = R"([
    {"type":"file","path":"config.json","size":1234},
    {"type":"directory","path":"subdir"},
    {"type":"file","path":"model.safetensors","size":999},
    {"type":"file","path":""},
    {"oops":"no type"}
  ])";
  FlexData tree = FlexData::from_json(json);
  auto files = hf_tree_files(tree);
  EXPECT_TRUE(files.size() == 2);
  EXPECT_TRUE(files[0].path == "config.json");
  EXPECT_TRUE(files[0].size == 1234u);
  EXPECT_TRUE(files[1].path == "model.safetensors");
  EXPECT_TRUE(files[1].size == 999u);
}

// The tree API is the only place the checksums come from, and they are
// what a resumed download is checked against. A parser that quietly
// dropped them would leave every check reporting "nothing published"
// and passing -- which reads exactly like a check that ran.
TEST(model_catalog, hf_tree_files_carries_the_published_checksums) {
  const char* json = R"([
    {"type":"file","path":"config.json","size":3366,
     "oid":"cb3e1a878a5c4a0505e0c9967da45820b2ca6bd3"},
    {"type":"file","path":"model.safetensors","size":135,
     "oid":"8dca545f7a23dbc24af451db83a558c31a97ef09",
     "lfs":{"oid":"52dc943eaf6093a2313271a7a0abc36d127b2a166","size":3034301037,
            "pointerSize":135}}
  ])";
  FlexData tree = FlexData::from_json(json);
  auto files = hf_tree_files(tree);
  ASSERT_TRUE(files.size() == 2);
  // A small file git stores itself: a blob id and no LFS object.
  EXPECT_TRUE(files[0].git_oid == "cb3e1a878a5c4a0505e0c9967da45820b2ca6bd3");
  EXPECT_TRUE(files[0].sha256.empty());
  EXPECT_TRUE(files[0].size == 3366u);
  // A shard: the LFS object id is the content SHA-256, and the LFS size
  // is the real one -- the outer `size` is the pointer file's.
  EXPECT_TRUE(files[1].sha256 == "52dc943eaf6093a2313271a7a0abc36d127b2a166");
  EXPECT_TRUE(files[1].git_oid == "8dca545f7a23dbc24af451db83a558c31a97ef09");
  EXPECT_TRUE(files[1].size == 3034301037ull);
}

TEST(model_catalog, hf_tree_files_non_array_is_empty) {
  FlexData obj = FlexData::make_object();
  EXPECT_TRUE(hf_tree_files(obj).empty());
}

// ---- Qwen3-ASR tokenizer.json synthesis (no transformers) ----------

// Synthesize tokenizer.json from vocab.json + merges.txt +
// tokenizer_config.json and verify it carries everything vpipe's
// Tokenizer parser consumes: BPE model, the vocab, the merges, the
// special tokens, a \p{L}/\p{N} pre-tokenizer, and no metaspace
// normalizer (so encoding stays byte-level).
TEST(model_catalog, build_qwen_asr_tokenizer_json_structure) {
  const char* vocab = R"({"!":0,"Ġthe":2,"a":3})";
  const char* merges = "#version: 0.2\nĠ t\nĠt h\n\n";
  const char* cfg = R"({
    "added_tokens_decoder": {
      "151643": {"content":"<|endoftext|>","special":true},
      "151704": {"content":"<asr_text>","special":true}
    }
  })";
  std::string err;
  std::string js = build_qwen_asr_tokenizer_json(vocab, merges, cfg, err);
  EXPECT_TRUE(err.empty());
  EXPECT_TRUE(!js.empty());

  FlexData root = FlexData::from_json(js);
  EXPECT_TRUE(root.is_object());
  auto ro = root.as_object();

  // model.type == BPE, vocab + merges carried through.
  EXPECT_TRUE(ro.contains("model"));
  FlexData model = ro.at("model");
  auto mo = model.as_object();
  EXPECT_TRUE(string(mo.at("type").as_string("")) == "BPE");
  FlexData v = mo.at("vocab");
  EXPECT_TRUE(v.is_object());
  EXPECT_TRUE(v.as_object().at("Ġthe").as_int(-1) == 2);
  FlexData m = mo.at("merges");
  EXPECT_TRUE(m.is_array());
  // 2 real merges (the "#version" header is dropped).
  EXPECT_TRUE(m.as_array().size() == 2);
  EXPECT_TRUE(string(m.as_array().at(0).as_string("")) == "Ġ t");

  // added_tokens: both specials present, keyed by id + content.
  FlexData added = ro.at("added_tokens");
  EXPECT_TRUE(added.is_array());
  EXPECT_TRUE(added.as_array().size() == 2);

  // No metaspace normalizer -> byte-level.
  EXPECT_TRUE(ro.contains("normalizer"));
  EXPECT_TRUE(ro.at("normalizer").is_null());

  // pre_tokenizer regex carries \p{L} and \p{N} (vpipe's Qwen scanner
  // trigger). Just check the substrings survive into the JSON.
  EXPECT_TRUE(js.find("\\\\p{L}") != string::npos
              || js.find("p{L}") != string::npos);
  EXPECT_TRUE(js.find("p{N}") != string::npos);
}

// An invalid vocab.json is reported, not crashed on.
TEST(model_catalog, build_qwen_asr_tokenizer_json_bad_vocab) {
  std::string err;
  std::string js =
      build_qwen_asr_tokenizer_json("not json", "a b\n", "", err);
  EXPECT_TRUE(js.empty());
  EXPECT_FALSE(err.empty());
}

// Empty tokenizer_config -> still valid, just no special tokens.
TEST(model_catalog, build_qwen_asr_tokenizer_json_no_config) {
  std::string err;
  std::string js =
      build_qwen_asr_tokenizer_json(R"({"a":0})", "a b\n", "", err);
  EXPECT_TRUE(err.empty());
  FlexData root = FlexData::from_json(js);
  EXPECT_TRUE(root.as_object().at("added_tokens").as_array().size() == 0);
}

// The Comfy-Org repack of MiniMax-H3: a SECOND entry for the same model,
// pinned to the precisions this build can read.
//
// Comfy-Org publishes five DiT variants and three text-encoder variants
// in one repo, most of them in their own int8_convrot / fp8_scaled /
// nvfp4 packings that no loader here opens. An unpinned fetch would pull
// ~250 GB of which ~180 GB is unreadable, so `files` is a whitelist --
// the same reason the GGUF entries pin one quant. This test is the
// guard on that whitelist: it is data, so nothing else fails when a
// packing we cannot read creeps back in.
TEST(model_catalog, comfy_org_minimax_h3_pins_readable_precisions) {
  const ModelCatalogEntry* e = catalog_by_path("Comfy-Org/MiniMax-H3");
  ASSERT_TRUE(e != nullptr);
  EXPECT_TRUE(e->weight_format == "comfyui");
  EXPECT_TRUE(e->model_type == "minimax-h3-fl2va");
  EXPECT_TRUE(!e->files.empty());
  for (const std::string& f : e->files) {
    // Readable precision, and the FL2VA partition (Ref2VA is a different
    // packed layout that this tree does not implement).
    const bool ok = f.find("_bf16") != std::string::npos
                 || f.find("_fp16") != std::string::npos
                 || f.find("_fp32") != std::string::npos;
    EXPECT_TRUE(ok);
    EXPECT_TRUE(f.find("int8_convrot") == std::string::npos);
    EXPECT_TRUE(f.find("fp8") == std::string::npos);
    EXPECT_TRUE(f.find("nvfp4") == std::string::npos);
    EXPECT_TRUE(f.find("ref2va") == std::string::npos);
  }
  // The released MiniMaxAI entry stays -- the point of this one is to be
  // a second opinion on that one, not to replace it.
  EXPECT_TRUE(catalog_by_path("MiniMaxAI/MiniMax-H3") != nullptr);
}

// The whitelist above says which files are READABLE; this one says the
// set is USABLE. They are different failures: a fetch can pin nothing but
// bf16 and still leave a directory that cannot make a video, which is
// what this entry did before the text encoder was pinned -- a DiT and two
// VAEs with no way to encode a prompt, reported as a successful fetch.
TEST(model_catalog, comfy_org_minimax_h3_fetches_a_runnable_set) {
  const ModelCatalogEntry* e = catalog_by_path("Comfy-Org/MiniMax-H3");
  ASSERT_TRUE(e != nullptr);
  bool dit = false, enc = false, vvae = false, avae = false;
  for (const std::string& f : e->files) {
    if (f.find("diffusion_models/") == 0) { dit = true; }
    if (f.find("text_encoders/") == 0) { enc = true; }
    if (f.find("video_vae") != std::string::npos) { vvae = true; }
    if (f.find("audio_vae") != std::string::npos) { avae = true; }
  }
  // All four components. Audio is not optional here: the DiT emits both
  // modalities from one packed sequence, so a set without the audio VAE
  // has a latent it cannot decode.
  EXPECT_TRUE(dit && enc && vvae && avae);

  // The tokenizer the repack does not ship, landing where the encoder's
  // own search looks (repo-root `tokenizer/`). Assert the DESTINATION,
  // not just that a companion exists: a correct file in the wrong place
  // fetches fine and fails at load.
  bool tok = false;
  for (const auto& c : e->companion_files) {
    EXPECT_TRUE(!c.repo.empty() && !c.file.empty() && !c.dest.empty());
    if (c.dest == "tokenizer/tokenizer.json") {
      tok = true;
      EXPECT_TRUE(c.repo == "MiniMaxAI/MiniMax-H3");
    }
  }
  EXPECT_TRUE(tok);
}

// Both MiniMax-H3 partitions, and that they are two ENTRIES rather than
// one. They share a repo, an encoder and both VAEs, and their DiTs are
// byte-identical in every respect but the weights -- so nothing about a
// checkout distinguishes them except which DiT file it holds, and the
// catalogue is where that distinction has to live.
TEST(model_catalog, minimax_h3_has_both_partitions) {
  const ModelCatalogEntry* fl = nullptr;
  const ModelCatalogEntry* rf = nullptr;
  for (const ModelCatalogEntry& e : model_catalog()) {
    if (e.model_type == "minimax-h3-fl2va" &&
        e.hf_path == "Comfy-Org/MiniMax-H3") { fl = &e; }
    // The repack's, explicitly: both publishers offer this partition, so
    // "the last one that matched" would be whichever the table lists
    // second and the file-list claims below would follow the wrong one.
    if (e.model_type == "minimax-h3-ref2va" &&
        e.hf_path == "Comfy-Org/MiniMax-H3") { rf = &e; }
  }
  ASSERT_TRUE(fl != nullptr);
  ASSERT_TRUE(rf != nullptr);
  EXPECT_TRUE(rf->hf_path == "Comfy-Org/MiniMax-H3");

  // The two pin DIFFERENT DiTs and the SAME everything else -- a
  // ref2va entry that pinned the fl2va DiT would fetch, load and
  // generate, conditioned on nothing.
  auto dit_of = [](const ModelCatalogEntry& e) {
    for (const std::string& f : e.files) {
      if (f.find("diffusion_models/") == 0) { return f; }
    }
    return std::string();
  };
  const std::string a = dit_of(*fl), b = dit_of(*rf);
  EXPECT_TRUE(!a.empty() && !b.empty());
  EXPECT_TRUE(a != b);
  EXPECT_TRUE(b.find("ref2va") != std::string::npos);
  // ...and NOT the pruned variant, which is a different model.
  EXPECT_TRUE(b.find("pruned") == std::string::npos);

  // Ref2VA takes reference video and audio as well as images; declaring
  // otherwise is what the web-ui and model-select filter on.
  bool video_in = false, audio_in = false;
  for (const std::string& i : rf->inputs) {
    if (i == "video") { video_in = true; }
    if (i == "audio") { audio_in = true; }
  }
  EXPECT_TRUE(video_in && audio_in);

  // Its tokenizer companion comes from the Ref2VA partition and lands
  // where the encoder looks.
  bool tok = false;
  for (const auto& c : rf->companion_files) {
    if (c.dest == "tokenizer/tokenizer.json") {
      tok = true;
      EXPECT_TRUE(c.file.find("Ref2VA/") == 0);
    }
  }
  EXPECT_TRUE(tok);
}

// BOTH publishers offer BOTH partitions.
//
// Ref2VA was catalogued from the repack first, and an entry offered in
// only one publisher's spelling reads as a partition only that publisher
// has -- where in fact MiniMaxAI ships `FL2VA/` and `Ref2VA/` side by
// side, each a complete pipeline. The released weights are also the
// reference the repack is diffed against, so the one that cannot be
// selected is the one a disagreement would be settled with.
TEST(model_catalog, minimaxai_publishes_both_partitions_too) {
  const ModelCatalogEntry* fl = nullptr;
  const ModelCatalogEntry* rf = nullptr;
  for (const ModelCatalogEntry& e : model_catalog()) {
    if (e.hf_path != "MiniMaxAI/MiniMax-H3") { continue; }
    if (e.model_type == "minimax-h3-fl2va")  { fl = &e; }
    if (e.model_type == "minimax-h3-ref2va") { rf = &e; }
  }
  ASSERT_TRUE(fl != nullptr);
  ASSERT_TRUE(rf != nullptr);
  EXPECT_TRUE(rf->version == "H3-Ref2VA");
  // Reference video and audio, like the repack's Ref2VA and unlike
  // either FL2VA -- this is what model-select and the web-ui filter on.
  EXPECT_TRUE(has_(rf->inputs, "video"));
  EXPECT_TRUE(has_(rf->inputs, "audio"));

  // The two subtrees are MIRRORS: same names, same shard counts, one
  // prefix apart. Asserted as a mapping rather than as a count, because
  // what breaks a fetch is a MISSING file, and a count matches just as
  // well when a path is wrong.
  EXPECT_TRUE(!fl->files.empty());
  EXPECT_TRUE(fl->files.size() == rf->files.size());
  for (size_t i = 0; i < fl->files.size(); ++i) {
    EXPECT_TRUE(fl->files[i].rfind("FL2VA/", 0) == 0);
    EXPECT_TRUE(rf->files[i].rfind("Ref2VA/", 0) == 0);
    EXPECT_TRUE(fl->files[i].substr(6) == rf->files[i].substr(7));
  }
  // A COMPLETE pipeline each: the DiT, the encoder and both VAEs. A
  // partition pinning only its DiT would fetch 66 GB that cannot encode
  // a prompt, since these are distinct paths from the other partition's
  // rather than the same file under a shared name.
  auto pins_ = [](const ModelCatalogEntry* e, const char* frag) {
    for (const auto& f : e->files) {
      if (f.find(frag) != std::string::npos) { return true; }
    }
    return false;
  };
  EXPECT_TRUE(pins_(rf, "/transformer/model-00013-of-00013.safetensors"));
  EXPECT_TRUE(pins_(rf, "/text_encoder/model-00014-of-00014.safetensors"));
  EXPECT_TRUE(pins_(rf, "/video_vae/source/model.safetensors"));
  EXPECT_TRUE(pins_(rf, "/audio_vae/model.safetensors"));
  EXPECT_TRUE(pins_(rf, "/tokenizer/tokenizer.json"));
  // ...and NOTHING from the other partition, which is the whole point of
  // pinning a subset of a repo that holds both.
  EXPECT_FALSE(pins_(rf, "FL2VA/"));
  EXPECT_FALSE(pins_(fl, "Ref2VA/"));

  std::printf("[model_catalog] MiniMaxAI/MiniMax-H3 publishes both "
              "partitions, %zu files each\n", rf->files.size());
}

// One repo, several models: `catalog_by_path` answers with the FIRST,
// which is the right answer only when there is one. The places this
// bites are the two MiniMax-H3 repos (each publishing both partitions,
// pinning different DiTs out of one directory) and the supplement repo
// (six archives) -- and on an H3 pair the first-match answer is
// silently wrong, because a caller asking for Ref2VA gets FL2VA's file
// list under Ref2VA's name.
//
// Which is why each of those entries carries a `name`: the registration
// key is what a consumer resolves, and two records over ONE directory
// can only be told apart by it.
TEST(model_catalog, several_models_can_share_one_repo) {
  for (const char* repo : {"Comfy-Org/MiniMax-H3", "MiniMaxAI/MiniMax-H3"}) {
    const auto h3 = catalog_all_by_path(repo);
    EXPECT_TRUE(h3.size() == 2);
    bool fl = false, ref = false;
    std::vector<std::string> keys;
    for (const auto* e : h3) {
      if (e->model_type == "minimax-h3-fl2va")  { fl = true; }
      if (e->model_type == "minimax-h3-ref2va") { ref = true; }
      // NAMED, and named DISTINCTLY. Empty would put both records under
      // the repo path, where the second fetch overwrites the first and
      // the surviving record carries one partition's model_type for
      // both.
      EXPECT_FALSE(e->name.empty());
      EXPECT_TRUE(e->name != e->hf_path);
      EXPECT_TRUE(!has_(keys, e->name));
      keys.push_back(e->name);
    }
    EXPECT_TRUE(fl && ref);
  }
  const auto h3 = catalog_all_by_path("Comfy-Org/MiniMax-H3");

  // And they pin DIFFERENT files, which is why picking the wrong one is
  // a wrong download rather than a cosmetic mislabel.
  auto pins = [](const ModelCatalogEntry* e, const char* frag) {
    for (const auto& f : e->files) {
      if (f.find(frag) != std::string::npos) { return true; }
    }
    return false;
  };
  for (const auto* e : h3) {
    const bool is_ref = e->model_type == "minimax-h3-ref2va";
    EXPECT_TRUE(pins(e, is_ref ? "ref2va" : "fl2va"));
    EXPECT_FALSE(pins(e, is_ref ? "fl2va" : "ref2va"));
  }

  const auto sup = catalog_all_by_path("tgo-app-dev/vpipe-supplement");
  EXPECT_TRUE(sup.size() > 1);

  // A repo publishing exactly one model still answers with one, so the
  // fetch stage's "no ambiguity, no key needed" path stays the norm.
  const auto one = catalog_all_by_path("Qwen/Qwen3.8-27B");
  EXPECT_TRUE(one.size() == 1);
  EXPECT_TRUE(catalog_all_by_path("no/such-repo").empty());

  std::printf("[model_catalog] Comfy-Org/MiniMax-H3 publishes %zu models, "
              "vpipe-supplement %zu\n", h3.size(), sup.size());
}

// THE SHIPPED PREPARE PIPELINES NAME A MODEL THAT WILL EXIST.
//
// docs/pipelines/prepare-*.vpipeline fetch a model and then quantize it,
// and the second stage names the first stage's result by its DB KEY. The
// key is not the repo path: model-fetch registers under `model_key` when
// set, else the catalogue entry's `name`, else the hf path -- so for any
// repo publishing more than one model, where the catalogue gives each a
// distinct name, the repo path is exactly the string that will NOT
// resolve.
//
// prepare-minimax-h3-8bit shipped that way. Its fetch pulled
// Comfy-Org/MiniMax-H3 with model_variant=fl2va, which registers as
// "Comfy-Org/MiniMax-H3-FL2VA", and its quantize asked for
// "Comfy-Org/MiniMax-H3" -- so the fetch succeeded and the quantize
// could not find its input. Both Ref2VA siblings had it right, which is
// what made it a one-file slip rather than a misunderstanding.
//
// Nothing else reads these files: they are documentation, so a wrong key
// is invisible until a user runs one. This walks them instead.
TEST(model_catalog, shipped_prepare_pipelines_name_a_resolvable_model)
{
#ifndef VPIPE_DOCS_PIPELINES_DIR
  std::printf("[catalog] no docs dir defined; skipped\n");
#else
  namespace fs = std::filesystem;
  const fs::path dir(VPIPE_DOCS_PIPELINES_DIR);
  if (!fs::is_directory(dir)) {
    std::printf("[catalog] %s is not a directory; skipped\n",
                dir.string().c_str());
    return;
  }
  // The key a model-fetch stage will register under, by the same rule
  // model-fetch-stage.cc applies.
  auto fetch_key = [](const std::string& path, const std::string& key,
                      const std::string& variant) {
    if (!key.empty()) { return key; }
    const ModelCatalogEntry* hit = nullptr;
    for (const ModelCatalogEntry& e : model_catalog()) {
      if (e.hf_path != path) { continue; }
      if (variant.empty()) { hit = &e; break; }
      // model_variant is matched against several descriptive fields; for
      // this check a case-insensitive substring over them is enough to
      // pick the entry the fetch would.
      auto has = [&](const std::string& f) {
        if (f.empty()) { return false; }
        std::string a = f, b = variant;
        std::transform(a.begin(), a.end(), a.begin(), ::tolower);
        std::transform(b.begin(), b.end(), b.begin(), ::tolower);
        return a.find(b) != std::string::npos;
      };
      if (has(e.version) || has(e.variant) || has(e.name) ||
          has(e.model_type)) {
        hit = &e;
        break;
      }
    }
    if (hit != nullptr && !hit->name.empty()) { return hit->name; }
    return path;
  };

  int checked = 0;
  for (const fs::directory_entry& de : fs::directory_iterator(dir)) {
    const std::string fn = de.path().filename().string();
    if (fn.rfind("prepare-", 0) != 0) { continue; }
    if (de.path().extension() != ".vpipeline") { continue; }
    std::string txt, line;
    {
      std::ifstream in(de.path());
      while (std::getline(in, line)) { txt += line + "\n"; }
    }
    FlexData fd = FlexData::from_json(txt);
    auto root = fd.as_object();
    // A malformed example pipeline is its own failure, and a silent
    // `continue` here would let one hide the rest of the file's checks.
    EXPECT_TRUE(root.contains("stages"));
    if (!root.contains("stages")) { continue; }
    const FlexData stages = root.at("stages");
    // Keys a stage in THIS pipeline produces, so a later src_model may
    // name an intermediate that does not exist on disk yet.
    std::vector<std::string> produced;
    for (const FlexData& st : stages.as_array()) {
      auto o = st.as_object();
      if (!o.contains("config")) { continue; }
      const FlexData cfg = o.at("config");
      auto c = cfg.as_object();
      auto str = [&](const char* k) {
        return c.contains(k) ? std::string(c.at(k).as_string("")) : "";
      };
      const std::string type = std::string(o.at("type").as_string(""));
      if (type == "model-fetch") {
        produced.push_back(fetch_key(str("model_path"), str("model_key"),
                                     str("model_variant")));
        continue;
      }
      if (type != "model-quantize") { continue; }
      const std::string src = str("src_model");
      if (src.empty()) { continue; }
      const bool from_pipeline =
          std::find(produced.begin(), produced.end(), src) != produced.end();
      if (!from_pipeline) {
        std::printf("[catalog] %s: quantize src_model '%s' is neither a key "
                    "an earlier stage registers nor its output\n",
                    fn.c_str(), src.c_str());
      }
      EXPECT_TRUE(from_pipeline);
      ++checked;
      if (!str("output_name").empty()) {
        produced.push_back(str("output_name"));
      }
    }
  }
  std::printf("[catalog] %d quantize stages across the shipped prepare "
              "pipelines name a model an earlier stage produces\n", checked);
  EXPECT_TRUE(checked > 0);
#endif
}

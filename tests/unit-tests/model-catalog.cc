#include "minitest.h"
#include "stages/model-catalog.h"
#include "stages/qwen-asr-tokenizer.h"
#include "common/flex-data.h"

#include <algorithm>
#include <string>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {
bool has_(const vector<string>& v, const string& x) {
  return std::find(v.begin(), v.end(), x) != v.end();
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
    const char* version;
    const char* param_class;
    const char* variant;
    const char* model_type;
    const char* name;
    const char* tar;
  };
  const Want wants[] = {
    {"3.5", "4B", "Vision tower CoreML 512x320 w8 (vpipe-supplement)",
     "qwen3.5-vision-encoder", "qwen3_5_mlx_4b_vision_vid_512x320",
     "qwen3_5_mlx_4b_vision_vid_512x320_w8.tar"},
    {"L", "1024x640", "CoreML w8 (vpipe-supplement)",
     "yolo", "yolox_l_1024x640", "yolox_l_1024x640_w8.tar"},
    {"VAD v6", "unified", "CoreML (vpipe-supplement)",
     "silero-vad", "silero_vad_unified_v6", "silero-vad-unified-v6.tar"},
  };
  for (const Want& w : wants) {
    const char* family = (w.model_type[0] == 'q') ? "Qwen"
                       : (w.model_type[0] == 'y') ? "YOLOX" : "Silero";
    const ModelCatalogEntry* e =
        catalog_find(family, w.version, w.param_class, w.variant);
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

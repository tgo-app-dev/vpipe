#include "stages/model-catalog.h"
#include "common/flex-data.h"

#include <algorithm>
#include <cctype>

namespace vpipe {

const std::vector<ModelCatalogEntry>&
model_catalog()
{
  // ==================================================================
  // MODEL CATALOGUE -- single edit point. Append an entry to add a
  // model; the selection menu rebuilds itself from this table.
  // ==================================================================
  static const std::vector<ModelCatalogEntry> kCatalog = {
    {.family = "Qwen", .version = "3.5", .param_class = "9B",
     .variant = "MLX 4-bit (lmstudio-community)",
     .hf_path = "lmstudio-community/Qwen3.5-9B-MLX-4bit",
     .model_type = "qwen3.5", .needs_tokenizer_json = false},
    {.family = "Qwen", .version = "3.5", .param_class = "9B",
     .variant = "MLX 4-bit (mlx-community)",
     .hf_path = "mlx-community/Qwen3.5-9B-MLX-4bit",
     .model_type = "qwen3.5", .needs_tokenizer_json = false},
    {.family = "Qwen", .version = "3.5", .param_class = "9B",
     .variant = "MLX OptiQ 4-bit (mlx-community)",
     .hf_path = "mlx-community/Qwen3.5-9B-OptiQ-4bit",
     .model_type = "qwen3.5", .needs_tokenizer_json = false},
    {.family = "Qwen", .version = "3.5", .param_class = "4B",
     .variant = "MLX 4-bit (lmstudio-community)",
     .hf_path = "lmstudio-community/Qwen3.5-4B-MLX-4bit",
     .model_type = "qwen3.5", .needs_tokenizer_json = false},
    {.family = "Qwen", .version = "3.5", .param_class = "4B",
     .variant = "MLX 4-bit (mlx-community)",
     .hf_path = "mlx-community/Qwen3.5-4B-MLX-4bit",
     .model_type = "qwen3.5", .needs_tokenizer_json = false},
    {.family = "Qwen", .version = "3.5", .param_class = "4B",
     .variant = "MLX OptiQ 4-bit (mlx-community)",
     .hf_path = "mlx-community/Qwen3.5-4B-OptiQ-4bit",
     .model_type = "qwen3.5", .needs_tokenizer_json = false},
    {.family = "Qwen", .version = "3.5", .param_class = "4B",
     .variant = "MLX 8-bit (lmstudio-community)",
     .hf_path = "lmstudio-community/Qwen3.5-4B-MLX-8bit",
     .model_type = "qwen3.5", .needs_tokenizer_json = false},
    {.family = "Qwen", .version = "3.5", .param_class = "4B",
     .variant = "MLX 8-bit (mlx-community)",
     .hf_path = "mlx-community/Qwen3.5-4B-MLX-8bit",
     .model_type = "qwen3.5", .needs_tokenizer_json = false},
    {.family = "Qwen", .version = "3.5", .param_class = "4B",
     .variant = "Q4_K_M GGUF +mmproj (unsloth)",
     .hf_path = "unsloth/Qwen3.5-4B-GGUF", .model_type = "qwen3.5",
     .files = {"Qwen3.5-4B-Q4_K_M.gguf",   // main quant (text)
               "mmproj-BF16.gguf"},        // BF16 multimodal projector
     .needs_tokenizer_json = false},
    {.family = "Qwen", .version = "3.5", .param_class = "4B",
     .variant = "MTP Q4_K_M GGUF +mmproj +imatrix (unsloth)",
     .hf_path = "unsloth/Qwen3.5-4B-MTP-GGUF", .model_type = "qwen3.5",
     .files = {"Qwen3.5-4B-Q4_K_M.gguf",   // main quant (MTP)
               "mmproj-BF16.gguf",         // BF16 multimodal projector
               "imatrix_unsloth.gguf_file"},  // imatrix
     .needs_tokenizer_json = false},
    {.family = "Qwen", .version = "3.5", .param_class = "2B",
     .variant = "MLX 4-bit (lmstudio-community)",
     .hf_path = "lmstudio-community/Qwen3.5-2B-MLX-4bit",
     .model_type = "qwen3.5", .needs_tokenizer_json = false},
    {.family = "Qwen", .version = "3.5", .param_class = "2B",
     .variant = "MLX 4-bit (mlx-community)",
     .hf_path = "mlx-community/Qwen3.5-2B-MLX-4bit",
     .model_type = "qwen3.5", .needs_tokenizer_json = false},
    {.family = "Qwen", .version = "3.5", .param_class = "2B",
     .variant = "MLX OptiQ 4-bit (mlx-community)",
     .hf_path = "mlx-community/Qwen3.5-2B-OptiQ-4bit",
     .model_type = "qwen3.5", .needs_tokenizer_json = false},
    {.family = "Qwen", .version = "3.5", .param_class = "2B",
     .variant = "MLX 8-bit (lmstudio-community)",
     .hf_path = "lmstudio-community/Qwen3.5-2B-MLX-8bit",
     .model_type = "qwen3.5", .needs_tokenizer_json = false},
    {.family = "Qwen", .version = "3.5", .param_class = "2B",
     .variant = "MLX 8-bit (mlx-community)",
     .hf_path = "mlx-community/Qwen3.5-2B-MLX-8bit",
     .model_type = "qwen3.5", .needs_tokenizer_json = false},
    {.family = "Qwen", .version = "3.5", .param_class = "2B",
     .variant = "Q4_K_M GGUF +mmproj (unsloth)",
     .hf_path = "unsloth/Qwen3.5-2B-GGUF", .model_type = "qwen3.5",
     .files = {"Qwen3.5-2B-Q4_K_M.gguf",   // main quant (text)
               "mmproj-BF16.gguf"},        // BF16 multimodal projector
     .needs_tokenizer_json = false},
    // ---- Qwen 3.6 (multimodal MTP, GGUF) -----------------------------
    {.family = "Qwen", .version = "3.6", .param_class = "27B",
     .variant = "Q4_K_M GGUF +mmproj +imatrix (unsloth)",
     .hf_path = "unsloth/Qwen3.6-27B-MTP-GGUF", .model_type = "qwen3.6",
     .files = {"Qwen3.6-27B-Q4_K_M.gguf",   // main quant (MTP)
               "mmproj-BF16.gguf",          // BF16 multimodal projector
               "imatrix_unsloth.gguf_file"},  // imatrix
     .needs_tokenizer_json = false},
    {.family = "Qwen", .version = "3-ASR", .param_class = "1.7B",
     .variant = "ASR MLX 8-bit (mlx-community)",
     .hf_path = "mlx-community/Qwen3-ASR-1.7B-8bit",
     .model_type = "qwen3-asr", .needs_tokenizer_json = true},
    {.family = "Qwen", .version = "3-ASR", .param_class = "0.6B",
     .variant = "ASR MLX 8-bit (mlx-community)",
     .hf_path = "mlx-community/Qwen3-ASR-0.6B-8bit",
     .model_type = "qwen3-asr", .needs_tokenizer_json = true},
    {.family = "Gemma", .version = "4", .param_class = "e4b",
     .variant = "MLX 4-bit (mlx-community)",
     .hf_path = "mlx-community/gemma-4-e4b-it-4bit",
     .model_type = "gemma4", .needs_tokenizer_json = false},
    {.family = "Gemma", .version = "4", .param_class = "12B",
     .variant = "GGUF QAT q4_0 (google, gated)",
     .hf_path = "google/gemma-4-12B-it-qat-q4_0-gguf",
     .model_type = "gemma4_unified", .needs_tokenizer_json = false},
    // ---- MOSS-TTS (text-to-speech: LM + audio codec) -----------------
    // The text-to-speech stage consumes TWO models: the MOSS-TTS LM
    // (the stage's hf_dir, model_type "moss-tts") and the
    // MOSS-Audio-Tokenizer codec (the stage's codec_dir, model_type
    // "moss-codec"). Both are whole-repo fetches (empty `files`):
    // sharded safetensors with no consolidated tokenizer.json to
    // synthesize. The stage's hf_dir/codec_dir fields filter the
    // suggestion dropdown on these two model_types.
    {.family = "MOSS", .version = "TTS", .param_class = "8B",
     .variant = "MLX 8-bit (mlx-community)",
     .hf_path = "mlx-community/MOSS-TTS-8B-8bit",
     .model_type = "moss-tts", .needs_tokenizer_json = false},
    {.family = "MOSS", .version = "Audio-Tokenizer", .param_class = "codec",
     .variant = "F32 (OpenMOSS-Team)",
     .hf_path = "OpenMOSS-Team/MOSS-Audio-Tokenizer",
     .model_type = "moss-codec", .needs_tokenizer_json = false},
    // ---- Supplementary CoreML models (vpipe-supplement) --------------
    // One pre-converted *.mlpackage per .tar; all share ONE repo, so each
    // entry pins its archive + a distinct `name` (= registration key /
    // extract subdir) and sets extract_archive so the fetcher unpacks the
    // .tar and registers the contained .mlpackage. The model_type is the
    // compatibility hint the stages + web-ui filter on.
    {.family = "Qwen", .version = "3.5", .param_class = "4B",
     .variant = "Vision tower CoreML 512x320 w8 (vpipe-supplement)",
     .hf_path = "tgo-app-dev/vpipe-supplement",
     .model_type = "qwen3.5-vision-encoder",
     .files = {"qwen3_5_mlx_4b_vision_vid_512x320_w8.tar"},
     .name = "qwen3_5_mlx_4b_vision_vid_512x320",
     .extract_archive = true},
    {.family = "Qwen", .version = "3.5", .param_class = "4B",
     .variant = "Vision tower CoreML 768x480 w8 (vpipe-supplement)",
     .hf_path = "tgo-app-dev/vpipe-supplement",
     .model_type = "qwen3.5-vision-encoder",
     .files = {"qwen3_5_mlx_4b_vision_vid_768x480_w8.tar"},
     .name = "qwen3_5_mlx_4b_vision_vid_768x480",
     .extract_archive = true},
    {.family = "Gemma", .version = "4", .param_class = "e4b",
     .variant = "Vision tower CoreML 768x480 w8 (vpipe-supplement)",
     .hf_path = "tgo-app-dev/vpipe-supplement",
     .model_type = "gemma4-vision-encoder",
     .files = {"gemma4_mlx_e4b_vision_768x480_w8.tar"},
     .name = "gemma4_mlx_e4b_vision_768x480",
     .extract_archive = true},
    {.family = "YOLOX", .version = "L", .param_class = "1024x640",
     .variant = "CoreML w8 (vpipe-supplement)",
     .hf_path = "tgo-app-dev/vpipe-supplement",
     .model_type = "yolo",
     .files = {"yolox_l_1024x640_w8.tar"},
     .name = "yolox_l_1024x640",
     .extract_archive = true},
    {.family = "Silero", .version = "VAD v6", .param_class = "unified",
     .variant = "CoreML (vpipe-supplement)",
     .hf_path = "tgo-app-dev/vpipe-supplement",
     .model_type = "silero-vad",
     .files = {"silero-vad-unified-v6.tar"},
     .name = "silero_vad_unified_v6",
     .extract_archive = true},
    {.family = "BEATs", .version = "iter3+", .param_class = "AS2M",
     .variant = "Audio tagging CoreML 10s (vpipe-supplement)",
     .hf_path = "tgo-app-dev/vpipe-supplement",
     .model_type = "audio-tagging",
     .files = {"beats_as2m_10s.tar"},
     .name = "beats_as2m_10s",
     .extract_archive = true},
  };
  return kCatalog;
}

namespace {

// Append `v` to `out` if not already present (order-preserving dedupe).
void
push_unique_(std::vector<std::string>& out, const std::string& v)
{
  if (std::find(out.begin(), out.end(), v) == out.end()) {
    out.push_back(v);
  }
}

std::string
lower_(std::string s)
{
  for (char& c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

// Drop a case-insensitive prefix from `s` if present; returns whether it
// matched.
bool
strip_prefix_(std::string& s, const std::string& prefix)
{
  if (s.size() >= prefix.size()
      && lower_(s.substr(0, prefix.size())) == prefix) {
    s.erase(0, prefix.size());
    return true;
  }
  return false;
}

}  // namespace

std::vector<std::string>
catalog_families()
{
  std::vector<std::string> out;
  for (const auto& e : model_catalog()) {
    push_unique_(out, e.family);
  }
  return out;
}

std::vector<std::string>
catalog_versions(const std::string& family)
{
  std::vector<std::string> out;
  for (const auto& e : model_catalog()) {
    if (e.family == family) {
      push_unique_(out, e.version);
    }
  }
  return out;
}

std::vector<std::string>
catalog_param_classes(const std::string& family, const std::string& version)
{
  std::vector<std::string> out;
  for (const auto& e : model_catalog()) {
    if (e.family == family && e.version == version) {
      push_unique_(out, e.param_class);
    }
  }
  return out;
}

std::vector<std::string>
catalog_variants(const std::string& family, const std::string& version,
                 const std::string& param_class)
{
  std::vector<std::string> out;
  for (const auto& e : model_catalog()) {
    if (e.family == family && e.version == version
        && e.param_class == param_class) {
      push_unique_(out, e.variant);
    }
  }
  return out;
}

const ModelCatalogEntry*
catalog_find(const std::string& family, const std::string& version,
             const std::string& param_class, const std::string& variant)
{
  for (const auto& e : model_catalog()) {
    if (e.family == family && e.version == version
        && e.param_class == param_class && e.variant == variant) {
      return &e;
    }
  }
  return nullptr;
}

const ModelCatalogEntry*
catalog_by_path(const std::string& hf_path)
{
  for (const auto& e : model_catalog()) {
    if (e.hf_path == hf_path) {
      return &e;
    }
  }
  return nullptr;
}

std::string
normalize_hf_path(const std::string& input)
{
  std::string s = input;
  // Trim surrounding whitespace.
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  if (s.empty()) {
    return {};
  }
  // Strip scheme + host so both full URLs and bare paths normalise.
  strip_prefix_(s, "https://");
  strip_prefix_(s, "http://");
  strip_prefix_(s, "www.");
  strip_prefix_(s, "huggingface.co/");
  // Drop query / fragment.
  s = s.substr(0, s.find_first_of("?#"));

  // Take the first two non-empty '/'-separated segments: owner/repo.
  std::string owner, repo;
  size_t i = 0;
  auto next_segment = [&](std::string& dst) {
    while (i < s.size() && s[i] == '/') { ++i; }
    size_t start = i;
    while (i < s.size() && s[i] != '/') { ++i; }
    dst = s.substr(start, i - start);
  };
  next_segment(owner);
  next_segment(repo);
  if (owner.empty() || repo.empty()) {
    return {};
  }
  return owner + "/" + repo;
}

std::vector<HfFile>
hf_tree_files(const FlexData& tree_json)
{
  std::vector<HfFile> out;
  if (!tree_json.is_array()) {
    return out;
  }
  auto arr = tree_json.as_array();
  for (std::size_t i = 0; i < arr.size(); ++i) {
    FlexData entry = arr.at(i);   // own a copy; views dangle off temporaries
    if (!entry.is_object()) {
      continue;
    }
    auto obj = entry.as_object();
    const std::string type = obj.contains("type")
        ? std::string(obj.at("type").as_string("")) : "";
    if (type != "file") {
      continue;
    }
    if (!obj.contains("path")) {
      continue;
    }
    std::string path(obj.at("path").as_string(""));
    if (path.empty()) {
      continue;
    }
    HfFile f;
    f.path = std::move(path);
    f.size = obj.contains("size") ? obj.at("size").as_uint(0) : 0;
    out.push_back(std::move(f));
  }
  return out;
}

}

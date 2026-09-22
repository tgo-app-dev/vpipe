#include "stages/model-detect.h"

#include "generative-models/detect-profile.h"

#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"

#include "common/flex-data.h"
#include "generative-models/shared/comfy-checkpoint.h"
#include "stages/model-catalog.h"
#include "stages/model-registry.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>

namespace vpipe {

namespace {

namespace fs = std::filesystem;

std::string
lower_(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return (char)std::tolower(c); });
  return s;
}

bool
has_(const std::string& hay_lc, const char* needle)
{
  return hay_lc.find(needle) != std::string::npos;
}

// Read a JSON file into a FlexData object. Returns a Null FlexData when
// the file is absent / unreadable / not an object -- callers test
// is_object() and move on, so a malformed config never throws out of
// detection.
FlexData
read_json_(const fs::path& p)
{
  std::error_code ec;
  if (!fs::exists(p, ec) || ec) {
    return {};
  }
  std::ifstream in(p);
  if (!in) {
    return {};
  }
  try {
    FlexData fd = FlexData::from_json(in);
    if (fd.is_object()) {
      return fd;
    }
  } catch (...) {
  }
  return {};
}

std::string
str_field_(const FlexData& obj, const char* key)
{
  if (!obj.is_object()) {
    return {};
  }
  auto o = obj.as_object();
  return o.contains(key) ? std::string(o.at(key).as_string("")) : std::string();
}

bool
has_field_(const FlexData& obj, const char* key)
{
  return obj.is_object() && obj.as_object().contains(key);
}

std::int64_t
int_field_(const FlexData& obj, const char* key, std::int64_t dflt)
{
  if (!obj.is_object()) {
    return dflt;
  }
  auto o = obj.as_object();
  return o.contains(key) ? o.at(key).as_int(dflt) : dflt;
}

// Split a name on the separators HF repos use, so "gemma-4-e4b-it-4bit"
// becomes {gemma,4,e4b,it,4bit}.
std::vector<std::string>
tokens_(const std::string& name)
{
  std::vector<std::string> out;
  std::string cur;
  for (char c : name) {
    if (c == '-' || c == '_' || c == '.' || c == ' ') {
      if (!cur.empty()) { out.push_back(cur); cur.clear(); }
      // '.' also separates, but "3.5" / "1.7B" must survive: re-join
      // below when both sides are numeric.
      if (c == '.') { out.push_back("."); }
    } else {
      cur.push_back(c);
    }
  }
  if (!cur.empty()) { out.push_back(cur); }
  // Re-join "1" "." "7B" -> "1.7B".
  std::vector<std::string> joined;
  for (size_t i = 0; i < out.size(); ++i) {
    if (out[i] == "." && !joined.empty() && i + 1 < out.size()) {
      joined.back() += "." + out[i + 1];
      ++i;
    } else if (out[i] != ".") {
      joined.push_back(out[i]);
    }
  }
  return joined;
}

// A parameter-count token: 4B, 1.7B, 27B, E4B (Gemma's effective sizes),
// optionally followed by the MoE active-parameter token (35B + A3B ->
// "35B-A3B", the spelling the catalogue uses).
bool
param_token_(const std::string& t)
{
  if (t.size() < 2) { return false; }
  size_t i = 0;
  if (t[0] == 'e' || t[0] == 'E') { i = 1; }
  bool digit = false;
  for (; i + 1 < t.size(); ++i) {
    if (std::isdigit((unsigned char)t[i])) { digit = true; }
    else if (t[i] != '.') { return false; }
  }
  return digit && (t.back() == 'b' || t.back() == 'B');
}

bool
active_param_token_(const std::string& t)
{
  if (t.size() < 3 || (t[0] != 'a' && t[0] != 'A')) { return false; }
  return param_token_(t.substr(1));
}

std::string
upper_(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return (char)std::toupper(c); });
  return s;
}

// "Qwen3.5-4B-MLX-4bit" -> "4B"; "gemma-4-e4b-it" -> "E4B";
// "Qwen3.6-35B-A3B-OptiQ" -> "35B-A3B". Empty when the name says nothing.
std::string
param_class_from_name_(const std::string& name)
{
  const auto tk = tokens_(name);
  for (size_t i = 0; i < tk.size(); ++i) {
    if (!param_token_(tk[i])) { continue; }
    std::string pc = upper_(tk[i]);
    if (i + 1 < tk.size() && active_param_token_(tk[i + 1])) {
      pc += "-" + upper_(tk[i + 1]);
    }
    return pc;
  }
  return {};
}

// family / version display labels for a runtime tag, matching the
// catalogue's spelling so a registered model sorts beside its
// catalogued siblings in the web-ui browser.
void
family_version_(const std::string& mt, std::string& family,
                std::string& version)
{
  struct Row { const char* mt; const char* family; const char* version; };
  static const Row kRows[] = {
      {"qwen3.5",                "Qwen",        "3.5"},
      {"qwen3.6",                "Qwen",        "3.6"},
      {"qwen3-asr",              "Qwen",        "3-ASR"},
      {"qwen3.5-vision-encoder", "Qwen",        "3.5"},
      {"gemma4",                 "Gemma",       "4"},
      {"gemma4_unified",         "Gemma",       "4"},
      {"gemma4-vision-encoder",  "Gemma",       "4"},
      {"llama3",                 "Llama",       "3"},
      {"krea2",                  "Krea",        "2"},
      {"krea2-lora",             "Krea",        "2"},
      {"flux2",                  "FLUX",        "2"},
      {"flux2-lora",             "FLUX",        "2"},
      {"qwen-image",             "Qwen-Image",  "2512"},
      {"qwen-image-edit",        "Qwen-Image",  "Edit-2511"},
      {"qwen-image-21",          "Qwen-Image",  "2.1"},
      {"qwen-image-21-dit",      "Qwen-Image",  "2.1"},
      {"z-image",                "Z-Image",     "1"},
      {"z-image-dit",            "Z-Image",     "1"},
      {"boogu-image",            "Boogu-Image", "0.1"},
      {"boogu-image-edit",       "Boogu-Image", "0.1-Edit"},
      {"wan-i2v",                "Wan",         "2.2-I2V"},
      {"wan-t2v",                "Wan",         "2.2-T2V"},
      {"minimax-h3-fl2va",       "MiniMax",     "H3-FL2VA"},
      {"minimax-h3-ref2va",      "MiniMax",     "H3-Ref2VA"},
      {"minimax-h3-dit",         "MiniMax",     "H3-FL2VA"},
      {"minimax-h3-text-encoder", "MiniMax",    "H3-FL2VA"},
      {"moss-tts",               "MOSS",        "TTS"},
      {"moss-tts-local",         "MOSS",        "TTS-Local"},
      {"moss-tts-realtime",      "MOSS",        "TTS-Realtime"},
      {"moss-codec",             "MOSS",        "Audio-Tokenizer"},
      {"moss-codec-v2",          "MOSS",        "Audio-Tokenizer"},
      {"yolo",                   "YOLOX",       "L"},
      {"silero-vad",             "Silero",      "VAD"},
      {"audio-tagging",          "BEATs",       "iter3+"},
  };
  for (const auto& r : kRows) {
    if (mt == r.mt) { family = r.family; version = r.version; return; }
  }
  // A REGISTERED family's own labels. The table above is the families
  // this tree implements; a plugin's belongs with the plugin, and
  // without it an out-of-tree checkpoint shows in a browser as a model
  // type and no name.
  if (const FlexData* p = genai::detect::for_model_type(mt)) {
    const std::string f = genai::detect::text(p, genai::detect::kLabelFamily,
                                              "");
    if (!f.empty()) { family = f; }
    const bool is_edit =
        genai::detect::text(p, genai::detect::kModelTypeEdit, "") == mt;
    const std::string v = genai::detect::text(
        p, is_edit ? genai::detect::kLabelVersionEdit
                   : genai::detect::kLabelVersion, "");
    if (!v.empty()) { version = v; }
  }
}

// ---- config.json (HF / MLX layout) -> runtime tag --------------------
//
// The config alone is not always decisive, so the directory NAME breaks
// the two ties it cannot:
//   * Qwen3.6 checkpoints still report model_type "qwen3_5" (same
//     architecture class), and
//   * every Gemma-4 reports "gemma4", but the E-sized (E2B/E4B) ones run
//     the per-layer-embedding path ("gemma4") while the 12B/26B/31B run
//     the unified one ("gemma4_unified") -- which is exactly the
//     distinction the catalogue draws by param_class.
std::string
lm_tag_(const std::string& cfg_type, const std::string& name_lc,
        const std::string& param_class)
{
  const std::string mt = lower_(cfg_type);
  auto starts = [&](const char* p) { return mt.rfind(p, 0) == 0; };

  if (starts("qwen3_asr") || starts("qwen3asr")) { return "qwen3-asr"; }
  if (starts("qwen3_5") || starts("qwen3_6")) {
    if (starts("qwen3_6") || has_(name_lc, "qwen3.6")
        || has_(name_lc, "qwen3_6")) {
      return "qwen3.6";
    }
    return "qwen3.5";
  }
  if (starts("gemma")) {
    // E2B / E4B -> the effective (gemma3n-style) path; every other size
    // is the unified one.
    return (!param_class.empty() && param_class[0] == 'E') ? "gemma4"
                                                           : "gemma4_unified";
  }
  if (starts("moss_tts_local"))    { return "moss-tts-local"; }
  if (starts("moss_tts_realtime")) { return "moss-tts-realtime"; }
  if (starts("moss_tts"))          { return "moss-tts"; }
  if (starts("llama"))             { return "llama3"; }
  return {};
}

// ---- transformer/config.json (diffusers layout) -> runtime tag -------
// `partition` is model_index.json's `_minimax_h3.partition` (empty for
// every other family): MiniMax-H3 ships two complete pipelines whose
// transformer/config.json files are BYTE-IDENTICAL, so the DiT config
// alone cannot tell FL2VA from Ref2VA and the pipeline manifest has to.
std::string
dit_tag_(const std::string& cls, const std::string& name_lc,
         const FlexData& cfg, const std::string& partition)
{
  const bool edit = has_(name_lc, "edit");
  // FLUX.2-klein-9b-kv reports this SAME class, and its config.json and tensor
  // names are identical to plain klein-9B's -- the two differ only in the
  // weights and in the attention recipe those weights were distilled for. So
  // "flux2" is the right answer for both, and the recipe is selected by the
  // consuming stage's `klein_kv` key rather than guessed here. Deliberately
  // NOT inferred from the name the way `edit` is below: a wrong `edit` guess
  // picks a sibling checkpoint, while a wrong recipe guess silently produces
  // wrong IMAGES from a model that loads and runs perfectly.
  // A REGISTERED family, asked FIRST only in the sense that its answer
  // is checked before the fall-through below -- the built-in names are
  // still matched first, so a plugin cannot take a class name this tree
  // owns. Without this a checkpoint of an out-of-tree family sitting in
  // a models directory cannot be named at all.
  //
  // Deliberately a LABELLING answer, not a loading one: what runs a
  // checkpoint is `claims()` on the model registries, which reads the
  // weights rather than a table.
  if (cls == "Flux2Transformer2DModel")      { return "flux2"; }
  if (cls == "Krea2Transformer2DModel")      { return "krea2"; }
  if (cls == "QwenImageTransformer2DModel")  { return "qwen-image-edit"; }
  // Qwen-Image-2.1. NOT split on `edit` the way Boogu is: one checkpoint
  // answers both tasks, because a reference image occupies vision slots
  // in the text encoder's own sequence instead of a separate
  // conditioning path. There is no sibling to pick wrongly.
  if (cls == "QwenImage21Transformer2DModel") { return "qwen-image-21"; }
  // Z-Image. ONE tag for both published checkpoints: Turbo and the
  // undistilled base ship byte-identical transformer configs, and what
  // separates them -- the scheduler shift and whether CFG is wanted --
  // is read from scheduler_config.json and the graph, never inferred
  // from a name. A tag per variant would promise a distinction this
  // file cannot actually make.
  if (cls == "ZImageTransformer2DModel") { return "z-image"; }
  if (cls == "BooguImageTransformer2DModel") {
    // The t2i and edit repos ship the SAME transformer config (only the
    // weights differ), so the name is the only signal.
    return edit ? "boogu-image-edit" : "boogu-image";
  }
  if (cls == "WanTransformer3DModel") {
    // Wan's t2v and i2v checkpoints share this class, but NOT their input
    // width: i2v takes 36 channels (16 noise + 4 first-frame mask + 16
    // conditioning latent) where t2v takes the bare 16. That is a property
    // of the weights rather than of the repo name, so read it rather than
    // guessing from the directory.
    return int_field_(cfg, "in_channels", 16) > 16 ? "wan-i2v" : "wan-t2v";
  }
  if (cls == "MiniMaxH3DiTModel") {
    // The two partitions ship BYTE-IDENTICAL transformer configs (same
    // 535 tensor names, same shapes), so the DiT alone cannot tell them
    // apart and the pipeline manifest has to -- and it matters, because
    // they are different TASKS over the same architecture: fl2va takes
    // an optional first/last frame, ref2va takes up to 9 reference
    // images / 3 clips / 3 audio clips and packs a block per reference.
    // A checkpoint tagged as the wrong one would load and run and
    // condition on nothing.
    if (partition == "fl2va")  { return "minimax-h3-fl2va"; }
    if (partition == "ref2va") { return "minimax-h3-ref2va"; }
    return {};
  }
  // Last: a REGISTERED family. Asked after every built-in name so a
  // plugin cannot take a class this tree owns.
  return genai::detect::model_type_for_class(cls, edit);
}

// ---- a Comfy-Org repack ----------------------------------------------
// One .safetensors per component under diffusion_models/ | vae/ |
// text_encoders/, each carrying its config in the safetensors
// `__metadata__` rather than in a config.json -- so there is no
// transformer/config.json for the diffusers probe above to find, and the
// directory would otherwise register as unrecognized.
//
// The DiT file is what names the model: its metadata's `image_model` is
// Comfy-Org's architecture tag, and the filename carries whatever the
// architecture alone cannot say (for MiniMax-H3, which of the two
// partitions the weights are).
//
// EVERY DiT in the repack has to agree, or there is no answer.
// Comfy-Org's MiniMax-H3 repo publishes BOTH partitions under
// diffusion_models/, and scan_repo walks it in filename order -- so
// taking the first tag registered a two-partition checkout as whichever
// sorts first, silently and always the same way.
std::string
comfy_repo_tag_(const fs::path& root)
{
  std::string tag;
  bool seen = false;
  for (const genai::comfy::Component& c :
       genai::comfy::scan_repo(root.string())) {
    if (c.role != "diffusion_models" || c.meta_key != "config") { continue; }
    FlexData md;
    if (!genai::comfy::metadata_json(c.file, "config", md, nullptr)) {
      continue;
    }
    if (!md.is_object() || !md.as_object().contains("transformer")) {
      continue;
    }
    const FlexData t = md.as_object().at("transformer");
    if (!t.is_object()) { continue; }
    const std::string im = str_field_(t, "image_model");
    const std::string name_lc = lower_(fs::path(c.file).filename().string());
    std::string one;
    if (im == "minimax_h3") {
      // The repack embeds no partition, so the FILENAME is the only
      // signal -- the weights themselves are indistinguishable. A
      // `pruned` DiT is a different model rather than a different
      // packing of this one, so it stays untagged.
      if (has_(name_lc, "pruned"))      { return {}; }
      else if (has_(name_lc, "fl2va"))  { one = "minimax-h3-fl2va"; }
      else if (has_(name_lc, "ref2va")) { one = "minimax-h3-ref2va"; }
      else                              { return {}; }
    }
    if (one.empty()) { continue; }
    if (seen && one != tag) { return {}; }   // two models, one directory
    tag  = one;
    seen = true;
  }
  return tag;
}

// What a Comfy-Org repo actually HOLDS, for the record's `variant`.
// A repack is published incrementally and pinned by a `files` whitelist,
// so a checkout is routinely a SUBSET -- and which subset decides what
// runs. Recording it is what turns "the pipeline failed" into "this copy
// has no text encoder", without a second walk of the tree.
std::string
comfy_components_(const fs::path& root)
{
  std::vector<std::string> roles;
  for (const genai::comfy::Component& c :
       genai::comfy::scan_repo(root.string())) {
    // Roles, not filenames: two VAEs under one role is the interesting
    // fact, not their names.
    std::string r = c.role;
    if (r == "diffusion_models") { r = "dit"; }
    else if (r == "text_encoders") { r = "text_encoder"; }
    bool seen = false;
    for (const std::string& s : roles) {
      if (s == r) { seen = true; break; }
    }
    if (!seen) { roles.push_back(r); }
  }
  if (roles.empty()) { return {}; }
  std::string out = "ComfyUI single-file (";
  for (std::size_t i = 0; i < roles.size(); ++i) {
    out += (i != 0 ? " + " : "") + roles[i];
  }
  return out + ")";
}

// A BARE DiT component directory (diffusers weights + config, no
// pipeline around it -- what model-quantize's DiT-only output is) gets
// the "<family>-dit" tag that generate-image's `dit_dir` picker filters
// on, NOT the pipeline tag: it is not loadable as a pipeline, and
// claiming otherwise would offer it to hf_dir where it would fail.
std::string
dit_component_tag_(const std::string& cls)
{
  if (cls == "Flux2Transformer2DModel")      { return "flux2-dit"; }
  if (cls == "Krea2Transformer2DModel")      { return "krea2-dit"; }
  if (cls == "QwenImageTransformer2DModel")  { return "qwen-image-edit-dit"; }
  if (cls == "QwenImage21Transformer2DModel") { return "qwen-image-21-dit"; }
  if (cls == "ZImageTransformer2DModel")     { return "z-image-dit"; }
  if (cls == "MageFlow")                     { return "mage-flow-dit"; }
  if (cls == "BooguImageTransformer2DModel") { return "boogu-image-dit"; }
  // Either Wan expert on its own (a quantized transformer/ or
  // transformer_2/ output) is one DiT component; which noise band it
  // covers is the pipeline's business, not the picker's.
  if (cls == "WanTransformer3DModel")        { return "wan-dit"; }
  // A bare H3 DiT carries no model_index.json, so which partition it
  // came from is not knowable here -- and does not matter to a picker.
  if (cls == "MiniMaxH3DiTModel")            { return "minimax-h3-dit"; }
  return {};
}

// Quantization / precision label for the record's `variant`.
std::string
variant_(const FlexData& cfg, const std::string& name_lc,
         const std::string& gguf_file)
{
  if (!gguf_file.empty()) {
    // "...-Q4_K_M.gguf" -> "GGUF Q4_K_M".
    const std::string base = fs::path(gguf_file).stem().string();
    const size_t dash = base.rfind('-');
    const std::string q = (dash == std::string::npos) ? std::string()
                                                      : base.substr(dash + 1);
    return q.empty() ? "GGUF" : ("GGUF " + upper_(q));
  }
  std::string bits;
  if (cfg.is_object()) {
    auto o = cfg.as_object();
    if (o.contains("quantization")) {
      FlexData q = o.at("quantization");
      if (q.is_object() && q.as_object().contains("bits")) {
        bits = std::to_string(q.as_object().at("bits").as_int(0));
      }
    }
  }
  const char* style = has_(name_lc, "optiq") ? "OptiQ "
                    : has_(name_lc, "mlx")   ? "MLX "
                                             : "";
  if (!bits.empty() && bits != "0") {
    return std::string(style) + bits + "-bit";
  }
  const std::string dtype = str_field_(cfg, "torch_dtype");
  if (dtype == "bfloat16") { return "bf16"; }
  if (dtype == "float16")  { return "f16"; }
  return std::string(style);
}

// Walk a directory once for the file count + byte total that the
// registry record reports (model-fetch records the same two facts from
// its download).
void
tree_size_(const fs::path& root, std::uint64_t& files, std::uint64_t& bytes)
{
  std::error_code ec;
  if (fs::is_regular_file(root, ec)) {
    files = 1;
    bytes = fs::file_size(root, ec);
    if (ec) { bytes = 0; }
    return;
  }
  fs::recursive_directory_iterator it(
      root, fs::directory_options::skip_permission_denied, ec);
  if (ec) { return; }
  for (const auto& e : it) {
    std::error_code fec;
    if (!e.is_regular_file(fec) || fec) { continue; }
    const auto sz = fs::file_size(e.path(), fec);
    if (!fec) { bytes += sz; }
    ++files;
  }
}

// True when the directory holds at least one file with `ext`.
bool
has_ext_(const fs::path& dir, const char* ext, std::string* found = nullptr)
{
  std::error_code ec;
  fs::directory_iterator it(dir, ec);
  if (ec) { return false; }
  for (const auto& e : it) {
    std::error_code fec;
    if (!e.is_regular_file(fec) || fec) { continue; }
    if (lower_(e.path().extension().string()) == ext) {
      if (found) { *found = e.path().filename().string(); }
      return true;
    }
  }
  return false;
}

// Copy a catalogue entry's curated metadata into the detection result.
void
from_catalog_(const ModelCatalogEntry& e, DetectedModel& d)
{
  d.model_type         = e.model_type;
  d.family             = e.family;
  d.version            = e.version;
  d.param_class        = e.param_class;
  d.variant            = e.variant;
  d.category           = catalog_category(e);
  d.parent_model_type  = e.parent_model_type;
  d.parent_param_class = e.parent_param_class;
  d.inputs             = e.inputs;
  d.outputs            = e.outputs;
  d.weight_format      = e.weight_format;
  if (d.inputs.empty() && d.outputs.empty()) {
    catalog_default_io(e.model_type, d.inputs, d.outputs);
  }
  d.files              = e.files;
  d.detected_by        = "catalog";
}

// WHICH catalogue entry a directory holds, when its repo publishes more
// than one.
//
// `catalog_by_path` answers with the first, which is right only for a
// repo with a single model -- and several here are not. MiniMax-H3 ships
// FL2VA and Ref2VA from one repo in BOTH packings, and lightx2v
// publishes eleven Turbo adapters from one repo whose parents differ
// (FL2VA and Ref2VA side by side), which is the field the LoRA pickers
// filter on. Taking the first offered a Ref2VA adapter to an FL2VA
// graph and registered a Ref2VA checkpoint as FL2VA -- the second being
// the worse of the two, since the partitions are byte-identical apart
// from the manifest and the wrong one loads, runs and conditions on
// nothing.
//
// The entries differ in the FILES they pin, and those files are on
// disk, so that is what decides. The rule is the unique strict maximum
// of files present, not "all of them": a hand-copied checkout that is
// missing a tokenizer is still unambiguously that model, while an exact
// tie -- both partitions in one directory, which is what Comfy-Org's
// repo actually holds -- has no right answer and gets none.
//
// Entries pinning nothing cannot take part: an empty list is present in
// every directory, so counting it as a match would make the unpinned
// entry win everywhere.
const ModelCatalogEntry*
catalog_pick_present_(const std::vector<const ModelCatalogEntry*>& cands,
                      const fs::path& root)
{
  const ModelCatalogEntry* best = nullptr;
  std::size_t best_n = 0;
  bool tied = false;
  for (const ModelCatalogEntry* e : cands) {
    if (e == nullptr || e->files.empty()) { continue; }
    std::size_t n = 0;
    std::error_code ec;
    for (const std::string& f : e->files) {
      if (fs::exists(root / f, ec) && !ec) { ++n; }
    }
    if (n == 0) { continue; }
    if (n > best_n) { best = e; best_n = n; tied = false; }
    else if (n == best_n) { tied = true; }
  }
  return tied ? nullptr : best;
}

// Whether the ABSENCE of a vision_config / audio_config in this family's
// config.json is evidence that the checkpoint drops that modality.
//
// It is evidence only for the families that signal their towers exactly
// that way at the config ROOT (Qwen3.5/3.6: vision_config; Gemma-4: both).
// Qwen3-ASR, for one, wraps everything under thinker_config and carries no
// audio_config at all -- reading its absence as "no audio" would turn a
// transcription model into a text-to-text one. Anything not listed here
// keeps its family defaults.
bool
io_trim_is_meaningful_(const std::string& mt)
{
  return mt == "qwen3.5" || mt == "qwen3.6" || mt == "gemma4"
         || mt == "gemma4_unified";
}

// Drop modalities the checkpoint plainly does not carry. The per-type
// defaults describe a FAMILY at its fullest (a Qwen3.5 repo with a vision
// tower takes images); a text-only dump of the same family does not, and
// claiming otherwise would offer it to a visual-qa field it cannot serve.
void
trim_io_to_config_(const FlexData& cfg, std::vector<std::string>& inputs)
{
  if (!cfg.is_object() || inputs.empty()) { return; }
  const bool vision = has_field_(cfg, "vision_config");
  const bool audio  = has_field_(cfg, "audio_config");
  std::vector<std::string> kept;
  for (const auto& m : inputs) {
    if ((m == "image" || m == "video") && !vision) { continue; }
    if (m == "audio" && !audio) { continue; }
    kept.push_back(m);
  }
  // A model that reads nothing would be nonsense: keep text as the floor.
  if (kept.empty()) { kept.push_back("text"); }
  inputs = kept;
}

}  // namespace

void
record_detected_fields(FlexData::ObjectView& rec, const DetectedModel& d)
{
  auto put = [&](const char* k, const std::string& v) {
    if (!v.empty()) { rec.insert_or_assign(k, FlexData::make_string(v)); }
  };
  put("model_type", d.model_type);
  put("family", d.family);
  put("version", d.version);
  put("param_class", d.param_class);
  put("variant", d.variant);
  put("category", d.category);
  put("parent_model_type", d.parent_model_type);
  put("parent_param_class", d.parent_param_class);
  put("detected_by", d.detected_by);
  put("weight_format", d.weight_format);
  auto put_list = [&](const char* k, const std::vector<std::string>& v) {
    if (v.empty()) { return; }
    FlexData arr = FlexData::make_array();
    {
      auto a = arr.as_array();
      for (const auto& s : v) { a.push_back(FlexData::make_string(s)); }
    }
    rec.insert_or_assign(k, std::move(arr));
  };
  put_list("inputs", d.inputs);
  put_list("outputs", d.outputs);
}

std::string
coreml_artifact(const std::string& path)
{
  if (path.empty()) { return {}; }
  auto is_bundle = [](const fs::path& p) {
    const std::string e = lower_(p.extension().string());
    return e == ".mlpackage" || e == ".mlmodelc";
  };
  const fs::path root(path);
  if (is_bundle(root)) { return path; }

  std::error_code ec;
  if (!fs::is_directory(root, ec) || ec) { return {}; }
  // .mlpackage is the shipped form, so it wins outright; a .mlmodelc is
  // remembered and used only if no package turns up, which keeps the
  // answer independent of directory iteration order.
  fs::path compiled;
  for (const auto& e : fs::directory_iterator(root, ec)) {
    if (ec) { break; }
    const fs::path& p = e.path();
    const std::string x = lower_(p.extension().string());
    if (x == ".mlpackage") { return p.string(); }
    if (x == ".mlmodelc" && compiled.empty()) { compiled = p; }
  }
  return compiled.empty() ? std::string() : compiled.string();
}

std::string
hf_path_from_local(const std::string& dir)
{
  fs::path p(dir);
  if (p.filename().empty()) { p = p.parent_path(); }   // trailing slash
  const std::string repo = p.filename().string();
  const std::string owner = p.parent_path().filename().string();
  if (repo.empty() || owner.empty()) { return {}; }
  return owner + "/" + repo;
}

DetectedModel
detect_model_dir(const std::string& dir, const std::string& hf_path_hint)
{
  DetectedModel d;
  std::error_code ec;
  const fs::path root(dir);
  if (!fs::exists(root, ec) || ec) {
    return d;
  }
  d.is_dir = fs::is_directory(root, ec) && !ec;
  tree_size_(root, d.file_count, d.total_bytes);

  const std::string base    = root.filename().string();
  const std::string base_lc = lower_(base);
  const std::string stem    = fs::path(base).stem().string();
  const std::string hf_path =
      hf_path_hint.empty() ? hf_path_from_local(dir) : hf_path_hint;

  // ---- 1. the catalogue, by name then by repo path -------------------
  // A catalogue hit is curated metadata -- always better than probing.
  // `name` first: the vpipe-supplement CoreML models share one hf_path
  // and are told apart only by name (which is their directory /
  // .mlpackage stem on disk).
  if (const ModelCatalogEntry* e = catalog_by_name(stem)) {
    from_catalog_(*e, d);
    return d;
  }
  if (!hf_path.empty()) {
    // One repo can publish several models, and then the path alone does
    // not name one -- what is actually on disk does. See
    // catalog_pick_present_.
    const std::vector<const ModelCatalogEntry*> cands =
        catalog_all_by_path(hf_path);
    const ModelCatalogEntry* pick =
        cands.size() == 1 ? cands.front()
                          : catalog_pick_present_(cands, root);
    if (pick != nullptr) {
      from_catalog_(*pick, d);
      return d;
    }
    // More than one and no way to tell: fall through and probe. The
    // probes below read the checkpoint rather than the path, and one of
    // them may still know (a Comfy repack names its partition in the
    // filename); if none does, the type stays empty, which is the right
    // answer for a directory holding two models.
  }

  // ---- 2. a diffusers pipeline (transformer/ + vae/) ------------------
  const FlexData dit_cfg = read_json_(root / "transformer" / "config.json");
  if (dit_cfg.is_object()) {
    // as_object() returns a VIEW, so the manifest has to outlive the
    // lookup -- bind it to a local rather than chaining off the call.
    const FlexData  mi = read_json_(root / "model_index.json");
    std::string     partition;
    if (mi.is_object()) {
      auto o = mi.as_object();
      if (o.contains("_minimax_h3")) {
        partition = str_field_(o.at("_minimax_h3"), "partition");
      }
    }
    d.model_type = dit_tag_(str_field_(dit_cfg, "_class_name"), base_lc,
                            dit_cfg, partition);
    if (!d.model_type.empty()) {
      d.detected_by = "diffusers";
    }
  }

  // ---- 2b. a Comfy-Org repack (no transformer/config.json) ------------
  if (d.model_type.empty()) {
    d.model_type = comfy_repo_tag_(root);
    if (!d.model_type.empty()) {
      d.detected_by   = "comfyui";
      d.weight_format = "comfyui";
      d.variant       = comfy_components_(root);
      catalog_default_io(d.model_type, d.inputs, d.outputs);
    }
  }

  // ---- 2b'. a QUANTIZED repack -----------------------------------
  // model-quantize's output is repack-SHAPED (diffusion_models/ + vae/ +
  // text_encoders/) but every component is a DIRECTORY of shards, so the
  // scan above -- which looks for one .safetensors per role -- finds
  // nothing, and the diffusers probe wants `transformer/config.json`
  // that a repack never had. Both quantized H3 builds therefore
  // registered as "type unknown" and no picker offered them.
  //
  // The partition comes from the config the producer wrote, for the
  // reason it writes `qkv_per_head` there: the filename that carried it
  // is gone, and the two partitions are byte-identical in every other
  // respect.
  if (d.model_type.empty()) {
    const FlexData q = read_json_(root / "diffusion_models" / "config.json");
    if (q.is_object()) {
      std::string part;
      auto qo = q.as_object();
      if (qo.contains(genai::MetalMiniMaxH3Transformer::kPartitionKey)) {
        const FlexData v =
            qo.at(genai::MetalMiniMaxH3Transformer::kPartitionKey);
        if (v.is_string()) { part = std::string(v.as_string()); }
      }
      d.model_type = dit_tag_(str_field_(q, "_class_name"), base_lc, q, part);
      if (!d.model_type.empty()) {
        d.detected_by   = "quantized-repack";
        d.weight_format = "safetensors";
        catalog_default_io(d.model_type, d.inputs, d.outputs);
      }
    }
  }

  // ---- 2c. a COMPONENT of an H3 pipeline ------------------------------
  // The text encoder is a stock Qwen3-VL, so its own config.json says
  // "qwen3_vl" and nothing more -- which is its architecture, not its
  // role. Both spellings of it (MiniMaxAI's text_encoder/ and a
  // quantized copy) therefore registered as "type unknown" and no picker
  // offered them. Two things can say otherwise:
  //   * `_vpipe_component`, stamped by model-quantize into the output of
  //     an encoder pass. A quantized sub-model has LEFT the pipeline
  //     that gave it meaning, so the producer is the last thing that
  //     knows what it was, and it writes it down.
  //   * the pipeline around it -- a `text_encoder/` whose parent carries
  //     model_index.json's `_minimax_h3`. That covers the released
  //     checkpoint, which predates any stamp.
  if (d.model_type.empty()) {
    const FlexData c0 = read_json_(root / "config.json");
    const std::string comp = str_field_(c0, "_vpipe_component");
    if (comp == "minimax-h3-text-encoder") {
      d.model_type  = comp;
      d.detected_by = "component-tag";
    } else if (base == "text_encoder") {
      const FlexData mi2 = read_json_(root.parent_path() / "model_index.json");
      if (has_field_(mi2, "_minimax_h3")) {
        d.model_type  = "minimax-h3-text-encoder";
        d.detected_by = "pipeline";
      }
    }
    if (!d.model_type.empty()) {
      // A component, not a runnable model -- it conditions the DiT, it
      // does not generate anything on its own.
      d.category = "component";
      d.param_class = "32B";
    }
  }

  // ---- 3. a plain HF / MLX checkpoint (config.json at the root) -------
  // Also catches a bare DiT component: a diffusers config carries
  // `_class_name` where an HF one carries `model_type`.
  FlexData cfg;
  if (d.model_type.empty()) {
    cfg = read_json_(root / "config.json");
    if (cfg.is_object()) {
      d.param_class = param_class_from_name_(base);
      const std::string cls = str_field_(cfg, "_class_name");
      if (!cls.empty()) {
        d.model_type = dit_component_tag_(cls);
        if (!d.model_type.empty()) {
          // A component, not a runnable model: no I/O of its own.
          d.category    = "component";
          d.detected_by = "diffusers-component";
        }
      } else {
        d.model_type =
            lm_tag_(str_field_(cfg, "model_type"), base_lc, d.param_class);
        if (!d.model_type.empty()) {
          d.detected_by = "config";
        }
      }
    }
  }

  // ---- 4. a GGUF checkpoint -------------------------------------------
  std::string gguf;
  if (d.model_type.empty() && d.is_dir && has_ext_(root, ".gguf", &gguf)) {
    // The repo NAME carries the family + size ("Qwen3.5-4B-GGUF"); the
    // .gguf file only carries the quant label.
    d.param_class = param_class_from_name_(base);
    if (has_(base_lc, "qwen3.6") || has_(base_lc, "qwen3_6")) {
      d.model_type = "qwen3.6";
    } else if (has_(base_lc, "qwen3.5") || has_(base_lc, "qwen3_5")) {
      d.model_type = "qwen3.5";
    } else if (has_(base_lc, "gemma")) {
      d.model_type = (!d.param_class.empty() && d.param_class[0] == 'E')
                         ? "gemma4" : "gemma4_unified";
    }
    if (!d.model_type.empty()) { d.detected_by = "gguf"; }
  }

  // ---- 5. a CoreML package (a tower / detector / VAD supplement) ------
  // Only the SHAPE is knowable here -- which model it belongs to is not,
  // so this sets the category and leaves model_type to the catalogue
  // (step 1) or the user's explicit override.
  const std::string ext_lc = lower_(root.extension().string());
  if (d.model_type.empty()
      && (ext_lc == ".mlpackage" || ext_lc == ".mlmodelc")) {
    d.category    = "supplement";
    d.detected_by = "coreml";
  }
  // A bare LoRA (weights, no config) is likewise a supplement of a model
  // we cannot name from the file alone.
  if (d.model_type.empty() && d.category.empty() && has_(base_lc, "lora")) {
    d.category = "supplement";
  }

  if (d.param_class.empty()) {
    d.param_class = param_class_from_name_(base);
  }
  // `variant_` reads a config.json's quantization block plus the
  // directory name, neither of which a Comfy-Org repack has -- so it
  // would blank the component list step 2b already worked out. Only
  // fill in what is still empty.
  if (d.variant.empty()) { d.variant = variant_(cfg, base_lc, gguf); }
  if (d.category.empty()) {
    d.category = "model";
  }
  family_version_(d.model_type, d.family, d.version);

  // ---- 6. I/O modalities ----------------------------------------------
  // Derived from the runtime tag (the same table the catalogue uses, so a
  // registered model reads identically to a fetched one), then trimmed to
  // what this particular checkpoint actually carries.
  //
  // ONLY WHEN STILL EMPTY, exactly as the variant and category above.
  // catalog_default_io APPENDS, and the detection steps that already know
  // their tag -- the Comfy repack and the QUANTIZED repack -- call it
  // themselves, so running it again here appended the same list twice:
  // a quantized MiniMax-H3 registered as
  // inputs ["text","image","text","image"]. The compatibility filter uses
  // includes() and survived that, which is why it went unnoticed; the
  // badges in the model picker did not.
  if (!d.model_type.empty()) {
    if (d.inputs.empty() && d.outputs.empty()) {
      catalog_default_io(d.model_type, d.inputs, d.outputs);
    }
    if (io_trim_is_meaningful_(d.model_type)) {
      trim_io_to_config_(cfg, d.inputs);
    }
  }
  return d;
}

std::string
resolve_vae_dir(const std::string& root)
{
  if (fs::exists(fs::path(root) / "vae" / "config.json")) {
    return (fs::path(root) / "vae").string();
  }
  // MiniMax-H3's video VAE. This is the OUTER directory, not the
  // `source/` one holding the weights: the outer config.json is the
  // one that names the class and carries latents_mean / latents_std,
  // while `source/config.json` is the legacy geometry. The model
  // loader descends the last level itself.
  // Both partition subdirectories: MiniMaxAI ships FL2VA/ and Ref2VA/ as
  // complete pipelines, and detection over a Ref2VA-only checkout that
  // probed FL2VA alone found no VAE at all.
  // VOSR ships its autoencoder under the name of the model it was
  // extracted from rather than under `vae/`, so a graph pointing every
  // stage at one checkpoint root would otherwise find no VAE and open
  // the root itself.
  if (fs::exists(fs::path(root) / "Qwen-Image-vae-2d" / "config.json")) {
    return (fs::path(root) / "Qwen-Image-vae-2d").string();
  }
  for (const fs::path& p : {fs::path(root) / "video_vae",
                            fs::path(root) / "FL2VA" / "video_vae",
                            fs::path(root) / "Ref2VA" / "video_vae"}) {
    if (fs::exists(p / "source" / "model.safetensors") &&
        fs::exists(p / "config.json")) {
      return p.string();
    }
  }
  // FlashVSR as published: its Wan 2.1 VAE is ONE torch file at the root
  // with no config beside it. Returned by FILE, the way a Comfy-Org
  // repack's component is, so the root's other checkpoints -- a 5.7 GB
  // denoiser among them -- are never taken for it. The source
  // projection's file is part of the test, so a directory that merely
  // holds a stray VAE of that name is not this.
  {
    const fs::path vae = fs::path(root) / "Wan2.1_VAE.pth";
    if (fs::exists(vae) && fs::exists(fs::path(root) / "LQ_proj_in.ckpt")) {
      return vae.string();
    }
  }
  return root;
}

std::string
resolve_vae_weights_path(const std::string& vae_dir)
{
  std::error_code ec;
  if (vae_dir.empty() || !fs::is_directory(fs::path(vae_dir), ec) || ec) {
    return vae_dir;
  }
  const fs::path dir(vae_dir);
  // Every layout MetalLlamaWeights::open_model globs for itself. Probed
  // first so a diffusers or sharded checkpoint keeps taking the path it
  // always did, whatever else happens to sit beside it.
  for (const char* known : {"model.safetensors.index.json",
                            "diffusion_pytorch_model.safetensors.index.json",
                            "model.safetensors",
                            "diffusion_pytorch_model.safetensors"}) {
    if (fs::exists(dir / known, ec) && !ec) { return vae_dir; }
  }
  std::string only;
  std::size_t n = 0;
  std::error_code lec;
  for (const auto& de : fs::directory_iterator(dir, lec)) {
    const fs::path& p = de.path();
    if (p.extension() != ".safetensors") { continue; }
    const std::string fn = p.filename().string();
    // A numbered shard is the index-less sharded layout, which
    // open_model globs on its own.
    if (fn.rfind("model-", 0) == 0 ||
        fn.rfind("diffusion_pytorch_model-", 0) == 0) {
      return vae_dir;
    }
    ++n;
    if (n == 1) { only = p.string(); }
  }
  return (n == 1) ? only : vae_dir;
}

std::string
resolve_vae_config_path(const SessionContextIntf* session,
                        const std::string&        model_ref,
                        const std::string&        vae_dir,
                        std::string_view          expect_class)
{
  std::error_code ec;
  // 1. The checkpoint's own. Probed first and unconditionally, so a file
  //    someone put there beats anything inferred below -- including a
  //    parent that disagrees with it.
  const fs::path own = fs::path(vae_dir) / "config.json";
  if (fs::exists(own, ec) && !ec) { return own.string(); }
  if (session == nullptr || model_ref.empty()) { return {}; }

  // 2. An installed model this VAE attaches to. The catalogue is what
  //    says which those are -- `parent_model_type` on the supplement,
  //    matched against the model_type of entries that are not
  //    themselves supplements -- so this is not a search of the disk for
  //    something plausible; it is the link the catalogue already
  //    records, followed.
  const std::string mt = resolve_model(session, model_ref).model_type;
  if (mt.empty()) { return {}; }
  std::string parent;
  for (const ModelCatalogEntry& e : model_catalog()) {
    if (e.model_type == mt && !e.parent_model_type.empty()) {
      parent = e.parent_model_type;
      break;
    }
  }
  if (parent.empty()) { return {}; }

  for (const ModelCatalogEntry& e : model_catalog()) {
    // A supplement is not a parent: a LoRA also carries model_type
    // "krea2-lora" and parent "krea2", and it has no vae/ at all.
    if (e.model_type != parent || !e.parent_model_type.empty()) { continue; }
    const ResolvedModel rm = resolve_model(session, e.hf_path);
    if (!rm.from_registry) { continue; }        // not installed here
    const fs::path p = fs::path(rm.dir) / "vae" / "config.json";
    if (!fs::exists(p, ec) || ec) { continue; }
    if (!expect_class.empty()) {
      // SAME FAMILY IS NOT SAME LATENT SPACE. Wan and Qwen-Image share
      // this VAE's architecture and shapes and have entirely different
      // latents_mean / latents_std, so a parent whose config names
      // another class is skipped rather than borrowed -- that is the one
      // way this fallback could produce a confident wrong answer.
      const FlexData cfg = read_json_(p);
      if (!cfg.is_object() ||
          str_field_(cfg, "_class_name") != expect_class) {
        continue;
      }
    }
    return p.string();
  }
  return {};
}

}

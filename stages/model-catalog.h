#ifndef STAGES_MODEL_CATALOG_H
#define STAGES_MODEL_CATALOG_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace vpipe {

class FlexData;

// One downloadable model in the internal HuggingFace catalogue.
//
// TO ADD A MODEL: append one designated-initialiser entry to kCatalog in
// model-catalog.cc -- that table is the single edit point. The four
// selection fields (family / version / param_class / variant) drive the
// interactive drill-down menu, so keep them short + human-readable.
// `hf_path` is the repo path AFTER "huggingface.co/" and is used BOTH as
// the download source and as the LMDB registration key, so it must be
// exact. `files` pins a SUBSET of repo files to fetch -- used for GGUF
// repos that ship many quantization points (and optional companions like
// an mmproj multimodal projector or an imatrix file) in one repo, so the
// catalogue points at just the wanted .gguf(s) instead of pulling every
// quant; empty means fetch the whole repo. `needs_tokenizer_json`
// requests the post-download tokenizer.json synthesis step (the Qwen3-ASR
// repos ship vocab.json + merges.txt but no consolidated tokenizer.json,
// which our runtime needs).
//
// `name` overrides the LMDB registration key (and the on-disk extraction
// subdir) -- needed when several entries share ONE hf_path repo (the
// vpipe-supplement repo ships one CoreML model per .tar), since the default
// key (hf_path) would collide. `extract_archive` post-processes each fetched
// `.tar` in `files`: it is unpacked into `<repo>/<name>/` and the registered
// local_path points at the single top-level `*.mlpackage` inside (the
// vpipe-supplement archive layout) rather than at the repo dir.
struct ModelCatalogEntry {
  std::string family;       // "Qwen", "Gemma"
  std::string version;      // "3.5", "3.6", "4", "3-ASR"
  std::string param_class;  // "27B", "9B", "4B", "E4B", "12B", "1.7B"
  std::string variant;      // label distinguishing publisher / quant
  std::string hf_path;      // "mlx-community/Qwen3.5-9B-MLX-4bit"
  std::string model_type;   // runtime hint: "qwen3.5"/"gemma4"/...
  // Supported input / output modalities, each a subset of
  // {"text","image","audio","video"}. Descriptive metadata surfaced in the
  // web-ui model browser (e.g. gemma-4-e4b in {text,image,audio,video} out
  // {text}; flux2 in {text,image} out {image}). Left empty for pure
  // components/datasets where a modality doesn't apply.
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  // Parent linkage for SUPPLEMENTS (vision towers, LoRAs). A non-empty
  // `parent_model_type` marks this entry as an attachment to a parent
  // model of that model_type (e.g. a vision tower -> "qwen3.5"; a LoRA ->
  // "krea2"). `parent_param_class` optionally pins the parent size ("4B",
  // "E4B"); empty means any size within the parent family. Used to match
  // a supplement to the parent model chosen in a sibling stage field.
  std::string parent_model_type;
  std::string parent_param_class;
  std::vector<std::string> files;  // repo files to fetch (one or more GGUF
                                   // quant(s) + mmproj/imatrix companions);
                                   // empty = fetch the whole repo
  // Small files pulled from ANOTHER repo into this entry's directory.
  //
  // This exists because a Comfy-Org repack is weights-ONLY: no tokenizer,
  // no configs. Pinning its own files alone yields a directory holding a
  // DiT and both VAEs that still cannot encode a prompt -- a fetch that
  // reports success and produces something unusable. The alternative, a
  // 144 GB sibling entry fetched for 11 MB of tokenizer, is not one.
  //
  // Kept SMALL on purpose. This is for the few MB that complete a repo,
  // not a way to assemble a model out of several: anything large enough
  // to be worth choosing belongs in its own entry, where the drill-down
  // menu can show it.
  struct Companion {
    std::string repo;  // an hf_path, e.g. "MiniMaxAI/MiniMax-H3"
    std::string file;  // path within `repo`
    std::string dest;  // where it lands under this entry's local dir
  };
  std::vector<Companion> companion_files;
  // On-disk weight format, when it is NOT the upstream diffusers/HF
  // layout. "comfyui" marks a Comfy-Org repack (the repos carrying the
  // `comfyui` HuggingFace tag: Comfy-Org/MiniMax-H3,
  // Comfy-Org/Wan-Animate-2, ...): one freely-named .safetensors per
  // component under diffusion_models/ | text_encoders/ | vae/, with each
  // component's config inside the file's safetensors `__metadata__`
  // instead of a config.json.
  //
  // This is recorded rather than derived because it can change how the
  // WEIGHTS are read, not just where the config comes from: Comfy-Org's
  // MiniMax-H3 conversion reorders the DiT's fused qkv projection under
  // the same tensor name and shape, so a loader that guesses wrong loads
  // cleanly and computes nonsense. Loaders still confirm it from the
  // file's own metadata -- this field is what lets the catalogue and the
  // registry record SAY which repack a directory is, for a picker or a
  // user who is choosing between two copies of one model.
  std::string weight_format;
  bool        needs_tokenizer_json = false;
  std::string name;         // registration key + extract subdir (when several
                            // entries share one hf_path); empty -> key=hf_path
  bool        extract_archive = false;  // unpack fetched .tar(s); local_path
                                        // -> the contained *.mlpackage
  // Dataset fetch (evaluation datasets, e.g. WikiText-2 / ARC-Challenge):
  // each pair is {full URL, destination filename}, downloaded VERBATIM into
  // the registered dir (the HF datasets-server /rows pages) instead of walking
  // a model repo's file tree. Keeps dataset text OUT of the binary (fetched on
  // demand by the user) so vpipe's Apache-2.0 license is unaffected. When
  // non-empty, the repo-tree download path is skipped.
  std::vector<std::pair<std::string, std::string>> dataset_files;
};

// The full catalogue: the built-in table followed by whatever plugins
// contributed, in registration order. Definition order is the display
// order within each drill-down group, so a plugin's models appear after
// the built-ins rather than interleaved with them.
const std::vector<ModelCatalogEntry>& model_catalog();

// Contribute catalogue entries from a PLUGIN.
//
// The built-in table in model-catalog.cc stays the single edit point for
// models that ship with vpipe; this is the seam for a model that does
// not, so a separately-licensed family can be downloadable, registrable
// and browsable without the entry living in this tree. Returns how many
// entries were taken.
//
// FIRST-WINS, per (name-or-hf_path, files) identity: an entry that would
// duplicate one already in the catalogue is dropped and NOT counted, so
// two plugins shipping the same repo cannot produce two menu rows for
// one model.
//
// LIFETIME. `model_catalog()` hands out a reference, and `catalog_by_*`
// hand out pointers to its elements, so registration must never move
// what a caller is already holding. It does not: each registration
// publishes a NEW snapshot and the previous ones are retained for the
// life of the process. Snapshots are a few hundred entries, and this is
// called a handful of times at plugin load, so the retained copies cost
// nothing measurable and no reference ever dangles.
//
// Call it from a plugin's vpipe_plugin_register. Plugins load before any
// pipeline is built, so entries are present before anything reads the
// catalogue -- but a later call is still safe, it just will not appear
// in a menu already rendered.
std::size_t register_catalog_entries(std::vector<ModelCatalogEntry> entries);

// Drill-down helpers: each returns the distinct values present at that
// level, in first-seen (catalogue) order, filtered by the levels already
// chosen above it.
std::vector<std::string> catalog_families();
std::vector<std::string> catalog_versions(const std::string& family);
std::vector<std::string>
catalog_param_classes(const std::string& family, const std::string& version);
std::vector<std::string>
catalog_variants(const std::string& family, const std::string& version,
                 const std::string& param_class);

// Resolve a fully-specified selection to its entry, or nullptr.
const ModelCatalogEntry*
catalog_find(const std::string& family, const std::string& version,
             const std::string& param_class, const std::string& variant);

// Look up a catalogue entry by exact hf_path (nullptr when not
// catalogued -- a user-typed path is still downloadable, just without
// the curated metadata).
const ModelCatalogEntry* catalog_by_path(const std::string& hf_path);

// EVERY catalogue entry published from `hf_path`, in catalogue order.
//
// One repo can hold several distinct models -- MiniMax-H3 ships its
// FL2VA and Ref2VA partitions from one Comfy-Org repo, and the
// vpipe-supplement repo holds six CoreML archives -- and the entries
// differ in which FILES they pin. `catalog_by_path` answers with the
// first, which is the right answer only when there is one; a caller
// that must not silently take the wrong partition asks for all of them
// and disambiguates.
std::vector<const ModelCatalogEntry*>
catalog_all_by_path(const std::string& hf_path);

// Look up a catalogue entry by its `name` (the registration key used when
// several entries share ONE hf_path -- the vpipe-supplement CoreML models).
// catalog_by_path can't disambiguate those (they share hf_path), so the
// registry-record enrichment matches by `name` first. nullptr on miss.
const ModelCatalogEntry* catalog_by_name(const std::string& name);

// Derived category of an entry: "dataset" (carries dataset_files),
// "supplement" (has a parent_model_type -- a tower / LoRA), else "model".
std::string catalog_category(const ModelCatalogEntry& e);

// Default input / output modalities for a runtime `model_type`, each a
// subset of {"text","image","audio","video"}. This is the table the
// catalogue falls back to for entries that don't record I/O explicitly,
// exposed so a model registered from disk (model-register, which detects
// its model_type) derives the SAME modalities as its catalogued
// siblings. Appends to `in` / `out`; an unknown type appends nothing.
void
catalog_default_io(const std::string& model_type,
                   std::vector<std::string>& in,
                   std::vector<std::string>& out);

// Serialize an entry's curated metadata (selection fields + model_type,
// category, inputs, outputs, parent linkage) to a FlexData object, for the
// web-ui model browser. Fetch-only fields (files, dataset URLs) are omitted.
FlexData catalog_entry_to_flex(const ModelCatalogEntry& e);

// Normalise a user-typed model reference to a bare "owner/repo" path:
//   "https://huggingface.co/owner/repo"     -> "owner/repo"
//   "huggingface.co/owner/repo/tree/main"   -> "owner/repo"
//   "owner/repo/"                           -> "owner/repo"
// Query/fragment and any path beyond the first two segments are dropped.
// Returns "" when the input has no owner/repo pair.
std::string normalize_hf_path(const std::string& input);

// One file listed by the HuggingFace tree API.
struct HfFile {
  std::string   path;       // repo-relative, e.g. "model.safetensors"
  std::uint64_t size = 0;   // bytes (0 when the API omits it)
  // What the repo publishes to check the downloaded bytes against.
  // HuggingFace stores anything big in LFS and names the object by the
  // SHA-256 of its CONTENT (`lfs.oid`); everything small enough to live
  // in git itself is named by the git blob id, a SHA-1 over
  // "blob <size>\0" followed by the content (`oid`). So every file
  // carries exactly one of these, and the API publishes no MD5 for any
  // of them.
  std::string   sha256;     // lfs.oid; empty when the file is not LFS
  std::string   git_oid;    // oid; empty when the API omits it
  // `xetHash`: the id the content-addressed store knows the file by, so
  // it can be rebuilt from deduplicated chunks instead of streamed
  // whole. Empty for a file the store does not hold.
  std::string   xet_hash;
};

// Extract the downloadable files (type == "file") from a HuggingFace
// "/api/models/<repo>/tree/<rev>" JSON array. Directories and malformed
// entries are skipped. Pure -- the caller fetches the JSON.
std::vector<HfFile> hf_tree_files(const FlexData& tree_json);

}

#endif

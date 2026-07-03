#ifndef STAGES_MODEL_CATALOG_H
#define STAGES_MODEL_CATALOG_H

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
  std::string param_class;  // "27B", "9B", "4B", "e4b", "12B", "1.7B"
  std::string variant;      // label distinguishing publisher / quant
  std::string hf_path;      // "mlx-community/Qwen3.5-9B-MLX-4bit"
  std::string model_type;   // runtime hint: "qwen3.5"/"gemma4"/...
  std::vector<std::string> files;  // repo files to fetch (one or more GGUF
                                   // quant(s) + mmproj/imatrix companions);
                                   // empty = fetch the whole repo
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

// The full catalogue. Definition order is the display order within each
// drill-down group.
const std::vector<ModelCatalogEntry>& model_catalog();

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
};

// Extract the downloadable files (type == "file") from a HuggingFace
// "/api/models/<repo>/tree/<rev>" JSON array. Directories and malformed
// entries are skipped. Pure -- the caller fetches the JSON.
std::vector<HfFile> hf_tree_files(const FlexData& tree_json);

}

#endif

#ifndef STAGES_MODEL_DETECT_H
#define STAGES_MODEL_DETECT_H

#include "common/flex-data.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vpipe {

// What a model directory on disk turns out to BE. Produced by
// detect_model_dir() for the model-register stage, so registering a
// hand-downloaded / hand-built model needs a path and nothing else: the
// runtime hint (`model_type`) and the I/O modalities that the stage
// pickers and the web-ui model browser filter on are derived from the
// directory rather than typed by the user.
//
// Every field is best-effort and may be empty; `model_type` is the one
// that matters (the stages filter on it), and it is either exact (a
// catalogue hit or a recognized config.json) or empty -- never guessed
// into a wrong family, because a wrong runtime hint offers a model to a
// stage that cannot load it.
struct DetectedModel {
  std::string model_type;    // runtime hint: "qwen3.5" / "flux2" / ...
  std::string family;        // display: "Qwen", "Gemma", "FLUX.2"
  std::string version;       // display: "3.5", "4"
  std::string param_class;   // "4B", "E4B", "35B-A3B" (from the dir name)
  std::string variant;       // "MLX 4-bit", "GGUF Q4_K", "bf16"
  std::string category;      // "model" | "supplement" | "dataset"
  std::vector<std::string> inputs;   // subset of text/image/audio/video
  std::vector<std::string> outputs;
  std::string parent_model_type;     // supplements: the model they attach to
  std::string parent_param_class;
  // How model_type was established, for the log line + summary beat:
  //   "catalog"    -- the directory matches a catalogued repo (curated)
  //   "config"     -- HF config.json model_type
  //   "diffusers"  -- transformer/config.json _class_name (DiT pipeline)
  //   "gguf"       -- a GGUF checkpoint's metadata/name
  //   "coreml"     -- an .mlpackage / .mlmodelc supplement
  //   ""           -- nothing recognizable
  std::string detected_by;
  // Layout facts, reported in the record so a later reader doesn't have
  // to walk the tree again.
  std::uint64_t file_count  = 0;
  std::uint64_t total_bytes = 0;
  bool          is_dir      = false;   // false for a bare .mlpackage/.gguf
};

// Inspect a model directory. `dir` is a filesystem path (a directory, or
// a bare .mlpackage / .gguf file). `hf_path_hint` is the "owner/repo"
// the caller believes this came from (empty -> derived from the two
// trailing path components); a catalogue hit on it supplies the curated
// metadata verbatim, which is always preferred over probing.
//
// Never throws: an unreadable / unrecognized directory yields a
// DetectedModel with an empty model_type and empty I/O, which registers
// fine -- it just carries no compatibility hint.
DetectedModel
detect_model_dir(const std::string& dir, const std::string& hf_path_hint = {});

// Write a detection's descriptive fields into a registry record object:
// model_type / family / version / param_class / variant / category /
// parent linkage / detected_by, plus the inputs + outputs arrays. Only
// non-empty fields are written, so a caller may fill in curated values
// before or after without having to null out what it doesn't know.
//
// Shared by every stage that registers a model (model-register,
// model-fetch, model-quantize, lora-fuse) so a record means the same
// thing however the model got there.
void
record_detected_fields(FlexData::ObjectView& rec, const DetectedModel& d);

// The loadable CoreML artifact for `path`.
//
// CoreML loads a *.mlpackage / *.mlmodelc BUNDLE, never the directory
// that happens to contain one, so a registry record a CoreML stage can
// use has to name the bundle itself. The vpipe-supplement archives
// unpack to exactly one bundle under a per-model directory, and this is
// the one place that layout is turned into a loadable path -- every
// CoreML consumer (realtime-vqa, visual-qa, yolo-detection,
// audio-tagging, audio-segment) just resolves its registry key and
// trusts what comes back.
//
// Returns `path` unchanged when it is already a bundle; the single
// top-level bundle directly under `path` when there is one; and an
// EMPTY string otherwise. Empty is the normal answer for a plain
// checkpoint directory -- a safetensors loader consumes the directory,
// so there is nothing to resolve -- which is why callers treat it as
// "keep what you had" rather than as an error.
//
// .mlpackage wins over .mlmodelc if a directory somehow holds both; the
// search is the immediate level only, matching how the archives unpack.
std::string coreml_artifact(const std::string& path);

// Best-effort "owner/repo" for a local path, from its two trailing
// components ("/models/mlx-community/gemma-4-e4b-it-4bit" ->
// "mlx-community/gemma-4-e4b-it-4bit"). Empty when the path has fewer
// than two components. Used both as the default registry key and as the
// catalogue lookup for a directory laid out the way model-fetch writes.
std::string hf_path_from_local(const std::string& dir);

}

#endif

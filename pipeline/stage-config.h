#ifndef STAGE_CONFIG_H
#define STAGE_CONFIG_H

#include "common/flex-data.h"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace vpipe {

// Declared type of a configuration value. Mirrors the FlexData kinds a
// stage actually consumes (Null is never a declared type). `Any` marks
// a key whose accepted kind varies by context. `Text` is a String in
// every backend respect (same FlexData kind, same def_str default,
// read via attr_str); it differs only as a UI hint -- the value is a
// multi-line string, so an editor renders a multi-line text box rather
// than a single-line input.
enum class ConfigType : unsigned char {
  Bool, Int, Uint, Real, String, Text, Array, Object, Any
};

// Stable lower-case name for a ConfigType ("bool", "int", "uint",
// "real", "string", "text", "array", "object", "any"). Never returns
// null.
std::string_view config_type_name(ConfigType) noexcept;

// Static, type-level declaration of one configuration key. A stage
// exposes its schema by returning a span over a file-static
// `constexpr ConfigKey[]` table from Stage::config_spec(). The struct
// is an aggregate so tables can be written with designated
// initializers:
//
//   constexpr ConfigKey kKeys[] = {
//     {.key = "top_n",   .type = ConfigType::Int,
//      .doc = "0 = draw all", .def_int = 10},
//     {.key = "hf_dir",  .type = ConfigType::String, .required = true,
//      .doc = "HuggingFace model directory"},
//   };
//
// Only the `def_*` field matching `type` is consulted, and only when
// `required` is false and the type is scalar. Composite types
// (Array/Object) default to an empty container; required keys default
// to Null. All string_view members must point at static storage
// (string literals) -- the table outlives every stage instance.
struct ConfigKey {
  std::string_view key;
  ConfigType       type     = ConfigType::Any;
  bool             required  = false;
  std::string_view doc      = {};

  bool             def_bool = false;
  std::int64_t     def_int  = 0;
  std::uint64_t    def_uint = 0;
  double           def_real = 0.0;
  std::string_view def_str  = {};

  // Optional UI hint: when set, this field's value is (or may be) a key
  // in the named LMDB sub-db, so an editor can SUGGEST the sub-db's keys
  // (the web-ui renders a datalist dropdown populated from
  // /api/db/keys). It is a hint, not a constraint -- free text is still
  // allowed (e.g. hf_dir also accepts a filesystem path). Only
  // meaningful for String keys.
  std::string_view suggest_db = {};

  // Optional companion to suggest_db: restrict the suggested keys to
  // model-registry records whose `model_type` field equals this (the
  // compatibility hint model-fetch writes, e.g. "yolo", "silero-vad",
  // "qwen3.5-vision-encoder"). Lets a CoreML-model field offer only the
  // type it can actually load. Ignored when suggest_db is unset.
  std::string_view suggest_db_type = {};

  // Optional: the name of a SHARED-MODEL CHANNEL this key belongs to.
  //
  // A channel is one model reference travelling from a source stage
  // (model-select) to every stage that latches it on a `model` iport.
  // Both ends name the channel here, and the difference between them is
  // whether they have a suggest_db_type of their own:
  //
  //   CONSUMER -- names the channel AND lists the model_types it can
  //     actually run. Its own picker keeps that narrower list, because
  //     generate-image cannot run a video family and vice versa.
  //   SOURCE -- names the channel and leaves suggest_db_type EMPTY.
  //     resolve_config_params() fills it with the UNION over every
  //     registered consumer of the channel.
  //
  // Derived rather than written down because the source cannot know the
  // consumers: a PLUGIN registers stages after this table is compiled,
  // and a hard-coded list at the source is a list that is wrong the
  // moment a plugin ships a family. It was: minimax-h3-fl2va ran a whole
  // text-to-video graph while being unpickable, and every out-of-tree
  // family was unpickable by construction.
  std::string_view model_channel = {};

  // Optional companions to suggest_db: required input / output modalities
  // (comma-separated subset of "text","image","audio","video"). The web-ui
  // model browser keeps only models whose catalogue I/O covers these -- a
  // model's inputs must include every need_inputs entry and its outputs
  // every need_outputs entry. E.g. a text-chat LM field sets
  // need_inputs="text", need_outputs="text". Ignored when suggest_db unset.
  std::string_view need_inputs  = {};
  std::string_view need_outputs = {};

  // Optional UI hint: this field is a filesystem path the stage reads or
  // writes (and which the web-ui filesystem sandbox confines -- see
  // path-sandbox.h). Lets an editor offer a "Browse..." file dialog.
  // Only meaningful for String keys (or an Array of path strings). NOTE:
  // model-manager paths (hf_dir, model_path, ...) are deliberately NOT
  // flagged here -- they are sandbox-exempt and get a model picker
  // instead of the sandbox browser.
  bool             is_path    = false;
  // When is_path: true, this path is WRITTEN (a save target) rather than
  // read; the browser opens in "save" mode (offers a filename field).
  bool             path_write = false;
  // When is_path: the kind of target -- "dir" selects a directory,
  // anything else (empty) selects a file. Static storage.
  std::string_view path_kind  = {};
  // When is_path: an optional filter category the browser maps to a set
  // of extensions ("image", "audio", "video", "text"); empty = no
  // filter (all files). Static storage.
  //
  // COMMA-SEPARATED for a field that takes more than one kind
  // ("image,video,audio"): the dialog offers the union first and the
  // individual categories after it. Used where the field cannot say
  // which kind a given file is -- video-ref-encoder's reference list
  // reads that from the file itself.
  std::string_view path_filter = {};
};

// Contribute model_types to a shared-model channel at RUN TIME.
//
// The ConfigKey side of a channel covers a plugin that brings its own
// stage: it declares what it runs and the union picks it up. It does
// NOT cover the other plugin shape -- a family that plugs into an
// EXISTING stage's own registry. LTX-2.5 registers a video family with
// generate-video and a VAE family with vae-decode, so those stages run
// it while their suggest_db_type, written when this tree was compiled,
// cannot name it.
//
// So the family registries call this, and a plugin that registers a
// family gets a working picker without knowing this exists. Idempotent
// per (channel, type); order of first appearance is preserved.
void register_channel_types(std::string_view channel,
                            std::string_view csv_model_types);

// Per-instance resolved descriptor: one ConfigKey paired with the
// value the stage instance was actually given. `default_value` is the
// schema default materialised as a FlexData; `current_value` is the
// value pulled from the instance's config tree, falling back to
// `default_value` when the key was not supplied. Stage-internal
// clamping / string->enum normalisation is NOT reflected here --
// `current_value` is the configured value resolved against the
// declared default, which is what a config round-trip needs.
struct ConfigParam {
  std::string key;
  ConfigType  type     = ConfigType::Any;
  bool        required = false;
  std::string doc;
  // Mirrors ConfigKey::suggest_db (empty when unset). Surfaced in the
  // config schema so the editor can offer a key-suggestion dropdown.
  std::string suggest_db;
  // Mirrors ConfigKey::suggest_db_type (empty when unset): filters the
  // suggestion dropdown to registry records of this model_type.
  std::string suggest_db_type;
  // Mirror ConfigKey::need_inputs / need_outputs (empty when unset):
  // required I/O modalities the model browser filters on.
  std::string need_inputs;
  std::string need_outputs;
  // Mirror ConfigKey's filesystem-path hints (see there). is_path drives
  // the editor's "Browse..." affordance; path_write / path_kind /
  // path_filter shape the file dialog.
  bool        is_path    = false;
  bool        path_write = false;
  std::string path_kind;
  std::string path_filter;
  FlexData    default_value;
  FlexData    current_value;
  // True when the instance's config tree actually contains this key
  // (vs. `current_value` having been filled from the schema default).
  // Lets clients distinguish "explicitly set" from "implicitly
  // defaulted", which matters for stages with mutually-exclusive
  // fields (chrono's frequency_hz vs period_*): the editor needs to
  // omit unset numeric fields from a write rather than sending the
  // default value, or both branches end up "present" at validation.
  bool        present  = false;
};

// Materialise a schema's default for one key as a FlexData (Null for
// required keys; empty container for Array/Object/Any).
FlexData config_default_value(const ConfigKey& key);

// Resolve a static spec against an instance's config tree, producing
// the per-key descriptors (keys, types, defaults, current values) in
// spec order. `config` may be any FlexData; only object members are
// matched, anything else resolves every key to its default.
std::vector<ConfigParam>
resolve_config_params(std::span<const ConfigKey> spec,
                      const FlexData&            config);

// Serialise resolved params to a FlexData array of objects, each:
//   { "key", "type", "required", "doc"?, "default", "current" }
// "doc" is omitted when empty. Suitable for JSON transport to clients.
FlexData config_params_to_flex(const std::vector<ConfigParam>& params);

}

#endif

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
// a key whose accepted kind varies by context.
enum class ConfigType : unsigned char {
  Bool, Int, Uint, Real, String, Array, Object, Any
};

// Stable lower-case name for a ConfigType ("bool", "int", "uint",
// "real", "string", "array", "object", "any"). Never returns null.
std::string_view config_type_name(ConfigType) noexcept;

// Static, type-level declaration of one configuration key. A stage
// exposes its schema by returning a span over a file-static
// `constexpr ConfigKey[]` table from Stage::config_spec(). The struct
// is an aggregate so tables can be written with designated
// initializers:
//
//   constexpr ConfigKey kKeys[] = {
//     {.key = "top_n",   .type = ConfigType::Int,  .def_int = 10,
//      .doc = "0 = draw all"},
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
};

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

#include "pipeline/stage-config.h"

#include <utility>

using namespace std;

namespace vpipe {

string_view
config_type_name(ConfigType t) noexcept
{
  switch (t) {
    case ConfigType::Bool:   return "bool";
    case ConfigType::Int:    return "int";
    case ConfigType::Uint:   return "uint";
    case ConfigType::Real:   return "real";
    case ConfigType::String: return "string";
    case ConfigType::Array:  return "array";
    case ConfigType::Object: return "object";
    case ConfigType::Any:    return "any";
  }
  return "any";
}

FlexData
config_default_value(const ConfigKey& k)
{
  if (k.required) {
    return FlexData::make_null();
  }
  switch (k.type) {
    case ConfigType::Bool:   return FlexData::make_bool(k.def_bool);
    case ConfigType::Int:    return FlexData::make_int(k.def_int);
    case ConfigType::Uint:   return FlexData::make_uint(k.def_uint);
    case ConfigType::Real:   return FlexData::make_real(k.def_real);
    case ConfigType::String: return FlexData::make_string(k.def_str);
    case ConfigType::Array:  return FlexData::make_array();
    case ConfigType::Object: return FlexData::make_object();
    case ConfigType::Any:    return FlexData::make_null();
  }
  return FlexData::make_null();
}

vector<ConfigParam>
resolve_config_params(span<const ConfigKey> spec, const FlexData& config)
{
  vector<ConfigParam> out;
  out.reserve(spec.size());

  const bool has_obj = config.is_object();
  for (const ConfigKey& k : spec) {
    ConfigParam p;
    p.key           = string(k.key);
    p.type          = k.type;
    p.required      = k.required;
    p.doc             = string(k.doc);
    p.suggest_db      = string(k.suggest_db);
    p.suggest_db_type = string(k.suggest_db_type);
    p.default_value   = config_default_value(k);

    p.current_value = p.default_value;   // fall back to the default
    p.present       = false;
    if (has_obj) {
      auto root = config.as_object();
      auto it   = root.find(k.key);
      if (it != root.end()) {
        p.current_value = (*it).second;
        p.present       = true;
      }
    }
    out.push_back(std::move(p));
  }
  return out;
}

FlexData
config_params_to_flex(const vector<ConfigParam>& params)
{
  FlexData arr = FlexData::make_array();
  auto av = arr.as_array();
  av.reserve(params.size());
  for (const ConfigParam& p : params) {
    FlexData obj = FlexData::make_object();
    auto ov = obj.as_object();
    ov.insert("key",      FlexData::make_string(p.key));
    ov.insert("type",     FlexData::make_string(config_type_name(p.type)));
    ov.insert("required", FlexData::make_bool(p.required));
    if (!p.doc.empty()) {
      ov.insert("doc", FlexData::make_string(p.doc));
    }
    if (!p.suggest_db.empty()) {
      ov.insert("suggest_db", FlexData::make_string(p.suggest_db));
    }
    if (!p.suggest_db_type.empty()) {
      ov.insert("suggest_db_type",
                FlexData::make_string(p.suggest_db_type));
    }
    ov.insert("default", p.default_value);
    ov.insert("current", p.current_value);
    ov.insert("present", FlexData::make_bool(p.present));
    av.push_back(std::move(obj));
  }
  return arr;
}

}

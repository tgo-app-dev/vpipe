#include "pipeline/stage-config.h"

#include "pipeline/stage-registry.h"
#include "pipeline/stage-spec.h"

#include <algorithm>
#include <cctype>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace vpipe {

namespace {

// Channel -> model_types contributed at run time (see
// register_channel_types). A vector of pairs, not a map: a handful of
// channels with a handful of types each, written a few times at plugin
// load and read when an editor asks for a config form.
//
// Function-local static so it is constructed on first use: plugin
// registration can run before this translation unit's static init.
std::vector<std::pair<std::string, std::vector<std::string>>>&
runtime_channel_types_()
{
  static std::vector<std::pair<std::string, std::vector<std::string>>> t;
  return t;
}

std::mutex&
runtime_channel_mu_()
{
  static std::mutex m;
  return m;
}

// Split a comma-separated list, trimming each entry. These lists are
// hand-written in stage specs and wrap across source lines.
void
split_into_(std::string_view csv, std::vector<std::string>* out)
{
  const std::string s(csv);
  size_t i = 0;
  while (i <= s.size()) {
    const size_t c = s.find(',', i);
    const size_t e = (c == std::string::npos) ? s.size() : c;
    std::string fam = s.substr(i, e - i);
    while (!fam.empty() && isspace((unsigned char)fam.front())) {
      fam.erase(fam.begin());
    }
    while (!fam.empty() && isspace((unsigned char)fam.back())) {
      fam.pop_back();
    }
    if (!fam.empty()
        && find(out->begin(), out->end(), fam) == out->end()) {
      out->push_back(std::move(fam));
    }
    if (c == std::string::npos) { break; }
    i = c + 1;
  }
}

// The model_types offered by the SOURCE of a shared-model channel: the
// union of what every registered CONSUMER of that channel declares.
//
// Walked from the registry on demand rather than cached, because the
// answer changes when a plugin registers -- which is the whole point --
// and this runs when an editor asks for one stage's config form, not in
// any hot path. First-seen order, so the built-ins keep their order and
// a plugin's families land after them.
string
channel_union_(string_view channel)
{
  vector<string> out;
  const StageRegistry& reg = StageRegistry::get();
  for (const auto& [id, name] : reg.all()) {
    (void)id;
    const StageSpec* sp = reg.spec(name);
    if (sp == nullptr) { continue; }
    for (const ConfigKey& k : sp->attrs) {
      if (k.model_channel != channel) { continue; }
      // A key with nothing to declare is a SOURCE, not a consumer;
      // folding its (empty) list in would be a no-op anyway, but the
      // skip keeps the two roles legible.
      if (k.suggest_db_type.empty()) { continue; }
      split_into_(k.suggest_db_type, &out);
    }
  }
  // ...and the families registered into an EXISTING stage rather than
  // brought with a stage of their own.
  {
    lock_guard<mutex> lk(runtime_channel_mu_());
    for (const auto& [ch, types] : runtime_channel_types_()) {
      if (ch != channel) { continue; }
      for (const string& t : types) {
        if (find(out.begin(), out.end(), t) == out.end()) {
          out.push_back(t);
        }
      }
    }
  }
  string csv;
  for (const string& f : out) {
    if (!csv.empty()) { csv += ','; }
    csv += f;
  }
  return csv;
}

}  // namespace

void
register_channel_types(string_view channel, string_view csv_model_types)
{
  if (channel.empty() || csv_model_types.empty()) { return; }
  lock_guard<mutex> lk(runtime_channel_mu_());
  auto& tbl = runtime_channel_types_();
  for (auto& [ch, types] : tbl) {
    if (ch == channel) { split_into_(csv_model_types, &types); return; }
  }
  tbl.push_back({string(channel), {}});
  split_into_(csv_model_types, &tbl.back().second);
}

string_view
config_type_name(ConfigType t) noexcept
{
  switch (t) {
    case ConfigType::Bool:   return "bool";
    case ConfigType::Int:    return "int";
    case ConfigType::Uint:   return "uint";
    case ConfigType::Real:   return "real";
    case ConfigType::String: return "string";
    case ConfigType::Text:   return "text";
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
    case ConfigType::Text:   return FlexData::make_string(k.def_str);
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
    // A channel SOURCE declares no families of its own; it offers what
    // the channel's consumers between them can run.
    if (!k.model_channel.empty() && k.suggest_db_type.empty()) {
      p.suggest_db_type = channel_union_(k.model_channel);
    }
    p.need_inputs     = string(k.need_inputs);
    p.need_outputs    = string(k.need_outputs);
    p.is_path         = k.is_path;
    p.path_write      = k.path_write;
    p.path_kind       = string(k.path_kind);
    p.path_filter     = string(k.path_filter);
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
    if (!p.need_inputs.empty()) {
      ov.insert("need_inputs", FlexData::make_string(p.need_inputs));
    }
    if (!p.need_outputs.empty()) {
      ov.insert("need_outputs", FlexData::make_string(p.need_outputs));
    }
    if (p.is_path) {
      ov.insert("is_path", FlexData::make_bool(true));
      if (p.path_write) {
        ov.insert("path_write", FlexData::make_bool(true));
      }
      if (!p.path_kind.empty()) {
        ov.insert("path_kind", FlexData::make_string(p.path_kind));
      }
      if (!p.path_filter.empty()) {
        ov.insert("path_filter", FlexData::make_string(p.path_filter));
      }
    }
    ov.insert("default", p.default_value);
    ov.insert("current", p.current_value);
    ov.insert("present", FlexData::make_bool(p.present));
    av.push_back(std::move(obj));
  }
  return arr;
}

}

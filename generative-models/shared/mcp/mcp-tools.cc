#include "generative-models/shared/mcp/mcp-tools.h"

#include "common/flex-data.h"

#include <cctype>
#include <ctime>
#include <exception>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace std;

namespace vpipe {

namespace {

// Trim ASCII whitespace from both ends of a view.
string_view
trim_(string_view s)
{
  const size_t b = s.find_first_not_of(" \t\r\n");
  if (b == string_view::npos) {
    return {};
  }
  const size_t e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

// Repair a common malformation in model-emitted tool calls: an UNQUOTED
// function name, e.g. {"name": write_file, "arguments": {...}}. Smaller
// / quantized models sometimes echo the `<function-name>` placeholder
// shape from the tool-use preamble without quoting it, which is invalid
// JSON. We wrap the bareword name value in quotes so the strict parser
// can then read the block. A no-op when the name is already quoted or
// absent. `arguments` stays untouched (it is a valid JSON object).
string
repair_unquoted_name_(string s)
{
  const size_t k = s.find("\"name\"");
  if (k == string::npos) {
    return s;
  }
  size_t c = s.find(':', k + 6);
  if (c == string::npos) {
    return s;
  }
  size_t p = c + 1;
  while (p < s.size()
         && std::isspace(static_cast<unsigned char>(s[p]))) {
    ++p;
  }
  if (p >= s.size() || s[p] == '"') {
    return s;   // already quoted (or nothing there)
  }
  const size_t start = p;
  auto ident = [](unsigned char ch) {
    return std::isalnum(ch) || ch == '_' || ch == '-' || ch == '.';
  };
  while (p < s.size() && ident(static_cast<unsigned char>(s[p]))) {
    ++p;
  }
  if (p == start) {
    return s;   // not a bareword we recognise
  }
  s.insert(p, "\"");
  s.insert(start, "\"");
  return s;
}

// Built-in tool: report the host's current local date and time. Args are
// accepted but ignored for now (the schema advertises none); the result
// is a small JSON object the model reads back verbatim.
string
get_current_time_(const string& /*args_json*/)
{
  const time_t now = time(nullptr);
  tm local{};
  localtime_r(&now, &local);
  char datetime[64];
  strftime(datetime, sizeof datetime, "%Y-%m-%d %H:%M:%S", &local);
  char tz[16];
  strftime(tz, sizeof tz, "%Z", &local);
  char iso[40];
  strftime(iso, sizeof iso, "%Y-%m-%dT%H:%M:%S%z", &local);
  FlexData r = FlexData::make_object();
  auto o = r.as_object();
  o.insert("datetime", FlexData::make_string(datetime));
  o.insert("timezone", FlexData::make_string(tz));
  o.insert("iso8601", FlexData::make_string(iso));
  o.insert("unix", FlexData::make_int(static_cast<int64_t>(now)));
  return r.to_json(false);
}

}  // namespace

void
McpToolRegistry::add(McpTool tool)
{
  _tools.push_back(std::move(tool));
}

const McpTool*
McpToolRegistry::find_(const string& name) const noexcept
{
  for (const auto& t : _tools) {
    if (t.name == name) {
      return &t;
    }
  }
  // Namespace-tolerant fallback: a model may call a declared tool under a
  // module namespace (the Gemma-4 12B emits `file_operations.write_file`
  // for `write_file`). Retry with the segment after the last '.'.
  const size_t dot = name.rfind('.');
  if (dot != string::npos && dot + 1 < name.size()) {
    const string base = name.substr(dot + 1);
    for (const auto& t : _tools) {
      if (t.name == base) {
        return &t;
      }
    }
  }
  return nullptr;
}

bool
McpToolRegistry::has(const string& name) const noexcept
{
  return find_(name) != nullptr;
}

string
McpToolRegistry::tools_json() const
{
  // One function spec per line, compact, in the OpenAI/Hermes shape
  // Qwen was trained to consume:
  //   {"type":"function","function":{"name",...,"parameters":{...}}}
  string out;
  for (const auto& t : _tools) {
    FlexData spec = FlexData::make_object();
    auto so = spec.as_object();
    so.insert("type", FlexData::make_string("function"));
    FlexData fn = FlexData::make_object();
    auto fo = fn.as_object();
    fo.insert("name", FlexData::make_string(t.name));
    fo.insert("description", FlexData::make_string(t.description));
    // Nest the parameter schema as JSON, not as a quoted string. Fall
    // back to an empty object when a tool ships a malformed schema.
    FlexData params;
    try {
      params = FlexData::from_json(
          t.parameters_json.empty() ? "{}" : t.parameters_json);
    } catch (const std::exception&) {
      params = FlexData::make_object();
    }
    fo.insert("parameters", std::move(params));
    so.insert("function", std::move(fn));
    if (!out.empty()) {
      out += '\n';
    }
    out += spec.to_json(false);
  }
  return out;
}

string
McpToolRegistry::dispatch(const McpToolCall& call) const
{
  // Resolve the (possibly module-namespaced) name against the registry.
  if (const McpTool* t = find_(call.name)) {
    if (!t->handler) {
      return "{\"error\":\"tool has no handler\"}";
    }
    try {
      return t->handler(call.arguments_json.empty() ? "{}"
                                                     : call.arguments_json);
    } catch (const std::exception& e) {
      FlexData err = FlexData::make_object();
      auto eo = err.as_object();
      eo.insert("error", FlexData::make_string(e.what()));
      return err.to_json(false);
    }
  }
  FlexData err = FlexData::make_object();
  auto eo = err.as_object();
  eo.insert("error", FlexData::make_string("unknown tool: " + call.name));
  return err.to_json(false);
}

namespace {

constexpr string_view kCallOpen  = "<tool_call>";
constexpr string_view kCallClose = "</tool_call>";

// The body of each <tool_call> block, in order; an unclosed one runs to
// the end of the text.
vector<string_view>
call_blocks_(string_view text)
{
  vector<string_view> blocks;
  size_t pos = 0;
  while (true) {
    const size_t o = text.find(kCallOpen, pos);
    if (o == string_view::npos) {
      break;
    }
    const size_t b = o + kCallOpen.size();
    const size_t c = text.find(kCallClose, b);
    blocks.push_back(c == string_view::npos ? text.substr(b)
                                            : text.substr(b, c - b));
    pos = c == string_view::npos ? text.size() : c + kCallClose.size();
  }
  return blocks;
}

// One block's Hermes JSON, {"name": ..., "arguments": ...}. False when
// it does not read as one.
bool
read_json_call_(string_view inner, McpToolCall* out)
{
  FlexData fd;
  bool parsed = false;
  const string_view body = trim_(inner);
  try {
    fd = FlexData::from_json(body);
    parsed = true;
  } catch (const std::exception&) {
    // Retry once after repairing an unquoted function name.
    try {
      fd = FlexData::from_json(repair_unquoted_name_(string(body)));
      parsed = true;
    } catch (const std::exception&) {
    }
  }
  if (!parsed || !fd.is_object()) {
    return false;
  }
  auto obj = fd.as_object();
  if (!obj.contains("name")) {
    return false;
  }
  FlexData name = obj.at("name");
  if (!name.is_string()) {
    return false;
  }
  out->name = string(name.get_string());
  out->arguments_json = "{}";
  if (obj.contains("arguments")) {
    FlexData a = obj.at("arguments");
    if (a.is_object() || a.is_array()) {
      out->arguments_json = a.to_json(false);
    } else if (a.is_string()) {
      out->arguments_json = string(a.get_string());
    }
  }
  return true;
}

// What a parameter's schema says its value is. A union of a string and
// null ("type": ["string", "null"]) is a string; any other union, JSON.
enum class ArgKind { Unknown, String, Integer, Number, Boolean, Json };

ArgKind
arg_kind_(const FlexData& type)
{
  if (type.is_string()) {
    const string_view t = type.get_string();
    if (t == "string")  { return ArgKind::String; }
    if (t == "integer") { return ArgKind::Integer; }
    if (t == "number")  { return ArgKind::Number; }
    if (t == "boolean") { return ArgKind::Boolean; }
    if (t == "object" || t == "array") { return ArgKind::Json; }
    return ArgKind::Unknown;
  }
  if (!type.is_array()) {
    return ArgKind::Unknown;
  }
  const auto ta = type.as_array();
  bool strings = false;
  for (size_t i = 0; i < ta.size(); ++i) {
    const FlexData t = ta[i];
    const string_view s = t.as_string();
    if (s == "string") {
      strings = true;
    } else if (s != "null") {
      return ArgKind::Json;
    }
  }
  return strings ? ArgKind::String : ArgKind::Json;
}

// Each tool's parameters by name, from tools_json's lines: the
// OpenAI / Hermes {"type":"function","function":{...}} shape, or the
// function object itself.
using ToolKinds = vector<pair<string, ArgKind>>;
using Schemas   = vector<pair<string, ToolKinds>>;

Schemas
schemas_(string_view tools_json)
{
  Schemas out;
  size_t pos = 0;
  while (pos < tools_json.size()) {
    const size_t nl = tools_json.find('\n', pos);
    const string_view line = tools_json.substr(
        pos, nl == string_view::npos ? string_view::npos : nl - pos);
    pos = nl == string_view::npos ? tools_json.size() : nl + 1;
    if (trim_(line).empty()) {
      continue;
    }
    try {
      FlexData spec = FlexData::from_json(line);
      if (!spec.is_object()) { continue; }
      if (spec.as_object().contains("function")) {
        spec = spec.as_object().at("function");
        if (!spec.is_object()) { continue; }
      }
      const auto so = spec.as_object();
      if (!so.contains("name") || !so.at("name").is_string()) { continue; }
      ToolKinds kinds;
      if (so.contains("parameters") && so.at("parameters").is_object()) {
        const FlexData params = so.at("parameters");
        const auto po = params.as_object();
        if (po.contains("properties") && po.at("properties").is_object()) {
          const FlexData props = po.at("properties");
          for (const auto& [key, p] : props.as_object()) {
            const bool typed = p.is_object()
                && p.as_object().contains("type");
            kinds.emplace_back(string(key),
                typed ? arg_kind_(p.as_object().at("type"))
                      : ArgKind::Unknown);
          }
        }
      }
      out.emplace_back(string(so.at("name").get_string()),
                       std::move(kinds));
    } catch (const std::exception&) {
      continue;   // a malformed spec types nothing
    }
  }
  return out;
}

// The text as JSON, or nothing when it is not JSON (trailing words
// included).
bool
as_json_(string_view text, FlexData* out)
{
  const string_view t = trim_(text);
  if (t.empty()) { return false; }
  try {
    *out = FlexData::from_json(t);
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

bool
iequals_(string_view a, string_view b)
{
  if (a.size() != b.size()) { return false; }
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::tolower(static_cast<unsigned char>(a[i]))
        != std::tolower(static_cast<unsigned char>(b[i]))) {
      return false;
    }
  }
  return true;
}

// A parameter's value, as its kind reads it. `known` is false when the
// tool has no schema here, so the value's own shape decides.
FlexData
arg_value_(string_view value, ArgKind kind, bool known)
{
  const string_view t = trim_(value);
  FlexData j;
  if (iequals_(t, "null")) {
    return FlexData::make_null();
  }
  switch (kind) {
  case ArgKind::String:
    return FlexData::make_string(value);
  case ArgKind::Integer:
  case ArgKind::Number:
    if (as_json_(t, &j) && (j.is_int() || j.is_uint() || j.is_real())) {
      return j;
    }
    return FlexData::make_string(value);
  case ArgKind::Boolean:
    if (iequals_(t, "true"))  { return FlexData::make_bool(true); }
    if (iequals_(t, "false")) { return FlexData::make_bool(false); }
    return FlexData::make_string(value);
  case ArgKind::Json:
    return as_json_(t, &j) ? j : FlexData::make_string(value);
  case ArgKind::Unknown:
    break;
  }
  // No type to go by: a tool without a schema here takes any JSON; a
  // parameter its schema leaves untyped (or does not name) only an
  // object or an array, the shapes no string is mistaken for.
  if (!known) {
    return as_json_(t, &j) ? j : FlexData::make_string(value);
  }
  if (!t.empty() && (t.front() == '{' || t.front() == '[')
      && as_json_(t, &j) && (j.is_object() || j.is_array())) {
    return j;
  }
  return FlexData::make_string(value);
}

// The template writes `<parameter=KEY>\nvalue\n</parameter>`: one line
// break either side of the value is the template's, not the value's.
string_view
strip_breaks_(string_view v)
{
  if (!v.empty() && v.front() == '\n') { v.remove_prefix(1); }
  if (!v.empty() && v.back() == '\n')  { v.remove_suffix(1); }
  if (!v.empty() && v.back() == '\r')  { v.remove_suffix(1); }
  return v;
}

// Every <function=NAME>...</function> in `text`, its parameters read
// by the tool's schema. A function's body runs to its </function>, else
// to the next <function=, else to the end; a parameter's value to its
// </parameter>, else to the next <parameter=, else to the body's end.
void
read_xml_calls_(string_view           text,
                const Schemas&        schemas,
                vector<McpToolCall>*  out)
{
  static constexpr string_view kFn     = "<function=";
  static constexpr string_view kFnEnd  = "</function>";
  static constexpr string_view kPar    = "<parameter=";
  static constexpr string_view kParEnd = "</parameter>";
  size_t pos = 0;
  while (true) {
    const size_t f = text.find(kFn, pos);
    if (f == string_view::npos) {
      break;
    }
    const size_t name_at = f + kFn.size();
    const size_t gt = text.find('>', name_at);
    if (gt == string_view::npos) {
      break;
    }
    const string name(trim_(text.substr(name_at, gt - name_at)));
    size_t end = text.find(kFnEnd, gt + 1);
    const size_t next = text.find(kFn, gt + 1);
    if (end == string_view::npos || (next != string_view::npos
                                     && next < end)) {
      end = next == string_view::npos ? text.size() : next;
      pos = end;
    } else {
      pos = end + kFnEnd.size();
    }
    const string_view body = text.substr(gt + 1, end - (gt + 1));
    if (name.empty()) {
      continue;
    }
    const ToolKinds* kinds = nullptr;
    for (const auto& [tool, k] : schemas) {
      if (tool == name) { kinds = &k; break; }
    }
    FlexData args = FlexData::make_object();
    auto ao = args.as_object();
    size_t p = 0;
    while (true) {
      const size_t a = body.find(kPar, p);
      if (a == string_view::npos) {
        break;
      }
      const size_t key_at = a + kPar.size();
      const size_t kg = body.find('>', key_at);
      if (kg == string_view::npos) {
        break;
      }
      const string key(trim_(body.substr(key_at, kg - key_at)));
      size_t v_end = body.find(kParEnd, kg + 1);
      const size_t v_next = body.find(kPar, kg + 1);
      size_t resume;
      if (v_end == string_view::npos || (v_next != string_view::npos
                                         && v_next < v_end)) {
        v_end = v_next == string_view::npos ? body.size() : v_next;
        resume = v_end;
      } else {
        resume = v_end + kParEnd.size();
      }
      const string_view value =
          strip_breaks_(body.substr(kg + 1, v_end - (kg + 1)));
      p = resume;
      if (key.empty()) {
        continue;
      }
      ArgKind kind = ArgKind::Unknown;
      if (kinds != nullptr) {
        for (const auto& [k, ak] : *kinds) {
          if (k == key) { kind = ak; break; }
        }
      }
      ao.insert_or_assign(key, arg_value_(value, kind, kinds != nullptr));
    }
    McpToolCall tc;
    tc.name = name;
    tc.arguments_json = args.to_json(false);
    out->push_back(std::move(tc));
  }
}

}  // namespace

vector<McpToolCall>
parse_tool_calls(const string& assistant_text)
{
  vector<McpToolCall> calls;
  for (const string_view inner : call_blocks_(assistant_text)) {
    McpToolCall tc;
    if (read_json_call_(inner, &tc)) {
      calls.push_back(std::move(tc));
    }
    // A malformed block is skipped; the scan goes on.
  }
  return calls;
}

vector<McpToolCall>
parse_xml_tool_calls(const string& assistant_text, string_view tools_json)
{
  static constexpr string_view kFn = "<function=";
  const Schemas schemas = schemas_(tools_json);
  vector<McpToolCall> calls;
  const vector<string_view> blocks = call_blocks_(assistant_text);
  if (blocks.empty()) {
    // The model's form without its wrapper: read as its parser reads
    // it (vLLM's Qwen3-Coder parser takes the whole reply then).
    read_xml_calls_(assistant_text, schemas, &calls);
    return calls;
  }
  for (const string_view inner : blocks) {
    if (inner.find(kFn) != string_view::npos) {
      read_xml_calls_(inner, schemas, &calls);
      continue;
    }
    McpToolCall tc;
    if (read_json_call_(inner, &tc)) {
      calls.push_back(std::move(tc));
    }
  }
  return calls;
}

McpToolRegistry
make_builtin_tool_registry()
{
  McpToolRegistry reg;
  reg.add(McpTool{
      .name        = "get_current_time",
      .description = "Get the host's current local date and time.",
      .parameters_json =
          "{\"type\":\"object\",\"properties\":{},\"required\":[]}",
      .handler = &get_current_time_,
  });
  return reg;
}

}

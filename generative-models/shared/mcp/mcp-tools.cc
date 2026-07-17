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

vector<McpToolCall>
parse_tool_calls(const string& assistant_text)
{
  static constexpr string_view kOpen  = "<tool_call>";
  static constexpr string_view kClose = "</tool_call>";
  vector<McpToolCall> calls;
  size_t pos = 0;
  while (true) {
    const size_t o = assistant_text.find(kOpen, pos);
    if (o == string::npos) {
      break;
    }
    const size_t js = o + kOpen.size();
    const size_t c  = assistant_text.find(kClose, js);
    const string_view inner = (c == string::npos)
        ? string_view(assistant_text).substr(js)
        : string_view(assistant_text).substr(js, c - js);
    pos = (c == string::npos) ? assistant_text.size() : c + kClose.size();

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
      continue;   // still malformed -- skip, keep scanning
    }
    auto obj = fd.as_object();
    if (!obj.contains("name")) {
      continue;
    }
    FlexData name = obj.at("name");
    if (!name.is_string()) {
      continue;
    }
    McpToolCall tc;
    tc.name = string(name.get_string());
    tc.arguments_json = "{}";
    if (obj.contains("arguments")) {
      FlexData a = obj.at("arguments");
      if (a.is_object() || a.is_array()) {
        tc.arguments_json = a.to_json(false);
      } else if (a.is_string()) {
        tc.arguments_json = string(a.get_string());
      }
    }
    calls.push_back(std::move(tc));
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

#ifndef VPIPE_GENERATIVE_MODELS_MCP_TOOLS_H
#define VPIPE_GENERATIVE_MODELS_MCP_TOOLS_H

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace vpipe {

// One locally-dispatchable tool, described in the shape an MCP server
// advertises one -- a name, a human-readable description, and a
// JSON-Schema object for its arguments -- plus a handler that runs it.
// `handler` receives the model-supplied arguments as a JSON object
// string ("{}" when the model passed none) and returns the tool's
// result as text; that text is fed straight back to the model inside a
// tool-response turn.
//
// This is the local half of MCP (Model Context Protocol): the tool
// surface plus a synchronous call path. A future transport can register
// tools whose handler proxies to a remote MCP server behind the same
// signature, so callers never learn whether a tool is built-in or
// remote.
struct McpTool {
  std::string name;
  std::string description;
  // JSON-Schema object describing the arguments, e.g.
  // {"type":"object","properties":{...},"required":[...]}.
  std::string parameters_json;
  std::function<std::string(const std::string& args_json)> handler;
};

// One tool invocation parsed out of an assistant turn.
struct McpToolCall {
  std::string name;
  std::string arguments_json;   // "{}" when the model emitted no args
};

// A registry of tools text-chat advertises to the model and dispatches
// the model's calls against.
class McpToolRegistry {
public:
  void add(McpTool tool);

  bool        empty() const noexcept { return _tools.empty(); }
  std::size_t size()  const noexcept { return _tools.size(); }
  bool        has(const std::string& name) const noexcept;

  // The `<tools>`-block body advertised to the model: one JSON object
  // per line in the OpenAI/Hermes function shape Qwen was trained on
  // ({"type":"function","function":{name,description,parameters}}).
  std::string tools_json() const;

  // Run one call. Returns the handler's result text; when no tool
  // matches `call.name`, returns a JSON error string so the model sees
  // (and can recover from) the miss rather than the turn silently
  // stalling.
  std::string dispatch(const McpToolCall& call) const;

private:
  // Resolve a possibly-namespaced tool name to a registered tool. Some
  // models call a declared tool under a module namespace -- e.g. the
  // Gemma-4 12B emits `file_operations.write_file` for `write_file`. Tries
  // the exact name, then the segment after the last '.'. Null if no match.
  const McpTool* find_(const std::string& name) const noexcept;

  std::vector<McpTool> _tools;
};

// Extract every <tool_call>...</tool_call> block the assistant emitted
// (Hermes/Qwen format), parsing each block's JSON into name + arguments.
// Malformed blocks are skipped. An empty result means the turn made no
// tool calls.
std::vector<McpToolCall>
parse_tool_calls(const std::string& assistant_text);

// Extract the calls of the XML function form Qwen3.5 / 3.6 / 3.8's own
// chat templates teach (Qwen3-Coder's):
//
//   <tool_call>
//   <function=NAME>
//   <parameter=KEY>
//   value
//   </parameter>
//   </function>
//   </tool_call>
//
// The template writes a string value as it is and anything else as
// JSON, so a value is read back by the type its tool's schema gives the
// parameter -- `tools_json`, one function spec a line as tools_json()
// writes them: a string stays the words written ("2026" a string where
// the schema says so), an integer, number or boolean is read as one, an
// object or array as JSON, `null` as null. Without a schema for the
// tool, a value that reads as JSON is taken as JSON. A <tool_call> block
// holding no <function=...> is read as Hermes JSON (parse_tool_calls),
// and a <function=...> written with no <tool_call> round it is read too.
// Unclosed tags run to the end of what holds them, as a decode cut off
// leaves them.
std::vector<McpToolCall>
parse_xml_tool_calls(const std::string& assistant_text,
                     std::string_view   tools_json);

// A registry seeded with the built-in tools -- currently just
// get_current_time, which reports the host's current local date and
// time. This is the starting point for MCP support in text-chat.
McpToolRegistry make_builtin_tool_registry();

}

#endif

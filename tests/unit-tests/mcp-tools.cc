// mcp-tools.cc -- tests for the text-chat MCP tool registry: the
// built-in tool surface, the <tools> advertisement JSON, dispatch, and
// the <tool_call> parser. All pure (no model / no metal), so these run
// in every build.

#include "minitest.h"

#include "common/flex-data.h"
#include "generative-models/shared/mcp/mcp-tools.h"

#include <string>
#include <vector>

using namespace std;
using namespace vpipe;

TEST(mcp_tools, builtin_registry_has_time_tool)
{
  McpToolRegistry reg = make_builtin_tool_registry();
  EXPECT_FALSE(reg.empty());
  EXPECT_TRUE(reg.has("get_current_time"));
  EXPECT_FALSE(reg.has("no_such_tool"));
}

TEST(mcp_tools, tools_json_is_valid_function_specs)
{
  McpToolRegistry reg = make_builtin_tool_registry();
  const string js = reg.tools_json();
  EXPECT_FALSE(js.empty());
  // Each line is one function spec; parse the first and check shape.
  const string first = js.substr(0, js.find('\n'));
  FlexData fd = FlexData::from_json(first);
  EXPECT_TRUE(fd.is_object());
  auto o = fd.as_object();
  EXPECT_TRUE(o.contains("type"));
  EXPECT_TRUE(o.contains("function"));
  FlexData fn = o.at("function");
  EXPECT_TRUE(fn.is_object());
  auto fo = fn.as_object();
  EXPECT_TRUE(string(fo.at("name").as_string("")) == "get_current_time");
  EXPECT_TRUE(fo.contains("description"));
  // parameters must be nested JSON, not a quoted string.
  EXPECT_TRUE(fo.at("parameters").is_object());
}

TEST(mcp_tools, dispatch_time_tool_returns_structured_result)
{
  McpToolRegistry reg = make_builtin_tool_registry();
  McpToolCall call;
  call.name = "get_current_time";
  call.arguments_json = "{}";
  const string res = reg.dispatch(call);
  FlexData fd = FlexData::from_json(res);
  EXPECT_TRUE(fd.is_object());
  auto o = fd.as_object();
  EXPECT_TRUE(o.contains("datetime"));
  EXPECT_TRUE(o.contains("iso8601"));
  EXPECT_TRUE(o.contains("unix"));
  // A real epoch second is comfortably past 2020-01-01.
  EXPECT_TRUE(o.at("unix").as_int(0) > 1577836800);
}

TEST(mcp_tools, dispatch_unknown_tool_reports_error)
{
  McpToolRegistry reg = make_builtin_tool_registry();
  McpToolCall call;
  call.name = "does_not_exist";
  const string res = reg.dispatch(call);
  FlexData fd = FlexData::from_json(res);
  EXPECT_TRUE(fd.is_object());
  EXPECT_TRUE(fd.as_object().contains("error"));
}

TEST(mcp_tools, parse_single_tool_call)
{
  const string reply =
      "Let me check.\n<tool_call>\n"
      "{\"name\": \"get_current_time\", \"arguments\": {}}\n"
      "</tool_call>";
  auto calls = parse_tool_calls(reply);
  EXPECT_TRUE(calls.size() == 1);
  if (!calls.empty()) {
    EXPECT_TRUE(calls[0].name == "get_current_time");
    // Empty args normalise to "{}".
    EXPECT_TRUE(calls[0].arguments_json == "{}");
  }
}

TEST(mcp_tools, parse_multiple_tool_calls_with_args)
{
  const string reply =
      "<tool_call>\n{\"name\": \"a\", \"arguments\": {\"x\": 1}}\n"
      "</tool_call>\n"
      "<tool_call>\n{\"name\": \"b\", \"arguments\": {}}\n</tool_call>";
  auto calls = parse_tool_calls(reply);
  EXPECT_TRUE(calls.size() == 2);
  if (calls.size() == 2) {
    EXPECT_TRUE(calls[0].name == "a");
    EXPECT_TRUE(calls[0].arguments_json.find("\"x\"") != string::npos);
    EXPECT_TRUE(calls[1].name == "b");
  }
}

TEST(mcp_tools, parse_repairs_unquoted_name)
{
  // Smaller/quantized models sometimes emit the function name unquoted
  // (mimicking the <function-name> placeholder). The parser repairs it.
  const string reply =
      "<tool_call>\n"
      "{\"name\": write_file, \"arguments\": {\"path\": \"m.txt\"}}\n"
      "</tool_call>";
  auto calls = parse_tool_calls(reply);
  EXPECT_TRUE(calls.size() == 1);
  if (!calls.empty()) {
    EXPECT_TRUE(calls[0].name == "write_file");
    EXPECT_TRUE(calls[0].arguments_json.find("m.txt") != string::npos);
  }
}

TEST(mcp_tools, parse_unescaped_quote_in_args_yields_no_call)
{
  // Documented failure mode: models sometimes emit an UNESCAPED quote
  // inside a string argument (here after "# X"), which makes the whole
  // <tool_call> object invalid JSON. It cannot be repaired unambiguously,
  // so parse yields zero calls -- the text-chat tool loop then logs a warn
  // and returns the raw block instead of running a tool.
  const string reply =
      "<tool_call>\n"
      "{\"name\": \"run_python\", \"arguments\": {\"code\": \"# X\"oops"
      "\\nprint(1)\"}}\n"
      "</tool_call>";
  auto calls = parse_tool_calls(reply);
  EXPECT_TRUE(calls.empty());
}

TEST(mcp_tools, parse_no_tool_call_is_empty)
{
  auto calls = parse_tool_calls("Just a plain answer, no tools here.");
  EXPECT_TRUE(calls.empty());
}

TEST(mcp_tools, parse_skips_malformed_block)
{
  // First block is not valid JSON -> skipped; second is well-formed.
  const string reply =
      "<tool_call>\nnot json at all\n</tool_call>\n"
      "<tool_call>\n{\"name\": \"ok\", \"arguments\": {}}\n</tool_call>";
  auto calls = parse_tool_calls(reply);
  EXPECT_TRUE(calls.size() == 1);
  if (!calls.empty()) {
    EXPECT_TRUE(calls[0].name == "ok");
  }
}

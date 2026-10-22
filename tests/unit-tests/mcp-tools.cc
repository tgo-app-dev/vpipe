// mcp-tools.cc -- tests for the text-chat MCP tool registry: the
// built-in tool surface, the <tools> advertisement JSON, dispatch, and
// the <tool_call> parser. All pure (no model / no metal), so these run
// in every build.

#include "minitest.h"

#include "common/flex-data.h"
#include "common/media-line.h"
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

// A model may call a declared tool under a module namespace -- the Gemma-4
// 12B emits e.g. `file_operations.get_current_time`. Dispatch resolves the
// segment after the last '.' to the registered tool; has() agrees.
TEST(mcp_tools, dispatch_resolves_namespaced_tool_name)
{
  McpToolRegistry reg = make_builtin_tool_registry();
  EXPECT_TRUE(reg.has("tools.get_current_time"));
  EXPECT_TRUE(reg.has("file_operations.get_current_time"));
  EXPECT_FALSE(reg.has("file_operations.no_such_tool"));

  McpToolCall call;
  call.name = "default_api.get_current_time";
  const string res = reg.dispatch(call);
  FlexData fd = FlexData::from_json(res);
  EXPECT_TRUE(fd.is_object());
  auto o = fd.as_object();
  EXPECT_TRUE(o.contains("datetime"));   // resolved + ran, not "unknown tool"
  EXPECT_FALSE(o.contains("error"));

  // A namespaced name whose base is still unknown stays an error.
  McpToolCall bad;
  bad.name = "file_operations.create_directory_recursive";
  FlexData bfd = FlexData::from_json(reg.dispatch(bad));
  EXPECT_TRUE(bfd.as_object().contains("error"));
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

// ---- the XML function form (Qwen3.5 / 3.6 / 3.8's own template) ------

namespace {

// add_text's spec as text-chat writes a tool a line.
const char* kAddTextSpec =
    "{\"type\":\"function\",\"function\":{\"name\":\"add_text\","
    "\"description\":\"Put words on a page.\",\"parameters\":{\"type\":"
    "\"object\",\"properties\":{\"text\":{\"type\":\"string\"},\"size\":"
    "{\"type\":\"integer\"},\"bold\":{\"type\":\"boolean\"},\"box\":"
    "{\"type\":\"array\"},\"opacity\":{\"type\":\"number\"},\"note\":"
    "{\"type\":[\"string\",\"null\"]},\"style\":{\"description\":\"any\"}},"
    "\"required\":[\"text\"]}}}";

FlexData
args_(const McpToolCall& c)
{
  return FlexData::from_json(c.arguments_json);
}

}  // namespace

// Each value read by the type its schema gives it: the template writes
// a string as it is and anything else as JSON, so "2026" is the words
// where the schema says string, and a number where it says integer.
TEST(mcp_tools, xml_call_is_typed_by_its_schema)
{
  const string reply =
      "I'll put the year on the cover.\n\n<tool_call>\n"
      "<function=add_text>\n"
      "<parameter=text>\n2026\n</parameter>\n"
      "<parameter=size>\n48\n</parameter>\n"
      "<parameter=bold>\ntrue\n</parameter>\n"
      "<parameter=box>\n[40, 30, 560, 120]\n</parameter>\n"
      "<parameter=opacity>\n0.5\n</parameter>\n"
      "<parameter=note>\nnull\n</parameter>\n"
      "<parameter=style>\n{\"align\": \"center\"}\n</parameter>\n"
      "</function>\n</tool_call>";
  auto calls = parse_xml_tool_calls(reply, kAddTextSpec);
  EXPECT_TRUE(calls.size() == 1);
  if (calls.size() != 1) { return; }
  EXPECT_TRUE(calls[0].name == "add_text");
  const FlexData a = args_(calls[0]);
  const auto o = a.as_object();
  EXPECT_TRUE(o.at("text").is_string());
  EXPECT_TRUE(o.at("text").as_string() == "2026");
  EXPECT_TRUE(o.at("size").is_int() && o.at("size").as_int() == 48);
  EXPECT_TRUE(o.at("bold").is_bool() && o.at("bold").as_bool());
  EXPECT_TRUE(o.at("box").is_array() && o.at("box").as_array().size() == 4);
  EXPECT_TRUE(o.at("opacity").as_real() == 0.5);
  EXPECT_TRUE(o.at("note").is_null());
  // Untyped in its schema: an object is taken as one.
  EXPECT_TRUE(o.at("style").is_object());
}

// A string keeps its own line breaks and spaces; only the one break the
// template puts either side of it goes. A number the schema calls a
// string stays the words; one it does not name is a string too.
TEST(mcp_tools, xml_string_keeps_its_lines)
{
  const string reply =
      "<tool_call>\n<function=add_text>\n"
      "<parameter=text>\n  Mercury\nVenus  \n</parameter>\n"
      "<parameter=size>\nlarge\n</parameter>\n"
      "<parameter=page>\n3\n</parameter>\n"
      "</function>\n</tool_call>";
  auto calls = parse_xml_tool_calls(reply, kAddTextSpec);
  EXPECT_TRUE(calls.size() == 1);
  if (calls.size() != 1) { return; }
  const FlexData a = args_(calls[0]);
  const auto o = a.as_object();
  EXPECT_TRUE(o.at("text").as_string() == "  Mercury\nVenus  ");
  // Not an integer: the words, for the tool to refuse by name.
  EXPECT_TRUE(o.at("size").as_string() == "large");
  EXPECT_TRUE(o.at("page").is_string() && o.at("page").as_string() == "3");
}

// Several calls, one block each or two in one; without a schema for the
// tool a value that reads as JSON is JSON.
TEST(mcp_tools, xml_calls_in_order_without_a_schema)
{
  const string reply =
      "<tool_call>\n<function=get_project>\n</function>\n</tool_call>\n"
      "<tool_call>\n<function=set_page>\n<parameter=page>\n2\n"
      "</parameter>\n<parameter=title>\nOrbits\n</parameter>\n"
      "</function>\n<function=view_page>\n<parameter=page>\n2\n"
      "</parameter>\n</function>\n</tool_call>";
  auto calls = parse_xml_tool_calls(reply, "");
  EXPECT_TRUE(calls.size() == 3);
  if (calls.size() != 3) { return; }
  EXPECT_TRUE(calls[0].name == "get_project");
  EXPECT_TRUE(calls[0].arguments_json == "{}");
  EXPECT_TRUE(calls[1].name == "set_page");
  const FlexData a = args_(calls[1]);
  EXPECT_TRUE(a.as_object().at("page").is_int());
  EXPECT_TRUE(a.as_object().at("title").as_string() == "Orbits");
  EXPECT_TRUE(calls[2].name == "view_page");
}

// A decode cut off leaves its tags open: what was written is read.
TEST(mcp_tools, xml_unclosed_tags_run_to_the_end)
{
  const string reply =
      "<tool_call>\n<function=add_text>\n<parameter=size>\n32\n"
      "<parameter=text>\nSolar System";
  auto calls = parse_xml_tool_calls(reply, kAddTextSpec);
  EXPECT_TRUE(calls.size() == 1);
  if (calls.size() != 1) { return; }
  const FlexData a = args_(calls[0]);
  EXPECT_TRUE(a.as_object().at("size").as_int() == 32);
  EXPECT_TRUE(a.as_object().at("text").as_string() == "Solar System");
}

// The Hermes JSON in a block, and the function form with no block round
// it, are read too; a reply with neither has no calls.
TEST(mcp_tools, xml_parser_takes_json_and_a_bare_function)
{
  auto json = parse_xml_tool_calls(
      "<tool_call>\n{\"name\": \"get_project\", \"arguments\": {}}\n"
      "</tool_call>", kAddTextSpec);
  EXPECT_TRUE(json.size() == 1);
  if (json.size() != 1) { return; }
  EXPECT_TRUE(json[0].name == "get_project");
  auto bare = parse_xml_tool_calls(
      "<function=add_text>\n<parameter=text>\nHi\n</parameter>\n"
      "</function>", kAddTextSpec);
  EXPECT_TRUE(bare.size() == 1);
  if (bare.size() != 1) { return; }
  EXPECT_TRUE(args_(bare[0]).as_object().at("text").as_string() == "Hi");
  EXPECT_TRUE(parse_xml_tool_calls("Done: 14 pages.", kAddTextSpec)
                  .empty());
  EXPECT_TRUE(parse_xml_tool_calls(
                  "<tool_call>\nnot a call\n</tool_call>", "").empty());
}

// A call the model drafted, or only mentioned, while it reasoned is not
// one it made: text-chat reads the calls of reply_without_thinking().
// Over the whole reply, a mention made a call with no arguments and a
// draft doubled the real one.
TEST(mcp_tools, a_call_made_while_reasoning_is_not_read)
{
  const std::string tools =
      "{\"type\":\"function\",\"function\":{\"name\":\"set_title\","
      "\"parameters\":{\"type\":\"object\",\"properties\":{\"title\":"
      "{\"type\":\"string\"}}}}}";
  const std::string ts(vpipe::media_line::kThinkStart);
  const std::string te(vpipe::media_line::kThinkEnd);
  const std::string call =
      "<tool_call>\n<function=set_title>\n<parameter=title>\nWater\n"
      "</parameter>\n</function>\n</tool_call>";
  const std::string mention =
      ts + "I could use <function=set_title> but they only asked.\n" + te
      + "\n\nIt would be \"Water\".";
  const std::string draft = ts + "Draft:\n" + call + "\n" + te + "\n\n" + call;
  auto read = [&](const std::string& reply) {
    return vpipe::parse_xml_tool_calls(
        vpipe::media_line::reply_without_thinking(reply), tools);
  };
  EXPECT_TRUE(vpipe::parse_xml_tool_calls(mention, tools).size() == 1);
  EXPECT_TRUE(vpipe::parse_xml_tool_calls(draft, tools).size() == 2);
  EXPECT_TRUE(read(mention).empty());
  const auto calls = read(draft);
  EXPECT_TRUE(calls.size() == 1);
  if (calls.size() != 1) { return; }
  EXPECT_TRUE(calls[0].name == "set_title"
              && calls[0].arguments_json == "{\"title\":\"Water\"}");
}

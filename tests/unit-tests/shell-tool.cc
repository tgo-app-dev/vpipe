// shell-tool.cc -- tests for the seatbelt-sandboxed `run_shell` MCP tool.
// macOS only (seatbelt); each case runs a real command through
// sandbox-exec, so they self-skip when it is not present.

#include "minitest.h"

#include "common/flex-data.h"
#include "generative-models/shared/mcp/mcp-tools.h"
#include "generative-models/shared/mcp/shell-tool.h"

#include <cstdlib>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

namespace fs = std::filesystem;

bool sandbox_available_()
{
  return ::access("/usr/bin/sandbox-exec", X_OK) == 0;
}

string make_tmpdir_(const char* tag)
{
  string tmpl = "/tmp/vpipe-shtool-";
  tmpl += tag;
  tmpl += "-XXXXXX";
  vector<char> buf(tmpl.begin(), tmpl.end());
  buf.push_back('\0');
  const char* p = ::mkdtemp(buf.data());
  return p ? string(p) : string();
}

}  // namespace

TEST(shell_tool, tool_shape)
{
  ShellToolOptions o;
  McpTool t = make_shell_tool(o);
  EXPECT_TRUE(t.name == "run_shell");
  EXPECT_TRUE(!!t.handler);
  EXPECT_TRUE(t.parameters_json.find("command") != string::npos);
}

TEST(shell_tool, missing_command_reports_error)
{
  ShellToolOptions o;
  McpTool t = make_shell_tool(o);
  const string out = t.handler("{}");
  FlexData fd = FlexData::from_json(out);
  EXPECT_TRUE(fd.is_object());
  EXPECT_TRUE(fd.as_object().contains("error"));
}

TEST(shell_tool, runs_command_and_captures_output)
{
  if (!sandbox_available_()) { return; }
  const string ws = make_tmpdir_("run");
  ASSERT_TRUE(!ws.empty());

  ShellToolOptions o;
  o.workspace = ws;
  McpTool t = make_shell_tool(o);

  // Write a file in the workspace (cwd) and echo -- both must succeed.
  const string out = t.handler(
      "{\"command\": \"echo hi-there > note.txt; echo done\"}");
  FlexData fd = FlexData::from_json(out);
  EXPECT_TRUE(fd.is_object());
  auto obj = fd.as_object();
  EXPECT_TRUE(obj.at("exit_code").as_int(-1) == 0);
  EXPECT_TRUE(string(obj.at("stdout").as_string("")).find("done")
              != string::npos);
  EXPECT_TRUE(fs::exists(ws + "/note.txt"));

  error_code ec;
  fs::remove_all(ws, ec);
}

TEST(shell_tool, confines_writes_to_workspace)
{
  if (!sandbox_available_()) { return; }
  const string ws      = make_tmpdir_("ws");
  const string outside = make_tmpdir_("out");
  ASSERT_TRUE(!ws.empty() && !outside.empty());

  ShellToolOptions o;
  o.workspace = ws;
  McpTool t = make_shell_tool(o);

  const string cmd =
      "{\"command\": \"echo ok > inside.txt; "
      "echo no > " + outside + "/escape.txt\"}";
  const string out = t.handler(cmd);
  FlexData fd = FlexData::from_json(out);
  EXPECT_TRUE(fd.is_object());
  EXPECT_TRUE(fs::exists(ws + "/inside.txt"));
  EXPECT_FALSE(fs::exists(outside + "/escape.txt"));

  error_code ec;
  fs::remove_all(ws, ec);
  fs::remove_all(outside, ec);
}

TEST(shell_tool, accepts_command_as_array_of_lines)
{
  if (!sandbox_available_()) { return; }
  const string ws = make_tmpdir_("arr");
  ASSERT_TRUE(!ws.empty());

  ShellToolOptions o;
  o.workspace = ws;
  McpTool t = make_shell_tool(o);

  const string out = t.handler(
      "{\"command\": [\"a=3\", \"b=4\", \"echo $((a+b))\"]}");
  FlexData fd = FlexData::from_json(out);
  EXPECT_TRUE(fd.is_object());
  auto obj = fd.as_object();
  EXPECT_TRUE(obj.at("exit_code").as_int(-1) == 0);
  EXPECT_TRUE(string(obj.at("stdout").as_string("")).find("7")
              != string::npos);

  error_code ec;
  fs::remove_all(ws, ec);
}

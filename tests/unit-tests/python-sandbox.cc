// python-sandbox.cc -- tests for the seatbelt-sandboxed Python runner and
// the `run_python` MCP tool. macOS only (seatbelt). Each case runs a real
// python3 through sandbox-exec, so they self-skip when neither is present.

#include "minitest.h"

#include "common/flex-data.h"
#include "generative-models/shared/mcp/mcp-tools.h"
#include "generative-models/shared/mcp/python-sandbox.h"

#include <string>
#include <unistd.h>

using namespace std;
using namespace vpipe;

namespace {

// True when the sandbox can actually run here (macOS + tools present).
bool sandbox_available_()
{
  return ::access("/usr/bin/sandbox-exec", X_OK) == 0
      && (::access("/opt/homebrew/bin/python3", X_OK) == 0
          || ::access("/usr/local/bin/python3", X_OK) == 0
          || ::access("/usr/bin/python3", X_OK) == 0);
}

}  // namespace

TEST(python_sandbox, basic_compute)
{
  if (!sandbox_available_()) { return; }
  auto r = run_python_sandboxed("print(6*7)");
  EXPECT_TRUE(r.ok);
  EXPECT_FALSE(r.timed_out);
  EXPECT_TRUE(r.exit_code == 0);
  EXPECT_TRUE(r.output.find("42") != string::npos);
}

TEST(python_sandbox, network_is_blocked)
{
  if (!sandbox_available_()) { return; }
  // DNS / connect must fail inside the sandbox; the script prints OK only
  // if it somehow reached the network.
  const char* code =
      "import socket\n"
      "try:\n"
      "    socket.create_connection(('example.com', 80), timeout=5)\n"
      "    print('NET_OK')\n"
      "except Exception as e:\n"
      "    print('NET_BLOCKED')\n";
  auto r = run_python_sandboxed(code);
  EXPECT_TRUE(r.ok);
  EXPECT_TRUE(r.output.find("NET_BLOCKED") != string::npos);
  EXPECT_TRUE(r.output.find("NET_OK") == string::npos);
}

TEST(python_sandbox, write_outside_scratch_denied)
{
  if (!sandbox_available_()) { return; }
  // Writing under /tmp root (outside the ephemeral scratch dir) must be
  // denied by the profile.
  const char* code =
      "try:\n"
      "    open('/tmp/vpipe-sbx-escape.txt','w').write('x')\n"
      "    print('WROTE')\n"
      "except Exception:\n"
      "    print('WRITE_DENIED')\n";
  auto r = run_python_sandboxed(code);
  EXPECT_TRUE(r.ok);
  EXPECT_TRUE(r.output.find("WRITE_DENIED") != string::npos);
}

TEST(python_sandbox, home_reads_denied)
{
  if (!sandbox_available_()) { return; }
  const char* real_home = ::getenv("HOME");
  if (!real_home || !*real_home) { return; }
  // Reading a file under the real home must be denied. Use the shell rc
  // as a stand-in secret; the profile denies the whole home subpath.
  string code =
      string("import os\n")
      + "p = " + "'" + string(real_home) + "/.zshrc'\n"
      + "try:\n"
      + "    open(p).read()\n"
      + "    print('READ_OK')\n"
      + "except Exception:\n"
      + "    print('READ_DENIED')\n";
  auto r = run_python_sandboxed(code);
  EXPECT_TRUE(r.ok);
  // Either the file doesn't exist or the read is denied; in both cases the
  // sensitive content never surfaces. We assert it was NOT read.
  EXPECT_TRUE(r.output.find("READ_OK") == string::npos);
}

TEST(python_sandbox, timeout_kills_runaway)
{
  if (!sandbox_available_()) { return; }
  PythonSandboxOptions o;
  o.timeout_ms = 800;
  auto r = run_python_sandboxed("while True:\n    pass\n", o);
  EXPECT_TRUE(r.ok);
  EXPECT_TRUE(r.timed_out);
}

TEST(python_sandbox, scratch_write_allowed)
{
  if (!sandbox_available_()) { return; }
  // Writing within the scratch cwd is permitted and readable back.
  const char* code =
      "open('out.txt','w').write('hello')\n"
      "print(open('out.txt').read())\n";
  auto r = run_python_sandboxed(code);
  EXPECT_TRUE(r.ok);
  EXPECT_TRUE(r.exit_code == 0);
  EXPECT_TRUE(r.output.find("hello") != string::npos);
}

TEST(python_sandbox, tool_wraps_result_as_json)
{
  if (!sandbox_available_()) { return; }
  McpTool t = make_python_tool();
  EXPECT_TRUE(t.name == "run_python");
  EXPECT_TRUE(!!t.handler);
  const string out = t.handler("{\"code\": \"print(1+1)\"}");
  FlexData fd = FlexData::from_json(out);
  EXPECT_TRUE(fd.is_object());
  auto o = fd.as_object();
  EXPECT_TRUE(o.contains("stdout"));
  EXPECT_TRUE(string(o.at("stdout").as_string("")).find("2")
              != string::npos);
  EXPECT_TRUE(o.at("exit_code").as_int(-1) == 0);
}

TEST(python_sandbox, tool_missing_code_reports_error)
{
  McpTool t = make_python_tool();
  const string out = t.handler("{}");
  FlexData fd = FlexData::from_json(out);
  EXPECT_TRUE(fd.is_object());
  EXPECT_TRUE(fd.as_object().contains("error"));
}

TEST(python_sandbox, tool_accepts_code_as_array_of_lines)
{
  if (!sandbox_available_()) { return; }
  McpTool t = make_python_tool();
  // Multi-line program supplied as an array of lines (indentation
  // preserved per line); joined with newlines and executed.
  const string out = t.handler(
      "{\"code\": [\"total = 0\", \"for i in range(4):\", "
      "\"    total += i\", \"print('sum', total)\"]}");
  FlexData fd = FlexData::from_json(out);
  EXPECT_TRUE(fd.is_object());
  auto o = fd.as_object();
  EXPECT_TRUE(o.at("exit_code").as_int(-1) == 0);
  EXPECT_TRUE(string(o.at("stdout").as_string("")).find("sum 6")
              != string::npos);
}

TEST(python_sandbox, tool_array_lines_avoids_quote_escaping)
{
  if (!sandbox_available_()) { return; }
  McpTool t = make_python_tool();
  // A line that itself contains double quotes -- as an array element the
  // model only escapes the JSON string, not a nested-in-a-blob string.
  const string out = t.handler(
      "{\"code\": [\"print(\\\"quoted ok\\\")\"]}");
  FlexData fd = FlexData::from_json(out);
  EXPECT_TRUE(fd.is_object());
  EXPECT_TRUE(string(fd.as_object().at("stdout").as_string(""))
              .find("quoted ok") != string::npos);
}

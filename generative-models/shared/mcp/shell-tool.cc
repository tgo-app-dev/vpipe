#include "generative-models/shared/mcp/shell-tool.h"

#include "common/command-sandbox.h"
#include "common/flex-data.h"

#include <exception>
#include <string>
#include <utility>

using namespace std;

namespace vpipe {

namespace {

// Read the `command` argument as either a single string or an ARRAY of
// lines joined with '\n' (same shape as run_python's `code`, which lets a
// model emit multi-line input without quote-escaping mistakes). Returns ""
// when absent / wrong type.
string
extract_command_(const FlexData& args)
{
  if (!args.is_object()) {
    return {};
  }
  auto o = args.as_object();
  if (!o.contains("command")) {
    return {};
  }
  const FlexData cmd = o.at("command");
  if (cmd.is_string()) {
    return string(cmd.get_string());
  }
  if (cmd.is_array()) {
    string out;
    auto arr = cmd.as_array();
    for (size_t i = 0; i < arr.size(); ++i) {
      if (i) {
        out.push_back('\n');
      }
      const FlexData line = arr[i];
      out += line.is_string() ? string(line.get_string())
                              : string(line.as_string(""));
    }
    return out;
  }
  return {};
}

}  // namespace

McpTool
make_shell_tool(const ShellToolOptions& opts)
{
  McpTool t;
  t.name = "run_shell";
  t.description =
      "Run a short shell command (/bin/sh -c) in a locked-down sandbox "
      "(no network, writes confined to your workspace, CPU/time limits) "
      "and return its combined stdout/stderr. Use for quick file/text "
      "operations in the workspace; it shares the workspace with the file "
      "tools.";
  t.parameters_json =
      "{\"type\":\"object\",\"properties\":{\"command\":{\"type\":"
      "[\"string\",\"array\"],\"items\":{\"type\":\"string\"},"
      "\"description\":\"Shell command to run via /bin/sh -c: either a "
      "single string, or an array of lines (joined with newlines).\"}},"
      "\"required\":[\"command\"]}";
  t.handler = [opts](const string& args_json) -> string {
    string command;
    try {
      command = extract_command_(FlexData::from_json(args_json));
    } catch (const std::exception&) {
      // fall through: empty command -> error result below
    }
    FlexData res = FlexData::make_object();
    auto ro = res.as_object();
    if (command.empty()) {
      ro.insert("error", FlexData::make_string(
          "missing 'command' argument (string or array of lines)"));
      return res.to_json(false);
    }

    CommandSandboxSpec spec;
    spec.enabled          = true;
    spec.allow_network    = opts.allow_network;
    spec.exec_allow       = opts.exec_allow;
    spec.cwd              = opts.workspace;
    spec.timeout_ms       = opts.timeout_ms;
    spec.cpu_seconds      = opts.cpu_seconds;
    spec.address_space_mb = opts.address_space_mb;
    spec.file_size_mb     = opts.file_size_mb;
    spec.max_procs        = opts.max_procs;
    if (!opts.workspace.empty()) {
      spec.writable_roots.push_back(opts.workspace);
    }
    for (const auto& w : opts.extra_writable) {
      spec.writable_roots.push_back(w);
    }

    // Capture combined stdout+stderr into one buffer, in line order, up to
    // the output cap. `truncated` records that we dropped later output.
    string output;
    bool   truncated = false;
    auto sink = [&output, &truncated, &opts](string_view line) {
      if (output.size() >= opts.max_output_bytes) {
        truncated = true;
        return;
      }
      string chunk(line);
      chunk.push_back('\n');
      const size_t room = opts.max_output_bytes - output.size();
      if (chunk.size() <= room) {
        output += chunk;
      } else {
        output.append(chunk, 0, room);
        truncated = true;
      }
    };
    CommandSandboxIo io;
    io.on_stdout_line = sink;
    io.on_stderr_line = sink;

    const CommandSandboxResult rc = run_shell_command(command, spec, io);
    if (!rc.ok) {
      ro.insert("error", FlexData::make_string(
          rc.error.empty() ? string("spawn failed") : rc.error));
      return res.to_json(false);
    }
    ro.insert("stdout", FlexData::make_string(output));
    ro.insert("exit_code", FlexData::make_int(rc.exit_code));
    if (rc.timed_out) {
      ro.insert("timed_out", FlexData::make_bool(true));
    }
    if (rc.signaled) {
      ro.insert("signal", FlexData::make_int(rc.term_signal));
    }
    if (truncated) {
      ro.insert("truncated", FlexData::make_bool(true));
    }
    return res.to_json(false);
  };
  return t;
}

}


#ifndef VPIPE_GENERATIVE_MODELS_SHELL_TOOL_H
#define VPIPE_GENERATIVE_MODELS_SHELL_TOOL_H

#include "generative-models/shared/mcp/mcp-tools.h"

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace vpipe {

// Knobs for the MCP `run_shell` tool. Like `run_python`, one call runs a
// short command under a macOS seatbelt with conservative defaults; the
// difference is the writable root is a PERSISTENT workspace (shared with
// the file tools) rather than an ephemeral scratch dir, so shell and file
// tools operate on the same files.
struct ShellToolOptions {
  // The only directory the command may write to (the chat file-tool
  // workspace). Reads outside stay permissive except the real $HOME.
  std::filesystem::path              workspace;
  // Extra writable roots (e.g. the session --white-list-path grants).
  std::vector<std::filesystem::path> extra_writable;
  bool        allow_network    = false;      // seatbelt denies net by default
  // Optional program whitelist (kernel-enforced); empty => any program.
  std::vector<std::string>           exec_allow;

  int         timeout_ms       = 15000;      // wall-clock; SIGKILL on expiry
  std::size_t max_output_bytes = 64 * 1024;  // combined stdout+stderr cap
  long        cpu_seconds      = 10;
  long        address_space_mb = 2048;
  long        file_size_mb     = 64;
  long        max_procs        = 64;
};

// Build the MCP `run_shell` tool backed by run_shell_command(). The handler
// expects {"command": "<sh>"} (a string, or an array of lines joined with
// newlines) and returns {stdout, exit_code, [timed_out], [signal],
// [truncated], [error]}. `opts` is captured, so `workspace` must outlive
// the registry's handlers.
McpTool make_shell_tool(const ShellToolOptions& opts);

}

#endif

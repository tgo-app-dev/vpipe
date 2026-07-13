#ifndef VPIPE_GENERATIVE_MODELS_PYTHON_SANDBOX_H
#define VPIPE_GENERATIVE_MODELS_PYTHON_SANDBOX_H

#include "generative-models/shared/mcp/mcp-tools.h"

#include <cstddef>
#include <string>

namespace vpipe {

// Knobs for one sandboxed Python run. Defaults are conservative -- a
// chat tool wants a fast calculator / data-muncher, not a long job.
struct PythonSandboxOptions {
  int    timeout_ms        = 10000;        // wall-clock; SIGKILL on expiry
  std::size_t max_output_bytes = 64 * 1024;// combined stdout+stderr cap
  long   cpu_seconds       = 10;           // RLIMIT_CPU (hard = +2s)
  long   address_space_mb  = 2048;         // RLIMIT_AS (0 => leave alone)
  long   file_size_mb      = 64;           // RLIMIT_FSIZE (0 => leave alone)
  long   max_procs         = 64;           // RLIMIT_NPROC (0 => leave alone)
};

// Outcome of a sandboxed run. `ok` means the child spawned and we
// collected an outcome (NOT that the script exited 0 -- check exit_code
// / timed_out / signaled for that). `error` carries a harness-level
// failure (python or sandbox-exec missing, spawn failed, non-macOS).
struct PythonSandboxResult {
  bool        ok = false;
  int         exit_code = -1;
  bool        timed_out = false;
  bool        signaled = false;
  int         term_signal = 0;
  std::string output;              // combined stdout+stderr, capped
  bool        output_truncated = false;
  std::string error;
};

// Run `code` as `python3 -I -B -c <code>` inside a macOS seatbelt
// sandbox with defense-in-depth:
//   * network denied
//   * writes confined to an ephemeral per-call scratch dir (which is
//     also $HOME/$TMPDIR for the child)
//   * reads of the invoking user's real $HOME denied (blocks ~/.ssh,
//     ~/.aws, keychains, dotfile secrets) while system + python stay
//     readable so CPython boots
//   * rlimits (CPU / address space / file size / nproc / no core)
//   * a wall-clock timeout that SIGKILLs the whole process group
//   * a minimal environment (no inherited secrets), no stdin, and an
//     output-size cap.
// The scratch dir is created and removed per call. macOS only; other
// platforms return ok=false with an error.
PythonSandboxResult
run_python_sandboxed(const std::string&          code,
                     const PythonSandboxOptions& opts = {});

// Build the MCP `run_python` tool backed by run_python_sandboxed(). The
// handler expects {"code": "<python source>"} and returns a JSON object
// {stdout, exit_code, timed_out, truncated, [error]}.
McpTool make_python_tool(const PythonSandboxOptions& opts = {});

}

#endif

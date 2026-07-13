#ifndef VPIPE_COMMAND_SANDBOX_H
#define VPIPE_COMMAND_SANDBOX_H

#include <cstddef>
#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace vpipe {

// Resource / policy knobs for one `sh -c <command>` run. All limits default
// to "off" (0 => leave the OS default): the shell STAGE streams unbounded,
// while the MCP `run_shell` tool dials in conservative caps.
//
// When `enabled`, the command runs under a macOS seatbelt profile that
// contains its SIDE EFFECTS: writes are denied everywhere except under
// `writable_roots` (+ /dev), the invoking user's real $HOME is unreadable
// (so secrets can't be exfiltrated), and -- unless `allow_network` -- all
// networking is denied. Reads of the system stay permissive so ordinary
// programs still run (a read-deny profile is fragile: the loader, shared
// caches and frameworks all need broad read access). When `enabled` is
// false the command runs natively (no seatbelt) -- the CLI / any host whose
// session is not filesystem-sandboxed.
struct CommandSandboxSpec {
  bool                                enabled = false;
  // Absolute directories the child may WRITE to (recursively). Populate
  // with the session sandbox root + any --white-list-path grants.
  std::vector<std::filesystem::path>  writable_roots;
  bool                                allow_network = false;

  // Optional command whitelist. When non-empty AND `enabled`, the seatbelt
  // denies process-exec* except the shell itself and these programs, so a
  // command can only launch a vetted set of binaries (a kernel-enforced
  // gate, not a fragile command-string parse). Each entry is either an
  // absolute path or a bare name resolved against a standard set of bin
  // dirs (/usr/bin, /bin, /usr/local/bin, /opt/homebrew/bin, ...). Empty
  // => any program may run (writes / network stay confined regardless).
  std::vector<std::string>            exec_allow;

  // Working directory for the child (created when `enabled` and missing).
  // Empty => inherit the parent CWD.
  std::filesystem::path               cwd;

  // Resource limits (0 => leave the OS default).
  int   timeout_ms       = 0;   // wall-clock; SIGKILL the group on expiry
  long  cpu_seconds      = 0;   // RLIMIT_CPU (hard = +2s)
  long  address_space_mb = 0;   // RLIMIT_AS
  long  file_size_mb     = 0;   // RLIMIT_FSIZE
  long  max_procs        = 0;   // RLIMIT_NPROC
};

// Line-oriented I/O hooks for a run. The runner splits the child's stdout
// and stderr into lines (trailing newline stripped) and invokes the
// matching hook as each completes, flushing any unterminated tail at EOF.
//
// `provide_stdin` (optional) is called on a private thread to source the
// child's stdin one line at a time; the runner appends '\n' and writes it.
// Its `cancel` argument becomes true once the child is gone or the run is
// aborting -- a blocking source (e.g. an interactive UI read) must poll it
// and give up promptly. Returning false closes the child's stdin (EOF).
// When `provide_stdin` is null the child's stdin is /dev/null.
//
// `should_cancel` (optional) is polled by the drain loop to abort a running
// child (its process group is SIGKILLed). Any hook may be null.
struct CommandSandboxIo {
  std::function<void(std::string_view line)> on_stdout_line;
  std::function<void(std::string_view line)> on_stderr_line;
  std::function<bool(std::string&                  line,
                     const std::function<bool()>&  cancel)>
                                              provide_stdin;
  std::function<bool()>                       should_cancel;
};

struct CommandSandboxResult {
  bool        ok = false;        // child spawned + reaped (NOT that exit==0)
  int         exit_code = -1;
  bool        timed_out = false;
  bool        signaled = false;
  int         term_signal = 0;
  bool        canceled = false;  // should_cancel fired
  std::string error;             // harness-level failure (spawn, or a
                                 // sandboxed run on a non-macOS host);
                                 // empty on a clean run
};

// Run `sh -c command` with the given confinement + I/O policy. Blocks the
// calling thread until the child exits, is killed (timeout / cancel), or
// its output closes. `provide_stdin`, if set, runs on an internal thread
// joined before return.
CommandSandboxResult
run_shell_command(const std::string&        command,
                  const CommandSandboxSpec& spec,
                  const CommandSandboxIo&   io);

}  // namespace vpipe

#endif  // VPIPE_COMMAND_SANDBOX_H

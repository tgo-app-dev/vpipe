#include "stages/shell-stage.h"
#include "common/beat-payload-intf.h"
#include "common/command-sandbox.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include <filesystem>
#include <utility>

using namespace std;

namespace vpipe {

ShellStage::ShellStage(const SessionContextIntf* s,
                       string                    id,
                       vector<InEdge>            iports,
                       FlexData                  config)
  : TypedStage<ShellStage>(s, std::move(id), std::move(iports),
                           std::move(config))
{
  _command = attr_str("command");
  if (_command.empty()) {
    fail_config(fmt(
        "ShellStage('{}'): config.command is required (non-empty string)",
        this->id()));
  }
  _forward_stdin = attr_bool("forward_stdin");
  _allow_network = attr_bool("allow_network");
  _timeout_ms    = attr_uint("timeout_ms");
  // `allow`: string array of programs the sandboxed command may exec. Bind
  // the owning FlexData to a local before as_array() (the view dangles
  // otherwise).
  FlexData allow = attr("allow");
  if (allow.is_array()) {
    auto arr = allow.as_array();
    _allow_commands.reserve(arr.size());
    for (size_t i = 0; i < arr.size(); ++i) {
      const FlexData v = arr[i];
      if (v.is_string()) {
        string name(v.get_string());
        if (!name.empty()) {
          _allow_commands.push_back(std::move(name));
        }
      }
    }
  }
}

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "command", .type = ConfigType::String, .required = true,
   .doc = "shell command passed verbatim to /bin/sh -c"},
  {.key = "forward_stdin", .type = ConfigType::Bool,
   .doc = "feed the UI's getline() to the command's stdin (interactive "
          "programs); otherwise stdin is /dev/null",
   .def_bool = false},
  {.key = "allow", .type = ConfigType::Array,
   .doc = "when the session is sandboxed, the only programs the command "
          "may exec (names or absolute paths); empty => any program (its "
          "writes/network stay confined regardless)"},
  {.key = "allow_network", .type = ConfigType::Bool,
   .doc = "permit network access when the session is sandboxed "
          "(no effect without the sandbox)",
   .def_bool = false},
  {.key = "timeout_ms", .type = ConfigType::Uint,
   .doc = "wall-clock timeout; the command's process group is killed on "
          "expiry. 0 => no limit",
   .def_uint = 0},
};
const PortSpec kIports[] = {
  {.name = "trigger", .doc = "any beat; each one runs the command",
   .type = nullptr, .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "shell",
  .doc       = "Sink: runs a shell command via /bin/sh on every input "
               "beat (any payload); stdout->info, stderr->warn, optional "
               "getline stdin. Sandboxed under the session file sandbox. "
               "EOS ends the stage.",
  .display_name = "Shell",
  .category  = StageCategory::Control,
  .iports    = kIports,
  .oports    = {},
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
ShellStage::spec() const noexcept
{
  return kSpec;
}

Job
ShellStage::process(RuntimeContext& ctx)
{
  auto t = co_await ctx.read(0);
  if (!t) {
    ctx.signal_done();
    co_return;
  }

  const SessionContextIntf* sess = session();

  CommandSandboxSpec spec;
  spec.enabled       = sess && sess->fs_sandboxed();
  spec.allow_network = _allow_network;
  spec.exec_allow    = _allow_commands;
  spec.timeout_ms    = static_cast<int>(_timeout_ms);
  if (spec.enabled) {
    // Confine writes to the sandbox root + any whitelisted mounts, and run
    // with the sandbox root as the working directory.
    std::filesystem::path root = sess->fs_sandbox_root();
    if (!root.empty()) {
      spec.writable_roots.push_back(root);
      spec.cwd = root;
    }
    for (const auto& w : sess->fs_whitelist()) {
      spec.writable_roots.push_back(w);
    }
  }

  CommandSandboxIo io;
  io.on_stdout_line = [sess](string_view line) {
    if (sess) {
      sess->info(fmt("{}", string(line)));
    }
  };
  io.on_stderr_line = [sess](string_view line) {
    if (sess) {
      sess->warn(fmt("{}", string(line)));
    }
  };
  io.should_cancel = [&ctx]() { return ctx.stop_requested(); };
  if (_forward_stdin) {
    io.provide_stdin =
        [sess](string& line, const function<bool()>& cancel) -> bool {
      if (!sess) {
        return false;
      }
      UiInputStatus st = sess->getline(fmt(""), line, cancel);
      return st == UiInputStatus::Ok;
    };
  }

  const CommandSandboxResult rc =
      run_shell_command(_command, spec, io);

  _invocations.fetch_add(1, std::memory_order_acq_rel);

  if (!rc.ok) {
    _last_status.store(-1, std::memory_order_release);
    session()->warn(fmt(
        "ShellStage('{}'): failed to run '{}': {}",
        this->id(), _command,
        rc.error.empty() ? string("spawn failed") : rc.error));
    co_return;
  }
  // Encode the outcome the way std::system() would: exit code, or a
  // negative signal marker for a killed child.
  const int status = rc.signaled ? -rc.term_signal : rc.exit_code;
  _last_status.store(status, std::memory_order_release);
  if (rc.timed_out) {
    session()->warn(fmt(
        "ShellStage('{}'): command timed out after {} ms and was killed",
        this->id(), _timeout_ms));
  }
}

VPIPE_REGISTER_STAGE(ShellStage)
VPIPE_REGISTER_SPEC(ShellStage, kSpec)

}

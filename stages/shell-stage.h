#ifndef VPIPE_STAGES_SHELL_STAGE_H
#define VPIPE_STAGES_SHELL_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace vpipe {

// Sink stage: 1 input (the trigger), 0 outputs. On each input beat
// (regardless of payload type) runs the configured shell command via
// /bin/sh. EOS on in-port 0 ends the stage.
//
// The command's stdout is forwarded line-by-line to the session UI's
// info() channel and its stderr to warn(), so a shell program's output
// surfaces through whatever front end is attached (e.g. the web-ui
// console) instead of the parent process's terminal. When `forward_stdin`
// is set, the child's stdin is sourced from the UI's getline() so an
// interactive program can prompt the operator; otherwise stdin is
// /dev/null.
//
// SANDBOX: when the session's filesystem sandbox is active (the web-ui
// default), the command runs under a macOS seatbelt that confines its
// side effects -- writes only under the sandbox root + any --white-list-path
// grants, no reads of the real $HOME, and (unless `allow_network`) no
// network -- so a triggered command can't touch the host outside the
// sandbox. An optional `allow` command whitelist further gates which
// programs may run. With the sandbox off (CLI), the command runs natively.
//
// Configuration (FlexData object on the 4th constructor parameter):
//   command        (string, required) -- passed verbatim to /bin/sh -c.
//   forward_stdin  (bool, default false) -- feed UI getline() to stdin.
//   allow          (array<string>) -- when sandboxed, the only programs
//                  the command may exec (names or absolute paths); empty
//                  => any (writes/network stay confined).
//   allow_network  (bool, default false) -- permit network when sandboxed.
//   timeout_ms     (uint, default 0) -- wall-clock kill; 0 => no limit.
//
// The command runs synchronously on the worker thread that delivers the
// trigger; the next trigger is not processed until it completes.
class ShellStage final : public TypedStage<ShellStage> {
public:
  static constexpr const char* kTypeName = "shell";

  ShellStage(const SessionContextIntf* session,
             std::string               id,
             std::vector<InEdge>       iports,
             FlexData                  config);

  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only accessors.
  const std::string& command() const noexcept { return _command; }
  uint64_t           invocations() const noexcept
  {
    return _invocations.load(std::memory_order_acquire);
  }
  int last_status() const noexcept
  {
    return _last_status.load(std::memory_order_acquire);
  }

private:
  std::string              _command;
  bool                     _forward_stdin = false;
  bool                     _allow_network = false;
  std::uint64_t            _timeout_ms = 0;
  std::vector<std::string> _allow_commands;
  std::atomic<uint64_t>    _invocations{0};
  std::atomic<int>         _last_status{0};
};

}

#endif

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
// Configuration (FlexData object on the 4th constructor parameter):
//   command   (string, required) -- passed verbatim to /bin/sh -c.
//
// The command runs synchronously on the worker thread that delivers
// the trigger; the next trigger is not processed until it completes.
// stderr/stdout inherit from the parent process.
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
  std::string           _command;
  std::atomic<uint64_t> _invocations{0};
  std::atomic<int>      _last_status{0};
};

}

#endif

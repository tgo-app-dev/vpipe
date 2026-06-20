#include "stages/shell-stage.h"
#include "common/beat-payload-intf.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include <cstdlib>
#include <stdexcept>
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
}

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "command", .type = ConfigType::String, .required = true,
   .doc = "shell command passed verbatim to /bin/sh -c"},
};
const PortSpec kIports[] = {
  {.name = "trigger", .doc = "any beat; each one runs the command",
   .type = nullptr, .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "shell",
  .doc       = "Sink: runs a shell command via /bin/sh on every input "
               "beat (any payload). EOS ends the stage.",
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

  // system(3) handles "command not found", quoting, redirection, etc.
  // It blocks the calling worker thread until the child exits; that
  // is acceptable for a sink whose whole purpose is "run a command on
  // every trigger".
  int rc = std::system(_command.c_str());
  _last_status.store(rc, std::memory_order_release);
  _invocations.fetch_add(1, std::memory_order_acq_rel);

  if (rc == -1) {
    session()->warn(fmt(
        "ShellStage('{}'): system() failed to spawn '{}'",
        this->id(), _command));
  }
}

VPIPE_REGISTER_STAGE(ShellStage)
VPIPE_REGISTER_SPEC(ShellStage, kSpec)

}

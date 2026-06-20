#include "stages/passthrough-stage.h"
#include "common/job.h"
#include "pipeline/runtime-context.h"
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace vpipe {

PassthroughStage::PassthroughStage(const SessionContextIntf* s,
                                   string                    id,
                                   vector<InEdge>            iports,
                                   FlexData                  config)
  : TypedStage<PassthroughStage>(s, std::move(id), std::move(iports),
                                 std::move(config))
{
  allocate_oports(spec().oports.size());
}

namespace {
const PortSpec kIports[] = {
  {.name = "in", .doc = "any beat", .type = nullptr, .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "out", .doc = "the input beat, forwarded unchanged",
   .type = nullptr, .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "passthrough",
  .doc       = "Identity 1-in/1-out stage: forwards each input beat "
               "unchanged. EOS on the input ends the stage.",
  .category  = StageCategory::Generic,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = {},
  .hidden    = true,   // built-in identity/test fixture, not user-added
};
}  // namespace

const StageSpec&
PassthroughStage::spec() const noexcept
{
  return kSpec;
}

Job
PassthroughStage::process(RuntimeContext& ctx)
{
  auto p = co_await ctx.read(0);
  if (!p) {
    ctx.signal_done();
    co_return;
  }
  co_await ctx.write(0, std::move(p));
}

VPIPE_REGISTER_STAGE(PassthroughStage)
VPIPE_REGISTER_SPEC(PassthroughStage, kSpec)

}

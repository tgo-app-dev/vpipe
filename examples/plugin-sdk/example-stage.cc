// Worked example of a vpipe plugin, built OUT OF TREE against an installed
// vpipe SDK via find_package(vpipe) + vpipe_add_plugin (see CMakeLists.txt
// and ../../docs/PLUGINS.md). It registers one stage, "example-echo",
// that forwards each input beat unchanged -- the smallest thing that
// exercises the whole contract (a TypedStage + its spec + the three C
// entry points).

#include "common/flex-data.h"
#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "plugin/plugin-abi.h"
#include "plugin/plugin-context.h"

#include <string>
#include <utility>
#include <vector>

using namespace vpipe;

namespace {

const PortSpec kIports[] = {
  {.name = "in", .doc = "any beat", .type = nullptr, .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "out", .doc = "the input beat, forwarded unchanged",
   .type = nullptr, .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name    = "example-echo",
  .doc          = "SDK example: forwards each input beat unchanged.",
  .display_name = "Example Echo",
  .category     = StageCategory::Generic,
  .iports       = kIports,
  .oports       = kOports,
};

class ExampleEchoStage : public TypedStage<ExampleEchoStage> {
public:
  static constexpr const char* kTypeName = "example-echo";

  ExampleEchoStage(const SessionContextIntf* s, std::string id,
                   std::vector<InEdge> iports, FlexData config)
    : TypedStage<ExampleEchoStage>(s, std::move(id), std::move(iports),
                                   std::move(config))
  {
    allocate_oports(spec().oports.size());
  }

  const StageSpec& spec() const noexcept override { return kSpec; }

  Job process(RuntimeContext& ctx) override
  {
    auto p = co_await ctx.read(0);
    if (!p) {
      ctx.signal_done();
      co_return;
    }
    if (ctx.has_consumers(0)) {
      co_await ctx.write(0, std::move(p));
    }
  }
};

const VpipePluginInfo kInfo = {
  VPIPE_PLUGIN_INFO_SCHEMA,
  "vpipe-sdk-example",
  "1.0.0",
  "vpipe",
  "MIT",
  "vpipe plugin SDK example: an echo stage (example-echo).",
};

void
plugin_register(vpipe::VpipePluginContext* ctx)
{
  ctx->register_stage<ExampleEchoStage>(&kSpec);
}

}  // namespace

VPIPE_PLUGIN_DEFINE(&kInfo, plugin_register)

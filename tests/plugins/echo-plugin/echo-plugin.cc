// A minimal vpipe plugin, built as a standalone MODULE .dylib linked
// against libvpipe. It lives ONLY in the plugin binary (not compiled into
// libvpipe), so its stage type "plugin-echo" enters the host registry
// only when the plugin is dlopen'd -- which is exactly what the
// plugin_loader unit test asserts. It doubles as the smallest worked
// example of the plugin contract (see docs/PLUGINS.md).

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
  .type_name = "plugin-echo",
  .doc       = "Example plugin stage: forwards each input beat unchanged. "
               "EOS on the input ends the stage.",
  .display_name = "Plugin Echo",
  .category  = StageCategory::Generic,
  .iports    = kIports,
  .oports    = kOports,
};

// An identity 1-in/1-out stage. Mirrors stages/passthrough-stage.cc.
class PluginEchoStage : public TypedStage<PluginEchoStage> {
public:
  static constexpr const char* kTypeName = "plugin-echo";

  PluginEchoStage(const SessionContextIntf* s, std::string id,
                  std::vector<InEdge> iports, FlexData config)
    : TypedStage<PluginEchoStage>(s, std::move(id), std::move(iports),
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
  "vpipe-echo-example",
  "1.0.0",
  "vpipe",
  "MIT",
  "Example vpipe plugin: an echo stage (plugin-echo).",
};

void
plugin_register(vpipe::VpipePluginContext* ctx)
{
  ctx->register_stage<PluginEchoStage>(&kSpec);
}

}  // namespace

VPIPE_PLUGIN_DEFINE(&kInfo, plugin_register)

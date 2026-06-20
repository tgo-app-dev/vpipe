#include "stages/call-stage.h"
#include "common/graph.h"
#include "common/job.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "pipeline/runtime-context.h"
#include <cstdint>
#include <utility>

using namespace std;

namespace vpipe {

CallStage::CallStage(const SessionContextIntf* s,
                     string                    id,
                     vector<InEdge>            iports,
                     FlexData                  config)
  : TypedStage<CallStage>(s, std::move(id), std::move(iports),
                          std::move(config))
{
  // Config validation is deferred to launch (see Stage::fail_config).
  // Structural validation (the referenced pipeline existing, port-count
  // agreement, recursion) happens later in initialize(), not here.
  const FlexData& cfg = this->config();
  if (!cfg.is_object()) {
    fail_config(fmt(
        "CallStage('{}'): config must be an object",
        this->id()));
  }
  FlexData empty_obj = FlexData::make_object();
  auto root = (cfg.is_object() ? cfg : empty_obj).as_object();

  if (root.contains("pipeline")) {
    FlexData pid_v = root.at("pipeline");
    if (pid_v.is_string()) {
      _pipeline_id = string(pid_v.get_string());
    } else {
      fail_config(fmt(
          "CallStage('{}'): config.pipeline must be a string",
          this->id()));
    }
  }
  if (_pipeline_id.empty()) {
    fail_config(fmt(
        "CallStage('{}'): config.pipeline is required (non-empty string)",
        this->id()));
  }

  uint64_t n_oports = 0;
  if (root.contains("num_oports")) {
    FlexData no_v = root.at("num_oports");
    if (no_v.is_uint()) {
      n_oports = no_v.get_uint();
    } else if (no_v.is_int()) {
      int64_t v = no_v.get_int();
      if (v < 0) {
        fail_config(fmt(
            "CallStage('{}'): config.num_oports must be >= 0",
            this->id()));
      } else {
        n_oports = static_cast<uint64_t>(v);
      }
    } else {
      fail_config(fmt(
          "CallStage('{}'): config.num_oports must be an integer",
          this->id()));
    }
  } else {
    fail_config(fmt(
        "CallStage('{}'): config.num_oports is required",
        this->id()));
  }

  allocate_oports(static_cast<unsigned>(n_oports));
}

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "pipeline", .type = ConfigType::String, .required = true,
   .doc = "id of the callee pipeline to inline"},
  {.key = "num_oports", .type = ConfigType::Any, .required = true,
   .doc = "output-port count; must equal callee.num_oports()"},
};
// Ports are structural and forwarded: the call-stage's iport/oport
// counts are taken from the wiring + config.num_oports, and their
// payload types match whatever the inlined callee carries. They are
// not enumerable here, so the spec declares no ports.
const StageSpec kSpec = {
  .type_name = "call",
  .doc       = "Invokes another pipeline as a function: the call-stage's "
               "iports/oports are wired to the callee's iports/oports and "
               "the runtime inlines the callee at launch. Structural -- it "
               "never runs.",
  .category  = StageCategory::Control,
  .iports    = {},
  .oports    = {},
  .attrs     = kAttrs,
  .hidden    = true,   // structural: inlined at launch, not user-added
};
}  // namespace

const StageSpec&
CallStage::spec() const noexcept
{
  return kSpec;
}

// All three lifecycle hooks are no-ops for a CallStage. The runtime
// erases this stage during the inlining pre-phase (it never goes
// into the live driver list), so these bodies are only reached if a
// future code path bypasses inlining; in that case we want them to
// terminate cleanly rather than hang the pipeline.
Job
CallStage::initialize(RuntimeContext& /*ctx*/)
{
  co_return;
}

Job
CallStage::process(RuntimeContext& ctx)
{
  ctx.signal_done();
  co_return;
}

Job
CallStage::drain(RuntimeContext& /*ctx*/)
{
  co_return;
}

const Graph*
lexical_lookup_pipeline(const Graph* origin, string_view name)
{
  // Match C++ name lookup: start at the innermost scope (the
  // origin's own children) and walk outward through each enclosing
  // scope until we find a match or run out.
  for (const Graph* g = origin; g != nullptr; g = g->parent_graph()) {
    const Graph* hit = g->graph(string(name));
    if (hit) {
      return hit;
    }
  }
  return nullptr;
}

VPIPE_REGISTER_STAGE(CallStage)
VPIPE_REGISTER_SPEC(CallStage, kSpec)

}

// Plugins the host must REFUSE, one per way a handshake can fail. Built
// three times from this file (see the root CMakeLists), each with a
// TypedStage that registers itself from a static initialiser -- the thing
// a refusal has to leave no trace of.
//
//   HANDSHAKE_OLD_ABI          reports plugin ABI 7, outside the window:
//                              refused from the FILE, never dlopen'd.
//   HANDSHAKE_NO_CONSTANT      the pre-8 shape, functions only, no
//                              `vpipe_plugin_abi`: refused from the file.
//   HANDSHAKE_MISSING_FEATURE  a current plugin requiring a feature this
//                              host lacks: dlopen'd, refused after, and
//                              its self-registered stage withdrawn.

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

#if defined(HANDSHAKE_OLD_ABI)
#define STAGE_NAME "plugin-handshake-old-abi"
#elif defined(HANDSHAKE_NO_CONSTANT)
#define STAGE_NAME "plugin-handshake-no-constant"
#else
#define STAGE_NAME "plugin-handshake-missing-feature"
#endif

class HandshakeStage : public TypedStage<HandshakeStage> {
public:
  static constexpr const char* kTypeName = STAGE_NAME;
  using TypedStage::TypedStage;
  Job process(RuntimeContext& ctx) override
  {
    ctx.signal_done();
    co_return;
  }
};

// Odr-use the id so the static registration really runs at load.
const StageTypeId kId = TypedStage<HandshakeStage>::type_id();

void
register_(VpipePluginContext* c)
{
  c->register_stage<HandshakeStage>();
}

const char* const kRequires[] = {"no-such-feature/1", nullptr};

const VpipePluginInfo kInfo = {
  VPIPE_PLUGIN_INFO_SCHEMA, "handshake-test", "0.0.0", "vpipe tests",
  "Apache-2.0", "a plugin the host must refuse", kRequires,
};

}  // namespace

#if defined(HANDSHAKE_OLD_ABI)
extern "C" __attribute__((visibility("default"), used))
const uint32_t vpipe_plugin_abi = 7u;
extern "C" __attribute__((visibility("default")))
uint32_t vpipe_plugin_abi_version(void) { return 7u; }
extern "C" __attribute__((visibility("default")))
const VpipePluginInfo* vpipe_plugin_info(void) { return &kInfo; }
extern "C" __attribute__((visibility("default")))
void vpipe_plugin_register(VpipePluginContext* c) { register_(c); }
#elif defined(HANDSHAKE_NO_CONSTANT)
extern "C" __attribute__((visibility("default")))
uint32_t vpipe_plugin_abi_version(void) { return VPIPE_PLUGIN_ABI_VERSION; }
extern "C" __attribute__((visibility("default")))
const VpipePluginInfo* vpipe_plugin_info(void) { return &kInfo; }
extern "C" __attribute__((visibility("default")))
void vpipe_plugin_register(VpipePluginContext* c) { register_(c); }
#else
VPIPE_PLUGIN_DEFINE(&kInfo, register_)
#endif

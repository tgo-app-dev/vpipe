#include "stages/load-text-stage.h"

#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace vpipe {

LoadTextStage::LoadTextStage(const SessionContextIntf* s,
                             string                    id,
                             vector<InEdge>            iports,
                             FlexData                  config)
  : TypedStage<LoadTextStage>(s, std::move(id), std::move(iports),
                              std::move(config))
{
  // Validation is deferred to launch (Stage::fail_config): the stage must
  // construct for any config so a graph can be built/edited before a path is
  // supplied.
  const FlexData& cfg = this->config();
  if (cfg.is_object()) {
    auto root = cfg.as_object();
    if (root.contains("path")) {
      FlexData p = root.at("path");
      if (p.is_string()) {
        _paths.emplace_back(string(p.as_string("")));
      } else if (p.is_array()) {
        for (FlexData e : p.as_array()) {
          if (e.is_string()) {
            _paths.emplace_back(string(e.as_string("")));
          } else {
            fail_config(fmt(
                "LoadTextStage('{}'): config.path array entries must be "
                "strings", this->id()));
          }
        }
      } else {
        fail_config(fmt(
            "LoadTextStage('{}'): config.path must be a string or array of "
            "strings", this->id()));
      }
    }
  }
  if (_paths.empty()) {
    fail_config(fmt(
        "LoadTextStage('{}'): config.path is required (non-empty string or "
        "array of strings)", this->id()));
  }
  // Confine every source path to the session file sandbox (a no-op when
  // the sandbox is disabled). An escape fails the stage's config.
  for (auto& p : _paths) {
    p = confine_local_(p, /*for_write=*/false);
  }
  for (const auto& p : _paths) {
    if (p.empty()) {
      fail_config(fmt(
          "LoadTextStage('{}'): config.path contains an empty entry",
          this->id()));
    }
  }
  allocate_oports(spec().oports.size());
}

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "path", .type = ConfigType::Any, .required = true,
   .doc = "text file path string or array of strings",
   .is_path = true, .path_filter = "text"},
};
const PortSpec kIports[] = {
  {.name = "trigger", .doc = "optional pacing beat (e.g. chrono); each one "
                             "reads + emits the next file",
   .type = nullptr, .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "text", .doc = "file contents as one FlexData string payload",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "load-text",
  .doc       = "Source: reads text files from the filesystem and emits each "
               "file's contents as a FlexData string (the text-input format). "
               "With a wired trigger iport one beat emits one file; unwired it "
               "emits all then ends.",
  .display_name = "Load Text",
  .category  = StageCategory::Text,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
LoadTextStage::spec() const noexcept
{
  return kSpec;
}

Job
LoadTextStage::process(RuntimeContext& ctx)
{
  if (_next >= _paths.size()) {
    ctx.signal_done();
    co_return;
  }

  // Pacing: when an iport is wired, gate each emission on one upstream beat
  // (only receipt matters). EOS upstream means we stop emitting too.
  if (ctx.num_iports() >= 1) {
    auto t = co_await ctx.read(0);
    if (!t) {
      ctx.signal_done();
      co_return;
    }
  }

  const string path = _paths[_next++];
  ifstream in(path, ios::binary);
  if (!in) {
    session()->warn(fmt(
        "LoadTextStage('{}'): failed to open '{}'", this->id(), path));
  } else {
    ostringstream ss;
    ss << in.rdbuf();
    co_await ctx.write(
        0, make_payload<FlexDataPayload>(FlexData::make_string(ss.str())));
  }

  // No iport => batch mode: signal done once the last file is emitted so the
  // driver closes our oport. With an iport wired the driver tears us down when
  // the upstream source EOSes; we never self-signal then.
  if (_next >= _paths.size() && ctx.num_iports() == 0) {
    ctx.signal_done();
  }
}

VPIPE_REGISTER_STAGE(LoadTextStage)
VPIPE_REGISTER_SPEC(LoadTextStage, kSpec)

}

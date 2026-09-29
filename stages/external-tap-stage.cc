#include "stages/external-tap-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/vpipe-format.h"
#include "stages/tensor-buffer.h"

#include <utility>

using namespace std;

namespace vpipe {

namespace {

const PortSpec kIports[] = {
  {.name = "in", .doc = "any beat", .type = nullptr},
};
const PortSpec kOports[] = {
  {.name = "out", .doc = "the same beats, in order, with any edits a "
                         "caller made in place",
   .type = nullptr},
};

const ConfigKey kAttrs[] = {
  {.key = "mode", .type = ConfigType::String,
   .doc = "\"pass\": beats flow freely and a queued read takes the next "
          "one. \"gate\": every TensorBeat waits for a read, so the "
          "pipeline runs at the caller's pace",
   .def_str = "pass"},
};

const ConfigKey kReadResults[] = {
  {.key = "seq", .type = ConfigType::Uint,
   .doc = "the beat's 1-based index through the tap this run"},
  {.key = "type", .type = ConfigType::String, .doc = "element type"},
  {.key = "shape", .type = ConfigType::Array, .doc = "the beat's shape"},
  {.key = "sideband", .type = ConfigType::Any,
   .doc = "the beat's sideband (null when it has none)"},
};
const BufferSpec kReadOut[] = {
  {.name = "data",
   .doc  = "the beat's own bytes, in place; edits travel downstream",
   .type = kTensorElementTypes, .writable = true},
};

const CommandSpec kCommands[] = {
  {.name = "read",
   .doc  = "expose the next TensorBeat in place and hold it -- the "
           "pipeline downstream waits -- until the command is closed",
   .results = kReadResults, .out = kReadOut, .holds = true},
};

const StageSpec kSpec = {
  .type_name    = "external-tap",
  .doc          = "Passthrough that lets code outside the pipeline read and "
                  "edit beats in place: a `read` command exposes the next "
                  "TensorBeat zero-copy and holds it until the command "
                  "closes. In gate mode every beat waits for a read. "
                  "Driven from the C++ / Python API.",
  .display_name = "External Tap",
  .category     = StageCategory::Control,
  .iports       = kIports,
  .oports       = kOports,
  .attrs        = kAttrs,
  .commands     = kCommands,
};

}  // namespace

ExternalTapStage::ExternalTapStage(const SessionContextIntf* session,
                                   string                    id,
                                   vector<InEdge>            iports,
                                   FlexData                  config)
  : TypedStage<ExternalTapStage>(session, std::move(id), std::move(iports),
                                 std::move(config))
{
  const string mode = attr_str("mode");
  if (mode == "gate") {
    _gate = true;
  } else if (mode != "pass") {
    fail_config(fmt("external-tap('{}'): mode '{}' is neither \"pass\" "
                    "nor \"gate\"", this->id(), mode));
  }
  allocate_oports(kSpec.oports.size());
}

const StageSpec&
ExternalTapStage::spec() const noexcept
{
  return kSpec;
}

Job
ExternalTapStage::process(RuntimeContext& ctx)
{
  unique_ptr<BeatPayloadIntf> beat = co_await ctx.read(0);
  if (!beat) {
    ctx.signal_done();
    co_return;
  }
  const auto* tb = dynamic_cast<const TensorBeatPayload*>(beat.get());
  if (tb == nullptr) {
    co_await ctx.write(0, std::move(beat));
    co_return;
  }
  const uint64_t seq = ++_seq;

  shared_ptr<StageCommand> cmd = ctx.try_command();
  if (!cmd && _gate) {
    // HOLD until someone reads it. Null means the pipeline is stopping.
    cmd = co_await ctx.next_command();
    if (!cmd) { co_return; }
  }
  if (!cmd) {
    co_await ctx.write(0, std::move(beat));
    co_return;
  }

  // The beat is SHARED with the reply for as long as the caller reads
  // it. Taken back whole when every view is gone; copied when the caller
  // kept one, so that memory stays the caller's and what moves on is
  // the beat as it was at close().
  auto hold = make_shared<unique_ptr<BeatPayloadIntf>>(std::move(beat));
  const auto& t = static_cast<const TensorBeatPayload&>(**hold);
  FlexData r = FlexData::make_object();
  {
    auto ro = r.as_object();
    ro.insert("seq", FlexData::make_uint(seq));
    ro.insert("type", FlexData::make_string(
                          element_type_name(element_type_of(t.dtype))));
    FlexData sh = FlexData::make_array();
    for (int64_t d : t.shape) {
      sh.as_array().push_back(FlexData::make_int(d));
    }
    ro.insert("shape", std::move(sh));
    ro.insert("sideband", t.sideband);
  }
  if (cmd->reply(std::move(r), {tensor_view(t, "data", hold, true)})) {
    co_await cmd->until_closed();
  }
  if (cmd->cancelled()) { co_return; }

  unique_ptr<BeatPayloadIntf> out =
      hold.use_count() == 1 ? std::move(*hold) : (*hold)->clone();
  hold.reset();
  co_await ctx.write(0, std::move(out));
  co_return;
}

VPIPE_REGISTER_STAGE(ExternalTapStage)
VPIPE_REGISTER_SPEC(ExternalTapStage, kSpec)

}

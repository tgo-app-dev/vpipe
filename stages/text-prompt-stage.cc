#include "stages/text-prompt-stage.h"

#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <string>
#include <utility>

using namespace std;

namespace vpipe {

TextPromptStage::TextPromptStage(const SessionContextIntf* s,
                                 string                    id,
                                 vector<InEdge>            iports,
                                 FlexData                  config)
  : TypedStage<TextPromptStage>(s, std::move(id), std::move(iports),
                                std::move(config))
{
  // Validation is deferred to launch (Stage::fail_config): the stage must
  // construct for any config so a graph can be built/edited before the text is
  // supplied.
  _text = attr_str("text");
  if (_text.empty()) {
    fail_config(fmt(
        "TextPromptStage('{}'): config.text is required (non-empty string)",
        this->id()));
  }
  allocate_oports(spec().oports.size());
}

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "text", .type = ConfigType::Text, .required = true,
   .doc = "the prompt text to emit as a FlexData string"},
};
const PortSpec kIports[] = {
  {.name = "trigger", .doc = "optional beat that gates re-emitting the text "
                             "(e.g. a chrono tick or a feedback loop)",
   .type = nullptr, .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "text", .doc = "the configured text as one FlexData string payload",
   .type = &typeid(FlexDataPayload),
   .tags = "text", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "text-prompt",
  .doc       = "Source: emits the configured `text` as a plain FlexData string "
               "(the text-input format, no media). One beat then done; with a "
               "trigger iport, one beat per inbound beat.",
  .display_name = "Text Prompt",
  .category  = StageCategory::Text,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
TextPromptStage::spec() const noexcept
{
  return kSpec;
}

void
TextPromptStage::reset_run_state()
{
  // Per-launch reset. Stopping a pipeline destroys the RUNTIME, not the
  // stages: only unload / re-materialize destroys a Stage, so a plain
  // Stop-then-Start re-enters initialize() with this source already
  // exhausted from the previous run. Without this it would emit nothing
  // and signal done immediately, and every stage downstream of the prompt beat
  // would sit idle while the pipeline "completed" in milliseconds.
  _done = false;
}

Job
TextPromptStage::process(RuntimeContext& ctx)
{
  // With a trigger iport wired, gate each emission on one inbound beat (the
  // payload type doesn't matter, only receipt); EOS upstream ends the stage.
  //
  // WIRED, not merely DECLARED. `{"src": "", "oport": 0}` is the spelling
  // this tree uses for an optional iport left unconnected -- the composer
  // writes it when an edge is deleted but the port row stays, and the
  // loader turns it into an InEdge{nullptr}. It still COUNTS in
  // num_iports(), so testing that alone takes the trigger branch on a
  // port nothing will ever write to: the read returns immediate EOS and
  // this source emits NOTHING, silently, with no warning. Every stage
  // downstream then sees EOS and the whole graph "completes" in
  // milliseconds having done no work.
  const bool has_trigger =
      ctx.num_iports() >= 1 && ctx.iport_connected(0);
  if (has_trigger) {
    auto t = co_await ctx.read(0);
    if (!t) {
      ctx.signal_done();
      co_return;
    }
  } else if (_done) {
    // No iport: one-shot -- already emitted, so close the oport.
    ctx.signal_done();
    co_return;
  }
  _done = true;

  co_await ctx.write(
      0, make_payload<FlexDataPayload>(FlexData::make_string(_text)));

  if (!has_trigger) {
    ctx.signal_done();
  }
}

VPIPE_REGISTER_STAGE(TextPromptStage)
VPIPE_REGISTER_SPEC(TextPromptStage, kSpec)

}

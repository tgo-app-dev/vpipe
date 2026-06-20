#include "stages/text-input-stage.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <string>
#include <utility>

using namespace std;

namespace vpipe {

TextInputStage::TextInputStage(const SessionContextIntf* s,
                               string                    id,
                               vector<InEdge>            iports,
                               FlexData                  config)
  : TypedStage<TextInputStage>(s, std::move(id), std::move(iports),
                               std::move(config))
{
  // Validation is deferred to launch (see Stage::fail_config). Every
  // field is optional, so the default configuration is valid. Attribute
  // defaults live in kSpec.attrs; attr_* resolves the value else that
  // default.
  _prompt = attr_str("prompt");
  {
    int64_t c = attr_int("count");
    if (c < 0) {
      fail_config(fmt(
          "TextInputStage('{}'): count must be >= 0 (got {})",
          this->id(), c));
    } else {
      _count = static_cast<uint64_t>(c);
    }
  }
  _present_first_without_beat = attr_bool("present_first_without_beat");

  allocate_oports(spec().oports.size());
}

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "prompt", .type = ConfigType::String,
   .doc = "text shown to the user before each read", .def_str = ">> "},
  {.key = "count", .type = ConfigType::Int,
   .doc = "lines to read; 0 = read until EOF", .def_int = 0},
  {.key = "present_first_without_beat", .type = ConfigType::Bool,
   .doc = "first prompt skips iport beat wait", .def_bool = true},
};
const PortSpec kIports[] = {
  {.name = "trigger", .doc = "optional beat that gates the next "
                             "prompt/read round (e.g. a feedback loop)",
   .type = nullptr, .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "text", .doc = "one FlexData string payload per read line",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "text-input",
  .doc       = "Source: prompts on stdout, reads a line from stdin, and "
               "emits it as a FlexData string. With a wired trigger iport "
               "each beat gates a fresh prompt/read round.",
  .display_name = "Text Input",
  .category  = StageCategory::Text,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
TextInputStage::spec() const noexcept
{
  return kSpec;
}

Job
TextInputStage::process(RuntimeContext& ctx)
{
  if (_count != 0 && _emitted >= _count) {
    ctx.signal_done();
    co_return;
  }
  if (ctx.stop_requested()) {
    ctx.signal_done();
    co_return;
  }

  // Feedback gating: when an iport is wired, wait for one beat per
  // round before presenting the next prompt -- with the optional
  // exception of the very first round (`present_first_without_beat`),
  // which prevents a self-deadlock when this stage is the trigger
  // source for a chat-style {text-input -> text-chat -> feedback-rx
  // ... feedback-tx -> text-input} loop.
  const bool wait_for_beat =
      ctx.num_iports() == 1 &&
      (_first_round_seen || !_present_first_without_beat);
  if (wait_for_beat) {
    auto trig = co_await ctx.read(0);
    if (!trig) {
      // Upstream closed -- end the input cycle.
      ctx.signal_done();
      co_return;
    }
  }
  _first_round_seen = true;

  // Prompt + read one line through the session's UI delegate. The
  // delegate owns prompt display, cooperative cancellation (it polls
  // the predicate so a pipeline stop is observed within ~50ms), and
  // the actual read -- stdin by default, or the browser console under
  // the web-ui delegate.
  string line;
  UiInputStatus st = session()->getline(
      fmt("{}", _prompt), line,
      [&ctx] { return ctx.stop_requested(); });
  if (st != UiInputStatus::Ok) {
    // Canceled (stop requested) or Eof (input closed): end the input
    // cycle cleanly so downstream consumers see EOS on their iport.
    ctx.signal_done();
    co_return;
  }
  ++_emitted;

  FlexData fd = FlexData::make_string(line);
  co_await ctx.write(0, make_payload<FlexDataPayload>(std::move(fd)));
}

VPIPE_REGISTER_STAGE(TextInputStage)
VPIPE_REGISTER_SPEC(TextInputStage, kSpec)

}

#include "pipeline/feedback-tx-stage.h"

#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/graph.h"
#include "common/vertex.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "pipeline/feedback-rx-stage.h"

#include <utility>

using namespace std;

namespace vpipe {

FeedbackTxStage::FeedbackTxStage(const SessionContextIntf* s,
                                 string                    id,
                                 vector<InEdge>            iports,
                                 FlexData                  config)
  : TypedStage<FeedbackTxStage>(s, std::move(id), std::move(iports),
                                std::move(config))
{
  // Config validation is deferred to launch (see Stage::fail_config);
  // the rx-stage lookup already happens in initialize().
  const FlexData& cfg = this->config();
  if (cfg.is_object()) {
    auto root = cfg.as_object();
    if (root.contains("from")) {
      _from_id = string(root.at("from").as_string(""));
    }
  }
  if (_from_id.empty()) {
    fail_config(fmt(
        "FeedbackTxStage('{}'): config.from (rx stage id) is required "
        "(non-empty string)", this->id()));
  }

  allocate_oports(1);
}

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "from", .type = ConfigType::String, .required = true,
   .doc = "id of the feedback-rx stage to relay"},
};
// 1 oport, type-erased: it re-emits whatever the rx side received,
// so the payload type is not known statically (left as "any").
const PortSpec kOports[] = {
  {.name = "out", .doc = "clone of the rx's latest beat (one-iter delay)",
   .type = nullptr, .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "feedback-tx",
  .doc       = "Source half of a single-clock-domain feedback pair: waits "
               "for the named feedback-rx to receive a new beat, then "
               "re-emits a clone of it on out-port 0.",
  .display_name = "Feedback Out",
  .category  = StageCategory::Control,
  .iports    = {},
  .oports    = kOports,
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
FeedbackTxStage::spec() const noexcept
{
  return kSpec;
}

Job
FeedbackTxStage::initialize(RuntimeContext&)
{
  Graph* g = this->graph();
  if (!g) {
    session()->error(fmt(
        "FeedbackTxStage('{}'): not attached to a graph", this->id()));
    co_return;
  }
  for (auto it = g->begin(); it != g->end(); ++it) {
    Vertex* v = *it;
    if (!v) { continue; }
    if (v->id() != _from_id) { continue; }
    auto* rx = dynamic_cast<FeedbackRxStage*>(v);
    if (!rx) {
      session()->error(fmt(
          "FeedbackTxStage('{}'): stage '{}' is not a feedback-rx",
          this->id(), _from_id));
      co_return;
    }
    _rx = rx;
    break;
  }
  if (!_rx) {
    session()->error(fmt(
        "FeedbackTxStage('{}'): no feedback-rx stage named '{}' in "
        "this pipeline", this->id(), _from_id));
  }
  // Per-launch reset, paired with FeedbackRxStage::initialize's
  // _seq=0: a persisted _last_seen from a prior run would otherwise
  // make wait_new_beat wait for a sequence number the reset rx never
  // reaches again.
  _last_seen = 0;
  co_return;
}

Job
FeedbackTxStage::process(RuntimeContext& ctx)
{
  if (!_rx) {
    // Misconfigured; bail without spinning. The initialize() path
    // already logged the error.
    ctx.signal_done();
    co_return;
  }

  // Block until the rx side has a beat we haven't relayed yet, OR
  // upstream closed. The awaiter resolves immediately when either
  // condition is already true.
  co_await _rx->wait_new_beat(_last_seen);

  std::shared_ptr<const BeatPayloadIntf> snap = _rx->snapshot();
  std::uint64_t now_seq = _rx->current_seq();

  if (now_seq <= _last_seen) {
    // Resumed via EOS without a new beat -- nothing more to relay.
    ctx.signal_done();
    co_return;
  }

  _last_seen = now_seq;
  if (!snap) {
    // Defensive: seq advanced but snapshot is null. Skip this round.
    co_return;
  }
  // Clone so the payload we hand to the oport is independently
  // owned. The cached snapshot stays in the rx for any future tx
  // calls (we don't take it away).
  std::unique_ptr<BeatPayloadIntf> out = snap->clone();
  co_await ctx.write(0, std::move(out));
}

VPIPE_REGISTER_STAGE(FeedbackTxStage)
VPIPE_REGISTER_SPEC(FeedbackTxStage, kSpec)

}

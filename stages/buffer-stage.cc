#include "stages/buffer-stage.h"
#include "common/job.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "pipeline/runtime-context.h"

#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace vpipe {

BufferStage::BufferStage(const SessionContextIntf* s,
                         string                    id,
                         vector<InEdge>            iports,
                         FlexData                  config)
  : TypedStage<BufferStage>(s, std::move(id), std::move(iports),
                            std::move(config))
{
  allocate_oports(spec().oports.size());
}

namespace {
// Each port is on its own clock: decoupling the sample clock from the
// emit clock is the whole point of the stage, so `in` / `advance` /
// `emit` must not be unified into one domain. `out` shares the emit
// group -- one output beat leaves per emit trigger, so it is paced by
// that clock and by nothing else.
const PortSpec kIports[] = {
  {.name = "in",
   .doc  = "any beat; one is sampled into the 1-beat buffer per "
           "advance trigger",
   .type = nullptr, .clock_group = 0},
  {.name = "advance",
   .doc  = "any beat; each one reads the next `in` beat into the "
           "buffer, replacing (dropping) the beat held before it",
   .type = nullptr, .clock_group = 1},
  {.name = "emit",
   .doc  = "any beat; each one sends a copy of the buffered beat to "
           "`out` (nothing, if no beat has been buffered yet)",
   .type = nullptr, .clock_group = 2},
};
const PortSpec kOports[] = {
  {.name = "out",
   .doc  = "a copy of the buffered beat, one per emit trigger",
   .type = nullptr, .clock_group = 2},
};
const StageSpec kSpec = {
  .type_name     = "buffer",
  .doc           = "One-beat sample-and-hold register. `advance` reads "
                   "the next `in` beat into a 1-beat buffer (dropping "
                   "whatever it held); `emit` sends a copy of the "
                   "buffered beat to `out`. Decouples an upstream rate "
                   "from a downstream one: advancing faster than "
                   "emitting drops input, emitting faster than "
                   "advancing repeats the held beat.",
  .display_name  = "Buffer",
  .category      = StageCategory::Control,
  .iports        = kIports,
  .oports        = kOports,
  .attrs         = {},
};
}  // namespace

const StageSpec&
BufferStage::spec() const noexcept
{
  return kSpec;
}

Job
BufferStage::initialize(RuntimeContext& ctx)
{
  // All three inputs are required for the stage to do anything: an
  // unwired port reads as permanent EOS, which silently degrades the
  // stage (no advance => nothing to emit; no emit => no output at
  // all). Say so once at launch rather than looking hung.
  static const char* kNames[] = { "in", "advance", "emit" };
  for (unsigned p = 0; p < 3; ++p) {
    if (!ctx.iport_connected(p)) {
      session()->log_normal(
          fmt("buffer('{}'): iport{} ('{}') is not connected; the "
              "stage can never {}.",
              id(), p, kNames[p],
              p == 2 ? "emit" : "buffer a beat"));
    }
  }
  co_return;
}

Job
BufferStage::process(RuntimeContext& ctx)
{
  // Once the emit trigger is closed nothing can ever leave this stage
  // again, so the held beat and any further input are moot. This is
  // also the stage's only exit: `in` or `advance` finishing just
  // freezes the buffer at its last value, which the emit clock may
  // legitimately keep re-sending.
  if (ctx.eos(2)) {
    ctx.signal_done();
    co_return;
  }

  // Wake on whichever trigger arrives first. A port already at EOS is
  // left out of the wait set -- read_any reports a closed port as
  // perpetually readable, so keeping it would spin.
  vector<unsigned> wait_ports;
  if (!ctx.eos(1)) {
    wait_ports.push_back(1);
  }
  wait_ports.push_back(2);
  co_await ctx.read_any(std::move(wait_ports));

  // Advance before emit: when both triggers are already pending, the
  // emit must send what this advance just sampled, not the previous
  // value. Draining each port's whole backlog keeps a burst of
  // triggers from queueing up behind one process() call.
  while (ctx.backlog(1) > 0) {
    auto trigger = co_await ctx.read(1);
    if (!trigger) {
      break;                 // advance closed; handled on a later call
    }
    if (ctx.eos(0)) {
      continue;              // source finished: nothing left to sample
    }
    // The rendezvous. Suspends until the producer delivers, so an
    // advance always costs exactly one `in` beat.
    auto beat = co_await ctx.read(0);
    if (!beat) {
      continue;              // raced the source to EOS
    }
    _held = std::move(beat);  // drops the beat held before this one
  }

  while (ctx.backlog(2) > 0) {
    auto trigger = co_await ctx.read(2);
    if (!trigger) {
      break;                 // emit closed; the next call ends the stage
    }
    if (_held) {
      co_await ctx.write(0, _held->clone());
    }
  }
  co_return;
}

VPIPE_REGISTER_STAGE(BufferStage)
VPIPE_REGISTER_SPEC(BufferStage, kSpec)

}

#ifndef BUFFER_STAGE_H
#define BUFFER_STAGE_H

#include "common/beat-payload-intf.h"
#include "pipeline/typed-stage.h"

#include <memory>
#include <string>
#include <vector>

namespace vpipe {

// A one-beat sample-and-hold register, externally clocked on both
// sides. It decouples a producer's rate from a consumer's rate when
// neither should backpressure the other:
//
//   iport0 "in"       any beat -- the sampled value
//   iport1 "advance"  any beat -- sample one `in` beat into the buffer
//   iport2 "emit"     any beat -- send a copy of the buffer to `out`
//   oport0 "out"      a copy of the buffered beat, one per emit trigger
//
// The buffer holds exactly one beat. `advance` is a RENDEZVOUS: every
// trigger consumes exactly one `in` beat, suspending until the producer
// delivers one, and the newly read beat REPLACES the held one -- so
// advancing faster than emitting drops the beats that were never
// emitted. Emitting faster than advancing re-sends the held beat (each
// emit gets its own clone), and an emit before the first advance sends
// nothing.
//
// Reading `in` moves the payload out of the edge buffer whenever this
// stage is the slowest cursor, so a held beat does not pin an upstream
// slot (under fanout the read falls back to clone + release, same as
// any other consumer).
class BufferStage final : public TypedStage<BufferStage> {
public:
  static constexpr const char* kTypeName = "buffer";

  BufferStage(const SessionContextIntf* session,
              std::string               id,
              std::vector<InEdge>       iports,
              FlexData                  config =
                  FlexData::make_object());

  Job initialize(RuntimeContext& ctx) override;
  Job process   (RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

private:
  // The held beat, or null before the first advance trigger (and after
  // an advance that raced the source to EOS). Owned outright: the beat
  // is handed to this stage by the read, and every emit sends a clone.
  std::unique_ptr<BeatPayloadIntf> _held;
};

}

#endif

#ifndef VPIPE_STAGES_TRIGGER_BEAT_H
#define VPIPE_STAGES_TRIGGER_BEAT_H

#include "common/beat-payload-intf.h"

#include <memory>
#include <string>

namespace vpipe {

// Empty payload exchanged between trigger-source stages (e.g. chrono)
// and trigger-sink stages (e.g. shell). Carries no data; the *fact*
// of receipt is the signal.
struct TriggerBeat {};

// Pipeline-edge payload form.
class TriggerPayload : public TriggerBeat, public BeatPayloadIntf {
public:
  std::unique_ptr<BeatPayloadIntf>
  clone() const override
  {
    return std::make_unique<TriggerPayload>();
  }

  std::string
  describe() const override
  {
    return "Trigger";
  }
};

}

#endif

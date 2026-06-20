#ifndef BEAT_PAYLOAD_INTF_H
#define BEAT_PAYLOAD_INTF_H

#include <memory>
#include <string>
#include <type_traits>
#include <utility>

namespace vpipe {

// Polymorphic payload base for pipeline beats. One pipeline edge
// carries an `std::unique_ptr<BeatPayloadIntf>` per beat; the buffer
// is the sole owner of the payload until either the slowest cursor
// acquires it (move-out) or all cursors advance past the slot.
//
// Concrete payload types derive from this interface and implement
// `clone()` and `describe()`. Fanout consumers that cannot acquire
// a slot (because a slower cursor still needs it) must clone the
// borrowed payload to forward it -- so derived classes should make
// `clone()` cheap, typically by holding any large data behind an
// internal `std::shared_ptr<DataBlock>` so clone is just a
// refcount bump.
//
// Type identification: stages declare their expected payload type
// per port (see Stage::iport_payload_type / oport_payload_type) and
// the pipeline-runtime verifies once at build time. Hot-path access
// is `static_cast<const ConcretePayload*>(p)` -- no RTTI per read.
class BeatPayloadIntf {
public:
  virtual ~BeatPayloadIntf() = default;

  // Fanout clone. Should be O(1) where possible (refcount bump on
  // an internal shared resource). Only do a deep copy when no
  // sharing is feasible.
  virtual std::unique_ptr<BeatPayloadIntf>
  clone() const = 0;

  // Short, human-readable, ASCII description for logs/traces.
  // Typical format: "<TypeName> <shape-or-key-fields>".
  virtual std::string
  describe() const = 0;
};

// Helper: build a `unique_ptr<BeatPayloadIntf>` from a concrete
// payload type derived from BeatPayloadIntf. Replaces the old
// `Beat::make<T>(x)` factory.
template <class T, class... Args>
std::unique_ptr<BeatPayloadIntf>
make_payload(Args&&... args)
{
  static_assert(std::is_base_of<BeatPayloadIntf, T>::value,
                "make_payload<T>: T must derive from BeatPayloadIntf");
  return std::make_unique<T>(std::forward<Args>(args)...);
}

}

#endif

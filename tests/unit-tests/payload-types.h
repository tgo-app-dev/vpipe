#ifndef VPIPE_TESTS_INTERNAL_PAYLOAD_TYPES_H
#define VPIPE_TESTS_INTERNAL_PAYLOAD_TYPES_H

#include "common/beat-payload-intf.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vpipe::test {

// Generic BeatPayloadIntf wrapper around a value of type T for use
// in tests that don't need a real domain payload.
template <class T>
class ValuePayload : public BeatPayloadIntf {
public:
  T value{};

  ValuePayload() = default;
  explicit
  ValuePayload(T v)
    : value(std::move(v))
  {}

  std::unique_ptr<BeatPayloadIntf>
  clone() const override
  {
    return std::make_unique<ValuePayload<T>>(value);
  }

  std::string
  describe() const override
  {
    return "TestValuePayload";
  }
};

using IntPayload    = ValuePayload<int>;
using UintPayload   = ValuePayload<unsigned>;
using VecIntPayload = ValuePayload<std::vector<int>>;

}

#endif

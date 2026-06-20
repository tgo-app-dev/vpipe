// CoreMLState.hpp -- MLState wrapper (CoreML 7+, macOS 15+).
#pragma once

#include "CoreMLPrivate.hpp"

#include <Foundation/Foundation.hpp>

namespace CML {

class MultiArray;

class State : public NS::Referencing<State> {
public:
  // -getMultiArrayForStateNamed:handler: takes an Objective-C block.
  // For wrapper convenience we provide a callback-pointer overload.
  // The handler receives a non-owning pointer; do not retain past the
  // callback.
  using MultiArrayHandler = void (*)(MultiArray*, void* userData);

  void getMultiArrayForStateNamed(const NS::String* name,
                                  MultiArrayHandler handler,
                                  void*             userData) const;
};

}

// Implementation lives in coreml-private.cc (block bridging is the
// only thing that cannot be done via a pure inline objc_msgSend).

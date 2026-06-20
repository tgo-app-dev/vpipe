// CoreMLUpdateProgressHandlers.hpp -- MLUpdateProgressHandlers.
//
// Configures which update events fire callbacks (`progressHandler`)
// and what runs at the end of the task (`completionHandler`).
// Block bridging happens in coreml-private.cc.
#pragma once

#include "CoreMLPrivate.hpp"

#include <Foundation/Foundation.hpp>

namespace CML {

class UpdateContext;

// MLUpdateProgressEvent flags from CoreML.h.
enum UpdateProgressEvent : NS::UInteger {
  UpdateProgressEventTrainingBegin = 1 << 0,
  UpdateProgressEventEpochEnd      = 1 << 1,
  UpdateProgressEventMiniBatchEnd  = 1 << 2,
};

class UpdateProgressHandlers
  : public NS::Referencing<UpdateProgressHandlers>
{
public:
  using Handler = void (*)(UpdateContext*, void* userData);

  static UpdateProgressHandlers* alloc();

  // The bridging adapter wraps each function-pointer pair into an
  // Objective-C block; see coreml-private.cc for the implementation.
  UpdateProgressHandlers* initForEvents(
      UpdateProgressEvent events,
      Handler             progressHandler,
      void*               progressUserData,
      Handler             completionHandler,
      void*               completionUserData);
};

}

_CML_INLINE CML::UpdateProgressHandlers*
CML::UpdateProgressHandlers::alloc()
{
  return NS::Object::alloc<UpdateProgressHandlers>(
      _CML_PRIVATE_CLS(MLUpdateProgressHandlers));
}

// initForEvents implementation lives in coreml-private.cc -- the
// Objective-C block creation requires actual blocks runtime, not just
// objc_msgSend.

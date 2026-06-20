// CoreMLUpdateTask.hpp -- MLUpdateTask wrapper (training/update).
#pragma once

#include "CoreMLPrivate.hpp"

#include <Foundation/Foundation.hpp>

namespace CML {

class BatchProvider;
class ModelConfiguration;
class UpdateProgressHandlers;
class UpdateContext;

class UpdateTask : public NS::Referencing<UpdateTask> {
public:
  using CompletionHandler =
      void (*)(UpdateContext*, void* userData);

  // Three-arg form:
  static UpdateTask* updateTaskForModelAtURL(
      const NS::URL*        modelURL,
      const BatchProvider*  trainingData,
      CompletionHandler     completion,
      void*                 userData,
      NS::Error**           error);

  // With configuration:
  static UpdateTask* updateTaskForModelAtURL(
      const NS::URL*              modelURL,
      const BatchProvider*        trainingData,
      const ModelConfiguration*   configuration,
      CompletionHandler           completion,
      void*                       userData,
      NS::Error**                 error);

  // Full progress-handler form:
  static UpdateTask* updateTaskForModelAtURL(
      const NS::URL*                modelURL,
      const BatchProvider*          trainingData,
      const ModelConfiguration*     configuration,
      const UpdateProgressHandlers* progressHandlers,
      NS::Error**                   error);

  void resume();
  void resumeWithParameters(const NS::Dictionary* parameters);
  void cancel();
};

}

// Static factories: implemented in coreml-private.cc -- they bridge
// the function-pointer completion handler to an Obj-C block.

_CML_INLINE void CML::UpdateTask::resume()
{
  NS::Object::sendMessage<void>(this, _CML_PRIVATE_SEL(resume));
}

_CML_INLINE void
CML::UpdateTask::resumeWithParameters(const NS::Dictionary* parameters)
{
  NS::Object::sendMessage<void>(
      this, _CML_PRIVATE_SEL(resumeWithParameters_), parameters);
}

_CML_INLINE void CML::UpdateTask::cancel()
{
  NS::Object::sendMessage<void>(this, _CML_PRIVATE_SEL(cancel));
}

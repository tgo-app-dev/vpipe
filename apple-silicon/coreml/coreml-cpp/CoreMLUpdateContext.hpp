// CoreMLUpdateContext.hpp -- MLUpdateContext wrapper.
#pragma once

#include "CoreMLPrivate.hpp"

#include <Foundation/Foundation.hpp>

namespace CML {

class Model;
class UpdateTask;

class UpdateContext : public NS::Referencing<UpdateContext> {
public:
  UpdateTask*    task()    const;
  Model*         model()   const;
  // MLUpdateProgressEvent flag value.
  NS::UInteger   event()   const;
  NS::Dictionary* metrics()    const;
  NS::Dictionary* parameters() const;
};

}

_CML_INLINE CML::UpdateTask* CML::UpdateContext::task() const
{
  return NS::Object::sendMessage<UpdateTask*>(
      this, _CML_PRIVATE_SEL(task));
}

_CML_INLINE CML::Model* CML::UpdateContext::model() const
{
  return NS::Object::sendMessage<Model*>(
      this, _CML_PRIVATE_SEL(model));
}

_CML_INLINE NS::UInteger CML::UpdateContext::event() const
{
  return NS::Object::sendMessage<NS::UInteger>(
      this, _CML_PRIVATE_SEL(event));
}

_CML_INLINE NS::Dictionary* CML::UpdateContext::metrics() const
{
  return NS::Object::sendMessage<NS::Dictionary*>(
      this, _CML_PRIVATE_SEL(metrics));
}

_CML_INLINE NS::Dictionary* CML::UpdateContext::parameters() const
{
  return NS::Object::sendMessage<NS::Dictionary*>(
      this, _CML_PRIVATE_SEL(parameters_));
}

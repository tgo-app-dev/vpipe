// CoreMLBatchProvider.hpp -- MLBatchProvider protocol wrapper.
//
// Like FeatureProvider, MLBatchProvider is a protocol. Concrete
// instances reach C++ via the result of MLModel batch prediction or
// via ArrayBatchProvider.
#pragma once

#include "CoreMLPrivate.hpp"

#include <Foundation/Foundation.hpp>

namespace CML {

class FeatureProvider;

class BatchProvider : public NS::Referencing<BatchProvider> {
public:
  NS::Integer       count()                          const;
  FeatureProvider*  featuresAtIndex(NS::Integer idx) const;
};

}

_CML_INLINE NS::Integer CML::BatchProvider::count() const
{
  return NS::Object::sendMessage<NS::Integer>(
      this, _CML_PRIVATE_SEL(count));
}

_CML_INLINE CML::FeatureProvider*
CML::BatchProvider::featuresAtIndex(NS::Integer idx) const
{
  return NS::Object::sendMessage<FeatureProvider*>(
      this, _CML_PRIVATE_SEL(featuresAtIndex_), idx);
}

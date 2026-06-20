// CoreMLFeatureProvider.hpp -- MLFeatureProvider protocol wrapper.
//
// MLFeatureProvider is an Objective-C protocol, not a class. The
// wrapper exposes it as a base type with the protocol's method
// surface available via objc_msgSend. Concrete instances reach C++
// callers as the result of MLModel prediction (Apple's internal
// classes conform to MLFeatureProvider) or via
// DictionaryFeatureProvider (which we expose as a real class).
//
// To implement a *custom* FeatureProvider in C++, see
// coreml-private.cc -- it offers a runtime-bridged adapter.
#pragma once

#include "CoreMLPrivate.hpp"

#include <Foundation/Foundation.hpp>

namespace CML {

class FeatureValue;

class FeatureProvider : public NS::Referencing<FeatureProvider> {
public:
  NS::Set*      featureNames()                     const;
  FeatureValue* featureValueForName(
      const NS::String* name) const;
};

}

_CML_INLINE NS::Set* CML::FeatureProvider::featureNames() const
{
  return NS::Object::sendMessage<NS::Set*>(
      this, _CML_PRIVATE_SEL(featureNames));
}

_CML_INLINE CML::FeatureValue*
CML::FeatureProvider::featureValueForName(
    const NS::String* name) const
{
  return NS::Object::sendMessage<FeatureValue*>(
      this, _CML_PRIVATE_SEL(featureValueForName_), name);
}

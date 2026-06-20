// CoreMLDictionaryFeatureProvider.hpp -- MLDictionaryFeatureProvider.
//
// Concrete MLFeatureProvider backed by an NS::Dictionary. The most
// convenient way to package C++-side inputs for prediction.
#pragma once

#include "CoreMLPrivate.hpp"
#include "CoreMLFeatureProvider.hpp"

#include <Foundation/Foundation.hpp>

namespace CML {

class DictionaryFeatureProvider
  : public NS::Referencing<DictionaryFeatureProvider, FeatureProvider>
{
public:
  static DictionaryFeatureProvider* alloc();

  // Initializes from {NSString*: MLFeatureValue*} (or other supported
  // values). Errors out if dictionary contains unsupported types.
  DictionaryFeatureProvider* initWithDictionary(
      const NS::Dictionary* dict,
      NS::Error**           error);

  NS::Dictionary* dictionary() const;
};

}

_CML_INLINE CML::DictionaryFeatureProvider*
CML::DictionaryFeatureProvider::alloc()
{
  return NS::Object::alloc<DictionaryFeatureProvider>(
      _CML_PRIVATE_CLS(MLDictionaryFeatureProvider));
}

_CML_INLINE CML::DictionaryFeatureProvider*
CML::DictionaryFeatureProvider::initWithDictionary(
    const NS::Dictionary* dict,
    NS::Error**           error)
{
  return NS::Object::sendMessage<DictionaryFeatureProvider*>(
      this, _CML_PRIVATE_SEL(initWithDictionary_error_), dict, error);
}

_CML_INLINE NS::Dictionary*
CML::DictionaryFeatureProvider::dictionary() const
{
  return NS::Object::sendMessage<NS::Dictionary*>(
      this, _CML_PRIVATE_SEL(dictionary));
}

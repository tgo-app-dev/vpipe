// CoreMLArrayBatchProvider.hpp -- MLArrayBatchProvider wrapper.
#pragma once

#include "CoreMLPrivate.hpp"
#include "CoreMLBatchProvider.hpp"

#include <Foundation/Foundation.hpp>

namespace CML {

class FeatureProvider;

class ArrayBatchProvider
  : public NS::Referencing<ArrayBatchProvider, BatchProvider>
{
public:
  static ArrayBatchProvider* alloc();

  // Initialize from NSArray<id<MLFeatureProvider>>.
  ArrayBatchProvider* initWithFeatureProviderArray(
      const NS::Array* providers);

  // Initialize from {NSString*: NSArray*}; the provider expands the
  // dict into per-row feature providers.
  ArrayBatchProvider* initWithDictionary(
      const NS::Dictionary* dictionaryOfArrays,
      NS::Error**           error);

  NS::Array* array() const;
};

}

_CML_INLINE CML::ArrayBatchProvider*
CML::ArrayBatchProvider::alloc()
{
  return NS::Object::alloc<ArrayBatchProvider>(
      _CML_PRIVATE_CLS(MLArrayBatchProvider));
}

_CML_INLINE CML::ArrayBatchProvider*
CML::ArrayBatchProvider::initWithFeatureProviderArray(
    const NS::Array* providers)
{
  return NS::Object::sendMessage<ArrayBatchProvider*>(
      this, _CML_PRIVATE_SEL(initWithFeatureProviderArray_),
      providers);
}

_CML_INLINE CML::ArrayBatchProvider*
CML::ArrayBatchProvider::initWithDictionary(
    const NS::Dictionary* dictionaryOfArrays,
    NS::Error**           error)
{
  return NS::Object::sendMessage<ArrayBatchProvider*>(
      this, _CML_PRIVATE_SEL(initWithDictionary_error_),
      dictionaryOfArrays, error);
}

_CML_INLINE NS::Array* CML::ArrayBatchProvider::array() const
{
  return NS::Object::sendMessage<NS::Array*>(
      this, _CML_PRIVATE_SEL(array));
}

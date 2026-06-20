// CoreMLModelAsset.hpp -- MLModelAsset wrapper.
#pragma once

#include "CoreMLPrivate.hpp"

#include <Foundation/Foundation.hpp>

namespace CML {

class ModelAsset : public NS::Referencing<ModelAsset> {
public:
  static ModelAsset* modelAssetWithURL(const NS::URL* url,
                                       NS::Error**    error);
  static ModelAsset* modelAssetWithSpecificationData(
      const NS::Data* spec,
      NS::Error**     error);
};

}

_CML_INLINE CML::ModelAsset*
CML::ModelAsset::modelAssetWithURL(const NS::URL* url,
                                   NS::Error**    error)
{
  return NS::Object::sendMessage<ModelAsset*>(
      _CML_PRIVATE_CLS(MLModelAsset),
      _CML_PRIVATE_SEL(modelAssetWithURL_error_),
      url, error);
}

_CML_INLINE CML::ModelAsset*
CML::ModelAsset::modelAssetWithSpecificationData(const NS::Data* spec,
                                                 NS::Error**     error)
{
  return NS::Object::sendMessage<ModelAsset*>(
      _CML_PRIVATE_CLS(MLModelAsset),
      _CML_PRIVATE_SEL(modelAssetWithSpecificationData_error_),
      spec, error);
}

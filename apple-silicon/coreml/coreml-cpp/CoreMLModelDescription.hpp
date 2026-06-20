// CoreMLModelDescription.hpp -- MLModelDescription wrapper.
#pragma once

#include "CoreMLPrivate.hpp"

#include <Foundation/Foundation.hpp>

namespace CML {

class ModelDescription
  : public NS::Copying<ModelDescription>
{
public:
  // {NSString*: MLFeatureDescription*}
  NS::Dictionary* inputDescriptionsByName()  const;
  NS::Dictionary* outputDescriptionsByName() const;
  NS::Dictionary* stateDescriptionsByName()  const;
  NS::Dictionary* trainingInputDescriptionsByName() const;
  NS::Dictionary* parameterDescriptionsByKey()      const;

  NS::String* predictedFeatureName()        const;
  NS::String* predictedProbabilitiesName()  const;

  NS::Dictionary* metadata() const;
  bool            isUpdatable() const;
};

}

_CML_INLINE NS::Dictionary*
CML::ModelDescription::inputDescriptionsByName() const
{
  return NS::Object::sendMessage<NS::Dictionary*>(
      this, _CML_PRIVATE_SEL(inputDescriptionsByName));
}

_CML_INLINE NS::Dictionary*
CML::ModelDescription::outputDescriptionsByName() const
{
  return NS::Object::sendMessage<NS::Dictionary*>(
      this, _CML_PRIVATE_SEL(outputDescriptionsByName));
}

_CML_INLINE NS::Dictionary*
CML::ModelDescription::stateDescriptionsByName() const
{
  return NS::Object::sendMessage<NS::Dictionary*>(
      this, _CML_PRIVATE_SEL(stateDescriptionsByName));
}

_CML_INLINE NS::Dictionary*
CML::ModelDescription::trainingInputDescriptionsByName() const
{
  return NS::Object::sendMessage<NS::Dictionary*>(
      this, _CML_PRIVATE_SEL(trainingInputDescriptionsByName));
}

_CML_INLINE NS::Dictionary*
CML::ModelDescription::parameterDescriptionsByKey() const
{
  return NS::Object::sendMessage<NS::Dictionary*>(
      this, _CML_PRIVATE_SEL(parameterDescriptionsByKey));
}

_CML_INLINE NS::String*
CML::ModelDescription::predictedFeatureName() const
{
  return NS::Object::sendMessage<NS::String*>(
      this, _CML_PRIVATE_SEL(predictedFeatureName));
}

_CML_INLINE NS::String*
CML::ModelDescription::predictedProbabilitiesName() const
{
  return NS::Object::sendMessage<NS::String*>(
      this, _CML_PRIVATE_SEL(predictedProbabilitiesName));
}

_CML_INLINE NS::Dictionary*
CML::ModelDescription::metadata() const
{
  return NS::Object::sendMessage<NS::Dictionary*>(
      this, _CML_PRIVATE_SEL(metadata));
}

_CML_INLINE bool CML::ModelDescription::isUpdatable() const
{
  return NS::Object::sendMessage<bool>(
      this, _CML_PRIVATE_SEL(isUpdatable));
}

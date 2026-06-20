// CoreMLFeatureDescription.hpp -- MLFeatureDescription wrapper.
#pragma once

#include "CoreMLPrivate.hpp"
#include "CoreMLFeatureValue.hpp"

#include <Foundation/Foundation.hpp>

namespace CML {

class MultiArrayConstraint;
class ImageConstraint;
class DictionaryConstraint;
class SequenceConstraint;
class FeatureValue;

class FeatureDescription
  : public NS::Copying<FeatureDescription>
{
public:
  NS::String*  name()       const;
  FeatureType  type()       const;
  bool         isOptional() const;

  MultiArrayConstraint*  multiArrayConstraint() const;
  ImageConstraint*       imageConstraint()      const;
  DictionaryConstraint*  dictionaryConstraint() const;
  SequenceConstraint*    sequenceConstraint()   const;

  bool isAllowedValue(const FeatureValue* value) const;
};

}

_CML_INLINE NS::String* CML::FeatureDescription::name() const
{
  return NS::Object::sendMessage<NS::String*>(
      this, _CML_PRIVATE_SEL(name));
}

_CML_INLINE CML::FeatureType CML::FeatureDescription::type() const
{
  return NS::Object::sendMessage<FeatureType>(
      this, _CML_PRIVATE_SEL(type));
}

_CML_INLINE bool CML::FeatureDescription::isOptional() const
{
  return NS::Object::sendMessage<bool>(
      this, _CML_PRIVATE_SEL(isOptional));
}

_CML_INLINE CML::MultiArrayConstraint*
CML::FeatureDescription::multiArrayConstraint() const
{
  return NS::Object::sendMessage<MultiArrayConstraint*>(
      this, _CML_PRIVATE_SEL(multiArrayConstraint));
}

_CML_INLINE CML::ImageConstraint*
CML::FeatureDescription::imageConstraint() const
{
  return NS::Object::sendMessage<ImageConstraint*>(
      this, _CML_PRIVATE_SEL(imageConstraint));
}

_CML_INLINE CML::DictionaryConstraint*
CML::FeatureDescription::dictionaryConstraint() const
{
  return NS::Object::sendMessage<DictionaryConstraint*>(
      this, _CML_PRIVATE_SEL(dictionaryConstraint));
}

_CML_INLINE CML::SequenceConstraint*
CML::FeatureDescription::sequenceConstraint() const
{
  return NS::Object::sendMessage<SequenceConstraint*>(
      this, _CML_PRIVATE_SEL(sequenceConstraint));
}

_CML_INLINE bool CML::FeatureDescription::isAllowedValue(
    const FeatureValue* value) const
{
  return NS::Object::sendMessage<bool>(
      this, _CML_PRIVATE_SEL(isAllowedValue_), value);
}

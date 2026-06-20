// CoreMLParameterDescription.hpp -- MLParameterDescription wrapper.
#pragma once

#include "CoreMLPrivate.hpp"

#include <Foundation/Foundation.hpp>

namespace CML {

class ParameterKey;
class NumericConstraint;

class ParameterDescription
  : public NS::Copying<ParameterDescription>
{
public:
  ParameterKey*       key()               const;
  NumericConstraint*  numericConstraint() const;
  NS::Object*         defaultValue()      const;
};

}

_CML_INLINE CML::ParameterKey* CML::ParameterDescription::key() const
{
  return NS::Object::sendMessage<ParameterKey*>(
      this, _CML_PRIVATE_SEL(key));
}

_CML_INLINE CML::NumericConstraint*
CML::ParameterDescription::numericConstraint() const
{
  return NS::Object::sendMessage<NumericConstraint*>(
      this, _CML_PRIVATE_SEL(numericConstraint));
}

_CML_INLINE NS::Object*
CML::ParameterDescription::defaultValue() const
{
  return NS::Object::sendMessage<NS::Object*>(
      this, _CML_PRIVATE_SEL(defaultValue));
}

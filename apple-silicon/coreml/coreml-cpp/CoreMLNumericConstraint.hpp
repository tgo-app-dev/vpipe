// CoreMLNumericConstraint.hpp -- MLNumericConstraint wrapper.
#pragma once

#include "CoreMLPrivate.hpp"

#include <Foundation/Foundation.hpp>

namespace CML {

class NumericConstraint : public NS::Referencing<NumericConstraint> {
public:
  NS::Number* minValue() const;
  NS::Number* maxValue() const;
  NS::Set*    enumeratedNumbers() const;
};

}

_CML_INLINE NS::Number* CML::NumericConstraint::minValue() const
{
  return NS::Object::sendMessage<NS::Number*>(
      this, _CML_PRIVATE_SEL(minValue));
}

_CML_INLINE NS::Number* CML::NumericConstraint::maxValue() const
{
  return NS::Object::sendMessage<NS::Number*>(
      this, _CML_PRIVATE_SEL(maxValue));
}

_CML_INLINE NS::Set* CML::NumericConstraint::enumeratedNumbers() const
{
  return NS::Object::sendMessage<NS::Set*>(
      this, _CML_PRIVATE_SEL(enumeratedNumbers));
}

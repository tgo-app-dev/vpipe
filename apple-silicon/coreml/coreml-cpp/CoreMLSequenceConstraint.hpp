// CoreMLSequenceConstraint.hpp -- MLSequenceConstraint wrapper.
#pragma once

#include "CoreMLPrivate.hpp"

#include <Foundation/Foundation.hpp>

namespace CML {

class NumericConstraint;

class SequenceConstraint : public NS::Referencing<SequenceConstraint> {
public:
  NS::Integer        valueType()       const;
  NumericConstraint* countConstraint() const;
};

}

_CML_INLINE NS::Integer CML::SequenceConstraint::valueType() const
{
  return NS::Object::sendMessage<NS::Integer>(
      this, _CML_PRIVATE_SEL(valueType));
}

_CML_INLINE CML::NumericConstraint*
CML::SequenceConstraint::countConstraint() const
{
  return NS::Object::sendMessage<NumericConstraint*>(
      this, _CML_PRIVATE_SEL(countConstraint));
}

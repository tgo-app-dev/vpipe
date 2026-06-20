// CoreMLMultiArrayConstraint.hpp -- MLMultiArrayConstraint wrapper.
#pragma once

#include "CoreMLPrivate.hpp"

#include <Foundation/Foundation.hpp>

namespace CML {

class MultiArrayShapeConstraint;

// MLMultiArrayDataType -- raw NSInteger values from CoreML.h.
enum MultiArrayDataType : NS::Integer {
  MultiArrayDataTypeInvalid = 0,
  MultiArrayDataTypeFloat32 = 0x10000 | 32,
  MultiArrayDataTypeFloat   = MultiArrayDataTypeFloat32,
  MultiArrayDataTypeFloat16 = 0x10000 | 16,
  MultiArrayDataTypeDouble  = 0x10000 | 64,
  MultiArrayDataTypeInt32   = 0x20000 | 32,
};

class MultiArrayConstraint
  : public NS::Referencing<MultiArrayConstraint>
{
public:
  NS::Array*                shape()           const;
  MultiArrayDataType        dataType()        const;
  MultiArrayShapeConstraint* shapeConstraint() const;
};

}

_CML_INLINE NS::Array* CML::MultiArrayConstraint::shape() const
{
  return NS::Object::sendMessage<NS::Array*>(
      this, _CML_PRIVATE_SEL(shape));
}

_CML_INLINE CML::MultiArrayDataType
CML::MultiArrayConstraint::dataType() const
{
  return NS::Object::sendMessage<MultiArrayDataType>(
      this, _CML_PRIVATE_SEL(dataType));
}

_CML_INLINE CML::MultiArrayShapeConstraint*
CML::MultiArrayConstraint::shapeConstraint() const
{
  return NS::Object::sendMessage<MultiArrayShapeConstraint*>(
      this, _CML_PRIVATE_SEL(shapeConstraint));
}

// CoreMLDictionaryConstraint.hpp -- MLDictionaryConstraint wrapper.
#pragma once

#include "CoreMLPrivate.hpp"

#include <Foundation/Foundation.hpp>

namespace CML {

class DictionaryConstraint
  : public NS::Referencing<DictionaryConstraint>
{
public:
  // 0=int64, 1=string  -- raw MLFeatureType for dict key.
  NS::Integer allowedKeyType() const;
};

}

_CML_INLINE NS::Integer
CML::DictionaryConstraint::allowedKeyType() const
{
  return NS::Object::sendMessage<NS::Integer>(
      this, _CML_PRIVATE_SEL(allowedKeyType));
}

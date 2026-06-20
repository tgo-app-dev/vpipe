// CoreMLParameterKey.hpp -- MLParameterKey wrapper.
#pragma once

#include "CoreMLPrivate.hpp"

#include <Foundation/Foundation.hpp>

namespace CML {

class ParameterKey : public NS::Copying<ParameterKey> {
public:
  // Underlying scope: int64 enum value.
  NS::Integer scope() const;
  NS::String* key()   const;
};

}

_CML_INLINE NS::Integer CML::ParameterKey::scope() const
{
  return NS::Object::sendMessage<NS::Integer>(
      this, _CML_PRIVATE_SEL(scope));
}

_CML_INLINE NS::String* CML::ParameterKey::key() const
{
  return NS::Object::sendMessage<NS::String*>(
      this, _CML_PRIVATE_SEL(key));
}

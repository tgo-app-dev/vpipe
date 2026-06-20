// CoreMLSequence.hpp -- MLSequence wrapper.
#pragma once

#include "CoreMLPrivate.hpp"

#include <Foundation/Foundation.hpp>

namespace CML {

class Sequence : public NS::Referencing<Sequence> {
public:
  static Sequence* sequenceWithStringArray(const NS::Array* strings);
  static Sequence* sequenceWithInt64Array(const NS::Array* ints);
  static Sequence* emptySequenceWithType(NS::Integer type);

  NS::Array* stringValues() const;
  NS::Array* int64Values()  const;
  NS::Integer type()        const;
};

}

_CML_INLINE CML::Sequence*
CML::Sequence::sequenceWithStringArray(const NS::Array* strings)
{
  return NS::Object::sendMessage<Sequence*>(
      _CML_PRIVATE_CLS(MLSequence),
      _CML_PRIVATE_SEL(sequenceWithStringArray_),
      strings);
}

_CML_INLINE CML::Sequence*
CML::Sequence::sequenceWithInt64Array(const NS::Array* ints)
{
  return NS::Object::sendMessage<Sequence*>(
      _CML_PRIVATE_CLS(MLSequence),
      _CML_PRIVATE_SEL(sequenceWithInt64Array_),
      ints);
}

_CML_INLINE CML::Sequence*
CML::Sequence::emptySequenceWithType(NS::Integer type)
{
  return NS::Object::sendMessage<Sequence*>(
      _CML_PRIVATE_CLS(MLSequence),
      _CML_PRIVATE_SEL(emptySequenceWithType_),
      type);
}

_CML_INLINE NS::Array* CML::Sequence::stringValues() const
{
  return NS::Object::sendMessage<NS::Array*>(
      this, _CML_PRIVATE_SEL(stringValues));
}

_CML_INLINE NS::Array* CML::Sequence::int64Values() const
{
  return NS::Object::sendMessage<NS::Array*>(
      this, _CML_PRIVATE_SEL(int64Values));
}

_CML_INLINE NS::Integer CML::Sequence::type() const
{
  return NS::Object::sendMessage<NS::Integer>(
      this, _CML_PRIVATE_SEL(type));
}

// CoreMLFeatureValue.hpp -- MLFeatureValue wrapper.
#pragma once

#include "CoreMLPrivate.hpp"

#include <Foundation/Foundation.hpp>

namespace CML {

class MultiArray;
class Sequence;

// MLFeatureType -- raw NSInteger values from CoreML.h.
enum FeatureType : NS::Integer {
  FeatureTypeInvalid    = 0,
  FeatureTypeInt64      = 1,
  FeatureTypeDouble     = 2,
  FeatureTypeString     = 3,
  FeatureTypeImage      = 4,
  FeatureTypeMultiArray = 5,
  FeatureTypeDictionary = 6,
  FeatureTypeSequence   = 7,
  FeatureTypeState      = 8,
};

class FeatureValue : public NS::Copying<FeatureValue> {
public:
  static FeatureValue* featureValueWithInt64(int64_t v);
  static FeatureValue* featureValueWithDouble(double v);
  static FeatureValue* featureValueWithString(const NS::String* v);
  static FeatureValue* featureValueWithMultiArray(
      const MultiArray* v);
  // Construct an MLFeatureValue from a CVPixelBufferRef (opaque void*).
  // Caller retains ownership of the pixel buffer until the feature
  // value has been consumed; CoreML's autoreleased instance copies
  // the underlying CVPixelBuffer reference internally.
  static FeatureValue* featureValueWithPixelBuffer(void* cv_pixel_buffer);
  static FeatureValue* featureValueWithDictionary(
      const NS::Dictionary* v,
      NS::Error**           error);
  static FeatureValue* featureValueWithSequence(const Sequence* v);
  static FeatureValue* undefinedFeatureValueWithType(FeatureType t);

  FeatureType         type()             const;
  int64_t             int64Value()       const;
  double              doubleValue()      const;
  NS::String*         stringValue()      const;
  MultiArray*         multiArrayValue()  const;
  NS::Dictionary*     dictionaryValue()  const;
  Sequence*           sequenceValue()    const;
  // CVPixelBufferRef as opaque void*.
  void*               imageBufferValue() const;

  bool isUndefined() const;
  bool isEqualToFeatureValue(const FeatureValue* other) const;
};

}

_CML_INLINE CML::FeatureValue*
CML::FeatureValue::featureValueWithInt64(int64_t v)
{
  return NS::Object::sendMessage<FeatureValue*>(
      _CML_PRIVATE_CLS(MLFeatureValue),
      _CML_PRIVATE_SEL(featureValueWithInt64_), v);
}

_CML_INLINE CML::FeatureValue*
CML::FeatureValue::featureValueWithDouble(double v)
{
  return NS::Object::sendMessage<FeatureValue*>(
      _CML_PRIVATE_CLS(MLFeatureValue),
      _CML_PRIVATE_SEL(featureValueWithDouble_), v);
}

_CML_INLINE CML::FeatureValue*
CML::FeatureValue::featureValueWithString(const NS::String* v)
{
  return NS::Object::sendMessage<FeatureValue*>(
      _CML_PRIVATE_CLS(MLFeatureValue),
      _CML_PRIVATE_SEL(featureValueWithString_), v);
}

_CML_INLINE CML::FeatureValue*
CML::FeatureValue::featureValueWithMultiArray(const MultiArray* v)
{
  return NS::Object::sendMessage<FeatureValue*>(
      _CML_PRIVATE_CLS(MLFeatureValue),
      _CML_PRIVATE_SEL(featureValueWithMultiArray_), v);
}

_CML_INLINE CML::FeatureValue*
CML::FeatureValue::featureValueWithPixelBuffer(void* cv_pixel_buffer)
{
  return NS::Object::sendMessage<FeatureValue*>(
      _CML_PRIVATE_CLS(MLFeatureValue),
      _CML_PRIVATE_SEL(featureValueWithPixelBuffer_),
      cv_pixel_buffer);
}

_CML_INLINE CML::FeatureValue*
CML::FeatureValue::featureValueWithDictionary(const NS::Dictionary* v,
                                              NS::Error**           error)
{
  return NS::Object::sendMessage<FeatureValue*>(
      _CML_PRIVATE_CLS(MLFeatureValue),
      _CML_PRIVATE_SEL(featureValueWithDictionary_error_), v, error);
}

_CML_INLINE CML::FeatureValue*
CML::FeatureValue::featureValueWithSequence(const Sequence* v)
{
  return NS::Object::sendMessage<FeatureValue*>(
      _CML_PRIVATE_CLS(MLFeatureValue),
      _CML_PRIVATE_SEL(featureValueWithSequence_), v);
}

_CML_INLINE CML::FeatureValue*
CML::FeatureValue::undefinedFeatureValueWithType(FeatureType t)
{
  return NS::Object::sendMessage<FeatureValue*>(
      _CML_PRIVATE_CLS(MLFeatureValue),
      _CML_PRIVATE_SEL(undefinedFeatureValueWithType_), t);
}

_CML_INLINE CML::FeatureType CML::FeatureValue::type() const
{
  return NS::Object::sendMessage<FeatureType>(
      this, _CML_PRIVATE_SEL(type));
}

_CML_INLINE int64_t CML::FeatureValue::int64Value() const
{
  return NS::Object::sendMessage<int64_t>(
      this, _CML_PRIVATE_SEL(int64Value));
}

_CML_INLINE double CML::FeatureValue::doubleValue() const
{
  return NS::Object::sendMessage<double>(
      this, _CML_PRIVATE_SEL(doubleValue));
}

_CML_INLINE NS::String* CML::FeatureValue::stringValue() const
{
  return NS::Object::sendMessage<NS::String*>(
      this, _CML_PRIVATE_SEL(stringValue));
}

_CML_INLINE CML::MultiArray*
CML::FeatureValue::multiArrayValue() const
{
  return NS::Object::sendMessage<MultiArray*>(
      this, _CML_PRIVATE_SEL(multiArrayValue));
}

_CML_INLINE NS::Dictionary*
CML::FeatureValue::dictionaryValue() const
{
  return NS::Object::sendMessage<NS::Dictionary*>(
      this, _CML_PRIVATE_SEL(dictionaryValue));
}

_CML_INLINE CML::Sequence* CML::FeatureValue::sequenceValue() const
{
  return NS::Object::sendMessage<Sequence*>(
      this, _CML_PRIVATE_SEL(sequenceValue));
}

_CML_INLINE void* CML::FeatureValue::imageBufferValue() const
{
  return NS::Object::sendMessage<void*>(
      this, _CML_PRIVATE_SEL(imageBufferValue));
}

_CML_INLINE bool CML::FeatureValue::isUndefined() const
{
  return NS::Object::sendMessage<bool>(
      this, _CML_PRIVATE_SEL(isUndefined));
}

_CML_INLINE bool CML::FeatureValue::isEqualToFeatureValue(
    const FeatureValue* other) const
{
  return NS::Object::sendMessage<bool>(
      this, _CML_PRIVATE_SEL(isEqualToFeatureValue_), other);
}

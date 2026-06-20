// CoreMLMultiArray.hpp -- MLMultiArray wrapper.
//
// Tensor container used as the canonical input/output for MLModel
// inference. Backs onto a CPU malloc by default, or wraps a
// caller-owned data pointer.
#pragma once

#include "CoreMLPrivate.hpp"
#include "CoreMLMultiArrayConstraint.hpp"

#include <Foundation/Foundation.hpp>

namespace CML {

class MultiArray : public NS::Referencing<MultiArray> {
public:
  static MultiArray* alloc();

  // Allocates a fresh CPU-backed buffer of `shape` ints, dataType
  // matching `dataType`. shape elements are NS::Number*.
  MultiArray* initWithShape(const NS::Array*    shape,
                            MultiArrayDataType  dataType,
                            NS::Error**         error);

  // Wraps a caller-owned pointer. `strides` is in elements (not
  // bytes). `deallocator` may be nullptr to disable cleanup.
  MultiArray* initWithDataPointer(
      void*                 dataPointer,
      const NS::Array*      shape,
      MultiArrayDataType    dataType,
      const NS::Array*      strides,
      const NS::Object*     deallocatorBlock,
      NS::Error**           error);

  MultiArrayDataType dataType()    const;
  NS::Array*         shape()       const;
  NS::Array*         strides()     const;

  // Direct buffer access. Use only on CPU-backed arrays where the
  // backing memory is contiguous; for non-contiguous storage use the
  // get*BytesWithHandler variants instead.
  void*              dataPointer() const;

  // Subscript helpers: shape-flat index space.
  NS::Number* objectAtIndexedSubscript(NS::Integer idx) const;
  void        setObjectAtIndexedSubscript(const NS::Number* value,
                                          NS::Integer       idx);

  // Multi-axis index: arr[ [i0, i1, i2] ].
  NS::Number* objectForKeyedSubscript(const NS::Array* idx) const;
  void        setObjectForKeyedSubscript(const NS::Number* value,
                                         const NS::Array*  idx);

  // Returns the underlying CVPixelBufferRef (id-typed) or nullptr.
  void* pixelBuffer() const;
};

}

_CML_INLINE CML::MultiArray* CML::MultiArray::alloc()
{
  return NS::Object::alloc<MultiArray>(
      _CML_PRIVATE_CLS(MLMultiArray));
}

_CML_INLINE CML::MultiArray*
CML::MultiArray::initWithShape(const NS::Array*   shape,
                               MultiArrayDataType dataType,
                               NS::Error**        error)
{
  return NS::Object::sendMessage<MultiArray*>(
      this, _CML_PRIVATE_SEL(initWithShape_dataType_error_),
      shape, dataType, error);
}

_CML_INLINE CML::MultiArray*
CML::MultiArray::initWithDataPointer(void*               dataPointer,
                                     const NS::Array*    shape,
                                     MultiArrayDataType  dataType,
                                     const NS::Array*    strides,
                                     const NS::Object*   deall,
                                     NS::Error**         error)
{
  return NS::Object::sendMessage<MultiArray*>(
      this,
      _CML_PRIVATE_SEL(
          initWithDataPointer_shape_dataType_strides_deallocator_error_),
      dataPointer, shape, dataType, strides, deall, error);
}

_CML_INLINE CML::MultiArrayDataType CML::MultiArray::dataType() const
{
  return NS::Object::sendMessage<MultiArrayDataType>(
      this, _CML_PRIVATE_SEL(dataType));
}

_CML_INLINE NS::Array* CML::MultiArray::shape() const
{
  return NS::Object::sendMessage<NS::Array*>(
      this, _CML_PRIVATE_SEL(shape));
}

_CML_INLINE NS::Array* CML::MultiArray::strides() const
{
  return NS::Object::sendMessage<NS::Array*>(
      this, _CML_PRIVATE_SEL(strides));
}

_CML_INLINE void* CML::MultiArray::dataPointer() const
{
  return NS::Object::sendMessage<void*>(
      this, _CML_PRIVATE_SEL(dataPointer));
}

_CML_INLINE NS::Number*
CML::MultiArray::objectAtIndexedSubscript(NS::Integer idx) const
{
  return NS::Object::sendMessage<NS::Number*>(
      this, _CML_PRIVATE_SEL(objectAtIndexedSubscript_), idx);
}

_CML_INLINE void
CML::MultiArray::setObjectAtIndexedSubscript(const NS::Number* value,
                                             NS::Integer       idx)
{
  NS::Object::sendMessage<void>(
      this, _CML_PRIVATE_SEL(setObject_atIndexedSubscript_),
      value, idx);
}

_CML_INLINE NS::Number*
CML::MultiArray::objectForKeyedSubscript(const NS::Array* idx) const
{
  return NS::Object::sendMessage<NS::Number*>(
      this, _CML_PRIVATE_SEL(objectForKeyedSubscript_), idx);
}

_CML_INLINE void
CML::MultiArray::setObjectForKeyedSubscript(const NS::Number* value,
                                            const NS::Array*  idx)
{
  NS::Object::sendMessage<void>(
      this, _CML_PRIVATE_SEL(setObject_forKeyedSubscript_),
      value, idx);
}

_CML_INLINE void* CML::MultiArray::pixelBuffer() const
{
  return NS::Object::sendMessage<void*>(
      this, _CML_PRIVATE_SEL(pixelBuffer));
}

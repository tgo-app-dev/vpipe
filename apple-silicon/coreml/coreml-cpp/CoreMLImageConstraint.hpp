// CoreMLImageConstraint.hpp -- MLImageConstraint wrapper.
#pragma once

#include "CoreMLPrivate.hpp"

#include <Foundation/Foundation.hpp>

namespace CML {

class ImageSizeConstraint;

class ImageConstraint : public NS::Referencing<ImageConstraint> {
public:
  NS::UInteger          pixelsWide()      const;
  NS::UInteger          pixelsHigh()      const;
  uint32_t              pixelFormatType() const;
  ImageSizeConstraint*  sizeConstraint()  const;
};

}

_CML_INLINE NS::UInteger CML::ImageConstraint::pixelsWide() const
{
  return NS::Object::sendMessage<NS::UInteger>(
      this, _CML_PRIVATE_SEL(pixelsWide));
}

_CML_INLINE NS::UInteger CML::ImageConstraint::pixelsHigh() const
{
  return NS::Object::sendMessage<NS::UInteger>(
      this, _CML_PRIVATE_SEL(pixelsHigh));
}

_CML_INLINE uint32_t CML::ImageConstraint::pixelFormatType() const
{
  return NS::Object::sendMessage<uint32_t>(
      this, _CML_PRIVATE_SEL(pixelFormatType));
}

_CML_INLINE CML::ImageSizeConstraint*
CML::ImageConstraint::sizeConstraint() const
{
  return NS::Object::sendMessage<ImageSizeConstraint*>(
      this, _CML_PRIVATE_SEL(sizeConstraint));
}

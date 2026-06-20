// CoreMLModelConfiguration.hpp -- MLModelConfiguration wrapper.
#pragma once

#include "CoreMLPrivate.hpp"

#include <Foundation/Foundation.hpp>

namespace CML {

// MLComputeUnits -- raw NSInteger values from CoreML.h.
enum ComputeUnits : NS::Integer {
  ComputeUnitsCPUOnly             = 0,
  ComputeUnitsCPUAndGPU           = 1,
  ComputeUnitsAll                 = 2,
  ComputeUnitsCPUAndNeuralEngine  = 3,
};

class ModelConfiguration
  : public NS::Copying<ModelConfiguration>
{
public:
  static ModelConfiguration* alloc();
  ModelConfiguration*        init();

  ComputeUnits     computeUnits() const;
  void             setComputeUnits(ComputeUnits units);

  NS::String*      modelDisplayName() const;
  void             setModelDisplayName(const NS::String* name);

  bool             allowLowPrecisionAccumulationOnGPU() const;
  void             setAllowLowPrecisionAccumulationOnGPU(bool flag);

  NS::Dictionary*  parameters() const;
  void             setParameters(const NS::Dictionary* params);

  // -preferredMetalDevice returns id<MTLDevice>; we expose as opaque
  // void* so callers don't have to pull in <Metal/Metal.hpp> through
  // CoreML. Cast to MTL::Device* if needed.
  void*            preferredMetalDevice() const;
  void             setPreferredMetalDevice(const void* device);
};

}

_CML_INLINE CML::ModelConfiguration* CML::ModelConfiguration::alloc()
{
  return NS::Object::alloc<ModelConfiguration>(
      _CML_PRIVATE_CLS(MLModelConfiguration));
}

_CML_INLINE CML::ModelConfiguration* CML::ModelConfiguration::init()
{
  return NS::Object::init<ModelConfiguration>();
}

_CML_INLINE CML::ComputeUnits
CML::ModelConfiguration::computeUnits() const
{
  return NS::Object::sendMessage<ComputeUnits>(
      this, _CML_PRIVATE_SEL(computeUnits));
}

_CML_INLINE void
CML::ModelConfiguration::setComputeUnits(ComputeUnits units)
{
  NS::Object::sendMessage<void>(
      this, _CML_PRIVATE_SEL(setComputeUnits_), units);
}

_CML_INLINE NS::String*
CML::ModelConfiguration::modelDisplayName() const
{
  return NS::Object::sendMessage<NS::String*>(
      this, _CML_PRIVATE_SEL(modelDisplayName));
}

_CML_INLINE void CML::ModelConfiguration::setModelDisplayName(
    const NS::String* name)
{
  NS::Object::sendMessage<void>(
      this, _CML_PRIVATE_SEL(setModelDisplayName_), name);
}

_CML_INLINE bool
CML::ModelConfiguration::allowLowPrecisionAccumulationOnGPU() const
{
  return NS::Object::sendMessage<bool>(
      this, _CML_PRIVATE_SEL(allowLowPrecisionAccumulationOnGPU));
}

_CML_INLINE void
CML::ModelConfiguration::setAllowLowPrecisionAccumulationOnGPU(
    bool flag)
{
  NS::Object::sendMessage<void>(
      this,
      _CML_PRIVATE_SEL(setAllowLowPrecisionAccumulationOnGPU_),
      flag);
}

_CML_INLINE NS::Dictionary*
CML::ModelConfiguration::parameters() const
{
  return NS::Object::sendMessage<NS::Dictionary*>(
      this, _CML_PRIVATE_SEL(parameters));
}

_CML_INLINE void CML::ModelConfiguration::setParameters(
    const NS::Dictionary* params)
{
  NS::Object::sendMessage<void>(
      this, _CML_PRIVATE_SEL(setParameters_), params);
}

_CML_INLINE void*
CML::ModelConfiguration::preferredMetalDevice() const
{
  return NS::Object::sendMessage<void*>(
      this, _CML_PRIVATE_SEL(preferredMetalDevice));
}

_CML_INLINE void CML::ModelConfiguration::setPreferredMetalDevice(
    const void* device)
{
  NS::Object::sendMessage<void>(
      this, _CML_PRIVATE_SEL(setPreferredMetalDevice_), device);
}

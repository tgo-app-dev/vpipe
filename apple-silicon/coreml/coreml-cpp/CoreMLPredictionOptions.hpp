// CoreMLPredictionOptions.hpp -- MLPredictionOptions wrapper.
#pragma once

#include "CoreMLPrivate.hpp"

#include <Foundation/Foundation.hpp>

namespace CML {

class PredictionOptions : public NS::Referencing<PredictionOptions> {
public:
  static PredictionOptions* alloc();
  PredictionOptions*        init();

  bool           usesCPUOnly() const;
  void           setUsesCPUOnly(bool flag);

  NS::Dictionary* outputBackings() const;
  void            setOutputBackings(const NS::Dictionary* backings);
};

}

_CML_INLINE CML::PredictionOptions* CML::PredictionOptions::alloc()
{
  return NS::Object::alloc<PredictionOptions>(
      _CML_PRIVATE_CLS(MLPredictionOptions));
}

_CML_INLINE CML::PredictionOptions* CML::PredictionOptions::init()
{
  return NS::Object::init<PredictionOptions>();
}

_CML_INLINE bool CML::PredictionOptions::usesCPUOnly() const
{
  return NS::Object::sendMessage<bool>(
      this, _CML_PRIVATE_SEL(usesCPUOnly));
}

_CML_INLINE void
CML::PredictionOptions::setUsesCPUOnly(bool flag)
{
  NS::Object::sendMessage<void>(
      this, _CML_PRIVATE_SEL(setUsesCPUOnly_), flag);
}

_CML_INLINE NS::Dictionary*
CML::PredictionOptions::outputBackings() const
{
  return NS::Object::sendMessage<NS::Dictionary*>(
      this, _CML_PRIVATE_SEL(outputBackings));
}

_CML_INLINE void CML::PredictionOptions::setOutputBackings(
    const NS::Dictionary* backings)
{
  NS::Object::sendMessage<void>(
      this, _CML_PRIVATE_SEL(setOutputBackings_), backings);
}

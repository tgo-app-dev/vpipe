// CoreML.hpp
//
// Umbrella header for the CoreML C++ wrapper. Includes Foundation
// (from extern/metal-cpp) plus all per-class headers in this
// directory. Public consumers should include this file rather than
// the individual class headers.
//
// Memory management uses the Foundation NS::Referencing convention:
//   * factory methods that return new objects: caller owns +1 retain
//     and must release / autorelease.
//   * +alloc() / -init() pattern follows Objective-C; prefer
//     NS::TransferPtr / NS::SharedPtr for RAII.

#pragma once

#include "Foundation/Foundation.hpp"

#include "CoreMLDefines.hpp"
#include "CoreMLPrivate.hpp"

#include "CoreMLArrayBatchProvider.hpp"
#include "CoreMLBatchProvider.hpp"
#include "CoreMLDictionaryConstraint.hpp"
#include "CoreMLDictionaryFeatureProvider.hpp"
#include "CoreMLFeatureDescription.hpp"
#include "CoreMLFeatureProvider.hpp"
#include "CoreMLFeatureValue.hpp"
#include "CoreMLImageConstraint.hpp"
#include "CoreMLModel.hpp"
#include "CoreMLModelAsset.hpp"
#include "CoreMLModelConfiguration.hpp"
#include "CoreMLModelDescription.hpp"
#include "CoreMLMultiArray.hpp"
#include "CoreMLMultiArrayConstraint.hpp"
#include "CoreMLNumericConstraint.hpp"
#include "CoreMLParameterDescription.hpp"
#include "CoreMLParameterKey.hpp"
#include "CoreMLPredictionOptions.hpp"
#include "CoreMLSequence.hpp"
#include "CoreMLSequenceConstraint.hpp"
#include "CoreMLState.hpp"
#include "CoreMLUpdateContext.hpp"
#include "CoreMLUpdateProgressHandlers.hpp"
#include "CoreMLUpdateTask.hpp"

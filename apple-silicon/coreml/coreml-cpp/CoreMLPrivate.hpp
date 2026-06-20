// CoreMLPrivate.hpp
//
// The selector / class registration table for the CoreML wrapper.
// Mirrors metal-cpp's MTLPrivate.hpp + MTLHeaderBridge.hpp split, but
// merged into one file because CoreML's surface is small enough not
// to need separate accumulation.
//
// Two compilation modes, controlled by CML_PRIVATE_IMPLEMENTATION:
//
//   * Default (CML_PRIVATE_IMPLEMENTATION not defined) -- emits
//     `extern void* s_kFoo;` / `extern SEL s_kfoo_;` declarations.
//     Every CoreML*.hpp header includes this in this mode and uses
//     _CML_PRIVATE_CLS / _CML_PRIVATE_SEL to reference the externs.
//
//   * Implementation (CML_PRIVATE_IMPLEMENTATION defined) -- emits
//     the corresponding definitions, calling objc_lookUpClass() and
//     sel_registerName() at startup. Defined in exactly one TU:
//     apple-silicon/coreml/coreml-cpp/coreml-private.cc.

#pragma once

#include "CoreMLDefines.hpp"

#include <objc/runtime.h>

#define _CML_PRIVATE_CLS(symbol)   (Private::Class::s_k##symbol)
#define _CML_PRIVATE_SEL(accessor) (Private::Selector::s_k##accessor)
#define _CML_PRIVATE_PRO(symbol)   (Private::Protocol::s_k##symbol)

#if defined(CML_PRIVATE_IMPLEMENTATION)

#ifdef METALCPP_SYMBOL_VISIBILITY_HIDDEN
#define _CML_PRIVATE_VISIBILITY __attribute__((visibility("hidden")))
#else
#define _CML_PRIVATE_VISIBILITY __attribute__((visibility("default")))
#endif

#define _CML_PRIVATE_DEF_CLS(symbol) \
    void* s_k##symbol _CML_PRIVATE_VISIBILITY = \
        objc_lookUpClass(#symbol)

#define _CML_PRIVATE_DEF_PRO(symbol) \
    void* s_k##symbol _CML_PRIVATE_VISIBILITY = \
        objc_getProtocol(#symbol)

#define _CML_PRIVATE_DEF_SEL(accessor, symbol) \
    SEL s_k##accessor _CML_PRIVATE_VISIBILITY = \
        sel_registerName(symbol)

#else

#define _CML_PRIVATE_DEF_CLS(symbol)           extern void* s_k##symbol
#define _CML_PRIVATE_DEF_PRO(symbol)           extern void* s_k##symbol
#define _CML_PRIVATE_DEF_SEL(accessor, symbol) extern SEL    s_k##accessor

#endif

namespace CML::Private::Class {
  _CML_PRIVATE_DEF_CLS(MLArrayBatchProvider);
  _CML_PRIVATE_DEF_CLS(MLDictionaryConstraint);
  _CML_PRIVATE_DEF_CLS(MLDictionaryFeatureProvider);
  _CML_PRIVATE_DEF_CLS(MLFeatureDescription);
  _CML_PRIVATE_DEF_CLS(MLFeatureValue);
  _CML_PRIVATE_DEF_CLS(MLImageConstraint);
  _CML_PRIVATE_DEF_CLS(MLModel);
  _CML_PRIVATE_DEF_CLS(MLModelAsset);
  _CML_PRIVATE_DEF_CLS(MLModelConfiguration);
  _CML_PRIVATE_DEF_CLS(MLModelDescription);
  _CML_PRIVATE_DEF_CLS(MLMultiArray);
  _CML_PRIVATE_DEF_CLS(MLMultiArrayConstraint);
  _CML_PRIVATE_DEF_CLS(MLNumericConstraint);
  _CML_PRIVATE_DEF_CLS(MLParameterDescription);
  _CML_PRIVATE_DEF_CLS(MLParameterKey);
  _CML_PRIVATE_DEF_CLS(MLPredictionOptions);
  _CML_PRIVATE_DEF_CLS(MLSequence);
  _CML_PRIVATE_DEF_CLS(MLSequenceConstraint);
  _CML_PRIVATE_DEF_CLS(MLState);
  _CML_PRIVATE_DEF_CLS(MLUpdateContext);
  _CML_PRIVATE_DEF_CLS(MLUpdateProgressHandlers);
  _CML_PRIVATE_DEF_CLS(MLUpdateTask);
}

namespace CML::Private::Protocol {
  _CML_PRIVATE_DEF_PRO(MLBatchProvider);
  _CML_PRIVATE_DEF_PRO(MLFeatureProvider);
}

namespace CML::Private::Selector {

  // ---- MLModel ----------------------------------------------------
  _CML_PRIVATE_DEF_SEL(modelWithContentsOfURL_error_,
                       "modelWithContentsOfURL:error:");
  _CML_PRIVATE_DEF_SEL(
      modelWithContentsOfURL_configuration_error_,
      "modelWithContentsOfURL:configuration:error:");
  _CML_PRIVATE_DEF_SEL(compileModelAtURL_error_,
                       "compileModelAtURL:error:");
  _CML_PRIVATE_DEF_SEL(modelDescription, "modelDescription");
  _CML_PRIVATE_DEF_SEL(configuration, "configuration");
  _CML_PRIVATE_DEF_SEL(predictionFromFeatures_error_,
                       "predictionFromFeatures:error:");
  _CML_PRIVATE_DEF_SEL(predictionFromFeatures_options_error_,
                       "predictionFromFeatures:options:error:");
  _CML_PRIVATE_DEF_SEL(
      predictionsFromBatch_options_error_,
      "predictionsFromBatch:options:error:");
  _CML_PRIVATE_DEF_SEL(predictionsFromBatch_error_,
                       "predictionsFromBatch:error:");
  _CML_PRIVATE_DEF_SEL(newState, "newState");
  _CML_PRIVATE_DEF_SEL(
      predictionFromFeatures_usingState_options_error_,
      "predictionFromFeatures:usingState:options:error:");
  _CML_PRIVATE_DEF_SEL(parameterValueForKey_error_,
                       "parameterValueForKey:error:");
  _CML_PRIVATE_DEF_SEL(
      loadContentsOfURL_configuration_completionHandler_,
      "loadContentsOfURL:configuration:completionHandler:");
  _CML_PRIVATE_DEF_SEL(
      loadModelAsset_configuration_completionHandler_,
      "loadModelAsset:configuration:completionHandler:");
  _CML_PRIVATE_DEF_SEL(
      predictionFromFeatures_completionHandler_,
      "predictionFromFeatures:completionHandler:");
  _CML_PRIVATE_DEF_SEL(
      predictionFromFeatures_options_completionHandler_,
      "predictionFromFeatures:options:completionHandler:");

  // ---- MLModelAsset -----------------------------------------------
  _CML_PRIVATE_DEF_SEL(modelAssetWithURL_error_,
                       "modelAssetWithURL:error:");
  _CML_PRIVATE_DEF_SEL(modelAssetWithSpecificationData_error_,
                       "modelAssetWithSpecificationData:error:");

  // ---- MLModelConfiguration ---------------------------------------
  _CML_PRIVATE_DEF_SEL(computeUnits, "computeUnits");
  _CML_PRIVATE_DEF_SEL(setComputeUnits_, "setComputeUnits:");
  _CML_PRIVATE_DEF_SEL(modelDisplayName, "modelDisplayName");
  _CML_PRIVATE_DEF_SEL(setModelDisplayName_, "setModelDisplayName:");
  _CML_PRIVATE_DEF_SEL(allowLowPrecisionAccumulationOnGPU,
                       "allowLowPrecisionAccumulationOnGPU");
  _CML_PRIVATE_DEF_SEL(setAllowLowPrecisionAccumulationOnGPU_,
                       "setAllowLowPrecisionAccumulationOnGPU:");
  _CML_PRIVATE_DEF_SEL(parameters, "parameters");
  _CML_PRIVATE_DEF_SEL(setParameters_, "setParameters:");
  _CML_PRIVATE_DEF_SEL(preferredMetalDevice, "preferredMetalDevice");
  _CML_PRIVATE_DEF_SEL(setPreferredMetalDevice_,
                       "setPreferredMetalDevice:");

  // ---- MLModelDescription -----------------------------------------
  _CML_PRIVATE_DEF_SEL(inputDescriptionsByName,
                       "inputDescriptionsByName");
  _CML_PRIVATE_DEF_SEL(outputDescriptionsByName,
                       "outputDescriptionsByName");
  _CML_PRIVATE_DEF_SEL(predictedFeatureName, "predictedFeatureName");
  _CML_PRIVATE_DEF_SEL(predictedProbabilitiesName,
                       "predictedProbabilitiesName");
  _CML_PRIVATE_DEF_SEL(metadata, "metadata");
  _CML_PRIVATE_DEF_SEL(stateDescriptionsByName,
                       "stateDescriptionsByName");
  _CML_PRIVATE_DEF_SEL(trainingInputDescriptionsByName,
                       "trainingInputDescriptionsByName");
  _CML_PRIVATE_DEF_SEL(parameterDescriptionsByKey,
                       "parameterDescriptionsByKey");
  _CML_PRIVATE_DEF_SEL(isUpdatable, "isUpdatable");

  // ---- MLFeatureDescription ---------------------------------------
  _CML_PRIVATE_DEF_SEL(name, "name");
  _CML_PRIVATE_DEF_SEL(type, "type");
  _CML_PRIVATE_DEF_SEL(isOptional, "isOptional");
  _CML_PRIVATE_DEF_SEL(multiArrayConstraint, "multiArrayConstraint");
  _CML_PRIVATE_DEF_SEL(imageConstraint, "imageConstraint");
  _CML_PRIVATE_DEF_SEL(dictionaryConstraint, "dictionaryConstraint");
  _CML_PRIVATE_DEF_SEL(sequenceConstraint, "sequenceConstraint");
  _CML_PRIVATE_DEF_SEL(isAllowedValue_, "isAllowedValue:");

  // ---- MLFeatureValue ---------------------------------------------
  _CML_PRIVATE_DEF_SEL(featureValueWithInt64_,
                       "featureValueWithInt64:");
  _CML_PRIVATE_DEF_SEL(featureValueWithDouble_,
                       "featureValueWithDouble:");
  _CML_PRIVATE_DEF_SEL(featureValueWithString_,
                       "featureValueWithString:");
  _CML_PRIVATE_DEF_SEL(featureValueWithMultiArray_,
                       "featureValueWithMultiArray:");
  _CML_PRIVATE_DEF_SEL(featureValueWithPixelBuffer_,
                       "featureValueWithPixelBuffer:");
  _CML_PRIVATE_DEF_SEL(featureValueWithDictionary_error_,
                       "featureValueWithDictionary:error:");
  _CML_PRIVATE_DEF_SEL(featureValueWithSequence_,
                       "featureValueWithSequence:");
  _CML_PRIVATE_DEF_SEL(undefinedFeatureValueWithType_,
                       "undefinedFeatureValueWithType:");
  _CML_PRIVATE_DEF_SEL(int64Value, "int64Value");
  _CML_PRIVATE_DEF_SEL(doubleValue, "doubleValue");
  _CML_PRIVATE_DEF_SEL(stringValue, "stringValue");
  _CML_PRIVATE_DEF_SEL(multiArrayValue, "multiArrayValue");
  _CML_PRIVATE_DEF_SEL(dictionaryValue, "dictionaryValue");
  _CML_PRIVATE_DEF_SEL(sequenceValue, "sequenceValue");
  _CML_PRIVATE_DEF_SEL(imageBufferValue, "imageBufferValue");
  _CML_PRIVATE_DEF_SEL(isUndefined, "isUndefined");
  _CML_PRIVATE_DEF_SEL(isEqualToFeatureValue_,
                       "isEqualToFeatureValue:");

  // ---- MLDictionaryFeatureProvider --------------------------------
  _CML_PRIVATE_DEF_SEL(initWithDictionary_error_,
                       "initWithDictionary:error:");
  _CML_PRIVATE_DEF_SEL(dictionary, "dictionary");
  _CML_PRIVATE_DEF_SEL(featureNames, "featureNames");
  _CML_PRIVATE_DEF_SEL(featureValueForName_, "featureValueForName:");

  // ---- MLArrayBatchProvider ---------------------------------------
  _CML_PRIVATE_DEF_SEL(
      initWithFeatureProviderArray_,
      "initWithFeatureProviderArray:");
  _CML_PRIVATE_DEF_SEL(initWithDictionary_, "initWithDictionary:");
  _CML_PRIVATE_DEF_SEL(array, "array");
  _CML_PRIVATE_DEF_SEL(count, "count");
  _CML_PRIVATE_DEF_SEL(featuresAtIndex_, "featuresAtIndex:");

  // ---- MLMultiArray -----------------------------------------------
  _CML_PRIVATE_DEF_SEL(initWithShape_dataType_error_,
                       "initWithShape:dataType:error:");
  _CML_PRIVATE_DEF_SEL(
      initWithDataPointer_shape_dataType_strides_deallocator_error_,
      "initWithDataPointer:shape:dataType:strides:deallocator:error:");
  _CML_PRIVATE_DEF_SEL(dataType, "dataType");
  _CML_PRIVATE_DEF_SEL(shape, "shape");
  _CML_PRIVATE_DEF_SEL(strides, "strides");
  _CML_PRIVATE_DEF_SEL(dataPointer, "dataPointer");
  _CML_PRIVATE_DEF_SEL(getBytesWithHandler_, "getBytesWithHandler:");
  _CML_PRIVATE_DEF_SEL(getMutableBytesWithHandler_,
                       "getMutableBytesWithHandler:");
  _CML_PRIVATE_DEF_SEL(objectAtIndexedSubscript_,
                       "objectAtIndexedSubscript:");
  _CML_PRIVATE_DEF_SEL(setObject_atIndexedSubscript_,
                       "setObject:atIndexedSubscript:");
  _CML_PRIVATE_DEF_SEL(objectForKeyedSubscript_,
                       "objectForKeyedSubscript:");
  _CML_PRIVATE_DEF_SEL(setObject_forKeyedSubscript_,
                       "setObject:forKeyedSubscript:");
  _CML_PRIVATE_DEF_SEL(pixelBuffer, "pixelBuffer");

  // ---- MLPredictionOptions ----------------------------------------
  _CML_PRIVATE_DEF_SEL(usesCPUOnly, "usesCPUOnly");
  _CML_PRIVATE_DEF_SEL(setUsesCPUOnly_, "setUsesCPUOnly:");
  _CML_PRIVATE_DEF_SEL(outputBackings, "outputBackings");
  _CML_PRIVATE_DEF_SEL(setOutputBackings_, "setOutputBackings:");

  // ---- Constraints ------------------------------------------------
  _CML_PRIVATE_DEF_SEL(shapeConstraint, "shapeConstraint");
  _CML_PRIVATE_DEF_SEL(allowedShapes, "allowedShapes");
  _CML_PRIVATE_DEF_SEL(pixelFormatType, "pixelFormatType");
  _CML_PRIVATE_DEF_SEL(pixelsWide, "pixelsWide");
  _CML_PRIVATE_DEF_SEL(pixelsHigh, "pixelsHigh");
  _CML_PRIVATE_DEF_SEL(sizeConstraint, "sizeConstraint");
  _CML_PRIVATE_DEF_SEL(allowedKeyType, "allowedKeyType");
  _CML_PRIVATE_DEF_SEL(valueType, "valueType");
  _CML_PRIVATE_DEF_SEL(countConstraint, "countConstraint");
  _CML_PRIVATE_DEF_SEL(minValue, "minValue");
  _CML_PRIVATE_DEF_SEL(maxValue, "maxValue");
  _CML_PRIVATE_DEF_SEL(enumeratedNumbers, "enumeratedNumbers");

  // ---- MLSequence -------------------------------------------------
  _CML_PRIVATE_DEF_SEL(sequenceWithStringArray_,
                       "sequenceWithStringArray:");
  _CML_PRIVATE_DEF_SEL(sequenceWithInt64Array_,
                       "sequenceWithInt64Array:");
  _CML_PRIVATE_DEF_SEL(emptySequenceWithType_,
                       "emptySequenceWithType:");
  _CML_PRIVATE_DEF_SEL(stringValues, "stringValues");
  _CML_PRIVATE_DEF_SEL(int64Values, "int64Values");

  // ---- MLState ----------------------------------------------------
  _CML_PRIVATE_DEF_SEL(getMultiArrayForStateNamed_handler_,
                       "getMultiArrayForStateNamed:handler:");

  // ---- MLParameterKey / MLParameterDescription --------------------
  _CML_PRIVATE_DEF_SEL(scope, "scope");
  _CML_PRIVATE_DEF_SEL(key, "key");
  _CML_PRIVATE_DEF_SEL(numericConstraint, "numericConstraint");
  _CML_PRIVATE_DEF_SEL(defaultValue, "defaultValue");

  // ---- MLUpdateTask -----------------------------------------------
  _CML_PRIVATE_DEF_SEL(
      updateTaskForModelAtURL_trainingData_completionHandler_,
      "updateTaskForModelAtURL:trainingData:completionHandler:");
  _CML_PRIVATE_DEF_SEL(
      updateTaskForModelAtURL_trainingData_configuration_completionHandler_,
      "updateTaskForModelAtURL:trainingData:configuration:"
      "completionHandler:");
  _CML_PRIVATE_DEF_SEL(
      updateTaskForModelAtURL_trainingData_configuration_progressHandlers_,
      "updateTaskForModelAtURL:trainingData:configuration:"
      "progressHandlers:");
  _CML_PRIVATE_DEF_SEL(resume, "resume");
  _CML_PRIVATE_DEF_SEL(cancel, "cancel");
  _CML_PRIVATE_DEF_SEL(resumeWithParameters_, "resumeWithParameters:");
  _CML_PRIVATE_DEF_SEL(model, "model");

  // ---- MLUpdateContext --------------------------------------------
  _CML_PRIVATE_DEF_SEL(task, "task");
  _CML_PRIVATE_DEF_SEL(event, "event");
  _CML_PRIVATE_DEF_SEL(metrics, "metrics");
  _CML_PRIVATE_DEF_SEL(parameters_, "parameters");
  _CML_PRIVATE_DEF_SEL(modelURL, "modelURL");

  // ---- MLUpdateProgressHandlers -----------------------------------
  _CML_PRIVATE_DEF_SEL(
      initForEvents_progressHandler_completionHandler_,
      "initForEvents:progressHandler:completionHandler:");

  // ---- generic NSObject-ish helpers used by the wrapper ----------
  _CML_PRIVATE_DEF_SEL(alloc, "alloc");
  _CML_PRIVATE_DEF_SEL(init, "init");
  _CML_PRIVATE_DEF_SEL(fileURLWithPath_, "fileURLWithPath:");
}

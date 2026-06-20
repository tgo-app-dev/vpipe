// coreml-private.cc
//
// The single translation unit that owns:
//
//   1. The Foundation selector / class table (NS_PRIVATE_IMPLEMENTATION).
//   2. The CoreML selector / class table (CML_PRIVATE_IMPLEMENTATION).
//   3. Out-of-line implementations of the few wrapper methods that
//      cannot be done with a pure objc_msgSend -- those that need to
//      construct an Objective-C block from a C function pointer.
//
// Everything else in coreml-cpp is header-only: classes, methods,
// and selector lookups all inline at the point of use.

// vpipe provides the Foundation/Metal NS::Private::* selector / class
// tables (NS_PRIVATE_IMPLEMENTATION) here. VPIPE_FOUNDATION_FROM_MLX is
// not set, so the guard below is always taken. The CoreML-specific
// CML::Private::* tables stay owned by us regardless.
#ifndef VPIPE_FOUNDATION_FROM_MLX
#define NS_PRIVATE_IMPLEMENTATION
#endif
#define CML_PRIVATE_IMPLEMENTATION

#include "apple-silicon/coreml/coreml-cpp/CoreML.hpp"

#include <objc/message.h>
#include <objc/objc.h>
#include <objc/runtime.h>

#include <Block.h>

namespace CML {

// ---- Block trampolines ---------------------------------------------
//
// CoreML's async APIs take Obj-C blocks of the form
//   void (^)(MLModel *m, NSError *e)  /  void (^)(NSDictionary *m, NSError *e)
//
// Clang's blocks runtime works in plain C/C++ if the source file is
// compiled with -fblocks. We declare each block type and pass it
// straight through objc_msgSend. The user's C function pointer +
// userData get captured by the block.

namespace {

// MLModel async load: void (^)(MLModel *, NSError *)
using LoadBlock = void (^)(void* model, void* error);

// MLModel async predict: void (^)(id<MLFeatureProvider>, NSError *)
using PredictBlock = void (^)(void* output, void* error);

// MLState handler: void (^)(MLMultiArray *)
using StateMultiArrayBlock = void (^)(void* multiArray);

// MLUpdateProgressHandlers progress + completion:
// void (^)(MLUpdateContext *)
using UpdateContextBlock = void (^)(void* updateContext);

}  // namespace

// ---- MLModel async load --------------------------------------------

void Model::loadContentsOfURL(const NS::URL*            url,
                              const ModelConfiguration* configuration,
                              LoadCompletion            completion,
                              void*                     userData)
{
  LoadBlock block = ^(void* model, void* error) {
    completion(static_cast<Model*>(model),
               static_cast<NS::Error*>(error),
               userData);
  };
  NS::Object::sendMessage<void>(
      _CML_PRIVATE_CLS(MLModel),
      _CML_PRIVATE_SEL(
          loadContentsOfURL_configuration_completionHandler_),
      url, configuration, block);
}

void Model::loadModelAsset(const ModelAsset*         asset,
                           const ModelConfiguration* configuration,
                           LoadCompletion            completion,
                           void*                     userData)
{
  LoadBlock block = ^(void* model, void* error) {
    completion(static_cast<Model*>(model),
               static_cast<NS::Error*>(error),
               userData);
  };
  NS::Object::sendMessage<void>(
      _CML_PRIVATE_CLS(MLModel),
      _CML_PRIVATE_SEL(
          loadModelAsset_configuration_completionHandler_),
      asset, configuration, block);
}

// ---- MLModel async predict -----------------------------------------

void Model::predictionFromFeatures(const FeatureProvider* input,
                                   PredictCompletion       completion,
                                   void*                   userData)
{
  PredictBlock block = ^(void* output, void* error) {
    completion(static_cast<FeatureProvider*>(output),
               static_cast<NS::Error*>(error),
               userData);
  };
  NS::Object::sendMessage<void>(
      this,
      _CML_PRIVATE_SEL(predictionFromFeatures_completionHandler_),
      input, block);
}

void Model::predictionFromFeatures(const FeatureProvider*   input,
                                   const PredictionOptions* options,
                                   PredictCompletion        completion,
                                   void*                    userData)
{
  PredictBlock block = ^(void* output, void* error) {
    completion(static_cast<FeatureProvider*>(output),
               static_cast<NS::Error*>(error),
               userData);
  };
  NS::Object::sendMessage<void>(
      this,
      _CML_PRIVATE_SEL(
          predictionFromFeatures_options_completionHandler_),
      input, options, block);
}

// ---- MLState multi-array handler -----------------------------------

void State::getMultiArrayForStateNamed(const NS::String* name,
                                       MultiArrayHandler handler,
                                       void*             userData) const
{
  StateMultiArrayBlock block = ^(void* multiArray) {
    handler(static_cast<MultiArray*>(multiArray), userData);
  };
  NS::Object::sendMessage<void>(
      this, _CML_PRIVATE_SEL(getMultiArrayForStateNamed_handler_),
      name, block);
}

// ---- MLUpdateProgressHandlers init ---------------------------------

UpdateProgressHandlers* UpdateProgressHandlers::initForEvents(
    UpdateProgressEvent events,
    Handler             progressHandler,
    void*               progressUserData,
    Handler             completionHandler,
    void*               completionUserData)
{
  UpdateContextBlock progressBlock = ^(void* ctx) {
    progressHandler(static_cast<UpdateContext*>(ctx),
                    progressUserData);
  };
  UpdateContextBlock completionBlock = ^(void* ctx) {
    completionHandler(static_cast<UpdateContext*>(ctx),
                      completionUserData);
  };
  return NS::Object::sendMessage<UpdateProgressHandlers*>(
      this,
      _CML_PRIVATE_SEL(
          initForEvents_progressHandler_completionHandler_),
      static_cast<NS::UInteger>(events),
      progressBlock,
      completionBlock);
}

// ---- MLUpdateTask static factories ---------------------------------

UpdateTask* UpdateTask::updateTaskForModelAtURL(
    const NS::URL*       modelURL,
    const BatchProvider* trainingData,
    CompletionHandler    completion,
    void*                userData,
    NS::Error**          error)
{
  UpdateContextBlock block = ^(void* ctx) {
    completion(static_cast<UpdateContext*>(ctx), userData);
  };
  return NS::Object::sendMessage<UpdateTask*>(
      _CML_PRIVATE_CLS(MLUpdateTask),
      _CML_PRIVATE_SEL(
          updateTaskForModelAtURL_trainingData_completionHandler_),
      modelURL, trainingData, block, error);
}

UpdateTask* UpdateTask::updateTaskForModelAtURL(
    const NS::URL*            modelURL,
    const BatchProvider*      trainingData,
    const ModelConfiguration* configuration,
    CompletionHandler         completion,
    void*                     userData,
    NS::Error**               error)
{
  UpdateContextBlock block = ^(void* ctx) {
    completion(static_cast<UpdateContext*>(ctx), userData);
  };
  return NS::Object::sendMessage<UpdateTask*>(
      _CML_PRIVATE_CLS(MLUpdateTask),
      _CML_PRIVATE_SEL(
          updateTaskForModelAtURL_trainingData_configuration_completionHandler_),
      modelURL, trainingData, configuration, block, error);
}

UpdateTask* UpdateTask::updateTaskForModelAtURL(
    const NS::URL*                modelURL,
    const BatchProvider*          trainingData,
    const ModelConfiguration*     configuration,
    const UpdateProgressHandlers* progressHandlers,
    NS::Error**                   error)
{
  return NS::Object::sendMessage<UpdateTask*>(
      _CML_PRIVATE_CLS(MLUpdateTask),
      _CML_PRIVATE_SEL(
          updateTaskForModelAtURL_trainingData_configuration_progressHandlers_),
      modelURL, trainingData, configuration, progressHandlers, error);
}

}  // namespace CML

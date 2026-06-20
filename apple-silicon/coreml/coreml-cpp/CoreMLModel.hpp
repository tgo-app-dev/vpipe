// CoreMLModel.hpp -- MLModel wrapper.
//
// The central CoreML inference / update entry point. All factory
// methods that take URL pointers expect a compiled model bundle
// (.mlmodelc directory) -- to load a raw .mlmodel, first invoke
// compileModelAtURL() to produce a tmp .mlmodelc, then load that.
#pragma once

#include "CoreMLPrivate.hpp"

#include <Foundation/Foundation.hpp>

namespace CML {

class FeatureProvider;
class BatchProvider;
class ModelDescription;
class ModelConfiguration;
class ModelAsset;
class PredictionOptions;
class State;

class Model : public NS::Referencing<Model> {
public:
  // ---- Synchronous load ------------------------------------------
  static Model* modelWithContentsOfURL(const NS::URL* url,
                                       NS::Error**    error);

  static Model* modelWithContentsOfURL(
      const NS::URL*            url,
      const ModelConfiguration* configuration,
      NS::Error**               error);

  // Compile a raw .mlmodel at `srcURL` to a temp .mlmodelc bundle;
  // returns the file URL. Caller is responsible for cleanup, since
  // CoreML places the result in the user's tmp dir.
  static NS::URL* compileModelAtURL(const NS::URL* srcURL,
                                    NS::Error**    error);

  // ---- Asynchronous load -----------------------------------------
  using LoadCompletion =
      void (*)(Model* model, NS::Error* error, void* userData);

  static void loadContentsOfURL(
      const NS::URL*            url,
      const ModelConfiguration* configuration,
      LoadCompletion            completion,
      void*                     userData);

  static void loadModelAsset(
      const ModelAsset*         asset,
      const ModelConfiguration* configuration,
      LoadCompletion            completion,
      void*                     userData);

  // ---- Description -----------------------------------------------
  ModelDescription*   modelDescription() const;
  ModelConfiguration* configuration()    const;

  // ---- Synchronous predict ---------------------------------------
  FeatureProvider* predictionFromFeatures(
      const FeatureProvider* input,
      NS::Error**            error);

  FeatureProvider* predictionFromFeatures(
      const FeatureProvider*    input,
      const PredictionOptions*  options,
      NS::Error**               error);

  BatchProvider* predictionsFromBatch(const BatchProvider* batch,
                                      NS::Error**          error);

  BatchProvider* predictionsFromBatch(
      const BatchProvider*      batch,
      const PredictionOptions*  options,
      NS::Error**               error);

  // ---- Asynchronous predict --------------------------------------
  using PredictCompletion = void (*)(FeatureProvider* output,
                                     NS::Error*       error,
                                     void*            userData);

  void predictionFromFeatures(const FeatureProvider* input,
                              PredictCompletion       completion,
                              void*                   userData);

  void predictionFromFeatures(const FeatureProvider*    input,
                              const PredictionOptions*  options,
                              PredictCompletion         completion,
                              void*                     userData);

  // ---- State (CoreML 7+) -----------------------------------------
  State* newState();

  FeatureProvider* predictionFromFeatures(
      const FeatureProvider*    input,
      State*                    state,
      const PredictionOptions*  options,
      NS::Error**               error);

  // ---- Parameter access (updatable models) -----------------------
  NS::Object* parameterValueForKey(const NS::Object* key,
                                   NS::Error**       error) const;
};

}

// ---- Inline implementations ----------------------------------------

_CML_INLINE CML::Model*
CML::Model::modelWithContentsOfURL(const NS::URL* url,
                                   NS::Error**    error)
{
  return NS::Object::sendMessage<Model*>(
      _CML_PRIVATE_CLS(MLModel),
      _CML_PRIVATE_SEL(modelWithContentsOfURL_error_),
      url, error);
}

_CML_INLINE CML::Model*
CML::Model::modelWithContentsOfURL(
    const NS::URL*            url,
    const ModelConfiguration* configuration,
    NS::Error**               error)
{
  return NS::Object::sendMessage<Model*>(
      _CML_PRIVATE_CLS(MLModel),
      _CML_PRIVATE_SEL(modelWithContentsOfURL_configuration_error_),
      url, configuration, error);
}

_CML_INLINE NS::URL*
CML::Model::compileModelAtURL(const NS::URL* srcURL,
                              NS::Error**    error)
{
  return NS::Object::sendMessage<NS::URL*>(
      _CML_PRIVATE_CLS(MLModel),
      _CML_PRIVATE_SEL(compileModelAtURL_error_),
      srcURL, error);
}

_CML_INLINE CML::ModelDescription*
CML::Model::modelDescription() const
{
  return NS::Object::sendMessage<ModelDescription*>(
      this, _CML_PRIVATE_SEL(modelDescription));
}

_CML_INLINE CML::ModelConfiguration*
CML::Model::configuration() const
{
  return NS::Object::sendMessage<ModelConfiguration*>(
      this, _CML_PRIVATE_SEL(configuration));
}

_CML_INLINE CML::FeatureProvider*
CML::Model::predictionFromFeatures(const FeatureProvider* input,
                                   NS::Error**            error)
{
  return NS::Object::sendMessage<FeatureProvider*>(
      this, _CML_PRIVATE_SEL(predictionFromFeatures_error_),
      input, error);
}

_CML_INLINE CML::FeatureProvider*
CML::Model::predictionFromFeatures(
    const FeatureProvider*   input,
    const PredictionOptions* options,
    NS::Error**              error)
{
  return NS::Object::sendMessage<FeatureProvider*>(
      this, _CML_PRIVATE_SEL(predictionFromFeatures_options_error_),
      input, options, error);
}

_CML_INLINE CML::BatchProvider*
CML::Model::predictionsFromBatch(const BatchProvider* batch,
                                 NS::Error**          error)
{
  return NS::Object::sendMessage<BatchProvider*>(
      this, _CML_PRIVATE_SEL(predictionsFromBatch_error_),
      batch, error);
}

_CML_INLINE CML::BatchProvider*
CML::Model::predictionsFromBatch(
    const BatchProvider*     batch,
    const PredictionOptions* options,
    NS::Error**              error)
{
  return NS::Object::sendMessage<BatchProvider*>(
      this, _CML_PRIVATE_SEL(predictionsFromBatch_options_error_),
      batch, options, error);
}

_CML_INLINE CML::State* CML::Model::newState()
{
  return NS::Object::sendMessage<State*>(
      this, _CML_PRIVATE_SEL(newState));
}

_CML_INLINE CML::FeatureProvider*
CML::Model::predictionFromFeatures(
    const FeatureProvider*   input,
    State*                   state,
    const PredictionOptions* options,
    NS::Error**              error)
{
  return NS::Object::sendMessage<FeatureProvider*>(
      this,
      _CML_PRIVATE_SEL(
          predictionFromFeatures_usingState_options_error_),
      input, state, options, error);
}

_CML_INLINE NS::Object*
CML::Model::parameterValueForKey(const NS::Object* key,
                                 NS::Error**       error) const
{
  return NS::Object::sendMessage<NS::Object*>(
      this, _CML_PRIVATE_SEL(parameterValueForKey_error_),
      key, error);
}

// loadContentsOfURL / loadModelAsset / async predict variants live
// in coreml-private.cc -- they wrap function-pointer completion in
// Objective-C blocks.

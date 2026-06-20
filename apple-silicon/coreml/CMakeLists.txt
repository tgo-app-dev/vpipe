# apple-silicon/coreml: CoreML C++ wrapper and inference stage. Only
# built on APPLE; gated by VPIPE_BUILD_APPLE_SILICON in the root
# CMakeLists.

set(LOCAL_SOURCES
  coreml-cpp/coreml-private.cc
  coreml-model-manager.cc
)
set(LOCAL_HEADERS
  coreml-cpp/CoreML.hpp
  coreml-cpp/CoreMLDefines.hpp
  coreml-cpp/CoreMLPrivate.hpp
  coreml-cpp/CoreMLArrayBatchProvider.hpp
  coreml-cpp/CoreMLBatchProvider.hpp
  coreml-cpp/CoreMLDictionaryConstraint.hpp
  coreml-cpp/CoreMLDictionaryFeatureProvider.hpp
  coreml-cpp/CoreMLFeatureDescription.hpp
  coreml-cpp/CoreMLFeatureProvider.hpp
  coreml-cpp/CoreMLFeatureValue.hpp
  coreml-cpp/CoreMLImageConstraint.hpp
  coreml-cpp/CoreMLModel.hpp
  coreml-cpp/CoreMLModelAsset.hpp
  coreml-cpp/CoreMLModelConfiguration.hpp
  coreml-cpp/CoreMLModelDescription.hpp
  coreml-cpp/CoreMLMultiArray.hpp
  coreml-cpp/CoreMLMultiArrayConstraint.hpp
  coreml-cpp/CoreMLNumericConstraint.hpp
  coreml-cpp/CoreMLParameterDescription.hpp
  coreml-cpp/CoreMLParameterKey.hpp
  coreml-cpp/CoreMLPredictionOptions.hpp
  coreml-cpp/CoreMLSequence.hpp
  coreml-cpp/CoreMLSequenceConstraint.hpp
  coreml-cpp/CoreMLState.hpp
  coreml-cpp/CoreMLUpdateContext.hpp
  coreml-cpp/CoreMLUpdateProgressHandlers.hpp
  coreml-cpp/CoreMLUpdateTask.hpp
  coreml-model-manager.h
)

list(TRANSFORM LOCAL_SOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(TRANSFORM LOCAL_HEADERS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(APPEND VPIPE_SOURCES ${LOCAL_SOURCES})
list(APPEND VPIPE_HEADERS ${LOCAL_HEADERS})
set(VPIPE_SOURCES ${VPIPE_SOURCES} PARENT_SCOPE)
set(VPIPE_HEADERS ${VPIPE_HEADERS} PARENT_SCOPE)

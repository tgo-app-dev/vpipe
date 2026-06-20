set(LOCAL_SOURCES
  clock-domain.cc
  edge-reader.cc
  feedback-rx-stage.cc
  feedback-tx-stage.cc
  oport-buffer.cc
  pipeline-handle.cc
  pipeline-runtime.cc
  pipeline-spec.cc
  pipeline.cc
  stage-config.cc
  stage-registry.cc
  stage-spec.cc
  stage.cc
)
set(LOCAL_HEADERS
  clock-domain.h
  edge-reader.h
  feedback-rx-stage.h
  feedback-tx-stage.h
  oport-buffer.h
  pipeline-handle-impl.h
  pipeline-runtime.h
  pipeline-spec.h
  pipeline.h
  runtime-context.h
  stage-config.h
  stage-registry.h
  stage-spec.h
  stage.h
  typed-stage.h
)

list(TRANSFORM LOCAL_SOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(TRANSFORM LOCAL_HEADERS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(APPEND VPIPE_SOURCES ${LOCAL_SOURCES})
list(APPEND VPIPE_HEADERS ${LOCAL_HEADERS})
set(VPIPE_SOURCES ${VPIPE_SOURCES} PARENT_SCOPE)
set(VPIPE_HEADERS ${VPIPE_HEADERS} PARENT_SCOPE)

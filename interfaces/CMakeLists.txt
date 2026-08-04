set(LOCAL_SOURCES
  log-delegate-intf.cc
  session-context-intf.cc
  session-services-intf.cc
  ui-delegate-intf.cc
)
set(LOCAL_HEADERS
  fs-sandbox-intf.h
  log-delegate-intf.h
  log-sink-intf.h
  perf-sink-intf.h
  runtime-services-intf.h
  session-context-intf.h
  session-services-intf.h
  ui-delegate-intf.h
  ui-port-intf.h
  ui-view-intf.h
)

list(TRANSFORM LOCAL_SOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(TRANSFORM LOCAL_HEADERS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(APPEND VPIPE_SOURCES ${LOCAL_SOURCES})
list(APPEND VPIPE_HEADERS ${LOCAL_HEADERS})
set(VPIPE_SOURCES ${VPIPE_SOURCES} PARENT_SCOPE)
set(VPIPE_HEADERS ${VPIPE_HEADERS} PARENT_SCOPE)

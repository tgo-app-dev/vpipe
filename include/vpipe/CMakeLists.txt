set(LOCAL_HEADERS
  pipeline-handle.h
  session-intf.h
  session-manager.h
  status.h
  vpipe.h
)

list(TRANSFORM LOCAL_HEADERS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(APPEND VPIPE_HEADERS ${LOCAL_HEADERS})
set(VPIPE_HEADERS ${VPIPE_HEADERS} PARENT_SCOPE)

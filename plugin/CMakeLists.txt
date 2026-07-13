# plugin: the runtime plugin system -- the ABI contract (plugin-abi.h),
# the C++ registration facade (plugin-context), and the dlopen loader
# (plugin-manager). Compiles into libvpipe so a plugin .dylib that links
# libvpipe observes the same registry singletons. See docs/PLUGINS.md.
set(LOCAL_SOURCES
  plugin-context.cc
  plugin-manager.cc
)
set(LOCAL_HEADERS
  plugin-abi.h
  plugin-context.h
  plugin-manager.h
)

list(TRANSFORM LOCAL_SOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(TRANSFORM LOCAL_HEADERS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(APPEND VPIPE_SOURCES ${LOCAL_SOURCES})
list(APPEND VPIPE_HEADERS ${LOCAL_HEADERS})
set(VPIPE_SOURCES ${VPIPE_SOURCES} PARENT_SCOPE)
set(VPIPE_HEADERS ${VPIPE_HEADERS} PARENT_SCOPE)

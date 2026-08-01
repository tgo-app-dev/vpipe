# ui: stage-provided GUI views -- the front-end panels a stage ships
# with itself instead of with the web-ui application.
#
#   ui-view-registry.{h,cc}  the backend catalogue (specs + assets)
#   sdk/                     the module contract every view imports
#   stages/<stage>/          one directory per view: its .js / .css /
#                            message catalogue AND its C++ backend
#
# Everything here links into libvpipe (VPIPE_SOURCES), and the .js/.css
# are embedded into that same library by embed-ui-assets.cmake below --
# so a front-end app serves a view by asking the registry, and a plugin
# .dylib can contribute views the app has never heard of.

set(LOCAL_SOURCES
  ui-view-registry.cc
  stages/preview/preview-view-backend.cc
  stages/compare-image/compare-image-view-backend.cc
  stages/create-mask/create-mask-view-backend.cc
)
set(LOCAL_HEADERS
  ui-view-registry.h
)

# Embed the front-end asset tree into libvpipe. CONFIGURE_DEPENDS
# re-globs at build time so adding/removing a .js or .css is picked up
# without a manual reconfigure; editing an existing one regenerates via
# the DEPENDS list.
file(GLOB_RECURSE VPIPE_UI_ASSET_FILES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/*.js"
     "${CMAKE_CURRENT_SOURCE_DIR}/*.css")
set(VPIPE_UI_ASSETS_CC
    "${CMAKE_CURRENT_BINARY_DIR}/ui-assets-generated.cc")
add_custom_command(
    OUTPUT  "${VPIPE_UI_ASSETS_CC}"
    COMMAND "${CMAKE_COMMAND}"
            -D "UI_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
            -D "OUTPUT=${VPIPE_UI_ASSETS_CC}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed-ui-assets.cmake"
    DEPENDS ${VPIPE_UI_ASSET_FILES}
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed-ui-assets.cmake"
    COMMENT "Embedding stage UI view assets into libvpipe"
    VERBATIM)

# The `vpipe` target is defined at ROOT scope, so it cannot see a custom
# command rule that lives only in this directory's makefile. Wrap the
# rule in a target the root can depend on, and mark the output GENERATED
# (a global property since CMake 3.20) so the root-scope object rule
# doesn't demand a producer of its own.
add_custom_target(vpipe_ui_assets DEPENDS "${VPIPE_UI_ASSETS_CC}")
set_source_files_properties("${VPIPE_UI_ASSETS_CC}"
                            PROPERTIES GENERATED TRUE)

list(TRANSFORM LOCAL_SOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(TRANSFORM LOCAL_HEADERS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(APPEND VPIPE_SOURCES ${LOCAL_SOURCES} "${VPIPE_UI_ASSETS_CC}")
list(APPEND VPIPE_HEADERS ${LOCAL_HEADERS})
set(VPIPE_SOURCES ${VPIPE_SOURCES} PARENT_SCOPE)
set(VPIPE_HEADERS ${VPIPE_HEADERS} PARENT_SCOPE)

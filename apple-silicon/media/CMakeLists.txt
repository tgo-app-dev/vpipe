# apple-silicon/media: stills and video through Apple's own frameworks --
# ImageIO for 16-bit PNG/TIFF, OpenEXR and HEIC, AVFoundation +
# VideoToolbox for ProRes (4444 with alpha) and HEVC Main10 (PQ/HLG) --
# so a picture keeps its bits, its alpha and its colour tags, which the
# FFmpeg path (8-bit planar RGB, BT.601/709 only) does not. Pictures
# cross as planar [C,H,W] TensorBeats, U8 or F16, tagged with the
# sideband colour keys of common/beat-keys.h.
#
# Apple-Silicon only; gated by VPIPE_BUILD_APPLE_SILICON in the parent
# apple-silicon/CMakeLists.txt. The frameworks are linked in the root
# CMakeLists (vpipe_link).

set(LOCAL_SOURCES
  color-tags.cc
  image-io.mm
  planar.cc
  video-io.mm
)
set(LOCAL_HEADERS
  color-tags.h
  image-io.h
  planar.h
  video-io.h
)
list(TRANSFORM LOCAL_HEADERS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(TRANSFORM LOCAL_SOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
# These files own Cocoa objects: ARC, set where the target lives (the
# root directory), since source properties are directory-scoped.
set(VPIPE_ARC_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/image-io.mm"
  "${CMAKE_CURRENT_SOURCE_DIR}/video-io.mm")
set_source_files_properties(${VPIPE_ARC_SOURCES}
  DIRECTORY ${PROJECT_SOURCE_DIR}
  PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
list(APPEND VPIPE_HEADERS ${LOCAL_HEADERS})
list(APPEND VPIPE_SOURCES ${LOCAL_SOURCES})

set(VPIPE_SOURCES ${VPIPE_SOURCES} PARENT_SCOPE)
set(VPIPE_HEADERS ${VPIPE_HEADERS} PARENT_SCOPE)

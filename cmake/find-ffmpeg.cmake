# Locate FFmpeg headers. We dlopen the libraries at runtime so we do not
# need their .so/.dylib at link time -- only the public headers, to type
# the function pointers in the curated `api` struct of each LibAv*Handle.
#
# Search order:
#   1. pkg-config (if available) -- fastest and most accurate.
#   2. find_path with hints for common Homebrew / Linux locations.
#
# On success, defines:
#   VPIPE_FFMPEG_INCLUDE_DIRS  - list of include dirs (deduplicated).
# On failure, fails with FATAL_ERROR.

set(_vpipe_ffmpeg_hints
  /opt/homebrew/include
  /usr/local/include
  /usr/include
)

find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
  pkg_check_modules(_vpipe_ffmpeg_pc QUIET
    libavformat
    libavcodec
    libavutil
    libswresample
  )
  if(_vpipe_ffmpeg_pc_FOUND)
    set(VPIPE_FFMPEG_INCLUDE_DIRS ${_vpipe_ffmpeg_pc_INCLUDE_DIRS})
  endif()
endif()

if(NOT VPIPE_FFMPEG_INCLUDE_DIRS)
  find_path(_vpipe_avformat_inc
    NAMES libavformat/avformat.h
    HINTS ${_vpipe_ffmpeg_hints})
  find_path(_vpipe_avcodec_inc
    NAMES libavcodec/avcodec.h
    HINTS ${_vpipe_ffmpeg_hints})
  find_path(_vpipe_avutil_inc
    NAMES libavutil/avutil.h
    HINTS ${_vpipe_ffmpeg_hints})
  find_path(_vpipe_swresample_inc
    NAMES libswresample/swresample.h
    HINTS ${_vpipe_ffmpeg_hints})

  if(NOT _vpipe_avformat_inc
     OR NOT _vpipe_avcodec_inc
     OR NOT _vpipe_avutil_inc
     OR NOT _vpipe_swresample_inc)
    message(FATAL_ERROR
      "FFmpeg headers not found. Install ffmpeg dev packages "
      "(e.g. 'brew install ffmpeg' on macOS, "
      "'apt install libavformat-dev libavcodec-dev libavutil-dev "
      "libswresample-dev' on Debian/Ubuntu).")
  endif()

  set(VPIPE_FFMPEG_INCLUDE_DIRS
    ${_vpipe_avformat_inc}
    ${_vpipe_avcodec_inc}
    ${_vpipe_avutil_inc}
    ${_vpipe_swresample_inc})
endif()

list(REMOVE_DUPLICATES VPIPE_FFMPEG_INCLUDE_DIRS)
message(STATUS "FFmpeg headers: ${VPIPE_FFMPEG_INCLUDE_DIRS}")

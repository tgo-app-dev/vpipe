# Locate FFmpeg headers. We dlopen the libraries at runtime so we do not
# need their .so/.dylib at link time -- only the public headers, to type
# the function pointers in the curated `api` struct of each LibAv*Handle.
#
# Search order:
#   1. pkg-config (if available) -- fastest and most accurate.
#   2. find_path with hints for common Homebrew / Linux locations.
#
# Both answers are then VALIDATED and CANONICALIZED, because a package
# manager moves headers out from under an already-configured build tree:
#
#   * Homebrew's pkg-config answers with a VERSION-PINNED Cellar path
#     (.../Cellar/ffmpeg/8.1.1/include). `brew upgrade ffmpeg` deletes that
#     directory -- and pkg_check_modules and find_path BOTH cache their
#     answer in the CMake cache and never look again, so re-running cmake
#     does not repair it. The tree goes on compiling with an -I that no
#     longer exists; the compiler ignores a missing -I silently, so this
#     surfaces as "libavformat/avformat.h not found" while the install on
#     disk looks perfectly healthy. MEASURED after an 8.1.1 -> 8.1.2_1
#     upgrade on this machine.
#
# So a Cellar path is rewritten to Homebrew's per-formula symlink
# (<prefix>/opt/ffmpeg/include), which is version-independent and which
# brew repoints on every upgrade; a cached path whose headers have since
# vanished is DISCARDED and re-detected, which heals a tree that was
# configured before the upgrade; and the pkg-config file is registered as
# a configure dependency, so the next `brew upgrade` re-runs this file by
# itself rather than waiting for someone to notice the breakage.
#
# On success, defines:
#   VPIPE_FFMPEG_INCLUDE_DIRS  - list of include dirs (deduplicated).
# On failure, fails with FATAL_ERROR.

set(_vpipe_ffmpeg_hints
  /opt/homebrew/include
  /usr/local/include
  /usr/include
)

# The headers that must be reachable. One list drives detection, staleness
# checking and the final assertion, so the three cannot drift apart.
set(_vpipe_ffmpeg_headers
  libavformat/avformat.h
  libavcodec/avcodec.h
  libavutil/avutil.h
  libswresample/swresample.h
)

# The cache entries the two search paths write. Dropping these is what
# makes a stale detection re-run instead of being served from the cache.
set(_vpipe_ffmpeg_cache_vars
  _vpipe_avformat_inc
  _vpipe_avcodec_inc
  _vpipe_avutil_inc
  _vpipe_swresample_inc
)

# TRUE when every header in `headers` is present somewhere in `dirs`.
function(_vpipe_ffmpeg_complete dirs headers out)
  foreach(_h IN LISTS headers)
    set(_found FALSE)
    foreach(_d IN LISTS dirs)
      if(NOT "${_d}" STREQUAL "" AND EXISTS "${_d}/${_h}")
        set(_found TRUE)
        break()
      endif()
    endforeach()
    if(NOT _found)
      set(${out} FALSE PARENT_SCOPE)
      return()
    endif()
  endforeach()
  set(${out} TRUE PARENT_SCOPE)
endfunction()

# Rewrite a Homebrew Cellar path to the formula's version-independent
# symlink, so a tree configured today survives the next upgrade:
#   /opt/homebrew/Cellar/ffmpeg/8.1.2_1/include
#     -> /opt/homebrew/opt/ffmpeg/include
# Left untouched when the symlink is absent or does not expose the same
# header -- a real path that works beats a tidy one that might not.
function(_vpipe_ffmpeg_stabilize path probe_header out)
  set(${out} "${path}" PARENT_SCOPE)
  if(NOT "${path}" MATCHES "^(.+)/Cellar/([^/]+)/[^/]+/include$")
    return()
  endif()
  set(_stable "${CMAKE_MATCH_1}/opt/${CMAKE_MATCH_2}/include")
  if(EXISTS "${_stable}/${probe_header}")
    set(${out} "${_stable}" PARENT_SCOPE)
  endif()
endfunction()

# ---- 1. drop anything cached that no longer resolves -------------------
# The staleness lives in the CACHE, not in VPIPE_FFMPEG_INCLUDE_DIRS --
# that one is a normal variable and is empty again on every configure, so
# checking it here would inspect nothing. What survives a re-run is
# pkg_check_modules' own cached answer (plus the guard that makes it skip
# the query entirely) and find_path's cached PATHs. Those are what a
# package-manager upgrade invalidates, so those are what get validated.
set(_vpipe_ffmpeg_cached "")
if(DEFINED _vpipe_ffmpeg_pc_INCLUDE_DIRS)
  list(APPEND _vpipe_ffmpeg_cached ${_vpipe_ffmpeg_pc_INCLUDE_DIRS})
endif()
foreach(_v IN LISTS _vpipe_ffmpeg_cache_vars)
  if(${_v})
    list(APPEND _vpipe_ffmpeg_cached "${${_v}}")
  endif()
endforeach()

if(_vpipe_ffmpeg_cached)
  # Every cached directory must still exist. Not "the set as a whole
  # resolves" -- one dead entry is one dead -I on the compile line, and
  # the compiler discards a missing -I without a word.
  set(_vpipe_ffmpeg_ok TRUE)
  foreach(_d IN LISTS _vpipe_ffmpeg_cached)
    if(NOT IS_DIRECTORY "${_d}")
      set(_vpipe_ffmpeg_ok FALSE)
      set(_vpipe_ffmpeg_dead "${_d}")
    endif()
  endforeach()
  if(NOT _vpipe_ffmpeg_ok)
    message(STATUS
      "FFmpeg headers moved since the last configure "
      "(${_vpipe_ffmpeg_dead} is gone) -- searching again")
    # pkg_check_modules short-circuits on this guard; clearing it is what
    # makes the query below actually re-run rather than replay the old
    # answer. The _pc_ result vars go too, so a partial re-run cannot
    # leave half of the previous install behind.
    unset(__pkg_config_checked__vpipe_ffmpeg_pc CACHE)
    foreach(_s IN ITEMS INCLUDE_DIRS CFLAGS CFLAGS_I CFLAGS_OTHER
                        LDFLAGS LDFLAGS_OTHER LIBRARIES LIBRARY_DIRS
                        FOUND INCLUDEDIR LIBDIR VERSION)
      unset(_vpipe_ffmpeg_pc_${_s} CACHE)
      unset(_vpipe_ffmpeg_pc_${_s})
    endforeach()
    foreach(_v IN LISTS _vpipe_ffmpeg_cache_vars)
      unset(${_v} CACHE)
    endforeach()
  endif()
endif()

# ---- 2. pkg-config -----------------------------------------------------
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND AND NOT VPIPE_FFMPEG_INCLUDE_DIRS)
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

# ---- 3. direct search --------------------------------------------------
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

  if(_vpipe_avformat_inc
     AND _vpipe_avcodec_inc
     AND _vpipe_avutil_inc
     AND _vpipe_swresample_inc)
    set(VPIPE_FFMPEG_INCLUDE_DIRS
      ${_vpipe_avformat_inc}
      ${_vpipe_avcodec_inc}
      ${_vpipe_avutil_inc}
      ${_vpipe_swresample_inc})
  endif()
endif()

# ---- 4. canonicalize + assert -----------------------------------------
set(_vpipe_ffmpeg_stable "")
foreach(_d IN LISTS VPIPE_FFMPEG_INCLUDE_DIRS)
  _vpipe_ffmpeg_stabilize("${_d}" "libavformat/avformat.h" _s)
  list(APPEND _vpipe_ffmpeg_stable "${_s}")
endforeach()
set(VPIPE_FFMPEG_INCLUDE_DIRS ${_vpipe_ffmpeg_stable})

_vpipe_ffmpeg_complete("${VPIPE_FFMPEG_INCLUDE_DIRS}"
  "${_vpipe_ffmpeg_headers}" _vpipe_ffmpeg_ok)
if(NOT _vpipe_ffmpeg_ok)
  message(FATAL_ERROR
    "FFmpeg headers not found. Install ffmpeg dev packages "
    "(e.g. 'brew install ffmpeg' on macOS, "
    "'apt install libavformat-dev libavcodec-dev libavutil-dev "
    "libswresample-dev' on Debian/Ubuntu).")
endif()

list(REMOVE_DUPLICATES VPIPE_FFMPEG_INCLUDE_DIRS)

# ---- 5. re-configure when the install changes --------------------------
# Watch the pkg-config file: a package manager rewrites (or repoints) it on
# upgrade, so the next BUILD re-runs cmake by itself and picks the new
# location up. That is the difference between this being automatic and
# being something a human diagnoses from a confusing compile error.
# MEASURED: CMAKE_CONFIGURE_DEPENDS fires both when a watched file is
# touched and when it is deleted.
#
# Watch the VERSION-INDEPENDENT path by preference. The obvious candidate,
# `pkg-config --variable=pcfiledir`, answers with the version-pinned Cellar
# directory -- and Homebrew keeps an old keg until `brew cleanup` runs, so
# that file can outlive the upgrade it is supposed to announce and the
# trigger silently never fires. The symlinks below are repointed at upgrade
# time whether or not the old version lingers.
set(_vpipe_ffmpeg_watch "")
foreach(_d IN LISTS VPIPE_FFMPEG_INCLUDE_DIRS)
  if("${_d}" MATCHES "^(.+)/opt/([^/]+)/include$")
    # <prefix>/opt/<formula>/... and <prefix>/lib/... are both stable; take
    # both, since which one a given package manager rewrites is its business.
    list(APPEND _vpipe_ffmpeg_watch
      "${CMAKE_MATCH_1}/opt/${CMAKE_MATCH_2}/lib/pkgconfig/libavformat.pc"
      "${CMAKE_MATCH_1}/lib/pkgconfig/libavformat.pc")
  endif()
endforeach()
if(NOT _vpipe_ffmpeg_watch AND PKG_CONFIG_FOUND)
  execute_process(
    COMMAND ${PKG_CONFIG_EXECUTABLE} --variable=pcfiledir libavformat
    OUTPUT_VARIABLE _vpipe_ffmpeg_pcdir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(_vpipe_ffmpeg_pcdir)
    list(APPEND _vpipe_ffmpeg_watch
      "${_vpipe_ffmpeg_pcdir}/libavformat.pc")
  endif()
endif()
foreach(_w IN LISTS _vpipe_ffmpeg_watch)
  if(EXISTS "${_w}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_w}")
  endif()
endforeach()

message(STATUS "FFmpeg headers: ${VPIPE_FFMPEG_INCLUDE_DIRS}")

# ---- 6. an ABI tag that forces a rebuild when the ABI moves ----------
# Read the soname majors out of the headers and hand them back so the
# build can put them ON THE COMPILE LINE. Changing a compile definition
# makes CMake rebuild every TU of the target, which is the point.
#
# WHY THIS EXISTS. vpipe dlopens FFmpeg but compiles against its
# headers, so the struct layouts baked into our objects must match the
# libraries loaded at run time. Header mtimes cannot be trusted to
# enforce that: Homebrew's bottles preserve UPSTREAM mtimes, so an
# upgrade can install headers that are OLDER than the objects already
# built against the previous version. make then sees everything as up
# to date and recompiles nothing.
#
# MEASURED, on the 7.1.1 -> 8.1.2 upgrade: the 8.1.2 headers arrived
# dated 2026-06-16, the existing objects were from 2026-08-09, and only
# the one file that had been edited by hand was rebuilt. The resulting
# libvpipe mixed 7.1.1 struct layouts with 8.1.2 libraries -- which does
# not fail at load (dlopen resolves every symbol by name) but corrupts
# every call that passes a struct. It surfaced as avcodec_open2 refusing
# a valid context with EINVAL, empty output files, and "decoded 0
# frames": a full day of confusing symptoms with no error naming the
# cause.
set(_vpipe_ffmpeg_abi "")
foreach(_pair "libavutil;LIBAVUTIL" "libavcodec;LIBAVCODEC"
              "libavformat;LIBAVFORMAT" "libswresample;LIBSWRESAMPLE"
              "libswscale;LIBSWSCALE")
  list(GET _pair 0 _dir)
  list(GET _pair 1 _prefix)
  set(_major "")
  foreach(_d IN LISTS VPIPE_FFMPEG_INCLUDE_DIRS)
    # Newer FFmpeg splits the major out into version_major.h; older
    # keeps it in version.h. Check both rather than assuming.
    foreach(_f "${_d}/${_dir}/version_major.h" "${_d}/${_dir}/version.h")
      if(NOT _major AND EXISTS "${_f}")
        file(STRINGS "${_f}" _line
             REGEX "^#define[ \t]+${_prefix}_VERSION_MAJOR[ \t]+[0-9]+")
        if(_line)
          string(REGEX MATCH "[0-9]+$" _major "${_line}")
        endif()
      endif()
    endforeach()
  endforeach()
  if(NOT _major)
    set(_major "x")
  endif()
  # NOT ";" as a separator: CMake treats a semicolon-bearing string as a
  # LIST, so the value would expand into several -D flags and break the
  # compile command outright ("sh: syntax error: unexpected end of
  # file"). A dash keeps it one token.
  string(APPEND _vpipe_ffmpeg_abi "${_dir}${_major}-")
endforeach()

set(VPIPE_FFMPEG_ABI_TAG "${_vpipe_ffmpeg_abi}")
message(STATUS "FFmpeg ABI: ${VPIPE_FFMPEG_ABI_TAG}")

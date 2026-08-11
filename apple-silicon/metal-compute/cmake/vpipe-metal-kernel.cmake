# vpipe-metal-kernel.cmake -- helper to declare a Metal compute
# kernel. Two build modes, chosen automatically (override with
# -DVPIPE_METAL_RUNTIME_COMPILE=ON/OFF):
#
#   * build-time metallib (default when the Metal toolchain is present):
#     compiles `.metal` -> `.air` -> `.metallib` via xcrun, then emits a
#     generated `.cc` that registers the compiled metallib BYTES with
#     MetalCompute's embedded registry at static-init.
#
#   * runtime-compile (fallback when metal/metallib are unavailable, e.g.
#     a Command-Line-Tools-only box with no Xcode Metal Toolchain): embeds
#     the include-flattened `.metal` SOURCE instead, and MetalCompute
#     compiles it via newLibraryWithSource: on first load_library().
#
# The generated `.cc` is appended to the caller's VPIPE_SOURCES list
# so it links into libvpipe.dylib alongside the framework -- no
# on-disk metallib at install time.
#
# Usage:
#   include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/vpipe-metal-kernel.cmake)
#   add_vpipe_metal_kernel(noop  SRC kernels/noop.metal)
#   add_vpipe_metal_kernel(saxpy SRC kernels/saxpy.metal
#                                DEPENDS msl/vpipe_tensor_view.metal)
#
# The first call resolves xcrun once and caches its path. Subsequent
# calls reuse it.

find_program(XCRUN_EXECUTABLE xcrun)

# Decide build-time-metallib vs runtime-compile mode. The offline Metal
# shader compiler (the `metal` + `metallib` xcrun subcommands) ships only
# with Xcode's Metal Toolchain; a Command-Line-Tools-only box lacks it.
# When it's missing, fall back to embedding source and compiling at load.
# The user can force either mode with -DVPIPE_METAL_RUNTIME_COMPILE=ON/OFF.
if(NOT DEFINED VPIPE_METAL_RUNTIME_COMPILE)
  set(_vp_have_metal FALSE)
  if(XCRUN_EXECUTABLE)
    execute_process(COMMAND ${XCRUN_EXECUTABLE} -sdk macosx -f metal
        RESULT_VARIABLE _vp_m_rc OUTPUT_QUIET ERROR_QUIET)
    execute_process(COMMAND ${XCRUN_EXECUTABLE} -sdk macosx -f metallib
        RESULT_VARIABLE _vp_ml_rc OUTPUT_QUIET ERROR_QUIET)
    if(_vp_m_rc EQUAL 0 AND _vp_ml_rc EQUAL 0)
      set(_vp_have_metal TRUE)
    endif()
  endif()
  if(_vp_have_metal)
    set(VPIPE_METAL_RUNTIME_COMPILE OFF CACHE BOOL
        "Embed .metal source + compile at runtime (no Metal toolchain needed)")
  else()
    set(VPIPE_METAL_RUNTIME_COMPILE ON CACHE BOOL
        "Embed .metal source + compile at runtime (no Metal toolchain needed)")
  endif()
endif()

if(VPIPE_METAL_RUNTIME_COMPILE)
  message(STATUS
      "vpipe Metal kernels: RUNTIME-COMPILE mode -- embedding .metal source; "
      "kernels compiled via newLibraryWithSource: at load (no metal/metallib "
      "toolchain required)")
elseif(NOT XCRUN_EXECUTABLE)
  message(FATAL_ERROR
      "xcrun not found in PATH; required to compile vpipe Metal kernels "
      "(or set -DVPIPE_METAL_RUNTIME_COMPILE=ON to embed source and compile "
      "at runtime instead)")
else()
  message(STATUS
      "vpipe Metal kernels: build-time metallib mode (xcrun metal/metallib)")
endif()

# Resolve the directory this file lives in so we can hand off to
# embed-metallib.cmake. CMAKE_CURRENT_FUNCTION_LIST_DIR is set per
# function invocation, so we must capture it at include time.
set(_VPIPE_METAL_KERNEL_DIR "${CMAKE_CURRENT_LIST_DIR}")

# A stamp file carrying the Metal compiler version, added to every .air
# command's DEPENDS.
#
# Without it the only dependencies are the .metal source and its headers,
# so updating Xcode's Metal Toolchain rebuilds NOTHING: every .air is
# newer than its unchanged source, and the build silently keeps shipping
# objects from the old compiler. That is not hypothetical -- the matrix-
# core kernels are the ones most likely to need a toolchain fix, and they
# are also the ones no pre-M5 machine can test, so a stale metallib would
# reach a release with nothing in between to catch it.
#
# file(WRITE) only touches the stamp when the content differs, so this
# costs a full metallib rebuild exactly when the compiler changed and
# nothing otherwise. It updates at CONFIGURE time: after a toolchain
# update, re-run cmake (the release script does).
if(NOT VPIPE_METAL_RUNTIME_COMPILE AND XCRUN_EXECUTABLE)
  set(VPIPE_METAL_TOOLCHAIN_STAMP
      "${CMAKE_BINARY_DIR}/metal-toolchain-version.stamp")
  execute_process(COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metal --version
      RESULT_VARIABLE _vp_ver_rc
      OUTPUT_VARIABLE _vp_metal_ver ERROR_VARIABLE _vp_metal_err
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(REGEX MATCH "metalfe-[0-9.]+" _vp_metal_tag "${_vp_metal_ver}")

  # Only a version we actually PARSED may touch the stamp. xcrun fails in
  # ways that still print to stdout -- an un-accepted Xcode licence after
  # an update is the common one -- and writing that text into the stamp
  # would both force a spurious full rebuild and record a "toolchain"
  # that is really an error message.
  if(_vp_ver_rc EQUAL 0 AND _vp_metal_tag)
    set(_vp_stamp_old "")
    if(EXISTS "${VPIPE_METAL_TOOLCHAIN_STAMP}")
      file(READ "${VPIPE_METAL_TOOLCHAIN_STAMP}" _vp_stamp_old)
    endif()
    if(NOT _vp_stamp_old STREQUAL "${_vp_metal_ver}")
      file(WRITE "${VPIPE_METAL_TOOLCHAIN_STAMP}" "${_vp_metal_ver}")
      if(_vp_stamp_old)
        message(STATUS
            "vpipe Metal kernels: toolchain changed -- rebuilding all metallibs")
      endif()
    endif()
    message(STATUS "vpipe Metal compiler: ${_vp_metal_tag}")
  else()
    if(NOT EXISTS "${VPIPE_METAL_TOOLCHAIN_STAMP}")
      file(WRITE "${VPIPE_METAL_TOOLCHAIN_STAMP}" "unknown")
    endif()
    message(WARNING
        "vpipe: could not read the Metal compiler version -- metallibs will "
        "NOT rebuild if the toolchain changed. Fix and re-run cmake.\n"
        "${_vp_metal_ver}${_vp_metal_err}")
  endif()
endif()

function(add_vpipe_metal_kernel KERNEL_NAME)
  cmake_parse_arguments(K "" "SRC;STD" "DEPENDS;DEFINES;FLAGS" ${ARGN})
  if(NOT K_SRC)
    message(FATAL_ERROR
        "add_vpipe_metal_kernel(${KERNEL_NAME}): SRC argument required")
  endif()

  # Optional -D preprocessor defines (e.g. DEFINES VPIPE_ELT=bfloat to
  # compile a bf16 variant of an element-type-generic kernel file).
  set(DEFINE_FLAGS "")
  foreach(D ${K_DEFINES})
    list(APPEND DEFINE_FLAGS "-D${D}")
  endforeach()

  # Optional Metal language standard (e.g. STD metal4.0) and extra
  # compile flags (FLAGS ...). The matrix-core kernels need metal4.0 so
  # the tensor / MetalPerformancePrimitives (matmul2d) intrinsics and
  # the __HAVE_TENSOR__ paths are enabled; the default-std kernels omit
  # both and compile exactly as before.
  set(STD_FLAG "")
  if(K_STD)
    set(STD_FLAG "-std=${K_STD}")
  endif()

  # Base dir for kernel sources, DEPENDS, and the -I include roots. The
  # .metal tree lives at the top-level gpu-kernels/metal/; the caller sets
  # VPIPE_METAL_KERNEL_DIR to it. Falls back to this CMakeLists' dir.
  if(NOT DEFINED VPIPE_METAL_KERNEL_DIR)
    set(VPIPE_METAL_KERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  endif()

  set(SRC      "${VPIPE_METAL_KERNEL_DIR}/${K_SRC}")
  set(AIR      "${CMAKE_CURRENT_BINARY_DIR}/${KERNEL_NAME}.air")
  set(LIB      "${CMAKE_CURRENT_BINARY_DIR}/${KERNEL_NAME}.metallib")
  set(EMBED_CC "${CMAKE_CURRENT_BINARY_DIR}/${KERNEL_NAME}_embed.cc")

  set(DEPENDS_ABS "")
  foreach(D ${K_DEPENDS})
    list(APPEND DEPENDS_ABS "${VPIPE_METAL_KERNEL_DIR}/${D}")
  endforeach()

  if(VPIPE_METAL_RUNTIME_COMPILE)
    # --- runtime-compile: embed include-flattened SOURCE ---------------
    # Map STD (e.g. metal4.0) to a numeric language-version code the
    # runtime compiler applies via MTLCompileOptions.
    set(LANG 0)
    if(K_STD MATCHES "^metal([0-9]+)\\.([0-9]+)$")
      math(EXPR LANG "${CMAKE_MATCH_1} * 10 + ${CMAKE_MATCH_2}")
    endif()

    # The AOT compiler resolves quoted includes against these -I roots;
    # replicate them for the flattener. '|'-join so the lists survive as
    # single args through add_custom_command.
    string(JOIN "|" _INC_ARG
        "${VPIPE_METAL_KERNEL_DIR}" "${VPIPE_METAL_KERNEL_DIR}/vendored")
    set(_DEF_ARG "")
    if(K_DEFINES)
      string(JOIN "|" _DEF_ARG ${K_DEFINES})
    endif()

    add_custom_command(
      OUTPUT ${EMBED_CC}
      COMMAND ${CMAKE_COMMAND}
              -D KERNEL_NAME=${KERNEL_NAME}
              -D SRC=${SRC}
              -D OUTPUT=${EMBED_CC}
              -D INCLUDE_DIRS=${_INC_ARG}
              -D DEFINES=${_DEF_ARG}
              -D LANG=${LANG}
              -P ${_VPIPE_METAL_KERNEL_DIR}/embed-metal-source.cmake
      DEPENDS "${SRC}" ${DEPENDS_ABS}
              ${_VPIPE_METAL_KERNEL_DIR}/embed-metal-source.cmake
      COMMENT "Embedding ${KERNEL_NAME}.metal SOURCE (runtime-compile)"
      VERBATIM
    )
  else()
    # --- build-time: compile .metal -> .air -> .metallib, embed bytes --
    add_custom_command(
      OUTPUT ${AIR}
      COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metal
              -gline-tables-only -frecord-sources
              -Wall -Wextra -fno-fast-math ${STD_FLAG} ${DEFINE_FLAGS} ${K_FLAGS}
              -I "${VPIPE_METAL_KERNEL_DIR}"
              -I "${VPIPE_METAL_KERNEL_DIR}/vendored"
              -c "${SRC}" -o "${AIR}"
      DEPENDS "${SRC}" ${DEPENDS_ABS} "${VPIPE_METAL_TOOLCHAIN_STAMP}"
      COMMENT "Compiling Metal kernel ${KERNEL_NAME}.metal -> .air"
      VERBATIM
    )

    add_custom_command(
      OUTPUT ${LIB}
      COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metallib "${AIR}" -o "${LIB}"
      DEPENDS ${AIR}
      COMMENT "Linking metallib ${KERNEL_NAME}.metallib"
      VERBATIM
    )

    add_custom_command(
      OUTPUT ${EMBED_CC}
      COMMAND ${CMAKE_COMMAND}
              -D KERNEL_NAME=${KERNEL_NAME}
              -D INPUT=${LIB}
              -D OUTPUT=${EMBED_CC}
              -P ${_VPIPE_METAL_KERNEL_DIR}/embed-metallib.cmake
      DEPENDS ${LIB}
              ${_VPIPE_METAL_KERNEL_DIR}/embed-metallib.cmake
      COMMENT "Embedding ${KERNEL_NAME}.metallib into vpipe TU"
      VERBATIM
    )
  endif()

  # Compile the embed TU in *this* directory so the custom_command
  # rules above bind to a real target's Makefile. The OBJECT library
  # has no link products of its own; root CMakeLists pulls the .o
  # into libvpipe via target_link_libraries. PIE is required because
  # libvpipe is a SHARED library.
  set(OBJ_TARGET vpipe_mc_kernel_${KERNEL_NAME})
  add_library(${OBJ_TARGET} OBJECT ${EMBED_CC})
  set_target_properties(${OBJ_TARGET} PROPERTIES
      POSITION_INDEPENDENT_CODE ON)

  # Bubble the object target up to the parent (apple-silicon) so it
  # can chain to the root scope. Collected into a single list that
  # the root CMakeLists links into vpipe.
  list(APPEND VPIPE_MC_KERNEL_OBJS ${OBJ_TARGET})
  set(VPIPE_MC_KERNEL_OBJS ${VPIPE_MC_KERNEL_OBJS} PARENT_SCOPE)
endfunction()

# vpipe-metal-kernel.cmake -- helper to declare a Metal compute
# kernel. Compiles `.metal` -> `.air` -> `.metallib` via xcrun, then
# emits a generated `.cc` that registers the metallib bytes with
# MetalCompute's process-wide embedded registry at static-init time.
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
if(NOT XCRUN_EXECUTABLE)
  message(FATAL_ERROR
      "xcrun not found in PATH; required to compile vpipe Metal kernels")
endif()

# Resolve the directory this file lives in so we can hand off to
# embed-metallib.cmake. CMAKE_CURRENT_FUNCTION_LIST_DIR is set per
# function invocation, so we must capture it at include time.
set(_VPIPE_METAL_KERNEL_DIR "${CMAKE_CURRENT_LIST_DIR}")

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

  add_custom_command(
    OUTPUT ${AIR}
    COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metal
            -gline-tables-only -frecord-sources
            -Wall -Wextra -fno-fast-math ${STD_FLAG} ${DEFINE_FLAGS} ${K_FLAGS}
            -I "${VPIPE_METAL_KERNEL_DIR}"
            -I "${VPIPE_METAL_KERNEL_DIR}/vendored"
            -c "${SRC}" -o "${AIR}"
    DEPENDS "${SRC}" ${DEPENDS_ABS}
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

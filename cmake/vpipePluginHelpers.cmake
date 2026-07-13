# CMake helpers for building vpipe plugins against an installed vpipe SDK
# (pulled in by find_package(vpipe)). See docs/PLUGINS.md.

# vpipe_add_plugin(<name> SOURCES <a.cc> [b.cc ...] [METAL] [COREML])
#
# Build a dlopen-loadable plugin: a MODULE .dylib linked against
# vpipe::vpipe (which supplies the SDK include dir + the shared libvpipe to
# resolve against). C++20 is required (coroutine stages). A two-level-
# namespace MODULE already errors on undefined symbols at link time, so a
# missing libvpipe symbol fails the plugin link rather than dlopen.
#
# Base plugins (stages / models / metal) need NO framework flags -- every
# Metal/CoreML call lives inside libvpipe and the SDK headers only forward-
# declare the framework types. Pass METAL / COREML only if the plugin uses
# an escape hatch (mtl_buffer(), SharedBuffer::wrap, CML::Model* model()).
function(vpipe_add_plugin name)
  cmake_parse_arguments(P "METAL;COREML" "" "SOURCES" ${ARGN})
  if(NOT P_SOURCES)
    message(FATAL_ERROR "vpipe_add_plugin(${name}): SOURCES is required")
  endif()
  add_library(${name} MODULE ${P_SOURCES})
  set_target_properties(${name} PROPERTIES PREFIX "" MACOSX_RPATH ON)
  target_link_libraries(${name} PRIVATE vpipe::vpipe)
  target_compile_features(${name} PRIVATE cxx_std_20)
  if(APPLE AND P_METAL)
    target_link_libraries(${name} PRIVATE
      "-framework Metal" "-framework Foundation")
  endif()
  if(APPLE AND P_COREML)
    target_link_libraries(${name} PRIVATE
      "-framework CoreML" "-framework Foundation")
  endif()
endfunction()

# vpipe_add_metal_library(<objlib> <name> SRC <file.metal> [DEFINES d1 ...])
#
# Compile a self-contained .metal offline to a .metallib and embed its
# bytes into an OBJECT library that a plugin links. The object exports two
# ordinary symbols the plugin hands to register_metal_library at load:
#   extern "C" const unsigned char <name>_metallib[];
#   extern "C" const unsigned long <name>_metallib_len;
# so the plugin does, in its vpipe_plugin_register:
#   ctx->register_metal_library("<name>", <name>_metallib, <name>_metallib_len);
#
# `name` is the runtime library name used with load_library(). DEFINES are
# passed as -D to the metal compiler (e.g. VPIPE_ELT=bfloat for a bf16
# twin). Requires the Xcode metal toolchain (xcrun).
function(vpipe_add_metal_library objlib name)
  cmake_parse_arguments(K "" "SRC" "DEFINES" ${ARGN})
  if(NOT K_SRC)
    message(FATAL_ERROR "vpipe_add_metal_library(${objlib}): SRC is required")
  endif()
  set(_defs "")
  foreach(d IN LISTS K_DEFINES)
    list(APPEND _defs "-D${d}")
  endforeach()
  set(_air "${CMAKE_CURRENT_BINARY_DIR}/${name}.air")
  set(_lib "${CMAKE_CURRENT_BINARY_DIR}/${name}.metallib")
  set(_cc  "${CMAKE_CURRENT_BINARY_DIR}/${name}_metallib.cc")
  get_filename_component(_src_abs "${K_SRC}" ABSOLUTE)
  add_custom_command(
    OUTPUT "${_cc}"
    COMMAND xcrun -sdk macosx metal ${_defs} -c "${_src_abs}" -o "${_air}"
    COMMAND xcrun -sdk macosx metallib "${_air}" -o "${_lib}"
    COMMAND ${CMAKE_COMMAND}
            -DINPUT=${_lib} -DOUTPUT=${_cc} -DNAME=${name}
            -P "${CMAKE_CURRENT_LIST_DIR}/vpipe-embed-metallib.cmake"
    DEPENDS "${_src_abs}"
            "${CMAKE_CURRENT_LIST_DIR}/vpipe-embed-metallib.cmake"
    COMMENT "vpipe_add_metal_library: ${name}.metal -> embedded metallib"
    VERBATIM)
  add_library(${objlib} OBJECT "${_cc}")
  set_target_properties(${objlib} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()

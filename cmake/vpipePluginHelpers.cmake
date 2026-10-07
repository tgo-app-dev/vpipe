# CMake helpers for building vpipe plugins against an installed vpipe SDK
# (pulled in by find_package(vpipe)). See docs/PLUGINS.md.

# Where THIS file lives, captured at include time.
#
# Not CMAKE_CURRENT_LIST_DIR inside the function bodies below: inside a
# CMake *function* that variable is evaluated when the function RUNS, so
# it names the caller's directory -- the plugin's source tree -- and the
# helper then looks there for a script that ships beside this file. That
# is a build error naming a path in the plugin's own tree, which reads
# like the plugin is missing something it never had.
set(VPIPE_PLUGIN_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")

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
  # No RPATH on the plugin. Linking vpipe::vpipe otherwise makes CMake
  # bake the BUILDING machine's libvpipe directory in as an absolute
  # LC_RPATH, which is the one thing that would stop an otherwise
  # portable .so from loading elsewhere -- and it buys nothing, because a
  # plugin is only ever dlopened INTO a process that has already loaded
  # libvpipe. dyld resolves the plugin's `@rpath/libvpipe.0.dylib`
  # against that image by install name; the host binary found it through
  # its own relocatable `@loader_path/../lib`. VERIFIED by deleting the
  # RPATH from a built plugin with install_name_tool: it still loads and
  # still registers its models.
  #
  # BUILD_WITH_INSTALL_RPATH makes the build-tree artifact carry the
  # install RPATH (empty) rather than the dependency-derived one, so the
  # .so is portable where it is built, not only once installed.
  #
  # Scoped to the MODULE deliberately. A plugin's own test executables
  # link libvpipe and run standalone, so they DO need an RPATH to find
  # it; stripping theirs would break them at exec.
  set_target_properties(${name} PROPERTIES
      BUILD_WITH_INSTALL_RPATH TRUE
      INSTALL_RPATH "")
  target_link_libraries(${name} PRIVATE vpipe::vpipe)
  # The toolkit (generative-models/shared) is compiled INTO the plugin, not
  # called in libvpipe: see the vpipe_toolkit target in vpipe's own
  # CMakeLists. Present on Apple-silicon builds of the SDK.
  if(TARGET vpipe::toolkit)
    target_link_libraries(${name} PRIVATE vpipe::toolkit)
  endif()
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
# twin).
#
# TWO MODES, chosen by PROBING, exactly as the host chooses for its own
# kernels. The offline Metal shader compiler (the `metal` and `metallib`
# xcrun subcommands) ships with Xcode's Metal Toolchain, which is a
# separately downloadable component -- so a Command-Line-Tools-only box,
# and a full-Xcode box that has not fetched the component, both lack it.
# When it is there, the kernel is compiled to a .metallib and its bytes
# are embedded. When it is not, the include-flattened MSL SOURCE is
# embedded instead and compiled at load through newLibraryWithSource:.
#
# THE PLUGIN'S CALL SITE IS THE SAME EITHER WAY. Both modes define the
# two symbols below, and in source mode the length is zero and the
# registration has already happened from a static initialiser --
# register_metal_library takes that pair as "already provided" rather
# than as an empty library. A plugin author should not have to know
# which toolchain the build machine had, which is the whole reason the
# host probes for its own kernels and the reason this now does too.
#
#   extern "C" const unsigned char <name>_metallib[];
#   extern "C" const unsigned long <name>_metallib_len;
#
# Force either mode with -DVPIPE_METAL_RUNTIME_COMPILE=ON/OFF.
function(vpipe_add_metal_library objlib name)
  cmake_parse_arguments(K "" "SRC" "DEFINES" ${ARGN})
  if(NOT K_SRC)
    message(FATAL_ERROR "vpipe_add_metal_library(${objlib}): SRC is required")
  endif()
  get_filename_component(_src_abs "${K_SRC}" ABSOLUTE)
  set(_cc  "${CMAKE_CURRENT_BINARY_DIR}/${name}_metallib.cc")

  # Probe once per project, and cache the verdict so every kernel in a
  # plugin agrees and the reason is printed once.
  if(NOT DEFINED VPIPE_METAL_RUNTIME_COMPILE)
    find_program(VPIPE_XCRUN_EXECUTABLE xcrun)
    set(_vp_have_metal FALSE)
    if(VPIPE_XCRUN_EXECUTABLE)
      execute_process(COMMAND ${VPIPE_XCRUN_EXECUTABLE} -sdk macosx -f metal
          RESULT_VARIABLE _vp_m_rc OUTPUT_QUIET ERROR_QUIET)
      execute_process(COMMAND ${VPIPE_XCRUN_EXECUTABLE} -sdk macosx -f metallib
          RESULT_VARIABLE _vp_ml_rc OUTPUT_QUIET ERROR_QUIET)
      if(_vp_m_rc EQUAL 0 AND _vp_ml_rc EQUAL 0)
        set(_vp_have_metal TRUE)
      endif()
    endif()
    if(_vp_have_metal)
      set(VPIPE_METAL_RUNTIME_COMPILE OFF CACHE BOOL
          "Embed .metal SOURCE and compile at load (no Metal toolchain)")
    else()
      set(VPIPE_METAL_RUNTIME_COMPILE ON CACHE BOOL
          "Embed .metal SOURCE and compile at load (no Metal toolchain)")
      message(STATUS
          "vpipe plugin kernels: RUNTIME-COMPILE mode -- `xcrun -sdk macosx "
          "-f metal` found no offline shader compiler, so .metal SOURCE is "
          "embedded and compiled at load. The Metal Toolchain is a separate "
          "Xcode component: `xcodebuild -downloadComponent MetalToolchain` "
          "(and `sudo xcode-select -s /Applications/Xcode.app` if this box "
          "is on the Command Line Tools) to build metallibs instead")
    endif()
  endif()

  if(VPIPE_METAL_RUNTIME_COMPILE)
    set(_embed "${VPIPE_PLUGIN_CMAKE_DIR}/embed-metal-source.cmake")
    if(NOT EXISTS "${_embed}")
      message(FATAL_ERROR
          "vpipe_add_metal_library(${objlib}): no Metal toolchain, and the "
          "installed vpipe does not ship embed-metal-source.cmake, so there "
          "is no fallback. Install the Metal Toolchain "
          "(`xcodebuild -downloadComponent MetalToolchain`), or use a vpipe "
          "install new enough to carry the script")
    endif()
    # '|'-joined, not ';': the lists have to survive add_custom_command's
    # argument splitting as single arguments.
    set(_def_arg "")
    string(JOIN "|" _def_arg ${K_DEFINES})
    get_filename_component(_src_dir "${_src_abs}" DIRECTORY)
    add_custom_command(
      OUTPUT "${_cc}"
      COMMAND ${CMAKE_COMMAND}
              -D KERNEL_NAME=${name}
              -D SRC=${_src_abs}
              -D OUTPUT=${_cc}
              -D INCLUDE_DIRS=${_src_dir}
              -D DEFINES=${_def_arg}
              -D LANG=0
              -D EMIT_METALLIB_STUB=1
              -P "${_embed}"
      DEPENDS "${_src_abs}" "${_embed}"
      COMMENT "vpipe_add_metal_library: ${name}.metal SOURCE (runtime-compile)"
      VERBATIM)
  else()
    set(_air "${CMAKE_CURRENT_BINARY_DIR}/${name}.air")
    set(_lib "${CMAKE_CURRENT_BINARY_DIR}/${name}.metallib")
    set(_defs "")
    foreach(d IN LISTS K_DEFINES)
      list(APPEND _defs "-D${d}")
    endforeach()
    # The deployment target, explicitly: CMake does not export
    # MACOSX_DEPLOYMENT_TARGET to a custom command, so the metal
    # compiler would otherwise target the BUILD MACHINE's OS. Both the
    # triple and the AIR bytecode version move with it, so a plugin
    # built on a newer OS would not load on the host's minimum at all.
    set(_vp_min_os "")
    if(CMAKE_OSX_DEPLOYMENT_TARGET)
      set(_vp_min_os -mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET})
    endif()
    add_custom_command(
      OUTPUT "${_cc}"
      COMMAND ${VPIPE_XCRUN_EXECUTABLE} -sdk macosx metal
              ${_vp_min_os} ${_defs}
              -c "${_src_abs}" -o "${_air}"
      COMMAND ${VPIPE_XCRUN_EXECUTABLE} -sdk macosx metallib
              "${_air}" -o "${_lib}"
      COMMAND ${CMAKE_COMMAND}
              -DINPUT=${_lib} -DOUTPUT=${_cc} -DNAME=${name}
              -P "${VPIPE_PLUGIN_CMAKE_DIR}/vpipe-embed-metallib.cmake"
      DEPENDS "${_src_abs}"
              "${VPIPE_PLUGIN_CMAKE_DIR}/vpipe-embed-metallib.cmake"
      COMMENT "vpipe_add_metal_library: ${name}.metal -> embedded metallib"
      VERBATIM)
  endif()
  add_library(${objlib} OBJECT "${_cc}")
  set_target_properties(${objlib} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()

# The plugin SDK, checked the way a plugin author meets it: installed to a
# scratch prefix, every installed header compiled ON ITS OWN, and the SDK
# example plugin built against that prefix alone.
#
# What it catches is the failure the source tree cannot show: a header
# that includes one that is not installed. Inside the tree every include
# resolves, so the gap surfaces only when someone outside builds a plugin
# -- which is how a streaming header came to need two FP8 headers the
# install did not ship.
#
# Run after a full build (cmake --install needs what it installs):
#   cmake --build <build> --target vpipe_sdk_check
#
# Then the PLUGIN ABI itself is compared against the committed snapshot
# for this ABI version (abi/plugin-abi-<N>.txt, see cmake/abi-snapshot.py):
# a removed or changed layout, virtual or exported symbol fails the check.
# With MODE=write the snapshot is regenerated instead (the
# vpipe_abi_snapshot target) -- which is where an ABI version is decided.
#
# Inputs: BUILD_DIR, SOURCE_DIR, CXX, PYTHON, LIB; optional MODE=write.

if(NOT BUILD_DIR OR NOT SOURCE_DIR OR NOT CXX)
  message(FATAL_ERROR "check-sdk: BUILD_DIR, SOURCE_DIR and CXX are required")
endif()

set(work   "${BUILD_DIR}/sdk-check")
set(prefix "${work}/prefix")
file(REMOVE_RECURSE "${work}")
file(MAKE_DIRECTORY "${work}/tu")

execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${BUILD_DIR}" --prefix "${prefix}"
  RESULT_VARIABLE rc OUTPUT_QUIET ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "check-sdk: install failed:\n${err}")
endif()

file(GLOB_RECURSE headers RELATIVE "${prefix}/include"
     "${prefix}/include/*.h")
list(SORT headers)
set(failed "")
set(n 0)
foreach(h IN LISTS headers)
  string(MAKE_C_IDENTIFIER "${h}" id)
  set(tu "${work}/tu/${id}.cc")
  file(WRITE "${tu}" "#include \"${h}\"\n")
  execute_process(
    COMMAND "${CXX}" -std=c++20 -fsyntax-only "-I${prefix}/include" "${tu}"
    RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE out)
  math(EXPR n "${n} + 1")
  if(NOT rc EQUAL 0)
    list(APPEND failed "${h}")
    string(REGEX MATCH "[^\n]*error:[^\n]*" first "${out}")
    message(STATUS "check-sdk: ${h}: ${first}")
  endif()
endforeach()

# The toolkit's sources, compiled against the installed headers ALONE: a
# plugin links them in, so anything they include has to be in the SDK.
if(EXISTS "${BUILD_DIR}/vpipe-toolkit-sources.txt")
  # And the package must export it under the name vpipe_add_plugin looks
  # for; under any other the helper links nothing, silently, and a plugin
  # that calls the toolkit fails only at its own link.
  file(GLOB targets_file "${prefix}/lib/cmake/vpipe/vpipeTargets.cmake")
  file(STRINGS "${targets_file}" toolkit_target
       REGEX "add_library\\(vpipe::toolkit ")
  if(NOT toolkit_target)
    list(APPEND failed "vpipe::toolkit (not exported under that name)")
    message(STATUS "check-sdk: the installed package does not define "
                   "vpipe::toolkit, which vpipe_add_plugin links")
  endif()
  file(READ "${BUILD_DIR}/vpipe-toolkit-sources.txt" toolkit)
  foreach(src IN LISTS toolkit)
    execute_process(
      COMMAND "${CXX}" -std=c++20 -fsyntax-only "-I${prefix}/include" "${src}"
      RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE out)
    math(EXPR n "${n} + 1")
    if(NOT rc EQUAL 0)
      list(APPEND failed "${src}")
      string(REGEX MATCH "[^\n]*error:[^\n]*" first "${out}")
      message(STATUS "check-sdk: toolkit ${src}: ${first}")
    endif()
  endforeach()
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -S "${SOURCE_DIR}/examples/plugin-sdk"
          -B "${work}/example" "-DCMAKE_PREFIX_PATH=${prefix}"
          "-DCMAKE_CXX_COMPILER=${CXX}"
  RESULT_VARIABLE rc OUTPUT_QUIET ERROR_VARIABLE err)
if(rc EQUAL 0)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${work}/example"
    RESULT_VARIABLE rc OUTPUT_VARIABLE err ERROR_VARIABLE err)
endif()
if(NOT rc EQUAL 0)
  list(APPEND failed "examples/plugin-sdk")
  message(STATUS "check-sdk: the SDK example did not build:\n${err}")
endif()

# ---- the ABI snapshot --------------------------------------------------------
file(STRINGS "${SOURCE_DIR}/plugin/plugin-abi.h" abi_line
     REGEX "^#define VPIPE_PLUGIN_ABI_VERSION")
string(REGEX MATCH "[0-9]+" abi "${abi_line}")
set(snapshot "${SOURCE_DIR}/abi/plugin-abi-${abi}.txt")
if(PYTHON AND LIB)
  if(MODE STREQUAL "write")
    execute_process(
      COMMAND "${PYTHON}" "${SOURCE_DIR}/cmake/abi-snapshot.py" write
              --prefix "${prefix}" --lib "${LIB}" --cxx "${CXX}"
              --out "${snapshot}"
      RESULT_VARIABLE rc)
    if(NOT rc EQUAL 0)
      message(FATAL_ERROR "check-sdk: writing the ABI snapshot failed")
    endif()
  elseif(NOT EXISTS "${snapshot}")
    list(APPEND failed "abi/plugin-abi-${abi}.txt (missing)")
    message(STATUS "check-sdk: no ABI snapshot for ABI ${abi}; create it "
                   "with the vpipe_abi_snapshot target")
  else()
    execute_process(
      COMMAND "${PYTHON}" "${SOURCE_DIR}/cmake/abi-snapshot.py" check
              --prefix "${prefix}" --lib "${LIB}" --cxx "${CXX}"
              --snapshot "${snapshot}"
      RESULT_VARIABLE rc)
    if(NOT rc EQUAL 0)
      list(APPEND failed "the plugin ABI (abi/plugin-abi-${abi}.txt)")
    endif()
  endif()
endif()

list(LENGTH failed nfail)
if(nfail GREATER 0)
  message(FATAL_ERROR
          "check-sdk: ${nfail} of ${n} headers (+ the example) failed: "
          "${failed}")
endif()
message(STATUS "check-sdk: ${n} installed headers and toolkit sources "
               "compile against the install alone; the SDK example builds")

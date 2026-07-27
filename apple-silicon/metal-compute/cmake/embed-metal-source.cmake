# embed-metal-source.cmake -- `cmake -P` script for the RUNTIME-COMPILE
# build mode. Used when the offline Metal toolchain (metal/metallib) is
# unavailable at build time. Instead of compiling a .metallib, it:
#
#   1. Flattens the .metal kernel by inlining every local `#include "..."`
#      exactly once (emulating #pragma once), resolving each against the
#      same roots the AOT compiler uses. `#include <...>` (metal_stdlib,
#      MetalPerformancePrimitives, ...) are left for the runtime compiler.
#   2. Prepends the -D defines (e.g. VPIPE_ELT=bfloat) as `#define` lines,
#      since newLibraryWithSource: has no include-path or macro CLI.
#   3. Emits a generated .cc that registers the MSL SOURCE with
#      MetalCompute's embedded registry, so load_library(KERNEL_NAME)
#      compiles it via newLibraryWithSource: on first use.
#
# Required -D inputs:
#   KERNEL_NAME  -- symbolic name load_library() resolves
#   SRC          -- absolute path to the root .metal source
#   OUTPUT       -- absolute path of the .cc to write
#   INCLUDE_DIRS -- '|'-joined list of -I roots for quoted includes
#   DEFINES      -- '|'-joined list of KEY=VAL / KEY defines (may be empty)
#   LANG         -- language-version code (0 default, 40 = metal4.0)
#
# Lists arrive '|'-joined (not ';'-joined) so they survive as single args
# through add_custom_command's argument splitting.

if(NOT DEFINED KERNEL_NAME)
  message(FATAL_ERROR "embed-metal-source.cmake: -D KERNEL_NAME=... required")
endif()
if(NOT DEFINED SRC)
  message(FATAL_ERROR "embed-metal-source.cmake: -D SRC=... required")
endif()
if(NOT DEFINED OUTPUT)
  message(FATAL_ERROR "embed-metal-source.cmake: -D OUTPUT=... required")
endif()
if(NOT DEFINED INCLUDE_DIRS)
  set(INCLUDE_DIRS "")
endif()
if(NOT DEFINED DEFINES)
  set(DEFINES "")
endif()
if(NOT DEFINED LANG)
  set(LANG 0)
endif()

string(REPLACE "|" ";" INCLUDE_DIRS "${INCLUDE_DIRS}")
string(REPLACE "|" ";" DEFINES      "${DEFINES}")

# Sentinels to protect characters that CMake list iteration mangles while
# we split content into lines on newlines:
#   ';'      -- the list delimiter
#   '[' ']'  -- unbalanced brackets make foreach(IN LISTS) merge elements
#   '\'      -- an escape char; a line-continuation '\' before the ';' we
#               inject becomes an escaped '\;' and merges the two lines
# MSL uses all of these heavily (macros with '\'-continuations, [[attrs]],
# array subscripts). Restored per-line before use.
string(ASCII 1 _SEMI)
string(ASCII 2 _LBRK)
string(ASCII 3 _RBRK)
string(ASCII 4 _BSL)

# GLOBAL properties persist across the recursive function calls below.
set_property(GLOBAL PROPERTY _VP_FLAT "")

function(_vp_inline_file _path)
  get_filename_component(_abs "${_path}" ABSOLUTE)
  string(MAKE_C_IDENTIFIER "vp_seen_${_abs}" _key)
  get_property(_seen GLOBAL PROPERTY ${_key})
  if(_seen)
    return()                          # include-once (emulate #pragma once)
  endif()
  set_property(GLOBAL PROPERTY ${_key} 1)

  if(NOT EXISTS "${_abs}")
    message(FATAL_ERROR
        "embed-metal-source(${KERNEL_NAME}): cannot read '${_abs}'")
  endif()
  file(READ "${_abs}" _content)
  string(REPLACE "\\" "${_BSL}"  _content "${_content}")  # protect '\'
  string(REPLACE ";" "${_SEMI}" _content "${_content}")   # protect ';'
  string(REPLACE "[" "${_LBRK}" _content "${_content}")   # protect '['
  string(REPLACE "]" "${_RBRK}" _content "${_content}")   # protect ']'
  string(REPLACE "\n" ";" _lines "${_content}")           # lines -> list

  get_filename_component(_dir "${_abs}" DIRECTORY)
  foreach(_line IN LISTS _lines)
    string(REPLACE "${_SEMI}" ";" _raw "${_line}")
    string(REPLACE "${_LBRK}" "[" _raw "${_raw}")
    string(REPLACE "${_RBRK}" "]" _raw "${_raw}")
    string(REPLACE "${_BSL}" "\\" _raw "${_raw}")
    if(_raw MATCHES "^[ \t]*#[ \t]*include[ \t]*\"([^\"]+)\"")
      set(_inc "${CMAKE_MATCH_1}")
      # Resolve like a compiler's quoted include: including file's dir
      # first, then each -I root in order.
      set(_hit "")
      if(EXISTS "${_dir}/${_inc}")
        set(_hit "${_dir}/${_inc}")
      else()
        foreach(_root IN LISTS INCLUDE_DIRS)
          if(EXISTS "${_root}/${_inc}")
            set(_hit "${_root}/${_inc}")
            break()
          endif()
        endforeach()
      endif()
      if(_hit)
        _vp_inline_file("${_hit}")
      else()
        # Keep unresolved quoted includes verbatim so the runtime
        # compiler reports them clearly instead of silently dropping.
        set_property(GLOBAL APPEND_STRING PROPERTY _VP_FLAT "${_raw}\n")
      endif()
    else()
      set_property(GLOBAL APPEND_STRING PROPERTY _VP_FLAT "${_raw}\n")
    endif()
  endforeach()
endfunction()

_vp_inline_file("${SRC}")
get_property(_flat GLOBAL PROPERTY _VP_FLAT)

# Prepend -D defines as #define lines (KEY=VAL -> `#define KEY VAL`).
set(_defs "// runtime-compile: injected -D defines\n")
foreach(_d IN LISTS DEFINES)
  if(_d STREQUAL "")
    # skip empty
  elseif(_d MATCHES "^([^=]+)=(.*)$")
    set(_defs "${_defs}#define ${CMAKE_MATCH_1} ${CMAKE_MATCH_2}\n")
  else()
    set(_defs "${_defs}#define ${_d} 1\n")
  endif()
endforeach()
set(_flat "${_defs}${_flat}")

# Write the flattened .metal beside the .cc (a debugging aid + a stable
# on-disk artifact), then embed its bytes as a NUL-terminated char array.
set(_flat_metal "${OUTPUT}.metal")
file(WRITE "${_flat_metal}" "${_flat}")
file(READ  "${_flat_metal}" _bytes HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," _csv "${_bytes}")

file(WRITE "${OUTPUT}"
"// Auto-generated by embed-metal-source.cmake. Do not edit.
//
// RUNTIME-COMPILE mode: embeds the (include-flattened) MSL source for
// kernel \"${KERNEL_NAME}\" and registers it so MetalCompute::load_library
// compiles it via newLibraryWithSource: on first use. Emitted when the
// offline Metal toolchain (metal/metallib) is unavailable at build time.

#include <cstddef>

namespace vpipe::metal_compute::_embedded {
void register_embedded_metal_source(const char* name,
                                    const char* src, std::size_t len,
                                    int lang_version);
}

namespace {

// unsigned char (not char): MSL sources can contain non-ASCII bytes
// (e.g. UTF-8 in comments) that would fail the narrowing conversion to
// signed char in a braced initializer. NUL-terminated so it binds
// directly as a C string; length excludes the terminator.
alignas(8) const unsigned char kSrc_${KERNEL_NAME}[] = {
${_csv}
0x00
};

struct RegSrc_${KERNEL_NAME} {
  RegSrc_${KERNEL_NAME}() {
    ::vpipe::metal_compute::_embedded::register_embedded_metal_source(
        \"${KERNEL_NAME}\",
        reinterpret_cast<const char*>(kSrc_${KERNEL_NAME}),
        sizeof(kSrc_${KERNEL_NAME}) - 1,
        ${LANG});
  }
};

const RegSrc_${KERNEL_NAME} g_regsrc_${KERNEL_NAME};

}  // namespace
")

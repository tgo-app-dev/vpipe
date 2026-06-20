// CoreMLDefines.hpp
//
// Visibility and inline macros for the CoreML C++ wrapper. Mirrors
// metal-cpp's NSDefines.hpp / MTLDefines.hpp.
//
// All other CoreML*.hpp headers in this directory pick up _CML_INLINE
// and _CML_EXPORT from here. Keep this file minimal -- adding macros
// here forces every translation unit that touches CoreML to see them.

#pragma once

#ifdef METALCPP_SYMBOL_VISIBILITY_HIDDEN
#define _CML_EXPORT __attribute__((visibility("hidden")))
#else
#define _CML_EXPORT __attribute__((visibility("default")))
#endif

#define _CML_EXTERN     extern "C" _CML_EXPORT
#define _CML_INLINE     inline __attribute__((always_inline))
#define _CML_WEAK_IMPORT __attribute__((weak_import))

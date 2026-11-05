// ane-int8-expected.h -- whether a rotated int8 ANE module must build on
// the box a test runs on.
//
// `ane_int8` (AneFeedForward::Options::int8) needs CoreML's int8 arrays,
// which arrive with macOS 26; before that the tier builds its fp16 module
// and says so, which is a correct outcome there and a broken one on 26.
// So a family test asserts the int8 arm ENGAGED where this is true and
// skips the arm where it is not -- an arm that silently ran fp16 would
// otherwise pass as itself. VPIPE_ANE_INT8 (0 | 1) overrides every
// module's choice, so with it set the arms are not what they say.
#pragma once

#include <cstdlib>

inline bool
ane_int8_expected()
{
  if (std::getenv("VPIPE_ANE_INT8") != nullptr) { return false; }
  if (__builtin_available(macOS 26.0, *)) { return true; }
  return false;
}

// The other half: under VPIPE_ANE_INT8 the fp16 arms are not fp16.
inline bool
ane_int8_env_free()
{
  return std::getenv("VPIPE_ANE_INT8") == nullptr;
}

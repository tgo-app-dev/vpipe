#ifndef VPIPE_GENERATIVE_MODELS_SHARED_ACCEL_SETTINGS_H
#define VPIPE_GENERATIVE_MODELS_SHARED_ACCEL_SETTINGS_H

// The acceleration settings a generating stage hands EVERY model family,
// as an open bag rather than as struct fields.
//
// ---------------------------------------------------------------------
// WHY A BAG, when typed fields read better
//
// Because typed fields put the host's vocabulary into the plugin ABI.
// `VideoGenRequest` carried `sage::Config sage; bool i8_gemm;`, and
// adding `sol::Config sol` beside them changed the struct's layout --
// which means every out-of-tree family, including ones that implement
// none of these, has to be rebuilt before it can be loaded again. The
// cost is paid by plugins that do not use the feature, for a decision
// they were not part of, and it is paid EVERY time a tier is added.
//
// That is the wrong trade for a channel whose whole purpose is optional
// hints. So the structs carry one pointer that never changes shape, and
// the vocabulary lives HERE, in a header that ships with the host and is
// read by code compiled into the caller.
//
// ---------------------------------------------------------------------
// THE CONTRACT
//
//   * A KEY IS ADDED, NEVER RENAMED AND NEVER REPURPOSED. An old plugin
//     reading `sol_tau` must keep getting what `sol_tau` has always
//     meant. If a setting's meaning changes, it gets a new key and the
//     old one keeps its old meaning for as long as anything might ask.
//   * ABSENT MEANS THE SHIPPED DEFAULT, and a reader supplies it. A
//     family therefore behaves identically against a host that predates
//     a key and one that has it turned off.
//   * A FAMILY IGNORES WHAT IT DOES NOT KNOW. The host never requires
//     any of this to be implemented; every tier here is an optional
//     accelerator, and a family that implements none is correct.
//   * ...BUT IT SHOULD SAY WHICH, because a knob that silently did
//     nothing and one that did something are indistinguishable in a
//     wall clock. `tiers_on()` is here for that log line.
//   * THE VALUES ARE SETTLED, not raw. The stage validates and corrects
//     before filling the bag -- an out-of-range `sol_key_block` is
//     already the value it fell back to -- so a family can use what it
//     reads without re-checking, and two families cannot disagree about
//     what the graph asked for.
//
// ---------------------------------------------------------------------
// HEADER-ONLY ON PURPOSE
//
// Every reader below is `inline` and compiles into the CALLER. A plugin
// built against an older host carries that host's readers and cannot see
// a key it was never told about; one built against a newer host sees the
// new key without the host having changed shape. The only libvpipe
// symbols involved are FlexData's own, which is the type this boundary
// already speaks -- `model_config` is the same idea for settings that
// belong to ONE family, where this is the same idea for settings that
// belong to all of them.
//
// This makes the acceleration VOCABULARY soft. It does not make the
// whole plugin ABI soft: FlexData's own layout is still shared, as are
// the config structs of any tier a family chooses to drive. The
// difference is the rate -- FlexData changes rarely and deliberately,
// where this list grows whenever someone has an idea.

#include "common/flex-data.h"
#include "generative-models/accel-keys.h"

#include <string>
#include <string_view>
#include <vector>

// TOOLKIT: compiled into each plugin (libvpipe_toolkit.a), not exported
// by libvpipe -- see the vpipe_toolkit target and plugin/plugin-abi.h.

namespace vpipe {
namespace genai {
namespace accel {

// ---- the keys -------------------------------------------------------
//
// In generative-models/accel-keys.h, a STABLE header: the key names are
// the host's contract with every plugin in the support window, where the
// readers below are the toolkit's own and travel with each plugin.

// ---- reading --------------------------------------------------------
//
// A null bag reads as every default, which is what a family gets from a
// host too old to send one -- and is the same thing it gets from a graph
// that asked for nothing.

inline bool
has(const FlexData* bag, std::string_view key)
{
  if (bag == nullptr || !bag->is_object()) { return false; }
  return bag->as_object().contains(key);
}

inline bool
flag(const FlexData* bag, std::string_view key, bool dflt = false)
{
  if (!has(bag, key)) { return dflt; }
  return bag->as_object().at(key).as_bool(dflt);
}

inline double
real(const FlexData* bag, std::string_view key, double dflt)
{
  if (!has(bag, key)) { return dflt; }
  return bag->as_object().at(key).as_real(dflt);
}

inline long long
integer(const FlexData* bag, std::string_view key, long long dflt)
{
  if (!has(bag, key)) { return dflt; }
  return (long long)bag->as_object().at(key).as_int(dflt);
}

inline std::string
text(const FlexData* bag, std::string_view key, const std::string& dflt = {})
{
  if (!has(bag, key)) { return dflt; }
  return std::string(bag->as_object().at(key).as_string(dflt));
}

// ---- writing --------------------------------------------------------
//
// For the STAGE, and for a test that wants to drive a family without
// one. Kept beside the readers so the two spellings of a key cannot
// drift apart -- they are the same constant.

inline void
set_flag(FlexData* bag, std::string_view key, bool v)
{
  if (bag == nullptr || !bag->is_object()) { return; }
  bag->as_object().insert_or_assign(key, FlexData::make_bool(v));
}

inline void
set_real(FlexData* bag, std::string_view key, double v)
{
  if (bag == nullptr || !bag->is_object()) { return; }
  bag->as_object().insert_or_assign(key, FlexData::make_real(v));
}

inline void
set_integer(FlexData* bag, std::string_view key, long long v)
{
  if (bag == nullptr || !bag->is_object()) { return; }
  bag->as_object().insert_or_assign(key, FlexData::make_int((int64_t)v));
}

inline void
set_text(FlexData* bag, std::string_view key, const std::string& v)
{
  if (bag == nullptr || !bag->is_object()) { return; }
  bag->as_object().insert_or_assign(key, FlexData::make_string(v));
}

// ---- what the graph turned on ---------------------------------------
//
// The ENABLE flag of every tier this reader knows about that the graph
// set true. A family logs the ones it does not implement:
//
//   for (auto t : accel::tiers_on(req.accel)) {
//     if (t != accel::kSolAttn) { warn(... " is not implemented here"); }
//   }
//
// It is deliberately not a list of every key present. What a user needs
// told is that a TIER they asked for is not going to happen; that
// `sol_tau` went unread by a family with no Sol is not news.
//
// A plugin built against an older host lists the tiers that host knew,
// which is exactly right: it cannot implement one it has never heard of,
// and cannot report on one either.
inline std::vector<std::string_view>
tiers_on(const FlexData* bag)
{
  std::vector<std::string_view> out;
  for (std::string_view k :
       {kI8Gemm, kSageAttn, kSolAttn, kAneFfn, kMotionCache}) {
    if (flag(bag, k)) { out.push_back(k); }
  }
  return out;
}

}  // namespace accel
}  // namespace genai
}  // namespace vpipe

#endif  // VPIPE_GENERATIVE_MODELS_SHARED_ACCEL_SETTINGS_H

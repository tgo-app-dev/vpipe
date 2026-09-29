#ifndef VPIPE_SPEC_EXTRA_H
#define VPIPE_SPEC_EXTRA_H

// The growth seam of a SPEC struct: StageSpec, PortSpec, BufferSpec,
// CommandSpec, ConfigKey, ServiceReq and UiViewSpec. Theirs is a span
// of string pairs rather than the FlexData every other value struct at
// the plugin boundary ends in (common/flex-bag.h), because their
// instances live in STATIC storage -- constexpr tables, function-local
// statics, registry pointers -- and a FlexData member would cost them
// constant initialisation and a trivial destructor. A value that is not
// a string travels as its JSON text.

#include <span>
#include <string_view>

#include "common/vpipe-api.h"

VPIPE_API_BEGIN

namespace vpipe {

// ONE EXTRA FACT about a spec, as a name and a value.
//
// It exists so the spec structs can stop growing. StageSpec, PortSpec
// and ConfigKey are AGGREGATES A PLUGIN BUILDS, with static storage, and
// hands over as pointers and spans -- so a field added here moves a
// layout a plugin already compiled, and every plugin that registers a
// stage has to be rebuilt for metadata it does not use. All three of
// this tree's plugins register stages.
//
// So new spec metadata goes in a span of these instead. Both halves are
// string_view over STATIC storage, exactly like every other member
// beside them, which keeps the aggregates brace-initializable and free
// of a non-trivial destructor.
//
// Names are lower-case and the host's, added and never repurposed.
// Two are defined today:
//
//   "category"  on a StageSpec -- a category NAME that wins over the
//               `category` enum when set, so a plugin whose stage is a
//               kind this tree has no enumerator for can say so without
//               an enum change (which is an ABI change).
//   "since"     a version string, for a composer that wants to mark
//               what is new.
//   "choices"   on a ConfigKey -- the fixed set of values a String key
//               accepts, comma-separated (the suggest_db_type spelling).
//               An editor offers them as a dropdown instead of a text
//               box; it reaches the config schema as a JSON array.
//
//               A HINT, like suggest_db: the stage still validates its
//               own value, and a key whose set is open (an FFmpeg codec
//               name, a locale tag) declares nothing here. Where the set
//               is genuinely closed, this is the ONE place it is written
//               down for the UI -- but the stage's own accept/reject
//               code is still the authority, so the two must be kept in
//               step by whoever edits either.
struct SpecExtra {
  std::string_view key;
  std::string_view value;
};

// The names defined so far (see above for what each means).
namespace spec_key {
inline constexpr std::string_view kCategory = "category";
inline constexpr std::string_view kSince    = "since";
inline constexpr std::string_view kChoices  = "choices";
}  // namespace spec_key

// Look one up. Empty when absent, which is what every spec written
// before a key existed produces.
std::string_view spec_extra(std::span<const SpecExtra> extra,
                            std::string_view key) noexcept;

}  // namespace vpipe

VPIPE_API_END

#endif

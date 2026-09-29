#ifndef VPIPE_COMMON_ABI_STRUCT_H
#define VPIPE_COMMON_ABI_STRUCT_H

// A TYPED append to StageSpec, the one struct that keeps this.
//
// Everything else at the plugin boundary grows through a bag: a
// FlexData `extra` on value structs (common/flex-bag.h), a SpecExtra
// span on spec structs (pipeline/spec-extra.h). StageSpec has its
// SpecExtra too, but it is also where a TYPED member has had to be
// added -- `commands`, a span of CommandSpec -- and a string pair cannot
// carry one. So StageSpec starts with `std::uint32_t struct_size =
// sizeof(StageSpec)`; the plugin's compiler writes its own sizeof there,
// and a host newer than the plugin asks before reading a trailing field:
//
//   if (abi_has(spec, &StageSpec::commands)) { ... }
//
// A host must never read a field abi_has() denies -- that memory belongs
// to whatever the older plugin put after the struct. Growth is
// APPEND-ONLY. See docs/PLUGINS.md, "Versioning".

#include <cstddef>
#include <cstdint>

#include "common/vpipe-api.h"

VPIPE_API_BEGIN

namespace vpipe {

template <class S, class M>
bool
abi_has(const S& s, const M S::*member) noexcept
{
  const auto* base  = reinterpret_cast<const char*>(&s);
  const auto* field = reinterpret_cast<const char*>(&(s.*member));
  return static_cast<std::size_t>(s.struct_size) >=
         static_cast<std::size_t>(field - base) + sizeof(M);
}

}

VPIPE_API_END

#endif

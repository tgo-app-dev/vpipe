#ifndef VPIPE_FLEX_BAG_H
#define VPIPE_FLEX_BAG_H

// THE GROWTH SEAM of every value struct that crosses the plugin boundary.
//
// A struct a plugin allocates, returns, keeps in a std::vector or is
// handed by value has its layout compiled into the plugin, so a field
// appended to it by a newer host is a field the plugin's copy does not
// have -- a misread at best, a write past the end at worst. Those that
// carry parameters rather than bulk data therefore end in ONE
// `FlexData extra`, and that is the only place a field is ever added
// once the ABI that shipped the struct is out:
//
//   * a new field is a KEY. Keys are the host's, lower_snake, added and
//     never renamed or repurposed -- a name is a promise to every
//     binary already built against it;
//   * an absent key means the behaviour from before the key existed.
//     That is what every older binary produces, so it is the default
//     every reader must take;
//   * the bag is null until something writes to it, and a null bag
//     costs one pointer: no allocation, no copy;
//   * a bag the struct OWNS is `extra`, by value. One it only BORROWS --
//     owned by the caller and valid for the call, the way the host lends
//     a family its arguments -- is `const FlexData* borrowed_extra`. The
//     name says which, rather than a plural that differs by one letter;
//   * every key is a named constant in a stable header, so the ABI
//     snapshot lists it and a rename fails the build.
//
// Structs whose instances live in STATIC storage (the stage, port,
// config-key and view specs) cannot hold a FlexData without giving up
// constant initialisation; theirs is the string-pair SpecExtra
// (pipeline/spec-extra.h). Structs that carry bulk data -- a tensor, a
// buffer, a frame -- keep their fixed layout, and grow through the
// FlexData they already point at (`borrowed_extra`, `sideband`).
//
// These helpers read and write one. Reading a bag that is null, or not
// an object, gives the default; writing to a null bag makes it an
// object first.

#include "common/flex-data.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

#include "common/vpipe-api.h"

VPIPE_API_BEGIN

namespace vpipe {
namespace bag {

inline bool
has(const FlexData& b, std::string_view key)
{
  return b.is_object() && b.as_object().contains(key);
}

inline bool
flag(const FlexData& b, std::string_view key, bool dflt = false)
{
  return has(b, key) ? b.as_object().at(key).as_bool(dflt) : dflt;
}

inline std::int64_t
integer(const FlexData& b, std::string_view key, std::int64_t dflt = 0)
{
  return has(b, key) ? b.as_object().at(key).as_int(dflt) : dflt;
}

inline std::uint64_t
unsigned_integer(const FlexData& b, std::string_view key,
                 std::uint64_t dflt = 0)
{
  return has(b, key) ? b.as_object().at(key).as_uint(dflt) : dflt;
}

inline double
real(const FlexData& b, std::string_view key, double dflt = 0.0)
{
  return has(b, key) ? b.as_object().at(key).as_real(dflt) : dflt;
}

inline std::string
text(const FlexData& b, std::string_view key, std::string_view dflt = {})
{
  return std::string(has(b, key) ? b.as_object().at(key).as_string(dflt)
                                 : dflt);
}

// The value itself, for a key that is not a scalar; null when absent.
inline FlexData
value(const FlexData& b, std::string_view key)
{
  return has(b, key) ? b.as_object().at(key) : FlexData();
}

// False only when `b` already holds something that is not an object --
// which no host-defined bag ever does.
inline bool
set(FlexData& b, std::string_view key, FlexData v)
{
  if (b.is_null()) { b = FlexData::make_object(); }
  if (!b.is_object()) { return false; }
  b.as_object().insert_or_assign(key, std::move(v));
  return true;
}

inline bool
set_flag(FlexData& b, std::string_view key, bool v)
{
  return set(b, key, FlexData::make_bool(v));
}

inline bool
set_integer(FlexData& b, std::string_view key, std::int64_t v)
{
  return set(b, key, FlexData::make_int(v));
}

inline bool
set_real(FlexData& b, std::string_view key, double v)
{
  return set(b, key, FlexData::make_real(v));
}

inline bool
set_text(FlexData& b, std::string_view key, std::string_view v)
{
  return set(b, key, FlexData::make_string(v));
}

// The same readers over a bag LENT by pointer (`borrowed_extra`, `accel`),
// where null means the host sent none.
inline bool
has(const FlexData* b, std::string_view key)
{
  return b != nullptr && has(*b, key);
}

inline bool
flag(const FlexData* b, std::string_view key, bool dflt = false)
{
  return b != nullptr ? flag(*b, key, dflt) : dflt;
}

inline std::int64_t
integer(const FlexData* b, std::string_view key, std::int64_t dflt = 0)
{
  return b != nullptr ? integer(*b, key, dflt) : dflt;
}

inline double
real(const FlexData* b, std::string_view key, double dflt = 0.0)
{
  return b != nullptr ? real(*b, key, dflt) : dflt;
}

inline std::string
text(const FlexData* b, std::string_view key, std::string_view dflt = {})
{
  return b != nullptr ? text(*b, key, dflt) : std::string(dflt);
}

}  // namespace bag
}  // namespace vpipe

VPIPE_API_END

#endif

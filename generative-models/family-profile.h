#ifndef VPIPE_GENERATIVE_MODELS_FAMILY_PROFILE_H
#define VPIPE_GENERATIVE_MODELS_FAMILY_PROFILE_H

// WHAT A MODEL FAMILY KNOWS ABOUT ITSELF THAT THE HOST CANNOT, as an
// open bag, for any subsystem that needs it.
//
// The host has several closed switches over family name: which prompt
// template to wrap, which tensor leaves to quantize, which encoder
// prefix to read. Each one is a table of facts about checkpoints, and
// each one is unextendable from outside -- so a model published after
// the host has no way to answer, and moving a family out of the tree
// means finding every table it appears in.
//
// This is the one seam for all of them. A profile is a FlexData keyed by
// (DOMAIN, FAMILY TAG): the domain names the subsystem asking
// ("conditioning", "quantize"), the tag names the family, and the bag
// carries whatever that domain's vocabulary is. The key vocabularies
// live in their own headers beside this one -- conditioner-profile.h,
// quantize-profile.h -- because a key belongs with the code that reads
// it.
//
// WHY ONE METHOD AND NOT ONE PER SUBSYSTEM. The plugin ABI is a strict
// equality cookie, so every entry point added to the registration facade
// is a decision a plugin author cannot opt out of. A domain STRING moves
// that decision out of the ABI: the next subsystem that wants per-family
// facts picks a domain name, writes a key header, and needs no new
// method, no vtable change and no version bump. That is the whole
// argument for the shape.
//
// WHAT A PROFILE IS NOT. It is data, never behaviour. A domain decides
// what its keys mean and what it does when a family says nothing; a
// family cannot hand the host code to run, cannot reach machinery the
// host did not offer, and cannot switch off a policy the host enforces.
// A family whose needs exceed its domain's vocabulary ships a stage.
//
// THE TWO PROPERTIES THAT MAKE THIS WORK, and they are the same pair the
// acceleration bag rests on, because "add a key and nothing breaks" is
// exactly their conjunction:
//
//   an unknown key is IGNORED     -- a newer plugin talking to an older
//                                    host, which is expected and free
//   a missing key is the DEFAULT  -- an older plugin talking to a newer
//                                    host, which reads the same as a
//                                    family that did not care
//
// A domain that KNOWS a key and cannot honour it must say so. That is a
// third case and not the same failure: it is a family being treated
// differently than it asked, which is silent everywhere else.

#include "common/flex-data.h"

#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "common/vpipe-api.h"

VPIPE_API_BEGIN

namespace vpipe::genai::profile {

// ---- readers ---------------------------------------------------------
//
// HEADER-ONLY, so they compile into the caller -- which is what makes
// the promise directional: a plugin built against an older host reads
// the keys that host documented and gets these defaults for anything it
// has never heard of.

inline bool
has(const FlexData* p, std::string_view key)
{
  return p != nullptr && p->is_object() && p->as_object().contains(key);
}

inline bool
flag(const FlexData* p, std::string_view key, bool dflt = false)
{
  if (!has(p, key)) { return dflt; }
  return p->as_object().at(key).as_bool(dflt);
}

inline long long
integer(const FlexData* p, std::string_view key, long long dflt = 0)
{
  if (!has(p, key)) { return dflt; }
  return (long long)p->as_object().at(key).as_int(dflt);
}

inline std::string
text(const FlexData* p, std::string_view key, const char* dflt = "")
{
  if (!has(p, key)) { return dflt; }
  return std::string(p->as_object().at(key).as_string(dflt));
}

// An array of strings, EMPTY when absent -- which a caller must treat as
// "said nothing", never as "said none". The two differ: a family that
// lists no quantizable leaves would have every one of them left dense,
// and a family that did not answer wants the host's own list.
inline std::vector<std::string>
strings(const FlexData* p, std::string_view key)
{
  std::vector<std::string> out;
  if (!has(p, key)) { return out; }
  const FlexData a = p->as_object().at(key);
  if (!a.is_array()) { return out; }
  auto arr = a.as_array();
  for (std::size_t i = 0; i < arr.size(); ++i) {
    std::string s(arr.at(i).as_string(""));
    if (!s.empty()) { out.push_back(std::move(s)); }
  }
  return out;
}

inline std::vector<int>
integers(const FlexData* p, std::string_view key)
{
  std::vector<int> out;
  if (!has(p, key)) { return out; }
  const FlexData a = p->as_object().at(key);
  if (!a.is_array()) { return out; }
  auto arr = a.as_array();
  for (std::size_t i = 0; i < arr.size(); ++i) {
    out.push_back((int)arr.at(i).as_int(0));
  }
  return out;
}

// ---- the registry ----------------------------------------------------
//
// Process-wide, and keyed by a PAIR rather than hung off a model family
// interface. Three reasons, and the third is the one that decided it: a
// profile is not the property of an image family (video families need
// them too, and a subsystem may want one for a checkpoint nothing loads
// a DiT from); binding it to an interface would put the same facts in
// two places; and adding a virtual method to an existing interface moves
// its vtable, which invalidates every plugin already built against it.
// Nothing here moved anyone's vtable.
class Registry {
  VPIPE_ABI_OPAQUE;   // host-owned: see vpipe/export.h
public:
  static Registry& get() noexcept;

  // Takes a copy. FIRST-WINS on (domain, family): a second profile for a
  // pair already present is ignored and returns false, so two plugins
  // shipping one model cannot silently reshape each other.
  bool add(std::string domain, std::string family, FlexData profile);

  // Null when nobody registered one -- which is every built-in family,
  // and is the case each domain's own defaults serve.
  const FlexData* find(std::string_view domain,
                       std::string_view family) const noexcept;

  // Every family with a profile in `domain`, for a diagnostic that wants
  // to say what is registered.
  std::vector<std::string> families(std::string_view domain) const;

private:
  Registry() = default;

  mutable std::mutex                                   _mu;
  std::map<std::pair<std::string, std::string>, FlexData> _by_key;
};

}  // namespace vpipe::genai::profile

VPIPE_API_END

#endif

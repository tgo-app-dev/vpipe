#include "generative-models/quantize-family-registry.h"

#include <exception>
#include <utility>

namespace vpipe::genai {

QuantizeFamilyRegistry&
QuantizeFamilyRegistry::get() noexcept
{
  static QuantizeFamilyRegistry instance;
  return instance;
}

bool
QuantizeFamilyRegistry::add(std::unique_ptr<QuantizableFamily> f)
{
  if (!f) { return false; }
  std::string_view tag;
  try {
    tag = f->tag();
  } catch (...) {
    return false;                     // a family that cannot name itself
  }
  if (tag.empty()) { return false; }
  // The family strings the stage's own detection produces. A plugin is
  // free to be one of these architectures, but taking the NAME makes the
  // log line ambiguous about which path packaged the model -- and the
  // registry is consulted FIRST, so a collision would read as a built-in
  // while running the plugin's component list.
  for (const char* built_in : {"krea2", "flux2", "mage", "wan", "boogu",
                               "qwen-image", "minimax-h3"}) {
    if (tag == built_in) { return false; }
  }
  std::lock_guard<std::mutex> lk(_mu);
  for (const auto& e : _families) {
    if (e->tag() == tag) { return false; }   // first-wins
  }
  _families.push_back(std::move(f));
  return true;
}

QuantizableFamily*
QuantizeFamilyRegistry::claim_for(const std::string& root) const
{
  if (root.empty()) { return nullptr; }
  std::lock_guard<std::mutex> lk(_mu);
  for (const auto& e : _families) {
    try {
      if (e->claims(root)) { return e.get(); }
    } catch (...) {
      // A family that throws while probing is skipped rather than
      // taking the pipeline down: the built-in detection below it is
      // still a correct answer for everything that is not this family.
      continue;
    }
  }
  return nullptr;
}

QuantizableFamily*
QuantizeFamilyRegistry::find(std::string_view tag) const noexcept
{
  std::lock_guard<std::mutex> lk(_mu);
  for (const auto& e : _families) {
    if (e->tag() == tag) { return e.get(); }
  }
  return nullptr;
}

std::vector<QuantizableFamily*>
QuantizeFamilyRegistry::all() const
{
  std::lock_guard<std::mutex> lk(_mu);
  std::vector<QuantizableFamily*> out;
  out.reserve(_families.size());
  for (const auto& e : _families) { out.push_back(e.get()); }
  return out;
}

}  // namespace vpipe::genai

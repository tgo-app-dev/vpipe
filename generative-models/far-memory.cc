#include "generative-models/far-memory.h"

#include <mutex>
#include <utility>
#include <vector>

namespace vpipe::genai {

namespace {

struct Registry {
  std::mutex                                       mu;
  std::vector<std::unique_ptr<FarMemoryTransport>> transports;
};

Registry&
registry() noexcept
{
  static Registry r;
  return r;
}

std::string_view
name_of(const FarMemoryTransport& t) noexcept
{
  try {
    return t.name();
  } catch (...) {
    return {};
  }
}

}

bool
register_far_memory_transport(std::unique_ptr<FarMemoryTransport> t)
{
  if (!t) { return false; }
  const std::string_view nm = name_of(*t);
  if (nm.empty()) { return false; }
  Registry& r = registry();
  std::lock_guard<std::mutex> lk(r.mu);
  for (const auto& e : r.transports) {
    if (name_of(*e) == nm) { return false; }
  }
  r.transports.push_back(std::move(t));
  return true;
}

FarMemoryTransport*
find_far_memory_transport(std::string_view name)
{
  Registry& r = registry();
  std::lock_guard<std::mutex> lk(r.mu);
  for (const auto& e : r.transports) {
    if (name_of(*e) == name) { return e.get(); }
  }
  return nullptr;
}

std::shared_ptr<FarMemory>
open_far_memory(const FlexData& bag, std::string* err)
{
  auto fail = [&](std::string why) -> std::shared_ptr<FarMemory> {
    if (err != nullptr) { *err = std::move(why); }
    return nullptr;
  };
  if (!bag.is_object()) { return fail("far_memory is not an object"); }
  auto o = bag.as_object();
  if (!o.contains("transport") || !o.at("transport").is_string()) {
    return fail("far_memory names no transport");
  }
  const std::string name(o.at("transport").as_string());
  FarMemoryTransport* t = find_far_memory_transport(name);
  if (t == nullptr) {
    return fail("no far memory transport '" + name + "' is registered");
  }
  const FlexData params =
      o.contains("params") ? o.at("params") : FlexData::make_object();
  try {
    std::string e;
    std::shared_ptr<FarMemory> m = t->open(params, &e);
    if (!m) { return fail(e.empty() ? "it did not open" : e); }
    if (m->bytes() == 0) { return fail("it lends no memory"); }
    return m;
  } catch (const std::exception& x) {
    return fail(std::string("the transport threw: ") + x.what());
  } catch (...) {
    return fail("the transport threw");
  }
}

}

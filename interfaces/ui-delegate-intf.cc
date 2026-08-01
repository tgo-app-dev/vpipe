#include "interfaces/ui-delegate-intf.h"

#include <algorithm>
#include <utility>

namespace vpipe {

std::uint64_t
UiInterruptRegistry::add(std::string label, UiInterruptHandler fn)
{
  if (!fn) { return 0; }
  std::lock_guard<std::mutex> lk(_mu);
  const std::uint64_t id = _next_id++;
  _entries.push_back(Entry{id, std::move(label), std::move(fn)});
  return id;
}

void
UiInterruptRegistry::remove(std::uint64_t id) noexcept
{
  if (id == 0) { return; }
  std::lock_guard<std::mutex> lk(_mu);
  _entries.erase(
      std::remove_if(_entries.begin(), _entries.end(),
                     [id](const Entry& e) { return e.id == id; }),
      _entries.end());
}

int
UiInterruptRegistry::dispatch()
{
  // The lock is held across the handler calls on purpose: it is what
  // makes remove() (a stage tearing down) wait for an in-flight
  // dispatch instead of racing it, so a handler is never invoked
  // against state that is already being destroyed. Handlers are
  // documented to be non-blocking and must not re-enter the registry.
  std::lock_guard<std::mutex> lk(_mu);
  int acted = 0;
  for (const Entry& e : _entries) {
    if (e.fn && e.fn()) { ++acted; }
  }
  return acted;
}

std::size_t
UiInterruptRegistry::size() const
{
  std::lock_guard<std::mutex> lk(_mu);
  return _entries.size();
}

}

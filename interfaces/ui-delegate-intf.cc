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

// ---- UiProgressRegistry ----------------------------------------------
//
// A vector, not a map: this holds a handful of entries at most (one per
// concurrent long-running operation), it is walked in full on every
// snapshot, and append order IS the display order the renderers want.

std::uint64_t
UiProgressRegistry::open(std::string desc)
{
  std::lock_guard<std::mutex> lk(_mu);
  Item it;
  it.id   = _next_id++;
  it.desc = std::move(desc);
  it.seq  = ++_seq;
  _items.push_back(std::move(it));
  ++_version;
  return _items.back().id;
}

void
UiProgressRegistry::update(std::uint64_t id, std::uint64_t done,
                           std::uint64_t total, std::string detail)
{
  if (id == 0) { return; }
  std::lock_guard<std::mutex> lk(_mu);
  for (Item& it : _items) {
    if (it.id != id) { continue; }
    it.done  = done;
    it.total = total;
    // An empty detail LEAVES the previous text alone rather than
    // clearing it: update(done, total) is the common call and it must
    // not wipe a detail that set_detail() established. Clearing is
    // deliberate -- set_detail("") -- not a side effect of counting.
    if (!detail.empty()) { it.detail = std::move(detail); }
    it.seq = ++_seq;
    ++_version;
    return;
  }
}

void
UiProgressRegistry::set_detail(std::uint64_t id, std::string detail)
{
  if (id == 0) { return; }
  std::lock_guard<std::mutex> lk(_mu);
  for (Item& it : _items) {
    if (it.id != id) { continue; }
    it.detail = std::move(detail);
    it.seq    = ++_seq;
    ++_version;
    return;
  }
}

void
UiProgressRegistry::close(std::uint64_t id) noexcept
{
  if (id == 0) { return; }
  std::lock_guard<std::mutex> lk(_mu);
  const auto before = _items.size();
  _items.erase(
      std::remove_if(_items.begin(), _items.end(),
                     [id](const Item& it) { return it.id == id; }),
      _items.end());
  // Only a real removal bumps the version, so a double close (finish()
  // then the destructor) does not wake every renderer for nothing.
  if (_items.size() != before) { ++_version; }
}

std::vector<UiProgressRegistry::Item>
UiProgressRegistry::snapshot() const
{
  std::lock_guard<std::mutex> lk(_mu);
  return _items;
}

std::uint64_t
UiProgressRegistry::version() const
{
  std::lock_guard<std::mutex> lk(_mu);
  return _version;
}

std::size_t
UiProgressRegistry::size() const
{
  std::lock_guard<std::mutex> lk(_mu);
  return _items.size();
}

}

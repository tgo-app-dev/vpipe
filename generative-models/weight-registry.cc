#include "generative-models/weight-registry.h"

#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <utility>

using namespace std;

namespace vpipe::genai {

void
WeightRegistry::Registration::reset() noexcept
{
  if (_reg != nullptr && _id != 0) {
    _reg->remove_(_id);
  }
  _reg = nullptr;
  _id  = 0;
}

WeightRegistry::Registration
WeightRegistry::add(WeightOwner* owner)
{
  if (owner == nullptr) { return {}; }
  lock_guard<mutex> lk(_mu);
  const uint64_t id = _next_id++;
  _entries.push_back(Entry{id, owner, false, 0});
  return Registration(this, id);
}

void
WeightRegistry::remove_(uint64_t id) noexcept
{
  lock_guard<mutex> lk(_mu);
  _entries.erase(
      std::remove_if(_entries.begin(), _entries.end(),
                     [id](const Entry& e) { return e.id == id; }),
      _entries.end());
}

WeightRegistry::Entry*
WeightRegistry::find_(WeightOwner* owner)
{
  for (Entry& e : _entries) {
    if (e.owner == owner) { return &e; }
  }
  return nullptr;
}

size_t
WeightRegistry::park(WeightOwner* owner)
{
  // The lock is held across for_each_weight on purpose: it is what
  // makes a concurrent remove_() (the owner being destroyed) wait
  // rather than race a walk of buffers that are going away.
  lock_guard<mutex> lk(_mu);
  Entry* e = find_(owner);
  if (e == nullptr || e->parked) { return 0; }

  size_t bytes = 0;
  owner->for_each_weight([&bytes](metal_compute::SharedBuffer& b) {
    // mark_inactive declines subviews, heap sub-allocations and wired
    // buffers, so only genuinely parkable bytes are counted.
    if (b.mark_inactive()) { bytes += b.byte_size(); }
  });
  e->parked = true;
  e->bytes  = bytes;
  // Both halves of the fact, under one lock. Told even when `bytes` is
  // 0 -- nothing was parkable, but `e->parked` is set either way, and an
  // owner whose flag disagreed with this entry is the failure this hook
  // exists to prevent. The cost of the honest answer is one reactivate
  // walk on next use that finds everything already non-volatile.
  owner->note_parked(true);
  if (_session != nullptr && bytes > 0) {
    _session->log_debug(fmt(
        "WeightRegistry: parked {} MB of '{}' (reclaimable under "
        "pressure; reactivates without a reload if untouched)",
        bytes >> 20, owner->weight_label()));
  }
  return bytes;
}

bool
WeightRegistry::reactivate(WeightOwner* owner, bool* reloaded)
{
  if (reloaded != nullptr) { *reloaded = false; }

  bool   intact = true;
  size_t bytes  = 0;
  {
    lock_guard<mutex> lk(_mu);
    Entry* e = find_(owner);
    if (e == nullptr) { return true; }     // unknown: nothing was parked
    if (!e->parked) { return true; }
    // Opened BEFORE the buffers are touched and closed only after any
    // reload below, so the owner can shut its readers out for the whole
    // span in which the contents are undefined. Taken unconditionally:
    // whether a reload is needed is not known until the walk finishes,
    // and reactivation is rare enough that the extra bracket on the
    // intact path costs nothing.
    owner->begin_restore();
    owner->for_each_weight(
        [&intact, &bytes](metal_compute::SharedBuffer& b) {
          // Every buffer must be reactivated, not just up to the first
          // casualty: leaving the rest volatile would let them be taken
          // during the reload that follows.
          const bool ok = b.reactivate();
          if (!ok) { intact = false; }
          bytes += b.byte_size();
        });
    e->parked = false;
    e->bytes  = 0;
    owner->note_parked(false);
  }
  if (intact) {
    owner->end_restore(true);
    if (_session != nullptr && bytes > 0) {
      _session->log_debug(fmt(
          "WeightRegistry: reactivated {} MB of '{}' intact (no reload)",
          bytes >> 20, owner->weight_label()));
    }
    return true;
  }

  // Something was reclaimed. The contents of the discarded buffers are
  // undefined, so this is not recoverable per-buffer from here -- the
  // owner has to re-read from disk.
  if (_session != nullptr) {
    _session->info(fmt(
        "WeightRegistry: '{}' had parked weights reclaimed under memory "
        "pressure; reloading from disk", owner->weight_label()));
  }
  const bool ok = owner->reload_weights();
  owner->end_restore(ok);
  if (reloaded != nullptr) { *reloaded = ok; }
  if (!ok && _session != nullptr) {
    _session->warn(fmt(
        "WeightRegistry: reload of '{}' FAILED after its parked weights "
        "were reclaimed; the model is unusable until it is rebuilt",
        owner->weight_label()));
  }
  return false;
}

size_t
WeightRegistry::park_all()
{
  vector<WeightOwner*> owners;
  {
    lock_guard<mutex> lk(_mu);
    owners.reserve(_entries.size());
    for (const Entry& e : _entries) {
      if (!e.parked) { owners.push_back(e.owner); }
    }
  }
  size_t total = 0;
  for (WeightOwner* o : owners) { total += park(o); }
  return total;
}

WeightRegistry::Stats
WeightRegistry::stats() const
{
  lock_guard<mutex> lk(_mu);
  Stats s;
  s.owners = _entries.size();
  for (const Entry& e : _entries) {
    if (e.parked) {
      ++s.parked_owners;
      s.parked_bytes += e.bytes;
    }
    s.total_bytes += e.bytes;
  }
  return s;
}

}  // namespace vpipe::genai

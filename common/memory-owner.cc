#include "common/memory-owner.h"

#include <atomic>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <utility>

namespace vpipe {

namespace {

// Owners the table holds: directories and stage types are few, and an id
// is never given back, so a process meets far fewer than this.
constexpr std::uint32_t kMaxOwners = 4096;

// The bytes of each (owner, use): slot owner * kMemoryUses + use. Slot 0
// -- owner 0, scratch -- is no one's.
std::atomic<std::uint64_t> g_bytes[kMaxOwners * kMemoryUses];

struct Owners {
  std::mutex mu;
  std::map<std::pair<std::string, std::string>, MemoryOwnerId> ids;
  // By id - 1.
  std::vector<std::pair<std::string, std::string>> names;
};

Owners&
owners()
{
  static Owners o;
  return o;
}

struct Tags {
  std::shared_mutex mu;
  std::map<std::pair<const void*, std::uint32_t>, MemoryOwnerId> owner;
  // Read without the lock: a pool with no stage registered (none yet, or
  // a session running no pipeline) pays one load a resume.
  std::atomic<std::size_t> count{0};
};

Tags&
tags()
{
  static Tags t;
  return t;
}

thread_local MemoryOwnerId t_owner = kNoMemoryOwner;
thread_local MemoryUse     t_use = MemoryUse::kScratch;

}  // namespace

const char*
memory_use_name(MemoryUse use) noexcept
{
  switch (use) {
  case MemoryUse::kScratch: return "scratch";
  case MemoryUse::kWeights: return "weights";
  case MemoryUse::kBlocks:  return "blocks";
  case MemoryUse::kExperts: return "experts";
  case MemoryUse::kKv:      return "kv";
  }
  return "scratch";
}

MemoryOwnerId
memory_owner(std::string_view dir, std::string_view kind)
{
  if (dir.empty() && kind.empty()) {
    return kNoMemoryOwner;
  }
  Owners& o = owners();
  std::lock_guard lk(o.mu);
  auto key = std::make_pair(std::string(dir), std::string(kind));
  if (auto it = o.ids.find(key); it != o.ids.end()) {
    return it->second;
  }
  if (o.names.size() + 1 >= kMaxOwners) {
    return kNoMemoryOwner;
  }
  o.names.push_back(key);
  const auto id = static_cast<MemoryOwnerId>(o.names.size());
  o.ids.emplace(std::move(key), id);
  return id;
}

MemoryOwnerScope::MemoryOwnerScope(MemoryOwnerId owner) noexcept
  : _was(t_owner), _was_use(t_use)
{
  t_owner = owner;
  t_use = MemoryUse::kScratch;
}

MemoryOwnerScope::MemoryOwnerScope(std::string_view dir,
                                   std::string_view kind)
  : MemoryOwnerScope(memory_owner(dir, kind))
{
}

MemoryOwnerScope::~MemoryOwnerScope()
{
  t_owner = _was;
  t_use = _was_use;
}

MemoryUseScope::MemoryUseScope(MemoryUse use) noexcept : _was(t_use)
{
  t_use = use;
}

MemoryUseScope::~MemoryUseScope()
{
  t_use = _was;
}

MemoryOwnerId
memory_owner_now() noexcept
{
  return t_owner;
}

MemoryUse
memory_use_now() noexcept
{
  return t_use;
}

std::uint32_t
memory_slot_now() noexcept
{
  // No owner: no one's, whatever the use -- slot 0.
  if (t_owner == kNoMemoryOwner || t_owner >= kMaxOwners) {
    return 0;
  }
  return t_owner * kMemoryUses + static_cast<std::uint32_t>(t_use);
}

void
memory_slot_add(std::uint32_t slot, std::size_t bytes) noexcept
{
  if (slot < kMaxOwners * kMemoryUses) {
    g_bytes[slot].fetch_add(bytes, std::memory_order_relaxed);
  }
}

void
memory_slot_sub(std::uint32_t slot, std::size_t bytes) noexcept
{
  if (slot < kMaxOwners * kMemoryUses) {
    g_bytes[slot].fetch_sub(bytes, std::memory_order_relaxed);
  }
}

void
set_tag_memory_owner(const void* session, std::uint32_t tag,
                     MemoryOwnerId owner)
{
  Tags& t = tags();
  std::unique_lock lk(t.mu);
  t.owner[{session, tag}] = owner;
  t.count.store(t.owner.size(), std::memory_order_relaxed);
}

void
clear_tag_memory_owner(const void* session, std::uint32_t tag)
{
  Tags& t = tags();
  std::unique_lock lk(t.mu);
  t.owner.erase({session, tag});
  t.count.store(t.owner.size(), std::memory_order_relaxed);
}

MemoryOwnerId
tag_memory_owner(const void* session, std::uint32_t tag) noexcept
{
  Tags& t = tags();
  if (t.count.load(std::memory_order_relaxed) == 0) {
    return kNoMemoryOwner;
  }
  std::shared_lock lk(t.mu);
  auto it = t.owner.find({session, tag});
  return it != t.owner.end() ? it->second : kNoMemoryOwner;
}

std::vector<OwnedMemory>
owned_memory(std::uint64_t* none)
{
  std::vector<std::pair<std::string, std::string>> names;
  {
    Owners& o = owners();
    std::lock_guard lk(o.mu);
    names = o.names;
  }
  std::vector<OwnedMemory> out;
  for (std::size_t id = 1; id <= names.size(); ++id) {
    for (int u = 0; u < kMemoryUses; ++u) {
      const std::uint64_t b =
          g_bytes[id * kMemoryUses + u].load(std::memory_order_relaxed);
      if (b == 0) {
        continue;
      }
      out.push_back({names[id - 1].first, names[id - 1].second,
                     static_cast<MemoryUse>(u), b});
    }
  }
  if (none != nullptr) {
    *none = g_bytes[0].load(std::memory_order_relaxed);
  }
  return out;
}

}  // namespace vpipe

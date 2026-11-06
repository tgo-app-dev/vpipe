#ifndef VPIPE_COMMON_MEMORY_OWNER_H
#define VPIPE_COMMON_MEMORY_OWNER_H

// WHOSE MEMORY a buffer is. Every buffer MetalCompute makes is counted
// under the OWNER in force on its thread as it is made -- a checkpoint's
// directory and what holds it -- and under what the bytes are FOR (their
// use), and uncounted there when its owning handle goes. A host's "what is
// in RAM" view reads the totals (SessionIntf::memory_report's `owned`).
//
// WHY. The figures a session could give before were the models kept warm
// between runs and each checkpoint's weight set: what a model holds while
// it RUNS -- a streaming DiT's kept blocks, a mixture of experts' expert
// cache, a language model's K/V, a forward's activations -- was only the
// process's Metal total less those. SEEN on a 128 GB Mac: 100.7 GB of
// 106.8 read as nobody's while a MiniMax H3 clip was made, a host unable
// to say it was the clip model's.
//
// WHO SETS IT. The thread pool, around each resume of a stage's coroutine:
// the owner the pipeline registered for that stage's vertex (its largest
// declared checkpoint, its type) -- so a stage's scratch is its model's.
// The model manager, around a language model's load. A weight set, around
// what it reads: its directory, the use weights (a tensor kept, a
// language model's read) or blocks (one read as a model streams). The
// pools with a use of their own -- a DiT's block loader, an expert cache,
// a context's K/V -- mark it where they allocate. What nothing claims is
// counted as no one's.
//
// A buffer made on a thread no scope covers -- a reader thread of a
// model's own -- is no one's: an owner is per thread, and a scope is
// cheap enough to open where it matters, not free enough to infer.
//
// INTERNAL: not installed and not exported. It is the session's
// accounting, not something a plugin calls (a plugin's stages are owned
// by the pool, as every stage is).

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace vpipe {

// What the bytes are for.
enum class MemoryUse : std::uint8_t {
  kScratch = 0,   // what it allocates to run -- the default
  kWeights,       // a checkpoint's weights, kept
  kBlocks,        // a streaming model's blocks, read as it streams
  kExperts,       // a mixture of experts' expert cache
  kKv,            // a language model's K/V and recurrent state
};
inline constexpr int kMemoryUses = 5;

// "scratch", "weights", "blocks", "experts", "kv".
const char* memory_use_name(MemoryUse use) noexcept;

// An owner, interned: 0 is no one.
using MemoryOwnerId = std::uint32_t;
inline constexpr MemoryOwnerId kNoMemoryOwner = 0;

// The owner a directory and what holds it name -- the same pair, the same
// id, for the process's life. Past the table's room, no one.
MemoryOwnerId memory_owner(std::string_view dir, std::string_view kind);

// This thread's owner while it lives (nested scopes restore the outer
// one), its use reset to scratch.
class MemoryOwnerScope {
public:
  explicit MemoryOwnerScope(MemoryOwnerId owner) noexcept;
  MemoryOwnerScope(std::string_view dir, std::string_view kind);
  ~MemoryOwnerScope();
  MemoryOwnerScope(const MemoryOwnerScope&)            = delete;
  MemoryOwnerScope& operator=(const MemoryOwnerScope&) = delete;

private:
  MemoryOwnerId _was;
  MemoryUse     _was_use;
};

// This thread's use while it lives, under the owner in force.
class MemoryUseScope {
public:
  explicit MemoryUseScope(MemoryUse use) noexcept;
  ~MemoryUseScope();
  MemoryUseScope(const MemoryUseScope&)            = delete;
  MemoryUseScope& operator=(const MemoryUseScope&) = delete;

private:
  MemoryUse _was;
};

// The owner and use in force on this thread.
MemoryOwnerId memory_owner_now() noexcept;
MemoryUse     memory_use_now() noexcept;

// THE COUNTERS (MetalCompute and SharedBuffer): the slot a buffer made
// now is counted in -- its owner and use; 0, no one's -- and the bytes
// it adds or takes away.
std::uint32_t memory_slot_now() noexcept;
void          memory_slot_add(std::uint32_t slot, std::size_t bytes) noexcept;
void          memory_slot_sub(std::uint32_t slot, std::size_t bytes) noexcept;

// A pipeline's stages, by their coroutines' tag (a vertex's id, within
// a session): the owner the pool puts on a resume. Registered at a
// launch, cleared as it ends. Any thread.
void          set_tag_memory_owner(const void* session, std::uint32_t tag,
                                   MemoryOwnerId owner);
void          clear_tag_memory_owner(const void* session, std::uint32_t tag);
MemoryOwnerId tag_memory_owner(const void* session,
                               std::uint32_t tag) noexcept;

// What is counted now, owner by owner and use by use: the slots holding
// bytes. `none` the bytes of buffers no one owned as they were made.
struct OwnedMemory {
  std::string   dir;
  std::string   kind;
  MemoryUse     use = MemoryUse::kScratch;
  std::uint64_t bytes = 0;
};
std::vector<OwnedMemory> owned_memory(std::uint64_t* none = nullptr);

}  // namespace vpipe

#endif

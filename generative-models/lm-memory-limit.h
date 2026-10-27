#ifndef VPIPE_GENERATIVE_MODELS_LM_MEMORY_LIMIT_H
#define VPIPE_GENERATIVE_MODELS_LM_MEMORY_LIMIT_H

// The MOST a language model loaded on this thread may hold, in all --
// its weights as held, a Mixture of Experts' expert cache, a
// conversation's state and its working memory -- while a scope is open:
// the host's load_spec::kMemoryLimitMb, which the model manager opens
// around the load it asks for.
//
// WHY. A box's memory is not all the language model's to take: a host
// running an agent on DeepSeek-V4.1-Flash or Qwen3.8-Flash-Next beside
// picture and video models wants the model to keep, say, 64 of a 256 GB
// box and leave the rest. Those models size their expert cache to the
// box (lm_expert_cache_bytes) -- 180 GB of one at 256 GB -- and a cache
// sized at load is not resized while the model lives. Under a limit the
// cache is what the limit leaves after the rest of the model and the
// run, at least a layer's experts (the floor any box runs at). A dense
// model cannot hold less than itself: it loads whole, as without one.
//
// AND ITS EXPERT CACHE (load_spec::kExpertCacheMb). The limit leaves the
// cache whatever the run does not take, so a host that books a long
// context gets a smaller cache, and one that wants the model kept beside
// its picture models must pick the limit by subtracting. A host asking
// for the cache itself -- the trade between a conversation's room, the
// room left beside the model, and decode speed, a person's to make --
// names its size: the cache is at most that, within the limit and the
// box as before (never more than the model's own sizing would take: a
// cache past the GPU's working set is ruinous), and every expert resident
// only where all of them fit within it.
//
// WHY A SCOPE. LoadedLanguageModel's constructors are in plugin ABI 8,
// so the figure reaches the load the way a WireOnAlloc room does: on
// the loading thread (the manager builds inline). The sizing functions
// take it as an argument; only the constructor reads it here.
//
// INTERNAL: not installed and not exported.

#include <cstddef>

namespace vpipe::genai {

class LmMemoryLimit {
public:
  // 0: no limit (an open scope of 0 lifts an outer one); `cache` the
  // expert cache asked for, 0 the model's own sizing.
  explicit LmMemoryLimit(std::size_t bytes, std::size_t cache = 0) noexcept
    : _outer(slot_())
  {
    slot_() = {bytes, cache};
  }
  ~LmMemoryLimit() { slot_() = _outer; }
  LmMemoryLimit(const LmMemoryLimit&)            = delete;
  LmMemoryLimit& operator=(const LmMemoryLimit&) = delete;

  // The limit in force on this thread; 0 none.
  static std::size_t current() noexcept { return slot_().limit; }
  // The expert cache asked for on this thread; 0 none.
  static std::size_t expert_cache() noexcept { return slot_().cache; }

private:
  struct Held {
    std::size_t limit = 0, cache = 0;
  };
  static Held& slot_() noexcept
  {
    thread_local Held held;
    return held;
  }

  Held _outer;
};

}  // namespace vpipe::genai

#endif

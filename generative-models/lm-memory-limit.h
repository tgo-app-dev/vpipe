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
  // 0: no limit (an open scope of 0 lifts an outer one).
  explicit LmMemoryLimit(std::size_t bytes) noexcept
    : _outer(slot_())
  {
    slot_() = bytes;
  }
  ~LmMemoryLimit() { slot_() = _outer; }
  LmMemoryLimit(const LmMemoryLimit&)            = delete;
  LmMemoryLimit& operator=(const LmMemoryLimit&) = delete;

  // The limit in force on this thread; 0 none.
  static std::size_t current() noexcept { return slot_(); }

private:
  static std::size_t& slot_() noexcept
  {
    thread_local std::size_t bytes = 0;
    return bytes;
  }

  std::size_t _outer;
};

}  // namespace vpipe::genai

#endif

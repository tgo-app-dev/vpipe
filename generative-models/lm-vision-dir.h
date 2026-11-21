#ifndef VPIPE_GENERATIVE_MODELS_LM_VISION_DIR_H
#define VPIPE_GENERATIVE_MODELS_LM_VISION_DIR_H

// Where a language model loaded on this thread finds a VISION TOWER its
// checkpoint does not carry, while a scope is open: the host's
// load_spec::kVisionDir, RESOLVED once (text-chat: named, else found
// beside the checkpoint), which the model manager opens around the load
// -- as LmMemoryLimit carries the memory keys, for the same reason:
// LoadedLanguageModel's constructors are in plugin ABI 8. The sizing
// functions take the folder as an argument; only the constructor reads
// it here.
//
// INTERNAL: not installed and not exported.

#include <string>
#include <utility>

namespace vpipe::genai {

class LmVisionDir {
public:
  explicit LmVisionDir(std::string dir) : _outer(slot_())
  {
    slot_() = std::move(dir);
  }
  ~LmVisionDir() { slot_() = std::move(_outer); }
  LmVisionDir(const LmVisionDir&)            = delete;
  LmVisionDir& operator=(const LmVisionDir&) = delete;

  // The folder in force on this thread; "" none named.
  static const std::string& current() noexcept { return slot_(); }

private:
  static std::string& slot_() noexcept
  {
    thread_local std::string held;
    return held;
  }

  std::string _outer;
};

}  // namespace vpipe::genai

#endif

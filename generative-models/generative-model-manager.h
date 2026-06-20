#ifndef VPIPE_GENERATIVE_MODELS_GENERATIVE_MODEL_MANAGER_H
#define VPIPE_GENERATIVE_MODELS_GENERATIVE_MODEL_MANAGER_H

#include "common/session-member.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace vpipe {
class SessionContextIntf;
}

// Header is portable: every nested type is either a built-in or a
// forward-declared opaque pointer, so common/session.cc can include
// this unconditionally and the unique_ptr<...> member destructor in
// ~Session sees a complete type even on builds where the .cc impl
// (apple-silicon only) isn't linked. Same trick we use for
// CoreMLModelManager.

namespace vpipe::genai {

class LoadedLanguageModel;   // defined in a later slice

// Parameters that fully determine which model is loaded and how the
// per-load resources (page pool size, KV dtype) are sized. The
// manager dedupes by the full spec, not just the directory: loading
// the same HF dir at a different dtype yields a separate
// LoadedLanguageModel.
struct LoadSpec {
  // Path to a Hugging Face style directory containing config.json,
  // tokenizer.json, and one or more *.safetensors shards.
  std::string   hf_dir;

  // Compute / weight dtype the model runs at. Encoded here as a
  // string ("bf16", "f16", "f32") so the header stays portable
  // (mlx::core::Dtype is an MLX type that we don't pull into the
  // public surface yet). The .cc impl maps the string to MLX at
  // load time; bad values error out cleanly.
  std::string   compute_dtype = "bf16";

  // K/V cache page size in tokens. Smaller pages waste less when
  // sequences are short; larger pages reduce indirection. Default
  // tuned for general-purpose chat workloads.
  int           page_tokens   = 512;

  // Page-pool capacity. seq_len <= page_tokens * max_pages across
  // all contexts of this model combined.
  std::uint32_t max_pages     = 4096;
};

// Session-shared cache of loaded language models. Two loads with the
// same LoadSpec (including dtype + page sizing) share one in-memory
// model; loads with different specs parallelise. Same machinery as
// CoreMLModelManager:
//
//   * `_cache` holds weak_ptrs so the model unloads when the last
//     caller drops its shared_ptr.
//   * `_mu` guards the map; the actual load runs OUTSIDE the lock so
//     concurrent loads of different keys don't serialise.
//   * A double-checked second lock-and-store path means at most one
//     load wastes work in the rare two-callers-same-key race; either
//     winner is correct.
//
// v1 scaffold: `load()` currently logs an `unimplemented` warning and
// returns nullptr. The full pipeline (ModelLoader -> Tokenizer ->
// ContextManager -> LlamaModelExec) lands in subsequent commits.
class GenerativeModelManager final : public SessionMember {
public:
  explicit GenerativeModelManager(const SessionContextIntf* session);

  GenerativeModelManager(const GenerativeModelManager&)            = delete;
  GenerativeModelManager& operator=(const GenerativeModelManager&) = delete;

  ~GenerativeModelManager() override = default;

  // Returns a shared model handle, loading on first request for this
  // spec. Returns nullptr on failure (the failure is reported through
  // session->warn()). v1 always returns nullptr.
  std::shared_ptr<LoadedLanguageModel>
  load(const LoadSpec& spec);

  // Diagnostics / tests: number of live entries in the cache. Walks
  // the map under `_mu`; not on any hot path.
  std::size_t cached_count() const;

private:
  struct Key {
    std::string   hf_dir;
    std::string   compute_dtype;
    int           page_tokens;
    std::uint32_t max_pages;

    bool operator==(const Key& o) const noexcept
    {
      return page_tokens == o.page_tokens
          && max_pages   == o.max_pages
          && compute_dtype == o.compute_dtype
          && hf_dir       == o.hf_dir;
    }
  };
  struct KeyHash {
    std::size_t operator()(const Key& k) const noexcept;
  };

  mutable std::mutex                                          _mu;
  std::unordered_map<Key,
                     std::weak_ptr<LoadedLanguageModel>,
                     KeyHash>                                 _cache;
};

}

#endif

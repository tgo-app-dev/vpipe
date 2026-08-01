#ifndef VPIPE_GENERATIVE_MODELS_GENERATIVE_MODEL_MANAGER_H
#define VPIPE_GENERATIVE_MODELS_GENERATIVE_MODEL_MANAGER_H

#include "common/session-member.h"
#include "generative-models/weight-registry.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

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
class WeightSet;             // generative-models/weight-set.h

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

  // ---- shared non-LM models ----------------------------------------
  //
  // The LM cache above dedups language models. This does the same for
  // everything else -- DiTs, VAEs, text encoders, TTS -- so a graph
  // whose conditioner, DiT, vae-encode and vae-decode all name one
  // model directory loads its weights ONCE instead of four times.
  //
  // `kind` names the model class ("boogu-dit", "flux2-vae"): it is part
  // of the key, so two different classes reading the same directory
  // never collide. `variant` is any further config that changes the
  // bytes (quantization width, streaming mode); "" when nothing does.
  // `build` runs OUTSIDE the lock and only on a miss.
  //
  // LIFETIME. The cache holds weak references, so the model dies with
  // its last holder exactly as an un-shared one did -- but "last
  // holder" now spans stages. A stage dropping its handle while a peer
  // still holds one keeps the weights alive, which is the point.
  template <class T>
  std::shared_ptr<T>
  shared_model(const std::string&                        kind,
               const std::string&                        dir,
               const std::string&                        variant,
               const std::function<std::shared_ptr<T>()>& build)
  {
    const std::string key = shared_key_(kind, dir, variant);
    if (auto hit = lookup_shared_(key)) {
      return std::static_pointer_cast<T>(hit);
    }
    std::shared_ptr<T> made = build ? build() : nullptr;
    if (!made) { return nullptr; }
    // Double-checked: another caller may have won the race while we
    // were loading. Either instance is correct; keep the stored one so
    // everyone converges on a single copy.
    if (auto hit = store_shared_(key, made)) {
      return std::static_pointer_cast<T>(hit);
    }
    return made;
  }

  // Look up WITHOUT loading: returns the live instance for this key or
  // null. Lets a caller reuse a SUPERSET another stage already built --
  // vae-decode reusing vae-encode's encoder-capable VAE, which decodes
  // just as well -- instead of loading its own narrower copy.
  template <class T>
  std::shared_ptr<T>
  existing_model(const std::string& kind,
                 const std::string& dir,
                 const std::string& variant)
  {
    if (auto hit = lookup_shared_(shared_key_(kind, dir, variant))) {
      return std::static_pointer_cast<T>(hit);
    }
    return nullptr;
  }

  // Diagnostics / tests: live entries in the shared-model cache.
  std::size_t shared_count() const;

  // ---- checkpoint weights ------------------------------------------
  //
  // The single place a checkpoint's weights are opened and cached. Ask
  // for a directory and get the ONE WeightSet for it; a second caller
  // naming the same directory gets the same object, so its weights are
  // reference-counted rather than loaded again. The set is unmapped
  // when the last holder drops it.
  //
  // This is dedup at the TENSOR level, which is what makes it robust:
  // two models over one checkpoint share every tensor they have in
  // common no matter what order they were built in or which subset each
  // one needs. (Deduping whole MODEL objects instead needs whoever
  // builds the superset to win the race, and stage initialize() bodies
  // run concurrently, so that ordering is not ours to assume.)
  //
  // Returns null when the directory holds no readable checkpoint.
  // `variant` separates sets that must NOT share bytes despite one
  // directory; leave it "" unless that is genuinely the case.
  std::shared_ptr<WeightSet> weight_set(const std::string& dir,
                                        const std::string& variant = {});

  // Live entries in the weight-set cache (diagnostics / tests).
  std::size_t weight_set_count() const;

  // True when `dir` already has an open weight set, i.e. a model naming
  // it again would share the bytes rather than add any. Matches on the
  // directory, whatever variant the set was opened under.
  bool holds_weights(const std::string& dir) const;

  // Weight bytes this session is holding right now, summed over open
  // checkpoints -- each counted ONCE however many models share it.
  //
  // This is the number a memory decision wants, and it is strictly
  // better than summing directory sizes off disk: it excludes what is
  // shared, and it reflects what a model is ACTUALLY holding rather
  // than what its files weigh. A streaming DiT is the clearest case --
  // it holds only its pinned prefix, so a peer stage sizing the box
  // against it sees the real (small) number instead of the full
  // checkpoint.
  //
  // Includes mapped bytes, which the OS can reclaim on its own; that is
  // the conservative reading and matches what the on-disk estimate it
  // replaces would have counted.
  std::size_t resident_weight_bytes() const;

  // ---- declared intent ---------------------------------------------
  //
  // A stage announcing, BEFORE any model loads, that this graph will
  // open `dir`. Until the declarations are cleared, resident_weight_bytes()
  // counts a declared checkpoint at whichever is larger: what it is
  // actually holding, or `expected_bytes`.
  //
  // The max() is what makes a decision taken mid-load correct. A peer
  // that is 30% through loading a 19 GB DiT genuinely holds 30% of it,
  // and a stage sizing the box against that number would conclude it has
  // room it does not have.
  //
  // Declarations persist for the whole run, and that is deliberate.
  //
  // The obvious alternative -- drop them once loading is done, on the
  // grounds that the real bytes are then authoritative -- is WRONG here,
  // because a weight set only knows what it CACHED. Models that read
  // uncached hold their weights in their own members: the LMs do
  // exactly this (see WeightSet::read), so a 2.9 GB checkpoint reports
  // ~0.6 GB of cached tensors and the rest is invisible. Clearing would
  // make an LM's weights vanish from the accounting the moment it
  // finished loading, which is precisely when peers start sizing the box
  // against it.
  //
  // So the estimate stands unless the model itself says otherwise --
  // see revise_declaration(), which the streaming DiTs use to report
  // the much smaller amount they actually keep.
  void declare_weights(const std::string& dir, std::size_t expected_bytes);

  // Correct a declaration downward (or upward) once the model knows what
  // it really holds. The streaming DiTs are the case that needs it: a
  // 20 GB checkpoint streamed block-by-block keeps only its pinned
  // prefix, and left at its on-disk estimate it would push every peer
  // into streaming they do not need.
  void revise_declaration(const std::string& dir, std::size_t bytes);

  // Drop every declaration. PipelineRuntime calls this at the start of
  // each launch, so one run's estimates never leak into the next.
  void clear_declarations();

  // True when `dir` is already accounted for -- open, or declared. The
  // question a footprint calculation asks before adding a directory's
  // on-disk size, so it does not count the same checkpoint twice.
  bool accounts_for(const std::string& dir) const;

  // ---- KV -----------------------------------------------------------
  //
  // KV / recurrent-state bytes across every language model this manager
  // is holding. Separate from the weights on purpose: weights are
  // allocated once at load and then constant, while KV grows with the
  // conversation. Accounting that only sees weights reads as healthy
  // right up to the point a long context exhausts the box.
  //
  // Note what this means for the load-time decisions: at the moment a
  // stage sizes the box, KV is ~0 for every model, so folding it into
  // that estimate would add nothing. It is the number a running memory
  // CAP needs, and what a memory report has to show to be honest.
  std::size_t resident_kv_bytes() const;

  // Everything this manager can account for: weights + KV. Still not the
  // whole process -- activations, the Metal heap, CoreML models and LMDB
  // sit outside it -- so treat it as the dominant term, not a total.
  std::size_t resident_bytes() const;

  // ---- memory cap ---------------------------------------------------
  //
  // A ceiling on what this session keeps ACTIVELY resident: weight bytes
  // plus KV. 0 (the default) means no cap.
  //
  // "Actively" is the load-bearing word. Going over the cap does NOT
  // fail a load -- it PARKS weights, handing their pages to the kernel
  // as purgeable. Parked pages survive whenever nothing else needs the
  // RAM (so the cost is usually nothing at all) and are reclaimed when
  // something does, and the next access to a parked set takes them back,
  // re-reading from disk only if they were actually taken. So the cap
  // bounds what the process INSISTS on holding, not what it has
  // allocated, and overshooting degrades throughput rather than
  // correctness.
  //
  // What it does not cover: activations, the Metal heap, CoreML models
  // and LMDB are outside the manager entirely. It is a cap on the
  // dominant term.
  void        set_memory_cap(std::size_t bytes);
  std::size_t memory_cap() const;

  // Weights currently held UNPARKED, plus KV. This is the number the cap
  // is compared against; resident_bytes() counts parked sets too.
  std::size_t active_bytes() const;

  // Park least-recently-used weight sets until active_bytes() is under
  // the cap, or nothing parkable is left. Returns the bytes handed over.
  //
  // Runs automatically after each checkpoint opens and each model loads.
  // Exposed for tests and for a caller that has just allocated something
  // large the manager cannot see.
  std::size_t enforce_memory_cap();

  // What is resident right now, per checkpoint. The point of routing
  // loads through the manager is that this question has an answer: how
  // much each model is holding, how much of it is OS-reclaimable
  // (mapped) versus owned, how many holders it has, and which optional
  // parts were actually needed. Feeds diagnostics and any policy that
  // has to decide what to give up first.
  struct WeightUsage {
    std::string dir;
    std::size_t bytes        = 0;
    std::size_t mapped_bytes = 0;   // clean file-backed; OS reclaims
    std::size_t copied_bytes = 0;   // owned; parkable + reloadable
    std::size_t tensors      = 0;
    std::size_t parts        = 0;
    long        holders      = 0;   // models keeping this set alive
  };
  std::vector<WeightUsage> weight_report() const;

  // Session-wide weight residency (see weight-registry.h). Models
  // register their weight buffers here so idle ones can be parked --
  // kept allocated but reclaimable -- instead of being destroyed and
  // reloaded. Shared by every model family, not just the LMs this
  // manager caches, because the RAM they compete for is one pool.
  WeightRegistry& weights() noexcept { return _weights; }

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

  // Canonicalized "kind|dir|variant" key for shared_model().
  std::string shared_key_(const std::string& kind,
                          const std::string& dir,
                          const std::string& variant) const;
  // Live entry for `key`, or null. Takes _shared_mu.
  std::shared_ptr<void> lookup_shared_(const std::string& key);
  // Store `v` unless another thread already stored one, in which case
  // the WINNER is returned (and the caller drops its own copy).
  std::shared_ptr<void> store_shared_(const std::string&           key,
                                      const std::shared_ptr<void>& v);

  mutable std::mutex                                          _shared_mu;
  std::unordered_map<std::string, std::weak_ptr<void>>        _shared;
  mutable std::mutex                                          _ws_mu;
  std::unordered_map<std::string, std::weak_ptr<WeightSet>>   _weight_sets;
  // canonical dir -> estimated bytes; see declare_weights().
  std::unordered_map<std::string, std::size_t>                _declared;
  WeightRegistry                                              _weights;
  std::atomic<std::size_t>                                    _memory_cap{0};
  mutable std::mutex                                          _mu;
  std::unordered_map<Key,
                     std::weak_ptr<LoadedLanguageModel>,
                     KeyHash>                                 _cache;
};

}

#endif

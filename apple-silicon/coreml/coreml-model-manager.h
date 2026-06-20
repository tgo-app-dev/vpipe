#ifndef VPIPE_APPLE_SILICON_COREML_MODEL_MANAGER_H
#define VPIPE_APPLE_SILICON_COREML_MODEL_MANAGER_H

#include "common/session-member.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace CML {
class Model;
}

namespace vpipe {

class SessionContextIntf;

// Per-output metadata captured at model-load time. `fixed == true`
// means CoreML reports a fully-specified shape (every dim > 0); the
// stage can pre-allocate output TensorBeats and bind them via
// PredictionOptions::setOutputBackings for zero-copy. `fixed == false`
// covers flexible / range / enumerated shapes -- the stage falls back
// to allocate-and-memcpy on the result.
struct CoreMLOutputDesc {
  std::vector<int64_t> shape;
  bool                 fixed = false;
};

// One loaded CoreML model. Wraps a retained `CML::Model*` plus a
// per-model std::mutex used to serialise prediction calls. CoreML's
// runtime serialises GPU/ANE prediction internally anyway; we
// serialise at our level so threads block on a cheap, observable
// mutex instead of stalling opaquely inside the framework.
//
// Non-copyable, non-movable. Lifetime is managed via shared_ptr from
// CoreMLModelManager.
class CoreMLLoadedModel final : public SessionMember {
public:
  // Loads `mlmodelc_path` (a directory; .mlmodel files are compiled
  // upstream in the manager). compute_units is the
  // CML::ModelConfiguration::setComputeUnits value: 0=CPUOnly,
  // 1=CPU+GPU, 2=All, 3=CPU+NeuralEngine. Errors via
  // session->error() (throws); the manager catches.
  CoreMLLoadedModel(const SessionContextIntf* session,
                    std::string_view          mlmodelc_path,
                    int                       compute_units);

  CoreMLLoadedModel(const CoreMLLoadedModel&)            = delete;
  CoreMLLoadedModel& operator=(const CoreMLLoadedModel&) = delete;

  ~CoreMLLoadedModel() override;

  CML::Model* model() const noexcept { return _model; }
  std::mutex& predict_mutex() noexcept { return _predict_mu; }

  // Output descriptors discovered at load time, keyed by output
  // feature name. Empty if introspection failed (the stage then
  // takes the memcpy fallback).
  const std::unordered_map<std::string, CoreMLOutputDesc>&
  output_descs() const noexcept { return _outputs; }

  // For diagnostics + tests.
  const std::string& path() const noexcept { return _path; }
  int compute_units() const noexcept { return _compute_units; }

private:
  CML::Model*                                              _model;
  std::mutex                                               _predict_mu;
  std::unordered_map<std::string, CoreMLOutputDesc>        _outputs;
  std::string                                              _path;
  int                                                      _compute_units;
};

// Session-shared cache of loaded CoreML models. Keyed by
// (canonical_path, compute_units) so two stages requesting the same
// model with the same compute units share a single load. Models are
// reference-counted via shared_ptr; once the last reference drops,
// the model unloads (the cache holds weak_ptr only).
//
// Threading: `load()` is safe to call concurrently. The cache map is
// guarded by `_mu`; the actual model load happens *outside* the map
// lock so concurrent loads of *different* keys parallelise. Two
// concurrent loads of the *same* key may both perform the load -- a
// rare race that wastes one load, never produces UB; the cache
// stabilises on whichever winner wins the second lock.
class CoreMLModelManager final : public SessionMember {
public:
  explicit CoreMLModelManager(const SessionContextIntf* session);

  CoreMLModelManager(const CoreMLModelManager&)            = delete;
  CoreMLModelManager& operator=(const CoreMLModelManager&) = delete;

  ~CoreMLModelManager() override = default;

  // Returns a shared model, loading on first request. `path` may be
  // a .mlmodel file (compiled to a tmp .mlmodelc inside) or a
  // .mlmodelc directory. Returns nullptr on failure (the failure is
  // already reported through session->warn()).
  std::shared_ptr<CoreMLLoadedModel>
  load(std::string_view path, int compute_units);

  // Test / diagnostics helpers.
  std::size_t cached_count() const;

private:
  struct Key {
    std::string path;
    int         compute_units;
    bool operator==(const Key& o) const noexcept
    { return compute_units == o.compute_units && path == o.path; }
  };
  struct KeyHash {
    std::size_t operator()(const Key& k) const noexcept;
  };

  mutable std::mutex                                      _mu;
  std::unordered_map<Key,
                     std::weak_ptr<CoreMLLoadedModel>,
                     KeyHash>                             _cache;
};

}

#endif

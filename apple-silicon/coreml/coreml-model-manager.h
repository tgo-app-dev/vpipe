#ifndef VPIPE_APPLE_SILICON_COREML_MODEL_MANAGER_H
#define VPIPE_APPLE_SILICON_COREML_MODEL_MANAGER_H

#include "common/session-member.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace CML {
class Model;
}

namespace vpipe {

class SessionContextIntf;

// Element type on the CoreML predict() I/O boundary. Independent of
// TensorBeat::DType -- it names the types MLMultiArray actually uses
// here: f32, IEEE half (f16), double (f64, an output-only shape some
// models emit), and int32 (recurrent-state inputs). A native consumer
// never sees a CML:: type, so this enum is the whole dtype vocabulary
// the plugin surface needs.
enum class CoreMLDType : std::uint8_t { F32, F16, F64, I32 };

// Byte width of one CoreMLDType element.
constexpr std::size_t
coreml_dtype_size(CoreMLDType d) noexcept
{
  switch (d) {
    case CoreMLDType::F32: return 4;
    case CoreMLDType::F16: return 2;
    case CoreMLDType::F64: return 8;
    case CoreMLDType::I32: return 4;
  }
  return 0;
}

// Whether a model feature is a dense tensor (MLMultiArray) or an image
// (bound as a CVPixelBuffer inside predict()).
enum class CoreMLFeatureKind : std::uint8_t { MultiArray, Image };

// Per-input metadata captured at load time (the input-side mirror of
// CoreMLOutputDesc). `fixed == true` means every dim / image size is
// fully specified. For an Image input, `pixel_format` is the model's
// CVPixelFormatType (predict() supports 32-bit BGRA only). `shape` /
// `dtype` describe a MultiArray input; `image_*` describe an Image.
struct CoreMLInputDesc {
  CoreMLFeatureKind    kind = CoreMLFeatureKind::MultiArray;
  std::vector<int64_t> shape;                     // MultiArray dims
  CoreMLDType          dtype = CoreMLDType::F32;   // MultiArray element
  int                  image_width  = 0;           // Image (0 == flex)
  int                  image_height = 0;
  std::uint32_t        pixel_format = 0;           // Image CVPixelFormat
  bool                 fixed = false;
};

// Per-output metadata captured at model-load time. `fixed == true`
// means CoreML reports a fully-specified shape (every dim > 0); the
// consumer can pre-allocate a buffer and hand it to predict() as an
// output `backing` for zero-copy. `fixed == false` covers flexible /
// range / enumerated shapes -- predict() then allocates and fills its
// own `owned` buffer. `dtype` is the model's native output element type.
struct CoreMLOutputDesc {
  std::vector<int64_t> shape;
  CoreMLDType          dtype = CoreMLDType::F32;
  bool                 fixed = false;
};

// One MODEL INPUT bound for CoreMLLoadedModel::predict(). Populate the
// form matching the model's input kind (see input_descs()):
//   * MultiArray -- set `data` (borrowed, laid out per `dtype`),
//     `shape`, and optionally `strides` (element strides; empty ==
//     row-major contiguous). `data` may point at UMA / Metal-shared
//     memory; CoreML reads it zero-copy.
//   * Image -- set `image` to width*height interleaved 8-bit BGRA
//     pixels (`image_row_bytes` 0 == tight `image_width*4`); predict()
//     stages it into a CVPixelBuffer of the model's fixed size.
// `name` may be left empty for a single-input model.
struct CoreMLPredictInput {
  std::string          name;

  // MultiArray form.
  const void*          data = nullptr;
  CoreMLDType          dtype = CoreMLDType::F32;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;

  // Image form (used when `image != nullptr`).
  const std::uint8_t*  image = nullptr;
  int                  image_width  = 0;
  int                  image_height = 0;
  std::size_t          image_row_bytes = 0;
};

// One MODEL OUTPUT requested from CoreMLLoadedModel::predict(). predict()
// fills `shape` and points `data` at the delivered bytes (dtype ==
// `want`). Provide `backing` (capacity `backing_elems` elements of
// `want`) to receive the result in place: when the model output is
// fixed-shape and its native dtype equals `want`, CoreML writes there
// zero-copy; otherwise predict() converts the result into `backing`
// (if it fits) or into the `owned` buffer. `name` may be left empty for
// a single-output model.
struct CoreMLPredictOutput {
  std::string               name;
  CoreMLDType               want = CoreMLDType::F32;
  void*                     backing = nullptr;
  std::size_t               backing_elems = 0;

  // Filled by predict().
  std::vector<int64_t>      shape;
  std::vector<std::uint8_t> owned;
  const void*               data = nullptr;
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

  // True once the underlying CoreML model bound successfully. The raw
  // CML::Model* is intentionally NOT exposed: every prediction goes
  // through predict() so a consumer never needs the coreml-cpp headers.
  bool valid() const noexcept { return _model != nullptr; }

  // Run one prediction. Binds every entry of `inputs` as a model input
  // feature, retrieves every entry of `outputs`, and returns true on
  // success. ALL CoreML / Foundation / CoreVideo marshaling and the
  // per-model serialization happen inside -- a caller never touches
  // coreml-cpp. `cpu_only` forces CPU execution for this call
  // (overriding the load-time compute units). On any failure logs via
  // session() and returns false (leaving `outputs` empty).
  bool predict(std::span<const CoreMLPredictInput> inputs,
               std::span<CoreMLPredictOutput>      outputs,
               bool                                cpu_only = false);

  // Input descriptors discovered at load time, keyed by input feature
  // name (the input-side mirror of output_descs()). Empty if
  // introspection failed.
  const std::unordered_map<std::string, CoreMLInputDesc>&
  input_descs() const noexcept { return _inputs_desc; }

  // Output descriptors discovered at load time, keyed by output
  // feature name. Empty if introspection failed (the stage then
  // takes the memcpy fallback).
  const std::unordered_map<std::string, CoreMLOutputDesc>&
  output_descs() const noexcept { return _outputs; }

  // Input / output feature names discovered at load time. CoreML does
  // not guarantee an enumeration order across keys, so a consumer that
  // wants to derive "the" feature name should require a single entry
  // (single-I/O models, the common case). Empty if introspection
  // failed.
  const std::vector<std::string>& input_names() const noexcept
  { return _input_names; }
  const std::vector<std::string>& output_names() const noexcept
  { return _output_names; }

  // For diagnostics + tests.
  const std::string& path() const noexcept { return _path; }
  int compute_units() const noexcept { return _compute_units; }

private:
  CML::Model*                                              _model;
  std::mutex                                               _predict_mu;
  std::unordered_map<std::string, CoreMLInputDesc>         _inputs_desc;
  std::unordered_map<std::string, CoreMLOutputDesc>        _outputs;
  std::vector<std::string>                                 _input_names;
  std::vector<std::string>                                 _output_names;
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

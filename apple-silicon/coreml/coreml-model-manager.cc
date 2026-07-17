#include "apple-silicon/coreml/coreml-model-manager.h"

#include "apple-silicon/coreml/coreml-cpp/CoreML.hpp"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <CoreVideo/CVPixelBuffer.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <system_error>
#include <utility>

using namespace std;

namespace vpipe {

namespace {

// Build NS::String from std::string (UTF-8). Returned object is
// autoreleased.
NS::String*
ns_str_(string_view s)
{
  return NS::String::string(string(s).c_str(),
                            NS::UTF8StringEncoding);
}

// Parse an NS::Array<NSNumber*> shape into a vector<int64_t>. Each
// dim that's <= 0 indicates a flexible shape; we surface that to
// the caller via the returned `fixed` flag.
struct ShapeRead {
  vector<int64_t> shape;
  bool            fixed = true;
};

ShapeRead
read_shape_(const NS::Array* arr)
{
  ShapeRead out;
  if (!arr) {
    out.fixed = false;
    return out;
  }
  NS::UInteger n = arr->count();
  out.shape.reserve(n);
  for (NS::UInteger i = 0; i < n; ++i) {
    auto* num = arr->object<NS::Number>(i);
    int64_t v = num ? num->longLongValue() : 0;
    if (v <= 0) {
      out.fixed = false;
    }
    out.shape.push_back(v);
  }
  // Empty shape (0-rank) is not "fixed" in any useful sense for
  // pre-allocation either.
  if (out.shape.empty()) {
    out.fixed = false;
  }
  return out;
}

// Map a CoreML MLMultiArray dtype to the neutral CoreMLDType. Unknown
// types fall back to F32 -- only the four listed occur in the models we
// run; a stray type is then read as f32 rather than crashing.
CoreMLDType
from_marray_dtype_(CML::MultiArrayDataType dt) noexcept
{
  switch (dt) {
    case CML::MultiArrayDataTypeFloat16: return CoreMLDType::F16;
    case CML::MultiArrayDataTypeDouble:  return CoreMLDType::F64;
    case CML::MultiArrayDataTypeInt32:   return CoreMLDType::I32;
    case CML::MultiArrayDataTypeFloat32:
    default:                             return CoreMLDType::F32;
  }
}

// Map the neutral CoreMLDType to the CoreML MLMultiArray dtype used to
// build/bind an array.
CML::MultiArrayDataType
to_marray_dtype_(CoreMLDType d) noexcept
{
  switch (d) {
    case CoreMLDType::F16: return CML::MultiArrayDataTypeFloat16;
    case CoreMLDType::I32: return CML::MultiArrayDataTypeInt32;
    case CoreMLDType::F64: return CML::MultiArrayDataTypeDouble;
    case CoreMLDType::F32: return CML::MultiArrayDataTypeFloat32;
  }
  return CML::MultiArrayDataTypeFloat32;
}

// Build an NS::Array<NSNumber*> from an int64 vector (shape / strides).
NS::Array*
ns_num_array_(const vector<int64_t>& v)
{
  vector<const NS::Object*> nums;
  nums.reserve(v.size());
  for (auto d : v) {
    nums.push_back(static_cast<const NS::Object*>(
        NS::Number::number(static_cast<long long>(d))));
  }
  return NS::Array::array(nums.data(),
                          static_cast<NS::UInteger>(nums.size()));
}

// Read element `i` of a CoreML MLMultiArray as a double (the widest
// lossless carrier for the f16/f32/f64/i32 set predict() decodes).
double
read_elem_(const void* base, CML::MultiArrayDataType dt,
           size_t i) noexcept
{
  switch (dt) {
    case CML::MultiArrayDataTypeFloat16:
      return static_cast<double>(static_cast<const _Float16*>(base)[i]);
    case CML::MultiArrayDataTypeDouble:
      return static_cast<const double*>(base)[i];
    case CML::MultiArrayDataTypeInt32:
      return static_cast<double>(static_cast<const int32_t*>(base)[i]);
    case CML::MultiArrayDataTypeFloat32:
    default:
      return static_cast<double>(static_cast<const float*>(base)[i]);
  }
}

// Write `val` into element `i` of a CoreMLDType-typed destination.
void
write_elem_(void* base, CoreMLDType want, size_t i, double val) noexcept
{
  switch (want) {
    case CoreMLDType::F16:
      static_cast<_Float16*>(base)[i] = static_cast<_Float16>(val);
      break;
    case CoreMLDType::F64:
      static_cast<double*>(base)[i] = val;
      break;
    case CoreMLDType::I32:
      static_cast<int32_t*>(base)[i] = static_cast<int32_t>(val);
      break;
    case CoreMLDType::F32:
      static_cast<float*>(base)[i] = static_cast<float>(val);
      break;
  }
}

// True when a CoreML output dtype and a requested CoreMLDType share an
// identical byte layout, so the result can be memcpy'd rather than
// element-converted.
bool
dtype_same_(CML::MultiArrayDataType dt, CoreMLDType want) noexcept
{
  return (dt == CML::MultiArrayDataTypeFloat32 && want == CoreMLDType::F32)
      || (dt == CML::MultiArrayDataTypeFloat16 && want == CoreMLDType::F16)
      || (dt == CML::MultiArrayDataTypeInt32   && want == CoreMLDType::I32)
      || (dt == CML::MultiArrayDataTypeDouble  && want == CoreMLDType::F64);
}

// Walk modelDescription.outputDescriptionsByName() and capture each
// MLMultiArray output's shape + dtype. Non-MultiArray outputs (image,
// dict, sequence) end up missing from the map and the stage falls back
// to the legacy memcpy path for them.
void
discover_outputs_(CML::Model* model,
                  unordered_map<string, CoreMLOutputDesc>* out)
{
  auto* desc = model->modelDescription();
  if (!desc) {
    return;
  }
  auto* dict = desc->outputDescriptionsByName();
  if (!dict) {
    return;
  }
  auto* keys = dict->keyEnumerator<NS::String>();
  if (!keys) {
    return;
  }
  while (auto* k = keys->nextObject()) {
    auto* fd = dict->object<CML::FeatureDescription>(k);
    if (!fd) {
      continue;
    }
    auto* mac = fd->multiArrayConstraint();
    if (!mac) {
      continue;
    }
    string name(k->utf8String());
    ShapeRead sr = read_shape_(mac->shape());
    out->emplace(std::move(name),
                 CoreMLOutputDesc{ std::move(sr.shape),
                                   from_marray_dtype_(mac->dataType()),
                                   sr.fixed });
  }
}

// Walk modelDescription.inputDescriptionsByName() and capture each
// input's kind + shape/dtype (MultiArray) or size/pixel-format (Image).
// The input-side mirror of discover_outputs_; predict() consults it to
// build the right feature and to pick an image input's pixel format.
void
discover_inputs_(CML::Model* model,
                 unordered_map<string, CoreMLInputDesc>* out)
{
  auto* desc = model->modelDescription();
  if (!desc) {
    return;
  }
  auto* dict = desc->inputDescriptionsByName();
  if (!dict) {
    return;
  }
  auto* keys = dict->keyEnumerator<NS::String>();
  if (!keys) {
    return;
  }
  while (auto* k = keys->nextObject()) {
    auto* fd = dict->object<CML::FeatureDescription>(k);
    if (!fd) {
      continue;
    }
    string          name(k->utf8String());
    CoreMLInputDesc d;
    if (auto* im = fd->imageConstraint()) {
      d.kind         = CoreMLFeatureKind::Image;
      d.image_width  = static_cast<int>(im->pixelsWide());
      d.image_height = static_cast<int>(im->pixelsHigh());
      d.pixel_format = im->pixelFormatType();
      d.fixed        = d.image_width > 0 && d.image_height > 0;
    } else if (auto* ma = fd->multiArrayConstraint()) {
      d.kind       = CoreMLFeatureKind::MultiArray;
      d.dtype      = from_marray_dtype_(ma->dataType());
      ShapeRead sr = read_shape_(ma->shape());
      d.shape      = std::move(sr.shape);
      d.fixed      = sr.fixed;
    } else {
      continue;   // unsupported input kind
    }
    out->emplace(std::move(name), std::move(d));
  }
}

// Collect every feature name from a modelDescription input/output
// dictionary into `out` (enumeration order; CoreML guarantees none).
// Used to derive the single input/output name of single-I/O models.
void
collect_names_(NS::Dictionary* dict, vector<string>* out)
{
  if (!dict) {
    return;
  }
  auto* keys = dict->keyEnumerator<NS::String>();
  if (!keys) {
    return;
  }
  while (auto* k = keys->nextObject()) {
    out->emplace_back(k->utf8String());
  }
}

// On-disk cache for the compiled .mlmodelc bundle CoreML produces
// from a raw .mlmodel/.mlpackage. CML::Model::compileModelAtURL
// drops the result into $TMPDIR and keeps it there until the caller
// moves or deletes it (per Apple's docs), so every load otherwise
// leaks a fresh ~600 MB-class bundle into /var/folders. We relocate
// the bundle into ~/Library/Caches/vpipe/coreml/ keyed by source
// path + source mtime; later loads short-circuit straight to the
// cached bundle.
std::filesystem::path
coreml_cache_dir_()
{
  const char* home = std::getenv("HOME");
  if (!home || !*home) {
    return {};
  }
  return std::filesystem::path(home)
       / "Library" / "Caches" / "vpipe" / "coreml";
}

std::string
coreml_cache_filename_(const std::string& canonical_path)
{
  std::error_code ec;
  auto t = std::filesystem::last_write_time(canonical_path, ec);
  long long mt = ec ? 0LL
      : static_cast<long long>(t.time_since_epoch().count());
  std::size_t h1 = std::hash<std::string>{}(canonical_path);
  std::size_t h2 = std::hash<long long>{}(mt);
  std::size_t h  = h1
      ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
  std::string stem =
      std::filesystem::path(canonical_path).stem().string();
  char hex[20];
  std::snprintf(hex, sizeof(hex), "%016llx",
                static_cast<unsigned long long>(h));
  return stem + "-" + hex + ".mlmodelc";
}

// Remove any older cached .mlmodelc bundles for the same source
// (same stem, different hash) so the cache stays one-per-source.
void
coreml_sweep_stale_cache_(const std::filesystem::path& cache_dir,
                          const std::string&           source_path,
                          const std::string&           keep_name)
{
  std::string stem =
      std::filesystem::path(source_path).stem().string();
  std::string prefix = stem + "-";
  std::string suffix = ".mlmodelc";

  std::error_code ec;
  std::filesystem::directory_iterator it(
      cache_dir,
      std::filesystem::directory_options::skip_permission_denied,
      ec);
  if (ec) {
    return;
  }
  for (const auto& entry : it) {
    const auto fname = entry.path().filename().string();
    if (fname == keep_name) {
      continue;
    }
    if (fname.size() <= prefix.size() + suffix.size()) {
      continue;
    }
    if (fname.compare(0, prefix.size(), prefix) != 0) {
      continue;
    }
    if (fname.compare(fname.size() - suffix.size(),
                      suffix.size(), suffix) != 0) {
      continue;
    }
    std::error_code rm_ec;
    std::filesystem::remove_all(entry.path(), rm_ec);
  }
}

string
canonicalize_(const SessionContextIntf* session, string_view path)
{
  // canonical() throws on missing files. We preserve the original
  // path on failure so the model load can still proceed (the load
  // itself will produce a clearer error). Sharing across two
  // textually-different but semantically-equal paths is a nice-to-
  // have, not a correctness requirement.
  try {
    auto p = filesystem::canonical(filesystem::path(string(path)));
    return p.string();
  } catch (const exception& e) {
    if (session) {
      session->log_debug(fmt(
          "CoreMLModelManager: canonical('{}') failed ({}); using "
          "raw path for cache key",
          path, e.what()));
    }
    return string(path);
  }
}

}

// ---- CoreMLLoadedModel ----------------------------------------------

CoreMLLoadedModel::CoreMLLoadedModel(const SessionContextIntf* session,
                                     string_view               path,
                                     int                       compute_units)
  : SessionMember(session)
  , _model(nullptr)
  , _path(path)
  , _compute_units(compute_units)
{
  auto* pool = NS::AutoreleasePool::alloc()->init();

  NS::String* path_ns = ns_str_(_path);
  NS::URL*    url     = NS::URL::fileURLWithPath(path_ns);

  auto ends_with = [](string_view s, string_view sfx) {
    return s.size() >= sfx.size()
        && s.compare(s.size() - sfx.size(), sfx.size(), sfx) == 0;
  };
  bool needs_compile = ends_with(_path, ".mlmodel")
                    || ends_with(_path, ".mlpackage");

  NS::Error* err = nullptr;
  if (needs_compile) {
    // Try the on-disk compile cache before falling back to a fresh
    // CoreML compile. Cache key includes the source dir's mtime, so
    // regenerating the mlpackage invalidates the entry naturally.
    std::filesystem::path cache_dir  = coreml_cache_dir_();
    std::filesystem::path cache_path = cache_dir.empty()
        ? std::filesystem::path()
        : cache_dir / coreml_cache_filename_(_path);

    std::error_code ec_exists;
    bool cache_hit = !cache_path.empty()
        && std::filesystem::exists(cache_path, ec_exists);

    if (cache_hit) {
      NS::String* cached_ns = ns_str_(cache_path.string());
      url = NS::URL::fileURLWithPath(cached_ns);
      this->session()->log_debug(fmt(
          "CoreMLLoadedModel: compile cache hit for '{}' -> '{}'",
          _path, cache_path.string()));
    } else {
      NS::URL* compiled = CML::Model::compileModelAtURL(url, &err);
      if (err || !compiled) {
        pool->release();
        this->session()->error(fmt(
            "CoreMLLoadedModel: compileModelAtURL failed for '{}'",
            _path));
        return;
      }
      url = compiled;

      // Relocate the freshly-compiled bundle from $TMPDIR into the
      // persistent cache. On any failure we leave `url` pointing at
      // the tmp bundle so the load still succeeds.
      if (!cache_path.empty()) {
        const char* tmp_cstr = compiled->fileSystemRepresentation();
        if (tmp_cstr && *tmp_cstr) {
          std::filesystem::path tmp_path(tmp_cstr);
          std::error_code ec_mk;
          std::filesystem::create_directories(cache_dir, ec_mk);
          if (ec_mk) {
            this->session()->log_debug(fmt(
                "CoreMLLoadedModel: mkdir '{}' failed ({}); leaving "
                "compiled bundle at tmp '{}'",
                cache_dir.string(), ec_mk.message(),
                tmp_path.string()));
          } else {
            std::error_code ec_mv;
            std::filesystem::rename(tmp_path, cache_path, ec_mv);
            if (!ec_mv) {
              NS::String* cached_ns = ns_str_(cache_path.string());
              url = NS::URL::fileURLWithPath(cached_ns);
              coreml_sweep_stale_cache_(
                  cache_dir, _path,
                  cache_path.filename().string());
              this->session()->log_debug(fmt(
                  "CoreMLLoadedModel: moved compiled '{}' to cache "
                  "'{}'", _path, cache_path.string()));
            } else {
              std::error_code ec_re;
              const bool target_now_exists =
                  std::filesystem::exists(cache_path, ec_re);
              if (target_now_exists) {
                std::error_code ec_rm;
                std::filesystem::remove_all(tmp_path, ec_rm);
                NS::String* cached_ns = ns_str_(cache_path.string());
                url = NS::URL::fileURLWithPath(cached_ns);
              } else {
                this->session()->log_debug(fmt(
                    "CoreMLLoadedModel: rename '{}' -> '{}' failed "
                    "({}); leaving compiled bundle at tmp",
                    tmp_path.string(), cache_path.string(),
                    ec_mv.message()));
              }
            }
          }
        }
      }
    }
  }

  auto* cfg = CML::ModelConfiguration::alloc()->init();
  cfg->setComputeUnits(static_cast<CML::ComputeUnits>(_compute_units));

  CML::Model* model =
      CML::Model::modelWithContentsOfURL(url, cfg, &err);
  cfg->release();
  if (err || !model) {
    pool->release();
    this->session()->error(fmt(
        "CoreMLLoadedModel: modelWithContentsOfURL failed for '{}'",
        _path));
    return;
  }
  _model = model->retain();

  // Discover input + output descriptors so predict() can build the
  // right features and the consuming stage can decide whether to take
  // the zero-copy outputBackings path, plus the input/output feature
  // names so a stage can derive them without config.
  discover_outputs_(_model, &_outputs);
  discover_inputs_(_model, &_inputs_desc);
  if (auto* mdesc = _model->modelDescription()) {
    collect_names_(mdesc->inputDescriptionsByName(),  &_input_names);
    collect_names_(mdesc->outputDescriptionsByName(), &_output_names);
  }

  pool->release();
}

CoreMLLoadedModel::~CoreMLLoadedModel()
{
  if (_model) {
    _model->release();
    _model = nullptr;
  }
}

bool
CoreMLLoadedModel::predict(span<const CoreMLPredictInput> inputs,
                           span<CoreMLPredictOutput>      outputs,
                           bool                           cpu_only)
{
  if (!_model) {
    return false;
  }

  auto warn = [this](const VpipeFormat& m) {
    if (this->session()) {
      this->session()->warn(m);
    }
  };

  // Resolve an empty feature name to the model's sole input/output.
  auto sole = [](const string& want, const vector<string>& names) {
    return (want.empty() && names.size() == 1) ? names[0] : want;
  };

  // Row-major contiguous element strides for a shape.
  auto contig_strides = [](const vector<int64_t>& shape) {
    vector<int64_t> s(shape.size(), 1);
    int64_t run = 1;
    for (size_t i = shape.size(); i-- > 0;) {
      s[i] = run;
      run *= shape[i];
    }
    return s;
  };
  auto elems = [](const vector<int64_t>& shape) {
    size_t n = 1;
    for (auto d : shape) {
      n *= static_cast<size_t>(d);
    }
    return n;
  };

  // Serialise per-model. CoreML serialises internally anyway; a cheap
  // observable mutex beats opaque blocking inside the framework.
  lock_guard<mutex> lk(_predict_mu);
  auto* pool = NS::AutoreleasePool::alloc()->init();

  // Manually-released objects, freed on every exit.
  vector<CVPixelBufferRef> pbs;
  vector<CML::MultiArray*> in_arrs;
  vector<CML::MultiArray*> back_arrs;
  CML::DictionaryFeatureProvider* dfp  = nullptr;
  CML::PredictionOptions*         opts = nullptr;
  auto cleanup = [&]() {
    if (dfp)  { dfp->release(); }
    if (opts) { opts->release(); }
    for (auto* a : back_arrs) { if (a) { a->release(); } }
    for (auto* a : in_arrs)   { if (a) { a->release(); } }
    for (auto pb : pbs)       { if (pb) { CFRelease(pb); } }
    pool->release();
  };

  NS::Error* err = nullptr;

  // ---- Build the input feature dictionary. ----
  const size_t              nin = inputs.size();
  vector<const NS::Object*> objs;
  vector<const NS::Object*> keys;
  objs.reserve(nin);
  keys.reserve(nin);

  for (const auto& in : inputs) {
    const string name = sole(in.name, _input_names);
    CML::FeatureValue* fv = nullptr;

    if (in.image) {
      // Image input: stage BGRA bytes into a CVPixelBuffer of the
      // caller's image size, using the model's pixel format.
      uint32_t pix_fmt = kCVPixelFormatType_32BGRA;
      auto di = _inputs_desc.find(name);
      if (di != _inputs_desc.end() && di->second.pixel_format) {
        pix_fmt = di->second.pixel_format;
      }
      CVPixelBufferRef pb = nullptr;
      CVReturn cv = CVPixelBufferCreate(
          kCFAllocatorDefault, static_cast<size_t>(in.image_width),
          static_cast<size_t>(in.image_height), pix_fmt, nullptr, &pb);
      if (cv != kCVReturnSuccess || !pb) {
        warn(fmt("CoreMLLoadedModel::predict: CVPixelBufferCreate "
                 "failed for input '{}'", name));
        cleanup();
        return false;
      }
      pbs.push_back(pb);
      if (CVPixelBufferLockBaseAddress(pb, 0) != kCVReturnSuccess) {
        warn(fmt("CoreMLLoadedModel::predict: pixel buffer lock "
                 "failed for input '{}'", name));
        cleanup();
        return false;
      }
      auto*  dst = static_cast<uint8_t*>(CVPixelBufferGetBaseAddress(pb));
      size_t bpr = CVPixelBufferGetBytesPerRow(pb);
      size_t src_row = in.image_row_bytes
          ? in.image_row_bytes
          : static_cast<size_t>(in.image_width) * 4;
      size_t row_bytes =
          std::min(bpr, static_cast<size_t>(in.image_width) * 4);
      for (int y = 0; y < in.image_height; ++y) {
        std::memcpy(dst + static_cast<size_t>(y) * bpr,
                    in.image + static_cast<size_t>(y) * src_row,
                    row_bytes);
      }
      CVPixelBufferUnlockBaseAddress(pb, 0);
      fv = CML::FeatureValue::featureValueWithPixelBuffer(pb);
    } else {
      // MultiArray input: zero-copy over the caller's borrowed buffer.
      vector<int64_t> strides =
          in.strides.empty() ? contig_strides(in.shape) : in.strides;
      NS::Array* shp = ns_num_array_(in.shape);
      NS::Array* str = ns_num_array_(strides);
      err = nullptr;
      auto* ma = CML::MultiArray::alloc()->initWithDataPointer(
          const_cast<void*>(in.data), shp, to_marray_dtype_(in.dtype),
          str, /*deallocator=*/nullptr, &err);
      if ((err || !ma) && in.strides.empty()) {
        // Fallback (contiguous inputs only): alloc + memcpy.
        err = nullptr;
        ma = CML::MultiArray::alloc()->initWithShape(
            shp, to_marray_dtype_(in.dtype), &err);
        if (!err && ma && in.data) {
          std::memcpy(ma->dataPointer(), in.data,
                      elems(in.shape) * coreml_dtype_size(in.dtype));
        }
      }
      if (err || !ma) {
        warn(fmt("CoreMLLoadedModel::predict: MLMultiArray init failed "
                 "for input '{}'", name));
        if (ma) { ma->release(); }
        cleanup();
        return false;
      }
      in_arrs.push_back(ma);
      fv = CML::FeatureValue::featureValueWithMultiArray(ma);
    }

    if (!fv) {
      warn(fmt("CoreMLLoadedModel::predict: feature value build failed "
               "for input '{}'", name));
      cleanup();
      return false;
    }
    objs.push_back(fv);
    keys.push_back(ns_str_(name));
  }

  NS::Dictionary* dict = NS::Dictionary::dictionary(
      objs.data(), keys.data(), static_cast<NS::UInteger>(nin));
  err = nullptr;
  dfp = CML::DictionaryFeatureProvider::alloc()
            ->initWithDictionary(dict, &err);
  if (err || !dfp) {
    warn(fmt("CoreMLLoadedModel::predict: feature provider init failed"));
    cleanup();
    return false;
  }

  // ---- PredictionOptions + zero-copy output backings. ----
  opts = CML::PredictionOptions::alloc()->init();
  opts->setUsesCPUOnly(cpu_only);

  vector<char> bound(outputs.size(), 0);   // outputs delivered zero-copy
  {
    vector<const NS::Object*> back_objs;
    vector<const NS::Object*> back_keys;
    for (size_t i = 0; i < outputs.size(); ++i) {
      auto& o = outputs[i];
      if (!o.backing) {
        continue;
      }
      const string oname = sole(o.name, _output_names);
      auto od = _outputs.find(oname);
      // Bind only when the shape is fully specified and the model's
      // native dtype matches what the caller wants (else CoreML would
      // write the wrong byte layout into the backing).
      if (od == _outputs.end() || !od->second.fixed
          || od->second.dtype != o.want) {
        continue;
      }
      const size_t n = elems(od->second.shape);
      if (o.backing_elems && o.backing_elems < n) {
        continue;
      }
      NS::Array* shp = ns_num_array_(od->second.shape);
      NS::Array* str = ns_num_array_(contig_strides(od->second.shape));
      err = nullptr;
      auto* ba = CML::MultiArray::alloc()->initWithDataPointer(
          o.backing, shp, to_marray_dtype_(o.want), str,
          /*deallocator=*/nullptr, &err);
      if (err || !ba) {
        if (ba) { ba->release(); }
        continue;   // fall back to copy-out below
      }
      back_arrs.push_back(ba);
      back_objs.push_back(ba);
      back_keys.push_back(ns_str_(oname));
      o.shape  = od->second.shape;
      o.data   = o.backing;
      bound[i] = 1;
    }
    if (!back_objs.empty()) {
      NS::Dictionary* bdict = NS::Dictionary::dictionary(
          back_objs.data(), back_keys.data(),
          static_cast<NS::UInteger>(back_objs.size()));
      opts->setOutputBackings(bdict);
    }
  }

  // ---- Predict. ----
  err = nullptr;
  auto* result = _model->predictionFromFeatures(dfp, opts, &err);
  if (err || !result) {
    warn(fmt("CoreMLLoadedModel::predict: predictionFromFeatures failed"));
    cleanup();
    return false;
  }

  // ---- Read outputs not already delivered via a zero-copy backing. ----
  bool ok = true;
  for (size_t i = 0; i < outputs.size() && ok; ++i) {
    auto& o = outputs[i];
    if (bound[i]) {
      continue;   // already in o.backing
    }
    const string oname = sole(o.name, _output_names);
    auto* val = result->featureValueForName(ns_str_(oname));
    auto* arr = val ? val->multiArrayValue() : nullptr;
    if (!arr) {
      warn(fmt("CoreMLLoadedModel::predict: output '{}' missing or not "
               "a MultiArray", oname));
      ok = false;
      break;
    }
    o.shape        = read_shape_(arr->shape()).shape;
    const size_t n = elems(o.shape);

    // Element strides of the result; decide contiguous vs strided read.
    vector<int64_t> ostr = read_shape_(arr->strides()).shape;
    bool contig = true;
    if (!ostr.empty() && ostr.size() == o.shape.size()) {
      int64_t run = 1;
      for (size_t k = o.shape.size(); k-- > 0;) {
        if (ostr[k] != run) { contig = false; break; }
        run *= o.shape[k];
      }
    }

    const CML::MultiArrayDataType dt  = arr->dataType();
    const void*                   src = arr->dataPointer();

    // Destination: caller backing when it fits, else the owned buffer.
    void* dstv;
    if (o.backing && (o.backing_elems == 0 || o.backing_elems >= n)) {
      dstv   = o.backing;
      o.data = o.backing;
    } else {
      o.owned.assign(n * coreml_dtype_size(o.want), 0);
      dstv   = o.owned.data();
      o.data = o.owned.data();
    }

    if (contig && dtype_same_(dt, o.want)) {
      std::memcpy(dstv, src, n * coreml_dtype_size(o.want));
    } else if (contig) {
      for (size_t e = 0; e < n; ++e) {
        write_elem_(dstv, o.want, e, read_elem_(src, dt, e));
      }
    } else {
      // Strided gather over o.shape using the reported element strides.
      vector<int64_t> idx(o.shape.size(), 0);
      for (size_t e = 0; e < n; ++e) {
        size_t off = 0;
        for (size_t d = 0; d < o.shape.size(); ++d) {
          off += static_cast<size_t>(idx[d])
               * static_cast<size_t>(ostr[d]);
        }
        write_elem_(dstv, o.want, e, read_elem_(src, dt, off));
        for (size_t d = o.shape.size(); d-- > 0;) {
          if (++idx[d] < o.shape[d]) { break; }
          idx[d] = 0;
        }
      }
    }
  }

  cleanup();
  return ok;
}

// ---- CoreMLModelManager ---------------------------------------------

size_t
CoreMLModelManager::KeyHash::operator()(const Key& k) const noexcept
{
  size_t h = hash<string>{}(k.path);
  h ^= hash<int>{}(k.compute_units) + 0x9e3779b9 + (h << 6) + (h >> 2);
  return h;
}

CoreMLModelManager::CoreMLModelManager(const SessionContextIntf* session)
  : SessionMember(session)
{
}

shared_ptr<CoreMLLoadedModel>
CoreMLModelManager::load(string_view path, int compute_units)
{
  Key key{ canonicalize_(session(), path), compute_units };

  // Fast path: existing live entry.
  {
    lock_guard<mutex> lk(_mu);
    auto it = _cache.find(key);
    if (it != _cache.end()) {
      if (auto sp = it->second.lock()) {
        return sp;
      }
      _cache.erase(it);
    }
  }

  // Miss: load outside the cache lock so concurrent loads of
  // different keys parallelise. session->error() inside the
  // CoreMLLoadedModel ctor throws; convert to a null return + log.
  shared_ptr<CoreMLLoadedModel> sp;
  try {
    sp = make_shared<CoreMLLoadedModel>(
        session(), key.path, key.compute_units);
  } catch (const exception& e) {
    if (session()) {
      session()->warn(fmt(
          "CoreMLModelManager::load('{}', compute_units={}): {}",
          key.path, key.compute_units, e.what()));
    }
    return nullptr;
  }
  if (!sp || !sp->valid()) {
    return nullptr;
  }

  // Re-acquire the lock and stash a weak ref. If a racing thread
  // already populated the entry with a live model, prefer the
  // existing one (returns the same shape but avoids parallel-load
  // duplication for a heartbeat). Either is correct.
  {
    lock_guard<mutex> lk(_mu);
    auto it = _cache.find(key);
    if (it != _cache.end()) {
      if (auto existing = it->second.lock()) {
        return existing;
      }
    }
    _cache[key] = sp;
  }
  return sp;
}

size_t
CoreMLModelManager::cached_count() const
{
  lock_guard<mutex> lk(_mu);
  size_t n = 0;
  for (const auto& kv : _cache) {
    if (!kv.second.expired()) {
      ++n;
    }
  }
  return n;
}

}

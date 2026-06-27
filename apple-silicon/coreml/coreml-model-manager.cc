#include "apple-silicon/coreml/coreml-model-manager.h"

#include "apple-silicon/coreml/coreml-cpp/CoreML.hpp"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

// Walk modelDescription.outputDescriptionsByName() and capture each
// MLMultiArray output's shape. Non-MultiArray outputs (image, dict,
// sequence) end up missing from the map and the stage falls back to
// the legacy memcpy path for them.
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
                 CoreMLOutputDesc{ std::move(sr.shape), sr.fixed });
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

  // Discover output descriptors so the consuming stage can decide
  // whether to take the zero-copy outputBackings path, plus the input/
  // output feature names so a stage can derive them without config.
  discover_outputs_(_model, &_outputs);
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
  if (!sp || !sp->model()) {
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

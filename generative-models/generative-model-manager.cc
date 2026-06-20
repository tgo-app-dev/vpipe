#include "generative-models/generative-model-manager.h"

#include "generative-models/shared/gguf-convert.h"
#include "generative-models/shared/gguf-file.h"
#include "generative-models/loaded-language-model.h"
#include "generative-models/model-loader.h"
#include "generative-models/tokenizer.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <chrono>
#include <exception>
#include <filesystem>
#include <functional>
#include <thread>
#include <utility>

using namespace std;

namespace vpipe::genai {

namespace {

// Map LoadSpec::compute_dtype string to the MLX-free ComputeDtype.
// Returns false on an unknown value; the loader logs and falls back to
// bfloat16 (Llama 3.x's training dtype).
bool
parse_compute_dtype_(const string& s, ComputeDtype* out)
{
  if (s == "bf16" || s == "bfloat16") {
    *out = ComputeDtype::BF16;
    return true;
  }
  if (s == "f16" || s == "float16") {
    *out = ComputeDtype::F16;
    return true;
  }
  if (s == "f32" || s == "float32") {
    *out = ComputeDtype::F32;
    return true;
  }
  return false;
}

// canonical()-with-fallback so two textually-different paths to the
// same model dedup. canonical() throws on missing files; on failure
// we keep the raw path -- the subsequent ModelLoader::load() will
// produce a clearer "config.json missing" warning.
string
canonicalize_(const SessionContextIntf* session, const string& path)
{
  try {
    return filesystem::canonical(filesystem::path(path)).string();
  } catch (const exception& e) {
    if (session) {
      session->log_debug(fmt(
          "GenerativeModelManager: canonical('{}') failed ({}); using "
          "raw path for cache key", path, e.what()));
    }
    return path;
  }
}

}

size_t
GenerativeModelManager::KeyHash::operator()(const Key& k) const noexcept
{
  size_t h = hash<string>{}(k.hf_dir);
  auto mix = [&h](size_t x) {
    h ^= x + 0x9e3779b9 + (h << 6) + (h >> 2);
  };
  mix(hash<string>{}(k.compute_dtype));
  mix(hash<int>{}(k.page_tokens));
  mix(hash<uint32_t>{}(k.max_pages));
  return h;
}

GenerativeModelManager::GenerativeModelManager(const SessionContextIntf* s)
  : SessionMember(s)
{
  // No work here. Every MLX op the manager performs (load, dequantize,
  // forward pass, free) is routed through Session::mlx_runtime() so
  // it runs on the dedicated MLX thread, where the (Stream, encoder)
  // TLS map is primed once and stays warm.
}

shared_ptr<LoadedLanguageModel>
GenerativeModelManager::load(const LoadSpec& spec)
{
  Key key{ canonicalize_(session(), spec.hf_dir),
           spec.compute_dtype,
           spec.page_tokens,
           spec.max_pages };

  // Fast path: existing live entry. Locked briefly; the lookup is
  // O(1) and weak_ptr::lock() is a couple of atomics.
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

  // Map the dtype string. We do this BEFORE the heavy IO so a typo
  // fails fast.
  ComputeDtype compute_dtype{ComputeDtype::BF16};
  if (!parse_compute_dtype_(spec.compute_dtype, &compute_dtype)) {
    if (session()) {
      session()->warn(fmt(
          "GenerativeModelManager::load('{}'): unknown compute_dtype "
          "'{}'; expected one of bf16 / f16 / f32",
          key.hf_dir, spec.compute_dtype));
    }
    return nullptr;
  }

  // The metal-compute LM runs inline on the calling thread
  // (LoadedLanguageModel::dispatch_ is a no-op when no runtime is
  // bound). Kept as an explicit null so the LM ctor's runtime
  // parameter has an obvious source.
  MlxRuntime* rt = nullptr;
  const string dir = key.hf_dir;
  const int    page_tokens = key.page_tokens;
  const uint32_t max_pages = key.max_pages;
  using ProfClock = std::chrono::steady_clock;
  const bool profile_load = std::getenv("VPIPE_LOAD_PROFILE") != nullptr;
  auto build =
      [this, &dir, compute_dtype, page_tokens, max_pages, rt,
       profile_load]
      () -> shared_ptr<LoadedLanguageModel> {
        auto t0 = ProfClock::now();
        ModelLoader loader(session());
        auto loaded = loader.load(dir);
        if (!loaded) {
          return nullptr;
        }
        auto t1 = ProfClock::now();
        if (profile_load && session()) {
          session()->info(fmt(
              "[load-profile] ModelLoader::load: {:.1f} ms",
              std::chrono::duration<double, std::milli>(t1 - t0).count()));
        }
        // Tokenizer: HF tokenizer.json when present, else (pure-GGUF dirs)
        // reconstruct it from the GGUF's embedded vocab/merges.
        filesystem::path tok_path =
            filesystem::path(dir) / "tokenizer.json";
        std::unique_ptr<Tokenizer> tok;
        if (filesystem::exists(tok_path)) {
          tok = Tokenizer::from_huggingface_json(
              tok_path.string(), session());
        } else if (std::string gp = find_gguf_in_dir(dir); !gp.empty()) {
          if (auto gf = GgufFile::open(gp)) {
            tok = Tokenizer::from_gguf(*gf, session());
          }
        }
        if (!tok) {
          return nullptr;
        }
        auto t2 = ProfClock::now();
        if (profile_load && session()) {
          session()->info(fmt(
              "[load-profile] Tokenizer::from_huggingface_json: {:.1f} ms",
              std::chrono::duration<double, std::milli>(t2 - t1).count()));
        }
        // `rt` is always null; the LM ctor's dispatch_() runs inline
        // on this thread.
        auto built = make_shared<LoadedLanguageModel>(
            std::move(*loaded), std::move(tok),
            compute_dtype, page_tokens, max_pages,
            rt, session(), std::string(dir));
        if (!built->valid()) {
          return nullptr;
        }
        auto t3 = ProfClock::now();
        if (profile_load && session()) {
          session()->info(fmt(
              "[load-profile] LoadedLanguageModel ctor total: {:.1f} ms",
              std::chrono::duration<double, std::milli>(t3 - t2).count()));
        }
        return built;
      };
  // Metal-only: run inline -- the metal backend loads its own weights
  // and the LM's dispatch_() runs inline when runtime == nullptr.
  shared_ptr<LoadedLanguageModel> lm = build();
  if (!lm) {
    if (session()) {
      session()->warn(fmt(
          "GenerativeModelManager::load('{}'): load failed "
          "(see prior warns)", key.hf_dir));
    }
    return nullptr;
  }

  // Re-lock and stash the result. If a racing thread already
  // installed an entry for the same key with a live shared_ptr we
  // hand back theirs (wastes one load, never produces UB; this is
  // the same trade-off CoreMLModelManager makes).
  {
    lock_guard<mutex> lk(_mu);
    auto it = _cache.find(key);
    if (it != _cache.end()) {
      if (auto existing = it->second.lock()) {
        return existing;
      }
    }
    _cache[key] = lm;
  }
  return lm;
}

size_t
GenerativeModelManager::cached_count() const
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

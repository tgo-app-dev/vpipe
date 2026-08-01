#include "generative-models/generative-model-manager.h"

#include "generative-models/shared/gguf-convert.h"
#include "generative-models/shared/gguf-file.h"
#include "generative-models/loaded-language-model.h"
#include "generative-models/model-loader.h"
#include "generative-models/tokenizer.h"
#include "generative-models/weight-set.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <chrono>
#include <exception>
#include <filesystem>
#include <functional>
#include <thread>
#include <algorithm>
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
std::string
GenerativeModelManager::shared_key_(const string& kind,
                                    const string& dir,
                                    const string& variant) const
{
  // Canonicalize the directory for the same reason the LM cache does:
  // two textually-different paths to one model must dedup.
  return kind + "|" + canonicalize_(session(), dir) + "|" + variant;
}

std::shared_ptr<void>
GenerativeModelManager::lookup_shared_(const string& key)
{
  lock_guard<mutex> lk(_shared_mu);
  auto it = _shared.find(key);
  if (it == _shared.end()) { return nullptr; }
  if (auto sp = it->second.lock()) { return sp; }
  _shared.erase(it);            // last holder is gone
  return nullptr;
}

std::shared_ptr<void>
GenerativeModelManager::store_shared_(const string&                key,
                                      const std::shared_ptr<void>& v)
{
  lock_guard<mutex> lk(_shared_mu);
  auto it = _shared.find(key);
  if (it != _shared.end()) {
    if (auto sp = it->second.lock()) {
      return sp;                // someone else won; caller uses theirs
    }
  }
  _shared[key] = v;
  return nullptr;
}

std::shared_ptr<WeightSet>
GenerativeModelManager::weight_set(const string& dir, const string& variant)
{
  if (dir.empty()) { return nullptr; }
  const string key = canonicalize_(session(), dir) + "|" + variant;

  {
    lock_guard<mutex> lk(_ws_mu);
    auto it = _weight_sets.find(key);
    if (it != _weight_sets.end()) {
      if (auto sp = it->second.lock()) { return sp; }
      _weight_sets.erase(it);
    }
  }

  // Opened OUTSIDE the lock: mapping a checkpoint touches the
  // filesystem, and two callers naming DIFFERENT models must not
  // serialise behind each other.
  shared_ptr<WeightSet> ws = WeightSet::open(dir, session());
  if (!ws) { return nullptr; }

  {
    lock_guard<mutex> lk(_ws_mu);
    auto it = _weight_sets.find(key);
    if (it != _weight_sets.end()) {
      // Someone else won the race. Hand back THEIR set and drop ours,
      // so everyone converges on one copy of the weights -- the whole
      // point. Ours has nothing materialised yet, so nothing is lost.
      if (auto sp = it->second.lock()) { return sp; }
    }
    _weight_sets[key] = ws;
  }
  // Registered after publication so the residency policy sees it, and
  // held by the set itself so it drops when the set does.
  ws->set_registration(_weights.add(ws.get()), &_weights);
  if (session() != nullptr) {
    session()->log_debug(fmt(
        "GenerativeModelManager: opened weight set '{}'{}", key,
        variant.empty() ? "" : " (variant)"));
  }
  // A new checkpoint is the moment the total can cross the cap.
  enforce_memory_cap();
  return ws;
}

bool
GenerativeModelManager::holds_weights(const string& dir) const
{
  if (dir.empty()) { return false; }
  const string pfx = canonicalize_(session(), dir) + "|";
  lock_guard<mutex> lk(_ws_mu);
  for (const auto& [k, w] : _weight_sets) {
    if (k.rfind(pfx, 0) == 0 && !w.expired()) { return true; }
  }
  return false;
}

void
GenerativeModelManager::declare_weights(const string& dir,
                                        std::size_t   expected_bytes)
{
  if (dir.empty()) { return; }
  const string key = canonicalize_(session(), dir);
  lock_guard<mutex> lk(_ws_mu);
  std::size_t& e = _declared[key];
  // Two stages naming one checkpoint declare it once, at the larger
  // estimate -- they are describing the same bytes, not two copies.
  if (expected_bytes > e) { e = expected_bytes; }
}

void
GenerativeModelManager::revise_declaration(const string& dir,
                                           std::size_t   bytes)
{
  if (dir.empty()) { return; }
  const string key = canonicalize_(session(), dir);
  lock_guard<mutex> lk(_ws_mu);
  auto it = _declared.find(key);
  // Only a checkpoint that was declared can be revised: an undeclared
  // one is not being counted from an estimate, so there is nothing to
  // correct.
  if (it != _declared.end()) { it->second = bytes; }
}

void
GenerativeModelManager::clear_declarations()
{
  lock_guard<mutex> lk(_ws_mu);
  _declared.clear();
}

bool
GenerativeModelManager::accounts_for(const string& dir) const
{
  if (dir.empty()) { return false; }
  const string canon = canonicalize_(session(), dir);
  const string pfx   = canon + "|";
  lock_guard<mutex> lk(_ws_mu);
  if (_declared.find(canon) != _declared.end()) { return true; }
  for (const auto& [k, w] : _weight_sets) {
    if (k.rfind(pfx, 0) == 0 && !w.expired()) { return true; }
  }
  return false;
}

std::size_t
GenerativeModelManager::resident_weight_bytes() const
{
  // Snapshot under the lock, measure outside it -- stats() takes each
  // set's own mutex and holding this one across that would put the
  // query in the way of a concurrent load.
  std::vector<std::pair<string, std::shared_ptr<WeightSet>>> live;
  std::unordered_map<string, std::size_t> declared;
  {
    lock_guard<mutex> lk(_ws_mu);
    live.reserve(_weight_sets.size());
    for (const auto& [k, w] : _weight_sets) {
      if (auto sp = w.lock()) {
        // Key back to the plain directory: two variants of one
        // checkpoint are two sets but must be summed under one dir so
        // a declaration for it compares against the whole.
        const auto bar = k.rfind('|');
        live.emplace_back(bar == string::npos ? k : k.substr(0, bar),
                          std::move(sp));
      }
    }
    declared = _declared;
  }

  std::unordered_map<string, std::size_t> per_dir;
  for (const auto& [dir, ws] : live) { per_dir[dir] += ws->stats().bytes; }
  // A declared checkpoint counts at the larger of what it HOLDS and what
  // it was estimated to need -- see declare_weights() for why the max
  // matters while a load is still in flight.
  for (const auto& [dir, want] : declared) {
    std::size_t& have = per_dir[dir];
    if (want > have) { have = want; }
  }
  std::size_t total = 0;
  for (const auto& [dir, b] : per_dir) { (void)dir; total += b; }
  return total;
}

std::size_t
GenerativeModelManager::resident_kv_bytes() const
{
  // Snapshot the live models under the lock, then measure outside it:
  // kv_bytes() takes each context manager's own mutex, and a decode in
  // flight holds that.
  std::vector<std::shared_ptr<LoadedLanguageModel>> live;
  {
    lock_guard<mutex> lk(_mu);
    live.reserve(_cache.size());
    for (const auto& [k, w] : _cache) {
      (void)k;
      if (auto sp = w.lock()) { live.push_back(std::move(sp)); }
    }
  }
  std::size_t total = 0;
  for (const auto& lm : live) { total += lm->kv_bytes(); }
  return total;
}

std::size_t
GenerativeModelManager::resident_bytes() const
{
  return resident_weight_bytes() + resident_kv_bytes();
}

void
GenerativeModelManager::set_memory_cap(std::size_t bytes)
{
  _memory_cap.store(bytes, memory_order_relaxed);
  if (session() != nullptr) {
    if (bytes == 0) {
      session()->log_debug(fmt(
          "GenerativeModelManager: memory cap removed"));
    } else {
      session()->info(fmt(
          "GenerativeModelManager: memory cap {} MB (weights + KV kept "
          "actively resident; over it, least-recently-used weights are "
          "parked rather than loads refused)", bytes >> 20));
    }
  }
  enforce_memory_cap();
}

std::size_t
GenerativeModelManager::memory_cap() const
{
  return _memory_cap.load(memory_order_relaxed);
}

std::size_t
GenerativeModelManager::active_bytes() const
{
  std::vector<std::shared_ptr<WeightSet>> live;
  {
    lock_guard<mutex> lk(_ws_mu);
    live.reserve(_weight_sets.size());
    for (const auto& [k, w] : _weight_sets) {
      (void)k;
      if (auto sp = w.lock()) { live.push_back(std::move(sp)); }
    }
  }
  std::size_t total = 0;
  for (const auto& ws : live) {
    // Parked bytes belong to the kernel now; they are exactly what the
    // cap was trying to give back, so counting them would make the
    // policy chase a number it can never reach.
    if (!ws->parked()) { total += ws->stats().bytes; }
  }
  return total + resident_kv_bytes();
}

std::size_t
GenerativeModelManager::enforce_memory_cap()
{
  const std::size_t cap = memory_cap();
  if (cap == 0) { return 0; }
  std::size_t active = active_bytes();
  if (active <= cap) { return 0; }

  // Snapshot the parkable sets, least-recently-used first. A set nobody
  // has touched for a while goes before one in active use, so the
  // common case is that the cap costs nothing: the parked pages are
  // never reclaimed and the next access is a state flip.
  std::vector<std::shared_ptr<WeightSet>> live;
  {
    lock_guard<mutex> lk(_ws_mu);
    live.reserve(_weight_sets.size());
    for (const auto& [k, w] : _weight_sets) {
      (void)k;
      if (auto sp = w.lock()) {
        if (!sp->parked()) { live.push_back(std::move(sp)); }
      }
    }
  }
  std::sort(live.begin(), live.end(),
            [](const std::shared_ptr<WeightSet>& a,
               const std::shared_ptr<WeightSet>& b) {
              return a->last_use() < b->last_use();
            });

  std::size_t freed = 0;
  for (const auto& ws : live) {
    if (active <= cap) { break; }
    const std::size_t got = _weights.park(ws.get());
    if (got == 0) { continue; }        // nothing parkable (all mapped)
    ws->set_parked(true);
    freed  += got;
    active -= (got > active) ? active : got;
    if (session() != nullptr) {
      session()->log_debug(fmt(
          "GenerativeModelManager: parked {} MB of '{}' (cap {} MB, active "
          "now ~{} MB)", got >> 20, ws->dir(), cap >> 20, active >> 20));
    }
  }
  if (freed > 0 && active > cap && session() != nullptr) {
    session()->warn(fmt(
        "GenerativeModelManager: still ~{} MB active after parking "
        "everything parkable (cap {} MB). Mapped weights and KV cannot be "
        "parked -- the cap is a target, not a guarantee",
        active >> 20, cap >> 20));
  }
  return freed;
}

std::vector<GenerativeModelManager::WeightUsage>
GenerativeModelManager::weight_report() const
{
  // Snapshot the live sets under the lock, then measure OUTSIDE it: a
  // set's stats() takes its own mutex, and holding the cache lock
  // across that would put the report in the way of a concurrent load.
  std::vector<std::shared_ptr<WeightSet>> live;
  {
    lock_guard<mutex> lk(_ws_mu);
    live.reserve(_weight_sets.size());
    for (const auto& [k, w] : _weight_sets) {
      (void)k;
      if (auto sp = w.lock()) { live.push_back(std::move(sp)); }
    }
  }
  std::vector<WeightUsage> out;
  out.reserve(live.size());
  for (const auto& ws : live) {
    const auto st = ws->stats();
    WeightUsage u;
    u.dir          = ws->dir();
    u.bytes        = st.bytes;
    u.mapped_bytes = st.mapped_bytes;
    u.copied_bytes = st.copied_bytes;
    u.tensors      = st.entries;
    u.parts        = st.parts;
    // Minus the reference this snapshot itself holds, so the number
    // reads as "models using it", which is what a caller means.
    u.holders      = ws.use_count() - 1;
    out.push_back(std::move(u));
  }
  return out;
}

std::size_t
GenerativeModelManager::weight_set_count() const
{
  lock_guard<mutex> lk(_ws_mu);
  std::size_t n = 0;
  for (const auto& [k, w] : _weight_sets) {
    (void)k;
    if (!w.expired()) { ++n; }
  }
  return n;
}

std::size_t
GenerativeModelManager::shared_count() const
{
  lock_guard<mutex> lk(_shared_mu);
  std::size_t n = 0;
  for (const auto& [k, w] : _shared) {
    (void)k;
    if (!w.expired()) { ++n; }
  }
  return n;
}


GenerativeModelManager::GenerativeModelManager(const SessionContextIntf* s)
  : SessionMember(s)
  , _weights(s)
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

#include "generative-models/generative-model-manager.h"

#include "generative-models/shared/gguf-convert.h"
#include "generative-models/shared/gguf-file.h"
#include "generative-models/loaded-language-model.h"
#include "generative-models/model-loader.h"
#include "generative-models/tokenizer.h"
#include "generative-models/weight-set.h"
#include "common/vpipe-format.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"

#include <sys/sysctl.h>

#include <cerrno>
#include <cstring>

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

  // Anything nobody borrows any more is settled first, so a set this
  // call is about to hand out is never sitting in the released state
  // while a model reads it.
  release_unheld_();

  {
    lock_guard<mutex> lk(_ws_mu);
    auto it = _weight_sets.find(key);
    if (it != _weight_sets.end()) {
      shared_ptr<WeightSet> sp = it->second;
      if (session() != nullptr && sp->parked()) {
        // NOT "no reload". A set the manager kept while nobody borrowed
        // it is PARKED, so what this saves for certain is re-OPENING the
        // checkpoint. The pages come back on the first read -- free if
        // the kernel left them alone, a re-read if it did not. Claiming
        // otherwise is how a reader learns to trust bytes nobody
        // promised.
        session()->log_debug(fmt(
            "GenerativeModelManager: '{}' borrowed again ({} MB, parked -- "
            "reactivated on first read)", key, sp->stats().bytes >> 20));
      }
      return sp;
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
      return it->second;
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
    if (k.rfind(pfx, 0) == 0 && w != nullptr) { return true; }
  }
  return false;
}

namespace {

// Believed physical RAM, honouring VPIPE_RAM_LIMIT_MB exactly as
// model_memory::phys_ram() and stream-sizing.h do -- the pool has to agree
// with the planning phase about the size of the box, or a graph that
// planned against a simulated 16 GB machine would wire against a real
// 64 GB one.
std::size_t
pool_physical_ram_()
{
  if (const char* e = std::getenv("VPIPE_RAM_LIMIT_MB")) {
    const long long mb = std::atoll(e);
    if (mb > 0) { return (std::size_t)mb << 20; }
  }
  std::uint64_t mem = 0;
  std::size_t len = sizeof(mem);
  if (::sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) != 0) { return 0; }
  return (std::size_t)mem;
}

}  // namespace

void
GenerativeModelManager::set_wired_pool_pct(int pct)
{
  _pool_pct.store(pct < 0 ? 0 : (pct > 95 ? 95 : pct),
                  std::memory_order_relaxed);
  // The absolute ask goes with it: setting one is choosing which of the
  // two is in force, and leaving the other standing would make the
  // reported limit depend on the order they were set in.
  _pool_bytes.store(0, std::memory_order_relaxed);
  // A raised ask deserves a fresh try. `_pool_granted` records that the
  // box refused once, and keeping it would make every later increase
  // silently inert -- including one an operator just typed.
  _pool_granted.store(0, std::memory_order_relaxed);
}

void
GenerativeModelManager::set_wired_pool_bytes(std::size_t bytes)
{
  _pool_bytes.store(bytes, std::memory_order_relaxed);
  _pool_granted.store(0, std::memory_order_relaxed);
}

void
GenerativeModelManager::reopen_wired_pool()
{
  // Only the CEILING is reset. `_pool_used` is what is really wired and
  // must not move, and the said-once flags stay set on purpose -- a box
  // that keeps refusing would otherwise warn once per forward, and the
  // first warning is the informative one.
  _pool_granted.store(0, std::memory_order_relaxed);
}

std::size_t
GenerativeModelManager::wired_pool_bytes() const
{
  return _pool_bytes.load(std::memory_order_relaxed);
}

std::size_t
GenerativeModelManager::wired_pool_device_max() const
{
  const std::size_t cached = _pool_devmax.load(std::memory_order_relaxed);
  if (cached > 0) { return cached; }
  const auto* sess = session();
  if (sess == nullptr || sess->services() == nullptr) { return 0; }
  auto* mc = sess->services()->metal_compute();
  if (mc == nullptr || !mc->valid()) { return 0; }
  const std::size_t rec = mc->memory_budget().recommended;
  if (rec > 0) { _pool_devmax.store(rec, std::memory_order_relaxed); }
  return rec;
}

GenerativeModelManager::~GenerativeModelManager()
{
  // See the declaration: the sets must be released while the registry
  // they deregister from is still alive.
  std::unordered_map<string, shared_ptr<WeightSet>> sets;
  {
    lock_guard<mutex> lk(_ws_mu);
    sets.swap(_weight_sets);
  }
  sets.clear();
}

// How many BORROWERS a set has: everything holding it except this
// manager's own reference.
//
// Racy by construction -- a peer may take a reference the instant after
// this reads -- and that is tolerable in exactly one direction. Reading
// one too many defers a park, which costs memory the box may want.
// Reading one too FEW parks a set somebody is using, and the reader is
// then handed pages the kernel is free to discard. So every caller here
// treats a nonzero answer as "in use" and never rounds it down.
static long
borrowers_(const shared_ptr<WeightSet>& ws)
{
  const long n = (long)ws.use_count() - 1;   // minus the manager's own
  return n < 0 ? 0 : n;
}

// THE BORROW RULE, in the one place every park in this file goes
// through: a set anything is borrowing is never parked.
//
// Parking makes pages purgeable, and the reader that loses them is the
// one holding cached SharedBuffer handles -- aliases of the very buffers
// the set would reactivate, read in a forward pass that never asks the
// set for anything and so never triggers the reactivation. There is no
// way to make that safe from here: the manager cannot know a live model
// is idle, and the model has no hook that fires before it reads. So the
// answer is not to.
//
// `borrowers` is passed IN rather than derived here, and that is
// load-bearing. use_count() is only true before the caller has taken a
// copy of its own; every site computes it under _ws_mu, in the same
// walk that selects the set, and hands it over. Deriving it inside this
// function would see the caller's temporary and report one borrower too
// many for every set -- which is to say it would park nothing, ever.
std::size_t
GenerativeModelManager::park_if_unborrowed_(const shared_ptr<WeightSet>& ws,
                                            long borrowers)
{
  if (ws == nullptr || ws->parked() || borrowers > 0) { return 0; }
  return _weights.park(ws.get());
}

std::size_t
GenerativeModelManager::release_unheld_()
{
  std::vector<shared_ptr<WeightSet>> to_park;
  std::vector<pair<string, shared_ptr<WeightSet>>> to_drop;
  {
    lock_guard<mutex> lk(_ws_mu);
    for (auto it = _weight_sets.begin(); it != _weight_sets.end();) {
      const shared_ptr<WeightSet>& ws = it->second;
      if (ws == nullptr || borrowers_(ws) > 0) { ++it; continue; }
      if (!ws->recyclable()) {
        // A set SPECIALISED to a run's parameters must not outlive the
        // run: handing it to a launch that does not share them gives
        // that launch weights which are silently wrong for it. Keeping
        // it is the one thing worse than re-reading it.
        to_drop.emplace_back(it->first, std::move(it->second));
        it = _weight_sets.erase(it);
        continue;
      }
      if (!ws->parked()) { to_park.push_back(ws); }
      ++it;
    }
  }
  // BOTH outside the lock. Parking walks the set's buffers and takes the
  // registry's mutex; dropping the last reference unmaps a checkpoint.
  // Holding _ws_mu across either would put every other model's open
  // behind it.
  std::size_t parked = 0;
  for (const shared_ptr<WeightSet>& ws : to_park) {
    // 0 borrowers by construction -- that is what put it in this list,
    // measured under the lock before `to_park` took its own reference.
    parked += park_if_unborrowed_(ws, 0);
  }
  if (session() != nullptr) {
    if (parked > 0) {
      session()->log_debug(fmt(
          "GenerativeModelManager: parked {} MB across {} checkpoint(s) "
          "nothing is borrowing (kept, purgeable, reactivated on the next "
          "read)", parked >> 20, to_park.size()));
    }
    for (const auto& [k, ws] : to_drop) {
      session()->log_debug(fmt(
          "GenerativeModelManager: dropped '{}' rather than keeping it -- "
          "not recyclable ({})", k,
          ws->unrecyclable_reason().empty() ? std::string("unstated")
                                            : ws->unrecyclable_reason()));
    }
  }
  to_drop.clear();
  return parked;
}

void
GenerativeModelManager::pool_weights(const string& dir, const string& variant)
{
  if (dir.empty()) { return; }
  // NOTHING TO HAND OVER any more: the manager already owns every set it
  // opened, so a stage that is finished has only to drop its own
  // reference. What this call still does -- and the reason it is worth
  // keeping at the call sites -- is ask the manager to SETTLE the
  // checkpoint now rather than at whatever unrelated moment happens to
  // reach release_unheld_() next.
  //
  // It is therefore safe in either order. The old contract required
  // pooling BEFORE the models were reset, because the manager's
  // reference was weak and there was nothing left to pool afterwards;
  // called before, this now finds the caller's own models still
  // borrowing and settles nothing, and the next manager call settles it
  // instead. Called AFTER, it settles immediately. Both are correct;
  // only the timing differs.
  (void)variant;
  release_unheld_();
}

std::size_t
GenerativeModelManager::drop_weights(const string& dir)
{
  if (dir.empty()) { return 0; }
  // THE BYTES BACK NOW, which is the one thing release_unheld_() will
  // not do for a recyclable checkpoint. Parking makes pages reclaimable;
  // it does not return them, and a caller freeing a text encoder so a
  // 1024px VAE decode fits needs them returned. `unload_when_idle:
  // destroy` is that caller, and it meant this before the manager owned
  // anything -- dropping the last reference used to unmap the
  // checkpoint, and now it does not.
  //
  // Still refuses a BORROWED set, for the same reason park does: one
  // stage does not get to unmap a checkpoint a peer is reading. A
  // borrowed one is left alone and settles when the last borrower goes.
  const string pfx = canonicalize_(session(), dir) + "|";
  std::vector<pair<string, shared_ptr<WeightSet>>> dropped;
  std::size_t freed = 0;
  {
    lock_guard<mutex> lk(_ws_mu);
    for (auto it = _weight_sets.begin(); it != _weight_sets.end();) {
      if (it->first.rfind(pfx, 0) != 0 || it->second == nullptr ||
          borrowers_(it->second) > 0) {
        ++it;
        continue;
      }
      freed += it->second->stats().bytes;
      dropped.emplace_back(it->first, std::move(it->second));
      it = _weight_sets.erase(it);
    }
  }
  // Outside the lock: the last reference going unmaps a checkpoint.
  if (!dropped.empty() && session() != nullptr) {
    session()->log_debug(fmt(
        "GenerativeModelManager: dropped {} checkpoint(s) under '{}', {} MB "
        "-- asked for outright, not parked", dropped.size(), dir,
        freed >> 20));
  }
  dropped.clear();
  return freed;
}

std::size_t
GenerativeModelManager::pooled_bytes() const
{
  std::vector<shared_ptr<WeightSet>> spare;
  {
    lock_guard<mutex> lk(_ws_mu);
    for (const auto& [k, w] : _weight_sets) {
      (void)k;
      if (w != nullptr && borrowers_(w) == 0) { spare.push_back(w); }
    }
  }
  // Counted AFTER the copies above are made, so `borrowers_` inside the
  // loop would see this vector's own references. It does not run again:
  // membership was decided under the lock.
  std::size_t n = 0;
  for (const auto& w : spare) { n += w->stats().bytes; }
  return n;
}

std::size_t
GenerativeModelManager::pool_evict(std::size_t want)
{
  if (want == 0) { return 0; }
  std::vector<pair<string, shared_ptr<WeightSet>>> dropped;
  std::size_t freed = 0;
  {
    lock_guard<mutex> lk(_ws_mu);
    // LEAST RECENTLY USED first, because these are all equally droppable
    // and the one nobody has touched is the one least likely to be
    // wanted next. The pool this replaced had no order at all -- it was
    // an unordered_map walked front to back -- which meant a caller
    // asking for one block's worth could take the checkpoint another
    // pipeline was about to reuse.
    std::vector<std::unordered_map<string, shared_ptr<WeightSet>>::iterator>
        spare;
    for (auto it = _weight_sets.begin(); it != _weight_sets.end(); ++it) {
      if (it->second != nullptr && borrowers_(it->second) == 0) {
        spare.push_back(it);
      }
    }
    std::sort(spare.begin(), spare.end(),
              [](const auto& a, const auto& b) {
                return a->second->last_use() < b->second->last_use();
              });
    for (auto& it : spare) {
      if (freed >= want) { break; }
      freed += it->second->stats().bytes;
      dropped.emplace_back(it->first, std::move(it->second));
      _weight_sets.erase(it);
    }
  }
  // Released OUTSIDE the lock: dropping the last reference unmaps a
  // checkpoint, and holding the manager's mutex across that would put
  // every other model's open behind it.
  if (!dropped.empty() && session() != nullptr) {
    session()->log_debug(fmt(
        "GenerativeModelManager: dropped {} unborrowed checkpoint(s), {} MB, "
        "to make room", dropped.size(), freed >> 20));
  }
  dropped.clear();
  return freed;
}

void
GenerativeModelManager::set_wired_pool_enforced(bool on)
{
  _pool_enforce.store(on, std::memory_order_relaxed);
}

bool
GenerativeModelManager::wired_pool_enforced() const
{
  return _pool_enforce.load(std::memory_order_relaxed);
}

int
GenerativeModelManager::wired_pool_pct() const
{
  return _pool_pct.load(std::memory_order_relaxed);
}

std::size_t
GenerativeModelManager::wired_pool_limit() const
{
  // The ASK: an absolute figure when one was given, else the share of
  // the box. Zero either way turns wiring off entirely, which is a
  // setting and not a failure.
  std::size_t ask = _pool_bytes.load(std::memory_order_relaxed);
  if (ask == 0) {
    const int pct = _pool_pct.load(std::memory_order_relaxed);
    if (pct <= 0) { return 0; }
    const std::size_t ram = pool_physical_ram_();
    if (ram == 0) { return 0; }
    ask = ram / 100 * (std::size_t)pct;
  }
  // THE DEVICE CAP, before the refusal ceiling. Wiring past what the GPU
  // can keep resident does not buy residency, it buys a working set the
  // driver pages against -- so an ask above it is clamped rather than
  // attempted, and the attempt-and-collapse path below is left for the
  // failures it can actually diagnose.
  const std::size_t devmax = wired_pool_device_max();
  if (devmax > 0 && ask > devmax) { ask = devmax; }
  // Then whatever the box turned out to grant. Recorded only after an
  // mlock refusal, and never above the ask -- an operator lowering the
  // limit must not be overruled by a ceiling discovered when it was
  // higher.
  const std::size_t granted = _pool_granted.load(std::memory_order_relaxed);
  if (granted > 0 && granted < ask) { return granted; }
  return ask;
}

std::size_t
GenerativeModelManager::wired_pool_used() const
{
  return _pool_used.load(std::memory_order_relaxed);
}

bool
GenerativeModelManager::wired_pool_can_take(std::size_t bytes) const
{
  const std::size_t lim = wired_pool_limit();
  if (lim == 0) { return false; }
  return _pool_used.load(std::memory_order_relaxed) + bytes <= lim;
}

std::size_t
GenerativeModelManager::wire_into_pool(metal_compute::SharedBuffer& b)
{
  const std::size_t n = b.byte_size();
  if (n == 0 || b.is_wired()) { return 0; }
  if (!wired_pool_can_take(n)) {
    // SAID ONCE. A caller sees only "0 bytes wired" and cannot tell a
    // full pool from a box that refused -- and those want opposite
    // fixes: raise wired_pool_mb for the first, look at what else is
    // wired for the second.
    if (!_pool_full_said.exchange(true, std::memory_order_relaxed)
        && session() != nullptr) {
      session()->info(fmt(
          "wired pool: full at {} MB of {} MB; {} MB more was asked for "
          "and stays reclaimable", wired_pool_used() >> 20,
          wired_pool_limit() >> 20, n >> 20));
    }
    return 0;
  }
  if (!b.set_wired(true)) {
    const int e = errno;
    // The box refused. Collapse the ceiling to what is already held so
    // callers stop asking -- the percentage was only ever an up-to, and
    // this is the box saying what it will actually give.
    const std::size_t used = _pool_used.load(std::memory_order_relaxed);
    _pool_granted.store(used > 0 ? used : 1, std::memory_order_relaxed);
    // WARN, not info: the pool has just collapsed to whatever happened
    // to be held, so every later request fails too and the run silently
    // loses the protection it was configured for. errno is the part
    // that says WHY -- ENOMEM against a limit far above what was asked
    // means the pages themselves would not wire, which is what a
    // read-only file mapping does.
    if (!_pool_refused_said.exchange(true, std::memory_order_relaxed)
        && session() != nullptr) {
      session()->warn(fmt(
          "wired pool: the box refused to wire {} MB (errno {}: {}) with "
          "{} MB of a {} MB pool in use. The pool is capped at what is "
          "already held; the rest of this run's weights stay reclaimable",
          n >> 20, e, std::strerror(e), used >> 20,
          wired_pool_limit() >> 20));
    }
    return 0;
  }
  _pool_used.fetch_add(n, std::memory_order_relaxed);
  return n;
}

void
GenerativeModelManager::unwire_from_pool(metal_compute::SharedBuffer& b)
{
  const std::size_t n = b.byte_size();
  if (n == 0 || !b.is_wired()) { return; }
  b.set_wired(false);
  std::size_t was = _pool_used.load(std::memory_order_relaxed);
  while (!_pool_used.compare_exchange_weak(
      was, was > n ? was - n : 0, std::memory_order_relaxed)) {}
}

void
GenerativeModelManager::declare_weights(const string& dir,
                                        std::size_t   expected_bytes,
                                        const string& phase,
                                        const string& last_phase,
                                        std::size_t   floor_bytes)
{
  if (dir.empty()) { return; }
  const string key = canonicalize_(session(), dir);
  lock_guard<mutex> lk(_ws_mu);
  std::size_t& e = _declared[key];
  // Two stages naming one checkpoint declare it once, at the larger
  // estimate -- they are describing the same bytes, not two copies.
  if (expected_bytes > e) { e = expected_bytes; }
  // The floor takes the LARGER too, and for the mirror-image reason: two
  // stages describing one checkpoint's smallest form disagree only by
  // being differently informed, and believing the smaller would promise
  // room on the strength of the more optimistic of them.
  if (floor_bytes > 0) {
    std::size_t& f = _declared_floor[key];
    if (floor_bytes > f) { f = floor_bytes; }
  }
  // The WIDER lifetime wins when two stages disagree. A checkpoint one
  // stage will drop after conditioning and another holds all run is
  // held all run; believing the shorter claim would subtract bytes that
  // are still there. First declaration wins only in that it seeds the
  // entry -- an unphased claim overwrites a phased one, never the
  // reverse.
  auto [it, fresh] = _phase.try_emplace(key, phase);
  if (!fresh && phase.empty()) { it->second.clear(); }
  // The interval's END takes the LATER of two claims, mirroring how the
  // wider lifetime wins above: a component two stages describe is alive
  // until the last of them is finished with it.
  if (!last_phase.empty()) {
    auto& le = _phase_last[key];
    if (le.empty() || phase_index_(last_phase) > phase_index_(le)) {
      le = last_phase;
    }
  }
}

void
GenerativeModelManager::set_declaration_phase(const string& dir,
                                              const string& phase)
{
  if (dir.empty() || phase.empty()) { return; }
  const string key = canonicalize_(session(), dir);
  lock_guard<mutex> lk(_ws_mu);
  if (_declared.find(key) == _declared.end()) { return; }
  auto [it, fresh] = _phase.try_emplace(key, phase);
  if (fresh) { return; }
  // An EMPTY entry is what the declare pass leaves: "not yet decided",
  // not a competing answer. Refining it is the whole point of this
  // call, so it must not be read as a disagreement -- doing so silently
  // discarded every refinement, and the arithmetic still looked
  // plausible because it fell back to the safe, conservative total.
  if (it->second.empty()) { it->second = phase; return; }
  // A different phase already decided: two stages disagree about when
  // this checkpoint is held, so it is held throughout. Persistent is
  // the only reading consistent with both, and it is the safe one.
  if (it->second != phase) { it->second.clear(); }
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
  // AUDIT the launch that is ending, before its state goes.
  //
  // A phase declaration is a promise that these bytes are gone before
  // the peers that sized against them run, and a DiT has already spent
  // an irreversible streaming decision on it. Nothing else in the
  // system can falsify that promise: if the encoder is never dropped,
  // the run simply thrashes, and no line in the log connects the thrash
  // to the claim that caused it. So the promise is checked here -- one
  // launch late, which is the earliest moment the whole run is over and
  // still cheap.
  //
  // Warned, never silently tolerated, for the same reason an unplanned
  // ResourceClaim is warned about: a wrong lifetime does not fail as a
  // wrong answer, it fails as non-deterministic sizing that no test
  // reliably catches.
  {
    std::vector<string> broken;
    {
      lock_guard<mutex> lk(_ws_mu);
      for (const auto& [dir, phase] : _phase) {
        if (phase.empty()) { continue; }
        if (_released.count(dir) != 0) { continue; }
        broken.push_back(dir);
      }
    }
    for (const string& dir : broken) {
      if (session() == nullptr) { break; }
      session()->warn(fmt(
          "GenerativeModelManager: '{}' was declared for phase-limited use "
          "but never reported being released; peers that sized against that "
          "promise were given memory this run never freed", dir));
    }
  }

  lock_guard<mutex> lk(_ws_mu);
  _declared.clear();
  _declared_floor.clear();
  // Same scope, same reason as the declarations they qualify: a phase
  // is stated per launch from the owning stage's policy, and a graph
  // without that stage must not inherit its claim.
  _phase.clear();
  _released.clear();
  // Cleared with the declarations, and for the same reason: it is a fact
  // about THIS run's sizing. A relaunch at a different resolution, or
  // with a peer removed from the graph, may not stream at all, and a
  // stale flag would keep every conditioner unloading for the rest of
  // the process.
  _streaming.clear();
}

void
GenerativeModelManager::note_streaming(const string& dir)
{
  if (dir.empty()) { return; }
  const string key = canonicalize_(session(), dir);
  lock_guard<mutex> lk(_ws_mu);
  _streaming.insert(key);
}

bool
GenerativeModelManager::any_streaming() const
{
  lock_guard<mutex> lk(_ws_mu);
  return !_streaming.empty();
}

void
GenerativeModelManager::declare_scratch(const string& label, size_t bytes,
                                        const string& phase,
                                        const string& last_phase)
{
  if (label.empty() || bytes == 0) { return; }
  lock_guard<mutex> lk(_ws_mu);
  ScratchClaim& c = _scratch[label];
  if (bytes > c.bytes) { c.bytes = bytes; }
  // Same widening rule as a weight declaration: two stages disagreeing
  // about when an arena is live means it is live throughout.
  if (c.phase.empty() && c.bytes == bytes) { c.phase = phase; }
  else if (c.phase != phase) { c.phase.clear(); }
  if (!last_phase.empty() &&
      phase_index_(last_phase) > phase_index_(c.last_phase)) {
    c.last_phase = last_phase;
  }
}

void
GenerativeModelManager::set_phase_order(std::vector<string> phases)
{
  lock_guard<mutex> lk(_ws_mu);
  _phase_order = std::move(phases);
}

int
GenerativeModelManager::phase_index_(const string& p) const
{
  if (p.empty()) { return -1; }
  for (std::size_t i = 0; i < _phase_order.size(); ++i) {
    if (_phase_order[i] == p) { return (int)i; }
  }
  return -1;
}

size_t
GenerativeModelManager::scratch_bytes(const string& phase) const
{
  lock_guard<mutex> lk(_ws_mu);
  size_t unphased = 0;
  unordered_map<string, size_t> by_phase;
  for (const auto& [label, c] : _scratch) {
    (void)label;
    if (c.phase.empty()) { unphased += c.bytes; continue; }
    by_phase[c.phase] += c.bytes;
  }
  if (!phase.empty()) {
    auto it = by_phase.find(phase);
    return unphased + (it == by_phase.end() ? 0 : it->second);
  }
  size_t widest = 0;
  for (const auto& [p, b] : by_phase) {
    (void)p;
    if (b > widest) { widest = b; }
  }
  return unphased + widest;
}

void
GenerativeModelManager::revise_scratch(const string& label, size_t bytes)
{
  if (label.empty()) { return; }
  lock_guard<mutex> lk(_ws_mu);
  auto it = _scratch.find(label);
  if (it == _scratch.end()) { return; }   // never declared: nothing to fix
  // SET, not max: this is the arena in flight, and a ledger that only
  // ever grew would describe the largest beat of the run forever.
  it->second.bytes = bytes;
}

void
GenerativeModelManager::clear_scratch()
{
  lock_guard<mutex> lk(_ws_mu);
  _scratch.clear();
}

void
GenerativeModelManager::note_phase_released(const string& dir)
{
  if (dir.empty()) { return; }
  const string key = canonicalize_(session(), dir);
  lock_guard<mutex> lk(_ws_mu);
  _released.insert(key);
}

size_t
GenerativeModelManager::park_weights(const string& dir)
{
  if (dir.empty()) { return 0; }
  // BY PREFIX, because `_weight_sets` is keyed by canonical dir AND
  // variant ("<canon>|<variant>", see weight_set()). Looking the bare
  // canonical dir up in that map cannot match anything -- not even the
  // ordinary empty-variant case, whose key still carries the separator
  // -- so this returned 0 for every checkpoint that ever existed, and
  // `unload_when_idle: park` reclaimed nothing anywhere.
  //
  // That zero was visible and was attributed to the other reason a park
  // gives back nothing (a model that reads UNCACHED has nothing in the
  // set to park). Both are real; this one came first and hid the other.
  // The memory-cap path was never affected: enforce_memory_cap() walks
  // `_weight_sets` itself and never asks by name.
  //
  // One directory may hold several variants (a quantized twin of the
  // same checkpoint); parking is per SET, so park them all and report
  // the total -- the caller asked about a directory.
  //
  // AND ONLY WHAT NOBODY IS BORROWING, which is the third reason this
  // can return 0 and the only one that is a refusal rather than an
  // absence. The manager owns these checkpoints; a stage asking to park
  // one is speaking for itself, and it cannot speak for a peer whose
  // model is mid-forward over the same bytes. Parking under that peer
  // hands it pages the kernel may discard, and nothing on its read path
  // would notice. So a borrowed set is skipped and SAID, and it settles
  // on its own through release_unheld_() once the last borrower lets go.
  const string pfx = canonicalize_(session(), dir) + "|";
  vector<pair<shared_ptr<WeightSet>, long>> sets;
  {
    lock_guard<mutex> lk(_ws_mu);
    for (const auto& [k, w] : _weight_sets) {
      if (k.rfind(pfx, 0) != 0 || w == nullptr) { continue; }
      // Counted HERE, under the lock and before the copy below adds a
      // reference of its own. Reading it afterwards would report one
      // borrower too many for every set and park nothing, ever.
      sets.emplace_back(w, borrowers_(w));
    }
  }
  size_t total = 0;
  size_t borrowed = 0;
  for (const auto& [ws, holders] : sets) {
    if (ws->parked()) { continue; }
    if (holders > 0) { ++borrowed; continue; }
    const size_t got = park_if_unborrowed_(ws, holders);
    if (got == 0) { continue; }      // nothing parkable (mapped/uncached)
    total += got;
  }
  if (borrowed > 0 && session() != nullptr) {
    session()->log_debug(fmt(
        "GenerativeModelManager: '{}' is still borrowed by {} model(s), so "
        "it is not parked; it settles when the last one lets go", dir,
        borrowed));
  }
  return total;
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
    if (k.rfind(pfx, 0) == 0 && w != nullptr) { return true; }
  }
  return false;
}

// Bytes per canonical directory: what each checkpoint costs right now,
// counting a declared one at its estimate while its load is in flight.
// Shared by resident_weight_bytes() and phase_footprint(), which differ
// only in how they add the result up.
std::unordered_map<string, std::size_t>
GenerativeModelManager::per_dir_bytes_(bool use_floor) const
{
  // Snapshot under the lock, measure outside it -- stats() takes each
  // set's own mutex and holding this one across that would put the
  // query in the way of a concurrent load.
  std::vector<std::pair<string, std::shared_ptr<WeightSet>>> live;
  std::unordered_map<string, std::size_t> declared;
  std::unordered_map<string, std::size_t> floors;
  {
    lock_guard<mutex> lk(_ws_mu);
    live.reserve(_weight_sets.size());
    for (const auto& [k, w] : _weight_sets) {
      if (const shared_ptr<WeightSet>& sp = w; sp != nullptr) {
        // Key back to the plain directory: two variants of one
        // checkpoint are two sets but must be summed under one dir so
        // a declaration for it compares against the whole.
        const auto bar = k.rfind('|');
        live.emplace_back(bar == string::npos ? k : k.substr(0, bar),
                          std::move(sp));
      }
    }
    declared = _declared;
    floors = _declared_floor;
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


  // A claim on a directory that CONTAINS another claim counts the inner
  // one twice, because dir_weights_bytes() is recursive while this map
  // is keyed by exact path.
  //
  // MEASURED on a Comfy-Org MiniMax-H3 repack: the repo root was claimed
  // at 117861 MB -- its own recursive scan, which already includes the
  // DiT and the text encoder -- alongside separate claims for the DiT
  // (63209 MB) and the encoder (49120 MB). The graph declared 230781 MB
  // against a true 118452 MB, and every peer sized itself against a box
  // twice as full as the one it was on.
  //
  // Only a DECLARATION-ONLY container is adjusted, and that is the whole
  // safety of it: those bytes came from the recursive scan, so the inner
  // claim's bytes are provably part of them. A live weight set's
  // stats().bytes are what that set actually holds and contain nothing
  // of a sibling's, so subtracting from one would under-count -- the
  // direction that thrashes.
  //
  // Phases survive because the INNER entry is the one kept: the encoder
  // stays phase-limited to conditioning instead of being folded into an
  // unphased root and counted through the denoise.
  {
    std::unordered_set<string> is_live;
    for (const auto& [dir, ws] : live) { (void)ws; is_live.insert(dir); }
    auto contains = [](const string& outer, const string& inner) {
      if (inner.size() <= outer.size()) { return false; }
      if (inner.compare(0, outer.size(), outer) != 0) { return false; }
      // A boundary, so "/models/foo" does not swallow "/models/foobar".
      return outer.back() == '/' || inner[outer.size()] == '/';
    };
    // Over a SNAPSHOT, and only the OUTERMOST contained entries.
    // Reading the map while reducing it would subtract values already
    // reduced, and a claim nested three deep -- a root, a component
    // directory, and a .safetensors named inside it, all of which are
    // legal claims on a repack -- would have the innermost subtracted
    // twice: once on its own and once inside its parent. Under-counting
    // is the direction that thrashes, so the containment walk has to be
    // over disjoint children.
    const std::unordered_map<string, std::size_t> orig = per_dir;
    for (auto& [outer, bytes] : per_dir) {
      if (bytes == 0 || is_live.count(outer) != 0) { continue; }
      for (const auto& [inner, inner_bytes] : orig) {
        if (inner == outer || !contains(outer, inner)) { continue; }
        bool nested = false;
        for (const auto& [mid, mid_bytes] : orig) {
          (void)mid_bytes;
          if (mid == outer || mid == inner) { continue; }
          if (contains(outer, mid) && contains(mid, inner)) {
            nested = true;
            break;
          }
        }
        if (nested) { continue; }
        bytes -= (inner_bytes > bytes) ? bytes : inner_bytes;
      }
    }
  }
  // Floors are applied LAST, after containment.
  //
  // Order matters and the other way round is wrong: a container's bytes
  // came from a RECURSIVE scan that counted the inner component at full
  // size, so full size is what has to be subtracted from it. Reducing
  // the inner first leaves the container still carrying the difference,
  // and the floor buys nothing -- which is exactly what it did: a graph
  // whose DiT streams to 3 GB still reported 118 GB, because a root-dir
  // claim above it had no floor of its own and had only been credited
  // 3 GB for a 63 GB component.
  if (use_floor) {
    for (const auto& [dir, floor] : floors) {
      auto it = per_dir.find(dir);
      if (it == per_dir.end() || floor == 0) { continue; }
      if (floor < it->second) { it->second = floor; }
    }
  }
  return per_dir;
}

std::size_t
GenerativeModelManager::resident_weight_bytes() const
{
  const auto per_dir = per_dir_bytes_();
  std::size_t total = 0;
  for (const auto& [dir, b] : per_dir) { (void)dir; total += b; }
  return total;
}

std::size_t
GenerativeModelManager::phase_footprint(const string& phase) const
{
  return phase_footprint_(phase, /*use_floor=*/false);
}

std::size_t
GenerativeModelManager::phase_footprint_floor(const string& phase) const
{
  return phase_footprint_(phase, /*use_floor=*/true);
}

std::size_t
GenerativeModelManager::phase_peak(
    std::vector<std::pair<string, std::size_t>>* by_phase) const
{
  // Floors, because this answers "can it run", and a component that can
  // stream will be told to stream before the answer is no.
  const auto per_dir = per_dir_bytes_(/*use_floor=*/true);

  std::unordered_map<string, string> first_of, last_of;
  std::unordered_map<string, ScratchClaim> scratch;
  std::vector<string> order;
  {
    lock_guard<mutex> lk(_ws_mu);
    first_of = _phase;
    last_of  = _phase_last;
    scratch  = _scratch;
    order    = _phase_order;
  }
  if (by_phase != nullptr) { by_phase->clear(); }
  if (order.empty()) { return 0; }

  // One entry per thing that occupies memory, reduced to the interval
  // it is alive over. Weights and scratch are the same question here --
  // bytes with a lifetime -- so they are walked together rather than
  // summed separately and added, which is what made a latent alive
  // across three phases impossible to express.
  struct Live { std::size_t bytes; int first; int last; };
  std::vector<Live> live;
  live.reserve(per_dir.size() + scratch.size());

  auto span = [&](const string& f, const string& l) {
    // Unphased is alive throughout; an interval that runs backwards is a
    // caller mistake and is treated as ending where it began rather than
    // as empty, because a claim narrowed to nothing UNDER-counts.
    int a = -1;
    for (std::size_t i = 0; i < order.size(); ++i) {
      if (order[i] == f) { a = (int)i; break; }
    }
    if (a < 0) { return std::pair<int, int>{0, (int)order.size() - 1}; }
    int b = a;
    for (std::size_t i = 0; i < order.size(); ++i) {
      if (order[i] == l) { b = (int)i; break; }
    }
    if (b < a) { b = a; }
    return std::pair<int, int>{a, b};
  };

  for (const auto& [dir, b] : per_dir) {
    if (b == 0) { continue; }
    auto it = first_of.find(dir);
    auto lt = last_of.find(dir);
    const auto [a, z] = span(it == first_of.end() ? string() : it->second,
                             lt == last_of.end() ? string() : lt->second);
    live.push_back({b, a, z});
  }
  for (const auto& [label, c] : scratch) {
    (void)label;
    if (c.bytes == 0) { continue; }
    const auto [a, z] = span(c.phase, c.last_phase);
    live.push_back({c.bytes, a, z});
  }

  std::size_t peak = 0;
  for (int p = 0; p < (int)order.size(); ++p) {
    std::size_t at = 0;
    for (const Live& l : live) {
      if (l.first <= p && p <= l.last) { at += l.bytes; }
    }
    if (by_phase != nullptr) { by_phase->emplace_back(order[(std::size_t)p], at); }
    if (at > peak) { peak = at; }
  }
  return peak;
}

std::size_t
GenerativeModelManager::phase_footprint_(const string& phase,
                                         bool use_floor) const
{
  const auto per_dir = per_dir_bytes_(use_floor);
  std::unordered_map<string, string> phase_of;
  {
    lock_guard<mutex> lk(_ws_mu);
    phase_of = _phase;
  }

  std::size_t persistent = 0;                 // held for the whole run
  std::unordered_map<string, std::size_t> by_phase;
  for (const auto& [dir, b] : per_dir) {
    auto it = phase_of.find(dir);
    // Undeclared, or declared without a phase: it is there the whole
    // time. Anything this function does not KNOW to be phased has to
    // land here, because the alternative is dropping real bytes from
    // somebody's estimate.
    if (it == phase_of.end() || it->second.empty()) {
      persistent += b;
      continue;
    }
    by_phase[it->second] += b;
  }

  if (!phase.empty()) {
    auto it = by_phase.find(phase);
    return persistent + (it == by_phase.end() ? 0 : it->second);
  }
  // No phase named: the peak the box must survive, which is persistent
  // plus whichever single phase is largest.
  std::size_t widest = 0;
  for (const auto& [p, b] : by_phase) {
    (void)p;
    if (b > widest) { widest = b; }
  }
  return persistent + widest;
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
      if (w != nullptr) { live.push_back(w); }
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
  std::vector<pair<std::shared_ptr<WeightSet>, long>> live;
  {
    lock_guard<mutex> lk(_ws_mu);
    live.reserve(_weight_sets.size());
    for (const auto& [k, w] : _weight_sets) {
      (void)k;
      // Borrowers counted HERE, before `live` takes a reference of its
      // own; see park_if_unborrowed_.
      if (w != nullptr && !w->parked()) { live.emplace_back(w, borrowers_(w)); }
    }
  }
  std::sort(live.begin(), live.end(),
            [](const auto& a, const auto& b) {
              return a.first->last_use() < b.first->last_use();
            });

  std::size_t freed = 0;
  std::size_t in_use = 0;
  for (const auto& [ws, holders] : live) {
    if (active <= cap) { break; }
    if (holders > 0) { in_use += ws->stats().bytes; continue; }
    const std::size_t got = park_if_unborrowed_(ws, holders);
    if (got == 0) { continue; }        // nothing parkable (all mapped)
    freed  += got;
    active -= (got > active) ? active : got;
    if (session() != nullptr) {
      session()->log_debug(fmt(
          "GenerativeModelManager: parked {} MB of '{}' (cap {} MB, active "
          "now ~{} MB)", got >> 20, ws->dir(), cap >> 20, active >> 20));
    }
  }
  if (active > cap && session() != nullptr) {
    // WHY it fell short, because the two reasons want different
    // responses. Mapped weights and KV are the cap's standing
    // limitation. Bytes a model is still BORROWING are a different
    // thing: they are not unreclaimable, they are in use, and what
    // returns them is the stage letting go -- an `unload_when_idle`
    // policy, not a bigger cap.
    if (in_use > 0) {
      session()->warn(fmt(
          "GenerativeModelManager: still ~{} MB active after parking "
          "everything it could (cap {} MB); ~{} MB of that is borrowed by "
          "live models and is not the manager's to park -- a stage has to "
          "let go of it first", active >> 20, cap >> 20, in_use >> 20));
    } else if (freed > 0) {
      session()->warn(fmt(
          "GenerativeModelManager: still ~{} MB active after parking "
          "everything parkable (cap {} MB). Mapped weights and KV cannot be "
          "parked -- the cap is a target, not a guarantee",
          active >> 20, cap >> 20));
    }
  }
  return freed;
}

std::size_t
GenerativeModelManager::reclaim_at_least(std::size_t bytes)
{
  if (bytes == 0) { return 0; }
  // THE POOL FIRST. It is spare capacity by construction -- checkpoints
  // nothing is using, kept only because a relaunch might want them -- so
  // spending it costs a future reload, where parking a LIVE model's
  // weights costs the model that is running now.
  const std::size_t from_pool = pool_evict(bytes);
  if (from_pool >= bytes) { return from_pool; }
  bytes -= from_pool;
  // Then whatever is left UNBORROWED, least-recently-used first. What
  // this pass cannot do is take from a live model: parking a set a model
  // is reading hands it pages the kernel may discard, and it has no way
  // to notice (park_if_unborrowed_). So the honest ceiling on this call
  // is "everything nobody is using", and a caller that still does not
  // fit has to make room by dropping something of its own -- which is
  // exactly what the image and video stages do, freeing their DiT before
  // a decode rather than hoping this call covers it.
  std::vector<pair<std::shared_ptr<WeightSet>, long>> live;
  {
    lock_guard<mutex> lk(_ws_mu);
    live.reserve(_weight_sets.size());
    for (const auto& [k, w] : _weight_sets) {
      (void)k;
      // Borrowers counted HERE; see park_if_unborrowed_.
      if (w != nullptr && !w->parked()) { live.emplace_back(w, borrowers_(w)); }
    }
  }
  std::sort(live.begin(), live.end(),
            [](const auto& a, const auto& b) {
              return a.first->last_use() < b.first->last_use();
            });

  std::size_t freed = 0;
  std::size_t in_use = 0;
  for (const auto& [ws, holders] : live) {
    if (freed >= bytes) { break; }
    if (holders > 0) { in_use += ws->stats().bytes; continue; }
    const std::size_t got = park_if_unborrowed_(ws, holders);
    if (got == 0) { continue; }        // nothing parkable (all mapped)
    freed += got;
    if (session() != nullptr) {
      session()->log_debug(fmt(
          "GenerativeModelManager: parked {} MB of '{}' to make room "
          "({} of {} MB requested)", got >> 20, ws->dir(), freed >> 20,
          bytes >> 20));
    }
  }
  if (freed + from_pool < bytes && in_use > 0 && session() != nullptr) {
    session()->log_debug(fmt(
        "GenerativeModelManager: {} MB short of the {} MB asked for; ~{} MB "
        "is borrowed by live models and is not the manager's to park",
        (bytes - freed) >> 20, (bytes + from_pool) >> 20, in_use >> 20));
  }
  // BOTH, or a caller that asked for 8 GB and got 6 from the pool and 2
  // from parking would be told 2 and conclude it had failed.
  return from_pool + freed;
}

std::vector<GenerativeModelManager::WeightUsage>
GenerativeModelManager::weight_report() const
{
  // Snapshot the live sets under the lock, then measure OUTSIDE it: a
  // set's stats() takes its own mutex, and holding the cache lock
  // across that would put the report in the way of a concurrent load.
  //
  // The borrower count is taken HERE, under the lock, because it is the
  // one field that stops being true the moment it is copied: the manager
  // holds a reference of its own and `live` is about to hold a second,
  // so measuring it later reports two holders for a set one model has.
  std::vector<pair<std::shared_ptr<WeightSet>, long>> live;
  {
    lock_guard<mutex> lk(_ws_mu);
    live.reserve(_weight_sets.size());
    for (const auto& [k, w] : _weight_sets) {
      (void)k;
      if (w != nullptr) { live.emplace_back(w, borrowers_(w)); }
    }
  }
  std::vector<WeightUsage> out;
  out.reserve(live.size());
  for (const auto& [ws, holders] : live) {
    const auto st = ws->stats();
    WeightUsage u;
    u.dir          = ws->dir();
    u.bytes        = st.bytes;
    u.mapped_bytes = st.mapped_bytes;
    u.copied_bytes = st.copied_bytes;
    u.tensors      = st.entries;
    u.parts        = st.parts;
    // "Models using it", which is what a caller means -- not counting
    // the manager, which holds every set it ever opened.
    u.holders      = holders;
    u.parked       = ws->parked();
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
    if (w != nullptr) { ++n; }
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

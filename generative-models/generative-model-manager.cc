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

  {
    lock_guard<mutex> lk(_ws_mu);
    // The POOL first, and it is taken OUT of it: a set in use is not
    // spare capacity, and leaving it there would let a later eviction
    // drop pages a live model is reading.
    auto p = _pool.find(key);
    if (p != _pool.end()) {
      shared_ptr<WeightSet> sp = std::move(p->second);
      _pool.erase(p);
      _weight_sets[key] = sp;
      if (session() != nullptr) {
        session()->log_debug(fmt(
            "GenerativeModelManager: recycled '{}' from the removable pool "
            "({} MB, no reload)", key, sp->stats().bytes >> 20));
      }
      return sp;
    }
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

namespace {

// Believed physical RAM, honouring VPIPE_RAM_LIMIT_MB exactly as
// model_memory::phys_ram() and stream-pin.h do -- the pool has to agree
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
  // See the declaration: the pooled sets must be released while the
  // registry they deregister from is still alive.
  std::unordered_map<string, shared_ptr<WeightSet>> pool;
  {
    lock_guard<mutex> lk(_ws_mu);
    pool.swap(_pool);
  }
  pool.clear();
}

void
GenerativeModelManager::pool_weights(const string& dir, const string& variant)
{
  if (dir.empty()) { return; }
  const string key = canonicalize_(session(), dir) + "|" + variant;
  shared_ptr<WeightSet> ws;
  {
    lock_guard<mutex> lk(_ws_mu);
    auto it = _weight_sets.find(key);
    if (it == _weight_sets.end()) { return; }
    ws = it->second.lock();
  }
  if (!ws) { return; }
  if (!ws->recyclable()) {
    // NOT an error, and said rather than silent: a checkpoint that
    // reloads every launch looks exactly like one the pool never saw,
    // and the difference is the whole reason the flag exists.
    if (session() != nullptr) {
      session()->log_debug(fmt(
          "GenerativeModelManager: '{}' is not recyclable ({}), so it is "
          "dropped rather than pooled", key,
          ws->unrecyclable_reason().empty() ? std::string("unstated")
                                            : ws->unrecyclable_reason()));
    }
    return;
  }
  // Purgeable on the way in. The pool holds the set so a relaunch can
  // find it; marking it lets the kernel take the pages meanwhile, which
  // is what makes pooling free rather than a claim on the box.
  const size_t parked = _weights.park(ws.get());
  {
    lock_guard<mutex> lk(_ws_mu);
    _pool[key] = std::move(ws);
  }
  if (session() != nullptr) {
    session()->log_debug(fmt(
        "GenerativeModelManager: '{}' released to the removable pool "
        "({} MB purgeable, recycled free by a relaunch)", key,
        parked >> 20));
  }
}

std::size_t
GenerativeModelManager::pooled_bytes() const
{
  std::vector<shared_ptr<WeightSet>> live;
  {
    lock_guard<mutex> lk(_ws_mu);
    live.reserve(_pool.size());
    for (const auto& [k, w] : _pool) { (void)k; live.push_back(w); }
  }
  std::size_t n = 0;
  for (const auto& w : live) { n += w->stats().bytes; }
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
    for (auto it = _pool.begin(); it != _pool.end() && freed < want;) {
      freed += it->second->stats().bytes;
      dropped.emplace_back(it->first, std::move(it->second));
      it = _pool.erase(it);
    }
  }
  // Released OUTSIDE the lock: dropping the last reference unmaps a
  // checkpoint, and holding the manager's mutex across that would put
  // every other model's open behind it.
  if (!dropped.empty() && session() != nullptr) {
    session()->log_debug(fmt(
        "GenerativeModelManager: dropped {} pooled checkpoint(s), {} MB, to "
        "make room", dropped.size(), freed >> 20));
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
  const string key = canonicalize_(session(), dir);
  shared_ptr<WeightSet> ws;
  {
    lock_guard<mutex> lk(_ws_mu);
    auto it = _weight_sets.find(key);
    if (it == _weight_sets.end()) { return 0; }
    ws = it->second.lock();
  }
  if (!ws || ws->parked()) { return 0; }
  const size_t got = _weights.park(ws.get());
  if (got == 0) { return 0; }        // nothing parkable (mapped/uncached)
  ws->set_parked(true);
  return got;
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
  // Same LRU snapshot the cap path takes: a set nobody has touched goes
  // before one in active use, so the common case is that this costs a
  // state flip and no re-read.
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
    if (freed >= bytes) { break; }
    const std::size_t got = _weights.park(ws.get());
    if (got == 0) { continue; }        // nothing parkable (all mapped)
    ws->set_parked(true);
    freed += got;
    if (session() != nullptr) {
      session()->log_debug(fmt(
          "GenerativeModelManager: parked {} MB of '{}' to make room "
          "({} of {} MB requested)", got >> 20, ws->dir(), freed >> 20,
          bytes >> 20));
    }
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

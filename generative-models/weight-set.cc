#include "generative-models/weight-set.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/vpipe-format.h"
#include "generative-models/generative-model-manager.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"

#include <chrono>
#include <cstdlib>

#include <utility>

using namespace std;

namespace vpipe::genai {

namespace {

// Read once at open: whether this build run verifies that cached
// tensors are never written after load. See the header.
bool
integrity_from_env_()
{
  const char* e = std::getenv("VPIPE_WEIGHT_INTEGRITY");
  return e != nullptr && *e != '\0' && *e != '0';
}

}  // namespace

WeightSet::WeightSet(string dir, MetalLlamaWeights wts,
                     const SessionContextIntf* session)
  : _session(session)
  , _integrity(integrity_from_env_())
  , _dir(std::move(dir))
  , _wts(std::move(wts))
{
}

uint64_t
WeightSet::hash_(const metal_compute::SharedBuffer& b) const
{
  if (!_integrity || b.empty() || b.contents() == nullptr) { return 0; }
  // FNV-1a over 8-byte words plus the tail. Not cryptographic and does
  // not need to be: it is catching an accidental write, not an attack.
  const auto* p    = static_cast<const unsigned char*>(b.contents());
  const size_t n   = b.byte_size();
  uint64_t     h   = 1469598103934665603ull;
  const size_t w   = n / 8;
  const auto*  q   = reinterpret_cast<const uint64_t*>(p);
  for (size_t i = 0; i < w; ++i) {
    h = (h ^ q[i]) * 1099511628211ull;
  }
  for (size_t i = w * 8; i < n; ++i) {
    h = (h ^ p[i]) * 1099511628211ull;
  }
  // Fold the length in so a truncation cannot collide with the whole.
  return (h ^ n) * 1099511628211ull;
}

WeightSet::~WeightSet() = default;

shared_ptr<WeightSet>
WeightSet::open(const string& dir, const SessionContextIntf* session)
{
  auto wts = MetalLlamaWeights::open_model(dir);
  if (!wts) {
    if (session != nullptr) {
      session->log_debug(fmt(
          "WeightSet: no readable checkpoint under '{}'", dir));
    }
    return nullptr;
  }
  // SAY SO IF THE PACK IS MISALIGNED, once per checkpoint and here
  // rather than where it bites.
  //
  // THE COST THIS NAMES IS THE READ, not the mapping.
  //
  // A misaligned tensor also cannot be used where it lies -- a Metal
  // buffer offset must be 16-byte aligned -- and that used to be the
  // headline here. It is deliberately not any more: zero-copy mapping is
  // not the direction. Asking the OS to manage 4 KB pages over a 10+ GB
  // checkpoint costs more than owning the bytes does, and a mapped
  // weight can be neither wired, parked nor pooled (see
  // docs/MODEL-MEMORY.md), so a loader that maps is outside every
  // mechanism that protects memory. Leading with a penalty on the path
  // this tree is leaving would point an operator at the wrong repair.
  //
  // What is left is the one that is paid on EVERY path: F_NOCACHE only
  // bypasses the buffer cache on a page-aligned offset, so a misaligned
  // checkpoint's reads silently fall back to buffered I/O and grow the
  // file cache one-for-one with what they read. That memory comes out of
  // the weights themselves.
  //
  // The remedy is the one the operator can act on, so it is part of the
  // message: vpipe's own quantizer used to write tensors in an order
  // that shifted everything after the first odd-sized one, and re-running
  // it lays the same weights down aligned.
  if (session != nullptr) {
    const auto a = wts->alignment();
    if (a.misaligned > 0 || a.bad_shards > 0) {
      const bool whole = a.bad_shards > 0;
      session->warn(fmt(
          "weights: '{}' is not aligned ({}). Every UNCACHED read of it "
          "silently falls back to buffered I/O, because F_NOCACHE only "
          "bypasses the cache on a page-aligned offset -- MEASURED, "
          "6.2 GB read from a misaligned checkpoint grew the file cache "
          "by 6.2 GB, which is memory taken from the weights themselves. "
          "vpipe reads it on page boundaries and copies, so the cache "
          "stays clean at the price of a memcpy. If this model was "
          "quantized by an earlier version of vpipe, re-running the "
          "quantization is recommended: it writes the same weights "
          "aligned and costs neither",
          dir,
          whole ? fmt("{} of {} shard data sections start off-boundary, "
                      "which shifts every tensor in them",
                      a.bad_shards, a.shards)()
                : fmt("{} of {} tensors start off-boundary",
                      a.misaligned, a.tensors)()));
    }
  }
  // make_shared is not usable: the constructor is private.
  return shared_ptr<WeightSet>(
      new WeightSet(dir, std::move(*wts), session));
}

void
WeightSet::set_parked(bool p) noexcept
{
  _parked.store(p, std::memory_order_relaxed);
}

bool
WeightSet::recyclable() const noexcept
{
  return _recyclable.load(std::memory_order_relaxed);
}

void
WeightSet::set_not_recyclable(string why)
{
  _recyclable.store(false, std::memory_order_relaxed);
  // FIRST reason wins: it is the one that made the set unrecyclable, and
  // a later caller overwriting it would report the symptom rather than
  // the cause.
  if (_unrecyclable_why.empty()) { _unrecyclable_why = std::move(why); }
}

const string&
WeightSet::unrecyclable_reason() const noexcept
{
  return _unrecyclable_why;
}

bool
WeightSet::parked() const noexcept
{
  return _parked.load(std::memory_order_relaxed);
}

uint64_t
WeightSet::last_use() const noexcept
{
  return _last_use.load(std::memory_order_relaxed);
}

void
WeightSet::ensure_active_()
{
  // One monotonic tick per access, shared across every set in the
  // process, so "least recently used" is comparable between them.
  static std::atomic<uint64_t> tick{0};
  _last_use.store(tick.fetch_add(1, std::memory_order_relaxed) + 1,
                  std::memory_order_relaxed);
  if (!_parked.load(std::memory_order_relaxed)) { return; }

  // Parked: take the pages back BEFORE handing any bytes out. Their
  // contents survive whenever nothing else needed the RAM, in which
  // case this is a state flip with no I/O; if the kernel did take
  // them, reactivate() drives reload_weights() and the bytes are
  // re-read from disk. Either way what the caller gets is valid.
  //
  // Safe to call with _mu already held (every caller holds it): the
  // mutex is recursive, and the begin_restore()/end_restore() bracket
  // reactivate() runs just nests one level deeper.
  _parked.store(false, std::memory_order_relaxed);
  if (_registry != nullptr) {
    bool reloaded = false;
    if (!_registry->reactivate(this, &reloaded) && !reloaded &&
        _session != nullptr) {
      _session->warn(fmt(
          "WeightSet('{}'): parked weights were reclaimed and could not be "
          "re-read; models built from this set are unusable", _dir));
    }
  }
}

void
WeightSet::set_registration(WeightRegistry::Registration reg,
                            WeightRegistry*              registry)
{
  // The OLD registration is released after _mu is dropped. Releasing it
  // takes the registry's lock, and the registry takes _mu (via
  // for_each_weight / begin_restore) while holding its own -- doing
  // both under _mu here would be the opposite order, i.e. a deadlock.
  // Today _reg is always empty at this point so the release is a no-op,
  // but the ordering should not depend on that.
  WeightRegistry::Registration old;
  {
    lock_guard<recursive_mutex> lk(_mu);
    old = std::move(_reg);
    _reg = std::move(reg);
    _registry = registry;
  }
}

bool
WeightSet::has(const string& name) const
{
  lock_guard<recursive_mutex> lk(_mu);
  return _wts->has(name);
}

metal_compute::SharedBuffer
WeightSet::alias_(const Entry& e) const
{
  // A whole-buffer subview: same bytes, same GPU offset, +1 on the
  // MTL refcount, and NOT accounted -- so the footprint counters and
  // the residency policy both keep treating the cached handle as the
  // one true owner of these bytes.
  if (e.buf.empty()) { return {}; }
  return e.buf.subview(0, e.buf.byte_size());
}

metal_compute::SharedBuffer
WeightSet::tensor(const string& name, metal_compute::MetalCompute* mc,
                  Residency res, const string& part)
{
  if (mc == nullptr) { return {}; }
  lock_guard<recursive_mutex> lk(_mu);
  if (_mc == nullptr) { _mc = mc; }
  ensure_active_();

  auto it = _cache.find(name);
  if (it != _cache.end()) { return alias_(it->second); }

  metal_compute::SharedBuffer buf =
      res == Residency::Mapped ? _wts->load_mapped(name, mc)
                               : _wts->load(name, mc);
  if (buf.empty()) { return {}; }

  Entry e;
  e.src_name = name;
  e.part     = part;
  // Record what we actually GOT, not what was asked for: load_mapped
  // falls back to a copy for GGUF-backed tensors and for file offsets
  // that are not GPU-bindable. An owned buffer is parkable and
  // reloadable; filing it as "mapped" would exclude it from both.
  e.res = buf.is_owned() ? Residency::Copied : Residency::Mapped;
  e.buf = std::move(buf);
  // Copied only: a Mapped tensor is already immutable (the shard is
  // mapped PROT_READ, so a write faults), and hashing one would
  // fault every page of it in -- the opposite of what mapping is for.
  e.hash = e.res == Residency::Copied ? hash_(e.buf) : 0;
  auto ins = _cache.emplace(name, std::move(e));
  return alias_(ins.first->second);
}

metal_compute::SharedBuffer
WeightSet::read(const string& name, metal_compute::MetalCompute* mc,
                Residency res)
{
  if (mc == nullptr) { return {}; }
  {
    lock_guard<recursive_mutex> lk(_mu);
    if (_mc == nullptr) { _mc = mc; }
    ensure_active_();
    // Two kinds of load mutate state shared through this set and so
    // have to stay serialised:
    //   Mapped -- load_mapped lazily wraps a whole shard.
    //   GGUF   -- the on-demand converter is not re-entrant.
    // is_gguf() is a whole-checkpoint property, so this costs nothing
    // per read -- an info() lookup here would be paid on every streamed
    // block for a fact that cannot change.
    if (res == Residency::Mapped) { return _wts->load_mapped(name, mc); }
    if (_wts->is_gguf()) { return _wts->load(name, mc); }
  }
  // Plain safetensors copy: a fresh allocation plus a memcpy out of a
  // read-only mmap, touching nothing this set owns. Deliberately done
  // OUTSIDE the lock -- which is also why the cost split below is
  // accumulated with relaxed atomics rather than under it.
  //
  // This is the block-streaming path, and streaming implies Copied (a
  // model that streams turns mmap OFF -- see the DiTs' _mmap_weights).
  // Holding the lock across the memcpy would make two pipelines sharing
  // one checkpoint take turns on every block read, which is exactly the
  // concurrency the sharing exists to enable.
  MetalLlamaWeights::LoadCost cost;
  metal_compute::SharedBuffer out = _wts->load(name, mc, &cost);
  _streamed_alloc_us.fetch_add((std::uint64_t)(cost.alloc_ms * 1000.0),
                               std::memory_order_relaxed);
  _streamed_fetch_us.fetch_add((std::uint64_t)(cost.fetch_ms * 1000.0),
                               std::memory_order_relaxed);
  return out;
}

metal_compute::SharedBuffer
WeightSet::stream_tensor(const string& name, metal_compute::MetalCompute* mc,
                         Residency res)
{
  metal_compute::SharedBuffer buf = read(name, mc, res);
  if (!buf.empty()) {
    _streamed_reads.fetch_add(1, std::memory_order_relaxed);
    _streamed_bytes.fetch_add(buf.byte_size(), std::memory_order_relaxed);
  }
  return buf;
}

bool
WeightSet::stream_into(const string& name, void* dst, std::size_t cap)
{
  if (dst == nullptr) { return false; }
  std::size_t nbytes = 0;
  {
    lock_guard<recursive_mutex> lk(_mu);
    if (!_wts) { return false; }
    ensure_active_();
    // GGUF converts on the way out, so there are no raw bytes to place.
    // Asked here rather than inside pread_into so the answer is taken
    // under the same lock that guards a reactivating set.
    if (_wts->is_gguf()) { return false; }
    const auto* ti = _wts->info(name);
    if (ti == nullptr) { return false; }
    nbytes = ti->nbytes;
  }
  // Outside the lock, for the reason read() spells out: this is the block
  // stream, and holding the lock across it makes two pipelines sharing one
  // checkpoint take turns on every block.
  const auto t0 = std::chrono::steady_clock::now();
  if (!_wts->pread_into(name, dst, cap)) { return false; }
  // All FETCH and no alloc, which is the whole point -- and it has to be
  // charged so the streaming profile keeps meaning what it says. Left
  // uncounted, a run that took this path would report its reads as free
  // and the alloc/fetch split would describe only whichever blocks
  // happened to go the other way.
  const double ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
  _streamed_fetch_us.fetch_add((std::uint64_t)(ms * 1000.0),
                               std::memory_order_relaxed);
  _streamed_reads.fetch_add(1, std::memory_order_relaxed);
  _streamed_bytes.fetch_add(nbytes, std::memory_order_relaxed);
  return true;
}

metal_compute::SharedBuffer
WeightSet::stream_derived(
    const function<metal_compute::SharedBuffer()>& build)
{
  if (!build) { return {}; }
  metal_compute::SharedBuffer buf = build();
  if (buf.empty()) { return buf; }
  _streamed_reads.fetch_add(1, std::memory_order_relaxed);
  _streamed_bytes.fetch_add(buf.byte_size(), std::memory_order_relaxed);
  return buf;
}

metal_compute::SharedBuffer
WeightSet::derived(const string&                                  key,
                   const function<metal_compute::SharedBuffer()>& build,
                   const string&                                  part)
{
  lock_guard<recursive_mutex> lk(_mu);
  ensure_active_();
  auto it = _cache.find(key);
  if (it != _cache.end()) { return alias_(it->second); }
  if (!build) { return {}; }

  metal_compute::SharedBuffer buf = build();
  if (buf.empty()) { return {}; }

  Entry e;
  e.src_name = {};                 // no name to re-read: never parked
  e.part     = part;
  e.res      = Residency::Copied;
  e.buf      = std::move(buf);
  e.hash     = hash_(e.buf);
  auto ins = _cache.emplace(key, std::move(e));
  return alias_(ins.first->second);
}

bool
WeightSet::part_ready(const string& part) const
{
  lock_guard<recursive_mutex> lk(_mu);
  auto it = _parts.find(part);
  return it != _parts.end() && it->second;
}

bool
WeightSet::ensure_part(const string& part, const function<bool()>& load)
{
  lock_guard<recursive_mutex> lk(_mu);
  auto it = _parts.find(part);
  if (it != _parts.end()) { return it->second; }
  const bool ok = load ? load() : false;
  _parts[part] = ok;
  if (_session != nullptr) {
    _session->log_debug(fmt(
        "WeightSet('{}'): part '{}' {}", _dir, part,
        ok ? "loaded" : "FAILED to load"));
  }
  return ok;
}

size_t
WeightSet::release_part(const string& part)
{
  lock_guard<recursive_mutex> lk(_mu);
  size_t freed = 0;
  for (auto it = _cache.begin(); it != _cache.end();) {
    if (it->second.part == part) {
      freed += it->second.buf.byte_size();
      it = _cache.erase(it);
    } else {
      ++it;
    }
  }
  _parts.erase(part);
  if (_session != nullptr && freed > 0) {
    _session->log_debug(fmt(
        "WeightSet('{}'): released part '{}' ({} MB; bytes come back only "
        "once every model holding an alias has dropped it)",
        _dir, part, freed >> 20));
  }
  return freed;
}

void
WeightSet::for_each_weight(
    const function<void(metal_compute::SharedBuffer&)>& cb)
{
  lock_guard<recursive_mutex> lk(_mu);
  for (auto& [k, e] : _cache) {
    (void)k;
    // Only tensors we can put back: an owned buffer whose source name we
    // still know. Mapped views decline to park anyway (they are subviews
    // of the shard wrap), and a derived tensor has no retained transform
    // to rebuild it with, so parking one would be unrecoverable.
    if (e.res == Residency::Copied && !e.src_name.empty()) { cb(e.buf); }
  }
}

bool
WeightSet::reload_weights()
{
  lock_guard<recursive_mutex> lk(_mu);
  if (!_wts || _mc == nullptr) { return false; }
  bool all_ok = true;
  size_t n = 0;
  for (auto& [k, e] : _cache) {
    (void)k;
    if (e.res != Residency::Copied || e.src_name.empty()) { continue; }
    // IN PLACE. The allocation survives a purge (only its contents are
    // discarded), so re-reading into the existing pointer keeps every
    // alias a live model is holding pointing at the right bytes.
    // Reloading into a fresh buffer would silently strand them.
    if (!_wts->read_into(e.src_name, e.buf.contents(), e.buf.byte_size())) {
      all_ok = false;
    } else {
      // Re-arm here, under the same lock that guards the write. This is
      // the whole disarm/re-arm story: because the registry is the ONLY
      // caller and this is the ONLY writer, nothing outside needs to
      // know the hash was ever stale.
      e.hash = hash_(e.buf);
      ++n;
    }
  }
  if (_session != nullptr) {
    _session->log_debug(fmt(
        "WeightSet('{}'): re-read {} reclaimed tensor(s) in place{}",
        _dir, n, all_ok ? "" : " (some FAILED)"));
  }
  return all_ok;
}

void
WeightSet::begin_restore()
{
  // Deliberately an UNSCOPED lock: it is released by end_restore(),
  // after the registry has finished reloading. That is the point --
  // every reader goes through _mu, so holding it here makes a
  // concurrent tensor() block until the bytes are valid again instead
  // of being handed a buffer whose pages the kernel took.
  //
  // _mu is recursive, so reload_weights() re-locking on this same
  // thread is fine.
  _mu.lock();
  _restoring.store(true, std::memory_order_relaxed);
}

void
WeightSet::end_restore(bool ok)
{
  // Defensive: an unmatched end_restore() must not unlock a mutex this
  // thread never took.
  if (!_restoring.exchange(false, std::memory_order_relaxed)) { return; }
  if (!ok && _session != nullptr) {
    _session->warn(fmt(
        "WeightSet('{}'): restore FAILED; cached tensors hold undefined "
        "contents and every model built from this set must be rebuilt",
        _dir));
  }
  _mu.unlock();
}

size_t
WeightSet::verify_integrity() const
{
  lock_guard<recursive_mutex> lk(_mu);
  if (!_integrity) { return 0; }
  size_t bad = 0;
  for (const auto& [k, e] : _cache) {
    if (e.hash == 0) { continue; }         // nothing recorded
    if (hash_(e.buf) == e.hash) { continue; }
    ++bad;
    if (_session != nullptr) {
      _session->warn(fmt(
          "WeightSet('{}'): tensor '{}' was MODIFIED after load. Cached "
          "tensors are shared between models, so this corrupts every "
          "other holder of it", _dir, k));
    }
  }
  return bad;
}

string
WeightSet::weight_label() const
{
  return _dir;
}

WeightSet::Stats
WeightSet::stats() const
{
  lock_guard<recursive_mutex> lk(_mu);
  Stats s;
  s.entries = _cache.size();
  for (const auto& [k, e] : _cache) {
    (void)k;
    const size_t b = e.buf.byte_size();
    s.bytes += b;
    if (e.res == Residency::Mapped) { s.mapped_bytes += b; }
    else                            { s.copied_bytes += b; }
  }
  for (const auto& [p, ok] : _parts) {
    (void)p;
    if (ok) { ++s.parts; }
  }
  s.streamed_reads = _streamed_reads.load(std::memory_order_relaxed);
  s.streamed_bytes = _streamed_bytes.load(std::memory_order_relaxed);
  s.streamed_alloc_ms =
      (double)_streamed_alloc_us.load(std::memory_order_relaxed) / 1000.0;
  s.streamed_fetch_ms =
      (double)_streamed_fetch_us.load(std::memory_order_relaxed) / 1000.0;
  return s;
}

shared_ptr<WeightSet>
open_weight_set(const string& dir, const SessionContextIntf* session,
                const string& variant)
{
  if (session != nullptr) {
    if (auto* mgr = session->services()->generative_model_manager()) {
      return mgr->weight_set(dir, variant);
    }
  }
  return WeightSet::open(dir, session);
}

}  // namespace vpipe::genai

#include "generative-models/moe-expert-cache.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <format>
#include <mutex>
#include <thread>

namespace vpipe::genai {

using metal_compute::SharedBuffer;

namespace {

// A slot's read: none, queued as a step's own, queued ahead, under way.
enum : std::uint8_t { kIdle = 0, kQueued = 1, kAheadQueued = 2, kRunning = 3 };

using Clock = std::chrono::steady_clock;

double
ms_since(Clock::time_point t0)
{
  return std::chrono::duration<double, std::milli>(Clock::now() - t0)
      .count();
}

}  // namespace

// The reader threads and the two queues they drain, a step's own misses
// before the reads ahead. A slot's read state (_io, _bad) and the queues are
// under `mu`, and the caller changes no slot's key while its read is queued
// or under way, so a reader takes the key under the lock and the bytes
// without it.
struct MoeExpertCache::Pool {
  std::mutex mu;
  std::condition_variable work_cv, done_cv;
  std::deque<int> demand, ahead;
  std::string err;                   // the first failed read's
  std::uint64_t bytes = 0;           // read so far
  bool stop = false;
  std::vector<std::thread> threads;
};

MoeExpertCache::~MoeExpertCache()
{
  if (_workers) {
    {
      std::lock_guard<std::mutex> lk(_workers->mu);
      _workers->stop = true;
      _workers->demand.clear();
      _workers->ahead.clear();
    }
    _workers->work_cv.notify_all();
    for (auto& t : _workers->threads) { t.join(); }
  }
}

std::unique_ptr<MoeExpertCache>
MoeExpertCache::create(metal_compute::MetalCompute* mc, const Options& opt,
                       Fill fill, std::string* err)
{
  auto fail = [&](std::string m) -> std::unique_ptr<MoeExpertCache> {
    if (err != nullptr) { *err = std::move(m); }
    return nullptr;
  };
  if (mc == nullptr || !fill || opt.layers <= 0 || opt.experts <= 0 ||
      opt.part_bytes.empty() || opt.resident < 0 || opt.transient <= 0) {
    return fail("expert cache: a bad layout");
  }
  std::unique_ptr<MoeExpertCache> c(new MoeExpertCache());
  c->_mc = mc;
  c->_opt = opt;
  c->_fill = std::move(fill);
  c->_resident = opt.resident;
  c->_transient = opt.transient;
  const int slots = opt.resident + opt.transient;
  for (std::size_t b : opt.part_bytes) {
    c->_slot_bytes += b;
    SharedBuffer p = mc->make_shared_buffer(b * (std::size_t)slots);
    if (p.empty()) {
      return fail(std::format("expert cache: could not allocate {} MB",
                              (b * (std::size_t)slots) >> 20));
    }
    c->_pool.push_back(std::move(p));
  }
  const std::size_t keys = (std::size_t)opt.layers * opt.experts;
  c->_slot_of.assign(keys, -1);
  c->_score.assign(keys, 0.0f);
  const std::size_t S = (std::size_t)slots;
  c->_key.assign(S, -1);
  c->_res.assign(S, 0);
  c->_is_ahead.assign(S, 0);
  c->_used.assign(S, 0);
  c->_io.assign(S, kIdle);
  c->_bad.assign(S, 0);
  c->_spare.resize(S);
  c->_spare_at.resize(S);
  for (int s = 0; s < slots; ++s) {   // every slot a spare, empty
    c->_spare[(std::size_t)s] = s;
    c->_spare_at[(std::size_t)s] = s;
  }
  for (int w : opt.ahead_widths) {
    if (w >= 0) { c->_aw.widths.push_back(std::min(w, opt.experts)); }
  }
  c->_aw.ms.assign(c->_aw.widths.size(), 0.0);
  c->_aw.n.assign(c->_aw.widths.size(), 0);
  c->_aw.width = c->_aw.widths.empty() ? 0 : c->_aw.widths[0];
  c->_aw.round = c->_aw.widths.size() > 1;            // else nothing to choose
  c->_workers = std::make_unique<Pool>();
  // VPIPE_MOE_EXPERT_THREADS=<n> sets the readers (an A/B: the drive
  // decides how many requests in flight it answers best)
  int threads = opt.threads;
  if (const char* e = std::getenv("VPIPE_MOE_EXPERT_THREADS")) {
    threads = std::clamp(std::atoi(e), 1, 64);
  }
  const int nt = std::max(1, threads - 1);       // the caller is one more
  for (int t = 0; t < nt; ++t) {
    c->_workers->threads.emplace_back([p = c.get()] { p->worker_(); });
  }
  return c;
}

void
MoeExpertCache::slot_ptrs_(int slot, std::uint8_t** dst) const
{
  for (std::size_t k = 0; k < _pool.size(); ++k) {
    dst[k] = static_cast<std::uint8_t*>(_pool[k].contents()) +
             (std::size_t)slot * _opt.part_bytes[k];
  }
}

// One read, the lock held on entry and on return (not across the read).
void
MoeExpertCache::run_(int s, std::unique_lock<std::mutex>& lk,
                     std::uint8_t** dst)
{
  Pool& w = *_workers;
  _io[(std::size_t)s] = kRunning;
  const std::int32_t key = _key[(std::size_t)s];
  lk.unlock();
  slot_ptrs_(s, dst);
  std::string e;
  const bool ok =
      _fill(key / _opt.experts, key % _opt.experts, dst, &e);
  lk.lock();
  _io[(std::size_t)s] = kIdle;
  _bad[(std::size_t)s] = ok ? 0 : 1;
  if (ok) {
    w.bytes += _slot_bytes;
  } else if (w.err.empty()) {
    w.err = e.empty() ? std::format("layer {} expert {} did not read",
                                    key / _opt.experts, key % _opt.experts)
                      : e;
  }
  w.done_cv.notify_all();
}

void
MoeExpertCache::worker_()
{
  Pool& w = *_workers;
  std::vector<std::uint8_t*> dst(_pool.size());
  std::unique_lock<std::mutex> lk(w.mu);
  for (;;) {
    w.work_cv.wait(lk, [&] {
      return w.stop || !w.demand.empty() || !w.ahead.empty();
    });
    if (w.stop) { return; }
    std::deque<int>& q = !w.demand.empty() ? w.demand : w.ahead;
    const int s = q.front();
    q.pop_front();
    run_(s, lk, dst.data());
  }
}

// Until every slot in `on` has no read queued or under way; `help` runs
// the queued demand reads on this thread too. False, with the first
// failure, when one of `on` did not read.
bool
MoeExpertCache::wait_(std::unique_lock<std::mutex>& lk,
                      const std::vector<int>& on, bool help,
                      std::string* err)
{
  Pool& w = *_workers;
  if (help) {
    std::vector<std::uint8_t*> dst(_pool.size());
    while (!w.demand.empty()) {
      const int s = w.demand.front();
      w.demand.pop_front();
      run_(s, lk, dst.data());
    }
  }
  w.done_cv.wait(lk, [&] {
    for (int s : on) {
      if (_io[(std::size_t)s] != kIdle) { return false; }
    }
    return true;
  });
  for (int s : on) {
    if (_bad[(std::size_t)s] != 0) {
      if (err != nullptr) { *err = "expert cache: " + w.err; }
      return false;
    }
  }
  return true;
}

// A spare this step may take: not used by it and not being read. An empty
// one first, else the least recently used. -1 when there is none.
int
MoeExpertCache::take_spare_() const
{
  int best = -1;
  for (int s : _spare) {
    const std::size_t i = (std::size_t)s;
    if (_used[i] == _step || _io[i] != kIdle) { continue; }
    if (_key[i] < 0) { return s; }
    if (best < 0 || _used[i] < _used[(std::size_t)best]) { best = s; }
  }
  return best;
}

// The least-selected resident expert this step does not need, or -1.
int
MoeExpertCache::victim_() const
{
  int v = -1;
  float vs = 0.0f;
  for (int s = 0; s < slots(); ++s) {
    const std::size_t i = (std::size_t)s;
    if (_res[i] == 0 || _used[i] == _step) { continue; }
    const float sc = _score[(std::size_t)_key[i]];
    if (v < 0 || sc < vs) { v = s; vs = sc; }
  }
  return v;
}

// Forget what slot s holds (its read, if any, has settled).
void
MoeExpertCache::drop_(int s)
{
  const std::size_t i = (std::size_t)s;
  if (_key[i] >= 0) { _slot_of[(std::size_t)_key[i]] = -1; }
  _key[i] = -1;
  _is_ahead[i] = 0;
  _bad[i] = 0;
}

void
MoeExpertCache::to_resident_(int s)
{
  const std::size_t i = (std::size_t)s;
  if (_res[i] != 0) { return; }
  const int at = _spare_at[i];
  const int last = _spare.back();
  _spare[(std::size_t)at] = last;
  _spare_at[(std::size_t)last] = at;
  _spare.pop_back();
  _spare_at[i] = -1;
  _res[i] = 1;
  ++_n_res;
}

void
MoeExpertCache::to_spare_(int s)
{
  const std::size_t i = (std::size_t)s;
  if (_res[i] == 0) { return; }
  _spare_at[i] = (int)_spare.size();
  _spare.push_back(s);
  _res[i] = 0;
  --_n_res;
}

void
MoeExpertCache::age_()
{
  for (float& s : _score) { s *= 0.5f; }
}

// Step `L` (of `rows` rows) blocked `blocked_ms` on reads: charge a decode
// step whose predecessor took its width, and pick the next step's.
void
MoeExpertCache::note_step_(int L, int rows, double blocked_ms)
{
  AheadChooser& a = _aw;
  const int nw = (int)a.widths.size();
  if (nw < 2) { return; }
  if (rows < 1 || rows > std::max(1, _opt.ahead_rows)) {
    // a prefill breaks the chain
    a.prev_layer = -2;
    return;
  }
  if (a.round && a.prev_layer == L - 1 && a.prev == a.cur) {
    a.ms[(std::size_t)a.cur] += blocked_ms;
    ++a.n[(std::size_t)a.cur];
  }
  a.prev_layer = L;
  a.prev = a.cur;
  auto set = [&](int i) {
    a.cur = i;
    a.width = a.widths[(std::size_t)i];
  };
  if (!a.round) {
    if (--a.keep > 0) { return; }
    std::fill(a.ms.begin(), a.ms.end(), 0.0);   // measure again
    std::fill(a.n.begin(), a.n.end(), 0);
    a.round = true;
    a.block = 0;
  }
  bool done = true;
  for (std::uint64_t k : a.n) { done = done && k >= AheadChooser::kSamples; }
  if (!done) {
    if (--a.block > 0) { return; }
    // The next block's width, drawn (splitmix64 of a counter).
    std::uint64_t z = ++a.draws * 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    set((int)((z ^ (z >> 31)) % (std::uint64_t)nw));
    a.block = AheadChooser::kBlock;
    return;
  }
  // The cheapest per step; a smaller width within 3% of it instead.
  std::vector<double> per((std::size_t)nw);
  int best = 0;
  for (int i = 0; i < nw; ++i) {
    const std::size_t u = (std::size_t)i;
    per[u] = a.ms[u] / (double)a.n[u];
    if (per[u] < per[(std::size_t)best]) { best = i; }
  }
  int pick = best;
  for (int i = 0; i < nw; ++i) {
    if (a.widths[(std::size_t)i] < a.widths[(std::size_t)pick] &&
        per[(std::size_t)i] <= 1.03 * per[(std::size_t)best]) {
      pick = i;
    }
  }
  set(pick);
  a.round = false;
  a.keep = AheadChooser::kKeep;
  if (const SessionContextIntf* sess = _mc->session()) {
    std::string m;
    for (int i = 0; i < nw; ++i) {
      m += std::format(" width {} {:.3f} ms", a.widths[(std::size_t)i],
                       per[(std::size_t)i]);
    }
    sess->log_debug(fmt("[moe-cache] read-ahead: a decode step blocked, "
                        "{} steps a width:{} -> width {}",
                        AheadChooser::kSamples, m, a.width));
  }
}

bool
MoeExpertCache::resolve(int L, const std::int32_t* eid, int np, int rows,
                        std::int32_t* slot, std::string* err,
                        const Ahead* ahead)
{
  const int E = _opt.experts;
  if (L < 0 || L >= _opt.layers || np < 0 || rows <= 0) {
    if (err != nullptr) { *err = "expert cache: a bad step"; }
    return false;
  }
  // The step's distinct experts and how many pairs each.
  std::vector<int> used;
  std::vector<int> count((std::size_t)E, 0);
  for (int p = 0; p < np; ++p) {
    const int e = eid[p];
    if (e < 0 || e >= E) {
      if (err != nullptr) {
        *err = std::format("expert cache: layer {} routed to expert {}", L, e);
      }
      return false;
    }
    if (count[(std::size_t)e]++ == 0) { used.push_back(e); }
  }
  ++_stats.steps;
  ++_step;
  const std::size_t base = (std::size_t)L * E;
  const float wt = 1.0f / (float)rows;
  for (int e : used) {
    _score[base + (std::size_t)e] += (float)count[(std::size_t)e] * wt;
  }
  if (L == 0 && ++_layer0_steps % (std::uint64_t)std::max(1, _opt.half_life)
                    == 0) {
    age_();
  }
  // How well the read-ahead for this step named it.
  if (_named_layer == L) {
    _stats.ahead_picks += used.size();
    for (int e : used) {
      if (std::find(_named.begin(), _named.end(), e) != _named.end()) {
        ++_stats.ahead_named;
      }
    }
  }
  _named_layer = -1;
  Pool& w = *_workers;
  std::unique_lock<std::mutex> lk(w.mu);
  // Settle the reads ahead: a queued one this step picks becomes its own
  // (read first); a queued one it does not is dropped; one that read and
  // failed is forgotten. Those still under way are settled later.
  {
    std::vector<int> keep;
    for (int s : _ahead) {
      const std::size_t i = (std::size_t)s;
      const std::int32_t key = _key[i];
      const bool picked = key >= 0 && (std::size_t)key >= base &&
                          (std::size_t)key < base + (std::size_t)E &&
                          count[(std::size_t)key - base] > 0;
      if (_io[i] == kAheadQueued) {
        w.ahead.erase(std::find(w.ahead.begin(), w.ahead.end(), s));
        if (picked) {
          _io[i] = kQueued;
          w.demand.push_back(s);
        } else {
          _io[i] = kIdle;
          drop_(s);
          ++_stats.ahead_dropped;
        }
      } else if (_io[i] == kRunning) {
        keep.push_back(s);
        continue;
      } else if (_bad[i] != 0) {
        drop_(s);
      }
    }
    _ahead.swap(keep);
  }
  // Hits, wherever they are held; a spare's is admitted by the miss rule.
  std::vector<int> miss, own, claimed;
  for (int e : used) {
    const std::size_t key = base + (std::size_t)e;
    ++_stats.picks;
    const int s = _slot_of[key];
    if (s < 0) {
      miss.push_back(e);
      continue;
    }
    const std::size_t i = (std::size_t)s;
    _used[i] = _step;
    if (_io[i] == kQueued) {
      // A read ahead that had not started: read now, as this step's own.
      _is_ahead[i] = 0;
      own.push_back(s);
      continue;
    }
    ++_stats.hits;
    if (_is_ahead[i] != 0) {
      ++_stats.ahead_hits;
      _is_ahead[i] = 0;
    }
    if (_io[i] == kRunning) { claimed.push_back(s); }
  }
  for (int e : used) {
    const std::size_t key = base + (std::size_t)e;
    const int s = _slot_of[key];
    if (s < 0 || _res[(std::size_t)s] != 0) { continue; }
    if (_n_res < _resident) {
      to_resident_(s);
      ++_stats.admitted;
    } else if (_resident > 0) {
      const int v = victim_();
      if (v >= 0 && _score[key] > _score[(std::size_t)_key[(std::size_t)v]]) {
        to_spare_(v);
        to_resident_(s);
        ++_stats.evicted;
        ++_stats.admitted;
      }
    }
  }
  // Misses, the most selected first: they have the first claim on room.
  std::sort(miss.begin(), miss.end(), [&](int a, int b) {
    return _score[base + (std::size_t)a] > _score[base + (std::size_t)b];
  });
  for (int e : miss) {
    const std::size_t key = base + (std::size_t)e;
    int s = take_spare_();
    if (s < 0 && !_ahead.empty()) {
      // Every spare is this step's or being read ahead: let the reads
      // ahead land, then take one of theirs.
      w.done_cv.wait(lk, [&] {
        for (int a : _ahead) {
          if (_io[(std::size_t)a] == kRunning) { return false; }
        }
        return true;
      });
      for (int a : _ahead) {
        if (_bad[(std::size_t)a] != 0) { drop_(a); }
      }
      _ahead.clear();
      s = take_spare_();
    }
    if (s < 0) {
      if (err != nullptr) {
        *err = std::format("expert cache: layer {} needs more than {} "
                           "spare slots", L, _transient);
      }
      return false;
    }
    drop_(s);
    bool admit = _n_res < _resident;
    int v = -1;
    if (!admit && _resident > 0) {
      v = victim_();
      admit = v >= 0 && _score[key] > _score[(std::size_t)_key[(std::size_t)v]];
    }
    if (admit) {
      if (v >= 0) {
        to_spare_(v);
        ++_stats.evicted;
      }
      to_resident_(s);
      ++_stats.admitted;
    } else {
      ++_stats.transient;
    }
    const std::size_t i = (std::size_t)s;
    _key[i] = (std::int32_t)key;
    _slot_of[key] = s;
    _used[i] = _step;
    _io[i] = kQueued;
    w.demand.push_back(s);
    own.push_back(s);
  }
  // Read ahead for the next step: what it is expected to pick and nothing
  // holds, into spares this step does not use.
  if (ahead != nullptr && ahead->layer >= 0 && ahead->layer < _opt.layers &&
      ahead->ranked != nullptr && ahead->n > 0) {
    const std::size_t nb = (std::size_t)ahead->layer * E;
    _named.assign(ahead->ranked, ahead->ranked + ahead->n);
    _named_layer = ahead->layer;
    for (int k = 0; k < ahead->n; ++k) {
      const int e = ahead->ranked[k];
      if (e < 0 || e >= E || _slot_of[nb + (std::size_t)e] >= 0) { continue; }
      const int s = take_spare_();
      if (s < 0) { break; }
      drop_(s);
      const std::size_t i = (std::size_t)s;
      _key[i] = (std::int32_t)(nb + (std::size_t)e);
      _slot_of[nb + (std::size_t)e] = s;
      _is_ahead[i] = 1;
      _used[i] = _step;
      _io[i] = kAheadQueued;
      w.ahead.push_back(s);
      _ahead.push_back(s);
      ++_stats.ahead_read;
    }
  }
  w.work_cv.notify_all();
  // This step's own reads (this thread reads too), then any it picked that
  // were already under way as reads ahead.
  // A read that failed must not be believed later: its slot is let go.
  auto forget_bad = [&](const std::vector<int>& on) {
    for (int s : on) {
      if (_bad[(std::size_t)s] != 0) {
        to_spare_(s);
        drop_(s);
      }
    }
  };
  if (_record) {
    _blocked.clear();
    for (const std::vector<int>* v : {&own, &claimed}) {
      for (int s : *v) {
        _blocked.push_back(_key[(std::size_t)s] % E);
      }
    }
  }
  bool ok = true;
  double blocked = 0.0;
  if (!own.empty()) {
    const auto t0 = Clock::now();
    ok = wait_(lk, own, true, err);
    blocked += ms_since(t0);
    _stats.read_ms += ms_since(t0);
    if (!ok) { forget_bad(own); }
  }
  if (ok && !claimed.empty()) {
    const auto t0 = Clock::now();
    if (!wait_(lk, claimed, false, nullptr)) {
      // A read ahead that failed: read it again, as this step's own.
      std::vector<int> again;
      for (int s : claimed) {
        if (_bad[(std::size_t)s] != 0) {
          _bad[(std::size_t)s] = 0;
          _io[(std::size_t)s] = kQueued;
          w.demand.push_back(s);
          again.push_back(s);
        }
      }
      ok = wait_(lk, again, true, err);
      if (!ok) { forget_bad(again); }
    }
    blocked += ms_since(t0);
    _stats.wait_ms += ms_since(t0);
  }
  _stats.bytes_read = w.bytes;
  if (!ok) { return false; }
  note_step_(L, rows, blocked);
  for (int p = 0; p < np; ++p) {
    slot[p] = _slot_of[base + (std::size_t)eid[p]];
  }
  return true;
}

int
MoeExpertCache::preload(const std::vector<std::pair<int, int>>& keys,
                        std::string* err)
{
  Pool& w = *_workers;
  std::unique_lock<std::mutex> lk(w.mu);
  ++_step;
  std::vector<int> own;
  for (const auto& [L, e] : keys) {
    if (_n_res >= _resident) { break; }
    if (L < 0 || L >= _opt.layers || e < 0 || e >= _opt.experts) { continue; }
    const std::size_t key = (std::size_t)L * _opt.experts + (std::size_t)e;
    if (_slot_of[key] >= 0) { continue; }
    const int s = take_spare_();
    if (s < 0) { break; }
    drop_(s);
    to_resident_(s);
    const std::size_t i = (std::size_t)s;
    _key[i] = (std::int32_t)key;
    _slot_of[key] = s;
    _used[i] = _step;
    _io[i] = kQueued;
    w.demand.push_back(s);
    own.push_back(s);
  }
  w.work_cv.notify_all();
  const auto t0 = Clock::now();
  const bool ok = wait_(lk, own, true, err);
  _stats.read_ms += ms_since(t0);
  _stats.bytes_read = w.bytes;
  if (!ok) {
    // A slot that did not load must not be believed.
    for (int s : own) {
      drop_(s);
      to_spare_(s);
    }
    return 0;
  }
  return (int)own.size();
}

void
MoeExpertCache::seed_scores(const std::vector<float>& s)
{
  if (s.size() == _score.size()) { _score = s; }
}

std::vector<std::pair<int, int>>
MoeExpertCache::ranked() const
{
  std::vector<std::size_t> ks;
  for (std::size_t k = 0; k < _score.size(); ++k) {
    if (_score[k] > 0.0f) { ks.push_back(k); }
  }
  std::sort(ks.begin(), ks.end(), [&](std::size_t a, std::size_t b) {
    return _score[a] > _score[b];
  });
  std::vector<std::pair<int, int>> out;
  out.reserve(ks.size());
  for (std::size_t k : ks) {
    out.push_back({(int)(k / (std::size_t)_opt.experts),
                   (int)(k % (std::size_t)_opt.experts)});
  }
  return out;
}

}  // namespace vpipe::genai

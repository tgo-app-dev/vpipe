#ifndef VPIPE_GENERATIVE_MODELS_MOE_EXPERT_CACHE_H
#define VPIPE_GENERATIVE_MODELS_MOE_EXPERT_CACHE_H

// THE ROUTED EXPERTS OF A MIXTURE-OF-EXPERTS MODEL, CACHED.
//
// A MoE layer reads only the experts its router picks -- 8 of 256 a token in
// the Qwen3.5 / Drama 3 family -- so most of a 35B model is idle at any
// moment, and a box too small for all of it can still run it: the rest of
// the model resident (and wired), the experts in a cache of SLOTS refilled
// from disk on a miss.
//
// A SLOT IS ONE EXPERT IN EVERY POOL. The pools are laid out exactly like a
// layer's resident expert slab (part k of slot s at s * part_bytes[k]), so
// the model's expert kernels run unchanged: they index the pool by slot as
// they index a slab by expert. resolve() is what turns a step's routed
// expert ids into slot ids.
//
// WHAT STAYS is decided by HOW OFTEN AN EXPERT IS SELECTED. Every (layer,
// expert) carries a score: each step adds the fraction of its rows that
// routed there (a decode token adds 1 per pick; a prefill of n rows adds
// what a token would, spread by the routing), and every `half_life` steps
// all scores halve, so the cache follows the content. A missed expert is
// ADMITTED to the resident set when there is room, or when it scores above
// the least-selected resident expert not needed this step -- which it then
// replaces. Otherwise it is read into a SPARE slot for this step: a one-off
// expert does not evict a hot one.
//
// SLOTS ARE FUNGIBLE. At most `resident` of them form the resident set;
// the rest are spares, and a spare keeps what it was last read with until
// it is taken again (least recently used first), so an expert one step
// read for itself, or a read ahead nobody picked, is still a hit if it is
// asked for before then. An expert found in a spare is admitted by the
// same rule as a miss -- the roles move, the bytes stay.
//
// READING AHEAD. A step may name what the NEXT step (the next layer of the
// same token) is likely to route to, most likely first; those not held are
// read behind this step's own misses and NOT waited for, so they arrive
// while the GPU runs this layer. The next resolve() takes what has arrived,
// waits only for the reads it picks that are still under way, and drops
// the ones that have not started. A read ahead never takes a slot this
// step uses.
//
// HOW MANY TO READ AHEAD IS MEASURED, NOT SET. It depends on the drive and
// the hit rate: where a decode's misses leave the drive idle, reading the
// next layer's likeliest experts while this one runs hides most of a
// step's wait; where they keep it busy, every guess not picked is a read
// the next step waits behind. MEASURED on an M4 Pro at 16 GB, one width
// for both: a Qwen3.6 MoE LLM 1.77x faster reading 8 ahead from the
// internal SSD and 0.92x from an external one, Drama 3's FP8 release
// 0.85x. So the cache compares Options::ahead_widths on the steps it is
// running, by the time each decode step BLOCKED on reads -- the only time
// the width changes (the GPU's work does not depend on it). A step's
// reads answer two widths -- the step before's, whose guess they are, and
// its own, whose reads ahead start beside them on the same drive -- and
// the two do not add: eight reads ahead compete with many misses only
// after a step that read none ahead. So what is compared is a step whose
// predecessor took the SAME width, what a width costs when every step
// takes it. A ROUND draws a width for each block of kBlock steps (hashed,
// so no layer keeps one) until every width has kSamples such steps: the
// text moves a step's cost 2-4x from one passage to the next, and widths
// compared over different passages -- a window of 256 steps each, as a
// first version did -- chose by the text; adding up a width's costs as
// predecessor and as successor, as a second did, chose 0 where 8 was 1.2x.
// The cheapest is kept for kKeep steps (a smaller width within 3% of it
// wins: a read ahead that buys nothing still costs the drive), then the
// round runs again. ahead_width() says the choice.
//
// IT IS NOT EXACT, and the error has a known lean. A width's effects
// outlast its block both ways -- reads still under way slow the next
// block, and experts read ahead stay in the spares as hits a frame later
// -- and an 8-step block catches the first less than the second, so it
// leans toward reading ahead where the drive is busy (an external drive:
// 8 estimated 1.5% ahead of 4 and 11% ahead of 0, where pinned 8 lost to
// 0 by 8%; the 3% rule held it at 4, which cost nothing). Blocks of a
// token measured the other lean (Drama 3 at 32 GB chose 0 against 8's
// 1.2x). MEASURED, the chooser against the best fixed width, M4 Pro: Qwen3.6
// at 16 GB 17.0 against 17.7 tok/s (none: 10.0), the same from the
// external drive 4.5 = 4.5, Drama 3 FP8 at 16 GB 3.4 against 3.5 frames/s,
// at 32 GB 9.4 against 10.2 (none: 8.5). It never ran slower than none.

// A FAR TIER. Given far memory (far-memory.h: another machine's DRAM lent
// over a link), the cache keeps a second set of experts there: those
// selected most often after the ones it holds, by the same scores. A miss
// whose expert is held far is read from there, not from disk. The far
// tier is filled by what the cache lets go anyway -- an expert the
// resident set evicts is WRITTEN to the far side (its slot pinned until
// the write is done) when there is room or it outscores the far tier's
// least selected -- so no read is made for it. Not an expert read for a
// step alone: most are not asked for again, and writing each cost the
// link as much as reading one back (DeepSeek V4.1 Flash, 64 GB cache:
// 35k written for 36k read), and an expert taken into the resident set from
// there leaves it: the two tiers hold different experts. A far call that
// fails lets the tier go; the model goes on from disk.
//
// Misses are read by a pool of threads, concurrently (an SSD answers more
// requests in flight with more bandwidth). The caller has made sure the GPU
// no longer reads any slot but the ones this step will -- in the models
// here, by waiting on the command buffer that computed the routes, which
// also ran everything before it.

#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/far-memory.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace vpipe::metal_compute { class MetalCompute; }

namespace vpipe::genai {

class MoeExpertCache {
public:
  // Reads expert `e` of layer `L` from the checkpoint into its slot: part k
  // at dst[k] (part_bytes[k] bytes). Called from worker threads, for
  // different experts at once.
  using Fill = std::function<bool(int L, int e, std::uint8_t* const* dst,
                                  std::string* err)>;

  struct Options {
    int layers = 0;                    // layer indices 0..layers-1
    int experts = 0;                   // experts a layer
    std::vector<std::size_t> part_bytes;
    int resident = 0;                  // slots kept by frequency
    int transient = 0;                 // spares: a step's misses not kept,
                                       // the reads ahead (>= experts)
    int threads = 6;
    int half_life = 256;               // steps of a layer, scores halve
    // The read-ahead widths a decode step chooses among (empty: never read
    // ahead; one: that width).
    std::vector<int> ahead_widths;
    // The most rows a step may have and still be one the widths are
    // compared on (a decode's 1; a speculative verify's few rows read
    // ahead for each row, and the chain they form is a decode's).
    int ahead_rows = 1;
    // The far tier (see A FAR TIER), and how many of the experts it holds
    // may be written at once (each pins a spare).
    std::shared_ptr<FarMemory> far;
    int far_writes = 4;
    // The Fill skips a part whose dst is null: an expert held far is then
    // read from BOTH -- some parts over the link, the rest from disk at
    // once, shared by how fast each has been -- so a miss waits on the
    // two together, not on one.
    bool fill_skips_null = false;
    // One-off misses written far too, not only what the resident set
    // lets go (VPIPE_MOE_FAR_ADMIT=all|victims overrides: an A/B).
    bool far_one_offs = false;
    // Bytes kept out of the far tier at the far memory's end, the model's
    // own (far_reserved_at(): a far compute's tables).
    std::uint64_t far_reserve = 0;
    // A ZERO SLOT past the others, for REMOTE pairs (see resolve()): asked
    // by a model whose far side may run experts, held only with a far
    // tier, counted in bytes().
    bool zero_slot = false;
  };

  struct Stats {
    std::uint64_t steps = 0;           // resolve() calls
    std::uint64_t picks = 0;           // (step, expert) uses
    std::uint64_t hits = 0;            // ...served without a read of their
                                       // own (resident, spare or ahead)
    std::uint64_t admitted = 0;        // experts taken into the resident set
    std::uint64_t transient = 0;       // misses read for their step alone
    std::uint64_t evicted = 0;         // experts the resident set let go
    std::uint64_t bytes_read = 0;      // the steps' own reads and ahead
    std::uint64_t ahead_picks = 0;     // picks of steps read ahead for
    std::uint64_t ahead_named = 0;     // ...that the read-ahead named
    std::uint64_t ahead_read = 0;      // experts read ahead
    std::uint64_t ahead_hits = 0;      // picks a read-ahead served
    std::uint64_t ahead_dropped = 0;   // read-aheads dropped unread
    std::uint64_t far_reads = 0;       // experts read from the far tier
    std::uint64_t far_writes = 0;      // ...written to it
    std::uint64_t far_evicted = 0;     // ...it let go for another
    std::uint64_t far_dropped = 0;     // writes not made: none could pin
    std::uint64_t far_bytes_read = 0;
    std::uint64_t far_split = 0;       // ...read from both at once
    std::uint64_t far_computed = 0;    // ...run there, not read (Remote)
    double read_ms = 0.0;              // resolve() reading its own misses
    double wait_ms = 0.0;              // ...waiting on reads ahead
    double hit_rate() const {
      return picks > 0 ? (double)hits / (double)picks : 0.0;
    }
  };

  // What the next step is expected to route to: layer `layer`'s experts
  // `ranked[0..n)`, the most likely first.
  struct Ahead {
    int layer = -1;
    const std::int32_t* ranked = nullptr;
    int n = 0;
  };

  // Empty with `err` set on a bad layout or an allocation the box refused.
  static std::unique_ptr<MoeExpertCache> create(
      metal_compute::MetalCompute* mc, const Options& opt, Fill fill,
      std::string* err = nullptr);
  ~MoeExpertCache();

  // One step of layer L: `np` routed (pair) expert ids in, the slot each
  // pair reads out (`slot` may be `eid`). `rows` is the step's token count
  // (a decode 1). Loads whatever is not held before returning; `ahead`, if
  // given, is read ahead behind it (see READING AHEAD).
  //
  // REMOTE (a far tier whose memory runs experts, far-memory.h's
  // far-compute/2; PROTOTYPE): given `remote`, a miss held far that the
  // resident set would not take is not read: its pairs are listed there
  // with its far slot, and handed the ZERO SLOT (every part zero, so the
  // model's kernels compute nothing for them) -- the caller has the far
  // side run it and puts its rows in. The far slots stay pinned until
  // remote_done(). A miss the resident set takes is read as ever, so a hot
  // expert still comes home; so is one whose pairs would take the step
  // past `cap` PAIRS (the caller's room for the far side's rows: at a
  // decode row's one token a pair an expert). A resolve that fails before
  // the pairs are listed unpins them itself; once listed (and `start`
  // called) a later failure -- a read of the step's own -- leaves them
  // listed and pinned: the caller calls remote_done() for any Remote that
  // came back with pairs, whatever resolve() returned.
  // `start`, if set, is called (under the cache's lock: it should only hand
  // the call off) as soon as the step's remote pairs are known -- before
  // resolve() waits on its own reads -- so the far side runs them while
  // the disk reads the rest.
  struct Remote {
    std::vector<int> pair;        // pair index into eid / slot
    std::vector<int> far_slot;    // where its expert is held
    std::function<void(const Remote&)> start;
    int cap = 1;                  // the most pairs run there a step
    void clear() { pair.clear(); far_slot.clear(); }
  };
  bool resolve(int L, const std::int32_t* eid, int np, int rows,
               std::int32_t* slot, std::string* err = nullptr,
               const Ahead* ahead = nullptr, Remote* remote = nullptr);
  // The far slots of a Remote released (the call answered or failed).
  void remote_done(const Remote& r);
  // A far slot's part k: its offset in the far memory.
  std::uint64_t far_part_offset(int far_slot, int k) const
  {
    return (std::uint64_t)far_slot * _slot_bytes + _part_off[(std::size_t)k];
  }
  // The slot whose parts are all zero (-1: none; there is one where
  // Options::zero_slot asks for it and a far tier is open).
  int zero_slot() const { return _zero; }

  // How many experts the next decode step should name to read ahead: the
  // cache's choice among Options::ahead_widths (0 when it has none).
  int ahead_width() const { return _aw.width; }

  // Fill the resident set with these experts, in order (most wanted first),
  // until it is full: a warm start from an earlier run's scores. Reads in
  // parallel; returns how many it loaded.
  int preload(const std::vector<std::pair<int, int>>& keys,
              std::string* err = nullptr);

  // The pool of part k: slot s at s * part_bytes[k].
  const metal_compute::SharedBuffer& pool(int k) const { return _pool[k]; }
  metal_compute::SharedBuffer& pool_mut(int k) { return _pool[k]; }
  int parts() const { return (int)_pool.size(); }
  // Slots the kernels may be handed (resident + transient): what they are
  // told is the expert count.
  int slots() const { return _resident + _transient; }
  int resident_slots() const { return _resident; }
  int resident_used() const { return _n_res; }
  // Whether slot `s` is in the resident set now (the rest are spares).
  bool slot_resident(int s) const {
    return s >= 0 && s < slots() && _res[(std::size_t)s] != 0;
  }
  std::size_t slot_bytes() const { return _slot_bytes; }
  std::size_t bytes() const
  {
    return _slot_bytes * ((std::size_t)slots() + (_zero >= 0 ? 1 : 0));
  }
  // The far tier: its slots, those holding an expert, and whether it is
  // still in use (a failure lets it go).
  int far_slots() const { return _far_slots; }
  // Where Options::far_reserve's bytes begin in the far memory.
  std::uint64_t far_reserved_at() const { return _far_reserved_at; }
  int far_used() const;
  bool far_on() const;
  // Whether the far tier holds layer L's expert e (written, or writing).
  bool far_holds(int L, int e) const;
  // Whether it holds any of layer L's experts ready to read or run (false
  // once the tier is let go).
  bool far_holds_any(int L) const;
  // What reading `bytes` of experts costs now, in seconds: at the paces
  // the cache has measured, the far tier's where it holds them, else the
  // disk's (0 while neither has been measured).
  double read_seconds(std::size_t bytes, bool far) const;

  // The selection scores, [layers][experts]: what an earlier run learned,
  // to seed the next (seed_scores), or to order a preload.
  const std::vector<float>& scores() const { return _score; }
  void seed_scores(const std::vector<float>& s);
  // The (layer, expert) keys by score, highest first (score > 0 only).
  std::vector<std::pair<int, int>> ranked() const;

  const Stats& stats() const { return _stats; }
  void reset_stats() { _stats = {}; }

  // Debug (a study of what to read ahead): with record_steps(true), the
  // experts that held the last step up -- read as its own, or read ahead
  // and still under way when it came.
  void record_steps(bool on) { _record = on; }
  const std::vector<int>& last_blocked() const { return _blocked; }
  // Whether a slot holds layer L's expert e, or is reading it.
  bool holds(int L, int e) const
  {
    return L >= 0 && L < _opt.layers && e >= 0 && e < _opt.experts &&
           _slot_of[(std::size_t)L * _opt.experts + (std::size_t)e] >= 0;
  }

private:
  MoeExpertCache() = default;
  struct Pool;                       // the reader threads
  void worker_();
  void run_(int s, std::unique_lock<std::mutex>& lk, std::uint8_t** dst);
  bool wait_(std::unique_lock<std::mutex>& lk, const std::vector<int>& on,
             bool help, std::string* err);
  void slot_ptrs_(int slot, std::uint8_t** dst) const;
  int take_spare_() const;
  int victim_() const;
  void drop_(int s);
  void to_resident_(int s);
  void to_spare_(int s);
  void age_();
  void note_step_(int L, int rows, double blocked_ms);
  // The far tier, under the pool's lock.
  void far_offer_(int s);
  void far_release_(int key);
  void far_fail_(const std::string& why);
  void far_writer_();

  // Choosing the read-ahead width (see HOW MANY TO READ AHEAD).
  struct AheadChooser {
    static constexpr int kBlock = 8;              // steps a draw lasts
    static constexpr int kSamples = 512;          // steps a width, a round
    static constexpr std::uint64_t kKeep = 16384; // steps a choice is kept
    std::vector<int> widths;
    // Blocked ms of steps whose predecessor took the same width, and how
    // many, per width.
    std::vector<double> ms;
    std::vector<std::uint64_t> n;
    int block = 0;                         // steps left in this draw
    int width = 0;                         // in force for the next step
    int cur = 0;                           // its index in `widths`
    bool round = true;
    std::uint64_t keep = 0;                // steps left keeping it
    std::uint64_t draws = 0;
    int prev_layer = -2, prev = -1;        // the step before: its width
  };
  AheadChooser _aw;

  metal_compute::MetalCompute* _mc = nullptr;
  Options _opt;
  Fill _fill;
  std::vector<metal_compute::SharedBuffer> _pool;
  std::size_t _slot_bytes = 0;
  int _resident = 0, _transient = 0, _n_res = 0;
  int _zero = -1;   // the zero slot: past every other, never handed out
  // [keys = layers * experts]: the slot holding it (or being read into
  // it), or -1; and its score.
  std::vector<std::int32_t> _slot_of;
  std::vector<float> _score;
  // [slots]: the key it holds, or -1; in the resident set; the step that
  // last used it (this step's are pinned); read ahead and not yet picked.
  std::vector<std::int32_t> _key;
  std::vector<std::uint8_t> _res, _is_ahead;
  std::vector<std::uint64_t> _used;
  // [slots], under the pool's lock: its read's state, and whether the
  // last one failed.
  std::vector<std::uint8_t> _io, _bad;
  // The spares, as a set (_spare_at[s] its index, or -1).
  std::vector<int> _spare, _spare_at;
  // Slots read ahead whose read the cache has not yet seen settle.
  std::vector<int> _ahead;
  // The last read-ahead's experts and layer (for Stats::ahead_named).
  std::vector<std::int32_t> _named;
  int _named_layer = -1;
  bool _record = false;
  std::vector<int> _blocked;
  std::uint64_t _step = 0, _layer0_steps = 0;
  // The far tier: [far slots] the key held (-1 free), written yet, reads
  // in flight; [keys] the far slot holding it; the writes queued (slot,
  // far slot); [slots] pinned by a write.
  std::shared_ptr<FarMemory> _far;
  int _far_slots = 0;
  std::uint64_t _far_reserved_at = 0;
  std::vector<std::int32_t> _far_key, _far_of;
  std::vector<std::uint8_t> _far_ready;
  std::vector<int> _far_reading, _far_free;
  std::vector<std::pair<int, int>> _far_queue;
  int _far_inflight = 0, _n_pinned = 0;
  bool _far_dead = false;
  bool _far_one_offs = false;   // a one-off miss written far too
  // An expert taken home keeps its far copy, let go only when the far
  // tier needs the room (copies of experts held home first): no write
  // when it is evicted again.
  bool _far_lazy = false;
  std::vector<std::uint8_t> _pinned;
  std::vector<std::size_t> _part_off;
  std::vector<std::thread> _far_threads;
  // Seconds a byte from each source, as measured (0 not yet): how an
  // expert held far is shared between the link and the disk.
  double _far_spb = 0.0, _disk_spb = 0.0;
  Stats _stats;
  std::unique_ptr<Pool> _workers;
};

}  // namespace vpipe::genai

#endif  // VPIPE_GENERATIVE_MODELS_MOE_EXPERT_CACHE_H

// The routed-expert cache (generative-models/moe-expert-cache.h): its
// policy on synthetic experts, and a Mixture-of-Experts LM run through it
// against the same model resident -- BYTE-IDENTICAL, since the cache only
// changes where the expert kernels find an expert's weights.
//
// Env:
//   VPIPE_QWEN35_MOE_TEST_MODEL_PATH  a Qwen3.5 / 3.6 35B-A3B MLX checkpoint
//                                     (affine experts) for the model test.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/session.h"
#include "generative-models/chat-template.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/loaded-language-model.h"
#include "generative-models/model-loader.h"
#include "generative-models/tokenizer.h"
#include "generative-models/moe-expert-cache.h"
#include "generative-models/qwen3/metal-qwen-model.h"

#include <mach/mach.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <thread>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::MetalCompute;

namespace {

// An expert's bytes: its (layer, expert) in every word, so a slot proves
// which expert it holds.
std::uint32_t
tag_(int L, int e)
{
  return 0x10000u * (std::uint32_t)(L + 1) + (std::uint32_t)e;
}

std::uint32_t
slot_tag_(const MoeExpertCache& c, int slot)
{
  std::uint32_t v;
  std::memcpy(&v,
              static_cast<const std::uint8_t*>(c.pool(0).contents()) +
                  (std::size_t)slot * 16,
              4);
  return v;
}

// Two loads compared byte for byte must run the same kernels, and two
// tuners pick theirs by timing at every load: the decode attention
// (VPIPE_QWEN_GQA_AUTOTUNE) and, at head dim 256, the prefill attention
// (VPIPE_QWEN_PREFILL_AUTOTUNE). Pinned for the scope.
struct PinnedTuners {
  static constexpr const char* kVars[2] = {"VPIPE_QWEN_GQA_AUTOTUNE",
                                           "VPIPE_QWEN_PREFILL_AUTOTUNE"};
  std::string was[2];
  PinnedTuners()
  {
    for (int i = 0; i < 2; ++i) {
      const char* v = std::getenv(kVars[i]);
      was[i] = v != nullptr ? v : "";
      ::setenv(kVars[i], "0", 1);
    }
  }
  ~PinnedTuners()
  {
    for (int i = 0; i < 2; ++i) {
      if (was[i].empty()) {
        ::unsetenv(kVars[i]);
      } else {
        ::setenv(kVars[i], was[i].c_str(), 1);
      }
    }
  }
};

}  // namespace

// Admission by room, then by how often an expert is selected: a one-off
// miss is read into a spare slot and evicts nothing (and is found there
// while the spare keeps it); an expert picked
// again and again overtakes the least-selected resident one; an expert the
// step itself needs is never the victim; scores halve every half_life
// steps; a preload fills the resident set in the order given.
TEST(moe_expert_cache, policy)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  int reads = 0;
  MoeExpertCache::Options o;
  o.layers = 2;
  o.experts = 8;
  o.part_bytes = {16, 8};
  o.resident = 4;
  o.transient = 8;
  o.threads = 3;
  o.half_life = 1000;
  std::string err;
  auto c = MoeExpertCache::create(
      mc, o,
      [&](int L, int e, std::uint8_t* const* dst, std::string*) {
        const std::uint32_t t = tag_(L, e);
        for (int i = 0; i < 4; ++i) { std::memcpy(dst[0] + 4 * i, &t, 4); }
        std::memcpy(dst[1], &t, 4);
        __atomic_add_fetch(&reads, 1, __ATOMIC_RELAXED);
        return true;
      },
      &err);
  ASSERT_TRUE(c != nullptr);
  if (c == nullptr) { return; }
  EXPECT_TRUE(c->slots() == 12 && c->resident_slots() == 4);
  auto step = [&](int L, std::vector<std::int32_t> eid, int rows) {
    std::vector<std::int32_t> slot(eid.size(), -1);
    EXPECT_TRUE(c->resolve(L, eid.data(), (int)eid.size(), rows,
                           slot.data(), &err));
    for (std::size_t p = 0; p < eid.size(); ++p) {
      EXPECT_TRUE(slot[p] >= 0 && slot[p] < c->slots());
      EXPECT_TRUE(slot_tag_(*c, slot[p]) == tag_(L, eid[p]));
    }
    return slot;
  };
  // Room: four experts admitted, each read once.
  step(0, {0, 1, 2, 3}, 1);
  EXPECT_TRUE(c->resident_used() == 4 && reads == 4);
  // Hits: nothing read.
  step(0, {3, 2, 1, 0}, 1);
  EXPECT_TRUE(reads == 4 && c->stats().hits == 4);
  // A one-off: a spare, no eviction.
  auto s = step(0, {5}, 1);
  EXPECT_TRUE(!c->slot_resident(s[0]) && c->stats().evicted == 0);
  // Expert 6 picked three times outscores the residents (2 each).
  step(0, {6}, 1);
  step(0, {6}, 1);
  EXPECT_TRUE(c->stats().evicted == 0);
  s = step(0, {6}, 1);
  EXPECT_TRUE(c->slot_resident(s[0]) && c->stats().evicted == 1);
  // ...found in the spare it was read into the first time, not read again.
  EXPECT_TRUE(reads == 6);
  // Pinned: steps that need every resident expert AND a newcomer leave no
  // victim -- the newcomer stays a spare each time, however its score
  // climbs past theirs.
  const std::uint64_t ev = c->stats().evicted;
  for (int i = 0; i < 4; ++i) {
    s = step(0, {6, 1, 2, 3, 4}, 1);
    EXPECT_TRUE(!c->slot_resident(s[4]));
  }
  EXPECT_TRUE(c->stats().evicted == ev);
  // A prefill: many rows, the same pair counts spread over them.
  step(0, {0, 0, 1, 2, 3, 4}, 6);
  // Hit rate is reported over picks.
  EXPECT_TRUE(c->stats().picks > 0 && c->stats().hit_rate() > 0.0);
  // Scores halve every half_life steps of layer 0.
  {
    MoeExpertCache::Options o2 = o;
    o2.half_life = 2;
    auto c2 = MoeExpertCache::create(
        mc, o2,
        [&](int, int, std::uint8_t* const*, std::string*) { return true; },
        &err);
    ASSERT_TRUE(c2 != nullptr);
    std::vector<std::int32_t> e{1}, sl(1);
    ASSERT_TRUE(c2->resolve(0, e.data(), 1, 1, sl.data(), &err));
    const float before = c2->scores()[1];
    ASSERT_TRUE(c2->resolve(0, e.data(), 1, 1, sl.data(), &err));
    EXPECT_TRUE(before == 1.0f && c2->scores()[1] == 1.0f);   // (1+1)/2
  }
  // A preload in the order given, until the resident set is full.
  {
    auto c3 = MoeExpertCache::create(
        mc, o,
        [&](int L, int e2, std::uint8_t* const* dst, std::string*) {
          const std::uint32_t t = tag_(L, e2);
          std::memcpy(dst[0], &t, 4);
          return true;
        },
        &err);
    ASSERT_TRUE(c3 != nullptr);
    const int n = c3->preload({{1, 5}, {0, 2}, {1, 1}, {0, 7}, {0, 3}}, &err);
    EXPECT_TRUE(n == 4 && c3->resident_used() == 4);
    std::vector<std::int32_t> e{5}, sl(1);
    ASSERT_TRUE(c3->resolve(1, e.data(), 1, 1, sl.data(), &err));
    EXPECT_TRUE(c3->stats().hits == 1 && slot_tag_(*c3, sl[0]) == tag_(1, 5));
  }
  // A routed id out of range is refused, not read.
  {
    std::vector<std::int32_t> e{9}, sl(1);
    EXPECT_FALSE(c->resolve(0, e.data(), 1, 1, sl.data(), &err));
  }
}

// Memory standing in for another machine's: a byte array, its calls
// counted, failing on demand.
class TestFarMemory final : public FarMemory {
public:
  explicit TestFarMemory(std::size_t bytes) : _mem(bytes) {}
  std::uint64_t bytes() const override { return _mem.size(); }
  bool read(const FarPiece* p, std::size_t n, std::string* err) override
  {
    if (fail.load()) {
      if (err != nullptr) { *err = "the link is down"; }
      return false;
    }
    for (std::size_t i = 0; i < n; ++i) {
      std::memcpy(p[i].local, _mem.data() + p[i].offset, p[i].bytes);
    }
    reads.fetch_add(1);
    return true;
  }
  bool write(const FarPiece* p, std::size_t n, std::string*) override
  {
    for (std::size_t i = 0; i < n; ++i) {
      std::memcpy(_mem.data() + p[i].offset, p[i].local, p[i].bytes);
    }
    writes.fetch_add(1);
    return true;
  }
  std::atomic<bool> fail{false};
  std::atomic<int> reads{0}, writes{0};

private:
  std::vector<std::uint8_t> _mem;
};

// THE FAR TIER: an expert read from disk for a step alone, or evicted from
// the resident set, is written to far memory; asked for again it is read
// from there, not from disk, with its own bytes; taken into the resident
// set it leaves the far tier; a far read that fails lets the tier go and
// the expert is read from disk.
// Both ways: and with `split`, the fill skipping a null part, an expert
// held far is read from both at once -- part 0 (the larger) over the
// link, part 1 from disk. Only part 0's disk reads are counted: the same
// counts either way.
TEST(moe_expert_cache, far_tier)
{
  for (bool split : {false, true}) {
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  std::atomic<int> reads{0};
  auto far = std::make_shared<TestFarMemory>(6 * 24);
  MoeExpertCache::Options o;
  o.layers = 1;
  o.experts = 16;
  o.part_bytes = {16, 8};
  o.resident = 2;
  o.transient = 4;
  o.threads = 2;
  o.half_life = 1000;
  o.far = far;
  o.fill_skips_null = split;
  o.far_one_offs = true;
  std::string err;
  auto c = MoeExpertCache::create(
      mc, o,
      [&](int L, int e, std::uint8_t* const* dst, std::string*) {
        const std::uint32_t t = tag_(L, e);
        if (dst[0] != nullptr) {
          for (int i = 0; i < 4; ++i) { std::memcpy(dst[0] + 4 * i, &t, 4); }
          reads.fetch_add(1);
        }
        if (dst[1] != nullptr) { std::memcpy(dst[1], &t, 4); }
        return true;
      },
      &err);
  ASSERT_TRUE(c != nullptr);
  if (c == nullptr) { return; }
  EXPECT_TRUE(c->far_slots() == 6 && c->far_on());
  auto step = [&](std::vector<std::int32_t> eid) {
    std::vector<std::int32_t> slot(eid.size(), -1);
    EXPECT_TRUE(c->resolve(0, eid.data(), (int)eid.size(), 1, slot.data(),
                           &err));
    for (std::size_t p = 0; p < eid.size(); ++p) {
      EXPECT_TRUE(slot_tag_(*c, slot[p]) == tag_(0, eid[p]));
      std::uint32_t t1;
      std::memcpy(&t1,
                  static_cast<const std::uint8_t*>(c->pool(1).contents()) +
                      (std::size_t)slot[p] * 8,
                  4);
      EXPECT_TRUE(t1 == tag_(0, eid[p]));
    }
  };
  auto written = [&](int n) {
    for (int i = 0; i < 1000 && far->writes.load() < n; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return far->writes.load() == n;
  };
  // Resident by room: nothing written far.
  step({0, 1});
  EXPECT_TRUE(reads.load() == 2 && far->writes.load() == 0);
  // One-offs: read into spares, each written far.
  for (int e = 2; e <= 6; ++e) {
    step({e});
    EXPECT_TRUE(written(e - 1));
  }
  EXPECT_TRUE(reads.load() == 7 && c->far_used() == 5);
  EXPECT_TRUE(c->far_holds(0, 2) && c->far_holds(0, 6));
  // Expert 2, its spare long taken: read from the far tier, not disk --
  // and now outscoring a resident, admitted: it leaves the far tier, and
  // the resident it evicts is written there.
  step({2});
  EXPECT_TRUE(reads.load() == 7 && far->reads.load() == 1);
  EXPECT_TRUE(c->stats().far_reads == 1);
  EXPECT_TRUE(c->stats().far_split == (split ? 1u : 0u));
  EXPECT_TRUE(!c->far_holds(0, 2));
  EXPECT_TRUE(written(6));
  EXPECT_TRUE(c->far_holds(0, 0) || c->far_holds(0, 1));
  // A far read that fails: the expert read from disk, the tier let go.
  far->fail.store(true);
  step({3});
  EXPECT_TRUE(reads.load() == 8 && !c->far_on());
  step({5});   // still in a spare
  EXPECT_TRUE(reads.load() == 8 && far->reads.load() == 1);
  }
  // By default only what the resident set lets go is written far: one-off
  // misses are not; an evicted resident is.
  {
    Session sess;
    MetalCompute* mc = sess.metal_compute();
    if (mc == nullptr) { return; }
    auto far = std::make_shared<TestFarMemory>(6 * 24);
    MoeExpertCache::Options o;
    o.layers = 1;
    o.experts = 16;
    o.part_bytes = {16, 8};
    o.resident = 2;
    o.transient = 4;
    o.threads = 2;
    o.half_life = 1000;
    o.far = far;
    std::string err;
    auto c = MoeExpertCache::create(
        mc, o,
        [&](int, int, std::uint8_t* const*, std::string*) { return true; },
        &err);
    ASSERT_TRUE(c != nullptr);
    std::vector<std::int32_t> sl(2);
    std::vector<std::int32_t> e01{0, 1}, e2{2};
    ASSERT_TRUE(c->resolve(0, e01.data(), 2, 1, sl.data(), &err));
    ASSERT_TRUE(c->resolve(0, e2.data(), 1, 1, sl.data(), &err));
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_TRUE(far->writes.load() == 0 && !c->far_holds(0, 2));
    // Picked again, 2 outscores a resident and evicts it: that one is
    // written far.
    ASSERT_TRUE(c->resolve(0, e2.data(), 1, 1, sl.data(), &err));
    for (int i = 0; i < 1000 && far->writes.load() < 1; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_TRUE(far->writes.load() == 1 &&
                (c->far_holds(0, 0) || c->far_holds(0, 1)));
  }
}

// READING AHEAD: a step names what the next one will pick; those it does
// not hold are read behind it, the next step finds them (no read of its
// own), and the ones it does not pick are dropped if still unread or kept
// in their spares if read. A read ahead never takes a slot the step that
// issued it uses -- the GPU reads those until the next step.
TEST(moe_expert_cache, reads_ahead)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  std::atomic<int> reads{0};
  MoeExpertCache::Options o;
  o.layers = 3;
  o.experts = 16;
  o.part_bytes = {16, 8};
  o.resident = 2;
  o.transient = 16;
  o.threads = 4;
  std::string err;
  auto c = MoeExpertCache::create(
      mc, o,
      [&](int L, int e, std::uint8_t* const* dst, std::string*) {
        const std::uint32_t t = tag_(L, e);
        for (int i = 0; i < 4; ++i) { std::memcpy(dst[0] + 4 * i, &t, 4); }
        std::memcpy(dst[1], &t, 4);
        ++reads;
        return true;
      },
      &err);
  ASSERT_TRUE(c != nullptr);
  if (c == nullptr) { return; }
  // Layer 0 reads its own; layer 1's guess is read behind it.
  std::vector<std::int32_t> e0{0, 1}, s0(2), g1{3, 4, 5};
  MoeExpertCache::Ahead a{1, g1.data(), 3};
  ASSERT_TRUE(c->resolve(0, e0.data(), 2, 1, s0.data(), &err, &a));
  EXPECT_TRUE(c->stats().ahead_read == 3);
  // The read-ahead never landed in step 0's slots.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_TRUE(slot_tag_(*c, s0[0]) == tag_(0, 0) &&
              slot_tag_(*c, s0[1]) == tag_(0, 1));
  // Layer 1 picks 3 and 4 (read ahead) and 7 (not): one read of its own.
  const int before = reads.load();
  std::vector<std::int32_t> e1{3, 7, 4}, s1(3);
  ASSERT_TRUE(c->resolve(1, e1.data(), 3, 1, s1.data(), &err));
  EXPECT_TRUE(reads.load() == before + 1);
  for (int p = 0; p < 3; ++p) {
    const std::size_t u = (std::size_t)p;
    EXPECT_TRUE(slot_tag_(*c, s1[u]) == tag_(1, e1[u]));
  }
  const auto& st = c->stats();
  EXPECT_TRUE(st.ahead_hits == 2 && st.ahead_picks == 3 &&
              st.ahead_named == 2);
  // Expert 5, read ahead and not picked, is still held: a hit later.
  std::vector<std::int32_t> e5{5}, s5(1);
  const std::uint64_t hits = st.hits;
  ASSERT_TRUE(c->resolve(1, e5.data(), 1, 1, s5.data(), &err));
  EXPECT_TRUE(c->stats().hits == hits + 1 && reads.load() == before + 1 &&
              slot_tag_(*c, s5[0]) == tag_(1, 5));
  // Spares run short: the step finds its three experts in three of four
  // spares (idle, read a step before), the guess names three -- one is
  // read ahead, and the step's own slots are left alone.
  {
    MoeExpertCache::Options o2 = o;
    o2.resident = 0;
    o2.transient = 4;
    auto c2 = MoeExpertCache::create(
        mc, o2,
        [&](int L, int e, std::uint8_t* const* dst, std::string*) {
          const std::uint32_t t = tag_(L, e);
          for (int i = 0; i < 4; ++i) { std::memcpy(dst[0] + 4 * i, &t, 4); }
          std::memcpy(dst[1], &t, 4);
          return true;
        },
        &err);
    ASSERT_TRUE(c2 != nullptr);
    if (c2 == nullptr) { return; }
    std::vector<std::int32_t> e{0, 1, 2}, sl(3), g{4, 5, 6};
    ASSERT_TRUE(c2->resolve(0, e.data(), 3, 1, sl.data(), &err));
    MoeExpertCache::Ahead a2{1, g.data(), 3};
    ASSERT_TRUE(c2->resolve(0, e.data(), 3, 1, sl.data(), &err, &a2));
    EXPECT_TRUE(c2->stats().hits == 3);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_TRUE(c2->stats().ahead_read == 1);
    for (int p = 0; p < 3; ++p) {
      EXPECT_TRUE(slot_tag_(*c2, sl[(std::size_t)p]) == tag_(0, p));
    }
  }
}

// Reads ahead still under way when the next step comes, steps picking
// some of them and not others, the resident set churning: every pick reads
// the expert it asked for, and no slot a step handed out changes before the
// next step (a slow reader makes the reads ahead overlap the steps).
TEST(moe_expert_cache, reads_ahead_under_way)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MoeExpertCache::Options o;
  o.layers = 4;
  o.experts = 32;
  o.part_bytes = {16, 8};
  o.resident = 24;
  o.transient = 32;
  o.threads = 4;
  std::string err;
  auto c = MoeExpertCache::create(
      mc, o,
      [&](int L, int e, std::uint8_t* const* dst, std::string*) {
        std::this_thread::sleep_for(std::chrono::microseconds(300));
        const std::uint32_t t = tag_(L, e);
        for (int i = 0; i < 4; ++i) { std::memcpy(dst[0] + 4 * i, &t, 4); }
        std::memcpy(dst[1], &t, 4);
        return true;
      },
      &err);
  ASSERT_TRUE(c != nullptr);
  if (c == nullptr) { return; }
  std::uint32_t rng = 12345;
  auto rnd = [&](int n) {
    rng = rng * 1664525u + 1013904223u;
    return (int)((rng >> 8) % (std::uint32_t)n);
  };
  // A skewed router: low ids far likelier, as real routes are.
  auto pick = [&] { return std::min(rnd(32), rnd(32)); };
  std::vector<std::int32_t> prev_slot, prev_tag;
  int bad = 0, moved = 0;
  std::vector<std::int32_t> guess;
  for (int t = 0; t < 400; ++t) {
    const int L = t % o.layers;
    const int rows = t % 23 == 0 ? 6 : 1;
    std::vector<std::int32_t> eid;
    for (int p = 0; p < 4 * rows; ++p) { eid.push_back(pick()); }
    // The guess for the next step, drawn from the same skew.
    std::vector<std::int32_t> next;
    for (int k = 0; k < 6; ++k) { next.push_back(pick()); }
    MoeExpertCache::Ahead a{(L + 1) % o.layers, next.data(), 6};
    // The last step's slots held still while its reads ahead ran: the GPU
    // reads them until this step begins.
    for (std::size_t i = 0; i < prev_slot.size(); ++i) {
      moved += slot_tag_(*c, prev_slot[i]) != (std::uint32_t)prev_tag[i];
    }
    prev_slot.clear();
    prev_tag.clear();
    std::vector<std::int32_t> slot(eid.size());
    if (!c->resolve(L, eid.data(), (int)eid.size(), rows, slot.data(), &err,
                    t % 3 == 2 ? nullptr : &a)) {
      ++bad;
      continue;
    }
    for (std::size_t p = 0; p < eid.size(); ++p) {
      bad += slot_tag_(*c, slot[p]) != tag_(L, eid[p]);
    }
    // Let some reads ahead still be running when the next step comes.
    std::this_thread::sleep_for(std::chrono::microseconds(rnd(600)));
    for (std::size_t p = 0; p < eid.size(); ++p) {
      prev_slot.push_back(slot[p]);
      prev_tag.push_back((std::int32_t)tag_(L, eid[p]));
    }
  }
  const auto& st = c->stats();
  std::printf("[moe-cache] 400 steps: %llu picks, %.1f%% hits, %llu read "
              "ahead, %llu of them picked, %llu dropped unread\n",
              (unsigned long long)st.picks, 100.0 * st.hit_rate(),
              (unsigned long long)st.ahead_read,
              (unsigned long long)st.ahead_hits,
              (unsigned long long)st.ahead_dropped);
  EXPECT_TRUE(bad == 0 && moved == 0);
  EXPECT_TRUE(st.ahead_read > 0 && st.ahead_hits > 0);
}

// THE WIDTH IS THE CACHE'S CHOICE: decode steps whose misses take 0.3 ms
// to read, 0.4 ms of "GPU" between them, and a guess that is always right
// -- reading ahead hides every miss, so the cache settles on it; with a
// guess that is never right it only adds reads, and the cache settles on
// none. A round is 512 steps a width whose predecessor took the same
// one, drawn in blocks of 8; two layers, so half the steps chain.
TEST(moe_expert_cache, chooses_its_read_ahead)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  for (const bool right : {true, false}) {
    MoeExpertCache::Options o;
    o.layers = 2;
    o.experts = 64;
    o.part_bytes = {16};
    o.resident = 0;
    o.transient = 8;
    o.threads = 3;
    o.ahead_widths = {0, 2};
    std::string err;
    auto c = MoeExpertCache::create(
        mc, o,
        [&](int, int, std::uint8_t* const*, std::string*) {
          std::this_thread::sleep_for(std::chrono::microseconds(300));
          return true;
        },
        &err);
    ASSERT_TRUE(c != nullptr);
    if (c == nullptr) { return; }
    // Step t picks two experts no spare still holds (the set rotates
    // through all 64); the guess names step t+1's, or two others.
    auto picks = [](int t) {
      return std::vector<std::int32_t>{(4 * t) % 64, (4 * t + 1) % 64};
    };
    for (int t = 0; t < 3000; ++t) {
      std::vector<std::int32_t> e = picks(t), sl(2);
      std::vector<std::int32_t> g = picks(t + 1);
      if (!right) { g = {(g[0] + 2) % 64, (g[1] + 2) % 64}; }
      const int w = c->ahead_width();
      MoeExpertCache::Ahead a{(t + 1) % 2, g.data(), w};
      ASSERT_TRUE(c->resolve(t % 2, e.data(), 2, 1, sl.data(), &err,
                             w > 0 ? &a : nullptr));
      std::this_thread::sleep_for(std::chrono::microseconds(400));
    }
    std::printf("[moe-cache] a guess %s: width %d after a round, %llu read "
                "ahead, %llu picked\n", right ? "always right" :
                "never right", c->ahead_width(),
                (unsigned long long)c->stats().ahead_read,
                (unsigned long long)c->stats().ahead_hits);
    EXPECT_TRUE(c->ahead_width() == (right ? 2 : 0));
  }
}

// A 35B-A3B MoE LM with its routed experts in a 2 GB cache -- 6% of them
// resident, every prefill layer through the transient slots -- against the
// same checkpoint resident: the prompt's logits and 32 greedy tokens
// BYTE-IDENTICAL. The attention tuners are pinned: left to vote they may
// pick other kernels in the second load.
TEST(moe_expert_cache, qwen_moe_is_resident)
{
  const char* path = std::getenv("VPIPE_QWEN35_MOE_TEST_MODEL_PATH");
  if (path == nullptr || *path == '\0') { return; }
  const PinnedTuners pinned;
  ModelLoader loader(nullptr);
  auto cfg = loader.load_config(path);
  ASSERT_TRUE(cfg.has_value());
  if (!cfg.has_value()) { return; }
  std::vector<std::int32_t> ids;
  for (int i = 0; i < 300; ++i) { ids.push_back((i * 131 + 7) % 2000 + 10); }
  struct Run {
    std::vector<float> logits;
    std::vector<std::int32_t> gen;
    std::size_t held = 0;
  };
  auto run = [&](std::size_t cache) {
    Run r;
    Session sess;
    MetalCompute* mc = sess.metal_compute();
    if (mc == nullptr) { return r; }
    (void)mc->memory_budget_settled(15000);
    auto mcfg = MetalQwenModel::config_from(*cfg);
    mcfg.page_tokens = 512;
    mcfg.max_pages = 8;
    mcfg.expert_cache_bytes = cache;
    auto m = MetalQwenModel::load(path, mc, mcfg);
    if (m == nullptr) { return r; }
    r.held = m->resident_bytes();
    r.logits = m->prefill(ids);
    std::int32_t t = 0;
    float bv = -1e30f;
    for (std::size_t i = 0; i < r.logits.size(); ++i) {
      if (r.logits[i] > bv) { bv = r.logits[i]; t = (std::int32_t)i; }
    }
    r.gen.push_back(t);
    for (int i = 0; i < 32; ++i) {
      t = m->forward_argmax(r.gen.back());
      if (t < 0) { break; }
      r.gen.push_back(t);
    }
    if (const MoeExpertCache* c = m->expert_cache()) {
      const auto& st = c->stats();
      std::printf("[moe_cache] %zu MB cache: %llu picks, hit rate %.1f%%, "
                  "%llu admitted, %llu transient, %llu evicted, %.0f MB "
                  "read in %.0f ms\n", cache >> 20,
                  (unsigned long long)st.picks, 100.0 * st.hit_rate(),
                  (unsigned long long)st.admitted,
                  (unsigned long long)st.transient,
                  (unsigned long long)st.evicted, st.bytes_read / 1048576.0,
                  st.read_ms);
    }
    return r;
  };
  const Run a = run(0);
  const Run b = run((std::size_t)2048 << 20);
  ASSERT_TRUE(!a.logits.empty() && !b.logits.empty());
  std::printf("[moe_cache] held %zu MB resident, %zu MB cached\n",
              a.held >> 20, b.held >> 20);
  EXPECT_TRUE(a.logits.size() == b.logits.size() &&
              std::memcmp(a.logits.data(), b.logits.data(),
                          a.logits.size() * 4) == 0);
  EXPECT_TRUE(a.gen == b.gen && a.gen.size() == 33);
  EXPECT_TRUE(b.held < a.held / 2);
}

// THE CACHE'S TURN AS A HANDSHAKE (gpu-host-handshake.h): where this GPU
// takes it, a cached Qwen MoE hands each layer's picks over inside the
// running GPU work on every decode path -- forward(), decode_step_fast(),
// the pipelined decode, a batched step of two branches -- and stays the
// resident model's to the bit; so does a run with every 13th turn
// abandoned (the GPU gives up and runs on stale slots; the step is redone
// with commits from the state it found: its linear layers' state written
// out of place, its input rows kept). VPIPE_QWEN_HANDSHAKE=0 runs the
// cached arms at a commit and wait a layer (the same bits).
TEST(moe_expert_cache, the_handshake_keeps_the_resident_bits)
{
  const char* path = std::getenv("VPIPE_QWEN35_MOE_TEST_MODEL_PATH");
  if (path == nullptr || *path == '\0') { return; }
  const PinnedTuners pinned;
  ModelLoader loader(nullptr);
  auto cfg = loader.load_config(path);
  ASSERT_TRUE(cfg.has_value());
  if (!cfg.has_value()) { return; }
  std::vector<std::int32_t> ids;
  for (int i = 0; i < 300; ++i) { ids.push_back((i * 131 + 7) % 2000 + 10); }
  auto argmax = [](const float* p, std::size_t n) {
    std::int32_t t = 0;
    for (std::size_t i = 1; i < n; ++i) {
      if (p[i] > p[t]) { t = (std::int32_t)i; }
    }
    return t;
  };
  struct Run {
    std::vector<float> lg;
    std::vector<std::int32_t> toks;
    bool ok = false, hs = false, cached = false;
    int redos = 0;
    double grew_mb = 0;   // the device's growth over the forward() steps
  };
  auto run = [&](std::size_t cache, int drop) {
    Run r;
    Session sess;
    MetalCompute* mc = sess.metal_compute();
    if (mc == nullptr) { return r; }
    (void)mc->memory_budget_settled(15000);
    auto mcfg = MetalQwenModel::config_from(*cfg);
    mcfg.page_tokens = 512;
    mcfg.max_pages = 16;
    mcfg.expert_cache_bytes = cache;
    auto m = MetalQwenModel::load(path, mc, mcfg);
    if (m == nullptr) { return r; }
    r.hs = m->expert_handshake();
    r.cached = m->expert_cache() != nullptr;
    m->xhs_drop_every(drop);
    const ContextId root = m->root_context();
    const std::size_t V = (std::size_t)mcfg.vocab;
    std::vector<float> lg = m->prefill(ids);
    if (lg.size() != V) { return r; }
    r.lg = lg;
    std::int32_t t = argmax(lg.data(), V);
    const std::size_t a0 = mc->memory_budget().allocated;
    for (int i = 0; i < 4; ++i) {           // forward()
      lg = m->forward(t);
      if (lg.size() != V) { return r; }
      r.lg.insert(r.lg.end(), lg.begin(), lg.end());
      t = argmax(lg.data(), V);
      r.toks.push_back(t);
    }
    r.grew_mb = ((double)mc->memory_budget().allocated - (double)a0) /
                1048576.0;
    for (int i = 0; i < 4; ++i) {           // decode_step_fast()
      t = m->decode_step_fast(root, t);
      if (t < 0) { return r; }
      r.toks.push_back(t);
    }
    std::vector<std::int32_t> pd;           // the pipelined decode
    if (!m->decode_pipelined(root, t, 6, pd) || pd.size() != 6) { return r; }
    r.toks.insert(r.toks.end(), pd.begin(), pd.end());
    // a batched step of two branches
    ContextManager* cx = m->context_manager();
    const std::vector<ContextId> bc{cx->branch(root), cx->branch(root)};
    std::vector<std::int32_t> in{pd.back(), (pd.back() + 1) % 1000 + 10};
    for (int i = 0; i < 3; ++i) {
      std::vector<float> out;
      if (!m->decode_batched_step(bc, in, {}, out) ||
          out.size() != 2 * V) {
        return r;
      }
      r.lg.insert(r.lg.end(), out.begin(), out.end());
      in = {argmax(out.data(), V), argmax(out.data() + V, V)};
      r.toks.insert(r.toks.end(), in.begin(), in.end());
    }
    r.redos = m->xhs_redos();
    r.ok = true;
    return r;
  };
  const Run a = run(0, 0);
  const Run b = run((std::size_t)2048 << 20, 0);
  ASSERT_TRUE(a.ok && b.ok);
  if (!a.ok || !b.ok) { return; }
  auto same = [&](const Run& x) {
    return x.lg.size() == a.lg.size() && x.toks == a.toks &&
           std::memcmp(x.lg.data(), a.lg.data(), a.lg.size() * 4) == 0;
  };
  // A context's linear-layer state, a slot: what the handshake's ring
  // holds twice more (run_bytes books it).
  std::size_t slot = 0;
  {
    const auto mc0 = MetalQwenModel::config_from(*cfg);
    for (int L = 0; L < mc0.n_layers; ++L) {
      if (mc0.layer_is_full(L)) { continue; }
      slot += (std::size_t)std::max(mc0.gdn_conv_kernel - 1, 0) *
                  mc0.gdn_conv_dim * 2 +
              (std::size_t)mc0.gdn_v_heads * mc0.gdn_v_dim * mc0.gdn_k_dim *
                  4;
    }
  }
  std::printf("[moe_cache] handshake: resident %d, cached %d; cached == "
              "resident %d (%zu logits, %zu tokens); the first decode steps "
              "allocated %.1f MB resident, %.1f MB cached (two slots of "
              "linear-layer state: %.1f MB)\n", a.hs, b.hs, same(b),
              a.lg.size(), a.toks.size(), a.grew_mb, b.grew_mb,
              2.0 * slot / 1048576.0);
  EXPECT_TRUE(!a.hs && !a.cached);
  // a checkpoint the cache refuses (mixed expert widths) proves nothing
  ASSERT_TRUE(b.cached);
  EXPECT_TRUE(same(b));
  EXPECT_TRUE(b.redos == 0);
  if (!b.hs) { return; }   // this GPU (or VPIPE_QWEN_HANDSHAKE=0) has none
  const Run c = run((std::size_t)2048 << 20, 13);
  ASSERT_TRUE(c.ok);
  if (!c.ok) { return; }
  std::printf("[moe_cache] every 13th turn abandoned: %d steps redone, "
              "== resident %d\n", c.redos, same(c));
  EXPECT_TRUE(c.redos > 0);
  EXPECT_TRUE(same(c));
}

namespace {

std::size_t
footprint_()
{
  task_vm_info_data_t i;
  mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&i, &n) !=
      KERN_SUCCESS) {
    return 0;
  }
  return (std::size_t)i.phys_footprint;
}

}  // namespace

// THE ENGINE'S CALL, through the model manager as text-chat loads an LM: on
// this box the 35B-A3B MoE loads resident; on a box simulated at 16 GB
// (VPIPE_RAM_LIMIT_MB) LoadedLanguageModel gives its routed experts a cache
// -- the load grows the process by a third as much -- and it still answers,
// the same greedy tokens as resident.
TEST(moe_expert_cache, lm_loader_caches_on_a_small_box)
{
  const char* path = std::getenv("VPIPE_QWEN35_MOE_TEST_MODEL_PATH");
  if (path == nullptr || *path == '\0') { return; }
  const PinnedTuners pinned;
  struct Run {
    std::vector<std::int32_t> gen;
    std::string text;
    std::size_t grew = 0;
    double tok_s = 0.0;
  };
  auto run = [&](const char* ram_mb) {
    Run r;
    if (ram_mb != nullptr) {
      ::setenv("VPIPE_RAM_LIMIT_MB", ram_mb, 1);
    } else {
      ::unsetenv("VPIPE_RAM_LIMIT_MB");
    }
    Session sess;
    MetalCompute* mc = sess.metal_compute();
    auto* mgr = sess.generative_model_manager();
    if (mc == nullptr || mgr == nullptr) { return r; }
    (void)mc->memory_budget_settled(15000);
    const std::size_t f0 = footprint_();
    LoadSpec spec;
    spec.hf_dir = path;
    spec.page_tokens = 512;
    spec.max_pages = 16;
    auto lm = mgr->load(spec);
    ::unsetenv("VPIPE_RAM_LIMIT_MB");
    if (!lm || !lm->valid()) { return r; }
    r.grew = footprint_() - std::min(footprint_(), f0);
    std::vector<std::int32_t> ids;
    // Thinking off: the answer, not a plan of it, inside 40 tokens.
    auto tpl = make_chat_template(lm->config(), lm->tokenizer(),
                                  /*disable_thinking=*/true);
    if (!tpl) { return r; }
    tpl->render_user_turn(
        "What is the capital of France? Answer in one short sentence.",
        /*is_first_turn=*/true, &ids);
    auto ctx = lm->make_context();
    std::int32_t t = lm->prefill(ctx, ids);
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 40 && t >= 0; ++i) {
      r.gen.push_back(t);
      t = lm->next_token_greedy(ctx);
    }
    r.tok_s = r.gen.size() /
              std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                            t0).count();
    r.text = lm->tokenizer().decode(
        std::span<const std::int32_t>(r.gen.data(), r.gen.size()));
    return r;
  };
  const Run a = run(nullptr);
  const Run b = run("16384");
  std::printf("[moe_cache] resident: grew %zu MB, %.1f tok/s | 16 GB box: "
              "grew %zu MB, %.1f tok/s\n  '%s'\n", a.grew >> 20, a.tok_s,
              b.grew >> 20, b.tok_s, b.text.c_str());
  ASSERT_TRUE(!a.gen.empty() && !b.gen.empty());
  EXPECT_TRUE(a.gen == b.gen);
  EXPECT_TRUE(b.grew * 3 < a.grew);
  std::string lower = b.text;
  for (char& ch : lower) { ch = (char)std::tolower((unsigned char)ch); }
  EXPECT_TRUE(lower.find("paris") != std::string::npos);
}

// REMOTE: given a Remote, a miss held far that the resident set would not
// take is not read: its pair gets the zero slot (all its parts zero) and
// is listed with the far slot holding it, whose parts are its bytes. One
// the resident set takes is read home as before; without a Remote the
// same miss is read from the far tier.
TEST(moe_expert_cache, remote_runs_a_far_expert_in_place)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto far = std::make_shared<TestFarMemory>(6 * 24);
  MoeExpertCache::Options o;
  o.layers = 1;
  o.experts = 16;
  o.part_bytes = {16, 8};
  o.resident = 2;
  o.transient = 4;
  o.threads = 2;
  o.half_life = 1000;
  o.far = far;
  o.far_one_offs = true;
  o.zero_slot = true;
  std::string err;
  std::atomic<int> reads{0};
  auto c = MoeExpertCache::create(
      mc, o,
      [&](int L, int e, std::uint8_t* const* dst, std::string*) {
        const std::uint32_t t = tag_(L, e);
        for (int i = 0; i < 4; ++i) { std::memcpy(dst[0] + 4 * i, &t, 4); }
        std::memcpy(dst[1], &t, 4);
        reads.fetch_add(1);
        return true;
      },
      &err);
  ASSERT_TRUE(c != nullptr);
  if (c == nullptr) { return; }
  ASSERT_TRUE(c->zero_slot() == c->slots());
  // the zero slot is held, and counted
  EXPECT_TRUE(c->bytes() == (std::size_t)(c->slots() + 1) * 24);
  std::vector<std::int32_t> sl(2);
  // 0 and 1 resident and hot; 2 a one-off, written far
  for (int k = 0; k < 3; ++k) {
    std::vector<std::int32_t> e{0, 1};
    ASSERT_TRUE(c->resolve(0, e.data(), 2, 1, sl.data(), &err));
  }
  {
    std::vector<std::int32_t> e{2};
    ASSERT_TRUE(c->resolve(0, e.data(), 1, 1, sl.data(), &err));
  }
  for (int i = 0; i < 1000 && far->writes.load() < 1; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_TRUE(c->far_holds(0, 2));
  // a spare's copy gone: the other spares taken by one-offs
  for (int e = 3; e <= 6; ++e) {
    std::vector<std::int32_t> ee{e};
    ASSERT_TRUE(c->resolve(0, ee.data(), 1, 1, sl.data(), &err));
  }
  const int disk = reads.load(), fr = far->reads.load();
  MoeExpertCache::Remote rem;
  std::vector<std::int32_t> e{2, 0};
  ASSERT_TRUE(c->resolve(0, e.data(), 2, 1, sl.data(), &err, nullptr, &rem));
  EXPECT_TRUE(sl[0] == c->zero_slot() && slot_tag_(*c, sl[1]) == tag_(0, 0));
  EXPECT_TRUE(rem.pair.size() == 1 && rem.pair[0] == 0);
  EXPECT_TRUE(reads.load() == disk && far->reads.load() == fr);
  EXPECT_TRUE(c->stats().far_computed == 1);
  if (rem.far_slot.size() == 1) {
    std::vector<FarPiece> pc(2);
    std::uint8_t a[16], b[8];
    pc[0] = {c->far_part_offset(rem.far_slot[0], 0), a, 16};
    pc[1] = {c->far_part_offset(rem.far_slot[0], 1), b, 8};
    ASSERT_TRUE(far->read(pc.data(), 2, &err));
    std::uint32_t t0, t1;
    std::memcpy(&t0, a + 12, 4);
    std::memcpy(&t1, b, 4);
    EXPECT_TRUE(t0 == tag_(0, 2) && t1 == tag_(0, 2));
  }
  for (int k = 0; k < 2; ++k) {
    const auto* z = static_cast<const std::uint8_t*>(c->pool(k).contents()) +
                    (std::size_t)c->zero_slot() * o.part_bytes[(std::size_t)k];
    bool zero = true;
    for (std::size_t i = 0; i < o.part_bytes[(std::size_t)k]; ++i) {
      zero = zero && z[i] == 0;
    }
    EXPECT_TRUE(zero);
  }
  c->remote_done(rem);
  // without a Remote it is read from the far tier
  const int fr2 = far->reads.load();
  ASSERT_TRUE(c->resolve(0, e.data(), 2, 1, sl.data(), &err));
  EXPECT_TRUE(slot_tag_(*c, sl[0]) == tag_(0, 2));
  EXPECT_TRUE(far->reads.load() == fr2 + 1 && reads.load() == disk);
}

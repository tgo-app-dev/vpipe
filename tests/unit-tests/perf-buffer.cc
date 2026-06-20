#include "minitest.h"
#include "common/perf-buffer.h"
#include "common/perf-event.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

using namespace std;
using namespace vpipe;

TEST(perf_buffer, records_and_caps_at_capacity) {
  PerfBuffer buf(8, chrono::steady_clock::now());
  for (uint32_t i = 0; i < 8; ++i) {
    buf.record(/*gvid*/ 1, /*type*/ 1, /*value*/ i);
  }
  EXPECT_TRUE(buf.size() == 8u);
  EXPECT_TRUE(buf.dropped() == 0u);
  for (uint32_t i = 0; i < 5; ++i) {
    buf.record(1, 2, i + 100);
  }
  EXPECT_TRUE(buf.size() == 8u);
  EXPECT_TRUE(buf.dropped() == 5u);
  // Drop-newest semantics: stored events are the first 8.
  for (size_t i = 0; i < buf.size(); ++i) {
    EXPECT_TRUE(buf.at(i).type  == 1u);
    EXPECT_TRUE(buf.at(i).value == i);
    EXPECT_TRUE(buf.at(i).stage_gvid == 1u);
  }
}

TEST(perf_buffer, timestamp_is_monotonic) {
  auto t0 = chrono::steady_clock::now();
  PerfBuffer buf(64, t0);
  for (uint32_t i = 0; i < 16; ++i) {
    buf.record(1, 0, 0);
  }
  EXPECT_TRUE(buf.size() == 16u);
  uint64_t prev = 0;
  for (size_t i = 0; i < buf.size(); ++i) {
    EXPECT_TRUE(buf.at(i).ns >= prev);
    prev = buf.at(i).ns;
  }
}

TEST(perf_buffer, clear_resets_and_reuses_slots) {
  PerfBuffer buf(4, chrono::steady_clock::now());
  for (uint32_t i = 0; i < 4; ++i) {
    buf.record(1, 1, i);
  }
  EXPECT_TRUE(buf.size() == 4u);
  buf.clear();
  EXPECT_TRUE(buf.size() == 0u);
  EXPECT_TRUE(buf.dropped() == 0u);
  buf.record(2, 7, 99);
  EXPECT_TRUE(buf.size() == 1u);
  EXPECT_TRUE(buf.at(0).stage_gvid == 2u);
  EXPECT_TRUE(buf.at(0).type       == 7u);
  EXPECT_TRUE(buf.at(0).value      == 99u);
}

TEST(perf_buffer, multi_thread_total_count_matches_capacity) {
  // Multi-producer (overflow-buffer scenario): the atomic
  // fetch_add gates per-slot writes so we see exactly capacity
  // entries and no slot stomps.
  PerfBuffer buf(1024, chrono::steady_clock::now());
  const int n_threads = 8;
  const int per_thread = 1000;
  vector<thread> ts;
  ts.reserve(n_threads);
  for (int t = 0; t < n_threads; ++t) {
    ts.emplace_back([&buf, t]() {
      for (int i = 0; i < per_thread; ++i) {
        buf.record(/*gvid*/ static_cast<uint32_t>(t),
                   /*type*/ static_cast<uint32_t>(t),
                   /*value*/ static_cast<uint64_t>(i));
      }
    });
  }
  for (auto& th : ts) {
    th.join();
  }
  EXPECT_TRUE(buf.size() == 1024u);
  EXPECT_TRUE(buf.dropped() ==
              size_t(n_threads) * size_t(per_thread) - 1024u);
  for (size_t i = 0; i < buf.size(); ++i) {
    EXPECT_TRUE(buf.at(i).type < uint32_t(n_threads));
    EXPECT_TRUE(buf.at(i).stage_gvid < uint32_t(n_threads));
  }
}

TEST(perf_buffer, zero_capacity_is_safe_noop) {
  PerfBuffer buf(0, chrono::steady_clock::now());
  EXPECT_TRUE(buf.capacity() == 0u);
  buf.record(1, 1, 1);
  buf.record(1, 2, 2);
  EXPECT_TRUE(buf.size() == 0u);
  EXPECT_TRUE(buf.dropped() == 2u);
}

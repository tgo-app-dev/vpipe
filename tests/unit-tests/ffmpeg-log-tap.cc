#include "minitest.h"
#include "common/ffmpeg-log-tap.h"

#include <atomic>
#include <string>
#include <thread>
#include <vector>

using namespace std;
using namespace vpipe;

TEST(ffmpeg_log_tap, install_then_remove_idempotent)
{
  atomic<int> count{0};
  LogTapHandle h = install_log_tap(
      [&count](int, string_view, string_view) {
        count.fetch_add(1, memory_order_relaxed);
      });
  EXPECT_TRUE(static_cast<bool>(h));

  ffmpeg_log_tap_dispatch(32, "rtsp", "hello");
  EXPECT_TRUE(count.load() == 1);

  remove_log_tap(h);
  ffmpeg_log_tap_dispatch(32, "rtsp", "post-remove");
  EXPECT_TRUE(count.load() == 1);

  // Removing twice or removing a zero handle is a no-op.
  remove_log_tap(h);
  remove_log_tap(LogTapHandle{});
  EXPECT_TRUE(count.load() == 1);
}

TEST(ffmpeg_log_tap, install_null_yields_zero_handle)
{
  LogTap empty;
  LogTapHandle h = install_log_tap(empty);
  EXPECT_FALSE(static_cast<bool>(h));
  // Dispatching with no taps must not crash.
  ffmpeg_log_tap_dispatch(0, "ffmpeg", "harmless");
}

TEST(ffmpeg_log_tap, multi_tap_dispatch)
{
  atomic<int> a{0}, b{0};
  LogTapHandle ha = install_log_tap(
      [&a](int, string_view, string_view) {
        a.fetch_add(1, memory_order_relaxed);
      });
  LogTapHandle hb = install_log_tap(
      [&b](int, string_view, string_view) {
        b.fetch_add(1, memory_order_relaxed);
      });

  ffmpeg_log_tap_dispatch(16, "rtp", "x");
  ffmpeg_log_tap_dispatch(16, "rtp", "y");
  EXPECT_TRUE(a.load() == 2);
  EXPECT_TRUE(b.load() == 2);

  remove_log_tap(ha);
  ffmpeg_log_tap_dispatch(16, "rtp", "z");
  EXPECT_TRUE(a.load() == 2);
  EXPECT_TRUE(b.load() == 3);
  remove_log_tap(hb);
}

TEST(ffmpeg_log_tap, dispatch_payload_matches)
{
  string seen_tag;
  string seen_msg;
  int    seen_level = 0;
  LogTapHandle h = install_log_tap(
      [&](int lvl, string_view tag, string_view msg) {
        seen_level = lvl;
        seen_tag.assign(tag);
        seen_msg.assign(msg);
      });
  ffmpeg_log_tap_dispatch(24, "udp", "max delay reached");
  EXPECT_TRUE(seen_level == 24);
  EXPECT_TRUE(seen_tag == "udp");
  EXPECT_TRUE(seen_msg == "max delay reached");
  remove_log_tap(h);
}

TEST(ffmpeg_log_tap, threaded_install_dispatch_remove)
{
  // Hammer the registry from multiple threads: each one installs a
  // tap, dispatches a bunch, and removes its tap. No deadlocks, no
  // dispatches to removed taps.
  constexpr int kThreads     = 8;
  constexpr int kPerThread   = 200;
  atomic<int>   global_count{0};
  atomic<bool>  fail{false};

  vector<thread> workers;
  workers.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    workers.emplace_back([&]() {
      atomic<int>  local{0};
      LogTapHandle h = install_log_tap(
          [&local](int, string_view, string_view) {
            local.fetch_add(1, memory_order_relaxed);
          });
      for (int i = 0; i < kPerThread; ++i) {
        ffmpeg_log_tap_dispatch(16, "rtsp", "tick");
      }
      remove_log_tap(h);
      // We can't predict cross-thread counts since other threads
      // dispatch concurrently, but THIS tap should have received
      // at least its own kPerThread dispatches.
      if (local.load() < kPerThread) {
        fail.store(true);
      }
      global_count.fetch_add(1, memory_order_relaxed);
    });
  }
  for (auto& w : workers) { w.join(); }
  EXPECT_FALSE(fail.load());
  EXPECT_TRUE(global_count.load() == kThreads);
}

// What FFmpeg said on THIS thread, WARNING or worse, while a capture
// lives: the reason a synchronous call refused. Another thread's lines
// (a decoder concealing a frame) are not this call's and are not kept;
// verbose lines are not reasons; scopes nest, the innermost collecting;
// the text is bounded.
TEST(ffmpeg_log_tap, a_capture_keeps_this_threads_reasons)
{
  {
    FfmpegLogCapture cap;
    ffmpeg_log_tap_dispatch(16, "SWR", "bad layout");          // ERROR
    ffmpeg_log_tap_dispatch(40, "SWR", "verbose chatter");     // VERBOSE
    std::thread other([] {
      ffmpeg_log_tap_dispatch(16, "h264", "concealing errors");
    });
    other.join();
    ffmpeg_log_tap_dispatch(24, "AVCodecContext", "a warning"); // WARNING
    EXPECT_TRUE(cap.text() == "SWR: bad layout; AVCodecContext: a warning");
    EXPECT_TRUE(cap.suffix() ==
                " (FFmpeg: SWR: bad layout; AVCodecContext: a warning)");
  }
  {
    FfmpegLogCapture outer;
    {
      FfmpegLogCapture inner;
      ffmpeg_log_tap_dispatch(16, "a", "inner");
      EXPECT_TRUE(inner.text() == "a: inner");
    }
    ffmpeg_log_tap_dispatch(16, "b", "outer");
    EXPECT_TRUE(outer.text() == "b: outer");
    EXPECT_TRUE(FfmpegLogCapture().suffix().empty());
  }
  {
    FfmpegLogCapture cap;
    const string line(200, 'x');
    for (int i = 0; i < 50; ++i) { ffmpeg_log_tap_dispatch(16, "t", line); }
    EXPECT_TRUE(cap.text().size() <= FfmpegLogCapture::kMax);
  }
  // No capture open: nothing to collect into, and nothing breaks.
  ffmpeg_log_tap_dispatch(16, "SWR", "nobody listening");
}

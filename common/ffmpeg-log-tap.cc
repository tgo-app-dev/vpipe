#include "common/ffmpeg-log-tap.h"

#include <atomic>
#include <shared_mutex>
#include <utility>
#include <vector>

namespace vpipe {

namespace {

struct Entry {
  std::uint64_t id;
  LogTap        tap;
};

// Singleton registry. Function-local statics so we don't get a static
// init order dependency with any global FFmpegLibraries instance.
//
// A shared_mutex, NOT a plain mutex: dispatch takes a SHARED lock for
// the full duration it invokes the taps, while install / remove take
// the EXCLUSIVE lock. This makes remove_log_tap synchronize with any
// in-flight dispatch -- once it returns, no dispatch is still inside
// the removed tap and no later dispatch can see it, so the caller may
// safely destroy whatever state the tap captured (e.g. the stack-local
// `std::atomic<bool>` rtsp-capture-stage installs). The previous
// snapshot-then-invoke-unlocked scheme let a dispatch call a tap after
// remove returned -> use-after-free of that captured state. Holding the
// shared lock across the callbacks is sound because the tap contract
// (see ffmpeg-log-tap.h) forbids taps from re-entering FFmpeg / the
// session, hence from re-entering install / remove / dispatch; multiple
// dispatches still run concurrently under the shared lock.
std::shared_mutex&
registry_mutex()
{
  static std::shared_mutex m;
  return m;
}

std::vector<Entry>&
registry()
{
  static std::vector<Entry> v;
  return v;
}

std::atomic<std::uint64_t>&
next_id_counter()
{
  static std::atomic<std::uint64_t> c{1};
  return c;
}

}

LogTapHandle
install_log_tap(LogTap tap)
{
  if (!tap) {
    return LogTapHandle{};
  }
  LogTapHandle h;
  h.id = next_id_counter().fetch_add(1, std::memory_order_relaxed);
  std::unique_lock<std::shared_mutex> g(registry_mutex());
  registry().push_back(Entry{h.id, std::move(tap)});
  return h;
}

void
remove_log_tap(LogTapHandle h)
{
  if (h.id == 0) {
    return;
  }
  std::unique_lock<std::shared_mutex> g(registry_mutex());
  auto& v = registry();
  for (auto it = v.begin(); it != v.end(); ++it) {
    if (it->id == h.id) {
      v.erase(it);
      return;
    }
  }
}

void
ffmpeg_log_tap_dispatch(int              level,
                        std::string_view tag,
                        std::string_view msg)
{
  // Invoke the taps under a SHARED lock held for the whole loop. This
  // keeps remove_log_tap (exclusive lock) from completing while a tap
  // it is removing is still executing, so a caller that removes its tap
  // and then drops the state the tap captured cannot be raced into a
  // use-after-free. Concurrent dispatches from different FFmpeg threads
  // still run in parallel (shared lock). The tap contract forbids
  // re-entering install / remove / dispatch, so holding the lock across
  // the callbacks cannot deadlock.
  std::shared_lock<std::shared_mutex> g(registry_mutex());
  for (const auto& e : registry()) {
    e.tap(level, tag, msg);
  }
}

}

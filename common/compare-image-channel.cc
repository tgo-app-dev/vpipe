#include "common/compare-image-channel.h"

#include <chrono>
#include <utility>

namespace vpipe {

void
CompareImageChannel::publish(Png a, Png b, int width, int height)
{
  {
    std::lock_guard<std::mutex> lk(_mu);
    if (_cur.closed) { return; }
    _cur.a      = std::move(a);
    _cur.b      = std::move(b);
    _cur.width  = width;
    _cur.height = height;
    ++_cur.version;
  }
  _cv.notify_all();
}

void
CompareImageChannel::close()
{
  {
    std::lock_guard<std::mutex> lk(_mu);
    if (_cur.closed) { return; }
    _cur.closed = true;
  }
  _cv.notify_all();
}

CompareImageChannel::Snapshot
CompareImageChannel::snapshot() const
{
  std::lock_guard<std::mutex> lk(_mu);
  return _cur;
}

CompareImageChannel::Snapshot
CompareImageChannel::wait_change(std::uint64_t since, int timeout_ms) const
{
  std::unique_lock<std::mutex> lk(_mu);
  _cv.wait_for(lk, std::chrono::milliseconds(timeout_ms),
               [&] { return _cur.closed || _cur.version != since; });
  return _cur;
}

bool
CompareImageChannel::closed() const noexcept
{
  std::lock_guard<std::mutex> lk(_mu);
  return _cur.closed;
}

}

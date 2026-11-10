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
    _pa = {};
    _pb = {};
    _render.reset();
  }
  _cv.notify_all();
}

void
CompareImageChannel::publish_pictures(Picture a, Picture b, int width,
                                      int height,
                                      std::shared_ptr<const Render> render)
{
  {
    std::lock_guard<std::mutex> lk(_mu);
    if (_cur.closed) { return; }
    const bool same_size = _cur.width == width && _cur.height == height;
    // A side the last publish already had -- its PNG, or one still owed
    // -- stays as it is; anything else is owed afresh.
    auto latch = [same_size](Pending& p, Png& png, Picture pic) {
      if (same_size && pic.rgb != nullptr && pic.rgb == p.pic.rgb &&
          (png != nullptr || p.owed)) {
        return;
      }
      png    = nullptr;
      p.owed = pic.rgb != nullptr;
      p.pic  = std::move(pic);
    };
    latch(_pa, _cur.a, std::move(a));
    latch(_pb, _cur.b, std::move(b));
    _cur.width  = width;
    _cur.height = height;
    ++_cur.version;
    _render = std::move(render);
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

// Until nothing is owed: a publish that lands during a render owes its
// sides in turn, and a reader must not be handed a version whose
// produced side reads as missing.
CompareImageChannel::Snapshot
CompareImageChannel::render_owed_() const
{
  {
    std::lock_guard<std::mutex> lk(_mu);
    if (!_pa.owed && !_pb.owed) { return _cur; }
  }
  std::lock_guard<std::mutex> rl(_render_mu);
  for (;;) {
    Picture pic;
    bool    a = false;
    int     w = 0, h = 0;
    std::shared_ptr<const Render> render;
    {
      std::lock_guard<std::mutex> lk(_mu);
      if (_pa.owed)      { pic = _pa.pic; a = true; }
      else if (_pb.owed) { pic = _pb.pic; }
      else               { return _cur; }
      w      = _cur.width;
      h      = _cur.height;
      render = _render;
    }
    Png png = render != nullptr ? (*render)(pic, w, h) : nullptr;
    {
      std::lock_guard<std::mutex> lk(_mu);
      Pending& p = a ? _pa : _pb;
      // Still the side that was rendered? Else a publish replaced it
      // and the loop renders what it owes instead.
      if (p.owed && p.pic.rgb == pic.rgb && _cur.width == w &&
          _cur.height == h) {
        (a ? _cur.a : _cur.b) = std::move(png);
        p.owed = false;
      }
    }
  }
}

CompareImageChannel::Snapshot
CompareImageChannel::snapshot() const
{
  return render_owed_();
}

CompareImageChannel::Snapshot
CompareImageChannel::wait_change(std::uint64_t since, int timeout_ms) const
{
  {
    std::unique_lock<std::mutex> lk(_mu);
    _cv.wait_for(lk, std::chrono::milliseconds(timeout_ms),
                 [&] { return _cur.closed || _cur.version != since; });
  }
  return render_owed_();
}

bool
CompareImageChannel::closed() const noexcept
{
  std::lock_guard<std::mutex> lk(_mu);
  return _cur.closed;
}

}

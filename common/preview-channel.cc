#include "common/preview-channel.h"

#include <chrono>
#include <cstring>
#include <utility>

namespace vpipe {

namespace {
// Per-subscriber backlog cap. A client that falls this far behind (a
// stalled tab, a throttled link) has its queue dropped and is re-seeded
// with the init segment (so its MediaSource rebuilds and resyncs to live),
// bounding memory without breaking playback. Generous enough that a
// healthy client never trips it.
constexpr std::size_t kMaxSubBytes = 24u * 1024u * 1024u;
}  // namespace

// One connection's pending messages. All access is under PreviewChannel::_mu.
struct PreviewChannel::Subscriber {
  std::deque<Blob> q;
  std::size_t      bytes = 0;
};

PreviewChannel::~PreviewChannel() = default;

PreviewChannel::Blob
PreviewChannel::msg_(std::uint8_t type, const std::uint8_t* payload,
                     std::size_t n)
{
  auto v = std::make_shared<std::vector<std::uint8_t>>();
  v->resize(1 + n);
  (*v)[0] = type;
  if (n > 0) {
    std::memcpy(v->data() + 1, payload, n);
  }
  return v;
}

void
PreviewChannel::broadcast_(const Blob& blob)
{
  for (auto& sub : _subs) {
    if (sub->bytes + blob->size() > kMaxSubBytes) {
      // Lagging client: drop the backlog and re-seed the init segment so
      // its MediaSource rebuilds + resyncs to the live edge.
      sub->q.clear();
      sub->bytes = 0;
      if (_init_blob) {
        sub->q.push_back(_init_blob);
        sub->bytes += _init_blob->size();
      }
    }
    sub->q.push_back(blob);
    sub->bytes += blob->size();
  }
  _cv.notify_all();
}

void
PreviewChannel::set_config(std::string json, bool has_video,
                           bool has_audio, int width, int height)
{
  std::lock_guard<std::mutex> lk(_mu);
  _has_video = has_video;
  _has_audio = has_audio;
  _width     = width;
  _height    = height;
  if (json == _config_json) {
    return;
  }
  _config_json = std::move(json);
  _config_blob = msg_(
      kMsgConfig,
      reinterpret_cast<const std::uint8_t*>(_config_json.data()),
      _config_json.size());
  if (!_closed) {
    broadcast_(_config_blob);
  }
}

void
PreviewChannel::set_init(const std::uint8_t* data, std::size_t n)
{
  Blob b = msg_(kMsgInit, data, n);
  std::lock_guard<std::mutex> lk(_mu);
  if (_closed) {
    return;
  }
  _init_blob = b;
  broadcast_(b);
}

void
PreviewChannel::push_fragment(const std::uint8_t* data, std::size_t n)
{
  if (n == 0) {
    return;
  }
  Blob b = msg_(kMsgFragment, data, n);
  std::lock_guard<std::mutex> lk(_mu);
  if (_closed) {
    return;
  }
  broadcast_(b);
}

void
PreviewChannel::push_audio(const float* const* planes, int channels,
                           int frames)
{
  if (channels <= 0 || frames <= 0 || !planes) {
    return;
  }
  const std::size_t plane_bytes =
      static_cast<std::size_t>(frames) * sizeof(float);
  std::vector<std::uint8_t> raw(
      static_cast<std::size_t>(channels) * plane_bytes);
  for (int c = 0; c < channels; ++c) {
    std::memcpy(raw.data() + static_cast<std::size_t>(c) * plane_bytes,
                planes[c], plane_bytes);
  }
  Blob b = msg_(kMsgAudio, raw.data(), raw.size());
  std::lock_guard<std::mutex> lk(_mu);
  if (_closed) {
    return;
  }
  broadcast_(b);
}

void
PreviewChannel::close()
{
  std::lock_guard<std::mutex> lk(_mu);
  _closed = true;
  _cv.notify_all();
}

bool PreviewChannel::has_video() const noexcept
{
  std::lock_guard<std::mutex> lk(_mu);
  return _has_video;
}
bool PreviewChannel::has_audio() const noexcept
{
  std::lock_guard<std::mutex> lk(_mu);
  return _has_audio;
}
int PreviewChannel::width() const noexcept
{
  std::lock_guard<std::mutex> lk(_mu);
  return _width;
}
int PreviewChannel::height() const noexcept
{
  std::lock_guard<std::mutex> lk(_mu);
  return _height;
}
bool PreviewChannel::closed() const noexcept
{
  std::lock_guard<std::mutex> lk(_mu);
  return _closed;
}

std::shared_ptr<PreviewChannel::Subscriber>
PreviewChannel::subscribe()
{
  auto sub = std::make_shared<Subscriber>();
  std::lock_guard<std::mutex> lk(_mu);
  if (_config_blob) {
    sub->q.push_back(_config_blob);
    sub->bytes += _config_blob->size();
  }
  if (_init_blob) {
    sub->q.push_back(_init_blob);
    sub->bytes += _init_blob->size();
  }
  _subs.push_back(sub);
  return sub;
}

void
PreviewChannel::unsubscribe(const std::shared_ptr<Subscriber>& sub)
{
  std::lock_guard<std::mutex> lk(_mu);
  for (auto it = _subs.begin(); it != _subs.end(); ++it) {
    if (*it == sub) {
      _subs.erase(it);
      break;
    }
  }
}

std::shared_ptr<const std::vector<std::uint8_t>>
PreviewChannel::wait_frame(const std::shared_ptr<Subscriber>& sub,
                           int timeout_ms)
{
  std::unique_lock<std::mutex> lk(_mu);
  _cv.wait_for(lk, std::chrono::milliseconds(timeout_ms),
               [&] { return !sub->q.empty() || _closed; });
  if (!sub->q.empty()) {
    Blob b = std::move(sub->q.front());
    sub->q.pop_front();
    sub->bytes -= b->size();
    return b;
  }
  return nullptr;   // timeout, or closed and drained
}

}

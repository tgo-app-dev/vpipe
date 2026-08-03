#include "apps/web-ui/view-api.h"
#include "apps/web-ui/api-common.h"

#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "ui/ui-view-registry.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace vpipe::webui {

// -------------------------------------------------------------------
// Stage-provided GUI views
// -------------------------------------------------------------------


HttpResponse
ViewApi::h_views_(const HttpRequest&)
{
  FlexData o = FlexData::make_object();
  FlexData arr = FlexData::make_array();
  {
    auto a = arr.as_array();
    for (const UiViewSpec* v : UiViewRegistry::get().all()) {
      FlexData e = FlexData::make_object();
      auto eo = e.as_object();
      eo.insert("id", fstr(string(v->id)));
      eo.insert("stage_type", fstr(string(v->stage_type)));
      eo.insert("module", fstr(string(v->module)));
      eo.insert("styles", fstr(string(v->styles)));
      eo.insert("label_key", fstr(string(v->label_key)));
      eo.insert("icon", fstr(string(v->icon)));
      a.push_back(std::move(e));
    }
  }
  auto oo = o.as_object();
  oo.insert("views", std::move(arr));
  return HttpResponse::json(200, o.to_json());
}

namespace {

// Backend -> client half of one view channel. The backend's own threads
// ENQUEUE here; the connection thread drains and writes, so a slow
// socket never blocks a producer and the WebSocket is only ever touched
// by one thread (which also keeps TLS legal).
class WsViewChannel final : public UiViewChannel {
public:
  // Bounds on the queue. The producer that matters -- a preview stage's
  // media -- already sheds load upstream (PreviewChannel drops a lagging
  // subscriber's backlog), so reaching these means the socket itself is
  // wedged: drop the channel and let the view reconnect.
  static constexpr size_t kMaxFrames = 512;
  static constexpr size_t kMaxBytes  = 32u << 20;

  void
  send(const FlexData& msg) override
  {
    enqueue_(false, msg.to_json());
  }

  void
  send_binary(const FlexData& header, const uint8_t* data,
              size_t size) override
  {
    const string h = header.to_json();
    const uint32_t hl = static_cast<uint32_t>(h.size());
    string frame;
    frame.reserve(4 + h.size() + size);
    const char le[4] = {
      static_cast<char>(hl & 0xff),
      static_cast<char>((hl >> 8) & 0xff),
      static_cast<char>((hl >> 16) & 0xff),
      static_cast<char>((hl >> 24) & 0xff),
    };
    frame.append(le, 4);
    frame.append(h);
    if (size > 0) {
      frame.append(reinterpret_cast<const char*>(data), size);
    }
    enqueue_(true, std::move(frame));
  }

  bool
  alive() const override
  {
    lock_guard<mutex> lk(_mu);
    return _alive;
  }

  // Block up to `ms` for something to send (or for the channel to die),
  // so an outbound message goes out immediately rather than waiting for
  // the next inbound poll.
  void
  wait_for_output(int ms)
  {
    unique_lock<mutex> lk(_mu);
    _cv.wait_for(lk, std::chrono::milliseconds(ms),
                 [this] { return !_q.empty() || !_alive; });
  }

  // Write everything queued. False once the peer is gone.
  bool
  flush(WebSocket& ws)
  {
    std::deque<Frame> out;
    {
      lock_guard<mutex> lk(_mu);
      if (!_alive) { return false; }
      out.swap(_q);
      _bytes = 0;
    }
    for (const Frame& f : out) {
      const bool ok =
          f.binary
            ? ws.send_binary(
                  reinterpret_cast<const uint8_t*>(f.data.data()),
                  f.data.size())
            : ws.send_text(f.data);
      if (!ok) { kill(); return false; }
    }
    return true;
  }

  // Mark dead and wake anything waiting on the channel.
  void
  kill()
  {
    {
      lock_guard<mutex> lk(_mu);
      _alive = false;
      _q.clear();
      _bytes = 0;
    }
    _cv.notify_all();
  }

private:
  struct Frame {
    bool   binary;
    string data;
  };

  void
  enqueue_(bool binary, string data)
  {
    {
      lock_guard<mutex> lk(_mu);
      if (!_alive) { return; }
      _bytes += data.size();
      _q.push_back(Frame{binary, std::move(data)});
      if (_q.size() > kMaxFrames || _bytes > kMaxBytes) {
        _alive = false;
        _q.clear();
        _bytes = 0;
      }
    }
    _cv.notify_all();
  }

  mutable mutex           _mu;
  std::condition_variable _cv;
  std::deque<Frame>       _q;
  size_t                  _bytes = 0;
  bool                    _alive = true;
};

}  // namespace

void
ViewApi::h_view_ws_(const HttpRequest& req, WebSocket& ws)
{
  string view_id;
  if (auto it = req.params.find("view"); it != req.params.end()) {
    view_id = it->second;
  }
  const UiViewSpec* spec = UiViewRegistry::get().find(view_id);
  if (spec == nullptr || spec->make_backend == nullptr) {
    return;   // unknown view: let the WebSocket close
  }

  // `ch` MUST outlive `backend`: the backend's worker threads write
  // through the channel, and on_close()/destruction is what joins them.
  WsViewChannel ch;
  std::unique_ptr<UiViewBackendIntf> backend = spec->make_backend(*view_host());
  if (!backend) { return; }
  backend->on_open(ch);

  // One thread, both directions: poll for client frames without
  // blocking, then wait briefly for anything the backend queued. An
  // outbound message leaves as soon as it is produced (the wait is
  // woken by the enqueue); an inbound one is picked up within a tick.
  //
  // This handler owns its connection for as long as the browser keeps
  // the panel open, so unlike a request/response route it CANNOT simply
  // run to completion -- it has to watch for shutdown. stop() waits only
  // a bounded time for handlers, and everything this one borrows (the
  // view host, and through it the SessionApi that owns the pipelines) is
  // destroyed right after that wait. Missing this leaves the loop
  // dereferencing a freed SessionApi and aborting on its destroyed mutex.
  constexpr int kTickMs = 20;
  string in;
  for (;;) {
    if (_ctx.server != nullptr && _ctx.server->stopping()) { break; }
    bool closed = false;
    for (;;) {
      const WebSocket::Msg m = ws.recv(in, 0);
      if (m == WebSocket::Msg::Closed) { closed = true; break; }
      if (m == WebSocket::Msg::None) { break; }
      if (m != WebSocket::Msg::Text) { continue; }   // no binary uplink
      FlexData msg;
      try {
        msg = FlexData::from_json(in);
      } catch (const std::exception&) {
        continue;   // malformed: ignore the frame, keep the channel
      }
      if (msg.is_object()) { backend->on_message(ch, msg); }
    }
    if (closed) { break; }
    ch.wait_for_output(kTickMs);
    if (!ch.flush(ws)) { break; }
  }

  ch.kill();
  backend->on_close();
}

void
ViewApi::register_routes(HttpServer& s)
{
  s.route("GET", "/api/ui/views",
          [this](const HttpRequest& r) { return h_views_(r); });
  s.route_ws("/api/ui/view/:view/ws",
             [this](const HttpRequest& r, WebSocket& ws) {
               h_view_ws_(r, ws);
             });
}

}

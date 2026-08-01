// Backend for the "compare-image" stage's GUI view.
//
// Same shape as the preview view's backend (ui/stages/preview/): the
// panel names a stage to follow and this pushes every transition and
// every new image pair. What differs is the payload -- a comparison is a
// PAIR published as a unit, not a stream, so each update is one state
// message followed by the images that exist.
//
// PROTOCOL (view -> backend):
//   {m:"list"}                      enumerate declared compare stages
//   {m:"watch", pipeline, stage}    follow this stage
//   {m:"unwatch"}                   stop following
//
// PROTOCOL (backend -> view):
//   {m:"stages", list:[{pipeline,stage,title,state,live}]}
//   {m:"waiting"|"playing"|"gone", pipeline, stage[, title]}
//   {m:"pair", width, height, has_a, has_b, version}
//   binary {m:"image", slot:"a"|"b"} + PNG bytes
//
// The state message ALWAYS precedes the images of that version, so the
// view knows which slots to blank before any bytes arrive; a slot that
// is false in `pair` gets no binary frame at all.

#include "ui/ui-view-registry.h"

#include "common/compare-image-channel.h"
#include "common/flex-data.h"
#include "pipeline/stage.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace vpipe {
namespace {

using namespace std::chrono_literals;

constexpr const char* kStageType = "compare-image";

std::string
str_field(const FlexData& obj, std::string_view key)
{
  if (!obj.is_object()) { return {}; }
  auto o = obj.as_object();
  if (!o.contains(key)) { return {}; }
  const FlexData v = o.at(key);
  return std::string(v.as_string(""));
}

FlexData
msg_obj(std::string_view kind)
{
  FlexData m = FlexData::make_object();
  auto mo = m.as_object();
  mo.insert("m", FlexData::make_string(kind));
  return m;
}

std::string
stage_title(const UiViewHostIntf::StageRef& r)
{
  std::string t = str_field(r.config, "title");
  return t.empty() ? r.stage : t;
}

class CompareViewBackend final : public UiViewBackendIntf {
public:
  explicit CompareViewBackend(UiViewHostIntf& host) : _host(host) {}

  ~CompareViewBackend() override { stop_worker_(); }

  void
  on_open(UiViewChannel& ch) override
  {
    _ch = &ch;
    _worker = std::thread([this] { run_(); });
  }

  void
  on_message(UiViewChannel& ch, const FlexData& msg) override
  {
    const std::string m = str_field(msg, "m");
    if (m == "list") {
      send_stage_list_(ch);
      return;
    }
    if (m == "watch" || m == "unwatch") {
      {
        std::lock_guard<std::mutex> lk(_mu);
        _want_pipeline = (m == "watch") ? str_field(msg, "pipeline") : "";
        _want_stage    = (m == "watch") ? str_field(msg, "stage") : "";
        _generation++;
        _reported.clear();
        _wake = true;
      }
      _cv.notify_all();
      return;
    }
  }

  void
  on_close() override
  {
    stop_worker_();
    _ch = nullptr;
  }

private:
  void
  stop_worker_()
  {
    {
      std::lock_guard<std::mutex> lk(_mu);
      if (_stop) { return; }
      _stop = true;
    }
    _cv.notify_all();
    if (_worker.joinable()) { _worker.join(); }
  }

  void
  send_stage_list_(UiViewChannel& ch)
  {
    FlexData m = msg_obj("stages");
    FlexData arr = FlexData::make_array();
    {
      auto a = arr.as_array();
      for (const auto& r : _host.find_stages(kStageType)) {
        FlexData e = FlexData::make_object();
        auto eo = e.as_object();
        eo.insert("pipeline", FlexData::make_string(r.pipeline));
        eo.insert("stage", FlexData::make_string(r.stage));
        eo.insert("title", FlexData::make_string(stage_title(r)));
        eo.insert("state", FlexData::make_string(r.state));
        eo.insert("live", FlexData::make_bool(r.live));
        a.push_back(std::move(e));
      }
    }
    auto mo = m.as_object();
    mo.insert("list", std::move(arr));
    ch.send(m);
  }

  // Announce a transition at most once per distinct state, so a stage
  // that sits stopped doesn't stream duplicate "waiting" frames.
  void
  report_(UiViewChannel& ch, std::string_view kind,
          const std::string& pipeline, const std::string& stage,
          const std::string& title)
  {
    const std::string tag =
        std::string(kind) + '\0' + pipeline + '\0' + stage;
    {
      std::lock_guard<std::mutex> lk(_mu);
      if (_reported == tag) { return; }
      _reported = tag;
    }
    FlexData m = msg_obj(kind);
    auto mo = m.as_object();
    mo.insert("pipeline", FlexData::make_string(pipeline));
    mo.insert("stage", FlexData::make_string(stage));
    if (!title.empty()) {
      mo.insert("title", FlexData::make_string(title));
    }
    ch.send(m);
  }

  // One published pair: the state message, then the PNGs that exist.
  void
  send_pair_(UiViewChannel& ch,
             const CompareImageChannel::Snapshot& s)
  {
    FlexData m = msg_obj("pair");
    {
      auto mo = m.as_object();
      mo.insert("width", FlexData::make_int(s.width));
      mo.insert("height", FlexData::make_int(s.height));
      mo.insert("has_a", FlexData::make_bool(s.a != nullptr));
      mo.insert("has_b", FlexData::make_bool(s.b != nullptr));
      mo.insert("version",
                FlexData::make_uint(static_cast<std::uint64_t>(s.version)));
    }
    ch.send(m);

    auto push = [&](const char* slot,
                    const CompareImageChannel::Png& png) {
      if (!png || png->empty()) { return; }
      FlexData h = msg_obj("image");
      auto ho = h.as_object();
      ho.insert("slot", FlexData::make_string(slot));
      ch.send_binary(h, png->data(), png->size());
    };
    push("a", s.a);
    push("b", s.b);
  }

  bool
  superseded_(std::uint64_t generation) const
  {
    std::lock_guard<std::mutex> lk(_mu);
    return _stop || _generation != generation;
  }

  // Follow one live stage's channel until it closes, the client goes
  // away, or the panel follows something else.
  void
  relay_(UiViewChannel& ch,
         const std::shared_ptr<CompareImageChannel>& cc,
         std::uint64_t generation)
  {
    std::uint64_t seen = 0;
    // Send whatever is already latched, so a panel mounting mid-run sees
    // the current pair instead of waiting for the next publish.
    {
      const auto s = cc->snapshot();
      if (s.version != 0) {
        send_pair_(ch, s);
        seen = s.version;
      }
    }
    while (!superseded_(generation) && ch.alive()) {
      const auto s = cc->wait_change(seen, 500);
      if (s.closed) { break; }
      if (s.version != seen) {
        seen = s.version;
        send_pair_(ch, s);
      }
    }
  }

  void
  run_()
  {
    for (;;) {
      std::string   pipeline;
      std::string   stage;
      std::uint64_t generation = 0;
      {
        std::unique_lock<std::mutex> lk(_mu);
        _cv.wait_for(lk, 500ms, [this] { return _stop || _wake; });
        if (_stop) { return; }
        _wake      = false;
        pipeline   = _want_pipeline;
        stage      = _want_stage;
        generation = _generation;
      }
      UiViewChannel* ch = _ch;
      if (ch == nullptr || !ch->alive()) { continue; }
      if (pipeline.empty()) { continue; }

      bool declared = false;
      std::string title = stage;
      for (const auto& r : _host.find_stages(kStageType)) {
        if (r.pipeline == pipeline && r.stage == stage) {
          declared = true;
          title    = stage_title(r);
          break;
        }
      }
      if (!declared) {
        report_(*ch, "gone", pipeline, stage, {});
        continue;
      }

      std::shared_ptr<CompareImageChannel> cc;
      if (Stage* s = _host.live_stage(pipeline, stage); s != nullptr) {
        auto* src = dynamic_cast<CompareImageSource*>(s);
        if (src != nullptr) { cc = src->compare_channel(); }
      }
      if (!cc || cc->closed()) {
        report_(*ch, "waiting", pipeline, stage, title);
        continue;
      }

      report_(*ch, "playing", pipeline, stage, title);
      relay_(*ch, cc, generation);
      {
        std::lock_guard<std::mutex> lk(_mu);
        if (_generation == generation) { _reported.clear(); }
      }
    }
  }

  UiViewHostIntf& _host;
  UiViewChannel*  _ch = nullptr;

  mutable std::mutex      _mu;
  std::condition_variable _cv;
  std::thread             _worker;
  bool                    _stop = false;
  bool                    _wake = false;
  std::string             _want_pipeline;
  std::string             _want_stage;
  std::uint64_t           _generation = 0;
  std::string             _reported;
};

std::unique_ptr<UiViewBackendIntf>
make_compare_backend(UiViewHostIntf& host)
{
  return std::make_unique<CompareViewBackend>(host);
}

const UiViewSpec kView{
  .id           = "compare-image",
  .stage_type   = kStageType,
  .module       = "/ui/stages/compare-image/compare-image-view.js",
  .styles       = "/ui/stages/compare-image/compare-image-view.css",
  .label_key    = "compare.panel",
  .icon         = "video",
  .make_backend = &make_compare_backend,
};

}

VPIPE_REGISTER_UI_VIEW(compare_image, kView)

}

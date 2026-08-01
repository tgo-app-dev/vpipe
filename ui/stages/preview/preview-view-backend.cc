// Backend for the "preview" stage's GUI view.
//
// The panel's JavaScript lives beside this file (preview-view.js) and is
// embedded into libvpipe with it; the front-end application only serves
// those bytes and forwards the messages below. Everything the panel
// knows about preview stages -- which exist, which are live, when one
// starts or stops, and the media itself -- comes from here.
//
// PROTOCOL (view -> backend):
//   {m:"list"}                      enumerate declared preview stages
//   {m:"watch", pipeline, stage}    follow this stage; relay it whenever
//                                   its pipeline is running
//   {m:"unwatch"}                   stop following
//
// PROTOCOL (backend -> view):
//   {m:"stages", list:[{pipeline,stage,title,state,live}]}
//   {m:"waiting", pipeline, stage}  followed, not running yet
//   {m:"playing", pipeline, stage, title}   relay started
//   {m:"gone", pipeline, stage}     followed stage no longer exists
//                                   (its pipeline was unloaded)
//   {m:"config", video:{codec,width,height}, audio:{sampleRate,channels}}
//   binary {m:"init"|"fragment"|"image"|"audio"} + payload
//
// The watch loop runs here rather than in the browser: the view states
// what it wants once and the backend pushes every transition, instead of
// the panel polling a REST endpoint on a timer.

#include "ui/ui-view-registry.h"

#include "common/flex-data.h"
#include "common/preview-channel.h"
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

// The stage type this view presents.
constexpr const char* kStageType = "preview";

// Read a string field out of a FlexData object. The entry is returned by
// value, so the FlexData must be bound to a local before as_string() --
// otherwise the string_view dangles into a destroyed temporary.
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

// A preview stage's display label: its `title` config, else its id.
std::string
stage_title(const UiViewHostIntf::StageRef& r)
{
  std::string t = str_field(r.config, "title");
  return t.empty() ? r.stage : t;
}

class PreviewViewBackend final : public UiViewBackendIntf {
public:
  explicit PreviewViewBackend(UiViewHostIntf& host) : _host(host) {}

  ~PreviewViewBackend() override { stop_worker_(); }

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
    if (m == "watch") {
      const std::string p = str_field(msg, "pipeline");
      const std::string s = str_field(msg, "stage");
      {
        std::lock_guard<std::mutex> lk(_mu);
        _want_pipeline = p;
        _want_stage    = s;
        _generation++;
        _reported.clear();
      }
      _cv.notify_all();
      return;
    }
    if (m == "unwatch") {
      {
        std::lock_guard<std::mutex> lk(_mu);
        _want_pipeline.clear();
        _want_stage.clear();
        _generation++;
        _reported.clear();
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

  // Every declared preview stage across the loaded pipelines, running or
  // not -- the panel's picker list.
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

  // Announce a state transition at most once per distinct state, so a
  // stage that stays stopped doesn't stream duplicate "waiting" frames.
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

  // Translate one PreviewChannel message (a 1-byte type tag followed by
  // its payload -- see common/preview-channel.h) into this view's
  // protocol: the config header becomes a structured message, the media
  // stays binary with a structured header.
  void
  relay_blob_(UiViewChannel& ch, const std::vector<std::uint8_t>& blob)
  {
    if (blob.empty()) { return; }
    const std::uint8_t  tag = blob[0];
    const std::uint8_t* pay = blob.data() + 1;
    const std::size_t   n   = blob.size() - 1;

    if (tag == PreviewChannel::kMsgConfig) {
      FlexData cfg;
      try {
        cfg = FlexData::from_json(std::string_view(
            reinterpret_cast<const char*>(pay), n));
      } catch (const std::exception&) {
        return;   // malformed config: drop rather than kill the relay
      }
      if (!cfg.is_object()) { return; }
      auto co = cfg.as_object();
      co.insert_or_assign("m", FlexData::make_string("config"));
      ch.send(cfg);
      return;
    }

    const char* kind = nullptr;
    switch (tag) {
      case PreviewChannel::kMsgInit:     kind = "init";     break;
      case PreviewChannel::kMsgFragment: kind = "fragment"; break;
      case PreviewChannel::kMsgAudio:    kind = "audio";    break;
      case PreviewChannel::kMsgImage:    kind = "image";    break;
      default: return;
    }
    const FlexData hdr = msg_obj(kind);
    ch.send_binary(hdr, pay, n);
  }

  // True once the panel asked for a different stage (or we are shutting
  // down): the caller abandons whatever it is relaying.
  bool
  superseded_(std::uint64_t generation) const
  {
    std::lock_guard<std::mutex> lk(_mu);
    return _stop || _generation != generation;
  }

  // Pump one live stage's channel into the view until it closes, the
  // client goes away, or the panel follows something else.
  void
  relay_(UiViewChannel& ch, const std::shared_ptr<PreviewChannel>& pc,
         std::uint64_t generation)
  {
    auto sub = pc->subscribe();
    while (!superseded_(generation) && ch.alive()) {
      auto blob = pc->wait_frame(sub, 500);
      if (blob) {
        relay_blob_(ch, *blob);
      } else if (pc->closed()) {
        break;   // the stage tore down (its pipeline stopped)
      }
      // else: the wait timed out with nothing queued -- loop and retry.
    }
    pc->unsubscribe(sub);
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
        // Wake early on a new designation; otherwise re-poll on a
        // cadence so a stage going live is picked up promptly.
        _cv.wait_for(lk, 500ms, [this] { return _stop || _wake; });
        if (_stop) { return; }
        _wake      = false;
        pipeline   = _want_pipeline;
        stage      = _want_stage;
        generation = _generation;
      }
      UiViewChannel* ch = _ch;
      if (ch == nullptr || !ch->alive()) { continue; }
      if (pipeline.empty()) { continue; }   // picker showing, nothing to do

      // Is the followed stage still declared anywhere? A pipeline that
      // was unloaded can never go live again, so tell the panel to drop
      // the designation rather than wait forever.
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

      std::shared_ptr<PreviewChannel> pc;
      if (Stage* s = _host.live_stage(pipeline, stage); s != nullptr) {
        if (auto* src = dynamic_cast<PreviewSource*>(s); src != nullptr) {
          pc = src->preview_channel();
        }
      }
      if (!pc || pc->closed()) {
        report_(*ch, "waiting", pipeline, stage, title);
        continue;
      }

      report_(*ch, "playing", pipeline, stage, title);
      relay_(*ch, pc, generation);
      // The relay ended (stream closed or designation changed): fall
      // back through the loop, which re-reports the current state.
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
  // The (pipeline, stage) the panel is following, and a counter bumped
  // on every change so a running relay notices it has been superseded.
  std::string             _want_pipeline;
  std::string             _want_stage;
  std::uint64_t           _generation = 0;
  // Last transition announced, so a steady state isn't re-sent.
  std::string             _reported;
};

std::unique_ptr<UiViewBackendIntf>
make_preview_backend(UiViewHostIntf& host)
{
  return std::make_unique<PreviewViewBackend>(host);
}

const UiViewSpec kView{
  .id           = "preview",
  .stage_type   = kStageType,
  .module       = "/ui/stages/preview/preview-view.js",
  .styles       = "/ui/stages/preview/preview-view.css",
  .label_key    = "preview.panel",
  .icon         = "video",
  .make_backend = &make_preview_backend,
};

}

VPIPE_REGISTER_UI_VIEW(preview, kView)

}

// Backend for the "create-mask" stage's GUI mask editor.
//
// Same skeleton as the compare-image backend -- the panel names a stage
// to follow and a worker relays that stage's channel -- with one thing
// neither preview nor compare-image has: traffic in the OTHER direction.
// A mask editor exists to hand something back, so this also carries the
// painted mask up to the stage.
//
// PROTOCOL (view -> backend):
//   {m:"list"}                      enumerate declared create-mask stages
//   {m:"watch", pipeline, stage}    follow this stage
//   {m:"unwatch"}                   stop following
//   {m:"commit", png:"<base64>"}    the painted mask; the stage emits
//                                   ONE beat per accepted commit
//
// PROTOCOL (backend -> view):
//   {m:"stages", list:[{pipeline,stage,title,state,live}]}
//   {m:"waiting"|"playing"|"gone", pipeline, stage[, title]}
//   {m:"frame", width, height, bg_width, bg_height, has_bg, has_mask,
//               version, editor:{mode,classes,colors,overlay_opacity,
//                                interactive}}
//   {m:"committed", seq}            the commit reached the stage
//   binary {m:"image", slot:"bg"|"mask"} + PNG bytes
//
// The frame message ALWAYS precedes the images of that version, so the
// view knows what to blank before any bytes arrive; a slot that is
// false in `frame` gets no binary frame at all.
//
// The commit rides as BASE64 inside a JSON message rather than as a
// binary frame: the view -> backend direction of the transport is
// structured messages only, and a commit is a once-per-click event, so
// the encoding it pays for is irrelevant next to the round trip it
// saves inventing.

#include "ui/ui-view-registry.h"

#include "common/flex-data.h"
#include "common/mask-editor-channel.h"
#include "common/media-line.h"
#include "pipeline/stage.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
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

constexpr const char* kStageType = "create-mask";

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

// The editor settings, as the view wants them: a mode NAME rather than
// an ordinal (the view switches on strings) and colours as "#rrggbb".
FlexData
editor_obj(const MaskEditorChannel::Editor& e)
{
  FlexData o = FlexData::make_object();
  auto oo = o.as_object();
  const char* mode = "binary";
  if (e.mode == MaskEditorChannel::Mode::Alpha) { mode = "alpha"; }
  if (e.mode == MaskEditorChannel::Mode::Class) { mode = "class"; }
  oo.insert("mode", FlexData::make_string(mode));
  oo.insert("classes", FlexData::make_int(e.classes));
  oo.insert("overlay_opacity",
            FlexData::make_real(static_cast<double>(e.overlay_opacity)));
  oo.insert("interactive", FlexData::make_bool(e.interactive));
  FlexData arr = FlexData::make_array();
  {
    auto a = arr.as_array();
    for (std::uint32_t c : e.colors) {
      char buf[8];
      std::snprintf(buf, sizeof(buf), "#%06x", c & 0xffffffu);
      a.push_back(FlexData::make_string(buf));
    }
  }
  oo.insert("colors", std::move(arr));
  return o;
}

class MaskViewBackend final : public UiViewBackendIntf {
public:
  explicit MaskViewBackend(UiViewHostIntf& host) : _host(host) {}

  ~MaskViewBackend() override { stop_worker_(); }

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
    if (m == "commit") {
      handle_commit_(ch, msg);
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

  // Resolve the followed stage's channel, or null. Taken fresh on each
  // use: the pipeline may have been relaunched under the panel, and the
  // Stage* is only valid while it stays launched.
  std::shared_ptr<MaskEditorChannel>
  channel_() const
  {
    std::string pipeline;
    std::string stage;
    {
      std::lock_guard<std::mutex> lk(_mu);
      pipeline = _want_pipeline;
      stage    = _want_stage;
    }
    if (pipeline.empty()) { return nullptr; }
    Stage* s = _host.live_stage(pipeline, stage);
    if (s == nullptr) { return nullptr; }
    auto* src = dynamic_cast<MaskEditorSource*>(s);
    return src != nullptr ? src->mask_channel() : nullptr;
  }

  // A painted mask on its way up. Acknowledged with the sequence the
  // stage assigned it, so the view can tell "the stage took it" from
  // "the socket ate it" -- the difference between a beat and nothing.
  void
  handle_commit_(UiViewChannel& ch, const FlexData& msg)
  {
    auto cc = channel_();
    if (!cc || cc->closed()) { return; }
    auto bytes = media_line::base64_decode(str_field(msg, "png"));
    if (!bytes || bytes->empty()) { return; }
    const std::uint64_t seq = cc->commit(std::move(*bytes));
    FlexData m = msg_obj("committed");
    auto mo = m.as_object();
    mo.insert("seq", FlexData::make_uint(seq));
    ch.send(m);
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

  void
  send_frame_(UiViewChannel& ch, const MaskEditorChannel::Frame& f)
  {
    FlexData m = msg_obj("frame");
    {
      auto mo = m.as_object();
      mo.insert("width", FlexData::make_int(f.width));
      mo.insert("height", FlexData::make_int(f.height));
      mo.insert("bg_width", FlexData::make_int(f.bg_width));
      mo.insert("bg_height", FlexData::make_int(f.bg_height));
      mo.insert("has_bg", FlexData::make_bool(f.background != nullptr));
      mo.insert("has_mask", FlexData::make_bool(f.mask != nullptr));
      mo.insert("version",
                FlexData::make_uint(static_cast<std::uint64_t>(f.version)));
      mo.insert("editor", editor_obj(f.editor));
    }
    ch.send(m);

    auto push = [&](const char* slot, const MaskEditorChannel::Bytes& png) {
      if (!png || png->empty()) { return; }
      FlexData h = msg_obj("image");
      auto ho = h.as_object();
      ho.insert("slot", FlexData::make_string(slot));
      ch.send_binary(h, png->data(), png->size());
    };
    push("bg", f.background);
    push("mask", f.mask);
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
         const std::shared_ptr<MaskEditorChannel>& cc,
         std::uint64_t generation)
  {
    std::uint64_t seen = 0;
    // Send whatever is already latched, so a panel mounting mid-run
    // opens on the current mask instead of waiting for the next change.
    {
      const auto f = cc->snapshot();
      if (f.version != 0) {
        send_frame_(ch, f);
        seen = f.version;
      }
    }
    while (!superseded_(generation) && ch.alive()) {
      const auto f = cc->wait_change(seen, 500);
      if (f.closed) { break; }
      if (f.version != seen) {
        seen = f.version;
        send_frame_(ch, f);
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

      std::shared_ptr<MaskEditorChannel> cc = channel_();
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
make_mask_backend(UiViewHostIntf& host)
{
  return std::make_unique<MaskViewBackend>(host);
}

const UiViewSpec kView{
  .id           = "create-mask",
  .stage_type   = kStageType,
  .module       = "/ui/stages/create-mask/create-mask-view.js",
  .styles       = "/ui/stages/create-mask/create-mask-view.css",
  .label_key    = "mask.panel",
  .icon         = "image",
  .make_backend = &make_mask_backend,
};

}

VPIPE_REGISTER_UI_VIEW(create_mask, kView)

}

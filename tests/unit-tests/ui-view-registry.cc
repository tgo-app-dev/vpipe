// Stage-provided GUI views: the backend half of the split.
//
// What matters here is that a view registers itself from its own TU, is
// discoverable by id, carries its front-end asset bytes in the same
// library, and that a backend can be constructed and driven over the
// FlexData protocol without any front end present. The preview view is
// the worked example, so it doubles as the fixture.

#include "minitest.h"

#include "common/flex-data.h"
#include "interfaces/ui-view-intf.h"
#include "pipeline/stage.h"
#include "ui/ui-view-registry.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

// Records everything a backend sends, so a test can assert on the
// protocol without a browser or a socket.
class CapturingChannel final : public UiViewChannel {
public:
  struct Sent {
    bool        binary = false;
    FlexData    msg;       // structured message, or a binary frame header
    size_t      bytes = 0; // payload size, binary frames only
  };

  void
  send(const FlexData& msg) override
  {
    lock_guard<mutex> lk(_mu);
    _sent.push_back(Sent{false, msg, 0});
  }

  void
  send_binary(const FlexData& header, const uint8_t*, size_t size) override
  {
    lock_guard<mutex> lk(_mu);
    _sent.push_back(Sent{true, header, size});
  }

  bool alive() const override { return true; }

  vector<Sent>
  drain()
  {
    lock_guard<mutex> lk(_mu);
    vector<Sent> out;
    out.swap(_sent);
    return out;
  }

private:
  mutable mutex _mu;
  vector<Sent>  _sent;
};

// A host with a scripted set of stages and no live pipeline: enough to
// exercise discovery, which is all a stopped-pipeline view can do.
class FakeHost final : public UiViewHostIntf {
public:
  vector<StageRef> stages;

  vector<StageRef>
  find_stages(string_view type_name) const override
  {
    if (type_name != "preview") { return {}; }
    return stages;
  }

  Stage*
  live_stage(string_view, string_view) const override
  {
    return nullptr;   // nothing launched in these tests
  }
};

string
str_of(const FlexData& obj, string_view key)
{
  if (!obj.is_object()) { return {}; }
  auto o = obj.as_object();
  if (!o.contains(key)) { return {}; }
  const FlexData v = o.at(key);
  return string(v.as_string(""));
}

FlexData
request(string_view m)
{
  FlexData f = FlexData::make_object();
  auto o = f.as_object();
  o.insert("m", FlexData::make_string(m));
  return f;
}

}  // namespace

// The preview stage registers its view from its own translation unit --
// nothing in the app or the test references it by symbol, so this also
// proves VPIPE_REGISTER_UI_VIEW fires at static-init.
TEST(ui_view_registry, preview_view_is_registered)
{
  const UiViewSpec* v = UiViewRegistry::get().find("preview");
  ASSERT_TRUE(v != nullptr);
  EXPECT_TRUE(v->stage_type == "preview");
  EXPECT_TRUE(v->module == "/ui/stages/preview/preview-view.js");
  EXPECT_TRUE(v->make_backend != nullptr);
  EXPECT_TRUE(!v->label_key.empty());
}

// A SECOND stage registering its own panel is the point of the split:
// nothing about the app, the registry, or the preview view changed to
// make room for it.
TEST(ui_view_registry, compare_image_view_is_registered)
{
  auto& reg = UiViewRegistry::get();
  const UiViewSpec* v = reg.find("compare-image");
  ASSERT_TRUE(v != nullptr);
  EXPECT_TRUE(v->stage_type == "compare-image");
  EXPECT_TRUE(v->make_backend != nullptr);
  EXPECT_TRUE(reg.find_asset(v->module) != nullptr);
  EXPECT_TRUE(reg.find_asset(v->styles) != nullptr);
  // Its message catalogue and the shared pan/zoom controller it imports
  // must be servable too, or the module fails to load in the browser.
  EXPECT_TRUE(
      reg.find_asset("/ui/stages/compare-image/compare-image-strings.js")
      != nullptr);
  EXPECT_TRUE(reg.find_asset("/ui/sdk/pan-zoom.js") != nullptr);
  // Two distinct views now coexist under one registry.
  EXPECT_TRUE(reg.all().size() >= 2);
  EXPECT_TRUE(reg.find("preview") != v);
}

// The mask editor is the first view that carries traffic UPWARDS -- it
// commits a painted mask back to its stage -- but it declares itself the
// same way the read-only panels do, which is what says the mechanism did
// not need widening to accept it.
TEST(ui_view_registry, create_mask_view_is_registered)
{
  auto& reg = UiViewRegistry::get();
  const UiViewSpec* v = reg.find("create-mask");
  ASSERT_TRUE(v != nullptr);
  EXPECT_TRUE(v->stage_type == "create-mask");
  EXPECT_TRUE(v->make_backend != nullptr);
  EXPECT_TRUE(reg.find_asset(v->module) != nullptr);
  EXPECT_TRUE(reg.find_asset(v->styles) != nullptr);
  EXPECT_TRUE(
      reg.find_asset("/ui/stages/create-mask/create-mask-strings.js")
      != nullptr);
  EXPECT_TRUE(reg.find_asset("/ui/sdk/pan-zoom.js") != nullptr);
  EXPECT_TRUE(reg.all().size() >= 3);
}

TEST(ui_view_registry, unknown_id_is_not_found)
{
  EXPECT_TRUE(UiViewRegistry::get().find("no-such-view") == nullptr);
  EXPECT_TRUE(UiViewRegistry::get().all().size() >= 1);
}

// The view's front-end sources are embedded in THIS library, not in the
// front-end app: the module, its stylesheet, its message catalogue and
// the shared channel SDK all resolve to real bytes.
TEST(ui_view_registry, view_assets_are_embedded)
{
  auto& reg = UiViewRegistry::get();
  for (const char* path : {
         "/ui/sdk/view-channel.js",
         "/ui/stages/preview/preview-view.js",
         "/ui/stages/preview/preview-view.css",
         "/ui/stages/preview/preview-strings.js",
       }) {
    const UiViewAsset* a = reg.find_asset(path);
    ASSERT_TRUE(a != nullptr);
    EXPECT_TRUE(a->size > 0);
    EXPECT_TRUE(a->data != nullptr);
  }
  EXPECT_TRUE(reg.find_asset("/ui/nope.js") == nullptr);
  // The spec's declared entry points must actually be servable.
  const UiViewSpec* v = reg.find("preview");
  ASSERT_TRUE(v != nullptr);
  EXPECT_TRUE(reg.find_asset(v->module) != nullptr);
  EXPECT_TRUE(reg.find_asset(v->styles) != nullptr);
}

// A "list" request is answered with the declared stages, including ones
// whose pipeline is only loaded -- the picker must be able to follow a
// stage that has not been launched.
TEST(ui_view_registry, preview_backend_answers_list)
{
  const UiViewSpec* v = UiViewRegistry::get().find("preview");
  ASSERT_TRUE(v != nullptr);

  FakeHost host;
  UiViewHostIntf::StageRef r;
  r.pipeline = "p1";
  r.stage    = "pv";
  r.state    = "stopped";
  r.live     = false;
  r.config   = FlexData::make_object();
  {
    auto co = r.config.as_object();
    co.insert("title", FlexData::make_string("Front door"));
  }
  host.stages.push_back(r);

  CapturingChannel ch;
  auto backend = v->make_backend(host);
  ASSERT_TRUE(backend != nullptr);
  backend->on_message(ch, request("list"));

  auto sent = ch.drain();
  ASSERT_TRUE(sent.size() == 1);
  EXPECT_TRUE(!sent[0].binary);
  EXPECT_TRUE(str_of(sent[0].msg, "m") == "stages");

  auto mo = sent[0].msg.as_object();
  ASSERT_TRUE(mo.contains("list"));
  const FlexData list = mo.at("list");
  auto la = list.as_array();
  ASSERT_TRUE(la.size() == 1);
  const FlexData e = la[0];
  EXPECT_TRUE(str_of(e, "pipeline") == "p1");
  EXPECT_TRUE(str_of(e, "stage") == "pv");
  // The `title` config becomes the display label; the id is the fallback.
  EXPECT_TRUE(str_of(e, "title") == "Front door");
  EXPECT_TRUE(str_of(e, "state") == "stopped");

  backend->on_close();
}

// Following a stage whose pipeline is not loaded at all reports "gone",
// so a saved layout drops a dead target instead of waiting forever.
TEST(ui_view_registry, preview_backend_reports_gone_for_unknown_stage)
{
  const UiViewSpec* v = UiViewRegistry::get().find("preview");
  ASSERT_TRUE(v != nullptr);

  FakeHost host;                      // declares no stages at all
  CapturingChannel ch;
  auto backend = v->make_backend(host);
  ASSERT_TRUE(backend != nullptr);
  backend->on_open(ch);               // starts the watch loop

  FlexData watch = request("watch");
  {
    auto wo = watch.as_object();
    wo.insert("pipeline", FlexData::make_string("ghost"));
    wo.insert("stage", FlexData::make_string("pv"));
  }
  backend->on_message(ch, watch);

  // The watch loop wakes on the new designation; poll briefly for it.
  bool saw_gone = false;
  for (int i = 0; i < 100 && !saw_gone; ++i) {
    for (const auto& s : ch.drain()) {
      if (!s.binary && str_of(s.msg, "m") == "gone") { saw_gone = true; }
    }
    if (!saw_gone) {
      this_thread::sleep_for(chrono::milliseconds(20));
    }
  }
  EXPECT_TRUE(saw_gone);

  backend->on_close();
}

// The host API's in-memory forms (include/vpipe/): a session config, a
// pipeline spec and command arguments passed as FlexData documents, and a
// command's result read back as one -- no JSON text written or parsed.
// Public calls, plus the SDK's common/flex-data.h for the document type,
// as a host application sees them -- and the Session behind the interface
// once, to see that a config document was actually read.

#include "minitest.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "vpipe/vpipe.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

FlexData
str_(string_view s)
{
  return FlexData::make_string(s);
}

// {"src": src, "oport": 0}
FlexData
edge_(string_view src)
{
  FlexData e = FlexData::make_object();
  auto o = e.as_object();
  o.insert("src", str_(src));
  o.insert("oport", FlexData::make_int(0));
  return e;
}

FlexData
stage_(string_view id, string_view type, FlexData iports, FlexData config)
{
  FlexData s = FlexData::make_object();
  auto o = s.as_object();
  o.insert("id", str_(id));
  o.insert("type", str_(type));
  o.insert("iports", std::move(iports));
  o.insert("config", std::move(config));
  return s;
}

// in (external-input) -> tap (external-tap, pass), built in code.
FlexData
graph_()
{
  FlexData stages = FlexData::make_array();
  auto a = stages.as_array();
  a.push_back(stage_("in", "external-input", FlexData::make_array(),
                     FlexData::make_object()));
  FlexData ip = FlexData::make_array();
  ip.as_array().push_back(edge_("in"));
  a.push_back(stage_("tap", "external-tap", std::move(ip),
                     FlexData::make_object()));
  FlexData g = FlexData::make_object();
  auto o = g.as_object();
  o.insert("id", str_("host-flex"));
  o.insert("stages", std::move(stages));
  o.insert("subpipelines", FlexData::make_array());
  return g;
}

DataBuffer
floats_(vector<float> v, vector<int64_t> shape)
{
  auto own = make_shared<vector<float>>(std::move(v));
  return DataBuffer::view(
      "data", own->data(),
      BufferLayout::contiguous(ElementType::F32, std::move(shape)), own);
}

}  // namespace

TEST(host_flex_api, session_from_a_config_document)
{
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("language", str_("zh-Hans"));
  const SessionIntf* s = SessionManager::get().create_session(cfg);
  ASSERT_TRUE(s != nullptr);
  if (s != nullptr) {
    // The document was read, not skipped: its language took (normalized).
    const auto* impl = dynamic_cast<const Session*>(s);
    ASSERT_TRUE(impl != nullptr);
    if (impl != nullptr) { EXPECT_TRUE(impl->language() == "zh-cn"); }
    SessionManager::get().destroy_session(s);
  }

  // Null is the built-in defaults; a non-object is refused (a warning)
  // and the defaults are used -- a session either way, as for bad text.
  for (const FlexData& c : {FlexData(), FlexData::make_array()}) {
    const SessionIntf* d = SessionManager::get().create_session(c);
    EXPECT_TRUE(d != nullptr);
    if (d != nullptr) {
      const auto* impl = dynamic_cast<const Session*>(d);
      EXPECT_TRUE(impl != nullptr && impl->language() == "en-us");
      SessionManager::get().destroy_session(d);
    }
  }
}

TEST(host_flex_api, pipeline_and_commands_without_json_text)
{
  auto* s = const_cast<SessionIntf*>(SessionManager::get().create_session());
  ASSERT_TRUE(s != nullptr);
  if (s == nullptr) { return; }
  PipelineHandle p = s->load_pipeline(graph_());
  ASSERT_TRUE(p.valid());
  if (!p.valid()) { SessionManager::get().destroy_session(s); return; }
  StageHandle in = p.stage("in");
  StageHandle tap = p.stage("tap");
  ASSERT_TRUE(in.valid() && tap.valid());
  ASSERT_TRUE(s->launch_pipeline(p).code == 0);

  // Arguments as a document, the result read back as one: the stage's
  // unsigned `seq` arrives as Uint, which JSON text would not say.
  {
    FlexData sb = FlexData::make_object();
    sb.as_object().insert("k", FlexData::make_uint(7));
    FlexData args = FlexData::make_object();
    args.as_object().insert("sideband", std::move(sb));
    CommandHandle push = in.command("push", args,
                                    {floats_({1.f, 2.f}, {2})});
    EXPECT_TRUE(push.wait(3000) == CommandState::Replied);
    const FlexData r = push.result();
    ASSERT_TRUE(r.is_object());
    if (r.is_object()) {
      const auto o = r.as_object();
      EXPECT_TRUE(o.contains("seq") && o.at("seq").is_uint() &&
                  o.at("seq").as_uint() == 1);
    }
  }

  // A document is checked against the declaration exactly as JSON is:
  // an unknown argument is refused, by name.
  {
    FlexData bad = FlexData::make_object();
    bad.as_object().insert("sidebnad", FlexData::make_object());
    CommandHandle typo = in.command("push", bad, {floats_({1.f}, {1})});
    EXPECT_TRUE(typo.state() == CommandState::Failed);
    EXPECT_TRUE(typo.error().find("sidebnad") != string::npos);
  }

  // Null arguments are no arguments; before a reply, result() is Null.
  {
    CommandHandle fin = in.command("finish", FlexData());
    EXPECT_TRUE(fin.wait(3000) == CommandState::Replied);
    EXPECT_TRUE(fin.result().is_object());
    CommandHandle none;
    EXPECT_TRUE(none.result().is_null());
  }

  (void)s->wait_pipelines(3000);
  EXPECT_TRUE(s->unload_pipeline(p).code == 0);
  SessionManager::get().destroy_session(s);
}

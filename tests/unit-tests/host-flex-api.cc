// The host API's in-memory forms (include/vpipe/): a session config, a
// pipeline spec and command arguments passed as FlexData documents, and a
// command's result read back as one -- no JSON text written or parsed.
// Public calls, plus the SDK's common/flex-data.h for the document type,
// as a host application sees them -- and the Session behind the interface
// once, to see that a config document was actually read.

#include "minitest.h"
#include "common/flex-data.h"
#include "common/host-report.h"
#include "common/session.h"
#include "common/vpipe-format.h"
#include "vpipe/vpipe.h"

#include <cstdint>
#include <memory>
#include <mutex>
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

// The live progress reports, read as a host reads them: the document
// /api/io/progress serves, from SessionIntf alone. The Session behind the
// interface opens the report, as a stage would through its context.
TEST(host_flex_api, progress_reports_as_a_document)
{
  const SessionIntf* s = SessionManager::get().create_session();
  ASSERT_TRUE(s != nullptr);
  if (s == nullptr) { return; }
  const auto* impl = dynamic_cast<const Session*>(s);
  ASSERT_TRUE(impl != nullptr);
  if (impl == nullptr) { SessionManager::get().destroy_session(s); return; }

  // The views borrow the document they view: hold each one by name.
  auto items = [&] { return s->progress().as_object().at("items"); };
  const uint64_t v0 = s->progress_version();
  const FlexData before = s->progress();
  EXPECT_TRUE(before.as_object().at("version").as_uint() == v0);
  EXPECT_TRUE(items().as_array().size() == 0);
  {
    UiProgress bar = impl->open_progress("denoise");
    bar.update(3, 10, "step 1/4");
    UiProgress dl = impl->open_progress("fetch");   // indeterminate
    EXPECT_TRUE(s->progress_version() > v0);
    const FlexData live = items();
    const auto a = live.as_array();
    ASSERT_TRUE(a.size() == 2);
    if (a.size() == 2) {
      // Oldest-opened first; ids only grow.
      const FlexData d0 = a[0];
      const FlexData f0 = a[1];
      const auto d = d0.as_object();
      EXPECT_TRUE(d.at("desc").as_string() == "denoise");
      EXPECT_TRUE(d.at("done").as_uint() == 3);
      EXPECT_TRUE(d.at("total").as_uint() == 10);
      EXPECT_TRUE(d.at("detail").as_string() == "step 1/4");
      EXPECT_TRUE(d.at("elapsed_ms").is_uint());
      const auto f = f0.as_object();
      EXPECT_TRUE(f.at("desc").as_string() == "fetch");
      EXPECT_TRUE(f.at("total").as_uint() == 0);
      EXPECT_TRUE(f.at("id").as_uint() > d.at("id").as_uint());
      // The most recently moved has the highest seq.
      bar.update(4, 10);
      const FlexData after = items();
      const auto b = after.as_array();
      const FlexData b0 = b[0];
      const FlexData b1 = b[1];
      EXPECT_TRUE(b0.as_object().at("seq").as_uint() >
                  b1.as_object().at("seq").as_uint());
    }
  }
  // Closed with their handles: nothing is live.
  EXPECT_TRUE(items().as_array().size() == 0);
  SessionManager::get().destroy_session(s);
}

// SessionIntf::set_log_listener: a host hears what the session reports
// -- info and warnings always, the diagnostic log at or above its level
// -- and an empty listener stops it.
TEST(host_flex_api, log_listener_hears_the_session)
{
  SessionIntf* s = const_cast<SessionIntf*>(
      SessionManager::get().create_session());
  ASSERT_TRUE(s != nullptr);
  if (s == nullptr) { return; }
  const auto* impl = dynamic_cast<const Session*>(s);
  ASSERT_TRUE(impl != nullptr);
  if (impl == nullptr) { SessionManager::get().destroy_session(s); return; }

  std::mutex mu;
  std::vector<std::pair<int, std::string>> heard;
  s->set_log_listener([&](int level, std::string_view text) {
    std::lock_guard<std::mutex> lk(mu);
    heard.emplace_back(level, std::string(text));
  });
  impl->info(fmt("hello {}", 7));
  impl->warn(fmt("careful"));
  impl->log_always(fmt("always"));
  impl->log_debug(fmt("debug, below the default level"));
  {
    std::lock_guard<std::mutex> lk(mu);
    ASSERT_TRUE(heard.size() == 3);
    if (heard.size() == 3) {
      EXPECT_TRUE(heard[0].first == 2 && heard[0].second == "hello 7");
      EXPECT_TRUE(heard[1].first == 1 && heard[1].second == "careful");
      EXPECT_TRUE(heard[2].first == 6 && heard[2].second == "always");
    }
  }
  s->set_log_listener({});
  impl->info(fmt("unheard"));
  {
    std::lock_guard<std::mutex> lk(mu);
    EXPECT_TRUE(heard.size() == 3);
  }
  SessionManager::get().destroy_session(s);
}

// SessionIntf::reports(): what a stage refused and why, as numbers a host
// can act on (common/host-report.h) -- the newest 32, oldest first, ids
// only growing, the version the newest.
TEST(host_flex_api, reports_reach_the_host_as_a_document)
{
  const SessionIntf* s = SessionManager::get().create_session();
  ASSERT_TRUE(s != nullptr);
  if (s == nullptr) { return; }
  const auto* impl = dynamic_cast<const Session*>(s);
  ASSERT_TRUE(impl != nullptr);
  if (impl == nullptr) { SessionManager::get().destroy_session(s); return; }

  const uint64_t v0 = s->reports_version();
  {
    FlexData d = FlexData::make_object();
    d.as_object().insert("need", FlexData::make_uint(3u << 30));
    d.as_object().insert("step", FlexData::make_string("denoise"));
    host_report(impl, "memory", std::move(d));
  }
  EXPECT_TRUE(s->reports_version() == v0 + 1);
  {
    const FlexData doc = s->reports();
    EXPECT_TRUE(doc.as_object().at("version").as_uint() == v0 + 1);
    const FlexData items = doc.as_object().at("items");
    ASSERT_TRUE(items.as_array().size() >= 1);
    const FlexData last = items.as_array()[items.as_array().size() - 1];
    const auto o = last.as_object();
    EXPECT_TRUE(o.at("id").as_uint() == v0 + 1);
    EXPECT_TRUE(o.at("kind").as_string() == "memory");
    EXPECT_TRUE(o.at("age_ms").is_uint());
    const FlexData data = o.at("data");
    EXPECT_TRUE(data.as_object().at("need").as_uint() == (3ull << 30));
    EXPECT_TRUE(data.as_object().at("step").as_string() == "denoise");
  }
  // Bounded: the newest 32 kept, the oldest gone.
  for (int i = 0; i < 40; ++i) {
    host_report(impl, "memory-plan", FlexData::make_object());
  }
  {
    const FlexData doc = s->reports();
    const FlexData items = doc.as_object().at("items");
    EXPECT_TRUE(items.as_array().size() == 32);
    const FlexData first = items.as_array()[0];
    EXPECT_TRUE(first.as_object().at("id").as_uint() == v0 + 41 - 31);
    EXPECT_TRUE(doc.as_object().at("version").as_uint() == v0 + 41);
  }
  // A context that is not vpipe's own Session drops the report.
  host_report(nullptr, "memory", FlexData::make_object());
  EXPECT_TRUE(s->reports_version() == v0 + 41);
  SessionManager::get().destroy_session(s);
}

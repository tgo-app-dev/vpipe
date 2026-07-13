#include "minitest.h"

#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/chrono-stage.h"
#include "stages/load-text-stage.h"
#include "stages/text-prompt-stage.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <streambuf>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

class CerrSilencer {
public:
  CerrSilencer() : _saved(cerr.rdbuf()), _null() { cerr.rdbuf(&_null); }
  ~CerrSilencer() { cerr.rdbuf(_saved); }
private:
  struct NullBuf : public streambuf {
    int overflow(int c) override { return c; }
  };
  streambuf* _saved;
  NullBuf    _null;
};

// Test-only sink: 1 iport, captures every payload until EOS.
class SinkCapture : public TypedStage<SinkCapture> {
public:
  static constexpr const char* kTypeName = "ut-sink-capture-text";
  using TypedStage::TypedStage;
  vector<unique_ptr<BeatPayloadIntf>> captured;
  Job
  process(RuntimeContext& ctx) override
  {
    auto p = co_await ctx.read(0);
    if (!p) { ctx.signal_done(); co_return; }
    captured.push_back(std::move(p));
  }
};

// The captured payload's string (empty if not a FlexData string payload).
string
captured_str_(const unique_ptr<BeatPayloadIntf>& p)
{
  const auto* fd = dynamic_cast<const FlexDataPayload*>(p.get());
  return fd ? string(fd->data.as_string("")) : string();
}

}  // namespace

// ===== text-prompt =====

// Construction succeeds for any config; a missing text is deferred to launch.
TEST(text_prompt_stage, config_text_required_deferred) {
  Session sess;
  TextPromptStage s(&sess, "tp", vector<InEdge>{}, FlexData::make_object());
  EXPECT_FALSE(s.config_error().empty());
  EXPECT_TRUE(s.num_oports() == 1);
}

// No iport: emits the configured text once as a FlexData string, then ends.
TEST(text_prompt_stage, emits_one_beat) {
  Session sess;
  CerrSilencer hush;
  auto pl = make_unique<Pipeline>("p", &sess);

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("text", FlexData::make_string("a fox in the snow"));
  auto tp_u = make_unique<TextPromptStage>(
      &sess, "tp", vector<InEdge>{}, std::move(cfg));
  auto* tp = static_cast<TextPromptStage*>(pl->insert_stage(std::move(tp_u)));

  auto sink_u = make_unique<SinkCapture>(
      &sess, "sink", vector<InEdge>{{tp, 0}}, FlexData::make_object());
  auto* sink = static_cast<SinkCapture*>(pl->insert_stage(std::move(sink_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  EXPECT_TRUE(sink->captured.size() == 1);
  if (!sink->captured.empty()) {
    EXPECT_TRUE(captured_str_(sink->captured[0]) == "a fox in the snow");
  }
}

// Trigger iport: each chrono tick re-emits the same text (3 ticks -> 3 beats).
TEST(text_prompt_stage, chrono_paced_reemits) {
  Session sess;
  CerrSilencer hush;
  auto pl = make_unique<Pipeline>("p", &sess);

  FlexData chrono_cfg = FlexData::from_json(
      R"({"frequency_hz":1000.0,"count":3})");
  auto ck_u = make_unique<ChronoStage>(
      &sess, "ck", vector<InEdge>{}, std::move(chrono_cfg));
  auto* ck = static_cast<ChronoStage*>(pl->insert_stage(std::move(ck_u)));

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("text", FlexData::make_string("prompt"));
  auto tp_u = make_unique<TextPromptStage>(
      &sess, "tp", vector<InEdge>{{ck, 0}}, std::move(cfg));
  auto* tp = static_cast<TextPromptStage*>(pl->insert_stage(std::move(tp_u)));

  auto sink_u = make_unique<SinkCapture>(
      &sess, "sink", vector<InEdge>{{tp, 0}}, FlexData::make_object());
  auto* sink = static_cast<SinkCapture*>(pl->insert_stage(std::move(sink_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  EXPECT_TRUE(sink->captured.size() == 3);
  if (sink->captured.size() == 3) {
    EXPECT_TRUE(captured_str_(sink->captured[0]) == "prompt");
    EXPECT_TRUE(captured_str_(sink->captured[2]) == "prompt");
  }
}

// ===== load-text =====

TEST(load_text_stage, config_path_string) {
  Session sess;
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("path", FlexData::make_string("/tmp/x.txt"));
  LoadTextStage s(&sess, "lt", vector<InEdge>{}, std::move(cfg));
  EXPECT_TRUE(s.paths().size() == 1);
  EXPECT_TRUE(s.paths()[0] == "/tmp/x.txt");
  EXPECT_TRUE(s.num_oports() == 1);
}

TEST(load_text_stage, config_path_required_deferred) {
  Session sess;
  LoadTextStage s(&sess, "lt", vector<InEdge>{}, FlexData::make_object());
  EXPECT_FALSE(s.config_error().empty());
}

// End-to-end: write a temp text file, run the pipeline, expect one FlexData
// string beat carrying the file's contents.
TEST(load_text_stage, reads_file) {
  Session sess;
  CerrSilencer hush;
  const string path =
      string("/tmp/vpipe-load-text-") + to_string(getpid()) + ".txt";
  const string contents = "a fox in the snow\nsecond line";
  { ofstream out(path, ios::binary); out << contents; }

  auto pl = make_unique<Pipeline>("p", &sess);
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("path", FlexData::make_string(path));
  auto lt_u = make_unique<LoadTextStage>(
      &sess, "lt", vector<InEdge>{}, std::move(cfg));
  auto* lt = static_cast<LoadTextStage*>(pl->insert_stage(std::move(lt_u)));

  auto sink_u = make_unique<SinkCapture>(
      &sess, "sink", vector<InEdge>{{lt, 0}}, FlexData::make_object());
  auto* sink = static_cast<SinkCapture*>(pl->insert_stage(std::move(sink_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();
  remove(path.c_str());

  EXPECT_TRUE(sink->captured.size() == 1);
  if (!sink->captured.empty()) {
    EXPECT_TRUE(captured_str_(sink->captured[0]) == contents);
  }
}

// Multi-path emits exactly one beat per file.
TEST(load_text_stage, multi_path_emits_each) {
  Session sess;
  CerrSilencer hush;
  const string p1 =
      string("/tmp/vpipe-load-text-") + to_string(getpid()) + "-a.txt";
  const string p2 =
      string("/tmp/vpipe-load-text-") + to_string(getpid()) + "-b.txt";
  { ofstream o(p1); o << "one"; }
  { ofstream o(p2); o << "two"; }

  auto pl = make_unique<Pipeline>("p", &sess);
  FlexData cfg = FlexData::make_object();
  {
    FlexData paths = FlexData::make_array();
    paths.as_array().push_back(string_view(p1));
    paths.as_array().push_back(string_view(p2));
    cfg.as_object().insert("path", std::move(paths));
  }
  auto lt_u = make_unique<LoadTextStage>(
      &sess, "lt", vector<InEdge>{}, std::move(cfg));
  auto* lt = static_cast<LoadTextStage*>(pl->insert_stage(std::move(lt_u)));

  auto sink_u = make_unique<SinkCapture>(
      &sess, "sink", vector<InEdge>{{lt, 0}}, FlexData::make_object());
  auto* sink = static_cast<SinkCapture*>(pl->insert_stage(std::move(sink_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();
  remove(p1.c_str());
  remove(p2.c_str());

  EXPECT_TRUE(sink->captured.size() == 2);
  if (sink->captured.size() == 2) {
    EXPECT_TRUE(captured_str_(sink->captured[0]) == "one");
    EXPECT_TRUE(captured_str_(sink->captured[1]) == "two");
  }
}

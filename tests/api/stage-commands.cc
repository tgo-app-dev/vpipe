// Stage commands through the PUBLIC API only (include/vpipe/), as an
// application linking libvpipe sees them: load a pipeline from JSON,
// find its stages by id, push data in, read it back out in place.

#include "minitest.h"
#include "vpipe/vpipe.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

// in (external-input) -> tap (external-tap, gate) -> sink (external-tap,
// pass, which nothing reads -- a place for the beats to go).
const char* kGraph = R"({
  "id": "api-commands",
  "stages": [
    {"id": "in",  "type": "external-input", "iports": [], "config": {}},
    {"id": "tap", "type": "external-tap",
     "iports": [{"src": "in", "oport": 0}], "config": {"mode": "gate"}},
    {"id": "sink", "type": "external-tap",
     "iports": [{"src": "tap", "oport": 0}], "config": {}}
  ],
  "subpipelines": []
})";

DataBuffer
floats_(const vector<float>& v, vector<int64_t> shape)
{
  auto own = make_shared<vector<float>>(v);
  return DataBuffer::view(
      "data", own->data(),
      BufferLayout::contiguous(ElementType::F32, std::move(shape)), own);
}

}  // namespace

TEST(api_stage_commands, push_then_read_in_place)
{
  auto* s = const_cast<SessionIntf*>(SessionManager::get().create_session());
  ASSERT_TRUE(s != nullptr);
  PipelineHandle p = s->load_pipeline(kGraph);
  ASSERT_TRUE(p.valid());
  StageHandle in  = p.stage("in");
  StageHandle tap = p.stage("tap");
  ASSERT_TRUE(in.valid() && tap.valid());
  EXPECT_TRUE(in.id() == "in");
  EXPECT_FALSE(p.stage("nope").valid());

  // Discoverable before launch; refused before launch.
  const string cj = tap.commands_json();
  EXPECT_TRUE(cj.find("\"read\"") != string::npos);
  EXPECT_TRUE(cj.find("\"holds\":true") != string::npos);
  {
    CommandHandle early = tap.command("read");
    EXPECT_TRUE(early.state() == CommandState::Failed);
    EXPECT_TRUE(early.error().find("not running") != string::npos);
  }

  ASSERT_TRUE(s->launch_pipeline(p).code == 0);

  {
    CommandHandle push = in.command("push", R"({"sideband":{"k":7}})",
                                    {floats_({1.f, 2.f, 3.f, 4.f}, {2, 2})});
    EXPECT_TRUE(push.wait(3000) == CommandState::Replied);
    EXPECT_TRUE(push.result_json().find("\"seq\":1") != string::npos);
  }

  {
    CommandHandle rd = tap.command("read");
    ASSERT_TRUE(rd.wait(3000) == CommandState::Replied);
    DataBuffer b;
    ASSERT_TRUE(rd.buffer("data", &b));
    EXPECT_TRUE(b.writable);
    EXPECT_TRUE(b.layout.type == ElementType::F32);
    EXPECT_TRUE(b.layout.shape == (vector<int64_t>{2, 2}));
    EXPECT_TRUE(static_cast<const float*>(b.data)[3] == 4.f);
    EXPECT_TRUE(rd.result_json().find("\"k\":7") != string::npos);
  }   // the handle's destructor CLOSES the command: the tap moves on

  // Refusals are values, not exceptions.
  CommandHandle typo = in.command("push", R"({"sidebnad":{}})",
                                  {floats_({1.f}, {1})});
  EXPECT_TRUE(typo.state() == CommandState::Failed);
  EXPECT_TRUE(typo.error().find("sidebnad") != string::npos);
  CommandHandle junk = in.command("push", "{not json");
  EXPECT_TRUE(junk.state() == CommandState::Failed);

  EXPECT_TRUE(s->stop_pipeline(p).code == 0);
  EXPECT_TRUE(s->unload_pipeline(p).code == 0);
  SessionManager::get().destroy_session(s);
}

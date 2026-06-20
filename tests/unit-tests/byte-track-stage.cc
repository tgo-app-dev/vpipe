#include "minitest.h"

#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "common/vertex.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/stage-registry.h"
#include "pipeline/typed-stage.h"
#include "stages/vision/byte-track-stage.h"

#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <streambuf>
#include <string>
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

// One FlexData detection record. The test builds a sequence of these
// (one per frame) and feeds them through byte-track.
struct DetSpec {
  int    cls;
  double x1, y1, x2, y2;
  double score;
};

FlexData
make_frame_(int frame_w, int frame_h, const vector<DetSpec>& dets)
{
  FlexData out = FlexData::make_object();
  auto root = out.as_object();
  root.insert("frame_width",  FlexData::make_int(frame_w));
  root.insert("frame_height", FlexData::make_int(frame_h));
  FlexData arr = FlexData::make_array();
  auto v = arr.as_array();
  for (const auto& d : dets) {
    FlexData entry = FlexData::make_object();
    auto e = entry.as_object();
    e.insert("class_id", FlexData::make_int(d.cls));
    e.insert("score",    FlexData::make_real(d.score));
    e.insert("x1",       FlexData::make_real(d.x1));
    e.insert("y1",       FlexData::make_real(d.y1));
    e.insert("x2",       FlexData::make_real(d.x2));
    e.insert("y2",       FlexData::make_real(d.y2));
    v.push_back(std::move(entry));
  }
  root.insert("detections", std::move(arr));
  return out;
}

// Emits a preconfigured sequence of FlexData frames, then signals
// done. One iteration of process() emits exactly one frame.
class FlexDataReplaySource : public TypedStage<FlexDataReplaySource> {
public:
  static constexpr const char* kTypeName = "ut-bt-flex-replay";
  using TypedStage::TypedStage;

  vector<FlexData> frames;

  Job process(RuntimeContext& ctx) override
  {
    if (_i >= frames.size()) {
      ctx.signal_done();
      co_return;
    }
    auto p = make_payload<FlexDataPayload>(frames[_i]);
    _i += 1;
    co_await ctx.write(0, std::move(p));
  }

private:
  size_t _i = 0;
};

// Collects FlexDataPayloads one per frame.
class FlexDataSink : public TypedStage<FlexDataSink> {
public:
  static constexpr const char* kTypeName = "ut-bt-flex-sink";
  using TypedStage::TypedStage;

  vector<FlexData> collected;

  Job process(RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) {
      ctx.signal_done();
      co_return;
    }
    const FlexDataPayload* p =
        dynamic_cast<const FlexDataPayload*>(in.get());
    if (p) {
      lock_guard<mutex> g(_mu);
      collected.push_back(p->data);
    }
  }

private:
  mutex _mu;
};

struct Harness {
  Session                  sess;
  unique_ptr<Pipeline>     pl;
  FlexDataReplaySource*    src  = nullptr;
  ByteTrackStage*          bt   = nullptr;
  FlexDataSink*            sink = nullptr;
};

unique_ptr<Harness>
build_(vector<FlexData> frames, FlexData bt_cfg = FlexData::make_object())
{
  auto h = make_unique<Harness>();
  h->pl = make_unique<Pipeline>("p", &h->sess);

  auto src_u = make_unique<FlexDataReplaySource>(
      &h->sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->frames = std::move(frames);
  src_u->allocate_oports(1);
  h->src = static_cast<FlexDataReplaySource*>(
      h->pl->insert_stage(std::move(src_u)));

  auto bt_u = make_unique<ByteTrackStage>(
      &h->sess, "bt", vector<InEdge>{{h->src, 0}}, std::move(bt_cfg));
  h->bt = static_cast<ByteTrackStage*>(
      h->pl->insert_stage(std::move(bt_u)));

  auto sink_u = make_unique<FlexDataSink>(
      &h->sess, "sink", vector<InEdge>{{h->bt, 0}},
      FlexData::make_object());
  h->sink = static_cast<FlexDataSink*>(
      h->pl->insert_stage(std::move(sink_u)));

  return h;
}

bool
run_(Harness& h)
{
  PipelineRuntime rt(h.pl.get(), &h.sess);
  if (!rt.launch()) {
    return false;
  }
  rt.wait_idle();
  rt.stop();
  return true;
}

// Pull the array of track ids present in one output frame (in input
// detection-order is not preserved; the test never relies on order).
set<int>
ids_in_frame_(const FlexData& frame)
{
  set<int> out;
  if (!frame.is_object()) { return out; }
  auto root = frame.as_object();
  if (!root.contains("detections")) { return out; }
  auto arr = root.at("detections");
  if (!arr.is_array()) { return out; }
  auto v = arr.as_array();
  for (size_t i = 0; i < v.size(); ++i) {
    auto e = v.at(i);
    if (!e.is_object()) { continue; }
    auto eo = e.as_object();
    if (!eo.contains("track_id")) { continue; }
    out.insert(static_cast<int>(eo.at("track_id").as_int(-1)));
  }
  return out;
}

size_t
count_detections_(const FlexData& frame)
{
  if (!frame.is_object()) { return 0; }
  auto root = frame.as_object();
  if (!root.contains("detections")) { return 0; }
  auto arr = root.at("detections");
  if (!arr.is_array()) { return 0; }
  return arr.as_array().size();
}

}  // namespace

TEST(byte_track_stage, type_is_registered)
{
  EXPECT_TRUE(StageRegistry::get().find_id("byte-track") !=
              StageTypeId::unknown);
}

TEST(byte_track_stage, stable_track_id_across_frames)
{
  // A single 50x50 box drifting right by 5 px per frame should be
  // tracked with one stable id from frame 2 onward. (Frame 1
  // activates the track; per the reference, only frame 1 sets
  // is_activated=true at activate-time; otherwise the track has to
  // be matched once before being emitted.)
  CerrSilencer hush;

  vector<FlexData> frames;
  for (int f = 0; f < 6; ++f) {
    const double x = 100.0 + f * 5.0;
    frames.push_back(make_frame_(640, 480, {
      { /*cls*/ 0, x, 100.0, x + 50.0, 150.0, /*score*/ 0.9 },
    }));
  }

  auto h = build_(std::move(frames));
  EXPECT_TRUE(run_(*h));

  // The first frame's emit may be empty (track Active only on f==1
  // by activate, then is_activated stays false until first update).
  // From the second frame onward we should consistently see one id.
  EXPECT_TRUE(h->sink->collected.size() == 6u);

  // Collect the *set of ids* from frame 1 onward; intersect across
  // frames to confirm one stable id.
  set<int> running;
  bool first = true;
  for (size_t i = 1; i < h->sink->collected.size(); ++i) {
    auto s = ids_in_frame_(h->sink->collected[i]);
    if (first) {
      running = s;
      first   = false;
    } else {
      set<int> inter;
      for (int id : s) {
        if (running.count(id)) { inter.insert(id); }
      }
      running = inter;
    }
  }
  EXPECT_TRUE(running.size() == 1u);
}

TEST(byte_track_stage, two_distinct_objects_get_distinct_ids)
{
  // Two non-overlapping boxes drifting in opposite directions.
  // Expect two distinct ids that each remain stable.
  CerrSilencer hush;

  vector<FlexData> frames;
  for (int f = 0; f < 5; ++f) {
    const double ax = 80.0  + f * 4.0;
    const double bx = 380.0 - f * 4.0;
    frames.push_back(make_frame_(640, 480, {
      { 0, ax,  80.0, ax + 40.0,  120.0, 0.9 },
      { 0, bx, 300.0, bx + 40.0,  340.0, 0.9 },
    }));
  }

  auto h = build_(std::move(frames));
  EXPECT_TRUE(run_(*h));

  // Look at the last frame: should contain 2 detections, with 2
  // distinct track ids.
  EXPECT_TRUE(h->sink->collected.size() == 5u);
  auto last = ids_in_frame_(h->sink->collected.back());
  EXPECT_TRUE(last.size() == 2u);
}

TEST(byte_track_stage, low_score_detection_never_spawns_track)
{
  // Every frame has one detection at score 0.4 -- below the default
  // high_thresh (0.6). The reference algorithm never spawns a new
  // track from such a detection; the output should be empty across
  // every frame.
  CerrSilencer hush;

  vector<FlexData> frames;
  for (int f = 0; f < 4; ++f) {
    frames.push_back(make_frame_(640, 480, {
      { 0, 100.0, 100.0, 150.0, 150.0, 0.4 },
    }));
  }

  auto h = build_(std::move(frames));
  EXPECT_TRUE(run_(*h));

  EXPECT_TRUE(h->sink->collected.size() == 4u);
  for (const auto& frame : h->sink->collected) {
    EXPECT_TRUE(count_detections_(frame) == 0u);
  }
}

TEST(byte_track_stage, output_preserves_frame_dimensions_and_class_id)
{
  // The output FlexData should echo frame_width / frame_height from
  // the input, and each tracked detection should carry class_id.
  CerrSilencer hush;

  vector<FlexData> frames;
  for (int f = 0; f < 4; ++f) {
    const double x = 100.0 + f * 3.0;
    frames.push_back(make_frame_(1280, 720, {
      { /*cls*/ 7, x, 100.0, x + 50.0, 160.0, 0.95 },
    }));
  }

  auto h = build_(std::move(frames));
  EXPECT_TRUE(run_(*h));

  EXPECT_TRUE(h->sink->collected.size() == 4u);
  // The last frame is past activation -- check its annotations.
  auto& fd = h->sink->collected.back();
  EXPECT_TRUE(fd.is_object());
  auto root = fd.as_object();
  EXPECT_TRUE(root.at("frame_width").as_int(0)  == 1280);
  EXPECT_TRUE(root.at("frame_height").as_int(0) == 720);
  // Bind the array FlexData into a named local before taking the
  // view -- ConstArrayView holds a pointer into the FlexData, and
  // the FlexData returned by root.at(...) is a temporary that would
  // otherwise die at the end of the full-expression.
  FlexData dets_fd = root.at("detections");
  auto dets = dets_fd.as_array();
  EXPECT_TRUE(dets.size() >= 1u);
  FlexData entry_fd = dets.at(0);
  auto e = entry_fd.as_object();
  EXPECT_TRUE(e.contains("class_id"));
  EXPECT_TRUE(e.at("class_id").as_int(-1) == 7);
  EXPECT_TRUE(e.contains("track_id"));
}

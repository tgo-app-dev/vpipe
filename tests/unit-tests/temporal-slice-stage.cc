// temporal-slice: Python's seq[start:end:step] over the beat stream.
//
// The oracle is Python's own rule, written out once in expect_() and
// applied to every case, because the whole value of this stage is that
// an author who knows what `frames[-1]` does already knows what this
// does. A per-case hand-written expectation would be a second guess at
// the same rule.
//
// Each beat carries its index as its payload, so a case asserts WHICH
// beats came through and in what order -- a count alone would pass a
// stage that emitted the right number of wrong frames.
//
//   vpipe_test --filter '*temporal_slice*'

#include "minitest.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/audio-video/temporal-slice-stage.h"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace vpipe;

namespace {

// Emits `count` beats, the i-th carrying i as its single element and in
// its sideband, then ends.
class IndexSource : public TypedStage<IndexSource> {
public:
  static constexpr const char* kTypeName = "ut-tsl-source";
  using TypedStage::TypedStage;

  int count = 0;
  int next  = 0;

  Job
  process(RuntimeContext& ctx) override
  {
    if (next >= count) { ctx.signal_done(); co_return; }
    const std::int64_t i = next++;
    TensorBeat tb;
    tb.dtype = TensorBeat::DType::U8;
    tb.shape = {1};
    tb.data.assign(1, (std::uint8_t)i);
    FlexData o = FlexData::make_object();
    o.as_object().insert_or_assign("index", FlexData::make_int(i));
    o.as_object().insert_or_assign("marker",
                                   FlexData::make_string("from-source"));
    tb.sideband = std::move(o);
    co_await ctx.write(0, make_payload<TensorBeatPayload>(std::move(tb)));
    co_return;
  }
};

class IndexSink : public TypedStage<IndexSink> {
public:
  static constexpr const char* kTypeName = "ut-tsl-sink";
  using TypedStage::TypedStage;

  std::vector<int> got;              // payload bytes, in arrival order
  std::vector<std::int64_t> sb_idx;  // sideband `index`, same order
  std::vector<std::string> markers;
  std::vector<std::size_t> ranks;

  Job
  process(RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) { ctx.signal_done(); co_return; }
    if (const auto* tb = dynamic_cast<const TensorBeatPayload*>(in.get())) {
      got.push_back(tb->data.empty() ? -1 : (int)tb->data[0]);
      ranks.push_back(tb->shape.size());
      FlexData sb = tb->sideband;
      if (sb.is_object()) {
        auto o = sb.as_object();
        sb_idx.push_back(o.contains("index") ? o.at("index").as_int(-1) : -1);
        markers.push_back(o.contains("marker")
                              ? std::string(o.at("marker").as_string(""))
                              : std::string());
      } else {
        sb_idx.push_back(-1);
        markers.push_back(std::string());
      }
    }
    co_return;
  }
};

struct Run {
  Session sess;
  std::unique_ptr<Pipeline> pl;
  IndexSink* sink = nullptr;
  TemporalSliceStage* slice = nullptr;
};

// source(count) -> temporal-slice(cfg) -> sink
bool
drive_(Run& r, int count, FlexData cfg)
{
  r.pl = std::make_unique<Pipeline>("p", &r.sess);
  auto src_u = std::make_unique<IndexSource>(&r.sess, "src",
                                             std::vector<InEdge>{},
                                             FlexData::make_object());
  src_u->count = count;
  auto* src = static_cast<IndexSource*>(r.pl->insert_stage(std::move(src_u)));
  src->allocate_oports(1);
  auto sl_u = std::make_unique<TemporalSliceStage>(
      &r.sess, "sl", std::vector<InEdge>{{src, 0}}, std::move(cfg));
  r.slice =
      static_cast<TemporalSliceStage*>(r.pl->insert_stage(std::move(sl_u)));
  auto sink_u = std::make_unique<IndexSink>(&r.sess, "sink",
                                            std::vector<InEdge>{{r.slice, 0}},
                                            FlexData::make_object());
  r.sink = static_cast<IndexSink*>(r.pl->insert_stage(std::move(sink_u)));
  PipelineRuntime rt(r.pl.get(), &r.sess);
  if (!rt.launch()) { return false; }
  rt.wait_idle();
  rt.stop();
  return true;
}

// source(count) -> { temporal-slice(cfg) -> sink,  witness }
//
// The witness is what makes this a FAN-OUT: with two cursors on the
// source's oport, the slice is not always the slowest one, so its
// read() takes the peek+clone path instead of moving the slot out (see
// EdgeReader::ReadAwaiter::await_resume). That is the case where a held
// beat could conceivably pin the producer, so it is the one to measure.
bool
drive_fanout_(Run& r, int count, FlexData cfg, int* witness_beats)
{
  r.pl = std::make_unique<Pipeline>("p", &r.sess);
  auto src_u = std::make_unique<IndexSource>(&r.sess, "src",
                                             std::vector<InEdge>{},
                                             FlexData::make_object());
  src_u->count = count;
  auto* src = static_cast<IndexSource*>(r.pl->insert_stage(std::move(src_u)));
  src->allocate_oports(1);
  auto sl_u = std::make_unique<TemporalSliceStage>(
      &r.sess, "sl", std::vector<InEdge>{{src, 0}}, std::move(cfg));
  r.slice =
      static_cast<TemporalSliceStage*>(r.pl->insert_stage(std::move(sl_u)));
  auto sink_u = std::make_unique<IndexSink>(&r.sess, "sink",
                                            std::vector<InEdge>{{r.slice, 0}},
                                            FlexData::make_object());
  r.sink = static_cast<IndexSink*>(r.pl->insert_stage(std::move(sink_u)));
  // The second reader of the SAME oport.
  auto wit_u = std::make_unique<IndexSink>(&r.sess, "wit",
                                           std::vector<InEdge>{{src, 0}},
                                           FlexData::make_object());
  auto* wit = static_cast<IndexSink*>(r.pl->insert_stage(std::move(wit_u)));
  PipelineRuntime rt(r.pl.get(), &r.sess);
  if (!rt.launch()) { return false; }
  rt.wait_idle();
  rt.stop();
  if (witness_beats) { *witness_beats = (int)wit->got.size(); }
  return true;
}

FlexData
cfg_(const int* start, const int* step, const int* end)
{
  FlexData c = FlexData::make_object();
  auto o = c.as_object();
  if (start) { o.insert_or_assign("start", FlexData::make_int(*start)); }
  if (step)  { o.insert_or_assign("step",  FlexData::make_int(*step)); }
  if (end)   { o.insert_or_assign("end",   FlexData::make_int(*end)); }
  return c;
}

// Python's slice rule, written once. `has_*` mirror a key being absent
// from the config rather than carrying a default, which is the
// distinction 0 cannot express.
std::vector<int>
expect_(int n, bool has_start, int start, int step, bool has_end, int end)
{
  int s = 0;
  if (has_start) {
    s = start < 0 ? std::max(n + start, 0) : std::min(start, n);
  }
  int e = n;
  if (has_end) {
    e = end < 0 ? std::max(n + end, 0) : std::min(end, n);
  }
  std::vector<int> out;
  for (int i = s; i < e; i += step) { out.push_back(i); }
  return out;
}

bool
same_(const std::vector<int>& a, const std::vector<int>& b)
{
  if (a.size() != b.size()) { return false; }
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) { return false; }
  }
  return true;
}

std::string
join_(const std::vector<int>& v)
{
  std::string s = "[";
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (i) { s += ","; }
    s += std::to_string(v[i]);
  }
  return s + "]";
}

}  // namespace

// The motivating case, and the one that cannot be done by streaming: the
// LAST beat of a clip, which is only identifiable once the source has
// said there are no more.
TEST(temporal_slice, start_minus_one_keeps_the_last_beat)
{
  Run r;
  const int start = -1;
  ASSERT_TRUE(drive_(r, 22, cfg_(&start, nullptr, nullptr)));
  ASSERT_TRUE(r.sink != nullptr);
  EXPECT_TRUE(same_(r.sink->got, {21}));
  // Held exactly one beat to do it, not the clip.
  EXPECT_TRUE(r.slice->seen() == 22);
  EXPECT_TRUE(r.slice->emitted() == 1);
}

// Every combination against Python's own answer. The negative cases are
// the point, but the non-negative ones share one code path with them and
// would break together.
TEST(temporal_slice, matches_python_slice_semantics)
{
  struct Case { bool hs; int s; int st; bool he; int e; };
  const Case cases[] = {
    {false, 0, 1, false, 0},     // [:]
    {true,  0, 1, false, 0},     // [0:]
    {true,  3, 1, false, 0},     // [3:]
    {true,  0, 2, false, 0},     // [::2]
    {true,  1, 3, false, 0},     // [1::3]
    {true, -1, 1, false, 0},     // [-1:]
    {true, -3, 1, false, 0},     // [-3:]
    {true, -5, 2, false, 0},     // [-5::2]
    {true,  0, 1, true,  4},     // [0:4]
    {true,  2, 1, true,  7},     // [2:7]
    {true,  2, 2, true,  9},     // [2:9:2]
    {true,  0, 1, true, -1},     // [:-1]   -- everything but the last
    {true,  0, 2, true, -2},     // [:-2:2]
    {true, -6, 1, true, -2},     // [-6:-2]
    {true, -6, 3, true, -1},     // [-6:-1:3]
    {true,  3, 1, true,  3},     // empty
    {true, 50, 1, false, 0},     // start past the end -> empty
    {true,-50, 1, false, 0},     // start before the start -> clamps to 0
    {true,  0, 1, true, 50},     // end past the end -> clamps to n
    {true,  0, 1, true,-50},     // end before the start -> empty
  };
  const int n = 10;
  for (const Case& c : cases) {
    Run r;
    const int s = c.s, st = c.st, e = c.e;
    ASSERT_TRUE(drive_(r, n, cfg_(c.hs ? &s : nullptr, &st,
                                  c.he ? &e : nullptr)));
    ASSERT_TRUE(r.sink != nullptr);
    const std::vector<int> want = expect_(n, c.hs, c.s, c.st, c.he, c.e);
    if (!same_(r.sink->got, want)) {
      std::printf("[temporal_slice] [%s:%s:%d] over %d -> got %s want %s\n",
                  c.hs ? std::to_string(c.s).c_str() : "",
                  c.he ? std::to_string(c.e).c_str() : "", c.st, n,
                  join_(r.sink->got).c_str(), join_(want).c_str());
    }
    EXPECT_TRUE(same_(r.sink->got, want));
  }
}

// The beat is FORWARDED, not rebuilt: same payload, same rank, and the
// sideband the source attached arrives intact. Rebuilding would lose
// whatever a producer put there that this stage has never heard of.
TEST(temporal_slice, forwards_the_beat_and_its_sideband_untouched)
{
  Run r;
  const int start = -2;
  ASSERT_TRUE(drive_(r, 6, cfg_(&start, nullptr, nullptr)));
  ASSERT_TRUE(r.sink != nullptr);
  ASSERT_TRUE(r.sink->got.size() == 2);
  if (r.sink->got.size() != 2) { return; }
  EXPECT_TRUE(same_(r.sink->got, {4, 5}));
  // The sideband travelled with the beat, including a key this stage
  // knows nothing about.
  EXPECT_TRUE(r.sink->sb_idx[0] == 4 && r.sink->sb_idx[1] == 5);
  EXPECT_TRUE(r.sink->markers[0] == "from-source");
  EXPECT_TRUE(r.sink->markers[1] == "from-source");
  // Not stacked: still rank 1, the shape the source emitted.
  EXPECT_TRUE(r.sink->ranks[0] == 1 && r.sink->ranks[1] == 1);
}

// A wholly non-negative slice never needs the total, so it must not hold
// anything -- it is the case that has to work on a live source that
// never reaches EOS. Asserted by watching the beats arrive BEFORE the
// source has finished.
TEST(temporal_slice, a_non_negative_slice_streams_without_holding)
{
  Run r;
  const int start = 0, step = 2;
  ASSERT_TRUE(drive_(r, 7, cfg_(&start, &step, nullptr)));
  ASSERT_TRUE(r.sink != nullptr);
  EXPECT_TRUE(same_(r.sink->got, {0, 2, 4, 6}));
  EXPECT_TRUE(r.slice->emitted() == 4);
}

// step <= 0 is refused at CONFIG time, not silently coerced: a stream is
// read once and in order, so a negative step names beats that are gone.
TEST(temporal_slice, a_non_positive_step_is_a_config_error)
{
  Session sess;
  for (int bad : {0, -1, -2}) {
    const int start = 0;
    TemporalSliceStage st(&sess, "sl", std::vector<InEdge>{},
                          cfg_(&start, &bad, nullptr));
    EXPECT_TRUE(!st.config_error().empty());
  }
  // ...and a positive one is accepted.
  const int start = 0, good = 3;
  TemporalSliceStage ok(&sess, "sl", std::vector<InEdge>{},
                        cfg_(&start, &good, nullptr));
  EXPECT_TRUE(ok.config_error().empty());
}

// An empty source is not an error, and a slice of nothing is nothing --
// it must not emit a beat to say so.
TEST(temporal_slice, an_empty_stream_yields_an_empty_slice)
{
  Run r;
  const int start = -1;
  ASSERT_TRUE(drive_(r, 0, cfg_(&start, nullptr, nullptr)));
  ASSERT_TRUE(r.sink != nullptr);
  EXPECT_TRUE(r.sink->got.empty());
  EXPECT_TRUE(r.slice->emitted() == 0);
}

// WHAT A NEGATIVE INDEX COSTS, measured rather than reasoned about: the
// window never holds more than |start| beats, and holding them neither
// stalls the producer nor starves a co-reader.
//
// The fan-out is the point. With a second cursor on the source's oport
// the slice is often not the slowest reader, so read() hands it a CLONE
// of the beat rather than moving the buffer slot out -- a private copy,
// which is exactly why holding it cannot pin the producer. The witness
// still receives every beat, which is the observable form of "the
// producer was never held back".
TEST(temporal_slice, a_negative_index_holds_only_that_many_beats)
{
  struct Case { int start; int want_hold; };
  const Case cases[] = {{-1, 1}, {-2, 2}, {-5, 5}};
  const int n = 40;
  for (const Case& c : cases) {
    Run r;
    int witness = 0;
    const int start = c.start;
    ASSERT_TRUE(drive_fanout_(r, n, cfg_(&start, nullptr, nullptr),
                              &witness));
    ASSERT_TRUE(r.slice != nullptr && r.sink != nullptr);
    if (r.slice == nullptr) { return; }
    if (r.slice->peak_hold() != c.want_hold) {
      std::printf("[temporal_slice] start=%d over %d beats: held %lld, "
                  "expected %d\n", c.start, n,
                  (long long)r.slice->peak_hold(), c.want_hold);
    }
    // EXACTLY |start|, not |start|+1: room is made before the new beat
    // is taken, so the window never transiently overshoots.
    EXPECT_TRUE(r.slice->peak_hold() == c.want_hold);
    // It really did see the whole stream while holding that little...
    EXPECT_TRUE(r.slice->seen() == n);
    EXPECT_TRUE((int)r.sink->got.size() == c.want_hold);
    // ...and the co-reader was never starved, so nothing was pinned.
    EXPECT_TRUE(witness == n);
  }
}

// A wholly non-negative slice holds NOTHING -- not one beat -- which is
// what lets it run on a source that never reaches EOS.
TEST(temporal_slice, a_non_negative_slice_holds_nothing)
{
  Run r;
  int witness = 0;
  const int start = 2, step = 3;
  ASSERT_TRUE(drive_fanout_(r, 40, cfg_(&start, &step, nullptr), &witness));
  ASSERT_TRUE(r.slice != nullptr);
  if (r.slice == nullptr) { return; }
  EXPECT_TRUE(r.slice->peak_hold() == 0);
  EXPECT_TRUE(witness == 40);
}

// No start, no step, no end is Python's `[:]`: every beat, in order,
// holding nothing. An unconfigured stage is a pass-through.
TEST(temporal_slice, an_unconfigured_slice_is_a_passthrough)
{
  Session sess;
  TemporalSliceStage st(&sess, "sl", std::vector<InEdge>{},
                        FlexData::make_object());
  EXPECT_TRUE(st.config_error().empty());

  Run r;
  ASSERT_TRUE(drive_(r, 6, FlexData::make_object()));
  ASSERT_TRUE(r.sink != nullptr);
  EXPECT_TRUE(same_(r.sink->got, {0, 1, 2, 3, 4, 5}));
  EXPECT_TRUE(r.slice->peak_hold() == 0);
}

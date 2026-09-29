// Stage command channels (pipeline/stage-command.h, docs/STAGE-COMMANDS.md):
// the layout grammar, the host's validation, and the protocol end to end
// through the two generic stages -- external-input (data IN, copied or
// leased) and external-tap (data OUT, in place, held).
//
// What these pin is the CONTRACT rather than any one stage:
//   * N commands are N requests, served in order -- never merged;
//   * a reply's buffers are the stage's own memory, and a lease's close
//     emits that very memory when the caller let go of it;
//   * a stage that holds a command holds the pipeline: nothing moves
//     downstream until the close;
//   * a stop ends every wait -- a held command and a queued one are both
//     cancelled, and stop() returns.

#include "minitest.h"

#include "apple-silicon/tensor-beat.h"
#include "common/job.h"
#include "common/session.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/stage-command.h"
#include "pipeline/typed-stage.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

// Latches every tensor it receives, and the address its bytes arrived
// at -- which is how the lease test sees that nothing was copied.
class CmdCollector : public TypedStage<CmdCollector> {
public:
  static constexpr const char* kTypeName = "ut-cmd-collector";
  using TypedStage::TypedStage;

  Job process(RuntimeContext& ctx) override
  {
    auto beat = co_await ctx.read(0);
    if (!beat) {
      ctx.signal_done();
      co_return;
    }
    if (const auto* tb =
            dynamic_cast<const TensorBeatPayload*>(beat.get())) {
      lock_guard<mutex> lk(_mu);
      _addrs.push_back(tb->bytes_());
      _beats.push_back(*tb);
    }
  }

  size_t count() const
  {
    lock_guard<mutex> lk(_mu);
    return _beats.size();
  }
  TensorBeat at(size_t i) const
  {
    lock_guard<mutex> lk(_mu);
    return i < _beats.size() ? _beats[i] : TensorBeat{};
  }
  const void* addr(size_t i) const
  {
    lock_guard<mutex> lk(_mu);
    return i < _addrs.size() ? _addrs[i] : nullptr;
  }

private:
  mutable mutex       _mu;
  vector<TensorBeat>  _beats;
  vector<const void*> _addrs;
};

// external-input -> [external-tap ->] collector, launched.
struct Rig {
  Session                     sess;
  unique_ptr<Pipeline>        pl;
  Stage*                      in  = nullptr;
  Stage*                      tap = nullptr;
  CmdCollector*               col = nullptr;
  unique_ptr<PipelineRuntime> rt;

  // tap_mode "" = no tap.
  explicit Rig(const string& tap_mode)
  {
    pl = make_unique<Pipeline>("p", &sess);
    in = pl->insert_stage("external-input", "in", {});
    Stage* feed = in;
    if (!tap_mode.empty()) {
      FlexData cfg = FlexData::make_object();
      cfg.as_object().insert("mode", FlexData::make_string(tap_mode));
      tap  = pl->insert_stage("external-tap", "tap", {{in, 0}}, cfg);
      feed = tap;
    }
    auto c = make_unique<CmdCollector>(&sess, "c", vector<InEdge>{{feed, 0}},
                                       FlexData::make_object());
    col = static_cast<CmdCollector*>(pl->insert_stage(std::move(c)));
    rt  = make_unique<PipelineRuntime>(pl.get(), &sess);
  }
  ~Rig() { rt->stop(); }

  bool launch() { return rt->launch(); }

  // Wait up to `ms` for the collector to hold `n` beats.
  bool
  collected(size_t n, int ms = 3000) const
  {
    for (int i = 0; i < ms / 5; ++i) {
      if (col->count() >= n) { return true; }
      this_thread::sleep_for(chrono::milliseconds(5));
    }
    return col->count() >= n;
  }
};

DataBuffer
u8_buffer_(const char* name, vector<uint8_t> v, vector<int64_t> shape)
{
  auto own = make_shared<vector<uint8_t>>(std::move(v));
  return DataBuffer::view(name, own->data(),
                          BufferLayout::contiguous(ElementType::U8,
                                                   std::move(shape)),
                          own);
}

uint64_t
result_uint_(const shared_ptr<StageCommand>& c, const char* key)
{
  const FlexData r = c->result();
  return r.is_object() && r.as_object().contains(key)
             ? r.as_object().at(key).as_uint(0) : 0;
}

}  // namespace

// ---- layout -------------------------------------------------------------

TEST(stage_command, element_types_round_trip)
{
  for (ElementType t : {ElementType::Bytes, ElementType::U8,
                        ElementType::I16, ElementType::U32, ElementType::I64,
                        ElementType::F16, ElementType::BF16, ElementType::F32,
                        ElementType::F64}) {
    ElementType back{};
    EXPECT_TRUE(parse_element_type(element_type_name(t), &back));
    EXPECT_TRUE(back == t);
  }
  EXPECT_FALSE(parse_element_type("float", nullptr));
  EXPECT_TRUE(element_size(ElementType::BF16) == 2);
  EXPECT_TRUE(element_size(ElementType::F64) == 8);
}

// Strides are in BYTES and may pad rows; the extent is what a buffer
// must at least hold, and a unit dimension's stride never matters.
TEST(stage_command, layout_extent_and_contiguity)
{
  BufferLayout l = BufferLayout::contiguous(ElementType::F32, {2, 3});
  EXPECT_TRUE(l.is_contiguous());
  EXPECT_TRUE(l.extent_bytes() == 24);
  EXPECT_TRUE(l.contiguous_strides() == (vector<int64_t>{12, 4}));

  l.strides = {16, 4};   // rows padded to 16 bytes
  EXPECT_FALSE(l.is_contiguous());
  EXPECT_TRUE(l.extent_bytes() == 16 + 12);

  BufferLayout u = BufferLayout::contiguous(ElementType::U8, {1, 4});
  u.strides = {999, 1};  // numpy leaves junk on unit dims
  EXPECT_TRUE(u.is_contiguous());

  BufferLayout z = BufferLayout::contiguous(ElementType::U8, {0, 4});
  EXPECT_TRUE(z.extent_bytes() == 0);
  DataBuffer empty = DataBuffer::view("e", nullptr, z);
  EXPECT_TRUE(empty.valid());
}

TEST(stage_command, shape_grammar)
{
  auto m = [](string_view pat, vector<int64_t> s) {
    return match_shape(pat, s, nullptr);
  };
  EXPECT_TRUE(m("", {5, 6, 7}));             // any shape
  EXPECT_TRUE(m("3,H,W", {3, 4, 5}));
  EXPECT_FALSE(m("3,H,W", {4, 4, 5}));
  EXPECT_FALSE(m("3,H,W", {3, 4}));          // rank
  EXPECT_TRUE(m("?,?", {9, 1}));
  EXPECT_TRUE(m("N,N", {4, 4}));             // a name binds once...
  EXPECT_FALSE(m("N,N", {4, 5}));            // ...and must agree
  EXPECT_TRUE(m("1,H,W|H,W", {6, 8}));       // alternatives
  EXPECT_TRUE(m("1,H,W|H,W", {1, 6, 8}));
  EXPECT_FALSE(m("1,H,W|H,W", {2, 6, 8}));
  EXPECT_TRUE(m("...,3", {7, 7, 3}));        // leading ellipsis
  EXPECT_TRUE(m("B,...", {2}));              // it may match nothing
  EXPECT_TRUE(m("B,...,C", {2, 5, 5, 4}));
  EXPECT_FALSE(m("...,3,...", {3}));         // two ellipses: invalid
  EXPECT_TRUE(m("|H", {}));                  // "" = rank 0
  EXPECT_FALSE(m("H", {}));

  // Bindings carry ACROSS buffers, and a failed match binds nothing.
  ShapeSymbols syms;
  EXPECT_TRUE(match_shape("H,W", {4, 6}, &syms));
  EXPECT_FALSE(match_shape("W,H", {4, 6}, &syms));
  EXPECT_TRUE(match_shape("W,H", {6, 4}, &syms));
  EXPECT_FALSE(match_shape("H,X", {5, 1}, &syms));
  EXPECT_TRUE(syms.bound.size() == 2);       // X was not kept
}

// The host checks buffers against the spec so a stage -- a plugin's
// included -- never sees one that does not fit its declaration.
TEST(stage_command, buffers_are_checked_against_the_spec)
{
  const BufferSpec spec[] = {
    {.name = "img", .type = "u8,f32", .shape = "3,H,W"},
    {.name = "mask", .type = "u8", .shape = "H,W", .optional = true,
     .contiguous = true},
    {.name = "png", .type = "bytes", .format = "image/png",
     .optional = true},
  };
  string err;
  auto img = [] {
    return u8_buffer_("img", vector<uint8_t>(3 * 4 * 5), {3, 4, 5});
  };

  vector<DataBuffer> ok = {img(), u8_buffer_("mask",
                                             vector<uint8_t>(20), {4, 5})};
  EXPECT_TRUE(check_buffers(spec, ok, false, &err));
  // An inbound buffer is the caller's memory: never writable for the
  // stage, whatever the caller said.
  ok[0].writable = true;
  EXPECT_TRUE(check_buffers(spec, ok, false, &err));
  EXPECT_FALSE(ok[0].writable);

  vector<DataBuffer> missing = {};
  EXPECT_FALSE(check_buffers(spec, missing, false, &err));
  EXPECT_TRUE(err.find("requires buffer 'img'") != string::npos);

  vector<DataBuffer> unknown = {img(), u8_buffer_("alpha", {1}, {1})};
  EXPECT_FALSE(check_buffers(spec, unknown, false, &err));
  EXPECT_TRUE(err.find("alpha") != string::npos &&
              err.find("img, mask, png") != string::npos);

  vector<DataBuffer> twice = {img(), img()};
  EXPECT_FALSE(check_buffers(spec, twice, false, &err));

  // H,W bind across buffers: a mask the image's size, or nothing.
  vector<DataBuffer> skew = {img(), u8_buffer_("mask",
                                               vector<uint8_t>(20), {5, 4})};
  EXPECT_FALSE(check_buffers(spec, skew, false, &err));
  EXPECT_TRUE(err.find("mask") != string::npos &&
              err.find("[5,4]") != string::npos);

  vector<DataBuffer> type = {img()};
  type[0].layout.type = ElementType::I16;
  EXPECT_FALSE(check_buffers(spec, type, false, &err));

  // A strided mask where the spec wants it contiguous.
  auto own = make_shared<vector<uint8_t>>(4 * 8);
  DataBuffer strided = DataBuffer::view(
      "mask", own->data(), BufferLayout::contiguous(ElementType::U8, {4, 5}),
      own);
  strided.layout.strides = {8, 1};
  strided.size = own->size();
  vector<DataBuffer> st = {img(), strided};
  EXPECT_FALSE(check_buffers(spec, st, false, &err));
  EXPECT_TRUE(err.find("contiguous") != string::npos);

  // A layout that reaches past the buffer's end.
  vector<DataBuffer> small = {img()};
  small[0].size = 10;
  EXPECT_FALSE(check_buffers(spec, small, false, &err));

  // "bytes": a flat u8 string becomes Bytes and takes the spec's format;
  // another format is refused.
  vector<DataBuffer> png = {img(), u8_buffer_("png", {1, 2, 3}, {3})};
  EXPECT_TRUE(check_buffers(spec, png, false, &err));
  EXPECT_TRUE(png[1].layout.type == ElementType::Bytes);
  EXPECT_TRUE(png[1].layout.format == "image/png");
  vector<DataBuffer> jpg = {img(), u8_buffer_("png", {1, 2, 3}, {3})};
  jpg[1].layout.format = "image/jpeg";
  EXPECT_FALSE(check_buffers(spec, jpg, false, &err));
  vector<DataBuffer> not_flat = {img(), u8_buffer_("png", {1, 2}, {1, 2})};
  EXPECT_FALSE(check_buffers(spec, not_flat, false, &err));

  // OUT: writable is the spec's to say.
  const BufferSpec out[] = {{.name = "data", .writable = true}};
  vector<DataBuffer> o = {u8_buffer_("data", {1}, {1})};
  EXPECT_TRUE(check_buffers(out, o, true, &err));
  EXPECT_TRUE(o[0].writable);
}

TEST(stage_command, args_are_checked_and_defaults_filled)
{
  const ConfigKey spec[] = {
    {.key = "n", .type = ConfigType::Uint, .required = true},
    {.key = "gain", .type = ConfigType::Real, .def_real = 0.5},
    {.key = "tag", .type = ConfigType::String, .def_str = "x"},
  };
  string err;
  FlexData a = FlexData::from_json(R"({"n": 3})");
  EXPECT_TRUE(check_args(spec, a, &err));
  EXPECT_TRUE(a.as_object().at("gain").as_real(0) == 0.5);
  EXPECT_TRUE(string(a.as_object().at("tag").as_string("")) == "x");

  FlexData missing = FlexData::make_object();
  EXPECT_FALSE(check_args(spec, missing, &err));
  EXPECT_TRUE(err.find("'n' is required") != string::npos);

  FlexData typo = FlexData::from_json(R"({"n": 3, "gian": 1})");
  EXPECT_FALSE(check_args(spec, typo, &err));
  EXPECT_TRUE(err.find("gian") != string::npos &&
              err.find("n, gain, tag") != string::npos);

  FlexData neg = FlexData::from_json(R"({"n": -1})");
  EXPECT_FALSE(check_args(spec, neg, &err));
  FlexData real_ok = FlexData::from_json(R"({"n": 1, "gain": 2})");
  EXPECT_TRUE(check_args(spec, real_ok, &err));   // an int is a real
  FlexData arr = FlexData::from_json("[1]");
  EXPECT_FALSE(check_args(spec, arr, &err));
  FlexData null = FlexData::make_null();
  const ConfigKey none[] = {{.key = "k", .type = ConfigType::Int}};
  EXPECT_TRUE(check_args(none, null, &err));      // null means {}
}

// ---- external-input: data IN ---------------------------------------------

// N pushes are N beats, in order, each replied once the oport took it.
TEST(stage_command, push_is_one_beat_per_command_in_order)
{
  Rig rig("");
  ASSERT_TRUE(rig.launch());
  vector<shared_ptr<StageCommand>> cmds;
  for (uint8_t v = 1; v <= 3; ++v) {
    FlexData args = FlexData::make_object();
    FlexData sb = FlexData::make_object();
    sb.as_object().insert("frame", FlexData::make_uint(v));
    args.as_object().insert("sideband", sb);
    cmds.push_back(open_stage_command(
        *rig.in, "push", std::move(args),
        {u8_buffer_("data", vector<uint8_t>(6, v), {2, 3})}));
  }
  for (size_t i = 0; i < cmds.size(); ++i) {
    EXPECT_TRUE(cmds[i]->wait(3000) == CommandState::Replied);
    EXPECT_TRUE(result_uint_(cmds[i], "seq") == i + 1);
  }
  ASSERT_TRUE(rig.collected(3));
  for (size_t i = 0; i < 3; ++i) {
    const TensorBeat tb = rig.col->at(i);
    EXPECT_TRUE(tb.shape == (vector<int64_t>{2, 3}));
    EXPECT_TRUE(tb.as_u8()[5] == i + 1);
    EXPECT_TRUE(tb.sideband.is_object() &&
                tb.sideband.as_object().at("frame").as_uint(0) == i + 1);
  }
  // A strided push is gathered into a contiguous beat.
  auto own = make_shared<vector<uint8_t>>(vector<uint8_t>{
      1, 2, 3, 0, 4, 5, 6, 0});           // [2,3] in rows of 4
  DataBuffer st = DataBuffer::view(
      "data", own->data(), BufferLayout::contiguous(ElementType::U8, {2, 3}),
      own);
  st.layout.strides = {4, 1};
  st.size = own->size();
  auto sc = open_stage_command(*rig.in, "push", FlexData::make_object(),
                               {std::move(st)});
  EXPECT_TRUE(sc->wait(3000) == CommandState::Replied);
  ASSERT_TRUE(rig.collected(4));
  const TensorBeat g = rig.col->at(3);
  EXPECT_TRUE(g.is_contiguous() && g.as_u8()[3] == 4 && g.as_u8()[5] == 6);
}

// A lease is written IN PLACE and its close emits that very memory: the
// collector's beat lives at the address the caller wrote to. A caller
// that keeps its view past close gets a copy downstream instead, so the
// memory it still holds is not shared with the pipeline.
TEST(stage_command, lease_emits_the_callers_writes_without_a_copy)
{
  Rig rig("");
  ASSERT_TRUE(rig.launch());
  FlexData args = FlexData::from_json(R"({"type": "f32", "shape": [2, 3]})");
  auto cmd = open_stage_command(*rig.in, "lease", std::move(args), {});
  ASSERT_TRUE(cmd->wait(3000) == CommandState::Replied);
  const void* addr = nullptr;
  {
    auto out = cmd->outputs();
    ASSERT_TRUE(out.size() == 1 && out[0].writable);
    ASSERT_TRUE(out[0].layout.type == ElementType::F32);
    auto* f = static_cast<float*>(out[0].data);
    for (int i = 0; i < 6; ++i) { f[i] = 0.5f * i; }
    addr = out[0].data;
  }                                       // the view goes here
  // Nothing moves until the close.
  this_thread::sleep_for(chrono::milliseconds(50));
  EXPECT_TRUE(rig.col->count() == 0);
  cmd->close();
  ASSERT_TRUE(rig.collected(1));
  EXPECT_TRUE(rig.col->addr(0) == addr);
  EXPECT_TRUE(rig.col->at(0).as_f32()[5] == 2.5f);

  // Keep the view: the pipeline gets a copy, the caller keeps its bytes.
  auto c2 = open_stage_command(
      *rig.in, "lease",
      FlexData::from_json(R"({"type": "u8", "shape": [4]})"), {});
  ASSERT_TRUE(c2->wait(3000) == CommandState::Replied);
  DataBuffer kept = c2->outputs()[0];
  memset(kept.data, 7, 4);
  c2->close();
  ASSERT_TRUE(rig.collected(2));
  EXPECT_TRUE(rig.col->addr(1) != kept.data);
  EXPECT_TRUE(rig.col->at(1).as_u8()[3] == 7);
  EXPECT_TRUE(static_cast<uint8_t*>(kept.data)[3] == 7);   // still ours

  // A bad geometry is the stage's to refuse.
  auto bad = open_stage_command(
      *rig.in, "lease",
      FlexData::from_json(R"({"type": "f64", "shape": [4]})"), {});
  EXPECT_TRUE(bad->wait(3000) == CommandState::Failed);
}

// `finish` ends the stream, and the stage then refuses rather than
// strands: a command to a stage that has ended fails at once.
TEST(stage_command, finish_ends_the_stream_and_refuses_what_follows)
{
  Rig rig("");
  ASSERT_TRUE(rig.launch());
  auto p = open_stage_command(*rig.in, "push", FlexData::make_object(),
                              {u8_buffer_("data", {1}, {1})});
  auto f = open_stage_command(*rig.in, "finish", FlexData::make_object(), {});
  EXPECT_TRUE(f->wait(3000) == CommandState::Replied);
  EXPECT_TRUE(result_uint_(f, "beats") == 1);
  // The EOS cascades: the collector ends too.
  for (int i = 0; i < 600 && rig.in->command_inbox() &&
                  !rig.in->command_inbox()->shut(); ++i) {
    this_thread::sleep_for(chrono::milliseconds(5));
  }
  auto late = open_stage_command(*rig.in, "push", FlexData::make_object(),
                                 {u8_buffer_("data", {1}, {1})});
  EXPECT_TRUE(late->state() == CommandState::Failed);
  EXPECT_TRUE(late->error().find("finished") != string::npos);
}

// ---- external-tap: data OUT, in place ------------------------------------

// GATE: nothing passes until a read, the read exposes the beat's own
// bytes writable, and the pipeline downstream waits for the close --
// then carries the caller's edit.
TEST(stage_command, tap_holds_the_pipeline_until_the_read_closes)
{
  Rig rig("gate");
  ASSERT_TRUE(rig.launch());
  for (uint8_t v : {10, 20}) {
    open_stage_command(*rig.in, "push", FlexData::make_object(),
                       {u8_buffer_("data", vector<uint8_t>(4, v), {4})});
  }
  this_thread::sleep_for(chrono::milliseconds(100));
  EXPECT_TRUE(rig.col->count() == 0);                 // gated

  auto rd = open_stage_command(*rig.tap, "read", FlexData::make_object(), {});
  ASSERT_TRUE(rd->wait(3000) == CommandState::Replied);
  EXPECT_TRUE(result_uint_(rd, "seq") == 1);
  {
    auto out = rd->outputs();
    ASSERT_TRUE(out.size() == 1 && out[0].writable);
    auto* px = static_cast<uint8_t*>(out[0].data);
    EXPECT_TRUE(px[0] == 10);
    px[0] = 99;                                       // edit in place
  }
  this_thread::sleep_for(chrono::milliseconds(100));
  EXPECT_TRUE(rig.col->count() == 0);                 // HELD
  rd->close();
  ASSERT_TRUE(rig.collected(1));
  EXPECT_TRUE(rig.col->at(0).as_u8()[0] == 99);       // the edit moved on
  this_thread::sleep_for(chrono::milliseconds(50));
  EXPECT_TRUE(rig.col->count() == 1);                 // #2 still gated

  auto rd2 = open_stage_command(*rig.tap, "read", FlexData::make_object(),
                                {});
  ASSERT_TRUE(rd2->wait(3000) == CommandState::Replied);
  EXPECT_TRUE(static_cast<uint8_t*>(rd2->outputs()[0].data)[0] == 20);
  rd2->close();
  ASSERT_TRUE(rig.collected(2));
}

// PASS: beats flow with no reads at all; a queued read takes the next.
TEST(stage_command, tap_pass_mode_flows_and_reads_the_next_beat)
{
  Rig rig("pass");
  ASSERT_TRUE(rig.launch());
  auto p1 = open_stage_command(*rig.in, "push", FlexData::make_object(),
                               {u8_buffer_("data", {1}, {1})});
  ASSERT_TRUE(rig.collected(1));
  auto rd = open_stage_command(*rig.tap, "read", FlexData::make_object(), {});
  EXPECT_TRUE(rd->wait(100) == CommandState::Pending);   // no beat yet
  open_stage_command(*rig.in, "push", FlexData::make_object(),
                     {u8_buffer_("data", {2}, {1})});
  ASSERT_TRUE(rd->wait(3000) == CommandState::Replied);
  EXPECT_TRUE(static_cast<uint8_t*>(rd->outputs()[0].data)[0] == 2);
  rd->close();
  ASSERT_TRUE(rig.collected(2));
}

// A stop ends every wait: the tap is suspended HOLDING a replied read
// and another read is queued behind it. stop() returns rather than
// waiting on a caller that is not coming back; the queued read is
// cancelled, and the answered one stays answered -- its buffer still
// readable -- because an answer, once given, is final.
TEST(stage_command, stop_cancels_held_and_queued_commands)
{
  Rig rig("gate");
  ASSERT_TRUE(rig.launch());
  open_stage_command(*rig.in, "push", FlexData::make_object(),
                     {u8_buffer_("data", {5}, {1})});
  auto held = open_stage_command(*rig.tap, "read", FlexData::make_object(),
                                 {});
  ASSERT_TRUE(held->wait(3000) == CommandState::Replied);
  auto queued = open_stage_command(*rig.tap, "read", FlexData::make_object(),
                                   {});
  // Keep a view across the stop: its memory must stay valid.
  DataBuffer view = held->outputs()[0];

  const auto t0 = chrono::steady_clock::now();
  rig.rt->stop();
  const auto ms = chrono::duration_cast<chrono::milliseconds>(
      chrono::steady_clock::now() - t0).count();
  EXPECT_TRUE(ms < 2000);
  EXPECT_TRUE(held->state() == CommandState::Replied);
  EXPECT_TRUE(held->outputs().size() == 1);
  EXPECT_TRUE(queued->state() == CommandState::Cancelled);
  EXPECT_TRUE(queued->error().find("stopping") != string::npos);
  EXPECT_TRUE(static_cast<uint8_t*>(view.data)[0] == 5);

  // And after the stop the stage is not running: refused, not queued.
  auto after = open_stage_command(*rig.tap, "read", FlexData::make_object(),
                                  {});
  EXPECT_TRUE(after->state() == CommandState::Failed);
  EXPECT_TRUE(after->error().find("not running") != string::npos);
}

// A caller that closes before the stage takes its command costs the
// stage nothing: the command is skipped, the next one served.
TEST(stage_command, a_command_closed_while_queued_is_skipped)
{
  Rig rig("gate");
  ASSERT_TRUE(rig.launch());
  auto gone = open_stage_command(*rig.tap, "read", FlexData::make_object(),
                                 {});
  gone->close();
  auto rd = open_stage_command(*rig.tap, "read", FlexData::make_object(), {});
  open_stage_command(*rig.in, "push", FlexData::make_object(),
                     {u8_buffer_("data", {3}, {1})});
  ASSERT_TRUE(rd->wait(3000) == CommandState::Replied);
  EXPECT_TRUE(gone->state() == CommandState::Closed);
  EXPECT_TRUE(result_uint_(rd, "seq") == 1);
  rd->close();
}

TEST(stage_command, describe_commands_lists_the_declaration)
{
  Session sess;
  Pipeline pl("p", &sess);
  Stage* tap = pl.insert_stage("external-tap", "t", {{nullptr, 0}});
  ASSERT_TRUE(tap != nullptr);
  const FlexData d = describe_commands(tap->spec());
  ASSERT_TRUE(d.is_array() && d.as_array().size() == 1);
  const FlexData rd = d.as_array()[0];
  auto o = rd.as_object();
  EXPECT_TRUE(string(o.at("name").as_string("")) == "read");
  EXPECT_TRUE(o.at("holds").as_bool(false));
  const FlexData out = o.at("out");
  ASSERT_TRUE(out.as_array().size() == 1);
  EXPECT_TRUE(out.as_array()[0].as_object().at("writable").as_bool(false));
  // Not launched: refused with the reason.
  auto c = open_stage_command(*tap, "read", FlexData::make_object(), {});
  EXPECT_TRUE(c->state() == CommandState::Failed);
  EXPECT_TRUE(c->error().find("not running") != string::npos);
}

// A stage object outlives the launch it ran in (Stage::reset_run_state):
// a second launch gets a fresh inbox, commands numbered from 1 again,
// and beats counted from 1 again -- and nothing left over from run 1.
TEST(stage_command, a_relaunch_starts_over)
{
  Rig rig("");
  for (int run = 0; run < 2; ++run) {
    // A runtime launches once; the host builds a new one per launch over
    // the same stages, and so does this.
    if (run > 0) {
      rig.rt = make_unique<PipelineRuntime>(rig.pl.get(), &rig.sess);
    }
    ASSERT_TRUE(rig.launch());
    auto p = open_stage_command(*rig.in, "push", FlexData::make_object(),
                                {u8_buffer_("data", {1}, {1})});
    EXPECT_TRUE(p->wait(3000) == CommandState::Replied);
    EXPECT_TRUE(p->id() == 1);
    EXPECT_TRUE(result_uint_(p, "seq") == 1);
    rig.rt->stop();
  }
}


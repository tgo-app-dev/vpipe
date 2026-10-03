// tensor-list: one beat from each wired input -> ONE list, in port order.
//
// Order is the whole contract: the multi-reference consumers pair item k
// of a picture list with item k of the latent list made from it, and the
// conditioner labels its pictures by position. So these tests read each
// item back -- shape AND bytes -- rather than counting items.

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
#include "stages/tensor-list-stage.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace vpipe;

namespace {

// Emits a fixed list of beats, then ends.
class BeatSource : public TypedStage<BeatSource> {
public:
  static constexpr const char* kTypeName = "ut-tl-source";
  using TypedStage::TypedStage;

  std::vector<TensorBeat> beats;
  std::size_t next = 0;

  Job
  process(RuntimeContext& ctx) override
  {
    if (next >= beats.size()) { ctx.signal_done(); co_return; }
    TensorBeat tb = beats[next++];
    co_await ctx.write(0, make_payload<TensorBeatPayload>(std::move(tb)));
    co_return;
  }
};

class ListSink : public TypedStage<ListSink> {
public:
  static constexpr const char* kTypeName = "ut-tl-sink";
  using TypedStage::TypedStage;

  // Per list: each item's shape and leading byte.
  std::vector<std::vector<std::vector<std::int64_t>>> shapes;
  std::vector<std::vector<std::uint8_t>> fills;
  std::vector<FlexData> sidebands;

  Job
  process(RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) { ctx.signal_done(); co_return; }
    if (const auto* lp = dynamic_cast<const TensorListPayload*>(in.get())) {
      std::vector<std::vector<std::int64_t>> s;
      std::vector<std::uint8_t> f;
      for (const auto& it : lp->items) {
        s.push_back(it.shape);
        f.push_back(it.byte_size() > 0 ? it.bytes_()[0] : 0);
      }
      shapes.push_back(std::move(s));
      fills.push_back(std::move(f));
      sidebands.push_back(lp->sideband);
    }
    co_return;
  }
};

// A planar u8 RGB "picture" whose every byte is `fill`.
TensorBeat
picture_(int h, int w, std::uint8_t fill)
{
  TensorBeat tb;
  tb.dtype = TensorBeat::DType::U8;
  tb.shape = {3, h, w};
  tb.data.resize((std::size_t)3 * h * w);
  std::memset(tb.data.data(), fill, tb.data.size());
  return tb;
}

struct Run {
  Session sess;
  std::unique_ptr<Pipeline> pl;
  ListSink* sink = nullptr;
  TensorListStage* list = nullptr;
};

// One source per entry of `inputs`; an EMPTY entry leaves that iport of
// the list stage unwired. Returns false when the launch was refused.
bool
drive_(Run& r, std::vector<std::vector<TensorBeat>> inputs,
       FlexData cfg = FlexData::make_object())
{
  r.pl = std::make_unique<Pipeline>("p", &r.sess);
  std::vector<InEdge> edges;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].empty()) {
      edges.push_back(InEdge{nullptr, 0});
      continue;
    }
    auto src_u = std::make_unique<BeatSource>(
        &r.sess, "src" + std::to_string(i), std::vector<InEdge>{},
        FlexData::make_object());
    src_u->beats = std::move(inputs[i]);
    auto* src =
        static_cast<BeatSource*>(r.pl->insert_stage(std::move(src_u)));
    src->allocate_oports(1);
    edges.push_back(InEdge{src, 0});
  }
  auto tl_u = std::make_unique<TensorListStage>(&r.sess, "list",
                                                std::move(edges),
                                                std::move(cfg));
  r.list = static_cast<TensorListStage*>(r.pl->insert_stage(std::move(tl_u)));
  auto sink_u = std::make_unique<ListSink>(&r.sess, "sink",
                                           std::vector<InEdge>{{r.list, 0}},
                                           FlexData::make_object());
  r.sink = static_cast<ListSink*>(r.pl->insert_stage(std::move(sink_u)));
  PipelineRuntime rt(r.pl.get(), &r.sess);
  if (!rt.launch()) { return false; }
  rt.wait_idle();
  rt.stop();
  return true;
}

}  // namespace

// Three pictures of three sizes -> one list, item k from iport k, each
// at its own size and with its own bytes (moved, not mixed).
TEST(tensor_list, one_beat_per_input_in_port_order)
{
  Run r;
  ASSERT_TRUE(drive_(r, {{picture_(4, 6, 11)},
                         {picture_(8, 2, 22)},
                         {picture_(3, 3, 33)}}));
  ASSERT_TRUE(r.sink->shapes.size() == 1);
  const auto& s = r.sink->shapes[0];
  ASSERT_TRUE(s.size() == 3);
  EXPECT_TRUE(s[0] == (std::vector<std::int64_t>{3, 4, 6}));
  EXPECT_TRUE(s[1] == (std::vector<std::int64_t>{3, 8, 2}));
  EXPECT_TRUE(s[2] == (std::vector<std::int64_t>{3, 3, 3}));
  EXPECT_TRUE(r.sink->fills[0] == (std::vector<std::uint8_t>{11, 22, 33}));
  EXPECT_TRUE(r.list->lists_emitted() == 1);
}

// Every declared port takes an item: a full list, in order.
TEST(tensor_list, every_port_fills_a_full_list)
{
  Run r;
  std::vector<std::vector<TensorBeat>> inputs;
  std::vector<std::uint8_t> want;
  for (unsigned i = 0; i < TensorListStage::kMaxItems; ++i) {
    inputs.push_back({picture_(2, 2, (std::uint8_t)(i + 1))});
    want.push_back((std::uint8_t)(i + 1));
  }
  ASSERT_TRUE(drive_(r, std::move(inputs)));
  ASSERT_TRUE(r.sink->fills.size() == 1);
  EXPECT_TRUE(r.sink->fills[0] == want);
}

// An unwired slot is skipped, not an empty item.
TEST(tensor_list, an_unwired_port_is_skipped)
{
  Run r;
  ASSERT_TRUE(drive_(r, {{picture_(2, 2, 1)}, {}, {picture_(2, 2, 3)}}));
  ASSERT_TRUE(r.sink->fills.size() == 1);
  EXPECT_TRUE(r.sink->fills[0] == (std::vector<std::uint8_t>{1, 3}));
}

// Each round is one list; a round an input cannot complete is dropped
// whole, never emitted short.
TEST(tensor_list, rounds_are_lists_and_a_short_round_is_dropped)
{
  Run r;
  ASSERT_TRUE(drive_(r, {{picture_(2, 2, 1), picture_(2, 2, 2),
                          picture_(2, 2, 5)},
                         {picture_(2, 2, 3), picture_(2, 2, 4)}}));
  ASSERT_TRUE(r.sink->fills.size() == 2);
  EXPECT_TRUE(r.sink->fills[0] == (std::vector<std::uint8_t>{1, 3}));
  EXPECT_TRUE(r.sink->fills[1] == (std::vector<std::uint8_t>{2, 4}));
}

// The list's own sideband comes from config.
TEST(tensor_list, the_list_sideband_is_the_config_one)
{
  Run r;
  FlexData cfg = FlexData::make_object();
  FlexData sb = FlexData::make_object();
  sb.as_object().insert_or_assign("role", FlexData::make_string("refs"));
  cfg.as_object().insert_or_assign("sideband", sb);
  ASSERT_TRUE(drive_(r, {{picture_(2, 2, 1)}}, cfg));
  ASSERT_TRUE(r.sink->sidebands.size() == 1);
  FlexData got = r.sink->sidebands[0];
  ASSERT_TRUE(got.is_object());
  EXPECT_TRUE(got.as_object().at("role").as_string("") == "refs");
}

// Wiring past the declared ports is refused, not run unchecked.
TEST(tensor_list, more_items_than_ports_is_a_config_error)
{
  Session sess;
  Pipeline pl("p", &sess);
  auto src_u = std::make_unique<BeatSource>(&sess, "src",
                                            std::vector<InEdge>{},
                                            FlexData::make_object());
  auto* src = static_cast<BeatSource*>(pl.insert_stage(std::move(src_u)));
  src->allocate_oports(1);
  std::vector<InEdge> edges(TensorListStage::kMaxItems + 1, InEdge{src, 0});
  TensorListStage tl(&sess, "list", std::move(edges),
                     FlexData::make_object());
  EXPECT_TRUE(!tl.config_error().empty());
}

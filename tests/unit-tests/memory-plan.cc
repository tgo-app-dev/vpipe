// The plan derived from the TOPOLOGY rather than from phase names.
//
// What these pin is the arithmetic, over synthetic stages -- a real
// graph brings a checkpoint and a GPU with it, and neither is needed to
// ask whether a buffer alive across three stages is counted at all three
// and once each.

#include "minitest.h"
#include "common/job.h"
#include "common/session.h"
#include "common/vertex.h"
#include "pipeline/memory-plan.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/stage.h"
#include "pipeline/typed-stage.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace vpipe;

namespace {

// A vertex with an id. The driver loop never runs here -- these tests
// exercise the arithmetic, and a real graph would bring a checkpoint and
// a GPU with it to ask a question that needs neither.
class MpStage : public vpipe::TypedStage<MpStage> {
public:
  static constexpr const char* kTypeName = "ut-mp-stage";
  using TypedStage::TypedStage;
  vpipe::Job process(vpipe::RuntimeContext& ctx) override
  {
    ctx.signal_done();
    co_return;
  }
};
VPIPE_REGISTER_STAGE(MpStage)

constexpr std::size_t kMB = 1ull << 20;

}  // namespace

// The shape the whole thing exists for: a chain where each stage holds
// something it lets go of, and a buffer that outlives its producer.
//
// Held separately the peak is the largest single stage; held wrongly --
// summed -- it is their total, which sizes a moment that never happens.
TEST(memory_plan, a_chain_peaks_at_its_widest_moment_not_its_total)
{
  Session sess;
  auto enc = std::make_unique<MpStage>(&sess, "enc", std::vector<vpipe::InEdge>{}); auto dit = std::make_unique<MpStage>(&sess, "dit", std::vector<vpipe::InEdge>{}); auto vae = std::make_unique<MpStage>(&sess, "vae", std::vector<vpipe::InEdge>{});
  std::vector<Stage*> stages{enc.get(), dit.get(), vae.get()};
  std::vector<LogicalEdge> edges{
      {enc.get(), 0, dit.get(), 0},
      {dit.get(), 0, vae.get(), 0},
  };

  std::vector<StageMemory> mem(3);
  mem[0].hold("s0", 48 * kMB, 4 * kMB, /*releases=*/true);
  mem[0].outputs = {1 * kMB};                  // conditioning -> dit
  mem[1].hold("s1", 63 * kMB, 3 * kMB, /*releases=*/true);
  mem[1].outputs = {2 * kMB};                  // latent -> vae
  mem[2].hold("s2", 5 * kMB, 0, /*releases=*/true);  // no smaller form

  const MemoryPlan p = compute_memory_plan(stages, edges, mem);
  EXPECT_TRUE(p.ordered);
  EXPECT_TRUE(p.steps.size() == 3);

  // enc: its own floor + the conditioning it just produced.
  EXPECT_TRUE(p.steps[0].at_floor == 5 * kMB);
  // dit: its floor + the conditioning still alive + the latent.
  EXPECT_TRUE(p.steps[1].at_floor == 6 * kMB);
  // vae: itself + the latent it is reading. The conditioning is GONE --
  // nothing downstream of dit consumes it.
  EXPECT_TRUE(p.steps[2].at_floor == 7 * kMB);
  EXPECT_TRUE(p.peak_floor == 7 * kMB);
  EXPECT_TRUE(p.tightest_floor == "vae");

  // Held resident, the peak is the widest single MOMENT and not the
  // total either: every stage here lets go, so the largest is the DiT
  // beside the conditioning it is still reading and the latent it is
  // writing. Summing all three (116 MB) would size a moment that never
  // happens -- which is the whole reason the plan walks positions.
  EXPECT_TRUE(p.peak_preload == 63 * kMB + 1 * kMB + 2 * kMB);
  EXPECT_TRUE(p.tightest_preload == "dit");
}

// A stage that does NOT let go is alive at every position, which is what
// makes a continuously running graph come out right without saying
// anything special about it.
TEST(memory_plan, a_stage_that_keeps_its_weights_is_counted_throughout)
{
  Session sess;
  auto a = std::make_unique<MpStage>(&sess, "a", std::vector<vpipe::InEdge>{}); auto b = std::make_unique<MpStage>(&sess, "b", std::vector<vpipe::InEdge>{});
  std::vector<Stage*> stages{a.get(), b.get()};
  std::vector<LogicalEdge> edges{{a.get(), 0, b.get(), 0}};

  std::vector<StageMemory> mem(2);
  mem[0].hold("s0", 10 * kMB);   // releases defaults to FALSE
  mem[1].hold("s1", 4 * kMB, 0, /*releases=*/true);

  const MemoryPlan p = compute_memory_plan(stages, edges, mem);
  EXPECT_TRUE(p.steps[0].at_floor == 10 * kMB);
  EXPECT_TRUE(p.steps[1].at_floor == 14 * kMB);
  EXPECT_TRUE(p.peak_floor == 14 * kMB);
}

// A fanout is the same bytes read twice, not twice the bytes -- and it
// stays alive until the LAST reader has run.
TEST(memory_plan, a_fanout_is_one_buffer_alive_until_its_last_reader)
{
  Session sess;
  auto src = std::make_unique<MpStage>(&sess, "src", std::vector<vpipe::InEdge>{}); auto m1 = std::make_unique<MpStage>(&sess, "m1", std::vector<vpipe::InEdge>{}); auto m2 = std::make_unique<MpStage>(&sess, "m2", std::vector<vpipe::InEdge>{});
  std::vector<Stage*> stages{src.get(), m1.get(), m2.get()};
  std::vector<LogicalEdge> edges{
      {src.get(), 0, m1.get(), 0},
      {src.get(), 0, m2.get(), 0},        // same oport, second consumer
      {m1.get(), 0, m2.get(), 1},         // forces m2 after m1
  };

  std::vector<StageMemory> mem(3);
  mem[0].outputs = {8 * kMB};
  const MemoryPlan p = compute_memory_plan(stages, edges, mem);
  // Counted ONCE per position, and still alive at m2 -- charging a graph
  // per edge would bill it for splitting a wire.
  EXPECT_TRUE(p.steps[0].at_floor == 8 * kMB);
  EXPECT_TRUE(p.steps[1].at_floor == 8 * kMB);
  EXPECT_TRUE(p.steps[2].at_floor == 8 * kMB);
  EXPECT_TRUE(p.peak_floor == 8 * kMB);
}

// Two of a kind are MAXED, not summed -- the case a fixed phase
// vocabulary cannot express, because both stages would name the same
// phase and be added together.
TEST(memory_plan, two_stages_of_one_kind_do_not_sum)
{
  Session sess;
  auto d1 = std::make_unique<MpStage>(&sess, "dit1", std::vector<vpipe::InEdge>{}); auto d2 = std::make_unique<MpStage>(&sess, "dit2", std::vector<vpipe::InEdge>{});
  std::vector<Stage*> stages{d1.get(), d2.get()};
  std::vector<LogicalEdge> edges{{d1.get(), 0, d2.get(), 0}};

  std::vector<StageMemory> mem(2);
  mem[0].hold("s0", 60 * kMB, 0, /*releases=*/true);
  mem[1].hold("s1", 60 * kMB, 0, /*releases=*/true);

  const MemoryPlan p = compute_memory_plan(stages, edges, mem);
  EXPECT_TRUE(p.peak_preload == 60 * kMB);
}

// An output nothing consumes still exists. save-video writes a file from
// a clip no other stage reads, and dropping it would under-count the
// moment it is largest.
TEST(memory_plan, an_unconsumed_output_is_still_counted)
{
  Session sess;
  auto a = std::make_unique<MpStage>(&sess, "a", std::vector<vpipe::InEdge>{});
  std::vector<Stage*> stages{a.get()};
  std::vector<StageMemory> mem(1);
  mem[0].outputs = {9 * kMB};
  const MemoryPlan p = compute_memory_plan(stages, {}, mem);
  EXPECT_TRUE(p.peak_floor == 9 * kMB);
}

// A cycle cannot be ordered, and the plan says so rather than reporting
// a confident number derived from an order it did not have.
TEST(memory_plan, a_cycle_is_reported_not_guessed_at)
{
  Session sess;
  auto a = std::make_unique<MpStage>(&sess, "a", std::vector<vpipe::InEdge>{}); auto b = std::make_unique<MpStage>(&sess, "b", std::vector<vpipe::InEdge>{});
  std::vector<Stage*> stages{a.get(), b.get()};
  std::vector<LogicalEdge> edges{{a.get(), 0, b.get(), 0},
                                 {b.get(), 0, a.get(), 0}};
  std::vector<StageMemory> mem(2);
  mem[0].hold("s0", kMB);
  mem[1].hold("s1", kMB);
  const MemoryPlan p = compute_memory_plan(stages, edges, mem);
  EXPECT_TRUE(!p.ordered);
  EXPECT_TRUE(p.steps.size() == 2);
}

// The same graph must produce the same plan twice: the number is meant
// to decide whether a run is allowed, and one that moves with
// unordered_map iteration cannot.
TEST(memory_plan, the_order_is_deterministic)
{
  Session sess;
  auto a = std::make_unique<MpStage>(&sess, "a", std::vector<vpipe::InEdge>{}); auto b = std::make_unique<MpStage>(&sess, "b", std::vector<vpipe::InEdge>{}); auto c = std::make_unique<MpStage>(&sess, "c", std::vector<vpipe::InEdge>{}); auto d = std::make_unique<MpStage>(&sess, "d", std::vector<vpipe::InEdge>{});
  // b and c are both ready at once; the tie must break the same way.
  std::vector<Stage*> stages{a.get(), b.get(), c.get(), d.get()};
  std::vector<LogicalEdge> edges{
      {a.get(), 0, b.get(), 0}, {a.get(), 0, c.get(), 0},
      {b.get(), 0, d.get(), 0}, {c.get(), 0, d.get(), 1},
  };
  std::vector<StageMemory> mem(4);
  for (std::size_t i = 0; i < 4; ++i) {
    mem[i].hold("s" + std::to_string(i), (i + 1) * kMB);
  }

  const MemoryPlan p1 = compute_memory_plan(stages, edges, mem);
  for (int i = 0; i < 8; ++i) {
    const MemoryPlan pn = compute_memory_plan(stages, edges, mem);
    EXPECT_TRUE(pn.peak_floor == p1.peak_floor);
    for (std::size_t k = 0; k < p1.steps.size(); ++k) {
      EXPECT_TRUE(pn.steps[k].stage_id == p1.steps[k].stage_id);
    }
  }
}

namespace {

// Records what it is told, so a test can check the channel rather than
// its effect.
class RecordingSink final : public MemoryPlanSink {
public:
  void revise(const Stage* s, const StageMemory& m) override
  {
    ++calls;
    last_stage = s;
    last = m;
  }
  int calls = 0;
  const Stage* last_stage = nullptr;
  StageMemory last;
};

}  // namespace

// The revise channel: a stage corrects what it declared, and the
// correction reaches whoever is holding the plan.
TEST(memory_plan, a_stage_revises_into_the_sink_it_was_given)
{
  Session sess;
  auto a = std::make_unique<MpStage>(&sess, "a",
                                     std::vector<vpipe::InEdge>{});
  RecordingSink sink;

  // With NO sink -- every stage built outside a launch, which is every
  // unit test -- a revision is dropped rather than crashing.
  StageMemory m;
  m.scratch = 7 * kMB;
  a->revise_memory(m);
  EXPECT_TRUE(sink.calls == 0);

  StageLifecycleAccess::set_memory_sink(a.get(), &sink);
  a->revise_memory(m);
  EXPECT_TRUE(sink.calls == 1);
  EXPECT_TRUE(sink.last_stage == a.get());
  EXPECT_TRUE(sink.last.scratch == 7 * kMB);

  // And taken back at stop, because a stage OUTLIVES the launch it ran
  // in -- a sink left behind is a dangling pointer the next beat writes
  // through.
  StageLifecycleAccess::set_memory_sink(a.get(), nullptr);
  a->revise_memory(m);
  EXPECT_TRUE(sink.calls == 1);
}

// A revision replaces the stage's whole entry, and the peak is
// recomputed against its PEERS -- which is the reason the sink is an
// interface rather than a field on the stage. An arena nobody could
// size before the first beat is the case it exists for.
TEST(memory_plan, a_revision_moves_the_peak_it_is_part_of)
{
  Session sess;
  auto a = std::make_unique<MpStage>(&sess, "a",
                                     std::vector<vpipe::InEdge>{});
  auto b = std::make_unique<MpStage>(&sess, "b",
                                     std::vector<vpipe::InEdge>{});
  std::vector<Stage*> stages{a.get(), b.get()};
  std::vector<LogicalEdge> edges{{a.get(), 0, b.get(), 0}};

  std::vector<StageMemory> mem(2);
  mem[0].hold("s0", 10 * kMB, 0, /*releases=*/true);
  mem[1].hold("s1", 2 * kMB, 0, /*releases=*/true);
  EXPECT_TRUE(compute_memory_plan(stages, edges, mem).peak_floor
              == 10 * kMB);

  // b learns its arena at the first beat: 40 MB nothing could have
  // known from configuration.
  mem[1].scratch = 40 * kMB;
  const MemoryPlan after = compute_memory_plan(stages, edges, mem);
  EXPECT_TRUE(after.peak_floor == 42 * kMB);
  EXPECT_TRUE(after.tightest_floor == "b");
}

// TWO STAGES, ONE CHECKPOINT. The case a plan that knows only bytes gets
// wrong: two generate-image stages over one model are two stages and one
// set of weights, and billing per stage charges the graph for a model it
// loaded once.
TEST(memory_plan, one_checkpoint_held_by_two_stages_is_counted_once)
{
  Session sess;
  auto a = std::make_unique<MpStage>(&sess, "a",
                                     std::vector<vpipe::InEdge>{});
  auto b = std::make_unique<MpStage>(&sess, "b",
                                     std::vector<vpipe::InEdge>{});
  std::vector<Stage*> stages{a.get(), b.get()};
  std::vector<LogicalEdge> edges{{a.get(), 0, b.get(), 0}};

  std::vector<StageMemory> mem(2);
  mem[0].hold("/models/shared", 20 * kMB);
  mem[1].hold("/models/shared", 20 * kMB);
  EXPECT_TRUE(compute_memory_plan(stages, edges, mem).peak_floor
              == 20 * kMB);

  // Differently informed about the same thing: the larger wins, because
  // they are describing one checkpoint and disagree only by estimate.
  mem[1].holdings[0].preload = 26 * kMB;
  EXPECT_TRUE(compute_memory_plan(stages, edges, mem).peak_floor
              == 26 * kMB);

  // UNNAMED holdings are never merged -- a stage that cannot say what it
  // holds is over-counted rather than wrongly shared with a neighbour.
  std::vector<StageMemory> anon(2);
  anon[0].hold("", 20 * kMB);
  anon[1].hold("", 20 * kMB);
  EXPECT_TRUE(compute_memory_plan(stages, edges, anon).peak_floor
              == 40 * kMB);
}

// A checkpoint one stage gives back and another keeps is KEPT: the merge
// takes the union of the lifetimes, not whichever it saw first.
TEST(memory_plan, a_shared_checkpoint_takes_the_wider_lifetime)
{
  Session sess;
  auto a = std::make_unique<MpStage>(&sess, "a",
                                     std::vector<vpipe::InEdge>{});
  auto b = std::make_unique<MpStage>(&sess, "b",
                                     std::vector<vpipe::InEdge>{});
  auto c = std::make_unique<MpStage>(&sess, "c",
                                     std::vector<vpipe::InEdge>{});
  std::vector<Stage*> stages{a.get(), b.get(), c.get()};
  std::vector<LogicalEdge> edges{{a.get(), 0, b.get(), 0},
                                 {b.get(), 0, c.get(), 0}};

  std::vector<StageMemory> mem(3);
  mem[0].hold("/m", 8 * kMB, 0, /*releases=*/true);   // gives it back
  mem[2].hold("/m", 8 * kMB, 0, /*releases=*/false);  // keeps it
  const MemoryPlan p = compute_memory_plan(stages, edges, mem);
  // Alive at every position, once.
  EXPECT_TRUE(p.steps[0].at_floor == 8 * kMB);
  EXPECT_TRUE(p.steps[1].at_floor == 8 * kMB);
  EXPECT_TRUE(p.steps[2].at_floor == 8 * kMB);
  EXPECT_TRUE(p.peak_floor == 8 * kMB);
}

// RECLAIMABLE is the `auto` unload policy in the plan's terms: it costs
// nothing on a box with no room and its full size on one with room. That
// is a different statement from `releases`, which is a lifetime ending.
TEST(memory_plan, a_reclaimable_holding_has_no_floor)
{
  Session sess;
  auto a = std::make_unique<MpStage>(&sess, "a",
                                     std::vector<vpipe::InEdge>{});
  std::vector<Stage*> stages{a.get()};
  std::vector<StageMemory> mem(1);
  mem[0].hold("/dit", 30 * kMB, /*floor=*/0, /*releases=*/false,
              /*reclaimable=*/true);
  const MemoryPlan p = compute_memory_plan(stages, {}, mem);
  EXPECT_TRUE(p.peak_floor == 0);
  EXPECT_TRUE(p.peak_preload == 30 * kMB);

  // Without it, the same holding has no smaller form and its floor is
  // its size -- which is what every non-reclaimable holding says.
  std::vector<StageMemory> kept(1);
  kept[0].hold("/dit", 30 * kMB);
  EXPECT_TRUE(compute_memory_plan(stages, {}, kept).peak_floor == 30 * kMB);
}

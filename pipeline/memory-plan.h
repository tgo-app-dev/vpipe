#ifndef PIPELINE_MEMORY_PLAN_H
#define PIPELINE_MEMORY_PLAN_H

// memory-plan.h -- what the graph must hold, derived from the graph.
//
// The phase model this sits beside asks every stage to name the phase it
// belongs to, from a fixed vocabulary, and takes the maximum across
// those names. That works for the graph shape the vocabulary was written
// for and degrades quietly everywhere else: two DiTs both name
// `denoise` and are SUMMED, a graph that is not
// conditioner-DiT-VAE has no phase to name so everything is counted as
// persistent, and a decode that feeds another denoise cannot say so
// because the running order is a constant list.
//
// The deeper problem is that a phase is ASSERTED by a stage about a
// global property it cannot see. A stage does not know whether it is the
// only DiT or the second of three, and `weight_claims_in_phase`
// documents its own preconditions as ones the planner cannot check.
// note_phase_released exists because of that: an audit, one launch late,
// for a promise nothing can verify up front.
//
// So this derives the same answer from the topology instead. The runtime
// already builds the post-inlining logical edge list for clock-domain
// analysis; ordering the stages by it gives every claim a position, and
// a buffer's lifetime falls out of who consumes it rather than being
// named. Two DiTs land at different positions and are maxed. A graph
// nobody anticipated works, because nothing had to anticipate it.
//
// WHAT A STAGE STILL HAS TO SAY, because topology cannot know it:
// whether it lets go. A position says what cannot run before what; it
// says nothing about a stage that keeps its weights for the whole run so
// the next launch is free. That is a policy, and `releases` is where the
// stage states it.
//
// POSITION IS NOT TIME, and the model does not pretend otherwise. Stages
// in a pipeline are coroutines that overlap; a topological position is a
// data dependency, not a schedule. What makes the arithmetic right
// anyway is that overlap is expressed through `releases`: a stage that
// runs the whole time does not release, so it is counted at every
// position. A generation graph is a chain and its positions read as
// time; a continuous capture graph has everything live at once and every
// stage says so.

#include "pipeline/clock-domain.h"

#include <cstddef>
#include <string>
#include <vector>

namespace vpipe {

class Stage;

// ONE thing a stage holds, and what it is.
//
// `source` is what makes the arithmetic right when a graph holds a
// checkpoint twice. Two generate-image stages over one model are two
// stages and ONE set of weights; counting the bytes per stage bills the
// graph for a model it loaded once. The plan cannot know that from the
// bytes alone -- they are just numbers -- so the stage names what it
// holds and the plan merges by that name.
//
// Empty means "unique to this stage", never merged. That is the right
// answer for anything a stage allocates for itself, and the safe one
// when a stage cannot identify what it holds: over-counting is the
// direction that refuses a run, not the one that thrashes.
struct StageHolding {
  std::string source;

  // What it costs when the box has room for it.
  std::size_t preload = 0;

  // The least it can be held at while still being HELD -- a
  // block-streaming DiT's trunk plus its slots. Zero means it has no
  // smaller form, and is read as `preload`.
  std::size_t floor = 0;

  // Gone when this stage is finished, rather than held for the run.
  bool releases = false;

  // Can be given back ENTIRELY under memory pressure, and reloaded (or
  // found still there) when next used. Its floor is therefore zero
  // however large it is: on a tight box it costs nothing, on a roomy one
  // it costs `preload` and saves the reload.
  //
  // This is the `auto` unload policy, said in the plan's own terms. It
  // is distinct from `releases`, which is about a LIFETIME that ends;
  // this is about bytes that can be reclaimed at any moment and are
  // still wanted.
  bool reclaimable = false;

  // What this holding costs on a box with no room to spare.
  std::size_t floor_bytes() const
  {
    if (reclaimable) { return 0; }
    return floor > 0 ? floor : preload;
  }
};

// What one stage costs the box, in its own terms.
//
// Two numbers per holding because the plan, not the stage, should decide
// which applies: whether a component CAN hold less is a property of the
// loader, whether it SHOULD is a property of the box, and a stage asked
// to answer both must guess at the second before it can see its peers.
// That guess is the ordering problem the planning phase exists to
// remove.
struct StageMemory {
  // Everything durable this stage holds. Several, because a stage can
  // hold more than one checkpoint with different lifetimes -- a DiT it
  // may give back beside an encoder it keeps -- and collapsing them to
  // one number loses exactly the distinction that matters.
  std::vector<StageHolding> holdings;

  // What the stage ALLOCATES while running, and frees after. Live only
  // at its own position, and never shared, so it needs no source.
  std::size_t scratch = 0;

  // Bytes handed downstream, indexed by oport. Alive from this stage
  // until the LAST consumer of that port has run -- which the plan works
  // out from the edges, so a stage never has to know who reads it.
  //
  // Shorter than the stage's oport count is fine; the rest are zero.
  std::vector<std::size_t> outputs;

  // The common case: one checkpoint, named.
  void hold(std::string source, std::size_t preload, std::size_t floor = 0,
            bool releases = false, bool reclaimable = false)
  {
    if (preload == 0) { return; }
    StageHolding h;
    h.source = std::move(source);
    h.preload = preload;
    h.floor = floor;
    h.releases = releases;
    h.reclaimable = reclaimable;
    holdings.push_back(std::move(h));
  }
};

// One position in the running order, for reporting.
struct MemoryPlanStep {
  std::string stage_id;
  std::size_t at_floor   = 0;   // everything streamable at its floor
  std::size_t at_preload = 0;   // everything resident
};

struct MemoryPlan {
  // The peak of each column. `floor` is what the box must hold for the
  // graph to run at all; `preload` is what it costs to keep everything
  // resident so a second launch pays no reload.
  std::size_t peak_floor   = 0;
  std::size_t peak_preload = 0;
  // The position each peak occurs at, for a message that can name it.
  std::string tightest_floor;
  std::string tightest_preload;
  std::vector<MemoryPlanStep> steps;
  // False when the graph could not be ordered (a cycle). The plan is
  // then computed in declaration order, which is still an upper bound on
  // nothing in particular -- callers should report rather than refuse.
  bool ordered = true;
};

// Where a stage's revision goes once the run has started.
//
// The plan is a SNAPSHOT taken before anything loads, from numbers a
// stage can know from its configuration. Several of the largest terms
// are not knowable then -- an activation arena is sized by the first
// beat's geometry, and a streaming model does not know what it will keep
// until it has kept it -- so the snapshot has to be correctable, or its
// unknowns are permanent.
//
// Deliberately an interface rather than a callback on the stage: what a
// revision has to reach is the OTHER stages' numbers, since the peak is
// a property of the graph. A stage that could only correct its own entry
// would be writing to something nothing reads.
class MemoryPlanSink {
public:
  virtual ~MemoryPlanSink() = default;

  // `stage` now expects to cost `m`. Called from the stage's own thread
  // while other stages are running, so implementations must be
  // thread-safe.
  virtual void revise(const Stage* stage, const StageMemory& m) = 0;
};

// Derive the plan. `mem[i]` describes `stages[i]`.
//
// `edges` is the post-inlining logical topology -- the same list
// clock-domain analysis takes, and for the same reason: using raw
// oport_edges() would be wrong once call-stages have been erased.
MemoryPlan compute_memory_plan(const std::vector<Stage*>&       stages,
                               const std::vector<LogicalEdge>&  edges,
                               const std::vector<StageMemory>&  mem);

}  // namespace vpipe

#endif

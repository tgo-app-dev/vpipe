// resource-plan.h -- the graph-wide resource planning phase.
//
// Some stages have to acquire something scarce and process-wide before
// they can run: model weights, a CoreML tower, a scratch arena. Those
// acquisitions interact -- a checkpoint two stages both name is loaded
// once, and a stage deciding how much memory it may use is really asking
// what its peers are about to take. So the decision cannot be made
// stage-locally at initialize() time, where every driver runs
// concurrently and each stage sees whichever subset of its peers happens
// to have gotten there first.
//
// The fix is a phase, not a lock: PipelineRuntime collects every stage's
// claims and hands them to the planners BEFORE the first driver starts.
// By the time any initialize() runs, the whole graph's intent is on the
// record, whoever acts on it.
//
// The runtime deliberately does not understand claims. It knows the
// ordering rule -- all planning completes before any initialize() begins
// -- and nothing about weights, checkpoints or bytes; `kind` selects a
// planner and `key` means whatever that planner says it means. That is
// what keeps a second resource kind from becoming a second edit to
// pipeline-runtime.cc.

#ifndef RESOURCE_PLAN_H
#define RESOURCE_PLAN_H

#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace vpipe {

class SessionContextIntf;

// One thing a stage intends to acquire, named in the vocabulary of
// whichever planner handles `kind`. Both fields are opaque to the
// pipeline core.
struct ResourceClaim {
  std::string kind;   // planner selector, e.g. model_memory::kWeightsKind
  std::string key;    // planner-defined, e.g. a checkpoint directory

  // Optional LIFETIME, in the planner's vocabulary. Two claims carrying
  // DIFFERENT non-empty phases assert that they are never held at the
  // same time, so a planner may take their maximum where it would
  // otherwise take their sum. Empty -- the default -- means "held for
  // the whole run", which is always the safe answer.
  //
  // The runtime does not interpret this any more than it interprets
  // `key`. What it is worth, and what the names mean, is entirely up to
  // the planner for `kind`; see model_memory::kPhaseCondition and the
  // ordering constraint documented with it.
  std::string phase;
};

// Consumes the claims of ONE kind across a launch.
//
// The bracket matters as much as the claims. begin_plan() runs on every
// registered planner at the start of every launch, whether or not this
// graph claims anything of that kind -- it is where a planner drops the
// PREVIOUS launch's state, which is launch-scoped knowledge only the
// runtime has. end_plan() is where a planner reports or acts on the
// complete picture.
//
// Planners are process-wide singletons and outlive any one pipeline, so
// they take the session per call rather than holding one, and must
// tolerate two pipelines launching at once. (Concurrent launches racing
// over one manager's declarations is a pre-existing hazard, unchanged
// here: keep per-launch state out of the planner, or make it atomic.)
class ResourcePlanner {
public:
  virtual ~ResourcePlanner() = default;

  // The claim kind this planner consumes. Must be stable and unique;
  // the registry keeps the first planner registered for a kind.
  virtual std::string_view kind() const noexcept = 0;

  virtual void begin_plan(const SessionContextIntf* /*session*/) {}

  // One claim of this planner's kind. Called once per claiming stage,
  // in graph order, between begin_plan and end_plan. `phase` is the
  // claim's declared lifetime, empty for the usual "whole run"; what it
  // means is this planner's business (see ResourceClaim::phase).
  virtual void claim(const SessionContextIntf* session,
                     const std::string&        key,
                     const std::string&        phase) = 0;

  // A refinement of an already-claimed key, from Stage::decide_resources,
  // delivered after EVERY stage has claimed. A planner that lets these
  // change what it reports must buffer them until end_plan: applying one
  // as it arrives would make it visible to the next stage's decision,
  // which is the order-dependence the two-pass split exists to remove.
  virtual void decide(const SessionContextIntf* /*session*/,
                      const std::string&        /*key*/,
                      const std::string&        /*phase*/) {}

  virtual void end_plan(const SessionContextIntf* /*session*/) {}
};

// Process-wide planner set, populated at static-init time by
// VPIPE_REGISTER_RESOURCE_PLANNER. Same singleton discipline as
// StageRegistry: a plugin library MUST link the host libvpipe shared so
// it observes this instance rather than forking a second one.
class ResourcePlannerRegistry {
public:
  static ResourcePlannerRegistry& get() noexcept;

  // Takes ownership. A second planner for an already-registered kind is
  // ignored (first registration sticks), so a duplicate-include footgun
  // cannot crash static init.
  void add(std::unique_ptr<ResourcePlanner> p);

  // Null when no planner claims `kind`. The runtime treats that as a
  // stage asking for something nothing in this build can provide, and
  // says so -- an unplanned claim is a silent return to the race the
  // phase exists to prevent, so it must never pass quietly.
  ResourcePlanner* find(std::string_view kind) const noexcept;

  std::vector<ResourcePlanner*> all() const;

private:
  ResourcePlannerRegistry() = default;

  std::vector<std::unique_ptr<ResourcePlanner>> _planners;
  mutable std::mutex                            _mu;
};

// Register a planner at static-init time. Namespace scope, in the .cc
// that implements the planner -- next to the policy it applies, so the
// two cannot be separated by someone editing one of them.
//
// Do NOT remove this under "looks unused": the planner is reached only
// by kind lookup, so without it nothing ever constructs the type and
// every claim of that kind goes unplanned.
#define VPIPE_REGISTER_RESOURCE_PLANNER(T)                              \
  namespace { const int _vpipe_planner_reg_##T = [] {                   \
    ::vpipe::ResourcePlannerRegistry::get().add(                        \
      std::make_unique<T>());                                           \
    return 0; }(); }

}

#endif

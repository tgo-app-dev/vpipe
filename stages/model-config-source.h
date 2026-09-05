#ifndef VPIPE_STAGES_MODEL_CONFIG_SOURCE_H
#define VPIPE_STAGES_MODEL_CONFIG_SOURCE_H

#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace vpipe {

// The contract every MODEL-SPECIFIC config source honours, and that
// `generate-video` reads. One header rather than a comment in each stage
// because a family added later -- in this tree or in a plugin -- has to
// match it exactly to be usable, and there is nothing to match against
// if the rules only exist in two stages that happen to agree.
//
// WHY THESE STAGES EXIST. A generative stage that serves several model
// families accumulates the UNION of their knobs: `generate-video` carried
// Wan's two guidance scales and expert boundary beside MiniMax-H3's two
// sigma shifts, condition timesteps and audio duration, with every key
// inert on the family that was not resident. That surface grows with each
// family, gives no hint which keys apply to the checkpoint in hand, and
// silently ignores the ones that do not -- so a knob set in good faith
// does nothing and says nothing. Moving each family's keys into a stage
// of its own makes the applicable set a WIRING decision: what is
// connected is what applies, and the graph shows which family it is for.
//
// THE BEAT. A `FlexDataPayload` object on an oport tagged `kConfigTag`,
// carrying `kFamilyKey` plus that family's own keys. The consumer matches
// the family against the resident checkpoint's and warns on a mismatch
// rather than applying keys the model has no use for -- a `wan2` config
// wired to an H3 checkpoint is a graph mistake, not a no-op.
//
// PARSING IS THE MODEL LAYER'S. The consumer passes the object down
// unread; each family's own `GenerationParams::from_flex` turns it into
// typed fields (MetalWanTransformer, MetalMiniMaxH3Transformer). So a new
// knob is added in the family that owns it and in the stage that emits
// it, and the consuming stage does not change at all.
//
// TRIGGER. Every config source takes an OPTIONAL trigger iport, matching
// `text-prompt`: unwired it emits one beat for the run and closes; wired
// it emits one beat per inbound beat (any payload -- receipt is the
// signal) and ends on upstream EOS. That is what lets the knobs change
// per request in a graph that generates continuously, rather than being
// fixed for the life of the pipeline.
namespace model_config {

// The oport tag a config source publishes and every `model_config`
// iport accepts. Also what the web-ui composer uses to offer only the
// sources a consumer can take.
//
// Deliberately NOT per-consumer ("generate-video-config"): one
// checkpoint's parameters are read by several stages of its graph -- the
// conditioner grounds a reference image the way that family's reference
// pipeline does, the DiT samples the way it samples -- and those are the
// same family's facts. One source per FAMILY, fanned out to whichever
// stages of the graph want it, beats one per (family x consumer), which
// would put the same number in two places and let them disagree.
inline constexpr std::string_view kConfigTag = "model-config";

// Which family the object's keys belong to. The vocabulary is the DiT
// family tag the consuming stage resolves from the checkpoint -- "wan",
// "minimax-h3" -- NOT the emitting stage's type name, so that a family
// spanning several checkpoint generations needs one config stage and not
// one per generation.
inline constexpr std::string_view kFamilyKey = "model_family";

// An empty config object already stamped with its family. Every source
// starts here and inserts its own keys.
inline FlexData
make_config(std::string_view family)
{
  FlexData fd = FlexData::make_object();
  fd.as_object().insert_or_assign(std::string(kFamilyKey),
                                  FlexData::make_string(std::string(family)));
  return fd;
}

// Copy whichever of the grounded-encode override keys `cfg` actually
// sets into `out`. An UNSET key is deliberately not written: the
// conditioner starts from genai::GroundedEncodeParams::for_family(),
// which holds each family's own reference-pipeline numbers, and a
// default emitted as if it were a choice would replace them with
// something that merely looks configured. Shared by every image-aware
// family's source so the four spellings cannot drift apart.
inline void
copy_grounded_keys(const FlexData& cfg, FlexData& out)
{
  static constexpr const char* kKeys[] = {
    "vl_long_edge", "vl_pixel_budget", "vl_min_pixels", "vl_max_pixels"};
  if (!cfg.is_object() || !out.is_object()) { return; }
  auto in = cfg.as_object();
  auto o  = out.as_object();
  for (const char* k : kKeys) {
    if (in.contains(k)) { o.insert_or_assign(k, in.at(k)); }
  }
}

// The family an incoming config beat is for, or "" when it is not an
// object or does not say. A beat that does not say is applied anyway --
// it can only have come from a source the graph wired deliberately --
// so this is for reporting, not gating.
inline std::string
family_of(const FlexData& fd)
{
  if (!fd.is_object()) { return {}; }
  auto o = fd.as_object();
  const std::string key(kFamilyKey);
  if (!o.contains(key)) { return {}; }
  return std::string(o.at(key).as_string(""));
}

}  // namespace model_config

// The trigger half of the contract, once, for every config source.
//
// A Derived source supplies only what is ITS OWN: a `kTypeName`, a
// `kSpec` with its family's attrs, a constructor that reads them, and
//
//   FlexData resolved_config() const;
//
// returning the beat (built on model_config::make_config). Everything
// below -- the one-shot / per-trigger decision, the per-launch reset,
// closing the oport -- is identical across families and was duplicated
// twice before there were seven of these. A plugin adding a family
// inherits it too, so a new source cannot get the trigger rule subtly
// wrong while looking right.
//
// Derived must still call allocate_oports() in its constructor and
// register itself; those are per-stage facts, not shared behaviour.
template <class Derived>
class ModelConfigSourceStage : public TypedStage<Derived> {
public:
  using TypedStage<Derived>::TypedStage;

  void
  reset_run_state() override
  {
    // Per-launch reset, as every source here does: stopping a pipeline
    // destroys the RUNTIME, not the stages, so without this a
    // Stop-then-Start would find the source already exhausted and its
    // consumer would wait on a beat that never comes.
    _done = false;
  }

  // The beat is a pure function of THIS STAGE'S CONFIG, so a consumer
  // can have it before any driver starts.
  //
  // Unconditional, and that is not an oversight about the trigger: a
  // trigger controls WHEN and HOW MANY TIMES resolved_config() is
  // emitted, never WHAT -- it is a const method over members only the
  // constructor writes. So the first beat and every later one are the
  // same value, which is exactly the promise apply_constant is given.
  //
  // What this buys: a consumer that latches something load-time from
  // this beat -- MiniMax-H3's `linear_branch`, a second 4.28 GB
  // checkpoint -- can declare it. declare_resources() runs before any
  // driver, so a checkpoint first seen at run time is one every peer
  // sized the box without. See Stage::constant_output.
  std::optional<FlexData>
  constant_output(unsigned oport) const override
  {
    if (oport != 0) { return std::nullopt; }
    return static_cast<const Derived*>(this)->resolved_config();
  }

  Job
  process(RuntimeContext& ctx) override
  {
    // `iport_connected` and not just `num_iports`: an unwired port is
    // spelled as a placeholder edge ({"src": ""}) in a saved pipeline,
    // so counting ports would read a one-shot source as trigger-driven
    // and it would block forever on a beat nothing sends.
    const bool has_trigger = ctx.num_iports() >= 1 && ctx.iport_connected(0);
    if (has_trigger) {
      auto t = co_await ctx.read(0);
      if (!t) { ctx.signal_done(); co_return; }
    } else if (_done) {
      ctx.signal_done();
      co_return;
    }
    _done = true;
    auto* self = static_cast<Derived*>(this);
    FlexData fd = self->resolved_config();
    self->report_config(fd);
    co_await ctx.write(0, make_payload<FlexDataPayload>(std::move(fd)));
    if (!has_trigger) { ctx.signal_done(); }
  }

  // What to log when a beat goes out. The default names the family and
  // leaves the numbers to the object; a source with a reading worth
  // spelling out (Wan's "boundary from the checkpoint") overrides it.
  void
  report_config(const FlexData& fd) const
  {
    const auto* self = static_cast<const Derived*>(this);
    if (self->session() == nullptr) { return; }
    self->session()->info(fmt("{}('{}'): {} model config -> {}",
                              Derived::kTypeName, self->id(),
                              model_config::family_of(fd), fd.to_json()));
  }

protected:
  bool _done = false;   // no-trigger one-shot guard
};

}  // namespace vpipe

#endif

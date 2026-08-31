#ifndef VPIPE_STAGES_AUDIO_VIDEO_TEMPORAL_SLICE_STAGE_H
#define VPIPE_STAGES_AUDIO_VIDEO_TEMPORAL_SLICE_STAGE_H

#include "common/beat-payload-intf.h"
#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vpipe {

// Python's `seq[start:end:step]`, with the BEAT STREAM as the sequence.
//
// The counterpart of temporal-stack, and its opposite: that one turns
// many time slices into one tensor, this one keeps some of the slices
// and drops the rest. Beats pass through UNTOUCHED -- same tensor, same
// rank, same sideband -- so the stage changes how many beats arrive and
// nothing about any of them. It is not temporal-decimation either: that
// one drops frames by CONTENT (motion, a rate ceiling), this one by
// INDEX, and an index slice is what a graph needs when it wants a
// particular frame rather than fewer frames.
//
// The motivating use is one line of config: `start: -1` on a
// MiniMax-H3 clip, wired into save-image, writes the LAST frame -- the
// keyframe that seeds the next FL2VA run.
//
// NEGATIVE INDICES ARE WHY THIS IS NOT A FILTER. `-1` means "the last
// beat", and which beat that is cannot be known until the source hits
// EOS. So a negative start or end makes the stage HOLD beats: it keeps
// a rolling window of the most recent
//
//   W = max(start < 0 ? -start : 0, end < 0 ? -end : 0)
//
// and decides those at EOS, when the count is finally known. W is the
// exact bound, not an approximation -- a beat with W beats behind it can
// already be decided, because it is provably outside any window a
// negative index can name. So `start: -1` holds ONE beat however long
// the clip is, and a wholly non-negative slice holds none and streams.
//
// The price of a negative index is therefore visible and self-imposed:
// it is whatever the author wrote. `start: -600` on a live source really
// does hold 600 beats, which is why the stage says at launch how many it
// will keep.
//
// ---- SLICING INSIDE ONE BEAT (`sequence: frames`) ----
//
// The default sequence is the beat STREAM, above. `sequence: frames`
// makes it the LEADING AXIS OF ONE BEAT instead: a stacked clip
// [T, 3, H, W] in, the selected frames out, ONE BEAT IN AND ONE BEAT
// OUT. It is the shape `vae-decode`'s clip oport and `temporal-stack`
// both produce.
//
// Two things follow, and both are the point:
//
//   * NO CLOCK-DOMAIN CROSSING. One beat in, one beat out, so the
//     oport reports the iport's clock group and the analyzer unifies
//     them. That is what lets this stage sit inside a FEEDBACK loop --
//     a feedback pair must stay within one clock domain, and the
//     stream mode, which drops beats, cannot.
//   * NO HOLDING. `T` is known from the beat, so `start: -1` resolves
//     immediately instead of waiting for EOS. Negative indices cost
//     nothing here.
//
// SQUEEZE IS EXPLICIT, AND THAT IS DELIBERATE. Selecting one frame can
// mean either of two things, and they are different requests to a
// model that conditions on references: [3, H, W] is a STILL, while
// [1, 3, H, W] is a one-frame CLIP. MiniMax-H3's reference encoder
// reads the two by RANK and sizes them by different rules -- a still
// gets its own short edge with no area cap, a clip gets the target
// canvas -- so reading one as the other encodes it at a fraction of
// the detail, from a tensor that is correctly shaped either way.
//
// There is no safe default for that, so `squeeze` is a switch rather
// than an inference: false (the default) keeps the time axis, true
// drops it when -- and only when -- the slice selected exactly one
// frame. Asking to squeeze several is warned about, once, and the axis
// is kept: the alternative is fabricating a shape the caller cannot
// have meant.
class TemporalSliceStage final : public TypedStage<TemporalSliceStage> {
public:
  static constexpr const char* kTypeName = "temporal-slice";

  TemporalSliceStage(const SessionContextIntf* s, std::string id,
                     std::vector<InEdge> iports, FlexData config);
  ~TemporalSliceStage() override;

  Job initialize(RuntimeContext& ctx) override;
  Job process(RuntimeContext& ctx) override;
  const StageSpec& spec() const noexcept override;

  // `sequence: frames` is one beat in, one beat out, so it does NOT
  // change the beat rate and the oport belongs in the iport's clock
  // group. The stream mode drops beats and keeps its own group. The
  // analyzer reads this rather than the static spec, which is what
  // makes the answer config-dependent at all.
  unsigned
  oport_clock_group(unsigned p) const noexcept override
  {
    if (p == 0 && _within_beat) { return 0; }
    return 1;
  }

  // Test-only. `peak_hold` is the most beats the window ever held at
  // once -- the stage's whole memory cost, and the thing a negative
  // index is charged for.
  std::uint64_t emitted() const noexcept { return _emitted; }
  std::uint64_t seen() const noexcept { return _index; }
  std::int64_t peak_hold() const noexcept { return _peak_hold; }

private:
  // How many trailing beats a negative index forces this stage to hold.
  // 0 for a wholly non-negative slice, which then streams.
  std::int64_t hold_() const noexcept;

  // Is beat `j` selected, once the total `n` is known? Python's rule,
  // exactly: clamp both ends into [0, n], then step from the start.
  bool selected_(std::int64_t j, std::int64_t n) const noexcept;

  // Is beat `j` selected, decided WITHOUT the total -- valid only for a
  // beat that has already fallen out of the hold window, which is what
  // makes the missing ends unable to change the answer.
  bool selected_streaming_(std::int64_t j) const noexcept;

  // Slice the leading axis of ONE beat instead of the beat stream.
  bool _within_beat = false;
  // Drop the time axis when the slice selected exactly one frame. Only
  // meaningful with `_within_beat`; see SQUEEZE above.
  bool _squeeze = false;
  // Latched so a mismatched squeeze is said once, not once per clip.
  bool _squeeze_warned = false;

  // One beat in, one beat out: slice the leading axis. Warns and
  // forwards nothing when the beat is not a tensor it can index.
  Job process_within_(RuntimeContext& ctx);

  std::int64_t _start = 0;
  std::int64_t _step  = 1;
  std::int64_t _end   = 0;
  bool _has_start = false;   // absent => from the first beat
  bool _has_end   = false;   // absent => until EOS

  // Per-RUN state. Stages outlive a stop/relaunch while the runtime does
  // not, so a window left full by the previous run would emit its beats
  // into this one's stream, at indices from a count that no longer
  // applies.
  std::int64_t  _index     = 0;   // index of the NEXT incoming beat
  std::uint64_t _emitted   = 0;
  std::int64_t  _peak_hold = 0;
  std::deque<std::pair<std::int64_t, std::unique_ptr<BeatPayloadIntf>>>
      _hold;
};

}  // namespace vpipe

#endif

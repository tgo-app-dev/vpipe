#ifndef VPIPE_STAGES_AUDIO_VIDEO_TEMPORAL_STACK_STAGE_H
#define VPIPE_STAGES_AUDIO_VIDEO_TEMPORAL_STACK_STAGE_H

#include "apple-silicon/tensor-beat.h"
#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <deque>
#include <string>
#include <utility>
#include <vector>

namespace vpipe {

// Many beats in, ONE beat out: the adapter between this tree's
// per-frame / per-chunk convention and a consumer that wants a whole
// clip or a whole waveform at once.
//
// Every producer here emits time one slice at a time -- video-to-rgb a
// planar [3, H, W] per frame, audio-to-pcm a fixed-duration [N] chunk --
// because that is what a pipeline streams. A model that conditions on a
// REFERENCE wants the other shape: MiniMax-H3's ref2va ports take one
// [frames, 3, H, W] clip or one [N] soundtrack per reference. Nothing
// bridged the two, so those ports had no producer at all.
//
// THE AXIS IS THE WHOLE DESIGN, and it differs by modality:
//
//   * VIDEO stacks on a NEW leading axis. [3, H, W] x T -> [T, 3, H, W]:
//     the frames stay whole and gain a time index.
//   * AUDIO concatenates on the LAST axis. [N] x k -> [sum N], and
//     planar [2, N] x k -> [2, sum N]. A soundtrack is one continuous
//     waveform; giving it a chunk axis would describe the arrival
//     schedule rather than the sound.
//
// Getting that backwards does not fail loudly, which is why it is not a
// single "stack" rule with a configurable axis and a sensible default:
// concatenating planar stereo on axis 0 yields [2+2, N], two channels
// glued end to end wearing the right rank, and stacking video on the
// last axis yields [3, H, W*T], a very wide single frame. Both are
// valid tensors and neither is the media.
//
// `auto` reads the modality off the beat -- rank 3 u8 is a frame, rank
// <= 2 f32 (or anything carrying a `sample_rate`) is audio -- and LATCHES
// it for the group, so an inconsistent producer is an error rather than
// a half-stacked tensor.
//
// The sideband is REWRITTEN, not forwarded. A group's rates are a
// property of the group: audio keeps `sample_rate` and recomputes
// `duration_us`; video must publish an `fps` that no frame beat carries
// (video-to-rgb emits `fps_num`/`fps_den`, and only when the source
// states them), so this stage resolves one -- from that pair, else from
// the span of `timestamp_us`, else from config -- and says which route
// it took. A clip conditioned at the wrong speed is the failure the
// consumers of this stage are arranged around; it must not be guessed.
class TemporalStackStage final : public TypedStage<TemporalStackStage> {
public:
  static constexpr const char* kTypeName = "temporal-stack";

  TemporalStackStage(const SessionContextIntf* s, std::string id,
                     std::vector<InEdge> iports, FlexData config);
  ~TemporalStackStage() override;

  Job process(RuntimeContext& ctx) override;
  Job initialize(RuntimeContext& ctx) override;
  const StageSpec& spec() const noexcept override;

  // What the group is being read as. `kAuto` never survives the first
  // beat -- it resolves to one of the others and latches.
  enum class Mode { kAuto, kVideo, kAudio, kGeneric };

  // Which modality a beat looks like, from its rank, dtype and
  // sideband. Public and static so the sensing has a test that needs no
  // runtime: it is the part that fails silently when it is wrong.
  static Mode sense(const TensorBeat& tb);

  // Test-only.
  std::uint64_t groups_emitted() const noexcept { return _emitted; }

private:
  // Reset the accumulator to empty, keeping the configured mode.
  void reset_group_();
  // How many beats to keep after an emit: `_overlap`, bounded by what
  // the group actually held and never the whole of it. 0 for audio and
  // whenever no overlap is configured.
  int retain_beats_() const noexcept;
  // Drop all but the last `keep` beats from the accumulator, rebuilding
  // the group's rate metadata from the beats that remain.
  void retain_tail_(int keep);
  // Append one beat. False (with a warned reason) when it does not match
  // the group already open.
  bool append_(const TensorBeat& tb, std::string* err);
  // Build the stacked beat from the accumulator. Empty shape when the
  // group is empty.
  TensorBeat build_() const;
  Job emit_(RuntimeContext& ctx);

  // How many beats of the group just emitted stay in the accumulator, so
  // the NEXT group opens with them. Frames, not seconds: this stage's
  // unit is the beat and `group_size` already counts them, so at 24 fps
  // an overlap of 24 is one second and the arithmetic is the caller's
  // rather than a rate this stage would have to re-derive at retention
  // time. 0 (the default) is the old behaviour -- groups share nothing.
  //
  // VIDEO and generic groups only. Audio beats are variable-length on
  // the time axis, so "the last N beats" is not a duration and the
  // retention would depend on how the source happened to chunk; the
  // seconds-accurate control for audio is `audio-to-pcm`'s
  // `chunk_overlap_s`, which works in samples and owns the chunking.
  int    _overlap  = 0;
  bool   _overlap_audio_warned = false;

  // The last `_overlap` beats' rate metadata, kept only when an overlap
  // is configured. The group's `timestamp_us` and its derived fps come
  // from the FIRST beat in the accumulator, so after a retention they
  // have to come from the first RETAINED beat -- carrying the emitted
  // group's would date the next clip to a frame it no longer starts on.
  struct TailMeta {
    bool          has_ts = false;
    std::uint64_t ts     = 0;
    FlexData      sb;
  };
  std::deque<TailMeta> _tail_meta;

  Mode   _mode_cfg = Mode::kAuto;   // as configured
  Mode   _mode     = Mode::kAuto;   // latched for the open group
  int    _group_size = 0;
  std::size_t _max_bytes = 0;
  double _fps_cfg  = 0.0;
  // Merged into every emitted sideband, last, so it wins.
  FlexData _sideband_cfg;

  // The open group: raw bytes back to back, plus the per-beat geometry
  // the first beat fixed.
  std::vector<std::uint8_t> _buf;
  std::vector<std::int64_t> _elem_shape;   // one beat's shape
  TensorBeat::DType         _dtype = TensorBeat::DType::F32;
  int                       _count = 0;    // beats in the group
  std::int64_t              _tail  = 0;    // audio: summed last axis

  // Sideband carried from the FIRST beat of the group, plus what the
  // rate resolution found.
  FlexData      _first_sideband;
  bool          _has_first_ts = false;
  std::uint64_t _first_ts_us  = 0;
  std::uint64_t _last_ts_us   = 0;
  int           _sample_rate  = 0;
  double        _fps_num      = 0.0;
  double        _fps_den      = 0.0;
  bool          _fps_reported = false;

  std::uint64_t _emitted = 0;
  bool          _capped  = false;
};

}  // namespace vpipe

#endif

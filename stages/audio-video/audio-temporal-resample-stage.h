#ifndef VPIPE_STAGES_AUDIO_VIDEO_AUDIO_TEMPORAL_RESAMPLE_STAGE_H
#define VPIPE_STAGES_AUDIO_VIDEO_AUDIO_TEMPORAL_RESAMPLE_STAGE_H

#include "apple-silicon/tensor-beat.h"
#include "common/av-filter-graph.h"
#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vpipe {

// `temporal-resample` for sound: PCM in at one rate and speed, PCM out
// at another, through a libavfilter chain.
//
// A waveform has TWO temporal knobs where a clip has one, and they are
// independent:
//
//   * `output_sample_rate` is the RESOLUTION -- how finely the same
//     seconds are sampled. It changes no duration and no pitch. It is
//     the knob a model forces: MiniMax-H3's audio VAE reads 32000 and
//     nothing else, Qwen3-ASR 16000.
//   * `speed` is the DURATION -- the same sound over more or fewer
//     seconds.
//
// Changing the duration raises the question the third knob answers.
// Play a waveform faster by handing the same samples out at a higher
// rate and the pitch rises with it; that is what tape does, and it is
// not always what is wanted. `pitch` picks:
//
//   maintain  the pitch is held while the duration changes (`atempo`,
//             a WSOLA time-stretch). The default, because it is the one
//             that leaves a voice recognisable.
//   follow    the pitch tracks the speed, chipmunk or drawl. The tape
//             behaviour, and free: it is a resample and nothing else.
//   raise     shift UP by `pitch_semitones`, independently of speed.
//   lower     shift DOWN by `pitch_semitones`, independently of speed.
//
// The last two are `asetrate` + `aresample` + `atempo`: relabel the
// rate (which moves pitch AND duration together), convert back, then
// stretch the duration back to where it was. That is the standard
// FFmpeg idiom and it is exactly why the three knobs compose -- one
// pitch factor and one tempo factor, with `speed` folded into the
// tempo.
//
// CLOCK DOMAINS. Same rule and the same reason as `temporal-resample`:
// a speed change means the stage does not emit one beat per beat it
// consumes, so its oport gets a clock of its own. A STACKED input --
// the whole waveform in one beat, what `temporal-stack` builds -- is
// 1:1, so its oport shares the iport's clock and crosses nothing. It is
// STATED (`stacked`), not sensed, because the clock analysis runs at
// launch, before any beat exists.
//
// Note that even a PURE rate change is not 1:1 while streaming: the
// filter owns its own block boundaries, so `aresample` hands back
// frames of its own length rather than one per chunk it was given.
// MEASURED: twelve 0.25 s chunks in, thirteen beats out. So the
// streaming oport is always on a clock of its own, and there is no
// setting that quietly makes it not be.
//
//   iport0  pcm   TensorBeatPayload f32, [N] mono or PLANAR [C, N].
//                 `sample_rate` on the sideband is REQUIRED -- a
//                 waveform read at the wrong rate is the wrong sound,
//                 and every producer here states it.
//   oport0  pcm   the same layout at `output_sample_rate`. Streaming,
//                 the chunk LENGTHS are the filter's own and not the
//                 input's; a consumer that needs fixed chunks should
//                 accumulate (every consumer in this tree does).
class AudioTemporalResampleStage final
    : public TypedStage<AudioTemporalResampleStage> {
public:
  static constexpr const char* kTypeName = "audio-temporal-resample";

  AudioTemporalResampleStage(const SessionContextIntf* session,
                             std::string               id,
                             std::vector<InEdge>       iports,
                             FlexData                  config);
  ~AudioTemporalResampleStage() override;

  Job initialize(RuntimeContext& ctx) override;
  Job process(RuntimeContext& ctx) override;
  void reset_run_state() override;

  const StageSpec& spec() const noexcept override;
  unsigned oport_clock_group(unsigned p) const noexcept override;

  enum class Pitch { kMaintain, kFollow, kRaise, kLower };

  // The libavfilter chain. `in_rate`/`out_rate` are the two sample
  // rates, `speed` the duration factor, `k` the pitch factor (1 = keep).
  //
  // Public and static because it is the whole behaviour of the stage
  // reduced to a string, and because the composition of the three knobs
  // -- tempo = speed / pitch -- is the part that is easy to get subtly
  // wrong and impossible to hear the difference of in a unit test.
  static std::string chain_for(int in_rate, int out_rate, double speed,
                               double k);
  // 2^(+-semitones/12), or the speed itself for `follow`, or 1.
  static double pitch_factor(Pitch p, double semitones, double speed);
  static const char* pitch_name(Pitch p) noexcept;

  // Test-only.
  Pitch  pitch()   const noexcept { return _pitch; }
  double speed()   const noexcept { return _speed; }
  bool   stacked() const noexcept { return _stacked; }
  int    output_sample_rate() const noexcept { return _out_rate; }
  std::uint64_t emitted() const noexcept { return _emitted; }

private:
  bool ensure_graph_(int rate, int channels);
  // Drain the sink, appending planar [channels, n] blocks.
  bool drain_(std::vector<std::vector<float>>* blocks, int* channels);
  bool push_block_(const float* pcm, int channels, int samples,
                   std::int64_t pts);
  FlexData out_sideband_(int samples, int rate);

  int    _out_rate = 0;          // 0 = keep the input's
  double _speed    = 1.0;
  Pitch  _pitch    = Pitch::kMaintain;
  double _semitones = 0.0;
  bool   _stacked  = false;

  AvFilterGraph _graph;
  int           _grate = 0, _gch = 0;
  std::int64_t  _pts = 0;
  AVFrame*      _in_frame  = nullptr;
  AVFrame*      _out_frame = nullptr;

  FlexData      _sb_seed;
  bool          _have_seed = false;
  std::uint64_t _first_ts_us = 0;
  bool          _have_ts   = false;
  std::uint64_t _out_samples = 0;   // emitted so far, for the timestamps
  std::uint64_t _emitted   = 0;
  bool          _shape_warned = false;
};

}  // namespace vpipe

#endif

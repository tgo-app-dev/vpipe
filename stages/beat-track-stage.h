#ifndef VPIPE_STAGES_BEAT_TRACK_STAGE_H
#define VPIPE_STAGES_BEAT_TRACK_STAGE_H

#include "common/flex-data.h"
#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vpipe {

class TensorBeatPayload;

namespace beat {
struct Beat;
}

// Beat tracking: PCM in, the beats' positions out, as they are decided.
//
// The tracker is librosa.beat.beat_track (librosa 0.11) -- Ellis's
// dynamic program over the median-aggregated mel onset envelope, the
// tempo from its autocorrelation under a log-normal prior -- with its
// arithmetic (stages/beat-tracker.h). Two modes:
//
//   stream  (default) beats come out while the sound goes on. librosa's
//           whole-track statistics (the dB ceiling, the tempo, the
//           envelope's deviation) are taken over what has been heard,
//           and a beat is handed out once every path the tracker could
//           still choose runs through it, so nothing handed out is ever
//           revised. The first decision waits for 8 s of sound past the
//           first onset (the tempogram's window); after that a beat
//           trails the sound by one to three beats. A stream that ends
//           before its first decision is decided exactly as `whole`.
//   whole   librosa's answer for the whole stream, handed out at its end.
//
// librosa hears a file through librosa.load: mono (the channels' mean)
// at 22050 Hz. So does this stage by default: `channels` "mix" takes
// the mean, and a stream at another rate is converted to `sample_rate`
// through libavfilter's `aresample` (librosa's own is soxr's; the beats
// do not notice). `sample_rate` 0 tracks at the stream's own rate, as
// librosa.beat.beat_track(y, sr=its rate) does -- the envelope's frames
// are then hop_length samples of THAT rate.
//
//   iport0  pcm    f32 PCM, [N] mono or PLANAR [C, N], `sample_rate` on
//                  the sideband (required: a beat at the wrong rate is at
//                  the wrong time). Any beat length; consecutive beats are
//                  one stream until the end of stream.
//
//   oport0  beats  FlexData, one object per `group` beats (the last of a
//                  stream may hold fewer):
//                    {
//                      "beats": [..],      // in `units`: seconds (real)
//                                          //   from the stream's first
//                                          //   sample, or that stream's
//                                          //   samples / envelope frames
//                                          //   (int)
//                      "units": "time" | "samples" | "frames",
//                      "index": uint,      // beats[0]'s ordinal
//                      "tempo": real,      // bpm, as the last of them was
//                                          //   placed (whole: the track's)
//                      "sample_rate": int, // the input's
//                      "hop_length": int, "analysis_rate": int,
//                      "channel": int,     // channels "each" / an index
//                      "timestamp_us": uint // the stream's first sample's
//                                           //   UTC, when the input says
//                    }
//                  On a clock of its own: beats come out when decided.
//
// Config:
//   mode         "stream" | "whole"
//   group        beats per output object, >= 1 (default 1)
//   units        "time" (seconds, default) | "samples" | "frames"
//   channels     "mix" (the mean, librosa.load's), "each" (a tracker per
//                channel), or a channel index
//   sample_rate  the rate the tracker hears (default 22050); 0 = the
//                input's
//   hop_length   envelope hop in samples (librosa's 512); at most 1000
//                envelope frames a second
//   start_bpm    the tempo prior's centre (librosa's 120)
//   tightness    how closely beats keep the tempo (librosa's 100)
//   trim         drop weak leading and trailing beats (librosa's true)
//   bpm          > 0: this tempo, not estimated (librosa's `bpm=`)
//   max_latency  stream: the longest a beat waits to be handed out,
//                seconds (default 4; 0 = until it is certain). Past it
//                the best path so far is said, and a later decision that
//                disagrees continues from it -- nothing said is revised.
class BeatTrackStage final : public TypedStage<BeatTrackStage> {
public:
  static constexpr const char* kTypeName = "beat-track";

  BeatTrackStage(const SessionContextIntf* session,
                 std::string               id,
                 std::vector<InEdge>       iports,
                 FlexData                  config);
  ~BeatTrackStage() override;

  Job process(RuntimeContext& ctx) override;
  void reset_run_state() override;

  const StageSpec& spec() const noexcept override;

  enum class Units { kTime, kSamples, kFrames };

  // Test-only.
  bool          whole() const noexcept { return _whole; }
  int           group() const noexcept { return _group; }
  Units         units() const noexcept { return _units; }
  int           analysis_rate() const noexcept { return _sr; }
  std::uint64_t beats_out() const noexcept { return _beats_out; }
  std::uint64_t groups_out() const noexcept { return _groups_out; }

  struct Channel;

private:
  bool open_(const TensorBeatPayload& tb, int channels, int rate);
  bool feed_(Channel& c, const float* x, std::size_t n);
  bool resample_(Channel& c, const float* x, std::size_t n, bool eos);
  void collect_(Channel& c);
  FlexData group_(Channel& c, std::size_t n) const;

  // Config.
  bool        _whole = false;
  int         _group = 1;
  Units       _units = Units::kTime;
  std::string _channels;
  int         _pick = -1;          // a channel index, else -1
  bool        _each = false;
  int         _sr = 22050;         // 0 = the input's
  int         _hop = 512;
  double      _start_bpm = 120.0, _tightness = 100.0, _bpm = 0.0;
  double      _max_latency = 4.0;
  bool        _trim = true;

  // The stream.
  std::vector<std::unique_ptr<Channel>> _ch;
  bool          _open = false, _failed = false;
  int           _in_rate = 0, _in_channels = 0, _rate = 0;
  bool          _have_ts = false;
  std::uint64_t _first_ts_us = 0;
  std::uint64_t _beats_out = 0, _groups_out = 0;
  // Groups ready to write, with their beat counts.
  std::vector<std::pair<FlexData, std::size_t>> _out;
};

}  // namespace vpipe

#endif  // VPIPE_STAGES_BEAT_TRACK_STAGE_H

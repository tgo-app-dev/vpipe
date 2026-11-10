#ifndef VPIPE_STAGES_BEAT_TRACKER_H
#define VPIPE_STAGES_BEAT_TRACKER_H

// librosa.beat.beat_track (librosa 0.11), on a stream.
//
// Ellis's dynamic-programming beat tracker as librosa runs it:
//
//   1. ONSET STRENGTH (librosa.onset.onset_strength, aggregate=median):
//      a centred STFT (n_fft 2048, a periodic Hann, zeros padded both
//      sides), the power through a 128-band Slaney mel filterbank to
//      sr / 2, power_to_db (amin 1e-10, top_db 80 below the LOUDEST
//      bin of the whole spectrogram), each band's first difference
//      clipped at zero, the median over the bands, shifted 3 frames
//      (lag 1 + n_fft / (2 hop)) so a frame's onset lands on it.
//   2. TEMPO (librosa.feature.tempo): the envelope's autocorrelation in
//      8-second Hann windows centred on every frame (linear ramps to 0
//      past either end), each normalised by its peak, averaged over the
//      track; the period of most log1p(1e6 x) under a log-normal prior
//      about start_bpm (sigma one octave), tempi above 320 bpm excluded.
//   3. BEATS: the envelope over its standard deviation (ddof 1),
//      smoothed by a Gaussian a 32nd of a period wide; a dynamic
//      program -- each frame's best predecessor between half and two
//      periods back, penalised tightness * log(gap / period)^2 -- with
//      no chain allowed to start below 1% of the strongest frame; the
//      last beat at the last local maximum of the cumulative score at
//      least half the median of them; the path traced back from it; and
//      the beats in the frames before the first, and after the last,
//      frame stronger than half the RMS of the beats' smoothed scores
//      dropped (`trim`).
//
// Every step reproduces librosa's arithmetic, float32 where librosa is
// float32 (the spectrogram, the envelope and its normalisation) and
// float64 where it is float64 (the tempogram, the local and cumulative
// scores: librosa hands its float64 tempo to numba's gufuncs, which
// therefore run their float64 kernels). Mode::kWhole IS
// librosa.beat.beat_track over the whole stream: it decides nothing
// until finish().
//
// Mode::kStream decides on the way. Three of librosa's choices need the
// whole track, and a stream takes them over what it has heard instead:
// the dB ceiling (the loudest bin so far), the tempo (the windows
// complete so far) and the envelope's deviation (all of it so far).
// The dynamic program runs a beat period behind the input -- a frame's
// local score reaches one period ahead -- and it uses librosa's own
// time-varying-tempo form, each frame at the tempo held when it was
// scored. A beat is COMMITTED once every path the tracker could still
// extend runs through it: the most recent frame on which all their
// backtracks meet. From then on no later frame may take a predecessor
// that does not descend from it, so a committed beat is never revised.
// It is HANDED OUT once it is also not one of the weak leading or
// trailing beats `trim` would drop: once a frame at or after it is
// stronger than the threshold over the beats committed so far (and,
// for the first, it is not before the first such frame).
//
// Most beats are certain two to four seconds after they are heard, but
// where two phases of the beat stay plausible for a while (a quiet or
// ambiguous passage) certainty can take ten or twenty. max_latency
// bounds the wait: a beat that old is handed out on the best path so
// far, and if a later decision disagrees the stream simply continues
// from it -- nothing handed out is revised, and the tracker is not held
// to a guess. MEASURED against librosa on two whole songs (3:34, 4:45):
// beats within a frame of librosa's 98.8-100% uncertain-free, 98.1% at
// 4 s, 96.9% at 2 s, 95.5% at 1 s. Nothing is
// decided until the tempogram's window (8 seconds) has been heard past
// the first onset; a stream that ends before its first decision is
// decided whole at its end, exactly as kWhole. After that a beat comes
// out a beat period plus the backtracks' convergence behind the sound
// -- typically one to three beats. Decisions go a spectrogram frame at
// a time, so they do not depend on how the samples were cut.
//
// A stream holds a bounded window, silences included. A leading silence
// is held as nothing: its dB rows and zero envelope frames go as they
// come, and its part of the dynamic program, run at the first decision,
// turns exactly periodic and is not held (VPIPE_BEAT_TRACK_LEAD_DP=0
// holds it -- the same scores to the bit). In a silence between two
// passages the candidate paths run side by side until the sound after
// it decides; past 2^16 envelope frames (25 minutes at 43 a second) the
// best path so far is committed. Frame rates over 1000 a second are
// refused.
//
// The tracker hears mono at Params::sample_rate; converting a stream to
// it is the caller's (the stage's) job.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace vpipe::beat {

struct Params {
  double sample_rate = 22050.0;  // librosa.load's default rate
  int    hop_length  = 512;
  int    n_fft       = 2048;
  int    n_mels      = 128;
  double start_bpm   = 120.0;
  double std_bpm     = 1.0;
  double ac_size     = 8.0;      // tempogram window, seconds
  double max_tempo   = 320.0;
  float  tightness   = 100.0f;
  bool   trim        = true;
  double bpm         = 0.0;      // > 0: this tempo, not estimated
  float  top_db      = 80.0f;
  // kStream: the longest a beat waits to be handed out, seconds; 0 =
  // until it is certain. Past it the best path so far is committed (and
  // a beat held for the trailing trim let go).
  double max_latency = 0.0;
};

enum class Mode { kStream, kWhole };

struct Beat {
  std::int64_t frame = 0;   // envelope frame: samples frame * hop_length
  double       bpm   = 0.0; // the tempo the beat was placed under
};

class Tracker {
public:
  Tracker(const Params& p, Mode mode);
  ~Tracker();
  Tracker(const Tracker&)            = delete;
  Tracker& operator=(const Tracker&) = delete;

  // Empty when the parameters are usable, else why not.
  const char* invalid() const noexcept;

  // Mono samples at Params::sample_rate, any length at a time.
  void push(const float* x, std::size_t n);
  // librosa's onset_envelope= route, kWhole only: frames of an envelope
  // computed elsewhere, in place of samples. A tracker takes one or the
  // other.
  void push_envelope(const float* env, std::size_t n);
  // The end of the stream: decides everything left.
  void finish();
  bool finished() const noexcept;

  // The beats handed out since the last call, in order.
  std::vector<Beat> take();

  // The tempo in force (bpm): the estimate so far, the final one after
  // finish(); 0 before any (and for a track without an onset, as
  // librosa's).
  double tempo() const noexcept;
  // Envelope frames so far; frames per second is sample_rate / hop.
  std::int64_t frames() const noexcept;
  // Beats handed out so far.
  std::int64_t beats_out() const noexcept;

  // ---- Introspection for tests (kWhole: the whole track's) ----
  const std::vector<float>& onset_envelope() const noexcept;
  const std::vector<double>& local_score() const noexcept;
  const std::vector<double>& cumulative_score() const noexcept;
  // The tempogram bin the tempo was read from (frames per beat).
  int tempo_bin() const noexcept;
  // The envelope's deviation the local score was divided by (the last).
  float onset_std() const noexcept;
  // Beats committed (kStream; handed out or held for the trim).
  std::int64_t committed() const noexcept;
  // The latest frame decided: every beat at or before it is final.
  std::int64_t commit_frame() const noexcept;
  // The most frames any of the tracker's histories has held at once so
  // far (kStream forgets what no decision can reach again).
  std::int64_t held_frames() const noexcept;

  struct Impl;

private:
  std::unique_ptr<Impl> _m;
};

// librosa.beat.beat_track(y=y, sr=p.sample_rate, ...) with units
// 'frames': the tempo and the beats, as one call.
struct Result {
  double                    tempo = 0.0;
  std::vector<std::int64_t> beats;
};
Result beat_track(const float* y, std::size_t n, const Params& p);

// The mel filterbank librosa.filters.mel builds (Slaney, fmax sr / 2),
// [n_mels, 1 + n_fft / 2] float32, for tests.
std::vector<float> mel_filterbank(const Params& p);

}  // namespace vpipe::beat

#endif  // VPIPE_STAGES_BEAT_TRACKER_H

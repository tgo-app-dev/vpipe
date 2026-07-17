#ifndef VPIPE_APPLE_SILICON_COREML_AUDIO_SEGMENT_STAGE_H
#define VPIPE_APPLE_SILICON_COREML_AUDIO_SEGMENT_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {

class CoreMLLoadedModel;

// Audio segmentation stage. Wraps a Silero VAD model (CoreML form) and
// emits ONLY timestamp markers that delimit speech utterances. No PCM
// bytes flow through this stage -- downstream consumers (audio-
// transcribe in streaming mode) slice their own synchronous PCM stream
// using the emitted [start_us, end_us) window.
//
//   iport0  TensorBeatPayload, f32 mono PCM. Rank-1 [N] or rank-2
//           [1, N]. The first-sample UTC timestamp is read from
//           sideband.timestamp_us; the rate from sideband.sample_rate
//           with the configured `sample_rate` as fallback. Consecutive
//           beats are treated as one continuous stream and concatenated
//           into a rolling window-sized buffer.
//
//   oport0  FlexDataPayload (object), one beat per closed utterance:
//             { start_us, end_us, index, is_partial }
//           Timestamp-only marker -- no PCM flows here; the consumer
//           slices its own PCM stream using [start_us, end_us).
//
// Pipeline placement: typically a sibling consumer of `audio-to-pcm`
// (fanout), so the same PCM stream feeds `audio-segment` (for
// boundaries) and `audio-transcribe` (for slicing).
//
//   audio-capture -> audio-to-pcm (small frames) -> audio-segment ----+
//                                              \                      |
//                                               +---> audio-transcribe (iport1)
//
// Inference is window/hop based, modelled after audio-tagging:
//   * each call sees `_window_samples` of contiguous PCM
//   * the cursor advances by `_hop_samples` after each call
//   * consecutive windows overlap by `window_samples - hop_samples`
//   * the FIRST window prepends `(window_samples - hop_samples)` zero
//     samples at the head so the model has a clean carryover context;
//     the "new audio" frame of every inference is the trailing
//     `_hop_samples`.
//
// Boundaries are derived from a hysteresis FSM driven by the per-window
// VAD probability:
//
//   * SILENCE: when prob >= speech_threshold for at least
//     min_speech_ms of contiguous high-prob windows, transition to
//     SPEECH and latch the segment start at (first crossing time
//     - pre_pad_ms).
//   * SPEECH:  when prob <  silence_threshold for at least
//     min_silence_ms of contiguous low-prob windows, close the segment
//     at (first low crossing + post_pad_ms) and return to SILENCE.
//   * A speech segment that grows past max_segment_s is force-closed
//     with is_partial=true; a new segment starts at the cut point.
//   * On drain (iport EOS), any open segment is flushed with
//     is_partial=true.
//
// The CoreML feature wiring is configurable so the stage works with
// different Silero exports. Two common shapes:
//
//   "Pure 512" (Silero JIT/ONNX -> CoreML, no STFT context inside the
//   graph):
//     input  "input"  : [1, 512] f32
//     input  "h"      : [2, 1, 64] f32
//     input  "c"      : [2, 1, 64] f32
//     input  "sr"     : int64 (optional)
//     output "output" : [1] f32 prob
//     output "hn"/"cn": next states
//
//   "Unified v6" (FluidAudio export; 512 new samples + 64-sample
//   internal carryover, single combined LSTM state of width 128):
//     input  "audio_input"     : [1, 576] f32  (= 64 ctx + 512 new)
//     input  "hidden_state"    : [1, 128] f32
//     input  "cell_state"      : [1, 128] f32
//     output "vad_output"      : [1, 1, 1] f32 prob
//     output "new_hidden_state": [1, 128]
//     output "new_cell_state"  : [1, 128]
//
// Defaults below match the unified-v6 model that vpipe ships with.
// Override the names/shapes/window_samples/hop_samples for other
// exports; setting `sr_feature_name` to "" disables the int64 sample
// rate input entirely.
//
// Config (FlexData object):
//   model_path           (string, required)  -- .mlpackage / .mlmodelc dir
//   compute_units        (int,    default 2) -- 0/1/2/3 (CoreML ComputeUnit)
//   sample_rate          (int,    default 16000)
//   window_samples       (int,    default 576) -- size of each inference
//                                                 input window
//   hop_samples          (int,    default 512) -- advance per inference;
//                                                 must be <= window_samples
//                                                 (overlap = window-hop)
//   speech_threshold     (real,   default 0.5)
//   silence_threshold    (real,   default 0.35) -- must be < speech_threshold
//   min_speech_ms        (int,    default 250)
//   min_silence_ms       (int,    default 400)
//   max_segment_s        (real,   default 12.0)
//   pre_pad_ms           (int,    default 100)
//   post_pad_ms          (int,    default 200)
//   input_feature_name   (string, default "audio_input")
//   prob_feature_name    (string, default "vad_output")
//   state_h_in_name      (string, default "hidden_state")
//   state_c_in_name      (string, default "cell_state")
//   state_h_out_name     (string, default "new_hidden_state")
//   state_c_out_name     (string, default "new_cell_state")
//   sr_feature_name      (string, default "")          -- "" disables
//   state_h_shape        (array<int>, default [1,128])
//   state_c_shape        (array<int>, default [1,128])
class AudioSegmentStage final : public TypedStage<AudioSegmentStage> {
public:
  static constexpr const char* kTypeName = "audio-segment";

  AudioSegmentStage(const SessionContextIntf* session,
                    std::string               id,
                    std::vector<InEdge>       iports,
                    FlexData                  config);
  ~AudioSegmentStage() override;

  Job initialize(RuntimeContext& ctx) override;
  Job process   (RuntimeContext& ctx) override;
  Job drain     (RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only accessors.
  const std::string& model_path()       const noexcept { return _model_path; }
  int           sample_rate()           const noexcept { return _sample_rate; }
  int           window_samples()        const noexcept { return _window_samples; }
  int           hop_samples()           const noexcept { return _hop_samples; }
  double        speech_threshold()      const noexcept { return _speech_threshold; }
  double        silence_threshold()     const noexcept { return _silence_threshold; }
  int           min_speech_ms()         const noexcept { return _min_speech_ms; }
  int           min_silence_ms()        const noexcept { return _min_silence_ms; }
  double        max_segment_s()         const noexcept { return _max_segment_s; }
  std::uint64_t segments_emitted()      const noexcept { return _segments_emitted; }
  std::uint64_t frames_run()            const noexcept { return _frames_run; }

private:
  // Run one window through the model. Returns false on any failure;
  // the caller skips advancing the FSM for that window.
  bool run_window_(const float* samples, float& prob_out);

  // Drive the hysteresis FSM with one new probability sample whose
  // window (specifically its NEW-audio segment) starts at
  // `frame_start_us` and ends at `frame_end_us`. May emit on oport 0.
  Job advance_fsm_(RuntimeContext& ctx,
                   float           prob,
                   std::uint64_t   frame_start_us,
                   std::uint64_t   frame_end_us);

  Job emit_segment_(RuntimeContext& ctx,
                    std::uint64_t   start_us,
                    std::uint64_t   end_us,
                    bool            is_partial);

  void reset_lstm_state_();

  // ---- Config attributes; defaults live in kSpec.attrs (kConfigKeys) and
  // are read in the constructor via attr_*. Scalar/string declarations carry
  // no non-zero default. ----
  std::string _model_path;
  std::string _models_db;
  int         _compute_units{};
  int         _sample_rate{};
  int         _window_samples{};
  int         _hop_samples{};
  double      _speech_threshold{};
  double      _silence_threshold{};
  int         _min_speech_ms{};
  int         _min_silence_ms{};
  double      _max_segment_s{};
  int         _pre_pad_ms{};
  int         _post_pad_ms{};
  std::string _input_feature_name;
  std::string _prob_feature_name;
  std::string _state_h_in_name;
  std::string _state_c_in_name;
  std::string _state_h_out_name;
  std::string _state_c_out_name;
  std::string _sr_feature_name;        // "" disables the sr input
  // Array attrs (no flat attr_* accessor): read off config() when present,
  // else these unified-v6 defaults stand.
  std::vector<std::int64_t> _state_h_shape = { 1, 128 };
  std::vector<std::int64_t> _state_c_shape = { 1, 128 };

  // Derived from window_samples - hop_samples; samples of carryover.
  int _context_overlap_samples = 0;

  // ---- Model handle ----
  std::shared_ptr<CoreMLLoadedModel> _loaded;
  bool                               _sr_warned = false;

  // ---- Rolling PCM buffer (mono f32 at _sample_rate) ----
  // `_buf[0]` corresponds to UTC `_buf_base_ts_us` when `_have_base_ts`.
  // At start-of-stream, when _context_overlap_samples > 0, _buf is
  // prepended with that many zero samples and `_buf_base_ts_us` is set
  // to `first_real_sample_ts - overlap_us` so the "new audio" frame
  // starts at the true first sample.
  std::vector<float> _buf;
  std::uint64_t      _buf_base_ts_us = 0;
  bool               _have_base_ts   = false;
  bool               _stream_started = false;

  // ---- LSTM state (zero-initialised in initialize) ----
  std::vector<float> _state_h;
  std::vector<float> _state_c;

  // ---- FSM ----
  enum class State { Silence, Speech };
  State          _state              = State::Silence;
  // Microseconds spent in the current candidate direction since the
  // prob first crossed the threshold.
  std::uint64_t  _candidate_run_us   = 0;
  // Timestamp of the candidate run's first qualifying frame; becomes
  // the speech start (with pre_pad applied) when the run qualifies.
  std::uint64_t  _candidate_start_us = 0;
  // Current open segment's start (valid when _state == Speech).
  std::uint64_t  _segment_start_us   = 0;
  // Latest frame's end_us seen -- used as the "now" cursor for emit.
  std::uint64_t  _last_frame_end_us  = 0;
  // Bookkeeping.
  std::uint64_t  _segments_emitted   = 0;
  std::uint64_t  _frames_run         = 0;
};

}

#endif

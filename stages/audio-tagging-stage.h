#ifndef VPIPE_APPLE_SILICON_COREML_AUDIO_TAGGING_STAGE_H
#define VPIPE_APPLE_SILICON_COREML_AUDIO_TAGGING_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace CML {
class PredictionOptions;
}

namespace vpipe {

class CoreMLLoadedModel;

// Audio tagging stage: runs the CED-base AudioSet tagger (a CoreML
// MultiArray model) over a sliding window of incoming PCM and emits
// the top-scoring class tags.
//
//   iport0  TensorBeatPayload, f32 mono 16 kHz PCM. Rank-1 [N] or
//           rank-2 [1, N]. The first-sample UTC timestamp is read
//           from sideband.timestamp_us (and the rate from
//           sideband.sample_rate, falling back to the `sample_rate`
//           config). Consecutive beats are treated as one continuous
//           stream and concatenated.
//
//   oport0  FlexDataPayload, one object per analysed window:
//             {
//               "timestamp_us": uint,   // UTC of the window's first
//                                       //   sample
//               "duration_us":  uint,   // window length (5 s)
//               "sample_rate":  int,
//               "window_index": uint,
//               "tags": [ { "label": str, "index": int,
//                           "score": real }, ... ]   // top_k, desc
//             }
//
// iport0 and oport0 are NOT in one clock domain: the model consumes a
// fixed `window_seconds` (5 s = 80000 samples) but the stage advances
// by `hop_seconds` (4 s) each run, so each window overlaps the prior
// by window-hop (1 s) for steadier tagging. Many input beats feed one
// output beat; the output cadence is one per hop's worth of audio.
//
// The model itself is a TensorType (MultiArray) CoreML mlpackage
// loaded through the session-shared CoreMLModelManager (so two stages
// pointing at the same file + compute_units share one load). Output
// is 527 sigmoid probabilities (FLOAT16); the 527 AudioSet display
// names are embedded in the stage (ced-audioset-labels.h).
//
// Config (FlexData object on the 4th constructor parameter):
//   model_path           (string)  -- .mlpackage / .mlmodelc dir.
//                                      Default: the bundled CED-base
//                                      5 s fp16 export.
//   input_feature_name   (string, default "waveform")
//   output_feature_name  (string, default "probabilities")
//   compute_units        (int,    default 2)  0=CPUOnly 1=CPU+GPU
//                                             2=All 3=CPU+ANE
//   sample_rate          (int,    default 16000) -- expected input
//                                             rate; must match the
//                                             model (16 kHz).
//   window_seconds       (real,   default 5.0)  -- must match the
//                                             model's input length.
//   hop_seconds          (real,   default 4.0)  -- advance per run;
//                                             (0, window_seconds].
//   top_k                (int,    default 5)    -- tags per window.
//   score_threshold      (real,   default 0.0) -- drop tags below
//                                             this probability.
class AudioTaggingStage final : public TypedStage<AudioTaggingStage> {
public:
  static constexpr const char* kTypeName = "audio-tagging";

  AudioTaggingStage(const SessionContextIntf* session,
                    std::string               id,
                    std::vector<InEdge>       iports,
                    FlexData                  config);
  ~AudioTaggingStage() override;

  Job initialize(RuntimeContext& ctx) override;
  Job process   (RuntimeContext& ctx) override;
  Job drain     (RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only accessors.
  const std::string& model_path()     const noexcept { return _model_path; }
  int                sample_rate()    const noexcept { return _sample_rate; }
  int                window_samples() const noexcept { return _window_samples; }
  int                hop_samples()    const noexcept { return _hop_samples; }
  int                top_k()          const noexcept { return _top_k; }
  double             score_threshold() const noexcept
  { return _score_threshold; }
  std::uint64_t      windows_emitted() const noexcept
  { return _windows_emitted; }

private:
  // Run one fixed-length window through the model. `samples` points at
  // `_window_samples` contiguous f32 values. Fills `probs_out` with
  // `_n_classes` sigmoid probabilities. Returns false on any failure
  // (the beat is then skipped). Synchronous: holds the per-model
  // predict mutex and an autorelease pool only for the call duration,
  // never across a coroutine suspend.
  bool run_window_(const float*        samples,
                   std::vector<float>& probs_out);

  // Build the output FlexData object for one window's probabilities.
  FlexData build_tags_(const std::vector<float>& probs,
                       std::uint64_t             ts_us) const;

  // ---- Config attributes; defaults live in kSpec.attrs and are read
  // in the constructor via attr_*. Declarations carry no non-zero
  // default. ----
  std::string _model_path;
  std::string _input_feature_name;
  std::string _output_feature_name;
  int         _compute_units{};
  int         _sample_rate{};
  double      _window_seconds{};
  double      _hop_seconds{};
  int         _top_k{};
  double      _score_threshold{};

  // ---- Derived ----
  int _window_samples = 0;   // window_seconds * sample_rate
  int _hop_samples    = 0;   // hop_seconds    * sample_rate
  int _n_classes      = 0;   // model output element count (527)

  // ---- Model handle ----
  std::shared_ptr<CoreMLLoadedModel> _loaded;
  CML::PredictionOptions*            _opts = nullptr;

  // ---- Sliding-window buffer ----
  // Pending mono samples not yet consumed by a window. `_buf[0]`
  // corresponds to UTC `_buf_base_ts_us` when `_have_base_ts`.
  std::vector<float> _buf;
  std::uint64_t      _buf_base_ts_us = 0;
  bool               _have_base_ts   = false;
  bool               _sr_warned      = false;

  // Bookkeeping / logging.
  std::uint64_t _windows_emitted = 0;
};

}

#endif

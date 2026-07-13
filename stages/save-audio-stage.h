#ifndef VPIPE_STAGES_SAVE_AUDIO_STAGE_H
#define VPIPE_STAGES_SAVE_AUDIO_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <string>

namespace vpipe {

class TensorBeatPayload;

// Save-audio sink: the audio counterpart to save-image, and the natural
// downstream of `text-to-speech` (which emits PCM).
//
//   iport0  TensorBeatPayload, f32 PCM. Accepts rank-1 [n_samples] (mono)
//           or rank-2 [channels, n_samples]. Sample rate is read from
//           `sideband.sample_rate` when present, else from the
//           `sample_rate` config, else 24000.
//
//   oport0  (OPTIONAL, unconditional) FlexDataPayload object describing
//           the file just written: keys `path` (string), `format`
//           (string), `samples` (int), `sample_rate` (int), `duration_s`
//           (real). Downstream consumers are optional (the runtime drops
//           writes when no cursor is attached).
//
// Per beat = one complete clip = one output file (encode -> flush ->
// close). This is NOT a rolling/segmented writer. When multiple beats
// arrive, an incrementing index is appended before the extension to
// avoid overwriting (out.aac, out-001.aac, out-002.aac, ...).
//
// WAV ("wav") is written directly with a 44-byte PCM header + int16
// samples and needs no ffmpeg. AAC / MP3 / M4A go through ffmpeg
// (encoder + swresample + a single-file muxer), mirroring the
// rtsp-capture audio transcode path. When a compressed format is
// requested but ffmpeg is unavailable, the stage warns and falls back
// to WAV (rewriting the extension to .wav).
//
// Config (FlexData object on the 4th constructor parameter; deferred-
// validated -- the ctor never throws, problems are recorded via
// fail_config()):
//   output_path  (string, required)  -- output file path. When it has no
//                                        extension, one is appended from
//                                        `format`.
//   format       (string)            -- "wav" | "aac" | "mp3" | "m4a"
//                                        (m4a => AAC in an mp4 container).
//                                        Default: inferred from the
//                                        output_path extension, else
//                                        "wav".
//   bitrate      (int, default 128000)-- AAC / MP3 target bitrate.
//   sample_rate  (int, default 0)    -- 0 = use the payload sideband; if
//                                        neither present, 24000.
class SaveAudioStage final : public TypedStage<SaveAudioStage> {
public:
  static constexpr const char* kTypeName = "save-audio";

  SaveAudioStage(const SessionContextIntf* session,
                 std::string               id,
                 std::vector<InEdge>       iports,
                 FlexData                  config);
  ~SaveAudioStage() override;

  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only accessors.
  const std::string& output_path() const noexcept { return _output_path; }
  const std::string& format()      const noexcept { return _format; }
  int           bitrate()          const noexcept { return _bitrate; }
  int           sample_rate_cfg()  const noexcept { return _sample_rate; }
  std::uint64_t files_written()    const noexcept { return _files_written; }

private:
  // Encode one clip of interleaved f32 PCM to `path` in the resolved
  // container. `pcm` is `channels * n_frames` interleaved samples.
  // Returns true on success. Errors are surfaced via session()->error
  // and leave the beat dropped (never crash).
  bool encode_wav_(const std::string& path, const float* pcm,
                   std::size_t n_frames, int channels, int sample_rate);
  bool encode_ffmpeg_(const std::string& path, const float* pcm,
                      std::size_t n_frames, int channels, int sample_rate,
                      const std::string& format);

  // Append `_files_written` as a zero-padded index before the extension
  // for the 2nd+ output file (out.aac -> out-001.aac).
  std::string next_output_path_(const std::string& ext) const;

  // ---- Config; defaults live in kSpec.attrs and are read in the ctor
  // via attr_*. ----
  std::string _output_path;
  std::string _format;       // resolved lower-case: wav | aac | mp3 | m4a
  int         _bitrate{};
  int         _sample_rate{};

  std::uint64_t _files_written = 0;
};

}

#endif

#ifndef VPIPE_STAGES_TEXT_TO_SPEECH_STAGE_H
#define VPIPE_STAGES_TEXT_TO_SPEECH_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

// The MOSS-TTS subsystem (MetalMossTtsModel + MetalMossCodec) is a
// from-scratch, MLX-free metal-compute stack: it builds on the
// VPIPE_BUILD_APPLE_SILICON axis and has NO MLX path (the moss model and
// codec are metal-only). Synthesis therefore runs on the metal-compute
// backend; this stage carries only that path. On non-Apple builds the
// stage is an inert stub (the constructor errors through session() and
// every beat emits nothing).
#ifdef VPIPE_BUILD_APPLE_SILICON
#include "generative-models/moss/metal-moss-tts-model.h"
#include "generative-models/moss/metal-moss-codec.h"
#include "generative-models/tokenizer.h"
#endif

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe::genai {
class Tokenizer;
}

namespace vpipe {

// Text-to-speech stage (MOSS-TTS, metal/no-MLX).
//
// Turns input text into a 24 kHz mono PCM waveform using the already-
// built MOSS-TTS metal models: a MetalMossTtsModel (an 8B Qwen3 backbone
// driving a [1+n_vq] delay-pattern code grid) generates RVQ audio codes,
// and a MetalMossCodec (the MOSS Audio Tokenizer decoder) turns those
// codes into PCM.
//
//   iport0  FlexDataPayload carrying the text to speak. Accepts either a
//           plain FlexData string OR a FlexData object with a "text"
//           key (mirrors text-chat's forgiving input handling).
//
//   oport0  TensorBeatPayload, rank-1 [n_samples] f32 PCM at 24 kHz, with
//           `sample_rate` set in the beat's sideband object. The oport is
//           unconditional; downstream consumers are optional (the runtime
//           drops writes when no cursor is attached).
//
// Per beat the stage:
//   1. Reads + normalizes the text (CR/CRLF -> space, drop ASCII control
//      chars except space, collapse whitespace runs, trim).
//   2. Renders the MOSS user_inst prompt and tokenizes it, then builds
//      the [seq][1+n_vq] input grid (channel 0 = text id, channels
//      1..n_vq = audio_pad_code, "no reference audio").
//   3. generate_delay_greedy(prompt, max_new_tokens) -> [G][1+n_vq].
//   4. De-delays the grid + drops all-pad frames -> codes [T][n_vq].
//   5. codec->decode(codes) -> [T*1920] f32 PCM @ 24 kHz.
//   6. Emits the PCM on oport0 as a TensorBeatPayload.
//
// Greedy generation only -- the MetalMossTtsModel exposes only greedy
// delay-pattern generation, so there is no sampler knob.
//
// Config (FlexData object on the 4th constructor parameter):
//   hf_dir         (string, required)      -- the MOSS-TTS LM directory
//                                             (config.json + safetensors).
//   codec_dir      (string, required)      -- the MOSS-Audio-Tokenizer
//                                             (codec) directory.
//   max_new_tokens (int, default 1024)     -- per-beat delay-pattern
//                                             generation budget (>= 1).
//
// Available on VPIPE_BUILD_APPLE_SILICON builds (the no-MLX metal path is
// the target; the moss models have no MLX path). On other builds the
// constructor logs an error through session() and the stage does nothing
// on each beat.
class TextToSpeechStage final : public TypedStage<TextToSpeechStage> {
public:
  static constexpr const char* kTypeName = "text-to-speech";

  TextToSpeechStage(const SessionContextIntf* session,
                    std::string               id,
                    std::vector<InEdge>       iports,
                    FlexData                  config);

  ~TextToSpeechStage() override;

  Job initialize(RuntimeContext& ctx) override;
  Job process   (RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only accessors.
  const std::string& hf_dir()    const noexcept { return _hf_dir; }
  const std::string& codec_dir() const noexcept { return _codec_dir; }
  int max_new_tokens()           const noexcept { return _max_new_tokens; }
  std::uint64_t clips_emitted()  const noexcept { return _clips_emitted; }

private:
  // Config attributes; defaults live in kSpec.attrs and are read in the
  // constructor via attr_*. Declarations carry no non-zero default.
  std::string _hf_dir;
  std::string _codec_dir;
  std::string _models_db;
  int         _max_new_tokens{};
  // TODO: flatten sampler config once the LM exposes sampling (greedy
  // delay-pattern generation is the only mode today).

#ifdef VPIPE_BUILD_APPLE_SILICON
  // Loaded lazily in initialize(); cleared on a load failure so the
  // stage stays inert (process() warns + emits nothing).
  std::unique_ptr<genai::MetalMossTtsModel> _lm;
  std::unique_ptr<genai::MetalMossCodec>    _codec;
  std::unique_ptr<genai::Tokenizer>         _tokenizer;
#endif

  // Bookkeeping for tests / logging.
  std::uint64_t _clips_emitted = 0;
};

}

#endif

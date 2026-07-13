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
#include "generative-models/moss/metal-moss-codec-v2.h"
#include "generative-models/moss/metal-moss-v15-model.h"
#include "generative-models/moss/metal-moss-rt-model.h"
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
// Handles BOTH MOSS-TTS variants from one stage, picked at load time from
// the LM dir's config.json `model_type`:
//
//   * "moss_tts" (the 8B delay-pattern model): a MetalMossTtsModel (8B Qwen3
//     backbone driving a [1+n_vq] delay-pattern code grid) generates RVQ
//     audio codes; a MetalMossCodec (MOSS Audio Tokenizer) decodes them to
//     24 kHz MONO PCM.
//   * "moss_tts_local" (v1.5): a MetalMossV15Model (Qwen3 backbone + per-frame
//     depth decoder, 12 RVQ codes/frame, greedy) feeding a MetalMossCodecV2
//     decoder -> 48 kHz STEREO PCM. The LM dir must be PRE-QUANTIZED (produce
//     it with the model-quantize stage); the codec_dir is a codec-v2 dir.
//
// Some config keys apply to only one variant (e.g. the sampling / voice-clone
// knobs are 8B-only; max_frames / instruction / language are v1.5-only) -- the
// unused keys are simply ignored for the loaded variant.
//
//   iport0  FlexDataPayload carrying the text to speak. Accepts either a
//           plain FlexData string OR a FlexData object with a "text"
//           key (mirrors text-chat's forgiving input handling). BARGE-IN
//           (interrupt_on_new_text, default on): if a new text beat arrives
//           here while the current utterance is still generating, the current
//           one is cut short -- the audio produced so far is flushed, then
//           generation stops and the newer text is served. Set the config
//           false to instead finish every utterance fully (queued FIFO).
//
//   iport1  OPTIONAL TensorBeatPayload, mono f32 PCM (any sample rate;
//           sideband.sample_rate honoured) -- a REFERENCE voice to clone.
//           When wired, the stage resamples it to the codec rate, encodes
//           it to RVQ codes (the MOSS-Audio-Tokenizer analysis path), and
//           splices those into the prompt's - Reference(s): section so the
//           synthesized speech adopts the reference's timbre. The latest
//           reference beat is STICKY -- it sets the voice for every later
//           utterance until a new reference arrives. This port is its OWN
//           clock domain (clock_group 1): it arrives independently of the
//           text stream / PCM output, not rate-locked to them, and is drained
//           non-blocking at the start of each beat. Loading the codec's
//           encoder ~doubles its resident weights, so it is loaded only when
//           iport1 is connected.
//
//   oport0  TensorBeatPayload f32 PCM ([channels, n_samples]; 24 kHz mono for
//           8B/realtime, 48 kHz stereo for v1.5), with `sample_rate` in the
//           beat's sideband. With stream_chunk_frames>0 (the default) the LM
//           decode and codec decode are INTERLEAVED: a chunk of PCM is emitted
//           every stream_chunk_frames generated frames (via the codec's
//           windowed-KV streaming decode), so a text beat produces a STREAM of
//           PCM beats with near-realtime first-audio latency instead of one big
//           beat at the end. stream_chunk_frames=0 restores the single-beat
//           one-shot decode. The oport is unconditional; downstream consumers
//           are optional (the runtime drops writes when no cursor is attached).
//
// Per beat the stage:
//   0. Drains any reference-audio beats on iport1 (voice cloning): resamples
//      each to the codec rate, encodes it to RVQ codes, and keeps it (sticky)
//      as the clone reference. Under voice_lock, the first generated voice is
//      cached as the reference instead (design-once).
//   1. Reads + normalizes the text (CR/CRLF -> space, drop ASCII control
//      chars except space, collapse whitespace runs, trim).
//   2. Builds the [seq][1+n_vq] input grid (channel 0 = text/control id,
//      channels 1..n_vq = audio codes). With a clone reference, splices it
//      into the - Reference(s): section (audio_user_slot + delay-patterned
//      reference codes); otherwise - Reference(s): None.
//   3. generate_delay(prompt, max_new_tokens, audio/text sampling, seed)
//      -> [G][1+n_vq].
//   4. De-delays the grid + drops all-pad frames -> codes [T][n_vq].
//   5. codec->decode(codes) -> [T*1920] f32 PCM @ 24 kHz.
//   6. Emits the PCM on oport0 as a TensorBeatPayload.
//
// Audio codes are SAMPLED (the MossTTSDelay-8B recommendation); the text
// channel defaults to greedy (vpipe re-emits the transcript there). All
// sampling knobs + the voice-lock seed are flattened config (see kAttrs).
//
// Config (FlexData object on the 4th constructor parameter; see kAttrs for
// the full flattened list incl. sampling + voice cloning):
//   hf_dir         (string, required)      -- the MOSS-TTS LM directory
//                                             (config.json + safetensors).
//   codec_dir      (string, required)      -- the MOSS-Audio-Tokenizer
//                                             (codec) directory.
//   max_new_tokens (int, default 1024)     -- per-beat delay-pattern
//                                             generation budget (>= 1).
//   voice_lock     (bool, default false)   -- reuse the first voice across
//                                             beats (design-once).
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
  int max_frames()               const noexcept { return _max_frames; }
  int stream_chunk_frames()      const noexcept { return _stream_chunk; }
  bool interrupt_on_new_text()   const noexcept
  { return _interrupt_on_new_text; }
  std::uint64_t clips_emitted()  const noexcept { return _clips_emitted; }

private:
  // Config attributes; defaults live in kSpec.attrs and are read in the
  // constructor via attr_*. Declarations carry no non-zero default.
  std::string _hf_dir;
  std::string _codec_dir;
  std::string _models_db;
  int         _max_new_tokens{};
  int         _stream_chunk{}; // emit PCM every N codec frames (0 = one-shot)
  bool        _interrupt_on_new_text{}; // barge-in: abort in-flight on new text
  bool        _codec_int8{};   // codec_quant == "int8": int8 g32 codec weights
  // v1.5-only config (ignored for the 8B variant).
  int         _max_frames{};
  std::string _instruction;
  std::string _language;
  // Flattened sampling config (separate audio + text channels); assembled into
  // MossSampling in process(). Defaults = MossTTSDelay-8B recommendations.
  double        _audio_temp{}, _audio_top_p{}, _audio_rep{};
  int           _audio_top_k{};
  double        _text_temp{}, _text_top_p{}, _text_rep{};
  int           _text_top_k{};
  std::uint64_t _sampler_seed{};
  // Voice cloning / lock. _with_encoder (set in the ctor from the iport
  // count) gates loading the codec's encode path -- only paid when a PCM
  // reference iport is wired. _voice_lock is the "design-once" mode: cache
  // the FIRST generated voice and reuse it as the reference for every later
  // beat, so the timbre stays consistent across different texts (a fixed
  // sampler_seed picks that first voice deterministically). An external
  // reference on iport1 overrides it. _voice_ref_seconds caps how much
  // reference audio is kept (longer prompts cost more per beat).
  bool          _with_encoder{};
  bool          _voice_lock{};
  double        _voice_ref_seconds{};

#ifdef VPIPE_BUILD_APPLE_SILICON
  // Loaded lazily in initialize(); cleared on a load failure so the
  // stage stays inert (process() warns + emits nothing). Exactly one variant
  // is loaded per stage (chosen from config.json model_type): the 8B pair
  // (_lm + _codec) OR the v1.5 pair (_lm_v15 + _codec_v15).
  std::unique_ptr<genai::MetalMossTtsModel> _lm;
  std::unique_ptr<genai::MetalMossCodec>    _codec;
  std::unique_ptr<genai::MetalMossV15Model> _lm_v15;
  std::unique_ptr<genai::MetalMossCodecV2>  _codec_v15;
  // Realtime variant (model_type moss_tts_realtime): a MetalMossRtModel
  // (Qwen3-1.7B backbone + 4-layer depth decoder, 16 RVQ codes/frame) feeding
  // the 24 kHz MetalMossCodec (_codec, shared with the 8B path). Loaded only
  // when the LM dir's config.json model_type is "moss_tts_realtime".
  std::unique_ptr<genai::MetalMossRtModel>  _lm_rt;
  std::unique_ptr<genai::Tokenizer>         _tokenizer;
  // Cached codec streaming state, reused across beats (the windowed K/V rings
  // are sized by stream_chunk_frames, not utterance length, so there is no need
  // to reallocate per beat -- reset() re-arms them). _stream_v15 pairs with the
  // codec-v2 (v1.5) path, _stream_v1 with the codec-v1 (8B/realtime) path.
  std::unique_ptr<genai::MetalMossCodecV2::StreamState> _stream_v15;
  std::unique_ptr<genai::MetalMossCodec::StreamState>   _stream_v1;
  // Active clone reference: RVQ codes [T][n_vq] spliced into the prompt.
  // Set from an iport1 PCM beat (encode) or, under _voice_lock, from the
  // first beat's own generated codes. Empty => no reference (plain TTS).
  std::vector<std::vector<std::int32_t>>    _ref_codes;
  bool                                      _ref_set = false;
#endif

  // Bookkeeping for tests / logging.
  std::uint64_t _clips_emitted = 0;
};

}

#endif

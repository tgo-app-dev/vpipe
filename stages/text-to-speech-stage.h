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
#include "generative-models/sampler.h"
#include "generative-models/moss/metal-moss-tts-model.h"
#include "generative-models/moss/metal-moss-codec.h"
#include "generative-models/moss/metal-moss-codec-v2.h"
#include "generative-models/moss/metal-moss-v15-model.h"
#include "generative-models/moss/metal-moss-rt-model.h"
#include "generative-models/drama3/drama3-prompt.h"
#include "generative-models/drama3/metal-drama3-model.h"
#include "generative-models/drama3/metal-fish-codec.h"
#include "generative-models/moe-expert-cache.h"
#include "generative-models/qwen3/metal-qwen-model.h"
#include "generative-models/tokenizer.h"
#endif

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace vpipe::genai {
class Tokenizer;
}

namespace vpipe {

// Text-to-speech stage (MOSS-TTS and Fish Audio Drama 3, metal/no-MLX).
//
// Handles every variant from one stage, picked at load time from the LM
// dir's config.json `model_type`:
//
//   * "fish_omni" -- Fish Audio Drama 3 (fishaudio/drama-3): a MetalDrama3
//     Model (a Qwen3.5-35B-A3B backbone with block-FP8 experts + a 13-layer
//     codebook-MTP tower, ten codes a frame) feeding the MetalFishCodec in
//     the same directory (codec.pth; codec_dir optional) -> 44.1 kHz MONO
//     PCM. The text may carry speaker tags (<|speaker:0|> .. <|speaker:15|>)
//     and inline direction ([laughs], [pause]). A reference voice on
//     iport1 needs its transcript (`reference_text`, or a "reference_text"
//     field on the text beat). The audio sampler's unwired default is the
//     reference's request default (T 1.0, top_p 0.9, top_k 30);
//     `max_frames` is the frame budget (each ~46 ms). The MOSS-only keys
//     (instruction, language, quality, ..., lm_quant, i8_gemm) do not
//     apply.
//
//   * "moss_tts_delay" / "moss_tts" -- the DELAY-PATTERN 8B: MOSS-TTS 1.0
//     and MOSS-TTS-v1.5 (one architecture). A MetalMossTtsModel (Qwen3-8B
//     backbone driving a [1+n_vq] delay-pattern code grid) generates RVQ
//     codes; a MetalMossCodec (MOSS Audio Tokenizer) decodes them to 24 kHz
//     MONO PCM. The prompt carries `language`, `instruction` and
//     `duration_tokens`; v1.5's text normalizer runs when its script is in
//     the LM dir (`text_normalizer`).
//   * "moss_tts_local" (Local-v1.5): a MetalMossV15Model (Qwen3 backbone +
//     per-frame depth decoder, 12 RVQ codes/frame) feeding a
//     MetalMossCodecV2 decoder -> 48 kHz STEREO PCM.
//   * "moss_tts_realtime": a MetalMossRtModel (Qwen3-1.7B backbone + depth
//     decoder, 16 codes/frame) feeding the 24 kHz MetalMossCodec.
//
// MEMORY (docs/MODEL-MEMORY.md). Every variant's core is an autoregressive
// Qwen3 backbone that reads all of its weights per generated frame, so it
// must stay inside RAM. An UNQUANTIZED checkpoint's backbone is held as
// affine w8 group-64 built in memory (`lm_quant`, ~53% of bf16); when even
// that does not fit, the backbone STREAMS its layers -- decided before
// load by plan_streaming, with what quantizing sheds as `retires` -- and
// grows a resident set into what the box will hold. Both ledgers declare
// what the models will hold (the streamable LM floor, the codec at f16)
// and are corrected from what they report after load and after every
// utterance; K/V and the codec working set are a scratch claim.
//
// Some config keys apply to only one variant (e.g. the voice-clone knobs
// do not reach the realtime variant; max_frames is the depth-decoder
// variants') -- the unused keys are simply ignored for the loaded
// variant. Sampling is NOT config: it comes from the two optional sampler
// iports below.
//
//   iport0  FlexDataPayload carrying the text to speak. Accepts either a
//           plain FlexData string OR a FlexData object with a "text"
//           key (mirrors text-chat's forgiving input handling). An object
//           may also carry "end_of_response" (bool) -- text-chat's `stream`
//           oport sets it, false on every chunk of a reply except the last.
//           BARGE-IN (interrupt_on_new_text, default on): if a new text beat
//           arrives here while the current utterance is still generating, the
//           current one is cut short -- the audio produced so far is flushed,
//           then generation stops and the newer text is served. The cut is
//           GATED by end_of_response: a chunk with end_of_response=false is
//           the same reply continuing, so it is always finished in full and a
//           waiting beat is served next (never cut); only a chunk that is
//           end_of_response (true, or a plain beat with no such field ->
//           defaults true) is cut when a newer beat waits. Set the config
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
//   iport2  OPTIONAL FlexDataPayload token-sampler spec (a `sampler-select`
//           stage) for the AUDIO-code channel; latched on the first beat.
//           Its seed doubles as the voice seed. UNLIKE every other sampler
//           port in the tree, an unwired iport2 does NOT fall back to greedy:
//           MOSS's audio head degenerates into silent loops under argmax, so
//           it keeps the model's own recommendation instead (temp 1.7 /
//           top_p 0.8 / top_k 25 for the delay-pattern variants and
//           Local-v1.5; temp 0.8 / top_p 0.6 / top_k 30 / rep 1.1 for the
//           realtime variant).
//
//   iport3  OPTIONAL FlexDataPayload token-sampler spec for the free-TEXT
//           channel; latched on the first beat. Unwired => greedy, which is
//           the right default: vpipe GENERATES the text channel (it re-emits
//           the transcript) and must follow it to reach audio_end, so
//           sampling it over-generates.
//
//           MossSampling has no min_p / presence-penalty knob, so a spec
//           setting either warns on latch and that knob is ignored.
//
//   oport0  TensorBeatPayload f32 PCM ([channels, n_samples]; 24 kHz mono for
//           the delay-pattern and realtime variants, 48 kHz stereo for
//           Local-v1.5), with `sample_rate` in the beat's sideband. With
//           stream_chunk_frames>0 (the default) the LM decode and codec
//           decode are INTERLEAVED: a chunk of PCM is emitted
//           every stream_chunk_frames generated frames (via the codec's
//           windowed-KV streaming decode), so a text beat produces a STREAM of
//           PCM beats with near-realtime first-audio latency instead of one big
//           beat at the end. stream_chunk_frames=0 restores the single-beat
//           one-shot decode. The oport is unconditional; downstream consumers
//           are optional (the runtime drops writes when no cursor is attached).
//
// Per beat the stage (delay-pattern variant):
//   0. Drains any reference-audio beats on iport1 (voice cloning): mixes
//      each down to mono, resamples it to the codec rate, loudness-
//      normalizes it (-20 dBFS, +-3 dB, as the reference processor does),
//      encodes it to RVQ codes and keeps it (sticky) as the clone
//      reference. Under voice_lock, the first generated voice is cached as
//      the reference instead (design-once).
//   1. Cleans the text: v1.5's robust normalizer, or the basic clean-up
//      (CR/CRLF -> space, control chars dropped, whitespace collapsed).
//   2. Builds the [seq][1+n_vq] input grid (channel 0 = text/control id,
//      channels 1..n_vq = audio codes) with the <user_inst> fields. With a
//      clone reference, splices it into - Reference(s): (audio_user_slot
//      rows + delay-patterned reference codes); otherwise None.
//   3. generate_delay(prompt, max_new_tokens, audio/text sampling, seed),
//      de-delaying rows into codec frames as they arrive.
//   4. Streams PCM: every stream_chunk_frames frames through the codec's
//      windowed streaming decode (bit-identical to one-shot), or one
//      decode of the whole utterance when stream_chunk_frames is 0.
//   5. Emits the PCM on oport0 as TensorBeatPayloads.
//
// Audio codes are SAMPLED (the MossTTSDelay-8B recommendation); the text
// channel defaults to greedy (vpipe re-emits the transcript there). Neither
// is a config key -- both come from the sampler iports above, and the audio
// spec's seed is the voice-lock seed.
//
// Config (FlexData object on the 4th constructor parameter; see kAttrs for
// the full flattened list incl. voice cloning):
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

  // The TTS LM and its audio codec -- two checkpoints, both large.
  // See Stage::declare_resources.
  std::vector<ResourceClaim> declare_resources() const override;
  // See Stage::declare_memory.
  StageMemory declare_memory() const override;
  Job process   (RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only accessors.
  const std::string& hf_dir()    const noexcept { return _hf_dir; }
  const std::string& codec_dir() const noexcept { return _codec_dir; }
  int max_new_tokens()           const noexcept { return _max_new_tokens; }
  int max_frames()               const noexcept { return _max_frames; }
  int duration_tokens()          const noexcept { return _duration_tokens; }
  const std::string& lm_quant()  const noexcept { return _lm_quant; }
  bool lm_streams()              const noexcept { return _lm_streams; }
  bool i8_gemm()                 const noexcept { return _i8_gemm; }
  const std::string& quality() const noexcept { return _quality; }
  const std::string& sound_event() const noexcept { return _sound_event; }
  const std::string& ambient_sound() const noexcept
  { return _ambient_sound; }
  bool wait_for_reference() const noexcept { return _wait_reference; }
  const std::string& text_normalizer() const noexcept
  { return _text_normalizer; }
  int stream_chunk_frames()      const noexcept { return _stream_chunk; }
  bool interrupt_on_new_text()   const noexcept
  { return _interrupt_on_new_text; }
  std::uint64_t clips_emitted()  const noexcept { return _clips_emitted; }
  // Test-only: whether the codec's (heavy) encode path will be loaded. True
  // only when the PCM-reference iport1 SLOT is wired -- the sampler iports
  // that follow it must not turn it on.
  bool with_encoder()            const noexcept { return _with_encoder; }
  const std::string& reference_text() const noexcept
  { return _reference_text; }
  bool chunk_text() const noexcept { return _d3_chunk_text; }
  const std::string& context() const noexcept { return _d3_context; }
  std::size_t configured_references() const noexcept
  { return _d3_ref_specs.size(); }
#ifdef VPIPE_BUILD_APPLE_SILICON
  // Test-only: the loaded Drama 3 model (null for the MOSS variants), the
  // voices it clones now (one per speaker), the takes it remembers for
  // `previous_takes`, and the last beat's chunk count.
  genai::MetalDrama3Model* drama3() const noexcept { return _d3.get(); }
  const std::vector<genai::drama3::Reference>& drama3_references() const
  { return _d3_refs; }
  std::size_t drama3_takes() const noexcept { return _d3_takes.size(); }
  struct Drama3Beat {
    int chunks = 0;        // the text's chunks
    int spoken = 0;        // of them, those that produced frames
    int previous = 0;      // previous_takes found
    int voices = 0;        // voices cloned
    int with_history = 0;  // chunks spoken after earlier takes
    int max_prompt = 0;    // the longest prompt, in rows
    int frames = 0;
  };
  const std::vector<Drama3Beat>& drama3_beats() const { return _d3_beats; }
#endif

private:
  // Config attributes; defaults live in kSpec.attrs and are read in the
  // constructor via attr_*. Declarations carry no non-zero default.
  std::string _hf_dir;
  std::string _codec_dir;
  int         _max_new_tokens{};
  int         _stream_chunk{}; // emit PCM every N codec frames (0 = one-shot)
  bool        _interrupt_on_new_text{}; // barge-in: abort in-flight on new text
  bool        _codec_int8{};   // codec_quant == "int8": int8 g32 codec weights
  // Prompt fields. instruction / language go into both the delay-pattern
  // (MOSS-TTS / MOSS-TTS-v1.5) and the Local-v1.5 prompt; duration_tokens
  // (the "- Tokens:" field, 0 = None) only into the delay-pattern one.
  // max_frames is the frame budget of the depth-decoder variants.
  int         _max_frames{};
  std::string _instruction;
  // The other whole-utterance fields (delay-pattern variants): quality,
  // sound event, ambient sound -- "None" unset, as the processor renders.
  std::string _quality;
  std::string _sound_event;
  std::string _ambient_sound;
  // wait_for_reference: the first beat waits for the audio-ref iport.
  bool        _wait_reference{};
  bool        _ref_waited{};
  std::string _language;
  int         _duration_tokens{};
  // "auto" | "robust" | "basic": which text clean-up runs before the
  // prompt is built (see kAttrs). Resolved per load into _robust_text.
  std::string _text_normalizer;
  bool        _robust_text{};
  // How an UNQUANTIZED LM backbone is held: "w8" (in-memory affine 8-bit,
  // the default) or "bf16". _lm_quant_bits is the resolved 8 / 0.
  std::string _lm_quant;
  int         _lm_quant_bits{};
  bool        _i8_gemm{};   // the backbone's int8 prefill tier (M5)
  bool        _autotune = true;  // its kernel tuners measure (kAutotune)
  std::uint64_t _i8_seen{};   // engagement count at the last revision
  int _kept_seen{-1};          // resident layers at the last revision

  // ---- memory accounting (docs/MODEL-MEMORY.md) -----------------------
  // Whether the LM backbone streams (plan_streaming's verdict, taken at
  // load and irreversible), the floor its holding was planned at, and the
  // per-stage scratch label. The plan is declared from the checkpoints
  // alone and corrected from what the models report after load and after
  // every utterance (revise_holdings_).
  bool        _lm_streams{};
  std::size_t _lm_floor{};
  std::size_t _lm_trunk{};          // non-backbone bytes (RT / Local)
  std::string _lm_dir_resolved;
  std::string _codec_dir_resolved;
  std::string scratch_label_() const;
  // Drama 3's codec.pth: in codec_dir when one is set (a directory holding
  // it, or the file), else beside the LM.
  std::string drama3_codec_path_(const std::string& lm_dir) const;
  std::size_t scratch_bytes_(const std::string& lm_dir,
                             std::size_t* kv_part = nullptr) const;
  void        revise_holdings_();
  // Voice cloning / lock. _with_encoder (set in the ctor from the iport1
  // SLOT, not the port count -- the sampler iports sit after it) gates
  // loading the codec's encode path, only paid when a PCM reference iport is
  // wired. _voice_lock is the "design-once" mode: cache the FIRST generated
  // voice and reuse it as the reference for every later beat, so the timbre
  // stays consistent across different texts (a fixed audio-sampler seed picks
  // that first voice deterministically). An external reference on iport1
  // overrides it. _voice_ref_seconds caps how much reference audio is kept
  // (longer prompts cost more per beat).
  bool          _with_encoder{};
  bool          _voice_lock{};
  double        _voice_ref_seconds{};
  // Drama 3: the transcript of the reference voice on iport1.
  std::string   _reference_text;
  // Drama 3: `chunk_text` (split long text as the reference server does),
  // `context` ("text": a beat's chunks share the history; "run": every
  // beat of the run does), and `references` (voices to clone, from files).
  struct D3RefSpec {
    std::string audio;
    std::string text;
  };
  bool                   _d3_chunk_text = true;
  std::string            _d3_context = "text";
  std::vector<D3RefSpec> _d3_ref_specs;

#ifdef VPIPE_BUILD_APPLE_SILICON
  // Per-channel token-sampler knobs, latched once off the OPTIONAL sampler
  // iports 2 (audio codes) and 3 (free text).
  //
  // The TEXT channel falls back to the SamplerParams default (argmax), like
  // every other LLM stage: vpipe GENERATES the text channel (it re-emits the
  // transcript) and must follow it to reach audio_end, so sampling it
  // over-generates.
  //
  // The AUDIO channel does NOT: MOSS degenerates into silent loops under
  // greedy, so an unwired iport2 keeps the model's own recommended sampling
  // (moss_audio_default_sampler(), set in the ctor; the realtime variant has
  // its own recommendation and applies it at its call site). The audio
  // spec's seed doubles as the voice seed -- it is the only RNG the
  // delay-pattern generator takes.
  genai::SamplerParams _audio_sp;
  genai::SamplerParams _text_sp;
  // _*_read: the port's one beat has been consumed (never re-read it, even
  // if it was unusable). _audio_sp_set: a VALID spec was accepted, so
  // _audio_sp is the operator's and not MOSS's default -- the realtime
  // variant needs the distinction because its own fallback differs from the
  // 8B/v1.5 one seeded in the ctor.
  bool                 _audio_sp_read = false;
  bool                 _audio_sp_set  = false;
  bool                 _text_sp_read  = false;
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
  // Fish Audio Drama 3 (model_type fish_omni): the speech LM and its codec.
  // The clone reference is [frames][10] codes + its transcript.
  std::unique_ptr<genai::MetalDrama3Model>  _d3;
  std::unique_ptr<genai::MetalFishCodec>    _fish;
  // Drama 3's streamed decode (stream_chunk_frames > 0), reused across
  // beats; its buffers are the codec's, so it goes before the codec.
  std::unique_ptr<genai::MetalFishCodec::DecodeStream> _fish_stream;
  // The voices cloned now, in prompt order: the configured `references`
  // first, then those from iport1 -- ONE PER SPEAKER (the transcript's
  // leading tag, 0 without one), a later one replacing an earlier. The
  // index of the latest from iport1 is what a text beat's
  // "reference_text" retitles.
  std::vector<genai::drama3::Reference>     _d3_refs;
  int                                       _d3_port_ref = -1;
  // context "run": the run's history (its first take pinned, the latest).
  genai::drama3::History                    _d3_run_hist;
  // The takes spoken this run, newest last (bounded), for a beat's
  // "previous_takes" to name by their text.
  std::deque<genai::drama3::Take>           _d3_takes;
  std::vector<Drama3Beat>                   _d3_beats;
  // The expert cache's counters at the last beat (for per-text deltas).
  genai::MoeExpertCache::Stats              _d3_xstats;
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

  // Drop every loaded model + its streaming state. Called at the top of
  // initialize() so a relaunch over a stopped-but-not-destroyed stage
  // frees the previous run's weights BEFORE loading the next copy,
  // instead of transiently holding both.
  void release_models_();
  void load_drama3_(const std::string& lm_dir,
                    metal_compute::MetalCompute* mc);
  // A voice into _d3_refs, replacing the one for its speaker.
  void d3_add_reference_(genai::drama3::Reference r, bool from_port);
  // Decode, trim and encode the configured `references`; "" or why not.
  std::string d3_load_references_();
#endif

  // Bookkeeping for tests / logging.
  std::uint64_t _clips_emitted = 0;
};

}

#endif

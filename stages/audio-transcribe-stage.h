#ifndef VPIPE_STAGES_AUDIO_TRANSCRIBE_STAGE_H
#define VPIPE_STAGES_AUDIO_TRANSCRIBE_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

// The LM subsystem (LoadedLanguageModel, sampler) is MLX-free and builds
// on the VPIPE_BUILD_APPLE_SILICON axis; transcription runs on the
// metal-compute backend (MetalAudioEncoder) when MLX is off.
#ifdef VPIPE_BUILD_APPLE_SILICON
#include "generative-models/loaded-language-model.h"
#include "generative-models/sampler.h"
#endif

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe::genai {
class LoadedLanguageModel;
class AudioEncoder;
class MetalAudioEncoder;
}

namespace vpipe {

// Audio transcription stage.
//
// Qwen3-ASR (the audio family today) only transcribes -- it ignores
// instruction/question prompting -- so this stage is transcription-
// only: one audio clip in, one transcript line out. No system-prompt
// instruction knob, no context branching.
//
//   iport0  TensorBeatPayload, f32 mono PCM samples. The expected
//           shape is rank-1 [n_samples] OR rank-2 [1, n_samples].
//           Sample rate is read from sideband.sample_rate when
//           present, otherwise from the `sample_rate` config below.
//
//           BLOCK MODE (iport1 unwired, the default): each incoming beat
//           is treated as ONE complete audio clip to transcribe.
//
//           STREAMING MODE (iport1 wired): PCM frames are buffered
//           into a rolling timestamp-indexed window of length
//           `pcm_buffer_s`. The stage waits for segment markers on
//           iport1; when one arrives it slices [start_us, end_us) out
//           of the buffer and pushes the resulting clip through the
//           same encode + prefill + decode pipeline used in block
//           mode. PCM beats on iport0 may be arbitrarily small frames
//           (e.g. 32 ms) -- see audio-to-pcm `chunk_duration_s` and
//           audio-segment.
//
//   iport1  (OPTIONAL) FlexDataPayload segment marker from `audio-segment`
//           (or any producer of that object). Wiring iport1 is what flips
//           the stage into streaming mode at construction time. The object
//           carries `start_us`, `end_us`, `index`, and `is_partial`; on
//           arrival the stage slices the PCM in [start_us, end_us) from its
//           rolling buffer and transcribes it.
//
//   iport2  (OPTIONAL) FlexDataPayload token-sampler spec from a
//           `sampler-select` stage. Latched on the first beat and reused for
//           every later clip. Unwired => greedy (argmax) decoding, which is
//           what transcription wants and what this stage has always done by
//           default. Wiring it does NOT flip streaming mode on -- that is
//           iport1's job, checked by slot.
//
//   oport0  (OPTIONAL) FlexDataPayload carrying the recognized transcript as
//           an object {text, start_us, end_us}. `text` is the decoded string;
//           `start_us`/`end_us` are the segment span (STREAMING mode only --
//           block mode has no timestamps, so they are omitted). Unconnected is
//           fine: the transcript is still logged via the UI delegate. Feed it
//           to save-text to save transcripts to a file.
//
// Per beat the stage:
//   1. Pulls the PCM samples.
//   2. Encodes them through the LM's AudioEncoder (Qwen3-ASR
//      family today) to produce [n_audio_tokens, lm_hidden]
//      embeddings.
//   3. Renders the family's ASR chat template (user turn wrapping
//      audio_pad tokens + assistant header + optional
//      `language X<asr_text>` hint).
//   4. Builds a TokenRef stream that interleaves text ids with
//      AudioTokens refs pointing into the encoder output.
//   5. prefill_multimodal + greedy decode until the chat
//      template's stop token (or max_new_tokens).
//   6. Surfaces the decoded text to the user via the session UI
//      delegate (session()->info).
//
// Config (FlexData object on the 4th constructor parameter):
//   hf_dir          (string, required)         -- ASR-LM HF-style dir
//   compute_dtype   (string, default "f16")    -- "bf16" | "f16" | "f32"
//   page_tokens     (int,    default 16)
//   max_pages       (int,    default 256)
//   max_new_tokens  (int,    default 256)
//   sample_rate     (int,    default 16000)    -- fallback when
//                                                 sideband.sample_rate
//                                                 is absent
//   language_hint   (string, default "")       -- when non-empty, the
//                                                 chat template
//                                                 pre-emits
//                                                 "language X<asr_text>"
//                                                 so the model only
//                                                 generates the
//                                                 transcript; otherwise
//                                                 the model
//                                                 auto-detects
//   pcm_buffer_s    (real,   default 30.0)     -- streaming-mode rolling
//                                                 PCM buffer length (s).
//                                                 Samples older than
//                                                 now-pcm_buffer_s are
//                                                 evicted to bound
//                                                 memory. Must exceed
//                                                 audio-segment's
//                                                 max_segment_s plus
//                                                 worst-case marker
//                                                 latency.
//   late_marker_skip(bool,   default true)     -- streaming mode only.
//                                                 When a marker arrives
//                                                 whose start_us has
//                                                 already been evicted
//                                                 from the buffer, skip
//                                                 it (log warn). When
//                                                 false, transcribe the
//                                                 available tail of the
//                                                 segment instead.
//
// Sampling is NOT config: it comes from a `sampler-select` stage on iport2
// (above). Unwired, the stage routes through the fast pipelined argmax path;
// a spec with ANY non-default knob switches it to the sampled path (sync
// next_token + host-pull logits + Sampler::sample). The per-clip INFO log
// tags " [sampled]" so the active path is visible at runtime.
class AudioTranscribeStage final
    : public TypedStage<AudioTranscribeStage> {
public:
  static constexpr const char* kTypeName = "audio-transcribe";

  AudioTranscribeStage(const SessionContextIntf* session,
                       std::string               id,
                       std::vector<InEdge>       iports,
                       FlexData                  config);
  ~AudioTranscribeStage() override;

  Job initialize(RuntimeContext& ctx) override;

  // The LM this stage holds. Declared before any stage initializes so
  // the diffusion stages -- which cannot see it from their own config --
  // size the box against it. See Stage::declare_resources.
  std::vector<ResourceClaim> declare_resources() const override;
  // See Stage::declare_memory.
  StageMemory declare_memory() const override;
  void reset_run_state() override;
  Job process   (RuntimeContext& ctx) override;
  Job drain     (RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only accessors.
  const std::string& hf_dir()         const noexcept { return _hf_dir; }
  const std::string& language_hint()  const noexcept { return _language_hint; }
  int                max_new_tokens() const noexcept { return _max_new_tokens; }
  int                sample_rate()    const noexcept { return _sample_rate; }
  std::uint64_t      clips_processed() const noexcept { return _clips_processed; }
  bool               streaming()      const noexcept { return _streaming; }
  double             pcm_buffer_s()   const noexcept { return _pcm_buffer_s; }
  bool               late_marker_skip() const noexcept
  { return _late_marker_skip; }
  std::uint64_t      segments_seen() const noexcept { return _segments_seen; }
  std::uint64_t      segments_dropped_late() const noexcept
  { return _segments_dropped_late; }

  // Resolve `ref` to a model directory: when it is a key in the model
  // registry, return that record's local_path -- a DB entry wins over a
  // same-named filesystem path; otherwise return `ref` unchanged so it is
  // used as a plain path. Public so tests can drive it directly
  // (initialize() calls it before loading the LM).
  std::string resolve_model_dir(const std::string& ref) const;

private:
#if defined(VPIPE_BUILD_APPLE_SILICON)
  // Metal/no-MLX counterpart: metal audio encode -> host-f32 embeddings
  // -> mROPE-free multimodal prefill -> greedy/host-sampled decode.
  bool m_transcribe_one_(const float*  pcm,
                         std::size_t   n_samples,
                         int           sample_rate);
#endif

#ifdef VPIPE_BUILD_APPLE_SILICON
  // Streaming mode: ingest one PCM beat into the rolling buffer. Drops
  // samples older than now-pcm_buffer_s. Returns false on a malformed
  // beat (already warned).
  bool ingest_pcm_beat_(const class TensorBeatPayload& tbp);

  // Streaming mode: slice the rolling buffer at [start_us, end_us)
  // and call the underlying encoder + prefill + decode path. Returns
  // false on a missing-range / late-marker skip.
  bool slice_and_transcribe_(std::uint64_t start_us,
                             std::uint64_t end_us);

  // Streaming mode: choose between MLX / metal `transcribe_one_`.
  bool transcribe_pcm_(const float* pcm,
                       std::size_t  n_samples,
                       int          sample_rate);
#endif

  // ---- Config attributes; defaults live in kSpec.attrs and are read
  // in the constructor via attr_*. Declarations carry no non-zero
  // default. ----
  std::string   _hf_dir;
  std::string   _compute_dtype;
  // page_tokens default (512) matches GenerativeModelManager. Smaller
  // pages multiply per-chunk prefill dispatch overhead by k =
  // ceil(prompt_len / page_tokens); at page=16 a 200-token ASR prompt
  // took 170 ms warm prefill, at page=512 it takes 66 ms (single chunk).
  int           _page_tokens{};
  std::uint32_t _max_pages{};
  int           _max_new_tokens{};
  int           _sample_rate{};
  std::string   _language_hint;
  // Streaming-mode knobs (only consulted when iport1 is wired).
  double        _pcm_buffer_s     = 30.0;
  bool          _late_marker_skip = true;

  // Streaming-mode rolling PCM ring buffer. Concatenated mono f32 at
  // `_sample_rate`. `_pcm_buf[0]` corresponds to UTC `_pcm_base_us`.
  std::vector<float> _pcm_buf;
  std::uint64_t      _pcm_base_us  = 0;
  bool               _pcm_have_ts  = false;
  // Latched at construction from a WIRED segment-marker iport1 SLOT (not
  // from the port count -- the sampler iport2 sits after it). Fixed for the
  // life of the stage.
  bool               _streaming    = false;
  std::uint64_t      _segments_seen         = 0;
  std::uint64_t      _segments_dropped_late = 0;

  // Transcript oport plumbing. m_transcribe_one_ is not a coroutine, so it
  // stashes each finished transcript here (as {text[, start_us, end_us]}) and
  // process() drains the queue to oport 0. `_cur_*_us` carry the segment span
  // for the in-flight clip (set by slice_and_transcribe_; absent in block
  // mode, which has no timestamps).
  std::vector<FlexData> _out_pending;
  std::uint64_t         _cur_start_us = 0;
  std::uint64_t         _cur_end_us   = 0;
  bool                  _cur_has_ts   = false;
#ifdef VPIPE_BUILD_APPLE_SILICON
  // Token-sampler knobs, latched once off the OPTIONAL sampler iport2 (a
  // `sampler-select` source). Defaults are argmax, so an unwired port keeps
  // the greedy decoding transcription wants.
  genai::SamplerParams _sampler_params;
  bool                 _sampler_latched = false;
  std::shared_ptr<genai::LoadedLanguageModel> _lm;
  bool _encoder_unavailable_warned = false;
#endif

#if defined(VPIPE_BUILD_APPLE_SILICON)
  // Borrowed metal audio tower (host-f32 embeddings). Cached at init.
  genai::MetalAudioEncoder* _m_audio = nullptr;
#endif

  // Bookkeeping for tests / logging.
  std::uint64_t _clips_processed = 0;
};

}

#endif

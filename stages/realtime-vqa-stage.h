#ifndef VPIPE_STAGES_REALTIME_VQA_STAGE_H
#define VPIPE_STAGES_REALTIME_VQA_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"


// No-MLX metal path: the LM subsystem (MLX-free) drives generation on the
// metal-compute backend. These headers carry no mlx::core types.
#if defined(VPIPE_BUILD_APPLE_SILICON)
#include "generative-models/chat-template.h"
#include "generative-models/shared/coreml-vision-encoder.h"
#include "generative-models/loaded-language-model.h"
#include "generative-models/gemma4/metal-gemma4-audio.h"
#include "generative-models/gemma4/metal-gemma4-vision.h"
#include "generative-models/qwen3/metal-qwen-vision.h"
#include "generative-models/sampler.h"
#endif

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace vpipe::genai {
class LoadedLanguageModel;
class VisionEncoder;
class MetalGemma4VisionEncoder;
class MetalGemma4AudioEncoder;
class Gemma4UnifiedEmbedder;
}

namespace vpipe {

// Reusable Shared/UMA buffer handle for the GPU resampler (defined in
// apple-silicon/tensor-storage.h; only the .cc needs the full type).
struct ExternalStorageHandle;

// Real-time visual-QA stage.
//
//   iport0  TensorBeatPayload, planar u8 [3, H, W] RGB video frames.
//           Sideband.timestamp_us (when present) drives the in-scene
//           "frame is substantially apart in time" boundary check.
//
//   iport1  TriggerPayload (or any payload sharing the clock domain
//           of a chrono stage). Used as the periodic driver — between
//           ticks the stage drains whatever frames arrived on iport0.
//           Two consecutive ticks with no frame in between close the
//           current scene.
//
//   iport2  (optional) FlexDataPayload from an audio-tagging stage:
//           { "timestamp_us": uint, "tags": [ { "label": str, ... },
//             ... ] }. NOT in the same clock domain as iport0/iport1 —
//           the audio tagger runs on its own (audio chunk) clock. Each
//           tick the stage pulls every classification whose window
//           timestamp is strictly before the latest received frame's
//           timestamp ("prior to its image data"), records the top
//           class when it is not "Silence", and weaves it into the
//           scene prompt as a `<t seconds> label` line aligned to the
//           video frame markers. Unwired (num_iports() < 3) disables
//           the audio path entirely.
//
//   oport0  FlexDataPayload per closed scene, with shape:
//             { "scene_index": uint,
//               "n_frames":    uint,
//               "first_ts_us": uint (when known),
//               "last_ts_us":  uint (when known),
//               "questions":   [ string, ... ],
//               "answers":     [ string, ... ] }
//
// Scene lifecycle:
//   - Frames accumulate into `_scene_images` while their timestamps
//     stay within `max_frame_gap_ms` of the previous frame.
//   - A frame whose timestamp gap exceeds `max_frame_gap_ms` closes
//     the current scene; the new frame starts the next one.
//   - When `idle_ticks_to_end` (default 2) consecutive triggers
//     arrive with no frames between them, the current scene closes.
//   - On scene close (only when at least one frame was captured): the
//     stage renders the chat prompt over the captured frames, prefills
//     once via `LoadedLanguageModel::prefill_multimodal`, branches the
//     base context per question (if multiple questions are configured),
//     and decodes — using `batched_step_pipelined` when 2+ branches
//     share the prefix. Answers are emitted on oport0.
//
// Config (FlexData object):
//   hf_dir              (string, required)   -- VLM HF-style dir
//   coreml_vision_path  (string, optional)   -- pre-converted CoreML
//                                               vision tower path (always
//                                               loaded with ComputeUnits
//                                               =All; leave unset to use
//                                               the GPU tower instead)
//   compute_dtype       (string, default "bf16")
//   language            (string, optional)       -- IETF UI/prompt locale
//                                                   for the built-in scene
//                                                   prompts (en-us | zh-cn
//                                                   | zh-tw). Empty inherits
//                                                   the session language at
//                                                   scene time.
//   page_tokens         (int,    default 512) -- KV tokens per page (the
//                                                page-pool cap is the LM
//                                                default, effectively
//                                                unbounded for VQA)
//   max_new_tokens      (int,    default 256)
//   vlm_input_width     (int,    default 0)  -- when both >0 AND the
//   vlm_input_height    (int,    default 0)     active tower runs on the
//                                                GPU (not CoreML), each
//                                                frame is GPU letterbox-
//                                                resampled to this fixed
//                                                resolution before encode
//                                                (0 = native).
//   max_frame_gap_ms    (int,    default 10000) -- inter-frame gap that
//                                                  closes a scene
//   idle_ticks_to_end   (int,    default 2)  -- consecutive trigger
//                                                 ticks with no frame
//                                                 that close a scene
//   max_frames_per_scene(int,    default 64) -- safety cap; closes
//                                                 the scene early when
//                                                 reached
//   catch_up_drop       (int,    default 0)  -- when > 0, at the start
//                                                 of each tick AND only
//                                                 when no scene is in
//                                                 progress, if the
//                                                 iport0 backlog
//                                                 exceeds this number,
//                                                 the stage skips
//                                                 catch_up_drop frames
//                                                 (advancing the read
//                                                 cursor) to close the
//                                                 lag toward the
//                                                 producer. Catch-up
//                                                 never runs mid-scene
//                                                 (would leave a gap
//                                                 in the scene
//                                                 timeline). 0 disables
//                                                 the catch-up logic
//                                                 entirely.
//   questions           (string|array<string>, required)
//   batched_decode      (bool,   default true)
//   disable_thinking    (bool,   optional)
//   video_fps           (real,   default 1.0) -- frames-per-second
//                                                 fallback cadence for
//                                                 the timestamp markers.
//                                                 Only used when
//                                                 timestamp_us is
//                                                 absent from the input
//                                                 sideband; when present
//                                                 (rtsp_vqa always sets
//                                                 it via video-to-rgb,
//                                                 preserved through
//                                                 temporal-decimation)
//                                                 the REAL per-frame
//                                                 timestamps drive the
//                                                 markers, so variable /
//                                                 decimated frame rates
//                                                 are handled exactly.
//                                                 With an MLX vision
//                                                 tower, frames are
//                                                 merged 2:1 and each
//                                                 merged token's marker
//                                                 is the AVERAGE of its
//                                                 pair's real times.
//   pipelined_decode    (bool,   default true) -- GPU-resident pipelined
//                                                  SINGLE-branch decode
//                                                  (metal). Batched decode
//                                                  always uses the shrinking
//                                                  path regardless.
//   sampler_temperature (real,   default 1.0)  -- flat decode-sampler
//   sampler_top_k       (int,    default 0)       knobs; all left at
//   sampler_top_p       (real,   default 1.0)     their defaults == greedy
//   sampler_min_p       (real,   default 0.0)     (argmax). Any non-default
//   sampler_repetition_penalty (real, default 1.0)  switches to sampled
//   sampler_presence_penalty   (real, default 0.0)  decoding.
//   sampler_seed        (uint,   default 0 = non-deterministic)
class RealtimeVqaStage final : public TypedStage<RealtimeVqaStage> {
public:
  static constexpr const char* kTypeName = "realtime-vqa";

  RealtimeVqaStage(const SessionContextIntf* session,
                   std::string               id,
                   std::vector<InEdge>       iports,
                   FlexData                  config);
  ~RealtimeVqaStage() override;

  Job initialize(RuntimeContext& ctx) override;
  Job process   (RuntimeContext& ctx) override;
  Job drain     (RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only accessors.
  const std::string&              hf_dir()    const noexcept { return _hf_dir; }
  const std::vector<std::string>& questions() const noexcept { return _questions; }
  const std::string& question_preamble() const noexcept
  { return _question_preamble; }
  // Resolved UI/prompt locale: the explicit "language" config (normalized)
  // when set, else "" meaning "inherit the session language at scene time"
  // (see effective_language_). Test-visible.
  const std::string& language() const noexcept { return _language; }
  int  max_frame_gap_ms() const noexcept { return _max_frame_gap_ms; }
  int  idle_ticks_to_end() const noexcept { return _idle_ticks_to_end; }
  bool prev_scene_recap() const noexcept { return _prev_scene_recap; }
  std::uint64_t scenes_closed() const noexcept { return _scenes_closed; }
  // Total iport2 audio-tagging beats released by the timestamp-gated drain
  // (across both the MLX and metal paths). Test-visible: a stuck audio
  // port (the bug this guards) leaves this at 0 while beats pile up.
  std::uint64_t audio_beats_released() const noexcept
  { return _audio_beats_released; }
  // Total audio soft tokens spliced into describe-prefixes this lifetime
  // (PCM iport2 + an LM audio encoder). 0 on the FlexData-tag path. Test-
  // visible so a PCM run can assert the audio was actually encoded + spliced.
  std::uint64_t audio_tokens_spliced() const noexcept
  { return _audio_tokens_spliced; }
#if defined(VPIPE_BUILD_APPLE_SILICON)
  // Size of the reusable per-question branch pool (metal path). Reserved
  // once on the first scene and reused thereafter; test-visible so a
  // multi-scene run can assert the pool stays put rather than churning.
  std::size_t branch_pool_size() const noexcept { return _branch_pool.size(); }
#endif

private:
  // ---- Path-agnostic helpers (compiled in BOTH builds) ----
  // These touch only scalars / strings / FlexData / LMDB / the chat
  // template -- never an MLX array or a metal SharedBuffer -- so the MLX
  // (process / close_scene_) and metal (m_process_ / m_close_scene_)
  // paths share ONE implementation. Keeping them here (not duplicated per
  // path) is what stops the two from drifting (e.g. the LMDB persistence
  // that the metal path was previously missing).

  // Render the scene-answer output beat as FlexData (scene_index,
  // n_frames, first/last_ts_us when has_ts, scene_description, questions,
  // answers).
  FlexData build_answer_(std::uint64_t                   scene_idx,
                         std::size_t                     n_frames,
                         std::uint64_t                   first_ts_us,
                         std::uint64_t                   last_ts_us,
                         bool                            has_ts,
                         const std::string&              scene_description,
                         const std::vector<std::string>& answers) const;

  // Build the user-turn text for question i: the (optional) per-question
  // preamble prepended to the question. Shared by both decode paths so the
  // answer-format instruction can't drift between MLX and metal.
  std::string question_prompt_(std::size_t i) const
  {
    if (_question_preamble.empty()) { return _questions[i]; }
    return _question_preamble + "\n\n" + _questions[i];
  }

  // The effective UI/prompt locale used to select the built-in scene
  // prompts (describe / preamble fragments / audio-interpret / the
  // question-preamble default). Returns the explicit "language" config
  // (already normalized) when set, else the session's current language
  // normalized, else the default locale. Resolved at scene time so an
  // inheriting stage tracks a later session set_language().
  std::string effective_language_() const;

  // Assemble the user-turn text preamble shared by both prompt builders:
  // the local date/time anchor, an optional prior-scene recap, and the
  // iport2 audio sound timeline (<X.Y seconds> markers, scene-relative to
  // base_ts_us). `audio` is the drained non-Silence events.
  std::string build_preamble_(
      std::uint64_t                                            first_ts_us,
      const std::string&                                       prev_desc,
      const std::vector<std::pair<std::uint64_t, std::string>>& audio,
      std::uint64_t                                            base_ts_us,
      bool                                                     audio_wired) const;

  // The prior scene's description to carry into THIS scene's describe
  // recap -- or empty when it should not be carried. Empty unless the
  // recap feature is enabled (_prev_scene_recap) AND the two scenes are
  // temporally continuous: the gap from the previous scene's last frame
  // (prev_last_ts_us) to this scene's first frame (this_first_ts_us) is
  // below the scene-splitting gap (_max_frame_gap_ms). A new, unrelated
  // scene gets an empty recap so the model can't echo a stale frame.
  std::string prev_recap_(const std::string& prev_desc,
                          std::uint64_t      prev_last_ts_us,
                          std::uint64_t      this_first_ts_us) const;

  // Lazily verify (or seed) the <camera>-video-questions sub-db; append a
  // new questions epoch keyed by ts_us when the stored set differs. No-op
  // once a camera is checked this lifetime. Best-effort (errors logged).
  void sync_questions_record_(const std::string& camera_name,
                              std::uint64_t      ts_us);

  // Persist the just-closed scene's QA bundle to <camera>-video-qa keyed
  // by the scene's first-frame time code. Errors logged and swallowed.
  void log_scene_qa_(const std::string&              camera_name,
                     std::uint64_t                   start_utc_us,
                     std::uint64_t                   end_utc_us,
                     const std::string&              description,
                     const std::vector<std::string>& answers);

  // Drain audio-tagging classifications from iport2 with ts <
  // boundary_ts_us, releasing them (relieves the upstream backpressure)
  // and weaving non-Silence ones into `scene_audio` when `scene_active`
  // and ts >= scene_first_ts_us. Shared by both paths (the caller passes
  // its own scene-active flag / first-frame ts / sound-event vector).
  Job consume_audio_(
      RuntimeContext& ctx, std::uint64_t boundary_ts_us, bool scene_active,
      std::uint64_t scene_first_ts_us,
      std::vector<std::pair<std::uint64_t, std::string>>& scene_audio);


  // ---- Config attributes; defaults live in kSpec.attrs and are read
  // in the constructor via attr_*. Declarations carry no non-zero
  // default. ----
  std::string              _hf_dir;
  std::string              _models_db;
  std::string              _coreml_vision_path;
  std::string              _compute_dtype;
  int                      _page_tokens{};
  int                      _max_new_tokens{};
  // Optional fixed VLM input resolution. When both > 0 AND the active
  // vision tower runs on the GPU (not the CoreML tower), each frame is
  // letterbox-resampled to (_vlm_in_w x _vlm_in_h) on the GPU before
  // encoding. 0 = pass frames at their native resolution.
  int                      _vlm_in_w{};
  int                      _vlm_in_h{};
  // Per-frame soft-token budget for the vision tower. 0 = the encoder's
  // still/default image budget (best detail; the right default for the
  // sparse-frame realtime regime). A positive value caps tokens/frame --
  // trade detail for prefill speed. Mainly affects Gemma-4 (whose dense-
  // video budget of ~64 tok/frame downsampled wide camera frames too
  // coarsely, causing vague / hallucinated descriptions).
  int                      _vlm_max_soft_tokens{};
  int                      _max_frame_gap_ms{};
  int                      _idle_ticks_to_end{};
  int                      _max_frames_per_scene{};
  // Catch-up policy for slow-consumer-vs-fast-producer scenarios.
  // 0 = disabled (default). When > 0, at the START of each process()
  // call (after consuming the chrono tick), if iport0's backlog
  // exceeds `_catch_up_drop`, the stage skips that many frames via
  // release_read so it lands `_catch_up_drop` positions closer to
  // the producer's write pointer. Skipped frames do NOT contribute
  // to any scene -- pick the value relative to acceptable scene
  // disruption.
  int                      _catch_up_drop{};
  bool                     _batched_decode{};
  // GPU-resident pipelined SINGLE-branch decode (metal); default true. It
  // overlaps host/GPU per token, a clear win for a lone branch. Batched
  // (multi-question) decode never pipelines -- the pipelined batched path
  // is constant-N and wastes work on staggered answers -- so this flag
  // governs single decode only (see m_decode_ vs m_decode_batched_).
  bool                     _pipelined_decode{};
  std::optional<bool>      _disable_thinking;
  std::vector<std::string> _questions;
  // Instruction prepended to EVERY per-question user turn (after the scene
  // description, inside each branch). Steers answer format -- e.g. "answer
  // yes/no/unknown first". Empty disables it. Config key "question_preamble".
  std::string              _question_preamble;
  // Resolved UI/prompt locale (config key "language"). "" means "inherit
  // the session language"; a non-empty value is a normalized supported tag
  // (en-us / zh-cn / zh-tw) that pins the built-in scene prompts. Read via
  // effective_language_() at scene time.
  std::string              _language;
  // When true (default), the previous scene's description is carried into
  // the next scene's describe prompt as a recap -- but ONLY when the two
  // scenes are temporally CONTINUOUS (small inter-scene gap). A new,
  // unrelated scene must not inherit the prior description, or the model
  // echoes it verbatim and the output locks onto a stale frame. Config
  // key "prev_scene_recap"; set false to disable the recap entirely.
  bool                     _prev_scene_recap{};
  float                    _video_fps{};


  // ---- No-MLX metal path (self-contained; MLX-ON build is unchanged) --
#if defined(VPIPE_BUILD_APPLE_SILICON)
  std::shared_ptr<genai::LoadedLanguageModel> _lm;
  std::unique_ptr<genai::ChatTemplate>        _chat_tpl;
  genai::MetalQwenVisionEncoder*              _mvis = nullptr;  // borrowed
  // Gemma-4 e4b metal ViT tower (borrowed; native-f16 SharedBuffer rows).
  // Mutually exclusive with _mvis/_mguni by family.
  genai::MetalGemma4VisionEncoder*            _mgvis = nullptr; // borrowed
  // Gemma-4 e4b metal USM audio encoder (borrowed; host-f32 rows). Encodes
  // scene PCM to soft tokens when the iport2 input is PCM (vs audio tags).
  genai::MetalGemma4AudioEncoder*             _mgaud = nullptr; // borrowed
  // Gemma-4-12B "unified" encoder-less shallow embedder (borrowed; host-f32
  // rows). Mutually exclusive with _mvis by family.
  genai::Gemma4UnifiedEmbedder*               _mguni = nullptr; // borrowed
  // Optional CoreML vision tower (host-f32 + metal-compute preproc).
  // Owned when coreml_vision_path is set; takes priority over _mvis.
  std::unique_ptr<genai::CoreMLVisionEncoder> _m_coreml;
  genai::SamplerParams                        _sampler_params;
  // Reusable per-question branch pool (reserve once, rebranch per scene):
  // avoids allocating + freeing N branch contexts (their KV pages + SSM/conv
  // buffers) every scene. Declared after _lm so it is destroyed (its pooled
  // contexts released) before the LM. Empty until the first scene reserves it.
  std::vector<genai::LoadedLanguageModel::Context> _branch_pool;
  // One encoded frame: owned host-f32 embeddings + POST-merger token
  // grid (normalised per encoder at encode time so m_close_scene_ stays
  // encoder-agnostic).
  struct MImg {
    metal_compute::SharedBuffer embeddings;   // native f16 [n_tokens, H]
    // gemma4_unified embedder output is host f32 [n_tokens * H] (spliced via
    // the TokenRef embeddings_host path); `embeddings` stays empty then.
    std::vector<float>          embeddings_host;
    int n_tokens = 0;
    int mh = 0;   // post-merger grid height
    int mw = 0;   // post-merger grid width
  };
  std::vector<MImg>                         _m_imgs;
  std::vector<std::uint64_t>                _m_frame_ts_us;
  std::uint64_t                            _m_first_ts_us = 0;
  std::uint64_t                            _m_last_ts_us  = 0;
  bool                                     _m_has_ts      = false;
  std::uint64_t                            _m_last_recv_ts_us = 0;
  std::string                             _m_prev_desc;
  // Last (real) frame timestamp of the previously-closed scene; with this
  // scene's first timestamp it gates the _m_prev_desc recap (continuity).
  std::uint64_t                           _m_prev_last_ts_us = 0;
  bool _m_vision_warned = false;
  bool _m_tpl_warned    = false;

  // Temporal-pair buffer (CoreML video export, image0+image1): the first
  // frame of a 2-frame merge, held until its partner arrives. Qwen3-VL
  // fuses temporal_patch_size (=2) consecutive frames into one token
  // grid. Mirrors the MLX path's _pending_* state. Unused (always empty)
  // for single-image towers, so they behave exactly as before.
  std::vector<std::uint8_t>                _m_pending_bytes;
  int                                      _m_pending_h     = 0;
  int                                      _m_pending_w     = 0;
  std::uint64_t                            _m_pending_ts_us = 0;
  bool                                     _m_has_pending   = false;

  // Active once a scene holds at least one encoded grid OR a buffered
  // first-of-pair frame, so gap / idle / drain boundaries flush the
  // lone frame instead of silently dropping it (matches scene_active_).
  bool m_scene_active_() const noexcept
  { return !_m_imgs.empty() || _m_has_pending; }
  bool m_encode_frame_(const std::uint8_t* rgb, int H, int W,
                       std::uint64_t ts_us, bool ts_present,
                       const metal_compute::SharedBuffer* src_buf = nullptr);
  // Validate, log, and append one encoded grid (+ its marker timestamp)
  // to the in-progress scene. Returns false (and warns) on a 0-token
  // encode. Shared by the single-frame and temporal-pair paths. The log
  // line mirrors the MLX path's scene-token message (real_ts_us is the
  // frame's raw timestamp for the human-readable tag; ts_present=false
  // marks a synthesised cadence timestamp).
  bool m_append_mimg_(MImg&& m, std::uint64_t marker_ts_us,
                      std::uint64_t real_ts_us, bool ts_present,
                      const char* how, int W, int H, double enc_s);
  // Flush a buffered first-of-pair frame at scene close (odd frame
  // count): encode_pair_host(f, f) replicates it across the temporal
  // patch, mirroring the reference last-frame padding.
  void m_flush_pending_();
  void m_reset_scene_();
  Job  m_process_(RuntimeContext& ctx);
  Job  m_close_scene_(RuntimeContext& ctx);
  std::string m_decode_(genai::LoadedLanguageModel::Context& ctx,
                        const genai::SamplerParams& sp);
  // Interpret a scene's audio (host-f32 encoder rows) to a short text
  // phrase in ISOLATION -- a dedicated [audio block + "what sound?"]
  // prefill+decode where the audio is salient (no visual context to drown
  // it). The phrase is added to the sound text timeline so the describe
  // and every question can read it. Returns "" on failure/no audio.
  std::string m_interpret_audio_(const std::vector<float>& audio_rows,
                                 int n_audio_tokens);
  // Batched (N-branch parallel) decode of already-prefilled question
  // branches on the metal backend. Dispatches to the GPU-resident
  // pipelined path (m_..._pipelined_) when the backend supports it, else
  // the synchronous loop (m_..._sync_). Returns answers[k] for children[k].
  std::vector<std::string> m_decode_batched_(
      std::vector<genai::LoadedLanguageModel::Context>& children);
  // Synchronous batched decode: each step batches the weight-bound
  // projections across the still-active branches (host samples each row's
  // [vocab] logits); branches dropped as they stop. Branches need NOT be at
  // the same seq_len.
  std::vector<std::string> m_decode_batched_sync_(
      std::vector<genai::LoadedLanguageModel::Context>& children);
  // Pipelined batched decode: GPU per-branch sampler keeps tokens on-device
  // + event-chained per-token command buffers overlap the host stop-check
  // with the next forward (m_bdecode_*). Constant-N. Greedy is token-exact
  // vs the synchronous path.
  std::vector<std::string> m_decode_batched_pipelined_(
      std::vector<genai::LoadedLanguageModel::Context>& children);
#endif

  // ---- Trigger / scene-boundary bookkeeping ----
  int                          _consec_idle_ticks = 0;
  bool                         _had_frame_this_tick = false;
  std::uint64_t                _scenes_closed  = 0;
  std::uint64_t                _next_scene_idx = 0;
  std::uint64_t                _audio_beats_released = 0;
  std::uint64_t                _audio_tokens_spliced = 0;

  // ---- Shared scene/audio/persistence state (BOTH paths) ----
  // camera_name latched from the first iport0 frame of the scene
  // ("unknown" when absent); namespaces the LMDB sub-dbs. Cleared per
  // scene by reset_scene_ / m_reset_scene_.
  std::string                  _scene_camera_name;
  // Cameras whose <camera>-video-questions epoch we've already verified
  // this stage lifetime (sync_questions_record_ dedup).
  std::unordered_set<std::string> _questions_checked_cameras;
  // Non-Silence sound events for the current scene (window ts_us, top
  // label), timestamp order, consecutive duplicates collapsed; woven into
  // the prompt, cleared per scene.
  std::vector<std::pair<std::uint64_t, std::string>> _scene_audio;
  // Raw scene PCM (mono f32), accumulated when iport2 carries PCM
  // TensorBeats (a pcm stage) instead of audio-tag FlexData AND the LM has
  // an audio encoder (Gemma-4 unified). At scene close it is encoded to
  // soft tokens and spliced into the describe-prefix as an inline audio
  // block (time-aligned with the video frames). Cleared per scene.
  std::vector<float>           _scene_pcm;
  std::uint64_t                _scene_pcm_first_ts_us = 0;
  int                          _scene_pcm_sr = 16000;
  // Latest received REAL frame ts -- the iport2 drain boundary ("prior to
  // its image data"). Monotonic ACROSS scenes (not reset per scene).
  std::uint64_t                _last_image_ts_us = 0;
  // iport2 reached EOS -- stop pulling (audio is auxiliary).
  bool                         _audio_eos = false;

#ifdef VPIPE_BUILD_APPLE_SILICON
  // ---- GPU VLM-input resampler (both builds) ----
  // When _vlm_in_w/_vlm_in_h are set AND the active tower runs on the GPU
  // (not the CoreML tower), letterbox-resample each frame to that fixed
  // resolution before encoding. Reuses the metal-compute letterbox
  // kernel; the src/dst Shared buffers are reused across frames.
  // Returns `rgb` unchanged (out_h=H,out_w=W) when no resample applies.
  const std::uint8_t* resample_frame_(const std::uint8_t* rgb, int H, int W,
                                      int* out_h, int* out_w);
  std::unique_ptr<ExternalStorageHandle> _resample_src;
  std::unique_ptr<ExternalStorageHandle> _resample_dst;
  bool _resample_warned = false;
#endif
};

}

#endif

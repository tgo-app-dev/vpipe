#ifndef VPIPE_STAGES_VISUAL_QA_STAGE_H
#define VPIPE_STAGES_VISUAL_QA_STAGE_H

#include "apple-silicon/aligned-allocator.h"
#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"


// No-MLX metal path: drive image-VQA on the metal-compute backend. These
// headers carry no mlx::core types.
#if defined(VPIPE_BUILD_APPLE_SILICON)
#include "generative-models/chat-template.h"
#include "generative-models/shared/coreml-vision-encoder.h"
#include "generative-models/loaded-language-model.h"
#include "generative-models/gemma4/metal-gemma4-vision.h"
#include "generative-models/qwen3/metal-qwen-vision.h"
#include "generative-models/sampler.h"
#endif

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace vpipe::genai {
class LoadedLanguageModel;
class VisionEncoder;
class Gemma4UnifiedEmbedder;
}

namespace vpipe {

// Sink stage: 1 input (TensorBeat RGB images), 0 outputs. Loads a
// vision-language model once in initialize(), then consumes
// TensorBeats on in-port 0 num_images at a time. Within a round:
//
//   * Each TensorBeat is acquired one at a time via ctx.read(0). The
//     acquire moves the slot out of the OportBuffer, advancing the
//     read cursor and unblocking any backpressured producer -- so
//     image bytes don't pile up while later beats in the same round
//     are still in flight.
//   * The image is passed straight through the loaded LM's
//     VisionEncoder (which is itself a stub today; the orchestration
//     is exercised end-to-end and ready for the real ViT body when it
//     lands). The encoded image-token embeddings are stashed for the
//     round.
//   * Once `num_images` encoded images are buffered, the stage asks
//     each configured question against the same image prefix and
//     surfaces each answer to the user via the session UI delegate
//     (session()->info), same as text-chat-stage.
//
// After a round closes the stage clears its per-round image buffer
// and starts the next round on the next incoming beat. EOS on iport
// 0 ends the stage (after flushing any partial round's remaining
// images through the encoder so the producer can release those
// OportBuffer slots cleanly).
//
// Configuration (FlexData object on the 4th constructor parameter):
//   hf_dir          (string, required)         -- VLM HF-style dir
//   coreml_vision_path
//                   (string, optional)         -- path to a pre-
//                                                 converted CoreML
//                                                 .mlpackage /
//                                                 .mlmodelc for the
//                                                 vision tower. When
//                                                 set, the stage runs
//                                                 the vision forward
//                                                 pass through CoreML
//                                                 (BGRA8888 ImageType
//                                                 input, fp16 fixed-
//                                                 size output)
//                                                 instead of the LM's
//                                                 bundled MLX tower.
//                                                 Preprocessing reuses
//                                                 the same Metal
//                                                 letterbox + RGB->BGRA
//                                                 GPU kernel
//                                                 yolo-detection-stage
//                                                 uses for ImageType
//                                                 CoreML models. Drops
//                                                 back to the MLX
//                                                 tower on load
//                                                 failure.
//   coreml_compute_units
//                   (int,    default 2)        -- CoreML compute-units
//                                                 selector for the
//                                                 vision model:
//                                                 0=CPUOnly, 1=CPU+GPU,
//                                                 2=All, 3=CPU+ANE.
//                                                 Only meaningful when
//                                                 coreml_vision_path
//                                                 is set.
//   compute_dtype   (string, default "bf16")   -- "bf16" | "f16" | "f32"
//   page_tokens     (int,    default 16)       -- ContextManager page
//                                                 size for the LLM K/V
//   max_pages       (int,    default 256)      -- per-LM page pool
//                                                 capacity
//   max_new_tokens  (int,    default 256)      -- per-question gen
//                                                 budget
//   num_images      (int,    default 1)        -- images per Q&A round
//                                                 (or frames per round
//                                                 in video mode)
//   questions       (string or array<string>, required) -- one or
//                                                 more questions
//                                                 asked of every round
//   video           (object, default unset)    -- enable video mode.
//     enabled       (bool,   default false)    -- if true, each
//                                                 incoming TensorBeat
//                                                 is treated as a
//                                                 video frame; the
//                                                 chat prompt
//                                                 interleaves
//                                                 `<{t:.1f} seconds>`
//                                                 markers between
//                                                 frames per the
//                                                 Qwen3-VL processor
//                                                 convention.
//     fps           (real,   default 1.0)      -- frames per second
//                                                 at which incoming
//                                                 TensorBeats were
//                                                 SAMPLED (the source
//                                                 sampling rate). To
//                                                 match the Qwen3-VL
//                                                 video path the stage
//                                                 merges temporal_patch
//                                                 _size (=2) consecutive
//                                                 frames into ONE token
//                                                 grid, so the effective
//                                                 token rate is fps/2.
//                                                 Each merged token's
//                                                 timestamp is the
//                                                 average of its two
//                                                 source frames' times,
//                                                 i.e. (2i+0.5)/fps for
//                                                 merged token i --
//                                                 matching the reference
//                                                 _calculate_timestamps
//                                                 (frame-pair averaging).
//   pause_ms_between_rounds (int, default 0)   -- sleep this many ms
//                                                 after each Q&A round
//                                                 finishes. Useful for
//                                                 multi-round benchmark
//                                                 scripts that want to
//                                                 dodge GPU thermal
//                                                 throttling across
//                                                 back-to-back rounds.
//   batched_decode (bool, default true)        -- when 2+ questions
//                                                 share an image prefix,
//                                                 decode their answers
//                                                 in parallel via
//                                                 LoadedLanguageModel
//                                                 ::batched_step_pipelined
//                                                 so the per-layer
//                                                 projection + MLP +
//                                                 lm_head matmuls
//                                                 amortise their weight
//                                                 reads across all
//                                                 active branches. Set
//                                                 to false to A/B
//                                                 against the legacy
//                                                 per-branch serial
//                                                 decode loop.
//   pre_image_prompt  (string, default "")     -- text inserted into
//                                                 the user turn BEFORE
//                                                 the image block, e.g.
//                                                 a task framing
//                                                 instruction the model
//                                                 should read prior to
//                                                 seeing the image.
//   post_image_prompt (string, default "")     -- text inserted into
//                                                 the user turn AFTER
//                                                 the image block (and
//                                                 before each question
//                                                 or, when
//                                                 decode_after_post_image
//                                                 is set, before the
//                                                 assistant header).
//                                                 Used to attach a
//                                                 grounding instruction
//                                                 or chain-of-thought
//                                                 cue to the shared
//                                                 prefix.
//   decode_after_post_image (bool, default false) -- when true the
//                                                 chat template closes
//                                                 the user turn right
//                                                 after the post-image
//                                                 prompt + opens the
//                                                 assistant turn; the
//                                                 stage decodes the
//                                                 model's reply against
//                                                 that prefix until it
//                                                 emits a stop token,
//                                                 commits the assistant
//                                                 close to the base
//                                                 context, and only
//                                                 then branches into
//                                                 the per-question
//                                                 sub-contexts. Useful
//                                                 for "look first,
//                                                 then answer" patterns
//                                                 where the model
//                                                 should produce a
//                                                 grounding caption or
//                                                 reasoning trace
//                                                 before per-question
//                                                 fanout.
//   disable_thinking  (bool, optional)         -- override the chat
//                                                 template's family-
//                                                 default thinking
//                                                 preamble. true forces
//                                                 the thinking-OFF
//                                                 sequence
//                                                 ("<think>\n\n</think>
//                                                 \n\n"); false forces
//                                                 thinking-ON
//                                                 ("<think>\n"). Absent
//                                                 = use family default
//                                                 (Qwen3-VL: ON;
//                                                 Qwen3 text-only:
//                                                 OFF). Ignored on
//                                                 families without a
//                                                 thinking concept.
//
// Only available on apple-silicon builds (the metal-compute LM
// subsystem). On other builds the constructor logs an error through
// session() and the stage does nothing on each beat.
class VisualQaStage final : public TypedStage<VisualQaStage> {
public:
  static constexpr const char* kTypeName = "visual-qa";

  VisualQaStage(const SessionContextIntf* session,
                std::string               id,
                std::vector<InEdge>       iports,
                FlexData                  config);
  ~VisualQaStage() override;

  Job initialize(RuntimeContext& ctx) override;
  Job process   (RuntimeContext& ctx) override;
  Job drain     (RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only accessors.
  const std::string& hf_dir()         const noexcept { return _hf_dir; }
  int                num_images()     const noexcept { return _num_images; }
  int                max_new_tokens() const noexcept
  { return _max_new_tokens; }
  const std::vector<std::string>&
  questions() const noexcept { return _questions; }

private:

  // Config attributes; flat-key defaults live in kSpec.attrs (read via
  // attr_*). The nested video.fps default (1.0) has no flat ConfigKey
  // form and is seeded once at the top of the constructor. Declarations
  // carry no non-zero default.
  std::string              _hf_dir;
  std::string              _models_db;
  // Optional path to a pre-converted CoreML .mlpackage / .mlmodelc
  // for the vision tower. When set, the stage runs the vision
  // forward pass through CoreML (using the Metal letterbox kernel
  // for input scaling + RGB->BGRA packing) instead of the MLX tower
  // bundled in the LM. The output is dropped into the same
  // TokenMuxer path so the LM is unaware of which backend produced
  // the image-token embeddings.
  std::string              _coreml_vision_path;
  int                      _coreml_compute_units{};
  std::string              _compute_dtype;
  int                      _page_tokens{};
  std::uint32_t            _max_pages{};
  int                      _max_new_tokens{};
  int                      _num_images{};
  int                      _pause_ms_between_rounds{};
  bool                     _batched_decode{};
  bool                     _i8_prefill{};   // LOSSY int8 prefill (opt-in)
  std::string              _pre_image_prompt;
  std::string              _post_image_prompt;
  bool                     _decode_after_post_image{};
  // Unset => use family default; set => override.
  std::optional<bool>      _disable_thinking;
  std::vector<std::string> _questions;
  bool                     _video_enabled = false;
  float                    _video_fps{};   // seeded to 1.0 in the ctor


  // ---- No-MLX metal path (self-contained; MLX-ON build is unchanged) --
  // Image mode only: per-frame metal vision encode + mROPE multimodal
  // prefill + per-question decode (greedy or host-sampled). Video 2:1
  // temporal merge and batched/deepstack decode stay on the MLX path.
#if defined(VPIPE_BUILD_APPLE_SILICON)
  std::shared_ptr<genai::LoadedLanguageModel> _lm;
  std::unique_ptr<genai::ChatTemplate>        _chat_tpl;
  genai::MetalQwenVisionEncoder*              _mvis = nullptr;  // borrowed
  // Gemma-4 metal vision tower (borrowed; mutually exclusive with _mvis
  // by family). Reports the POST-merger grid directly (no /S).
  genai::MetalGemma4VisionEncoder*            _mgvis = nullptr; // borrowed
  // Gemma-4-12B "unified" encoder-less shallow embedder (borrowed; host-f32
  // rows). Mutually exclusive with _mgvis/_mvis by family.
  genai::Gemma4UnifiedEmbedder*               _mguni = nullptr; // borrowed
  // Optional CoreML vision tower (host-f32 + metal-compute preproc).
  // Owned when coreml_vision_path is set; takes priority over _mvis.
  std::unique_ptr<genai::CoreMLVisionEncoder> _m_coreml;
  genai::SamplerParams                        _sampler_params;
  // One encoded image: owned host-f32 embeddings + POST-merger token
  // grid (normalised here so m_run_round_ is encoder-agnostic -- the
  // metal tower reports pre-merger patches, the CoreML tower reports
  // the post-merger grid directly).
  struct MImg {
    metal_compute::SharedBuffer embeddings;   // native f16 [n_tokens, H]
    // gemma4_unified embedder output is host f32 [n_tokens * H] (spliced via
    // the TokenRef embeddings_host path); `embeddings` stays empty then.
    std::vector<float>          embeddings_host;
    int n_tokens = 0;
    int mh = 0;   // post-merger grid height
    int mw = 0;   // post-merger grid width
  };
  std::vector<MImg> _m_imgs;
  bool _m_vision_warned = false;
  bool _m_tpl_warned    = false;
  bool _m_video_warned  = false;
  // Run one Q&A round against the metal-encoded images in _m_imgs.
  void        m_run_round_();
  std::string m_decode_(genai::LoadedLanguageModel::Context& ctx,
                        const genai::SamplerParams& sp);
#endif

  // Beats-this-round counter. Independent of `_round_images.size()`
  // so a stubbed encoder still advances the round and the read
  // pointer stays in sync with what we've consumed off the iport.
  int _seen_this_round = 0;
};

}

#endif

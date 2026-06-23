#ifndef VPIPE_STAGES_TEXT_CHAT_STAGE_H
#define VPIPE_STAGES_TEXT_CHAT_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

// The language-model subsystem (chat template, LoadedLanguageModel,
// sampler) is MLX-free and builds on the VPIPE_BUILD_APPLE_SILICON axis;
// generation runs on the metal-compute backend when MLX is off.
#ifdef VPIPE_BUILD_APPLE_SILICON
#include "generative-models/chat-template.h"
#include "generative-models/loaded-language-model.h"
#include "generative-models/sampler.h"
#endif

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace vpipe::genai {
class LoadedLanguageModel;
}

namespace vpipe {

// Stage: 1 input, 1 output. The model is loaded once in
// initialize() and a single K/V chat context is acquired there too;
// every subsequent beat on in-port 0 appends a user turn to that
// context, runs chat-templated inference using the per-model
// ChatTemplate the LM provides (Llama-3 headers, ChatML, etc.),
// and streams the assistant's tokens to the user as they are produced
// via the session UI delegate's live text stream
// (session()->open_text_stream()). The
// context carries the conversation history across beats, so the
// model "remembers" prior turns until the user types `/clear`,
// which resets the K/V cache and re-seeds the next turn's session-
// start tokens. EOS on in-port 0 ends the stage.
//
// Per assistant turn the stage emits a FlexData object on out-port 0
// with keys:
//   text        (string) -- the full assistant response text
//   prefill_ms  (int)    -- prefill wall time in milliseconds
//   decode_ms   (int)    -- decode wall time in milliseconds
//   ctx_pos     (int)    -- K/V context length after this turn
// The oport is unconditional; downstream consumers are optional --
// the runtime drops writes when no cursor is attached. The intended
// consumer is a feedback-rx + feedback-tx pair driving the input
// stage in an interactive chat loop.
//
// Configuration (FlexData object on the 4th constructor parameter):
//   hf_dir         (string, required)         -- path to a Hugging-
//                                                Face style Llama
//                                                3.x directory.
//   compute_dtype  (string, default "bf16")   -- "bf16" | "f16" | "f32"
//   page_tokens    (int,    default 16)       -- ContextManager page
//                                                size.
//   max_pages      (int,    default 256)      -- per-LM page pool
//                                                capacity (caps the
//                                                K/V cache size).
//   max_new_tokens (int,    default 128)      -- per-turn generation
//                                                budget.
//   disable_thinking (bool, optional)         -- override the chat
//                                                template's family-
//                                                default thinking
//                                                mode. true forces the
//                                                thinking-OFF preamble
//                                                ("<think>\n\n</think>
//                                                \n\n"); false forces
//                                                thinking-ON
//                                                ("<think>\n"). Absent
//                                                = use family default
//                                                (Qwen3 text-only:
//                                                OFF; Qwen3-VL: ON;
//                                                Llama-3 / ChatML
//                                                ignore this flag).
//   sampler_temperature (real, default 1.0)  -- flat decode-sampler
//   sampler_top_k       (int,  default 0)       knobs; all at their
//   sampler_top_p       (real, default 1.0)     defaults == greedy
//   sampler_min_p       (real, default 0.0)     (argmax). Any non-
//   sampler_repetition_penalty (real, default 1.0) default switches to
//   sampler_presence_penalty   (real, default 0.0) sampled decoding.
//   sampler_seed        (uint, default 0 = non-deterministic)
//   mtp              (bool,   default true)    -- use the MTP speculative-
//                                                decode head when the
//                                                loaded model carries one
//                                                (Qwen3.5-OptiQ / GGUF
//                                                NextN; metal path only).
//                                                Token-exact, so perf-only;
//                                                false forces the standard
//                                                pdecode loop.
//
// Available on VPIPE_BUILD_APPLE_SILICON builds (both the MLX and the
// no-MLX metal path, which is the default). On other builds the
// constructor logs an error through session() and the stage does
// nothing on each beat.
class TextChatStage final : public TypedStage<TextChatStage> {
public:
  static constexpr const char* kTypeName = "text-chat";

  TextChatStage(const SessionContextIntf* session,
                std::string               id,
                std::vector<InEdge>       iports,
                FlexData                  config);

  ~TextChatStage() override;

  Job initialize(RuntimeContext& ctx) override;
  Job process   (RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only accessors.
  const std::string& hf_dir() const noexcept { return _hf_dir; }
  int max_new_tokens() const noexcept { return _max_new_tokens; }
#if defined(VPIPE_BUILD_APPLE_SILICON)
  // The flattened `mtp` opt-out (metal path only). Test-visible.
  bool mtp_enabled() const noexcept { return _mtp_enabled; }
#endif

private:
  // Config attributes; defaults live in kSpec.attrs and are read in the
  // constructor via attr_*. Declarations carry no non-zero default.
  std::string _hf_dir;
  std::string _models_db;
  std::string _compute_dtype;
  int         _page_tokens{};
  std::uint32_t _max_pages{};
  int         _max_new_tokens{};
  // Unset => use family default; set => override.
  std::optional<bool> _disable_thinking;
  // Loaded lazily in initialize(); cleared on shutdown.
  std::shared_ptr<genai::LoadedLanguageModel> _lm;
#ifdef VPIPE_BUILD_APPLE_SILICON
  genai::SamplerParams _sampler_params;
  // MTP speculative-decode opt-out (config "mtp", default true; read in the
  // ctor). Gates the metal MTP fast path; token-exact, so it only changes
  // decode speed. The MLX path has no MTP head, hence no-MLX only.
  bool _mtp_enabled{};
  // MTP prefix-seed preference (config "mtp_prefix_seed", default true; read in
  // the ctor, applied to the LM at launch). Decode- vs prefill-throughput.
  bool _mtp_prefix_seed{};
  // Stage-local chat template built in initialize() with the
  // disable_thinking override applied (when set). Falls back to the
  // LM's family-default template when _disable_thinking is unset.
  // Declared AFTER _lm so it destructs first (it borrows the LM's
  // tokenizer).
  std::unique_ptr<genai::ChatTemplate> _chat_tpl;
  // Persistent K/V context across beats; reset on `/clear`. Declared
  // AFTER _lm so it destructs first — its release path dispatches
  // through the LM's MlxRuntime, which must still be alive.
  genai::LoadedLanguageModel::Context _chat_ctx;
#endif
  // True once the BOS token has been prefilled into _chat_ctx. The
  // first turn pushes `<|begin_of_text|>` ahead of the user header;
  // subsequent turns skip it because the cache already starts with
  // BOS. `/clear` flips this back to false.
  bool _seeded = false;
};

}

#endif

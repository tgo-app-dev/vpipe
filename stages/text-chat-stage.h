#ifndef VPIPE_STAGES_TEXT_CHAT_STAGE_H
#define VPIPE_STAGES_TEXT_CHAT_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "generative-models/shared/mcp/file-sandbox.h"
#include "generative-models/shared/mcp/mcp-tools.h"
#include "generative-models/shared/mcp/url-fetch.h"

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
// A user turn may embed image/audio attachments as media-line markers
// (common/media-line.h; produced by text-input's `media` mode or typed
// by hand). Markers are processed BEFORE the tokenizer: each item is
// decoded with FFmpeg to the encoder's native input, encoded through
// the model's own vision/audio tower, and its embeddings are spliced
// at the marker position via prefill_multimodal_metal. A modality the
// loaded model does not provide (no tower / no pad token) warns and
// drops that attachment, keeping the surrounding text.
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
// Out-port 1 ("stream") emits the SAME reply progressively, as a series
// of FlexData objects with keys:
//   text            (string) -- one chunk of the reply
//   end_of_response (bool)   -- true only on the final chunk of the turn
// A chunk is flushed roughly every ~20 words at a sentence/clause
// punctuation boundary (English + Chinese; see common/text-stream-chunk.h),
// so a streaming consumer -- notably text-to-speech, whose barge-in
// respects end_of_response -- can start before the turn finishes. Like
// out-port 0 it is unconditional and downstream-optional. When
// `stream_answer_only` is set the reasoning (<think>) and tool-call blocks
// are folded OUT of this port so a speaking consumer voices only the
// answer; out-port 0 still carries the full text.
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
//   enable_tools   (bool,   default false)    -- MCP tool calling. When
//                                                set and the chat
//                                                template supports tools
//                                                (ChatML / Qwen), the
//                                                first prompt advertises
//                                                the built-in tools
//                                                (get_current_time) and
//                                                the stage runs any
//                                                <tool_call> the model
//                                                emits, feeding the
//                                                results back for another
//                                                decode round.
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
  bool tools_enabled() const noexcept { return _enable_tools; }
  bool python_tool_enabled() const noexcept { return _enable_python_tool; }
  bool file_tools_enabled() const noexcept { return _enable_file_tools; }
  bool shell_tool_enabled() const noexcept { return _enable_shell_tool; }
  bool web_tools_enabled() const noexcept { return _enable_web_tools; }
  bool allow_system_temp() const noexcept { return _allow_system_temp; }
  bool stream_answer_only() const noexcept { return _stream_answer_only; }
  const std::string& file_sandbox_mode() const noexcept
  { return _file_sandbox_mode; }
  const std::string& file_sandbox_root() const noexcept
  { return _file_sandbox_root; }
  // Test-only: run the file-sandbox root resolution (normally done inside
  // initialize(), after the model load, which the test skips) and return
  // the resolved root.
  const std::string& resolve_file_sandbox_for_test()
  { setup_file_sandbox_(); return _file_sandbox_root; }
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
  // Enable MCP tool calling (config "enable_tools", default false). When
  // set and the model's chat template renders the tool scaffold, the
  // first prompt advertises the built-in tools (get_current_time) and
  // the stage runs any <tool_call> the model emits, feeding results back.
  bool        _enable_tools{};
  // Enable the sandboxed Python tool (config "enable_python_tool",
  // default false). Independent opt-in from _enable_tools (executing
  // model-written code is a bigger risk); either flag activates the tool
  // loop. Adds a `run_python` tool backed by a seatbelt sandbox.
  bool        _enable_python_tool{};
  // Enable the confined file-I/O tools (config "enable_file_tools",
  // default false). Adds read_file / write_file / list_files backed by a
  // FileSandbox rooted at _file_sandbox_root; paths cannot escape it.
  bool        _enable_file_tools{};
  // Workspace root for the file tools. An explicit "file_sandbox_dir"
  // wins; otherwise "file_sandbox" picks the mode: "per-chat" (default) is
  // an ephemeral per-stage temp dir created in initialize() and removed in
  // the destructor (_file_sandbox_ephemeral tracks that); "persistent" is
  // the session sandbox root ($CWD/sandbox), kept across chats.
  std::string _file_sandbox_dir;
  std::string _file_sandbox_mode;      // "per-chat" | "persistent"
  std::string _file_sandbox_root;      // resolved root actually used
  bool        _file_sandbox_ephemeral{};
  std::shared_ptr<FileSandbox> _file_sandbox;
  // Enable the sandboxed shell tool (config "enable_shell_tool", default
  // false). Adds a `run_shell` tool backed by a seatbelt sandbox, sharing
  // the file-tool workspace (_file_sandbox) as its writable root. Executes
  // model-written commands -- a deliberate opt-in like the python tool.
  bool        _enable_shell_tool{};
  // Enable the web tools (config "enable_web_tools", default false). Adds
  // fetch_url / scrape_page. SSRF-guarded: requests to private/localhost
  // targets are refused unless _web_allow_private (config
  // "web_allow_private", default false) is set.
  bool        _enable_web_tools{};
  bool        _web_allow_private{};
  // Allow the sandboxed run_shell / run_python tools to use the per-user
  // system temp (config "allow_system_temp", default false). Off = all temp
  // stays under the workspace/CWD; on = tools that hardcode the system temp
  // (e.g. the macOS `mktemp` CLI) work, at the cost of temp outside the CWD.
  bool        _allow_system_temp{};
  // Fold reasoning (<think>) and tool-call blocks OUT of the streaming
  // out-port (index 1) so a speaking consumer voices only the answer
  // (config "stream_answer_only", default false). Out-port 0 always
  // carries the full text.
  bool        _stream_answer_only{};
  // The locally-dispatchable tool registry, seeded in initialize() when
  // tools are enabled and supported. Empty otherwise.
  McpToolRegistry _tools;
  // Unset => use family default; set => override.
  std::optional<bool> _disable_thinking;
  // Loaded lazily in initialize(); cleared on shutdown.
  std::shared_ptr<genai::LoadedLanguageModel> _lm;
#ifdef VPIPE_BUILD_APPLE_SILICON
  genai::SamplerParams _sampler_params;
  // Borrowed media-encoder towers (owned by the LM; cached in
  // initialize()). Null when the checkpoint has no such tower. Used by
  // the media-line turn path: a user line carrying attachment markers
  // decodes each item (FFmpeg) and encodes it through the model's own
  // tower, then splices the embeddings at the marker positions via
  // prefill_multimodal_metal.
  genai::MetalQwenVisionEncoder*   _mvis     = nullptr;
  genai::MetalGemma4VisionEncoder* _mgvis    = nullptr;
  genai::Gemma4UnifiedEmbedder*    _mguni    = nullptr;
  genai::MetalAudioEncoder*        _m_audio  = nullptr;
  genai::MetalGemma4AudioEncoder*  _mg_audio = nullptr;
  // MTP speculative-decode opt-out (config "mtp", default true; read in the
  // ctor). Gates the metal MTP fast path; token-exact, so it only changes
  // decode speed. The MLX path has no MTP head, hence no-MLX only.
  bool _mtp_enabled{};
  bool _i8_prefill{};   // LOSSY dynamic-int8 prefill GEMMs (opt-in)
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

  // Resolve/create the file-tool workspace root and register the file
  // tools into _tools. Called from initialize() when enable_file_tools is
  // set and the template supports tools. Sets _file_sandbox(_root) and
  // marks _file_sandbox_ephemeral when it created a temp dir.
  void setup_file_sandbox_();
};

}

#endif

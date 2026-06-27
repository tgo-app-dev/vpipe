#ifndef VPIPE_GENERATIVE_MODELS_CHAT_TEMPLATE_H
#define VPIPE_GENERATIVE_MODELS_CHAT_TEMPLATE_H

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace vpipe::genai {

class Tokenizer;

// Per-model-family chat-template renderer.
//
// Different LLM families wrap user/assistant turns in different
// special tokens:
//   * Llama-3 family:
//       <|begin_of_text|> (BOS once per session)
//       <|start_header_id|>{role}<|end_header_id|>\n\n
//       {content}<|eot_id|>
//       <|start_header_id|>assistant<|end_header_id|>\n\n
//   * Qwen3.5 / ChatML family (no BOS):
//       <|im_start|>{role}\n{content}<|im_end|>\n
//       <|im_start|>assistant\n
//
// The interface is intentionally narrow: a chat stage only ever needs
// to (1) append a user turn + assistant-prompt header to the prompt-id
// vector, (2) ask which ids should terminate decoding, and (3) get
// the single id to commit to the K/V cache after the model's response
// so the next user turn sees a clean turn boundary.
//
// Construction goes through `make_chat_template`, which dispatches on
// the model's architecture string (LlamaForCausalLM/Qwen2ForCausalLM/
// MistralForCausalLM all share the Llama-3 family;
// Qwen3_5ForConditionalGeneration uses ChatML). Tokens unavailable in
// the tokenizer are reported as warns at construction time so callers
// see them once at load time rather than per-turn.
class ChatTemplate {
public:
  virtual ~ChatTemplate() = default;

  ChatTemplate(const ChatTemplate&)            = delete;
  ChatTemplate& operator=(const ChatTemplate&) = delete;

  // Append a complete user turn followed by the assistant
  // generation-prompt header to `dst`. The first call in a fresh
  // K/V context should pass `is_first_turn=true` so families that
  // require a session-start token (Llama-3's `<|begin_of_text|>`)
  // emit it exactly once. On subsequent turns we pass false because
  // the cache already begins with BOS.
  virtual void render_user_turn(std::string_view      content,
                                bool                  is_first_turn,
                                std::vector<std::int32_t>* dst) const = 0;

  // Append a multimodal user turn: each entry in
  // `image_token_counts` describes one image whose pre-encoded
  // embeddings the caller has ready. The template inserts the
  // family-specific vision-start/vision-end wrapper around N copies
  // of the image-pad sentinel per image (so the LM can route image
  // positions to vision embeddings at prefill time). Default impl
  // ignores images and falls back to text-only render_user_turn();
  // VLM-family subclasses override.
  virtual void
  render_user_turn_vlm(std::string_view              content,
                       std::span<const int>          image_token_counts,
                       bool                          is_first_turn,
                       std::vector<std::int32_t>*    dst) const
  {
    (void)image_token_counts;
    render_user_turn(content, is_first_turn, dst);
  }

  // Token id used as the image-token placeholder in the rendered id
  // stream. The visual-qa stage swaps these positions for image
  // embeddings via TokenMuxer's ImageTokens branch at prefill time.
  // Returns -1 for templates without vision support.
  virtual std::int32_t image_pad_token_id() const noexcept
  { return -1; }

  // Token id used as the per-frame VIDEO placeholder in the rendered id
  // stream. Stages splice these positions to frame embeddings via
  // TokenMuxer's ImageTokens branch at prefill time. Defaults to
  // image_pad_token_id() -- families that reuse the image placeholder
  // for video (Qwen3-VL) need no override; Gemma-4 overrides to its
  // distinct video soft token (`<|video|>`, 258884).
  virtual std::int32_t video_pad_token_id() const noexcept
  { return image_pad_token_id(); }

  // Token id used as the audio-token placeholder in the rendered id
  // stream (Qwen3-ASR `<|audio_pad|>`). The audio-transcribe stage
  // swaps these positions for audio embeddings via TokenMuxer's
  // AudioTokens branch at prefill time. Returns -1 for templates
  // without audio support.
  virtual std::int32_t audio_pad_token_id() const noexcept
  { return -1; }

  // Multimodal AUDIO user turn (Gemma-4 audio understanding): a user
  // turn carrying an audio block (audio_start + N audio_pad + audio_end)
  // followed by the question text, then the assistant-turn open. Same
  // shape as render_user_turn_vlm but with audio sentinels. The default
  // ignores the audio and renders text only (families without audio).
  virtual void
  render_user_turn_audio(std::string_view              content,
                         int                           n_audio_tokens,
                         bool                          is_first_turn,
                         std::vector<std::int32_t>*    dst) const
  {
    (void)n_audio_tokens;
    render_user_turn(content, is_first_turn, dst);
  }

  // Append an INLINE audio block to an already-open user turn (i.e. between
  // render_video_prefix and render_vlm_completion): optional `caption` text
  // then audio_start + N audio_pad + audio_end. Lets one multimodal turn mix
  // video frames AND audio so the model can correlate sight + sound
  // (realtime-vqa). The N audio_pad slots are overlaid with audio-encoder
  // embeddings by prefill_multimodal. Returns false (no block emitted) for
  // families without audio support. Default: no-op.
  virtual bool
  render_audio_block(std::string_view              caption,
                     int                           n_audio_tokens,
                     std::vector<std::int32_t>*    dst) const
  {
    (void)caption;
    (void)n_audio_tokens;
    (void)dst;
    return false;
  }

  // ASR / audio-LM prompt rendering.
  //
  // Produces the full token stream for a single-turn audio-conditioned
  // prompt:
  //
  //   <|im_start|>system\n{system_prompt}<|im_end|>\n
  //   <|im_start|>user\n
  //     <|audio_start|><|audio_pad|>×N<|audio_end|>
  //   <|im_end|>\n
  //   <|im_start|>assistant\n
  //   [language {lang_hint}<asr_text>]
  //
  // The N audio_pad ids are slot fillers the caller's TokenMuxer
  // swaps for real audio-encoder embeddings at prefill time.
  // `language_hint` is the case-insensitive name of a language the
  // template should pre-emit on the assistant side -- when set, the
  // model only generates the transcription text. Empty hint defers
  // to the model's language-detection prefix
  // ("language English<asr_text>...").
  //
  // Default impl returns false. Audio-aware subclasses (Qwen3-ASR)
  // override.
  virtual bool
  render_asr_prompt(std::string_view              system_prompt,
                    int                           audio_token_count,
                    std::string_view              language_hint,
                    std::vector<std::int32_t>*    dst) const
  {
    (void)system_prompt; (void)audio_token_count;
    (void)language_hint; (void)dst;
    return false;
  }

  // VLM split rendering: render_vlm_prefix emits the shared-across-
  // questions portion of a user turn -- the user-role open and the
  // vision-wrapped image-pad blocks (no question text). render_vlm_
  // completion emits the per-question tail -- the question text,
  // the turn end, and the assistant-role open + extras. Together
  // they produce the same token stream as render_user_turn_vlm.
  //
  // Used by VisualQaStage to prefill the vision block once and
  // branch the K/V context per question. Returns false on families
  // that don't support the split (e.g. text-only families).
  virtual bool
  render_vlm_prefix(std::span<const int>          image_token_counts,
                    bool                          is_first_turn,
                    std::vector<std::int32_t>*    dst) const
  {
    (void)image_token_counts; (void)is_first_turn; (void)dst;
    return false;
  }

  virtual bool
  render_vlm_completion(std::string_view              content,
                        std::vector<std::int32_t>*    dst) const
  {
    (void)content; (void)dst;
    return false;
  }

  // Extended VLM prefix render with optional pre-image / post-image
  // text segments and an optional close_turn flag. Layout:
  //
  //   <|im_start|>user
  //   [pre_image_prompt]
  //   <vision block>+
  //   [post_image_prompt]
  //   if close_turn:
  //     <|im_end|>
  //     <|im_start|>assistant
  //     <assistant extras>
  //
  // When close_turn==false the user turn is left OPEN (same shape as
  // render_vlm_prefix's output, just with pre/post text injected).
  // The caller follows up with render_vlm_completion(question, ...)
  // to close the user turn + open the assistant turn -- this matches
  // the existing prefix+completion split.
  //
  // When close_turn==true the call closes the user turn AND opens
  // the assistant turn in one shot, with no question text between
  // the post-image prompt and the assistant header. The caller
  // decodes the assistant's pre-question reply against this prefix;
  // subsequent question turns then start a fresh user turn (via
  // render_user_turn with is_first_turn=false).
  //
  // Returns false on families that don't support the split. The
  // default impl accepts the call only when pre_image_prompt and
  // post_image_prompt are empty AND close_turn==false, in which case
  // it delegates to render_vlm_prefix; otherwise it returns false
  // so callers can fall back to a different rendering path.
  virtual bool
  render_vlm_prefix_ex(
      std::span<const int>          image_token_counts,
      bool                          is_first_turn,
      std::string_view              pre_image_prompt,
      std::string_view              post_image_prompt,
      bool                          close_turn,
      std::vector<std::int32_t>*    dst) const
  {
    if (!pre_image_prompt.empty() || !post_image_prompt.empty()
        || close_turn) {
      return false;
    }
    return render_vlm_prefix(image_token_counts, is_first_turn, dst);
  }

  // VLM split rendering for video. Emits the shared-across-questions
  // portion of a video user turn -- the user-role open and the
  // timestamped vision-wrapped frame blocks -- WITHOUT the question
  // text or the assistant-role open. Pair with render_vlm_completion
  // (which emits content + close_user + assistant_open) to produce
  // the same token stream as render_user_turn_video.
  //
  // Used by RealtimeVqaStage to prefill the video prefix once and
  // branch the K/V context per question. Returns false on families
  // that don't support the split (default).
  virtual bool
  render_video_prefix(
      std::span<const float>        frame_timestamps_seconds,
      std::span<const int>          image_token_counts,
      bool                          is_first_turn,
      std::vector<std::int32_t>*    dst) const
  {
    (void)frame_timestamps_seconds;
    (void)image_token_counts;
    (void)is_first_turn;
    (void)dst;
    return false;
  }

  // Extended video prefix: optional `pre_image_prompt` plaintext is
  // inserted between the user-turn open and the first frame block.
  // Useful for chaining scene descriptions across calls (the caller
  // injects "Here's the summary of the prior scene: ..." so the LM
  // sees it as preamble inside the user turn that contains the new
  // frames). Default impl falls back to the 4-arg variant ONLY when
  // pre_image_prompt is empty; non-empty pre_image_prompt on a
  // template that hasn't overridden this returns false so the caller
  // can pick a fallback path.
  virtual bool
  render_video_prefix(
      std::span<const float>        frame_timestamps_seconds,
      std::span<const int>          image_token_counts,
      bool                          is_first_turn,
      std::string_view              pre_image_prompt,
      std::vector<std::int32_t>*    dst) const
  {
    if (!pre_image_prompt.empty()) {
      return false;
    }
    return render_video_prefix(frame_timestamps_seconds,
                               image_token_counts, is_first_turn,
                               dst);
  }

  // Multimodal turn for video. Same shape as render_user_turn_vlm
  // but each frame's vision-wrapped block is preceded by a
  // `<{timestamp_seconds:.1f} seconds>` text marker (rendered through
  // the tokenizer as plain text). `frame_timestamps_seconds.size()`
  // must equal `image_token_counts.size()` -- one timestamp per
  // temporal-merged frame. Default implementation drops timestamps
  // and delegates to render_user_turn_vlm so text-only families and
  // image-only VLM families keep their existing behaviour.
  // Convenience overload: render the full multimodal video user turn
  // with an optional pre-image plaintext prepended. Default impl
  // forwards to the 4-arg variant when pre_image_prompt is empty and
  // falls back to render_user_turn_vlm (no timestamps, no pre-text)
  // when pre_image_prompt is non-empty AND the template hasn't
  // overridden this hook.
  virtual void
  render_user_turn_video(
      std::string_view              content,
      std::span<const float>        frame_timestamps_seconds,
      std::span<const int>          image_token_counts,
      bool                          is_first_turn,
      std::string_view              pre_image_prompt,
      std::vector<std::int32_t>*    dst) const
  {
    if (pre_image_prompt.empty()) {
      render_user_turn_video(content, frame_timestamps_seconds,
                             image_token_counts, is_first_turn, dst);
      return;
    }
    (void)frame_timestamps_seconds;
    render_user_turn_vlm(content, image_token_counts,
                         is_first_turn, dst);
  }

  virtual void
  render_user_turn_video(
      std::string_view              content,
      std::span<const float>        frame_timestamps_seconds,
      std::span<const int>          image_token_counts,
      bool                          is_first_turn,
      std::vector<std::int32_t>*    dst) const
  {
    (void)frame_timestamps_seconds;
    render_user_turn_vlm(content, image_token_counts,
                         is_first_turn, dst);
  }

  // The id to commit to the K/V cache after a generated assistant
  // response, so the next user-turn prefill sees a clean turn
  // boundary. -1 means "the assistant turn closes itself" (a model
  // that emits `<|im_end|>` as its stop token doesn't need an
  // explicit close). Most callers will commit unconditionally and
  // skip when this is -1.
  virtual std::int32_t assistant_close_token_id() const = 0;

  // Returns true if `id` should terminate the decode loop.
  // Implementations check against a small set of family-specific
  // stop tokens (eot/eom/end_of_text for Llama-3; im_end/endoftext
  // for ChatML) without scanning the whole vocab.
  virtual bool is_stop_token(std::int32_t id) const = 0;

  // Sanitize a fully-decoded assistant turn for display/storage: strip
  // any model-internal reasoning the checkpoint emits but that should not
  // surface to the user (e.g. Gemma-4's `<|channel>thought ...<channel|>`
  // block, mirroring the checkpoint's own strip_thinking macro). The
  // default is identity; families with a reasoning channel override it.
  // Callers that always want a clean answer (e.g. realtime-vqa, where
  // thinking is disabled) run their decoded text through this.
  virtual std::string sanitize_output(std::string text) const
  { return text; }

  // Family name for log / telemetry. Stable string literal; no need
  // to copy.
  virtual std::string_view family_name() const = 0;

protected:
  ChatTemplate() = default;
};

// Factory: dispatches on the HF architecture string from
// ModelConfig::architecture. Returns nullptr when the family is
// unknown -- caller should log a warn and leave the LM without a
// chat template (raw prefill still works for callers that build their
// own tokens). The tokenizer is borrowed for special-token id
// lookups; it must outlive the returned ChatTemplate.
//
// `disable_thinking` overrides the family-default thinking mode for
// the Qwen3 family (which has both thinking-on `<think>\n` and
// thinking-off `<think>\n\n</think>\n\n` assistant preambles in its
// chat template). Defaults: Qwen3 text-only is family-default
// thinking-OFF; Qwen3-VL is family-default thinking-ON. Passing an
// explicit value flips the bit. Ignored on families without a
// thinking concept (Llama-3, vanilla ChatML / Qwen2).
std::unique_ptr<ChatTemplate>
make_chat_template(const std::string&    architecture,
                   const Tokenizer&      tokenizer,
                   std::optional<bool>   disable_thinking = std::nullopt);

}

#endif

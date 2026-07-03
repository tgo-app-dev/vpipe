#include "generative-models/chat-template.h"

#include "common/media-line.h"
#include "generative-models/tokenizer.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vpipe::genai {

namespace {

// Convenience: encode `text` via the tokenizer and append to dst.
void
append_text_(const Tokenizer&            tok,
             std::string_view            text,
             std::vector<std::int32_t>*  dst)
{
  auto piece = tok.encode(std::string(text));
  dst->insert(dst->end(), piece.begin(), piece.end());
}

// Convenience: look up a special token id. Returns -1 when missing,
// which is the same sentinel Tokenizer::special_token_id uses.
std::int32_t
sp_(const Tokenizer& tok, const char* name)
{
  return tok.special_token_id(name);
}

// Push `id` to dst when valid. Returns false if id < 0 so the caller
// can flag a missing-special-token warning at construction time.
bool
push_sp_(std::int32_t                id,
         std::vector<std::int32_t>*  dst)
{
  if (id < 0) {
    return false;
  }
  dst->push_back(id);
  return true;
}

// ---------------------------------------------------------------------
// Llama-3 chat template.
// ---------------------------------------------------------------------
class Llama3ChatTemplate final : public ChatTemplate {
public:
  explicit Llama3ChatTemplate(const Tokenizer& tok)
    : _tok(tok)
    , _bos      (sp_(tok, "<|begin_of_text|>"))
    , _hdr_start(sp_(tok, "<|start_header_id|>"))
    , _hdr_end  (sp_(tok, "<|end_header_id|>"))
    , _eot      (sp_(tok, "<|eot_id|>"))
    , _eom      (sp_(tok, "<|eom_id|>"))
    , _end_text (sp_(tok, "<|end_of_text|>"))
  {}

  void render_user_turn(std::string_view              content,
                        bool                          is_first_turn,
                        std::vector<std::int32_t>*    dst) const override
  {
    // BOS once per session.
    if (is_first_turn) {
      (void)push_sp_(_bos, dst);
    }
    // User turn: header + content + eot.
    (void)push_sp_(_hdr_start, dst);
    append_text_(_tok, "user", dst);
    (void)push_sp_(_hdr_end, dst);
    append_text_(_tok, "\n\n", dst);
    append_text_(_tok, content, dst);
    (void)push_sp_(_eot, dst);
    // Assistant generation-prompt header: model continues from here.
    (void)push_sp_(_hdr_start, dst);
    append_text_(_tok, "assistant", dst);
    (void)push_sp_(_hdr_end, dst);
    append_text_(_tok, "\n\n", dst);
  }

  std::int32_t assistant_close_token_id() const override
  {
    return _eot;
  }

  bool is_stop_token(std::int32_t id) const override
  {
    // -1 lookups never match (guards against missing-tokenizer-tokens
    // turning is_stop_token into "always true").
    if (id < 0) {
      return false;
    }
    return id == _eot || id == _eom || id == _end_text;
  }

  std::string_view family_name() const override { return "llama3"; }

private:
  const Tokenizer& _tok;
  const std::int32_t _bos;
  const std::int32_t _hdr_start;
  const std::int32_t _hdr_end;
  const std::int32_t _eot;
  const std::int32_t _eom;
  const std::int32_t _end_text;
};

// ---------------------------------------------------------------------
// ChatML template (Qwen2.5 family + the structural shape used by
// every later ChatML model).
// ---------------------------------------------------------------------
class ChatMLChatTemplate : public ChatTemplate {
public:
  explicit ChatMLChatTemplate(const Tokenizer& tok)
    : _tok(tok)
    , _im_start  (sp_(tok, "<|im_start|>"))
    , _im_end    (sp_(tok, "<|im_end|>"))
    , _end_of_txt(sp_(tok, "<|endoftext|>"))
  {}

  void render_user_turn(std::string_view              content,
                        bool                          is_first_turn,
                        std::vector<std::int32_t>*    dst) const override
  {
    // ChatML has no session-start token.
    (void)is_first_turn;
    // User turn: <|im_start|>user\n{content}<|im_end|>\n
    (void)push_sp_(_im_start, dst);
    append_text_(_tok, "user\n", dst);
    append_text_(_tok, content, dst);
    (void)push_sp_(_im_end, dst);
    append_text_(_tok, "\n", dst);
    // Assistant generation-prompt header. Vanilla ChatML stops here;
    // model families that prepend additional control tokens override
    // append_assistant_extras_().
    (void)push_sp_(_im_start, dst);
    append_text_(_tok, "assistant\n", dst);
    append_assistant_extras_(dst);
  }

  std::int32_t assistant_close_token_id() const override
  {
    // <|im_end|> closes the assistant turn. The model normally emits
    // it as its own stop token; we still return it here so the
    // chat stage commits it on the (rare) max_new_tokens-cutoff path
    // when the model didn't reach the stop on its own.
    return _im_end;
  }

  bool is_stop_token(std::int32_t id) const override
  {
    if (id < 0) {
      return false;
    }
    return id == _im_end || id == _end_of_txt;
  }

  std::string_view family_name() const override { return "chatml"; }

protected:
  const Tokenizer& tok_() const noexcept { return _tok; }

  // Override hook. Default (vanilla ChatML, Qwen2.5) emits nothing.
  // Qwen3-family subclasses inject the thinking-disabled
  // `<think>\n\n</think>\n\n` prefix here.
  virtual void
  append_assistant_extras_(std::vector<std::int32_t>* /*dst*/) const
  {}

private:
  const Tokenizer& _tok;
  const std::int32_t _im_start;
  const std::int32_t _im_end;
  const std::int32_t _end_of_txt;
};

// Qwen3-family variant: same ChatML scaffold, but emits either the
// thinking-DISABLED prefix `<think>\n\n</think>\n\n` or the
// thinking-ENABLED prefix `<think>\n` after the assistant header. The
// model's published chat_template.jinja defaults to thinking-on (the
// model produces a reasoning block then a visible answer); the
// `disable_thinking` ctor parameter flips between modes. Default
// here is thinking-OFF so a vanilla text chat stage gets a visible
// answer without parsing out the reasoning block; Qwen3VLChatTemplate
// overrides the default to thinking-ON because the VL variant was
// primarily trained / served with `<think>\n`.
class Qwen3ChatTemplate : public ChatMLChatTemplate {
public:
  explicit Qwen3ChatTemplate(const Tokenizer& tok,
                             bool             disable_thinking = true)
    : ChatMLChatTemplate(tok)
    , _disable_thinking(disable_thinking)
    , _think_open (sp_(tok, "<think>"))
    , _think_close(sp_(tok, "</think>"))
  {}

  std::string_view family_name() const override { return "qwen3-chatml"; }

  bool thinking_disabled() const noexcept { return _disable_thinking; }

  // Thinking-ON extras end with `<think>\n`, so generation starts
  // inside the reasoning block; the opening token never streams
  // (it's part of the prompt). See the base-class docs.
  bool assistant_prompt_opens_thinking() const noexcept override
  { return !_disable_thinking; }

protected:
  // Emit the `<think>...` preamble as the SINGLE special-token ids the
  // tokenizer registered for `<think>` (id 248068) and `</think>` (id
  // 248069) on the Qwen3 family. The HF chat_template.jinja renders
  // these as text, but the HF tokenizer pipeline splits added-tokens
  // out before BPE so the model receives the single token ids. Our
  // encoder doesn't segment by added_tokens, so a naive
  // `append_text_(tok, "<think>\n", dst)` would BPE the literal "<",
  // "think", ">" chars and miss the trained boundary entirely --
  // which is exactly what produced the bogus `</think>` echoes in
  // disable_thinking=true output. When the special tokens aren't in
  // the vocab (older Qwen3 dev builds) fall back to the text path.
  void
  append_assistant_extras_(std::vector<std::int32_t>* dst) const override
  {
    if (_disable_thinking) {
      if (_think_open >= 0 && _think_close >= 0) {
        // <think>\n\n</think>\n\n
        dst->push_back(_think_open);
        append_text_(tok_(), "\n\n", dst);
        dst->push_back(_think_close);
        append_text_(tok_(), "\n\n", dst);
      } else {
        append_text_(tok_(), "<think>\n\n</think>\n\n", dst);
      }
    } else {
      if (_think_open >= 0) {
        // <think>\n
        dst->push_back(_think_open);
        append_text_(tok_(), "\n", dst);
      } else {
        append_text_(tok_(), "<think>\n", dst);
      }
    }
  }

private:
  bool         _disable_thinking;
  std::int32_t _think_open;
  std::int32_t _think_close;
};

// Qwen3-VL variant: ChatML scaffold + thinking-ENABLED assistant
// extras PLUS multimodal user turn rendering. The Qwen3-VL chat
// template wraps each image's image-pad placeholders inside
// <|vision_start|>...<|vision_end|>:
//
//   <|im_start|>user
//   <|vision_start|><|image_pad|>...<|image_pad|><|vision_end|>
//   <|vision_start|><|image_pad|>...<|image_pad|><|vision_end|>
//   {user text}<|im_end|>
//   <|im_start|>assistant
//   <think>\n
//
// IMPORTANT: VLM uses thinking-ENABLED preamble (`<think>\n`) while
// the text-only Qwen3 template uses thinking-DISABLED
// (`<think>\n\n</think>\n\n`). Qwen3.5-VL appears to have been
// primarily trained / served with thinking-on for multimodal
// prompts — this is what mlx-vlm's `apply_chat_template` produces
// by default. With thinking-OFF the LM intermittently predicts
// <|im_end|> as the very first token for less-saturated images
// (the model can't commit to a final answer in one shot from
// vision-only context, and "no answer needed" is the safest
// argmax). Thinking-on gives the LM a reasoning runway and matches
// the training distribution. Output now contains a visible
// `<think>...</think>` block followed by the final answer; the
// visual-qa stage streams both through to the user.
//
// The image-pad placeholder count per image is set by the vision
// encoder's EncodedImage::n_tokens; the visual-qa stage passes those
// counts into render_user_turn_vlm.
class Qwen3VLChatTemplate final : public Qwen3ChatTemplate {
public:
  explicit Qwen3VLChatTemplate(const Tokenizer& tok,
                               bool             disable_thinking = false)
    // Qwen3-VL was primarily trained / served with thinking-ON; pass
    // the inverse default to the Qwen3 base so the family-default for
    // VLM users is `<think>\n`. The base class's
    // append_assistant_extras_ handles both modes uniformly.
    : Qwen3ChatTemplate(tok, disable_thinking)
    , _vision_start(sp_(tok, "<|vision_start|>"))
    , _vision_end  (sp_(tok, "<|vision_end|>"))
    , _image_pad   (sp_(tok, "<|image_pad|>"))
  {}

  std::string_view family_name() const override
  { return "qwen3-vl-chatml"; }

  std::int32_t image_pad_token_id() const noexcept override
  { return _image_pad; }

  void
  render_user_turn_vlm(std::string_view              content,
                       std::span<const int>          image_token_counts,
                       bool                          is_first_turn,
                       std::vector<std::int32_t>*    dst) const override
  {
    (void)render_vlm_prefix(image_token_counts, is_first_turn, dst);
    (void)render_vlm_completion(content, dst);
  }

  // Mixed in-line turn: text runs and image blocks in caller order.
  // Qwen3.5-VL has no audio channel, so any Audio chunk rejects the
  // whole render (nothing appended) and the caller falls back.
  bool
  render_user_turn_media(std::span<const MediaChunk>   chunks,
                         bool                          is_first_turn,
                         std::vector<std::int32_t>*    dst) const override
  {
    (void)is_first_turn;   // ChatML has no session-start token
    if (_image_pad < 0) {
      return false;
    }
    for (const auto& c : chunks) {
      if (c.kind == MediaChunk::Kind::Audio) {
        return false;
      }
    }
    (void)push_sp_(im_start_(), dst);
    append_text_(tok_(), "user\n", dst);
    for (const auto& c : chunks) {
      if (c.kind == MediaChunk::Kind::Text) {
        if (!c.text.empty()) {
          append_text_(tok_(), c.text, dst);
        }
      } else if (c.n_tokens > 0) {
        (void)push_sp_(_vision_start, dst);
        dst->insert(dst->end(),
                    static_cast<std::size_t>(c.n_tokens), _image_pad);
        (void)push_sp_(_vision_end, dst);
      }
    }
    (void)push_sp_(im_end_(), dst);
    append_text_(tok_(), "\n", dst);
    (void)push_sp_(im_start_(), dst);
    append_text_(tok_(), "assistant\n", dst);
    append_assistant_extras_(dst);
    return true;
  }

  // Shared-prefix render. Used by VisualQaStage to prefill the
  // vision block once across questions; pairs with
  // render_vlm_completion.
  bool
  render_vlm_prefix(std::span<const int>          image_token_counts,
                    bool                          /*is_first_turn*/,
                    std::vector<std::int32_t>*    dst) const override
  {
    // <|im_start|>user\n
    (void)push_sp_(im_start_(), dst);
    append_text_(tok_(), "user\n", dst);
    // One vision-wrapped block per image. The image-pad placeholders
    // are dropped in by repeating the id N times; prefill_multimodal
    // overlays vision embeddings at those positions via TokenMuxer's
    // ImageTokens branch.
    for (int n : image_token_counts) {
      if (n <= 0) {
        continue;
      }
      (void)push_sp_(_vision_start, dst);
      if (_image_pad >= 0) {
        dst->insert(dst->end(), static_cast<std::size_t>(n),
                    _image_pad);
      }
      (void)push_sp_(_vision_end, dst);
    }
    return true;
  }

  bool
  render_vlm_completion(std::string_view              content,
                        std::vector<std::int32_t>*    dst) const override
  {
    append_text_(tok_(), content, dst);
    (void)push_sp_(im_end_(), dst);
    append_text_(tok_(), "\n", dst);
    (void)push_sp_(im_start_(), dst);
    append_text_(tok_(), "assistant\n", dst);
    append_assistant_extras_(dst);
    return true;
  }

  // Extended prefix: pre_image_prompt and post_image_prompt sit just
  // outside the vision-wrapped image-pad block; close_turn closes the
  // user turn + opens the assistant turn (no question text in
  // between). This is the shape the visual-qa stage uses when it
  // wants the model to produce a pre-question reply against the
  // image-grounded post-image prompt before fanning out to per-
  // question branches.
  bool
  render_vlm_prefix_ex(
      std::span<const int>          image_token_counts,
      bool                          /*is_first_turn*/,
      std::string_view              pre_image_prompt,
      std::string_view              post_image_prompt,
      bool                          close_turn,
      std::vector<std::int32_t>*    dst) const override
  {
    (void)push_sp_(im_start_(), dst);
    append_text_(tok_(), "user\n", dst);
    if (!pre_image_prompt.empty()) {
      append_text_(tok_(), pre_image_prompt, dst);
    }
    for (int n : image_token_counts) {
      if (n <= 0) {
        continue;
      }
      (void)push_sp_(_vision_start, dst);
      if (_image_pad >= 0) {
        dst->insert(dst->end(), static_cast<std::size_t>(n),
                    _image_pad);
      }
      (void)push_sp_(_vision_end, dst);
    }
    if (!post_image_prompt.empty()) {
      append_text_(tok_(), post_image_prompt, dst);
    }
    if (close_turn) {
      (void)push_sp_(im_end_(), dst);
      append_text_(tok_(), "\n", dst);
      (void)push_sp_(im_start_(), dst);
      append_text_(tok_(), "assistant\n", dst);
      append_assistant_extras_(dst);
    }
    return true;
  }

  // Video variant. Each frame's vision-wrapped block is preceded by
  // a `<{seconds:.1f} seconds>` text marker. The marker is encoded
  // by the tokenizer the same way any other text run is -- there's
  // no dedicated timestamp special token -- so the LM sees the same
  // textual cue Qwen3-VL was trained on (Qwen3VLProcessor injects
  // exactly this string before each temporal-merged frame's
  // <|vision_start|>). Caller passes one timestamp per entry in
  // image_token_counts; if the two spans disagree in length we fall
  // back to render_user_turn_vlm (no timestamps).
  void
  render_user_turn_video(
      std::string_view              content,
      std::span<const float>        frame_timestamps_seconds,
      std::span<const int>          image_token_counts,
      bool                          is_first_turn,
      std::vector<std::int32_t>*    dst) const override
  {
    render_user_turn_video(content, frame_timestamps_seconds,
                           image_token_counts, is_first_turn,
                           std::string_view{}, dst);
  }

  void
  render_user_turn_video(
      std::string_view              content,
      std::span<const float>        frame_timestamps_seconds,
      std::span<const int>          image_token_counts,
      bool                          is_first_turn,
      std::string_view              pre_image_prompt,
      std::vector<std::int32_t>*    dst) const override
  {
    if (!render_video_prefix(frame_timestamps_seconds,
                             image_token_counts, is_first_turn,
                             pre_image_prompt, dst)) {
      render_user_turn_vlm(content, image_token_counts,
                           is_first_turn, dst);
      return;
    }
    (void)render_vlm_completion(content, dst);
  }

  // Shared video prefix: user-turn open + per-frame timestamp marker
  // + vision-wrapped image-pad blocks. NO question text, NO user
  // turn close. Pairs with render_vlm_completion to produce the same
  // token stream as render_user_turn_video. Falls back to false when
  // the per-frame timestamp count disagrees with the image count so
  // the caller can pick a non-split path.
  bool
  render_video_prefix(
      std::span<const float>        frame_timestamps_seconds,
      std::span<const int>          image_token_counts,
      bool                          is_first_turn,
      std::vector<std::int32_t>*    dst) const override
  {
    return render_video_prefix(frame_timestamps_seconds,
                               image_token_counts, is_first_turn,
                               std::string_view{}, dst);
  }

  // 5-arg variant: optionally injects `pre_image_prompt` as plaintext
  // BETWEEN the user-role open and the first frame block. The text
  // is encoded the same way render_user_turn_vlm encodes inline text
  // (tokenizer BPE on the raw string), so callers can include
  // arbitrary preamble — e.g. "Here's the summary of the prior
  // scene: ...".
  bool
  render_video_prefix(
      std::span<const float>        frame_timestamps_seconds,
      std::span<const int>          image_token_counts,
      bool                          /*is_first_turn*/,
      std::string_view              pre_image_prompt,
      std::vector<std::int32_t>*    dst) const override
  {
    if (frame_timestamps_seconds.size() != image_token_counts.size()) {
      return false;
    }
    (void)push_sp_(im_start_(), dst);
    append_text_(tok_(), "user\n", dst);
    if (!pre_image_prompt.empty()) {
      append_text_(tok_(), pre_image_prompt, dst);
    }
    for (std::size_t i = 0; i < image_token_counts.size(); ++i) {
      const int n = image_token_counts[i];
      if (n <= 0) {
        continue;
      }
      // Format "<X.Y seconds>" without iostream/std::format to keep
      // the dependency surface flat; one decimal mirrors the HF
      // reference (`f"<{curr_time:.1f} seconds>"`).
      char buf[32];
      std::snprintf(buf, sizeof buf, "<%.1f seconds>",
                    static_cast<double>(frame_timestamps_seconds[i]));
      append_text_(tok_(), buf, dst);
      (void)push_sp_(_vision_start, dst);
      if (_image_pad >= 0) {
        dst->insert(dst->end(), static_cast<std::size_t>(n),
                    _image_pad);
      }
      (void)push_sp_(_vision_end, dst);
    }
    return true;
  }

protected:
  // Exposed for the VLM subclass via inherited accessors (declared
  // private in ChatMLChatTemplate). Forwarder helpers keep the
  // subclass's internals clean.
  std::int32_t im_start_() const noexcept
  {
    return sp_(tok_(), "<|im_start|>");
  }
  std::int32_t im_end_() const noexcept
  {
    return sp_(tok_(), "<|im_end|>");
  }

private:
  const std::int32_t _vision_start;
  const std::int32_t _vision_end;
  const std::int32_t _image_pad;
};

// ---------------------------------------------------------------------
// Qwen3-ASR template.
//
// Single-turn audio-conditioned chat. The published template emits:
//
//   <|im_start|>system\n{system_prompt}\n<|im_end|>\n
//   <|im_start|>user\n
//     <|audio_start|><|audio_pad|>×N<|audio_end|>
//   <|im_end|>\n
//   <|im_start|>assistant\n
//   [language {lang_hint}<asr_text>]
//
// There's no <think> preamble (ASR is single-pass transcription, not
// reasoning), and no per-turn vision wrapper -- inherits ChatML
// scaffolding for `<|im_start|>...<|im_end|>` framing but overrides
// the entire prompt rendering via `render_asr_prompt`. Stop tokens
// are the standard ChatML pair (im_end + endoftext).
// ---------------------------------------------------------------------
class Qwen3AsrChatTemplate final : public ChatMLChatTemplate {
public:
  explicit Qwen3AsrChatTemplate(const Tokenizer& tok)
    : ChatMLChatTemplate(tok)
    , _audio_start(sp_(tok, "<|audio_start|>"))
    , _audio_end  (sp_(tok, "<|audio_end|>"))
    , _audio_pad  (sp_(tok, "<|audio_pad|>"))
  {}

  std::string_view family_name() const override
  { return "qwen3-asr-chatml"; }

  std::int32_t audio_pad_token_id() const noexcept override
  { return _audio_pad; }

  bool
  render_asr_prompt(std::string_view              system_prompt,
                    int                           audio_token_count,
                    std::string_view              language_hint,
                    std::vector<std::int32_t>*    dst) const override
  {
    if (!dst || audio_token_count <= 0 || _audio_pad < 0) {
      return false;
    }
    const std::int32_t im_start = sp_(tok_(), "<|im_start|>");
    const std::int32_t im_end   = sp_(tok_(), "<|im_end|>");
    if (im_start < 0 || im_end < 0) {
      return false;
    }

    // <|im_start|>system\n{content}\n<|im_end|>\n
    // The mlx-audio reference appends '\n' to a non-empty system
    // prompt; on an empty prompt the section is just
    // "<|im_start|>system\n<|im_end|>\n" with no body line.
    dst->push_back(im_start);
    append_text_(tok_(), "system\n", dst);
    if (!system_prompt.empty()) {
      append_text_(tok_(), system_prompt, dst);
      append_text_(tok_(), "\n", dst);
    }
    dst->push_back(im_end);
    append_text_(tok_(), "\n", dst);

    // <|im_start|>user\n
    //   <|audio_start|><|audio_pad|>×N<|audio_end|>
    // <|im_end|>\n
    dst->push_back(im_start);
    append_text_(tok_(), "user\n", dst);
    if (_audio_start >= 0) {
      dst->push_back(_audio_start);
    }
    for (int i = 0; i < audio_token_count; ++i) {
      dst->push_back(_audio_pad);
    }
    if (_audio_end >= 0) {
      dst->push_back(_audio_end);
    }
    dst->push_back(im_end);
    append_text_(tok_(), "\n", dst);

    // <|im_start|>assistant\n
    // [language {lang_hint}<asr_text>]
    dst->push_back(im_start);
    append_text_(tok_(), "assistant\n", dst);
    if (!language_hint.empty()) {
      // Match the mlx-audio reference. `<asr_text>` is a SPECIAL
      // ADDED token (id 151704 on Qwen3-ASR); BPE-encoding it as
      // text splits "<", "asr_text", ">" into 3+ sub-tokens which
      // the model has never seen in that combination and reacts to
      // by emitting <|im_end|> immediately on long-form clips. Push
      // the single special id directly.
      append_text_(tok_(), "language ", dst);
      append_text_(tok_(), language_hint, dst);
      const std::int32_t asr_text = sp_(tok_(), "<asr_text>");
      if (asr_text >= 0) {
        dst->push_back(asr_text);
      } else {
        // Older Qwen3-ASR fixtures without the added-token entry:
        // fall back to text encoding so at least the chat template
        // remains well-formed.
        append_text_(tok_(), "<asr_text>", dst);
      }
    }
    return true;
  }

protected:
  // Qwen3-ASR has no thinking preamble.
  void
  append_assistant_extras_(std::vector<std::int32_t>* /*dst*/) const override
  {}

private:
  const std::int32_t _audio_start;
  const std::int32_t _audio_end;
  const std::int32_t _audio_pad;
};

// ---------------------------------------------------------------------
// Gemma-4 chat template.
//
//   <bos>
//   <start_of_turn>user\n{content}<end_of_turn>\n
//   <start_of_turn>model\n
//
// The role strings ("user" / "model") and the newlines are plain text;
// the turn markers + BOS are special tokens. For the gemma-4 tokenizer
// they decode as <bos>(2), <|turn>(105 == start_of_turn), <turn|>(106
// == end_of_turn). Stop tokens (config eos_token_id): <eos>(1),
// <turn|>(106), <|tool_response>(50).
// ---------------------------------------------------------------------
class GemmaChatTemplate final : public ChatTemplate {
public:
  // `thinking_variant` is used for the gemma4_unified family (12B), whose
  // reasoning is steered at render time: `thinking_on` selects between
  // emitting a leading `<|think|>` system turn (reasoning ON) and pre-filling
  // an empty `<|channel>thought\n<channel|>` on every model turn (reasoning
  // OFF -- meant to make the model skip its own thought block).
  // NOTE: e4b ALSO carries a reasoning channel, but it is constructed plain
  // (thinking_variant=false): the empty-thought prefill makes e4b answer in
  // open meta-reasoning rather than skip it, so consumers that want a clean
  // answer strip the channel post-hoc via sanitize_output() instead.
  explicit GemmaChatTemplate(const Tokenizer& tok,
                             bool             thinking_variant = false,
                             bool             thinking_on      = false)
    : _tok(tok)
    , _bos      (sp_(tok, "<bos>"))
    , _sot      (sp_(tok, "<|turn>"))
    , _eot      (sp_(tok, "<turn|>"))
    , _eos      (sp_(tok, "<eos>"))
    , _tool_resp(sp_(tok, "<|tool_response>"))
    , _boi      (sp_(tok, "<|image>"))    // begin-of-image  (255999)
    , _img_pad  (sp_(tok, "<|image|>"))   // image soft tok  (258880)
    , _eoi      (sp_(tok, "<image|>"))    // end-of-image    (258882)
    , _boa      (sp_(tok, "<|audio>"))    // begin-of-audio  (256000)
    , _aud_pad  (sp_(tok, "<|audio|>"))   // audio soft tok  (258881)
    , _eoa      (sp_(tok, "<audio|>"))    // end-of-audio    (258883)
    , _vid_pad  (sp_(tok, "<|video|>"))   // video soft tok  (258884)
    , _think       (sp_(tok, "<|think|>"))    // enable-reasoning marker (98)
    , _channel_open(sp_(tok, "<|channel>"))   // thought channel open   (100)
    , _channel_close(sp_(tok, "<channel|>"))  // thought channel close  (101)
    , _thinking_variant(thinking_variant)
    , _thinking_on(thinking_on)
  {}

  // Turn opener: <bos> on the first turn, plus -- for the thinking variant
  // with reasoning ON -- a leading `<|turn>system\n<|think|>\n<turn|>\n`
  // block that arms the model's reasoning. Replaces the bare bos push so
  // every render path picks the behaviour up.
  void emit_turn_open_(bool is_first_turn,
                       std::vector<std::int32_t>* dst) const
  {
    if (!is_first_turn) {
      return;
    }
    (void)push_sp_(_bos, dst);
    if (_thinking_variant && _thinking_on && _think >= 0) {
      (void)push_sp_(_sot, dst);
      append_text_(_tok, "system\n", dst);
      (void)push_sp_(_think, dst);
      append_text_(_tok, "\n", dst);
      (void)push_sp_(_eot, dst);
      append_text_(_tok, "\n", dst);
    }
  }

  // Model-turn opener: `<|turn>model\n`, plus -- for the thinking variant
  // with reasoning OFF -- an empty `<|channel>thought\n<channel|>` so the
  // model emits its answer immediately instead of an own reasoning block.
  void emit_model_open_(std::vector<std::int32_t>* dst) const
  {
    (void)push_sp_(_sot, dst);
    append_text_(_tok, "model\n", dst);
    if (_thinking_variant && !_thinking_on
        && _channel_open >= 0 && _channel_close >= 0) {
      (void)push_sp_(_channel_open, dst);
      append_text_(_tok, "thought\n", dst);
      (void)push_sp_(_channel_close, dst);
    }
  }

  void render_user_turn(std::string_view              content,
                        bool                          is_first_turn,
                        std::vector<std::int32_t>*    dst) const override
  {
    emit_turn_open_(is_first_turn, dst);
    (void)push_sp_(_sot, dst);
    append_text_(_tok, "user\n", dst);
    append_text_(_tok, content, dst);
    (void)push_sp_(_eot, dst);
    append_text_(_tok, "\n", dst);
    emit_model_open_(dst);
  }

  // ---- VLM surface (image soft tokens) ----------------------------
  std::int32_t image_pad_token_id() const noexcept override
  { return _img_pad; }

  std::int32_t audio_pad_token_id() const noexcept override
  { return _aud_pad; }

  // Gemma-4 video uses a DISTINCT soft token (`<|video|>`, 258884), not
  // the image placeholder (258880). Stages splice frame embeddings at
  // these positions.
  std::int32_t video_pad_token_id() const noexcept override
  { return _vid_pad; }

  // Audio user turn: <bos>(first)<|turn>user\n <|audio>{pad x N}<audio|>
  // {content} <turn|>\n<|turn>model\n.
  void
  render_user_turn_audio(std::string_view              content,
                         int                           n_audio_tokens,
                         bool                          is_first_turn,
                         std::vector<std::int32_t>*    dst) const override
  {
    emit_turn_open_(is_first_turn, dst);
    (void)push_sp_(_sot, dst);
    append_text_(_tok, "user\n", dst);
    if (n_audio_tokens > 0) {
      (void)push_sp_(_boa, dst);
      if (_aud_pad >= 0) {
        dst->insert(dst->end(), static_cast<std::size_t>(n_audio_tokens),
                    _aud_pad);
      }
      (void)push_sp_(_eoa, dst);
    }
    append_text_(_tok, content, dst);
    (void)push_sp_(_eot, dst);
    append_text_(_tok, "\n", dst);
    emit_model_open_(dst);
  }

  void
  render_user_turn_vlm(std::string_view              content,
                       std::span<const int>          image_token_counts,
                       bool                          is_first_turn,
                       std::vector<std::int32_t>*    dst) const override
  {
    (void)render_vlm_prefix(image_token_counts, is_first_turn, dst);
    (void)render_vlm_completion(content, dst);
  }

  // Mixed in-line turn: text runs, image blocks (boi + pad×n + eoi)
  // and audio blocks (boa + pad×n + eoa) in caller order -- Gemma-4
  // is the tree's only image+audio chat family, so both modalities
  // can share one turn. Rejects (nothing appended) when a requested
  // modality's soft tokens are missing from the tokenizer.
  bool
  render_user_turn_media(std::span<const MediaChunk>   chunks,
                         bool                          is_first_turn,
                         std::vector<std::int32_t>*    dst) const override
  {
    for (const auto& c : chunks) {
      if (c.kind == MediaChunk::Kind::Image && _img_pad < 0) {
        return false;
      }
      if (c.kind == MediaChunk::Kind::Audio
          && (_aud_pad < 0 || _boa < 0 || _eoa < 0)) {
        return false;
      }
    }
    emit_turn_open_(is_first_turn, dst);
    (void)push_sp_(_sot, dst);
    append_text_(_tok, "user\n", dst);
    for (const auto& c : chunks) {
      if (c.kind == MediaChunk::Kind::Text) {
        if (!c.text.empty()) {
          append_text_(_tok, c.text, dst);
        }
      } else if (c.n_tokens > 0) {
        const bool img = c.kind == MediaChunk::Kind::Image;
        (void)push_sp_(img ? _boi : _boa, dst);
        dst->insert(dst->end(), static_cast<std::size_t>(c.n_tokens),
                    img ? _img_pad : _aud_pad);
        (void)push_sp_(img ? _eoi : _eoa, dst);
      }
    }
    (void)push_sp_(_eot, dst);
    append_text_(_tok, "\n", dst);
    emit_model_open_(dst);
    return true;
  }

  // Shared image-block prefix: <bos>(first)<|turn>user\n then one
  // <|image> {pad x n} <image|> block per image. Pairs with
  // render_vlm_completion. The pad placeholders are overlaid with
  // vision embeddings by prefill_multimodal at those positions.
  bool
  render_vlm_prefix(std::span<const int>          image_token_counts,
                    bool                          is_first_turn,
                    std::vector<std::int32_t>*    dst) const override
  {
    emit_turn_open_(is_first_turn, dst);
    (void)push_sp_(_sot, dst);
    append_text_(_tok, "user\n", dst);
    for (int n : image_token_counts) {
      if (n <= 0) {
        continue;
      }
      (void)push_sp_(_boi, dst);
      if (_img_pad >= 0) {
        dst->insert(dst->end(), static_cast<std::size_t>(n), _img_pad);
      }
      (void)push_sp_(_eoi, dst);
    }
    return true;
  }

  bool
  render_vlm_completion(std::string_view              content,
                        std::vector<std::int32_t>*    dst) const override
  {
    append_text_(_tok, content, dst);
    (void)push_sp_(_eot, dst);
    append_text_(_tok, "\n", dst);
    emit_model_open_(dst);
    return true;
  }

  bool
  render_vlm_prefix_ex(
      std::span<const int>          image_token_counts,
      bool                          is_first_turn,
      std::string_view              pre_image_prompt,
      std::string_view              post_image_prompt,
      bool                          close_turn,
      std::vector<std::int32_t>*    dst) const override
  {
    emit_turn_open_(is_first_turn, dst);
    (void)push_sp_(_sot, dst);
    append_text_(_tok, "user\n", dst);
    if (!pre_image_prompt.empty()) {
      append_text_(_tok, pre_image_prompt, dst);
    }
    for (int n : image_token_counts) {
      if (n <= 0) {
        continue;
      }
      (void)push_sp_(_boi, dst);
      if (_img_pad >= 0) {
        dst->insert(dst->end(), static_cast<std::size_t>(n), _img_pad);
      }
      (void)push_sp_(_eoi, dst);
    }
    if (!post_image_prompt.empty()) {
      append_text_(_tok, post_image_prompt, dst);
    }
    if (close_turn) {
      (void)push_sp_(_eot, dst);
      append_text_(_tok, "\n", dst);
      emit_model_open_(dst);
    }
    return true;
  }

  // ---- VLM video surface ------------------------------------------
  // Mirrors the token-exact image render (render_vlm_prefix) but emits
  // the Gemma video block: one `MM:SS ` text marker + `<|image>`(boi)
  // {video_pad x n} `<image|>`(eoi) per frame, frames separated by a
  // single space ( the HF processor's `" ".join(...)` over per-frame
  // `f"{mm:02d}:{ss:02d} {boi}{video_token*n}{eoi}"` ). Pairs with
  // render_vlm_completion for the question + assistant open.
  void
  render_user_turn_video(
      std::string_view              content,
      std::span<const float>        frame_timestamps_seconds,
      std::span<const int>          image_token_counts,
      bool                          is_first_turn,
      std::vector<std::int32_t>*    dst) const override
  {
    render_user_turn_video(content, frame_timestamps_seconds,
                           image_token_counts, is_first_turn,
                           std::string_view{}, dst);
  }

  void
  render_user_turn_video(
      std::string_view              content,
      std::span<const float>        frame_timestamps_seconds,
      std::span<const int>          image_token_counts,
      bool                          is_first_turn,
      std::string_view              pre_image_prompt,
      std::vector<std::int32_t>*    dst) const override
  {
    if (!render_video_prefix(frame_timestamps_seconds,
                             image_token_counts, is_first_turn,
                             pre_image_prompt, dst)) {
      render_user_turn_vlm(content, image_token_counts,
                           is_first_turn, dst);
      return;
    }
    (void)render_vlm_completion(content, dst);
  }

  bool
  render_video_prefix(
      std::span<const float>        frame_timestamps_seconds,
      std::span<const int>          image_token_counts,
      bool                          is_first_turn,
      std::vector<std::int32_t>*    dst) const override
  {
    return render_video_prefix(frame_timestamps_seconds,
                               image_token_counts, is_first_turn,
                               std::string_view{}, dst);
  }

  bool
  render_video_prefix(
      std::span<const float>        frame_timestamps_seconds,
      std::span<const int>          image_token_counts,
      bool                          is_first_turn,
      std::string_view              pre_image_prompt,
      std::vector<std::int32_t>*    dst) const override
  {
    if (frame_timestamps_seconds.size() != image_token_counts.size()) {
      return false;
    }
    emit_turn_open_(is_first_turn, dst);
    (void)push_sp_(_sot, dst);
    append_text_(_tok, "user\n", dst);
    if (!pre_image_prompt.empty()) {
      append_text_(_tok, pre_image_prompt, dst);
    }
    bool first_frame = true;
    for (std::size_t i = 0; i < image_token_counts.size(); ++i) {
      const int n = image_token_counts[i];
      if (n <= 0) {
        continue;
      }
      // " ".join: a single space precedes every frame block but the
      // first. Each block opens with "MM:SS " (trailing space before
      // boi). int(secs/60):int(secs%60) matches the HF reference's
      // int(secs//60):int(secs%60) for secs >= 0.
      const int isec = static_cast<int>(frame_timestamps_seconds[i]);
      char buf[40];
      std::snprintf(buf, sizeof buf, "%s%02d:%02d ",
                    first_frame ? "" : " ", isec / 60, isec % 60);
      append_text_(_tok, buf, dst);
      first_frame = false;
      (void)push_sp_(_boi, dst);
      if (_vid_pad >= 0) {
        dst->insert(dst->end(), static_cast<std::size_t>(n), _vid_pad);
      }
      (void)push_sp_(_eoi, dst);
    }
    return true;
  }

  // Inline audio block (mixes with the video frames in one user turn): an
  // optional caption + <audio_start> {audio_pad x N} <audio_end>.
  bool
  render_audio_block(std::string_view              caption,
                     int                           n_audio_tokens,
                     std::vector<std::int32_t>*    dst) const override
  {
    if (n_audio_tokens <= 0 || _aud_pad < 0 || _boa < 0 || _eoa < 0) {
      return false;
    }
    if (!caption.empty()) {
      append_text_(_tok, caption, dst);
    }
    (void)push_sp_(_boa, dst);
    dst->insert(dst->end(), static_cast<std::size_t>(n_audio_tokens),
                _aud_pad);
    (void)push_sp_(_eoa, dst);
    return true;
  }

  std::int32_t assistant_close_token_id() const override
  {
    // <end_of_turn> closes the assistant turn (the model normally emits
    // it as its own stop token).
    return _eot;
  }

  bool is_stop_token(std::int32_t id) const override
  {
    if (id < 0) {
      return false;
    }
    return id == _eos || id == _eot || id == _tool_resp;
  }

  // Strip the reasoning channel from a decoded turn, mirroring the
  // checkpoint's own `strip_thinking` jinja macro: split on the
  // channel-close marker; for each segment drop everything from a
  // channel-open marker onward, keep the rest; trim. This removes both
  // the `<|channel>`/`<channel|>` markers AND the thought text between
  // them (and a trailing thought left unclosed by a truncated decode),
  // leaving only the user-facing answer.
  std::string sanitize_output(std::string text) const override
  {
    // The stream detokenizer rewrites the channel tokens to the
    // unified vpipe thinking markers, so streamed text carries the
    // marker form; bulk-decoded or externally-sourced text may still
    // carry the raw channel form. Strip both.
    auto strip_blocks = [](std::string in, std::string_view open,
                           std::string_view close) -> std::string {
      if (in.find(open) == std::string::npos) {
        return in;   // common case: nothing to strip
      }
      std::string out;
      out.reserve(in.size());
      const std::string_view tv(in);
      std::size_t start = 0;
      while (true) {
        const std::size_t c = tv.find(close, start);
        const std::string_view seg = (c == std::string_view::npos)
            ? tv.substr(start)
            : tv.substr(start, c - start);
        const std::size_t o = seg.find(open);
        out.append(o == std::string_view::npos ? seg
                                               : seg.substr(0, o));
        if (c == std::string_view::npos) {
          break;
        }
        start = c + close.size();
      }
      return out;
    };
    std::string out = strip_blocks(std::move(text),
                                   "<|channel>", "<channel|>");
    out = strip_blocks(std::move(out), vpipe::media_line::kThinkStart,
                       vpipe::media_line::kThinkEnd);
    const std::size_t b = out.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) {
      return {};
    }
    const std::size_t e = out.find_last_not_of(" \t\r\n");
    return out.substr(b, e - b + 1);
  }

  std::string_view family_name() const override { return "gemma"; }

private:
  const Tokenizer&   _tok;
  const std::int32_t _bos;
  const std::int32_t _sot;
  const std::int32_t _eot;
  const std::int32_t _eos;
  const std::int32_t _tool_resp;
  const std::int32_t _boi;
  const std::int32_t _img_pad;
  const std::int32_t _eoi;
  const std::int32_t _boa;
  const std::int32_t _aud_pad;
  const std::int32_t _eoa;
  const std::int32_t _vid_pad;
  const std::int32_t _think;          // <|think|> (98); -1 if absent
  const std::int32_t _channel_open;   // <|channel> (100)
  const std::int32_t _channel_close;  // <channel|> (101)
  const bool         _thinking_variant;
  const bool         _thinking_on;
};

}

std::unique_ptr<ChatTemplate>
make_chat_template(const std::string&    architecture,
                   const Tokenizer&      tokenizer,
                   std::optional<bool>   disable_thinking)
{
  // Llama family (and ALSO Qwen2 dense / Mistral 7B variants, which
  // all use the Llama-3 header tokens because their tokenizers extend
  // the Llama vocab). Mirror the dispatch table used by
  // make_model_exec_ in loaded-language-model.cc so the chat template
  // tracks the exec class.
  if (architecture == "LlamaForCausalLM"
      || architecture == "MistralForCausalLM") {
    // Llama-3 has no thinking concept; ignore the flag.
    return std::make_unique<Llama3ChatTemplate>(tokenizer);
  }
  // Qwen2 dense uses vanilla ChatML; Qwen3.5 layers the
  // thinking flag on top.
  if (architecture == "Qwen2ForCausalLM") {
    return std::make_unique<ChatMLChatTemplate>(tokenizer);
  }
  if (architecture == "Qwen3ASRForConditionalGeneration") {
    // ASR is single-pass transcription -- no thinking flag.
    (void)disable_thinking;
    return std::make_unique<Qwen3AsrChatTemplate>(tokenizer);
  }
  if (architecture == "Gemma4ForConditionalGeneration") {
    // e4b IS a reasoning checkpoint (its tokenizer carries
    // <|think|>/<|channel>/<channel|> and it intermittently emits a
    // `<|channel>thought ...<channel|>` block, e.g. on multi-part prompts).
    // We deliberately do NOT prefill an empty thought channel to "disable"
    // it: empirically that makes e4b answer in open meta-reasoning ("The
    // user wants me to ...") instead of the actual content. Instead we keep
    // the plain template and strip any thought channel from the decoded
    // output (sanitize_output, matching the checkpoint's own strip_thinking
    // macro). So the thinking flag does not apply at render time here.
    (void)disable_thinking;
    return std::make_unique<GemmaChatTemplate>(tokenizer);
  }
  if (architecture == "Gemma4UnifiedForConditionalGeneration") {
    // 12B gemma4_unified is a reasoning model. Reasoning is ON only when the
    // caller explicitly clears disable_thinking; the checkpoint's canonical
    // default (and an explicit disable) prefills an empty thought channel so
    // the model skips reasoning. Same `<|turn>` format as e4b otherwise.
    const bool thinking_on =
        disable_thinking.has_value() && !disable_thinking.value();
    return std::make_unique<GemmaChatTemplate>(
        tokenizer, /*thinking_variant=*/true, thinking_on);
  }
  if (architecture == "Qwen3_5ForConditionalGeneration"
      || architecture == "Qwen3_5MoeForConditionalGeneration") {
    // VLM-capable when the tokenizer ships vision sentinel tokens.
    // Text-only Qwen3 checkpoints don't have <|vision_start|> in
    // their vocab; we fall back to plain Qwen3ChatTemplate so the
    // text-only flow continues to work.
    const bool is_vlm =
        tokenizer.special_token_id("<|vision_start|>") >= 0
        && tokenizer.special_token_id("<|vision_end|>")   >= 0
        && tokenizer.special_token_id("<|image_pad|>")    >= 0;
    if (is_vlm) {
      // Qwen3-VL family default: thinking-ON.
      return std::make_unique<Qwen3VLChatTemplate>(
          tokenizer, disable_thinking.value_or(false));
    }
    // Qwen3 text-only family default: thinking-OFF.
    return std::make_unique<Qwen3ChatTemplate>(
        tokenizer, disable_thinking.value_or(true));
  }
  return nullptr;
}

}

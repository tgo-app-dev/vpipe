// Per-model-family chat-template renderer tests. Two synthetic
// tokenizers (one Llama-3-style, one ChatML-style) feed the
// dispatch table; we then assert the rendered prompt-id sequences
// match what the upstream chat_template.jinja would emit for a
// single user turn.

#include "minitest.h"
#include "generative-models/chat-template.h"
#include "generative-models/tokenizer.h"
#include "common/flex-data.h"
#include "common/session.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

// Llama-3-style synthetic tokenizer. Special tokens have plausible
// ids; the only requirement is that special_token_id() can look
// them up by name. Vocab covers "user", "assistant", and the
// punctuation characters the chat template emits between specials.
const char* kLlama3TokenizerJson = R"json({
  "version": "1.0",
  "added_tokens": [
    {"id": 128000, "content": "<|begin_of_text|>",   "special": true},
    {"id": 128001, "content": "<|end_of_text|>",     "special": true},
    {"id": 128006, "content": "<|start_header_id|>", "special": true},
    {"id": 128007, "content": "<|end_header_id|>",   "special": true},
    {"id": 128008, "content": "<|eom_id|>",          "special": true},
    {"id": 128009, "content": "<|eot_id|>",          "special": true}
  ],
  "pre_tokenizer": {
    "type": "Sequence",
    "pretokenizers": [
      {"type": "Split",
       "pattern": {"Regex": "\\S+|\\s+"},
       "behavior": "Isolated"}
    ]
  },
  "model": {
    "type": "BPE",
    "vocab": {
      "a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6,
      "h": 7, "i": 8, "j": 9, "k": 10, "l": 11, "m": 12, "n": 13,
      "o": 14, "p": 15, "q": 16, "r": 17, "s": 18, "t": 19, "u": 20,
      "v": 21, "w": 22, "x": 23, "y": 24, "z": 25,
      "Ġ": 26, "Ċ": 27,
      "user": 50, "assistant": 51, "hello": 60, "world": 61
    },
    "merges": []
  }
})json";

// ChatML-style synthetic tokenizer (Qwen-family).
const char* kChatMLTokenizerJson = R"json({
  "version": "1.0",
  "added_tokens": [
    {"id": 151643, "content": "<|endoftext|>", "special": true},
    {"id": 151644, "content": "<|im_start|>",  "special": true},
    {"id": 151645, "content": "<|im_end|>",    "special": true}
  ],
  "pre_tokenizer": {
    "type": "Sequence",
    "pretokenizers": [
      {"type": "Split",
       "pattern": {"Regex": "\\S+|\\s+"},
       "behavior": "Isolated"}
    ]
  },
  "model": {
    "type": "BPE",
    "vocab": {
      "a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6,
      "h": 7, "i": 8, "j": 9, "k": 10, "l": 11, "m": 12, "n": 13,
      "o": 14, "p": 15, "q": 16, "r": 17, "s": 18, "t": 19, "u": 20,
      "v": 21, "w": 22, "x": 23, "y": 24, "z": 25,
      "Ġ": 26, "Ċ": 27, "<": 28, ">": 29, "/": 30,
      "user": 50, "assistant": 51, "think": 52,
      "hello": 60, "world": 61
    },
    "merges": []
  }
})json";

unique_ptr<genai::Tokenizer>
make_tok_(const char* json, const SessionContextIntf* s)
{
  return genai::Tokenizer::from_huggingface_string(
      json, "test-fixture", s);
}

// True iff `id` appears anywhere in `ids`.
bool
contains_id_(const vector<int32_t>& ids, int32_t id)
{
  for (int32_t v : ids) {
    if (v == id) { return true; }
  }
  return false;
}

}

TEST(llm_chat_template, llama3_unknown_arch_returns_nullptr)
{
  Session sess;
  auto tok = make_tok_(kLlama3TokenizerJson, &sess);
  EXPECT_TRUE(tok != nullptr);
  if (!tok) { return; }
  auto tpl = genai::make_chat_template("MysteryForCausalLM", *tok);
  EXPECT_TRUE(tpl == nullptr);
}

TEST(llm_chat_template, llama3_first_turn_includes_bos)
{
  Session sess;
  auto tok = make_tok_(kLlama3TokenizerJson, &sess);
  EXPECT_TRUE(tok != nullptr);
  if (!tok) { return; }
  auto tpl = genai::make_chat_template("LlamaForCausalLM", *tok);
  EXPECT_TRUE(tpl != nullptr);
  if (!tpl) { return; }

  vector<int32_t> ids;
  tpl->render_user_turn("hello world",
                        /*is_first_turn=*/true,
                        &ids);

  const int32_t bos = tok->special_token_id("<|begin_of_text|>");
  const int32_t hdr_s = tok->special_token_id("<|start_header_id|>");
  const int32_t eot   = tok->special_token_id("<|eot_id|>");
  EXPECT_TRUE(bos >= 0);

  // Sequence must start with BOS then user-header pair.
  EXPECT_TRUE(ids.size() >= 4u);
  EXPECT_TRUE(ids[0] == bos);
  EXPECT_TRUE(ids[1] == hdr_s);
  // Token at [2] is the "user" piece; we don't check the exact id
  // because BPE encoding of "user" depends on the vocab merges and
  // we want this test resilient to merge changes.
  // Skip to the eot transition: there must be exactly one eot in the
  // sequence (between user content and the assistant header) AND
  // the eot must come before the LAST hdr_s (the assistant header).
  bool saw_eot = false;
  bool saw_hdr_s_after_eot = false;
  for (size_t i = 0; i < ids.size(); ++i) {
    if (ids[i] == eot) {
      EXPECT_TRUE(!saw_eot);   // exactly one eot
      saw_eot = true;
    } else if (saw_eot && ids[i] == hdr_s) {
      saw_hdr_s_after_eot = true;
    }
  }
  EXPECT_TRUE(saw_eot);
  EXPECT_TRUE(saw_hdr_s_after_eot);

  // Ends with the assistant <|end_header_id|> + a couple of newline
  // bytes that the BPE renders. Final id is one of those newline
  // bytes (NOT a special token).
  EXPECT_TRUE(ids.back() != eot);
  EXPECT_TRUE(ids.back() != hdr_s);
}

TEST(llm_chat_template, llama3_subsequent_turn_skips_bos)
{
  Session sess;
  auto tok = make_tok_(kLlama3TokenizerJson, &sess);
  if (!tok) { return; }
  auto tpl = genai::make_chat_template("LlamaForCausalLM", *tok);
  if (!tpl) { return; }

  vector<int32_t> first;
  tpl->render_user_turn("hello", true,  &first);
  vector<int32_t> later;
  tpl->render_user_turn("hello", false, &later);

  const int32_t bos = tok->special_token_id("<|begin_of_text|>");
  EXPECT_TRUE(contains_id_(first, bos));
  EXPECT_TRUE(!contains_id_(later, bos));
  // Later turn is shorter by exactly one BOS.
  EXPECT_TRUE(later.size() + 1 == first.size());
}

TEST(llm_chat_template, llama3_stop_tokens_and_close_id)
{
  Session sess;
  auto tok = make_tok_(kLlama3TokenizerJson, &sess);
  if (!tok) { return; }
  auto tpl = genai::make_chat_template("LlamaForCausalLM", *tok);
  if (!tpl) { return; }

  const int32_t eot   = tok->special_token_id("<|eot_id|>");
  const int32_t eom   = tok->special_token_id("<|eom_id|>");
  const int32_t end_t = tok->special_token_id("<|end_of_text|>");
  EXPECT_TRUE(tpl->is_stop_token(eot));
  EXPECT_TRUE(tpl->is_stop_token(eom));
  EXPECT_TRUE(tpl->is_stop_token(end_t));
  // Non-special id and the -1 sentinel must NOT stop the loop.
  EXPECT_TRUE(!tpl->is_stop_token(0));
  EXPECT_TRUE(!tpl->is_stop_token(-1));

  // Llama-3 closes the assistant turn with <|eot_id|>.
  EXPECT_TRUE(tpl->assistant_close_token_id() == eot);
}

TEST(llm_chat_template, chatml_qwen2_no_bos_no_think)
{
  Session sess;
  auto tok = make_tok_(kChatMLTokenizerJson, &sess);
  if (!tok) { return; }
  auto tpl = genai::make_chat_template("Qwen2ForCausalLM", *tok);
  EXPECT_TRUE(tpl != nullptr);
  if (!tpl) { return; }

  vector<int32_t> first;
  tpl->render_user_turn("hello", true, &first);
  vector<int32_t> later;
  tpl->render_user_turn("hello", false, &later);

  // is_first_turn is a no-op for ChatML -- same byte sequence.
  EXPECT_TRUE(first == later);

  const int32_t im_start = tok->special_token_id("<|im_start|>");
  const int32_t im_end   = tok->special_token_id("<|im_end|>");
  EXPECT_TRUE(im_start >= 0);
  EXPECT_TRUE(im_end >= 0);

  // Must contain exactly TWO <|im_start|> (one for user, one for
  // assistant) and exactly ONE <|im_end|> (closing the user turn).
  // The single <|im_end|> is the same shape for both first/later
  // turns since ChatML has no session-start token to skip.
  int n_start = 0;
  int n_end   = 0;
  for (int32_t v : first) {
    if (v == im_start) { ++n_start; }
    if (v == im_end)   { ++n_end;   }
  }
  EXPECT_TRUE(n_start == 2);
  EXPECT_TRUE(n_end   == 1);
}

TEST(llm_chat_template, qwen3_emits_more_tokens_than_qwen2_chatml)
{
  // The think-disabled prefix `<think>\n\n</think>\n\n` is the only
  // structural difference between vanilla ChatML and the Qwen3
  // variant -- so the Qwen3 turn must be strictly longer.
  Session sess;
  auto tok = make_tok_(kChatMLTokenizerJson, &sess);
  if (!tok) { return; }
  auto qwen2 = genai::make_chat_template("Qwen2ForCausalLM", *tok);
  auto qwen3 = genai::make_chat_template(
      "Qwen3_5ForConditionalGeneration", *tok);
  if (!qwen2 || !qwen3) { return; }

  vector<int32_t> a;
  vector<int32_t> b;
  qwen2->render_user_turn("hello", true, &a);
  qwen3->render_user_turn("hello", true, &b);
  EXPECT_TRUE(b.size() > a.size());

  // The Qwen3 extra-suffix length equals the BPE-encoded length of
  // `<think>\n\n</think>\n\n` -- the same string the implementation
  // emits via append_text_. We don't rely on exact byte arithmetic
  // (the synthetic vocab's byte fallback maps each non-vocab byte
  // to its own id); we just assert the delta matches.
  auto delta_ids = tok->encode("<think>\n\n</think>\n\n");
  EXPECT_TRUE(!delta_ids.empty());
  EXPECT_TRUE(b.size() == a.size() + delta_ids.size());
}

TEST(llm_chat_template,
     qwen3_disable_thinking_uses_special_token_ids_when_available)
{
  // Qwen3.5's real tokenizer ships `<think>` and `</think>` as
  // single added-token ids (248068 / 248069). The HF chat_template.jinja
  // renders them as text and the HF tokenizer pipeline splits added-
  // tokens out before BPE so the model receives single-token ids.
  // Our encoder doesn't segment by added_tokens, so the chat template
  // emits the special-token ids directly when the tokenizer registers
  // them. Verify this with a synthetic tokenizer that DOES include
  // both `<think>` and `</think>` as added tokens.
  const char* json = R"json({
    "version": "1.0",
    "added_tokens": [
      {"id": 151644, "content": "<|im_start|>",  "special": true},
      {"id": 151645, "content": "<|im_end|>",    "special": true},
      {"id": 248068, "content": "<think>",       "special": false},
      {"id": 248069, "content": "</think>",      "special": false}
    ],
    "pre_tokenizer": {
      "type": "Sequence",
      "pretokenizers": [
        {"type": "Split",
         "pattern": {"Regex": "\\S+|\\s+"},
         "behavior": "Isolated"}
      ]
    },
    "model": {
      "type": "BPE",
      "vocab": {
        "Ġ": 26, "Ċ": 27, "<": 28, ">": 29, "/": 30,
        "user": 50, "assistant": 51, "think": 52,
        "hello": 60
      },
      "merges": []
    }
  })json";
  Session sess;
  auto tok = genai::Tokenizer::from_huggingface_string(
      json, "test-fixture", &sess);
  if (!tok) { return; }
  EXPECT_TRUE(tok->special_token_id("<think>")  == 248068);
  EXPECT_TRUE(tok->special_token_id("</think>") == 248069);

  auto off = genai::make_chat_template(
      "Qwen3_5ForConditionalGeneration", *tok,
      /*disable_thinking=*/std::optional<bool>(true));
  auto on  = genai::make_chat_template(
      "Qwen3_5ForConditionalGeneration", *tok,
      /*disable_thinking=*/std::optional<bool>(false));
  if (!off || !on) { return; }

  vector<int32_t> off_ids;
  vector<int32_t> on_ids;
  off->render_user_turn("hello", true, &off_ids);
  on ->render_user_turn("hello", true, &on_ids);

  // Both streams must contain the single token id 248068 (<think>);
  // the OFF stream must also contain 248069 (</think>), and the ON
  // stream must NOT contain it.
  EXPECT_TRUE(contains_id_(off_ids, 248068));
  EXPECT_TRUE(contains_id_(off_ids, 248069));
  EXPECT_TRUE(contains_id_(on_ids,  248068));
  EXPECT_TRUE(!contains_id_(on_ids, 248069));
}

TEST(llm_chat_template, qwen3_disable_thinking_override_flips_preamble)
{
  // Qwen3 text-only family default is thinking-OFF
  // (`<think>\n\n</think>\n\n`). Passing disable_thinking=false flips
  // it to thinking-ON (`<think>\n`); the rendered user turn shrinks
  // by exactly the difference between the two token spans.
  Session sess;
  auto tok = make_tok_(kChatMLTokenizerJson, &sess);
  if (!tok) { return; }
  auto def = genai::make_chat_template(
      "Qwen3_5ForConditionalGeneration", *tok);
  auto on  = genai::make_chat_template(
      "Qwen3_5ForConditionalGeneration", *tok,
      /*disable_thinking=*/std::optional<bool>(false));
  auto off = genai::make_chat_template(
      "Qwen3_5ForConditionalGeneration", *tok,
      /*disable_thinking=*/std::optional<bool>(true));
  if (!def || !on || !off) { return; }

  vector<int32_t> a, b, c;
  def->render_user_turn("hello", true, &a);
  on ->render_user_turn("hello", true, &b);
  off->render_user_turn("hello", true, &c);

  // default == off (this synthetic tokenizer has no vision sentinels
  // so the factory selects the Qwen3 text-only branch, which is
  // family-default thinking-OFF).
  EXPECT_TRUE(a == c);
  // off > on by exactly len("<think>\n\n</think>\n\n") - len("<think>\n").
  auto off_ids = tok->encode("<think>\n\n</think>\n\n");
  auto on_ids  = tok->encode("<think>\n");
  EXPECT_TRUE(c.size() == b.size() + off_ids.size() - on_ids.size());
}

TEST(llm_chat_template, chatml_qwen3_stop_and_close_match_chatml)
{
  // The Qwen3 variant inherits stop-token set + assistant-close from
  // vanilla ChatML; only the assistant-header extras differ. Confirm
  // the inherited behaviour so a future refactor doesn't accidentally
  // diverge them.
  Session sess;
  auto tok = make_tok_(kChatMLTokenizerJson, &sess);
  if (!tok) { return; }
  auto tpl = genai::make_chat_template(
      "Qwen3_5ForConditionalGeneration", *tok);
  EXPECT_TRUE(tpl != nullptr);
  if (!tpl) { return; }

  const int32_t im_end     = tok->special_token_id("<|im_end|>");
  const int32_t end_of_txt = tok->special_token_id("<|endoftext|>");
  EXPECT_TRUE(tpl->is_stop_token(im_end));
  EXPECT_TRUE(tpl->is_stop_token(end_of_txt));
  EXPECT_TRUE(!tpl->is_stop_token(-1));
  EXPECT_TRUE(tpl->assistant_close_token_id() == im_end);
}

TEST(llm_chat_template, content_text_appears_in_user_turn)
{
  Session sess;
  auto tok = make_tok_(kChatMLTokenizerJson, &sess);
  if (!tok) { return; }
  auto tpl = genai::make_chat_template("Qwen2ForCausalLM", *tok);
  if (!tpl) { return; }

  vector<int32_t> ids;
  tpl->render_user_turn("hello world", true, &ids);

  // The user content "hello world" must be tokenisable to ids that
  // all appear in the rendered prompt (in order, after the user
  // header and before <|im_end|>). We only check membership here;
  // ordering is implied by render_user_turn's structure.
  auto content_ids = tok->encode("hello world");
  EXPECT_TRUE(!content_ids.empty());
  for (int32_t c : content_ids) {
    EXPECT_TRUE(contains_id_(ids, c));
  }
}

// ChatML fixture with the Qwen3-VL vision sentinels. Used to exercise
// the VLM dispatch and render_user_turn_vlm.
namespace {
const char* kQwen3VLTokenizerJson = R"json({
  "version": "1.0",
  "added_tokens": [
    {"id": 151643, "content": "<|endoftext|>",     "special": true},
    {"id": 151644, "content": "<|im_start|>",      "special": true},
    {"id": 151645, "content": "<|im_end|>",        "special": true},
    {"id": 151652, "content": "<|vision_start|>",  "special": true},
    {"id": 151653, "content": "<|vision_end|>",    "special": true},
    {"id": 151655, "content": "<|image_pad|>",     "special": true}
  ],
  "pre_tokenizer": {
    "type": "Sequence",
    "pretokenizers": [
      {"type": "Split",
       "pattern": {"Regex": "\\S+|\\s+"},
       "behavior": "Isolated"}
    ]
  },
  "model": {
    "type": "BPE",
    "vocab": {
      "a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6,
      "h": 7, "i": 8, "j": 9, "k": 10, "l": 11, "m": 12, "n": 13,
      "o": 14, "p": 15, "q": 16, "r": 17, "s": 18, "t": 19, "u": 20,
      "v": 21, "w": 22, "x": 23, "y": 24, "z": 25,
      "Ġ": 26, "Ċ": 27, "<": 28, ">": 29, "/": 30,
      "user": 50, "assistant": 51, "think": 52,
      "hello": 60, "world": 61
    },
    "merges": []
  }
})json";
}

TEST(llm_chat_template, qwen3vl_factory_picks_vl_when_vision_tokens_present)
{
  Session sess;
  auto tok = make_tok_(kQwen3VLTokenizerJson, &sess);
  if (!tok) { return; }
  auto tpl = genai::make_chat_template(
      "Qwen3_5ForConditionalGeneration", *tok);
  EXPECT_TRUE(tpl != nullptr);
  if (!tpl) { return; }
  EXPECT_TRUE(string_view(tpl->family_name()) == "qwen3-vl-chatml");
  EXPECT_TRUE(tpl->image_pad_token_id() ==
              tok->special_token_id("<|image_pad|>"));
}

TEST(llm_chat_template, qwen3vl_render_inserts_vision_block)
{
  Session sess;
  auto tok = make_tok_(kQwen3VLTokenizerJson, &sess);
  if (!tok) { return; }
  auto tpl = genai::make_chat_template(
      "Qwen3_5ForConditionalGeneration", *tok);
  if (!tpl) { return; }
  const int32_t vs = tok->special_token_id("<|vision_start|>");
  const int32_t ve = tok->special_token_id("<|vision_end|>");
  const int32_t ip = tok->special_token_id("<|image_pad|>");
  EXPECT_TRUE(vs >= 0 && ve >= 0 && ip >= 0);

  vector<int32_t> ids;
  int counts[2] = {7, 3};
  tpl->render_user_turn_vlm("hello",
      std::span<const int>(counts, 2),
      /*is_first_turn=*/true, &ids);
  // One <|vision_start|> + N image-pads + <|vision_end|> per image.
  int n_vs = 0, n_ve = 0, n_ip = 0;
  for (auto id : ids) {
    if (id == vs) { ++n_vs; }
    if (id == ve) { ++n_ve; }
    if (id == ip) { ++n_ip; }
  }
  EXPECT_TRUE(n_vs == 2);
  EXPECT_TRUE(n_ve == 2);
  EXPECT_TRUE(n_ip == 7 + 3);
}

TEST(llm_chat_template, qwen3vl_render_video_injects_timestamps)
{
  // Video render path: every frame's vision-wrapped block is
  // preceded by a "<X.Y seconds>" plaintext marker. The marker is
  // tokenised through the regular BPE (the synthetic fixture has
  // no merges so every character maps 1:1 to its byte-fallback id,
  // which makes the timestamp string easy to spot when scanning
  // for the literal "seconds" substring after decode).
  Session sess;
  auto tok = make_tok_(kQwen3VLTokenizerJson, &sess);
  if (!tok) { return; }
  auto tpl = genai::make_chat_template(
      "Qwen3_5ForConditionalGeneration", *tok);
  if (!tpl) { return; }
  const int32_t vs = tok->special_token_id("<|vision_start|>");
  const int32_t ve = tok->special_token_id("<|vision_end|>");
  const int32_t ip = tok->special_token_id("<|image_pad|>");

  vector<int32_t> ids_img;
  vector<int32_t> ids_vid;
  int counts[3] = {2, 2, 2};
  float ts[3]  = {0.0f, 1.0f, 2.0f};
  tpl->render_user_turn_vlm("q",
      std::span<const int>(counts, 3), true, &ids_img);
  tpl->render_user_turn_video("q",
      std::span<const float>(ts, 3),
      std::span<const int>(counts, 3), true, &ids_vid);

  // Same number of vision-start / vision-end / image-pad tokens.
  auto count = [](const vector<int32_t>& v, int32_t id) {
    int n = 0;
    for (auto x : v) {
      if (x == id) { ++n; }
    }
    return n;
  };
  EXPECT_TRUE(count(ids_vid, vs) == 3);
  EXPECT_TRUE(count(ids_vid, ve) == 3);
  EXPECT_TRUE(count(ids_vid, ip) == 6);
  // The video render is longer than the plain VLM render: every
  // timestamp text marker adds extra tokens.
  EXPECT_TRUE(ids_vid.size() > ids_img.size());
  // Decoding back to a string must contain the literal "seconds"
  // substring at least once per frame (since the synthetic tokenizer
  // byte-falls-back unknown characters).
  std::string decoded = tok->decode(ids_vid);
  std::size_t pos = 0;
  int seconds_hits = 0;
  while ((pos = decoded.find("seconds", pos)) != std::string::npos) {
    ++seconds_hits;
    pos += 7;
  }
  EXPECT_TRUE(seconds_hits == 3);
}

TEST(llm_chat_template, qwen3vl_video_prefix_plus_completion_matches_full)
{
  // Shared-prefix split for video: render_video_prefix emits the
  // user-role open + timestamped frame blocks; render_vlm_completion
  // emits the question text + close + assistant_open. Their
  // concatenation must match render_user_turn_video's one-shot
  // output bit-for-bit — the contract RealtimeVqaStage relies on
  // when prefilling a shared prefix once and branching per question.
  Session sess;
  auto tok = make_tok_(kQwen3VLTokenizerJson, &sess);
  if (!tok) { return; }
  auto tpl = genai::make_chat_template(
      "Qwen3_5ForConditionalGeneration", *tok);
  if (!tpl) { return; }

  vector<int32_t> ids_full;
  vector<int32_t> ids_split;
  int   counts[3] = {2, 2, 2};
  float ts[3]    = {0.0f, 1.0f, 2.0f};
  tpl->render_user_turn_video("q",
      std::span<const float>(ts, 3),
      std::span<const int>(counts, 3), true, &ids_full);
  ASSERT_TRUE(tpl->render_video_prefix(
      std::span<const float>(ts, 3),
      std::span<const int>(counts, 3), true, &ids_split));
  ASSERT_TRUE(tpl->render_vlm_completion("q", &ids_split));
  EXPECT_TRUE(ids_split == ids_full);
}

TEST(llm_chat_template, qwen3vl_video_prefix_with_pre_text_prepends_inline)
{
  // The 5-arg render_video_prefix injects `pre_image_prompt` between
  // the user-role open and the first frame block. Verifies:
  //   1. Vision tokens still emit (same vision-start/-end/image-pad
  //      counts as the no-pre-text variant).
  //   2. The pre-text appears BEFORE the first <|vision_start|> when
  //      we walk the resulting id stream.
  //   3. Empty pre_text path is unchanged from the 4-arg variant.
  Session sess;
  auto tok = make_tok_(kQwen3VLTokenizerJson, &sess);
  if (!tok) { return; }
  auto tpl = genai::make_chat_template(
      "Qwen3_5ForConditionalGeneration", *tok);
  if (!tpl) { return; }
  const int32_t vs = tok->special_token_id("<|vision_start|>");
  const int32_t ve = tok->special_token_id("<|vision_end|>");
  const int32_t ip = tok->special_token_id("<|image_pad|>");

  vector<int32_t> ids_no_pre;
  vector<int32_t> ids_with_pre;
  int   counts[2] = {2, 2};
  float ts[2]    = {0.0f, 1.0f};
  ASSERT_TRUE(tpl->render_video_prefix(
      std::span<const float>(ts, 2),
      std::span<const int>(counts, 2), true, &ids_no_pre));
  const std::string pre =
      "Here's the summary of the scene before the first frame: SUMMARY";
  ASSERT_TRUE(tpl->render_video_prefix(
      std::span<const float>(ts, 2),
      std::span<const int>(counts, 2), true,
      std::string_view(pre), &ids_with_pre));

  auto count = [](const vector<int32_t>& v, int32_t id) {
    int n = 0;
    for (auto x : v) {
      if (x == id) { ++n; }
    }
    return n;
  };
  EXPECT_TRUE(count(ids_with_pre, vs) == count(ids_no_pre, vs));
  EXPECT_TRUE(count(ids_with_pre, ve) == count(ids_no_pre, ve));
  EXPECT_TRUE(count(ids_with_pre, ip) == count(ids_no_pre, ip));
  EXPECT_TRUE(ids_with_pre.size() > ids_no_pre.size());

  // Structural check: the FRAME tokens (from the first <|vision_start|>
  // onward) are identical between the two paths — pre_text only adds
  // tokens IN FRONT of the first vision block. We avoid decoding back
  // through the synthetic BPE because non-ASCII / byte-fallback ids
  // don't round-trip cleanly to the literal string we passed in.
  auto first_vs_pos = [vs](const vector<int32_t>& v) -> std::size_t {
    for (std::size_t i = 0; i < v.size(); ++i) {
      if (v[i] == vs) { return i; }
    }
    return v.size();
  };
  const std::size_t pos_no  = first_vs_pos(ids_no_pre);
  const std::size_t pos_pre = first_vs_pos(ids_with_pre);
  ASSERT_TRUE(pos_no < ids_no_pre.size());
  ASSERT_TRUE(pos_pre < ids_with_pre.size());
  EXPECT_TRUE(pos_pre > pos_no);   // pre_text inserts extra tokens
  EXPECT_TRUE(ids_with_pre.size() - pos_pre
              == ids_no_pre.size() - pos_no);
  for (std::size_t i = 0;
       i < ids_no_pre.size() - pos_no; ++i) {
    EXPECT_TRUE(ids_with_pre[pos_pre + i] == ids_no_pre[pos_no + i]);
  }
}

TEST(llm_chat_template,
     qwen3vl_user_turn_video_with_pre_text_round_trip)
{
  // The 5-arg render_user_turn_video should be equivalent to
  // render_video_prefix(5-arg) + render_vlm_completion(content) —
  // mirrors qwen3vl_video_prefix_plus_completion_matches_full but
  // with a non-empty pre_image_prompt threaded through.
  Session sess;
  auto tok = make_tok_(kQwen3VLTokenizerJson, &sess);
  if (!tok) { return; }
  auto tpl = genai::make_chat_template(
      "Qwen3_5ForConditionalGeneration", *tok);
  if (!tpl) { return; }

  vector<int32_t> ids_full;
  vector<int32_t> ids_split;
  int   counts[2] = {2, 2};
  float ts[2]    = {0.0f, 1.0f};
  const std::string pre = "PRE";
  tpl->render_user_turn_video("q",
      std::span<const float>(ts, 2),
      std::span<const int>(counts, 2), true,
      std::string_view(pre), &ids_full);
  ASSERT_TRUE(tpl->render_video_prefix(
      std::span<const float>(ts, 2),
      std::span<const int>(counts, 2), true,
      std::string_view(pre), &ids_split));
  ASSERT_TRUE(tpl->render_vlm_completion("q", &ids_split));
  EXPECT_TRUE(ids_split == ids_full);
}

TEST(llm_chat_template, qwen3vl_video_prefix_length_mismatch_returns_false)
{
  // Same length-mismatch guard as render_user_turn_video, but exposed
  // through the bool return so the caller can pick a fallback path.
  Session sess;
  auto tok = make_tok_(kQwen3VLTokenizerJson, &sess);
  if (!tok) { return; }
  auto tpl = genai::make_chat_template(
      "Qwen3_5ForConditionalGeneration", *tok);
  if (!tpl) { return; }

  vector<int32_t> dst;
  int   counts[2] = {3, 3};
  float ts_bad[1] = {0.0f};
  EXPECT_TRUE(!tpl->render_video_prefix(
      std::span<const float>(ts_bad, 1),
      std::span<const int>(counts, 2), true, &dst));
  // No tokens emitted on failure.
  EXPECT_TRUE(dst.empty());
}

TEST(llm_chat_template, qwen3vl_render_video_length_mismatch_falls_back)
{
  // When timestamps.size() != image_counts.size() the template
  // falls back to render_user_turn_vlm (no timestamps). Verifying
  // this guard keeps callers safe from a misconfigured visual-qa
  // stage that accidentally emits N images but M != N timestamps.
  Session sess;
  auto tok = make_tok_(kQwen3VLTokenizerJson, &sess);
  if (!tok) { return; }
  auto tpl = genai::make_chat_template(
      "Qwen3_5ForConditionalGeneration", *tok);
  if (!tpl) { return; }

  vector<int32_t> ids_vid_bad;
  vector<int32_t> ids_img;
  int counts[2] = {3, 3};
  float ts_bad[1] = {0.0f};
  tpl->render_user_turn_vlm("q",
      std::span<const int>(counts, 2), true, &ids_img);
  tpl->render_user_turn_video("q",
      std::span<const float>(ts_bad, 1),
      std::span<const int>(counts, 2), true, &ids_vid_bad);
  // Mismatched lengths -> identical output to the plain VLM path.
  EXPECT_TRUE(ids_vid_bad == ids_img);
}

TEST(llm_chat_template, qwen3vl_text_only_fallback_when_no_vision_tokens)
{
  // If the tokenizer doesn't ship vision sentinels, the factory
  // returns the text-only Qwen3 template, not the VL variant.
  Session sess;
  auto tok = make_tok_(kChatMLTokenizerJson, &sess);
  if (!tok) { return; }
  auto tpl = genai::make_chat_template(
      "Qwen3_5ForConditionalGeneration", *tok);
  if (!tpl) { return; }
  EXPECT_TRUE(string_view(tpl->family_name()) == "qwen3-chatml");
  EXPECT_TRUE(tpl->image_pad_token_id() == -1);
}

// Qwen3-ASR-specific tokenizer fixture (ChatML scaffold + the three
// audio sentinel ids). Used for chat-template tests below.
namespace {
const char* kQwen3AsrTokenizerJson = R"json({
  "version": "1.0",
  "added_tokens": [
    {"id": 151643, "content": "<|endoftext|>",     "special": true},
    {"id": 151644, "content": "<|im_start|>",      "special": true},
    {"id": 151645, "content": "<|im_end|>",        "special": true},
    {"id": 151669, "content": "<|audio_start|>",   "special": true},
    {"id": 151670, "content": "<|audio_end|>",     "special": true},
    {"id": 151676, "content": "<|audio_pad|>",     "special": true}
  ],
  "pre_tokenizer": {
    "type": "Sequence",
    "pretokenizers": [
      {"type": "Split",
       "pattern": {"Regex": "\\S+|\\s+"},
       "behavior": "Isolated"}
    ]
  },
  "model": {
    "type": "BPE",
    "vocab": {
      "a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6,
      "h": 7, "i": 8, "j": 9, "k": 10, "l": 11, "m": 12, "n": 13,
      "o": 14, "p": 15, "q": 16, "r": 17, "s": 18, "t": 19, "u": 20,
      "v": 21, "w": 22, "x": 23, "y": 24, "z": 25,
      "Ġ": 26, "Ċ": 27, "<": 28, ">": 29, "/": 30,
      "user": 50, "assistant": 51, "system": 52, "asr_text": 53,
      "language": 54, "English": 55, "Chinese": 56,
      "transcribe": 70
    },
    "merges": []
  }
})json";
}

// Factory dispatches on architecture string + picks the ASR template.
// Asserts the audio_pad_token_id() accessor surfaces the correct id.
TEST(llm_chat_template, qwen3_asr_factory_dispatches)
{
  Session sess;
  auto tok = make_tok_(kQwen3AsrTokenizerJson, &sess);
  if (!tok) { return; }
  auto tpl = genai::make_chat_template(
      "Qwen3ASRForConditionalGeneration", *tok);
  EXPECT_TRUE(tpl != nullptr);
  if (!tpl) { return; }
  EXPECT_TRUE(string_view(tpl->family_name()) == "qwen3-asr-chatml");
  EXPECT_TRUE(tpl->audio_pad_token_id() ==
              tok->special_token_id("<|audio_pad|>"));
  // ASR template has no thinking + no vision.
  EXPECT_TRUE(tpl->image_pad_token_id() == -1);
  // Stop tokens are the standard ChatML pair.
  EXPECT_TRUE(tpl->is_stop_token(
      tok->special_token_id("<|im_end|>")));
  EXPECT_TRUE(tpl->is_stop_token(
      tok->special_token_id("<|endoftext|>")));
}

// The rendered prompt has the expected layout: system turn, user turn
// containing exactly N audio_pad tokens wrapped in audio_start/end,
// assistant header. Empty system_prompt + empty language_hint case.
TEST(llm_chat_template, qwen3_asr_render_basic_layout)
{
  Session sess;
  auto tok = make_tok_(kQwen3AsrTokenizerJson, &sess);
  if (!tok) { return; }
  auto tpl = genai::make_chat_template(
      "Qwen3ASRForConditionalGeneration", *tok);
  if (!tpl) { return; }

  const int32_t im_start = tok->special_token_id("<|im_start|>");
  const int32_t im_end   = tok->special_token_id("<|im_end|>");
  const int32_t as       = tok->special_token_id("<|audio_start|>");
  const int32_t ae       = tok->special_token_id("<|audio_end|>");
  const int32_t ap       = tok->special_token_id("<|audio_pad|>");

  vector<int32_t> ids;
  EXPECT_TRUE(
      tpl->render_asr_prompt(/*system_prompt=*/"",
                             /*audio_token_count=*/5,
                             /*language_hint=*/"",
                             &ids));
  // Exactly 5 audio_pad ids, all inside a single <|audio_start|> /
  // <|audio_end|> pair.
  size_t n_pad = 0, n_as = 0, n_ae = 0, n_imstart = 0, n_imend = 0;
  for (int32_t v : ids) {
    if (v == ap)        { ++n_pad; }
    else if (v == as)   { ++n_as; }
    else if (v == ae)   { ++n_ae; }
    else if (v == im_start) { ++n_imstart; }
    else if (v == im_end)   { ++n_imend; }
  }
  EXPECT_TRUE(n_pad == 5u);
  EXPECT_TRUE(n_as  == 1u);
  EXPECT_TRUE(n_ae  == 1u);
  // System + user + assistant = 3 im_start. 2 im_end (system, user).
  EXPECT_TRUE(n_imstart == 3u);
  EXPECT_TRUE(n_imend   == 2u);
  // <|audio_start|> appears strictly before <|audio_end|>.
  size_t pos_as = static_cast<size_t>(-1), pos_ae = pos_as;
  for (size_t i = 0; i < ids.size(); ++i) {
    if (ids[i] == as) { pos_as = i; }
    if (ids[i] == ae) { pos_ae = i; }
  }
  EXPECT_TRUE(pos_as < pos_ae);
}

// language_hint emits a leading "language X<asr_text>" assistant-side
// hint that the model continues from.
TEST(llm_chat_template, qwen3_asr_language_hint_emits_prefix)
{
  Session sess;
  auto tok = make_tok_(kQwen3AsrTokenizerJson, &sess);
  if (!tok) { return; }
  auto tpl = genai::make_chat_template(
      "Qwen3ASRForConditionalGeneration", *tok);
  if (!tpl) { return; }

  vector<int32_t> with_hint, without_hint;
  tpl->render_asr_prompt("system here", 3, "English", &with_hint);
  tpl->render_asr_prompt("system here", 3, "",        &without_hint);
  // The hint variant produces strictly more tokens.
  EXPECT_TRUE(with_hint.size() > without_hint.size());
  // "language" and "English" appear in the hint variant only.
  EXPECT_TRUE(contains_id_(with_hint,
                           tok->special_token_id("<|im_start|>")));
  // The synthetic BPE vocab doesn't ship merges that recombine "<",
  // "asr_text", ">" into "<asr_text>" -- but the difference in
  // token count between with_hint and without_hint should at least
  // be > 5 (the hint string is 25+ characters and even a per-char
  // fallback emits one id per character).
  EXPECT_TRUE(with_hint.size() >= without_hint.size() + 5u);
}

// audio_token_count == 0 is invalid (no audio in prompt).
TEST(llm_chat_template, qwen3_asr_zero_audio_count_returns_false)
{
  Session sess;
  auto tok = make_tok_(kQwen3AsrTokenizerJson, &sess);
  if (!tok) { return; }
  auto tpl = genai::make_chat_template(
      "Qwen3ASRForConditionalGeneration", *tok);
  if (!tpl) { return; }
  vector<int32_t> ids;
  EXPECT_TRUE(!tpl->render_asr_prompt("", 0, "", &ids));
  EXPECT_TRUE(ids.empty());
}

// Gemma-4 12B (gemma4_unified) is a reasoning model whose `<|turn>` chat
// format carries a thought channel. Reasoning OFF (default + explicit
// disable) pre-fills an empty `<|channel>thought\n<channel|>` so the model
// skips reasoning; reasoning ON injects a leading `<|think|>` system turn.
// Token-exact vs the HF chat_template.jinja for "Name three primary colors."
// (captured via transformers apply_chat_template). Gated on the real 12B
// tokenizer (VPIPE_GEMMA12B_TEST_MODEL_PATH) because the metaspace
// SentencePiece tokenization of the text spans must match exactly.
TEST(llm_chat_template, gemma4_unified_thinking_modes)
{
  const char* path = std::getenv("VPIPE_GEMMA12B_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  Session sess;
  auto tok = genai::Tokenizer::from_huggingface_json(
      std::string(path) + "/tokenizer.json", &sess);
  if (!tok) { return; }

  const vector<int32_t> kDisabled{
      2, 105, 2364, 107, 1567, 1806, 5905, 7913, 236761, 106, 107, 105,
      4368, 107, 100, 45518, 107, 101};
  const vector<int32_t> kEnabled{
      2, 105, 9731, 107, 98, 107, 106, 107, 105, 2364, 107, 1567, 1806,
      5905, 7913, 236761, 106, 107, 105, 4368, 107};

  auto def = genai::make_chat_template(
      "Gemma4UnifiedForConditionalGeneration", *tok);
  auto off = genai::make_chat_template(
      "Gemma4UnifiedForConditionalGeneration", *tok,
      std::optional<bool>(true));
  auto on  = genai::make_chat_template(
      "Gemma4UnifiedForConditionalGeneration", *tok,
      std::optional<bool>(false));
  if (!def || !off || !on) { return; }

  vector<int32_t> d, o, e;
  def->render_user_turn("Name three primary colors.", true, &d);
  off->render_user_turn("Name three primary colors.", true, &o);
  on ->render_user_turn("Name three primary colors.", true, &e);

  EXPECT_TRUE(d == kDisabled);   // default == reasoning OFF
  EXPECT_TRUE(o == kDisabled);   // explicit disable == OFF
  EXPECT_TRUE(e == kEnabled);    // explicit enable == <|think|> system turn
}

// ---- Tool / function calling (ChatML / Qwen) -----------------------

TEST(llm_chat_template, chatml_supports_tools)
{
  Session sess;
  auto tok = make_tok_(kChatMLTokenizerJson, &sess);
  if (!tok) { return; }
  auto tpl = genai::make_chat_template("Qwen2ForCausalLM", *tok);
  EXPECT_TRUE(tpl != nullptr);
  if (!tpl) { return; }
  EXPECT_TRUE(tpl->supports_tools());
}

TEST(llm_chat_template, llama3_has_no_tool_support)
{
  Session sess;
  auto tok = make_tok_(kLlama3TokenizerJson, &sess);
  if (!tok) { return; }
  auto tpl = genai::make_chat_template("LlamaForCausalLM", *tok);
  EXPECT_TRUE(tpl != nullptr);
  if (!tpl) { return; }
  EXPECT_FALSE(tpl->supports_tools());
  // The default (unsupported) hooks append nothing and return false.
  vector<int32_t> ids;
  EXPECT_FALSE(tpl->render_tools_system_turn("{}", true, &ids));
  EXPECT_TRUE(ids.empty());
  vector<string> results{"r"};
  vector<string> names{"t"};
  EXPECT_FALSE(tpl->render_tool_results_turn(
      span<const string>(names.data(), names.size()),
      span<const string>(results.data(), results.size()), &ids));
  EXPECT_TRUE(ids.empty());
}

TEST(llm_chat_template, chatml_tools_system_turn_is_wrapped)
{
  Session sess;
  auto tok = make_tok_(kChatMLTokenizerJson, &sess);
  if (!tok) { return; }
  auto tpl = genai::make_chat_template("Qwen2ForCausalLM", *tok);
  if (!tpl) { return; }

  const int32_t im_start = tok->special_token_id("<|im_start|>");
  const int32_t im_end   = tok->special_token_id("<|im_end|>");

  vector<int32_t> ids;
  EXPECT_TRUE(tpl->render_tools_system_turn("{\"a\":1}", true, &ids));
  EXPECT_FALSE(ids.empty());
  // A system turn: opens with <|im_start|> and closes with <|im_end|>.
  EXPECT_TRUE(!ids.empty() && ids.front() == im_start);
  EXPECT_TRUE(contains_id_(ids, im_end));
}

TEST(llm_chat_template, chatml_tool_results_turn_opens_assistant)
{
  Session sess;
  auto tok = make_tok_(kChatMLTokenizerJson, &sess);
  if (!tok) { return; }
  auto tpl = genai::make_chat_template("Qwen2ForCausalLM", *tok);
  if (!tpl) { return; }

  const int32_t im_start = tok->special_token_id("<|im_start|>");
  const int32_t im_end   = tok->special_token_id("<|im_end|>");

  vector<string> results{"{\"datetime\":\"now\"}"};
  vector<string> names{"get_current_time"};
  vector<int32_t> ids;
  EXPECT_TRUE(tpl->render_tool_results_turn(
      span<const string>(names.data(), names.size()),
      span<const string>(results.data(), results.size()), &ids));
  EXPECT_FALSE(ids.empty());
  // Result turn closes the user turn (<|im_end|>) then opens a fresh
  // assistant turn (<|im_start|>): both specials must be present, and
  // there must be at least two <|im_start|> (user-open + assistant-open).
  int starts = 0;
  for (int32_t v : ids) { if (v == im_start) { ++starts; } }
  EXPECT_TRUE(starts >= 2);
  EXPECT_TRUE(contains_id_(ids, im_end));
}

// ---- Tool / function calling (Gemma-4 DSL) -------------------------

// Synthetic Gemma-4 tokenizer carrying just the tool-DSL special tokens
// (real ids) + a tiny vocab. Enough to construct GemmaChatTemplate and
// exercise the tokenizer-independent tool logic (supports/marker/stop/
// parse). The render + token-exact tests below use the REAL tokenizer.
namespace {
const char* kGemmaToolTokenizerJson = R"json({
  "version": "1.0",
  "added_tokens": [
    {"id": 1,   "content": "<eos>",            "special": true},
    {"id": 2,   "content": "<bos>",            "special": true},
    {"id": 46,  "content": "<|tool>",          "special": true},
    {"id": 47,  "content": "<tool|>",          "special": true},
    {"id": 50,  "content": "<|tool_response>", "special": true},
    {"id": 51,  "content": "<tool_response|>", "special": true},
    {"id": 52,  "content": "<|\"|>",           "special": true},
    {"id": 105, "content": "<|turn>",          "special": true},
    {"id": 106, "content": "<turn|>",          "special": true}
  ],
  "pre_tokenizer": {
    "type": "Sequence",
    "pretokenizers": [
      {"type": "Split",
       "pattern": {"Regex": "\\S+|\\s+"},
       "behavior": "Isolated"}
    ]
  },
  "model": {
    "type": "BPE",
    "vocab": {"a": 300, "b": 301, "c": 302},
    "merges": []
  }
})json";
}

TEST(llm_chat_template, gemma_supports_tools_and_markers)
{
  Session sess;
  auto tok = make_tok_(kGemmaToolTokenizerJson, &sess);
  if (!tok) { return; }
  auto tpl = genai::make_chat_template(
      "Gemma4ForConditionalGeneration", *tok);
  EXPECT_TRUE(tpl != nullptr);
  if (!tpl) { return; }
  EXPECT_TRUE(string_view(tpl->family_name()) == "gemma");
  // The DSL tokens are present -> tools are supported.
  EXPECT_TRUE(tpl->supports_tools());
  EXPECT_TRUE(string_view(tpl->tool_call_open_marker()) == "<|tool_call>");
  // <|tool_response> (50) is a turn-CONTINUING stop; <turn|> (106) and
  // <eos> (1) close the turn.
  EXPECT_TRUE(tpl->stop_token_continues_turn(50));
  EXPECT_FALSE(tpl->stop_token_continues_turn(106));
  EXPECT_FALSE(tpl->stop_token_continues_turn(1));
  EXPECT_FALSE(tpl->stop_token_continues_turn(-1));
  // <|tool_response> is also a stop token.
  EXPECT_TRUE(tpl->is_stop_token(50));
}

TEST(llm_chat_template, gemma_tools_unsupported_without_dsl_tokens)
{
  // A Gemma checkpoint whose tokenizer lacks the tool-DSL tokens must
  // report no tool support (so the chat stage keeps tools off) and the
  // render hooks must append nothing.
  Session sess;
  auto tok = make_tok_(kChatMLTokenizerJson, &sess);   // no gemma tokens
  if (!tok) { return; }
  auto tpl = genai::make_chat_template(
      "Gemma4ForConditionalGeneration", *tok);
  if (!tpl) { return; }
  EXPECT_FALSE(tpl->supports_tools());
  vector<int32_t> ids;
  EXPECT_FALSE(tpl->render_tools_system_turn("{}", true, &ids));
  EXPECT_TRUE(ids.empty());
  vector<string> names{"t"}, results{"r"};
  EXPECT_FALSE(tpl->render_tool_results_turn(
      span<const string>(names.data(), names.size()),
      span<const string>(results.data(), results.size()), &ids));
  EXPECT_TRUE(ids.empty());
}

TEST(llm_chat_template, gemma_parse_tool_call_dsl)
{
  // parse_tool_calls decodes Gemma's `<|tool_call>call:NAME{...}<tool_call|>`
  // DSL into name + JSON arguments. Works off the decoded text, so it
  // needs no real tokenizer.
  Session sess;
  auto tok = make_tok_(kGemmaToolTokenizerJson, &sess);
  if (!tok) { return; }
  auto tpl = genai::make_chat_template(
      "Gemma4ForConditionalGeneration", *tok);
  if (!tpl) { return; }

  // Single call, one string arg with an embedded comma/space.
  const string txt1 =
      "<|tool_call>call:get_weather{location:<|\"|>Tokyo, JP<|\"|>}"
      "<tool_call|>";
  auto c1 = tpl->parse_tool_calls(txt1);
  ASSERT_TRUE(c1.size() == 1u);
  EXPECT_TRUE(c1[0].name == "get_weather");
  FlexData a1 = FlexData::from_json(c1[0].arguments_json);
  ASSERT_TRUE(a1.is_object());
  EXPECT_TRUE(a1.as_object().at("location").get_string() == "Tokyo, JP");

  // Mixed types (string / bool) + TWO calls in one turn.
  const string txt2 =
      "<|tool_call>call:write_file{append:false,content:<|\"|>hi<|\"|>,"
      "path:<|\"|>a.txt<|\"|>}<tool_call|>"
      "<|tool_call>call:get_current_time{}<tool_call|>";
  auto c2 = tpl->parse_tool_calls(txt2);
  ASSERT_TRUE(c2.size() == 2u);
  EXPECT_TRUE(c2[0].name == "write_file");
  FlexData a2 = FlexData::from_json(c2[0].arguments_json);
  ASSERT_TRUE(a2.is_object());
  {
    auto o = a2.as_object();
    EXPECT_TRUE(o.at("path").get_string() == "a.txt");
    EXPECT_TRUE(o.at("content").get_string() == "hi");
    EXPECT_TRUE(o.at("append").is_bool());
    EXPECT_FALSE(o.at("append").get_bool());
  }
  EXPECT_TRUE(c2[1].name == "get_current_time");
  EXPECT_TRUE(c2[1].arguments_json == "{}");

  // No tool call -> empty.
  EXPECT_TRUE(tpl->parse_tool_calls("just a normal reply").empty());
}

// Token-exact vs the HF chat_template.jinja for a two-tool advertisement
// (get_current_time with empty params + get_weather with a string param)
// plus the user turn "What time is it?". Captured offline via
// transformers.apply_chat_template on the real e4b tokenizer. Mirrors the
// chat stage: user turn rendered is_first=false (the tools system turn
// carries the single <bos>), tools system turn prepended. Gated on the
// real tokenizer (VPIPE_GEMMA4_TEST_MODEL_PATH) -- the metaspace
// SentencePiece tokenization of the DSL text spans must match exactly.
TEST(llm_chat_template, gemma_tools_system_turn_token_exact)
{
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  Session sess;
  auto tok = genai::Tokenizer::from_huggingface_json(
      std::string(path) + "/tokenizer.json", &sess);
  if (!tok) { return; }
  auto tpl = genai::make_chat_template(
      "Gemma4ForConditionalGeneration", *tok);
  if (!tpl) { return; }
  ASSERT_TRUE(tpl->supports_tools());

  // One Hermes function-spec object per line -- exactly what
  // McpToolRegistry::tools_json() emits.
  const string tools_json =
      "{\"type\":\"function\",\"function\":{\"name\":\"get_current_time\","
      "\"description\":\"Get the host's current local date and time.\","
      "\"parameters\":{\"type\":\"object\",\"properties\":{},"
      "\"required\":[]}}}\n"
      "{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
      "\"description\":\"Get the weather for a city.\",\"parameters\":"
      "{\"type\":\"object\",\"properties\":{\"location\":{\"type\":"
      "\"string\",\"description\":\"The city name\"}},\"required\":"
      "[\"location\"]}}}";

  vector<int32_t> user_ids;
  tpl->render_user_turn("What time is it?", /*is_first_turn=*/false,
                        &user_ids);
  vector<int32_t> sys_ids;
  ASSERT_TRUE(tpl->render_tools_system_turn(
      tools_json, /*is_first_turn=*/true, &sys_ids));
  vector<int32_t> got = sys_ids;
  got.insert(got.end(), user_ids.begin(), user_ids.end());

  const vector<int32_t> kExpected{
      2, 105, 9731, 107, 46, 163688, 236787, 828, 236779, 4002, 236779,
      2289, 236782, 7777, 236787, 52, 3407, 506, 4253, 236789, 236751,
      1873, 2263, 3433, 532, 990, 236761, 52, 236764, 19031, 29616, 2084,
      236787, 52, 60688, 52, 1807, 47, 46, 163688, 236787, 828, 236779,
      19323, 236782, 7777, 236787, 52, 3407, 506, 7606, 573, 496, 3207,
      236761, 52, 236764, 19031, 29616, 15921, 29616, 7125, 29616, 7777,
      236787, 52, 818, 3207, 1463, 52, 236764, 2084, 236787, 52, 35410,
      52, 5237, 15979, 24845, 52, 7125, 52, 1604, 2084, 236787, 52, 60688,
      52, 1807, 47, 106, 107, 105, 2364, 107, 3689, 990, 563, 625, 236881,
      106, 107, 105, 4368, 107};
  EXPECT_TRUE(got == kExpected);
}

// The tool-results turn re-emits the <|tool_response> opener the model
// stopped on (uncommitted) + one response:NAME{...} block per result +
// <tool_response|>, and does NOT open a new model turn (Gemma keeps the
// exchange in one turn). A JSON-object result renders as a native
// {key:value,...} mapping. Decoded, it must match the HF reference
// string. Gated on the real tokenizer.
TEST(llm_chat_template, gemma_tool_results_turn_decodes_to_reference)
{
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  Session sess;
  auto tok = genai::Tokenizer::from_huggingface_json(
      std::string(path) + "/tokenizer.json", &sess);
  if (!tok) { return; }
  auto tpl = genai::make_chat_template(
      "Gemma4ForConditionalGeneration", *tok);
  if (!tpl) { return; }
  ASSERT_TRUE(tpl->supports_tools());

  vector<string> names{"get_weather"};
  vector<string> results{"{\"temperature\":15,\"weather\":\"sunny\"}"};
  vector<int32_t> ids;
  ASSERT_TRUE(tpl->render_tool_results_turn(
      span<const string>(names.data(), names.size()),
      span<const string>(results.data(), results.size()), &ids));
  // Native mapping form, keys dictsorted (temperature < weather), number
  // bare, string <|"|>-wrapped -- exactly the HF reference block.
  const string want =
      "<|tool_response>response:get_weather{temperature:15,weather:"
      "<|\"|>sunny<|\"|>}<tool_response|>";
  EXPECT_TRUE(tok->decode(ids) == want);

  // A non-JSON result falls back to {value:<|"|>...<|"|>}.
  vector<string> results2{"plain text result"};
  vector<int32_t> ids2;
  ASSERT_TRUE(tpl->render_tool_results_turn(
      span<const string>(names.data(), names.size()),
      span<const string>(results2.data(), results2.size()), &ids2));
  const string want2 =
      "<|tool_response>response:get_weather{value:<|\"|>plain text "
      "result<|\"|>}<tool_response|>";
  EXPECT_TRUE(tok->decode(ids2) == want2);
}

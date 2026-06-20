// Tokenizer round-trip tests. The fixture is a tiny BPE tokenizer
// constructed inline (no external file). It covers:
//
//   * Encoding ASCII bytes through the pre-tokenizer + byte-level +
//     BPE merge path, looking up resulting pieces in the vocab.
//   * Decoding the same ids back to the input string.
//   * Streaming decode of multi-byte UTF-8 sequences split across
//     tokens.
//   * Special token lookup and inline-during-decode rendering.
//
// The Llama-3.1-style \p{L}/\p{N} pre-tokenizer regex is out of
// scope for v1 (std::regex doesn't support Unicode categories);
// this fixture uses ECMAScript-compatible patterns the encoder
// supports today.

#include "minitest.h"
#include "generative-models/tokenizer.h"
#include "stages/qwen-asr-tokenizer.h"
#include "common/session.h"

#include <cstdint>
#include <string>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

// Synthetic tokenizer.json. Vocab contains:
//   * each single byte-level codepoint we'll need (a, b, c, h, e, l,
//     o, w, r, d, Ġ for U+0120/space, Ã for 0xC3, ± for 0xB1)
//   * a few merged tokens reachable by the merges below ("ab",
//     "abc", "he", "ll", "hell", "hello", "Ġh", "Ġhello")
//
// Merges are applied in priority order (lower index wins). The
// chain to reach "hello" from ["h","e","l","l","o"] is:
//   h e  -> he
//   l l  -> ll
//   he ll -> hell
//   hell o -> hello
// And the chain to reach "Ġhello" from [Ġ, h, e, l, l, o]:
//   Ġ h     -> Ġh   (priority 6)
//   ... but "Ġh ello" requires "ello" to exist first, which we
// don't bother with -- after Ġh, the pre-tokenizer's prior split
// has already separated " " from "hello", so "hello" stays one
// chunk and gets merged independently.
const char* kTokenizerJson = R"json({
  "version": "1.0",
  "added_tokens": [
    {"id": 100, "content": "<|begin_of_text|>", "special": true},
    {"id": 101, "content": "<|end_of_text|>",   "special": true}
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
      "a": 0, "b": 1, "c": 2, "h": 3, "e": 4, "l": 5, "o": 6,
      "w": 7, "r": 8, "d": 9,
      "Ġ": 10,
      "Ã": 11, "±": 12,
      "ab": 13, "abc": 14,
      "he": 15, "ll": 16, "hell": 17, "hello": 18,
      "Ġh": 19, "world": 20
    },
    "merges": [
      "a b",
      "ab c",
      "h e",
      "l l",
      "he ll",
      "hell o",
      "Ġ h",
      "w o",
      "wo r",
      "wor l",
      "worl d"
    ]
  }
})json";

unique_ptr<genai::Tokenizer>
make_(const SessionContextIntf* s)
{
  return genai::Tokenizer::from_huggingface_string(
      kTokenizerJson, "test-fixture", s);
}

}

TEST(llm_tokenizer, parses_synthetic_fixture)
{
  Session sess;
  auto tok = make_(&sess);
  EXPECT_TRUE(tok != nullptr);
  if (!tok) { return; }
  // 21 regular vocab entries (ids 0..20) + 2 special (ids 100, 101)
  // means the high-water id is 101, so vocab_size is 102.
  EXPECT_TRUE(tok->vocab_size() == 102);
  EXPECT_TRUE(tok->merge_count() == 11u);
  EXPECT_TRUE(tok->special_token_count() == 2u);
  EXPECT_TRUE(tok->has_pre_tokenizer());
}

// Integration: the native (no-transformers) Qwen3-ASR tokenizer.json
// synthesizer must produce a file the REAL Tokenizer parses + round-
// trips. Feed it the same vocab/merges/special-tokens as the hand-
// written fixture above, just split into the slow-tokenizer inputs the
// Qwen3-ASR repos actually ship (vocab.json + merges.txt +
// tokenizer_config.json).
TEST(llm_tokenizer, synthesized_qwen_tokenizer_round_trips)
{
  const char* vocab_json = R"json({
      "a": 0, "b": 1, "c": 2, "h": 3, "e": 4, "l": 5, "o": 6,
      "w": 7, "r": 8, "d": 9, "Ġ": 10, "Ã": 11, "±": 12,
      "ab": 13, "abc": 14, "he": 15, "ll": 16, "hell": 17,
      "hello": 18, "Ġh": 19, "world": 20
  })json";
  const char* merges_txt =
      "#version: 0.2\n"
      "a b\nab c\nh e\nl l\nhe ll\nhell o\nĠ h\nw o\nwo r\n"
      "wor l\nworl d\n";
  const char* cfg_json = R"json({
    "added_tokens_decoder": {
      "100": {"content": "<|begin_of_text|>", "special": true},
      "101": {"content": "<|end_of_text|>",   "special": true}
    }
  })json";

  std::string err;
  std::string js =
      build_qwen_asr_tokenizer_json(vocab_json, merges_txt, cfg_json, err);
  EXPECT_TRUE(err.empty());

  Session sess;
  auto tok = genai::Tokenizer::from_huggingface_string(js, "synth", &sess);
  EXPECT_TRUE(tok != nullptr);
  if (!tok) { return; }

  // Vocab / merges / specials all carried through (same as the fixture).
  EXPECT_TRUE(tok->vocab_size() == 102);
  EXPECT_TRUE(tok->merge_count() == 11u);
  EXPECT_TRUE(tok->special_token_count() == 2u);

  // BPE still merges to the expected single tokens, and round-trips.
  auto ids = tok->encode("abc");
  EXPECT_TRUE(ids.size() == 1u && ids[0] == 14);
  EXPECT_TRUE(tok->decode(ids) == "abc");
  ids = tok->encode("hello");
  EXPECT_TRUE(ids.size() == 1u && ids[0] == 18);
  EXPECT_TRUE(tok->decode(tok->encode("hello world")) == "hello world");

  // Special token resolvable by name.
  EXPECT_TRUE(tok->special_token_id("<|begin_of_text|>") == 100);
}

TEST(llm_tokenizer, ascii_round_trip_through_bpe)
{
  Session sess;
  auto tok = make_(&sess);
  EXPECT_TRUE(tok != nullptr);
  if (!tok) { return; }

  auto ids = tok->encode("abc");
  EXPECT_TRUE(ids.size() == 1u);
  EXPECT_TRUE(ids[0] == 14);   // merged "abc"

  string back = tok->decode(ids);
  EXPECT_TRUE(back == "abc");

  // "hello" merges all the way to a single token.
  ids = tok->encode("hello");
  EXPECT_TRUE(ids.size() == 1u);
  EXPECT_TRUE(ids[0] == 18);
  EXPECT_TRUE(tok->decode(ids) == "hello");
}

TEST(llm_tokenizer, pre_tokenizer_splits_on_whitespace)
{
  Session sess;
  auto tok = make_(&sess);
  EXPECT_TRUE(tok != nullptr);
  if (!tok) { return; }

  // The fixture's pre-tokenizer (\S+|\s+) splits " hello" into
  // [" ", "hello"]. " " encodes byte-level as Ġ (id 10).
  auto ids = tok->encode(" hello");
  EXPECT_TRUE(ids.size() == 2u);
  EXPECT_TRUE(ids[0] == 10);   // Ġ
  EXPECT_TRUE(ids[1] == 18);   // hello
  EXPECT_TRUE(tok->decode(ids) == " hello");

  // Symmetric for "hello world": "hello", " ", "world".
  ids = tok->encode("hello world");
  EXPECT_TRUE(ids.size() == 3u);
  EXPECT_TRUE(ids[0] == 18);
  EXPECT_TRUE(ids[1] == 10);
  EXPECT_TRUE(ids[2] == 20);
  EXPECT_TRUE(tok->decode(ids) == "hello world");
}

TEST(llm_tokenizer, multibyte_utf8_round_trip)
{
  Session sess;
  auto tok = make_(&sess);
  EXPECT_TRUE(tok != nullptr);
  if (!tok) { return; }

  // "ñ" = U+00F1 = UTF-8 0xC3 0xB1. Byte-level maps each byte to
  // its own codepoint (both bytes are in the self-mapping range
  // 0xAE..0xFF), so the BPE sees ["Ã", "±"]. No merge applies; ids
  // are vocab["Ã"]=11, vocab["±"]=12.
  const string input = "\xC3\xB1";  // "ñ"
  auto ids = tok->encode(input);
  EXPECT_TRUE(ids.size() == 2u);
  EXPECT_TRUE(ids[0] == 11);
  EXPECT_TRUE(ids[1] == 12);
  EXPECT_TRUE(tok->decode(ids) == input);
}

TEST(llm_tokenizer, streaming_decoder_holds_partial_codepoint)
{
  Session sess;
  auto tok = make_(&sess);
  EXPECT_TRUE(tok != nullptr);
  if (!tok) { return; }

  // Feed the two halves of "ñ" one at a time. The first byte alone
  // is an incomplete UTF-8 sequence, so step() must return "" and
  // buffer the partial; the second byte completes the codepoint
  // and step() returns the full "ñ".
  auto sd = tok->make_stream_decoder();
  string chunk1 = tok->step(sd, 11);
  EXPECT_TRUE(chunk1.empty());

  string chunk2 = tok->step(sd, 12);
  EXPECT_TRUE(chunk2 == "\xC3\xB1");
}

TEST(llm_tokenizer, streaming_decoder_matches_bulk_decode)
{
  Session sess;
  auto tok = make_(&sess);
  EXPECT_TRUE(tok != nullptr);
  if (!tok) { return; }

  vector<int32_t> ids = tok->encode("hello world");
  string bulk = tok->decode(ids);

  string streamed;
  auto sd = tok->make_stream_decoder();
  for (auto id : ids) {
    streamed += tok->step(sd, id);
  }
  // After all ids the pending buffer must be empty (every codepoint
  // is complete by the end).
  EXPECT_TRUE(sd.pending.empty());
  EXPECT_TRUE(streamed == bulk);
  EXPECT_TRUE(streamed == "hello world");
}

TEST(llm_tokenizer, special_token_id_lookup)
{
  Session sess;
  auto tok = make_(&sess);
  EXPECT_TRUE(tok != nullptr);
  if (!tok) { return; }

  EXPECT_TRUE(tok->special_token_id("<|begin_of_text|>") == 100);
  EXPECT_TRUE(tok->special_token_id("<|end_of_text|>")   == 101);
  EXPECT_TRUE(tok->special_token_id("<|not_a_token|>")   == -1);
}

TEST(llm_tokenizer, special_tokens_render_literal_in_decode)
{
  Session sess;
  auto tok = make_(&sess);
  EXPECT_TRUE(tok != nullptr);
  if (!tok) { return; }

  // Caller prepends BOS explicitly; the encoder does NOT recognise
  // "<|begin_of_text|>" as a single special token in input text.
  vector<int32_t> ids;
  ids.push_back(tok->special_token_id("<|begin_of_text|>"));
  for (auto id : tok->encode("hello")) { ids.push_back(id); }
  ids.push_back(tok->special_token_id("<|end_of_text|>"));

  string out = tok->decode(ids);
  EXPECT_TRUE(out == "<|begin_of_text|>hello<|end_of_text|>");
}

TEST(llm_tokenizer, empty_input_round_trips)
{
  Session sess;
  auto tok = make_(&sess);
  EXPECT_TRUE(tok != nullptr);
  if (!tok) { return; }

  auto ids = tok->encode("");
  EXPECT_TRUE(ids.empty());
  EXPECT_TRUE(tok->decode(ids).empty());
}

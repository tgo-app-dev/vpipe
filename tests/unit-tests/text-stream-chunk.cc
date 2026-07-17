#include "minitest.h"
#include "common/text-stream-chunk.h"

#include <cstddef>
#include <string>
#include <vector>

using namespace vpipe;
namespace ts = vpipe::text_stream;

namespace {
// An English sentence of `n` space-joined words, then `tail` verbatim
// (e.g. "." or "").
std::string words_(int n, const char* tail) {
  std::string s;
  for (int i = 0; i < n; ++i) {
    if (i) { s += ' '; }
    s += "word";
  }
  s += tail;
  return s;
}
// The UTF-8 string `u` repeated `n` times.
std::string repeat_(const char* u, int n) {
  std::string s;
  for (int i = 0; i < n; ++i) { s += u; }
  return s;
}

// The unified thinking markers (as media-line.h emits) and the two tool-call
// families the stage arms MetaFilter with.
constexpr const char* kTS = "<|__vpipe_think_start__|>";
constexpr const char* kTE = "<|__vpipe_think_end__|>";
constexpr const char* kQOpen  = "<tool_call>";
constexpr const char* kQClose = "</tool_call>";
constexpr const char* kGOpen  = "<|tool_call>";
constexpr const char* kGClose = "<tool_call|>";

// Feed `chunks` through the filter and return the concatenated visible text.
std::string run_filter_(ts::MetaFilter& f,
                        const std::vector<std::string>& chunks) {
  std::string out;
  auto sink = [&out](const std::string& s) { out += s; };
  for (const auto& c : chunks) { f.feed(c, sink); }
  f.flush(sink);
  return out;
}

// Feed `input` one BYTE at a time (markers straddle every boundary).
std::string run_filter_bytes_(ts::MetaFilter& f, const std::string& input) {
  std::string out;
  auto sink = [&out](const std::string& s) { out += s; };
  for (char ch : input) { f.feed(std::string(1, ch), sink); }
  f.flush(sink);
  return out;
}
}  // namespace

TEST(meta_filter, passthrough_when_no_markers) {
  ts::MetaFilter f(kTS, kTE, kQOpen, kQClose);
  EXPECT_TRUE(run_filter_(f, {"Hello ", "world."}) == "Hello world.");
}

TEST(meta_filter, removes_thinking_block) {
  ts::MetaFilter f(kTS, kTE, kQOpen, kQClose);
  const std::string out = run_filter_(
      f, {kTS, "let me reason about it", kTE, "The answer is 42."});
  EXPECT_TRUE(out == "The answer is 42.");
}

TEST(meta_filter, removes_thinking_split_across_bytes) {
  // The markers are cut at every byte boundary; the block must still be
  // recognised and removed, and NO visible byte may be lost.
  ts::MetaFilter f(kTS, kTE, kQOpen, kQClose);
  const std::string input =
      std::string("Before. ") + kTS + "secret plan" + kTE + "Visible answer.";
  EXPECT_TRUE(run_filter_bytes_(f, input) == "Before. Visible answer.");
}

TEST(meta_filter, removes_qwen_tool_call) {
  ts::MetaFilter f(kTS, kTE, kQOpen, kQClose);
  const std::string out = run_filter_(
      f, {"Sure.", kQOpen, "{\"name\":\"get_time\"}", kQClose, "Done."});
  EXPECT_TRUE(out == "Sure.Done.");
}

TEST(meta_filter, removes_gemma_tool_call) {
  ts::MetaFilter f(kTS, kTE, kGOpen, kGClose);
  const std::string out = run_filter_bytes_(
      f, std::string("A") + kGOpen + "call:f{}" + kGClose + "B");
  EXPECT_TRUE(out == "AB");
}

TEST(meta_filter, unterminated_block_runs_to_end) {
  // A reasoning block with no close token runs to end-of-message and is
  // dropped; the visible text before it survives.
  ts::MetaFilter f(kTS, kTE, kQOpen, kQClose);
  const std::string out =
      run_filter_(f, {"Visible ", kTS, "thinking with no close"});
  EXPECT_TRUE(out == "Visible ");
}

TEST(meta_filter, disarmed_tool_markers_not_suppressed) {
  // Empty tool markers (tools off): a literal tool-call marker in prose is
  // NOT suppressed. Only the thinking span is armed.
  ts::MetaFilter f(kTS, kTE, "", "");
  const std::string in = "code like <tool_call> stays </tool_call> visible";
  EXPECT_TRUE(run_filter_bytes_(f, in) == in);
}

TEST(meta_filter, false_start_marker_prefix_is_emitted) {
  // A run that begins like a marker prefix but never completes must be
  // emitted verbatim (held-back bytes are never lost).
  ts::MetaFilter f(kTS, kTE, kQOpen, kQClose);
  const std::string out = run_filter_(f, {"I paid $5 <|", "__ not a marker"});
  EXPECT_TRUE(out == "I paid $5 <|__ not a marker");
}

TEST(meta_filter, mixed_think_and_tool) {
  ts::MetaFilter f(kTS, kTE, kQOpen, kQClose);
  const std::string input =
      std::string(kTS) + "reason" + kTE + "Part one. " + kQOpen + "json"
      + kQClose + "Part two.";
  // Both chunked and byte-split must yield the same visible answer.
  ts::MetaFilter g(kTS, kTE, kQOpen, kQClose);
  EXPECT_TRUE(run_filter_(f, {input}) == "Part one. Part two.");
  EXPECT_TRUE(run_filter_bytes_(g, input) == "Part one. Part two.");
}

TEST(text_stream_chunk, count_words_english) {
  EXPECT_TRUE(ts::count_words("") == 0);
  EXPECT_TRUE(ts::count_words("hello") == 1);
  EXPECT_TRUE(ts::count_words("hello world") == 2);
  EXPECT_TRUE(ts::count_words("  hello,   world!  ") == 2);
  // Apostrophe stays inside the word.
  EXPECT_TRUE(ts::count_words("it's a test") == 3);
  // Punctuation-only string has no words.
  EXPECT_TRUE(ts::count_words("...!?") == 0);
}

TEST(text_stream_chunk, count_words_chinese) {
  // Each Han character is one word; punctuation is not counted.
  EXPECT_TRUE(ts::count_words(repeat_("字", 5)) == 5);
  EXPECT_TRUE(ts::count_words("你好世界。") == 4);
  // Mixed English + Chinese: 2 Han + 1 Latin word.
  EXPECT_TRUE(ts::count_words("你好 world") == 3);
}

TEST(text_stream_chunk, ends_at_break) {
  EXPECT_TRUE(ts::ends_at_break("Hello world."));
  EXPECT_TRUE(ts::ends_at_break("Hello world.   "));   // ignores trailing space
  EXPECT_TRUE(ts::ends_at_break("Hello, "));
  EXPECT_FALSE(ts::ends_at_break("Hello world"));
  EXPECT_FALSE(ts::ends_at_break(""));
  // Chinese full-width punctuation.
  EXPECT_TRUE(ts::ends_at_break("你好世界。"));
  EXPECT_TRUE(ts::ends_at_break("你好，"));
  EXPECT_FALSE(ts::ends_at_break("你好世界"));
}

TEST(text_stream_chunk, chunker_flushes_at_word_target_and_punct) {
  // 19 words + period: below the word target -> not ready.
  ts::Chunker c(20, 60);
  EXPECT_FALSE(c.push(words_(19, ".")));
  EXPECT_FALSE(c.empty());
  (void)c.take();
  EXPECT_TRUE(c.empty());

  // 20 words WITHOUT punctuation: past target but no break -> not ready;
  // adding a period then crosses the boundary.
  ts::Chunker c2(20, 60);
  EXPECT_FALSE(c2.push(words_(20, "")));
  EXPECT_TRUE(c2.push("."));
}

TEST(text_stream_chunk, chunker_hard_cap_without_punct) {
  // 40 words, no punctuation at all -> forced flush at the hard cap.
  ts::Chunker c(20, 40);
  EXPECT_TRUE(c.push(words_(40, "")));
}

TEST(text_stream_chunk, chunker_chinese) {
  // 20 Han characters then a full-width stop -> ready (target + break).
  ts::Chunker c(20, 60);
  EXPECT_FALSE(c.push(repeat_("字", 19)));
  EXPECT_TRUE(c.push("字。"));
}

TEST(text_stream_chunk, streaming_reassembles_input) {
  // Feed a multi-sentence English + Chinese text one UTF-8 codepoint at a
  // time; the concatenation of every flushed chunk plus the final tail
  // must equal the original, and several sentences must produce several
  // flushed beats.
  const std::string input =
      "The quick brown fox jumps over the lazy dog. "
      "Pack my box with five dozen liquor jugs, then rest a while. "
      "你好世界，这是一个中文测试。再来一句更长的话来触发切分吧。";
  ts::Chunker c(8, 40);
  std::string              reassembled;
  std::vector<std::string> flushed;
  for (std::size_t i = 0; i < input.size(); ) {
    std::size_t j = i;
    (void)ts::next_codepoint(input, j);   // advance exactly one codepoint
    if (c.push(input.substr(i, j - i))) {
      flushed.push_back(c.take());
    }
    i = j;
  }
  if (!c.empty()) { flushed.push_back(c.take()); }
  for (const auto& f : flushed) { reassembled += f; }
  EXPECT_TRUE(reassembled == input);
  EXPECT_TRUE(flushed.size() >= 2);
}

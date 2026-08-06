// The SentencePiece Unigram tokenizer, token-exact against the reference.
//
// Token-exact is the only bar worth setting here, and not out of
// strictness. Unigram is a SCORING model: it picks the segmentation whose
// token log-probabilities sum highest. A wrong implementation therefore
// does not emit garbage -- it emits a different, entirely plausible
// segmentation of the same string, and the model is then conditioned on
// something subtly other than the prompt. There is no rel-L2 that catches
// that; either the ids match or they do not.
//
// The cases in the golden are chosen for the parts that can differ:
// strings where greedy longest-match and Viterbi disagree (the reason the
// model cannot be approximated with a longest-match scan at all), runs of
// spaces (the T5 normalizer collapses them), leading and trailing spaces
// and the empty string (where the metaspace prepend rule decides the
// first token), CJK and emoji (where a byte-stepping fallback would split
// a character), and unknown characters (which take the unk penalty).
//
// Env: VPIPE_WAN_TEST_MODEL_PATH = the Wan model root,
// VPIPE_WAN_TOKENIZER_GOLDEN = the dir the dumper wrote. Skips if unset.

#include "minitest.h"

#include "common/flex-data.h"
#include "common/session.h"
#include "generative-models/tokenizer.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;

TEST(unigram_tokenizer, matches_reference_ids)
{
  const char* root = std::getenv("VPIPE_WAN_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_WAN_TOKENIZER_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  std::ifstream in(std::string(gd) + "/unigram_ids.json");
  if (!in) { return; }             // golden not dumped -> skip
  FlexData man;
  try {
    man = FlexData::from_json(in);
  } catch (...) {
    return;
  }
  ASSERT_TRUE(man.is_object());

  Session sess;
  auto tok = Tokenizer::from_huggingface_json(
      std::string(root) + "/tokenizer/tokenizer.json", &sess);
  // The load itself is half the test: before Unigram was supported this
  // returned null with "unsupported model.type='Unigram'", which is what
  // took the whole Wan conditioner offline.
  ASSERT_TRUE(tok != nullptr);

  auto mo = man.as_object();
  ASSERT_TRUE(mo.contains("cases"));
  FlexData cases_fd = mo.at("cases");
  ASSERT_TRUE(cases_fd.is_array());
  auto cases = cases_fd.as_array();

  int n_cases = 0, n_bad = 0, n_ids = 0;
  for (std::size_t i = 0; i < cases.size(); ++i) {
    FlexData c = cases.at(i);
    if (!c.is_object()) { continue; }
    auto co = c.as_object();
    const std::string text(co.at("text").as_string(""));
    auto want_of = [&](const char* key) {
      std::vector<std::int32_t> v;
      if (!co.contains(key)) { return v; }
      FlexData fd = co.at(key);
      if (!fd.is_array()) { return v; }
      auto a = fd.as_array();
      v.reserve(a.size());
      for (std::size_t k = 0; k < a.size(); ++k) {
        v.push_back((std::int32_t)a.at(k).as_int(-1));
      }
      return v;
    };
    auto check = [&](const char* what, const std::vector<std::int32_t>& got,
                     const std::vector<std::int32_t>& want) {
      bool ok = got.size() == want.size();
      for (std::size_t k = 0; ok && k < got.size(); ++k) {
        if (got[k] != want[k]) { ok = false; }
      }
      if (!ok) {
        ++n_bad;
        std::string g, w;
        for (auto v : got)  { g += std::to_string(v) + " "; }
        for (auto v : want) { w += std::to_string(v) + " "; }
        std::printf("[unigram] MISMATCH (%s) \"%s\"\n   got  %s\n"
                    "   want %s\n", what, text.c_str(), g.c_str(), w.c_str());
      }
    };
    const std::vector<std::int32_t> want_raw = want_of("ids");
    const std::vector<std::int32_t> want_cln = want_of("ids_clean");
    ++n_cases;
    n_ids += (int)want_raw.size() + (int)want_cln.size();
    // (a) the tokenizer alone must reproduce tokenizer.json's own
    //     pipeline exactly -- that is the contract of the file.
    check("raw", tok->encode(text), want_raw);
    // (b) the tokenizer BEHIND whitespace_clean must reproduce what the
    //     reference's callers get, which is what actually conditions the
    //     model. The two differ on whitespace, and it is the second that
    //     the Wan conditioner is wired to produce.
    check("cleaned", tok->encode(Tokenizer::whitespace_clean(text)),
          want_cln);
  }
  std::printf("[unigram] %d/%d cases token-exact (%d reference ids)\n",
              n_cases - n_bad, n_cases, n_ids);
  EXPECT_TRUE(n_cases > 0);
  EXPECT_TRUE(n_bad == 0);

  // The two ids the conditioner reaches for by name. Both come from
  // added_tokens rather than the scored vocab, so they exercise a
  // different lookup than everything above.
  EXPECT_TRUE(tok->special_token_id("</s>") ==
              (std::int32_t)mo.at("eos_id").as_int(-1));
  EXPECT_TRUE(tok->special_token_id("<pad>") ==
              (std::int32_t)mo.at("pad_id").as_int(-1));
}

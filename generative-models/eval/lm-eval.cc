#include "generative-models/eval/lm-eval.h"

#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "generative-models/loaded-language-model.h"
#include "generative-models/tokenizer.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/ui-delegate-intf.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <vector>

namespace vpipe::genai {

namespace {

// The model's beginning-of-sequence token id, or -1 if it has none. Gemma /
// SentencePiece models are BOS-sensitive (a missing <bos> produces garbage
// next-token distributions); ChatML / Qwen has no BOS (returns -1, no prepend).
int
bos_id_(const Tokenizer& tok)
{
  for (const char* n : {"<bos>", "<|begin_of_text|>", "<s>"}) {
    const int id = tok.special_token_id(n);
    if (id >= 0) { return id; }
  }
  return -1;
}

// log_softmax(logits)[tok] = logits[tok] - logsumexp(logits). Double-accumulated.
double
log_prob_at_(const std::vector<float>& logits, int tok)
{
  if (tok < 0 || tok >= (int)logits.size()) { return -1e30; }
  float m = logits[0];
  for (float v : logits) { if (v > m) { m = v; } }
  double s = 0.0;
  for (float v : logits) { s += std::exp((double)v - (double)m); }
  return ((double)logits[tok] - (double)m) - std::log(s);
}

// Sorted list of rows-*.json (else any *.json) page files in a dataset dir.
std::vector<std::filesystem::path>
dataset_pages_(const std::string& dir)
{
  namespace fs = std::filesystem;
  std::set<fs::path> rows, other;
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(dir, ec)) {
    if (ec) { break; }
    if (!e.is_regular_file()) { continue; }
    const fs::path& p = e.path();
    if (p.extension() != ".json") { continue; }
    if (p.filename().string().rfind("rows-", 0) == 0) { rows.insert(p); }
    else { other.insert(p); }
  }
  std::vector<fs::path> out;
  if (!rows.empty()) { out.assign(rows.begin(), rows.end()); }
  else { out.assign(other.begin(), other.end()); }
  return out;
}

std::string
read_file_(const std::filesystem::path& p)
{
  std::ifstream in(p, std::ios::binary);
  if (!in) { return {}; }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// Throttled in-place progress bar -- redraws on a carriage-return only
// when the integer percentage changes (the frame is space-padded so a
// shorter redraw fully overwrites a longer prior one).
void
eval_progress_(vpipe::UiTextStream* bar, const char* tag, int done,
               int total, int& last_pct)
{
  if (bar == nullptr || total <= 0) { return; }
  int pct = static_cast<int>(static_cast<long>(done) * 100 / total);
  if (pct < 0) { pct = 0; } else if (pct > 100) { pct = 100; }
  if (pct == last_pct) { return; }
  last_pct = pct;
  constexpr int W = 24;
  const int fill = pct * W / 100;
  std::string b(static_cast<std::size_t>(fill), '#');
  b += std::string(static_cast<std::size_t>(W - fill), '-');
  std::string line = fmt("\r[{}] {}% {} ({}/{})", b, pct, tag, done,
                         total)();
  while (line.size() < 64) { line += ' '; }   // wipe stale tail
  bar->write(line);
}

}  // namespace

bool
load_wikitext_dir(const std::string& dir, std::string& out_text,
                  std::string& err)
{
  out_text.clear();
  const auto pages = dataset_pages_(dir);
  if (pages.empty()) {
    err = "no rows-*.json pages in '" + dir +
          "' (fetch the 'wikitext-2-raw-test' dataset with model-fetch)";
    return false;
  }
  for (const auto& p : pages) {
    const std::string body = read_file_(p);
    if (body.empty()) { continue; }
    FlexData root;
    try { root = FlexData::from_json(body); } catch (...) { continue; }
    if (!root.is_object()) { continue; }
    const auto ro = root.as_object();
    if (!ro.contains("rows")) { continue; }
    const FlexData rows_fd = ro.at("rows");
    if (!rows_fd.is_array()) { continue; }
    const auto rows = rows_fd.as_array();
    for (std::size_t i = 0; i < rows.size(); ++i) {
      const FlexData rentry = rows[i];
      if (!rentry.is_object()) { continue; }
      const auto reo = rentry.as_object();
      if (!reo.contains("row")) { continue; }
      const FlexData row = reo.at("row");
      if (!row.is_object()) { continue; }
      const auto rowo = row.as_object();
      if (rowo.contains("text")) {
        out_text += std::string(rowo.at("text").as_string(""));
      }
    }
  }
  if (out_text.empty()) {
    err = "no 'text' rows parsed from '" + dir + "'";
    return false;
  }
  return true;
}

bool
load_arc_dir(const std::string& dir, std::vector<ArcItem>& out,
             std::string& err)
{
  out.clear();
  const auto pages = dataset_pages_(dir);
  if (pages.empty()) {
    err = "no rows-*.json pages in '" + dir +
          "' (fetch the 'arc-challenge-test' dataset with model-fetch)";
    return false;
  }
  for (const auto& p : pages) {
    const std::string body = read_file_(p);
    if (body.empty()) { continue; }
    FlexData root;
    try { root = FlexData::from_json(body); } catch (...) { continue; }
    if (!root.is_object()) { continue; }
    const auto ro = root.as_object();
    if (!ro.contains("rows")) { continue; }
    const FlexData rows_fd = ro.at("rows");
    if (!rows_fd.is_array()) { continue; }
    const auto rows = rows_fd.as_array();
    for (std::size_t i = 0; i < rows.size(); ++i) {
      const FlexData rentry = rows[i];
      if (!rentry.is_object()) { continue; }
      const auto reo = rentry.as_object();
      if (!reo.contains("row")) { continue; }
      const FlexData row = reo.at("row");
      if (!row.is_object()) { continue; }
      const auto rowo = row.as_object();
      if (!rowo.contains("question") || !rowo.contains("choices") ||
          !rowo.contains("answerKey")) {
        continue;
      }
      const FlexData ch = rowo.at("choices");
      if (!ch.is_object()) { continue; }
      const auto cho = ch.as_object();
      if (!cho.contains("text") || !cho.contains("label")) { continue; }
      const FlexData texts_fd = cho.at("text");
      const FlexData labels_fd = cho.at("label");
      if (!texts_fd.is_array() || !labels_fd.is_array()) { continue; }
      const auto texts = texts_fd.as_array();
      const auto labels = labels_fd.as_array();
      if (texts.size() < 2 || texts.size() != labels.size()) { continue; }
      const std::string ak(rowo.at("answerKey").as_string(""));
      int answer = -1;
      ArcItem item;
      item.question = std::string(rowo.at("question").as_string(""));
      for (std::size_t c = 0; c < texts.size(); ++c) {
        item.choices.emplace_back(texts[c].as_string(""));
        if (std::string(labels[c].as_string("")) == ak) {
          answer = (int)c;
        }
      }
      if (answer < 0 || item.question.empty()) { continue; }
      item.answer = answer;
      out.push_back(std::move(item));
    }
  }
  if (out.empty()) {
    err = "no ARC questions parsed from '" + dir + "'";
    return false;
  }
  return true;
}

PerplexityResult
eval_wikitext2_perplexity(LoadedLanguageModel& lm, std::string_view text,
                          int max_tokens, const SessionContextIntf* session)
{
  PerplexityResult r;
  std::vector<std::int32_t> ids = lm.tokenizer().encode(text);
  if (max_tokens > 0 && (int)ids.size() > max_tokens) {
    ids.resize((std::size_t)max_tokens);
  }
  // Prepend BOS so the first real token is scored in-distribution (and so
  // BOS-sensitive models like Gemma see a valid sequence start).
  const int bos = bos_id_(lm.tokenizer());
  if (bos >= 0) { ids.insert(ids.begin(), (std::int32_t)bos); }
  if (ids.size() < 2) { return r; }

  LoadedLanguageModel::Context ctx = lm.make_context();
  if (!ctx.valid()) { return r; }
  // Prefill the first token; last_logits_host() then predicts ids[1].
  std::array<std::int32_t, 1> first{ ids[0] };
  if (lm.prefill(ctx, std::span<const std::int32_t>(first)) < 0) { return r; }

  std::unique_ptr<vpipe::UiTextStream> bar;
  if (session) { bar = session->open_text_stream(); }
  if (session) {
    session->log_normal(
        fmt("eval: WikiText-2 perplexity over {} tokens", ids.size()));
  }

  double nll = 0.0;
  long long n = 0;
  int ppl_pct = -1;
  for (std::size_t i = 1; i < ids.size(); ++i) {
    eval_progress_(bar.get(), "perplexity", (int)i, (int)ids.size(), ppl_pct);
    const std::vector<float>& lg = lm.last_logits_host();
    if (lg.empty()) { break; }
    nll += -log_prob_at_(lg, ids[i]);
    ++n;
    if (i + 1 < ids.size()) {
      // Feed the true token; advances last_logits_host() to predict ids[i+1].
      if (lm.next_token(ctx, ids[i]) < 0) { break; }
    }
  }
  if (bar) { bar->end(); }
  if (n <= 0) { return r; }
  r.ok         = true;
  r.n_tokens   = n;
  r.avg_nll    = nll / (double)n;
  r.perplexity = std::exp(r.avg_nll);
  return r;
}

ArcResult
eval_arc_challenge(LoadedLanguageModel& lm, const std::vector<ArcItem>& items,
                   int n_samples, std::uint64_t seed,
                   const SessionContextIntf* session)
{
  ArcResult r;
  const int total = (int)items.size();
  if (total == 0) { return r; }

  // Deterministic random sample of indices.
  std::vector<int> idx(total);
  std::iota(idx.begin(), idx.end(), 0);
  std::mt19937_64 rng(seed);
  std::shuffle(idx.begin(), idx.end(), rng);
  const int take = (n_samples > 0 && n_samples < total) ? n_samples : total;

  const Tokenizer& tok = lm.tokenizer();
  const int bos = bos_id_(tok);

  std::unique_ptr<vpipe::UiTextStream> bar;
  if (session) { bar = session->open_text_stream(); }
  if (session) {
    session->log_normal(fmt("eval: ARC-Challenge over {} samples", take));
  }

  int correct = 0, scored = 0;
  int arc_pct = -1;
  for (int s = 0; s < take; ++s) {
    eval_progress_(bar.get(), "ARC", s, take, arc_pct);
    const ArcItem& item = items[(std::size_t)idx[(std::size_t)s]];
    const int nc = (int)item.choices.size();
    const int gold = item.answer;
    if (nc < 2 || gold < 0 || gold >= nc) { continue; }

    std::vector<std::int32_t> stem =
        tok.encode("Question: " + item.question + "\nAnswer:");
    if (stem.empty()) { continue; }
    if (bos >= 0) { stem.insert(stem.begin(), (std::int32_t)bos); }

    int best = -1;
    double best_norm = -1e300;
    bool ok_q = true;
    for (int c = 0; c < nc; ++c) {
      const std::vector<std::int32_t> cont =
          tok.encode(" " + item.choices[(std::size_t)c]);
      if (cont.empty()) { ok_q = false; break; }
      // Fresh context per choice: prefill the stem, then teacher-force the
      // continuation, summing the per-token log-probability.
      LoadedLanguageModel::Context ctx = lm.make_context();
      if (!ctx.valid()) { ok_q = false; break; }
      if (lm.prefill(ctx, std::span<const std::int32_t>(stem)) < 0) {
        ok_q = false; break;
      }
      double lp = 0.0;
      bool ok_c = true;
      for (std::size_t j = 0; j < cont.size(); ++j) {
        const std::vector<float>& lg = lm.last_logits_host();
        if (lg.empty()) { ok_c = false; break; }
        lp += log_prob_at_(lg, cont[j]);
        if (j + 1 < cont.size()) {
          if (lm.next_token(ctx, cont[j]) < 0) { ok_c = false; break; }
        }
      }
      if (!ok_c) { ok_q = false; break; }
      const double norm = lp / (double)cont.size();   // length-normalized
      if (norm > best_norm) { best_norm = norm; best = c; }
    }
    if (!ok_q || best < 0) { continue; }
    ++scored;
    const bool correct_q = (best == gold);
    if (correct_q) { ++correct; }
    if (session) {
      session->log_debug(fmt("  arc {}/{}: {}", s + 1, take,
                             correct_q ? "ok" : "miss"));
    }
  }
  if (bar) { bar->end(); }
  if (scored <= 0) { return r; }
  r.ok       = true;
  r.n        = scored;
  r.correct  = correct;
  r.accuracy = (double)correct / (double)scored;
  return r;
}

bool
capture_token_logits(LoadedLanguageModel& lm, std::string_view text,
                     int max_tokens, std::vector<std::int32_t>& out_ids,
                     std::vector<float>& out_logits, int& out_n, int& out_vocab,
                     const SessionContextIntf* session)
{
  out_ids.clear();
  out_logits.clear();
  out_n     = 0;
  out_vocab = lm.config().vocab_size;

  std::vector<std::int32_t> ids = lm.tokenizer().encode(text);
  if (max_tokens > 0 && (int)ids.size() > max_tokens) {
    ids.resize((std::size_t)max_tokens);
  }
  // BOS-prepend, same as the perplexity probe (in-distribution first token).
  const int bos = bos_id_(lm.tokenizer());
  if (bos >= 0) { ids.insert(ids.begin(), (std::int32_t)bos); }
  if (ids.size() < 2) { return false; }

  LoadedLanguageModel::Context ctx = lm.make_context();
  if (!ctx.valid()) { return false; }
  std::array<std::int32_t, 1> first{ ids[0] };
  if (lm.prefill(ctx, std::span<const std::int32_t>(first)) < 0) {
    return false;
  }

  std::unique_ptr<vpipe::UiTextStream> bar;
  if (session) { bar = session->open_text_stream(); }
  if (session) {
    session->log_normal(
        fmt("eval: capturing A logits over {} tokens", ids.size()));
  }

  const int vocab = out_vocab;
  out_logits.reserve((std::size_t)(ids.size() - 1) * (std::size_t)vocab);
  int cap_pct = -1;
  for (std::size_t i = 1; i < ids.size(); ++i) {
    eval_progress_(bar.get(), "capture", (int)i, (int)ids.size(), cap_pct);
    const std::vector<float>& lg = lm.last_logits_host();
    if ((int)lg.size() != vocab) { break; }
    out_logits.insert(out_logits.end(), lg.begin(), lg.end());
    out_ids.push_back(ids[i - 1]);   // the id whose output predicts ids[i]
    ++out_n;
    if (i + 1 < ids.size()) {
      if (lm.next_token(ctx, ids[i]) < 0) { break; }
    }
  }
  // Record the final input id too, so B replays the EXACT fed sequence.
  if (out_n > 0) { out_ids.push_back(ids[(std::size_t)out_n]); }
  if (bar) { bar->end(); }
  return out_n > 0;
}

AbDivergenceResult
ab_divergence(LoadedLanguageModel& lm_b, const std::vector<std::int32_t>& ids,
              const std::vector<float>& a_logits, int a_n, int vocab,
              const SessionContextIntf* session)
{
  AbDivergenceResult r;
  if (a_n <= 0 || vocab <= 0) { return r; }
  if ((int)ids.size() < a_n + 1) { return r; }
  if ((long long)a_logits.size() < (long long)a_n * vocab) { return r; }

  // KL needs a shared vocab/tokenizer; a mismatch is meaningless.
  if (lm_b.config().vocab_size != vocab) {
    if (session) {
      session->log_normal(fmt(
          "eval: A-vs-B divergence skipped: vocab mismatch (A={}, B={})",
          vocab, lm_b.config().vocab_size));
    }
    return r;
  }

  LoadedLanguageModel::Context ctx = lm_b.make_context();
  if (!ctx.valid()) { return r; }
  std::array<std::int32_t, 1> first{ ids[0] };
  if (lm_b.prefill(ctx, std::span<const std::int32_t>(first)) < 0) {
    return r;
  }

  std::unique_ptr<vpipe::UiTextStream> bar;
  if (session) { bar = session->open_text_stream(); }
  if (session) {
    session->log_normal(
        fmt("eval: A-vs-B divergence over {} positions", a_n));
  }

  double kl_sum = 0.0, rel_sum = 0.0;
  int n = 0;
  int div_pct = -1;
  for (int t = 0; t < a_n; ++t) {
    eval_progress_(bar.get(), "divergence", t, a_n, div_pct);
    const std::vector<float>& b = lm_b.last_logits_host();
    if ((int)b.size() != vocab) { break; }
    const float* a = a_logits.data() + (std::size_t)t * vocab;

    // Per-row maxima for numerically-stable softmax / log-softmax.
    float amax = a[0], bmax = b[0];
    for (int v = 0; v < vocab; ++v) {
      if (a[v] > amax) { amax = a[v]; }
      if (b[v] > bmax) { bmax = b[v]; }
    }
    double asum = 0.0, bsum = 0.0;
    for (int v = 0; v < vocab; ++v) {
      asum += std::exp((double)a[v] - (double)amax);
      bsum += std::exp((double)b[v] - (double)bmax);
    }
    const double log_asum = std::log(asum);
    const double log_bsum = std::log(bsum);

    // KL(P_A || P_B) = sum P_A (logP_A - logP_B), with the row-max shifts
    // cancelling inside (logP_A - logP_B). Also the logit relative-L2.
    double kl = 0.0, diff2 = 0.0, an2 = 0.0;
    for (int v = 0; v < vocab; ++v) {
      const double la  = ((double)a[v] - (double)amax) - log_asum;  // logP_A
      const double lb  = ((double)b[v] - (double)bmax) - log_bsum;  // logP_B
      const double pa  = std::exp(la);
      kl += pa * (la - lb);
      const double d = (double)a[v] - (double)b[v];
      diff2 += d * d;
      an2   += (double)a[v] * (double)a[v];
    }
    if (kl < 0.0) { kl = 0.0; }   // clamp fp drift below 0
    kl_sum  += kl;
    rel_sum += std::sqrt(diff2) / std::max(std::sqrt(an2), 1e-12);
    ++n;
    if (t + 1 < a_n) {
      if (lm_b.next_token(ctx, ids[t + 1]) < 0) { break; }
    }
  }
  if (bar) { bar->end(); }
  if (n <= 0) { return r; }
  r.ok     = true;
  r.n      = n;
  r.kl     = kl_sum / (double)n;
  r.rel_l2 = rel_sum / (double)n;
  return r;
}

}  // namespace vpipe::genai

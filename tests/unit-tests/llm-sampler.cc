// Sampler unit tests. Drives the host-side sampler with synthetic
// logit vectors so we can verify the parameter behaviour without
// loading a real model.

#include "minitest.h"

#include "generative-models/sampler.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <set>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

// A tiny logit vector with a clear ordering. Index 3 is the argmax.
const vector<float>&
fixture_logits_()
{
  static const vector<float> v = {
      -1.0f, 0.0f, 0.5f, 5.0f, 2.0f, 0.1f, -3.0f, 1.0f,
  };
  return v;
}

}  // namespace

TEST(llm_sampler, defaults_reduce_to_argmax)
{
  genai::Sampler s(genai::SamplerParams{});
  EXPECT_TRUE(s.is_argmax());
  auto v = fixture_logits_();
  for (int i = 0; i < 10; ++i) {
    int32_t id = s.sample(std::span<const float>(v.data(), v.size()));
    EXPECT_TRUE(id == 3);
  }
}

TEST(llm_sampler, temperature_zero_is_argmax)
{
  genai::SamplerParams p;
  p.temperature = 0.0f;
  genai::Sampler s(p);
  EXPECT_TRUE(s.is_argmax());
  auto v = fixture_logits_();
  EXPECT_TRUE(s.sample(std::span<const float>(v.data(), v.size())) == 3);
}

TEST(llm_sampler, top_k_one_is_argmax)
{
  genai::SamplerParams p;
  p.top_k = 1;
  genai::Sampler s(p);
  EXPECT_TRUE(s.is_argmax());
  auto v = fixture_logits_();
  EXPECT_TRUE(s.sample(std::span<const float>(v.data(), v.size())) == 3);
}

TEST(llm_sampler, fixed_seed_is_deterministic)
{
  genai::SamplerParams p;
  p.temperature = 1.0f;
  p.top_k       = 4;
  p.seed        = 12345;
  genai::Sampler a(p);
  genai::Sampler b(p);
  auto v = fixture_logits_();
  for (int i = 0; i < 20; ++i) {
    int32_t ia = a.sample(std::span<const float>(v.data(), v.size()));
    int32_t ib = b.sample(std::span<const float>(v.data(), v.size()));
    EXPECT_TRUE(ia == ib);
  }
}

TEST(llm_sampler, top_k_only_picks_from_top_k)
{
  genai::SamplerParams p;
  p.temperature = 1.0f;
  p.top_k       = 2;
  p.seed        = 7;
  genai::Sampler s(p);
  auto v = fixture_logits_();
  // Top-2 of fixture: idx 3 (5.0) and idx 4 (2.0). Sampler must
  // pick one of those two over many draws.
  for (int i = 0; i < 50; ++i) {
    int32_t id = s.sample(std::span<const float>(v.data(), v.size()));
    EXPECT_TRUE(id == 3 || id == 4);
  }
}

TEST(llm_sampler, top_p_keeps_high_probability_mass)
{
  genai::SamplerParams p;
  p.temperature = 1.0f;
  p.top_p       = 0.9f;
  p.seed        = 42;
  genai::Sampler s(p);
  // With temperature=1 and the fixture, idx 3 dominates the
  // softmax mass; top-p=0.9 should keep mostly idx 3 (and a tiny
  // tail of idx 4). Run many samples and confirm idx 3 wins
  // overwhelmingly.
  auto v = fixture_logits_();
  int count_3 = 0;
  int total   = 200;
  for (int i = 0; i < total; ++i) {
    int32_t id = s.sample(std::span<const float>(v.data(), v.size()));
    if (id == 3) { ++count_3; }
  }
  EXPECT_TRUE(count_3 > total * 8 / 10);
}

TEST(llm_sampler, min_p_drops_long_tail)
{
  // min_p relative to max — at min_p=0.5 only tokens with prob
  // >= 0.5 * max_prob survive. With softmax(fixture/T=1), idx 3
  // is ~95% of the mass; nothing else stays above 0.5 * 0.95 =
  // 0.475. So we expect idx 3 every time.
  genai::SamplerParams p;
  p.temperature = 1.0f;
  p.min_p       = 0.5f;
  p.seed        = 99;
  genai::Sampler s(p);
  auto v = fixture_logits_();
  for (int i = 0; i < 30; ++i) {
    EXPECT_TRUE(
        s.sample(std::span<const float>(v.data(), v.size())) == 3);
  }
}

TEST(llm_sampler, repetition_penalty_depresses_seen_token)
{
  // Build flatter logits so the penalty's effect on the argmax is
  // visible. Without the penalty, idx 3 (logit=2) is the argmax.
  // After dividing l[3] by rp=10 -> 0.2, idx 3's softmax mass
  // drops well below the others.
  //
  // Each iteration uses a FRESH sampler (so _seen only contains
  // the primed token) — otherwise _seen accumulates from prior
  // picks and dilutes the penalty across all logits.
  const vector<float> v = { 1.0f, 1.0f, 1.0f, 2.0f, 1.0f, 1.0f };
  int count_3 = 0;
  const int N = 200;
  for (int i = 0; i < N; ++i) {
    genai::SamplerParams p;
    p.temperature        = 1.0f;
    p.repetition_penalty = 10.0f;
    p.seed               = 11u + static_cast<uint64_t>(i);
    genai::Sampler s(p);
    s.prime(std::array<int32_t, 1>{3});
    int32_t id = s.sample(std::span<const float>(v.data(), v.size()));
    if (id == 3) { ++count_3; }
  }
  // Unpenalised P(3) ≈ 35%; penalised P(3) ≈ 8%. Allow plenty of
  // slack for FP / seed variance.
  EXPECT_TRUE(count_3 < N / 5);
}

TEST(llm_sampler, presence_penalty_subtracts_from_seen)
{
  // Same shape as the repetition-penalty test; uses the additive
  // presence penalty instead. presence_pen=10 - large enough that
  // idx 3's logit drops from 2 -> -8, well below the unseen
  // logits at 1.
  const vector<float> v = { 1.0f, 1.0f, 1.0f, 2.0f, 1.0f, 1.0f };
  int count_3 = 0;
  const int N = 200;
  for (int i = 0; i < N; ++i) {
    genai::SamplerParams p;
    p.temperature      = 1.0f;
    p.presence_penalty = 10.0f;
    p.seed             = 13u + static_cast<uint64_t>(i);
    genai::Sampler s(p);
    s.prime(std::array<int32_t, 1>{3});
    int32_t id = s.sample(std::span<const float>(v.data(), v.size()));
    if (id == 3) { ++count_3; }
  }
  EXPECT_TRUE(count_3 < N / 10);
}

TEST(llm_sampler, mixed_params_does_not_nan_or_crash)
{
  genai::SamplerParams p;
  p.temperature        = 0.7f;
  p.top_k              = 10;
  p.top_p              = 0.92f;
  p.min_p              = 0.03f;
  p.repetition_penalty = 1.1f;
  p.presence_penalty   = 0.05f;
  p.seed               = 0xC0FFEE;
  genai::Sampler s(p);
  // Wider logit vector; sample a bunch, all results in range.
  vector<float> v(128);
  for (size_t i = 0; i < v.size(); ++i) {
    v[i] = static_cast<float>(i % 7) * 0.3f
         + static_cast<float>(((i * 91) % 13)) * 0.05f;
  }
  for (int it = 0; it < 100; ++it) {
    int32_t id = s.sample(std::span<const float>(v.data(), v.size()));
    EXPECT_TRUE(id >= 0);
    EXPECT_TRUE(id < static_cast<int32_t>(v.size()));
  }
}

// Helper: extract top-K (values + indices) from a flat logit vector
// the way LoadedLanguageModel::batched_topk_last_logits would, so the
// sample_topk tests don't need an MLX runtime.
static void
extract_topk_(const vector<float>& logits, int k,
              vector<float>& top_vals, vector<int32_t>& top_idx)
{
  const int V = static_cast<int>(logits.size());
  k = std::min(k, V);
  vector<int32_t> idx(V);
  for (int i = 0; i < V; ++i) { idx[i] = i; }
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                    [&](int32_t a, int32_t b) {
                      return logits[a] > logits[b];
                    });
  top_vals.resize(k);
  top_idx.resize(k);
  for (int i = 0; i < k; ++i) {
    top_idx[i] = idx[i];
    top_vals[i] = logits[idx[i]];
  }
}

TEST(llm_sampler, topk_argmax_matches_sample_argmax)
{
  // sample_topk under argmax-equivalent params must pick the same id
  // as sample() does.
  genai::SamplerParams p;
  genai::Sampler s(p);
  auto v = fixture_logits_();
  vector<float> vals;
  vector<int32_t> idx;
  extract_topk_(v, 4, vals, idx);
  for (int i = 0; i < 10; ++i) {
    int32_t id = s.sample_topk(
        std::span<const float>(vals.data(), vals.size()),
        std::span<const int32_t>(idx.data(), idx.size()));
    EXPECT_TRUE(id == 3);
  }
}

TEST(llm_sampler, topk_under_top_p_picks_within_topk)
{
  // sample_topk should only ever return ids that appear in top_idx;
  // never an out-of-set index.
  genai::SamplerParams p;
  p.temperature = 1.0f;
  p.top_p       = 0.9f;
  p.seed        = 99;
  genai::Sampler s(p);
  auto v = fixture_logits_();
  vector<float> vals;
  vector<int32_t> idx;
  extract_topk_(v, 3, vals, idx);  // top-3 of fixture
  std::set<int32_t> allowed(idx.begin(), idx.end());
  for (int i = 0; i < 100; ++i) {
    int32_t id = s.sample_topk(
        std::span<const float>(vals.data(), vals.size()),
        std::span<const int32_t>(idx.data(), idx.size()));
    EXPECT_TRUE(allowed.count(id) == 1);
  }
}

TEST(llm_sampler, topk_matches_sample_distribution_when_topk_covers_support)
{
  // When K_internal covers everything top_k_param would have kept,
  // sample_topk and sample() must produce IDENTICAL draws (same RNG
  // seed, same params, same prime). This is the correctness contract
  // for the batched-GPU-topk path.
  const vector<float> v = {
      0.1f, 0.5f, 1.2f, 3.0f, 0.7f, 2.1f, -0.4f, 1.8f,
      0.0f, 2.4f, 0.3f, 1.0f, -1.2f, 0.6f, 1.5f, 0.8f,
  };
  vector<float> vals;
  vector<int32_t> idx;
  extract_topk_(v, 5, vals, idx);

  genai::SamplerParams p;
  p.temperature = 0.8f;
  p.top_k       = 5;
  p.top_p       = 0.95f;
  p.seed        = 4242;

  genai::Sampler a(p), b(p);
  for (int i = 0; i < 50; ++i) {
    int32_t ia = a.sample(std::span<const float>(v.data(), v.size()));
    int32_t ib = b.sample_topk(
        std::span<const float>(vals.data(), vals.size()),
        std::span<const int32_t>(idx.data(), idx.size()));
    EXPECT_TRUE(ia == ib);
  }
}

TEST(llm_sampler, topk_repetition_penalty_depresses_seen_in_set)
{
  // Penalty applied to a seen token whose vocab id IS in the top-K
  // view should depress its sampling probability, just like sample().
  const vector<float> v = { 1.0f, 1.0f, 1.0f, 2.0f, 1.0f, 1.0f };
  vector<float> vals;
  vector<int32_t> idx;
  extract_topk_(v, 4, vals, idx);  // covers idx 3
  int count_3 = 0;
  const int N = 200;
  for (int i = 0; i < N; ++i) {
    genai::SamplerParams p;
    p.temperature        = 1.0f;
    p.repetition_penalty = 10.0f;
    p.seed               = 23u + static_cast<uint64_t>(i);
    genai::Sampler s(p);
    s.prime(std::array<int32_t, 1>{3});
    int32_t id = s.sample_topk(
        std::span<const float>(vals.data(), vals.size()),
        std::span<const int32_t>(idx.data(), idx.size()));
    if (id == 3) { ++count_3; }
  }
  EXPECT_TRUE(count_3 < N / 4);
}

TEST(llm_sampler, reset_forgets_prior_picks)
{
  // After reset(), the prime() set is cleared so the penalty no
  // longer applies to idx 3 on the very next sample. Test by
  // priming idx 3 then resetting before the first draw, across
  // many fresh seeds — most picks should land back on idx 3 (the
  // original argmax of the un-penalised softmax).
  const vector<float> v = { 1.0f, 1.0f, 1.0f, 2.0f, 1.0f, 1.0f };
  int count_3 = 0;
  const int N = 100;
  for (int i = 0; i < N; ++i) {
    genai::SamplerParams p;
    p.temperature        = 1.0f;
    p.repetition_penalty = 10.0f;
    p.seed               = 17u + static_cast<uint64_t>(i);
    genai::Sampler s(p);
    s.prime(std::array<int32_t, 1>{3});
    s.reset();
    int32_t id = s.sample(std::span<const float>(v.data(), v.size()));
    if (id == 3) { ++count_3; }
  }
  EXPECT_TRUE(count_3 > N / 4);
}

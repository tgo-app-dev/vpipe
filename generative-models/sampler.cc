#include "generative-models/sampler.h"

#include "common/flex-data.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace vpipe::genai {

namespace {

constexpr float kFloatEps = 1e-9f;

bool
near_one_(float v, float eps = 1e-6f) noexcept
{
  return std::fabs(v - 1.0f) <= eps;
}

bool
near_zero_(float v, float eps = 1e-6f) noexcept
{
  return std::fabs(v) <= eps;
}

}  // namespace

Sampler::Sampler(SamplerParams p)
  : _params(p)
{
  if (_params.seed == 0) {
    std::random_device rd;
    std::seed_seq sseq{rd(), rd(), rd(), rd()};
    _rng.seed(sseq);
  } else {
    _rng.seed(_params.seed);
  }
}

bool
Sampler::is_argmax() const noexcept
{
  // Argmax-equivalent when:
  //   * temperature <= 0 (explicit greedy), OR
  //   * top_k == 1 (only the top logit can be picked), OR
  //   * every knob is at its disabled / no-effect default AND
  //     temperature == 1 (so the pre-sample probability ordering
  //     matches the logit ordering and the multinomial draw of a
  //     1-prob distribution returns the same id every time).
  if (_params.temperature <= 0.0f) {
    return true;
  }
  if (_params.top_k == 1) {
    return true;
  }
  return near_one_(_params.temperature)
      && _params.top_k <= 0
      && _params.top_p >= 1.0f - 1e-9f
      && _params.min_p <= 0.0f
      && near_one_(_params.repetition_penalty)
      && near_zero_(_params.presence_penalty);
}

void
Sampler::prime(std::span<const std::int32_t> tokens)
{
  for (auto t : tokens) {
    if (t >= 0) {
      _seen.insert(t);
    }
  }
}

void
Sampler::reset()
{
  _seen.clear();
}

std::int32_t
Sampler::sample(std::span<const float> logits)
{
  const std::size_t V = logits.size();
  if (V == 0) {
    return -1;
  }

  // Fast path: any parameter combination that's mathematically
  // equivalent to argmax — temperature <= 0, top_k == 1, or every
  // knob at its disabled default. Skips the full softmax + filter
  // chain entirely.
  if (is_argmax()) {
    std::size_t best = 0;
    float best_v = logits[0];
    for (std::size_t i = 1; i < V; ++i) {
      if (logits[i] > best_v) {
        best_v = logits[i];
        best   = i;
      }
    }
    _seen.insert(static_cast<std::int32_t>(best));
    return static_cast<std::int32_t>(best);
  }

  _scratch.assign(logits.begin(), logits.end());
  float* l = _scratch.data();

  // 1. Repetition penalty. HuggingFace convention: divide positive
  // logits by `rp`, multiply negative logits by `rp`. rp > 1 makes
  // already-seen tokens less likely; rp < 1 (unusual) makes them
  // more likely.
  if (!near_one_(_params.repetition_penalty) && !_seen.empty()) {
    const float rp = _params.repetition_penalty;
    for (std::int32_t id : _seen) {
      if (id < 0 || static_cast<std::size_t>(id) >= V) { continue; }
      const float v = l[id];
      l[id] = (v >= 0.0f) ? (v / rp) : (v * rp);
    }
  }

  // 2. Presence penalty. OpenAI convention: subtract `pp` from every
  // already-seen token's logit. Independent of frequency.
  if (!near_zero_(_params.presence_penalty) && !_seen.empty()) {
    const float pp = _params.presence_penalty;
    for (std::int32_t id : _seen) {
      if (id < 0 || static_cast<std::size_t>(id) >= V) { continue; }
      l[id] -= pp;
    }
  }

  // 3. Temperature scaling.
  if (!near_one_(_params.temperature)) {
    const float inv_t = 1.0f / _params.temperature;
    for (std::size_t i = 0; i < V; ++i) {
      l[i] *= inv_t;
    }
  }

  // 4. Softmax to probabilities (subtract max for stability).
  float max_l = l[0];
  for (std::size_t i = 1; i < V; ++i) {
    if (l[i] > max_l) { max_l = l[i]; }
  }
  float sum_e = 0.0f;
  for (std::size_t i = 0; i < V; ++i) {
    const float e = std::exp(l[i] - max_l);
    l[i] = e;
    sum_e += e;
  }
  if (sum_e <= kFloatEps) {
    // Degenerate — pick argmax of original logits as a last resort.
    std::size_t best = 0;
    float best_v = logits[0];
    for (std::size_t i = 1; i < V; ++i) {
      if (logits[i] > best_v) {
        best_v = logits[i];
        best   = i;
      }
    }
    _seen.insert(static_cast<std::int32_t>(best));
    return static_cast<std::int32_t>(best);
  }
  for (std::size_t i = 0; i < V; ++i) {
    l[i] /= sum_e;
  }

  // 5. Top-k filter: zero out probabilities outside the top k.
  if (_params.top_k > 0 && static_cast<std::size_t>(_params.top_k) < V) {
    // Pull indices and partial-sort by probability descending.
    static thread_local std::vector<std::int32_t> idx_buf;
    idx_buf.resize(V);
    for (std::size_t i = 0; i < V; ++i) {
      idx_buf[i] = static_cast<std::int32_t>(i);
    }
    const std::size_t k = static_cast<std::size_t>(_params.top_k);
    std::nth_element(
        idx_buf.begin(), idx_buf.begin() + (k - 1), idx_buf.end(),
        [&](std::int32_t a, std::int32_t b) {
          return l[a] > l[b];
        });
    // Find the kth-largest probability — the cutoff.
    float cutoff = l[idx_buf[k - 1]];
    for (std::size_t i = 0; i < V; ++i) {
      if (l[i] < cutoff) { l[i] = 0.0f; }
    }
  }

  // 6. Min-p filter: drop tokens whose probability is below
  //    min_p * max_prob (after the top-k mask).
  if (_params.min_p > 0.0f) {
    float max_p = 0.0f;
    for (std::size_t i = 0; i < V; ++i) {
      if (l[i] > max_p) { max_p = l[i]; }
    }
    const float floor = _params.min_p * max_p;
    for (std::size_t i = 0; i < V; ++i) {
      if (l[i] < floor) { l[i] = 0.0f; }
    }
  }

  // 7. Top-p (nucleus) filter: sort by probability descending and
  //    keep the smallest prefix whose cumulative probability >= p.
  if (_params.top_p < 1.0f - 1e-9f) {
    static thread_local std::vector<std::int32_t> idx_buf;
    idx_buf.resize(V);
    for (std::size_t i = 0; i < V; ++i) {
      idx_buf[i] = static_cast<std::int32_t>(i);
    }
    std::sort(idx_buf.begin(), idx_buf.end(),
              [&](std::int32_t a, std::int32_t b) {
                return l[a] > l[b];
              });
    float running_sum = 0.0f;
    std::size_t cutoff_idx = V;
    for (std::size_t i = 0; i < V; ++i) {
      running_sum += l[idx_buf[i]];
      if (running_sum >= _params.top_p) {
        cutoff_idx = i + 1;
        break;
      }
    }
    if (cutoff_idx == 0) {
      cutoff_idx = 1;
    }
    for (std::size_t i = cutoff_idx; i < V; ++i) {
      l[idx_buf[i]] = 0.0f;
    }
  }

  // 8. Renormalise.
  float new_sum = 0.0f;
  for (std::size_t i = 0; i < V; ++i) {
    new_sum += l[i];
  }
  if (new_sum <= kFloatEps) {
    // Every filter dropped everything — pick the argmax of the
    // pre-filter probabilities (which equals the original logits'
    // argmax modulo the penalty modifications).
    std::size_t best = 0;
    float best_v = logits[0];
    for (std::size_t i = 1; i < V; ++i) {
      if (logits[i] > best_v) {
        best_v = logits[i];
        best   = i;
      }
    }
    _seen.insert(static_cast<std::int32_t>(best));
    return static_cast<std::int32_t>(best);
  }
  const float inv_sum = 1.0f / new_sum;
  for (std::size_t i = 0; i < V; ++i) {
    l[i] *= inv_sum;
  }

  // 9. Multinomial sample via inverse CDF.
  std::uniform_real_distribution<float> u01(0.0f, 1.0f);
  const float r = u01(_rng);
  float cdf = 0.0f;
  std::int32_t picked = -1;
  for (std::size_t i = 0; i < V; ++i) {
    cdf += l[i];
    if (cdf >= r) {
      picked = static_cast<std::int32_t>(i);
      break;
    }
  }
  if (picked < 0) {
    // FP rounding edge case: r ~= 1.0 and cdf never crossed. Walk
    // back to the last non-zero entry.
    for (std::size_t i = V; i-- > 0;) {
      if (l[i] > 0.0f) {
        picked = static_cast<std::int32_t>(i);
        break;
      }
    }
    if (picked < 0) { picked = 0; }
  }
  _seen.insert(picked);
  return picked;
}

std::int32_t
Sampler::sample_topk(std::span<const float>        top_vals,
                     std::span<const std::int32_t> top_idx)
{
  const std::size_t K = top_vals.size();
  if (K == 0 || top_idx.size() != K) {
    return -1;
  }

  // Fast path: argmax over the top-K view. Picks the entry with the
  // largest raw logit and returns its vocab id.
  if (is_argmax()) {
    std::size_t best = 0;
    float best_v = top_vals[0];
    for (std::size_t i = 1; i < K; ++i) {
      if (top_vals[i] > best_v) {
        best_v = top_vals[i];
        best   = i;
      }
    }
    const std::int32_t id = top_idx[best];
    if (id >= 0) { _seen.insert(id); }
    return id;
  }

  // Sort the top-K view by vocab-id ascending so the final CDF walk
  // visits entries in the same order as sample()'s 0..V-1 traversal
  // (which steps past zero-prob non-set entries without advancing the
  // CDF). With matching traversal order, the same RNG `r` produces
  // the same picked vocab-id -- bit-for-bit identical to sample()
  // whenever the top-K view covers the full surviving support.
  // Argpartition's output is unordered, so we always do this sort
  // here; the K-element sort is microseconds.
  static thread_local std::vector<std::int32_t> order_buf;
  order_buf.resize(K);
  for (std::size_t i = 0; i < K; ++i) {
    order_buf[i] = static_cast<std::int32_t>(i);
  }
  std::sort(order_buf.begin(), order_buf.end(),
            [&](std::int32_t a, std::int32_t b) {
              return top_idx[a] < top_idx[b];
            });
  // Working buffers hold the K raw logits and the K vocab-ids in
  // ascending-id order. Index `i` in these buffers maps to vocab id
  // ids_sorted[i].
  _scratch.resize(K);
  static thread_local std::vector<std::int32_t> ids_sorted;
  ids_sorted.resize(K);
  for (std::size_t i = 0; i < K; ++i) {
    const std::int32_t pos = order_buf[i];
    _scratch[i]   = top_vals[pos];
    ids_sorted[i] = top_idx[pos];
  }
  float* l = _scratch.data();

  // 1. Repetition penalty. Only entries whose vocab id is in the
  //    seen-set get modified; a seen token that fell outside the
  //    top-K is left alone -- the GPU's top-K already eliminated it
  //    from contention, so dropping the penalty doesn't change the
  //    sampled distribution (its probability is zero either way).
  if (!near_one_(_params.repetition_penalty) && !_seen.empty()) {
    const float rp = _params.repetition_penalty;
    for (std::size_t i = 0; i < K; ++i) {
      const std::int32_t id = ids_sorted[i];
      if (id < 0) { continue; }
      if (_seen.find(id) == _seen.end()) { continue; }
      const float v = l[i];
      l[i] = (v >= 0.0f) ? (v / rp) : (v * rp);
    }
  }

  // 2. Presence penalty.
  if (!near_zero_(_params.presence_penalty) && !_seen.empty()) {
    const float pp = _params.presence_penalty;
    for (std::size_t i = 0; i < K; ++i) {
      const std::int32_t id = ids_sorted[i];
      if (id < 0) { continue; }
      if (_seen.find(id) == _seen.end()) { continue; }
      l[i] -= pp;
    }
  }

  // 3. Temperature scaling.
  if (!near_one_(_params.temperature)) {
    const float inv_t = 1.0f / _params.temperature;
    for (std::size_t i = 0; i < K; ++i) {
      l[i] *= inv_t;
    }
  }

  // 4. Softmax over the K logits (subtract max for stability).
  float max_l = l[0];
  for (std::size_t i = 1; i < K; ++i) {
    if (l[i] > max_l) { max_l = l[i]; }
  }
  float sum_e = 0.0f;
  for (std::size_t i = 0; i < K; ++i) {
    const float e = std::exp(l[i] - max_l);
    l[i] = e;
    sum_e += e;
  }
  if (sum_e <= kFloatEps) {
    // Degenerate -- pick argmax of the original UNMODIFIED top_vals
    // (vocab-id ascending within the working buffers; top_vals is
    // in the caller's original argpartition order so we use the same
    // ascending traversal via order_buf).
    std::size_t best = 0;
    float best_v = top_vals[order_buf[0]];
    for (std::size_t i = 1; i < K; ++i) {
      const float v = top_vals[order_buf[i]];
      if (v > best_v) { best_v = v; best = i; }
    }
    const std::int32_t id = ids_sorted[best];
    if (id >= 0) { _seen.insert(id); }
    return id;
  }
  for (std::size_t i = 0; i < K; ++i) {
    l[i] /= sum_e;
  }

  // 5. Top-K filter is a NO-OP here -- the GPU already extracted the
  //    top-K of the raw logits and `top_vals` / `top_idx` ARE that
  //    extraction. If params.top_k > 0 and params.top_k < K we still
  //    honour it by zeroing the entries past the kth largest prob.
  if (_params.top_k > 0 && static_cast<std::size_t>(_params.top_k) < K) {
    static thread_local std::vector<std::int32_t> idx_buf;
    idx_buf.resize(K);
    for (std::size_t i = 0; i < K; ++i) {
      idx_buf[i] = static_cast<std::int32_t>(i);
    }
    const std::size_t k = static_cast<std::size_t>(_params.top_k);
    std::nth_element(
        idx_buf.begin(), idx_buf.begin() + (k - 1), idx_buf.end(),
        [&](std::int32_t a, std::int32_t b) {
          return l[a] > l[b];
        });
    const float cutoff = l[idx_buf[k - 1]];
    for (std::size_t i = 0; i < K; ++i) {
      if (l[i] < cutoff) { l[i] = 0.0f; }
    }
  }

  // 6. Min-p filter.
  if (_params.min_p > 0.0f) {
    float max_p = 0.0f;
    for (std::size_t i = 0; i < K; ++i) {
      if (l[i] > max_p) { max_p = l[i]; }
    }
    const float floor = _params.min_p * max_p;
    for (std::size_t i = 0; i < K; ++i) {
      if (l[i] < floor) { l[i] = 0.0f; }
    }
  }

  // 7. Top-p filter.
  if (_params.top_p < 1.0f - 1e-9f) {
    static thread_local std::vector<std::int32_t> idx_buf;
    idx_buf.resize(K);
    for (std::size_t i = 0; i < K; ++i) {
      idx_buf[i] = static_cast<std::int32_t>(i);
    }
    std::sort(idx_buf.begin(), idx_buf.end(),
              [&](std::int32_t a, std::int32_t b) {
                return l[a] > l[b];
              });
    float running_sum = 0.0f;
    std::size_t cutoff_idx = K;
    for (std::size_t i = 0; i < K; ++i) {
      running_sum += l[idx_buf[i]];
      if (running_sum >= _params.top_p) {
        cutoff_idx = i + 1;
        break;
      }
    }
    if (cutoff_idx == 0) { cutoff_idx = 1; }
    for (std::size_t i = cutoff_idx; i < K; ++i) {
      l[idx_buf[i]] = 0.0f;
    }
  }

  // 8. Renormalise.
  float new_sum = 0.0f;
  for (std::size_t i = 0; i < K; ++i) { new_sum += l[i]; }
  if (new_sum <= kFloatEps) {
    // Filters wiped everything -- fall back to argmax of the
    // ORIGINAL (pre-penalty, pre-filter) top_vals.
    std::size_t best = 0;
    float best_v = top_vals[order_buf[0]];
    for (std::size_t i = 1; i < K; ++i) {
      const float v = top_vals[order_buf[i]];
      if (v > best_v) { best_v = v; best = i; }
    }
    const std::int32_t id = ids_sorted[best];
    if (id >= 0) { _seen.insert(id); }
    return id;
  }
  const float inv_sum = 1.0f / new_sum;
  for (std::size_t i = 0; i < K; ++i) { l[i] *= inv_sum; }

  // 9. Multinomial sample via inverse CDF over the K probs, walked
  //    in vocab-id ascending order so the iteration matches
  //    sample()'s 0..V-1 traversal (zero entries don't advance CDF).
  std::uniform_real_distribution<float> u01(0.0f, 1.0f);
  const float r = u01(_rng);
  float cdf = 0.0f;
  std::size_t picked_pos = K;
  for (std::size_t i = 0; i < K; ++i) {
    cdf += l[i];
    if (cdf >= r) { picked_pos = i; break; }
  }
  if (picked_pos >= K) {
    for (std::size_t i = K; i-- > 0;) {
      if (l[i] > 0.0f) { picked_pos = i; break; }
    }
    if (picked_pos >= K) { picked_pos = 0; }
  }
  const std::int32_t picked = ids_sorted[picked_pos];
  if (picked >= 0) { _seen.insert(picked); }
  return picked;
}

SamplerParams
parse_sampler_config(const FlexData& obj)
{
  SamplerParams p;
  if (!obj.is_object()) {
    return p;
  }
  auto root = obj.as_object();
  if (root.contains("temperature")) {
    p.temperature = static_cast<float>(
        root.at("temperature").as_real(p.temperature));
  }
  if (root.contains("top_k")) {
    p.top_k = static_cast<int>(
        root.at("top_k").as_int(p.top_k));
  }
  if (root.contains("top_p")) {
    p.top_p = static_cast<float>(
        root.at("top_p").as_real(p.top_p));
  }
  if (root.contains("min_p")) {
    p.min_p = static_cast<float>(
        root.at("min_p").as_real(p.min_p));
  }
  if (root.contains("repetition_penalty")) {
    p.repetition_penalty = static_cast<float>(
        root.at("repetition_penalty").as_real(p.repetition_penalty));
  }
  if (root.contains("presence_penalty")) {
    p.presence_penalty = static_cast<float>(
        root.at("presence_penalty").as_real(p.presence_penalty));
  }
  if (root.contains("seed")) {
    p.seed = static_cast<std::uint64_t>(
        root.at("seed").as_uint(p.seed));
  }
  return p;
}

}

#include "generative-models/moss/metal-moss-local-model.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "generative-models/llama3/metal-llama-weights.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace vpipe::genai {

using metal_compute::CommandStream;
using metal_compute::ComputeEncoder;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

float bf16_to_f32(std::uint16_t b)
{
  std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// Load a 2-D BF16/F16 tensor as f16. Returns empty on failure.
SharedBuffer
load_f16_2d(const MetalLlamaWeights& wts, MetalCompute* mc,
            const std::string& nm)
{
  const auto* info = wts.info(nm);
  if (info == nullptr || info->shape.size() != 2) { return {}; }
  SharedBuffer raw = wts.load(nm, mc);
  if (raw.empty()) { return {}; }
  const std::size_t n = (std::size_t)info->shape[0] * info->shape[1];
  SharedBuffer out = mc->make_shared_buffer(n * 2);
  if (out.empty()) { return {}; }
  if (info->dtype == "F16") {
    std::memcpy(out.contents(), raw.contents(), n * 2);
  } else if (info->dtype == "BF16") {
    const auto* s = static_cast<const std::uint16_t*>(raw.contents());
    auto* d = static_cast<_Float16*>(out.contents());
    for (std::size_t i = 0; i < n; ++i) { d[i] = (_Float16)bf16_to_f32(s[i]); }
  } else {
    return {};
  }
  return out;
}

// Host sampler over logit[0..n): temperature softmax -> top_k -> top_p
// nucleus -> sample with `rng`. Mirrors the MOSS-8B host sampler's semantics
// (temperature <= 0 => greedy argmax); no repetition penalty (each audio
// codebook is sampled independently, no per-code history).
int
sample_logits_local_(const float* logit, int n, const MossSampling& sp,
                      std::mt19937_64& rng)
{
  if (sp.temperature <= 0.0f) {                       // greedy
    int best = 0;
    float bv = logit[0];
    for (int i = 1; i < n; ++i) {
      if (logit[i] > bv) { bv = logit[i]; best = i; }
    }
    return best;
  }
  std::vector<int> cand((std::size_t)n);
  for (int i = 0; i < n; ++i) { cand[(std::size_t)i] = i; }
  // top_k: keep the k highest-logit candidates, sorted descending.
  int k = (sp.top_k > 0 && sp.top_k < n) ? sp.top_k : n;
  auto gt = [&](int a, int b) { return logit[a] > logit[b]; };
  if (k < n) {
    std::nth_element(cand.begin(), cand.begin() + (k - 1), cand.end(), gt);
    cand.resize((std::size_t)k);
  }
  std::sort(cand.begin(), cand.end(), gt);
  // temperature softmax over the kept set, then top_p nucleus truncation.
  const float inv_t = 1.0f / sp.temperature;
  const float mx = logit[cand[0]] * inv_t;
  std::vector<float> prob(cand.size());
  double sum = 0.0;
  for (std::size_t i = 0; i < cand.size(); ++i) {
    const float p = std::exp(logit[(std::size_t)cand[i]] * inv_t - mx);
    prob[i] = p;
    sum += p;
  }
  std::size_t keep = cand.size();
  if (sp.top_p < 1.0f) {
    const double thresh = sp.top_p * sum;
    double cum = 0.0;
    keep = 0;
    for (std::size_t i = 0; i < cand.size(); ++i) {
      cum += prob[i];
      ++keep;
      if (cum >= thresh) { break; }
    }
  }
  double ksum = 0.0;
  for (std::size_t i = 0; i < keep; ++i) { ksum += prob[i]; }
  const double target =
      std::uniform_real_distribution<double>(0.0, 1.0)(rng) * ksum;
  double acc = 0.0;
  for (std::size_t i = 0; i < keep; ++i) {
    acc += prob[i];
    if (acc >= target) { return cand[i]; }
  }
  return cand[keep - 1];
}

}  // namespace

std::unique_ptr<MetalMossLocalModel>
MetalMossLocalModel::load(const std::string& model_dir, MetalCompute* mc,
                          const Config& cfg)
{
  auto wts = MetalLlamaWeights::open_model(model_dir);
  if (!wts.has_value()) { return nullptr; }
  auto self = std::make_unique<MetalMossLocalModel>();
  if (!self->init_(*wts, mc, cfg)) { return nullptr; }
  return self;
}

bool
MetalMossLocalModel::init_(const MetalLlamaWeights& wts, MetalCompute* mc,
                           const Config& cfg)
{
  _mc = mc;
  _cfg = cfg;
  _lt = MetalMossLocalTransformer::load(wts, mc, cfg.lt);
  if (!_lt) { return false; }

  _heads.resize((std::size_t)cfg.n_vq);
  _embeds.resize((std::size_t)cfg.n_vq);
  for (int k = 0; k < cfg.n_vq; ++k) {
    _heads[(std::size_t)k] = load_f16_2d(
        wts, mc, "audio_lm_heads." + std::to_string(k) + ".weight");
    _embeds[(std::size_t)k] = load_f16_2d(
        wts, mc, "audio_embeddings." + std::to_string(k) + ".weight");
    if (_heads[(std::size_t)k].empty() || _embeds[(std::size_t)k].empty()) {
      return false;
    }
  }
  _logits = mc->make_shared_buffer((std::size_t)cfg.audio_vocab * 2);
  _einp   = mc->make_shared_buffer((std::size_t)cfg.lt.hidden * 2);
  _h0     = mc->make_shared_buffer((std::size_t)cfg.lt.hidden * 2);
  _codes  = mc->make_shared_buffer(
      (std::size_t)cfg.n_vq * sizeof(std::int32_t));
  _lc_ws  = mc->make_shared_buffer((std::size_t)cfg.audio_vocab * 2);
  if (_logits.empty() || _einp.empty() || _h0.empty() || _codes.empty() ||
      _lc_ws.empty()) {
    return false;
  }

  _lib_dense = mc->load_library("dense_gemm");
  _lib_elt   = mc->load_library("llm_elementwise");
  if (!_lib_dense.valid() || !_lib_elt.valid()) { return false; }
  _fn_gemv         = _lib_dense.function("dense_gemv_t_f16");
  _fn_argmax       = _lib_elt.function("argmax_f16");
  _fn_sample       = _lib_elt.function("lc_sample_f16");
  _fn_embed_gather = _lib_elt.function("embed_gather_f16");
  _fn_copy         = _lib_elt.function("copy_f16");
  return _fn_gemv.valid() && _fn_argmax.valid() && _fn_sample.valid() &&
         _fn_embed_gather.valid() && _fn_copy.valid();
}

int
MetalMossLocalModel::argmax_head_(int k, const SharedBuffer& h)
{
  const int K = _cfg.lt.hidden, N = _cfg.audio_vocab;
  CommandStream s = _mc->make_command_stream();
  {
    ComputeEncoder e = s.begin_compute();
    e.set_function(_fn_gemv);
    e.set_buffer(0, h);
    e.set_buffer(1, _heads[(std::size_t)k]);
    e.set_buffer(2, _logits);
    e.set_constant(3, K);
    e.set_constant(4, N);
    e.dispatch({32, (unsigned)(((N + 7) / 8) * 2), 1}, {32, 2, 1});
  }
  s.commit().wait();
  const auto* lg = static_cast<const _Float16*>(_logits.contents());
  int best = 0;
  float bv = (float)lg[0];
  for (int i = 1; i < N; ++i) {
    const float v = (float)lg[i];
    if (v > bv) { bv = v; best = i; }
  }
  return best;
}

int
MetalMossLocalModel::sample_head_(int k, const SharedBuffer& h,
                                  const MossSampling& sp,
                                  std::mt19937_64& rng)
{
  const int K = _cfg.lt.hidden, N = _cfg.audio_vocab;
  CommandStream s = _mc->make_command_stream();
  {
    ComputeEncoder e = s.begin_compute();
    e.set_function(_fn_gemv);
    e.set_buffer(0, h);
    e.set_buffer(1, _heads[(std::size_t)k]);
    e.set_buffer(2, _logits);
    e.set_constant(3, K);
    e.set_constant(4, N);
    e.dispatch({32, (unsigned)(((N + 7) / 8) * 2), 1}, {32, 2, 1});
  }
  s.commit().wait();
  const auto* lg = static_cast<const _Float16*>(_logits.contents());
  std::vector<float> f32((std::size_t)N);
  for (int i = 0; i < N; ++i) { f32[(std::size_t)i] = (float)lg[i]; }
  return sample_logits_local_(f32.data(), N, sp, rng);
}

void
MetalMossLocalModel::gather_embed_(int k, int code)
{
  const int H = _cfg.lt.hidden;
  const auto* src = static_cast<const _Float16*>(
      _embeds[(std::size_t)k].contents());
  std::memcpy(_einp.contents(), src + (std::size_t)code * H,
              (std::size_t)H * 2);
}

void
MetalMossLocalModel::encode_head_(ComputeEncoder& e, int k,
                                  const SharedBuffer& h)
{
  const int K = _cfg.lt.hidden, N = _cfg.audio_vocab;
  e.set_function(_fn_gemv);
  e.set_buffer(0, h);
  e.set_buffer(1, _heads[(std::size_t)k]);
  e.set_buffer(2, _logits);
  e.set_constant(3, K);
  e.set_constant(4, N);
  e.dispatch({32, (unsigned)(((N + 7) / 8) * 2), 1}, {32, 2, 1});
}

void
MetalMossLocalModel::encode_argmax_(ComputeEncoder& e, int k)
{
  e.set_function(_fn_argmax);
  e.set_buffer(0, _logits);
  e.set_buffer(1, _codes, (std::size_t)k * sizeof(std::int32_t));
  e.set_constant(2, _cfg.audio_vocab);
  e.dispatch({256, 1, 1}, {256, 1, 1});
}

void
MetalMossLocalModel::encode_sample_(ComputeEncoder& e, int k,
                                    const MossSampling& sp, std::uint32_t seed)
{
  const int   n_iter   = 24;      // bisection iters (unused when use_hist=1)
  const int   use_hist = 1;       // histogram nucleus (qwen default)
  const float min_p    = 0.0f;
  e.set_function(_fn_sample);
  e.set_buffer(0, _logits);
  e.set_buffer(1, _codes, (std::size_t)k * sizeof(std::int32_t));
  e.set_constant(2, _cfg.audio_vocab);
  e.set_constant(3, sp.temperature);
  e.set_constant(4, sp.top_p);
  e.set_constant(5, seed);
  e.set_buffer(6, _lc_ws);
  e.set_constant(7, n_iter);
  e.set_constant(8, sp.top_k);
  e.set_constant(9, min_p);
  e.set_constant(10, use_hist);
  e.dispatch({256, 1, 1}, {256, 1, 1});
}

void
MetalMossLocalModel::encode_gather_(ComputeEncoder& e, int k)
{
  const int H = _cfg.lt.hidden;
  e.set_function(_fn_embed_gather);
  e.set_buffer(0, _codes, (std::size_t)k * sizeof(std::int32_t));  // ids[0]
  e.set_buffer(1, _embeds[(std::size_t)k]);
  e.set_buffer(2, _einp);
  e.set_constant(3, H);
  e.dispatch({(unsigned)H, 1, 1}, {256, 1, 1});
}

std::vector<int>
MetalMossLocalModel::decode_frame_fused_(const SharedBuffer& seed,
                                         const MossSampling* sp,
                                         std::uint64_t seed64)
{
  const int NV = _cfg.n_vq, H = _cfg.lt.hidden;
  std::vector<int> codes((std::size_t)NV, 0);
  _lt->reset_frame();
  CommandStream s = _mc->make_command_stream();
  if (!s.valid()) { return codes; }
  {
    ComputeEncoder e = s.begin_compute();
    for (int k = 0; k < NV; ++k) {
      if (k == 0) {
        _lt->encode_step(e, seed, 0);
      } else {
        encode_gather_(e, k - 1);       // _embeds[k-1](_codes[k-1]) -> _einp
        _lt->encode_step(e, _einp, k);
      }
      const SharedBuffer& h = _lt->hidden();
      if (k == 0) {
        // Snapshot the pos-0 hidden for the continue/stop head: the next
        // step overwrites the transformer's _hidden, but text_decision_
        // reads last_h0() after this frame's single commit.
        e.set_function(_fn_copy);
        e.set_buffer(0, h);
        e.set_buffer(1, _h0);
        e.set_constant(2, 0);           // off
        e.set_constant(3, H);           // N
        e.dispatch({(unsigned)H, 1, 1}, {256, 1, 1});
      }
      encode_head_(e, k, h);
      if (sp == nullptr) {
        encode_argmax_(e, k);
      } else {
        // Decorrelate the per-codebook RNG within the frame (splitmix mix).
        const std::uint32_t sd = (std::uint32_t)(
            (seed64 ^ (0x9E3779B97F4A7C15ULL * (std::uint64_t)(k + 1))) >> 29);
        encode_sample_(e, k, *sp, sd);
      }
    }
  }
  s.commit().wait();
  const auto* c = static_cast<const std::int32_t*>(_codes.contents());
  for (int k = 0; k < NV; ++k) { codes[(std::size_t)k] = (int)c[k]; }
  return codes;
}

std::vector<int>
MetalMossLocalModel::decode_frame_greedy(const SharedBuffer& seed)
{
  return decode_frame_fused_(seed, nullptr, 0);
}

std::vector<int>
MetalMossLocalModel::decode_frame_greedy_ref(const SharedBuffer& seed)
{
  std::vector<int> codes((std::size_t)_cfg.n_vq, 0);
  _lt->reset_frame();
  const SharedBuffer* h = _lt->step(seed);
  if (h == nullptr) { return codes; }
  std::memcpy(_h0.contents(), h->contents(), (std::size_t)_cfg.lt.hidden * 2);
  codes[0] = argmax_head_(0, *h);
  for (int k = 1; k < _cfg.n_vq; ++k) {
    gather_embed_(k - 1, codes[(std::size_t)(k - 1)]);
    h = _lt->step(_einp);
    if (h == nullptr) { break; }
    codes[(std::size_t)k] = argmax_head_(k, *h);
  }
  return codes;
}

std::vector<int>
MetalMossLocalModel::decode_frame_sampled(const SharedBuffer& seed,
                                          const MossSampling& sp,
                                          std::uint64_t& rng_state)
{
  const std::uint64_t s = rng_state != 0 ? rng_state : 0x123456789abcdefULL;
  const MossSampling* spp = sp.temperature > 0.0f ? &sp : nullptr;
  std::vector<int> codes = decode_frame_fused_(seed, spp, s);
  // Advance the deterministic per-frame stream (LCG) for the next frame.
  rng_state = s * 6364136223846793005ULL + 1442695040888963407ULL;
  return codes;
}

void
MetalMossLocalModel::decode_frame_teacher(const SharedBuffer& seed,
                                          const std::vector<int>& tf_codes,
                                          std::vector<float>& hiddens_out,
                                          std::vector<int>& argmax_out)
{
  const int H = _cfg.lt.hidden, NV = _cfg.n_vq;
  hiddens_out.assign((std::size_t)NV * H, 0.0f);
  argmax_out.assign((std::size_t)NV, 0);
  auto copy_hidden = [&](int k, const SharedBuffer& h) {
    const auto* s = static_cast<const _Float16*>(h.contents());
    float* d = hiddens_out.data() + (std::size_t)k * H;
    for (int i = 0; i < H; ++i) { d[i] = (float)s[i]; }
  };

  _lt->reset_frame();
  const SharedBuffer* h = _lt->step(seed);
  if (h == nullptr) { return; }
  copy_hidden(0, *h);
  argmax_out[0] = argmax_head_(0, *h);
  for (int k = 1; k < NV; ++k) {
    gather_embed_(k - 1, tf_codes[(std::size_t)(k - 1)]);
    h = _lt->step(_einp);
    if (h == nullptr) { return; }
    copy_hidden(k, *h);
    argmax_out[(std::size_t)k] = argmax_head_(k, *h);
  }
}

}  // namespace vpipe::genai

#include "generative-models/moss/metal-moss-rt-model.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/perf-event.h"
#include "common/perf-scope.h"
#include "generative-models/context-manager.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
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
  float f; std::memcpy(&f, &u, 4); return f;
}
std::uint16_t f32_to_bf16(float f)
{
  std::uint32_t x; std::memcpy(&x, &f, 4);
  const std::uint32_t r = x + 0x7fffu + ((x >> 16) & 1u);   // round-to-nearest
  return (std::uint16_t)(r >> 16);
}
SharedBuffer load_bf16(const MetalLlamaWeights& wts, MetalCompute* mc,
                       const std::string& nm)
{
  if (wts.info(nm) == nullptr) { return {}; }
  return wts.load(nm, mc);   // already bf16 in the checkpoint
}
// Load a BF16/F16 tensor as f16 (the local transformer + heads run f16 so the
// f16 fusion kernels -- argmax/lc_sample/embed_gather -- apply directly).
SharedBuffer load_f16(const MetalLlamaWeights& wts, MetalCompute* mc,
                      const std::string& nm)
{
  const auto* info = wts.info(nm);
  if (info == nullptr) { return {}; }
  SharedBuffer raw = wts.load(nm, mc);
  if (raw.empty()) { return {}; }
  std::size_t n = 1;
  for (auto d : info->shape) { n *= (std::size_t)d; }
  SharedBuffer out = mc->make_shared_buffer(n * 2);
  if (out.empty()) { return {}; }
  auto* d = static_cast<_Float16*>(out.contents());
  if (info->dtype == "F16") {
    std::memcpy(out.contents(), raw.contents(), n * 2);
  } else if (info->dtype == "BF16") {
    const auto* s = static_cast<const std::uint16_t*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) { d[i] = (_Float16)bf16_to_f32(s[i]); }
  } else {
    return {};
  }
  return out;
}

}  // namespace

std::unique_ptr<MetalMossRtModel>
MetalMossRtModel::load(const std::string& quant_dir, MetalCompute* mc,
                       const Config& cfg)
{
  auto wts = MetalLlamaWeights::open_model(quant_dir);
  if (!wts.has_value()) { return nullptr; }
  auto self = std::make_unique<MetalMossRtModel>();
  if (!self->init_(*wts, mc, cfg, quant_dir)) { return nullptr; }
  return self;
}

bool
MetalMossRtModel::init_(const MetalLlamaWeights& wts, MetalCompute* mc,
                        const Config& cfg, const std::string& quant_dir)
{
  _mc = mc;
  _session = mc->session();
  _cfg = cfg;
  // The local/depth transformer decodes in f16 (use_bf16=false) so the whole
  // frame can fuse through the f16 on-GPU sampler + embed-gather kernels; the
  // backbone stays bf16 (verified). Both are 8-bit affine.
  _cfg.local.use_bf16 = false;
  _bb = MetalQwenModel::load(quant_dir, mc, _cfg.backbone);
  if (!_bb) { return false; }
  _lt = MetalQwenModel::load(quant_dir, mc, _cfg.local);
  if (!_lt) { return false; }

  const int NV = cfg.n_vq;
  _text_embed = load_bf16(wts, mc, "embed_tokens.0.weight");   // backbone, bf16
  if (_text_embed.empty()) { return false; }
  _audio_embed.resize((std::size_t)NV);
  for (int i = 0; i < NV; ++i) {
    _audio_embed[(std::size_t)i] =
        load_bf16(wts, mc, "embed_tokens." + std::to_string(i + 1) + ".weight");
    if (_audio_embed[(std::size_t)i].empty()) { return false; }
  }
  // Local feedback embeds + output heads run f16 (the fused depth loop).
  _local_embed.resize((std::size_t)(NV - 1));
  for (int i = 0; i < NV - 1; ++i) {
    _local_embed[(std::size_t)i] = load_f16(
        wts, mc,
        "local_transformer.model.embed_tokens." + std::to_string(i) + ".weight");
    if (_local_embed[(std::size_t)i].empty()) { return false; }
  }
  _heads.resize((std::size_t)NV);
  for (int i = 0; i < NV; ++i) {
    _heads[(std::size_t)i] = load_f16(
        wts, mc,
        "local_transformer.local_lm_heads." + std::to_string(i) + ".weight");
    if (_heads[(std::size_t)i].empty()) { return false; }
  }

  _logits = mc->make_shared_buffer((std::size_t)cfg.audio_vocab * 2);
  _lc_ws  = mc->make_shared_buffer((std::size_t)cfg.audio_vocab * 2);
  _codes  = mc->make_shared_buffer((std::size_t)NV * sizeof(std::int32_t));
  _hist   = mc->make_shared_buffer(
      (std::size_t)NV * kRepWindow * sizeof(std::int32_t));
  _hist_len.assign((std::size_t)NV, 0);
  _hist_host.assign((std::size_t)NV, {});
  _lib_dense = mc->load_library("dense_gemm");        // f16
  _lib_elt   = mc->load_library("llm_elementwise");
  if (!_lib_dense.valid() || !_lib_elt.valid()) { return false; }
  _fn_gemv   = _lib_dense.function("dense_gemv_t_f16");
  _fn_argmax = _lib_elt.function("argmax_f16");
  _fn_sample = _lib_elt.function("lc_sample_f16");
  _fn_gather = _lib_elt.function("embed_gather_f16");
  _fn_rep    = _lib_elt.function("rep_penalty_f16");
  return _fn_gemv.valid() && _fn_argmax.valid() && _fn_sample.valid() &&
         _fn_gather.valid() && _fn_rep.valid() && !_logits.empty() &&
         !_lc_ws.empty() && !_codes.empty() && !_hist.empty();
}

SharedBuffer
MetalMossRtModel::assemble_embeds_(
    const std::vector<std::vector<std::int32_t>>& grid, int start, int n)
{
  const int H = _cfg.hidden, NV = _cfg.n_vq, pad = _cfg.audio_pad_code;
  SharedBuffer out = _mc->make_shared_buffer((std::size_t)n * H * 2);
  auto* o = static_cast<std::uint16_t*>(out.contents());
  const auto* te = static_cast<const std::uint16_t*>(_text_embed.contents());
  for (int r = 0; r < n; ++r) {
    const auto& row = grid[(std::size_t)(start + r)];
    std::uint16_t* orow = o + (std::size_t)r * H;
    // Start from the text embed (bf16), then ADD each active audio embed,
    // rounding back to bf16 after each add (matches the reference MLX bf16 sum).
    const int tid = row[0];
    std::memcpy(orow, te + (std::size_t)tid * H, (std::size_t)H * 2);
    for (int i = 0; i < NV; ++i) {
      const int code = row[(std::size_t)(i + 1)];
      if (code == pad) { continue; }          // padding_idx 1024 => zero embed
      const auto* ae = static_cast<const std::uint16_t*>(
          _audio_embed[(std::size_t)i].contents());
      const std::uint16_t* arow = ae + (std::size_t)code * H;
      for (int h = 0; h < H; ++h) {
        orow[h] = f32_to_bf16(bf16_to_f32(orow[h]) + bf16_to_f32(arow[h]));
      }
    }
  }
  return out;
}

void
MetalMossRtModel::encode_head_(ComputeEncoder& e, int k, const SharedBuffer& h)
{
  const int K = _cfg.hidden, N = _cfg.audio_vocab;
  e.set_function(_fn_gemv);
  e.set_buffer(0, h);
  e.set_buffer(1, _heads[(std::size_t)k]);
  e.set_buffer(2, _logits);
  e.set_constant(3, K);
  e.set_constant(4, N);
  e.dispatch({32, (unsigned)(((N + 7) / 8) * 2), 1}, {32, 2, 1});
}

void
MetalMossRtModel::encode_argmax_(ComputeEncoder& e, int k)
{
  e.set_function(_fn_argmax);
  e.set_buffer(0, _logits);
  e.set_buffer(1, _codes, (std::size_t)k * sizeof(std::int32_t));
  e.set_constant(2, _cfg.audio_vocab);
  e.dispatch({256, 1, 1}, {256, 1, 1});
}

void
MetalMossRtModel::encode_sample_(ComputeEncoder& e, int k,
                                 const MossSampling& sp, std::uint32_t seed)
{
  const int   n_iter   = 24;    // bisection iters (unused when use_hist=1)
  const int   use_hist = 1;     // histogram nucleus
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
MetalMossRtModel::encode_gather_(ComputeEncoder& e, int k,
                                 const SharedBuffer& out)
{
  e.set_function(_fn_gather);
  e.set_buffer(0, _codes, (std::size_t)k * sizeof(std::int32_t));  // ids[0]
  e.set_buffer(1, _local_embed[(std::size_t)k]);
  e.set_buffer(2, out);
  e.set_constant(3, _cfg.hidden);
  e.dispatch({(unsigned)_cfg.hidden, 1, 1}, {256, 1, 1});
}

void
MetalMossRtModel::encode_rep_penalty_(ComputeEncoder& e, int k, float penalty)
{
  const int n = _hist_len[(std::size_t)k];
  if (n <= 0 || penalty == 1.0f) { return; }
  e.set_function(_fn_rep);
  e.set_buffer(0, _logits);
  e.set_buffer(1, _hist, (std::size_t)k * kRepWindow * sizeof(std::int32_t));
  e.set_constant(2, n);
  e.set_constant(3, penalty);
  e.set_constant(4, _cfg.audio_vocab);
  e.dispatch({1, 1, 1}, {1, 1, 1});
}

std::vector<int>
MetalMossRtModel::decode_frame_fused_(const SharedBuffer& seed_bb,
                                      const MossSampling* sp,
                                      std::uint64_t seed64)
{
  const int NV = _cfg.n_vq, H = _cfg.hidden;
  std::vector<int> codes((std::size_t)NV, 0);

  // Refresh the GPU repetition-penalty history from the rolling host history
  // (the last kRepWindow codes per codebook, across previous frames).
  const bool rep_on = sp != nullptr && sp->repetition_penalty != 1.0f;
  if (rep_on) {
    auto* hd = static_cast<std::int32_t*>(_hist.contents());
    for (int k = 0; k < NV; ++k) {
      const auto& hk = _hist_host[(std::size_t)k];
      const int n = std::min<int>(kRepWindow, (int)hk.size());
      _hist_len[(std::size_t)k] = n;
      for (int j = 0; j < n; ++j) {
        hd[(std::size_t)k * kRepWindow + j] = hk[hk.size() - (std::size_t)n + j];
      }
    }
  }

  ContextManager* cm = _lt->context_manager();
  const ContextId lcid = cm->acquire_root();
  // The local decode input residual (f16 [H]); pos 0 = the backbone seed
  // (bf16 -> f16 host convert), pos k = the on-GPU embed gather targets it.
  SharedBuffer& dx = _lt->decode_input_buffer();
  if (dx.empty()) { cm->release(lcid); return codes; }
  {
    auto* d = static_cast<_Float16*>(dx.contents());
    const auto* s = static_cast<const std::uint16_t*>(seed_bb.contents());
    for (int i = 0; i < H; ++i) { d[i] = (_Float16)bf16_to_f32(s[i]); }
  }

  CommandStream st = _mc->make_command_stream();
  if (!st.valid()) { cm->release(lcid); return codes; }
  {
    ComputeEncoder e = st.begin_compute();
    for (int k = 0; k < NV; ++k) {
      if (k > 0) {
        // Feed codebook (k-1)'s sampled code back in: gather its local embed
        // straight into the decode input buffer (GPU; no host round-trip).
        encode_gather_(e, k - 1, dx);
      }
      const SharedBuffer* h = _lt->encode_decode_prewritten(e, lcid);
      if (h == nullptr) { break; }
      encode_head_(e, k, *h);
      if (sp == nullptr) {
        encode_argmax_(e, k);
      } else {
        if (rep_on) { encode_rep_penalty_(e, k, sp->repetition_penalty); }
        // Decorrelate the per-codebook RNG within the frame (splitmix mix).
        const std::uint32_t sd = (std::uint32_t)(
            (seed64 ^ (0x9E3779B97F4A7C15ULL * (std::uint64_t)(k + 1))) >> 29);
        encode_sample_(e, k, *sp, sd);
      }
    }
  }
  st.commit().wait();
  cm->release(lcid);
  const auto* c = static_cast<const std::int32_t*>(_codes.contents());
  for (int k = 0; k < NV; ++k) { codes[(std::size_t)k] = (int)c[k]; }
  // Append this frame's codes to the rolling per-codebook history (capped).
  if (rep_on) {
    for (int k = 0; k < NV; ++k) {
      auto& hk = _hist_host[(std::size_t)k];
      hk.push_back(codes[(std::size_t)k]);
      if ((int)hk.size() > kRepWindow) {
        hk.erase(hk.begin(), hk.end() - kRepWindow);
      }
    }
  }
  return codes;
}

std::vector<int>
MetalMossRtModel::local_frame(const SharedBuffer& seed_hidden,
                              const MossSampling& sp, std::uint64_t& rng_state)
{
  const MossSampling* spp = sp.temperature > 0.0f ? &sp : nullptr;
  const std::uint64_t s = rng_state != 0 ? rng_state : 0x123456789abcdefULL;
  std::vector<int> codes = decode_frame_fused_(seed_hidden, spp, s);
  rng_state = s * 6364136223846793005ULL + 1442695040888963407ULL;
  return codes;
}

std::vector<std::vector<int>>
MetalMossRtModel::generate(
    const std::vector<std::vector<std::int32_t>>& prompt_grid,
    const std::vector<std::int32_t>& text_ids, int max_frames,
    const MossSampling& audio_sp, std::uint64_t seed, const FrameCb& on_frame)
{
  std::vector<std::vector<int>> frames;
  const int H = _cfg.hidden, NV = _cfg.n_vq;
  if (prompt_grid.empty() || max_frames <= 0) { return frames; }

  // Fresh repetition-penalty history for this utterance.
  for (auto& hk : _hist_host) { hk.clear(); }

  std::uint64_t rng = seed != 0 ? seed
                                : (std::uint64_t)std::random_device{}();

  // Prefill grid = prompt prefix + first min(len,12) text rows, BOS at ch1 of
  // the last prefill row (matches inferencer._build_prefill_batch).
  std::vector<std::vector<std::int32_t>> full = prompt_grid;
  const int prefill_n = std::min<int>((int)text_ids.size(), 12);
  for (int j = 0; j < prefill_n; ++j) {
    std::vector<std::int32_t> r((std::size_t)(NV + 1),
                                (std::int32_t)_cfg.audio_pad_code);
    r[0] = text_ids[(std::size_t)j];
    full.push_back(std::move(r));
  }
  if (prefill_n > 0) { full.back()[1] = _cfg.audio_bos; }
  const int seq = (int)full.size();

  ContextManager* cm = _bb->context_manager();
  const ContextId cid = cm->acquire_root();

  SharedBuffer seed_h;
  {
    PerfAuxScope _perf(_session, kPerfLaneLLM, kGvidLlmPrefill,
                       kPerfLlmPrefillBegin, (std::uint64_t)seq);
    SharedBuffer emb = assemble_embeds_(full, 0, seq);
    seed_h = _bb->forward_embeddings_hidden(cid, emb, seq);
  }
  if (seed_h.empty()) { cm->release(cid); return frames; }
  const SharedBuffer* cur = &seed_h;   // live backbone hidden (prefill, then
                                       // _bb's internal decode scratch)

  // The whole depth frame is fused into ONE command buffer (on-GPU sample +
  // embed-gather); rep_pen is dropped in the fused path (sampled on-GPU).
  const MossSampling* spp = audio_sp.temperature > 0.0f ? &audio_sp : nullptr;
  auto next_frame = [&](const SharedBuffer& seed) {
    std::vector<int> c = decode_frame_fused_(seed, spp, rng);
    rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return c;
  };
  // One decode profiler event PER FRAME (value 1 = one 12.5 Hz frame), same as
  // the v1.5 path -- so the timeline shows per-frame decode steps, not a single
  // block spanning the whole utterance. Frame 0's backbone hidden came from the
  // prefill; frames 1+ wrap the backbone step + the fused local depth frame.
  std::vector<int> codes;
  {
    PerfAuxScope _perf(_session, kPerfLaneLLM, kGvidLlmDecode,
                       kPerfLlmDecodeBegin, 1);
    codes = next_frame(*cur);
  }
  if (codes.empty() || codes[0] == _cfg.audio_eos) {
    cm->release(cid); return frames;
  }
  frames.push_back(codes);
  if (on_frame && !on_frame(codes)) { cm->release(cid); return frames; }

  int text_idx = prefill_n;
  for (int f = 1; f < max_frames; ++f) {
    const int nxt = text_idx < (int)text_ids.size()
                        ? text_ids[(std::size_t)text_idx] : _cfg.text_pad;
    ++text_idx;
    std::vector<std::vector<std::int32_t>> row(
        1, std::vector<std::int32_t>((std::size_t)(NV + 1)));
    row[0][0] = nxt;
    for (int i = 0; i < NV; ++i) {
      row[0][(std::size_t)(i + 1)] = codes[(std::size_t)i];
    }
    {
      PerfAuxScope _perf(_session, kPerfLaneLLM, kGvidLlmDecode,
                         kPerfLlmDecodeBegin, 1);
      SharedBuffer step_emb = assemble_embeds_(row, 0, 1);
      // Optimized qmv single-token backbone decode (reuses _bb decode scratch).
      cur = _bb->decode_embedding_hidden(cid, step_emb);
      codes = (cur == nullptr) ? std::vector<int>{} : next_frame(*cur);
    }
    if (codes.empty() || codes[0] == _cfg.audio_eos) { break; }
    frames.push_back(codes);
    if (on_frame && !on_frame(codes)) { break; }
  }
  (void)H;
  cm->release(cid);
  return frames;
}

}  // namespace vpipe::genai

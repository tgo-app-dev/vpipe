#include "generative-models/moss/metal-moss-v15-model.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/perf-event.h"
#include "common/perf-scope.h"
#include "generative-models/context-manager.h"
#include "generative-models/llama3/metal-llama-weights.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>

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
  const auto* info = wts.info(nm);
  if (info == nullptr) { return {}; }
  return wts.load(nm, mc);   // already bf16 in the checkpoint
}
}  // namespace

std::unique_ptr<MetalMossV15Model>
MetalMossV15Model::load(const std::string& quant_dir, MetalCompute* mc,
                        const Config& cfg)
{
  auto wts = MetalLlamaWeights::open_model(quant_dir);
  if (!wts.has_value()) { return nullptr; }
  auto self = std::make_unique<MetalMossV15Model>();
  if (!self->init_(*wts, mc, cfg, quant_dir)) { return nullptr; }
  return self;
}

bool
MetalMossV15Model::init_(const MetalLlamaWeights& wts, MetalCompute* mc,
                         const Config& cfg, const std::string& quant_dir)
{
  _mc = mc;
  _session = mc->session();
  _cfg = cfg;
  _bb = MetalQwenModel::load(quant_dir, mc, cfg.backbone);
  if (!_bb) { return false; }
  _lm = MetalMossLocalModel::load(quant_dir, mc, cfg.local);
  if (!_lm) { return false; }

  _text_embed = load_bf16(wts, mc, "transformer.embed_tokens.weight");
  if (_text_embed.empty()) { return false; }
  _audio_embed.resize((std::size_t)cfg.local.n_vq);
  for (int i = 0; i < cfg.local.n_vq; ++i) {
    _audio_embed[(std::size_t)i] =
        load_bf16(wts, mc, "audio_embeddings." + std::to_string(i) + ".weight");
    if (_audio_embed[(std::size_t)i].empty()) { return false; }
  }
  // local_text_lm_head is bf16 [2, hidden] -> f16.
  {
    const std::string nm = "local_text_lm_head.weight";
    const auto* info = wts.info(nm);
    SharedBuffer raw = wts.load(nm, mc);
    if (info == nullptr || raw.empty()) { return false; }
    const std::size_t n = 2u * (std::size_t)cfg.local.lt.hidden;
    _ltext_head = mc->make_shared_buffer(n * 2);
    auto* d = static_cast<_Float16*>(_ltext_head.contents());
    const auto* s = static_cast<const std::uint16_t*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) { d[i] = (_Float16)bf16_to_f32(s[i]); }
  }

  const int H = cfg.local.lt.hidden;
  _seed_bb = mc->make_shared_buffer((std::size_t)H * 2);
  _seed_f16 = mc->make_shared_buffer((std::size_t)H * 2);
  _ltext_logits = mc->make_shared_buffer(2 * 2);
  _lib_dense = mc->load_library("dense_gemm");
  if (!_lib_dense.valid()) { return false; }
  _fn_gemv = _lib_dense.function("dense_gemv_t_f16");
  return _fn_gemv.valid() && !_seed_bb.empty() && !_seed_f16.empty();
}

SharedBuffer
MetalMossV15Model::assemble_embeds_(
    const std::vector<std::vector<std::int32_t>>& grid, int start, int n)
{
  const int H = _cfg.local.lt.hidden, NV = _cfg.local.n_vq;
  const int pad = _cfg.audio_pad_code;
  SharedBuffer out = _mc->make_shared_buffer((std::size_t)n * H * 2);
  auto* o = static_cast<std::uint16_t*>(out.contents());
  const auto* te = static_cast<const std::uint16_t*>(_text_embed.contents());
  for (int r = 0; r < n; ++r) {
    const auto& row = grid[(std::size_t)(start + r)];
    const int tid = row[0];
    std::uint16_t* orow = o + (std::size_t)r * H;
    for (int j = 0; j < H; ++j) {
      float acc = bf16_to_f32(te[(std::size_t)tid * H + j]);
      for (int i = 0; i < NV; ++i) {
        const int code = row[(std::size_t)(i + 1)];
        if (code != pad) {
          const auto* ae =
              static_cast<const std::uint16_t*>(_audio_embed[(std::size_t)i]
                                                    .contents());
          acc += bf16_to_f32(ae[(std::size_t)code * H + j]);
        }
      }
      orow[j] = f32_to_bf16(acc);
    }
  }
  return out;
}

int
MetalMossV15Model::text_decision_(const SharedBuffer& h0)
{
  const int H = _cfg.local.lt.hidden;
  CommandStream s = _mc->make_command_stream();
  {
    ComputeEncoder e = s.begin_compute();
    e.set_function(_fn_gemv);
    e.set_buffer(0, h0); e.set_buffer(1, _ltext_head);
    e.set_buffer(2, _ltext_logits);
    e.set_constant(3, H); e.set_constant(4, 2);
    e.dispatch({32, 2, 1}, {32, 2, 1});
  }
  s.commit().wait();
  const auto* lg = static_cast<const _Float16*>(_ltext_logits.contents());
  const int idx = ((float)lg[1] > (float)lg[0]) ? 1 : 0;
  return idx == 0 ? _cfg.audio_assistant_slot : _cfg.audio_end_token;
}

std::vector<std::vector<int>>
MetalMossV15Model::generate(
    const std::vector<std::vector<std::int32_t>>& grid, int max_frames,
    const MossSampling& audio_sp, std::uint64_t seed)
{
  std::vector<std::vector<int>> frames;
  const int H = _cfg.local.lt.hidden, NV = _cfg.local.n_vq;
  const int seq = (int)grid.size();
  if (seq <= 0) { return frames; }

  const bool sample = audio_sp.temperature > 0.0f;
  std::uint64_t rng = seed != 0 ? seed
                                : (std::uint64_t)std::random_device{}();

  // Opt-in per-phase decode profiler (VPIPE_MOSS15_PROFILE): times the four
  // serial per-frame phases (local RVQ decode, continue/stop head, CPU embed
  // assembly, backbone decode step) so the throughput bottleneck is visible.
  const bool prof = std::getenv("VPIPE_MOSS15_PROFILE") != nullptr;
  using clk = std::chrono::steady_clock;
  auto ms = [](clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  double t_local = 0, t_text = 0, t_asm = 0, t_bb = 0;
  int nf = 0;
  const auto t_gen0 = clk::now();

  ContextManager* cm = _bb->context_manager();
  const ContextId cid = cm->acquire_root();

  SharedBuffer emb = assemble_embeds_(grid, 0, seq);
  SharedBuffer s0;
  {
    PerfAuxScope _perf(_session, kPerfLaneLLM, kGvidLlmPrefill,
                       kPerfLlmPrefillBegin, (std::uint64_t)seq);
    s0 = _bb->forward_embeddings_hidden(cid, emb, seq);
  }
  if (s0.empty()) { cm->release(cid); return frames; }
  std::memcpy(_seed_bb.contents(), s0.contents(), (std::size_t)H * 2);

  for (int f = 0; f < max_frames; ++f) {
    PerfAuxScope _perf(_session, kPerfLaneLLM, kGvidLlmDecode,
                       kPerfLlmDecodeBegin, 1);
    const auto* sb = static_cast<const std::uint16_t*>(_seed_bb.contents());
    auto* sf = static_cast<_Float16*>(_seed_f16.contents());
    for (int j = 0; j < H; ++j) { sf[j] = (_Float16)bf16_to_f32(sb[j]); }

    const auto t0 = clk::now();
    std::vector<int> codes =
        sample ? _lm->decode_frame_sampled(_seed_f16, audio_sp, rng)
               : _lm->decode_frame_greedy(_seed_f16);
    const auto t1 = clk::now();
    const int tok = text_decision_(_lm->last_h0());
    const auto t2 = clk::now();
    if (tok == _cfg.audio_end_token) { break; }
    frames.push_back(codes);

    std::vector<std::vector<std::int32_t>> row(
        1, std::vector<std::int32_t>((std::size_t)(1 + NV)));
    row[0][0] = _cfg.audio_assistant_slot;
    for (int i = 0; i < NV; ++i) {
      row[0][(std::size_t)(i + 1)] = codes[(std::size_t)i];
    }
    SharedBuffer re = assemble_embeds_(row, 0, 1);
    const auto t3 = clk::now();
    const SharedBuffer* nh = _bb->decode_embedding_hidden(cid, re);
    if (nh == nullptr || nh->empty()) { break; }
    std::memcpy(_seed_bb.contents(), nh->contents(), (std::size_t)H * 2);
    const auto t4 = clk::now();
    if (prof) {
      t_local += ms(t0, t1); t_text += ms(t1, t2);
      t_asm += ms(t2, t3);   t_bb += ms(t3, t4); ++nf;
    }
  }

  if (prof && nf > 0) {
    const double wall = ms(t_gen0, clk::now());
    std::fprintf(stderr,
        "[moss15-prof] %d frames | %.1f frame/s | per-frame ms: "
        "local=%.2f text=%.2f asm=%.2f backbone=%.2f (sum=%.2f)\n",
        nf, 1000.0 * nf / wall, t_local / nf, t_text / nf, t_asm / nf,
        t_bb / nf, (t_local + t_text + t_asm + t_bb) / nf);
  }

  cm->release(cid);
  return frames;
}

}  // namespace vpipe::genai

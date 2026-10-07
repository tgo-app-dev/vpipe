#include "generative-models/moss/metal-moss-tts-model.h"

#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/weight-set.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "common/perf-event.h"
#include "common/perf-scope.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace vpipe::genai {

using metal_compute::ComputeEncoder;
using metal_compute::SharedBuffer;

namespace {
inline float
bf16_to_f32_(std::uint16_t b)
{
  std::uint32_t bits = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}
inline std::uint16_t
f32_to_bf16_(float f)
{
  std::uint32_t b;
  std::memcpy(&b, &f, sizeof(b));
  b += 0x7fffu + ((b >> 16) & 1u);   // round-to-nearest-even
  return (std::uint16_t)(b >> 16);
}
int
geti_(const FlexData::ConstObjectView& o, const char* k, int d)
{
  return o.contains(k) ? (int)o.at(k).as_int(d) : d;
}
double
getr_(const FlexData::ConstObjectView& o, const char* k, double d)
{
  return o.contains(k) ? o.at(k).as_real(d) : d;
}

// Host sampler over logit[0..n): temperature / top_k / top_p, with an optional
// repetition penalty against `seen` (HF style: scale a seen logit by 1/pen if
// >0 else *pen). `blocked` (small) excludes ids; temperature <= 0 => greedy
// argmax. `cand`/`prob` are caller scratch (reused across steps, no per-call
// alloc). Returns the chosen index.
int
sample_logits_(const float* logit, int n, const MossSampling& sp,
               const std::vector<int>& blocked,
               const std::unordered_set<int>* seen,
               std::vector<int>& cand, std::vector<float>& prob,
               std::mt19937_64& rng)
{
  auto is_blocked = [&](int i) {
    for (int b : blocked) { if (b == i) { return true; } }
    return false;
  };
  const bool rep = sp.repetition_penalty != 1.0f && seen != nullptr;
  auto adj = [&](int i) -> float {     // logit after repetition penalty
    float v = logit[i];
    if (rep && seen->count(i)) {
      v = (v > 0.0f) ? v / sp.repetition_penalty : v * sp.repetition_penalty;
    }
    return v;
  };
  if (sp.temperature <= 0.0f) {                       // greedy
    float bv = -std::numeric_limits<float>::infinity();
    int best = -1;
    for (int i = 0; i < n; ++i) {
      if (is_blocked(i)) { continue; }
      const float v = adj(i);
      if (v > bv) { bv = v; best = i; }
    }
    return best;
  }
  cand.clear();
  for (int i = 0; i < n; ++i) { if (!is_blocked(i)) { cand.push_back(i); } }
  if (cand.empty()) { return 0; }
  // top_k: select the k highest-adjusted-logit candidates, sorted desc.
  int k = (sp.top_k > 0 && sp.top_k < (int)cand.size())
              ? sp.top_k
              : (int)cand.size();
  auto gt = [&](int a, int b) { return adj(a) > adj(b); };
  if (k < (int)cand.size()) {
    std::nth_element(cand.begin(), cand.begin() + (k - 1), cand.end(), gt);
    cand.resize((std::size_t)k);
  }
  std::sort(cand.begin(), cand.end(), gt);
  // temperature softmax over the kept set, then top_p nucleus truncation.
  const float inv_t = 1.0f / sp.temperature;
  const float mx = adj(cand[0]) * inv_t;
  prob.resize(cand.size());
  double sum = 0.0;
  for (std::size_t i = 0; i < cand.size(); ++i) {
    const float p = std::exp(adj(cand[i]) * inv_t - mx);
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

std::unique_ptr<MetalMossTtsModel>
MetalMossTtsModel::load(const std::string& model_dir,
                        metal_compute::MetalCompute* mc,
                        const LoadOptions& opts)
{
  // No session to ask, so this opens a PRIVATE set: correct, just not
  // shared with whatever else has the same checkpoint open.
  return load(WeightSet::open(model_dir, nullptr), mc, opts);
}

std::unique_ptr<MetalMossTtsModel>
MetalMossTtsModel::load(std::shared_ptr<WeightSet> ws,
                        metal_compute::MetalCompute* mc,
                        const LoadOptions& opts)
{
  if (mc == nullptr || !mc->valid() || ws == nullptr) {
    return nullptr;
  }
  const std::string model_dir = ws->dir();
  auto self = std::unique_ptr<MetalMossTtsModel>(new MetalMossTtsModel());
  self->_ws = ws;
  self->_mc = mc;
  Config cfg;

  // Backbone (dense Qwen3) shape, overridable from config.json/language_config.
  MetalQwenModel::Config bc;
  bc.n_layers   = 36;
  bc.hidden     = 4096;
  bc.n_heads    = 32;
  bc.n_kv_heads = 8;
  bc.head_dim   = 128;
  bc.ffn_inner  = 12288;
  bc.vocab      = 155648;
  bc.rope_theta = 1.0e6f;
  bc.rms_eps    = 1e-6f;
  bc.quant_bits = 8;      // when quantized and config.json does not say

  {
    std::ifstream in(model_dir + "/config.json");
    if (in) {
      try {
        FlexData root = FlexData::from_json(in);
        if (root.is_object()) {
          const auto ro = root.as_object();
          cfg.n_vq            = geti_(ro, "n_vq", cfg.n_vq);
          cfg.audio_vocab     = geti_(ro, "audio_vocab_size", cfg.audio_vocab);
          cfg.audio_pad_code  = geti_(ro, "audio_pad_code", cfg.audio_vocab);
          cfg.sampling_rate   = geti_(ro, "sampling_rate", cfg.sampling_rate);
          cfg.audio_start     =
              geti_(ro, "audio_start_token_id", cfg.audio_start);
          cfg.audio_end       =
              geti_(ro, "audio_end_token_id", cfg.audio_end);
          cfg.audio_user_slot =
              geti_(ro, "audio_user_slot_token_id", cfg.audio_user_slot);
          cfg.audio_gen_slot  =
              geti_(ro, "audio_assistant_gen_slot_token_id",
                    cfg.audio_gen_slot);
          cfg.audio_delay_slot =
              geti_(ro, "audio_assistant_delay_slot_token_id",
                    cfg.audio_delay_slot);
          // A model-quantize'd (or mlx-community) dir states its affine
          // width; a raw bf16 checkpoint has no block and loads dense.
          if (ro.contains("quantization")) {
            const FlexData q = ro.at("quantization");
            if (q.is_object()) {
              bc.quant_bits = geti_(q.as_object(), "bits", bc.quant_bits);
            }
          }
          if (ro.contains("language_config")) {
            const FlexData lc = ro.at("language_config");
            if (lc.is_object()) {
              const auto lco = lc.as_object();
              bc.n_layers   = geti_(lco, "num_hidden_layers", bc.n_layers);
              bc.hidden     = geti_(lco, "hidden_size", bc.hidden);
              bc.n_heads    = geti_(lco, "num_attention_heads", bc.n_heads);
              bc.n_kv_heads = geti_(lco, "num_key_value_heads", bc.n_kv_heads);
              bc.head_dim   = geti_(lco, "head_dim", bc.head_dim);
              bc.ffn_inner  = geti_(lco, "intermediate_size", bc.ffn_inner);
              bc.vocab      = geti_(lco, "vocab_size", bc.vocab);
              bc.rope_theta = (float)getr_(lco, "rope_theta", bc.rope_theta);
              bc.rms_eps    = (float)getr_(lco, "rms_norm_eps", bc.rms_eps);
              cfg.pad_token = geti_(lco, "pad_token_id", cfg.pad_token);
            }
          }
        }
      } catch (...) {
        // Fall back to the 8B defaults baked above.
      }
    }
  }
  cfg.hidden = bc.hidden;
  cfg.vocab  = bc.vocab;

  bc.rotary_dim        = bc.head_dim;   // Qwen3 full rotary
  bc.full_attn_interval = 1;            // dense: every layer is full-attn
  bc.tie_embeddings    = false;
  bc.use_bf16          = true;          // match the reference dtype
  bc.dense             = true;
  // Plain Qwen3 RMSNorm (weight * h). Consulted only when the backbone is
  // raw bf16 (MOSS-TTS / MOSS-TTS-v1.5 as published): left at the Qwen3.5
  // default it folds +1 into every norm and the model emits no audio.
  bc.zero_centered_norm = false;
  bc.attn_output_gate  = false;
  bc.weight_prefix     = "language_";   // -> language_model.layers / .norm
  bc.backbone_only     = true;
  bc.max_seq           = 16384;
  bc.page_tokens       = 256;
  // How the backbone is held. Both apply only to an unquantized checkpoint
  // (the backbone ignores them for a pack): in-memory w8, and layer
  // streaming through decode when the plan said the box cannot hold it.
  bc.load_quant_bits   = opts.quant_bits == 8 ? 8 : 0;
  bc.stream_decode     = opts.stream;

  self->_backbone = MetalQwenModel::load(ws, mc, bc);
  if (!self->_backbone) {
    return nullptr;
  }
  self->_cfg = cfg;

  // bf16 embedding tables + output heads (raw, unquantized in the ckpt).
  // From the SAME set as the backbone: this used to open the checkpoint a
  // second time. Copied, not Mapped -- mapping wraps the whole shard and
  // the backbone reads it by copy.
  WeightSet& wts = *ws;
  const auto cp = WeightSet::Residency::Copied;
  // trunk_w8: the text embedding and text head at w8 too. Built here from
  // an uncached read, as the backbone builds its w8 layers, and handed to
  // the backbone so they are wired with its trunk. The rows the audio
  // phase reads every step stay exact bf16: the slot tokens in, and the
  // two the text channel ranks out.
  if (opts.quant_bits == 8 && opts.trunk_w8 && cfg.vocab % 8 == 0) {
    self->_lib_qmv = mc->load_library("affine_qmv_bf16");
    self->_fn_qmv8 = self->_lib_qmv.function("affine_qmv_w8g64");
  }
  if (self->_fn_qmv8.valid()) {
    const std::vector<int> in_rows = {
        cfg.audio_gen_slot, cfg.audio_delay_slot, cfg.audio_start,
        cfg.audio_end, cfg.audio_user_slot, cfg.pad_token};
    const std::vector<int> out_rows = {cfg.audio_gen_slot,
                                       cfg.audio_delay_slot};
    SharedBuffer eq = self->quantize_table_(
        wts, "language_model.embed_tokens.weight", in_rows,
        self->_embed_exact, &self->_e_soff, &self->_e_boff);
    SharedBuffer hq;
    if (!eq.empty()) {
      hq = self->quantize_table_(wts, "lm_heads.0.weight", out_rows,
                                 self->_head_exact, &self->_h_soff,
                                 &self->_h_boff);
    }
    if (!eq.empty() && !hq.empty()) {
      self->_embed_q  = &self->_backbone->adopt_trunk(std::move(eq));
      self->_head0_q  = &self->_backbone->adopt_trunk(std::move(hq));
      self->_trunk_w8 = true;
    } else {
      self->_embed_exact.clear();
      self->_head_exact.clear();
    }
  }
  if (!self->_trunk_w8) {
    self->_embed_tokens =
        wts.tensor("language_model.embed_tokens.weight", mc, cp);
    if (self->_embed_tokens.empty()) {
      return nullptr;
    }
  }
  self->_emb_ext.resize((std::size_t)cfg.n_vq);
  for (int i = 0; i < cfg.n_vq; ++i) {
    const std::string nm = "emb_ext." + std::to_string(i) + ".weight";
    self->_emb_ext[(std::size_t)i] = wts.tensor(nm, mc, cp);
    if (self->_emb_ext[(std::size_t)i].empty()) {
      return nullptr;
    }
  }
  self->_heads.resize((std::size_t)cfg.n_vq + 1);
  self->_head_out.resize((std::size_t)cfg.n_vq + 1);
  for (int i = 0; i <= cfg.n_vq; ++i) {
    const std::string nm = "lm_heads." + std::to_string(i) + ".weight";
    const auto* info = wts.src().info(nm);
    if (i > 0 || !self->_trunk_w8) {   // the w8 text head is _head0_q
      self->_heads[(std::size_t)i] = wts.tensor(nm, mc, cp);
      if (self->_heads[(std::size_t)i].empty()) {
        return nullptr;
      }
    }
    self->_head_out[(std::size_t)i] =
        (info != nullptr && !info->shape.empty())
            ? (int)info->shape[0]
            : (i == 0 ? cfg.vocab : cfg.audio_vocab + 1);
  }

  // Dense bf16 GEMV for the heads (same kernel the backbone uses post-dequant).
  self->_lib_dense = mc->load_library("dense_gemm_bf16");
  self->_fn_dense_gemv = self->_lib_dense.function("dense_gemv_t_f16");
  if (!self->_fn_dense_gemv.valid()) {
    return nullptr;
  }
  const std::size_t logit_elts =
      (std::size_t)cfg.vocab + (std::size_t)cfg.n_vq * (cfg.audio_vocab + 1);
  self->_logits_buf = mc->make_shared_buffer(logit_elts * 2);
  // Now that the tables and heads are loaded -- they are this checkpoint's
  // trunk, read on every token -- wire what the model keeps: trunk first,
  // then the resident layers.
  self->_backbone->wire_resident();
  return self;
}

MetalMossTtsModel::MemoryPlan
MetalMossTtsModel::plan_memory(const std::string& model_dir, int quant_bits,
                               bool trunk_w8)
{
  int n_layers = 36;
  {
    std::ifstream in(model_dir + "/config.json");
    if (in) {
      try {
        FlexData root = FlexData::from_json(in);
        if (root.is_object() && root.as_object().contains("language_config")) {
          const FlexData lc = root.as_object().at("language_config");
          if (lc.is_object()) {
            n_layers = geti_(lc.as_object(), "num_hidden_layers", n_layers);
          }
        }
      } catch (...) {}
    }
  }
  return plan_memory(model_dir, "language_model.layers.", n_layers,
                     quant_bits, trunk_w8);
}

MetalMossTtsModel::MemoryPlan
MetalMossTtsModel::plan_memory(const std::string& model_dir,
                               const std::string& layer_prefix, int n_layers,
                               int quant_bits, bool trunk_w8)
{
  MemoryPlan p;
  auto wts = MetalLlamaWeights::open_model(model_dir);
  if (!wts.has_value()) { return p; }
  for (const std::string& nm : wts->tensor_names()) {
    if (const auto* ti = wts->info(nm)) { p.disk += (std::size_t)ti->nbytes; }
  }
  // What the trunk_w8 tables shed: each bf16 [N, K] held as w8 group-64,
  // as load() builds them (the handful of exact rows is noise).
  std::size_t trunk_shed = 0;
  if (quant_bits == 8 && trunk_w8) {
    for (const char* nm : {"language_model.embed_tokens.weight",
                           "lm_heads.0.weight"}) {
      const auto* ti = wts->info(nm);
      if (ti == nullptr || ti->dtype != "BF16" || ti->shape.size() != 2 ||
          ti->shape[1] % 64 != 0) {
        trunk_shed = 0;
        break;
      }
      const std::size_t w8 = (std::size_t)(ti->shape[0] * ti->shape[1]) +
                             (std::size_t)(ti->shape[0] * (ti->shape[1] / 64))
                                 * 4;
      trunk_shed += (std::size_t)ti->nbytes - w8;
    }
  }
  p.n_layers = n_layers;
  const auto rs = MetalQwenModel::raw_sizes(
      *wts, layer_prefix, n_layers, quant_bits == 8 ? 8 : 0);
  if (rs.n_layers != n_layers || rs.raw == 0) {
    // Neither raw nor a w8 pack (a 4-bit or mixed one): held as it sits,
    // and the backbone does not stream it.
    p.preload = p.disk - trunk_shed;
    p.retires = trunk_shed;
    p.floor   = 0;
    return p;
  }
  p.raw     = true;
  p.pack    = rs.pack;
  p.trunk   = p.disk - rs.raw - trunk_shed;
  p.layer   = rs.layer;
  p.preload = p.trunk + rs.built;
  p.retires = p.disk > p.preload ? p.disk - p.preload : 0;
  // Streaming holds the trunk plus a slot PAIR -- each a layer's sources
  // and the products built from them (for a pack, mostly the same bytes).
  p.floor   = p.trunk + 2 * rs.slot;
  return p;
}

std::size_t
MetalMossTtsModel::resident_bytes() const
{
  std::size_t n = _backbone ? (std::size_t)_backbone->resident_bytes() : 0;
  n += _embed_tokens.byte_size() + _logits_buf.byte_size();
  for (const auto& b : _emb_ext) { n += b.byte_size(); }
  for (const auto& b : _heads) { n += b.byte_size(); }
  // trunk_w8: the backbone owns these (adopt_trunk) but does not book them.
  if (_embed_q != nullptr) { n += _embed_q->byte_size(); }
  if (_head0_q != nullptr) { n += _head0_q->byte_size(); }
  for (const auto& kv : _embed_exact) { n += kv.second.size() * 2; }
  for (const auto& kv : _head_exact) { n += kv.second.size() * 2; }
  return n;
}

void
MetalMossTtsModel::set_residency_reserve(std::size_t bytes)
{
  if (_backbone) { _backbone->set_residency_reserve(bytes); }
}

void
MetalMossTtsModel::set_residency_schedule(int forwards)
{
  if (_backbone) { _backbone->set_residency_schedule(forwards); }
}

std::size_t
MetalMossTtsModel::release_resident_layers(std::size_t bytes)
{
  return _backbone ? _backbone->release_resident_layers(bytes) : 0;
}

bool
MetalMossTtsModel::streams() const
{
  return _backbone && _backbone->streams_decode();
}

bool
MetalMossTtsModel::streams_pack() const
{
  return _backbone && _backbone->streams_pack();
}

bool
MetalMossTtsModel::quantized_on_load() const
{
  return _backbone && _backbone->quantized_on_load();
}

int
MetalMossTtsModel::resident_layer_count() const
{
  return _backbone ? _backbone->resident_layer_count() : 0;
}

int
MetalMossTtsModel::n_layers() const
{
  return _backbone ? _backbone->config().n_layers : 0;
}

void
MetalMossTtsModel::set_i8_gemm(bool on)
{
  if (_backbone) { _backbone->set_i8_gemm(on); }
}

bool
MetalMossTtsModel::i8_gemm_enabled() const
{
  return _backbone && _backbone->i8_gemm_enabled();
}

std::uint64_t
MetalMossTtsModel::i8_gemm_count() const
{
  return _backbone ? _backbone->i8_gemm_count() : 0;
}

std::size_t
MetalMossTtsModel::kv_bytes() const
{
  if (!_backbone) { return 0; }
  ContextManager* cm = _backbone->context_manager();
  return cm != nullptr ? cm->resident_bytes() : 0;
}

SharedBuffer
MetalMossTtsModel::quantize_table_(
    WeightSet& wts, const std::string& name, const std::vector<int>& exact,
    std::unordered_map<int, std::vector<std::uint16_t>>& rows,
    std::size_t* soff, std::size_t* boff)
{
  const auto* ti = wts.src().info(name);
  if (ti == nullptr || ti->dtype != "BF16" || ti->shape.size() != 2 ||
      ti->shape[1] % 64 != 0 || ti->shape[0] <= 0) {
    return {};
  }
  const int N = (int)ti->shape[0], K = (int)ti->shape[1];
  // The build kernel the backbone's in-memory w8 layers go through, so the
  // tables quantize exactly as a layer would (scales in bf16, codes fit to
  // the rounded scale and bias).
  metal_compute::ComputeLibrary lib = _mc->load_library("affine_dequant_bf16");
  metal_compute::ComputeFunction fn = lib.function("affine_quant_rows_w8g64");
  if (!fn.valid()) { return {}; }
  // Consumed here, so read UNCACHED: caching it would keep the bf16 table
  // beside its w8 copy.
  SharedBuffer src = wts.read(name, _mc, WeightSet::Residency::Copied);
  if (src.empty()) { return {}; }
  const std::size_t codes = (std::size_t)N * K;
  const std::size_t sc = (std::size_t)N * (K / 64) * 2;
  SharedBuffer q = _mc->make_shared_buffer(codes + 2 * sc);
  if (q.empty()) { return {}; }
  metal_compute::CommandStream st = _mc->make_command_stream();
  {
    ComputeEncoder enc = st.begin_compute();
    const int one = 1, zero = 0;
    enc.set_function(fn);
    enc.set_buffer(0, src);
    enc.set_buffer(1, q, 0);
    enc.set_buffer(2, q, codes);
    enc.set_buffer(3, q, codes + sc);
    enc.set_constant(4, K);
    enc.set_constant(5, N);
    enc.set_constant(6, one);
    enc.set_constant(7, zero);
    enc.dispatch({(unsigned)(K / 64), (unsigned)N, 1}, {32, 8, 1});
  }
  if (!st.commit().wait_ok()) { return {}; }
  const auto* sp = static_cast<const std::uint16_t*>(src.contents());
  for (int id : exact) {
    if (id < 0 || id >= N || rows.count(id) != 0) { continue; }
    rows[id].assign(sp + (std::size_t)id * K, sp + (std::size_t)(id + 1) * K);
  }
  *soff = codes;
  *boff = codes + sc;
  return q;
}

void
MetalMossTtsModel::embed_row_(int id, std::uint16_t* out) const
{
  const int H = _cfg.hidden;
  if (!_trunk_w8) {
    const auto* et =
        static_cast<const std::uint16_t*>(_embed_tokens.contents());
    std::memcpy(out, et + (std::size_t)id * H, (std::size_t)H * 2);
    return;
  }
  const auto it = _embed_exact.find(id);
  if (it != _embed_exact.end()) {
    std::memcpy(out, it->second.data(), (std::size_t)H * 2);
    return;
  }
  // Dequantize the row as affine_dequant_w8g64 does: s * q + b in f32,
  // rounded once to bf16.
  const auto* base = static_cast<const std::uint8_t*>(_embed_q->contents());
  const std::uint8_t* q = base + (std::size_t)id * H;
  const int G = H / 64;
  const auto* s = reinterpret_cast<const std::uint16_t*>(base + _e_soff) +
                  (std::size_t)id * G;
  const auto* b = reinterpret_cast<const std::uint16_t*>(base + _e_boff) +
                  (std::size_t)id * G;
  for (int g = 0; g < G; ++g) {
    const float sf = bf16_to_f32_(s[g]), bf = bf16_to_f32_(b[g]);
    for (int j = 0; j < 64; ++j) {
      const int k = g * 64 + j;
      out[k] = f32_to_bf16_(sf * (float)q[k] + bf);
    }
  }
}

SharedBuffer
MetalMossTtsModel::assemble_embeds_(
    const std::vector<std::vector<std::int32_t>>& rows, int start, int n)
{
  const int H = _cfg.hidden, NV = _cfg.n_vq;
  SharedBuffer x = _mc->make_shared_buffer((std::size_t)n * H * 2);
  auto* xp = static_cast<std::uint16_t*>(x.contents());
  // Match the reference's accumulation: inputs_embeds starts as the bf16
  // text embedding, then each audio-code embedding is added one at a time
  // with the result ROUNDED BACK TO BF16 after every add (MLX bf16 add).
  // A single f32 accumulation rounded once at the end diverges by ULPs and
  // flips occasional near-tie argmaxes.
  std::vector<std::uint16_t> cur((std::size_t)H);
  for (int r = 0; r < n; ++r) {
    const auto& row = rows[(std::size_t)(start + r)];
    embed_row_(row[0], cur.data());
    for (int cb = 0; cb < NV; ++cb) {
      const int code = row[(std::size_t)(1 + cb)];
      const auto* arow =
          static_cast<const std::uint16_t*>(_emb_ext[(std::size_t)cb].contents())
          + (std::size_t)code * H;
      for (int h = 0; h < H; ++h) {
        cur[(std::size_t)h] = f32_to_bf16_(
            bf16_to_f32_(cur[(std::size_t)h]) + bf16_to_f32_(arow[h]));
      }
    }
    auto* xr = xp + (std::size_t)r * H;
    for (int h = 0; h < H; ++h) { xr[h] = cur[(std::size_t)h]; }
  }
  return x;
}

void
MetalMossTtsModel::dispatch_heads_(ComputeEncoder& enc, const SharedBuffer& hn,
                                   bool need_full_text)
{
  const int H = _cfg.hidden, NV = _cfg.n_vq;
  const int V = _cfg.vocab, AC = _cfg.audio_vocab + 1;
  auto gemv = [&](const SharedBuffer& w, std::size_t yoff, int N) {
    enc.set_function(_fn_dense_gemv);
    enc.set_buffer(0, hn, 0);
    enc.set_buffer(1, w);
    enc.set_buffer(2, _logits_buf, yoff * 2);
    enc.set_constant(3, H);
    enc.set_constant(4, N);
    enc.dispatch({32, (unsigned)(((N + 7) / 8) * 2), 1}, {32, 2, 1});
  };
  // 32 audio heads always; the full text head only when needed.
  std::size_t off = (std::size_t)V;
  for (int cb = 0; cb < NV; ++cb) {
    gemv(_heads[(std::size_t)(1 + cb)], off, AC);
    off += (std::size_t)AC;
  }
  if (!need_full_text) { return; }
  if (!_trunk_w8) {
    gemv(_heads[0], 0, V);
    return;
  }
  enc.set_function(_fn_qmv8);
  enc.set_buffer(0, *_head0_q, 0);
  enc.set_buffer(1, *_head0_q, _h_soff);
  enc.set_buffer(2, *_head0_q, _h_boff);
  enc.set_buffer(3, hn, 0);
  enc.set_buffer(4, _logits_buf, 0);
  enc.set_constant(5, H);
  enc.set_constant(6, V);
  enc.dispatch({32, (unsigned)(V / 4), 1}, {32, 2, 1});
}

void
MetalMossTtsModel::read_heads_(const SharedBuffer& hn, bool need_full_text,
                               bool need_2slots,
                               std::vector<float>& text_logits,
                               std::vector<std::vector<float>>& audio_logits)
{
  const int H = _cfg.hidden, NV = _cfg.n_vq;
  const int V = _cfg.vocab, AC = _cfg.audio_vocab + 1;
  const auto* lb = static_cast<const std::uint16_t*>(_logits_buf.contents());
  audio_logits.assign((std::size_t)NV, {});
  std::size_t off = (std::size_t)V;
  for (int cb = 0; cb < NV; ++cb) {
    auto& al = audio_logits[(std::size_t)cb];
    al.resize((std::size_t)AC);
    for (int i = 0; i < AC; ++i) { al[(std::size_t)i] = bf16_to_f32_(lb[off + i]); }
    off += (std::size_t)AC;
  }
  if (need_full_text) {
    text_logits.resize((std::size_t)V);
    for (int i = 0; i < V; ++i) { text_logits[(std::size_t)i] = bf16_to_f32_(lb[i]); }
  } else if (need_2slots) {
    // Inside audio the text channel only ranks the gen/delay slot tokens:
    // two cheap host dot products hn . text_head_row[slot] (bf16 UMA reads).
    text_logits.resize((std::size_t)V);
    const auto* h = static_cast<const std::uint16_t*>(hn.contents());
    for (int slot : {_cfg.audio_gen_slot, _cfg.audio_delay_slot}) {
      // trunk_w8 keeps exactly these two rows of the text head in bf16.
      const std::uint16_t* row = nullptr;
      if (_trunk_w8) {
        const auto it = _head_exact.find(slot);
        if (it == _head_exact.end()) { continue; }
        row = it->second.data();
      } else {
        row = static_cast<const std::uint16_t*>(_heads[0].contents()) +
              (std::size_t)slot * H;
      }
      float acc = 0.0f;
      for (int k = 0; k < H; ++k) {
        acc += bf16_to_f32_(h[k]) * bf16_to_f32_(row[k]);
      }
      text_logits[(std::size_t)slot] = acc;
    }
  }
}

void
MetalMossTtsModel::head_logits_(const SharedBuffer& hn, bool need_full_text,
                                bool need_2slots,
                                std::vector<float>& text_logits,
                                std::vector<std::vector<float>>& audio_logits)
{
  metal_compute::CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    dispatch_heads_(enc, hn, need_full_text);
  }
  stream.commit().wait();
  read_heads_(hn, need_full_text, need_2slots, text_logits, audio_logits);
}

std::vector<std::vector<std::int32_t>>
MetalMossTtsModel::generate_delay(
    const std::vector<std::vector<std::int32_t>>& prompt, int max_new_tokens,
    const MossSampling& audio_sp, const MossSampling& text_sp,
    std::uint64_t seed, const RowCb& on_row)
{
  std::vector<std::vector<std::int32_t>> out;
  const int NV = _cfg.n_vq;
  const int seq = (int)prompt.size();
  if (seq <= 0) { return out; }
  ContextManager* cm = _backbone->context_manager();
  const ContextId cid = cm->acquire_root();

  // Sampler state: one RNG (seed 0 => nondeterministic), reused scratch, and
  // optional repetition-penalty history (per-codebook audio + text), allocated
  // only when the respective penalty is enabled.
  std::mt19937_64 rng(seed != 0 ? seed : std::random_device{}());
  std::vector<int> samp_cand;
  std::vector<float> samp_prob;
  const std::vector<int> kNoBlock;
  const bool audio_rep = audio_sp.repetition_penalty != 1.0f;
  const bool text_rep  = text_sp.repetition_penalty != 1.0f;
  std::vector<std::unordered_set<int>> audio_seen(
      audio_rep ? (std::size_t)NV : 0);
  std::unordered_set<int> text_seen;

  // Optional per-phase timing (VPIPE_MOSS_PROFILE).
  const bool kProf = std::getenv("VPIPE_MOSS_PROFILE") != nullptr;
  using Clk = std::chrono::steady_clock;
  auto ms = [](Clk::time_point a, Clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  double t_head = 0, t_fwd = 0, t_asm = 0, t_prefill = 0;
  int prof_steps = 0;

  // Prefill the prompt -> last position's normed hidden. Decode steps then
  // reuse the optimized qmv decode path (cur_hn tracks the live hidden:
  // the prefill buffer first, then the backbone's internal _d_hn scratch).
  const auto p0 = Clk::now();
  SharedBuffer x;
  SharedBuffer prefill_hn;
  {
    // text-prefill perf block (LLM lane). On the FIRST synthesis this also
    // absorbs the cold Metal pipeline-state compilation + first-touch weight
    // residency, so the block is huge on clip 1 and small thereafter.
    PerfAuxScope _pre(_session, kPerfLaneLLM, kGvidLlmPrefill,
                      kPerfLlmPrefillBegin, (std::uint64_t)seq);
    x = assemble_embeds_(prompt, 0, seq);
    prefill_hn = _backbone->forward_embeddings_hidden(cid, x, seq);
  }
  if (kProf) { t_prefill = ms(p0, Clk::now()); }
  if (prefill_hn.empty()) { cm->release(cid); return out; }
  const SharedBuffer* cur_hn = &prefill_hn;

  constexpr std::int64_t kSentinel = std::numeric_limits<std::int64_t>::max();
  constexpr float kNegInf = -std::numeric_limits<float>::infinity();
  bool is_stopping = false;
  bool is_audio = false;
  int audio_lengths = 0;
  std::int64_t delayed_lengths = kSentinel;

  // Continuation bootstrap from the prompt's tail (matches the reference).
  const int last_text = prompt[(std::size_t)(seq - 1)][0];
  const bool is_continuation =
      (last_text == _cfg.audio_start || last_text == _cfg.audio_gen_slot);
  int audio_start_idx = -1;
  for (int i = 0; i < seq; ++i) {
    if (prompt[(std::size_t)i][0] == _cfg.audio_start) { audio_start_idx = i; }
  }
  is_audio = is_continuation && audio_start_idx != -1;
  if (is_audio) { audio_lengths = seq - audio_start_idx; }

  std::vector<float> text_logits;
  std::vector<std::vector<float>> audio_logits;
  // Head needs from the delay state: text samples only when delayed_lengths >
  // n_vq (else it is forced); inside audio it ranks just the 2 slot tokens,
  // outside audio it needs the full vocab head.
  auto need_full = [&] {
    return (!is_stopping && delayed_lengths > NV) && !is_audio;
  };
  auto need_2 = [&] {
    return (!is_stopping && delayed_lengths > NV) && is_audio;
  };
  // Step 0's heads run standalone on the prefill hidden; every later step's
  // heads are FUSED into the decode that produced its hidden (one commit).
  head_logits_(*cur_hn, need_full(), need_2(), text_logits, audio_logits);

  for (int t = 0; t < max_new_tokens; ++t) {
    if (kProf) { ++prof_steps; }
    // One text-decode perf event PER generated token (matches the LM /
    // realtime-vqa per-token decode timeline): brackets this step's token
    // selection + the fused decode forward that produces the next hidden.
    PerfAuxScope _dec(_session, kPerfLaneLLM, kGvidLlmDecode,
                      kPerfLlmDecodeBegin, 1);

    // ---- text channel (delay-pattern state machine) ------------------
    int next_text = _cfg.pad_token;
    if (!is_stopping && delayed_lengths < NV) {
      next_text = _cfg.audio_delay_slot;
    } else if (!is_stopping && delayed_lengths == NV) {
      next_text = _cfg.audio_end;
      is_audio = false;
    } else if (!is_stopping && delayed_lengths > NV) {
      if (is_audio) {
        // keep only {gen_slot, delay_slot}; t==0 also drops delay_slot.
        float bv = kNegInf;
        int best = -1;
        const int cand[2] = {_cfg.audio_gen_slot, _cfg.audio_delay_slot};
        for (int id : cand) {
          if (t == 0 && id == _cfg.audio_delay_slot) { continue; }
          if (text_logits[(std::size_t)id] > bv) {
            bv = text_logits[(std::size_t)id];
            best = id;
          }
        }
        next_text = best;
      } else {
        // Sample the free text token, excluding the control slots ({pad,
        // gen_slot, delay_slot, audio_end} + im_end until past the delay
        // window). temp<=0 reproduces the original argmax (lowest id on ties).
        std::vector<int> blk = {_cfg.pad_token, _cfg.audio_gen_slot,
                                _cfg.audio_delay_slot, _cfg.audio_end};
        if (t <= NV) { blk.push_back(_cfg.im_end); }
        const int best = sample_logits_(
            text_logits.data(), _cfg.vocab, text_sp, blk,
            text_rep ? &text_seen : nullptr, samp_cand, samp_prob, rng);
        next_text = best;
        if (text_rep && best >= 0) { text_seen.insert(best); }
      }
    }

    if (next_text == _cfg.audio_start) { is_audio = true; }
    if (next_text == _cfg.im_end) { is_stopping = true; }

    // ---- audio channels (per-codebook delay activation) --------------
    std::vector<std::int32_t> next_audio((std::size_t)NV, _cfg.audio_pad_code);
    for (int cb = 0; cb < NV; ++cb) {
      const bool pre_audio = audio_lengths > cb;
      const bool post_audio = (delayed_lengths == kSentinel)
                                  ? true
                                  : ((std::int64_t)cb > delayed_lengths - 1);
      if (!(pre_audio && post_audio)) { continue; }
      const auto& cl = audio_logits[(std::size_t)cb];
      const int best = sample_logits_(
          cl.data(), _cfg.audio_vocab, audio_sp, kNoBlock,
          audio_rep ? &audio_seen[(std::size_t)cb] : nullptr,
          samp_cand, samp_prob, rng);
      next_audio[(std::size_t)cb] = (best >= 0) ? best : 0;
      if (audio_rep && best >= 0) { audio_seen[(std::size_t)cb].insert(best); }
    }

    // ---- delay bookkeeping (order matches the reference) -------------
    if (next_text == _cfg.audio_start || next_text == _cfg.audio_gen_slot ||
        next_text == _cfg.audio_delay_slot) {
      ++audio_lengths;
    }
    if (next_text == _cfg.audio_end) { audio_lengths = 0; }
    if (delayed_lengths == kSentinel && next_text == _cfg.audio_delay_slot) {
      delayed_lengths = 0;
    }
    if (delayed_lengths != kSentinel) { ++delayed_lengths; }
    if (delayed_lengths > NV) { delayed_lengths = kSentinel; }

    std::vector<std::int32_t> row;
    row.reserve((std::size_t)(1 + NV));
    row.push_back(next_text);
    for (int cb = 0; cb < NV; ++cb) { row.push_back(next_audio[(std::size_t)cb]); }
    out.push_back(row);

    // The caller sees the row first (a streaming de-delay may complete a
    // frame on it), then may stop -- barge-in: the delayed codes still in
    // flight (the last ~n_vq rows) never complete and the de-delay drops
    // them, so the audio ends a hair early but the new text starts now.
    const bool keep_going = !on_row || on_row(row);
    if (is_stopping || !keep_going) { break; }

    // Feed the new row back through the optimized qmv decode path, FUSING the
    // next step's heads (the state is now t+1) into the same command buffer.
    const bool nf1 = need_full(), n2_1 = need_2();
    const auto a0 = Clk::now();
    std::vector<std::vector<std::int32_t>> one(1, row);
    SharedBuffer xe = assemble_embeds_(one, 0, 1);
    const auto a1 = Clk::now();
    const std::function<void(ComputeEncoder&, const SharedBuffer&)> hook =
        [&](ComputeEncoder& enc, const SharedBuffer& d_hn) {
          dispatch_heads_(enc, d_hn, nf1);
        };
    cur_hn = _backbone->decode_embedding_hidden(cid, xe, -1, &hook);
    if (kProf) { t_fwd += ms(a1, Clk::now()); t_asm += ms(a0, a1); }
    if (cur_hn == nullptr) { break; }
    const auto rh0 = Clk::now();
    read_heads_(*cur_hn, nf1, n2_1, text_logits, audio_logits);
    if (kProf) { t_head += ms(rh0, Clk::now()); }
  }
  if (kProf && prof_steps > 0) {
    std::printf("[moss-prof] %d steps | prefill %.1f ms | per-step: "
                "decode-fwd %.1f ms + heads %.1f ms + embed %.2f ms "
                "(decode total %.1f ms/step)\n",
                prof_steps, t_prefill, t_fwd / prof_steps,
                t_head / prof_steps, t_asm / prof_steps,
                (t_fwd + t_head + t_asm) / prof_steps);
  }
  cm->release(cid);
  return out;
}

std::vector<MetalMossTtsModel::AudioMismatch>
MetalMossTtsModel::teacher_force_audio_mismatches(
    const std::vector<std::vector<std::int32_t>>& prompt,
    const std::vector<std::vector<std::int32_t>>& ref_rows)
{
  std::vector<AudioMismatch> out;
  const int NV = _cfg.n_vq;
  constexpr float kNegInf = -std::numeric_limits<float>::infinity();
  // Verification only inspects the audio heads; skip the text head.
  teacher_force_logits(
      prompt, ref_rows, /*full_text=*/false,
      [&](int r, const std::vector<std::vector<float>>& audio_logits,
          const std::vector<float>&) {
        const auto& ref = ref_rows[(std::size_t)r];
        for (int cb = 0; cb < NV; ++cb) {
          const int ref_code = ref[(std::size_t)(1 + cb)];
          if (ref_code == _cfg.audio_pad_code) { continue; }   // inactive
          const auto& cl = audio_logits[(std::size_t)cb];
          float bv = kNegInf;
          int best = 0;
          for (int c = 0; c < _cfg.audio_vocab; ++c) {
            if (cl[(std::size_t)c] > bv) { bv = cl[(std::size_t)c]; best = c; }
          }
          if (best != ref_code) {
            out.push_back({r, cb, best, ref_code, bv,
                           cl[(std::size_t)ref_code]});
          }
        }
      });
  return out;
}

void
MetalMossTtsModel::teacher_force_logits(
    const std::vector<std::vector<std::int32_t>>& prompt,
    const std::vector<std::vector<std::int32_t>>& rows, bool full_text,
    const LogitsCb& cb)
{
  const int seq = (int)prompt.size();
  if (seq <= 0 || rows.empty()) { return; }
  ContextManager* cm = _backbone->context_manager();
  const ContextId cid = cm->acquire_root();

  SharedBuffer x = assemble_embeds_(prompt, 0, seq);
  SharedBuffer prefill_hn = _backbone->forward_embeddings_hidden(cid, x, seq);
  if (prefill_hn.empty()) { cm->release(cid); return; }
  const SharedBuffer* cur_hn = &prefill_hn;

  std::vector<float> text_logits;
  std::vector<std::vector<float>> audio_logits;
  for (int r = 0; r < (int)rows.size(); ++r) {
    text_logits.clear();
    head_logits_(*cur_hn, full_text, /*need_2slots=*/false, text_logits,
                 audio_logits);
    cb(r, audio_logits, text_logits);
    std::vector<std::vector<std::int32_t>> one(1, rows[(std::size_t)r]);
    SharedBuffer xe = assemble_embeds_(one, 0, 1);
    cur_hn = _backbone->decode_embedding_hidden(cid, xe);
    if (cur_hn == nullptr) { break; }
  }
  cm->release(cid);
}

}  // namespace vpipe::genai

#include "generative-models/yue2/metal-yue2-model.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/kernel-contract.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "common/perf-event.h"
#include "common/perf-scope.h"
#include "common/vpipe-format.h"
#include "generative-models/context-manager.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/qwen3/metal-qwen-model.h"
#include "generative-models/shared/ane-ffn.h"
#include "generative-models/shared/i8-gemm.h"
#include "generative-models/shared/metal-sage-attention.h"
#include "generative-models/shared/metal-sol-attention.h"
#include "generative-models/shared/mma-tile.h"
#include "generative-models/weight-set.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace vpipe::genai {

using metal_compute::CommandStream;
using metal_compute::ComputeEncoder;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;
namespace contract = metal_compute::contract;

namespace {

inline float
bf16_bits_to_f32_(std::uint16_t b)
{
  const std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

inline std::uint16_t
f32_to_bf16_bits_(float f)
{
  const float r = yue2::bf16_round(f);
  std::uint32_t u;
  std::memcpy(&u, &r, 4);
  return (std::uint16_t)(u >> 16);
}

// fmt() defers its formatting; an error string is wanted now.
template <typename... A>
std::string
sfmt_(std::string_view f, A&&... a)
{
  return std::vformat(f, std::make_format_args(a...));
}

// The steel attention tiles. The matrix-core entry tiles twice as wide.
constexpr int kBqSteel = 32, kBkSteel = 16;
constexpr int kBqNax   = 64, kBkNax   = 32;

// matmul2d's 128-row tile only pays once M amortizes it.
constexpr int kMmaMinRows = 64;

}  // namespace

// ---- config -------------------------------------------------------------

bool
MetalYue2Model::config_from_dir(const std::string& dir, Config* out,
                                std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  std::ifstream in(std::filesystem::path(dir) / "config.json");
  if (!in) { return fail("no config.json under " + dir); }
  FlexData fd;
  try {
    fd = FlexData::from_json(in);
  } catch (const std::exception& e) {
    return fail(std::string("config.json: ") + e.what());
  }
  if (!fd.is_object()) { return fail("config.json is not an object"); }
  auto o = fd.as_object();
  if (!o.contains("model_type") ||
      o.at("model_type").as_string() != "yue2") {
    return fail("not a YuE2 checkpoint (model_type != \"yue2\")");
  }
  if (o.contains("latent_type") && o.at("latent_type").as_string() != "vae") {
    return fail("YuE2 inference supports only latent_type \"vae\"");
  }
  Config c;
  auto geti = [&](const char* k, int d) {
    return o.contains(k) ? (int)o.at(k).as_int(d) : d;
  };
  auto getf = [&](const char* k, float d) {
    return o.contains(k) ? (float)o.at(k).as_real(d) : d;
  };
  c.n_layers   = geti("num_hidden_layers", c.n_layers);
  c.hidden     = geti("hidden_size", c.hidden);
  c.n_heads    = geti("num_attention_heads", c.n_heads);
  c.n_kv_heads = geti("num_key_value_heads", c.n_kv_heads);
  c.head_dim   = geti("head_dim", c.head_dim);
  c.ffn        = geti("intermediate_size", c.ffn);
  c.vocab      = geti("vocab_size", c.vocab);
  c.rope_theta = getf("rope_theta", c.rope_theta);
  c.rms_eps    = getf("rms_norm_eps", c.rms_eps);
  c.latent_dim = geti("latent_dim", c.latent_dim);
  c.max_latent_frames = geti("max_latent_frames", c.max_latent_frames);
  c.timestep_shift    = getf("timestep_shift", c.timestep_shift);
  c.context    = geti("max_position_embeddings", c.context);
  // The protocol's ids are fixed; a vocabulary that does not hold them is
  // a different model, whatever its config says.
  if (c.vocab != yue2::kVocab || c.latent_dim != yue2::kLatentDim) {
    return fail(sfmt_("unexpected YuE2 geometry: vocab {} latent_dim {}",
                    c.vocab, c.latent_dim));
  }
  if (c.head_dim != 128 || c.n_heads % c.n_kv_heads != 0) {
    return fail("YuE2 attention needs head_dim 128 and grouped heads");
  }
  *out = c;
  return true;
}

bool
MetalYue2Model::is_yue2_dir(const std::string& dir)
{
  Config c;
  return config_from_dir(dir, &c, nullptr);
}

// ---- load ---------------------------------------------------------------

MetalYue2Model::~MetalYue2Model() = default;

std::unique_ptr<MetalYue2Model>
MetalYue2Model::load(const std::string& dir, MetalCompute* mc,
                     std::string* err)
{
  Config cfg;
  if (!config_from_dir(dir, &cfg, err)) { return nullptr; }
  // No session to ask, so this opens a PRIVATE set.
  return load(WeightSet::open(dir, nullptr), mc, cfg, err);
}

std::unique_ptr<MetalYue2Model>
MetalYue2Model::load(std::shared_ptr<WeightSet> ws, MetalCompute* mc,
                     const Config& cfg, std::string* err)
{
  if (ws == nullptr || mc == nullptr) {
    if (err != nullptr) { *err = "no checkpoint or no metal device"; }
    return nullptr;
  }
  std::unique_ptr<MetalYue2Model> m(new MetalYue2Model());
  if (!m->init_(ws, mc, cfg, err)) { return nullptr; }
  return m;
}

bool
MetalYue2Model::init_(const std::shared_ptr<WeightSet>& ws, MetalCompute* mc,
                      const Config& cfg, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  _ws = ws;
  _mc = mc;
  _cfg = cfg;
  WeightSet& wts = *ws;

  // ---- the AR backbone: a plain Qwen3 dense stack under these names --
  MetalQwenModel::Config q;
  q.n_layers   = cfg.n_layers;
  q.hidden     = cfg.hidden;
  q.n_heads    = cfg.n_heads;
  q.n_kv_heads = cfg.n_kv_heads;
  q.head_dim   = cfg.head_dim;
  q.ffn_inner  = cfg.ffn;
  q.vocab      = cfg.vocab;
  q.rope_theta = cfg.rope_theta;
  q.rms_eps    = cfg.rms_eps;
  q.rotary_dim = cfg.head_dim;            // full rotary, no partial factor
  q.tie_embeddings     = false;
  q.use_bf16           = true;
  q.dense              = true;
  q.zero_centered_norm = false;           // plain Qwen3 RMSNorm
  // The affine kernels are chosen for this width whether or not the stack
  // is quantized; a w8 build needs them to be the w8 ones.
  q.quant_bits         = 8;
  q.load_quant_bits    = cfg.ar_quant_bits == 8 ? 8 : 0;
  q.attn_output_gate   = false;
  q.qk_norm            = true;
  q.backbone_only      = true;            // embed + head are ours
  q.weight_prefix      = "";
  q.model_seg          = "model.";
  // One context holds a prefix plus a whole phase; the NAR's prefill of a
  // chunk is shorter than the context by construction.
  q.max_seq            = cfg.context + 256;
  _lm = MetalQwenModel::load(ws, mc, q);
  if (!_lm) { return fail("the YuE2 AR backbone did not load"); }

  // ---- the AR head and embedding: cached, kept as the checkpoint has
  // them (bf16), sliced per phase at dispatch --------------------------
  _embed = wts.tensor("model.embed_tokens.weight", mc,
                      WeightSet::Residency::Copied);
  _head = wts.tensor("lm_head.weight", mc, WeightSet::Residency::Copied);
  if (_embed.empty() || _head.empty()) {
    return fail("model.embed_tokens / lm_head missing");
  }
  const std::size_t H = (std::size_t)cfg.hidden;
  _logits = mc->make_shared_buffer((std::size_t)cfg.vocab * 2);
  _d_emb = mc->make_shared_buffer(H * 2);

  // ---- the NAR weights --------------------------------------------------
  auto t = [&](const std::string& nm) {
    return wts.tensor(nm, mc, WeightSet::Residency::Copied);
  };
  _nar.resize((std::size_t)cfg.n_layers);
  for (int L = 0; L < cfg.n_layers; ++L) {
    const std::string p = "model.layers." + std::to_string(L) + ".";
    NarLayer& n = _nar[(std::size_t)L];
    n.in_ln  = t(p + "nar_input_layernorm.weight");
    n.mlp_ln = t(p + "nar_pre_mlp_layernorm.weight");
    n.q      = t(p + "nar_self_attn.q_proj.weight");
    n.k      = t(p + "nar_self_attn.k_proj.weight");
    n.v      = t(p + "nar_self_attn.v_proj.weight");
    n.o      = t(p + "nar_self_attn.o_proj.weight");
    n.q_norm = t(p + "nar_self_attn.q_norm.weight");
    n.k_norm = t(p + "nar_self_attn.k_norm.weight");
    n.gate   = t(p + "nar_mlp.gate_proj.weight");
    n.up     = t(p + "nar_mlp.up_proj.weight");
    n.down   = t(p + "nar_mlp.down_proj.weight");
    if (n.in_ln.empty() || n.mlp_ln.empty() || n.q.empty() || n.k.empty() ||
        n.v.empty() || n.o.empty() || n.q_norm.empty() || n.k_norm.empty() ||
        n.gate.empty() || n.up.empty() || n.down.empty()) {
      return fail(sfmt_("NAR layer {} is incomplete", L));
    }
  }
  _final_ln  = t("model.norm.weight");
  _vae2llm_w = t("vae2llm.weight");
  _vae2llm_b = t("vae2llm.bias");
  _llm2vae_w = t("llm2vae.weight");
  _llm2vae_b = t("llm2vae.bias");
  _pos       = t("latent_pos_embed.pe");
  if (_final_ln.empty() || _vae2llm_w.empty() || _vae2llm_b.empty() ||
      _llm2vae_w.empty() || _llm2vae_b.empty() || _pos.empty()) {
    return fail("the NAR's model-level tensors are incomplete");
  }

  // The timestep embedder runs on the host (one row per evaluation);
  // widen its bf16 weights once. Read uncached: consumed here.
  auto widen = [&](const std::string& nm, std::vector<float>* dst) {
    SharedBuffer b = wts.read(nm, mc, WeightSet::Residency::Copied);
    if (b.empty()) { return false; }
    const std::size_t n = b.byte_size() / 2;
    const auto* s = static_cast<const std::uint16_t*>(b.contents());
    dst->resize(n);
    for (std::size_t i = 0; i < n; ++i) { (*dst)[i] = bf16_bits_to_f32_(s[i]); }
    return true;
  };
  if (!widen("time_embedder.mlp.0.weight", &_te0_w) ||
      !widen("time_embedder.mlp.0.bias", &_te0_b) ||
      !widen("time_embedder.mlp.2.weight", &_te2_w) ||
      !widen("time_embedder.mlp.2.bias", &_te2_b)) {
    return fail("time_embedder weights missing");
  }
  if (_te0_b.size() != H || _te2_b.size() != H ||
      _te0_w.size() != H * 256 || _te2_w.size() != H * H) {
    return fail("time_embedder has an unexpected shape");
  }

  // ---- kernels ----------------------------------------------------------
  _lib_gemm = mc->load_library("dense_gemm_bf16");
  _lib_elt  = mc->load_library("llm_elementwise_bf16");
  _lib_rms  = mc->load_library("rms_norm_bf16");
  _lib_rope = mc->load_library("rope_bf16");
  _lib_attn = mc->load_library("attn_steel");
  _fn_gemv        = _lib_gemm.function("dense_gemv_t_f16");
  _fn_gemm        = _lib_gemm.function("dense_gemm_t_f16");
  _fn_gemm_bm64   = _lib_gemm.function("dense_gemm_t_bm64_f16");
  _fn_gemm_bm64bn64 = _lib_gemm.function("dense_gemm_t_bm64bn64_f16");
  _fn_rms         = _lib_rms.function("rms_norm_f16");
  _fn_rms_heads   = _lib_rope.function("rms_norm_heads_strided_f16");
  _fn_tr_rope     = _lib_rope.function("transpose_rope_half_part_ftab_f16");
  _fn_kv_gather   = _lib_elt.function("kv_gather_paged_f16");
  _fn_add         = _lib_elt.function("residual_add_f16");
  _fn_add4        = _lib_elt.function("residual_add_v4_f16");
  _fn_bias_rows   = _lib_elt.function("bias_add_rows_f16");
  _fn_swiglu      = _lib_elt.function("swiglu_f16");
  _fn_swiglu4     = _lib_elt.function("swiglu_v4_f16");
  _fn_transpose   = _lib_elt.function("transpose_abd_f16");
  if (!_fn_gemv.valid() || !_fn_gemm.valid() || !_fn_gemm_bm64.valid() ||
      !_fn_gemm_bm64bn64.valid() || !_fn_rms.valid() ||
      !_fn_rms_heads.valid() || !_fn_tr_rope.valid() || !_fn_add.valid() ||
      !_fn_bias_rows.valid() || !_fn_swiglu.valid() ||
      !_fn_transpose.valid() || !_fn_kv_gather.valid()) {
    return fail("a YuE2 kernel did not resolve");
  }
  // M5 matrix cores: matmul2d GEMMs and the NAX flash attention. Gated on
  // the hardware -- matmul2d EMULATES without matrix cores.
  if (mc->supports_matrix_cores()) {
    if (std::getenv("VPIPE_YUE2_NO_MMA2") == nullptr) {
      _lib_mma = mc->load_library("dense_gemm_mma_bf16");
      _fn_mma = _lib_mma.function("dense_gemm_mma_t_n128_f16");
      _fn_mma_deep = _lib_mma.function("dense_gemm_mma_t_n128x256_f16");
      _use_mma = _fn_mma.valid() && _fn_mma_deep.valid();
    }
    if (std::getenv("VPIPE_YUE2_NO_NAX_ATTN") == nullptr) {
      _lib_attn_nax = mc->load_library("attn_steel_nax");
      _use_nax = _lib_attn_nax.valid();
    }
  }
  if (_logits.empty() || _d_emb.empty()) {
    return fail("YuE2 scratch allocation failed");
  }
  return true;
}

std::size_t
MetalYue2Model::resident_bytes() const
{
  std::size_t b = _embed.byte_size() + _head.byte_size() +
                  _final_ln.byte_size() + _vae2llm_w.byte_size() +
                  _vae2llm_b.byte_size() + _llm2vae_w.byte_size() +
                  _llm2vae_b.byte_size() + _pos.byte_size();
  for (const NarLayer& n : _nar) {
    b += n.in_ln.byte_size() + n.mlp_ln.byte_size() + n.q.byte_size() +
         n.k.byte_size() + n.v.byte_size() + n.o.byte_size() +
         n.q_norm.byte_size() + n.k_norm.byte_size() + n.gate.byte_size() +
         n.up.byte_size() + n.down.byte_size();
  }
  // The AR stack as the backbone holds it, not as the config implies.
  if (_lm) { b += (std::size_t)_lm->resident_bytes(); }
  return b;
}

bool
MetalYue2Model::ar_quantized() const
{
  return _lm != nullptr && _lm->quantized_on_load();
}

std::size_t
MetalYue2Model::weight_bytes(const std::string& dir, int ar_quant_bits)
{
  auto wts = MetalLlamaWeights::open_model(dir);
  if (!wts.has_value()) { return 0; }
  std::size_t disk = 0;
  for (const std::string& nm : wts->tensor_names()) {
    if (const auto* ti = wts->info(nm)) { disk += (std::size_t)ti->nbytes; }
  }
  if (ar_quant_bits != 8) { return disk; }
  Config c;
  if (!config_from_dir(dir, &c)) { return disk; }
  // The AR stack is the plain Qwen3 names under model.layers. (the NAR's
  // are nar_*, which this does not count); built, it is w8.
  const auto rs = MetalQwenModel::raw_sizes(*wts, "model.layers.",
                                            c.n_layers, 8);
  if (rs.n_layers != c.n_layers || rs.pack || rs.raw > disk) { return disk; }
  return disk - rs.raw + rs.built;
}

std::size_t
MetalYue2Model::nar_row_bytes_(const Config& c)
{
  // ensure_scratch_: the latent in and out, x / h / o at the hidden
  // width, q and the head-major query and attention outputs at q width,
  // k / v at kv width, gate and up at the feed-forward width. bf16.
  const std::size_t H = (std::size_t)c.hidden;
  const std::size_t qd = (std::size_t)c.n_heads * c.head_dim;
  const std::size_t kvd = (std::size_t)c.n_kv_heads * c.head_dim;
  const std::size_t F = (std::size_t)c.ffn;
  const std::size_t Z = (std::size_t)c.latent_dim;
  return (2 * Z + 3 * H + 4 * qd + 2 * kvd + 2 * F) * 2;
}

MetalYue2Model::RunBytes
MetalYue2Model::run_bytes(const Config& c, int prefix, int frames,
                          bool guided)
{
  RunBytes r;
  if (frames <= 0) { return r; }
  const std::size_t kv_tok = (std::size_t)c.n_layers * 2 *
                             (std::size_t)c.n_kv_heads * c.head_dim * 2;
  // The AR pool: MetalQwenModel's ContextManager, pages of kPage tokens,
  // capacity grown by doubling from one page to cover the highest page
  // id in use -- every page the song phase holds at once, both contexts
  // when guided. Its cap (max_seq / page * 16) is far above any song.
  constexpr int kPage = 256;   // MetalQwenModel::Config::page_tokens
  const int ctx = std::min(c.context, prefix + frames);
  const int per = (ctx + kPage - 1) / kPage + 1;   // + the page in progress
  const int pages = guided ? 2 * per : per;
  int cap = 1, prev = 0;
  while (cap < pages) { prev = cap; cap *= 2; }
  const std::size_t page_bytes = (std::size_t)kPage * kv_tok;
  r.held = (std::size_t)cap * page_bytes +
           (std::size_t)(frames + 2) * nar_row_bytes_(c);
  // The chunk: its AR tokens (prefix, codes, MUSIC_END) and its NAR rows
  // (START, latents, END), head-major, every layer.
  const std::size_t chunk = (std::size_t)(ctx + 1 + frames + 2) * kv_tok;
  r.transient = std::max(chunk, (std::size_t)prev * page_bytes);
  return r;
}

std::size_t
MetalYue2Model::held_scratch_bytes() const
{
  std::size_t b = 0;
  if (_lm != nullptr) {
    if (ContextManager* cm = _lm->context_manager()) {
      b += cm->resident_bytes();
    }
  }
  for (const SharedBuffer* p : {&_n_in, &_n_x, &_n_h, &_n_q, &_n_k, &_n_vv,
                                &_n_qh, &_n_aoh, &_n_ao, &_n_o, &_n_g, &_n_u,
                                &_n_v, &_n_t}) {
    b += p->byte_size();
  }
  return b;
}

// ---- AR -----------------------------------------------------------------

SharedBuffer
MetalYue2Model::embed_(const std::vector<std::int32_t>& ids, std::size_t from,
                      std::size_t n)
{
  const std::size_t H = (std::size_t)_cfg.hidden;
  SharedBuffer x = _mc->make_shared_buffer(n * H * 2);
  if (x.empty()) { return x; }
  auto* d = static_cast<std::uint8_t*>(x.contents());
  const auto* e = static_cast<const std::uint8_t*>(_embed.contents());
  for (std::size_t i = 0; i < n; ++i) {
    const std::int32_t id = ids[from + i];
    std::memcpy(d + i * H * 2, e + (std::size_t)id * H * 2, H * 2);
  }
  return x;
}

void
MetalYue2Model::encode_head_(ComputeEncoder& enc, const SharedBuffer& h,
                             const yue2::Window& w)
{
  // y[rows] = head[row0 : row0 + rows] @ h. The GEMV reads whole 8-row
  // groups, so it may touch up to 7 rows past the window -- inside the
  // head for both phase windows, and their writes are masked.
  const int H = _cfg.hidden;
  enc.set_function(_fn_gemv);
  enc.set_buffer(0, h);
  enc.set_buffer(1, _head, (std::size_t)w.row0 * (std::size_t)H * 2);
  enc.set_buffer(2, _logits);
  enc.set_constant(3, H);
  enc.set_constant(4, w.rows);
  enc.dispatch({32, (unsigned)(((w.rows + 7) / 8) * 2), 1}, {32, 2, 1});
}

void
MetalYue2Model::read_logits_(const yue2::Window& w,
                             std::vector<float>* out) const
{
  out->resize((std::size_t)w.rows);
  const auto* s = static_cast<const std::uint16_t*>(_logits.contents());
  for (int i = 0; i < w.rows; ++i) {
    (*out)[(std::size_t)i] = bf16_bits_to_f32_(s[i]);
  }
}

bool
MetalYue2Model::prefill_logits(const std::vector<std::int32_t>& prefix,
                               std::vector<float>* logits, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (prefix.empty()) { return fail("empty prefix"); }
  ContextManager* cm = _lm->context_manager();
  const ContextId c = cm->acquire_root();
  SharedBuffer x = embed_(prefix, 0, prefix.size());
  SharedBuffer h = _lm->forward_embeddings_hidden(c, x, (int)prefix.size());
  cm->release(c);
  if (h.empty()) { return fail("prefill failed"); }
  const yue2::Window all{0, _cfg.vocab, 0};
  CommandStream s = _mc->make_command_stream();
  {
    ComputeEncoder enc = s.begin_compute();
    encode_head_(enc, h, all);
  }
  std::string gerr;
  if (!s.commit().wait_ok(&gerr)) { return fail(gerr); }
  read_logits_(all, logits);
  return true;
}

bool
MetalYue2Model::decode(const std::vector<std::int32_t>& prefix,
                       yue2::Phase phase, const yue2::Sampling& sp,
                       std::uint64_t seed,
                       const std::vector<std::int32_t>* negative, float cfg,
                       bool legacy, const TokenFn& on_token,
                       DecodeResult* out, std::string* err)
{
  return run_decode_(prefix, phase, sp, seed, negative, cfg, legacy, on_token,
                     nullptr, nullptr, out, err);
}

bool
MetalYue2Model::decode_forced(const std::vector<std::int32_t>& prefix,
                              yue2::Phase phase, const yue2::Sampling& sp,
                              const std::vector<std::int32_t>& forced,
                              std::vector<std::int32_t>* picks,
                              std::string* err)
{
  DecodeResult out;
  picks->clear();
  return run_decode_(prefix, phase, sp, 1, nullptr, 1.0f, false, {}, &forced,
                     picks, &out, err);
}

bool
MetalYue2Model::run_decode_(const std::vector<std::int32_t>& prefix,
                            yue2::Phase phase, const yue2::Sampling& sp,
                            std::uint64_t seed,
                            const std::vector<std::int32_t>* negative,
                            float cfg, bool legacy, const TokenFn& on_token,
                            const std::vector<std::int32_t>* forced,
                            std::vector<std::int32_t>* picks,
                            DecodeResult* out, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  using clk = std::chrono::steady_clock;
  const auto t0 = clk::now();
  *out = DecodeResult{};
  std::string serr;
  if (!sp.valid(&serr)) { return fail("sampling: " + serr); }
  if (prefix.empty()) { return fail("empty prefix"); }
  const bool guided = cfg != 1.0f;
  if (guided && (negative == nullptr || negative->empty())) {
    return fail("CFG needs a negative prefix");
  }
  const std::size_t budget = (std::size_t)sp.max_tokens;
  if (prefix.size() + budget > (std::size_t)_cfg.context ||
      (guided && negative->size() + budget > (std::size_t)_cfg.context)) {
    return fail(sfmt_("prefix {} + budget {} exceeds the {}-token context",
                    prefix.size(), budget, _cfg.context));
  }

  const SessionContextIntf* session = _mc->session();
  yue2::TokenPicker picker(phase, sp, seed, legacy);
  const yue2::Window w = picker.window();
  ContextManager* cm = _lm->context_manager();
  const ContextId cc = cm->acquire_root();
  const ContextId uc = guided ? cm->acquire_root() : ContextId{};
  struct Release {
    ContextManager* cm; ContextId a, b;
    ~Release() {
      cm->release(a);
      if (b.valid()) { cm->release(b); }
    }
  } rel{cm, cc, uc};

  std::vector<float> cond, uncond, mixed;
  // Prefill one branch and leave its window's logits in `dst`.
  auto prefill = [&](ContextId c, const std::vector<std::int32_t>& ids,
                     std::vector<float>* dst) -> bool {
    PerfAuxScope _perf(session, kPerfLaneLLM, kGvidLlmPrefill,
                       kPerfLlmPrefillBegin, (std::uint64_t)ids.size());
    SharedBuffer x = embed_(ids, 0, ids.size());
    SharedBuffer h = _lm->forward_embeddings_hidden(c, x, (int)ids.size());
    if (h.empty()) { return false; }
    CommandStream s = _mc->make_command_stream();
    {
      ComputeEncoder enc = s.begin_compute();
      encode_head_(enc, h, w);
    }
    if (!s.commit().wait_ok()) { return false; }
    read_logits_(w, dst);
    return true;
  };
  // One decode step of one branch, the head fused into its buffer.
  const std::function<void(ComputeEncoder&, const SharedBuffer&)> head =
      [&](ComputeEncoder& enc, const SharedBuffer& h) {
        encode_head_(enc, h, w);
      };
  auto step = [&](ContextId c, std::int32_t id,
                  std::vector<float>* dst) -> bool {
    const std::size_t H = (std::size_t)_cfg.hidden;
    std::memcpy(_d_emb.contents(),
                static_cast<const std::uint8_t*>(_embed.contents()) +
                    (std::size_t)id * H * 2,
                H * 2);
    const SharedBuffer* hn = _lm->decode_embedding_hidden(c, _d_emb, -1,
                                                          &head);
    if (hn == nullptr) { return false; }
    read_logits_(w, dst);
    return true;
  };

  if (!prefill(cc, prefix, &cond)) { return fail("prefill failed"); }
  if (guided && !prefill(uc, *negative, &uncond)) {
    return fail("negative prefill failed");
  }
  out->prefill_seconds =
      std::chrono::duration<double>(clk::now() - t0).count();

  bool eos = false;
  const int steps = forced != nullptr
                        ? std::min(sp.max_tokens, (int)forced->size())
                        : sp.max_tokens;
  for (int k = 0; k < steps; ++k) {
    const float* lg = cond.data();
    if (guided) {
      mixed.resize(cond.size());
      yue2::cfg_combine(cond.data(), uncond.data(), cfg, mixed.data(),
                        (int)cond.size());
      lg = mixed.data();
    }
    std::int32_t id = picker.pick(lg, k);
    if (forced != nullptr) {
      picks->push_back(id);
      id = (*forced)[(std::size_t)k];
    }
    if (on_token && !on_token(id, k)) {
      out->stopped = true;
      break;
    }
    if (id == w.end) {
      eos = true;
      break;
    }
    picker.accept(id);
    out->ids.push_back(id);
    if (k + 1 >= steps) { break; }
    PerfAuxScope _perf(session, kPerfLaneLLM, kGvidLlmDecode,
                       kPerfLlmDecodeBegin, 1);
    if (!step(cc, id, &cond)) { return fail("decode step failed"); }
    if (guided && !step(uc, id, &uncond)) {
      return fail("negative decode step failed");
    }
  }
  out->truncated = !eos && !out->stopped;
  out->seconds = std::chrono::duration<double>(clk::now() - t0).count();
  return true;
}

// ---- NAR ----------------------------------------------------------------

// ---- the acceleration tiers ----------------------------------------------

std::size_t
MetalYue2Model::ane_runtime_bytes()
{
  // The module is one block's feed-forward, sized by the chunk; the song's
  // length does not move it.
  const Config c;
  return AneFeedForward::runtime_bytes(
      c.hidden, c.ffn, 0, AneFeedForward::chunk_rows("VPIPE_YUE2_ANE_CHUNK"));
}

bool
MetalYue2Model::set_accel(const Accel& a, std::string* err)
{
  const SessionContextIntf* ss = _mc->session();
  _accel = a;
  _ane.reset();
  _ane_tried_seq = 0;
  _sage.reset();
  _sol.reset();
  _i8.reset();
  _sol_warned = false;
  if (a.i8_gemm) {
    // Declines itself without matrix cores, and VPIPE_I8_GEMM overrides.
    auto i8 = std::make_unique<I8GemmContext>(_mc, true, /*bf16=*/true);
    if (i8->enabled()) {
      _i8 = std::move(i8);
    } else if (ss != nullptr) {
      ss->log_normal(fmt("yue2: i8_gemm unavailable here -- dense GEMMs"));
    }
  }
  if (a.sol.enabled) {
    std::string serr;
    _sol = MetalSolAttention::load(_mc, /*bf16=*/true, &serr);
    if (_sol == nullptr && ss != nullptr) {
      ss->log_normal(fmt("yue2: sol_attn unavailable -- {}",
                         serr.empty() ? "kernels did not load" : serr));
    }
  }
  bool fatal = false;
  _sage = MetalSageAttention::load_for_model(_mc, /*bf16=*/true, a.sage,
                                             "MetalYue2Model", &fatal);
  if (fatal) {
    if (err != nullptr) { *err = "sage_attn's kernels did not load"; }
    return false;
  }
  if (_sage && !_use_nax) {
    // The int8 QK lives in the matrix-core flash entry only.
    if (ss != nullptr) {
      ss->log_normal(fmt("yue2: sage_attn is off -- the matrix-core flash "
                         "entry is unavailable here"));
    }
    _sage.reset();
  }
  if (ss != nullptr) {
    ss->info(fmt("yue2: flow-matching tiers -- ane_ffn {}, i8_gemm {}, "
                 "sage_attn {}, sol_attn {}{}",
                 a.ane_ffn ? "requested" : "off", _i8 ? "on" : "off",
                 _sage ? "on" : "off",
                 _sol ? "on" : "off",
                 _sol ? sfmt_(" (tau {}, {} dense layer(s), {})", a.sol.tau,
                              a.sol.dense_layers,
                              _sol->uses_matrix_cores() ? "NAX" : "ALU")
                      : std::string()));
  }
  return true;
}

bool
MetalYue2Model::ane_setup_(int seq)
{
  if (_ane != nullptr) { return true; }
  // The module is sized at its first forward. A song too short for one
  // chunk declines, and a LONGER one later is asked again -- the share is
  // a property of the geometry, and a stage serves songs of any length.
  if (seq <= 2 * _ane_tried_seq || _mc->session() == nullptr) {
    return false;
  }
  _ane_tried_seq = seq;
  AneFeedForward::Options o;
  o.session = _mc->session();
  o.mc      = _mc;
  o.tag     = "yue2";
  o.hidden  = _cfg.hidden;
  o.ffn     = _cfg.ffn;
  o.seq     = seq;
  o.rows    = _accel.ane_rows;
  o.chunk   = AneFeedForward::chunk_rows("VPIPE_YUE2_ANE_CHUNK");
  o.profile = std::getenv("VPIPE_YUE2_ANE_PROFILE") != nullptr;
  _ane = AneFeedForward::create(o);
  return _ane != nullptr;
}

bool
MetalYue2Model::ane_eligible_(int L) const
{
  if (_ane == nullptr) { return false; }
  const int cap = _accel.ane_layers > 0
                      ? std::min(_accel.ane_layers, _cfg.n_layers)
                      : _cfg.n_layers;
  return L < cap;
}

void
MetalYue2Model::ane_stage_(int L)
{
  if (_ane == nullptr) { return; }
  // Dense bf16 gate, up and down, each its own [out, in] matrix.
  auto src = [](const SharedBuffer& w) {
    AneFfnSource s;
    s.w = &w;
    return s;
  };
  const NarLayer& n = _nar[(std::size_t)L];
  _ane->stage(L, src(n.gate), src(n.up), src(n.down), 64);
}

bool
MetalYue2Model::ensure_scratch_(int rows)
{
  if (rows <= _scratch_rows) { return true; }
  const std::size_t R = (std::size_t)rows;
  const std::size_t H = (std::size_t)_cfg.hidden;
  const std::size_t qd = (std::size_t)_cfg.n_heads * _cfg.head_dim;
  const std::size_t F = (std::size_t)_cfg.ffn;
  const std::size_t Z = (std::size_t)_cfg.latent_dim;
  _n_in = _mc->make_shared_buffer(R * Z * 2);
  _n_x  = _mc->make_shared_buffer(R * H * 2);
  _n_h  = _mc->make_shared_buffer(R * H * 2);
  const std::size_t kvd = (std::size_t)_cfg.n_kv_heads * _cfg.head_dim;
  _n_q  = _mc->make_shared_buffer(R * qd * 2);
  _n_k  = _mc->make_shared_buffer(R * kvd * 2);
  _n_vv = _mc->make_shared_buffer(R * kvd * 2);
  _n_qh = _mc->make_shared_buffer(R * qd * 2);
  _n_aoh = _mc->make_shared_buffer(R * qd * 2);
  _n_ao = _mc->make_shared_buffer(R * qd * 2);
  _n_o  = _mc->make_shared_buffer(R * H * 2);
  _n_g  = _mc->make_shared_buffer(R * F * 2);
  _n_u  = _mc->make_shared_buffer(R * F * 2);
  _n_v  = _mc->make_shared_buffer(R * Z * 2);
  if (_n_t.empty()) { _n_t = _mc->make_shared_buffer(H * 2); }
  if (_n_in.empty() || _n_x.empty() || _n_h.empty() || _n_q.empty() ||
      _n_k.empty() || _n_vv.empty() || _n_qh.empty() || _n_aoh.empty() ||
      _n_ao.empty() || _n_o.empty() || _n_g.empty() || _n_u.empty() ||
      _n_v.empty() || _n_t.empty()) {
    _scratch_rows = 0;
    return false;
  }
  _scratch_rows = rows;
  return true;
}

void
MetalYue2Model::gemm_(ComputeEncoder& enc, const SharedBuffer& x,
                      std::size_t xe, const SharedBuffer& w,
                      const SharedBuffer* bias, const SharedBuffer& y,
                      std::size_t ye, int M, int N, int K)
{
  // The int8 route first, from the same dense operand; a shape it declines
  // falls through with nothing encoded.
  if (_i8 && _i8->gemm(enc, x, xe, w, y, ye, M, N, K)) {
    ++_stats.i8_gemms;
    if (bias != nullptr) {
      const std::uint32_t total = (std::uint32_t)((std::size_t)M * N);
      enc.set_function(_fn_bias_rows);
      enc.set_buffer(0, y, ye * 2);
      enc.set_buffer(1, *bias);
      enc.set_constant(2, N);
      enc.set_constant(3, total);
      enc.dispatch({total, 1, 1}, {256, 1, 1});
    }
    return;
  }
  if (_use_mma && M >= kMmaMinRows && N >= 16 && M <= mma_row_band(N, K)) {
    const bool wide = mma_use_wide_tile(N, K);
    const int RN = wide ? 256 : 128;
    enc.set_function(wide ? _fn_mma_deep : _fn_mma);
    enc.set_buffer(0, x, xe * 2);
    // Bound twice: the biasless form takes the weight in the bias slot.
    enc.set_buffer(1, w);
    enc.set_buffer(2, w);
    enc.set_buffer(3, y, ye * 2);
    enc.set_constant(4, K);
    enc.set_constant(5, N);
    enc.set_constant(6, M);
    enc.set_constant(7, 0);
    enc.dispatch({(unsigned)(((N + RN - 1) / RN) * 256),
                  (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
    if (bias != nullptr) {
      const std::uint32_t total = (std::uint32_t)((std::size_t)M * N);
      enc.set_function(_fn_bias_rows);
      enc.set_buffer(0, y, ye * 2);
      enc.set_buffer(1, *bias);
      enc.set_constant(2, N);
      enc.set_constant(3, total);
      enc.dispatch({total, 1, 1}, {256, 1, 1});
    }
    return;
  }
  // No matrix cores: the widest steel tile BOTH extents fill. The BM64
  // tiles only pay once the rows amortize them, and BN64 wants an output
  // wide enough to fill it -- a narrower one wastes half of every tile,
  // which is what BM64xBN32 is for.
  //
  // THE CHOICE IS TOTAL ON PURPOSE. Every projection of this checkpoint
  // is at least 64 wide (latent_dim 64, kv 1024, hidden 2048, ffn 6144),
  // so it is the BN64 arm that runs here and the middle arm serves a
  // narrower latent than this one ships. Written as three arms anyway,
  // because the alternative is what this replaced: BM64xBN32 loaded and
  // required to resolve at load while no shape could ever reach it, so
  // the model would have refused to load over a kernel it never used.
  int bm = 32, bn = 32;
  const metal_compute::ComputeFunction* fn = &_fn_gemm;
  if (M >= 128) {
    bm = 64;
    if (N >= 64) {
      fn = &_fn_gemm_bm64bn64;
      bn = 64;
    } else {
      fn = &_fn_gemm_bm64;
    }
  }
  enc.set_function(*fn);
  enc.set_buffer(0, x, xe * 2);
  enc.set_buffer(1, w);
  enc.set_buffer(2, bias != nullptr ? *bias : w);
  enc.set_buffer(3, y, ye * 2);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  enc.set_constant(6, M);
  enc.set_constant(7, bias != nullptr ? 1 : 0);
  enc.dispatch({(unsigned)(((N + bn - 1) / bn) * 32),
                (unsigned)(((M + bm - 1) / bm) * 2), 2}, {32, 2, 2});
}

const metal_compute::ComputeFunction*
MetalYue2Model::attn_fn_(int q_rows, int k_rows, bool i8)
{
  const int BQ = _use_nax ? kBqNax : kBqSteel;
  const int BK = _use_nax ? kBkNax : kBkSteel;
  const unsigned key = (i8 ? 8u : 0u) | (_use_nax ? 4u : 0u) |
                       ((q_rows % BQ) == 0 ? 2u : 0u) |
                       ((k_rows % BK) == 0 ? 1u : 0u);
  auto it = _attn_fns.find(key);
  if (it == _attn_fns.end()) {
    metal_compute::FunctionConstants fc;
    fc.set_bool(contract::kAttnAlignQ, (q_rows % BQ) == 0)
        .set_bool(contract::kAttnAlignK, (k_rows % BK) == 0)
        .set_bool(contract::kAttnHasMask, false)
        .set_bool(contract::kAttnCausal, false)
        .set_bool(contract::kAttnSinks, false)
        .set_bool(contract::kAttnQkInt8, i8);
    metal_compute::ComputeFunction fn =
        _use_nax ? _lib_attn_nax.function("attn_steel_nax_h_bd128_bf16", fc)
                 : _lib_attn.function("attn_steel_h_bd128_bf16", fc);
    it = _attn_fns.emplace(key, std::move(fn)).first;
  }
  return it->second.valid() ? &it->second : nullptr;
}

bool
MetalYue2Model::setup_chunk_(const std::vector<std::int32_t>& ar_tokens,
                             int frames, Chunk* c, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  const int n = (int)ar_tokens.size();
  c->ar_len = n;
  c->frames = frames;
  c->nar_rows = frames + 2;
  const int L_tot = n + c->nar_rows;
  if (L_tot > _cfg.context) {
    return fail(sfmt_("chunk of {} AR + {} latent rows exceeds the context",
                    n, c->nar_rows));
  }
  if (c->nar_rows > _cfg.max_latent_frames) {
    return fail("chunk exceeds the latent position table");
  }
  const int D = _cfg.head_dim;
  const int KV = _cfg.n_kv_heads;
  const std::size_t kvd = (std::size_t)KV * D;

  // The AR weights prefill the chunk's tokens once, causally. Their K/V
  // is what every NAR evaluation attends; MetalQwenModel leaves it in its
  // paged pool, from which it is gathered HEAD-MAJOR, page by page, into
  // the head of each layer's [kv_heads, ar_len + nar_rows, D] buffers.
  ContextManager* cm = _lm->context_manager();
  const ContextId cid = cm->acquire_root();
  {
    PerfAuxScope _perf(_mc->session(), kPerfLaneLLM, kGvidLlmPrefill,
                       kPerfLlmPrefillBegin, (std::uint64_t)n);
    SharedBuffer x = embed_(ar_tokens, 0, (std::size_t)n);
    SharedBuffer h = _lm->forward_embeddings_hidden(cid, x, n);
    if (x.empty() || h.empty()) {
      cm->release(cid);
      return fail("the chunk's AR prefill failed");
    }
  }
  const std::vector<PageId> pages = cm->pages_of(cid);
  const std::vector<int> valid = cm->pages_valid_of(cid);
  const int P = cm->page_tokens();
  const std::size_t pstride = cm->page_stride_bytes();
  c->k.resize((std::size_t)_cfg.n_layers);
  c->v.resize((std::size_t)_cfg.n_layers);
  CommandStream s = _mc->make_command_stream();
  {
    ComputeEncoder enc = s.begin_compute();
    for (int L = 0; L < _cfg.n_layers; ++L) {
      SharedBuffer& kb = c->k[(std::size_t)L];
      SharedBuffer& vb = c->v[(std::size_t)L];
      kb = _mc->make_shared_buffer((std::size_t)L_tot * kvd * 2);
      vb = _mc->make_shared_buffer((std::size_t)L_tot * kvd * 2);
      const SharedBuffer* kp = cm->kpool(L);
      const SharedBuffer* vp = cm->vpool(L);
      if (kb.empty() || vb.empty() || kp == nullptr || vp == nullptr) {
        cm->release(cid);
        return fail("KV allocation failed");
      }
      int row = 0;
      for (std::size_t pi = 0; pi < pages.size(); ++pi) {
        const int cnt = pi < valid.size() ? valid[pi] : 0;
        if (cnt <= 0) { continue; }
        const std::size_t poff = (std::size_t)pages[pi].v * pstride;
        for (int side = 0; side < 2; ++side) {
          enc.set_function(_fn_kv_gather);
          enc.set_buffer(0, side == 0 ? kb : vb);
          enc.set_buffer(1, side == 0 ? *kp : *vp, poff);
          enc.set_constant(2, P);
          enc.set_constant(3, D);
          enc.set_constant(4, L_tot);
          enc.set_constant(5, row);
          enc.dispatch({(unsigned)D, (unsigned)cnt, (unsigned)KV},
                       {(unsigned)D, 1, 1});
        }
        row += cnt;
      }
      if (row != n) {
        cm->release(cid);
        return fail(sfmt_("paged KV holds {} rows for {} tokens", row, n));
      }
    }
  }
  std::string gerr;
  const bool ok = s.commit().wait_ok(&gerr);
  cm->release(cid);
  if (!ok) { return fail("KV gather: " + gerr); }

  // Sol's exact sink: the text prefix, everything before the first codec
  // token -- instruction, tags, lyrics and the score. A centroid over
  // those is a prompt half-read.
  c->sink = n;
  for (int i = 0; i < n; ++i) {
    const std::int32_t t = ar_tokens[(std::size_t)i];
    if (t >= yue2::kCodecOffset && t < yue2::kCodecOffset + yue2::kCodecSize) {
      c->sink = i;
      break;
    }
  }

  // RoPE of the NAR rows: positions continue the AR sequence. The tables
  // are f32 but hold bf16-rounded values, as the reference casts its
  // cos/sin to the activations' dtype before rotating.
  const int half = D / 2;
  c->cos = _mc->make_shared_buffer((std::size_t)c->nar_rows * half * 4);
  c->sin = _mc->make_shared_buffer((std::size_t)c->nar_rows * half * 4);
  if (c->cos.empty() || c->sin.empty()) { return fail("rope tables"); }
  {
    auto* cs = static_cast<float*>(c->cos.contents());
    auto* sn = static_cast<float*>(c->sin.contents());
    std::vector<float> inv(half);
    for (int i = 0; i < half; ++i) {
      const float e = (float)(2 * i) / (float)D;
      inv[(std::size_t)i] = 1.0f / std::pow(_cfg.rope_theta, e);
    }
    for (int r = 0; r < c->nar_rows; ++r) {
      const float pos = (float)(n + r);
      for (int i = 0; i < half; ++i) {
        const float a = pos * inv[(std::size_t)i];
        cs[(std::size_t)r * half + i] = yue2::bf16_round(std::cos(a));
        sn[(std::size_t)r * half + i] = yue2::bf16_round(std::sin(a));
      }
    }
  }

  // The steel attention's parameter block: the NAR rows' queries against
  // [AR prefix ; NAR rows], every key visible.
  c->params = _mc->make_shared_buffer(sizeof(contract::SteelAttnParams));
  if (c->params.empty()) { return fail("attention params"); }
  {
    const int BQ = _use_nax ? kBqNax : kBqSteel;
    const int BK = _use_nax ? kBkNax : kBkSteel;
    const int Hq = _cfg.n_heads;
    const int qL = c->nar_rows, kL = L_tot;
    auto* p = static_cast<contract::SteelAttnParams*>(c->params.contents());
    std::memset(p, 0, sizeof(*p));
    p->B = 1;
    p->H = Hq;
    p->D = D;
    p->qL = qL;
    p->kL = kL;
    p->gqa_factor = Hq / KV;
    p->scale = 1.0f / std::sqrt((float)D);
    p->NQ = (qL + BQ - 1) / BQ;
    p->NK = (kL + BK - 1) / BK;
    p->NQ_aligned = qL / BQ;
    p->NK_aligned = kL / BK;
    p->qL_rem = qL - p->NQ_aligned * BQ;
    p->kL_rem = kL - p->NK_aligned * BK;
    p->qL_off = 0;
    // All four head-major: [heads, rows, D].
    p->Q_strides[0] = (std::int64_t)Hq * qL * D;
    p->Q_strides[1] = (std::int64_t)qL * D;
    p->Q_strides[2] = D;
    p->K_strides[0] = (std::int64_t)KV * kL * D;
    p->K_strides[1] = (std::int64_t)kL * D;
    p->K_strides[2] = D;
    p->V_strides[0] = p->K_strides[0];
    p->V_strides[1] = p->K_strides[1];
    p->V_strides[2] = D;
    p->O_strides[0] = p->Q_strides[0];
    p->O_strides[1] = p->Q_strides[1];
    p->O_strides[2] = D;
  }
  if (!ensure_scratch_(c->nar_rows)) { return fail("NAR scratch"); }
  return true;
}

void
MetalYue2Model::time_embedding_(double raw_t,
                                std::vector<std::uint16_t>* out) const
{
  // The reference rounds the raw timestep to bf16 to make a tensor, takes
  // its sigmoid there and applies the shift in bf16 too.
  const float rb = yue2::bf16_round((float)raw_t);
  float ts = yue2::bf16_round(1.0f / (1.0f + std::exp(-rb)));
  const float shift = _cfg.timestep_shift;
  if (shift != 1.0f) {
    const float num = yue2::bf16_round(shift * ts);
    const float den = yue2::bf16_round(
        1.0f + yue2::bf16_round((shift - 1.0f) * ts));
    ts = yue2::bf16_round(num / den);
  }
  // Sinusoidal [cos | sin] over 256 frequencies, then Linear -> SiLU ->
  // Linear, each rounded to bf16 as the bf16 MLP rounds it.
  const int half = 128;
  const std::size_t H = (std::size_t)_cfg.hidden;
  std::vector<float> e(2 * (std::size_t)half);
  const float lg = (float)(-std::log(10000.0));
  for (int i = 0; i < half; ++i) {
    const float f = std::exp(lg * (float)i / (float)half);
    const float a = ts * f;
    e[(std::size_t)i] = yue2::bf16_round(std::cos(a));
    e[(std::size_t)(half + i)] = yue2::bf16_round(std::sin(a));
  }
  std::vector<float> h1(H);
  for (std::size_t o = 0; o < H; ++o) {
    const float* w = _te0_w.data() + o * 256;
    float acc = 0.0f;
    for (int i = 0; i < 256; ++i) { acc += w[i] * e[(std::size_t)i]; }
    const float y = yue2::bf16_round(acc + _te0_b[o]);
    h1[o] = yue2::bf16_round(y / (1.0f + std::exp(-y)));
  }
  out->resize(H);
  for (std::size_t o = 0; o < H; ++o) {
    const float* w = _te2_w.data() + o * H;
    float acc = 0.0f;
    for (std::size_t i = 0; i < H; ++i) { acc += w[i] * h1[i]; }
    (*out)[o] = f32_to_bf16_bits_(acc + _te2_b[o]);
  }
}

bool
MetalYue2Model::eval_(Chunk& c, const std::vector<float>& state,
                      double raw_t, std::vector<float>* v, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  const int R = c.nar_rows;
  // THE FLOW MATCHING ON THE LLM LANE. This is a denoise in everything
  // but name -- two evaluations a midpoint step, and most of a song's
  // time on every machine -- so it takes the same gvid the image DiTs
  // do. Without it the profiler showed the AR phases and the KV gather
  // and then a blank stretch exactly where the time goes.
  PerfAuxScope _perf(_mc->session(), kPerfLaneLLM, kGvidLlmDit,
                     kPerfLlmDitBegin, (std::uint64_t)R);
  const int Z = _cfg.latent_dim;
  const int H = _cfg.hidden;
  const int D = _cfg.head_dim;
  const int Hq = _cfg.n_heads;
  const int KV = _cfg.n_kv_heads;
  const int qd = Hq * D;
  const int kvd = KV * D;
  const int F = _cfg.ffn;
  const int L_tot = c.ar_len + R;
  if ((int)state.size() != c.frames * Z) { return fail("state shape"); }

  // The int8 split's width, for the shapes an earlier evaluation recorded.
  // HERE because it runs its own command streams: evaluation 1 records,
  // 2 measures, every one after reads the cache. Every shape here is
  // short of the deep-K cliff (K <= 6144), so the context starts them
  // unsplit and weighs only S=2 against that -- which at 5347 rows ran
  // 0.67-0.90x of the single op (M5 Pro); untuned S=2 had made the tier
  // a loss over a song, 1258 ms per evaluation against dense 1219.
  if (_i8) { _i8->tune_pending(_mc); }
  const auto t_eval = std::chrono::steady_clock::now();

  // [START, latents, END]: the two borders carry zeros through vae2llm.
  {
    auto* d = static_cast<std::uint16_t*>(_n_in.contents());
    std::memset(d, 0, (std::size_t)R * Z * 2);
    for (std::size_t i = 0; i < state.size(); ++i) {
      d[(std::size_t)Z + i] = f32_to_bf16_bits_(state[i]);
    }
  }
  {
    std::vector<std::uint16_t> te;
    time_embedding_(raw_t, &te);
    std::memcpy(_n_t.contents(), te.data(), te.size() * 2);
  }

  const std::size_t n_rh = (std::size_t)R * H;
  CommandStream stream = _mc->make_command_stream();
  ComputeEncoder enc = stream.begin_compute();
  // Commit what is encoded, wait, and open a new buffer. The ANE split
  // needs it twice per block: once so its input is readable, once to
  // join the GPU's share before the residual reads every row.
  auto drain = [&](double* ms) -> bool {
    enc.end();
    const auto t0 = std::chrono::steady_clock::now();
    std::string ge;
    const bool ok = stream.commit().wait_ok(&ge);
    if (ms != nullptr) {
      *ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count();
    }
    stream = _mc->make_command_stream();
    enc = stream.begin_compute();
    if (!ok && err != nullptr) {
      *err = ge.empty() ? "NAR evaluation failed" : ge;
    }
    return ok;
  };
  auto add = [&](const SharedBuffer& a, const SharedBuffer& b,
                 std::size_t n) {
    if ((n % 4) == 0 && _fn_add4.valid()) {
      enc.set_function(_fn_add4);
      enc.set_buffer(0, a);
      enc.set_buffer(1, b);
      enc.set_buffer(2, a);
      enc.set_constant(3, (int)(n / 4));
      enc.dispatch({(unsigned)(n / 4), 1, 1}, {256, 1, 1});
      return;
    }
    enc.set_function(_fn_add);
    enc.set_buffer(0, a);
    enc.set_buffer(1, b);
    enc.set_buffer(2, a);
    enc.set_constant(3, (int)n);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
  };
  auto rms = [&](const SharedBuffer& x, const SharedBuffer& w,
                 const SharedBuffer& y) {
    enc.set_function(_fn_rms);
    enc.set_buffer(0, x);
    enc.set_buffer(1, w);
    enc.set_buffer(2, y);
    enc.set_constant(3, H);
    enc.set_constant(4, _cfg.rms_eps);
    enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
  };
  // Per-head RMSNorm in place on a token-major [R, heads * D] projection.
  auto norm_heads = [&](const SharedBuffer& x, const SharedBuffer& gamma,
                        int heads) {
    enc.set_function(_fn_rms_heads);
    enc.set_buffer(0, x);
    enc.set_buffer(1, gamma);
    enc.set_constant(2, R);
    enc.set_constant(3, heads);
    enc.set_constant(4, D);
    enc.set_constant(5, heads * D);
    enc.set_constant(6, 0);
    enc.set_constant(7, _cfg.rms_eps);
    enc.set_constant(8, D);
    enc.dispatch({32, (unsigned)(R * heads), 1}, {32, 1, 1});
  };
  // Token-major [R, heads * D] -> head-major rows [row0, row0 + R) of a
  // [heads, rows_per_head, D] buffer, rotating the first `rot` channels.
  // rot 0 is a plain transpose, which is how V goes.
  auto to_heads = [&](const SharedBuffer& in, const SharedBuffer& out,
                      int heads, int rows_per_head, int row0, int rot) {
    enc.set_function(_fn_tr_rope);
    enc.set_buffer(0, in);
    enc.set_buffer(1, out, (std::size_t)row0 * D * 2);
    enc.set_buffer(2, c.cos);
    enc.set_buffer(3, c.sin);
    enc.set_constant(4, heads);
    enc.set_constant(5, rows_per_head);
    enc.set_constant(6, D);
    enc.set_constant(7, rot);
    enc.set_constant(8, heads * D);
    enc.set_constant(9, 0);
    enc.set_constant(10, D);
    enc.dispatch({(unsigned)D, (unsigned)R, (unsigned)heads},
                 {(unsigned)D, 1, 1});
  };
  // The GPU's feed-forward over rows [r0, r0 + n) of _n_h into the same
  // rows of _n_o: its share of a split, every row of an unsplit block,
  // and the rows the ANE lost. The scratch is used from its first row.
  auto gpu_ff = [&](const NarLayer& ly, int r0, int n) {
    if (n <= 0) { return; }
    const std::size_t e0 = (std::size_t)r0 * H;
    gemm_(enc, _n_h, e0, ly.gate, nullptr, _n_g, 0, n, F, H);
    gemm_(enc, _n_h, e0, ly.up, nullptr, _n_u, 0, n, F, H);
    const std::size_t nf = (std::size_t)n * F;
    if ((nf % 4) == 0 && _fn_swiglu4.valid()) {
      enc.set_function(_fn_swiglu4);
      enc.set_buffer(0, _n_g);
      enc.set_buffer(1, _n_u);
      enc.set_buffer(2, _n_g);
      enc.set_constant(3, (int)(nf / 4));
      enc.dispatch({(unsigned)(nf / 4), 1, 1}, {256, 1, 1});
    } else {
      enc.set_function(_fn_swiglu);
      enc.set_buffer(0, _n_g);
      enc.set_buffer(1, _n_u);
      enc.set_buffer(2, _n_g);
      enc.set_constant(3, (int)nf);
      enc.dispatch({(unsigned)nf, 1, 1}, {256, 1, 1});
    }
    gemm_(enc, _n_g, 0, ly.down, nullptr, _n_o, e0, n, H, F);
  };
  // The attention: Q [Hq, R, D] against K/V [KV, L_tot, D], head-major,
  // into _n_aoh. Sol where it routes, else steel -- with Sage's int8 QK
  // when it is armed.
  const float scale = 1.0f / std::sqrt((float)D);
  auto attend = [&](int L, SharedBuffer& kb, SharedBuffer& vb) -> bool {
    const bool sage_here =
        _sage != nullptr && L >= _accel.sage.dense_layers;
    if (_sol != nullptr && L >= _accel.sol.dense_layers) {
      sol::Config sc = _accel.sol;
      sc.sink_start  = 0;
      sc.sink_tokens = c.sink;
      // SAGE ON SOL'S EXACT HALF, quantizing K RAW: the smoothing is exact
      // under ONE softmax and Sol splits the row across two.
      bool sage_ok = false;
      if (sage_here && _sol->uses_matrix_cores()) {
        const MetalSageAttention::Operand qo{&_n_qh, 0, D, R * D};
        const MetalSageAttention::Operand ko{&kb, 0, D, L_tot * D};
        std::string gerr;
        sage_ok = _sage->prepare(enc, qo, ko, Hq, KV, R, L_tot, D,
                                 _sol->query_block(), _sol->key_block_unit(),
                                 MetalSolAttention::compose(_accel.sage),
                                 &gerr);
      }
      _sol->set_sage(sage_ok ? _sage.get() : nullptr);
      MetalSolAttention::Band band;
      band.q_tokens = R;
      band.k_tokens = L_tot;
      band.q_rows   = R;
      band.k_rows   = L_tot;
      band.q_off    = c.ar_len;          // the queries' place in the keys
      band.q_row0   = 0;
      std::string serr;
      if (_sol->encode(enc, _n_qh, kb, vb, _n_aoh, Hq, KV, band, D, scale,
                       sc, &serr)) {
        // The total is the call's; the kept count accumulates on the GPU
        // until reset, so it is read once the evaluation has run.
        ++_stats.sol_layers;
        _stats.sol_total += _sol->total_blocks();
        if (sage_ok) { ++_stats.sage_layers; }
        return true;
      }
      if (!_sol_warned && _mc->session() != nullptr) {
        _sol_warned = true;
        _mc->session()->warn(fmt(
            "yue2: sol_attn declined ({}); the dense attention runs", serr));
      }
    }
    bool i8 = false;
    if (sage_here) {
      const MetalSageAttention::Operand qo{&_n_qh, 0, D, R * D};
      const MetalSageAttention::Operand ko{&kb, 0, D, L_tot * D};
      std::string gerr;
      i8 = _sage->prepare(enc, qo, ko, Hq, KV, R, L_tot, D,
                          MetalSageAttention::nax_query_block(),
                          MetalSageAttention::nax_key_block(), _accel.sage,
                          &gerr);
    }
    const metal_compute::ComputeFunction* afn = attn_fn_(R, L_tot, i8);
    if (afn == nullptr) { return false; }
    const int BQ = _use_nax ? kBqNax : kBqSteel;
    enc.set_function(*afn);
    enc.set_buffer(0, _n_qh);
    enc.set_buffer(1, kb);
    enc.set_buffer(2, vb);
    enc.set_buffer(3, _n_aoh);
    enc.set_buffer(4, c.params);
    if (i8) {
      _sage->bind(enc);
      ++_stats.sage_layers;
    }
    enc.dispatch({(unsigned)(32 * ((R + BQ - 1) / BQ)), (unsigned)(4 * Hq),
                  1}, {32, 4, 1});
    return true;
  };

  // x = vae2llm(latents) + time + position.
  gemm_(enc, _n_in, 0, _vae2llm_w, &_vae2llm_b, _n_x, 0, R, H, Z);
  enc.set_function(_fn_bias_rows);
  enc.set_buffer(0, _n_x);
  enc.set_buffer(1, _n_t);
  enc.set_constant(2, H);
  enc.set_constant(3, (std::uint32_t)n_rh);
  enc.dispatch({(unsigned)n_rh, 1, 1}, {256, 1, 1});
  add(_n_x, _pos, n_rh);

  bool first_split = true;
  const std::size_t tail = (std::size_t)c.ar_len;
  for (int L = 0; L < _cfg.n_layers; ++L) {
    const NarLayer& ly = _nar[(std::size_t)L];
    SharedBuffer& kb = c.k[(std::size_t)L];
    SharedBuffer& vb = c.v[(std::size_t)L];

    // THE ANE'S PLAN, decided before anything is encoded: its weights
    // stage under this block's attention.
    using Plan = AneFeedForward::Plan;
    Plan plan = Plan::kGpu;
    if (_accel.ane_ffn && ane_setup_(R) && ane_eligible_(L)) {
      plan = _ane->plan_block();
    }
    if (plan != Plan::kGpu && (first_split || _ane->needs_barrier())) {
      // An unsplit block commits nothing mid-block, so the split point's
      // drain would wait for everything queued before it. Drain first.
      if (!drain(nullptr)) { return false; }
      first_split = false;
    }
    if (plan == Plan::kSplit) { ane_stage_(L); }

    rms(_n_x, ly.in_ln, _n_h);
    gemm_(enc, _n_h, 0, ly.q, nullptr, _n_q, 0, R, qd, H);
    gemm_(enc, _n_h, 0, ly.k, nullptr, _n_k, 0, R, kvd, H);
    gemm_(enc, _n_h, 0, ly.v, nullptr, _n_vv, 0, R, kvd, H);
    norm_heads(_n_q, ly.q_norm, Hq);
    norm_heads(_n_k, ly.k_norm, KV);
    // Q head-major on its own; K and V behind the AR prefix's rows.
    to_heads(_n_q, _n_qh, Hq, R, 0, D);
    to_heads(_n_k, kb, KV, L_tot, (int)tail, D);
    to_heads(_n_vv, vb, KV, L_tot, (int)tail, 0);
    if (!attend(L, kb, vb)) {
      return fail("the steel attention did not resolve");
    }
    // [Hq, R, D] -> token-major [R, Hq * D] for the output projection.
    enc.set_function(_fn_transpose);
    enc.set_buffer(0, _n_aoh);
    enc.set_buffer(1, _n_ao);
    enc.set_constant(2, Hq);
    enc.set_constant(3, R);
    enc.set_constant(4, D);
    enc.dispatch({(unsigned)D, (unsigned)R, (unsigned)Hq},
                 {(unsigned)D, 1, 1});
    gemm_(enc, _n_ao, 0, ly.o, nullptr, _n_o, 0, R, H, qd);
    add(_n_x, _n_o, n_rh);
    rms(_n_x, ly.mlp_ln, _n_h);

    // ---- the ANE split point ------------------------------------------
    // _n_h holds the feed-forward input for every row. THE ORDER IS
    // LOAD-BEARING: drain, START THE ANE on the tail rows, and only then
    // encode the GPU's -- starting it after the GPU's gate/up are encoded
    // would leave only the down projection to hide behind.
    const bool split = plan == Plan::kSplit;
    const bool probe = plan == Plan::kProbe;
    int a_rows = 0;
    double drain_ms = 0.0;
    if (split || probe) {
      if (!drain(&drain_ms)) { return false; }
      if (split && _ane->join_stage(L)) {
        a_rows = _ane->begin(_n_h, _n_o, R);
      } else if (split && _mc->session() != nullptr) {
        _mc->session()->warn(fmt("yue2: staging NAR block {}'s weights for "
                                 "the ANE failed; it keeps the GPU "
                                 "feed-forward", L));
      }
    }
    const auto t_g0 = std::chrono::steady_clock::now();
    gpu_ff(ly, 0, R - a_rows);
    if (split || probe) {
      double gpu_ms = 0.0;
      enc.end();
      std::string ge;
      const bool ok = stream.commit().wait_ok(&ge);
      stream = _mc->make_command_stream();
      enc = stream.begin_compute();
      gpu_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t_g0).count();
      if (!ok) {
        if (a_rows > 0) { (void)_ane->finish(L, R, 0.0, drain_ms); }
        return fail(ge.empty() ? "NAR evaluation failed" : ge);
      }
      if (probe) { _ane->note_probe(drain_ms, gpu_ms); }
      // THE GPU FALLBACK: _n_h is only the tier's input, so rows it lost
      // can be run again from it.
      if (!_ane->finish(L, R, gpu_ms, drain_ms) && a_rows > 0) {
        const int lost = _ane->lost_row0();
        if (lost < 0 || lost >= R) { return fail("the ANE lost its rows"); }
        gpu_ff(ly, lost, R - lost);
        a_rows = std::max(0, lost - (R - a_rows));
      }
      _stats.ane_rows += a_rows;
    }
    _stats.ffn_rows += R;
    add(_n_x, _n_o, n_rh);
  }
  rms(_n_x, _final_ln, _n_h);
  gemm_(enc, _n_h, 0, _llm2vae_w, &_llm2vae_b, _n_v, 0, R, Z, H);
  if (!drain(nullptr)) { return false; }
  if (_sol != nullptr) {
    _stats.sol_exact += _sol->exact_blocks();
    _sol->reset_counts();
  }
  ++_stats.evals;
  _stats.ms += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t_eval).count();

  // The borders are dropped: only the latent rows have a velocity.
  v->resize((std::size_t)c.frames * Z);
  const auto* src = static_cast<const std::uint16_t*>(_n_v.contents());
  for (std::size_t i = 0; i < v->size(); ++i) {
    (*v)[i] = bf16_bits_to_f32_(src[(std::size_t)Z + i]);
  }
  return true;
}

bool
MetalYue2Model::velocity(const std::vector<std::int32_t>& ar_tokens,
                         const std::vector<float>& state, double raw_t,
                         std::vector<float>* v, std::string* err)
{
  const int Z = _cfg.latent_dim;
  if (state.empty() || state.size() % (std::size_t)Z != 0) {
    if (err != nullptr) { *err = "state is not [T, latent_dim]"; }
    return false;
  }
  Chunk c;
  if (!setup_chunk_(ar_tokens, (int)(state.size() / Z), &c, err)) {
    return false;
  }
  std::vector<float> st(state.size());
  for (std::size_t i = 0; i < st.size(); ++i) {
    st[i] = yue2::bf16_round(state[i]);
  }
  return eval_(c, st, raw_t, v, err);
}

bool
MetalYue2Model::synthesize(const std::vector<std::int32_t>& prefix,
                           const std::vector<std::int32_t>& codec,
                           std::uint64_t seed, int steps,
                           std::vector<float>* latents,
                           const ProgressFn& progress, std::string* err,
                           const std::vector<float>* noise)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  const int Z = _cfg.latent_dim;
  const int frames = (int)codec.size();
  if (prefix.empty() || frames < 1) { return fail("empty prefix or codec"); }
  if (steps < 1) { return fail("steps must be >= 1"); }
  for (std::int32_t cd : codec) {
    if (cd < 0 || cd >= yue2::kCodecSize) {
      return fail("a codec id is outside the codebook");
    }
  }
  const auto ranges = yue2::chunk_ranges(frames, (int)prefix.size(),
                                         _cfg.context);
  if (ranges.empty()) { return fail("the prefix leaves no acoustic context"); }
  // The whole song's noise is drawn once and cut at the chunk borders, so
  // a chunked song sees the same noise its frames would see unchunked.
  std::vector<float> nz;
  if (noise != nullptr) {
    if (noise->size() != (std::size_t)frames * Z) {
      return fail("noise is not [frames, latent_dim]");
    }
    nz = *noise;
  } else {
    nz = yue2::torch_cpu_randn(seed, (std::size_t)frames * Z);
  }

  latents->clear();
  latents->reserve((std::size_t)frames * Z);
  _stats = NarStats{};
  const int total = (int)ranges.size() * steps * 2;
  int done = 0;
  const double dt = 1.0 / (double)steps;
  auto raw_of = [](double t) {
    const double r = std::log(t / (1.0 - t));
    return std::clamp(r, -20.0, 20.0);   // t = 1 is +inf before the clamp
  };
  for (const auto& [a, b] : ranges) {
    std::vector<std::int32_t> ar = prefix;
    for (int i = a; i < b; ++i) {
      ar.push_back(codec[(std::size_t)i] + yue2::kCodecOffset);
    }
    ar.push_back(yue2::kMusicEnd);
    Chunk c;
    if (!setup_chunk_(ar, b - a, &c, err)) { return false; }
    const std::size_t n = (std::size_t)(b - a) * Z;
    std::vector<float> st(n), mid(n), v1, v2;
    for (std::size_t i = 0; i < n; ++i) {
      st[i] = yue2::bf16_round(nz[(std::size_t)a * Z + i]);
    }
    // Midpoint integration from t = 1 (noise) to t = 0, every update
    // rounded to bf16 as the reference's bf16 state is.
    const float half_dt = (float)(dt / 2.0);
    const float fdt = (float)dt;
    for (int k = 0; k < steps; ++k) {
      const double t = 1.0 - k * dt;
      if (!eval_(c, st, raw_of(t), &v1, err)) { return false; }
      if (progress && !progress(++done, total)) { return fail(""); }
      for (std::size_t i = 0; i < n; ++i) {
        mid[i] = yue2::bf16_round(st[i] - yue2::bf16_round(v1[i] * half_dt));
      }
      if (!eval_(c, mid, raw_of(t - dt / 2.0), &v2, err)) { return false; }
      if (progress && !progress(++done, total)) { return fail(""); }
      for (std::size_t i = 0; i < n; ++i) {
        st[i] = yue2::bf16_round(st[i] - yue2::bf16_round(v2[i] * fdt));
      }
    }
    for (float x : st) {
      if (!std::isfinite(x)) {
        return fail("flow matching produced non-finite latents");
      }
    }
    latents->insert(latents->end(), st.begin(), st.end());
  }
  return true;
}

}  // namespace vpipe::genai

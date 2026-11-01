#include "generative-models/lfm3/metal-lfm3-model.h"

#include "generative-models/lfm3/lfm3-weights.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/shared/i8-gemm.h"
#include "generative-models/weight-set.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-bag.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <utility>

namespace vpipe::genai {

using metal_compute::ComputeEncoder;
using metal_compute::SharedBuffer;

namespace {

std::string
str_(const FlexData& o, std::string_view k)
{
  return std::string(bag::text(o, k));
}

// The inverse frequencies of LFM3's RoPE for one layer kind: the first
// floor(prf * D) / 2 pairs follow the FULL head's progression,
// theta^(-2i / D), the rest are zero (those dims stay put).
bool
inv_freq_(const FlexData& rp, int D, std::vector<float>* out,
          std::string* why)
{
  if (str_(rp, "rope_type") != "default" && bag::has(rp, "rope_type")) {
    *why = "rope type '" + str_(rp, "rope_type") + "' is not supported";
    return false;
  }
  const double theta = bag::real(rp, "rope_theta", 0.0);
  if (theta <= 0.0) {
    *why = "no rope_theta";
    return false;
  }
  const double prf = bag::real(rp, "partial_rotary_factor", 1.0);
  const int ang = (int)(prf * D) / 2;
  out->assign((std::size_t)D / 2, 0.0f);
  for (int i = 0; i < ang && i < D / 2; ++i) {
    // torch: 1.0 / (base ** (arange(0, 2*ang, 2, float) / D)), in f32
    const float e = (float)(2 * i) / (float)D;
    (*out)[(std::size_t)i] = 1.0f / std::pow((float)theta, e);
  }
  return true;
}

}  // namespace

bool
MetalLfm3Model::Config::from_dir(const std::string& dir, Config* out,
                                 std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = "LFM3 config: " + std::move(m); }
    return false;
  };
  std::ifstream in(std::filesystem::path(dir) / "config.json");
  if (!in) { return fail("no config.json under " + dir); }
  std::stringstream ss;
  ss << in.rdbuf();
  FlexData root;
  try {
    root = FlexData::from_json(ss.str());
  } catch (const std::exception& e) {
    return fail(e.what());
  }
  const FlexData tc = bag::has(root, "text_config")
                          ? bag::value(root, "text_config") : root;
  if (str_(tc, "model_type") != "lfm3") {
    return fail("model_type is not lfm3");
  }
  Config c;
  c.n_layers = (int)bag::integer(tc, "num_hidden_layers", 0);
  c.hidden = (int)bag::integer(tc, "hidden_size", 0);
  c.n_heads = (int)bag::integer(tc, "num_attention_heads", 0);
  c.n_kv_heads = (int)bag::integer(tc, "num_key_value_heads", 0);
  c.head_dim = (int)bag::integer(tc, "head_dim",
                                 c.n_heads > 0 ? c.hidden / c.n_heads : 0);
  c.ffn = (int)bag::integer(tc, "intermediate_size", 0);
  c.vocab = (int)bag::integer(tc, "vocab_size", 0);
  c.conv_k = (int)bag::integer(tc, "conv_L_cache", 0);
  c.window = (int)bag::integer(tc, "sliding_window", 0);
  c.eps = (float)bag::real(tc, "norm_eps", 1e-5);
  const std::int64_t mpe = bag::integer(tc, "max_position_embeddings", 0);
  if (mpe > 0) { c.max_seq = (int)std::min<std::int64_t>(mpe, 1 << 20); }
  if (c.n_layers <= 0 || c.hidden <= 0 || c.n_heads <= 0
      || c.n_kv_heads <= 0 || c.head_dim <= 0 || c.ffn <= 0
      || c.vocab <= 0 || c.n_heads % c.n_kv_heads != 0) {
    return fail("missing or inconsistent sizes");
  }
  if (c.conv_k < 2 || c.conv_k > 9) {
    return fail("conv_L_cache " + std::to_string(c.conv_k) +
                " is outside 2..9");
  }
  if (c.head_dim != 128) {
    return fail("head_dim " + std::to_string(c.head_dim) +
                " (the attention kernels are head_dim 128)");
  }
  if (bag::flag(tc, "conv_bias", false)) {
    return fail("conv_bias is not supported");
  }
  if (!bag::flag(tc, "use_sandwich_norm", false)) {
    return fail("a config without use_sandwich_norm is not supported");
  }
  const FlexData lt = bag::value(tc, "layer_types");
  if (!lt.is_array() || (int)lt.as_array().size() != c.n_layers) {
    return fail("layer_types does not list every layer");
  }
  for (FlexData e : lt.as_array()) {
    const std::string t(e.as_string(""));
    if (t == "conv") {
      c.kinds.push_back(Kind::Conv);
    } else if (t == "sliding_attention") {
      c.kinds.push_back(Kind::Sliding);
    } else if (t == "full_attention") {
      c.kinds.push_back(Kind::Full);
    } else {
      return fail("layer type '" + t + "' is not supported");
    }
  }
  const FlexData rp = bag::value(tc, "rope_parameters");
  std::string why;
  bool need_full = false, need_sliding = false;
  for (Kind k : c.kinds) {
    need_full |= k == Kind::Full;
    need_sliding |= k == Kind::Sliding;
  }
  if (need_full
      && !inv_freq_(bag::value(rp, "full_attention"), c.head_dim,
                    &c.inv_freq_full, &why)) {
    return fail("full_attention: " + why);
  }
  if (need_sliding) {
    if (c.window <= 0) { return fail("no sliding_window"); }
    if (!inv_freq_(bag::value(rp, "sliding_attention"), c.head_dim,
                   &c.inv_freq_sliding, &why)) {
      return fail("sliding_attention: " + why);
    }
  }
  *out = std::move(c);
  return true;
}

std::size_t
MetalLfm3Model::weight_bytes(const std::string& model_dir)
{
  Config c;
  if (!Config::from_dir(model_dir, &c, nullptr)) { return 0; }
  const std::size_t H = (std::size_t)c.hidden, F = (std::size_t)c.ffn;
  std::size_t e = (std::size_t)c.vocab * H + H;   // tied table, final norm
  for (Kind k : c.kinds) {
    e += 4 * H + 3 * H * F;                       // norms, feed-forward
    if (k == Kind::Conv) {
      e += 3 * H * H + H * (std::size_t)c.conv_k + H * H;
    } else {
      e += (std::size_t)c.qkvg() * H + H * (std::size_t)c.qd() +
           2 * (std::size_t)c.head_dim;
    }
  }
  return e * 2;
}

std::size_t
MetalLfm3Model::run_bytes(const Config& c, std::int64_t kv_tokens,
                          int contexts, bool i8)
{
  const std::size_t R = (std::size_t)kPieceRows;
  const std::size_t H = (std::size_t)c.hidden, F = (std::size_t)c.ffn;
  const std::size_t D = (std::size_t)c.head_dim;
  const std::size_t big =
      std::max({3 * H, (std::size_t)c.qkvg(), 2 * F});
  const std::size_t mid = std::max(H, (std::size_t)c.qd());
  std::size_t per_row = 2 * (3 * H + big + mid + F);
  per_row += 2 * 2 * D * ((std::size_t)c.n_heads + c.n_kv_heads);  // q/a, k/v
  per_row += 4 * 4 * (D / 2) + 4;                         // tables, ids
  std::size_t b = R * per_row + (std::size_t)c.vocab * 2;
  // decode attention partials (split <= 32)
  b += (std::size_t)c.n_heads * 32 * (D + 2) * 4;
  // the paged K/V: every attention layer, at the doubling's capacity
  int attn = 0;
  for (Kind k : c.kinds) { attn += k != Kind::Conv; }
  const std::size_t pt = (std::size_t)std::max(c.page_tokens, 1);
  std::size_t pages = (std::size_t)((std::max<std::int64_t>(kv_tokens, 1) +
                                     (std::int64_t)pt - 1) /
                                    (std::int64_t)pt);
  // the pool doubles to the pages it needs, never past its cap, a
  // layer's old pool alive while it copies into the new
  std::size_t cap = 1;
  while (cap < pages) { cap *= 2; }
  if (c.max_pages > 0) { cap = std::min(cap, (std::size_t)c.max_pages); }
  const std::size_t layer_page = pt * 2 * (std::size_t)c.kd() * 2;
  b += cap * layer_page * (std::size_t)attn + (cap / 2) * layer_page;
  // each context's conv state
  const std::size_t conv = (std::size_t)(c.n_layers - attn) *
                           (std::size_t)(c.conv_k - 1) * H * 2;
  b += (std::size_t)std::max(contexts, 1) * conv;
  if (i8) {
    // The int8 context's grow-only scratches (shared/i8-gemm.cc), held
    // until the model goes: a piece's activation codes and group-64
    // scales at the widest K, one weight's codes and scales at the
    // largest, and the split tuner's output and planes at the widest N
    // -- below K 9216 it weighs S = 2 alone, every K here is.
    const std::size_t NQ = (std::size_t)c.qkvg(), QD = (std::size_t)c.qd();
    const std::size_t ka = std::max({H, F, QD});
    const std::size_t nmax = std::max({3 * H, NQ, 2 * F, H});
    const std::pair<std::size_t, std::size_t> wts[] = {
        {3 * H, H}, {H, H}, {NQ, H}, {H, QD}, {2 * F, H}, {H, F}};
    std::size_t wmax = 0;
    for (const auto& [n, k] : wts) {
      wmax = std::max(wmax, n * k + n * (k / 64) * 2);
    }
    const std::size_t planes =
        std::min(2 * R * nmax * 4, (std::size_t)512 << 20);
    b += R * ka + R * (ka / 64) * 2 + wmax + R * nmax * 2 + planes;
  }
  return b;
}

MetalLfm3Model::~MetalLfm3Model() = default;

void
MetalLfm3Model::set_i8_gemm(bool on)
{
  // The context self-gates on matrix cores and its kernels (f16: this
  // model's buffers); VPIPE_I8_GEMM overrides `on` either way. Asked
  // again for the same state, the context and what it settled stay.
  if (_i8_want == (on ? 1 : 0)) { return; }
  _i8_want = on ? 1 : 0;
  auto i8 = std::make_unique<I8GemmContext>(_mc, on, false);
  _i8 = i8->enabled() ? std::move(i8) : nullptr;
}

std::unique_ptr<MetalLfm3Model>
MetalLfm3Model::load(std::shared_ptr<WeightSet> ws,
                     metal_compute::MetalCompute* mc, const Config& cfg,
                     std::string* err)
{
  auto fail = [&](std::string m) -> std::unique_ptr<MetalLfm3Model> {
    if (err != nullptr) { *err = "MetalLfm3Model: " + std::move(m); }
    return nullptr;
  };
  if (mc == nullptr || !mc->valid() || ws == nullptr) {
    return fail("no GPU or no weights");
  }
  if (cfg.page_tokens % 16 != 0) {
    return fail("page_tokens must be a multiple of 16");
  }
  auto m = std::unique_ptr<MetalLfm3Model>(new MetalLfm3Model());
  m->_cfg = cfg;
  m->_mc = mc;
  m->_ws = std::move(ws);

  m->_lib_dense = mc->load_library("dense_gemm");
  m->_lib_rms = mc->load_library("rms_norm");
  m->_lib_rope = mc->load_library("rope");
  m->_lib_elt = mc->load_library("llm_elementwise");
  m->_lib_sdpa = mc->load_library("sdpa");
  m->_lib_attn = mc->load_library("attn_steel");
  m->_lib_lfm3 = mc->load_library("lfm3");
  m->_fn_gemm = m->_lib_dense.function("dense_gemm_t_bm64bn64_f16");
  m->_fn_gemm32 = m->_lib_dense.function("dense_gemm_t_f16");
  m->_fn_gemv_batch = m->_lib_lfm3.function("lfm3_gemv_batch_f16");
  m->_fn_rms = m->_lib_rms.function("rms_norm_f16");
  m->_fn_rms_add = m->_lib_rms.function("rms_add_f16");
  m->_fn_rms_rope = m->_lib_rope.function("rms_rope_half_ftab_tm_f16");
  m->_fn_transpose =
      m->_lib_rope.function("transpose_rope_half_part_ftab_f16");
  m->_fn_embed = m->_lib_elt.function("embed_gather_f16");
  m->_fn_swiglu = m->_lib_elt.function("swiglu_split_gate_first_f16");
  m->_fn_kv_write = m->_lib_elt.function("kv_write_paged_f16");
  m->_fn_dec_attn = m->_lib_sdpa.function("sdpa_paged_gqa_vec_f16");
  m->_fn_dec_merge = m->_lib_sdpa.function("sdpa_gqa_merge_f16");
  m->_fn_attn = m->_lib_attn.function("attn_steel_paged_bd128");
  m->_fn_attn_win = m->_lib_attn.function("attn_steel_paged_win_bd128");
  m->_fn_conv = m->_lib_lfm3.function("lfm3_shortconv_f16");
  m->_fn_conv_state = m->_lib_lfm3.function("lfm3_shortconv_state_f16");
  m->_fn_conv_step = m->_lib_lfm3.function("lfm3_shortconv_step_f16");
  m->_fn_gate = m->_lib_lfm3.function("lfm3_gate_heads_f16");
  for (const auto* f :
       {&m->_fn_gemm, &m->_fn_gemm32, &m->_fn_gemv_batch, &m->_fn_rms,
        &m->_fn_rms_add, &m->_fn_rms_rope, &m->_fn_transpose, &m->_fn_embed,
        &m->_fn_swiglu, &m->_fn_kv_write, &m->_fn_dec_attn,
        &m->_fn_dec_merge, &m->_fn_attn, &m->_fn_attn_win, &m->_fn_conv,
        &m->_fn_conv_state, &m->_fn_conv_step, &m->_fn_gate}) {
    if (!f->valid()) { return fail("a kernel did not load"); }
  }
  if (mc->supports_matrix_cores()
      && std::getenv("VPIPE_LFM3_NO_MMA2") == nullptr) {
    m->_lib_mma = mc->load_library("dense_gemm_mma");
    m->_fn_mma = m->_lib_mma.function("dense_gemm_mma_t_n128_f16");
    m->_fn_mma_deep = m->_lib_mma.function("dense_gemm_mma_t_n128x256_f16");
    m->_use_mma2 = m->_fn_mma.valid() && m->_fn_mma_deep.valid();
  }
  // Off until the stage asks (set_i8_gemm); VPIPE_I8_GEMM=1 forces it on.
  m->set_i8_gemm(false);

  // ---- weights -----------------------------------------------------
  Lfm3Weights w(m->_ws.get(), mc, "lfm3/f16/");
  const std::string pre =
      w.has("model.language_model.embed_tokens.weight")
          ? "model.language_model."
          : "model.";
  m->_embed = w.f16(pre + "embed_tokens.weight");
  m->_final_norm = w.f16(pre + "embedding_norm.weight");
  if (m->_embed.empty() || m->_final_norm.empty()) {
    return fail("no embed_tokens / embedding_norm under '" + pre + "'");
  }
  const int H = cfg.hidden;
  m->_layers.resize((std::size_t)cfg.n_layers);
  for (int L = 0; L < cfg.n_layers; ++L) {
    const std::string p = pre + "layers." + std::to_string(L) + ".";
    Layer& ly = m->_layers[(std::size_t)L];
    ly.kind = cfg.kinds[(std::size_t)L];
    ly.op_norm = w.f16(p + "operator_norm.weight");
    ly.post_op_norm = w.f16(p + "post_operator_norm.weight");
    ly.ffn_norm = w.f16(p + "ffn_norm.weight");
    ly.post_ffn_norm = w.f16(p + "post_ffn_norm.weight");
    ly.w13 = w.fused({p + "feed_forward.w1.weight",
                      p + "feed_forward.w3.weight"});
    ly.w2 = w.f16(p + "feed_forward.w2.weight");
    bool ok = !ly.op_norm.empty() && !ly.post_op_norm.empty()
        && !ly.ffn_norm.empty() && !ly.post_ffn_norm.empty()
        && !ly.w13.empty() && !ly.w2.empty();
    if (ly.kind == Kind::Conv) {
      ly.in_w = w.f16(p + "conv.in_proj.weight");
      ly.out_w = w.f16(p + "conv.out_proj.weight");
      ly.conv_w = w.f16(p + "conv.conv.weight");   // [H, 1, K] -> [H, K]
      ok = ok && !ly.in_w.empty() && !ly.out_w.empty()
          && ly.conv_w.byte_size() >= (std::size_t)H * cfg.conv_k * 2;
    } else {
      const std::string a = p + "self_attn.";
      ly.qkvg_w = w.fused({a + "q_proj.weight", a + "k_proj.weight",
                           a + "v_proj.weight", a + "gate_proj.weight"});
      ly.o_w = w.f16(a + "o_proj.weight");
      ly.q_norm = w.f16(a + "q_norm.weight");
      ly.k_norm = w.f16(a + "k_norm.weight");
      ok = ok && !ly.qkvg_w.empty() && !ly.o_w.empty()
          && !ly.q_norm.empty() && !ly.k_norm.empty()
          && ly.qkvg_w.byte_size() ==
                 (std::size_t)cfg.qkvg() * (std::size_t)H * 2;
    }
    if (!ok) {
      return fail("layer " + std::to_string(L) + ": a weight is missing "
                  "or has an unexpected shape");
    }
  }

  // ---- the context manager: paged K/V for the attention layers, the
  // conv state as a linear layer's (a 4-byte recurrent state beside it:
  // the manager refuses an empty one)
  ContextManager::Spec cs;
  cs.metal = mc;
  cs.n_layers = cfg.n_layers;
  cs.n_kv_heads = cfg.n_kv_heads;
  cs.head_dim = cfg.head_dim;
  cs.max_seq = cfg.max_seq;
  cs.page_tokens = cfg.page_tokens;
  cs.max_pages = (std::uint32_t)std::max(cfg.max_pages, 0);
  cs.is_linear_layer.assign((std::size_t)cfg.n_layers, false);
  for (int L = 0; L < cfg.n_layers; ++L) {
    cs.is_linear_layer[(std::size_t)L] =
        cfg.kinds[(std::size_t)L] == Kind::Conv;
  }
  cs.ssm_conv_kernel = cfg.conv_k;
  cs.ssm_conv_dim = cfg.hidden;
  cs.ssm_num_k_heads = 1;
  cs.ssm_num_v_heads = 1;
  cs.ssm_k_head_dim = 1;
  cs.ssm_v_head_dim = 1;
  m->_ctx = std::make_unique<ContextManager>(cs, mc->session());
  return m;
}

bool
MetalLfm3Model::ensure_scratch_(int rows, int logit_rows, int segs)
{
  const Config& c = _cfg;
  const std::size_t H = (std::size_t)c.hidden, F = (std::size_t)c.ffn;
  const std::size_t D = (std::size_t)c.head_dim;
  auto buf = [&](std::size_t bytes) {
    return _mc->make_shared_buffer(std::max<std::size_t>(bytes, 16));
  };
  if (rows > _s.rows) {
    const std::size_t R = (std::size_t)rows;
    const std::size_t big =
        std::max({3 * H, (std::size_t)c.qkvg(), 2 * F});
    const std::size_t mid = std::max(H, (std::size_t)c.qd());
    _s.x = buf(R * H * 2);
    _s.hn = buf(R * H * 2);
    _s.big = buf(R * big * 2);
    _s.mid = buf(R * mid * 2);
    _s.act = buf(R * F * 2);
    _s.o = buf(R * H * 2);
    _s.qt = buf((std::size_t)c.n_heads * R * D * 2);
    _s.at = buf((std::size_t)c.n_heads * R * D * 2);
    _s.kt = buf((std::size_t)c.n_kv_heads * R * D * 2);
    _s.vt = buf((std::size_t)c.n_kv_heads * R * D * 2);
    for (auto* t : {&_s.cos_f, &_s.sin_f, &_s.cos_s, &_s.sin_s}) {
      *t = buf(R * (D / 2) * 4);
    }
    _s.ids = buf(R * 4);
    _s.rows = rows;
  }
  if (logit_rows > _s.logits) {
    _s.lg = buf((std::size_t)logit_rows * (std::size_t)c.vocab * 2);
    _s.lrows = buf((std::size_t)logit_rows * H * 2);
    _s.logits = logit_rows;
  }
  if (_s.oacc.empty()) {
    _s.oacc = buf((std::size_t)c.n_heads * 32 * D * 4);
    _s.m = buf((std::size_t)c.n_heads * 32 * 4);
    _s.l = buf((std::size_t)c.n_heads * 32 * 4);
  }
  const std::size_t pgt =
      (std::size_t)std::max(segs, 1) * 2 *
      (std::size_t)_ctx->max_pages() * 3 * 4;
  if (_s.pgt.byte_size() < pgt) { _s.pgt = buf(pgt); }
  return !_s.x.empty() && !_s.big.empty() && !_s.lg.empty()
      && !_s.pgt.empty();
}

void
MetalLfm3Model::gemm_(ComputeEncoder& enc, const SharedBuffer& x,
                      std::size_t xoff, const SharedBuffer& w,
                      const SharedBuffer& y, std::size_t yoff, int M, int N,
                      int K, bool int8)
{
  if (M <= 8) {
    // the decode rows: the weight read once for all of them
    enc.set_function(_fn_gemv_batch);
    enc.set_buffer(0, x, xoff);
    enc.set_buffer(1, w);
    enc.set_buffer(2, y, yoff);
    enc.set_constant(3, K);
    enc.set_constant(4, N);
    enc.set_constant(5, M);
    enc.dispatch({32, (unsigned)(((N + 7) / 8) * 2), 1}, {32, 2, 1});
    return;
  }
  // The int8 prefill (opt-in, LOSSY): the activation and the f16 weight
  // quantized per call, the int8 matmul; it declines under its row floor
  // with nothing encoded. Offsets in elements there.
  if (int8 && _i8 &&
      _i8->gemm(enc, x, xoff / 2, w, y, yoff / 2, M, N, K)) {
    ++_i8_count;
    return;
  }
  if (_use_mma2 && M >= 64) {
    // matmul2d y = x @ w^T; the tensor extents clamp the tails
    const bool deep = K >= 6144;
    const int BN = deep ? 256 : 128;
    enc.set_function(deep ? _fn_mma_deep : _fn_mma);
    enc.set_buffer(0, x, xoff);
    enc.set_buffer(1, w);
    enc.set_buffer(2, w);              // bias slot unused (has_bias 0)
    enc.set_buffer(3, y, yoff);
    enc.set_constant(4, K);
    enc.set_constant(5, N);
    enc.set_constant(6, M);
    enc.set_constant(7, 0);
    enc.dispatch({(unsigned)(((N + BN - 1) / BN) * 256),
                  (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
    return;
  }
  // steel: the 64 x 64 tile from 64 rows, the 32 x 32 one below
  const bool wide = M >= 64;
  const int bm = wide ? 64 : 32, bn = wide ? 64 : 32;
  enc.set_function(wide ? _fn_gemm : _fn_gemm32);
  enc.set_buffer(0, x, xoff);
  enc.set_buffer(1, w);
  enc.set_buffer(2, w);                // bias slot unused (has_bias 0)
  enc.set_buffer(3, y, yoff);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  enc.set_constant(6, M);
  enc.set_constant(7, 0);
  enc.dispatch({(unsigned)(((N + bn - 1) / bn) * 32),
                (unsigned)(((M + bm - 1) / bm) * 2), 2}, {32, 2, 2});
}

bool
MetalLfm3Model::forward(std::span<const Segment> segs,
                        std::span<const std::int32_t> ids,
                        const std::vector<float>* emb, int logits_from,
                        std::vector<float>* logits,
                        std::vector<std::vector<float>>* capture)
{
  const Config& c = _cfg;
  const int H = c.hidden, D = c.head_dim, Hq = c.n_heads,
            Hkv = c.n_kv_heads;
  const int qd = c.qd(), kd = c.kd(), NQ = c.qkvg(), F = c.ffn;
  const int G = Hq / Hkv;
  if (segs.empty()) { return false; }
  using Clock = std::chrono::steady_clock;
  static const bool kProfile = std::getenv("VPIPE_LFM3_PROFILE") != nullptr;
  const auto t_in = Clock::now();
  auto t_enc = t_in, t_gpu = t_in;
  int total = 0;
  for (const Segment& s : segs) {
    if (s.n <= 0 || s.row0 < 0) { return false; }
    total = std::max(total, s.row0 + s.n);
  }
  const bool from_emb = emb != nullptr;
  if (from_emb ? emb->size() < (std::size_t)total * H
               : ids.size() < (std::size_t)total) {
    return false;
  }
  if (logits != nullptr && (logits_from < 0 || logits_from >= total)) {
    return false;
  }
  const int nlog = logits != nullptr ? total - logits_from : 0;
  if (!ensure_scratch_(total, std::max(nlog, 1), (int)segs.size())) {
    return false;
  }

  // ---- append every segment's rows; its page tables ---------------
  struct Chunk { std::size_t page_off; int slot; int src_off; int cnt; };
  struct Plan {
    int p0 = 0, n_pages = 0, n_win = 0, win_q = 0;
    std::size_t pgt_off = 0, win_off = 0;   // bytes into _s.pgt
    std::vector<Chunk> chunks;
  };
  std::vector<Plan> plans(segs.size());
  const int maxp = _ctx->max_pages();
  const std::size_t tbl = (std::size_t)maxp * 3 * 4;
  auto* pg = static_cast<std::uint8_t*>(_s.pgt.contents());
  for (std::size_t i = 0; i < segs.size(); ++i) {
    const Segment& s = segs[i];
    Plan& pl = plans[i];
    pl.p0 = -1;
    for (int done = 0; done < s.n; ) {
      const int cnt =
          std::min(s.n - done, _ctx->next_append_capacity(s.cid));
      ContextManager::AppendSlot a = _ctx->append(s.cid, cnt);
      if (!a.valid()) { return false; }
      if (pl.p0 < 0) { pl.p0 = a.position; }
      pl.chunks.push_back({(std::size_t)a.page_id.v *
                               _ctx->page_stride_bytes(),
                           a.slot_offset, done, cnt});
      done += cnt;
    }
  }
  for (std::size_t i = 0; i < segs.size(); ++i) {
    Plan& pl = plans[i];
    pl.pgt_off = (2 * i) * tbl;
    pl.win_off = (2 * i + 1) * tbl;
    auto* t = reinterpret_cast<std::int32_t*>(pg + pl.pgt_off);
    pl.n_pages = _ctx->fill_page_table(segs[i].cid, t);
    // The decode row's window: keys [wlo, q], the table and the query's
    // position shifted by wlo so the decode kernel's [0, q'] is exactly
    // the window (it reads a key's position only as gstart + slot).
    const int q = pl.p0 + segs[i].n - 1;
    const int wlo = std::max(0, q - c.window + 1);
    pl.win_q = q - wlo;
    auto* w = reinterpret_cast<std::int32_t*>(pg + pl.win_off);
    for (int p = 0; p < pl.n_pages; ++p) {
      if (t[p * 3 + 2] + t[p * 3 + 1] <= wlo) { continue; }
      w[pl.n_win * 3 + 0] = t[p * 3 + 0];
      w[pl.n_win * 3 + 1] = t[p * 3 + 1];
      w[pl.n_win * 3 + 2] = t[p * 3 + 2] - wlo;
      ++pl.n_win;
    }
  }

  // ---- inputs and the RoPE tables (f32, per row) -------------------
  {
    auto* cf = static_cast<float*>(_s.cos_f.contents());
    auto* sf = static_cast<float*>(_s.sin_f.contents());
    auto* cs = static_cast<float*>(_s.cos_s.contents());
    auto* ss = static_cast<float*>(_s.sin_s.contents());
    const int hd2 = D / 2;
    for (std::size_t i = 0; i < segs.size(); ++i) {
      for (int r = 0; r < segs[i].n; ++r) {
        const int row = segs[i].row0 + r;
        const float pos = (float)(plans[i].p0 + r);
        for (int k = 0; k < hd2; ++k) {
          const std::size_t o = (std::size_t)row * hd2 + k;
          if (!c.inv_freq_full.empty()) {
            const float a = pos * c.inv_freq_full[(std::size_t)k];
            cf[o] = std::cos(a);
            sf[o] = std::sin(a);
          }
          if (!c.inv_freq_sliding.empty()) {
            const float a = pos * c.inv_freq_sliding[(std::size_t)k];
            cs[o] = std::cos(a);
            ss[o] = std::sin(a);
          }
        }
      }
    }
    if (from_emb) {
      auto* x = static_cast<_Float16*>(_s.x.contents());
      for (std::size_t i = 0; i < (std::size_t)total * H; ++i) {
        x[i] = (_Float16)(*emb)[i];
      }
    } else {
      std::memcpy(_s.ids.contents(), ids.data(), (std::size_t)total * 4);
    }
  }

  // The int8 split's width for shapes an earlier piece recorded: here,
  // with no encoder open (it runs its own streams), and at a prefill's
  // forward only -- a decode step or a verify never meets the int8 floor,
  // so the first token after a prompt does not wait for the tuner. A
  // no-op once settled.
  if (_i8 && total > 8) { _i8->tune_pending(_mc); }
  t_enc = Clock::now();
  metal_compute::CommandStream stream = _mc->make_command_stream();
  ComputeEncoder enc = stream.begin_compute();
  auto rms = [&](const SharedBuffer& xin, std::size_t xoff,
                 const SharedBuffer& wt, const SharedBuffer& y,
                 std::size_t yoff, int R) {
    enc.set_function(_fn_rms);
    enc.set_buffer(0, xin, xoff);
    enc.set_buffer(1, wt);
    enc.set_buffer(2, y, yoff);
    enc.set_constant(3, H);
    enc.set_constant(4, c.eps);
    enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
  };
  // x += rms(o) * wt
  auto rms_add = [&](const SharedBuffer& wt) {
    const float one = 1.0f;
    enc.set_function(_fn_rms_add);
    enc.set_buffer(0, _s.o);
    enc.set_buffer(1, wt);
    enc.set_buffer(2, _s.x);
    enc.set_buffer(3, _s.x);
    enc.set_constant(4, H);
    enc.set_constant(5, c.eps);
    enc.set_constant(6, one);
    enc.dispatch({256, (unsigned)total, 1}, {256, 1, 1});
  };
  // per-head q / k norm + RoPE, in place, every row of the fused output
  auto norm_rope = [&](std::size_t col, int heads, const SharedBuffer& g,
                       bool sliding) {
    enc.set_function(_fn_rms_rope);
    enc.set_buffer(0, _s.big, col * 2);
    enc.set_buffer(1, sliding ? _s.cos_s : _s.cos_f);
    enc.set_buffer(2, sliding ? _s.sin_s : _s.sin_f);
    enc.set_buffer(3, g);
    enc.set_constant(4, heads);
    enc.set_constant(5, total);
    enc.set_constant(6, D);
    enc.set_constant(7, NQ);
    enc.set_constant(8, c.eps);
    enc.dispatch({32, (unsigned)total, (unsigned)heads}, {32, 1, 1});
  };
  // [n, heads, D] slice of the fused row -> head-major [heads, n, D]
  auto to_heads = [&](int row0, int n, std::size_t col, int heads,
                      const SharedBuffer& out) {
    const int rot = 0;
    enc.set_function(_fn_transpose);
    enc.set_buffer(0, _s.big, (std::size_t)row0 * NQ * 2);
    enc.set_buffer(1, out);
    enc.set_buffer(2, _s.cos_f);
    enc.set_buffer(3, _s.sin_f);
    enc.set_constant(4, heads);
    enc.set_constant(5, n);
    enc.set_constant(6, D);
    enc.set_constant(7, rot);
    enc.set_constant(8, NQ);
    enc.set_constant(9, (int)col);
    enc.set_constant(10, D);
    enc.dispatch({(unsigned)D, (unsigned)n, (unsigned)heads},
                 {(unsigned)D, 1, 1});
  };
  auto kv_write = [&](const SharedBuffer& src, std::size_t src_off_b,
                      int n_src, const SharedBuffer& pool, const Chunk& ch,
                      int src_row) {
    enc.set_function(_fn_kv_write);
    enc.set_buffer(0, src, src_off_b);
    enc.set_buffer(1, pool, ch.page_off);
    enc.set_constant(2, _ctx->page_tokens());
    enc.set_constant(3, D);
    enc.set_constant(4, n_src);
    enc.set_constant(5, src_row);
    enc.set_constant(6, ch.slot);
    enc.dispatch({(unsigned)D, (unsigned)ch.cnt, (unsigned)Hkv},
                 {(unsigned)D, 1, 1});
  };
  auto gate = [&](const SharedBuffer& a, int row0, int n) {
    enc.set_function(_fn_gate);
    enc.set_buffer(0, a);
    enc.set_buffer(1, _s.big, ((std::size_t)row0 * NQ + qd + 2 * kd) * 2);
    enc.set_buffer(2, _s.mid, (std::size_t)row0 * qd * 2);
    enc.set_constant(3, Hq);
    enc.set_constant(4, n);
    enc.set_constant(5, D);
    enc.set_constant(6, NQ);
    enc.set_constant(7, qd);
    enc.dispatch({(unsigned)D, (unsigned)n, (unsigned)Hq},
                 {(unsigned)D, 1, 1});
  };
  const float scale = 1.0f / std::sqrt((float)D);
  const int page_tokens = _ctx->page_tokens();

  if (!from_emb) {
    enc.set_function(_fn_embed);
    enc.set_buffer(0, _s.ids);
    enc.set_buffer(1, _embed);
    enc.set_buffer(2, _s.x);
    enc.set_constant(3, H);
    enc.dispatch({(unsigned)H, (unsigned)total, 1}, {256, 1, 1});
  }

  for (int L = 0; L < c.n_layers; ++L) {
    const Layer& ly = _layers[(std::size_t)L];
    rms(_s.x, 0, ly.op_norm, _s.hn, 0, total);
    if (ly.kind == Kind::Conv) {
      gemm_(enc, _s.hn, 0, ly.in_w, _s.big, 0, total, 3 * H, H);
      for (const Segment& s : segs) {
        const SharedBuffer* st = _ctx->conv_state(s.cid, L);
        if (st == nullptr) { return false; }
        const std::size_t in_off = (std::size_t)s.row0 * 3 * H * 2;
        const std::size_t out_off = (std::size_t)s.row0 * H * 2;
        if (s.n == 1) {
          enc.set_function(_fn_conv_step);
          enc.set_buffer(0, _s.big, in_off);
          enc.set_buffer(1, *st);
          enc.set_buffer(2, ly.conv_w);
          enc.set_buffer(3, _s.mid, out_off);
          enc.set_constant(4, H);
          enc.set_constant(5, c.conv_k);
          enc.dispatch({(unsigned)H, 1, 1}, {256, 1, 1});
          continue;
        }
        const int stride = 3 * H;
        enc.set_function(_fn_conv);
        enc.set_buffer(0, _s.big, in_off);
        enc.set_buffer(1, *st);
        enc.set_buffer(2, ly.conv_w);
        enc.set_buffer(3, _s.mid, out_off);
        enc.set_constant(4, H);
        enc.set_constant(5, s.n);
        enc.set_constant(6, c.conv_k);
        enc.set_constant(7, stride);
        enc.set_constant(8, H);
        enc.dispatch({(unsigned)H, (unsigned)s.n, 1}, {256, 1, 1});
        enc.set_function(_fn_conv_state);
        enc.set_buffer(0, _s.big, in_off);
        enc.set_buffer(1, *st);
        enc.set_constant(2, H);
        enc.set_constant(3, s.n);
        enc.set_constant(4, c.conv_k);
        enc.set_constant(5, stride);
        enc.dispatch({(unsigned)H, 1, 1}, {256, 1, 1});
      }
      gemm_(enc, _s.mid, 0, ly.out_w, _s.o, 0, total, H, H);
    } else {
      const bool sliding = ly.kind == Kind::Sliding;
      gemm_(enc, _s.hn, 0, ly.qkvg_w, _s.big, 0, total, NQ, H);
      norm_rope(0, Hq, ly.q_norm, sliding);
      norm_rope((std::size_t)qd, Hkv, ly.k_norm, sliding);
      const SharedBuffer& kp = *_ctx->kpool(L);
      const SharedBuffer& vp = *_ctx->vpool(L);
      for (std::size_t i = 0; i < segs.size(); ++i) {
        const Segment& s = segs[i];
        const Plan& pl = plans[i];
        const std::size_t row_b = (std::size_t)s.row0 * NQ * 2;
        if (s.n == 1) {
          kv_write(_s.big, row_b + (std::size_t)qd * 2, 1, kp,
                   pl.chunks[0], 0);
          kv_write(_s.big, row_b + (std::size_t)(qd + kd) * 2, 1, vp,
                   pl.chunks[0], 0);
          const int keys = sliding ? pl.win_q + 1 : pl.p0 + 1;
          const int split = std::clamp(keys / 128, 1, 32);
          const int qpos = sliding ? pl.win_q : pl.p0;
          const int npg = sliding ? pl.n_win : pl.n_pages;
          enc.set_function(_fn_dec_attn);
          enc.set_buffer(0, _s.big, row_b);
          enc.set_buffer(1, kp);
          enc.set_buffer(2, vp);
          enc.set_buffer(3, _s.oacc);
          enc.set_buffer(4, _s.m);
          enc.set_buffer(5, _s.l);
          enc.set_constant(6, scale);
          enc.set_constant(7, D);
          enc.set_constant(8, Hq);
          enc.set_constant(9, Hkv);
          enc.set_constant(10, qpos);
          enc.set_constant(11, page_tokens);
          enc.set_constant(12, npg);
          enc.set_buffer(13, _s.pgt, sliding ? pl.win_off : pl.pgt_off);
          enc.set_constant(14, split);
          enc.dispatch({32, (unsigned)Hq, (unsigned)split},
                       {32, (unsigned)G, 1});
          enc.set_function(_fn_dec_merge);
          enc.set_buffer(0, _s.oacc);
          enc.set_buffer(1, _s.m);
          enc.set_buffer(2, _s.l);
          enc.set_buffer(3, _s.at);
          enc.set_constant(4, D);
          enc.set_constant(5, split);
          enc.set_constant(6, Hq);
          enc.dispatch({(unsigned)(Hq * D), 1, 1}, {256, 1, 1});
          gate(_s.at, s.row0, 1);
          continue;
        }
        to_heads(s.row0, s.n, 0, Hq, _s.qt);
        to_heads(s.row0, s.n, (std::size_t)qd, Hkv, _s.kt);
        to_heads(s.row0, s.n, (std::size_t)(qd + kd), Hkv, _s.vt);
        for (const Chunk& ch : pl.chunks) {
          kv_write(_s.kt, 0, s.n, kp, ch, ch.src_off);
          kv_write(_s.vt, 0, s.n, vp, ch, ch.src_off);
        }
        enc.set_function(sliding ? _fn_attn_win : _fn_attn);
        enc.set_buffer(0, _s.qt);
        enc.set_buffer(1, kp);
        enc.set_buffer(2, vp);
        enc.set_buffer(3, _s.at);
        enc.set_constant(4, scale);
        enc.set_constant(5, D);
        enc.set_constant(6, Hq);
        enc.set_constant(7, Hkv);
        enc.set_constant(8, s.n);
        enc.set_constant(9, pl.p0);
        enc.set_constant(10, page_tokens);
        enc.set_constant(11, pl.n_pages);
        enc.set_buffer(12, _s.pgt, pl.pgt_off);
        enc.set_constant(13, c.window);
        enc.dispatch({(unsigned)(32 * ((s.n + 31) / 32)),
                      (unsigned)(4 * Hq), 1}, {32, 4, 1});
        gate(_s.at, s.row0, s.n);
      }
      gemm_(enc, _s.mid, 0, ly.o_w, _s.o, 0, total, H, qd);
    }
    rms_add(ly.post_op_norm);
    rms(_s.x, 0, ly.ffn_norm, _s.hn, 0, total);
    gemm_(enc, _s.hn, 0, ly.w13, _s.big, 0, total, 2 * F, H);
    enc.set_function(_fn_swiglu);
    enc.set_buffer(0, _s.big);
    enc.set_buffer(1, _s.act);
    enc.set_constant(2, total);
    enc.set_constant(3, F);
    enc.dispatch({(unsigned)(total * F), 1, 1}, {256, 1, 1});
    gemm_(enc, _s.act, 0, ly.w2, _s.o, 0, total, H, F);
    rms_add(ly.post_ffn_norm);
    if (capture != nullptr) {
      enc.end();
      if (!stream.commit().wait_ok()) { return false; }
      const auto* xs = static_cast<const _Float16*>(_s.x.contents());
      std::vector<float> row((std::size_t)total * H);
      for (std::size_t i = 0; i < row.size(); ++i) { row[i] = (float)xs[i]; }
      capture->push_back(std::move(row));
      enc = stream.begin_compute();
    }
  }
  if (nlog > 0) {
    rms(_s.x, (std::size_t)logits_from * H * 2, _final_norm, _s.lrows, 0,
        nlog);
    gemm_(enc, _s.lrows, 0, _embed, _s.lg, 0, nlog, c.vocab, H, false);
  }
  enc.end();
  t_gpu = Clock::now();
  std::string why;
  if (!stream.commit().wait_ok(&why)) {
    if (const SessionContextIntf* sess = _mc->session()) {
      sess->warn(fmt("MetalLfm3Model: the forward failed on the GPU ({})",
                     why));
    }
    return false;
  }
  const auto t_done = Clock::now();
  if (logits != nullptr) {
    const auto* lg = static_cast<const _Float16*>(_s.lg.contents());
    logits->resize((std::size_t)nlog * c.vocab);
    for (std::size_t i = 0; i < logits->size(); ++i) {
      (*logits)[i] = (float)lg[i];
    }
  }
  if (kProfile) {
    auto ms = [](Clock::duration d) {
      return std::chrono::duration<double, std::milli>(d).count();
    };
    if (const SessionContextIntf* sess = _mc->session()) {
      sess->info(fmt("MetalLfm3Model: {} rows in {} segments | host "
                     "{:.2f} ms, encode {:.2f}, GPU {:.2f}, logits {:.2f}",
                     total, segs.size(), ms(t_enc - t_in),
                     ms(t_gpu - t_enc), ms(t_done - t_gpu),
                     ms(Clock::now() - t_done)));
    }
  }
  return true;
}

bool
MetalLfm3Model::prefill(ContextId cid, std::span<const std::int32_t> ids,
                        const std::vector<float>* emb,
                        std::vector<float>* logits)
{
  const int n = emb != nullptr
      ? (int)(emb->size() / (std::size_t)_cfg.hidden) : (int)ids.size();
  if (n <= 0) { return false; }
  std::vector<float> piece_emb;
  for (int at = 0; at < n; at += kPieceRows) {
    const int k = std::min(kPieceRows, n - at);
    const bool last = at + k == n;
    Segment s{cid, 0, k};
    const std::vector<float>* e = nullptr;
    if (emb != nullptr) {
      if (at == 0 && k == n) {
        e = emb;
      } else {
        piece_emb.assign(emb->begin() + (long)at * _cfg.hidden,
                         emb->begin() + (long)(at + k) * _cfg.hidden);
        e = &piece_emb;
      }
    }
    const std::span<const std::int32_t> pid =
        emb != nullptr ? std::span<const std::int32_t>()
                       : ids.subspan((std::size_t)at, (std::size_t)k);
    if (!forward(std::span<const Segment>(&s, 1), pid, e,
                 last ? k - 1 : 0, last ? logits : nullptr)) {
      return false;
    }
  }
  return true;
}

std::vector<float>
MetalLfm3Model::embed_rows(std::span<const std::int32_t> ids) const
{
  const int H = _cfg.hidden;
  std::vector<float> out(ids.size() * (std::size_t)H);
  const auto* t = static_cast<const _Float16*>(_embed.contents());
  for (std::size_t i = 0; i < ids.size(); ++i) {
    const std::int32_t id = ids[i];
    if (id < 0 || id >= _cfg.vocab) { return {}; }
    const _Float16* r = t + (std::size_t)id * H;
    for (int k = 0; k < H; ++k) { out[i * H + k] = (float)r[k]; }
  }
  return out;
}

std::size_t
MetalLfm3Model::kv_bytes() const
{
  return _ctx != nullptr ? _ctx->resident_bytes() : 0;
}

}  // namespace vpipe::genai

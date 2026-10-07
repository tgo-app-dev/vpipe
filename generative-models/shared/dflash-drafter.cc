#include "generative-models/shared/dflash-drafter.h"

#include "common/flex-data.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/weight-set.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <utility>

namespace vpipe::genai {

using metal_compute::ComputeEncoder;
using metal_compute::ComputeFunction;
using metal_compute::ComputeLibrary;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

std::optional<FlexData>
read_json_(const std::string& path)
{
  std::ifstream in(path);
  if (!in) { return std::nullopt; }
  try {
    return FlexData::from_json(in);
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

std::vector<int>
int_list_(const FlexData& v)
{
  std::vector<int> out;
  if (!v.is_array()) { return out; }
  const std::span<const std::int64_t> sp = v.as_int_span();
  if (!sp.empty()) {
    for (const std::int64_t x : sp) { out.push_back((int)x); }
    return out;
  }
  const FlexData::ConstArrayView a = v.as_array();
  for (std::size_t i = 0; i < a.size(); ++i) {
    out.push_back((int)a[i].as_int(0));
  }
  return out;
}

float
bf16_to_f32_(std::uint16_t b)
{
  const std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

std::uint16_t
f32_to_bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0) {
    return (std::uint16_t)((u >> 16) | 0x40);   // keep a NaN a NaN
  }
  u += 0x7fffu + ((u >> 16) & 1u);              // round to nearest even
  return (std::uint16_t)(u >> 16);
}

// Element size of a checkpoint dtype the drafter reads (0 = unsupported).
int
src_elt_bytes_(const std::string& dt)
{
  if (dt == "BF16" || dt == "F16") { return 2; }
  if (dt == "F32") { return 4; }
  return 0;
}

// Convert `n` elements of `src` (checkpoint dtype `dt`) into `dst` in the
// compute dtype (bf16 when `bf16`, else f16).
void
convert_(const void* src, const std::string& dt, std::uint16_t* dst,
         std::size_t n, bool bf16)
{
  if (dt == "BF16") {
    const auto* s = static_cast<const std::uint16_t*>(src);
    if (bf16) {
      std::memcpy(dst, s, n * 2);
      return;
    }
    for (std::size_t i = 0; i < n; ++i) {
      const _Float16 h = (_Float16)bf16_to_f32_(s[i]);
      std::memcpy(&dst[i], &h, 2);
    }
    return;
  }
  if (dt == "F16") {
    const auto* s = static_cast<const _Float16*>(src);
    if (!bf16) {
      std::memcpy(dst, s, n * 2);
      return;
    }
    for (std::size_t i = 0; i < n; ++i) {
      dst[i] = f32_to_bf16_((float)s[i]);
    }
    return;
  }
  const auto* s = static_cast<const float*>(src);   // F32
  for (std::size_t i = 0; i < n; ++i) {
    if (bf16) {
      dst[i] = f32_to_bf16_(s[i]);
    } else {
      const _Float16 h = (_Float16)s[i];
      std::memcpy(&dst[i], &h, 2);
    }
  }
}

std::size_t
align_up_(std::size_t v, std::size_t a)
{
  return (v + a - 1) / a * a;
}

}  // namespace

// ---------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------

std::optional<DFlashConfig>
DFlashConfig::from_dir(const std::string& dir, std::string* err)
{
  auto fail = [&](const std::string& why) -> std::optional<DFlashConfig> {
    if (err) { *err = why; }
    return std::nullopt;
  };
  const std::optional<FlexData> doc = read_json_(dir + "/config.json");
  if (!doc || !doc->is_object()) {
    return fail("no readable config.json in '" + dir + "'");
  }
  const FlexData::ConstObjectView root = doc->as_object();
  bool is_v1 = false, is_v2 = false;
  if (root.contains("architectures")) {
    const FlexData arch = root.at("architectures");
    if (arch.is_array()) {
      const FlexData::ConstArrayView a = arch.as_array();
      for (std::size_t i = 0; i < a.size(); ++i) {
        // a[i] is a temporary: copy the string out before it goes.
        const std::string s(a[i].as_string());
        if (s == "DFlash2DraftModel") { is_v2 = true; }
        if (s == "DFlashDraftModel") { is_v1 = true; }
      }
    }
  }
  if (!is_v1 && !is_v2) {
    return fail("config.json names neither DFlashDraftModel nor "
                "DFlash2DraftModel");
  }
  auto geti = [&](const char* k, int def) {
    return root.contains(k) ? (int)root.at(k).as_int(def) : def;
  };
  auto getr = [&](const char* k, double def) {
    return root.contains(k) ? root.at(k).as_real(def) : def;
  };
  DFlashConfig c;
  c.hidden = geti("hidden_size", 0);
  c.n_layers = geti("num_hidden_layers", 0);
  c.n_heads = geti("num_attention_heads", 0);
  c.n_kv_heads = geti("num_key_value_heads", c.n_heads);
  c.head_dim = geti("head_dim", c.n_heads > 0 ? c.hidden / c.n_heads : 0);
  c.ffn = geti("intermediate_size", 0);
  c.vocab = geti("vocab_size", 0);
  c.rms_eps = (float)getr("rms_norm_eps", 1e-6);
  c.num_target_layers = geti("num_target_layers", 0);
  double theta = getr("rope_theta", 0.0);
  for (const char* rk : {"rope_parameters", "rope_scaling"}) {
    if (theta > 0.0 || !root.contains(rk)) { continue; }
    const FlexData rp = root.at(rk);
    if (!rp.is_object()) { continue; }
    const FlexData::ConstObjectView rv = rp.as_object();
    if (rv.contains("rope_type")) {
      const std::string t(rv.at("rope_type").as_string());
      if (!t.empty() && t != "default") {
        return fail("unsupported drafter rope_type '" + t + "'");
      }
    }
    if (rv.contains("rope_theta")) {
      theta = rv.at("rope_theta").as_real(0.0);
    }
  }
  c.rope_theta = (float)(theta > 0.0 ? theta : 10000.0);
  if (root.contains("hidden_act")) {
    const std::string act(root.at("hidden_act").as_string());
    if (act != "silu") {
      return fail("unsupported drafter hidden_act '" + act + "'");
    }
  }
  if (root.contains("attention_bias") &&
      root.at("attention_bias").as_bool(false)) {
    return fail("drafter attention_bias is not supported");
  }

  // dflash_config (values may also sit at the top level, as the
  // reference's _draft_value reads them).
  FlexData dfc = FlexData::make_object();
  if (root.contains("dflash_config")) { dfc = root.at("dflash_config"); }
  const FlexData::ConstObjectView dv = dfc.as_object();
  auto dget = [&](const char* k) -> std::optional<FlexData> {
    if (dv.contains(k)) { return dv.at(k); }
    if (root.contains(k)) { return root.at(k); }
    return std::nullopt;
  };
  if (auto v = dget("block_size")) { c.block_size = (int)v->as_int(16); }
  if (auto v = dget("mask_token_id")) { c.mask_token_id = (int)v->as_int(0); }
  else { return fail("dflash_config has no mask_token_id"); }
  if (auto v = dget("target_layer_ids")) { c.target_layer_ids = int_list_(*v); }
  if (c.target_layer_ids.empty()) {
    // The reference's build_target_layer_ids: spread over [1, N - 3].
    const int nl = c.n_layers, nt = c.num_target_layers;
    if (nl <= 0 || nt <= 0) {
      return fail("no target_layer_ids and no num_target_layers");
    }
    if (nl == 1) {
      c.target_layer_ids.push_back(nt / 2);
    } else {
      const int start = 1, end = nt - 3, span = end - start;
      for (int i = 0; i < nl; ++i) {
        c.target_layer_ids.push_back((int)std::lround(
            start + (double)i * span / (double)(nl - 1)));
      }
    }
  }
  if (auto v = dget("input_embedding_scale")) {
    c.input_embedding_scale = (float)v->as_real(1.0);
  }
  if (auto v = dget("output_multiplier")) {
    c.output_multiplier = (float)v->as_real(1.0);
  }
  if (auto v = dget("final_logit_softcapping")) {
    if (!v->is_null()) { c.final_logit_softcap = (float)v->as_real(0.0); }
  }
  if (is_v2) {
    if (auto v = dget("conv_kernel_size")) {
      c.conv_kernel = (int)v->as_int(0);
    }
    if (auto v = dget("conv_group_size")) { c.conv_group = (int)v->as_int(0); }
    if (auto v = dget("selector_rank")) { c.selector_rank = (int)v->as_int(0); }
    if (auto v = dget("selector_top_k")) {
      c.selector_top_k = (int)v->as_int(0);
    }
    if (c.conv_kernel <= 0 || c.conv_group <= 0 || c.selector_rank <= 0 ||
        c.selector_top_k <= 0) {
      return fail("DFlash2DraftModel without conv_* / selector_* fields");
    }
    if (c.selector_top_k != 16) {
      return fail("selector_top_k " + std::to_string(c.selector_top_k) +
                  " (only 16 is implemented)");
    }
    if (c.selector_rank % 32 != 0 || c.selector_rank > 1024) {
      return fail("selector_rank must be a multiple of 32 up to 1024");
    }
    if (c.hidden % c.conv_group != 0) {
      return fail("hidden_size is not a multiple of conv_group_size");
    }
  }

  // Per-layer attention: layer_types (sliding vs full) and is_causal.
  std::vector<std::string> types;
  if (root.contains("layer_types")) {
    const FlexData lt = root.at("layer_types");
    if (lt.is_array()) {
      const FlexData::ConstArrayView a = lt.as_array();
      for (std::size_t i = 0; i < a.size(); ++i) {
        types.emplace_back(a[i].as_string());
      }
    }
  }
  if (types.empty()) {
    types.assign((std::size_t)c.n_layers, "full_attention");
  }
  if ((int)types.size() != c.n_layers) {
    return fail("layer_types length != num_hidden_layers");
  }
  const int window = geti("sliding_window", 0);
  const bool has_causal =
      root.contains("is_causal") && !root.at("is_causal").is_null();
  const bool causal_all =
      has_causal && root.at("is_causal").as_bool(false);
  for (const std::string& t : types) {
    const bool sliding = (t == "sliding_attention");
    if (!sliding && t != "full_attention") {
      return fail("unsupported drafter layer type '" + t + "'");
    }
    if (sliding && window <= 0) {
      return fail("sliding_attention layer without sliding_window");
    }
    c.layer_window.push_back(sliding ? window : 0);
    // The reference: causal iff sliding, unless is_causal says otherwise.
    c.layer_causal.push_back(has_causal ? causal_all : sliding);
  }

  if (c.hidden <= 0 || c.n_layers <= 0 || c.n_heads <= 0 ||
      c.n_kv_heads <= 0 || c.head_dim <= 0 || c.ffn <= 0 || c.vocab <= 0) {
    return fail("drafter config is missing a dimension");
  }
  if (c.n_heads % c.n_kv_heads != 0) {
    return fail("num_attention_heads is not a multiple of "
                "num_key_value_heads");
  }
  if (c.head_dim % 64 != 0 || c.head_dim > 256) {
    return fail("head_dim must be a multiple of 64 up to 256");
  }
  if (c.n_heads / c.n_kv_heads > 32) {
    return fail("more than 32 query heads per kv head");
  }
  if (c.hidden % 64 != 0 || c.ffn % 64 != 0) {
    return fail("hidden / intermediate size not a multiple of 64");
  }
  if (c.block_size < 2) {
    return fail("block_size must be at least 2");
  }
  return c;
}

bool
is_dflash_drafter_dir(const std::string& dir)
{
  return !dir.empty() && DFlashConfig::from_dir(dir).has_value();
}

// ---------------------------------------------------------------------
// Implementation state
// ---------------------------------------------------------------------

namespace {

// y[M, N] = x[M, K] @ W^T. bits 0: W dense in the compute dtype; 4 / 8:
// affine group-64 (codes, scales, biases -- subviews of one buffer).
struct Lin {
  int N = 0, K = 0, bits = 0;
  SharedBuffer w, s, b;
  std::size_t bytes = 0;
};

struct DLayer {
  SharedBuffer in_ln, post_ln, q_norm, k_norm;
  Lin qkv;            // fused q|k|v [qd + 2 kvd, H]
  Lin o;              // [H, qd]
  Lin gate, up;       // [ffn, H]
  Lin down;           // [H, ffn]
  // DFlash 2: the convolutions' kernel projections + base kernels.
  Lin attn_kp, mlp_kp;           // [2 * ks * G, H]
  SharedBuffer attn_base, mlp_base;   // [2, ks, H]
  // This layer's context cache.
  int cap = 0;
  SharedBuffer kc, vc;           // [Hkv, cap, D]
  SharedBuffer tags;             // [cap] int32, -1 = empty
};

}  // namespace

struct MetalDFlashDrafter::Impl {
  std::shared_ptr<WeightSet> ws;
  MetalCompute* mc = nullptr;
  std::size_t esz = 2;

  std::vector<DLayer> layers;
  Lin fc;                                  // [H, n_taps * H]
  SharedBuffer hidden_norm, norm;
  Lin hp;                                  // selector hidden_projection
  SharedBuffer pred_cb, succ_cb;           // [V, rank]
  SharedBuffer inv_freq;                   // [D / 2] f32
  std::size_t weight_bytes = 0;

  // The drafter's own kernels are bf16; the ones that read the target's
  // logits (top-k, argmax, scale, softcap) take the target's dtype.
  ComputeLibrary lib_df, lib_rms, lib_elt, lib_qmv, lib_qmm, lib_dense,
      lib_df_t, lib_elt_t;
  ComputeFunction fn_conv, fn_qk_prep, fn_kv_store, fn_attn, fn_topk_part,
      fn_topk_merge, fn_select, fn_copy_rows, fn_argmax, fn_rms,
      fn_residual, fn_swiglu, fn_scale, fn_softcap;
  // Boundary converters, used only when the target is f16; and the
  // embedding scale, which acts on the drafter's own bf16 rows.
  ComputeFunction fn_h2b, fn_b2h, fn_copy_rows_h2b, fn_scale_d;
  bool target_bf16 = true;
  // Affine GEMM tiers for the held width, and the dense pair.
  ComputeFunction fn_qmv, fn_qmv_b2, fn_qmv_b4, fn_qmv_b8, fn_qmm;
  ComputeFunction fn_dgemm, fn_dgemv;

  // Scratch sized for Lmax block rows and kIngestChunk context rows.
  static constexpr int kIngestChunk = 256;
  static constexpr int kTopkParts = 64;
  int Lmax = 0;
  SharedBuffer hb, xn, xc, qkv, att, ao, ao2, dyn, g, u, hn, logits;
  SharedBuffer pval, pidx, cval, cidx, hpo;
  SharedBuffer cat, ctx, ctxn, kv;
  // f16-target boundary scratch: the embeddings as the target writes them,
  // the normed hidden as its head reads it.
  SharedBuffer emb_t, hn_t;
  std::size_t scratch_bytes = 0;
  SharedBuffer dbg;                         // set_debug_layers
  ComputeFunction fn_copy;

  void gemm(ComputeEncoder& enc, const Lin& W, const SharedBuffer& x,
            std::size_t xoff, const SharedBuffer& y, std::size_t yoff,
            int M, int row0 = 0, int rows = -1) const;
  void rms(ComputeEncoder& enc, const SharedBuffer& x, std::size_t xoff,
           const SharedBuffer& w, const SharedBuffer& y, std::size_t yoff,
           int R, int H, float eps) const;
  void residual(ComputeEncoder& enc, const SharedBuffer& a,
                const SharedBuffer& b, int n) const;
  void conv(ComputeEncoder& enc, const SharedBuffer& x,
            const SharedBuffer& base, std::size_t base_off,
            const SharedBuffer& out, int C, int L, int ks, int gsz,
            int dyn_stride, int dyn_off) const;
  void qk_prep(ComputeEncoder& enc, const SharedBuffer& x, std::size_t xoff,
               const SharedBuffer& w, int T, int H, int D, float eps,
               int pos0, int ld) const;
};

void
MetalDFlashDrafter::Impl::gemm(ComputeEncoder& enc, const Lin& W,
                               const SharedBuffer& x, std::size_t xoff,
                               const SharedBuffer& y, std::size_t yoff,
                               int M, int row0, int rows) const
{
  const int N = rows < 0 ? W.N - row0 : rows;
  const int K = W.K;
  if (W.bits == 0) {
    const std::size_t woff = (std::size_t)row0 * K * esz;
    if (M == 1) {
      enc.set_function(fn_dgemv);
      enc.set_buffer(0, x, xoff);
      enc.set_buffer(1, W.w, woff);
      enc.set_buffer(2, y, yoff);
      enc.set_constant(3, K);
      enc.set_constant(4, N);
      enc.dispatch({32, (unsigned)(((N + 7) / 8) * 2), 1}, {32, 2, 1});
      return;
    }
    enc.set_function(fn_dgemm);
    enc.set_buffer(0, x, xoff);
    enc.set_buffer(1, W.w, woff);
    enc.set_buffer(2, W.w, woff);   // bias slot unused (has_bias = 0)
    enc.set_buffer(3, y, yoff);
    enc.set_constant(4, K);
    enc.set_constant(5, N);
    enc.set_constant(6, M);
    enc.set_constant(7, 0);
    enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                  (unsigned)(((M + 31) / 32) * 2), 2}, {32, 2, 2});
    return;
  }
  const std::size_t woff = (std::size_t)row0 * K * W.bits / 8;
  const std::size_t soff = (std::size_t)row0 * (K / 64) * esz;
  auto bind = [&](const ComputeFunction& fn) {
    enc.set_function(fn);
    enc.set_buffer(0, W.w, woff);
    enc.set_buffer(1, W.s, soff);
    enc.set_buffer(2, W.b, soff);
    enc.set_buffer(3, x, xoff);
    enc.set_buffer(4, y, yoff);
    enc.set_constant(5, K);
    enc.set_constant(6, N);
  };
  if (M == 1) {
    bind(fn_qmv);
    enc.dispatch({32, (unsigned)(N / 4), 1}, {32, 2, 1});
    return;
  }
  // Rows up to which the batched GEMV runs rather than the steel GEMM.
  // MEASURED on an M4 Pro the two tie at a draft's 8 rows (both run out of
  // ALU, not bandwidth); VPIPE_DFLASH_GEMV_MAX moves the cut for an A/B on
  // a GPU where they may not (1 = steel from 2 rows).
  static const int gemv_max = [] {
    const char* e = std::getenv("VPIPE_DFLASH_GEMV_MAX");
    return e != nullptr ? std::atoi(e) : 8;
  }();
  if (M <= gemv_max) {
    // Batched GEMV: the weight read once for every row of the tile.
    const ComputeFunction& fn =
        M >= 5 ? fn_qmv_b8 : (M >= 3 ? fn_qmv_b4 : fn_qmv_b2);
    const int maxm = M >= 5 ? 8 : (M >= 3 ? 4 : 2);
    bind(fn);
    enc.set_constant(7, M);
    enc.dispatch({32, (unsigned)(N / 4), (unsigned)((M + maxm - 1) / maxm)},
                 {32, 2, 1});
    return;
  }
  bind(fn_qmm);
  enc.set_constant(7, M);
  enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                (unsigned)(((M + 31) / 32) * 2), 2}, {32, 2, 2});
}

void
MetalDFlashDrafter::Impl::rms(ComputeEncoder& enc, const SharedBuffer& x,
                              std::size_t xoff, const SharedBuffer& w,
                              const SharedBuffer& y, std::size_t yoff, int R,
                              int H, float eps) const
{
  enc.set_function(fn_rms);
  enc.set_buffer(0, x, xoff);
  enc.set_buffer(1, w);
  enc.set_buffer(2, y, yoff);
  enc.set_constant(3, H);
  enc.set_constant(4, eps);
  enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
}

void
MetalDFlashDrafter::Impl::residual(ComputeEncoder& enc, const SharedBuffer& a,
                                   const SharedBuffer& b, int n) const
{
  enc.set_function(fn_residual);
  enc.set_buffer(0, a);
  enc.set_buffer(1, b);
  enc.set_buffer(2, a);
  enc.set_constant(3, n);
  enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
}

void
MetalDFlashDrafter::Impl::conv(ComputeEncoder& enc, const SharedBuffer& x,
                               const SharedBuffer& base, std::size_t base_off,
                               const SharedBuffer& out, int C, int L, int ks,
                               int gsz, int dyn_stride, int dyn_off) const
{
  enc.set_function(fn_conv);
  enc.set_buffer(0, x);
  enc.set_buffer(1, dyn);
  enc.set_buffer(2, base, base_off);
  enc.set_buffer(3, out);
  enc.set_constant(4, C);
  enc.set_constant(5, L);
  enc.set_constant(6, ks);
  enc.set_constant(7, gsz);
  enc.set_constant(8, dyn_stride);
  enc.set_constant(9, dyn_off);
  enc.dispatch({(unsigned)C, (unsigned)L, 1}, {256, 1, 1});
}

void
MetalDFlashDrafter::Impl::qk_prep(ComputeEncoder& enc, const SharedBuffer& x,
                                  std::size_t xoff, const SharedBuffer& w,
                                  int T, int H, int D, float eps, int pos0,
                                  int ld) const
{
  enc.set_function(fn_qk_prep);
  enc.set_buffer(0, x, xoff);
  enc.set_buffer(1, w);
  enc.set_buffer(2, inv_freq);
  enc.set_constant(3, H);
  enc.set_constant(4, D);
  enc.set_constant(5, eps);
  enc.set_constant(6, pos0);
  enc.set_constant(7, ld);
  enc.dispatch({32, (unsigned)(T * H), 1}, {32, 1, 1});
}

// ---------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------

MetalDFlashDrafter::~MetalDFlashDrafter() = default;

const std::string&
MetalDFlashDrafter::dir() const noexcept
{
  return _impl->ws->dir();
}

std::unique_ptr<MetalDFlashDrafter>
MetalDFlashDrafter::load(std::shared_ptr<WeightSet> ws, MetalCompute* mc,
                         const Options& opt, std::string* err)
{
  auto fail = [&](const std::string& why)
      -> std::unique_ptr<MetalDFlashDrafter> {
    if (err) { *err = why; }
    return nullptr;
  };
  if (!ws || mc == nullptr) { return fail("no weight set / device"); }
  if (opt.quant_bits != 0 && opt.quant_bits != 4 && opt.quant_bits != 8) {
    return fail("draft quant_bits must be 0, 4 or 8");
  }
  std::string cerr;
  std::optional<DFlashConfig> cfg = DFlashConfig::from_dir(ws->dir(), &cerr);
  if (!cfg) { return fail(cerr); }

  std::unique_ptr<MetalDFlashDrafter> d(new MetalDFlashDrafter());
  d->_cfg = *cfg;
  d->_opt = opt;
  d->_impl = std::make_unique<Impl>();
  Impl& m = *d->_impl;
  const DFlashConfig& c = d->_cfg;
  m.ws = ws;
  m.mc = mc;
  // The drafter's own storage and arithmetic: bf16, always.
  const bool bf16 = true;
  const std::string sfx = "_bf16";
  const std::string tsfx = opt.target_bf16 ? "_bf16" : "";
  m.target_bf16 = opt.target_bf16;

  // ---- kernels ----
  m.lib_df = mc->load_library("dflash" + sfx);
  m.lib_rms = mc->load_library("rms_norm" + sfx);
  m.lib_elt = mc->load_library("llm_elementwise" + sfx);
  m.lib_df_t = mc->load_library("dflash" + tsfx);
  m.lib_elt_t = mc->load_library("llm_elementwise" + tsfx);
  m.fn_conv = m.lib_df.function("dflash_conv_f16");
  m.fn_qk_prep = m.lib_df.function("dflash_qk_prep_f16");
  m.fn_kv_store = m.lib_df.function("dflash_kv_store_f16");
  m.fn_attn = m.lib_df.function("dflash_attn_f16");
  m.fn_topk_part = m.lib_df_t.function("dflash_topk_part_f16");
  m.fn_topk_merge = m.lib_df_t.function("dflash_topk_merge_f16");
  m.fn_select = m.lib_df.function("dflash_select_f16");
  m.fn_copy_rows = m.lib_df.function("dflash_copy_rows_f16");
  m.fn_argmax = m.lib_df_t.function("dflash_argmax_f16");
  m.fn_rms = m.lib_rms.function("rms_norm_fast_f16");
  m.fn_residual = m.lib_elt.function("residual_add_f16");
  m.fn_swiglu = m.lib_elt.function("swiglu_f16");
  m.fn_scale = m.lib_elt_t.function("scale_inplace_f16");
  m.fn_softcap = m.lib_elt_t.function("softcap_f16");
  m.fn_h2b = m.lib_df.function("dflash_h2b");
  m.fn_b2h = m.lib_df.function("dflash_b2h");
  m.fn_copy_rows_h2b = m.lib_df.function("dflash_copy_rows_h2b");
  m.fn_scale_d = m.lib_elt.function("scale_inplace_f16");
  for (const ComputeFunction* f :
       {&m.fn_conv, &m.fn_qk_prep, &m.fn_kv_store, &m.fn_attn,
        &m.fn_topk_part, &m.fn_topk_merge, &m.fn_select, &m.fn_copy_rows,
        &m.fn_argmax, &m.fn_rms, &m.fn_residual, &m.fn_swiglu, &m.fn_scale,
        &m.fn_softcap, &m.fn_h2b, &m.fn_b2h, &m.fn_copy_rows_h2b,
        &m.fn_scale_d}) {
    if (!f->valid()) { return fail("a DFlash kernel failed to load"); }
  }
  if (opt.quant_bits != 0) {
    const std::string g = opt.quant_bits == 8 ? "w8g64" : "w4g64";
    m.lib_qmv = mc->load_library("affine_qmv" + sfx);
    m.lib_qmm = mc->load_library("affine_qmm_steel" + sfx);
    m.fn_qmv = m.lib_qmv.function("affine_qmv_" + g);
    m.fn_qmv_b2 = m.lib_qmv.function("affine_qmv_batch_" + g);
    m.fn_qmv_b4 = m.lib_qmv.function("affine_qmv_batch4_xp_" + g);
    m.fn_qmv_b8 = m.lib_qmv.function("affine_qmv_batch8_xp2_" + g);
    m.fn_qmm = m.lib_qmm.function("affine_qmm_steel_" + g);
    if (!m.fn_qmv.valid() || !m.fn_qmv_b2.valid() || !m.fn_qmv_b4.valid() ||
        !m.fn_qmv_b8.valid() || !m.fn_qmm.valid()) {
      return fail("the affine " + g + " GEMM kernels failed to load");
    }
  } else {
    m.lib_dense = mc->load_library("dense_gemm" + sfx);
    m.fn_dgemm = m.lib_dense.function("dense_gemm_t_f16");
    m.fn_dgemv = m.lib_dense.function("dense_gemv_t_f16");
    if (!m.fn_dgemm.valid() || !m.fn_dgemv.valid()) {
      return fail("the dense GEMM kernels failed to load");
    }
  }
  m.esz = 2;

  // ---- weights ----
  const MetalLlamaWeights& src = ws->src();
  std::string pfx;
  if (!ws->has("fc.weight") && ws->has("model.fc.weight")) { pfx = "model."; }
  if (!ws->has(pfx + "fc.weight")) {
    return fail("drafter checkpoint has no fc.weight");
  }
  const std::string dtname = bf16 ? "bf16" : "f16";
  bool ok = true;
  std::string why;

  // A vector the model keeps as-is, in the compute dtype.
  auto vec = [&](const std::string& name) -> SharedBuffer {
    const auto* ti = src.info(name);
    if (ti == nullptr || src_elt_bytes_(ti->dtype) == 0) {
      ok = false;
      why = "missing or unsupported tensor '" + name + "'";
      return {};
    }
    const std::string key = "dflash/" + dtname + "/" + name;
    return ws->derived(key, [&]() -> SharedBuffer {
      SharedBuffer raw = ws->read(name, mc, WeightSet::Residency::Copied);
      if (raw.empty()) { return {}; }
      const std::size_t n = raw.byte_size() / (std::size_t)
          src_elt_bytes_(ti->dtype);
      SharedBuffer out = mc->make_shared_buffer(n * 2);
      if (out.empty()) { return {}; }
      convert_(raw.contents(), ti->dtype,
               static_cast<std::uint16_t*>(out.contents()), n, bf16);
      return out;
    });
  };

  // A linear [sum(rows of `names`), K], the parts stacked by rows. Dense:
  // the compute dtype; quantized: affine group-64 at opt.quant_bits,
  // packed codes | scales | biases in one buffer.
  auto lin = [&](const std::vector<std::string>& names, Lin& out) {
    if (!ok) { return; }
    int N = 0, K = -1;
    std::string dt;
    for (const std::string& nm : names) {
      const auto* ti = src.info(nm);
      if (ti == nullptr || ti->shape.size() != 2 ||
          src_elt_bytes_(ti->dtype) == 0) {
        ok = false;
        why = "missing or unsupported tensor '" + nm + "'";
        return;
      }
      if (K >= 0 && ti->shape[1] != K) {
        ok = false;
        why = "fused parts disagree in width at '" + nm + "'";
        return;
      }
      if (!dt.empty() && ti->dtype != dt) {
        ok = false;
        why = "fused parts disagree in dtype at '" + nm + "'";
        return;
      }
      K = (int)ti->shape[1];
      N += (int)ti->shape[0];
      dt = ti->dtype;
    }
    out.N = N;
    out.K = K;
    out.bits = opt.quant_bits;
    std::string key = "dflash/" + dtname + "/w" +
                      std::to_string(opt.quant_bits) + "g64/";
    for (const std::string& nm : names) { key += nm + "+"; }
    const int bits = opt.quant_bits;
    if (bits != 0 && K % 64 != 0) {
      ok = false;
      why = "width of '" + names[0] + "' is not a multiple of 64";
      return;
    }
    if (bits != 0 && N % 4 != 0) {
      ok = false;
      why = "rows of '" + names[0] + "' are not a multiple of 4";
      return;
    }
    const std::size_t wbytes = (std::size_t)N * K * (bits == 0 ? 2 : bits) /
                               (bits == 0 ? 1 : 8);
    const std::size_t sbytes = bits == 0 ? 0 : (std::size_t)N * (K / 64) * 2;
    const std::size_t soff = align_up_(wbytes, 256);
    const std::size_t boff = align_up_(soff + sbytes, 256);
    const std::size_t total = bits == 0 ? wbytes : boff + sbytes;
    SharedBuffer packed = ws->derived(key, [&]() -> SharedBuffer {
      SharedBuffer outb = mc->make_shared_buffer(total);
      if (outb.empty()) { return {}; }
      if (bits == 0) {
        std::size_t row = 0;
        for (const std::string& nm : names) {
          SharedBuffer raw = ws->read(nm, mc, WeightSet::Residency::Copied);
          if (raw.empty()) { return {}; }
          const std::size_t n = raw.byte_size() /
                                (std::size_t)src_elt_bytes_(dt);
          convert_(raw.contents(), dt,
                   static_cast<std::uint16_t*>(outb.contents()) + row * K,
                   n, bf16);
          row += n / (std::size_t)K;
        }
        return outb;
      }
      // Quantize on the GPU in the SOURCE's 16-bit type (no conversion of
      // the big tensor); scales / biases come out in that type and are
      // narrowed to the compute dtype afterwards if they differ.
      const bool src_bf16 = (dt != "F16");
      ComputeLibrary ql =
          mc->load_library(src_bf16 ? "affine_dequant_bf16" : "affine_dequant");
      ComputeFunction qf = ql.function(bits == 8 ? "affine_quant_rows_w8g64"
                                                 : "affine_quant_rows_w4g64");
      if (!qf.valid()) { return {}; }
      SharedBuffer sb = outb.subview(soff, sbytes);
      SharedBuffer bb = outb.subview(boff, sbytes);
      int row = 0;
      for (const std::string& nm : names) {
        SharedBuffer raw = ws->read(nm, mc, WeightSet::Residency::Copied);
        if (raw.empty()) { return {}; }
        const auto* ti = src.info(nm);
        const int rows = (int)ti->shape[0];
        if (dt == "F32") {
          // Narrow to bf16 first: the quantizer reads a 16-bit source.
          const std::size_t n = (std::size_t)rows * K;
          SharedBuffer nb = mc->make_shared_buffer(n * 2);
          if (nb.empty()) { return {}; }
          convert_(raw.contents(), dt,
                   static_cast<std::uint16_t*>(nb.contents()), n, true);
          raw = std::move(nb);
        }
        metal_compute::CommandStream qs = mc->make_command_stream();
        {
          ComputeEncoder enc = qs.begin_compute();
          enc.set_function(qf);
          enc.set_buffer(0, raw);
          enc.set_buffer(1, outb);
          enc.set_buffer(2, sb);
          enc.set_buffer(3, bb);
          enc.set_constant(4, K);
          enc.set_constant(5, rows);
          const int one = 1;
          enc.set_constant(6, one);
          enc.set_constant(7, row);
          enc.dispatch({(unsigned)(K / 64), (unsigned)rows, 1}, {32, 1, 1});
        }
        qs.commit().wait();
        row += rows;
      }
      if (src_bf16 != bf16) {
        // Narrow the scales / biases to the compute dtype in place.
        for (std::size_t off : {soff, boff}) {
          auto* p = reinterpret_cast<std::uint16_t*>(
              static_cast<char*>(outb.contents()) + off);
          const std::size_t n = sbytes / 2;
          if (src_bf16) {
            for (std::size_t i = 0; i < n; ++i) {
              const _Float16 h = (_Float16)bf16_to_f32_(p[i]);
              std::memcpy(&p[i], &h, 2);
            }
          } else {
            for (std::size_t i = 0; i < n; ++i) {
              _Float16 h;
              std::memcpy(&h, &p[i], 2);
              p[i] = f32_to_bf16_((float)h);
            }
          }
        }
      }
      return outb;
    });
    if (packed.empty()) {
      ok = false;
      why = "failed to build '" + names[0] + "'";
      return;
    }
    out.bytes = packed.byte_size();
    if (bits == 0) {
      out.w = std::move(packed);
    } else {
      out.w = packed.subview(0, wbytes);
      out.s = packed.subview(soff, sbytes);
      out.b = packed.subview(boff, sbytes);
    }
    m.weight_bytes += out.bytes;
  };
  auto keepvec = [&](const std::string& name) -> SharedBuffer {
    SharedBuffer v = vec(name);
    m.weight_bytes += v.byte_size();
    return v;
  };

  lin({pfx + "fc.weight"}, m.fc);
  if (ok && m.fc.K != (int)c.target_layer_ids.size() * c.hidden) {
    return fail("fc.weight width is not n_taps x hidden_size");
  }
  m.hidden_norm = keepvec(pfx + "hidden_norm.weight");
  m.norm = keepvec(pfx + "norm.weight");
  m.layers.resize((std::size_t)c.n_layers);
  for (int L = 0; L < c.n_layers && ok; ++L) {
    const std::string p = pfx + "layers." + std::to_string(L) + ".";
    DLayer& ly = m.layers[(std::size_t)L];
    ly.in_ln = keepvec(p + "input_layernorm.weight");
    ly.post_ln = keepvec(p + "post_attention_layernorm.weight");
    ly.q_norm = keepvec(p + "self_attn.q_norm.weight");
    ly.k_norm = keepvec(p + "self_attn.k_norm.weight");
    lin({p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight",
         p + "self_attn.v_proj.weight"}, ly.qkv);
    lin({p + "self_attn.o_proj.weight"}, ly.o);
    lin({p + "mlp.gate_proj.weight"}, ly.gate);
    lin({p + "mlp.up_proj.weight"}, ly.up);
    lin({p + "mlp.down_proj.weight"}, ly.down);
    if (c.v2()) {
      lin({p + "attention_conv.kernel_projection.weight"}, ly.attn_kp);
      lin({p + "mlp_conv.kernel_projection.weight"}, ly.mlp_kp);
      ly.attn_base = keepvec(p + "attention_conv.base_kernel");
      ly.mlp_base = keepvec(p + "mlp_conv.base_kernel");
    }
    if (ok && ly.qkv.N != c.qd() + 2 * c.kvd()) {
      return fail("q/k/v projections do not match the config's heads");
    }
  }
  if (ok && c.v2()) {
    lin({pfx + "candidate_selector.hidden_projection.weight"}, m.hp);
    m.pred_cb = keepvec(pfx + "candidate_selector.predecessor_codebook");
    m.succ_cb = keepvec(pfx + "candidate_selector.successor_codebook");
    if (ok && (m.hp.N != c.selector_rank ||
               m.pred_cb.byte_size() !=
                   (std::size_t)c.vocab * c.selector_rank * 2)) {
      return fail("candidate selector shapes do not match the config");
    }
  }
  if (!ok) { return fail(why); }

  // RoPE inverse frequencies over the full head, as HF's default rope.
  m.inv_freq = mc->make_shared_buffer((std::size_t)c.head_dim / 2 * 4);
  {
    auto* f = static_cast<float*>(m.inv_freq.contents());
    for (int i = 0; i < c.head_dim / 2; ++i) {
      const float e = (float)(2 * i) / (float)c.head_dim;
      f[i] = 1.0f / std::pow(c.rope_theta, e);
    }
  }

  // ---- context caches ----
  for (int L = 0; L < c.n_layers; ++L) {
    DLayer& ly = m.layers[(std::size_t)L];
    const int w = c.layer_window[(std::size_t)L];
    ly.cap = w > 0 ? w : std::max(opt.max_context, 1024);
    const std::size_t kvb =
        (std::size_t)c.n_kv_heads * ly.cap * c.head_dim * 2;
    ly.kc = mc->make_shared_buffer(kvb);
    ly.vc = mc->make_shared_buffer(kvb);
    ly.tags = mc->make_shared_buffer((std::size_t)ly.cap * 4);
    if (ly.kc.empty() || ly.vc.empty() || ly.tags.empty()) {
      return fail("failed to allocate the context cache");
    }
    m.scratch_bytes += 2 * kvb + (std::size_t)ly.cap * 4;
  }

  // ---- scratch ----
  m.Lmax = std::max(c.block_size, 16);
  const int Lx = m.Lmax, H = c.hidden, V = c.vocab;
  const int nT = (int)c.target_layer_ids.size();
  const int ks = std::max(c.conv_kernel, 1);
  const int G = c.conv_group > 0 ? H / c.conv_group : 1;
  auto buf = [&](std::size_t bytes) {
    SharedBuffer b = mc->make_shared_buffer(std::max<std::size_t>(bytes, 4));
    m.scratch_bytes += b.byte_size();
    return b;
  };
  m.hb = buf((std::size_t)Lx * H * 2);
  m.xn = buf((std::size_t)Lx * H * 2);
  m.xc = buf((std::size_t)Lx * H * 2);
  m.qkv = buf((std::size_t)Lx * (c.qd() + 2 * c.kvd()) * 2);
  m.att = buf((std::size_t)Lx * c.qd() * 2);
  m.ao = buf((std::size_t)Lx * H * 2);
  m.ao2 = buf((std::size_t)Lx * H * 2);
  m.dyn = buf((std::size_t)Lx * 2 * ks * G * 2);
  m.g = buf((std::size_t)Lx * c.ffn * 2);
  m.u = buf((std::size_t)Lx * c.ffn * 2);
  m.hn = buf((std::size_t)Lx * H * 2);
  m.logits = buf((std::size_t)Lx * V * 2);
  m.pval = buf((std::size_t)Lx * Impl::kTopkParts * 16 * 4);
  m.pidx = buf((std::size_t)Lx * Impl::kTopkParts * 16 * 4);
  m.cval = buf((std::size_t)Lx * 16 * 4);
  m.cidx = buf((std::size_t)Lx * 16 * 4);
  m.hpo = buf((std::size_t)Lx * std::max(c.selector_rank, 1) * 2);
  const int ch = Impl::kIngestChunk;
  m.cat = buf((std::size_t)ch * nT * H * 2);
  m.ctx = buf((std::size_t)ch * H * 2);
  m.ctxn = buf((std::size_t)ch * H * 2);
  m.kv = buf((std::size_t)ch * 2 * c.kvd() * 2);
  if (!opt.target_bf16) {
    m.emb_t = buf((std::size_t)Lx * H * 2);
    m.hn_t = buf((std::size_t)Lx * H * 2);
    if (m.emb_t.empty() || m.hn_t.empty()) {
      return fail("failed to allocate drafter scratch");
    }
  }
  for (const SharedBuffer* b :
       {&m.hb, &m.xn, &m.xc, &m.qkv, &m.att, &m.ao, &m.ao2, &m.dyn, &m.g,
        &m.u, &m.hn, &m.logits, &m.pval, &m.pidx, &m.cval, &m.cidx, &m.hpo,
        &m.cat, &m.ctx, &m.ctxn, &m.kv}) {
    if (b->empty()) { return fail("failed to allocate drafter scratch"); }
  }
  d->reset();
  return d;
}

std::size_t
MetalDFlashDrafter::resident_bytes() const noexcept
{
  return _impl ? _impl->weight_bytes + _impl->scratch_bytes : 0;
}

void
MetalDFlashDrafter::reset()
{
  for (DLayer& ly : _impl->layers) {
    if (!ly.tags.empty()) {
      std::memset(ly.tags.contents(), 0xff, ly.tags.byte_size());
    }
  }
}

void
MetalDFlashDrafter::fill_block_ids(std::int32_t* dst, std::int32_t anchor,
                                   int L) const
{
  dst[0] = anchor;
  for (int i = 1; i < L; ++i) { dst[i] = _cfg.mask_token_id; }
}

const SharedBuffer&
MetalDFlashDrafter::last_hidden() const noexcept
{
  return _impl->hn;
}

const SharedBuffer&
MetalDFlashDrafter::last_cand_val() const noexcept
{
  return _impl->cval;
}

const SharedBuffer&
MetalDFlashDrafter::last_cand_idx() const noexcept
{
  return _impl->cidx;
}

void
MetalDFlashDrafter::set_debug_layers(bool on)
{
  Impl& m = *_impl;
  if (!on) {
    m.dbg = SharedBuffer{};
    return;
  }
  if (!m.fn_copy.valid()) { m.fn_copy = m.lib_elt.function("copy_f16"); }
  m.dbg = m.mc->make_shared_buffer((std::size_t)_cfg.n_layers * m.Lmax *
                                   _cfg.hidden * 2);
}

const SharedBuffer&
MetalDFlashDrafter::debug_layers() const noexcept
{
  return _impl->dbg;
}

// ---------------------------------------------------------------------
// Context ingest
// ---------------------------------------------------------------------

void
MetalDFlashDrafter::encode_ingest(ComputeEncoder& enc,
                                  const SharedBuffer& taps, int tap_rows,
                                  int row0, int rows, int pos0)
{
  Impl& m = *_impl;
  const DFlashConfig& c = _cfg;
  const int H = c.hidden, D = c.head_dim, Hkv = c.n_kv_heads;
  const int kvd = c.kvd(), qd = c.qd();
  const int nT = (int)c.target_layer_ids.size();
  const std::size_t es = m.esz;
  for (int r = 0; r < rows; r += Impl::kIngestChunk) {
    const int mrows = std::min(Impl::kIngestChunk, rows - r);
    // Concatenate the tapped layers' rows: cat[:, j*H .. (j+1)*H).
    for (int j = 0; j < nT; ++j) {
      enc.set_function(m.target_bf16 ? m.fn_copy_rows : m.fn_copy_rows_h2b);
      enc.set_buffer(0, taps,
                     ((std::size_t)j * tap_rows + row0 + r) * H * es);
      enc.set_buffer(1, m.cat, (std::size_t)j * H * es);
      enc.set_constant(2, H);
      enc.set_constant(3, H);
      const int dld = nT * H;
      enc.set_constant(4, dld);
      enc.dispatch({(unsigned)H, (unsigned)mrows, 1}, {256, 1, 1});
    }
    m.gemm(enc, m.fc, m.cat, 0, m.ctx, 0, mrows);
    m.rms(enc, m.ctx, 0, m.hidden_norm, m.ctxn, 0, mrows, H, c.rms_eps);
    const int p = pos0 + r;
    for (int L = 0; L < c.n_layers; ++L) {
      DLayer& ly = m.layers[(std::size_t)L];
      // The k|v rows of the fused q|k|v projection.
      m.gemm(enc, ly.qkv, m.ctxn, 0, m.kv, 0, mrows, qd, 2 * kvd);
      m.qk_prep(enc, m.kv, 0, ly.k_norm, mrows, Hkv, D, c.rms_eps, p,
                2 * kvd);
      enc.set_function(m.fn_kv_store);
      enc.set_buffer(0, m.kv, 0);
      enc.set_buffer(1, m.kv, (std::size_t)kvd * es);
      enc.set_buffer(2, ly.kc);
      enc.set_buffer(3, ly.vc);
      enc.set_buffer(4, ly.tags);
      enc.set_constant(5, Hkv);
      enc.set_constant(6, D);
      enc.set_constant(7, ly.cap);
      enc.set_constant(8, p);
      const int ld = 2 * kvd;
      enc.set_constant(9, ld);
      enc.dispatch({(unsigned)D, (unsigned)mrows, (unsigned)Hkv},
                   {(unsigned)D, 1, 1});
    }
  }
}

// ---------------------------------------------------------------------
// Block draft
// ---------------------------------------------------------------------

void
MetalDFlashDrafter::encode_draft(ComputeEncoder& enc, DFlashTarget& target,
                                 const SharedBuffer& block_ids, int L,
                                 int pos0, const SharedBuffer& out_ids,
                                 int out_off)
{
  Impl& m = *_impl;
  const DFlashConfig& c = _cfg;
  L = std::min(L, m.Lmax);
  if (L < 2) { return; }
  const int H = c.hidden, D = c.head_dim, Hq = c.n_heads, Hkv = c.n_kv_heads;
  const int qd = c.qd(), kvd = c.kvd(), ldq = qd + 2 * kvd;
  const std::size_t es = m.esz;
  const float eps = c.rms_eps;
  const int ks = c.conv_kernel, gsz = c.conv_group;
  const int G = c.v2() ? H / gsz : 0;
  const int dstride = 2 * ks * G;

  if (m.target_bf16) {
    target.encode_embed(enc, block_ids, 0, m.hb, L);
  } else {
    target.encode_embed(enc, block_ids, 0, m.emb_t, L);
    enc.set_function(m.fn_h2b);
    enc.set_buffer(0, m.emb_t);
    enc.set_buffer(1, m.hb);
    const int ne = L * H;
    enc.set_constant(2, ne);
    enc.dispatch({(unsigned)ne, 1, 1}, {256, 1, 1});
  }
  if (c.input_embedding_scale != 1.0f) {
    enc.set_function(m.fn_scale_d);
    enc.set_buffer(0, m.hb);
    const int n = L * H;
    enc.set_constant(1, n);
    enc.set_constant(2, c.input_embedding_scale);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
  }
  const float scale = 1.0f / std::sqrt((float)D);
  for (int li = 0; li < c.n_layers; ++li) {
    DLayer& ly = m.layers[(std::size_t)li];
    // ---- attention sublayer ----
    m.rms(enc, m.hb, 0, ly.in_ln, m.xn, 0, L, H, eps);
    const SharedBuffer* a_in = &m.xn;
    if (c.v2()) {
      m.gemm(enc, ly.attn_kp, m.xn, 0, m.dyn, 0, L);
      m.conv(enc, m.xn, ly.attn_base, 0, m.xc, H, L, ks, gsz, dstride, 0);
      a_in = &m.xc;
    }
    m.gemm(enc, ly.qkv, *a_in, 0, m.qkv, 0, L);
    m.qk_prep(enc, m.qkv, 0, ly.q_norm, L, Hq, D, eps, pos0, ldq);
    m.qk_prep(enc, m.qkv, (std::size_t)qd * es, ly.k_norm, L, Hkv, D, eps,
              pos0, ldq);
    enc.set_function(m.fn_attn);
    enc.set_buffer(0, m.qkv, 0);
    enc.set_buffer(1, m.qkv, (std::size_t)qd * es);
    enc.set_buffer(2, m.qkv, (std::size_t)(qd + kvd) * es);
    enc.set_buffer(3, ly.kc);
    enc.set_buffer(4, ly.vc);
    enc.set_buffer(5, ly.tags);
    enc.set_buffer(6, m.att);
    enc.set_constant(7, L);
    enc.set_constant(8, Hq);
    enc.set_constant(9, Hkv);
    enc.set_constant(10, D);
    enc.set_constant(11, ly.cap);
    enc.set_constant(12, pos0);
    const int window = c.layer_window[(std::size_t)li];
    enc.set_constant(13, window);
    const int causal = c.layer_causal[(std::size_t)li] ? 1 : 0;
    enc.set_constant(14, causal);
    enc.set_constant(15, scale);
    enc.set_constant(16, ldq);
    enc.dispatch({(unsigned)(32 * L), (unsigned)Hq, 1},
                 {32, (unsigned)(Hq / Hkv), 1});
    m.gemm(enc, ly.o, m.att, 0, m.ao, 0, L);
    if (c.v2()) {
      m.conv(enc, m.ao, ly.attn_base, (std::size_t)ks * H * es, m.ao2, H, L,
             ks, gsz, dstride, ks * G);
      m.residual(enc, m.hb, m.ao2, L * H);
    } else {
      m.residual(enc, m.hb, m.ao, L * H);
    }
    // ---- MLP sublayer ----
    m.rms(enc, m.hb, 0, ly.post_ln, m.xn, 0, L, H, eps);
    const SharedBuffer* m_in = &m.xn;
    if (c.v2()) {
      m.gemm(enc, ly.mlp_kp, m.xn, 0, m.dyn, 0, L);
      m.conv(enc, m.xn, ly.mlp_base, 0, m.xc, H, L, ks, gsz, dstride, 0);
      m_in = &m.xc;
    }
    m.gemm(enc, ly.gate, *m_in, 0, m.g, 0, L);
    m.gemm(enc, ly.up, *m_in, 0, m.u, 0, L);
    enc.set_function(m.fn_swiglu);
    enc.set_buffer(0, m.g);
    enc.set_buffer(1, m.u);
    enc.set_buffer(2, m.g);
    const int nf = L * c.ffn;
    enc.set_constant(3, nf);
    enc.dispatch({(unsigned)nf, 1, 1}, {256, 1, 1});
    m.gemm(enc, ly.down, m.g, 0, m.ao, 0, L);
    if (c.v2()) {
      m.conv(enc, m.ao, ly.mlp_base, (std::size_t)ks * H * es, m.ao2, H, L,
             ks, gsz, dstride, ks * G);
      m.residual(enc, m.hb, m.ao2, L * H);
    } else {
      m.residual(enc, m.hb, m.ao, L * H);
    }
    if (!m.dbg.empty() && m.fn_copy.valid()) {
      enc.set_function(m.fn_copy);
      enc.set_buffer(0, m.hb);
      enc.set_buffer(1, m.dbg, (std::size_t)li * L * H * es);
      const int zero = 0, cnt = L * H;
      enc.set_constant(2, zero);
      enc.set_constant(3, cnt);
      enc.dispatch({(unsigned)cnt, 1, 1}, {256, 1, 1});
    }
  }
  m.rms(enc, m.hb, 0, m.norm, m.hn, 0, L, H, eps);

  // ---- drafts from block rows 1 .. L-1 ----
  const int R = L - 1;
  const int V = c.vocab;
  const std::size_t row1 = (std::size_t)H * es;
  if (m.target_bf16) {
    target.encode_head(enc, m.hn, row1, m.logits, R);
  } else {
    enc.set_function(m.fn_b2h);
    enc.set_buffer(0, m.hn, row1);
    enc.set_buffer(1, m.hn_t);
    const int nh = R * H;
    enc.set_constant(2, nh);
    enc.dispatch({(unsigned)nh, 1, 1}, {256, 1, 1});
    target.encode_head(enc, m.hn_t, 0, m.logits, R);
  }
  if (c.output_multiplier != 1.0f) {
    enc.set_function(m.fn_scale);
    enc.set_buffer(0, m.logits);
    const int n = R * V;
    enc.set_constant(1, n);
    enc.set_constant(2, c.output_multiplier);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
  }
  if (c.final_logit_softcap > 0.0f) {
    enc.set_function(m.fn_softcap);
    enc.set_buffer(0, m.logits);
    enc.set_buffer(1, m.logits);
    const int n = R * V;
    enc.set_constant(2, n);
    enc.set_constant(3, c.final_logit_softcap);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
  }
  if (!c.v2()) {
    enc.set_function(m.fn_argmax);
    enc.set_buffer(0, m.logits);
    enc.set_buffer(1, out_ids);
    enc.set_constant(2, V);
    enc.set_constant(3, out_off);
    enc.dispatch({(unsigned)(256 * R), 1, 1}, {256, 1, 1});
    return;
  }
  const int P = Impl::kTopkParts;
  const int chunk = (V + P - 1) / P;
  enc.set_function(m.fn_topk_part);
  enc.set_buffer(0, m.logits);
  enc.set_buffer(1, m.pval);
  enc.set_buffer(2, m.pidx);
  enc.set_constant(3, V);
  enc.set_constant(4, chunk);
  enc.set_constant(5, P);
  enc.dispatch({(unsigned)(256 * P), (unsigned)R, 1}, {256, 1, 1});
  enc.set_function(m.fn_topk_merge);
  enc.set_buffer(0, m.pval);
  enc.set_buffer(1, m.pidx);
  enc.set_buffer(2, m.cval);
  enc.set_buffer(3, m.cidx);
  enc.set_constant(4, P);
  enc.dispatch({(unsigned)(256 * R), 1, 1}, {256, 1, 1});
  m.gemm(enc, m.hp, m.hn, row1, m.hpo, 0, R);
  enc.set_function(m.fn_select);
  enc.set_buffer(0, m.cval);
  enc.set_buffer(1, m.cidx);
  enc.set_buffer(2, m.hpo);
  enc.set_buffer(3, m.pred_cb);
  enc.set_buffer(4, m.succ_cb);
  enc.set_buffer(5, block_ids, 0);
  enc.set_buffer(6, out_ids);
  enc.set_constant(7, R);
  enc.set_constant(8, c.selector_rank);
  enc.set_constant(9, out_off);
  enc.dispatch({(unsigned)c.selector_rank, 1, 1},
               {(unsigned)c.selector_rank, 1, 1});
}

}  // namespace vpipe::genai

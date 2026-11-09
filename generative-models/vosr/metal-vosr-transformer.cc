#include "generative-models/vosr/metal-vosr-transformer.h"

#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/shared/dit-gpu-progress.h"
#include "generative-models/weight-set.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <utility>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

using metal_compute::CommandStream;
using metal_compute::ComputeEncoder;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

// C++ mirror of mlx::steel::AttnParams -- the param block the vendored
// steel flash kernel reads. One copy per DiT in this tree, deliberately:
// the struct is the kernel's ABI.
struct SteelAttnParams {
  int B, H, D;
  int qL, kL;
  int gqa_factor;
  float scale;
  int NQ, NK;
  int NQ_aligned, NK_aligned;
  int qL_rem, kL_rem, qL_off;
  std::int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
};
// The flash tiles live on the class (attn_bq_ / attn_bk_): they follow
// the KERNEL ARM, not the head width, and both arms are reachable in one
// process through VPIPE_VOSR_NO_ATTN_NAX.

inline std::uint16_t
f32_to_bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

inline float
bf16_to_f32_(std::uint16_t h)
{
  const std::uint32_t u = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

SharedBuffer
to_bf16_(WeightSet& ws, MetalCompute* mc, const std::string& nm)
{
  const auto* info = ws.src().info(nm);
  if (info == nullptr || info->shape.empty()) { return {}; }
  std::size_t n = 1;
  for (auto d : info->shape) { n *= (std::size_t)d; }
  // Uncached: the fp32 source is consumed by the conversion and dropped.
  // The checkpoint ships fp32 at 5.6 GB, so caching the raw tensor would
  // double the peak for no gain.
  SharedBuffer raw = ws.read(nm, mc, WeightSet::Residency::Copied);
  if (raw.empty()) { return {}; }
  if (info->dtype == "BF16") { return raw; }
  SharedBuffer out = mc->make_shared_buffer(n * 2);
  if (out.empty()) { return {}; }
  auto* d = static_cast<std::uint16_t*>(out.contents());
  if (info->dtype == "F32") {
    const auto* s = static_cast<const float*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) { d[i] = f32_to_bf16_(s[i]); }
  } else if (info->dtype == "F16") {
    const auto* s = static_cast<const _Float16*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) { d[i] = f32_to_bf16_((float)s[i]); }
  } else {
    return {};
  }
  return out;
}

// The directory holding a VOSR `args.json`, searched from `dir`.
//
// Three places, because three different callers point at three
// different levels of one checkpoint. `dir` ITSELF is the natural one.
// A CHILD covers the layout model-fetch produces: the release publishes
// several models from one repo, so a fetched root holds `VOSR2/` beside
// `Qwen-Image-vae-2d/` and the config is a level down. A PARENT covers
// the weights directory, which is `checkpoints/` under the config.
//
// Children before parents on purpose: a fetched root has both a child
// that answers and no parent that does, and reversing the order would
// make a stray args.json somewhere above a checkout win over the real
// one inside it.
std::string
args_root_(const std::string& dir)
{
  namespace fs = std::filesystem;
  std::error_code ec;
  const fs::path base = fs::absolute(dir, ec);
  if (ec) { return {}; }
  if (fs::exists(base / "args.json", ec)) { return base.string(); }
  if (fs::is_directory(base, ec)) {
    std::vector<std::string> kids;
    for (const auto& e : fs::directory_iterator(base, ec)) {
      if (e.is_directory(ec) && fs::exists(e.path() / "args.json", ec)) {
        kids.push_back(e.path().string());
      }
    }
    // Deterministic when a repo publishes more than one: the caller
    // asked about a root, not about a variant, and picking by
    // directory-iteration order would answer differently per machine.
    if (!kids.empty()) {
      std::sort(kids.begin(), kids.end());
      return kids.front();
    }
  }
  fs::path cur = base;
  for (int i = 0; i < 5; ++i) {
    if (!cur.has_parent_path() || cur.parent_path() == cur) { break; }
    cur = cur.parent_path();
    if (fs::exists(cur / "args.json", ec)) { return cur.string(); }
  }
  return {};
}

std::string
blk_(int i, const char* rest)
{
  return "blocks." + std::to_string(i) + "." + rest;
}

// Starting offsets covering `length` with `tile`-wide windows, the
// reference's _make_tile_grid: a fixed stride, plus a final flush-right
// window when the stride leaves a remainder.
std::vector<int>
tile_grid_(int length, int tile, int overlap)
{
  std::vector<int> pos;
  if (length <= tile) { pos.push_back(0); return pos; }
  const int stride = std::max(tile - overlap, 1);
  for (int p = 0; p + tile <= length; p += stride) { pos.push_back(p); }
  if (pos.empty() || pos.back() + tile < length) {
    pos.push_back(length - tile);
  }
  return pos;
}

}  // namespace

MetalVosrTransformer::~MetalVosrTransformer() = default;

bool
MetalVosrTransformer::read_config(const std::string& dir, Config* cfg,
                                  std::string* why)
{
  namespace fs = std::filesystem;
  auto fail = [&](const char* m) {
    if (why != nullptr) { *why = m; }
    return false;
  };
  const fs::path found = args_root_(dir);
  if (found.empty()) { return fail("no args.json"); }
  std::ifstream in(found / "args.json");
  if (!in) { return fail("args.json unreadable"); }
  FlexData fd = FlexData::from_json(in);
  if (!fd.is_object()) { return fail("args.json is not an object"); }
  auto o = fd.as_object();
  auto geti = [&](const char* k, int d) {
    return o.contains(k) ? (int)o.at(k).as_real((double)d) : d;
  };
  auto getb = [&](const char* k, bool d) {
    return o.contains(k) ? o.at(k).as_bool(d) : d;
  };

  // The four ways a VOSR checkpoint can be one this class does not run.
  // Each of them LOADS if waved through and then computes nonsense, so
  // they are refusals rather than warnings.
  if (o.contains("ae_type")) {
    const std::string ae(o.at("ae_type").as_string("qwen"));
    if (ae != "qwen") { return fail("only the Qwen-Image VAE variant"); }
  }
  // THE ONE-STEP MARK MUST BE THERE. VOSR 2.0 says `distill_type:
  // onestep`; the release's other 1.4B checkpoints have the same block
  // stack and the same tensor shapes, so nothing else tells them apart.
  // VOSR_1.4B_ms (multi-step: 25 steps at guidance 2.0) carries no
  // distill_type at all -- read as absent = one-step, it loaded and ran
  // one unguided step -- and VOSR_1.4B_os says `rcgm`, VOSR 1's own
  // distillation, which conditions differently.
  {
    const std::string dt(o.contains("distill_type")
                             ? o.at("distill_type").as_string("")
                             : std::string_view{});
    if (dt.empty()) {
      return fail("no distill_type: a multi-step VOSR checkpoint (VOSR 1's "
                  "VOSR_*_ms); only VOSR 2.0's one-step restorer runs here");
    }
    if (dt == "rcgm") {
      return fail("distill_type rcgm: VOSR 1's one-step checkpoint "
                  "(VOSR_*_os), which this tree does not run; only VOSR "
                  "2.0's one-step restorer does");
    }
    if (dt != "onestep") {
      return fail("distill_type is not VOSR 2.0's (onestep)");
    }
  }
  if (getb("auxiliary_time_cond", false)) {
    return fail("auxiliary_time_cond is not implemented");
  }
  if (!getb("use_rope", true) || !getb("use_rmsnorm", true) ||
      !getb("use_swiglu", true) || !getb("use_qknorm", true)) {
    return fail("the block recipe differs from VOSR 2.0's");
  }
  if (getb("wo_shift", false)) { return fail("wo_shift is not implemented"); }

  cfg->dim      = geti("dim", cfg->dim);
  cfg->depth    = geti("depth", cfg->depth);
  cfg->n_heads  = geti("num_heads", cfg->n_heads);
  cfg->patch    = geti("patch_size", cfg->patch);
  cfg->enc_dim  = geti("enc_dim", cfg->enc_dim);
  cfg->dinov2_size = geti("dinov2_size", cfg->dinov2_size);
  if (cfg->n_heads > 0) { cfg->head_dim = cfg->dim / cfg->n_heads; }
  const int mlp_ratio = geti("mlp_ratio", 4);
  // SwiGLUFFN(hidden, int(2/3 * mlp_ratio * hidden)) -- the 2/3 keeps a
  // gated MLP's parameter count level with an ungated one's.
  cfg->ffn = (int)((2.0 / 3.0) * (double)(mlp_ratio * cfg->dim));
  cfg->enc_mlp = cfg->dim * geti("encdim_ratio", 3);
  const int res = geti("resolution", 512);
  cfg->train_grid = res / 8 / (cfg->patch > 0 ? cfg->patch : 1);
  // layer_dinov2b_list is a one-element list in every shipped config;
  // the tower needs the layer index, the DiT does not.
  if (o.contains("layer_dinov2b_list")) {
    FlexData lst = o.at("layer_dinov2b_list");
    auto arr = lst.as_array();
    if (arr.size() >= 1) { cfg->enc_layer = (int)arr[0].as_real(17.0); }
  }
  return true;
}

std::string
MetalVosrTransformer::weights_dir(const std::string& root)
{
  namespace fs = std::filesystem;
  std::error_code ec;
  auto holds = [&](const fs::path& d) {
    if (!fs::is_directory(d, ec)) { return false; }
    for (const auto& e : fs::directory_iterator(d, ec)) {
      if (e.path().extension() == ".safetensors") { return true; }
    }
    return false;
  };
  // Relative to the directory holding args.json, not to what the caller
  // passed: a fetched root has the config a level down, and its
  // `checkpoints/` a level below that.
  const std::string a = args_root_(root);
  const fs::path r = a.empty() ? fs::path(root) : fs::path(a);
  if (holds(r / "checkpoints")) { return (r / "checkpoints").string(); }
  if (holds(r / "clean_weights")) { return (r / "clean_weights").string(); }
  if (holds(r)) { return r.string(); }
  return {};
}

std::string
MetalVosrTransformer::weights_path(const std::string& root)
{
  namespace fs = std::filesystem;
  const std::string d = weights_dir(root);
  if (d.empty()) { return {}; }
  std::error_code ec;
  // The reference's own search order, and it matters: a checkpoint
  // directory can hold the EMA weights beside the raw ones, and the EMA
  // set is the one every published result was produced with.
  for (const char* known : {"ema_model.safetensors", "model.safetensors"}) {
    if (fs::exists(fs::path(d) / known, ec)) {
      return (fs::path(d) / known).string();
    }
  }
  std::vector<std::string> hits;
  for (const auto& e : fs::directory_iterator(d, ec)) {
    if (e.path().extension() == ".safetensors") {
      hits.push_back(e.path().string());
    }
  }
  if (hits.empty()) { return {}; }
  std::sort(hits.begin(), hits.end());
  return hits.front();
}

std::unique_ptr<MetalVosrTransformer>
MetalVosrTransformer::load(const std::string& dir, MetalCompute* mc,
                           const Config& cfg, std::string* err)
{
  auto ws = WeightSet::open(dir, nullptr);
  if (!ws) {
    if (err != nullptr) { *err = "cannot open the checkpoint"; }
    return nullptr;
  }
  return load(std::move(ws), mc, cfg, err);
}

std::unique_ptr<MetalVosrTransformer>
MetalVosrTransformer::load(std::shared_ptr<WeightSet> ws, MetalCompute* mc,
                           const Config& cfg, std::string* err)
{
  auto fail = [&](const char* m) -> std::unique_ptr<MetalVosrTransformer> {
    if (err != nullptr) { *err = m; }
    return nullptr;
  };
  if (!ws || mc == nullptr) { return fail("no checkpoint or no backend"); }
  std::unique_ptr<MetalVosrTransformer> m(new MetalVosrTransformer());
  m->_mc  = mc;
  m->_cfg = cfg;
  m->_ws  = std::move(ws);

  m->_lib_gemm = mc->load_library("dense_gemm_bf16");
  m->_lib_elt  = mc->load_library("llm_elementwise_bf16");
  m->_lib_rms  = mc->load_library("rms_norm_bf16");
  m->_lib_rope = mc->load_library("rope_bf16");
  m->_lib_attn = mc->load_library("attn_steel");
  // MATRIX CORES, on by default where the GPU has them -- the same rule
  // krea2, flux2, boogu and (since this arc) qwen-image use, and the
  // restorer was written before any of it. Two independent switches
  // because the two kernels fail independently.
  if (mc->supports_matrix_cores()) {
    if (std::getenv("VPIPE_VOSR_NO_MMA2") == nullptr) {
      m->_lib_dense_mma = mc->load_library("dense_gemm_mma_bf16");
      m->_fn_gemm_mma =
          m->_lib_dense_mma.function("dense_gemm_mma_t_n128_f16");
      // ...and the bias pass it needs, because no matmul2d tile applies
      // one. Without it the matrix-core route is refused rather than run
      // biasless: a dropped bias is not a slower answer, it is a
      // different one.
      m->_fn_bias_rows = m->_lib_elt.function("bias_add_rows_f16");
      m->_use_mma2 = m->_fn_gemm_mma.valid() && m->_fn_bias_rows.valid();
    }
    if (std::getenv("VPIPE_VOSR_NO_ATTN_NAX") == nullptr) {
      m->_lib_attn_nax = mc->load_library("attn_steel_nax");
      m->_use_attn_nax = m->_lib_attn_nax.valid();
    }
  }
  m->_fn_gemm      = m->_lib_gemm.function("dense_gemm_t_bm64_f16");
  m->_fn_rms       = m->_lib_rms.function("rms_norm_fast_f16");
  m->_fn_ln        = m->_lib_elt.function("layer_norm_affine_f16");
  m->_fn_adaln     = m->_lib_elt.function("adaln_modulate_f16");
  m->_fn_gated     = m->_lib_elt.function("gated_residual_f16");
  m->_fn_residual  = m->_lib_elt.function("residual_add_f16");
  m->_fn_hslice    = m->_lib_elt.function("head_slice_f16");
  m->_fn_transpose = m->_lib_elt.function("transpose_abd_f16");
  m->_fn_swiglu    = m->_lib_elt.function("swiglu_split_gate_first_f16");
  m->_fn_silu      = m->_lib_elt.function("mul_sigmoid_f16");
  m->_fn_gelu      = m->_lib_elt.function("gelu_tanh_ff_f16");
  m->_fn_trope     = m->_lib_rope.function("transpose_rope_pair_ftab_f16");
  m->_fn_qknorm    = m->_lib_rope.function("rms_norm_heads_strided_f16");
  if (!m->_fn_gemm.valid() || !m->_fn_rms.valid() || !m->_fn_ln.valid() ||
      !m->_fn_adaln.valid() || !m->_fn_gated.valid() ||
      !m->_fn_residual.valid() || !m->_fn_hslice.valid() ||
      !m->_fn_transpose.valid() || !m->_fn_swiglu.valid() ||
      !m->_fn_silu.valid() || !m->_fn_gelu.valid() ||
      !m->_fn_trope.valid() || !m->_fn_qknorm.valid() ||
      !m->_lib_attn.valid()) {
    return fail("a metal kernel this model needs is missing");
  }
  m->_ap_self  = mc->make_shared_buffer(sizeof(SteelAttnParams));
  m->_ap_cross = mc->make_shared_buffer(sizeof(SteelAttnParams));
  if (m->_ap_self.empty() || m->_ap_cross.empty()) {
    return fail("attention parameter allocation failed");
  }
  // ---- the three opt-in accelerations -------------------------------
  //
  // All independent of each other and of the matrix-core switches above:
  // i8_gemm chooses how a block's GEMMs are computed, sol_attn which key
  // blocks are attended, and sage_attn how the attended ones are
  // multiplied. None reads the others.
  {
    auto i8 = std::make_unique<I8GemmContext>(mc, cfg.i8_gemm,
                                              /*bf16=*/true);
    i8->set_autotune(cfg.autotune);
    if (i8->enabled()) { m->_i8 = std::move(i8); }
  }
  {
    bool sage_fatal = false;
    m->_sage = MetalSageAttention::load_for_model(
        mc, /*bf16=*/true, cfg.sage, "MetalVosrTransformer", &sage_fatal);
    if (sage_fatal) { return nullptr; }
    // Sage rides on the NAX flash entry, which is where the int8 QK
    // lives. Said rather than implied: a run that asked for it and
    // quietly got dense attention would be reported as a Sage run.
    if (m->_sage && !m->_use_attn_nax) {
      if (mc->session() != nullptr) {
        mc->session()->log_normal(fmt(
            "MetalVosrTransformer: sage_attn is off -- the matrix-core "
            "flash entry is unavailable here"));
      }
      m->_sage.reset();
    }
  }
  if (cfg.sol.enabled) {
    // HEAD DIM 64, which Sol asserted against until this model. Both
    // flash kernels tile 64 and 128 identically, so the routing, the CSR
    // and the block statistics are unchanged; see the note in
    // MetalSolAttention::encode.
    std::string serr;
    m->_sol = MetalSolAttention::load(mc, /*bf16=*/true, &serr);
    if (!m->_sol) {
      if (mc->session() != nullptr) {
        mc->session()->log_normal(fmt(
            "MetalVosrTransformer: sol_attn requested but {}", serr));
      }
      return nullptr;
    }
    if (mc->session() != nullptr) {
      mc->session()->log_normal(fmt(
          "MetalVosrTransformer: Sol-Attn ON -- tau {:.2f}, layers {}+, "
          "local radius {}", (double)cfg.sol.tau, cfg.sol.dense_layers,
          cfg.sol.local_radius));
    }
  }
  if (mc->session() != nullptr &&
      (m->_use_mma2 || m->_use_attn_nax)) {
    mc->session()->log_debug(fmt(
        "MetalVosrTransformer: matrix cores -- GEMM {}, attention {}",
        m->_use_mma2 ? "matmul2d" : "ALU",
        m->_use_attn_nax ? "attn_steel_nax" : "ALU steel"));
  }
  if (!m->load_weights_(*m->_ws, err)) { return nullptr; }
  return m;
}

bool
MetalVosrTransformer::load_weights_(WeightSet& ws, std::string* err)
{
  const int D = _cfg.dim, C = _cfg.latent_ch, P = _cfg.patch;
  // NAME THE TENSOR THAT WAS NOT THERE. A restorer that declines to load
  // is otherwise indistinguishable from a wrong path, a wrong variant
  // and a truncated download, and all three look like "inert".
  auto take = [&](const std::string& nm, SharedBuffer& dst) {
    dst = to_bf16_(ws, _mc, nm);
    if (dst.empty()) {
      if (err != nullptr && err->empty()) {
        *err = "missing or unreadable tensor '" + nm + "'";
      }
      return false;
    }
    _bytes += dst.byte_size();
    return true;
  };

  if (!take("x_embedder.proj.weight", _patch_w)) { return false; }
  if (!take("x_embedder.proj.bias", _patch_b)) { return false; }
  // 2 * C input channels: the low-quality latent and the noise, packed
  // side by side. A checkpoint whose patch projection says otherwise is
  // not the model this class implements.
  if (_patch_w.byte_size() !=
      (std::size_t)D * (2 * C) * P * P * 2) {
    if (err != nullptr) {
      *err = "the patch projection is not " + std::to_string(2 * C) +
             " channels wide; this is not the VOSR 2.0 restorer";
    }
    return false;
  }
  if (!take("t_embedder.mlp.0.weight", _t0_w)) { return false; }
  if (!take("t_embedder.mlp.0.bias", _t0_b)) { return false; }
  if (!take("t_embedder.mlp.2.weight", _t2_w)) { return false; }
  if (!take("t_embedder.mlp.2.bias", _t2_b)) { return false; }
  if (!take("t_block.1.weight", _tblock_w)) { return false; }
  if (!take("t_block.1.bias", _tblock_b)) { return false; }
  if (!take("layer_norm.weight", _cln_w)) { return false; }
  if (!take("layer_norm.bias", _cln_b)) { return false; }
  if (!take("mlp_ca.fc1.weight", _ca1_w)) { return false; }
  if (!take("mlp_ca.fc1.bias", _ca1_b)) { return false; }
  if (!take("mlp_ca.fc2.weight", _ca2_w)) { return false; }
  if (!take("mlp_ca.fc2.bias", _ca2_b)) { return false; }
  if (!take("final_layer.adaLN_modulation.1.weight", _fmod_w)) { return false; }
  if (!take("final_layer.adaLN_modulation.1.bias", _fmod_b)) { return false; }
  if (!take("final_layer.norm_final.weight", _fnorm)) { return false; }
  if (!take("final_layer.linear.weight", _fout_w)) { return false; }
  if (!take("final_layer.linear.bias", _fout_b)) { return false; }

  _blocks.resize((std::size_t)_cfg.depth);
  for (int i = 0; i < _cfg.depth; ++i) {
    Block& b = _blocks[(std::size_t)i];
    const bool ok =
        take(blk_(i, "norm1.weight"), b.n1) &&
        take(blk_(i, "norm2.weight"), b.n2) &&
        take(blk_(i, "scale_shift_table"), b.sst) &&
        take(blk_(i, "attn.qkv.weight"), b.qkv_w) &&
        take(blk_(i, "attn.qkv.bias"), b.qkv_b) &&
        take(blk_(i, "attn.q_norm.weight"), b.qn) &&
        take(blk_(i, "attn.k_norm.weight"), b.kn) &&
        take(blk_(i, "attn.proj.weight"), b.proj_w) &&
        take(blk_(i, "attn.proj.bias"), b.proj_b) &&
        take(blk_(i, "cross_attn.q_linear.weight"), b.cq_w) &&
        take(blk_(i, "cross_attn.q_linear.bias"), b.cq_b) &&
        take(blk_(i, "cross_attn.k_linear.weight"), b.ck_w) &&
        take(blk_(i, "cross_attn.k_linear.bias"), b.ck_b) &&
        take(blk_(i, "cross_attn.v_linear.weight"), b.cv_w) &&
        take(blk_(i, "cross_attn.v_linear.bias"), b.cv_b) &&
        take(blk_(i, "cross_attn.q_norm.weight"), b.cqn) &&
        take(blk_(i, "cross_attn.k_norm.weight"), b.ckn) &&
        take(blk_(i, "cross_attn.proj.weight"), b.cproj_w) &&
        take(blk_(i, "cross_attn.proj.bias"), b.cproj_b) &&
        take(blk_(i, "mlp.w12.weight"), b.w12_w) &&
        take(blk_(i, "mlp.w12.bias"), b.w12_b) &&
        take(blk_(i, "mlp.w3.weight"), b.w3_w) &&
        take(blk_(i, "mlp.w3.bias"), b.w3_b);
    if (!ok) { return false; }
  }
  return true;
}

int
MetalVosrTransformer::default_tile(int lh, int lw, const Config& cfg)
{
  // The grid the weights were distilled at, expressed back in latent
  // cells: train_grid tokens of `patch` cells each. For the shipped
  // checkpoint that is 32 * 2 = 64, i.e. 512 output pixels.
  const int p = cfg.patch > 0 ? cfg.patch : 1;
  const int trained = cfg.train_grid > 0 ? cfg.train_grid * p : 0;
  if (trained <= 0) { return 0; }
  // Already inside it: one pass IS the reference's path here, and
  // tiling a picture that does not need it would only add seams.
  if (lh <= trained && lw <= trained) { return 0; }
  return trained;
}

void
MetalVosrTransformer::build_rope_(int gh, int gw, SharedBuffer& cos_out,
                                 SharedBuffer& sin_out) const
{
  // EVA-style 2D vision RoPE: the head dim splits in half, the first
  // half carrying the row coordinate and the second the column, each in
  // adjacent PAIRS that share one frequency.
  //
  // Positions are RESCALED, not extrapolated: coordinate p on an axis of
  // length g maps to p / g * train_grid, so a 4x upscale's 128-wide grid
  // still lands inside the range the table was trained over. That is the
  // reference's own _get_dynamic_rope. It only ever ran on square grids;
  // scaling each axis by its OWN length is the generalisation, and the
  // only one that keeps both axes inside the trained range.
  const int D = _cfg.head_dim;
  const int half = D / 2;             // per-axis width
  const int pairs = half / 2;         // frequencies per axis
  const int T = gh * gw;
  auto* cb = static_cast<float*>(cos_out.contents());
  auto* sb = static_cast<float*>(sin_out.contents());
  std::vector<double> freq((std::size_t)pairs);
  for (int i = 0; i < pairs; ++i) {
    freq[(std::size_t)i] =
        1.0 / std::pow(_cfg.rope_theta, (2.0 * (double)i) / (double)half);
  }
  const double pt = (double)_cfg.train_grid;
  for (int y = 0; y < gh; ++y) {
    const double ty = (double)y / (double)gh * pt;
    for (int x = 0; x < gw; ++x) {
      const double tx = (double)x / (double)gw * pt;
      float* c = cb + (std::size_t)(y * gw + x) * D;
      float* s = sb + (std::size_t)(y * gw + x) * D;
      for (int i = 0; i < pairs; ++i) {
        const double ay = ty * freq[(std::size_t)i];
        const double ax = tx * freq[(std::size_t)i];
        const float cy = (float)std::cos(ay), sy = (float)std::sin(ay);
        const float cx = (float)std::cos(ax), sx = (float)std::sin(ax);
        c[2 * i] = cy; c[2 * i + 1] = cy;
        s[2 * i] = sy; s[2 * i + 1] = sy;
        c[half + 2 * i] = cx; c[half + 2 * i + 1] = cx;
        s[half + 2 * i] = sx; s[half + 2 * i + 1] = sx;
      }
    }
  }
  (void)T;
}

bool
MetalVosrTransformer::build_attn_(int qL, int kL, bool self)
{
  SharedBuffer& pb = self ? _ap_self : _ap_cross;
  const int NH = _cfg.n_heads, HD = _cfg.head_dim;
  auto* p = static_cast<SteelAttnParams*>(pb.contents());
  p->B = 1; p->H = NH; p->D = HD;
  p->qL = qL; p->kL = kL;
  p->gqa_factor = 1;
  p->scale = 1.0f / std::sqrt((float)HD);
  // EVERY NQ/NK BELOW COMES FROM THESE, so this one line is what makes
  // the params describe the kernel that will actually run. head_dim 64
  // tiles identically to 128 on both kernels, which is why nothing else
  // here moves with the width.
  const int BQ = attn_bq_(), BK = attn_bk_();
  p->NQ = (qL + BQ - 1) / BQ;
  p->NK = (kL + BK - 1) / BK;
  p->NQ_aligned = qL / BQ;
  p->NK_aligned = kL / BK;
  p->qL_rem = qL - p->NQ_aligned * BQ;
  p->kL_rem = kL - p->NK_aligned * BK;
  p->qL_off = 0;
  p->Q_strides[0] = (std::int64_t)NH * qL * HD;
  p->Q_strides[1] = (std::int64_t)qL * HD;
  p->Q_strides[2] = HD;
  p->K_strides[0] = (std::int64_t)NH * kL * HD;
  p->K_strides[1] = (std::int64_t)kL * HD;
  p->K_strides[2] = HD;
  p->V_strides[0] = p->K_strides[0];
  p->V_strides[1] = p->K_strides[1];
  p->V_strides[2] = HD;
  p->O_strides[0] = p->Q_strides[0];
  p->O_strides[1] = p->Q_strides[1];
  p->O_strides[2] = HD;
  auto build = [&](bool i8) {
    metal_compute::FunctionConstants fc;
    fc.set_bool(200, (qL % BQ) == 0).set_bool(201, (kL % BK) == 0)
        .set_bool(300, false).set_bool(301, false).set_bool(302, false)
        .set_bool(sage::kQkInt8Constant, i8);
    return _use_attn_nax
        ? _lib_attn_nax.function("attn_steel_nax_h_bd64_bf16", fc)
        : _lib_attn.function("attn_steel_h_bd64_bf16", fc);
  };
  metal_compute::ComputeFunction fn = build(false);
  if (!fn.valid()) { return false; }
  // The int8-QK twin, on the matrix-core entry only -- Sage's
  // dense_layers leaves the leading blocks on the plain kernel, so both
  // have to exist within one forward.
  metal_compute::ComputeFunction fn8;
  if (_sage && _use_attn_nax) { fn8 = build(true); }
  if (self) {
    _fn_attn_self = std::move(fn);
    _fn_attn_self_i8 = std::move(fn8);
  } else {
    _fn_attn_cross = std::move(fn);
    _fn_attn_cross_i8 = std::move(fn8);
  }
  return true;
}

void
MetalVosrTransformer::encode_gemm_(ComputeEncoder& enc,
                                   const SharedBuffer& x, std::size_t xe,
                                   const SharedBuffer& w,
                                   const SharedBuffer& b,
                                   const SharedBuffer& y, std::size_t ye,
                                   int M, int N, int K, bool bias)
{
  // MATRIX CORES FIRST, on the same terms as every sibling DiT: the
  // matmul2d tile takes the SAME operands the ALU kernel does, bias slot
  // and flag included, so this is a dispatch-shape change and not a
  // different arithmetic. K here is 1536 (the projections), 4096 (the
  // feed-forward's down leg) or 1024/4608 (the conditioning MLP), all
  // under the 6144 the wider tile wants -- so the n128 region is the
  // only one this model reaches.
  // Accelerated mode first, bias and all: it takes the same operands the
  // tiles below do and declines with nothing encoded on a shape it does
  // not want, so it sits ahead of the tile choice rather than beside it.
  // SharedBuffer is move-only, so the no-bias case names a static empty
  // one rather than constructing a temporary in the argument list.
  static const SharedBuffer kNoBias;
  if (_i8 && _i8->gemm(enc, x, xe, w, (bias && !b.empty()) ? b : kNoBias, y,
                       ye, M, N, K)) {
    return;
  }
  if (_use_mma2 && _fn_gemm_mma.valid() && _fn_bias_rows.valid() &&
      M >= 64) {
    // THE MATMUL2D TILES DO NOT APPLY A BIAS. Every dense_gemm_mma entry
    // takes the slot for signature compatibility with the ALU kernel and
    // then discards it -- `(void)has_bias; (void)bias;` -- because the
    // families that reached them first (krea2, flux2, qwen) have
    // bias-free projections and never noticed. VOSR's every projection
    // carries one, so binding it here and trusting the flag produced an
    // output uncorrelated with the ALU path's (rel-L2 1.03, caught by
    // vosr_accel.the_accelerations_agree_with_the_alu_baseline).
    //
    // So the bias is a second pass. One read-modify-write over [M, N]
    // against a contraction over K -- 1536 or 4096 here -- which is a
    // few percent, and it is the honest shape until a tile grows an
    // epilogue.
    constexpr int RN = 128;
    enc.set_function(_fn_gemm_mma);
    enc.set_buffer(0, x, xe * 2);
    enc.set_buffer(1, w);
    enc.set_buffer(2, w);              // unread; the slot must be bound
    enc.set_buffer(3, y, ye * 2);
    enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, M);
    enc.set_constant(7, 0);
    enc.dispatch({(unsigned)(((N + RN - 1) / RN) * 256),
                  (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
    if (bias && !b.empty()) {
      enc.set_function(_fn_bias_rows);
      enc.set_buffer(0, y, ye * 2);
      enc.set_buffer(1, b);
      enc.set_constant(2, N);
      const unsigned total = (unsigned)((std::size_t)M * N);
      enc.set_constant(3, total);
      enc.dispatch({(unsigned)N, (unsigned)M, 1}, {32, 8, 1});
    }
    return;
  }
  enc.set_function(_fn_gemm);
  enc.set_buffer(0, x, xe * 2);
  enc.set_buffer(1, w);
  enc.set_buffer(2, bias && !b.empty() ? b : w);
  enc.set_buffer(3, y, ye * 2);
  enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, M);
  enc.set_constant(7, (bias && !b.empty()) ? 1 : 0);
  enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
}

bool
MetalVosrTransformer::alloc_scratch_(Scratch* s, int T, int K) const
{
  const int D = _cfg.dim, FF = _cfg.ffn, C = _cfg.latent_ch, P = _cfg.patch;
  auto buf = [&](std::size_t elems) {
    return _mc->make_shared_buffer(elems * 2);
  };
  s->T = T; s->K = K;
  s->pix  = buf((std::size_t)T * (2 * C) * P * P);
  s->x    = buf((std::size_t)T * D);
  s->nrm  = buf((std::size_t)T * D);
  s->mod  = buf((std::size_t)6 * D);
  s->qkv  = buf((std::size_t)T * 3 * D);
  s->q    = buf((std::size_t)T * D);
  s->k    = buf((std::size_t)T * D);
  s->v    = buf((std::size_t)T * D);
  s->qt   = buf((std::size_t)T * D);
  s->kt   = buf((std::size_t)T * D);
  s->vt   = buf((std::size_t)T * D);
  s->at   = buf((std::size_t)T * D);
  s->att  = buf((std::size_t)T * D);
  s->o    = buf((std::size_t)T * D);
  s->ff   = buf((std::size_t)T * 2 * FF);
  s->ffo  = buf((std::size_t)T * FF);
  s->fsil = buf((std::size_t)D);
  s->fmod = buf((std::size_t)2 * D);
  s->ck   = buf((std::size_t)K * D);
  s->cv   = buf((std::size_t)K * D);
  s->ckt  = buf((std::size_t)K * D);
  s->cvt  = buf((std::size_t)K * D);
  s->outp = buf((std::size_t)T * P * P * C);
  return !(s->pix.empty() || s->x.empty() || s->qkv.empty() ||
           s->ff.empty() || s->ffo.empty() || s->fsil.empty() ||
           s->fmod.empty() || s->ck.empty() || s->outp.empty());
}

SharedBuffer
MetalVosrTransformer::project_cond_(const void* cond, int rows)
{
  const int D = _cfg.dim, E = _cfg.enc_dim, M = _cfg.enc_mlp;
  if (cond == nullptr || rows <= 0) { return {}; }
  SharedBuffer in = _mc->make_shared_buffer((std::size_t)rows * E * 2);
  SharedBuffer nrm = _mc->make_shared_buffer((std::size_t)rows * E * 2);
  SharedBuffer mid = _mc->make_shared_buffer((std::size_t)rows * M * 2);
  SharedBuffer out = _mc->make_shared_buffer((std::size_t)rows * D * 2);
  if (in.empty() || nrm.empty() || mid.empty() || out.empty()) { return {}; }
  std::memcpy(in.contents(), cond, (std::size_t)rows * E * 2);
  CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    enc.set_function(_fn_ln);
    enc.set_buffer(0, in); enc.set_buffer(1, _cln_w);
    enc.set_buffer(2, _cln_b); enc.set_buffer(3, nrm);
    enc.set_constant(4, E); enc.set_constant(5, _cfg.cond_norm_eps);
    enc.dispatch({256, (unsigned)rows, 1}, {256, 1, 1});
    auto gemm = [&](const SharedBuffer& x, const SharedBuffer& w,
                    const SharedBuffer& b, const SharedBuffer& y, int Mm,
                    int N, int K) {
      encode_gemm_(enc, x, 0, w, b, y, 0, Mm, N, K, /*bias=*/true);
    };
    gemm(nrm, _ca1_w, _ca1_b, mid, rows, M, E);
    // timm's Mlp with GELU(approximate="tanh") -- the tanh approximation
    // is what the reference builds, not the erf form.
    enc.set_function(_fn_gelu);
    enc.set_buffer(0, mid); enc.set_buffer(1, mid);
    enc.set_constant(2, rows * M);
    enc.dispatch({(unsigned)(rows * M), 1, 1}, {256, 1, 1});
    gemm(mid, _ca2_w, _ca2_b, out, rows, D, M);
  }
  if (!stream.commit().wait_ok(nullptr)) { return {}; }
  return out;
}

bool
MetalVosrTransformer::build_time_(float t, SharedBuffer* c, SharedBuffer* c0)
{
  const int D = _cfg.dim, F = _cfg.freq_dim;
  SharedBuffer freq = _mc->make_shared_buffer((std::size_t)F * 2);
  SharedBuffer h1 = _mc->make_shared_buffer((std::size_t)D * 2);
  SharedBuffer sil = _mc->make_shared_buffer((std::size_t)D * 2);
  *c  = _mc->make_shared_buffer((std::size_t)D * 2);
  *c0 = _mc->make_shared_buffer((std::size_t)6 * D * 2);
  if (freq.empty() || h1.empty() || sil.empty() || c->empty() ||
      c0->empty()) {
    return false;
  }
  {
    // The DiT sinusoid: cos first, then sin, over F/2 log-spaced
    // frequencies. Built in double on the host and rounded once.
    auto* d = static_cast<std::uint16_t*>(freq.contents());
    const int half = F / 2;
    for (int i = 0; i < half; ++i) {
      const double w =
          std::exp(-std::log(10000.0) * (double)i / (double)half);
      const double a = (double)t * w;
      d[i] = f32_to_bf16_((float)std::cos(a));
      d[half + i] = f32_to_bf16_((float)std::sin(a));
    }
  }
  CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    auto gemm = [&](const SharedBuffer& x, const SharedBuffer& w,
                    const SharedBuffer& b, const SharedBuffer& y, int N,
                    int K) {
      enc.set_function(_fn_gemm);
      enc.set_buffer(0, x); enc.set_buffer(1, w); enc.set_buffer(2, b);
      enc.set_buffer(3, y);
      enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, 1);
      enc.set_constant(7, 1);
      enc.dispatch({(unsigned)(((N + 31) / 32) * 32), 2, 2}, {32, 2, 2});
    };
    auto silu = [&](const SharedBuffer& x, const SharedBuffer& y, int n) {
      enc.set_function(_fn_silu);
      enc.set_buffer(0, x); enc.set_buffer(1, x); enc.set_buffer(2, y);
      enc.set_constant(3, n);
      enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
    };
    gemm(freq, _t0_w, _t0_b, h1, D, F);
    silu(h1, sil, D);
    gemm(sil, _t2_w, _t2_b, *c, D, D);       // c = t_embedder(t)
    silu(*c, sil, D);
    gemm(sil, _tblock_w, _tblock_b, *c0, 6 * D, D);
  }
  return stream.commit().wait_ok(nullptr);
}

bool
MetalVosrTransformer::forward_(Scratch& s, const SharedBuffer& zc, int K,
                               const SharedBuffer& c, const SharedBuffer& c0,
                               int gh, int gw,
                               const DitBlockProgressFn& block_progress,
                               std::string* err)
{
  const int D = _cfg.dim, NH = _cfg.n_heads, HD = _cfg.head_dim;
  const int FF = _cfg.ffn, C = _cfg.latent_ch, P = _cfg.patch;
  const int T = gh * gw;
  const int PV = 2 * C * P * P;

  // The int8 split's width, for shapes an earlier step recorded. Here
  // because it runs its own command streams and so needs no encoder open.
  if (_i8) { _i8->tune_pending(_mc); }
  CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    auto gemm = [&](const SharedBuffer& x, const SharedBuffer& w,
                    const SharedBuffer& b, const SharedBuffer& y, int Mm,
                    int N, int Kd) {
      encode_gemm_(enc, x, 0, w, b, y, 0, Mm, N, Kd, !b.empty());
    };
    auto rms = [&](const SharedBuffer& x, const SharedBuffer& w,
                   const SharedBuffer& y, int rows) {
      enc.set_function(_fn_rms);
      enc.set_buffer(0, x); enc.set_buffer(1, w); enc.set_buffer(2, y);
      enc.set_constant(3, D); enc.set_constant(4, _cfg.norm_eps);
      enc.dispatch({256, (unsigned)rows, 1}, {256, 1, 1});
    };
    // out = x * (1 + mod[scale_i]) + mod[shift_i], both broadcast over
    // the token rows.
    auto adaln = [&](const SharedBuffer& x, const SharedBuffer& m,
                     int shift_i, int scale_i, const SharedBuffer& y) {
      enc.set_function(_fn_adaln);
      enc.set_buffer(0, x);
      enc.set_buffer(1, m, (std::size_t)scale_i * D * 2);
      enc.set_buffer(2, m, (std::size_t)shift_i * D * 2);
      enc.set_buffer(3, y);
      enc.set_constant(4, D); enc.set_constant(5, T * D);
      enc.dispatch({(unsigned)(T * D), 1, 1}, {256, 1, 1});
    };
    auto gated = [&](const SharedBuffer& h, int gate_i,
                     const SharedBuffer& sub) {
      enc.set_function(_fn_gated);
      enc.set_buffer(0, h);
      enc.set_buffer(1, s.mod, (std::size_t)gate_i * D * 2);
      enc.set_buffer(2, sub);
      enc.set_constant(3, D); enc.set_constant(4, T * D);
      enc.dispatch({(unsigned)(T * D), 1, 1}, {256, 1, 1});
    };
    auto hslice = [&](const SharedBuffer& in_b, const SharedBuffer& out_b,
                      int rows, int stride, int off) {
      enc.set_function(_fn_hslice);
      enc.set_buffer(0, in_b); enc.set_buffer(1, out_b);
      enc.set_constant(2, rows); enc.set_constant(3, stride);
      enc.set_constant(4, D); enc.set_constant(5, off);
      enc.set_constant(6, 0); enc.set_constant(7, 0);
      enc.dispatch({(unsigned)(rows * D), 1, 1}, {256, 1, 1});
    };
    auto qknorm = [&](const SharedBuffer& x, const SharedBuffer& g, int rows,
                      int stride, int off) {
      enc.set_function(_fn_qknorm);
      enc.set_buffer(0, x); enc.set_buffer(1, g);
      enc.set_constant(2, rows); enc.set_constant(3, NH);
      enc.set_constant(4, HD); enc.set_constant(5, stride);
      enc.set_constant(6, off); enc.set_constant(7, _cfg.norm_eps);
      enc.set_constant(8, HD);
      enc.dispatch({32, (unsigned)(rows * NH), 1}, {32, 1, 1});
    };
    auto transpose = [&](const SharedBuffer& in_b, const SharedBuffer& out_b,
                         int A, int Bd) {
      enc.set_function(_fn_transpose);
      enc.set_buffer(0, in_b); enc.set_buffer(1, out_b);
      enc.set_constant(2, A); enc.set_constant(3, Bd);
      enc.set_constant(4, HD);
      enc.dispatch({(unsigned)HD, (unsigned)Bd, (unsigned)A},
                   {(unsigned)HD, 1, 1});
    };
    // Fused [T,NH,HD] -> [NH,T,HD] transpose + interleaved-pair RoPE.
    auto trope = [&](const SharedBuffer& in_b, const SharedBuffer& out_b) {
      enc.set_function(_fn_trope);
      enc.set_buffer(0, in_b); enc.set_buffer(1, out_b);
      enc.set_buffer(2, _rcos); enc.set_buffer(3, _rsin);
      enc.set_constant(4, NH); enc.set_constant(5, T);
      enc.set_constant(6, HD);
      enc.dispatch({(unsigned)(HD / 2), (unsigned)T, (unsigned)NH},
                   {(unsigned)(HD / 2), 1, 1});
    };
    auto attn = [&](const SharedBuffer& q, const SharedBuffer& k,
                    const SharedBuffer& v, SharedBuffer& out, bool self,
                    int L) {
      // SOL, on the SELF attention only. The cross attention reads the
      // conditioning tokens -- a few hundred DINOv2 rows against the
      // image's tens of thousands -- so there is nothing to be sparse
      // in, and a routed prompt is a prompt half-read.
      const bool sol_here =
          self && _sol && _cfg.sol.enabled && L >= _cfg.sol.dense_layers;
      // SAGE, on whichever half runs the flash kernel. Independent of
      // Sol: one decides which key blocks are attended, the other how
      // the attended ones are multiplied.
      const bool sage_here =
          _sage && _cfg.sage.enabled && L >= _cfg.sage.dense_layers;
      const int kL = self ? T : K;
      bool sage_ok = false;
      if (sage_here) {
        const MetalSageAttention::Operand qo{&q, 0, HD, T * HD};
        const MetalSageAttention::Operand ko{&k, 0, HD, kL * HD};
        std::string gerr;
        // Sol's exact half and the dense kernel tile identically here,
        // so ONE prologue serves either. compose() clears the key
        // smoothing when Sol is on: that shift is exact under one
        // softmax and Sol's row is split across two.
        const int bq = sol_here ? _sol->query_block() : attn_bq_();
        const int bk = sol_here ? _sol->key_block_unit() : attn_bk_();
        sage_ok = _sage->prepare(
            enc, qo, ko, NH, NH, T, kL, HD, bq, bk,
            sol_here ? MetalSolAttention::compose(_cfg.sage) : _cfg.sage,
            &gerr);
      }
      if (sol_here) {
        sol::Config sc = _cfg.sol;
        // No packed prompt in this sequence: the restorer's self
        // attention is image tokens alone, so there is no sink.
        sc.sink_start = 0;
        sc.sink_tokens = 0;
        _sol->set_sage(sage_ok ? _sage.get() : nullptr);
        std::string serr;
        if (!_sol->encode(enc, q, k, v, out, NH, T, HD, 1.0f /
                          std::sqrt((float)HD), sc, &serr)) {
          return false;
        }
        return true;
      }
      const metal_compute::ComputeFunction& i8fn =
          self ? _fn_attn_self_i8 : _fn_attn_cross_i8;
      const bool use_i8 = sage_ok && i8fn.valid();
      enc.set_function(use_i8 ? i8fn
                              : (self ? _fn_attn_self : _fn_attn_cross));
      enc.set_buffer(0, q); enc.set_buffer(1, k); enc.set_buffer(2, v);
      enc.set_buffer(3, out);
      enc.set_buffer(4, self ? _ap_self : _ap_cross);
      if (use_i8) { _sage->bind(enc); }
      const unsigned nqb =
          (unsigned)((T + attn_bq_() - 1) / attn_bq_());
      enc.dispatch({32 * nqb, 4 * (unsigned)NH, 1}, {32, 4, 1});
      return true;
    };

    gemm(s.pix, _patch_w, _patch_b, s.x, T, D, PV);

    for (int L = 0; L < _cfg.depth; ++L) {
      report_block(stream, block_progress, L, _cfg.depth);
      const Block& b = _blocks[(std::size_t)L];
      // The six modulation vectors: the block's own table plus the
      // shared timestep projection. Order is the reference's chunk
      // order -- shift, scale, gate for the attention, then the same
      // three for the MLP.
      enc.set_function(_fn_residual);
      enc.set_buffer(0, c0); enc.set_buffer(1, b.sst);
      enc.set_buffer(2, s.mod);
      enc.set_constant(3, 6 * D);
      enc.dispatch({(unsigned)(6 * D), 1, 1}, {256, 1, 1});

      // ---- self attention ------------------------------------------
      rms(s.x, b.n1, s.nrm, T);
      adaln(s.nrm, s.mod, 0, 1, s.nrm);
      gemm(s.nrm, b.qkv_w, b.qkv_b, s.qkv, T, 3 * D, D);
      qknorm(s.qkv, b.qn, T, 3 * D, 0);
      qknorm(s.qkv, b.kn, T, 3 * D, D);
      hslice(s.qkv, s.q, T, 3 * D, 0);
      hslice(s.qkv, s.k, T, 3 * D, D);
      hslice(s.qkv, s.v, T, 3 * D, 2 * D);
      trope(s.q, s.qt);
      trope(s.k, s.kt);
      transpose(s.v, s.vt, T, NH);
      if (!attn(s.qt, s.kt, s.vt, s.at, true, L)) {
        if (err != nullptr) { *err = "vosr: self attention failed"; }
        return false;
      }
      transpose(s.at, s.att, NH, T);
      gemm(s.att, b.proj_w, b.proj_b, s.o, T, D, D);
      gated(s.x, 2, s.o);

      // ---- cross attention to the DINOv2 tokens ---------------------
      //
      // UNGATED: the reference adds it straight onto the residual with
      // no gate_msa-style term, which is what makes the conditioning
      // impossible for the model to switch off.
      gemm(s.x, b.cq_w, b.cq_b, s.q, T, D, D);
      gemm(zc, b.ck_w, b.ck_b, s.ck, K, D, D);
      gemm(zc, b.cv_w, b.cv_b, s.cv, K, D, D);
      qknorm(s.q, b.cqn, T, D, 0);
      qknorm(s.ck, b.ckn, K, D, 0);
      transpose(s.q, s.qt, T, NH);
      transpose(s.ck, s.ckt, K, NH);
      transpose(s.cv, s.cvt, K, NH);
      if (!attn(s.qt, s.ckt, s.cvt, s.at, false, L)) {
        if (err != nullptr) { *err = "vosr: cross attention failed"; }
        return false;
      }
      transpose(s.at, s.att, NH, T);
      gemm(s.att, b.cproj_w, b.cproj_b, s.o, T, D, D);
      enc.set_function(_fn_residual);
      enc.set_buffer(0, s.x); enc.set_buffer(1, s.o); enc.set_buffer(2, s.x);
      enc.set_constant(3, T * D);
      enc.dispatch({(unsigned)(T * D), 1, 1}, {256, 1, 1});

      // ---- SwiGLU MLP ------------------------------------------------
      rms(s.x, b.n2, s.nrm, T);
      adaln(s.nrm, s.mod, 3, 4, s.nrm);
      gemm(s.nrm, b.w12_w, b.w12_b, s.ff, T, 2 * FF, D);
      enc.set_function(_fn_swiglu);
      enc.set_buffer(0, s.ff); enc.set_buffer(1, s.ffo);
      enc.set_constant(2, T); enc.set_constant(3, FF);
      enc.dispatch({(unsigned)(T * FF), 1, 1}, {256, 1, 1});
      gemm(s.ffo, b.w3_w, b.w3_b, s.o, T, D, FF);
      gated(s.x, 5, s.o);
    }

    // ---- final layer -------------------------------------------------
    // Its modulation comes from `c` (the raw timestep embedding), NOT
    // from the six-way table the blocks read.
    enc.set_function(_fn_silu);
    enc.set_buffer(0, c); enc.set_buffer(1, c); enc.set_buffer(2, s.fsil);
    enc.set_constant(3, D);
    enc.dispatch({(unsigned)D, 1, 1}, {256, 1, 1});
    gemm(s.fsil, _fmod_w, _fmod_b, s.fmod, 1, 2 * D, D);
    rms(s.x, _fnorm, s.nrm, T);
    adaln(s.nrm, s.fmod, 0, 1, s.nrm);
    gemm(s.nrm, _fout_w, _fout_b, s.outp, T, P * P * C, D);
  }
  if (!stream.commit().wait_ok(err)) { return false; }
  return true;
}

bool
MetalVosrTransformer::restore(const RestoreRequest& req,
                              std::vector<float>* out, std::string* err)
{
  auto fail = [&](const char* m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  const int C = _cfg.latent_ch, P = _cfg.patch, D = _cfg.dim;
  const int lh = req.lh, lw = req.lw;
  if (req.lq == nullptr || lh <= 0 || lw <= 0) { return fail("no latent"); }
  if (req.cond == nullptr || req.cond_rows <= 0) {
    return fail("no conditioning");
  }
  if (lh % P != 0 || lw % P != 0) {
    return fail("the latent grid must be a multiple of the patch size");
  }
  const int steps = std::max(1, req.steps);

  // ---- tiles -----------------------------------------------------------
  int tile = req.tile;
  int overlap = req.tile_overlap;
  if (tile > 0) {
    tile = std::max((tile / P) * P, P);
    tile = std::min(tile, std::min(lh, lw));
    overlap = std::max(overlap, tile / 8);
    overlap = std::min(std::max(overlap, 0), tile - 1);
  }
  const bool tiled = tile > 0 && (tile < lh || tile < lw);
  const int th = tiled ? tile : lh;
  const int tw = tiled ? tile : lw;
  std::vector<int> hpos = tiled ? tile_grid_(lh, th, overlap)
                                : std::vector<int>{0};
  std::vector<int> wpos = tiled ? tile_grid_(lw, tw, overlap)
                                : std::vector<int>{0};

  // The conditioning grid, for the per-tile crop. The tower squashes its
  // input to a square, so the token grid is square whatever the picture
  // was.
  const int fs = (int)std::lround(std::sqrt((double)req.cond_rows));
  if (fs * fs != req.cond_rows) {
    return fail("the conditioning token grid is not square");
  }

  // ---- the shared noise field -----------------------------------------
  //
  // ONE field for the whole latent even when tiling, which is what keeps
  // overlapping tiles agreeing: two tiles denoising the same pixel from
  // different noise would disagree by more than the blend can hide.
  const std::size_t n_lat = (std::size_t)C * lh * lw;
  std::vector<float> z(n_lat);
  if (req.noise != nullptr) {
    std::memcpy(z.data(), req.noise, n_lat * sizeof(float));
  } else {
    std::mt19937_64 rng(req.seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (std::size_t i = 0; i < n_lat; ++i) { z[i] = nd(rng); }
  }

  const int gh = th / P, gw = tw / P;
  const int T = gh * gw;

  // Rope tables for the tile grid.
  if (_rope_gh != gh || _rope_gw != gw) {
    _rcos = _mc->make_shared_buffer((std::size_t)T * _cfg.head_dim * 4);
    _rsin = _mc->make_shared_buffer((std::size_t)T * _cfg.head_dim * 4);
    if (_rcos.empty() || _rsin.empty()) { return fail("rope alloc failed"); }
    build_rope_(gh, gw, _rcos, _rsin);
    _rope_gh = gh; _rope_gw = gw;
  }

  // ---- conditioning, projected once per distinct crop ------------------
  struct TileCond { SharedBuffer zc; int rows = 0; };
  std::vector<TileCond> tcond(hpos.size() * wpos.size());
  const auto* cond16 = static_cast<const std::uint16_t*>(req.cond);
  const int E = _cfg.enc_dim;
  int max_k = 0;
  for (std::size_t hi = 0; hi < hpos.size(); ++hi) {
    for (std::size_t wi = 0; wi < wpos.size(); ++wi) {
      TileCond& tc = tcond[hi * wpos.size() + wi];
      if (!tiled) {
        tc.zc = project_cond_(req.cond, req.cond_rows);
        tc.rows = req.cond_rows;
      } else {
        // The reference re-runs its tower on the tile's PIXELS, which a
        // graph whose conditioner ran once cannot do. Cropping the
        // feature grid is the reference's own alternative
        // (_crop_venc_features) and the only one available here; it is
        // why a tiled run is not bit-comparable with the reference.
        const int y0 = hpos[hi], x0 = wpos[wi];
        const int fh_s = (int)((double)y0 / lh * fs);
        const int fh_e = std::max((int)((double)(y0 + th) / lh * fs),
                                  fh_s + 1);
        const int fw_s = (int)((double)x0 / lw * fs);
        const int fw_e = std::max((int)((double)(x0 + tw) / lw * fs),
                                  fw_s + 1);
        const int rows = (fh_e - fh_s) * (fw_e - fw_s);
        std::vector<std::uint16_t> crop((std::size_t)rows * E);
        int r = 0;
        for (int y = fh_s; y < fh_e; ++y) {
          for (int x = fw_s; x < fw_e; ++x, ++r) {
            std::memcpy(&crop[(std::size_t)r * E],
                        cond16 + (std::size_t)(y * fs + x) * E,
                        (std::size_t)E * 2);
          }
        }
        tc.zc = project_cond_(crop.data(), rows);
        tc.rows = rows;
      }
      if (tc.zc.empty()) { return fail("conditioning projection failed"); }
      max_k = std::max(max_k, tc.rows);
    }
  }

  Scratch s;
  if (!alloc_scratch_(&s, T, max_k)) { return fail("scratch alloc failed"); }
  if (!build_attn_(T, T, true)) { return fail("attention build failed"); }
  _attn_q = T; _attn_kv = 0;

  // Gaussian blend weights over a tile, the reference's _gaussian_weights.
  std::vector<float> gw_w;
  if (tiled) {
    gw_w.resize((std::size_t)th * tw);
    const float var = 0.01f;
    const float mh = (float)(th - 1) / 2.0f, mw = (float)(tw - 1) / 2.0f;
    for (int y = 0; y < th; ++y) {
      const float a = ((float)y - mh) / (float)th;
      const float wy = std::exp(-(a * a) / (2.0f * var));
      for (int x = 0; x < tw; ++x) {
        const float bx = ((float)x - mw) / (float)tw;
        gw_w[(std::size_t)y * tw + x] =
            wy * std::exp(-(bx * bx) / (2.0f * var));
      }
    }
  }

  std::vector<float> uacc, wacc;
  if (tiled) { uacc.assign(n_lat, 0.0f); wacc.assign(n_lat, 0.0f); }

  for (int i = 0; i < steps; ++i) {
    const float t_cur = 1.0f - (float)i / (float)steps;
    const float t_nxt = 1.0f - (float)(i + 1) / (float)steps;
    const float dt = t_cur - t_nxt;
    SharedBuffer c, c0;
    if (!build_time_(t_cur, &c, &c0)) { return fail("timestep embed failed"); }
    if (tiled) {
      std::fill(uacc.begin(), uacc.end(), 0.0f);
      std::fill(wacc.begin(), wacc.end(), 0.0f);
    }
    for (std::size_t hi = 0; hi < hpos.size(); ++hi) {
      for (std::size_t wi = 0; wi < wpos.size(); ++wi) {
        const int y0 = hpos[hi], x0 = wpos[wi];
        const TileCond& tc = tcond[hi * wpos.size() + wi];
        if (_attn_kv != tc.rows) {
          if (!build_attn_(T, tc.rows, false)) {
            return fail("cross-attention build failed");
          }
          _attn_kv = tc.rows;
        }
        // Pack the tile's patch rows: 2*C channels, the low-quality
        // latent then the noise, in (channel, row, col) order -- the
        // flattening of the reference's Conv2d patch projection.
        {
          auto* d = static_cast<std::uint16_t*>(s.pix.contents());
          const int PV = 2 * C * P * P;
          for (int py = 0; py < gh; ++py) {
            for (int px = 0; px < gw; ++px) {
              std::uint16_t* row = d + (std::size_t)(py * gw + px) * PV;
              for (int ch = 0; ch < 2 * C; ++ch) {
                const float* src = (ch < C ? req.lq : z.data())
                                 + (std::size_t)(ch % C) * lh * lw;
                for (int ih = 0; ih < P; ++ih) {
                  const int yy = y0 + py * P + ih;
                  const float* sr = src + (std::size_t)yy * lw + x0 + px * P;
                  std::uint16_t* dr =
                      row + (std::size_t)ch * P * P + ih * P;
                  for (int iw = 0; iw < P; ++iw) {
                    dr[iw] = f32_to_bf16_(sr[iw]);
                  }
                }
              }
            }
          }
        }
        std::string ferr;
        if (!forward_(s, tc.zc, tc.rows, c, c0, gh, gw, req.block_progress,
                      &ferr)) {
          if (err != nullptr) { *err = ferr; }
          return false;
        }
        // Unpatchify into the velocity. The final linear's columns are
        // ordered (row, col, channel) -- the channel is FASTEST here
        // where it was slowest going in.
        const auto* op = static_cast<const std::uint16_t*>(s.outp.contents());
        const int PC = P * P * C;
        for (int py = 0; py < gh; ++py) {
          for (int px = 0; px < gw; ++px) {
            const std::uint16_t* row =
                op + (std::size_t)(py * gw + px) * PC;
            for (int ih = 0; ih < P; ++ih) {
              for (int iw = 0; iw < P; ++iw) {
                const int yy = py * P + ih, xx = px * P + iw;
                const std::uint16_t* cell =
                    row + (std::size_t)(ih * P + iw) * C;
                for (int ch = 0; ch < C; ++ch) {
                  const float u = bf16_to_f32_(cell[ch]);
                  const std::size_t gidx =
                      (std::size_t)ch * lh * lw
                      + (std::size_t)(y0 + yy) * lw + (x0 + xx);
                  if (tiled) {
                    const float w = gw_w[(std::size_t)yy * tw + xx];
                    uacc[gidx] += u * w;
                    wacc[gidx] += w;
                  } else {
                    z[gidx] -= dt * u;
                  }
                }
              }
            }
          }
        }
      }
    }
    if (tiled) {
      for (std::size_t j = 0; j < n_lat; ++j) {
        if (wacc[j] > 0.0f) { z[j] -= dt * (uacc[j] / wacc[j]); }
      }
    }
    if (req.progress && !req.progress(i + 1, steps)) {
      return fail("cancelled");
    }
  }
  (void)D;
  *out = std::move(z);
  return true;
}

}  // namespace genai
}  // namespace vpipe

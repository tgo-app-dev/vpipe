#include "generative-models/krea2/metal-krea2-vae.h"

#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/weight-set.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <tuple>
#include <vector>

namespace vpipe {
namespace genai {

using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;
using metal_compute::ComputeEncoder;
using metal_compute::CommandStream;

namespace {

// Namespace for this class's derived-tensor cache keys. A WeightSet is
// shared by everything reading one checkpoint, so a key has to say which
// class's transform produced the bytes, not just which tensor they came
// from.
constexpr const char* kKey = "krea2-vae/";

// Read a raw checkpoint tensor as float (F32/F16/BF16 source).
std::vector<float>
read_f32_(const MetalLlamaWeights& wts, MetalCompute* mc, const std::string& nm,
          std::size_t& n_out)
{
  const auto* info = wts.info(nm);
  std::vector<float> v;
  if (info == nullptr || info->shape.empty()) { n_out = 0; return v; }
  std::size_t n = 1;
  for (auto d : info->shape) { n *= (std::size_t)d; }
  SharedBuffer raw = wts.load(nm, mc);
  if (raw.empty()) { n_out = 0; return v; }
  v.resize(n);
  if (info->dtype == "F32") {
    std::memcpy(v.data(), raw.contents(), n * 4);
  } else if (info->dtype == "F16") {
    const auto* s = static_cast<const _Float16*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) { v[i] = (float)s[i]; }
  } else if (info->dtype == "BF16") {
    const auto* s = static_cast<const std::uint16_t*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) {
      std::uint32_t u = (std::uint32_t)s[i] << 16;
      float f; std::memcpy(&f, &u, 4); v[i] = f;
    }
  } else {
    n_out = 0; return {};
  }
  n_out = n;
  return v;
}

SharedBuffer
f16_buf_(MetalCompute* mc, const float* src, std::size_t n)
{
  SharedBuffer b = mc->make_shared_buffer(n * 2);
  if (b.empty()) { return {}; }
  auto* d = static_cast<_Float16*>(b.contents());
  for (std::size_t i = 0; i < n; ++i) { d[i] = (_Float16)src[i]; }
  return b;
}

}  // namespace

// Load a 3x3 conv as a dense-gemm weight [Cout, 9*Cin], flattened (ky,kx,cin)
// to pair with im2col_hwc_3x3. `from3d` sources a [Cout,Cin,3,3,3] causal
// conv3d and keeps only the kt=2 temporal slice (single-frame decode); else a
// [Cout,Cin,3,3] 2D conv (the upsample resample.1).
MetalKrea2Vae::Conv
MetalKrea2Vae::load_conv3x3_(WeightSet& ws, const std::string& nm,
                             bool from3d)
{
  Conv c;
  const auto* info = ws.src().info(nm + ".weight");
  if (info == nullptr || info->shape.empty()) { return c; }
  const auto& sh = info->shape;
  const int Cout = (int)sh[0];
  const int Cin = (int)sh[1];
  const int kt = from3d ? (int)sh[2] : 1;   // 3 for conv3d, absent for conv2d
  const int kty = from3d ? (kt - 1) : 0;    // kt=2 slice index (last)
  c.cin = Cin; c.cout = Cout; c.k = 9 * Cin;

  // The flattened [Cout, 9*Cin] weight and its HWIO twin are DERIVED --
  // built from the checkpoint's bytes, not copied out of them -- so they
  // are cached under keys that name the transform AND the layout it read
  // (3d vs 2d source), never the tensor name alone.
  const std::string k3 = std::string(kKey) + "c3x3|" + nm +
                         (from3d ? "|3d" : "|2d");
  std::vector<float> flat;
  auto build_flat = [&]() {
    std::size_t n = 0;
    std::vector<float> w = read_f32_(ws.src(), _mc, nm + ".weight", n);
    if (w.empty()) { return; }
    flat.assign((std::size_t)Cout * 9 * Cin, 0.0f);
    for (int o = 0; o < Cout; ++o) {
      for (int ky = 0; ky < 3; ++ky) {
        for (int kx = 0; kx < 3; ++kx) {
          for (int i = 0; i < Cin; ++i) {
            std::size_t si;
            if (from3d) {
              // [Cout,Cin,3,3,3]: (o,i,kt,ky,kx)
              si = (((((std::size_t)o * Cin + i) * 3 + kty) * 3 + ky) * 3)
                   + kx;
            } else {
              // [Cout,Cin,3,3]: (o,i,ky,kx)
              si = ((((std::size_t)o * Cin + i) * 3 + ky) * 3) + kx;
            }
            const std::size_t di =
                ((std::size_t)o * 9 + (ky * 3 + kx)) * Cin + i;
            flat[di] = w[si];
          }
        }
      }
    }
  };
  c.w = ws.derived(k3, [&]() -> SharedBuffer {
    build_flat();
    if (flat.empty()) { return {}; }
    return f16_buf_(_mc, flat.data(), flat.size());
  }, _part);
  if (c.w.empty()) { return Conv{}; }
  // HWIO twin for the NAX hardware conv (out-channel fastest). `flat` is
  // [Cout, 9*Cin] with (ky,kx,ci) columns -- permute from it directly so
  // the conv3d slice handling above is inherited.
  // The HWIO twin also feeds conv3x3_small_cout_, which runs on any GPU -- so
  // build it for a small-cout shape even where the hardware conv is off,
  // otherwise that kernel loads and then declines every call on an empty
  // c.whwio.
  const bool want_small_cout_twin =
      _fn_conv_small_cout.valid() && Cout > 0 && Cout <= kSmallCoutMax;
  if (_use_hwconv || want_small_cout_twin) {
    c.whwio = ws.derived(k3 + "|hwio", [&]() -> SharedBuffer {
      // A cache hit on c.w above skips build_flat(), so re-run it here
      // when this twin is the miss.
      if (flat.empty()) { build_flat(); }
      if (flat.empty()) { return {}; }
      std::vector<float> hwio((std::size_t)9 * Cin * Cout);
      for (int o = 0; o < Cout; ++o) {
        for (int t = 0; t < 9; ++t) {
          for (int i = 0; i < Cin; ++i) {
            hwio[((std::size_t)t * Cin + i) * Cout + o] =
                flat[((std::size_t)o * 9 + t) * Cin + i];
          }
        }
      }
      return f16_buf_(_mc, hwio.data(), hwio.size());
    }, _part);
  }
  c.b = load_vec_(ws, nm + ".bias");
  return c;
}

// Load a 1x1 conv ([Cout,Cin,1,1,1] or [Cout,Cin,1,1]) as dense-gemm weight
// [Cout, Cin] (the trailing singleton dims flatten away).
// Pick and load the wide-query-tile mid-attention kernel. The mid-block
// attention is BANDWIDTH-bound: every threadgroup walks the whole of K and V,
// so the traffic is ceil(hw/BQ) * 2 * hw * D * 2 bytes and BQ is the only term
// that moves it. See sdpa_mma.metal for the tile sweep behind kAttnBq and for
// why BQ > 8 needs the register accumulator.
void MetalKrea2Vae::load_wide_attn_(int mid_d)
{
  // All three tiles load; autotune_mid_attn_ picks between them.
  const std::string base = "sdpa_full_mma2_d" + std::to_string(mid_d) + "_q";
  _fn_sdpa_full_wide16 = _lib_sdpa_mma.function(base + "16_f16");
  _fn_sdpa_full_wide32 = _lib_sdpa_mma.function(base + "32_f16");
  _fn_sdpa_full_wide64 = _lib_sdpa_mma.function(base + "64_f16");
}

bool
MetalKrea2Vae::mid_attn_available_(MidAttn k) const
{
  switch (k) {
    case MidAttn::kScalar: return _fn_sdpa.valid();
    case MidAttn::kSmm:    return _fn_sdpa_full_smm.valid();
    case MidAttn::kMma8:   return _fn_sdpa_full_mma.valid();
    case MidAttn::kWide16: return _fn_sdpa_full_wide16.valid();
    case MidAttn::kWide32: return _fn_sdpa_full_wide32.valid();
    case MidAttn::kWide64: return _fn_sdpa_full_wide64.valid();
    case MidAttn::kMat:    return false;   // no materialized path in this VAE
  }
  return false;
}

// Every member speaks one buffer contract; only the grid differs.
void
MetalKrea2Vae::encode_mid_attn_(ComputeEncoder& enc, MidAttn kind,
                                const SharedBuffer& q, const SharedBuffer& k,
                                const SharedBuffer& v, const SharedBuffer& att,
                                std::size_t hw, int C, float scale)
{
  enc.set_buffer(0, q); enc.set_buffer(1, k); enc.set_buffer(2, v);
  enc.set_buffer(3, att);
  enc.set_constant(4, scale); enc.set_constant(5, (int)hw);
  enc.set_constant(6, C); enc.set_constant(7, 1); enc.set_constant(8, 1);
  enc.set_constant(9, (int)hw); enc.set_constant(10, (int)hw);
  switch (kind) {
    case MidAttn::kWide16:
    case MidAttn::kWide32:
    case MidAttn::kWide64: {
      const int bq = (kind == MidAttn::kWide16) ? 16
                   : (kind == MidAttn::kWide32) ? 32 : 64;
      enc.set_function(kind == MidAttn::kWide16 ? _fn_sdpa_full_wide16
                     : kind == MidAttn::kWide32 ? _fn_sdpa_full_wide32
                                                : _fn_sdpa_full_wide64);
      const unsigned nt = attn_threads_(bq);
      enc.dispatch({nt, 1, (unsigned)(((int)hw + bq - 1) / bq)}, {nt, 1, 1});
      break;
    }
    case MidAttn::kMma8:
      enc.set_function(_fn_sdpa_full_mma);
      enc.dispatch({4 * 32, 1, (unsigned)((hw + 7) / 8)}, {4 * 32, 1, 1});
      break;
    case MidAttn::kSmm: {
      enc.set_function(_fn_sdpa_full_smm);
      const unsigned nt = 4u * (unsigned)(C / 64) * 32u;   // WM*WD*32
      enc.dispatch({nt, 1, (unsigned)((hw + 31) / 32)}, {nt, 1, 1});
      break;
    }
    default:
      enc.set_function(_fn_sdpa);                    // scalar O(N^2)
      enc.dispatch({32, 1, (unsigned)hw}, {32, 1, 1});
      break;
  }
}

void
MetalKrea2Vae::autotune_mid_attn_(MetalCompute* mc, int C)
{
  // Capability guess, kept when the tune is skipped or an override names a
  // member outright.
  _attn_pick = MidAttn::kScalar;
  const bool smm_ok = mid_attn_available_(MidAttn::kSmm) &&
                      (C % 64 == 0) && (C <= 512);
  if (smm_ok) { _attn_pick = MidAttn::kSmm; }
  if (mc->supports_matrix_cores()) {
    if (mid_attn_available_(MidAttn::kMma8))   { _attn_pick = MidAttn::kMma8; }
    if (mid_attn_available_(MidAttn::kWide32)) { _attn_pick = MidAttn::kWide32; }
  }
  if (std::getenv("VPIPE_KREA2_VAE_ATTN_SMM") != nullptr && smm_ok) {
    _attn_pick = MidAttn::kSmm; return;
  }
  if (const char* e = std::getenv("VPIPE_KREA2_VAE_ATTN_BQ")) {
    const int b = std::atoi(e);
    if (b == 8  && mid_attn_available_(MidAttn::kMma8))   { _attn_pick = MidAttn::kMma8; }
    if (b == 16 && mid_attn_available_(MidAttn::kWide16)) { _attn_pick = MidAttn::kWide16; }
    if (b == 32 && mid_attn_available_(MidAttn::kWide32)) { _attn_pick = MidAttn::kWide32; }
    if (b == 64 && mid_attn_available_(MidAttn::kWide64)) { _attn_pick = MidAttn::kWide64; }
    return;
  }
  std::vector<MidAttn> cands;
  for (MidAttn k : {MidAttn::kSmm, MidAttn::kMma8, MidAttn::kWide16,
                    MidAttn::kWide32, MidAttn::kWide64}) {
    if (k == MidAttn::kSmm && !smm_ok) { continue; }
    if (mid_attn_available_(k)) { cands.push_back(k); }
  }
  std::string detail;
  const auto t0 = std::chrono::steady_clock::now();
  _attn_pick = vae_mid_attn::autotune<MetalCompute, ComputeEncoder>(
      mc, C, cands, _attn_pick,
      [this](ComputeEncoder& enc, MidAttn kind, const SharedBuffer& qq,
             const SharedBuffer& kk, const SharedBuffer& vv,
             const SharedBuffer& oo, std::size_t hw, int c, float sc,
             const vae_mid_attn::Alloc&, const vae_mid_attn::Release&) {
        encode_mid_attn_(enc, kind, qq, kk, vv, oo, hw, c, sc);
      },
      &detail);
  const double tune_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
  if (mc->session() != nullptr && !detail.empty()) {
    // VPIPE_VAE_ATTN_TUNE_LOG promotes this to a normal log line. Which member
    // won is the whole point of the tune and differs per machine, so it has to
    // be observable without a debug build when porting to a new GPU.
    const auto line = fmt(
        "MetalKrea2Vae: mid-attn autotune (D={}) -> {} [{}] in {} ms",
        C, vae_mid_attn::name(_attn_pick), detail, (long long)tune_ms);
    if (std::getenv("VPIPE_VAE_ATTN_TUNE_LOG") != nullptr) {
      mc->session()->log_normal(line);
    } else {
      mc->session()->log_debug(line);
    }
  }
}


MetalKrea2Vae::Conv
MetalKrea2Vae::load_conv1x1_(WeightSet& ws, const std::string& nm)
{
  Conv c;
  const auto* info = ws.src().info(nm + ".weight");
  if (info == nullptr || info->shape.size() < 2) { return c; }
  const auto& sh = info->shape;
  c.cout = (int)sh[0]; c.cin = (int)sh[1]; c.k = c.cin;
  c.w = load_vec_(ws, nm + ".weight");
  if (c.w.empty()) { return Conv{}; }
  c.b = load_vec_(ws, nm + ".bias");
  return c;
}

// Every scalar/vector/matrix tensor the VAE keeps is stored as f16
// regardless of its on-disk dtype, so "read it as f32 and narrow" IS the
// transform and the result is cached like any other derived tensor.
SharedBuffer
MetalKrea2Vae::load_vec_(WeightSet& ws, const std::string& nm)
{
  return ws.derived(std::string(kKey) + "f16|" + nm, [&]() -> SharedBuffer {
    std::size_t n = 0;
    std::vector<float> v = read_f32_(ws.src(), _mc, nm, n);
    if (v.empty()) { return {}; }
    return f16_buf_(_mc, v.data(), n);
  }, _part);
}

bool
MetalKrea2Vae::load_resblock_(WeightSet& wts,
                              const std::string& pre, ResBlock& rb, int cin,
                              int cout)
{
  rb.cin = cin; rb.cout = cout;
  rb.n1g = load_vec_(wts, pre + "norm1.gamma");
  rb.n2g = load_vec_(wts, pre + "norm2.gamma");
  rb.c1 = load_conv3x3_(wts, pre + "conv1", true);
  rb.c2 = load_conv3x3_(wts, pre + "conv2", true);
  rb.has_short = (cin != cout);
  if (rb.has_short) { rb.shortcut = load_conv1x1_(wts, pre + "conv_shortcut"); }
  return !rb.n1g.empty() && !rb.n2g.empty() && !rb.c1.w.empty() &&
         !rb.c2.w.empty() && (!rb.has_short || !rb.shortcut.w.empty());
}

std::unique_ptr<MetalKrea2Vae>
MetalKrea2Vae::load(const std::string& model_dir, MetalCompute* mc,
                    const Config& cfg, bool with_encoder)
{
  return load(WeightSet::open(model_dir, nullptr), mc, cfg, with_encoder);
}

std::unique_ptr<MetalKrea2Vae>
MetalKrea2Vae::load(std::shared_ptr<WeightSet> ws_in, MetalCompute* mc,
                    const Config& cfg, bool with_encoder)
{
  if (mc == nullptr || !ws_in) { return nullptr; }
  WeightSet& wts = *ws_in;

  auto m = std::unique_ptr<MetalKrea2Vae>(new MetalKrea2Vae());
  m->_ws = std::move(ws_in);
  m->_mc = mc;
  m->_cfg = cfg;

  m->_lib_gemm = mc->load_library("dense_gemm");
  m->_lib_elt  = mc->load_library("llm_elementwise");
  m->_lib_rms  = mc->load_library("rms_norm");
  m->_lib_sdpa = mc->load_library("sdpa");
  m->_fn_gemm_bias   = m->_lib_gemm.function("dense_gemm_bias_f16");
  m->_fn_rms         = m->_lib_rms.function("rms_norm_fast_f16");
  m->_fn_mul_sigmoid = m->_lib_elt.function("mul_sigmoid_f16");
  m->_fn_residual    = m->_lib_elt.function("residual_add_f16");
  m->_fn_clamp       = m->_lib_elt.function("clamp_f16");
  m->_fn_sdpa        = m->_lib_sdpa.function("sdpa_full_f16");
  m->_fn_sdpa_full_smm = m->_lib_sdpa.function("sdpa_full_mma_f16");
  // Direct small-cout 3x3 (see conv3x3_small_cout_). Loaded HERE, ahead of the
  // first load_conv3x3_, because that decides whether to build the HWIO twin
  // this kernel reads.
  m->_fn_conv_small_cout =
      std::getenv("VPIPE_VAE_NO_SMALL_COUT_CONV") == nullptr
          ? m->_lib_elt.function("conv3x3_hwc_small_cout_f16")
          : metal_compute::ComputeFunction{};
  m->_fn_im2col      = m->_lib_elt.function("im2col_hwc_3x3_f16");
  m->_fn_im2col_s2   = m->_lib_elt.function("im2col_hwc_3x3_s2_f16");
  m->_fn_im2col_tiled = m->_lib_elt.function("im2col_hwc_3x3_tiled_f16");
  m->_fn_im2col_s2_tiled =
      m->_lib_elt.function("im2col_hwc_3x3_s2_tiled_f16");
  m->_fn_upsample    = m->_lib_elt.function("upsample_nearest2x_hwc_f16");
  if (!m->_fn_gemm_bias.valid() || !m->_fn_rms.valid() ||
      !m->_fn_mul_sigmoid.valid() || !m->_fn_residual.valid() ||
      !m->_fn_clamp.valid() || !m->_fn_sdpa.valid() ||
      !m->_fn_im2col.valid() || !m->_fn_im2col_s2.valid() ||
      !m->_fn_im2col_tiled.valid() || !m->_fn_im2col_s2_tiled.valid() ||
      !m->_fn_upsample.valid()) {
    return nullptr;
  }
  // M5 matrix-core dense GEMM (matmul2d) for the conv/1x1 GEMMs, mirroring the
  // LM + gemma-vision NAX path. The VAE runs at large M (M = H*W pixels), so the
  // tiled matmul2d amortizes well. bias is folded by a separate bias_add_rows
  // pass (the mma kernel has no bias slot). Steel is kept as the fallback.
  if (mc->supports_matrix_cores() &&
      std::getenv("VPIPE_KREA2_NO_MMA2") == nullptr) {
    m->_lib_dense_mma = mc->load_library("dense_gemm_mma");
    m->_fn_dense_mma = m->_lib_dense_mma.function("dense_gemm_mma_t_n128_f16");
    m->_fn_dense_mma_deep =
        m->_lib_dense_mma.function("dense_gemm_mma_t_n128x256_f16");
    m->_fn_bias_add = m->_lib_elt.function("bias_add_rows_f16");
    m->_use_mma2 = m->_fn_dense_mma.valid() && m->_fn_dense_mma_deep.valid() &&
                   m->_fn_bias_add.valid();
    if (const char* e = std::getenv("VPIPE_KREA2_MMA_MIN_M")) {
      m->_mma_min_m = std::atoi(e);
    }
    if (const char* e = std::getenv("VPIPE_KREA2_VAE_MMA_MAX_M")) {
      m->_mma_max_m = std::atoi(e);   // 0 => no upper cap (A/B the corruption)
    }
    if (const char* e = std::getenv("VPIPE_KREA2_VAE_MMA_MIN_N")) {
      m->_mma_min_n = std::atoi(e);   // route tiny-N GEMMs (conv_out) to steel
    }
    // Fused 3x3 conv2d (matmul2d over threadgroup-staged im2col): bit-identical
    // to im2col + matmul2d (krea2_vae.decode_conv2d_vs_im2col). It used to be
    // opt-in on the strength of a single whole-decode A/B that called it ~1.7x
    // slower on M5 -- but that measurement ran it on EVERY conv, and the answer
    // is not uniform: the round trip it skips scales with 9*cin while the halo
    // it re-gathers scales with cout, so a narrow-cout conv can win where a
    // wide one loses badly. The kernels are therefore loaded whenever the matrix
    // units are there, and autotune_conv3x3_ picks per shape at first use.
    // Needs bias_add (_use_mma2) for the bias fold.
    if (m->_use_mma2 && std::getenv("VPIPE_KREA2_NO_CONV2D") == nullptr) {
      m->_lib_conv2d = mc->load_library("conv2d_mma");
      m->_fn_conv2d_s1 = m->_lib_conv2d.function("conv2d_mma_3x3_s1_f16");
      m->_fn_conv2d_s2 = m->_lib_conv2d.function("conv2d_mma_3x3_s2_f16");
      m->_use_conv2d = m->_fn_conv2d_s1.valid() && m->_fn_conv2d_s2.valid();
    }
  }
  // FULL matrix-core flash-attention for the mid-block self-attention (head_dim
  // = mid channel dim = base*dim_mult[-1]); replaces the scalar O(N^2)
  // sdpa_full_f16 that dominates decode at high res. Loaded independently of
  // supports_matrix_cores(): the MPP matmul2d op EMULATES on pre-M5 GPUs (see
  // metal-compute.h), so this tiled flash beats the scalar O(N^2) path there
  // too (the steel flash the DiT uses on M4 tops out at head_dim 256, below the
  // VAE's 384/512). `_fn_sdpa_full_mma.valid()` is the real gate -- a GPU that
  // can't run the metal4.0 tensor kernel keeps the scalar fallback.
  // VPIPE_KREA2_NO_MMA_ATTN forces scalar.
  if (std::getenv("VPIPE_KREA2_NO_MMA_ATTN") == nullptr) {
    const int mid_d = cfg.base_dim * cfg.dim_mult[3];
    const char* fn = (mid_d == 384) ? "sdpa_full_mma2_d384_f16"
                   : (mid_d == 512) ? "sdpa_full_mma2_d512_f16"
                                    : nullptr;
    if (fn != nullptr) {
      m->_lib_sdpa_mma = mc->load_library("sdpa_mma");
      m->_fn_sdpa_full_mma = m->_lib_sdpa_mma.function(fn);
      m->load_wide_attn_(mid_d);
    }
    m->autotune_mid_attn_(mc, mid_d);
    // Prefer matmul2d flash only where the matrix units make it worthwhile
    // (M5); on M4/older the simdgroup_matrix flash (_fn_sdpa_full_smm) wins.
    m->_use_attn_mma2 = mc->supports_matrix_cores();
  }
  // NAX hardware convolution2d for the 3x3 convs (see conv3x3_hw_); decided
  // BEFORE the weights load so load_conv3x3_ builds the HWIO twins. Needs
  // _fn_bias_add for the bias fold, so load it here too when the mma block
  // above was disabled (steel A/B).
  if (mc->supports_matrix_cores() &&
      std::getenv("VPIPE_VAE_NO_HWCONV") == nullptr) {
    m->_lib_convhw = mc->load_library("conv2d_mma");
    m->_fn_conv_hw_s1 = m->_lib_convhw.function("conv2d_hw_3x3_s1_f16");
    m->_fn_conv_hw_s2 = m->_lib_convhw.function("conv2d_hw_3x3_s2_f16");
    if (!m->_fn_bias_add.valid()) {
      m->_fn_bias_add = m->_lib_elt.function("bias_add_rows_f16");
    }
    m->_use_hwconv = m->_fn_conv_hw_s1.valid() &&
                     m->_fn_conv_hw_s2.valid() && m->_fn_bias_add.valid();
  }

  const int base = cfg.base_dim;                         // 96
  // Decoder channel dims: [dim*dim_mult[-1]] + dim*dim_mult[::-1].
  const int dims0 = base * cfg.dim_mult[3];              // 384

  m->_post_quant = m->load_conv1x1_(wts, "post_quant_conv");
  m->_conv_in    = m->load_conv3x3_(wts, "decoder.conv_in", true);

  bool ok = !m->_post_quant.w.empty() && !m->_conv_in.w.empty();
  ok = ok && m->load_resblock_(wts, "decoder.mid_block.resnets.0.",
                               m->_mid_res0, dims0, dims0);
  ok = ok && m->load_resblock_(wts, "decoder.mid_block.resnets.1.",
                               m->_mid_res1, dims0, dims0);
  // mid-block attention (single head over the spatial map).
  m->_mid_attn.dim = dims0;
  m->_mid_attn.ng = m->load_vec_(wts, "decoder.mid_block.attentions.0.norm.gamma");
  {
    // to_qkv is one 1x1 conv [3*dim, dim]; split output channels into q/k/v.
    std::size_t n = 0, nb = 0;
    std::vector<float> qkv, qkvb;
    const std::string qbase = "decoder.mid_block.attentions.0.to_qkv";
    auto read_qkv = [&]() {
      if (!qkv.empty()) { return; }
      qkv  = read_f32_(wts.src(), mc, qbase + ".weight", n);
      qkvb = read_f32_(wts.src(), mc, qbase + ".bias", nb);
    };
    const int C = dims0;
    {
      // Each third of the fused to_qkv is its own derived tensor, keyed
      // by which third it is.
      auto slice = [&](int off) {
        Conv c; c.cin = C; c.cout = C; c.k = C;
        const std::string k = std::string(kKey) + "qkv|" + qbase + "|" +
                              std::to_string(off);
        c.w = wts.derived(k + "|w", [&]() -> SharedBuffer {
          read_qkv();
          if (qkv.size() != (std::size_t)3 * C * C) { return {}; }
          return f16_buf_(mc, qkv.data() + (std::size_t)off * C * C,
                          (std::size_t)C * C);
        }, m->_part);
        c.b = wts.derived(k + "|b", [&]() -> SharedBuffer {
          read_qkv();
          if (qkvb.size() != (std::size_t)3 * C) { return {}; }
          return f16_buf_(mc, qkvb.data() + (std::size_t)off * C,
                          (std::size_t)C);
        }, m->_part);
        return c;
      };
      m->_mid_attn.q = slice(0);
      m->_mid_attn.k = slice(1);
      m->_mid_attn.v = slice(2);
    }
  }
  m->_mid_attn.proj = m->load_conv1x1_(wts, "decoder.mid_block.attentions.0.proj");
  ok = ok && !m->_mid_attn.ng.empty() && !m->_mid_attn.q.w.empty() &&
       !m->_mid_attn.k.w.empty() && !m->_mid_attn.v.w.empty() &&
       !m->_mid_attn.proj.w.empty();

  // Upsample blocks. dims = [384, 384, 384, 192, 96]; up i maps dims[i]->dims[i+1]
  // (for i>0 the resnet in_dim is halved by the previous upsample conv), with an
  // upsample conv (out -> out/2, spatial 2x) for i != 3.
  const int dims[5] = {dims0, base * cfg.dim_mult[3], base * cfg.dim_mult[2],
                       base * cfg.dim_mult[1], base * cfg.dim_mult[0]};
  m->_up_blocks.resize(4);
  for (int i = 0; i < 4; ++i) {
    UpBlock& ub = m->_up_blocks[(std::size_t)i];
    int in_dim = dims[i];
    if (i > 0) { in_dim = in_dim / 2; }
    const int out_dim = dims[i + 1];
    ub.resnets.resize((std::size_t)cfg.num_res_blocks + 1);
    int cin = in_dim;
    for (int r = 0; r <= cfg.num_res_blocks; ++r) {
      ok = ok && m->load_resblock_(
          wts, "decoder.up_blocks." + std::to_string(i) + ".resnets." +
                   std::to_string(r) + ".",
          ub.resnets[(std::size_t)r], cin, out_dim);
      cin = out_dim;
    }
    ub.has_up = (i != 3);
    if (ub.has_up) {
      ub.up_dim = out_dim;
      ub.up = m->load_conv3x3_(
          wts, "decoder.up_blocks." + std::to_string(i) + ".upsamplers.0.resample.1",
          false);
      ok = ok && !ub.up.w.empty();
    }
  }

  m->_norm_out_g = m->load_vec_(wts, "decoder.norm_out.gamma");
  m->_conv_out = m->load_conv3x3_(wts, "decoder.conv_out", true);
  ok = ok && !m->_norm_out_g.empty() && !m->_conv_out.w.empty();

  if (!ok) { return nullptr; }
  if (with_encoder && !m->ensure_encoder()) { return nullptr; }
  return m;
}

bool
MetalKrea2Vae::ensure_encoder()
{
  if (_has_encoder) { return true; }
  if (!_ws) { return false; }
  // Through the WeightSet so the encoder half loads ONCE per checkpoint
  // however many VAEs over it end up needing it -- and, when nobody
  // does, never.
  const bool ok = _ws->ensure_part("encoder", [this]() {
    _part = "encoder";
    const bool r = load_encoder_(*_ws);
    _part.clear();
    return r;
  });
  if (!ok) { return false; }
  // A second VAE over a checkpoint whose encoder another one already
  // loaded still has to populate ITS OWN struct members; ensure_part
  // reports the cached success without re-running the loader, so bind
  // the (now cache-hit) tensors here.
  if (_enc_conv_in.w.empty()) {
    _part = "encoder";
    const bool r = load_encoder_(*_ws);
    _part.clear();
    if (!r) { return false; }
  }
  _has_encoder = true;
  return true;
}

SharedBuffer
MetalKrea2Vae::unwhiten(const SharedBuffer& z, int h8, int w8)
{
  const int C = _cfg.z_dim;
  const std::size_t hw = (std::size_t)h8 * w8;
  if ((int)_cfg.latents_mean.size() != C ||
      (int)_cfg.latents_std.size() != C ||
      z.byte_size() < (std::size_t)C * hw * 2) {
    return {};
  }
  SharedBuffer out = _mc->make_shared_buffer((std::size_t)C * hw * 2);
  const auto* s = static_cast<const _Float16*>(z.contents());
  auto* d = static_cast<_Float16*>(out.contents());
  for (int c = 0; c < C; ++c) {
    const float mu = _cfg.latents_mean[(std::size_t)c];
    const float sd = _cfg.latents_std[(std::size_t)c];
    for (std::size_t p = 0; p < hw; ++p) {
      const std::size_t i = (std::size_t)c * hw + p;
      d[i] = (_Float16)((float)s[i] * sd + mu);
    }
  }
  return out;
}

void
MetalKrea2Vae::conv_gemm_bias_(ComputeEncoder& enc, const SharedBuffer& x,
                               const SharedBuffer& w, const SharedBuffer& b,
                               const SharedBuffer& y, int M, int N, int K,
                               int y_row0)
{
  // y[M,N] = x[M,K] @ w[N,K]^T (+ bias[N]), written into y at output row y_row0
  // (a row-tiled im2col conv feeds one band into out[y_row0:y_row0+M]). M5
  // matmul2d for tall M, else steel.
  const std::size_t ybase = (std::size_t)y_row0 * N;   // element offset into y
  if (_use_mma2 && M >= _mma_min_m && N >= _mma_min_n) {
    // Tile-adaptive matrix-core dense GEMM (no bias): 128x128 for K < 6144,
    // 128x256 for deeper K. The matmul2d tensor extents clamp M/N tails, so M
    // and N need not be tile multiples.
    const bool deep = (K >= 6144);
    const int BN = deep ? 256 : 128;
    // The MPP matmul2d op silently corrupts output rows past M ~= 2^19 (a
    // >=1024px decode has M = H*W = 2^20 and went grey from GEMM row 2^19), so
    // split a tall GEMM into row-chunks of at most _mma_max_m and dispatch each
    // over its own row-range. The dense_gemm_mma tensors are column-major (the
    // row dim's stride is K for x / N for y), so a contiguous r0*K / r0*N
    // ELEMENT offset (x2 for f16 bytes) selects rows [r0, r0+mc) exactly.
    // _mma_max_m == 0 disables the split (single dispatch; A/B the corruption).
    const int chunk = (_mma_max_m > 0 && M > _mma_max_m) ? _mma_max_m : M;
    for (int r0 = 0; r0 < M; r0 += chunk) {
      const int mc = (M - r0 < chunk) ? (M - r0) : chunk;
      enc.set_function(deep ? _fn_dense_mma_deep : _fn_dense_mma);
      enc.set_buffer(0, x, (std::size_t)r0 * K * 2);
      enc.set_buffer(1, w);
      enc.set_buffer(2, w);        // bias slot unused (has_bias=0)
      enc.set_buffer(3, y, (ybase + (std::size_t)r0 * N) * 2);
      enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, mc);
      enc.set_constant(7, 0);
      enc.dispatch({(unsigned)(((N + BN - 1) / BN) * 256),
                    (unsigned)((mc + 127) / 128), 1}, {256, 1, 1});
    }
    if (!b.empty()) {
      // Fold bias[N] across ALL M rows at once (a plain kernel, no M limit).
      const std::size_t total = (std::size_t)M * N;
      enc.set_function(_fn_bias_add);
      enc.set_buffer(0, y, ybase * 2); enc.set_buffer(1, b);
      enc.set_constant(2, N); enc.set_constant(3, (unsigned)total);
      // 2D grid {N, M}: gid = row*N + col (a 1D {M*N} grid overflows past ~2K).
      enc.dispatch({(unsigned)N, (unsigned)M, 1}, {256, 1, 1});
    }
    return;
  }
  enc.set_function(_fn_gemm_bias);
  enc.set_buffer(0, x); enc.set_buffer(1, w);
  enc.set_buffer(2, b.empty() ? w : b); enc.set_buffer(3, y, ybase * 2);
  enc.set_constant(4, M); enc.set_constant(5, N); enc.set_constant(6, K);
  enc.set_constant(7, b.empty() ? 0 : 1);
  enc.dispatch({(unsigned)(((N + 15) / 16) * 16),
                (unsigned)(((M + 15) / 16) * 16), 1}, {16, 16, 1});
}

void
MetalKrea2Vae::tiled_conv3x3_(ComputeEncoder& enc, const SharedBuffer& in,
                              const SharedBuffer& out, int H, int W,
                              const Conv& c, int stride, const SharedBuffer& col,
                              std::size_t cap)
{
  const int OH = (stride == 2) ? H / 2 : H;
  const int OW = (stride == 2) ? W / 2 : W;
  const std::size_t ohw = (std::size_t)OH * OW;
  const std::size_t per_row = (std::size_t)9 * c.cin;
  std::size_t tile_rows = (per_row > 0) ? (cap / per_row) : ohw;
  // Safe-cap so each band's GEMM is one un-chunked matmul2d under the
  // M-corruption threshold (measured 2^18 clean / 2^19 banded for large-K 3x3).
  if (_mma_max_m > 0 && tile_rows > (std::size_t)_mma_max_m / 2) {
    tile_rows = (std::size_t)_mma_max_m / 2;
  }
  if (tile_rows > ohw) { tile_rows = ohw; }
  if (tile_rows < 1) { tile_rows = 1; }
  const metal_compute::ComputeFunction& fn =
      (stride == 2) ? _fn_im2col_s2_tiled : _fn_im2col_tiled;
  // The whole [H*W, cin] input stays live so the 3x3 halo spans band edges; the
  // shared `col` is reused across bands (serial dispatch orders band b's GEMM
  // before band b+1 overwrites col).
  for (std::size_t r0 = 0; r0 < ohw; r0 += tile_rows) {
    const int mc = (int)std::min(tile_rows, ohw - r0);
    enc.set_function(fn);
    enc.set_buffer(0, in); enc.set_buffer(1, col);
    enc.set_constant(2, H); enc.set_constant(3, W); enc.set_constant(4, c.cin);
    enc.set_constant(5, (int)r0); enc.set_constant(6, mc);
    enc.dispatch({(unsigned)(9 * c.cin), (unsigned)mc, 1}, {64, 1, 1});
    conv_gemm_bias_(enc, col, c.w, c.b, out, mc, c.cout, c.k, (int)r0);
  }
}

bool
MetalKrea2Vae::conv3x3_small_cout_(ComputeEncoder& enc, const SharedBuffer& in,
                                   const Conv& c, const SharedBuffer& out,
                                   int H, int W, int stride)
{
  // stride 1 only: the stride-2 downsample uses the asymmetric (0,1,0,1)
  // padding, and every small-cout conv in this VAE is stride 1 anyway.
  if (!_fn_conv_small_cout.valid() || c.whwio.empty() || stride != 1) {
    return false;
  }
  if (c.cout <= 0 || c.cout > kSmallCoutMax || c.cin <= 0) { return false; }
  const int has_bias = c.b.empty() ? 0 : 1;
  enc.set_function(_fn_conv_small_cout);
  enc.set_buffer(0, in); enc.set_buffer(1, c.whwio);
  // Every buffer slot must be bound even when unread; the kernel gates on
  // has_bias, so point the unused slot at a live buffer rather than nothing.
  enc.set_buffer(2, has_bias != 0 ? c.b : c.whwio);
  enc.set_buffer(3, out);
  enc.set_constant(4, W); enc.set_constant(5, H);
  enc.set_constant(6, c.cin); enc.set_constant(7, c.cout);
  enc.set_constant(8, has_bias);
  enc.dispatch({(unsigned)W, (unsigned)H, 1}, {256, 1, 1});
  return true;
}

bool
MetalKrea2Vae::conv3x3_hw_(ComputeEncoder& enc, const SharedBuffer& in,
                           const Conv& c, const SharedBuffer& out,
                           int H, int W, int stride)
{
  if (!_use_hwconv || c.whwio.empty()) { return false; }
  const int OH = H / stride, OW = W / stride;
  if ((OW % 8) != 0 || (OH % 8) != 0 || (c.cout % 64) != 0) { return false; }
  // The MPP conv op indexes its source/dest through int32 tensor extents; fall
  // back to the (uint-safe) im2col path before cin*W*H or cout*OW*OH would
  // overflow a signed int (~3K px), so a very large decode degrades to im2col
  // instead of silently corrupting.
  constexpr std::size_t kIdxMax = 0x7fffffffull;
  if ((std::size_t)c.cin * W * H > kIdxMax ||
      (std::size_t)c.cout * OW * OH > kIdxMax) {
    return false;
  }
  enc.set_function(stride == 2 ? _fn_conv_hw_s2 : _fn_conv_hw_s1);
  enc.set_buffer(0, in);
  enc.set_buffer(1, c.whwio);
  enc.set_buffer(2, out);
  enc.set_constant(3, W); enc.set_constant(4, H);
  enc.set_constant(5, c.cin); enc.set_constant(6, c.cout);
  if (stride == 2) {
    // Probe-verified offset mode 3: the ASYMMETRIC (0,1,0,1) downsample
    // padding (diffusers Downsample2D / im2col_hwc_3x3_s2 convention).
    enc.set_constant(7, 3);
  }
  enc.dispatch({(unsigned)((OW / 8) * 128), (unsigned)(OH / 8),
                (unsigned)(c.cout / 64)}, {128, 1, 1});
  if (!c.b.empty()) {          // fold bias (the hw op has no bias slot)
    const std::size_t rows = (std::size_t)OH * OW;
    const std::size_t total = rows * c.cout;
    enc.set_function(_fn_bias_add);
    enc.set_buffer(0, out); enc.set_buffer(1, c.b);
    enc.set_constant(2, c.cout); enc.set_constant(3, (unsigned)total);
    // 2D grid {cout, rows}: gid = row*cout + col. A 1D {total} grid overflows
    // both the int index and the grid dimension past ~3K px.
    enc.dispatch({(unsigned)c.cout, (unsigned)rows, 1}, {256, 1, 1});
  }
  return true;
}

bool
MetalKrea2Vae::conv2d_mma_(ComputeEncoder& enc, const SharedBuffer& in,
                           const SharedBuffer& w, const SharedBuffer& b,
                           const SharedBuffer& out, int H, int W, int Cin,
                           int Cout, int OH, int OW, int stride)
{
  if (!_use_conv2d) { return false; }
  // Measured, per shape -- not a capability bit. See vae-conv3x3-tune.h.
  if (conv_route_(Cin, Cout, stride) != vae_conv3x3::Kind::kOnChip) {
    return false;
  }
  // out[OH*OW, Cout] = conv3x3(in[H*W, Cin], w[Cout, 9*Cin]) on the matrix units
  // (im2col staged in threadgroup memory). BM/BN/SG mirror the kernel's tile.
  const int BM = 64, BN = 64, SG = 4, M = OH * OW;
  enc.set_function(stride == 2 ? _fn_conv2d_s2 : _fn_conv2d_s1);
  enc.set_buffer(0, in); enc.set_buffer(1, w); enc.set_buffer(2, out);
  enc.set_constant(3, H); enc.set_constant(4, W); enc.set_constant(5, Cin);
  enc.set_constant(6, Cout); enc.set_constant(7, OH); enc.set_constant(8, OW);
  enc.dispatch({(unsigned)(((Cout + BN - 1) / BN) * SG * 32),
                (unsigned)((M + BM - 1) / BM), 1}, {(unsigned)(SG * 32), 1, 1});
  if (!b.empty()) {              // fold bias[Cout] across the M output pixels
    const std::size_t total = (std::size_t)M * Cout;
    enc.set_function(_fn_bias_add);
    enc.set_buffer(0, out); enc.set_buffer(1, b);
    enc.set_constant(2, Cout); enc.set_constant(3, (unsigned)total);
    // 2D grid {Cout, M}: gid = pixel*Cout + c (1D {M*Cout} overflows past ~2K).
    enc.dispatch({(unsigned)Cout, (unsigned)M, 1}, {256, 1, 1});
  }
  return true;
}

void
MetalKrea2Vae::conv3x3_fallback_(ComputeEncoder& enc, const SharedBuffer& in,
                                 const SharedBuffer& out, int H, int W,
                                 const Conv& c, int stride,
                                 const SharedBuffer& col, std::size_t cap)
{
  const int OH = H / stride, OW = W / stride;
  if (conv2d_mma_(enc, in, c.w, c.b, out, H, W, c.cin, c.cout, OH, OW,
                  stride)) {
    return;
  }
  tiled_conv3x3_(enc, in, out, H, W, c, stride, col, cap);
}

vae_conv3x3::Kind
MetalKrea2Vae::conv_route_(int cin, int cout, int stride) const
{
  if (_conv_force_fused) { return vae_conv3x3::Kind::kOnChip; }
  const auto it = _conv_pick.find(std::make_tuple(cin, cout, stride));
  return (it != _conv_pick.end()) ? it->second : vae_conv3x3::Kind::kIm2col;
}

// Decide, once the resolution is known, WHICH 3x3 shapes will reach the
// fallback at all -- and tune only those.
//
// This VAE is the opposite case from the FLUX.2 one. base_dim 96 is not a
// multiple of 64, so conv3x3_hw_ declines the whole top-resolution level
// (up_blocks[3]'s three 96->96 resblocks, plus the 192->96 upsample conv that
// feeds them) at EVERY resolution -- six full-size convs whose im2col scratch
// decode_peak_bytes already calls the single largest buffer in a decode. So
// the fallback is not a large-decode corner here; it is on the critical path
// of every image. Everything else rides the hardware conv until the resolution
// itself pushes it off (a grid that is not a multiple of 8, or cin*W*H past
// int32), which is what the pyramid walk below checks.
void
MetalKrea2Vae::maybe_tune_conv_(int H, int W, const SharedBuffer& col,
                                std::size_t cap)
{
  if (_mc == nullptr) { return; }
  if (!_conv_tuned) {
    _conv_tuned = true;
    // An explicit force wins outright rather than seeding the map: it has to
    // cover the shapes the tune skips (cout <= kSmallCoutMax, and any the
    // hardware conv takes), or the A/B silently compares im2col with itself.
    // Latched HERE and not re-read per call, so an A/B can unset it between
    // the two legs' first decodes without the first leg flipping back.
    _conv_force_fused = std::getenv("VPIPE_KREA2_CONV2D") != nullptr;
  }
  if (_conv_force_fused || !_use_conv2d) { return; }
  // Does the hardware conv still take the shapes it is eligible for? If not,
  // every 3x3 lands in the fallback and every shape is worth tuning.
  bool all = !_use_hwconv;
  if (!all) {
    constexpr std::size_t kIdxMax = 0x7fffffffull;
    int h = H, w = W;                      // walk the decoder's pyramid
    const int base = _cfg.base_dim;
    for (int lvl = 0; lvl < 4 && !all; ++lvl) {
      const int cin = base * _cfg.dim_mult[lvl];
      if ((h % 8) != 0 || (w % 8) != 0) { all = true; break; }
      if ((std::size_t)cin * w * h > kIdxMax) { all = true; break; }
      h /= 2; w /= 2;
    }
  }
  // Shapes already decided are skipped, so this can run more than once and
  // only ever ADD: the encoder half arrives late (ensure_encoder() may follow
  // the first decode, and its stride-2 downsamples are shapes the decoder does
  // not have), and a later, larger image can push convs off the hardware conv
  // that a smaller one kept on it. Latching after the first call would leave
  // both silently untuned.
  std::vector<std::tuple<int, int, int>> shapes;
  auto want = [&](const Conv& c, int stride) {
    if (c.cin <= 0 || c.cout <= kSmallCoutMax) { return; }
    if (!all && (c.cout % 64) == 0) { return; }   // the hardware conv has it
    const auto key = std::make_tuple(c.cin, c.cout, stride);
    if (_conv_pick.find(key) != _conv_pick.end()) { return; }
    for (const auto& s : shapes) { if (s == key) { return; } }
    shapes.push_back(key);
  };
  auto want_rb = [&](const ResBlock& rb) { want(rb.c1, 1); want(rb.c2, 1); };
  want(_conv_in, 1); want_rb(_mid_res0); want_rb(_mid_res1);
  for (const UpBlock& ub : _up_blocks) {
    for (const ResBlock& rb : ub.resnets) { want_rb(rb); }
    if (ub.has_up) { want(ub.up, 1); }
  }
  if (_has_encoder) {
    want(_enc_conv_in, 1); want_rb(_enc_mid_res0); want_rb(_enc_mid_res1);
    for (const DownStage& ds : _enc_down) {
      for (const ResBlock& rb : ds.resnets) { want_rb(rb); }
      if (ds.has_down) { want(ds.down, 2); }
    }
    want(_enc_conv_out, 1);
  }
  if (shapes.empty()) { return; }
  autotune_conv3x3_(_mc, shapes, col, cap);
}

// Time both fallback routes per shape and keep the winner. The probe runs
// through conv3x3_fallback_ -- the SHIPPING entrance -- with _conv_pick
// steering it, and over the caller's REAL im2col scratch, so what is measured
// is the route (and the banding) a decode would actually take.
void
MetalKrea2Vae::autotune_conv3x3_(MetalCompute* mc,
                                 const std::vector<std::tuple<int, int, int>>&
                                     shapes,
                                 const SharedBuffer& col, std::size_t cap)
{
  // The probe resolution SCALES WITH THE SHAPE. In this U-net channel count
  // and resolution are inversely related -- the 384-channel convs run on the
  // latent grid, the 96-channel ones at full size -- so a single probe
  // resolution is wrong in both directions. Sizing it as dims[-1]/cout tracks
  // the pyramid, and since a U-net holds work roughly constant per level, it
  // keeps every shape's tune about equally cheap.
  int probe_base = 64;
  if (const char* e = std::getenv("VPIPE_VAE_CONV_TUNE_HW")) {
    probe_base = std::max(32, std::atoi(e));
  }
  const int top = _cfg.base_dim * _cfg.dim_mult[3];
  const std::vector<vae_conv3x3::Kind> cands = {vae_conv3x3::Kind::kIm2col,
                                                vae_conv3x3::Kind::kOnChip};
  std::string detail;
  const auto t0 = std::chrono::steady_clock::now();
  for (const auto& sh : shapes) {
    const int cin = std::get<0>(sh), cout = std::get<1>(sh);
    const int stride = std::get<2>(sh);
    int H = std::min(256, std::max(probe_base,
                                   probe_base * (top / std::max(1, cout))));
    H -= H % (8 * stride);                       // whole 8x8 dest tiles
    if (H <= 0) { continue; }
    const int W = H;
    Conv c;
    c.cin = cin; c.cout = cout; c.k = 9 * cin;
    c.w = mc->make_shared_buffer((std::size_t)cout * 9 * cin * 2);
    c.b = mc->make_shared_buffer((std::size_t)cout * 2);
    SharedBuffer in = mc->make_shared_buffer((std::size_t)H * W * cin * 2);
    SharedBuffer out = mc->make_shared_buffer(
        (std::size_t)(H / stride) * (W / stride) * cout * 2);
    if (c.w.empty() || c.b.empty() || in.empty() || out.empty()) {
      continue;                                  // leave this shape on im2col
    }
    {
      auto* pw = static_cast<_Float16*>(c.w.contents());
      for (std::size_t i = 0; i < (std::size_t)cout * 9 * cin; ++i) {
        pw[i] = (_Float16)(((float)(i % 37) - 18.0f) * 0.002f);
      }
      auto* pb = static_cast<_Float16*>(c.b.contents());
      for (int i = 0; i < cout; ++i) { pb[i] = (_Float16)0.01f; }
      auto* pi = static_cast<_Float16*>(in.contents());
      for (std::size_t i = 0; i < (std::size_t)H * W * cin; ++i) {
        pi[i] = (_Float16)(((float)(i % 53) - 26.0f) * 0.01f);
      }
    }
    std::string one;
    _conv_pick[sh] = vae_conv3x3::autotune(
        "krea2", cin, cout, stride, cands, vae_conv3x3::Kind::kIm2col,
        [&](int i) -> double {
          _conv_pick[sh] = cands[(std::size_t)i];
          return autotune_time(mc, 1, [&](ComputeEncoder& enc) {
            conv3x3_fallback_(enc, in, out, H, W, c, stride, col, cap);
          });
        },
        &one);
    if (!detail.empty()) { detail += ", "; }
    detail += std::to_string(cin) + "->" + std::to_string(cout) +
              (stride == 2 ? "/s2 " : " ") + vae_conv3x3::name(_conv_pick[sh]);
    if (!one.empty()) { detail += " (" + one + ")"; }
  }
  const double tune_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
  if (mc->session() != nullptr && !detail.empty()) {
    const auto line = fmt("MetalKrea2Vae: conv3x3 autotune -- {} in {} ms",
                          detail, (long long)tune_ms);
    if (std::getenv("VPIPE_VAE_ATTN_TUNE_LOG") != nullptr) {
      mc->session()->log_normal(line);
    } else {
      mc->session()->log_debug(line);
    }
  }
}

std::size_t
MetalKrea2Vae::decode_peak_bytes(int h8, int w8) const noexcept
{
  // Preflight estimate (f16 = 2 bytes/elt). The per-up-block command-buffer
  // split (default on; VPIPE_KREA2_NO_VAE_SPLIT opts out) commits + frees each
  // up-level before the next, so the resident peak is ONE level's working set,
  // not the summed up-path. That peak is the top-res im2col scratch
  // [Hout*Wout, 9*base]: conv_out (3 ch) and the top-level resblock convs
  // (base ch) can't use the hw conv (cout % 64 != 0), so they fall back to
  // im2col even in hwconv mode -- it is the single largest buffer. Budget it +
  // ~50% for the level's input/output/carry activations. A miss here is caught
  // cleanly by the per-level wait_ok() backstop (never a corrupt image), so
  // this need not be wildly conservative -- an over-estimate just rejects
  // feasible decodes on a memory-bounded box, which is what regressed 1024px.
  if (h8 <= 0 || w8 <= 0) { return 0; }
  const std::size_t Hout = (std::size_t)h8 * 8;
  const std::size_t Wout = (std::size_t)w8 * 8;
  const std::size_t base = (std::size_t)_cfg.base_dim;
  const std::size_t im2col = Hout * Wout * 9 * base * 2;
  if (std::getenv("VPIPE_KREA2_NO_VAE_SPLIT") == nullptr) {
    return im2col + im2col / 2;                        // split on: one level
  }
  // Split off: the whole up-path is one command buffer -- keep the summed,
  // conservative figure (~16 GB at 1024px; the split is why it fits at all).
  const std::size_t top =
      Hout * Wout * (base * (std::size_t)_cfg.dim_mult[1]) * 2;
  return _use_hwconv ? top * 10 : top * 9 * 2;
}

int
MetalKrea2Vae::decode_tile_side_(std::size_t budget) const noexcept
{
  if (budget == 0) { return 0; }
  // Walk down in 8-cell steps; decode_peak_bytes is monotone in the side,
  // so the first fit is the largest. A side is latent cells (8 px each),
  // so 256 is a 2048^2 window -- the whole image at the size that made
  // tiling necessary, and the right place to start looking down from.
  for (int s = 256; s >= kTileMin8; s -= 8) {
    if (decode_peak_bytes(s, s) <= budget) { return s; }
  }
  return 0;
}

SharedBuffer
MetalKrea2Vae::decode_tiled_(const SharedBuffer& z, int h8, int w8,
                             int tile8, std::string* err)
{
  auto fail = [&](std::string m) -> SharedBuffer {
    if (err != nullptr) { *err = std::move(m); }
    return {};
  };
  const int Cz = _cfg.z_dim;
  const int px = 8;                              // output pixels per cell
  const int H = h8 * px, W = w8 * px;
  if (tile8 < kTileMin8) { return fail("tiled decode: window too small"); }
  const int ov = std::max(2, tile8 * kTileOvNum / kTileOvDen);
  const int step = tile8 - ov;
  if (step < 1) { return fail("tiled decode: overlap exceeds the window"); }

  const std::size_t hw = (std::size_t)H * W;
  std::vector<float> acc((std::size_t)3 * hw, 0.0f);   // weighted RGB sum
  std::vector<float> wsum(hw, 0.0f);                   // weight sum
  const auto* zsrc = static_cast<const _Float16*>(z.contents());
  if (zsrc == nullptr) { return fail("tiled decode: latent not host-visible"); }

  // Cross-fade weight for an OUTPUT pixel: ramps up over the overlap unless
  // the window starts at the image edge, down over it unless it ends there,
  // so interior windows sum to 1 across a seam. The ramp width is the
  // overlap in PIXELS -- ramping over `ov` cells instead would fade across
  // an eighth of the overlap and leave a visible seam.
  const int ovp = ov * px;
  auto ramp = [&](int i, int n, bool at_lo, bool at_hi) {
    float w = 1.0f;
    if (!at_lo && i < ovp)      { w = (float)(i + 1) / (float)(ovp + 1); }
    if (!at_hi && i >= n - ovp) {
      const float t = (float)(n - i) / (float)(ovp + 1);
      w = std::min(w, t);
    }
    return std::max(w, 1e-3f);
  };

  int ntiles = 0;
  for (int y0 = 0; y0 < h8; y0 += step) {
    const int th = std::min(tile8, h8 - y0);
    if (th <= 0) { break; }
    const bool y_lo = (y0 == 0), y_hi = (y0 + th >= h8);
    for (int x0 = 0; x0 < w8; x0 += step) {
      const int tw = std::min(tile8, w8 - x0);
      if (tw <= 0) { break; }
      const bool x_lo = (x0 == 0), x_hi = (x0 + tw >= w8);
      // Slice the latent window out of z[Cz, h8, w8] (channel-first).
      SharedBuffer zt = _mc->make_shared_buffer((std::size_t)Cz * th * tw * 2);
      if (zt.empty()) {
        return fail("tiled decode: window latent alloc failed");
      }
      auto* zd = static_cast<_Float16*>(zt.contents());
      for (int c = 0; c < Cz; ++c) {
        for (int y = 0; y < th; ++y) {
          const _Float16* srow =
              zsrc + ((std::size_t)c * h8 + (y0 + y)) * w8 + x0;
          _Float16* drow = zd + ((std::size_t)c * th + y) * tw;
          for (int x = 0; x < tw; ++x) { drow[x] = srow[x]; }
        }
      }
      std::string terr;
      SharedBuffer rgb = decode(zt, th, tw, &terr);
      if (rgb.empty()) {
        return fail(terr.empty() ? std::string("tiled decode: window failed")
                                 : terr);
      }
      // Cross-fade the window into the accumulator (planar RGB, f16).
      const int tH = th * px, tW = tw * px;
      const std::size_t thw = (std::size_t)tH * tW;
      const auto* rs = static_cast<const _Float16*>(rgb.contents());
      for (int y = 0; y < tH; ++y) {
        const float wy = ramp(y, tH, y_lo, y_hi);
        const std::size_t orow = (std::size_t)(y0 * px + y) * W + x0 * px;
        for (int x = 0; x < tW; ++x) {
          const float wgt = wy * ramp(x, tW, x_lo, x_hi);
          const std::size_t op = orow + x;
          const std::size_t tp = (std::size_t)y * tW + x;
          for (int c = 0; c < 3; ++c) {
            acc[(std::size_t)c * hw + op] +=
                wgt * (float)rs[(std::size_t)c * thw + tp];
          }
          wsum[op] += wgt;
        }
      }
      ++ntiles;
      if (x_hi) { break; }
    }
    if (y_hi) { break; }
  }

  SharedBuffer out = _mc->make_shared_buffer((std::size_t)3 * hw * 2);
  if (out.empty()) { return fail("tiled decode: output alloc failed"); }
  auto* od = static_cast<_Float16*>(out.contents());
  for (std::size_t p = 0; p < hw; ++p) {
    const float inv = (wsum[p] > 0.0f) ? 1.0f / wsum[p] : 0.0f;
    for (int c = 0; c < 3; ++c) {
      od[(std::size_t)c * hw + p] =
          (_Float16)(acc[(std::size_t)c * hw + p] * inv);
    }
  }
  if (_mc->session() != nullptr) {
    _mc->session()->log_normal(fmt(
        "Krea-2 VAE decode: TILED {}x{} from {} windows of {} latent cells "
        "(overlap {}) -- per-window attention, cross-faded seams",
        W, H, ntiles, tile8, ov));
  }
  return out;
}

SharedBuffer
MetalKrea2Vae::decode(const SharedBuffer& z, int h8, int w8, std::string* err)
{
  auto fail = [&](std::string m) -> SharedBuffer {
    if (err != nullptr) { *err = std::move(m); }
    return {};
  };
  const int Cz = _cfg.z_dim;
  const std::size_t hw0 = (std::size_t)h8 * w8;
  if (z.byte_size() < (std::size_t)Cz * hw0 * 2) {
    return fail("input latent smaller than [z_dim, h8, w8]");
  }
  MetalCompute* mc = _mc;
  const int Hout = h8 * 8, Wout = w8 * 8;
  const int base = _cfg.base_dim;                    // 96

  // Preflight: refuse to start a decode that clearly won't fit in the GPU's
  // current working-set headroom, rather than allocating our way into an
  // out-of-memory / page-fault mid-decode (which corrupts the output). A
  // shortfall here is the "prevent" half of over-commit handling.
  std::size_t decode_headroom = 0;   // free working set, for the im2col band cap
  {
    const MetalCompute::MemoryBudget mb = mc->memory_budget();
    decode_headroom = (mb.recommended != 0) ? mb.headroom : 0;
    const std::size_t need = decode_peak_bytes(h8, w8);
    const bool gpu_short = mb.recommended != 0 && !mb.fits(need);
    const bool ram_short = !mb.fits_physical(need);
    // Before failing, try to make it fit. The window is sized against
    // whichever budget is tighter, so a shortfall on either becomes a
    // smaller window rather than a rejection after the denoise has
    // already been paid for.
    const char* forced_env = std::getenv("VPIPE_VAE_TILE");
    const int   forced     = (forced_env != nullptr) ? std::atoi(forced_env) : 0;
    if (gpu_short || ram_short || forced > 0) {
      std::size_t budget = (std::size_t)-1;
      if (mb.available_physical != 0) {
        budget = (std::size_t)((double)mb.available_physical * 0.90);
      }
      if (mb.recommended != 0) {
        const auto gpu = (std::size_t)((double)mb.headroom * 0.95);
        if (gpu < budget) { budget = gpu; }
      }
      const int tile8 = (forced > 0) ? forced : decode_tile_side_(budget);
      // Strictly smaller than the image in at least one axis, or the
      // window decode would recurse into this same branch forever.
      if (tile8 > 0 && (tile8 < h8 || tile8 < w8) &&
          std::getenv("VPIPE_VAE_NO_TILE") == nullptr) {
        return decode_tiled_(z, h8, w8, tile8, err);
      }
    }
    if (gpu_short) {
      return fail(fmt(
          "insufficient GPU memory for a {}x{} decode: need ~{} MB, {} MB "
          "free of {} MB working set (lower the resolution, tile, or free "
          "other resident models)", Wout, Hout, need >> 20,
          mb.headroom >> 20, mb.recommended >> 20)());
    }
    // True-physical-pressure backstop (reclaimable RAM incl. evictable mmap'd
    // weights) so a shortfall rejects cleanly instead of a mid-decode GPU OOM.
    if (ram_short) {
      return fail(fmt(
          "insufficient free RAM for a {}x{} decode: need ~{} MB, ~{} MB "
          "reclaimable (close other apps, lower the resolution, or free "
          "resident models)", Wout, Hout, need >> 20,
          mb.available_physical >> 20)());
    }
  }

  // The WHOLE decode runs in ONE command stream: dispatches are enqueued on a
  // single serial ComputeEncoder (so each op sees the previous op's writes)
  // and committed once. Every intermediate buffer must outlive commit().wait()
  // -- so they are held in `keep` (a deque: element references stay valid as
  // more are pushed) and the primitives return references into it. The im2col
  // scratch is a single max-sized buffer reused across all 3x3 convs (safe:
  // serial ordering guarantees a conv's gemm reads it before the next conv's
  // im2col overwrites it).
  // Buffer pool with reuse: the VAE is a serial feed-forward chain, so a
  // released buffer is safe to reuse for a later op (serial dispatch orders the
  // reuse after the last read; an in-use buffer is never handed out). Bounds the
  // live set to the concurrent working set instead of the whole decode, so a
  // high-res decode fits one command buffer. VPIPE_KREA2_NO_VAE_POOL disables it
  // (then the per-level command-buffer split below bounds the peak instead).
  struct Slot { SharedBuffer buf; std::size_t cap; bool used; };
  std::deque<Slot> pool;
  bool alloc_ok = true;   // set false if any held allocation comes back empty
  const bool use_pool = std::getenv("VPIPE_KREA2_NO_VAE_POOL") == nullptr;
  auto alloc = [&](std::size_t e) -> SharedBuffer& {
    const std::size_t bytes = e * 2;
    if (use_pool) {
      for (auto& s : pool) {
        if (!s.used && !s.buf.empty() && s.cap >= bytes) {
          s.used = true;
          return s.buf;
        }
      }
    }
    pool.push_back(Slot{mc->make_shared_buffer(bytes), bytes, true});
    if (pool.back().buf.empty()) { alloc_ok = false; }
    return pool.back().buf;
  };
  auto release = [&](const SharedBuffer& b) {
    if (!use_pool) { return; }
    for (auto& s : pool) {
      if (&s.buf == &b) { s.used = false; return; }
    }
  };
  // Max im2col = the highest-resolution 3x3 conv (the up-block upsample conv at
  // full res, cin = base*dim_mult[1]). Row-tiled im2col (see the flux2 VAE):
  // stream each conv's [H*W, 9*cin] in output-row bands so the shared col
  // scratch is bounded to ONE band (a full-res conv is otherwise multi-GB), on
  // BOTH the hw-conv path (conv_out + top resblocks fall back to im2col) and the
  // non-matrix-core M4 path (all convs). `im2col_cap` (ELEMS) caps a band by
  // memory (reserve the level activation pool = decode_peak_bytes, band gets the
  // rest of the headroom) AND correctness (k_safe_band rows, matmul2d
  // M-corruption). Each conv derives band rows = im2col_cap / (9*cin), re-capped
  // at k_safe_band. VPIPE_KREA2_VAE_BAND_ROWS overrides the band.
  const std::size_t wide = (std::size_t)base * _cfg.dim_mult[1];   // widest cin
  const std::size_t k_safe_band =
      _mma_max_m > 0 ? (std::size_t)_mma_max_m / 2 : (std::size_t)Hout * Wout;
  const std::size_t full_band = (std::size_t)Hout * Wout * 9 * wide;
  const std::size_t floor_band = (std::size_t)Wout * 9 * wide * 8;   // 8 rows
  std::size_t im2col_cap = full_band;
  if (decode_headroom > 0) {
    const std::size_t reserve = decode_peak_bytes(h8, w8);   // pool reserve bytes
    const std::size_t avail_el = decode_headroom > reserve
        ? (decode_headroom - reserve) / 2 : 0;
    im2col_cap = std::min(full_band, std::max(floor_band, avail_el));
  }
  im2col_cap = std::min(im2col_cap, k_safe_band * 9 * wide);
  if (const char* e = std::getenv("VPIPE_KREA2_VAE_BAND_ROWS")) {
    const long r = std::atol(e);
    if (r > 0) { im2col_cap = (std::size_t)r * 9 * wide; }
  }
  SharedBuffer im2col_scratch = mc->make_shared_buffer(im2col_cap * 2);
  if (im2col_scratch.empty()) {
    return fail("im2col band scratch allocation failed (out of GPU memory)");
  }
  // First decode: measure the 3x3 fallback. Over THIS scratch and cap, so the
  // probe bands exactly as the convs below it will.
  maybe_tune_conv_(Hout, Wout, im2col_scratch, im2col_cap);

  CommandStream stream = mc->make_command_stream();
  int H = h8, W = w8;
  const SharedBuffer* rgb_ptr = nullptr;
  // Per-up-block command-buffer split: bound each command buffer to ~one
  // resolution level so the whole decode's running set (which grows to > the
  // GPU wired limit at high res -- channels grow as spatial shrinks) never
  // shares one command buffer. `carry` crosses each boundary. Opt out with
  // VPIPE_KREA2_NO_VAE_SPLIT (fine at small res).
  SharedBuffer carry;
  const bool vae_split = std::getenv("VPIPE_KREA2_NO_VAE_SPLIT") == nullptr;
  bool split_ok = true;
  {
    ComputeEncoder enc = stream.begin_compute();

    auto gemm_bias = [&](const SharedBuffer& x, const SharedBuffer& w,
                         const SharedBuffer& b, const SharedBuffer& y, int M,
                         int N, int K) {
      conv_gemm_bias_(enc, x, w, b, y, M, N, K);
    };
    // 3x3 conv2d: im2col (into the shared scratch, or a held buffer if it
    // overflows) then gemm_bias. Spatial size preserved.
    auto conv3x3 = [&](const SharedBuffer& in, int H, int W,
                       const Conv& c) -> SharedBuffer& {
      SharedBuffer& out = alloc((std::size_t)H * W * c.cout);
      // NAX hardware conv first; then the (opt-in) fused tgmem conv; else the
      // row-tiled im2col + gemm_bias (bands through im2col_scratch).
      if (conv3x3_hw_(enc, in, c, out, H, W, /*stride=*/1)) { return out; }
      // Before the fused/im2col paths: conv_out (-> 3) can use neither the
      // hardware conv (cout % 64) nor an im2col that is worth its scratch.
      if (conv3x3_small_cout_(enc, in, c, out, H, W, /*stride=*/1)) {
        return out;
      }
      conv3x3_fallback_(enc, in, out, H, W, c, /*stride=*/1, im2col_scratch,
                        im2col_cap);
      return out;
    };
    auto conv1x1 = [&](const SharedBuffer& in, std::size_t hw,
                       const Conv& c) -> SharedBuffer& {
      SharedBuffer& out = alloc(hw * c.cout);
      gemm_bias(in, c.w, c.b, out, (int)hw, c.cout, c.cin);
      return out;
    };
    // QwenImageRMS_norm over the channel axis: x/||x|| * sqrt(C) * gamma. The
    // rms kernel's x*rsqrt(mean+eps)*w matches with w=gamma and eps ~ 0.
    auto normc = [&](const SharedBuffer& in, std::size_t hw, int C,
                     const SharedBuffer& g) -> SharedBuffer& {
      SharedBuffer& out = alloc(hw * C);
      const float eps = 1e-12f / (float)C;
      enc.set_function(_fn_rms);
      enc.set_buffer(0, in); enc.set_buffer(1, g); enc.set_buffer(2, out);
      enc.set_constant(3, C); enc.set_constant(4, eps);
      enc.dispatch({256, (unsigned)hw, 1}, {256, 1, 1});
      return out;
    };
    auto silu = [&](const SharedBuffer& x, std::size_t n) {   // in place
      enc.set_function(_fn_mul_sigmoid);
      enc.set_buffer(0, x); enc.set_buffer(1, x); enc.set_buffer(2, x);
      enc.set_constant(3, (int)n);
      enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
    };
    auto resadd = [&](const SharedBuffer& a, const SharedBuffer& b,
                      std::size_t n) -> SharedBuffer& {
      SharedBuffer& out = alloc(n);
      enc.set_function(_fn_residual);
      enc.set_buffer(0, a); enc.set_buffer(1, b); enc.set_buffer(2, out);
      enc.set_constant(3, (int)n);
      enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
      return out;
    };
    auto upsample = [&](const SharedBuffer& in, int H, int W,
                        int C) -> SharedBuffer& {
      SharedBuffer& out = alloc((std::size_t)4 * H * W * C);
      const std::size_t rows = (std::size_t)4 * H * W;     // 2H * 2W pixels
      enc.set_function(_fn_upsample);
      enc.set_buffer(0, in); enc.set_buffer(1, out);
      enc.set_constant(2, H); enc.set_constant(3, W); enc.set_constant(4, C);
      // 2D grid {C, 4*H*W}: gid = pixel*C + c (a 1D {4*H*W*C} grid overflows
      // the int index and the grid dimension past ~3K px).
      enc.dispatch({(unsigned)C, (unsigned)rows, 1}, {256, 1, 1});
      return out;
    };
    auto resblock = [&](const ResBlock& rb, const SharedBuffer& x, int H,
                        int W) -> SharedBuffer& {
      const std::size_t hw = (std::size_t)H * W;
      SharedBuffer& t = normc(x, hw, rb.cin, rb.n1g);
      silu(t, hw * rb.cin);
      SharedBuffer& t1 = conv3x3(t, H, W, rb.c1);
      release(t);
      SharedBuffer& t2 = normc(t1, hw, rb.cout, rb.n2g);
      release(t1);
      silu(t2, hw * rb.cout);
      SharedBuffer& t3 = conv3x3(t2, H, W, rb.c2);
      release(t2);
      if (rb.has_short) {                          // in != out (up1.resnets.0)
        SharedBuffer& h = conv1x1(x, hw, rb.shortcut);
        SharedBuffer& out = resadd(t3, h, hw * rb.cout);
        release(t3); release(h);
        return out;
      }
      SharedBuffer& out = resadd(t3, x, hw * rb.cout);   // identity shortcut
      release(t3);
      return out;
    };
    // Single-head spatial self-attention (mid block).
    auto attention = [&](const Attn& a, const SharedBuffer& x, int H,
                         int W) -> SharedBuffer& {
      const std::size_t hw = (std::size_t)H * W;
      const int C = a.dim;
      SharedBuffer& n = normc(x, hw, C, a.ng);
      SharedBuffer& q = conv1x1(n, hw, a.q);
      SharedBuffer& k = conv1x1(n, hw, a.k);
      SharedBuffer& v = conv1x1(n, hw, a.v);
      release(n);                          // consumed by q, k, v
      SharedBuffer& att = alloc(hw * C);
      const float scale = 1.0f / std::sqrt((float)C);
      // The member is chosen ONCE, by measurement, in autotune_mid_attn_.
      encode_mid_attn_(enc, _attn_pick, q, k, v, att, hw, C, scale);
      release(q); release(k); release(v);  // consumed by the sdpa
      SharedBuffer& p = conv1x1(att, hw, a.proj);
      release(att);
      SharedBuffer& out = resadd(p, x, hw * C);
      release(p);
      return out;
    };

    // Input z is channel-first [Cz, h8, w8] -> channel-last [hw, Cz] (host).
    SharedBuffer& x0 = alloc(hw0 * Cz);
    {
      const auto* s = static_cast<const _Float16*>(z.contents());
      auto* d = static_cast<_Float16*>(x0.contents());
      for (int c = 0; c < Cz; ++c) {
        for (std::size_t p = 0; p < hw0; ++p) {
          d[p * Cz + c] = s[(std::size_t)c * hw0 + p];
        }
      }
    }
    auto flush = [&](const SharedBuffer*& xp) {
      if (!vae_split) { return; }
      enc.end();
      std::string ge;
      if (!stream.commit().wait_ok(&ge)) { split_ok = false; }
      for (auto& s : pool) {
        if (&s.buf == xp) { carry = std::move(s.buf); break; }
      }
      pool.clear();
      xp = &carry;
      stream = mc->make_command_stream();
      enc = stream.begin_compute();
    };

    const SharedBuffer* x = &conv1x1(x0, hw0, _post_quant);   // post_quant (1x1)
    release(x0);
    auto step = [&](const SharedBuffer& nx) { release(*x); x = &nx; };
    step(conv3x3(*x, H, W, _conv_in));           // conv_in (3x3, Cz -> dims0)
    step(resblock(_mid_res0, *x, H, W));
    step(attention(_mid_attn, *x, H, W));
    step(resblock(_mid_res1, *x, H, W));
    for (const UpBlock& ub : _up_blocks) {
      for (const ResBlock& rb : ub.resnets) { step(resblock(rb, *x, H, W)); }
      if (ub.has_up) {
        step(upsample(*x, H, W, ub.up_dim));     // nearest 2x
        H *= 2; W *= 2;
        step(conv3x3(*x, H, W, ub.up));          // resample.1 (out -> out/2)
      }
      flush(x);            // pool off: bound the working set to ~one up-block
    }
    SharedBuffer& xn = normc(*x, (std::size_t)H * W, base, _norm_out_g);
    release(*x);
    silu(xn, (std::size_t)H * W * base);
    SharedBuffer& rgb = conv3x3(xn, H, W, _conv_out);        // -> [hw, 3]
    release(xn);
    const int n = H * W * 3;                                  // clamp to [-1,1]
    enc.set_function(_fn_clamp);
    enc.set_buffer(0, rgb); enc.set_buffer(1, rgb);
    enc.set_constant(2, n); enc.set_constant(3, -1.0f); enc.set_constant(4, 1.0f);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
    rgb_ptr = &rgb;
  }
  // Detect an over-commit that slipped past the preflight: a held buffer that
  // came back empty, or a GPU out-of-memory / page-fault at execution (the
  // usual UMA case -- Shared allocations succeed virtually, then the GPU
  // faults on non-resident pages). Either way, fail loudly instead of
  // returning the half-written / uninitialized output.
  if (!alloc_ok) {
    return fail("a decode intermediate allocation failed (out of GPU memory)");
  }
  if (!split_ok) {
    return fail("GPU decode ran out of memory at a level boundary "
                "(lower the resolution)");
  }
  std::string gpu_err;
  if (!stream.commit().wait_ok(&gpu_err)) {
    return fail(gpu_err.empty() ? std::string("GPU decode failed") : gpu_err);
  }

  // Transpose channel-last [hw, 3] -> channel-first [3, H, W].
  SharedBuffer out = mc->make_shared_buffer((std::size_t)3 * H * W * 2);
  {
    const auto* s = static_cast<const _Float16*>(rgb_ptr->contents());
    auto* d = static_cast<_Float16*>(out.contents());
    const std::size_t hw = (std::size_t)H * W;
    for (std::size_t p = 0; p < hw; ++p) {
      for (int c = 0; c < 3; ++c) { d[(std::size_t)c * hw + p] = s[p * 3 + c]; }
    }
  }
  return out;
}

bool
MetalKrea2Vae::load_encoder_(WeightSet& ws)
{
  WeightSet& wts = ws;
  MetalCompute* mc = _mc;
  const int base = _cfg.base_dim;                    // 96
  // Encoder dims = [base*u for u in [1] + dim_mult] = [96, 96, 192, 384, 384].
  const int dims[5] = {base, base * _cfg.dim_mult[0], base * _cfg.dim_mult[1],
                       base * _cfg.dim_mult[2], base * _cfg.dim_mult[3]};
  const int dtop = base * _cfg.dim_mult[3];          // 384

  _enc_conv_in = load_conv3x3_(wts, "encoder.conv_in", true);   // 3 -> base
  bool ok = !_enc_conv_in.w.empty();

  // 4 down stages; the flat down_blocks index runs res,res,[resample],...
  _enc_down.resize(4);
  int idx = 0;
  for (int i = 0; i < 4; ++i) {
    DownStage& st = _enc_down[(std::size_t)i];
    const int out_dim = dims[i + 1];
    st.resnets.resize((std::size_t)_cfg.num_res_blocks);
    int cin = dims[i];
    for (int r = 0; r < _cfg.num_res_blocks; ++r) {
      ok = ok && load_resblock_(
          wts, "encoder.down_blocks." + std::to_string(idx) + ".",
          st.resnets[(std::size_t)r], cin, out_dim);
      cin = out_dim; ++idx;
    }
    st.has_down = (i != 3);
    if (st.has_down) {
      st.down = load_conv3x3_(
          wts, "encoder.down_blocks." + std::to_string(idx) + ".resample.1",
          false);                                    // 2D stride-2 conv
      ok = ok && !st.down.w.empty();
      ++idx;
    }
  }

  ok = ok && load_resblock_(wts, "encoder.mid_block.resnets.0.",
                            _enc_mid_res0, dtop, dtop);
  ok = ok && load_resblock_(wts, "encoder.mid_block.resnets.1.",
                            _enc_mid_res1, dtop, dtop);
  _enc_mid_attn.dim = dtop;
  _enc_mid_attn.ng = load_vec_(wts, "encoder.mid_block.attentions.0.norm.gamma");
  {
    std::size_t n = 0, nb = 0;
    std::vector<float> qkv, qkvb;
    const std::string base = "encoder.mid_block.attentions.0.to_qkv";
    auto read_qkv = [&]() {
      if (!qkv.empty()) { return; }
      qkv  = read_f32_(wts.src(), mc, base + ".weight", n);
      qkvb = read_f32_(wts.src(), mc, base + ".bias", nb);
    };
    const int C = dtop;
    {
      // The fused to_qkv is SPLIT into three convs, so each slice is its
      // own derived tensor keyed by which third it is.
      auto slice = [&](int off) {
        Conv c; c.cin = C; c.cout = C; c.k = C;
        const std::string k = std::string(kKey) + "qkv|" + base + "|" +
                              std::to_string(off);
        c.w = wts.derived(k + "|w", [&]() -> SharedBuffer {
          read_qkv();
          if (qkv.size() != (std::size_t)3 * C * C) { return {}; }
          return f16_buf_(mc, qkv.data() + (std::size_t)off * C * C,
                          (std::size_t)C * C);
        }, _part);
        c.b = wts.derived(k + "|b", [&]() -> SharedBuffer {
          read_qkv();
          if (qkvb.size() != (std::size_t)3 * C) { return {}; }
          return f16_buf_(mc, qkvb.data() + (std::size_t)off * C,
                          (std::size_t)C);
        }, _part);
        return c;
      };
      _enc_mid_attn.q = slice(0);
      _enc_mid_attn.k = slice(1);
      _enc_mid_attn.v = slice(2);
    }
  }
  _enc_mid_attn.proj = load_conv1x1_(wts, "encoder.mid_block.attentions.0.proj");
  ok = ok && !_enc_mid_attn.ng.empty() && !_enc_mid_attn.q.w.empty() &&
       !_enc_mid_attn.proj.w.empty();

  _enc_norm_out_g = load_vec_(wts, "encoder.norm_out.gamma");
  _enc_conv_out = load_conv3x3_(wts, "encoder.conv_out", true);   // dtop -> z*2
  _quant_conv = load_conv1x1_(wts, "quant_conv");                 // z*2 -> z*2
  ok = ok && !_enc_norm_out_g.empty() && !_enc_conv_out.w.empty() &&
       !_quant_conv.w.empty();
  return ok;
}

SharedBuffer
MetalKrea2Vae::encode(const SharedBuffer& img, int H, int W)
{
  if (!_has_encoder) { return {}; }
  const std::size_t hw = (std::size_t)H * W;
  if (img.byte_size() < (std::size_t)3 * hw * 2) { return {}; }
  const int Cz = _cfg.z_dim;
  if ((int)_cfg.latents_mean.size() != Cz ||
      (int)_cfg.latents_std.size() != Cz) {
    return {};
  }
  MetalCompute* mc = _mc;
  const int base = _cfg.base_dim;

  // Buffer pool with reuse (see decode()). VPIPE_KREA2_NO_VAE_POOL disables it.
  struct Slot { SharedBuffer buf; std::size_t cap; bool used; };
  std::deque<Slot> pool;
  bool alloc_ok = true;
  const bool use_pool = std::getenv("VPIPE_KREA2_NO_VAE_POOL") == nullptr;
  auto alloc = [&](std::size_t e) -> SharedBuffer& {
    const std::size_t bytes = e * 2;
    if (use_pool) {
      for (auto& s : pool) {
        if (!s.used && !s.buf.empty() && s.cap >= bytes) {
          s.used = true;
          return s.buf;
        }
      }
    }
    pool.push_back(Slot{mc->make_shared_buffer(bytes), bytes, true});
    if (pool.back().buf.empty()) { alloc_ok = false; }
    return pool.back().buf;
  };
  auto release = [&](const SharedBuffer& b) {
    if (!use_pool) { return; }
    for (auto& s : pool) {
      if (&s.buf == &b) { s.used = false; return; }
    }
  };
  // Row-tiled im2col band scratch (see decode() / the flux2 VAE): stream each
  // conv's [OH*OW, 9*cin] in output-row bands so the shared col scratch is one
  // band, not the full-res base-channel [hw, 9*base] (multi-GB at high res).
  // Peak full-res im2col is the base-ch s1 convs; deeper levels shrink. Cap the
  // band by memory (headroom -- no encode preflight, query it here) AND
  // correctness (k_safe_band rows, matmul2d M-corruption).
  const std::size_t k_safe_band =
      _mma_max_m > 0 ? (std::size_t)_mma_max_m / 2 : hw;
  const std::size_t full_band = hw * 9 * (std::size_t)base;
  const std::size_t floor_band = (std::size_t)W * 9 * base * 8;
  const std::size_t act_reserve = hw * (std::size_t)base * 7;
  std::size_t im2col_cap = full_band;
  {
    const MetalCompute::MemoryBudget mb = mc->memory_budget();
    if (mb.recommended != 0) {
      const std::size_t avail_el = mb.headroom > act_reserve * 2
          ? (mb.headroom - act_reserve * 2) / 2 : 0;
      im2col_cap = std::min(full_band, std::max(floor_band, avail_el));
    }
  }
  im2col_cap = std::min(im2col_cap, k_safe_band * 9 * (std::size_t)base);
  if (const char* e = std::getenv("VPIPE_KREA2_VAE_BAND_ROWS")) {
    const long r = std::atol(e);
    if (r > 0) { im2col_cap = (std::size_t)r * 9 * base; }
  }
  SharedBuffer im2col_scratch = mc->make_shared_buffer(im2col_cap * 2);
  if (im2col_scratch.empty()) { return {}; }
  maybe_tune_conv_(H, W, im2col_scratch, im2col_cap);

  CommandStream stream = mc->make_command_stream();
  int Hc = H, Wc = W;
  const SharedBuffer* last = nullptr;
  int last_ch = 0;
  // Per-down-stage command-buffer split (see decode()). `carry` crosses each
  // boundary; VPIPE_KREA2_NO_VAE_SPLIT opts out.
  SharedBuffer carry;
  const bool vae_split = std::getenv("VPIPE_KREA2_NO_VAE_SPLIT") == nullptr;
  bool split_ok = true;
  {
    ComputeEncoder enc = stream.begin_compute();
    auto gemm_bias = [&](const SharedBuffer& x, const SharedBuffer& w,
                         const SharedBuffer& b, const SharedBuffer& y, int M,
                         int N, int K) {
      conv_gemm_bias_(enc, x, w, b, y, M, N, K);
    };
    auto conv3x3g = [&](const SharedBuffer& in, int H, int W, const Conv& c,
                        bool stride2) -> SharedBuffer& {
      const int OH = stride2 ? H / 2 : H, OW = stride2 ? W / 2 : W;
      SharedBuffer& out = alloc((std::size_t)OH * OW * c.cout);
      if (conv3x3_hw_(enc, in, c, out, H, W, stride2 ? 2 : 1)) {
        return out;
      }
      if (!stride2 &&
          conv3x3_small_cout_(enc, in, c, out, H, W, /*stride=*/1)) {
        return out;
      }
      // Fused conv2d where the tune picked it, else row-tiled im2col (which
      // streams bands through im2col_scratch; the s1 and s2 tiled kernels
      // share the tpig.y*(9*cin)+tpig.x layout).
      conv3x3_fallback_(enc, in, out, H, W, c, stride2 ? 2 : 1, im2col_scratch,
                        im2col_cap);
      return out;
    };
    auto conv1x1 = [&](const SharedBuffer& in, std::size_t hwv,
                       const Conv& c) -> SharedBuffer& {
      SharedBuffer& out = alloc(hwv * c.cout);
      gemm_bias(in, c.w, c.b, out, (int)hwv, c.cout, c.cin);
      return out;
    };
    auto normc = [&](const SharedBuffer& in, std::size_t hwv, int C,
                     const SharedBuffer& g) -> SharedBuffer& {
      SharedBuffer& out = alloc(hwv * C);
      const float eps = 1e-12f / (float)C;
      enc.set_function(_fn_rms);
      enc.set_buffer(0, in); enc.set_buffer(1, g); enc.set_buffer(2, out);
      enc.set_constant(3, C); enc.set_constant(4, eps);
      enc.dispatch({256, (unsigned)hwv, 1}, {256, 1, 1});
      return out;
    };
    auto silu = [&](const SharedBuffer& x, std::size_t n) {
      enc.set_function(_fn_mul_sigmoid);
      enc.set_buffer(0, x); enc.set_buffer(1, x); enc.set_buffer(2, x);
      enc.set_constant(3, (int)n);
      enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
    };
    auto resadd = [&](const SharedBuffer& a, const SharedBuffer& b,
                      std::size_t n) -> SharedBuffer& {
      SharedBuffer& out = alloc(n);
      enc.set_function(_fn_residual);
      enc.set_buffer(0, a); enc.set_buffer(1, b); enc.set_buffer(2, out);
      enc.set_constant(3, (int)n);
      enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
      return out;
    };
    auto resblock = [&](const ResBlock& rb, const SharedBuffer& x, int H,
                        int W) -> SharedBuffer& {
      const std::size_t hwv = (std::size_t)H * W;
      SharedBuffer& t = normc(x, hwv, rb.cin, rb.n1g);
      silu(t, hwv * rb.cin);
      SharedBuffer& t1 = conv3x3g(t, H, W, rb.c1, false);
      SharedBuffer& t2 = normc(t1, hwv, rb.cout, rb.n2g);
      silu(t2, hwv * rb.cout);
      SharedBuffer& t3 = conv3x3g(t2, H, W, rb.c2, false);
      if (rb.has_short) {
        SharedBuffer& h = conv1x1(x, hwv, rb.shortcut);
        return resadd(t3, h, hwv * rb.cout);
      }
      return resadd(t3, x, hwv * rb.cout);
    };
    auto attention = [&](const Attn& a, const SharedBuffer& x, int H,
                         int W) -> SharedBuffer& {
      const std::size_t hwv = (std::size_t)H * W;
      const int C = a.dim;
      SharedBuffer& n = normc(x, hwv, C, a.ng);
      SharedBuffer& q = conv1x1(n, hwv, a.q);
      SharedBuffer& k = conv1x1(n, hwv, a.k);
      SharedBuffer& v = conv1x1(n, hwv, a.v);
      release(n);                          // consumed by q, k, v
      SharedBuffer& att = alloc(hwv * C);
      const float scale = 1.0f / std::sqrt((float)C);
      // The member is chosen ONCE, by measurement, in autotune_mid_attn_.
      // hwv, NOT hw: the encoder's enclosing scope has its own `hw` (the
      // full-resolution pixel count) and passing that here compiles cleanly
      // while running the attention over a sequence ~64x too long.
      encode_mid_attn_(enc, _attn_pick, q, k, v, att, hwv, C, scale);
      release(q); release(k); release(v);  // consumed by the sdpa
      SharedBuffer& p = conv1x1(att, hwv, a.proj);
      release(att);
      SharedBuffer& out = resadd(p, x, hwv * C);
      release(p);
      return out;
    };

    // Input RGB [3, H, W] -> channel-last [hw, 3] (host).
    SharedBuffer& x0 = alloc(hw * 3);
    {
      const auto* s = static_cast<const _Float16*>(img.contents());
      auto* d = static_cast<_Float16*>(x0.contents());
      for (int c = 0; c < 3; ++c) {
        for (std::size_t p = 0; p < hw; ++p) {
          d[p * 3 + c] = s[(std::size_t)c * hw + p];
        }
      }
    }
    auto flush = [&](const SharedBuffer*& xp) {
      if (!vae_split) { return; }
      enc.end();
      std::string ge;
      if (!stream.commit().wait_ok(&ge)) { split_ok = false; }
      for (auto& s : pool) {
        if (&s.buf == xp) { carry = std::move(s.buf); break; }
      }
      pool.clear();
      xp = &carry;
      stream = mc->make_command_stream();
      enc = stream.begin_compute();
    };

    const SharedBuffer* x = &conv3x3g(x0, Hc, Wc, _enc_conv_in, false);
    release(x0);
    auto step = [&](const SharedBuffer& nx) { release(*x); x = &nx; };
    for (const DownStage& st : _enc_down) {
      for (const ResBlock& rb : st.resnets) { step(resblock(rb, *x, Hc, Wc)); }
      if (st.has_down) {
        step(conv3x3g(*x, Hc, Wc, st.down, true));  // stride-2 downsample
        Hc /= 2; Wc /= 2;
      }
      flush(x);            // pool off: bound the working set to ~one down-stage
    }
    const int dtop = base * _cfg.dim_mult[3];        // 384
    step(resblock(_enc_mid_res0, *x, Hc, Wc));
    step(attention(_enc_mid_attn, *x, Hc, Wc));
    step(resblock(_enc_mid_res1, *x, Hc, Wc));
    SharedBuffer& xn = normc(*x, (std::size_t)Hc * Wc, dtop, _enc_norm_out_g);
    release(*x);
    silu(xn, (std::size_t)Hc * Wc * dtop);
    SharedBuffer& conv = conv3x3g(xn, Hc, Wc, _enc_conv_out, false);  // ->z*2
    release(xn);
    SharedBuffer& q = conv1x1(conv, (std::size_t)Hc * Wc, _quant_conv);
    release(conv);
    last = &q;
    last_ch = _cfg.z_dim * 2;
  }
  if (!alloc_ok || !split_ok) { return {}; }
  stream.commit().wait();

  // Posterior MODE = first z_dim channels; whiten (x-mean)/std; transpose
  // channel-last [hw', z*2] -> channel-first [z_dim, H/8, W/8].
  const std::size_t hwp = (std::size_t)Hc * Wc;
  SharedBuffer out = mc->make_shared_buffer((std::size_t)Cz * hwp * 2);
  {
    const auto* s = static_cast<const _Float16*>(last->contents());
    auto* d = static_cast<_Float16*>(out.contents());
    for (int c = 0; c < Cz; ++c) {
      const float mu = _cfg.latents_mean[(std::size_t)c];
      const float sd = _cfg.latents_std[(std::size_t)c];
      for (std::size_t p = 0; p < hwp; ++p) {
        const float v = (float)s[p * (std::size_t)last_ch + c];
        d[(std::size_t)c * hwp + p] = (_Float16)((v - mu) / sd);
      }
    }
  }
  return out;
}

}  // namespace genai
}  // namespace vpipe

#include "generative-models/flux2/metal-flux2-vae.h"

#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/weight-set.h"
#include "generative-models/shared/kernel-autotune.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
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
constexpr const char* kKey = "flux2-vae/";

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

// 3x3 conv [Cout,Cin,3,3] -> dense-gemm weight [Cout, 9*Cin] flattened
// (ky,kx,cin) to pair with im2col_hwc_3x3.
MetalFlux2Vae::Conv
MetalFlux2Vae::load_conv3x3_(WeightSet& wts, const std::string& nm)
{
  Conv c;
  const auto* info = wts.src().info(nm + ".weight");
  if (info == nullptr || info->shape.size() < 2) { return c; }
  const auto& sh = info->shape;
  const int Cout = (int)sh[0], Cin = (int)sh[1];
  c.cin = Cin; c.cout = Cout; c.k = 9 * Cin;

  // The flattened [Cout, 9*Cin] weight and its HWIO twin are DERIVED --
  // built from the checkpoint's bytes rather than copied out of them --
  // so each is cached under a key naming the transform, not the tensor.
  std::vector<float> w;
  auto read_w = [&]() {
    if (!w.empty()) { return; }
    std::size_t n = 0;
    w = read_f32_(wts.src(), _mc, nm + ".weight", n);
  };
  const std::string k3 = std::string(kKey) + "c3x3|" + nm;
  c.w = wts.derived(k3, [&]() -> SharedBuffer {
    read_w();
    if (w.empty()) { return {}; }
    std::vector<float> flat((std::size_t)Cout * 9 * Cin);
    for (int o = 0; o < Cout; ++o) {
      for (int ky = 0; ky < 3; ++ky) {
        for (int kx = 0; kx < 3; ++kx) {
          for (int i = 0; i < Cin; ++i) {
            const std::size_t si =
                ((((std::size_t)o * Cin + i) * 3 + ky) * 3) + kx;
            const std::size_t di =
                ((std::size_t)o * 9 + (ky * 3 + kx)) * Cin + i;
            flat[di] = w[si];
          }
        }
      }
    }
    return f16_buf_(_mc, flat.data(), flat.size());
  }, _part);
  if (c.w.empty()) { return Conv{}; }
  // HWIO twin (out-channel fastest). Two consumers, and they do NOT have
  // the same availability:
  //
  //   * conv3x3_hw_, the NAX hardware convolution -- matrix cores (M5+).
  //   * conv3x3_small_cout_, which keeps the short output vector in
  //     registers -- plain arithmetic, no matrix cores needed.
  //
  // Building the twin only for the first left the second dead on M4: the
  // kernel loads, then declines every call because c.whwio is empty, and
  // conv_out (cout=3) falls back to im2col AT FULL RESOLUTION -- the
  // multi-GB materialization for a ~5 GFLOP convolution that the small-cout
  // path exists to avoid. So build it whenever EITHER consumer can use it.
  // Cheap: the small-cout twin is only ever built for cout <= 32, i.e. the
  // final conv, so it costs 9*Cin*32 halves at worst.
  const bool want_small_cout_twin =
      _fn_conv_small_cout.valid() && Cout > 0 && Cout <= kSmallCoutMax;
  if (_use_hwconv || want_small_cout_twin) {
    c.whwio = wts.derived(k3 + "|hwio", [&]() -> SharedBuffer {
      read_w();
      if (w.empty()) { return {}; }
      std::vector<float> hwio((std::size_t)9 * Cin * Cout);
      for (int o = 0; o < Cout; ++o) {
        for (int ky = 0; ky < 3; ++ky) {
          for (int kx = 0; kx < 3; ++kx) {
            for (int i = 0; i < Cin; ++i) {
              const std::size_t si =
                  ((((std::size_t)o * Cin + i) * 3 + ky) * 3) + kx;
              hwio[(((std::size_t)(ky * 3 + kx) * Cin) + i) * Cout + o] =
                  w[si];
            }
          }
        }
      }
      return f16_buf_(_mc, hwio.data(), hwio.size());
    }, _part);
  }
  c.b = load_vec_(wts, nm + ".bias");
  return c;
}

// 1x1 conv or Linear [Cout,Cin(,1,1)] -> dense-gemm weight [Cout, Cin].
MetalFlux2Vae::Conv
MetalFlux2Vae::load_conv1x1_(WeightSet& wts, const std::string& nm)
{
  Conv c;
  const auto* info = wts.src().info(nm + ".weight");
  if (info == nullptr || info->shape.size() < 2) { return c; }
  const auto& sh = info->shape;
  c.cout = (int)sh[0]; c.cin = (int)sh[1]; c.k = c.cin;
  // A 1x1 conv's [Cout,Cin(,1,1)] weight flattens to [Cout, Cin] with no
  // reordering, so it needs nothing beyond the f16 narrowing load_vec_
  // already does -- and shares its cache entry.
  c.w = load_vec_(wts, nm + ".weight");
  if (c.w.empty()) { return Conv{}; }
  c.b = load_vec_(wts, nm + ".bias");
  return c;
}

MetalFlux2Vae::GNorm
MetalFlux2Vae::load_gnorm_(WeightSet& wts, const std::string& nm)
{
  GNorm gn;
  gn.g = load_vec_(wts, nm + ".weight");
  gn.b = load_vec_(wts, nm + ".bias");
  const auto* info = wts.src().info(nm + ".weight");
  if (info != nullptr && !info->shape.empty()) { gn.c = (int)info->shape[0]; }
  return gn;
}

// Every tensor the VAE keeps is stored as f16 regardless of its on-disk
// dtype, so "read as f32 and narrow" IS the transform and the result is
// cached like any other derived tensor.
SharedBuffer
MetalFlux2Vae::load_vec_(WeightSet& wts, const std::string& nm)
{
  return wts.derived(std::string(kKey) + "f16|" + nm, [&]() -> SharedBuffer {
    std::size_t n = 0;
    std::vector<float> v = read_f32_(wts.src(), _mc, nm, n);
    if (v.empty()) { return {}; }
    return f16_buf_(_mc, v.data(), n);
  }, _part);
}

bool
MetalFlux2Vae::load_resblock_(WeightSet& wts,
                              const std::string& pre, ResBlock& rb, int cin,
                              int cout)
{
  rb.cin = cin; rb.cout = cout;
  rb.n1 = load_gnorm_(wts, pre + "norm1");
  rb.c1 = load_conv3x3_(wts, pre + "conv1");
  rb.n2 = load_gnorm_(wts, pre + "norm2");
  rb.c2 = load_conv3x3_(wts, pre + "conv2");
  rb.has_short = (cin != cout);
  if (rb.has_short) {
    rb.shortcut = load_conv1x1_(wts, pre + "conv_shortcut");
    if (rb.shortcut.w.empty()) { return false; }
  }
  return !rb.n1.g.empty() && !rb.n1.b.empty() && !rb.c1.w.empty() &&
         !rb.n2.g.empty() && !rb.n2.b.empty() && !rb.c2.w.empty();
}

MetalFlux2Vae::Attn
MetalFlux2Vae::load_attn_(WeightSet& wts, const std::string& pre,
                          int dim)
{
  Attn a;
  a.dim = dim;
  a.n = load_gnorm_(wts, pre + "group_norm");
  a.q = load_conv1x1_(wts, pre + "to_q");
  a.k = load_conv1x1_(wts, pre + "to_k");
  a.v = load_conv1x1_(wts, pre + "to_v");
  a.proj = load_conv1x1_(wts, pre + "to_out.0");
  return a;
}

std::unique_ptr<MetalFlux2Vae>
MetalFlux2Vae::load(const std::string& model_dir, MetalCompute* mc,
                    const Config& cfg_in, bool with_encoder)
{
  return load(WeightSet::open(model_dir, nullptr), mc, cfg_in, with_encoder);
}

std::unique_ptr<MetalFlux2Vae>
MetalFlux2Vae::load(std::shared_ptr<WeightSet> ws_in, MetalCompute* mc,
                    const Config& cfg_in, bool with_encoder)
{
  if (mc == nullptr || !ws_in) { return nullptr; }
  WeightSet& wts = *ws_in;
  const std::string model_dir = ws_in->dir();

  auto m = std::unique_ptr<MetalFlux2Vae>(new MetalFlux2Vae());
  m->_ws = std::move(ws_in);
  m->_mc = mc;

  // Size from vae/config.json so the same code serves every AutoencoderKLFlux2
  // (the FLUX.2 family shares one VAE, so klein-4B and klein-9B match, but read
  // the config anyway to stay robust). Absent keys keep the Config default.
  Config cfg = cfg_in;
  {
    namespace fs = std::filesystem;
    std::ifstream in(fs::path(model_dir) / "config.json");
    if (in) {
      FlexData fd = FlexData::from_json(in);
      if (fd.is_object()) {
        auto o = fd.as_object();
        auto geti = [&](const char* k, int cur) -> int {
          return o.contains(k) ? (int)o.at(k).as_int(cur) : cur;
        };
        cfg.in_channels     = geti("in_channels", cfg.in_channels);
        cfg.latent_channels = geti("latent_channels", cfg.latent_channels);
        cfg.norm_groups     = geti("norm_num_groups", cfg.norm_groups);
        cfg.layers_per_block = geti("layers_per_block", cfg.layers_per_block);
        if (o.contains("block_out_channels")) {
          FlexData bo = o.at("block_out_channels");
          if (bo.is_array()) {
            auto av = bo.as_array();
            for (int i = 0; i < 4 && i < (int)av.size(); ++i) {
              cfg.block_out[i] = (int)av[i].as_int(cfg.block_out[i]);
            }
          }
        }
        if (o.contains("patch_size")) {
          FlexData ps = o.at("patch_size");
          if (ps.is_array() && ps.as_array().size() > 0) {
            cfg.patch = (int)ps.as_array()[0].as_int(cfg.patch);
          } else if (ps.is_int()) {
            cfg.patch = (int)ps.as_int(cfg.patch);
          }
        } else {
          // A plain diffusers AutoencoderKL (the FLUX.1 VAE Boogu uses) has no
          // patch_size at all: the latent IS [latent_channels, H/8, W/8].
          cfg.patch = 1;
        }
        // The 1x1 moment convs are optional on the plain AutoencoderKL.
        if (o.contains("use_quant_conv")) {
          cfg.use_quant_conv = o.at("use_quant_conv").as_bool(true);
        }
        if (o.contains("use_post_quant_conv")) {
          cfg.use_post_quant_conv = o.at("use_post_quant_conv").as_bool(true);
        }
        // Scalar whitening (plain AutoencoderKL); the FLUX.2 VAE carries a
        // BatchNorm instead and leaves these absent.
        if (o.contains("scaling_factor")) {
          cfg.scaling_factor = (float)o.at("scaling_factor").as_real(0.0);
        }
        if (o.contains("shift_factor")) {
          cfg.shift_factor = (float)o.at("shift_factor").as_real(0.0);
        }
      }
    }
  }
  m->_cfg = cfg;

  m->_lib_gemm = mc->load_library("dense_gemm");
  m->_lib_elt  = mc->load_library("llm_elementwise");
  m->_lib_sdpa = mc->load_library("sdpa");
  m->_fn_gemm_bias   = m->_lib_gemm.function("dense_gemm_bias_f16");
  // Simdgroup-MMA dense GEMM, x[M,K] @ w[N,K]^T -- the SAME shape the conv
  // im2col produces, and the same kernels the Krea-2 DiT and MOSS codec use.
  // Without these the non-matrix-core path fell to dense_gemm_bias_f16, which
  // is the scalar one-thread-per-output-element GEMM its own comment calls a
  // fallback; at VAE conv shapes that is ~25x off this box's ALU roofline.
  // VPIPE_FLUX2_VAE_NO_STEEL_GEMM=1 restores it (A/B).
  if (std::getenv("VPIPE_FLUX2_VAE_NO_STEEL_GEMM") == nullptr) {
    m->_fn_gemm_t_bm64     = m->_lib_gemm.function("dense_gemm_t_bm64_f16");
    m->_fn_gemm_t_bm64bn64 =
        m->_lib_gemm.function("dense_gemm_t_bm64bn64_f16");
  }
  // Direct conv: the same MMA with the im2col done on-chip. Kept behind an
  // A/B switch because the trade is bandwidth-dependent -- the scratch
  // round-trip this removes is cheap on a high-bandwidth part (the M5 tensor
  // twin of this idea is SLOWER there) and expensive on M4.
  if (std::getenv("VPIPE_VAE_NO_DIRECT_CONV") == nullptr) {
    m->_fn_conv3x3_s1_bn64 =
        m->_lib_gemm.function("conv3x3_gemm_s1_bn64_f16");
    m->_fn_conv3x3_s1_bn32 =
        m->_lib_gemm.function("conv3x3_gemm_s1_bn32_f16");
    m->_fn_conv3x3_s2_bn64 =
        m->_lib_gemm.function("conv3x3_gemm_s2_bn64_f16");
    m->_fn_conv3x3_s2_bn32 =
        m->_lib_gemm.function("conv3x3_gemm_s2_bn32_f16");
    m->_fn_conv3x3_s1_bn128 =
        m->_lib_gemm.function("conv3x3_gemm_s1_bn128_f16");
    m->_fn_conv3x3_s2_bn128 =
        m->_lib_gemm.function("conv3x3_gemm_s2_bn128_f16");
  }
  m->_fn_groupnorm   = m->_lib_elt.function("group_norm_f16");
  m->_fn_mul_sigmoid = m->_lib_elt.function("mul_sigmoid_f16");
  m->_fn_residual    = m->_lib_elt.function("residual_add_f16");
  m->_fn_clamp       = m->_lib_elt.function("clamp_f16");
  m->_fn_sdpa        = m->_lib_sdpa.function("sdpa_full_f16");
  // Materialized-attention pieces (see mat_attn_band_): the row softmax lives
  // in the sdpa lib, the V transpose in the elementwise one.
  m->_fn_softmax_rows = m->_lib_sdpa.function("causal_softmax_rows_f16");
  m->_fn_transpose    = m->_lib_elt.function("transpose_abd_f16");
  m->_fn_sdpa_full_smm = m->_lib_sdpa.function("sdpa_full_mma_f16");
  m->_fn_im2col      = m->_lib_elt.function("im2col_hwc_3x3_f16");
  m->_fn_im2col_tiled = m->_lib_elt.function("im2col_hwc_3x3_tiled_f16");
  m->_fn_im2col_s2   = m->_lib_elt.function("im2col_hwc_3x3_s2_f16");
  m->_fn_im2col_s2_tiled =
      m->_lib_elt.function("im2col_hwc_3x3_s2_tiled_f16");
  m->_fn_upsample    = m->_lib_elt.function("upsample_nearest2x_hwc_f16");
  m->_fn_bias_add    = m->_lib_elt.function("bias_add_rows_f16");
  // Two-pass group norm (see the header). Best-effort: any missing entry
  // point leaves _fast_gnorm false and the single-pass kernel in charge.
  m->_fn_conv_small_cout =
      std::getenv("VPIPE_VAE_NO_SMALL_COUT_CONV") == nullptr
          ? m->_lib_elt.function("conv3x3_hwc_small_cout_f16")
          : metal_compute::ComputeFunction{};
  m->_fn_gn_stats  = m->_lib_elt.function("group_norm_chan_stats_f16");
  m->_fn_gn_reduce = m->_lib_elt.function("group_norm_group_stats_f32");
  m->_fn_gn_apply  = m->_lib_elt.function("group_norm_apply_f16");
  m->_fast_gnorm = m->_fn_gn_stats.valid() && m->_fn_gn_reduce.valid()
                   && m->_fn_gn_apply.valid()
                   && std::getenv("VPIPE_VAE_NO_FAST_GNORM") == nullptr;
  if (m->_fast_gnorm) {
    // Size off the widest channel count any norm in this model will see, so
    // the two scratch buffers are allocated once and reused by every norm.
    int cmax = 0;
    for (int c : m->_cfg.block_out) { cmax = std::max(cmax, c); }
    cmax = std::max(cmax, m->_cfg.latent_channels);
    m->_gn_part =
        mc->make_shared_buffer((std::size_t)kGnBlocks * 2 * cmax * 4);
    m->_gn_stats = mc->make_shared_buffer((std::size_t)cmax * 2 * 4);
    m->_fast_gnorm = !m->_gn_part.empty() && !m->_gn_stats.empty();
  }
  if (!m->_fn_gemm_bias.valid() || !m->_fn_groupnorm.valid() ||
      !m->_fn_mul_sigmoid.valid() || !m->_fn_residual.valid() ||
      !m->_fn_clamp.valid() || !m->_fn_sdpa.valid() || !m->_fn_im2col.valid() ||
      !m->_fn_im2col_tiled.valid() || !m->_fn_im2col_s2_tiled.valid() ||
      !m->_fn_im2col_s2.valid() || !m->_fn_upsample.valid() ||
      !m->_fn_bias_add.valid()) {
    return nullptr;
  }
  // M5 matrix-core dense GEMM (matmul2d) for the conv/1x1 GEMMs, mirroring the
  // Krea-2 VAE. The VAE runs at large M (M = H*W pixels), so the tiled
  // matmul2d amortizes well; bias is folded by a separate bias_add_rows pass
  // (the mma kernel has no bias slot). Steel is kept as the fallback. The
  // fused conv2d_mma is NOT ported: measured ~1.7x SLOWER on M5 UMA than
  // im2col + matmul2d (see the Krea-2 VAE), so the im2col path stays.
  if (mc->supports_matrix_cores() &&
      std::getenv("VPIPE_FLUX2_NO_MMA2") == nullptr) {
    m->_lib_dense_mma = mc->load_library("dense_gemm_mma");
    m->_fn_dense_mma = m->_lib_dense_mma.function("dense_gemm_mma_t_n128_f16");
    m->_fn_dense_mma_deep =
        m->_lib_dense_mma.function("dense_gemm_mma_t_n128x256_f16");
    m->_use_mma2 = m->_fn_dense_mma.valid() && m->_fn_dense_mma_deep.valid();
    if (const char* e = std::getenv("VPIPE_FLUX2_MMA_MIN_M")) {
      m->_mma_min_m = std::atoi(e);
    }
    if (const char* e = std::getenv("VPIPE_FLUX2_VAE_MMA_MAX_M")) {
      m->_mma_max_m = std::atoi(e);   // 0 => no split (reproduces the bug)
    }
    if (const char* e = std::getenv("VPIPE_FLUX2_VAE_MMA_MIN_N")) {
      m->_mma_min_n = std::atoi(e);   // route tiny-N GEMMs (conv_out) to steel
    }
  }
  // FULL matrix-core flash-attention for the mid-block self-attention (head_dim
  // = mid channel dim = block_out[-1]); replaces the scalar O(N^2) sdpa_full_f16
  // that dominates decode at high res. Loaded independently of
  // supports_matrix_cores(): the MPP matmul2d op EMULATES on pre-M5 GPUs (see
  // metal-compute.h), so this tiled flash beats the scalar O(N^2) path there too
  // (the steel flash the DiT uses on M4 tops out at head_dim 256, below the
  // VAE's 384/512). `_fn_sdpa_full_mma.valid()` is the real gate -- a GPU that
  // can't run the metal4.0 tensor kernel keeps the scalar fallback.
  // VPIPE_FLUX2_NO_MMA_ATTN forces scalar.
  if (std::getenv("VPIPE_FLUX2_NO_MMA_ATTN") == nullptr) {
    const int mid_d = cfg.block_out[3];
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
  // NAX hardware convolution2d for the 3x3 convs (see conv3x3_hw_): decided
  // BEFORE the weights load so load_conv3x3_ builds the HWIO twins.
  if (mc->supports_matrix_cores() &&
      std::getenv("VPIPE_VAE_NO_HWCONV") == nullptr) {
    m->_lib_convhw = mc->load_library("conv2d_mma");
    m->_fn_conv_hw_s1 = m->_lib_convhw.function("conv2d_hw_3x3_s1_f16");
    m->_fn_conv_hw_s2 = m->_lib_convhw.function("conv2d_hw_3x3_s2_f16");
    m->_use_hwconv =
        m->_fn_conv_hw_s1.valid() && m->_fn_conv_hw_s2.valid();
  }

  const int top = cfg.block_out[3];               // 512
  bool ok = true;

  // ---- decoder ----
  if (cfg.use_post_quant_conv) {
    m->_post_quant = m->load_conv1x1_(wts, "post_quant_conv");
  }
  m->_conv_in = m->load_conv3x3_(wts, "decoder.conv_in");   // latent -> top
  ok = ok && !m->_conv_in.w.empty();
  ok = ok && m->load_resblock_(wts, "decoder.mid_block.resnets.0.",
                               m->_mid_res0, top, top);
  ok = ok && m->load_resblock_(wts, "decoder.mid_block.resnets.1.",
                               m->_mid_res1, top, top);
  m->_mid_attn = m->load_attn_(wts, "decoder.mid_block.attentions.0.", top);
  ok = ok && !m->_mid_attn.n.g.empty() && !m->_mid_attn.q.w.empty() &&
       !m->_mid_attn.proj.w.empty();

  // Up blocks over the reversed block_out [512,512,256,128]; layers_per_block+1
  // resnets each; upsample on all but the last.
  const int rev[4] = {cfg.block_out[3], cfg.block_out[2], cfg.block_out[1],
                      cfg.block_out[0]};
  m->_up_blocks.resize(4);
  int cin = top;
  for (int i = 0; i < 4; ++i) {
    UpBlock& ub = m->_up_blocks[(std::size_t)i];
    const int out_dim = rev[i];
    ub.resnets.resize((std::size_t)cfg.layers_per_block + 1);
    for (int r = 0; r <= cfg.layers_per_block; ++r) {
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
          wts, "decoder.up_blocks." + std::to_string(i) + ".upsamplers.0.conv");
      ok = ok && !ub.up.w.empty();
    }
  }
  m->_norm_out = m->load_gnorm_(wts, "decoder.conv_norm_out");
  m->_conv_out = m->load_conv3x3_(wts, "decoder.conv_out");   // block_out[0]->3
  ok = ok && !m->_norm_out.g.empty() && !m->_conv_out.w.empty();

  // ---- latent whitening (BatchNorm running stats ONLY: the pipeline applies
  //      (x - running_mean)/sqrt(running_var + eps), NOT the bn affine
  //      gamma/beta). Fold to per-channel a*x + b: a = 1/sqrt(var+eps),
  //      b = -mean*a. Decode inverts (x = (z - b)/a = z/a + mean). ----
  {
    const int C = cfg.dit_channels();            // 128
    std::size_t n = 0;
    std::vector<float> mean = read_f32_(wts.src(), mc, "bn.running_mean", n);
    std::vector<float> var  = read_f32_(wts.src(), mc, "bn.running_var", n);
    m->_bn_a.assign((std::size_t)C, 1.0f);
    m->_bn_b.assign((std::size_t)C, 0.0f);
    if ((int)mean.size() == C && (int)var.size() == C) {
      const float eps = 1e-4f;                   // batch_norm_eps
      for (int c = 0; c < C; ++c) {
        const float a = 1.0f / std::sqrt(var[(std::size_t)c] + eps);
        m->_bn_a[(std::size_t)c] = a;
        m->_bn_b[(std::size_t)c] = -mean[(std::size_t)c] * a;
      }
    } else if (cfg.scaling_factor != 0.0f) {
      // Plain AutoencoderKL: the SCALAR whitening z = (x - shift) * scale folds
      // into the same per-channel affine, so encode (a*x + b) and decode
      // ((z - b)/a) need no special case.
      for (int c = 0; c < C; ++c) {
        m->_bn_a[(std::size_t)c] = cfg.scaling_factor;
        m->_bn_b[(std::size_t)c] = -cfg.shift_factor * cfg.scaling_factor;
      }
    }
  }

  if (!ok) { return nullptr; }
  if (with_encoder && !m->ensure_encoder()) { return nullptr; }
  return m;
}

bool
MetalFlux2Vae::ensure_encoder()
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

bool
MetalFlux2Vae::load_encoder_(WeightSet& wts)
{
  const Config& cfg = _cfg;
  const int L = cfg.latent_channels;
  bool ok = true;
  _enc_conv_in = load_conv3x3_(wts, "encoder.conv_in");   // 3 -> block_out[0]
  ok = ok && !_enc_conv_in.w.empty();

  _enc_down.resize(4);
  int cin = cfg.block_out[0];
  for (int i = 0; i < 4; ++i) {
    DownStage& st = _enc_down[(std::size_t)i];
    const int out_dim = cfg.block_out[i];
    st.resnets.resize((std::size_t)cfg.layers_per_block);
    for (int r = 0; r < cfg.layers_per_block; ++r) {
      ok = ok && load_resblock_(
          wts, "encoder.down_blocks." + std::to_string(i) + ".resnets." +
                   std::to_string(r) + ".",
          st.resnets[(std::size_t)r], cin, out_dim);
      cin = out_dim;
    }
    st.has_down = (i != 3);
    if (st.has_down) {
      st.down = load_conv3x3_(
          wts, "encoder.down_blocks." + std::to_string(i) +
                   ".downsamplers.0.conv");
      ok = ok && !st.down.w.empty();
    }
  }
  const int top = cfg.block_out[3];
  ok = ok && load_resblock_(wts, "encoder.mid_block.resnets.0.", _enc_mid_res0,
                            top, top);
  ok = ok && load_resblock_(wts, "encoder.mid_block.resnets.1.", _enc_mid_res1,
                            top, top);
  _enc_mid_attn = load_attn_(wts, "encoder.mid_block.attentions.0.", top);
  ok = ok && !_enc_mid_attn.n.g.empty() && !_enc_mid_attn.q.w.empty();
  _enc_norm_out = load_gnorm_(wts, "encoder.conv_norm_out");
  _enc_conv_out = load_conv3x3_(wts, "encoder.conv_out");   // top -> 2*L
  if (cfg.use_quant_conv) {
    _quant_conv = load_conv1x1_(wts, "quant_conv");         // 2L -> 2L
  }
  ok = ok && !_enc_norm_out.g.empty() && !_enc_conv_out.w.empty() &&
       (!cfg.use_quant_conv || !_quant_conv.w.empty());
  (void)L;
  return ok;
}

// Pick and load the wide-query-tile mid-attention kernel. The mid-block
// attention is BANDWIDTH-bound, not compute-bound: every threadgroup walks the
// whole of K and V, so the traffic is ceil(hw/BQ) * 2 * hw * D * 2 bytes and BQ
// is the only term that moves it. At 1024x768 (hw = 12288, D = 512) the BQ=8
// kernel moves 36 GB and takes ~166 ms, in BOTH decode and encode -- the single
// largest item in either profile. See sdpa_mma.metal for the derivation and for
// why BQ > 8 needs the register accumulator.
void MetalFlux2Vae::load_wide_attn_(int mid_d)
{
  // All three tiles load; autotune_mid_attn_ picks between them. The tile that
  // wins is a property of the GPU (register file vs bandwidth), so a constant
  // chosen by a sweep on one machine is the wrong default on the next.
  const std::string base = "sdpa_full_mma2_d" + std::to_string(mid_d) + "_q";
  _fn_sdpa_full_wide16 = _lib_sdpa_mma.function(base + "16_f16");
  _fn_sdpa_full_wide32 = _lib_sdpa_mma.function(base + "32_f16");
  _fn_sdpa_full_wide64 = _lib_sdpa_mma.function(base + "64_f16");
}

bool
MetalFlux2Vae::mid_attn_available_(MidAttn k) const
{
  switch (k) {
    case MidAttn::kScalar: return _fn_sdpa.valid();
    case MidAttn::kSmm:    return _fn_sdpa_full_smm.valid();
    case MidAttn::kMma8:   return _fn_sdpa_full_mma.valid();
    case MidAttn::kWide16: return _fn_sdpa_full_wide16.valid();
    case MidAttn::kWide32: return _fn_sdpa_full_wide32.valid();
    case MidAttn::kWide64: return _fn_sdpa_full_wide64.valid();
    case MidAttn::kMat:
      return _fn_gemm_t_bm64bn64.valid() && _fn_softmax_rows.valid() &&
             _fn_transpose.valid();
  }
  return false;
}

void
MetalFlux2Vae::encode_mid_attn_(
    ComputeEncoder& enc, MidAttn kind, const SharedBuffer& q,
    const SharedBuffer& k, const SharedBuffer& v, const SharedBuffer& att,
    std::size_t hw, int C, float scale,
    const std::function<SharedBuffer&(std::size_t)>& alloc,
    const std::function<void(const SharedBuffer&)>& release)
{
  if (kind == MidAttn::kMat) {
    // dense_t computes x[M,K] @ w[N,K]^T, so PV needs V as [C, hw].
    SharedBuffer& vT = alloc(hw * (std::size_t)C);
    enc.set_function(_fn_transpose);
    enc.set_buffer(0, v); enc.set_buffer(1, vT);
    enc.set_constant(2, (int)hw);           // A
    enc.set_constant(3, C);                 // B
    enc.set_constant(4, 1);                 // D (last dim)
    enc.dispatch({1u, (unsigned)C, (unsigned)hw}, {1u, 32u, 8u});
    const int bq = mat_attn_band_(hw);
    SharedBuffer& sc = alloc((std::size_t)bq * hw);
    auto gemm_t = [&](const SharedBuffer& xb, std::size_t xoff,
                      const SharedBuffer& wb, const SharedBuffer& yb,
                      std::size_t yoff, int Kk, int Nn, int Mm) {
      enc.set_function(_fn_gemm_t_bm64bn64);
      enc.set_buffer(0, xb, xoff * 2);
      enc.set_buffer(1, wb); enc.set_buffer(2, wb);   // bias slot unused
      enc.set_buffer(3, yb, yoff * 2);
      enc.set_constant(4, Kk); enc.set_constant(5, Nn);
      enc.set_constant(6, Mm); enc.set_constant(7, 0);
      enc.dispatch({(unsigned)(((Nn + 63) / 64) * 32),
                    (unsigned)(((Mm + 63) / 64) * 2), 2}, {32, 2, 2});
    };
    for (int q0 = 0; q0 < (int)hw; q0 += bq) {
      const int rows = std::min(bq, (int)hw - q0);
      // scores[rows, hw] = q[q0.., C] @ k[hw, C]^T
      gemm_t(q, (std::size_t)q0 * C, k, sc, 0, C, (int)hw, rows);
      // Plain scaled row softmax: q_offset = hw puts every key at or below the
      // causal bound, so the mask never fires (window/banded off).
      enc.set_function(_fn_softmax_rows);
      enc.set_buffer(0, sc);
      enc.set_constant(1, rows); enc.set_constant(2, (int)hw);
      enc.set_constant(3, (int)hw);         // q_offset -> unmasked
      enc.set_constant(4, scale);
      enc.set_constant(5, 0);               // window
      enc.set_constant(6, 0);               // banded
      enc.dispatch({256u, (unsigned)rows, 1u}, {256u, 1, 1});
      // att[q0.., C] = P[rows, hw] @ vT[C, hw]^T
      gemm_t(sc, 0, vT, att, (std::size_t)q0 * C, (int)hw, C, rows);
    }
    release(sc); release(vT);
    return;
  }
  // Every flash member speaks one buffer contract; only the grid differs.
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

// Time every available member on a synthetic mid block and keep the winner.
// autotune_vote interleaves the candidates each round and scores the per-round
// winner, so the SoC-power-gated clock cancels instead of handing whichever arm
// ran first an advantage.
void
MetalFlux2Vae::autotune_mid_attn_(MetalCompute* mc, int C)
{
  // Capability guess, used as-is when the tune is skipped or cannot run.
  _attn_pick = MidAttn::kScalar;
  const bool smm_ok = mid_attn_available_(MidAttn::kSmm) &&
                      (C % 64 == 0) && (C <= 512);
  if (smm_ok) { _attn_pick = MidAttn::kSmm; }
  if (mc->supports_matrix_cores()) {
    if (mid_attn_available_(MidAttn::kMma8))   { _attn_pick = MidAttn::kMma8; }
    if (mid_attn_available_(MidAttn::kWide32)) { _attn_pick = MidAttn::kWide32; }
  }
  // Honour the existing manual overrides ahead of any measurement.
  if (std::getenv("VPIPE_FLUX2_VAE_ATTN_SMM") != nullptr && smm_ok) {
    _attn_pick = MidAttn::kSmm; return;
  }
  if (std::getenv("VPIPE_VAE_MAT_ATTN_MMA2") != nullptr &&
      mid_attn_available_(MidAttn::kMat)) {
    _attn_pick = MidAttn::kMat; return;
  }
  if (const char* e = std::getenv("VPIPE_FLUX2_VAE_ATTN_BQ")) {
    const int b = std::atoi(e);
    if (b == 8  && mid_attn_available_(MidAttn::kMma8))   { _attn_pick = MidAttn::kMma8; }
    if (b == 16 && mid_attn_available_(MidAttn::kWide16)) { _attn_pick = MidAttn::kWide16; }
    if (b == 32 && mid_attn_available_(MidAttn::kWide32)) { _attn_pick = MidAttn::kWide32; }
    if (b == 64 && mid_attn_available_(MidAttn::kWide64)) { _attn_pick = MidAttn::kWide64; }
    return;
  }
  // Candidates: everything that loaded, minus the scalar fallback (orders
  // slower, and only there for a GPU with nothing else). The shared tuner
  // times them through the SAME encode_mid_attn_ the decode uses.
  std::vector<MidAttn> cands;
  for (MidAttn k : {MidAttn::kSmm, MidAttn::kMma8, MidAttn::kWide16,
                    MidAttn::kWide32, MidAttn::kWide64, MidAttn::kMat}) {
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
             const vae_mid_attn::Alloc& al, const vae_mid_attn::Release& re) {
        encode_mid_attn_(enc, kind, qq, kk, vv, oo, hw, c, sc, al, re);
      },
      &detail);
  const double tune_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
  if (mc->session() != nullptr && !detail.empty()) {
    // VPIPE_VAE_ATTN_TUNE_LOG promotes this to a normal log line. Which member
    // won is the whole point of the tune and differs per machine, so it has to
    // be observable without a debug build when porting to a new GPU.
    const auto line = fmt(
        "MetalFlux2Vae: mid-attn autotune (D={}) -> {} [{}] in {} ms",
        C, vae_mid_attn::name(_attn_pick), detail, (long long)tune_ms);
    if (std::getenv("VPIPE_VAE_ATTN_TUNE_LOG") != nullptr) {
      mc->session()->log_normal(line);
    } else {
      mc->session()->log_debug(line);
    }
  }
}

// Time the two 3x3 fallback routes on a synthetic conv and keep the winner.
// The probe runs through tiled_conv3x3_ -- the SHIPPING entrance -- with
// _conv_pick steering it, so what is measured is the route a decode would
// actually take, im2col arm and matmul2d GEMM included. Benching the kernels
// in isolation is what made the earlier conv-phase numbers misleading: that
// harness compared the on-chip gather against a STEEL GEMM, not against the
// matrix-core one an M5 really uses.
// Will any 3x3 in this pass miss the hardware conv? That op needs whole 8x8
// destination tiles, cout % 64, and int32-addressable extents, so it declines
// on a grid that is not a multiple of 8 or once cin*W*H passes 2^31. If none
// of that bites, the fallback carries nothing and the tune would be pure load
// time. conv_out is excluded: it has its own small-cout kernel.
void
MetalFlux2Vae::maybe_tune_conv_(int H, int W)
{
  if (_conv_tuned || _mc == nullptr) { return; }
  _conv_tuned = true;
  // An explicit force still has to populate the map, or the override silently
  // does nothing at a resolution where the fallback would not otherwise run --
  // which is exactly how the direct-conv A/B went vacuous once before.
  if (std::getenv("VPIPE_VAE_DIRECT_CONV_MMA2") != nullptr) {
    _conv_force_onchip = true;
    return;
  }
  bool needed = !_use_hwconv;
  if (!needed) {
    constexpr std::size_t kIdxMax = 0x7fffffffull;
    int h = H, w = W;                       // walk the decoder's pyramid
    for (int lvl = 3; lvl >= 0 && !needed; --lvl) {
      const int cin = _cfg.block_out[lvl];
      if ((h % 8) != 0 || (w % 8) != 0) { needed = true; break; }
      if ((std::size_t)cin * w * h > kIdxMax) { needed = true; break; }
      h /= 2; w /= 2;
    }
  }
  if (!needed) { return; }
  autotune_conv3x3_(_mc);
}

vae_conv3x3::Kind
MetalFlux2Vae::conv_route_(int cin, int cout) const
{
  if (_conv_force_onchip) { return vae_conv3x3::Kind::kOnChip; }
  const auto it = _conv_pick.find(std::make_pair(cin, cout));
  return (it != _conv_pick.end()) ? it->second : vae_conv3x3::Kind::kIm2col;
}

void
MetalFlux2Vae::autotune_conv3x3_(MetalCompute* mc)
{
  const bool have_direct =
      _fn_conv3x3_s1_bn32.valid() || _fn_conv3x3_s1_bn64.valid();
  if (!have_direct) { return; }                 // nothing to choose

  // Every distinct (cin, cout) this decoder's 3x3s use, minus the shapes that
  // never reach the fallback: cin % 32 != 0 (the on-chip gather declines) and
  // cout <= kSmallCoutMax (conv_out, which has its own kernel).
  std::vector<std::pair<int, int>> shapes;
  auto want = [&](const Conv& c) {
    if (c.cin <= 0 || (c.cin % 32) != 0 || c.cout <= kSmallCoutMax) { return; }
    const auto key = std::make_pair(c.cin, c.cout);
    for (const auto& s : shapes) { if (s == key) { return; } }
    shapes.push_back(key);
  };
  auto want_rb = [&](const ResBlock& rb) {
    want(rb.c1); want(rb.c2);
  };
  want(_conv_in); want_rb(_mid_res0); want_rb(_mid_res1);
  for (const UpBlock& ub : _up_blocks) {
    for (const ResBlock& rb : ub.resnets) { want_rb(rb); }
    if (ub.has_up) { want(ub.up); }
  }
  if (shapes.empty()) { return; }
  // The probe resolution SCALES WITH THE SHAPE, because in a U-net decoder
  // channel count and resolution are inversely related: the 512-channel convs
  // run at the smallest grid and the 128-channel ones at full size. Probing
  // every shape at one resolution is wrong in both directions -- MEASURED, a
  // flat 128x128 probe called 128->128 for im2col (2.099 vs 2.255 ms) while at
  // the 256x256/512x512/768x768 where that conv actually runs the on-chip
  // gather wins by 1.10x. Sizing the probe as block_out[3]/cout tracks the
  // pyramid and, because a U-net holds work roughly constant per level, keeps
  // every shape's tune about equally cheap.
  int probe_base = 64;
  if (const char* e = std::getenv("VPIPE_VAE_CONV_TUNE_HW")) {
    probe_base = std::max(32, std::atoi(e));
  }
  const std::vector<vae_conv3x3::Kind> cands = {
      vae_conv3x3::Kind::kIm2col, vae_conv3x3::Kind::kOnChip};
  std::string detail;
  const auto t0 = std::chrono::steady_clock::now();
  for (const auto& sh : shapes) {
    const int cin = sh.first, cout = sh.second;
    const int top = _cfg.block_out[3] > 0 ? _cfg.block_out[3] : cout;
    const int H = std::min(256, std::max(probe_base,
                                         probe_base * (top / std::max(1, cout))));
    const int W = H;
    Conv c;
    c.cin = cin; c.cout = cout; c.k = 9 * cin;
    c.w = mc->make_shared_buffer((std::size_t)cout * 9 * cin * 2);
    c.b = mc->make_shared_buffer((std::size_t)cout * 2);
    SharedBuffer in  = mc->make_shared_buffer((std::size_t)H * W * cin * 2);
    SharedBuffer out = mc->make_shared_buffer((std::size_t)H * W * cout * 2);
    const std::size_t cap = (std::size_t)H * W * 9 * cin;
    SharedBuffer col = mc->make_shared_buffer(cap * 2);
    if (c.w.empty() || c.b.empty() || in.empty() || out.empty() ||
        col.empty()) {
      continue;                                 // leave this shape on im2col
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
    // The probe runs through tiled_conv3x3_, the SHIPPING entrance, with the
    // map steering it -- so what is timed is the route a decode really takes,
    // matmul2d GEMM included. Benching the kernels standalone is what made the
    // conv-phase numbers misleading: that harness compared the gather against
    // a STEEL GEMM, not the matrix-core one this path actually reaches.
    _conv_pick[sh] = vae_conv3x3::autotune(
        "flux2", cin, cout, /*stride=*/1, cands, vae_conv3x3::Kind::kIm2col,
        [&](int i) -> double {
          _conv_pick[sh] = cands[(std::size_t)i];
          return autotune_time(mc, 1, [&](ComputeEncoder& enc) {
            tiled_conv3x3_(enc, in, out, H, W, c, /*stride=*/1, col, cap);
          });
        },
        &one);
    if (!detail.empty()) { detail += ", "; }
    detail += std::to_string(cin) + "->" + std::to_string(cout) + " " +
              vae_conv3x3::name(_conv_pick[sh]);
    if (!one.empty()) { detail += " (" + one + ")"; }
  }
  const double tune_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
  if (mc->session() != nullptr && !detail.empty()) {
    const auto line = fmt("MetalFlux2Vae: conv3x3 autotune -- {} in {} ms",
                          detail, (long long)tune_ms);
    if (std::getenv("VPIPE_VAE_ATTN_TUNE_LOG") != nullptr) {
      mc->session()->log_normal(line);
    } else {
      mc->session()->log_debug(line);
    }
  }
}

void
MetalFlux2Vae::group_norm_(ComputeEncoder& enc, const SharedBuffer& in,
                           const SharedBuffer& gamma, const SharedBuffer& beta,
                           const SharedBuffer& out, int rows, int C, int G,
                           float eps)
{
  if (!_fast_gnorm || _gn_part.empty() || _gn_stats.empty()) {
    enc.set_function(_fn_groupnorm);
    enc.set_buffer(0, in); enc.set_buffer(1, gamma); enc.set_buffer(2, beta);
    enc.set_buffer(3, out);
    enc.set_constant(4, rows); enc.set_constant(5, C);
    enc.set_constant(6, G); enc.set_constant(7, eps);
    enc.dispatch({256, (unsigned)G, 1}, {256, 1, 1});
    return;
  }
  // More row blocks than rows would leave empty blocks contributing zero
  // partials -- harmless, but the reduce then averages over a `rows` that
  // does not match what was summed, so clamp instead.
  const int NB = std::min(kGnBlocks, std::max(1, rows));
  enc.set_function(_fn_gn_stats);
  enc.set_buffer(0, in); enc.set_buffer(1, _gn_part);
  enc.set_constant(2, rows); enc.set_constant(3, C); enc.set_constant(4, NB);
  enc.dispatch({(unsigned)(256 * NB), 1, 1}, {256, 1, 1});
  enc.set_function(_fn_gn_reduce);
  enc.set_buffer(0, _gn_part); enc.set_buffer(1, _gn_stats);
  enc.set_constant(2, rows); enc.set_constant(3, C); enc.set_constant(4, G);
  enc.set_constant(5, NB); enc.set_constant(6, eps);
  enc.dispatch({(unsigned)(256 * G), 1, 1}, {256, 1, 1});
  enc.set_function(_fn_gn_apply);
  enc.set_buffer(0, in); enc.set_buffer(1, gamma); enc.set_buffer(2, beta);
  enc.set_buffer(3, out); enc.set_buffer(4, _gn_stats);
  enc.set_constant(5, C); enc.set_constant(6, G);
  enc.dispatch({(unsigned)C, (unsigned)rows, 1}, {256, 1, 1});
}

bool
MetalFlux2Vae::conv3x3_small_cout_(ComputeEncoder& enc, const SharedBuffer& in,
                                   const Conv& c, const SharedBuffer& out,
                                   int H, int W, int stride)
{
  // stride 1 only: the stride-2 downsample uses the asymmetric (0,1,0,1)
  // padding, and every small-cout conv in these VAEs is stride 1 anyway.
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
MetalFlux2Vae::conv3x3_hw_(ComputeEncoder& enc, const SharedBuffer& in,
                           const Conv& c, const SharedBuffer& out,
                           int H, int W, int stride)
{
  // NAX hardware convolution2d (probe-established semantics -- see
  // conv2d_mma.metal): the op reads the full NHWC activation itself,
  // zero-filled pad-1 halo included; no im2col scratch or DRAM round-trip.
  // Whole 8x8 dest tiles + 64-channel tiles required; others (the 3-ch
  // conv_in/out, tiny latent grids) keep the im2col path.
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

void
MetalFlux2Vae::conv_gemm_bias_(ComputeEncoder& enc, const SharedBuffer& x,
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
    // >=1024px decode has M = H*W = 2^20), so split a tall GEMM into row-chunks
    // of at most _mma_max_m and dispatch each over its own row-range. The
    // dense_gemm_mma tensors are column-major (the row dim's stride is K for x /
    // N for y), so a contiguous r0*K / r0*N ELEMENT offset (x2 for f16 bytes)
    // selects rows [r0, r0+mc). _mma_max_m == 0 disables the split (A/B).
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
  // No matrix cores: the simdgroup-MMA GEMM, not the scalar one. Same
  // math, same [M,K]x[N,K]^T shape; bias is folded afterwards by
  // bias_add_rows exactly as the matmul2d path does (these entry points
  // have a bias slot, but the proven-in-this-file route is the separate
  // pass, and it costs one cheap dispatch).
  const bool wide_n = N >= 64 && _fn_gemm_t_bm64bn64.valid();
  const metal_compute::ComputeFunction& gt =
      wide_n ? _fn_gemm_t_bm64bn64 : _fn_gemm_t_bm64;
  if (gt.valid()) {
    const int bm = 64, bn = wide_n ? 64 : 32;
    enc.set_function(gt);
    enc.set_buffer(0, x); enc.set_buffer(1, w); enc.set_buffer(2, w);
    enc.set_buffer(3, y, ybase * 2);
    enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, M);
    enc.set_constant(7, 0);
    enc.dispatch({(unsigned)(((N + bn - 1) / bn) * 32),
                  (unsigned)(((M + bm - 1) / bm) * 2), 2}, {32, 2, 2});
    if (!b.empty()) {
      enc.set_function(_fn_bias_add);
      enc.set_buffer(0, y, ybase * 2); enc.set_buffer(1, b);
      enc.set_constant(2, N);
      enc.set_constant(3, (unsigned)((std::size_t)M * N));
      // 2D grid {N, M}: a 1D {M*N} grid overflows at VAE sizes.
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

int
MetalFlux2Vae::mat_attn_band_(std::size_t hw) const noexcept
{
  if (hw == 0) { return 0; }
  // Largest 64-row-aligned band whose [band, hw] f16 scores fit the cap. 64 is
  // the GEMM's BM, so a band boundary never splits an output tile.
  std::size_t band = kMatAttnScoreBytes / (hw * 2);
  if (band > hw) { band = hw; }
  band &= ~(std::size_t)63;
  if (band < 64) { band = std::min<std::size_t>(hw, 64); }
  return (int)band;
}

bool
MetalFlux2Vae::direct_conv3x3_(ComputeEncoder& enc, const SharedBuffer& in,
                               const SharedBuffer& out, int H, int W,
                               const Conv& c, int stride)
{
  // Whether this GPU wants the on-chip gather at all is MEASURED in
  // autotune_conv3x3_ (see vae-conv3x3-tune.h). It is not a capability
  // question: this kernel runs the contraction on simdgroup MMA, and where
  // matmul2d is available the im2col arm it replaces already reaches the
  // matrix units -- so on an M5 stepping in front of that is a downgrade,
  // while on an M4 Pro the round trip it removes is the whole cost.
  if (conv_route_(c.cin, c.cout) != vae_conv3x3::Kind::kOnChip) {
    return false;
  }
  // The gather only vectorizes when a K tile is one tap, i.e. cin % 32 == 0
  // (BK). Anything else falls to the per-element divide form, which measured
  // SLOWER than im2col -- so keep those shapes (conv_in's tiny cin) on im2col.
  const bool s2 = (stride == 2);
  if (c.cin % 32 != 0) { return false; }
  // BN=64 measured FASTER than BN=128 once the gather vectorized (13.00 vs
  // 13.90 ms at cout=128): with the gather no longer the cost, the narrower
  // tile's lower threadgroup-memory footprint wins on occupancy. The wider
  // entry stays for A/B under VPIPE_VAE_CONV_BN128.
  static const bool want128 = std::getenv("VPIPE_VAE_CONV_BN128") != nullptr;
  int bn = 32;
  const metal_compute::ComputeFunction* fnp =
      s2 ? &_fn_conv3x3_s2_bn32 : &_fn_conv3x3_s1_bn32;
  if (want128 && c.cout >= 128 &&
      (s2 ? _fn_conv3x3_s2_bn128 : _fn_conv3x3_s1_bn128).valid()) {
    bn = 128;
    fnp = s2 ? &_fn_conv3x3_s2_bn128 : &_fn_conv3x3_s1_bn128;
  } else if (c.cout >= 64 &&
             (s2 ? _fn_conv3x3_s2_bn64 : _fn_conv3x3_s1_bn64).valid()) {
    bn = 64;
    fnp = s2 ? &_fn_conv3x3_s2_bn64 : &_fn_conv3x3_s1_bn64;
  }
  const metal_compute::ComputeFunction& fn = *fnp;
  if (!fn.valid()) { return false; }
  const int OH = (stride == 2) ? H / 2 : H;
  const int OW = (stride == 2) ? W / 2 : W;
  const int M = OH * OW;
  if (M <= 0 || c.cin <= 0 || c.cout <= 0) { return false; }
  const int bm = 64;
  enc.set_function(fn);
  enc.set_buffer(0, in);
  enc.set_buffer(1, c.w);
  enc.set_buffer(2, c.b.empty() ? c.w : c.b);
  enc.set_buffer(3, out);
  enc.set_constant(4, H);      enc.set_constant(5, W);
  enc.set_constant(6, c.cin);  enc.set_constant(7, c.cout);
  enc.set_constant(8, OH);     enc.set_constant(9, OW);
  enc.set_constant(10, c.b.empty() ? 0 : 1);
  enc.dispatch({(unsigned)(((c.cout + bn - 1) / bn) * 32),
                (unsigned)(((M + bm - 1) / bm) * 2), 2}, {32, 2, 2});
  return true;
}

void
MetalFlux2Vae::tiled_conv3x3_(ComputeEncoder& enc, const SharedBuffer& in,
                              const SharedBuffer& out, int H, int W,
                              const Conv& c, int stride, const SharedBuffer& col,
                              std::size_t cap)
{
  // No scratch, no row banding, no im2col round-trip -- the whole conv is one
  // dispatch when the direct kernel is available.
  if (direct_conv3x3_(enc, in, out, H, W, c, stride)) { return; }
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

std::size_t
MetalFlux2Vae::decode_peak_bytes(int h16, int w16) const noexcept
{
  if (h16 <= 0 || w16 <= 0) { return 0; }
  // Pixels per latent cell: the conv trunk is 8x, times the patch factor.
  const std::size_t px = (std::size_t)8 * _cfg.patch;
  const std::size_t Hout = (std::size_t)h16 * px;
  const std::size_t Wout = (std::size_t)w16 * px;
  // The top up-block's FIRST resnet reads block_out[1] channels (256) at full
  // res (the upsampled level-2 output, before it reduces to block_out[0]=128),
  // so the largest full-res im2col is [Hout*Wout, 9*block_out[1]] -- TWICE the
  // conv_out/block_out[0] figure. Budget the real peak (block_out[0] alone
  // under-modelled it 2x and risked a false-pass -> a mid-flight OOM).
  const std::size_t base =
      (std::size_t)std::max(_cfg.block_out[0], _cfg.block_out[1]);
  const std::size_t top = Hout * Wout * base * 2;  // one full-res base-ch f16 buf
  // The per-up-block command-buffer split (default on; VPIPE_FLUX2_NO_VAE_SPLIT
  // opts out) commits + frees each up-level, so the resident peak is ONE
  // level's working set, not the summed up-path.
  const bool split = std::getenv("VPIPE_FLUX2_NO_VAE_SPLIT") == nullptr;
  if (_use_hwconv) {
    // Hardware-conv path (DEFAULT): the big convs run through conv3x3_hw_ and
    // NEVER materialize the [Hout*Wout, 9*base] im2col scratch -- only the tiny
    // 3-ch fallback convs im2col (their own small col buffers). The resident
    // peak is the per-level activation pool. MEASURED ~5.5x `top` at 1024^2
    // (2.8 GB decode delta) on the split path; budget 7x for the hw-conv
    // workspace + margin (a miss is caught cleanly by the per-level wait_ok()).
    // The old 9x-im2col figure (13.5x top split-on) was a ~2.3x PHANTOM -- it
    // budgeted scratch this path never allocates and, once the doubled `base`
    // pushed it to 6912 MB, falsely rejected a 1024 decode sharing the box with
    // other resident models (it fit in ~3 GB the whole time).
    return split ? top * 7 : top * 10;     // split frees per level; no-split sums
  }
  // im2col fallback (VPIPE_VAE_NO_HWCONV, or any non-matrix-core M4 GPU): the big
  // convs materialize im2col, but conv3x3 ROW-TILES it into bands bounded to fit
  // the free headroom (im2col_cap), so the split-on peak is the activation pool
  // + one band -- the SAME order as the hw path (the band never exceeds the
  // headroom). Budget top*7 like the hw path; the actual band shrinks to fit.
  // Split-off keeps everything in one command buffer (pool holds every level),
  // so keep the conservative summed im2col figure there.
  if (split) { return top * 7; }
  const std::size_t im2col = Hout * Wout * 9 * base * 2;
  return im2col * 2;
}

int
MetalFlux2Vae::decode_tile_side_(std::size_t budget) const noexcept
{
  if (budget == 0) { return 0; }
  // Walk down in 4-cell steps; decode_peak_bytes is monotone in the side, so
  // the first fit is the largest. (A side is latent cells; one cell is
  // patch*8 output pixels.)
  for (int s = 128; s >= kTileMin16; s -= 4) {
    if (decode_peak_bytes(s, s) <= budget) { return s; }
  }
  return 0;
}

SharedBuffer
MetalFlux2Vae::decode_tiled_(const SharedBuffer& z, int h16, int w16,
                             int tile16, std::string* err)
{
  auto fail = [&](std::string m) -> SharedBuffer {
    if (err != nullptr) { *err = std::move(m); }
    return {};
  };
  const int Cdit = _cfg.dit_channels();
  const int px = _cfg.patch * 8;                 // output pixels per cell
  const int H = h16 * px, W = w16 * px;
  if (tile16 < kTileMin16) { return fail("tiled decode: window too small"); }
  const int ov = std::max(2, tile16 * kTileOvNum / kTileOvDen);
  const int step = tile16 - ov;
  if (step < 1) { return fail("tiled decode: overlap exceeds the window"); }

  const std::size_t hw = (std::size_t)H * W;
  std::vector<float> acc((std::size_t)3 * hw, 0.0f);   // weighted RGB sum
  std::vector<float> wsum(hw, 0.0f);                   // weight sum
  const auto* zsrc = static_cast<const _Float16*>(z.contents());
  if (zsrc == nullptr) { return fail("tiled decode: latent not host-visible"); }

  // Cross-fade weight for an OUTPUT pixel: ramps up over the overlap unless the
  // window starts at the image edge, down over it unless the window ends there,
  // so interior windows sum to 1 across a seam. The ramp width is the overlap
  // in PIXELS (ov is latent cells, one cell = px pixels) -- ramping over `ov`
  // pixels instead fades across 1/16th of the overlap and leaves a visible
  // seam, which the per-column check below catches.
  const int ovp = ov * px;
  auto ramp = [&](int i, int n, bool at_lo, bool at_hi) {
    float w = 1.0f;
    if (!at_lo && i < ovp)          { w = (float)(i + 1) / (float)(ovp + 1); }
    if (!at_hi && i >= n - ovp) {
      const float t = (float)(n - i) / (float)(ovp + 1);
      w = std::min(w, t);
    }
    return std::max(w, 1e-3f);
  };

  int ntiles = 0;
  for (int y0 = 0; y0 < h16; y0 += step) {
    const int th = std::min(tile16, h16 - y0);
    if (th <= 0) { break; }
    const bool y_lo = (y0 == 0), y_hi = (y0 + th >= h16);
    for (int x0 = 0; x0 < w16; x0 += step) {
      const int tw = std::min(tile16, w16 - x0);
      if (tw <= 0) { break; }
      const bool x_lo = (x0 == 0), x_hi = (x0 + tw >= w16);
      // Slice the latent window out of z[Cdit, h16, w16] (channel-first).
      SharedBuffer zt =
          _mc->make_shared_buffer((std::size_t)Cdit * th * tw * 2);
      if (zt.empty()) { return fail("tiled decode: window latent alloc failed"); }
      auto* zd = static_cast<_Float16*>(zt.contents());
      for (int c = 0; c < Cdit; ++c) {
        for (int y = 0; y < th; ++y) {
          const _Float16* srow =
              zsrc + ((std::size_t)c * h16 + (y0 + y)) * w16 + x0;
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
      od[(std::size_t)c * hw + p] = (_Float16)(acc[(std::size_t)c * hw + p] * inv);
    }
  }
  if (_mc->session() != nullptr) {
    _mc->session()->log_normal(fmt(
        "FLUX.2 VAE decode: TILED {}x{} from {} windows of {} latent cells "
        "(overlap {}) -- per-window attention, cross-faded seams",
        W, H, ntiles, tile16, ov));
  }
  return out;
}

SharedBuffer
MetalFlux2Vae::decode(const SharedBuffer& z, int h16, int w16, std::string* err)
{
  auto fail = [&](std::string m) -> SharedBuffer {
    if (err != nullptr) { *err = std::move(m); }
    return {};
  };
  const int Cdit = _cfg.dit_channels();          // 128
  const int L = _cfg.latent_channels;            // 32
  const int P = _cfg.patch;                      // 2
  const std::size_t hw16 = (std::size_t)h16 * w16;
  if (z.byte_size() < (std::size_t)Cdit * hw16 * 2) {
    return fail("input latent smaller than [dit_channels, h16, w16]");
  }
  MetalCompute* mc = _mc;
  const int G = _cfg.norm_groups;
  const float geps = _cfg.norm_eps;
  int h8 = h16 * P, w8 = w16 * P;                // latent spatial after unpatch
  const int Hout = h8 * 8, Wout = w8 * 8;        // conv trunk is 8x

  std::size_t decode_headroom = 0;   // free working set, for the im2col band cap
  {
    const MetalCompute::MemoryBudget mb = mc->memory_budget();
    decode_headroom = (mb.recommended != 0) ? mb.headroom : 0;
    const std::size_t need = decode_peak_bytes(h16, w16);
    bool gpu_short = (mb.recommended != 0) && !mb.fits(need);
    bool ram_short = !mb.fits_physical(need);
    // Test/capacity hook: pretend the decode budget is this many MB. The
    // auto-switch is a DECISION about memory, and a 64 GB box can never take
    // the short branch on its own -- without this the fallback would only ever
    // be exercised on the hardware that needs it. (VPIPE_RAM_LIMIT_MB does not
    // apply here: that one sizes model_memory against total RAM, not this live
    // per-decode budget.)
    std::size_t budget_override = 0;
    if (const char* e = std::getenv("VPIPE_VAE_BUDGET_MB")) {
      budget_override = (std::size_t)std::max(0, std::atoi(e)) << 20;
      if (budget_override != 0) {
        gpu_short = need > budget_override;
        ram_short = need > budget_override;
      }
    }
    // Force-tile knob (A/B and testing on a box that would otherwise fit).
    int forced = 0;
    if (const char* e = std::getenv("VPIPE_VAE_TILE")) { forced = std::atoi(e); }
    if (gpu_short || ram_short || forced > 0) {
      // Fall back to a TILED decode rather than refusing: the peak then tracks
      // one window, not the output area. See decode_tiled_ for what this costs
      // numerically (per-window attention). Tiling only helps if the window is
      // actually smaller than the image -- if the whole latent already fits in
      // one window and still does not fit in memory, there is nothing to split,
      // so fail as before (this is also what terminates the recursion, since
      // decode_tiled_ re-enters decode() per window).
      // Size the window against the SAME MARGINED budgets the accept tests
      // above use. fits()/fits_physical() keep 5%/10% back, so sizing against
      // the raw figures hands back the largest window inside that reserve --
      // which the window's own decode() then rejects. And because that window
      // is the whole latent, the recursion has nowhere left to go and the
      // decode fails outright instead of tiling. MEASURED on a 16 GB box at
      // 2048x2048: need 7406 MB against 7796 MB reclaimable picked a 92-cell
      // window, 10% over what fits_physical would accept, and no image above
      // ~1.5K could decode at all.
      std::size_t budget = (std::size_t)-1;
      if (mb.available_physical != 0) {
        budget = (std::size_t)((double)mb.available_physical * 0.90);
      }
      if (mb.recommended != 0) {
        const auto gpu = (std::size_t)((double)mb.headroom * 0.95);
        if (gpu < budget) { budget = gpu; }
      }
      if (budget_override != 0) { budget = budget_override; }
      int tile16 = (forced > 0) ? forced : decode_tile_side_(budget);
      if (tile16 > 0 && (tile16 < h16 || tile16 < w16) &&
          std::getenv("VPIPE_VAE_NO_TILE") == nullptr) {
        return decode_tiled_(z, h16, w16, tile16, err);
      }
    }
    if (gpu_short) {
      return fail(fmt(
          "insufficient GPU memory for a {}x{} decode: need ~{} MB, {} MB free "
          "of {} MB working set (lower the resolution or free other resident "
          "models)", Wout, Hout, need >> 20, mb.headroom >> 20,
          mb.recommended >> 20)());
    }
    // True-physical-pressure backstop: reclaimable RAM (counting mmap'd/clean
    // weight pages the OS can evict) must also cover the decode, else the GPU
    // command buffer would OOM mid-flight instead of a clean rejection here.
    if (ram_short) {
      return fail(fmt(
          "insufficient free RAM for a {}x{} decode: need ~{} MB, ~{} MB "
          "reclaimable (close other apps, lower the resolution, or free "
          "resident models)", Wout, Hout, need >> 20,
          mb.available_physical >> 20)());
    }
  }

  // Un-bn + unpatchify on host: z[Cdit, h16, w16] (channel-first) ->
  // latent channel-last [h8*w8, L]. pixel-unshuffle convention
  // (c, ph, pw) -> channel c*P*P + ph*P + pw at (2i+ph, 2j+pw).
  // NOTE: VERIFY the unshuffle channel order + bn inversion vs a golden.
  const std::size_t hw8 = (std::size_t)h8 * w8;
  maybe_tune_conv_(Hout, Wout);
  SharedBuffer latent = mc->make_shared_buffer(hw8 * (std::size_t)L * 2);
  if (latent.empty()) { return fail("latent allocation failed"); }
  {
    const auto* s = static_cast<const _Float16*>(z.contents());
    auto* d = static_cast<_Float16*>(latent.contents());
    for (int cc = 0; cc < Cdit; ++cc) {
      const float a = _bn_a[(std::size_t)cc], b = _bn_b[(std::size_t)cc];
      const int c = cc / (P * P);
      const int ph = (cc % (P * P)) / P;
      const int pw = cc % P;
      for (int i = 0; i < h16; ++i) {
        for (int j = 0; j < w16; ++j) {
          const float zv = (float)s[((std::size_t)cc * h16 + i) * w16 + j];
          const float x = (zv - b) / a;          // inverse bn
          const int oi = i * P + ph, oj = j * P + pw;
          d[((std::size_t)oi * w8 + oj) * L + c] = (_Float16)x;
        }
      }
    }
  }

  // Buffer pool: `alloc` reuses a released buffer of sufficient capacity rather
  // than allocating a fresh one every op. The VAE is a serial feed-forward
  // chain (DispatchType::Serial), so reusing a released buffer is safe -- serial
  // dispatch orders the reuse strictly after the last read of the old contents,
  // and an in-use buffer is never handed out (so an op's output never aliases a
  // live input). This bounds the live set to the concurrent working set (~3
  // top-res buffers) instead of the whole decode, so a 1024^2 decode fits one
  // command buffer. VPIPE_FLUX2_NO_VAE_POOL disables reuse (then the per-level
  // command-buffer split below bounds the peak instead).
  struct Slot { SharedBuffer buf; std::size_t cap; bool used; };
  std::deque<Slot> pool;
  bool alloc_ok = true;
  const bool use_pool = std::getenv("VPIPE_FLUX2_NO_VAE_POOL") == nullptr;
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
  // im2col band scratch (ROW-TILED). im2col convs stream their [H*W, 9*cin]
  // expansion in output-row bands so the shared col scratch is bounded to ONE
  // band -- a 1024^2 im2col is otherwise ~2.4 GB (conv_out, 128-ch, even on the
  // hw-conv path) to ~4.6 GB (256-ch, the non-hw M4 path). `im2col_cap` (ELEMS)
  // caps a band by TWO limits: (1) memory -- reserve the level's activation pool
  // (~decode_peak_bytes = 7x one full-res buffer), give the band the rest of the
  // headroom, floored at a few output rows; (2) CORRECTNESS -- the matmul2d op
  // corrupts output rows past M ~ 2^18-2^19 for the large-K 3x3 im2col GEMMs
  // (MEASURED at 1024: a 2^19-row band bands the image, 2^18 is clean; this is
  // LOWER than the small-K _mma_max_m=2^19 the chunk split assumes), so cap a
  // band at kSafeBand = _mma_max_m/2 rows for the widest conv, which also keeps
  // each band a single un-chunked GEMM. Each conv derives its own band rows =
  // im2col_cap / (9*cin), re-capped at kSafeBand below.
  // VPIPE_FLUX2_VAE_BAND_ROWS overrides the memory budget (still safe-capped).
  const std::size_t base_max =
      (std::size_t)std::max(_cfg.block_out[0], _cfg.block_out[1]);
  const std::size_t big_cin =
      _use_hwconv ? (std::size_t)_cfg.block_out[0] : base_max;
  const std::size_t k_safe_band =                            // corruption cap
      _mma_max_m > 0 ? (std::size_t)_mma_max_m / 2 : (std::size_t)Hout * Wout;
  const std::size_t full_band =
      (std::size_t)Hout * Wout * 9 * big_cin;                // full im2col elems
  const std::size_t floor_band = (std::size_t)Wout * 9 * big_cin * 8;  // 8 rows
  const std::size_t act_reserve = (std::size_t)Hout * Wout * base_max * 7;
  std::size_t im2col_cap = full_band;                        // roomy default
  if (decode_headroom > 0) {
    const std::size_t avail_el =                             // bytes -> elems
        decode_headroom > act_reserve * 2
            ? (decode_headroom - act_reserve * 2) / 2 : 0;
    im2col_cap = std::min(full_band, std::max(floor_band, avail_el));
  }
  // Never size the scratch past one safe band for the widest conv -- a bigger
  // buffer just wastes UMA (the per-conv band is safe-capped at k_safe_band).
  im2col_cap = std::min(im2col_cap, k_safe_band * 9 * big_cin);
  if (const char* e = std::getenv("VPIPE_FLUX2_VAE_BAND_ROWS")) {
    const long r = std::atol(e);
    if (r > 0) { im2col_cap = (std::size_t)r * 9 * big_cin; }
  }
  SharedBuffer im2col_scratch = mc->make_shared_buffer(im2col_cap * 2);
  if (im2col_scratch.empty()) { return fail("im2col band scratch alloc failed"); }

  CommandStream stream = mc->make_command_stream();
  int H = h8, W = w8;
  const SharedBuffer* rgb_ptr = nullptr;
  // Fallback for when the buffer pool is disabled: split the decode across
  // command buffers at up-block boundaries, committing + freeing per level so
  // the running set (which sums to > the GPU wired limit at 1024^2) never
  // shares one command buffer. `carry` holds the one activation that crosses a
  // boundary. Opt in with VPIPE_FLUX2_VAE_SPLIT; a no-op when the pool is on
  // (the pool already bounds the peak within one command buffer).
  // The pool reuses buffers WITHIN a command buffer; the split commits + frees
  // ACROSS resolution levels. They compose: a single command buffer can only
  // rebind buffers (earlier dispatches still reference them until commit), so
  // the pool alone holds the union of every distinct size; committing per level
  // lets those be freed. Both default on for the minimum peak. `carry` crosses
  // each boundary. VPIPE_FLUX2_NO_VAE_SPLIT keeps it all in one command buffer.
  SharedBuffer carry;
  const bool vae_split = std::getenv("VPIPE_FLUX2_NO_VAE_SPLIT") == nullptr;
  const bool vprof = std::getenv("VPIPE_FLUX2_VAE_PROFILE") != nullptr;
  bool split_ok = true;
  {
    ComputeEncoder enc = stream.begin_compute();
    auto conv3x3 = [&](const SharedBuffer& in, int H, int W,
                       const Conv& c) -> SharedBuffer& {
      const std::size_t hw = (std::size_t)H * W;
      SharedBuffer& out = alloc(hw * c.cout);
      // NAX hardware conv when the shape tiles; else row-tiled im2col + GEMM
      // (streams the [hw, 9*cin] im2col in bands through im2col_scratch).
      if (conv3x3_hw_(enc, in, c, out, H, W, /*stride=*/1)) { return out; }
      // Small cout (the final conv to 3 / 2*latent) never satisfies the
      // hardware conv's cout % 64; a direct pass beats materializing im2col.
      if (conv3x3_small_cout_(enc, in, c, out, H, W, /*stride=*/1)) {
        return out;
      }
      tiled_conv3x3_(enc, in, out, H, W, c, /*stride=*/1, im2col_scratch,
                     im2col_cap);
      return out;
    };
    auto conv1x1 = [&](const SharedBuffer& in, std::size_t hw,
                       const Conv& c) -> SharedBuffer& {
      SharedBuffer& out = alloc(hw * c.cout);
      conv_gemm_bias_(enc, in, c.w, c.b, out, (int)hw, c.cout, c.cin);
      return out;
    };
    auto gnorm = [&](const SharedBuffer& in, std::size_t hw, const GNorm& n)
        -> SharedBuffer& {
      SharedBuffer& out = alloc(hw * n.c);
      group_norm_(enc, in, n.g, n.b, out, (int)hw, n.c, G, geps);
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
    auto upsample = [&](const SharedBuffer& in, int H, int W, int C)
        -> SharedBuffer& {
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
    // Residual input `x` is held live through to the resadd (read twice); the
    // internal temporaries are released as soon as their consumer is enqueued.
    auto resblock = [&](const ResBlock& rb, const SharedBuffer& x, int H, int W)
        -> SharedBuffer& {
      const std::size_t hw = (std::size_t)H * W;
      SharedBuffer& t = gnorm(x, hw, rb.n1);
      silu(t, hw * rb.cin);
      SharedBuffer& t1 = conv3x3(t, H, W, rb.c1);
      release(t);
      SharedBuffer& t2 = gnorm(t1, hw, rb.n2);
      release(t1);
      silu(t2, hw * rb.cout);
      SharedBuffer& t3 = conv3x3(t2, H, W, rb.c2);
      release(t2);
      if (rb.has_short) {
        SharedBuffer& h = conv1x1(x, hw, rb.shortcut);
        SharedBuffer& out = resadd(t3, h, hw * rb.cout);
        release(t3); release(h);
        return out;
      }
      SharedBuffer& out = resadd(t3, x, hw * rb.cout);
      release(t3);
      return out;
    };
    auto attention = [&](const Attn& a, const SharedBuffer& x, int H, int W)
        -> SharedBuffer& {
      const std::size_t hw = (std::size_t)H * W;
      const int C = a.dim;
      SharedBuffer& n = gnorm(x, hw, a.n);
      SharedBuffer& q = conv1x1(n, hw, a.q);
      SharedBuffer& k = conv1x1(n, hw, a.k);
      SharedBuffer& v = conv1x1(n, hw, a.v);
      release(n);                          // consumed by q, k, v
      SharedBuffer& att = alloc(hw * C);
      const float scale = 1.0f / std::sqrt((float)C);
      // The member is chosen ONCE, by measurement, in autotune_mid_attn_.
      encode_mid_attn_(enc, _attn_pick, q, k, v, att, hw, C, scale,
                       [&](std::size_t n) -> SharedBuffer& { return alloc(n); },
                       [&](const SharedBuffer& b) { release(b); });
      release(q); release(k); release(v);  // consumed by the sdpa
      SharedBuffer& p = conv1x1(att, hw, a.proj);
      release(att);
      SharedBuffer& out = resadd(p, x, hw * C);
      release(p);
      return out;
    };

    // Per-section GPU timing (VPIPE_FLUX2_VAE_PROFILE). Like the DiT profiler,
    // `mark` closes the command buffer and waits, so the sections serialize
    // (no cross-section overlap) -- the point is per-section GPU wall time,
    // not a faithful end-to-end number. Unlike `flush` it leaves the pool
    // alone: the buffers stay alive and reusable across the boundary, so
    // marking does not change what the decode allocates. No-op when unset.
    std::vector<std::pair<std::string, double>> vmarks;
    auto vnow = [] { return std::chrono::steady_clock::now(); };
    auto vlast = vnow();
    auto mark = [&](const std::string& name) {
      if (!vprof) { return; }
      enc.end();
      std::string ge;
      if (!stream.commit().wait_ok(&ge)) { split_ok = false; }
      vmarks.emplace_back(name,
                          std::chrono::duration<double, std::milli>(
                              vnow() - vlast).count());
      stream = mc->make_command_stream();
      enc = stream.begin_compute();
      vlast = vnow();
    };

    // Commit the current command buffer, free the level's intermediates, and
    // carry `*xp` (the level output) into the next command buffer. See `carry`.
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

    // post_quant_conv is optional (absent on the plain AutoencoderKL).
    const SharedBuffer* x = _post_quant.w.empty()
                                ? &latent
                                : &conv1x1(latent, hw8, _post_quant);
    // Feed-forward chain: after each op, release the activation it just consumed
    // so the pool can reuse that buffer (resblock/attention release their own
    // temporaries internally; each holds its residual input through the resadd).
    auto step = [&](const SharedBuffer& nx) { release(*x); x = &nx; };
    step(conv3x3(*x, H, W, _conv_in));
    mark(fmt("conv_in {}x{}", H, W)());
    step(resblock(_mid_res0, *x, H, W));
    mark("mid_res0");
    step(attention(_mid_attn, *x, H, W));
    mark("mid_attn");
    step(resblock(_mid_res1, *x, H, W));
    mark("mid_res1");
    for (const UpBlock& ub : _up_blocks) {
      const int lvl = (int)(&ub - _up_blocks.data());
      for (const ResBlock& rb : ub.resnets) { step(resblock(rb, *x, H, W)); }
      mark(fmt("up{} res x{} {}x{}c{}", lvl, (int)ub.resnets.size(), H, W,
               ub.resnets.empty() ? 0 : ub.resnets.back().cout)());
      if (ub.has_up) {
        step(upsample(*x, H, W, ub.up_dim));
        H *= 2; W *= 2;
        step(conv3x3(*x, H, W, ub.up));
        mark(fmt("up{} upsample+conv -> {}x{}c{}", lvl, H, W,
                 ub.up.cout)());
      }
      flush(x);            // pool off: bound the working set to ~one up-block
    }
    SharedBuffer& xn = gnorm(*x, (std::size_t)H * W, _norm_out);
    release(*x);
    silu(xn, (std::size_t)H * W * _cfg.block_out[0]);
    mark(fmt("norm_out+silu {}x{}", H, W)());
    SharedBuffer& rgb = conv3x3(xn, H, W, _conv_out);
    release(xn);
    const int n = H * W * 3;
    enc.set_function(_fn_clamp);
    enc.set_buffer(0, rgb); enc.set_buffer(1, rgb);
    enc.set_constant(2, n); enc.set_constant(3, -1.0f);
    enc.set_constant(4, 1.0f);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
    mark(fmt("norm_out+conv_out {}x{}", H, W)());
    rgb_ptr = &rgb;
    if (vprof && mc->session() != nullptr) {
      double tot = 0;
      for (const auto& m : vmarks) { tot += m.second; }
      std::string body;
      for (const auto& m : vmarks) {
        body += fmt("\n  {:<34} {} ms ({}%)", m.first, (long)m.second,
                    (long)(tot > 0 ? m.second * 100.0 / tot : 0))();
      }
      mc->session()->log_normal(fmt(
          "FLUX.2 VAE decode profile ({}x{} out, GPU {} ms total):{}",
          H, W, (long)tot, body));
    }
  }
  if (!alloc_ok) { return fail("a decode intermediate allocation failed"); }
  if (!split_ok) { return fail("GPU decode ran out of memory at a level "
                               "boundary (lower the resolution)"); }
  std::string gpu_err;
  if (!stream.commit().wait_ok(&gpu_err)) {
    return fail(gpu_err.empty() ? std::string("GPU decode failed") : gpu_err);
  }
  SharedBuffer out = mc->make_shared_buffer((std::size_t)3 * H * W * 2);
  {
    const auto t_rb = std::chrono::steady_clock::now();
    const auto* s = static_cast<const _Float16*>(rgb_ptr->contents());
    auto* d = static_cast<_Float16*>(out.contents());
    const std::size_t hw = (std::size_t)H * W;
    for (std::size_t p = 0; p < hw; ++p) {
      for (int c = 0; c < 3; ++c) { d[(std::size_t)c * hw + p] = s[p * 3 + c]; }
    }
    if (vprof && mc->session() != nullptr) {
      mc->session()->log_normal(fmt(
          "FLUX.2 VAE decode: interleaved->planar readback (CPU) {} ms",
          (long)std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - t_rb).count()));
    }
  }
  return out;
}

SharedBuffer
MetalFlux2Vae::encode(const SharedBuffer& img, int H0, int W0)
{
  if (!_has_encoder) { return {}; }
  const int L = _cfg.latent_channels;
  const int P = _cfg.patch;
  const int G = _cfg.norm_groups;
  const float geps = _cfg.norm_eps;
  MetalCompute* mc = _mc;
  if (img.byte_size() < (std::size_t)3 * H0 * W0 * 2) { return {}; }
  maybe_tune_conv_(H0, W0);

  // Buffer pool with reuse (see decode()): bounds the live set to the
  // concurrent working set. VPIPE_FLUX2_NO_VAE_POOL disables reuse.
  struct Slot { SharedBuffer buf; std::size_t cap; bool used; };
  std::deque<Slot> pool;
  bool alloc_ok = true;
  const bool use_pool = std::getenv("VPIPE_FLUX2_NO_VAE_POOL") == nullptr;
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
  // Row-tiled im2col band scratch (see decode()): stream each conv's [H*W, 9*cin]
  // (or the s2 downsample's [(H/2)(W/2), 9*cin]) in output-row bands so the
  // shared col scratch is bounded to ONE band -- a full-res encode conv is
  // otherwise ~2.4 GB. Cap the band by memory (headroom, no encode preflight so
  // query it here) AND correctness (k_safe_band rows, matmul2d M-corruption).
  const std::size_t base_max =
      (std::size_t)std::max(_cfg.block_out[0], _cfg.block_out[1]);
  const std::size_t big_cin =
      _use_hwconv ? (std::size_t)_cfg.block_out[0] : base_max;
  const std::size_t k_safe_band =
      _mma_max_m > 0 ? (std::size_t)_mma_max_m / 2 : (std::size_t)H0 * W0;
  const std::size_t full_band = (std::size_t)H0 * W0 * 9 * big_cin;
  const std::size_t floor_band = (std::size_t)W0 * 9 * big_cin * 8;
  const std::size_t act_reserve = (std::size_t)H0 * W0 * base_max * 7;
  std::size_t im2col_cap = full_band;
  {
    const MetalCompute::MemoryBudget mb = mc->memory_budget();
    if (mb.recommended != 0) {
      const std::size_t avail_el = mb.headroom > act_reserve * 2
          ? (mb.headroom - act_reserve * 2) / 2 : 0;
      im2col_cap = std::min(full_band, std::max(floor_band, avail_el));
    }
  }
  im2col_cap = std::min(im2col_cap, k_safe_band * 9 * big_cin);
  if (const char* e = std::getenv("VPIPE_FLUX2_VAE_BAND_ROWS")) {
    const long r = std::atol(e);
    if (r > 0) { im2col_cap = (std::size_t)r * 9 * big_cin; }
  }
  SharedBuffer im2col_scratch = mc->make_shared_buffer(im2col_cap * 2);
  if (im2col_scratch.empty()) { return {}; }

  // Channel-first [3,H,W] -> channel-last [H*W, 3].
  SharedBuffer& x0 = alloc((std::size_t)H0 * W0 * 3);
  {
    const auto* s = static_cast<const _Float16*>(img.contents());
    auto* d = static_cast<_Float16*>(x0.contents());
    const std::size_t hw = (std::size_t)H0 * W0;
    for (std::size_t p = 0; p < hw; ++p) {
      for (int c = 0; c < 3; ++c) { d[p * 3 + c] = s[(std::size_t)c * hw + p]; }
    }
  }

  const SharedBuffer* mean_ptr = nullptr;
  int H = H0, W = W0;
  CommandStream stream = mc->make_command_stream();
  // Per-down-stage command-buffer split; composes with the pool (see decode()).
  // Both default on for the minimum peak. `carry` crosses each boundary.
  SharedBuffer carry;
  const bool vae_split = std::getenv("VPIPE_FLUX2_NO_VAE_SPLIT") == nullptr;
  const bool vprof = std::getenv("VPIPE_FLUX2_VAE_PROFILE") != nullptr;
  bool split_ok = true;
  {
    ComputeEncoder enc = stream.begin_compute();
    auto conv3x3 = [&](const SharedBuffer& in, int H, int W,
                       const Conv& c) -> SharedBuffer& {
      SharedBuffer& out = alloc((std::size_t)H * W * c.cout);
      if (conv3x3_hw_(enc, in, c, out, H, W, /*stride=*/1)) { return out; }
      // Small cout (the final conv to 3 / 2*latent) never satisfies the
      // hardware conv's cout % 64; a direct pass beats materializing im2col.
      if (conv3x3_small_cout_(enc, in, c, out, H, W, /*stride=*/1)) {
        return out;
      }
      tiled_conv3x3_(enc, in, out, H, W, c, /*stride=*/1, im2col_scratch,
                     im2col_cap);
      return out;
    };
    auto conv3x3_s2 = [&](const SharedBuffer& in, int H, int W,
                          const Conv& c) -> SharedBuffer& {
      SharedBuffer& out = alloc((std::size_t)(H / 2) * (W / 2) * c.cout);
      if (conv3x3_hw_(enc, in, c, out, H, W, /*stride=*/2)) { return out; }
      tiled_conv3x3_(enc, in, out, H, W, c, /*stride=*/2, im2col_scratch,
                     im2col_cap);
      return out;
    };
    auto conv1x1 = [&](const SharedBuffer& in, std::size_t hw,
                       const Conv& c) -> SharedBuffer& {
      SharedBuffer& out = alloc(hw * c.cout);
      conv_gemm_bias_(enc, in, c.w, c.b, out, (int)hw, c.cout, c.cin);
      return out;
    };
    auto gnorm = [&](const SharedBuffer& in, std::size_t hw, const GNorm& n)
        -> SharedBuffer& {
      SharedBuffer& out = alloc(hw * n.c);
      group_norm_(enc, in, n.g, n.b, out, (int)hw, n.c, G, geps);
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
    auto resblock = [&](const ResBlock& rb, const SharedBuffer& x, int H, int W)
        -> SharedBuffer& {
      const std::size_t hw = (std::size_t)H * W;
      SharedBuffer& t = gnorm(x, hw, rb.n1);
      silu(t, hw * rb.cin);
      SharedBuffer& t1 = conv3x3(t, H, W, rb.c1);
      release(t);
      SharedBuffer& t2 = gnorm(t1, hw, rb.n2);
      release(t1);
      silu(t2, hw * rb.cout);
      SharedBuffer& t3 = conv3x3(t2, H, W, rb.c2);
      release(t2);
      if (rb.has_short) {
        SharedBuffer& h = conv1x1(x, hw, rb.shortcut);
        SharedBuffer& out = resadd(t3, h, hw * rb.cout);
        release(t3); release(h);
        return out;
      }
      SharedBuffer& out = resadd(t3, x, hw * rb.cout);
      release(t3);
      return out;
    };
    auto attention = [&](const Attn& a, const SharedBuffer& x, int H, int W)
        -> SharedBuffer& {
      const std::size_t hw = (std::size_t)H * W;
      const int C = a.dim;
      SharedBuffer& n = gnorm(x, hw, a.n);
      SharedBuffer& q = conv1x1(n, hw, a.q);
      SharedBuffer& k = conv1x1(n, hw, a.k);
      SharedBuffer& v = conv1x1(n, hw, a.v);
      release(n);                          // consumed by q, k, v
      SharedBuffer& att = alloc(hw * C);
      const float scale = 1.0f / std::sqrt((float)C);
      // The member is chosen ONCE, by measurement, in autotune_mid_attn_.
      encode_mid_attn_(enc, _attn_pick, q, k, v, att, hw, C, scale,
                       [&](std::size_t n) -> SharedBuffer& { return alloc(n); },
                       [&](const SharedBuffer& b) { release(b); });
      release(q); release(k); release(v);  // consumed by the sdpa
      SharedBuffer& p = conv1x1(att, hw, a.proj);
      release(att);
      SharedBuffer& out = resadd(p, x, hw * C);
      release(p);
      return out;
    };

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

    // Per-section GPU timing (VPIPE_FLUX2_VAE_PROFILE); see decode().
    std::vector<std::pair<std::string, double>> vmarks;
    auto vnow = [] { return std::chrono::steady_clock::now(); };
    auto vlast = vnow();
    auto mark = [&](const std::string& name) {
      if (!vprof) { return; }
      enc.end();
      std::string ge;
      if (!stream.commit().wait_ok(&ge)) { split_ok = false; }
      vmarks.emplace_back(name,
                          std::chrono::duration<double, std::milli>(
                              vnow() - vlast).count());
      stream = mc->make_command_stream();
      enc = stream.begin_compute();
      vlast = vnow();
    };

    const SharedBuffer* x = &conv3x3(x0, H, W, _enc_conv_in);
    mark(fmt("conv_in {}x{}", H, W)());
    release(x0);
    auto step = [&](const SharedBuffer& nx) { release(*x); x = &nx; };
    for (const DownStage& st : _enc_down) {
      const int lvl = (int)(&st - _enc_down.data());
      for (const ResBlock& rb : st.resnets) { step(resblock(rb, *x, H, W)); }
      mark(fmt("down{} res x{} {}x{}c{}", lvl, (int)st.resnets.size(), H, W,
               st.resnets.empty() ? 0 : st.resnets.back().cout)());
      if (st.has_down) {
        step(conv3x3_s2(*x, H, W, st.down));
        H /= 2; W /= 2;
        mark(fmt("down{} stride2 -> {}x{}", lvl, H, W)());
      }
      flush(x);            // pool off: bound the working set to ~one down-stage
    }
    step(resblock(_enc_mid_res0, *x, H, W));
    mark("mid_res0");
    step(attention(_enc_mid_attn, *x, H, W));
    mark("mid_attn");
    step(resblock(_enc_mid_res1, *x, H, W));
    mark("mid_res1");
    SharedBuffer& xn = gnorm(*x, (std::size_t)H * W, _enc_norm_out);
    release(*x);
    silu(xn, (std::size_t)H * W * _cfg.block_out[3]);
    SharedBuffer& moments = conv3x3(xn, H, W, _enc_conv_out);   // [hw, 2L]
    release(xn);
    mark(fmt("norm_out+conv_out {}x{}", H, W)());
    if (vprof && mc->session() != nullptr) {
      double tot = 0;
      for (const auto& m : vmarks) { tot += m.second; }
      std::string body;
      for (const auto& m : vmarks) {
        body += fmt("\n  {:<34} {} ms ({}%)", m.first, (long)m.second,
                    (long)(tot > 0 ? m.second * 100.0 / tot : 0))();
      }
      mc->session()->log_normal(fmt(
          "FLUX.2 VAE ENCODE profile ({}x{} in, GPU {} ms total):{}",
          H0, W0, (long)tot, body));
    }
    // quant_conv is optional (absent on the plain AutoencoderKL): then the
    // conv_out moments ARE the posterior parameters.
    if (_quant_conv.w.empty()) {
      mean_ptr = &moments;
    } else {
      mean_ptr = &conv1x1(moments, (std::size_t)H * W, _quant_conv);
      release(moments);
    }
  }
  if (!alloc_ok || !split_ok) { return {}; }
  std::string gpu_err;
  if (!stream.commit().wait_ok(&gpu_err)) { return {}; }

  // Take the mode (first L channels of the 2L moments), patchify [2,2], apply
  // bn -> DiT latent channel-first [dit_channels, H/P, W/P].
  const int h16 = H / P, w16 = W / P;
  const int Cdit = _cfg.dit_channels();
  SharedBuffer out = mc->make_shared_buffer((std::size_t)Cdit * h16 * w16 * 2);
  {
    const auto* s = static_cast<const _Float16*>(mean_ptr->contents());
    auto* d = static_cast<_Float16*>(out.contents());
    const int twoL = 2 * L;
    for (int c = 0; c < L; ++c) {
      for (int ph = 0; ph < P; ++ph) {
        for (int pw = 0; pw < P; ++pw) {
          const int cc = c * P * P + ph * P + pw;
          const float a = _bn_a[(std::size_t)cc], b = _bn_b[(std::size_t)cc];
          for (int i = 0; i < h16; ++i) {
            for (int j = 0; j < w16; ++j) {
              const int si = i * P + ph, sj = j * P + pw;
              const float m = (float)s[((std::size_t)si * W + sj) * twoL + c];
              d[((std::size_t)cc * h16 + i) * w16 + j] = (_Float16)(a * m + b);
            }
          }
        }
      }
    }
  }
  return out;
}

}  // namespace genai
}  // namespace vpipe

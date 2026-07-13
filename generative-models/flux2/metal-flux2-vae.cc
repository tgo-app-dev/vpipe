#include "generative-models/flux2/metal-flux2-vae.h"

#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;
using metal_compute::ComputeEncoder;
using metal_compute::CommandStream;

namespace {

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
MetalFlux2Vae::load_conv3x3_(const MetalLlamaWeights& wts,
                             const std::string& nm)
{
  Conv c;
  std::size_t n = 0;
  std::vector<float> w = read_f32_(wts, _mc, nm + ".weight", n);
  if (w.empty()) { return c; }
  const auto& sh = wts.info(nm + ".weight")->shape;
  const int Cout = (int)sh[0], Cin = (int)sh[1];
  c.cin = Cin; c.cout = Cout; c.k = 9 * Cin;
  std::vector<float> flat((std::size_t)Cout * 9 * Cin);
  for (int o = 0; o < Cout; ++o) {
    for (int ky = 0; ky < 3; ++ky) {
      for (int kx = 0; kx < 3; ++kx) {
        for (int i = 0; i < Cin; ++i) {
          const std::size_t si =
              ((((std::size_t)o * Cin + i) * 3 + ky) * 3) + kx;
          const std::size_t di = ((std::size_t)o * 9 + (ky * 3 + kx)) * Cin + i;
          flat[di] = w[si];
        }
      }
    }
  }
  c.w = f16_buf_(_mc, flat.data(), flat.size());
  // HWIO twin for the NAX hardware conv (out-channel fastest).
  if (_use_hwconv) {
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
    c.whwio = f16_buf_(_mc, hwio.data(), hwio.size());
  }
  std::size_t nb = 0;
  std::vector<float> b = read_f32_(wts, _mc, nm + ".bias", nb);
  if (!b.empty()) { c.b = f16_buf_(_mc, b.data(), nb); }
  return c;
}

// 1x1 conv or Linear [Cout,Cin(,1,1)] -> dense-gemm weight [Cout, Cin].
MetalFlux2Vae::Conv
MetalFlux2Vae::load_conv1x1_(const MetalLlamaWeights& wts,
                             const std::string& nm)
{
  Conv c;
  std::size_t n = 0;
  std::vector<float> w = read_f32_(wts, _mc, nm + ".weight", n);
  if (w.empty()) { return c; }
  const auto& sh = wts.info(nm + ".weight")->shape;
  c.cout = (int)sh[0]; c.cin = (int)sh[1]; c.k = c.cin;
  c.w = f16_buf_(_mc, w.data(), (std::size_t)c.cout * c.cin);
  std::size_t nb = 0;
  std::vector<float> b = read_f32_(wts, _mc, nm + ".bias", nb);
  if (!b.empty()) { c.b = f16_buf_(_mc, b.data(), nb); }
  return c;
}

MetalFlux2Vae::GNorm
MetalFlux2Vae::load_gnorm_(const MetalLlamaWeights& wts, const std::string& nm)
{
  GNorm gn;
  gn.g = load_vec_(wts, nm + ".weight");
  gn.b = load_vec_(wts, nm + ".bias");
  const auto* info = wts.info(nm + ".weight");
  if (info != nullptr && !info->shape.empty()) { gn.c = (int)info->shape[0]; }
  return gn;
}

SharedBuffer
MetalFlux2Vae::load_vec_(const MetalLlamaWeights& wts, const std::string& nm)
{
  std::size_t n = 0;
  std::vector<float> v = read_f32_(wts, _mc, nm, n);
  if (v.empty()) { return {}; }
  return f16_buf_(_mc, v.data(), n);
}

bool
MetalFlux2Vae::load_resblock_(const MetalLlamaWeights& wts,
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
MetalFlux2Vae::load_attn_(const MetalLlamaWeights& wts, const std::string& pre,
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
  if (mc == nullptr) { return nullptr; }
  auto wtsopt = MetalLlamaWeights::open_model(model_dir);
  if (!wtsopt.has_value()) { return nullptr; }
  const MetalLlamaWeights& wts = *wtsopt;

  auto m = std::unique_ptr<MetalFlux2Vae>(new MetalFlux2Vae());
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
        }
      }
    }
  }
  m->_cfg = cfg;

  m->_lib_gemm = mc->load_library("dense_gemm");
  m->_lib_elt  = mc->load_library("llm_elementwise");
  m->_lib_sdpa = mc->load_library("sdpa");
  m->_fn_gemm_bias   = m->_lib_gemm.function("dense_gemm_bias_f16");
  m->_fn_groupnorm   = m->_lib_elt.function("group_norm_f16");
  m->_fn_mul_sigmoid = m->_lib_elt.function("mul_sigmoid_f16");
  m->_fn_residual    = m->_lib_elt.function("residual_add_f16");
  m->_fn_clamp       = m->_lib_elt.function("clamp_f16");
  m->_fn_sdpa        = m->_lib_sdpa.function("sdpa_full_f16");
  m->_fn_sdpa_full_smm = m->_lib_sdpa.function("sdpa_full_mma_f16");
  m->_fn_im2col      = m->_lib_elt.function("im2col_hwc_3x3_f16");
  m->_fn_im2col_s2   = m->_lib_elt.function("im2col_hwc_3x3_s2_f16");
  m->_fn_upsample    = m->_lib_elt.function("upsample_nearest2x_hwc_f16");
  m->_fn_bias_add    = m->_lib_elt.function("bias_add_rows_f16");
  if (!m->_fn_gemm_bias.valid() || !m->_fn_groupnorm.valid() ||
      !m->_fn_mul_sigmoid.valid() || !m->_fn_residual.valid() ||
      !m->_fn_clamp.valid() || !m->_fn_sdpa.valid() || !m->_fn_im2col.valid() ||
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
    }
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
  m->_post_quant = m->load_conv1x1_(wts, "post_quant_conv");
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
    std::vector<float> mean = read_f32_(wts, mc, "bn.running_mean", n);
    std::vector<float> var  = read_f32_(wts, mc, "bn.running_var", n);
    m->_bn_a.assign((std::size_t)C, 1.0f);
    m->_bn_b.assign((std::size_t)C, 0.0f);
    if ((int)mean.size() == C && (int)var.size() == C) {
      const float eps = 1e-4f;                   // batch_norm_eps
      for (int c = 0; c < C; ++c) {
        const float a = 1.0f / std::sqrt(var[(std::size_t)c] + eps);
        m->_bn_a[(std::size_t)c] = a;
        m->_bn_b[(std::size_t)c] = -mean[(std::size_t)c] * a;
      }
    }
  }

  if (with_encoder) {
    if (!m->load_encoder_(wts)) { return nullptr; }
    m->_has_encoder = true;
  }
  if (!ok) { return nullptr; }
  return m;
}

bool
MetalFlux2Vae::load_encoder_(const MetalLlamaWeights& wts)
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
  _quant_conv = load_conv1x1_(wts, "quant_conv");           // 2L -> 2L
  ok = ok && !_enc_norm_out.g.empty() && !_enc_conv_out.w.empty() &&
       !_quant_conv.w.empty();
  (void)L;
  return ok;
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
                               const SharedBuffer& y, int M, int N, int K)
{
  // y[M,N] = x[M,K] @ w[N,K]^T (+ bias[N]). M5 matmul2d for tall M, else steel.
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
      enc.set_buffer(3, y, (std::size_t)r0 * N * 2);
      enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, mc);
      enc.set_constant(7, 0);
      enc.dispatch({(unsigned)(((N + BN - 1) / BN) * 256),
                    (unsigned)((mc + 127) / 128), 1}, {256, 1, 1});
    }
    if (!b.empty()) {
      // Fold bias[N] across ALL M rows at once (a plain kernel, no M limit).
      const std::size_t total = (std::size_t)M * N;
      enc.set_function(_fn_bias_add);
      enc.set_buffer(0, y); enc.set_buffer(1, b);
      enc.set_constant(2, N); enc.set_constant(3, (unsigned)total);
      // 2D grid {N, M}: gid = row*N + col (a 1D {M*N} grid overflows past ~2K).
      enc.dispatch({(unsigned)N, (unsigned)M, 1}, {256, 1, 1});
    }
    return;
  }
  enc.set_function(_fn_gemm_bias);
  enc.set_buffer(0, x); enc.set_buffer(1, w);
  enc.set_buffer(2, b.empty() ? w : b); enc.set_buffer(3, y);
  enc.set_constant(4, M); enc.set_constant(5, N); enc.set_constant(6, K);
  enc.set_constant(7, b.empty() ? 0 : 1);
  enc.dispatch({(unsigned)(((N + 15) / 16) * 16),
                (unsigned)(((M + 15) / 16) * 16), 1}, {16, 16, 1});
}

std::size_t
MetalFlux2Vae::decode_peak_bytes(int h16, int w16) const noexcept
{
  if (h16 <= 0 || w16 <= 0) { return 0; }
  const std::size_t Hout = (std::size_t)h16 * 16;
  const std::size_t Wout = (std::size_t)w16 * 16;
  const std::size_t base = (std::size_t)_cfg.block_out[0];
  // The per-up-block command-buffer split (default on; VPIPE_FLUX2_NO_VAE_SPLIT
  // opts out) commits + frees each up-level, so the resident peak is ONE
  // level's working set, not the summed up-path. That peak is the top-res
  // im2col scratch [Hout*Wout, 9*base]: conv_out (3 ch) can't use the hw conv
  // (cout % 64 != 0) so it falls back to im2col even in hwconv mode -- it is the
  // single largest buffer. Budget it + ~50% for the level's I/O activations. A
  // miss is caught cleanly by the per-level wait_ok() backstop; the previous
  // top*10 hwconv figure UNDER-modelled this im2col and risked a false pass.
  const std::size_t im2col = Hout * Wout * 9 * base * 2;
  if (std::getenv("VPIPE_FLUX2_NO_VAE_SPLIT") == nullptr) {
    return im2col + im2col / 2;             // split on: one up-level
  }
  // Split off: the whole up-path is one command buffer -- keep the summed,
  // conservative figure.
  const std::size_t top = Hout * Wout * base * 2;
  return _use_hwconv ? top * 10 : top * 9 * 2;
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
  const int Hout = h16 * 16, Wout = w16 * 16;

  {
    const MetalCompute::MemoryBudget mb = mc->memory_budget();
    const std::size_t need = decode_peak_bytes(h16, w16);
    if (mb.recommended != 0 && !mb.fits(need)) {
      return fail(fmt(
          "insufficient GPU memory for a {}x{} decode: need ~{} MB, {} MB free "
          "of {} MB working set (lower the resolution or free other resident "
          "models)", Wout, Hout, need >> 20, mb.headroom >> 20,
          mb.recommended >> 20)());
    }
    // True-physical-pressure backstop: reclaimable RAM (counting mmap'd/clean
    // weight pages the OS can evict) must also cover the decode, else the GPU
    // command buffer would OOM mid-flight instead of a clean rejection here.
    if (!mb.fits_physical(need)) {
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
  // im2col scratch: with the NAX hw conv active, only the small fallback convs
  // (3-ch conv_in/out, non-tiling shapes) use im2col and they alloc their own
  // col buffers -- so skip the multi-GB up-front scratch (a 1024^2 conv reserves
  // ~2.4 GB it never touches on the hw path). Without hw conv, keep the shared
  // scratch sized for the largest conv.
  const std::size_t im2col_cap =
      _use_hwconv ? 0 : (std::size_t)Hout * Wout * 9 * _cfg.block_out[0];
  SharedBuffer im2col_scratch;
  if (im2col_cap != 0) {
    im2col_scratch = mc->make_shared_buffer(im2col_cap * 2);
    if (im2col_scratch.empty()) { return fail("im2col scratch alloc failed"); }
  }

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
  bool split_ok = true;
  {
    ComputeEncoder enc = stream.begin_compute();
    auto conv3x3 = [&](const SharedBuffer& in, int H, int W,
                       const Conv& c) -> SharedBuffer& {
      const std::size_t hw = (std::size_t)H * W;
      SharedBuffer& out = alloc(hw * c.cout);
      // NAX hardware conv when the shape tiles; else im2col + GEMM.
      if (conv3x3_hw_(enc, in, c, out, H, W, /*stride=*/1)) { return out; }
      const std::size_t cols = hw * 9 * c.cin;
      const SharedBuffer& col =
          cols <= im2col_cap ? im2col_scratch : alloc(cols);
      enc.set_function(_fn_im2col);
      enc.set_buffer(0, in); enc.set_buffer(1, col);
      enc.set_constant(2, H); enc.set_constant(3, W);
      enc.set_constant(4, c.cin);
      // 2D grid {9*cin, hw} keeps each grid dimension small (a 1D {cols} grid is
      // > 2^31 for a 1024px conv); the im2col kernel reconstructs the flat index
      // as tpig.y*(9*cin)+tpig.x.
      enc.dispatch({(unsigned)(9 * c.cin), (unsigned)hw, 1}, {64, 1, 1});
      conv_gemm_bias_(enc, col, c.w, c.b, out, (int)hw, c.cout, c.k);
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
      enc.set_function(_fn_groupnorm);
      enc.set_buffer(0, in); enc.set_buffer(1, n.g); enc.set_buffer(2, n.b);
      enc.set_buffer(3, out);
      enc.set_constant(4, (int)hw); enc.set_constant(5, n.c);
      enc.set_constant(6, G); enc.set_constant(7, geps);
      enc.dispatch({256, (unsigned)G, 1}, {256, 1, 1});
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
      enc.set_buffer(0, q); enc.set_buffer(1, k); enc.set_buffer(2, v);
      enc.set_buffer(3, att);
      enc.set_constant(4, scale); enc.set_constant(5, (int)hw);
      enc.set_constant(6, C); enc.set_constant(7, 1); enc.set_constant(8, 1);
      enc.set_constant(9, (int)hw); enc.set_constant(10, (int)hw);
      // matmul2d flash on M5 (hardware matrix cores); simdgroup_matrix flash
      // (sdpa_full_mma_f16) elsewhere -- it needs no matrix cores, so it is the
      // fast path on M4/older where emulated matmul2d is slow. Scalar last.
      const bool smm = _fn_sdpa_full_smm.valid() && (C % 64 == 0) && (C <= 512);
      const bool mma2 = _fn_sdpa_full_mma.valid() && _use_attn_mma2;
      if (mma2 && !(smm && std::getenv("VPIPE_FLUX2_VAE_ATTN_SMM"))) {
        enc.set_function(_fn_sdpa_full_mma);           // matmul2d (M5)
        enc.dispatch({4 * 32, 1, (unsigned)((hw + 7) / 8)}, {4 * 32, 1, 1});
      } else if (smm) {
        enc.set_function(_fn_sdpa_full_smm);           // simdgroup_matrix
        const unsigned nt = 4u * (unsigned)(C / 64) * 32u;   // WM*WD*32
        enc.dispatch({nt, 1, (unsigned)((hw + 31) / 32)}, {nt, 1, 1});
      } else {
        enc.set_function(_fn_sdpa);                    // scalar O(N^2)
        enc.dispatch({32, 1, (unsigned)hw}, {32, 1, 1});
      }
      release(q); release(k); release(v);  // consumed by the sdpa
      SharedBuffer& p = conv1x1(att, hw, a.proj);
      release(att);
      SharedBuffer& out = resadd(p, x, hw * C);
      release(p);
      return out;
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

    const SharedBuffer* x = &conv1x1(latent, hw8, _post_quant);
    // Feed-forward chain: after each op, release the activation it just consumed
    // so the pool can reuse that buffer (resblock/attention release their own
    // temporaries internally; each holds its residual input through the resadd).
    auto step = [&](const SharedBuffer& nx) { release(*x); x = &nx; };
    step(conv3x3(*x, H, W, _conv_in));
    step(resblock(_mid_res0, *x, H, W));
    step(attention(_mid_attn, *x, H, W));
    step(resblock(_mid_res1, *x, H, W));
    for (const UpBlock& ub : _up_blocks) {
      for (const ResBlock& rb : ub.resnets) { step(resblock(rb, *x, H, W)); }
      if (ub.has_up) {
        step(upsample(*x, H, W, ub.up_dim));
        H *= 2; W *= 2;
        step(conv3x3(*x, H, W, ub.up));
      }
      flush(x);            // pool off: bound the working set to ~one up-block
    }
    SharedBuffer& xn = gnorm(*x, (std::size_t)H * W, _norm_out);
    release(*x);
    silu(xn, (std::size_t)H * W * _cfg.block_out[0]);
    SharedBuffer& rgb = conv3x3(xn, H, W, _conv_out);
    release(xn);
    const int n = H * W * 3;
    enc.set_function(_fn_clamp);
    enc.set_buffer(0, rgb); enc.set_buffer(1, rgb);
    enc.set_constant(2, n); enc.set_constant(3, -1.0f);
    enc.set_constant(4, 1.0f);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
    rgb_ptr = &rgb;
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
    const auto* s = static_cast<const _Float16*>(rgb_ptr->contents());
    auto* d = static_cast<_Float16*>(out.contents());
    const std::size_t hw = (std::size_t)H * W;
    for (std::size_t p = 0; p < hw; ++p) {
      for (int c = 0; c < 3; ++c) { d[(std::size_t)c * hw + p] = s[p * 3 + c]; }
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
  // See decode(): the hw conv path never touches the big shared scratch.
  const std::size_t im2col_cap =
      _use_hwconv ? 0 : (std::size_t)H0 * W0 * 9 * _cfg.block_out[0];
  SharedBuffer im2col_scratch;
  if (im2col_cap != 0) {
    im2col_scratch = mc->make_shared_buffer(im2col_cap * 2);
    if (im2col_scratch.empty()) { return {}; }
  }

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
  bool split_ok = true;
  {
    ComputeEncoder enc = stream.begin_compute();
    auto conv3x3 = [&](const SharedBuffer& in, int H, int W,
                       const Conv& c) -> SharedBuffer& {
      const std::size_t hw = (std::size_t)H * W;
      SharedBuffer& out = alloc(hw * c.cout);
      // NAX hardware conv when the shape tiles; else im2col + GEMM.
      if (conv3x3_hw_(enc, in, c, out, H, W, /*stride=*/1)) { return out; }
      const std::size_t cols = hw * 9 * c.cin;
      const SharedBuffer& col =
          cols <= im2col_cap ? im2col_scratch : alloc(cols);
      enc.set_function(_fn_im2col);
      enc.set_buffer(0, in); enc.set_buffer(1, col);
      enc.set_constant(2, H); enc.set_constant(3, W);
      enc.set_constant(4, c.cin);
      // 2D grid {9*cin, hw} keeps each grid dimension small (a 1D {cols} grid is
      // > 2^31 for a 1024px conv); the im2col kernel reconstructs the flat index
      // as tpig.y*(9*cin)+tpig.x.
      enc.dispatch({(unsigned)(9 * c.cin), (unsigned)hw, 1}, {64, 1, 1});
      conv_gemm_bias_(enc, col, c.w, c.b, out, (int)hw, c.cout, c.k);
      return out;
    };
    auto conv3x3_s2 = [&](const SharedBuffer& in, int H, int W,
                          const Conv& c) -> SharedBuffer& {
      const int OH = H / 2, OW = W / 2;
      const std::size_t ohw = (std::size_t)OH * OW;
      SharedBuffer& out = alloc(ohw * c.cout);
      // NAX hardware conv when the shape tiles; else im2col + GEMM.
      if (conv3x3_hw_(enc, in, c, out, H, W, /*stride=*/2)) { return out; }
      const std::size_t cols = ohw * 9 * c.cin;
      const SharedBuffer& col =
          cols <= im2col_cap ? im2col_scratch : alloc(cols);
      enc.set_function(_fn_im2col_s2);
      enc.set_buffer(0, in); enc.set_buffer(1, col);
      enc.set_constant(2, H); enc.set_constant(3, W);
      enc.set_constant(4, c.cin);
      // 2D grid {9*cin, (H/2)*(W/2)} keeps each dimension small (a 1D {cols}
      // grid is > 2^31 for a >=~2K conv); the kernel reconstructs the flat
      // index as tpig.y*(9*cin)+tpig.x.
      enc.dispatch({(unsigned)(9 * c.cin), (unsigned)ohw, 1}, {64, 1, 1});
      conv_gemm_bias_(enc, col, c.w, c.b, out, (int)ohw, c.cout, c.k);
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
      enc.set_function(_fn_groupnorm);
      enc.set_buffer(0, in); enc.set_buffer(1, n.g); enc.set_buffer(2, n.b);
      enc.set_buffer(3, out);
      enc.set_constant(4, (int)hw); enc.set_constant(5, n.c);
      enc.set_constant(6, G); enc.set_constant(7, geps);
      enc.dispatch({256, (unsigned)G, 1}, {256, 1, 1});
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
      enc.set_buffer(0, q); enc.set_buffer(1, k); enc.set_buffer(2, v);
      enc.set_buffer(3, att);
      enc.set_constant(4, scale); enc.set_constant(5, (int)hw);
      enc.set_constant(6, C); enc.set_constant(7, 1); enc.set_constant(8, 1);
      enc.set_constant(9, (int)hw); enc.set_constant(10, (int)hw);
      // matmul2d flash on M5 (hardware matrix cores); simdgroup_matrix flash
      // (sdpa_full_mma_f16) elsewhere -- it needs no matrix cores, so it is the
      // fast path on M4/older where emulated matmul2d is slow. Scalar last.
      const bool smm = _fn_sdpa_full_smm.valid() && (C % 64 == 0) && (C <= 512);
      const bool mma2 = _fn_sdpa_full_mma.valid() && _use_attn_mma2;
      if (mma2 && !(smm && std::getenv("VPIPE_FLUX2_VAE_ATTN_SMM"))) {
        enc.set_function(_fn_sdpa_full_mma);           // matmul2d (M5)
        enc.dispatch({4 * 32, 1, (unsigned)((hw + 7) / 8)}, {4 * 32, 1, 1});
      } else if (smm) {
        enc.set_function(_fn_sdpa_full_smm);           // simdgroup_matrix
        const unsigned nt = 4u * (unsigned)(C / 64) * 32u;   // WM*WD*32
        enc.dispatch({nt, 1, (unsigned)((hw + 31) / 32)}, {nt, 1, 1});
      } else {
        enc.set_function(_fn_sdpa);                    // scalar O(N^2)
        enc.dispatch({32, 1, (unsigned)hw}, {32, 1, 1});
      }
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

    const SharedBuffer* x = &conv3x3(x0, H, W, _enc_conv_in);
    release(x0);
    auto step = [&](const SharedBuffer& nx) { release(*x); x = &nx; };
    for (const DownStage& st : _enc_down) {
      for (const ResBlock& rb : st.resnets) { step(resblock(rb, *x, H, W)); }
      if (st.has_down) {
        step(conv3x3_s2(*x, H, W, st.down));
        H /= 2; W /= 2;
      }
      flush(x);            // pool off: bound the working set to ~one down-stage
    }
    step(resblock(_enc_mid_res0, *x, H, W));
    step(attention(_enc_mid_attn, *x, H, W));
    step(resblock(_enc_mid_res1, *x, H, W));
    SharedBuffer& xn = gnorm(*x, (std::size_t)H * W, _enc_norm_out);
    release(*x);
    silu(xn, (std::size_t)H * W * _cfg.block_out[3]);
    SharedBuffer& moments = conv3x3(xn, H, W, _enc_conv_out);   // [hw, 2L]
    release(xn);
    mean_ptr = &conv1x1(moments, (std::size_t)H * W, _quant_conv);
    release(moments);
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

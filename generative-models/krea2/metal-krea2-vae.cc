#include "generative-models/krea2/metal-krea2-vae.h"

#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;
using metal_compute::ComputeEncoder;
using metal_compute::CommandStream;

namespace {

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
MetalKrea2Vae::load_conv3x3_(const MetalLlamaWeights& wts, const std::string& nm,
                             bool from3d)
{
  Conv c;
  std::size_t n = 0;
  std::vector<float> w = read_f32_(wts, _mc, nm + ".weight", n);
  if (w.empty()) { return c; }
  const auto* info = wts.info(nm + ".weight");
  const auto& sh = info->shape;
  const int Cout = (int)sh[0];
  const int Cin = (int)sh[1];
  const int kt = from3d ? (int)sh[2] : 1;   // 3 for conv3d, absent for conv2d
  const int kty = from3d ? (kt - 1) : 0;    // kt=2 slice index (last)
  c.cin = Cin; c.cout = Cout; c.k = 9 * Cin;
  std::vector<float> flat((std::size_t)Cout * 9 * Cin);
  for (int o = 0; o < Cout; ++o) {
    for (int ky = 0; ky < 3; ++ky) {
      for (int kx = 0; kx < 3; ++kx) {
        for (int i = 0; i < Cin; ++i) {
          std::size_t si;
          if (from3d) {
            // [Cout,Cin,3,3,3]: (o,i,kt,ky,kx)
            si = (((((std::size_t)o * Cin + i) * 3 + kty) * 3 + ky) * 3) + kx;
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
  c.w = f16_buf_(_mc, flat.data(), flat.size());
  // HWIO twin for the NAX hardware conv (out-channel fastest). `flat` is
  // [Cout, 9*Cin] with (ky,kx,ci) columns -- permute from it directly so
  // the conv3d slice handling above is inherited.
  if (_use_hwconv) {
    std::vector<float> hwio((std::size_t)9 * Cin * Cout);
    for (int o = 0; o < Cout; ++o) {
      for (int t = 0; t < 9; ++t) {
        for (int i = 0; i < Cin; ++i) {
          hwio[((std::size_t)t * Cin + i) * Cout + o] =
              flat[((std::size_t)o * 9 + t) * Cin + i];
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

// Load a 1x1 conv ([Cout,Cin,1,1,1] or [Cout,Cin,1,1]) as dense-gemm weight
// [Cout, Cin] (the trailing singleton dims flatten away).
MetalKrea2Vae::Conv
MetalKrea2Vae::load_conv1x1_(const MetalLlamaWeights& wts, const std::string& nm)
{
  Conv c;
  std::size_t n = 0;
  std::vector<float> w = read_f32_(wts, _mc, nm + ".weight", n);
  if (w.empty()) { return c; }
  const auto& sh = wts.info(nm + ".weight")->shape;
  c.cout = (int)sh[0]; c.cin = (int)sh[1]; c.k = c.cin;
  c.w = f16_buf_(_mc, w.data(), n);
  std::size_t nb = 0;
  std::vector<float> b = read_f32_(wts, _mc, nm + ".bias", nb);
  if (!b.empty()) { c.b = f16_buf_(_mc, b.data(), nb); }
  return c;
}

SharedBuffer
MetalKrea2Vae::load_vec_(const MetalLlamaWeights& wts, const std::string& nm)
{
  std::size_t n = 0;
  std::vector<float> v = read_f32_(wts, _mc, nm, n);
  if (v.empty()) { return {}; }
  return f16_buf_(_mc, v.data(), n);
}

bool
MetalKrea2Vae::load_resblock_(const MetalLlamaWeights& wts,
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
  if (mc == nullptr) { return nullptr; }
  auto wtsopt = MetalLlamaWeights::open_model(model_dir);
  if (!wtsopt.has_value()) { return nullptr; }
  const MetalLlamaWeights& wts = *wtsopt;

  auto m = std::unique_ptr<MetalKrea2Vae>(new MetalKrea2Vae());
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
  m->_fn_im2col      = m->_lib_elt.function("im2col_hwc_3x3_f16");
  m->_fn_im2col_s2   = m->_lib_elt.function("im2col_hwc_3x3_s2_f16");
  m->_fn_upsample    = m->_lib_elt.function("upsample_nearest2x_hwc_f16");
  if (!m->_fn_gemm_bias.valid() || !m->_fn_rms.valid() ||
      !m->_fn_mul_sigmoid.valid() || !m->_fn_residual.valid() ||
      !m->_fn_clamp.valid() || !m->_fn_sdpa.valid() ||
      !m->_fn_im2col.valid() || !m->_fn_im2col_s2.valid() ||
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
    // Fused 3x3 conv2d (matmul2d over threadgroup-staged im2col): correct (bit-
    // identical to im2col + matmul2d -- krea2_vae.decode_conv2d_vs_im2col) but
    // ~1.7x SLOWER on M5, because its high UMA bandwidth makes the im2col DRAM
    // round-trip cheap while the fused per-tile halo re-gather + barriers cost
    // more than they save. So it is OPT-IN (VPIPE_KREA2_CONV2D=1) -- parked for a
    // halo-tile rewrite / a lower-bandwidth chip -- and the default keeps the
    // faster im2col + matmul2d path. Needs bias_add (_use_mma2) for the bias fold.
    if (m->_use_mma2 && std::getenv("VPIPE_KREA2_CONV2D") != nullptr) {
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
    }
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
    std::size_t n = 0;
    std::vector<float> qkv = read_f32_(
        wts, mc, "decoder.mid_block.attentions.0.to_qkv.weight", n);
    std::size_t nb = 0;
    std::vector<float> qkvb = read_f32_(
        wts, mc, "decoder.mid_block.attentions.0.to_qkv.bias", nb);
    const int C = dims0;
    if (qkv.size() == (std::size_t)3 * C * C && qkvb.size() == (std::size_t)3 * C) {
      auto slice = [&](int off) {
        Conv c; c.cin = C; c.cout = C; c.k = C;
        c.w = f16_buf_(mc, qkv.data() + (std::size_t)off * C * C, (std::size_t)C * C);
        c.b = f16_buf_(mc, qkvb.data() + (std::size_t)off * C, (std::size_t)C);
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

  if (with_encoder) {
    if (!m->load_encoder_(wts)) { return nullptr; }
    m->_has_encoder = true;
  }

  if (!ok) { return nullptr; }
  return m;
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
  {
    const MetalCompute::MemoryBudget mb = mc->memory_budget();
    const std::size_t need = decode_peak_bytes(h8, w8);
    if (mb.recommended != 0 && !mb.fits(need)) {
      return fail(fmt(
          "insufficient GPU memory for a {}x{} decode: need ~{} MB, {} MB "
          "free of {} MB working set (lower the resolution, tile, or free "
          "other resident models)", Wout, Hout, need >> 20,
          mb.headroom >> 20, mb.recommended >> 20)());
    }
    // True-physical-pressure backstop (reclaimable RAM incl. evictable mmap'd
    // weights) so a shortfall rejects cleanly instead of a mid-decode GPU OOM.
    if (!mb.fits_physical(need)) {
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
  // full res, cin = base*dim_mult[1]); a guard falls back to a held buffer if
  // some conv ever needs more, so correctness never depends on the estimate.
  // With the NAX hw conv active only the small fallback convs use im2col (they
  // alloc their own col buffers), so skip the multi-GB up-front scratch. Else
  // keep the shared scratch sized for the largest conv.
  const std::size_t im2col_cap =
      _use_hwconv ? 0 : (std::size_t)Hout * Wout * 9 * (base * _cfg.dim_mult[1]);
  SharedBuffer im2col_scratch;
  if (im2col_cap != 0) {
    im2col_scratch = mc->make_shared_buffer(im2col_cap * 2);
    if (im2col_scratch.empty()) {
      return fail("im2col scratch allocation failed (out of GPU memory)");
    }
  }

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
      const std::size_t hw = (std::size_t)H * W;
      SharedBuffer& out = alloc(hw * c.cout);
      // NAX hardware conv first; then the (opt-in) fused tgmem conv; else
      // im2col + gemm_bias into the same output buffer.
      if (conv3x3_hw_(enc, in, c, out, H, W, /*stride=*/1)) { return out; }
      if (conv2d_mma_(enc, in, c.w, c.b, out, H, W, c.cin, c.cout, H, W, 1)) {
        return out;
      }
      const std::size_t cols = hw * 9 * c.cin;
      const SharedBuffer& col =
          cols <= im2col_cap ? im2col_scratch : alloc(cols);
      enc.set_function(_fn_im2col);
      enc.set_buffer(0, in); enc.set_buffer(1, col);
      enc.set_constant(2, H); enc.set_constant(3, W); enc.set_constant(4, c.cin);
      // 2D grid {9*cin, hw} keeps each dimension small (a 1D {cols} grid is
      // > 2^31 for a >=1K conv); the kernel reconstructs tpig.y*(9*cin)+tpig.x.
      enc.dispatch({(unsigned)(9 * c.cin), (unsigned)hw, 1}, {64, 1, 1});
      gemm_bias(col, c.w, c.b, out, (int)hw, c.cout, c.k);
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
      if (mma2 && !(smm && std::getenv("VPIPE_KREA2_VAE_ATTN_SMM"))) {
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
MetalKrea2Vae::load_encoder_(const MetalLlamaWeights& wts)
{
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
    std::vector<float> qkv = read_f32_(
        wts, mc, "encoder.mid_block.attentions.0.to_qkv.weight", n);
    std::vector<float> qkvb = read_f32_(
        wts, mc, "encoder.mid_block.attentions.0.to_qkv.bias", nb);
    const int C = dtop;
    if (qkv.size() == (std::size_t)3 * C * C && qkvb.size() == (std::size_t)3 * C) {
      auto slice = [&](int off) {
        Conv c; c.cin = C; c.cout = C; c.k = C;
        c.w = f16_buf_(mc, qkv.data() + (std::size_t)off * C * C, (std::size_t)C * C);
        c.b = f16_buf_(mc, qkvb.data() + (std::size_t)off * C, (std::size_t)C);
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
  // Peak im2col = the full-res base-channel s1 convs (hw*9*base); every deeper
  // level shrinks (hw/4 per stage, channels x2). A held fallback covers any
  // surprise.
  // See decode(): the hw conv path never touches the big shared scratch.
  const std::size_t im2col_cap =
      _use_hwconv ? 0 : hw * 9 * (std::size_t)base;
  SharedBuffer im2col_scratch;
  if (im2col_cap != 0) {
    im2col_scratch = mc->make_shared_buffer(im2col_cap * 2);
    if (im2col_scratch.empty()) { return {}; }
  }

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
      const std::size_t ohw = (std::size_t)OH * OW;
      SharedBuffer& out = alloc(ohw * c.cout);
      if (conv3x3_hw_(enc, in, c, out, H, W, stride2 ? 2 : 1)) {
        return out;
      }
      if (conv2d_mma_(enc, in, c.w, c.b, out, H, W, c.cin, c.cout, OH, OW,
                      stride2 ? 2 : 1)) {
        return out;
      }
      const std::size_t cols = ohw * 9 * c.cin;
      const SharedBuffer& col =
          cols <= im2col_cap ? im2col_scratch : alloc(cols);
      enc.set_function(stride2 ? _fn_im2col_s2 : _fn_im2col);
      enc.set_buffer(0, in); enc.set_buffer(1, col);
      enc.set_constant(2, H); enc.set_constant(3, W); enc.set_constant(4, c.cin);
      // 2D grid {9*cin, OH*OW} keeps each dimension small (a 1D {cols} grid is
      // > 2^31 for a >=~2K conv); the kernel reconstructs tpig.y*(9*cin)+tpig.x
      // (identical for the s1 and s2 im2col kernels).
      enc.dispatch({(unsigned)(9 * c.cin), (unsigned)ohw, 1}, {64, 1, 1});
      gemm_bias(col, c.w, c.b, out, (int)ohw, c.cout, c.k);
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
      enc.set_buffer(0, q); enc.set_buffer(1, k); enc.set_buffer(2, v);
      enc.set_buffer(3, att);
      enc.set_constant(4, scale); enc.set_constant(5, (int)hwv);
      enc.set_constant(6, C); enc.set_constant(7, 1); enc.set_constant(8, 1);
      enc.set_constant(9, (int)hwv); enc.set_constant(10, (int)hwv);
      if (_fn_sdpa_full_mma.valid()) {         // matrix-core FULL flash attn
        enc.set_function(_fn_sdpa_full_mma);
        enc.dispatch({4 * 32, 1, (unsigned)((hwv + 7) / 8)}, {4 * 32, 1, 1});
      } else {                                 // scalar O(N^2) fallback
        enc.set_function(_fn_sdpa);
        enc.dispatch({32, 1, (unsigned)hwv}, {32, 1, 1});
      }
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

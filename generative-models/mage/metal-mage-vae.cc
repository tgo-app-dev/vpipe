#include "generative-models/mage/metal-mage-vae.h"

#include "generative-models/llama3/metal-llama-weights.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;
using metal_compute::ComputeEncoder;
using metal_compute::CommandStream;

namespace {

constexpr const char* kEncPre = "student.dconv_encoder.";

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

float silu_(float x) { return x / (1.0f + std::exp(-x)); }

}  // namespace

MetalMageVae::~MetalMageVae() = default;

// ---------------------------------------------------------------------------
// Load helpers
// ---------------------------------------------------------------------------

// Linear / 1x1 conv -> dense-GEMM weight [Cout, Cin] (trailing 1x1 dims of a
// Conv2d flatten away).
MetalMageVae::Conv
MetalMageVae::load_linear_(const MetalLlamaWeights& wts, const std::string& nm)
{
  Conv c;
  std::size_t n = 0;
  std::vector<float> w = read_f32_(wts, _mc, nm + ".weight", n);
  if (w.empty()) { return c; }
  const auto& sh = wts.info(nm + ".weight")->shape;
  c.cout = (int)sh[0];
  c.cin = (int)(n / (std::size_t)c.cout);
  c.k = c.cin;
  c.w = f16_buf_(_mc, w.data(), n);
  std::size_t nb = 0;
  std::vector<float> b = read_f32_(wts, _mc, nm + ".bias", nb);
  if (!b.empty()) { c.b = f16_buf_(_mc, b.data(), nb); }
  return c;
}

// 3x3 conv -> [Cout, 9*Cin] flattened (ky,kx,cin), pairing with
// im2col_hwc_3x3 (same convention as the Krea-2 / FLUX.2 VAEs).
MetalMageVae::Conv
MetalMageVae::load_conv3x3_(const MetalLlamaWeights& wts, const std::string& nm)
{
  Conv c;
  std::size_t n = 0;
  std::vector<float> w = read_f32_(wts, _mc, nm + ".weight", n);
  if (w.empty()) { return c; }
  const auto& sh = wts.info(nm + ".weight")->shape;   // [Cout, Cin, 3, 3]
  const int Cout = (int)sh[0], Cin = (int)sh[1];
  c.cin = Cin; c.cout = Cout; c.k = 9 * Cin;
  std::vector<float> flat((std::size_t)Cout * 9 * Cin);
  for (int o = 0; o < Cout; ++o) {
    for (int ky = 0; ky < 3; ++ky) {
      for (int kx = 0; kx < 3; ++kx) {
        for (int i = 0; i < Cin; ++i) {
          flat[((std::size_t)o * 9 + (ky * 3 + kx)) * Cin + i] =
              w[((((std::size_t)o * Cin + i) * 3 + ky) * 3) + kx];
        }
      }
    }
  }
  c.w = f16_buf_(_mc, flat.data(), flat.size());
  std::size_t nb = 0;
  std::vector<float> b = read_f32_(wts, _mc, nm + ".bias", nb);
  if (!b.empty()) { c.b = f16_buf_(_mc, b.data(), nb); }
  return c;
}

// Patch conv (kernel == stride == `patch`) -> [Cout, Cin*p*p]. PyTorch's
// [Cout, Cin, p, p] already flattens to the (cin, ky, kx) column order the
// host patchify emits, so this is a straight copy.
MetalMageVae::Conv
MetalMageVae::load_patch_conv_(const MetalLlamaWeights& wts,
                               const std::string& nm)
{
  Conv c;
  std::size_t n = 0;
  std::vector<float> w = read_f32_(wts, _mc, nm + ".weight", n);
  if (w.empty()) { return c; }
  const auto& sh = wts.info(nm + ".weight")->shape;
  c.cout = (int)sh[0];
  c.cin = (int)sh[1];
  c.k = (int)(n / (std::size_t)c.cout);
  c.w = f16_buf_(_mc, w.data(), n);
  std::size_t nb = 0;
  std::vector<float> b = read_f32_(wts, _mc, nm + ".bias", nb);
  if (!b.empty()) { c.b = f16_buf_(_mc, b.data(), nb); }
  return c;
}

// Depthwise 3x3 ([C, 1, 3, 3]) -> [C, 9] (the natural flatten).
MetalMageVae::DwConv
MetalMageVae::load_dwconv_(const MetalLlamaWeights& wts, const std::string& nm)
{
  DwConv d;
  std::size_t n = 0;
  std::vector<float> w = read_f32_(wts, _mc, nm + ".weight", n);
  if (w.empty()) { return d; }
  d.c = (int)wts.info(nm + ".weight")->shape[0];
  d.w = f16_buf_(_mc, w.data(), n);
  std::size_t nb = 0;
  std::vector<float> b = read_f32_(wts, _mc, nm + ".bias", nb);
  if (!b.empty()) { d.b = f16_buf_(_mc, b.data(), nb); }
  return d;
}

SharedBuffer
MetalMageVae::load_vec_(const MetalLlamaWeights& wts, const std::string& nm)
{
  std::size_t n = 0;
  std::vector<float> v = read_f32_(wts, _mc, nm, n);
  if (v.empty()) { return {}; }
  return f16_buf_(_mc, v.data(), n);
}

// TimestepEmbedder(0): timestep_embedding(0, 256) is [cos(0)]*128 ++
// [sin(0)]*128 = [1]*128 ++ [0]*128 regardless of the frequency table, so
// the whole embedder collapses to mlp.2(SiLU(mlp.0(that))) -- host math.
std::vector<float>
MetalMageVae::t_embed_zero_(const MetalLlamaWeights& wts,
                            const std::string& t_pre) const
{
  std::size_t n1 = 0, nb1 = 0, n2 = 0, nb2 = 0;
  auto* self = const_cast<MetalMageVae*>(this);
  std::vector<float> w1 = read_f32_(wts, self->_mc, t_pre + "mlp.0.weight", n1);
  std::vector<float> b1 = read_f32_(wts, self->_mc, t_pre + "mlp.0.bias", nb1);
  std::vector<float> w2 = read_f32_(wts, self->_mc, t_pre + "mlp.2.weight", n2);
  std::vector<float> b2 = read_f32_(wts, self->_mc, t_pre + "mlp.2.bias", nb2);
  if (w1.empty() || w2.empty()) { return {}; }
  const int H = (int)nb1;             // hidden
  const int F = (int)(n1 / (std::size_t)H);   // frequency_embedding_size (256)
  std::vector<float> h((std::size_t)H, 0.0f);
  for (int o = 0; o < H; ++o) {
    float acc = b1[(std::size_t)o];
    // emb[i] = 1 for i < F/2 (cos(0)), 0 otherwise -- only the first half
    // of each weight row contributes.
    for (int i = 0; i < F / 2; ++i) { acc += w1[(std::size_t)o * F + i]; }
    h[(std::size_t)o] = silu_(acc);
  }
  std::vector<float> c((std::size_t)H, 0.0f);
  for (int o = 0; o < H; ++o) {
    float acc = b2[(std::size_t)o];
    for (int i = 0; i < H; ++i) { acc += w2[(std::size_t)o * H + i] * h[i]; }
    c[(std::size_t)o] = acc;
  }
  return c;
}

// adaLN_modulation = Sequential(SiLU, Linear(H, 6H)) applied to the constant
// t_embedder(0) -- the whole thing folds to a fixed [6H] vector.
std::vector<float>
MetalMageVae::fold_adaln_(const MetalLlamaWeights& wts,
                          const std::string& t_pre,
                          const std::string& adaln_pre) const
{
  std::vector<float> c = t_embed_zero_(wts, t_pre);
  if (c.empty()) { return {}; }
  auto* self = const_cast<MetalMageVae*>(this);
  std::size_t nw = 0, nb = 0;
  std::vector<float> w = read_f32_(wts, self->_mc, adaln_pre + ".weight", nw);
  std::vector<float> b = read_f32_(wts, self->_mc, adaln_pre + ".bias", nb);
  if (w.empty() || b.empty()) { return {}; }
  const int H = (int)c.size();
  const int O = (int)nb;              // 6*H
  std::vector<float> s((std::size_t)H);
  for (int i = 0; i < H; ++i) { s[(std::size_t)i] = silu_(c[(std::size_t)i]); }
  std::vector<float> out((std::size_t)O);
  for (int o = 0; o < O; ++o) {
    float acc = b[(std::size_t)o];
    for (int i = 0; i < H; ++i) { acc += w[(std::size_t)o * H + i] * s[i]; }
    out[(std::size_t)o] = acc;
  }
  return out;
}

bool
MetalMageVae::load_dico_(const MetalLlamaWeights& wts, const std::string& pre,
                         DiCoBlock& b, int c, bool with_mod,
                         bool with_norm_affine)
{
  b.c = c;
  b.c1 = load_linear_(wts, pre + "conv1");
  b.c2 = load_dwconv_(wts, pre + "conv2");
  b.c3 = load_linear_(wts, pre + "conv3");
  b.c4 = load_linear_(wts, pre + "conv4");
  b.c5 = load_linear_(wts, pre + "conv5");
  b.ca = load_linear_(wts, pre + "ca.1");
  if (with_norm_affine) {
    b.n1w = load_vec_(wts, pre + "norm1.weight");
    b.n1b = load_vec_(wts, pre + "norm1.bias");
    b.n2w = load_vec_(wts, pre + "norm2.weight");
    b.n2b = load_vec_(wts, pre + "norm2.bias");
    if (b.n1w.empty() || b.n1b.empty() || b.n2w.empty() || b.n2b.empty()) {
      return false;
    }
  }
  return !b.c1.empty() && !b.c2.w.empty() && !b.c3.empty() && !b.c4.empty()
         && !b.c5.empty() && !b.ca.empty() && (!with_mod || b.has_mod);
}

bool
MetalMageVae::load_encoder_(const MetalLlamaWeights& wts)
{
  const std::string p = kEncPre;
  _enc_patch = load_patch_conv_(wts, p + "patch_cond_embed");
  if (_enc_patch.empty()) { return false; }

  _enc_head.resize((std::size_t)_cfg.n_head_blocks);
  for (int i = 0; i < _cfg.n_head_blocks; ++i) {
    const std::string bp = p + "head_blocks." + std::to_string(i) + ".";
    if (!load_dico_(wts, bp, _enc_head[(std::size_t)i], _cfg.enc_head_dim,
                    false, true)) {
      return false;
    }
  }
  _enc_proj_down = load_linear_(wts, p + "proj_down");

  // fuse_proj sees cat([cond, z_proj(z_t)]) with z_t == 0, so the z half is
  // the constant z_proj.bias. Fold W[:, hidden:] @ z_proj.bias into the
  // fuse bias and keep only the `cond` columns of the weight -- the concat
  // disappears entirely.
  {
    std::size_t nw = 0, nb = 0, nzb = 0;
    std::vector<float> w = read_f32_(wts, _mc, p + "fuse_proj.weight", nw);
    std::vector<float> b = read_f32_(wts, _mc, p + "fuse_proj.bias", nb);
    std::vector<float> zb = read_f32_(wts, _mc, p + "z_proj.bias", nzb);
    if (w.empty() || b.empty() || zb.empty()) { return false; }
    const int H = _cfg.hidden;
    const int O = (int)nb;                 // == H
    const int K = (int)(nw / (std::size_t)O);   // == 2*H
    std::vector<float> wc((std::size_t)O * H);
    std::vector<float> bc((std::size_t)O);
    for (int o = 0; o < O; ++o) {
      float acc = b[(std::size_t)o];
      for (int i = 0; i < H; ++i) {
        wc[(std::size_t)o * H + i] = w[(std::size_t)o * K + i];
        acc += w[(std::size_t)o * K + H + i] * zb[(std::size_t)i];
      }
      bc[(std::size_t)o] = acc;
    }
    _enc_fuse.cout = O; _enc_fuse.cin = H; _enc_fuse.k = H;
    _enc_fuse.w = f16_buf_(_mc, wc.data(), wc.size());
    _enc_fuse.b = f16_buf_(_mc, bc.data(), bc.size());
  }

  _enc_blocks.resize((std::size_t)_cfg.n_blocks);
  for (int i = 0; i < _cfg.n_blocks; ++i) {
    const std::string bp = p + "blocks." + std::to_string(i) + ".";
    std::vector<float> mod =
        fold_adaln_(wts, p + "t_embedder.", bp + "adaLN_modulation.1");
    if (mod.empty()) { return false; }
    _enc_blocks[(std::size_t)i].mod = f16_buf_(_mc, mod.data(), mod.size());
    _enc_blocks[(std::size_t)i].has_mod = true;
    if (!load_dico_(wts, bp, _enc_blocks[(std::size_t)i], _cfg.hidden, true,
                    false)) {
      return false;
    }
  }
  _enc_no_w = load_vec_(wts, p + "norm_out.weight");
  _enc_no_b = load_vec_(wts, p + "norm_out.bias");
  _enc_proj_out = load_linear_(wts, p + "proj_out");
  return !_enc_proj_down.empty() && !_enc_no_w.empty() && !_enc_no_b.empty()
         && !_enc_proj_out.empty();
}

bool
MetalMageVae::load_resblock_(const MetalLlamaWeights& wts,
                             const std::string& pre, ResBlock& rb)
{
  rb.n1w = load_vec_(wts, pre + "norm1.weight");
  rb.n1b = load_vec_(wts, pre + "norm1.bias");
  rb.n2w = load_vec_(wts, pre + "norm2.weight");
  rb.n2b = load_vec_(wts, pre + "norm2.bias");
  rb.c1 = load_conv3x3_(wts, pre + "conv1");
  rb.c2 = load_conv3x3_(wts, pre + "conv2");
  return !rb.n1w.empty() && !rb.n1b.empty() && !rb.n2w.empty()
         && !rb.n2b.empty() && !rb.c1.empty() && !rb.c2.empty();
}

bool
MetalMageVae::load_attn_(const MetalLlamaWeights& wts, const std::string& pre,
                         AttnBlock& a)
{
  a.nw = load_vec_(wts, pre + "norm.weight");
  a.nb = load_vec_(wts, pre + "norm.bias");
  a.q = load_linear_(wts, pre + "q");
  a.k = load_linear_(wts, pre + "k");
  a.v = load_linear_(wts, pre + "v");
  a.proj = load_linear_(wts, pre + "proj_out");
  return !a.nw.empty() && !a.nb.empty() && !a.q.empty() && !a.k.empty()
         && !a.v.empty() && !a.proj.empty();
}

bool
MetalMageVae::load_decoder_(const MetalLlamaWeights& wts)
{
  const std::string p = "pipeline.";
  const std::string dp = p + "y_embedder.decoder.";
  _dec_conv_in = load_conv3x3_(wts, dp + "conv_in");
  if (!load_resblock_(wts, dp + "block.0.", _dec_res0)
      || !load_attn_(wts, dp + "block.1.", _dec_attn0)
      || !load_resblock_(wts, dp + "block.2.", _dec_res1)
      || !load_attn_(wts, dp + "block.3.", _dec_attn1)
      || !load_resblock_(wts, dp + "block.4.", _dec_res2)) {
    return false;
  }
  _dec_no_w = load_vec_(wts, dp + "norm_out.weight");
  _dec_no_b = load_vec_(wts, dp + "norm_out.bias");
  _dec_conv_out = load_conv3x3_(wts, dp + "conv_out");

  // s_embedder = proj2(cat([proj1(noise), cond])) with noise == 0 and proj1
  // bias-free, so the first `latent_channels` input columns are dead --
  // keep only the `cond` columns and the concat disappears.
  {
    std::size_t nw = 0, nb = 0;
    std::vector<float> w = read_f32_(wts, _mc, p + "s_embedder.proj2.weight",
                                     nw);
    std::vector<float> b = read_f32_(wts, _mc, p + "s_embedder.proj2.bias", nb);
    if (w.empty() || b.empty()) { return false; }
    const int O = (int)nb;                      // hidden
    const int K = (int)(nw / (std::size_t)O);   // latent + hidden
    const int Z = K - _cfg.hidden;
    std::vector<float> wc((std::size_t)O * _cfg.hidden);
    for (int o = 0; o < O; ++o) {
      for (int i = 0; i < _cfg.hidden; ++i) {
        wc[(std::size_t)o * _cfg.hidden + i] = w[(std::size_t)o * K + Z + i];
      }
    }
    _dec_s_embed.cout = O; _dec_s_embed.cin = _cfg.hidden;
    _dec_s_embed.k = _cfg.hidden;
    _dec_s_embed.w = f16_buf_(_mc, wc.data(), wc.size());
    _dec_s_embed.b = f16_buf_(_mc, b.data(), b.size());
  }

  _dec_blocks.resize((std::size_t)_cfg.n_blocks);
  for (int i = 0; i < _cfg.n_blocks; ++i) {
    const std::string bp = p + "blocks." + std::to_string(i) + ".";
    std::vector<float> mod =
        fold_adaln_(wts, p + "t_embedder.", bp + "adaLN_modulation.1");
    if (mod.empty()) { return false; }
    _dec_blocks[(std::size_t)i].mod = f16_buf_(_mc, mod.data(), mod.size());
    _dec_blocks[(std::size_t)i].has_mod = true;
    if (!load_dico_(wts, bp, _dec_blocks[(std::size_t)i], _cfg.hidden, true,
                    false)) {
      return false;
    }
  }

  // y_embedder_x emits channel (f * P^2 + q); the per-pixel head consumes it
  // as [patch_pixel q][feature f]. Permuting the weight ROWS to (q, f) at
  // load makes the GEMM output a CONTIGUOUS [hw*P^2, x_dim] reshape, so no
  // runtime gather is needed.
  {
    std::size_t nw = 0, nb = 0;
    std::vector<float> w = read_f32_(wts, _mc, p + "y_embedder_x.weight", nw);
    std::vector<float> b = read_f32_(wts, _mc, p + "y_embedder_x.bias", nb);
    if (w.empty() || b.empty()) { return false; }
    const int O = (int)nb;                      // x_dim * P^2
    const int K = (int)(nw / (std::size_t)O);   // hidden
    const int PP = _cfg.patch * _cfg.patch;
    const int F = O / PP;                       // == x_dim
    std::vector<float> wp((std::size_t)O * K), bp2((std::size_t)O);
    for (int f = 0; f < F; ++f) {
      for (int q = 0; q < PP; ++q) {
        const std::size_t src = (std::size_t)f * PP + q;
        const std::size_t dst = (std::size_t)q * F + f;
        std::memcpy(&wp[dst * K], &w[src * K], (std::size_t)K * sizeof(float));
        bp2[dst] = b[src];
      }
    }
    _dec_y_x.cout = O; _dec_y_x.cin = K; _dec_y_x.k = K;
    _dec_y_x.w = f16_buf_(_mc, wp.data(), wp.size());
    _dec_y_x.b = f16_buf_(_mc, bp2.data(), bp2.size());
  }

  // dec_net.cond_embed already emits (q, f) order (it is reshaped to
  // [P^2, x_dim]), so it loads unpermuted.
  _dec_cond_embed = load_linear_(wts, p + "dec_net.cond_embed");
  _dec_input_proj = load_linear_(wts, p + "dec_net.input_proj");

  // x_embedder is Linear(in_ch + x_dim + max_freqs^2 -> x_dim) over
  // cat([patch pixels, cond features, DCT position]). The patch pixels are
  // the zero noise, so those columns are dead; the DCT block is a constant
  // per intra-patch pixel, so cols [in+x_dim ..] @ dct + bias folds into a
  // [P^2, x_dim] table. What survives at runtime is a [x_dim, x_dim] GEMM.
  {
    std::size_t nw = 0, nb = 0;
    const std::string xe = p + "x_embedder.embedder.0";
    std::vector<float> w = read_f32_(wts, _mc, xe + ".weight", nw);
    std::vector<float> b = read_f32_(wts, _mc, xe + ".bias", nb);
    if (w.empty() || b.empty()) { return false; }
    const int O = (int)nb;                      // x_dim
    const int K = (int)(nw / (std::size_t)O);   // 3 + x_dim + max_freqs^2
    const int NF = _cfg.max_freqs * _cfg.max_freqs;
    const int c0 = K - NF - O;                  // == in_channels (3)
    std::vector<float> wc((std::size_t)O * O);
    for (int o = 0; o < O; ++o) {
      for (int i = 0; i < O; ++i) {
        wc[(std::size_t)o * O + i] = w[(std::size_t)o * K + c0 + i];
      }
    }
    _dec_x_embed.cout = O; _dec_x_embed.cin = O; _dec_x_embed.k = O;
    _dec_x_embed.w = f16_buf_(_mc, wc.data(), wc.size());
    // DCT basis (NerfEmbedder.fetch_pos): for intra-patch pixel q = iy*P+ix,
    // dct[q][a*mf+b] = cos(px*fa*pi) * cos(py*fb*pi) / (1 + fa*fb), with
    // px/py on linspace(0,1,P) and fa/fb on linspace(0,max_freqs,max_freqs).
    const int P = _cfg.patch, MF = _cfg.max_freqs;
    const int PP = P * P;
    std::vector<float> tbl((std::size_t)PP * O);
    std::vector<float> dct((std::size_t)NF);
    for (int iy = 0; iy < P; ++iy) {
      for (int ix = 0; ix < P; ++ix) {
        const int q = iy * P + ix;
        const float py = (P > 1) ? (float)iy / (float)(P - 1) : 0.0f;
        const float px = (P > 1) ? (float)ix / (float)(P - 1) : 0.0f;
        for (int a = 0; a < MF; ++a) {
          const float fa = (MF > 1) ? (float)a * (float)MF / (float)(MF - 1)
                                    : 0.0f;
          for (int bb = 0; bb < MF; ++bb) {
            const float fb = (MF > 1) ? (float)bb * (float)MF / (float)(MF - 1)
                                      : 0.0f;
            dct[(std::size_t)a * MF + bb] =
                std::cos(px * fa * (float)M_PI)
                * std::cos(py * fb * (float)M_PI) / (1.0f + fa * fb);
          }
        }
        for (int o = 0; o < O; ++o) {
          float acc = b[(std::size_t)o];
          for (int j = 0; j < NF; ++j) {
            acc += w[(std::size_t)o * K + c0 + O + j] * dct[(std::size_t)j];
          }
          tbl[(std::size_t)q * O + o] = acc;
        }
      }
    }
    _dec_x_const = f16_buf_(_mc, tbl.data(), tbl.size());
  }

  _dec_mlp.resize((std::size_t)_cfg.n_mlp_res);
  for (int i = 0; i < _cfg.n_mlp_res; ++i) {
    const std::string rp = p + "dec_net.res_blocks." + std::to_string(i) + ".";
    MlpResBlock& r = _dec_mlp[(std::size_t)i];
    r.lnw = load_vec_(wts, rp + "in_ln.weight");
    r.lnb = load_vec_(wts, rp + "in_ln.bias");
    r.adaln = load_linear_(wts, rp + "adaLN_modulation.1");
    r.fc1 = load_linear_(wts, rp + "mlp.0");
    r.fc2 = load_linear_(wts, rp + "mlp.2");
    if (r.lnw.empty() || r.lnb.empty() || r.adaln.empty() || r.fc1.empty()
        || r.fc2.empty()) {
      return false;
    }
  }
  _dec_final_n = load_vec_(wts, p + "final_layer.norm.weight");
  _dec_final_lin = load_linear_(wts, p + "final_layer.linear");

  return !_dec_conv_in.empty() && !_dec_no_w.empty() && !_dec_no_b.empty()
         && !_dec_conv_out.empty() && !_dec_cond_embed.empty()
         && !_dec_input_proj.empty() && !_dec_final_n.empty()
         && !_dec_final_lin.empty();
}

std::unique_ptr<MetalMageVae>
MetalMageVae::load(const std::string& model_dir, MetalCompute* mc,
                   const Config& cfg, bool with_encoder)
{
  if (mc == nullptr) { return nullptr; }
  auto wtsopt = MetalLlamaWeights::open_model(model_dir);
  if (!wtsopt.has_value()) { return nullptr; }
  const MetalLlamaWeights& wts = *wtsopt;

  auto m = std::unique_ptr<MetalMageVae>(new MetalMageVae());
  m->_mc = mc;
  m->_cfg = cfg;

  m->_lib_gemm = mc->load_library("dense_gemm");
  m->_lib_elt  = mc->load_library("llm_elementwise");
  m->_lib_rms  = mc->load_library("rms_norm");
  m->_lib_sdpa = mc->load_library("sdpa");
  m->_fn_gemm_bias  = m->_lib_gemm.function("dense_gemm_bias_f16");
  m->_fn_rms        = m->_lib_rms.function("rms_norm_fast_f16");
  m->_fn_gelu       = m->_lib_elt.function("gelu_erf_f16");
  m->_fn_swish      = m->_lib_elt.function("mul_sigmoid_f16");
  m->_fn_residual   = m->_lib_elt.function("residual_add_f16");
  m->_fn_im2col     = m->_lib_elt.function("im2col_hwc_3x3_f16");
  m->_fn_ln_plain   = m->_lib_elt.function("layer_norm_plain_f16");
  m->_fn_ln_affine  = m->_lib_elt.function("layer_norm_affine_f16");
  m->_fn_ln_mod     = m->_lib_elt.function("layernorm_modulate_f16");
  m->_fn_gated      = m->_lib_elt.function("gated_residual_f16");
  m->_fn_groupnorm  = m->_lib_elt.function("group_norm_f16");
  m->_fn_dw3x3      = m->_lib_elt.function("depthwise_conv2d_3x3_hwc_f16");
  m->_fn_col_mean   = m->_lib_elt.function("col_mean_f16");
  m->_fn_mul_sig    = m->_lib_elt.function("mul_rows_sigmoid_f16");
  m->_fn_bias_add   = m->_lib_elt.function("bias_add_rows_f16");
  m->_fn_copy       = m->_lib_elt.function("copy_f16");
  m->_fn_sdpa       = m->_lib_sdpa.function("sdpa_full_f16");
  m->_fn_tile_gather  = m->_lib_elt.function("tile_gather_clamp_hwc_f16");
  m->_fn_tile_scatter = m->_lib_elt.function("tile_scatter_hwc_f16");
  m->_fn_ln_mod_rows  = m->_lib_elt.function("layer_norm_mod_rows_f16");
  m->_fn_gated_rows   = m->_lib_elt.function("gated_residual_rows_f16");
  m->_fn_add_rows_mod = m->_lib_elt.function("add_rows_mod_f16");
  if (!m->_fn_gemm_bias.valid() || !m->_fn_gelu.valid()
      || !m->_fn_ln_affine.valid() || !m->_fn_ln_mod.valid()
      || !m->_fn_gated.valid() || !m->_fn_dw3x3.valid()
      || !m->_fn_col_mean.valid() || !m->_fn_mul_sig.valid()
      || !m->_fn_bias_add.valid() || !m->_fn_residual.valid()
      || !m->_fn_groupnorm.valid() || !m->_fn_swish.valid()
      || !m->_fn_im2col.valid() || !m->_fn_rms.valid() || !m->_fn_sdpa.valid()
      || !m->_fn_copy.valid() || !m->_fn_tile_gather.valid()
      || !m->_fn_tile_scatter.valid() || !m->_fn_ln_mod_rows.valid()
      || !m->_fn_gated_rows.valid() || !m->_fn_add_rows_mod.valid()) {
    return nullptr;
  }
  // M5 matrix-core dense GEMM for the 1x1 / patch GEMMs (M = H*W pixels, so
  // the tiled matmul2d amortizes well), mirroring the Krea-2 / FLUX.2 VAEs.
  if (mc->supports_matrix_cores()
      && std::getenv("VPIPE_MAGE_NO_MMA2") == nullptr) {
    m->_lib_dense_mma = mc->load_library("dense_gemm_mma");
    m->_fn_dense_mma = m->_lib_dense_mma.function("dense_gemm_mma_t_n128_f16");
    m->_fn_dense_mma_deep =
        m->_lib_dense_mma.function("dense_gemm_mma_t_n128x256_f16");
    m->_use_mma2 = m->_fn_dense_mma.valid() && m->_fn_dense_mma_deep.valid();
    if (const char* e = std::getenv("VPIPE_MAGE_VAE_MMA_MAX_M")) {
      m->_mma_max_m = std::atoi(e);
    }
  }

  // The decoder half is ~105M params and every caller decodes, so it always
  // loads; the encoder (~67M) is optional.
  if (!m->load_decoder_(wts)) { return nullptr; }
  if (with_encoder) {
    if (!m->load_encoder_(wts)) { return nullptr; }
    m->_has_encoder = true;
  }
  return m;
}

// ---------------------------------------------------------------------------
// Forward helpers
// ---------------------------------------------------------------------------

void
MetalMageVae::gemm_(ComputeEncoder& enc, const SharedBuffer& x, std::size_t xe,
                    const Conv& c, const SharedBuffer& y, std::size_t ye, int M)
{
  const int N = c.cout, K = c.k;
  if (_use_mma2 && M >= _mma_min_m && N >= _mma_min_n && K >= _mma_min_k) {
    const bool deep = (K >= 6144);
    const int BN = deep ? 256 : 128;
    // Row-split under the matmul2d high-M corruption threshold (see the
    // Krea-2 VAE: the MPP op silently corrupts output rows past M ~ 2^19).
    const int chunk = (_mma_max_m > 0 && M > _mma_max_m) ? _mma_max_m : M;
    for (int r0 = 0; r0 < M; r0 += chunk) {
      const int mc = (M - r0 < chunk) ? (M - r0) : chunk;
      enc.set_function(deep ? _fn_dense_mma_deep : _fn_dense_mma);
      enc.set_buffer(0, x, (xe + (std::size_t)r0 * K) * 2);
      enc.set_buffer(1, c.w);
      enc.set_buffer(2, c.w);          // bias slot unused
      enc.set_buffer(3, y, (ye + (std::size_t)r0 * N) * 2);
      enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, mc);
      enc.set_constant(7, 0);
      enc.dispatch({(unsigned)(((N + BN - 1) / BN) * 256),
                    (unsigned)((mc + 127) / 128), 1}, {256, 1, 1});
    }
    if (!c.b.empty()) {
      enc.set_function(_fn_bias_add);
      enc.set_buffer(0, y, ye * 2); enc.set_buffer(1, c.b);
      enc.set_constant(2, N);
      enc.set_constant(3, (unsigned)((std::size_t)M * N));
      enc.dispatch({(unsigned)N, (unsigned)M, 1}, {256, 1, 1});
    }
    return;
  }
  enc.set_function(_fn_gemm_bias);
  enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, c.w);
  enc.set_buffer(2, c.b.empty() ? c.w : c.b); enc.set_buffer(3, y, ye * 2);
  enc.set_constant(4, M); enc.set_constant(5, N); enc.set_constant(6, K);
  enc.set_constant(7, c.b.empty() ? 0 : 1);
  enc.dispatch({(unsigned)(((N + 15) / 16) * 16),
                (unsigned)(((M + 15) / 16) * 16), 1}, {16, 16, 1});
}

// One DiCo block, in place on x [H*W, C].
//   trunk (has_mod): x = x + gate_msa * conv3(ca(gelu(dw(conv1(
//                            modulate(LN(x), shift_msa, scale_msa))))))
//                    x = x + gate_mlp * conv5(gelu(conv4(
//                            modulate(LN(x), shift_mlp, scale_mlp))))
//   head  (affine LN, ungated): the same body with plain residuals.
void
MetalMageVae::dico_(ComputeEncoder& enc, const DiCoBlock& b,
                    const SharedBuffer& x, int H, int W, const SharedBuffer& t1,
                    const SharedBuffer& t2, const SharedBuffer& pool)
{
  const int C = b.c;
  const int HW = H * W;
  const std::size_t n = (std::size_t)HW * C;
  const int ffn = b.c4.cout;

  auto norm_into = [&](const SharedBuffer& dst, int which) {
    if (b.has_mod) {
      // shift at [which*3C], scale at [which*3C + C] (chunk order is
      // shift, scale, gate per half).
      const std::size_t base = (std::size_t)which * 3 * C;
      enc.set_function(_fn_ln_mod);
      enc.set_buffer(0, x); enc.set_buffer(1, b.mod, (base + C) * 2);
      enc.set_buffer(2, b.mod, base * 2); enc.set_buffer(3, dst);
      enc.set_constant(4, C); enc.set_constant(5, _cfg.norm_eps);
    } else {
      enc.set_function(_fn_ln_affine);
      enc.set_buffer(0, x);
      enc.set_buffer(1, which == 0 ? b.n1w : b.n2w);
      enc.set_buffer(2, which == 0 ? b.n1b : b.n2b);
      enc.set_buffer(3, dst);
      enc.set_constant(4, C); enc.set_constant(5, _cfg.norm_eps);
    }
    enc.dispatch({256, (unsigned)HW, 1}, {256, 1, 1});
  };

  // ---- conv branch ----
  norm_into(t1, 0);
  gemm_(enc, t1, 0, b.c1, t2, 0, HW);            // conv1 1x1
  enc.set_function(_fn_dw3x3);                   // conv2 depthwise 3x3
  enc.set_buffer(0, t2); enc.set_buffer(1, b.c2.w);
  enc.set_buffer(2, b.c2.b.empty() ? b.c2.w : b.c2.b);
  enc.set_buffer(3, t1);
  enc.set_constant(4, H); enc.set_constant(5, W); enc.set_constant(6, C);
  enc.set_constant(7, b.c2.b.empty() ? 0 : 1);
  enc.dispatch({(unsigned)C, (unsigned)HW, 1}, {256, 1, 1});
  enc.set_function(_fn_gelu);
  enc.set_buffer(0, t1); enc.set_buffer(1, t1);
  enc.set_constant(2, (int)n);
  enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
  // channel attention: spatial mean -> 1x1 -> sigmoid -> per-channel scale
  enc.set_function(_fn_col_mean);
  enc.set_buffer(0, t1); enc.set_buffer(1, pool);
  enc.set_constant(2, HW); enc.set_constant(3, C);
  enc.dispatch({(unsigned)(C * 256), 1, 1}, {256, 1, 1});
  gemm_(enc, pool, 0, b.ca, pool, (std::size_t)C, 1);
  enc.set_function(_fn_mul_sig);
  enc.set_buffer(0, t1); enc.set_buffer(1, pool, (std::size_t)C * 2);
  enc.set_constant(2, C); enc.set_constant(3, (unsigned)n);
  enc.dispatch({(unsigned)C, (unsigned)HW, 1}, {256, 1, 1});
  gemm_(enc, t1, 0, b.c3, t2, 0, HW);            // conv3 1x1
  if (b.has_mod) {
    enc.set_function(_fn_gated);                 // x += gate_msa * t2
    enc.set_buffer(0, x); enc.set_buffer(1, b.mod, (std::size_t)2 * C * 2);
    enc.set_buffer(2, t2);
    enc.set_constant(3, C); enc.set_constant(4, (int)n);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
  } else {
    enc.set_function(_fn_residual);              // x += t2
    enc.set_buffer(0, x); enc.set_buffer(1, t2); enc.set_buffer(2, x);
    enc.set_constant(3, (int)n);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
  }

  // ---- FFN branch ----
  norm_into(t1, 1);
  gemm_(enc, t1, 0, b.c4, t2, 0, HW);            // conv4 1x1 -> ffn
  enc.set_function(_fn_gelu);
  enc.set_buffer(0, t2); enc.set_buffer(1, t2);
  enc.set_constant(2, (int)((std::size_t)HW * ffn));
  enc.dispatch({(unsigned)((std::size_t)HW * ffn), 1, 1}, {256, 1, 1});
  gemm_(enc, t2, 0, b.c5, t1, 0, HW);            // conv5 ffn -> C
  if (b.has_mod) {
    enc.set_function(_fn_gated);
    enc.set_buffer(0, x); enc.set_buffer(1, b.mod, (std::size_t)5 * C * 2);
    enc.set_buffer(2, t1);
    enc.set_constant(3, C); enc.set_constant(4, (int)n);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
  } else {
    enc.set_function(_fn_residual);
    enc.set_buffer(0, x); enc.set_buffer(1, t1); enc.set_buffer(2, x);
    enc.set_constant(3, (int)n);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
  }
}

// ---------------------------------------------------------------------------
// encode
// ---------------------------------------------------------------------------
SharedBuffer
MetalMageVae::encode(const SharedBuffer& img, int H, int W, std::string* err)
{
  auto fail = [&](const char* m) -> SharedBuffer {
    if (err != nullptr) { *err = m; }
    return {};
  };
  if (!_has_encoder) { return fail("MageVAE: encoder not loaded"); }
  const int P = _cfg.patch;
  if (H <= 0 || W <= 0 || (H % P) != 0 || (W % P) != 0) {
    return fail("MageVAE: H and W must be positive multiples of 16");
  }
  const int h = H / P, w = W / P;
  const int HW = h * w;
  const int HD = _cfg.enc_head_dim, C = _cfg.hidden;
  const int Kp = _enc_patch.k;                 // 3*P*P

  // Host patchify: col[p, ci*P*P + ky*P + kx] = img[ci, y*P+ky, x*P+kx],
  // matching PyTorch's Conv2d weight flatten (see load_patch_conv_).
  SharedBuffer col = _mc->make_shared_buffer((std::size_t)HW * Kp * 2);
  if (col.empty()) { return fail("MageVAE: patch buffer alloc failed"); }
  {
    const auto* src = static_cast<const _Float16*>(img.contents());
    auto* dst = static_cast<_Float16*>(col.contents());
    const std::size_t plane = (std::size_t)H * W;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        _Float16* row = dst + ((std::size_t)y * w + x) * Kp;
        for (int ci = 0; ci < 3; ++ci) {
          for (int ky = 0; ky < P; ++ky) {
            const std::size_t s0 = (std::size_t)ci * plane
                                 + (std::size_t)(y * P + ky) * W + x * P;
            _Float16* d0 = row + (std::size_t)ci * P * P + (std::size_t)ky * P;
            for (int kx = 0; kx < P; ++kx) { d0[kx] = src[s0 + kx]; }
          }
        }
      }
    }
  }

  const std::size_t big = (std::size_t)HW * (std::size_t)HD * _cfg.mlp_ratio;
  SharedBuffer xh = _mc->make_shared_buffer((std::size_t)HW * HD * 2);
  SharedBuffer t1 = _mc->make_shared_buffer(big * 2);
  SharedBuffer t2 = _mc->make_shared_buffer(big * 2);
  SharedBuffer xs = _mc->make_shared_buffer((std::size_t)HW * C * 2);
  SharedBuffer pool = _mc->make_shared_buffer((std::size_t)HD * 2 * 2);
  SharedBuffer out = _mc->make_shared_buffer(
      (std::size_t)HW * _enc_proj_out.cout * 2);
  if (xh.empty() || t1.empty() || t2.empty() || xs.empty() || pool.empty()
      || out.empty()) {
    return fail("MageVAE: encode scratch alloc failed");
  }

  CommandStream stream = _mc->make_command_stream();
  {
  ComputeEncoder enc = stream.begin_compute();

  gemm_(enc, col, 0, _enc_patch, xh, 0, HW);            // patch_cond_embed
  for (auto& b : _enc_head) { dico_(enc, b, xh, h, w, t1, t2, pool); }
  gemm_(enc, xh, 0, _enc_proj_down, xs, 0, HW);         // 768 -> 384
  // fuse_proj with the z_proj(0) half folded into the bias at load. Lands
  // back in xh (sized for the wider head width, so it holds the trunk too),
  // which the 21 DiCo blocks then run on in place.
  gemm_(enc, xs, 0, _enc_fuse, xh, 0, HW);
  for (auto& b : _enc_blocks) { dico_(enc, b, xh, h, w, t1, t2, pool); }
  enc.set_function(_fn_ln_affine);                      // norm_out (affine)
  enc.set_buffer(0, xh); enc.set_buffer(1, _enc_no_w);
  enc.set_buffer(2, _enc_no_b); enc.set_buffer(3, t1);
  enc.set_constant(4, C); enc.set_constant(5, _cfg.norm_eps);
  enc.dispatch({256, (unsigned)HW, 1}, {256, 1, 1});
  gemm_(enc, t1, 0, _enc_proj_out, out, 0, HW);         // -> 2*latent
  }
  std::string gpu_err;
  if (!stream.commit().wait_ok(&gpu_err)) {
    return fail(gpu_err.empty() ? "MageVAE: encode GPU commit failed"
                                : gpu_err.c_str());
  }

  // Keep the MEAN half (sample_posterior=false) and hand back channel-first
  // [latent, h, w].
  const int Z = _cfg.latent_channels;
  const int O = _enc_proj_out.cout;
  SharedBuffer z = _mc->make_shared_buffer((std::size_t)Z * HW * 2);
  if (z.empty()) { return fail("MageVAE: latent alloc failed"); }
  {
    const auto* s = static_cast<const _Float16*>(out.contents());
    auto* d = static_cast<_Float16*>(z.contents());
    for (int c = 0; c < Z; ++c) {
      for (int p = 0; p < HW; ++p) {
        d[(std::size_t)c * HW + p] = s[(std::size_t)p * O + c];
      }
    }
  }
  return z;
}

// im2col 3x3 (pad 1, stride 1) -> dense GEMM.
void
MetalMageVae::conv3x3_(ComputeEncoder& enc, const SharedBuffer& in,
                       const Conv& c, const SharedBuffer& out, int H, int W,
                       const SharedBuffer& col)
{
  enc.set_function(_fn_im2col);
  enc.set_buffer(0, in); enc.set_buffer(1, col);
  enc.set_constant(2, H); enc.set_constant(3, W); enc.set_constant(4, c.cin);
  enc.dispatch({(unsigned)(9 * c.cin), (unsigned)(H * W), 1}, {64, 1, 1});
  gemm_(enc, col, 0, c, out, 0, H * W);
}

void
MetalMageVae::groupnorm_(ComputeEncoder& enc, const SharedBuffer& x,
                         const SharedBuffer& g, const SharedBuffer& b,
                         const SharedBuffer& out, int HW, int C)
{
  enc.set_function(_fn_groupnorm);
  enc.set_buffer(0, x); enc.set_buffer(1, g); enc.set_buffer(2, b);
  enc.set_buffer(3, out);
  enc.set_constant(4, HW); enc.set_constant(5, C);
  enc.set_constant(6, _cfg.norm_groups); enc.set_constant(7, _cfg.norm_eps);
  enc.dispatch({256, (unsigned)_cfg.norm_groups, 1}, {256, 1, 1});
}

// h = x + conv2(swish(norm2(conv1(swish(norm1(x)))))). in == out channels
// throughout the CoD decoder, so there is no shortcut projection.
void
MetalMageVae::resblock_(ComputeEncoder& enc, const ResBlock& rb,
                        const SharedBuffer& x, int H, int W,
                        const SharedBuffer& t1, const SharedBuffer& t2,
                        const SharedBuffer& col)
{
  const int HW = H * W, C = rb.c1.cin;
  const std::size_t n = (std::size_t)HW * C;
  groupnorm_(enc, x, rb.n1w, rb.n1b, t1, HW, C);
  enc.set_function(_fn_swish);                   // x * sigmoid(x)
  enc.set_buffer(0, t1); enc.set_buffer(1, t1); enc.set_buffer(2, t1);
  enc.set_constant(3, (int)n);
  enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
  conv3x3_(enc, t1, rb.c1, t2, H, W, col);
  groupnorm_(enc, t2, rb.n2w, rb.n2b, t1, HW, rb.c1.cout);
  enc.set_function(_fn_swish);
  enc.set_buffer(0, t1); enc.set_buffer(1, t1); enc.set_buffer(2, t1);
  enc.set_constant(3, (int)n);
  enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
  conv3x3_(enc, t1, rb.c2, t2, H, W, col);
  enc.set_function(_fn_residual);
  enc.set_buffer(0, x); enc.set_buffer(1, t2); enc.set_buffer(2, x);
  enc.set_constant(3, (int)n);
  enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
}

// x = x + proj_out(attn(q,k,v)) with the attention computed independently
// inside each `attn_patch` x `attn_patch` tile (single head, head_dim = C).
// The reference replicate-pads to whole tiles; the gather folds that in as a
// coordinate clamp, and the scatter drops the padded positions.
void
MetalMageVae::attnblock_(ComputeEncoder& enc, const AttnBlock& a,
                         const SharedBuffer& x, int H, int W,
                         const SharedBuffer& t1, const SharedBuffer& qkv)
{
  const int HW = H * W, C = a.q.cin, d = _cfg.attn_patch;
  const int nph = (H + d - 1) / d, npw = (W + d - 1) / d;
  const int nt = nph * npw, dd = d * d;
  const std::size_t tile = (std::size_t)dd * C;
  const std::size_t all = (std::size_t)nt * tile;

  groupnorm_(enc, x, a.nw, a.nb, t1, HW, C);
  // q, k, v, att live back to back in `qkv`.
  gemm_(enc, t1, 0, a.q, qkv, 0, HW);
  gemm_(enc, t1, 0, a.k, qkv, (std::size_t)HW * C, HW);
  gemm_(enc, t1, 0, a.v, qkv, (std::size_t)2 * HW * C, HW);
  const std::size_t gq = (std::size_t)3 * HW * C;         // gathered tiles
  for (int i = 0; i < 3; ++i) {
    enc.set_function(_fn_tile_gather);
    enc.set_buffer(0, qkv, (std::size_t)i * HW * C * 2);
    enc.set_buffer(1, qkv, (gq + (std::size_t)i * all) * 2);
    enc.set_constant(2, H); enc.set_constant(3, W); enc.set_constant(4, C);
    enc.set_constant(5, d); enc.set_constant(6, npw);
    enc.dispatch({(unsigned)C, (unsigned)dd, (unsigned)nt}, {64, 1, 1});
  }
  const std::size_t ao = gq + (std::size_t)3 * all;       // attention output
  const float scale = 1.0f / std::sqrt((float)C);
  for (int t = 0; t < nt; ++t) {
    const std::size_t off = (std::size_t)t * tile;
    enc.set_function(_fn_sdpa);
    enc.set_buffer(0, qkv, (gq + off) * 2);
    enc.set_buffer(1, qkv, (gq + all + off) * 2);
    enc.set_buffer(2, qkv, (gq + 2 * all + off) * 2);
    enc.set_buffer(3, qkv, (ao + off) * 2);
    enc.set_constant(4, scale); enc.set_constant(5, dd);
    enc.set_constant(6, C); enc.set_constant(7, 1); enc.set_constant(8, 1);
    enc.set_constant(9, dd); enc.set_constant(10, dd);
    enc.dispatch({32, 1, (unsigned)dd}, {32, 1, 1});
  }
  enc.set_function(_fn_tile_scatter);
  enc.set_buffer(0, qkv, ao * 2); enc.set_buffer(1, t1);
  enc.set_constant(2, H); enc.set_constant(3, W); enc.set_constant(4, C);
  enc.set_constant(5, d); enc.set_constant(6, npw);
  enc.dispatch({(unsigned)C, (unsigned)dd, (unsigned)nt}, {64, 1, 1});
  gemm_(enc, t1, 0, a.proj, qkv, 0, HW);
  enc.set_function(_fn_residual);
  enc.set_buffer(0, x); enc.set_buffer(1, qkv); enc.set_buffer(2, x);
  enc.set_constant(3, (int)((std::size_t)HW * C));
  enc.dispatch({(unsigned)((std::size_t)HW * C), 1, 1}, {256, 1, 1});
}

// ---------------------------------------------------------------------------
// decode
// ---------------------------------------------------------------------------
SharedBuffer
MetalMageVae::decode(const SharedBuffer& z, int h, int w, std::string* err)
{
  auto fail = [&](const char* m) -> SharedBuffer {
    if (err != nullptr) { *err = m; }
    return {};
  };
  if (_dec_conv_in.empty()) { return fail("MageVAE: decoder not loaded"); }
  if (h <= 0 || w <= 0) { return fail("MageVAE: bad latent size"); }
  const int P = _cfg.patch;
  const int HW = h * w, C = _cfg.hidden, Z = _cfg.latent_channels;
  const int X = _cfg.x_dim, PP = P * P;
  const std::size_t M = (std::size_t)HW * PP;   // per-pixel MLP rows
  const int d = _cfg.attn_patch;
  const int nt = ((h + d - 1) / d) * ((w + d - 1) / d);

  // Latent channel-first [Z, h, w] -> channel-last [hw, Z] (host).
  SharedBuffer zl = _mc->make_shared_buffer((std::size_t)HW * Z * 2);
  if (zl.empty()) { return fail("MageVAE: latent buffer alloc failed"); }
  {
    const auto* s = static_cast<const _Float16*>(z.contents());
    auto* dst = static_cast<_Float16*>(zl.contents());
    for (int c = 0; c < Z; ++c) {
      for (int p = 0; p < HW; ++p) {
        dst[(std::size_t)p * Z + c] = s[(std::size_t)c * HW + p];
      }
    }
  }

  const std::size_t hwc = (std::size_t)HW * C;
  const std::size_t ffn = hwc * (std::size_t)_cfg.mlp_ratio;
  const std::size_t yxw = (std::size_t)HW * PP * X;   // == M * X
  SharedBuffer cond = _mc->make_shared_buffer(hwc * 2);
  SharedBuffer s = _mc->make_shared_buffer(hwc * 2);
  SharedBuffer t1 = _mc->make_shared_buffer(ffn * 2);
  SharedBuffer t2 = _mc->make_shared_buffer(ffn * 2);
  SharedBuffer col = _mc->make_shared_buffer(hwc * 9 * 2);
  SharedBuffer qkv = _mc->make_shared_buffer(
      ((std::size_t)3 * hwc + (std::size_t)4 * nt * d * d * C) * 2);
  SharedBuffer pool = _mc->make_shared_buffer((std::size_t)C * 2 * 2);
  SharedBuffer yx = _mc->make_shared_buffer(yxw * 2);
  SharedBuffer ce = _mc->make_shared_buffer(yxw * 2);
  SharedBuffer xm = _mc->make_shared_buffer(M * X * 2);
  SharedBuffer hm = _mc->make_shared_buffer(M * X * 2);
  SharedBuffer h2 = _mc->make_shared_buffer(M * X * 2);
  SharedBuffer mod = _mc->make_shared_buffer(M * 3 * X * 2);
  SharedBuffer outm = _mc->make_shared_buffer(M * _dec_final_lin.cout * 2);
  if (cond.empty() || s.empty() || t1.empty() || t2.empty() || col.empty()
      || qkv.empty() || pool.empty() || yx.empty() || ce.empty() || xm.empty()
      || hm.empty() || h2.empty() || mod.empty() || outm.empty()) {
    return fail("MageVAE: decode scratch alloc failed (lower the resolution)");
  }

  CommandStream stream = _mc->make_command_stream();
  {
  ComputeEncoder enc = stream.begin_compute();

  // ---- CoD decoder: latent -> [hw, hidden] conditioning ----
  conv3x3_(enc, zl, _dec_conv_in, cond, h, w, col);
  resblock_(enc, _dec_res0, cond, h, w, t1, t2, col);
  attnblock_(enc, _dec_attn0, cond, h, w, t1, qkv);
  resblock_(enc, _dec_res1, cond, h, w, t1, t2, col);
  attnblock_(enc, _dec_attn1, cond, h, w, t1, qkv);
  resblock_(enc, _dec_res2, cond, h, w, t1, t2, col);
  groupnorm_(enc, cond, _dec_no_w, _dec_no_b, t1, HW, C);
  enc.set_function(_fn_swish);
  enc.set_buffer(0, t1); enc.set_buffer(1, t1); enc.set_buffer(2, t1);
  enc.set_constant(3, (int)hwc);
  enc.dispatch({(unsigned)hwc, 1, 1}, {256, 1, 1});
  conv3x3_(enc, t1, _dec_conv_out, cond, h, w, col);

  // ---- DiCo trunk on the s stream ----
  gemm_(enc, cond, 0, _dec_s_embed, s, 0, HW);
  for (auto& b : _dec_blocks) { dico_(enc, b, s, h, w, t1, t2, pool); }

  // ---- per-pixel MLP head ----
  // Both projections emit (patch_pixel, feature) order, so their [hw, PP*X]
  // outputs ARE the [M, X] per-pixel tensors with no reshuffle.
  gemm_(enc, cond, 0, _dec_y_x, yx, 0, HW);        // NerfEmbedder features
  gemm_(enc, s, 0, _dec_cond_embed, ce, 0, HW);    // per-pixel adaLN source
  enc.set_function(_fn_swish);                     // adaLN_modulation's SiLU,
  enc.set_buffer(0, ce); enc.set_buffer(1, ce); enc.set_buffer(2, ce);
  enc.set_constant(3, (int)yxw);                   // hoisted (shared by all
  enc.dispatch({(unsigned)yxw, 1, 1}, {256, 1, 1});   // three res blocks)

  gemm_(enc, yx, 0, _dec_x_embed, xm, 0, (int)M);
  enc.set_function(_fn_add_rows_mod);               // + the DCT/bias constant
  enc.set_buffer(0, xm); enc.set_buffer(1, _dec_x_const);
  enc.set_constant(2, X); enc.set_constant(3, PP);
  enc.set_constant(4, (unsigned)(M * X));
  enc.dispatch({(unsigned)X, (unsigned)M, 1}, {256, 1, 1});
  gemm_(enc, xm, 0, _dec_input_proj, hm, 0, (int)M);
  enc.set_function(_fn_copy);
  enc.set_buffer(0, hm); enc.set_buffer(1, xm);
  enc.set_constant(2, 0); enc.set_constant(3, (int)(M * X));
  enc.dispatch({(unsigned)(M * X), 1, 1}, {256, 1, 1});

  for (auto& r : _dec_mlp) {
    gemm_(enc, ce, 0, r.adaln, mod, 0, (int)M);
    enc.set_function(_fn_ln_mod_rows);
    enc.set_buffer(0, xm); enc.set_buffer(1, r.lnw); enc.set_buffer(2, r.lnb);
    enc.set_buffer(3, mod); enc.set_buffer(4, hm);
    enc.set_constant(5, X); enc.set_constant(6, _cfg.norm_eps);
    enc.set_constant(7, (unsigned)M);
    enc.dispatch({(unsigned)M, 1, 1}, {256, 1, 1});
    gemm_(enc, hm, 0, r.fc1, h2, 0, (int)M);
    enc.set_function(_fn_swish);
    enc.set_buffer(0, h2); enc.set_buffer(1, h2); enc.set_buffer(2, h2);
    enc.set_constant(3, (int)(M * X));
    enc.dispatch({(unsigned)(M * X), 1, 1}, {256, 1, 1});
    gemm_(enc, h2, 0, r.fc2, hm, 0, (int)M);
    enc.set_function(_fn_gated_rows);
    enc.set_buffer(0, xm); enc.set_buffer(1, mod); enc.set_buffer(2, hm);
    enc.set_constant(3, X); enc.set_constant(4, (unsigned)M);
    enc.dispatch({(unsigned)M, 1, 1}, {256, 1, 1});
  }
  enc.set_function(_fn_rms);                       // final_layer.norm
  enc.set_buffer(0, xm); enc.set_buffer(1, _dec_final_n); enc.set_buffer(2, hm);
  enc.set_constant(3, X); enc.set_constant(4, _cfg.norm_eps);
  enc.dispatch({256, (unsigned)M, 1}, {256, 1, 1});
  gemm_(enc, hm, 0, _dec_final_lin, outm, 0, (int)M);
  }
  std::string gpu_err;
  if (!stream.commit().wait_ok(&gpu_err)) {
    return fail(gpu_err.empty() ? "MageVAE: decode GPU commit failed"
                                : gpu_err.c_str());
  }

  // Fold the per-patch pixels back into a channel-first image (host). Patch
  // p = (y, x) row-major, intra-patch pixel q = (ky, kx) row-major.
  const int OC = _dec_final_lin.cout;              // 3
  const int H = h * P, W = w * P;
  SharedBuffer img = _mc->make_shared_buffer((std::size_t)OC * H * W * 2);
  if (img.empty()) { return fail("MageVAE: image alloc failed"); }
  {
    const auto* src = static_cast<const _Float16*>(outm.contents());
    auto* dst = static_cast<_Float16*>(img.contents());
    const std::size_t plane = (std::size_t)H * W;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const std::size_t p = (std::size_t)y * w + x;
        for (int ky = 0; ky < P; ++ky) {
          for (int kx = 0; kx < P; ++kx) {
            const _Float16* v = src + (p * PP + (std::size_t)ky * P + kx) * OC;
            const std::size_t o = (std::size_t)(y * P + ky) * W + x * P + kx;
            for (int c = 0; c < OC; ++c) {
              dst[(std::size_t)c * plane + o] = v[c];
            }
          }
        }
      }
    }
  }
  return img;
}

}  // namespace genai
}  // namespace vpipe

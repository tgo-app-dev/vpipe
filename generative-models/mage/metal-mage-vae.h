#ifndef GENERATIVE_MODELS_MAGE_METAL_MAGE_VAE_H
#define GENERATIVE_MODELS_MAGE_METAL_MAGE_VAE_H

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <memory>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

class MetalLlamaWeights;   // fwd

// MageVAE (microsoft/Mage-Flow), run in f16 on the metal-compute backend.
//
// NOT an AutoencoderKL. It is a "symmetric one-step diffusion codec": both
// halves are DiCo conv trunks (1x1 conv -> depthwise 3x3 -> GELU -> channel
// attention -> 1x1, then a 1x1 FFN at mlp_ratio 4) evaluated at a FIXED
// diffusion time t=0, mapping RGB <-> a 128-channel, 16x-downsampled latent.
//
// Three properties of the t=0 inference point collapse most of the nominal
// work, and this implementation bakes all three in at load:
//  * adaLN is CONSTANT. Every DiCo block's modulation is
//    Linear(SiLU(t_embedder(0))), independent of the input, so all 21+21
//    blocks get a fixed [6*hidden] vector computed once on the host
//    (`fold_adaln_`) instead of a per-forward MLP.
//  * The diffusion inputs are ZERO. The encoder's z_t and the decoder's
//    noise are both zero tensors, so the encoder's z_proj(0) collapses to
//    its bias and the decoder's s_embedder.proj1 (bias=False) contributes
//    nothing -- its proj2 weight is sliced at load to the `cond` columns.
//  * The posterior is NOT sampled (`sample_posterior:false`), so encode
//    returns the mean half of proj_out and is fully deterministic.
//
// Layout mirrors the Krea-2 / FLUX.2 VAEs: a channel-last [H*W, C] interior
// so 1x1 convs are dense GEMMs, LayerNorm2d is a row-wise norm, and the CoD
// decoder's 3x3 convs are im2col -> GEMM. Channel-first <-> channel-last
// conversion happens on the host at the call boundaries only.
class MetalMageVae {
 public:
  struct Config {
    int latent_channels = 128;   // z_ch
    int patch            = 16;   // downsample_factor (also the patch size)
    int enc_head_dim     = 768;  // encoder head_blocks width
    int hidden           = 384;  // DiCo trunk width (encoder and decoder)
    int n_head_blocks    = 2;    // encoder pre-trunk blocks
    int n_blocks         = 21;   // DiCo blocks per trunk (num_cond_blocks)
    int x_dim            = 32;   // decoder per-pixel MLP width (hidden_size_x)
    int n_mlp_res        = 3;    // dec_net res blocks (num_blocks - cond)
    int mlp_ratio        = 4;
    int attn_patch       = 32;   // CoD decoder AttnBlock tile
    int norm_groups      = 32;
    int max_freqs        = 8;    // NerfEmbedder DCT basis
    float norm_eps       = 1e-6f;
  };

  // `with_encoder` also loads the `student.dconv_encoder.*` half (the edit
  // flow needs it to encode reference images); a decode-only pipeline leaves
  // it false to save the ~67M params.
  static std::unique_ptr<MetalMageVae>
  load(const std::string& model_dir, metal_compute::MetalCompute* mc,
       const Config& cfg, bool with_encoder = true);

  ~MetalMageVae();

  // Encode RGB [3, H, W] channel-first, in [-1, 1], into the latent
  // [latent_channels, H/patch, W/patch] channel-first (the posterior MEAN --
  // the logvar half of proj_out is discarded). H and W must be multiples of
  // `patch`. Empty on failure / no encoder.
  metal_compute::SharedBuffer
  encode(const metal_compute::SharedBuffer& img, int H, int W,
         std::string* err = nullptr);

  // Decode a latent [latent_channels, h, w] channel-first into RGB
  // [3, h*patch, w*patch] channel-first. NOT clamped -- the reference clamps
  // to [-1, 1] at the PIL conversion, which the caller owns.
  metal_compute::SharedBuffer
  decode(const metal_compute::SharedBuffer& z, int h, int w,
         std::string* err = nullptr);

  bool has_encoder() const { return _has_encoder; }
  const Config& config() const { return _cfg; }

  // True when the M5 matrix-core matmul2d (NAX) path is active for the 1x1 /
  // conv GEMMs. For the mma-vs-steel A/B test to assert the path engaged.
  bool uses_mma2() const { return _use_mma2; }

 private:
  MetalMageVae() = default;

  // A conv/linear weight as a dense-GEMM [Cout, K] (+ bias [Cout]):
  //   1x1 / Linear : K = Cin
  //   3x3          : K = 9*Cin, flattened (ky,kx,cin) to pair with im2col
  //   patch p      : K = p*p*Cin, flattened (cin,ky,kx) to pair with the
  //                  host patchify (PyTorch Conv2d weight order)
  struct Conv {
    metal_compute::SharedBuffer w, b;
    int cin = 0, cout = 0, k = 0;
    bool empty() const { return w.empty(); }
  };
  // Depthwise 3x3 (groups == C): weight [C, 9], bias [C]. No GEMM form --
  // depthwise_conv2d_3x3_hwc reads the 9 neighbours per (pixel, channel).
  struct DwConv {
    metal_compute::SharedBuffer w, b;
    int c = 0;
  };
  // The shared DiCo body. `mod` is the folded [6*hidden] adaLN constant for
  // the trunk blocks; the encoder's head_blocks have no adaLN and instead
  // carry affine LayerNorm2d params (n1w/n1b, n2w/n2b).
  struct DiCoBlock {
    metal_compute::SharedBuffer mod;              // [6*C], trunk blocks only
    metal_compute::SharedBuffer n1w, n1b, n2w, n2b;   // head_blocks only
    Conv c1, c3, c4, c5, ca;
    DwConv c2;
    int c = 0;
    bool has_mod = false;
  };
  // CoD decoder ResnetBlock (GroupNorm-32 + swish + 3x3, in == out).
  struct ResBlock {
    metal_compute::SharedBuffer n1w, n1b, n2w, n2b;
    Conv c1, c2;
  };
  // CoD decoder AttnBlock: GroupNorm + 1x1 q/k/v/proj, attention computed
  // over non-overlapping `attn_patch` x `attn_patch` tiles (replicate-padded
  // to a whole number of tiles), single head, head_dim = the full channel
  // count.
  struct AttnBlock {
    metal_compute::SharedBuffer nw, nb;
    Conv q, k, v, proj;
  };
  // dec_net per-pixel residual MLP (width x_dim), adaLN-conditioned on the
  // per-patch latent -- NOT constant-folded (it varies per position).
  struct MlpResBlock {
    metal_compute::SharedBuffer lnw, lnb;   // in_ln (affine LayerNorm)
    Conv adaln;                             // -> 3*x_dim
    Conv fc1, fc2;
  };

  // ---- load helpers ----
  Conv load_linear_(const MetalLlamaWeights& w, const std::string& nm);
  Conv load_conv3x3_(const MetalLlamaWeights& w, const std::string& nm);
  Conv load_patch_conv_(const MetalLlamaWeights& w, const std::string& nm);
  DwConv load_dwconv_(const MetalLlamaWeights& w, const std::string& nm);
  metal_compute::SharedBuffer load_vec_(const MetalLlamaWeights& w,
                                        const std::string& nm);
  bool load_dico_(const MetalLlamaWeights& w, const std::string& pre,
                  DiCoBlock& b, int c, bool with_mod, bool with_norm_affine);
  bool load_resblock_(const MetalLlamaWeights& w, const std::string& pre,
                      ResBlock& rb);
  bool load_attn_(const MetalLlamaWeights& w, const std::string& pre,
                  AttnBlock& a);
  bool load_encoder_(const MetalLlamaWeights& w);
  bool load_decoder_(const MetalLlamaWeights& w);
  // Host-fold the t=0 adaLN constant for one trunk: returns [6*hidden].
  // `t_pre` is the TimestepEmbedder prefix, `adaln` the block's modulation
  // Linear. Cross-checked against the reference's own folded buffers.
  std::vector<float> fold_adaln_(const MetalLlamaWeights& w,
                                 const std::string& t_pre,
                                 const std::string& adaln_pre) const;
  // Host-compute t_embedder(0) -> [hidden] (shared by every block in a trunk).
  std::vector<float> t_embed_zero_(const MetalLlamaWeights& w,
                                   const std::string& t_pre) const;

  // ---- forward helpers ----
  // y[M,N] = x[M,K] @ w[N,K]^T (+ bias[N]) at element offsets, matmul2d on M5
  // (row-split under the high-M corruption threshold) else steel.
  void gemm_(metal_compute::ComputeEncoder& enc,
             const metal_compute::SharedBuffer& x, std::size_t xe,
             const Conv& c, const metal_compute::SharedBuffer& y,
             std::size_t ye, int M);
  // One DiCo block in place on `x` [HW, C], using `tmp`/`tmp2` scratch.
  void dico_(metal_compute::ComputeEncoder& enc, const DiCoBlock& b,
             const metal_compute::SharedBuffer& x, int H, int W,
             const metal_compute::SharedBuffer& tmp,
             const metal_compute::SharedBuffer& tmp2,
             const metal_compute::SharedBuffer& pool);
  // im2col 3x3 -> GEMM (pad 1, stride 1) into `out`.
  void conv3x3_(metal_compute::ComputeEncoder& enc,
                const metal_compute::SharedBuffer& in, const Conv& c,
                const metal_compute::SharedBuffer& out, int H, int W,
                const metal_compute::SharedBuffer& col);
  void groupnorm_(metal_compute::ComputeEncoder& enc,
                  const metal_compute::SharedBuffer& x,
                  const metal_compute::SharedBuffer& g,
                  const metal_compute::SharedBuffer& b,
                  const metal_compute::SharedBuffer& out, int HW, int C);
  void resblock_(metal_compute::ComputeEncoder& enc, const ResBlock& rb,
                 const metal_compute::SharedBuffer& x, int H, int W,
                 const metal_compute::SharedBuffer& t1,
                 const metal_compute::SharedBuffer& t2,
                 const metal_compute::SharedBuffer& col);
  void attnblock_(metal_compute::ComputeEncoder& enc, const AttnBlock& a,
                  const metal_compute::SharedBuffer& x, int H, int W,
                  const metal_compute::SharedBuffer& t1,
                  const metal_compute::SharedBuffer& qkv);

  metal_compute::MetalCompute* _mc = nullptr;
  Config _cfg;

  // ---- encoder (student.dconv_encoder.*) ----
  bool _has_encoder = false;
  Conv _enc_patch;                       // patch_cond_embed (16x16 s16)
  std::vector<DiCoBlock> _enc_head;      // head_blocks (affine LN, no adaLN)
  Conv _enc_proj_down;                   // 768 -> 384
  Conv _enc_fuse;                        // 768 -> 384, applied to [cond|zc]
  metal_compute::SharedBuffer _enc_zc;   // z_proj(0) = its bias, [hidden]
  std::vector<DiCoBlock> _enc_blocks;    // 21
  metal_compute::SharedBuffer _enc_no_w, _enc_no_b;   // norm_out (affine)
  Conv _enc_proj_out;                    // 384 -> 2*latent (mean | logvar)

  // ---- decoder (pipeline.*) ----
  Conv _dec_conv_in;                     // y_embedder.decoder.conv_in 3x3
  ResBlock _dec_res0, _dec_res1, _dec_res2;
  AttnBlock _dec_attn0, _dec_attn1;
  metal_compute::SharedBuffer _dec_no_w, _dec_no_b;   // norm_out GroupNorm
  Conv _dec_conv_out;                    // 3x3 384 -> 384
  Conv _dec_s_embed;                     // s_embedder.proj2, cond cols only
  std::vector<DiCoBlock> _dec_blocks;    // 21
  Conv _dec_y_x;                         // y_embedder_x, rows permuted to (q,f)
  Conv _dec_cond_embed;                  // dec_net.cond_embed -> [q, x_dim]
  Conv _dec_x_embed;                     // x_embedder cols 3..35 only
  metal_compute::SharedBuffer _dec_x_const;   // [p*p, x_dim] DCT+bias constant
  Conv _dec_input_proj;
  std::vector<MlpResBlock> _dec_mlp;
  metal_compute::SharedBuffer _dec_final_n;   // final_layer RMSNorm [x_dim]
  Conv _dec_final_lin;                        // x_dim -> 3

  // Libraries + kernels.
  metal_compute::ComputeLibrary _lib_gemm, _lib_elt, _lib_rms, _lib_sdpa;
  metal_compute::ComputeFunction _fn_gemm_bias, _fn_rms, _fn_gelu, _fn_swish,
      _fn_residual, _fn_im2col, _fn_ln_plain, _fn_ln_affine, _fn_ln_mod,
      _fn_gated, _fn_groupnorm, _fn_dw3x3, _fn_col_mean, _fn_mul_sig,
      _fn_bias_add, _fn_sdpa, _fn_copy,
      // decoder-only: tiled attention + the per-pixel MLP head
      _fn_tile_gather, _fn_tile_scatter, _fn_ln_mod_rows, _fn_gated_rows,
      _fn_add_rows_mod;
  // M5 matrix-core dense GEMM, as in the Krea-2 / FLUX.2 VAEs.
  metal_compute::ComputeLibrary _lib_dense_mma;
  metal_compute::ComputeFunction _fn_dense_mma, _fn_dense_mma_deep;
  bool _use_mma2 = false;
  int  _mma_min_m = 64;
  int  _mma_min_n = 16;
  // The per-pixel MLP head runs GEMMs at K = N = x_dim (32) over a huge M.
  // A 128-wide matmul2d tile wastes almost all of itself on those shapes, so
  // route anything shallower than this to steel (the decoder head only --
  // every encoder/trunk GEMM has K >= 384).
  int  _mma_min_k = 64;
  int  _mma_max_m = 1 << 19;   // matmul2d high-M corruption guard
};

}  // namespace genai
}  // namespace vpipe

#endif

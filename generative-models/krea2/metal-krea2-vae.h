#ifndef GENERATIVE_MODELS_KREA2_METAL_KREA2_VAE_H
#define GENERATIVE_MODELS_KREA2_METAL_KREA2_VAE_H

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <memory>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

// Qwen-Image VAE decoder (AutoencoderKLQwenImage), run in f16 on the metal-
// compute backend. The full model is a 3D causal-conv video VAE, but a still
// image is a single-frame latent -- and for one frame every QwenImageCausal
// Conv3d left-pads two zero frames in time (empty feat_cache), so only the
// kt=2 temporal slice of each 3x3x3 weight contributes and every upsample3d
// time_conv is skipped (marked "Rep"). The decode therefore collapses to a
// pure 2D conv net, run here in a channel-last [H*W, C] layout so RMS-norm,
// 1x1 conv, SiLU and the mid-block attention reuse the existing LM kernels
// and 3x3 convs become im2col -> dense_gemm.
//
// Handles the text-to-image `krea2` model's VAE stage: an unpacked,
// un-whitened latent [z_dim, H/8, W/8] -> RGB [3, H, W] in [-1, 1].
class MetalKrea2Vae {
 public:
  struct Config {
    int base_dim       = 96;
    int z_dim          = 16;
    int dim_mult[4]    = {1, 2, 4, 4};
    int num_res_blocks = 2;
    // Per-channel latent statistics (config latents_mean / latents_std) used
    // by unwhiten(); the pipeline applies latents = latents * std + mean.
    std::vector<float> latents_mean;
    std::vector<float> latents_std;
  };

  // `with_encoder` also loads the encoder weights (for the img2img encode
  // path); the decode-only path (text-to-image) leaves it false to save RAM.
  static std::unique_ptr<MetalKrea2Vae>
  load(const std::string& model_dir, metal_compute::MetalCompute* mc,
       const Config& cfg, bool with_encoder = false);

  // Decode a single-frame latent `z` laid out channel-first [z_dim, h8, w8]
  // (already unpacked + un-whitened) into an RGB image [3, h8*8, w8*8],
  // channel-first, clamped to [-1, 1]. Empty on failure. When `err` is
  // non-null it receives a reason on failure -- distinguishing an
  // over-commit (a preflight budget shortfall, a failed buffer allocation,
  // or a GPU out-of-memory / page-fault detected at commit) from other
  // errors, so the caller can surface it instead of emitting a corrupt image.
  metal_compute::SharedBuffer
  decode(const metal_compute::SharedBuffer& z, int h8, int w8,
         std::string* err = nullptr);

  // Conservative estimate of the peak GPU memory (bytes) a decode() at the
  // given latent size needs: the reused im2col scratch plus the held
  // intermediate activations. For a preflight memory_budget() check.
  std::size_t decode_peak_bytes(int h8, int w8) const noexcept;

  // Encode an RGB image [3, H, W] (channel-first, in [-1,1]) into the WHITENED
  // latent [z_dim, H/8, W/8] (channel-first) -- the posterior mode (mean),
  // whitened (x-mean)/std -- ready for img2img. Empty on failure / no encoder.
  metal_compute::SharedBuffer
  encode(const metal_compute::SharedBuffer& img, int H, int W);
  bool has_encoder() const { return _has_encoder; }

  // In-place per-channel un-whiten of a channel-first latent [z_dim, h8, w8]:
  // z[c] = z[c] * latents_std[c] + latents_mean[c]. Returns a fresh buffer.
  metal_compute::SharedBuffer
  unwhiten(const metal_compute::SharedBuffer& z, int h8, int w8);

  const Config& config() const { return _cfg; }

 private:
  MetalKrea2Vae() = default;

  // A conv weight is stored as f16 [Cout, K] ready for dense_gemm (W[N,K]):
  //  - 1x1 conv:  K = Cin  (bias [Cout]).
  //  - 3x3 conv:  K = 9*Cin, flattened (ky,kx,cin) to pair with im2col.
  struct Conv {
    metal_compute::SharedBuffer w, b;
    // HWIO twin ([3,3,Cin,Cout], out-channel fastest) for the NAX hardware
    // convolution2d; built at load when _use_hwconv.
    metal_compute::SharedBuffer whwio;
    int cin = 0, cout = 0, k = 0;   // k = Cin (1x1) or 9*Cin (3x3)
  };
  struct ResBlock {
    metal_compute::SharedBuffer n1g, n2g;   // RMS gammas (standard, no +1)
    Conv c1, c2, shortcut;
    bool has_short = false;
    int cin = 0, cout = 0;
  };
  struct Attn {
    metal_compute::SharedBuffer ng;
    Conv q, k, v, proj;             // 1x1 convs (to_qkv split into q/k/v)
    int dim = 0;
  };
  struct UpBlock {
    std::vector<ResBlock> resnets;
    bool has_up = false;
    Conv up;                        // resample.1: 3x3 conv (dim -> dim/2)
    int up_dim = 0;                 // channels feeding the upsample conv
  };
  struct DownStage {                // encoder down_blocks stage
    std::vector<ResBlock> resnets;  // num_res_blocks per stage
    bool has_down = false;
    Conv down;                      // resample.1: 3x3 STRIDE-2 conv (dim->dim)
  };

  Conv load_conv3x3_(const class MetalLlamaWeights& w, const std::string& nm,
                     bool from3d);
  Conv load_conv1x1_(const class MetalLlamaWeights& w, const std::string& nm);
  metal_compute::SharedBuffer load_vec_(const class MetalLlamaWeights& w,
                                        const std::string& nm);
  bool load_resblock_(const class MetalLlamaWeights& w, const std::string& pre,
                      ResBlock& rb, int cin, int cout);

  metal_compute::MetalCompute* _mc = nullptr;
  Config _cfg;

  Conv _post_quant;                 // 1x1 conv z_dim -> z_dim
  Conv _conv_in;                    // 3x3 conv z_dim -> dims[0]
  ResBlock _mid_res0, _mid_res1;
  Attn _mid_attn;
  std::vector<UpBlock> _up_blocks;  // 4
  metal_compute::SharedBuffer _norm_out_g;
  Conv _conv_out;                   // 3x3 conv base_dim -> 3

  // ---- encoder (loaded only when with_encoder) ----
  bool _has_encoder = false;
  Conv _enc_conv_in;                // 3x3 conv 3 -> base
  std::vector<DownStage> _enc_down; // 4 stages (3 with a stride-2 downsample)
  ResBlock _enc_mid_res0, _enc_mid_res1;
  Attn _enc_mid_attn;
  metal_compute::SharedBuffer _enc_norm_out_g;
  Conv _enc_conv_out;               // 3x3 conv base*dim_mult[-1] -> z_dim*2
  Conv _quant_conv;                 // 1x1 conv z_dim*2 -> z_dim*2

  bool load_encoder_(const class MetalLlamaWeights& w);

  // Shared conv/1x1 GEMM y[M,N] = x[M,K] @ w[N,K]^T (+ bias[N]). On M5 (matrix
  // cores) this rides the hardware matmul2d dense GEMM when M is tall enough to
  // amortize the tiled kernel; otherwise the steel dense_gemm_bias. Used by
  // both decode() and encode() (their conv3x3/conv1x1 helpers).
  void conv_gemm_bias_(metal_compute::ComputeEncoder& enc,
                       const metal_compute::SharedBuffer& x,
                       const metal_compute::SharedBuffer& w,
                       const metal_compute::SharedBuffer& b,
                       const metal_compute::SharedBuffer& y, int M, int N,
                       int K);

  // Fused 3x3 conv2d on the matrix units (im2col staged in threadgroup memory ->
  // matmul2d): out[OH*OW, Cout] = conv(in[H*W, Cin], w[Cout, 9*Cin]) + bias, pad
  // 1, given stride. Skips the DRAM im2col scratch of conv_gemm_bias's path.
  // Only used when _use_conv2d (matrix cores + kernels present); returns false
  // so callers fall back to im2col otherwise.
  bool conv2d_mma_(metal_compute::ComputeEncoder& enc,
                   const metal_compute::SharedBuffer& in,
                   const metal_compute::SharedBuffer& w,
                   const metal_compute::SharedBuffer& b,
                   const metal_compute::SharedBuffer& out, int H, int W, int Cin,
                   int Cout, int OH, int OW, int stride);

  // NAX hardware convolution2d (M5+, probe-established semantics): the op
  // reads the full NHWC activation itself (zero-filled pad-1 halo included)
  // -- no im2col scratch or DRAM round-trip; ~2-3x the im2col+matmul2d path
  // at decoder shapes, bit-identical. Needs whole 8x8 dest tiles +
  // 64-channel tiles; bias folds via bias_add_rows. Tried FIRST by the conv
  // lambdas; false -> the callers keep their existing routes.
  // VPIPE_VAE_NO_HWCONV=1 opts out (shared with the flux2 VAE).
  bool conv3x3_hw_(metal_compute::ComputeEncoder& enc,
                   const metal_compute::SharedBuffer& in, const Conv& c,
                   const metal_compute::SharedBuffer& out, int H, int W,
                   int stride);
  metal_compute::ComputeLibrary _lib_convhw;
  metal_compute::ComputeFunction _fn_conv_hw_s1, _fn_conv_hw_s2;
  bool _use_hwconv = false;

  // Libraries + kernel functions (all reused from the LM path except the VAE
  // kernels im2col_hwc_3x3{,_s2} / upsample_nearest2x_hwc in llm_elementwise).
  metal_compute::ComputeLibrary _lib_gemm, _lib_elt, _lib_rms, _lib_sdpa;
  metal_compute::ComputeFunction _fn_gemm_bias, _fn_rms, _fn_mul_sigmoid,
      _fn_residual, _fn_clamp, _fn_sdpa, _fn_im2col, _fn_im2col_s2, _fn_upsample;
  // Matrix-core FULL flash-attention for the mid-block self-attention (D = the
  // mid channel dim). Replaces the scalar O(N^2) sdpa_full_f16 that dominates
  // decode at high res. Loaded best-effort (matrix cores only, D in {384,512});
  // _fn_sdpa stays the fallback. VPIPE_KREA2_NO_MMA_ATTN forces the scalar path.
  metal_compute::ComputeLibrary _lib_sdpa_mma;
  metal_compute::ComputeFunction _fn_sdpa_full_mma;   // matmul2d (M5)
  // simdgroup_matrix FULL flash (sdpa_full_mma_f16, D%64==0 && D<=512): the
  // non-matmul2d flash used on pre-M5 GPUs (emulated matmul2d is slow there);
  // from _lib_sdpa. Preferred over _fn_sdpa_full_mma when no matrix cores.
  metal_compute::ComputeFunction _fn_sdpa_full_smm;
  bool _use_attn_mma2 = false;   // prefer matmul2d flash (true on M5 only)

  // M5-only matrix-core dense GEMM (matmul2d/NAX): the conv/1x1 GEMMs run on the
  // hardware matrix units instead of steel simdgroup_matrix. Gated on
  // supports_matrix_cores() (Apple10+) so M4/older never load it and keep the
  // steel path (byte-identical). VPIPE_KREA2_NO_MMA2=1 forces steel on M5 (A/B).
  // Bias is applied as a separate bias_add_rows pass (matmul2d has no bias fold).
  metal_compute::ComputeLibrary _lib_dense_mma;
  metal_compute::ComputeFunction _fn_dense_mma, _fn_dense_mma_deep, _fn_bias_add;
  bool _use_mma2 = false;
  int  _mma_min_m = 64;   // matmul2d only wins once M amortizes the 128-row tile
  // Min output cols for the matmul2d path: a tiny N (e.g. the 3-channel
  // conv_out) wastes the 128/256-wide tile, so route N < _mma_min_n to steel.
  // VPIPE_KREA2_VAE_MMA_MIN_N overrides (A/B; 0 = always matmul2d).
  int  _mma_min_n = 16;
  // Max rows per matmul2d dispatch: the MPP matmul2d op corrupts output rows
  // past M ~= 2^19 (a >=1024px decode has M = H*W = 2^20 and went grey from
  // GEMM row 2^19; 512px decodes, M = 2^18, stay clean). conv_gemm_bias_ splits
  // a taller GEMM into row-chunks of this size (each its own dispatch over a
  // sub-range of x/y) so every chunk stays under the limit. VPIPE_KREA2_VAE_
  // MMA_MAX_M overrides (0 = no split -- single dispatch, reproduces the bug).
  int  _mma_max_m = 1 << 19;

  // M5-only fused 3x3 conv2d (matmul2d over threadgroup-staged im2col): runs the
  // VAE 3x3 convs without materializing the [H*W, 9*Cin] im2col scratch in DRAM.
  // Correct (bit-identical) but ~1.7x slower than im2col+matmul2d on M5's high
  // UMA bandwidth, so OPT-IN via VPIPE_KREA2_CONV2D=1 (default keeps im2col).
  metal_compute::ComputeLibrary _lib_conv2d;
  metal_compute::ComputeFunction _fn_conv2d_s1, _fn_conv2d_s2;
  bool _use_conv2d = false;
};

}  // namespace genai
}  // namespace vpipe

#endif

#ifndef GENERATIVE_MODELS_FLUX2_METAL_FLUX2_VAE_H
#define GENERATIVE_MODELS_FLUX2_METAL_FLUX2_VAE_H

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <memory>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

class MetalLlamaWeights;   // fwd
class WeightSet;           // generative-models/weight-set.h

// FLUX.2 VAE (AutoencoderKLFlux2), run in f16 on the metal-compute backend. A
// standard diffusers 2D AutoencoderKL (GroupNorm-32 + SiLU ResnetBlock2D conv
// net, mid-block single-head attention, 8x spatial via 3 down/up samples) with
// two FLUX.2-specific wrappers on the latent:
//   * patch_size [2,2]: the encoder's latent is pixel-UNSHUFFLED (2x2) to
//     prod(patch)*latent = 4*32 = 128 channels at H/16 -- so the latent the DiT
//     consumes is [128, H/16, W/16] (the DiT has patch_size 1, in_channels 128).
//   * a BatchNorm2d over those 128 channels (running stats -> a per-channel
//     affine) is the whitening; decode inverts it before unpatchifying.
//
// It ALSO serves the PLAIN diffusers `AutoencoderKL` -- the FLUX.1 VAE that
// Boogu-Image uses -- which is the same conv trunk with the two wrappers turned
// off: patch 1 (the latent IS [16, H/8, W/8]), no quant/post-quant 1x1 convs,
// and a SCALAR whitening (z = (x - shift_factor) * scaling_factor) instead of
// the BatchNorm. Both come straight off vae/config.json, so one code path
// serves both: patch 1 makes the pixel-(un)shuffle an identity and the scalar
// factors fold into the same per-channel affine the BatchNorm produces.
//
// Channel-last [H*W, C] activation layout (as Krea-2's VAE) so 3x3 convs are
// im2col -> dense_gemm and 1x1/attention reuse the LM/vision kernels.
//
// Bring-up scope: correct-first (dense f16, im2col GEMM convs). The patch/bn
// latent flow + downsample padding convention are a best reading of
// AutoencoderKLFlux2; VERIFY against a diffusers golden.
class MetalFlux2Vae {
 public:
  struct Config {
    int in_channels     = 3;
    int latent_channels = 32;
    int block_out[4]    = {128, 256, 512, 512};
    int layers_per_block = 2;
    int norm_groups     = 32;
    int patch           = 2;        // patch_size (per spatial dim); 1 = plain
    float norm_eps      = 1e-6f;    // GroupNorm eps
    // The plain AutoencoderKL can be built without the 1x1 moment convs
    // (use_quant_conv / use_post_quant_conv false -- the FLUX.1 VAE Boogu
    // uses); then encode/decode skip them entirely.
    bool use_quant_conv      = true;
    bool use_post_quant_conv = true;
    // SCALAR latent whitening z = (x - shift) * scale. Used only when the
    // checkpoint has no BatchNorm running stats (scale 0 = "not configured",
    // which leaves the whitening to the bn tensors).
    float shift_factor   = 0.0f;
    float scaling_factor = 0.0f;
    // Channels the DiT sees = prod(patch)*latent_channels.
    int dit_channels() const { return patch * patch * latent_channels; }
  };

  // Prefer the WeightSet overload: the set is the manager's shared,
  // reference-counted view of the checkpoint, so a second VAE over the
  // same directory reuses these tensors instead of loading its own copy.
  // The dir overload opens a PRIVATE set (tests, and callers with no
  // session to ask).
  static std::unique_ptr<MetalFlux2Vae>
  load(const std::string& model_dir, metal_compute::MetalCompute* mc,
       const Config& cfg, bool with_encoder = false);

  static std::unique_ptr<MetalFlux2Vae>
  load(std::shared_ptr<WeightSet> ws, metal_compute::MetalCompute* mc,
       const Config& cfg, bool with_encoder = false);

  // Materialise the encoder half now, if it is not already loaded. The
  // point of deferring it: a graph that never receives a reference image
  // never calls this, so the encoder's weights are never read. Cheap and
  // idempotent once loaded. False if the encoder cannot be loaded.
  bool ensure_encoder();

  // Decode a latent [dit_channels, h16, w16] (the DiT-facing patchified+whitened
  // latent) into an RGB image [3, h16*8*patch, w16*8*patch], channel-first, in
  // [-1,1] (so 16x per cell at patch 2, 8x on the plain AutoencoderKL).
  // Empty on failure; *err set on an over-commit/allocation/GPU failure.
  metal_compute::SharedBuffer
  decode(const metal_compute::SharedBuffer& z, int h16, int w16,
         std::string* err = nullptr);

  std::size_t decode_peak_bytes(int h16, int w16) const noexcept;

  // Encode an RGB image [3, H, W] (channel-first, in [-1,1]) into the DiT-facing
  // latent [dit_channels, H/(8*patch), W/(8*patch)] (patchified + whitened
  // posterior mode).
  metal_compute::SharedBuffer
  encode(const metal_compute::SharedBuffer& img, int H, int W);
  bool has_encoder() const { return _has_encoder; }

  const Config& config() const { return _cfg; }

 private:
  MetalFlux2Vae() = default;

  // A conv weight stored as dense-gemm W[Cout, K]: 1x1 -> K=Cin; 3x3 -> K=9*Cin
  // flattened (ky,kx,cin) to pair with im2col_hwc_3x3. When the NAX hardware
  // conv is active (_use_hwconv), 3x3 weights ALSO carry an HWIO twin
  // ([3,3,Cin,Cout], out-channel fastest) -- the layout the convolution2d op
  // consumes.
  struct Conv {
    metal_compute::SharedBuffer w, b, whwio;
    int cin = 0, cout = 0, k = 0;
  };
  // GroupNorm affine (gamma/beta [C]).
  struct GNorm { metal_compute::SharedBuffer g, b; int c = 0; };
  struct ResBlock {
    GNorm n1, n2;
    Conv c1, c2, shortcut;
    bool has_short = false;
    int cin = 0, cout = 0;
  };
  struct Attn {
    GNorm n;
    Conv q, k, v, proj;   // separate 1x1 convs (to_q/k/v/to_out.0)
    int dim = 0;
  };
  struct UpBlock {
    std::vector<ResBlock> resnets;
    bool has_up = false;
    Conv up;              // upsamplers.0.conv (3x3, after nearest-2x)
    int up_dim = 0;
  };
  struct DownStage {
    std::vector<ResBlock> resnets;
    bool has_down = false;
    Conv down;            // downsamplers.0.conv (3x3 stride-2)
  };

  Conv load_conv3x3_(WeightSet& w, const std::string& nm);
  Conv load_conv1x1_(WeightSet& w, const std::string& nm);
  GNorm load_gnorm_(WeightSet& w, const std::string& nm);
  metal_compute::SharedBuffer load_vec_(WeightSet& w,
                                        const std::string& nm);
  bool load_resblock_(WeightSet& w, const std::string& pre,
                      ResBlock& rb, int cin, int cout);
  Attn load_attn_(WeightSet& w, const std::string& pre, int dim);
  bool load_encoder_(WeightSet& w);

  // The checkpoint these weights come from, held for this model's whole
  // life: the cached tensors are aliases into buffers it owns (and
  // mapped ones alias its mmap), so it has to outlive them.
  std::shared_ptr<WeightSet> _ws;
  // Part the loaders currently attribute their tensors to; "" is the
  // always-resident trunk, "encoder" the half release_part() can drop.
  std::string _part;

  metal_compute::MetalCompute* _mc = nullptr;
  Config _cfg;

  // Decoder path.
  Conv _post_quant;                 // post_quant_conv (1x1, latent->latent)
  Conv _conv_in;                    // decoder.conv_in (3x3, latent->block_out[-1])
  ResBlock _mid_res0, _mid_res1;
  Attn _mid_attn;
  std::vector<UpBlock> _up_blocks;
  GNorm _norm_out;                  // decoder.conv_norm_out
  Conv _conv_out;                   // decoder.conv_out (3x3 -> 3)

  // Latent whitening: BatchNorm2d over dit_channels folded to per-channel affine
  // bn(x) = a*x + b (a = gamma/sqrt(var+eps), b = beta - a*mean). Decode inverts
  // (x = (z - b) / a); encode applies (a*x + b).
  std::vector<float> _bn_a, _bn_b;

  // Encoder path (loaded only when with_encoder).
  bool _has_encoder = false;
  Conv _enc_conv_in;                // 3 -> block_out[0]
  std::vector<DownStage> _enc_down;
  ResBlock _enc_mid_res0, _enc_mid_res1;
  Attn _enc_mid_attn;
  GNorm _enc_norm_out;
  Conv _enc_conv_out;               // -> 2*latent (mean, logvar)
  Conv _quant_conv;                 // quant_conv (1x1, 2*latent -> 2*latent)

  // y_row0: write the M-row result into y starting at output row y_row0 (the
  // tiled im2col conv feeds one band at a time into out[r0:r0+M]); 0 = at y[0].
  void conv_gemm_bias_(metal_compute::ComputeEncoder& enc,
                       const metal_compute::SharedBuffer& x,
                       const metal_compute::SharedBuffer& w,
                       const metal_compute::SharedBuffer& b,
                       const metal_compute::SharedBuffer& y, int M, int N, int K,
                       int y_row0 = 0);

  // Row-tiled im2col 3x3 conv (stride 1 or 2) into a pre-alloc'd `out`: streams
  // output-row bands through `col` (bounded to `cap` ELEMS) instead of the full
  // [OH*OW, 9*cin] scratch. Each band's GEMM is one un-chunked dispatch under
  // the matmul2d M-corruption threshold (band <= _mma_max_m/2). Shared by
  // decode + encode; the caller does the hw-conv fast path first.
  void tiled_conv3x3_(metal_compute::ComputeEncoder& enc,
                      const metal_compute::SharedBuffer& in,
                      const metal_compute::SharedBuffer& out, int H, int W,
                      const Conv& c, int stride,
                      const metal_compute::SharedBuffer& col, std::size_t cap);

  metal_compute::ComputeLibrary _lib_gemm, _lib_elt, _lib_sdpa;
  metal_compute::ComputeFunction _fn_gemm_bias, _fn_groupnorm, _fn_mul_sigmoid,
      _fn_residual, _fn_clamp, _fn_sdpa, _fn_im2col, _fn_im2col_s2,
      _fn_im2col_tiled, _fn_im2col_s2_tiled, _fn_upsample, _fn_bias_add;
  // Matrix-core FULL flash-attention for the mid-block self-attention (D = the
  // mid channel dim block_out[-1]). Replaces the scalar O(N^2) sdpa_full_f16
  // that dominates decode at high res. Best-effort (matrix cores + D in
  // {384,512}); _fn_sdpa stays the fallback. VPIPE_FLUX2_NO_MMA_ATTN forces it.
  metal_compute::ComputeLibrary _lib_sdpa_mma;
  metal_compute::ComputeFunction _fn_sdpa_full_mma;   // matmul2d (M5)
  // simdgroup_matrix FULL flash (sdpa_full_mma_f16, D%64==0 && D<=512): the
  // non-matmul2d flash used on pre-M5 GPUs (emulated matmul2d is slow there);
  // from _lib_sdpa. Preferred over _fn_sdpa_full_mma when no matrix cores.
  metal_compute::ComputeFunction _fn_sdpa_full_smm;
  bool _use_attn_mma2 = false;   // prefer matmul2d flash (true on M5 only)

  // M5 matrix-core dense GEMM (matmul2d) for the conv/1x1 GEMMs, mirroring
  // the Krea-2 VAE. The VAE runs at large M (M = H*W pixels), so the tiled
  // matmul2d amortizes well; bias is folded by a separate bias_add_rows pass
  // (the mma kernel has no bias slot). Steel stays the fallback (small M /
  // non-matrix-core GPUs). VPIPE_FLUX2_NO_MMA2 A/B off (shared with the DiT).
  metal_compute::ComputeLibrary _lib_dense_mma;
  metal_compute::ComputeFunction _fn_dense_mma, _fn_dense_mma_deep;
  bool _use_mma2 = false;
  int  _mma_min_m = 64;
  // Min output cols for the matmul2d path: a tiny N (e.g. the 3-channel
  // conv_out) wastes the 128/256-wide tile, so route N < _mma_min_n to steel.
  // VPIPE_FLUX2_VAE_MMA_MIN_N overrides (A/B; 0 = always matmul2d).
  int  _mma_min_n = 16;
  // Max rows per matmul2d dispatch: the MPP matmul2d op corrupts output rows
  // past M ~= 2^19 (a >=1024px decode has M = H*W = 2^20). conv_gemm_bias_
  // splits a taller GEMM into row-chunks of this size (each its own dispatch
  // over a sub-range of x/y) so every chunk stays under the limit; the
  // dense_gemm_mma tensors are column-major so a contiguous r0*K / r0*N element
  // offset selects rows [r0, r0+mc). VPIPE_FLUX2_VAE_MMA_MAX_M overrides (0 = no
  // split). See the Krea-2 VAE conv_gemm_bias_ for the diagnosed root cause.
  int  _mma_max_m = 1 << 19;

  // NAX hardware convolution2d (M5+) for the 3x3 s1/s2 convs: the op reads
  // the full NHWC activation itself (zero-filled pad-1 halo included) -- no
  // im2col scratch, no DRAM round-trip; ~2-3x the im2col+matmul2d path at
  // decoder shapes, bit-identical. Gated on matrix cores; needs whole 8x8
  // dest tiles + 64-channel tiles (others keep im2col). Bias folds via
  // bias_add_rows like the mma path. VPIPE_VAE_NO_HWCONV=1 opts out (A/B).
  // conv3x3_hw_ returns false -- nothing encoded -- when not applicable.
  bool conv3x3_hw_(metal_compute::ComputeEncoder& enc,
                   const metal_compute::SharedBuffer& in, const Conv& c,
                   const metal_compute::SharedBuffer& out, int H, int W,
                   int stride);
  metal_compute::ComputeLibrary _lib_convhw;
  metal_compute::ComputeFunction _fn_conv_hw_s1, _fn_conv_hw_s2;
  bool _use_hwconv = false;
};

}  // namespace genai
}  // namespace vpipe

#endif

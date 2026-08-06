#ifndef GENERATIVE_MODELS_FLUX2_METAL_FLUX2_VAE_H
#define GENERATIVE_MODELS_FLUX2_METAL_FLUX2_VAE_H

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/shared/kernel-sets/vae-conv3x3-tune.h"
#include "generative-models/shared/kernel-sets/vae-mid-attn-tune.h"

#include <functional>
#include <map>
#include <utility>
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
  // Two-pass group norm. _fn_groupnorm runs one threadgroup per group, which
  // on a full-resolution VAE activation (~100M elements) is ~8K threads and a
  // few GB/s -- MEASURED ~120 ms for the single norm_out at 1024x768, ~26% of
  // a decode once every resblock's two norms are counted. These split the
  // reduction so both passes saturate: per-channel partials over row blocks,
  // a tiny per-group reduce, then a fully parallel apply. Falls back to
  // _fn_groupnorm when unavailable or under VPIPE_VAE_NO_FAST_GNORM.
  metal_compute::ComputeFunction _fn_gn_stats, _fn_gn_reduce, _fn_gn_apply;
  bool _fast_gnorm = false;
  // Direct 3x3 for a small output-channel count. The hardware conv needs
  // cout % 64 == 0, so every VAE's final conv (to 3, or to 2*latent) drops to
  // im2col AT FULL RESOLUTION and pays ~1.8 GB of materialization for a
  // ~5 GFLOP convolution -- MEASURED 764 ms for 128->3 at 1024x768. This
  // keeps the small output vector in registers and reads the activation
  // directly. VPIPE_VAE_NO_SMALL_COUT_CONV falls back to im2col.
  metal_compute::ComputeFunction _fn_conv_small_cout;
  // Above this the im2col+GEMM path is the better trade (the register
  // accumulators and the per-tap re-read stop paying).
  static constexpr int kSmallCoutMax = 32;
  // Which 3x3 fallback route this GPU prefers once the hardware conv and the
  // small-cout kernel have declined -- MEASURED at load, not branched on
  // supports_matrix_cores(). See vae-conv3x3-tune.h: the on-chip gather wins
  // outright on an M4 Pro and LOSES 1.24-1.28x on an M5, where the im2col arm
  // it replaces feeds matmul2d.
  // PER (cin, cout): the winner is not a property of the GPU alone. MEASURED
  // on M5 at a 256x256 probe -- 128->128 and 256->128 go to the on-chip gather
  // (1.10x), while 256->256, 512->256 and 512->512 go to im2col + matmul2d by
  // 1.41x, 1.43x and 1.95x. cout is what swings it: the gather re-reads the
  // activation once per output tile, so a wide cout pays for it repeatedly
  // while im2col materializes once and hands the GEMM a wide N. A single
  // global answer loses either way, which is why a whole-decode A/B read
  // "on-chip is 1.24x slower" while a 128-channel probe read the opposite.
  std::map<std::pair<int, int>, vae_conv3x3::Kind> _conv_pick;
  vae_conv3x3::Kind conv_route_(int cin, int cout) const;
  void autotune_conv3x3_(metal_compute::MetalCompute* mc);
  // LAZY, and only when the fallback will actually carry work. Where the
  // hardware conv is available every 3x3 in this decoder goes to it (all of
  // block_out is a multiple of 64), so the tune's answer is never read at
  // ordinary resolutions -- it starts mattering once conv3x3_hw_ declines,
  // which is cin*W*H past 2^31 (4096x4096 at cin=128) or a grid that is not a
  // multiple of 8. Tuning at load would spend ~0.5 s on every M5 load for a
  // result nothing consults. Called at the top of decode()/encode(), where the
  // resolution is finally known; the tuner memoizes, so it runs once.
  bool _conv_tuned = false;
  // VPIPE_VAE_DIRECT_CONV_MMA2 forces the on-chip gather for EVERY shape,
  // including the ones autotune_conv3x3_ does not tune (cout <= kSmallCoutMax
  // normally goes to conv3x3_small_cout_ instead). An override that only
  // reached the tuned shapes would leave the direct-conv A/B comparing im2col
  // against itself -- which it silently did once already.
  bool _conv_force_onchip = false;
  void maybe_tune_conv_(int H, int W);
  // Returns false when the shape is not a fit, leaving the caller on im2col.
  bool conv3x3_small_cout_(metal_compute::ComputeEncoder& enc,
                           const metal_compute::SharedBuffer& in,
                           const Conv& c,
                           const metal_compute::SharedBuffer& out,
                           int H, int W, int stride);
  // Reused across every norm in a decode: partials [NB][2*C] f32 and the
  // per-group [G][2] (mean, inv) both size off the LARGEST C in the model.
  metal_compute::SharedBuffer _gn_part, _gn_stats;
  static constexpr int kGnBlocks = 512;   // row blocks in the stats pass
  // One group norm over `rows` x `C` channel-last, into `out`.
  void group_norm_(metal_compute::ComputeEncoder& enc,
                   const metal_compute::SharedBuffer& in,
                   const metal_compute::SharedBuffer& gamma,
                   const metal_compute::SharedBuffer& beta,
                   const metal_compute::SharedBuffer& out,
                   int rows, int C, int G, float eps);
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
  // Wide-query-tile twin of the above (sdpa_full_mma2_dN_qBQ_f16). Identical
  // math and f32 accumulators; it keeps O in registers instead of threadgroup
  // memory, which is what lets BQ exceed 8 and so divides the K/V bandwidth
  // this attention is bound by. Preferred over _fn_sdpa_full_mma whenever it
  // loaded. VPIPE_FLUX2_VAE_ATTN_BQ picks the tile (8 = the narrow kernel).
  metal_compute::ComputeFunction _fn_sdpa_full_wide16, _fn_sdpa_full_wide32,
      _fn_sdpa_full_wide64;
  void load_wide_attn_(int mid_d);
  // Simdgroup count the kernel of query tile `bq` was instantiated with; the
  // dispatch has to match it exactly (matmul2d's execution scope is UB
  // otherwise). Mirrors the SAW_INST table in sdpa_mma.metal.
  static unsigned attn_threads_(int bq)
  {
    return (bq == 16 ? 4u : bq == 32 ? 8u : 16u) * 32u;
  }

  // The interchangeable mid-block attention kernels. All compute the same
  // D=384/512 single-head full attention and are cross-verified against each
  // other; they differ only in how they get there, and WHICH IS FASTEST IS A
  // PROPERTY OF THE GPU, not of the shape. On an M4 Pro the materialized
  // banded GEMM beat the flash kernel 9.1x; on an M5 the matrix-core flash
  // beats it 1.08-1.46x at every size measured (mid tokens 16k..66k). Neither
  // is "the" answer, and a query tile picked by a sweep on one machine is not
  // the tile another wants, so the choice is MEASURED at load rather than
  // predicted from supports_matrix_cores().
  using MidAttn = vae_mid_attn::Kind;
  MidAttn _attn_pick = MidAttn::kScalar;
  // True when this member's kernels loaded on this GPU.
  bool mid_attn_available_(MidAttn k) const;
  // Encode ONE mid attention with the chosen member. `alloc`/`release` supply
  // scratch: the decode's pool in the real path, plain buffers in the tuner --
  // which is what lets both share this code instead of the tuner re-deriving
  // the band loop it is supposed to be measuring.
  void encode_mid_attn_(metal_compute::ComputeEncoder& enc, MidAttn kind,
                        const metal_compute::SharedBuffer& q,
                        const metal_compute::SharedBuffer& k,
                        const metal_compute::SharedBuffer& v,
                        const metal_compute::SharedBuffer& att,
                        std::size_t hw, int C, float scale,
                        const std::function<metal_compute::SharedBuffer&(
                            std::size_t)>& alloc,
                        const std::function<void(
                            const metal_compute::SharedBuffer&)>& release);
  // Time every available member on a synthetic mid block and keep the winner
  // (vae_mid_attn::autotune). Falls back to the capability guess when the tune
  // cannot run or an override names a member outright.
  void autotune_mid_attn_(metal_compute::MetalCompute* mc, int C);

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
  // Simdgroup-MMA dense GEMM for the non-matrix-core conv path (see
  // conv_gemm_bias_). Best-effort: invalid leaves the scalar fallback.
  metal_compute::ComputeFunction _fn_gemm_t_bm64, _fn_gemm_t_bm64bn64;
  // Direct (im2col-free) 3x3 conv: gathers the 3x3 neighbourhood into
  // threadgroup memory and runs the SAME simdgroup MMA as the GEMMs above, so
  // the [H*W, 9*cin] scratch is never written or read back. On M4 the im2col
  // round-trip was ~56% of a large conv even after its index math was fixed.
  // VPIPE_VAE_NO_DIRECT_CONV falls back to im2col + conv_gemm_bias_.
  metal_compute::ComputeFunction _fn_conv3x3_s1_bn64, _fn_conv3x3_s1_bn32,
      _fn_conv3x3_s2_bn64, _fn_conv3x3_s2_bn32, _fn_conv3x3_s1_bn128,
      _fn_conv3x3_s2_bn128;
  // Returns false when the shape is not a fit, leaving the caller on im2col.
  bool direct_conv3x3_(metal_compute::ComputeEncoder& enc,
                       const metal_compute::SharedBuffer& in,
                       const metal_compute::SharedBuffer& out, int H, int W,
                       const Conv& c, int stride);
  // MATERIALIZED attention for the mid block. The flash kernels run the whole
  // D=512 SINGLE-head contraction in one threadgroup, which forces the D axis
  // to be split across simdgroups and a cross-simdgroup softmax reduction per
  // key block; MEASURED 0.65 TFLOP/s against ~4 TFLOP/s for the dense GEMM on
  // the same box. Q.K^T -> row softmax -> P.V through those GEMMs instead.
  // The score matrix is O(hw^2) (8.6 GB at 2Kx2K), so queries run in BANDS:
  // mat_attn_band_ picks the largest band whose scores fit kMatAttnScoreBytes.
  // VPIPE_VAE_NO_MAT_ATTN restores the flash path (A/B).
  static constexpr std::size_t kMatAttnScoreBytes = 256u << 20;   // 256 MB
  int mat_attn_band_(std::size_t hw) const noexcept;

  // TILED decode. A whole-image decode's peak scales with the OUTPUT area, so
  // a resolution that fits a 64 GB box can exceed a 16 GB one; decode() used to
  // reject those outright. Instead, split the LATENT into overlapping windows,
  // decode each through the normal path, and cross-fade the RGB results, which
  // bounds the peak at one window regardless of output size.
  //
  // This is NOT numerically identical to a whole-image decode: the mid-block
  // attention is GLOBAL, so per-window attention sees only its own window, and
  // the convolutions see zero padding at window edges rather than real
  // neighbours. The overlap cross-fade hides the conv seam; the attention
  // difference is real and is why tiling stays a FALLBACK, entered only when
  // the whole-image decode does not fit (or VPIPE_VAE_TILE forces it).
  // VPIPE_VAE_NO_TILE restores the old hard failure.
  static constexpr int kTileMin16 = 8;      // smallest useful latent window
  static constexpr int kTileOvNum = 1;      // overlap = 1/4 of the window
  static constexpr int kTileOvDen = 4;
  // Largest square latent window whose decode peak fits `budget` (0 if none).
  int decode_tile_side_(std::size_t budget) const noexcept;
  metal_compute::SharedBuffer decode_tiled_(
      const metal_compute::SharedBuffer& z, int h16, int w16, int tile16,
      std::string* err);
  metal_compute::ComputeFunction _fn_softmax_rows, _fn_transpose;

  metal_compute::ComputeLibrary _lib_convhw;
  metal_compute::ComputeFunction _fn_conv_hw_s1, _fn_conv_hw_s2;
  bool _use_hwconv = false;
};

}  // namespace genai
}  // namespace vpipe

#endif

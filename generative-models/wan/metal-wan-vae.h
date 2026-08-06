#ifndef GENERATIVE_MODELS_WAN_METAL_WAN_VAE_H
#define GENERATIVE_MODELS_WAN_METAL_WAN_VAE_H

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/shared/kernel-sets/vae-mid-attn-tune.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

class WeightSet;   // generative-models/weight-set.h

// The Wan video VAE (AutoencoderKLWan), run in f16 on the metal-compute
// backend: RGB video <-> a 16-channel latent at 8x spatial and 4x TEMPORAL
// compression.
//
// This is the general form of the net MetalKrea2Vae already runs. The
// Qwen-Image VAE (AutoencoderKLQwenImage, used by Krea-2 and
// Qwen-Image-Edit) is this same architecture -- same base_dim 96, same
// dim_mult {1,2,4,4}, same two residual blocks per stage, same tensor
// NAMES -- with the time axis pinned to a single frame. There, every
// causal conv3d left-pads two zero frames, so only the kt=2 temporal slice
// of each 3x3x3 weight survives and every temporal resample is skipped;
// what is left is a pure 2D conv net. Here the time axis is restored:
//
//   * every conv is CAUSAL in time. Output frame t reads input frames
//     t-2, t-1, t, with zero frames before the start of the sequence.
//   * two of the three resample stages resample in TIME as well as space
//     (`temperal_downsample`), via a (3,1,1) `time_conv`.
//   * the video is processed in CHUNKS with a per-conv carry of the last
//     two input frames, which is what makes the memory bounded: a chunk
//     is one latent frame on the way out and four video frames on the way
//     in, never the whole clip.
//
// The frame arithmetic follows from the chunking rather than from a plain
// stride: the first chunk is a single frame that skips the temporal
// resamples entirely, so
//
//     encode: F video frames  -> T = 1 + (F - 1) / 4 latent frames
//     decode: T latent frames -> F = 1 + 4 * (T - 1) video frames
//
// and F must satisfy F % 4 == 1 (81, 121, ... -- the frame counts the Wan
// pipelines use).
//
// Layout matches MetalKrea2Vae: a channel-last [frames*H*W, C] interior so
// RMS-norm, 1x1 convs, SiLU and the mid-block attention are the existing LM
// kernels and the spatial 3x3 convs are im2col -> dense_gemm. The causal
// conv3d adds one kernel (im2col_hwc_3x3x3_tiled_f16), which gathers its
// three temporal taps from three separately-bound frame views rather than
// from a materialized window -- so carrying frames across a chunk boundary
// costs a binding, not a copy.
class MetalWanVae {
 public:
  struct Config {
    int base_dim       = 96;
    int z_dim          = 16;
    int dim_mult[4]    = {1, 2, 4, 4};
    int num_res_blocks = 2;
    // config `temperal_downsample`: whether resample stage i also halves
    // the frame count. The decoder walks it REVERSED (temperal_upsample).
    // Two trues => the 4x temporal compression above.
    bool temperal_downsample[3] = {false, true, true};
    // Per-channel latent statistics (config latents_mean / latents_std)
    // used by unwhiten(); the pipeline applies latents = latents*std + mean.
    std::vector<float> latents_mean;
    std::vector<float> latents_std;
  };

  // Read a Config out of a diffusers `vae/config.json`. False when the
  // file is missing or is not an AutoencoderKLWan config.
  static bool config_from_json(const std::string& vae_dir, Config& out,
                               std::string* err = nullptr);

  // `with_encoder` also loads the encoder weights (the image-to-video
  // conditioning path needs them); a decode-only graph leaves it false so
  // the encoder half is never read.
  //
  // Prefer the WeightSet overload: the set is the manager's shared,
  // reference-counted view of the checkpoint, so a second VAE over the
  // same directory reuses these tensors instead of loading its own copy.
  // The dir overload opens a PRIVATE set (tests, and callers with no
  // session to ask).
  static std::unique_ptr<MetalWanVae>
  load(const std::string& model_dir, metal_compute::MetalCompute* mc,
       const Config& cfg, bool with_encoder = false);

  static std::unique_ptr<MetalWanVae>
  load(std::shared_ptr<WeightSet> ws, metal_compute::MetalCompute* mc,
       const Config& cfg, bool with_encoder = false);

  // Materialise the encoder half now, if it is not already loaded. Cheap
  // and idempotent once loaded. False if it cannot be loaded.
  bool ensure_encoder();

  // How many video frames a T-frame latent decodes to, and how many latent
  // frames an F-frame video encodes to. Both are exact, not estimates --
  // the chunking makes the first frame a special case.
  static int video_frames(int latent_frames)
  {
    return latent_frames > 0 ? 1 + 4 * (latent_frames - 1) : 0;
  }
  static int latent_frames(int video_frames)
  {
    return video_frames > 0 ? 1 + (video_frames - 1) / 4 : 0;
  }

  // Called once per decoded CHUNK with that chunk's frames, channel-first
  // [3, n, h8*8, w8*8] f16 in [-1, 1], and the index of its first frame in
  // the clip. Chunk 0 is one frame; every later chunk is four. Streaming
  // the output this way is not an optimization detail -- it is what keeps
  // a long clip from having to exist in memory at once (81 frames of 720p
  // f16 RGB is ~450 MB before anything downstream touches it).
  using FrameSink =
      std::function<bool(const metal_compute::SharedBuffer& rgb, int frame0,
                         int n_frames)>;

  // Decode a latent video `z` laid out channel-first [z_dim, T, h8, w8]
  // (already un-whitened) into video_frames(T) RGB frames, delivered to
  // `on_frame`. Returns false on failure, with a reason in `err` --
  // distinguishing an over-commit (a preflight budget shortfall, a failed
  // allocation, or a GPU out-of-memory detected at commit) from other
  // errors, so the caller can surface it instead of emitting a corrupt
  // clip. A sink that returns false aborts the decode cleanly.
  bool decode(const metal_compute::SharedBuffer& z, int T, int h8, int w8,
              const FrameSink& on_frame, std::string* err = nullptr);

  // Single-frame convenience: decode a [z_dim, 1, h8, w8] latent to one
  // RGB frame [3, h8*8, w8*8]. This is the call that reduces to exactly
  // what MetalKrea2Vae::decode does, and the test that holds the two to
  // each other uses it.
  metal_compute::SharedBuffer
  decode_frame(const metal_compute::SharedBuffer& z, int h8, int w8,
               std::string* err = nullptr);

  // Encode an RGB video [3, F, H, W] (channel-first, in [-1,1], F % 4 == 1)
  // into the WHITENED latent [z_dim, T, H/8, W/8] (channel-first) -- the
  // posterior mode (mean), whitened (x-mean)/std. Empty on failure / no
  // encoder.
  metal_compute::SharedBuffer
  encode(const metal_compute::SharedBuffer& video, int F, int H, int W,
         std::string* err = nullptr);

  bool has_encoder() const { return _has_encoder; }

  // In-place per-channel un-whiten of a channel-first latent video
  // [z_dim, T, h8, w8]: z[c] = z[c]*latents_std[c] + latents_mean[c].
  // Returns a fresh buffer.
  metal_compute::SharedBuffer
  unwhiten(const metal_compute::SharedBuffer& z, int T, int h8, int w8);

  // Conservative estimate of the peak GPU memory (bytes) one decode CHUNK
  // at the given latent size needs. Per chunk, not per clip -- the clip is
  // streamed. For a preflight memory_budget() check.
  std::size_t decode_peak_bytes(int h8, int w8) const noexcept;

  const Config& config() const { return _cfg; }

 private:
  MetalWanVae() = default;

  // A convolution stored as a dense-GEMM weight [Cout, K] (+ bias [Cout]),
  // with K covering every tap so one GEMM realizes the whole convolution:
  //   causal 3x3x3 : K = 27*Cin, flattened (kt,ky,kx,cin)  -- kt = 3
  //   plain 3x3    : K =  9*Cin, flattened (ky,kx,cin)     -- kt = 1
  //   time (3,1,1) : K =  3*Cin, flattened (kt,cin)        -- kt = 3, ks = 1
  //   1x1          : K =    Cin                            -- kt = 1, ks = 1
  struct Conv {
    metal_compute::SharedBuffer w, b;
    int cin = 0, cout = 0, k = 0;
    int kt = 1;      // temporal taps
    int ks = 1;      // spatial taps (9 for 3x3, else 1)
    bool empty() const { return w.empty(); }
  };
  struct ResBlock {
    metal_compute::SharedBuffer n1g, n2g;   // RMS gammas (standard, no +1)
    Conv c1, c2, shortcut;                  // c1/c2 causal 3x3x3
    bool has_short = false;
    int cin = 0, cout = 0;
  };
  struct Attn {
    metal_compute::SharedBuffer ng;
    Conv q, k, v, proj;             // 1x1 convs (to_qkv split into q/k/v)
    int dim = 0;
  };
  // One resample stage. `time` is the (3,1,1) conv that is present only on
  // a temporal stage; `space` is the plain 2D 3x3 (stride 2 down / stride 1
  // after a nearest 2x up).
  struct Resample {
    Conv space;
    Conv time;
    bool temporal = false;
    bool present  = false;
  };
  struct UpBlock {
    std::vector<ResBlock> resnets;
    Resample up;
    int up_dim = 0;                 // channels feeding the upsample conv
  };
  struct DownStage {
    std::vector<ResBlock> resnets;
    Resample down;
  };

  // ---- per-forward temporal state ------------------------------------
  //
  // The carry of ONE causal conv: its last up-to-two INPUT frames, kept so
  // the next chunk's first output frames can reach back across the chunk
  // boundary. The reference calls this feat_cache and indexes it by a
  // counter that walks the convs in traversal order; `Carry` is one entry
  // and the counter lives in decode()/encode().
  //
  // Two slots addressed MODULO the running input-frame count, not a
  // shifted window: a chunk can be a single frame (every conv above the
  // first temporal upsample sees t == 1), so a shifting cache would have
  // to move the older frame down on every chunk, and that copy would be a
  // GPU round trip for nothing. Frame j of the sequence is at slot j % 2
  // and is live while total-2 <= j < total.
  //
  // `seen` marks a temporal resample that has consumed its first chunk.
  // That chunk is passed through untouched (the reference's "Rep"
  // sentinel), which is why the frame arithmetic has a +1 in it.
  struct Carry {
    metal_compute::SharedBuffer buf;   // 2 slots of [hw, cin] f16
    long long   total = 0;             // input frames consumed so far
    std::size_t hw    = 0;
    int         cin   = 0;
    bool        seen  = false;
  };

  // Everything one chunk's dispatches need: the encoder they are queued
  // on, the buffer pool they allocate from, and the shared im2col band.
  struct Ctx;

  // ---- loading --------------------------------------------------------
  Conv load_conv3d_(WeightSet& ws, const std::string& nm);
  Conv load_conv2d_(WeightSet& ws, const std::string& nm);
  Conv load_time_conv_(WeightSet& ws, const std::string& nm);
  Conv load_conv1x1_(WeightSet& ws, const std::string& nm);
  metal_compute::SharedBuffer load_vec_(WeightSet& ws, const std::string& nm);
  bool load_resblock_(WeightSet& ws, const std::string& pre, ResBlock& rb,
                      int cin, int cout);
  bool load_attn_(WeightSet& ws, const std::string& pre, Attn& a, int dim);
  bool load_encoder_(WeightSet& ws);

  metal_compute::MetalCompute* _mc = nullptr;
  Config _cfg;

  // ---- decoder --------------------------------------------------------
  Conv _post_quant;                 // 1x1 conv z_dim -> z_dim
  Conv _conv_in;                    // causal 3x3x3 z_dim -> dims[0]
  ResBlock _mid_res0, _mid_res1;
  Attn _mid_attn;
  std::vector<UpBlock> _up_blocks;  // 4
  metal_compute::SharedBuffer _norm_out_g;
  Conv _conv_out;                   // causal 3x3x3 base_dim -> 3

  // ---- encoder (loaded only when with_encoder) ------------------------
  bool _has_encoder = false;
  Conv _enc_conv_in;                // causal 3x3x3 3 -> base
  std::vector<DownStage> _enc_down; // 4 stages (3 with a downsample)
  ResBlock _enc_mid_res0, _enc_mid_res1;
  Attn _enc_mid_attn;
  metal_compute::SharedBuffer _enc_norm_out_g;
  Conv _enc_conv_out;               // causal 3x3x3 base*mult[-1] -> z_dim*2
  Conv _quant_conv;                 // 1x1 conv z_dim*2 -> z_dim*2

  // The checkpoint these weights come from, held for this model's whole
  // life: the cached tensors are aliases into buffers it owns, so it has
  // to outlive them.
  std::shared_ptr<WeightSet> _ws;
  // Part the loaders currently attribute their tensors to; "" is the
  // always-resident trunk, "encoder" the half release_part() can drop.
  std::string _part;

  // ---- forward primitives ---------------------------------------------
  // y[M,N] = x[M,K] @ w[N,K]^T (+ bias[N]) written at output row y_row0.
  // Matrix-core matmul2d for tall M, else the steel dense GEMM.
  void gemm_bias_(Ctx& cx, const metal_compute::SharedBuffer& x,
                  const metal_compute::SharedBuffer& w,
                  const metal_compute::SharedBuffer& b,
                  const metal_compute::SharedBuffer& y, int M, int N, int K,
                  int y_row0 = 0, std::size_t x_off_rows = 0);

  // One output FRAME of a convolution, gathering its taps from `taps`
  // (kt frame views, each [hw, cin]) into the shared band scratch and
  // running the GEMM. Handles all four Conv shapes above.
  void conv_frame_(Ctx& cx, const Conv& c,
                   const metal_compute::SharedBuffer* const taps[3],
                   const std::size_t tap_off[3],
                   const metal_compute::SharedBuffer& out,
                   std::size_t out_row0, int H, int W, int stride);

  // Whole-chunk convolution. `in` is [(t_in)*hw, cin] and `carry` supplies
  // the frames before it; `out` gets t_out frames. The mapping from output
  // frame to input frames is the causal one for a 3x3x3 / (3,1,1) stride-1
  // conv, and the strided one for the encoder's temporal downsample.
  metal_compute::SharedBuffer&
  conv_chunk_(Ctx& cx, const Conv& c, const metal_compute::SharedBuffer& in,
              int t_in, int H, int W, int stride, Carry* carry);

  // Pointwise / rowwise helpers, all over a whole chunk at once.
  metal_compute::SharedBuffer&
  normc_(Ctx& cx, const metal_compute::SharedBuffer& in, std::size_t rows,
         int C, const metal_compute::SharedBuffer& g);
  void silu_(Ctx& cx, const metal_compute::SharedBuffer& x, std::size_t n);
  metal_compute::SharedBuffer&
  resadd_(Ctx& cx, const metal_compute::SharedBuffer& a,
          const metal_compute::SharedBuffer& b, std::size_t n);
  metal_compute::SharedBuffer&
  upsample2x_(Ctx& cx, const metal_compute::SharedBuffer& in, int t, int H,
              int W, int C);
  metal_compute::SharedBuffer&
  resblock_(Ctx& cx, const ResBlock& rb, const metal_compute::SharedBuffer& x,
            int t, int H, int W, Carry* c1, Carry* c2);
  metal_compute::SharedBuffer&
  attention_(Ctx& cx, const Attn& a, const metal_compute::SharedBuffer& x,
             int t, int H, int W);

  // The temporal halves of the two resample stages. `up` doubles the frame
  // count (except on the first chunk, which it passes through untouched);
  // `down` folds the carried frame in and strides by two.
  metal_compute::SharedBuffer&
  time_up_(Ctx& cx, const Resample& rs, const metal_compute::SharedBuffer& x,
           int& t, std::size_t hw, int C, Carry* carry);
  metal_compute::SharedBuffer&
  time_down_(Ctx& cx, const Resample& rs, const metal_compute::SharedBuffer& x,
             int& t, std::size_t hw, int C, Carry* carry);

  // Copy the trailing min(2, t) frames of `x` into `carry` for the next
  // chunk. A GPU-side copy: the frames are still only on the GPU timeline.
  void save_carry_(Ctx& cx, Carry& carry, const metal_compute::SharedBuffer& x,
                   int t, std::size_t hw, int cin);

  // ---- kernels ---------------------------------------------------------
  metal_compute::ComputeLibrary _lib_gemm, _lib_elt, _lib_rms, _lib_sdpa;
  metal_compute::ComputeFunction _fn_gemm_bias, _fn_rms, _fn_mul_sigmoid,
      _fn_residual, _fn_clamp, _fn_sdpa, _fn_upsample, _fn_copy;
  // The 2D im2col twins (the plain-3x3 resample convs) and the 3D one that
  // is this VAE's addition.
  metal_compute::ComputeFunction _fn_im2col_tiled, _fn_im2col_s2_tiled,
      _fn_im2col3d_tiled, _fn_concat3, _fn_time_unshuffle;

  // Matrix-core dense GEMM (matmul2d/NAX). Same rationale and the same
  // guards as the Qwen-Image VAE: bias folds as a separate bias_add_rows
  // pass, and a tall GEMM is split at _mma_max_m because the matmul2d op
  // corrupts output rows past ~2^19.
  metal_compute::ComputeLibrary _lib_dense_mma;
  metal_compute::ComputeFunction _fn_dense_mma, _fn_dense_mma_deep,
      _fn_bias_add;
  bool _use_mma2  = false;
  int  _mma_min_m = 64;
  int  _mma_min_n = 16;
  int  _mma_max_m = 1 << 19;

  // Mid-block attention, picked by MEASUREMENT at load (vae-mid-attn-tune.h)
  // exactly as in the Qwen-Image VAE -- the attention here is the same
  // single-head spatial one, run per frame.
  metal_compute::ComputeLibrary _lib_sdpa_mma;
  metal_compute::ComputeFunction _fn_sdpa_full_mma, _fn_sdpa_full_smm,
      _fn_sdpa_full_wide16, _fn_sdpa_full_wide32, _fn_sdpa_full_wide64;
  using MidAttn = vae_mid_attn::Kind;
  MidAttn _attn_pick = MidAttn::kScalar;
  bool mid_attn_available_(MidAttn k) const;
  void load_wide_attn_(int mid_d);
  void encode_mid_attn_(metal_compute::ComputeEncoder& enc, MidAttn kind,
                        const metal_compute::SharedBuffer& q,
                        const metal_compute::SharedBuffer& k,
                        const metal_compute::SharedBuffer& v,
                        const metal_compute::SharedBuffer& att,
                        std::size_t hw, int C, float scale);
  void autotune_mid_attn_(metal_compute::MetalCompute* mc, int C);
  static unsigned attn_threads_(int bq)
  {
    return (bq == 16 ? 4u : bq == 32 ? 8u : 16u) * 32u;
  }
};

}  // namespace genai
}  // namespace vpipe

#endif

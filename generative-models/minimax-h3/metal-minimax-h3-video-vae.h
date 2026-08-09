#ifndef GENERATIVE_MODELS_MINIMAX_H3_METAL_MINIMAX_H3_VIDEO_VAE_H
#define GENERATIVE_MODELS_MINIMAX_H3_METAL_MINIMAX_H3_VIDEO_VAE_H

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/minimax-h3/minimax-h3-layout.h"
#include "generative-models/shared/dit-block-progress.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

class WeightSet;

// The MiniMax-H3 video VAE decoder (`MiniMaxH3VideoViTDecoder3d`).
//
// This VAE is not shaped like any of the others here. The encoder is a
// causal 3D CNN, but the DECODER is a 36-layer, 2048-wide **ViT**: every
// latent voxel becomes one token, the tokens attend to each other with
// full self-attention, and each one is finally projected into a whole
// 4 x 16 x 16 pixel block. That is where the checkpoint's 10.4 GB goes --
// 2.4B of its ~2.6B parameters are the decoder's transformer blocks.
//
// Two consequences follow from the decoder being a transformer:
//
//   * TILING IS STRUCTURAL, not an optimization. Attention is quadratic
//     in the number of latent voxels, and a 5-second 768p clip is
//     ~129,000 of them -- which is not a slow decode, it is an
//     impossible one. The released config tiles at 256 PIXELS (16 latent
//     cells) with a 64-pixel minimum overlap, so a tile is ~1,280
//     tokens. This class decodes whatever grid it is handed; laying out
//     the tiles and cross-fading them is the caller's job.
//   * it reuses the DiT's kernels almost exactly -- partial rotate-half
//     rope, per-head q/k RMS, the value-first SwiGLU. The differences
//     are that q/k/v/out carry BIASES, the q/k norms have no affine at
//     all, the residual gate is a learned per-CHANNEL vector rather than
//     a per-row modulation, and the output norm is a LayerNorm.
//
// The rope is the one place the two genuinely differ: the DiT's
// coordinates are absolute and scaled by 32, while this one normalizes
// each axis to [-1, 1) and multiplies by 2*pi, so the grid depends on
// the TILE's extent rather than on the clip's. A tile decoded on its own
// therefore sees a different rope grid than the same voxels would inside
// a larger tile -- which is why the reference's tile size is part of the
// checkpoint's config and not a free knob.
class MetalMiniMaxH3VideoVae {
 public:
  struct Config {
    int z_channels   = 24;
    int out_channels = 3;
    int dim          = 2048;   // heads * head_dim
    int n_heads      = 32;
    int head_dim     = 64;
    int n_layers     = 36;
    int ffn_inner    = 8192;   // w1 is 2x this wide (fused value|gate)
    int patch        = 16;     // spatial compression
    int patch_t      = 4;      // temporal compression
    int n_register   = 4;
    double rope_theta     = 100.0;
    double rope_dim_ratio = 0.75;
    float norm_eps        = 1e-5f;

    // ---- the CNN encoder ------------------------------------------
    // `space_down` / `time_down` multiply out to `patch` / `patch_t`,
    // and their per-level values are what decide which levels carry a
    // downsampling convolution at all -- a level whose two factors are
    // both 1 has none. `ch_mult` scales `base_ch`; a level's INPUT
    // width is the previous level's output (the first repeats
    // `base_ch`), which is where the three 1x1x1 shortcut convolutions
    // sit.
    int in_channels = 3;
    int base_ch     = 128;
    int res_blocks  = 2;
    int gn_groups   = 32;
    float enc_norm_eps = 1e-6f;
    std::vector<int> ch_mult    = {1, 2, 2, 4, 4, 8};
    std::vector<int> space_down = {2, 2, 2, 2, 1, 1};
    std::vector<int> time_down  = {1, 2, 2, 1, 1, 1};
    // The temporal chunking `encode()` is a single step of: the
    // reference encodes `clip_length` pixel frames at a time and drops
    // the last `token_drop` latent frames of the whole sequence.
    int clip_length = 17;
    int token_drop  = 3;
    // Spatial tiling, in PIXELS. Both halves must use the same values:
    // a tile's rope is built from its own extent, so an encode and a
    // decode that disagree here compute different functions rather than
    // merely seaming differently.
    int tile_size        = 256;
    int tile_overlap_min = 64;
    // Per-channel latent whitening. The DiT works in NORMALIZED latent
    // space -- `(z - mean) / std` -- so a decode has to undo it first.
    // Empty when the checkpoint states neither, which means identity.
    std::vector<float> latents_mean;
    std::vector<float> latents_std;

    // Derived temporal-chunk geometry. `clip_length` is deliberately
    // NOT a multiple of `patch_t`, so a chunk's leading `frame_pre_pad`
    // pixel frames belong to an implicit pad and the decoder has to
    // re-derive them.
    int tokens_per_chunk() const
    {
      return patch_t > 0 ? (clip_length + patch_t - 1) / patch_t : 0;
    }
    int frame_pre_pad() const
    {
      return patch_t > 0 ? ((-clip_length % patch_t) + patch_t) % patch_t : 0;
    }
    int token_overlap() const
    {
      const int c = tokens_per_chunk();
      return c > 0 ? ((-token_drop % c) + c) % c : 0;
    }
    int frame_overlap() const
    {
      const int f = token_overlap() * patch_t - frame_pre_pad();
      return f > 0 ? f : 0;
    }

    // int(head_dim * rope_dim_ratio) = 48 of 64 channels rotate. The
    // angles are 3 axes x (rope_dim / (2*3)) frequencies concatenated and
    // then duplicated, so the table needs rope_dim/2 entries per row.
    int rope_dim() const { return (int)((double)head_dim * rope_dim_ratio); }
    int rope_freqs() const { return rope_dim() / 6; }
    int patch_elems() const
    {
      return out_channels * patch_t * patch * patch;   // 3072
    }
  };

  // Read a Config out of the VAE's `source/config.json`. `vae_dir` may be
  // the `video_vae` directory, its `source` subdirectory, or the
  // partition / repository root.
  static bool config_from_json(const std::string& vae_dir, Config& out,
                               std::string* err = nullptr);

  static std::string resolve_vae_dir(const std::string& path);

  static std::unique_ptr<MetalMiniMaxH3VideoVae>
  load(const std::string& vae_dir, metal_compute::MetalCompute* mc,
       const Config& cfg);

  static std::unique_ptr<MetalMiniMaxH3VideoVae>
  load(std::shared_ptr<WeightSet> ws, metal_compute::MetalCompute* mc,
       const Config& cfg);

  ~MetalMiniMaxH3VideoVae();

  // Decode one latent grid. `z` is CHANNEL-FIRST [z_channels, T, h, w]
  // bf16 -- the layout the DiT's unpatchify produces -- and the result is
  // channel-first [out_channels, T*4, h*16, w*16] bf16.
  //
  // `post_quant_conv` is applied here: the reference's decode is
  // `decoder(post_quant_conv(z))`, and splitting them would leave a
  // 1x1x1 convolution stranded in the caller.
  //
  // Empty on failure, with a reason in `err`.
  metal_compute::SharedBuffer
  decode(const metal_compute::SharedBuffer& z, int T, int h, int w,
         std::string* err = nullptr);

  // Encode ONE temporal clip -- the reference's `_encode_clip` with
  // tiling off. `x` is CHANNEL-FIRST [in_channels, T, H, W] bf16 in
  // [-1, 1] and the result is channel-first
  // [2*z_channels, encoded_frames(T), H/patch, W/patch] bf16: the
  // MOMENTS, mean and log-variance concatenated on the channel axis,
  // with quant_conv already applied.
  //
  // H and W must be multiples of `patch`. This decodes whatever extent
  // it is handed; splitting a frame into tiles and blending the
  // overlaps is the caller's job, as it is for decode(). Unlike the
  // decoder, tiling here is an ordinary memory optimization -- a
  // convolution has no quadratic term -- but the two must tile the SAME
  // way for a round trip to line up.
  //
  // Empty on failure, with a reason in `err`.
  metal_compute::SharedBuffer
  encode(const metal_compute::SharedBuffer& x, int T, int H, int W,
         std::string* err = nullptr);

  // Latent frames one clip of `T` pixel frames encodes to. The causal
  // padding makes this ceil over each temporal stride rather than a
  // plain division: 17 frames become 9, then 5.
  int encoded_frames(int T) const;

  // Encode a WHOLE video: `clip_length`-frame chunks, each spatially
  // tiled and stitched, with the trailing `token_drop` latent frames
  // dropped from the result. `x` is channel-first
  // [in_channels, T, H, W] bf16 and the result is channel-first
  // [2*z_channels, video_latent_frames(T), H/patch, W/patch].
  //
  // A video whose length is not a whole number of clips is extended by
  // REPEATING its last frame. A single frame is a special case and goes
  // through the spatial path alone -- padding it up to a clip would run
  // the temporal path over `clip_length` copies of one image and return
  // a different number of latent frames, which is not the conditioning
  // this model was trained with.
  metal_compute::SharedBuffer
  encode_video(const metal_compute::SharedBuffer& x, int T, int H, int W,
               int* latent_frames = nullptr, std::string* err = nullptr);

  // Decode a whole latent video, mirroring that chunking: consecutive
  // chunks overlap by `frame_overlap` pixel frames and are linearly
  // cross-faded, and latent frames are repeated at the end when the
  // length is not a whole number of chunks.
  metal_compute::SharedBuffer
  decode_video(const metal_compute::SharedBuffer& z, int LT, int lh, int lw,
               int* out_frames = nullptr, std::string* err = nullptr);

  // Latent frames a `T`-frame video encodes to, and pixel frames an
  // `LT`-frame latent decodes to. NOT inverses: `token_drop` discards
  // the tail of the encode on purpose, so a round trip is shorter than
  // it started. 0 when `LT` is too short to form a chunk.
  int video_latent_frames(int T) const;
  int decoded_frames(int LT) const;

  const Config& config() const { return _cfg; }

  // Per-TILE progress out of decode_video / decode_tiled_. A 960x544
  // clip is ~15 tiles per chunk of a 2.4B ViT -- around a minute during
  // which nothing else this class does is observable -- so without it a
  // caller can only report the decode as one indivisible step.
  //
  // Fires on the encode thread between tiles; must be cheap and must not
  // re-enter the decoder. See VaeTileProgressFn for the counting rule.
  void set_tile_progress(genai::VaeTileProgressFn fn)
  {
    _tile_progress = std::move(fn);
  }

 private:
  MetalMiniMaxH3VideoVae() = default;

  struct Linear {
    metal_compute::SharedBuffer w, b;
    metal_compute::SharedBuffer codes, scales, qbias;
    bool quantized = false;
    int  bits = 0;
    bool empty() const { return quantized ? codes.empty() : w.empty(); }
  };

  struct Block {
    metal_compute::SharedBuffer n1, n2, s1, s2;
    Linear qkv, out, w1, w2;
  };

  // A causal conv3d flattened into a GEMM. `spatial` distinguishes the
  // 3x3x3 convolutions, whose weight is [cout, 27*cin] laid out
  // (kt, ky, kx, cin) to pair with the im2col, from the 1x1x1 shortcut
  // and quant convolutions, which need no gather at all and run as a
  // plain [cout, cin] Linear over every voxel at once.
  struct Conv3d {
    Linear l;
    int  cin = 0, cout = 0, k = 0;
    bool spatial = false;
    bool empty() const { return l.empty(); }
  };

  struct EncResnet {
    metal_compute::SharedBuffer n1w, n1b, n2w, n2b;
    Conv3d c1, c2, skip;              // skip empty when cin == cout
    int cin = 0, cout = 0;
  };

  struct EncLevel {
    std::vector<EncResnet> res;
    Conv3d down;                      // empty when the level does not
    int space = 1, time = 1;          // downsample
    int cin = 0, cout = 0;
  };

  metal_compute::SharedBuffer weight_(WeightSet& ws, const std::string& nm);
  Linear linear_(WeightSet& ws, const std::string& nm);
  Conv3d conv3d_(WeightSet& ws, const std::string& nm);
  bool   load_encoder_(WeightSet& ws);

  void gemm_(metal_compute::ComputeEncoder& enc,
             const metal_compute::SharedBuffer& x, std::size_t x_off,
             const Linear& l, const metal_compute::SharedBuffer& y,
             std::size_t y_off, int M, int N, int K);
  // Dequant-once into _w_deq (quantized) or the weight as-is (dense), then
  // one dense matmul2d tile. False when the shape or the machine does not
  // take it, which leaves gemm_ on its existing path.
  bool gemm_mma_(metal_compute::ComputeEncoder& enc,
                 const metal_compute::SharedBuffer& x, std::size_t x_off,
                 const Linear& l, const metal_compute::SharedBuffer& y,
                 std::size_t y_off, int M, int N, int K);

  // cos/sin over the [-1, 1) normalized (t, h, w) grid of ONE tile, plus
  // the register and cls rows, which all sit at position 0.
  void build_rope_(int T, int h, int w, metal_compute::SharedBuffer& cos_out,
                   metal_compute::SharedBuffer& sin_out) const;

  struct Scratch {
    int rows = -1;
    metal_compute::SharedBuffer rcos, rsin;
    metal_compute::SharedBuffer x, nm, qkv, qh, kh, vh, oh, ob, ff, patches;
    metal_compute::SharedBuffer zt, ones;
  };
  bool ensure_scratch_(int rows, int voxels);
  Scratch _s;

  // One causal conv3d over a whole clip: `in` is [Ti, H*W, cin] and
  // `out` becomes [To, Ho*Wo, cout], with the output frame count that
  // `strides` implies. Returns it, or 0 on a bad shape.
  int enc_conv_(metal_compute::ComputeEncoder& enc, const Conv3d& c,
                const metal_compute::SharedBuffer& in, int Ti, int H, int W,
                const metal_compute::SharedBuffer& out, int stride_t,
                int stride_s);
  void enc_gn_(metal_compute::ComputeEncoder& enc,
               const metal_compute::SharedBuffer& in,
               const metal_compute::SharedBuffer& gamma,
               const metal_compute::SharedBuffer& beta,
               const metal_compute::SharedBuffer& out, int T, int rows,
               int C, bool silu);

  // Three interchangeable activation buffers, each sized for the WIDEST
  // level, plus one im2col band. Three is what a resnet needs: its
  // input has to survive until the residual add, while the two
  // convolutions inside it are already using the other two.
  struct EncScratch {
    int T = -1, H = -1, W = -1;
    metal_compute::SharedBuffer a, b, c, col;
    std::size_t col_cap = 0;          // elements, not rows: the band a
  };                                  // conv can afford depends on its cin
  bool ensure_enc_scratch_(int T, int H, int W);
  EncScratch _es;

  // One clip / one latent grid, spatially tiled and stitched. Falls
  // straight through to encode()/decode() when the extent fits in a
  // single tile.
  metal_compute::SharedBuffer
  encode_tiled_(const metal_compute::SharedBuffer& x, int T, int H, int W,
                std::string* err);
  metal_compute::SharedBuffer
  decode_tiled_(const metal_compute::SharedBuffer& z, int LT, int lh, int lw,
                std::string* err);

  // Cross-fade a grid of channel-first [C, T, th, tw] tiles into one
  // [C, T, out_h, out_w]. `ratio` converts the split's PIXEL units into
  // the tiles' own units -- `patch` for a latent grid, 1 for pixels.
  metal_compute::SharedBuffer
  stitch_(const std::vector<metal_compute::SharedBuffer>& tiles,
          const minimax_h3::TileSplit& ys, const minimax_h3::TileSplit& xs,
          int ratio, int C, int T, int out_h, int out_w);

  metal_compute::MetalCompute* _mc = nullptr;
  Config _cfg;
  std::shared_ptr<WeightSet> _ws;

  Linear _post_quant, _proj_in, _proj_out;
  metal_compute::SharedBuffer _register_tokens;
  metal_compute::SharedBuffer _norm_out_w, _norm_out_b;
  std::vector<Block> _blocks;

  Conv3d _enc_conv_in, _enc_conv_out, _quant_conv;
  metal_compute::SharedBuffer _enc_norm_w, _enc_norm_b;
  std::vector<EncLevel> _enc_levels;
  // Which WeightSet part the loaders currently attribute bytes to. The
  // encoder is loaded on first encode() and is releasable on its own,
  // so a graph that only decodes never pays for it.
  std::string _part;

  int _quant_bits = 0;
  int _quant_group = 64;

  metal_compute::ComputeLibrary _lib_gemm, _lib_elt, _lib_rms, _lib_rope,
      _lib_attn, _lib_attn_nax, _lib_qmm, _lib_dense_mma, _lib_dequant;
  // ---- M5 matrix cores, as in the DiT: dequant-once + dense matmul2d ----
  // The VAE runs once per clip rather than once per block per step, so this
  // is a smaller lever than the DiT's -- but the ViT half of the decoder is
  // a real transformer over every voxel, so the shapes are large enough to
  // reach the tile.
  metal_compute::ComputeFunction _fn_dense_mma, _fn_dense_mma_deep;
  metal_compute::ComputeFunction _fn_dequant4, _fn_dequant8;
  metal_compute::SharedBuffer _w_deq;
  // Tile progress, and the counters it reports against. `_prog_total`
  // is set by decode_video (chunks x tiles) so the bar spans the whole
  // clip; a bare decode_tiled_ call sets it to its own tile count.
  genai::VaeTileProgressFn _tile_progress;
  int _prog_done = 0;
  int _prog_total = 0;
  bool _use_mma2 = false;
  int _mma_min_m = 64;
  bool _attn_nax = false;
  metal_compute::ComputeFunction _fn_gemm, _fn_rms, _fn_rms_heads, _fn_trope,
      _fn_gated, _fn_swiglu, _fn_transpose, _fn_ln, _fn_bias_add, _fn_sdpa,
      _fn_qmm4, _fn_qmm8, _fn_gn_frames, _fn_im2col_r, _fn_residual;
  metal_compute::SharedBuffer _attn_p;
  metal_compute::ComputeFunction _fn_attn;
  int _attn_rows = -1;
  bool _steel_ok = false;
};

}  // namespace genai
}  // namespace vpipe

#endif

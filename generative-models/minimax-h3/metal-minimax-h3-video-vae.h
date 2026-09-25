#ifndef GENERATIVE_MODELS_MINIMAX_H3_METAL_MINIMAX_H3_VIDEO_VAE_H
#define GENERATIVE_MODELS_MINIMAX_H3_METAL_MINIMAX_H3_VIDEO_VAE_H

#include "generative-models/shared/ane-ffn.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/minimax-h3/minimax-h3-layout.h"
#include "generative-models/shared/dit-block-progress.h"
#include "generative-models/shared/mma-splitk.h"

#include <cstddef>
#include <functional>
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
//     rope, per-head q/k RMS, the gate-first SwiGLU. The differences
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
    int ffn_inner    = 8192;   // w1 is 2x this wide (fused gate|up)
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

    // ---- the ANE tier (LOSSY, opt-in) -------------------------------
    // The decoder blocks' feed-forward split between the GPU and the Neural
    // Engine by ROWS, per tile, per block (see shared/ane-ffn.h). About 70%
    // of the decode's arithmetic is that feed-forward; attention stays on the
    // GPU, where the ANE measured 1.8 TOPS for a whole block. `ane_rows` is
    // the ANE's share, 0 auto-balancing; `ane_layers` caps the blocks, 0 all.
    // Set from vae-decode's `ane_*` keys once the plan has granted the
    // module; VPIPE_H3_VVAE_ANE_CHUNK / VPIPE_H3_VVAE_ANE_PROFILE for tuning.
    bool  ane_ffn    = false;
    float ane_rows   = 0.0f;
    int   ane_layers = 0;
    // Per-channel latent whitening. The DiT works in NORMALIZED latent
    // space -- `(z - mean) / std` -- so a decode has to undo it first.
    // Empty when the checkpoint states neither, which means identity.
    std::vector<float> latents_mean;
    std::vector<float> latents_std;

    // ---- the single-latent-frame IMAGE decode (checkpoint-declared) --
    // Whether this checkpoint's decoder was trained to turn ONE latent
    // frame straight into one picture, and which of the `patch_t` pixel
    // frames that token expands to is that picture.
    //
    // A CAPABILITY OF THE WEIGHTS, never of the request, which is why it
    // is read from the file rather than configured. Every H3 checkpoint
    // has the same geometry and the same tensor names, so nothing about
    // the shapes distinguishes the two -- run `decode_image` on the
    // stock video decoder and it returns a picture-shaped buffer full of
    // the wrong thing. The released image checkpoint says so in its
    // `__metadata__` (`h3_t1_direct` / `h3_t1_output_slice`); a
    // checkpoint that says nothing does not have it.
    //
    // WHY A SLICE AND NOT FRAME 0. The decoder emits a whole `patch_t`
    // block per token and the chunked video path throws the leading
    // `frame_pre_pad()` of them away as an implicit pad, so the first
    // frame a chunk really carries is index 3 -- which is exactly the
    // slice the released checkpoint names. Read it anyway: agreeing with
    // a derivation is not the same as being one, and a future fine-tune
    // is free to park the picture somewhere else.
    bool t1_direct       = false;
    int  t1_output_slice = 0;

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

  // Decode ONE latent frame as ONE picture: channel-first
  // [z_channels, 1, h, w] bf16 in, channel-first
  // [out_channels, h*patch, w*patch] bf16 out -- a still, with no time
  // axis at all rather than a length-1 one.
  //
  // This is the decode `decode_video` CANNOT do. Its chunking spends
  // `token_drop` latent frames priming the temporal window, so it needs
  // `tokens_per_chunk() + token_drop` = 7 latent frames before it will
  // decode anything and refuses 1 outright -- while `encode_video`
  // turns a single still into exactly 1 latent frame. A checkpoint with
  // `t1_direct` closes that gap: its decoder was fine-tuned to read one
  // token on its own.
  //
  // False on a checkpoint that does not declare the capability, because
  // the failure is otherwise invisible: the stock video decoder returns
  // a correctly shaped picture that merely looks wrong, and the rope is
  // why. `build_rope_` normalizes the temporal axis by `T` ITSELF, so a
  // lone latent lands at coordinate 0.0 -- the middle of the spread a
  // 7-frame chunk gives its frames, and a grid no voxel occupies during
  // a video decode. The fine-tune is what learned it.
  //
  // Spatially tiled exactly as `decode_video` tiles, and the tiling is
  // REQUIRED rather than an optimization. MEASURED on a 512px centre
  // crop round-tripped through the released image checkpoint: 30.97 dB
  // PSNR at the checkpoint's own 256 tile against 23.27 dB decoding the
  // whole 512 in one go -- and the checkpoint reports 30.44 dB over a
  // 64-image 512px set, so the tiled arm is the one that reproduces it.
  //
  // The reason is the rope and not the seams. `build_rope_` normalizes
  // each axis to the extent it is HANDED, so decoding a whole image puts
  // every token at a coordinate no 256px tile occupies, and this decoder
  // was fine-tuned on a 256 -> 384 px curriculum. Raising `tile_size`
  // for an image therefore does not trade seam for compute; it computes
  // a different and worse function.
  //
  // Empty on failure, with a reason in `err`.
  metal_compute::SharedBuffer
  decode_image(const metal_compute::SharedBuffer& z, int h, int w,
               std::string* err = nullptr);

  // Whether `decode_image` will run at all -- `t1_direct` under the
  // name a caller asks its question in.
  bool can_decode_image() const { return _cfg.t1_direct; }

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

  // The same two answers over an EXPLICIT config, so the planning phase
  // can have them before any checkpoint is opened. The instance methods
  // are these with `_cfg`, so the two cannot drift.
  static int encoded_frames_for(const Config& c, int T);
  static int video_latent_frames_for(const Config& c, int T);

  // Bytes of the LATENT this VAE's geometry implies for a clip.
  //
  // From the DEFAULT config, which is what a plan can know: the ratios
  // are the model's, not the checkpoint's, and a repack ships no
  // config.json to read them from. Replacing a hand-written bound --
  // 8x spatial and 4x temporal with 32 channels was wrong in every term
  // for this family (16x, a chunked temporal rule, and 24 channels) and
  // over-stated the latent ~4x.
  static std::size_t latent_bytes(int width, int height, int frames);

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

  // Per-TILE progress out of decode_video / decode_tiled_, and out of
  // encode_video / encode_tiled_ the same way. A 960x544 clip is ~15
  // tiles per chunk of a 2.4B ViT -- around a minute during which
  // nothing else this class does is observable -- so without it a
  // caller can only report the decode as one indivisible step. The
  // encode is the same shape and as long: MEASURED ~1.8 s per frame at
  // 896x512, so a 90-frame reference is two and a half silent minutes.
  // encode_video reports against clips x tiles for the whole clip.
  //
  // Fires on the encode thread between tiles; must be cheap and must not
  // re-enter the decoder. See VaeTileProgressFn for the counting rule.
  // Bytes the ANE feed-forward tier holds for this config's tile shape;
  // see AneFeedForward::runtime_bytes. Static: the plan asks before load.
  static std::size_t ane_runtime_bytes(const Config& c) noexcept;
  // Rows per ANE predict: VPIPE_H3_VVAE_ANE_CHUNK, or 1024 -- a tile is
  // 1797 rows, so one 1024-row chunk is a 57% share, near the balance.
  static int ane_chunk_rows() noexcept;
  // Whether the tier tried to arm, and whether it did.
  bool ane_attempted() const noexcept { return _ane_tried; }
  bool ane_armed() const noexcept { return _ane != nullptr; }

  void set_tile_progress(genai::VaeTileProgressFn fn)
  {
    _tile_progress = std::move(fn);
  }

  // A COOPERATIVE STOP, checked between tiles and between clips of an
  // ENCODE. Optional; unset means nothing ever stops.
  //
  // Those are real interruption points and not merely convenient ones:
  // a tile commits its command stream and WAITS on it, so a check
  // between two of them is reached with the GPU idle and the stop is
  // bounded by one tile. That is the distinction this tree keeps having
  // to make -- checking between units of work only interrupts anything
  // when the units are separately committed, and where a loop encodes
  // the whole job into one command buffer the same check can only
  // observe a request that was already set.
  //
  // On a stop the call returns an EMPTY buffer and leaves `err`
  // untouched, because a stop is not a failure. A caller distinguishes
  // it from a real failure by asking its own stop predicate, which is
  // what encode_references does.
  void set_stop(std::function<bool()> fn) { _stop = std::move(fn); }
  bool stopping() const { return _stop && _stop(); }

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
  // Rows one dispatch may write into an N-wide destination.
  //
  // Two limits, and the tighter one is the same number. The mma tiles
  // address through `dextents<int32_t, 2>`, so a destination past 2^31
  // BYTES makes the tile whose base crosses the line stop storing --
  // silently, leaving what was already there. And `M * N` is an int in
  // the bias and steel dispatches, which is the ELEMENT count. Bytes
  // bind first, so a band under 2^31 bytes is under 2^31 elements too.
  //
  // The DECODE never comes near either: it runs per 256x256 tile, so M
  // is ~10^4 whatever the canvas. The ENCODE can, because encode() is
  // documented to take whatever extent it is handed and encode_tiled_
  // hands it the WHOLE frame when the frame does not tile. A 1344x768
  // clip through that path is [17547264, 128] = 4.5 GB, twice over the
  // line, and `t * rows * cout` overflows the int it was narrowed to.
  // VPIPE_H3_VVAE_ROW_BAND forces one (smaller only -- see the clamp).
  //
  // `widest` is the WIDEST ROW of any operand in elements, i.e.
  // max(N, K) and not N: the mma kernels wrap x, W and y alike in
  // int32 extents, so the x operand crosses the line first whenever
  // K > N. Callers pass the max; see gemm_.
  int row_band_(int widest) const;
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

  // ---- the ANE tier ------------------------------------------------
  // Built at the first decode, from its tile's row count. Null == every
  // block keeps the GPU feed-forward.
  std::unique_ptr<AneFeedForward> _ane;
  bool _ane_tried       = false;
  bool _ane_skip_warned = false;
  bool ane_setup_(int rows);
  bool ane_eligible_(int L, const Block& b);
  // w1 is [gate | up] rows with a [2*ffn] bias, gate first; w2 [dim, ffn].
  void ane_stage_(int L, const Block& b);

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
  // Deep-K split for the encoder's conv GEMMs (a 3x3x3 conv is an im2col
  // GEMM with K = 27*cin, so the deep levels contract over 13824 / 27648).
  // Tuned per shape in encode(), which is the only place the band size --
  // and therefore M -- is known before a stream opens.
  MmaSplitK _splitk;
  // Tile choice per conv shape, MEASURED. The shared rule is K-only
  // (mma_use_wide_tile: K >= 6144 -> the 256-wide tile), and for this
  // tower's narrow-N convs it is wrong: at N=512, K=13824 the 128-wide tile
  // runs 13867 GF/s against the wide tile's 10151. Two threadgroups across
  // the output width is not enough to fill the machine, and no rule keyed on
  // K can see that. Empty until encode() tunes; `_force_tile` is how the
  // tuner drives this same dispatch (0 = decide, 1 = n128, 2 = n256).
  struct TileTune { int N, K, M; bool wide; };
  std::vector<TileTune> _tile_tuned;
  int _force_tile = 0;
  // Tile progress, and the counters it reports against. `_prog_total`
  // is set by decode_video (chunks x tiles) so the bar spans the whole
  // clip; a bare decode_tiled_ call sets it to its own tile count.
  genai::VaeTileProgressFn _tile_progress;
  std::function<bool()>    _stop;
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

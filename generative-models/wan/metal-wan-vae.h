#ifndef GENERATIVE_MODELS_WAN_METAL_WAN_VAE_H
#define GENERATIVE_MODELS_WAN_METAL_WAN_VAE_H

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/shared/kernel-sets/vae-mid-attn-tune.h"

#include <functional>
#include <memory>
#include <unordered_map>
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

  // A Config for a natively-named Wan 2.1 VAE that has NO config of its
  // own -- FlashVSR publishes `Wan2.1_VAE.pth` bare, with nothing beside
  // it. The TENSORS decide: native names (`conv1`, `decoder.conv1`, not
  // diffusers' `quant_conv` / `conv_in`) and a 32-wide `conv1`, which is
  // z_dim 16 -- the Wan 2.2 VAE is a different, wider net and is refused.
  // The geometry is then Wan 2.1's fixed one and the whitening statistics
  // its published ones. False, `out` untouched, for anything else.
  static bool config_for_native_checkpoint(const std::string& path,
                                           Config& out,
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
  // The next frame count at or above `n` that this chunking can represent.
  // Counts that are not 4k+1 have no latent form at all -- the first frame
  // is its own chunk and every later chunk is exactly four -- so a caller
  // handed 80 has to become 81 rather than be truncated to 77: rounding
  // DOWN would silently deliver less video than was asked for, where
  // rounding up costs at most three frames nobody notices.
  //
  // Named beside its inverses on purpose. This rule is the VAE's, not the
  // DiT's or a stage's, and every place that has to honour it should be
  // reading it from the same one line. MiniMax-H3's counterpart is
  // `minimax_h3::align_num_frames`.
  static int align_num_frames(int n)
  {
    if (n <= 1) { return 1; }
    return n + ((1 - n % 4) + 4) % 4;
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

  // The GPU memory (bytes) a decode at this latent size holds at its peak,
  // for the preflight and for the plan. NOT per chunk, although the clip
  // streams: the causal carries -- two input frames of every 3x3x3 conv,
  // seven of them at full resolution -- are allocated by the first chunk
  // and held to the last, and they are the largest term. The im2col band
  // is not in it; the band takes whatever headroom this leaves.
  // `frame_split` is whether the post-temporal tail runs a frame at a time
  // (the default; VPIPE_WAN_VAE_NO_FRAME_SPLIT turns it off).
  //
  // `tail_frac` is the share of the plane the TILED tail carries at once --
  // 1.0 untiled, and (tile + halo) / whole for a spatial tiling. It scales
  // only what runs below the last temporal upsample, because that is what
  // tiles: the blocks above it, and the mid attention among them, keep the
  // whole plane. See decode_tile_outer_() and choose_tail_tiles_().
  //
  // `tail_frac` < 1 is the TILE-OUTER decode, which holds the clip's split-
  // plane frames and RGB and so grows with its length: `latent_frames` is
  // that length, and is read only then.
  //
  // `deep` is the tiling that starts below the MID ATTENTION rather than
  // below the last temporal upsample: a wider halo, a far smaller store
  // and head, for a box the shallow one does not fit.
  std::size_t decode_peak_bytes(int h8, int w8) const noexcept;
  static std::size_t decode_peak_bytes(const Config& cfg, int h8, int w8,
                                       bool frame_split = true,
                                       double tail_frac = 1.0,
                                       int latent_frames = 1,
                                       bool deep = false) noexcept;

  // How many tiles the TAIL (everything below the last temporal upsample)
  // is split into so a decode fits `headroom`, and the plane share one
  // tile then holds. {1, 1} when the whole plane already fits, which is
  // every geometry small enough to have worked before this existed.
  //
  // Tiling BELOW the attention is what makes it exact: the mid-block
  // attention is the only layer here whose receptive field is the whole
  // plane, and it runs above the split. Everything under it is convs, a
  // nearest upsample, and a per-pixel RMS over channels -- all local -- so
  // a tile carrying `halo` pixels of real neighbourhood produces interior
  // pixels that do not know they were tiled.
  //
  // Two depths. SHALLOW tiles the tail below the last temporal upsample, on
  // the split plane, with a halo of a few pixels. DEEP tiles everything
  // below the mid attention, on the LATENT plane: the halo is wider (~12
  // latent pixels of reach, 16 aligned) and so is the recompute, but the
  // head shrinks to the mid block and the store to one latent frame of it
  // per chunk -- what a box too small for the shallow floor needs.
  struct TailTiles {
    int    ty    = 1;
    int    tx    = 1;
    int    bh    = 0;     // tile height in tile-plane pixels, a multiple of 8
    int    bw    = 0;
    int    halo  = 0;     // tile-plane pixels of overlap per side, also 8s
    double frac  = 1.0;   // largest (tile + halo) / whole
    bool   deep  = false; // tile plane = latent (deep) or split plane
    // Laid out exactly as given (VPIPE_WAN_VAE_TILE), rather than planned
    // a band at a time from what is free as the decode goes.
    bool   forced = false;
  };
  // Tiles and halo are rounded UP TO 8 on purpose. The matrix-core 3x3
  // convolution tiles its destination 8x8 and declines a plane whose H or
  // W is not a multiple of 8 -- a tile that misses that is still correct,
  // because the im2col gather picks it up, but it gives away the ~3x that
  // path is worth on an M5. The split plane is itself a multiple of 8, so
  // aligned tiles keep every extent aligned, halo included.
  static int align8_(int v) { return ((v + 7) / 8) * 8; }
  // `fits` reports whether the returned grid actually gets under
  // `headroom`. When nothing does, the FINEST grid tried comes back with
  // `fits` false -- a refusal can then say what was already attempted
  // instead of quoting the whole-plane figure nobody was going to use.
  // The `ty` x `tx` grid as the decode would cut it (8-aligned tiles and
  // halo), or {1, 1} where that grid is all halo. VPIPE_WAN_VAE_TILE forces
  // one; a test asks for the figure it will be held to.
  TailTiles tail_tiles_for(int h8, int w8, int ty, int tx,
                           bool deep = false) const noexcept;
  TailTiles choose_tail_tiles_(int h8, int w8, std::size_t headroom,
                               bool frame_split, int latent_frames,
                               bool* fits = nullptr) const noexcept;
  // Pixels of halo one tile needs so its interior is bit-exact: one per
  // 3x3 conv below the split, each counted at its own resolution. The deep
  // one counts every conv below the mid block, in latent pixels.
  int tail_halo_() const noexcept;
  int tail_halo_deep_() const noexcept;
  // The split-plane size (the tail's input) for a latent of this size.
  void tail_plane_(int h8, int w8, int* hs, int* ws) const noexcept;

  const Config& config() const { return _cfg; }

  // ROWS one matmul2d dispatch may cover here, for an [M, K] source and an
  // [M, N] destination. Two independent limits, and the chunk is the
  // smaller:
  //
  //   kMmaMaxM   matmul2d corrupts output rows past ~2^19 regardless of
  //              width (the Krea-2 VAE's grey-bottom at 1024px).
  //   mma_row_band  no operand's byte span may pass 2^31.
  //
  // The row cap alone was the whole rule, and it bounds the DESTINATION --
  // cout is at most 384 here, so 2^19 rows is 200 MB and never close. The
  // source is the im2col band, and this VAE is the one in the tree that can
  // make it large: its causal conv3d has 27 taps where a 2D VAE has 9, so
  // K = 27 * cin reaches 10368. MEASURED at the decoder's own geometry, the
  // source span the row cap permits is 2.39 GB at 1280x720 (192-channel
  // level, K = 5184, 230400 rows per frame) and 2.72 GB at 1920x1080 --
  // both past the line, while 832x480 stays under it at 1.36 GB.
  //
  // VPIPE_WAN_VAE_MMA_MAX_M overrides the row cap only; the span band is
  // not negotiable, being a property of the kernels rather than a tuning
  // choice.
  static constexpr int kMmaMaxM = 1 << 19;
  static int mma_row_chunk(int M, int N, int K, int max_m = kMmaMaxM);

 private:
  // decode_peak_bytes in TERMS, by region: the mid block (with conv_in),
  // the up blocks to the last temporal upsample, and the tail below it --
  // so a whole, shallow-tiled or deep-tiled decode can each sum what it
  // holds at once. `mode` 0 whole, 1 shallow, 2 deep; `frac` a tile's
  // share of the plane where that mode tiles.
  enum { kMid = 0, kUp = 1, kTail = 2 };
  struct DecodeTerms {
    std::size_t carries[3] = {0, 0, 0};
    std::size_t pool[3]    = {0, 0, 0};
    std::size_t mid_el   = 0;   // one mid-block frame, elements
    std::size_t split_el = 0;   // one split-plane frame, elements
    std::size_t out_hw   = 0;   // output pixels per frame
    std::size_t t        = 1;   // frames in a steady chunk
  };
  static DecodeTerms decode_terms_(const Config& cfg, int h8, int w8,
                                   bool frame_split, double frac,
                                   int mode) noexcept;
  // What ONE tile of share `frac` holds while it runs the clip, and the
  // share of what is free one tile may take on a tight box.
  static std::size_t tile_working_bytes_(const Config& cfg, int h8, int w8,
                                         double frac, bool deep) noexcept;
  static double tile_share_() noexcept;
  MetalWanVae() = default;

  // A convolution stored as a dense-GEMM weight [Cout, K] (+ bias [Cout]),
  // with K covering every tap so one GEMM realizes the whole convolution:
  //   causal 3x3x3 : K = 27*Cin, flattened (kt,ky,kx,cin)  -- kt = 3
  //   plain 3x3    : K =  9*Cin, flattened (ky,kx,cin)     -- kt = 1
  //   time (3,1,1) : K =  3*Cin, flattened (kt,cin)        -- kt = 3, ks = 1
  //   1x1          : K =    Cin                            -- kt = 1, ks = 1
  // DIFFUSERS name -> the spelling this checkpoint actually uses. Empty
  // for a diffusers-named one; see wname_() in the .cc.
  std::unordered_map<std::string, std::string> _names;
  const std::string& wname_(const std::string& diffusers_name) const;

  struct Conv {
    metal_compute::SharedBuffer w, b;
    // HWIO twin for the hardware conv, [3,3,Cin',Cout] out-channel
    // fastest, where Cin' is 3*cin for a 3x3x3 conv run over its three
    // taps CONCATENATED channel-wise. Empty when that route is off.
    metal_compute::SharedBuffer whwio;
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
  // Where gather band `r0` of `rows` starts, so no band of a plane runs
  // under matmul2d's row floor (see the definition).
  std::size_t band_start_(std::size_t r0, std::size_t rows,
                          std::size_t ohw) const noexcept;
  void conv_frame_(Ctx& cx, const Conv& c,
                   const metal_compute::SharedBuffer* const taps[3],
                   const std::size_t tap_off[3],
                   const metal_compute::SharedBuffer& out,
                   std::size_t out_row0, int H, int W, int stride);

  // The same output frame on the gather-free hardware conv: a 3x3x3 conv
  // concatenates its three taps and runs ONE 2D conv over them. False
  // when the shape does not tile (grid not a multiple of 8, Cout neither
  // a multiple of 32 nor at most kSmallCoutMax, an int32 extent) and the
  // caller gathers instead.
  bool conv3x3_hw_(Ctx& cx, const Conv& c,
                   const metal_compute::SharedBuffer* const taps[3],
                   const std::size_t tap_off[3],
                   const metal_compute::SharedBuffer& out,
                   std::size_t out_row0, int H, int W);

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

  // The decode's two halves, either side of the last temporal upsample
  // (`split_index_`): the head of one latent frame to the split plane,
  // and the tail from there to RGB. Shared by the whole-plane decode and
  // the tile-outer one, which only runs them in a different order.
  std::size_t split_index_() const noexcept;
  // The head in its two parts: the mid block (to the latent plane's one
  // frame) and the up blocks to the split. A deep tile runs the second
  // on its own patch of the first's output.
  const metal_compute::SharedBuffer*
  mid_chunk_(Ctx& cx, const metal_compute::SharedBuffer& z, int T, int f,
             int h8, int w8, std::vector<Carry>& carry, std::size_t& ci);
  const metal_compute::SharedBuffer*
  up_to_split_(Ctx& cx, const metal_compute::SharedBuffer& in, int& t,
               int& H, int& W, std::vector<Carry>& carry, std::size_t& ci);
  const metal_compute::SharedBuffer*
  head_chunk_(Ctx& cx, const metal_compute::SharedBuffer& z, int T, int f,
              int h8, int w8, std::vector<Carry>& carry, std::size_t& ci,
              int& t, int& H, int& W);
  metal_compute::SharedBuffer*
  tail_(Ctx& cx, const metal_compute::SharedBuffer& in, int nt, int& th,
        int& tw, std::vector<Carry>& carry, std::size_t& ci);
  // The tail over a chunk's `t` split-plane frames, a frame at a time.
  metal_compute::SharedBuffer*
  tail_frames_(Ctx& cx, const metal_compute::SharedBuffer& x, int t, int& H,
               int& W, std::vector<Carry>& carry, std::size_t& ci);
  // A decode whose tail does not fit whole: the head over the clip into a
  // store, then each tile through every frame. See the definition.
  bool decode_tile_outer_(const metal_compute::SharedBuffer& z, int T, int h8,
                          int w8, const TailTiles& tiles, std::size_t col_cap,
                          const FrameSink& on_frame, std::string* err);

  // Copy the trailing min(2, t) frames of `x` into `carry` for the next
  // chunk. A GPU-side copy: the frames are still only on the GPU timeline.
  void save_carry_(Ctx& cx, Carry& carry, const metal_compute::SharedBuffer& x,
                   int t, std::size_t hw, int cin);

  // ---- kernels ---------------------------------------------------------
  metal_compute::ComputeLibrary _lib_gemm, _lib_elt, _lib_rms, _lib_sdpa;
  metal_compute::ComputeFunction _fn_gemm_bias, _fn_rms, _fn_mul_sigmoid,
      _fn_residual, _fn_clamp, _fn_sdpa, _fn_upsample, _fn_copy,
      _fn_copy_rect;
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
  int  _mma_max_m = kMmaMaxM;

  // The gather-free hardware conv (MPP convolution2d) over a 64- or a
  // 32-channel destination tile, and the per-pixel small-cout conv for the
  // head's 3-channel output, which neither tile can serve.
  // Each has an ACCUMULATING twin (out += conv), so a causal 3x3x3 conv
  // runs as one call per temporal tap with no three-frame concat.
  static constexpr int kSmallCoutMax = 32;
  metal_compute::ComputeLibrary _lib_convhw;
  metal_compute::ComputeFunction _fn_conv_hw64, _fn_conv_hw32,
      _fn_conv_small_cout, _fn_conv_hw64_acc, _fn_conv_hw32_acc,
      _fn_conv_small_cout_acc;
  bool _use_hwconv = false;

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

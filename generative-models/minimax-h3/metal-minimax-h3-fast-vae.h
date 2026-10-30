#ifndef GENERATIVE_MODELS_MINIMAX_H3_METAL_MINIMAX_H3_FAST_VAE_H
#define GENERATIVE_MODELS_MINIMAX_H3_METAL_MINIMAX_H3_FAST_VAE_H

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/shared/ane-ffn.h"
#include "generative-models/shared/dit-flops.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

class I8GemmContext;
class WeightSet;

// MiniMax-H3's FAST video decoder: the non-causal 24-layer decoder that
// Open-H3-Regenerate-2K ships in `vae_fast/` (from NVIDIA's StreamRender-H3
// fast H3 VAE work), a LOSSY stand-in for the released ViT decoder
// (MetalMiniMaxH3VideoVae) -- the release puts it ~30 dB from it.
//
// Its blocks ARE the released decoder's (24 of the 36, distilled): RMSNorm,
// a fused per-head q|k|v with biases, q/k RMS without affine, rotary on 48
// of 64 channels, a learned per-channel residual gate, a gate-first SwiGLU
// x4. What differs is how tokens meet:
//
//   * NO TILES. The whole clip is one token grid, a token a latent voxel,
//     and attention is LOCAL: a window of 5 latent frames x 16 x 16 tokens,
//     the odd layers' windows shifted by 2 frames and 8 tokens (the grid
//     padded at the front, Video-Swin style), so nothing is decoded twice
//     and nothing is cross-faded.
//   * GLOBAL TOKENS. Every native block of 5 latent frames owns five (four
//     learned registers and a zero one); a window attends the global tokens
//     of the blocks its frames touch, and a block's global tokens attend
//     every token of the block.
//   * WINDOW-LOCAL ROTARY: ids from a token's place in its window -- time
//     normalized by 5, space by 16 -- so nothing depends on the absolute
//     position and any length decodes the same way.
//
// Output: ImageNet-normalized pixels, the first 3 of the 4 frames of every
// block's first latent dropped -- 5 latents give 17 frames, as the released
// decoder's chunks do, so the two emit the same frame count for a latent.
//
// THE WORK. ~2 * 67M FLOPs a token a layer in its GEMMs: a 2K clip of 37
// latents (532,800 tokens) is ~1.7 PFLOP, ~3.7x less than the released
// decoder's overlapping 256-pixel tiles and 7-latent chunks at 36 layers.
//
// HOW IT RUNS here. The residual stream is f32 for the whole clip (as the
// release keeps it), GEMM operands bf16. A layer's attention goes a
// TEMPORAL WINDOW at a time: that slab's rows are normed and projected in
// bands, their q / k / v scattered window by window into packed pools (no
// padding: a window holds only its real tokens, plus its blocks' global
// keys), the windows batched by shape into the steel / matrix-core flash
// kernel, the output gathered back to token order and projected. A
// block's tokens' keys also land in a two-block ring, where the block's
// global tokens attend them once its last frame is in. The feed-forward
// then runs over every row at once (so the ANE can take a share of them).
// Memory: the f32 stream and a bf16 row buffer for the clip (12 bytes a
// token per channel pair -- 6.5 GB for that 2K clip) plus one slab's pools.
//
// TIERS: matmul2d GEMMs and the matrix-core attention on an M5; steel's
// 64x64 GEMM and flash attention elsewhere; i8_gemm at any tier (the
// block projections held in the int8 layout from fast up); the ANE taking
// a share of every feed-forward's rows (ane_ffn), as the released
// decoder's tier does.
class MetalMiniMaxH3FastVae {
 public:
  struct Config {
    int   z_channels   = 24;
    int   out_channels = 3;
    int   dim          = 2048;
    int   n_heads      = 32;
    int   head_dim     = 64;
    int   n_layers     = 24;
    int   ffn_inner    = 8192;
    int   patch        = 16;
    int   patch_t      = 4;
    int   n_register   = 4;
    int   window       = 16;     // tokens a side
    int   block        = 5;      // latent frames a window / native block
    int   t_shift      = 2;      // odd layers' temporal shift
    int   frames_to_trim = 3;    // of a block's first latent's patch_t
    double time_norm      = 5.0;
    double rope_theta     = 100.0;
    double rope_dim_ratio = 0.75;
    float  norm_eps       = 1e-5f;
    // The released VAE's latent statistics: the decoder un-whitens.
    std::vector<float> latents_mean, latents_std;

    // i8_gemm: -1 off, 0 accurate, 1 fast, 2 token, 3 token_plain (the
    // stages' tier names, by number). Matrix cores only.
    int   i8_tier    = -1;
    // The ANE tier (LOSSY, fp16): its share of every feed-forward's rows,
    // 0 auto-balancing; `ane_layers` caps the blocks, 0 all.
    bool  ane_ffn    = false;
    float ane_rows   = 0.0f;
    int   ane_layers = 0;

    int rope_dim() const { return (int)((double)head_dim * rope_dim_ratio); }
    int patch_elems() const { return out_channels * patch_t * patch * patch; }
    int global_tokens() const { return n_register + 1; }
  };

  // The file to open, from what a stage was given: the .safetensors
  // itself; a directory holding `vae_fast/<one file>` (the
  // Open-H3-Regenerate-2K layout); or a directory with exactly one
  // .safetensors. Empty, with a reason, else.
  static std::string resolve_weights(const std::string& path,
                                     std::string* err = nullptr);

  // The config off the weights (layer count, widths) and the released
  // VAE's config.json (`vae_dir` as MetalMiniMaxH3VideoVae resolves it:
  // latent statistics, heads, rotary). False with a reason when the file
  // is not this decoder.
  static bool config_from(const WeightSet& ws, const std::string& vae_dir,
                          Config* out, std::string* err = nullptr);

  static std::unique_ptr<MetalMiniMaxH3FastVae>
  load(std::shared_ptr<WeightSet> ws, metal_compute::MetalCompute* mc,
       const Config& cfg, std::string* err = nullptr);

  ~MetalMiniMaxH3FastVae();

  // The i8 tier a load with `i8_tier` runs on a box with (or without)
  // matrix cores -- the int8 context's own reading of the config and
  // VPIPE_I8_GEMM / VPIPE_I8_TIER / VPIPE_I8_TOKEN_ROT -- so a plan and the
  // load agree. -1 off, 0 accurate, 1 fast, 2 token, 3 token_plain.
  static int effective_i8_tier(int i8_tier, bool matrix_cores);

  // Pixel frames `T` latent frames decode to: 4 T less 3 a native block.
  static int decoded_frames(const Config& c, int T);
  int decoded_frames(int T) const { return decoded_frames(_cfg, T); }

  // Decode `z`, the WHITENED latent channel-first [z_channels, T, h, w]
  // f32, into channel-first [out_channels, decoded_frames(T), h*16, w*16]
  // ImageNet-normalized pixels, bf16. Empty on failure with a reason in
  // `err`; empty with `err` untouched on a stop.
  metal_compute::SharedBuffer
  decode(const float* z, int T, int h, int w, std::string* err = nullptr);

  // Device bytes decode() allocates beside the weights at its high water
  // for a [T, h, w] latent, its output included. From the config alone,
  // so a plan can ask before load; the instance method is this with
  // _cfg.
  static std::size_t scratch_bytes(const Config& c, int T, int h, int w);
  std::size_t scratch_bytes(int T, int h, int w) const
  {
    return scratch_bytes(_cfg, T, h, w);
  }
  // Every weight as held: bf16 projections (or their int8 layout), f32
  // norms, gates and the head's bias. From the file's headers and a tier,
  // for a plan.
  static std::size_t weight_bytes(const std::string& file, int i8_tier);
  std::size_t weight_bytes() const noexcept { return _weight_bytes; }

  // The ANE tier's module bytes (one runtime-weight module for every
  // layer), for the plan's claim.
  static std::size_t ane_runtime_bytes(const Config& c) noexcept;

  // Progress after each layer: (done, total) layers.
  void set_progress(std::function<void(int, int)> fn)
  {
    _progress = std::move(fn);
  }
  // A cooperative stop, checked between layers (each commits and waits).
  void set_stop(std::function<bool()> fn) { _stop = std::move(fn); }

  const Config& config() const noexcept { return _cfg; }
  bool matrix_cores() const noexcept { return _use_mma2; }
  bool nax_attention() const noexcept { return _attn_nax; }
  // The i8 tier that engaged (-1 none), and the GEMMs each route ran in
  // the last decode.
  int i8_tier() const noexcept { return _i8_tier; }
  long long i8_gemms() const noexcept { return _i8_gemms; }
  long long dense_gemms() const noexcept { return _dense_gemms; }
  bool ane_attempted() const noexcept { return _ane_tried; }
  bool ane_armed() const noexcept { return _ane != nullptr; }
  bool ane_gpu_only() const noexcept
  {
    return _ane != nullptr && _ane->gpu_only();
  }
  // The ANE's rows over the last decode (0 when it took none).
  long long ane_rows_taken() const noexcept { return _ane_rows; }
  const GpuFlopsTally& gpu_flops() const noexcept { return _flops; }

 private:
  MetalMiniMaxH3FastVae() = default;

  // A projection [N, K]: bf16 `w`, or -- under a tier that holds it -- its
  // int8 layout (gc > 0); a bf16 bias.
  struct Proj {
    metal_compute::SharedBuffer w, b;
    metal_compute::SharedBuffer codes, scales;
    int gc = 0;
    int n = 0, k = 0;
  };
  struct Block {
    metal_compute::SharedBuffer n1, n2, s1, s2;     // f32 [D]
    Proj qkv, out, w1, w2;
  };

  // One layer's windows over the clip, for one shift: per temporal
  // window (a slab), where each token's q and k rows go, the classes of
  // windows batched into one attention dispatch, and the global keys each
  // window takes.
  struct Slab {
    int t_lo = 0, t_hi = 0;           // latent frames [t_lo, t_hi)
    int q_rows = 0, k_rows = 0;
    std::vector<int> qrow, krow;      // per token of the slab
    struct Class { int q0, k0, count, qL, kL; };
    std::vector<Class> classes;
    std::vector<int> gdst, gsrc;      // global keys into the k/v pools
    // Native blocks whose last frame is in this slab: their global
    // tokens attend once it has run.
    std::vector<int> done;
  };
  struct Geometry {
    int T = 0, H = 0, W = 0, nb = 0;
    std::vector<Slab> slabs[2];       // [shift]
    metal_compute::SharedBuffer tables;   // every slab's qrow|krow|gdst|gsrc
    std::vector<std::size_t> off[2];      // per slab: tables' int offset
  };
  bool build_geometry_(int T, int h, int w, std::string* err);

  bool load_(std::string* err);
  bool bind_kernels_(std::string* err);
  Proj proj_(const std::string& name, int N, int K, std::string* err);

  // y = x W^T + b over M rows, on the int8 route, matmul2d or steel.
  void gemm_(metal_compute::ComputeEncoder& enc,
             const metal_compute::SharedBuffer& x, std::size_t xe,
             const Proj& p, const metal_compute::SharedBuffer& y,
             std::size_t ye, int M);
  void dense_(metal_compute::ComputeEncoder& enc,
              const metal_compute::SharedBuffer& x, std::size_t xe,
              const metal_compute::SharedBuffer& w,
              const metal_compute::SharedBuffer* bias,
              const metal_compute::SharedBuffer& y, std::size_t ye, int M,
              int N, int K);
  void norm_(metal_compute::ComputeEncoder& enc,
             const metal_compute::SharedBuffer& x, std::size_t row0,
             const metal_compute::SharedBuffer& w,
             const metal_compute::SharedBuffer* b,
             const metal_compute::SharedBuffer& out, std::size_t orow0,
             int rows, bool ln);
  void gated_(metal_compute::ComputeEncoder& enc, std::size_t row0,
              const metal_compute::SharedBuffer& s,
              const metal_compute::SharedBuffer& y, std::size_t yrow0,
              int rows);
  // The flash kernel over `count` windows of qL queries x kL keys, token-
  // major ([rows, heads * 64]) pools from rows q0 / k0.
  void attend_(metal_compute::ComputeEncoder& enc,
               const metal_compute::SharedBuffer& q, std::size_t q0,
               const metal_compute::SharedBuffer& k,
               const metal_compute::SharedBuffer& v, std::size_t k0,
               const metal_compute::SharedBuffer& o, int count, int qL,
               int kL);
  void copy_rows_(metal_compute::ComputeEncoder& enc,
                  const metal_compute::SharedBuffer& src,
                  const metal_compute::SharedBuffer& dst,
                  const metal_compute::SharedBuffer* src2,
                  const metal_compute::SharedBuffer* dst2,
                  const metal_compute::SharedBuffer& smap, std::size_t s_off,
                  const metal_compute::SharedBuffer* dmap, std::size_t d_off,
                  int d0, int n);
  bool ensure_scratch_(int T, int h, int w);

  // ANE: built at the first decode from its row count.
  bool ane_setup_(int rows);
  bool ane_eligible_(int L) const;
  void ane_stage_(int L);

  std::shared_ptr<WeightSet>   _ws;
  metal_compute::MetalCompute* _mc = nullptr;
  Config _cfg;

  std::vector<Block> _blocks;
  metal_compute::SharedBuffer _embed_a, _embed_c;   // f32 [D, Z], [D]
  metal_compute::SharedBuffer _registers;           // f32 [n_register, D]
  metal_compute::SharedBuffer _ln_w, _ln_b;         // f32 [D]
  Proj _head;                                       // [patch_elems, D]
  metal_compute::SharedBuffer _head_bias;           // f32
  metal_compute::SharedBuffer _rope;                // f32 [2][3][16][8]
  std::size_t _weight_bytes = 0;

  std::unique_ptr<I8GemmContext> _i8;
  int _i8_tier = -1;
  long long _i8_gemms = 0, _dense_gemms = 0;
  metal_compute::SharedBuffer _wdeq;    // a held layout widened, < floor

  struct Scratch {
    int T = -1, h = -1, w = -1;
    metal_compute::SharedBuffer x;      // f32 [rows, D]
    metal_compute::SharedBuffer hb;     // bf16 [rows, D]
    metal_compute::SharedBuffer qkv;    // bf16 [band, max(3D, F)]
    metal_compute::SharedBuffer ff;     // bf16 [band, 2F]
    metal_compute::SharedBuffer y;      // bf16 [band, D]
    metal_compute::SharedBuffer qp, kp, vp, op;   // one slab's pools
    metal_compute::SharedBuffer ringk, ringv;     // two blocks
    metal_compute::SharedBuffer gq, gk, gv, go;   // global rows
    metal_compute::SharedBuffer ident;            // int [max(band, G)]
    int band = 0, ring_rows = 0;
  };
  Scratch _s;
  Geometry _geo;

  metal_compute::ComputeLibrary _lib_gemm, _lib_elt, _lib_regen, _lib_mma,
      _lib_attn, _lib_attn_nax;
  metal_compute::ComputeFunction _fn_gemm, _fn_mma, _fn_mma_wide,
      _fn_bias_add, _fn_swiglu, _fn_embed, _fn_norm, _fn_gated, _fn_post,
      _fn_copy, _fn_unpatch;
  // The flash kernel per (align_Q, align_K).
  metal_compute::ComputeFunction _fn_attn[2][2];
  bool _use_mma2 = false;
  bool _attn_nax = false;
  int  _mma_min_m = 64;

  std::unique_ptr<AneFeedForward> _ane;
  bool _ane_tried = false;
  long long _ane_rows = 0;

  std::function<void(int, int)> _progress;
  std::function<bool()> _stop;
  GpuFlopsTally _flops;
};

}  // namespace genai
}  // namespace vpipe

#endif

#ifndef GENERATIVE_MODELS_VOSR_METAL_VOSR_TRANSFORMER_H
#define GENERATIVE_MODELS_VOSR_METAL_VOSR_TRANSFORMER_H

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/shared/i8-gemm.h"
#include "generative-models/shared/metal-sage-attention.h"
#include "generative-models/shared/metal-sol-attention.h"
#include "generative-models/shared/sage-attention.h"
#include "generative-models/shared/sol-attention.h"
#include "generative-models/shared/dit-block-progress.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

class WeightSet;   // generative-models/weight-set.h

// VOSR's LightningDiT: the one-step, VISION-ONLY image restorer.
//
// A single-stream DiT, 36 blocks at 1536 wide. Per block: adaLN-modulated
// self-attention with 2D RoPE and per-head qk RMS-norm, then CROSS-
// attention to the DINOv2 tokens (ungated -- it is a plain residual, not
// a gated one), then an adaLN-modulated SwiGLU MLP. Six modulation
// vectors per block come from one timestep embedding through a shared
// t_block plus a per-block scale_shift_table.
//
// WHAT MAKES IT A RESTORER rather than a generator is the input: the
// patch embedding reads 32 channels, the LOW-QUALITY latent concatenated
// with the noise latent. There is no text anywhere in the model, and
// nothing to condition on but the picture.
//
// THE FLOW LOOP LIVES HERE, not in the caller, because tiling and
// sampling are entangled: a tiled run blends per-tile VELOCITIES under
// one shared noise field and only then takes the step, so a caller
// driving tiles itself would have to own the sampler too.
class MetalVosrTransformer {
 public:
  struct Config {
    int dim          = 1536;
    int depth        = 36;
    int n_heads      = 24;
    int head_dim     = 64;
    int patch        = 2;
    int latent_ch    = 16;    // the VAE's z_dim; in_channels is 2x this
    int ffn          = 4096;  // int(2/3 * mlp_ratio * dim)
    int enc_dim      = 1024;  // DINOv2 width
    int enc_mlp      = 4608;  // dim * encdim_ratio
    int freq_dim     = 256;   // timestep sinusoid width
    // The patch grid the RoPE was trained at (resolution / 8 / patch).
    // Positions at other grids are RESCALED into this range rather than
    // extrapolated, which is what the reference does and why a 4x
    // upscale does not fall off the end of the table.
    int train_grid   = 32;
    float norm_eps   = 1e-6f;
    // The conditioning LayerNorm is a bare `nn.LayerNorm`, so it carries
    // torch's DEFAULT epsilon, not the 1e-6 the RMSNorms were built
    // with. A tenth of a percent on every conditioning row, which is
    // small and is not zero.
    float cond_norm_eps = 1e-5f;
    double rope_theta = 10000.0;
    // The DINOv2 layer the checkpoint was trained against. Carried here
    // only so the stage can configure the tower to match; the DiT itself
    // never reads it.
    int enc_layer    = 17;
    int dinov2_size  = 448;

    // ---- acceleration, all four opt-in and independent ------------
    //
    // The restorer is head_dim 64 where every other DiT here is 128,
    // which is the only thing that made this more than four flag
    // assignments: the steel entries exist at bd64 with the SAME tiles
    // (ALU 32/16, NAX 64/32), so nothing about the geometry changes,
    // but Sol had asserted head_dim 128 since it shipped.
    bool i8_gemm = false;
    // Whether the kernel tuners MEASURE (the accel key `autotune`,
    // accel-keys.h): false takes the choice believed fastest on this
    // GPU, timing nothing.
    bool autotune = true;
    sage::Config sage;
    sol::Config  sol;
  };

  // Read a VOSR `args.json` (the file beside the checkpoint) into `cfg`.
  // Keys it does not carry keep their defaults. False when there is no
  // readable args.json, or when it describes a model this class does not
  // implement (a multi-step checkpoint, an SD2 autoencoder, an auxiliary
  // time condition) -- refusing is the point: every one of those loads
  // to the wrong answer rather than to an error.
  static bool read_config(const std::string& dir, Config* cfg,
                          std::string* why = nullptr);

  // THE TILE A RUN SHOULD TAKE when the graph did not choose one, in
  // LATENT cells; 0 when the picture already fits the grid the weights
  // were distilled at and nothing needs splitting.
  //
  // Pure arithmetic, and public, because it is a decision a stage has to
  // make BEFORE it builds a request and a test has to be able to pin
  // without a checkpoint. See the note on RestoreRequest::tile for why
  // the answer is the trained grid rather than "as large as fits".
  static int default_tile(int lh, int lw, const Config& cfg);

  // The DIRECTORY the weights sit in under a VOSR checkpoint root:
  // `checkpoints/` beside `args.json`, or that directory itself. Empty
  // when neither holds a readable safetensors. This is what the memory
  // plan is asked about, because a plan names directories.
  static std::string weights_dir(const std::string& root);

  // The FILE to load, which is not the same question. VOSR names its
  // checkpoint `ema_model.safetensors`, and a directory holding one
  // freely-named file is not a layout the checkpoint opener globs -- it
  // recognises `model.safetensors` and the diffusers spellings and
  // nothing else. So the loader is handed the file, the way a Comfy-Org
  // repack's components are. Empty when there is none.
  static std::string weights_path(const std::string& root);

  static std::unique_ptr<MetalVosrTransformer>
  load(const std::string& model_dir, metal_compute::MetalCompute* mc,
       const Config& cfg, std::string* err = nullptr);

  static std::unique_ptr<MetalVosrTransformer>
  load(std::shared_ptr<WeightSet> ws, metal_compute::MetalCompute* mc,
       const Config& cfg, std::string* err = nullptr);

  ~MetalVosrTransformer();

  struct RestoreRequest {
    // The low-quality latent, channel-first f32 [latent_ch, lh, lw],
    // WHITENED -- exactly what `vae-encode` emits.
    const float* lq = nullptr;
    int lh = 0, lw = 0;
    // DINOv2 tokens, bf16 [cond_rows, enc_dim]. The grid is assumed
    // square (it is: the tower squashes its input), which is what makes
    // a tile's slice of it computable.
    const void* cond = nullptr;
    int cond_rows = 0;
    int steps = 1;
    std::uint64_t seed = 0;
    // The starting noise, channel-first f32 over the same shape as `lq`.
    // Null draws it from `seed`, which is what a graph does. Supplying
    // it is how a REFERENCE COMPARISON is possible at all: two runtimes
    // cannot agree on a pseudo-random field, so the check has to hold
    // it fixed and compare what the model does with it.
    const float* noise = nullptr;
    // Latent-space tile side and overlap. 0 runs the whole picture in
    // one pass.
    //
    // ONE PASS IS ONLY THE REFERENCE'S PATH AT THE TRAINED GRID, and
    // this is the sentence to read before changing it. The reference's
    // one-step sampler calls a forward that applies `self.feat_rope`
    // unconditionally -- a table built once for the training grid -- so
    // at any other token count its `q * freqs_cos` does not broadcast
    // and the run cannot happen at all. Its answer for a bigger picture
    // is to tile, and its tile arithmetic exists to put every tile back
    // on that grid. We instead build a table for whatever grid we are
    // handed, which is the reference's `forward_flexible`, a function it
    // ships and never calls here: the rescale then puts every other
    // token on a half-integer position the weights never saw, two axes
    // at period two, which is a checkerboard over exactly the content
    // the model has to synthesise rather than pass through. Reported
    // from the field on faces at a 1024 output, clean at 512 and clean
    // tiled. So default_tile() above, not 0.
    //
    // A tiled run does still differ from the reference in one way: it
    // CROPS the conditioning grid per tile where the reference re-runs
    // its tower on the tile's pixels.
    int tile = 0;
    int tile_overlap = 0;
    // False ABORTS. `step` is one-based and counts finished steps.
    std::function<bool(int step, int total)> progress;
    // Two-phase, so the report lands on the GPU's clock rather
    // than the encode thread's. See shared/dit-gpu-progress.h.
    DitBlockProgressFn block_progress;
  };

  // Run the restoration. `out` receives the channel-first f32 latent
  // [latent_ch, lh, lw], still whitened -- `vae-decode` un-whitens it.
  // False after warning through nothing: the caller reports.
  bool restore(const RestoreRequest& req, std::vector<float>* out,
               std::string* err = nullptr);

  const Config& config() const noexcept { return _cfg; }
  std::uint64_t resident_bytes() const noexcept { return _bytes; }

  // The latent side the reference trained on, for the log line a caller
  // writes before it spends a minute.
  int train_grid() const noexcept { return _cfg.train_grid * _cfg.patch; }

 private:
  MetalVosrTransformer() = default;

  struct Block {
    metal_compute::SharedBuffer n1, n2;              // RMSNorm [dim]
    metal_compute::SharedBuffer sst;                 // [6, dim]
    metal_compute::SharedBuffer qkv_w, qkv_b;        // [3*dim, dim]
    metal_compute::SharedBuffer qn, kn;              // RMSNorm [head_dim]
    metal_compute::SharedBuffer proj_w, proj_b;
    metal_compute::SharedBuffer cq_w, cq_b, ck_w, ck_b, cv_w, cv_b;
    metal_compute::SharedBuffer cqn, ckn;
    metal_compute::SharedBuffer cproj_w, cproj_b;
    metal_compute::SharedBuffer w12_w, w12_b;        // [2*ffn, dim]
    metal_compute::SharedBuffer w3_w, w3_b;          // [dim, ffn]
  };

  // Everything one forward over `T` tokens needs. Allocated once per
  // restore() and reused for every tile of every step -- the tiles are
  // uniform, so the largest is the only size that matters.
  struct Scratch {
    int T = 0, K = 0;
    metal_compute::SharedBuffer pix, x, nrm, mod, qkv, q, k, v;
    metal_compute::SharedBuffer qt, kt, vt, at, att, o, ff, ffo;
    metal_compute::SharedBuffer fsil, fmod;
    metal_compute::SharedBuffer ck, cv, ckt, cvt;
    metal_compute::SharedBuffer outp;
  };

  bool load_weights_(WeightSet& ws, std::string* err);
  bool alloc_scratch_(Scratch* s, int T, int K) const;
  // y[M,N] = x[M,K] @ w[N,K]^T (+ b), through whichever dense tile this
  // box has. ONE spelling, because the model has three call sites and
  // they must not drift about which kernel or which bias convention.
  void encode_gemm_(metal_compute::ComputeEncoder& enc,
                    const metal_compute::SharedBuffer& x, std::size_t xe,
                    const metal_compute::SharedBuffer& w,
                    const metal_compute::SharedBuffer& b,
                    const metal_compute::SharedBuffer& y, std::size_t ye,
                    int M, int N, int K, bool bias);
  // The RoPE tables for a `gh` x `gw` patch grid, f32 [gh*gw, head_dim].
  void build_rope_(int gh, int gw, metal_compute::SharedBuffer& cos_out,
                   metal_compute::SharedBuffer& sin_out) const;
  // Project DINOv2 tokens into the DiT width: LayerNorm then the shared
  // two-layer GELU MLP. [rows, enc_dim] bf16 -> [rows, dim] bf16.
  metal_compute::SharedBuffer project_cond_(const void* cond, int rows);
  // The timestep conditioning for one step: `c` (the final layer's) and
  // `c0` (the blocks' six-way table), both bf16.
  bool build_time_(float t, metal_compute::SharedBuffer* c,
                   metal_compute::SharedBuffer* c0);
  // One velocity evaluation over a tile. `pix` must already hold the
  // packed [T, 2*latent_ch*p*p] patch rows.
  bool forward_(Scratch& s, const metal_compute::SharedBuffer& zc, int K,
                const metal_compute::SharedBuffer& c,
                const metal_compute::SharedBuffer& c0, int gh, int gw,
                const DitBlockProgressFn& block_progress,
                std::string* err);

  metal_compute::MetalCompute* _mc = nullptr;
  Config                       _cfg;
  std::uint64_t                _bytes = 0;

  metal_compute::SharedBuffer _patch_w, _patch_b;   // x_embedder.proj
  metal_compute::SharedBuffer _t0_w, _t0_b, _t2_w, _t2_b;   // t_embedder
  metal_compute::SharedBuffer _tblock_w, _tblock_b;
  metal_compute::SharedBuffer _cln_w, _cln_b;       // layer_norm (enc_dim)
  metal_compute::SharedBuffer _ca1_w, _ca1_b, _ca2_w, _ca2_b;   // mlp_ca
  metal_compute::SharedBuffer _fmod_w, _fmod_b;     // final adaLN
  metal_compute::SharedBuffer _fnorm;               // RMSNorm [dim]
  metal_compute::SharedBuffer _fout_w, _fout_b;     // [p*p*C, dim]
  std::vector<Block>          _blocks;

  // RoPE tables, rebuilt when the grid changes.
  metal_compute::SharedBuffer _rcos, _rsin;
  int _rope_gh = 0, _rope_gw = 0;

  // Steel flash, rebuilt when a (qL, kL) changes. Self and cross are
  // tracked apart because only the cross one moves with the tile.
  metal_compute::SharedBuffer  _ap_self, _ap_cross;
  metal_compute::ComputeFunction _fn_attn_self, _fn_attn_cross;
  // The int8-QK twins, built only when sage_attn asked and the box has
  // the matrix-core entry it lives in.
  metal_compute::ComputeFunction _fn_attn_self_i8, _fn_attn_cross_i8;
  int _attn_q = 0, _attn_kv = 0;
  bool build_attn_(int qL, int kL, bool self);
  // The flash kernel's tiles for the arm this box took. head_dim 64 and
  // 128 tile identically, so these follow the KERNEL and not the width.
  int attn_bq_() const { return _use_attn_nax ? 64 : 32; }
  int attn_bk_() const { return _use_attn_nax ? 32 : 16; }

  // Matrix cores: the matmul2d dense GEMM and the NAX flash attention,
  // both on by default where the GPU has them, as in every sibling DiT.
  // VPIPE_VOSR_NO_MMA2 / VPIPE_VOSR_NO_ATTN_NAX are the A/Bs.
  bool _use_mma2 = false, _use_attn_nax = false;
  metal_compute::ComputeLibrary  _lib_dense_mma, _lib_attn_nax;
  metal_compute::ComputeFunction _fn_gemm_mma, _fn_bias_rows;
  // Dynamic-int8 GEMMs, and the two attention accelerations. Null unless
  // the config asked and the box can; every use is guarded.
  std::unique_ptr<I8GemmContext>       _i8;
  std::unique_ptr<MetalSageAttention>  _sage;
  std::unique_ptr<MetalSolAttention>   _sol;

  metal_compute::ComputeLibrary _lib_gemm, _lib_elt, _lib_rms, _lib_rope,
      _lib_attn;
  metal_compute::ComputeFunction _fn_gemm, _fn_rms, _fn_ln, _fn_adaln,
      _fn_gated, _fn_residual, _fn_hslice, _fn_transpose, _fn_trope,
      _fn_qknorm, _fn_swiglu, _fn_silu, _fn_gelu;

  std::shared_ptr<WeightSet> _ws;
};

}  // namespace genai
}  // namespace vpipe

#endif

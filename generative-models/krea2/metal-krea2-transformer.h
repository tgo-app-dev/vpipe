#ifndef GENERATIVE_MODELS_KREA2_METAL_KREA2_TRANSFORMER_H
#define GENERATIVE_MODELS_KREA2_METAL_KREA2_TRANSFORMER_H

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

class MetalLlamaWeights;   // fwd: kept source mmap for streaming-blocks mode
class I8GemmContext;       // fwd (shared/i8-gemm.h)

// Krea-2-Turbo denoiser (Krea2Transformer2DModel): a single-stream MMDiT
// flow-matching transformer, run in f16 on the metal-compute backend.
//
// Text conditioning enters as a stack of hidden states tapped from 12 layers
// of the Qwen3-VL text encoder (see the M2 encoder tap). A small text-fusion
// tower (2 layerwise blocks over the layer axis -> a Linear(12->1) projector
// -> 2 refiner blocks over the token axis) collapses that stack into one
// sequence of text features; txt_in projects it to the transformer width.
// (The image path -- img_in, the 28 adaLN/RoPE/gated transformer blocks and
// the final layer -- is added incrementally; M3a implements the text tower.)
class MetalKrea2Transformer {
 public:
  struct Config {
    int   hidden          = 6144;   // attention_head_dim * num_attention_heads
    int   n_heads         = 48;
    int   n_kv_heads      = 12;     // GQA
    int   head_dim        = 128;
    int   ffn             = 16384;
    int   n_layers        = 28;
    int   in_channels     = 64;     // 16 latent ch * patch 2 * patch 2
    int   timestep_dim    = 256;
    // Text-fusion tower.
    int   text_hidden     = 2560;
    int   text_heads      = 20;
    int   text_kv_heads   = 20;     // no GQA in the fusion blocks
    int   text_ffn        = 6912;
    int   n_text_layers   = 12;     // tapped encoder layers
    int   n_layerwise     = 2;
    int   n_refiner       = 2;
    float norm_eps        = 1e-5f;
    float rope_theta      = 1000.0f;
    int   rope_t          = 32;     // axes_dims_rope
    int   rope_h          = 48;
    int   rope_w          = 48;
    // Accelerated mode (LOSSY, opt-in): dynamic-int8 GEMMs for the big
    // block matmuls (see shared/i8-gemm.h). VPIPE_I8_GEMM overrides.
    bool  i8_gemm         = false;
    int   text_head_dim() const { return text_hidden / text_heads; }  // 128
  };

  // `stream_blocks` (memory-bounded AWQ calibration on small boxes): keep the
  // 28 main transformer_blocks OFF the wired heap -- forward_dit loads each from
  // the retained source mmap just before use and frees it after, so peak block
  // memory is ~1 block (~0.85 GB bf16) instead of all 28 (~24 GB). The small
  // parts (text-fusion tower, conditioning, final layer) are still preloaded.
  static std::unique_ptr<MetalKrea2Transformer>
  load(const std::string& model_dir, metal_compute::MetalCompute* mc,
       const Config& cfg, bool stream_blocks = false);

  ~MetalKrea2Transformer();   // out-of-line: _stream_wts is a fwd-declared type

  // Cooperative-stop hook for the (long, disk-bound) streaming-blocks forward:
  // forward_dit polls this before each main block and bails early (returns
  // empty) when it returns true. Used by the AWQ calibration collector to honor
  // a pipeline stop within ~one block. No effect on the preloaded path.
  void set_stream_stop(std::function<bool()> stop) { _stream_stop = std::move(stop); }

  // M3a: run the text-fusion tower + txt_in on the (text_seq, n_text_layers,
  // text_hidden) f16 encoder-tap stack -> the (text_seq, hidden) fused text
  // conditioning fed to the DiT's joint sequence. `ehs` is laid out
  // [token][layer][text_hidden] (row-major), matching M2's forward_embeddings_
  // taps output reshaped to (seq, 12, 2560). Empty on failure.
  metal_compute::SharedBuffer
  forward_text(const metal_compute::SharedBuffer& ehs, int text_seq);

  // A reference-image conditioning input (Qwen-Image-Edit-2509 multi-reference).
  // `latents` is a pre-VAE-encoded, packed reference latent already in the DiT
  // input space, token-major [seq, in_channels] f16 (the same layout as the
  // generated `latents`). Reference tokens are concatenated AFTER the generated
  // image tokens, share the image-stream modulation and take part in the full
  // joint attention, but carry a per-reference RoPE FRAME (axis-0) offset so
  // each reference occupies a distinct position band (Qwen _compute_video_freqs
  // frame index: generated = 0, ref i = i+1). The model predicts velocity only
  // for the generated tokens; reference tokens are dropped. seq = grid_h*grid_w.
  struct RefImage {
    metal_compute::SharedBuffer latents;   // [seq, in_channels] f16, token-major
    int seq    = 0;
    int grid_h = 0;
    int grid_w = 0;
  };

  // M3b/M3c: one denoiser step. `fused_text` is the (text_seq, hidden) output
  // of forward_text; `latents` the packed (img_seq, in_channels) noisy image;
  // `timestep` the flow-matching time (= sigma = scheduler t / 1000); the
  // (grid_h, grid_w) latent grid drives the image RoPE coordinates. Returns
  // the (img_seq, in_channels) predicted velocity. When `stop_after_block >= 0`
  // it instead returns the joint (text_seq+img_seq, hidden) hidden state after
  // that 0-indexed block (staged verification against d_blk*). Empty on fail.
  // `refs` are optional reference images (Qwen-Image-Edit multi-reference); each
  // is embedded by the same img_in, appended to the image-token stream with its
  // own RoPE frame offset, and attended jointly. Empty (the default) = plain
  // text-to-image. NOTE: reference conditioning is a LEARNED capability -- it
  // only steers generation on a reference-trained (Qwen-Image-Edit-class)
  // checkpoint; on a plain text-to-image distill the tokens run but are inert.
  metal_compute::SharedBuffer
  forward_dit(const metal_compute::SharedBuffer& fused_text, int text_seq,
              const metal_compute::SharedBuffer& latents, int img_seq,
              int grid_h, int grid_w, float timestep,
              int stop_after_block = -1,
              const std::vector<RefImage>& refs = {});

  const Config& config() const { return _cfg; }

  // Drop the per-forward scratch: the reusable DitScratch buffers (activations
  // + RoPE), the dequant/split-K scratches, and the i8 accel scratches. All
  // are lazily rebuilt on the next forward_dit (ensure_dit_scratch_ keys on
  // shape; _w_deq/_splitk/i8 re-grow on demand), so this is a pure memory
  // reclaim -- ~1-2 GB at 1024px -- for a memory-bounded box where the idle
  // DiT would otherwise crowd out a large downstream VAE decode. Call only
  // when no forward is in flight (the caller has read back the latent).
  void release_forward_scratch();

  // ---- On-device AWQ calibration ------------------------------------------
  // Accumulate per-input-channel |activation| abs-max at each main-block Linear
  // input while enabled, tapped inside the verified forward_dit (guarded; no
  // effect when off). Unlike an LM, the DiT's distribution depends on the
  // denoising timestep -- so the collector calls forward_dit over prompts x the
  // turbo sigmas and the abs-max accumulates across all of them (col_absmax is
  // read+write). Four taps per block: qkv/gate input (adaLN norm1 out), to_out.0
  // input (gated attn out), ff gate/up input (adaLN norm2 out), ff down input
  // (SwiGLU out). Feeds the DiT AWQ fold. calib_begin zeroes the accumulators.
  void calib_begin();
  void calib_end() { _calib_on = false; }
  bool calibrating() const { return _calib_on; }
  // [n_layers][channels] abs-max (hidden for qkv/o/gateup, ffn for down).
  std::vector<std::vector<float>> calib_qkv() const;
  std::vector<std::vector<float>> calib_o() const;
  std::vector<std::vector<float>> calib_gateup() const;
  std::vector<std::vector<float>> calib_down() const;

 private:
  MetalKrea2Transformer() = default;

  // A Linear weight: either a dense f16 matrix (`w`, [N,K]) or an MLX-affine
  // quantized triple (`codes` U32 packed [N, K*bits/32] + `scales`/`qbias` F16
  // [N, K/group]). `gemm_` dispatches dense_gemm_t or affine_qmm_steel
  // accordingly. Populated by load_qw_ (auto-detects `<name>.scales`).
  struct QWeight {
    metal_compute::SharedBuffer w;                   // f16 dense
    metal_compute::SharedBuffer codes, scales, qbias;  // affine quant
    bool quantized = false;
    int  bits = 0;                                   // 4 or 8 (per-weight; mixed)
    bool empty() const { return quantized ? codes.empty() : w.empty(); }
  };

  // A Krea2 attention+SwiGLU block (used by the fusion tower and the main image
  // blocks). Norms are zero-centered in the checkpoint -> +1 folded at load.
  // `n1/n2` pre-norms; `qn/kn` per-head q/k RMSNorm. Projections may be
  // quantized (4/8-bit affine) or dense f16.
  struct Block {
    metal_compute::SharedBuffer n1, n2;         // pre-attn / pre-ff norms (+1)
    QWeight q, k, v, gate;                       // attn projections (no bias)
    metal_compute::SharedBuffer qn, kn;         // per-head q/k norm (+1)
    QWeight o;                                  // attn to_out.0 (no bias)
    QWeight ff_gate, ff_up, ff_down;            // SwiGLU
    // Fused interleaved gate|up [2*ffn, K] (row 2g = gate g, 2g+1 = up g).
    // Built at load for the MAIN blocks of a quantized checkpoint when
    // _fuse_ff (ff_gate/ff_up are then released); empty otherwise (dense
    // checkpoints, the text tower). The steel path runs it through the fused
    // affine_qmm_swiglu epilogue; the M5 mma path dequants it whole and folds
    // the interleaved matmul2d output with swiglu_interleaved.
    QWeight ff_gu;
    metal_compute::SharedBuffer sst;            // adaLN scale_shift_table (main)
  };

  bool load_block_(const class MetalLlamaWeights& wts, const std::string& pre,
                   Block& b, bool main_block);
  // Load a Linear weight: quantized (affine triple) when `<name>.scales` is
  // present, else dense f16.
  QWeight load_qw_(const class MetalLlamaWeights& wts, const std::string& name);

  // M5 matrix-core GEMM fast path for one Linear: y[M,N] (element offset ye) =
  // xin[M,K] @ dequant(w)[N,K]^T on the hardware matmul2d units. Dense weights
  // feed the tiled kernel directly; affine-quant weights are first expanded into
  // the reusable f16 scratch _w_deq (affine_dequant) -- the same dequant-once ->
  // dense_gemm_mma shape the LM prefill uses. Returns true when it ran; false
  // (not matrix-core capable / M below the tile-amortize threshold / degenerate
  // N) leaves y untouched so the caller runs its steel dense_gemm_t / qmm path.
  bool gemm_mma_(metal_compute::ComputeEncoder& enc,
                 const metal_compute::SharedBuffer& xin, const QWeight& w,
                 const metal_compute::SharedBuffer& y, std::size_t ye, int M,
                 int N, int K);

  // Dynamic-int8 accelerated GEMMs (Config::i8_gemm / VPIPE_I8_GEMM);
  // null when off. Tried first in gemm_mma_ for qualifying shapes.
  std::unique_ptr<I8GemmContext> _i8;

  metal_compute::MetalCompute* _mc = nullptr;
  Config _cfg;
  int _quant_bits = 0;             // 0 = dense; 4 or 8 when quantized
  int _quant_group = 64;

  std::vector<Block> _layerwise;   // text_fusion.layerwise_blocks
  std::vector<Block> _refiner;     // text_fusion.refiner_blocks
  QWeight _projector;                                  // [1, n_text_layers]
  metal_compute::SharedBuffer _txt_norm;               // txt_in.norm (+1)
  QWeight _txt_l1w; metal_compute::SharedBuffer _txt_l1b;   // txt_in.linear_1
  QWeight _txt_l2w; metal_compute::SharedBuffer _txt_l2b;   // txt_in.linear_2

  // ---- DiT image path ----
  QWeight _img_in_w; metal_compute::SharedBuffer _img_in_b;   // img_in (64->H)
  QWeight _te_l1w, _te_l2w;
  metal_compute::SharedBuffer _te_l1b, _te_l2b;              // time_embed
  QWeight _tmp_w; metal_compute::SharedBuffer _tmp_b;        // time_mod_proj
  std::vector<Block> _blocks;                          // 28 transformer_blocks
  // Streaming-blocks mode: _blocks stays empty; forward_dit loads each block on
  // demand from _stream_wts (the retained source mmap) and frees it after use.
  bool _stream_blocks = false;
  // Zero-copy weights: quantized Linears load as READ-ONLY views into the
  // retained source mmap (newBufferWithBytesNoCopy) rather than owned copies,
  // so the OS reclaims those clean file pages under memory pressure. On by
  // default; disable with VPIPE_KREA2_NO_MMAP_WEIGHTS. Requires retaining
  // _stream_wts for the model's lifetime (as streaming mode already does).
  bool _mmap_weights = true;
  std::unique_ptr<MetalLlamaWeights> _stream_wts;
  std::function<bool()> _stream_stop;   // polled per block in streaming mode
  metal_compute::SharedBuffer _final_sst;              // final_layer (2, hidden)
  metal_compute::SharedBuffer _final_norm;             // final_layer.norm (+1)
  QWeight _final_lw; metal_compute::SharedBuffer _final_lb;   // final_layer.linear

  // One contiguous image-token segment for the RoPE build: all its tokens share
  // the axis-0 (frame/index) coordinate `frame` and tile a grid_h x grid_w
  // latent grid row-major (axis-1 = row, axis-2 = col). Segment 0 is the
  // generated image (frame 0); each reference adds one (frame = index).
  struct ImgSeg { int frame; int grid_h; int grid_w; int seq; };

  // Build the [seq, head_dim] cos/sin RoPE tables (3-axis, adjacent-pair) for
  // the joint sequence (text rows at the origin, then the image segments in
  // order -- the generated grid plus any reference grids, each in its own frame
  // band).
  void build_rope_tables_(int text_seq, const std::vector<ImgSeg>& segs,
                          metal_compute::SharedBuffer& cos_out,
                          metal_compute::SharedBuffer& sin_out);

  // Per-shape persistent scratch for forward_dit: the RoPE tables (which are
  // timestep-INDEPENDENT) + all block scratch. Across the N denoising steps of
  // one image the (seq, grid) shape is identical, so ensure_dit_scratch_ builds
  // this ONCE and reuses it -- the buffers' contents are overwritten each call.
  // Keyed by (seq, grid_h, grid_w, layout-sig); the sig captures the reference
  // segmentation so a changed reference set rebuilds the RoPE (buffers depend
  // only on total seq, but the RoPE depends on the exact split).
  struct DitScratch {
    int seq = -1, gh = -1, gw = -1;               // shape key (-1 = unbuilt)
    std::size_t sig = 0;                           // reference-layout signature
    metal_compute::SharedBuffer te_in, rcos, rsin, joint;
    metal_compute::SharedBuffer te1, temb, tmp, tmod, mod;
    metal_compute::SharedBuffer n1, n2, nm, gate, att, o;
    metal_compute::SharedBuffer q, k, v, qt, kt, vt, atb;
    metal_compute::SharedBuffer g, u, gu, modf;
  };
  DitScratch _dit;
  // `segs` is the full image layout (generated + refs); grid_h/grid_w are the
  // generated grid (the shape-key + buffer-size anchor).
  void ensure_dit_scratch_(int text_seq, int grid_h, int grid_w,
                           const std::vector<ImgSeg>& segs);

  // Libraries + kernel functions.
  metal_compute::ComputeLibrary _lib_gemm, _lib_elt, _lib_rms, _lib_sdpa,
      _lib_vis, _lib_rope, _lib_qmm;
  metal_compute::ComputeFunction _fn_gemm, _fn_gemm_bias, _fn_rms, _fn_swiglu,
      _fn_mul_sigmoid, _fn_residual, _fn_transpose, _fn_sdpa, _fn_gelu_tanh,
      _fn_rope_table, _fn_adaln, _fn_gated, _fn_qmm4, _fn_qmm8, _fn_bias_add,
      _fn_colabsmax;
  // Steel MMA flash-attention (attn_steel_h_bd128, head_dim 128, GQA-aware) for
  // the joint-sequence attention. The scalar sdpa_full_f16 is O(seq^2) at <2%
  // FLOP efficiency and DOMINATES at high resolution (~80% of a 1024px step,
  // seq 4096); the steel kernel is the register-resident flash MLX ships, fast
  // on M4 without matrix cores, and handles GQA via gqa_factor. Loaded
  // best-effort; forward_dit falls back to _fn_sdpa when absent or when
  // VPIPE_KREA2_NO_STEEL_ATTN is set (A/B + safety). align_Q/align_K are shape-
  // dependent function constants, so the concrete function is built per-forward.
  metal_compute::ComputeLibrary _lib_attn;
  metal_compute::SharedBuffer _attn_params;   // SteelAttnParams block
  bool _steel_attn_ok = false;
  // M5 matrix-core (matmul2d) NAX steel flash (attn_steel_nax_h_bd128): the
  // vision tower's M5 attention, ported to the DiT's head_dim 128. Preferred
  // over the ALU steel when supports_matrix_cores(); ALU steel is the M4 path
  // + the fallback (VPIPE_KREA2_NO_ATTN_NAX forces it for A/B).
  metal_compute::ComputeLibrary _lib_attn_nax;
  bool _use_attn_nax = false;
  // Larger-BM bf16 GEMM tiles for the block projections (amortize f16 weight
  // bandwidth at M~=seq). _gemm_tile selects: 0 = BM32, 1 = BM64 (default,
  // ~5% faster), 2 = BM64/BN64 (spills, slower); set in load(), overridable via
  // VPIPE_KREA2_GEMM_TILE for A/B.
  metal_compute::ComputeFunction _fn_gemm_bm64, _fn_gemm_bm64bn64;
  int _gemm_tile = 1;
  // Larger-BM steel qmm tile for the quantized block projections (M ~= seq):
  // the same tuning as the dense BM64 above -- each dequantized [BN,BK] weight
  // tile serves 2x the output rows, halving code re-reads + per-FLOP dequant
  // ALU work. g64-only entry points; routed in forward_dit for M >= 128.
  // _qmm_tile: 1 = BM64 (default when the functions are present), 0 = the
  // 32x32 tile, 2 = BM128 (WM=4, 256 threads) for the tall-M block GEMMs at
  // high resolution (M = seq >= 1024). BM128 quarters the fused gate|up code
  // re-reads (that GEMM had stayed at BM=32), but MEASURED ~7-8% SLOWER on
  // M4 Pro at M=4106/1024px across every GEMM section (ff-up 6663->7176,
  // qkv/o/ff-down similarly up; attn -- unchanged kernel -- flat, so not
  // thermal): these GEMMs are compute/occupancy-bound, not weight-bandwidth-
  // bound [[krea2-dit-perf]], so cutting weight re-reads attacks a non-
  // bottleneck while the 128-row/256-thread tile halves the threadgroup count
  // (33 vs 65 M-tiles) and cuts latency-hiding. So BM128 stays opt-in
  // (default BM64); kept only for an M5 matrix-core retest (bigger tiles may
  // feed the matrix units better). Correctness bit-identical to BM64 (rel-L2
  // = 0, w4 + w8; forward_dit_bm128_matches_bm64). VPIPE_KREA2_QMM_TILE=2.
  metal_compute::ComputeFunction _fn_qmm4_bm64, _fn_qmm8_bm64;
  metal_compute::ComputeFunction _fn_qmm4_bm128, _fn_qmm8_bm128;
  int _qmm_tile = 1;
  // FP16-pipe (half-accumulate) twins of the big-GEMM kernels. The default
  // steel BlockMMA widens every fragment to f32 (FP32 pipe); the _acc16
  // variants keep the simdgroup MMA on half8x8 -- worthwhile only where the
  // f16 matrix rate exceeds f32. MEASURED on M4 Pro (gemm_mma.alu_rate
  // probes): f16 10.4 vs f32 10.1 TFLOP/s and mixed slower -- perf-NEUTRAL
  // there, while f16 accumulation costs accuracy (w4 golden 0.0456->0.0506,
  // w8 0.0161->0.0290). So _acc16 stays opt-in (VPIPE_KREA2_ACC16=1),
  // routed only to the tall-M DiT block GEMMs; re-probe before enabling on
  // a new GPU generation.
  metal_compute::ComputeFunction _fn_qmm4_bm64_a16, _fn_qmm8_bm64_a16,
      _fn_qmm_swiglu4_a16, _fn_qmm_swiglu8_a16, _fn_gemm_bm64_a16;
  bool _acc16 = false;
  // Fused SwiGLU FF over the interleaved gate|up weight. Steel path: one
  // affine_qmm_swiglu GEMM (register-local silu(gate)*up in the store
  // epilogue) replaces the two ff GEMMs + the swiglu pass, with no gate/up
  // intermediates. M5 mma path: dequant + one matmul2d over the same
  // interleaved weight -> _dit.gu [seq, 2*ffn], folded to [seq, ffn] by
  // swiglu_interleaved (the LM prefill's fused-FF shape). g64-only kernels.
  // VPIPE_KREA2_NO_FUSED_FF opts out (A/B).
  metal_compute::ComputeFunction _fn_qmm_swiglu4, _fn_qmm_swiglu8,
      _fn_swiglu_inter;
  // BM=128 twins of the fused SwiGLU GEMM (the biggest DiT GEMM, the only
  // quantized tile that had stayed at BM=32). Routed when _qmm_tile == 2 and
  // seq >= 1024.
  metal_compute::ComputeFunction _fn_qmm_swiglu4_bm128, _fn_qmm_swiglu8_bm128;
  bool _fuse_ff = false;

  // M5-only matrix-core dense GEMM (matmul2d/NAX) for the DiT projections. Same
  // NAX path the LM prefill + gemma-vision use, gated on supports_matrix_cores()
  // (Apple10+) so M4/older keep the steel dense_gemm_t / affine_qmm_steel
  // (byte-identical). VPIPE_KREA2_NO_MMA2=1 forces steel on M5 (A/B). The
  // quantized DiT dequant-expands each weight into _w_deq before the dense
  // matmul2d; _fn_dequant4/8 are the group-matched affine_dequant kernels
  // (loaded only for a quantized checkpoint). matmul2d rounds bias in a separate
  // bias_add_rows pass (_fn_bias_add), so it is loaded whenever mma is active.
  metal_compute::ComputeLibrary _lib_dense_mma, _lib_dequant;
  metal_compute::ComputeFunction _fn_dense_mma, _fn_dense_mma_deep,
      _fn_dense_mma_tn2, _fn_dequant4, _fn_dequant8;
  metal_compute::SharedBuffer _w_deq;   // reusable [N,K] f16 dequant scratch
  bool _use_mma2 = false;
  int  _mma_min_m = 64;   // matmul2d only wins once M amortizes the 128-row tile
  // Split-K deep-reduction GEMM for the very deep K (ff-down, K=16384): the
  // single-op full reduction runs ~0.7x the K<=9728 rate, so gemm_mma_ routes
  // K that is a multiple of kSplitKC (>= 2 chunks) through the split-K kernel
  // (grid.z = K/kSplitKC partial planes) + a residual_add fold. _splitk holds
  // the [n_splits, M, N] f16 partials (reused across calls). Gated on M and on
  // VPIPE_KREA2_NO_SPLITK (A/B).
  static constexpr int kSplitKC = 8192;
  metal_compute::ComputeFunction _fn_dense_mma_splitk;
  metal_compute::SharedBuffer _splitk;  // [n_splits, M, N] f16 partial planes
  bool _use_splitk = false;

  // ---- AWQ calibration accumulators (per main block; live while _calib_on) --
  bool _calib_on = false;
  std::vector<metal_compute::SharedBuffer> _cb_qkv, _cb_o, _cb_gu, _cb_dn;
  std::vector<std::vector<float>> read_calib_(
      const std::vector<metal_compute::SharedBuffer>& acc, int dim) const;
};

}  // namespace genai
}  // namespace vpipe

#endif

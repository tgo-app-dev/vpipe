#ifndef GENERATIVE_MODELS_QWEN_IMAGE_METAL_QWEN_IMAGE_TRANSFORMER_H
#define GENERATIVE_MODELS_QWEN_IMAGE_METAL_QWEN_IMAGE_TRANSFORMER_H

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

class MetalLlamaWeights;   // fwd

// Qwen-Image-Edit-2511 denoiser (QwenImageTransformer2DModel): a DUAL-STREAM
// MMDiT flow-matching transformer, run in f16 on the metal-compute backend.
//
// Unlike the single-stream Krea-2 DiT, every block carries SEPARATE image and
// text projections (to_q/k/v + add_q/k/v), MLPs and adaLN modulation; the two
// streams are joined only in the attention (concat [text, image], one full
// non-causal SDPA, split back). Norms are plain LayerNorm (no affine); the MLP
// is GELU (tanh-approx), not SwiGLU; q/k carry a per-head RMSNorm. The image
// RoPE is 3-axis (frame/height/width, axes [16,56,56], adjacent-pair, with the
// scale_rope height/width centering); the text tokens take a contiguous
// position band above the image grid.
//
// Reuses the metal kernel toolbox (dense GEMM+bias, rms_norm, layer_norm,
// rope_pair_table, adaln_modulate, gated_residual, gelu_tanh, steel flash
// attention) shared with the Krea-2 / vision paths. Weights load as raw bf16 ->
// f16 (the base model is bf16; affine-quant support is a later milestone).
class MetalQwenImageTransformer {
 public:
  struct Config {
    int   hidden       = 3072;   // num_attention_heads * attention_head_dim
    int   n_heads      = 24;
    int   head_dim     = 128;
    int   n_layers     = 60;
    int   in_channels  = 64;     // 16 latent ch * patch 2 * patch 2
    int   txt_dim      = 3584;   // joint_attention_dim (Qwen2.5-VL hidden)
    int   ffn          = 12288;  // FeedForward inner (4 * hidden)
    int   time_proj    = 256;    // sinusoidal timestep channels
    float norm_eps     = 1e-6f;
    int   rope_theta   = 10000;
    int   axes[3]      = {16, 56, 56};   // axes_dims_rope (sum = head_dim)
  };

  // A reference-image conditioning input (Qwen-Image-Edit multi-reference): a
  // pre-VAE-encoded, 2x2-packed reference latent already in the DiT input space,
  // token-major [seq, in_channels] f16. Reference tokens are embedded by the
  // same img_in, appended AFTER the generated image tokens, and attended
  // jointly; each reference occupies its own RoPE frame band (frame = index+1)
  // and is modulated at timestep 0 (clean; zero_cond_t). Velocity is predicted
  // only for the generated tokens. seq = grid_h*grid_w.
  struct RefImage {
    metal_compute::SharedBuffer latents;   // [seq, in_channels] f16, token-major
    int seq = 0, grid_h = 0, grid_w = 0;
  };

  // `stream_blocks` (memory-bounded, for small boxes / 16GB): don't preload the
  // 60 dual-stream blocks -- retain the source mmap and load/free each block on
  // demand inside forward(), so ~one block (~0.7GB) is resident instead of the
  // whole 20B DiT. The preloaded top-level weights (embedders, time, norm_out)
  // stay. Slower (a commit+wait per block), but fits AWQ calibration / a full
  // denoise on 16GB. Mirrors the Krea-2 / FLUX-2 streaming path.
  //
  // `pin_frac` (streaming only): when > 0, pin a LEADING prefix of blocks
  // resident so pinned + running stays within that fraction of physical RAM
  // (e.g. 0.60). Pinned blocks are read once and reused across every forward;
  // only the tail streams -- trades spare RAM for speed. 0 => pure streaming.
  static std::unique_ptr<MetalQwenImageTransformer>
  load(const std::string& model_dir, metal_compute::MetalCompute* mc,
       const Config& cfg, bool stream_blocks = false, double pin_frac = 0.0);

  ~MetalQwenImageTransformer();

  // Cooperative stop, polled once per block in streaming mode (a pipeline stop
  // is then honored within ~one block instead of a whole 60-block forward).
  // forward() returns an empty buffer when it fires. No effect when preloaded.
  void set_stream_stop(std::function<bool()> stop) {
    _stream_stop = std::move(stop);
  }

  // Leading blocks pinned resident in streaming mode (0 = pure streaming, or
  // preloaded). For logging the RAM-for-speed decision.
  int pinned_blocks() const { return _pinned; }

  // True when the M5 matrix-core matmul2d (NAX) path is active for the block
  // GEMMs (supports_matrix_cores() + kernels loaded + VPIPE_QIE_NO_MMA2 unset).
  // For the mma-vs-steel A/B test to assert the path actually engaged.
  bool uses_mma2() const { return _use_mma2; }

  // One denoiser step. `hidden` is the packed [gen_seq, in_channels] noisy
  // image (gen_seq = grid_h*grid_w); `txt` the [txt_seq, txt_dim] encoder
  // prompt embeddings (post encoder final-norm); `sigma` the flow-matching time
  // (= timestep). Returns the [gen_seq, in_channels] predicted velocity.
  // `refs` are optional reference images (edit conditioning). When
  // `stop_after_block >= 0` it instead returns the image-stream hidden state
  // [gen_seq, hidden] after that 0-indexed block (staged verification).
  metal_compute::SharedBuffer
  forward(const metal_compute::SharedBuffer& hidden, int gen_seq,
          const metal_compute::SharedBuffer& txt, int txt_seq,
          int grid_h, int grid_w, float sigma,
          const std::vector<RefImage>& refs = {},
          int stop_after_block = -1);

  const Config& config() const { return _cfg; }

  // ---- On-device AWQ activation calibration ------------------------------
  // Between calib_begin() and calib_end(), forward() taps the per-input-channel
  // running abs-max at each quantized linear's input into per-group accumulators
  // (dual-stream: img/txt x qkv/o/fc1/fc2). calib_stats() returns them as
  // group -> [n_layers*dim] f32 (dim = hidden for qkv/o/fc1, ffn for fc2). The
  // taps are guarded (no effect when off) so this is the model's REAL activation
  // distribution across the calibration prompts x sigmas. Fed to the quantizer's
  // activation-aware weight clipping (model-quantize target=dit awq=true).
  void calib_begin();
  void calib_end() { _calib_on = false; }
  bool calibrating() const { return _calib_on; }
  std::map<std::string, std::vector<float>> calib_stats() const;

 private:
  MetalQwenImageTransformer() = default;

  // Load a checkpoint tensor as a bf16 SharedBuffer (BF16 memcpy'd; F16/F32
  // converted). The whole forward runs bf16.
  metal_compute::SharedBuffer to_elt_(const MetalLlamaWeights& wts,
                                      const std::string& name);
  bool load_linear_(const MetalLlamaWeights& wts, const std::string& pre,
                    metal_compute::SharedBuffer& w,
                    metal_compute::SharedBuffer& b);

  // A Linear's weight, dense (bf16) OR affine group-quantized (w4/w8 g64). The
  // quantizer writes `<name>.weight` as U32 packed codes (+ `.scales`/`.biases`
  // F16 -> converted to bf16), keyed by the presence of `<name>.scales`. Bias
  // (`<name>.bias`) stays a separate bf16 buffer, added after the qmm.
  struct QWeight {
    metal_compute::SharedBuffer w;                         // dense bf16 [N,K]
    metal_compute::SharedBuffer codes, scales, qbias;      // affine quant
    bool quantized = false;
    int  bits = 0;                                         // 4 or 8
    bool empty() const { return quantized ? codes.empty() : w.empty(); }
  };
  // Load a (possibly-quantized) linear's weight into `qw` and its bias into `b`.
  bool load_linear_q_(const MetalLlamaWeights& wts, const std::string& pre,
                      QWeight& qw, metal_compute::SharedBuffer& b);
  QWeight load_qw_(const MetalLlamaWeights& wts, const std::string& name);

  // M5 matrix-core matmul2d biasless GEMM y[M,N] = x[M,K] @ w[N,K]^T (dense
  // weight direct, or a quantized weight dequant-expanded once into _w_deq).
  // `xe` is x's element offset (the dual-stream img/txt sub-slice). Returns
  // false (y untouched) when matrix cores are absent, M is below the tile-
  // amortize threshold, or N is degenerate -> the caller keeps its steel
  // dense_gemm_t / affine_qmm_steel path. On success the caller folds bias
  // with a bias_add_rows pass (matmul2d has no bias slot).
  bool gemm_mma_(metal_compute::ComputeEncoder& enc,
                 const metal_compute::SharedBuffer& xin, std::size_t xe,
                 const QWeight& w, const metal_compute::SharedBuffer& y,
                 std::size_t ye, int M, int N, int K);

  // Per-stream, per-block weights. The attention projections + FeedForward MLPs
  // (the model-quantize leaf set) are QWeight (dense or w4/w8); the adaLN
  // modulation + per-head q/k norms stay bf16.
  struct Block {
    QWeight img_mod_w, txt_mod_w;                        // AdaLN mod Linear(H,6H)
    metal_compute::SharedBuffer img_mod_b, txt_mod_b;    // dense by default;
                                                         // quantized when opted
                                                         // in (model-quantize
                                                         // quant_modulation)
    QWeight qw, kw, vw, ow;                              // image attn
    metal_compute::SharedBuffer qb, kb, vb, ob;
    QWeight aqw, akw, avw, aow;                          // text attn
    metal_compute::SharedBuffer aqb, akb, avb, aob;
    metal_compute::SharedBuffer nq, nk, naq, nak;       // q/k RMSNorm [head_dim]
    QWeight img_fc1_w, img_fc2_w, txt_fc1_w, txt_fc2_w;
    metal_compute::SharedBuffer img_fc1_b, img_fc2_b, txt_fc1_b, txt_fc2_b;
  };
  // Load block L's per-stream weights from `wts` into `b` (dense or quantized).
  // Used both to preload and, in streaming mode, per-block inside forward().
  bool load_block_(const MetalLlamaWeights& wts, int L, Block& b);

  metal_compute::MetalCompute* _mc = nullptr;
  Config _cfg;
  int _quant_bits = 0;      // 0 = dense bf16; 4 / 8 = affine group-quant
  int _quant_group = 64;

  metal_compute::SharedBuffer _img_in_w, _img_in_b;      // in_channels -> H
  metal_compute::SharedBuffer _txt_norm_w;               // RMSNorm txt_dim
  metal_compute::SharedBuffer _txt_in_w, _txt_in_b;      // txt_dim -> H
  metal_compute::SharedBuffer _t1_w, _t1_b, _t2_w, _t2_b;   // time embed
  metal_compute::SharedBuffer _normout_w, _normout_b;    // AdaLN-continuous
  metal_compute::SharedBuffer _projout_w, _projout_b;    // H -> in_channels
  std::vector<Block> _blocks;                            // empty when streaming

  // Streaming mode (stream_blocks): _blocks holds only the pinned prefix
  // (_pinned blocks, possibly 0); blocks L >= _pinned are loaded from the
  // retained source mmap on demand in forward() and freed after use.
  bool _stream_blocks = false;
  int _pinned = 0;                  // pinned leading blocks (streaming only)
  std::unique_ptr<MetalLlamaWeights> _stream_wts;
  std::function<bool()> _stream_stop;

  // One contiguous image-token segment for the RoPE build (frame band + grid).
  struct ImgSeg { int frame, grid_h, grid_w, seq; };
  // Build the [seq, head_dim] adjacent-pair cos/sin RoPE tables for the joint
  // sequence (text rows first, then the image segments -- generated grid + any
  // references, each in its own frame band). Host-computed (small), replicating
  // diffusers QwenEmbedRope (3-axis, scale_rope centering, txt position band).
  void build_rope_(int txt_seq, const std::vector<ImgSeg>& segs,
                   metal_compute::SharedBuffer& cos_out,
                   metal_compute::SharedBuffer& sin_out);
  // Host-compute the [time_proj] sinusoidal timestep embedding of `sigma`.
  std::vector<float> time_proj_(float sigma) const;

  // Libraries + kernel functions. The whole DiT runs in bf16 (the residual
  // stream reaches ~1e7, far past f16's 65504) -> the *_bf16 metallib variants.
  metal_compute::ComputeLibrary _lib_gemm, _lib_elt, _lib_rms, _lib_sdpa,
      _lib_rope, _lib_qmm, _lib_attn;
  metal_compute::ComputeFunction _fn_gemm, _fn_gemm_bias, _fn_rms, _fn_layernorm,
      _fn_silu, _fn_gelu, _fn_residual, _fn_transpose, _fn_sdpa, _fn_rope_table,
      _fn_adaln, _fn_gated;
  // Fused transpose+RoPE for the q/k path (token-major -> head-major roped in one
  // pass; saves the separate rope's read+write). Best-effort: null -> the split
  // transpose + rope path. VPIPE_QIE_NO_FUSE_ROPE disables.
  metal_compute::ComputeFunction _fn_transpose_rope;
  // Dense-GEMM fast-path twins: the BM=64 steel tile (tall-M block GEMMs, halves
  // weight re-reads) + a GEMV (the M=1 conditioning / modulation / head rows,
  // bias-less -> a row bias-add follows). Best-effort (fall back to _fn_gemm).
  metal_compute::ComputeFunction _fn_gemm_bm64, _fn_gemv;

  // Steel register-resident flash attention (bf16, head_dim 128) for the joint
  // attention -- replaces the scalar O(seq^2) sdpa at high resolution (mirrors
  // the Krea-2 / FLUX.2 DiTs). `_attn_params` holds the MLX steel AttnParams
  // block (filled per forward). Best-effort: falls back to _fn_sdpa when the
  // kernel is unavailable (VPIPE_QIE_NO_STEEL_ATTN forces the scalar path).
  metal_compute::SharedBuffer _attn_params;
  bool _steel_attn_ok = false;
  // Affine qmm (loaded only when quantized): w4g64 / w8g64 (bf16 variant) +
  // the row-broadcast bias add applied after the (bias-less) qmm.
  metal_compute::ComputeFunction _fn_qmm4, _fn_qmm8, _fn_bias_add;
  // Wider steel-qmm tiles for the tall-M (M ~= seq) DiT block GEMMs: BM=64
  // amortizes each [BN,BK] weight tile over 64 output rows (half the code
  // re-reads of the BM=32 base), BM=128 halves it again at high resolution.
  // Selected by M in gemm_bias_q; small-M (txt, head) shapes keep the base tile.
  // _qmm_tile: 0 = base only, 1 = +BM64, 2 = +BM128. VPIPE_QIE_QMM_TILE overrides.
  metal_compute::ComputeFunction _fn_qmm4_bm64, _fn_qmm8_bm64;
  metal_compute::ComputeFunction _fn_qmm4_bm128, _fn_qmm8_bm128;
  int _qmm_tile = 0;
  // Peak M=1 quantized GEMV (the LLM-decode kernel) for the per-block modulation
  // projection when the mod weights are quantized -- the steel qmm tile wastes
  // 31/32 rows at M=1, while affine_qmv streams the codes at ~peak bandwidth.
  // Loaded only when quantized; null -> the mod GEMV falls back to gemm_bias_q.
  metal_compute::ComputeLibrary     _lib_qmv;
  metal_compute::ComputeFunction    _fn_qmv4, _fn_qmv8;

  // M5-only matrix-core dense GEMM (matmul2d / NAX) for the DiT block GEMMs --
  // the same NAX path Krea-2 / FLUX.2 use, here in the *_bf16 metallib variant
  // (QIE's residual stream runs bf16). Gated on supports_matrix_cores()
  // (Apple10+) so M4/older keep the steel dense_gemm_t / affine_qmm_steel path
  // (byte-identical). A quantized weight is dequant-expanded into _w_deq
  // (affine_dequant, bf16) then run through the SAME dense matmul2d
  // (dequant-once -> dense-GEMM, matching affine_qmm_steel to f32 rounding).
  // matmul2d has no bias slot -> gemm_bias_q folds bias with a bias_add_rows
  // pass. VPIPE_QIE_NO_MMA2=1 forces steel on M5 (A/B).
  metal_compute::ComputeLibrary _lib_dense_mma, _lib_dequant;
  metal_compute::ComputeFunction _fn_dense_mma, _fn_dense_mma_deep,
      _fn_dense_mma_tn2, _fn_dequant4, _fn_dequant8;
  metal_compute::SharedBuffer _w_deq;   // reusable [N,K] bf16 dequant scratch
  bool _use_mma2 = false;
  int  _mma_min_m = 64;   // matmul2d wins once M amortizes the 128-row tile
  // Split-K deep-reduction GEMM for very deep K (a multiple of kSplitKC, >= 2
  // chunks): grid.z = K/kSplitKC partial planes + a residual_add fold. QIE's
  // deepest K is ff-down 12288 (not a multiple), so this rarely fires; kept
  // for parity with Krea-2 + future dims. VPIPE_QIE_NO_SPLITK opts out.
  static constexpr int kSplitKC = 8192;
  metal_compute::ComputeFunction _fn_dense_mma_splitk;
  metal_compute::SharedBuffer _splitk;  // [n_splits, M, N] bf16 partial planes
  bool _use_splitk = false;

  // AWQ calibration: per-group running col-absmax accumulators (each a single
  // [n_layers*dim] bf16 buffer), the tap kernel, and the on/off guard.
  metal_compute::ComputeFunction _fn_colabsmax;
  bool _calib_on = false;
  std::map<std::string, metal_compute::SharedBuffer> _calib_acc;
};

}  // namespace genai
}  // namespace vpipe

#endif

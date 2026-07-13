#ifndef GENERATIVE_MODELS_FLUX2_METAL_FLUX2_TRANSFORMER_H
#define GENERATIVE_MODELS_FLUX2_METAL_FLUX2_TRANSFORMER_H

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
class I8GemmContext;       // fwd (shared/i8-gemm.h)

// FLUX.2-klein denoiser (Flux2Transformer2DModel): a FLUX-topology flow-matching
// transformer -- N_double DUAL-STREAM (MMDiT joint) blocks followed by N_single
// SINGLE-STREAM blocks -- run in f16 on the metal-compute backend.
//
// Unlike the Krea-2 (Qwen-Image) MMDiT, FLUX.2 keeps the image and text streams
// SEPARATE through the double blocks (each with its own modulation + FF), joins
// them for a joint attention (K/V/Q concatenated [text, image]), then runs the
// single blocks over the concatenated sequence with a FUSED qkv+mlp projection.
//
// Text conditioning enters PRE-COMPUTED: the Qwen3 text encoder's hidden states
// tapped from 3 layers ({10,20,30}) are concatenated to [text_seq, 7680]
// (= 3 x 2560) by the stage; context_embedder projects that to the transformer
// width. (Krea-2's text-fusion tower has no analogue here.)
//
// Modulation is SHARED across blocks: three model-level Flux2Modulation Linears
// (double img: 2 param-sets, double txt: 2, single: 1) turn the timestep
// embedding into shift/scale/gate vectors reused by every block of that kind.
// The distilled klein has guidance_embeds=false (no CFG guidance channel).
//
// Perf tiers (mirroring Krea-2): steel GEMMs + fused-SwiGLU FF everywhere,
// steel/NAX flash attention, M5 matrix-core matmul2d (+ split-K deep-K) for
// the block GEMMs (gemm_mma_), and block streaming for memory-bounded boxes.
class MetalFlux2Transformer {
 public:
  struct Config {
    int   hidden        = 3072;   // inner_dim = attention_head_dim * num_heads
    int   n_heads       = 24;
    int   head_dim      = 128;
    int   in_channels   = 128;    // 32 latent ch * patch 2 * patch 2
    int   joint_dim     = 7680;   // context (text) embedding dim (3 x 2560)
    int   n_double      = 5;      // num_layers (dual-stream blocks)
    int   n_single      = 20;     // num_single_layers
    float mlp_ratio     = 3.0f;
    int   timestep_dim  = 256;    // timestep_guidance_channels
    float norm_eps      = 1e-6f;
    float rope_theta    = 2000.0f;
    int   axes_dim[4]   = {32, 32, 32, 32};   // axes_dims_rope (sum = head_dim)
    // Guidance-distilled variants (klein-9B) carry a guidance_embedder that
    // embeds the guidance scale and adds it to the timestep embedding; the
    // distilled 4B has guidance_embeds=false (timestep embedding only). Set
    // from config.json at load(); the guidance_embedder weights load only when
    // true.
    bool  guidance_embeds = false;
    // Derived dims read from the checkpoint at load() (mlp hidden sizes vary by
    // the exact Flux2FeedForward / Flux2SwiGLU construction); 0 = "read from the
    // weight shapes".
    int   double_ff_hidden = 0;   // ff.linear_in out dim  (~ hidden * mlp_ratio)
    int   single_mlp_in    = 0;   // to_qkv_mlp_proj out - 3*hidden (swiglu in)
    int   out_channels     = 0;   // proj_out out dim (patch^2 * out_ch); 0 -> in
    // Accelerated mode (LOSSY, opt-in): dynamic-int8 GEMMs for the big
    // block matmuls (see shared/i8-gemm.h). ~2x the f16 matmul2d rate on
    // matrix-core GPUs at int8 quality (rel-L2 ~1e-2 per GEMM). Env
    // VPIPE_I8_GEMM=0|1 overrides.
    bool  i8_gemm = false;
  };

  // `stream_blocks` (memory-bounded, for small boxes): the double + single
  // blocks are NOT preloaded; each is re-read from the retained source mmap
  // just before use in forward_dit and freed after, so peak block RAM is ~one
  // block instead of the whole DiT (the embedders / modulation / final layer
  // are still preloaded). ~2-3x slower per step (weights re-read per forward).
  static std::unique_ptr<MetalFlux2Transformer>
  load(const std::string& model_dir, metal_compute::MetalCompute* mc,
       const Config& cfg, bool stream_blocks = false);

  ~MetalFlux2Transformer();

  // Cooperative stop polled per block in streaming mode, so a pipeline stop is
  // honored within ~one block instead of a whole forward. No-op preloaded.
  void set_stream_stop(std::function<bool()> stop) {
    _stream_stop = std::move(stop);
  }

  // A reference-image conditioning input (FLUX.2 multi-reference / Kontext-
  // style). `latents` is a pre-VAE-encoded, patchified reference latent already
  // in the DiT input space, packed token-major [seq, in_channels] f16 (the same
  // layout as the generated `latents`). Reference tokens are concatenated AFTER
  // the generated image tokens, share the image-stream modulation and take part
  // in the full joint attention, but carry a per-reference RoPE T (time/index)
  // offset so each reference occupies a distinct position band (Flux2
  // _prepare_image_ids: T = scale + scale*index, scale = 10 -> ref0 T=10, ref1
  // T=20). The model predicts velocity only for the generated tokens; reference
  // tokens are dropped from the output. seq = grid_h * grid_w.
  struct RefImage {
    metal_compute::SharedBuffer latents;   // [seq, in_channels] f16, token-major
    int seq    = 0;
    int grid_h = 0;
    int grid_w = 0;
  };

  // One denoiser step. `context` is the [text_seq, joint_dim] pre-computed text
  // embedding (from the encoder tap concat); `latents` the packed [img_seq,
  // in_channels] noisy image; `timestep` the flow-matching time (sigma). The
  // (grid_h, grid_w) latent grid drives the image-axis RoPE coordinates.
  // `guidance` is the embedded guidance scale for guidance-distilled variants
  // (klein-9B); < 0 (the default) or a model without guidance_embeds skips the
  // guidance embedding and uses the timestep embedding alone.
  // `refs` are optional reference images (up to a handful); each is embedded by
  // the same x_embedder, appended to the image-token stream with its own RoPE
  // T offset, and attended jointly. Empty (the default) = plain text-to-image.
  // Returns the [img_seq, out_channels] predicted velocity (generated tokens
  // only; reference tokens are not returned). Empty on failure.
  metal_compute::SharedBuffer
  forward_dit(const metal_compute::SharedBuffer& context, int text_seq,
              const metal_compute::SharedBuffer& latents, int img_seq,
              int grid_h, int grid_w, float timestep, float guidance = -1.0f,
              const std::vector<RefImage>& refs = {});

  const Config& config() const { return _cfg; }

  // ---- On-device AWQ calibration ------------------------------------------
  // Accumulate per-input-channel |activation| abs-max at each quantizable Linear
  // input while enabled, tapped inside forward_dit (guarded; no effect when
  // off). Like Krea-2 the DiT distribution is timestep-dependent, so the
  // collector runs forward_dit over prompts x sigmas and the abs-max
  // accumulates. calib_stats() returns {group -> flat [rows*dim]} feeding the
  // clip-only AWQ fold (see metal-flux2-calibration + the quantizer's flux2
  // dit_act). Groups: dbl_{norm1,attn,norm2,ffact}_{img,txt}, sgl_{norm,cat},
  // emb_{x,ctx,proj}.
  void calib_begin();
  void calib_end() { _calib_on = false; }
  bool calibrating() const { return _calib_on; }
  std::map<std::string, std::vector<float>> calib_stats() const;

 private:
  MetalFlux2Transformer() = default;

  // A Linear weight: dense f16 [N,K] or an affine-quantized triple
  // (codes/scales/qbias). `gemm_` dispatches accordingly. Loaded by load_qw_.
  struct QWeight {
    metal_compute::SharedBuffer w;                     // f16 dense [N,K]
    metal_compute::SharedBuffer codes, scales, qbias;  // affine quant
    bool quantized = false;
    int  bits = 0;                                     // 4 | 8 (per-weight)
    int  n = 0, k = 0;                                 // out, in dims
    bool empty() const { return quantized ? codes.empty() : w.empty(); }
  };

  // Dual-stream (Flux2TransformerBlock). norm1/norm2 are affine-free LayerNorm;
  // modulation (shift/scale/gate) comes from the shared model-level Linears.
  // Attention: separate img (to_q/k/v) + txt (add_q/k/v_proj) projections joined
  // for one attention; per-head q/k RMSNorm; to_out.0 (img) + to_add_out (txt).
  // FF: img (ff.linear_in/out) + txt (ff_context.linear_in/out), GELU-tanh act.
  struct DoubleBlock {
    QWeight q, k, v, o;                     // image attn (to_q/k/v/to_out.0)
    QWeight aq, ak, av, ao;                 // text  attn (add_*_proj/to_add_out)
    metal_compute::SharedBuffer qn, kn;     // per-head q/k RMSNorm (img)
    metal_compute::SharedBuffer aqn, akn;   // per-head q/k RMSNorm (txt)
    QWeight ff_in, ff_out;                  // image FF (linear_in/out)
    QWeight cff_in, cff_out;                // text  FF (ff_context.*)
  };
  // Single-stream (Flux2SingleTransformerBlock) over the concat [text; image].
  // norm is affine-free LayerNorm; modulation from single_stream_modulation.
  // qkv_mlp = to_qkv_mlp_proj -> [3*hidden | 2*single_mlp] split; swiglu the mlp
  // half; to_out over concat[attn, mlp].
  struct SingleBlock {
    QWeight qkv_mlp;                        // to_qkv_mlp_proj (unfused path)
    // Fused path (_fuse_ff): to_qkv_mlp_proj split into the attention qkv rows
    // [0:3H] and the INTERLEAVED gate|up mlp rows [3H:], so the mlp runs as a
    // fused-SwiGLU GEMM (silu(gate)*up direct, no [seq, 2*SMLP] intermediate).
    QWeight qkv, mlp_gu;
    metal_compute::SharedBuffer qn, kn;     // per-head q/k RMSNorm
    QWeight o;                             // to_out
  };

  QWeight load_qw_(const MetalLlamaWeights& wts, const std::string& name);
  // Reorder a fused [2*INNER,K] gate|up weight's rows from concatenated (gate
  // block then up block) to INTERLEAVED (row 2g = gate_g, 2g+1 = up_g), in
  // place, so the fused-SwiGLU epilogue reads even col = gate, odd col = up.
  void interleave_gu_(QWeight& qw);
  // Extract rows [start, start+count) of a QWeight into a fresh QWeight (dense
  // or quantized -- the codes/scales/biases rows are the N dimension).
  QWeight slice_rows_(const QWeight& src, int start, int count);
  bool load_double_(const MetalLlamaWeights& wts, const std::string& pre,
                    DoubleBlock& b);
  bool load_single_(const MetalLlamaWeights& wts, const std::string& pre,
                    SingleBlock& b);

  metal_compute::MetalCompute* _mc = nullptr;
  Config _cfg;
  int _quant_bits = 0;             // 0 dense; 4|8 quantized
  int _quant_group = 64;

  // Embedders + shared modulation + final layer.
  QWeight _x_embed;                                  // x_embedder (in_ch->hidden)
  QWeight _ctx_embed;                                // context_embedder (7680->h)
  QWeight _t_emb1, _t_emb2;   // timestep_embedder.linear_1 / linear_2 (+bias)
  metal_compute::SharedBuffer _t_emb1_b, _t_emb2_b;
  QWeight _g_emb1, _g_emb2;   // guidance_embedder.linear_1 / linear_2 (+bias);
  metal_compute::SharedBuffer _g_emb1_b, _g_emb2_b;   // only when guidance_embeds
  QWeight _mod_img, _mod_txt, _mod_single;   // Flux2Modulation.linear (no bias)
  QWeight _proj_out;                                 // proj_out (hidden->out_ch)
  QWeight _norm_out_lin;      // AdaLayerNormContinuous.linear (hidden->2*hidden)
  std::vector<DoubleBlock> _double;   // preloaded (empty when streaming)
  std::vector<SingleBlock> _single;   // preloaded (empty when streaming)

  // Streaming mode: the double/single blocks are loaded on demand from the
  // retained source mmap (_stream_wts) per forward and freed after use.
  bool _stream_blocks = false;
  // Zero-copy mmap of the quantized weight tensors (codes/scales/qbias) as
  // read-only views aliasing the retained source mmap (_stream_wts), instead of
  // owned copies. The pages are clean + file-backed, so the OS can reclaim the
  // DiT's resident footprint under memory pressure (e.g. a large VAE decode)
  // and re-fault from the file later -- the Krea-2 DiT does the same. On by
  // default when preloading; off in streaming mode (blocks already re-read JIT)
  // and via VPIPE_FLUX2_NO_MMAP_WEIGHTS. Requires the model on local SSD.
  bool _mmap_weights = false;
  std::unique_ptr<MetalLlamaWeights> _stream_wts;
  std::function<bool()> _stream_stop;

  // Constant affine-free LayerNorm params (weight=1, bias=0), sized `hidden`.
  metal_compute::SharedBuffer _ln_w1, _ln_b0;

  // One contiguous image-token segment for the RoPE build: all its tokens share
  // the axis-0 (T = time/index) coordinate `t_off` and tile a grid_h x grid_w
  // latent grid row-major (axis-1 = row, axis-2 = col; axis-3 = 0). Segment 0 is
  // the generated image (t_off 0); each reference adds one (t_off = 10*(i+1)).
  struct ImgSeg { int t_off; int grid_h; int grid_w; int seq; };

  // Build the [seq, head_dim] cos/sin 4-axis RoPE tables for the joint sequence
  // (text rows at the origin on axis-3, then the image segments in order). The
  // segment list carries the generated grid plus any reference grids so each
  // reference lands in its own T band. Rebuilt per shape.
  void build_rope_tables_(int text_seq, const std::vector<ImgSeg>& segs,
                          metal_compute::SharedBuffer& cos_out,
                          metal_compute::SharedBuffer& sin_out);

  // Libraries + kernel functions.
  metal_compute::ComputeLibrary _lib_gemm, _lib_elt, _lib_rms, _lib_sdpa,
      _lib_vis, _lib_rope, _lib_ln, _lib_qmm;
  metal_compute::ComputeFunction _fn_gemm, _fn_gemm_bias, _fn_rms, _fn_swiglu,
      _fn_residual, _fn_transpose, _fn_sdpa, _fn_gelu_tanh, _fn_rope_table,
      _fn_adaln, _fn_gated, _fn_layernorm, _fn_bias_add, _fn_qmm4, _fn_qmm8,
      _fn_headslice, _fn_mulsig, _fn_concat, _fn_transpose_rs;
  // Larger-tile dense f16 GEMM twins (fewer weight re-reads at the DiT's big
  // M = seq). _gemm_tile: 0 = 32x32 base, 1 = BM64, 2 = BM64xBN64; _acc16 uses
  // the f16-accumulate BM64 twin. Selected in load() (VPIPE_FLUX2_GEMM_TILE /
  // VPIPE_FLUX2_GEMM_ACC16), applied for M >= 128.
  metal_compute::ComputeFunction _fn_gemm_bm64, _fn_gemm_bm64bn64,
      _fn_gemm_bm64_a16;
  int  _gemm_tile = 0;
  bool _acc16 = false;
  // BM128 affine-qmm twins (g64 only): a 128-row tile reads each weight tile
  // once across 128 output rows -- the win for the DiT's big M = seq quant
  // GEMMs. _qmm_tile 1 = use BM128 for M >= 1024. VPIPE_FLUX2_QMM_TILE.
  metal_compute::ComputeFunction _fn_qmm4_bm128, _fn_qmm8_bm128;
  int _qmm_tile = 0;
  // Fused-SwiGLU FF (BM64): one GEMM whose register-local epilogue writes
  // silu(gate)*up directly from the INTERLEAVED linear_in weight, killing the
  // [seq, 2*INNER] intermediate + the slice/swiglu passes. Dense (bf16) +
  // w4/w8 twins, float or FP16-pipe (acc16) accumulate. _fuse_ff gates it;
  // ff.linear_in / ff_context.linear_in are row-interleaved at load when on.
  metal_compute::ComputeFunction _fn_ff_swiglu, _fn_ff_swiglu_a16,
      _fn_qmm_swiglu4_bm64, _fn_qmm_swiglu8_bm64,
      _fn_qmm_swiglu4_bm64_a16, _fn_qmm_swiglu8_bm64_a16;
  // Strided-output twins (write the mlp straight into scat[:, H:], no concat):
  // quant qmm _rs (+acc16) + the unfused swiglu_rs.
  metal_compute::ComputeFunction _fn_qmm_swiglu4_bm64_rs, _fn_qmm_swiglu8_bm64_rs,
      _fn_qmm_swiglu4_bm64_rs_a16, _fn_qmm_swiglu8_bm64_rs_a16, _fn_swiglu_rs;
  bool _fuse_ff = false;
  bool _ff_acc16 = true;

  // Steel flash-attention (head_dim 128, GQA-aware). Best-effort; scalar SDPA
  // fallback when absent or VPIPE_FLUX2_NO_STEEL_ATTN is set.
  metal_compute::ComputeLibrary _lib_attn;
  metal_compute::SharedBuffer _attn_params;
  bool _steel_attn_ok = false;
  // M5 matrix-core steel attention (attn_steel_nax_h_bd128); falls back to the
  // ALU steel (attn_steel_h_bd128) off matrix-core GPUs. VPIPE_FLUX2_NO_ATTN_NAX.
  metal_compute::ComputeLibrary _lib_attn_nax;
  bool _use_attn_nax = false;

  // M5 matrix-core matmul2d for the block/projection GEMMs (mirrors Krea-2):
  // dense weights feed dense_gemm_mma directly; quantized weights are dequant-
  // expanded into the reusable f16 scratch _w_deq (affine_dequant) then run the
  // SAME dense matmul2d -- the dequant-once -> dense-GEMM shape, which matches
  // affine_qmm_steel to f32 rounding. Gated on matrix cores (M4/older keep
  // steel); VPIPE_FLUX2_NO_MMA2 A/B off. gemm_mma_ encodes the dequant (when
  // quantized) + matmul; returns false -- caller keeps its steel path -- when
  // off-gate or M/N too small to amortize the 128-row tile.
  bool gemm_mma_(metal_compute::ComputeEncoder& enc,
                 const metal_compute::SharedBuffer& x, std::size_t xe,
                 const QWeight& w, const metal_compute::SharedBuffer& y,
                 std::size_t ye, int M, int N, int K);
  metal_compute::ComputeLibrary _lib_dense_mma, _lib_dequant;
  metal_compute::ComputeFunction _fn_dense_mma, _fn_dense_mma_deep,
      _fn_dense_mma_tn2, _fn_dense_mma_splitk, _fn_dequant4, _fn_dequant8;
  // Dynamic-int8 accelerated GEMMs (Config::i8_gemm / VPIPE_I8_GEMM);
  // null when off. Tried first in gemm_mma_ for qualifying shapes.
  std::unique_ptr<I8GemmContext> _i8;
  bool _use_mma2 = false;
  int  _mma_min_m = 64;   // matmul2d only wins once M amortizes the 128 tile
  metal_compute::SharedBuffer _w_deq;   // reusable [N,K] f16 dequant scratch
  // Very deep K (the single-stream to_out: K = H + SMLP = 16384 on the 9B)
  // runs split-K (grid.z = K/kSplitKC partial planes) + a residual_add fold --
  // the single-op full reduction sits on the deep-K cliff (see Krea-2).
  // _splitk holds the [splits, M, N] partial planes. VPIPE_FLUX2_NO_SPLITK.
  static constexpr int kSplitKC = 8192;
  metal_compute::SharedBuffer _splitk;
  bool _use_splitk = false;

  // AWQ calibration: one f16 [rows*dim] abs-max accumulator per group (live
  // while _calib_on), tapped by col_absmax_f16 at each quantizable Linear input.
  bool _calib_on = false;
  metal_compute::ComputeFunction _fn_colabsmax;
  std::map<std::string, metal_compute::SharedBuffer> _calib_acc;
};

}  // namespace genai
}  // namespace vpipe

#endif

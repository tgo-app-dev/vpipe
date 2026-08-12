#ifndef GENERATIVE_MODELS_WAN_METAL_WAN_TRANSFORMER_H
#define GENERATIVE_MODELS_WAN_METAL_WAN_TRANSFORMER_H

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/flex-data.h"
#include "generative-models/shared/dit-block-progress.h"

#include <memory>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

class WeightSet;   // generative-models/weight-set.h

// The Wan denoiser (WanTransformer3DModel): the 14B DiT at the centre of
// every Wan 2.1/2.2 video model, run in BF16 on the metal-compute backend.
//
// Unlike the MMDiTs already in this tree (Krea-2 / Qwen-Image-Edit /
// FLUX.2), this is a SINGLE-STREAM transformer: the text never joins the
// residual stream. Each block is self-attention over the video tokens,
// then CROSS-attention into the fixed 512-token text conditioning, then a
// feed-forward -- so the text is 512 keys, not 512 more rows of the
// sequence, and the video token count alone sets the attention cost.
//
// Four details are this model's and not shared with the MMDiT siblings:
//
//   * the norms are LayerNorm (mean subtracted), not RMSNorm, and the
//     reference runs them in fp32. norm1/norm3 carry no weights at all
//     (elementwise_affine=False) -- a missing-tensor check that keys on
//     "every norm has a gamma" will read that as a broken checkpoint.
//   * q/k normalization is `rms_norm_across_heads`: one RMS over the whole
//     5120-wide projection BEFORE the head split, not the usual per-head
//     norm. Normalizing per head would be a different function.
//   * the feed-forward is UNGATED -- gelu(proj(x)) through one 13824-wide
//     hidden -- where every other DiT here is SwiGLU or GEGLU.
//   * RoPE is 3-axis over (frame, row, column) with the head split
//     44/42/42 (h_dim = w_dim = 2*(head_dim/6), t_dim takes the
//     remainder), adjacent-pair, applied to self-attention only. The
//     cross-attention into text carries no position signal.
//
// Conditioning is CHANNEL-WISE for image-to-video: Wan 2.2 dropped the
// CLIP image tower Wan 2.1-I2V had (`image_dim` null), so `in_channels`
// 36 = 16 noise + 4 first-frame mask + 16 VAE latent of the conditioning
// image, and there is no image cross-attention path. A text-to-video
// checkpoint is the same class with in_channels 16; both load here.
//
// A14B ships as TWO experts -- `transformer/` (high noise) and
// `transformer_2/` (low noise) -- switched at the scheduler's
// boundary_ratio. They are separate checkpoints, so they are separate
// WeightSets and separate instances of this class; at bf16 one is ~28 GB,
// which is why the caller holds one at a time rather than both.
class MetalWanTransformer {
 public:
  struct Config {
    int   n_heads      = 40;
    int   head_dim     = 128;
    int   hidden       = 5120;    // n_heads * head_dim
    int   ffn          = 13824;
    int   n_layers     = 40;
    int   in_channels  = 36;      // 16 noise + 4 mask + 16 image (I2V)
    int   out_channels = 16;
    int   text_dim     = 4096;    // umT5-XXL width
    int   freq_dim     = 256;     // sinusoidal timestep width
    int   patch_t      = 1;
    int   patch_h      = 2;
    int   patch_w      = 2;
    float norm_eps     = 1e-6f;
    double rope_theta  = 10000.0;

    // The 3-axis RoPE split, derived exactly as the reference derives it:
    // h = w = 2*(head_dim/6), t = head_dim - h - w. 44/42/42 at 128.
    int rope_h() const { return 2 * (head_dim / 6); }
    int rope_w() const { return 2 * (head_dim / 6); }
    int rope_t() const { return head_dim - 2 * (2 * (head_dim / 6)); }
    int patch_elems() const { return in_channels * patch_t * patch_h * patch_w; }
    int out_patch_elems() const
    {
      return out_channels * patch_t * patch_h * patch_w;
    }
  };

  // Read a Config out of a diffusers `transformer/config.json`. False when
  // the file is missing or is not a WanTransformer3DModel config.
  static bool config_from_json(const std::string& dit_dir, Config& out,
                               std::string* err = nullptr);

  // What a CALLER chooses per generation, as against Config, which is what
  // the checkpoint IS. It lives here rather than in the driving stage
  // because every field is a fact about this family: a stage that also
  // drives other video DiTs would otherwise carry Wan's guidance rules,
  // Wan's expert boundary and the next family's knobs side by side, and
  // have to know which apply to what is resident.
  struct GenerationParams {
    // Classifier-free guidance. Wan is NOT distilled: without a negative
    // conditioning there is nothing to guide away from, and the caller
    // forces this to 1 rather than running a second forward for nothing.
    double guidance_scale   = 3.5;
    // The LOW-noise expert's, on a two-expert checkpoint (A14B). Two
    // scales rather than one because the pair is trained that way -- the
    // experts see different sigma ranges and want different guidance.
    double guidance_scale_2 = 3.5;
    // The sigma at which the low-noise expert takes over. 0 = a single
    // expert, which is what every other family here is.
    double boundary_ratio   = 0.9;
    // Whether `boundary_ratio` above was ASKED for or is just the
    // default. It decides who wins against the checkpoint's own
    // model_index.json: an explicit choice does, a default does not.
    // Without this the file would always win and the key would be inert.
    bool boundary_ratio_set = false;

    // Parse the `wan2-model-config` beat. Absent keys keep the defaults
    // above; `err` collects what was ignored, and is not fatal.
    static GenerationParams from_flex(const FlexData& fd,
                                      std::string* err = nullptr);

    // Which expert `sigma` belongs to: 0 = high noise (`transformer`),
    // 1 = low noise (`transformer_2`). Always 0 when there is no
    // boundary. The schedule descends, so a loop asking this every step
    // crosses at most once.
    int expert_for(double sigma) const
    {
      return (boundary_ratio > 0.0 && sigma < boundary_ratio) ? 1 : 0;
    }
    double guidance_for(int expert) const
    {
      return expert == 1 ? guidance_scale_2 : guidance_scale;
    }
  };

  // `boundary_ratio` as the CHECKPOINT states it. It lives in the
  // pipeline's model_index.json rather than in either expert's own
  // config, because it is a property of the pair; an explicit null there
  // means a single-expert pipeline and reads as 0. Returns `fallback`
  // when the file is missing or unreadable.
  static double boundary_ratio_from_index(const std::string& root,
                                          double fallback);

  // Prefer the WeightSet overload: the set is the manager's shared,
  // reference-counted view of the checkpoint. The dir overload opens a
  // PRIVATE set (tests, and callers with no session to ask).
  static std::unique_ptr<MetalWanTransformer>
  load(const std::string& dit_dir, metal_compute::MetalCompute* mc,
       const Config& cfg);

  static std::unique_ptr<MetalWanTransformer>
  load(std::shared_ptr<WeightSet> ws, metal_compute::MetalCompute* mc,
       const Config& cfg);

  ~MetalWanTransformer();   // out-of-line: _ws is a fwd-declared type

  // Project the text encoder's output into the transformer width:
  // [text_seq, text_dim] bf16 -> [text_seq, hidden] bf16. Timestep-
  // independent, so a caller runs this ONCE per prompt and passes the
  // result to every forward() of the denoise loop (twice per generation
  // under classifier-free guidance -- once for the prompt, once for the
  // negative). Empty on failure, with a reason in `err`.
  metal_compute::SharedBuffer
  encode_text(const metal_compute::SharedBuffer& text, int text_seq,
              std::string* err = nullptr);

  // One denoiser evaluation.
  //
  // `latents` is the CHANNEL-FIRST [in_channels, T, h, w] bf16 input --
  // the same layout MetalWanVae uses for a latent video, with the I2V
  // conditioning channels already concatenated by the caller. `text_proj`
  // is encode_text()'s output. `timestep` is the scheduler's timestep on
  // its native 0..num_train scale (i.e. sigma * 1000), NOT the sigma.
  //
  // Returns the predicted velocity, channel-first [out_channels, T, h, w]
  // bf16. Empty on failure, with a reason in `err`.
  metal_compute::SharedBuffer
  forward(const metal_compute::SharedBuffer& latents, int T, int h, int w,
          const metal_compute::SharedBuffer& text_proj, int text_seq,
          float timestep, std::string* err = nullptr);

  const Config& config() const { return _cfg; }

  // Cap on the quantized-GEMM tile height: 0 = BM32 only, 1 = +BM64,
  // 2 = +BM128 (the default, when the checkpoint's group size has the
  // wide twins). Exposed because the only way to compare two tiles
  // without the ~4% thermal spread between processes is to ALTERNATE
  // them inside one, which needs the choice to be settable per call.
  // Clamped to what was actually built.
  void set_qmm_tile(int cap);
  int qmm_tile() const { return _qmm_tile; }

  // Per-block progress, fired as each block is entered. One forward of
  // this model at video resolution is minutes of otherwise-silent work,
  // so a caller showing movement inside a single denoise step is not a
  // nicety.
  void set_block_progress(DitBlockProgressFn fn)
  {
    _block_progress = std::move(fn);
  }

 private:
  MetalWanTransformer() = default;

  // A Linear: either a dense bf16 [N, K] matrix, or an MLX-affine
  // quantized triple (U32 `codes` [N, K*bits/32] + `scales`/`qbias`
  // [N, K/group]), plus the bias every projection in this model has
  // (T5's have none; these all do). load_linear_ auto-detects which by
  // looking for a `<name>.scales` sibling, so one checkpoint can be
  // dense and another quantized with no config to keep in sync.
  struct Linear {
    metal_compute::SharedBuffer w, b;
    metal_compute::SharedBuffer codes, scales, qbias;
    bool quantized = false;
    int  bits = 0;                       // 4 or 8, per weight (mixed ok)
    bool empty() const { return quantized ? codes.empty() : w.empty(); }
  };

  struct Block {
    Linear q1, k1, v1, o1;                       // self-attention
    metal_compute::SharedBuffer qn1, kn1;        // RMS over the full width
    Linear q2, k2, v2, o2;                       // cross-attention into text
    metal_compute::SharedBuffer qn2, kn2;
    metal_compute::SharedBuffer n2w, n2b;        // norm2 (LayerNorm, affine)
    Linear ff_in, ff_out;                        // 5120 -> 13824 -> 5120
    metal_compute::SharedBuffer sst;             // scale_shift_table [6, H]
  };

  // A checkpoint tensor as bf16. The Wan checkpoints ship FP32, so this is
  // a genuine transform and goes through derived() rather than tensor():
  // the model keeps the converted bytes for its whole life and the source
  // f32 is dropped, which is exactly what a derived entry is.
  metal_compute::SharedBuffer weight_(WeightSet& ws, const std::string& nm);
  Linear linear_(WeightSet& ws, const std::string& nm);
  bool load_block_(WeightSet& ws, int i, Block& b);

  // y[M,N] = x[M,K] @ w[N,K]^T (+ bias[N] when `l.b` is present).
  void gemm_(metal_compute::ComputeEncoder& enc,
             const metal_compute::SharedBuffer& x, const Linear& l,
             const metal_compute::SharedBuffer& y, int M, int N, int K);

  // The [seq, head_dim] f32 cos/sin tables for one (T, h, w) patch grid.
  // f32 on purpose: RoPE error is STRUCTURED, so bf16 tables would compound
  // across 40 blocks and every denoise step rather than average out.
  void build_rope_(int T, int ph, int pw, metal_compute::SharedBuffer& cos_out,
                   metal_compute::SharedBuffer& sin_out) const;

  // Per-shape scratch, rebuilt only when the token grid changes -- the
  // denoise loop runs the same shape every step.
  struct Scratch {
    int seq = -1, text_seq = -1;
    metal_compute::SharedBuffer rcos, rsin;
    metal_compute::SharedBuffer x, joint, nm, qb, kb, vb;
    metal_compute::SharedBuffer qh, kh, vh, oh, ob, ffb, outp;
    metal_compute::SharedBuffer tk, tv, tkh, tvh;
    metal_compute::SharedBuffer te_in, te1, temb, tproj, mod, mod2;
  };
  bool ensure_scratch_(int T, int ph, int pw, int text_seq);
  Scratch _s;

  metal_compute::MetalCompute* _mc = nullptr;
  Config _cfg;
  std::shared_ptr<WeightSet> _ws;
  DitBlockProgressFn _block_progress;

  Linear _patch;                                // patch_embedding (as Linear)
  Linear _time1, _time2, _time_proj;            // condition_embedder time path
  Linear _text1, _text2;                        // condition_embedder text path
  std::vector<Block> _blocks;
  metal_compute::SharedBuffer _final_sst;       // scale_shift_table [2, H]
  Linear _proj_out;

  // Quantized checkpoints: bits/group come from the transformer's
  // config.json `quantization` block (the loader still auto-detects the
  // per-tensor width, so a mixed 4/8 checkpoint runs as-is).
  int _quant_bits = 0;                   // 0 = dense
  int _quant_group = 64;

  metal_compute::ComputeLibrary _lib_gemm, _lib_elt, _lib_rms, _lib_rope,
      _lib_attn, _lib_qmm;
  // The steel affine GEMM in three tile heights. Which one runs is chosen
  // per CALL from M: the video sequence is thousands of rows, where the
  // BM=32 tile re-reads each weight tile once per 32 rows and spends the
  // step on weight traffic rather than arithmetic. 0 = only BM32 built
  // (g32 checkpoints, which have no wide twins), 1 = +BM64, 2 = +BM128.
  metal_compute::ComputeFunction _fn_qmm4, _fn_qmm8;
  metal_compute::ComputeFunction _fn_qmm4_bm64, _fn_qmm8_bm64;
  metal_compute::ComputeFunction _fn_qmm4_bm128, _fn_qmm8_bm128;
  int _qmm_tile = 0;
  metal_compute::ComputeFunction _fn_gemm, _fn_rms, _fn_ln_mod, _fn_ln_affine,
      _fn_gelu, _fn_residual, _fn_gated, _fn_transpose, _fn_trope, _fn_silu,
      _fn_sdpa, _fn_bias_add;
  // Steel flash-attention (attn_steel_h_bd128_bf16). Two param blocks: the
  // self-attention shape (qL = kL = seq) and the cross-attention one
  // (qL = seq, kL = text_seq), which differ in the alignment function
  // constants as well as the lengths.
  metal_compute::SharedBuffer _attn_p_self, _attn_p_cross;
  metal_compute::ComputeFunction _fn_attn_self, _fn_attn_cross;
  int _attn_seq = -1, _attn_kv = -1;   // shape the two above were built for
  bool _steel_ok = false;
};

}  // namespace genai
}  // namespace vpipe

#endif

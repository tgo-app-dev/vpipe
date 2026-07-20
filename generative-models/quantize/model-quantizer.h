#ifndef VPIPE_GENAI_QUANTIZE_MODEL_QUANTIZER_H
#define VPIPE_GENAI_QUANTIZE_MODEL_QUANTIZER_H

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace vpipe::metal_compute {
class MetalCompute;
}

namespace vpipe::genai {

struct QuantizeOptions {
  int   bits  = 8;            // backbone affine bit-width (4 | 8)
  int   group = 64;           // affine group size (32 | 64)
  float clip  = 1.0f;         // 1.0 = plain min/max (AWQ/unsloth clip hook)
  // Linear "leaf" names (the dotted segment just before ".weight") to
  // quantize; every other tensor passes through verbatim. Empty => the
  // default set (standard attention + MLP projections shared by the
  // Llama / Qwen / MOSS backbones). Prefix-agnostic, so it works across
  // checkpoints regardless of the module path.
  std::vector<std::string> quant_linears;
  std::uint64_t shard_max_bytes = 5ull << 30;

  // Optional submodule SCOPE: when non-empty, ONLY tensors whose full name
  // contains this substring are eligible to quantize; everything else passes
  // through verbatim. Lets a multi-modal LLM quantize one part in isolation --
  // e.g. scope "language_model." to quantize just the text backbone and leave
  // the (precision-sensitive) vision/audio towers bf16, or scope "visual." to
  // quantize just the vision tower. Empty (default) => the whole model.
  std::string quant_scope;
  // Broadened leaf rule for a scoped submodule: when true (with quant_scope
  // set), quantize EVERY 2D fp weight in scope whose leaf is not a norm /
  // embedding (ignore quant_linears). Vision/audio towers name their linears
  // differently (qkv / linear_fc1 / proj / ...), so the standard leaf set
  // misses them; this quantizes the tower wholesale. Off => the leaf set still
  // gates (right for the language backbone, whose leaves are standard).
  bool quant_all_in_scope = false;

  // Also quantize the token-embedding table + (untied) lm_head, matching the
  // MLX convention (its checkpoints quantize embed/lm_head). Needed so a
  // standard LM (Qwen/Llama) RELOADS for inference -- the affine load path
  // expects them quantized. OFF by default so backbone-only models (MOSS,
  // whose wrapper keeps a bf16 embed for its host-side gather + audio embeds)
  // are unaffected.
  bool quant_embeddings = false;

  // Zero-centered RMSNorm convention (the raw-HF Qwen3.5/3.6 family): the
  // checkpoint stores RMSNorm weights centered at 0 and the model applies
  // (1 + weight). vpipe's RMSNorm kernel multiplies by `weight` directly
  // (correct for MLX-converted checkpoints, which pre-fold the +1), so when
  // quantizing a raw-HF checkpoint we must add 1.0 to the affected norm
  // weights (input_layernorm, post_attention_layernorm, q_norm, k_norm, and
  // the final model norm) on the way out. The GATED GDN norm
  // (linear_attn.norm, ones-init, applied as `weight`) is excluded.
  bool norm_offset = false;

  // SmoothQuant activation-aware smoothing. When enabled, the quantizer runs
  // a LAYER-AWARE pass over <layer_prefix>layers.N: it derives per-input-
  // channel smoothing scales from calibration activation stats (calib_dir) +
  // the weight magnitudes, and folds them fp-equivalently (qkv-input <- norm,
  // gate/up-input <- norm, down-input <- up rows) before quantizing -- moving
  // quant difficulty off the spiky channels. (o/v GQA fold deferred.)
  bool        smoothquant   = false;
  float       smooth_alpha  = 0.5f;
  // AWQ weight clipping (the second half of the AWQ recipe, paired with the
  // scale search). When set (with smoothquant), each group's [min,max] is
  // shrunk toward its midpoint by the per-group clip ratio that minimizes the
  // activation-weighted group-affine reconstruction error -- saturating a
  // group's outliers to tighten the step for its inliers. Host-side (the fold
  // pass already holds f32 weights); the clamped weights quantize at clip=1.
  // OFF by default: on MOSS-v1.5 (+ the small calib) it REGRESSES end drift on
  // top of the scale search (proxy-vs-end divergence; see the
  // awq_reduces_4bit_drift test). Kept opt-in -- standard AWQ, may help other
  // models / larger calibration sets.
  bool        awq_clip      = false;
  std::string calib_dir;            // dir with calib_{qkv,o,gateup,down}.f32
  std::string layer_prefix  = "transformer.layers.";  // arch layer root
  int         n_layers      = 0;    // >0 required when smoothquant

  // unsloth-style mixed precision (sensitivity quant). When enabled, bit
  // width is assigned per-LAYER: every quantizable linear in a layer shares
  // one width (the runtime's mixed path corrupts a layer whose projections
  // carry DIFFERENT widths -- the fused q|k|v and gate|up groups assume a
  // uniform width -- and per-layer assignment is what unsloth / llama.cpp do).
  // Layers are ranked by aggregate sensitivity = sum over the layer's linears
  // of how much promotion base->high helps (the group-affine reconstruction-
  // error drop); the most-sensitive `mixed_frac` fraction of layers is
  // promoted to `high_bits`. The result is a genuine mixed checkpoint (the
  // loader auto-detects per-tensor bits from each weight's packed-column
  // count). The mixed-affine decode kernels are w{4,8}g64 only, so mixed
  // REQUIRES bits=4, high_bits=8, group=64, and n_layers>0 + layer_prefix.
  bool  mixed      = false;
  int   high_bits  = 8;
  float mixed_frac = 0.25f;          // fraction of LAYERS promoted to high_bits

  // Leaves (weight_leaf_ names) to quantize at `high_bits` instead of `bits`,
  // independent of `mixed`. For a precision-sensitive tensor type that is too
  // lossy at 4-bit but fine at 8-bit -- e.g. the Qwen-Image-Edit AdaLN
  // modulation ("1"): body @ w4, modulation @ w8. The loader auto-detects the
  // per-tensor bit width, so a checkpoint mixing widths loads + infers as-is.
  std::vector<std::string> high_bit_leaves;

  // Krea-2 DiT activation-aware weight CLIPPING (the fold-free half of AWQ; the
  // adaLN smoothing fold is obstructed by the shared timestep-dependent
  // time_mod_proj shift). When set, each quantized transformer_blocks.{L}
  // Linear's weight groups are clipped by awq_clip_search using its per-input-
  // channel activation from `calib_dir` (calib_{qkv,o,gateup,down}.f32, produced
  // by collect_dit_calibration) before quantizing at clip=1. Non-block DiT
  // Linears (img_in / txt_in / time_embed / text_fusion / final) quantize plain.
  // Requires calib_dir + n_layers (the main-block count).
  bool  dit_awq    = false;
  // DiT family for the AWQ activation mapping: "" / "krea2" -> the Krea-2
  // calib_{qkv,o,gateup,down}.f32 scheme; "flux2" -> per-group files
  // (dbl_*/sgl_*/emb_* from collect_flux2_calibration) mapped by the FLUX
  // topology (double/single blocks + embedders). Clip-only for flux2 (the
  // exact ff.down<-ff.up fold is Krea-2-specific).
  std::string dit_family;
};

// M0 per-tensor group-affine quantizer. Opens a source safetensors model,
// quantizes the configured linear weights to the MLX-affine layout
// (U32 codes + F16 scales + F16 biases), passes every other tensor through
// byte-for-byte (embeddings, heads, norms, the MOSS local transformer),
// rewrites config.json's top-level `quantization` block, and copies the
// remaining sidecar files (tokenizer, etc.). The output reloads via
// MetalLlamaWeights. Each tensor is processed one at a time, so peak RAM is
// ~one tensor -- already friendly to large models. Calibration / SmoothQuant
// / AWQ / per-layer streaming calibration arrive in later milestones.
class ModelQuantizer {
public:
  explicit ModelQuantizer(metal_compute::MetalCompute* mc) : _mc(mc) {}

  // Quantize in_dir -> out_dir. Returns false and sets *err on any failure.
  // `stop` is polled at the bulk per-tensor / per-layer loop boundaries so a
  // pipeline stop request aborts the (long) quantize cooperatively.
  bool run(const std::string& in_dir, const std::string& out_dir,
           const QuantizeOptions& opt, std::string* err,
           const std::function<bool()>& stop = [] { return false; }) const;

  static std::vector<std::string> default_quant_linears();

private:
  metal_compute::MetalCompute* _mc = nullptr;
};

}  // namespace vpipe::genai

#endif

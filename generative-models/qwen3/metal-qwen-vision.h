#ifndef VPIPE_GENERATIVE_MODELS_METAL_QWEN_VISION_H
#define VPIPE_GENERATIVE_MODELS_METAL_QWEN_VISION_H

// MetalQwenVisionEncoder -- procedural Qwen3-VL vision tower (ViT) on the
// metal-compute framework, no MLX in the forward. The dense counterpart
// of the MLX Qwen3VLVisionEncoder (generative-models/shared/vision-encoder.cc):
// CPU preprocess (smart-resize + bilinear + normalize) -> merger-order
// patchify -> patch-embed GEMM -> bilinear pos-embed -> N pre-LN ViT
// blocks (LayerNorm, dense qkv, 2D-RoPE, full SDPA, dense proj, GELU-tanh
// MLP) -> patch merger (LayerNorm, dense fc1, GELU-erf, dense fc2) ->
// [n_tokens, out_hidden] image embeddings, ready to splice into the LM at
// the image-token placeholders. Vision weights are dense BF16 (converted
// to F16 at load); every kernel it calls is oracle-verified in
// tests/unit-tests/metal-compute-llm-ops.cc.

#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe { class SessionContextIntf; }
namespace vpipe::metal_compute { class MetalCompute; }

namespace vpipe::genai {

struct ModelConfig;
class WeightSet;       // generative-models/weight-set.h

class MetalQwenVisionEncoder {
public:
  struct Config {
    int depth          = 24;
    int hidden         = 1024;
    int n_heads        = 16;
    int patch_size     = 16;
    int spatial_merge  = 2;
    int temporal_patch = 2;
    int out_hidden     = 2560;
    int num_pos_embed  = 2304;   // G*G learned position table (G=48)
    int intermediate   = 4096;   // ViT MLP inner; also merger fc1 out
    float image_mean[3] = {0.5f, 0.5f, 0.5f};
    float image_std[3]  = {0.5f, 0.5f, 0.5f};
    float ln_eps       = 1e-6f;
    // smart_resize pixel bounds (preprocessor_config.json size.shortest_edge /
    // size.longest_edge). The defaults are the Qwen2/3-VL processor defaults
    // the LM path has always used; a repo that ships its own bounds (e.g.
    // Mage-Flow's 65536 / 16777216) must set them or a small / very wide
    // reference image is left un-upscaled where the reference upscales it.
    int min_pixels     = 56 * 56;
    int max_pixels     = 28 * 28 * 1280;
    // The VIDEO processor's own bounds, which are a different pair from
    // the image ones in every checkpoint that ships both (MiniMax-H3:
    // 65536 / 16777216 for images, 4096 / 25165824 for video). They also
    // budget differently -- the frame count multiplies into the pixel
    // total -- so a clip resized by the image rule is silently wrong
    // rather than merely differently sized. Defaults are the
    // Qwen3VLVideoProcessor signature's.
    std::int64_t video_min_pixels = 128 * 128;
    std::int64_t video_max_pixels = 16 * 16 * 2 * 2 * 2 * 6144;
    // When non-empty, the tower loads from this mmproj-*.gguf (llama.cpp
    // CLIP layout) instead of the safetensors `model_dir`. Same BF16/F32
    // weights, just CLIP tensor names + a split patch-embed conv.
    std::string gguf_mmproj;
    // Safetensors weight-name prefix for the tower. The Qwen3-VL LM checkpoints
    // store it under "vision_tower." (default); the Krea-2 diffusion text
    // encoder (a qwen3_vl checkpoint) stores the same tower under "visual.".
    std::string weight_prefix = "vision_tower.";
    // Qwen3-VL deepstack: ViT block indices whose post-block hidden states are
    // additionally merged (postshuffle-norm merger) into extra visual features
    // injected into the LM at layers 0..N-1. Empty => no deepstack (the tower
    // emits only the final merger tokens). Populated from config_from.
    std::vector<int> deepstack_indexes;
    // Storage/compute element type for the whole tower: f16 (default) or
    // bf16. This is a FIDELITY-TO-THE-REFERENCE knob, not an accuracy one --
    // f16 is ~3x closer to an fp32 oracle than bf16 (0.0142 vs 0.0425 rel-L2
    // on this tower: bf16 has 8 mantissa bits to f16's 11, and its range
    // advantage buys nothing when the tower peaks near 2238 against f16's
    // 65504 ceiling). Set it when the point is to REPRODUCE a reference that
    // runs bf16 -- Mage-Flow's pipeline casts its whole text encoder, tower
    // included, so its conditioning carries bf16 tower noise that an f16
    // tower cannot match no matter how accurate it is.
    // NOTE: the steel / NAX flash-attention kernels are f16-only, so a bf16
    // tower falls back to the scalar (or matmul2d) SDPA and is slower.
    bool use_bf16 = false;
    int head_dim() const { return hidden / n_heads; }   // 64
  };

  struct Result {
    // Native compute-dtype [n_tokens * out_hidden] row-major, on the GPU
    // (UMA). The ELEMENT TYPE follows the encoder: f16, or bf16 when it was
    // built with Config::use_bf16 -- ask is_bf16(), do not assume. Both are
    // 2 bytes, so a consumer that guesses wrong reads plausible-looking
    // garbage rather than crashing.
    // Splice straight into the LM via prefill_multimodal_metal -- no
    // host f32 round-trip. Empty buffer / n_tokens==0 on failure.
    metal_compute::SharedBuffer embeddings;
    int n_tokens   = 0;
    int out_hidden = 0;
    int grid_h     = 0;
    int grid_w     = 0;
    // Temporal grid cells: 1 for an image, `frames / temporal_patch` for
    // a clip. The consumer needs it because Qwen3-VL labels a video ONE
    // BLOCK PER CELL, so this is the number of `<|vision_start|>` runs
    // the prompt has to carry, not a detail of the tensor's shape.
    int grid_t     = 1;
    // The element type of `embeddings` and `deepstack`, carried on the
    // RESULT so a consumer that outlives the encoder handle (or holds
    // results from several) does not have to ask a tower which one made
    // this. Both types are two bytes, so a wrong guess reads plausible
    // garbage rather than crashing.
    bool is_bf16   = false;
    // Qwen3-VL deepstack features, one per config.deepstack_indexes entry
    // (in that order): native f16 [n_tokens * out_hidden]. Empty when the
    // tower has no deepstack mergers. Added to the LM hidden states at the
    // image-token rows after LM layers 0,1,... (see MetalQwenModel forward).
    std::vector<metal_compute::SharedBuffer> deepstack;
  };

  static std::unique_ptr<MetalQwenVisionEncoder> load(
      const std::string& model_dir,
      metal_compute::MetalCompute* mc,
      const Config& cfg);

  // Preferred: the tower lives in the SAME checkpoint directory as the
  // LM it feeds, so passing the LM's set means the ViT's weights are
  // cached beside the LM's rather than in a second private mmap of the
  // same shards. `ws` may be null when cfg.gguf_mmproj names an mmproj
  // GGUF -- that path reads its own file and has no model dir at all.
  //
  // The returned tower KEEPS the set (its tensors are views into it).
  static std::unique_ptr<MetalQwenVisionEncoder> load(
      std::shared_ptr<WeightSet> ws,
      metal_compute::MetalCompute* mc,
      const Config& cfg);

  // Derive the ViT Config from a parsed ModelConfig's vision_config so
  // the tower self-sizes to any family member (the 9B ViT is depth 27 /
  // hidden 1152 / out_hidden 4096 vs the 4B's 24 / 1024 / 2560). Single
  // source of truth, shared by the production loader and the tests.
  static Config config_from(const ModelConfig& c);

  // Encode one RGB image: rgb is [3, H, W] U8 channel-first contiguous.
  Result encode(const std::uint8_t* rgb, int H, int W);

  // Encode a CLIP: `rgb` is `n_frames` consecutive [3, H, W] U8 planes.
  //
  // Frames are merged in groups of `temporal_patch` (2), so a clip of
  // 2n frames becomes n grid cells and each cell is one vision block in
  // the prompt. An ODD count repeats the last frame, which is what the
  // reference's processor does rather than dropping it -- so the caller
  // gets ceil(n_frames / 2) cells either way and never silently loses
  // the tail.
  //
  // The clip is resized by the VIDEO rule (Config::video_*_pixels), in
  // which the frame count is part of the pixel budget: the same frames
  // passed one at a time through encode() would land on a different
  // canvas and produce different tokens.
  Result encode_video(const std::uint8_t* rgb, int n_frames, int H, int W);

  const Config& config() const { return _cfg; }
  // The element type of Result::embeddings / deepstack (and of every weight
  // buffer). Reflects the A/B env overrides too, so it is the truth rather
  // than a restatement of Config::use_bf16.
  bool is_bf16() const noexcept { return _bf16; }

  // Attach a session so encode() is recorded on the profiler's LLM lane
  // (vision-tower category) when no CoreML tower is configured and the
  // ViT runs on the GPU. Optional; without it the encode still runs but
  // is invisible to the profiler. Mirrors MetalGemma4VisionEncoder.
  void set_session(const SessionContextIntf* s) { _session = s; }

private:
  MetalQwenVisionEncoder() = default;

  // The one implementation behind encode() and encode_video().
  // `n_frames == 1` is the image path: the single frame is tiled across
  // the temporal patch, which is what the reference does too.
  Result encode_(const std::uint8_t* rgb, int n_frames, int H, int W,
                 bool video);

  struct Block {
    metal_compute::SharedBuffer n1w, n1b, qkvw, qkvb, ow, ob;
    metal_compute::SharedBuffer n2w, n2b, fc1w, fc1b, fc2w, fc2b;
  };

  Config _cfg;
  metal_compute::MetalCompute* _mc = nullptr;
  const SessionContextIntf* _session = nullptr;   // profiler, optional
  metal_compute::ComputeLibrary _lib_gemm, _lib_vis, _lib_elt, _lib_sdpa,
      _lib_attn, _lib_dense_mma, _lib_sdpa_mma, _lib_attn_nax;
  metal_compute::ComputeFunction _fn_gemm, _fn_ln, _fn_gelu_tanh, _fn_gelu_erf,
      _fn_vrope, _fn_sdpa_full, _fn_head_slice, _fn_transpose, _fn_residual;
  // Steel MMA dense GEMM (fast path; the scalar _fn_gemm is the fallback
  // when this isn't available). Bias is folded into the kernel.
  metal_compute::ComputeFunction _fn_gemm_t;
  // M5 matrix-core (matmul2d) dense GEMM -- the same NAX path the LM prefill
  // uses (dense_gemm_mma, 128x128 for K<6144 else 128x256). The kernel omits
  // bias, so _fn_bias_add folds the vision linear bias back over the rows.
  // _use_mma2 gates it (tensor GPU only); else the steel/scalar path above.
  metal_compute::ComputeFunction _fn_dense_mma, _fn_dense_mma_deep, _fn_bias_add;
  bool _use_mma2 = false;
  // M5 matrix-core (matmul2d/NAX) non-causal flash attention for the ViT
  // (head_dim 64). Replaces the steel simdgroup_matrix attn -- the O(n^2)
  // term that dominates the tower at large grids. _use_mma2_attn gates it
  // (tensor GPU + hd==64); else the steel/scalar path stays.
  metal_compute::ComputeFunction _fn_sdpa_mma_d64;
  bool _use_mma2_attn = false;
  // MLX's matrix-core (NAX) steel flash attention -- the kernel MLX itself
  // dispatches on M5 (attn_steel_nax, bq=64/bk=32, register-resident softmax).
  // The preferred steel path when available; the ALU _lib_attn steel is the
  // M4 fallback. _use_attn_nax gates it (supports_matrix_cores + a usable bd).
  bool _use_attn_nax = false;
  // ZERO-PAD width for the attention head dim, or 0 for native. The flash
  // kernels are instantiated at bd64 / bd120 / bd128 only, so a tower whose
  // head_dim is none of those (Qwen3-VL 8B/9B: hidden 1152 / 16 heads = 72)
  // had NO flash kernel at all and fell to the scalar O(n^2) sdpa across every
  // block. Padding q/k/v out to bd128 with zeros is exact -- zero lanes add
  // nothing to the QK dot, and the widened V's padding columns are dropped on
  // the way back -- and it is the same trick the Boogu DiT uses for its
  // head_dim 120 (see kPadAttn there, where the padded route also measured
  // FASTER than a native bd120 instantiation, because the pad buys aligned
  // block loads). `scale` stays 1/sqrt(head_dim), not 1/sqrt(128).
  // VPIPE_QWEN_VISION_NO_ATTN_PAD forces the scalar path back.
  int _attn_pad_d = 0;
  metal_compute::ComputeFunction _fn_tr_pad, _fn_tr_unpad;
  // Config::use_bf16, cached: selects the _bf16 metallibs and the host-side
  // float <-> storage conversions. The kernels' ENTRY-POINT names are
  // unchanged (they keep their _f16 suffix); only the library differs.
  bool _bf16 = false;
  // Steel flash-attention (MMA) param block; the per-encode pipeline (with
  // align/causal function constants) is created in encode() from _lib_attn.
  metal_compute::SharedBuffer _attn_params;

  std::vector<Block> _blocks;
  metal_compute::SharedBuffer _pe_w, _pe_b;   // patch embed [hidden,1536],[hidden]
  metal_compute::SharedBuffer _pos_w;          // pos table [num_pos_embed, hidden] f16
  metal_compute::SharedBuffer _mnw, _mnb, _mfc1w, _mfc1b, _mfc2w, _mfc2b;
  // Qwen3-VL deepstack mergers (one per _cfg.deepstack_indexes). Same shape as
  // the main merger but POSTSHUFFLE norm (LayerNorm over S*S*hidden, applied to
  // the merged window), so `dnw`/`dnb` are mdim-wide (vs the main merger's
  // hidden-wide norm). Empty when the tower has no deepstack.
  struct DsMerger {
    metal_compute::SharedBuffer dnw, dnb, fc1w, fc1b, fc2w, fc2b;
  };
  std::vector<DsMerger> _ds;
  std::vector<float> _rope_inv_freq;           // [head_dim/4]
  int _feat_dim = 0;                           // temporal*patch*patch*3
  // The checkpoint, held for this tower's whole life: every weight above
  // is either an alias of a buffer the set owns or a view into its mmap.
  // Null on the GGUF-mmproj path, which owns its tensors outright.
  std::shared_ptr<WeightSet> _ws;
};

}  // namespace vpipe::genai

#endif

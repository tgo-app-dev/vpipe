#ifndef VPIPE_GENERATIVE_MODELS_MODEL_LOADER_H
#define VPIPE_GENERATIVE_MODELS_MODEL_LOADER_H

#include "common/session-member.h"


#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace vpipe {
class SessionContextIntf;
}

namespace vpipe::genai {

// Parsed view of a Hugging Face `config.json`. Holds the subset of
// fields the LLM subsystem cares about; unknown keys are ignored.
// Values are populated with defaults so a partial config still
// loads, but the caller is expected to validate that the relevant
// fields are sensible for its target architecture (e.g.
// LlamaModelExec asserts architecture == "LlamaForCausalLM",
// n_layers > 0, etc.).
struct ModelConfig {
  std::string  architecture;                // e.g. "LlamaForCausalLM"
  int          n_layers              = 0;   // hidden_layers
  // Multi-Token-Prediction (NextN/Eagle) draft layers bundled after the
  // main blocks. >0 means a GGUF carries `nextn_predict_layers` extra blocks
  // (at indices n_layers .. n_layers+num_nextn_layers-1) exposed as the MTP
  // head; n_layers already EXCLUDES them. 0 for every non-MTP checkpoint.
  int          num_nextn_layers      = 0;
  int          n_heads               = 0;   // attention_heads
  int          n_kv_heads            = 0;   // key_value_heads (GQA)
  int          head_dim              = 0;   // (derived if absent)
  int          hidden                = 0;   // hidden_size
  int          ffn_inner             = 0;   // intermediate_size
  int          vocab_size            = 0;
  float        rope_theta            = 0.0f;
  bool         tie_word_embeddings   = false;
  float        rms_eps               = 1e-5f;
  int          max_position_embeddings = 0;

  // Optional Llama-3-style RoPE frequency rescaling. `kind` is "llama3"
  // when this block is present and the model uses the long-context
  // rescaling described in the Llama 3.1 paper; empty otherwise. The
  // four floats and `original_max_position_embeddings` are only
  // meaningful when kind == "llama3".
  struct RopeScaling {
    std::string  kind;                       // "" or "llama3"
    float        factor                = 1.0f;
    float        low_freq_factor       = 1.0f;
    float        high_freq_factor      = 1.0f;
    int          original_max_position_embeddings = 0;
  };
  RopeScaling  rope_scaling;

  // Optional MLX-style quantization. `bits == 0` means "weights are
  // raw float (no quantization)"; `bits` of 4 or 8 with a `group_size`
  // means every nn.Linear (and the embedding table) is replaced by
  // (weight: uint32-packed, scales: fp16, biases: fp16) triples per
  // MLX's affine quantization scheme. The loader doesn't dequantize;
  // it just surfaces the metadata so LlamaModelExec knows to call
  // quantized_matmul / dequantize against the .scales/.biases
  // sidecars.
  struct Quantization {
    int          bits        = 0;
    int          group_size  = 0;
    // True for symmetric quants repacked to affine with a REDUNDANT bias
    // (GGUF Q4_0: bias == -8*scale, exact in fp16). Lets the decode GEMVs
    // skip the bias read (scale-only); asymmetric MLX-affine leaves it false.
    bool         symmetric   = false;
    bool         enabled() const noexcept { return bits > 0; }
  };
  Quantization quantization;

  // ---- Qwen3.5 / hybrid-attention fields -----------------------------
  // Non-empty when the model interleaves classic full attention with a
  // linear-attention (Gated DeltaNet, Mamba-2 style) variant. Index L
  // is true iff layer L runs the linear-attention branch. For pure
  // dense models (Llama et al.) this stays empty and the executor takes
  // the existing single-path code.
  std::vector<bool>  is_linear_layer;

  // Linear-attention sizing (Qwen3.5 GatedDeltaNet). Meaningful only
  // when is_linear_layer has at least one true entry.
  //   key_dim  = linear_num_k_heads * linear_k_head_dim
  //   value_dim= linear_num_v_heads * linear_v_head_dim
  //   conv_dim = 2 * key_dim + value_dim   (input to the depthwise
  //                                         conv1d before the recurrence)
  int                linear_num_k_heads  = 0;
  int                linear_num_v_heads  = 0;
  int                linear_k_head_dim   = 0;
  int                linear_v_head_dim   = 0;
  int                linear_conv_kernel  = 0;

  // Spacing between full-attention layers (Qwen3.5: 4 -> every 4th
  // layer is full attention; pattern is [linear,linear,linear,full]).
  // 0 for non-hybrid models.
  int                full_attention_interval = 0;

  // Partial-rotary fraction for the full-attention layers. 1.0 means
  // every head_dim gets rotated (Llama default); Qwen3.5 sets 0.25 so
  // only the first head_dim/4 dims are rotated. Stored as fraction so
  // the executor can compute the rotary slice as
  //   rotary_dim = int(head_dim * partial_rotary_factor).
  float              partial_rotary_factor = 1.0f;

  // Multimodal RoPE section splits (Qwen3.5: [11, 11, 10] for the
  // temporal / height / width axes). Empty for non-mrope models.
  // The text-only first cut treats all positions as temporal (axis 0)
  // and ignores the interleave; saved here so a later VLM pass can
  // wire vision-token positions in without re-reading config.json.
  std::vector<int>   mrope_section;

  // Qwen3.5 full-attention layers split q_proj's output into [Q | gate];
  // attention output is multiplied by sigmoid(gate) before o_proj.
  // false for vanilla Llama/Qwen2.5.
  bool               attn_output_gate = false;

  // ---- Mixture-of-Experts fields (Qwen3.5-MoE) ---------------------
  // Populated for Qwen3_5MoeForConditionalGeneration. num_experts > 0
  // means EVERY decoder layer's MLP is a SparseMoeBlock: a softmax router
  // (mlp.gate, w8) selects top-k of num_experts experts (mlp.switch_mlp,
  // batched 3D w4 tensors), their outputs are score-weighted and summed,
  // plus an always-on shared expert (mlp.shared_expert, dense w4) gated by
  // sigmoid(mlp.shared_expert_gate, w8). 0 for dense models. The hybrid
  // GDN+full-attn backbone is unchanged; MoE only replaces the MLP.
  int                num_experts             = 0;   // total experts (256)
  int                num_experts_per_tok     = 0;   // top-k (8)
  int                moe_intermediate_size   = 0;   // per-expert inner (512)
  int                shared_expert_inter     = 0;   // shared-expert inner (512)
  bool               norm_topk_prob          = true;// renorm top-k weights

  // ---- Vision tower config (Qwen3-VL family) -----------------------
  // Populated from the nested `vision_config` object in HF config.json
  // when the architecture is Qwen3_5ForConditionalGeneration (or any
  // future VLM family). `present == true` indicates the field has been
  // read; consumers should branch on this to skip vision-tower code
  // paths for text-only models.
  struct VisionConfig {
    bool    present                  = false;
    int     depth                    = 0;     // num transformer blocks
    int     hidden_size              = 0;     // ViT hidden dim
    int     intermediate_size        = 0;     // ViT MLP inner dim
    int     num_heads                = 0;     // per-block attn heads
    int     in_channels              = 3;
    int     patch_size               = 0;     // spatial patch (e.g. 16)
    int     spatial_merge_size       = 0;     // 2 -> 2x2 patch merger
    int     temporal_patch_size      = 0;     // 1 for image; 2 for VL
    int     out_hidden_size          = 0;     // == LM hidden_size
    int     num_position_embeddings  = 0;     // sqrt(this) = grid
    // DeepStack: vision-tower layer indices whose intermediate hidden
    // states get re-merged via a postshuffle-norm patch merger and
    // injected back into the LLM at corresponding layers. For Qwen3-
    // VL default this is [8, 16, 24]. v1 (this implementation) does
    // NOT wire the LLM-side injection -- the encoder still produces
    // the main merger output, just without the DeepStack quality
    // boost. Field is parsed so a follow-up implementation can find
    // the right layers.
    std::vector<int> deepstack_visual_indexes;
    // Image preprocessing constants. Matches Qwen2/3-VL defaults
    // unless config.json overrides them via the processor section
    // (not currently parsed here -- the loader uses defaults).
    float   image_mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    float   image_std[3]  = {0.26862954f, 0.26130258f, 0.27577711f};

    // ---- Gemma-4 vision tower extras (gemma4_vision) ----------------
    // Gemma-4's ViT differs from Qwen3-VL: it average-pools patches by
    // their 2-D position down to a fixed soft-token budget, uses a
    // learned 2-axis position-embedding table, and 2-D RoPE. These are
    // 0 / unset for the Qwen path. Selected by architecture ==
    // "Gemma4ForConditionalGeneration". Gemma image preproc is
    // rescale-only (image_mean overridden to 0, image_std to 1).
    int     vit_head_dim            = 0;   // per-head dim (Gemma 64)
    int     pooling_kernel_size     = 0;   // avg-pool kernel (Gemma 3)
    int     default_output_length   = 0;   // soft-token budget (Gemma 280)
    // Per-frame soft-token budget for VIDEO (Gemma processor default 70,
    // vs 280 for stills). 0/unset for Qwen (smart-resize fixed by px).
    int     video_default_output_length = 0;
    int     position_embedding_size = 0;   // learned pos-table rows (10240)
    float   vit_rope_theta          = 0.0f;// 2-D RoPE base (Gemma 100)

    // ---- Gemma-4-12B "unified" (gemma4_unified, GGUF) ---------------
    // Encoder-LESS shallow patch embedder (no ViT). Weights live in a
    // SEPARATE mmproj GGUF (projector_type gemma4uv). When set, the loader
    // builds a Gemma4UnifiedEmbedder instead of the e4b MetalGemma4Vision
    // tower. `mmproj_path` is the sibling mmproj-*.gguf.
    bool        unified = false;
    std::string mmproj_path;
  };
  VisionConfig vision;

  // ---- Audio encoder config (Qwen3-ASR family) ---------------------
  // Populated from the nested `audio_config` object in HF config.json
  // (which itself is nested under `thinker_config` for Qwen3-ASR).
  // `present == true` indicates the field has been read; consumers
  // should branch on this for audio-aware code paths. All defaults
  // match the Qwen3-ASR-0.6B WhisperFeatureExtractor + audio tower.
  struct AudioConfig {
    bool    present                 = false;
    int     d_model                 = 0;     // encoder hidden dim
    int     encoder_layers          = 0;     // num transformer blocks
    int     encoder_attention_heads = 0;     // per-block attn heads
    int     encoder_ffn_dim         = 0;     // encoder MLP inner dim
    int     downsample_hidden_size  = 0;     // conv2d stem out channels
    int     num_mel_bins            = 128;
    int     max_source_positions    = 1500;  // sinusoidal pos length
    int     output_dim              = 0;     // == LM hidden_size
    int     n_window                = 50;    // raw chunk = n_window*2
    int     n_window_infer          = 800;   // window for block-attn
    int     conv_chunksize          = 500;
    bool    scale_embedding         = false;
    // WhisperFeatureExtractor params (preprocessor_config.json).
    int     n_fft                   = 400;
    int     hop_length              = 160;
    int     chunk_length_s          = 30;    // 30-second chunk
    int     sample_rate             = 16000;

    // ---- Gemma-4 USM Conformer extras (gemma4_audio) ---------------
    // Populated when the architecture is Gemma-4. The Conformer differs
    // from the Qwen3-ASR Whisper-style encoder: SSCP conv subsample,
    // chunked-local attention with relative-position embeddings + logit
    // softcapping, light depthwise Conv1d, macaron FFNs. 0/unset for
    // Qwen.
    int     conv_kernel_size        = 5;
    int     attention_chunk_size    = 0;
    int     attention_context_left  = 0;
    int     attention_context_right = 0;
    float   attention_logit_cap     = 0.0f;
    float   gradient_clipping       = 0.0f;
    float   residual_weight         = 0.5f;
    float   audio_rms_eps           = 1.0e-6f;
    std::vector<int> subsampling_conv_channels;   // e.g. {128, 32}

    // ---- Gemma-4-12B "unified" (gemma4_unified, GGUF) ---------------
    // Encoder-LESS: a single linear projection (projector_type gemma4ua) in
    // the sibling mmproj GGUF. FFT-free (raw 640-sample frames). Shares the
    // Gemma4UnifiedEmbedder with vision.
    bool        unified = false;
    std::string mmproj_path;
  };
  AudioConfig audio;

  // Audio-token slots used by audio-LM chat templates. Only meaningful
  // when AudioConfig::present is true. -1 means the field wasn't set.
  std::int32_t audio_start_token_id = -1;
  std::int32_t audio_end_token_id   = -1;
  std::int32_t audio_pad_token_id   = -1;

  // Languages the model claims to support (Qwen3-ASR `support_languages`).
  // Empty for non-multilingual models. Used by chat-template builders
  // that emit a "language X<asr_text>" prefill when the caller passes
  // a fixed language.
  std::vector<std::string> support_languages;

  // ---- Gemma-4 text family (Gemma4ForConditionalGeneration) --------
  // Populated from `text_config` when the architecture is the Gemma-4
  // family. `present == true` selects Gemma4ModelExec. Gemma-4 differs
  // enough from the dense / hybrid execs that its sizing lives in its
  // own struct rather than overloading the Qwen3.5 hybrid fields:
  //   * per-layer-type head_dim (sliding 256 vs full `global_head_dim`),
  //   * cross-layer KV sharing (the last `num_kv_shared_layers` reuse an
  //     earlier same-type layer's K/V),
  //   * per-layer-type RoPE (sliding "default" vs full "proportional"),
  //   * per-layer-input (PLE) embeddings + gate,
  //   * sliding-window attention + final logit softcapping.
  struct Gemma4 {
    bool   present                    = false;
    int    head_dim_sliding           = 0;     // text_config.head_dim
    int    head_dim_full              = 0;     // global_head_dim
    int    num_kv_shared_layers       = 0;     // last N layers share KV
    int    hidden_per_layer_input     = 0;     // hidden_size_per_layer_input
    int    vocab_per_layer_input      = 0;     // vocab_size_per_layer_input
    int    sliding_window             = 0;     // local-attention window
    float  final_logit_softcapping    = 0.0f;  // tanh(z/c)*c; 0 => off
    // rope_parameters.{sliding,full}_attention.
    float  rope_theta_sliding         = 1.0e4f;
    float  rope_theta_full            = 1.0e6f;
    float  full_partial_rotary_factor = 0.25f;
    // Per-layer attention kind: true => full_attention, else sliding.
    // Size == n_layers once parsed.
    std::vector<bool> is_full_layer;
    // gemma4_unified (12B): full-attention layers share K as V (no v_proj)
    // and use `num_global_kv_heads` K/V heads, while sliding layers keep the
    // generic `n_kv_heads`. e4b leaves these at the defaults (k_eq_v false,
    // num_global_kv_heads 0) so every layer uses n_kv_heads with a v_proj.
    bool   attention_k_eq_v           = false;  // full layers: values = keys
    int    num_global_kv_heads        = 0;      // full-layer K/V heads (0=off)
    // Per-layer K/V head count (size == n_layers once finalized). Sliding =
    // n_kv_heads; full = num_global_kv_heads when k_eq_v, else n_kv_heads.
    std::vector<int>  layer_n_kv_heads;
  };
  Gemma4 gemma4;
};

// Result of a successful ModelLoader::load(). Tensors are keyed by
// the Hugging Face name they ship with (e.g.
// "model.layers.0.self_attn.q_proj.weight") so downstream ModelExec
// code can look them up without renaming.
struct LoadedWeights {
  ModelConfig                                       config;
};

// Stateless loader that materialises a Hugging Face style model
// directory into a LoadedWeights record. Three layout variants are
// recognised:
//
//   1. Single-shard: `<hf_dir>/model.safetensors`.
//   2. Indexed multi-shard:
//      `<hf_dir>/model.safetensors.index.json` with a `weight_map`
//      object mapping each tensor name to a shard filename. Every
//      referenced shard is loaded; the resulting tensor maps are
//      merged. A tensor name that appears in two shards is an
//      error.
//   3. Failure: neither file present, or config.json missing /
//      malformed. The loader reports through session()->warn() and
//      returns nullopt.
//
// Pure I/O + parsing; no forward pass. No global / static state
// retained between calls; calling load() twice on the same dir does
// two filesystem reads (callers cache through
// GenerativeModelManager).
class ModelLoader : public SessionMember {
public:
  explicit ModelLoader(const SessionContextIntf* session);

  ModelLoader(const ModelLoader&)            = delete;
  ModelLoader& operator=(const ModelLoader&) = delete;

  // Load config + weights from `hf_dir`. Returns nullopt on failure
  // (failure is already reported through session()->warn()).
  std::optional<LoadedWeights>
  load(std::string_view hf_dir) const;

  // Parse only `<hf_dir>/config.json` (plus preprocessor mean/std for
  // VLMs) into a ModelConfig -- no safetensors materialised. Cheap way
  // to learn a checkpoint's shape (layers, hidden, tie/untie, vision
  // dims) without paying the multi-GB tensor load; lets size-specific
  // setup (e.g. the metal exec Config) be derived from any model in a
  // family. Returns nullopt if config.json is missing or malformed.
  std::optional<ModelConfig>
  load_config(std::string_view hf_dir) const;
};

}

#endif

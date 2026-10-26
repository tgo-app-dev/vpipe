#ifndef VPIPE_GENERATIVE_MODELS_DEEPSEEK_V41_DEEPSEEK_V41_CONFIG_H
#define VPIPE_GENERATIVE_MODELS_DEEPSEEK_V41_DEEPSEEK_V41_CONFIG_H

// DeepSeek-V4.1-Flash's config.json (model_type deepseek_v41), read into
// the release's own ModelArgs vocabulary (inference/model.py), with what
// each layer's attention does worked out once.
//
// THE ATTENTION LAYOUT (CSA2). Every layer attends a sliding WINDOW of its
// own raw latents; a layer with compress_ratios[L] = r > 0 also attends
// COMPRESSED latents, r consecutive tokens pooled into one. Those are made
// by a KV SOURCE layer and read by the layers after it until the next
// source; which compressed positions a query attends is picked by an
// INDEX SOURCE (the indexer's top-k) and reused by the layers after it
// until the next. The indexer's keys are made by the kv source too. A
// CANDIDATE SOURCE restricts every later index source to the blocks it
// kept (a two-level top-k). Released: layers 0-1 window only, 2-19
// ratio 2 (sources 2, 8, 14), 20-39 ratio 1 (the decoder: its compressed
// latents are layer 20's, the encoder's output), index sources every
// sixth layer then every fourth.

#include <cstdint>
#include <string>
#include <vector>

namespace vpipe::genai {

struct DeepseekV41Config {
  int   vocab = 129280;
  int   dim = 5120;
  int   n_layers = 40;
  int   n_mtp = 0;                 // DSpark stages after the backbone
  int   n_heads = 64;
  int   head_dim = 512;
  int   rope_dim = 64;             // the rotated tail of each head
  int   q_lora = 1280;
  int   o_lora = 1024;
  int   o_groups = 8;
  int   window = 128;
  float eps = 1e-20f;
  // MoE
  int   moe_inter = 2304;
  int   n_experts = 384;
  int   top_k = 6;
  int   n_shared = 1;
  float route_scale = 1.5f;
  float swiglu_limit = 10.0f;
  bool  norm_topk = true;
  std::string score_func = "sqrtsoftplus";
  // Rotary: window-only layers at rope_theta without YaRN, compressed
  // layers at compress_rope_theta with it.
  float rope_theta = 10000.0f;
  float compress_rope_theta = 160000.0f;
  float rope_factor = 16.0f;
  int   beta_fast = 32;
  int   beta_slow = 1;
  int   orig_seq = 65536;           // 0: no YaRN
  // CSA2
  std::vector<int> compress_ratios;  // n_layers + n_mtp entries
  std::vector<int> kv_sources, index_sources;
  int   candidate_source = -1;
  int   candidate_blocks = 0;
  int   candidate_block = 0;
  int   index_heads = 32;
  int   index_dim = 128;
  int   index_topk = 512;
  // hyper-connections
  int   hc = 4;
  int   hc_iters = 20;
  float hc_eps = 1e-6f;
  // engram
  std::vector<int> engram_layers;
  std::vector<std::int64_t> engram_rows;
  int   engram_ngram = 4;
  int   engram_vocab = 16000000;
  int   engram_heads = 8;
  int   engram_dim = 256;
  int   engram_pad = 2;
  int   engram_cvocab = 0;          // the compressed token vocabulary
  // DSpark
  int   dspark_block = 0;
  int   dspark_noise = 0;
  std::vector<int> dspark_targets;
  int   dspark_markov = 256;
  int   dspark_experts = 0;
  int   dspark_top_k = 0;
  // vision
  int   vis_layers = 0;
  int   vis_dim = 1024;
  int   vis_heads = 16;
  int   vis_inter = 2816;
  int   vis_patch = 14;
  float vis_rope_theta = 10000.0f;
  int   vis_downsample = 3;
  int   vis_max_tokens = 1024;
  int   vis_min_pixels = 295936;
  int   vis_max_wh_ratio = 0;      // 0: none
  int   image_token = 129264;
  int   bos = 0;
  int   eos = 1;
  bool  fp8 = true;                // quantization_config: E4M3 projections
  bool  fp4_experts = true;        // ... and E2M1 routed experts
  // An oMLX pack that requantizes modules to MLX's affine codes (oQ2e /
  // oQ2.5e / oQ3e); which modules, and their widths, the tensors say.
  bool  affine = false;

  // What layer L's attention does (L < n_layers).
  struct Plan {
    int  ratio = 0;                // 0: the window only
    bool kv_source = false;        // makes the compressed latents
    bool index_source = false;     // runs an indexer
    bool owns_index_keys = false;  // makes the indexer's keys
    bool candidate_source = false;
    bool uses_candidates = false;
    int  kv_from = -1;             // the kv source whose latents it reads
    int  index_from = -1;          // the index source whose picks it uses
    int  keys_from = -1;           // the layer whose indexer keys it reads
  };
  std::vector<Plan> plan;          // [n_layers]

  int  engram_cols() const { return (engram_ngram - 1) * engram_heads; }
  int  engram_slot(int L) const;   // index in engram_layers, or -1
  // The causal encoder-decoder's first decoder layer (the README's CED:
  // the decoder's global KV projected from the encoder's last hidden
  // states): the one ratio-1 KV source whose latents and index keys every
  // layer after it reads, no engram layer at or past it; -1 when the plan
  // is not that shape.
  int  ced_mid() const;
  int  mix_hc() const { return (2 + hc) * hc; }

  // False with `err` set when the file is missing, not this model, or
  // outside what the port runs.
  static bool from_dir(const std::string& dir, DeepseekV41Config* out,
                       std::string* err);
  // From config.json's text (tests).
  static bool from_json_text(const std::string& text, DeepseekV41Config* out,
                             std::string* err);
};

}  // namespace vpipe::genai

#endif  // VPIPE_GENERATIVE_MODELS_DEEPSEEK_V41_DEEPSEEK_V41_CONFIG_H

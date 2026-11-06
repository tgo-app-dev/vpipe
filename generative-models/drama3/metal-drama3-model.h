#ifndef VPIPE_GENERATIVE_MODELS_DRAMA3_METAL_DRAMA3_MODEL_H
#define VPIPE_GENERATIVE_MODELS_DRAMA3_METAL_DRAMA3_MODEL_H

// MetalDrama3Model -- Fish Audio Drama 3's speech LM (fishaudio/drama-3,
// config.json model_type "fish_omni"): text and voice references in, one
// ten-codebook codec frame per step out.
//
//   backbone   a Qwen3.5-35B-A3B text stack (40 layers, 3 gated-DeltaNet
//              to 1 full attention, 256 routed experts top-8 + a gated
//              shared expert) on MetalQwenModel, its routed experts held
//              as the checkpoint's block-FP8 codes (_fp8_experts), the
//              rest bf16. It reads one row per frame: a text token's
//              embedding, or for an audio frame (E[semantic_begin + c0] +
//              E_cb[c1]) / sqrt(2).
//   main head  the backbone's lm_head SLICED to the only ids the
//              reference ever draws: end of speech and semantic codes
//              1..4095 (16 MB of a 1 GB head). The draw is the
//              reference's top-k / top-p with repetition-aware sampling.
//   tower      MetalDrama3Tower: 13 layers, codes 1..9.
//
// A generated frame is ONE command buffer: the backbone's decode step on
// the embedding the previous frame left in its input, the main head and
// its draw, the tower with its nine draws, and the next frame's embedding
// -- all on the GPU; the host reads the frame back.
//
// The prompt (reference-bracket-v1) is built by drama3-prompt.h.

#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/drama3/metal-drama3-tower.h"
#include "generative-models/qwen3/metal-qwen-model.h"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace vpipe::metal_compute {
class MetalCompute;
}

namespace vpipe::genai {

class WeightSet;
class MetalLlamaWeights;

struct Drama3Config {
  MetalQwenModel::Config   backbone;
  MetalDrama3Tower::Config tower;
  int eos            = 248046;    // <|im_end|> ends the speech
  int semantic_begin = 248099;    // <|semantic:0|>
  int semantic_end   = 252194;    // <|semantic:4095|>
  int text_vocab     = 252288;
  // How the routed experts are stored: "fp8" (block FP8, fishaudio/
  // drama-3) or "int4" (GPTQ, fishaudio/drama-3-int4).
  std::string experts = "fp8";
  int frame_width() const { return 1 + tower.num_codebooks; }
  // The main head's candidates: end of speech, then codes 1..end.
  int main_candidates() const { return 1 + semantic_end - semantic_begin; }

  // config.json of a fishaudio/drama-3 directory. Refuses, by name, every
  // variant this port does not run (a different tower, rotary layout or
  // prompt template).
  static bool from_dir(const std::string& dir, Drama3Config* out,
                       std::string* err = nullptr);
  static bool is_drama3_dir(const std::string& dir);
};

class MetalDrama3Model {
public:
  static constexpr int kFrame = 11;     // [main, c0 .. c9]

  struct LoadOptions {
    // Rows a sequence may hold: prompt + generated frames.
    int max_rows = 8192 + 2048;
    // The backbone's routed experts in a cache of this many bytes, read
    // from disk on a miss and kept by how often they are selected
    // (MetalQwenModel::Config::expert_cache_bytes); 0 = all resident.
    std::size_t expert_cache_bytes = 0;
    // Whether the backbone's kernel tuners measure (the accel key
    // `autotune`; MetalQwenModel::Config::autotune).
    bool autotune = true;
  };

  static std::unique_ptr<MetalDrama3Model> load(
      std::shared_ptr<WeightSet> ws, metal_compute::MetalCompute* mc,
      const Drama3Config& cfg, const LoadOptions& opt,
      std::string* err = nullptr);
  ~MetalDrama3Model();

  const Drama3Config& config() const { return _cfg; }
  // The longest sequence (prompt + frames) it was loaded for.
  int max_rows() const { return _opt.max_rows; }
  MetalQwenModel* backbone() { return _lm.get(); }
  MetalDrama3Tower* tower() { return _tower.get(); }

  // Teacher-forced forward over `n` frames ([n, 11] ints, row-major): each
  // row's final-normed backbone hidden ([n, hidden]), main-head logits
  // ([n, main_candidates]) and codebook-head logits ([9, n, 4096]) given
  // the NEXT row (the last row's next is end of speech). Any output may be
  // null. The test seam against the reference's own forward.
  bool teacher_forced(const std::vector<std::int32_t>& frames, int n,
                      std::vector<float>* hidden,
                      std::vector<float>* main_logits,
                      std::vector<float>* head_logits,
                      std::string* err = nullptr);

  struct Sampling {
    // The reference's request defaults (sglang-omni speech_request.py).
    float temperature = 1.0f;
    float top_p       = 0.9f;
    int   top_k       = 30;
    std::uint32_t seed = 0;
    int   max_frames  = 2048;
  };
  using Frame = std::array<std::int32_t, 10>;    // c0 .. c9

  // Generate after `prompt` ([n_prompt, 11]) until end of speech or
  // max_frames. on_frame(frame, index) sees each frame as it is drawn and
  // may stop the run (return false). `frames` (optional) gets them all.
  // `main_logits` (tests): each frame's main-head logits as drawn from,
  // [frames + 1, main_candidates] (the end-of-speech frame's included).
  bool generate(const std::vector<std::int32_t>& prompt, int n_prompt,
                const Sampling& sp,
                const std::function<bool(const Frame&, int)>& on_frame,
                std::vector<Frame>* frames, std::string* err = nullptr,
                std::vector<float>* main_logits = nullptr);

  // The DECODE path, teacher-forced (tests): the first `n_prompt` rows of
  // `frames` prefill as a prompt does, then every later row is fed one at
  // a time through the backbone's decode step and a one-row tower, with
  // the next row as its label. Logits of rows n_prompt-1 .. n-1 in the
  // teacher_forced layouts ([rows, main_candidates], [9, rows, 4096]).
  bool teacher_decode(const std::vector<std::int32_t>& frames, int n,
                      int n_prompt, std::vector<float>* main_logits,
                      std::vector<float>* head_logits,
                      std::string* err = nullptr,
                      std::vector<float>* hidden = nullptr);

  // The last generate()'s timing: the prompt's prefill (backbone + tower),
  // the first frame after it, and the frames after that.
  struct RunStats {
    int    prompt_rows = 0;
    int    frames      = 0;
    bool   ended       = false;    // end of speech (not the budget, not a
                                   // caller's stop)
    double prefill_ms  = 0.0;      // embed + backbone + tower:
    double embed_ms    = 0.0;
    double backbone_ms = 0.0;
    double first_ms    = 0.0;
    double frames_per_s = 0.0;     // after the first
  };
  const RunStats& last_run() const { return _stats; }

  // Weights held now: the backbone, the tower, the tables and heads here.
  std::size_t resident_bytes() const;
  // What a load of `dir` will hold, from the checkpoint alone (for a plan
  // made before anything loads): every tensor but the two vocabulary-wide
  // tables -- the embedding is mapped, the head sliced -- plus the slice
  // and the embedding's semantic band. 0 when the checkpoint does not open.
  static std::size_t weight_bytes(const std::string& dir);
  // The routed experts' share of weight_bytes(), and one layer's: what an
  // expert cache stands in for (model_memory::moe_expert_cache).
  static std::size_t routed_expert_bytes(const std::string& dir,
                                         std::size_t* per_layer = nullptr);
  // The working set of a sequence of `rows` rows (prompt + frames): the
  // backbone's K/V and recurrent state, the tower's K/V, and the prefill's
  // transients for a `prompt_rows`-row prompt.
  static std::size_t run_bytes(const Drama3Config& c, int rows,
                               int prompt_rows);

private:
  MetalDrama3Model() = default;

  bool init_(std::string* err);
  // Backbone input rows [n, hidden] for frames [n, 11] (host gather of the
  // text rows from the mapped table, combine on the GPU).
  metal_compute::SharedBuffer embed_(const std::int32_t* frames, int n);

  metal_compute::MetalCompute* _mc = nullptr;
  Drama3Config _cfg;
  LoadOptions _opt;
  std::shared_ptr<WeightSet> _ws;
  std::unique_ptr<MetalQwenModel> _lm;
  std::unique_ptr<MetalDrama3Tower> _tower;

  // The embedding table stays ON DISK (1 GB; a prompt reads a few hundred
  // of its rows, by pread at the tensor's place in its shard); its
  // semantic band is resident for the GPU's per-frame gather.
  struct RowFile {
    int fd = -1;
    std::uint64_t base = 0;      // the tensor's first byte in the file
    std::size_t row = 0;         // bytes a row
    std::int64_t rows = 0;
    ~RowFile();
    bool read(std::int64_t r0, std::int64_t n, void* dst) const;
  };
  static bool open_rows_(const std::string& dir, const std::string& name,
                         const MetalLlamaWeights& src, RowFile* out);
  RowFile _embed_rows;
  metal_compute::SharedBuffer _embed_sem;    // [4096, hidden]
  metal_compute::SharedBuffer _head;         // [main_candidates, hidden]

  metal_compute::ComputeLibrary _lib_d3;
  metal_compute::ComputeFunction _fn_embed, _fn_sample_main;
  // Per-frame GPU state.
  metal_compute::SharedBuffer _frame[2], _ras, _mlogits, _hrow, _hrow_in;
  RunStats _stats;
};

}  // namespace vpipe::genai

#endif  // VPIPE_GENERATIVE_MODELS_DRAMA3_METAL_DRAMA3_MODEL_H

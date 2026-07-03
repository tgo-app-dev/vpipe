#ifndef VPIPE_GENAI_MOSS_METAL_MOSS_V15_MODEL_H
#define VPIPE_GENAI_MOSS_METAL_MOSS_V15_MODEL_H

// MOSS-TTS-Local-Transformer-v1.5 full LM: the 8-bit Qwen3 backbone +
// MetalMossLocalModel (depth decoder + audio heads) + the per-frame generate
// loop. Per frame: backbone produces the last hidden (prefill, then one decode
// step per subsequent frame, KV-cached); that seeds the local decoder which
// emits n_vq codes; local_text_lm_head(pos-0 hidden) decides continue/stop;
// the next grid row [audio_assistant_slot, codes...] is re-embedded
// (text_embed + sum audio_embeddings) and fed to the backbone. Loads entirely
// from a quantized model dir (backbone 8-bit; embeds/heads/local bf16).

#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/moss/metal-moss-local-model.h"
#include "generative-models/qwen3/metal-qwen-model.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe::metal_compute {
class MetalCompute;
}
namespace vpipe { class SessionContextIntf; }

namespace vpipe::genai {

class MetalLlamaWeights;

class MetalMossV15Model {
public:
  struct Config {
    MetalQwenModel::Config     backbone;   // v1.5 8-bit dense Qwen3
    MetalMossLocalModel::Config local;     // depth decoder + heads
    int audio_pad_code      = 1024;
    int audio_assistant_slot = 151656;
    int audio_end_token      = 151670;
  };

  static std::unique_ptr<MetalMossV15Model> load(
      const std::string& quant_dir, metal_compute::MetalCompute* mc,
      const Config& cfg);

  // Generation. `grid` is [seq][1+n_vq] int32 (channel 0 text id, 1..n_vq
  // audio codes / audio_pad). Returns up to max_frames frames, each n_vq
  // codes; stops early when local_text_lm_head selects audio_end. When
  // `audio_sp.temperature > 0` the per-frame RVQ codes are SAMPLED (audio
  // MUST be sampled -- greedy degenerates into silence after the first
  // sentence); `seed` seeds the deterministic audio sampler (0 => fresh).
  // The continue/stop decision stays greedy. Defaults => greedy (callers and
  // golden/teacher-forced tests stay token-exact).
  std::vector<std::vector<int>> generate(
      const std::vector<std::vector<std::int32_t>>& grid, int max_frames,
      const MossSampling& audio_sp = {}, std::uint64_t seed = 0);

  MetalQwenModel*      backbone() { return _bb.get(); }
  MetalMossLocalModel* local() { return _lm.get(); }

  // Assemble n grid rows -> bf16 [n*hidden] (text embed + sum audio embeds).
  // Public for tests (verify against the HF inputs_embeds golden).
  metal_compute::SharedBuffer assemble_embeds(
      const std::vector<std::vector<std::int32_t>>& grid, int start, int n)
  { return assemble_embeds_(grid, start, n); }

private:
  bool init_(const MetalLlamaWeights& wts, metal_compute::MetalCompute* mc,
             const Config& cfg, const std::string& quant_dir);
  // Assemble n grid rows -> bf16 [n*hidden] (text embed + sum audio embeds).
  metal_compute::SharedBuffer assemble_embeds_(
      const std::vector<std::vector<std::int32_t>>& grid, int start, int n);
  int text_decision_(const metal_compute::SharedBuffer& h0);

  metal_compute::MetalCompute* _mc = nullptr;
  const SessionContextIntf* _session = nullptr;   // profiling sink (null=off)
  Config _cfg;
  std::unique_ptr<MetalQwenModel>      _bb;
  std::unique_ptr<MetalMossLocalModel> _lm;

  metal_compute::SharedBuffer _text_embed;               // bf16 [vocab,hidden]
  std::vector<metal_compute::SharedBuffer> _audio_embed; // bf16 n_vq x [V,h]
  metal_compute::SharedBuffer _ltext_head;               // f16 [2,hidden]
  metal_compute::SharedBuffer _seed_bb;                  // bf16 [hidden]
  metal_compute::SharedBuffer _seed_f16;                 // f16 [hidden]
  metal_compute::SharedBuffer _ltext_logits;             // f16 [2]

  metal_compute::ComputeLibrary  _lib_dense;
  metal_compute::ComputeFunction _fn_gemv;
};

}  // namespace vpipe::genai

#endif

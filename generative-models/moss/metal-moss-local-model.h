#ifndef VPIPE_GENAI_MOSS_METAL_MOSS_LOCAL_MODEL_H
#define VPIPE_GENAI_MOSS_METAL_MOSS_LOCAL_MODEL_H

// MOSS-TTS-Local-Transformer-v1.5 generation head: the depth/local transformer
// (MetalMossLocalTransformer) plus the n_vq audio code embeddings + n_vq audio
// LM heads (bf16 -> f16). decode_frame_greedy() runs the per-frame codebook
// loop: backbone seed -> local step (pos 0) -> head_0 argmax -> code_0, then
// for k=1..n_vq-1 feed audio_embeddings[k-1](code) -> local step (pos k) ->
// head_k argmax -> code_k. Returns the n_vq greedy codes for the frame.
//
// (The Qwen3 backbone + the full text->grid generate loop are wired by the
// enclosing v1.5 model; this class owns the depth-decode half + heads.)

#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/moss/metal-moss-local-transformer.h"
#include "generative-models/moss/metal-moss-tts-model.h"   // MossSampling

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace vpipe::metal_compute {
class MetalCompute;
class ComputeEncoder;
}

namespace vpipe::genai {

class MetalLlamaWeights;

class MetalMossLocalModel {
public:
  struct Config {
    MetalMossLocalTransformer::Config lt;   // hidden/n_head/head_dim/inner...
    int n_vq        = 12;
    int audio_vocab = 1024;                 // codebook size (heads N + embed rows)
  };

  static std::unique_ptr<MetalMossLocalModel> load(
      const std::string& model_dir, metal_compute::MetalCompute* mc,
      const Config& cfg);

  // Greedy per-frame codebook decode from the backbone seed [hidden] f16.
  std::vector<int> decode_frame_greedy(const metal_compute::SharedBuffer& seed);

  // Reference (pre-fusion) serial greedy decode: one command buffer + host
  // argmax per codebook. Kept for the fused-vs-serial token-exact equivalence
  // test; production uses decode_frame_greedy (the fused single-buffer path).
  std::vector<int> decode_frame_greedy_ref(
      const metal_compute::SharedBuffer& seed);

  // Sampled per-frame codebook decode: identical structure to
  // decode_frame_greedy, but each codebook's head logits are SAMPLED
  // (temperature / top_k / top_p; temperature <= 0 => argmax) instead of
  // argmax. `rng_state` carries the deterministic PRNG state across frames
  // (advanced in place). MOSS audio must be sampled -- greedy degenerates
  // into silence after the first sentence.
  std::vector<int> decode_frame_sampled(
      const metal_compute::SharedBuffer& seed, const MossSampling& sp,
      std::uint64_t& rng_state);

  // Teacher-forced variant: feed tf_codes[k-1] as the codebook-(k-1) feedback
  // (instead of the model's own argmax), so every step sees the SAME input as
  // a reference run. Fills hiddens_out [n_vq*hidden] f32 (per-step ln_f hidden)
  // and argmax_out [n_vq] (the model's greedy pick at each step). Isolates the
  // per-step compute from greedy flip-propagation.
  void decode_frame_teacher(const metal_compute::SharedBuffer& seed,
                            const std::vector<int>& tf_codes,
                            std::vector<float>& hiddens_out,
                            std::vector<int>& argmax_out);

  MetalMossLocalTransformer* local() { return _lt.get(); }

  // The pos-0 local hidden from the most recent decode_frame_* call (the
  // hidden the local_text_lm_head reads to decide continue/stop). [hidden] f16.
  const metal_compute::SharedBuffer& last_h0() const { return _h0; }

private:
  bool init_(const MetalLlamaWeights& wts, metal_compute::MetalCompute* mc,
             const Config& cfg);
  int  argmax_head_(int k, const metal_compute::SharedBuffer& h);
  // Run codebook-k head GEMV, pull logits to host, and SAMPLE one code.
  int  sample_head_(int k, const metal_compute::SharedBuffer& h,
                    const MossSampling& sp, std::mt19937_64& rng);
  void gather_embed_(int k, int code);

  // Fused whole-frame decode: encode all n_vq (local step -> head GEMV ->
  // on-GPU argmax/sample -> embed-gather) dispatches into ONE command buffer
  // so the codebook loop never round-trips to the host between codebooks
  // (the old per-codebook argmax_head_/step commit+wait was the decode
  // bottleneck). `sp==nullptr` => greedy argmax (token-exact); else the
  // per-codebook logits are sampled on-GPU (lc_sample_f16, seed varied per
  // codebook from `seed64`). Returns the n_vq codes.
  std::vector<int> decode_frame_fused_(const metal_compute::SharedBuffer& seed,
                                       const MossSampling* sp,
                                       std::uint64_t seed64);
  // Encoder helpers (no commit): codebook-k head GEMV -> _logits; on-GPU
  // argmax/sample -> _codes[k]; gather _embeds[k](_codes[k]) -> _einp.
  void encode_head_(metal_compute::ComputeEncoder& e, int k,
                    const metal_compute::SharedBuffer& h);
  void encode_argmax_(metal_compute::ComputeEncoder& e, int k);
  void encode_sample_(metal_compute::ComputeEncoder& e, int k,
                      const MossSampling& sp, std::uint32_t seed);
  void encode_gather_(metal_compute::ComputeEncoder& e, int k);

  metal_compute::MetalCompute* _mc = nullptr;
  Config _cfg;
  std::unique_ptr<MetalMossLocalTransformer> _lt;

  std::vector<metal_compute::SharedBuffer> _heads;   // n_vq x [vocab, hidden]
  std::vector<metal_compute::SharedBuffer> _embeds;  // n_vq x [vocab, hidden]
  metal_compute::SharedBuffer _logits;               // f16 [audio_vocab]
  metal_compute::SharedBuffer _einp;                 // f16 [hidden]
  metal_compute::SharedBuffer _h0;                   // f16 [hidden] (pos-0)
  metal_compute::SharedBuffer _codes;                // int32 [n_vq] (on-GPU)
  metal_compute::SharedBuffer _lc_ws;                // f16 [audio_vocab] ws

  metal_compute::ComputeLibrary  _lib_dense, _lib_elt;
  metal_compute::ComputeFunction _fn_gemv, _fn_argmax, _fn_sample;
  metal_compute::ComputeFunction _fn_embed_gather, _fn_copy;
};

}  // namespace vpipe::genai

#endif

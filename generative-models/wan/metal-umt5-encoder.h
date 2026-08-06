#ifndef GENERATIVE_MODELS_WAN_METAL_UMT5_ENCODER_H
#define GENERATIVE_MODELS_WAN_METAL_UMT5_ENCODER_H

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

class WeightSet;   // generative-models/weight-set.h

// The umT5-XXL encoder (UMT5EncoderModel), which is the text tower of every
// Wan video model. Encoder-only -- the checkpoint ships no decoder half --
// and run in BF16 on the metal-compute backend.
//
// bf16 is not a preference here. T5-XXL's residual stream carries
// activations far outside f16's range (this is the long-standing
// "T5 in fp16 gives NaNs" problem), and the checkpoint is bf16 on disk, so
// running bf16 both avoids the overflow and means the weights load with no
// conversion at all -- they are cached as-is rather than derived.
//
// Structurally it is a plain pre-norm encoder, but three details are T5's
// and not shared with any of the LMs in this tree:
//
//   * the norm is RMS with NO mean subtraction and NO bias, accumulated in
//     fp32 (T5LayerNorm).
//   * attention scores are NOT scaled by 1/sqrt(d). T5 folds that into
//     initialization, so dividing here would be wrong by a constant factor
//     per layer.
//   * position information arrives ONLY as an additive relative-position
//     bias on the attention scores, bucketed over 32 buckets of distance.
//     There is no rotary embedding. And unlike T5, umT5 carries its own
//     bias table in EVERY layer rather than sharing layer 0's.
//
// The feed-forward is gated: wo(gelu_tanh(wi_0(x)) * wi_1(x)).
class MetalUmt5Encoder {
 public:
  struct Config {
    int d_model    = 4096;
    int d_ff       = 10240;
    int n_heads    = 64;
    int d_kv       = 64;
    int n_layers   = 24;
    int vocab      = 256384;
    float norm_eps = 1e-6f;
    int rel_buckets     = 32;
    int rel_max_dist    = 128;
  };

  // Read a Config out of a transformers `text_encoder/config.json`. False
  // when the file is missing or is not a umt5 encoder config.
  static bool config_from_json(const std::string& enc_dir, Config& out,
                               std::string* err = nullptr);

  // Prefer the WeightSet overload: the set is the manager's shared,
  // reference-counted view of the checkpoint. The dir overload opens a
  // PRIVATE set (tests, and callers with no session to ask).
  static std::unique_ptr<MetalUmt5Encoder>
  load(const std::string& model_dir, metal_compute::MetalCompute* mc,
       const Config& cfg);

  static std::unique_ptr<MetalUmt5Encoder>
  load(std::shared_ptr<WeightSet> ws, metal_compute::MetalCompute* mc,
       const Config& cfg);

  // Encode `ids` (already tokenized and padded to `padded_len` with the
  // pad id) into the last hidden state, bf16 [padded_len, d_model].
  //
  // `n_valid` is how many leading ids are real: keys past it are masked
  // out of the attention, and -- following the Wan pipeline -- the rows
  // past it are ZEROED rather than left as whatever the encoder produced
  // for padding. That zero tail is what the DiT cross-attends to, so it is
  // part of the contract, not a tidy-up.
  //
  // Empty on failure, with a reason in `err`.
  metal_compute::SharedBuffer
  encode(const std::vector<std::int32_t>& ids, int n_valid,
         std::string* err = nullptr);

  // Sequence length the attention kernel can hold in threadgroup memory.
  static constexpr int kMaxSeq = 1024;

  const Config& config() const { return _cfg; }

 private:
  MetalUmt5Encoder() = default;

  struct Layer {
    metal_compute::SharedBuffer n1, n2;        // [d_model] RMS gammas
    metal_compute::SharedBuffer q, k, v, o;    // [d_model, d_model]
    metal_compute::SharedBuffer rel;           // [32, n_heads]
    metal_compute::SharedBuffer wi0, wi1, wo;  // [d_ff, d_model] / [d_model, d_ff]
  };

  metal_compute::SharedBuffer weight_(WeightSet& ws, const std::string& nm);

  // y[M,N] = x[M,K] @ w[N,K]^T. No bias anywhere in T5.
  void gemm_(metal_compute::ComputeEncoder& enc,
             const metal_compute::SharedBuffer& x,
             const metal_compute::SharedBuffer& w,
             const metal_compute::SharedBuffer& y, int M, int N, int K);

  // The per-(query,key) bucket table the attention kernel reads. Built once
  // per sequence length on the host, in the float arithmetic the reference
  // uses, because the log's truncation decides the bucket at the boundaries.
  std::vector<std::uint8_t> bucket_table_(int L) const;

  metal_compute::MetalCompute* _mc = nullptr;
  Config _cfg;
  metal_compute::SharedBuffer _embed;       // [vocab, d_model]
  std::vector<Layer> _layers;
  metal_compute::SharedBuffer _final_norm;  // [d_model]
  std::shared_ptr<WeightSet> _ws;

  metal_compute::ComputeLibrary _lib_gemm, _lib_elt, _lib_rms;
  // geglu IS T5's gated feed-forward: gelu_tanh(wi_0(x)) * wi_1(x) is the
  // same expression Gemma-4's MLP gate uses, so the kernel is shared.
  metal_compute::ComputeFunction _fn_gemm, _fn_rms, _fn_geglu, _fn_residual,
      _fn_attn;
};

}  // namespace genai
}  // namespace vpipe

#endif

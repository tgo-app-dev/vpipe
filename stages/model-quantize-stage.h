#ifndef VPIPE_STAGES_MODEL_QUANTIZE_STAGE_H
#define VPIPE_STAGES_MODEL_QUANTIZE_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace vpipe {

// Source stage: 0 inputs, 0 outputs. One-shot offline model quantization.
// Reads a source HF/safetensors model, quantizes its backbone linear weights
// to the MLX-affine group-quant format (passing embeddings / heads / norms /
// any auxiliary modules through unchanged), writes the result, registers it in
// the models DB, then signals done. Pure filesystem side-effect; no Beats.
//
// Multi-architecture: the plain group-affine and per-layer mixed-precision
// paths work for any standard-named transformer (Llama / Qwen2 / Qwen3 /
// Qwen3.5 hybrid / Gemma-4 / MOSS) -- arch, layer_prefix and n_layers are
// auto-detected from the source (config.json + a probe of the safetensors
// layer layout; see genai::detect_quant_arch). AWQ additionally requires a
// standard-layernorm stack -- full attention OR Qwen3.5 gated-DeltaNet, both
// rooted at input_layernorm (blocked for Gemma-style FFN norms and MoE MLPs);
// on-device auto-calibration additionally requires a Qwen3-family backbone
// (dense or the Qwen3.5 hybrid), else supply calib_dir.
//
// Registry-aware:
//   * src_model may be a models-DB key (registered by model-fetch) OR a path.
//   * output_name names the result: a bare name (or "org/name") -> the output
//     dir is inferred as <cwd>/models/<output_name> and the result is
//     registered in the models DB under that key. An absolute/relative path
//     ("/..", "./..") is used verbatim and NOT registered.
//   * arch / n_layers / layer_prefix are auto-detected from the source
//     config.json for a known model_type (moss_tts_local); explicit config
//     overrides the detection.
//
// Config (FlexData object):
//   src_model     (string, required) -- source model: a models-DB key or a
//                  bf16/f16 safetensors directory path.
//   output_name   (string, required) -- result name (-> <cwd>/models/<name>,
//                  registered) or an explicit path ("/..", "./..").
//   models_db     (string, default "models") -- LMDB sub-db for src lookup +
//                  output registration.
//   bits          (uint, default 8)  -- backbone affine bit-width (4 | 8).
//   group_size    (uint, default 64) -- affine group size (32 | 64).
//   arch          (string, default "") -- model family tag; empty => auto-
//                  detect from the source config.json model_type.
//   skip_existing (bool, default true) -- skip if the output config.json
//                  already exists.
//   awq           (bool, default false) -- AWQ activation-aware smoothing
//                  (per-layer fp-equivalent scale search). Calibration:
//                  supply calib_dir, else it auto-calibrates on-device for a
//                  known arch.
//   awq_clip      (bool, default false) -- the paired per-group weight clip
//                  search (needs awq). OFF by default: it can regress end
//                  drift on small calibration sets.
//   calib_dir     (string) -- dir with calib_{qkv,o,gateup,down}.f32 stats;
//                  empty => on-device auto-calibration for a known arch.
//   mixed         (bool, default false) -- unsloth-style PER-LAYER mixed
//                  precision: promote the most-sensitive `mixed_frac` of
//                  layers to `high_bits`. Requires bits=4, high_bits=8,
//                  group_size=64.
//   high_bits     (uint, default 8) -- promoted bit-width for mixed.
//   mixed_frac    (real, default 0.25) -- fraction of LAYERS promoted.
//   layer_prefix  (string, default "") -- arch layer-root prefix for awq/mixed
//                  per-layer tensors; empty => auto-detect.
//   n_layers      (uint, default 0) -- layer count for awq/mixed; 0 => auto-
//                  detect from the source config.json.
//
// Drives genai::ModelQuantizer (plain group-affine, AWQ, and mixed paths).
class ModelQuantizeStage final : public TypedStage<ModelQuantizeStage> {
public:
  static constexpr const char* kTypeName = "model-quantize";

  ModelQuantizeStage(const SessionContextIntf* session,
                     std::string               id,
                     std::vector<InEdge>       iports,
                     FlexData                  config);

  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only inspectors.
  const std::string& src_model() const noexcept { return _src_model; }
  const std::string& output_name() const noexcept { return _output_name; }
  int  bits() const noexcept { return _bits; }
  int  group_size() const noexcept { return _group_size; }
  bool skip_existing() const noexcept { return _skip_existing; }
  bool awq() const noexcept { return _awq; }
  bool mixed() const noexcept { return _mixed; }
  float mixed_frac() const noexcept { return _mixed_frac; }
  const std::string& calib_dir() const noexcept { return _calib_dir; }
  int n_layers() const noexcept { return _n_layers; }

  // Test seam: run the quantization once. Returns true on success (or when
  // skipped because the output already exists). Logs + returns false on
  // error.
  bool quantize_once(const std::function<bool()>& stop = [] {
    return false;
  });

private:
  // Register the quantized output dir in the models DB under `key`.
  void register_output_(const std::string& key, const std::string& dir,
                        const std::string& arch, int bits);

  // Config attributes; defaults live in kSpec.attrs and are read in the
  // constructor via attr_*. Declarations carry no non-zero default.
  std::string _src_model;
  std::string _output_name;
  std::string _models_db;
  std::string _arch;
  int         _bits{};
  int         _group_size{};
  bool        _skip_existing{};
  // AWQ + mixed precision.
  bool        _awq{};
  bool        _awq_clip{};
  std::string _calib_dir;
  bool        _mixed{};
  int         _high_bits{};
  float       _mixed_frac{};
  std::string _layer_prefix;
  int         _n_layers{};
};

}  // namespace vpipe

#endif

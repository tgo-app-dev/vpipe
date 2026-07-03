#ifndef VPIPE_STAGES_MODEL_BENCHMARK_STAGE_H
#define VPIPE_STAGES_MODEL_BENCHMARK_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

// The sampler config (decode test) lives in the LM subsystem, which is
// MLX-free and builds on the VPIPE_BUILD_APPLE_SILICON axis.
#ifdef VPIPE_BUILD_APPLE_SILICON
#include "generative-models/sampler.h"
#endif

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace vpipe {

// Source stage: 0 inputs, 0 outputs. One-shot LM throughput benchmark.
// Loads one language model (a models-DB key or a directory path) and
// times prefill + decode at several context lengths, recording GPU
// temperature / frequency / utilization / power during each test, then
// logs a Markdown report through session()->info(). Pure read-only;
// emits no Beats.
//
// Three benchmarks, each run for every configured context length:
//   1. Full prefill   -- time a whole N-token prompt prefill.
//   2. Incremental     -- time a `prefill_chunk`-token prefill when the
//                         KV already holds N tokens (prefill speed vs
//                         context depth).
//   3. Decode          -- time decoding `decode_tokens` tokens after an
//                         N-token prefill.
//
// Config (FlexData object):
//   model         (string, required) -- a models-DB key or a dir path.
//   models_db     (string, default "models") -- LMDB sub-db for keys.
//   contexts      (string, default "1024,2048,4096") -- comma-separated
//                 context lengths to benchmark (opt into 8k/16k by
//                 listing them).
//   decode_tokens (uint, default 128) -- tokens decoded per decode test.
//   prefill_chunk (uint, default 1024) -- chunk size for the
//                 incremental-prefill-at-depth probe.
//   warmup        (uint, default 1) -- warmup iterations (discarded).
//   seed          (uint, default 1234) -- synthetic-prompt RNG seed.
//   cooldown_s    (real, default 0.0) -- seconds to sleep after each
//                 timed test (let GPU thermals settle); skipped after
//                 the last test.
//   temperature / top_k / top_p / min_p / repetition_penalty /
//   presence_penalty -- decode sampler config (greedy by default so
//                 results stay deterministic).
//   mtp           (bool, default false) -- when the model carries an MTP
//                 head, run the decode test via MTP speculative decode
//                 (real-text prompt + acceptance rate) instead of the
//                 pdecode run-ahead path.
class ModelBenchmarkStage final
    : public TypedStage<ModelBenchmarkStage> {
public:
  static constexpr const char* kTypeName = "model-benchmark";

  ModelBenchmarkStage(const SessionContextIntf* session,
                      std::string               id,
                      std::vector<InEdge>       iports,
                      FlexData                  config);

  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only inspectors.
  const std::string&      model()        const noexcept { return _model; }
  const std::vector<int>& contexts()     const noexcept
  { return _contexts; }
  int decode_tokens() const noexcept { return _decode_tokens; }
  int prefill_chunk() const noexcept { return _prefill_chunk; }

  // Test seam: run the benchmark once. Returns true on success (a report
  // was produced); logs + returns false on error. `stop()` is polled
  // cooperatively between tests, between decode tokens, and during the
  // cooldown sleep -- when it goes true the run stops at the next check and
  // a PARTIAL report (the tests completed so far) is emitted.
  bool benchmark_once(const std::function<bool()>& stop = [] {
    return false;
  });

private:
#ifdef VPIPE_BUILD_APPLE_SILICON
  // Build the decode sampler params from the configured sampling knobs
  // (reusing `seed`). Greedy/argmax by default.
  genai::SamplerParams sampler_params_() const;
#endif
  // First `n` tokens of the flattened built-in calibration corpus (tiled
  // if the corpus is shorter than `n`); empty until built / on failure.
  std::vector<std::int32_t> real_prompt_(int n) const;

  std::string      _model;
  std::string      _models_db;
  std::vector<int> _contexts;
  int              _decode_tokens{};
  int              _prefill_chunk{};
  int              _warmup{};
  std::uint64_t    _seed{};
  double           _cooldown_s{};
  double           _temperature{};
  std::uint64_t    _top_k{};
  double           _top_p{};
  double           _min_p{};
  double           _repetition_penalty{};
  double           _presence_penalty{};
  bool             _mtp{};
  bool             _mtp_require_exact{};
  // Cached flattened real-text decode corpus (built once when MTP mode
  // is active; the prefill/incremental tests keep using the synthetic
  // prompt).
  std::vector<std::int32_t> _real_corpus;
};

}  // namespace vpipe

#endif

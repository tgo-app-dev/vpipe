#ifndef VPIPE_GENERATIVE_MODELS_DEEPSEEK_V41_METAL_DEEPSEEK_V41_MODEL_EXEC_H
#define VPIPE_GENERATIVE_MODELS_DEEPSEEK_V41_METAL_DEEPSEEK_V41_MODEL_EXEC_H

// The ModelExec for DeepSeek-V4.1-Flash (MetalDeepseekV41Model): the
// token-id path LoadedLanguageModel drives an own-KV exec through --
// prefill, decode a token at a time, the host logits. The model keeps
// every context's state (windows, compressed latents, index keys, the
// n-gram history) itself, keyed by ContextId.

#include "generative-models/deepseek-v41/metal-deepseek-v41-model.h"
#include "generative-models/model-exec.h"
#include "generative-models/speculative-decode.h"

#include <functional>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace vpipe { class SessionContextIntf; }
namespace vpipe::metal_compute { class MetalCompute; }

namespace vpipe::genai {

class MetalDeepseekV41ModelExec : public ModelExec, private SpecDecodeExt {
public:
  MetalDeepseekV41ModelExec(const std::string&                    model_dir,
                            metal_compute::MetalCompute*          mc,
                            const MetalDeepseekV41Model::Options& opt,
                            const SessionContextIntf*             session);

  bool valid() const noexcept override { return _model != nullptr; }
  bool owns_kv() const noexcept override { return true; }
  std::size_t kv_bytes() const override
  {
    return _model != nullptr ? _model->kv_bytes() : 0;
  }
  int context_seq_len(ContextId ctx) const override
  {
    return _model != nullptr ? _model->seq_len(ctx.v) : 0;
  }
  bool branch_context(ContextId parent, ContextId child) override
  {
    return _model != nullptr && _model->branch(parent.v, child.v);
  }
  void release_context(ContextId ctx) override
  {
    if (_model != nullptr) { _model->release(ctx.v); }
  }

  std::int32_t
  prefill(ContextId ctx, std::span<const std::int32_t> tokens) override;
  std::int32_t decode_one(ContextId ctx, std::int32_t id) override;
  const std::vector<float>& last_logits_host() const override
  {
    return _logits;
  }
  void set_eval_per_layer(bool) noexcept override {}
  void set_suppressed_tokens(std::span<const std::int32_t> ids) override
  {
    _suppress.assign(ids.begin(), ids.end());
  }

  MetalDeepseekV41Model* model() { return _model.get(); }

  // ---- speculative decode with the DSpark head (Options::dspark) ------
  // A round drafts the block after the pending token, verifies it and the
  // drafts in one forward, keeps the prefix the verify agrees with
  // (greedy: its argmax; a sampler: one draw a kept token, in order -- the
  // serial loop's draws) and forgets the rest (truncate). Token-exact
  // against the serial decode: a verify's rows are its decode's.
  void* query_extension(std::string_view id) noexcept override;
  bool supports_mtp() const override
  {
    return _model != nullptr && _model->has_dspark();
  }
  bool
  mtp_generate(ContextId ctx, std::int32_t first_token, int max_tokens,
               int rope_first, const GpuSamplerParams& sp,
               const std::function<bool(std::int32_t)>& is_stop,
               const std::function<bool(std::span<const std::int32_t>)>&
                   on_tokens,
               int* produced, bool* hit_stop) override;
  // Tests: replace a round's drafts (the block after `pending`) before
  // the verify -- a known-right continuation with planted errors makes
  // every accepted length happen on a model whose head drafts noise.
  using DraftOverride = std::function<void(
      std::int32_t pending, std::vector<std::int32_t>* drafts)>;
  void set_draft_override(DraftOverride f) { _draft_override = std::move(f); }
  // What the drafts' confidence said and what the verify kept, and what a
  // round cost (benches): per block row i and tenth b of sigmoid(conf[i]),
  // the rounds that reached row i (every draft before it kept) and those
  // of them that kept it, at [i * 10 + b]; the verify's ms and count by
  // its rows ([rows]); the drafts' ms.
  struct DraftStats {
    std::vector<std::int64_t> reached, kept;
    std::vector<double> verify_ms;
    std::vector<std::int64_t> verifies;
    double draft_ms = 0.0;
    std::int64_t drafts = 0;
  };
  const DraftStats& draft_stats() const { return _dstats; }
  // The speculative decode's modes (VPIPE_DSV41_MTP_DRAFT / _DEPTH win):
  // chained drafts (a DSpark pass a draft over the real drafts so far) in
  // place of the one-pass block, and omlx's acceptance-only depth (one
  // past the last round's kept drafts) in place of the measured rate.
  // Every mode keeps the plain decode's tokens.
  void set_mtp_modes(bool chain, bool acceptance)
  {
    _mtp_chain = chain;
    _mtp_acceptance = acceptance;
    _acc_depth = 3;
  }

  // A prefill whose rows at `positions` (indices into ids) are `rows`
  // ([n_mm, dim] f32): an image's span, from MetalDeepseekV41Vision.
  std::int32_t prefill_multimodal_ids(ContextId ctx,
                                      std::span<const std::int32_t> ids,
                                      const std::vector<float>& rows,
                                      int n_mm,
                                      const std::vector<int>& positions);

  // The routed experts' bytes in a checkpoint, what the rest of the model
  // holds once loaded (the FP8 projections as their codes and a scale per
  // 32 of a row; not the n-gram tables, which are read by row, nor the
  // DSpark stages), what its vision tower holds (`vision`, optional) and
  // the checkpoint's bytes on disk (`disk`, optional).
  static bool expert_bytes(const std::string& model_dir,
                           std::size_t* experts, std::size_t* rest,
                           std::size_t* vision = nullptr,
                           std::size_t* disk = nullptr);
  // The expert cache's size for this box (0: every expert resident) --
  // the engine's call, on the rule the other MoE models take: what the
  // model holds, its peers', the plan's scratch, `run` (K/V and
  // activations) and the irreversible 8 GB headroom against total RAM;
  // and, where `mc` is given, the GPU's own limit: everything the model
  // keeps is a Metal buffer, and past the device's recommended working
  // set the GPU pages them in and out of every command buffer (MEASURED,
  // M4 Pro 64 GB: a 39 GB cache beside the ~17 GB rest -- the FP8 then
  // widened -- 55.7 GB against a 55.7 GB working set, decoded a token in
  // 46 s, 43 of them GPU waits; 16 GB, 3.6 s, 0.44 of them the GPU).
  // VPIPE_DSV41_EXPERT_CACHE_MB pins it (0 = resident).
  // `dspark`: the DSpark head loads beside it (Options::dspark).
  // `limit` (bytes, 0 none): the most the model may hold in all, `run`
  // in it (load_spec::kMemoryLimitMb) -- the cache what it leaves.
  static std::size_t lm_expert_cache_bytes(
      const SessionContextIntf* session, const std::string& model_dir,
      std::size_t run, const metal_compute::MetalCompute* mc = nullptr,
      bool dspark = false, std::size_t limit = 0);
  // Whether the DSpark draft head pays on this box -- the engine's call,
  // which text-chat asks before loading the head: only where every expert
  // is resident. Where they are cached, a checked draft reads its own
  // missed experts (~11-13 ms a row on top of a ~30 ms step) and the draft
  // itself costs ~5 ms, against ~0.6 tokens kept a round. MEASURED on the
  // M5 Ultra (internal SSD, 512 greedy tokens from cold), plain against
  // DSpark (the measured draft count; one draft pinned): VPIPE_RAM_LIMIT_MB
  // 128 GB 20.5 against 19.5-19.8 (19.3), 256 GB 21.4 against 20.3-20.7
  // (19.9). THE TEXT MOVES IT: since the parallel expert reads, the
  // story prompt at 256 GB is 24.9 against 24.2 (128 GB 24.4 against
  // 22.1), but ten GSM8K questions in one process (3.1-3.3 tokens a
  // round) 39.3-39.6 against 51.5-52.8 (128 GB 27.0-27.4 against
  // 28.0-29.1). VPIPE_DSV41_DSPARK=0|1 pins it. Under a `limit` (the
  // most the model may hold), every expert resident within it.
  static bool dspark_pays(const SessionContextIntf* session,
                          const std::string& model_dir, std::size_t run,
                          const metal_compute::MetalCompute* mc = nullptr,
                          std::size_t limit = 0);

private:
  // ---- SpecDecodeExt -------------------------------------------------
  std::string_view drafter() const noexcept override;
  bool attach_drafter(const std::string& dir, int bits,
                      std::string* err) override;
  bool generate(const SpecDecodeRequest& rq, SpecDecodeResult* out) override;
  DraftOverride _draft_override;
  DraftStats _dstats;
  // HOW MANY DRAFTS A ROUND VERIFIES, measured (see generate()): the
  // verify's ms by its rows and the draft's, as moving averages; the
  // confidence calibration, its counts halved every kHalve rounds.
  struct DraftChoice {
    static constexpr int kExplore = 8;     // one round in this many tries
    static constexpr int kHalve = 512;     // rounds, calibration halves
    std::vector<double> v;                 // [1] a plain step's ms, [r] an
                                           // r-row verify's ratio to it
    std::vector<int> nv;                   // [rows] samples
    double d = 0.0;                        // draft ms
    int nd = 0;
    double spark = 0.0;                    // tokens / ms, drafted rounds
    std::vector<double> reached, kept;     // [i * 10 + bin]
    std::uint64_t rounds = 0;
  };
  DraftChoice _choice;
  // The cost of a verify of `rows` rows, interpolated where unmeasured.
  double verify_cost_(int rows) const;
  // How many of a block's drafts (conf[0..k_max)) to verify: the most
  // tokens a ms by the calibration and the measured costs.
  int verify_count_(const std::vector<float>& conf, int k_max,
                    double* rate, double draft_ms = -1.0) const;
  double kept_share_(float conf, int at) const;
  // the draft and depth modes (see generate(); env overrides)
  bool _mtp_chain = false, _mtp_acceptance = false;
  int _acc_depth = 3;                     // the acceptance rule's next depth

  std::int32_t run_(ContextId ctx, std::span<const std::int32_t> ids,
                    const MetalDeepseekV41Model::Media* media = nullptr);

  std::unique_ptr<MetalDeepseekV41Model> _model;
  const SessionContextIntf* _session = nullptr;
  std::vector<float> _logits;
  std::vector<std::int32_t> _suppress;
};

}  // namespace vpipe::genai

#endif  // VPIPE_GENERATIVE_MODELS_DEEPSEEK_V41_METAL_DEEPSEEK_V41_MODEL_EXEC_H

#ifndef VPIPE_GENERATIVE_MODELS_SPECULATIVE_DECODE_H
#define VPIPE_GENERATIVE_MODELS_SPECULATIVE_DECODE_H

// Speculative decoding as a capability of a language-model exec: a
// DRAFTER proposes tokens, the model verifies a block of them in one
// forward, and the longest prefix it agrees with is kept. The drafter
// decides only how many tokens a forward yields, never which: greedy
// output is the model's own greedy output, a sampler's is its own
// per-slot samples.
//
// Two kinds of drafter exist in this tree, behind one interface:
//
//   mtp      a Multi-Token-Prediction head bundled with (or published
//            beside) the model: one extra decoder layer drafting one or
//            two tokens a round.
//   dflash   a DFlash / DFlash 2 block-diffusion drafter (z-lab, Inco
//   dflash2  AI): a small separate model conditioned on the target's
//            hidden states, drafting a whole block in one forward. Named
//            to a model by LoadSpec::extra[load_spec::kDraftDir]
//            (generative-model-manager.h, beside the MTP drafter's
//            load_spec::kMtpDir).
//
// An exec that can do this answers ModelExec::query_extension(
// kExtensionId) with a SpecDecodeExt*. LoadedLanguageModel wraps it
// (spec_decode_available / spec_generate / attach_draft_model), which is
// what a stage calls.

#include "common/flex-data.h"
#include "common/vpipe-api.h"
#include "generative-models/context-manager.h"   // ContextId
#include "generative-models/model-exec.h"        // GpuSamplerParams

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>

VPIPE_API_BEGIN

namespace vpipe::genai {

namespace spec_decode {

// ModelExec::query_extension id; the answer is a SpecDecodeExt*.
inline constexpr std::string_view kExtensionId =
    "vpipe.speculative-decode/1";

// SpecDecodeExt::drafter() values.
inline constexpr std::string_view kDrafterMtp     = "mtp";
inline constexpr std::string_view kDrafterDFlash  = "dflash";
inline constexpr std::string_view kDrafterDFlash2 = "dflash2";

}  // namespace spec_decode

// One speculative decode, from a prefilled context. `first_token` is the
// prefill's already-decided next token, NOT yet appended; the decode
// appends up to max_tokens tokens counting it. The functions are
// borrowed for the call.
struct SpecDecodeRequest {
  ContextId                    ctx;
  std::int32_t                 first_token = -1;
  int                          max_tokens  = 0;
  // Rotary position of first_token: -1 sequential text, >= 0 the
  // mROPE-advanced position a multimodal prefill left.
  int                          rope_first  = -1;
  // Greedy (default) or the sampler the verify draws each slot with.
  GpuSamplerParams             sampler{};
  // The prompt the penalty seen-set starts from (pdecode_begin's).
  std::span<const std::int32_t> prime;
  // Ends the decode at the first accepted stop token, which is NOT kept.
  const std::function<bool(std::int32_t)>*                  is_stop = nullptr;
  // Each round's newly kept tokens, for streaming; false aborts.
  const std::function<bool(std::span<const std::int32_t>)>* on_tokens =
      nullptr;
  // Tokens a round verifies (anchor + drafts): 0 = the drafter's choice
  // (DFlash adapts it to the measured cost of a round on this machine).
  int                          block = 0;
  // Anything this struct has no field for: see common/flex-bag.h.
  const FlexData*              borrowed_extra = nullptr;
};

struct SpecDecodeResult {
  int  produced = 0;      // tokens appended, a stop token excluded
  bool hit_stop = false;  // a stop token ended it
  long rounds   = 0;      // verify forwards
  long drafted  = 0;      // draft tokens offered to the verifier
  long accepted = 0;      // of those, kept
  // Anything this struct has no field for: see common/flex-bag.h.
  FlexData extra;
};

// The capability. Owned by the exec that returned it; valid for the
// exec's lifetime.
class SpecDecodeExt {
public:
  // spec_decode::kDrafter*, or "" when nothing is attached / usable.
  virtual std::string_view drafter() const noexcept = 0;
  // Attach a separate drafter from `dir` ("" detaches), held at `bits`
  // (see load_spec::kDraftBits). False with *err when it cannot draft for
  // this model; the model is then unchanged.
  virtual bool attach_drafter(const std::string& dir, int bits,
                              std::string* err) = 0;
  // Run one speculative decode. False when no drafter is usable or the
  // first round failed (nothing appended); the caller then decodes the
  // ordinary way.
  virtual bool generate(const SpecDecodeRequest& rq,
                        SpecDecodeResult* out) = 0;

protected:
  ~SpecDecodeExt() = default;
};

}  // namespace vpipe::genai

VPIPE_API_END

#endif

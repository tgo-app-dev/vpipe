#ifndef VPIPE_GENERATIVE_MODELS_SPEC_KEYS_H
#define VPIPE_GENERATIVE_MODELS_SPEC_KEYS_H

// The keys of the SPEC beats: the FlexData objects `scheduler-select`,
// `diffusion-sampler-select` and `sampler-select` emit, which reach a
// model family as VideoGenRequest / ImageGenRequest `scheduler_spec` and
// `sampler_spec` and an LLM stage as its sampler beat.
//
// A family that samples its own way reads these directly, so they are
// part of the plugin contract: keys and the enumerated VALUES alike are
// added, never renamed or repurposed, and an absent key means the
// default a spec had before the key existed. A value a reader does not
// know is warned about and replaced by its default -- never silently
// substituted by the nearest one.

#include <string_view>

#include "common/vpipe-api.h"

VPIPE_API_BEGIN

namespace vpipe {
namespace genai {

// ---- the flow-matching SCHEDULER: the discrete sigma grid -----------
namespace scheduler_spec {

// What the object is. Value: kFlowMatch.
inline constexpr std::string_view kScheduler = "scheduler";
inline constexpr std::string_view kFlowMatch = "flow_match";

// The grid's shape. Values below; default kSimple.
inline constexpr std::string_view kType        = "type";
inline constexpr std::string_view kSimple      = "simple";
inline constexpr std::string_view kKarras      = "karras";
inline constexpr std::string_view kExponential = "exponential";
inline constexpr std::string_view kBooguV1     = "boogu_v1";
inline constexpr std::string_view kUnipcFlow   = "unipc_flow";

inline constexpr std::string_view kSteps = "steps";   // int
inline constexpr std::string_view kShift = "shift";   // real
// How `shift` bends the grid. Values: kShiftExponential (default),
// kShiftLinear.
inline constexpr std::string_view kShiftType        = "shift_type";
inline constexpr std::string_view kShiftExponential = "exponential";
inline constexpr std::string_view kShiftLinear      = "linear";
inline constexpr std::string_view kRho = "rho";       // real; karras

// Resolution-dependent shift (the diffusers "use_dynamic_shifting").
inline constexpr std::string_view kDynamicShift  = "dynamic_shift";  // bool
inline constexpr std::string_view kBaseShift     = "base_shift";     // real
inline constexpr std::string_view kMaxShift      = "max_shift";      // real
inline constexpr std::string_view kShiftTerminal = "shift_terminal"; // real
inline constexpr std::string_view kBaseSeq       = "base_seq";       // int
inline constexpr std::string_view kMaxSeq        = "max_seq";        // int
inline constexpr std::string_view kSeqLen        = "seq_len";        // int
inline constexpr std::string_view kNumTrain      = "num_train";      // int

}  // namespace scheduler_spec

// ---- the flow-matching SAMPLER: how a step is taken ------------------
namespace sampler_spec {

// What the object is. Value: kFlowMatch.
inline constexpr std::string_view kSampler   = "sampler";
inline constexpr std::string_view kFlowMatch = "flow_match";

// The step rule. Values below; default kEuler.
inline constexpr std::string_view kMethod   = "method";
inline constexpr std::string_view kEuler    = "euler";
inline constexpr std::string_view kHeun     = "heun";
inline constexpr std::string_view kDpmpp2m  = "dpmpp_2m";
inline constexpr std::string_view kDpmppSde = "dpmpp_sde";
inline constexpr std::string_view kDmd      = "dmd";
inline constexpr std::string_view kUnipc    = "unipc";

inline constexpr std::string_view kEta    = "eta";      // real
inline constexpr std::string_view kSNoise = "s_noise";  // real
inline constexpr std::string_view kSeed   = "seed";     // int
// kDmd only: the sigma the student was conditioned at. Real.
inline constexpr std::string_view kConditioningSigma = "conditioning_sigma";
// kUnipc only: the solver's order (int) and variant (kBh1 / kBh2).
inline constexpr std::string_view kOrder      = "order";
inline constexpr std::string_view kSolverType = "solver_type";
inline constexpr std::string_view kBh1        = "bh1";
inline constexpr std::string_view kBh2        = "bh2";

}  // namespace sampler_spec

// ---- the LANGUAGE-MODEL sampler: how a token is drawn ----------------
namespace token_sampler_spec {

// What the object is. Value: kToken.
inline constexpr std::string_view kSampler = "sampler";
inline constexpr std::string_view kToken   = "token";

inline constexpr std::string_view kTemperature       = "temperature";  // real
inline constexpr std::string_view kTopK              = "top_k";        // int
inline constexpr std::string_view kTopP              = "top_p";        // real
inline constexpr std::string_view kMinP              = "min_p";        // real
inline constexpr std::string_view kRepetitionPenalty = "repetition_penalty";
inline constexpr std::string_view kPresencePenalty   = "presence_penalty";
inline constexpr std::string_view kSeed              = "seed";         // uint

}  // namespace token_sampler_spec

}  // namespace genai
}  // namespace vpipe

VPIPE_API_END

#endif

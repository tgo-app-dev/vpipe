// model-memory-host.h -- sizing for the host's own models, beside the
// shared rules of model-memory.h.
//
// NOT INSTALLED AND NOT EXPORTED, on purpose. What is declared here sizes
// machinery no plugin can reach (the routed-expert cache lives inside the
// host's Qwen stack, which the SDK does not ship), so putting it in
// model-memory.h would export a symbol no plugin can use -- and a plugin
// that called it anyway would fail to load on every ABI-8 host built
// before it, ahead of any version or feature check. Keep it here until the
// machinery it sizes is offered to plugins, and then behind a feature.

#ifndef VPIPE_STAGES_MODEL_MEMORY_HOST_H
#define VPIPE_STAGES_MODEL_MEMORY_HOST_H

#include "generative-models/lm-run-memory.h"
#include "stages/model-memory.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace vpipe {
class SessionContextIntf;
namespace model_memory {

// THE ROUTED-EXPERT CACHE a Mixture-of-Experts model takes on this box
// (MetalQwenModel::Config::expert_cache_bytes): 0 when its whole preload
// fits beside its peers -- plan_streaming's rule, total RAM and the
// irreversible headroom, so every stage answers alike -- else its experts'
// bytes LESS the shortfall, never below one layer's worth (`layer_experts`:
// a prefill step may route to every expert of a layer at once). `dir` is
// counted at its preload whether or not it is declared yet; `experts` is
// what its routed experts hold resident, `scratch` what it allocates to
// run (K/V, activations) beside them.
std::size_t moe_expert_cache(const SessionContextIntf* session,
                             const std::string& dir, std::size_t preload,
                             std::size_t experts, std::size_t layer_experts,
                             std::size_t scratch,
                             std::size_t headroom = kStreamHeadroom);

// THE CLAIMS OF A STAGE THAT HOLDS A LANGUAGE MODEL in `dir` (text-chat,
// visual-qa, realtime-vqa), one rule for all of them:
//   * its weights -- STREAMABLE at the routed-expert cache's floor for a
//     Mixture-of-Experts model (the rest of the model and one layer of
//     experts; a hyper-connection model's n-gram table, read by row, no
//     part of it), else held whole;
//   * a Qwen vision tower's activation scratch at the largest picture it
//     admits -- `picture_pixels`, or the checkpoint's own bound (16.7 MP
//     on a Qwen3-VL repo: ~5 GB) when 0, or `picture_tokens` merged
//     tokens' worth when that is set, none when < 0 -- unphased and labelled
//     "vision:<label>": per STAGE, since two stages encode at once;
//   * with `run.draft_head`, the model's own draft head where loading it
//     is optional (DeepSeek-V4.1's DSpark), held whole beside the weights;
//   * what it RUNS in beside them -- its K/V and per-context state, a
//     prefill's activations, the decode scratch -- as `run` shapes it
//     (genai::lm_run_bytes), unphased and labelled lm_run_label(dir): per
//     CHECKPOINT, as the stages naming one share its loaded model. The
//     loader sizes a routed-expert cache beside that same figure.
// `memory_limit` (bytes, 0 none) is the host's limit on what the model
// holds (genai::load_spec::kMemoryLimitMb): a Mixture of Experts' weights
// are then booked at what its cache sized within the limit leaves it
// holding (kClaimHeldBytes), never under the floor, so the plan its peers
// size against has the room the limit gives them. A dense model is booked
// whole, as it loads. `expert_cache` (bytes, 0 none) is the cache the host
// asks for (genai::load_spec::kExpertCacheMb): the weights are booked at
// the rest of the model and that cache at most, within the limit alike.
std::vector<ResourceClaim> lm_claims(const SessionContextIntf* session,
                                     const std::string& dir,
                                     const std::string& label,
                                     const genai::LmRunShape& run,
                                     std::int64_t picture_pixels = 0,
                                     int picture_tokens = 0,
                                     std::size_t memory_limit = 0,
                                     std::size_t expert_cache = 0);

// int, on a weights claim's `extra`: the most the component will HOLD,
// when that is less than its checkpoint weighs and it is not a floor -- a
// Mixture of Experts whose routed-expert cache is sized within a host's
// memory limit. The weights planner books the checkpoint at it (never
// under the claim's floor, never over the checkpoint). Read through
// DeliveringResourceClaim (pipeline/resource-claim-delivery.h). Host-only:
// the host's language-model stages set it; no plugin is offered it yet.
inline constexpr std::string_view kClaimHeldBytes = "held_bytes";

// THE RESIDENCY RESERVE of a model resident in `phase` that grows a
// resident set by measurement -- a conditioning encoder streaming through
// the raw-source slots (MetalQwenModel::Config::stream_decode): the widest
// activation scratch the plan books for any OTHER phase. Those arenas are
// what may allocate while the encoder still holds its set -- a DiT's
// forward and a VAE decode of an earlier prompt run beside the next
// prompt's conditioning, and an encoder kept past its beat (kept warm, a
// `keep` policy) sits beside this one's -- and no reading of free memory
// can see them coming. Its own phase is left out: what the encoder's
// forward allocates is in its hands before it admits a layer.
std::size_t residency_reserve_beside(const SessionContextIntf* session,
                                     std::string_view phase);

}  // namespace model_memory
}  // namespace vpipe

#endif  // VPIPE_STAGES_MODEL_MEMORY_HOST_H

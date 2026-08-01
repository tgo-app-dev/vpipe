// model-memory.h -- shared host-memory sizing for the model-holding stages.
//
// The diffusion stages each hold a big chunk of weights (a DiT, a text encoder,
// a VAE) and each has to answer the same question: is this box able to keep my
// weights resident beside everybody else's, or should I drop them when idle and
// reload on the next beat? The inputs to that decision -- physical RAM and the
// on-disk weight bytes of a component directory -- were file-static in
// text-to-image-stage.cc; they live here so the conditioner and the VAE stages
// reach the SAME numbers rather than a second opinion.
//
// Weight bytes come from the .safetensors sizes, which is a good proxy for the
// wired footprint: the loaders either mmap the file or copy it into a
// SharedBuffer at the same width. Quantized checkpoints therefore report their
// quantized size, which is the point -- a 4-bit Boogu is ~5.6 GB, not ~20 GB.

#ifndef VPIPE_STAGES_MODEL_MEMORY_H
#define VPIPE_STAGES_MODEL_MEMORY_H

#include <cstddef>
#include <string>
#include <vector>

namespace vpipe {
class SessionContextIntf;
namespace model_memory {

// Working-set headroom for a decision that can be REVISED later.
// unload_when_idle is the case: it is a per-beat behaviour flag, decided
// after the init barrier when every peer has loaded, and changing it
// costs nothing but the next idle point.
inline constexpr std::size_t kHeadroom = 6ull << 30;

// Headroom for a decision that is baked in at CONSTRUCTION and cannot
// practically be revised. Block streaming is the case: it is an argument
// to MetalXTransformer::load, so changing the answer afterwards means
// destroying and rebuilding the DiT -- a full reload of up to ~20 GB to
// correct a guess, which is worse than the guess.
//
// Wider than kHeadroom on purpose, because the costs are asymmetric.
// Failing to stream when you should have means thrash or an OOM kill;
// streaming when you needn't costs ~2-3x per step. Under any residual
// uncertainty the cheap error is to stream, so the irreversible decision
// gets the bigger cushion.
//
// Declarations (Stage::declare_models) removed the largest source of
// that uncertainty -- peers this stage cannot see from its own config --
// but not all of it: CoreML models, LMDB, and activation scratch are
// still outside the manager's accounting.
inline constexpr std::size_t kStreamHeadroom = 8ull << 30;

// Total physical RAM in bytes, 0 when the query fails (callers treat 0 as
// "unknown" and keep the roomier behaviour rather than guessing small).
std::size_t phys_ram();

// Sum of the .safetensors bytes under `dir`, recursively. 0 if unreadable.
std::size_t dir_weights_bytes(const std::string& dir);

// The weight footprint of keeping `dirs` resident, in bytes.
//
// A UNION, not a sum, and that distinction is the whole point. Two things
// make a plain per-directory sum wrong now that the model manager owns
// checkpoints:
//
//   * A directory named twice -- by two stages of one graph, or by two
//     pipelines sharing a model -- is loaded ONCE. Counting it twice
//     over-estimates and pushes a box into streaming it does not need.
//   * A model some OTHER stage loaded is real memory this stage cannot
//     see from its own config. Leaving it out under-estimates.
//
// So this starts from what the manager is actually HOLDING (already
// deduped, one entry per checkpoint) and adds only those `dirs` it does
// not hold yet, estimated from their on-disk bytes. A checkpoint that is
// open contributes what it really holds, not what its files weigh --
// which for a streaming DiT is just the pinned prefix.
//
// With no session or no manager this degrades to the old behaviour: the
// on-disk sum of `dirs`, deduped against itself.
std::size_t weight_footprint(const SessionContextIntf*        session,
                             const std::vector<std::string>&  dirs);

// Would keeping `dirs` resident, plus `headroom` for working set, fail to
// fit in physical RAM? False when RAM is unknown, so an unreadable sysctl
// never turns unloading on by itself.
//
// Deliberately measured against TOTAL RAM rather than what is free right
// now. This is a once-per-run decision that every stage in a graph must
// answer the same way, and it has to be reproducible -- VPIPE_RAM_LIMIT_MB
// exists so a 16 GB box can be simulated on a big one. Free memory moves
// underfoot: the first stage to initialize would see an empty machine and
// the last a full one, so the same graph would stream or not depending on
// stage order and on whatever else the box happened to be doing.
//
// The live budget (MetalCompute::memory_budget) is the right tool for the
// other kind of question -- "will this TRANSIENT spike fit right now" --
// and that is where the stages already use it, e.g. freeing a DiT before
// a large VAE decode.
bool bounded(const SessionContextIntf*       session,
             const std::vector<std::string>& dirs,
             std::size_t                     headroom);

// The block-streaming decision for a DiT, and how much of RAM its pinned
// prefix may use. Three families asked this identically (FLUX.2,
// Qwen-Image-Edit, Boogu) with the rule copy-pasted; it lives here so
// they cannot drift, and so all of them get the manager-aware footprint.
//
// `stream` is true when the DiT plus its encoder plus everything else
// the session already holds will not fit in RAM with `headroom` to
// spare. `pin_frac` then sizes the pinned prefix against what will be
// resident BESIDES the DiT, which is what the prefix has to coexist
// with. Both are 0/false when RAM is unknown.
struct StreamPlan {
  bool        stream    = false;
  double      pin_frac  = 0.0;
  std::size_t footprint = 0;   // DiT + encoder + everything resident
  std::size_t others    = 0;   // the same, minus the DiT's own bytes
};

StreamPlan plan_streaming(const SessionContextIntf* session,
                          const std::string&        dit_dir,
                          const std::string&        enc_dir,
                          std::size_t               headroom);

// The `unload_when_idle` config value shared by the model-holding stages:
//   auto (default) -- decide from RAM vs the pipeline's weight footprint
//   always         -- drop the weights after every beat
//   never          -- keep them resident
enum class UnloadPolicy { kAuto, kAlways, kNever };

// Parse a config string; unknown values return kAuto and set `*bad` so the
// caller can warn (stage config is deferred-validated -- never throw).
UnloadPolicy parse_unload_policy(const std::string& s, bool* bad = nullptr);

const char* unload_policy_name(UnloadPolicy p);

}  // namespace model_memory
}  // namespace vpipe

#endif

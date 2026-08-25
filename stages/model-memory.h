// model-memory.h -- shared host-memory sizing for the model-holding stages.
//
// The diffusion stages each hold a big chunk of weights (a DiT, a text encoder,
// a VAE) and each has to answer the same question: is this box able to keep my
// weights resident beside everybody else's, or should I drop them when idle and
// reload on the next beat? The inputs to that decision -- physical RAM and the
// on-disk weight bytes of a component directory -- were file-static in
// generate-image-stage.cc; they live here so the conditioner and the VAE stages
// reach the SAME numbers rather than a second opinion.
//
// Weight bytes come from the .safetensors sizes, which is a good proxy for the
// wired footprint: the loaders either mmap the file or copy it into a
// SharedBuffer at the same width. Quantized checkpoints therefore report their
// quantized size, which is the point -- a 4-bit Boogu is ~5.6 GB, not ~20 GB.

#ifndef VPIPE_STAGES_MODEL_MEMORY_H
#define VPIPE_STAGES_MODEL_MEMORY_H

#include "pipeline/resource-plan.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace vpipe {
class SessionContextIntf;
namespace model_memory {

// The ResourceClaim kind consumed by the model-weight planner, which
// turns a claimed checkpoint directory into a GenerativeModelManager
// declaration. The constant lives here rather than in the pipeline core
// on purpose: the runtime routes claims by kind without knowing that
// "model weights" is one of the things a graph can want.
inline constexpr std::string_view kWeightsKind = "model-weights";

// Wrap checkpoint directories as claims, for a stage's
// declare_resources() override:
//
//   std::vector<ResourceClaim>
//   FooStage::declare_resources() const
//   {
//     return model_memory::weight_claims({_model_dir, _encoder_dir});
//   }
//
// Empty directories are dropped, so a conditionally-configured
// component needs no branch at the call site.
std::vector<ResourceClaim> weight_claims(std::vector<std::string> dirs);

// One weight claim that also states the FLOOR this component can be
// reduced to -- what a block-streaming DiT holds when it streams: its
// trunk plus the in-flight slots, not the checkpoint.
//
// Separate from weight_claims() rather than an extra argument on it,
// because a floor is per-DIRECTORY and the plural form takes a list. A
// caller with two streamable components makes two claims.
//
// `floor` above the checkpoint's own size is clamped by the planner: it
// would otherwise report a streamed total larger than the preloaded one,
// and this figure is meant to decide whether a graph can run at all.
ResourceClaim weight_claim_streamable(std::string dir, std::size_t floor);

// ---- phases ----------------------------------------------------------
//
// A generation graph runs in stages that do not overlap: the text
// encoder produces one conditioning and is done, THEN a DiT denoises
// for minutes, THEN a VAE turns the latent into pixels. Summing all
// three sizes the box for a moment that never happens -- and on a
// MiniMax-H3 graph the sum is 94 GB where the largest single phase is
// 78 GB, which is the difference between streaming a 33B DiT and
// holding it.
//
// So a claim may name the phase it belongs to, and the planner takes
// the maximum across phases rather than their sum (see
// GenerativeModelManager::phase_footprint). Empty -- the default and
// what weight_claims() above produces -- means held for the whole run.
//
// THE VOCABULARY IS FIXED, and deliberately: an unrecognised phase is
// warned about and treated as persistent. A typo that quietly became
// its own phase would be counted apart from the claims it actually
// coexists with, and that error UNDER-counts, which is the direction
// that thrashes.
// THE PHASES ARE ORDERED, and a claim spans an INTERVAL of them.
//
// A single phase per claim can only express "this is resident while its
// own stage runs", and the thing that actually sizes a constrained box
// is not that -- it is what COEXISTS at each moment, and the largest
// term is usually something produced by one phase and consumed by a
// later one. The output latent is alive from the denoise that writes it
// until the last decode that reads it; the conditioning is alive from
// the encoder that produces it until the denoise that consumes it.
// Neither belongs to one phase, and counting them as persistent (the
// old answer for anything not single-phase) puts them in every phase
// including the ones they are absent from.
//
// So what a memory-constrained box must hold is
//
//   max over phases p of ( sum of every claim alive during p )
//
// which for a generation graph reads as the four terms below -- and no
// two of them are ever live together:
//
//   condition     encoder floor + encoder scratch + conditioning
//   denoise       conditioning + DiT floor + DiT scratch + latent
//   decode-audio  latent + audio VAE floor + its scratch + PCM
//   decode        latent + video VAE floor + its scratch + frames
//
// On an ABUNDANT box the same graph may keep all of it resident at once
// so a second launch pays no reload, which is the sum rather than the
// max. Both are reported; the max is what decides whether a graph can
// run, the sum is what it costs to make relaunches free.
//
// The ORDER below is the order phases run in, and intervals are
// inclusive of both ends. Adding a phase means putting it in the right
// place here -- see phase_order().
inline constexpr std::string_view kPhaseCondition   = "condition";
inline constexpr std::string_view kPhaseDenoise     = "denoise";
// The two decodes are separate phases because on a constrained box they
// do not overlap either: an audio VAE and a video VAE are loaded and
// dropped independently, and summing them sizes a moment that does not
// happen. Both read the same latent, which is why the latent's interval
// has to REACH both rather than belong to one.
inline constexpr std::string_view kPhaseDecodeAudio = "decode-audio";
inline constexpr std::string_view kPhaseDecode      = "decode";

// Position of `phase` in the running order, or -1 when it is not a
// known phase. An unphased claim is alive throughout and has no index.
int phase_order(std::string_view phase);

// How many phases there are, for a caller walking all of them.
int phase_count();

// Claims for weights this stage holds only during `phase`.
//
// TWO CONDITIONS, both required, and neither is checkable by the
// planner:
//
//   1. The release must be certain from CONFIG, before anything loads.
//      An idle policy of `destroy` qualifies. `park` does NOT --
//      park_weights() returns 0 for a set that reads uncached, which is
//      every text encoder today, so a parked encoder is still entirely
//      resident and a peer that subtracted it is short by its whole
//      size.
//
//   2. The decision to release must not itself be taken by consulting a
//      phased figure. `auto` resolves against bounded(), which sums
//      everything on purpose; were it to read the phased number it
//      would see the room its own claim invented and conclude it need
//      not release after all.
//
// The release is reported at unload via
// GenerativeModelManager::note_phase_released, and a phase claim whose
// release never arrives is warned about at the end of the run.
std::vector<ResourceClaim>
weight_claims_in_phase(std::vector<std::string> dirs,
                       std::string_view         phase);

// ---- activation scratch ----------------------------------------------
//
// The second thing that has to fit, and until now the one nothing
// declared. Weights are what a model HOLDS; scratch is what it
// ALLOCATES to run, and for a VAE decode the second dwarfs the first --
// FLUX.2's VAE weighs 160 MB on disk and its decode peaks around 2.8 GB
// at 1024x1024. A graph accounted purely by weights therefore reads as
// roomy right up to the allocation that does not fit.
//
// Declared as an ordinary claim so it is on the books before anything
// runs, and PHASED like weights, because scratch exists only while its
// stage is running: a decode's arena is not resident during the
// denoise, so `kPhaseDecode` keeps it out of the DiT's sizing and in the
// box-level peak.
inline constexpr std::string_view kScratchKind = "activation-scratch";

// A PRESENCE marker for an arena whose size is not knowable at plan
// time -- an image edit's geometry comes from the reference image, so
// there is no height/width in any config to estimate from.
//
// Declaring this rather than nothing is what keeps the plan
// authoritative about WHAT exists while leaving runtime to supply HOW
// MUCH, which is exactly the contract weights already have
// (declare_weights / revise_declaration, where revise also refuses to
// create). Without it a stage would have to introduce an arena the plan
// never saw, and then nothing distinguishes a legitimate late truth
// from a typo'd label.
//
// Negligible by construction: 4 KB cannot move any sizing decision, and
// the first beat replaces it with the real figure before the decision
// it feeds is acted on.
inline constexpr std::size_t kUnknownArena = 4096;

// `label` names the allocation for the log (e.g. "vae-decode"); it does
// not have to be unique, but two claims sharing one label are counted
// ONCE, at the larger, on the assumption they describe the same arena.
std::vector<ResourceClaim>
scratch_claims(std::string label, std::size_t bytes, std::string_view phase);

// A PAYLOAD: bytes that outlive the phase producing them.
//
// Same machinery as an arena -- a label, a size, a lifetime -- and a
// separate name because the two are different facts and a reader sizing
// a graph needs to tell them apart. A scratch arena is torn down when
// its stage finishes; a payload is handed DOWNSTREAM, so it is alive
// from the phase that writes it through the last phase that reads it.
//
// These are the terms a weights-and-scratch accounting cannot see, and
// on a constrained box they are often the largest thing in the moment
// they exist: a video latent sits through both decodes, the decoded
// frame buffer is larger than the VAE that produced it, and a
// conditioning outlives the encoder that made it by the whole denoise.
std::vector<ResourceClaim>
payload_claims(std::string label, std::size_t bytes,
               std::string_view first_phase, std::string_view last_phase);

// Declared scratch for `phase`, or with an empty phase the widest single
// phase -- the same peak rule weights use. Kept separate from
// weight_footprint() rather than folded into it: a caller sizing a
// decode wants both, but a caller asking what a CHECKPOINT costs must
// not silently get an arena as well.
std::size_t scratch_footprint(const SessionContextIntf* session,
                              std::string_view          phase = {});


// What a VAE decode of `width` x `height` will allocate, estimated from
// `<root>/vae/config.json` alone -- no model load, so it is answerable
// during the planning phase.
//
// The formula is the VAE's own, selected by `_class_name`: the FLUX.2
// AutoencoderKL peaks at ~7 full-resolution base-channel buffers, the
// Qwen-Image one at an im2col scratch plus ~50% for the level's I/O.
// Both mirror the decode_peak_bytes() on the corresponding VAE class,
// and the runtime reclaim checks in generate-image use those directly --
// this is the plan-time estimate of the same quantity.
//
// An UNRECOGNISED VAE gets the larger of the two, deliberately: a decode
// arena that is under-declared reads as room that is not there, and the
// stage that believed it has already decided something it cannot undo.
// 0 only when there is no readable VAE config at all.
std::size_t vae_decode_scratch_bytes(const std::string& root,
                                     int width, int height);

// The same for a VIDEO decode, where the dominant transient is not the
// convolution arena but the OUTPUT: [3, frames, H, W] at bf16 plus the
// planar-U8 clip the stage buffers behind it, which is 9 bytes per
// output pixel and grows linearly with length.
//
// A BOUND, not the truth. The real figure depends on how many pixel
// frames the VAE expands the latent into, which is a property of the
// loaded model -- so vae-decode revises this to the exact number on
// every beat (GenerativeModelManager::revise_scratch). Declaring the
// config geometry first is what gives peers something to size against
// before the first clip exists.
std::size_t video_decode_scratch_bytes(int width, int height, int frames);

// Should a stage drop its weights between beats, given what this beat
// actually needs?
//
// Re-asked per beat, in BOTH directions, because the arena is a property
// of the beat: a run of image edits at mixed sizes has one large frame
// and several small ones, and a rule that could only tighten would make
// every small frame after the large one pay a reload it did not need.
// The decision is taken after a decode completes, so it is never on the
// critical path -- what it costs is the reload before the NEXT beat.
//
// The band is what stops that becoming churn. An arena sitting on the
// threshold would otherwise flip every beat, and a flip is not free:
// MiniMax-H3's video VAE is 10.4 GB, so a needless drop-and-reload is
// seconds of disk. So: unload when it genuinely does not fit, keep when
// it fits with room to spare, and HOLD the current answer in between.
// The band is proportional (bytes/8) rather than a constant, so it
// scales with whatever the stage is actually decoding.
//
// `current` is the answer in force, returned unchanged inside the band.
bool resolve_idle_unload(std::size_t ram, std::size_t peers,
                         std::size_t arena, bool current);

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
// Declarations (Stage::declare_resources) removed the largest source of
// that uncertainty -- peers this stage cannot see from its own config --
// but not all of it: CoreML models, LMDB, and activation scratch are
// still outside the manager's accounting.
inline constexpr std::size_t kStreamHeadroom = 8ull << 30;

// Total physical RAM in bytes, 0 when the query fails (callers treat 0 as
// "unknown" and keep the roomier behaviour rather than guessing small).
std::size_t phys_ram();

// Sum of the .safetensors bytes under `dir`, recursively. 0 if unreadable.
std::size_t dir_weights_bytes(const std::string& dir);

// The floor a checkpoint under `dir` can be reduced to if it streams its
// repeating unit: everything outside that unit, plus the two in-flight
// slots a streamer refills into.
//
// `stems` are the prefixes to try, WITHOUT the index -- "blocks." for a
// MiniMax-H3 DiT, "model.layers." for an LM -- and the first that
// matches anything wins. Trying several is safe because a stem that
// matches nothing yields 0, and a prefix match from position 0 cannot
// confuse "blocks." with "transformer_blocks.".
//
// Returns 0 when NO stem matches, which means nothing here streams and
// the checkpoint has no smaller form.
//
// A model with TWO stacks (an image DiT's double and single) is measured
// against BOTH: it streams both and keeps a slot pair for each, so the
// floor is the trunk plus a pair per stack. Stems are tried in order and
// the first prefix match wins, so a list holding one stem that is a
// prefix of another must put the longer first.
std::size_t streaming_floor_bytes(const std::string&                   dir,
                                  const std::vector<std::string_view>& stems,
                                  std::string_view exclude = {});

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
//
// `phase` is the caller's own phase, when it has one. Naming it drops
// the peers that will not be resident while this stage runs; leaving it
// empty asks for the box-level peak instead (persistent plus the widest
// single phase), which is what a stage that does not know when it runs
// has to assume. Either way `dirs` are always added -- they are what
// this caller is about to hold, whatever anybody else does.
std::size_t weight_footprint(const SessionContextIntf*        session,
                             const std::vector<std::string>&  dirs,
                             std::string_view                 phase = {});

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
  // LM LAYER pinning ONLY. Every DiT that used this is retired: the five
  // block-streaming families grow a resident set by measuring instead
  // (BlockResidency), which is what a fraction of total RAM decided
  // before the run could never do.
  //
  // The language models are not retired because they have nothing to
  // grow into -- metal-qwen-model and metal-gemma-model pin a prefix of
  // layers and have no residency policy behind it, so removing this
  // would leave the text encoders streaming everything with no way back.
  // Give them BlockResidency and this field goes too.
  double      pin_frac  = 0.0;
  std::size_t footprint = 0;   // DiT + encoder + everything resident
  std::size_t others    = 0;   // the same, minus the DiT's own bytes
  std::size_t retires   = 0;   // what `dit_retires` took off the footprint
  std::size_t transient = 0;   // peers that let go before this model runs
};

// `dit_retires` is what the DiT will RELEASE after load and before the
// denoise -- weights it reads once and then drops, so they are never
// part of the set that has to coexist. MiniMax-H3's AdaLN bake is the
// case this exists for: it retires every `adaln_proj` projection, 24.3 GB
// of a 61.7 GB bf16 checkpoint (39% of it), and without this the
// irreversible streaming decision is taken against a model 39% of which
// is about to stop existing.
//
// Pass it ONLY when the release is certain. Being wrong in this
// direction is the expensive one: a model that declines to stream and
// then holds more than predicted thrashes, and nothing later can undo
// the decision. Zero -- the default -- is always safe.
StreamPlan plan_streaming(const SessionContextIntf* session,
                          const std::string&        dit_dir,
                          const std::string&        enc_dir,
                          std::size_t               headroom,
                          std::size_t               dit_retires = 0);

// The `unload_when_idle` config value shared by the model-holding stages.
// THREE things can happen to idle weights, not two:
//
//   destroy -- free them. The bytes are gone, the accounting shrinks, and
//              the next beat pays a full reload from disk. The only one
//              that reliably gives a peer room RIGHT NOW.
//   park    -- hand them to the kernel as purgeable. They survive unless
//              something else needs the RAM, and the next beat reactivates
//              without touching the disk if they did. Never worse than
//              destroy: same reclaim value, cheaper when the pages live.
//   keep    -- hold them pinned. A second run is free, but nothing can
//              reclaim them, so under pressure the OS compresses and
//              swaps them instead -- paid in both directions, for weights
//              that are read-only and could have been re-read from a file.
//   auto    -- resolve from the box (see the stages' resolve_*_policy_).
//
// `park` is the state this vocabulary existed to name and could not: the
// old spelling had only always/never, so a stage that wanted "let go IF
// somebody needs it" had to choose between two wrong answers. The legacy
// names are kept as aliases so existing pipeline JSON keeps working.
enum class UnloadPolicy {
  kAuto,
  kDestroy,
  kPark,
  kKeep,
  kAlways = kDestroy,        // legacy spelling
  kNever  = kKeep,           // legacy spelling
};

// True when any model in this run decided to stream its blocks -- i.e.
// somebody is already paying per-step disk reads for want of RAM.
//
// The signal a stage should weigh before keeping anything resident, and
// it does NOT come from arithmetic: it is recorded by plan_streaming()
// when a DiT actually takes that decision. See
// GenerativeModelManager::note_streaming for why the footprint numbers
// alone cannot see it (a streaming DiT revises its declaration down, and
// the revision reads as roominess).
bool peer_streams(const SessionContextIntf* session);

// Parse a config string; unknown values return kAuto and set `*bad` so the
// caller can warn (stage config is deferred-validated -- never throw).
UnloadPolicy parse_unload_policy(const std::string& s, bool* bad = nullptr);

const char* unload_policy_name(UnloadPolicy p);

}  // namespace model_memory
}  // namespace vpipe

#endif

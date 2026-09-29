#ifndef VPIPE_GENERATIVE_MODELS_CONDITIONER_PROFILE_H
#define VPIPE_GENERATIVE_MODELS_CONDITIONER_PROFILE_H

// How a MODEL FAMILY wants its prompt conditioned, as an open bag.
//
// This is the counterpart to shared/accel-settings.h, drawn for the same
// reason and in the same shape. There, the host tells a family what the
// graph asked for; here, a family tells the host what its checkpoint
// needs. Both are FlexData rather than a struct, and the argument is
// identical: a typed field puts one side's vocabulary into the other's
// ABI, so adding a knob invalidates every plugin binary including the
// ones that do not use it.
//
// WHAT THIS IS NOT. It is not a way to bring your own conditioner. The
// MACHINERY stays in `diffusion-conditioner` -- the tokenizer, the
// Qwen3-VL tower, the deepstack injection, the embedding muxer, the
// mROPE grid -- because that machinery is large, shared by six families,
// and reaching it would mean exporting half the generative-model stack
// through the plugin SDK. What a profile carries is the FACTS that
// differ between checkpoints: a weight prefix, a template, how many
// leading tokens to drop, where the hidden state is tapped. A family
// whose conditioning needs machinery that is not here ships its own
// stage instead, the way LTX-2.5 does.
//
// WHAT A PROFILE CANNOT DO, and this one is a policy rather than a
// limitation: it cannot turn a mandatory content screen off. Whether a
// family's prompts are screened is decided by the host from the family
// tag, never from a key in here. A profile that could omit the screen
// would be a bypass, which is exactly what the screen exists to prevent.
//
// A KEY THIS HOST DOES NOT KNOW COSTS NOTHING and is ignored, which is
// what makes the bag open. A key it DOES know and cannot honour is
// WARNED -- see the checks in diffusion-conditioner-stage.cc. The two
// are not the same failure: the first is a newer plugin talking to an
// older host, which is expected; the second is a family being
// conditioned differently than it asked, which is silent everywhere
// else.
//
// Registration is by FAMILY TAG -- the same string the model registry's
// `tag()` returns and the same one a model_config beat carries in
// `model_family`, so one name identifies a family everywhere. First
// wins, so a profile cannot be replaced by a later plugin.

#include "generative-models/family-profile.h"

#include <string_view>

#include "common/vpipe-api.h"

VPIPE_API_BEGIN

namespace vpipe::genai::cond {

// The domain name to register under. See family-profile.h: one method
// takes a domain string, so a new subsystem needs no new entry point.
inline constexpr std::string_view kDomain = "conditioning";

// ---- the key vocabulary ---------------------------------------------
//
// ADD, NEVER RENAME OR REPURPOSE. A key's meaning is a promise to every
// binary already built against it; if a setting's meaning changes it
// gets a NEW key and the old one keeps meaning what it always did. That
// is what lets this file grow without the ABI version moving.

// Which of the host's encoder paths runs. Unset means "qwen3-vl" (the
// Qwen3-VL tower plus its language model), which is what every image
// family here uses and is the ONLY value the profile path drives today.
// The others -- "qwen3-dense", "qwen2.5-vl", "umt5" -- name the paths
// the host has for its built-ins, and asking for one is WARNED rather
// than silently ignored. A family needing an encoder that is not here
// ships its own conditioner stage.
inline constexpr std::string_view kEncoderKind = "encoder_kind";
// The subdirectory of the model root the encoder lives in. Unset means
// "text_encoder"; Boogu-Image says "mllm".
inline constexpr std::string_view kEncoderSubdir = "encoder_subdir";
// The checkpoint prefix the language model's tensors carry, e.g.
// "model.language_model." for a Qwen3VLForConditionalGeneration wrapper
// against a bare "language_model.". Unset means none.
inline constexpr std::string_view kWeightPrefix = "weight_prefix";
// ...and the vision tower's, e.g. "model.visual." against "visual.".
inline constexpr std::string_view kVisionPrefix = "vision_prefix";
// Load the language model WITHOUT its output head. True is the norm for
// a conditioning encoder, which never decodes. A family whose screen or
// captioner GENERATES on the same weights says false, and then pays for
// the head once instead of keeping a second embedding table.
inline constexpr std::string_view kBackboneOnly = "backbone_only";
// Page-pool ceiling for the encoder's context, in tokens. Only a CAP:
// pages are allocated lazily. A family whose screen runs a long system
// policy needs a bigger one than its prompts imply.
inline constexpr std::string_view kMaxSeq = "max_seq";

// Where the conditioning is tapped. "last", the final hidden state, is
// the default and the only value the profile path implements; "multi"
// names what FLUX.2 and Krea-2 do with `hidden_tap_layers`, and asking
// for it is warned. Getting this wrong conditions a DiT on a tensor
// several layers from the one it was trained against, which nothing
// downstream can detect.
inline constexpr std::string_view kHiddenTap = "hidden_tap";
inline constexpr std::string_view kHiddenTapLayers = "hidden_tap_layers";

// The chat template around the prompt, and how many leading tokens the
// family drops from the encoded sequence. Two sets: the text-only one
// and the one used when a reference image is wired.
inline constexpr std::string_view kPromptPrefix = "prompt_prefix";
inline constexpr std::string_view kPromptSuffix = "prompt_suffix";
inline constexpr std::string_view kDropPrefix   = "drop_prefix";
inline constexpr std::string_view kEditPrefix   = "edit_prefix";
inline constexpr std::string_view kEditSuffix   = "edit_suffix";
inline constexpr std::string_view kEditDrop     = "edit_drop_prefix";
// The per-reference label inside the edit template: Mage-Flow renders
// "Image N: ", Qwen-Image-Edit "Picture N: ", Boogu-Image nothing.
inline constexpr std::string_view kRefLabel = "ref_label";

// "sequential" -- plain arange positions, which a model that overrides
// position_ids needs -- or "mrope2d", the 2-D grid the stock Qwen3-VL
// text model builds. Sequential is the default and the only value the
// profile path implements; "mrope2d" is warned. Getting this wrong
// conditions the DiT on the right tokens at the wrong places, which
// looks like a slightly worse model rather than an error.
inline constexpr std::string_view kPositions = "positions";

// The grounded encode's preprocessing, when a reference image rides the
// vision tower. Pixels, not tokens.
inline constexpr std::string_view kGroundedLongEdge  = "grounded_long_edge";
inline constexpr std::string_view kGroundedMinPixels = "grounded_min_pixels";
inline constexpr std::string_view kGroundedMaxPixels = "grounded_max_pixels";

// ---- readers ---------------------------------------------------------
//
// The generic ones, named here so a reader of this file does not have to
// know where they live. See family-profile.h for what they promise.
using profile::flag;
using profile::has;
using profile::integer;
using profile::integers;
using profile::strings;
using profile::text;

// This domain's profile for `family`, or null when no plugin registered
// one -- which is every built-in family, and is the case the host's own
// per-family defaults serve.
inline const FlexData*
find(std::string_view family) noexcept
{
  return profile::Registry::get().find(kDomain, family);
}

}  // namespace vpipe::genai::cond

VPIPE_API_END

#endif

#ifndef VPIPE_GENERATIVE_MODELS_QUANTIZE_PROFILE_H
#define VPIPE_GENERATIVE_MODELS_QUANTIZE_PROFILE_H

// What `model-quantize` needs to know about a family's DiT, as an open
// bag. See family-profile.h for the shape and the argument for it.
//
// This is the SECOND seam that stage has and it answers a different
// question from the first. `QuantizableFamily` (quantize-family-registry.h)
// describes a REPACK: where each component lives, which file to pick,
// what its metadata key is. This describes a diffusers-layout DiT: what
// its transformer config calls itself, and which tensor leaves are
// matrices worth quantizing.
//
// A family may want either, both, or neither. They are not alternatives:
// a checkpoint published in both layouts needs both, and its leaf set is
// the same either way because leaf names are a property of the weights.
//
// WHY THIS IS DATA AND NOT A CALLBACK. Everything that differs between
// DiTs here is a LIST OF NAMES. The matching rule (last dot-component
// before `.weight`), the group sizing, the mixed-precision ranking, the
// writer and the config it stamps are the same work for every family --
// and a family that could override them could produce a checkpoint that
// loads, runs, and generates the wrong thing. Naming leaves is the part
// only the family knows; the rest is the stage's and should stay there.

#include "generative-models/family-profile.h"

#include <string_view>

#include "common/vpipe-api.h"

VPIPE_API_BEGIN

namespace vpipe::genai::quant {

inline constexpr std::string_view kDomain = "quantize";

// ---- the key vocabulary ---------------------------------------------
//
// ADD, NEVER RENAME OR REPURPOSE.

// The `_class_name` this family's transformer/config.json carries, e.g.
// "AcmeTransformer2DModel". This is what lets the stage RECOGNISE a
// diffusers
// checkpoint whose family it does not implement -- the built-in chain is
// a closed switch over class names, and without this an out-of-tree
// family falls through it and is quantized with someone else's leaf set
// or not at all.
//
// It must be EXACT and it must be yours. Claiming a class name another
// family owns quantizes their weights with your scope, which produces a
// plausible checkpoint that is wrong.
inline constexpr std::string_view kDitClassName = "dit_class_name";

// The tensor leaves to quantize, as an array of strings.
//
// A leaf is the LAST dot-component before `.weight`, which is what the
// matcher compares -- so `attn.to_out.0` is "0" and
// `img_mlp.net.0.proj` is "proj". That makes leaves SHORT and prone to
// collide, which is what `quant_exclude` is for.
//
// What to leave out is the harder half and it is the same everywhere:
// the patch and context embedders, the output projection and the
// timestep MLP are all small and all precision-sensitive -- every value
// enters or leaves through them -- so quantizing them dominates the
// error while saving almost nothing.
//
// Empty means "said nothing", never "quantize none": a family that
// omits this gets the host's own list for its tag, and a family the
// host does not know gets nothing quantized at all rather than
// everything.
inline constexpr std::string_view kQuantLinears = "quant_linears";

// Name substrings kept DENSE even when their leaf matched. Appended to
// the stage's own `quant_exclude`, never replacing it. This is how a
// family whose feed-forward `linear_1` collides with a timestep
// embedder's `linear_1` keeps the second one out.
inline constexpr std::string_view kQuantExclude = "quant_exclude";

// The adaLN modulation leaves, quantized only under the stage's
// `quant_modulation` opt-in and always at 8 bits whatever the body runs.
//
// Separate from `quant_linears` because the default answer differs: the
// modulation is what the residual scale rides on, so 4-bit wrecks it
// while 8-bit is fine, and it stays dense unless a caller asks. A family
// with nothing here simply has no opt-in.
inline constexpr std::string_view kModulationLinears = "modulation_linears";
// What to call them in the log line, e.g. "*_mod.1". Cosmetic, and worth
// having: "quantizing the AdaLN modulation" without saying which tensors
// is a sentence nobody can check.
inline constexpr std::string_view kModulationLabel = "modulation_label";

// ---- readers ---------------------------------------------------------
using profile::flag;
using profile::has;
using profile::integer;
using profile::strings;
using profile::text;

inline const FlexData*
find(std::string_view family) noexcept
{
  return profile::Registry::get().find(kDomain, family);
}

// The family whose profile claims this transformer `_class_name`, or
// empty. Linear over the registered families, which is a handful and is
// asked once per quantize run.
std::string family_for_class(std::string_view class_name);

}  // namespace vpipe::genai::quant

VPIPE_API_END

#endif

#ifndef VPIPE_GENERATIVE_MODELS_DETECT_PROFILE_H
#define VPIPE_GENERATIVE_MODELS_DETECT_PROFILE_H

// How a checkpoint of this family is RECOGNISED on disk, and what to
// call it. See family-profile.h for the shape and the argument for it.
//
// The host has two closed switches here and a plugin can extend
// neither. `model-detect` maps a transformer `_class_name` to a runtime
// model type and then that type to a family and version label; the
// catalogue maps the same type to input and output modalities. Between
// them they are what identifies a model the user already has on disk,
// as opposed to one they fetched from a catalogue row.
//
// The consequence today is visible: LTX-2.5 and SenseNova-U1.5 appear
// in neither, so a checkpoint of either sitting in a models directory
// cannot be named at all. It works anyway, because the catalogue row
// that fetched it recorded the type -- but a directory that arrived any
// other way is unidentifiable, and the failure is silent.
//
// WHAT THIS IS FOR AND WHAT IT IS NOT. It answers "what is this
// directory", which is a labelling question. It does NOT decide what
// loads it: that is `claims()` on the model registries, which reads the
// checkpoint rather than a table and is the thing a wrong answer
// actually costs. A family should register both and they should agree;
// where they disagree, the registry wins, because it looked.

#include "generative-models/family-profile.h"

#include <string>
#include <string_view>
#include <vector>

#include "common/vpipe-api.h"

VPIPE_API_BEGIN

namespace vpipe::genai::detect {

inline constexpr std::string_view kDomain = "detect";

// ---- the key vocabulary ---------------------------------------------
//
// ADD, NEVER RENAME OR REPURPOSE.

// The transformer `_class_name` values this family owns, as an array.
// An array rather than one string because a family may ship several
// architectures under one label, and because the t2i and edit repos of
// a family commonly share a class.
inline constexpr std::string_view kClassNames = "class_names";

// The runtime model type to report, and the variant to report instead
// when the directory looks image-conditioned. Both are the strings a
// catalogue row's `model_type` carries, so one name identifies a family
// through the browser, the fetcher and the stages.
//
// `model_type_edit` is optional: a family whose two instantiations
// share a transformer config has no other signal, and a family with one
// instantiation should leave it unset rather than duplicate the first.
inline constexpr std::string_view kModelType     = "model_type";
inline constexpr std::string_view kModelTypeEdit = "model_type_edit";

// What to call it in a browser: "Mage-Flow" and "Edit". Cosmetic, and
// the reason the table exists at all -- a model type is an identifier,
// not a name a person reads.
inline constexpr std::string_view kLabelFamily  = "label_family";
inline constexpr std::string_view kLabelVersion = "label_version";
inline constexpr std::string_view kLabelVersionEdit = "label_version_edit";

// Input and output modalities, each a subset of
// {"text","image","audio","video"}. The catalogue derives these for a
// DETECTED model, where a catalogued one carries its own.
inline constexpr std::string_view kInputs  = "inputs";
inline constexpr std::string_view kOutputs = "outputs";

// ---- readers ---------------------------------------------------------
using profile::has;
using profile::strings;
using profile::text;

inline const FlexData*
find(std::string_view family) noexcept
{
  return profile::Registry::get().find(kDomain, family);
}

// The model type a registered family reports for this transformer
// `_class_name`, or empty when none claims it. `edit` selects the
// image-conditioned variant where a family declared one.
std::string model_type_for_class(std::string_view class_name, bool edit);

// The profile whose `model_type` (or `model_type_edit`) is this one, or
// null. This is the reverse lookup the label and modality tables need:
// they are keyed by model type, and a profile is keyed by family tag.
const FlexData* for_model_type(std::string_view model_type);

}  // namespace vpipe::genai::detect

VPIPE_API_END

#endif

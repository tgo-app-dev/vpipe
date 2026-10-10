#ifndef VPIPE_GENERATIVE_MODELS_GEN_INPUT_H
#define VPIPE_GENERATIVE_MODELS_GEN_INPUT_H

// AN OPTIONAL NAMED INPUT to a generation, looked up by name instead of
// carried as a struct field.
//
// This exists because the request structs were growing, and growing in
// a way that costs everyone. VideoGenRequest wears the history plainly:
// `cond`, then `neg`, then `audio_cond`, then `ref`, then `ref_last`,
// then `ref_video_rows` and `ref_audio_rows`. Every one of those arrived
// with a capability, changed the struct's layout, and invalidated every
// family binary -- including the families that ignore it. It is the
// same shape as the acceleration settings before they became a bag, and
// it has the same answer.
//
// So: the typed fields that exist STAY, because they work and they are
// documented. What changes is where the NEXT input goes. A mask, a
// control image, a depth map, a third reference, a per-region prompt --
// each of those is a name and a tensor, and none of them needs a field.
//
// WHY A CALLBACK AND NOT A FLEXDATA. A FlexData cannot hold a
// `const float*` to a tensor the stage already owns without either
// copying it or smuggling a pointer through an integer. The lookup is
// one member whose layout never moves, and it borrows exactly the way
// the typed pointers beside it do.
//
// Scalars are a different question and have a different answer: the
// `borrowed_extra` FlexData beside this one. Use that for a number or a
// flag, and this for anything with a shape.

#include "common/flex-data.h"

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "common/vpipe-api.h"

VPIPE_API_BEGIN

namespace vpipe::genai {

// ---- THE CONDITIONING'S SIDEBAND -------------------------------------
//
// What a conditioner says about the conditioning beside its rows, and
// what reaches a family as VideoGenRequest / ImageGenRequest
// `cond_sideband`. Keys the host's own conditioners write; a family
// reads the ones it understands and ignores the rest, and a plugin
// conditioner feeding its own family may add keys of its own. The
// content screen's verdict rides here too (sideband::kContentBlocked,
// common/beat-keys.h).
namespace cond_sideband {

// Per vision slot of the prompt, 1 where a REFERENCE IMAGE's embedding
// sits. Int array. A second prompt with a different reference count has
// a different mask, so a family reads it fresh every beat.
inline constexpr std::string_view kImgSlots = "img_slots";
// Per reference, its LATENT grid in tokens. Int arrays, one per axis.
inline constexpr std::string_view kRefGridH = "ref_grid_h";
inline constexpr std::string_view kRefGridW = "ref_grid_w";

// The REFERENCE PLAN a reference encoder hands a video family: one
// object per reference, in the order their rows follow the caption's.
inline constexpr std::string_view kReferences = "references";
// Per reference: what it is. Values kImage (default), kVideo, kAudio.
inline constexpr std::string_view kKind  = "kind";
inline constexpr std::string_view kImage = "image";
inline constexpr std::string_view kVideo = "video";
inline constexpr std::string_view kAudio = "audio";
// Per reference: its latent extent. Ints.
inline constexpr std::string_view kLatentFrames = "latent_frames";
inline constexpr std::string_view kLatentHeight = "latent_height";
inline constexpr std::string_view kLatentWidth  = "latent_width";
inline constexpr std::string_view kAudioLatents = "audio_latents";
// Per caption token, the tag of what it belongs to. Int array.
inline constexpr std::string_view kTokenTags = "token_tags";

}  // namespace cond_sideband

// ---- A NAMED TENSOR'S DATA FORMAT ----------------------------------
//
// Carried in the tensor's `extra` bag, as the element type's NAME, so a
// format this header has not heard of yet costs a new value rather than
// a new field. `elem_size` stays beside it on purpose: a consumer that
// does not recognise a newer name can still size, copy and forward the
// bytes, and says "unsupported" rather than misreading them.
//
// ABSENT means f32 -- the only format a tensor could be before the key
// existed without saying so -- which is why a 2-byte tensor with no
// dtype is refused rather than guessed at: reading bf16 as f16 gives
// values of roughly the right magnitude and entirely the wrong content.
namespace named_tensor {

inline constexpr std::string_view kDtype = "dtype";

// Values of kDtype. Names are the host's, added and never repurposed.
inline constexpr std::string_view kF32     = "f32";
inline constexpr std::string_view kF16     = "f16";
inline constexpr std::string_view kBf16    = "bf16";
inline constexpr std::string_view kI32     = "i32";
inline constexpr std::string_view kI16     = "i16";
inline constexpr std::string_view kI8      = "i8";
inline constexpr std::string_view kU8      = "u8";
inline constexpr std::string_view kFp8E4m3 = "fp8_e4m3";
inline constexpr std::string_view kFp8E5m2 = "fp8_e5m2";

// Bytes per element of a known dtype, 0 for a name this header does not
// know.
inline int
dtype_bytes(std::string_view dtype) noexcept
{
  if (dtype == kF32 || dtype == kI32) { return 4; }
  if (dtype == kF16 || dtype == kBf16 || dtype == kI16) { return 2; }
  if (dtype == kI8 || dtype == kU8 || dtype == kFp8E4m3 ||
      dtype == kFp8E5m2) {
    return 1;
  }
  return 0;
}

}  // namespace named_tensor

// A tensor the stage is lending for the duration of the call. BORROWED,
// like every other pointer on a request: valid until the generate call
// returns, and owned by a beat the stage is holding.
struct NamedTensor {
  const void*      data = nullptr;
  // Row-major over these extents. Empty means the input exists and the
  // stage could not describe its shape, which a family should treat as
  // a refusal rather than guess at.
  std::vector<int> shape;
  // Bytes per element, whatever the format -- enough to size and copy a
  // tensor whose dtype the reader does not know. Set it with dtype()
  // below rather than by hand, so the two cannot disagree.
  int              elem_size = 4;
  // The DATA FORMAT (named_tensor::kDtype) and anything else this struct
  // has no field for: see common/flex-bag.h.
  FlexData         extra;

  std::size_t elems() const noexcept
  {
    if (data == nullptr || shape.empty()) { return 0; }
    std::size_t n = 1;
    for (const int d : shape) {
      if (d <= 0) { return 0; }
      n *= (std::size_t)d;
    }
    return n;
  }

  // The element type's name: the `dtype` key when present, f32 when it
  // is absent and the elements are 4 bytes, and EMPTY otherwise -- a
  // 2-byte tensor that does not say which 2-byte format it is.
  std::string dtype() const
  {
    if (extra.is_object() && extra.as_object().contains(named_tensor::kDtype)) {
      return std::string(
          extra.as_object().at(named_tensor::kDtype).as_string(""));
    }
    return elem_size == 4 ? std::string(named_tensor::kF32) : std::string();
  }

  // Say what the elements are, keeping `elem_size` in step. False, and
  // nothing changed, for a name this header does not know the width of
  // -- set extra[kDtype] and elem_size yourself for one of those.
  bool set_dtype(std::string_view name)
  {
    const int b = named_tensor::dtype_bytes(name);
    if (b == 0) { return false; }
    if (!extra.is_object()) { extra = FlexData::make_object(); }
    extra.as_object().insert_or_assign(named_tensor::kDtype,
                                       FlexData::make_string(name));
    elem_size = b;
    return true;
  }

  bool is(std::string_view name) const { return dtype() == name; }
};

// Look one up. False means the graph wired nothing under that name,
// which is the answer for every name on a graph that predates it -- so
// a family MUST work when this returns false for everything.
//
// NAMES ARE THE HOST'S and are lower_snake. They are added, never
// renamed or repurposed: a name is a promise to every binary already
// asking for it. The set is empty today, which is the honest state of
// it -- this is the seam, not a backlog.
using NamedInputFn =
    std::function<bool(std::string_view name, NamedTensor* out)>;

// ---- and its mirror: A NAMED OUTPUT, handed back MID-GENERATION ------
//
// The result struct is what a generation leaves behind; this is what it
// shows on the way. A live preview is the first -- the model's clean
// estimate at a step, which the host decodes and streams -- and the
// same argument as the input seam says it should not be the last to cost
// a field: an attention map, a per-step audio latent, a debug tap are
// each a name and a tensor.
//
// TWO CALLS, ASKED BEFORE BUILT. `wanted(name, step, total)` first, and
// only on a yes does the family build the tensor and hand it to
// `output(name, step, total, t)`. Building a video's clean estimate
// means unpacking a latent the size of the clip, and a family that did
// it every step for a host that previews every eighth would pay for the
// seven nobody asked for.
//
// Both are ALWAYS installed by the host (answering "no" and ignoring,
// when nothing downstream listens), so a family calls them without a
// null check. `step` is 1-based of `total`, as in `progress`. The tensor
// is BORROWED for the call: the host copies what it keeps.
//
// NAMES are the host's, lower_snake, added and never repurposed, exactly
// as for inputs. A family that cannot produce one never calls `output`
// with it, and nothing is waiting.
using OutputWantedFn =
    std::function<bool(std::string_view name, int step, int total)>;
using NamedOutputFn = std::function<void(
    std::string_view name, int step, int total, const NamedTensor& t)>;

// The model's CLEAN ESTIMATE at a step -- x0, not the noisy state --
// in the same latent space, layout and shape as the generation's result
// latent (VideoGenResult::video [z, T, h, w], ImageGenResult::latent),
// f32. The host decodes it with a tiny autoencoder trained for that
// space; see stages/latent-preview.h.
inline constexpr std::string_view kOutputPreviewX0 = "preview_x0";

// The model_config keys that turn previews ON -- which tiny autoencoder,
// how often, how large, which frames. A family's config source declares
// them under these names (the host's generate stages read them from the
// model_config beat), so every family is configured the same way.
namespace preview_key {
inline constexpr std::string_view kVae     = "preview_vae";
inline constexpr std::string_view kEvery   = "preview_every";
inline constexpr std::string_view kMaxEdge = "preview_max_edge";
inline constexpr std::string_view kFrames  = "preview_frames";
}  // namespace preview_key

// ---- WHAT A STAGE TELLS A FAMILY IT LOADS ----------------------------
//
// Keys of ImageModelCreateArgs / VideoModelCreateArgs `borrowed_extra`.
namespace create_args {

// The LABEL the stage booked this family's CoreML claim under (a family
// claims its ANE module with model_memory::coreml_claims; the stage
// keeps it only when the graph asked for `ane_ffn`, and relabels it to
// itself, since a family cannot know which stage it runs in and two
// stages sharing a label would share one grant). The stage has already
// read the grant: a refused tier arrives with `ane_ffn` off in the
// accel bag. A family revises what it actually holds through
// revise_scratch("coreml:" + label, bytes). Absent: the stage booked
// nothing for the family, which keeps whatever label it chose itself.
inline constexpr std::string_view kCoreMLLabel = "coreml_label";

}  // namespace create_args

}  // namespace vpipe::genai

VPIPE_API_END

#endif

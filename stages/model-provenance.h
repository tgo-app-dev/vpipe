#ifndef VPIPE_STAGES_MODEL_PROVENANCE_H
#define VPIPE_STAGES_MODEL_PROVENANCE_H

// What produced these pixels, and how a sink records it.
//
// Only the stage that ran the model knows which one it was, and the
// file is written several stages later. So a generator stamps
// `model_name` onto its output beat's sideband and everything between
// carries it forward:
//
//   generate-image -> vae-decode -> save-image        (EXIF Software)
//   generate-video -> vae-decode -> rgb-to-video ->
//                     save-video                      (a container tag)
//   generate-video -> audio-vae-decode -> save-video  (the same tag)
//
// Two things live here rather than at those call sites. The READ,
// because six stages do it and a sideband key spelled differently in
// one of them breaks SILENTLY -- the graph still runs and the file just
// comes out unmarked. And the STRING, because save-image and save-video
// have to record the same words for the same run: an image and a video
// out of one graph disagreeing about what made them is worse than
// neither being marked at all.
//
// Nothing here invents a name. A beat that reaches a sink without one
// did not come from a model here -- a plain load -> save transcode is
// the ordinary case -- and stamping it would be vpipe claiming
// authorship of someone else's footage.

#include "common/flex-data.h"
#include "vpipe/vpipe.h"
#include "common/beat-keys.h"

#include <string>
#include <utility>

namespace vpipe::provenance {

// The sideband key every stage in the chain agrees on.
inline constexpr std::string_view kSidebandKey = sideband::kModelName;

// The model as the USER named it ("local/MiniMax-H3-FL2VA-8bit"), or ""
// when the producer sent none.
inline std::string
model_name(const FlexData& sideband)
{
  if (!sideband.is_object()) { return {}; }
  FlexData sb = sideband;                 // as_object() is a view
  auto o = sb.as_object();
  if (!o.contains(kSidebandKey)) { return {}; }
  return std::string(o.at(kSidebandKey).as_string(""));
}

// Stamp it, leaving whatever else is on the sideband alone. An empty
// name writes nothing -- see the note about authorship above.
inline void
set_model_name(FlexData& sideband, const std::string& model)
{
  if (model.empty()) { return; }
  FlexData o = sideband.is_object() ? sideband : FlexData::make_object();
  o.as_object().insert_or_assign(kSidebandKey,
                                 FlexData::make_string(model));
  sideband = std::move(o);
}

// Carry it across a stage. Only this one key: the rest of a source
// beat's sideband describes the LATENT, not what the stage made of it.
inline void
carry_model_name(const FlexData& src, FlexData& dst)
{
  set_model_name(dst, model_name(src));
}

// The model as a file may name it. A local ABSOLUTE path -- a host that
// resolved the model itself passes one -- is cut to its last two
// components, <org>/<repo> in the models layout: a file someone shares
// must not carry their home directory. Any other name is kept as given.
inline std::string
public_model_name(const std::string& model)
{
  if (model.empty() || model.front() != '/') { return model; }
  std::string_view s = model;
  while (s.size() > 1 && s.back() == '/') { s.remove_suffix(1); }
  const std::size_t last = s.rfind('/');
  if (last == 0) { return std::string(s.substr(1)); }
  const std::size_t prev = s.rfind('/', last - 1);
  return std::string(s.substr(prev + 1));
}

// What a sink records, in the one wording both of them use:
//
//   "Vpipe 0.1 d950473 with local/MiniMax-H3-FL2VA-8bit"
//
// The build hash is in it because the version number alone does not
// identify a build -- most clips come off a tree between releases, and
// the hash is what makes a file reproducible.
inline std::string
software_string(const std::string& model)
{
  std::string s = "Vpipe ";
  s += vpipe_version_number();
  s += " ";
  s += vpipe_build_hash();
  if (!model.empty()) {
    s += " with ";
    s += public_model_name(model);
  }
  return s;
}

}  // namespace vpipe::provenance

#endif

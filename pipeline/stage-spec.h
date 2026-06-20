#ifndef STAGE_SPEC_H
#define STAGE_SPEC_H

#include "pipeline/stage-config.h"

#include <span>
#include <string_view>
#include <typeinfo>

namespace vpipe {

// Coarse functional grouping of a stage, used to organize tooling (the
// web-ui toolbox) and for at-a-glance classification. A stage with no
// declared category is Generic.
enum class StageCategory : unsigned char {
  Generic, Audio, Video, Text, Control, Database, Network
};

// Stable lower-case name ("generic", "audio", "video", "text",
// "control", "database", "network"). Never returns null.
std::string_view stage_category_name(StageCategory) noexcept;

// Static, type-level declaration of one stage port (input or output).
// `type` is the concrete beat payload type carried on the port
// (&typeid(ConcretePayload)); nullptr means "untyped / any", which the
// runtime and the composer treat as compatible with everything. Like
// ConfigKey, all string_view members must point at static storage --
// the owning StageSpec outlives every stage instance.
struct PortSpec {
  std::string_view      name;            // short label, e.g. "frames"
  std::string_view      doc;             // one-line description
  const std::type_info* type        = nullptr;
  unsigned              clock_group  = 0;
};

// One formal description of a stage type: its human description,
// category, declared input/output ports, and configuration attributes
// (reusing ConfigKey, whose def_* fields are the single source of truth
// for attribute defaults). A stage exposes it by returning a reference
// to a file-static `kSpec` from Stage::spec(); the same object is
// registered with the StageRegistry (VPIPE_REGISTER_SPEC) so tooling can
// enumerate it without constructing an instance.
//
// `kSpec`, `kIports`, `kOports` are file-static `const` (not constexpr:
// PortSpec.type uses &typeid which is not a constant expression);
// `attrs` may point at a `constexpr ConfigKey[]` table. The base default
// (Stage::spec) returns an empty Generic spec.
struct StageSpec {
  std::string_view           type_name;
  std::string_view           doc;
  // Optional human-friendly label for tooling (the web-ui toolbox shows
  // it in place of `type_name` when set). Empty -> fall back to
  // type_name. Like the other string_views, must point at static
  // storage. Declared right after `doc` so existing designated
  // initializers (which skip it) stay valid.
  std::string_view           display_name;
  StageCategory              category = StageCategory::Generic;
  std::span<const PortSpec>  iports;
  std::span<const PortSpec>  oports;
  std::span<const ConfigKey> attrs;
  // When true, tooling (the web-ui composer toolbox) omits this stage
  // from the palette of stages a user can add. The spec is still
  // registered + returned by the stage-types API so an already-present
  // instance (e.g. loaded from a saved pipeline) still renders with its
  // ports/docs -- this only hides it from the "add a new stage" list.
  // For structural / internal stages (call, passthrough).
  bool                       hidden = false;
};

}

#endif

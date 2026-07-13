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
  Generic, Audio, Visual, Vision, Generative, Text, Control, Database,
  Network, Preparation
};

// Stable lower-case name ("generic", "audio", "visual", "vision",
// "generative", "text", "control", "database", "network",
// "preparation"). Never returns null. "visual" is media I/O and frame
// transforms (image/video load/save, decode/encode, rgb, decimation);
// "vision" is perception/understanding (detection, tracking, overlay,
// visual QA); "generative" is model-driven synthesis (text-to-image /
// -speech, VAE codec, sampler/scheduler selection).
std::string_view stage_category_name(StageCategory) noexcept;

// Static, type-level declaration of one stage port (input or output).
// `type` is the concrete beat payload type carried on the port
// (&typeid(ConcretePayload)); nullptr means "untyped / any", which the
// runtime and the composer treat as compatible with everything. Like
// ConfigKey, all string_view members must point at static storage --
// the owning StageSpec outlives every stage instance.
//
// `tags` is an OPTIONAL finer-grained payload constraint layered on top
// of `type`: a comma-separated list of semantic tags with OR meaning --
// the port produces / accepts ANY of the listed tags. Two ports are
// tag-compatible iff either declares no tags OR their tag sets intersect
// (see port_tags_compatible). It distinguishes payloads that share one
// beat type but are not interchangeable -- e.g. rtsp-capture emits both
// its video and audio as EncodedSegment, so its video oport tags
// "video-encoder-segments" and video-to-rgb's iport requires the same,
// while the audio oport tags "audio-encoder-segments" and is refused.
// Empty (the default) keeps the legacy behaviour: type alone decides.
struct PortSpec {
  std::string_view      name;            // short label, e.g. "frames"
  std::string_view      doc;             // one-line description
  const std::type_info* type        = nullptr;
  std::string_view      tags;            // comma-separated; OR semantics
  unsigned              clock_group  = 0;
};

// True iff a producer output port advertising the tag list `produced`
// may feed a consumer input port advertising `accepted`. Both are
// comma-separated tag lists (surrounding whitespace ignored, empty
// entries skipped). An empty / all-blank list on EITHER side means "no
// tag constraint" and is compatible with anything; otherwise the two are
// compatible iff they share at least one tag (OR semantics). This is the
// deeper check the runtime and the web-ui composer apply on top of the
// beat-type match.
bool port_tags_compatible(std::string_view produced,
                          std::string_view accepted) noexcept;

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

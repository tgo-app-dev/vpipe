#ifndef STAGE_SPEC_H
#define STAGE_SPEC_H

#include "pipeline/stage-config.h"

#include <cstdint>
#include <span>
#include <string_view>
#include <typeinfo>

#include "common/vpipe-api.h"

VPIPE_API_BEGIN

namespace vpipe {

// Coarse functional grouping of a stage, used to organize tooling (the
// web-ui toolbox) and for at-a-glance classification. A stage with no
// declared category is Generic.
enum class StageCategory : unsigned char {
  Generic, Audio, Visual, Vision, Generative, Text, Control, Database,
  Network, Preparation, ModelSpecificConfig
};

// Stable lower-case name ("generic", "audio", "visual", "vision",
// "generative", "text", "control", "database", "network",
// "preparation", "model-specific-config"). Never returns null. "visual"
// is media I/O and frame transforms (image/video load/save,
// decode/encode, rgb, decimation); "vision" is perception/understanding
// (detection, tracking, overlay, visual QA); "generative" is
// model-driven synthesis (text-to-image / -speech, VAE codec,
// sampler/scheduler selection).
//
// "model-specific-config" is its own category rather than a corner of
// "generative" because these stages are the only ones whose
// APPLICABILITY is decided by the checkpoint: a `wan2-model-config` is
// meaningless in a MiniMax-H3 graph, where every generative stage is
// meaningful in both. Grouping them apart is what lets tooling offer the
// one that matches the resident family instead of a flat list where
// picking wrong looks like picking right. See stages/model-config-source.h.
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
  // Anything this struct has no field for. See SpecExtra.
  std::span<const SpecExtra> extra;
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

// ---- command channels -------------------------------------------------
//
// A stage may accept COMMANDS while its pipeline runs: named requests
// from outside the graph (the C++ API's StageHandle::command, the
// Python binding, a GUI view) that carry JSON arguments and, optionally,
// buffers in either direction. The spec DECLARES them, so a caller can
// discover them and the host can refuse a malformed one before the
// stage ever sees it. docs/STAGE-COMMANDS.md is the full contract; the
// stage side is pipeline/stage-command.h.

// One named buffer a command carries. The LAYOUT fields are patterns,
// checked by the host against the concrete DataBuffer:
//
//   type    accepted element types, comma-separated ("u8,f32"); empty =
//           any. "bytes" is an opaque byte string -- any one-byte
//           element type, rank 1, contiguous -- and `format` says what
//           it encodes.
//   shape   accepted shapes: alternatives separated by '|', dimensions
//           by ','. A dimension is an integer (exactly that size), a
//           NAME (any size, but the SAME size wherever that name
//           appears in this command's buffers of the same direction),
//           '?' (any size) or '...' (any number of dimensions, at most
//           once per alternative). Empty = any shape. "1,H,W|H,W" takes
//           a mask with or without its channel axis; "H,W,3" packed RGB.
//   format  for "bytes": the media type ("image/png"). A buffer that
//           names no format takes this one; one that names another is
//           refused.
struct BufferSpec {
  std::string_view name;
  std::string_view doc;
  std::string_view type;
  std::string_view shape;
  std::string_view format;
  // IN: the caller may omit it. OUT: the stage may omit it.
  bool             optional   = false;
  // OUT only: the caller may WRITE the stage's bytes before closing.
  bool             writable   = false;
  // The buffer must be C-contiguous (no padded or permuted strides).
  bool             contiguous = false;
  // Anything this struct has no field for. See SpecExtra.
  std::span<const SpecExtra> extra;
};

// One command a stage accepts.
struct CommandSpec {
  std::string_view            name;
  std::string_view            doc;
  // The JSON arguments, as ConfigKeys: type, required and doc are
  // enforced/used exactly as for configuration. An argument not listed
  // here is refused, so a typo is an error rather than a default.
  std::span<const ConfigKey>  args;
  // The keys of the JSON result, for discovery. Not enforced.
  std::span<const ConfigKey>  results;
  std::span<const BufferSpec> in;       // caller -> stage
  std::span<const BufferSpec> out;      // stage -> caller, in place
  // The stage HOLDS its output while this command's reply is exposed:
  // everything downstream waits until the caller closes it. A promise
  // to the caller about the pipeline, stated so it can be discovered.
  bool                        holds = false;
  // Anything this struct has no field for. See SpecExtra.
  std::span<const SpecExtra>  extra;
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
  // A plugin's spec is PLUGIN memory the host reads, so it says how much
  // of it there is (common/abi-struct.h). Leave it defaulted.
  std::uint32_t              struct_size = sizeof(StageSpec);
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
  // Anything this struct has no field for. See SpecExtra.
  std::span<const SpecExtra> extra;
  // When true, tooling (the web-ui composer toolbox) omits this stage
  // from the palette of stages a user can add. The spec is still
  // registered + returned by the stage-types API so an already-present
  // instance (e.g. loaded from a saved pipeline) still renders with its
  // ports/docs -- this only hides it from the "add a new stage" list.
  // For structural / internal stages (call, passthrough).
  bool                       hidden = false;
  // The commands this stage accepts while running. Empty = none, and
  // then no inbox is created for it. LAST so every existing designated
  // initializer stays valid.
  std::span<const CommandSpec> commands;
};

// The command named `name` in `spec`, or null.
const CommandSpec* find_command(const StageSpec& spec,
                                std::string_view name) noexcept;

// spec.commands, read only when the spec -- plugin memory -- is new enough
// to have the field (common/abi-struct.h). Host code reads it through this.
std::span<const CommandSpec> spec_commands(const StageSpec& spec) noexcept;

}

VPIPE_API_END

#endif

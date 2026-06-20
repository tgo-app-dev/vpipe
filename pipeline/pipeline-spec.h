#ifndef PIPELINE_SPEC_H
#define PIPELINE_SPEC_H

#include "common/flex-data.h"
#include <memory>

namespace vpipe {

class Pipeline;
class SessionContextIntf;

// Pipeline serialization.
//
// The on-wire form is a FlexData document with the schema:
//
//   {
//     "id":     "<pipeline id>",
//     "stages": [
//       {
//         "id":     "<stage id>",
//         "type":   "<registered stage type name>",
//         "iports": [ { "src": "<stage id>", "oport": <int> }, ... ],
//         "config": { ... }   // optional; default {}
//       }, ...
//     ],
//     "subpipelines": [ <pipeline spec>, ... ]
//   }
//
// The `iports` array is positional: entry `i` maps to the new
// stage's iport `i`. Each `iports.src` must reference a stage
// declared earlier in the same pipeline's `stages` array (forward
// references are rejected). Cross-pipeline edges are not modelled;
// each pipeline is a closed unit.
//
// FlexData itself round-trips through both JSON and a binary
// format; choosing between them at the file boundary is the
// caller's responsibility (see Session::load_pipeline /
// store_pipeline for the convention).

// Serialise the pipeline (and any nested sub-pipelines) into a
// FlexData spec. Stages whose vertex is not a Stage (shouldn't
// happen in well-formed pipelines, but the cast is checked) are
// skipped silently.
FlexData pipeline_to_spec(const Pipeline& pl);

// Build a fresh Pipeline from a spec FlexData document. Returns
// nullptr on any structural / validation error; in that case the
// failure is reported through `session->warn(...)`. The returned
// pipeline is detached -- the caller is responsible for handing it
// to a PipelineHandleImpl (Session::load_pipeline) so it gets
// owned and exposed through the public handle API.
std::unique_ptr<Pipeline>
pipeline_from_spec(const FlexData&           spec,
                   const SessionContextIntf* session);

}

#endif

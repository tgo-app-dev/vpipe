#ifndef CALL_STAGE_H
#define CALL_STAGE_H

#include "common/flex-data.h"
#include "common/vertex.h"
#include "pipeline/typed-stage.h"
#include <string>
#include <string_view>
#include <vector>

namespace vpipe {

class Graph;
class Job;
class RuntimeContext;

// "call" stage: invokes another pipeline as a function. The call-
// stage's iports are wired to the called pipeline's iports and
// the call-stage's oports are sourced from the called pipeline's
// oports. The call-stage itself is structural -- it never executes
// at runtime; the PipelineRuntime detects it during launch and
// inlines the called pipeline (rewriting edges to skip past both
// the call-stage and the called pipeline's iport/oport carrier
// vertices).
//
// Configuration:
//   {
//     "pipeline":   "<id>",   // required, the callee's id
//     "num_oports": <int>     // required, must equal callee.num_oports()
//   }
//
// The iport count is taken from the InEdge vector passed at
// construction. The constructor calls allocate_oports(num_oports)
// so the call-stage carries the right output-port shape; runtime
// validates that both counts match the resolved callee at launch.
//
// Visibility: the callee id is resolved with C++ lambda-style
// lexical scoping at launch time -- the call-stage's enclosing
// graph and every parent up the chain is searched in turn; the
// closest match wins.
class CallStage : public TypedStage<CallStage> {
public:
  static constexpr const char* kTypeName = "call";

  CallStage(const SessionContextIntf*,
            std::string         id,
            std::vector<InEdge> iports,
            FlexData            config);

  Job initialize(RuntimeContext&) override;
  Job process   (RuntimeContext&) override;
  Job drain     (RuntimeContext&) override;

  const StageSpec& spec() const noexcept override;

  const std::string& called_pipeline_id() const noexcept
  { return _pipeline_id; }

private:
  std::string _pipeline_id;
};

// Lexical-scope lookup of a pipeline by id, starting from `origin`.
// Search order, mirroring C++ block-scope name lookup:
//   1. origin's own children:    origin->graph(name)
//   2. origin's parent's graphs: origin->parent_graph()->graph(name)
//   3. ...up to the root
// The closest match wins (innermost shadowing). Returns nullptr
// when no enclosing scope provides a pipeline with the given id.
const Graph*
lexical_lookup_pipeline(const Graph* origin, std::string_view name);

}

#endif

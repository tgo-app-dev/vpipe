#ifndef PIPELINE_H
#define PIPELINE_H

#include "common/graph.h"
#include "pipeline/stage.h"
#include <string>
#include <string_view>
#include <vector>

namespace vpipe {

class Pipeline : public Graph {
public:
  Pipeline(std::string id, const SessionContextIntf*);
  ~Pipeline() override = default;

  // Typed insertion. Forwards to Graph::insert_vertex with a
  // unique_ptr<Vertex>(stage.release()), and returns the same
  // stage cast back to Stage*.
  Stage* insert_stage(StagePtr);

  // Convenience: look up `type_name` in the registry, construct the
  // stage with this Pipeline's session, and insert. Returns nullptr
  // if the type isn't registered. The trailing `config` is forwarded
  // to the registry-built stage; defaulted to an empty object.
  Stage* insert_stage(std::string_view    type_name,
                      std::string         id,
                      std::vector<InEdge> iports,
                      FlexData            config
                        = FlexData::make_object());

protected:
  // Defense-in-depth: every vertex inserted through Graph's API must
  // be a Stage. Asserts in debug builds; no-op in release.
  void on_insert_vertex(Vertex*) override;

  // A Stage may only be rewired while it is not running -- editing the
  // iports of a live stage would desync the frozen edge buffers built
  // at launch. Non-Stage vertices (if any) are unconstrained.
  bool can_move_iport(Vertex* v, unsigned iport) const override;

  // An iport move changed the stage's inputs; flag it so the next
  // launch is known to re-run initialize() against the new wiring.
  void on_iport_moved(Vertex* v, unsigned iport) override;
};

}

#endif

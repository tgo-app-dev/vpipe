#include "pipeline/pipeline.h"
#include "pipeline/stage-registry.h"

#include <cassert>
#include <memory>
#include <utility>

using namespace std;

namespace vpipe {

Pipeline::Pipeline(string id, const SessionContextIntf* s)
  : Graph(std::move(id), s)
{
}

Stage*
Pipeline::insert_stage(StagePtr stage)
{
  Stage* p = stage.get();
  auto v = unique_ptr<Vertex>(stage.release());
  Vertex* iv = insert_vertex(std::move(v));
  (void)iv;
  assert(iv == p);
  return p;
}

Stage*
Pipeline::insert_stage(string_view    type_name,
                       string         id,
                       vector<InEdge> iports,
                       FlexData       config)
{
  StagePtr s = StageRegistry::get().create(type_name, session(),
                                           std::move(id),
                                           std::move(iports),
                                           std::move(config));
  if (!s) {
    return nullptr;
  }
  return insert_stage(std::move(s));
}

void
Pipeline::on_insert_vertex(Vertex* v)
{
  assert(dynamic_cast<Stage*>(v) != nullptr);
  (void)v;
}

bool
Pipeline::can_move_iport(Vertex* v, unsigned /*iport*/) const
{
  if (auto* s = dynamic_cast<Stage*>(v)) {
    return !s->running();
  }
  return true;
}

void
Pipeline::on_iport_moved(Vertex* v, unsigned /*iport*/)
{
  if (auto* s = dynamic_cast<Stage*>(v)) {
    StageLifecycleAccess::set_needs_init(s, true);
  }
}

}

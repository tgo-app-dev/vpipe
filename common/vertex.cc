#include "common/vertex.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include <cassert>

using namespace std;

namespace vpipe {

Vertex::Vertex(const SessionContextIntf* s,
               string id,
               vector<InEdge> iports)
  : SessionMember(s)
  , _id(std::move(id))
  , _iports(std::move(iports))
  , _graph(nullptr)
  , _gvid(0)
{
}

const string
Vertex::id() const
{
  return _id;
}

const vector<InEdge>&
Vertex::iport_edges() const
{
  return _iports;
}

const unordered_set<OutEdge>&
Vertex::oport_edges(unsigned oport) const
{
  return _oports[oport];
}

unsigned
Vertex::num_iports() const
{
  return static_cast<unsigned>(_iports.size());
}

unsigned
Vertex::num_oports() const
{
  return static_cast<unsigned>(_oports.size());
}

void
Vertex::allocate_oports(unsigned n)
{
  _oports.resize(n);
  _oport_policies.resize(n);
}

void
Vertex::set_oport_policy(unsigned oport, OportPolicy pol)
{
  if (pol.capacity > OportPolicy::kMaxCapacity) {
    session()->warn(fmt(
        "Vertex({}): oport[{}] requested capacity {} exceeds hard "
        "cap {}; clamping",
        _id, oport, pol.capacity, OportPolicy::kMaxCapacity));
    pol.capacity = OportPolicy::kMaxCapacity;
  }
  _oport_policies.at(oport) = pol;
}

const OportPolicy&
Vertex::oport_policy(unsigned oport) const
{
  return _oport_policies.at(oport);
}

unsigned
Vertex::num_fanouts() const
{
  unsigned n = 0;
  for (const auto& port : _oports) {
    n += static_cast<unsigned>(port.size());
  }
  return n;
}

unsigned
Vertex::num_fanouts(unsigned p) const
{
  return static_cast<unsigned>(_oports[p].size());
}

Graph*
Vertex::graph() const
{
  return _graph;
}

void
Vertex::gvid(unsigned id)
{
  _gvid = id;
}

unsigned
Vertex::gvid() const
{
  return _gvid;
}

void
Vertex::commit_to_graph(Graph* g)
{
  for (unsigned i = 0; i < _iports.size(); ++i) {
    Vertex* src = _iports[i].v;
    unsigned src_p = _iports[i].p;
    if (!src) {
      continue;
    }
    if (g) {
      src->attach_fanout(src_p, OutEdge{this, i});
    } else {
      src->detach_fanout(src_p, OutEdge{this, i});
    }
  }
  _graph = g;
}

void
Vertex::set_iport_edge(unsigned iport, InEdge e)
{
  assert(iport < _iports.size());
  _iports[iport] = e;
}

void
Vertex::attach_fanout(unsigned oport, OutEdge e)
{
  assert(oport < _oports.size());
  _oports[oport].insert(e);
}

void
Vertex::detach_fanout(unsigned oport, OutEdge e)
{
  assert(oport < _oports.size());
  _oports[oport].erase(e);
}

void
VertexGraphAccess::commit_to_graph(Vertex* v, Graph* g)
{
  v->commit_to_graph(g);
}

void
VertexGraphAccess::gvid(Vertex* v, unsigned id)
{
  v->gvid(id);
}

unsigned
VertexGraphAccess::gvid(const Vertex* v)
{
  return v->gvid();
}

void
VertexGraphAccess::attach_fanout(Vertex* v, unsigned oport, OutEdge e)
{
  v->attach_fanout(oport, e);
}

void
VertexGraphAccess::detach_fanout(Vertex* v, unsigned oport, OutEdge e)
{
  v->detach_fanout(oport, e);
}

void
VertexGraphAccess::set_iport_edge(Vertex* v, unsigned iport, InEdge e)
{
  v->set_iport_edge(iport, e);
}

}


#include "common/graph.h"
#include "common/vertex.h"
#include <cassert>
#include <limits>

using namespace std;

namespace vpipe {

namespace {

bool
is_head_vertex(const Vertex* v)
{
  for (const auto& e : v->iport_edges()) {
    if (e.v) {
      return false;
    }
  }
  return true;
}

}

Graph::Graph(string id, const SessionContextIntf* s)
  : SessionMember(s)
  , _id(std::move(id))
{
}

Graph::~Graph()
{
  for (auto& kv : _vertices) {
    VertexGraphAccess::commit_to_graph(kv.second.get(), nullptr);
  }
}

const string&
Graph::id() const
{
  return _id;
}

unsigned
Graph::allocate_gvid()
{
  unsigned current = _next_gvid.load();
  while (current != 0) {
    if (_next_gvid.compare_exchange_weak(current, current + 1)) {
      return current;
    }
  }
  // Wrap path: counter exhausted. Scan _vertices for a free id.
  // Not MT-safe; requires external serialization like the rest of
  // insert_vertex's body.
  assert(_vertices.size() < numeric_limits<unsigned>::max());
  unsigned id = 1;
  while (_vertices.count(id)) {
    ++id;
  }
  return id;
}

Vertex*
Graph::insert_vertex(VertexPtr v)
{
  Vertex* p = v.get();
  on_insert_vertex(p);

  unsigned id = allocate_gvid();
  VertexGraphAccess::gvid(p, id);
  VertexGraphAccess::commit_to_graph(p, this);
  if (is_head_vertex(p)) {
    _head_vertices.insert(p);
  }
  _vertices.emplace(id, std::move(v));
  return p;
}

void
Graph::remove_vertex(Vertex* v)
{
  unsigned id = VertexGraphAccess::gvid(v);
  assert(_vertices.count(id));
  assert(v->num_fanouts() == 0);
  assert(!_port_vertices.count(v));
  on_remove_vertex(v);

  VertexGraphAccess::gvid(v, 0);
  VertexGraphAccess::commit_to_graph(v, nullptr);
  _head_vertices.erase(v);
  _vertices.erase(id);
}

VertexPtr
Graph::release_vertex(Vertex* v)
{
  unsigned id = VertexGraphAccess::gvid(v);
  assert(_vertices.count(id));
  assert(v->num_fanouts() == 0);
  assert(!_port_vertices.count(v));
  on_release_vertex(v);

  VertexGraphAccess::gvid(v, 0);
  VertexGraphAccess::commit_to_graph(v, nullptr);
  _head_vertices.erase(v);
  auto node = _vertices.extract(id);
  return std::move(node.mapped());
}

unsigned
Graph::num_vertices() const
{
  return static_cast<unsigned>(_vertices.size());
}

const unordered_set<Vertex*>&
Graph::head_vertices() const
{
  return _head_vertices;
}

Graph::vertex_iterator
Graph::begin()
{
  return vertex_iterator(_vertices.begin());
}

Graph::vertex_iterator
Graph::end()
{
  return vertex_iterator(_vertices.end());
}

Graph::const_vertex_iterator
Graph::begin() const
{
  return const_vertex_iterator(_vertices.begin());
}

Graph::const_vertex_iterator
Graph::end() const
{
  return const_vertex_iterator(_vertices.end());
}

void
Graph::assign_iports(vector<InEdge> v)
{
  for ([[maybe_unused]] const auto& e : v) {
    assert(e.v && e.v->num_iports() == 0);
  }
  _iports = std::move(v);
  rebuild_port_vertices();
}

void
Graph::assign_oports(vector<OutEdge> v)
{
  for ([[maybe_unused]] const auto& e : v) {
    assert(e.v && e.v->num_oports() == 0);
  }
  _oports = std::move(v);
  rebuild_port_vertices();
}

const unordered_set<Vertex*>&
Graph::port_vertices() const
{
  return _port_vertices;
}

void
Graph::rebuild_port_vertices()
{
  _port_vertices.clear();
  for (const auto& e : _iports) {
    _port_vertices.insert(e.v);
  }
  for (const auto& e : _oports) {
    _port_vertices.insert(e.v);
  }
}

void
Graph::update_head_membership(Vertex* v)
{
  if (is_head_vertex(v)) {
    _head_vertices.insert(v);
  } else {
    _head_vertices.erase(v);
  }
}

bool
Graph::move_iport_to(Vertex* v, unsigned iport, InEdge new_src)
{
  assert(v);
  assert(v->graph() == this);
  if (iport >= v->num_iports()) {
    return false;
  }
  // A non-null new source must live in this graph and actually expose
  // the requested oport, so the fanout we attach below is real.
  if (new_src.v) {
    if (new_src.v->graph() != this
        || new_src.p >= new_src.v->num_oports()) {
      return false;
    }
  }
  if (!can_move_iport(v, iport)) {
    return false;
  }

  const InEdge old = v->iport_edges()[iport];
  if (old == new_src) {
    return true;  // no-op (covers null -> null)
  }

  // `v` is committed to this graph, so its non-null iports already
  // have matching fanouts on their sources. Swap the reverse edge:
  // detach the old, re-point the forward edge, attach the new.
  if (old.v) {
    VertexGraphAccess::detach_fanout(old.v, old.p, OutEdge{v, iport});
  }
  VertexGraphAccess::set_iport_edge(v, iport, new_src);
  if (new_src.v) {
    VertexGraphAccess::attach_fanout(
        new_src.v, new_src.p, OutEdge{v, iport});
  }

  update_head_membership(v);
  on_iport_moved(v, iport);
  return true;
}

unsigned
Graph::num_iports() const
{
  return static_cast<unsigned>(_iports.size());
}

unsigned
Graph::num_oports() const
{
  return static_cast<unsigned>(_oports.size());
}

Graph*
Graph::insert_graph(GraphPtr g)
{
  assert(_graphs.find(g->id()) == _graphs.end());
  Graph* p = g.get();
  p->parent_graph(this);
  _graphs.emplace(p->id(), std::move(g));
  return p;
}

void
Graph::remove_graph(Graph* g)
{
  g->parent_graph(nullptr);
  _graphs.erase(g->id());
}

GraphPtr
Graph::release_graph(Graph* g)
{
  auto it = _graphs.find(g->id());
  if (it == _graphs.end()) {
    return nullptr;
  }
  g->parent_graph(nullptr);
  GraphPtr p = std::move(it->second);
  _graphs.erase(it);
  return p;
}

Graph*
Graph::graph(string id)
{
  auto it = _graphs.find(id);
  if (it == _graphs.end()) {
    return nullptr;
  }
  return it->second.get();
}

const Graph*
Graph::graph(string id) const
{
  auto it = _graphs.find(id);
  if (it == _graphs.end()) {
    return nullptr;
  }
  return it->second.get();
}

Graph*
Graph::parent_graph()
{
  return _parent_graph;
}

const Graph*
Graph::parent_graph() const
{
  return _parent_graph;
}

void
Graph::parent_graph(Graph* g)
{
  _parent_graph = g;
}

}


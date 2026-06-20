#ifndef GRAPH_H
#define GRAPH_H

#include "common/vertex.h"
#include <atomic>
#include <memory>
#include <string>
#include <unordered_map>

namespace vpipe {

class Graph;
typedef std::unique_ptr<Graph> GraphPtr;

// Thread-safety:
// - Read-only accessors (id, num_vertices, head_vertices,
//   port_vertices, num_iports, num_oports, parent_graph,
//   graph(string) const, const begin/end) may be called concurrently
//   with each other when no mutator is in progress on this graph or
//   any of its vertices.
// - Mutators require external serialization, with one exception: the
//   pre-wrap gvid allocation inside insert_vertex is lock-free
//   MT-safe (CAS loop on _next_gvid). The wrap-path scan that takes
//   over after ~2^32 inserts is *not* MT-safe and, like the rest of
//   insert_vertex's body, must run under external serialization.
class Graph : public SessionMember {
public:
  class vertex_iterator {
    using map_iter =
      std::unordered_map<unsigned, VertexPtr>::iterator;
  public:
    vertex_iterator() = default;
    explicit vertex_iterator(map_iter it) : _it(it) {}
    Vertex* operator*() const { return _it->second.get(); }
    vertex_iterator& operator++() { ++_it; return *this; }
    bool operator==(const vertex_iterator& o) const
    { return _it == o._it; }
    bool operator!=(const vertex_iterator& o) const
    { return _it != o._it; }
  private:
    map_iter _it;
  };

  class const_vertex_iterator {
    using map_iter =
      std::unordered_map<unsigned, VertexPtr>::const_iterator;
  public:
    const_vertex_iterator() = default;
    explicit const_vertex_iterator(map_iter it) : _it(it) {}
    const Vertex* operator*() const { return _it->second.get(); }
    const_vertex_iterator& operator++() { ++_it; return *this; }
    bool operator==(const const_vertex_iterator& o) const
    { return _it == o._it; }
    bool operator!=(const const_vertex_iterator& o) const
    { return _it != o._it; }
  private:
    map_iter _it;
  };

  Graph(std::string, const SessionContextIntf*);
  virtual ~Graph();

  const std::string& id() const;
  Vertex*   insert_vertex(VertexPtr);
  void      remove_vertex(Vertex*);
  VertexPtr release_vertex(Vertex*);
  unsigned  num_vertices() const;
  const std::unordered_set<Vertex*>& head_vertices() const;

  vertex_iterator begin();
  vertex_iterator end();
  const_vertex_iterator begin() const;
  const_vertex_iterator end() const;

  void assign_iports(std::vector<InEdge>);
  void assign_oports(std::vector<OutEdge>);
  unsigned num_iports() const;
  unsigned num_oports() const;
  const std::unordered_set<Vertex*>& port_vertices() const;

  // Re-point one of a vertex's iports after construction, so a graph
  // built incrementally (e.g. by an editor) can be rewired without
  // tearing the vertex down. `v` must be a vertex owned by this graph
  // and `iport` must index into its iport list.
  //
  // `new_src` is the new upstream {source vertex, source oport}. A
  // NULL source (new_src.v == nullptr -- the default) DISCONNECTS the
  // iport: it becomes unwired and `v` re-joins the head-vertex set if
  // all of its iports are now null.
  //
  // The bidirectional invariant is preserved: the old source's fanout
  // for {v, iport} is detached, the new source's fanout is attached,
  // and head-vertex membership is updated.
  //
  // Returns true on success (including a no-op move to the same edge).
  // Returns false, leaving the graph unchanged, when:
  //   - `iport` is out of range,
  //   - `new_src.v` is non-null but is not owned by this graph, or its
  //     oport index is out of range,
  //   - a subclass guard rejects the move (see can_move_iport -- e.g.
  //     a Pipeline refuses to rewire a stage whose driver is running).
  bool move_iport_to(Vertex* v, unsigned iport,
                     InEdge new_src = InEdge{nullptr, 0});

  // Read-only access to the graph-level external port lists. Each
  // iport is an InEdge {carrier, p}: data conceptually arrives at
  // carrier.oport(p) and fans out to internal consumers. Each oport
  // is an OutEdge {carrier, p}: data arrives at carrier.iport(p)
  // from internal producers and is exposed to external consumers.
  // Both vectors live as long as no further assign_iports/oports
  // call mutates them. Used by call-stage inlining at launch.
  const std::vector<InEdge>&  iports() const noexcept { return _iports; }
  const std::vector<OutEdge>& oports() const noexcept { return _oports; }

  Graph* insert_graph(GraphPtr);
  void   remove_graph(Graph*);
  GraphPtr release_graph(Graph*);
  Graph* graph(std::string);
  const Graph* graph(std::string) const;
  Graph* parent_graph();
  const Graph* parent_graph() const;

  // Read-only access to the sub-graph map, used by
  // pipeline-spec.cc when serialising sub-pipelines. The reference
  // is valid until the next mutator on this graph.
  const std::unordered_map<std::string, GraphPtr>&
  graphs() const { return _graphs; }

protected:
  virtual void on_insert_vertex(Vertex*) {};
  // Not invoked from ~Graph: vertices owned by a destructing graph are
  // torn down without calling these hooks.
  virtual void on_remove_vertex(Vertex*) {};
  virtual void on_release_vertex(Vertex*) {};

  // Guard hook for move_iport_to. A subclass may veto an iport move;
  // the default always permits it. Pipeline overrides this to refuse
  // rewiring a Stage whose driver is currently running.
  virtual bool can_move_iport(Vertex* /*v*/, unsigned /*iport*/) const
  { return true; }

  // Notification hook fired after move_iport_to mutates an iport.
  // Pipeline overrides this to mark the moved Stage as needing
  // re-initialization. The default is a no-op.
  virtual void on_iport_moved(Vertex* /*v*/, unsigned /*iport*/) {};

private:
  void parent_graph(Graph*);
  void rebuild_port_vertices();
  void update_head_membership(Vertex*);
  unsigned allocate_gvid();

private:
  std::string _id;
  std::unordered_map<unsigned, VertexPtr> _vertices;
  std::unordered_set<Vertex*> _head_vertices;
  std::unordered_map<std::string, GraphPtr> _graphs;
  std::vector<InEdge>  _iports;
  std::vector<OutEdge> _oports;
  std::unordered_set<Vertex*> _port_vertices;
  std::atomic<unsigned> _next_gvid{1};
  Graph* _parent_graph = nullptr;
};

}


#endif


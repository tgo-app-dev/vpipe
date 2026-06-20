#ifndef VERTEX_H
#define VERTEX_H

#include "common/oport-policy.h"
#include "common/session-member.h"
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace vpipe {

class Vertex;
typedef std::unique_ptr<Vertex> VertexPtr;

class Graph;

struct OutEdge {
  Vertex*  v;
  unsigned p;
};

struct InEdge {
  Vertex*  v;
  unsigned p;
};

inline bool
operator==(const OutEdge& a, const OutEdge& b)
{
  return a.v == b.v && a.p == b.p;
}

inline bool
operator==(const InEdge& a, const InEdge& b)
{
  return a.v == b.v && a.p == b.p;
}

}

namespace std {

template <>
struct hash<vpipe::OutEdge> {
  size_t operator()(const vpipe::OutEdge& e) const noexcept
  {
    size_t h1 = hash<vpipe::Vertex*>{}(e.v);
    size_t h2 = hash<unsigned>{}(e.p);
    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
  }
};

template <>
struct hash<vpipe::InEdge> {
  size_t operator()(const vpipe::InEdge& e) const noexcept
  {
    size_t h1 = hash<vpipe::Vertex*>{}(e.v);
    size_t h2 = hash<unsigned>{}(e.p);
    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
  }
};

}

namespace vpipe {

class VertexGraphAccess;

// Thread-safety: Vertex performs no internal synchronization.
// - id(), iport_edges(), num_iports() read state fixed at construction
//   and are safe to call concurrently from any number of threads.
// - graph(), oport_edges(p), num_oports(), num_fanouts() expose
//   graph-managed state. Reading them concurrently with any Graph
//   operation that touches this vertex - or any vertex naming this
//   one as a source - is a data race.
class Vertex : public SessionMember {
public:
  Vertex(const SessionContextIntf*, std::string, std::vector<InEdge>);
  virtual ~Vertex() = default;

  const std::string id() const;
  const std::vector<InEdge>& iport_edges() const;
  const std::unordered_set<OutEdge>& oport_edges(unsigned oport) const;
  unsigned num_iports() const;
  unsigned num_oports() const;
  unsigned num_fanouts() const;
  unsigned num_fanouts(unsigned) const;

  void allocate_oports(unsigned);

  // Per-oport policy. Default-constructed entries leave both fields
  // at their built-in defaults (capacity 0 -> session default;
  // mode = Backpressure). Stages that need a different policy on a
  // particular oport call set_oport_policy in their constructor,
  // right after allocate_oports.
  void               set_oport_policy(unsigned oport, OportPolicy);
  const OportPolicy& oport_policy(unsigned oport) const;

  Graph* graph() const;

private:
  friend class VertexGraphAccess;

  void commit_to_graph(Graph*);
  void gvid(unsigned);
  unsigned gvid() const;
  void attach_fanout(unsigned oport, OutEdge);
  void detach_fanout(unsigned oport, OutEdge);
  void set_iport_edge(unsigned iport, InEdge);

private:
  std::string _id;
  std::vector<InEdge> _iports;
  std::vector<std::unordered_set<OutEdge>> _oports;
  std::vector<OportPolicy> _oport_policies;
  Graph* _graph;
  unsigned _gvid;
};

class VertexGraphAccess {
public:
  static void commit_to_graph(Vertex* v, Graph* g);
  static void gvid(Vertex* v, unsigned id);
  static unsigned gvid(const Vertex* v);
  static void attach_fanout(Vertex* v, unsigned oport, OutEdge e);
  static void detach_fanout(Vertex* v, unsigned oport, OutEdge e);
  // Re-point one of v's iports. Used by Graph::move_iport_to to edit
  // a vertex's upstream after construction; the Graph is responsible
  // for keeping the matching oport fanout sets consistent.
  static void set_iport_edge(Vertex* v, unsigned iport, InEdge e);
};

}


#endif


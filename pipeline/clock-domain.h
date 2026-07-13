#ifndef PIPELINE_CLOCK_DOMAIN_H
#define PIPELINE_CLOCK_DOMAIN_H

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace vpipe {

class Stage;

// Identifies a single port on a stage. Used as the union-find element
// in the clock-domain analysis.
struct PortKey {
  enum class Kind : uint8_t { In, Out };
  Stage*   stage;
  Kind     kind;
  unsigned port;

  bool
  operator==(const PortKey& o) const noexcept
  {
    return stage == o.stage && kind == o.kind && port == o.port;
  }
};

}

namespace std {

template <>
struct hash<vpipe::PortKey> {
  size_t
  operator()(const vpipe::PortKey& k) const noexcept
  {
    size_t h = hash<vpipe::Stage*>{}(k.stage);
    h ^= static_cast<size_t>(k.kind) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= hash<unsigned>{}(k.port)    + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

}

namespace vpipe {

// Output of compute_clock_domains().
//
// `port_domain` maps every port (iport and oport) of every stage in
// the pipeline to a globally-numbered domain id in [0, num_domains).
//
// `stages_in_domain[d]` is the list of stages whose iports and oports
// all map to domain `d` (single-clock stages -- they "live in" that
// domain).
//
// `crossers` contains every stage with ports in two or more distinct
// domains (e.g. load-video reading packets in one domain and
// emitting frames in another).
//
// `intra_domain_cycle` is set to true if the directed sub-graph
// induced by any single domain contains a cycle. Such cycles need a
// `{drain, feed}` stage pair to be valid runtime topology -- that
// work is deferred; the runtime rejects launch when this flag is
// set.
struct ClockDomainAssignment {
  unsigned                                   num_domains = 0;
  std::unordered_map<PortKey, unsigned>      port_domain;
  std::vector<std::vector<Stage*>>           stages_in_domain;
  std::vector<Stage*>                        crossers;
  bool                                       intra_domain_cycle = false;
  // The first stage detected to be on a cycle (for diagnostics).
  Stage*                                     cycle_stage = nullptr;
};

// One directed cross-stage edge, in logical (post-inlining) form:
// `src` produces on its `src_oport`, `dst` consumes on its
// `dst_iport`. Pipeline-runtime constructs this list after flatten
// + edge rewrites so the analysis sees the inlined graph rather
// than raw oport_edges() of stages with phantom downstreams.
struct LogicalEdge {
  Stage*   src;
  unsigned src_oport;
  Stage*   dst;
  unsigned dst_iport;
};

// Run the analysis. `stages` is the full pipeline membership; the
// caller is responsible for passing every stage in the launched
// pipeline (a partial list yields meaningless port-domain ids on the
// missing stages). `edges` is the list of cross-stage logical
// edges; it replaces the implicit walk over `oport_edges()` so the
// analysis is correct for inlined call-stage graphs.
ClockDomainAssignment compute_clock_domains(
    const std::vector<Stage*>&      stages,
    const std::vector<LogicalEdge>& edges);

// Back-compat overload: derives the edge list from raw
// `oport_edges()` of the supplied stages. Use this when the graph
// has no call-stages / inlining; pipeline-runtime uses the
// explicit form so the inliner's rewrites are reflected.
ClockDomainAssignment compute_clock_domains(
    const std::vector<Stage*>& stages);

}

#endif

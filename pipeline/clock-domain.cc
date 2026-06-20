#include "pipeline/clock-domain.h"

#include "common/vertex.h"
#include "pipeline/stage.h"

#include <algorithm>
#include <set>
#include <unordered_set>
#include <utility>

using namespace std;

namespace vpipe {

namespace {

// Disjoint-set forest keyed on PortKey.
struct UnionFind {
  unordered_map<PortKey, PortKey> parent;
  unordered_map<PortKey, unsigned> rank_;

  void
  add(const PortKey& k)
  {
    if (parent.find(k) == parent.end()) {
      parent[k] = k;
      rank_[k]  = 0;
    }
  }

  PortKey
  find(PortKey k)
  {
    while (!(parent[k] == k)) {
      PortKey p = parent[k];
      parent[k] = parent[p];  // path compression (one-step halving)
      k = parent[k];
    }
    return k;
  }

  void
  unify(const PortKey& a, const PortKey& b)
  {
    PortKey ra = find(a);
    PortKey rb = find(b);
    if (ra == rb) {
      return;
    }
    unsigned ra_rank = rank_[ra];
    unsigned rb_rank = rank_[rb];
    if (ra_rank < rb_rank) {
      parent[ra] = rb;
    } else if (ra_rank > rb_rank) {
      parent[rb] = ra;
    } else {
      parent[rb] = ra;
      rank_[ra]  = ra_rank + 1;
    }
  }
};

// Adjacency: per source stage, the list of (downstream stage)
// reached through any of the source's oports (in this domain or
// not). Built from the explicit edge list so the cycle check sees
// the same logical topology as the unifier.
using Adjacency =
    unordered_map<Stage*, vector<Stage*>>;

Adjacency
build_adjacency_(const vector<LogicalEdge>& edges)
{
  Adjacency adj;
  for (const LogicalEdge& e : edges) {
    adj[e.src].push_back(e.dst);
  }
  return adj;
}

// DFS-based cycle detection over the sub-graph induced by a single
// clock domain. Standard three-colour algorithm (white/gray/black).
bool
domain_has_cycle_(const vector<Stage*>&        stages_in_d,
                  const unordered_set<Stage*>& stages_in_d_set,
                  const Adjacency&             adj,
                  Stage**                      hit_out)
{
  enum Color { White, Gray, Black };
  unordered_map<Stage*, Color> color;
  for (Stage* s : stages_in_d) {
    color[s] = White;
  }

  // Iterative DFS so deep graphs don't blow the call stack.
  vector<pair<Stage*, size_t>> stk;  // (stage, next-neighbor-idx)
  for (Stage* root : stages_in_d) {
    if (color[root] != White) {
      continue;
    }
    color[root] = Gray;
    stk.push_back({root, 0});
    while (!stk.empty()) {
      auto& [s, p] = stk.back();
      bool advanced = false;
      auto adj_it = adj.find(s);
      const vector<Stage*>* nbrs =
          (adj_it != adj.end()) ? &adj_it->second : nullptr;
      while (nbrs && p < nbrs->size()) {
        Stage* dst = (*nbrs)[p];
        ++p;
        if (stages_in_d_set.find(dst) == stages_in_d_set.end()) {
          continue;
        }
        auto it = color.find(dst);
        if (it == color.end() || it->second == White) {
          color[dst] = Gray;
          stk.push_back({dst, 0});
          advanced = true;
          break;
        }
        if (it->second == Gray) {
          if (hit_out) {
            *hit_out = dst;
          }
          return true;
        }
      }
      if (!advanced) {
        color[s] = Black;
        stk.pop_back();
      }
    }
  }
  return false;
}

}

ClockDomainAssignment
compute_clock_domains(const vector<Stage*>& stages)
{
  // Derive the edge list from raw oport_edges() and delegate.
  // Used by tests / callers that have no call-stage inlining to
  // worry about.
  vector<LogicalEdge> edges;
  for (Stage* s : stages) {
    for (unsigned p = 0; p < s->num_oports(); ++p) {
      for (const OutEdge& e : s->oport_edges(p)) {
        Stage* dst = dynamic_cast<Stage*>(e.v);
        if (!dst) {
          continue;
        }
        edges.push_back(LogicalEdge{s, p, dst, e.p});
      }
    }
  }
  return compute_clock_domains(stages, edges);
}

ClockDomainAssignment
compute_clock_domains(const vector<Stage*>&      stages,
                      const vector<LogicalEdge>& edges)
{
  ClockDomainAssignment out;

  // 1. Seed every port of every stage as its own class.
  UnionFind uf;
  for (Stage* s : stages) {
    for (unsigned i = 0; i < s->num_iports(); ++i) {
      uf.add(PortKey{s, PortKey::Kind::In, i});
    }
    for (unsigned p = 0; p < s->num_oports(); ++p) {
      uf.add(PortKey{s, PortKey::Kind::Out, p});
    }
  }

  // 2. Within each stage, unify ports that report the same
  // clock_group. Group ids are stage-local; we just bucket ports by
  // (stage, group, kind?) -- actually two ports with the same group
  // id on the same stage are on the same clock regardless of kind,
  // so iport_clock_group(i) == oport_clock_group(p) means they share
  // a clock too.
  for (Stage* s : stages) {
    // Map clock_group_id -> first port we saw with that id. New
    // ports with the same id get unified with the first.
    unordered_map<unsigned, PortKey> first_in_group;
    auto handle = [&](PortKey k, unsigned grp) {
      auto it = first_in_group.find(grp);
      if (it == first_in_group.end()) {
        first_in_group[grp] = k;
      } else {
        uf.unify(it->second, k);
      }
    };
    for (unsigned i = 0; i < s->num_iports(); ++i) {
      handle(PortKey{s, PortKey::Kind::In, i},
             s->iport_clock_group(i));
    }
    for (unsigned p = 0; p < s->num_oports(); ++p) {
      handle(PortKey{s, PortKey::Kind::Out, p},
             s->oport_clock_group(p));
    }
  }

  // 3. Unify across fanout edges: a producer's oport and its
  // consumer's iport are necessarily on the same clock. The edge
  // list is logical (post-inlining); the analysis is correct for
  // call-stage-inlined graphs because the inliner has already
  // resolved phantoms.
  for (const LogicalEdge& e : edges) {
    uf.unify(PortKey{e.src, PortKey::Kind::Out, e.src_oport},
             PortKey{e.dst, PortKey::Kind::In,  e.dst_iport});
  }

  // 4. Number the resulting equivalence classes 0..N-1.
  unordered_map<PortKey, unsigned> rep_to_id;
  for (auto& kv : uf.parent) {
    PortKey rep = uf.find(kv.first);
    auto it = rep_to_id.find(rep);
    if (it == rep_to_id.end()) {
      unsigned id = static_cast<unsigned>(rep_to_id.size());
      rep_to_id[rep] = id;
      out.port_domain[kv.first] = id;
    } else {
      out.port_domain[kv.first] = it->second;
    }
  }
  out.num_domains = static_cast<unsigned>(rep_to_id.size());
  out.stages_in_domain.resize(out.num_domains);

  // 5. Classify each stage as single-domain or crosser.
  for (Stage* s : stages) {
    set<unsigned> seen;
    auto note = [&](PortKey k) {
      auto it = out.port_domain.find(k);
      if (it != out.port_domain.end()) {
        seen.insert(it->second);
      }
    };
    for (unsigned i = 0; i < s->num_iports(); ++i) {
      note(PortKey{s, PortKey::Kind::In, i});
    }
    for (unsigned p = 0; p < s->num_oports(); ++p) {
      note(PortKey{s, PortKey::Kind::Out, p});
    }
    if (seen.size() == 1) {
      out.stages_in_domain[*seen.begin()].push_back(s);
    } else if (seen.size() > 1) {
      out.crossers.push_back(s);
    }
    // seen.size() == 0 means a stage with zero ports; skip silently.
  }

  // 6. Per-domain cycle check. The sub-graph for domain d is
  // {stages_in_domain[d]} restricted to edges whose source-oport and
  // dest-iport both live in d. Crossers are not part of any
  // sub-graph (they sit on the boundaries) so cycles routed through
  // a crosser don't count as intra-domain cycles.
  Adjacency adj = build_adjacency_(edges);
  for (unsigned d = 0; d < out.num_domains; ++d) {
    const auto& list = out.stages_in_domain[d];
    if (list.size() < 2) {
      continue;
    }
    unordered_set<Stage*> in_d(list.begin(), list.end());
    Stage* hit = nullptr;
    if (domain_has_cycle_(list, in_d, adj, &hit)) {
      out.intra_domain_cycle = true;
      out.cycle_stage = hit;
      break;
    }
  }

  return out;
}

}

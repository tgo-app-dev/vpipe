#include "pipeline/memory-plan.h"

#include "pipeline/stage.h"

#include <algorithm>
#include <queue>
#include <string>
#include <unordered_map>

namespace vpipe {

namespace {

// Kahn over the logical edges, restricted to `stages`.
//
// Ties are broken by declaration order rather than by whatever the queue
// happens to hold, so the plan a graph produces does not depend on
// unordered_map iteration -- two runs of one pipeline must report the
// same peak or the number cannot be used to refuse anything.
//
// Returns false on a cycle, having filled `pos` with declaration order.
// The runtime refuses intra-domain cycles separately; this only has to
// avoid producing a confident number from a graph it could not order.
bool
topological_positions_(const std::vector<Stage*>&      stages,
                       const std::vector<LogicalEdge>& edges,
                       std::vector<int>*               pos)
{
  const std::size_t n = stages.size();
  pos->assign(n, 0);
  std::unordered_map<const Stage*, std::size_t> index;
  for (std::size_t i = 0; i < n; ++i) { index[stages[i]] = i; }

  std::vector<std::vector<std::size_t>> out(n);
  std::vector<int> indeg(n, 0);
  for (const LogicalEdge& e : edges) {
    auto s = index.find(e.src);
    auto d = index.find(e.dst);
    if (s == index.end() || d == index.end()) { continue; }
    if (s->second == d->second) { continue; }        // self-loop
    out[s->second].push_back(d->second);
    ++indeg[d->second];
  }

  // A min-heap on declaration index is what makes it deterministic.
  std::priority_queue<std::size_t, std::vector<std::size_t>,
                      std::greater<std::size_t>> ready;
  for (std::size_t i = 0; i < n; ++i) {
    if (indeg[i] == 0) { ready.push(i); }
  }
  std::size_t placed = 0;
  while (!ready.empty()) {
    const std::size_t i = ready.top();
    ready.pop();
    (*pos)[i] = (int)placed++;
    for (std::size_t j : out[i]) {
      if (--indeg[j] == 0) { ready.push(j); }
    }
  }
  if (placed == n) { return true; }
  for (std::size_t i = 0; i < n; ++i) { (*pos)[i] = (int)i; }
  return false;
}

}  // namespace

MemoryPlan
compute_memory_plan(const std::vector<Stage*>&      stages,
                    const std::vector<LogicalEdge>& edges,
                    const std::vector<StageMemory>& mem)
{
  MemoryPlan plan;
  const std::size_t n = std::min(stages.size(), mem.size());
  if (n == 0) { return plan; }

  std::vector<int> pos;
  plan.ordered = topological_positions_(stages, edges, &pos);
  pos.resize(n, 0);

  // Everything that occupies memory, reduced to the span of positions it
  // is alive over. Holds, scratch and cross-stage buffers are the same
  // question here -- bytes with a lifetime -- so they are walked
  // together rather than summed separately and added.
  struct Live {
    std::size_t floor_bytes;
    std::size_t preload_bytes;
    int         first;
    int         last;
  };
  std::vector<Live> live;
  live.reserve(n * 2);

  const int last_pos = (int)n - 1;

  // Holdings, MERGED BY SOURCE.
  //
  // Two stages naming one checkpoint hold the same bytes, so the graph
  // pays for them once. Bytes take the LARGER of what the two said --
  // they are describing one thing and disagree only by being differently
  // informed -- and the lifetime takes the UNION, because a checkpoint
  // one stage gives back and another keeps is kept.
  //
  // An unnamed holding is never merged, which is what makes `source`
  // optional: a stage that cannot identify what it holds gets its own
  // entry and is over-counted rather than wrongly shared with a
  // neighbour.
  {
    struct Merged { std::size_t fl, pre; int first, last; };
    std::unordered_map<std::string, Merged> by_source;
    for (std::size_t i = 0; i < n; ++i) {
      for (const StageHolding& h : mem[i].holdings) {
        const std::size_t fl = h.floor_bytes();
        if (h.preload == 0 && fl == 0) { continue; }
        const int a = h.releases ? pos[i] : 0;
        const int z = h.releases ? pos[i] : last_pos;
        if (h.source.empty()) {
          live.push_back({fl, h.preload, a, z});
          continue;
        }
        auto it = by_source.find(h.source);
        if (it == by_source.end()) {
          by_source.emplace(h.source, Merged{fl, h.preload, a, z});
          continue;
        }
        Merged& g = it->second;
        if (fl > g.fl) { g.fl = fl; }
        if (h.preload > g.pre) { g.pre = h.preload; }
        if (a < g.first) { g.first = a; }
        if (z > g.last) { g.last = z; }
      }
    }
    for (const auto& [src, g] : by_source) {
      (void)src;
      live.push_back({g.fl, g.pre, g.first, g.last});
    }
  }

  for (std::size_t i = 0; i < n; ++i) {
    if (mem[i].scratch > 0) {
      live.push_back({mem[i].scratch, mem[i].scratch, pos[i], pos[i]});
    }
  }

  // Cross-stage buffers. ONE buffer per (producer, oport) however many
  // consumers it has -- a fanout is the same bytes read twice, and
  // counting it per edge would charge a graph for splitting a wire.
  // It lives until the LAST of those consumers has run.
  std::unordered_map<const Stage*, std::size_t> index;
  for (std::size_t i = 0; i < n; ++i) { index[stages[i]] = i; }
  std::unordered_map<std::uint64_t, int> last_use;
  std::unordered_map<std::uint64_t, std::size_t> producer;
  for (const LogicalEdge& e : edges) {
    auto s = index.find(e.src);
    auto d = index.find(e.dst);
    if (s == index.end() || d == index.end()) { continue; }
    const StageMemory& m = mem[s->second];
    if (e.src_oport >= m.outputs.size()) { continue; }
    if (m.outputs[e.src_oport] == 0) { continue; }
    const std::uint64_t key =
        ((std::uint64_t)s->second << 32) | (std::uint64_t)e.src_oport;
    producer[key] = s->second;
    auto it = last_use.find(key);
    if (it == last_use.end() || pos[d->second] > it->second) {
      last_use[key] = pos[d->second];
    }
  }
  for (const auto& [key, upto] : last_use) {
    const std::size_t si = producer[key];
    const std::size_t bytes = mem[si].outputs[(std::size_t)(key & 0xffffffffu)];
    const int from = pos[si];
    live.push_back({bytes, bytes, from, upto < from ? from : upto});
  }

  // A stage's output that NOTHING consumes still exists -- save-video
  // writes a file from a clip nobody else reads -- so it is alive at the
  // producer's own position rather than dropped.
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t p = 0; p < mem[i].outputs.size(); ++p) {
      if (mem[i].outputs[p] == 0) { continue; }
      const std::uint64_t key = ((std::uint64_t)i << 32) | (std::uint64_t)p;
      if (last_use.count(key) != 0) { continue; }
      live.push_back({mem[i].outputs[p], mem[i].outputs[p],
                      pos[i], pos[i]});
    }
  }

  // Walk the positions. `steps` is indexed by POSITION, so the report
  // reads in running order rather than declaration order.
  std::vector<const Stage*> at(n, nullptr);
  for (std::size_t i = 0; i < n; ++i) { at[(std::size_t)pos[i]] = stages[i]; }
  plan.steps.resize(n);
  for (int p = 0; p < (int)n; ++p) {
    std::size_t f = 0, pre = 0;
    for (const Live& l : live) {
      if (l.first <= p && p <= l.last) {
        f += l.floor_bytes;
        pre += l.preload_bytes;
      }
    }
    MemoryPlanStep& st = plan.steps[(std::size_t)p];
    st.stage_id = at[(std::size_t)p] != nullptr ? at[(std::size_t)p]->id()
                                                : std::string();
    st.at_floor = f;
    st.at_preload = pre;
    if (f > plan.peak_floor) {
      plan.peak_floor = f;
      plan.tightest_floor = st.stage_id;
    }
    if (pre > plan.peak_preload) {
      plan.peak_preload = pre;
      plan.tightest_preload = st.stage_id;
    }
  }
  return plan;
}

}  // namespace vpipe

#include "pipeline/pipeline-runtime.h"
#include "common/graph.h"
#include "common/thread-pool.h"
#include "common/vertex.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "pipeline/clock-domain.h"
#include "pipeline/edge-reader.h"
#include "pipeline/feedback-rx-stage.h"
#include "pipeline/feedback-tx-stage.h"
#include "pipeline/oport-buffer.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/stage.h"
#include "stages/call-stage.h"
#include <cstring>
#include <exception>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

using namespace std;

namespace vpipe {

// Pipeline-wide barrier between initialize() and process(). Every
// stage driver calls arrive() exactly once after its initialize()
// returns (success or exception) and then co_awaits wait(); no
// driver proceeds to the process() loop until all N stages have
// arrived. The wait is coroutine-friendly: a driver that has to
// block frees its worker thread by suspending; the last arrive()
// re-schedules every parked driver via the session ThreadPool so the
// barrier never deadlocks the pool even when N exceeds worker count.
class InitBarrier {
public:
  InitBarrier(unsigned expected, ThreadPool* pool) noexcept
    : _expected(expected)
    , _pool(pool)
  {}

  // Signal that one stage's initialize() has returned. When the last
  // stage arrives, every parked waiter is re-scheduled on the pool.
  void arrive()
  {
    std::vector<std::coroutine_handle<>> wake;
    {
      std::lock_guard<std::mutex> lk(_mu);
      ++_arrived;
      if (_arrived >= _expected && !_open) {
        _open = true;
        wake.swap(_waiters);
      }
    }
    for (auto h : wake) {
      _pool->schedule(h);
    }
  }

  struct Awaiter {
    InitBarrier* barrier;

    bool await_ready() const noexcept
    {
      std::lock_guard<std::mutex> lk(barrier->_mu);
      return barrier->_open;
    }

    bool await_suspend(std::coroutine_handle<> h)
    {
      std::lock_guard<std::mutex> lk(barrier->_mu);
      if (barrier->_open) {
        return false;
      }
      barrier->_waiters.push_back(h);
      return true;
    }

    void await_resume() const noexcept {}
  };

  Awaiter wait() noexcept { return Awaiter{this}; }

private:
  std::mutex                            _mu;
  std::vector<std::coroutine_handle<>>  _waiters;
  unsigned                              _arrived  = 0;
  unsigned                              _expected = 0;
  bool                                  _open     = false;
  ThreadPool*                           _pool     = nullptr;
};

PipelineRuntime::PipelineRuntime(Pipeline*                 pipeline,
                                 const SessionContextIntf* session)
  : SessionMember(session)
  , _pipeline(pipeline)
{
}

PipelineRuntime::~PipelineRuntime()
{
  if (_running.load(memory_order_acquire)) {
    try {
      stop();
    } catch (...) {
      // Best effort: a destructor must not propagate.
    }
  }
  // Member destruction order (reverse declaration): _drivers
  // destructs first, calling .destroy() on each handle. Drivers are
  // at final_suspend by now, so the framework just frees the frame.
}

namespace {

// Driver coroutine. One per Stage. Drives the full
// initialize/process/drain lifecycle: initialize() once after
// everything is wired; a pipeline-wide barrier that holds every
// driver until all N stages have returned from initialize();
// process() in a loop until the runtime stops or the stage signals
// done; drain() once before output edges are closed. On exit the
// driver closes outputs (so downstream sees EOS) and signals
// completion via the supplied counter / mutex / cv.
Job
stage_driver_(Stage*                    stage,
              RuntimeContext*           ctx,
              InitBarrier*              init_barrier,
              std::atomic<unsigned>*    completed,
              std::mutex*               done_mu,
              std::condition_variable*  done_cv);

// ---------------------------------------------------------------------
// Inlining pre-phase: flatten the root pipeline by recursively
// replacing every CallStage with the body of the called sub-graph.
// Produces a flat list of *real* stages plus an upstream rewrite map
// the edge-build phase uses to "look through" erased phantom
// vertices (call-stages and the called graphs' iport/oport
// carriers).
// ---------------------------------------------------------------------

struct Endpoint {
  Vertex*  v = nullptr;
  unsigned p = 0;
  bool operator==(const Endpoint& o) const noexcept
  { return v == o.v && p == o.p; }
};

struct EndpointHash {
  size_t operator()(const Endpoint& e) const noexcept
  {
    size_t h = std::hash<Vertex*>{}(e.v);
    h ^= std::hash<unsigned>{}(e.p) +
         0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

struct FlattenCtx {
  // Output: real stages that will run, in flatten-order.
  vector<Stage*> flat;

  // Upstream rewrite: map raw {phantom_vertex, oport_on_phantom}
  // to a closer-to-real {vertex, oport}. Resolution chases until
  // the source is no longer a key; the result is a real stage in
  // `flat` (validated at edge-build time).
  unordered_map<Endpoint, Endpoint, EndpointHash> src_rewrite;

  // Vertices erased during flatten: call-stages and iport/oport
  // carriers from every inlined graph. They never appear in
  // `flat`, never have a driver, never have an EdgeBuffer.
  unordered_set<Vertex*> erased;

  // Cycle detection (recursive call graph).
  unordered_set<const Graph*> in_progress;
  // Duplicate-call detection (each pipeline may be inlined at
  // most once per launch).
  unordered_set<const Graph*> already_inlined;
};

// Resolve a (vertex, oport) pair through the rewrite chain to its
// real producer. Iterates until no more rewrite applies; in a
// well-formed flatten the loop terminates in O(call-nesting depth).
Endpoint
resolve_src_(Endpoint e,
             const unordered_map<Endpoint, Endpoint, EndpointHash>&
                 rewrite)
{
  while (true) {
    auto it = rewrite.find(e);
    if (it == rewrite.end()) {
      return e;
    }
    e = it->second;
  }
}

bool
is_carrier_(const Graph& g, Vertex* v)
{
  for (const InEdge& e : g.iports()) {
    if (e.v == v) return true;
  }
  for (const OutEdge& e : g.oports()) {
    if (e.v == v) return true;
  }
  return false;
}

// Walk `g`, recursing into every call-stage's referenced sub-graph
// in turn. Populates `fx.flat`, `fx.src_rewrite`, `fx.erased`. On
// any error, emits one warn and returns false.
//
// `g` is taken non-const because we iterate with the non-const
// vertex iterator (the const iterator yields `const Vertex*` and
// our flat list and rewrite map both want plain `Vertex*`). We
// don't mutate the graph itself.
bool
flatten_(Graph*                    g,
         FlattenCtx&               fx,
         const SessionContextIntf* session)
{
  if (fx.in_progress.count(g)) {
    session->warn(fmt(
        "PipelineRuntime: pipeline '{}' calls itself (transitive "
        "or direct); refusing to launch", g->id()));
    return false;
  }
  if (fx.already_inlined.count(g)) {
    session->warn(fmt(
        "PipelineRuntime: pipeline '{}' is referenced by more "
        "than one call-stage; the first-cut inliner allows each "
        "callee to be inlined at most once per launch",
        g->id()));
    return false;
  }
  fx.in_progress.insert(g);
  fx.already_inlined.insert(g);

  // Carriers vanish from the flat graph -- they're pure markers
  // for the graph's external port boundary.
  for (const InEdge& e : g->iports()) {
    fx.erased.insert(e.v);
  }
  for (const OutEdge& e : g->oports()) {
    fx.erased.insert(e.v);
  }

  for (auto it = g->begin(); it != g->end(); ++it) {
    Vertex* v = *it;
    if (is_carrier_(*g, v)) {
      continue;
    }

    auto* cs = dynamic_cast<CallStage*>(v);
    if (!cs) {
      auto* s = dynamic_cast<Stage*>(v);
      if (!s) {
        session->warn(fmt(
            "PipelineRuntime: pipeline '{}' has non-Stage vertex; "
            "skipped", g->id()));
        continue;
      }
      fx.flat.push_back(s);
      continue;
    }

    // Resolve the called pipeline by lexical scope.
    const Graph* callee =
        lexical_lookup_pipeline(g, cs->called_pipeline_id());
    if (!callee) {
      session->warn(fmt(
          "PipelineRuntime: call-stage '{}' references unknown "
          "pipeline '{}' (no enclosing scope provides it)",
          cs->id(), cs->called_pipeline_id()));
      return false;
    }
    if (cs->num_iports() != callee->num_iports()) {
      session->warn(fmt(
          "PipelineRuntime: call-stage '{}' has {} iport(s) but "
          "callee '{}' declares {} iport(s)",
          cs->id(), cs->num_iports(),
          callee->id(), callee->num_iports()));
      return false;
    }
    if (cs->num_oports() != callee->num_oports()) {
      session->warn(fmt(
          "PipelineRuntime: call-stage '{}' has {} oport(s) but "
          "callee '{}' declares {} oport(s)",
          cs->id(), cs->num_oports(),
          callee->id(), callee->num_oports()));
      return false;
    }

    fx.erased.insert(cs);

    // Wire CallStage iports -> callee iports.
    // CS.iport[i] has InEdge{U, up} (the external upstream).
    // callee.iport[i] is InEdge{Cin, cp}: every internal stage
    // that reads {Cin, cp} now should read {U, up} instead.
    for (unsigned i = 0; i < cs->num_iports(); ++i) {
      const InEdge& cs_iport  = cs->iport_edges()[i];
      const InEdge& cee_iport = callee->iports()[i];
      if (!cs_iport.v) {
        session->warn(fmt(
            "PipelineRuntime: call-stage '{}' iport {} is unwired",
            cs->id(), i));
        return false;
      }
      fx.src_rewrite[Endpoint{cee_iport.v, cee_iport.p}] =
          Endpoint{cs_iport.v, cs_iport.p};
    }

    // Wire callee oports -> CallStage downstream.
    // callee.oport[j] is OutEdge{Cout, cq}: Cout's iport[cq] is
    // wired back to the real internal producer (P, pp). Anyone
    // reading {CS, j} externally should now read {P, pp}.
    for (unsigned j = 0; j < cs->num_oports(); ++j) {
      const OutEdge& cee_oport = callee->oports()[j];
      Vertex* cout = cee_oport.v;
      unsigned cq  = cee_oport.p;
      if (!cout || cq >= cout->num_iports()) {
        session->warn(fmt(
            "PipelineRuntime: callee '{}' oport[{}] carrier is "
            "malformed (carrier null or iport index out of range)",
            callee->id(), j));
        return false;
      }
      const InEdge& prod = cout->iport_edges()[cq];
      if (!prod.v) {
        session->warn(fmt(
            "PipelineRuntime: callee '{}' oport[{}] carrier "
            "iport[{}] has no internal producer wired",
            callee->id(), j, cq));
        return false;
      }
      fx.src_rewrite[Endpoint{cs, j}] =
          Endpoint{prod.v, prod.p};
    }

    // The const-cast is safe: `callee` is owned by some live
    // Graph in our session and was always non-const at insertion
    // time. lexical_lookup_pipeline is const-qualified so it can
    // be reused on read-only paths, but here we need a non-const
    // pointer to iterate via the non-const vertex iterator.
    if (!flatten_(const_cast<Graph*>(callee), fx, session)) {
      return false;
    }
  }

  fx.in_progress.erase(g);
  return true;
}

}

bool
PipelineRuntime::launch()
{
  if (_launched) {
    session()->warn(fmt(
      "PipelineRuntime: launch() called twice on '{}'; ignored",
      _pipeline->id()));
    return false;
  }
  _launched = true;

  // Phase 0: flatten. Inline every CallStage's body in-place and
  // collect the upstream rewrites we need to skip past phantoms.
  FlattenCtx fx;
  if (!flatten_(_pipeline, fx, session())) {
    return false;
  }
  vector<Stage*>&        stages = fx.flat;
  unordered_set<Vertex*> stage_set(stages.begin(), stages.end());
  if (stages.empty()) {
    session()->warn(fmt(
      "PipelineRuntime: pipeline '{}' has no stages",
      _pipeline->id()));
    return false;
  }

  // Resolve every real-stage iport to its (real_src, real_src_p).
  // The result feeds clock-domain analysis, edge-buffer build, and
  // the per-stage in_bufs construction. Each (resolved) entry must
  // land on a stage that's actually in `stages` -- otherwise the
  // graph is malformed (e.g. an iport reads from a carrier that
  // wasn't filled by any inlined call-stage).
  using EdgeKey = tuple<Vertex*, unsigned, Vertex*, unsigned>;
  vector<vector<Endpoint>> iport_src; // [stage_idx][iport] = real src
  iport_src.resize(stages.size());
  vector<LogicalEdge> logical_edges;
  for (size_t si = 0; si < stages.size(); ++si) {
    Stage* s = stages[si];
    iport_src[si].resize(s->num_iports());
    for (unsigned i = 0; i < s->num_iports(); ++i) {
      const InEdge& raw = s->iport_edges()[i];
      if (!raw.v) {
        // A DISCONNECTED (optional) iport: no producer. Keep the
        // positional slot -- it gets a null-parent EdgeReader below
        // (permanent EOS) -- but contribute no clock-domain edge.
        iport_src[si][i] = Endpoint{nullptr, 0};
        continue;
      }
      Endpoint src = resolve_src_({raw.v, raw.p}, fx.src_rewrite);
      if (stage_set.find(src.v) == stage_set.end()) {
        session()->warn(fmt(
          "PipelineRuntime: stage '{}' iport {} cannot resolve to a "
          "real producer (resolved upstream is not in the flat "
          "stage list -- usually a malformed call-stage / carrier "
          "graph)", s->id(), i));
        return false;
      }
      iport_src[si][i] = src;
      auto* src_stage = dynamic_cast<Stage*>(src.v);
      if (src_stage) {
        logical_edges.push_back(
            LogicalEdge{src_stage, src.p, s, i});
      }
    }
  }

  // Also include any oport-side fanout edges that aren't already
  // covered by the iport-side resolution above. Symmetric edges
  // (producer's fanout matched by consumer's iport) appear once
  // either way; asymmetric fanouts (e.g. a directly-attached
  // fanout that the consumer doesn't list as an upstream, used in
  // some clock-domain cycle tests) only appear on the oport side.
  // Including them here keeps clock-domain analysis seeing the
  // full topology even though edge_idx is iport-driven.
  {
    set<tuple<Stage*, unsigned, Stage*, unsigned>> seen;
    for (const LogicalEdge& e : logical_edges) {
      seen.insert({e.src, e.src_oport, e.dst, e.dst_iport});
    }
    for (Stage* s : stages) {
      for (unsigned p = 0; p < s->num_oports(); ++p) {
        for (const OutEdge& oe : s->oport_edges(p)) {
          auto* dst = dynamic_cast<Stage*>(oe.v);
          if (!dst) {
            continue;
          }
          if (stage_set.find(dst) == stage_set.end()) {
            continue;
          }
          auto key = make_tuple(s, p, dst, oe.p);
          if (seen.insert(key).second) {
            logical_edges.push_back(LogicalEdge{s, p, dst, oe.p});
          }
        }
      }
    }
  }

  // Phase 1.5: clock-domain analysis. Observational + validation:
  // log the partitioning at info level so users can see how their
  // graph was carved into clock domains, and reject the launch if
  // any single domain contains a directed cycle (those need a
  // {drain, feed} stage pair, which is future work). The explicit
  // edge list is the post-inlining logical topology -- using it
  // (instead of raw oport_edges()) keeps the analysis correct
  // when call-stages have been erased.
  ClockDomainAssignment cd = compute_clock_domains(stages, logical_edges);
  {
    ostringstream summary;
    summary << "PipelineRuntime: pipeline '" << _pipeline->id()
            << "' has " << cd.num_domains << " clock domain"
            << (cd.num_domains == 1 ? "" : "s");
    for (unsigned d = 0; d < cd.num_domains; ++d) {
      summary << "; domain " << d << " = {";
      bool first = true;
      for (Stage* s : cd.stages_in_domain[d]) {
        if (!first) summary << ", ";
        summary << s->id();
        first = false;
      }
      summary << "}";
    }
    if (!cd.crossers.empty()) {
      summary << "; crossers = {";
      bool first = true;
      for (Stage* s : cd.crossers) {
        if (!first) summary << ", ";
        summary << s->id();
        first = false;
      }
      summary << "}";
    }
    string s = summary.str();
    session()->info(fmt("{}", s));
  }
  if (cd.intra_domain_cycle) {
    session()->warn(fmt(
      "PipelineRuntime: pipeline '{}' has a directed cycle within a "
      "single clock domain (first stage on cycle: '{}'); intra-domain "
      "feedback requires a {{drain, feed}} stage pair, which is not "
      "yet supported. Refusing to launch.",
      _pipeline->id(),
      cd.cycle_stage ? cd.cycle_stage->id() : "?"));
    return false;
  }

  // Phase 1.6: feedback-pair validation. Every feedback-tx stage names
  // (via config.from) the feedback-rx stage it relays. The pair forms
  // a logical one-beat-delay register that only makes sense when both
  // endpoints already participate in the same clock domain via the
  // rest of the graph (the "register" lives inside one clock, not
  // across two). Verify rx exists and tx.oport=0 + rx.iport=0 fell
  // into the same domain.
  for (Stage* s : stages) {
    auto* tx = dynamic_cast<FeedbackTxStage*>(s);
    if (!tx) { continue; }
    FeedbackRxStage* rx = tx->rx();
    if (!rx) {
      // initialize() will not have run yet; resolve here so we can
      // perform the clock-domain check before launch. Re-uses
      // tx's stored from_id.
      const std::string& from_id = tx->from_id();
      for (Stage* candidate : stages) {
        if (candidate->id() != from_id) { continue; }
        rx = dynamic_cast<FeedbackRxStage*>(candidate);
        break;
      }
    }
    if (!rx) {
      session()->warn(fmt(
        "PipelineRuntime: pipeline '{}': feedback-tx '{}' names "
        "feedback-rx '{}' but no such stage exists in the pipeline. "
        "Refusing to launch.",
        _pipeline->id(), tx->id(), tx->from_id()));
      return false;
    }
    auto it_tx = cd.port_domain.find(
        PortKey{tx, PortKey::Kind::Out, 0});
    auto it_rx = cd.port_domain.find(
        PortKey{rx, PortKey::Kind::In, 0});
    if (it_tx == cd.port_domain.end()
        || it_rx == cd.port_domain.end()) {
      session()->warn(fmt(
        "PipelineRuntime: pipeline '{}': feedback pair tx='{}' / "
        "rx='{}' missing from clock-domain map (tx oport or rx iport "
        "unwired?). Refusing to launch.",
        _pipeline->id(), tx->id(), rx->id()));
      return false;
    }
    if (it_tx->second != it_rx->second) {
      session()->warn(fmt(
        "PipelineRuntime: pipeline '{}': feedback pair tx='{}' (domain "
        "{}) and rx='{}' (domain {}) are in different clock domains; "
        "feedback edges must stay within one clock domain. Refusing "
        "to launch.",
        _pipeline->id(), tx->id(), it_tx->second,
        rx->id(), it_rx->second));
      return false;
    }
  }

  // Phase 2a: allocate one OportBuffer per (producer, oport). Every
  // oport gets one, even if it has no consumers wired up -- a stage's
  // close_outputs() still needs something to call close() on for an
  // unwired oport, and producers can write opportunistically (the
  // buffer drops on the floor with no cursors attached).
  map<pair<Vertex*, unsigned>, OportBuffer*> oport_idx;
  for (Stage* s : stages) {
    for (unsigned p = 0; p < s->num_oports(); ++p) {
      const auto& pol = s->oport_policy(p);
      auto buf = make_unique<OportBuffer>(session(), pol);
      oport_idx[{s, p}] = buf.get();
      _oport_bufs.push_back(std::move(buf));
    }
  }

  // Phase 2b: allocate one EdgeReader per consumer iport, attaching
  // a fresh cursor to the producer's OportBuffer. iport_src reflects
  // the inlined graph so we never wire to a phantom vertex.
  map<EdgeKey, EdgeReader*> reader_idx;
  for (size_t si = 0; si < stages.size(); ++si) {
    Stage* s = stages[si];
    for (unsigned i = 0; i < s->num_iports(); ++i) {
      Endpoint src = iport_src[si][i];
      // Disconnected (optional) iport: no producer buffer. Give it a
      // null-parent EdgeReader that reads as permanent EOS, and index
      // it under the same {nullptr,0,s,i} key phase 3 rebuilds.
      if (!src.v) {
        auto reader = make_unique<EdgeReader>(nullptr, 0);
        reader_idx[{nullptr, 0, s, i}] = reader.get();
        _readers.push_back(std::move(reader));
        _edge_labels.push_back(EdgeBufferStat{
            std::string("(unwired)"), 0, s->id(), i,
            /*backlog=*/0, /*capacity=*/0, /*dropped=*/0,
            /*closed=*/true});
        continue;
      }
      // Type-compatibility check: producer's oport_payload_type vs
      // consumer's iport_payload_type. If both declare a type they
      // must match. Untyped (nullptr) on either side is permitted
      // (legacy / test stages, untyped passthrough).
      Stage* prod = dynamic_cast<Stage*>(src.v);
      if (prod) {
        const std::type_info* pt =
            prod->oport_payload_type(src.p);
        const std::type_info* ct = s->iport_payload_type(i);
        // Compare by mangled name, not type_info identity: a plugin
        // .dylib defines its own typeid objects, so `*pt != *ct` (value
        // compare) can misfire across images. The fast pointer check
        // short-circuits the common in-image case; the name() strcmp is
        // the cross-image fallback the project's tests already use.
        if (pt && ct && pt != ct
            && std::strcmp(pt->name(), ct->name()) != 0) {
          throw std::runtime_error(
              "PipelineRuntime: payload type mismatch on edge "
              + prod->id() + "[" + std::to_string(src.p) + "] -> "
              + s->id() + "[" + std::to_string(i) + "]: producer "
              + pt->name() + " vs consumer " + ct->name());
        }
        // Deeper check: payload TAGS (a finer constraint on top of the
        // beat type; OR semantics). Untagged on either side is permitted.
        const std::string_view ptg = prod->oport_payload_tags(src.p);
        const std::string_view ctg = s->iport_payload_tags(i);
        if (!port_tags_compatible(ptg, ctg)) {
          throw std::runtime_error(
              "PipelineRuntime: payload tag mismatch on edge "
              + prod->id() + "[" + std::to_string(src.p) + "] -> "
              + s->id() + "[" + std::to_string(i) + "]: producer tags {"
              + std::string(ptg) + "} vs consumer tags {"
              + std::string(ctg) + "}");
        }
      }
      OportBuffer* obuf = oport_idx.at({src.v, src.p});
      unsigned cur_idx  = obuf->attach_cursor();
      auto reader = make_unique<EdgeReader>(obuf, cur_idx);
      reader_idx[{src.v, src.p, s, i}] = reader.get();
      _readers.push_back(std::move(reader));
      // Label this edge (parallel to _readers) for edge_buffer_stats().
      _edge_labels.push_back(EdgeBufferStat{
          prod ? prod->id() : std::string("?"), src.p, s->id(), i,
          /*backlog=*/0, /*capacity=*/0, /*dropped=*/0, /*closed=*/false});
    }
  }

  // Phase 3: build per-stage RuntimeContext. in_readers come from
  // reader_idx (one cursor per iport); out_bufs come from oport_idx
  // (one shared buffer per oport).
  _contexts.reserve(stages.size());
  for (size_t si = 0; si < stages.size(); ++si) {
    Stage* s = stages[si];

    vector<EdgeReader*> in_readers;
    in_readers.reserve(s->num_iports());
    for (unsigned i = 0; i < s->num_iports(); ++i) {
      Endpoint src = iport_src[si][i];
      EdgeKey key{src.v, src.p, s, i};
      in_readers.push_back(reader_idx.at(key));
    }

    vector<OportBuffer*> out_bufs;
    out_bufs.reserve(s->num_oports());
    for (unsigned p = 0; p < s->num_oports(); ++p) {
      out_bufs.push_back(oport_idx.at({s, p}));
    }

    _contexts.push_back(make_unique<RuntimeContext>(
      std::move(in_readers), std::move(out_bufs), &_stop));
  }

  // Phase 4: spawn drivers and schedule.
  _expected = static_cast<unsigned>(stages.size());
  _init_barrier = std::make_unique<InitBarrier>(
      _expected, session()->thread_pool());
  // Mark every driven stage running before any driver is scheduled, so
  // a concurrent editor sees the pipeline as live the instant launch
  // commits. stop() clears these once the drivers have drained.
  _live_stages = stages;
  for (Stage* s : _live_stages) {
    StageLifecycleAccess::set_running(s, true);
  }
  _running.store(true, memory_order_release);
  _drivers.reserve(stages.size());
  for (size_t i = 0; i < stages.size(); ++i) {
    Job driver = stage_driver_(stages[i], _contexts[i].get(),
                               _init_barrier.get(),
                               &_completed, &_done_mu, &_done_cv);
    // Tag the driver coroutine with the stage's gvid so its first
    // resume-slice (and any later ones) attribute to the stage.
    driver.set_perf_tag(VertexGraphAccess::gvid(stages[i]));
    auto h = driver.handle();
    _drivers.push_back(std::move(driver));
    session()->thread_pool()->schedule(h);
  }
  return true;
}

std::vector<PipelineRuntime::EdgeBufferStat>
PipelineRuntime::edge_buffer_stats() const
{
  std::vector<EdgeBufferStat> out;
  out.reserve(_readers.size());
  for (std::size_t k = 0; k < _readers.size(); ++k) {
    EdgeReader*       r = _readers[k].get();
    const OportBuffer* b = r ? r->parent() : nullptr;
    EdgeBufferStat e = _edge_labels[k];   // label fields prefilled
    e.backlog  = r ? r->backlog() : 0u;
    e.capacity = b ? b->capacity() : 0u;
    e.dropped  = r ? r->dropped() : 0u;
    e.closed   = b ? b->closed() : false;
    out.push_back(std::move(e));
  }
  return out;
}

void
PipelineRuntime::pause()
{
  _stop.store(true, memory_order_release);
}

void
PipelineRuntime::stop()
{
  pause();
  // Wake any drivers currently suspended on a buffer wait list.
  for (auto& buf : _oport_bufs) {
    buf->close();
  }
  wait_idle();
  _running.store(false, memory_order_release);
  // Drivers have drained: the stages are quiescent and safe to rewire
  // again, so drop the running flag we set at launch.
  for (Stage* s : _live_stages) {
    StageLifecycleAccess::set_running(s, false);
  }
}

// Spin until every driver coroutine is officially at final_suspend.
// `_completed.fetch_add(...)` in stage_driver_ runs INSIDE the
// coroutine body before the implicit co_return, so by the time
// _completed reaches _expected the coroutine has signalled "done"
// but has not yet transitioned to the suspended-at-final_suspend
// state. Destroying the Job in that window (e.g. when ~PipelineRuntime
// tears down `_drivers`) destroys the coroutine frame while it is
// still executing its return epilogue -- libmalloc reports this
// later as a "memory corruption of free block" abort.
//
// The window is microseconds; yield is enough.
void
PipelineRuntime::wait_drivers_suspended_()
{
  for (const Job& d : _drivers) {
    auto h = d.handle();
    if (!h) { continue; }
    while (!h.done()) {
      std::this_thread::yield();
    }
  }
}

void
PipelineRuntime::wait_idle()
{
  {
    unique_lock<mutex> lk(_done_mu);
    _done_cv.wait(lk, [this] {
      return _completed.load(memory_order_acquire) >= _expected;
    });
  }
  wait_drivers_suspended_();
}

bool
PipelineRuntime::wait_idle(int timeout_ms)
{
  bool ok;
  {
    unique_lock<mutex> lk(_done_mu);
    auto pred = [this] {
      return _completed.load(memory_order_acquire) >= _expected;
    };
    if (timeout_ms < 0) {
      _done_cv.wait(lk, pred);
      ok = true;
    } else {
      ok = _done_cv.wait_for(
          lk, std::chrono::milliseconds(timeout_ms), pred);
    }
  }
  if (ok) {
    wait_drivers_suspended_();
  }
  return ok;
}

namespace {

Job
stage_driver_(Stage*                    stage,
              RuntimeContext*           ctx,
              InitBarrier*              init_barrier,
              std::atomic<unsigned>*    completed,
              std::mutex*               done_mu,
              std::condition_variable*  done_cv)
{
  const SessionContextIntf* session = stage->session();
  // Tag this stage's coroutines with its gvid so the ThreadPool can
  // attribute each worker resume-slice (schedule/unschedule) to it.
  const std::uint32_t gvid = VertexGraphAccess::gvid(stage);

  bool init_ok = false;
  // A stage that recorded a configuration problem at construction
  // (fail_config) is skipped here rather than at construction time, so
  // a graph can be built/inspected/edited before required fields are
  // supplied. Its initialize()/process() do not run; it still arrives
  // at the barrier and drains so peers don't hang.
  if (!stage->config_error().empty()) {
    session->warn(fmt(
      "stage '{}' invalid config: {}; skipping initialize/process",
      stage->id(), stage->config_error()));
  } else {
    try {
      Job j = stage->initialize(*ctx);
      j.set_perf_tag(gvid);
      co_await std::move(j);
      init_ok = true;
      // initialize() ran against the current wiring -- any pending
      // "rewired since last init" flag set by move_iport_to is now
      // satisfied.
      StageLifecycleAccess::set_needs_init(stage, false);
    } catch (const exception& e) {
      session->warn(fmt(
        "stage '{}' initialize: {}; skipping process, entering drain",
        stage->id(), e.what()));
    } catch (...) {
      session->warn(fmt(
        "stage '{}' initialize: non-std exception; skipping process, "
        "entering drain",
        stage->id()));
    }
  }

  // Pipeline-wide barrier: no stage's process() runs until every
  // stage in this pipeline has returned from initialize() (success
  // or exception). Stages that failed init still arrive so their
  // peers don't hang; they just skip the process() loop below.
  init_barrier->arrive();
  co_await init_barrier->wait();

  if (init_ok) {
    try {
      while (!ctx->stop_requested() && !ctx->done()) {
        Job j = stage->process(*ctx);
        j.set_perf_tag(gvid);
        co_await std::move(j);
      }
    } catch (const exception& e) {
      session->warn(fmt(
        "stage '{}' process: {}; entering drain",
        stage->id(), e.what()));
    } catch (...) {
      session->warn(fmt(
        "stage '{}' process: non-std exception; entering drain",
        stage->id()));
    }
  }

  // drain runs even if initialize/process failed so stages can
  // release in-flight resources (encoder buffers, file trailers,
  // ...) without leaking.
  try {
    Job j = stage->drain(*ctx);
    j.set_perf_tag(gvid);
    co_await std::move(j);
  } catch (const exception& e) {
    session->warn(fmt(
      "stage '{}' drain: {}; continuing",
      stage->id(), e.what()));
  } catch (...) {
    session->warn(fmt(
      "stage '{}' drain: non-std exception; continuing",
      stage->id()));
  }

  ctx->close_outputs();

  {
    lock_guard<mutex> lk(*done_mu);
    completed->fetch_add(1, memory_order_release);
  }
  done_cv->notify_all();

  co_return;
}

}

}

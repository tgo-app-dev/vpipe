#ifndef STAGE_H
#define STAGE_H

#include "common/flex-data.h"
#include "common/vertex.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "pipeline/stage-config.h"
#include "pipeline/stage-spec.h"
#include <atomic>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <typeinfo>
#include <vector>

namespace vpipe {

enum class StageTypeId : unsigned { unknown = 0 };

class Job;
class RuntimeContext;
class StageLifecycleAccess;

class Stage : public Vertex {
public:
  // The 4th parameter is the stage's configuration as a FlexData
  // tree. Defaulted to an empty object so call sites that predate
  // configurable stages keep compiling unchanged. Derived stages
  // read it through the protected `config()` accessor in their
  // constructors.
  Stage(const SessionContextIntf*, std::string id,
        std::vector<InEdge> iports,
        FlexData config = FlexData::make_object());
  ~Stage() override = default;

  // Numeric type id assigned at static init by TypedStage; unique
  // in the running process but NOT stable across runs / builds.
  // Persist stages by type_name(), never by type().
  virtual StageTypeId      type()      const = 0;
  // Stable, human-readable canonical name (e.g. "scale", "decode").
  virtual std::string_view type_name() const = 0;

  // Stage lifecycle, all driven from the per-stage driver coroutine
  // on the Session ThreadPool:
  //
  //   initialize(ctx)         -- once, after the pipeline is fully
  //                              assembled (every EdgeBuffer and
  //                              RuntimeContext is wired). Use it
  //                              for one-shot setup; may emit on
  //                              output ports (e.g. send a stream
  //                              params header). Default no-op.
  //
  //   process(ctx)            -- per iteration, looped until the
  //                              runtime stops or the stage signals
  //                              done via ctx.signal_done().
  //                              Inside the body, use
  //                                auto t = co_await ctx.read(p);
  //                                co_await ctx.write(p, std::move(*t));
  //                              and call ctx.signal_done() to exit
  //                              the loop after the current call.
  //
  //   drain(ctx)              -- once after the loop exits (either
  //                              the stage signalled done or stop
  //                              was requested), before the runtime
  //                              closes this stage's output edges.
  //                              Use it to flush in-flight state:
  //                              codec packet queues, file trailers,
  //                              etc. Default no-op.
  //
  // A future process_batch(RuntimeContext&, unsigned n) sits beside
  // process(); the driver will pick one based on a per-stage
  // capability flag.
  virtual Job initialize(RuntimeContext& ctx);
  virtual Job process   (RuntimeContext& ctx) = 0;
  virtual Job drain     (RuntimeContext& ctx);

  // Per-port clock group. Ports with the same clock_group on the
  // same stage share a clock; the graph analyzer (see
  // pipeline/clock-domain.h) walks fanout edges and unifies
  // connected port-groups into globally-numbered *clock domains*.
  //
  // Default: read the clock group from spec().{i,o}ports[p] (0 when
  // the port isn't declared) -- so a "single-clock" stage and any
  // stage that declares its ports in kSpec needs no override. A stage
  // can still override these directly if it computes groups
  // dynamically. The numeric value is local to the stage; only
  // same-stage equality matters.
  virtual unsigned
  iport_clock_group(unsigned p) const noexcept
  {
    const StageSpec& s = spec();
    return p < s.iports.size() ? s.iports[p].clock_group : 0;
  }
  virtual unsigned
  oport_clock_group(unsigned p) const noexcept
  {
    const StageSpec& s = spec();
    return p < s.oports.size() ? s.oports[p].clock_group : 0;
  }

  // Per-port payload type declaration. PipelineRuntime checks that
  // every wired edge has a compatible producer-output / consumer-
  // input pair: equality is required when both sides declare a
  // type; a null on either side is treated as "untyped" (legacy /
  // test stages) and skipped. Default: read from spec().{i,o}ports[p]
  // (nullptr when the port isn't declared), so declaring ports in
  // kSpec opts a stage into type-checking.
  virtual const std::type_info*
  iport_payload_type(unsigned p) const noexcept
  {
    const StageSpec& s = spec();
    return p < s.iports.size() ? s.iports[p].type : nullptr;
  }
  virtual const std::type_info*
  oport_payload_type(unsigned p) const noexcept
  {
    const StageSpec& s = spec();
    return p < s.oports.size() ? s.oports[p].type : nullptr;
  }

  // Per-port payload TAGS: an optional finer-grained constraint on top of
  // the beat type (comma-separated, OR semantics; see PortSpec::tags and
  // port_tags_compatible). PipelineRuntime and the web-ui composer require
  // a producer oport's tags and a consumer iport's tags to be compatible
  // in addition to the beat types. Default: read from spec().{i,o}ports[p]
  // (empty when undeclared -> no constraint).
  virtual std::string_view
  iport_payload_tags(unsigned p) const noexcept
  {
    const StageSpec& s = spec();
    return p < s.iports.size() ? s.iports[p].tags : std::string_view{};
  }
  virtual std::string_view
  oport_payload_tags(unsigned p) const noexcept
  {
    const StageSpec& s = spec();
    return p < s.oports.size() ? s.oports[p].tags : std::string_view{};
  }

  // Read-only access to the configuration tree the stage was
  // constructed with. Public so pipeline serialization can
  // round-trip a stage's config back into the spec; derived
  // stages also use it from their ctor bodies.
  const FlexData& config() const noexcept { return _config; }

  // ---- Stage specification ------------------------------------
  //
  // The formal description of this stage type: category, docs, ports
  // (name/doc/beat-type/clock), and config attributes. A stage
  // returns a reference to its file-static `kSpec`; the base default
  // is an empty Generic spec. The same object is registered with the
  // StageRegistry via VPIPE_REGISTER_SPEC so tooling can read it
  // without constructing an instance. See pipeline/stage-spec.h.
  virtual const StageSpec& spec() const noexcept;

  StageCategory    category()    const noexcept { return spec().category; }
  std::string_view description() const noexcept { return spec().doc; }

  // ---- Configuration introspection ----------------------------
  //
  // Static, type-level declaration of the configuration keys this
  // stage understands: their declared types, defaults, and one-line
  // docs. Default: spec().attrs -- a stage declares its attributes
  // once in kSpec. (A stage with no spec advertises no schema.)
  virtual std::span<const ConfigKey>
  config_spec() const noexcept { return spec().attrs; }

  // Per-instance resolved view: every key from config_spec() paired
  // with its declared type, default, and the current value this
  // instance was given (config(), falling back to the default). This
  // is what a client requests to learn a stage's configuration keys,
  // types, and current values.
  std::vector<ConfigParam> config_params() const;

  // config_params() serialised to a FlexData array of objects, for
  // JSON transport across the binding (see config_params_to_flex).
  FlexData config_schema() const;

  // ---- Deferred configuration validation -----------------------
  //
  // A stage MUST construct successfully for ANY configuration --
  // including an empty/default one -- so tools can build, inspect and
  // edit a stage graph before required fields are supplied. Instead of
  // throwing from the constructor, a stage records the first problem
  // via fail_config(). The pipeline runtime checks config_error()
  // before launch: a stage with a non-empty error is logged and
  // skipped (its initialize() and process() never run), while the rest
  // of the pipeline launches normally. Empty => the configuration is
  // valid.
  const std::string& config_error() const noexcept { return _config_error; }

  // ---- Lifecycle state (driven by PipelineRuntime) ------------
  //
  // A stage is "running" from the moment the PipelineRuntime schedules
  // its driver at launch() until the runtime has stopped and every
  // driver has drained. Topology edits to a running stage are unsafe
  // (the edge buffers are frozen at launch), so Graph::move_iport_to
  // refuses them via Pipeline::can_move_iport.
  bool running() const noexcept
  {
    return _running.load(std::memory_order_acquire);
  }

  // True when this stage's wiring changed since it was last
  // initialized -- i.e. an iport was moved (Pipeline::on_iport_moved)
  // after the last successful initialize(). A tool can read this to
  // know a re-launch is required for the edit to take effect; the
  // runtime clears it once initialize() runs against the new
  // topology. A freshly constructed stage reports false (its initial
  // wiring is its ctor wiring).
  bool needs_init() const noexcept
  {
    return _needs_init.load(std::memory_order_acquire);
  }

  // ---- Performance tracing ------------------------------------
  //
  // Producer hot path. The event is routed into the calling
  // thread's per-worker PerfBuffer owned by the session (or the
  // overflow buffer for non-worker callers). When profiling is
  // off, this is a single virtual call into a no-op default impl
  // on SessionContextIntf -- but we early-out via an atomic-flag
  // check first so the disabled cost is one load + branch.
  //
  // `type` is a stage-local id; override perf_event_name() to
  // translate it to a string at dump time. `value` is an optional
  // uint64 payload (frame number, byte count, queue depth, etc.).
  void
  record_perf_event(std::uint32_t type, std::uint64_t value = 0) noexcept
  {
    if (auto* sess = session()) {
      if (sess->profiling_enabled()) {
        sess->record_perf_event(
            VertexGraphAccess::gvid(this), type, value);
      }
    }
  }

  // Auxiliary-lane variant: records into a logical activity timeline
  // (perf-event.h PerfAuxLane, e.g. the ANE lane for CoreML jobs)
  // rather than the calling worker's lane, but still tagged with THIS
  // stage's gvid so the viewer names + colors it by the stage. `type`
  // follows the begin/end parity convention (even begin, odd end).
  void
  record_perf_event_aux(unsigned      lane,
                        std::uint32_t type,
                        std::uint64_t value = 0) noexcept
  {
    if (auto* sess = session()) {
      if (sess->profiling_enabled()) {
        sess->record_perf_event_aux(
            lane, VertexGraphAccess::gvid(this), type, value);
      }
    }
  }

  // Translate a stage-local event type id to a human-readable
  // name. Default returns a synthetic "event_<N>" string. Stages
  // override to give meaningful names; the override is non-static
  // and may produce instance-specific labels. Called only at dump
  // time, so cost is irrelevant.
  virtual std::string perf_event_name(std::uint32_t type) const;

protected:
  // Record a configuration problem from a constructor instead of
  // throwing. Keeps the first message; later calls are ignored. Never
  // throws, so the rest of the constructor still runs (with defaults).
  // Takes the same VpipeFormat as session()->error() so existing
  // `fail_config(fmt("...", ...))` call sites read identically.
  void fail_config(const VpipeFormat& message);

  // ---- Attribute resolution (defaults live in the spec) --------
  //
  // Read a configuration attribute: the value from config() when the
  // key is present (coerced to the requested kind), else the schema
  // default from config_spec()'s matching ConfigKey.def_* (false / 0 /
  // 0.0 / "" / empty container when the key is unknown). Stages call
  // these in their constructor so no non-zero attribute default lives
  // in a member initializer -- the spec is the single source of truth.
  // Must be called from the derived stage's own constructor body (the
  // vtable resolves spec()/config_spec() to the derived overrides
  // there).
  bool        attr_bool(std::string_view key) const;
  std::int64_t  attr_int (std::string_view key) const;
  std::uint64_t attr_uint(std::string_view key) const;
  double      attr_real(std::string_view key) const;
  std::string attr_str (std::string_view key) const;
  // Like attr_str, but for a LOCAL file path: the value is confined to
  // the session file sandbox (chroot-like) via
  // session()->confine_path. A network URL (rtsp/http(s)/...) is
  // returned unchanged (it is not a filesystem path); a file:// prefix
  // is stripped before confining. On a sandbox escape the stage's config
  // is failed (fail_config) and "" is returned. When `for_write` the
  // parent directory is created. Non-const because it may fail_config.
  std::string attr_path(std::string_view key, bool for_write);
  // Confine one already-read path/URL string to the file sandbox (the
  // shared core of attr_path). Network URLs pass through; file:// is
  // stripped; a sandbox escape calls fail_config and returns "". For
  // stages that parse their own path arrays (load-image / load-text).
  std::string confine_local_(std::string_view raw, bool for_write);
  // Composite / any-typed attribute as a FlexData (the configured
  // value, else the schema default container).
  FlexData    attr     (std::string_view key) const;

private:
  friend class StageLifecycleAccess;

  // Fetch the raw config value for `key` into `out`; true iff present.
  bool attr_present_(std::string_view key, FlexData& out) const;

  FlexData          _config;
  std::string       _config_error;
  std::atomic<bool> _running{false};
  std::atomic<bool> _needs_init{false};
};

using StagePtr = std::unique_ptr<Stage>;

// Privileged setters for the two lifecycle flags above, mirroring
// VertexGraphAccess: the PipelineRuntime marks a stage running across
// its launch/stop span and clears needs_init after initialize(), and
// Pipeline sets needs_init when an iport move changes the stage's
// inputs. Kept out of the public surface so ordinary stage code can
// only observe the flags, never forge them.
class StageLifecycleAccess {
public:
  static void set_running(Stage* s, bool running);
  static void set_needs_init(Stage* s, bool needs_init);
};

}

#endif

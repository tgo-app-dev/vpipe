#include "stages/chrono-stage.h"
#include "common/beat-payload-intf.h"
#include "common/thread-pool.h"
#include "stages/trigger-beat.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "stages/trigger-beat.h"
#include <chrono>
#include <coroutine>
#include <stdexcept>
#include <thread>
#include <utility>

using namespace std;

namespace vpipe {

namespace {

// Suspends the awaiting coroutine and re-schedules it on the pool after
// `delay`, instead of blocking the worker with a sleep. The worker is
// returned to the pool while we wait, so a periodic source no longer
// pins a thread (which showed up as a permanently "empty" profiler
// lane -- the worker was stuck mid-resume in this_thread::sleep_for).
struct TimerAwaiter {
  ThreadPool*                     pool;
  chrono::steady_clock::duration  delay;

  bool
  await_ready() const noexcept
  {
    return pool == nullptr ||
           delay <= chrono::steady_clock::duration::zero();
  }

  void
  await_suspend(coroutine_handle<> h) const noexcept
  {
    // Hoist members into locals before publishing the handle (the
    // timer can resume it on another worker the instant it fires).
    ThreadPool* p = pool;
    auto        d = delay;
    p->schedule_after(d, h);
  }

  void await_resume() const noexcept {}
};

double
read_real_(const FlexData& v)
{
  if (v.is_real()) {
    return v.get_real();
  }
  if (v.is_int()) {
    return static_cast<double>(v.get_int());
  }
  if (v.is_uint()) {
    return static_cast<double>(v.get_uint());
  }
  return 0.0;
}

}

ChronoStage::ChronoStage(const SessionContextIntf* s,
                         string                    id,
                         vector<InEdge>            iports,
                         FlexData                  config)
  : TypedStage<ChronoStage>(s, std::move(id), std::move(iports),
                            std::move(config))
{
  // Validation is deferred to launch (see Stage::fail_config): the
  // stage must construct for any config so a graph can be built/edited
  // before a period is supplied.
  const FlexData& cfg = this->config();
  bool    have_hz     = false;
  bool    have_period = false;
  double  period_secs = 0.0;
  double  hz          = 0.0;
  bool    count_set   = false;
  int64_t count_val   = 0;

  if (cfg.is_object()) {
    auto root = cfg.as_object();
    have_hz = root.contains("frequency_hz");

    auto add_unit = [&](const char* key, double scale_to_seconds) {
      if (!root.contains(key)) {
        return;
      }
      have_period = true;
      period_secs += read_real_(root.at(key)) * scale_to_seconds;
    };
    add_unit("period_seconds", 1.0);
    add_unit("period_minutes", 60.0);
    add_unit("period_hours",   3600.0);
    add_unit("period_days",    86400.0);

    if (have_hz) {
      hz = read_real_(root.at("frequency_hz"));
    }
    if (root.contains("count")) {
      count_set = true;
      count_val = root.at("count").as_int(0);
    }
  }

  if (!cfg.is_object()) {
    fail_config(fmt(
        "ChronoStage('{}'): config must be an object", this->id()));
  } else if (have_hz && have_period) {
    fail_config(fmt(
        "ChronoStage('{}'): config may set either frequency_hz or "
        "period_*, not both", this->id()));
  } else if (!have_hz && !have_period) {
    fail_config(fmt(
        "ChronoStage('{}'): config must set frequency_hz or one or "
        "more of period_seconds/period_minutes/period_hours/"
        "period_days", this->id()));
  } else if (have_hz && !(hz > 0.0)) {
    fail_config(fmt(
        "ChronoStage('{}'): frequency_hz must be > 0 (got {})",
        this->id(), hz));
  } else if (!have_hz && !(period_secs > 0.0)) {
    fail_config(fmt(
        "ChronoStage('{}'): period must be > 0 (got {} seconds)",
        this->id(), period_secs));
  }
  if (count_set && count_val < 0) {
    fail_config(fmt(
        "ChronoStage('{}'): count must be >= 0 (got {})",
        this->id(), count_val));
  }

  // Convert to integer nanoseconds; clamp microscopic / invalid periods
  // up to 1ns so steady_clock arithmetic stays sane even when the
  // config is invalid (the stage is skipped at launch in that case).
  double seconds = have_hz ? (hz > 0.0 ? 1.0 / hz : 0.0) : period_secs;
  auto ns =
      chrono::duration_cast<chrono::nanoseconds>(
          chrono::duration<double>(seconds));
  if (ns.count() <= 0) {
    ns = chrono::nanoseconds(1);
  }
  _period = ns;
  if (count_set && count_val >= 0) {
    _count = static_cast<uint64_t>(count_val);
  }

  allocate_oports(spec().oports.size());
}

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "frequency_hz", .type = ConfigType::Real,
   .doc = "ticks per second; mutually exclusive with period_*"},
  {.key = "period_seconds", .type = ConfigType::Real,
   .doc = "period contribution in seconds (period_* stack additively)"},
  {.key = "period_minutes", .type = ConfigType::Real,
   .doc = "period contribution in minutes"},
  {.key = "period_hours", .type = ConfigType::Real,
   .doc = "period contribution in hours"},
  {.key = "period_days", .type = ConfigType::Real,
   .doc = "period contribution in days"},
  {.key = "count", .type = ConfigType::Int,
   .doc = "beats to emit then signal done; 0 = forever", .def_int = 0},
};
const PortSpec kOports[] = {
  {.name = "tick", .doc = "periodic TriggerBeat", .type = &typeid(TriggerPayload),
   .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "chrono",
  .doc       = "Periodic source: emits a TriggerBeat at a fixed rate "
               "(frequency_hz or period_*), optionally N times.",
  .display_name = "Timer",
  .category  = StageCategory::Control,
  .iports    = {},
  .oports    = kOports,
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
ChronoStage::spec() const noexcept
{
  return kSpec;
}

Job
ChronoStage::process(RuntimeContext& ctx)
{
  if (_count != 0 && _emitted >= _count) {
    ctx.signal_done();
    co_return;
  }

  // Wait until the next tick WITHOUT pinning a worker: suspend and let
  // the pool's timer re-schedule us. Chunk the wait so a stop request
  // is still observed within ~50ms regardless of the configured period.
  auto deadline = chrono::steady_clock::now() + _period;
  constexpr auto kChunk = chrono::milliseconds(50);
  ThreadPool* pool = session() ? session()->thread_pool() : nullptr;
  while (true) {
    if (ctx.stop_requested()) {
      ctx.signal_done();
      co_return;
    }
    auto now = chrono::steady_clock::now();
    if (now >= deadline) {
      break;
    }
    auto remaining = deadline - now;
    auto chunk     = remaining < kChunk ? remaining : kChunk;
    if (pool) {
      co_await TimerAwaiter{pool, chunk};
    } else {
      // No pool (e.g. a unit context): fall back to a blocking sleep.
      this_thread::sleep_for(chunk);
    }
  }

  ++_emitted;
  if (_period >= chrono::nanoseconds(2000000000)) {
    session()->log_verbose(fmt("chrono('{}'): issuing beat.", id()));
  }
  co_await ctx.write(0, make_payload<TriggerPayload>());
}

VPIPE_REGISTER_STAGE(ChronoStage)
VPIPE_REGISTER_SPEC(ChronoStage, kSpec)

}

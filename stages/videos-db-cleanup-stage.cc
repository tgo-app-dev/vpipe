#include "stages/videos-db-cleanup-stage.h"
#include "common/lmdb-cursor.h"
#include "common/lmdb-db.h"
#include "common/lmdb-env.h"
#include "common/lmdb-txn.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <chrono>
#include <cstdint>
#include <exception>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace vpipe {

namespace {

// 8-byte big-endian encoding of microseconds-since-epoch. Matches
// the LMDB key format `be64_us_key` in rtsp-capture-stage.cc, so
// string-comparison of two keys equals numeric comparison of the
// underlying microseconds.
std::string
be64_us_key_(std::chrono::system_clock::time_point t)
{
  uint64_t us = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          t.time_since_epoch()).count());
  std::string out;
  out.resize(8);
  for (int i = 0; i < 8; ++i) {
    out[7 - i] = static_cast<char>(us & 0xff);
    us >>= 8;
  }
  return out;
}

void
stop_aware_sleep_(RuntimeContext&            ctx,
                  std::chrono::milliseconds  total)
{
  using namespace std::chrono;
  auto deadline = steady_clock::now() + total;
  constexpr auto kChunk = milliseconds(50);
  while (true) {
    if (ctx.stop_requested()) { return; }
    auto now = steady_clock::now();
    if (now >= deadline) { return; }
    auto remaining = deadline - now;
    std::this_thread::sleep_for(remaining < kChunk ? remaining
                                                   : kChunk);
  }
}

}

VideosDbCleanupStage::VideosDbCleanupStage(
    const SessionContextIntf* s,
    std::string               id,
    std::vector<InEdge>       iports,
    FlexData                  config)
  : TypedStage<VideosDbCleanupStage>(s, std::move(id),
                                     std::move(iports),
                                     std::move(config))
{
  // Validation is deferred to launch (see Stage::fail_config).
  // Attribute defaults live in kSpec.attrs; attr_* resolves the
  // configured value else that default.
  _camera_name      = attr_str("camera_name");
  _videos_db_suffix = attr_str("videos_db_suffix");
  {
    uint64_t v = attr_uint("retention_seconds");
    if (v == 0) {
      fail_config(fmt(
          "VideosDbCleanupStage('{}'): retention_seconds must be > 0",
          this->id()));
    } else {
      _retention = std::chrono::seconds(v);
    }
  }
  {
    uint64_t v = attr_uint("sweep_interval_seconds");
    if (v == 0) { v = 1; }
    _sweep_interval = std::chrono::seconds(v);
  }
  _run_once = attr_bool("run_once");

  if (_camera_name.empty()) {
    fail_config(fmt(
        "VideosDbCleanupStage('{}'): config.camera_name is required",
        this->id()));
  }

  allocate_oports(spec().oports.size());
}

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "camera_name", .type = ConfigType::String, .required = true,
   .doc = "selects the <camera_name><suffix> sub-db"},
  {.key = "videos_db_suffix", .type = ConfigType::String,
   .doc = "suffix appended to camera_name", .def_str = "-videos"},
  {.key = "retention_seconds", .type = ConfigType::Uint,
   .doc = "delete records older than this; >0", .def_uint = 86400},
  {.key = "sweep_interval_seconds", .type = ConfigType::Uint,
   .doc = "wall-clock seconds between sweeps; >=1", .def_uint = 3600},
  {.key = "run_once", .type = ConfigType::Bool,
   .doc = "sweep once then signal done", .def_bool = true},
};
const StageSpec kSpec = {
  .type_name = "videos-db-cleanup",
  .doc       = "Source: periodically sweeps the <camera><suffix> LMDB "
               "segment index and deletes records older than "
               "retention_seconds. Pure LMDB side-effect; 0 in / 0 out.",
  .display_name = "Video Cleanup",
  .category  = StageCategory::Database,
  .iports    = {},
  .oports    = {},
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
VideosDbCleanupStage::spec() const noexcept
{
  return kSpec;
}

std::size_t
VideosDbCleanupStage::sweep_once()
{
  LmdbEnv* env = session()->lmdb_env();
  if (!env) {
    session()->warn(fmt(
        "VideosDbCleanupStage('{}'): session lmdb_env() unavailable; "
        "skipping sweep", this->id()));
    return 0;
  }

  const std::string videos_db_name =
      _camera_name + _videos_db_suffix;
  const auto now = std::chrono::system_clock::now();
  const auto cutoff_tp =
      now - std::chrono::duration_cast<
                std::chrono::system_clock::duration>(_retention);
  const std::string cutoff_key = be64_us_key_(cutoff_tp);

  std::vector<std::string> to_delete;
  try {
    LmdbDb  db(*env, videos_db_name);
    // Read pass: collect keys to delete (cursor visits in
    // ascending order; once we hit a key >= cutoff every later key
    // is younger and can be skipped).
    {
      LmdbTxn txn(*env, LmdbTxn::Mode::ReadOnly);
      LmdbCursor cur(txn, db);
      std::string_view k, v;
      if (cur.first(k, v)) {
        do {
          if (k >= std::string_view(cutoff_key)) { break; }
          to_delete.emplace_back(k);  // copy bytes out of LMDB memory
        } while (cur.next(k, v));
      }
    }
    if (to_delete.empty()) {
      return 0;
    }
    // Write pass: delete each collected key.
    LmdbTxn txn(*env, LmdbTxn::Mode::ReadWrite);
    std::size_t deleted = 0;
    for (const auto& k : to_delete) {
      if (db.del(txn, k)) {
        ++deleted;
      }
    }
    txn.commit();
    return deleted;
  } catch (const std::exception& e) {
    session()->warn(fmt(
        "VideosDbCleanupStage('{}'): sweep of '{}' threw: {}",
        this->id(), videos_db_name, e.what()));
    return 0;
  } catch (...) {
    session()->warn(fmt(
        "VideosDbCleanupStage('{}'): sweep of '{}' threw unknown",
        this->id(), videos_db_name));
    return 0;
  }
}

Job
VideosDbCleanupStage::process(RuntimeContext& ctx)
{
  using namespace std::chrono;

  if (ctx.stop_requested()) {
    ctx.signal_done();
    co_return;
  }

  // Wait first on every call EXCEPT the first one, so the operator
  // sees an immediate sweep on launch and confirmation that the
  // stage is alive.
  if (_did_first_sweep) {
    stop_aware_sleep_(ctx,
        duration_cast<milliseconds>(_sweep_interval));
    if (ctx.stop_requested()) {
      ctx.signal_done();
      co_return;
    }
  }

  const std::string videos_db_name =
      _camera_name + _videos_db_suffix;
  session()->info(fmt(
      "VideosDbCleanupStage('{}'): sweeping '{}' (retention={}s)",
      this->id(), videos_db_name,
      static_cast<long long>(_retention.count())));

  std::size_t deleted = sweep_once();
  _did_first_sweep = true;

  session()->info(fmt(
      "VideosDbCleanupStage('{}'): sweep of '{}' deleted {} "
      "record(s)",
      this->id(), videos_db_name, deleted));
  
  if (_run_once) {
    ctx.signal_done();
    co_return;
  }
}

VPIPE_REGISTER_STAGE(VideosDbCleanupStage)
VPIPE_REGISTER_SPEC(VideosDbCleanupStage, kSpec)

}

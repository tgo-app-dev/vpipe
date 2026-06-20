#ifndef VPIPE_STAGES_VIDEOS_DB_CLEANUP_STAGE_H
#define VPIPE_STAGES_VIDEOS_DB_CLEANUP_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace vpipe {

// Source stage: 0 inputs, 0 outputs. Periodically sweeps the
// <camera_name><videos_db_suffix> LMDB sub-db (the segment index
// rtsp-capture writes) and deletes records whose start_utc is
// older than `retention_seconds`. Pure side-effect on LMDB; emits
// no Beats.
//
// Key insight: the records are keyed by 8-byte big-endian
// microseconds-since-epoch, so a forward cursor walk visits them
// in time order. Once we hit a key >= the cutoff, every later key
// is younger and can be skipped.
//
// Config (FlexData object):
//   camera_name             (string, required) -- selects the
//                            <camera_name><videos_db_suffix> sub-db.
//   videos_db_suffix        (string, default "-videos")
//   retention_seconds       (uint, default 86400 = 1 day) --
//                            entries with start_utc older than
//                            now - retention_seconds are deleted.
//   sweep_interval_seconds  (uint, default 3600 = 1 hour) -- wall-
//                            clock interval between sweeps. The
//                            first sweep runs immediately on
//                            launch; subsequent sweeps wait this
//                            long. Bounded to >= 1s.
//
// Stop is honoured promptly: the inter-sweep wait is broken into
// 50ms chunks that poll stop_requested(). The sweep itself runs
// inside a single LMDB write txn -- ranges large enough to take
// noticeably long should configure sweep_interval_seconds
// accordingly so the writer mutex doesn't stay held.
class VideosDbCleanupStage final
  : public TypedStage<VideosDbCleanupStage>
{
public:
  static constexpr const char* kTypeName = "videos-db-cleanup";

  VideosDbCleanupStage(const SessionContextIntf* session,
                       std::string               id,
                       std::vector<InEdge>       iports,
                       FlexData                  config);

  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only inspectors.
  const std::string& camera_name() const noexcept
  { return _camera_name; }
  const std::string& videos_db_suffix() const noexcept
  { return _videos_db_suffix; }
  std::chrono::seconds retention() const noexcept
  { return _retention; }
  std::chrono::seconds sweep_interval() const noexcept
  { return _sweep_interval; }

  // Test seam: run a single sweep against the current session's
  // LmdbEnv and return the number of records deleted. Same logic
  // process() drives on its timer. Errors (missing env, txn
  // failures) get logged and produce a 0 return.
  std::size_t sweep_once();

private:

  // Config attributes; defaults live in kSpec.attrs and are read in the
  // constructor via attr_*. Declarations carry no non-zero default.
  std::string          _camera_name;
  std::string          _videos_db_suffix;
  std::chrono::seconds _retention{};
  std::chrono::seconds _sweep_interval{};
  bool                 _did_first_sweep = false;
  bool                 _run_once{};
};

}

#endif

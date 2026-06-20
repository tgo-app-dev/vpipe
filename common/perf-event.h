#ifndef PERF_EVENT_H
#define PERF_EVENT_H

#include <cstdint>

namespace vpipe {

// One performance-trace record. Fixed size, POD, written into a
// per-worker PerfBuffer by Session::record_perf_event(). Designed
// for minimum producer-side cost: a single 24-byte slot store with
// no allocation, no formatting, no string handling.
//
// Field summary:
//   ns         -- delta from the session's profiling anchor, in
//                 nanoseconds. The anchor is captured by
//                 Session::enable_profiling() (steady_clock); a
//                 companion system_clock reading is also recorded
//                 in the dump for cross-correlation. uint64 is wide
//                 enough for any practical session.
//   value      -- optional uint64 payload chosen by the stage --
//                 a frame index, byte count, in-flight queue depth,
//                 etc. 0 if unused.
//   stage_gvid -- graph-vertex id of the stage that recorded the
//                 event. The dump uses this to group events by
//                 stage and to resolve the type id to a name via
//                 Stage::perf_event_name().
//   type       -- a stage-local event type id chosen by the stage.
//                 Translated to a string at dump time. Values are
//                 local to the stage instance.
//
// worker_id is NOT stored here -- it is implicit in the PerfBuffer
// index that holds the event. The dump emits it as a separate
// column computed from the buffer index.
struct PerfEvent {
  uint64_t ns;
  uint64_t value;
  uint32_t stage_gvid;
  uint32_t type;
};

static_assert(sizeof(PerfEvent) == 24, "PerfEvent must be 24 bytes");

// ---- Event-type id convention -------------------------------------
//
// A `type` id is interpreted by parity so begin/end pairs can be drawn
// as one block:
//   * EVEN id ((type & 1) == 0):
//       - a TRANSIENT (instantaneous) event when it has no matching
//         odd partner, OR
//       - the BEGIN of a duration event.
//   * ODD id ((type & 1) == 1):
//       - the END of the duration event whose begin id is (type - 1).
// So a duration event is the pair (even N = begin, odd N+1 = end); a
// transient event is a lone even id. A visualizer draws a paired
// begin/end as one block and a lone even id as a point.
inline constexpr bool
perf_type_is_end(std::uint32_t type) noexcept
{
  return (type & 1u) != 0u;
}
inline constexpr std::uint32_t
perf_type_begin_of(std::uint32_t end_type) noexcept
{
  return end_type - 1u;
}

// Reserved type ids emitted by the pipeline runtime (the ThreadPool)
// itself, bracketing each worker resume-slice: the worker records a
// "schedule" (begin) right before resuming a coroutine and an
// "unschedule" (end) right after it suspends or finishes. Both records
// land in the SAME worker's buffer -- a resume and its return happen on
// one thread -- so each thread shows clean schedule->unschedule pairs.
// Picked high (and begin even / end odd per the convention) so they
// never collide with a stage's own small, stage-local type ids.
inline constexpr std::uint32_t kPerfRuntimeBase = 0xFFFF0000u;
inline constexpr std::uint32_t kPerfSchedule    = kPerfRuntimeBase + 0u;
inline constexpr std::uint32_t kPerfUnschedule  = kPerfRuntimeBase + 1u;

// ---- Auxiliary (non-worker) perf lanes ----------------------------
//
// Beyond the per-worker buffers + the overflow buffer, the session
// keeps a fixed set of AUXILIARY lanes: logical activity timelines
// that are NOT pipeline worker threads. They are recorded via
// SessionContextIntf::record_perf_event_aux(lane, gvid, type, value)
// from any thread -- the LLM MLX-runtime worker, CoreML callback
// threads, or a pool worker -- routed by lane id, not by the calling
// thread. This is what lets LLM forward-pass activity (which runs on a
// single dedicated worker, not a pool worker) and Apple-Neural-Engine
// jobs show up as their own dedicated timelines instead of colliding
// in the shared overflow lane. The PerfAuxScope RAII helper
// (common/perf-scope.h) is the ergonomic way to bracket a begin/end.
enum PerfAuxLane : unsigned {
  kPerfLaneLLM = 0u,   // LLM/VLM forward-pass activity
  kPerfLaneANE = 1u,   // Apple Neural Engine (CoreML) jobs
  kPerfAuxLaneCount = 2u,
};

// Human label per aux lane (dump_profiling uses it for the lane name).
inline constexpr const char* kPerfAuxLaneName[kPerfAuxLaneCount] = {
  "LLM", "ANE",
};

// Synthetic "stage" gvids naming the LLM-lane activities (the ANE lane
// reuses the real CoreML stage's gvid, so it self-names). Picked high
// so they never collide with real graph-vertex ids (small indices).
inline constexpr std::uint32_t kPerfAuxGvidBase = 0xF0000000u;
inline constexpr std::uint32_t kGvidLlmPrefill  = kPerfAuxGvidBase + 0u;
inline constexpr std::uint32_t kGvidLlmDecode   = kPerfAuxGvidBase + 1u;
inline constexpr std::uint32_t kGvidLlmVision   = kPerfAuxGvidBase + 2u;
inline constexpr std::uint32_t kGvidLlmAudio    = kPerfAuxGvidBase + 3u;

// Per-activity begin type ids (end = begin + 1, per the parity
// convention). Distinct pairs so the begin/end pairing is unambiguous
// even though every LLM activity shares the one LLM lane.
inline constexpr std::uint32_t kPerfLlmPrefillBegin = 0u;
inline constexpr std::uint32_t kPerfLlmDecodeBegin  = 2u;
inline constexpr std::uint32_t kPerfLlmVisionBegin  = 4u;
inline constexpr std::uint32_t kPerfLlmAudioBegin   = 6u;

// The LLM-lane synthetic stages, consumed by dump_profiling to emit
// `stages` entries (id = block label; begin/end named in the dump).
struct PerfAuxStageDesc {
  std::uint32_t gvid;
  std::uint32_t begin_type;
  const char*   id;
};
inline constexpr PerfAuxStageDesc kPerfAuxStages[] = {
  { kGvidLlmPrefill, kPerfLlmPrefillBegin, "text-prefill"  },
  { kGvidLlmDecode,  kPerfLlmDecodeBegin,  "text-decode"   },
  { kGvidLlmVision,  kPerfLlmVisionBegin,  "vision-tower"  },
  { kGvidLlmAudio,   kPerfLlmAudioBegin,   "audio-encoder" },
};

// ANE-lane CoreML predict begin/end (recorded with the real CoreML
// stage's gvid so the block is named/colored by the actual stage).
inline constexpr std::uint32_t kPerfAnePredictBegin = 0u;

}

#endif

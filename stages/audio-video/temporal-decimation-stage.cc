#include "stages/audio-video/temporal-decimation-stage.h"

#include "apple-silicon/metal-compute/image-ops.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/tensor-beat.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "pipeline/runtime-context.h"

#include <algorithm>
#include <cstring>
#include <utility>

namespace vpipe {

namespace {

// Parse a JSON array of mixed int / string entries into the focus
// class sets. Values that look like integers go into `ids`; values
// that look like strings go into `names`. Anything else is logged
// and skipped.
void
parse_focus_classes_(const FlexData&                   arr,
                     const SessionContextIntf*         session,
                     const std::string&                stage_id,
                     std::unordered_set<int>*          ids,
                     std::unordered_set<std::string>*  names)
{
  if (!arr.is_array()) { return; }
  auto view = arr.as_array();
  for (std::size_t i = 0; i < view.size(); ++i) {
    const FlexData& v = view.at(i);
    if (v.is_int() || v.is_uint()) {
      ids->insert(static_cast<int>(v.as_int(-1)));
    } else if (v.is_string()) {
      names->insert(std::string(v.as_string("")));
    } else if (session) {
      session->warn(fmt(
          "temporal-decimation('{}'): focus_classes[{}] is not an "
          "int or string; ignoring entry",
          stage_id, i));
    }
  }
}

// Sum a per-tile u8 SAD heatmap, optionally restricting the sum to
// tile cells whose center falls inside any focus-class detection
// bbox. Returns
//     total_motion       = sum(diff) / (255 * tile_w * tile_h)
//     focus_motion_share = sum_focus / sum_total   (0 when no
//                                                   focus class hit
//                                                   or total == 0)
// `out_n_dets` (when non-null) receives the total detection count in
// det_fd; `out_n_focus` (when non-null) receives the count of those
// detections whose class matched a focus_class entry.
void
reduce_diff_heatmap_(const std::uint8_t*                     diff,
                     int                                     tile_w,
                     int                                     tile_h,
                     int                                     frame_w,
                     int                                     frame_h,
                     const FlexData*                         det_fd,
                     const std::unordered_set<int>&          focus_ids,
                     const std::unordered_set<std::string>&  focus_names,
                     double*                                 out_total,
                     double*                                 out_focus_share,
                     int*                                    out_n_dets,
                     int*                                    out_n_focus)
{
  std::uint64_t sum_total = 0;
  std::uint64_t sum_focus = 0;
  const int N = tile_w * tile_h;
  int n_dets  = 0;
  int n_focus = 0;

  // Detection bbox set in source-pixel coords.
  struct Box { float x1, y1, x2, y2; };
  std::vector<Box> focus_boxes;
  if (det_fd && det_fd->is_object()) {
    auto root = det_fd->as_object();
    if (root.contains("detections")) {
      const FlexData& dets = root.at("detections");
      if (dets.is_array()) {
        auto arr = dets.as_array();
        n_dets = static_cast<int>(arr.size());
        if (!focus_ids.empty() || !focus_names.empty()) {
          focus_boxes.reserve(arr.size());
          for (std::size_t i = 0; i < arr.size(); ++i) {
            const FlexData& d = arr.at(i);
            if (!d.is_object()) { continue; }
            auto obj = d.as_object();
            bool hit = false;
            if (!focus_ids.empty() && obj.contains("class_id")) {
              int cid = static_cast<int>(obj.at("class_id").as_int(-1));
              if (focus_ids.find(cid) != focus_ids.end()) {
                hit = true;
              }
            }
            if (!hit && !focus_names.empty()
                && obj.contains("class_name")) {
              std::string cn(obj.at("class_name").as_string(""));
              if (focus_names.find(cn) != focus_names.end()) {
                hit = true;
              }
            }
            if (!hit) { continue; }
            ++n_focus;
            Box b{
                static_cast<float>(obj.contains("x1")
                    ? obj.at("x1").as_real(0.0) : 0.0),
                static_cast<float>(obj.contains("y1")
                    ? obj.at("y1").as_real(0.0) : 0.0),
                static_cast<float>(obj.contains("x2")
                    ? obj.at("x2").as_real(0.0) : 0.0),
                static_cast<float>(obj.contains("y2")
                    ? obj.at("y2").as_real(0.0) : 0.0),
            };
            if (b.x2 > b.x1 && b.y2 > b.y1) {
              focus_boxes.push_back(b);
            }
          }
        }
      }
    }
  }
  if (out_n_dets)  { *out_n_dets  = n_dets;  }
  if (out_n_focus) { *out_n_focus = n_focus; }

  const float tile_px_w =
      static_cast<float>(frame_w) / static_cast<float>(tile_w);
  const float tile_px_h =
      static_cast<float>(frame_h) / static_cast<float>(tile_h);

  for (int ty = 0; ty < tile_h; ++ty) {
    for (int tx = 0; tx < tile_w; ++tx) {
      const std::uint64_t v =
          static_cast<std::uint64_t>(diff[ty * tile_w + tx]);
      sum_total += v;
      if (focus_boxes.empty()) { continue; }
      const float cx = (static_cast<float>(tx) + 0.5f) * tile_px_w;
      const float cy = (static_cast<float>(ty) + 0.5f) * tile_px_h;
      for (const auto& b : focus_boxes) {
        if (cx >= b.x1 && cx <= b.x2
            && cy >= b.y1 && cy <= b.y2) {
          sum_focus += v;
          break;
        }
      }
    }
  }

  const double denom_total =
      255.0 * static_cast<double>(N);
  *out_total = (denom_total > 0.0)
      ? static_cast<double>(sum_total) / denom_total
      : 0.0;
  if (sum_total == 0) {
    *out_focus_share = 0.0;
  } else {
    *out_focus_share =
        static_cast<double>(sum_focus) / static_cast<double>(sum_total);
  }
}

}  // namespace

TemporalDecimationStage::TemporalDecimationStage(
    const SessionContextIntf* s,
    std::string               id,
    std::vector<InEdge>       iports,
    FlexData                  cfg)
  : TypedStage<TemporalDecimationStage>(s, std::move(id),
                                        std::move(iports),
                                        std::move(cfg))
{
  allocate_oports(spec().oports.size());

  // Scalar attribute defaults live in kSpec.attrs; attr_* resolves the
  // configured value else that default.
  _max_avg_fps       = attr_real("max_avg_fps");
  _max_consec_drops  =
      static_cast<unsigned>(attr_uint("max_consecutive_drops"));
  _tile_w            = static_cast<int>(attr_int("tile_w"));
  _tile_h            = static_cast<int>(attr_int("tile_h"));
  _motion_threshold  = attr_real("motion_threshold");
  _focus_motion_gain = attr_real("focus_motion_gain");
  _bucket_capacity   = attr_real("bucket_capacity");

  const FlexData& root = this->config();
  if (root.is_object()) {
    auto obj = root.as_object();
    // "target_fps" is a legacy alias used only when "max_avg_fps" (the
    // canonical key, whose default is in kSpec) was not supplied.
    if (!obj.contains("max_avg_fps") && obj.contains("target_fps")) {
      _max_avg_fps = obj.at("target_fps").as_real(_max_avg_fps);
    }
    if (obj.contains("focus_classes")) {
      parse_focus_classes_(obj.at("focus_classes"),
                           session(),
                           this->id(),
                           &_focus_class_ids,
                           &_focus_class_names);
    }
  }

  // Clamps repair out-of-range overrides.
  if (_max_avg_fps <= 0.0)    { _max_avg_fps = 30.0; }
  if (_tile_w < 1)            { _tile_w = 1; }
  if (_tile_h < 1)            { _tile_h = 1; }
  if (_tile_w > 1024)         { _tile_w = 1024; }
  if (_tile_h > 1024)         { _tile_h = 1024; }
  if (_bucket_capacity < 1.0) { _bucket_capacity = 1.0; }
}

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "max_avg_fps", .type = ConfigType::Real,
   .doc = "ceiling keep rate; sub-1 allowed", .def_real = 30.0},
  {.key = "target_fps", .type = ConfigType::Real,
   .doc = "legacy alias for max_avg_fps", .def_real = 30.0},
  {.key = "max_consecutive_drops", .type = ConfigType::Uint,
   .doc = "force-keep floor: max drops in a row", .def_uint = 5},
  {.key = "tile_w", .type = ConfigType::Int,
   .doc = "motion-signature tile cols, clamped [1,1024]",
   .def_int = 32},
  {.key = "tile_h", .type = ConfigType::Int,
   .doc = "motion-signature tile rows, clamped [1,1024]",
   .def_int = 18},
  {.key = "motion_threshold", .type = ConfigType::Real,
   .doc = "priority bar above which a frame may keep", .def_real = 0.02},
  {.key = "focus_motion_gain", .type = ConfigType::Real,
   .doc = "weight on focus-class motion share", .def_real = 4.0},
  {.key = "bucket_capacity", .type = ConfigType::Real,
   .doc = "token-bucket burst cap, clamped >= 1", .def_real = 2.0},
  {.key = "focus_classes", .type = ConfigType::Array,
   .doc = "mixed int class_id / string class_name focus list"},
};
const PortSpec kIports[] = {
  {.name = "frames", .doc = "planar u8 RGB TensorBeat [3,H,W]",
   .type = &typeid(TensorBeatPayload),
   .tags = "rgb-frames", .clock_group = 0},
  {.name = "detections", .doc = "optional FlexData YOLO detections, read "
                                "1:1 with frames",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "frames", .doc = "kept frames (forwarded TensorBeat); dropped "
                            "frames produce no output",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "temporal-decimation",
  .doc       = "Token-bucket frame dropper: keeps frames up to an average "
               "FPS ceiling, biased toward motion (and focus-class motion "
               "when detections are wired on iport1). Forwards kept frames.",
  .display_name = "Frame Dropper",
  .category  = StageCategory::Visual,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
TemporalDecimationStage::spec() const noexcept
{
  return kSpec;
}

TemporalDecimationStage::~TemporalDecimationStage() = default;

Job
TemporalDecimationStage::initialize(RuntimeContext& ctx)
{
  if (session()) {
    _mc = session()->metal_compute();
  }
  _has_iport1 = (ctx.num_iports() >= 2);
  if (!_mc || !_mc->valid()) {
    if (session()) {
      session()->warn(fmt(
          "temporal-decimation('{}'): metal-compute unavailable; the "
          "motion-score kernels are disabled and every frame will be "
          "forwarded as-is", this->id()));
    }
  }
  // Config banner: print the resolved knobs so users can verify
  // what's actually wired up. Useful when chasing "why is everything
  // dropping" / "why is nothing dropping" misconfigurations.
  if (session()) {
    std::string focus_summary;
    if (_focus_class_ids.empty() && _focus_class_names.empty()) {
      focus_summary = "(none)";
    } else {
      focus_summary = "ids=[";
      bool first = true;
      for (int id : _focus_class_ids) {
        if (!first) { focus_summary += ","; }
        focus_summary += std::to_string(id);
        first = false;
      }
      focus_summary += "] names=[";
      first = true;
      for (const auto& n : _focus_class_names) {
        if (!first) { focus_summary += ","; }
        focus_summary += "\"";
        focus_summary += n;
        focus_summary += "\"";
        first = false;
      }
      focus_summary += "]";
    }
    session()->info(fmt(
        "temporal-decimation('{}'): config max_avg_fps={:.2f}, "
        "motion_threshold={:.4f}, focus_motion_gain={:.2f}, "
        "bucket_capacity={:.2f}, max_consecutive_drops={}, "
        "tile={}x{}, iport1={}, focus_classes={}, metal={}",
        this->id(), _max_avg_fps, _motion_threshold,
        _focus_motion_gain, _bucket_capacity, _max_consec_drops,
        _tile_w, _tile_h,
        _has_iport1 ? "wired" : "unwired",
        focus_summary,
        (_mc && _mc->valid()) ? "ok" : "unavailable"));
  }
  // No edge writes on initialize.
  co_return;
}

bool
TemporalDecimationStage::ensure_buffers_(int in_w, int in_h)
{
  if (!_mc || !_mc->valid()) { return false; }
  const std::size_t sig_bytes =
      static_cast<std::size_t>(_tile_w)
    * static_cast<std::size_t>(_tile_h);
  const std::size_t src_bytes =
      static_cast<std::size_t>(3)
    * static_cast<std::size_t>(in_w)
    * static_cast<std::size_t>(in_h);

  if (!_sig_a) {
    _sig_a = metal_compute::make_shared_storage(*_mc, sig_bytes, session());
    _sig_b = metal_compute::make_shared_storage(*_mc, sig_bytes, session());
    _diff  = metal_compute::make_shared_storage(*_mc, sig_bytes, session());
    if (!_sig_a || !_sig_b || !_diff) {
      _sig_a.reset(); _sig_b.reset(); _diff.reset();
      return false;
    }
    _sig_cur  = _sig_a.get();
    _sig_prev = _sig_b.get();
  }

  if (!_src_stage || _src_w != in_w || _src_h != in_h) {
    _src_stage =
        metal_compute::make_shared_storage(*_mc, src_bytes, session());
    if (!_src_stage) { return false; }
    _src_w = in_w;
    _src_h = in_h;
    _have_prev = false;   // dims changed; invalidate prev signature.
  }
  return true;
}

bool
TemporalDecimationStage::score_frame_(
    const std::uint8_t* src_bytes,
    std::size_t         src_bytes_len,
    int                 in_w,
    int                 in_h,
    const FlexData*     det_fd,
    double*             out_total_motion,
    double*             out_focus_share,
    int*                out_n_dets,
    int*                out_n_focus)
{
  *out_total_motion = 0.0;
  *out_focus_share  = 0.0;
  if (out_n_dets)  { *out_n_dets  = 0; }
  if (out_n_focus) { *out_n_focus = 0; }
  if (!ensure_buffers_(in_w, in_h)) { return false; }

  const std::size_t need =
      static_cast<std::size_t>(3)
    * static_cast<std::size_t>(in_w)
    * static_cast<std::size_t>(in_h);
  if (src_bytes_len < need) { return false; }

  // Stage the source bytes into the Shared MTL::Buffer. On Apple-
  // Silicon UMA this is just a host-side memcpy; the GPU sees the
  // bytes once the kernel is dispatched.
  std::memcpy(_src_stage->contents, src_bytes, need);

  // 1. Downsample to a u8 luma signature.
  if (!metal_compute::motion_signature_u8(
          *_mc, *_src_stage, in_w, in_h,
          *_sig_cur, _tile_w, _tile_h, session())) {
    return false;
  }

  // First frame: no prior signature -> no motion score available.
  // Caller treats `false` from score_frame_ as "force keep".
  if (!_have_prev) {
    // Make this signature the prev for the next call (no swap; just
    // remember we have a baseline).
    std::swap(_sig_cur, _sig_prev);
    _have_prev = true;
    return false;
  }

  // 2. Per-tile abs diff vs prev.
  if (!metal_compute::motion_diff_u8(
          *_mc, *_sig_cur, *_sig_prev, *_diff, _tile_w, _tile_h,
          session())) {
    std::swap(_sig_cur, _sig_prev);
    return false;
  }

  // 3. CPU reduction over the (small) diff buffer; integrate inside
  //    focus-class bboxes if any are present.
  reduce_diff_heatmap_(
      _diff->contents, _tile_w, _tile_h, in_w, in_h,
      det_fd, _focus_class_ids, _focus_class_names,
      out_total_motion, out_focus_share,
      out_n_dets, out_n_focus);

  // 4. Ping-pong: this frame's signature becomes prev for the next.
  std::swap(_sig_cur, _sig_prev);
  return true;
}

bool
TemporalDecimationStage::decide_keep_for_test(
    double        total_motion,
    double        focus_motion_share,
    std::uint64_t timestamp_us)
{
  // Token bucket refill from inter-frame interval. When we don't
  // have a previous timestamp, give the bucket one frame worth of
  // tokens so the first frame is always kept (matches the timestamp-
  // present path: the first frame's score_frame_ returns false and
  // the caller force-keeps).
  double refill;
  if (_have_prev_ts && timestamp_us > _prev_ts_us) {
    const double dt =
        static_cast<double>(timestamp_us - _prev_ts_us) * 1e-6;
    refill = dt * _max_avg_fps;
  } else {
    // No previous timestamp: refill by one frame's worth so the
    // first call has at least one token to spend.
    refill = 1.0;
  }
  _tokens += refill;
  if (_tokens > _bucket_capacity) {
    _tokens = _bucket_capacity;
  }
  _last_refill  = refill;
  _prev_ts_us   = timestamp_us;
  _have_prev_ts = true;

  const double priority =
      total_motion + _focus_motion_gain * focus_motion_share;
  const bool   force_keep =
      _consec_drops >= _max_consec_drops;
  const bool   above_thresh =
      priority >= _motion_threshold;

  // Policy: max_avg_fps is a CEILING, not a target. We never
  // force-keep just to drain the bucket — when motion is below the
  // threshold the bucket simply accumulates (capped at
  // bucket_capacity) and the keep rate falls naturally below
  // max_avg_fps, governed only by max_consecutive_drops. A keep
  // requires (a) motion above threshold, (b) at least one token
  // available, or (c) the consecutive-drops cap has fired.
  bool keep = false;
  if (force_keep) {
    keep = true;
  } else if (above_thresh && _tokens >= 1.0) {
    keep = true;
  }

  if (keep) {
    _tokens = std::max(0.0, _tokens - 1.0);
    _consec_drops = 0;
    ++_kept_count;
  } else {
    ++_consec_drops;
    ++_dropped_count;
  }
  return keep;
}

Job
TemporalDecimationStage::process(RuntimeContext& ctx)
{
  auto in0 = co_await ctx.read(0);
  if (!in0) {
    ctx.signal_done();
    co_return;
  }
  std::unique_ptr<BeatPayloadIntf> in1;
  if (ctx.num_iports() >= 2) {
    auto p = co_await ctx.read(1);
    if (!p) {
      ctx.signal_done();
      co_return;
    }
    in1 = std::move(p);
  }

  const auto* tin = dynamic_cast<const TensorBeatPayload*>(in0.get());
  if (!tin
      || tin->dtype != TensorBeat::DType::U8
      || tin->shape.size() != 3
      || tin->shape[0] != 3) {
    if (session()) {
      session()->warn(fmt(
          "temporal-decimation('{}'): iport0 must be a u8 [3, H, W] "
          "TensorBeat; forwarding frame as-is", this->id()));
    }
    co_await ctx.write(0, std::move(in0));
    co_return;
  }

  // Frame dims and timestamp.
  const int in_h = static_cast<int>(tin->shape[1]);
  const int in_w = static_cast<int>(tin->shape[2]);
  std::uint64_t ts_us = 0;
  bool ts_present = false;
  if (tin->sideband.is_object()) {
    auto sb = tin->sideband.as_object();
    if (sb.contains("timestamp_us")) {
      ts_us = sb.at("timestamp_us").as_uint(0);
      ts_present = true;
    }
  }

  // Detection FlexData (optional).
  const FlexData* det_fd = nullptr;
  if (in1) {
    if (auto* fdp = dynamic_cast<const FlexDataPayload*>(in1.get())) {
      det_fd = &fdp->data;
    }
  }

  // Score the frame. score_frame_ returns false on the first frame
  // (no prior signature) and on any GPU dispatch failure; in both
  // cases we force-keep so the pipeline keeps moving and we get a
  // baseline signature established.
  double total_motion = 0.0;
  double focus_share  = 0.0;
  int    n_dets       = 0;
  int    n_focus      = 0;
  const bool have_score = score_frame_(
      tin->as_u8(),
      tin->byte_size() - static_cast<std::size_t>(tin->storage_offset),
      in_w, in_h, det_fd,
      &total_motion, &focus_share, &n_dets, &n_focus);

  // Decide keep / drop.
  bool keep;
  // Pacing hint: when the input sideband has no timestamp_us, fake
  // an interval of 1/target_fps so the bucket refills at exactly
  // target_fps per call. Match the test-harness path so both routes
  // converge on the same per-frame state.
  if (!ts_present) {
    ts_us = _prev_ts_us
          + static_cast<std::uint64_t>(1e6 / _max_avg_fps);
  }
  // Snapshot state BEFORE decide_keep_for_test so the debug log can
  // show the pre-refill values that drove the decision.
  const std::uint64_t pre_idx        = _kept_count + _dropped_count;
  const double        pre_tokens     = _tokens;
  const unsigned      pre_consec     = _consec_drops;
  const double        priority       =
      total_motion + _focus_motion_gain * focus_share;
  if (!have_score) {
    // First frame (or kernel unavailable): always keep. Run the
    // refill bookkeeping via the same helper so _last_refill /
    // _prev_ts_us / _have_prev_ts stay consistent with the rest of
    // the loop. Then spend one token explicitly.
    (void)decide_keep_for_test(/*total_motion=*/1.0,
                               /*focus_motion_share=*/0.0,
                               ts_us);
    // decide_keep_for_test already updated _kept_count / _consec_drops
    // / _tokens. We pass a high motion score so the keep is taken
    // through the above_thresh branch, which mirrors what an
    // unmissable high-priority frame would have produced.
    keep = true;
  } else {
    keep = decide_keep_for_test(total_motion, focus_share, ts_us);
  }

  // Reason classification for the debug log. Mirrors the policy
  // inside decide_keep_for_test so the user can spot-check why a
  // given frame went the way it did.
  const char* reason;
  if (!have_score) {
    reason = "first_or_kernel_miss";
  } else if (pre_consec >= _max_consec_drops) {
    reason = "force_keep_consec_cap";
  } else if (priority < _motion_threshold) {
    reason = "drop_below_thresh";
  } else if (pre_tokens + _last_refill < 1.0) {
    reason = "drop_no_tokens";
  } else if (keep) {
    reason = "keep_motion_and_tokens";
  } else {
    reason = "drop_other";
  }

  // Per-frame debug log. Gated by the session's debug level so it's
  // off in production but kicks in once the user sets debug_level
  // ("debug" or "verbose").
  if (session()) {
    session()->log_debug(fmt(
        "temporal-decimation('{}'): frame {} {}x{} ts={} "
        "motion={:.4f} focus_share={:.4f} prio={:.4f} thr={:.4f} "
        "dets={} focus_dets={} tokens={:.3f}->{:.3f} refill={:.3f} "
        "consec={} -> {} ({})",
        this->id(), pre_idx, in_w, in_h, ts_us,
        total_motion, focus_share, priority, _motion_threshold,
        n_dets, n_focus, pre_tokens, _tokens, _last_refill,
        pre_consec, keep ? "KEEP" : "DROP", reason));
  }

  // Roll-up window stats for the periodic summary.
  ++_win_frames;
  if (keep) { ++_win_kept; } else { ++_win_dropped; }
  if (have_score) {
    ++_win_with_score;
    _win_motion_sum += total_motion;
    if (total_motion > _win_motion_max) {
      _win_motion_max = total_motion;
    }
    _win_focus_sum  += focus_share;
    if (focus_share  > _win_focus_max)  {
      _win_focus_max  = focus_share;
    }
  }
  _win_tokens_sum    += pre_tokens;
  _win_dets_total    += static_cast<std::uint64_t>(n_dets);
  _win_focus_matches += static_cast<std::uint64_t>(n_focus);

  if (_win_frames >= kSummaryEveryFrames && session()) {
    const double n      = static_cast<double>(_win_frames);
    const double ns     = _win_with_score > 0
        ? static_cast<double>(_win_with_score) : 1.0;
    // Percent printed as f-then-literal-% — std::format (C++20/23)
    // doesn't implement Python-style %-conversion, so `{:.1%}` would
    // throw format_error and surface as "<formatter threw>".
    const double drop_rate_pct =
        100.0 * static_cast<double>(_win_dropped) / n;
    const std::uint64_t total_seen = _kept_count + _dropped_count;
    session()->log_verbose(fmt(
        "temporal-decimation('{}'): window {} frames (total {}): "
        "kept={} dropped={} drop_rate={:.1f}% motion mean={:.4f} "
        "max={:.4f} focus_share mean={:.4f} max={:.4f} "
        "tokens_avg={:.3f} dets_avg={:.1f} focus_dets={}",
        this->id(), _win_frames, total_seen,
        _win_kept, _win_dropped, drop_rate_pct,
        _win_motion_sum / ns, _win_motion_max,
        _win_focus_sum  / ns, _win_focus_max,
        _win_tokens_sum / n,
        static_cast<double>(_win_dets_total) / n,
        _win_focus_matches));
    // Reset window.
    _win_frames        = 0;
    _win_kept          = 0;
    _win_dropped       = 0;
    _win_motion_sum    = 0.0;
    _win_motion_max    = 0.0;
    _win_focus_sum     = 0.0;
    _win_focus_max     = 0.0;
    _win_tokens_sum    = 0.0;
    _win_with_score    = 0;
    _win_dets_total    = 0;
    _win_focus_matches = 0;
  }

  if (!keep) {
    co_return;
  }

  // Augment the sideband and forward the input payload unchanged
  // otherwise. The TensorBeatPayload we received still has its data
  // (Shared MTL::Buffer or AlignedVector); we move it through so
  // downstream consumers can use it zero-copy.
  auto* mut_tin = const_cast<TensorBeatPayload*>(tin);
  if (!mut_tin->sideband.is_object()) {
    mut_tin->sideband = FlexData::make_object();
  }
  auto sb = mut_tin->sideband.as_object();
  if (have_score) {
    sb.insert_or_assign(
        "decim_motion_score", FlexData::make_real(total_motion));
    if (_has_iport1) {
      sb.insert_or_assign(
          "decim_focus_share", FlexData::make_real(focus_share));
    }
  }
  sb.insert_or_assign(
      "decim_consec_drops",
      FlexData::make_uint(static_cast<std::uint64_t>(_consec_drops)));
  sb.insert_or_assign("decim_kept", FlexData::make_bool(true));

  co_await ctx.write(0, std::move(in0));
}

VPIPE_REGISTER_STAGE(TemporalDecimationStage)
VPIPE_REGISTER_SPEC(TemporalDecimationStage, kSpec)

}

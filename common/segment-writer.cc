#include "common/segment-writer.h"
#include "common/ffmpeg-libraries.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <chrono>
#include <cstdio>
#include <ctime>

namespace vpipe {

bool
decide_rollover(bool    is_video_key,
                int64_t elapsed_ms,
                int64_t target_ms)
{
  if (!is_video_key) {
    return false;
  }
  return elapsed_ms >= target_ms;
}

namespace {

// Compose <stem>_<YYYYMMDDTHHMMSSZ>.mp4 from a UTC time-point.
std::string
format_filename(std::string_view stem,
                std::chrono::system_clock::time_point t)
{
  using namespace std::chrono;
  auto secs = time_point_cast<seconds>(t);
  std::time_t tt = system_clock::to_time_t(secs);
  std::tm tm_utc{};
  ::gmtime_r(&tt, &tm_utc);
  char buf[40];
  std::snprintf(buf, sizeof(buf),
                "%04d%02d%02dT%02d%02d%02dZ",
                tm_utc.tm_year + 1900, tm_utc.tm_mon + 1, tm_utc.tm_mday,
                tm_utc.tm_hour, tm_utc.tm_min, tm_utc.tm_sec);
  std::string out;
  out.reserve(stem.size() + 32);
  out.append(stem);
  out.push_back('_');
  out.append(buf);
  out.append(".mp4");
  return out;
}

}

SegmentWriter::SegmentWriter(const FFmpegLibraries*    libs,
                             const SessionContextIntf* session,
                             SegmentSpec               spec,
                             bool                      stream_copy_video,
                             bool                      stream_copy_audio)
  : _libs(libs)
  , _session(session)
  , _spec(std::move(spec))
  , _stream_copy_v(stream_copy_video)
  , _stream_copy_a(stream_copy_audio)
{
}

SegmentWriter::~SegmentWriter()
{
  if (_ofctx) {
    close_current_file_();
  }
}

bool
SegmentWriter::ready() const noexcept
{
  return _ofctx != nullptr && !_closed;
}

bool
SegmentWriter::wants_keyframe() const noexcept
{
  if (!ready()) {
    return false;
  }
  auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - _file_open_steady)
      .count();
  int64_t target_ms =
      static_cast<int64_t>(_spec.target_duration.count()) * 1000;
  return elapsed >= target_ms;
}

bool
SegmentWriter::open(const std::vector<OutputStreamSpec>& streams)
{
  if (streams.empty() || !streams[0].is_video) {
    _session->warn(fmt(
        "SegmentWriter: open() requires a video stream at index 0"));
    return false;
  }
  _streams = streams;
  _out_tbs.assign(_streams.size(), AVRational{1, 90000});
  _anchor_us  = 0;
  _anchor_set = false;
  _closed     = false;
  return open_new_file_();
}

std::string
SegmentWriter::make_filename_() const
{
  std::string path = _spec.output_dir;
  if (!path.empty() && path.back() != '/') {
    path.push_back('/');
  }
  path.append(format_filename(_spec.filename_stem, _file_open_utc));
  return path;
}

bool
SegmentWriter::open_new_file_()
{
  const auto& fmt_api = _libs->avformat().api;
  const auto& cdc_api = _libs->avcodec().api;

  _file_open_utc    = std::chrono::system_clock::now();
  _file_open_steady = std::chrono::steady_clock::now();
  _current_path     = make_filename_();

  AVFormatContext* ofctx = nullptr;
  int rc = fmt_api.alloc_output_context2(&ofctx, nullptr, "mp4",
                                         _current_path.c_str());
  if (rc < 0 || !ofctx) {
    _session->warn(fmt(
        "SegmentWriter: alloc_output_context2 failed ({}) for '{}'",
        rc, _current_path));
    return false;
  }

  // Create output streams matching the requested specs.
  for (size_t i = 0; i < _streams.size(); ++i) {
    AVStream* os = fmt_api.new_stream(ofctx, nullptr);
    if (!os) {
      _session->warn(fmt(
          "SegmentWriter: new_stream failed at idx {}", i));
      fmt_api.free_context(ofctx);
      return false;
    }
    rc = cdc_api.parameters_copy(os->codecpar, _streams[i].codecpar);
    if (rc < 0) {
      _session->warn(fmt(
          "SegmentWriter: parameters_copy failed at idx {}", i));
      fmt_api.free_context(ofctx);
      return false;
    }
    // mp4 prefers tag = 0 so it picks a sensible default per codec.
    os->codecpar->codec_tag = 0;
    // Seed time-base; muxer may rewrite during write_header.
    os->time_base = _streams[i].input_time_base;
  }

  if (!(ofctx->oformat->flags & AVFMT_NOFILE)) {
    rc = fmt_api.avio_open(&ofctx->pb, _current_path.c_str(),
                           AVIO_FLAG_WRITE);
    if (rc < 0) {
      _session->warn(fmt(
          "SegmentWriter: avio_open '{}' failed ({})",
          _current_path, rc));
      fmt_api.free_context(ofctx);
      return false;
    }
  }

  rc = fmt_api.write_header(ofctx, nullptr);
  if (rc < 0) {
    _session->warn(fmt(
        "SegmentWriter: write_header for '{}' failed ({})",
        _current_path, rc));
    if (!(ofctx->oformat->flags & AVFMT_NOFILE)) {
      fmt_api.avio_closep(&ofctx->pb);
    }
    fmt_api.free_context(ofctx);
    return false;
  }

  // Capture the post-header time-bases the muxer settled on.
  for (size_t i = 0; i < _streams.size(); ++i) {
    _out_tbs[i] = ofctx->streams[i]->time_base;
  }
  // The PTS anchor is per-file, not per-stream: cross-stream offsets
  // (audio leading video, or vice versa) must be preserved or A/V
  // sync will drift after rollover.
  _anchor_set = false;
  _anchor_us  = 0;

  _ofctx = ofctx;
  _last_packet_utc = _file_open_utc;
  _session->info(fmt(
      "SegmentWriter: opened '{}'", _current_path));
  return true;
}

void
SegmentWriter::close_current_file_()
{
  if (!_ofctx) {
    return;
  }
  const auto& fmt_api = _libs->avformat().api;
  fmt_api.write_trailer(_ofctx);
  if (!(_ofctx->oformat->flags & AVFMT_NOFILE) && _ofctx->pb) {
    fmt_api.avio_closep(&_ofctx->pb);
  }
  fmt_api.free_context(_ofctx);
  _ofctx = nullptr;
}

bool
SegmentWriter::mux_after_open_(AVPacket* pkt, int out_idx)
{
  if (!_ofctx || out_idx < 0
      || static_cast<size_t>(out_idx) >= _streams.size()) {
    return false;
  }
  const auto& util_api = _libs->avutil().api;
  const auto& fmt_api  = _libs->avformat().api;

  AVRational in_tb  = _streams[out_idx].input_time_base;
  AVRational out_tb = _out_tbs[out_idx];
  AVRational us_tb  = AVRational{1, 1000000};

  int64_t in_pts_us = AV_NOPTS_VALUE;
  if (pkt->pts != AV_NOPTS_VALUE) {
    in_pts_us = util_api.rescale_q(pkt->pts, in_tb, us_tb);
  }
  int64_t in_dts_us = AV_NOPTS_VALUE;
  if (pkt->dts != AV_NOPTS_VALUE) {
    in_dts_us = util_api.rescale_q(pkt->dts, in_tb, us_tb);
  }

  // Establish the per-file anchor from the first packet on ANY
  // stream so cross-stream offsets survive rollover.
  if (!_anchor_set) {
    int64_t anchor_us = 0;
    if (in_dts_us != AV_NOPTS_VALUE) {
      anchor_us = in_dts_us;
    } else if (in_pts_us != AV_NOPTS_VALUE) {
      anchor_us = in_pts_us;
    }
    _anchor_us  = anchor_us;
    _anchor_set = true;
  }

  int64_t out_pts = AV_NOPTS_VALUE;
  int64_t out_dts = AV_NOPTS_VALUE;
  if (in_pts_us != AV_NOPTS_VALUE) {
    int64_t delta = in_pts_us - _anchor_us;
    if (delta < 0) { delta = 0; }
    out_pts = util_api.rescale_q(delta, us_tb, out_tb);
  }
  if (in_dts_us != AV_NOPTS_VALUE) {
    int64_t delta = in_dts_us - _anchor_us;
    if (delta < 0) { delta = 0; }
    out_dts = util_api.rescale_q(delta, us_tb, out_tb);
  }
  int64_t out_dur = 0;
  if (pkt->duration > 0) {
    out_dur = util_api.rescale_q(pkt->duration, in_tb, out_tb);
  }

  pkt->stream_index = out_idx;
  pkt->pts          = out_pts;
  pkt->dts          = out_dts;
  pkt->duration     = out_dur;
  pkt->pos          = -1;

  int rc = fmt_api.interleaved_write_frame(_ofctx, pkt);
  // interleaved_write_frame transfers ownership of the packet's
  // buffer (or unrefs on failure), so the caller does not unref
  // after a successful call. Either way, returning here leaves the
  // packet logically consumed.
  if (rc < 0) {
    _session->warn(fmt(
        "SegmentWriter: interleaved_write_frame failed ({}) "
        "on output stream {}", rc, out_idx));
    return false;
  }
  _last_packet_utc = std::chrono::system_clock::now();
  return true;
}

std::optional<SegmentInfo>
SegmentWriter::write_packet(AVPacket* pkt, int out_idx, bool is_video_key)
{
  if (!ready() || out_idx < 0
      || static_cast<size_t>(out_idx) >= _streams.size()) {
    return std::nullopt;
  }

  // Rollover check happens before muxing the packet. If the writer
  // decides to roll, the just-arrived keyframe becomes the first
  // packet of the NEW file -- that keeps each file starting on an
  // IDR.
  if (_streams[out_idx].is_video) {
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - _file_open_steady)
        .count();
    int64_t target_ms =
        static_cast<int64_t>(_spec.target_duration.count()) * 1000;
    if (decide_rollover(is_video_key, elapsed, target_ms)) {
      SegmentInfo finished;
      finished.path              = _current_path;
      finished.start_utc         = _file_open_utc;
      finished.end_utc           = _last_packet_utc;
      finished.duration_us       =
          std::chrono::duration_cast<std::chrono::microseconds>(
              _last_packet_utc - _file_open_utc).count();
      finished.has_audio         = _streams.size() >= 2;
      finished.stream_copy_video = _stream_copy_v;
      finished.stream_copy_audio = _stream_copy_a;
      close_current_file_();
      if (!open_new_file_()) {
        _closed = true;
        return finished;
      }
      auto written = mux_after_open_(pkt, out_idx);
      (void)written;  // best-effort; failure already logged
      return finished;
    }
  }

  (void)mux_after_open_(pkt, out_idx);
  return std::nullopt;
}

SegmentInfo
SegmentWriter::close()
{
  SegmentInfo info;
  if (_ofctx) {
    info.path              = _current_path;
    info.start_utc         = _file_open_utc;
    info.end_utc           = _last_packet_utc;
    info.duration_us       =
        std::chrono::duration_cast<std::chrono::microseconds>(
            _last_packet_utc - _file_open_utc).count();
    info.has_audio         = _streams.size() >= 2;
    info.stream_copy_video = _stream_copy_v;
    info.stream_copy_audio = _stream_copy_a;
    close_current_file_();
  }
  _closed = true;
  return info;
}

}

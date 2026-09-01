#include "common/av-filter-graph.h"

#include "common/ffmpeg-libraries.h"

#include <cmath>
#include <cstdio>
#include <string>

using namespace std;

namespace vpipe {

namespace {

void
set_err_(string* err, string msg)
{
  if (err != nullptr) { *err = std::move(msg); }
}

string
av_err_(const FFmpegLibraries* libs, int rc)
{
  char buf[256] = {0};
  if (libs != nullptr && libs->avutil().api.strerror != nullptr) {
    libs->avutil().api.strerror(rc, buf, sizeof buf);
  }
  return buf[0] != '\0' ? string(buf) : ("error " + to_string(rc));
}

}  // namespace

AVRational
av_rational_from_double(double v, int max_den)
{
  if (!(v > 0.0) || !std::isfinite(v)) { return AVRational{0, 1}; }
  // Stern-Brocot / continued fractions. Kept local rather than calling
  // av_d2q so this needs no extra dlopen'd symbol for ten lines of
  // arithmetic.
  long long p0 = 0, q0 = 1, p1 = 1, q1 = 0;
  double x = v;
  for (int i = 0; i < 32; ++i) {
    const long long a = (long long)std::floor(x);
    const long long p2 = a * p1 + p0, q2 = a * q1 + q0;
    if (q2 > max_den || p2 > (long long)max_den * 4096) { break; }
    p0 = p1; q0 = q1; p1 = p2; q1 = q2;
    const double frac = x - (double)a;
    if (frac < 1e-9) { break; }
    x = 1.0 / frac;
  }
  if (q1 <= 0 || p1 <= 0) { return AVRational{0, 1}; }
  return AVRational{(int)p1, (int)q1};
}

AvFilterGraph::~AvFilterGraph()
{
  close();
}

void
AvFilterGraph::close() noexcept
{
  if (_graph != nullptr && _libs != nullptr &&
      _libs->avfilter().api.graph_free != nullptr) {
    _libs->avfilter().api.graph_free(&_graph);
  }
  _graph = nullptr;
  _src   = nullptr;
  _sink  = nullptr;
}

bool
AvFilterGraph::open(const FFmpegLibraries* libs, const VideoIn& in,
                    const string& chain, string* err)
{
  if (in.width <= 0 || in.height <= 0) {
    set_err_(err, "filter graph: the input frame size is empty");
    return false;
  }
  if (in.frame_rate.num <= 0 || in.frame_rate.den <= 0) {
    set_err_(err, "filter graph: the input frame rate is not positive -- it "
                  "is what the chain divides by and cannot be guessed");
    return false;
  }
  // The time base is 1/rate, so an input frame's pts is its INDEX. That
  // is the one stamping scheme with no rounding to accumulate over a
  // long clip.
  _in_tb = AVRational{in.frame_rate.den, in.frame_rate.num};
  char args[256];
  std::snprintf(args, sizeof args,
                "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:"
                "pixel_aspect=1/1:frame_rate=%d/%d",
                in.width, in.height, in.pix_fmt, _in_tb.num, _in_tb.den,
                in.frame_rate.num, in.frame_rate.den);
  // The trailing `format` pins the SINK to what the caller feeds in, so
  // a chain whose filters only speak YUV (minterpolate) is converted
  // there and back by libavfilter rather than by the stage.
  char fmt[64];
  std::snprintf(fmt, sizeof fmt, "format=%d", in.pix_fmt);
  const string full = chain.empty() ? string(fmt) : chain + "," + fmt;
  return build_(libs, "buffer", args, "buffersink", full, err);
}

bool
AvFilterGraph::open(const FFmpegLibraries* libs, const AudioIn& in,
                    const string& chain, string* err)
{
  if (in.sample_rate <= 0) {
    set_err_(err, "filter graph: the input sample rate is not positive");
    return false;
  }
  if (in.channels != 1 && in.channels != 2) {
    set_err_(err, "filter graph: mono or stereo only, got " +
                  to_string(in.channels) + " channels");
    return false;
  }
  if (libs == nullptr || libs->avutil().api.get_sample_fmt_name == nullptr) {
    set_err_(err, "filter graph: FFmpeg is unavailable");
    return false;
  }
  const char* sfmt = libs->avutil().api.get_sample_fmt_name(
      (enum AVSampleFormat)in.sample_fmt);
  if (sfmt == nullptr) {
    set_err_(err, "filter graph: unknown sample format");
    return false;
  }
  _in_tb = AVRational{1, in.sample_rate};
  const char* layout = in.channels == 2 ? "stereo" : "mono";
  char args[256];
  std::snprintf(args, sizeof args,
                "time_base=1/%d:sample_rate=%d:sample_fmt=%s:"
                "channel_layout=%s",
                in.sample_rate, in.sample_rate, sfmt, layout);
  // Same idea as the video `format`: whatever the chain converted to
  // internally, what leaves the sink is the caller's own layout. The
  // RATE is deliberately NOT pinned -- a chain ending in `aresample`
  // owns the output rate, and pinning it here would silently undo it.
  char af[128];
  std::snprintf(af, sizeof af,
                "aformat=sample_fmts=%s:channel_layouts=%s", sfmt, layout);
  const string full = chain.empty() ? string(af) : chain + "," + af;
  return build_(libs, "abuffer", args, "abuffersink", full, err);
}

bool
AvFilterGraph::build_(const FFmpegLibraries* libs, const char* src_name,
                      const string& src_args, const char* sink_name,
                      const string& chain, string* err)
{
  close();
  _libs = libs;
  if (libs == nullptr || !libs->avfilter().valid()) {
    set_err_(err, "filter graph: libavfilter is not loaded, so no temporal "
                  "resampling is available on this build");
    return false;
  }
  const auto& F = libs->avfilter().api;
  const auto& U = libs->avutil().api;

  const AVFilter* srcf  = F.get_by_name(src_name);
  const AVFilter* sinkf = F.get_by_name(sink_name);
  if (srcf == nullptr || sinkf == nullptr) {
    set_err_(err, string("filter graph: this FFmpeg has no ") + src_name +
                  "/" + sink_name);
    return false;
  }
  _graph = F.graph_alloc();
  if (_graph == nullptr) {
    set_err_(err, "filter graph: allocation failed");
    return false;
  }
  _src  = F.graph_alloc_filter(_graph, srcf, "in");
  _sink = F.graph_alloc_filter(_graph, sinkf, "out");
  if (_src == nullptr || _sink == nullptr) {
    set_err_(err, "filter graph: could not allocate the endpoints");
    close();
    return false;
  }
  int rc = F.init_str(_src, src_args.c_str());
  if (rc < 0) {
    set_err_(err, "filter graph: the source rejected '" + src_args + "': " +
                  av_err_(libs, rc));
    close();
    return false;
  }
  rc = F.init_str(_sink, nullptr);
  if (rc < 0) {
    set_err_(err, "filter graph: the sink would not initialise: " +
                  av_err_(libs, rc));
    close();
    return false;
  }

  // parse_ptr's `inputs` are the chain's DANGLING inputs -- which our
  // source feeds -- and its `outputs` the dangling outputs the sink
  // takes. The naming reads backwards exactly once and then never
  // again.
  AVFilterInOut* outputs = F.inout_alloc();
  AVFilterInOut* inputs  = F.inout_alloc();
  if (outputs == nullptr || inputs == nullptr) {
    if (outputs != nullptr) { F.inout_free(&outputs); }
    if (inputs  != nullptr) { F.inout_free(&inputs); }
    set_err_(err, "filter graph: allocation failed");
    close();
    return false;
  }
  outputs->name       = U.strdup("in");
  outputs->filter_ctx = _src;
  outputs->pad_idx    = 0;
  outputs->next       = nullptr;
  inputs->name        = U.strdup("out");
  inputs->filter_ctx  = _sink;
  inputs->pad_idx     = 0;
  inputs->next        = nullptr;

  rc = F.graph_parse_ptr(_graph, chain.c_str(), &inputs, &outputs, nullptr);
  F.inout_free(&inputs);
  F.inout_free(&outputs);
  if (rc < 0) {
    set_err_(err, "filter graph: '" + chain + "' would not parse: " +
                  av_err_(libs, rc));
    close();
    return false;
  }
  rc = F.graph_config(_graph, nullptr);
  if (rc < 0) {
    set_err_(err, "filter graph: '" + chain + "' would not configure: " +
                  av_err_(libs, rc));
    close();
    return false;
  }
  return true;
}

bool
AvFilterGraph::push(AVFrame* frame, string* err)
{
  if (!is_open()) {
    set_err_(err, "filter graph: not open");
    return false;
  }
  // KEEP_REF so the caller's frame stays the caller's; without it
  // libavfilter takes the buffers and hands back an empty shell, which
  // is a use-after-move the compiler cannot see.
  const int rc = _libs->avfilter().api.buffersrc_add_frame_flags(
      _src, frame, frame != nullptr ? AV_BUFFERSRC_FLAG_KEEP_REF : 0);
  if (rc < 0) {
    set_err_(err, "filter graph: the source refused a frame: " +
                  av_err_(_libs, rc));
    return false;
  }
  return true;
}

AvFilterGraph::Pull
AvFilterGraph::pull(AVFrame* out, string* err)
{
  if (!is_open() || out == nullptr) {
    set_err_(err, "filter graph: not open");
    return Pull::kError;
  }
  _libs->avutil().api.frame_unref(out);
  const int rc = _libs->avfilter().api.buffersink_get_frame(_sink, out);
  if (rc == AVERROR(EAGAIN)) { return Pull::kAgain; }
  if (rc == AVERROR_EOF)     { return Pull::kEof; }
  if (rc < 0) {
    set_err_(err, "filter graph: the sink failed: " + av_err_(_libs, rc));
    return Pull::kError;
  }
  return Pull::kFrame;
}

}  // namespace vpipe

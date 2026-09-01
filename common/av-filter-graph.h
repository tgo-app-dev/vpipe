#ifndef VPIPE_COMMON_AV_FILTER_GRAPH_H
#define VPIPE_COMMON_AV_FILTER_GRAPH_H

#include <string>

extern "C" {
#include <libavfilter/avfilter.h>
#include <libavutil/frame.h>
#include <libavutil/rational.h>
}

// One libavfilter graph, driven a frame at a time.
//
// WHY THIS EXISTS. Temporal resampling is a solved problem with a
// decade of tuning behind it -- `fps`, `framerate`, `tmix`,
// `minterpolate` for pictures, `atempo`, `asetrate`, `aresample` for
// sound -- and every one of those lives in libavfilter. Re-deriving
// them here would mean re-deriving WSOLA and motion estimation, badly.
// So the stages describe a filter CHAIN and this drives it.
//
// It is a push/pull pump and nothing else: no format guessing, no
// buffering policy, no timestamps invented on the caller's behalf. A
// filter chain answers with as many frames as it likes for each one it
// is given -- none, one, or several -- which is exactly why the stages
// that use it have to say whether their oport shares the iport's clock.
namespace vpipe {

class FFmpegLibraries;

class AvFilterGraph {
public:
  // What the graph is fed. One of the two `open` overloads picks the
  // buffer source (`buffer` / `abuffer`) and with it the whole graph's
  // media type; a chain that does not match its source fails to
  // configure, which is a build-time error with a reason rather than a
  // wrong answer.
  struct VideoIn {
    int        width   = 0;
    int        height  = 0;
    int        pix_fmt = 0;               // an AVPixelFormat
    // The rate the frames were sampled at. Both are required: `fps` and
    // `minterpolate` read the input rate off the link, and a chain told
    // the wrong one resamples by the wrong ratio while reporting
    // nothing. The time base is its reciprocal, so a frame's pts is its
    // INDEX and there is no rounding to accumulate.
    AVRational frame_rate{0, 1};
  };
  struct AudioIn {
    int sample_rate = 0;
    int channels    = 0;                  // 1 or 2
    int sample_fmt  = 0;                  // an AVSampleFormat
  };

  AvFilterGraph() = default;
  ~AvFilterGraph();
  AvFilterGraph(const AvFilterGraph&)            = delete;
  AvFilterGraph& operator=(const AvFilterGraph&) = delete;

  // Build the graph. `chain` is the same syntax `ffmpeg -vf` / `-af`
  // takes ("fps=24", "tmix=frames=3,fps=24", "atempo=0.8,aresample=32000").
  // An output format filter is appended so what comes out of the sink is
  // what the caller asked to be fed IN -- planar RGB stays planar RGB,
  // planar float stays planar float -- with libavfilter inserting the
  // conversions the chain's own filters need. False with a reason.
  bool open(const FFmpegLibraries* libs, const VideoIn& in,
            const std::string& chain, std::string* err);
  bool open(const FFmpegLibraries* libs, const AudioIn& in,
            const std::string& chain, std::string* err);

  bool is_open() const noexcept { return _graph != nullptr; }
  void close() noexcept;

  // Feed one frame, or NULL to signal end of stream. The frame is
  // referenced, not taken: the caller keeps ownership either way.
  bool push(AVFrame* frame, std::string* err);

  enum class Pull { kFrame, kAgain, kEof, kError };
  // Take one output frame. kAgain means the chain wants more input;
  // kEof means it is finished, which only follows a push(nullptr).
  // `out` is unref'd first, so one frame can be reused across a drain.
  Pull pull(AVFrame* out, std::string* err);

  // The buffer source's time base, so a caller can stamp its input
  // frames in the units the graph will read them in. Video: 1/frame_rate,
  // so a pts is a frame index. Audio: 1/sample_rate, so a pts is a
  // sample index.
  AVRational input_time_base() const noexcept { return _in_tb; }

private:
  bool build_(const FFmpegLibraries* libs, const char* src_name,
              const std::string& src_args, const char* sink_name,
              const std::string& chain, std::string* err);

  const FFmpegLibraries* _libs  = nullptr;
  AVFilterGraph*         _graph = nullptr;
  AVFilterContext*       _src   = nullptr;
  AVFilterContext*       _sink  = nullptr;
  AVRational             _in_tb{0, 1};
};

// A double as a rational, by continued fractions, so a rate like
// 29.97 becomes 30000/1001 rather than 2997/100. libavfilter reads the
// input rate off the link and `fps` divides by it; a rate that is a
// hair off is a chain that drops one frame somewhere in every few
// thousand and says nothing about it.
AVRational av_rational_from_double(double v, int max_den = 1000000);

}  // namespace vpipe

#endif

// The libavfilter pump the two temporal-resample stages are built on.
//
// It is tested on its own because everything above it is a filter-chain
// STRING: if the pump drops frames, mis-stamps them, or hands the sink a
// format the chain cannot take, the stages report a plausible-looking
// beat count and nothing says why. These assertions are the ones the
// stages then get to assume.
#include "minitest.h"
#include "common/av-filter-graph.h"
#include "common/ffmpeg-libraries.h"
#include "common/session.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
#include <libavutil/pixfmt.h>
#include <libavutil/samplefmt.h>
}

using namespace std;
using namespace vpipe;

namespace {

// Every test needs the libraries and a frame to push; a box without
// FFmpeg skips vacuously, which is the rule the media tests already use.
const FFmpegLibraries*
libs_or_null_(Session& s)
{
  try {
    const FFmpegLibraries* l = s.ffmpeg_libraries();
    if (l != nullptr && l->valid() && l->avfilter().valid()) { return l; }
  } catch (...) {
  }
  return nullptr;
}

struct Frame {
  const FFmpegLibraries* libs;
  AVFrame*               f = nullptr;
  explicit Frame(const FFmpegLibraries* l) : libs(l)
  {
    f = libs->avutil().api.frame_alloc();
  }
  ~Frame()
  {
    if (f != nullptr) { libs->avutil().api.frame_free(&f); }
  }
};

}  // namespace

TEST(av_filter_graph, a_double_becomes_the_rational_a_rate_actually_is)
{
  // 30000/1001, not 2997/100: the chain divides by the input rate, and
  // a rate a hair off drops a frame every few thousand silently.
  const AVRational a = av_rational_from_double(30000.0 / 1001.0);
  EXPECT_TRUE(a.num == 30000 && a.den == 1001);
  const AVRational b = av_rational_from_double(24.0);
  EXPECT_TRUE(b.num == 24 && b.den == 1);
  const AVRational c = av_rational_from_double(23.976023976);
  EXPECT_TRUE(c.num == 24000 && c.den == 1001);
  // Nonsense is refused rather than approximated.
  const AVRational z = av_rational_from_double(0.0);
  EXPECT_TRUE(z.num == 0);
  std::printf("[av_filter_graph] 29.97 -> %d/%d, 23.976 -> %d/%d\n",
              a.num, a.den, c.num, c.den);
}

TEST(av_filter_graph, video_fps_halves_the_frame_count)
{
  Session s;
  const FFmpegLibraries* libs = libs_or_null_(s);
  if (libs == nullptr) { return; }

  constexpr int W = 32, H = 16;
  AvFilterGraph::VideoIn in;
  in.width = W; in.height = H;
  in.pix_fmt = AV_PIX_FMT_GBRP;
  in.frame_rate = AVRational{30, 1};

  AvFilterGraph g;
  string err;
  ASSERT_TRUE(g.open(libs, in, "fps=15", &err));
  EXPECT_TRUE(err.empty());
  // 1/rate, so a pts is a frame index.
  EXPECT_TRUE(g.input_time_base().num == 1 &&
              g.input_time_base().den == 30);

  Frame src(libs), out(libs);
  ASSERT_TRUE(src.f != nullptr && out.f != nullptr);
  src.f->format = AV_PIX_FMT_GBRP;
  src.f->width  = W;
  src.f->height = H;
  ASSERT_TRUE(libs->avutil().api.frame_get_buffer(src.f, 0) >= 0);

  int pulled = 0;
  auto drain = [&]() {
    while (true) {
      const auto r = g.pull(out.f, &err);
      if (r == AvFilterGraph::Pull::kFrame) { ++pulled; continue; }
      EXPECT_TRUE(r != AvFilterGraph::Pull::kError);
      break;
    }
  };
  for (int i = 0; i < 30; ++i) {
    // A ramp per frame, so a chain that duplicated instead of dropping
    // would be visible in the values as well as the count.
    for (int p = 0; p < 3; ++p) {
      std::memset(src.f->data[p], (unsigned char)(i * 8),
                  (size_t)src.f->linesize[p] * H);
    }
    src.f->pts = i;
    ASSERT_TRUE(g.push(src.f, &err));
    drain();
  }
  ASSERT_TRUE(g.push(nullptr, &err));
  drain();
  // 30 frames of 30 fps is one second; at 15 fps that is 15 out, and
  // the filter's own start-up rounding is worth one frame either way.
  EXPECT_TRUE(pulled >= 14 && pulled <= 16);
  std::printf("[av_filter_graph] 30 frames @30fps -> %d @15fps\n", pulled);
}

TEST(av_filter_graph, audio_atempo_halves_the_duration)
{
  Session s;
  const FFmpegLibraries* libs = libs_or_null_(s);
  if (libs == nullptr) { return; }

  constexpr int SR = 32000, N = 1024;
  AvFilterGraph::AudioIn in;
  in.sample_rate = SR;
  in.channels    = 2;
  in.sample_fmt  = AV_SAMPLE_FMT_FLTP;

  AvFilterGraph g;
  string err;
  ASSERT_TRUE(g.open(libs, in, "atempo=2.0", &err));
  EXPECT_TRUE(err.empty());

  Frame src(libs), out(libs);
  ASSERT_TRUE(src.f != nullptr && out.f != nullptr);
  src.f->format      = AV_SAMPLE_FMT_FLTP;
  src.f->sample_rate = SR;
  src.f->nb_samples  = N;
  src.f->ch_layout   = AVChannelLayout AV_CHANNEL_LAYOUT_STEREO;
  ASSERT_TRUE(libs->avutil().api.frame_get_buffer(src.f, 0) >= 0);

  long long got = 0;
  auto drain = [&]() {
    while (true) {
      const auto r = g.pull(out.f, &err);
      if (r == AvFilterGraph::Pull::kFrame) {
        got += out.f->nb_samples;
        // The sink gives back the layout it was fed, which is what the
        // trailing aformat is for.
        EXPECT_TRUE(out.f->format == AV_SAMPLE_FMT_FLTP);
        EXPECT_TRUE(out.f->ch_layout.nb_channels == 2);
        continue;
      }
      EXPECT_TRUE(r != AvFilterGraph::Pull::kError);
      break;
    }
  };
  // One second in.
  const int blocks = SR / N;
  for (int b = 0; b < blocks; ++b) {
    for (int c = 0; c < 2; ++c) {
      float* d = reinterpret_cast<float*>(src.f->data[c]);
      for (int i = 0; i < N; ++i) {
        const double t = (double)(b * N + i) / SR;
        d[i] = 0.4f * (float)std::sin(2.0 * M_PI * 440.0 * t);
      }
    }
    src.f->pts = (long long)b * N;
    ASSERT_TRUE(g.push(src.f, &err));
    drain();
  }
  ASSERT_TRUE(g.push(nullptr, &err));
  drain();
  const long long want = (long long)blocks * N / 2;
  EXPECT_TRUE(got > want - SR / 20 && got < want + SR / 20);
  std::printf("[av_filter_graph] %d samples @2.0x -> %lld (want ~%lld)\n",
              blocks * N, got, want);
}

TEST(av_filter_graph, a_chain_that_cannot_configure_says_which_and_why)
{
  Session s;
  const FFmpegLibraries* libs = libs_or_null_(s);
  if (libs == nullptr) { return; }

  AvFilterGraph::VideoIn in;
  in.width = 16; in.height = 16;
  in.pix_fmt = AV_PIX_FMT_GBRP;
  in.frame_rate = AVRational{24, 1};

  AvFilterGraph g;
  string err;
  // An AUDIO filter in a video graph. It has to fail at BUILD time with
  // the chain in the message -- the alternative is a stage that starts,
  // accepts frames and produces nothing.
  EXPECT_FALSE(g.open(libs, in, "atempo=2.0", &err));
  EXPECT_FALSE(err.empty());
  EXPECT_TRUE(err.find("atempo") != string::npos);
  EXPECT_FALSE(g.is_open());

  // A rate that was never resolved is refused before anything is built:
  // it is what the chain divides by.
  in.frame_rate = AVRational{0, 1};
  err.clear();
  EXPECT_FALSE(g.open(libs, in, "fps=12", &err));
  EXPECT_TRUE(err.find("frame rate") != string::npos);
  std::printf("[av_filter_graph] refusals: %s\n", err.c_str());
}

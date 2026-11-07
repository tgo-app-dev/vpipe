#include "minitest.h"
#include "common/ffmpeg-libraries.h"
#include "common/ffmpeg-log-tap.h"
#include "common/library-handle.h"
#include "common/session.h"

extern "C" {
#include <libavcodec/version.h>
#include <libavformat/version.h>
#include <libavutil/version.h>
#include <libswresample/version.h>
#include <libswscale/version.h>
}

#include <map>

#include <iostream>
#include <streambuf>
#include <string>

using namespace std;
using namespace vpipe;

namespace {

// Suppresses cerr while a load is being attempted -- the bootstrap
// stdout delegate writes warnings to cerr on Optional load misses, and
// we don't want to pollute test output when the dev environment lacks
// FFmpeg.
class CerrSilencer {
public:
  CerrSilencer() : _saved(cerr.rdbuf()), _null() { cerr.rdbuf(&_null); }
  ~CerrSilencer() { cerr.rdbuf(_saved); }
private:
  struct NullBuf : public streambuf {
    int overflow(int c) override { return c; }
  };
  streambuf* _saved;
  NullBuf    _null;
};

}

TEST(ffmpeg_libraries, avutil_loads_or_skips) {
  Session s;
  CerrSilencer hush;
  LibAvUtil lib(&s, LibraryHandle::LoadMode::Optional);
  if (!lib.valid()) {
    return;
  }
  EXPECT_TRUE(lib.api.version != nullptr);
  EXPECT_TRUE(lib.api.version() != 0u);
  EXPECT_TRUE(lib.version_major() >= 56u);
  EXPECT_TRUE(lib.api.frame_alloc != nullptr);
  EXPECT_TRUE(lib.api.frame_free  != nullptr);
  EXPECT_TRUE(lib.api.strerror    != nullptr);
}

TEST(ffmpeg_libraries, avcodec_loads_or_skips) {
  Session s;
  CerrSilencer hush;
  LibAvCodec lib(&s, LibraryHandle::LoadMode::Optional);
  if (!lib.valid()) {
    return;
  }
  EXPECT_TRUE(lib.api.version != nullptr);
  EXPECT_TRUE(lib.api.version() != 0u);
  EXPECT_TRUE(lib.version_major() >= 58u);
  EXPECT_TRUE(lib.api.send_packet     != nullptr);
  EXPECT_TRUE(lib.api.receive_frame   != nullptr);
  EXPECT_TRUE(lib.api.alloc_context3  != nullptr);
  EXPECT_TRUE(lib.api.find_decoder    != nullptr);
}

TEST(ffmpeg_libraries, avformat_loads_or_skips) {
  Session s;
  CerrSilencer hush;
  LibAvFormat lib(&s, LibraryHandle::LoadMode::Optional);
  if (!lib.valid()) {
    return;
  }
  EXPECT_TRUE(lib.api.version != nullptr);
  EXPECT_TRUE(lib.api.version() != 0u);
  EXPECT_TRUE(lib.version_major() >= 58u);
  EXPECT_TRUE(lib.api.open_input  != nullptr);
  EXPECT_TRUE(lib.api.read_frame  != nullptr);
  EXPECT_TRUE(lib.api.write_header != nullptr);
}

TEST(ffmpeg_libraries, swresample_loads_or_skips) {
  Session s;
  CerrSilencer hush;
  LibSwResample lib(&s, LibraryHandle::LoadMode::Optional);
  if (!lib.valid()) {
    return;
  }
  EXPECT_TRUE(lib.api.version != nullptr);
  EXPECT_TRUE(lib.api.version() != 0u);
  EXPECT_TRUE(lib.version_major() >= 3u);
  EXPECT_TRUE(lib.api.alloc           != nullptr);
  EXPECT_TRUE(lib.api.alloc_set_opts2 != nullptr);
  EXPECT_TRUE(lib.api.convert         != nullptr);
}

TEST(ffmpeg_libraries, swscale_loads_or_skips) {
  Session s;
  CerrSilencer hush;
  LibSwScale lib(&s, LibraryHandle::LoadMode::Optional);
  if (!lib.valid()) {
    return;
  }
  EXPECT_TRUE(lib.api.version != nullptr);
  EXPECT_TRUE(lib.api.version() != 0u);
  EXPECT_TRUE(lib.version_major() >= 5u);
  EXPECT_TRUE(lib.api.get_context  != nullptr);
  EXPECT_TRUE(lib.api.scale        != nullptr);
  EXPECT_TRUE(lib.api.free_context != nullptr);
}

TEST(ffmpeg_libraries, composite_loads_or_skips) {
  Session s;
  CerrSilencer hush;
  FFmpegLibraries libs(&s, LibraryHandle::LoadMode::Optional);
  if (!libs.valid()) {
    return;
  }
  EXPECT_TRUE(libs.avutil().valid());
  EXPECT_TRUE(libs.avcodec().valid());
  EXPECT_TRUE(libs.avformat().valid());
  EXPECT_TRUE(libs.swresample().valid());
  EXPECT_TRUE(libs.swscale().valid());
  EXPECT_TRUE(libs.avutil().api.version()     != 0u);
  EXPECT_TRUE(libs.avcodec().api.version()    != 0u);
  EXPECT_TRUE(libs.avformat().api.version()   != 0u);
  EXPECT_TRUE(libs.swresample().api.version() != 0u);
  EXPECT_TRUE(libs.swscale().api.version()    != 0u);
}

// WHICH FFmpeg the loader takes: the first candidate of the MAJOR this
// build's headers describe, never one of another -- the code reads
// FFmpeg's structures at those headers' offsets. A fake probe stands in
// for dlopen, so the rule is checked with no second FFmpeg installed.
TEST(ffmpeg_libraries, a_library_of_another_major_is_passed_over) {
  auto v = [](unsigned maj, unsigned min, unsigned mic) {
    return (maj << 16) | (min << 8) | mic;
  };
  std::map<string, unsigned> on_disk = {
      {"old.dylib", v(59, 39, 100)},
      {"new.dylib", v(60, 26, 101)},
      {"newer.dylib", v(61, 3, 100)},
  };
  auto probe = [&](const string& path, unsigned* ver) {
    auto it = on_disk.find(path);
    if (it == on_disk.end()) { return false; }   // does not load
    *ver = it->second;
    return true;
  };

  // Another major first, then a missing one, then the right one.
  FfmpegPick p = pick_ffmpeg_library(
      {"old.dylib", "absent.dylib", "new.dylib"}, "libavutil", 60, false,
      probe);
  EXPECT_TRUE(p.path == "new.dylib" && p.version == v(60, 26, 101));
  EXPECT_TRUE(!p.other_major && p.why.empty());
  EXPECT_TRUE(p.passed_over == "'old.dylib' 59.39.100");

  // Only other majors: refused, every candidate found named.
  p = pick_ffmpeg_library({"old.dylib", "newer.dylib"}, "libavutil", 60,
                          false, probe);
  printf("[ffmpeg_libraries] refusal: %s\n", p.why.c_str());
  EXPECT_TRUE(p.path.empty());
  EXPECT_TRUE(p.why.find("'old.dylib' 59.39.100") != string::npos);
  EXPECT_TRUE(p.why.find("'newer.dylib' 61.3.100") != string::npos);
  EXPECT_TRUE(p.why.find("major 60") != string::npos);
  EXPECT_TRUE(p.why.find("VPIPE_FFMPEG_ANY_MAJOR") != string::npos);

  // VPIPE_FFMPEG_ANY_MAJOR: the first loadable, said so.
  p = pick_ffmpeg_library({"old.dylib", "newer.dylib"}, "libavutil", 60,
                          true, probe);
  EXPECT_TRUE(p.path == "old.dylib" && p.other_major && !p.why.empty());

  // Nothing loads: no pick and no refusal -- the absent-FFmpeg path.
  p = pick_ffmpeg_library({"absent.dylib"}, "libavutil", 60, false, probe);
  EXPECT_TRUE(p.path.empty() && p.why.empty());
}

// The FFmpeg this run loaded is the major this build was compiled
// against, library by library.
TEST(ffmpeg_libraries, the_loaded_libraries_are_this_builds_major) {
  Session s;
  const FFmpegLibraries* libs = nullptr;
  try {
    libs = s.ffmpeg_libraries();
  } catch (...) {
    return;
  }
  if (!libs || !libs->valid()) { return; }
  printf("[ffmpeg_libraries] loaded avutil %u, avcodec %u, avformat %u, "
         "swresample %u, swscale %u; built against %d, %d, %d, %d, %d\n",
         libs->avutil().version_major(), libs->avcodec().version_major(),
         libs->avformat().version_major(),
         libs->swresample().version_major(),
         libs->swscale().version_major(), LIBAVUTIL_VERSION_MAJOR,
         LIBAVCODEC_VERSION_MAJOR, LIBAVFORMAT_VERSION_MAJOR,
         LIBSWRESAMPLE_VERSION_MAJOR, LIBSWSCALE_VERSION_MAJOR);
  EXPECT_TRUE(libs->avutil().version_major() ==
              (unsigned)LIBAVUTIL_VERSION_MAJOR);
  EXPECT_TRUE(libs->avcodec().version_major() ==
              (unsigned)LIBAVCODEC_VERSION_MAJOR);
  EXPECT_TRUE(libs->avformat().version_major() ==
              (unsigned)LIBAVFORMAT_VERSION_MAJOR);
  EXPECT_TRUE(libs->swresample().version_major() ==
              (unsigned)LIBSWRESAMPLE_VERSION_MAJOR);
  EXPECT_TRUE(libs->swscale().version_major() ==
              (unsigned)LIBSWSCALE_VERSION_MAJOR);
}

// A refusal's reason, in FFmpeg's words, captured on the calling thread
// (FfmpegLogCapture) -- through the real library: swr_alloc_set_opts2
// handed the layout audio-to-pcm and media-decode used to build past two
// channels (MONO, its count overwritten to 6) refuses it, and says why
// only through av_log. The well-formed layouts are taken.
TEST(ffmpeg_libraries, a_refusal_carries_ffmpegs_reason) {
  Session s;
  const FFmpegLibraries* libs = nullptr;
  try {
    libs = s.ffmpeg_libraries();
  } catch (...) {
    return;
  }
  if (!libs || !libs->valid()) { return; }
  auto alloc = [&](const AVChannelLayout& in, string* said) {
    FfmpegLogCapture cap;
    AVChannelLayout out = ffmpeg_channel_layout(2);
    SwrContext* sw = nullptr;
    const int rc = libs->swresample().api.alloc_set_opts2(
        &sw, &out, AV_SAMPLE_FMT_FLT, 16000, &in, AV_SAMPLE_FMT_S16, 48000,
        0, nullptr);
    if (sw) {
      if (rc >= 0) { (void)libs->swresample().api.init(sw); }
      libs->swresample().api.free(&sw);
    }
    *said = cap.text();
    return rc;
  };
  AVChannelLayout old_six = AV_CHANNEL_LAYOUT_MONO;
  old_six.nb_channels = 6;
  string said;
  const int rc_old = alloc(old_six, &said);
  printf("[ffmpeg_libraries] the old 6-channel layout: rc %d, FFmpeg said "
         "\"%s\"\n", rc_old, said.c_str());
  EXPECT_TRUE(rc_old < 0);
  EXPECT_TRUE(said.find("invalid") != string::npos);
  // The layouts ffmpeg_channel_layout builds: taken, and nothing said.
  AVChannelLayout five_one = AV_CHANNEL_LAYOUT_5POINT1;
  for (const AVChannelLayout& l :
       {ffmpeg_channel_layout(6), ffmpeg_channel_layout(6, &five_one),
        ffmpeg_channel_layout(3), ffmpeg_channel_layout(2)}) {
    EXPECT_TRUE(alloc(l, &said) >= 0);
    EXPECT_TRUE(said.empty());
  }
  // From the frame only when it is a well-formed NATIVE layout.
  EXPECT_TRUE(ffmpeg_channel_layout(6, &five_one).order ==
              AV_CHANNEL_ORDER_NATIVE);
  EXPECT_TRUE(ffmpeg_channel_layout(6, &old_six).order ==
              AV_CHANNEL_ORDER_UNSPEC);
}

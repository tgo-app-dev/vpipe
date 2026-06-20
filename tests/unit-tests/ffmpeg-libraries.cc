#include "minitest.h"
#include "common/ffmpeg-libraries.h"
#include "common/library-handle.h"
#include "common/session.h"

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

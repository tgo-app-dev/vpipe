#include "minitest.h"
#include "common/ffmpeg-libraries.h"
#include "common/library-handle.h"
#include "common/media-decode.h"
#include "common/session.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

// Synthesize a tiny binary PPM (P6): 4x2, top row solid red, bottom
// row solid blue. FFmpeg's ppm decoder handles it without any codec
// dependency, so the test runs on any box with FFmpeg installed.
vector<uint8_t>
make_ppm_()
{
  const string hdr = "P6\n4 2\n255\n";
  vector<uint8_t> out(hdr.begin(), hdr.end());
  auto px = [&](int r, int g, int b) {
    out.push_back(static_cast<uint8_t>(r));
    out.push_back(static_cast<uint8_t>(g));
    out.push_back(static_cast<uint8_t>(b));
  };
  for (int x = 0; x < 4; ++x) { px(255, 0, 0); }   // row 0: red
  for (int x = 0; x < 4; ++x) { px(0, 0, 255); }   // row 1: blue
  return out;
}

// Synthesize a 0.25 s, 8 kHz, s16 mono WAV carrying a 440 Hz sine at
// ~0.5 amplitude. PCM-in-WAV needs no external codec either.
vector<uint8_t>
make_wav_()
{
  constexpr int      kRate    = 8000;
  constexpr int      kSamples = kRate / 4;
  constexpr uint32_t kDataLen = kSamples * 2;

  vector<uint8_t> out;
  auto u32 = [&](uint32_t v) {
    for (int i = 0; i < 4; ++i) {
      out.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xff));
    }
  };
  auto u16 = [&](uint16_t v) {
    out.push_back(static_cast<uint8_t>(v & 0xff));
    out.push_back(static_cast<uint8_t>(v >> 8));
  };
  auto tag = [&](const char* t) {
    out.insert(out.end(), t, t + 4);
  };
  tag("RIFF"); u32(36 + kDataLen); tag("WAVE");
  tag("fmt "); u32(16);
  u16(1);                       // PCM
  u16(1);                       // mono
  u32(kRate);
  u32(kRate * 2);               // byte rate
  u16(2);                       // block align
  u16(16);                      // bits/sample
  tag("data"); u32(kDataLen);
  for (int i = 0; i < kSamples; ++i) {
    const double v =
        0.5 * std::sin(2.0 * M_PI * 440.0 * i / kRate);
    u16(static_cast<uint16_t>(
        static_cast<int16_t>(v * 32767.0)));
  }
  return out;
}

}

TEST(media_decode, image_bytes_to_planar_rgb) {
  Session s;
  const FFmpegLibraries* libs = nullptr;
  try {
    libs = s.ffmpeg_libraries();
  } catch (...) {
    return;   // no FFmpeg on this box -- skip vacuously
  }
  if (!libs || !libs->valid()) {
    return;
  }
  const auto ppm = make_ppm_();
  string err;
  auto img = decode_image_bytes(
      libs, span<const uint8_t>(ppm.data(), ppm.size()), &err);
  ASSERT_TRUE(img.has_value());
  EXPECT_TRUE(img->width == 4);
  EXPECT_TRUE(img->height == 2);
  ASSERT_TRUE(img->rgb.size() == 3u * 2u * 4u);
  // Planar [3,H,W], channel order R,G,B. Row 0 red, row 1 blue.
  // sws GBRP conversion is exact for these saturated primaries.
  const uint8_t* R = img->rgb.data();
  const uint8_t* G = R + 8;
  const uint8_t* B = G + 8;
  EXPECT_TRUE(R[0] > 240 && G[0] < 16 && B[0] < 16);      // (0,0) red
  EXPECT_TRUE(R[7] < 16 && G[7] < 16 && B[7] > 240);      // (1,3) blue

  // Malformed bytes fail cleanly with a reason.
  vector<uint8_t> junk = {0, 1, 2, 3};
  err.clear();
  auto bad = decode_image_bytes(
      libs, span<const uint8_t>(junk.data(), junk.size()), &err);
  EXPECT_TRUE(!bad.has_value());
  EXPECT_TRUE(!err.empty());
}

TEST(media_decode, audio_bytes_resampled_to_target) {
  Session s;
  const FFmpegLibraries* libs = nullptr;
  try {
    libs = s.ffmpeg_libraries();
  } catch (...) {
    return;
  }
  if (!libs || !libs->valid()) {
    return;
  }
  const auto wav = make_wav_();
  string err;
  auto au = decode_audio_bytes(
      libs, span<const uint8_t>(wav.data(), wav.size()), 16000, &err);
  ASSERT_TRUE(au.has_value());
  EXPECT_TRUE(au->sample_rate == 16000);
  // 0.25 s at 16 kHz = 4000 samples; allow resampler edge slack.
  EXPECT_TRUE(au->pcm.size() > 3900 && au->pcm.size() < 4100);
  // Values in [-1,1] and the sine's RMS is ~0.35 (0.5/sqrt(2)).
  double peak = 0.0, sq = 0.0;
  for (float v : au->pcm) {
    peak = std::max(peak, static_cast<double>(std::fabs(v)));
    sq += static_cast<double>(v) * v;
  }
  const double rms = std::sqrt(sq / au->pcm.size());
  EXPECT_TRUE(peak <= 1.0);
  EXPECT_TRUE(peak > 0.4);
  EXPECT_TRUE(rms > 0.3 && rms < 0.4);
}

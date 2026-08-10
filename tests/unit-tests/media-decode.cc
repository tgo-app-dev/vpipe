#include "minitest.h"
#include "common/ffmpeg-libraries.h"
#include "common/library-handle.h"
#include "common/media-decode.h"
#include "common/session.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
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

namespace {

// Synthesize a stereo WAV: LEFT carries a 440 Hz sine, RIGHT is silent.
// That asymmetry is the point -- it is what tells a planar [2, N] result
// apart from an interleaved one, and it pins the channel ORDER.
vector<uint8_t>
make_stereo_wav_()
{
  constexpr int      kRate    = 8000;
  constexpr int      kSamples = kRate / 4;
  constexpr uint32_t kDataLen = kSamples * 4;      // 2 ch * s16

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
  auto tag = [&](const char* t) { out.insert(out.end(), t, t + 4); };
  tag("RIFF"); u32(36 + kDataLen); tag("WAVE");
  tag("fmt "); u32(16);
  u16(1);                       // PCM
  u16(2);                       // stereo
  u32(kRate);
  u32(kRate * 4);               // byte rate
  u16(4);                       // block align
  u16(16);                      // bits/sample
  tag("data"); u32(kDataLen);
  for (int i = 0; i < kSamples; ++i) {
    const double v = 0.5 * std::sin(2.0 * M_PI * 440.0 * i / kRate);
    u16(static_cast<uint16_t>(static_cast<int16_t>(v * 32767.0)));
    u16(0);                     // right: silence
  }
  return out;
}

// Synthesize an N-frame YUV4MPEG2 clip at a stated frame rate, frame i
// at luma `20 + 20 * i`. Y4M is a raw container FFmpeg demuxes with no
// codec dependency, and -- unlike a bare pixel dump -- it CARRIES the
// frame rate, which is the field a reference conditioned at the wrong
// speed loses.
vector<uint8_t>
make_y4m_(int frames, int w, int h, int fps_num, int fps_den)
{
  string hdr = "YUV4MPEG2 W" + to_string(w) + " H" + to_string(h) +
               " F" + to_string(fps_num) + ":" + to_string(fps_den) +
               " Ip A1:1 C420mpeg2\n";
  vector<uint8_t> out(hdr.begin(), hdr.end());
  for (int f = 0; f < frames; ++f) {
    const string fh = "FRAME\n";
    out.insert(out.end(), fh.begin(), fh.end());
    const uint8_t luma = static_cast<uint8_t>(20 + 20 * f);
    out.insert(out.end(), static_cast<size_t>(w) * h, luma);
    out.insert(out.end(), static_cast<size_t>(w / 2) * (h / 2), 128);
    out.insert(out.end(), static_cast<size_t>(w / 2) * (h / 2), 128);
  }
  return out;
}

// Write bytes to a temp file and hand back the path; decode_video_file
// takes a path (a clip and its soundtrack come out of ONE open
// container, which the in-memory AVIO shim does not model).
string
write_temp_(const vector<uint8_t>& bytes, const char* name)
{
  const string path =
      (std::filesystem::temp_directory_path() / name).string();
  FILE* f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) { return {}; }
  std::fwrite(bytes.data(), 1, bytes.size(), f);
  std::fclose(f);
  return path;
}

}

TEST(media_decode, audio_bytes_to_planar_stereo) {
  Session s;
  const FFmpegLibraries* libs = nullptr;
  try {
    libs = s.ffmpeg_libraries();
  } catch (...) {
    return;   // no FFmpeg on this box -- skip vacuously
  }
  if (!libs || !libs->valid()) { return; }

  const auto wav = make_stereo_wav_();
  string err;
  auto a = decode_audio_bytes(
      libs, span<const uint8_t>(wav.data(), wav.size()), 16000, &err, 2);
  ASSERT_TRUE(a.has_value());
  EXPECT_TRUE(a->channels == 2);
  EXPECT_TRUE(a->sample_rate == 16000);
  const size_t n = a->pcm.size() / 2;
  ASSERT_TRUE(n > 0);
  // Planar: the whole LEFT run first, then the whole RIGHT one. An
  // interleaved result would put silence in every other sample of the
  // first half and fail the energy check on both.
  double el = 0.0, er = 0.0;
  for (size_t i = 0; i < n; ++i) {
    el += a->pcm[i] * a->pcm[i];
    er += a->pcm[n + i] * a->pcm[n + i];
  }
  EXPECT_TRUE(el > 0.01 * static_cast<double>(n));
  EXPECT_TRUE(er < 1e-6 * static_cast<double>(n));

  // Mono is still the default and still one flat run.
  auto m = decode_audio_bytes(
      libs, span<const uint8_t>(wav.data(), wav.size()), 16000, &err);
  ASSERT_TRUE(m.has_value());
  EXPECT_TRUE(m->channels == 1);
  EXPECT_TRUE(m->pcm.size() == n);
  std::printf("[media_decode] stereo: %zu samples/ch, left energy %.4f, "
              "right %.6f\n", n, el / static_cast<double>(n),
              er / static_cast<double>(n));
}

TEST(media_decode, video_file_frames_rate_and_bound) {
  Session s;
  const FFmpegLibraries* libs = nullptr;
  try {
    libs = s.ffmpeg_libraries();
  } catch (...) {
    return;
  }
  if (!libs || !libs->valid()) { return; }

  const auto y4m = make_y4m_(/*frames=*/10, /*w=*/8, /*h=*/4, 24, 1);
  const string path = write_temp_(y4m, "vpipe-media-decode-test.y4m");
  ASSERT_TRUE(!path.empty());

  string err;
  auto v = decode_video_file(libs, path, /*max_seconds=*/0.0,
                             /*audio_sample_rate=*/0, &err);
  ASSERT_TRUE(v.has_value());
  EXPECT_TRUE(v->num_frames == 10);
  EXPECT_TRUE(v->width == 8 && v->height == 4);
  EXPECT_TRUE(std::fabs(v->fps - 24.0) < 1e-6);
  EXPECT_TRUE(v->audio_channels == 0);        // no audio stream
  ASSERT_TRUE(v->rgb.size() == 10u * 3u * 4u * 8u);
  // Frame-major planar [3,H,W] planes, in order: the synthesized luma
  // rises frame by frame, so a decode that shuffled or dropped frames
  // would break the monotonicity.
  const size_t plane = 3u * 4u * 8u;
  bool rising = true;
  for (int f = 1; f < 10; ++f) {
    if (v->rgb[f * plane] <= v->rgb[(f - 1) * plane]) { rising = false; }
  }
  EXPECT_TRUE(rising);

  // `max_seconds` is a memory bound, not a nicety: planar RGB is 6 MB a
  // 1080p frame. Two frames of slack are deliberate (the rate resample
  // downstream can round the last slot past the cut).
  auto cut = decode_video_file(libs, path, /*max_seconds=*/0.125,
                               /*audio_sample_rate=*/0, &err);
  ASSERT_TRUE(cut.has_value());
  EXPECT_TRUE(cut->num_frames <= 5);
  EXPECT_TRUE(cut->num_frames >= 3);
  std::printf("[media_decode] y4m: %d frames at %.2f fps %dx%d; a 0.125 s "
              "bound reads %d\n", v->num_frames, v->fps, v->width, v->height,
              cut->num_frames);
  std::filesystem::remove(path);
}

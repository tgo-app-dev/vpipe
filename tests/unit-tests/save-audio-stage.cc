// SaveAudioStage tests.
//
// Always-on: config plumbing (kTypeName, fail_config on missing
// output_path, format inferred from extension) + the WAV path
// end-to-end (encode a synthetic 1 s sine to a temp .wav, re-read the
// 44-byte header, assert sample_rate / channels / bits + the data chunk
// holds ~1 s of samples).
//
// ffmpeg-gated: AAC + MP3 + M4A. When session()->ffmpeg_libraries() is
// valid the test encodes to temp files, asserts the file is non-trivially
// sized, then RE-OPENS each with ffmpeg (avformat_open_input +
// find_stream_info) to confirm a valid audio stream with ~1 s duration
// and the right codec.

#include "minitest.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/ffmpeg-libraries.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "common/vertex.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/save-audio-stage.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include <streambuf>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

using namespace std;
using namespace vpipe;

namespace {

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

std::string
make_tempdir_()
{
  auto base = std::filesystem::temp_directory_path()
            / "vpipe-save-audio-XXXXXX";
  std::string tmpl = base.string();
  if (!::mkdtemp(tmpl.data())) {
    throw std::runtime_error("mkdtemp failed");
  }
  return tmpl;
}

// One mono f32 PCM clip (a 440 Hz sine, 1 s @ 24000 Hz) as a rank-1
// TensorBeat with sideband.sample_rate set, then signals done.
class OneSineSource : public TypedStage<OneSineSource> {
public:
  static constexpr const char* kTypeName = "ut-audio-encode-sine";
  using TypedStage::TypedStage;

  int sr = 24000;
  int n  = 24000;   // 1 s

  Job process(RuntimeContext& ctx) override
  {
    if (_sent) { ctx.signal_done(); co_return; }
    _sent = true;
    TensorBeat tb;
    tb.dtype = TensorBeat::DType::F32;
    tb.shape = { n };
    tb.resize_contiguous(static_cast<size_t>(n));
    float* p = tb.as_f32();
    const float two_pi = 2.0f * static_cast<float>(std::numbers::pi);
    for (int i = 0; i < n; ++i) {
      const float t = static_cast<float>(i) / static_cast<float>(sr);
      p[i] = 0.6f * std::sin(two_pi * 440.0f * t);
    }
    tb.sideband = FlexData::make_object();
    tb.sideband.as_object().insert("sample_rate",
                                   FlexData::make_int(sr));
    co_await ctx.write(0, make_payload<TensorBeatPayload>(std::move(tb)));
  }

private:
  bool _sent = false;
};

// Read a little-endian u16/u32 from a byte buffer at `off`.
std::uint16_t
rd16_(const std::vector<unsigned char>& b, std::size_t off)
{
  return static_cast<std::uint16_t>(b[off])
       | static_cast<std::uint16_t>(b[off + 1]) << 8;
}
std::uint32_t
rd32_(const std::vector<unsigned char>& b, std::size_t off)
{
  return static_cast<std::uint32_t>(b[off])
       | static_cast<std::uint32_t>(b[off + 1]) << 8
       | static_cast<std::uint32_t>(b[off + 2]) << 16
       | static_cast<std::uint32_t>(b[off + 3]) << 24;
}

std::vector<unsigned char>
slurp_(const std::string& path)
{
  std::ifstream in(path, std::ios::binary);
  return std::vector<unsigned char>(
      std::istreambuf_iterator<char>(in),
      std::istreambuf_iterator<char>());
}

// Drive a one-beat pipeline: the sine source -> a SaveAudioStage with
// `cfg`. Returns true iff the runtime launched (assertions are made by
// the caller, since ASSERT_TRUE only works inside a TEST body).
bool
run_one_(Session& sess, FlexData cfg)
{
  Pipeline pl("save-audio-smoke", &sess);
  auto src = make_unique<OneSineSource>(
      &sess, "sine", vector<InEdge>{}, FlexData::make_object());
  src->allocate_oports(1);
  auto* sinesrc =
      static_cast<OneSineSource*>(pl.insert_stage(std::move(src)));

  auto enc = make_unique<SaveAudioStage>(
      &sess, "enc", vector<InEdge>{ { sinesrc, 0 } }, std::move(cfg));
  pl.insert_stage(std::move(enc));

  PipelineRuntime rt(&pl, &sess);
  const bool launched = rt.launch();
  if (launched) {
    rt.wait_idle();
    rt.stop();
  }
  return launched;
}

}  // namespace

// ---- Always-on config tests ------------------------------------------

TEST(save_audio_stage, type_is_registered) {
  EXPECT_TRUE(string_view(SaveAudioStage::kTypeName) == "save-audio");
}

TEST(save_audio_stage, missing_output_path_deferred) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::make_object();
  SaveAudioStage s(&sess, "enc", vector<InEdge>{}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

TEST(save_audio_stage, format_inferred_from_extension) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(R"({"output_path":"/tmp/out.mp3"})");
  SaveAudioStage s(&sess, "enc", vector<InEdge>{}, std::move(cfg));
  EXPECT_TRUE(s.config_error().empty());
  EXPECT_TRUE(s.format() == "mp3");
  EXPECT_TRUE(s.bitrate() == 128000);
}

TEST(save_audio_stage, format_default_wav_when_no_extension) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(R"({"output_path":"/tmp/out"})");
  SaveAudioStage s(&sess, "enc", vector<InEdge>{}, std::move(cfg));
  EXPECT_TRUE(s.config_error().empty());
  EXPECT_TRUE(s.format() == "wav");
}

TEST(save_audio_stage, explicit_format_overrides_extension) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(
      R"({"output_path":"/tmp/out.dat","format":"aac","bitrate":96000})");
  SaveAudioStage s(&sess, "enc", vector<InEdge>{}, std::move(cfg));
  EXPECT_TRUE(s.config_error().empty());
  EXPECT_TRUE(s.format() == "aac");
  EXPECT_TRUE(s.bitrate() == 96000);
}

TEST(save_audio_stage, bad_format_deferred) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(
      R"({"output_path":"/tmp/out","format":"flac"})");
  SaveAudioStage s(&sess, "enc", vector<InEdge>{}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

TEST(save_audio_stage, spec_ports) {
  Session sess;
  CerrSilencer hush;
  SaveAudioStage s(&sess, "enc", vector<InEdge>{ { nullptr, 0 } },
                     FlexData::from_json(R"({"output_path":"/tmp/o.wav"})"));
  const StageSpec& sp = s.spec();
  EXPECT_TRUE(sp.iports.size() == 1u);
  EXPECT_TRUE(sp.oports.size() == 1u);
  if (sp.iports.size() == 1u) {
    EXPECT_TRUE(sp.iports[0].name == "pcm");
  }
}

// ---- WAV end-to-end (always-on; no ffmpeg) ---------------------------

TEST(save_audio_stage, wav_roundtrip) {
  const std::string dir = make_tempdir_();
  const std::string wav = dir + "/sine.wav";
  {
    Session sess;
    CerrSilencer hush;
    FlexData cfg = FlexData::make_object();
    cfg.as_object().insert("output_path", FlexData::make_string(wav));
    ASSERT_TRUE(run_one_(sess, std::move(cfg)));
  }

  // Parse the 44-byte header.
  std::vector<unsigned char> b = slurp_(wav);
  ASSERT_TRUE(b.size() > 44u);
  EXPECT_TRUE(std::memcmp(b.data(), "RIFF", 4) == 0);
  EXPECT_TRUE(std::memcmp(b.data() + 8, "WAVE", 4) == 0);
  EXPECT_TRUE(std::memcmp(b.data() + 12, "fmt ", 4) == 0);
  const std::uint16_t audio_fmt = rd16_(b, 20);
  const std::uint16_t channels  = rd16_(b, 22);
  const std::uint32_t sr        = rd32_(b, 24);
  const std::uint16_t bits      = rd16_(b, 34);
  EXPECT_TRUE(audio_fmt == 1);          // PCM
  EXPECT_TRUE(channels == 1);
  EXPECT_TRUE(sr == 24000);
  EXPECT_TRUE(bits == 16);
  EXPECT_TRUE(std::memcmp(b.data() + 36, "data", 4) == 0);
  const std::uint32_t data_bytes = rd32_(b, 40);
  // 24000 mono int16 samples = 48000 bytes (allow exact).
  EXPECT_TRUE(data_bytes == 48000u);
  std::printf("[save-audio] wav: %zu bytes, %u Hz, %u ch, %u-bit, "
              "data=%u bytes (~%.2fs)\n",
              b.size(), sr, channels, bits, data_bytes,
              data_bytes / 2.0 / sr);

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

// ---- AAC / MP3 / M4A end-to-end (ffmpeg-gated) -----------------------

namespace {

// Re-open `path` with ffmpeg, return {duration_s, codec_id} for the
// first audio stream (duration_s < 0 on failure). Logs nothing.
struct ProbeResult {
  double duration_s = -1.0;
  int    codec_id   = -1;
};

ProbeResult
ffprobe_(const FFmpegLibraries& libs, const std::string& path)
{
  ProbeResult r;
  const auto& fmt_api = libs.avformat().api;
  AVFormatContext* ic = nullptr;
  if (fmt_api.open_input(&ic, path.c_str(), nullptr, nullptr) < 0
      || !ic) {
    return r;
  }
  if (fmt_api.find_stream_info(ic, nullptr) < 0) {
    fmt_api.close_input(&ic);
    return r;
  }
  for (unsigned i = 0; i < ic->nb_streams; ++i) {
    AVStream* st = ic->streams[i];
    if (st->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
      r.codec_id = st->codecpar->codec_id;
      if (st->duration != AV_NOPTS_VALUE && st->time_base.den > 0) {
        r.duration_s = st->duration
            * static_cast<double>(st->time_base.num)
            / st->time_base.den;
      }
      break;
    }
  }
  if (r.duration_s < 0 && ic->duration != AV_NOPTS_VALUE) {
    r.duration_s = ic->duration / static_cast<double>(AV_TIME_BASE);
  }
  fmt_api.close_input(&ic);
  return r;
}

// Encode the synthetic sine to `dir/sine.<ext>` with `format`, then
// ffprobe it. Returns {file_size, probe}. Assertions are made by the
// caller in the TEST body.
struct EncodeProbe {
  bool        launched = false;
  std::size_t size     = 0;
  ProbeResult probe;
};

EncodeProbe
encode_and_probe_(const FFmpegLibraries& libs, const std::string& dir,
                  const std::string& ext, const std::string& format)
{
  EncodeProbe r;
  const std::string out = dir + "/sine." + ext;
  {
    Session sess;
    CerrSilencer hush;
    FlexData cfg = FlexData::make_object();
    {
      auto o = cfg.as_object();
      o.insert("output_path", FlexData::make_string(out));
      o.insert("format", FlexData::make_string(format));
      o.insert("bitrate", FlexData::make_int(128000));
    }
    r.launched = run_one_(sess, std::move(cfg));
  }
  std::error_code ec;
  const auto sz = std::filesystem::file_size(out, ec);
  r.size  = ec ? 0 : static_cast<std::size_t>(sz);
  r.probe = ffprobe_(libs, out);
  return r;
}

}  // namespace

TEST(save_audio_stage, aac_mp3_m4a_roundtrip) {
  Session probe_sess;
  const FFmpegLibraries* libs = probe_sess.ffmpeg_libraries();
  if (!libs || !libs->valid()) {
    std::printf("[save-audio] ffmpeg unavailable; skipping AAC/MP3/M4A "
                "cases\n");
    return;
  }
  const std::string dir = make_tempdir_();

  struct Case { const char* ext; const char* format; int want_codec; };
  const Case cases[] = {
    { "aac", "aac", AV_CODEC_ID_AAC },
    { "m4a", "m4a", AV_CODEC_ID_AAC },
    { "mp3", "mp3", AV_CODEC_ID_MP3 },
  };
  for (const Case& c : cases) {
    const EncodeProbe r = encode_and_probe_(*libs, dir, c.ext, c.format);
    std::printf("[save-audio] %s: launched=%d %zu bytes, probe "
                "dur=%.2fs codec=%d (want %d)\n",
                c.format, static_cast<int>(r.launched), r.size,
                r.probe.duration_s, r.probe.codec_id, c.want_codec);
    EXPECT_TRUE(r.launched);
    EXPECT_TRUE(r.size > 1024u);               // non-trivial
    EXPECT_TRUE(r.probe.codec_id == c.want_codec);
    // ~1 s, allow encoder priming/padding slack.
    EXPECT_TRUE(r.probe.duration_s > 0.8 && r.probe.duration_s < 1.3);
  }

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

// Multiple beats append an incrementing index before the extension. Drive
// two sine clips through one stage instance and assert both files exist.
namespace {

class TwoSineSource : public TypedStage<TwoSineSource> {
public:
  static constexpr const char* kTypeName = "ut-audio-encode-sine2";
  using TypedStage::TypedStage;

  int sr = 24000;
  int n  = 12000;   // 0.5 s each

  Job process(RuntimeContext& ctx) override
  {
    if (_sent >= 2) { ctx.signal_done(); co_return; }
    ++_sent;
    TensorBeat tb;
    tb.dtype = TensorBeat::DType::F32;
    tb.shape = { n };
    tb.resize_contiguous(static_cast<size_t>(n));
    float* p = tb.as_f32();
    const float two_pi = 2.0f * static_cast<float>(std::numbers::pi);
    for (int i = 0; i < n; ++i) {
      const float t = static_cast<float>(i) / static_cast<float>(sr);
      p[i] = 0.5f * std::sin(two_pi * 440.0f * t);
    }
    tb.sideband = FlexData::make_object();
    tb.sideband.as_object().insert("sample_rate",
                                   FlexData::make_int(sr));
    co_await ctx.write(0, make_payload<TensorBeatPayload>(std::move(tb)));
  }

private:
  int _sent = 0;
};

}  // namespace

TEST(save_audio_stage, multiple_beats_index_suffix) {
  const std::string dir = make_tempdir_();
  const std::string wav = dir + "/clip.wav";
  {
    Session  sess;
    CerrSilencer hush;
    Pipeline pl("save-audio-multi", &sess);
    auto src = make_unique<TwoSineSource>(
        &sess, "sine2", vector<InEdge>{}, FlexData::make_object());
    src->allocate_oports(1);
    auto* s2 = static_cast<TwoSineSource*>(pl.insert_stage(std::move(src)));

    FlexData cfg = FlexData::make_object();
    cfg.as_object().insert("output_path", FlexData::make_string(wav));
    auto enc = make_unique<SaveAudioStage>(
        &sess, "enc", vector<InEdge>{ { s2, 0 } }, std::move(cfg));
    auto* enc_stage =
        static_cast<SaveAudioStage*>(pl.insert_stage(std::move(enc)));

    PipelineRuntime rt(&pl, &sess);
    ASSERT_TRUE(rt.launch());
    rt.wait_idle();
    rt.stop();
    EXPECT_TRUE(enc_stage->files_written() == 2u);
  }
  // First file is the plain path, second gets a "-001" index suffix.
  EXPECT_TRUE(std::filesystem::exists(dir + "/clip.wav"));
  EXPECT_TRUE(std::filesystem::exists(dir + "/clip-001.wav"));
  std::printf("[save-audio] multi: clip.wav + clip-001.wav present\n");

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

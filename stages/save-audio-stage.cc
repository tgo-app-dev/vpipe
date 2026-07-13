#include "stages/save-audio-stage.h"

#include "common/beat-payload-intf.h"
#include "common/ffmpeg-libraries.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include "apple-silicon/tensor-beat.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

namespace vpipe {

namespace {

// Lower-case + trim an ASCII string in place (formats are short ASCII).
std::string
ascii_lower_(std::string s)
{
  for (char& c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

// Return the lower-cased extension (without the dot) of `path`, or "" if
// there is none. Only the final path component is considered, so a
// dotted directory name does not count.
std::string
extension_of_(const std::string& path)
{
  const std::size_t slash = path.find_last_of('/');
  const std::size_t start = (slash == std::string::npos) ? 0 : slash + 1;
  const std::size_t dot   = path.find_last_of('.');
  if (dot == std::string::npos || dot < start || dot + 1 >= path.size()) {
    return "";
  }
  return ascii_lower_(path.substr(dot + 1));
}

// "wav"/"aac"/"mp3"/"m4a" -> true. Anything else is rejected.
bool
is_known_format_(const std::string& f)
{
  return f == "wav" || f == "aac" || f == "mp3" || f == "m4a";
}

// Small planar-float FIFO sitting between swresample's output and the
// audio encoder's input. The encoder requires exactly frame_size
// samples per send_frame call (the native FFmpeg AAC encoder lacks
// AV_CODEC_CAP_VARIABLE_FRAME_SIZE), but an arbitrary input clip is some
// other length. The FIFO accumulates planar floats until a full encoder
// frame is available, then hands it out. (Copied from the rtsp-capture
// audio transcode path.)
struct AudioPlanarFloatFifo {
  int                             channels   = 0;
  int                             frame_size = 0;
  std::vector<std::vector<float>> planes;

  void reset(int nb_channels, int nb_samples_per_frame)
  {
    channels   = nb_channels;
    frame_size = nb_samples_per_frame;
    planes.assign(static_cast<size_t>(channels), {});
  }

  int filled() const
  {
    return planes.empty()
         ? 0
         : static_cast<int>(planes.front().size());
  }

  // Append nb_samples planar floats from src[0..channels-1]. Caller must
  // guarantee src has `channels` valid planes.
  void push_planar(const uint8_t* const* src, int nb_samples)
  {
    for (int ch = 0; ch < channels; ++ch) {
      auto* f = reinterpret_cast<const float*>(src[ch]);
      planes[static_cast<size_t>(ch)]
          .insert(planes[static_cast<size_t>(ch)].end(),
                  f, f + nb_samples);
    }
  }

  // Drain one full frame_size-sample frame into dst[0..channels-1].
  // Returns true on success, false if the FIFO doesn't hold enough.
  bool pull_frame(uint8_t* const* dst)
  {
    if (filled() < frame_size) {
      return false;
    }
    for (int ch = 0; ch < channels; ++ch) {
      auto& src = planes[static_cast<size_t>(ch)];
      std::memcpy(dst[ch], src.data(),
                  static_cast<size_t>(frame_size) * sizeof(float));
      src.erase(src.begin(), src.begin() + frame_size);
    }
    return true;
  }

  // Drain a final partial frame, zero-padding the tail to frame_size.
  // Returns the number of REAL (non-pad) samples drained, or 0 if empty.
  int pull_padded_frame(uint8_t* const* dst)
  {
    const int have = filled();
    if (have <= 0) {
      return 0;
    }
    const int real = std::min(have, frame_size);
    for (int ch = 0; ch < channels; ++ch) {
      auto& src = planes[static_cast<size_t>(ch)];
      std::memcpy(dst[ch], src.data(),
                  static_cast<size_t>(real) * sizeof(float));
      if (real < frame_size) {
        std::memset(reinterpret_cast<float*>(dst[ch]) + real, 0,
                    static_cast<size_t>(frame_size - real)
                        * sizeof(float));
      }
      src.erase(src.begin(), src.begin() + real);
    }
    return real;
  }
};

constexpr ConfigKey kAttrs[] = {
  {.key = "output_path", .type = ConfigType::String, .required = true,
   .doc = "output file path; when it has no extension, one is appended "
          "from `format`",
   .is_path = true, .path_write = true, .path_filter = "audio"},
  {.key = "format", .type = ConfigType::String,
   .doc = "wav | aac | mp3 | m4a (m4a => AAC in mp4). Default: inferred "
          "from the output_path extension, else wav",
   .def_str = ""},
  {.key = "bitrate", .type = ConfigType::Int,
   .doc = "AAC / MP3 target bitrate (bits/s)", .def_int = 128000},
  {.key = "sample_rate", .type = ConfigType::Int,
   .doc = "0 = use the payload sideband.sample_rate; if neither present, "
          "24000",
   .def_int = 0},
};
const PortSpec kIports[] = {
  {.name = "pcm",
   .doc = "TensorBeat f32 PCM, rank-1 [n_samples] (mono) or rank-2 "
          "[channels, n_samples]; sideband.sample_rate honoured. One "
          "complete clip per beat.",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "info",
   .doc = "OPTIONAL FlexData object describing the file just written: "
          "{path, format, samples, sample_rate, duration_s}. Downstream "
          "consumers are optional.",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "save-audio",
  .doc       = "Sink: encodes each incoming PCM clip (one beat = one clip) "
               "to a file: AAC / MP3 / M4A via FFmpeg, or 16-bit WAV "
               "directly. The audio counterpart to save-image; pairs with "
               "text-to-speech. Multiple beats append an incrementing index "
               "before the extension.",
  .display_name = "Save Audio",
  .category  = StageCategory::Audio,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

SaveAudioStage::SaveAudioStage(const SessionContextIntf* s,
                               std::string               id,
                               std::vector<InEdge>       iports,
                               FlexData                  config)
  : TypedStage<SaveAudioStage>(s, std::move(id), std::move(iports),
                               std::move(config))
{
  // Construction must succeed for any config (see Stage::fail_config):
  // a stage must construct so a graph can be built/edited before
  // required fields are supplied. Config problems are recorded via
  // fail_config (first message wins) and deferred to launch.
  _output_path = attr_path("output_path", true);
  _bitrate     = static_cast<int>(attr_int("bitrate"));
  _sample_rate = static_cast<int>(attr_int("sample_rate"));

  // Resolve the format: explicit `format` wins; else infer from the
  // output_path extension; else "wav".
  std::string fmt_cfg = ascii_lower_(attr_str("format"));
  if (fmt_cfg.empty()) {
    const std::string ext = extension_of_(_output_path);
    fmt_cfg = ext.empty() ? std::string("wav") : ext;
  }
  _format = fmt_cfg;

  if (_output_path.empty()) {
    fail_config(fmt(
        "SaveAudioStage('{}'): config.output_path is required "
        "(non-empty string)", this->id()));
  }
  if (!is_known_format_(_format)) {
    fail_config(fmt(
        "SaveAudioStage('{}'): unsupported format '{}' (expected one "
        "of wav | aac | mp3 | m4a)", this->id(), _format));
  }
  if (_bitrate <= 0) {
    fail_config(fmt(
        "SaveAudioStage('{}'): bitrate must be > 0 (got {})",
        this->id(), _bitrate));
  }
  if (_sample_rate < 0) {
    fail_config(fmt(
        "SaveAudioStage('{}'): sample_rate must be >= 0 (got {})",
        this->id(), _sample_rate));
  }

  // Always allocate the info oport. When nothing is wired downstream the
  // runtime allocates an OportBuffer with no cursors and writes are
  // silently dropped, so a sink-style pipeline still works.
  allocate_oports(spec().oports.size());
}

SaveAudioStage::~SaveAudioStage() = default;

const StageSpec&
SaveAudioStage::spec() const noexcept
{
  return kSpec;
}

std::string
SaveAudioStage::next_output_path_(const std::string& ext) const
{
  // Start from output_path; ensure it ends in `.ext`. For the 2nd+ file
  // insert "-NNN" before the extension to avoid overwrite.
  std::string base = _output_path;
  // Strip a trailing extension equal to one of the known formats so we
  // can re-attach `ext` consistently (the configured extension may be
  // .m4a while the inferred container ext differs only by request).
  std::string have_ext = extension_of_(base);
  if (!have_ext.empty() && is_known_format_(have_ext)) {
    base.erase(base.size() - have_ext.size() - 1);  // drop ".ext"
  }
  std::string out = base;
  if (_files_written > 0) {
    char idx[16];
    std::snprintf(idx, sizeof(idx), "-%03llu",
                  static_cast<unsigned long long>(_files_written));
    out += idx;
  }
  out += ".";
  out += ext;
  return out;
}

bool
SaveAudioStage::encode_wav_(const std::string& path, const float* pcm,
                              std::size_t n_frames, int channels,
                              int sample_rate)
{
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    session()->error(fmt(
        "SaveAudioStage('{}'): cannot open '{}' for writing",
        this->id(), path));
    return false;
  }
  // 16-bit PCM WAV (see metal-lm-smoke moss_tts_end_to_end_wav). For
  // multi-channel, samples are interleaved in `pcm` already.
  const std::uint16_t ch        = static_cast<std::uint16_t>(channels);
  const std::uint16_t bits      = 16;
  const std::uint16_t block_al  = static_cast<std::uint16_t>(channels * 2);
  const std::uint32_t byte_rate =
      static_cast<std::uint32_t>(sample_rate) * block_al;
  const std::uint32_t n_total =
      static_cast<std::uint32_t>(n_frames)
      * static_cast<std::uint32_t>(channels);
  const std::uint32_t data_bytes = n_total * 2;
  auto u32 = [&](std::uint32_t v) {
    out.write(reinterpret_cast<const char*>(&v), 4);
  };
  auto u16 = [&](std::uint16_t v) {
    out.write(reinterpret_cast<const char*>(&v), 2);
  };
  out.write("RIFF", 4); u32(36 + data_bytes); out.write("WAVE", 4);
  out.write("fmt ", 4); u32(16); u16(1); u16(ch);
  u32(static_cast<std::uint32_t>(sample_rate)); u32(byte_rate);
  u16(block_al); u16(bits);
  out.write("data", 4); u32(data_bytes);
  for (std::uint32_t i = 0; i < n_total; ++i) {
    const float s = std::max(-1.0f, std::min(1.0f, pcm[i]));
    const int v = static_cast<int>(std::lround(s * 32767.0f));
    u16(static_cast<std::uint16_t>(static_cast<std::int16_t>(v)));
  }
  out.close();
  if (!out) {
    session()->error(fmt(
        "SaveAudioStage('{}'): write error finalizing '{}'",
        this->id(), path));
    return false;
  }
  return true;
}

bool
SaveAudioStage::encode_ffmpeg_(const std::string& path, const float* pcm,
                                 std::size_t n_frames, int channels,
                                 int sample_rate, const std::string& format)
{
  const FFmpegLibraries* libs = session()->ffmpeg_libraries();
  if (!libs || !libs->valid()) {
    session()->error(fmt(
        "SaveAudioStage('{}'): ffmpeg unavailable for format '{}'",
        this->id(), format));
    return false;
  }
  const auto& util_api = libs->avutil().api;
  const auto& cdc_api  = libs->avcodec().api;
  const auto& fmt_api  = libs->avformat().api;
  const auto& swr_api  = libs->swresample().api;

  // RAII-ish cleanup: collect everything we allocate, free on the way
  // out via a single goto-free `done:`-style lambda. We keep raw
  // pointers + a `ok` flag for explicit, ordered teardown.
  AVCodecContext*  a_enc        = nullptr;
  AVFormatContext* ofctx        = nullptr;
  SwrContext*      swr          = nullptr;
  AVFrame*         enc_in_frame = nullptr;
  AVFrame*         swr_out      = nullptr;
  AVPacket*        pkt          = nullptr;
  bool             pb_opened    = false;
  bool             ok           = false;

  auto cleanup = [&]() {
    if (pkt)          { cdc_api.packet_free(&pkt); }
    if (swr_out)      { util_api.frame_free(&swr_out); }
    if (enc_in_frame) { util_api.frame_free(&enc_in_frame); }
    if (swr)          { swr_api.free(&swr); }
    if (a_enc)        { cdc_api.free_context(&a_enc); }
    if (ofctx) {
      if (pb_opened && ofctx->pb) { fmt_api.avio_closep(&ofctx->pb); }
      fmt_api.free_context(ofctx);
      ofctx = nullptr;
    }
  };
  auto fail = [&](const std::string& msg) -> bool {
    session()->error(fmt("SaveAudioStage('{}'): {}", this->id(), msg));
    cleanup();
    return false;
  };

  // -- 1. Pick + open the encoder. ---
  const bool is_mp3 = (format == "mp3");
  const AVCodec* enc = nullptr;
  if (is_mp3) {
    enc = cdc_api.find_encoder_by_name("libmp3lame");
    if (!enc) { enc = cdc_api.find_encoder(AV_CODEC_ID_MP3); }
  } else {
    enc = cdc_api.find_encoder(AV_CODEC_ID_AAC);
  }
  if (!enc) {
    return fail(std::format("no {} encoder available in this ffmpeg build",
                    is_mp3 ? "MP3" : "AAC"));
  }
  a_enc = cdc_api.alloc_context3(enc);
  if (!a_enc) { return fail(std::format("avcodec_alloc_context3 failed")); }

  // Encoder sample_fmt: both AAC and libmp3lame accept FLTP. swresample
  // produces FLTP from our interleaved-f32 input, so FLTP is the uniform
  // choice; if open2 rejects it for MP3 (e.g. the native MP3 encoder,
  // which wants S16P), retry with S16P below.
  AVSampleFormat enc_fmt = AV_SAMPLE_FMT_FLTP;

  // Channel layout: 1 -> mono, 2 -> stereo, else default for the count.
  AVChannelLayout out_layout;
  std::memset(&out_layout, 0, sizeof(out_layout));
  if (channels == 1) {
    out_layout = AV_CHANNEL_LAYOUT_MONO;
  } else if (channels == 2) {
    out_layout = AV_CHANNEL_LAYOUT_STEREO;
  } else {
    out_layout = AV_CHANNEL_LAYOUT_MASK(
        channels, (1ULL << channels) - 1);
  }

  a_enc->sample_rate = sample_rate;
  a_enc->ch_layout   = out_layout;
  a_enc->sample_fmt  = enc_fmt;
  a_enc->bit_rate    = static_cast<int64_t>(_bitrate);
  a_enc->time_base   = AVRational{1, sample_rate};

  if (cdc_api.open2(a_enc, enc, nullptr) < 0) {
    // Retry MP3 with S16P (the native MP3 encoder's preferred format).
    bool reopened = false;
    if (is_mp3) {
      cdc_api.free_context(&a_enc);
      a_enc = cdc_api.alloc_context3(enc);
      if (a_enc) {
        a_enc->sample_rate = sample_rate;
        a_enc->ch_layout   = out_layout;
        a_enc->sample_fmt  = AV_SAMPLE_FMT_S16P;
        a_enc->bit_rate    = static_cast<int64_t>(_bitrate);
        a_enc->time_base   = AVRational{1, sample_rate};
        reopened = (cdc_api.open2(a_enc, enc, nullptr) >= 0);
      }
    }
    if (!reopened) {
      return fail(std::format(
          "avcodec_open2 failed (sample_rate {} / channels {} may be "
          "unsupported by {})", sample_rate, channels,
          is_mp3 ? "MP3" : "AAC"));
    }
  }
  const int frame_size =
      a_enc->frame_size > 0 ? a_enc->frame_size : 1024;

  // -- 2. swresample: interleaved f32 input -> the encoder's format. ---
  AVChannelLayout in_layout;
  std::memset(&in_layout, 0, sizeof(in_layout));
  if (channels == 1) {
    in_layout = AV_CHANNEL_LAYOUT_MONO;
  } else if (channels == 2) {
    in_layout = AV_CHANNEL_LAYOUT_STEREO;
  } else {
    in_layout = AV_CHANNEL_LAYOUT_MASK(channels, (1ULL << channels) - 1);
  }
  if (swr_api.alloc_set_opts2(
          &swr,
          &a_enc->ch_layout, a_enc->sample_fmt, a_enc->sample_rate,
          &in_layout, AV_SAMPLE_FMT_FLT, sample_rate,
          0, nullptr) < 0
      || !swr) {
    return fail(std::format("swr_alloc_set_opts2 failed"));
  }
  if (swr_api.init(swr) < 0) {
    return fail(std::format("swr_init failed"));
  }

  // -- 3. Muxer: infer the container from the path extension. ---
  if (fmt_api.alloc_output_context2(&ofctx, nullptr, nullptr,
                                    path.c_str()) < 0
      || !ofctx) {
    return fail(std::format("avformat_alloc_output_context2 failed for '{}'",
                    path));
  }
  AVStream* stream = fmt_api.new_stream(ofctx, enc);
  if (!stream) { return fail(std::format("avformat_new_stream failed")); }
  if (cdc_api.parameters_from_context(stream->codecpar, a_enc) < 0) {
    return fail(std::format("avcodec_parameters_from_context failed"));
  }
  stream->time_base = a_enc->time_base;
  if (!(ofctx->oformat->flags & AVFMT_NOFILE)) {
    if (fmt_api.avio_open(&ofctx->pb, path.c_str(),
                          AVIO_FLAG_WRITE) < 0) {
      return fail(std::format("avio_open failed for '{}'", path));
    }
    pb_opened = true;
  }
  if (fmt_api.write_header(ofctx, nullptr) < 0) {
    return fail(std::format("avformat_write_header failed for '{}'", path));
  }

  // -- 4. Scratch frames + packet. ---
  pkt = cdc_api.packet_alloc();
  if (!pkt) { return fail(std::format("av_packet_alloc failed")); }

  enc_in_frame = util_api.frame_alloc();
  if (!enc_in_frame) { return fail(std::format("av_frame_alloc failed")); }
  enc_in_frame->nb_samples  = frame_size;
  enc_in_frame->format      = a_enc->sample_fmt;
  enc_in_frame->sample_rate = a_enc->sample_rate;
  enc_in_frame->ch_layout   = a_enc->ch_layout;
  if (util_api.frame_get_buffer(enc_in_frame, 0) < 0) {
    return fail(std::format("frame_get_buffer (encoder input) failed"));
  }

  // swr scratch: convert the WHOLE clip at once. swr may emit up to
  // out_count = n_frames * out_rate / in_rate + delay; here in/out rate
  // match so n_frames + a small slack is plenty.
  const int swr_cap = static_cast<int>(n_frames) + 1024;
  swr_out = util_api.frame_alloc();
  if (!swr_out) { return fail(std::format("av_frame_alloc (swr) failed")); }
  swr_out->nb_samples  = swr_cap;
  swr_out->format      = a_enc->sample_fmt;
  swr_out->sample_rate = a_enc->sample_rate;
  swr_out->ch_layout   = a_enc->ch_layout;
  if (util_api.frame_get_buffer(swr_out, 0) < 0) {
    return fail(std::format("frame_get_buffer (swr scratch) failed"));
  }

  AudioPlanarFloatFifo fifo;
  fifo.reset(channels, frame_size);
  int64_t next_pts = 0;

  // Convert the whole interleaved-f32 clip through swr -> planar floats,
  // push into the FIFO.
  {
    const uint8_t* in_ptr = reinterpret_cast<const uint8_t*>(pcm);
    const uint8_t* in_planes[1] = { in_ptr };
    const int n_out = swr_api.convert(
        swr,
        swr_out->data, swr_out->nb_samples,
        in_planes, static_cast<int>(n_frames));
    if (n_out < 0) {
      return fail(std::format("swr_convert failed"));
    }
    if (n_out > 0) {
      fifo.push_planar(swr_out->data, n_out);
    }
    // Flush any samples swr buffered internally (none for matched rates,
    // but uniform handling makes resample just work).
    for (;;) {
      const int n_flush = swr_api.convert(
          swr, swr_out->data, swr_out->nb_samples, nullptr, 0);
      if (n_flush <= 0) { break; }
      fifo.push_planar(swr_out->data, n_flush);
    }
  }

  // Send-frame + drain-packets helper.
  auto drain_packets = [&]() -> bool {
    for (;;) {
      const int rc = cdc_api.receive_packet(a_enc, pkt);
      if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) { return true; }
      if (rc < 0) { return false; }
      pkt->stream_index = 0;
      cdc_api.packet_rescale_ts(pkt, a_enc->time_base, stream->time_base);
      if (fmt_api.interleaved_write_frame(ofctx, pkt) < 0) {
        cdc_api.packet_unref(pkt);
        return false;
      }
      cdc_api.packet_unref(pkt);
    }
  };

  // Pull full frames out of the FIFO and encode them.
  while (fifo.pull_frame(enc_in_frame->data)) {
    enc_in_frame->pts = next_pts;
    next_pts += frame_size;
    if (cdc_api.send_frame(a_enc, enc_in_frame) < 0) {
      return fail(std::format("avcodec_send_frame failed"));
    }
    if (!drain_packets()) {
      return fail(std::format("encode / write_frame failed"));
    }
  }
  // Final partial frame: zero-pad to frame_size (the native AAC encoder
  // needs exactly frame_size). Only send it when there were leftover
  // real samples.
  if (fifo.filled() > 0) {
    const int real = fifo.pull_padded_frame(enc_in_frame->data);
    if (real > 0) {
      enc_in_frame->pts = next_pts;
      next_pts += frame_size;
      if (cdc_api.send_frame(a_enc, enc_in_frame) < 0) {
        return fail(std::format("avcodec_send_frame (final) failed"));
      }
      if (!drain_packets()) {
        return fail(std::format("encode / write_frame (final) failed"));
      }
    }
  }

  // -- 5. Flush the encoder + finalize the file. ---
  if (cdc_api.send_frame(a_enc, nullptr) < 0) {
    return fail(std::format("avcodec_send_frame(nullptr) flush failed"));
  }
  if (!drain_packets()) {
    return fail(std::format("flush drain failed"));
  }
  if (fmt_api.write_trailer(ofctx) < 0) {
    return fail(std::format("av_write_trailer failed"));
  }

  ok = true;
  cleanup();
  return ok;
}

Job
SaveAudioStage::process(RuntimeContext& ctx)
{
  auto p = co_await ctx.read(0);
  if (!p) {
    ctx.signal_done();
    co_return;
  }
  const auto* tbp = dynamic_cast<const TensorBeatPayload*>(p.get());
  if (!tbp) {
    session()->warn(fmt(
        "SaveAudioStage('{}'): expected TensorBeatPayload on in-port 0, "
        "got {}; dropping beat", this->id(), p->describe()));
    co_return;
  }
  if (tbp->dtype != TensorBeat::DType::F32) {
    session()->warn(fmt(
        "SaveAudioStage('{}'): expected TensorBeat dtype=F32 (f32 PCM), "
        "got {}; dropping beat",
        this->id(), TensorBeat::name_of(tbp->dtype)));
    co_return;
  }

  // Accept rank-1 [n_samples] (mono) or rank-2 [channels, n_samples].
  int         channels = 1;
  std::size_t n_frames = 0;
  if (tbp->shape.size() == 1) {
    channels = 1;
    n_frames = static_cast<std::size_t>(tbp->shape[0]);
  } else if (tbp->shape.size() == 2) {
    channels = static_cast<int>(tbp->shape[0]);
    n_frames = static_cast<std::size_t>(tbp->shape[1]);
  } else {
    session()->warn(fmt(
        "SaveAudioStage('{}'): expected TensorBeat shape [N] or "
        "[channels,N], got rank={}; dropping beat",
        this->id(), static_cast<int>(tbp->shape.size())));
    co_return;
  }
  if (channels < 1 || n_frames == 0) {
    co_return;
  }

  // Sample rate: config wins when > 0, else the payload sideband, else
  // 24000.
  int sr = _sample_rate;
  if (sr <= 0) {
    sr = 24000;
    if (tbp->sideband.is_object()) {
      auto sb = tbp->sideband.as_object();
      if (sb.contains("sample_rate")) {
        sr = static_cast<int>(sb.at("sample_rate").as_int(24000));
      }
    }
  }
  if (sr <= 0) { sr = 24000; }

  // Materialize the PCM as a contiguous interleaved f32 buffer. The
  // TensorBeat for [channels, N] stores channel-major; the WAV/ffmpeg
  // paths want frame-interleaved. For the common mono case the layout is
  // identical, so we only re-interleave when channels > 1.
  AlignedVector<float> contig = tbp->materialize_contiguous_as<float>();
  if (contig.size() < n_frames * static_cast<std::size_t>(channels)) {
    session()->warn(fmt(
        "SaveAudioStage('{}'): materialized buffer too small "
        "({} floats vs {} channels * {} frames); dropping beat",
        this->id(), contig.size(), channels, n_frames));
    co_return;
  }
  std::vector<float> interleaved;
  const float* pcm = contig.data();
  if (channels > 1) {
    interleaved.resize(n_frames * static_cast<std::size_t>(channels));
    for (int ch = 0; ch < channels; ++ch) {
      const float* src = contig.data()
          + static_cast<std::size_t>(ch) * n_frames;
      for (std::size_t i = 0; i < n_frames; ++i) {
        interleaved[i * static_cast<std::size_t>(channels)
                    + static_cast<std::size_t>(ch)] = src[i];
      }
    }
    pcm = interleaved.data();
  }

  // Resolve the format + output path. A compressed format requested when
  // ffmpeg is unavailable falls back to WAV (extension rewritten).
  std::string format = _format;
  if (format != "wav") {
    const FFmpegLibraries* libs = session()->ffmpeg_libraries();
    if (!libs || !libs->valid()) {
      session()->warn(fmt(
          "SaveAudioStage('{}'): format '{}' requested but ffmpeg is "
          "unavailable; falling back to WAV",
          this->id(), format));
      format = "wav";
    }
  }
  // The container ext == format, except m4a stays "m4a".
  const std::string ext = format;
  const std::string out_path = next_output_path_(ext);

  bool wrote = false;
  if (format == "wav") {
    wrote = encode_wav_(out_path, pcm, n_frames, channels, sr);
  } else {
    wrote = encode_ffmpeg_(out_path, pcm, n_frames, channels, sr, format);
    if (!wrote) {
      // ffmpeg failed mid-encode: leave whatever partial file ffmpeg may
      // have created, but do not crash. The error was already logged.
      co_return;
    }
  }
  if (!wrote) {
    co_return;
  }

  const double duration_s =
      static_cast<double>(n_frames) / static_cast<double>(sr);
  session()->info(fmt(
      "SaveAudioStage('{}'): wrote '{}' ({}, {} frames x {} ch, "
      "{:.2f}s @ {} Hz)",
      this->id(), out_path, format, n_frames, channels, duration_s, sr));

  ++_files_written;

  // Emit the info object (unconditional; dropped if nothing is wired).
  FlexData fd = FlexData::make_object();
  {
    auto root = fd.as_object();
    root.insert("path", FlexData::make_string(out_path));
    root.insert("format", FlexData::make_string(format));
    root.insert("samples",
                FlexData::make_int(static_cast<std::int64_t>(n_frames)));
    root.insert("sample_rate", FlexData::make_int(sr));
    root.insert("duration_s", FlexData::make_real(duration_s));
  }
  co_await ctx.write(0, make_payload<FlexDataPayload>(std::move(fd)));
  co_return;
}

VPIPE_REGISTER_STAGE(SaveAudioStage)
VPIPE_REGISTER_SPEC(SaveAudioStage, kSpec)

}

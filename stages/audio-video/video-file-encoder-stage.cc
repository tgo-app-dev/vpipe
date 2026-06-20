#include "stages/audio-video/video-file-encoder-stage.h"
#include "common/beat-payload-intf.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include <stdexcept>
#include <string>
#include <utility>

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/error.h>
}

using namespace std;

namespace vpipe {

namespace {

void
fill_dict_from_options_(const FlexData& opts,
                        const LibAvUtil& avu,
                        AVDictionary**   out)
{
  if (!opts.is_object()) {
    return;
  }
  for (auto entry : opts.as_object()) {
    if (!entry.second.is_string()) {
      continue;
    }
    string key(entry.first);
    string val(entry.second.get_string());
    avu.api.dict_set(out, key.c_str(), val.c_str(), 0);
  }
}

}

VideoFileEncoderStage::VideoFileEncoderStage
  (const SessionContextIntf* s,
   string                    id,
   vector<InEdge>            iports_in,
   FlexData                  config)
  : TypedStage<VideoFileEncoderStage>(s, std::move(id),
                                      std::move(iports_in),
                                      std::move(config))
  , _libs(s->ffmpeg_libraries())
{
  // Nested video/audio defaults: no flat ConfigKey representation, so
  // they are seeded here (the single source) and overridden from the
  // video.* / audio.* sub-objects below.
  _video_codec    = "libx264";
  _video_bitrate  = 2'000'000;
  _video_preset   = "medium";
  _video_gop_size = 60;
  _audio_codec    = "aac";
  _audio_bitrate  = 128'000;

  // Flat attribute defaults live in kSpec.attrs; attr_* resolves the
  // configured value else that default.
  _output_url   = attr_str("output_url");
  _format       = attr_str("format");
  _enable_video = attr_bool("enable_video");
  _enable_audio = attr_bool("enable_audio");

  const FlexData& cfg = this->config();
  if (cfg.is_object()) {
    auto root = cfg.as_object();
    if (root.contains("video")) {
      FlexData v_val = root.at("video");
      if (v_val.is_object()) {
        auto vo = v_val.as_object();
        if (vo.contains("codec")) {
          _video_codec = string(vo.at("codec").as_string(_video_codec));
        }
        if (vo.contains("bitrate")) {
          _video_bitrate = vo.at("bitrate").as_int(_video_bitrate);
        }
        if (vo.contains("preset")) {
          _video_preset =
            string(vo.at("preset").as_string(_video_preset));
        }
        if (vo.contains("crf")) {
          _video_has_crf = true;
          _video_crf = static_cast<int>(vo.at("crf").as_int(0));
        }
        if (vo.contains("gop_size")) {
          _video_gop_size =
            static_cast<int>(vo.at("gop_size").as_int(_video_gop_size));
        }
        if (vo.contains("options")) {
          _video_options = vo.at("options");
        }
      }
    }
    if (root.contains("audio")) {
      FlexData a_val = root.at("audio");
      if (a_val.is_object()) {
        auto ao = a_val.as_object();
        if (ao.contains("codec")) {
          _audio_codec = string(ao.at("codec").as_string(_audio_codec));
        }
        if (ao.contains("bitrate")) {
          _audio_bitrate = ao.at("bitrate").as_int(_audio_bitrate);
        }
        if (ao.contains("options")) {
          _audio_options = ao.at("options");
        }
      }
    }
    if (root.contains("muxer_options")) {
      _muxer_options = root.at("muxer_options");
    }
  }

  // Validation is deferred to launch (see Stage::fail_config).
  if (_output_url.empty()) {
    fail_config(fmt(
      "VideoFileEncoderStage('{}'): config.output_url is required",
      this->id()));
  }
  if (!_enable_video && !_enable_audio) {
    fail_config(fmt(
      "VideoFileEncoderStage('{}'): at least one of enable_video / "
      "enable_audio must be true",
      this->id()));
  }

  // Assign port indices.
  unsigned next = 0;
  if (_enable_video) {
    _video_port = static_cast<int>(next++);
  }
  if (_enable_audio) {
    _audio_port = static_cast<int>(next++);
  }
  if (this->num_iports() != next) {
    fail_config(fmt(
      "VideoFileEncoderStage('{}'): expected {} input edge(s) "
      "(enable_video={}, enable_audio={}) but got {}",
      this->id(), next, _enable_video, _enable_audio,
      this->num_iports()));
  }

  allocate_oports(spec().oports.size());   // sink: 0 oports

  // Allocate scratch packet for receive_packet drains.
  _enc_pkt = _libs->avcodec().api.packet_alloc();
  if (!_enc_pkt) {
    fail_config(fmt(
      "VideoFileEncoderStage('{}'): av_packet_alloc failed",
      this->id()));
  }
}

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "output_url", .type = ConfigType::String, .required = true,
   .doc = "output file path / URL"},
  {.key = "format", .type = ConfigType::String,
   .doc = "container format; \"\" = inferred", .def_str = ""},
  {.key = "enable_video", .type = ConfigType::Bool,
   .doc = "encode a video stream", .def_bool = true},
  {.key = "enable_audio", .type = ConfigType::Bool,
   .doc = "encode an audio stream", .def_bool = true},
  {.key = "video", .type = ConfigType::Object,
   .doc = "video opts: codec/bitrate/preset/crf/gop_size/options"},
  {.key = "audio", .type = ConfigType::Object,
   .doc = "audio opts: codec/bitrate/options"},
  {.key = "muxer_options", .type = ConfigType::Object,
   .doc = "av_dict string opts for the muxer"},
};
// Canonical iports (video first, audio second); either may be disabled
// via config, so the live count comes from the wiring. Each port
// carries a StreamParams header then FrameRef beats (mixed -> untyped).
// Clock groups are authoritatively reported by iport_clock_group()
// (video 0, audio 1).
const PortSpec kIports[] = {
  {.name = "video", .doc = "VideoStreamParams header then video FrameRefs",
   .type = nullptr, .clock_group = 0},
  {.name = "audio", .doc = "AudioStreamParams header then audio FrameRefs",
   .type = nullptr, .clock_group = 1},
};
const StageSpec kSpec = {
  .type_name = "video-file-encoder",
  .doc       = "Sink: encodes incoming video/audio frames (H.264 + AAC "
               "by default) and muxes them into a container file. 0 "
               "oports; iports run on independent per-stream clocks.",
  .display_name = "Video File Writer",
  .category  = StageCategory::Video,
  .iports    = kIports,
  .oports    = {},
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
VideoFileEncoderStage::spec() const noexcept
{
  return kSpec;
}

VideoFileEncoderStage::~VideoFileEncoderStage()
{
  if (!_finalized) {
    // Best-effort cleanup if pipeline never reached EOS path.
    finalize_();
  }
  if (_enc_pkt) {
    _libs->avcodec().api.packet_free(&_enc_pkt);
  }
  if (_venc) {
    _libs->avcodec().api.free_context(&_venc);
  }
  if (_aenc) {
    _libs->avcodec().api.free_context(&_aenc);
  }
  if (_ofctx) {
    if (_ofctx->pb) {
      _libs->avformat().api.avio_closep(&_ofctx->pb);
    }
    _libs->avformat().api.free_context(_ofctx);
    _ofctx = nullptr;
  }
}

string
VideoFileEncoderStage::av_err_(int rc) const
{
  char buf[256];
  _libs->avutil().api.strerror(rc, buf, sizeof buf);
  return string(buf);
}

void
VideoFileEncoderStage::ensure_output_format_()
{
  if (_ofctx) {
    return;
  }
  AVFormatContext* ofctx = nullptr;
  const char* fmt_name = _format.empty() ? nullptr : _format.c_str();
  int rc = _libs->avformat().api.alloc_output_context2(
    &ofctx, nullptr, fmt_name, _output_url.c_str());
  if (rc < 0 || !ofctx) {
    session()->error(fmt(
        "VideoFileEncoderStage('{}'): alloc_output_context2 failed: "
        "{}", this->id(), av_err_(rc)));
  }
  _ofctx = ofctx;
}

void
VideoFileEncoderStage::init_video_encoder_(const VideoStreamParams& p)
{
  ensure_output_format_();

  const AVCodec* codec = _libs->avcodec().api.find_encoder_by_name(
    _video_codec.c_str());
  if (!codec) {
    session()->error(fmt(
        "VideoFileEncoderStage('{}'): video encoder '{}' not found",
        this->id(), _video_codec));
  }

  _venc = _libs->avcodec().api.alloc_context3(codec);
  if (!_venc) {
    session()->error(fmt(
        "VideoFileEncoderStage('{}'): alloc_context3 (video) failed",
        this->id()));
  }

  _venc->width     = p.width;
  _venc->height    = p.height;
  _venc->pix_fmt   = static_cast<AVPixelFormat>(p.pix_fmt);
  _venc->bit_rate  = _video_bitrate;
  _venc->gop_size  = _video_gop_size;

  // Encoder time_base: prefer 1/frame_rate; fall back to input
  // stream's time_base; final fallback 1/25.
  if (p.frame_rate.num > 0 && p.frame_rate.den > 0) {
    _venc->time_base = AVRational{p.frame_rate.den, p.frame_rate.num};
    _venc->framerate = p.frame_rate;
  } else if (p.time_base.num > 0 && p.time_base.den > 0) {
    _venc->time_base = p.time_base;
  } else {
    _venc->time_base = AVRational{1, 25};
  }

  if (_ofctx->oformat
      && (_ofctx->oformat->flags & AVFMT_GLOBALHEADER)) {
    _venc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  }

  AVDictionary* opts = nullptr;
  _libs->avutil().api.dict_set(&opts, "preset",
                              _video_preset.c_str(), 0);
  if (_video_has_crf) {
    string crf_s = to_string(_video_crf);
    _libs->avutil().api.dict_set(&opts, "crf", crf_s.c_str(), 0);
  }
  fill_dict_from_options_(_video_options, _libs->avutil(), &opts);

  int rc = _libs->avcodec().api.open2(_venc, codec, &opts);
  _libs->avutil().api.dict_free(&opts);
  if (rc < 0) {
    session()->error(fmt(
        "VideoFileEncoderStage('{}'): avcodec_open2 (video) failed: "
        "{}", this->id(), av_err_(rc)));
  }

  _vstream = _libs->avformat().api.new_stream(_ofctx, codec);
  if (!_vstream) {
    session()->error(fmt(
        "VideoFileEncoderStage('{}'): avformat_new_stream (video) "
        "failed", this->id()));
  }
  rc = _libs->avcodec().api.parameters_from_context(_vstream->codecpar,
                                                   _venc);
  if (rc < 0) {
    session()->error(fmt(
        "VideoFileEncoderStage('{}'): parameters_from_context "
        "(video) failed: {}", this->id(), av_err_(rc)));
  }
  _vstream->time_base = _venc->time_base;
}

void
VideoFileEncoderStage::init_audio_encoder_(const AudioStreamParams& p)
{
  ensure_output_format_();

  const AVCodec* codec = _libs->avcodec().api.find_encoder_by_name(
    _audio_codec.c_str());
  if (!codec) {
    session()->error(fmt(
        "VideoFileEncoderStage('{}'): audio encoder '{}' not found",
        this->id(), _audio_codec));
  }

  _aenc = _libs->avcodec().api.alloc_context3(codec);
  if (!_aenc) {
    session()->error(fmt(
        "VideoFileEncoderStage('{}'): alloc_context3 (audio) failed",
        this->id()));
  }

  _aenc->sample_rate = p.sample_rate;
  _aenc->sample_fmt  = static_cast<AVSampleFormat>(p.sample_fmt);
  // Bitwise copy is correct for AV_CHANNEL_ORDER_NATIVE (mask is a
  // value type). For CUSTOM the u.map pointer aliases the
  // upstream-owned buffer; safe as long as the producer outlives this
  // encoder, which is true within a single PipelineRuntime lifetime.
  _aenc->ch_layout   = p.ch_layout;
  _aenc->bit_rate    = _audio_bitrate;
  _aenc->time_base   = AVRational{1, p.sample_rate};

  if (_ofctx->oformat
      && (_ofctx->oformat->flags & AVFMT_GLOBALHEADER)) {
    _aenc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  }

  AVDictionary* opts = nullptr;
  fill_dict_from_options_(_audio_options, _libs->avutil(), &opts);

  int rc = _libs->avcodec().api.open2(_aenc, codec, &opts);
  _libs->avutil().api.dict_free(&opts);
  if (rc < 0) {
    session()->error(fmt(
        "VideoFileEncoderStage('{}'): avcodec_open2 (audio) failed: "
        "{}", this->id(), av_err_(rc)));
  }

  _astream = _libs->avformat().api.new_stream(_ofctx, codec);
  if (!_astream) {
    session()->error(fmt(
        "VideoFileEncoderStage('{}'): avformat_new_stream (audio) "
        "failed", this->id()));
  }
  rc = _libs->avcodec().api.parameters_from_context(_astream->codecpar,
                                                   _aenc);
  if (rc < 0) {
    session()->error(fmt(
        "VideoFileEncoderStage('{}'): parameters_from_context "
        "(audio) failed: {}", this->id(), av_err_(rc)));
  }
  _astream->time_base = _aenc->time_base;
}

bool
VideoFileEncoderStage::ready_to_write_header_() const noexcept
{
  bool v_ok = (!_enable_video) || _video_initialized;
  bool a_ok = (!_enable_audio) || _audio_initialized;
  return v_ok && a_ok;
}

void
VideoFileEncoderStage::open_output_and_write_header_()
{
  // For non-NOFILE muxers we need to open the IO context.
  if (_ofctx->oformat
      && !(_ofctx->oformat->flags & AVFMT_NOFILE))
  {
    int rc = _libs->avformat().api.avio_open(&_ofctx->pb,
                                             _output_url.c_str(),
                                             AVIO_FLAG_WRITE);
    if (rc < 0) {
      session()->error(fmt(
          "VideoFileEncoderStage('{}'): avio_open('{}') failed: {}",
          this->id(), _output_url, av_err_(rc)));
    }
  }

  AVDictionary* mux_opts = nullptr;
  fill_dict_from_options_(_muxer_options, _libs->avutil(), &mux_opts);
  int rc = _libs->avformat().api.write_header(_ofctx, &mux_opts);
  _libs->avutil().api.dict_free(&mux_opts);
  if (rc < 0) {
    session()->error(fmt(
        "VideoFileEncoderStage('{}'): avformat_write_header failed: "
        "{}", this->id(), av_err_(rc)));
  }
}

void
VideoFileEncoderStage::drain_encoder_(AVCodecContext* enc, AVStream* st)
{
  while (true) {
    int rc = _libs->avcodec().api.receive_packet(enc, _enc_pkt);
    if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) {
      break;
    }
    if (rc < 0) {
      session()->warn(fmt(
        "encoder('{}'): receive_packet: {}",
        this->id(), av_err_(rc)));
      break;
    }
    _enc_pkt->stream_index = st->index;
    _libs->avcodec().api.packet_rescale_ts(_enc_pkt,
                                          enc->time_base,
                                          st->time_base);
    rc = _libs->avformat().api.interleaved_write_frame(_ofctx,
                                                      _enc_pkt);
    if (rc < 0) {
      session()->warn(fmt(
        "encoder('{}'): interleaved_write_frame: {}",
        this->id(), av_err_(rc)));
    }
    // interleaved_write_frame consumes the packet (unrefs).
  }
}

void
VideoFileEncoderStage::encode_and_mux_(unsigned port, AVFrame* frame)
{
  AVCodecContext* enc = nullptr;
  AVStream*       st  = nullptr;
  if (static_cast<int>(port) == _video_port) {
    enc = _venc;
    st  = _vstream;
  } else if (static_cast<int>(port) == _audio_port) {
    enc = _aenc;
    st  = _astream;
  }
  if (!enc || !st) {
    return;
  }
  int rc = _libs->avcodec().api.send_frame(enc, frame);
  if (rc < 0) {
    session()->warn(fmt(
      "encoder('{}'): send_frame port {}: {}",
      this->id(), port, av_err_(rc)));
    return;
  }
  drain_encoder_(enc, st);
}

void
VideoFileEncoderStage::finalize_()
{
  if (_finalized) {
    return;
  }
  if (_header_written) {
    if (_venc) {
      _libs->avcodec().api.send_frame(_venc, nullptr);
      drain_encoder_(_venc, _vstream);
    }
    if (_aenc) {
      _libs->avcodec().api.send_frame(_aenc, nullptr);
      drain_encoder_(_aenc, _astream);
    }
    int rc = _libs->avformat().api.write_trailer(_ofctx);
    if (rc < 0) {
      session()->warn(fmt(
        "encoder('{}'): write_trailer: {}",
        this->id(), av_err_(rc)));
    }
  }
  if (_ofctx && _ofctx->pb
      && _ofctx->oformat
      && !(_ofctx->oformat->flags & AVFMT_NOFILE))
  {
    _libs->avformat().api.avio_closep(&_ofctx->pb);
  }
  _finalized = true;
}

Job
VideoFileEncoderStage::initialize(RuntimeContext& /*ctx*/)
{
  // Output context, encoder contexts, and the file IO context all
  // depend on stream parameters that arrive as the first token on
  // each input port. Setup is therefore deferred to process().
  co_return;
}

Job
VideoFileEncoderStage::drain(RuntimeContext& /*ctx*/)
{
  // Flush each encoder, write the muxer trailer, close the IO
  // context. Idempotent via _finalized.
  finalize_();
  co_return;
}

Job
VideoFileEncoderStage::process(RuntimeContext& ctx)
{
  // Are we entirely done?
  bool v_done = (_video_port < 0) || _video_eos;
  bool a_done = (_audio_port < 0) || _audio_eos;
  if (v_done && a_done) {
    ctx.signal_done();
    co_return;
  }

  // Pick a non-EOS port via round-robin.
  int target = _next_port;
  for (int tries = 0; tries < 2; ++tries) {
    if (target == _video_port && !_video_eos && _video_port >= 0) {
      break;
    }
    if (target == _audio_port && !_audio_eos && _audio_port >= 0) {
      break;
    }
    target = (target + 1) % static_cast<int>(this->num_iports());
  }
  unsigned port = static_cast<unsigned>(target);

  auto t = co_await ctx.read(port);
  if (!t) {
    if (static_cast<int>(port) == _video_port) {
      _video_eos = true;
    } else if (static_cast<int>(port) == _audio_port) {
      _audio_eos = true;
    }
    if (this->num_iports() > 1) {
      _next_port =
        (target + 1) % static_cast<int>(this->num_iports());
    }
    co_return;
  }

  bool& init = (static_cast<int>(port) == _video_port)
                 ? _video_initialized : _audio_initialized;
  if (!init) {
    try {
      if (static_cast<int>(port) == _video_port) {
        const auto* p = dynamic_cast<const VideoStreamParamsPayload*>(
            t.get());
        if (!p) {
          session()->error(fmt(
              "VideoFileEncoderStage('{}'): expected "
              "VideoStreamParams header on video port",
              this->id()));
        }
        init_video_encoder_(*p);
      } else {
        const auto* p = dynamic_cast<const AudioStreamParamsPayload*>(
            t.get());
        if (!p) {
          session()->error(fmt(
              "VideoFileEncoderStage('{}'): expected "
              "AudioStreamParams header on audio port",
              this->id()));
        }
        init_audio_encoder_(*p);
      }
      init = true;
      if (ready_to_write_header_() && !_header_written) {
        open_output_and_write_header_();
        _header_written = true;
      }
    } catch (const exception& e) {
      session()->warn(fmt(
        "encoder('{}'): init port {}: {}; stopping",
        this->id(), port, e.what()));
      ctx.signal_done();
      co_return;
    }
  } else {
    if (!_header_written) {
      // Producer-contract violation: frame before paired header.
      session()->warn(fmt(
        "encoder('{}'): port {} produced a frame before all paired "
        "headers arrived; dropping",
        this->id(), port));
    } else {
      const auto* fp = dynamic_cast<const FrameRefPayload*>(t.get());
      if (!fp || !fp->ref) {
        session()->warn(fmt(
          "encoder('{}'): non-FrameRef token on port {}; dropping",
          this->id(), port));
      } else {
        encode_and_mux_(port, fp->ref.get());
      }
    }
  }

  if (this->num_iports() > 1) {
    _next_port = (target + 1) % static_cast<int>(this->num_iports());
  }
}

VPIPE_REGISTER_STAGE(VideoFileEncoderStage)
VPIPE_REGISTER_SPEC(VideoFileEncoderStage, kSpec)

}

#include "stages/audio-video/audio-to-pcm-stage.h"

#include "common/beat-payload-intf.h"
#include "common/encoded-segment.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

extern "C" {
#include <libavutil/error.h>
#include <libavutil/opt.h>
}

using namespace std;

namespace vpipe {

namespace {

const char*
sample_fmt_name_(const FFmpegLibraries* libs, enum AVSampleFormat f)
{
  if (!libs) { return "?"; }
  const char* s = libs->avutil().api.get_sample_fmt_name(f);
  return s ? s : "?";
}

void
attach_sideband_(TensorBeat&    tb,
                 std::uint64_t  ts_us,
                 int            sample_rate,
                 std::uint64_t  duration_us)
{
  FlexData o = FlexData::make_object();
  o.as_object().insert_or_assign("timestamp_us",
                                 FlexData::make_uint(ts_us));
  o.as_object().insert_or_assign("sample_rate",
                                 FlexData::make_int(sample_rate));
  o.as_object().insert_or_assign("duration_us",
                                 FlexData::make_uint(duration_us));
  tb.sideband = std::move(o);
}

}

AudioToPcmStage::AudioToPcmStage(const SessionContextIntf* s,
                                 string                    id,
                                 vector<InEdge>            iports,
                                 FlexData                  config)
  : TypedStage<AudioToPcmStage>(s, std::move(id), std::move(iports),
                                std::move(config))
{
  // Declare our single output port BEFORE anything else can read
  // num_oports(). Without this call num_oports() defaults to 0, the
  // pipeline runtime allocates an empty `_out_bufs`, and the first
  // `ctx.write(0, ...)` reads past `_out_bufs.data()` -- crashes
  // with a null deref inside the WriteAwaiter setup (the exact
  // failure was `emit_chunk_(.resume) + 1036`, loading
  // `_out_bufs[0]->_buf` from an empty vector). Every other emitting
  // stage in the tree does this in its ctor; sink stages
  // (visual-qa, audio-transcribe) skip it because they have 0 oports.
  allocate_oports(spec().oports.size());

  // Validation is deferred to launch (see Stage::fail_config): the
  // stage must construct for any config. Each early return after a
  // fail_config just stops further setup; the runtime skips a stage
  // whose config_error() is non-empty. Attribute defaults live in
  // kSpec.attrs; attr_* resolves the configured value else that default.
  {
    int v = static_cast<int>(attr_int("output_sample_rate"));
    if (v < 1000 || v > 384000) {
      fail_config(fmt(
          "AudioToPcmStage('{}'): output_sample_rate {} outside "
          "[1000, 384000]", this->id(), v));
      return;
    }
    _output_sample_rate = v;
  }
  {
    double v = attr_real("chunk_duration_s");
    if (v <= 0.0 || v > 600.0) {
      fail_config(fmt(
          "AudioToPcmStage('{}'): chunk_duration_s {} outside "
          "(0, 600]", this->id(), v));
      return;
    }
    _chunk_duration_s = v;
  }
  {
    double v = attr_real("max_chunk_duration_s");
    if (v < _chunk_duration_s) {
      fail_config(fmt(
          "AudioToPcmStage('{}'): max_chunk_duration_s {} < "
          "chunk_duration_s {}",
          this->id(), v, _chunk_duration_s));
      return;
    }
    _max_chunk_duration_s = v;
  }
  {
    int v = static_cast<int>(attr_int("oport_capacity"));
    if (v < 1) {
      fail_config(fmt(
          "AudioToPcmStage('{}'): oport_capacity must be >= 1",
          this->id()));
      return;
    }
    _oport_capacity = static_cast<unsigned>(v);
  }
  {
    int e = static_cast<int>(attr_int("emit_log_every"));
    if (e < 1) {
      fail_config(fmt(
          "AudioToPcmStage('{}'): emit_log_every must be >= 1 (got {})",
          this->id(), e));
      return;
    }
    _emit_log_every = static_cast<unsigned>(e);
  }
  _flush_on_eos = attr_bool("flush_on_eos");

  _libs = session()->ffmpeg_libraries();
  if (!_libs) {
    fail_config(fmt(
        "AudioToPcmStage('{}'): ffmpeg libraries not available on "
        "this session", this->id()));
    return;
  }
  _pkt   = _libs->avcodec().api.packet_alloc();
  _frame = _libs->avutil().api.frame_alloc();
  if (!_pkt || !_frame) {
    fail_config(fmt(
        "AudioToPcmStage('{}'): packet/frame alloc failed",
        this->id()));
    return;
  }
  // Pre-reserve enough room for ~max chunk to avoid mid-stream
  // reallocations. mono float32 at _output_sample_rate.
  _chunk_buf.reserve(
      static_cast<std::size_t>(_output_sample_rate)
      * static_cast<std::size_t>(_max_chunk_duration_s + 1.0));
  session()->info(fmt(
      "AudioToPcmStage('{}'): output_sample_rate={} Hz, "
      "chunk_duration_s={}, max_chunk_duration_s={}, "
      "emit_log_every={}",
      this->id(), _output_sample_rate, _chunk_duration_s,
      _max_chunk_duration_s, _emit_log_every));
}

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "output_sample_rate", .type = ConfigType::Int,
   .doc = "resampler target Hz, [1000,384000]", .def_int = 16000},
  {.key = "chunk_duration_s", .type = ConfigType::Real,
   .doc = "min emitted chunk seconds, (0,600]", .def_real = 10.0},
  {.key = "max_chunk_duration_s", .type = ConfigType::Real,
   .doc = "hard-cap chunk seconds; >= chunk_duration_s",
   .def_real = 30.0},
  {.key = "oport_capacity", .type = ConfigType::Int,
   .doc = "oport buffer depth in chunks, >= 1", .def_int = 4},
  {.key = "flush_on_eos", .type = ConfigType::Bool,
   .doc = "emit partial chunk when source closes", .def_bool = true},
  {.key = "emit_log_every", .type = ConfigType::Int,
   .doc = "log one INFO line every Nth emitted chunk (>= 1)",
   .def_int = 1},
};
const PortSpec kIports[] = {
  {.name = "audio", .doc = "EncodedSegment with kind=Audio (e.g. AAC "
                           "from rtsp-capture or raw PCM)",
   .type = &typeid(EncodedSegmentPayload),
   .tags = "audio-encoder-segments", .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "pcm", .doc = "mono F32 PCM TensorBeat [N] at "
                         "output_sample_rate; sideband ts/sr/duration",
   .type = &typeid(TensorBeatPayload),
   .tags = "pcm-samples", .clock_group = 1},
};
const StageSpec kSpec = {
  .type_name = "audio-to-pcm",
  .doc       = "Decodes audio EncodedSegments and resamples to mono F32 "
               "PCM at output_sample_rate, emitting fixed-duration "
               "chunks. Crosses packet-rate -> chunk-rate clocks.",
  .display_name = "Audio → PCM",
  .category  = StageCategory::Audio,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
AudioToPcmStage::spec() const noexcept
{
  return kSpec;
}

AudioToPcmStage::~AudioToPcmStage()
{
  teardown_resampler_();
  teardown_decoder_();
  if (_libs) {
    if (_frame) { _libs->avutil().api.frame_free(&_frame); }
    if (_pkt)   { _libs->avcodec().api.packet_free(&_pkt); }
  }
}

void
AudioToPcmStage::teardown_decoder_()
{
  if (_libs && _dctx) {
    _libs->avcodec().api.free_context(&_dctx);
    _dctx = nullptr;
  }
  _last_codec_id = 0;
}

void
AudioToPcmStage::teardown_resampler_()
{
  if (_libs && _swr) {
    _libs->swresample().api.free(&_swr);
    _swr = nullptr;
  }
  _last_in_sample_rate = 0;
  _last_in_channels    = 0;
  _last_in_sample_fmt  = AV_SAMPLE_FMT_NONE;
}

bool
AudioToPcmStage::ensure_decoder_(const EncodedSegment& seg)
{
  if (_dctx && _last_codec_id == seg.codec_id) {
    return true;
  }
  if (_dctx) {
    // Codec switched mid-stream (rare). Tear down and rebuild.
    session()->warn(fmt(
        "AudioToPcmStage('{}'): codec_id switch {}->{}; reopening "
        "decoder", this->id(),
        static_cast<int>(_last_codec_id),
        static_cast<int>(seg.codec_id)));
    teardown_decoder_();
    teardown_resampler_();
  }
  AVCodecID id_ = static_cast<AVCodecID>(seg.codec_id);
  const AVCodec* codec = _libs->avcodec().api.find_decoder(id_);
  if (!codec) {
    session()->error(fmt(
        "AudioToPcmStage('{}'): no decoder for audio codec_id={}",
        this->id(), static_cast<int>(id_)));
    return false;
  }
  AVCodecContext* cctx = _libs->avcodec().api.alloc_context3(codec);
  if (!cctx) {
    session()->error(fmt(
        "AudioToPcmStage('{}'): avcodec_alloc_context3 failed",
        this->id()));
    return false;
  }
  cctx->sample_rate = static_cast<int>(seg.sample_rate);
  // Channel layout: rtsp-capture's EncodedSegment only records the
  // channel count, not a full layout. For AAC the layout is encoded
  // in extradata (AudioSpecificConfig); for raw PCM-like codecs we
  // fall back to mono/stereo defaults by channel count.
  if (seg.channels > 0) {
    if (seg.channels == 1) {
      cctx->ch_layout = AV_CHANNEL_LAYOUT_MONO;
    } else if (seg.channels == 2) {
      cctx->ch_layout = AV_CHANNEL_LAYOUT_STEREO;
    } else {
      AVChannelLayout cl = AV_CHANNEL_LAYOUT_MONO;
      cl.nb_channels = static_cast<int>(seg.channels);
      cctx->ch_layout = cl;
    }
  }
  if (!seg.extradata.empty()) {
    cctx->extradata_size = static_cast<int>(seg.extradata.size());
    cctx->extradata = static_cast<uint8_t*>(
        _libs->avutil().api.malloc(
            cctx->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE));
    if (!cctx->extradata) {
      session()->error(fmt(
          "AudioToPcmStage('{}'): extradata alloc failed",
          this->id()));
      _libs->avcodec().api.free_context(&cctx);
      return false;
    }
    std::memcpy(cctx->extradata, seg.extradata.data(),
                cctx->extradata_size);
    std::memset(cctx->extradata + cctx->extradata_size, 0,
                AV_INPUT_BUFFER_PADDING_SIZE);
  }
  int rc = _libs->avcodec().api.open2(cctx, codec, nullptr);
  if (rc < 0) {
    char buf[256] = {0};
    _libs->avutil().api.strerror(rc, buf, sizeof(buf));
    session()->error(fmt(
        "AudioToPcmStage('{}'): avcodec_open2 failed: {} ({})",
        this->id(), buf, rc));
    _libs->avcodec().api.free_context(&cctx);
    return false;
  }
  _dctx = cctx;
  _last_codec_id = seg.codec_id;
  session()->info(fmt(
      "AudioToPcmStage('{}'): opened decoder codec_id={} "
      "in_sr={} in_ch={} (resampler builds on first frame)",
      this->id(), static_cast<int>(id_),
      cctx->sample_rate, cctx->ch_layout.nb_channels));
  return true;
}

bool
AudioToPcmStage::ensure_resampler_(int             in_sample_rate,
                                   int             in_channels,
                                   enum AVSampleFormat in_fmt)
{
  if (_swr
      && in_sample_rate == _last_in_sample_rate
      && in_channels    == _last_in_channels
      && in_fmt         == _last_in_sample_fmt) {
    return true;
  }
  if (_swr) {
    session()->info(fmt(
        "AudioToPcmStage('{}'): rebuilding resampler "
        "(sr {}->{} ch {}->{} fmt {}->{})",
        this->id(), _last_in_sample_rate, in_sample_rate,
        _last_in_channels, in_channels,
        sample_fmt_name_(_libs, _last_in_sample_fmt),
        sample_fmt_name_(_libs, in_fmt)));
    teardown_resampler_();
  }
  AVChannelLayout in_layout = AV_CHANNEL_LAYOUT_MONO;
  if (in_channels == 1) {
    in_layout = AV_CHANNEL_LAYOUT_MONO;
  } else if (in_channels == 2) {
    in_layout = AV_CHANNEL_LAYOUT_STEREO;
  } else {
    in_layout.nb_channels = in_channels;
  }
  AVChannelLayout out_layout = AV_CHANNEL_LAYOUT_MONO;
  SwrContext* sw = nullptr;
  int rc = _libs->swresample().api.alloc_set_opts2(
      &sw,
      &out_layout, AV_SAMPLE_FMT_FLT, _output_sample_rate,
      &in_layout,  in_fmt,            in_sample_rate,
      0, nullptr);
  if (rc < 0 || !sw) {
    session()->error(fmt(
        "AudioToPcmStage('{}'): swr_alloc_set_opts2 failed (rc={})",
        this->id(), rc));
    return false;
  }
  rc = _libs->swresample().api.init(sw);
  if (rc < 0) {
    session()->error(fmt(
        "AudioToPcmStage('{}'): swr_init failed (rc={})",
        this->id(), rc));
    _libs->swresample().api.free(&sw);
    return false;
  }
  _swr                 = sw;
  _last_in_sample_rate = in_sample_rate;
  _last_in_channels    = in_channels;
  _last_in_sample_fmt  = in_fmt;
  return true;
}

bool
AudioToPcmStage::decode_one_(const EncodedSegment& seg)
{
  if (!ensure_decoder_(seg)) { return false; }
  // Feed the AAC frame to the decoder. Mirror video-to-rgb's
  // pattern: packet_unref BEFORE each use to drop any side data /
  // ref-counted buffer state from the prior packet, then assign raw
  // pointer + size (unowned-data mode). Do NOT null out fields
  // after send_packet -- packet_unref on the next iteration clears
  // them correctly, and clearing them manually fights any deferred
  // bookkeeping FFmpeg may have done.
  _libs->avcodec().api.packet_unref(_pkt);
  _pkt->data = const_cast<uint8_t*>(seg.data.data());
  _pkt->size = static_cast<int>(seg.data.size());
  int rc = _libs->avcodec().api.send_packet(_dctx, _pkt);
  if (rc < 0 && rc != AVERROR(EAGAIN)) {
    char buf[256] = {0};
    _libs->avutil().api.strerror(rc, buf, sizeof(buf));
    if (!_decoder_warned) {
      session()->warn(fmt(
          "AudioToPcmStage('{}'): avcodec_send_packet: {} (rc={})",
          this->id(), buf, rc));
      _decoder_warned = true;
    }
    return false;
  }
  bool any_decoded = false;
  while (true) {
    rc = _libs->avcodec().api.receive_frame(_dctx, _frame);
    if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) {
      break;
    }
    if (rc < 0) {
      char buf[256] = {0};
      _libs->avutil().api.strerror(rc, buf, sizeof(buf));
      session()->warn(fmt(
          "AudioToPcmStage('{}'): avcodec_receive_frame: {} (rc={})",
          this->id(), buf, rc));
      break;
    }
    any_decoded = true;
    // Build / refresh the resampler if the frame's format changed.
    if (!ensure_resampler_(
            _frame->sample_rate,
            _frame->ch_layout.nb_channels,
            static_cast<enum AVSampleFormat>(_frame->format))) {
      _libs->avutil().api.frame_unref(_frame);
      continue;
    }
    // Resample. The swr's out_samples upper bound is
    //   delay + in_samples * out_sr / in_sr + 32 (slack).
    int64_t delay = _libs->swresample().api.get_delay(
        _swr, _frame->sample_rate);
    int64_t max_out =
        delay + (static_cast<int64_t>(_frame->nb_samples)
                 * _output_sample_rate)
                / std::max(1, _frame->sample_rate) + 32;
    std::size_t prev = _chunk_buf.size();
    _chunk_buf.resize(prev + static_cast<std::size_t>(max_out));
    float* dst = _chunk_buf.data() + prev;
    int n_out = _libs->swresample().api.convert(
        _swr, reinterpret_cast<uint8_t**>(&dst),
        static_cast<int>(max_out),
        const_cast<const uint8_t**>(_frame->extended_data),
        _frame->nb_samples);
    if (n_out < 0) {
      char buf[256] = {0};
      _libs->avutil().api.strerror(n_out, buf, sizeof(buf));
      session()->warn(fmt(
          "AudioToPcmStage('{}'): swr_convert failed: {} (rc={})",
          this->id(), buf, n_out));
      _chunk_buf.resize(prev);
      _libs->avutil().api.frame_unref(_frame);
      break;
    }
    _chunk_buf.resize(prev + static_cast<std::size_t>(n_out));
    _libs->avutil().api.frame_unref(_frame);
  }
  if (any_decoded) {
    ++_packets_decoded;
    // First packet of a fresh chunk: record its timestamp.
    if (!_chunk_has_ts) {
      _chunk_first_ts_us = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(
              seg.start_utc.time_since_epoch()).count());
      _chunk_has_ts = true;
    }
  }
  return any_decoded;
}

Job
AudioToPcmStage::emit_chunk_(RuntimeContext& ctx)
{
  if (_chunk_buf.empty()) { co_return; }
  TensorBeat tb;
  tb.dtype = TensorBeat::DType::F32;
  tb.shape = { static_cast<int64_t>(_chunk_buf.size()) };
  tb.strides.clear();  // row-major contiguous 1-D
  const std::size_t nbytes = _chunk_buf.size() * sizeof(float);
  tb.data.resize(nbytes);
  std::memcpy(tb.data.data(), _chunk_buf.data(), nbytes);
  const std::uint64_t duration_us = static_cast<std::uint64_t>(
      _chunk_buf.size() * 1'000'000ULL / _output_sample_rate);
  attach_sideband_(tb, _chunk_first_ts_us,
                   _output_sample_rate, duration_us);
  if (_emit_log_every == 1
      || (_chunks_emitted % _emit_log_every) == 0) {
    session()->log_verbose(fmt(
        "AudioToPcmStage('{}'): emit chunk #{} samples={} dur={:.2f}s "
        "ts_us={}",
        this->id(), _chunks_emitted, _chunk_buf.size(),
        _chunk_buf.size() / static_cast<double>(_output_sample_rate),
        _chunk_first_ts_us));
  }
  _chunk_buf.clear();
  _chunk_has_ts = false;
  _chunk_first_ts_us = 0;
  ++_chunks_emitted;
  // Build the payload as a separate local BEFORE the suspend.
  // Inlining the make_payload<...>(std::move(tb)) expression inside
  // the co_await argument leaves the temporary's lifetime tangled
  // with the suspend's frame layout; on the resume path the move-
  // from TensorBeat's destructor sequencing was crashing in
  // .resume + 1036. Hoisting the payload to its own local makes
  // the lifetime line up with the rest of the function locals and
  // matches the pattern in stages that work (visual-qa-stage,
  // realtime-vqa-stage all build the payload, then co_await write
  // on a clearly-bound name).
  auto payload = make_payload<TensorBeatPayload>(std::move(tb));
  co_await ctx.write(0, std::move(payload));
  co_return;
}

Job
AudioToPcmStage::process(RuntimeContext& ctx)
{
  auto p = co_await ctx.read(0);
  if (!p) {
    if (_flush_on_eos && !_chunk_buf.empty()) {
      co_await emit_chunk_(ctx);
    }
    ctx.signal_done();
    co_return;
  }
  const auto* esp = dynamic_cast<const EncodedSegmentPayload*>(p.get());
  if (!esp) {
    session()->warn(fmt(
        "AudioToPcmStage('{}'): expected EncodedSegmentPayload on "
        "iport 0, got {}; dropping beat",
        this->id(), p->describe()));
    co_return;
  }
  if (esp->kind != EncodedSegment::Kind::Audio) {
    session()->warn(fmt(
        "AudioToPcmStage('{}'): expected audio segment, got video; "
        "dropping beat (wire to rtsp-capture's oport 1, not 0)",
        this->id()));
    co_return;
  }
  decode_one_(*esp);
  const double cur_s =
      static_cast<double>(_chunk_buf.size()) / _output_sample_rate;
  if (cur_s >= _chunk_duration_s) {
    co_await emit_chunk_(ctx);
  } else if (cur_s >= _max_chunk_duration_s) {
    // Safety cap. Should only trip when chunk_duration_s and
    // max_chunk_duration_s are configured close together.
    co_await emit_chunk_(ctx);
  }
  co_return;
}

Job
AudioToPcmStage::drain(RuntimeContext& ctx)
{
  if (_flush_on_eos && !_chunk_buf.empty()) {
    co_await emit_chunk_(ctx);
  }
  co_return;
}

VPIPE_REGISTER_STAGE(AudioToPcmStage)
VPIPE_REGISTER_SPEC(AudioToPcmStage, kSpec)

}

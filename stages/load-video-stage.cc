#include "common/encoded-segment.h"
#include "stages/load-video-stage.h"
#include "common/beat-payload-intf.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include <algorithm>
#include <chrono>
#include <cstring>
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

// Reads an object<string,string> from FlexData and forwards each pair
// into the AVDictionary via the avutil api.
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

LoadVideoStage::LoadVideoStage
  (const SessionContextIntf* s,
   string                    id,
   vector<InEdge>            iports,
   FlexData                  config)
  : TypedStage<LoadVideoStage>(s, std::move(id),
                                      std::move(iports),
                                      std::move(config))
  , _libs(s->services()->ffmpeg_libraries())
{
  // Stage's _config now holds the moved-in FlexData. Read everything
  // we need up-front; throw via session()->error if `input_url` is
  // missing because there's no graceful default.
  // Attribute defaults live in kSpec.attrs; attr_* resolves the
  // configured value else that default.
  // ONE path or an ARRAY of them, the spelling load-image, load-text and
  // video-ref-encoder already use -- and the reason the composer's file
  // browser offers multi-select here at all (it reads the key's declared
  // type, not a flag of its own: anything but `string` picks several).
  //
  // Two or more are JOINED, in the order given, into one stream. That is
  // what a caller wants from a video source handed several files, and it
  // is the same join `format: "concat"` spells by hand -- so this key now
  // covers the common case and the list file stays for the uncommon one
  // (per-clip `inpoint`/`outpoint` trims, which a bare path cannot carry).
  {
    const FlexData& cfg = this->config();
    if (cfg.is_object()) {
      auto root = cfg.as_object();
      if (root.contains("input_url")) {
        FlexData u = root.at("input_url");
        if (u.is_string()) {
          _inputs.emplace_back(string(u.as_string("")));
        } else if (u.is_array()) {
          for (FlexData e : u.as_array()) {
            if (e.is_string()) {
              _inputs.emplace_back(string(e.as_string("")));
            } else {
              fail_config(fmt(
                  "LoadVideoStage('{}'): config.input_url array entries "
                  "must be path strings", this->id()));
            }
          }
        } else {
          fail_config(fmt(
              "LoadVideoStage('{}'): config.input_url must be a string or "
              "an array of strings", this->id()));
        }
      }
    }
    if (_inputs.empty()) {
      fail_config(fmt(
          "LoadVideoStage('{}'): config.input_url is required (a path/URL "
          "or an array of them)", this->id()));
    }
    // Confine each to the session file sandbox exactly as the single
    // path did (attr_path's core); network URLs pass through unchanged.
    for (auto& u : _inputs) {
      u = confine_local_(u, /*for_write=*/false);
    }
    for (const auto& u : _inputs) {
      if (u.empty()) {
        fail_config(fmt(
            "LoadVideoStage('{}'): config.input_url contains an empty "
            "entry", this->id()));
      }
    }
    // The single-input spelling stays byte-for-byte what it was: one
    // entry opens its own URL and never reaches the concat demuxer.
    _input_url = _inputs.empty() ? string() : _inputs.front();
  }
  _format             = attr_str("format");
  _enable_video       = attr_bool("enable_video");
  _enable_audio       = attr_bool("enable_audio");
  _video_stream_index = static_cast<int>(attr_int("video_stream_index"));
  _audio_stream_index = static_cast<int>(attr_int("audio_stream_index"));
  _read_timeout_ms    = static_cast<int>(attr_int("read_timeout_ms"));
  _start_s            = attr_real("start_s");
  _duration_s         = attr_real("duration_s");
  _open_options       = attr("options");

  if (_start_s < 0.0) {
    fail_config(fmt("LoadVideoStage('{}'): start_s must be >= 0, got {}",
                    this->id(), _start_s));
  }
  if (_duration_s < 0.0) {
    fail_config(fmt("LoadVideoStage('{}'): duration_s must be >= 0 (0 = to "
                    "the end of the file), got {}", this->id(), _duration_s));
  }
  _start_us = static_cast<std::int64_t>(_start_s * 1e6);
  _has_stop = _duration_s > 0.0;
  _stop_us  = _has_stop
                  ? _start_us + static_cast<std::int64_t>(_duration_s * 1e6)
                  : 0;

  // Validation is deferred to launch (see Stage::fail_config).
  if (_input_url.empty()) {
    fail_config(fmt(
      "LoadVideoStage('{}'): config.input_url is required",
      this->id()));
  }
  if (!_enable_video && !_enable_audio) {
    fail_config(fmt(
      "LoadVideoStage('{}'): at least one of enable_video / "
      "enable_audio must be true",
      this->id()));
  }

  // Assign port indices: video first, then audio.
  unsigned next = 0;
  if (_enable_video) {
    _video_port = static_cast<int>(next++);
  }
  if (_enable_audio) {
    _audio_port = static_cast<int>(next++);
  }
  allocate_oports(next);
}

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "input_url", .type = ConfigType::Any, .required = true,
   .doc = "file path or network URL (rtsp/http/...), or an ARRAY of them. "
          "An array is JOINED into one stream in the order given -- the "
          "clips play back to back, both oports run over the join, and "
          "`start_s`/`duration_s` address the joined timeline. Pick "
          "several files at once in the composer's file browser. Only "
          "local files can be joined this way; a network URL belongs on "
          "its own",
   .is_path = true, .path_filter = "video"},
  {.key = "format", .type = ConfigType::String,
   .doc = "forced demuxer by name, \"\" = autodetect. Joining clips does "
          "NOT need this any more -- give `input_url` an array instead. "
          "`concat` remains for what an array cannot say: a LIST file of "
          "`file` / `inpoint` / `outpoint` lines, which trims each clip as "
          "it joins it (it then needs `options: {\"safe\": \"0\"}` for "
          "absolute paths). Setting it alongside an array is an error, not "
          "a silent winner. A name no demuxer answers to is an error too, "
          "not a silent fall back to probing",
   .def_str = ""},
  {.key = "enable_video", .type = ConfigType::Bool,
   .doc = "emit video oport", .def_bool = true},
  {.key = "enable_audio", .type = ConfigType::Bool,
   .doc = "emit audio oport", .def_bool = true},
  {.key = "video_stream_index", .type = ConfigType::Int,
   .doc = "stream to use; -1 = first video", .def_int = -1},
  {.key = "audio_stream_index", .type = ConfigType::Int,
   .doc = "stream to use; -1 = first audio", .def_int = -1},
  {.key = "start_s", .type = ConfigType::Real,
   .doc = "seek to this media time (seconds from the start of the file) "
          "before reading the first packet; 0 (default) starts at the "
          "beginning. KEYFRAME-accurate: the packets between the keyframe "
          "and start_s are emitted, because the frames after it are "
          "decoded from them",
   .def_real = 0.0},
  {.key = "duration_s", .type = ConfigType::Real,
   .doc = "stop this many seconds past start_s; 0 (default) reads to the "
          "end of the file. Each enabled stream stops when it passes the "
          "end, and the run ends when both have",
   .def_real = 0.0},
  {.key = "read_timeout_ms", .type = ConfigType::Int,
   .doc = "network read timeout ms; 0 = none", .def_int = 0},
  {.key = "options", .type = ConfigType::Object,
   .doc = "av_dict of string opts for input open"},
};
// Canonical oports (video first, audio second); either may be disabled
// via config, so the live count is assigned dynamically in the ctor.
// The payload is rtsp-capture's, so a file and a camera are the same
// input to everything downstream. Clock groups are authoritatively
// reported by oport_clock_group() (video 0, audio 1) -- unlike the
// camera's single capture clock, a file's two streams advance
// independently.
const PortSpec kOports[] = {
  {.name = "video", .doc = "ONE EncodedSegment per video packet, carrying "
                           "the stream's codec_id / width / height / "
                           "fps_num / fps_den / extradata. Wire it to "
                           "video-to-rgb, which owns the decode",
   .type = &typeid(EncodedSegmentPayload),
   .tags = "video-encoder-segments", .clock_group = 0},
  {.name = "audio", .doc = "ONE EncodedSegment per audio packet, carrying "
                           "the stream's codec_id / sample_rate / channels "
                           "/ extradata. Wire it to audio-to-pcm, which "
                           "owns the decode and the resample",
   .type = &typeid(EncodedSegmentPayload),
   .tags = "audio-encoder-segments", .clock_group = 1},
};
const StageSpec kSpec = {
  .type_name = "load-video",
  .doc       = "Source: demuxes a video file or network URL and emits its "
               "encoded video/audio packets on independent per-stream "
               "clocks -- the file-based sibling of rtsp-capture. Feeds "
               "video-to-rgb / audio-to-pcm, which own the decode. Given "
               "SEVERAL files it joins them into one stream, in order.",
  .display_name = "Load Video",
  .category  = StageCategory::Visual,
  .iports    = {},
  .oports    = kOports,
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
LoadVideoStage::spec() const noexcept
{
  return kSpec;
}

LoadVideoStage::~LoadVideoStage()
{
  if (_pkt) {
    _libs->avcodec().api.packet_free(&_pkt);
  }
  if (_fctx) {
    _libs->avformat().api.close_input(&_fctx);
  }
  // A preset `pb` sets AVFMT_FLAG_CUSTOM_IO, so close_input left the
  // AVIOContext alone -- free it here, and its CURRENT buffer first
  // (FFmpeg may have reallocated it, so this is not the pointer we
  // handed in).
  if (_avio) {
    _libs->avutil().api.freep(&_avio->buffer);
    _libs->avformat().api.avio_context_free(&_avio);
  }
}

int
LoadVideoStage::concat_read_(void* opaque, std::uint8_t* buf, int size)
{
  auto* self = static_cast<LoadVideoStage*>(opaque);
  if (self->_concat_pos >= self->_concat_list.size()) {
    return AVERROR_EOF;
  }
  const size_t n = std::min(static_cast<size_t>(size),
                            self->_concat_list.size()
                              - self->_concat_pos);
  std::memcpy(buf, self->_concat_list.data() + self->_concat_pos, n);
  self->_concat_pos += n;
  return static_cast<int>(n);
}

std::int64_t
LoadVideoStage::concat_seek_(void* opaque, std::int64_t offset, int whence)
{
  auto* self = static_cast<LoadVideoStage*>(opaque);
  const auto size = static_cast<std::int64_t>(self->_concat_list.size());
  if (whence & AVSEEK_SIZE) {
    return size;
  }
  std::int64_t base = 0;
  switch (whence & ~AVSEEK_FORCE) {
  case SEEK_SET: base = 0; break;
  case SEEK_CUR: base = static_cast<std::int64_t>(self->_concat_pos); break;
  case SEEK_END: base = size; break;
  default:       return AVERROR(EINVAL);
  }
  const std::int64_t np = base + offset;
  if (np < 0 || np > size) {
    return AVERROR(EINVAL);
  }
  self->_concat_pos = static_cast<size_t>(np);
  return np;
}

string
LoadVideoStage::av_err_(int rc) const
{
  char buf[256];
  _libs->avutil().api.strerror(rc, buf, sizeof buf);
  return string(buf);
}

int
LoadVideoStage::pick_stream_(int media_type,
                                    int requested) const noexcept
{
  if (!_fctx) {
    return -1;
  }
  for (unsigned i = 0; i < _fctx->nb_streams; ++i) {
    AVStream* s = _fctx->streams[i];
    if (s->codecpar->codec_type != media_type) {
      continue;
    }
    if (requested < 0) {
      return static_cast<int>(i);
    }
    if (static_cast<int>(i) == requested) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

void
LoadVideoStage::cache_stream_(int stream_idx, bool video)
{
  if (_fctx == nullptr || stream_idx < 0) { return; }
  AVStream* st = _fctx->streams[stream_idx];
  StreamMeta& m = video ? _vmeta : _ameta;
  m.codec_id  = (unsigned)st->codecpar->codec_id;
  m.time_base = st->time_base;
  if (st->codecpar->extradata != nullptr &&
      st->codecpar->extradata_size > 0) {
    m.extradata.assign(
        st->codecpar->extradata,
        st->codecpar->extradata + st->codecpar->extradata_size);
  }
  if (video) {
    m.width  = (unsigned)st->codecpar->width;
    m.height = (unsigned)st->codecpar->height;
    // THE CONTAINER'S COLOUR DESCRIPTION, carried so a decoder built
    // from `extradata` alone is not left with only the bitstream's. An
    // H.264 SPS may carry a VUI and when it does it wins, but a file
    // tagged only in the `colr` box has nothing in the bitstream at all
    // -- and reading that as untagged costs a 601-for-709 hue tilt that
    // nothing downstream reports. See EncodedSegment.
    m.color_range     = (int)st->codecpar->color_range;
    m.colorspace      = (int)st->codecpar->color_space;
    m.color_primaries = (int)st->codecpar->color_primaries;
    m.color_trc       = (int)st->codecpar->color_trc;
    // avg_frame_rate first, r_frame_rate as the fallback, 0/0 when the
    // container advertises neither -- the same three-way answer
    // rtsp-capture gives, so video-to-rgb's sideband means one thing
    // whichever source it is reading.
    AVRational fr = st->avg_frame_rate;
    if (fr.num <= 0 || fr.den <= 0) { fr = st->r_frame_rate; }
    if (fr.num > 0 && fr.den > 0) {
      m.fps_num = (unsigned)fr.num;
      m.fps_den = (unsigned)fr.den;
    }
  } else {
    m.sample_rate = (unsigned)st->codecpar->sample_rate;
    m.channels    = (unsigned)st->codecpar->ch_layout.nb_channels;
    // The encoder's priming, which the container knows, and the TRUE
    // length, which it does not -- an mp4's audio duration counts the
    // block codec's padding at both ends. A file vpipe wrote records
    // the real count under kAudioSamplesMetaKey; see EncodedSegment.
    m.skip_head = st->codecpar->initial_padding > 0
                      ? (std::int64_t)st->codecpar->initial_padding
                      : 0;
    m.total_samples = 0;
    {
      const auto& avu = _libs->avutil().api;
      for (AVDictionary* d : {st->metadata,
                              _fctx != nullptr ? _fctx->metadata : nullptr}) {
        if (d == nullptr) { continue; }
        AVDictionaryEntry* e =
            avu.dict_get(d, kAudioSamplesMetaKey, nullptr, 0);
        if (e != nullptr && e->value != nullptr) {
          const long long v = std::strtoll(e->value, nullptr, 10);
          if (v > 0) { m.total_samples = (std::int64_t)v; break; }
        }
      }
    }
  }
}

bool
LoadVideoStage::advance_to_next_input_()
{
  // The joined timeline continues where this part ended. Both streams
  // are considered: they do not end together, and rewinding the clock
  // to the shorter one would overlap the next part on that port.
  const std::int64_t end_us = std::max(_vmeta.last_us, _ameta.last_us);
  ++_seq_index;
  _libs->avformat().api.close_input(&_fctx);
  _fctx = nullptr;
  if (!open_input_()) {
    session()->warn(fmt(
        "LoadVideoStage('{}'): joining stopped at part {} of {}: '{}' "
        "could not be opened", this->id(), _seq_index + 1, _inputs.size(),
        _inputs[_seq_index]));
    return false;
  }
  _seq_base_us = end_us;
  // A fresh file restarts its own clock at zero; the per-stream cursors
  // have to restart with it or a packet without a pts would be dated
  // from the previous part.
  _vmeta.last_us = 0;
  _ameta.last_us = 0;
  session()->info(fmt(
      "LoadVideoStage('{}'): joined part {} of {} -- '{}' continues at "
      "{:.3f} s", this->id(), _seq_index + 1, _inputs.size(),
      _inputs[_seq_index], (double)_seq_base_us / 1e6));
  return true;
}

std::unique_ptr<EncodedSegmentPayload>
LoadVideoStage::segment_(bool video)
{
  StreamMeta& m = video ? _vmeta : _ameta;
  auto seg = std::make_unique<EncodedSegmentPayload>();
  seg->kind = video ? EncodedSegment::Kind::Video
                    : EncodedSegment::Kind::Audio;
  seg->path      = _input_url;
  seg->codec_id  = m.codec_id;
  seg->extradata = m.extradata;
  if (video) {
    seg->width   = m.width;
    seg->height  = m.height;
    seg->fps_num = m.fps_num;
    seg->fps_den = m.fps_den;
    seg->color_range     = m.color_range;
    seg->colorspace      = m.colorspace;
    seg->color_primaries = m.color_primaries;
    seg->color_trc       = m.color_trc;
  } else {
    seg->sample_rate = m.sample_rate;
    seg->channels    = m.channels;
    seg->skip_head     = m.skip_head;
    seg->total_samples = m.total_samples;
    seg->part_index    = (unsigned)_seq_index;
  }
  // The container's own answer to "can a decoder start here", which is
  // the only one a non-H.264 codec has -- nothing in an FFV1 packet
  // says it. Audio ignores it.
  seg->key_frame = (_pkt->flags & AV_PKT_FLAG_KEY) != 0;
  seg->data.assign(_pkt->data, _pkt->data + _pkt->size);

  const AVRational us = {1, 1000000};
  std::int64_t t = m.last_us;
  if (_pkt->pts != AV_NOPTS_VALUE) {
    t = _libs->avutil().api.rescale_q(_pkt->pts, m.time_base, us);
  }
  std::int64_t d = 0;
  if (_pkt->duration > 0) {
    d = _libs->avutil().api.rescale_q(_pkt->duration, m.time_base, us);
  }
  m.last_us = t + d;
  // `t` is this PART's own clock; the beat carries the joined one.
  const std::int64_t jt = t + _seq_base_us;
  seg->duration_us = d;
  seg->start_utc =
      std::chrono::system_clock::time_point(std::chrono::microseconds(jt));
  seg->end_utc =
      std::chrono::system_clock::time_point(
          std::chrono::microseconds(jt + d));
  return seg;
}

// EVERY JOINED INPUT MUST DECODE THE SAME WAY, and nothing downstream
// can cope if they do not: a decoder is opened once, from the FIRST
// segment's parameters, and packets from a segment that disagrees are
// rejected one by one. The symptom is silent and asymmetric -- the
// stream that changed simply stops after the first file, while the
// other one plays to the end -- so it reads as "the second clip has no
// sound" rather than as a mismatch. Measured on an ALAC part joined to
// an AAC one: 13.9 s of video against 10.2 s of audio and a WARN per
// packet.
//
// The concat demuxer will not tell us: it presents ONE stream set, the
// first file's. So each input is opened here and compared before any of
// them is read.
void
LoadVideoStage::check_inputs_agree_()
{
  if (_inputs.size() < 2) { return; }
  struct Sig { int vid = -1, aud = -1, rate = 0, ch = 0; };
  Sig first;
  for (std::size_t i = 0; i < _inputs.size(); ++i) {
    AVFormatContext* f = nullptr;
    if (_libs->avformat().api.open_input(&f, _inputs[i].c_str(),
                                         nullptr, nullptr) < 0) {
      continue;    // the join itself will fail on it, with a better message
    }
    if (_libs->avformat().api.find_stream_info(f, nullptr) < 0) {
      _libs->avformat().api.close_input(&f);
      continue;
    }
    Sig sig;
    for (unsigned k = 0; k < f->nb_streams; ++k) {
      const AVCodecParameters* cp = f->streams[k]->codecpar;
      if (cp->codec_type == AVMEDIA_TYPE_VIDEO && sig.vid < 0) {
        sig.vid = (int)cp->codec_id;
      } else if (cp->codec_type == AVMEDIA_TYPE_AUDIO && sig.aud < 0) {
        sig.aud  = (int)cp->codec_id;
        sig.rate = cp->sample_rate;
        sig.ch   = cp->ch_layout.nb_channels;
      }
    }
    _libs->avformat().api.close_input(&f);
    if (i == 0) { first = sig; continue; }
    auto name = [&](int id) {
      const AVCodec* c = id < 0 ? nullptr
          : _libs->avcodec().api.find_decoder((AVCodecID)id);
      return c && c->name ? c->name : "none";
    };
    if (sig.vid != first.vid || sig.aud != first.aud
        || sig.rate != first.rate || sig.ch != first.ch) {
      session()->error(fmt(
          "LoadVideoStage('{}'): the joined inputs do not agree on how "
          "they decode, so everything after the first would be dropped. "
          "'{}' is video {} / audio {} at {} Hz x{}, but '{}' is video "
          "{} / audio {} at {} Hz x{}. Re-encode them to one format "
          "before joining", this->id(),
          _inputs.front(), name(first.vid), name(first.aud), first.rate,
          first.ch, _inputs[i], name(sig.vid), name(sig.aud), sig.rate,
          sig.ch));
      return;
    }
  }
}

bool
LoadVideoStage::open_input_()
{
  AVDictionary* opts = nullptr;
  fill_dict_from_options_(_open_options, _libs->avutil(), &opts);
  if (_read_timeout_ms > 0) {
    string val = to_string(static_cast<long long>(_read_timeout_ms)
                            * 1000LL);
    _libs->avutil().api.dict_set(&opts, "stimeout", val.c_str(), 0);
  }

  AVFormatContext* fctx = nullptr;
  // A FORCED demuxer, when `format` names one. Autodetection covers every
  // ordinary file, and there is one family it cannot cover at all: a
  // demuxer whose input is a LIST rather than media. `format: "concat"`
  // over a text file of `file` / `outpoint` lines is how a graph joins
  // clips without a stage that re-encodes them one by one -- and probing
  // decides that same file is ANSI art (a 640x400 `tty` stream), because
  // that is what a text file looks like to a prober. So the key has to
  // reach the open, and an unknown name is a clean failure here rather
  // than a mystery stream later.
  const AVInputFormat* forced = nullptr;
  if (!_format.empty()) {
    forced = _libs->avformat().api.find_input_format(_format.c_str());
    if (forced == nullptr) {
      session()->error(fmt(
          "LoadVideoStage('{}'): no demuxer named '{}'; leaving the format "
          "to autodetection", this->id(), _format));
    }
  }

  int rc = 0;
  if (_inputs.size() > 1) {
    // SEVERAL inputs are opened ONE AT A TIME rather than handed to the
    // concat demuxer. The demuxer would present them as a single
    // flattened stream, and a block codec's padding is then only
    // visible for the first file -- every interior seam keeps its
    // encoder priming, which is audible as a gap at every join. Opening
    // each part in turn keeps its own gapless metadata; `_seq_base_us`
    // carries the joined timeline across the boundary.
    if (forced != nullptr) {
      session()->error(fmt(
          "LoadVideoStage('{}'): `format: \"{}\"` cannot be combined with "
          "an array of inputs -- an array IS the join. Give one path to "
          "force a demuxer, or drop `format`", this->id(), _format));
    }
    _input_url = _inputs[_seq_index];
  }
  {
    rc = _libs->avformat().api.open_input(&fctx, _input_url.c_str(),
                                          forced, &opts);
  }
  _libs->avutil().api.dict_free(&opts);
  if (rc < 0) {
    session()->error(fmt(
        "LoadVideoStage('{}'): avformat_open_input({}) "
        "failed: {}", this->id(), input_doc_(), av_err_(rc)));
    return false;
  }
  _fctx = fctx;

  rc = _libs->avformat().api.find_stream_info(_fctx, nullptr);
  if (rc < 0) {
    session()->error(fmt(
        "LoadVideoStage('{}'): find_stream_info failed: {}",
        this->id(), av_err_(rc)));
    return false;
  }

  if (_video_port >= 0) {
    _v_stream_idx = pick_stream_(AVMEDIA_TYPE_VIDEO,
                                 _video_stream_index);
    if (_v_stream_idx < 0) {
      session()->warn(fmt(
        "LoadVideoStage('{}'): no video stream in {}; "
        "video oport will be closed immediately",
        this->id(), input_doc_()));
    } else {
      cache_stream_(_v_stream_idx, /*video=*/true);
    }
  }
  if (_audio_port >= 0) {
    _a_stream_idx = pick_stream_(AVMEDIA_TYPE_AUDIO,
                                 _audio_stream_index);
    if (_a_stream_idx < 0) {
      session()->warn(fmt(
        "LoadVideoStage('{}'): no audio stream in {}; "
        "audio oport will be closed immediately",
        this->id(), input_doc_()));
    } else {
      cache_stream_(_a_stream_idx, /*video=*/false);
    }
  }

  // Seek AFTER the streams are chosen: the target is in a stream's own
  // time base, and it is the VIDEO stream's when there is one -- seeking
  // on the audio stream would land between video keyframes and hand the
  // decoder a stream that starts mid-GOP.
  if (_start_us > 0 && (_v_stream_idx >= 0 || _a_stream_idx >= 0)) {
    const bool on_video = _v_stream_idx >= 0;
    const int  sidx     = on_video ? _v_stream_idx : _a_stream_idx;
    const AVRational stb = on_video ? _vmeta.time_base : _ameta.time_base;
    const AVRational us  = {1, 1000000};
    const std::int64_t ts =
        _libs->avutil().api.rescale_q(_start_us, us, stb);
    const int srr = _libs->avformat().api.seek_frame(_fctx, sidx, ts,
                                                     AVSEEK_FLAG_BACKWARD);
    if (srr < 0) {
      // Not fatal: a stream with no index refuses the seek, and reading
      // from the start still produces the right WINDOW -- `duration_s`
      // still ends it, and only the work to reach it is wasted.
      session()->warn(fmt(
          "LoadVideoStage('{}'): seek to {:.3f}s failed ({}); reading from "
          "the start instead", this->id(), _start_s, av_err_(srr)));
    }
  }
  _vmeta.last_us = _start_us;
  _ameta.last_us = _start_us;
  if (_start_us > 0 || _has_stop) {
    session()->info(fmt("LoadVideoStage('{}'): {}{}", this->id(),
                        input_doc_(), window_doc_()));
  }

  if (_pkt == nullptr) {
    // One packet for the whole run: re-opening for the next joined part
    // must not leak the previous one.
    _pkt = _libs->avcodec().api.packet_alloc();
    if (!_pkt) {
      session()->error(fmt(
          "LoadVideoStage('{}'): av_packet_alloc failed",
          this->id()));
      return false;
    }
  }
  return true;
}

std::string
LoadVideoStage::input_doc_() const
{
  if (_inputs.size() <= 1) {
    return "'" + _input_url + "'";
  }
  return fmt("{} inputs joined, '{}' … '{}'", _inputs.size(),
             _inputs.front(), _inputs.back())();
}

std::string
LoadVideoStage::window_doc_() const
{
  if (_start_us <= 0 && !_has_stop) { return std::string(); }
  if (!_has_stop) {
    return fmt(", from {:.3f}s to the end", _start_s)();
  }
  return fmt(", window [{:.3f}s, {:.3f}s)", _start_s,
             _start_s + _duration_s)();
}

bool
LoadVideoStage::all_past_end_() const noexcept
{
  if (_video_port >= 0 && _v_stream_idx >= 0 && !_vmeta.past_end) {
    return false;
  }
  if (_audio_port >= 0 && _a_stream_idx >= 0 && !_ameta.past_end) {
    return false;
  }
  return true;
}

void
LoadVideoStage::reset_run_state()
{
  if (_pkt != nullptr && _libs != nullptr) {
    _libs->avcodec().api.packet_free(&_pkt);
  }
  if (_fctx != nullptr && _libs != nullptr) {
    _libs->avformat().api.close_input(&_fctx);
  }
  _pkt          = nullptr;
  _fctx         = nullptr;
  _v_stream_idx = -1;
  _a_stream_idx = -1;
  _vmeta        = StreamMeta{};
  _ameta        = StreamMeta{};
  _eof          = false;
  _lead_in_logged = false;
  _v_packets    = 0;
  _a_packets    = 0;
}

Job
LoadVideoStage::initialize(RuntimeContext& ctx)
{
  try {
    open_input_();
  } catch (const exception& e) {
    session()->warn(fmt(
      "decoder('{}'): {}; stopping", this->id(), e.what()));
    ctx.signal_done();
    co_return;
  }

  // No header beat. Every field a StreamParams header used to carry --
  // geometry, cadence, rate, channels -- rides on EVERY segment, which
  // is what lets a consumer join a stream late and what makes this
  // interchangeable with rtsp-capture. A header would also have to be
  // distinguished by try_get<T> from the beats after it, and a consumer
  // that forgot would read it as a packet.
  co_return;
}

Job
LoadVideoStage::process(RuntimeContext& ctx)
{
  if (!_fctx || !_pkt) {
    // initialize() failed earlier; nothing to do.
    ctx.signal_done();
    co_return;
  }
  if (_eof) { ctx.signal_done(); co_return; }

  const int rc = _libs->avformat().api.read_frame(_fctx, _pkt);
  if (rc == AVERROR_EOF) {
    // The end of ONE part is not the end of the run when several inputs
    // are being joined: carry the timeline forward and open the next.
    if (_seq_index + 1 < _inputs.size()) {
      if (advance_to_next_input_()) { co_return; }
    }
    _eof = true;
    session()->info(fmt(
        "LoadVideoStage('{}'): end of {} after {} video + {} audio "
        "packet(s)", this->id(), input_doc_(), _v_packets, _a_packets));
    ctx.signal_done();
    co_return;
  }
  if (rc < 0) {
    session()->warn(fmt(
      "LoadVideoStage('{}'): read_frame: {}", this->id(), av_err_(rc)));
    ctx.signal_done();
    co_return;
  }

  // Which port, if any. A packet from a stream nobody asked for -- a
  // second language, a cover image, the video track when enable_video
  // is off -- is dropped, and the driver calls again.
  int port = -1;
  bool video = false;
  if (_pkt->stream_index == _v_stream_idx && _video_port >= 0) {
    port = _video_port;
    video = true;
  } else if (_pkt->stream_index == _a_stream_idx && _audio_port >= 0) {
    port = _audio_port;
  }
  if (port < 0 || _pkt->size <= 0) {
    _libs->avcodec().api.packet_unref(_pkt);
    co_return;
  }

  // Past the end of the window? Tested per STREAM, on the packet's
  // start: the two do not interleave evenly, so ending the run on the
  // first stream to get there would truncate its partner. The packet
  // straddling the end is kept whole -- this stage does not decode and
  // so cannot cut inside one.
  //
  // Nothing is dropped at the START. The seek landed on the keyframe at
  // or before `start_s` and the frames after it are decoded from the
  // packets in between, so they are emitted; see the header.
  StreamMeta& m = video ? _vmeta : _ameta;
  if (_has_stop) {
    const AVRational us = {1, 1000000};
    std::int64_t t = m.last_us;
    if (_pkt->pts != AV_NOPTS_VALUE) {
      t = _libs->avutil().api.rescale_q(_pkt->pts, m.time_base, us);
    }
    if (t >= _stop_us) {
      m.past_end = true;
      _libs->avcodec().api.packet_unref(_pkt);
      if (all_past_end_()) {
        _eof = true;
        session()->info(fmt(
            "LoadVideoStage('{}'): reached {:.3f}s of {} after {} video + "
            "{} audio packet(s)", this->id(), _start_s + _duration_s,
            input_doc_(), _v_packets, _a_packets));
        ctx.signal_done();
      }
      co_return;
    }
  }

  auto seg = segment_(video);
  if (!_lead_in_logged && _start_us > 0) {
    _lead_in_logged = true;
    const std::int64_t first_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            seg->start_utc.time_since_epoch()).count();
    session()->info(fmt(
        "LoadVideoStage('{}'): seek for start_s={:.3f}s landed at {:.3f}s "
        "-- {:.3f}s of lead-in emitted so the decoder has its keyframe",
        this->id(), _start_s, first_us / 1e6,
        (_start_us - first_us) / 1e6));
  }
  if (video) { ++_v_packets; } else { ++_a_packets; }
  _libs->avcodec().api.packet_unref(_pkt);
  // The payload is bound to its own local before the suspend: a
  // temporary built inside the co_await argument has its lifetime
  // tangled with the frame layout across the resume.
  co_await ctx.write(static_cast<unsigned>(port), std::move(seg));
  co_return;
}

VPIPE_REGISTER_STAGE(LoadVideoStage)
VPIPE_REGISTER_SPEC(LoadVideoStage, kSpec)

}

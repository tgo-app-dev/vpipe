#include "stages/audio-video/video-to-rgb-stage.h"
#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/encoded-segment.h"
#include "common/oport-policy.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "apple-silicon/metal-compute/image-ops.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#endif

#include <chrono>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/error.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixfmt.h>
}

using namespace std;

namespace vpipe {

namespace {

// EncodedSegment::data carries an H.264 bitstream that needs to be
// split into per-frame access units before being fed to
// avcodec_send_packet — the decoder takes one AU per call, and
// feeding it a multi-AU blob makes it decode only the first AU and
// discard the rest.
//
// Two wire framings are supported, picked per Beat by sniffing the
// leading bytes:
//   * AVCC (4-byte BE length prefix per NAL) -- produced by
//     extract_encoded_segments (file-based catch-up path) and by
//     test fixtures.
//   * Annex-B (3- or 4-byte start codes 0x000001 / 0x00000001) --
//     produced by rtsp-capture's live per-GOP publisher (RTSP
//     depacketizer and global-header encoder both deliver annex-B).
//
// Access-unit boundary rule is identical for both framings: a VCL
// NAL (nal_unit_type 1 or 5) begins a new access unit. Any non-VCL
// NALs (AUD/SEI/SPS/PPS/...) preceding it belong to that AU. This
// is the well-known "primary-coded-picture starts a new AU"
// heuristic from the H.264 spec; sufficient for IP-camera streams
// that emit one slice per picture (no multi-slice frames).
//
// Returns a vector of (offset, length) byte ranges into `data` —
// each range is one AU's worth of NAL units in its native framing,
// ready to hand to avcodec_send_packet as is.

std::vector<std::pair<size_t, size_t>>
split_avcc_aus_(const uint8_t* data, size_t size) noexcept
{
  std::vector<std::pair<size_t, size_t>> aus;
  size_t pos       = 0;
  size_t au_start  = 0;
  bool   have_vcl  = false;
  while (pos + 4 <= size) {
    const uint32_t nal_len =
        (static_cast<uint32_t>(data[pos    ]) << 24) |
        (static_cast<uint32_t>(data[pos + 1]) << 16) |
        (static_cast<uint32_t>(data[pos + 2]) <<  8) |
         static_cast<uint32_t>(data[pos + 3]);
    if (nal_len == 0 || pos + 4 + nal_len > size) {
      break;            // malformed framing -- bail out
    }
    const uint8_t nal_type = data[pos + 4] & 0x1F;
    const bool    is_vcl   = (nal_type == 1 || nal_type == 5);
    if (is_vcl && have_vcl) {
      // Boundary: previous AU ends here, new AU starts at `pos`.
      aus.emplace_back(au_start, pos - au_start);
      au_start = pos;
    }
    if (is_vcl) {
      have_vcl = true;
    }
    pos += 4 + nal_len;
  }
  if (au_start < size) {
    aus.emplace_back(au_start, size - au_start);
  }
  return aus;
}

// Detect annex-B framing by checking for a start code at offset 0.
// AVCC's leading bytes are a 32-bit BE length, never starting with
// {0,0,0,1} (that would mean a 1-byte NAL, which doesn't exist in
// valid streams) and never with {0,0,1} unless the next length byte
// happens to be 1 (a 256-byte+ NAL whose top byte is 0x01 -- also
// effectively impossible for the first NAL of a GOP).
bool
looks_annexb_(const uint8_t* data, size_t size) noexcept
{
  if (size >= 4
      && data[0] == 0 && data[1] == 0
      && data[2] == 0 && data[3] == 1) {
    return true;
  }
  if (size >= 3
      && data[0] == 0 && data[1] == 0 && data[2] == 1) {
    return true;
  }
  return false;
}

// Walk one AU and return true if it contains an IDR NAL (type 5).
// This is the gate the live-decode sync state machine uses: an
// IDR is the only AU a freshly-opened or freshly-flushed H.264
// decoder can safely consume (especially under VideoToolbox
// hwaccel, where a non-IDR first packet returns AVERROR_UNKNOWN
// and leaves the decoder in an unrecoverable error state).
bool
au_contains_idr_(const uint8_t* au, size_t size, bool annexb) noexcept
{
  if (annexb) {
    size_t i = 0;
    while (i + 2 < size) {
      int sc = 0;
      if (i + 3 < size
          && au[i]   == 0 && au[i+1] == 0
          && au[i+2] == 0 && au[i+3] == 1) {
        sc = 4;
      } else if (au[i] == 0 && au[i+1] == 0 && au[i+2] == 1) {
        sc = 3;
      }
      if (sc == 0) { ++i; continue; }
      if (i + sc >= size) { break; }
      if ((au[i + sc] & 0x1F) == 5) { return true; }
      i += sc + 1;
    }
  } else {
    size_t pos = 0;
    while (pos + 4 < size) {
      const uint32_t nal_len =
          (static_cast<uint32_t>(au[pos    ]) << 24)
        | (static_cast<uint32_t>(au[pos + 1]) << 16)
        | (static_cast<uint32_t>(au[pos + 2]) <<  8)
        |  static_cast<uint32_t>(au[pos + 3]);
      if (nal_len == 0 || pos + 4 + nal_len > size) { break; }
      if ((au[pos + 4] & 0x1F) == 5) { return true; }
      pos += 4 + nal_len;
    }
  }
  return false;
}

// Walk one AU and return a compact summary of its constituent NAL
// unit types ("7,8,5,1x4,6,1x12" for SPS+PPS+IDR+4×P+SEI+12×P).
// Triage-only: tells us at a glance whether a failing AU is one
// frame or a multi-frame blob the splitter accidentally merged.
std::string
au_nal_summary_(const uint8_t* au, size_t size, bool annexb)
{
  std::vector<int> types;
  types.reserve(8);
  if (annexb) {
    size_t i = 0;
    while (i + 2 < size) {
      int sc = 0;
      if (i + 3 < size
          && au[i  ] == 0 && au[i+1] == 0
          && au[i+2] == 0 && au[i+3] == 1) {
        sc = 4;
      } else if (au[i  ] == 0 && au[i+1] == 0 && au[i+2] == 1) {
        sc = 3;
      }
      if (sc == 0) { ++i; continue; }
      if (i + sc >= size) { break; }
      types.push_back(au[i + sc] & 0x1F);
      i += sc + 1;
    }
  } else {
    size_t pos = 0;
    while (pos + 4 < size) {
      const uint32_t nal_len =
          (static_cast<uint32_t>(au[pos    ]) << 24)
        | (static_cast<uint32_t>(au[pos + 1]) << 16)
        | (static_cast<uint32_t>(au[pos + 2]) <<  8)
        |  static_cast<uint32_t>(au[pos + 3]);
      if (nal_len == 0 || pos + 4 + nal_len > size) { break; }
      types.push_back(au[pos + 4] & 0x1F);
      pos += 4 + nal_len;
    }
  }
  std::string out;
  for (size_t i = 0; i < types.size(); ) {
    size_t j = i + 1;
    while (j < types.size() && types[j] == types[i]) { ++j; }
    if (!out.empty()) { out.push_back(','); }
    out += std::to_string(types[i]);
    if (j - i > 1) {
      out += "x";
      out += std::to_string(j - i);
    }
    i = j;
  }
  return out;
}

// Hex-dump the first up-to-n bytes of `buf`, space-separated.
std::string
hex_prefix_(const uint8_t* buf, size_t size, size_t n)
{
  static const char* H = "0123456789abcdef";
  const size_t k = size < n ? size : n;
  std::string out;
  out.reserve(k * 3);
  for (size_t i = 0; i < k; ++i) {
    if (i) { out.push_back(' '); }
    out.push_back(H[(buf[i] >> 4) & 0xF]);
    out.push_back(H[ buf[i]       & 0xF]);
  }
  return out;
}

// Hex-encode `db_key` (raw 8-byte BE microseconds-since-epoch key
// the capture stage writes into the `<camera>-videos` LMDB sub-db).
// Useful in logs because the raw bytes are not printable.
std::string
db_key_hex_(std::string_view k)
{
  static const char* H = "0123456789abcdef";
  std::string out;
  out.reserve(k.size() * 2);
  for (unsigned char c : k) {
    out.push_back(H[(c >> 4) & 0xF]);
    out.push_back(H[ c       & 0xF]);
  }
  return out;
}

std::vector<std::pair<size_t, size_t>>
split_annexb_aus_(const uint8_t* data, size_t size) noexcept
{
  // Walk start codes one-pass and decide AU boundaries on the fly.
  std::vector<std::pair<size_t, size_t>> aus;
  size_t au_start = 0;
  bool   have_au  = false;
  bool   have_vcl = false;
  size_t i = 0;
  auto scan_start_code = [&](size_t p) -> int {
    // Returns 4 for {0,0,0,1}, 3 for {0,0,1}, else 0.
    if (p + 3 < size
        && data[p] == 0 && data[p+1] == 0
        && data[p+2] == 0 && data[p+3] == 1) {
      return 4;
    }
    if (p + 2 < size
        && data[p] == 0 && data[p+1] == 0 && data[p+2] == 1) {
      return 3;
    }
    return 0;
  };
  while (i + 2 < size) {
    int sc = scan_start_code(i);
    if (sc == 0) { ++i; continue; }
    const size_t nal_hdr = i + sc;
    if (nal_hdr >= size) { break; }
    const uint8_t nal_type = data[nal_hdr] & 0x1F;
    const bool    is_vcl   = (nal_type == 1 || nal_type == 5);
    if (!have_au) {
      au_start = i;
      have_au  = true;
    } else if (is_vcl && have_vcl) {
      aus.emplace_back(au_start, i - au_start);
      au_start = i;
      have_vcl = false;
    }
    if (is_vcl) { have_vcl = true; }
    i = nal_hdr + 1;
  }
  if (have_au && au_start < size) {
    aus.emplace_back(au_start, size - au_start);
  }
  return aus;
}

}

VideoToRgbStage::VideoToRgbStage(const SessionContextIntf* s,
                                 string                    id,
                                 vector<InEdge>            iports,
                                 FlexData                  config)
  : TypedStage<VideoToRgbStage>(s, std::move(id), std::move(iports),
                                std::move(config))
  , _libs(s->ffmpeg_libraries())
{
  // Scalar attribute defaults live in kSpec.attrs; attr_* resolves the
  // configured value else that default.
  const FlexData& cfg = this->config();
  _normalize      = attr_bool("normalize");
  _oport_capacity = static_cast<unsigned>(attr_uint("oport_capacity"));

  {
    std::string s = attr_str("hwaccel");
    if      (s == "auto")         { _hw_mode = HwMode::Auto; }
    else if (s == "videotoolbox") { _hw_mode = HwMode::Videotoolbox; }
    else if (s == "none")         { _hw_mode = HwMode::None; }
    else {
      session()->warn(fmt(
        "video-to-rgb('{}'): unknown hwaccel '{}' -- using 'auto'",
        this->id(), s));
      _hw_mode = HwMode::Auto;
    }
  }
  {
    std::string s = attr_str("output_dtype");
    if      (s == "f32") { _output_dtype = TensorBeat::DType::F32; }
    else if (s == "u8")  { _output_dtype = TensorBeat::DType::U8; }
    else {
      session()->warn(fmt(
        "video-to-rgb('{}'): unknown output_dtype '{}' -- using 'f32'",
        this->id(), s));
      _output_dtype = TensorBeat::DType::F32;
    }
  }
  // output_width / output_height: optional pair. Both keys must be set
  // together or omitted together. > 0 required when set. Even
  // dimensions are recommended because the NV12 chroma plane needs
  // chroma-aligned crops; we accept odd values and let the crop
  // computation round them down internally. Validation is deferred to
  // launch (see Stage::fail_config).
  if (cfg.is_object()) {
    auto root = cfg.as_object();
    const bool has_ow = root.contains("output_width");
    const bool has_oh = root.contains("output_height");
    if (has_ow != has_oh) {
      fail_config(fmt(
        "video-to-rgb('{}'): output_width and output_height must be "
        "set together (or both omitted)",
        this->id()));
    }
    if (has_ow && has_oh) {
      int64_t ow = root.at("output_width").as_int(0);
      int64_t oh = root.at("output_height").as_int(0);
      if (ow <= 0 || oh <= 0) {
        fail_config(fmt(
          "video-to-rgb('{}'): output_width / output_height must be "
          "> 0 (got {}x{})", this->id(), ow, oh));
      } else {
        _output_width  = static_cast<int>(ow);
        _output_height = static_cast<int>(oh);
      }
    }
  }
  if (_oport_capacity == 0) {
    _oport_capacity = 4;
  }

  allocate_oports(spec().oports.size());
  set_oport_policy(0, {_oport_capacity, OverrunPolicy::Backpressure});

#ifdef VPIPE_BUILD_APPLE_SILICON
  // Cache the session's Metal runtime pointer. May be nullptr (non-
  // apple builds) or non-null but !valid() (Metal init failed). The
  // Metal fast path only engages when valid() and the frame is a
  // videotoolbox-decoded CVPixelBuffer with u8 output.
  _mc = s->metal_compute();
#endif

  // Install an FFmpeg log tap that captures the most recent
  // codec-related warning / error into _last_ffmpeg_log. When
  // libavcodec's hwaccel wrapper hides a VideoToolbox OSStatus
  // behind AVERROR_UNKNOWN, the actual error text shows up here
  // (e.g. "videotoolbox: VTDecompressionSessionDecodeFrame()
  // failed: -12909"). decode_segment_'s error WARN reads the
  // latest captured line and surfaces it alongside our own
  // context for triage. Callback runs on an FFmpeg thread and
  // MUST be short + non-reentrant; it just locks _ffmpeg_log_mtx
  // and assigns the string. The session log call is done back on
  // the pipeline thread by decode_segment_.
  _ffmpeg_log_tap = install_log_tap(
      [this](int level, std::string_view tag, std::string_view msg) {
        if (level > AV_LOG_WARNING) { return; }
        // Filter to codec / videotoolbox / h264 tags so unrelated
        // FFmpeg noise (rtsp, rtp, mov, etc.) doesn't drown the
        // diagnostic.
        const bool relevant =
              tag == "videotoolbox"
           || tag == "h264"
           || tag == "AVCodecContext"
           || tag.find("h264") != std::string_view::npos;
        if (!relevant) { return; }
        std::lock_guard<std::mutex> lk(_ffmpeg_log_mtx);
        _last_ffmpeg_log = std::string(tag) + ": " + std::string(msg);
      });
}

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "normalize", .type = ConfigType::Bool,
   .doc = "F32 only: divide bytes by 255 -> [0,1]", .def_bool = true},
  {.key = "oport_capacity", .type = ConfigType::Uint,
   .doc = "oport buffer depth", .def_uint = 4},
  {.key = "hwaccel", .type = ConfigType::String,
   .doc = "auto | videotoolbox | none", .def_str = "auto"},
  {.key = "output_dtype", .type = ConfigType::String,
   .doc = "f32 | u8", .def_str = "f32"},
  {.key = "output_width", .type = ConfigType::Int,
   .doc = "rescale target width; pair w/ output_height", .def_int = 0},
  {.key = "output_height", .type = ConfigType::Int,
   .doc = "rescale target height; pair w/ output_width", .def_int = 0},
};
const PortSpec kIports[] = {
  {.name = "segments", .doc = "EncodedSegment: AVCC H.264 + extradata",
   .type = &typeid(EncodedSegmentPayload),
   .tags = "video-encoder-segments", .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "frames", .doc = "planar RGB TensorBeat [3,H,W] (F32 or U8)",
   .type = &typeid(TensorBeatPayload),
   .tags = "rgb-frames", .clock_group = 1},
};
const StageSpec kSpec = {
  .type_name = "video-to-rgb",
  .doc       = "Decodes H.264 EncodedSegments (FFmpeg SW/VideoToolbox) "
               "to planar RGB TensorBeats, one per frame; optional "
               "crop+rescale. Crosses segment-rate -> frame-rate clocks.",
  .display_name = "Video → RGB",
  .category  = StageCategory::Visual,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
VideoToRgbStage::spec() const noexcept
{
  return kSpec;
}

VideoToRgbStage::~VideoToRgbStage()
{
  remove_log_tap(_ffmpeg_log_tap);
  teardown_sws_();
  teardown_codec_();
  if (_hw_device_ctx) {
    _libs->avutil().api.buffer_unref(&_hw_device_ctx);
  }
}

string
VideoToRgbStage::av_err_(int rc) const
{
  char buf[256];
  _libs->avutil().api.strerror(rc, buf, sizeof buf);
  return string(buf);
}

void
VideoToRgbStage::teardown_codec_()
{
  if (_yuv_in) {
    _libs->avutil().api.frame_free(&_yuv_in);
  }
  if (_sw_frame) {
    _libs->avutil().api.frame_free(&_sw_frame);
  }
  if (_pkt) {
    _libs->avcodec().api.packet_free(&_pkt);
  }
  if (_hw.ctx) {
    _libs->avcodec().api.free_context(&_hw.ctx);
  }
  if (_sw.ctx) {
    _libs->avcodec().api.free_context(&_sw.ctx);
  }
  _hw = CodecSlot{};
  _sw = CodecSlot{};
  _last_codec_id = 0;
  _hw_broken_this_gop    = false;
  _hw_drops_since_desync = 0;
  _sw_drops_since_desync = 0;
}

void
VideoToRgbStage::teardown_sws_()
{
  if (_sws_chroma) {
    _libs->swscale().api.free_context(_sws_chroma);
    _sws_chroma = nullptr;
  }
  if (_sws_rgb) {
    _libs->swscale().api.free_context(_sws_rgb);
    _sws_rgb = nullptr;
  }
  if (_yuv444) {
    _libs->avutil().api.frame_free(&_yuv444);
  }
  if (_gbrp) {
    _libs->avutil().api.frame_free(&_gbrp);
  }
  _last_w = _last_h = 0;
  _last_dec_pix_fmt = -1;
}

namespace {

// Shared body of ensure_codec_hw_ / ensure_codec_sw_. `try_hwaccel`
// controls whether videotoolbox is attempted at all. Returns the
// fully-opened AVCodecContext on success (caller takes ownership),
// or nullptr on failure (already error-logged via `sess`). Sets
// *out_hw_active to true iff hwaccel was wired up.
//
// Lives outside the class so both slots can share the heavy
// alloc/open dance without duplicating ~100 lines.
struct OpenCodecParams {
  const FFmpegLibraries*    libs                = nullptr;
  const SessionContextIntf* sess                = nullptr;
  std::string               id;                 // for log prefix
  const EncodedSegment*     seg                 = nullptr;
  bool                      try_hwaccel         = false;
  AVBufferRef**             hw_device_ctx_inout = nullptr;
  enum AVPixelFormat*       hw_pix_fmt_inout    = nullptr;
  void*                     get_format_opaque   = nullptr;
  enum AVPixelFormat       (*get_format_cb)(AVCodecContext*,
                                            const enum AVPixelFormat*)
                                              = nullptr;
  bool*                     out_hw_active       = nullptr;
};

AVCodecContext*
open_h264_codec_(const OpenCodecParams& p)
{
  *p.out_hw_active = false;

  AVCodecID id_ = static_cast<AVCodecID>(p.seg->codec_id);
  const AVCodec* codec = p.libs->avcodec().api.find_decoder(id_);
  if (!codec) {
    p.sess->error(fmt(
      "video-to-rgb('{}'): no decoder for codec_id={}",
      p.id, static_cast<int>(id_)));
    return nullptr;
  }
  AVCodecContext* cctx = p.libs->avcodec().api.alloc_context3(codec);
  if (!cctx) {
    p.sess->error(fmt(
      "video-to-rgb('{}'): avcodec_alloc_context3 failed", p.id));
    return nullptr;
  }
  cctx->width       = static_cast<int>(p.seg->width);
  cctx->height      = static_cast<int>(p.seg->height);
  cctx->thread_type = FF_THREAD_SLICE;

  if (p.try_hwaccel) {
    enum AVPixelFormat want = AV_PIX_FMT_NONE;
    for (int i = 0; ; ++i) {
      const AVCodecHWConfig* hwc =
          p.libs->avcodec().api.get_hw_config(codec, i);
      if (!hwc) { break; }
      if (hwc->device_type == AV_HWDEVICE_TYPE_VIDEOTOOLBOX
          && (hwc->methods
              & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX)) {
        want = hwc->pix_fmt;
        break;
      }
    }
    if (want != AV_PIX_FMT_NONE && !*p.hw_device_ctx_inout) {
      int rc = p.libs->avutil().api.hwdevice_ctx_create(
          p.hw_device_ctx_inout, AV_HWDEVICE_TYPE_VIDEOTOOLBOX,
          nullptr, nullptr, 0);
      if (rc < 0) {
        p.sess->warn(fmt(
          "video-to-rgb('{}'): av_hwdevice_ctx_create videotoolbox: "
          "{} -- this slot will run software", p.id, rc));
        *p.hw_device_ctx_inout = nullptr;
        want = AV_PIX_FMT_NONE;
      }
    }
    if (want != AV_PIX_FMT_NONE && *p.hw_device_ctx_inout) {
      cctx->hw_device_ctx =
          p.libs->avutil().api.buffer_ref(*p.hw_device_ctx_inout);
      cctx->opaque     = p.get_format_opaque;
      cctx->get_format = p.get_format_cb;
      *p.hw_pix_fmt_inout = want;
      *p.out_hw_active   = true;
    }
  }

  if (!p.seg->extradata.empty()) {
    const size_t n = p.seg->extradata.size();
    void* buf = p.libs->avutil().api.malloc(
        n + AV_INPUT_BUFFER_PADDING_SIZE);
    if (!buf) {
      p.libs->avcodec().api.free_context(&cctx);
      p.sess->error(fmt(
        "video-to-rgb('{}'): av_malloc({} bytes) failed",
        p.id, n + AV_INPUT_BUFFER_PADDING_SIZE));
      return nullptr;
    }
    std::memcpy(buf, p.seg->extradata.data(), n);
    std::memset(static_cast<uint8_t*>(buf) + n, 0,
                AV_INPUT_BUFFER_PADDING_SIZE);
    cctx->extradata      = static_cast<uint8_t*>(buf);
    cctx->extradata_size = static_cast<int>(n);
  }

  int rc = p.libs->avcodec().api.open2(cctx, codec, nullptr);
  if (rc < 0) {
    p.libs->avcodec().api.free_context(&cctx);
    p.sess->error(fmt(
      "video-to-rgb('{}'): avcodec_open2 ({}-only path): rc={}",
      p.id, p.try_hwaccel ? "hw" : "sw", rc));
    return nullptr;
  }
  return cctx;
}

}

void
VideoToRgbStage::ensure_codec_hw_(const EncodedSegment& seg)
{
  if (_hw.ctx && _last_codec_id == seg.codec_id) {
    return;
  }
  if (_hw.ctx) {
    session()->info(fmt(
      "video-to-rgb('{}'): codec switch mid-stream: codec_id "
      "{} -> {} (re-initialising HW decoder)",
      this->id(),
      static_cast<int>(_last_codec_id),
      static_cast<int>(seg.codec_id)));
  } else if (!_sw.ctx) {
    // Only log "first segment" once across the stage lifetime,
    // i.e. when neither slot has been opened yet.
    session()->info(fmt(
      "video-to-rgb('{}'): first segment: codec_id={} {}x{} "
      "extradata={}b db_key={} parent='{}'",
      this->id(),
      static_cast<int>(seg.codec_id), seg.width, seg.height,
      seg.extradata.size(),
      db_key_hex_(seg.db_key), seg.path));
  }
  // Tear down ONLY the HW slot (not _sw).
  if (_hw.ctx) {
    _libs->avcodec().api.free_context(&_hw.ctx);
  }
  _hw = CodecSlot{};

  OpenCodecParams p{};
  p.libs                  = _libs;
  p.sess                  = session();
  p.id                    = this->id();
  p.seg                   = &seg;
  p.try_hwaccel           = (_hw_mode != HwMode::None);
  p.hw_device_ctx_inout   = &_hw_device_ctx;
  p.hw_pix_fmt_inout      = &_hw_pix_fmt;
  p.get_format_opaque     = this;
  p.get_format_cb         = &VideoToRgbStage::get_format_;
  p.out_hw_active         = &_hw.hw_active;
  AVCodecContext* cctx = open_h264_codec_(p);
  if (!cctx) { return; }
  _hw.ctx    = cctx;
  _hw.synced = false;
  if (_hw.hw_active && !_hw.hw_logged) {
    session()->info(fmt(
      "video-to-rgb('{}'): using videotoolbox hwaccel "
      "(hw_pix_fmt={})", this->id(),
      static_cast<int>(_hw_pix_fmt)));
    _hw.hw_logged = true;
  }
  _last_codec_id = seg.codec_id;

  // Lazy alloc of shared scratch on first context.
  if (!_pkt) {
    _pkt = _libs->avcodec().api.packet_alloc();
  }
  if (!_yuv_in) {
    _yuv_in = _libs->avutil().api.frame_alloc();
  }
  if (!_pkt || !_yuv_in) {
    session()->error(fmt(
      "video-to-rgb('{}'): packet/frame alloc failed", this->id()));
    teardown_codec_();
  }
}

void
VideoToRgbStage::ensure_codec_sw_(const EncodedSegment& seg)
{
  if (_sw.ctx && _last_codec_id == seg.codec_id) {
    return;
  }
  if (_sw.ctx) {
    // Codec change while a SW slot exists -- rebuild.
    _libs->avcodec().api.free_context(&_sw.ctx);
  }
  _sw = CodecSlot{};

  OpenCodecParams p{};
  p.libs                  = _libs;
  p.sess                  = session();
  p.id                    = this->id();
  p.seg                   = &seg;
  p.try_hwaccel           = false;        // SW-only by construction
  // The SW slot must NOT touch the shared HW device context; pass
  // a local null pointer so open_h264_codec_ skips the hwaccel
  // probe entirely (try_hwaccel=false already does, this is
  // belt-and-suspenders).
  AVBufferRef* unused_hw_ctx = nullptr;
  enum AVPixelFormat unused_hw_pix_fmt = AV_PIX_FMT_NONE;
  p.hw_device_ctx_inout   = &unused_hw_ctx;
  p.hw_pix_fmt_inout      = &unused_hw_pix_fmt;
  p.get_format_opaque     = nullptr;
  p.get_format_cb         = nullptr;
  bool unused_hw_active   = false;
  p.out_hw_active         = &unused_hw_active;

  AVCodecContext* cctx = open_h264_codec_(p);
  if (!cctx) { return; }
  _sw.ctx       = cctx;
  _sw.synced    = false;
  _sw.hw_active = false;
  session()->info(fmt(
    "video-to-rgb('{}'): opened software fallback decoder "
    "(codec_id={} {}x{})",
    this->id(), static_cast<int>(seg.codec_id),
    seg.width, seg.height));
}

void
VideoToRgbStage::emitted_dims_(int src_w, int src_h,
                               int* out_w, int* out_h) const noexcept
{
  if (_output_width > 0 && _output_height > 0) {
    *out_w = _output_width;
    *out_h = _output_height;
  } else {
    *out_w = src_w;
    *out_h = src_h;
  }
}

namespace {

// Mirror of MetalRuntime's centered_crop_: pick a crop region of the
// source that exactly matches the output aspect ratio, with even
// crop dimensions so YUV-4:2:0 chroma plane offsets stay aligned.
// Lives here so the CPU swscale path uses identical crop arithmetic.
struct CenteredCrop {
  int crop_w;
  int crop_h;
  int crop_left;
  int crop_top;
};

CenteredCrop
centered_crop_(int in_w, int in_h, int out_w, int out_h)
{
  const long long in_x_oh =
      static_cast<long long>(in_w) * out_h;
  const long long ow_x_ih =
      static_cast<long long>(out_w) * in_h;
  CenteredCrop c{};
  if (in_x_oh == ow_x_ih) {
    c.crop_w    = in_w;
    c.crop_h    = in_h;
    c.crop_left = 0;
    c.crop_top  = 0;
    return c;
  }
  if (in_x_oh > ow_x_ih) {
    c.crop_h    = in_h;
    c.crop_w    =
        static_cast<int>(static_cast<long long>(in_h) * out_w / out_h);
    if (c.crop_w & 1) { --c.crop_w; }
    if (c.crop_w > in_w) { c.crop_w = in_w & ~1; }
    c.crop_left = (in_w - c.crop_w) / 2;
    if (c.crop_left & 1) { --c.crop_left; }
    c.crop_top  = 0;
  } else {
    c.crop_w    = in_w;
    c.crop_h    =
        static_cast<int>(static_cast<long long>(in_w) * out_h / out_w);
    if (c.crop_h & 1) { --c.crop_h; }
    if (c.crop_h > in_h) { c.crop_h = in_h & ~1; }
    c.crop_top  = (in_h - c.crop_h) / 2;
    if (c.crop_top & 1) { --c.crop_top; }
    c.crop_left = 0;
  }
  return c;
}

}  // namespace

void
VideoToRgbStage::ensure_sws_(int w, int h, int dec_pix_fmt)
{
  if (_sws_chroma && _sws_rgb
      && w == _last_w && h == _last_h
      && dec_pix_fmt == _last_dec_pix_fmt) {
    return;
  }
  // Surface resolution / pix_fmt changes -- they invalidate any
  // size-bound downstream stage and are useful when triaging "why
  // did the decoder start failing mid-stream".
  if (_last_w != 0 || _last_h != 0) {
    session()->info(fmt(
      "video-to-rgb('{}'): sws geometry change: {}x{} pix_fmt={} "
      "-> {}x{} pix_fmt={}",
      this->id(), _last_w, _last_h, _last_dec_pix_fmt,
      w, h, dec_pix_fmt));
  } else {
    session()->info(fmt(
      "video-to-rgb('{}'): sws initial geometry: {}x{} pix_fmt={}",
      this->id(), w, h, dec_pix_fmt));
  }
  teardown_sws_();

  int out_w = 0;
  int out_h = 0;
  emitted_dims_(w, h, &out_w, &out_h);
  const CenteredCrop cc = centered_crop_(w, h, out_w, out_h);
  const int src_w = cc.crop_w;
  const int src_h = cc.crop_h;

  // sws_chroma: cropped src @ dec_pix_fmt → YUV444P at out dims.
  // Combines chroma upsample with the geometric resample so we don't
  // need a second sws pass for the scaling; when out == src and no
  // crop is in effect this degenerates to the original same-size
  // chroma upsample.
  _sws_chroma = _libs->swscale().api.get_context(
      src_w, src_h, static_cast<AVPixelFormat>(dec_pix_fmt),
      out_w, out_h, AV_PIX_FMT_YUV444P,
      SWS_BICUBIC, nullptr, nullptr, nullptr);
  if (!_sws_chroma) {
    session()->error(fmt(
      "video-to-rgb('{}'): sws_getContext (chroma) failed",
      this->id()));
    return;
  }
  _sws_rgb = _libs->swscale().api.get_context(
      out_w, out_h, AV_PIX_FMT_YUV444P,
      out_w, out_h, AV_PIX_FMT_GBRP,
      SWS_POINT, nullptr, nullptr, nullptr);
  if (!_sws_rgb) {
    teardown_sws_();
    session()->error(fmt(
      "video-to-rgb('{}'): sws_getContext (rgb) failed",
      this->id()));
    return;
  }

  _yuv444 = _libs->avutil().api.frame_alloc();
  _gbrp   = _libs->avutil().api.frame_alloc();
  if (!_yuv444 || !_gbrp) {
    teardown_sws_();
    session()->error(fmt(
      "video-to-rgb('{}'): frame_alloc failed", this->id()));
    return;
  }
  _yuv444->format = AV_PIX_FMT_YUV444P;
  _yuv444->width  = out_w;
  _yuv444->height = out_h;
  _gbrp->format = AV_PIX_FMT_GBRP;
  _gbrp->width  = out_w;
  _gbrp->height = out_h;
  int rc = _libs->avutil().api.frame_get_buffer(_yuv444, 32);
  if (rc < 0) {
    teardown_sws_();
    session()->error(fmt(
      "video-to-rgb('{}'): frame_get_buffer yuv444: {}",
      this->id(), av_err_(rc)));
    return;
  }
  rc = _libs->avutil().api.frame_get_buffer(_gbrp, 32);
  if (rc < 0) {
    teardown_sws_();
    session()->error(fmt(
      "video-to-rgb('{}'): frame_get_buffer gbrp: {}",
      this->id(), av_err_(rc)));
    return;
  }

  _last_w = w;
  _last_h = h;
  _last_dec_pix_fmt = dec_pix_fmt;
}

enum AVPixelFormat
VideoToRgbStage::get_format_(AVCodecContext* ctx,
                             const enum AVPixelFormat* fmts) noexcept
{
  auto* self = static_cast<VideoToRgbStage*>(ctx->opaque);
  const enum AVPixelFormat want = self ? self->_hw_pix_fmt
                                       : AV_PIX_FMT_NONE;
  for (const enum AVPixelFormat* p = fmts;
       *p != AV_PIX_FMT_NONE; ++p) {
    if (*p == want) {
      return *p;
    }
  }
  // Hardware unavailable for this stream — fall back to the first
  // software fmt offered so decoding still progresses. Mark the
  // HW slot as not-hw-active so receive_frame's downstream path
  // skips the hwframe-transfer step (this callback only ever fires
  // on the HW slot; the SW slot is opened with get_format=nullptr).
  if (self) {
    self->_hw.hw_active = false;
  }
  return fmts[0];
}

namespace {

// Attach a {"timestamp_us": <uint64>, "camera_name": <str>,
// "fps_num": <uint>, "fps_den": <uint>} sideband object to `tb`. Used
// by both the Metal fast path and the CPU swscale path so consumers
// always see the same shape. Empty camera_name omits its field so
// consumers that don't care aren't forced to ignore an empty value;
// the fps pair is emitted only when known (both > 0), so a sink can
// distinguish "source cadence available" from "unknown". The fps comes
// from the source EncodedSegment (rtsp-capture's input stream rate) and
// lets a downstream sink (hls-broadcast) adopt the original frame rate.
void
attach_sideband_(TensorBeat&      tb,
                 std::uint64_t    ts_us,
                 std::string_view camera_name,
                 unsigned         fps_num,
                 unsigned         fps_den)
{
  FlexData o = FlexData::make_object();
  o.as_object().insert_or_assign("timestamp_us",
                                 FlexData::make_uint(ts_us));
  if (!camera_name.empty()) {
    o.as_object().insert_or_assign("camera_name",
                                   FlexData::make_string(camera_name));
  }
  if (fps_num > 0 && fps_den > 0) {
    o.as_object().insert_or_assign("fps_num", FlexData::make_uint(fps_num));
    o.as_object().insert_or_assign("fps_den", FlexData::make_uint(fps_den));
  }
  tb.sideband = std::move(o);
}

}

std::unique_ptr<BeatPayloadIntf>
VideoToRgbStage::frame_to_tensor_beat_(const AVFrame*   src,
                                       std::uint64_t    timestamp_us,
                                       std::string_view camera_name)
{
  int out_w = 0;
  int out_h = 0;
  emitted_dims_(src->width, src->height, &out_w, &out_h);
  const CenteredCrop cc =
      centered_crop_(src->width, src->height, out_w, out_h);

  // The swscale context is built at (cropped_src) → (out dims). Feed
  // it pointers into the crop region of each plane so the source rect
  // lines up with the configured src dims. Plane 0 (Y) advances by
  // crop_top * linesize[0] + crop_left; chroma planes (1, 2 for
  // YUV420P/YUVJ420P, the only formats the SW decode path produces)
  // advance by halved offsets. NV12 hits the Metal fast path; this
  // branch is for sw-decoded yuv-planar input only.
  const uint8_t* src_data[4]    = {nullptr, nullptr, nullptr, nullptr};
  int            src_linesize[4]= {0, 0, 0, 0};
  const int chroma_left = cc.crop_left / 2;
  const int chroma_top  = cc.crop_top  / 2;
  for (int p = 0; p < 4; ++p) {
    src_linesize[p] = src->linesize[p];
    if (!src->data[p]) { continue; }
    const bool is_chroma = (p == 1 || p == 2);
    const int  off_y     = is_chroma ? chroma_top  : cc.crop_top;
    const int  off_x     = is_chroma ? chroma_left : cc.crop_left;
    src_data[p] = src->data[p]
        + static_cast<size_t>(off_y) * src_linesize[p]
        + off_x;
  }

  // Step 1: cropped src → yuv444p at out dims (chroma upsample fused
  // with the geometric resample).
  _libs->swscale().api.scale(
      _sws_chroma,
      const_cast<uint8_t* const*>(src_data),
      src_linesize, 0, cc.crop_h,
      _yuv444->data, _yuv444->linesize);

  // Step 2: yuv444p → gbrp at out dims (point sample; both contexts
  // are sized at out dims so this is a pure pixel-format conversion).
  _libs->swscale().api.scale(
      _sws_rgb,
      _yuv444->data, _yuv444->linesize, 0, out_h,
      _gbrp->data,   _gbrp->linesize);

  // From here on, the TensorBeat geometry is the OUTPUT size, NOT the
  // source's. Bind h / w to the output for the row-copy loops below.
  const int h = out_h;
  const int w = out_w;

  // Pick a uniform per-row pitch P that fits every plane's linesize.
  // GBRP planes typically share linesize but FFmpeg can hand back
  // distinct values.
  int P = _gbrp->linesize[0];
  if (_gbrp->linesize[1] > P) { P = _gbrp->linesize[1]; }
  if (_gbrp->linesize[2] > P) { P = _gbrp->linesize[2]; }

  TensorBeat tb;
  tb.dtype          = _output_dtype;
  tb.shape          = {3, h, w};
  tb.storage_offset = 0;
  const size_t esz        = tb.element_byte_size();
  const size_t row_stride = static_cast<size_t>(P == w ? w : P);
  if (P == w) {
    // No padding -- emit as plain contiguous (strides empty).
    tb.data.assign(static_cast<size_t>(3) * h * w * esz, 0);
  } else {
    tb.strides = {static_cast<int64_t>(h) * P, P, 1};
    tb.data.assign(static_cast<size_t>(3) * h * P * esz, 0);
  }

  // GBRP plane indices: G=0, B=1, R=2. TensorBeat wants R, G, B.
  const int src_plane_for_channel[3] = {2, 0, 1};

  if (_output_dtype == TensorBeat::DType::U8) {
    // Fast path: GBRP bytes go straight into the destination plane
    // (channel-permuted). No float conversion at all.
    uint8_t* dst_base = tb.as_u8();
    for (int c = 0; c < 3; ++c) {
      const int      sp         = src_plane_for_channel[c];
      const uint8_t* src        = _gbrp->data[sp];
      const int      src_stride = _gbrp->linesize[sp];
      uint8_t*       dst_plane  = dst_base
          + static_cast<size_t>(c) * h * row_stride;
      for (int y = 0; y < h; ++y) {
        std::memcpy(dst_plane + static_cast<size_t>(y) * row_stride,
                    src + static_cast<size_t>(y) * src_stride,
                    static_cast<size_t>(w));
        // Columns w..P stay zero-initialised.
      }
    }
  } else {
    float* dst_base = tb.as_f32();
    for (int c = 0; c < 3; ++c) {
      const int      sp         = src_plane_for_channel[c];
      const uint8_t* src        = _gbrp->data[sp];
      const int      src_stride = _gbrp->linesize[sp];
      float*         dst_plane  = dst_base
          + static_cast<size_t>(c) * h * row_stride;
      for (int y = 0; y < h; ++y) {
        const uint8_t* src_row =
            src + static_cast<size_t>(y) * src_stride;
        float* dst_row =
            dst_plane + static_cast<size_t>(y) * row_stride;
        if (_normalize) {
          for (int x = 0; x < w; ++x) {
            dst_row[x] =
                static_cast<float>(src_row[x]) * (1.0f / 255.0f);
          }
        } else {
          for (int x = 0; x < w; ++x) {
            dst_row[x] = static_cast<float>(src_row[x]);
          }
        }
        // Columns w..P stay as the zero-initialised pad.
      }
    }
  }

  attach_sideband_(tb, timestamp_us, camera_name,
                   _last_fps_num, _last_fps_den);
  return make_payload<TensorBeatPayload>(std::move(tb));
}

Job
VideoToRgbStage::try_decode_au_(RuntimeContext& ctx,
                                CodecSlot&      slot,
                                const std::uint8_t* data,
                                std::size_t     len,
                                bool            has_idr,
                                std::uint64_t   timestamp_us,
                                std::string_view camera_name,
                                int*            out_rc,
                                std::size_t*    out_emitted)
{
  *out_rc      = 0;
  *out_emitted = 0;
  if (!slot.ctx) {
    *out_rc = AVERROR(EINVAL);
    co_return;
  }
  // Sync gate: a slot needs an IDR to come out of "unsynced" state.
  if (!slot.synced && !has_idr) {
    *out_rc = AVERROR(EAGAIN);   // "not an error, just skip"
    co_return;
  }
  _libs->avcodec().api.packet_unref(_pkt);
  _pkt->data = const_cast<std::uint8_t*>(data);
  _pkt->size = static_cast<int>(len);

  int rc = _libs->avcodec().api.send_packet(slot.ctx, _pkt);
  if (rc < 0 && rc != AVERROR(EAGAIN)) {
    *out_rc = rc;
    co_return;
  }

  while (true) {
    rc = _libs->avcodec().api.receive_frame(slot.ctx, _yuv_in);
    if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) { break; }
    if (rc < 0) {
      *out_rc = rc;
      co_return;
    }
#ifdef VPIPE_BUILD_APPLE_SILICON
    // Metal fast path: only for HW slot's CVPixelBuffer frames.
    // The output TensorBeat is allocated on a Shared MTLBuffer
    // (storage_class == Shared) so the kernel writes its bytes
    // directly into the consumer-visible buffer -- no readback
    // memcpy from a transient Metal output buffer, and any
    // downstream Metal kernel (e.g. yolo-detection's letterbox)
    // can bind these bytes again without a host round trip. The
    // CPU-cached path stays available as fallback if the Shared
    // allocation fails (e.g. a very large frame exhausting GPU
    // address space), or as a graceful retreat if the kernel
    // dispatch fails for any reason.
    if (slot.hw_active
        && _yuv_in->format == _hw_pix_fmt
        && _output_dtype == TensorBeat::DType::U8
        && _mc && _mc->valid()
        && _yuv_in->data[3] != nullptr) {
      int out_w = 0;
      int out_h = 0;
      emitted_dims_(_yuv_in->width, _yuv_in->height,
                    &out_w, &out_h);
      const size_t need =
          static_cast<size_t>(3) * out_h * out_w;
      TensorBeat tb;
      tb.dtype = TensorBeat::DType::U8;
      tb.shape = {3, out_h, out_w};
      // Prefer Shared MTL::Buffer storage; falls back to CpuCached
      // alloc + readback if Shared allocation or kernel dispatch
      // fails. The single Metal kernel handles NV12 -> centered-
      // crop -> bilinear-rescale -> planar RGB in one compute pass.
      auto shared =
          metal_compute::make_shared_storage(*_mc, need, session());
      bool mok = false;
      if (shared) {
        mok = metal_compute::nv12_to_planar_rgb_u8_shared(
            *_mc,
            reinterpret_cast<void*>(_yuv_in->data[3]),
            *shared,
            _yuv_in->width,
            _yuv_in->height,
            out_w, out_h, session());
        if (mok) {
          tb.external = std::move(shared);
        }
        // On failure `shared` is dropped at scope exit, releasing
        // the MTL::Buffer.
      }
      if (!mok) {
        tb.resize_contiguous(need);
        mok = metal_compute::nv12_to_planar_rgb_u8(
            *_mc,
            reinterpret_cast<void*>(_yuv_in->data[3]),
            tb.as_u8(),
            tb.data.size(),
            _yuv_in->width,
            _yuv_in->height,
            out_w, out_h, session());
      }
      if (mok) {
        if (!_metal_logged) {
          const bool rescale = (out_w != _yuv_in->width
                                || out_h != _yuv_in->height);
          session()->info(fmt(
            "video-to-rgb('{}'): using metal NV12->RGB fast path "
            "(src={}x{} out={}x{} u8{}, storage={})",
            this->id(),
            _yuv_in->width, _yuv_in->height,
            out_w, out_h,
            rescale ? " + center-crop+rescale" : "",
            tb.storage_class() == TensorStorageClass::Shared
              ? "shared" : "cpu"));
          _metal_logged = true;
        }
        attach_sideband_(tb, timestamp_us, camera_name,
                         _last_fps_num, _last_fps_den);
        co_await ctx.write(0,
            make_payload<TensorBeatPayload>(std::move(tb)));
        ++(*out_emitted);
        _libs->avutil().api.frame_unref(_yuv_in);
        continue;
      }
      // Metal failed; fall through to hwframe_transfer + sws.
    }
#endif
    const AVFrame* src = _yuv_in;
    if (slot.hw_active && _yuv_in->format == _hw_pix_fmt) {
      if (!_sw_frame) {
        _sw_frame = _libs->avutil().api.frame_alloc();
      }
      if (_sw_frame) {
        int trc = _libs->avutil().api.hwframe_transfer_data(
            _sw_frame, _yuv_in, 0);
        if (trc < 0) {
          session()->warn(fmt(
            "video-to-rgb('{}'): hwframe_transfer_data: {}",
            this->id(), av_err_(trc)));
          _libs->avutil().api.frame_unref(_yuv_in);
          continue;
        }
        src = _sw_frame;
      }
    }
    ensure_sws_(src->width, src->height, src->format);
    if (_sws_chroma && _sws_rgb) {
      auto tb = frame_to_tensor_beat_(src, timestamp_us, camera_name);
      co_await ctx.write(0, std::move(tb));
      ++(*out_emitted);
    }
    if (src == _sw_frame) {
      _libs->avutil().api.frame_unref(_sw_frame);
    }
    _libs->avutil().api.frame_unref(_yuv_in);
  }

  if (*out_emitted > 0) {
    slot.synced = true;
  }
}

Job
VideoToRgbStage::decode_segment_(RuntimeContext& ctx,
                                 const EncodedSegment& seg)
{
  ensure_codec_hw_(seg);
  if (!_hw.ctx) {
    // Primary slot failed to open at all (e.g. unknown codec_id).
    // ensure_codec_hw_ has already error-logged via session->error
    // which throws, but if we got here the throw was suppressed.
    co_return;
  }

  // Per-frame timestamp basis: the source Beat's start_utc. For
  // rtsp-capture Beats this is the wall-clock time the AU arrived
  // (one Beat = one AU = one frame, so all emitted frames inherit
  // this single value). For the file-extract path each multi-AU
  // Beat shares its containing segment's start_utc; downstream
  // consumers that need per-frame PTS within a multi-AU Beat are
  // out of scope today.
  const std::uint64_t base_ts_us = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          seg.start_utc.time_since_epoch()).count());

  // Latch the segment's camera_name on the stage so flush_decoder_
  // (run at EOS, with no source segment in hand) can still stamp
  // reorder-drain frames with the most recent producer's name. The
  // source frame rate rides along the same way so every emitted frame
  // (including reorder-drain frames) carries it in its sideband.
  _last_camera_name = seg.camera_name;
  _last_fps_num     = seg.fps_num;
  _last_fps_den     = seg.fps_den;

  // Split the concatenated bitstream into per-frame access units.
  // The H.264 decoder takes one AU per send_packet call; feeding a
  // multi-AU blob makes it decode only the first AU. Framing is
  // sniffed from the leading bytes (annex-B vs AVCC).
  const bool annexb = looks_annexb_(seg.data.data(), seg.data.size());
  const auto aus = annexb
      ? split_annexb_aus_(seg.data.data(), seg.data.size())
      : split_avcc_aus_  (seg.data.data(), seg.data.size());
  if (aus.empty()) {
    session()->warn(fmt(
      "video-to-rgb('{}'): segment had no parseable access units "
      "in {} bytes ({} framing)",
      this->id(), seg.data.size(), annexb ? "annex-b" : "avcc"));
    co_return;
  }

  // We deliberately do NOT drain + flush_buffers between Beats; see
  // the comment in the previous revision. flush_buffers fires only
  // when a slot hits a hard decode error (below) or during the
  // pipeline-wide drain() at EOS via flush_decoder_().

  // Per-AU loop: try HW slot first (unless _hw_broken_this_gop is
  // set for this GOP); on hard error fall back to SW slot for the
  // rest of this GOP. _hw_broken_this_gop is cleared at every IDR
  // boundary so HW gets retried on every new GOP — recovering
  // automatically once VT works again.
  size_t emitted_hw     = 0;
  size_t emitted_sw     = 0;
  size_t hw_failures    = 0;
  size_t sw_failures    = 0;
  int    last_hw_rc     = 0;
  int    last_sw_rc     = 0;
  size_t au_idx         = 0;
  for (const auto& [off, len] : aus) {
    const std::uint8_t* p = seg.data.data() + off;
    const bool au_idr = au_contains_idr_(p, len, annexb);
    if (au_idr) {
      // New GOP: clear the HW-broken flag so HW is retried.
      _hw_broken_this_gop = false;
    }

    // Routing: prefer HW unless this GOP has been marked broken.
    // If SW hasn't been opened yet (no prior HW failure ever), we
    // also fall through to HW-only.
    const bool sw_ready  = (_sw.ctx != nullptr);
    const bool prefer_hw = !_hw_broken_this_gop || !sw_ready;
    CodecSlot& first  = prefer_hw ? _hw : _sw;
    CodecSlot& second = prefer_hw ? _sw : _hw;
    const bool first_is_hw  = (&first == &_hw);

    // Sync-gate: we can ONLY attempt this AU if (a) some slot is
    // already synced (can consume a P-slice), or (b) the AU has an
    // IDR (any slot can sync on it). Otherwise drop.
    const bool any_synced = _hw.synced || _sw.synced;
    if (!any_synced && !au_idr) {
      if (_hw_drops_since_desync == 0 && _sw_drops_since_desync == 0) {
        session()->info(fmt(
          "video-to-rgb('{}'): decoder unsynced; dropping non-IDR "
          "AUs until next IDR (codec_id={} {}x{} hw_active={} "
          "hw_broken_this_gop={} db_key={} parent='{}')",
          this->id(), static_cast<int>(seg.codec_id),
          seg.width, seg.height, _hw.hw_active, _hw_broken_this_gop,
          db_key_hex_(seg.db_key), seg.path));
      }
      if (first_is_hw) {
        ++_hw_drops_since_desync;
      } else {
        ++_sw_drops_since_desync;
      }
      ++au_idx;
      continue;
    }

    // First attempt.
    size_t this_emitted = 0;
    int    rc_first     = 0;
    co_await try_decode_au_(ctx, first, p, len, au_idr, base_ts_us,
                            seg.camera_name,
                            &rc_first, &this_emitted);
    if (rc_first == 0 || rc_first == AVERROR(EAGAIN)) {
      // Success path. (EAGAIN = the slot couldn't sync yet but it
      // wasn't an error.) Frames already emitted; carry on.
      if (first_is_hw) emitted_hw += this_emitted;
      else             emitted_sw += this_emitted;
      ++au_idx;
      continue;
    }

    // Hard error on first slot. Capture the FFmpeg log tap's last
    // message (the VideoToolbox / h264 OSStatus that libavcodec's
    // wrapper hides behind AVERROR_UNKNOWN), flush this slot, and
    // try the fallback.
    std::string tap_msg;
    {
      std::lock_guard<std::mutex> lk(_ffmpeg_log_mtx);
      tap_msg = _last_ffmpeg_log;
    }
    _libs->avcodec().api.flush_buffers(first.ctx);
    first.synced = false;
    if (first_is_hw) {
      ++hw_failures;
      last_hw_rc          = rc_first;
      _hw_broken_this_gop = true;
    } else {
      ++sw_failures;
      last_sw_rc = rc_first;
    }
    if (first_is_hw && hw_failures <= 3) {
      session()->warn(fmt(
        "video-to-rgb('{}'): HW send_packet AU#{}/{} @off={} ({}b) "
        "rc={} ({:#x}, {}) framing={} hw_active={} db_key={} "
        "parent='{}' nal_types=[{}] first16=[{}]{}",
        this->id(), au_idx, aus.size(), off, len,
        rc_first, static_cast<unsigned>(rc_first), av_err_(rc_first),
        annexb ? "annex-b" : "avcc", _hw.hw_active,
        db_key_hex_(seg.db_key), seg.path,
        au_nal_summary_(p, len, annexb),
        hex_prefix_(p, len, 16),
        tap_msg.empty() ? "" :
          fmt(" ffmpeg=[{}]", tap_msg)()));
    } else if (!first_is_hw && sw_failures <= 3) {
      session()->warn(fmt(
        "video-to-rgb('{}'): SW send_packet AU#{}/{} @off={} ({}b) "
        "rc={} ({:#x}, {}) framing={} db_key={} parent='{}' "
        "nal_types=[{}] first16=[{}]{}",
        this->id(), au_idx, aus.size(), off, len,
        rc_first, static_cast<unsigned>(rc_first), av_err_(rc_first),
        annexb ? "annex-b" : "avcc",
        db_key_hex_(seg.db_key), seg.path,
        au_nal_summary_(p, len, annexb),
        hex_prefix_(p, len, 16),
        tap_msg.empty() ? "" :
          fmt(" ffmpeg=[{}]", tap_msg)()));
    }

    // Lazy SW open if HW failed and SW slot isn't here yet.
    if (first_is_hw && !_sw.ctx) {
      ensure_codec_sw_(seg);
    }

    // Fallback attempt on the other slot.
    if (second.ctx && (second.synced || au_idr)) {
      size_t fallback_emitted = 0;
      int    rc_second        = 0;
      co_await try_decode_au_(ctx, second, p, len, au_idr, base_ts_us,
                              seg.camera_name,
                              &rc_second, &fallback_emitted);
      if (rc_second == 0 || rc_second == AVERROR(EAGAIN)) {
        if (&second == &_hw) emitted_hw += fallback_emitted;
        else                 emitted_sw += fallback_emitted;
        ++au_idx;
        continue;
      }
      _libs->avcodec().api.flush_buffers(second.ctx);
      second.synced = false;
      if (&second == &_hw) {
        ++hw_failures;
        last_hw_rc          = rc_second;
        _hw_broken_this_gop = true;
      } else {
        ++sw_failures;
        last_sw_rc = rc_second;
      }
    }
    ++au_idx;
  }

  // Per-segment summary log. Three branches:
  //  (a) any failures: WARN with both slot statuses + tap context.
  //  (b) first frame from a previously-unsynced slot: INFO recovery.
  //  (c) multi-AU success: existing INFO (file-extract / test path).
  const size_t total_emitted = emitted_hw + emitted_sw;
  if (hw_failures > 0 || sw_failures > 0) {
    std::string tap_msg;
    {
      std::lock_guard<std::mutex> lk(_ffmpeg_log_mtx);
      tap_msg = _last_ffmpeg_log;
    }
    session()->warn(fmt(
      "video-to-rgb('{}'): segment finished with errors: "
      "hw_failures={} sw_failures={} emitted_hw={} emitted_sw={} "
      "(total AUs={}); bytes={} codec_id={} {}x{} framing={} "
      "extradata={}b hw_broken_this_gop={} db_key={} parent='{}' "
      "last_hw_rc={:#x} last_sw_rc={:#x} ffmpeg=[{}]. "
      "Flushed failed slot(s); HW will retry at next IDR.",
      this->id(), hw_failures, sw_failures,
      emitted_hw, emitted_sw, aus.size(),
      seg.data.size(), static_cast<int>(seg.codec_id),
      seg.width, seg.height, annexb ? "annex-b" : "avcc",
      seg.extradata.size(), _hw_broken_this_gop,
      db_key_hex_(seg.db_key), seg.path,
      static_cast<unsigned>(last_hw_rc),
      static_cast<unsigned>(last_sw_rc),
      tap_msg));
  }
  // Per-slot resync INFO.
  if (emitted_hw > 0 && _hw_drops_since_desync > 0) {
    session()->info(fmt(
      "video-to-rgb('{}'): HW slot synced on IDR ({} AU(s) dropped); "
      "codec_id={} {}x{} db_key={}",
      this->id(), _hw_drops_since_desync,
      static_cast<int>(seg.codec_id), seg.width, seg.height,
      db_key_hex_(seg.db_key)));
    _hw_drops_since_desync = 0;
  }
  if (emitted_sw > 0 && _sw_drops_since_desync > 0) {
    session()->info(fmt(
      "video-to-rgb('{}'): SW slot synced on IDR ({} AU(s) dropped); "
      "codec_id={} {}x{} db_key={}",
      this->id(), _sw_drops_since_desync,
      static_cast<int>(seg.codec_id), seg.width, seg.height,
      db_key_hex_(seg.db_key)));
    _sw_drops_since_desync = 0;
  }
  // File-extract path INFO (multi-AU Beat).
  if (hw_failures == 0 && sw_failures == 0 && aus.size() > 1) {
    session()->info(fmt(
        "video-to-rgb('{}'): decoded segment ({} bytes, codec_id={}, "
        "{} framing, {} AU(s)) -> {} frame(s)",
        this->id(), seg.data.size(),
        static_cast<int>(seg.codec_id),
        annexb ? "annex-b" : "avcc",
        aus.size(), total_emitted));
  }
}

Job
VideoToRgbStage::flush_decoder_(RuntimeContext& ctx)
{
  // Drain both slots at pipeline EOS. Either may have B-frame
  // reorder buffers pending; we want to emit whatever frames the
  // underlying decoder still holds before the stage tears down.
  //
  // Stop-requested fast path: PipelineRuntime::stop() closes every
  // oport before joining drivers, so a co_await ctx.write here
  // would throw "write to closed buffer" and surface as a
  // confusing "stage drain: ; continuing" warning. There is no
  // consumer left to deliver the held frames to either way, so
  // skip the flush entirely on stop. (Natural upstream EOS still
  // falls through and flushes B-frame reorder buffers normally.)
  if (ctx.stop_requested()) {
    co_return;
  }
  for (CodecSlot* slot_ptr : {&_hw, &_sw}) {
    CodecSlot& slot = *slot_ptr;
    if (!slot.ctx) { continue; }
    int rc = _libs->avcodec().api.send_packet(slot.ctx, nullptr);
    if (rc < 0 && rc != AVERROR(EAGAIN)) {
      session()->warn(fmt(
        "video-to-rgb('{}'): flush send_packet ({}): {}",
        this->id(), slot.hw_active ? "hw" : "sw", av_err_(rc)));
      continue;
    }
    while (!ctx.stop_requested()) {
      rc = _libs->avcodec().api.receive_frame(slot.ctx, _yuv_in);
      if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) { break; }
      if (rc < 0) { break; }
      const AVFrame* src = _yuv_in;
      if (slot.hw_active && _yuv_in->format == _hw_pix_fmt) {
        if (!_sw_frame) {
          _sw_frame = _libs->avutil().api.frame_alloc();
        }
        if (_sw_frame) {
          int trc = _libs->avutil().api.hwframe_transfer_data(
              _sw_frame, _yuv_in, 0);
          if (trc < 0) {
            _libs->avutil().api.frame_unref(_yuv_in);
            continue;
          }
          src = _sw_frame;
        }
      }
      ensure_sws_(src->width, src->height, src->format);
      if (_sws_chroma && _sws_rgb) {
        // No source AU for B-frame reorder buffers drained at EOS;
        // stamp them with wall-clock now so consumers still see a
        // monotonic-ish timestamp_us rather than zero.
        const std::uint64_t now_us = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch())
              .count());
        auto tb = frame_to_tensor_beat_(src, now_us,
                                        _last_camera_name);
        co_await ctx.write(0, std::move(tb));
      }
      if (src == _sw_frame) {
        _libs->avutil().api.frame_unref(_sw_frame);
      }
      _libs->avutil().api.frame_unref(_yuv_in);
    }
  }
}

Job
VideoToRgbStage::process(RuntimeContext& ctx)
{
  auto beat_opt = co_await ctx.read(0);
  if (!beat_opt) {
    ctx.signal_done();
    co_return;
  }
  const auto* seg = dynamic_cast<const EncodedSegmentPayload*>(beat_opt.get());
  if (!seg || seg->kind != EncodedSegment::Kind::Video) {
    session()->warn(fmt(
      "video-to-rgb('{}'): ignoring non-video / wrong-type beat",
      this->id()));
    co_return;
  }
  if (seg->data.empty()) {
    co_return;
  }
  co_await decode_segment_(ctx, *seg);
}

Job
VideoToRgbStage::drain(RuntimeContext& ctx)
{
  co_await flush_decoder_(ctx);
}

VPIPE_REGISTER_STAGE(VideoToRgbStage)
VPIPE_REGISTER_SPEC(VideoToRgbStage, kSpec)

}

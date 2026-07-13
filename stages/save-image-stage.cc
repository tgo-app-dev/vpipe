#include "stages/save-image-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/ffmpeg-libraries.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/dict.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
}

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace vpipe {

namespace {

// One small unique_ptr deleter per FFmpeg resource, each carrying the
// FFmpegLibraries* so destruction routes through the right api table.
struct CctxDeleter {
  const FFmpegLibraries* libs;
  void
  operator()(AVCodecContext* p) const noexcept
  {
    if (p) { libs->avcodec().api.free_context(&p); }
  }
};
using CctxPtr = unique_ptr<AVCodecContext, CctxDeleter>;

struct PktDeleter {
  const FFmpegLibraries* libs;
  void
  operator()(AVPacket* p) const noexcept
  {
    if (p) { libs->avcodec().api.packet_free(&p); }
  }
};
using PktPtr = unique_ptr<AVPacket, PktDeleter>;

struct FrameDeleter {
  const FFmpegLibraries* libs;
  void
  operator()(AVFrame* p) const noexcept
  {
    if (p) { libs->avutil().api.frame_free(&p); }
  }
};
using FramePtr = unique_ptr<AVFrame, FrameDeleter>;

struct SwsDeleter {
  const FFmpegLibraries* libs;
  void
  operator()(SwsContext* p) const noexcept
  {
    if (p) { libs->swscale().api.free_context(p); }
  }
};
using SwsPtr = unique_ptr<SwsContext, SwsDeleter>;

// One supported output format: its config key, the FFmpeg encoder name,
// whether it is lossy (quality-controlled), and the preferred encode
// pixel format (validated against the encoder's list at open time).
struct FormatDef {
  const char*   key;
  const char*   encoder;
  bool          lossy;
  AVPixelFormat pref_pix_fmt;
};
const FormatDef kFormats[] = {
  {"png",  "png",   false, AV_PIX_FMT_RGB24},
  {"jpeg", "mjpeg", true,  AV_PIX_FMT_YUVJ444P},
  {"jpg",  "mjpeg", true,  AV_PIX_FMT_YUVJ444P},
  {"webp", "webp",  true,  AV_PIX_FMT_YUV420P},
  {"bmp",  "bmp",   false, AV_PIX_FMT_BGR24},
  {"tiff", "tiff",  false, AV_PIX_FMT_RGB24},
  {"tif",  "tiff",  false, AV_PIX_FMT_RGB24},
};

const FormatDef*
lookup_format(const string& key)
{
  for (const auto& f : kFormats) {
    if (key == f.key) { return &f; }
  }
  return nullptr;
}

string
lower_ext(const string& path)
{
  const auto dot = path.find_last_of('.');
  if (dot == string::npos) { return {}; }
  string ext = path.substr(dot + 1);
  for (char& c : ext) { c = (char)tolower((unsigned char)c); }
  return ext;
}

// Detect a single printf integer conversion ("%d", "%04d", "%i", ...) so
// a path template can index a stream of images. Returns true when exactly
// one such conversion is present (a literal "%%" is not a conversion).
bool
has_int_conversion(const string& s)
{
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] != '%') { continue; }
    if (i + 1 < s.size() && s[i + 1] == '%') { ++i; continue; }   // %%
    size_t j = i + 1;
    while (j < s.size() &&
           (s[j] == '-' || s[j] == '+' || s[j] == ' ' ||
            s[j] == '#' || s[j] == '0')) { ++j; }                 // flags
    while (j < s.size() && isdigit((unsigned char)s[j])) { ++j; } // width
    if (j < s.size() &&
        (s[j] == 'd' || s[j] == 'i' || s[j] == 'u' ||
         s[j] == 'x' || s[j] == 'X')) {
      return true;
    }
  }
  return false;
}

}   // namespace

SaveImageStage::SaveImageStage(const SessionContextIntf* s,
                                 string                    id,
                                 vector<InEdge>            iports,
                                 FlexData                  config)
  : TypedStage<SaveImageStage>(s, std::move(id), std::move(iports),
                                std::move(config))
  , _libs(s ? s->ffmpeg_libraries() : nullptr)
{
  // Deferred validation (see Stage::fail_config): construct for any config
  // so a graph can be built/edited before `path` is supplied.
  _path        = attr_path("path", true);
  _format      = attr_str("format");
  _quality     = (int)attr_int("quality");
  _compression = (int)attr_int("compression");
  _lossless    = attr_bool("lossless");
  if (_quality <= 0)     { _quality = 90; }
  if (_compression <= 0) { _compression = 6; }
  _quality     = std::clamp(_quality, 1, 100);
  _compression = std::clamp(_compression, 0, 9);

  if (_path.empty()) {
    fail_config(fmt(
        "SaveImageStage('{}'): config.path is required (the output image "
        "file path)", this->id()));
  }
  // Resolve the format: explicit config wins, else the path extension,
  // else png. Lower-case + validate against the supported set.
  string key = _format;
  for (char& c : key) { c = (char)tolower((unsigned char)c); }
  if (key.empty()) { key = lower_ext(_path); }
  if (key.empty()) { key = "png"; }
  const FormatDef* fd = lookup_format(key);
  if (fd == nullptr) {
    fail_config(fmt(
        "SaveImageStage('{}'): unsupported format '{}' (want one of "
        "png, jpeg, webp, bmp, tiff)", this->id(), key));
  } else {
    _format = fd->key;
  }
  if (!_libs || !_libs->valid()) {
    fail_config(fmt(
        "SaveImageStage('{}'): FFmpegLibraries unavailable", this->id()));
  }
}

SaveImageStage::~SaveImageStage() = default;

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "path", .type = ConfigType::String, .required = true,
   .doc = "output image file path; a printf integer conversion "
          "(e.g. frame-%04d.png) indexes a stream of images, else later "
          "images get a -NNNNNN suffix",
   .is_path = true, .path_write = true, .path_filter = "image"},
  {.key = "format", .type = ConfigType::String, .required = false,
   .doc = "png | jpeg (jpg) | webp | bmp | tiff; default from the path "
          "extension, else png"},
  {.key = "quality", .type = ConfigType::Int, .required = false,
   .doc = "lossy codecs (jpeg, lossy webp): 1..100, higher is better "
          "(default 90)"},
  {.key = "compression", .type = ConfigType::Int, .required = false,
   .doc = "PNG zlib level 0..9, higher is smaller + slower (default 6)"},
  {.key = "lossless", .type = ConfigType::Bool, .required = false,
   .doc = "webp lossless mode (default false)"},
};
const PortSpec kIports[] = {
  {.name = "image", .doc = "planar U8 RGB TensorBeat [3,H,W] (load-image / "
                           "vae-decode format)",
   .type = &typeid(TensorBeatPayload),
   .tags = "rgb-frames", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "save-image",
  .doc       = "Sink: encodes each planar U8 RGB TensorBeat [3,H,W] to an "
               "image file (PNG/JPEG/WebP/BMP/TIFF) via FFmpeg. The inverse "
               "of load-image; quality/compression knobs control the store.",
  .display_name = "Save Image",
  .category  = StageCategory::Visual,
  .iports    = kIports,
  .oports    = {},
  .attrs     = kAttrs,
};
}   // namespace

const StageSpec&
SaveImageStage::spec() const noexcept
{
  return kSpec;
}

string
SaveImageStage::av_err_(int rc) const
{
  char buf[256];
  _libs->avutil().api.strerror(rc, buf, sizeof buf);
  return string(buf);
}

string
SaveImageStage::resolve_path_(uint64_t index) const
{
  // A printf integer conversion gives the caller full control over the
  // per-image name.
  if (has_int_conversion(_path)) {
    char buf[2048];
    std::snprintf(buf, sizeof buf, _path.c_str(),
                  (int)(unsigned)index);
    return string(buf);
  }
  // No token: the first image writes to `path` verbatim; each later image
  // gets a zero-padded suffix before the extension so a stream does not
  // clobber itself.
  if (index == 0) { return _path; }
  char suf[16];
  std::snprintf(suf, sizeof suf, "-%06u", (unsigned)index);
  const auto dot = _path.find_last_of('.');
  const auto slash = _path.find_last_of("/\\");
  if (dot != string::npos && (slash == string::npos || dot > slash)) {
    return _path.substr(0, dot) + suf + _path.substr(dot);
  }
  return _path + suf;
}

bool
SaveImageStage::encode_(const BeatPayloadIntf& beat, const string& out_path)
{
  const auto* tbp = dynamic_cast<const TensorBeatPayload*>(&beat);
  if (tbp == nullptr || tbp->dtype != TensorBeat::DType::U8 ||
      tbp->shape.size() != 3 || tbp->shape[0] != 3) {
    session()->warn(fmt(
        "SaveImageStage('{}'): expected a U8 [3,H,W] RGB TensorBeat, got "
        "{}; dropping beat", this->id(), beat.describe()));
    return false;
  }
  const int H = (int)tbp->shape[1];
  const int W = (int)tbp->shape[2];
  if (H <= 0 || W <= 0) {
    session()->warn(fmt(
        "SaveImageStage('{}'): invalid image dimensions {}x{}; dropping "
        "beat", this->id(), W, H));
    return false;
  }

  const FormatDef* fd = lookup_format(_format);
  if (fd == nullptr) { return false; }   // ctor validated; defensive.

  const AVCodec* codec =
      _libs->avcodec().api.find_encoder_by_name(fd->encoder);
  if (codec == nullptr) {
    session()->error(fmt(
        "SaveImageStage('{}'): FFmpeg has no '{}' encoder; cannot write "
        "'{}'", this->id(), fd->encoder, out_path));
    return false;
  }

  // Encode pixel format: the per-codec preferred one (open2 gives a clear
  // error if the encoder rejects it). Lossless webp needs a full RGB
  // format -- chroma-subsampled YUV would not round-trip -- so promote it.
  AVPixelFormat pix = fd->pref_pix_fmt;
  if (_format == "webp" && _lossless) { pix = AV_PIX_FMT_BGRA; }

  AVCodecContext* raw_cctx = _libs->avcodec().api.alloc_context3(codec);
  if (raw_cctx == nullptr) {
    session()->error(fmt(
        "SaveImageStage('{}'): alloc_context3 failed", this->id()));
    return false;
  }
  CctxPtr cctx(raw_cctx, CctxDeleter{_libs});
  cctx->width     = W;
  cctx->height    = H;
  cctx->pix_fmt   = pix;
  cctx->time_base = AVRational{1, 1};   // stills; must be non-zero.
  cctx->compression_level = _compression;   // honoured by png (+ others).

  AVDictionary* opts = nullptr;
  if (fd->lossy) {
    // Fixed-quality (qscale) mode. mjpeg qscale is 1 (best) .. 31 (worst);
    // map quality 100->1, 1->31 linearly.
    const int qscale = std::clamp(
        (int)std::lround((100.0 - _quality) * 30.0 / 99.0) + 1, 1, 31);
    cctx->flags |= AV_CODEC_FLAG_QSCALE;
    cctx->global_quality = qscale * FF_QP2LAMBDA;
    // libwebp reads its own 0..100 `quality` + `lossless` private options.
    _libs->avutil().api.dict_set(
        &opts, "quality", std::to_string(_quality).c_str(), 0);
    if (_lossless) {
      _libs->avutil().api.dict_set(&opts, "lossless", "1", 0);
    }
  }

  int rc = _libs->avcodec().api.open2(cctx.get(), codec, &opts);
  _libs->avutil().api.dict_free(&opts);
  if (rc < 0) {
    session()->error(fmt(
        "SaveImageStage('{}'): avcodec_open2('{}') failed: {}",
        this->id(), fd->encoder, av_err_(rc)));
    return false;
  }

  FramePtr frame(_libs->avutil().api.frame_alloc(), FrameDeleter{_libs});
  if (!frame) { return false; }
  frame->format = pix;
  frame->width  = W;
  frame->height = H;
  rc = _libs->avutil().api.frame_get_buffer(frame.get(), 32);
  if (rc < 0) {
    session()->error(fmt(
        "SaveImageStage('{}'): frame_get_buffer failed: {}",
        this->id(), av_err_(rc)));
    return false;
  }
  if (fd->lossy) { frame->quality = cctx->global_quality; }

  // Our tensor is planar R,G,B [3,H,W]; feed swscale as GBRP (plane order
  // G,B,R). Handle both contiguous (strides empty -> row pitch W) and the
  // padded 3-stride form load-image can hand back.
  const uint8_t* base = tbp->as_u8();
  const int64_t plane_stride =
      tbp->strides.empty() ? (int64_t)H * W : tbp->strides[0];
  const int row_pitch =
      tbp->strides.empty() ? W : (int)tbp->strides[1];
  const uint8_t* plane_r = base + 0 * plane_stride;
  const uint8_t* plane_g = base + 1 * plane_stride;
  const uint8_t* plane_b = base + 2 * plane_stride;
  const uint8_t* src_data[4]   = {plane_g, plane_b, plane_r, nullptr};
  int            src_pitch[4]  = {row_pitch, row_pitch, row_pitch, 0};

  SwsPtr sws(_libs->swscale().api.get_context(
                 W, H, AV_PIX_FMT_GBRP, W, H, pix,
                 SWS_BILINEAR, nullptr, nullptr, nullptr),
             SwsDeleter{_libs});
  if (!sws) {
    session()->error(fmt(
        "SaveImageStage('{}'): sws_getContext (gbrp->{}) failed",
        this->id(), (int)pix));
    return false;
  }
  _libs->swscale().api.scale(sws.get(), src_data, src_pitch, 0, H,
                             frame->data, frame->linesize);

  PktPtr pkt(_libs->avcodec().api.packet_alloc(), PktDeleter{_libs});
  if (!pkt) { return false; }

  rc = _libs->avcodec().api.send_frame(cctx.get(), frame.get());
  if (rc < 0) {
    session()->error(fmt(
        "SaveImageStage('{}'): send_frame failed: {}",
        this->id(), av_err_(rc)));
    return false;
  }
  // Flush: still-image encoders emit one packet after an EOF signal.
  _libs->avcodec().api.send_frame(cctx.get(), nullptr);

  namespace fs = std::filesystem;
  std::error_code ec;
  const fs::path parent = fs::path(out_path).parent_path();
  if (!parent.empty()) { fs::create_directories(parent, ec); }
  std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    session()->error(fmt(
        "SaveImageStage('{}'): cannot open '{}' for writing",
        this->id(), out_path));
    return false;
  }
  bool wrote = false;
  for (;;) {
    rc = _libs->avcodec().api.receive_packet(cctx.get(), pkt.get());
    if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) { break; }
    if (rc < 0) {
      session()->error(fmt(
          "SaveImageStage('{}'): receive_packet failed: {}",
          this->id(), av_err_(rc)));
      _libs->avcodec().api.packet_unref(pkt.get());
      return false;
    }
    out.write(reinterpret_cast<const char*>(pkt->data), pkt->size);
    wrote = true;
    _libs->avcodec().api.packet_unref(pkt.get());
  }
  out.flush();
  if (!wrote || !out.good()) {
    session()->error(fmt(
        "SaveImageStage('{}'): no encoded data written to '{}'",
        this->id(), out_path));
    return false;
  }
  return true;
}

Job
SaveImageStage::initialize(RuntimeContext& /*ctx*/)
{
  if (_path.empty() || !_libs || !_libs->valid()) { co_return; }
  session()->info(fmt(
      "SaveImageStage('{}'): writing {} to '{}' (quality {}, compression "
      "{}{})", this->id(), _format, _path, _quality, _compression,
      _lossless ? ", lossless" : ""));
  co_return;
}

Job
SaveImageStage::process(RuntimeContext& ctx)
{
  auto in = co_await ctx.read(0);
  if (!in) { ctx.signal_done(); co_return; }   // upstream EOS -> end.
  if (_path.empty() || !_libs || !_libs->valid()) { co_return; }   // inert.

  const string out_path = resolve_path_(_seen++);
  if (encode_(*in, out_path)) {
    ++_written;
    session()->log_verbose(fmt(
        "SaveImageStage('{}'): wrote '{}'", this->id(), out_path));
  }
  // Sink: no ctx.write.
}

VPIPE_REGISTER_STAGE(SaveImageStage)
VPIPE_REGISTER_SPEC(SaveImageStage, kSpec)

}

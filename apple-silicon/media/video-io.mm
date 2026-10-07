#include "apple-silicon/media/video-io.h"

#include "apple-silicon/media/planar.h"

#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <VideoToolbox/VideoToolbox.h>

#include <chrono>
#include <cmath>
#include <numeric>
#include <thread>

namespace vpipe::apple_media {

namespace {

// ---- CICP <-> Core Video's colour strings --------------------------------

struct CodePoint {
  CFStringRef cv;
  int         cicp;
};

const CodePoint kPrimaries[] = {
  {kCVImageBufferColorPrimaries_ITU_R_709_2, 1},
  {kCVImageBufferColorPrimaries_ITU_R_2020, 9},
  {kCVImageBufferColorPrimaries_P3_D65, 12},
  {kCVImageBufferColorPrimaries_DCI_P3, 11},
  {kCVImageBufferColorPrimaries_SMPTE_C, 6},
  {kCVImageBufferColorPrimaries_EBU_3213, 5},
};
const CodePoint kTransfers[] = {
  {kCVImageBufferTransferFunction_ITU_R_709_2, 1},
  {kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ, 16},
  {kCVImageBufferTransferFunction_ITU_R_2100_HLG, 18},
  {kCVImageBufferTransferFunction_Linear, 8},
  {kCVImageBufferTransferFunction_sRGB, 13},
  {kCVImageBufferTransferFunction_ITU_R_2020, 14},
  {kCVImageBufferTransferFunction_SMPTE_240M_1995, 7},
};
const CodePoint kMatrices[] = {
  {kCVImageBufferYCbCrMatrix_ITU_R_709_2, 1},
  {kCVImageBufferYCbCrMatrix_ITU_R_601_4, 6},
  {kCVImageBufferYCbCrMatrix_ITU_R_2020, 9},
  {kCVImageBufferYCbCrMatrix_SMPTE_240M_1995, 7},
};

template <std::size_t N>
int
cicp_of(CFTypeRef v, const CodePoint (&table)[N])
{
  if (!v || CFGetTypeID(v) != CFStringGetTypeID()) { return 2; }
  for (const auto& e : table) {
    if (CFEqual(v, e.cv)) { return e.cicp; }
  }
  return 2;
}

template <std::size_t N>
CFStringRef
cv_of(int cicp, const CodePoint (&table)[N], CFStringRef fallback)
{
  for (const auto& e : table) {
    if (e.cicp == cicp) { return e.cv; }
  }
  return fallback;
}

// Colour of a format description or a pixel buffer's attachments.
void
tags_from(CFTypeRef (*get)(const void*, CFStringRef), const void* obj,
          ColorTags* t)
{
  const int p = cicp_of(get(obj, kCVImageBufferColorPrimariesKey),
                        kPrimaries);
  const int tr = cicp_of(get(obj, kCVImageBufferTransferFunctionKey),
                         kTransfers);
  const int m = cicp_of(get(obj, kCVImageBufferYCbCrMatrixKey), kMatrices);
  if (p != 2) { t->primaries = p; }
  if (tr != 2) { t->transfer = tr; }
  if (m != 2) { t->matrix = m; }
}

CFTypeRef
fd_get(const void* fd, CFStringRef key)
{
  return CMFormatDescriptionGetExtension((CMFormatDescriptionRef)fd, key);
}

CFTypeRef
attachment_get(const void* buf, CFStringRef key)
{
  // A borrowed reference: the buffer keeps it alive while it is used.
  CFTypeRef v = CVBufferCopyAttachment((CVBufferRef)buf, key, nullptr);
  if (v) { CFAutorelease(v); }
  return v;
}

std::uint16_t
be16(const std::uint8_t* p)
{
  return (std::uint16_t)((p[0] << 8) | p[1]);
}

std::uint32_t
be32(const std::uint8_t* p)
{
  return ((std::uint32_t)p[0] << 24) | ((std::uint32_t)p[1] << 16) |
         ((std::uint32_t)p[2] << 8) | p[3];
}

void
put16(std::uint8_t* p, double v)
{
  const auto u = (std::uint16_t)std::lround(v);
  p[0] = (std::uint8_t)(u >> 8);
  p[1] = (std::uint8_t)u;
}

void
put32(std::uint8_t* p, double v)
{
  const auto u = (std::uint32_t)std::llround(v);
  p[0] = (std::uint8_t)(u >> 24);
  p[1] = (std::uint8_t)(u >> 16);
  p[2] = (std::uint8_t)(u >> 8);
  p[3] = (std::uint8_t)u;
}

// ST 2086 as the bitstream holds it (24 bytes, big-endian, primaries in
// SEI order G, B, R) <-> the sideband's R, G, B, white, max, min.
std::vector<double>
mastering_from(CFTypeRef v)
{
  std::vector<double> out;
  if (!v || CFGetTypeID(v) != CFDataGetTypeID() ||
      CFDataGetLength((CFDataRef)v) < 24) {
    return out;
  }
  const std::uint8_t* b = CFDataGetBytePtr((CFDataRef)v);
  const int sei[3] = {2, 0, 1};  // R is the third, G the first
  for (int c = 0; c < 3; ++c) {
    out.push_back(be16(b + sei[c] * 4));
    out.push_back(be16(b + sei[c] * 4 + 2));
  }
  out.push_back(be16(b + 12));
  out.push_back(be16(b + 14));
  out.push_back(be32(b + 16));
  out.push_back(be32(b + 20));
  return out;
}

NSData*
mastering_data(const std::vector<double>& m)
{
  std::uint8_t b[24] = {};
  const int sei[3] = {1, 2, 0};  // G, B, R from R, G, B
  for (int k = 0; k < 3; ++k) {
    put16(b + k * 4, m[sei[k] * 2]);
    put16(b + k * 4 + 2, m[sei[k] * 2 + 1]);
  }
  put16(b + 12, m[6]);
  put16(b + 14, m[7]);
  put32(b + 16, m[8]);
  put32(b + 20, m[9]);
  return [NSData dataWithBytes:b length:24];
}

std::vector<double>
content_light_from(CFTypeRef v)
{
  if (!v || CFGetTypeID(v) != CFDataGetTypeID() ||
      CFDataGetLength((CFDataRef)v) < 4) {
    return {};
  }
  const std::uint8_t* b = CFDataGetBytePtr((CFDataRef)v);
  return {(double)be16(b), (double)be16(b + 2)};
}

NSData*
content_light_data(const std::vector<double>& c)
{
  std::uint8_t b[4] = {};
  put16(b, c[0]);
  put16(b + 2, c[1]);
  return [NSData dataWithBytes:b length:4];
}

std::string
fourcc(FourCharCode c)
{
  std::string s(4, ' ');
  for (int i = 0; i < 4; ++i) { s[i] = (char)((c >> (24 - 8 * i)) & 0xff); }
  return s;
}

std::string
describe(NSError* e)
{
  return e ? std::string(e.localizedDescription.UTF8String) : "unknown";
}

void
load_keys(id<AVAsynchronousKeyValueLoading> obj, NSArray<NSString*>* keys)
{
  dispatch_semaphore_t sem = dispatch_semaphore_create(0);
  [obj loadValuesAsynchronouslyForKeys:keys
                     completionHandler:^{
                       dispatch_semaphore_signal(sem);
                     }];
  dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
}

bool
is_prores(const std::string& codec)
{
  return codec.rfind("prores", 0) == 0;
}

}  // namespace

void
fps_fraction(double fps, int* num, int* den)
{
  struct Ntsc {
    double fps;
    int    num, den;
  };
  static const Ntsc k[] = {{23.976, 24000, 1001}, {29.97, 30000, 1001},
                           {47.952, 48000, 1001}, {59.94, 60000, 1001},
                           {119.88, 120000, 1001}};
  for (const auto& e : k) {
    if (std::fabs(fps - e.fps) < 0.005) {
      *num = e.num;
      *den = e.den;
      return;
    }
  }
  *num = (int)std::lround(fps * 1000.0);
  *den = 1000;
  const int g = std::gcd(*num, *den);
  if (g > 1) {
    *num /= g;
    *den /= g;
  }
  if (*num <= 0) {
    *num = 24;
    *den = 1;
  }
}

// ---- reading ---------------------------------------------------------------

struct VideoReader::Impl {
  AVAssetReader*            reader = nil;
  AVAssetReaderTrackOutput* output = nil;
  VideoInfo                 info;
  VideoReadOptions          opt;
};

VideoReader::VideoReader() : _impl(std::make_unique<Impl>()) {}
VideoReader::~VideoReader()
{
  if (_impl->reader &&
      _impl->reader.status == AVAssetReaderStatusReading) {
    [_impl->reader cancelReading];
  }
}

const VideoInfo&
VideoReader::info() const
{
  return _impl->info;
}

bool
VideoReader::open(const std::string& path, const VideoReadOptions& opt,
                  std::string* err)
{
  @autoreleasepool {
    _impl->opt = opt;
    NSURL* url = [NSURL fileURLWithPath:@(path.c_str())];
    AVURLAsset* asset = [AVURLAsset URLAssetWithURL:url options:@{
      AVURLAssetPreferPreciseDurationAndTimingKey : @YES
    }];
    __block NSArray<AVAssetTrack*>* tracks = nil;
    __block NSError* terr = nil;
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    [asset loadTracksWithMediaType:AVMediaTypeVideo
                 completionHandler:^(NSArray<AVAssetTrack*>* t, NSError* e) {
                   tracks = t;
                   terr = e;
                   dispatch_semaphore_signal(sem);
                 }];
    dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
    AVAssetTrack* track = tracks.firstObject;
    if (!track) {
      if (err) {
        *err = "no video track in '" + path + "': " + describe(terr);
      }
      return false;
    }
    load_keys(track, @[ @"formatDescriptions", @"nominalFrameRate",
                        @"naturalSize", @"timeRange", @"minFrameDuration" ]);
    VideoInfo& vi = _impl->info;
    const CGSize sz = track.naturalSize;
    vi.width = (int)sz.width;
    vi.height = (int)sz.height;
    const CMTime mfd = track.minFrameDuration;
    if (CMTIME_IS_VALID(mfd) && mfd.value > 0) {
      const int g = std::gcd((int)mfd.timescale, (int)mfd.value);
      vi.fps_num = (int)mfd.timescale / g;
      vi.fps_den = (int)mfd.value / g;
    } else {
      fps_fraction(track.nominalFrameRate, &vi.fps_num, &vi.fps_den);
    }
    vi.duration_s = CMTimeGetSeconds(track.timeRange.duration);
    vi.frames = (std::int64_t)std::llround(vi.duration_s * vi.fps_num /
                                           vi.fps_den);
    CMFormatDescriptionRef fd = (__bridge CMFormatDescriptionRef)
        track.formatDescriptions.firstObject;
    if (fd) {
      const FourCharCode sub = CMFormatDescriptionGetMediaSubType(fd);
      vi.codec = fourcc(sub);
      tags_from(fd_get, fd, &vi.color);
      CFTypeRef fr = fd_get(fd, kCMFormatDescriptionExtension_FullRangeVideo);
      vi.color.full_range = fr && CFBooleanGetValue((CFBooleanRef)fr);
      int bits = 8;
      if (CFTypeRef b = fd_get(fd,
              kCMFormatDescriptionExtension_BitsPerComponent)) {
        CFNumberGetValue((CFNumberRef)b, kCFNumberIntType, &bits);
      }
      const bool p4444 = sub == kCMVideoCodecType_AppleProRes4444 ||
                         sub == kCMVideoCodecType_AppleProRes4444XQ;
      if (p4444) {
        bits = 12;
      } else if (sub == kCMVideoCodecType_AppleProRes422 ||
                 sub == kCMVideoCodecType_AppleProRes422HQ ||
                 sub == kCMVideoCodecType_AppleProRes422LT ||
                 sub == kCMVideoCodecType_AppleProRes422Proxy) {
        bits = 10;
      }
      vi.color.source_bits = bits;
      CFTypeRef a = fd_get(fd,
          kCMFormatDescriptionExtension_ContainsAlphaChannel);
      int depth = 0;
      if (CFTypeRef d = fd_get(fd, kCMFormatDescriptionExtension_Depth)) {
        CFNumberGetValue((CFNumberRef)d, kCFNumberIntType, &depth);
      }
      vi.has_alpha = (a && CFBooleanGetValue((CFBooleanRef)a)) ||
                     (p4444 && depth == 32);
      CFTypeRef mode = fd_get(fd,
          kCMFormatDescriptionExtension_AlphaChannelMode);
      vi.color.premultiplied = mode && CFEqual(mode,
          kCMFormatDescriptionAlphaChannelMode_PremultipliedAlpha);
      vi.color.mastering = mastering_from(fd_get(fd,
          kCMFormatDescriptionExtension_MasteringDisplayColorVolume));
      vi.color.content_light = content_light_from(fd_get(fd,
          kCMFormatDescriptionExtension_ContentLightLevelInfo));
    }
    NSError* rerr = nil;
    AVAssetReader* reader = [[AVAssetReader alloc] initWithAsset:asset
                                                           error:&rerr];
    if (!reader) {
      if (err) { *err = "cannot read '" + path + "': " + describe(rerr); }
      return false;
    }
    if (opt.start_s > 0 || opt.duration_s > 0) {
      const CMTime start = CMTimeMakeWithSeconds(opt.start_s, 600000);
      const CMTime dur = opt.duration_s > 0
          ? CMTimeMakeWithSeconds(opt.duration_s, 600000)
          : kCMTimePositiveInfinity;
      reader.timeRange = CMTimeRangeMake(start, dur);
    }
    // The decoder converts to RGB (its matrix), leaving primaries and
    // transfer as they are: the samples are the source's code values.
    // F16 and F32 pictures both come from half-float buffers.
    const bool f16 = opt.dtype != TensorBeat::DType::U8;
    NSDictionary* settings = @{
      (id)kCVPixelBufferPixelFormatTypeKey :
          @(f16 ? kCVPixelFormatType_64RGBAHalf : kCVPixelFormatType_32BGRA)
    };
    AVAssetReaderTrackOutput* out =
        [AVAssetReaderTrackOutput assetReaderTrackOutputWithTrack:track
                                                   outputSettings:settings];
    out.alwaysCopiesSampleData = NO;
    if (![reader canAddOutput:out]) {
      if (err) { *err = "cannot decode the video track of '" + path + "'"; }
      return false;
    }
    [reader addOutput:out];
    if (![reader startReading]) {
      if (err) {
        *err = "cannot start reading '" + path + "': " +
               describe(reader.error);
      }
      return false;
    }
    _impl->reader = reader;
    _impl->output = out;
    return true;
  }
}

bool
VideoReader::read(TensorBeat* out, std::int64_t* pts_us, std::string* err)
{
  @autoreleasepool {
    if (!_impl->output) { return false; }
    for (;;) {
      CMSampleBufferRef sb = [_impl->output copyNextSampleBuffer];
      if (!sb) {
        if (_impl->reader.status == AVAssetReaderStatusFailed && err) {
          *err = "decoding failed: " + describe(_impl->reader.error);
        }
        return false;
      }
      CVImageBufferRef ib = CMSampleBufferGetImageBuffer(sb);
      if (!ib) {  // a sample with no picture (a marker)
        CFRelease(sb);
        continue;
      }
      const CMTime pts = CMSampleBufferGetPresentationTimeStamp(sb);
      if (pts_us) {
        *pts_us = (std::int64_t)std::llround(CMTimeGetSeconds(pts) * 1e6);
      }
      const VideoInfo& vi = _impl->info;
      const bool f16 = _impl->opt.dtype != TensorBeat::DType::U8;
      CVPixelBufferLockBaseAddress(ib, kCVPixelBufferLock_ReadOnly);
      const int channels = _impl->opt.keep_alpha && vi.has_alpha ? 4 : 3;
      const bool ok = interleaved_to_planar(
          CVPixelBufferGetBaseAddress(ib), CVPixelBufferGetBytesPerRow(ib),
          (int)CVPixelBufferGetWidth(ib), (int)CVPixelBufferGetHeight(ib),
          f16 ? Interleaved::RGBAHalf : Interleaved::BGRA8, channels,
          _impl->opt.dtype, false, out, err);
      CVPixelBufferUnlockBaseAddress(ib, kCVPixelBufferLock_ReadOnly);
      ColorTags t = vi.color;
      // What the decoded buffer says it holds wins over the track -- for
      // an untagged track that is how the decoder read it (BT.601 for
      // SD, BT.709 for HD), the matrix included, which a re-encode then
      // reuses. The samples are RGB; the matrix is kept for that.
      tags_from(attachment_get, ib, &t);
      if (vi.color.matrix != 2) { t.matrix = vi.color.matrix; }
      CFRelease(sb);
      if (!ok) { return false; }
      t.write(out->sideband);
      return true;
    }
  }
}

// ---- writing ---------------------------------------------------------------

bool
video_codec_known(std::string_view c)
{
  return c == "prores4444" || c == "prores4444xq" || c == "prores422hq" ||
         c == "prores422" || c == "prores422lt" || c == "prores422proxy" ||
         c == "hevc" || c == "hevc8" || c == "h264";
}

struct VideoWriter::Impl {
  AVAssetWriter*                        writer = nil;
  AVAssetWriterInput*                   input = nil;
  AVAssetWriterInputPixelBufferAdaptor* adaptor = nil;
  VideoWriteOptions                     opt;
  CFStringRef                           primaries = nullptr;
  CFStringRef                           transfer = nullptr;
  CFStringRef                           matrix = nullptr;
  bool                                  finished = false;
};

VideoWriter::VideoWriter() : _impl(std::make_unique<Impl>()) {}
VideoWriter::~VideoWriter()
{
  if (_impl->writer && !_impl->finished &&
      _impl->writer.status == AVAssetWriterStatusWriting) {
    [_impl->writer cancelWriting];
  }
}

// H.264's profile at a level, as VideoToolbox names them; null for one
// it has not. "" is High, at the level the encoder picks.
static CFStringRef
h264_profile_level(const std::string& profile, const std::string& level)
{
  struct Row {
    const char* profile;
    const char* level;
    CFStringRef value;
  };
  static const Row kRows[] = {
    {"baseline", "", kVTProfileLevel_H264_Baseline_AutoLevel},
    {"baseline", "3.0", kVTProfileLevel_H264_Baseline_3_0},
    {"baseline", "3.1", kVTProfileLevel_H264_Baseline_3_1},
    {"baseline", "3.2", kVTProfileLevel_H264_Baseline_3_2},
    {"baseline", "4.0", kVTProfileLevel_H264_Baseline_4_0},
    {"baseline", "4.1", kVTProfileLevel_H264_Baseline_4_1},
    {"baseline", "4.2", kVTProfileLevel_H264_Baseline_4_2},
    {"baseline", "5.0", kVTProfileLevel_H264_Baseline_5_0},
    {"baseline", "5.1", kVTProfileLevel_H264_Baseline_5_1},
    {"baseline", "5.2", kVTProfileLevel_H264_Baseline_5_2},
    {"main", "", kVTProfileLevel_H264_Main_AutoLevel},
    {"main", "3.0", kVTProfileLevel_H264_Main_3_0},
    {"main", "3.1", kVTProfileLevel_H264_Main_3_1},
    {"main", "3.2", kVTProfileLevel_H264_Main_3_2},
    {"main", "4.0", kVTProfileLevel_H264_Main_4_0},
    {"main", "4.1", kVTProfileLevel_H264_Main_4_1},
    {"main", "4.2", kVTProfileLevel_H264_Main_4_2},
    {"main", "5.0", kVTProfileLevel_H264_Main_5_0},
    {"main", "5.1", kVTProfileLevel_H264_Main_5_1},
    {"main", "5.2", kVTProfileLevel_H264_Main_5_2},
    {"high", "", kVTProfileLevel_H264_High_AutoLevel},
    {"high", "3.0", kVTProfileLevel_H264_High_3_0},
    {"high", "3.1", kVTProfileLevel_H264_High_3_1},
    {"high", "3.2", kVTProfileLevel_H264_High_3_2},
    {"high", "4.0", kVTProfileLevel_H264_High_4_0},
    {"high", "4.1", kVTProfileLevel_H264_High_4_1},
    {"high", "4.2", kVTProfileLevel_H264_High_4_2},
    {"high", "5.0", kVTProfileLevel_H264_High_5_0},
    {"high", "5.1", kVTProfileLevel_H264_High_5_1},
    {"high", "5.2", kVTProfileLevel_H264_High_5_2},
  };
  const std::string p = profile.empty() ? "high" : profile;
  const std::string l = level == "auto" ? "" : level;
  for (const Row& r : kRows) {
    if (p == r.profile && l == r.level) { return r.value; }
  }
  return nullptr;
}

bool
h264_settings_known(const std::string& profile, const std::string& level,
                    const std::string& entropy)
{
  return h264_profile_level(profile, level) != nullptr &&
         (entropy.empty() || entropy == "auto" || entropy == "cabac" ||
          entropy == "cavlc");
}

bool
VideoWriter::open(const std::string& path, const VideoWriteOptions& opt,
                  std::string* err)
{
  @autoreleasepool {
    _impl->opt = opt;
    if (!video_codec_known(opt.codec)) {
      if (err) { *err = "unknown video codec '" + opt.codec + "'"; }
      return false;
    }
    if (opt.width <= 0 || opt.height <= 0 || opt.fps_num <= 0 ||
        opt.fps_den <= 0) {
      if (err) { *err = "a video needs a size and a frame rate"; }
      return false;
    }
    NSString* p = @(path.c_str());
    NSString* ext = p.pathExtension.lowercaseString;
    const bool mp4 = [ext isEqualToString:@"mp4"] ||
                     [ext isEqualToString:@"m4v"];
    if (mp4 && is_prores(opt.codec)) {
      if (err) { *err = "ProRes goes in a QuickTime movie (.mov), not MP4"; }
      return false;
    }
    [[NSFileManager defaultManager] removeItemAtPath:p error:nil];
    NSError* werr = nil;
    AVAssetWriter* writer = [AVAssetWriter
        assetWriterWithURL:[NSURL fileURLWithPath:p]
                  fileType:mp4 ? AVFileTypeMPEG4 : AVFileTypeQuickTimeMovie
                     error:&werr];
    if (!writer) {
      if (err) { *err = "cannot write '" + path + "': " + describe(werr); }
      return false;
    }
    const std::string& c = opt.codec;
    AVVideoCodecType type =
        c == "prores4444"     ? AVVideoCodecTypeAppleProRes4444
      : c == "prores4444xq"   ? AVVideoCodecTypeAppleProRes4444XQ
      : c == "prores422hq"    ? AVVideoCodecTypeAppleProRes422HQ
      : c == "prores422"      ? AVVideoCodecTypeAppleProRes422
      : c == "prores422lt"    ? AVVideoCodecTypeAppleProRes422LT
      : c == "prores422proxy" ? AVVideoCodecTypeAppleProRes422Proxy
      : c == "h264"           ? AVVideoCodecTypeH264
                              : AVVideoCodecTypeHEVC;
    // The frames' colour, as the file will declare it. A video file has
    // no sRGB transfer in practice: sRGB-encoded frames are tagged
    // BT.709, as video tools expect.
    const ColorTags& t = opt.color;
    const int transfer = t.transfer == 13 || t.transfer == 2 ? 1
                                                             : t.transfer;
    _impl->primaries = cv_of(t.primaries, kPrimaries,
                             kCVImageBufferColorPrimaries_ITU_R_709_2);
    _impl->transfer = cv_of(transfer, kTransfers,
                            kCVImageBufferTransferFunction_ITU_R_709_2);
    const int m = t.matrix > 0 && t.matrix != 2 ? t.matrix
                : t.primaries == 9              ? 9 : 1;
    _impl->matrix = cv_of(m, kMatrices,
                          kCVImageBufferYCbCrMatrix_ITU_R_709_2);
    NSMutableDictionary* comp = [NSMutableDictionary dictionary];
    if (c == "hevc") {
      comp[AVVideoProfileLevelKey] =
          (__bridge NSString*)kVTProfileLevel_HEVC_Main10_AutoLevel;
    } else if (c == "hevc8") {
      comp[AVVideoProfileLevelKey] =
          (__bridge NSString*)kVTProfileLevel_HEVC_Main_AutoLevel;
    } else if (c == "h264") {
      CFStringRef pl = h264_profile_level(opt.profile, opt.level);
      if (!pl) {
        if (err) {
          *err = "no H.264 profile '" + opt.profile + "' at level '" +
                 opt.level + "'";
        }
        return false;
      }
      comp[AVVideoProfileLevelKey] = (__bridge NSString*)pl;
      // Baseline has CAVLC alone; the others CABAC unless asked.
      if (opt.profile != "baseline" && !opt.entropy.empty()) {
        comp[AVVideoH264EntropyModeKey] = opt.entropy == "cavlc"
            ? AVVideoH264EntropyModeCAVLC : AVVideoH264EntropyModeCABAC;
      }
    }
    if (!is_prores(c)) {
      if (opt.bitrate > 0) { comp[AVVideoAverageBitRateKey] = @(opt.bitrate); }
      if (opt.quality > 0) {
        comp[(__bridge NSString*)kVTCompressionPropertyKey_Quality] =
            @(opt.quality);
      }
      // A peak: so many bytes in any one second.
      if (opt.max_bitrate > 0) {
        comp[(__bridge NSString*)kVTCompressionPropertyKey_DataRateLimits] =
            @[ @(opt.max_bitrate / 8), @1 ];
      }
      if (opt.keyframe_interval > 0) {
        comp[AVVideoMaxKeyFrameIntervalKey] = @(opt.keyframe_interval);
      }
      // Baseline has no B-frames: none, whatever was asked.
      const bool baseline = c == "h264" && opt.profile == "baseline";
      if (baseline || opt.frame_reordering >= 0) {
        comp[AVVideoAllowFrameReorderingKey] =
            @(!baseline && opt.frame_reordering > 0);
      }
    }
    if (c == "hevc" && t.is_hdr()) {
      if (t.mastering.size() == 10) {
        comp[(__bridge NSString*)
                 kVTCompressionPropertyKey_MasteringDisplayColorVolume] =
            mastering_data(t.mastering);
      }
      if (t.content_light.size() == 2) {
        comp[(__bridge NSString*)
                 kVTCompressionPropertyKey_ContentLightLevelInfo] =
            content_light_data(t.content_light);
      }
    }
    if (opt.alpha && (c == "prores4444" || c == "prores4444xq")) {
      comp[(__bridge NSString*)kVTCompressionPropertyKey_AlphaChannelMode] =
          (__bridge NSString*)(t.premultiplied
                                   ? kVTAlphaChannelMode_PremultipliedAlpha
                                   : kVTAlphaChannelMode_StraightAlpha);
    }
    NSMutableDictionary* settings = [@{
      AVVideoCodecKey : type,
      AVVideoWidthKey : @(opt.width),
      AVVideoHeightKey : @(opt.height),
      AVVideoColorPropertiesKey : @{
        AVVideoColorPrimariesKey : (__bridge NSString*)_impl->primaries,
        AVVideoTransferFunctionKey : (__bridge NSString*)_impl->transfer,
        AVVideoYCbCrMatrixKey : (__bridge NSString*)_impl->matrix,
      },
    } mutableCopy];
    if (comp.count) { settings[AVVideoCompressionPropertiesKey] = comp; }
    AVAssetWriterInput* input = nil;
    @try {
      input = [AVAssetWriterInput assetWriterInputWithMediaType:AVMediaTypeVideo
                                                 outputSettings:settings];
    } @catch (NSException* e) {
      if (err) {
        *err = std::string("the encoder refused its settings: ") +
               e.reason.UTF8String;
      }
      return false;
    }
    input.expectsMediaDataInRealTime = NO;
    // Frame times on the rate's own clock (1001 ticks a frame at 30000
    // for 29.97), so the file's frame rate reads back exactly.
    input.mediaTimeScale = (CMTimeScale)opt.fps_num;
    writer.movieTimeScale = (CMTimeScale)opt.fps_num;
    if (![writer canAddInput:input]) {
      if (err) { *err = "cannot add a " + c + " track to '" + path + "'"; }
      return false;
    }
    [writer addInput:input];
    // Half-float RGBA in, whatever the codec: 10 and 12 bits (and an
    // HDR transfer's code values) reach the encoder intact.
    NSDictionary* attrs = @{
      (id)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_64RGBAHalf),
      (id)kCVPixelBufferWidthKey : @(opt.width),
      (id)kCVPixelBufferHeightKey : @(opt.height),
      (id)kCVPixelBufferIOSurfacePropertiesKey : @{},
    };
    AVAssetWriterInputPixelBufferAdaptor* adaptor =
        [AVAssetWriterInputPixelBufferAdaptor
            assetWriterInputPixelBufferAdaptorWithAssetWriterInput:input
                                       sourcePixelBufferAttributes:attrs];
    if (![writer startWriting]) {
      if (err) {
        *err = "cannot start writing '" + path + "': " +
               describe(writer.error);
      }
      return false;
    }
    [writer startSessionAtSourceTime:kCMTimeZero];
    _impl->writer = writer;
    _impl->input = input;
    _impl->adaptor = adaptor;
    return true;
  }
}

bool
VideoWriter::write(const TensorBeat& pic, std::int64_t index,
                   std::string* err)
{
  @autoreleasepool {
    Impl& w = *_impl;
    if (!w.writer || w.finished) {
      if (err) { *err = "the video writer is not open"; }
      return false;
    }
    if (pic.shape.size() != 3 || pic.shape[1] != w.opt.height ||
        pic.shape[2] != w.opt.width) {
      if (err) { *err = "a frame is not the video's size"; }
      return false;
    }
    // The encoder takes frames as fast as it can compress them.
    const auto give_up = std::chrono::steady_clock::now() +
                         std::chrono::seconds(60);
    while (!w.input.readyForMoreMediaData) {
      if (w.writer.status != AVAssetWriterStatusWriting ||
          std::chrono::steady_clock::now() > give_up) {
        if (err) { *err = "the encoder stopped: " + describe(w.writer.error); }
        return false;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CVPixelBufferRef pb = nullptr;
    if (CVPixelBufferPoolCreatePixelBuffer(nullptr, w.adaptor.pixelBufferPool,
                                           &pb) != kCVReturnSuccess || !pb) {
      if (err) { *err = "no pixel buffer from the encoder's pool"; }
      return false;
    }
    CVPixelBufferLockBaseAddress(pb, 0);
    const bool ok = planar_to_interleaved(
        pic, Interleaved::RGBAHalf, CVPixelBufferGetBaseAddress(pb),
        CVPixelBufferGetBytesPerRow(pb), false, err, !w.opt.alpha);
    CVPixelBufferUnlockBaseAddress(pb, 0);
    if (!ok) {
      CVPixelBufferRelease(pb);
      return false;
    }
    // The samples are in the colour the file declares: nothing to
    // convert but the matrix.
    CVBufferSetAttachment(pb, kCVImageBufferColorPrimariesKey, w.primaries,
                          kCVAttachmentMode_ShouldPropagate);
    CVBufferSetAttachment(pb, kCVImageBufferTransferFunctionKey, w.transfer,
                          kCVAttachmentMode_ShouldPropagate);
    CVBufferSetAttachment(pb, kCVImageBufferYCbCrMatrixKey, w.matrix,
                          kCVAttachmentMode_ShouldPropagate);
    const CMTime t = CMTimeMake(index * w.opt.fps_den, w.opt.fps_num);
    const BOOL appended = [w.adaptor appendPixelBuffer:pb
                                  withPresentationTime:t];
    CVPixelBufferRelease(pb);
    if (!appended) {
      if (err) { *err = "the encoder refused a frame: " +
                        describe(w.writer.error); }
      return false;
    }
    return true;
  }
}

bool
VideoWriter::finish(std::string* err)
{
  @autoreleasepool {
    Impl& w = *_impl;
    if (!w.writer || w.finished) { return w.finished; }
    w.finished = true;
    [w.input markAsFinished];
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    [w.writer finishWritingWithCompletionHandler:^{
      dispatch_semaphore_signal(sem);
    }];
    dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
    if (w.writer.status != AVAssetWriterStatusCompleted) {
      if (err) { *err = "the movie was not finished: " +
                        describe(w.writer.error); }
      return false;
    }
    return true;
  }
}

}  // namespace vpipe::apple_media

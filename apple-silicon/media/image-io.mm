#include "apple-silicon/media/image-io.h"

#include "apple-silicon/media/planar.h"

#import <CoreGraphics/CoreGraphics.h>
#import <CoreImage/CoreImage.h>
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>

#include <cstring>
#include <utility>
#include <vector>

namespace vpipe::apple_media {

namespace {

// ---- colour spaces <-> CICP --------------------------------------------

struct NamedSpace {
  CFStringRef name;
  int         primaries;
  int         transfer;
};

// The named Core Graphics spaces CICP can say, extended variants
// included (the same encoding, unclamped).
const NamedSpace*
named_spaces(std::size_t* n)
{
  static const NamedSpace k[] = {
    {kCGColorSpaceSRGB, 1, 13},
    {kCGColorSpaceExtendedSRGB, 1, 13},
    {kCGColorSpaceLinearSRGB, 1, 8},
    {kCGColorSpaceExtendedLinearSRGB, 1, 8},
    {kCGColorSpaceITUR_709, 1, 1},
    {kCGColorSpaceDisplayP3, 12, 13},
    {kCGColorSpaceExtendedDisplayP3, 12, 13},
    {kCGColorSpaceLinearDisplayP3, 12, 8},
    {kCGColorSpaceExtendedLinearDisplayP3, 12, 8},
    {kCGColorSpaceDisplayP3_PQ, 12, 16},
    {kCGColorSpaceDisplayP3_HLG, 12, 18},
    {kCGColorSpaceITUR_2020, 9, 14},
    {kCGColorSpaceExtendedITUR_2020, 9, 14},
    {kCGColorSpaceITUR_2020_sRGBGamma, 9, 13},
    {kCGColorSpaceLinearITUR_2020, 9, 8},
    {kCGColorSpaceExtendedLinearITUR_2020, 9, 8},
    {kCGColorSpaceITUR_2100_PQ, 9, 16},
    {kCGColorSpaceITUR_2100_HLG, 9, 18},
  };
  *n = sizeof(k) / sizeof(k[0]);
  return k;
}

// The CICP a colour space declares; false for one CICP cannot name (an
// arbitrary ICC profile) or one that is not RGB.
bool
cicp_from_space(CGColorSpaceRef cs, ColorTags* t)
{
  if (!cs || CGColorSpaceGetModel(cs) != kCGColorSpaceModelRGB) {
    return false;
  }
  CFStringRef name = CGColorSpaceCopyName(cs);
  if (!name) { return false; }
  std::size_t n = 0;
  const NamedSpace* k = named_spaces(&n);
  bool found = false;
  for (std::size_t i = 0; i < n && !found; ++i) {
    if (CFEqual(name, k[i].name)) {
      t->primaries = k[i].primaries;
      t->transfer = k[i].transfer;
      t->matrix = 0;
      t->full_range = true;
      found = true;
    }
  }
  CFRelease(name);
  return found;
}

// The Core Graphics space for a picture tagged `t`: `extended` for float
// samples (which may run past 0..1), `linear` for the same primaries in
// linear light. Untagged or unknown pictures are sRGB.
CGColorSpaceRef
space_for(const ColorTags& t, bool extended, bool linear)
{
  const int p = (t.primaries == 9 || t.primaries == 12) ? t.primaries : 1;
  const int tr = linear ? 8 : t.transfer;
  CFStringRef name = extended ? kCGColorSpaceExtendedSRGB
                              : kCGColorSpaceSRGB;
  if (tr == 8) {
    name = p == 9    ? (extended ? kCGColorSpaceExtendedLinearITUR_2020
                                 : kCGColorSpaceLinearITUR_2020)
         : p == 12   ? (extended ? kCGColorSpaceExtendedLinearDisplayP3
                                 : kCGColorSpaceLinearDisplayP3)
                     : (extended ? kCGColorSpaceExtendedLinearSRGB
                                 : kCGColorSpaceLinearSRGB);
  } else if (tr == 16) {
    name = p == 12 ? kCGColorSpaceDisplayP3_PQ : kCGColorSpaceITUR_2100_PQ;
  } else if (tr == 18) {
    name = p == 12 ? kCGColorSpaceDisplayP3_HLG : kCGColorSpaceITUR_2100_HLG;
  } else if (p == 9) {
    name = tr == 13 ? kCGColorSpaceITUR_2020_sRGBGamma
         : (extended ? kCGColorSpaceExtendedITUR_2020
                     : kCGColorSpaceITUR_2020);
  } else if (p == 12) {
    name = extended ? kCGColorSpaceExtendedDisplayP3 : kCGColorSpaceDisplayP3;
  } else if (tr == 1 || tr == 6 || tr == 14 || tr == 15) {
    name = kCGColorSpaceITUR_709;
  }
  return CGColorSpaceCreateWithName(name);
}

CGImageSourceRef
open_source(const std::string& path, std::string* err)
{
  NSURL* url = [NSURL fileURLWithPath:@(path.c_str())];
  CGImageSourceRef src = CGImageSourceCreateWithURL(
      (__bridge CFURLRef)url, nullptr);
  if (!src && err) { *err = "ImageIO cannot open '" + path + "'"; }
  return src;
}

bool
is_raw_type(CFStringRef type)
{
  if (!type) { return false; }
  UTType* t = [UTType typeWithIdentifier:(__bridge NSString*)type];
  return t && [t conformsToType:UTTypeRAWImage];
}

// One context for every development: Core Image contexts are costly to
// make and safe to share across threads. The working space is extended
// LINEAR sRGB, said rather than defaulted, because the Rendered clamp
// below is only an sRGB clamp in a space with sRGB's primaries; nothing
// is clamped on the way to any other output space.
CIContext*
raw_context()
{
  static CIContext* ctx = [] {
    CGColorSpaceRef ws =
        CGColorSpaceCreateWithName(kCGColorSpaceExtendedLinearSRGB);
    CIContext* c = [CIContext contextWithOptions:@{
      kCIContextWorkingColorSpace : (__bridge id)ws,
      kCIContextWorkingFormat : @(kCIFormatRGBAh),
      kCIContextCacheIntermediates : @NO,
    }];
    CGColorSpaceRelease(ws);
    return c;
  }();
  return ctx;
}

// A camera RAW, developed (RawOptions) into a planar picture. CIRAWFilter
// rather than ImageIO's own decode: ImageIO hands back the sensor's
// orientation in the camera's default look, Display P3, with nothing to
// say about it; the filter applies the orientation and takes the look.
std::unique_ptr<TensorBeatPayload>
develop_raw(const std::string& path, TensorBeat::DType dtype,
            bool keep_alpha, const RawOptions& o, int source_bits,
            ColorTags* tags, std::string* err)
{
  NSURL* url = [NSURL fileURLWithPath:@(path.c_str())];
  CIRAWFilter* f = [CIRAWFilter filterWithImageURL:url];
  if (!f) {
    if (err) {
      *err = "Core Image cannot develop the RAW '" + path +
             "' (a camera this macOS does not know?)";
    }
    return nullptr;
  }
  const bool linear = o.look == RawOptions::Look::Linear;
  f.exposure = (float)o.exposure;
  if (linear) {
    f.boostAmount = 0;                           // no global tone curve
    if (f.localToneMapSupported) { f.localToneMapAmount = 0; }
    f.extendedDynamicRangeAmount = 2;            // keep the headroom
  }
  CIImage* img = f.outputImage;
  if (!img) {
    if (err) { *err = "Core Image developed nothing from '" + path + "'"; }
    return nullptr;
  }
  if (!linear) {
    // Display-referred: what an sRGB export of the photo holds. Clamped
    // in the (linear sRGB) working space, which is the sRGB clamp.
    img = [img imageByApplyingFilter:@"CIColorClamp" withInputParameters:@{
      @"inputMinComponents" : [CIVector vectorWithX:0 Y:0 Z:0 W:0],
      @"inputMaxComponents" : [CIVector vectorWithX:1 Y:1 Z:1 W:1],
    }];
  }
  const CGRect r = CGRectIntegral(img.extent);
  const int W = (int)r.size.width;
  const int H = (int)r.size.height;
  if (W <= 0 || H <= 0) {
    if (err) { *err = "the RAW '" + path + "' developed to nothing"; }
    return nullptr;
  }
  CGColorSpaceRef cs = CGColorSpaceCreateWithName(
      linear ? kCGColorSpaceExtendedLinearITUR_2020 : kCGColorSpaceSRGB);
  const std::size_t row = (std::size_t)W * 8;
  std::vector<std::uint8_t> rgba(row * (std::size_t)H);
  [raw_context() render:img toBitmap:rgba.data() rowBytes:(ptrdiff_t)row
                 bounds:r format:kCIFormatRGBAh colorSpace:cs];
  CGColorSpaceRelease(cs);
  auto out = std::make_unique<TensorBeatPayload>();
  if (!interleaved_to_planar(rgba.data(), row, W, H, Interleaved::RGBAHalf,
                             keep_alpha ? 4 : 3, dtype, false, out.get(),
                             err)) {
    return nullptr;
  }
  ColorTags t = ColorTags::srgb();
  if (linear) {
    t.primaries = 9;
    t.transfer = 8;
  }
  t.source_bits = source_bits;
  if (tags) { *tags = t; }
  t.write(out->sideband);
  return out;
}

CFStringRef
uti_for(std::string_view format)
{
  if (format == "png") { return CFSTR("public.png"); }
  if (format == "tiff" || format == "tif") { return CFSTR("public.tiff"); }
  if (format == "exr") { return CFSTR("com.ilm.openexr-image"); }
  if (format == "jpeg" || format == "jpg") { return CFSTR("public.jpeg"); }
  if (format == "heic") { return CFSTR("public.heic"); }
  return nullptr;
}

}  // namespace

bool
probe_still(const std::string& path, StillInfo* out, std::string* err)
{
  @autoreleasepool {
    CGImageSourceRef src = open_source(path, err);
    if (!src) { return false; }
    NSDictionary* p = CFBridgingRelease(
        CGImageSourceCopyPropertiesAtIndex(src, 0, nullptr));
    CFStringRef type = CGImageSourceGetType(src);
    if (type) { out->uti = [(__bridge NSString*)type UTF8String]; }
    out->raw = is_raw_type(type);
    CFRelease(src);
    if (!p) {
      if (err) { *err = "ImageIO found no picture in '" + path + "'"; }
      return false;
    }
    out->width = [p[(id)kCGImagePropertyPixelWidth] intValue];
    out->height = [p[(id)kCGImagePropertyPixelHeight] intValue];
    const int depth = [p[(id)kCGImagePropertyDepth] intValue];
    out->bits = depth > 0 ? depth : 8;
    out->is_float = [p[(id)kCGImagePropertyIsFloat] boolValue];
    out->has_alpha = [p[(id)kCGImagePropertyHasAlpha] boolValue];
    NSNumber* orient = p[(id)kCGImagePropertyOrientation];
    out->orientation = orient ? orient.intValue : 1;
    // A RAW comes out of read_still upright: say the size it comes out at.
    if (out->raw && out->orientation >= 5 && out->orientation <= 8) {
      std::swap(out->width, out->height);
    }
    return out->width > 0 && out->height > 0;
  }
}

bool
prefers_imageio(const StillInfo& s)
{
  return s.raw || s.bits > 8 || s.is_float ||
         s.uti == "com.ilm.openexr-image" || s.uti == "public.heic" ||
         s.uti == "public.heif";
}

std::unique_ptr<TensorBeatPayload>
read_still(const std::string& path, TensorBeat::DType dtype, bool keep_alpha,
           ColorTags* tags, std::string* err, const RawOptions& raw)
{
  @autoreleasepool {
    CGImageSourceRef src = open_source(path, err);
    if (!src) { return nullptr; }
    if (is_raw_type(CGImageSourceGetType(src))) {
      NSDictionary* p = CFBridgingRelease(
          CGImageSourceCopyPropertiesAtIndex(src, 0, nullptr));
      CFRelease(src);
      const int depth = [p[(id)kCGImagePropertyDepth] intValue];
      return develop_raw(path, dtype, keep_alpha, raw,
                         depth > 0 ? depth : 16, tags, err);
    }
    // Float stays float (EXR, an HDR still), instead of ImageIO's
    // default of tone-mapping it into 8 or 16 bits.
    NSDictionary* opts = @{(id)kCGImageSourceShouldAllowFloat : @YES};
    CGImageRef img = CGImageSourceCreateImageAtIndex(
        src, 0, (__bridge CFDictionaryRef)opts);
    CFRelease(src);
    if (!img) {
      if (err) { *err = "ImageIO cannot decode '" + path + "'"; }
      return nullptr;
    }
    const int W = (int)CGImageGetWidth(img);
    const int H = (int)CGImageGetHeight(img);
    const CGImageAlphaInfo ai = CGImageGetAlphaInfo(img);
    const bool has_alpha = ai != kCGImageAlphaNone &&
                           ai != kCGImageAlphaNoneSkipLast &&
                           ai != kCGImageAlphaNoneSkipFirst;
    ColorTags t;
    CGColorSpaceRef cs = CGImageGetColorSpace(img);
    // F32 draws as F16 does, widened by interleaved_to_planar.
    const bool f16 = dtype != TensorBeat::DType::U8;
    // Draw in the file's own space when CICP can name it: no conversion,
    // the samples as stored. Otherwise convert to (extended) sRGB.
    CGColorSpaceRef draw = nullptr;
    if (cicp_from_space(cs, &t)) {
      draw = CGColorSpaceRetain(cs);
    } else {
      t = ColorTags::srgb();
      draw = CGColorSpaceCreateWithName(f16 ? kCGColorSpaceExtendedSRGB
                                            : kCGColorSpaceSRGB);
    }
    t.source_bits = (int)CGImageGetBitsPerComponent(img);
    t.premultiplied = false;
    // Premultiplied when there is alpha to keep or to undo (what a
    // bitmap context holds), undone below; skip-alpha otherwise, so
    // nothing is composited.
    const uint32_t alpha = has_alpha || keep_alpha
        ? (uint32_t)kCGImageAlphaPremultipliedLast
        : (uint32_t)kCGImageAlphaNoneSkipLast;
    const uint32_t info =
        f16 ? (alpha | (uint32_t)kCGBitmapFloatComponents |
               (uint32_t)kCGBitmapByteOrder16Little)
            : (alpha | (uint32_t)kCGBitmapByteOrder32Big);
    const std::size_t bpp = f16 ? 8 : 4;
    CGContextRef ctx = CGBitmapContextCreate(nullptr, W, H, f16 ? 16 : 8,
                                             (std::size_t)W * bpp, draw,
                                             info);
    CGColorSpaceRelease(draw);
    if (!ctx) {
      CGImageRelease(img);
      if (err) { *err = "cannot make a bitmap for '" + path + "'"; }
      return nullptr;
    }
    CGContextSetBlendMode(ctx, kCGBlendModeCopy);
    CGContextSetInterpolationQuality(ctx, kCGInterpolationNone);
    CGContextDrawImage(ctx, CGRectMake(0, 0, W, H), img);
    CGImageRelease(img);
    auto out = std::make_unique<TensorBeatPayload>();
    const bool ok = interleaved_to_planar(
        CGBitmapContextGetData(ctx), CGBitmapContextGetBytesPerRow(ctx), W,
        H, f16 ? Interleaved::RGBAHalf : Interleaved::RGBA8,
        keep_alpha ? 4 : 3, dtype, has_alpha, out.get(), err);
    CGContextRelease(ctx);
    if (!ok) { return nullptr; }
    if (tags) { *tags = t; }
    t.write(out->sideband);
    return out;
  }
}

bool
still_format_known(std::string_view format)
{
  return uti_for(format) != nullptr;
}

bool
encode_still(const TensorBeat& pic, const ColorTags& tags,
             const StillWriteOptions& opt, std::vector<std::uint8_t>* out,
             std::string* err)
{
  @autoreleasepool {
    CFStringRef uti = uti_for(opt.format);
    if (!uti) {
      if (err) { *err = "ImageIO does not write '" + opt.format + "'"; }
      return false;
    }
    if (pic.shape.size() != 3) {
      if (err) { *err = "a picture is [C,H,W]"; }
      return false;
    }
    const bool exr = opt.format == "exr";
    const bool deep = !exr && opt.bits == 16 &&
                      (opt.format == "png" || opt.format == "tiff" ||
                       opt.format == "tif");
    const bool alpha = pic.shape[0] == 4;
    const int W = (int)pic.shape[2];
    const int H = (int)pic.shape[1];
    // The samples as they are, in the space their tags name: half for
    // EXR, 16 or 8 bits otherwise.
    const Interleaved fmt = exr ? Interleaved::RGBAHalf
                          : deep ? Interleaved::RGBA16 : Interleaved::RGBA8;
    const std::size_t row = (std::size_t)W * bytes_per_pixel(fmt);
    NSMutableData* px = [NSMutableData dataWithLength:row * H];
    // EXR is premultiplied (OpenEXR's convention); the others straight.
    if (!planar_to_interleaved(pic, fmt, px.mutableBytes, row, exr, err)) {
      return false;
    }
    uint32_t info = 0;
    if (exr) {
      info = (uint32_t)(alpha ? kCGImageAlphaPremultipliedLast
                              : kCGImageAlphaNoneSkipLast) |
             (uint32_t)kCGBitmapFloatComponents |
             (uint32_t)kCGBitmapByteOrder16Little;
    } else {
      info = (uint32_t)(alpha ? kCGImageAlphaLast
                              : kCGImageAlphaNoneSkipLast) |
             (uint32_t)(deep ? kCGBitmapByteOrder16Little
                             : kCGBitmapByteOrderDefault);
    }
    CGColorSpaceRef cs = space_for(tags, exr, false);
    CGDataProviderRef dp =
        CGDataProviderCreateWithCFData((__bridge CFDataRef)px);
    const int bpc = exr || deep ? 16 : 8;
    CGImageRef img = CGImageCreate(W, H, bpc, bpc * 4, row, cs, info, dp,
                                   nullptr, false,
                                   kCGRenderingIntentDefault);
    CGDataProviderRelease(dp);
    if (img && exr && !tags.is_linear()) {
      // OpenEXR holds linear light: the picture is drawn into the linear
      // space of the same primaries (Core Graphics converts the
      // transfer, PQ and HLG included).
      CGColorSpaceRef lin = space_for(tags, true, true);
      CGContextRef ctx = CGBitmapContextCreate(
          nullptr, W, H, 16, (std::size_t)W * 8, lin,
          (uint32_t)(alpha ? kCGImageAlphaPremultipliedLast
                           : kCGImageAlphaNoneSkipLast) |
              (uint32_t)kCGBitmapFloatComponents |
              (uint32_t)kCGBitmapByteOrder16Little);
      CGColorSpaceRelease(lin);
      CGImageRef linear = nullptr;
      if (ctx) {
        CGContextSetBlendMode(ctx, kCGBlendModeCopy);
        CGContextDrawImage(ctx, CGRectMake(0, 0, W, H), img);
        linear = CGBitmapContextCreateImage(ctx);
        CGContextRelease(ctx);
      }
      CGImageRelease(img);
      img = linear;
    }
    CGColorSpaceRelease(cs);
    if (!img) {
      if (err) { *err = "cannot make a CGImage to encode"; }
      return false;
    }
    NSMutableData* file = [NSMutableData data];
    CGImageDestinationRef dst = CGImageDestinationCreateWithData(
        (__bridge CFMutableDataRef)file, uti, 1, nullptr);
    bool ok = false;
    if (dst) {
      NSDictionary* props = @{};
      if (opt.format == "jpeg" || opt.format == "jpg" ||
          opt.format == "heic") {
        props = @{(id)kCGImageDestinationLossyCompressionQuality :
                      @(opt.quality)};
      }
      CGImageDestinationAddImage(dst, img, (__bridge CFDictionaryRef)props);
      ok = CGImageDestinationFinalize(dst);
      CFRelease(dst);
    }
    CGImageRelease(img);
    if (!ok) {
      if (err) { *err = "ImageIO could not encode " + opt.format; }
      return false;
    }
    const auto* b = static_cast<const std::uint8_t*>(file.bytes);
    out->assign(b, b + file.length);
    return true;
  }
}

}  // namespace vpipe::apple_media

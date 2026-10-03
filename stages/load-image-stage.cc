#include "stages/load-image-stage.h"
#include "stages/image-decode.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "apple-silicon/media/image-io.h"
#endif

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/image-metadata.h"
#include "common/media-line.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
}

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace vpipe {

LoadImageStage::LoadImageStage(const SessionContextIntf* s,
                               string                    id,
                               vector<InEdge>            iports,
                               FlexData                  config)
  : TypedStage<LoadImageStage>(s, std::move(id), std::move(iports),
                               std::move(config))
  , _libs(s ? s->services()->ffmpeg_libraries() : nullptr)
{
  // Validation is deferred to launch (see Stage::fail_config): the
  // stage must construct for any config so a graph can be built/edited
  // before a url is supplied.
  _keep_alpha = attr_str("alpha") == "keep";
  _dtype = attr_str("dtype");
  _decoder = attr_str("decoder");
  const string raw = attr_str("raw");
  _raw_linear = raw == "linear";
  _raw_exposure = attr_real("raw_exposure");
  if (raw != "rendered" && raw != "linear") {
    fail_config(fmt("LoadImageStage('{}'): raw must be rendered or linear",
                    this->id()));
  }
  if (_dtype != "u8" && _dtype != "f16" && _dtype != "auto") {
    fail_config(fmt("LoadImageStage('{}'): dtype must be u8, f16 or auto",
                    this->id()));
  }
  if (_decoder != "auto" && _decoder != "ffmpeg" && _decoder != "imageio") {
    fail_config(fmt(
        "LoadImageStage('{}'): decoder must be auto, ffmpeg or imageio",
        this->id()));
  }
#ifndef VPIPE_BUILD_APPLE_SILICON
  if (_decoder == "imageio" || _dtype == "f16") {
    fail_config(fmt("LoadImageStage('{}'): ImageIO and F16 need the "
                    "Apple build", this->id()));
  }
#endif
  const FlexData& cfg = this->config();
  if (cfg.is_object()) {
    auto root = cfg.as_object();
    if (root.contains("url")) {
      FlexData u = root.at("url");
      if (u.is_string()) {
        _urls.emplace_back(string(u.as_string("")));
      } else if (u.is_array()) {
        for (FlexData e : u.as_array()) {
          if (e.is_string()) {
            _urls.emplace_back(string(e.as_string("")));
          } else {
            fail_config(fmt(
                "LoadImageStage('{}'): config.url array entries must be "
                "strings", this->id()));
          }
        }
      } else {
        fail_config(fmt(
            "LoadImageStage('{}'): config.url must be a string or array "
            "of strings", this->id()));
      }
    }
  }
  if (_urls.empty()) {
    fail_config(fmt(
        "LoadImageStage('{}'): config.url is required "
        "(non-empty string or array of strings)", this->id()));
  }
  // Confine local file sources to the session file sandbox (network URLs
  // pass through; no-op when the sandbox is disabled). Escape fails config.
  for (auto& url : _urls) {
    url = confine_local_(url, /*for_write=*/false);
  }
  for (const auto& url : _urls) {
    if (url.empty()) {
      fail_config(fmt(
          "LoadImageStage('{}'): config.url contains an empty entry",
          this->id()));
    }
  }
  if (!_libs || !_libs->valid()) {
    fail_config(fmt(
        "LoadImageStage('{}'): FFmpegLibraries unavailable",
        this->id()));
  }
  allocate_oports(spec().oports.size());
}

namespace {
constexpr SpecExtra kAlphaChoices[] = {
  {spec_key::kChoices, "drop,keep"},
};
constexpr SpecExtra kDtypeChoices[] = {
  {spec_key::kChoices, "u8,f16,auto"},
};
constexpr SpecExtra kDecoderChoices[] = {
  {spec_key::kChoices, "auto,ffmpeg,imageio"},
};
constexpr SpecExtra kRawChoices[] = {
  {spec_key::kChoices, "rendered,linear"},
};
constexpr ConfigKey kAttrs[] = {
  {.key = "url", .type = ConfigType::Any, .required = true,
   .doc = "image path/URL string or array of strings",
   .is_path = true, .path_filter = "image"},
  {.key = "alpha", .type = ConfigType::String, .required = false,
   .doc = "drop | keep. `keep` emits FOUR channels [4,H,W] RGBA -- what "
          "an RGBA VAE (Qwen-Image-2.1) encodes and what save-image "
          "writes to PNG. A source with no alpha then gets a fully "
          "opaque one, so the shape is the graph's choice and not the "
          "file's. Unset => drop, which is [3,H,W] and what every other "
          "consumer here reads; a transparent PNG loaded that way keeps "
          "whatever colour sits under its transparent pixels rather "
          "than being composited",
   .def_str = "drop", .extra = kAlphaChoices},
  {.key = "dtype", .type = ConfigType::String, .required = false,
   .doc = "u8 | f16 | auto. u8 (the default) is what every consumer "
          "reads. f16 keeps what is deeper than 8 bits -- 16-bit PNG and "
          "TIFF, 10/12-bit HEIC, OpenEXR's float (values above 1 "
          "included): samples 0..1 over the file's code range in its own "
          "transfer, tagged with its colour (sideband color_primaries, "
          "color_transfer, ...). auto: f16 for a file deeper than 8 bits "
          "or float -- a camera RAW is -- u8 otherwise",
   .def_str = "u8", .extra = kDtypeChoices},
  {.key = "decoder", .type = ConfigType::String, .required = false,
   .doc = "auto | ffmpeg | imageio. imageio (Apple) reads 16-bit, float "
          "(OpenEXR) and HEIC stills as stored and keeps the colour space "
          "the file declares; FFmpeg's path is 8-bit and untagged. auto: "
          "ImageIO for those files, camera RAW and an f16 read, FFmpeg "
          "otherwise",
   .def_str = "auto", .extra = kDecoderChoices},
  {.key = "raw", .type = ConfigType::String, .required = false,
   .doc = "rendered | linear: how a camera RAW (CR2, CR3, NEF, ARW, RAF, "
          "DNG, ...) is DEVELOPED -- it has no colour space to read, the "
          "development decides one. rendered (the default): the camera's "
          "own look, as a viewer shows the file -- its tone curve, "
          "noise reduction, highlight recovery -- as sRGB 0..1, what a "
          "model was trained on. linear: scene-linear, no tone curve, "
          "the sensor's highlight headroom kept above 1, in linear "
          "BT.2020 so no camera colour clips; read it as f16. Either "
          "way the picture comes out UPRIGHT (the EXIF orientation "
          "applied, and the metadata oport's EXIF says 1). Ignored for "
          "other files",
   .def_str = "rendered", .extra = kRawChoices},
  {.key = "raw_exposure", .type = ConfigType::Real, .required = false,
   .doc = "exposure for a camera RAW, in EV on top of the camera's "
          "baseline (+1 doubles the light). Ignored for other files",
   .def_real = 0.0},
};
const PortSpec kIports[] = {
  {.name = "trigger", .doc = "optional pacing beat (e.g. chrono); each "
                             "one decodes + emits the next image",
   .type = nullptr, .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "image", .doc = "decoded image as a planar TensorBeat, U8 or "
                           "(per `dtype`) F16 -- RGB [3,H,W], or RGBA "
                           "[4,H,W] when the `alpha` config says keep",
   .type = &typeid(TensorBeatPayload),
   .tags = "rgb-frames", .clock_group = 0},
  {.name = "metadata", .doc = "FlexData {url, width, height, exif{...}, "
                              "exif_tiff_b64} carried with the image; one "
                              "beat per image, paired with oport 0. Feeds "
                              "save-image's metadata iport",
   .type = &typeid(FlexDataPayload),
   .tags = "image-metadata", .clock_group = 0},
  {.name = "images",
   .doc = "EVERY image of `url` as ONE LIST beat, in url order, each in "
          "the `image` port's format and at its own size -- the reference "
          "set of a multi-reference edit, for vae-encode's `images` and the "
          "conditioner's `ref_images`. Emitted once per run, ahead of any "
          "pacing; with only this port wired the stage ends after it. An "
          "image that cannot be decoded is left out and said so",
   .type = &typeid(TensorListPayload),
   .tags = "rgb-frames-list", .clock_group = 0},
};
constexpr unsigned kImagesPort = 2;
const StageSpec kSpec = {
  .type_name = "load-image",
  .doc       = "Source: decodes still images from files/URLs to planar "
               "RGB TensorBeats -- FFmpeg (U8), or ImageIO for 16-bit, "
               "OpenEXR and HEIC (U8 or F16, colour tagged). With a wired "
               "trigger iport one beat emits one image; unwired it emits "
               "all then ends.",
  .display_name = "Load Image",
  .category  = StageCategory::Visual,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
LoadImageStage::spec() const noexcept
{
  return kSpec;
}

LoadImageStage::~LoadImageStage() = default;

unique_ptr<BeatPayloadIntf>
LoadImageStage::decode_url_(const string& url) const
{
#ifdef VPIPE_BUILD_APPLE_SILICON
  // ImageIO for what FFmpeg would flatten, or when asked. Only a local
  // file: ImageIO does not fetch.
  string local = url;
  if (local.rfind("file://", 0) == 0) { local.erase(0, 7); }
  const bool remote = local.find("://") != string::npos;
  if (!remote && _decoder != "ffmpeg") {
    apple_media::StillInfo info;
    string perr;
    const bool known = apple_media::probe_still(local, &info, &perr);
    const bool deep = known && apple_media::prefers_imageio(info);
    const bool f16 = _dtype == "f16" || (_dtype == "auto" && deep);
    if (_decoder == "imageio" || f16 || deep) {
      apple_media::RawOptions raw;
      raw.look = _raw_linear ? apple_media::RawOptions::Look::Linear
                             : apple_media::RawOptions::Look::Rendered;
      raw.exposure = _raw_exposure;
      if (info.raw && _raw_linear && !f16) {
        session()->warn(fmt(
            "LoadImageStage('{}'): a linear RAW in 8 bits bands; read it "
            "with dtype f16", this->id()));
      }
      string err;
      auto out = apple_media::read_still(
          local, f16 ? TensorBeat::DType::F16 : TensorBeat::DType::U8,
          _keep_alpha, nullptr, &err, raw);
      if (!out) {
        session()->warn(fmt("LoadImageStage('{}'): {}", this->id(), err));
      }
      return out;
    }
  }
#endif
  ImageDecodeOptions opt;
  opt.keep_alpha = _keep_alpha;
  string err;
  auto out = decode_image_file(_libs, url, opt, &err);
  if (!out) {
    session()->warn(fmt("LoadImageStage('{}'): {}", this->id(), err));
  }
  return out;
}

unique_ptr<BeatPayloadIntf>
LoadImageStage::build_metadata_(const string&          url,
                                const BeatPayloadIntf& image) const
{
  FlexData o = FlexData::make_object();
  o.as_object().insert_or_assign("url", FlexData::make_string(url));

  const auto* tbp = dynamic_cast<const TensorBeatPayload*>(&image);
  if (tbp != nullptr && tbp->shape.size() == 3) {
    o.as_object().insert_or_assign(
        "height", FlexData::make_uint((uint64_t)tbp->shape[1]));
    o.as_object().insert_or_assign(
        "width", FlexData::make_uint((uint64_t)tbp->shape[2]));
  }

  // EXIF comes from the FILE, not the decoder: FFmpeg surfaces none of it
  // for stills (see common/image-metadata.h). Only a local path can be
  // re-read here -- a remote URL would mean a second fetch, so those simply
  // carry no exif rather than paying for one.
  string local = url;
  if (local.rfind("file://", 0) == 0) { local.erase(0, 7); }
  const bool is_remote = local.find("://") != string::npos;
  if (!is_remote) {
    vector<uint8_t> tiff = imgmeta::read_exif_blob(local);
#ifdef VPIPE_BUILD_APPLE_SILICON
    // A camera RAW was developed UPRIGHT (decode_url_), so the EXIF that
    // goes with it must not ask a viewer to turn it again.
    if (!tiff.empty() && _decoder != "ffmpeg") {
      apple_media::StillInfo info;
      string perr;
      if (apple_media::probe_still(local, &info, &perr) && info.raw &&
          info.orientation != 1) {
        imgmeta::exif_set_orientation(tiff, 1);
      }
    }
#endif
    if (!tiff.empty()) {
      FlexData ex = imgmeta::parse_exif(tiff);
      if (ex.is_object()) {
        o.as_object().insert_or_assign("exif", std::move(ex));
      }
      o.as_object().insert_or_assign(
          "exif_tiff_b64",
          FlexData::make_string(media_line::base64_encode(tiff)));
    }
  }
  return make_payload<FlexDataPayload>(std::move(o));
}

void
LoadImageStage::reset_run_state()
{
  // Per-launch reset. Stopping a pipeline destroys the RUNTIME, not the
  // stages: only unload / re-materialize destroys a Stage, so a plain
  // Stop-then-Start re-enters initialize() with this source already
  // exhausted from the previous run. Without this it would emit nothing
  // and signal done immediately -- its url list is already consumed, so
  // nothing downstream would see an image and the pipeline would
  // "complete" in milliseconds.
  _next = 0;
  _list_emitted = false;
}

Job
LoadImageStage::process(RuntimeContext& ctx)
{
  // The whole url list as ONE beat, once per run, when that port is wired.
  if (!_list_emitted && ctx.num_oports() > kImagesPort &&
      ctx.has_consumers(kImagesPort)) {
    _list_emitted = true;
    auto list = std::make_unique<TensorListPayload>();
    for (const string& url : _urls) {
      auto p = decode_url_(url);
      auto* tb = p ? dynamic_cast<TensorBeatPayload*>(p.get()) : nullptr;
      if (tb == nullptr) {
        session()->warn(fmt(
            "LoadImageStage('{}'): '{}' could not be decoded; the `images` "
            "list goes out without it", this->id(), url));
        continue;
      }
      list->items.push_back(std::move(static_cast<TensorBeat&>(*tb)));
    }
    co_await ctx.write(kImagesPort, std::move(list));
    // Nothing reads the one-at-a-time ports: the list was the whole job.
    const bool singles = ctx.has_consumers(0) ||
                         (ctx.num_oports() > 1 && ctx.has_consumers(1));
    if (!singles) {
      ctx.signal_done();
      co_return;
    }
  }

  if (_next >= _urls.size()) {
    ctx.signal_done();
    co_return;
  }

  // Pacing: when an iport is wired, gate each emission on one beat
  // upstream (typically a chrono tick). The payload type doesn't
  // matter -- only the fact of receipt does. EOS upstream means we
  // stop emitting too.
  //
  // WIRED, not merely DECLARED -- see the same note in
  // text-prompt-stage.cc. An optional iport left unconnected
  // (`{"src": "", "oport": 0}`, what the composer writes when an edge
  // is deleted but the port row stays) still counts in num_iports(),
  // so testing that alone waits on a port nothing will ever write and
  // this source loads NOTHING, silently.
  const bool has_trigger =
      ctx.num_iports() >= 1 && ctx.iport_connected(0);
  if (has_trigger) {
    auto t = co_await ctx.read(0);
    if (!t) {
      ctx.signal_done();
      co_return;
    }
  }

  const string url = _urls[_next++];
  auto payload = decode_url_(url);
  if (payload) {
    // Build the metadata beat BEFORE the image moves out of `payload` (it
    // reads the decoded dimensions), and emit it only when oport 1 is wired
    // -- reading EXIF costs a file read, so an unwired graph pays nothing.
    unique_ptr<BeatPayloadIntf> meta;
    if (ctx.num_oports() > 1 && ctx.has_consumers(1)) {
      meta = build_metadata_(url, *payload);
    }
    co_await ctx.write(0, std::move(payload));
    if (meta) { co_await ctx.write(1, std::move(meta)); }
  }

  // No TRIGGER => batch mode: signal done as soon as we've emitted (or
  // tried to emit) the last URL so the driver closes our oport. With a
  // trigger wired the driver tears us down when the upstream source
  // EOSes; we never self-signal in that case. Reads `has_trigger` and
  // not `num_iports() == 0` for the reason above -- a dangling port
  // would otherwise leave a batch run never closing its oport.
  if (_next >= _urls.size() && !has_trigger) {
    ctx.signal_done();
  }
}

VPIPE_REGISTER_STAGE(LoadImageStage)
VPIPE_REGISTER_SPEC(LoadImageStage, kSpec)

}

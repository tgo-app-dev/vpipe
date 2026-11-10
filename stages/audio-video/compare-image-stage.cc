#include "stages/audio-video/compare-image-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/vpipe-format.h"
#include "generative-models/generative-model-manager.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "pipeline/stage-spec.h"
#include "stages/model-memory.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>
}

using namespace std;

namespace vpipe {

namespace {

// Parse "#RRGGBB" / "RRGGBB" into r,g,b bytes. Defaults to BLACK on a
// malformed value -- a comparison pads with black so the padded border
// reads as "no image here", not as part of the picture (image-resample
// defaults to 114 grey instead, matching the historical letterbox).
void
parse_hex_color_(string s, uint8_t* r, uint8_t* g, uint8_t* b)
{
  *r = *g = *b = 0;
  if (!s.empty() && s[0] == '#') { s = s.substr(1); }
  if (s.size() >= 6) {
    auto hx = [&](size_t off) -> long {
      return strtol(s.substr(off, 2).c_str(), nullptr, 16);
    };
    *r = static_cast<uint8_t>(hx(0));
    *g = static_cast<uint8_t>(hx(2));
    *b = static_cast<uint8_t>(hx(4));
  }
}

inline uint8_t
clamp_byte_(float v)
{
  if (v <= 0.0f)   { return 0; }
  if (v >= 255.0f) { return 255; }
  return static_cast<uint8_t>(lrintf(v));
}

constexpr ConfigKey kAttrs[] = {
  {.key = "input_normalized", .type = ConfigType::Bool,
   .doc = "F32 input in [0,1] vs [0,255]", .def_bool = true},
  {.key = "title", .type = ConfigType::String,
   .doc = "optional label shown in the web-ui compare picker; empty = "
          "the stage id", .def_str = ""},
  {.key = "pad_color", .type = ConfigType::String,
   .doc = "#RRGGBB fill for the padded border of whichever image is "
          "smaller than the common size", .def_str = "#000000"},
};
const PortSpec kIports[] = {
  {.name = "a", .doc = "image A: planar RGB [3,H,W] or RGBA [4,H,W] "
          "(shown over a checkerboard) TensorBeat (F32 or "
                       "U8). Optional -- unwired or silent shows black.",
   .type = &typeid(TensorBeatPayload),
   .tags = "rgb-frames", .clock_group = 0},
  {.name = "b", .doc = "image B: planar RGB [3,H,W] or RGBA [4,H,W] "
          "(shown over a checkerboard) TensorBeat (F32 or "
                       "U8). Optional -- unwired or silent shows black.",
   .type = &typeid(TensorBeatPayload),
   .tags = "rgb-frames", .clock_group = 1},
};
const StageSpec kSpec = {
  .type_name = "compare-image",
  .doc       = "Sink: pairs two images for comparison in its own web-ui "
               "view -- A only, B only, side-by-side (left/right or "
               "top/bottom) with a centre divider, or a draggable "
               "vertical / horizontal wipe. Zoom and pan stay in sync "
               "across both. Mismatched resolutions are matched to "
               "max(Wa,Wb) x max(Ha,Hb) with the pad policy (fit, "
               "centre, pad_color border). A missing input shows black. "
               "0 oports.",
  .display_name = "Compare Images",
  .category  = StageCategory::Visual,
  .iports    = kIports,
  .oports    = {},
  .attrs     = kAttrs,
};

}  // namespace

static CompareImageChannel::Png
render_png_(const FFmpegLibraries* libs, const uint8_t pad[3],
            const CompareImageChannel::Picture& side, int w, int h);

CompareImageStage::CompareImageStage(const SessionContextIntf* s,
                                     string                    id,
                                     vector<InEdge>            iports,
                                     FlexData                  config)
  : TypedStage<CompareImageStage>(s, std::move(id), std::move(iports),
                                  std::move(config))
  , _libs(s->services()->ffmpeg_libraries())
{
  _input_normalized = attr_bool("input_normalized");
  _title            = attr_str("title");
  parse_hex_color_(attr_str("pad_color"), &_pad_r, &_pad_g, &_pad_b);

  allocate_oports(spec().oports.size());   // sink: 0 oports
  _channel = std::make_shared<CompareImageChannel>();
  // The libraries are the session's and outlive the stage; the colour
  // is copied. Nothing of the stage's is captured: the channel, and a
  // viewer rendering through it, can outlive the stage.
  {
    const FFmpegLibraries* libs = _libs;
    const std::array<uint8_t, 3> pad{_pad_r, _pad_g, _pad_b};
    _render = std::make_shared<const CompareImageChannel::Render>(
        [libs, pad](const CompareImageChannel::Picture& side, int w,
                    int h) {
          return render_png_(libs, pad.data(), side, w, h);
        });
  }
  if (_libs == nullptr ||
      _libs->avcodec().api.find_encoder_by_name("png") == nullptr) {
    session()->warn(fmt(
        "compare-image('{}'): PNG encoder unavailable; view disabled",
        this->id()));
  }

  session()->info(fmt(
      "compare-image('{}'): image A/B comparison view", this->id()));
}

CompareImageStage::~CompareImageStage()
{
  teardown_();
}

const StageSpec&
CompareImageStage::spec() const noexcept
{
  return kSpec;
}

// Only the DESTRUCTOR closes the channel. Running out of input is not a
// reason to: a comparison is a still result, and the last published pair
// stays worth looking at after the pipeline has drained -- closing here
// would drop a mounted panel back to "waiting" the instant the images it
// was asked to compare became final. A live video preview closes on
// teardown because a stopped stream has nothing left to show; this is
// the opposite case.
void
CompareImageStage::teardown_()
{
  if (_torn) { return; }
  _torn = true;
  if (_channel) { _channel->close(); }
}

void
CompareImageStage::resolve_roles_(RuntimeContext& ctx)
{
  if (_roles_resolved) { return; }
  _want_a = ctx.num_iports() >= 1 && ctx.iport_connected(0);
  _want_b = ctx.num_iports() >= 2 && ctx.iport_connected(1);
  _roles_resolved = true;
  session()->info(fmt(
      "compare-image('{}'): roles a={} b={}", this->id(), _want_a, _want_b));
}

// The transparency backdrop, in the convention every image viewer
// uses. OVER A CHECKERBOARD RATHER THAN A FLAT COLOUR, because this is
// a comparison view: flattening onto white would make a transparent
// region indistinguishable from a white one, so two images that differ
// ONLY in their alpha would be shown as identical -- which is the one
// thing a comparison must never do.
static std::uint8_t
checker_(int x, int y)
{
  return ((x / 8) + (y / 8)) % 2 == 0 ? 0xFF : 0xCC;
}

bool
CompareImageStage::unpack_(const TensorBeat& tb, Side& side) const
{
  if (tb.shape.size() != 3
      || (tb.shape[0] != 3 && tb.shape[0] != 4)) {
    return false;
  }
  const int CH = static_cast<int>(tb.shape[0]);
  const int H = static_cast<int>(tb.shape[1]);
  const int W = static_cast<int>(tb.shape[2]);
  if (H <= 0 || W <= 0) { return false; }

  const size_t plane   = static_cast<size_t>(H) * W;
  const size_t expect  = static_cast<size_t>(CH) * plane;
  auto                  rgb = make_shared<vector<uint8_t>>(3 * plane);
  vector<uint8_t>&      out = *rgb;

  // alpha in [0,1] for pixel i, or 1 when the beat carries none.
  auto blend = [&](size_t i, int c, float v, float a01) {
    if (CH == 3) { return clamp_byte_(v); }
    const int x = static_cast<int>(i % static_cast<size_t>(W));
    const int y = static_cast<int>(i / static_cast<size_t>(W));
    (void)c;
    return clamp_byte_(v * a01 +
                       static_cast<float>(checker_(x, y)) * (1.0f - a01));
  };

  if (tb.dtype == TensorBeat::DType::U8) {
    const uint8_t*         src = nullptr;
    AlignedVector<uint8_t> tmp;
    if (tb.is_contiguous() && tb.byte_size() == expect) {
      src = tb.as_u8();
    } else {
      tmp = tb.materialize_contiguous();
      src = tmp.data();
    }
    for (size_t i = 0; i < plane; ++i) {
      const float a01 =
          CH == 4 ? static_cast<float>(src[3 * plane + i]) / 255.0f : 1.0f;
      for (int c = 0; c < 3; ++c) {
        out[i * 3 + c] =
            blend(i, c, static_cast<float>(src[c * plane + i]), a01);
      }
    }
  } else if (tb.dtype == TensorBeat::DType::F32) {
    const float*         src = nullptr;
    AlignedVector<float> tmp;
    if (tb.is_contiguous() && tb.byte_size() == expect * sizeof(float)) {
      src = tb.as_f32();
    } else {
      tmp = tb.materialize_contiguous_as<float>();
      src = tmp.data();
    }
    const float scale = _input_normalized ? 255.0f : 1.0f;
    for (size_t i = 0; i < plane; ++i) {
      const float a01 =
          CH == 4 ? clamp_byte_(src[3 * plane + i] * scale) / 255.0f : 1.0f;
      for (int c = 0; c < 3; ++c) {
        out[i * 3 + c] = blend(i, c, src[c * plane + i] * scale, a01);
      }
    }
  } else {
    return false;   // unsupported dtype: keep whatever we had
  }

  side.rgb   = std::move(rgb);
  side.w     = W;
  side.h     = H;
  side.valid = true;
  return true;
}

// Draw `src` at the common `dst_w` x `dst_h` into rows `linesize` apart:
// copied verbatim when it is that size, else scaled to the largest fit
// that keeps its aspect, centred, on a `pad` border. What the stage's
// canvas held before -- now drawn straight into the encoder's frame, so a
// render keeps one copy of the picture rather than two.
static void
fit_into_(const CompareImageChannel::Picture& src, int dst_w, int dst_h,
          const uint8_t pad[3], uint8_t* dst, int linesize)
{
  const vector<uint8_t>& rgb = *src.rgb;
  // Already the target size -> verbatim copy, no resample. This is the
  // common case (both inputs agree) and must not go through the
  // interpolator, which would soften a pixel-exact image.
  if (src.width == dst_w && src.height == dst_h) {
    for (int y = 0; y < dst_h; ++y) {
      std::memcpy(dst + static_cast<size_t>(y) * linesize,
                  rgb.data() + static_cast<size_t>(y) * dst_w * 3,
                  static_cast<size_t>(dst_w) * 3);
    }
    return;
  }
  // Fill with pad_color first; the fitted image overwrites its box.
  for (int y = 0; y < dst_h; ++y) {
    uint8_t* row = dst + static_cast<size_t>(y) * linesize;
    for (int x = 0; x < dst_w; ++x) {
      row[x * 3 + 0] = pad[0];
      row[x * 3 + 1] = pad[1];
      row[x * 3 + 2] = pad[2];
    }
  }

  // pad policy: largest fit preserving aspect ratio, centred.
  const double sx = static_cast<double>(dst_w) / src.width;
  const double sy = static_cast<double>(dst_h) / src.height;
  const double s  = std::min(sx, sy);
  int fit_w = static_cast<int>(std::lround(src.width * s));
  int fit_h = static_cast<int>(std::lround(src.height * s));
  fit_w = std::clamp(fit_w, 1, dst_w);
  fit_h = std::clamp(fit_h, 1, dst_h);
  const int off_x = (dst_w - fit_w) / 2;
  const int off_y = (dst_h - fit_h) / 2;

  // Bilinear sample, matching image-resample's default algorithm.
  for (int y = 0; y < fit_h; ++y) {
    const double fy = (y + 0.5) * src.height / fit_h - 0.5;
    const int    y0 = static_cast<int>(std::floor(fy));
    const double wy = fy - y0;
    const int    y0c = std::clamp(y0, 0, src.height - 1);
    const int    y1c = std::clamp(y0 + 1, 0, src.height - 1);
    uint8_t* drow =
        dst + static_cast<size_t>(y + off_y) * linesize + off_x * 3;
    for (int x = 0; x < fit_w; ++x) {
      const double fx = (x + 0.5) * src.width / fit_w - 0.5;
      const int    x0 = static_cast<int>(std::floor(fx));
      const double wx = fx - x0;
      const int    x0c = std::clamp(x0, 0, src.width - 1);
      const int    x1c = std::clamp(x0 + 1, 0, src.width - 1);
      for (int c = 0; c < 3; ++c) {
        const double p00 =
            rgb[(static_cast<size_t>(y0c) * src.width + x0c) * 3 + c];
        const double p01 =
            rgb[(static_cast<size_t>(y0c) * src.width + x1c) * 3 + c];
        const double p10 =
            rgb[(static_cast<size_t>(y1c) * src.width + x0c) * 3 + c];
        const double p11 =
            rgb[(static_cast<size_t>(y1c) * src.width + x1c) * 3 + c];
        const double top = p00 + (p01 - p00) * wx;
        const double bot = p10 + (p11 - p10) * wx;
        drow[x * 3 + c] =
            clamp_byte_(static_cast<float>(top + (bot - top) * wy));
      }
    }
  }
}

// A viewer's render of one side: an encoder, its frame and its packet
// made for this picture and FREED with it. The stage kept all three for
// the run -- a frame of the whole picture and FFmpeg's packet sized for
// it, 250 MiB at 42 MP -- and encoded on the pipeline's thread whether
// or not anything was watching.
static CompareImageChannel::Png
render_png_(const FFmpegLibraries* libs, const uint8_t pad[3],
            const CompareImageChannel::Picture& side, int w, int h)
{
  if (libs == nullptr || side.rgb == nullptr || w <= 0 || h <= 0) {
    return nullptr;
  }
  const auto& cod = libs->avcodec().api;
  const auto& utl = libs->avutil().api;
  const AVCodec* codec = cod.find_encoder_by_name("png");
  if (codec == nullptr) { return nullptr; }
  AVCodecContext* ctx = cod.alloc_context3(codec);
  AVFrame*        frm = utl.frame_alloc();
  AVPacket*       pkt = cod.packet_alloc();
  CompareImageChannel::Png out;
  if (ctx != nullptr && frm != nullptr && pkt != nullptr) {
    ctx->width     = w;
    ctx->height    = h;
    ctx->pix_fmt   = AV_PIX_FMT_RGB24;
    ctx->time_base = AVRational{1, 1};
    frm->format = AV_PIX_FMT_RGB24;
    frm->width  = w;
    frm->height = h;
    if (cod.open2(ctx, codec, nullptr) >= 0 &&
        utl.frame_get_buffer(frm, 32) >= 0) {
      fit_into_(side, w, h, pad, frm->data[0], frm->linesize[0]);
      frm->pts = 0;
      // PNG makes ONE packet a frame, in the encoder's own buffer (kept
      // in the context) copied out into a refcounted one. The frame and
      // the context go before the PNG is copied out of that packet: with
      // them held, a render peaked at four buffers of the picture's size
      // (MEASURED 4.1x the picture, incompressible content), three now.
      const int rc = cod.send_frame(ctx, frm) >= 0
                         ? cod.receive_packet(ctx, pkt) : -1;
      utl.frame_free(&frm);
      cod.free_context(&ctx);
      if (rc >= 0 && pkt->size > 0) {
        out = make_shared<const vector<uint8_t>>(pkt->data,
                                                 pkt->data + pkt->size);
      }
    }
  }
  if (pkt != nullptr) { cod.packet_free(&pkt); }
  if (frm != nullptr) { utl.frame_free(&frm); }
  if (ctx != nullptr) { cod.free_context(&ctx); }
  return out;
}

void
CompareImageStage::publish_()
{
  if (!_a.valid && !_b.valid) {
    _pub_w = 0;
    _pub_h = 0;
    _channel->publish(nullptr, nullptr, 0, 0);
    return;
  }

  // Common size = the larger extent on each axis, so neither image is
  // ever downscaled (a comparison must not lose detail it was given).
  const int w = std::max(_a.valid ? _a.w : 0, _b.valid ? _b.w : 0);
  const int h = std::max(_a.valid ? _a.h : 0, _b.valid ? _b.h : 0);
  if (w <= 0 || h <= 0) { return; }
  _pub_w = w;
  _pub_h = h;

  auto pic = [](const Side& s) {
    return s.valid ? CompareImageChannel::Picture{s.rgb, s.w, s.h}
                   : CompareImageChannel::Picture{};
  };
  _channel->publish_pictures(pic(_a), pic(_b), w, h, _render);
}

std::string
CompareImageStage::scratch_label_() const
{
  return fmt("compare-image:{}", this->id())();
}

std::size_t
CompareImageStage::pair_bytes(int wa, int ha, int wb, int hb, int w, int h)
{
  const std::size_t sides = (std::size_t)3 *
      ((std::size_t)std::max(0, wa) * std::max(0, ha) +
       (std::size_t)std::max(0, wb) * std::max(0, hb));
  if (w <= 0 || h <= 0) { return sides; }
  // A render's peak (render_png_): the frame (rows padded), the PNG
  // encoder's buffer -- a deflate bound a row plus each IDAT's 12 bytes,
  // and its minimum -- and the packet it is copied into; then, the frame
  // and the encoder gone, the packet and the PNG copied out of it. Beside
  // that the other side's PNG, at most a packet, and a quarter of the
  // picture and a MiB for FFmpeg's and zlib's own, which grows with the
  // picture. MEASURED, a pair of noise (the worst a PNG can be), the
  // second side padded: 4.05x the common picture at 1024 x 768, 4.14x
  // at 4096 x 3072, against this bound's 4.84x / 4.31x.
  const std::size_t row   = (std::size_t)w * 3 + 1;
  const std::size_t raw   = (std::size_t)w * 3 * h;
  const std::size_t frame = ((std::size_t)w * 3 + 128) * h;
  const std::size_t pkt   = 32768 + (std::size_t)h * (row + row / 2048 + 64);
  return sides + frame + 3 * pkt + raw / 4 + ((std::size_t)1 << 20);
}

std::vector<ResourceClaim>
CompareImageStage::declare_resources() const
{
  return model_memory::scratch_claims(
      scratch_label_(), pair_bytes(1024, 1024, 1024, 1024, 1024, 1024),
      std::string_view());
}

StageMemory
CompareImageStage::declare_memory() const
{
  StageMemory m;
  m.scratch = _scratch_stated > 0
                  ? _scratch_stated
                  : pair_bytes(1024, 1024, 1024, 1024, 1024, 1024);
  return m;
}

void
CompareImageStage::revise_scratch_(std::size_t bytes)
{
  if (bytes == 0 || bytes == _scratch_stated || session() == nullptr ||
      session()->services() == nullptr) {
    return;
  }
  _scratch_stated = bytes;
  if (auto* mgr = session()->services()->generative_model_manager()) {
    mgr->revise_scratch(scratch_label_(), bytes);
  }
  StageMemory m = declare_memory();
  m.scratch = bytes;
  revise_memory(m);
}

Job
CompareImageStage::process(RuntimeContext& ctx)
{
  ++_process_calls;
  resolve_roles_(ctx);

  if (!_want_a && !_want_b) {
    // Nothing wired: publish the empty pair once so a mounted view shows
    // black rather than waiting, then retire. The channel stays OPEN
    // (see teardown_) so the panel reports the empty result rather than
    // dropping back to "waiting".
    publish_();
    ctx.signal_done();
    co_return;
  }

  // Wake on whichever input arrives; both are optional and they run in
  // separate clock groups, so blocking on one would stall the other.
  vector<unsigned> ports;
  if (_want_a) { ports.push_back(0); }
  if (_want_b) { ports.push_back(1); }

  // WAIT ON THE LIVE INPUTS ONLY. read_any treats a drained-and-closed
  // port as perpetually ready, so keeping one in the wait set makes the
  // wait return AT ONCE -- and while the other input has not produced
  // yet, this stage then wakes, drains nothing, finds all_eos false
  // because that other input is still live, and re-arms immediately.
  //
  // That spin is not merely wasteful. It occupies a worker for as long
  // as the live producer takes to speak, and under enough CPU
  // oversubscription it can starve the very producer it is waiting for
  // -- so the pair never publishes AT ALL. It is the one input wired
  // and silent case, i.e. exactly what
  // `one_side_only_leaves_other_black` covers; that test reached a
  // 10-SECOND deadline about one run in 100-150 under a 12-way load,
  // which is what a starved producer looks like from the outside. A
  // flat sleep could not distinguish it from "slow" and reported it as
  // rare test noise for a long time.
  //
  // An empty live set means every wired input is finished, and the
  // retirement check below is what acts on that -- so the wait is
  // simply skipped rather than being asked to block on nothing.
  vector<unsigned> live;
  for (unsigned p : ports) {
    if (!ctx.eos(p)) { live.push_back(p); }
  }
  if (!live.empty()) { co_await ctx.read_any(live); }

  // Drain both backlogs, keeping only the newest frame per side -- an
  // unread backlog is still consumed so the producer isn't
  // back-pressured.
  bool changed = false;
  for (unsigned p : ports) {
    const uint32_t avail = ctx.backlog(p);
    for (uint32_t i = 0; i < avail; ++i) {
      auto beat = co_await ctx.read(p);
      if (!beat) { break; }
      const auto* tb = dynamic_cast<const TensorBeatPayload*>(beat.get());
      if (tb == nullptr) { continue; }
      // The pair this beat makes, in the plan before its picture is.
      if (tb->shape.size() == 3) {
        const int bh = (int)tb->shape[1], bw = (int)tb->shape[2];
        const Side& o = p == 0 ? _b : _a;
        const int ow = o.valid ? o.w : 0, oh = o.valid ? o.h : 0;
        revise_scratch_(pair_bytes(bw, bh, ow, oh, std::max(bw, ow),
                                   std::max(bh, oh)));
      }
      if (unpack_(*tb, p == 0 ? _a : _b)) { changed = true; }
    }
  }
  if (changed) { publish_(); }

  // read_any treats a closed port as perpetually ready, so a loop that
  // only re-armed would spin once every input is drained and closed.
  bool all_eos = true;
  for (unsigned p : ports) {
    if (!ctx.eos(p)) { all_eos = false; break; }
  }
  if (all_eos || ctx.stop_requested()) {
    // Retire, but leave the channel open holding the final pair -- the
    // comparison is the output, and it outlives the inputs that made it.
    ctx.signal_done();
    co_return;
  }
  co_return;
}

VPIPE_REGISTER_STAGE(CompareImageStage)
VPIPE_REGISTER_SPEC(CompareImageStage, kSpec)

}

#include "stages/audio-video/alpha-mix-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "pipeline/runtime-context.h"
#include "pipeline/stage-spec.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace vpipe {

namespace {

// Parse "#RRGGBB" / "RRGGBB". Defaults to WHITE on a malformed value,
// which is the flatten colour the rest of this tree already uses --
// save-image, the diffusion conditioner and vae-encode all composite
// onto white, and a compositor that disagreed with them would make the
// same picture look different depending on which stage flattened it.
void
parse_hex_color_(string s, uint8_t* r, uint8_t* g, uint8_t* b)
{
  *r = *g = *b = 255;
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

constexpr SpecExtra kOutputChoices[] = {
  {"choices", "rgb,rgba"},
};

constexpr ConfigKey kAttrs[] = {
  {.key = "output", .type = ConfigType::String, .required = false,
   .doc = "rgb | rgba. `rgba` keeps the composite's own alpha; `rgb` "
          "flattens it onto `background` afterwards. Unset => rgb, "
          "because a composite is usually the step that FINISHES an "
          "image and [3,H,W] is what every consumer here reads",
   .def_str = "rgb", .extra = kOutputChoices},
  {.key = "background", .type = ConfigType::String, .required = false,
   .doc = "#RRGGBB shown wherever the composite is still transparent. "
          "Only read when output is rgb -- with rgba the transparency "
          "is kept rather than filled. Unset => #FFFFFF, matching what "
          "save-image and the diffusion conditioner flatten onto, so "
          "one picture does not change colour depending on which stage "
          "flattened it",
   .def_str = "#FFFFFF"},
  {.key = "input_normalized", .type = ConfigType::Bool, .required = false,
   .doc = "F32 input in [0,1] vs [0,255]. U8 is unambiguous and ignores "
          "this",
   .def_bool = true},
};

const PortSpec kIports[] = {
  {.name = "front",
   .doc = "the image composited ON TOP: planar RGB [3,H,W] or RGBA "
          "[4,H,W] TensorBeat (F32 or U8). Three channels means opaque",
   .type = &typeid(TensorBeatPayload),
   .tags = "rgb-frames", .clock_group = 0},
  {.name = "back",
   .doc = "the image composited UNDER: same formats, same size as "
          "`front`",
   .type = &typeid(TensorBeatPayload),
   .tags = "rgb-frames", .clock_group = 0},
  {.name = "alpha",
   .doc = "OPTIONAL mask in create-mask's own output format -- [1,H,W] "
          "or [H,W], U8 0..255 or F32 coverage in [0,1], at the images' "
          "size. It MULTIPLIES `front`'s alpha, so against an RGB front "
          "it simply IS the alpha and against an RGBA one it restricts "
          "where the front shows without discarding the transparency it "
          "already had",
   .type = &typeid(TensorBeatPayload),
   .tags = "mask-frames", .clock_group = 0},
};

const PortSpec kOports[] = {
  {.name = "image",
   .doc = "the composite: planar [3,H,W] or [4,H,W] per `output`, in "
          "`front`'s dtype",
   .type = &typeid(TensorBeatPayload),
   .tags = "rgb-frames", .clock_group = 0},
};

const StageSpec kSpec = {
  .type_name = "alpha-mix",
  .doc       = "Composites `front` over `back` (Porter-Duff over), with "
               "an optional mask on the front's alpha. Either input may "
               "be RGB or RGBA; the result is emitted as RGBA or "
               "flattened onto a background colour. The mask port takes "
               "create-mask's output directly.",
  .display_name = "Alpha Mix",
  .category  = StageCategory::Visual,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

AlphaMixStage::AlphaMixStage(const SessionContextIntf* session,
                             string                    id,
                             vector<InEdge>            iports,
                             FlexData                  config)
  : TypedStage<AlphaMixStage>(session, std::move(id), std::move(iports),
                              std::move(config))
{
  const string out = attr_str("output");
  _rgba_out = out == "rgba";
  if (!out.empty() && out != "rgb" && out != "rgba") {
    fail_config(fmt(
        "AlphaMixStage('{}'): output must be \"rgb\" or \"rgba\" (got "
        "\"{}\")", this->id(), out));
  }
  const string bg = attr_str("background");
  parse_hex_color_(bg.empty() ? string("#FFFFFF") : bg, &_bg_r, &_bg_g,
                   &_bg_b);
  _input_normalized = attr_bool("input_normalized");
  allocate_oports(spec().oports.size());
}

const StageSpec&
AlphaMixStage::spec() const noexcept
{
  return kSpec;
}

AlphaMixStage::Plane
AlphaMixStage::read_image_(const TensorBeat& tb) const
{
  Plane p;
  if (tb.shape.size() != 3) { return p; }
  const int C = static_cast<int>(tb.shape[0]);
  if (C != 3 && C != 4) { return p; }
  p.h = static_cast<int>(tb.shape[1]);
  p.w = static_cast<int>(tb.shape[2]);
  if (p.w <= 0 || p.h <= 0) { return p; }
  const size_t hw = static_cast<size_t>(p.h) * p.w;
  p.rgb.assign(3 * hw, 0.0f);
  p.a.assign(hw, 1.0f);
  p.had_alpha = C == 4;

  if (tb.dtype == TensorBeat::DType::U8) {
    const auto src = tb.materialize_contiguous();
    if (src.size() < static_cast<size_t>(C) * hw) { return p; }
    for (size_t i = 0; i < 3 * hw; ++i) {
      p.rgb[i] = static_cast<float>(src[i]) / 255.0f;
    }
    if (C == 4) {
      for (size_t i = 0; i < hw; ++i) {
        p.a[i] = static_cast<float>(src[3 * hw + i]) / 255.0f;
      }
    }
  } else if (tb.dtype == TensorBeat::DType::F32) {
    const auto src = tb.materialize_contiguous_as<float>();
    if (src.size() < static_cast<size_t>(C) * hw) { return p; }
    const float s = _input_normalized ? 1.0f : 1.0f / 255.0f;
    for (size_t i = 0; i < 3 * hw; ++i) { p.rgb[i] = src[i] * s; }
    if (C == 4) {
      for (size_t i = 0; i < hw; ++i) { p.a[i] = src[3 * hw + i] * s; }
    }
  } else {
    return p;
  }
  for (float& v : p.a) { v = std::min(1.0f, std::max(0.0f, v)); }
  p.ok = true;
  return p;
}

AlphaMixStage::Plane
AlphaMixStage::read_mask_(const TensorBeat& tb, int w, int h) const
{
  Plane p;
  // create-mask emits [1,H,W]; [H,W] is the same thing written shorter
  // and its `in-mask` port accepts both, so this does too.
  int mh = 0, mw = 0;
  if (tb.shape.size() == 3 && tb.shape[0] == 1) {
    mh = static_cast<int>(tb.shape[1]);
    mw = static_cast<int>(tb.shape[2]);
  } else if (tb.shape.size() == 2) {
    mh = static_cast<int>(tb.shape[0]);
    mw = static_cast<int>(tb.shape[1]);
  } else {
    return p;
  }
  if (mw != w || mh != h) { return p; }
  const size_t hw = static_cast<size_t>(h) * w;
  p.a.assign(hw, 1.0f);
  p.w = w;
  p.h = h;
  if (tb.dtype == TensorBeat::DType::U8) {
    const auto src = tb.materialize_contiguous();
    if (src.size() < hw) { return p; }
    for (size_t i = 0; i < hw; ++i) {
      p.a[i] = static_cast<float>(src[i]) / 255.0f;
    }
  } else if (tb.dtype == TensorBeat::DType::F32) {
    const auto src = tb.materialize_contiguous_as<float>();
    if (src.size() < hw) { return p; }
    // A mask is COVERAGE, and `input_normalized` deliberately does NOT
    // apply to it: create-mask's f32 mode is documented as [0,1]
    // whatever the images around it are, so scaling this by 255 in a
    // 0..255 f32 graph would turn every mask into nearly zero -- an
    // invisible front, which reads as "the mask did nothing".
    for (size_t i = 0; i < hw; ++i) { p.a[i] = src[i]; }
  } else {
    return p;
  }
  for (float& v : p.a) { v = std::min(1.0f, std::max(0.0f, v)); }
  p.ok = true;
  return p;
}

Job
AlphaMixStage::process(RuntimeContext& ctx)
{
  auto fb = co_await ctx.read(0);
  if (!fb) { ctx.signal_done(); co_return; }
  auto bb = co_await ctx.read(1);
  if (!bb) { ctx.signal_done(); co_return; }

  const auto* ftb = dynamic_cast<const TensorBeatPayload*>(fb.get());
  const auto* btb = dynamic_cast<const TensorBeatPayload*>(bb.get());
  if (ftb == nullptr || btb == nullptr) {
    session()->warn(fmt(
        "AlphaMixStage('{}'): expected image TensorBeats on front/back; "
        "dropping beat", this->id()));
    co_return;
  }
  const Plane F = read_image_(*ftb);
  const Plane B = read_image_(*btb);
  if (!F.ok || !B.ok) {
    session()->warn(fmt(
        "AlphaMixStage('{}'): front/back must be planar U8/F32 RGB "
        "[3,H,W] or RGBA [4,H,W]; got {} and {}; dropping beat",
        this->id(), fb->describe(), bb->describe()));
    co_return;
  }
  if (F.w != B.w || F.h != B.h) {
    // REFUSED rather than fitted. A compositor that silently scaled one
    // layer would move every pixel of it against the other, and
    // image-resample exists to say how that should happen.
    session()->warn(fmt(
        "AlphaMixStage('{}'): front is {}x{} and back is {}x{} -- a "
        "composite needs one coordinate space; put an image-resample "
        "on whichever should change. Dropping beat",
        this->id(), F.w, F.h, B.w, B.h));
    co_return;
  }

  const size_t hw = static_cast<size_t>(F.h) * F.w;
  vector<float> fa = F.a;
  if (ctx.num_iports() > 2 && ctx.iport_connected(2)) {
    auto mb = co_await ctx.read(2);
    const auto* mtb = mb ? dynamic_cast<const TensorBeatPayload*>(mb.get())
                         : nullptr;
    if (mtb == nullptr) {
      session()->warn(fmt(
          "AlphaMixStage('{}'): the alpha iport is wired but carried no "
          "mask; compositing on the front's own alpha", this->id()));
    } else {
      const Plane M = read_mask_(*mtb, F.w, F.h);
      if (!M.ok) {
        session()->warn(fmt(
            "AlphaMixStage('{}'): the mask must be [1,H,W] or [H,W] at "
            "{}x{} (U8 0..255 or F32 0..1); got {}; compositing on the "
            "front's own alpha", this->id(), F.w, F.h, mb->describe()));
      } else {
        // MULTIPLY, which is the one rule that serves both cases -- see
        // the header. Against an RGB front (alpha 1) it IS the mask.
        for (size_t i = 0; i < hw; ++i) { fa[i] *= M.a[i]; }
      }
    }
  }

  // ---- Porter-Duff over -----------------------------------------------
  //
  // Accumulated PREMULTIPLIED, which is the form in which the blend is a
  // plain weighted sum, then divided back out once at the end.
  //
  // THE TERM THAT GETS DROPPED is the back's own alpha: its weight is
  // `aB * (1 - aF)`, not `(1 - aF)`. Weighting by `(1 - aF)` alone makes
  // a transparent BACK behave as an opaque one, which is invisible in
  // the ordinary case (an opaque back, aB == 1, where the two agree) and
  // only shows when both layers are partly transparent. Pinned by
  // alpha_mix_stage.the_back_alpha_weights_the_back.
  const int OC = _rgba_out ? 4 : 3;
  TensorBeat out;
  out.dtype = ftb->dtype;
  out.shape = { OC, F.h, F.w };
  out.sideband = ftb->sideband;
  const bool u8 = ftb->dtype == TensorBeat::DType::U8;
  const float wscale = u8 ? 255.0f : (_input_normalized ? 1.0f : 255.0f);
  out.resize_contiguous(static_cast<size_t>(OC) * hw);
  uint8_t* d8 = out.bytes_();
  float*   df = reinterpret_cast<float*>(d8);
  const float bg[3] = { _bg_r / 255.0f, _bg_g / 255.0f, _bg_b / 255.0f };

  for (size_t i = 0; i < hw; ++i) {
    const float af = fa[i];
    const float ab = B.a[i];
    const float wb = ab * (1.0f - af);
    const float ao = af + wb;
    for (int c = 0; c < 3; ++c) {
      const float pf = F.rgb[static_cast<size_t>(c) * hw + i] * af;
      const float pb = B.rgb[static_cast<size_t>(c) * hw + i] * wb;
      float v;
      if (_rgba_out) {
        // Back to unpremultiplied for the emitted beat. At ao == 0
        // there is no colour to recover and any value is as good as
        // another; zero keeps it out of an unpremultiplied consumer's
        // arithmetic.
        v = ao > 1e-6f ? (pf + pb) / ao : 0.0f;
      } else {
        v = pf + pb + bg[c] * (1.0f - ao);
      }
      v = std::min(1.0f, std::max(0.0f, v)) * wscale;
      const size_t o = static_cast<size_t>(c) * hw + i;
      if (u8) { d8[o] = static_cast<uint8_t>(lrintf(v)); }
      else    { df[o] = v; }
    }
    if (_rgba_out) {
      const float v = std::min(1.0f, std::max(0.0f, ao)) * wscale;
      const size_t o = 3 * hw + i;
      if (u8) { d8[o] = static_cast<uint8_t>(lrintf(v)); }
      else    { df[o] = v; }
    }
  }
  co_await ctx.write(0, make_payload<TensorBeatPayload>(std::move(out)));
}

VPIPE_REGISTER_STAGE(AlphaMixStage)
VPIPE_REGISTER_SPEC(AlphaMixStage, kSpec)

}  // namespace vpipe

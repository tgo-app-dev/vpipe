#ifndef VPIPE_STAGES_CREATE_MASK_STAGE_H
#define VPIPE_STAGES_CREATE_MASK_STAGE_H

#include "common/ffmpeg-libraries.h"
#include "common/job.h"
#include "common/mask-editor-channel.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
}

namespace vpipe {

struct TensorBeat;

// Authors a mask -- by hand in its own GUI editor, or by passing one
// through from an input -- and emits it, or the reference image with it
// painted on top.
//
// THREE KINDS OF MASK, chosen by `mask_mode` and honoured end to end
// (the brush, the input unpack, the resample and the output all switch
// on it):
//   binary  two states. A sample is 0 or 255 and nothing between; the
//           brush is a hard disc and a resample is nearest-neighbour,
//           because an interpolated edge would invent a third state.
//   alpha   soft coverage 0..255. The brush's HARDNESS sets how the
//           value decays from the centre of the stroke to its rim, and
//           a resample is bilinear.
//   class   multi-class. A sample is a class INDEX in [0, classes),
//           index 0 being background. Nearest-neighbour again, for the
//           same reason as binary: an average of class 1 and class 3 is
//           not a class.
//
// The mask oport always carries ONE channel -- [1,H,W], tagged
// mask-frames -- whichever kind it is. `class_colors` is presentation
// only: it drives the editor and the overlay, never the index map that
// leaves the stage, so a palette can be changed without invalidating a
// consumer.
//
// TWO WAYS TO RUN.
//   interactive (default). The stage publishes the reference image and
//     the current mask to its editor panel and waits. On COMMIT it
//     emits exactly one beat, then goes back to waiting -- so the
//     stage is long-lived and one click means one beat. It does NOT
//     retire when its inputs reach EOS: a still reference image is the
//     normal case, and the editor must outlive it.
//   interactive: false. No GUI at all. The mask arrives on in-mask,
//     is overlaid / converted per the configuration, and is emitted as
//     it arrives. The stage retires when its inputs do.
//
// GEOMETRY. The mask canvas is `width` x `height`; either dimension
// left at 0 is inferred from the other against the reference image's
// aspect ratio, and with both at 0 the canvas matches the reference
// image (or, with no reference, the incoming mask). In the editor the
// canvas is STRETCHED over the reference image, so the two always
// register regardless of their resolutions. An overlay output is
// produced at the REFERENCE image's resolution with the mask stretched
// onto it; a mask output keeps the canvas resolution.
//
// iport 0: ref-image -- planar RGB TensorBeat [3,H,W] (F32 or U8).
//          Optional: the editor's background, the overlay's base. With
//          none, the editor paints over black.
// iport 1: in-mask   -- planar mask TensorBeat [1,H,W] (F32 or U8);
//          [H,W] is accepted too. Optional: seeds the editor, and is
//          the whole input in non-interactive mode.
// oport 0: the mask [1,H,W] (mask-frames), or the overlaid image
//          [3,H,W] (rgb-frames), per `output`.
//
// Config (FlexData object):
//   mask_mode        "binary" | "alpha" | "class"   (default "binary")
//   classes          class count incl. background   (default 3)
//   class_colors     comma-separated #RRGGBB        (default: built-in)
//   output           "mask" | "overlay"             (default "mask")
//   overlay_color    #RRGGBB for binary/alpha       (default "#ff3b30")
//   overlay_opacity  0..1                           (default 0.5)
//   output_dtype     "u8" | "f32"                   (default "u8")
//   width, height    mask canvas size, 0 = infer    (default 0)
//   interactive      run the editor                 (default true)
//   input_normalized F32 input in [0,1] vs [0,255]  (default true)
//   title            editor picker label            (default "")
class CreateMaskStage final
  : public TypedStage<CreateMaskStage>
  , public MaskEditorSource
{
public:
  static constexpr const char* kTypeName = "create-mask";

  CreateMaskStage(const SessionContextIntf* session,
                  std::string               id,
                  std::vector<InEdge>       iports,
                  FlexData                  config);
  ~CreateMaskStage() override;

  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  std::shared_ptr<MaskEditorChannel> mask_channel() const override
  { return _channel; }

  // Test-only inspectors.
  int  mask_width()  const noexcept { return _mw; }
  int  mask_height() const noexcept { return _mh; }
  bool have_ref()    const noexcept { return _ref_valid; }
  bool have_mask()   const noexcept { return !_mask.empty(); }
  MaskEditorChannel::Mode mode() const noexcept { return _mode; }

  // Resolve the configured canvas against a source size, the way
  // image-resample resolves its target: a dimension left at 0 is
  // inferred from the other so the source aspect ratio is preserved.
  void resolve_canvas_(int src_w, int src_h,
                       int* out_w, int* out_h) const noexcept;

private:
  void teardown_();

  void resolve_roles_(RuntimeContext& ctx);

  // Unpack a planar RGB TensorBeat [3,H,W] into `_ref` as packed RGB24.
  // Returns false (leaving the reference untouched) for an unsupported
  // dtype or a malformed shape.
  bool unpack_ref_(const TensorBeat& tb);

  // Unpack a mask TensorBeat [1,H,W] (or [H,W]) into `out` at `*w` x
  // `*h`, one byte per sample. In Class mode a value is a class index
  // whatever the dtype; otherwise an F32 value is coverage, scaled by
  // `input_normalized`.
  bool unpack_mask_(const TensorBeat& tb, std::vector<std::uint8_t>* out,
                    int* w, int* h) const;

  // Resample a one-channel mask. Nearest-neighbour in Binary and Class
  // mode (an interpolated sample would not be a state or a class);
  // bilinear in Alpha mode. A source already at the target size is
  // copied verbatim.
  void resample_mask_(const std::vector<std::uint8_t>& src, int sw, int sh,
                      int dw, int dh,
                      std::vector<std::uint8_t>* dst) const;

  // Paint the mask over a packed-RGB24 base at the base's resolution,
  // stretching the canvas to fit. Class 0 / value 0 paints nothing.
  void overlay_(const std::vector<std::uint8_t>& base, int bw, int bh,
                std::vector<std::uint8_t>* dst) const;

  // The colour a sample is painted with, and how strongly, at
  // `overlay_opacity`. Returns false when the sample paints nothing.
  bool sample_color_(std::uint8_t v, std::uint8_t* r, std::uint8_t* g,
                     std::uint8_t* b, float* a) const;

  // PNG-encode packed RGB24 (`comps` == 3) or GRAY8 (`comps` == 1).
  // Returns null on failure; the encoder is rebuilt when the geometry
  // changes, so the two callers do not share one context.
  struct PngEncoder {
    AVCodecContext* ctx   = nullptr;
    AVFrame*        frame = nullptr;
    int             w     = 0;
    int             h     = 0;
    bool            bad   = false;
  };
  MaskEditorChannel::Bytes
  encode_png_(PngEncoder* enc, const std::uint8_t* src, int w, int h,
              int comps);
  void free_encoder_(PngEncoder* enc);

  // Decode a committed PNG (any pixel format the browser produced) and
  // normalise it to one byte per sample at `*w` x `*h`. False on a
  // malformed or undecodable payload -- a bad commit is dropped, never
  // fatal.
  bool decode_commit_(const std::vector<std::uint8_t>& png,
                      std::vector<std::uint8_t>* out, int* w, int* h);

  // Bring the canvas up to date with what is now known about the
  // geometry, carrying an existing mask across a size change rather
  // than dropping it. `fallback_w/h` is the size to resolve against
  // when there is no reference image yet.
  void refresh_canvas_(int fallback_w, int fallback_h);

  // Push the current reference / mask / geometry to the editor.
  void publish_();

  // Build the output beat for the current mask. Null when there is
  // nothing to emit yet.
  std::unique_ptr<BeatPayloadIntf> make_output_() const;

  const FFmpegLibraries* _libs = nullptr;

  // Config.
  MaskEditorChannel::Mode _mode = MaskEditorChannel::Mode::Binary;
  int                     _classes = 3;
  std::vector<std::uint32_t> _colors;
  bool         _overlay_out      = false;
  float        _overlay_opacity  = 0.5f;
  bool         _f32_out          = false;
  int          _cfg_w = 0, _cfg_h = 0;
  bool         _interactive      = true;
  bool         _input_normalized = true;
  std::string  _title;

  // Reference image, packed RGB24 at its own size.
  std::vector<std::uint8_t> _ref;
  int  _rw = 0, _rh = 0;
  bool _ref_valid = false;

  // The mask being authored, one byte per sample at _mw x _mh.
  std::vector<std::uint8_t> _mask;
  int _mw = 0, _mh = 0;

  PngEncoder _rgb_png;
  PngEncoder _gray_png;
  AVPacket*  _pkt = nullptr;

  AVCodecContext* _dec     = nullptr;
  SwsContext*     _sws     = nullptr;
  bool            _dec_bad = false;

  std::uint64_t _seen_commit = 0;
  bool _roles_resolved = false;
  bool _want_ref       = false;
  bool _want_mask      = false;
  bool _torn           = false;
  bool _warned_idle    = false;

  std::shared_ptr<MaskEditorChannel> _channel;
};

}

#endif

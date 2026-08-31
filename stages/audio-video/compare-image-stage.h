#ifndef VPIPE_STAGES_COMPARE_IMAGE_STAGE_H
#define VPIPE_STAGES_COMPARE_IMAGE_STAGE_H

#include "common/compare-image-channel.h"
#include "common/ffmpeg-libraries.h"
#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
}

namespace vpipe {

struct TensorBeat;

// Sink stage that pairs two image inputs for side-by-side / wipe
// comparison in its own GUI view (ui/stages/compare-image/).
//
// It keeps the LATEST frame from each input and republishes the pair
// whenever either changes; there is no cadence and no encoder state.
// Both images are PNG-encoded (full quality -- a comparison is looked
// at, not streamed) and pushed through a CompareImageChannel.
//
// RESOLUTION MATCHING. The two inputs need not agree. The pair is
// published at the COMMON size max(Wa,Wb) x max(Ha,Hb), and an image
// smaller than that is fitted with the `pad` policy -- the same
// semantics as image-resample's `pad`: scale to the largest size that
// fits while preserving aspect ratio, centre it, and fill the remainder
// with `pad_color`. Matching server-side (rather than in the browser)
// is what lets the view treat A and B as one coordinate space, which is
// what makes the split modes pixel-exact and the zoom/pan sync trivial.
//
// A missing input is simply absent from the pair: the view shows black
// in its place, and black for both before anything arrives.
//
// iport 0: image A -- planar RGB TensorBeat [3,H,W] (F32 or U8).
// iport 1: image B -- same. Either may be left unwired.
// oports : none (sink).
//
// Config (FlexData object):
//   input_normalized (bool, default true) -- F32 input in [0,1] vs
//                    [0,255]
//   title            (string, default "") -- optional picker label;
//                    empty = the stage id.
//   pad_color        (string, default "#000000") -- fill for the padded
//                    border of the smaller image.
class CompareImageStage final
  : public TypedStage<CompareImageStage>
  , public CompareImageSource
{
public:
  static constexpr const char* kTypeName = "compare-image";

  CompareImageStage(const SessionContextIntf* session,
                    std::string               id,
                    std::vector<InEdge>       iports,
                    FlexData                  config);
  ~CompareImageStage() override;

  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  std::shared_ptr<CompareImageChannel> compare_channel() const override
  { return _channel; }

  // Test-only inspectors.
  int  common_width()  const noexcept { return _pub_w; }
  int  common_height() const noexcept { return _pub_h; }
  bool have_a()        const noexcept { return _a.valid; }
  bool have_b()        const noexcept { return _b.valid; }
  // How many times process() has run SINCE CONSTRUCTION. The wait set
  // covers only LIVE inputs precisely so this stays small while one
  // side is closed and the other has not spoken yet; a stage that waits
  // on a closed port instead re-arms in a tight loop, and this climbs
  // into the hundreds of thousands. That is the difference between
  // blocking and spinning, and it is not otherwise observable from
  // outside the stage.
  std::uint64_t process_calls() const noexcept { return _process_calls; }

private:
  // One input's latest frame, unpacked to packed RGB24 at its OWN size.
  // Padding to the common size happens at publish time, so a later
  // change to the other input re-fits this one without needing the
  // original TensorBeat.
  struct Side {
    std::vector<std::uint8_t> rgb;   // packed RGB24, w*h*3
    int  w = 0;
    int  h = 0;
    bool valid = false;
  };

  void resolve_roles_(RuntimeContext& ctx);

  // Unpack a planar RGB TensorBeat [3,H,W] into `side` as packed RGB24.
  // Returns false (leaving `side` untouched) for an unsupported dtype or
  // a malformed shape.
  bool unpack_(const TensorBeat& tb, Side& side) const;

  // Scale-and-centre `src` into a `dst_w` x `dst_h` packed-RGB24 buffer
  // using the pad policy; the border is `pad_color`. A source already at
  // the target size is copied verbatim (no resample).
  void pad_fit_(const Side& src, int dst_w, int dst_h,
                std::vector<std::uint8_t>* dst) const;

  // PNG-encode a packed-RGB24 buffer. Returns null on failure (the
  // encoder is built lazily and re-built when the size changes).
  CompareImageChannel::Png
  encode_png_(const std::vector<std::uint8_t>& rgb, int w, int h);

  void teardown_();

  // Re-fit both sides to the current common size and publish the pair.
  void publish_();

  const FFmpegLibraries* _libs = nullptr;

  // Config.
  bool         _input_normalized = true;
  std::string  _title;
  std::uint8_t _pad_r = 0, _pad_g = 0, _pad_b = 0;

  Side _a;
  Side _b;

  // Last published common size; the PNG encoder is keyed to it.
  int _pub_w = 0;
  int _pub_h = 0;

  AVCodecContext* _png     = nullptr;
  AVFrame*        _png_frm = nullptr;
  AVPacket*       _png_pkt = nullptr;
  int             _png_w   = 0;    // size the encoder was opened at
  int             _png_h   = 0;
  bool            _png_bad = false;

  bool _roles_resolved = false;
  bool _want_a = false;
  bool _want_b = false;
  bool _torn   = false;

  // Test-only bookkeeping; see process_calls().
  std::uint64_t _process_calls = 0;

  std::shared_ptr<CompareImageChannel> _channel;
};

}

#endif

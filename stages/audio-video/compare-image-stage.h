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

  // What the stage holds, in the plan: see pair_bytes().
  std::vector<ResourceClaim> declare_resources() const override;
  StageMemory declare_memory() const override;

  // The bytes a pair holds: each side's picture, packed RGB24 at its own
  // size, kept for the run -- and a viewer's render at the common size
  // (`w` x `h`): the frame it draws into, the PNG encoder's buffer and
  // its packet, sized for a picture that does not compress, the PNG
  // copied out, the other side's PNG kept. No signal says whether a
  // viewer will mount, so the render is counted whenever the stage runs.
  static std::size_t pair_bytes(int wa, int ha, int wb, int hb, int w,
                                int h);

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
  // One input's latest frame, unpacked to packed RGB24 at its OWN size,
  // and SHARED with the channel, which pads and encodes it only when a
  // viewer asks (CompareImageChannel::publish_pictures) -- so a later
  // change to the other input re-fits this one without the original
  // TensorBeat, and nothing is encoded for a graph nobody watches.
  struct Side {
    std::shared_ptr<const std::vector<std::uint8_t>> rgb;  // w*h*3
    int  w = 0;
    int  h = 0;
    bool valid = false;
  };

  void resolve_roles_(RuntimeContext& ctx);

  // Unpack a planar RGB TensorBeat [3,H,W] into `side` as packed RGB24.
  // Returns false (leaving `side` untouched) for an unsupported dtype or
  // a malformed shape.
  bool unpack_(const TensorBeat& tb, Side& side) const;

  void teardown_();

  // Publish the current pair at its common size, for a viewer to render.
  void publish_();

  // The plan's figure: two 1024^2 sides at launch, then each beat's real
  // pair, stated BEFORE its picture is unpacked.
  std::string scratch_label_() const;
  void revise_scratch_(std::size_t bytes);
  std::size_t _scratch_stated = 0;

  const FFmpegLibraries* _libs = nullptr;
  // What a viewer runs to turn a side into a PNG (render_png_ over the
  // libraries and the pad colour, nothing of the stage's).
  std::shared_ptr<const CompareImageChannel::Render> _render;

  // Config.
  bool         _input_normalized = true;
  std::string  _title;
  std::uint8_t _pad_r = 0, _pad_g = 0, _pad_b = 0;

  Side _a;
  Side _b;

  // Last published common size.
  int _pub_w = 0;
  int _pub_h = 0;

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

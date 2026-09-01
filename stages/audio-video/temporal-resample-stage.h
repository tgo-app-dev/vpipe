#ifndef VPIPE_STAGES_AUDIO_VIDEO_TEMPORAL_RESAMPLE_STAGE_H
#define VPIPE_STAGES_AUDIO_VIDEO_TEMPORAL_RESAMPLE_STAGE_H

#include "apple-silicon/tensor-beat.h"
#include "common/av-filter-graph.h"
#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {

// The temporal counterpart of `image-resample`: frames in at one rate,
// frames out at another, through a libavfilter chain.
//
// `image-resample` is the spatial half and this is the temporal one, and
// the two have the same shape of problem. Choosing an output rate is the
// easy part; what decides whether the result looks right is what happens
// to the information that does not survive the change -- which is
// ALIASING, and it is why this is a stage rather than a `stride` on
// something else. Decimating 60 fps to 24 by taking every other-and-a-
// half frame is nearest-neighbour sampling of a signal nobody
// band-limited: a wheel turns backwards, a pan judders, and the frames
// that are kept are each individually perfect. `method` is the filter
// applied BEFORE the decimation, and it is the whole point of the stage.
//
//   nearest  `fps` -- drop and duplicate. Cheapest, and what every
//            naive "every Nth frame" does. Aliases.
//   blend    `framerate` -- blend the two source frames an output frame
//            falls between. Cheap, and enough for small ratios.
//   average  `tmix` then `fps` -- box-average the source frames that
//            fall inside each output frame's interval, then decimate.
//            The temporal analogue of area-averaging a downscale, and
//            the right answer when the ratio is large. Reads as motion
//            blur, because that is what it is.
//   motion   `minterpolate` -- estimate motion and synthesise the
//            in-between frames. The one that helps when going UP in
//            rate, where there is no information to average.
//
// CLOCK DOMAINS -- read this before wiring it.
//
// A rate change means the stage does not emit one beat per beat it
// consumes, so its oport is on a clock of its own and the graph shows it
// as a crosser. That is correct and it is also a constraint: a feedback
// loop cannot close through a domain boundary.
//
// A STACKED input escapes this. When the whole time axis arrives in one
// beat ([T, 3, H, W] -- what `temporal-stack` builds and what a
// reference-conditioned model takes), the stage is 1:1 -- one clip in,
// one clip out -- so its oport shares the iport's clock and nothing is
// crossed.
//
// Which of the two it is has to be STATED (`stacked`), not sensed. The
// clock analysis runs at launch, before any beat exists, so there is
// nothing to sense yet; and a stage that guessed would move a domain
// boundary between runs of the same graph. A beat that contradicts the
// setting is warned about by name rather than reinterpreted.
//
//   iport0  frames   TensorBeatPayload, planar RGB, u8 or f32 -- the
//                    two `video-to-rgb` emits, and it defaults to f32.
//                    [3, H, W] per beat when `stacked` is false; ONE
//                    [T, 3, H, W] clip when it is true. The dtype is
//                    carried through: a chain runs in GBRP or GBRPF32
//                    and what leaves is what arrived.
//   oport0  frames   the same shape, at `output_fps`. Its clock group is
//                    the iport's when `stacked`, its own otherwise.
//
// The INPUT rate is resolved from the beat -- `fps` on a clip,
// `fps_num`/`fps_den` on frames -- else from `input_fps`, and a request
// with neither is refused. It is what the chain divides by: a rate
// guessed wrong resamples by the wrong ratio and produces a clip that is
// the wrong LENGTH, with every frame in it individually fine.
class TemporalResampleStage final : public TypedStage<TemporalResampleStage> {
public:
  static constexpr const char* kTypeName = "temporal-resample";

  TemporalResampleStage(const SessionContextIntf* session,
                        std::string               id,
                        std::vector<InEdge>       iports,
                        FlexData                  config);
  ~TemporalResampleStage() override;

  Job initialize(RuntimeContext& ctx) override;
  Job process(RuntimeContext& ctx) override;
  void reset_run_state() override;

  const StageSpec& spec() const noexcept override;

  // See the CLOCK DOMAINS note above: the oport shares the iport's
  // group only when the whole time axis rides in one beat.
  unsigned oport_clock_group(unsigned p) const noexcept override;

  // How a rate change is filtered. Public so the chain builder has a
  // test that needs neither FFmpeg nor a runtime.
  enum class Method { kNearest, kBlend, kAverage, kMotion };

  // The libavfilter chain for one request. `in_fps`/`out_fps` are the
  // resolved rates; `avg_frames` is the `tmix` window, 0 to derive it
  // from the ratio. Empty when the rates are equal and the method adds
  // nothing -- the caller then forwards the beat untouched, which is
  // the one case worth not paying for.
  static std::string chain_for(Method m, double in_fps, double out_fps,
                               int avg_frames);
  static const char* method_name(Method m) noexcept;

  // Test-only.
  Method        method()     const noexcept { return _method; }
  bool          stacked()    const noexcept { return _stacked; }
  double        output_fps() const noexcept { return _out_fps; }
  std::uint64_t emitted()    const noexcept { return _emitted; }

private:
  // Resolve the source rate for one beat: `input_fps` when configured,
  // else the beat's own sideband. 0 when neither said.
  double input_fps_of_(const TensorBeat& tb) const;
  // Open a graph for this geometry, if one is not already open for it.
  // False with a warned reason.
  bool ensure_graph_(int w, int h, double in_fps, int pix_fmt);
  // Drain the sink into `out`, appending one beat's worth of pixels per
  // frame -- BYTES, so one entry is 3 * w * h * element size. False on
  // a graph error (already warned).
  bool drain_(std::vector<std::vector<std::uint8_t>>* frames);
  // Feed one [3, H, W] plane triple. False (warned) on a push failure.
  bool push_frame_(const std::uint8_t* rgb, int w, int h, std::int64_t pts);

  // The seed sideband with the RATE keys rewritten: an output beat
  // carries the rate it actually has, not the one it was resampled
  // from. Everything else the source said (camera_name, model
  // provenance) rides along -- the pixels are still that source's.
  // Both advance `_out_index`, which is what the output timestamps are
  // counted from.
  FlexData frame_sideband_();
  FlexData clip_sideband_(int frames);

  double _out_fps    = 0.0;
  double _in_fps_cfg = 0.0;
  Method _method     = Method::kNearest;
  int    _avg_frames = 0;
  bool   _stacked    = false;

  // The open graph and the geometry it was built for. A frame stream
  // keeps one for the whole run; a stacked clip builds and flushes one
  // per beat, because a flush ends the graph.
  AvFilterGraph _graph;
  int           _gw = 0, _gh = 0;
  // The pixel format the open graph was built for, so a source that
  // switches dtype rebuilds rather than reinterpreting float as bytes.
  int           _gfmt = 0;
  int           _gesz = 1;         // bytes per component
  double        _gfps = 0.0;
  std::int64_t  _pts  = 0;
  // Reused across pushes and pulls; owned so a run allocates none.
  AVFrame*      _in_frame  = nullptr;
  AVFrame*      _out_frame = nullptr;

  // Sideband carried onto the output beats, from the first input beat
  // of the run (streaming) or of this clip (stacked).
  FlexData      _sb_seed;
  bool          _have_seed = false;
  std::uint64_t _first_ts_us = 0;
  bool          _have_ts     = false;
  std::uint64_t _out_index   = 0;
  std::uint64_t _emitted     = 0;
  bool          _shape_warned = false;
};

}  // namespace vpipe

#endif

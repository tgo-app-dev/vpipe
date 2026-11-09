#ifndef VPIPE_STAGES_AUDIO_VIDEO_TEMPORAL_RESAMPLE_STAGE_H
#define VPIPE_STAGES_AUDIO_VIDEO_TEMPORAL_RESAMPLE_STAGE_H

#include "apple-silicon/tensor-beat.h"
#include "common/av-filter-graph.h"
#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/memory-plan.h"
#include "pipeline/resource-plan.h"
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
//   rife     RIFE (Practical-RIFE 4.22-4.26, generative-models/rife/) --
//            a learned flow network synthesises the frame AT each output
//            time from the two source frames around it, on the GPU.
//            Far better than `motion` at what `motion` is for (going UP
//            in rate; slow motion), and like it no anti-alias going
//            down: an output time between two frames gets the frame at
//            that time, not an average. Needs the model (`hf_dir`).
//            Not libavfilter: the stage times the output frames itself
//            -- output frame k at source position k * in / out, the
//            source frame itself where that is whole -- and holds the
//            last frame over the end of the clip, so a clip keeps its
//            length (N frames at r in are N / r seconds out).
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

  // Only `rife` holds a model: its checkpoint and the GPU scratch of the
  // geometry it runs (booked at 1920x1080 until the first frame says).
  std::vector<ResourceClaim> declare_resources() const override;
  StageMemory declare_memory() const override;

  const StageSpec& spec() const noexcept override;

  // See the CLOCK DOMAINS note above: the oport shares the iport's
  // group only when the whole time axis rides in one beat.
  unsigned oport_clock_group(unsigned p) const noexcept override;

  // How a rate change is filtered. Public so the chain builder has a
  // test that needs neither FFmpeg nor a runtime.
  enum class Method { kNearest, kBlend, kAverage, kMotion, kRife };

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
  // `rife`: frames synthesised (the rest were source frames, or held
  // over a scene cut), and pairs held as cuts.
  std::uint64_t interpolated() const noexcept { return _interpolated; }
  std::uint64_t cuts_held()    const noexcept { return _cuts_held; }

  // Where output frame `k` falls in the source: frame `*i0` and the
  // fraction `*t` of the way to the next, for a source at `in_fps`
  // resampled to `out_fps` -- in exact rational arithmetic, so 30000/1001
  // into 60000/1001 lands on whole frames every other output and not a
  // hair off them. False for rates that do not convert.
  static bool output_position(std::int64_t k, double in_fps, double out_fps,
                              std::int64_t* i0, double* t);
  // How many frames a clip of `n` source frames resamples to: every
  // output time before the clip's end, n / in_fps seconds.
  static std::int64_t output_count(std::int64_t n, double in_fps,
                                   double out_fps);

private:
  struct Rife;   // the `rife` method's model and frame window (.cc)

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

  // `rife`.
  std::string   _hf_dir;
  double        _rife_scale = 0.0;     // 0: the driver's rule per size
  double        _scene_threshold = 0.2;
  std::unique_ptr<Rife> _rife;
  std::uint64_t _interpolated = 0;
  std::uint64_t _cuts_held = 0;
  // The checkpoint file for `_hf_dir`; empty + *err when it names none.
  std::string rife_file_(std::string* err = nullptr) const;
  std::string rife_scratch_label_() const;
  // Load at initialize; drop at the end of the run.
  bool rife_load_();
  void rife_release_();
  // The `rife` beat paths. `out` gets each output frame's planar bytes.
  bool rife_stream_frame_(std::unique_ptr<BeatPayloadIntf> p, double in_fps,
                          std::vector<std::vector<std::uint8_t>>* out);
  void rife_stream_end_(std::vector<std::vector<std::uint8_t>>* out);
  bool rife_clip_(const TensorBeat& clip, double in_fps,
                  std::vector<std::vector<std::uint8_t>>* out);
  // Output frame at `t` of the way from source frame `i0` to the next,
  // both already in the window (t == 0: frame i0 itself).
  bool rife_pair_(std::int64_t i0, double t, std::vector<std::uint8_t>* out);
  // Book the scratch of the geometry the window now runs at.
  void rife_revise_scratch_();
};

}  // namespace vpipe

#endif

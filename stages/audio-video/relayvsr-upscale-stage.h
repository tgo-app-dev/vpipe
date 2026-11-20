#ifndef VPIPE_STAGES_AUDIO_VIDEO_RELAYVSR_UPSCALE_STAGE_H
#define VPIPE_STAGES_AUDIO_VIDEO_RELAYVSR_UPSCALE_STAGE_H

#include "common/flex-data.h"
#include "common/job.h"
#include "pipeline/memory-plan.h"
#include "pipeline/resource-plan.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {

class TensorBeatPayload;

// 4x / 2x VIDEO SUPER-RESOLUTION by RelayVSR (kopper/RelayVSR, 4x;
// lightx2v/RelayVSR_X2, 2x; ComfyUI's int8 repack of both,
// Comfy-Org/RelayVSR; arXiv 2609.37850; generative-models/relayvsr/):
// frames in, the same frames at the model's scale out, one beat for one
// beat, in order.
//
// Two models run in relay over the stream. A large one-step DiT writes a
// KEYFRAME every `window - 1` frames (and at the last), and a small decoder
// turns each frame into its 4x frame directly in RGB, conditioned on the
// two keyframes around it and the frame before -- so a frame leaves once
// the window it sits in is complete, at most `window - 1` frames after it
// arrived. A clip ending mid-window is keyframed at its last frame.
//
// Its sibling is FlashVSR+ (generate-video's `flashvsr` family, which
// denoises latents a chunk at a time and needs vae-decode after it). This
// one is a single stage: no conditioning, no VAE, no scheduler.
//
//   iport0  frames   TensorBeatPayload, planar RGB [3, H, W] per beat: u8,
//                    or f32 / f16 in [0, 1] (video-to-rgb's two). Every
//                    frame of a run one size; any size (the model pads to
//                    8 and cuts back).
//   oport0  frames   planar RGB [3, sH, sW] (s the scale): u8, or f32 in
//                    [0, 1]
//                    (`output_dtype`), the input's sideband carried over
//                    -- its rate and timestamp are the output's.
//
// Acceleration (both opt-in): `i8_gemm` / `i8_gemm_tier` run both models'
// block projections on the int8 tiers (matrix cores, M5 on), and
// `ane_ffn` splits both models' feed-forwards by rows with the Neural
// Engine (an M4's lever; a GPU that is faster alone keeps it). See
// docs/RELAYVSR.md for what each measured.
//
// Memory: the two models hold 8.4 GB of weights (5.2 GB from ComfyUI's
// int8 repack), and every one of the keyframe DiT's 30 blocks keeps the
// last four keyframes' keys and values -- 0.4 GB a block at a 960 x 544
// input at 4x (a quarter of that at 2x, whose tokens are 1/16 of the frame
// rather than 1/8). Booked at every launch at `output_width` x
// `output_height` over the scale when the graph gives them, else at a
// 3840 x 2176 output; revised to the run's frame size at its first
// frame, and withdrawn with the models at the end of the stream. Nothing
// here streams, so a run that cannot fit is REFUSED before it allocates:
// at the launch when the output size is given and the run alone is over
// the GPU's working set (the weights are not read), and at the first
// frame against what is left then. A complete window goes out one frame
// at a time (decoded, copied once into its beat, written), so the host
// holds one output frame, not a window of them.
class RelayVsrUpscaleStage final : public TypedStage<RelayVsrUpscaleStage> {
public:
  static constexpr const char* kTypeName = "relayvsr-upscale";

  RelayVsrUpscaleStage(const SessionContextIntf* session, std::string id,
                       std::vector<InEdge> iports, FlexData config);
  ~RelayVsrUpscaleStage() override;

  Job initialize(RuntimeContext& ctx) override;
  Job process(RuntimeContext& ctx) override;
  void reset_run_state() override;

  std::vector<ResourceClaim> declare_resources() const override;
  StageMemory declare_memory() const override;

  const StageSpec& spec() const noexcept override;

  // The GPU working set this Mac gives a process: Metal's recommended
  // size, scaled to the RAM the session believes (VPIPE_RAM_LIMIT_MB). A
  // property of the device and the setting, the same at every launch:
  // what the launch's refusal is judged against. 0 without a GPU.
  std::size_t gpu_working_set() const;

  // Test-only.
  std::uint64_t emitted() const noexcept { return _emitted; }
  int window() const noexcept { return _window; }
  int loads() const noexcept { return _loads; }   // the models, loaded

private:
  struct Model;

  // The two checkpoint files of `hf_dir`; false + *err when it names none.
  bool files_(std::string* wan, std::string* flash,
              std::string* err = nullptr) const;
  // The files and both models' configs, the acceleration applied as the
  // load will run it (the int8 tier this GPU takes; the ANE as asked --
  // the plan's grant is decided at load). Every booking reads these.
  struct Configs;
  bool configs_(Configs* out, std::string* err = nullptr) const;
  std::string ane_label_(const char* model) const;
  std::string scratch_label_() const;
  // What a stream of this frame size holds beside the weights.
  std::uint64_t run_bytes_(int h, int w) const;
  // The input frame the launch plans: the output size given (or 3840 x
  // 2176) over the model's scale. False, and the reason, when the output
  // size given is not a multiple of the scale (the size is still set).
  bool plan_in_(const Configs& c, int* h, int* w,
                std::string* why = nullptr) const;
  bool load_();
  // Whether `weights` (still to load) and a stream's `run` fit the box;
  // false: reported (a "memory" report) and the refusal's sentence in
  // *why, for the caller to raise -- session()->error() throws, so it
  // gives back what it holds first. kLaunch judges the run alone against
  // the whole working set, kFirstFrame against what is left of it and
  // against reclaimable RAM.
  enum class Gate { kLaunch, kFirstFrame };
  bool fits_(std::uint64_t weights, std::uint64_t run, int h, int w,
             Gate gate, std::string* why);
  // The models let go (release_), and everything this stage booked
  // withdrawn: the weights' declarations, the run's arena and the ANE
  // modules' bookings.
  void release_();
  void withdraw_(const std::string& wan, const std::string& flash);
  void revise_scratch_(int h, int w);
  // A frame's sideband for the output: the input's, as it came.
  FlexData out_sideband_(std::int64_t index);
  // An output frame's beat: its one host copy, from the stream's buffer.
  std::unique_ptr<TensorBeatPayload> beat_(std::int64_t index, int h, int w,
                                           const void* px,
                                           std::size_t bytes);

  std::string _hf_dir;
  int _window = 16;
  int _scale = 0;                // asked: 0 the model's, 2 or 4
  int _out_w = 0, _out_h = 0;    // output_width / output_height
  bool _plan_given = false;      // ...given
  std::uint64_t _seed = 42;
  std::string _out_dtype = "auto";
  int _i8_tier = -1;             // asked: -1 off, 0 accurate .. 3
  bool _ane_ffn = false;
  float _ane_rows = 0.0f;
  bool _sage_attn = false;       // asked; the configs take it on matrix
  bool _sage_pv8 = false;        // cores alone
  bool _autotune = true;

  std::unique_ptr<Model> _m;
  int _loads = 0;
  // The sideband of every frame still in the window, by index.
  std::vector<std::pair<std::int64_t, FlexData>> _sb;
  std::uint64_t _emitted = 0;
  std::uint64_t _scratch_stated = 0;
  bool _f32_out = false;
  bool _warned_shape = false;
  // The ANE modules' bookings set to what each model holds (its module's
  // bytes when it armed, 0 when it declined) -- once, after the first
  // window.
  bool _ane_booked = false;
  void book_ane_();
};

}  // namespace vpipe

#endif

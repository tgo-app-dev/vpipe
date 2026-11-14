#ifndef VPIPE_GENERATIVE_MODELS_RELAYVSR_RELAYVSR_STREAM_H
#define VPIPE_GENERATIVE_MODELS_RELAYVSR_RELAYVSR_STREAM_H

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/relayvsr/metal-relayvsr-decoder.h"
#include "generative-models/relayvsr/metal-relayvsr-wan.h"

#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

namespace relayvsr {
class Ops;
}

// RelayVSR (Wang et al., "Large-Small Model Collaboration for Efficient
// Real-World Video Super-Resolution", arXiv 2609.37850; kopperx/RelayVSR,
// weights kopper/RelayVSR, Apache-2.0): streaming 4x video super-
// resolution by two models in relay. A large one-step DiT
// (MetalRelayVsrWan) writes a KEYFRAME latent every `window - 1` frames,
// and a small decoder (MetalRelayVsrDecoder) turns every frame into its 4x
// frame between the two keyframes around it.
//
// WHERE ITS FILES ARE. The release is two safetensors side by side,
// `wan.safetensors` (format refvsr-wan-v1) and `flash.safetensors`
// (refvsr-flash-v1), each carrying its architecture in its metadata.
// `path` is the directory holding both or either file; a file named
// otherwise is told by its format.
struct RelayVsrFiles {
  std::string wan;
  std::string flash;
};
bool resolve_relayvsr_files(const std::string& path, RelayVsrFiles* out,
                            std::string* err = nullptr);

// THE STREAM, as the release's CollaborativeFrameStream runs it, frame in,
// frames out:
//
//   keyframes at 0, W-1, 2(W-1), ... and at the LAST frame;
//   each window [start .. end] decoded against its two keyframes with a
//   fresh video history, frame by frame;
//   a window's first frame -- the previous window's last, already out --
//   decoded again (it is the next frame's history) but not emitted.
//
// So a frame comes out once the window it is in is complete: at most W - 1
// frames after it went in. A complete window is decoded ONE FRAME A CALL
// (next()), each frame into the stream's own buffer: a caller that hands
// each on before asking for the next holds one frame, not a window of
// them (16 frames of 4K f32 are 1.6 GB of host memory).
//
// The keyframe noise is torch's CPU normal draw (TorchCpuRandn) from one
// generator seeded per stream, in keyframe order -- what the reference
// draws, on the CPU rather than CUDA's generator.
//
// Frames are planar [3, h, w]: u8, or f32 / f16 in [0, 1]. Out comes
// planar [3, 4h, 4w]: u8 (the release's truncation) or f32 in [0, 1].
// Every frame of a stream must be the same size.
class RelayVsrStream {
 public:
  struct Options {
    int window = 16;             // --window-size: keyframes every W - 1
    std::uint64_t seed = 42;
    bool f32_out = false;
  };
  enum class In : int { kU8 = 0, kF32 = 1, kF16 = 2 };
  // A frame out: planar [3, h, w], u8 or f32, `bytes` long at `px` -- the
  // stream's own buffer, valid until the next call on the stream.
  struct Frame {
    std::int64_t index = 0;      // the frame's index in the stream
    int h = 0, w = 0;            // 4x the input
    const void* px = nullptr;
    std::size_t bytes = 0;
  };

  RelayVsrStream(MetalRelayVsrWan* wan, MetalRelayVsrDecoder* dec,
                 metal_compute::MetalCompute* mc, const Options& opt);
  ~RelayVsrStream();

  // The next frame in. When it completes a window (its end keyframed),
  // ready() turns true; drain it with next() before the next push, which
  // is refused until then. `stop`, when set, is asked between blocks and
  // frames; a stop returns false with no error and leaves the stream to
  // reset().
  bool push(const void* px, In kind, int h, int w, std::string* err,
            const std::function<bool()>& stop = {});
  // The end of the source: the last window, keyframed at its last frame,
  // made ready (when it has a frame not yet out). Once drained, the stream
  // is reset.
  bool finish(std::string* err, const std::function<bool()>& stop = {});
  // Whether a complete window has frames still to come out.
  bool ready() const noexcept { return _ready; }
  // The next frame of the ready window, decoded now. False with no error
  // when none is ready (or on a stop); false with *err on a failure.
  bool next(Frame* f, std::string* err,
            const std::function<bool()>& stop = {});
  // Forget everything; the next push starts a stream.
  void reset();

  std::int64_t frames_in() const noexcept { return _n_in; }
  std::int64_t frames_out() const noexcept { return _n_out; }
  int keyframes() const noexcept { return _n_key; }

  // TEST HOOK: each keyframe's latent as it is made (f32 [48, ph/4,
  // pw/4]) and the noise it was made from.
  std::function<void(std::int64_t frame, const std::vector<float>& noise,
                     const metal_compute::SharedBuffer& latent)>
      on_keyframe;

  // What a stream of this frame size holds beside the two models' weights:
  // the keyframe ring and working set, three primed keyframes, the
  // decoder's working set and history, the window's frames.
  static std::uint64_t run_bytes(const MetalRelayVsrWan::Config& wan,
                                 const MetalRelayVsrDecoder::Config& dec,
                                 int h, int w, int window);

 private:
  struct Held {
    std::int64_t index = 0;
    metal_compute::SharedBuffer px;   // the planar source, as it came
  };
  bool keyframe_(const Held& f, MetalRelayVsrDecoder::Condition* cond,
                 std::string* err, const std::function<bool()>& stop);
  // After a window's last frame: its frames but the last (the next
  // window's first) recycled, its end keyframe the next one's start --
  // or, for the last window, reset().
  void close_window_();
  bool upload_(const Held& f, bool bf16, metal_compute::SharedBuffer* dst,
               std::string* err);

  MetalRelayVsrWan* _wan;
  MetalRelayVsrDecoder* _dec;
  metal_compute::MetalCompute* _mc;
  Options _opt;
  std::unique_ptr<relayvsr::Ops> _op16, _op_bf16;
  struct Rng;
  std::unique_ptr<Rng> _rng;

  int _h = 0, _w = 0, _ph = 0, _pw = 0;
  In _fmt = In::kU8;
  std::int64_t _n_in = 0, _n_out = 0;
  int _n_key = 0;
  bool _first_window = true;
  bool _ready = false;                // a complete window to decode
  bool _final = false;                // ... and it is the last
  std::size_t _pos = 0;               // the next frame of it to decode
  std::deque<Held> _window;           // [start .. ]
  std::vector<metal_compute::SharedBuffer> _spare;   // recycled frames
  MetalRelayVsrDecoder::Condition _start, _end;
  metal_compute::SharedBuffer _lr16, _lr_bf16, _latent, _frame;
  metal_compute::CommandStream _cs;
  bool _have_cs = false;
};

}  // namespace genai
}  // namespace vpipe

#endif

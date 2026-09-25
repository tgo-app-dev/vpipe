#ifndef VPIPE_STAGES_LATENT_PREVIEW_H
#define VPIPE_STAGES_LATENT_PREVIEW_H

#include "common/flex-data.h"
#include "pipeline/resource-plan.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "generative-models/shared/metal-tae-decoder.h"
#endif

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace vpipe {

class RuntimeContext;
class SessionContextIntf;

// A LATENT PREVIEW: the denoiser's clean estimate at some step, decoded by
// a tiny autoencoder (TAE) and pushed out as pixels while the generation
// runs.
//
// THE SPLIT. What a preview costs and how often it is worth paying for
// is a property of the MODEL -- a TAE is trained for one latent space,
// and a 33B video step leaves room for a preview every step where a
// 1-second image step does not -- so the knobs ride on the family's own
// `<family>-model-config` source, next to its sigma shifts. The
// denoising stage reads them off the model_config beat and owns the
// port; this file is what the two share, so a family adds four keys and
// a stage adds one port, and neither re-derives the rest.
//
// THE PORT is push-style, like rtsp-capture's: DropOldest at a small
// depth, written from a thread the stage owns. A preview nobody is
// watching is dropped rather than holding the generation up, and a slow
// viewer sees the newest one. Because DropOldest takes exactly one
// consumer, wire it to one stage.

// The model_config keys, shared by every family's config source. The
// names are the contract; a family's config stage declares them with
// these docs so every family reads the same.
namespace latent_preview {

inline constexpr std::string_view kVaeKey      = "preview_vae";
inline constexpr std::string_view kEveryKey    = "preview_every";
inline constexpr std::string_view kMaxEdgeKey  = "preview_max_edge";
inline constexpr std::string_view kFramesKey   = "preview_frames";

inline constexpr std::string_view kVaeDoc =
    "OPTIONAL tiny autoencoder for LIVE PREVIEWS: after a denoising step, "
    "the model's clean estimate is decoded by it and written to the "
    "denoiser's `preview` oport, which a `preview` stage plays directly. "
    "A registered model, a directory holding one .safetensors, or a path "
    "to one. It must be trained for THIS model's latent space. Empty (the "
    "default) turns previews off, and so does leaving the port unwired -- "
    "then nothing is loaded and nothing is decoded";
inline constexpr std::string_view kEveryDoc =
    "render a preview after every Nth denoising step; the last step always "
    "renders. The decode runs on a thread of its own and never makes the "
    "generation wait: a preview still queued when the next is due is "
    "REPLACED rather than kept. It does share the GPU with the denoiser, "
    "so a larger N buys back that GPU time. 0 turns previews off";
inline constexpr std::string_view kMaxEdgeDoc =
    "longest edge of the preview picture, in pixels; the decoded picture "
    "is box-filtered down by a whole factor until it fits (so 512 turns "
    "960x576 into 480x288). The decode itself always runs at the TAE's "
    "own size: shrinking the latent instead blurs it and shifts colour. "
    "0 sends the picture as decoded";
inline constexpr std::string_view kFramesDoc =
    "OPTIONAL cap on a video preview's length, in frames: 0 (the default) "
    "previews the whole clip, N decodes only its first N frames. Exact, "
    "not an approximation -- a video TAE's memory runs forward only -- and "
    "the one knob that makes a preview CHEAPER";

}  // namespace latent_preview

// The keys as one value. `vae` empty means no preview was asked for.
//
// HOST-SIDE ONLY. A plugin's config source emits the keys by the names
// above; it does not hold this struct, whose layout is not part of the
// plugin contract and may grow.
struct LatentPreviewSpec {
  std::string vae;
  int every      = 1;
  int max_edge   = 512;
  int max_frames = 0;

  bool enabled() const noexcept { return !vae.empty() && every > 0; }
  // Whether step `step` (1-based, of `total`) renders.
  bool due(int step, int total) const noexcept;

  // Read the keys off a model_config beat. Absent keys keep their
  // defaults; no `preview_vae` is a disabled spec.
  static LatentPreviewSpec from_model_config(const FlexData& cfg);
  // Write them into a config source's beat -- only when `vae` is set, so
  // a graph that asked for no preview says nothing at all.
  void emit(FlexData& cfg) const;

  bool
  operator==(const LatentPreviewSpec& o) const noexcept
  {
    return vae == o.vae && every == o.every && max_edge == o.max_edge &&
           max_frames == o.max_frames;
  }
};

// The integer box-filter factor that brings a `w` x `h` picture's longest
// edge within `max_edge` (1 when it already fits, or max_edge is 0).
int latent_preview_shrink(int w, int h, int max_edge);

#ifdef VPIPE_BUILD_APPLE_SILICON

// What previewing a `w` x `h` output (`frames` of it, 1 for an image)
// books in the plan, for a denoiser's declare_resources(): the TAE's
// weights, and its decode scratch plus the host's frame buffers for the
// denoise phase. The scratch is EXACT -- MetalTaeDecoder::plan() reads it
// off the TAE's headers, the sizing the decoder allocates from -- and the
// latent geometry is the TAE's own (the smallest latent clip its `trim`
// turns into `frames`, at its spatial factor). Empty when `spec` names no
// TAE or the TAE does not resolve: nothing will load then either.
std::vector<ResourceClaim>
latent_preview_claims(const SessionContextIntf* session,
                      std::string_view owner, const LatentPreviewSpec& spec,
                      genai::MetalTaeDecoder::Trim trim, int w, int h,
                      int frames);

// Renders previews for one denoising stage: holds the TAE, a worker
// thread, and a one-slot mailbox. Created once per stage and reused
// across generations; begin()/end() bracket one generation.
//
//   configure(spec, trim)      when the model_config beat or the family
//                              changes (a different vae drops the TAE)
//   begin(ctx, port, ...)      before the denoise: loads the TAE on first
//                              use and starts the worker. False when the
//                              port has no consumer or the TAE will not
//                              load -- the generation runs either way
//   due(step, total)           the loop asks before building a latent
//   submit(...)                hands over [C, T, h, w]; never waits on
//                              the GPU
//   end(ctx)                   after the denoise: finishes what is queued
//                              (or abandons it on a stop), then joins
//
// Every render is one beat: planar U8 [F, 3, H, W] for a video (the
// `rgb-clip` shape a preview stage loops) and [3, H, W] for an image,
// with {frames, fps, fps_num, fps_den, step, steps} on the sideband.
class LatentPreviewer {
public:
  LatentPreviewer(const SessionContextIntf* session, std::string owner);
  ~LatentPreviewer();

  LatentPreviewer(const LatentPreviewer&)            = delete;
  LatentPreviewer& operator=(const LatentPreviewer&) = delete;

  void configure(const LatentPreviewSpec& spec,
                 genai::MetalTaeDecoder::Trim trim);
  const LatentPreviewSpec& spec() const noexcept { return _spec; }

  // `fps` / `video_frames` are the clip the REAL decoder will make, so a
  // preview with fewer frames (a 2D TAE makes one per latent frame) is
  // stamped with the rate that plays it over the same duration. An image
  // passes 0 / 0 and gets a [3, H, W] beat.
  bool begin(RuntimeContext& ctx, unsigned port, double fps,
             int video_frames);

  // One generation's previews as a scope: begin() on construction, end()
  // when it is left -- after the outputs are written on the ordinary path,
  // so the last step's preview renders while the real decode is already
  // working -- and on every early return too. `release` also drops the
  // TAE afterwards (a stage unloading when idle sets it).
  class Scope {
  public:
    Scope(LatentPreviewer* p, RuntimeContext& ctx, unsigned port,
          double fps, int video_frames)
    {
      if (p != nullptr && p->begin(ctx, port, fps, video_frames)) {
        _p = p;
        _ctx = &ctx;
      }
    }
    ~Scope()
    {
      if (_p == nullptr) { return; }
      _p->end(*_ctx);
      if (release || _p->releases_each_generation()) { _p->release(); }
    }
    Scope(const Scope&)            = delete;
    Scope& operator=(const Scope&) = delete;
    bool active() const noexcept { return _p != nullptr; }
    bool release = false;

  private:
    LatentPreviewer*      _p   = nullptr;
    const RuntimeContext* _ctx = nullptr;
  };
  bool active() const noexcept { return _worker.joinable(); }
  bool due(int step, int total) const noexcept;
  void submit(std::vector<float> latent, int C, int T, int h, int w,
              int step, int total);
  void end(const RuntimeContext& ctx);

  // Drop the decoder and its weights (the stage's idle unload).
  void release();

  // DROP IT AFTER EVERY GENERATION, rather than keeping it for the next
  // one. A stage-wide policy, set once where the previewer is built,
  // because it is a property of how that stage runs and not of any one
  // generation -- and because it is what makes the DENOISE-PHASE claim
  // latent_preview_claims files true. Held across generations, the TAE
  // and its scratch (726 MB for a 1024^2 taef1) are still resident
  // through the decode phase, which the claim says they are not, and
  // the decode is where a tight box is tightest. The cost is a reload
  // per generation: ~10-23 MB read and narrowed to f16, against a
  // generation measured in seconds.
  void release_each_generation(bool on) noexcept { _release_each = on; }
  bool releases_each_generation() const noexcept { return _release_each; }

  // What the TAE holds, 0 before the first load.
  std::size_t resident_bytes() const noexcept;

  // Counters for the current (or last) generation, for the log and tests.
  std::uint64_t rendered() const noexcept { return _rendered.load(); }
  std::uint64_t replaced() const noexcept { return _replaced.load(); }

private:
  struct Job {
    std::vector<float> latent;
    int C = 0, T = 0, h = 0, w = 0;
    int step = 0, total = 0;
  };

  bool load_(std::string* err);
  void worker_();
  void render_(Job& job);

  const SessionContextIntf* _session = nullptr;
  std::string               _owner;
  LatentPreviewSpec         _spec;
  genai::MetalTaeDecoder::Trim _trim = genai::MetalTaeDecoder::Trim::kLeading;

  std::string                             _loaded_vae;
  std::unique_ptr<genai::MetalTaeDecoder> _dec;
  // A load that failed is not retried every generation: the same file
  // would fail the same way, and the warning would repeat per clip.
  std::string _failed_vae;
  // See release_each_generation(). Off by default: keeping the decoder
  // is the cheaper answer wherever the claim already says so.
  bool _release_each = false;

  // Set by begin(), read by the worker.
  RuntimeContext* _ctx = nullptr;
  unsigned        _port = 0;
  double          _fps = 0.0;
  int             _video_frames = 0;

  std::thread             _worker;
  std::mutex              _mu;
  std::condition_variable _cv;
  bool                    _have_job = false;
  Job                     _job;
  bool                    _finishing = false;   // drain, then exit
  std::atomic<bool>       _abandon{false};      // exit now
  // Set by the worker when the port closes under it; read by due().
  std::atomic<bool>       _port_closed{false};

  std::atomic<std::uint64_t> _rendered{0};
  std::atomic<std::uint64_t> _replaced{0};
};

#endif  // VPIPE_BUILD_APPLE_SILICON

}  // namespace vpipe

#endif

#ifndef VPIPE_STAGES_AUDIO_VAE_ENCODE_STAGE_H
#define VPIPE_STAGES_AUDIO_VAE_ENCODE_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "stages/model-memory.h"
#include "generative-models/vae-model-registry.h"
#endif

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {

// Audio VAE encode: PCM in, a latent SOUNDTRACK out — the reference a
// generator conditions on when you want it to keep an existing track
// rather than invent one. The mirror of `audio-vae-decode`.
//
//   iport0  TensorBeatPayload f32 PCM, rank-1 [N] mono or rank-2
//           [channels, N] PLANAR, in [-1, 1] — what `audio-to-pcm` and
//           `audio-vae-decode` emit. `sample_rate` is read from the
//           sideband; the `sample_rate` config is the fallback for a
//           producer that stamps none.
//
//   iport1  OPTIONAL FlexDataPayload model reference (a `model-select`
//           source), latched once; OVERRIDES hf_dir.
//
//   oport0  TensorBeatPayload f32, the ENCODER's own latent shape —
//           for a family targeting `generate-video`'s `ref_audio_rows`
//           that is [rows, dim], sideband {sample_rate, seconds}.
//
// ---- IT ACCUMULATES, AND WHY THERE IS NO SEPARATE AGGREGATOR ----
//
// This stage buffers every PCM beat and encodes ONCE, at drain (iport
// EOS). It is not a per-beat transform, and a graph cannot get the same
// answer by encoding chunks and concatenating latents: an audio VAE is
// causal and compresses time (4x for LTX-2.5), so a chunk boundary is a
// place where the convolutions never saw the neighbouring samples and
// where the latent's own time base restarts. Concatenated chunk latents
// are the wrong length AND wrong at every seam.
//
// The same argument is why `vae-encode` takes a clip rather than frames,
// and why neither of them is fed by a latent-combining stage. Aggregate
// BEFORE the encoder or not at all.
//
// `max_seconds` bounds the buffer: a reference soundtrack is a clip, not
// a live feed, and a stage that grew without limit on an open microphone
// would fail as an OOM rather than as a message.
//
// ONE BEAT PER RUN, and the oport closes behind it. A `generate-video`
// asked for a SECOND request then reads a closed port, gets nothing, and
// generates unreferenced — the same shape as `vae-encode` feeding
// `ref_latent0` from one image. A graph that wants a reference on every
// request needs a source that emits on every request.
//
// Config (FlexData object):
//   hf_dir       (string, OPTIONAL) — the model root
//   sample_rate  (int, default 0)   — fallback when a beat carries none
//   max_seconds  (real, default 30) — refuse to buffer more than this
//   unload_when_idle (string)       — auto|always|never
//
// THERE IS NO BUILT-IN FAMILY. Every audio encoder reachable from here
// comes from `VaeModelFamily::load_audio_encoder`, i.e. from a plugin.
// The stage says so plainly when nothing claims the checkpoint rather
// than falling back to something that would encode at the wrong latent
// geometry.
class AudioVaeEncodeStage final : public TypedStage<AudioVaeEncodeStage> {
public:
  static constexpr const char* kTypeName = "audio-vae-encode";

  AudioVaeEncodeStage(const SessionContextIntf* session,
                      std::string               id,
                      std::vector<InEdge>       iports,
                      FlexData                  config);
  ~AudioVaeEncodeStage() override;

  Job initialize(RuntimeContext& ctx) override;

  std::vector<ResourceClaim> declare_resources() const override;

  // Latch a `model-select` constant before the planning phase, so
  // the claim above is made against the model this graph will
  // actually run rather than against an empty hf_dir.
  void apply_constant(unsigned iport, const FlexData& beat) override;
  void reset_run_state() override;
  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only accessors.
  const std::string& hf_dir() const noexcept { return _hf_dir; }
  const std::string& family() const noexcept { return _family; }
  std::uint64_t latents_emitted() const noexcept { return _latents; }

private:
  std::string _hf_dir;
  std::string _family;
  std::uint64_t _latents = 0;
  int    _cfg_rate    = 0;
  double _max_seconds = 30.0;

  bool _model_latched  = false;
  bool _load_attempted = false;

  // The accumulated PCM, PLANAR [channels][n]. `_rate` is what the first
  // beat said; a later beat that disagrees is refused rather than
  // resampled, because splicing two rates is a reference that is the
  // right length and the wrong pitch in the middle.
  std::vector<std::vector<float>> _pcm;
  int  _rate = 0;
  bool _overflowed = false;

#ifdef VPIPE_BUILD_APPLE_SILICON
  genai::VaeModelFamily*                  _vae_family = nullptr;
  std::unique_ptr<genai::AudioVaeEncoder> _plugin_enc;
  bool _family_probed = false;

  void ensure_loaded_();

  model_memory::UnloadPolicy _unload_cfg = model_memory::UnloadPolicy::kAuto;
  bool _unload_idle = false;
#endif

  // Append one beat's samples. False (with a message already logged)
  // when the beat does not fit the stream being built.
  bool accumulate_(const class TensorBeatPayload& tb);
};

}  // namespace vpipe

#endif

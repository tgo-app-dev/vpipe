#ifndef VPIPE_STAGES_AUDIO_VAE_DECODE_STAGE_H
#define VPIPE_STAGES_AUDIO_VAE_DECODE_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "stages/model-memory.h"
#include "generative-models/vae-model-registry.h"
#include "generative-models/minimax-h3/metal-minimax-h3-audio-vae.h"
#endif

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {

// Audio VAE decode: turns the latent SOUNDTRACK an omni generator emits
// into PCM. The counterpart of `vae-decode`, and a separate stage rather
// than a branch of it because the two produce different things -- that
// one emits RGB frames, this one emits samples, and a stage's output
// type is what a graph wires against.
//
//   iport0  TensorBeatPayload f32 [stereo, latent_channels, frames]
//           latent audio, per-channel WHITENED (the boundary
//           `generate-video`'s oport1 emits). The stage un-whitens
//           (z * std + mean) and decodes.
//
//   iport1  OPTIONAL FlexDataPayload model reference (a `model-select`
//           source). Latched on the first beat; OVERRIDES hf_dir so the
//           conditioner / DiT / decoders of one graph share a model
//           choice. When wired, the load is DEFERRED to the first
//           process(), since the beat only arrives after the init
//           barrier.
//
//   oport0  TensorBeatPayload f32 PCM [channels, n_samples], PLANAR and
//           clamped to [-1, 1], sideband {sample_rate} -- what
//           `save-audio` reads.
//
// Config (FlexData object):
//   hf_dir            (string, OPTIONAL) -- the model root; the audio VAE
//                                           is read from
//                                           <hf_dir>/audio_vae (or
//                                           hf_dir itself if it already
//                                           is one)
//   unload_when_idle  (string) -- auto|always|never
class AudioVaeDecodeStage final : public TypedStage<AudioVaeDecodeStage> {
public:
  static constexpr const char* kTypeName = "audio-vae-decode";

  AudioVaeDecodeStage(const SessionContextIntf* session,
                      std::string               id,
                      std::vector<InEdge>       iports,
                      FlexData                  config);
  ~AudioVaeDecodeStage() override;

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
  const std::string& hf_dir()  const noexcept { return _hf_dir; }
  const std::string& family()  const noexcept { return _family; }
  std::uint64_t      clips_emitted() const noexcept { return _clips; }

private:
  // See VaeDecodeStage::vae_dir_for_release_.
  std::string vae_dir_for_release_() const;

  // See Stage::declare_memory.
  StageMemory declare_memory() const override;

  // Pass TWO. The phase belongs HERE, not in declare_resources: a
  // phased claim in pass one is ignored (and warned about), because a
  // stage deciding a phase must be able to see every peer's
  // declaration, and in pass one only some have arrived.
  std::vector<ResourceClaim> decide_resources() const override;

  std::string _hf_dir;
  std::string _family;              // "minimax-h3"
  std::uint64_t _clips = 0;

  bool _model_latched  = false;
  bool _load_attempted = false;

#ifdef VPIPE_BUILD_APPLE_SILICON
  std::unique_ptr<genai::MetalMiniMaxH3AudioVae> _h3_vae;

  // ---- an out-of-tree family (VaeModelRegistry) -------------------------
  // The SAME registry `vae-decode` consults -- a family that generates
  // both modalities answers for both, so this is load_audio_decoder() on
  // VaeModelFamily rather than a second registry. BORROWED; the registry
  // outlives every stage. Dispatch below is guarded on the POINTER, not
  // on the `_family` string.
  genai::VaeModelFamily*                  _vae_family = nullptr;
  std::unique_ptr<genai::AudioVaeDecoder> _plugin_dec;
  bool _family_probed = false;

  void ensure_loaded_();

  model_memory::UnloadPolicy _unload_cfg = model_memory::UnloadPolicy::kAuto;
  bool _unload_idle     = false;
  bool _unload_resolved = false;
  bool _unloaded        = false;
  void unload_vae_();
  void reload_vae_();
#endif
};

}  // namespace vpipe

#endif

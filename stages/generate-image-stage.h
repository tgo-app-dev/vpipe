#ifndef VPIPE_STAGES_GENERATE_IMAGE_STAGE_H
#define VPIPE_STAGES_GENERATE_IMAGE_STAGE_H

#include "apple-silicon/tensor-beat.h"
#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

// The Krea-2-Turbo text-to-image stack (Qwen3-VL text encoder + Krea2
// MMDiT + FlowMatchEuler sampler) is a from-scratch, MLX-free metal-compute
// path on the VPIPE_BUILD_APPLE_SILICON axis. On non-Apple builds this stage
// is an inert stub.
#ifdef VPIPE_BUILD_APPLE_SILICON
#include "generative-models/krea2/flow-sampler.h"
#include "generative-models/krea2/metal-krea2-transformer.h"
#include "generative-models/flux2/metal-flux2-transformer.h"
#include "generative-models/qwen-image/metal-qwen-image-transformer.h"
#include "generative-models/boogu/metal-boogu-transformer.h"
#include "generative-models/mage/metal-mage-flow-transformer.h"
#include "generative-models/mage/mage-watermark.h"
#endif

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {

namespace genai { class WeightSet; }   // generative-models/weight-set.h

// Text-to-image (DiT) stage: the denoiser half of the diffusion split -- it
// consumes ready-made conditioning from a `diffusion-conditioner` stage (which
// owns the tokenizer + text encoder + vision tower) and runs the family DiT
// (Krea2 12B MMDiT / FLUX.2 / Qwen-Image-Edit dual-stream) + FlowMatchEuler
// sampler on the metal-compute backend, then emits the unpacked latent for the
// vae-decode stage. Pair it with a diffusion-conditioner on the SAME hf_dir.
//
//   iport0  TensorBeatPayload conditioning from the diffusion-conditioner stage
//           (family-shaped + typed: krea2 f16 [n,12,2560]; flux2 f16
//           [n,3*enc_hidden]; qwen-image-edit bf16 [n_real,3584];
//           boogu-image bf16 [n,4096]).
//
//   iport1  OPTIONAL TensorBeatPayload neg_conditioning (same shape/type, the
//           conditioner's oport1). Drives classifier-free guidance: when it is
//           present AND guidance_scale != 1, each denoise step runs the DiT a
//           second time on the negative conditioning and combines
//           v = v_neg + scale*(v_pos - v_neg). Read per generation when a beat is
//           available. Unwired (or scale 1) => no CFG (single DiT pass, the
//           token-exact turbo default).
//
//   iport2  OPTIONAL FlexDataPayload model reference (from a `model-select`
//           source). Latched once; OVERRIDES the hf_dir config so the
//           conditioner / DiT / vae stages of a graph share one model. When
//           wired, the (heavy) DiT load is DEFERRED to the first process() (the
//           beat only arrives after the init barrier).
//
//   iport3  OPTIONAL FlexDataPayload sampler spec (from a
//           `diffusion-sampler-select` stage -- NOT the LLM `sampler-select`).
//           Latched on the first beat and reused for every prompt;
//           selects the integrator (euler / heun / dpmpp_2m / dpmpp_sde) + its
//           knobs. Unwired => the config default (euler, the token-exact turbo).
//
//   iport4  OPTIONAL FlexDataPayload scheduler spec (from a `scheduler-select`
//           stage). Latched likewise; selects the sigma schedule (simple /
//           karras / exponential), steps and shift. Unwired => config default
//           (simple / `steps` / shift 1.15).
//
//   iport5  OPTIONAL TensorBeatPayload reference latent 0 -- an f32 latent
//   iport6  OPTIONAL TensorBeatPayload reference latent 1 (channel-first,
//           the format the vae-encode stage emits: FLUX.2 [dit_channels, H/16,
//           W/16], Krea-2 [z_dim, H/8, W/8]). Read once when a beat is available
//           and cached. FLUX.2 uses them as multi-reference conditioning: each
//           reference is embedded, appended to the image-token stream and given
//           its own RoPE T offset so the two references occupy distinct position
//           bands. Krea-2 has no multi-ref path, so ref latent 0 doubles as the
//           img2img init (mixed into the noise at the strength-selected sigma)
//           and ref latent 1 is ignored. Unwired => text-to-image from noise.
//
//   oport0  TensorBeatPayload, an f32 latent [z_dim, H/8, W/8] (channel-first,
//           unpacked, still whitened -- the vae-decode stage un-whitens).
//
// Config (FlexData object):
//   hf_dir     (string, OPTIONAL) -- the model dir; the transformer/ subfolder's
//                                    _class_name selects the DiT family. (The
//                                    text_encoder/ + tokenizer/ live here too but
//                                    are the diffusion-conditioner's concern.) A
//                                    model-select source on iport2 overrides it;
//                                    required only when iport2 is unwired.
//   dit_dir    (string, optional) -- override for the DiT (transformer) dir,
//                                    e.g. a model-quantize'd 4/8-bit DiT; else
//                                    <hf_dir>/transformer.
//   height     (int, default 256) -- output height  (must be a multiple of 16).
//   width      (int, default 256) -- output width   (must be a multiple of 16).
//   steps      (int, default 8)   -- turbo sampler steps.
//   seed       (int, default 0)   -- initial-noise RNG seed.
//   guidance_scale (real, default 1) -- classifier-free guidance scale. 1
//                                    disables CFG (single DiT pass). >1 with a
//                                    negative prompt on iport1 runs a second
//                                    DiT pass per step and pushes the velocity
//                                    away from the negative (v_neg + scale*
//                                    (v_pos - v_neg)); ~2x DiT cost per step.
//   init_latents (string, optional) -- debug/repro: a raw f32 file of packed
//                                    initial latents [img_seq, z_dim*4] to use
//                                    instead of RNG noise (for golden anchoring).
//
// OUTPUT SIZE. An explicit width/height always wins. When NEITHER is set and a
// reference latent is wired on iport5, the size is INFERRED from it:
// ref_latent0 is vae-encode's output for the source image, so the output is
// its H/W times the family's VAE factor (8 for Krea-2 / Qwen-Image-Edit, 16
// for FLUX.2 / Mage-Flow -- see latent_scale_). An edit therefore comes out at
// the source resolution with no size config at all. Falls back to 256x256 when
// there is no reference either.
class GenerateImageStage final : public TypedStage<GenerateImageStage> {
public:
  static constexpr const char* kTypeName = "generate-image";

  GenerateImageStage(const SessionContextIntf* session,
                   std::string               id,
                   std::vector<InEdge>       iports,
                   FlexData                  config);
  ~GenerateImageStage() override;

  Job initialize(RuntimeContext& ctx) override;

  // The DiT this stage loads plus the text encoder it must coexist with
  // -- the same two directories plan_streaming() sizes against.
  std::vector<ResourceClaim> declare_resources() const override;
  // See Stage::declare_memory.
  StageMemory declare_memory() const override;

  // Latch a `model-select` constant before the planning phase, so
  // the claim above is made against the model this graph will
  // actually run rather than against an empty hf_dir.
  void apply_constant(unsigned iport, const FlexData& beat) override;

private:
  // Tell the manager what a STREAMING DiT actually keeps resident, which
  // is its pinned prefix rather than the whole checkpoint its files
  // suggest. Without this every peer sizes the box against ~20 GB that
  // was never there. No-op when the DiT is not streaming.
  void revise_dit_declaration_(const std::string& dit_dir) const;
public:
  void reset_run_state() override;
  Job process   (RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only accessors.
  const std::string& hf_dir()        const noexcept { return _hf_dir; }
  std::uint64_t      latents_emitted() const noexcept { return _latents_emitted; }
  // The adapter this stage would load with, after its own config and
  // any model_config beat. Exposed because the interesting failure is
  // that it is EMPTY when the graph thought it had set one, and every
  // other way of seeing that needs a checkpoint.
  const std::string& lora_ref(int slot = 0) const noexcept
  {
    return _lora[(std::size_t)(slot > 0 ? 1 : 0)].path;
  }
  double lora_scale(int slot = 0) const noexcept
  {
    return _lora[(std::size_t)(slot > 0 ? 1 : 0)].scale;
  }

private:
  // Stamp the generating model onto a latent beat's sideband. The chain
  // generate-image -> vae-decode -> save-image carries it through so a saved
  // file can record WHAT produced it (save-image writes it into EXIF
  // Software). Merges into any sideband already present.
  void tag_model_(TensorBeat& tb) const;

  std::string _hf_dir;
  std::string _dit_dir;
  std::string _init_latents;
  int         _height{};
  int         _width{};
  int         _steps{};
  double      _strength{};      // img2img: 0 => text-to-image (pure noise)
  double      _guidance_scale{};     // CFG scale; 1 => disabled (single pass)
  bool        _infer_size{};    // no width/height configured: size from iport5
  bool        _i8_gemm{};            // LOSSY dynamic-int8 DiT GEMMs (opt-in)
  std::uint64_t _seed{};
  std::uint64_t _latents_emitted = 0;

  // "krea2" | "flux2" | "qwen-image-edit" | "mage-flow" | "boogu-image" (from
  // the transformer _class_name).
  std::string _family = "krea2";

  // Model iport bookkeeping (used only when a model-select source is wired):
  // latch the model beat once, and load the DiT at most once (lazily, since the
  // beat only arrives after the init barrier).
  bool _model_latched  = false;
  bool _load_attempted = false;
  bool _cfg_latched    = false;

  // The last model-config beat, held UNPARSED. Which family's parser
  // reads it is not known until the checkpoint resolves, and the two
  // beats arrive on different ports in either order.
  FlexData _model_cfg;

#ifdef VPIPE_BUILD_APPLE_SILICON
  // The resident family's own parameters, as ITS model layer defines
  // them. Both held rather than one variant: the families share no keys,
  // and a struct wide enough for both would describe neither.
  genai::MetalFlux2Transformer::GenerationParams _flux2_params;
  genai::mage_wm::Params _wm_params;
  // Re-parse `_model_cfg` for the family that is now resident. Called
  // when either input changes -- a new config beat, or a model that just
  // resolved. Returns false when the beat is for another family, which
  // is reported and then ignored.
  bool apply_model_config_();
  // Resolve _hf_dir + load the family DiT (idempotent: the _load_attempted
  // guard runs the body at most once). Called from initialize() when the model
  // comes from config, or from the first process() when it arrives on the model
  // iport. No-op when _hf_dir is still empty.
  void ensure_loaded_();

  // Loaded lazily by ensure_loaded_ (one DiT per the detected family); left
  // null on failure (stage stays inert). The text encoder + tokenizer + vision
  // tower live in the paired diffusion-conditioner stage, not here.
  std::unique_ptr<genai::MetalKrea2Transformer>   _dit;
  std::unique_ptr<genai::MetalFlux2Transformer>   _flux2_dit;
  std::unique_ptr<genai::MetalQwenImageTransformer> _qie_dit;
  // Mage-Flow's NR-MMDiT IS MetalQwenImageTransformer (a different Config);
  // kept in its own handle so the two families never share load state.
  std::unique_ptr<genai::MetalMageFlowTransformer> _mage_dit;
  // Boogu-Image's 10B NextDiT (its own class: five block kinds and three
  // refiner stacks share nothing with the MMDiTs above).
  std::unique_ptr<genai::MetalBooguTransformer> _boogu_dit;
  // Mage-Flow's STATIC flow shift from <root>/scheduler/scheduler_config.json
  // (6.0 in every published checkpoint; use_dynamic_shifting is false).
  double _mage_shift = 6.0;
  // On a memory-bounded box (DiT + the conditioner's resident encoder can't fit
  // a large decode too), drop the DiT's per-forward scratch after each
  // generation so it doesn't crowd out the downstream vae-decode. Set from the
  // same RAM heuristic that decides DiT streaming.
  bool _release_scratch = false;

  // FLUX.2 free/reload: on a memory-bounded box a PRELOADED ~9 GB DiT leaves
  // no working-set room for a 1024px vae-decode. After a generation, if the
  // pending decode won't fit alongside the resident DiT, free the DiT weights
  // (they are mmap'd; the destructor releases the working set) BEFORE the
  // latent is published, then reload lazily when the next prompt arrives. The
  // load params are cached here so process() can reload without re-deriving.
  std::string _flux2_dit_dir;            // resolved transformer dir (reload)
  // Same pair for Qwen-Image-Edit. The 20B DiT is the largest of the
  // image families, so it is the one a 1024px vae-decode is most likely
  // to need the room from -- see free_qie_dit_for_decode_.
  std::string _qie_dit_dir;
  bool        _qie_stream = false;
  bool        _flux2_stream   = false;   // streaming mode used at load
  std::string _krea2_dit_dir;            // resolved Krea-2 DiT dir (reload)
  // Streaming mode used at load. Cached for the SAME reason
  // _flux2_stream is: a memory-driven free after a generation
  // reloads through load_krea2_dit_(), and a reload that forgot the
  // flag would come back PRELOADED -- quietly undoing the decision
  // on the box that needed it most.
  bool        _krea2_stream   = false;
  // The runtime adapter. The PATH is LOAD-TIME -- it decides whether a
  // fused-activation weave can be built at all -- so a later beat that
  // changes it is warned about rather than silently ignored; the scale
  // is live and is pushed straight at the model.
  //
  // Held as the REFERENCE the config gave (a registry key, a directory
  // or a file), not as a resolved path: it is compared against the next
  // beat's, and resolving first would make two spellings of one adapter
  // read as a change.
  // ONE pair, not one per family: a stage holds exactly one DiT, so it
  // has exactly one adapter. Seeded from this stage's OWN `lora` /
  // `lora_scale` config and overridden by a model_config beat that
  // names them.
  // The runtime LoRA SLOTS, off this stage's own config and any
  // model_config beat, in the order the DiT binds them. Two, so a
  // distillation or identity adapter and a style one can ride together
  // -- see kMaxLoraSlots on the transformers. The PATH is load-time; the
  // STRENGTH is not, and each slot's moves on its own.
  struct LoraSlot {
    std::string path;
    double      scale = 1.0;
  };
  static constexpr int kLoraSlots = 2;
  std::array<LoraSlot, kLoraSlots> _lora;
  int         _vae_base       = 128;     // VAE base ch (decode-peak est.)
  bool        _dit_unloaded   = false;   // freed after a gen; reload next beat

  // Turn what the config named -- a registry key, a directory holding
  // one .safetensors, or a file -- into the single file a loader opens.
  // Empty (with a warning naming the reason) when it cannot, so the
  // caller generates WITHOUT the adapter rather than failing the graph.
  std::string adapter_file_(const std::string& ref) const;

  // The slots as the family's own LoraSpec list, in the order the graph
  // named them -- which is the order the DiT binds and the order
  // `lora_scale(i)` addresses afterwards. A slot whose file cannot be
  // resolved is DROPPED rather than left as a hole, so the surviving
  // adapters keep consecutive indices and a strength beat still reaches
  // them. Templated because the two families spell LoraSpec separately
  // (they are the same two fields, but a shared type would put one
  // family's adapter contract in the other's header).
  template <class Spec>
  std::vector<Spec> lora_specs_() const
  {
    std::vector<Spec> out;
    for (const LoraSlot& sl : _lora) {
      if (sl.path.empty()) { continue; }
      Spec sp;
      sp.path = adapter_file_(sl.path);
      if (sp.path.empty()) { continue; }   // adapter_file_ warned
      sp.scale = (float)sl.scale;
      out.push_back(std::move(sp));
    }
    return out;
  }

  // (Re)load the FLUX.2 DiT from the cached params. Returns false on failure.
  bool load_flux2_dit_();
  // Free the FLUX.2 DiT if the pending gen_w x gen_h vae-decode won't fit the
  // current GPU working-set headroom alongside it. No-op otherwise / non-flux2.
  // Put this image's decode arena on the books. See the definition:
  // an edit graph has no config geometry to declare from.
  void publish_decode_arena_(std::size_t peak) const;
  void free_flux2_dit_for_decode_(int gen_w, int gen_h);
  // Same, for the Krea-2 DiT (its ~7 GB of resident weights otherwise starve
  // the shared MetalKrea2Vae's 1024px decode); reloaded on the next prompt.
  bool load_krea2_dit_();
  void free_krea2_dit_for_decode_(int gen_w, int gen_h);
  // Qwen-Image-Edit's pair of the same. The DiT is idle once its latent
  // is read back, so on a box that cannot hold 20B of weights beside the
  // decode arena it is given to the pool and reloaded on the next prompt.
  bool load_qie_dit_();
  void free_qie_dit_for_decode_(int gen_w, int gen_h);
  // Same, for Boogu-Image. Its 10B NextDiT is the largest DiT here (~20 GB
  // bf16, ~5.6 GB at 4-bit) and it shares the box with a resident Qwen3-VL
  // mllm, so on a 16 GB box it is freed for the decode even at 4-bit.
  std::string _boogu_dit_dir;
  bool        _boogu_stream   = false;
  bool load_boogu_dit_();
  // The manager's shared view of a checkpoint's weights (weight-set.h).
  std::shared_ptr<genai::WeightSet> weight_set_(const std::string& dir) const;
  void free_boogu_dit_for_decode_(int gen_w, int gen_h);

  // The active sampler (integrator) + scheduler (sigma schedule) specs. Seeded
  // from config (euler + simple / _steps / shift 1.15) and each overridden by
  // the first beat latched off the sampler / scheduler iports.
  genai::FlowSamplerSpec   _sampler_spec;
  genai::FlowSchedulerSpec _scheduler_spec;
  bool                     _sampler_latched   = false;
  bool                     _scheduler_latched = false;

  // A reference-image latent as it arrives on a ref-latent iport: the
  // vae-encode output, channel-first f32 [c, h, w] (FLUX.2 dit_channels @ H/16,
  // Krea-2 z_dim @ H/8). Cached across prompts once a beat is seen.
  struct RefLatent {
    std::vector<float> chw;
    int c = 0, h = 0, w = 0;
    bool empty() const { return chw.empty(); }
  };
  // Cached reference latents from iport4 / iport5 (read once when a beat is
  // available, reused for every later prompt like the negative prompt).
  RefLatent _ref[2];

  // The conditioning -> unpacked-latent forward: fuse the tapped text
  // conditioning through the DiT's text tower, sample the FlowMatchEuler turbo
  // steps and unpack. `cond` is the diffusion-conditioner's krea2 conditioning
  // (f16 [n_real, 12, EH]); `cond_neg` is the optional negative conditioning
  // (empty => no CFG). When `init_latent` is non-null (a whitened latent
  // [z, H/8, W/8], from the vae-encode stage) with strength>0 it is an img2img
  // run: scale_noise init, denoising only the tail steps. `init_packed` (when
  // non-null) overrides the packed init (repro / golden). `refs` are optional
  // reference-conditioning latents ([z_dim, h, w] channel-first) packed to DiT
  // tokens + given per-reference RoPE frame offsets. Returns the unpacked latent
  // [z, H/8, W/8] (whitened) or empty. When `emit_step` is set it is called with
  // the unpacked latent AFTER each sampler step (same [z, H/8, W/8] format) so
  // process() can stream it LIVE to the step_latent oport for a per-step preview.
  std::vector<float>
  generate_(const metal_compute::SharedBuffer& cond, int n_real,
            const metal_compute::SharedBuffer& cond_neg, int n_real_neg,
            int gen_h, int gen_w,
            const std::vector<float>* init_packed,
            const std::vector<float>* init_latent,
            const std::vector<RefLatent>& refs,
            const std::function<void(const std::vector<float>&)>& emit_step =
                {}) const;

  // FLUX.2 forward: `context` is the diffusion-conditioner's flux2 conditioning
  // (f16 [n, 3*enc_hidden] concatenated taps) -> FLUX DiT sampler loop (no
  // text-fusion tower, no CFG). `refs` are optional reference images
  // (patchify-packed to DiT tokens + given per-reference RoPE T offsets inside
  // the DiT). Returns the DiT-facing latent [dit_channels, H/16, W/16]
  // (channel-first) or empty.
  std::vector<float>
  generate_flux2_(const metal_compute::SharedBuffer& context, int n_real,
                  int gen_h, int gen_w,
                  const std::vector<RefLatent>& refs,
                  const std::vector<float>* init_packed = nullptr,
                  const std::function<void(const std::vector<float>&)>&
                      emit_step = {}) const;

  // Qwen-Image-Edit forward: `txt` is the diffusion-conditioner's qwen-image-edit
  // conditioning (bf16 [n_real, 3584] image-aware last-hidden, POST final-norm);
  // run the dual-stream MetalQwenImageTransformer over pure-noise packed latents
  // with the M2 dynamic-shift sampler and (when `txt_neg` is set + scale>1)
  // norm-preserving true-CFG. `refs` are reference latents (vae-encode output
  // [16, h, w]) packed 2x2 to DiT tokens + appended in their own RoPE frame
  // bands. Returns the unpacked, whitened latent [16, H/8, W/8] (channel-first)
  // or empty.
  // Mage-Flow forward: `txt` is the diffusion-conditioner's mage-flow
  // conditioning (bf16 [n_real, 2560] single last-hidden tap, POST final
  // norm). MageVAE downsamples 16x and the DiT is patch_size 1, so a latent
  // pixel IS a token: no 2x2 pack/unpack anywhere on this path. `refs` are
  // MageVAE reference latents ([128, h, w] channel-first from vae-encode),
  // appended CLEAN after the target in their own RoPE frame bands -- the
  // sampler steps only the target tokens. FlowMatchEuler with the checkpoint's
  // STATIC shift 6.0 (scheduler_config.json, use_dynamic_shifting false).
  // Returns the latent [128, H/16, W/16] (channel-first) or empty.
  std::vector<float>
  generate_mage_(const metal_compute::SharedBuffer& txt, int n_real,
                 const metal_compute::SharedBuffer& txt_neg, int n_real_neg,
                 int gen_h, int gen_w,
                 const std::vector<float>* init_packed,
                 const std::vector<RefLatent>& refs,
                 const std::function<void(const std::vector<float>&)>&
                     emit_step = {}) const;

  // Boogu-Image forward: `txt` is the diffusion-conditioner's boogu-image
  // conditioning (bf16 [n_real, 4096] mllm last-hidden POST final norm, over
  // the WHOLE templated sequence -- Boogu drops no prefix). Runs the NextDiT
  // over patch-packed latents with either the DMD student's ascending-sigma
  // jump+renoise loop (the distilled Turbo default) or Euler along the v1
  // logistic time-shift schedule (Base/Edit, with optional CFG). `refs` are
  // vae-encode reference latents ([16, h, w] channel-first), packed 2x2 and
  // embedded by the DiT's own ref_image_patch_embedder. Returns the unpacked,
  // whitened latent [16, H/8, W/8] (channel-first) or empty.
  std::vector<float>
  generate_boogu_(const metal_compute::SharedBuffer& txt, int n_real,
                  const metal_compute::SharedBuffer& txt_neg, int n_real_neg,
                  int gen_h, int gen_w,
                  const std::vector<float>* init_packed,
                  const std::vector<RefLatent>& refs,
                  const std::function<void(const std::vector<float>&)>&
                      emit_step = {}) const;

  std::vector<float>
  generate_qie_(const metal_compute::SharedBuffer& txt, int n_real,
                const metal_compute::SharedBuffer& txt_neg, int n_real_neg,
                int gen_h, int gen_w,
                const std::vector<float>* init_packed,
                const std::vector<RefLatent>& refs,
                const std::function<void(const std::vector<float>&)>&
                    emit_step = {}) const;
#endif
};

}

#endif

#ifndef VPIPE_STAGES_DIFFUSION_CONDITIONER_STAGE_H
#define VPIPE_STAGES_DIFFUSION_CONDITIONER_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

// The prompt/image -> conditioning half of the diffusion split. Owns the
// tokenizer, the text encoder, and (for image-aware families) a vision tower
// (Qwen2.5-VL for Qwen-Image-Edit, Qwen3-VL for Krea-2 edit); emits the
// per-family conditioning tensor the generate-image (DiT) stage consumes. This
// is the from-scratch, MLX-free metal-compute path on the
// VPIPE_BUILD_APPLE_SILICON axis; an inert stub off it.
#ifdef VPIPE_BUILD_APPLE_SILICON
#include "generative-models/mage/mage-screen.h"
#include "generative-models/qwen-image/metal-qwen25-vision.h"
#include "generative-models/qwen3/metal-qwen-model.h"
#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/minimax-h3/minimax-h3-text-encoder.h"
#include "generative-models/wan/metal-umt5-encoder.h"
#include "generative-models/qwen3/metal-qwen-vision.h"
#include "generative-models/shared/grounded-encode-params.h"
#include "generative-models/tokenizer.h"
#include "generative-models/weight-set.h"
#include "stages/model-memory.h"
#endif

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {

// DiffusionConditionerStage: prompt text (+ optional reference image for
// image-aware models) -> conditioning embeddings for a diffusion DiT.
//
// This is the encoder half pulled out of the generate-image stage, so the raw
// reference IMAGE goes to the VLM here (semantic understanding) while the VAE
// LATENT of that same image goes to the DiT stage (spatial detail) -- two
// distinct consumers, no longer conflated on one stage. The conditioner and the
// generate-image stage are a MATCHED PAIR keyed by the same `hf_dir`: the
// conditioning tensor's shape + semantics are family-specific.
//
//   iport0  prompt        FlexDataPayload (string or {text: ...}), required.
//   iport1  negative       OPTIONAL FlexDataPayload negative prompt (for the
//                          DiT's classifier-free guidance). Its conditioning is
//                          emitted on oport1.
//   iport2  model          OPTIONAL FlexDataPayload model reference (from a
//                          `model-select` source). Latched once; OVERRIDES the
//                          hf_dir config so the conditioner / DiT / vae stages
//                          of a graph share one model. When wired, the encoder
//                          load is DEFERRED to the first process() (the beat
//                          arrives only after the init barrier).
//   iport3  ref_image      OPTIONAL planar U8 RGB TensorBeat [3,H,W] (load-image
//                          format). Image-aware families run it through their
//                          vision tower to make the prompt embeds understand the
//                          source: Qwen-Image-Edit via Qwen2.5-VL, Krea-2 edit
//                          (identity-edit LoRA) via Qwen3-VL. FLUX.2 ignores it.
//                          Latched once.
//   iport4  ref_image2     OPTIONAL SECOND reference image, same format.
//                          Qwen-Image-Edit-2511 is a MULTI-reference edit model
//                          and Mage-Flow-Edit's template has a per-reference
//                          body, so on those families both pictures have to be
//                          understood by the VLM -- the DiT's ref_latent1 only
//                          carries the second picture's spatial detail. Each
//                          reference gets its own vision block (the family's own
//                          convention: "Picture N: " for Qwen-Image-Edit,
//                          "Image N: " for Mage-Flow, bare back-to-back blocks
//                          for Boogu), its own 2-D mROPE band and its own
//                          deepstack run. Krea-2 is single-reference by design
//                          (ComfyUI-Krea2Edit) and ignores it with a warning.
//
//   oport0  conditioning   TensorBeatPayload bf16, family-shaped:
//                            krea2  [n_real, 12, 2560] (12-tap; the DiT fuses)
//                            flux2  [n_real, 3*enc_hidden] (concatenated taps)
//                            qwen-image-edit [n_real, 3584] (last-hidden,
//                              image-aware, POST encoder final-norm)
//                            boogu-image [n, 4096] (last-hidden, image-aware,
//                              POST final-norm, and n is the WHOLE templated
//                              sequence -- Boogu drops no prefix)
//                            minimax-h3 [n, 5120] -- Qwen3-VL-32B
//                              hidden_states[50] of 64, UN-NORMED, over a
//                              VERBATIM prompt (no chat template, no
//                              special tokens, no padding)
//   oport1  neg_conditioning  same shape, emitted only when a negative is set.
//
// MAGE-FLOW CONTENT SCREEN. On the `mage-flow` family every prompt is first
// run through the model's own content-policy classifier (mage-screen.h) --
// mandatory, no config key, no port to leave unwired. A refused prompt gets
// a one-row conditioning beat tagged `content_blocked` on its sideband, which
// generate-image turns into a skipped denoise and vae-decode into a blank
// refusal image. With a reference image the classifier judges the SOURCE
// PICTURE as well as the instruction. It fails CLOSED.
//
// Config (FlexData object):
//   hf_dir     (string, OPTIONAL) -- the model dir (text_encoder/, transformer/,
//                                    tokenizer/); the transformer's _class_name
//                                    selects the family + encoder. A
//                                    model-select source on iport2 overrides it;
//                                    required only when iport2 is unwired.
class DiffusionConditionerStage final
    : public TypedStage<DiffusionConditionerStage> {
public:
  static constexpr const char* kTypeName = "diffusion-conditioner";

  DiffusionConditionerStage(const SessionContextIntf* session,
                            std::string               id,
                            std::vector<InEdge>       iports,
                            FlexData                  config);
  ~DiffusionConditionerStage() override;

  Job initialize(RuntimeContext& ctx) override;

  // The text encoder this stage loads, plus the DiT it is paired with --
  // the same two directories its footprint query uses, so the
  // declaration and the estimate cannot disagree.
  std::vector<ResourceClaim> declare_resources() const override;
  void reset_run_state() override;
  Job process   (RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  const std::string& hf_dir() const noexcept { return _hf_dir; }
  std::uint64_t conditionings_emitted() const noexcept { return _emitted; }
  // How many prompts the Mage-Flow content screen refused. Counted in the
  // emitted total too: a refusal IS a conditioning beat (a blocked one).
  std::uint64_t blocked_by_policy() const noexcept { return _blocked; }

private:
  std::string _hf_dir;
  std::string _enc_dir;
  // krea2 | flux2 | qwen-image-edit | mage-flow | boogu-image
  std::string _family = "krea2";
  std::uint64_t _emitted = 0;
  std::uint64_t _blocked = 0;      // refused by the Mage-Flow content screen
  // Image-aware families: always emit a grounded negative (empty prompt ok) on
  // oport1 so the DiT can run CFG>1 (Krea-2 edit deletion recipe). Config flag.
  bool _grounded_negative = false;
  // Model iport bookkeeping (used only when a model-select source is wired):
  // latch the model beat once, and load the encoder at most once (lazily,
  // since the beat only arrives after the init barrier).
  bool _model_latched  = false;
  bool _load_attempted = false;
  bool _cfg_latched    = false;
  // The last model-config beat, held UNPARSED: which family reads it is
  // not known until the checkpoint resolves, and the two beats arrive on
  // different ports in either order.
  FlexData _model_cfg;
  // The idle-unload decision is taken at the first process() (post
  // init-barrier), not at load; see resolve_unload_policy_().
  bool _unload_resolved = false;
  std::vector<std::string> _peer_dirs;

#ifdef VPIPE_BUILD_APPLE_SILICON
  // How the resident family wants a reference image prepared before its
  // vision tower sees it. Seeded from the model layer's per-family
  // numbers the moment the family is known, then overlaid with whatever
  // the config beat set. See genai::GroundedEncodeParams.
  genai::GroundedEncodeParams _ground;
  // Re-seed `_ground` for the resident family and re-apply `_model_cfg`.
  // Runs on every change to either, because the two arrive on different
  // ports in either order.
  void apply_model_config_();
  // Resolve _hf_dir + load the tokenizer / text encoder / embeds (idempotent:
  // the _load_attempted guard runs the body at most once). Called from
  // initialize() (config model) or the first process() (model iport). No-op
  // when _hf_dir is still empty.
  void ensure_loaded_();

  // The encoder checkpoint, from the session's model manager. The text
  // encoder, the vision tower and the embedding table below all live in
  // THIS one directory and used to open it three times over (plus once
  // more per conditioning call, for the final-norm vector); they now
  // share this. Held for the stage's life: their tensors come from it.
  std::shared_ptr<genai::WeightSet>              _enc_ws;
  std::unique_ptr<genai::MetalQwenModel>          _encoder;
  // The Wan family's tower is a umT5-XXL ENCODER, not a decoder-only
  // LM, so it is its own member rather than another config of
  // _encoder -- different weights, different attention (an additive
  // relative-position bias, no rotary) and no embedding table to
  // gather separately.
  std::unique_ptr<genai::MetalUmt5Encoder>        _umt5;
  // MiniMax-H3's tower: a Qwen3-VL-32B tapped at an INTERMEDIATE layer
  // rather than run to its last hidden state, which is why it is its own
  // class rather than another encoder_config_*() over _encoder.
  std::unique_ptr<genai::MiniMaxH3TextEncoder>   _h3_enc;
  std::unique_ptr<genai::Tokenizer>               _tokenizer;
  metal_compute::SharedBuffer                     _embed;      // encoder embeds
  mutable std::unique_ptr<genai::MetalQwen25Vision> _vision;   // QIE, lazy
  mutable std::unique_ptr<genai::MetalQwenVisionEncoder> _vision3;  // krea2, lazy
  int _enc_hidden = 2560;

  // Cached negative prompt + raw reference images, latched once like the DiT
  // stage's negative / ref-latent inputs. kMaxRefs is the number of ref_image
  // iports; `_n_ref` is how many actually arrived (they latch independently, so
  // a graph may wire only the second).
  static constexpr int kMaxRefs = 2;
  std::string               _negative_prompt;
  bool                      _negative_latched = false;
  std::vector<std::uint8_t> _ref_rgb[kMaxRefs];
  int _ref_rgb_h[kMaxRefs] = {0, 0};
  int _ref_rgb_w[kMaxRefs] = {0, 0};
  int _n_ref = 0;                  // contiguous count actually latched
  // Qwen3-VL deepstack features (krea2 grounded encode), bf16 [n_img, EH] each,
  // one per vision deepstack merger. Computed once in vision_tokens_, ADDED to
  // the encoder hidden states at the image rows after LM layers 0.. by encode_.
  mutable std::vector<metal_compute::SharedBuffer> _ds_feats;
  // Per-reference merged vision grid (mh, mw) and token count -- the 3-axis
  // mROPE position_ids give each reference its OWN band, so these are per
  // image, not global.
  mutable int _img_mh[kMaxRefs] = {0, 0};
  mutable int _img_mw[kMaxRefs] = {0, 0};
  mutable int _img_tok[kMaxRefs] = {0, 0};
  mutable int _img_n = 0;          // references the tower actually encoded

  bool load_encoder_(metal_compute::MetalCompute* mc);

  // ---- idle unload -------------------------------------------------------
  // The text encoder is the second-largest resident block in a diffusion
  // pipeline (Boogu's Qwen3-VL mllm is ~16 GB bf16 / ~4.7 GB at 4-bit) and it
  // is needed only while a prompt is being encoded -- the DiT then runs for
  // seconds to minutes with the encoder sitting idle. On a box that cannot hold
  // both, drop it once the conditioning beats are out and reload it when the
  // next prompt arrives. `unload_when_idle`: auto (RAM heuristic) |
  // destroy | park | keep (legacy: always = destroy, never = keep).
  // Resolved once, after the init barrier, into _idle_action.
  model_memory::UnloadPolicy _unload_cfg = model_memory::UnloadPolicy::kAuto;
  // What happens to the encoder between prompts. Never kAuto after
  // resolve_unload_policy_() has run.
  model_memory::UnloadPolicy _idle_action = model_memory::UnloadPolicy::kKeep;
  bool _unloaded    = false;   // dropped; reload before the next encode
  std::string _root_dir;       // resolved pipeline root (peer weight sizing)

  // Resolve `_idle_action` from the post-barrier footprint. Idempotent;
  // called at the top of every process().
  void resolve_unload_policy_();
  // Whichever of destroy/park/keep was resolved, applied at an idle
  // point. The single call site for "the prompt is done with me".
  void release_encoder_when_idle_();
  void park_encoder_();
  void unload_encoder_();
  bool reload_encoder_();

  // Encode `text` (+ the cached reference image, for image-aware families) into
  // the family conditioning tensor; returns the bf16 buffer + sets `n_real`.
  // `vtok`/`n_img` carry the (already-computed) vision tokens for QIE (empty
  // for text-only). Empty buffer on failure.
  metal_compute::SharedBuffer
  encode_(const std::string& text, const char* which, int& n_real,
          const metal_compute::SharedBuffer& vtok, int n_img) const;

  // Run the reference image through the family vision tower -> vision tokens,
  // sets n_img. QIE: Qwen2.5-VL, bf16 [n_img, 3584]. Krea-2 edit: Qwen3-VL,
  // f16 [n_img, 2560]. Empty when no ref image / not image-aware.
  metal_compute::SharedBuffer vision_tokens_(metal_compute::MetalCompute* mc,
                                             int& n_img) const;

  // Mage-Flow's MANDATORY content screen over the encoder this stage already
  // owns (see generative-models/mage/mage-screen.h). With a reference image
  // the multimodal EDIT policy runs -- it judges the source picture as well
  // as the instruction. Never throws; blocks on every failure.
  genai::MageScreenVerdict screen_(const std::string&                 prompt,
                                   const metal_compute::SharedBuffer& vtok,
                                   int                                n_img)
      const;
#endif
};

}  // namespace vpipe

#endif

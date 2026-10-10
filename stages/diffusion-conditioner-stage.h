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
#include "generative-models/content-screen.h"
#ifdef VPIPE_BUILD_APPLE_SILICON
#include "generative-models/qwen-image/metal-qwen25-vision.h"
#include "generative-models/qwen3/metal-qwen-model.h"
#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/minimax-h3/minimax-h3-text-encoder.h"
#include "generative-models/wan/metal-umt5-encoder.h"
#include "generative-models/qwen3/metal-qwen-vision.h"
#include "generative-models/shared/grounded-encode-params.h"
#include "generative-models/tokenizer.h"
#include "generative-models/vosr/metal-dinov2-encoder.h"
#include "generative-models/weight-set.h"
#include "stages/generation-input.h"
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
//                          Qwen-Image-Edit-2511 is a MULTI-reference edit model,
//                          and a family whose template has a per-reference body
//                          needs the same, so on those families both pictures
//                          have to be understood by the VLM -- the DiT's
//                          ref_latent1 only carries the second picture's spatial
//                          detail. Each reference gets its own vision block (the
//                          family's own convention: "Picture N: " for
//                          Qwen-Image-Edit, a profile's `ref_label`, bare
//                          back-to-back blocks for Boogu), its own 2-D mROPE
//                          band and its own
//                          deepstack run. Krea-2 is single-reference by design
//                          (ComfyUI-Krea2Edit) and ignores it with a warning.
//   iport6  ref_images     OPTIONAL reference images as ONE LIST (load-image's
//                          `images`): as many as the graph has, each in
//                          ref_image's format, following whatever is on
//                          iport3 / iport4 in list order. Last, so a graph
//                          written before it keeps its port numbers.
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
// CONTENT SCREEN. A family that provides one (generative-models/
// content-screen.h) has every prompt first judged by it, on this stage's
// own encoder -- mandatory, no config key, no port to leave unwired. A
// refused prompt gets a one-row conditioning beat tagged `content_blocked`
// on its sideband, which generate-image turns into a skipped denoise and
// vae-decode into a blank refusal image. With reference images every
// SOURCE PICTURE is judged as well as the instruction. It fails CLOSED.
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

  // Whether the encoder is phase-limited -- a decision, so it is taken
  // in the second planning pass where the whole graph has declared and
  // every stage gets the same answer. See Stage::decide_resources.
  std::vector<ResourceClaim> decide_resources() const override;

  // See Stage::declare_memory. The encoder plus the conditioning on
  // oport 0; the DiT is generate-video's to declare, not this stage's,
  // even though this stage names its directory in a claim.
  StageMemory declare_memory() const override;

  // Latch a `model-select` constant before the planning phase, so
  // the claim above is made against the model this graph will
  // actually run rather than against an empty hf_dir.
  void apply_constant(unsigned iport, const FlexData& beat) override;
  void reset_run_state() override;
  Job process   (RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  const std::string& hf_dir() const noexcept { return _hf_dir; }
  std::uint64_t conditionings_emitted() const noexcept { return _emitted; }
  // How many prompts the family's content screen refused. Counted in the
  // emitted total too: a refusal IS a conditioning beat (a blocked one).
  std::uint64_t blocked_by_policy() const noexcept { return _blocked; }

private:
  std::string _hf_dir;
  std::string _enc_dir;
  // VOSR only: the DINOv2 checkpoint, which is not under _hf_dir. Empty
  // until initialize() resolves it.
  std::string _venc_dir;
  // How a reference image pairs with the prompts: the `reference_mode`
  // config key, as the mode of each reference input. "auto" is the rule
  // every input here follows (stages/generation-input.h), which serves an
  // edit graph and a folder of pictures alike.
  GenerationInput::Mode _ref_mode = GenerationInput::Mode::kAuto;
  // krea2 | flux2 | qwen-image-edit | boogu-image | ... | a registered
  // family's tag
  std::string _family = "krea2";
  // Within the qwen-image-edit family, WHICH of the two Qwen-Image
  // recipes this checkpoint takes. They are one architecture published
  // as two diffusers pipelines with two system prompts, and the weights
  // cannot tell them apart, so this is read from the pipeline the
  // checkpoint declares (or the model_type the registry recorded) and
  // never from whether a reference image happens to be wired. See
  // qwen_image_is_t2i_.
  bool _qie_t2i = false;
  // What the models DB recorded for this checkpoint, when a model-select
  // source named one. Only used to ask a registered family's `claims`,
  // which is the one probe a bare weights directory can answer.
  std::string _model_type;
  // True once the checkpoint has been resolved and the family is the
  // one that will run. Before it, `_family` is a default nobody chose,
  // and calling a config beat a mismatch against it is noise about a
  // decision that has not been taken.
  bool _family_settled = false;
  // The family's conditioning profile, when a plugin registered one --
  // an open FlexData bag of the facts that differ between checkpoints.
  // Null for every built-in family, and null is the case the host's own
  // per-family defaults serve. BORROWED from a process-wide registry
  // that outlives every stage. See
  // generative-models/conditioner-profile.h.
  const FlexData* _profile = nullptr;
  // The family's content screen, when it provides one through its model
  // family's `query_extension`. Its presence is the whole decision: no
  // key turns it off. BORROWED from the registry, which outlives every
  // stage. See generative-models/content-screen.h.
  genai::screen::ContentScreen* _screen = nullptr;
  std::uint64_t _emitted = 0;
  std::uint64_t _blocked = 0;      // refused by the family's content screen
  // Image-aware families: always emit a grounded negative (empty prompt ok) on
  // oport1 so the DiT can run CFG>1 (Krea-2 edit deletion recipe). Config flag.
  bool _grounded_negative = false;
  // Model iport bookkeeping (used only when a model-select source is wired):
  // latch the model beat once, and load the encoder at most once (lazily,
  // since the beat only arrives after the init barrier).
  bool _model_latched  = false;
  bool _load_attempted = false;
  // The model config: a generation input, broadcast when its source sends
  // once -- which a config source without a trigger does.
  GenerationInput _cfg_in;
  // The prompt (iport0), a generation input like the rest: one prompt
  // and a folder of reference pictures condition every picture on it.
  GenerationInput _prompt_in;
  std::string     _prompt;
  // The prompt beats say they are a series (beat::kBatch): hold the
  // encoder until end of stream. And the current one may be empty.
  bool            _batch = false;
  bool            _batch_checked = false;
  // A batch on a box that holds the encoder whole loads it resident
  // (see process()); sticky for the stage's life, like the load itself.
  bool            _resident_for_batch = false;
  bool            _allow_empty = false;
  // A batch on a STREAMED encoder holds prompts back and encodes them
  // layer-major, this many at a time (VPIPE_COND_BATCH).
  std::vector<std::string> _pend;
  int             _batch_max = 32;
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
  // VOSR's conditioner: a DINOv2 ViT-L/14 and nothing else. It is the
  // WHOLE conditioning -- there is no prompt, no tokenizer and no text
  // encoder on this path, so every text member above stays null.
  std::unique_ptr<genai::MetalDinov2Encoder>      _dinov2;
  int _dino_size = 448;
  metal_compute::SharedBuffer                     _embed;      // encoder embeds
  mutable std::unique_ptr<genai::MetalQwen25Vision> _vision;   // QIE, lazy
  mutable std::unique_ptr<genai::MetalQwenVisionEncoder> _vision3;  // krea2, lazy
  int _enc_hidden = 2560;

  // The negative prompt + raw reference images, each a generation input
  // paired with the prompt the way generate-image pairs its own.
  std::string               _negative_prompt;
  GenerationInput           _negative_in;
  // One reference picture, RGB planar U8 [3, h, w].
  struct RefImage {
    std::vector<std::uint8_t> rgb;
    int h = 0, w = 0;
    bool empty() const { return rgb.empty(); }
  };
  // What each reference INPUT holds: ref_image / ref_image2 (kRefPorts
  // of them), then the ref_images list, each a generation input in
  // `reference_mode`.
  static constexpr int kRefPorts = 2;
  RefImage              _ref_port[kRefPorts];
  GenerationInput       _ref_port_in[kRefPorts];
  std::vector<RefImage> _ref_list;
  GenerationInput       _ref_list_in;
  // THE REFERENCES, as the model sees them: the two ports, then the
  // list, packed -- so "reference i" is the i-th picture the VLM sees
  // even when only the second port is wired. No fixed count.
  std::vector<std::vector<std::uint8_t>> _ref_rgb;
  std::vector<int> _ref_rgb_h, _ref_rgb_w;
  int _n_ref = 0;                  // == _ref_rgb.size()
  // Qwen3-VL deepstack features (krea2 grounded encode), bf16 [n_img, EH] each,
  // one per vision deepstack merger. Computed once in vision_tokens_, ADDED to
  // the encoder hidden states at the image rows after LM layers 0.. by encode_.
  mutable std::vector<metal_compute::SharedBuffer> _ds_feats;
  // Per-reference merged vision grid (mh, mw) and token count -- the 3-axis
  // mROPE position_ids give each reference its OWN band, so these are per
  // image, not global.
  mutable std::vector<int> _img_mh, _img_mw, _img_tok;
  // Reference i's merged grid, 0 past the end -- which is what a prompt
  // with no reference reads for reference 0.
  int
  img_mh_(std::size_t i) const
  {
    return i < _img_mh.size() ? _img_mh[i] : 0;
  }
  int
  img_mw_(std::size_t i) const
  {
    return i < _img_mw.size() ? _img_mw[i] : 0;
  }
  mutable int _img_n = 0;          // references the tower actually encoded

  // Qwen-Image-2.1's joint-sequence bookkeeping, published on the
  // conditioning beat's sideband. Nothing downstream can recover it:
  // an image row's embedding is a tower output, not a token id, so
  // which rows are image slots is knowable only here.
  mutable std::vector<std::uint8_t> _qi21_slots;
  mutable std::vector<int> _qi21_grid_h, _qi21_grid_w;
  mutable int _qi21_nref = 0;

  bool load_encoder_(metal_compute::MetalCompute* mc);
  // The VOSR path: one picture in, one conditioning beat out, driven by
  // the ref_image iport rather than by a prompt.
  bool load_dinov2_(metal_compute::MetalCompute* mc,
                    const std::string& root);
  Job  process_vosr_(RuntimeContext& ctx);

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

  // Encoder / DiT directories for this stage's model. Shared by both
  // planning passes so they name the same checkpoints -- the decide
  // pass refines a declaration keyed by path.
  void resolve_component_dirs_(std::string* enc_out,
                               std::string* dit_out) const;

  // Encode `text` (+ the cached reference image, for image-aware families) into
  // the family conditioning tensor; returns the bf16 buffer + sets `n_real`.
  // `vtok`/`n_img` carry the (already-computed) vision tokens for QIE (empty
  // for text-only). Empty buffer on failure.
  metal_compute::SharedBuffer
  encode_(const std::string& text, const char* which, int& n_real,
          const metal_compute::SharedBuffer& vtok, int n_img) const;

  // Qwen-Image-2.1: the template's ids for `text` and how many leading
  // ones the reference drops (the system turn).
  bool qi21_ids_(const std::string& text, int nref,
                 std::vector<std::int32_t>* ids, int* drop) const;
  // The encoder's input rows for `ids`, gathered from its embedding table.
  metal_compute::SharedBuffer
  embed_rows_(const std::vector<std::int32_t>& ids) const;
  // The conditioning beat's Qwen-Image-2.1 sideband: the slot mask and
  // each reference's latent grid.
  static FlexData qi21_sideband_(const std::vector<std::uint8_t>& slots,
                                 int nref, const std::vector<int>& grid_h,
                                 const std::vector<int>& grid_w);
  // The prompts held back from a batch on a streamed encoder, encoded in
  // one layer-major pass and emitted in order.
  bool batchable_() const;
  Job flush_batch_(RuntimeContext& ctx);

  // Run the reference image through the family vision tower -> vision tokens,
  // sets n_img. QIE: Qwen2.5-VL, bf16 [n_img, 3584]. Krea-2 edit: Qwen3-VL,
  // f16 [n_img, 2560]. Empty when no ref image / not image-aware.
  metal_compute::SharedBuffer vision_tokens_(metal_compute::MetalCompute* mc,
                                             int& n_img) const;

  // The family's MANDATORY content screen over the encoder this stage
  // already owns (see generative-models/content-screen.h). With reference
  // images it runs once PER REFERENCE, each pass bound to that picture,
  // and blocks if any pass does. Never throws; blocks on every failure
  // it can see.
  genai::screen::Verdict screen_(const std::string&                 prompt,
                                 const metal_compute::SharedBuffer& vtok,
                                 int                                n_img)
      const;
#endif
};

}  // namespace vpipe

#endif

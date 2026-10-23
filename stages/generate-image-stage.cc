#include "stages/generate-image-stage.h"
#include "generative-models/gen-input.h"
#include "common/beat-keys.h"

#include "stages/model-memory.h"
#include "stages/model-provenance.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/oport-policy.h"
#include "common/vpipe-format.h"
#include "generative-models/image-model-registry.h"
#include "generative-models/krea2/krea2-native-checkpoint.h"
#include "generative-models/shared/dit-block-progress.h"
#include "generative-models/shared/i8-gemm.h"
#include "generative-models/shared/torch-randn.h"
#include "stages/denoise-progress.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/weight-set.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "stages/model-config-source.h"
#include "stages/model-registry.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/shared/runtime-lora.h"

// The stage offers exactly the slots a DiT binds: fewer leaves some
// unreachable, more names adapters the loader warns about and drops.
static_assert(vpipe::lora_slots::kSlots == vpipe::genai::lora::Stack::kMax,
              "stages/lora-slots.h and lora::Stack disagree on the slots");
#endif

#include <type_traits>
#include <sys/sysctl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vpipe {

namespace {

// The Qwen-Image-Edit DiT runs bf16 (its residual stream exceeds f16's 65504);
// its packed-latent + velocity buffers are raw bf16, not _Float16.
inline std::uint16_t
f32_to_bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
inline float
bf16_to_f32_(std::uint16_t b)
{
  std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// How a reference latent pairs with the conditioning beats -- the
// ctor's own three, for the editor's dropdown.
constexpr SpecExtra kReferenceModeChoices[] = {
  {spec_key::kChoices, "auto,latch,per_beat"},
};
constexpr SpecExtra kI8TierChoices[] = {
  {spec_key::kChoices, "accurate,fast,token,token_plain"},
};
constexpr SpecExtra kSeedSequenceExtra[] = {
  {spec_key::kChoices, kSeedSequenceChoices},
};
// The catalogued families this stage can adapt, for every LoRA field's
// picker. A comma list so the picker offers each and refuses the rest --
// a Krea-2 adapter on a FLUX.2 DiT binds nothing and says so, but not
// until it has been fetched. (Z-Image has no catalogued adapter to
// offer.)
constexpr std::string_view kLoraTypes =
    "krea2-lora,flux2-lora,flux3-lora,qwen-image-21-lora,fibo2-lora";
// Slots 3..8 here: the shared doc, plus what is true only on this stage.
constexpr std::string_view kLoraMoreDoc =
    "one more runtime LoRA slot, read exactly as `lora2` is: applied "
    "alongside the others (every adapted projection adds s B (A x) per "
    "slot), the same accepted forms, LOAD-time, with its own live "
    "strength in the matching `_scale` key. Up to eight adapters run at "
    "once; each costs one more pair of skinny GEMMs, and the base weight "
    "is still read once. The same key on the family's model-config beat "
    "OVERRIDES this";
const ConfigKey kAttrs[] = {
  {.key = "hf_dir", .type = ConfigType::String, .required = false,
   .doc = "Krea-2-Turbo / FLUX.2 model dir (text_encoder/, transformer/, "
          "tokenizer/); an original or model-quantize'd (self-contained) "
          "pipeline. OPTIONAL: a model-select source on the model iport "
          "overrides it",
   .suggest_db = kModelRegistryDb, .suggest_db_type =
       "krea2,flux2,qwen-image,qwen-image-edit,qwen-image-21,"
       "boogu-image,boogu-image-edit,z-image,flux3,moss-image,fibo2,vosr",
   .model_channel = "diffusion-model"},
  {.key = "dit_dir", .type = ConfigType::String, .required = false,
   .doc = "override DiT dir (e.g. a quantized 4/8-bit DiT); else "
          "<hf_dir>/transformer. Krea-2 also takes a single .safetensors "
          "in Krea's own layout -- a community fine-tune, FP8 included -- "
          "and Qwen-Image-2.1 a single-file DiT under its diffusers names "
          "(unsloth's FP8) or a Draw Things .ckpt, each read as it is, with "
          "no conversion step",
   .suggest_db = kModelRegistryDb,
   .suggest_db_type =
       "krea2-dit,flux2-dit,qwen-image-edit-dit,qwen-image-21-dit,"
       "boogu-image-dit,z-image-dit,flux3-dit,moss-image-dit,fibo2-dit",
   .is_path = true, .path_filter = "*.safetensors,*.ckpt"},
  {.key = "strength", .type = ConfigType::Real, .required = false,
   .doc = "img2img strength in [0,1]; 0 (default) = text-to-image from noise "
          "(the init latent arrives on the `latent` iport from vae-encode)"},
  {.key = "height", .type = ConfigType::Int, .required = false,
   .doc = "output height, multiple of 16. Unset TOGETHER with width = infer "
          "from ref_latent0 (iport5) at its family's VAE scale; 256 if there "
          "is no reference either"},
  {.key = "width", .type = ConfigType::Int, .required = false,
   .doc = "output width, multiple of 16. Unset TOGETHER with height = infer "
          "from ref_latent0 (iport5); 256 if there is no reference either"},
  {.key = "steps", .type = ConfigType::Int, .required = false,
   .doc = "turbo sampler steps (default 8; 1 on a VOSR restorer, which is "
          "distilled to one and gains nothing from more)"},
  {.key = "seed", .type = ConfigType::Int, .required = false,
   .doc = "initial-noise RNG seed (default 0). The run's FIRST generation "
          "uses it; `seed_sequence` decides what later ones use"},
  {.key = "seed_sequence", .type = ConfigType::String, .required = false,
   .doc = "the seed of each generation when a run makes several -- a "
          "prompt per beat, a folder of references. \"increment\" "
          "(default): `seed`, then `seed`+1, `seed`+2, ... in generation "
          "order, so any one of them can be re-run alone by setting its "
          "number. \"randomize\": ignore `seed` and draw each from the "
          "clock; every seed used is logged, and fits the `seed` field "
          "for a re-run. \"keep\": `seed` for every generation, which "
          "compares prompts on the same noise",
   .def_str = "increment", .extra = kSeedSequenceExtra},
  {.key = "guidance_scale", .type = ConfigType::Real, .required = false,
   .doc = "classifier-free guidance scale; 1 (default) disables CFG. >1 with "
          "a negative prompt on iport1 runs a 2nd DiT pass per step"},
  {.key = "init_latents", .type = ConfigType::String, .required = false,
   .doc = "debug: raw f32 packed initial latents [img_seq, 64] (repro/golden)"},
  {.key = "ane_ffn", .type = ConfigType::Bool, .required = false,
   .doc = "accelerated FFN (LOSSY, fp16): run the block feed-forward on "
          "the Apple Neural Engine, which on an M4 is the only unused "
          "compute on the die -- the GPU's f16 GEMM already sits at "
          "84-92% of roofline and its integer MAD is a QUARTER of the "
          "f16 rate, so there is no int8 lever there. The rows are split "
          "between the two engines and run concurrently. The ANE runs "
          "ONE shared module whose weights are runtime inputs, staged "
          "per block into IOSurface buffers: its memory is fixed (~1.4 "
          "GB for Krea-2, however many blocks use it), it compiles once "
          "per shape and is cached after, and it measured 10.4 TOPS on "
          "a Krea-2 feed-forward. The resource plan decides whether the "
          "module fits. A 4/8-bit affine checkpoint is dequantized "
          "into those buffers per block. Ignored by families that do "
          "not implement it -- they say so in the log"},
  {.key = "ane_qkv", .type = ConfigType::Bool, .required = false,
   .doc = "the fused q|k|v projection on the ANE too, as a SECOND module "
          "beside the feed-forward's, split the same way and sharing the "
          "one ANE worker. REQUIRES ane_ffn: the plan books both modules "
          "as one unit and grants them together. It costs its own weight "
          "slots and staging on top of the feed-forward's, so the plan "
          "may decline the pair where it would have granted the "
          "feed-forward alone. Implemented by Krea-2 (q|k|v|gate), "
          "FLUX.2 (the double blocks' image q|k|v), Qwen-Image-2.1 and "
          "Z-Image (q|k|v of every main block). MEASURED on an M4 Pro "
          "at 1024^2 with the feed-forward already there: Qwen-Image-2.1 "
          "1.08x more per forward. VPIPE_KREA2_ANE_QKV=1 / "
          "VPIPE_FLUX2_ANE_QKV=1 / VPIPE_QWEN_IMAGE21_ANE_QKV=1 / "
          "VPIPE_Z_IMAGE_ANE_QKV=1 also turn it on"},
  {.key = "ane_rows", .type = ConfigType::Real, .required = false,
   .doc = "share of the FFN's ROWS given to the ANE, the GPU taking the "
          "rest CONCURRENTLY (rows are independent in a feed-forward, so "
          "the split is exact -- no partial sums, no seam). 0 (default) "
          "lets the family balance from the two engines' measured rates; "
          "~12.6 TOPS ANE against ~7 TFLOP/s GPU puts it near 0.64. 1 "
          "gives the whole FFN to the ANE and idles the GPU for its "
          "duration. The crossing itself is free: measured at -0.3 to "
          "-1.0 ms against a GPU-only control with the same commit "
          "count, i.e. the ANE phase overlaps the GPU's commit/wait "
          "rather than serialising behind it"},
  {.key = "ane_templates", .type = ConfigType::String, .required = false,
   .doc = "directory of precompiled ANE module templates. Unused by the "
          "runtime-weight feed-forward, which emits its own graph; "
          "accepted so existing graphs keep loading"},
  {.key = "ane_layers", .type = ConfigType::Int, .required = false,
   .doc = "cap on how many blocks use the ANE feed-forward; 0 (default) "
          "means every dense block. Not a memory setting -- the blocks "
          "share one module whose cost is fixed -- so set it only to "
          "cover fewer blocks on purpose, e.g. for a benchmark"},
  {.key = "sage_attn", .type = ConfigType::Bool, .required = false,
   .doc = "accelerated attention (LOSSY): SageAttention runs the QK^T "
          "product of the flash kernel in INT8, one scale per attention "
          "block, with K quantized as K - mean(K) over tokens -- which is "
          "EXACT rather than approximate, since a per-channel shift moves "
          "every score in a row equally and softmax does not see it. P*V "
          "stays in the tensor dtype. MEASURED on an M5, 8 heads x 20036 "
          "rows x 128: 156.6 -> 130.6 ms including the quantization "
          "prologue (1.20x), at cosine 0.99992 against a double-precision "
          "reference where the f16 kernel is also 0.99992. MATRIX CORES "
          "ONLY -- the int8 fragment MMA is an M5 instruction with no ALU "
          "fallback, so a box without them says so once and runs dense. "
          "Taken by flux2, krea2 and qwen-image; INDEPENDENT of i8_gemm "
          "and settable with it. Off by default: it is an approximation, "
          "and which images it is safe on is a judgement about the "
          "model, not about the kernel",
   .def_bool = false},
  {.key = "sage_dense_layers", .type = ConfigType::Int, .required = false,
   .doc = "leading transformer blocks left in the tensor dtype, untouched "
          "by SageAttention. 0 -- the default -- quantizes every block",
   .def_int = 0},
  {.key = "sage_pv8", .type = ConfigType::Bool, .required = false,
   .doc = "with sage_attn: the P*V product in INT8 TOO (LOSSY, more so "
          "than sage_attn alone). The flash kernel holds the product in "
          "int32 across key blocks with no f32 output tile -- P as uint8 "
          "against the row's running max, V with one scale per channel "
          "over the sequence, offset by 128 -- and rescales the held sum "
          "only when a row's max grows; Q goes per token. Every step and "
          "every row, the prompt's included. MEASURED on an M5 Pro, 8 heads "
          "x 20036 rows x 128 bf16: 44.0 ms against sage_attn's 50.6. "
          "Taken by flux2, flux3, krea2, moss-image, qwen-image and "
          "z-image; matrix cores only, as sage_attn",
   .def_bool = false},
  {.key = "i8_gemm", .type = ConfigType::Bool, .required = false,
   .doc = "accelerated mode (LOSSY): dynamic-int8 GEMMs for the DiT's big "
          "block matmuls, ~2x their f16 rate at int8 quality; IGNORED "
          "without NAX matmul2d (matrix-core GPU + kernels). Default "
          "false; env VPIPE_I8_GEMM overrides"},
  {.key = "i8_gemm_tier", .type = ConfigType::String, .required = false,
   .doc = "which int8 GEMM i8_gemm runs: 'accurate' (default; groups of 64 "
          "on both operands), 'fast' (the block weights held as rotated "
          "int8 at a group of 512: faster on the M5's matrix cores, ~1.1x "
          "accurate on an Ultra), 'token' (int8 per weight row x per token "
          "after a 256-wide Hadamard rotation: the fastest, ~1.4x accurate "
          "on an M5 Ultra at 1024^2) or 'token_plain' (the same unrotated: "
          "no faster, and visibly further from the dense picture; it "
          "reads a Draw Things i8x checkpoint's codes as stored, which "
          "'token' requantizes). A w4 pack's block weights take 'fast' "
          "and both token tiers too: the engine keeps their int8 copy "
          "where the box has room for it (twice the w4 bytes) and "
          "otherwise converts each before its GEMM, at ~3-5% of the "
          "denoise. Taken by "
          "qwen-image-21, krea2, flux2, z-image and moss-image on their "
          "image-row "
          "projections; others run accurate. Env VPIPE_I8_TIER overrides",
   .def_str = "accurate", .extra = kI8TierChoices},
  {.key = "keep_loaded", .type = ConfigType::Real, .required = false,
   .doc = "seconds to keep the DiT BUILT after this stage ends (its graph "
          "unloaded), on a box with room for it: the next graph asking for "
          "the same checkpoint and settings is handed it resident -- no "
          "read, no int8 layout made again. Renewed at each end; released "
          "sooner when a launch that does not use it needs the room. 0 "
          "(the default) lets it go with the graph. Qwen-Image-2.1, "
          "Krea-2, FLUX.2, Z-Image, FLUX.3 and MOSS-Image",
   .def_real = 0.0},
  // Sol-Attn. The same five keys generate-video carries, with the same
  // spellings and the same defaults, because they are one vocabulary
  // (generative-models/shared/accel-settings.h) and a graph moved from
  // one stage to the other should not have to be re-authored.
  //
  // KREA-2 TAKES IT, which is the day this comment used to anticipate:
  // the keys were carried for REGISTERED families, inert for the
  // built-ins, and the interface needed no change when one of them
  // implemented the tier -- `kcfg.sol = _sol` is the whole host-side
  // wiring. The others (flux2, qwen-image-edit, boogu-image) still
  // ignore it and the stage says so once, per family rather than for
  // every family as it used to.
  //
  // The bag remains how an out-of-tree image family learns what the
  // graph asked for, which is the coupling it exists to remove.
  {.key = "sol_attn", .type = ConfigType::Bool, .required = false,
   .doc = "accelerated attention (LOSSY): Sol-Attn routes whole key "
          "blocks during the online softmax, attending the ones a proxy "
          "score keeps and standing in for the rest with each block's key "
          "centroid and value sum. Cost falls with the fraction of blocks "
          "kept, which is a property of the DATA and is reported per run. "
          "Taken by KREA-2 among the built-in image families, over its "
          "joint [text; image] self-attention: the text prefix is "
          "attended EXACTLY, because a centroid over prompt tokens is a "
          "prompt half-read, and the tier declines below 16 routing "
          "blocks -- under about 512x512 -- where the local band, the "
          "sink and the always-exact tail already cover the sequence. "
          "Also taken by FLUX.2, over the same joint self-attention -- "
          "including reference-image editing, where the references sit "
          "IN the sequence. The one FLUX.2 shape it declines is "
          "`klein_kv`, whose cached steps attend the live queries over "
          "sequence-plus-spliced-reference keys and whose fill steps run "
          "two query groups: the keys are then not the queries, and a "
          "centroid would summarise the wrong sequence. Those steps stay "
          "dense automatically -- the gate is the geometry, not the "
          "flag. qwen-image-edit and boogu-image ignore it, and the "
          "stage says so once when it is set. REGISTERED families read it "
          "out of the acceleration bag (see generative-models/"
          "image-model-registry.h). MEASURED IN THE MODEL on an M4 Pro, "
          "on the video family that takes it (minimax-h3 at 20036 rows): "
          "a routed block's attention goes 3480 -> 1348 ms (2.58x) "
          "keeping 26.1% of key blocks. KREA-2'S OWN SPEED IS NOT "
          "MEASURED; its correctness is (routing off and every layer "
          "dense are byte-identical). An image DiT's sequence is shorter "
          "than a video one's, so expect less. "
          "INDEPENDENT OF i8_gemm and of sage_attn and settable with "
          "either: i8_gemm chooses how the block's GEMMs are computed, "
          "sage_attn how the kept blocks' QK product is, and this one "
          "WHICH key blocks are computed at all. Off by default -- it is "
          "an approximation, and which images it is safe on is a "
          "judgement about the model, not about the kernel",
   .def_bool = false},
  {.key = "sol_tau", .type = ConfigType::Real, .required = false,
   .doc = "Sol-Attn routing threshold, in STANDARD DEVIATIONS of a query "
          "block's own proxy-score distribution -- which is what lets one "
          "number serve every head, layer and resolution where a fixed "
          "block count would not. The threshold is mean + tau * sd, so "
          "HIGHER keeps fewer blocks: quality falls and speed rises, "
          "monotonically in both.\n"
          "0 IS THE DEFAULT HERE, against 1.0 on generate-video, because "
          "the higher value this stage used to ship was reported to show "
          "artifacts on real image graphs. It is the most accurate point "
          "that still routes.\n"
          "TAU 0 IS NOT DENSE: the threshold sits at the mean, so every "
          "below-average block is still dropped. It reduces artifacts "
          "rather than removing them, and `sol_attn false` is the only "
          "unapproximated setting.\n"
          "WHAT ANY TAU CAN SAVE IS A PROPERTY OF THE RESOLUTION, not of "
          "tau. Attention is O(seq^2) where the GEMMs are linear in it, "
          "so Sol's share of a step -- and so the whole tier's value -- "
          "grows with the picture: at ordinary image sizes there is "
          "little to divide however tau is set, and it earns its keep at "
          "the large ones. Raise tau toward 0.5-0.8 on a graph that "
          "tolerates it, where the speed actually is.\n"
          "generate-video keeps 1.0: its sequences are several times "
          "longer, which is where its published profile was measured and "
          "where this stage's report has not been re-checked",
   .def_real = 0.0},
  {.key = "sol_dense_layers", .type = ConfigType::Int, .required = false,
   .doc = "leading transformer blocks left DENSE, untouched by Sol-Attn. "
          "The first blocks are where the residual stream is least "
          "redundant. 0 routes every block",
   .def_int = 1},
  {.key = "sol_key_block", .type = ConfigType::Int, .required = false,
   .doc = "keys summarised by one centroid, and the unit the routing "
          "decides on. 32 or 64 only; 0 (default) takes 64. THE TWO TRADE "
          "AGAINST EACH OTHER RATHER THAN ONE DOMINATING, which is why "
          "both are offered. MEASURED on an M4 Pro at 56 heads x 20036 "
          "rows against dense steel: 64 keeps 19.1% of key blocks and "
          "runs 3.54x, 32 keeps 22.9-26.0% and runs 2.79x -- so 32 does "
          "MORE exact work, is slower, and is also the more accurate of "
          "the two, because a centroid over 32 keys stands in for them "
          "better. 64 is the speed choice, 32 the fidelity one. Larger "
          "blocks were measured and are worse on both counts, so they "
          "are not offered",
   .def_int = 0},
  {.key = "sol_local_radius", .type = ConfigType::Int, .required = false,
   .doc = "key blocks either side of a query's own block that are always "
          "exact, never routed. A block centroid is least representative "
          "exactly where a query discriminates most, so the near diagonal "
          "is not put to the vote. -1 routes even the query's own block, "
          "which is an ablation rather than a setting",
   .def_int = 1},
  // The adapter lives HERE, on the stage that loads the DiT, and not
  // only on the family's config source.
  //
  // It is not a family-specific knob the way klein_kv is -- every DiT
  // here can carry one -- so it belongs beside `i8_gemm`, which is the
  // other cross-family argument to the same load. Putting it ONLY on
  // the config source made it possible to configure an adapter, see the
  // source report it, and have nothing happen: that beat fans out to
  // whichever stages a graph wires it to, and a graph that wired it to
  // the diffusion-conditioner alone -- for the `vl_*` keys it also
  // carries -- dropped the adapter without a word.
  //
  // A model_config beat that NAMES these still wins, so one source can
  // still drive a whole graph.
  {.key = "lora", .type = ConfigType::String, .required = false,
   .doc = "a LoRA applied AT RUNTIME -- every adapted projection computes "
          "W x + scale * B (A x) rather than having the delta folded into "
          "the weights, which for a quantized base is the difference "
          "between keeping a small correction and rounding it away. A "
          "registered model, a directory holding one .safetensors, or a "
          "path to one. Krea-2, FLUX.2, Z-Image and Qwen-Image-2.1. "
          "LOAD-time: it decides how "
          "the blocks are BUILT, so a beat that changes it once the DiT "
          "is up is reported and ignored. A `lora` on the family's "
          "model-config beat OVERRIDES this",
   .suggest_db = kModelRegistryDb,
   .suggest_db_type = kLoraTypes,
   // AND a file browser: a LoRA is as often a download sitting in
   // the sandbox (Civitai) as a catalogued model, and the editor
   // offers both pickers when a field carries both hints.
   .is_path = true, .path_filter = "weights"},
  {.key = "lora_scale", .type = ConfigType::Real, .required = false,
   .doc = "the adapter's strength, applied PER FORWARD. Live: it rides the "
          "GEMM as a constant, so it can be swept without a reload. 1.0 = "
          "as trained; 0 skips both adapter GEMMs, so off is exactly off",
   .def_real = 1.0},
  {.key = "lora2", .type = ConfigType::String, .required = false,
   .doc = "a SECOND runtime LoRA, applied alongside the first: every "
          "adapted projection computes W x + s1 B1 (A1 x) + s2 B2 (A2 x). "
          "The pairing this exists for is a distillation or identity "
          "adapter in `lora` and a style one here, which is the one "
          "actually being dialled -- but nothing distinguishes the slots: "
          "either may hold either, and `lora2` alone is an ordinary "
          "request. Same accepted forms as `lora`, and LOAD-time in the "
          "same way. A `lora2` on the family's model-config beat "
          "OVERRIDES this",
   .suggest_db = kModelRegistryDb,
   .suggest_db_type = kLoraTypes,
   // AND a file browser: a LoRA is as often a download sitting in
   // the sandbox (Civitai) as a catalogued model, and the editor
   // offers both pickers when a field carries both hints.
   .is_path = true, .path_filter = "weights"},
  {.key = "lora2_scale", .type = ConfigType::Real, .required = false,
   .doc = "`lora2`'s strength, per FORWARD and independent of "
          "`lora_scale` -- which is the point of a second slot: one "
          "adapter stays where it was trained while the other is swept. "
          "0 skips its two GEMMs", .def_real = 1.0},
  // Slots 3..8, which read exactly as `lora2` does.
  lora_slots::path_key(2, kLoraTypes, kLoraMoreDoc),
  lora_slots::scale_key(2),
  lora_slots::path_key(3, kLoraTypes, kLoraMoreDoc),
  lora_slots::scale_key(3),
  lora_slots::path_key(4, kLoraTypes, kLoraMoreDoc),
  lora_slots::scale_key(4),
  lora_slots::path_key(5, kLoraTypes, kLoraMoreDoc),
  lora_slots::scale_key(5),
  lora_slots::path_key(6, kLoraTypes, kLoraMoreDoc),
  lora_slots::scale_key(6),
  lora_slots::path_key(7, kLoraTypes, kLoraMoreDoc),
  lora_slots::scale_key(7),
  {.key = "reference_mode", .type = ConfigType::String, .required = false,
   .doc = "how the reference latents pair with the conditioning beats. "
          "\"auto\" (the default) is the rule every input of this stage "
          "follows: a reference whose source sends ONE beat and ends is "
          "used for every generation (one source picture, many prompts), "
          "and one that keeps sending is consumed a beat per generation "
          "(a folder to restore or edit). \"latch\" holds the FIRST "
          "reference for the whole run even from a source that stays "
          "open -- a live input that will never end its stream. "
          "\"per_beat\" wants a new reference for EVERY generation and "
          "ends the run when they stop, even after one",
   .def_str = "auto", .extra = kReferenceModeChoices},
};

// The keys that MOVED to the per-family config stages. Named so a
// pipeline written against the old union says what to do instead of
// silently running the defaults -- an unknown key is not an error
// anywhere in this runtime, which is what makes a quiet removal the
// wrong way to do this. `klein_kv` is the one that matters most: it
// selects a RECIPE, so ignoring it does not degrade the image, it makes
// a different one.
struct MovedKey { const char* key; const char* to; };
constexpr MovedKey kMovedKeys[] = {
  {"klein_kv",      "flux2-model-config"},
};
const PortSpec kIports[] = {
  {.name = "conditioning", .doc = "conditioning tensor from a diffusion-"
                                  "conditioner stage (family-shaped + typed)",
   .type = &typeid(TensorBeatPayload),
   .tags = "conditioning", .clock_group = 0},
  {.name = "neg_conditioning", .doc = "OPTIONAL negative conditioning (the "
                                      "conditioner's oport1) for classifier-free "
                                      "guidance",
   .type = &typeid(TensorBeatPayload),
   .tags = "conditioning", .clock_group = 0},
  {.name = "model", .doc = "OPTIONAL shared model reference from a model-select "
                           "source; overrides the hf_dir config",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
  {.name = "sampler",
   .doc = "OPTIONAL sampler spec FlexData (diffusion-sampler-select)",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
  {.name = "scheduler", .doc = "OPTIONAL scheduler spec FlexData (scheduler-select)",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
  {.name = "ref_latent0", .doc = "OPTIONAL reference latent 0 (channel-first "
                                 "f32 from vae-encode); FLUX.2/QIE conditioning "
                                 "/ Krea-2 img2img init",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
  {.name = "ref_latent1", .doc = "OPTIONAL reference latent 1 (channel-first "
                                 "f32 from vae-encode); FLUX.2/QIE 2nd reference "
                                 "(distinct RoPE position); ignored by Krea-2",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
  {.name = "model_config",
   .doc = "OPTIONAL model-specific parameters from the resident family's own "
          "config source -- flux2-model-config (the klein-kv recipe and its "
          "`lora`), or krea2-model-config (its `lora`). An out-of-tree "
          "family's own config source lands here too. Passed to that "
          "family's own params "
          "struct UNREAD, so this stage carries none of the knobs. Wiring it "
          "DEFERS the DiT load to the first beat, because a recipe like "
          "klein_kv -- or an adapter, which decides how the blocks are built "
          "-- has to be known before the weights are read. THE ADAPTER KEYS "
          "ARE READ HERE AND NOWHERE ELSE: a krea2-model-config wired only to "
          "the diffusion-conditioner delivers its `vl_*` keys and its `lora` "
          "goes nowhere, silently",
   .type = &typeid(FlexDataPayload),
   .tags = "model-config", .clock_group = 0},
  {.name = "ref_latents",
   .doc = "OPTIONAL reference latents as ONE LIST (vae-encode's `latents`): "
          "as many as the graph has, each at its own size, for a "
          "multi-reference family. They follow whatever is on ref_latent0 / "
          "ref_latent1, in list order, and the family sees them all as its "
          "references; one that takes fewer uses the first ones and says so",
   .type = &typeid(TensorListPayload),
   .tags = "latent-list", .clock_group = 0},
};

// The model iport (a model-select source) overrides hf_dir. Inserted after the
// conditioning + neg_conditioning inputs, so it is iport2; sampler / scheduler /
// ref_latent0 / ref_latent1 follow at iports 3 / 4 / 5 / 6. (Referenced only
// from the Apple-gated code; maybe_unused for the inert non-Apple build.)
[[maybe_unused]] constexpr unsigned kModelPort = 2;
[[maybe_unused]] constexpr unsigned kModelCfgPort = 7;
// ...and the reference LIST after them all, so every graph written before
// it keeps its port numbers.
[[maybe_unused]] constexpr unsigned kRefListPort = 8;
const PortSpec kOports[] = {
  {.name = "latent",
   .doc = "f32 latent [z_dim, H/8, W/8] (unpacked, whitened)",
   .type = &typeid(TensorBeatPayload),
   .tags = "latent", .clock_group = 0},
  {.name = "step_latent",
   .doc = "OPTIONAL per-denoise-step latent (one beat per sampler step, SAME "
          "format as `latent`) -- connect a vae-decode here to visualize the "
          "denoising progression for debugging. Only emitted when connected.",
   .type = &typeid(TensorBeatPayload),
   .tags = "latent", .clock_group = 0},
  // Its OWN clock: several previews per image, at the step rate.
  {.name = "preview",
   .doc = "OPTIONAL live preview while the image denoises: planar U8 RGB "
          "[3, H, W], one beat per rendered step -- the model's clean "
          "estimate, decoded by the tiny autoencoder the family's config "
          "source names (`preview_vae`). Feed a `preview` stage. Push-style "
          "(DropOldest, one consumer): a preview nobody reads is dropped, "
          "never waited for. Unwired, nothing is decoded. Far cheaper than "
          "`step_latent` through the full VAE, and it shows the image the "
          "model is heading for rather than the noisy state",
   .type = &typeid(TensorBeatPayload),
   .tags = "rgb-frames", .clock_group = 1},
};
[[maybe_unused]] constexpr unsigned kPreviewPort = 2;
// Deep enough that the preview stage's tick can pick one up, shallow
// enough that a beat nobody reads is not hoarded.
constexpr unsigned kPreviewDepth = 2;
const StageSpec kSpec = {
  .type_name = "generate-image",
  .doc       = "Diffusion DiT denoiser: conditioning (from a diffusion-"
               "conditioner stage) -> family MMDiT -> FlowMatchEuler -> latent, "
               "on the metal-compute backend. The denoiser half of the split "
               "(feed vae-decode). A run makes one image per generation: an "
               "input whose source sends one beat and ends serves every "
               "generation, one that keeps sending is consumed a beat per "
               "generation, and the run ends when a consumed input does. "
               "`seed_sequence` sets each generation's seed.",
  .display_name = "Generate Image",
  .category  = StageCategory::Generative,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};


// The block stems of every DiT this stage drives, and the FLOOR they
// imply.
//
// ONE list, used by declare_memory() and declare_resources() both. They
// answer different questions -- what the plan reports, and what the
// resource phase may refuse a graph on -- but they must answer them
// about the same bytes, and a stem list maintained twice is a silent
// per-family divergence: a family missing from one copy is reported as
// streamable and judged as though it were not.
//
// A stem matching nothing contributes nothing, so naming all four costs
// one lookup each and cannot mis-measure a checkpoint that uses only
// one of them.
// THE INT8 LAYOUTS ON A W4 PACK, in bytes, two ways. On the fast and
// token tiers (i8_gemm_tier) a w4 block projection is either HELD as its
// int8 layout -- twice its nibbles, so `held` is every one's codes again
// -- or kept as stored and converted before each GEMM into one int8
// scratch the model's I8GemmContext keeps, `scratch` the largest such
// projection's codes and scales. Which one is the engine's call
// (hold_w4_int8_). Zero for a family with no such tier and a pack with
// no w4 projection. Taken over every w4 tensor of a block, where the
// families convert only the ones at the image's rows: the error
// over-states, which is the direction that refuses rather than thrashes.
struct Int8LayoutBytes {
  std::size_t scratch = 0;
  std::size_t held = 0;
};

Int8LayoutBytes
int8_layout_bytes_(const std::string& family, const std::string& dit)
{
  Int8LayoutBytes out;
  std::vector<std::string_view> stems;
  if (family == "z-image" || family == "fibo2") {
    stems = {"noise_refiner.", "layers."};
  } else if (family == "moss-image") {
    stems = {"layers."};
  } else if (family == "krea2" || family == "qwen-image-21") {
    stems = {"transformer_blocks."};
  } else if (family == "flux2") {
    stems = {"transformer_blocks.", "single_transformer_blocks."};
  } else {
    return out;
  }
  auto wts = genai::MetalLlamaWeights::open_model(dit);
  if (!wts.has_value()) { return out; }
  const std::string_view suffix = ".weight";
  for (const std::string& nm : wts->tensor_names()) {
    if (nm.size() <= suffix.size() ||
        nm.compare(nm.size() - suffix.size(), suffix.size(), suffix) != 0) {
      continue;
    }
    bool in_stack = false;
    for (std::string_view st : stems) {
      if (nm.rfind(std::string(st), 0) == 0) { in_stack = true; }
    }
    if (!in_stack) { continue; }
    // Fibo 2.0's injection blocks keep their cross-attention in the
    // trunk, off the layout: it runs once a prompt, at the text's rows.
    if (family == "fibo2" &&
        nm.find(".cross_attention.") != std::string::npos) {
      continue;
    }
    // FLUX.2's double blocks keep their TEXT stream (add_*, to_add_out,
    // ff_context) off the layout -- it runs at the prompt's rows, under
    // the int8 floor -- so it neither grows nor needs scratch; counted,
    // klein-9B's hold decision read ~0.87 GB it would never spend.
    if (family == "flux2" &&
        (nm.find(".attn.add_") != std::string::npos ||
         nm.find(".attn.to_add_out.") != std::string::npos ||
         nm.find(".ff_context.") != std::string::npos)) {
      continue;
    }
    const std::string stem = nm.substr(0, nm.size() - suffix.size());
    const auto* ci = wts->info(nm);
    const auto* sc = wts->info(stem + ".scales");
    if (ci == nullptr || sc == nullptr || ci->shape.size() != 2 ||
        sc->shape.size() != 2 || sc->shape[1] <= 0) {
      continue;
    }
    // Group 64, the only one the tiers convert; w4 only.
    const int N = (int)ci->shape[0];
    const int K = (int)(sc->shape[1] * 64);
    if ((long)ci->shape[1] * 32 != (long)K * 4) { continue; }
    out.held += (std::size_t)ci->nbytes;
    // Scales at the coarse tier's finest group (256), or the token
    // tier's f32 a row, whichever is larger.
    out.scratch = std::max(
        out.scratch,
        std::max(genai::I8GemmContext::fast_scratch_bytes(N, K, 256),
                 genai::I8GemmContext::fast_scratch_bytes(
                     N, K, genai::I8GemmContext::kTokenGroup)));
  }
  return out;
}

std::string t2i_family_(const std::string& transformer_dir);   // below

// A DiT checkpoint's quantization width from its config.json (0 dense,
// unreadable, or a single file), for the plan's figures.
int
dit_config_quant_bits_(const std::string& dit)
{
  std::ifstream in(std::filesystem::path(dit) / "config.json");
  if (!in) { return 0; }
  FlexData fd = FlexData::from_json(in);
  if (!fd.is_object() || !fd.as_object().contains("quantization")) {
    return 0;
  }
  FlexData q = fd.as_object().at("quantization");   // keep the owner
  if (!q.is_object() || !q.as_object().contains("bits")) { return 0; }
  return (int)q.as_object().at("bits").as_int(0);
}

std::size_t
dit_floor_bytes_(const std::string& root, const std::string& dit)
{
  // VOSR HAS NO STREAMING FORM, and its checkpoint looks like one that
  // does: its 36 blocks are named `blocks.N.`, which the single-stack
  // stem below matches. MetalVosrTransformer copies every block in at
  // load, so a floor measured from those names is a promise nothing
  // keeps -- for VOSR 2.0, 417 MB against the ~2659 MB it holds at bf16,
  // the error that ADMITS a graph rather than refusing one. The published
  // layout escaped it only by accident: `checkpoints/ema_model.safetensors`
  // is not a name open_model() opens in a directory, so the measurement
  // came back 0; a checkpoint shipped as `model.safetensors` (which the
  // loader also accepts) would have been booked at the block-name floor.
  // Zero is "cannot be reduced", and now it is said on purpose.
  genai::MetalVosrTransformer::Config vc;
  if (genai::MetalVosrTransformer::read_config(root, &vc, nullptr)) {
    return 0;
  }
  // Z-IMAGE ANSWERS FOR ITSELF. Its streamed stack is ONE stack spelled
  // under two prefixes (`noise_refiner.` then `layers.`) sharing one
  // slot pair, while `context_refiner.` is held in both modes; and its
  // Turbo checkpoint is F32, so the bytes held are half the bytes on
  // disk. The stem list below can express neither, and without a stem
  // that matches at all it returns 0 -- "cannot be reduced" -- which
  // declared a 24.6 GB DiT at its full size and put the plan's peak
  // 8 GB over this box for a model that actually floors near 1.5.
  {
    genai::MetalZImageTransformer::Config zc;
    if (genai::MetalZImageTransformer::Config::read_dims(dit, &zc)) {
      return genai::MetalZImageTransformer::streaming_floor_bytes(dit);
    }
  }
  // MOSS-IMAGE ANSWERS FOR ITSELF: its two text blocks share the
  // `layers.`-free trunk with the model-level weights, and one slot pair
  // serves the 42 main blocks.
  {
    genai::MetalMossImageTransformer::Config mc;
    if (genai::MetalMossImageTransformer::Config::read_dims(dit, &mc)) {
      return genai::MetalMossImageTransformer::streaming_floor_bytes(dit);
    }
  }
  // FIBO 2.0 ANSWERS FOR ITSELF: one stack under two prefixes
  // (noise_refiner then layers), its injection blocks' cross-attention
  // and the Perceiver held in the trunk.
  {
    genai::MetalFibo2Transformer::Config bc;
    if (genai::MetalFibo2Transformer::Config::read_dims(dit, &bc)) {
      return genai::MetalFibo2Transformer::streaming_floor_bytes(dit);
    }
  }
  // FLUX.3 ANSWERS FOR ITSELF too: its four stacks (three per-mode, one
  // joint) are ONE block shape sharing ONE slot pair, where the stem list
  // below would book a pair per stem.
  {
    genai::MetalFlux3Transformer::Config fc;
    if (genai::MetalFlux3Transformer::Config::read_dims(dit, &fc)) {
      return genai::MetalFlux3Transformer::streaming_floor_bytes(dit);
    }
  }
  // EVERY stack, not just the first. These models stream BOTH of their
  // stacks and keep a slot pair for each, so a stem list that names one
  // leaves the other in the trunk -- MEASURED on FLUX.2-klein-9B, whose
  // 24 single blocks are 10.5 GB of a 17.3 GB checkpoint: naming only
  // `transformer_blocks.` reported a floor of 12324 MB against a true
  // 2848, which is a declaration that says almost nothing.
  static const std::vector<std::string_view> kStems = {
      "transformer_blocks.",         // FLUX.2 doubles, Krea-2,
                                     // Qwen-Image, Qwen-Image-2.1
      "single_transformer_blocks.",  // FLUX.2 singles
      "double_stream_layers.",       // Boogu doubles
      "single_stream_layers.",       // Boogu singles
      "double_blocks.",              // Comfy-Org repacks
      "single_blocks.",
      "blocks.",                     // single-stack spellings
  };
  std::size_t floor = model_memory::streaming_floor_bytes(dit, kStems);
  // A SLOT THAT KEEPS ITS SOURCE holds more than the block on disk: the
  // products are built out of the raw tensor after every refill, and the
  // raw tensor stays (see the BlockSlots notes on tensors with no
  // checkpoint name). Krea-2's fused SwiGLU weaves `ff_gu` beside the
  // split gate / up it reads; FLUX.2's single block splits
  // `to_qkv_mlp_proj` into q|k|v and the MLP's gate|up beside the raw
  // projection. Priced at the disk block, each pair sat short by twice
  // that tensor -- ~600 MB on FLUX.2-klein-9B. Booked whether or not the
  // fused path engages, which is the side that refuses.
  if (floor > 0) {
    const std::string fam = t2i_family_(dit);
    std::vector<std::string> keep;
    if (fam == "krea2") {
      keep = {"transformer_blocks.0.ff.gate.", "transformer_blocks.0.ff.up."};
    } else if (fam == "flux2") {
      keep = {"single_transformer_blocks.0.attn.to_qkv_mlp_proj."};
    }
    if (!keep.empty()) {
      if (auto w = genai::MetalLlamaWeights::open_model(dit)) {
        std::size_t kept = 0;
        for (const std::string& nm : w->tensor_names()) {
          for (const std::string& k : keep) {
            if (nm.rfind(k, 0) != 0) { continue; }
            if (const auto* ti = w->info(nm)) {
              kept += (std::size_t)ti->nbytes;
            }
          }
        }
        floor += 2 * kept;
      }
    }
  }
  return floor;
}

}  // namespace

GenerateImageStage::GenerateImageStage(const SessionContextIntf* s,
                                   std::string               id,
                                   std::vector<InEdge>       iports,
                                   FlexData                  config)
  : TypedStage<GenerateImageStage>(s, std::move(id), std::move(iports),
                                 std::move(config))
{
  _hf_dir    = attr_str("hf_dir");
  _dit_dir   = attr_str("dit_dir");
  // THE RESTORER'S TILING MOVED to `vosr-model-config`, because it is
  // one family's knob on a stage that serves seven. A graph that still
  // carries the keys here is told once rather than left to wonder why
  // the number stopped mattering -- an unknown key is not otherwise
  // reported anywhere, which is how a knob set in good faith goes quiet.
  {
    const FlexData& cfg = this->config();
    if (cfg.is_object()) {
      auto o = cfg.as_object();
      for (const char* k : {"tile_size", "tile_overlap"}) {
        if (!o.contains(k)) { continue; }
        session()->warn(fmt(
            "GenerateImageStage('{}'): `{}` is no longer read here -- it "
            "moved to the `vosr-model-config` stage, wired to this stage's "
            "model_config iport. Unwired, the restorer tiles at the "
            "resolution its checkpoint was distilled at, which is what "
            "this key was usually set to", this->id(), k));
      }
    }
  }
  _tile_overlap = (int)attr_int("tile_overlap");
  {
    using Mode = GenerationInput::Mode;
    const std::string rm = attr_str("reference_mode");
    if (rm == "latch")         { _ref_mode = Mode::kLatch; }
    else if (rm == "per_beat") { _ref_mode = Mode::kFresh; }
    else if (!rm.empty() && rm != "auto") {
      // Deferred-validated config: warn and take the default.
      session()->warn(fmt(
          "GenerateImageStage('{}'): reference_mode '{}' is not "
          "auto|latch|per_beat; using auto", this->id(), rm));
    }
    for (GenerationInput& in : _ref_port_in) { in.set_mode(_ref_mode); }
    _ref_list_in.set_mode(_ref_mode);
  }
  _init_latents = attr_str("init_latents");
  _height = (int)attr_int("height");
  _width  = (int)attr_int("width");
  _steps  = (int)attr_int("steps");
  // Whether the GRAPH chose a step count, which the default
  // below is about to make unknowable. A one-step distilled
  // family needs to tell "8 because nobody said" apart from
  // "8 because somebody did".
  _steps_set = _steps > 0;
  _strength = attr_real("strength");
  _guidance_scale = attr_real("guidance_scale");
  // Accelerated mode (LOSSY, opt-in): dynamic-int8 GEMMs for the DiT's
  // big block matmuls (~2x their f16 rate on matrix-core GPUs at int8
  // quality). Env VPIPE_I8_GEMM=0|1 overrides.
  //
  // This one STAYS here, unlike klein_kv beside it, and the line
  // between them is worth stating: i8_gemm chooses a KERNEL, so it is
  // a property of the machine and the caller's quality budget, and a
  // family without int8 kernels simply runs f16 -- no worse, just not
  // faster. klein_kv chooses a RECIPE, so on the wrong checkpoint it
  // does not run slower, it runs wrong.
  _i8_gemm = attr_bool("i8_gemm");
  _keep_loaded = attr_real("keep_loaded");
  _sage.enabled      = attr_bool("sage_attn");
  _sage.dense_layers = (int)attr_int("sage_dense_layers");
  _sage.pv8          = attr_bool("sage_pv8");
  _sol.enabled      = attr_bool("sol_attn");
  _sol.tau          = (float)attr_real("sol_tau");
  _sol.dense_layers = (int)attr_int("sol_dense_layers");
  _sol.local_radius = (int)attr_int("sol_local_radius");
  // 32 or 64, and 0 means the default. WARN AND FALL BACK rather than
  // refuse the stage, which is what `unload_when_idle` beside it does
  // for the same class of key: a perf knob spelled wrong should not take
  // a graph down. Named in the warning, because silently running a
  // different block size than was asked for is how a benchmark comes to
  // report the wrong number.
  {
    const std::int64_t kb = attr_int("sol_key_block");
    if (kb == 0 || kb == 32 || kb == 64) {
      _sol.key_block = (int)kb;
    } else {
      _sol.key_block = 0;
      // The message distinguishes the two rejects, because they are not
      // the same fact: 128 and 256 RUN and are simply worse on both
      // speed and accuracy, so declining them is a policy; anything not
      // a multiple of the steel kernel's 16-key block cannot run at all.
      session()->warn(fmt(
          "GenerateImageStage('{}'): sol_key_block {} is not 32 or 64; "
          "using 64. {}", this->id(), (long long)kb,
          kb > 64
              ? "Larger blocks were measured and are slower AND less "
                "accurate, so they are not offered"
              : "The routing block is also the unit the exact half walks, "
                "so it cannot be smaller than 32"));
    }
  }
  // THE BAG, and the typed members are read BACK OUT OF IT, exactly as
  // generate-video does it. The in-tree DiTs take a typed Config -- they
  // are compiled with this stage and have no ABI to protect -- and a
  // registered family gets the bag itself; those are two readings of one
  // config, and the failure if they drift is a family running at
  // settings the log line did not describe.
  _accel = FlexData::make_object();
  genai::accel::set_flag(&_accel, genai::accel::kI8Gemm, _i8_gemm);
  {
    const std::string t = attr_str("i8_gemm_tier");
    const bool known = t == genai::accel::kI8TierFast ||
                       t == genai::accel::kI8TierToken ||
                       t == genai::accel::kI8TierTokenPlain;
    if (!t.empty() && !known && t != genai::accel::kI8TierAccurate) {
      fail_config(fmt("GenerateImageStage('{}'): i8_gemm_tier must be "
                      "'accurate', 'fast', 'token' or 'token_plain' (got "
                      "'{}')", this->id(), t));
    }
    genai::accel::set_text(&_accel, genai::accel::kI8GemmTier,
                           known ? t
                                 : std::string(genai::accel::kI8TierAccurate));
  }
  genai::accel::set_flag(&_accel, genai::accel::kSageAttn, _sage.enabled);
  genai::accel::set_integer(&_accel, genai::accel::kSageDenseLayers,
                            _sage.dense_layers);
  genai::accel::set_flag(&_accel, genai::accel::kSagePv8, _sage.pv8);
  // Written whether the tier is on or off, and with the SETTLED value:
  // `sol_key_block` here is what the check above fell back to, not what
  // the graph said, so no family has to repeat that validation or can
  // reach a different answer than the log line did.
  genai::accel::set_flag(&_accel, genai::accel::kSolAttn, _sol.enabled);
  genai::accel::set_real(&_accel, genai::accel::kSolTau, (double)_sol.tau);
  genai::accel::set_integer(&_accel, genai::accel::kSolKeyBlock,
                            _sol.key_block);
  genai::accel::set_integer(&_accel, genai::accel::kSolDenseLayers,
                            _sol.dense_layers);
  genai::accel::set_integer(&_accel, genai::accel::kSolLocalRadius,
                            _sol.local_radius);
  // The ANE tier. An empty `ane_templates` does NOT disable it: a family
  // that can emit its own graph needs no template, and only the family
  // knows whether it can, so the decision belongs there and not here.
  // `ane_layers` caps the blocks that use it;
  {
    const std::string tdir = attr_str("ane_templates");
    const bool on = attr_bool("ane_ffn");
    double rows = attr_real("ane_rows");
    if (!(rows >= 0.0) || rows > 1.0) { rows = 0.0; }
    long long layers = attr_int("ane_layers");
    if (layers < 0) { layers = 0; }
    genai::accel::set_flag(&_accel, genai::accel::kAneFfn, on);
    // Only with the feed-forward: the two modules are one booking.
    genai::accel::set_flag(&_accel, genai::accel::kAneQkv,
                           on && attr_bool("ane_qkv"));
    genai::accel::set_real(&_accel, genai::accel::kAneRows, rows);
    genai::accel::set_text(&_accel, genai::accel::kAneTemplates, tdir);
    genai::accel::set_integer(&_accel, genai::accel::kAneLayers, layers);
  }
  // ...and back out, which is the half that makes it one decision. If a
  // key is ever written under one name and read under another, this is
  // where it stops being true rather than three plugins away.
  _sage    = genai::sage::config_from_flex(&_accel);
  _sol     = genai::sol::config_from_flex(&_accel);
  _i8_gemm = genai::accel::flag(&_accel, genai::accel::kI8Gemm);
  {
    const std::string t =
        genai::accel::text(&_accel, genai::accel::kI8GemmTier);
    _i8_tier = t == genai::accel::kI8TierTokenPlain ? 3
               : t == genai::accel::kI8TierToken    ? 2
               : t == genai::accel::kI8TierFast     ? 1
                                                    : 0;
  }
  for (int i = 0; i < kLoraSlots; ++i) {
    _lora[(std::size_t)i].path  = attr_str(lora_slots::kPath[i]);
    _lora[(std::size_t)i].scale = attr_real(lora_slots::kScale[i]);
  }
  _lora_model_slot.fill(-1);
  _seed_base = (std::uint64_t)attr_int("seed");
  _seed      = _seed_base;
  {
    const std::string sq = attr_str("seed_sequence");
    if (!sq.empty() && !parse_seed_sequence(sq, &_seed_seq)) {
      // Deferred-validated config: warn and take the default.
      session()->warn(fmt(
          "GenerateImageStage('{}'): seed_sequence '{}' is not "
          "increment|randomize|keep; using increment", this->id(), sq));
    }
  }
  // The family-specific keys are gone from this stage; a pipeline still
  // carrying one gets told where it went. Warning rather than failing
  // because the key is now inert, not wrong -- the graph runs, it just
  // runs on defaults.
  {
    auto o = this->config().as_object();
    for (const auto& mk : kMovedKeys) {
      if (o.contains(mk.key)) {
        session()->warn(fmt(
            "GenerateImageStage('{}'): config key '{}' moved to the '{}' "
            "stage and is IGNORED here; wire one to the model_config iport "
            "or the family runs its defaults", this->id(), mk.key, mk.to));
      }
    }
  }
  // Neither axis configured => infer the size from ref_latent0 when a
  // reference is wired (process()); both stay 0 so the check below is vacuous
  // and the log reads "auto". A PARTIALLY specified size is not half-inferred
  // -- the given axis stands and the other keeps the historical 256 default.
  _infer_size = (_height <= 0 && _width <= 0);
  if (!_infer_size) {
    if (_height <= 0) { _height = 256; }
    if (_width  <= 0) { _width  = 256; }
  }
  if (_steps  <= 0) { _steps  = 8; }
  if (_guidance_scale <= 0.0) { _guidance_scale = 1.0; }   // <=0 => CFG off
  if (_strength < 0.0 || _strength > 1.0) {
    fail_config(fmt(
        "GenerateImageStage('{}'): strength must be in [0,1] (got {})",
        this->id(), _strength));
  }
  // hf_dir is OPTIONAL: a model-select source on the model iport can supply it
  // instead, so "no model at all" is reported at initialize()/process() time
  // (when iport connectivity is known), not here.
  if (_height % 16 != 0 || _width % 16 != 0) {
    fail_config(fmt(
        "GenerateImageStage('{}'): height/width must be multiples of 16 (got "
        "{}x{})", this->id(), _height, _width));
  }
  allocate_oports(spec().oports.size());
  set_oport_policy(kPreviewPort, {kPreviewDepth, OverrunPolicy::DropOldest});
#ifdef VPIPE_BUILD_APPLE_SILICON
  _scheduler_spec.steps = _steps;   // config default; port beats override
#endif
}

// Carry the settled ANE tier out of the accel bag and onto a family's
// config. WITHOUT THIS the keys are documented, validated and put in
// the bag, and then nothing reads them -- the tier is reachable from a
// test and from no graph at all, which is exactly how it shipped
// unreachable the first time.
//
// Templated because every family that grows the tier takes the same
// three fields; the bag is the shared vocabulary, this is the one place
// that spends it.
std::string
GenerateImageStage::ane_claim_label_() const
{
  return "ane-ffn/" + std::string(this->id());
}

template <typename Cfg>
void
GenerateImageStage::apply_ane_(Cfg& cfg) const
{
  namespace a = genai::accel;
  if (!a::flag(&_accel, a::kAneFfn)) { return; }
  cfg.session       = session();
  cfg.ane_templates = a::text(&_accel, a::kAneTemplates);
  cfg.ane_rows      = (float)a::real(&_accel, a::kAneRows, 0.0);
  // Before the bytes below, which must cover BOTH modules or the grant
  // is for less than the family will build.
  cfg.ane_qkv       = a::flag(&_accel, a::kAneQkv);
  // WHETHER the tier runs is the plan's call; HOW MANY blocks use it is
  // not a memory question.
  //
  // The feed-forward runs on ONE shared runtime-weight module whose
  // weights are staged per block into IOSurface slots, so its cost is
  // fixed -- the module's staged weights plus the slots, ~1.4 GB for
  // Krea-2 -- however many blocks use it. The plan grants that one unit
  // or nothing, and without it the config carries no session, which is
  // how the family is told to keep its GPU path. Read HERE rather than at
  // claim time because the grant is not computed until every stage has
  // claimed.
  const std::size_t bytes = genai::MetalKrea2Transformer::ane_runtime_bytes(
      cfg, genai::MetalKrea2Transformer::ane_plan_seq(_width, _height));
  if (model_memory::coreml_grant(session(), ane_claim_label_(), bytes, 1) <=
      0) {
    cfg.session = nullptr;
    session()->info(fmt(
        "GenerateImageStage('{}'): the ANE feed-forward was requested but "
        "the plan left no room for its module ({} MB); keeping the GPU "
        "feed-forward", this->id(), bytes >> 20));
    return;
  }
  // `ane_layers` is only a CAP now -- a run covering fewer blocks, say a
  // benchmark. 0 means every dense block.
  cfg.ane_layers =
      (int)std::max<long long>(0, a::integer(&_accel, a::kAneLayers, 0));
}

GenerateImageStage::~GenerateImageStage()
{
  keep_models_warm_();
}

namespace {

// The warm holds' kinds (GenerativeModelManager::keep_model_warm): the
// class each casts back to.
constexpr const char* kWarmKrea2  = "krea2-dit";
constexpr const char* kWarmFlux2  = "flux2-dit";
constexpr const char* kWarmQi21   = "qwen-image-21-dit";
constexpr const char* kWarmZImage = "z-image-dit";
constexpr const char* kWarmFlux3  = "flux3-dit";
constexpr const char* kWarmMoss   = "moss-image-dit";
constexpr const char* kWarmFibo2  = "fibo2-dit";

}  // namespace

// What this stage built, handed to the manager as it ends to be kept
// warm (`keep_loaded`, GenerativeModelManager::keep_model_warm) -- only
// where the box has room for it: a bounded box streams because it does
// not, and frees the DiT for the decode.
void
GenerateImageStage::keep_models_warm_()
{
  if (_keep_loaded <= 0 || _dit_bounded || _dit_warm_key.empty() ||
      session() == nullptr) {
    return;
  }
  keep_warm_dit_(kWarmKrea2, _krea2_dit_dir, _dit, "Krea-2");
  keep_warm_dit_(kWarmFlux2, _flux2_dit_dir, _flux2_dit, "FLUX.2");
  keep_warm_dit_(kWarmQi21, _qi21_dit_dir, _qi21_dit, "Qwen-Image-2.1");
  keep_warm_dit_(kWarmZImage, _zi_dit_dir, _zi_dit, "Z-Image");
  keep_warm_dit_(kWarmFlux3, _flux3_dit_dir, _flux3_dit, "FLUX.3");
  keep_warm_dit_(kWarmMoss, _moss_dit_dir, _moss_dit, "MOSS-Image");
  keep_warm_dit_(kWarmFibo2, _fibo2_dit_dir, _fibo2_dit, "Fibo 2.0");
}

template <class Cfg, class Spec>
std::string
GenerateImageStage::warm_key_(const Cfg& c, bool stream,
                              const std::vector<Spec>& loras,
                              const std::string& extra)
{
  std::string k = stream ? "stream" : "preload";
  if constexpr (requires { c.i8_gemm; }) {
    k += fmt("|i8 {}", c.i8_gemm ? 1 : 0)();
  }
  if constexpr (requires { c.i8_tier; }) { k += fmt(" {}", c.i8_tier)(); }
  if constexpr (requires { c.i8_hold_w4; }) {
    k += fmt(" {}", c.i8_hold_w4 ? 1 : 0)();
  }
  // Every sage::Config field, the int8 P*V's included: a family reads
  // them off its own config at every forward (which flash function, which
  // prologue), so a warm DiT under another sage_pv8 runs the other arm --
  // and an A/B of it in one process measures the first arm twice. The
  // P*V's only where Sage is on, and its two refinements only with it.
  if constexpr (requires { c.sage; }) {
    const bool pv8 = c.sage.enabled && c.sage.pv8;
    k += fmt("|sage {} {} {} {}", c.sage.enabled ? 1 : 0,
             c.sage.dense_layers, c.sage.smooth_k ? 1 : 0, pv8 ? 1 : 0)();
    if (pv8) {
      k += fmt(" {} {}", c.sage.pv8_from_step,
               c.sage.pv8_all_rows ? 1 : 0)();
    }
  }
  if constexpr (requires { c.sol; }) {
    k += fmt("|sol {} {} {} {} {} {} {}", c.sol.enabled ? 1 : 0, c.sol.tau,
             c.sol.dense_layers, c.sol.local_radius, c.sol.sink_start,
             c.sol.sink_tokens, c.sol.key_block)();
  }
  if constexpr (requires { c.ane_ffn; }) {
    k += fmt("|ane {}", c.ane_ffn ? 1 : 0)();
  }
  // Krea-2 and FLUX.2 arm the tier by a session, which the plan's grant
  // withholds: a build with the ANE's modules and one without differ by
  // that alone.
  if constexpr (requires { c.session; }) {
    k += fmt("|ane {}", c.session != nullptr ? 1 : 0)();
  }
  if constexpr (requires { c.ane_qkv; }) {
    k += fmt(" {}", c.ane_qkv ? 1 : 0)();
  }
  if constexpr (requires { c.ane_rows; }) { k += fmt(" {}", c.ane_rows)(); }
  if constexpr (requires { c.ane_layers; }) {
    k += fmt(" {}", c.ane_layers)();
  }
  if constexpr (requires { c.ane_templates; }) {
    k += "|" + std::string(c.ane_templates);
  }
  if constexpr (requires { c.klein_kv; }) {
    k += fmt("|kv {}", c.klein_kv ? 1 : 0)();
  }
  // The adapters by file: a strength is live (set_lora_scale), set again
  // on every take -- see take_warm_dit_.
  for (const auto& l : loras) { k += fmt("|lora {}", l.path)(); }
  return k + extra;
}

template <class Dit>
std::shared_ptr<Dit>
GenerateImageStage::take_warm_dit_(const char* kind, const std::string& dir,
                                   const std::string& key, const char* name)
{
  _dit_warm_key = key;
  if (_keep_loaded <= 0 || _dit_bounded || session() == nullptr) {
    return nullptr;
  }
  auto* mgr = session()->services()->generative_model_manager();
  if (mgr == nullptr) { return nullptr; }
  auto warm = mgr->take_warm<Dit>(kind, dir, key);
  if (warm) {
    _dit_taken_warm = true;
    // Each adapter's strength as THIS stage has it: a live change in the
    // graph that kept it (a model_config beat) stayed on the model, and
    // adapter_keys pushes a strength only when it changes.
    for (int i = 0; i < kLoraSlots; ++i) {
      const int ms = _lora_model_slot[(std::size_t)i];
      if (ms >= 0 && warm->lora_modules(ms) > 0) {
        warm->set_lora_scale(ms, (float)_lora[(std::size_t)i].scale);
      }
    }
    std::size_t bytes = 0;
    if constexpr (requires { warm->resident_bytes(); }) {
      bytes = (std::size_t)warm->resident_bytes();
    }
    session()->info(fmt(
        "GenerateImageStage('{}'): the {} DiT kept warm is taken back{}",
        this->id(), name,
        bytes > 0 ? fmt(" ({} MB resident)", bytes >> 20)() : std::string()));
  }
  return warm;
}

template <class Dit>
void
GenerateImageStage::keep_warm_dit_(const char* kind, const std::string& dir,
                                   std::shared_ptr<Dit>& dit,
                                   const char* name)
{
  if (!dit || dir.empty()) { return; }
  auto* mgr = session()->services()->generative_model_manager();
  if (mgr == nullptr) { return; }
  // What a forward sizes per picture goes; the weights stay.
  dit->release_forward_scratch();
  (void)dit->release_idle_slots();
  // What the hold keeps, for the next launch's fit test: the weights the
  // model holds (the checkpoint's size where it does not say), and the
  // ANE's modules -- CoreML's bytes, which no ledger here sees.
  std::size_t bytes = 0;
  if constexpr (requires { dit->resident_bytes(); }) {
    bytes = (std::size_t)dit->resident_bytes();
  } else {
    bytes = model_memory::dir_weights_bytes(dir);
  }
  if (dit->ane_armed()) {
    if constexpr (requires { Dit::ane_plan_seq(_width, _height); }) {
      bytes += Dit::ane_runtime_bytes(dit->config(),
                                      Dit::ane_plan_seq(_width, _height));
    } else {   // FLUX.2: sized by its image rows
      bytes += Dit::ane_runtime_bytes(
          dit->config(), ((_width > 0 ? _width : 1024) / 16) *
                             ((_height > 0 ? _height : 1024) / 16));
    }
  }
  session()->info(fmt(
      "GenerateImageStage('{}'): the {} DiT kept warm for {:.0f} s{}",
      this->id(), name, _keep_loaded,
      bytes > 0 ? fmt(" ({} MB)", bytes >> 20)() : std::string()));
  mgr->keep_model_warm(kind, dir, _dit_warm_key, std::move(dit), bytes,
                       _keep_loaded);
}

const StageSpec&
GenerateImageStage::spec() const noexcept
{
  return kSpec;
}

void
GenerateImageStage::apply_constant(unsigned iport, const FlexData& beat)
{
  // Pre-launch twin of the runtime latch in process(): the same
  // beat and the same parse, early enough that declare_resources()
  // sees the model. Bookkeeping only -- nothing loads here; the
  // pipeline is not assembled yet (see Stage::apply_constant).
  if (iport == kModelCfgPort) {
    // Planning only -- the live preview's TAE and Fibo 2.0's guidance
    // (whether a negative's text is held beside the prompt's) are booked
    // from it; the runtime latch parses the beat as before.
    _plan_model_cfg = beat;
    return;
  }
  if (iport != kModelPort) { return; }
  apply_model_select_beat(beat, _hf_dir);
}

#ifdef VPIPE_BUILD_APPLE_SILICON

namespace {

// Output pixels per ref_latent cell, i.e. the factor the paired vae-encode
// applied when it produced ref_latent0 for this family:
//   krea2 / qwen-image-edit   [16, H/8,  W/8 ]   -> 8
//   boogu-image               [16, H/8,  W/8 ]   -> 8
//   flux2                     [C,  H/16, W/16]   -> 16
// (FLUX.2's VAE is 8x but vae-encode emits its 2x2-packed DiT latent, so the
// pixels-per-cell figure is 16 there too.)
// A transformer class this stage does NOT run, or empty when the root's
// class is one of its own (or there is no class to read).
//
// It exists so that a checkpoint nothing here implements costs a message
// rather than a wrong model: `t2i_family_` falls through to "krea2", so
// without this a family that lives in a plugin nobody loaded would load
// the Krea-2 path at full cost and denoise its checkpoint through a 12B
// config.
std::string
unclaimed_class_(const std::string& transformer_dir)
{
  namespace fs = std::filesystem;
  std::ifstream in(fs::path(transformer_dir) / "config.json");
  if (!in) { return {}; }
  FlexData fd = FlexData::from_json(in);
  if (!fd.is_object()) { return {}; }
  auto obj = fd.as_object();
  if (!obj.contains("_class_name")) { return {}; }
  const std::string cls(obj.at("_class_name").as_string(""));
  for (const char* own : {"Krea2Transformer2DModel",
                          "Flux2Transformer2DModel",
                          "QwenImageTransformer2DModel",
                          "QwenImage21Transformer2DModel",
                          "ZImageTransformer2DModel",
                          "Flux3Transformer2DModel",
                          "MOSSImageTransformer",
                          "BriaFibo2Transformer2DModel",
                          "BooguImageTransformer2DModel"}) {
    if (cls == own) { return {}; }
  }
  return cls;
}

// Which built-in denoisers actually DISPATCH Sol-Attn -- verified
// against `_sol->encode` in each transformer, not against what their
// configs accept. A family here that does not dispatch would run dense
// in silence; a family missing from here that does would be warned
// about a tier it is running.
bool
family_implements_sol_(const std::string& family)
{
  return family == "krea2" || family == "flux2" ||
         family == "qwen-image-21" || family == "z-image" ||
         family == "fibo2" ||
         family == "flux3" || family == "moss-image";
}

int
latent_scale_(const std::string& family)
{
  // Pixels per reference-latent cell: the VAE's spatial stride. 16 for
  // FLUX.2 (8x VAE + 2x patch), and 16 for Qwen-Image-2.1 -- whose VAE
  // is five levels, so the stride is the VAE's alone with no patch to
  // multiply by.
  // NOT Z-Image, which looks like FLUX.2 and is not: the 2x patch is
  // the DIT's, not the VAE's. MetalFlux2Vae emits
  // [channels, H/(8*patch), W/(8*patch)] and Z-Image's VAE is patch 1,
  // so a reference latent cell is 8 pixels and inferring a size at 16
  // would ask for twice the picture.
  // FLUX.3 takes FLUX.2's VAE unchanged, so its cell is 16 too, and so
  // does MOSS-Image (the small decoder changes widths, not strides).
  // Fibo 2.0 takes FLUX.2's VAE too (its own 2x patch is the same one
  // the VAE stage's latent already carries).
  if (family == "flux2" || family == "flux3" || family == "qwen-image-21" ||
      family == "moss-image" || family == "fibo2") {
    return 16;
  }
  return 8;
}

// FLUX.2 empirical flow-shift mu (diffusers Flux2Pipeline.compute_empirical_mu):
// the sigma time-shift is resolution- AND step-dependent (a fixed shift washes
// out at other resolutions -> grey). `image_seq_len` is the packed image token
// count (grid_h*grid_w); `num_steps` the sampler steps. vpipe's FlowScheduler
// `shift` field IS this mu (time_shift(sigma, mu, exponential)).
double
flux2_empirical_mu_(int image_seq_len, int num_steps)
{
  const double a1 = 8.73809524e-05, b1 = 1.89833333;
  const double a2 = 0.00016927,     b2 = 0.45666666;
  const double n = (double)image_seq_len;
  if (n > 4300.0) { return a2 * n + b2; }
  const double m_200 = a2 * n + b2;
  const double m_10  = a1 * n + b1;
  const double a = (m_200 - m_10) / 190.0;
  const double b = m_200 - 200.0 * a;
  return a * (double)num_steps + b;
}

// The transformer family from <root>/transformer/config.json `_class_name`:
// "Flux2Transformer2DModel" -> "flux2"; "QwenImageTransformer2DModel" ->
// "qwen-image-edit"; else "krea2".
//
// A family that lives in a plugin is handled by the registry instead, and
// a class neither knows is refused before this -- see unclaimed_class_()
// -- because the fall-through here is "krea2", and silently loading a 12B
// Krea-2 config over another model's checkpoint is the worst failure this
// stage has.
std::string
t2i_family_(const std::string& transformer_dir)
{
  namespace fs = std::filesystem;
  // A Krea-2 DiT in Krea's own single-file layout (a community fine-tune,
  // usually FP8) has no config.json at all, so it is recognised by its
  // tensor names -- a READING, where the "krea2" at the bottom is a
  // guess.
  if (genai::krea2::is_native_dit_file(transformer_dir)) { return "krea2"; }
  // Likewise a Qwen-Image-2.1 DiT published as ONE file (unsloth's FP8):
  // no config, so read off its tensor names.
  if (genai::MetalQwenImage21Transformer::is_single_file_dit(
          transformer_dir)) {
    return "qwen-image-21";
  }
  std::ifstream in(fs::path(transformer_dir) / "config.json");
  if (in) {
    FlexData fd = FlexData::from_json(in);
    if (fd.is_object()) {
      auto obj = fd.as_object();
      if (obj.contains("_class_name")) {
        const std::string cls(obj.at("_class_name").as_string(""));
        if (cls == "Flux2Transformer2DModel") { return "flux2"; }
        // FLUX.3 Image Klein: nothing like FLUX.2's network, and named
        // because the fall-through below would load a Krea-2 config over
        // its weights.
        if (cls == "Flux3Transformer2DModel") { return "flux3"; }
        // MOSS-Image: a single-stream DiT on FLUX.2's VAE; named for the
        // same reason as the rest.
        if (cls == "MOSSImageTransformer") { return "moss-image"; }
        // Fibo 2.0: Z-Image's block around a Perceiver and injection
        // blocks, on FLUX.2's VAE.
        if (cls == "BriaFibo2Transformer2DModel") { return "fibo2"; }
        if (cls == "QwenImageTransformer2DModel") { return "qwen-image-edit"; }
        // Qwen-Image-2.1. A DIFFERENT network from the line above --
        // single-stream, /16 VAE, RGBA -- and named here because the
        // fall-through is "krea2", which would load a Krea-2 config over
        // these weights and run.
        if (cls == "ZImageTransformer2DModel") { return "z-image"; }
        if (cls == "QwenImage21Transformer2DModel") {
          return "qwen-image-21";
        }
        if (cls == "BooguImageTransformer2DModel") { return "boogu-image"; }
      }
    }
  }
  return "krea2";
}

// Physical RAM + component weight bytes now live in stages/model-memory.h so
// the conditioner and the VAE stages size the box the same way this one does.
using model_memory::phys_ram;

// A Qwen-Image-2.1 checkpoint's OWN schedule, which is not one thing:
// the base model shifts linspace(1, 1/S, S) by a per-image mu and
// stretches the tail onto 0.02 (use_dynamic_shifting), while a distilled
// checkpoint -- Qwen-Image-2.1-Turbo -- turns dynamic shifting off and
// ships its raw nodes as `sample_sigmas` in model_index.json, which
// diffusers' QwenImage21Pipeline then uses as the schedule and the step
// count. Read from the files rather than assumed, because the two are
// byte-identical networks and only these files tell them apart. A file
// that is absent keeps the base model's value.
}  // namespace

GenerateImageStage::Qi21Schedule
GenerateImageStage::qi21_checkpoint_schedule(const std::string& root)
{
  namespace fs = std::filesystem;
  GenerateImageStage::Qi21Schedule ck;
  {
    std::ifstream in(fs::path(root) / "scheduler" / "scheduler_config.json");
    FlexData fd = in ? FlexData::from_json(in) : FlexData::make_null();
    if (fd.is_object()) {
      auto o = fd.as_object();
      auto real = [&](const char* k, double d) {
        if (!o.contains(k)) { return d; }
        const FlexData v = o.at(k);
        return v.is_null() ? 0.0 : v.as_real(d);   // null = off
      };
      auto num = [&](const char* k, int d) {
        return o.contains(k) ? (int)o.at(k).as_int(d) : d;
      };
      if (o.contains("use_dynamic_shifting")) {
        ck.dynamic = o.at("use_dynamic_shifting").as_bool(ck.dynamic);
      }
      if (o.contains("time_shift_type")) {
        ck.shift_type = std::string(o.at("time_shift_type").as_string(
            ck.shift_type));
      }
      ck.shift = real("shift", ck.shift);
      ck.shift_terminal = real("shift_terminal", ck.shift_terminal);
      ck.base_shift = real("base_shift", ck.base_shift);
      ck.max_shift = real("max_shift", ck.max_shift);
      ck.base_seq = num("base_image_seq_len", ck.base_seq);
      ck.max_seq = num("max_image_seq_len", ck.max_seq);
    }
  }
  {
    std::ifstream in(fs::path(root) / "model_index.json");
    FlexData fd = in ? FlexData::from_json(in) : FlexData::make_null();
    if (fd.is_object() && fd.as_object().contains("sample_sigmas")) {
      FlexData a = fd.as_object().at("sample_sigmas");   // keep the owner
      if (a.is_array()) {
        for (const auto& v : a.as_array()) {
          const double x = v.as_real(-1.0);
          if (!(x > 0.0 && x <= 1.0)) { ck.sample_sigmas.clear(); break; }
          ck.sample_sigmas.push_back(x);
        }
      }
    }
  }
  return ck;
}

namespace {

// The DiT weights directory for a checkpoint root, at PLANNING time --
// before initialize() has resolved a family and while the two ledgers
// still have to name the same bytes.
//
// `transformer/` for every diffusers tree, and VOSR's own `checkpoints/`
// for the one family that ships no diffusers tree at all. Getting this
// wrong is quiet in the worst way: a directory that does not exist
// weighs zero, so the largest thing the graph loads is planned as
// nothing and the box is admitted to a run it cannot hold.
std::string
planning_dit_dir_(const std::string& root)
{
  namespace fs = std::filesystem;
  std::error_code ec;
  const fs::path t = fs::path(root) / "transformer";
  if (fs::is_directory(t, ec)) { return t.string(); }
  genai::MetalVosrTransformer::Config vc;
  if (genai::MetalVosrTransformer::read_config(root, &vc, nullptr)) {
    const std::string w = genai::MetalVosrTransformer::weights_dir(root);
    if (!w.empty()) { return w; }
  }
  return t.string();
}


// FLUX.2 VAE decode-peak channel count from <root>/vae/config.json. The top
// up-block's FIRST resnet reads block_out_channels[1] (256) at FULL res (the
// upsampled level-2 output) before it reduces to block_out_channels[0] (128),
// so the largest full-res im2col scales with max(block_out[0], block_out[1]) --
// mirror MetalFlux2Vae::decode_peak_bytes, which was widened to that max in the
// 2K uint32-overflow fix (using block_out[0] alone under-modelled the peak 2x,
// so the DiT-reclaim decision could skip freeing and then OOM the vae-decode).
// 128 when unreadable -- the stock AutoencoderKLFlux2 block_out[0] default.
int
flux2_vae_base_(const std::string& root)
{
  namespace fs = std::filesystem;
  std::ifstream in(fs::path(root) / "vae" / "config.json");
  if (in) {
    FlexData fd = FlexData::from_json(in);
    if (fd.is_object()) {
      auto o = fd.as_object();
      // The DECODER's widths where they differ (`decoder_block_out_
      // channels`, the FLUX.2 small decoder MOSS-Image ships): this sizes
      // a decode. `null` there (FLUX.3's) is "as the encoder's".
      const char* key = o.contains("decoder_block_out_channels") &&
                                o.at("decoder_block_out_channels").is_array()
                            ? "decoder_block_out_channels"
                            : "block_out_channels";
      if (o.contains(key) && o.at(key).is_array()) {
        FlexData bo = o.at(key);
        auto arr = bo.as_array();       // owner `bo` outlives this view
        // block_out_channels parses as a JSON INTEGER array (mode Int), so read
        // each element with as_real() -- as_real_span() only exposes native-real
        // arrays and returns EMPTY here (which had silently pinned this to 128).
        if (arr.size() >= 2) {
          const int b0 = (int)arr[0].as_real(0.0);
          const int b1 = (int)arr[1].as_real(0.0);
          if (b0 > 0 && b1 > 0) { return std::max(b0, b1); }
        }
        if (arr.size() >= 1) {
          const int b0 = (int)arr[0].as_real(0.0);
          if (b0 > 0) { return b0; }
        }
      }
    }
  }
  return 128;
}

// Force or suppress the free-the-DiT-for-the-decode reclaim.
//
//   VPIPE_IMAGE_DIT_RECLAIM=1  always free   =0  never free
//   unset                      decide from the memory budget (the default)
//
// Exists because the path is otherwise reachable only on a box tight
// enough to need it, which is exactly the box a test is least likely to
// be running on -- so the reclaim, the hand-off to the weight pool and
// the reload it implies went unexercised anywhere but in production on a
// 16 GB machine. `0` is the operator's side of the same switch: on a box
// with room to spare, never giving the DiT up saves the reload on every
// prompt.
//
// Returns 1 / 0 / -1 for force / suppress / decide.
int
forced_dit_reclaim_()
{
  const char* e = std::getenv("VPIPE_IMAGE_DIT_RECLAIM");
  if (e == nullptr || *e == '\0') { return -1; }
  return std::atoi(e) != 0 ? 1 : 0;
}

// ---- VAE decode peaks ---------------------------------------------------
//
// The number a streamed DiT reserves for, the number its free-for-decode
// tests against, and the number published to the plan as the decode
// arena are all THE SAME number. They were three copies of one
// expression per family, which is how the Qwen-Image-Edit reserve came to
// be computing the FLUX.2 shape against a `_vae_base` its arm had never
// set. One definition per VAE topology, so a drift is a compile error
// rather than a silently generous or stingy reserve.
//
// Both are f16 (2 bytes/element) and both model ONE up-level's working
// set: the per-up-block command-buffer split (default on) commits and
// frees each level before the next, so the resident peak is a level, not
// the summed up-path.

// AutoencoderKLFlux2, and the plain diffusers AutoencoderKL that
// Boogu-Image uses. The big convs run the NAX hardware conv and never
// materialize a 9*base im2col, so the peak is the per-level activation
// pool -- MEASURED at ~7x one full-res base-channel buffer at 1024^2
// (~2.8 GB), NOT the 9x-im2col figure, which was a ~2.3x phantom that
// over-estimated the decode ~2x. `base` = max(block_out[0..1]); see
// flux2_vae_base_ and MetalFlux2Vae::decode_peak_bytes.
std::size_t
flux2_decode_peak_(int w, int h, int base)
{
  if (w <= 0 || h <= 0 || base <= 0) { return 0; }
  // Seven full-resolution planes plus the band MetalFlux2Vae allocates on
  // every decode (whole-image, capped at 128 MB), as its decode_peak_bytes.
  const std::size_t hwb = (std::size_t)h * (std::size_t)w * (std::size_t)base;
  return hwb * 2 * 7 + std::min(hwb * 9 * 2, (std::size_t)(128u << 20));
}

// AutoencoderKLQwenImage -- Krea-2 AND Qwen-Image-Edit, which run the
// same MetalKrea2Vae. Two terms, mirroring
// MetalKrea2Vae::decode_peak_bytes: the top level's activations (six
// full-resolution base-channel f16 tensors, five of them measured live)
// plus the im2col band, which a GPU without the hardware conv pays on
// every 3x3. The band is CAPPED there (kDecodeBandMax, 128 MB) rather
// than sized from free headroom, which is the only reason a fixed figure
// here can be an upper bound at all. Booked unconditionally: this runs
// before anything is loaded, so it cannot know whether this GPU gathers.
// `base` = the checkpoint's base_dim, see krea2_vae_base_.
std::size_t
qwen_decode_peak_(int w, int h, int base, bool residual = false)
{
  if (w <= 0 || h <= 0 || base <= 0) { return 0; }
  const std::size_t hw = (std::size_t)h * (std::size_t)w;
  // MetalKrea2Vae::decode_peak_bytes' units -- nine for a residual
  // decoder (Qwen-Image-2.1's), six for the plain one -- plus the one an
  // int8 tier adds, because vae-decode's `i8_gemm` is not this stage's
  // to see, and the decode's fixed term.
  constexpr std::size_t kFixed = 256u << 20;   // its kDecodeFixedBytes
  const std::size_t acts =
      hw * (std::size_t)base * 2 * (residual ? 10 : 7) + kFixed;
  // Widest cin the decoder bands for is base*dim_mult[1], and dim_mult is
  // {1,2,4,4} for this family.
  const std::size_t band = std::min(hw * 9 * (std::size_t)base * 2 * 2,
                                    (std::size_t)(128ull << 20));
  return acts + band;
}

// The Krea-2 VAE (WanVAE-style, shared with Qwen-Image-Edit) sizes its conv
// stack from `base_dim`, not diffusers' block_out_channels. Read it for the
// decode-peak estimate; fall back to the checkpoint default (96).
int
krea2_vae_base_(const std::string& root)
{
  namespace fs = std::filesystem;
  std::ifstream in(fs::path(root) / "vae" / "config.json");
  if (in) {
    FlexData fd = FlexData::from_json(in);
    if (fd.is_object()) {
      auto o = fd.as_object();
      if (o.contains("base_dim")) {
        const int b = (int)o.at("base_dim").as_real(0.0);
        if (b > 0) { return b; }
      }
    }
  }
  return 96;
}

// The DECODER's base, which is the one the decode peak is about. Most
// checkpoints on this VAE are symmetric and `decoder_base_dim` is
// absent, so this is `krea2_vae_base_` for all of them -- but
// Qwen-Image-2.1 is 96 encoder / 144 decoder, and reading the encoder's
// number there under-books the peak by a third. That is the quietest
// possible version of this bug: the estimate decides whether the DiT is
// freed before the decode, so a short answer does not fail, it just
// runs the decode in less room than it needs.
int
krea2_vae_dec_base_(const std::string& root)
{
  namespace fs = std::filesystem;
  std::ifstream in(fs::path(root) / "vae" / "config.json");
  if (in) {
    FlexData fd = FlexData::from_json(in);
    if (fd.is_object()) {
      auto o = fd.as_object();
      if (o.contains("decoder_base_dim")) {
        const int b = (int)o.at("decoder_base_dim").as_real(0.0);
        if (b > 0) { return b; }
      }
    }
  }
  return krea2_vae_base_(root);
}

}  // namespace

void
GenerateImageStage::reset_run_state()
{
  // If a previous run left the weights UNLOADED (the idle-unload
  // policy drops them between beats), let this launch load them again:
  // ensure_loaded_'s once-only guard is per-Stage, not per-launch, so
  // without this the stage stays inert for the whole run. When the
  // weights are still held we deliberately leave the guard set --
  // reloading on top of a resident copy is exactly what doubles peak
  // memory.
  if (!_dit && !_flux2_dit && !_qie_dit && !_boogu_dit && !_qi21_dit &&
      !_zi_dit && !_flux3_dit && !_moss_dit && !_fibo2_dit) {
    _load_attempted = false;
    _dit_unloaded   = false;
    // The holding correction describes a DiT that is no longer here. A
    // reloaded one starts at its trunk and grows again, so keeping the
    // old figures would suppress the first revision that matters --
    // `_dit_revised` is compared against to avoid replanning for an
    // unchanged number, and a stale one reads as unchanged.
    _dit_load_floor = 0;
    _dit_revised    = 0;
  }

  // Per-launch reset: the stage survives a stop/relaunch, and the
  // sources upstream re-emit on every launch. Without this the
  // re-emitted beat is never latched and this stage keeps the previous
  // run's selection.
  _model_latched     = false;
  _sampler_latched   = false;
  _scheduler_latched = false;
  _model_cfg         = FlexData{};
  // Every per-generation input starts over: a run's beats pair among
  // themselves, never with the last run's, and the seed sequence starts
  // again from `seed`.
  _cfg_in.reset();
  _cond_in.reset();
  _sampler_in.reset();
  _scheduler_in.reset();
  _cond_beat.reset();
  _neg_beat.reset();
  _generation = 0;
  _seed       = _seed_base;
  // The reference latents are held across generations (`_ref`), which
  // is right within a run and wrong across one: an input still holding
  // is not read, so on the next launch the stage would never consume
  // the reference beat the vae-encode upstream just produced. That beat
  // would sit on the wire forever (the edge shows a permanent backlog)
  // and the run would silently re-use the PREVIOUS image's reference.
  _ref_port[0] = RefLatent{};
  _ref_port[1] = RefLatent{};
  _ref_port_in[0].reset();
  _ref_port_in[1].reset();
  _ref_list.clear();
  _ref_list_in.reset();
  _ref.clear();
  // ...AND THE VALUES PARSED OUT OF IT. `_model_cfg` going empty is not
  // enough: the parsed derivatives live in their own members, and an
  // ABSENT key means "no opinion" on purpose -- which is right WITHIN a
  // run and wrong across one. A relaunch may be a different graph
  // entirely, and a graph with no z-image-model-config would otherwise
  // inherit the previous run's guidance. On Z-Image that is visible:
  // the distilled Turbo checkpoint wants 0 and takes a stale 4 as an
  // instruction, producing an over-guided picture rather than an error.
  //
  // Only this family's are reset here. The same shape exists for
  // Qwen-Image-2.1's `use_kv_cache` and FLUX.2's GenerationParams, and
  // changing those is a decision about their defaults rather than a
  // fix to this one.
  _zi_guidance  = 0.0;
  _zi_cfg_norm  = 0.0;
  _zi_cfg_trunc = 1.0;
  // FLUX.3's schedule override: a graph that stops naming one must not
  // keep running it.
  _flux3_alpha_cfg = 0.0;
  // Fibo 2.0's guidance: "no opinion" again; and the plan's copy of the
  // model config, read by declare_resources, which has run by now.
  _fibo2_guidance_cfg = -1.0;
  _plan_model_cfg = FlexData{};
  // MOSS-Image's guidance, skipped blocks and shift: "no opinion" again.
  _moss_guidance_cfg = -1.0;
  _moss_skip_set = false;
  _moss_skip_cfg.clear();
  _moss_shift_cfg = 0.0;
  _moss_slg_layers.clear();
  _moss_slg_scale = 0.0;
  _moss_slg_start = 0.0;
  _moss_slg_stop = 1.0;
  // Qwen-Image-2.1's schedule keys, which carry no default of their own
  // to argue about: a graph that stops naming a turbo schedule must not
  // keep running it.
  _qi21_sigmas.clear();
  _qi21_shift_terminal = -1.0;
  _qi21_sched_noted = false;
#ifdef VPIPE_BUILD_APPLE_SILICON
  // The preview keys too: a relaunch whose graph no longer names a
  // preview VAE must not inherit one.
  if (_preview) {
    _preview->configure(LatentPreviewSpec{},
                        genai::MetalTaeDecoder::Trim::kLeading);
  }
#endif
}

void
GenerateImageStage::revise_dit_declaration_(const std::string& dit_dir) const
{
  auto* mgr = session() ? session()->services()->generative_model_manager() : nullptr;
  if (mgr == nullptr || dit_dir.empty()) { return; }
  // WHAT THE SET HOLDS IS ONLY THE TRUNK on a streaming DiT, and the
  // trunk is the small half. Retained tensors go through
  // tensor()/derived() and are cached in the set, but the blocks
  // residency PROMOTES are the model's OWN buffers -- not cache
  // entries, and so invisible here. A family that can answer for its
  // whole holding is asked instead; the set is the fallback for the
  // ones that cannot.
  std::size_t held = dit_resident_bytes_();
  if (held == 0) {
    auto ws = mgr->weight_set(dit_dir);
    if (!ws) { return; }
    held = ws->stats().bytes;
  }
  mgr->revise_declaration(dit_dir, held);
  session()->log_debug(fmt(
      "GenerateImageStage('{}'): streaming DiT keeps {} MB resident; revised "
      "from its {} MB on disk so peers do not size against weights "
      "that are never there", this->id(), held >> 20,
      model_memory::dir_weights_bytes(dit_dir) >> 20));
}

// What the LIVE DiT says it holds, or 0 when this family cannot answer
// cheaply. Only the streaming-aware ones are asked -- a preloaded DiT's
// declaration is already right and re-stating it per generation would
// replan the graph for nothing.
std::size_t
GenerateImageStage::dit_resident_bytes_() const
{
  if (_qi21_dit) { return (std::size_t)_qi21_dit->resident_bytes(); }
  if (_zi_dit) { return (std::size_t)_zi_dit->resident_bytes(); }
  if (_flux3_dit) { return (std::size_t)_flux3_dit->resident_bytes(); }
  if (_moss_dit) { return (std::size_t)_moss_dit->resident_bytes(); }
  if (_fibo2_dit) { return (std::size_t)_fibo2_dit->resident_bytes(); }
  // Krea-2, FLUX.2, Boogu and Qwen-Image-Edit keep no total of their own:
  // what their set CACHED (the trunk, and every block when preloaded --
  // held int8 layouts included, which are derived entries) plus what
  // residency PROMOTED (the model's own buffers) plus the slot pairs
  // while they exist. Asked of none of them before, a streaming one was
  // declared at its load-time trunk for the whole run -- MEASURED on
  // Krea-2 under 48 GB: ~3.5 GB declared while residency held all 28
  // blocks, 23.2 GB wired.
  auto cached_plus = [&](const std::string& dir, std::size_t promoted,
                         std::size_t slots) -> std::size_t {
    auto* mgr = session() != nullptr
                    ? session()->services()->generative_model_manager()
                    : nullptr;
    auto ws = mgr != nullptr ? mgr->weight_set(dir) : nullptr;
    const std::size_t cached = ws ? ws->stats().bytes : 0;
    return cached + promoted + slots;
  };
  if (_dit) {
    return cached_plus(_krea2_dit_dir, _dit->resident_block_bytes(),
                       _dit->slot_bytes());
  }
  if (_flux2_dit) {
    return cached_plus(_flux2_dit_dir, _flux2_dit->resident_block_bytes(),
                       _flux2_dit->slot_bytes());
  }
  if (_boogu_dit) {
    return cached_plus(_boogu_dit_dir, _boogu_dit->resident_block_bytes(),
                       _boogu_dit->slot_bytes());
  }
  if (_qie_dit) {
    return cached_plus(_qie_dit_dir, _qie_dit->resident_block_bytes(),
                       _qie_dit->slot_bytes());
  }
  // A registered family's generator answers the same question. Without
  // it a streaming plugin DiT whose residency grows stayed declared at
  // the floor its load reported -- the trunk -- for the whole run.
  if (_plugin_gen) { return (std::size_t)_plugin_gen->resident_bytes(); }
  return 0;
}

// THE WORKING SET MOVES DURING THE RUN, and a declaration taken once at
// load says otherwise.
//
// A streaming DiT loads holding its trunk and grows through the denoise
// as block residency admits what the box turns out to have room for --
// MEASURED on this family, 1088 MB declared at load against 7360 MB
// held three forwards later. Revised only at load, the manager reports
// the first number for the rest of the run, and a peer sizing against
// it keeps weights the box no longer has room for. That is the
// oversubscription that ends in swap, and it is silent: every
// individual figure is one somebody really said.
//
// So the correction is asked again after every generation, which is the
// same rule generate-video applies per clip. Both ledgers move: the
// manager's, which is what peers read through weight_footprint(), and
// the plan's holding, which is what the phase figures are computed
// from. A revision replans the graph, so it is only published when the
// number actually moved.
// A streamed DiT residency kept whole gives its slot pair back between
// generations, before the holding below is read -- the holding is then the
// preload's, and the decode that follows does not run beside two blocks
// nothing reads.
void
GenerateImageStage::release_idle_slots_()
{
  if (_dit) { (void)_dit->release_idle_slots(); }
  if (_flux2_dit) { (void)_flux2_dit->release_idle_slots(); }
  if (_qie_dit) { (void)_qie_dit->release_idle_slots(); }
  if (_qi21_dit) { (void)_qi21_dit->release_idle_slots(); }
  if (_zi_dit) { (void)_zi_dit->release_idle_slots(); }
  if (_flux3_dit) { (void)_flux3_dit->release_idle_slots(); }
  if (_moss_dit) { (void)_moss_dit->release_idle_slots(); }
  if (_fibo2_dit) { (void)_fibo2_dit->release_idle_slots(); }
  if (_boogu_dit) { (void)_boogu_dit->release_idle_slots(); }
}

void
GenerateImageStage::correct_dit_holding_(const char* when) const
{
  const std::size_t held = dit_resident_bytes_();
  if (held == 0 || _dit_holding_dir.empty()) { return; }
  auto* mgr = session() ? session()->services()->generative_model_manager()
                        : nullptr;
  if (mgr == nullptr) { return; }
  mgr->revise_declaration(_dit_holding_dir, held);

  StageMemory m = declare_memory();
  StageHolding* h = nullptr;
  for (auto& x : m.holdings) {
    if (x.source == _dit_holding_dir) { h = &x; break; }
  }
  // Named separately from the encoder on purpose (see declare_memory),
  // which is what makes it findable here -- resident_bytes() is this
  // DiT's total and cannot be apportioned across several holdings.
  if (h == nullptr) { return; }
  _dit_load_floor = correct_loaded_holding(*h, held, _dit_load_floor);
  if (h->preload == _dit_revised) { return; }
  _dit_revised = h->preload;
  revise_memory(m);
  session()->log_debug(fmt(
      "GenerateImageStage('{}'): the DiT holds {} MB {}; the plan holds it "
      "at {} MB, floor {} MB", this->id(), held >> 20, when,
      h->preload >> 20, h->floor >> 20));
}

namespace {

// Whether a registered family's own claims name `dir` -- the rule by
// which a family answers for the conditioning encoder: it claimed it, so
// the stage does not book it a second time on the family's behalf.
bool
family_claims_dir_(const genai::ImageModelFamily& fam,
                   const std::string& root, const std::string& dir)
{
  for (const ResourceClaim& c : fam.declare_resources(root)) {
    if (c.kind == model_memory::kWeightsKind && c.key == dir) { return true; }
  }
  return false;
}

}  // namespace

// int8_layout_bytes_() where this stage's DiT will take the int8
// layouts: i8_gemm on the fast or token tier, on matrix cores (the only
// box the tiers engage on); zero otherwise.
// THE DiT THIS STAGE LOADS, which is what both ledgers must name: the
// `dit_dir` a graph sets (a w4 / w8 / mixed pack, an FP8 file, a Draw
// Things .ckpt), resolved as the load resolves it, and the checkpoint's
// own only when it sets none. Planning the root's copy while loading
// another put a DiT nothing loads in the plan and left the real one
// undeclared -- so every revision of it was a no-op and the holding
// correction after a generation never found it. MEASURED on Qwen-Image-
// 2.1 with a 4 GB w4 pack: 13.6 GB of the release's bf16 DiT booked
// beside it, and the conditioner's streaming decision sized against it.
// With a config-only hf_dir (the usual .ckpt graph) the DiT was planned
// at nothing at all.
std::string
GenerateImageStage::planned_dit_dir_(const std::string& root) const
{
  if (_dit_dir.empty()) { return planning_dit_dir_(root); }
  return resolve_model_dir(session(), _dit_dir);
}

// WHETHER A CLAIMED ANE TIER MAY RUN is the plan's call, read where the
// grant exists. Krea-2 and FLUX.2 asked; Qwen-Image-2.1, Z-Image, FLUX.3
// and MOSS-Image built their CoreML modules whatever the plan said --
// ~0.7 GB on MOSS-Image a plan had refused.
bool
GenerateImageStage::ane_granted_(std::size_t bytes, const char* family) const
{
  if (bytes > 0 && model_memory::coreml_grant(session(), ane_claim_label_(),
                                              bytes, 1) > 0) {
    return true;
  }
  session()->info(fmt(
      "GenerateImageStage('{}'): the ANE feed-forward was requested but the "
      "plan left no room for {}'s modules ({} MB); keeping the GPU",
      this->id(), family, bytes >> 20));
  return false;
}

// What a STREAMED dense checkpoint's slots add on the int8 layouts: each
// keeps its bf16 source as the refill target AND the layout quantized
// from it (QWeight::ds in each family), so a slot is ~1.5 blocks. Block
// 0's converted projections of each streamed stack at int8, with the
// finest group's scales, for each of the stack's two slots.
std::size_t
GenerateImageStage::int8_slot_bytes_(const std::string& dit) const
{
  if (!_i8_gemm || _i8_tier <= 0 || session() == nullptr) { return 0; }
  auto* mc = session()->services()->metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return 0; }
  if (dit_config_quant_bits_(dit) != 0) { return 0; }
  // Per family: block 0 of every streamed stack (each stack has its own
  // slot pair), and the 2-D tensors there the layout leaves dense -- the
  // modulation tables, and FLUX.2's double-block text stream, which never
  // clears the int8 row floor.
  struct Stack {
    const char* prefix;
    std::vector<const char*> dense;
  };
  const std::string fam = t2i_family_(dit);
  std::vector<Stack> stacks;
  if (fam == "qwen-image-21") {
    stacks = {{"transformer_blocks.0.", {}}};
  } else if (fam == "krea2") {
    stacks = {{"transformer_blocks.0.", {"scale_shift_table"}}};
  } else if (fam == "z-image" || fam == "fibo2") {
    // Fibo 2.0's first main block is a plain one (no cross-attention).
    stacks = {{"layers.0.", {"adaLN"}}};
  } else if (fam == "moss-image") {
    stacks = {{"layers.0.", {}}};
  } else if (fam == "flux2") {
    stacks = {{"transformer_blocks.0.",
               {"add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
                "ff_context"}},
              {"single_transformer_blocks.0.", {}}};
  } else {
    return 0;
  }
  auto w = genai::MetalLlamaWeights::open_model(dit);
  if (!w.has_value()) { return 0; }
  const std::string_view sx = ".weight";
  std::size_t n = 0;
  for (const Stack& st : stacks) {
    const std::string pre = st.prefix;
    for (const std::string& nm : w->tensor_names()) {
      if (nm.rfind(pre, 0) != 0 || nm.size() <= sx.size() ||
          nm.compare(nm.size() - sx.size(), sx.size(), sx) != 0) {
        continue;
      }
      bool dense = false;
      for (const char* d : st.dense) {
        if (nm.find(d) != std::string::npos) { dense = true; }
      }
      const auto* ti = w->info(nm);
      if (dense || ti == nullptr || ti->shape.size() != 2) { continue; }
      const std::size_t N = (std::size_t)ti->shape[0];
      const std::size_t K = (std::size_t)ti->shape[1];
      n += N * K + std::max(N * (K / 256) * 2, N * 4);
    }
  }
  return 2 * n;
}

std::size_t
GenerateImageStage::int8_held_bytes_(const std::string& dit) const
{
  if (!_i8_gemm || _i8_tier <= 0 || session() == nullptr) { return 0; }
  auto* mc = session()->services()->metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return 0; }
  return int8_layout_bytes_(t2i_family_(dit), dit).held;
}

std::size_t
GenerateImageStage::int8_scratch_bytes_(const std::string& dit) const
{
  if (!_i8_gemm || _i8_tier <= 0 || session() == nullptr) { return 0; }
  auto* mc = session()->services()->metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return 0; }
  return int8_layout_bytes_(t2i_family_(dit), dit).scratch;
}

// THE ENGINE'S CALL on a w4 pack's int8 layout (the int8 tiers): HOLD it
// -- the fastest, at twice the w4 projections' bytes -- where the box has
// room, which is plan_streaming's own preload rule asked again with that
// growth added; CONVERT IT PER GEMM into one scratch where it has not, or
// where the blocks stream (a slot holds the pack as stored). MEASURED on
// the M5 Ultra, a 20-step 1024^2 Qwen-Image-2.1 w4 generation: per call
// costs +3.0% (fast) to +4.1% (token, token_plain) of the denoise against
// a layout held, which saves 1.8x of the w4 blocks' bytes.
// VPIPE_I8_W4_HOLD=0 / 1 overrides, for the A/B.
bool
GenerateImageStage::hold_w4_int8_(const model_memory::StreamPlan& plan,
                                  bool streamed, const std::string& dit)
{
  const std::size_t held = int8_held_bytes_(dit);
  if (held == 0) { return false; }
  bool hold = !streamed &&
              model_memory::preload_fits_extra(plan, held,
                                               model_memory::kStreamHeadroom);
  const char* forced = std::getenv("VPIPE_I8_W4_HOLD");
  if (forced != nullptr) { hold = std::atoi(forced) != 0; }
  const char* why = forced != nullptr ? "VPIPE_I8_W4_HOLD"
                    : hold            ? "the box has room"
                    : streamed        ? "the blocks stream"
                                      : "the box has no room";
  session()->info(fmt(
      "GenerateImageStage('{}'): the w4 projections' int8 layout {}",
      this->id(),
      hold ? fmt("HELD ({} MB more; {})", held >> 20, why)()
           : fmt("CONVERTED PER GEMM into a {} MB scratch ({}; holding it "
                 "would take {} MB more)", int8_scratch_bytes_(dit) >> 20,
                 why, held >> 20)()));
  // The report ledger booked the layout held -- the preload at its
  // largest, before anything decided. Converted per GEMM, it is not, so
  // the holding drops it now rather than at the first generation's
  // correction.
  _i8_hold_w4 = hold;
  _i8_hold_decided = true;
  if (!hold) { revise_memory(declare_memory()); }
  return hold;
}

StageMemory
GenerateImageStage::declare_memory() const
{
  StageMemory m;
  if (_hf_dir.empty()) { return m; }
  namespace fs = std::filesystem;
  const std::string root = resolve_model_dir(session(), _hf_dir);
  const std::string mllm = (fs::path(root) / "mllm").string();
  std::error_code ec;
  const std::string enc = fs::exists(mllm, ec)
                              ? mllm
                              : (fs::path(root) / "text_encoder").string();
  const std::string dit = planned_dit_dir_(root);
  // NAMED SEPARATELY, both of them. A DiT and an encoder have different
  // sizes, different streaming forms and -- on other stages -- different
  // lifetimes, and collapsing them to one number loses the distinction
  // the plan exists to make. Naming them is also what keeps a second
  // generate-image over the same model from being billed twice.
  // A REGISTERED FAMILY ANSWERS FOR ITSELF here too, and for the same
  // reason: `dit` below is a diffusers path a plugin's checkpoint need
  // not have. Its holdings replace the DiT's, never the encoder's --
  // the text encoder is the conditioner's, and it is resident beside
  // whichever DiT this is.
  bool held_by_family = false;
  bool enc_answered = false;
  if (genai::ImageModelFamily* fam =
          genai::ImageModelRegistry::get().claim_for(
              session(), root, resolve_model(session(), _hf_dir).model_type)) {
    // `releases` / `reclaimable` are the STAGE's to say, not the
    // family's: whether idle weights are dropped is this stage's
    // policy. It has none -- it holds for the run -- so both stay
    // false, and a family that set them is overruled rather than
    // believed.
    for (const auto& h : fam->declare_holdings(root)) {
      m.hold(h.source, h.preload, h.floor);
    }
    held_by_family = true;
    enc_answered = family_claims_dir_(*fam, root, enc);
  }
  if (!held_by_family) {
    // A w4 pack's int8 layout: held where the box has room (the
    // preload, at its largest), a scratch where it streams (the floor).
    const std::size_t floor = dit_floor_bytes_(root, dit) +
                              int8_slot_bytes_(dit);
    const bool held = !_i8_hold_decided || _i8_hold_w4;
    m.hold(dit,
           model_memory::dir_weights_bytes(dit) +
               (held ? int8_held_bytes_(dit) : 0),
           floor);
  }
  if (!enc_answered) { m.hold(enc, model_memory::dir_weights_bytes(enc)); }
  // No unload policy on this stage: it holds both for the run. Freeing
  // the DiT for a decode that would not otherwise fit
  // (free_flux2_dit_for_decode_ and its siblings) is a TRANSIENT, not a
  // lifetime, and saying `releases` here would tell peers the weights
  // are gone after this stage runs when they are not.
  return m;
}

std::vector<ResourceClaim>
GenerateImageStage::declare_resources() const
{
  if (_hf_dir.empty()) { return {}; }
  namespace fs = std::filesystem;
  const std::string root = resolve_model_dir(session(), _hf_dir);
  const std::string mllm = (fs::path(root) / "mllm").string();
  std::error_code ec;
  const std::string enc = fs::exists(mllm, ec)
                              ? mllm
                              : (fs::path(root) / "text_encoder").string();
  // THE DiT CARRIES ITS FLOOR, and that is the difference between a
  // graph being reported and a graph being refused.
  //
  // declare_memory() above states the same floor, but into a different
  // ledger: that one is what a run REPORTS. The floor on a
  // ResourceClaim is what phase_footprint_floor() and phase_peak() read,
  // and those are the only place a graph is turned away. Claimed without
  // one, a streaming DiT is judged at its full on-disk size -- so
  // "everything streamable at its floor" reads the same as "everything
  // preloaded", and a 17 GB checkpoint that runs on a couple of blocks
  // is weighed as 17 GB against the box.
  // A REGISTERED FAMILY DECLARES ITS OWN, because the built-in shape
  // below assumes a diffusers `transformer/` subdir and a plugin's
  // checkpoint commonly has no such thing. Empty means the family
  // declined, and a hole in this ledger reads as room that is not
  // there -- so the encoder claim is still added either way.
  if (genai::ImageModelFamily* fam =
          genai::ImageModelRegistry::get().claim_for(
              session(), root, resolve_model(session(), _hf_dir).model_type)) {
    // A family's CoreML claim -- the ANE module a family implementing
    // `ane_ffn` holds -- is kept only when the graph asked for the tier,
    // and under THIS stage's label: the family cannot know the stage's
    // id, and two stages sharing a label would share one grant. The
    // grant is read back at load, and the label handed to the family
    // there (genai::create_args::kCoreMLLabel), as generate-video does.
    const bool ane = genai::accel::flag(&_accel, genai::accel::kAneFfn);
    std::vector<ResourceClaim> pout;
    for (auto& c : fam->declare_resources(root)) {
      std::size_t ub = 0;
      int un = 0;
      if (model_memory::parse_coreml_claim(c, nullptr, &ub, &un)) {
        if (!ane) { continue; }
        for (auto& r : model_memory::coreml_claims(ane_claim_label_(), ub,
                                                   un, c.phase)) {
          pout.push_back(std::move(r));
        }
        continue;
      }
      pout.push_back(std::move(c));
    }
    // A registered family renders previews through the output seam, so
    // its preview is booked exactly as a built-in's is.
    for (auto& c : latent_preview_claims(
             session(), this->id(),
             LatentPreviewSpec::from_model_config(_plan_model_cfg),
             genai::MetalTaeDecoder::Trim::kLeading,
             _width > 0 ? _width : 1024, _height > 0 ? _height : 1024, 1)) {
      pout.push_back(std::move(c));
    }
    // THE ENCODER, unless the family answered for it. A family whose
    // conditioner is its OWN stage claims the encoder itself -- phased
    // to `condition` when that stage lets it go after the prompt -- and
    // an unphased claim added here would book it through the denoise
    // regardless. MEASURED with a plugin family whose conditioner holds
    // a 32 GB MLLM: the encoder was released after the prompt, yet this
    // claim kept it in the denoise phase and the family's 12 GB DiT
    // streamed from disk on a 64 GB box.
    if (!family_claims_dir_(*fam, root, enc)) {
      for (auto& c : model_memory::weight_claims({enc})) {
        pout.push_back(std::move(c));
      }
    }
    // THE DECODE ARENA, as for the built-ins below. This branch returned
    // before it, so no registered family's decode ever reached the plan
    // -- and vae-decode's per-beat revision cannot create a label that
    // was never declared, so it stayed absent for the whole run. The
    // size comes from root/vae's config like a built-in's; a VAE that
    // config does not describe gets the presence marker to revise.
    std::size_t parena =
        model_memory::vae_decode_scratch_bytes(root, _width, _height);
    if (parena == 0) { parena = model_memory::kUnknownArena; }
    for (auto& c : model_memory::scratch_claims("vae-decode", parena,
                                                model_memory::kPhaseDecode)) {
      pout.push_back(std::move(c));
    }
    return pout;
  }
  const std::string dit = planned_dit_dir_(root);
  // (A w4 pack's per-GEMM conversion scratch is the int8 context's, and
  // is booked with the activations below -- not in the weights' floor. A
  // dense checkpoint's slots carry their int8 layout too, and that IS a
  // holding: int8_slot_bytes_.)
  const std::size_t floor =
      dit_floor_bytes_(root, dit) + int8_slot_bytes_(dit);
  std::vector<ResourceClaim> out{model_memory::weight_claim_streamable(
      dit, floor)};
  for (auto& c : model_memory::weight_claims({enc})) {
    out.push_back(std::move(c));
  }

  // The DECODE ARENA this stage's output implies.
  //
  // Declared HERE, by the stage that has the geometry, because nothing
  // else in the graph does: vae-decode takes its size from whatever
  // latent arrives, so at planning time it cannot name a number, and
  // sizing itself against peer WEIGHTS is the proxy it has been using.
  // The arena is the larger quantity for every image family -- FLUX.2's
  // VAE weighs 160 MB and its decode peaks near 2.8 GB at 1024^2.
  //
  // kPhaseDecode, so it is not charged to the denoise: the arena does
  // not exist while the DiT runs. It still lands in the box-level peak,
  // which is where it belongs -- generate-image frees its DiT for a
  // decode that will not fit (free_flux2_dit_for_decode_ and its
  // siblings), and the peak is what says whether that will be needed.
  //
  // An over-estimate here costs caution; the geometry is config, so it
  // is settled before anything loads.
  //
  // With NO geometry -- an edit graph takes its size from the reference
  // image, so height/width are absent -- this is kUnknownArena: a
  // presence marker, negligible by construction, that gives the runtime
  // figure something to revise. Declaring nothing would leave
  // publish_decode_arena_ with no entry to correct.
  std::size_t arena =
      model_memory::vae_decode_scratch_bytes(root, _width, _height);
  if (arena == 0) { arena = model_memory::kUnknownArena; }
  for (auto& c : model_memory::scratch_claims("vae-decode", arena,
                                              model_memory::kPhaseDecode)) {
    out.push_back(std::move(c));
  }

  // ANE RESIDENCY: one unit.
  //
  // The bytes a CoreML model holds are invisible to every other ledger in
  // this process -- not a weight set, not a SharedBuffer, not even a Metal
  // allocation -- so without this claim the tier's module appears nowhere,
  // beside a DiT that has revised its own declaration DOWN to a streaming
  // floor. The feed-forward runs on ONE shared runtime-weight module
  // (weights staged per block), so the cost is a single fixed unit:
  // granted or not, never partial. Claimed in the DENOISE phase, the only
  // one the DiT is alive in.
  //
  // Only for the family that HAS the tier. Claiming for a graph whose
  // config sites never call apply_ane_() would declare bytes nothing
  // allocates, which is the same lie pointing the other way.
  //
  // A QUANTIZED checkpoint claims the same unit: its feed-forward is
  // dequantized into the module's fp16 inputs per block, so the module
  // costs what it costs a dense one.
  if (genai::accel::flag(&_accel, genai::accel::kAneFfn) &&
      t2i_family_(dit) == "krea2") {
    genai::MetalKrea2Transformer::Config kc;
    kc.ane_qkv = genai::accel::flag(&_accel, genai::accel::kAneQkv);
    for (auto& c : model_memory::coreml_claims(
             ane_claim_label_(),
             genai::MetalKrea2Transformer::ane_runtime_bytes(
                 kc,
                 genai::MetalKrea2Transformer::ane_plan_seq(_width, _height)),
             1, model_memory::kPhaseDenoise)) {
      out.push_back(std::move(c));
    }
  }
  // FLUX.2's tiers -- the double feed-forward (and q|k|v) and the single
  // projection -- are fixed modules too, summed into ONE unit: granted
  // together or not at all. Sized from the checkpoint's own dims, since the
  // klein-4B and -9B modules differ by ~1.8x.
  // Qwen-Image-2.1's feed-forward tier: ONE runtime-weight module, so a
  // single fixed unit, granted or not. Claimed only for the family that
  // calls it -- booking bytes nothing allocates is the same lie the
  // other way round.
  if (genai::accel::flag(&_accel, genai::accel::kAneFfn) &&
      t2i_family_(dit) == "qwen-image-21") {
    genai::MetalQwenImage21Transformer::Config qc;
    (void)genai::MetalQwenImage21Transformer::Config::read_dims(dit, &qc);
    qc.ane_qkv = genai::accel::flag(&_accel, genai::accel::kAneQkv);
    for (auto& c : model_memory::coreml_claims(
             ane_claim_label_(),
             genai::MetalQwenImage21Transformer::ane_runtime_bytes(
                 qc,
                 genai::MetalQwenImage21Transformer::ane_plan_seq(_width,
                                                                  _height)),
             1, model_memory::kPhaseDenoise)) {
      out.push_back(std::move(c));
    }
  }
  // Z-Image's ANE tier is the feed-forward over its MAIN stack only --
  // the refiners run a shorter sequence than the module is sized for.
  // Claimed in UNITS because CoreML holds those bytes and no other
  // ledger in this process can see them.
  if (genai::accel::flag(&_accel, genai::accel::kAneFfn) &&
      t2i_family_(dit) == "z-image") {
    genai::MetalZImageTransformer::Config zc;
    (void)genai::MetalZImageTransformer::Config::read_dims(dit, &zc);
    zc.ane_qkv = genai::accel::flag(&_accel, genai::accel::kAneQkv);
    for (auto& c : model_memory::coreml_claims(
             ane_claim_label_(),
             genai::MetalZImageTransformer::ane_runtime_bytes(
                 zc, genai::MetalZImageTransformer::ane_plan_seq(_width,
                                                                 _height)),
             1, model_memory::kPhaseDenoise)) {
      out.push_back(std::move(c));
    }
  }
  // FLUX.3's ACTIVATIONS, which no weight claim can see and which are
  // most of the denoise at large sizes: 8.1 GB at 4096^2 against an
  // 11.1 GB DiT -- and on a box that streams the DiT, nearly all of what
  // the denoise holds. MEASURED there before this claim existed: the
  // plan's denoise phase read 989 MB on the M5 for a forward that held
  // 9.4 GB. Sized from the geometry, so an edit graph -- which takes its
  // size from the reference -- books the presence marker instead.
  if (t2i_family_(dit) == "flux3") {
    genai::MetalFlux3Transformer::Config fc;
    std::size_t bytes = model_memory::kUnknownArena;
    if (_width > 0 && _height > 0 &&
        genai::MetalFlux3Transformer::Config::read_dims(dit, &fc)) {
      // The attention tiers allocate their own scratch beside the planes.
      fc.sage = _sage;
      fc.sol = _sol;
      bytes = genai::MetalFlux3Transformer::forward_scratch_bytes(
          fc, genai::MetalFlux3Transformer::ane_plan_seq(_width, _height));
    }
    for (auto& c : model_memory::scratch_claims(
             fmt("flux3-dit:{}", this->id())(), bytes,
             model_memory::kPhaseDenoise)) {
      out.push_back(std::move(c));
    }
  }
  // FIBO 2.0's ACTIVATIONS, for FLUX.3's reason. An edit attends over
  // the picture AND its reference (one at the picture's size, as the
  // shipped graphs wire it), beside a ~1k-token prompt (an edit's picture
  // alone is 576 of them). An edit graph takes its size from the
  // reference, which no plan can see; it books the edit the shipped
  // graphs run -- 1024^2, the model's trained area, as ane_plan_seq
  // assumes -- rather than the presence marker, which left every
  // launch-time decision blind to ~1 GB. A text-to-image graph (Dev)
  // has no reference and is booked as an edit all the same: the plan
  // cannot see the wiring, and over is the safe side. A GUIDED graph
  // (Dev: its guidance above 1, from the model-config constant or this
  // stage's own) holds a negative's text beside the prompt's, booked at
  // the prompt's length. generate_fibo2_ revises it all to the real
  // rows.
  if (t2i_family_(dit) == "fibo2") {
    genai::MetalFibo2Transformer::Config bc;
    std::size_t bytes = model_memory::kUnknownArena;
    const int w = _width > 0 ? _width : 1024;
    const int h = _height > 0 ? _height : 1024;
    if (genai::MetalFibo2Transformer::Config::read_dims(dit, &bc)) {
      bc.i8_gemm = _i8_gemm;
      bc.i8_tier = _i8_tier;
      bc.sage = _sage;
      bc.sol = _sol;
      auto* mcm = session() != nullptr
                      ? session()->services()->metal_compute()
                      : nullptr;
      double g = _guidance_scale;
      if (_plan_model_cfg.is_object()) {
        auto o = _plan_model_cfg.as_object();
        const std::string fam = model_config::family_of(_plan_model_cfg);
        if ((fam.empty() || fam == "fibo2") && o.contains("guidance_scale") &&
            o.at("guidance_scale").as_real(-1.0) >= 0.0) {
          g = o.at("guidance_scale").as_real(-1.0);
        }
      }
      bytes = genai::MetalFibo2Transformer::forward_scratch_bytes(
          bc, 2 * (w / 16) * (h / 16), 1024,
          dit_config_quant_bits_(dit) > 0,
          mcm != nullptr && mcm->supports_matrix_cores(),
          g > 1.0 ? 1024 : -1);
    }
    for (auto& c : model_memory::scratch_claims(
             fmt("fibo2-dit:{}", this->id())(), bytes,
             model_memory::kPhaseDenoise)) {
      out.push_back(std::move(c));
    }
  }
  // MOSS-Image's ACTIVATIONS, for the same reason as FLUX.3's: a
  // streamed DiT holds little beside them. Sized at the picture with a
  // 512-token prompt (the reference allows 4096; a longer prompt grows
  // the joint sequence by ~25 KB a row a plane).
  if (t2i_family_(dit) == "moss-image") {
    genai::MetalMossImageTransformer::Config mc;
    std::size_t bytes = model_memory::kUnknownArena;
    if (_width > 0 && _height > 0 &&
        genai::MetalMossImageTransformer::Config::read_dims(dit, &mc)) {
      mc.i8_gemm = _i8_gemm;
      mc.i8_tier = _i8_tier;
      mc.sage = _sage;
      mc.sol = _sol;
      auto* mcm = session() != nullptr
                      ? session()->services()->metal_compute()
                      : nullptr;
      bytes = genai::MetalMossImageTransformer::forward_scratch_bytes(
          mc, _height / genai::moss_image::kPixelsPerToken,
          _width / genai::moss_image::kPixelsPerToken, 512, _sol.enabled,
          dit_config_quant_bits_(dit) > 0,
          mcm != nullptr && mcm->supports_matrix_cores());
    }
    for (auto& c : model_memory::scratch_claims(
             fmt("moss-image-dit:{}", this->id())(), bytes,
             model_memory::kPhaseDenoise)) {
      out.push_back(std::move(c));
    }
  }
  // QWEN-IMAGE-2.1's, Z-IMAGE's, KREA-2's AND FLUX.2's ACTIVATIONS, for
  // FLUX.3's reason: no weight claim can see them, and a streamed DiT
  // holds little beside them. None of the four booked any before -- the
  // denoise phase carried weights alone. Sized at the picture with a
  // 512-token prompt; the int8 / Sage / Sol terms only where the box has
  // the matrix cores they run on.
  {
    const std::string fam = t2i_family_(dit);
    auto* mc = session() != nullptr
                   ? session()->services()->metal_compute()
                   : nullptr;
    const bool mx = mc != nullptr && mc->supports_matrix_cores();
    const int img = _width > 0 && _height > 0
                        ? (_width / 16) * (_height / 16) : 0;
    const int bits = dit_config_quant_bits_(dit);
    std::size_t bytes = 0;
    if (img > 0 && fam == "qwen-image-21") {
      genai::MetalQwenImage21Transformer::Config c;
      if (genai::MetalQwenImage21Transformer::Config::read_dims(dit, &c)) {
        c.i8_gemm = _i8_gemm; c.i8_tier = _i8_tier;
        c.sage = _sage; c.sol = _sol;
        bytes = genai::MetalQwenImage21Transformer::forward_scratch_bytes(
            c, img, 512, /*cfg=*/true, bits > 0, mx);
      }
    } else if (img > 0 && fam == "z-image") {
      genai::MetalZImageTransformer::Config c;
      if (genai::MetalZImageTransformer::Config::read_dims(dit, &c)) {
        c.i8_gemm = _i8_gemm; c.i8_tier = _i8_tier;
        c.sage = _sage; c.sol = _sol;
        bytes = genai::MetalZImageTransformer::forward_scratch_bytes(
            c, img, 512, bits > 0, mx);
      }
    } else if (img > 0 && fam == "krea2") {
      genai::MetalKrea2Transformer::Config c;   // Turbo's dims
      c.i8_gemm = _i8_gemm; c.i8_tier = _i8_tier;
      c.sage = _sage; c.sol = _sol;
      bytes = genai::MetalKrea2Transformer::forward_scratch_bytes(
          c, img, 512, mx);
    } else if (img > 0 && fam == "flux2") {
      genai::MetalFlux2Transformer::Config c;
      if (genai::MetalFlux2Transformer::read_dims(dit, &c)) {
        c.i8_gemm = _i8_gemm; c.i8_tier = _i8_tier;
        c.sage = _sage; c.sol = _sol;
        bytes = genai::MetalFlux2Transformer::forward_scratch_bytes(
            c, img, 512, bits, mx);
      }
    }
    // No geometry (an edit graph sizes from its reference): the
    // presence marker, as FLUX.3's.
    if (img == 0 && (fam == "qwen-image-21" || fam == "z-image" ||
                     fam == "krea2" || fam == "flux2")) {
      bytes = model_memory::kUnknownArena;
    }
    if (bytes > 0) {
      for (auto& c : model_memory::scratch_claims(
               fmt("{}-dit:{}", fam, this->id())(), bytes,
               model_memory::kPhaseDenoise)) {
        out.push_back(std::move(c));
      }
    }
  }
  // MOSS-Image's tiers: the main blocks' SITU feed-forward and, beside it,
  // q / k / v as one matmul -- fixed modules, one unit, in the denoise.
  if (genai::accel::flag(&_accel, genai::accel::kAneFfn) &&
      t2i_family_(dit) == "moss-image") {
    genai::MetalMossImageTransformer::Config mc;
    (void)genai::MetalMossImageTransformer::Config::read_dims(dit, &mc);
    mc.ane_qkv = genai::accel::flag(&_accel, genai::accel::kAneQkv);
    for (auto& c : model_memory::coreml_claims(
             ane_claim_label_(),
             genai::MetalMossImageTransformer::ane_runtime_bytes(
                 mc, genai::MetalMossImageTransformer::ane_plan_seq(
                         _width, _height)),
             1, model_memory::kPhaseDenoise)) {
      out.push_back(std::move(c));
    }
  }
  // Fibo 2.0's tiers are Z-Image's: the main stack's feed-forward and,
  // with ane_qkv, its q/k/v -- a module each.
  if (genai::accel::flag(&_accel, genai::accel::kAneFfn) &&
      t2i_family_(dit) == "fibo2") {
    genai::MetalFibo2Transformer::Config bc;
    (void)genai::MetalFibo2Transformer::Config::read_dims(dit, &bc);
    bc.ane_qkv = genai::accel::flag(&_accel, genai::accel::kAneQkv);
    for (auto& c : model_memory::coreml_claims(
             ane_claim_label_(),
             genai::MetalFibo2Transformer::ane_runtime_bytes(
                 bc, genai::MetalFibo2Transformer::ane_plan_seq(_width,
                                                                _height)),
             1, model_memory::kPhaseDenoise)) {
      out.push_back(std::move(c));
    }
  }
  // FLUX.3's tier is the joint blocks' feed-forward: ONE runtime-weight
  // module, one unit, in the denoise phase.
  if (genai::accel::flag(&_accel, genai::accel::kAneFfn) &&
      t2i_family_(dit) == "flux3") {
    genai::MetalFlux3Transformer::Config fc;
    (void)genai::MetalFlux3Transformer::Config::read_dims(dit, &fc);
    for (auto& c : model_memory::coreml_claims(
             ane_claim_label_(),
             genai::MetalFlux3Transformer::ane_runtime_bytes(
                 fc, genai::MetalFlux3Transformer::ane_plan_seq(_width,
                                                                _height)),
             1, model_memory::kPhaseDenoise)) {
      out.push_back(std::move(c));
    }
  }
  if (genai::accel::flag(&_accel, genai::accel::kAneFfn) &&
      t2i_family_(dit) == "flux2") {
    genai::MetalFlux2Transformer::Config fc;
    (void)genai::MetalFlux2Transformer::read_dims(dit, &fc);
    fc.ane_qkv = genai::accel::flag(&_accel, genai::accel::kAneQkv);
    for (auto& c : model_memory::coreml_claims(
             ane_claim_label_(),
             genai::MetalFlux2Transformer::ane_runtime_bytes(
                 fc, ((_width > 0 ? _width : 1024) / 16) * ((_height > 0 ? _height : 1024) / 16)),
             1,
             model_memory::kPhaseDenoise)) {
      out.push_back(std::move(c));
    }
  }
  // The live preview's TAE and its decode -- exact, from the TAE's
  // headers; see latent_preview_claims. NOT gated on a family list:
  // the spec is what decides, and it is empty unless this family's
  // model-config named a `preview_vae`, so a family that renders no
  // preview claims nothing. A list here would have to be kept in step
  // with the model-config stages that carry the key, and would fail
  // silently in the direction that under-books. The registered-family
  // branch above asks the same way. The spec is the model-config
  // CONSTANT's (apply_constant): `_model_cfg` is the runtime latch, empty
  // at a first launch's plan and the last run's at a relaunch's.
  for (auto& c : latent_preview_claims(
           session(), this->id(),
           LatentPreviewSpec::from_model_config(_plan_model_cfg),
           genai::MetalTaeDecoder::Trim::kLeading,
           _width > 0 ? _width : 1024, _height > 0 ? _height : 1024, 1)) {
    out.push_back(std::move(c));
  }
  return out;
}

// Both directions of one question: what did the caller ask for, and
// which family is resident. It runs on every change to either, because a
// config beat and a model reference arrive on different ports and either
// can be first -- parsing at only one of those moments loses whichever
// came earlier. False when the beat is for a DIFFERENT family, which is
// reported and then ignored.
bool
GenerateImageStage::apply_model_config_()
{
  const std::string want = model_config::family_of(_model_cfg);
  if (!want.empty() && want != _family) {
    session()->warn(fmt(
        "GenerateImageStage('{}'): the model_config beat is for the '{}' "
        "family but the resident checkpoint is '{}'; IGNORING it and using "
        "the defaults. Wire the config stage that matches the checkpoint",
        this->id(), want, _family));
    return false;
  }
#ifdef VPIPE_BUILD_APPLE_SILICON
  // The preview keys are the same on every family's config source and
  // the port is this stage's, so they are read here, beside the family's
  // own keys. Every image TAE trims the leading raw frames: a still is
  // one latent frame, and the temporal TAEs keep its last sub-frame.
  if (!_preview) {
    _preview = std::make_unique<LatentPreviewer>(session(),
                                                 std::string(this->id()));
    // AN IMAGE STAGE DROPS ITS TAE AFTER EVERY GENERATION. Its preview
    // scratch is claimed for the denoise phase alone (see
    // latent_preview_claims) and at 1024^2 that is hundreds of
    // megabytes -- held into the decode, it would be memory the plan
    // says is not there, on the one box and phase where that matters.
    // A reload costs a read of ~10-23 MB per generation.
    _preview->release_each_generation(true);
  }
  _preview->configure(LatentPreviewSpec::from_model_config(_model_cfg),
                      genai::MetalTaeDecoder::Trim::kLeading);
#endif
  std::string perr;
  // The adapter keys, read the same way by every family that has one.
  // The PATH is a LOAD-TIME argument -- it decides whether a
  // fused-activation weave could be built at all -- so a beat that
  // changes it once the DiT is up is reported rather than dropped in
  // silence. The STRENGTH is live: it rides the adapter's GEMM as a
  // constant, so it can be swept without rebuilding anything.
  //
  // Every SLOT, read the same way. Slot 0 is the plain `lora` keys and
  // slot i the `lora<i+1>` ones (stages/lora-slots.h), so a graph that
  // names one adapter reads exactly as it always did and each further
  // one is additive. Which slot holds what is the caller's order, not a
  // role the model knows.
  auto adapter_keys = [&](const char* what, bool loaded, auto&& push_scale) {
    if (!_model_cfg.is_object()) { return; }
    FlexData cfg = _model_cfg;
    auto o = cfg.as_object();
    for (int i = 0; i < kLoraSlots; ++i) {
      LoraSlot& sl = _lora[(std::size_t)i];
      // An ABSENT key keeps what a previous beat set. A beat carrying
      // other keys and not these is saying nothing about the adapter,
      // and reading that as "drop it" would make an unrelated
      // re-emission silently un-adapt the run.
      const std::string_view kp = lora_slots::kPath[i];
      const std::string_view ks = lora_slots::kScale[i];
      const std::string lp = o.contains(kp)
          ? std::string(o.at(kp).as_string("")) : sl.path;
      const double ls = o.contains(ks) ? o.at(ks).as_real(1.0) : sl.scale;
      if (loaded && lp != sl.path) {
        session()->warn(fmt(
            "GenerateImageStage('{}'): the {} DiT is already loaded, so the "
            "model_config beat's `{}` is IGNORED -- it is a load-time "
            "argument. `{}` still applies",
            this->id(), what, kp, ks));
      } else {
        sl.path = lp;
      }
      if (ls != sl.scale) {
        sl.scale = ls;
        // Through the packing lora_specs_() recorded: config slot i is
        // the model's slot _lora_model_slot[i], or none.
        const int ms = _lora_model_slot[(std::size_t)i];
        if (ms >= 0) { push_scale(ms, (float)ls); }
      }
    }
  };
  if (_family == "z-image") {
    // The two published checkpoints ship BYTE-IDENTICAL transformer
    // configs; what separates them is the scheduler shift and whether
    // CFG is wanted. So the guidance is a graph decision, defaulted to
    // the distilled Turbo's 0 -- an absent key is "no opinion", not a
    // demand for the base model's 4.
    if (_model_cfg.is_object()) {
      auto o = _model_cfg.as_object();
      if (o.contains("guidance_scale")) {
        _zi_guidance = o.at("guidance_scale").as_real(_zi_guidance);
      }
      if (o.contains("cfg_normalization")) {
        _zi_cfg_norm = o.at("cfg_normalization").as_real(_zi_cfg_norm);
      }
      if (o.contains("cfg_truncation")) {
        _zi_cfg_trunc = o.at("cfg_truncation").as_real(_zi_cfg_trunc);
      }
    }
  }
  if (_family == "qwen-image-21") {
    // `use_kv_cache` -- absent means "no opinion", so the family's own
    // default (on) stands rather than being replaced by a false.
    if (_model_cfg.is_object()) {
      auto o = _model_cfg.as_object();
      if (o.contains("use_kv_cache")) {
        _qi21_use_kv_cache = o.at("use_kv_cache").as_bool(true);
      }
      // A schedule's raw sigma nodes: all in (0, 1], strictly falling.
      // Anything else is refused rather than half-applied -- a wrong
      // node is a different sampler that still produces a picture.
      if (o.contains("sigmas")) {
        const FlexData a = o.at("sigmas");
        std::vector<double> v;
        bool ok = a.is_array();
        if (ok) {
          auto arr = a.as_array();
          for (std::size_t i = 0; i < arr.size(); ++i) {
            const double x = arr.at(i).as_real(-1.0);
            ok = ok && x > 0.0 && x <= 1.0 && (v.empty() || x < v.back());
            v.push_back(x);
          }
        }
        if (ok && !v.empty()) {
          _qi21_sigmas = std::move(v);
        } else {
          session()->warn(fmt(
              "GenerateImageStage('{}'): model_config `sigmas` must be a "
              "non-empty list of values in (0, 1], strictly falling; "
              "IGNORED, the default schedule runs", this->id()));
        }
      }
      if (o.contains("shift_terminal")) {
        const double t = o.at("shift_terminal").as_real(-1.0);
        if (t >= 0.0 && t < 1.0) { _qi21_shift_terminal = t; }
      }
    }
  }
  if (_family == "qwen-image-21") {
    adapter_keys("Qwen-Image-2.1", _qi21_dit != nullptr,
                 [&](int i, float s) {
                   if (_qi21_dit && _qi21_dit->lora_modules(i) > 0) {
                     _qi21_dit->set_lora_scale(i, s);
                   }
                 });
  }
  if (_family == "flux3") {
    // `alpha` reshapes the schedule (t = a s / (1 + (a - 1) s)); absent is
    // "no opinion", and the checkpoint's own (6.93) stands.
    if (_model_cfg.is_object()) {
      auto o = _model_cfg.as_object();
      if (o.contains("alpha")) {
        const double a = o.at("alpha").as_real(0.0);
        if (a > 0.0) {
          _flux3_alpha_cfg = a;
        } else {
          session()->warn(fmt(
              "GenerateImageStage('{}'): model_config `alpha` must be > 0; "
              "IGNORED, the checkpoint's {} stands", this->id(),
              _flux3_alpha));
        }
      }
    }
    adapter_keys("FLUX.3", _flux3_dit != nullptr, [&](int i, float s) {
      if (_flux3_dit && _flux3_dit->lora_modules(i) > 0) {
        _flux3_dit->set_lora_scale(i, s);
      }
    });
  }
  if (_family == "fibo2") {
    adapter_keys("Fibo 2.0", _fibo2_dit != nullptr, [&](int i, float s) {
      if (_fibo2_dit && _fibo2_dit->lora_modules(i) > 0) {
        _fibo2_dit->set_lora_scale(i, s);
      }
    });
    // guidance_scale: Dev's 5 (the model card's recipe, against the
    // conditioner's negative on iport1), Edit Turbo's 1. Absent is "no
    // opinion".
    if (_model_cfg.is_object()) {
      auto o = _model_cfg.as_object();
      if (o.contains("guidance_scale")) {
        const double g = o.at("guidance_scale").as_real(-1.0);
        if (g >= 0.0) {
          _fibo2_guidance_cfg = g;
        } else {
          session()->warn(fmt(
              "GenerateImageStage('{}'): model_config `guidance_scale` must "
              "be >= 0; IGNORED", this->id()));
        }
      }
    }
  }
  if (_family == "moss-image") {
    // guidance_scale (the reference's 3; 1 runs no negative at all),
    // uncond_skip_layers (its [22]; [] disables the skip) and shift (the
    // checkpoint's 5). Absent is "no opinion".
    if (_model_cfg.is_object()) {
      auto o = _model_cfg.as_object();
      if (o.contains("guidance_scale")) {
        const double g = o.at("guidance_scale").as_real(-1.0);
        if (g >= 0.0) {
          _moss_guidance_cfg = g;
        } else {
          session()->warn(fmt(
              "GenerateImageStage('{}'): model_config `guidance_scale` must "
              "be >= 0; IGNORED", this->id()));
        }
      }
      if (o.contains("uncond_skip_layers")) {
        const FlexData a = o.at("uncond_skip_layers");
        std::vector<int> v;
        bool ok = a.is_array();
        if (ok) {
          auto arr = a.as_array();
          for (std::size_t i = 0; i < arr.size(); ++i) {
            const int x = (int)arr.at(i).as_int(-1);
            ok = ok && x >= 0;
            v.push_back(x);
          }
        }
        if (ok) {
          _moss_skip_set = true;
          _moss_skip_cfg = std::move(v);
        } else {
          session()->warn(fmt(
              "GenerateImageStage('{}'): model_config `uncond_skip_layers` "
              "must be a list of block indices >= 0; IGNORED", this->id()));
        }
      }
      if (o.contains("shift")) {
        const double sh = o.at("shift").as_real(0.0);
        if (sh > 0.0) { _moss_shift_cfg = sh; }
      }
      if (o.contains("slg_layers") && o.at("slg_layers").is_array()) {
        const FlexData a = o.at("slg_layers");
        auto arr = a.as_array();
        _moss_slg_layers.clear();
        for (std::size_t i = 0; i < arr.size(); ++i) {
          const int x = (int)arr.at(i).as_int(-1);
          if (x >= 0) { _moss_slg_layers.push_back(x); }
        }
      }
      if (o.contains("slg_scale")) {
        _moss_slg_scale = std::max(0.0, o.at("slg_scale").as_real(0.0));
      }
      if (o.contains("slg_start")) {
        _moss_slg_start = o.at("slg_start").as_real(0.0);
      }
      if (o.contains("slg_stop")) {
        _moss_slg_stop = o.at("slg_stop").as_real(1.0);
      }
    }
    adapter_keys("MOSS-Image", _moss_dit != nullptr, [&](int i, float sc) {
      if (_moss_dit && _moss_dit->lora_modules(i) > 0) {
        _moss_dit->set_lora_scale(i, sc);
      }
    });
  }
  if (_family == "flux2") {
    _flux2_params =
        genai::MetalFlux2Transformer::GenerationParams::from_flex(_model_cfg,
                                                                  &perr);
    adapter_keys("FLUX.2", _flux2_dit != nullptr, [&](int i, float s) {
      if (_flux2_dit && _flux2_dit->lora_modules(i) > 0) {
        _flux2_dit->set_lora_scale(i, s);
      }
    });
  } else if (_family == "z-image") {
    adapter_keys("Z-Image", _zi_dit != nullptr, [&](int i, float s) {
      if (_zi_dit && _zi_dit->lora_modules(i) > 0) {
        _zi_dit->set_lora_scale(i, s);
      }
    });
  } else if (_family == "krea2") {
    adapter_keys("Krea-2", _dit != nullptr, [&](int i, float s) {
      if (_dit && _dit->lora_modules(i) > 0) {
        _dit->set_lora_scale(i, s);
      }
    });
  } else if (_family == "vosr") {
    // The restorer's tiling. LIVE, not load-time: the tile is an
    // argument to each restore rather than to the DiT's construction,
    // so a beat that changes it mid-run takes effect on the next
    // picture. PRESENCE is the reading: unset means the trained grid
    // and 0 means one pass, and those are different answers.
    if (_model_cfg.is_object()) {
      FlexData cfg = _model_cfg;
      auto o = cfg.as_object();
      if (o.contains("tile_size")) {
        _tile_size     = (int)o.at("tile_size").as_int(0);
        _tile_size_set = true;
      }
      if (o.contains("tile_overlap")) {
        _tile_overlap = (int)o.at("tile_overlap").as_int(32);
      }
    }
  }
  if (!perr.empty() && !_model_cfg.is_null()) {
    session()->warn(fmt("GenerateImageStage('{}'): model_config: {}",
                        this->id(), perr));
  }
  return true;
}

Job
GenerateImageStage::initialize(RuntimeContext& ctx)
{
  // Defer the (heavy) DiT load when a model-select source feeds the model iport
  // (its beat only arrives after the init barrier, in process()). Otherwise
  // load now from the config hf_dir, as before.
  const bool model_from_iport =
      ctx.num_iports() > kModelPort && ctx.iport_connected(kModelPort);
  // The SAME deferral for a wired model_config, and for a sharper
  // reason: `klein_kv` selects the attention recipe, so load() has to
  // know it. Loading here and reading the config afterwards would build
  // the DiT under the wrong recipe and then never rebuild it -- and
  // FLUX.2's two variants both load cleanly either way, so the mistake
  // would surface as wrong images rather than as an error.
  const bool cfg_from_iport =
      ctx.num_iports() > kModelCfgPort && ctx.iport_connected(kModelCfgPort);
  if (!model_from_iport && !cfg_from_iport) { ensure_loaded_(); }
  co_return;
}

void
GenerateImageStage::ensure_loaded_()
{
  if (_load_attempted) { return; }   // idempotent: load at most once
  _load_attempted = true;
  if (_hf_dir.empty()) {
    session()->error(fmt(
        "GenerateImageStage('{}'): no model -- set config.hf_dir or wire a "
        "model-select source to the model iport; inert", this->id()));
    return;
  }
  auto* mc = session() ? session()->services()->metal_compute() : nullptr;
  if (mc == nullptr) {
    session()->error(fmt(
        "GenerateImageStage('{}'): no metal-compute backend; the stage is inert",
        this->id()));
    return;
  }
  namespace fs = std::filesystem;
  const std::string root = resolve_model_dir(session(), _hf_dir);
  // The text encoder lives in the paired diffusion-conditioner stage, but its
  // weights are resident in the SAME process while the DiT runs, so the
  // encoder-coexistence streaming heuristics below still budget for it.
  // Boogu-Image calls its encoder `mllm/` (a full multimodal LLM, not a
  // diffusers text_encoder): sizing it as text_encoder/ silently scored the
  // encoder at ZERO bytes, so the streaming decision under-counted by the
  // largest peer in the pipeline and a bounded box would preload instead.
  std::string enc_dir = (fs::path(root) / "mllm").string();
  {
    std::error_code eec;
    if (!fs::is_directory(enc_dir, eec)) {
      enc_dir = (fs::path(root) / "text_encoder").string();
    }
  }
  const std::string dit_dir = _dit_dir.empty()
      ? (fs::path(root) / "transformer").string()
      : resolve_model_dir(session(), _dit_dir);
  // AN OUT-OF-TREE FAMILY FIRST, and if one claims this checkpoint it
  // is the whole of the dispatch: `_family` becomes its tag and every
  // built-in DiT member below stays null. Asked with `root` rather than
  // `dit_dir` because a family's `claims` looks at the checkpoint the
  // way it is published, which for most new models is not a diffusers
  // `transformer/` subdir at all.
  _plugin_family = genai::ImageModelRegistry::get().claim_for(
      session(), root, resolve_model(session(), _hf_dir).model_type);
  // A graph that named a `dit_dir` may have pointed `hf_dir` at a
  // directory holding only the encoder and the VAE -- which is what
  // happens when a quantized DiT is written beside a checkpoint rather
  // than into it. Ask again with the DiT the graph actually named.
  //
  // Second, not first: the root is the model's identity, and a family
  // must not be able to claim a run on the strength of a weights
  // directory when another family already claimed the checkpoint.
  if (_plugin_family == nullptr && !_dit_dir.empty()) {
    _plugin_family = genai::ImageModelRegistry::get().claim_for(
        session(), dit_dir, resolve_model(session(), _hf_dir).model_type);
  }
  // A CHECKPOINT NOTHING HERE RUNS. Checked only once the registry has
  // declined it, so a loaded plugin never sees this -- and refused rather
  // than warned, because the fall-through below is "krea2": without it a
  // family whose plugin was not loaded would load a 12B Krea-2 config
  // over its checkpoint, spend minutes and emit noise.
  if (_plugin_family == nullptr) {
    const std::string cls = unclaimed_class_(dit_dir);
    if (!cls.empty()) {
      session()->error(fmt(
          "GenerateImageStage('{}'): '{}' is a '{}' checkpoint, which no "
          "built-in family runs and no loaded plugin claims -- load the "
          "plugin that provides it (vpipe --plugin <path>) and re-run. The "
          "stage is inert rather than guessing a family, because the guess "
          "would be a different model at full cost", this->id(), root, cls));
      return;
    }
  }
  _family = _plugin_family != nullptr ? std::string(_plugin_family->tag())
                                      : t2i_family_(dit_dir);
  // VOSR ships no diffusers tree -- an args.json beside a `checkpoints/`
  // directory -- so `t2i_family_` cannot see it and falls through to the
  // "krea2" default. Its own reader is the only thing that can identify
  // it, and it refuses every variant this stage does not implement
  // (multi-step, 0.5B/SD2, auxiliary time conditioning), which matters
  // more here than usual: those load through the same tensor names and
  // then compute the wrong answer rather than failing.
  // FLUX.3 in BFL's own single-file layout: refused by name, because the
  // "krea2" fall-through below would otherwise take it.
  if (_plugin_family == nullptr && genai::flux3::is_bfl_root_layout(root)) {
    session()->error(fmt("GenerateImageStage('{}'): {}; inert", this->id(),
                         genai::flux3::bfl_root_layout_message(root)));
    return;
  }
  if (_plugin_family == nullptr && _family == "krea2") {
    std::string vwhy;
    genai::MetalVosrTransformer::Config vc;
    if (genai::MetalVosrTransformer::read_config(root, &vc, &vwhy)) {
      _family   = "vosr";
      _vosr_cfg = vc;
      // The FILE, not the directory: `ema_model.safetensors` is not a
      // name the checkpoint opener globs for, so a directory here opens
      // nothing.
      _vosr_dir = _dit_dir.empty()
          ? genai::MetalVosrTransformer::weights_path(root)
          : dit_dir;
    }
  }
  // NO CONFIG AND NO CLAIM: `_family` above is the fall-through, not a
  // reading. Say so, because the alternative is a log line that names
  // "krea2" with the same confidence it would have had from a config --
  // and the load that follows fails on missing tensors several seconds
  // later, describing a family nobody chose.
  //
  // A WARNING rather than a refusal, deliberately: a fused or
  // hand-assembled DiT directory with no config.json is a supported
  // thing to point `dit_dir` at, and turning that into an error would
  // break a working path to improve a message. The families that are
  // recognised-but-absent are refused above; this is the residue.
  //
  // NOT for VOSR: the probe above READ its args.json, so the family is a
  // reading and not a fall-through, and saying otherwise sends an
  // operator looking for a plugin that does not exist.
  if (_plugin_family == nullptr && _family != "vosr" &&
      !genai::krea2::is_native_dit_file(dit_dir) &&
      !genai::MetalQwenImage21Transformer::is_single_file_dit(dit_dir)) {
    std::error_code cec;
    if (!fs::exists(fs::path(dit_dir) / "config.json", cec)) {
      session()->warn(fmt(
          "GenerateImageStage('{}'): '{}' has no transformer config and no "
          "registered family claimed it, so the family was GUESSED as "
          "'{}'. If this is a single-file or ComfyUI-packed checkpoint, "
          "the family that reads it is a plugin that is not loaded",
          this->id(), dit_dir, _family));
    }
  }
  // The family is only now known, so this is the first moment a config
  // beat that already arrived can be parsed by anything -- and it has to
  // happen BEFORE the load below, since klein_kv is an argument to it.
  apply_model_config_();
  if (_plugin_family != nullptr) {
    session()->info(fmt(
        "GenerateImageStage('{}'): loading the '{}' DiT from '{}' (an "
        "out-of-tree family)", this->id(), _family, root));
    // This stage's own `lora` .. `lora8` keys feed the BUILT-IN families'
    // loaders. A registered family reads its adapters from its own
    // model_config (its config stage), so keys set here would otherwise
    // do nothing at all -- an adapter that silently never applies.
    for (int i = 0; i < kLoraSlots; ++i) {
      if (!_lora[(std::size_t)i].path.empty()) {
        session()->warn(fmt(
            "GenerateImageStage('{}'): `{}` is set on generate-image, but "
            "'{}' is an out-of-tree family that takes adapters on its own "
            "model-config stage -- set it there; it is IGNORED here",
            this->id(), lora_slots::kPath[i], _family));
      }
    }
    genai::ImageModelCreateArgs args;
    args.root       = root;
    args.model_type = resolve_model(session(), _hf_dir).model_type;
    // THE ANE TIER'S GRANT, for a family that booked a module (see
    // declare_resources). A refusal turns the key OFF in the bag, which
    // the family and every later request carry, so it is told no rather
    // than holding a module the plan left no room for. The label goes
    // with the args so the family can revise what it really holds.
    FlexData create_extra = FlexData::make_object();
    if (genai::accel::flag(&_accel, genai::accel::kAneFfn)) {
      for (const ResourceClaim& c : _plugin_family->declare_resources(root)) {
        std::size_t ub = 0;
        int un = 0;
        if (!model_memory::parse_coreml_claim(c, nullptr, &ub, &un)) {
          continue;
        }
        create_extra.as_object().insert_or_assign(
            genai::create_args::kCoreMLLabel,
            FlexData::make_string(ane_claim_label_()));
        if (model_memory::coreml_grant(session(), ane_claim_label_(), ub,
                                       un) <= 0) {
          genai::accel::set_flag(&_accel, genai::accel::kAneFfn, false);
          session()->info(fmt(
              "GenerateImageStage('{}'): the ANE feed-forward was requested "
              "but the plan left no room for {}'s module ({} MB); the "
              "family keeps its GPU feed-forward", this->id(), _family,
              ub >> 20));
        }
        break;
      }
    }
    args.borrowed_extra = &create_extra;
    // The DiT the graph NAMED, if it named one. Passed only when the
    // config actually set it: `dit_dir` defaults to <root>/transformer,
    // and handing a family that default would tell it the graph made a
    // choice it did not make -- on a checkpoint published as a single
    // file there is no transformer/ at all.
    //
    // Without this a registered family could not be pointed at a
    // quantized DiT beside its checkpoint, which every built-in family
    // has always been able to do.
    if (!_dit_dir.empty()) { args.dit_dir = dit_dir; }
    args.metal      = session()->services()->metal_compute();
    args.session    = session();
    // The stage's own residency verdict, so a family need not re-derive
    // it. It is free to disagree -- it knows its own sizes -- but it
    // should say so.
    args.prefer_streaming =
        model_memory::plan_streaming(session(), dit_dir, enc_dir,
                                     model_memory::kStreamHeadroom).stream;
    args.model_config = _model_cfg.is_object() ? &_model_cfg : nullptr;
    args.accel        = &_accel;
    // The picture the graph intends to make, through the family's own
    // grid: a load-time decision that scales with the beat wants the
    // right order of magnitude rather than none.
    {
      int gh = 16, gw = 16;
      _plugin_family->size_grid(root, &gh, &gw);
      auto up = [](int v, int g) { return g > 0 ? ((v + g - 1) / g) * g : v; };
      args.width  = up(_width, gw);
      args.height = up(_height, gh);
    }
    _plugin_gen = _plugin_family->load(args);
    if (!_plugin_gen) {
      // The family warned through args.session; saying it twice adds
      // nothing, but the stage has to record that it is inert.
      session()->warn(fmt(
          "GenerateImageStage('{}'): the '{}' family did not load; this "
          "stage is inert", this->id(), _family));
    } else {
      session()->info(fmt(
          "GenerateImageStage('{}'): '{}' ready -- latent {} channels at "
          "1/{} spatial", this->id(), _family,
          _plugin_gen->latent_channels(),
          _plugin_gen->spatial_compression()));
      // The holding the per-generation correction revises: the family's
      // own first holding, which is the name the plan merged it under.
      const auto hs = _plugin_family->declare_holdings(root);
      if (!hs.empty()) { _dit_holding_dir = hs.front().source; }
    }
    return;
  }
  const std::string size_desc =
      _infer_size ? std::string("auto (from ref_latent0)")
                  : std::to_string(_width) + "x" + std::to_string(_height);
  session()->log_debug(fmt(
      "GenerateImageStage('{}'): init root='{}' dit='{}' family={} "
      "(default {}, {} steps, seed {}, strength {})", this->id(), root,
      dit_dir, _family, size_desc, _steps, _seed, _strength));

  session()->info(fmt(
      "GenerateImageStage('{}'): loading {} DiT from '{}'", this->id(),
      // NAMED PER FAMILY, including the ones whose label is only a log
      // line: the fall-through is "Krea2 MMDiT", so a family without a
      // row here announces itself as a different model while loading
      // correctly. Two already did.
      _family == "flux2" ? "FLUX.2"
      : _family == "flux3" ? "FLUX.3 Image Klein"
      : _family == "fibo2" ? "Fibo 2.0"
      : _family == "moss-image" ? "MOSS-Image single-stream"
      : _family == "qwen-image-edit" ? "Qwen-Image-Edit MMDiT"
      : _family == "qwen-image-21" ? "Qwen-Image-2.1 single-stream"
      : _family == "boogu-image" ? "Boogu-Image NextDiT"
      : _family == "z-image" ? "Z-Image NextDiT"
      : _family == "vosr" ? "VOSR LightningDiT"
      : "Krea2 MMDiT", _family == "vosr" ? _vosr_dir : dit_dir));
  if (_family == "vosr") {
    if (_vosr_dir.empty()) {
      session()->error(fmt(
          "GenerateImageStage('{}'): '{}' looks like a VOSR checkpoint but "
          "holds no readable safetensors; inert", this->id(), root));
      return;
    }
    if (!load_vosr_dit_()) {
      session()->error(fmt(
          "GenerateImageStage('{}'): failed to load the VOSR restorer from "
          "'{}'; inert", this->id(), _vosr_dir));
      return;
    }
    session()->info(fmt(
        "GenerateImageStage('{}'): VOSR LightningDiT ready -- {} blocks at "
        "{} wide, one-step, latent {} channels at 1/8 spatial (trained at "
        "{}px)", this->id(), _vosr_cfg.depth, _vosr_cfg.dim,
        _vosr_cfg.latent_ch, _vosr->train_grid() * 8));
    return;
  }
  if (_family == "flux2") {
    // Stream the DiT blocks when the box can't hold encoder + DiT together
    // (e.g. the 18 GB 9B DiT + a 16 GB encoder on a 16/32 GB box); ~2-3x slower
    // per step but bounds peak RAM to ~one block. VPIPE_FLUX2_STREAM forces it.
    // STREAM, THEN RETAIN; `bounded` is the tight box's.
    bool stream_blocks;
    bool bounded = false;
    {
      const auto plan = model_memory::plan_streaming(
          session(), dit_dir, enc_dir, model_memory::kStreamHeadroom);
      stream_blocks = model_memory::stream_blocks_by_default(plan);
      if (const char* e = std::getenv("VPIPE_FLUX2_STREAM")) {
        stream_blocks = (std::atoi(e) != 0);
      }
      bounded = plan.stream && stream_blocks;
      session()->log_debug(fmt(
          "GenerateImageStage('{}'): FLUX.2 footprint {} GB (others {} GB) "
          "+ {} GB headroom vs {} GB RAM -> {}", this->id(),
          plan.footprint >> 30, plan.others >> 30,
          model_memory::kStreamHeadroom >> 30, phys_ram() >> 30,
          !stream_blocks ? "PRELOAD"
          : bounded      ? "STREAM blocks"
                         : "STREAM blocks, kept as the box allows"));
      // HELD by the box's room, streamed or not: residency converts each
      // block it keeps into the layout; a slot converts per GEMM.
      _i8_hold_w4 = hold_w4_int8_(plan, bounded, dit_dir);
    }
    // Before the take below: a tight box takes no DiT kept warm, whose
    // promoted blocks the floor it was admitted on does not count.
    _dit_bounded = bounded;
    genai::MetalFlux2Transformer::Config fcfg;
    fcfg.i8_gemm = _i8_gemm;
    fcfg.i8_tier = _i8_tier;
    fcfg.i8_hold_w4 = _i8_hold_w4;
    fcfg.sage = _sage;
    fcfg.sol  = _sol;
    // The ANE tiers, whose claim declare_resources booked: WHETHER they run
    // is the plan's grant, read here where the grant exists. Sized from the
    // checkpoint's own dims -- klein-4B and -9B differ by ~1.8x -- and a
    // refusal leaves the config without a session, which is how the DiT is
    // told to keep its GPU path.
    if (genai::accel::flag(&_accel, genai::accel::kAneFfn)) {
      namespace a = genai::accel;
      genai::MetalFlux2Transformer::Config dims = fcfg;
      (void)genai::MetalFlux2Transformer::read_dims(dit_dir, &dims);
      dims.ane_qkv = a::flag(&_accel, a::kAneQkv);
      const std::size_t bytes =
          genai::MetalFlux2Transformer::ane_runtime_bytes(
              dims, ((_width > 0 ? _width : 1024) / 16) * ((_height > 0 ? _height : 1024) / 16));
      if (bytes > 0 && model_memory::coreml_grant(
                           session(), ane_claim_label_(), bytes, 1) > 0) {
        fcfg.session    = session();
        fcfg.ane_qkv    = a::flag(&_accel, a::kAneQkv);
        fcfg.ane_rows   = (float)a::real(&_accel, a::kAneRows, 0.0);
        fcfg.ane_layers = (int)std::max<long long>(
            0, a::integer(&_accel, a::kAneLayers, 0));
      } else {
        session()->info(fmt(
            "GenerateImageStage('{}'): the ANE feed-forward was requested "
            "but the plan left no room for FLUX.2's modules ({} MB); keeping "
            "the GPU", this->id(), bytes >> 20));
      }
    }
    _flux2_params.apply_to(fcfg);
    // Nothing in the checkpoint says which recipe it wants, and getting it
    // backwards costs plausible-looking wrong images rather than an error --
    // so when the path LOOKS like the other variant, say so. A hint, never an
    // override: the config key stays authoritative.
    {
      std::string dl = dit_dir;
      for (auto& ch : dl) { ch = (char)std::tolower((unsigned char)ch); }
      const bool looks_kv = dl.find("-kv") != std::string::npos
                            || dl.find("_kv") != std::string::npos;
      if (looks_kv != _flux2_params.klein_kv) {
        session()->warn(fmt(
            "GenerateImageStage('{}'): '{}' looks like {} but klein_kv={}. The "
            "-kv checkpoint isolates reference tokens; running either variant "
            "under the other's attention produces wrong images silently",
            this->id(), dit_dir,
            looks_kv ? "FLUX.2-klein-9b-kv" : "plain FLUX.2-klein",
            _flux2_params.klein_kv ? "true" : "false"));
      }
    }
    {
      using Dit = genai::MetalFlux2Transformer;
      const auto loras = lora_specs_<Dit::LoraSpec>();
      _flux2_dit = take_warm_dit_<Dit>(kWarmFlux2, dit_dir,
                                       warm_key_(fcfg, stream_blocks, loras),
                                       "FLUX.2");
      if (!_flux2_dit) {
        _flux2_dit = Dit::load(weight_set_(dit_dir), mc, fcfg, stream_blocks,
                               loras);
      }
    }
    if (!_flux2_dit) {
      session()->error(fmt(
          "GenerateImageStage('{}'): failed to load the FLUX.2 DiT from '{}'; "
          "inert", this->id(), dit_dir));
      return;
    }
    // When the DiT had to stream (box can't hold encoder + DiT together), drop
    // the DiT's per-forward scratch after each generation so it doesn't crowd
    // out a large downstream VAE decode.
    _release_scratch = bounded;
    if (bounded) { revise_dit_declaration_(dit_dir); }
    // Cache the load params + VAE base so a memory-driven free after a
    // generation (free_flux2_dit_for_decode_) can reload identically.
    _flux2_dit_dir  = dit_dir;
    // The holding correct_dit_holding_ moves as residency grows.
    _dit_holding_dir = dit_dir;
    _flux2_stream   = stream_blocks;
    _vae_base       = flux2_vae_base_(root);
  } else if (_family == "qwen-image-edit") {
    // Dual-stream Qwen-Image-Edit DiT (20B). Stream the blocks when the box
    // can't hold encoder + DiT together (else it can't run on a 16 GB box), and
    // PIN as many leading blocks resident as fit ALONGSIDE the per-prompt
    // encoder: pinned + encoder + headroom <= RAM. The encoder is reloaded for
    // each prompt's conditioning and stays resident while the DiT weights do,
    // so -- unlike the calibration collector, which frees the encoder before
    // loading the DiT and can pin 60% -- the stage must budget around it. On a
    // 16 GB box with the ~14 GB bf16 encoder that leaves no room (pin 0 => pure
    // streaming, ~one block resident); a roomier box pins more. Pinned blocks
    // are read once + reused, only the tail streams. VPIPE_QIE_STREAM /
    // VPIPE_QIE_PIN_FRAC override.
    const auto plan = model_memory::plan_streaming(
        session(), dit_dir, enc_dir, model_memory::kStreamHeadroom);
    // STREAM, THEN RETAIN; `bounded` is the tight box's.
    bool stream_blocks = model_memory::stream_blocks_by_default(plan);
    if (const char* e = std::getenv("VPIPE_QIE_STREAM")) {
      stream_blocks = (std::atoi(e) != 0);
    }
    const bool bounded = plan.stream && stream_blocks;
    session()->log_debug(fmt(
        "GenerateImageStage('{}'): Qwen-Image-Edit footprint {} GB (others {} "
        "GB) + {} GB headroom vs {} GB RAM -> {}", this->id(),
        plan.footprint >> 30, plan.others >> 30,
        model_memory::kStreamHeadroom >> 30, phys_ram() >> 30,
        !stream_blocks ? "PRELOAD"
        : bounded      ? "STREAM blocks"
                       : "STREAM blocks, kept as the box allows"));
    genai::MetalQwenImageTransformer::Config qcfg;
    qcfg.sage    = _sage;
    qcfg.i8_gemm = _i8_gemm;
    _qie_dit = genai::MetalQwenImageTransformer::load(
        weight_set_(dit_dir), mc, qcfg, stream_blocks);
    if (!_qie_dit) {
      session()->error(fmt(
          "GenerateImageStage('{}'): failed to load the Qwen-Image-Edit DiT from "
          "'{}'; inert", this->id(), dit_dir));
      return;
    }
    if (stream_blocks) {
      // No prefix any more: the pinned-prefix policy was a fraction of
      // TOTAL RAM decided before the run and blind to the machine, and
      // the measured resident set (shared/block-residency.h) replaced
      // it. Growth fills the stack from here as the box allows.
      session()->info(fmt(
          "GenerateImageStage('{}'): Qwen-Image-Edit DiT streaming {} blocks; "
          "the resident set grows as the box allows", this->id(),
          qcfg.n_layers));
    }
    _release_scratch = bounded;
    _dit_bounded = bounded;
    if (bounded) { revise_dit_declaration_(dit_dir); }
    // Cache the load params + VAE base so a memory-driven free after a
    // generation (free_qie_dit_for_decode_) can reload identically. The
    // Config is default-constructed above and everything else comes from
    // the checkpoint's own config.json, so the dir and the streaming
    // decision are the whole of it.
    _qie_dit_dir = dit_dir;
    // The holding correct_dit_holding_ moves as residency grows.
    _dit_holding_dir = dit_dir;
    _qie_stream  = stream_blocks;
    // AutoencoderKLQwenImage, which is the SAME VAE Krea-2 runs (see
    // vae_family_ in vae-decode-stage.cc: anything that is not Flux2 /
    // Wan / MiniMax-H3 or a registered family opens as one). So it sizes from `base_dim`,
    // not from diffusers' block_out_channels -- reading it the FLUX.2 way
    // would have silently pinned the estimate to the 128 default on a
    // checkpoint whose base_dim is 96.
    _vae_base = krea2_vae_base_(root);
  } else if (_family == "qwen-image-21") {
    // A 7B DiT at ~14.2 GB bf16, beside a ~17.5 GB Qwen3-VL encoder.
    // The same plan_streaming() rule the other five families take --
    // not a sixth answer to one question.
    // VPIPE_QWEN_IMAGE21_STREAM overrides.
    const auto plan = model_memory::plan_streaming(
        session(), dit_dir, enc_dir, model_memory::kStreamHeadroom);
    // STREAM, THEN RETAIN on every box (stream_blocks_by_default), the
    // verdict on the box kept apart: `bounded` is what only a tight box
    // wants -- the load deferred past the conditioning, the declaration
    // revised to the trunk, the forward's scratch released. A streamed
    // block is the preloaded one, int8 layouts included (a dense source
    // quantized at the refill, a held w4 layout converted at promotion).
    bool stream_blocks = model_memory::stream_blocks_by_default(plan);
    if (const char* e = std::getenv("VPIPE_QWEN_IMAGE21_STREAM")) {
      stream_blocks = (std::atoi(e) != 0);
    }
    const bool bounded = plan.stream && stream_blocks;
    session()->log_debug(fmt(
        "GenerateImageStage('{}'): Qwen-Image-2.1 footprint {} GB (others {} "
        "GB) + {} GB headroom vs {} GB RAM -> {}", this->id(),
        plan.footprint >> 30, plan.others >> 30,
        model_memory::kStreamHeadroom >> 30, phys_ram() >> 30,
        !stream_blocks ? "PRELOAD"
        : bounded      ? "STREAM blocks"
                       : "STREAM blocks, kept as the box allows"));
    _qi21_dit_dir = dit_dir;
    // The holding correct_dit_holding_ moves as residency grows.
    _dit_holding_dir = dit_dir;
    _qi21_stream = stream_blocks;
    // HELD by the box's room, streamed or not: residency converts each
    // block it keeps into the layout (MetalQwenImage21Transformer::
    // clone_block_); a slot converts per GEMM either way.
    _i8_hold_w4 = hold_w4_int8_(plan, bounded, dit_dir);
    _qi21_ck = qi21_checkpoint_schedule(root);
    session()->info(fmt(
        "GenerateImageStage('{}'): Qwen-Image-2.1 schedule from the "
        "checkpoint: {}{}, {}", this->id(),
        _qi21_ck.sample_sigmas.empty()
            ? std::string("linspace grid")
            : fmt("{} sigma nodes (sample_sigmas)",
                  _qi21_ck.sample_sigmas.size())(),
        _qi21_ck.dynamic
            ? std::string(", per-image dynamic shift")
            : fmt(", static shift {}", _qi21_ck.shift)(),
        _qi21_ck.shift_terminal > 0.0
            ? fmt("terminal stretch to {}", _qi21_ck.shift_terminal)()
            : std::string("no terminal stretch")));
    // Its VAE is the Qwen-Image one generalized, so the same reader
    // serves it -- but the DECODER's base, which is 144 here against
    // the encoder's 96, because every peak this feeds is a decode.
    _vae_base = krea2_vae_dec_base_(root);
    _release_scratch = bounded;
    _dit_bounded = bounded;
    if (bounded) { revise_dit_declaration_(dit_dir); }
    if (bounded) {
      // Deferred for the same reason Boogu defers: both stages load in
      // initialize(), so a preloaded DiT beside a resident encoder is
      // the peak, and it is the peak that decides whether the box
      // copes. The DiT loads on the first conditioning beat, by which
      // time the conditioner has let its encoder go.
      _dit_unloaded = true;
      session()->info(fmt(
          "GenerateImageStage('{}'): memory-bounded -- the Qwen-Image-2.1 "
          "DiT loads on the first conditioning beat (block streaming on) "
          "and is freed for the vae-decode, so it never shares the box "
          "with the text encoder", this->id()));
    } else if (!load_qwen_image21_dit_()) {
      session()->error(fmt(
          "GenerateImageStage('{}'): failed to load the Qwen-Image-2.1 DiT "
          "from '{}'; inert", this->id(), dit_dir));
      return;
    }
  } else if (_family == "z-image") {
    // A 6B DiT beside a ~8 GB Qwen3-4B encoder. The same plan_streaming()
    // rule the other families take -- not a seventh answer to one
    // question. VPIPE_Z_IMAGE_STREAM overrides.
    //
    // NOTE the Turbo checkpoint is F32 on disk (24.6 GB against the
    // base's 12.3 GB bf16), so plan_streaming sees twice the footprint
    // for the same 6B of parameters -- which is the right answer for
    // the READ and pessimistic for what is held, since every block is
    // narrowed on the way in.
    const auto plan = model_memory::plan_streaming(
        session(), dit_dir, enc_dir, model_memory::kStreamHeadroom);
    // STREAM, THEN RETAIN; `bounded` is the tight box's.
    bool stream_blocks = model_memory::stream_blocks_by_default(plan);
    if (const char* e = std::getenv("VPIPE_Z_IMAGE_STREAM")) {
      stream_blocks = (std::atoi(e) != 0);
    }
    const bool bounded = plan.stream && stream_blocks;
    session()->log_debug(fmt(
        "GenerateImageStage('{}'): Z-Image footprint {} GB (others {} GB) + "
        "{} GB headroom vs {} GB RAM -> {}", this->id(),
        plan.footprint >> 30, plan.others >> 30,
        model_memory::kStreamHeadroom >> 30, phys_ram() >> 30,
        !stream_blocks ? "PRELOAD"
        : bounded      ? "STREAM blocks"
                       : "STREAM blocks, kept as the box allows"));
    _zi_dit_dir = dit_dir;
    // The holding correct_dit_holding_ moves as residency grows.
    _dit_holding_dir = dit_dir;
    _zi_stream = stream_blocks;
    // HELD by the box's room, streamed or not (see Qwen-Image-2.1's).
    _i8_hold_w4 = hold_w4_int8_(plan, bounded, dit_dir);
    {
      bool dyn = false;
      _zi_shift = genai::flow_shift_from_config(root, &dyn);
      if (dyn) {
        session()->warn(fmt(
            "GenerateImageStage('{}'): this Z-Image checkpoint asks for "
            "use_dynamic_shifting, which wants a per-image mu rather "
            "than the constant shift this family runs; using the "
            "constant", this->id()));
      }
      session()->log_debug(fmt(
          "GenerateImageStage('{}'): Z-Image flow shift {} from the "
          "checkpoint's scheduler_config", this->id(),
          _zi_shift > 0.0 ? std::to_string(_zi_shift)
                          : std::string("(absent, using 3.0)")));
    }
    // The plain diffusers AutoencoderKL -- the FLUX.1 VAE -- so it
    // sizes from block_out_channels like FLUX.2's does, not from a
    // Qwen-Image `base_dim`.
    _vae_base = flux2_vae_base_(root);
    _release_scratch = bounded;
    _dit_bounded = bounded;
    if (bounded) { revise_dit_declaration_(dit_dir); }
    if (bounded) {
      // Deferred for the same reason Boogu and 2.1 defer: both stages
      // load in initialize(), so a preloaded DiT beside a resident
      // encoder is the peak, and it is the peak that decides whether
      // the box copes.
      _dit_unloaded = true;
      session()->info(fmt(
          "GenerateImageStage('{}'): memory-bounded -- the Z-Image DiT "
          "loads on the first conditioning beat (block streaming on) and "
          "is freed for the vae-decode, so it never shares the box with "
          "the text encoder", this->id()));
    } else if (!load_z_image_dit_()) {
      session()->error(fmt(
          "GenerateImageStage('{}'): failed to load the Z-Image DiT from "
          "'{}'; inert", this->id(), dit_dir));
      return;
    }
  } else if (_family == "flux3") {
    // A 5.6B DiT (11 GB bf16) beside a Qwen3-VL-4B encoder (~8 GB). The
    // one plan_streaming() rule; VPIPE_FLUX3_STREAM overrides.
    const auto plan = model_memory::plan_streaming(
        session(), dit_dir, enc_dir, model_memory::kStreamHeadroom);
    // STREAM, THEN RETAIN; `bounded` is the tight box's.
    bool stream_blocks = model_memory::stream_blocks_by_default(plan);
    if (const char* e = std::getenv("VPIPE_FLUX3_STREAM")) {
      stream_blocks = (std::atoi(e) != 0);
    }
    const bool bounded = plan.stream && stream_blocks;
    session()->log_debug(fmt(
        "GenerateImageStage('{}'): FLUX.3 footprint {} GB (others {} GB) + "
        "{} GB headroom vs {} GB RAM -> {}", this->id(),
        plan.footprint >> 30, plan.others >> 30,
        model_memory::kStreamHeadroom >> 30, phys_ram() >> 30,
        !stream_blocks ? "PRELOAD"
        : bounded      ? "STREAM blocks"
                       : "STREAM blocks, kept as the box allows"));
    _flux3_dit_dir = dit_dir;
    _dit_holding_dir = dit_dir;
    _flux3_stream = stream_blocks;
    // THE SAMPLER THE CHECKPOINT STATES: <root>/config.json carries the
    // schedule's alpha and the distilled step count. A graph's own
    // scheduler-select or `steps` still wins on the count.
    {
      namespace fs = std::filesystem;
      std::ifstream in(fs::path(root) / "config.json");
      if (in) {
        try {
          FlexData fd = FlexData::from_json(in);
          if (fd.is_object() && fd.as_object().contains("sampler")) {
            const FlexData smp = fd.as_object().at("sampler");
            if (smp.is_object()) {
              auto so = smp.as_object();
              if (so.contains("alpha")) {
                _flux3_alpha = so.at("alpha").as_real(_flux3_alpha);
              }
              if (so.contains("steps")) {
                _flux3_steps = (int)so.at("steps").as_int(_flux3_steps);
              }
            }
          }
        } catch (...) {
        }
      }
    }
    // FLUX.2's VAE, unchanged (stored F32), so FLUX.2's decode peak.
    _vae_base = flux2_vae_base_(root);
    _release_scratch = bounded;
    _dit_bounded = bounded;
    if (bounded) { revise_dit_declaration_(dit_dir); }
    if (bounded) {
      // Deferred like the other streaming families: both stages load in
      // initialize(), and a DiT beside a resident encoder is the peak.
      _dit_unloaded = true;
      session()->info(fmt(
          "GenerateImageStage('{}'): memory-bounded -- the FLUX.3 DiT loads "
          "on the first conditioning beat (block streaming on) and is freed "
          "for the vae-decode, so it never shares the box with the text "
          "encoder", this->id()));
    } else if (!load_flux3_dit_()) {
      session()->error(fmt(
          "GenerateImageStage('{}'): failed to load the FLUX.3 DiT from "
          "'{}'; inert", this->id(), dit_dir));
      return;
    }
  } else if (_family == "fibo2") {
    // An 8B DiT (16 GB bf16) beside a Qwen3-VL-4B encoder (~9 GB). The
    // one plan_streaming() rule; VPIPE_FIBO2_STREAM overrides.
    const auto plan = model_memory::plan_streaming(
        session(), dit_dir, enc_dir, model_memory::kStreamHeadroom);
    // STREAM, THEN RETAIN; `bounded` is the tight box's.
    bool stream_blocks = model_memory::stream_blocks_by_default(plan);
    if (const char* e = std::getenv("VPIPE_FIBO2_STREAM")) {
      stream_blocks = (std::atoi(e) != 0);
    }
    const bool bounded = plan.stream && stream_blocks;
    session()->log_debug(fmt(
        "GenerateImageStage('{}'): Fibo 2.0 footprint {} GB (others {} GB) "
        "+ {} GB headroom vs {} GB RAM -> {}", this->id(),
        plan.footprint >> 30, plan.others >> 30,
        model_memory::kStreamHeadroom >> 30, phys_ram() >> 30,
        !stream_blocks ? "PRELOAD"
        : bounded      ? "STREAM blocks"
                       : "STREAM blocks, kept as the box allows"));
    _fibo2_dit_dir = dit_dir;
    _dit_holding_dir = dit_dir;
    _fibo2_stream = stream_blocks;
    // THE SCHEDULE THE CHECKPOINT STATES: the dynamic shift's ends.
    {
      namespace fs = std::filesystem;
      std::ifstream in(fs::path(root) / "scheduler" /
                       "scheduler_config.json");
      if (in) {
        try {
          FlexData fd = FlexData::from_json(in);
          if (fd.is_object()) {
            auto o = fd.as_object();
            auto getf = [&](const char* k, double d) {
              return o.contains(k) ? o.at(k).as_real(d) : d;
            };
            auto geti = [&](const char* k, int d) {
              return o.contains(k) ? (int)o.at(k).as_int(d) : d;
            };
            _fibo2_base_shift = getf("base_shift", _fibo2_base_shift);
            _fibo2_max_shift = getf("max_shift", _fibo2_max_shift);
            _fibo2_base_seq = geti("base_image_seq_len", _fibo2_base_seq);
            _fibo2_max_seq = geti("max_image_seq_len", _fibo2_max_seq);
          }
        } catch (...) {
        }
      }
    }
    // FLUX.2's VAE (AutoencoderKLFlux2), so FLUX.2's decode peak.
    _vae_base = flux2_vae_base_(root);
    _release_scratch = bounded;
    _dit_bounded = bounded;
    if (bounded) { revise_dit_declaration_(dit_dir); }
    if (bounded) {
      _dit_unloaded = true;
      session()->info(fmt(
          "GenerateImageStage('{}'): memory-bounded -- the Fibo 2.0 DiT "
          "loads on the first conditioning beat (block streaming on) and is "
          "freed for the vae-decode, so it never shares the box with the "
          "text encoder", this->id()));
    } else if (!load_fibo2_dit_()) {
      session()->error(fmt(
          "GenerateImageStage('{}'): failed to load the Fibo 2.0 DiT from "
          "'{}'; inert", this->id(), dit_dir));
      return;
    }
  } else if (_family == "moss-image") {
    // A 5B DiT (9.5 GB bf16) beside a Qwen3.5-9B encoder (~18 GB). The
    // one plan_streaming() rule; VPIPE_MOSS_IMAGE_STREAM overrides.
    const auto plan = model_memory::plan_streaming(
        session(), dit_dir, enc_dir, model_memory::kStreamHeadroom);
    // STREAM, THEN RETAIN; `bounded` is the tight box's.
    bool stream_blocks = model_memory::stream_blocks_by_default(plan);
    if (const char* e = std::getenv("VPIPE_MOSS_IMAGE_STREAM")) {
      stream_blocks = (std::atoi(e) != 0);
    }
    const bool bounded = plan.stream && stream_blocks;
    session()->log_debug(fmt(
        "GenerateImageStage('{}'): MOSS-Image footprint {} GB (others {} "
        "GB) + {} GB headroom vs {} GB RAM -> {}", this->id(),
        plan.footprint >> 30, plan.others >> 30,
        model_memory::kStreamHeadroom >> 30, phys_ram() >> 30,
        !stream_blocks ? "PRELOAD"
        : bounded      ? "STREAM blocks"
                       : "STREAM blocks, kept as the box allows"));
    _moss_dit_dir = dit_dir;
    _dit_holding_dir = dit_dir;
    _moss_stream = stream_blocks;
    // HELD by the box's room, streamed or not (see Qwen-Image-2.1's).
    _i8_hold_w4 = hold_w4_int8_(plan, bounded, dit_dir);
    // THE SAMPLER THE CHECKPOINT STATES: MOSSImageScheduler's shift,
    // training_shift and training grid.
    {
      namespace fs = std::filesystem;
      std::ifstream in(fs::path(root) / "scheduler" /
                       "scheduler_config.json");
      if (in) {
        try {
          FlexData fd = FlexData::from_json(in);
          if (fd.is_object()) {
            auto so = fd.as_object();
            if (so.contains("shift")) {
              _moss_shift = so.at("shift").as_real(_moss_shift);
            }
            if (so.contains("training_shift")) {
              _moss_train_shift =
                  so.at("training_shift").as_real(_moss_train_shift);
            }
            if (so.contains("num_train_timesteps")) {
              _moss_train_steps = (int)so.at("num_train_timesteps")
                                      .as_int(_moss_train_steps);
            }
          }
        } catch (...) {
        }
      }
    }
    // FLUX.2's VAE with the small decoder: sized from its decoder widths.
    _vae_base = flux2_vae_base_(root);
    _release_scratch = bounded;
    _dit_bounded = bounded;
    if (bounded) { revise_dit_declaration_(dit_dir); }
    if (bounded) {
      _dit_unloaded = true;
      session()->info(fmt(
          "GenerateImageStage('{}'): memory-bounded -- the MOSS-Image DiT "
          "loads on the first conditioning beat (block streaming on) and is "
          "freed for the vae-decode, so it never shares the box with the "
          "text encoder", this->id()));
    } else if (!load_moss_image_dit_()) {
      session()->error(fmt(
          "GenerateImageStage('{}'): failed to load the MOSS-Image DiT from "
          "'{}'; inert", this->id(), dit_dir));
      return;
    }
  } else if (_family == "boogu-image") {
    // Boogu's 10B NextDiT. At ~20 GB bf16 it streams on any box that cannot
    // hold it beside the resident Qwen3-VL mllm (~16 GB), which is every box
    // short of ~48 GB; the three refiner stacks stay resident either way.
    // VPIPE_BOOGU_STREAM / VPIPE_BOOGU_PIN_FRAC override.
    const auto plan = model_memory::plan_streaming(
        session(), dit_dir, enc_dir, model_memory::kStreamHeadroom);
    // STREAM, THEN RETAIN; `bounded` is the tight box's.
    bool stream_blocks = model_memory::stream_blocks_by_default(plan);
    if (const char* e = std::getenv("VPIPE_BOOGU_STREAM")) {
      stream_blocks = (std::atoi(e) != 0);
    }
    const bool bounded = plan.stream && stream_blocks;
    session()->log_debug(fmt(
        "GenerateImageStage('{}'): Boogu footprint {} GB (others {} GB) + {} "
        "GB headroom vs {} GB RAM -> {}", this->id(), plan.footprint >> 30,
        plan.others >> 30, model_memory::kStreamHeadroom >> 30,
        phys_ram() >> 30,
        !stream_blocks ? "PRELOAD"
        : bounded      ? "STREAM blocks"
                       : "STREAM blocks, kept as the box allows"));
    // Cache the load params so the stage can free the DiT for a big decode and
    // reload it on the next prompt (free_boogu_dit_for_decode_).
    _boogu_dit_dir = dit_dir;
    // The holding correct_dit_holding_ moves as residency grows.
    _dit_holding_dir = dit_dir;
    _boogu_stream = stream_blocks;
    _vae_base = flux2_vae_base_(root);   // Boogu's VAE is a plain AutoencoderKL
    _release_scratch = bounded;
    _dit_bounded = bounded;
    if (bounded) { revise_dit_declaration_(dit_dir); }
    if (bounded) {
      // DEFER the load on a bounded box. Streaming and per-prompt reload only
      // help if the DiT is not resident WHILE the conditioner's encoder is:
      // both stages load in initialize(), so a preloaded DiT plus a resident
      // Qwen3-VL mllm is the peak, and it is the peak that decides whether the
      // box copes (measured: 17.7 GB resident for a 4-bit Boogu when both load
      // up front, 8.7 GB when they take turns). The DiT loads on its first
      // conditioning beat -- by which time the conditioner has dropped the
      // encoder -- and is freed again for the vae-decode.
      _dit_unloaded = true;
      session()->info(fmt(
          "GenerateImageStage('{}'): memory-bounded -- the Boogu-Image DiT loads "
          "on the first conditioning beat (block streaming on) and is freed for "
          "the vae-decode, so it never shares the box with the text encoder",
          this->id()));
    } else if (!load_boogu_dit_()) {
      session()->error(fmt(
          "GenerateImageStage('{}'): failed to load the Boogu-Image DiT from "
          "'{}'; inert", this->id(), dit_dir));
      return;
    }
    } else {
    // Stream the 28 transformer blocks when the box cannot hold the DiT
    // beside the conditioner's encoder -- the same rule the other four
    // image families take, from the same plan_streaming(), rather than a
    // fifth answer to one question.
    //
    // KREA-2 WAS THE FAMILY THAT NEVER ASKED. It loaded unconditionally
    // and decided only whether to drop per-forward scratch, on the
    // reasoning below: its weights are mmapped and therefore evictable
    // under pressure. That holds for a QUANTIZED checkpoint. It does not
    // hold for raw bf16, which is read `Copied` like every other LM and
    // tower in this tree (see the WeightSet residency rule) -- so the 28
    // blocks land in dirty, non-reclaimable buffers and the box swaps
    // instead of evicting. MEASURED: ~7 GB of swap on a 24 GB M5 Pro
    // with the bf16 weights, where the plan says stream.
    // STREAM, THEN RETAIN; `bounded` is the tight box's.
    bool stream_blocks;
    bool bounded = false;
    {
      const auto plan = model_memory::plan_streaming(
          session(), dit_dir, enc_dir, model_memory::kStreamHeadroom);
      stream_blocks = model_memory::stream_blocks_by_default(plan);
      if (const char* e = std::getenv("VPIPE_KREA2_STREAM")) {
        stream_blocks = (std::atoi(e) != 0);
      }
      bounded = plan.stream && stream_blocks;
      _dit_bounded = bounded;
      session()->log_debug(fmt(
          "GenerateImageStage('{}'): Krea-2 footprint {} GB (others {} GB) "
          "+ {} GB headroom vs {} GB RAM -> {}", this->id(),
          plan.footprint >> 30, plan.others >> 30,
          model_memory::kStreamHeadroom >> 30, phys_ram() >> 30,
          !stream_blocks ? "PRELOAD"
          : bounded      ? "STREAM blocks"
                         : "STREAM blocks, kept as the box allows"));
      // HELD by the box's room, streamed or not (see Qwen-Image-2.1's).
      _i8_hold_w4 = hold_w4_int8_(plan, bounded, dit_dir);
    }
    genai::MetalKrea2Transformer::Config kcfg;
    kcfg.i8_gemm = _i8_gemm;
    kcfg.i8_tier = _i8_tier;
    kcfg.i8_hold_w4 = _i8_hold_w4;
    kcfg.sage = _sage;
    kcfg.sol  = _sol;
    apply_ane_(kcfg);
    {
      using Dit = genai::MetalKrea2Transformer;
      const auto loras = lora_specs_<Dit::LoraSpec>();
      _dit = take_warm_dit_<Dit>(kWarmKrea2, dit_dir,
                                 warm_key_(kcfg, stream_blocks, loras),
                                 "Krea-2");
      if (!_dit) {
        _dit = Dit::load(weight_set_(dit_dir), mc, kcfg, stream_blocks,
                         loras);
      }
    }
    if (!_dit) {
      session()->error(fmt(
          "GenerateImageStage('{}'): failed to load the DiT from '{}'; inert",
          this->id(), dit_dir));
      return;
    }
    // Cache the resolved dir (reload after a decode-driven free) + the VAE base
    // channels for the decode-peak estimate (shared MetalKrea2Vae -> base_dim).
    _krea2_dit_dir = dit_dir;
    // The holding correct_dit_holding_ moves as residency grows.
    _dit_holding_dir = dit_dir;
    _krea2_stream  = stream_blocks;
    _vae_base = krea2_vae_base_(root);
    if (bounded) { revise_dit_declaration_(dit_dir); }
    // Drop the DiT's per-forward scratch after each generation when the
    // box cannot hold encoder + DiT + a large VAE decode at once, so its
    // working set does not crowd out the downstream vae-decode.
    //
    // Kept as its OWN question rather than folded into the streaming
    // one: releasing scratch is the milder measure and answers to the
    // smaller kHeadroom, so a box that is tight without being tight
    // enough to stream still gets it -- which is what this stage did
    // before streaming was wired in, and there is no reason to take it
    // away.
    const std::vector<std::string> peers = {dit_dir, enc_dir};
    _release_scratch =
        bounded ||
        model_memory::bounded(session(), peers, model_memory::kHeadroom);
    session()->log_debug(fmt(
        "GenerateImageStage('{}'): Krea-2 scratch: footprint {} GB + {} GB "
        "headroom vs {} GB RAM -> {}", this->id(),
        model_memory::weight_footprint(session(), peers) >> 30,
        model_memory::kHeadroom >> 30, phys_ram() >> 30,
        _release_scratch ? "RELEASE" : "keep"));
  }
  // A TIER ASKED FOR THAT WILL NOT HAPPEN, said once, here -- where the
  // family is finally known and before a minute of denoising.
  //
  // This is the failure mode the acceleration bag has and the reason it
  // is worth a line: a tier is a key in an untyped object, so asking a
  // family for one it does not implement is not a type error, not a load
  // error and not visible in the picture. It is a run that is slower
  // than the operator believes it is. Only sol_attn is named because it
  // is the only one none of the built-ins takes; i8_gemm and sage_attn
  // report their own non-engagement from inside the models that do.
  //
  // Reached only on the BUILT-IN path -- the registered-family branch
  // returns above -- so a plugin that implements Sol is never told its
  // own tier is inert.
  // NOT FOR EVERY FAMILY. A family that implements the tier would
  // otherwise be told its attention ran dense at the moment it is
  // routing -- a log that CONTRADICTS the run, which is worse than no
  // log. The rest still ignore the key, and a graph that set it
  // deserves to know the attention it asked to approximate ran dense.
  //
  // KEPT AS ONE NAMED PREDICATE because the list has now gone stale
  // twice in this file: it read {krea2, flux2} while Qwen-Image-2.1
  // and Z-Image both dispatched Sol, so both were warned that a tier
  // they were actively running did not exist. The check that catches
  // it is `grep -c "_sol->encode"` over the transformers -- the
  // dispatch, not the declaration.
  if (_sol.enabled && !family_implements_sol_(_family)) {
    session()->warn(fmt(
        "GenerateImageStage('{}'): sol_attn is set and the built-in '{}' "
        "denoiser does not implement it, so the attention runs dense. It "
        "reaches a REGISTERED image family through the acceleration bag",
        this->id(), _family));
  }
  // The int8 FAST tier, the same way: one named family takes it (its
  // model logs that it did, or why not); the rest run the accurate tier,
  // and a graph that asked for fast deserves to know which it got.
  if (_i8_gemm && _i8_tier != 0 && _family != "qwen-image-21" &&
      _family != "krea2" && _family != "flux2" && _family != "z-image" &&
      _family != "moss-image" && _family != "fibo2") {
    session()->warn(fmt(
        "GenerateImageStage('{}'): i8_gemm_tier is '{}' and the built-in "
        "'{}' denoiser implements only the accurate tier, which it runs",
        this->id(),
        _i8_tier == 1 ? "fast" : _i8_tier == 2 ? "token" : "token_plain",
        _family));
  }
  session()->log_debug(fmt(
      "GenerateImageStage('{}'): {} DiT ready{}",
      this->id(), _family,
      _strength > 0.0 ? "; img2img init latent expected on a ref_latent iport"
                      : ""));
}

// The config names a MODEL (a registry key), a directory or a file;
// this is the one place that turns any of those into the single
// .safetensors a loader opens. A key beats a directory scan because
// several adapters of one repo land side by side, and only the record
// says which one the key meant.
//
// A failure WARNS and returns empty rather than failing the graph: an
// unreadable adapter should cost the adapter, not the run -- and the
// image that comes back un-adapted is obvious, where a stage that
// refused to start is a mystery.
std::string
GenerateImageStage::adapter_file_(const std::string& ref) const
{
  if (ref.empty()) { return {}; }
  std::string lerr;
  std::string f = resolve_adapter_file(session(), ref, &lerr);
  if (f.empty()) {
    session()->warn(fmt(
        "GenerateImageStage('{}'): {} -- generating WITHOUT the adapter",
        this->id(), lerr));
    return f;
  }
  // SAID OUT LOUD, on the UI delegate, beside the "loading <family> DiT
  // from ..." line. An adapter that never arrives is otherwise
  // indistinguishable from one that was never configured: the config
  // source reports what it EMITTED, and if that beat is wired somewhere
  // that does not read the key -- a krea2-model-config on the
  // conditioner alone, say -- nothing downstream says a word. This line
  // is the one place a user can see that the key got here.
  session()->info(fmt(
      "GenerateImageStage('{}'): runtime LoRA '{}'", this->id(), f));
  return f;
}

bool
GenerateImageStage::load_flux2_dit_()
{
  auto* mc = session() ? session()->services()->metal_compute() : nullptr;
  if (mc == nullptr || _flux2_dit_dir.empty()) { return false; }
  genai::MetalFlux2Transformer::Config fcfg;
  fcfg.i8_gemm = _i8_gemm;
  fcfg.i8_tier = _i8_tier;
  fcfg.i8_hold_w4 = _i8_hold_w4;
  fcfg.sage = _sage;
  // The RELOAD path too. A tier set here and not there comes back
  // missing on the generation after a decode-driven free, which is the
  // quietest way to lose it.
  fcfg.sol  = _sol;
  _flux2_params.apply_to(fcfg);
  // The streaming flag the first load used, and the adapter with it. A
  // reload that dropped either would come back preloaded, or
  // un-adapted, on the graph that asked for both.
  using Dit = genai::MetalFlux2Transformer;
  const auto loras = lora_specs_<Dit::LoraSpec>();
  _flux2_dit = take_warm_dit_<Dit>(kWarmFlux2, _flux2_dit_dir,
                                   warm_key_(fcfg, _flux2_stream, loras),
                                   "FLUX.2");
  if (!_flux2_dit) {
    _flux2_dit = Dit::load(weight_set_(_flux2_dit_dir), mc, fcfg,
                           _flux2_stream, loras);
  }
  return (bool)_flux2_dit;
}

// State the decode arena this image implies, before deciding anything
// about it.
//
// The plan could only bound this, and for an EDIT graph it could not
// even do that: the output geometry comes from the reference image, so
// there is no height/width in any config to estimate from and
// declare_resources() correctly declares nothing. This is where the
// number becomes known -- the same expression the reclaim check below
// uses, so the ledger and the decision cannot drift apart.
//
// Published BEFORE the budget guards, and unconditionally: whether THIS
// stage needs to free its DiT is a different question from how much
// vae-decode is about to allocate, and a roomy box must still put the
// arena on the books for whoever sizes next.
//
// A REVISION, so declare_resources must have claimed the label -- with
// kUnknownArena when it had no geometry to size from. That is what
// keeps the plan authoritative about what exists.
void
GenerateImageStage::publish_decode_arena_(std::size_t peak) const
{
  if (peak == 0 || session() == nullptr ||
      session()->services() == nullptr) {
    return;
  }
  if (auto* mgr = session()->services()->generative_model_manager()) {
    mgr->revise_scratch("vae-decode", peak);
  }
}

void
GenerateImageStage::free_flux2_dit_for_decode_(int gen_w, int gen_h)
{
  if (!_flux2_dit || gen_w <= 0 || gen_h <= 0) { return; }
  const std::size_t peak = flux2_decode_peak_(gen_w, gen_h, _vae_base);
  publish_decode_arena_(peak);
  auto* mc = session() ? session()->services()->metal_compute() : nullptr;
  if (mc == nullptr) { return; }
  const auto mb = mc->memory_budget();
  const int forced = forced_dit_reclaim_();
  if (forced == 0) { return; }           // operator says never
  if (forced < 0) {
    if (mb.recommended == 0) { return; }   // budget query unavailable
    // Free if EITHER budget is short: fits() (our Metal working set)
    // misses pressure from OTHER apps' resident memory; fits_physical()
    // (host-wide reclaimable RAM) catches it, so external pressure
    // triggers the reclaim instead of the decode later failing its own
    // physical backstop.
    if (mb.fits(peak) && mb.fits_physical(peak)) { return; }
  }
  // The DiT is idle now (its latent is read back). Free the ~9 GB of mmap'd
  // weights so the downstream vae-decode stage's working set has room; the DiT
  // reloads lazily when the next prompt arrives. Done BEFORE the latent is
  // published, so the decode sees the freed working set.
  // The numbers are reported either way, but a FORCED free must not
  // read as a measured one -- "13072 MB headroom < 162 MB needed" is a
  // comparison that never happened, and chasing it wastes the next
  // reader's afternoon.
  session()->info(fmt(
      "GenerateImageStage('{}'): freeing the FLUX.2 DiT for the {}x{} "
      "vae-decode (~{} MB): {} MB working-set headroom, ~{} MB "
      "reclaimable{}; reloads on the next prompt", this->id(),
      gen_w, gen_h, peak >> 20, mb.headroom >> 20,
      mb.available_physical >> 20,
      forced > 0 ? " -- FORCED by VPIPE_IMAGE_DIT_RECLAIM" : ""));
  // SETTLED WITH THE MANAGER, which owns the checkpoint: the reset below
  // only ends this stage's borrow. The order against it used to be the
  // whole point -- the manager's reference was weak, so there was
  // nothing left to pool afterwards -- and is now free either way.
  //
  // This free is the pool's best case. The DiT is being given up to make
  // room for a decode that may or may not actually need it: pooled, the
  // pages stay purgeable, so if the decode does need them
  // reclaim_at_least() spends the pool first and they go -- and if it
  // does not, the next prompt recycles the checkpoint instead of
  // re-reading it. Dropping outright took the reload every time and
  // could not tell the two cases apart.
  if (auto* mgr = session()->services()->generative_model_manager()) {
    mgr->pool_weights(_flux2_dit_dir);
  }
  _flux2_dit.reset();
  _dit_unloaded = true;
}

// The manager's shared, reference-counted view of a checkpoint. Two
// pipelines running the same model get the same set and share its
// weights; the DiT holds the ONLY reference from this stage, deliberately
// -- free_*_dit_for_decode_ has to actually free, and it would not if the
// stage kept a handle of its own.
std::shared_ptr<genai::WeightSet>
GenerateImageStage::weight_set_(const std::string& dir) const
{
  if (dir.empty()) { return nullptr; }
  // Falls back to a private set when the session has no manager; see
  // open_weight_set().
  return genai::open_weight_set(dir, session());
}

bool
GenerateImageStage::load_vosr_dit_()
{
  auto* mc = session() ? session()->services()->metal_compute() : nullptr;
  if (mc == nullptr || _vosr_dir.empty()) { return false; }
  // No streaming decision: at 1.4B the restorer is 2.8 GB bf16, small
  // enough that the block-streaming machinery would cost more than it
  // saves on any box that can run the VAE at all. The conditioner's
  // DINOv2 is another 0.6 GB and is dropped between beats by its own
  // idle policy.
  // The four accelerations, from the same stage keys every other family
  // here reads. The restorer is head_dim 64 where the rest are 128,
  // which is what Sol had to be generalised for.
  genai::MetalVosrTransformer::Config vcfg = _vosr_cfg;
  vcfg.i8_gemm = _i8_gemm;
  vcfg.sage    = _sage;
  vcfg.sol     = _sol;
  std::string err;
  _vosr = genai::MetalVosrTransformer::load(weight_set_(_vosr_dir), mc,
                                            vcfg, &err);
  if (!_vosr && !err.empty()) {
    session()->error(fmt("GenerateImageStage('{}'): VOSR restorer: {}",
                         this->id(), err));
  }
  return (bool)_vosr;
}

bool
GenerateImageStage::load_qwen_image21_dit_()
{
  auto* mc = session() ? session()->services()->metal_compute() : nullptr;
  if (mc == nullptr || _qi21_dit_dir.empty()) { return false; }
  genai::MetalQwenImage21Transformer::Config cfg;
  if (!genai::MetalQwenImage21Transformer::Config::read_dims(_qi21_dit_dir,
                                                             &cfg)) {
    session()->warn(fmt(
        "GenerateImageStage('{}'): '{}' does not hold a "
        "QwenImage21Transformer2DModel config", this->id(), _qi21_dit_dir));
    return false;
  }
  cfg.i8_gemm = _i8_gemm;
  cfg.i8_tier = _i8_tier;
  cfg.i8_hold_w4 = _i8_hold_w4;
  cfg.sage = _sage;
  cfg.sol = _sol;
  cfg.ane_ffn = genai::accel::flag(&_accel, genai::accel::kAneFfn);
  cfg.ane_qkv = genai::accel::flag(&_accel, genai::accel::kAneQkv);
  using Dit = genai::MetalQwenImage21Transformer;
  if (cfg.ane_ffn &&
      !ane_granted_(
          Dit::ane_runtime_bytes(cfg, Dit::ane_plan_seq(_width, _height)),
          "Qwen-Image-2.1")) {
    cfg.ane_ffn = false;
    cfg.ane_qkv = false;
  }
  cfg.ane_rows = (int)genai::accel::integer(&_accel, genai::accel::kAneRows,
                                            0);
  cfg.ane_layers = (int)genai::accel::integer(&_accel,
                                              genai::accel::kAneLayers, 0);
  const auto loras = lora_specs_<Dit::LoraSpec>();
  _qi21_dit = take_warm_dit_<Dit>(kWarmQi21, _qi21_dit_dir,
                                  warm_key_(cfg, _qi21_stream, loras),
                                  "Qwen-Image-2.1");
  if (_qi21_dit) { return true; }
  _qi21_dit = genai::MetalQwenImage21Transformer::load(
      weight_set_(_qi21_dit_dir), mc, cfg, _qi21_stream, loras);
  if (_qi21_dit && _qi21_stream) {
    session()->info(fmt(
        "GenerateImageStage('{}'): Qwen-Image-2.1 DiT streaming {} blocks; "
        "the resident set grows as the box allows", this->id(),
        _qi21_dit->config().n_layers));
  }
  return (bool)_qi21_dit;
}

void
GenerateImageStage::free_qwen_image21_dit_for_decode_(int gen_w, int gen_h)
{
  if (!_qi21_dit) { return; }
  auto* mc = session() ? session()->services()->metal_compute() : nullptr;
  if (mc == nullptr) { return; }
  // The VAE decode that follows is a /16 one, so its peak is a quarter
  // of what the same picture costs an /8 family -- the latent grid is
  // half in each axis. Sized from the checkpoint either way.
  const std::size_t peak =
      qwen_decode_peak_(gen_w, gen_h, _vae_base > 0 ? _vae_base : 96,
                        /*residual=*/true);
  const std::size_t budget = mc->memory_budget().available_physical;
  if (peak <= budget) { return; }
  session()->info(fmt(
      "GenerateImageStage('{}'): freeing the Qwen-Image-2.1 DiT before a "
      "{}x{} decode ({} MB wanted, {} MB free)", this->id(), gen_w, gen_h,
      peak >> 20, budget >> 20));
  publish_decode_arena_(peak);
  if (auto* mgr = session()->services()->generative_model_manager()) {
    mgr->pool_weights(_qi21_dit_dir);
  }
  _qi21_dit.reset();
  _dit_unloaded = true;
}

bool
GenerateImageStage::load_z_image_dit_()
{
  auto* mc = session() ? session()->services()->metal_compute() : nullptr;
  if (mc == nullptr || _zi_dit_dir.empty()) { return false; }
  genai::MetalZImageTransformer::Config cfg;
  if (!genai::MetalZImageTransformer::Config::read_dims(_zi_dit_dir, &cfg)) {
    session()->warn(fmt(
        "GenerateImageStage('{}'): '{}' does not hold a "
        "ZImageTransformer2DModel config", this->id(), _zi_dit_dir));
    return false;
  }
  cfg.i8_gemm = _i8_gemm;
  cfg.i8_tier = _i8_tier;
  cfg.i8_hold_w4 = _i8_hold_w4;
  cfg.sage = _sage;
  cfg.sol = _sol;
  cfg.ane_ffn = genai::accel::flag(&_accel, genai::accel::kAneFfn);
  cfg.ane_qkv = genai::accel::flag(&_accel, genai::accel::kAneQkv);
  using Dit = genai::MetalZImageTransformer;
  if (cfg.ane_ffn &&
      !ane_granted_(
          Dit::ane_runtime_bytes(cfg, Dit::ane_plan_seq(_width, _height)),
          "Z-Image")) {
    cfg.ane_ffn = false;
    cfg.ane_qkv = false;
  }
  cfg.ane_rows = (int)genai::accel::integer(&_accel, genai::accel::kAneRows,
                                            0);
  cfg.ane_layers = (int)genai::accel::integer(&_accel,
                                              genai::accel::kAneLayers, 0);
  const auto loras = lora_specs_<Dit::LoraSpec>();
  _zi_dit = take_warm_dit_<Dit>(kWarmZImage, _zi_dit_dir,
                                warm_key_(cfg, _zi_stream, loras),
                                "Z-Image");
  if (_zi_dit) { return true; }
  _zi_dit = Dit::load(weight_set_(_zi_dit_dir), mc, cfg, _zi_stream, loras);
  if (_zi_dit && _zi_stream) {
    session()->info(fmt(
        "GenerateImageStage('{}'): Z-Image DiT streaming {} of {} blocks; "
        "the resident set grows as the box allows{}", this->id(),
        _zi_dit->config().n_mod_blocks(), _zi_dit->config().n_blocks(),
        _zi_dit->narrows_f32()
            ? " (the checkpoint is F32, so every block is narrowed to "
              "bf16 on the way in -- twice the bytes to read, half the "
              "bytes to hold)"
            : ""));
  }
  return (bool)_zi_dit;
}

void
GenerateImageStage::free_z_image_dit_for_decode_(int gen_w, int gen_h)
{
  if (!_zi_dit) { return; }
  auto* mc = session() ? session()->services()->metal_compute() : nullptr;
  if (mc == nullptr) { return; }
  // The plain AutoencoderKL that FLUX.1 and Boogu use, so the same peak.
  const std::size_t peak =
      flux2_decode_peak_(gen_w, gen_h, _vae_base > 0 ? _vae_base : 128);
  // SETTLED: the last forward's scratch (8 GB at 4096^2) is still on its
  // way back to the system when this runs.
  const std::size_t budget =
      mc->memory_budget_settled().available_physical;
  if (peak <= budget) { return; }
  session()->info(fmt(
      "GenerateImageStage('{}'): freeing the Z-Image DiT before a {}x{} "
      "decode ({} MB wanted, {} MB free)", this->id(), gen_w, gen_h,
      peak >> 20, budget >> 20));
  publish_decode_arena_(peak);
  if (auto* mgr = session()->services()->generative_model_manager()) {
    mgr->pool_weights(_zi_dit_dir);
  }
  _zi_dit.reset();
  _dit_unloaded = true;
}

bool
GenerateImageStage::load_flux3_dit_()
{
  auto* mc = session() ? session()->services()->metal_compute() : nullptr;
  if (mc == nullptr || _flux3_dit_dir.empty()) { return false; }
  genai::MetalFlux3Transformer::Config cfg;
  if (!genai::MetalFlux3Transformer::Config::read_dims(_flux3_dit_dir,
                                                       &cfg)) {
    session()->warn(fmt(
        "GenerateImageStage('{}'): '{}' does not hold a "
        "Flux3Transformer2DModel config this build runs", this->id(),
        _flux3_dit_dir));
    return false;
  }
  cfg.i8_gemm = _i8_gemm;
  cfg.sage = _sage;
  cfg.sol = _sol;
  cfg.ane_ffn = genai::accel::flag(&_accel, genai::accel::kAneFfn);
  cfg.ane_qkv = genai::accel::flag(&_accel, genai::accel::kAneQkv);
  using Dit = genai::MetalFlux3Transformer;
  if (cfg.ane_ffn &&
      !ane_granted_(
          Dit::ane_runtime_bytes(cfg, Dit::ane_plan_seq(_width, _height)),
          "FLUX.3")) {
    cfg.ane_ffn = false;
    cfg.ane_qkv = false;
  }
  cfg.ane_rows = (int)genai::accel::integer(&_accel, genai::accel::kAneRows,
                                            0);
  cfg.ane_layers = (int)genai::accel::integer(&_accel,
                                              genai::accel::kAneLayers, 0);
  const auto loras = lora_specs_<Dit::LoraSpec>();
  _flux3_dit = take_warm_dit_<Dit>(kWarmFlux3, _flux3_dit_dir,
                                   warm_key_(cfg, _flux3_stream, loras),
                                   "FLUX.3");
  if (_flux3_dit) { return true; }
  _flux3_dit = Dit::load(weight_set_(_flux3_dit_dir), mc, cfg, _flux3_stream,
                         loras);
  if (_flux3_dit && _flux3_stream) {
    session()->info(fmt(
        "GenerateImageStage('{}'): FLUX.3 DiT streaming its {} blocks; the "
        "resident set grows as the box allows", this->id(),
        _flux3_dit->n_blocks()));
  }
  return (bool)_flux3_dit;
}

void
GenerateImageStage::free_flux3_dit_for_decode_(int gen_w, int gen_h)
{
  if (!_flux3_dit) { return; }
  auto* mc = session() ? session()->services()->metal_compute() : nullptr;
  if (mc == nullptr) { return; }
  const std::size_t peak =
      flux2_decode_peak_(gen_w, gen_h, _vae_base > 0 ? _vae_base : 128);
  // SETTLED: the last forward's scratch (8 GB at 4096^2) is still on its
  // way back to the system when this runs.
  const std::size_t budget =
      mc->memory_budget_settled().available_physical;
  if (peak <= budget) { return; }
  session()->info(fmt(
      "GenerateImageStage('{}'): freeing the FLUX.3 DiT before a {}x{} "
      "decode ({} MB wanted, {} MB free)", this->id(), gen_w, gen_h,
      peak >> 20, budget >> 20));
  publish_decode_arena_(peak);
  if (auto* mgr = session()->services()->generative_model_manager()) {
    mgr->pool_weights(_flux3_dit_dir);
  }
  _flux3_dit.reset();
  _dit_unloaded = true;
}

bool
GenerateImageStage::load_fibo2_dit_()
{
  auto* mc = session() ? session()->services()->metal_compute() : nullptr;
  if (mc == nullptr || _fibo2_dit_dir.empty()) { return false; }
  genai::MetalFibo2Transformer::Config cfg;
  if (!genai::MetalFibo2Transformer::Config::read_dims(_fibo2_dit_dir,
                                                       &cfg)) {
    session()->warn(fmt(
        "GenerateImageStage('{}'): '{}' does not hold a "
        "BriaFibo2Transformer2DModel config this build runs", this->id(),
        _fibo2_dit_dir));
    return false;
  }
  cfg.i8_gemm = _i8_gemm;
  cfg.i8_tier = _i8_tier;
  cfg.i8_hold_w4 = _i8_hold_w4;
  cfg.sage = _sage;
  cfg.sol = _sol;
  cfg.ane_ffn = genai::accel::flag(&_accel, genai::accel::kAneFfn);
  cfg.ane_qkv = genai::accel::flag(&_accel, genai::accel::kAneQkv);
  using Dit = genai::MetalFibo2Transformer;
  if (cfg.ane_ffn &&
      !ane_granted_(
          Dit::ane_runtime_bytes(cfg, Dit::ane_plan_seq(_width, _height)),
          "Fibo 2.0")) {
    cfg.ane_ffn = false;
    cfg.ane_qkv = false;
  }
  cfg.ane_rows = (int)genai::accel::integer(&_accel, genai::accel::kAneRows,
                                            0);
  cfg.ane_layers = (int)genai::accel::integer(&_accel,
                                              genai::accel::kAneLayers, 0);
  const auto loras = lora_specs_<Dit::LoraSpec>();
  _fibo2_dit = take_warm_dit_<Dit>(kWarmFibo2, _fibo2_dit_dir,
                                   warm_key_(cfg, _fibo2_stream, loras),
                                   "Fibo 2.0");
  if (_fibo2_dit) { return true; }
  _fibo2_dit = Dit::load(weight_set_(_fibo2_dit_dir), mc, cfg, _fibo2_stream,
                         loras);
  if (_fibo2_dit && _fibo2_stream) {
    session()->info(fmt(
        "GenerateImageStage('{}'): Fibo 2.0 DiT streaming its {} blocks; "
        "the resident set grows as the box allows", this->id(),
        _fibo2_dit->config().n_blocks()));
  }
  return (bool)_fibo2_dit;
}

void
GenerateImageStage::free_fibo2_dit_for_decode_(int gen_w, int gen_h)
{
  if (!_fibo2_dit) { return; }
  auto* mc = session() ? session()->services()->metal_compute() : nullptr;
  if (mc == nullptr) { return; }
  const std::size_t peak =
      flux2_decode_peak_(gen_w, gen_h, _vae_base > 0 ? _vae_base : 128);
  // SETTLED: the last forward's scratch is still on its way back to the
  // system when this runs.
  const std::size_t budget =
      mc->memory_budget_settled().available_physical;
  if (peak <= budget) { return; }
  session()->info(fmt(
      "GenerateImageStage('{}'): freeing the Fibo 2.0 DiT before a {}x{} "
      "decode ({} MB wanted, {} MB free)", this->id(), gen_w, gen_h,
      peak >> 20, budget >> 20));
  publish_decode_arena_(peak);
  if (auto* mgr = session()->services()->generative_model_manager()) {
    mgr->pool_weights(_fibo2_dit_dir);
  }
  _fibo2_dit.reset();
  _dit_unloaded = true;
}

bool
GenerateImageStage::load_moss_image_dit_()
{
  auto* mc = session() ? session()->services()->metal_compute() : nullptr;
  if (mc == nullptr || _moss_dit_dir.empty()) { return false; }
  genai::MetalMossImageTransformer::Config cfg;
  if (!genai::MetalMossImageTransformer::Config::read_dims(_moss_dit_dir,
                                                          &cfg)) {
    session()->warn(fmt(
        "GenerateImageStage('{}'): '{}' does not hold a MOSSImageTransformer "
        "config this build runs", this->id(), _moss_dit_dir));
    return false;
  }
  cfg.i8_gemm = _i8_gemm;
  cfg.i8_tier = _i8_tier;
  cfg.i8_hold_w4 = _i8_hold_w4;
  cfg.sage = _sage;
  cfg.sol = _sol;
  cfg.ane_ffn = genai::accel::flag(&_accel, genai::accel::kAneFfn);
  cfg.ane_qkv = genai::accel::flag(&_accel, genai::accel::kAneQkv);
  using Dit = genai::MetalMossImageTransformer;
  if (cfg.ane_ffn &&
      !ane_granted_(
          Dit::ane_runtime_bytes(cfg, Dit::ane_plan_seq(_width, _height)),
          "MOSS-Image")) {
    cfg.ane_ffn = false;
    cfg.ane_qkv = false;
  }
  cfg.ane_rows = (int)genai::accel::integer(&_accel, genai::accel::kAneRows,
                                            0);
  cfg.ane_layers = (int)genai::accel::integer(&_accel,
                                              genai::accel::kAneLayers, 0);
  const auto loras = lora_specs_<Dit::LoraSpec>();
  _moss_dit = take_warm_dit_<Dit>(kWarmMoss, _moss_dit_dir,
                                  warm_key_(cfg, _moss_stream, loras),
                                  "MOSS-Image");
  if (_moss_dit) { return true; }
  _moss_dit = Dit::load(weight_set_(_moss_dit_dir), mc, cfg, _moss_stream,
                        loras);
  if (_moss_dit && _moss_stream) {
    session()->info(fmt(
        "GenerateImageStage('{}'): MOSS-Image DiT streaming its {} blocks; "
        "the resident set grows as the box allows", this->id(),
        _moss_dit->config().n_layers));
  }
  return (bool)_moss_dit;
}

void
GenerateImageStage::free_moss_image_dit_for_decode_(int gen_w, int gen_h)
{
  if (!_moss_dit) { return; }
  auto* mc = session() ? session()->services()->metal_compute() : nullptr;
  if (mc == nullptr) { return; }
  const std::size_t peak =
      flux2_decode_peak_(gen_w, gen_h, _vae_base > 0 ? _vae_base : 128);
  // VPIPE_IMAGE_DIT_RECLAIM forces (1) or suppresses (0) the free; unset,
  // the budget decides -- read SETTLED, since the last forward's scratch
  // is still on its way back.
  const int forced = forced_dit_reclaim_();
  if (forced == 0) { return; }
  const std::size_t budget =
      mc->memory_budget_settled().available_physical;
  if (forced < 0 && peak <= budget) { return; }
  session()->info(fmt(
      "GenerateImageStage('{}'): freeing the MOSS-Image DiT before a {}x{} "
      "decode ({} MB wanted, {} MB free{})", this->id(), gen_w, gen_h,
      peak >> 20, budget >> 20, forced > 0 ? ", FORCED" : ""));
  publish_decode_arena_(peak);
  if (auto* mgr = session()->services()->generative_model_manager()) {
    mgr->pool_weights(_moss_dit_dir);
  }
  _moss_dit.reset();
  _dit_unloaded = true;
}

bool
GenerateImageStage::load_boogu_dit_()
{
  auto* mc = session() ? session()->services()->metal_compute() : nullptr;
  if (mc == nullptr || _boogu_dit_dir.empty()) { return false; }
  genai::MetalBooguTransformer::Config bcfg;
  bcfg.i8_gemm = _i8_gemm;
  // NOT boogu, deliberately. Its attention pads head_dim 120 to the
  // bd128 kernel and memoizes a specialised function per DISTINCT
  // sequence length -- the refiners, the image self-attention and the
  // joint attention all run at different ones -- so Sage there is an
  // int8 twin per length and a prologue per (length, block), against
  // padded operands. That is a different piece of work from the three
  // families above, not a fifth copy of it.
  _boogu_dit = genai::MetalBooguTransformer::load(weight_set_(_boogu_dit_dir),
                                                 mc, bcfg, _boogu_stream);
  if (_boogu_dit && _boogu_stream) {
    // No prefix any more: the pinned-prefix policy was a fraction of
    // TOTAL RAM decided before the run and blind to the machine, and
    // the measured resident set (shared/block-residency.h) replaced
    // it. Growth fills the stack from here as the box allows.
    session()->info(fmt(
        "GenerateImageStage('{}'): Boogu DiT streaming {} blocks; the resident "
        "set grows as the box allows", this->id(),
        _boogu_dit->config().n_double + _boogu_dit->config().n_single));
  }
  return (bool)_boogu_dit;
}

void
GenerateImageStage::free_boogu_dit_for_decode_(int gen_w, int gen_h)
{
  if (!_boogu_dit || gen_w <= 0 || gen_h <= 0) { return; }
  // Boogu's VAE is the plain AutoencoderKL through MetalFlux2Vae.
  const std::size_t peak = flux2_decode_peak_(gen_w, gen_h, _vae_base);
  publish_decode_arena_(peak);
  auto* mc = session() ? session()->services()->metal_compute() : nullptr;
  if (mc == nullptr) { return; }
  const auto mb = mc->memory_budget();
  const int forced = forced_dit_reclaim_();
  if (forced == 0) { return; }           // operator says never
  if (forced < 0) {
    if (mb.recommended == 0) { return; }   // budget query unavailable
  }
  // Forced (1) means free regardless; suppressed (0) returned above.
  const bool decode_runs_beside_us =
      forced < 0 && mb.fits(peak) && mb.fits_physical(peak);
  // Tell the DiT how much to leave clear when it decides whether to keep
  // a streamed block. Growth must not eat the arena the decode needs --
  // that trades a disk read for an allocation failure. ZERO when the DiT
  // is about to be freed for the decode: reserving room for a
  // coexistence that does not happen costs the whole denoise, and
  // BlockResidency reads a zero reserve as an ANSWER rather than as
  // never having been told.
  _boogu_dit->set_residency_reserve(decode_runs_beside_us ? peak : 0);
  if (decode_runs_beside_us) { return; }
  // The numbers are reported either way, but a FORCED free must not
  // read as a measured one -- "13072 MB headroom < 162 MB needed" is a
  // comparison that never happened, and chasing it wastes the next
  // reader's afternoon.
  session()->info(fmt(
      "GenerateImageStage('{}'): freeing the Boogu-Image DiT for the {}x{} "
      "vae-decode (~{} MB): {} MB working-set headroom, ~{} MB "
      "reclaimable{}; reloads on the next prompt", this->id(),
      gen_w, gen_h, peak >> 20, mb.headroom >> 20,
      mb.available_physical >> 20,
      forced > 0 ? " -- FORCED by VPIPE_IMAGE_DIT_RECLAIM" : ""));
  // SETTLED WITH THE MANAGER, which owns the checkpoint: the reset below
  // only ends this stage's borrow. The order against it used to be the
  // whole point -- the manager's reference was weak, so there was
  // nothing left to pool afterwards -- and is now free either way.
  //
  // This free is the pool's best case. The DiT is being given up to make
  // room for a decode that may or may not actually need it: pooled, the
  // pages stay purgeable, so if the decode does need them
  // reclaim_at_least() spends the pool first and they go -- and if it
  // does not, the next prompt recycles the checkpoint instead of
  // re-reading it. Dropping outright took the reload every time and
  // could not tell the two cases apart.
  if (auto* mgr = session()->services()->generative_model_manager()) {
    mgr->pool_weights(_boogu_dit_dir);
  }
  _boogu_dit.reset();
  _dit_unloaded = true;
}

bool
GenerateImageStage::load_qie_dit_()
{
  auto* mc = session() ? session()->services()->metal_compute() : nullptr;
  if (mc == nullptr || _qie_dit_dir.empty()) { return false; }
  // Default-constructed, exactly as the first load was: everything that
  // varies (quantization, zero_cond_t, the block count) is read from the
  // checkpoint's config.json inside load().
  //
  // THE ACCELERATION SETTINGS BELONG HERE TOO. This is the RELOAD path --
  // free_qie_dit_for_decode_ drops the DiT so a large VAE decode fits,
  // and the next generation lands here. A config built fresh without
  // them means the first image of a run is accelerated and every one
  // after it is not, which reads as the setting working and then
  // quietly stopping.
  genai::MetalQwenImageTransformer::Config qcfg;
  qcfg.sage    = _sage;
  qcfg.i8_gemm = _i8_gemm;
  _qie_dit = genai::MetalQwenImageTransformer::load(
      weight_set_(_qie_dit_dir), mc, qcfg, _qie_stream);
  return (bool)_qie_dit;
}

void
GenerateImageStage::free_qie_dit_for_decode_(int gen_w, int gen_h)
{
  if (!_qie_dit || gen_w <= 0 || gen_h <= 0 || _vae_base <= 0) { return; }
  const std::size_t peak = qwen_decode_peak_(gen_w, gen_h, _vae_base);
  publish_decode_arena_(peak);
  auto* mc = session() ? session()->services()->metal_compute() : nullptr;
  if (mc == nullptr) { return; }
  const auto mb = mc->memory_budget();
  const int forced = forced_dit_reclaim_();
  if (forced == 0) { return; }           // operator says never
  if (forced < 0) {
    if (mb.recommended == 0) { return; }   // budget query unavailable
    // Free if EITHER budget is short: fits() (our Metal working set)
    // misses pressure from OTHER apps' resident memory; fits_physical()
    // (host-wide reclaimable RAM) catches it, so external pressure
    // triggers the reclaim instead of the decode later failing its own
    // physical backstop.
    if (mb.fits(peak) && mb.fits_physical(peak)) { return; }
  }
  // The numbers are reported either way, but a FORCED free must not
  // read as a measured one -- "13072 MB headroom < 162 MB needed" is a
  // comparison that never happened, and chasing it wastes the next
  // reader's afternoon.
  session()->info(fmt(
      "GenerateImageStage('{}'): freeing the Qwen-Image-Edit DiT for the {}x{} "
      "vae-decode (~{} MB): {} MB working-set headroom, ~{} MB "
      "reclaimable{}; reloads on the next prompt", this->id(),
      gen_w, gen_h, peak >> 20, mb.headroom >> 20,
      mb.available_physical >> 20,
      forced > 0 ? " -- FORCED by VPIPE_IMAGE_DIT_RECLAIM" : ""));
  // SETTLED WITH THE MANAGER, which owns the checkpoint: the reset below
  // only ends this stage's borrow. The order against it used to be the
  // whole point -- the manager's reference was weak, so there was
  // nothing left to pool afterwards -- and is now free either way.
  //
  // This free is the pool's best case. The DiT is being given up to make
  // room for a decode that may or may not actually need it: pooled, the
  // pages stay purgeable, so if the decode does need them
  // reclaim_at_least() spends the pool first and they go -- and if it
  // does not, the next prompt recycles the checkpoint instead of
  // re-reading 20B of weights. Dropping outright took the reload every
  // time and could not tell the two cases apart.
  //
  // The destructor below is also what gives this model's share of the
  // WIRED POOL back (see MetalQwenImageTransformer::~...): freeing a
  // wired buffer unwires it in the kernel but does not decrement the
  // pool's counter, so a per-prompt free that skipped it would leak the
  // whole share every prompt until nothing could wire at all.
  if (auto* mgr = session()->services()->generative_model_manager()) {
    mgr->pool_weights(_qie_dit_dir);
  }
  _qie_dit.reset();
  _dit_unloaded = true;
}

bool
GenerateImageStage::load_krea2_dit_()
{
  auto* mc = session() ? session()->services()->metal_compute() : nullptr;
  if (mc == nullptr || _krea2_dit_dir.empty()) { return false; }
  genai::MetalKrea2Transformer::Config kcfg;
  kcfg.i8_gemm = _i8_gemm;
  kcfg.i8_tier = _i8_tier;
  kcfg.i8_hold_w4 = _i8_hold_w4;
  kcfg.sage = _sage;
  apply_ane_(kcfg);
  // The streaming flag the first load used, and the adapter with it. A
  // reload that dropped either would come back preloaded, or un-adapted,
  // on the graph that asked for both.
  using Dit = genai::MetalKrea2Transformer;
  const auto loras = lora_specs_<Dit::LoraSpec>();
  _dit = take_warm_dit_<Dit>(kWarmKrea2, _krea2_dit_dir,
                             warm_key_(kcfg, _krea2_stream, loras),
                             "Krea-2");
  if (!_dit) {
    _dit = Dit::load(weight_set_(_krea2_dit_dir), mc, kcfg, _krea2_stream,
                     loras);
  }
  return (bool)_dit;
}

void
GenerateImageStage::free_krea2_dit_for_decode_(int gen_w, int gen_h)
{
  if (!_dit || gen_w <= 0 || gen_h <= 0) { return; }
  // Checked AFTER release_forward_scratch(), so the weights are only
  // freed when dropping the scratch was not enough.
  const std::size_t peak = qwen_decode_peak_(gen_w, gen_h, _vae_base);
  publish_decode_arena_(peak);
  auto* mc = session() ? session()->services()->metal_compute() : nullptr;
  if (mc == nullptr) { return; }
  const auto mb = mc->memory_budget();
  const int forced = forced_dit_reclaim_();
  if (forced == 0) { return; }           // operator says never
  if (forced < 0) {
    if (mb.recommended == 0) { return; }   // budget query unavailable
    // Free if EITHER budget is short. fits() (our Metal working set)
    // misses pressure from OTHER apps' resident memory; fits_physical()
    // (host-wide reclaimable RAM) catches it -- otherwise external
    // pressure leaves the DiT resident and the decode just fails its own
    // physical backstop instead.
    if (mb.fits(peak) && mb.fits_physical(peak)) { return; }
  }
  // The numbers are reported either way, but a FORCED free must not
  // read as a measured one -- "13072 MB headroom < 162 MB needed" is a
  // comparison that never happened, and chasing it wastes the next
  // reader's afternoon.
  session()->info(fmt(
      "GenerateImageStage('{}'): freeing the Krea-2 DiT for the {}x{} "
      "vae-decode (~{} MB): {} MB working-set headroom, ~{} MB "
      "reclaimable{}; reloads on the next prompt", this->id(),
      gen_w, gen_h, peak >> 20, mb.headroom >> 20,
      mb.available_physical >> 20,
      forced > 0 ? " -- FORCED by VPIPE_IMAGE_DIT_RECLAIM" : ""));
  // SETTLED WITH THE MANAGER, which owns the checkpoint: the reset below
  // only ends this stage's borrow. The order against it used to be the
  // whole point -- the manager's reference was weak, so there was
  // nothing left to pool afterwards -- and is now free either way.
  //
  // This free is the pool's best case. The DiT is being given up to make
  // room for a decode that may or may not actually need it: pooled, the
  // pages stay purgeable, so if the decode does need them
  // reclaim_at_least() spends the pool first and they go -- and if it
  // does not, the next prompt recycles the checkpoint instead of
  // re-reading it. Dropping outright took the reload every time and
  // could not tell the two cases apart.
  if (auto* mgr = session()->services()->generative_model_manager()) {
    mgr->pool_weights(_krea2_dit_dir);
  }
  _dit.reset();
  _dit_unloaded = true;
}

namespace {

// Copy a conditioning TensorBeat (2 bytes/elt, f16 or bf16) into a metal
// SharedBuffer for the DiT. The element type is family-fixed and interpreted by
// the DiT (f16 for krea2/flux2, bf16 for qwen-image-edit); here it is opaque
// bytes. Empty on allocation failure.
metal_compute::SharedBuffer
cond_to_shared_(metal_compute::MetalCompute* mc, const TensorBeatPayload& tb)
{
  const auto bytes = tb.materialize_contiguous();
  metal_compute::SharedBuffer b = mc->make_shared_buffer(bytes.size());
  if (!b.empty()) { std::memcpy(b.contents(), bytes.data(), bytes.size()); }
  return b;
}

}  // namespace

namespace {

// A live preview's CLEAN ESTIMATE, x0 = x - sigma v, from the LAST model
// evaluation of a step the previewer renders (for a multi-evaluation
// sampler, the one at the lowest noise). Every flow family here returns
// its velocity in FlowSampler's convention, so one tap serves them all.
// Nothing is built on a step that is not armed.
struct X0Tap {
  bool want = false;
  std::vector<float> x0;
  void
  arm(bool due)
  {
    want = due;
    x0.clear();
  }
  void
  take(const std::vector<float>& x, double sigma,
       const std::vector<float>& v)
  {
    if (!want || v.size() != x.size()) { return; }
    x0.resize(x.size());
    for (std::size_t k = 0; k < x.size(); ++k) {
      x0[k] = x[k] - (float)sigma * v[k];
    }
  }
  bool ready() const noexcept { return want && !x0.empty(); }
};

}  // namespace

std::vector<float>
GenerateImageStage::generate_(const metal_compute::SharedBuffer& cond, int n_real,
                            const metal_compute::SharedBuffer& cond_neg,
                            int n_real_neg, int gen_h, int gen_w,
                            const std::vector<float>* init_packed,
                            const std::vector<float>* init_latent,
                            const std::vector<RefLatent>& refs,
                            const std::function<void(const std::vector<float>&)>&
                                emit_step) const
{
  auto* mc = session()->services()->metal_compute();
  using metal_compute::SharedBuffer;
  const int H = gen_h, W = gen_w;
  const int lh = H / 8, lw = W / 8;          // latent H/W
  const int gh = H / 16, gw = W / 16;        // 2x2-patch grid
  const int img_seq = gh * gw;
  const int IC = 64;                         // z_dim(16) * patch(2) * patch(2)

  // How much room the streamed DiT must leave clear to keep a block
  // resident. Krea-2 has carried the residency machinery -- measurement,
  // eviction, ratchet -- since it was written, and never grew a single
  // block, because nothing ever set this: a reserve left at its default
  // means "this model was never taught what its activations cost", and
  // growth stays off. Wired here so the policy is live rather than
  // decorative.
  //
  // Same rule as FLUX.2, against Krea-2's own decode peak
  // (free_krea2_dit_for_decode_): reserve the VAE decode only when it
  // would actually run beside us. When it would not, that free drops the
  // whole DiT first, which is strictly more than a reserve could hold
  // clear -- so reserving it there protects a coexistence that never
  // happens and costs every block of every step.
  if (_vae_base > 0 && _dit) {
    const std::size_t peak = qwen_decode_peak_(W, H, _vae_base);
    const auto mb = mc->memory_budget();
    // No budget to read -> the free bails out too, so the DiT survives.
    const bool decode_runs_beside_us =
        mb.recommended == 0 || (mb.fits(peak) && mb.fits_physical(peak));
    // Plus the ANE module when the plan granted it: CoreML holds its staged
    // weights beside every forward, and a block kept resident into that
    // memory is the same failure as one kept into the decode.
    std::size_t ane = 0;
    if (genai::accel::flag(&_accel, genai::accel::kAneFfn)) {
      genai::MetalKrea2Transformer::Config kc;
      kc.ane_qkv = genai::accel::flag(&_accel, genai::accel::kAneQkv);
      const std::size_t bytes =
          genai::MetalKrea2Transformer::ane_runtime_bytes(
              kc, genai::MetalKrea2Transformer::ane_plan_seq(_width, _height));
      if (model_memory::coreml_grant(session(), ane_claim_label_(), bytes,
                                     1) > 0) {
        ane = bytes;
      }
    }
    // ...and Sol-Attn's scratch, for the same reason: what it could not
    // carve from the buffers the forward lends it is memory a resident
    // block must not grow into.
    const std::size_t sol_b = _dit->sol_scratch_bytes(
        genai::MetalKrea2Transformer::ane_plan_seq(_width, _height));
    _dit->set_residency_reserve((decode_runs_beside_us ? peak : 0) + ane +
                                sol_b);
    // And the RATES for this schedule. Both of BlockResidency's defaults
    // are tuned for a ~30-step run: on a short one the probe would reach
    // full residency only near the end, where nothing is left to use it.
    // After the reserve, because the probe is sized against what is left
    // once the reserve is taken.
    _dit->set_residency_schedule(
        _scheduler_spec.steps > 0 ? _scheduler_spec.steps : 1);
    session()->log_debug(fmt(
        "GenerateImageStage('{}'): Krea-2 residency reserve {} MB -- the "
        "{}x{} vae-decode wants {} MB and {}",
        this->id(), (decode_runs_beside_us ? peak : 0) >> 20, W, H,
        peak >> 20,
        decode_runs_beside_us ? "fits beside the DiT"
                              : "does not, so the DiT is freed before it"));
  }

  // The conditioner emits the 12-tap f16 conditioning [n_real, 12, EH]; the
  // DiT's text-fusion tower fuses it into the DiT-facing text [n_real, hidden].
  SharedBuffer fused = _dit->forward_text(cond, n_real);
  if (fused.empty()) { return {}; }
  session()->log_debug(fmt(
      "GenerateImageStage('{}'): fused conditioning [{}, hidden]; "
      "{}x{} grid {}x{} img_seq {}", this->id(), n_real, W, H, gh, gw,
      img_seq));

  // Classifier-free guidance: fuse the negative conditioning too and, at each
  // denoise step, push the velocity away from it. Skipped when no negative
  // conditioning is wired or the scale is a no-op (1.0), so the single-pass
  // turbo default stays token-exact.
  bool cfg = !cond_neg.empty() && n_real_neg > 0 && _guidance_scale != 1.0;
  SharedBuffer fused_neg;
  if (cfg) {
    fused_neg = _dit->forward_text(cond_neg, n_real_neg);
    if (fused_neg.empty()) {
      session()->warn(fmt(
          "GenerateImageStage('{}'): negative-conditioning fuse failed; running "
          "without classifier-free guidance", this->id()));
      cfg = false;
    } else {
      session()->info(fmt(
          "GenerateImageStage('{}'): CFG on, scale {}, negative [{} rows]",
          this->id(), (float)_guidance_scale, n_real_neg));
    }
  }

  // 6. build the sampler (integrator) + scheduler (sigma schedule) from the
  //    active specs (config defaults or latched select-stage beats). The
  //    distilled turbo default (euler + simple / steps / shift 1.15 exp) is the
  //    token-exact schedule.
  genai::FlowSampler sampler(_sampler_spec, _scheduler_spec);
  const int S = sampler.steps();
  const std::vector<double>& sig = sampler.sigmas();

  // img2img start step: run only the tail (t_start .. S). t2i => start 0.
  int start = 0;
  if (_strength > 0.0) {
    const double init_ts = std::min((double)S * _strength, (double)S);
    start = (int)std::max((double)S - init_ts, 0.0);
    if (start >= S) { start = S - 1; }
  }
  const double sig0 = sig[(std::size_t)start];
  session()->log_debug(fmt(
      "GenerateImageStage('{}'): sampler={} scheduler={} steps={} start={} "
      "sig0={} (strength {})", this->id(), _sampler_spec.method,
      _scheduler_spec.type, S, start, (float)sig0, _strength));

  // 7. initial packed latents [img_seq, 64].
  std::vector<float> packed((std::size_t)img_seq * IC);
  if (init_packed != nullptr && init_packed->size() == packed.size()) {
    packed = *init_packed;                       // supplied (repro / golden)
  } else if (init_latent != nullptr && _strength > 0.0 &&
             init_latent->size() == (std::size_t)16 * lh * lw) {
    // img2img: the whitened latent [16, lh, lw] arrives from the vae-encode
    // stage on the `latent` iport. Mix it with noise at sig0 (scale_noise),
    // then pack.
    const float* lp = init_latent->data();
    std::mt19937_64 rng(_seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> mixed((std::size_t)16 * lh * lw);
    for (std::size_t i = 0; i < mixed.size(); ++i) {
      const float n = nd(rng);
      mixed[i] = (float)sig0 * n + (float)(1.0 - sig0) * lp[i];
    }
    // pack [16, lh, lw] -> [img_seq, 64] (inverse of the unpack below).
    for (int c = 0; c < 16; ++c) {
      for (int y = 0; y < lh; ++y) {
        for (int xx = 0; xx < lw; ++xx) {
          const int a = y / 2, ph = y % 2, b = xx / 2, pw = xx % 2;
          const std::size_t t = (std::size_t)a * gw + b;
          packed[t * IC + (std::size_t)c * 4 + ph * 2 + pw] =
              mixed[((std::size_t)c * lh + y) * lw + xx];
        }
      }
    }
  } else {
    std::mt19937_64 rng(_seed);                  // text-to-image: pure noise
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : packed) { v = nd(rng); }
  }

  // 8. sampler denoising loop (from `start` for img2img). The interchangeable
  //    integrator drives the DiT via this denoise callback: upload the
  //    candidate latent, run the DiT at the given sigma, read back the velocity.
  SharedBuffer latbuf = mc->make_shared_buffer((std::size_t)img_seq * IC * 2);
  if (latbuf.empty()) { return {}; }
  // Reference-image conditioning (built once, constant across denoise steps).
  // A ref latent arrives channel-first [16, rlh, rlw] (vae-encode output); pack
  // it 2x2 into token-major [rseq, 64] (the same patchify the generated latent
  // uses) so the DiT embeds it via img_in. Each reference lands in its own RoPE
  // frame band inside the DiT. Odd-dim or non-16-channel refs are skipped.
  std::vector<genai::MetalKrea2Transformer::RefImage> ri;
  for (const auto& r : refs) {
    if (r.empty() || r.c != 16 || (r.h % 2) != 0 || (r.w % 2) != 0) {
      if (!r.empty()) {
        session()->warn(fmt(
            "GenerateImageStage('{}'): krea2 reference latent [{}, {}, {}] must be "
            "16-channel with even H/W; ignoring", this->id(), r.c, r.h, r.w));
      }
      continue;
    }
    const int rgh = r.h / 2, rgw = r.w / 2, rseq = rgh * rgw;
    SharedBuffer rb = mc->make_shared_buffer((std::size_t)rseq * IC * 2);
    if (rb.empty()) { continue; }
    auto* d = static_cast<_Float16*>(rb.contents());
    std::memset(d, 0, rb.byte_size());
    for (int cc = 0; cc < 16; ++cc) {
      for (int y = 0; y < r.h; ++y) {
        for (int x = 0; x < r.w; ++x) {
          const int a = y / 2, ph = y % 2, bcol = x / 2, pw = x % 2;
          const std::size_t t = (std::size_t)a * rgw + bcol;
          d[t * IC + (std::size_t)cc * 4 + ph * 2 + pw] =
              (_Float16)r.chw[((std::size_t)cc * r.h + y) * r.w + x];
        }
      }
    }
    genai::MetalKrea2Transformer::RefImage img;
    img.latents = std::move(rb);
    img.seq = rseq; img.grid_h = rgh; img.grid_w = rgw;
    ri.push_back(std::move(img));
  }
  if (!ri.empty()) {
    session()->info(fmt(
        "GenerateImageStage('{}'): Krea-2 conditioning on {} reference image(s)",
        this->id(), ri.size()));
  }
  bool dit_ok = true;
  const float gscale = (float)_guidance_scale;
  // Opened before the denoise callable so the per-block hook is
  // live for the very first forward.
  UiProgress bar = session()->open_progress("denoise");
  DenoiseProgress prog(&bar, S - start, cfg ? 2 : 1);
  ScopedBlockProgress<std::remove_reference_t<decltype(*_dit)>>
      prog_guard(_dit.get(), prog);
  // Once per generation, after the first forward -- which is where the ANE
  // tier arms or declines -- set the module booking to what is HELD: the
  // bytes at this generation's sequence when armed, 0 when it declined
  // (emitter unverified, nothing eligible). Both ways, so a declined size
  // followed by one that arms does not leave the ledger at zero.
  // revise_scratch() refuses to create, so this is a no-op when the plan
  // booked nothing.
  bool ane_checked = false;
  // A live preview's clean estimate, packed like the state; unpacked
  // once the step is done.
  X0Tap tap;
  auto denoise = [&](const std::vector<float>& cand,
                     double sigma) -> std::vector<float> {
    auto* lb = static_cast<_Float16*>(latbuf.contents());
    for (std::size_t k = 0; k < cand.size(); ++k) { lb[k] = (_Float16)cand[k]; }
    SharedBuffer vel = _dit->forward_dit(fused, n_real, latbuf, img_seq, gh, gw,
                                         (float)sigma, -1, ri);
    prog.end_forward();
    if (!ane_checked) {
      ane_checked = true;
      if (_dit->ane_attempted()) {
        if (auto* mgr = session()->services()->generative_model_manager()) {
          genai::MetalKrea2Transformer::Config kc;
          kc.ane_qkv = _dit->ane_qkv_armed();
          // Either tier holding means the booking stands. When only the
          // qkv one armed this still counts the feed-forward's term,
          // which errs HIGH on purpose: under-booking hands memory the
          // modules are holding to somebody else, and that is the
          // failure worth avoiding.
          mgr->revise_scratch(
              "coreml:" + ane_claim_label_(),
              (_dit->ane_armed() || _dit->ane_qkv_armed())
                  ? genai::MetalKrea2Transformer::ane_runtime_bytes(
                        kc, n_real + img_seq)
                  : 0);
        }
      }
    }
    if (vel.empty()) { dit_ok = false; return {}; }
    const auto* vp = static_cast<const _Float16*>(vel.contents());
    std::vector<float> v(cand.size());
    for (std::size_t k = 0; k < v.size(); ++k) { v[k] = (float)vp[k]; }
    if (cfg) {
      // Second DiT pass on the SAME candidate (latbuf unchanged), conditioned
      // on the negative prompt: v = v_neg + scale*(v_pos - v_neg).
      SharedBuffer veln = _dit->forward_dit(fused_neg, n_real_neg, latbuf,
                                            img_seq, gh, gw, (float)sigma, -1,
                                            ri);
      prog.end_forward();
      if (veln.empty()) { dit_ok = false; return {}; }
      const auto* np = static_cast<const _Float16*>(veln.contents());
      for (std::size_t k = 0; k < v.size(); ++k) {
        const float vneg = (float)np[k];
        v[k] = vneg + gscale * (v[k] - vneg);
      }
    }
    tap.take(cand, sigma, v);
    return v;
  };
  sampler.reset();   // clear multistep history / reseed the SDE RNG for this run
  const bool prof = std::getenv("VPIPE_KREA2_PROFILE") != nullptr;
  double dit_ms = 0.0;
  int dit_calls = 0;
  auto denoise_p = [&](const std::vector<float>& cand, double sigma) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<float> v = denoise(cand, sigma);
    dit_ms += std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t0).count();
    ++dit_calls;
    return v;
  };
  // Denoising is the slow part (several seconds of DiT forwards). Paint an
  // in-place progress bar on the user-facing stream (model-quantize style)
  // and time the whole loop so the operator sees steady progress instead
  // of a silent stall, plus a wall-clock summary when the latent is done.
  // unpack packed [img_seq, 64] -> channel-first latent [16, lh, lw] (2x2). Used
  // per-step (debug step_latents) and for the final return.
  auto unpack = [&](const std::vector<float>& pk) {
    std::vector<float> latent((std::size_t)16 * lh * lw);
    for (int c = 0; c < 16; ++c) {
      for (int y = 0; y < lh; ++y) {
        for (int xx = 0; xx < lw; ++xx) {
          const int a = y / 2, ph = y % 2, b = xx / 2, pw = xx % 2;
          const std::size_t t = (std::size_t)a * gw + b;
          latent[((std::size_t)c * lh + y) * lw + xx] =
              pk[t * IC + (std::size_t)c * 4 + ph * 2 + pw];
        }
      }
    }
    return latent;
  };
  const int nsteps = S - start;
  const auto gen_t0 = std::chrono::steady_clock::now();
  for (int i = start; i < S; ++i) {
    session()->log_debug(fmt(
        "GenerateImageStage('{}'): denoise step {}/{} sigma {}", this->id(),
        i + 1, S, (float)sig[(std::size_t)i]));
    tap.arm(_preview && _preview->due(i - start + 1, nsteps));
    sampler.step(i, packed,
                 prof ? genai::FlowSampler::DenoiseFn(denoise_p) : denoise);
    if (!dit_ok) { return {}; }
    if (emit_step) { emit_step(unpack(packed)); }
    if (tap.ready()) {
      // One latent frame: [16, lh, lw] is [16, 1, lh, lw].
      _preview->submit(unpack(tap.x0), 16, 1, lh, lw, i - start + 1,
                       nsteps);
    }
    prog.end_step(i - start);
  }
  const double gen_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - gen_t0).count();
  session()->info(fmt(
      "GenerateImageStage('{}'): latent generated in {:.2f}s ({} denoise "
      "steps, {} ms/step)", this->id(), gen_s, nsteps,
      nsteps ? (long)(gen_s * 1000.0 / nsteps) : 0));
  if (prof) {
    session()->log_normal(fmt(
        "GenerateImageStage('{}'): DiT {} forward_dit calls, {} ms total, {} "
        "ms/call (seq {}+{}={})", this->id(), dit_calls, (long)dit_ms,
        dit_calls ? (long)(dit_ms / dit_calls) : 0, n_real, img_seq,
        n_real + img_seq));
  }

  // 9. unpack [img_seq, 64] -> channel-first latent [16, lh, lw].
  return unpack(packed);
}

std::vector<float>
GenerateImageStage::generate_flux2_(const metal_compute::SharedBuffer& context,
                                  int n_real, int gen_h, int gen_w,
                                  const std::vector<RefLatent>& refs,
                                  const std::vector<float>* init_packed,
                                  const std::function<void(
                                      const std::vector<float>&)>& emit_step)
    const
{
  auto* mc = session()->services()->metal_compute();
  using metal_compute::SharedBuffer;
  const int H = gen_h, W = gen_w;
  const int gh = H / 16, gw = W / 16;        // VAE latent grid (128ch @ H/16)
  const int img_seq = gh * gw;
  const int IC = _flux2_dit->config().in_channels;   // 128
  // How much room the DiT must leave clear when it decides whether to keep
  // a streamed block resident. Its own activations are allocated before
  // the budget it grows against is read, so what is left to protect is the
  // VAE DECODE that runs after the denoise -- but ONLY where the decode
  // runs beside us.
  //
  // When it does not fit beside us, free_flux2_dit_for_decode_ drops the
  // whole ~9 GB DiT before the latent is published. It re-asks with the
  // grown set resident, and it frees strictly more than any reserve could
  // have held clear, so the decode cannot be starved by growth. Reserving
  // its peak on that path protects a coexistence that never happens, and
  // pays for it by streaming every block of every step.
  //
  // So reserve it only where it is real -- where the decode WOULD fit
  // beside us, and growth is the one thing that could tip us into a free
  // we did not need (which costs a reload on the next prompt, not a
  // failure). Same predicate as the free, measured rather than assumed;
  // the post-denoise check stays the authority on a borderline box.
  if (_vae_base > 0) {
    const std::size_t peak = flux2_decode_peak_(W, H, _vae_base);
    const auto mb = mc->memory_budget();
    // No budget to read -> the free bails out too, so the DiT survives.
    const bool decode_runs_beside_us =
        mb.recommended == 0 || (mb.fits(peak) && mb.fits_physical(peak));
    // Plus Sol-Attn's scratch, for the same reason the decode peak is
    // here: what it could not carve from the two buffers the forward
    // lends it is memory a promoted block must not grow into.
    //
    // REFERENCE TOKENS COUNT TOWARD IT. With klein_kv off -- the better-
    // quality reference path, and the only reference geometry Sol
    // actually routes -- the references sit IN the joint sequence
    // (forward_dit: seq = text + generated + refs), so sizing this on
    // the generated image alone under-books exactly where the tier is
    // used. Counted with the SAME filter the pack loop below applies, so
    // the figure tracks the sequence that is really built rather than an
    // upper bound on it. Only the prompt half is allowed for rather than
    // measured -- a few hundred rows against thousands -- because
    // under-booking is the failure that matters.
    int flux2_ref_rows = 0;
    for (const auto& r : refs) {
      if (!r.empty() && r.c == IC) { flux2_ref_rows += r.h * r.w; }
    }
    const std::size_t flux2_sol =
        _flux2_dit->sol_scratch_bytes(img_seq + flux2_ref_rows + 512);
    _flux2_dit->set_residency_reserve(
        (decode_runs_beside_us ? peak : 0) + flux2_sol);
    // And the RATES for this schedule. Both of BlockResidency's defaults
    // are tuned for a ~30-step run: on a short one the probe would reach
    // full residency only near the end, where nothing is left to use it.
    // After the reserve, because the probe is sized against what is left
    // once the reserve is taken.
    _flux2_dit->set_residency_schedule(
        _scheduler_spec.steps > 0 ? _scheduler_spec.steps : 1);
    session()->log_debug(fmt(
        "GenerateImageStage('{}'): FLUX.2 residency reserve {} MB -- the "
        "{}x{} vae-decode wants {} MB and {}",
        this->id(), (decode_runs_beside_us ? peak : 0) >> 20, W, H,
        peak >> 20,
        decode_runs_beside_us ? "fits beside the DiT"
                              : "does not, so the DiT is freed before it"));
  }
  // `context` is the diffusion-conditioner's flux2 conditioning: the {9,18,27}
  // encoder taps concatenated per token -> f16 [n_real, joint_dim=3*enc_hidden].

  // Sampler (FlowMatchEuler; klein distilled -> no CFG). FLUX.2's flow-shift
  // (mu) is resolution- AND step-dependent (compute_empirical_mu): a fixed
  // shift is only right near the base resolution and washes the latent to grey
  // elsewhere (e.g. 1024). Recompute mu from the image token count + steps and
  // override the scheduler shift (unless a scheduler-select beat explicitly set
  // a non-default shift, which the operator then owns).
  genai::FlowSchedulerSpec sched = _scheduler_spec;
  if (!_scheduler_latched) {
    sched.shift = flux2_empirical_mu_(img_seq, sched.steps);
    session()->log_debug(fmt(
        "GenerateImageStage('{}'): FLUX.2 flow-shift mu = {} (img_seq {}, {} "
        "steps)", this->id(), (float)sched.shift, img_seq, sched.steps));
  }
  genai::FlowSampler sampler(_sampler_spec, sched);
  const int S = sampler.steps();

  // Initial packed latents [img_seq, IC] (pure noise; the VAE already
  // patchified, so no extra 2x2 packing here).
  std::vector<float> packed((std::size_t)img_seq * IC);
  {
    std::mt19937_64 rng(_seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : packed) { v = nd(rng); }
  }
  if (init_packed != nullptr && init_packed->size() == packed.size()) {
    packed = *init_packed;                       // supplied (repro / golden)
  }
  SharedBuffer latbuf = mc->make_shared_buffer((std::size_t)img_seq * IC * 2);
  if (latbuf.empty()) { return {}; }
  // Embedded guidance (guidance-distilled variants, e.g. base FLUX.2 dev/pro):
  // when the DiT has a guidance_embedder AND the user raises guidance_scale
  // above 1, feed guidance = guidance_scale so it is embedded into the timestep
  // modulation (single forward pass, the base pipeline's semantics). The klein
  // reference passes guidance=None, so the default (guidance_scale 1.0) -> -1
  // reproduces the distilled golden exactly.
  const float guid = (_flux2_dit->config().guidance_embeds &&
                      _guidance_scale > 1.0)
                         ? (float)_guidance_scale
                         : -1.0f;
  if (guid >= 0.0f) {
    session()->info(fmt(
        "GenerateImageStage('{}'): FLUX.2 embedded guidance scale {}",
        this->id(), guid));
  }
  // Reference-image conditioning (built once, constant across denoise steps):
  // patchify-pack each vae-encode latent [IC, rh, rw] into a token-major [rh*rw,
  // IC] f16 buffer. The DiT appends these tokens to the image stream and gives
  // each its own RoPE T offset. Non-IC-channel or empty refs are skipped.
  std::vector<genai::MetalFlux2Transformer::RefImage> ri;
  for (const auto& r : refs) {
    if (r.empty()) { continue; }
    if (r.c != IC) {
      session()->warn(fmt(
          "GenerateImageStage('{}'): reference latent has {} channels, expected "
          "{} (DiT in_channels); ignoring", this->id(), r.c, IC));
      continue;
    }
    const int rh = r.h, rw = r.w, rseq = rh * rw;
    SharedBuffer rb = mc->make_shared_buffer((std::size_t)rseq * IC * 2);
    if (rb.empty()) { continue; }
    auto* d = static_cast<_Float16*>(rb.contents());
    for (int y = 0; y < rh; ++y) {
      for (int x = 0; x < rw; ++x) {
        const std::size_t t = (std::size_t)y * rw + x;
        for (int cc = 0; cc < IC; ++cc) {
          d[t * IC + cc] =
              (_Float16)r.chw[((std::size_t)cc * rh + y) * rw + x];
        }
      }
    }
    genai::MetalFlux2Transformer::RefImage img;
    img.latents = std::move(rb);
    img.seq = rseq; img.grid_h = rh; img.grid_w = rw;
    ri.push_back(std::move(img));
  }
  if (!ri.empty()) {
    session()->info(fmt(
        "GenerateImageStage('{}'): FLUX.2 conditioning on {} reference image(s)",
        this->id(), ri.size()));
  }
  bool dit_ok = true;
  // Reference K/V cache for the -kv checkpoint, scoped to THIS generation: the
  // first denoise step fills it, the rest reuse it. A local means a new run
  // (or new reference set) always starts from an empty one.
  genai::MetalFlux2Transformer::KvCache kvc;
  // VPIPE_FLUX2_NO_KV_CACHE recomputes the reference band every step while
  // keeping the recipe itself intact. That is the only honest way to price the
  // cache: turning `klein_kv` off instead would change the ATTENTION, so the
  // two arms would not be computing the same thing.
  genai::MetalFlux2Transformer::KvCache* kvp =
      (_flux2_params.klein_kv && !ri.empty() &&
       std::getenv("VPIPE_FLUX2_NO_KV_CACHE") == nullptr) ? &kvc : nullptr;
  // Opened before the denoise callable so the per-block hook is
  // live for the very first forward.
  UiProgress bar = session()->open_progress("denoise");
  DenoiseProgress prog(&bar, S, 1);
  ScopedBlockProgress<std::remove_reference_t<decltype(*_flux2_dit)>>
      prog_guard(_flux2_dit.get(), prog);
  // Set once the first forward has shown whether the ANE tiers armed.
  bool flux2_ane_checked = false;
  X0Tap tap;   // a live preview's clean estimate (packed)
  auto denoise = [&](const std::vector<float>& cand,
                     double sigma) -> std::vector<float> {
    auto* lb = static_cast<_Float16*>(latbuf.contents());
    for (std::size_t k = 0; k < cand.size(); ++k) { lb[k] = (_Float16)cand[k]; }
    SharedBuffer vel = _flux2_dit->forward_dit(context, n_real, latbuf, img_seq,
                                               gh, gw, (float)sigma, guid, ri,
                                               kvp);
    prog.end_forward();
    // After the first forward, whose blocks are where the ANE tiers arm or
    // decline: book what is HELD -- the modules when any armed, 0 when none
    // did. revise_scratch() refuses to create, so this is a no-op when the
    // plan booked nothing.
    if (!flux2_ane_checked && _flux2_dit->ane_attempted()) {
      flux2_ane_checked = true;
      if (auto* mgr = session()->services()->generative_model_manager()) {
        mgr->revise_scratch(
            "coreml:" + ane_claim_label_(),
            _flux2_dit->ane_armed()
                ? genai::MetalFlux2Transformer::ane_runtime_bytes(
                      _flux2_dit->config(), img_seq)
                : 0);
      }
    }
    if (vel.empty()) { dit_ok = false; return {}; }
    const auto* vp = static_cast<const _Float16*>(vel.contents());
    std::vector<float> v(cand.size());
    for (std::size_t k = 0; k < v.size(); ++k) { v[k] = (float)vp[k]; }
    tap.take(cand, sigma, v);
    return v;
  };
  sampler.reset();
  // Per-step DiT timing (VPIPE_FLUX2_PROFILE), mirroring the Krea-2 path's
  // VPIPE_KREA2_PROFILE: time each forward_dit + log a ms/call summary.
  const bool prof = std::getenv("VPIPE_FLUX2_PROFILE") != nullptr;
  double dit_ms = 0.0;
  int dit_calls = 0;
  auto denoise_p = [&](const std::vector<float>& cand, double sigma) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<float> v = denoise(cand, sigma);
    dit_ms += std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t0).count();
    ++dit_calls;
    return v;
  };
  // Unpack [img_seq, IC] (token-major) -> channel-first [IC, gh, gw]. Used
  // per-step (debug step_latents) and for the final return.
  auto unpack = [&](const std::vector<float>& pk) {
    std::vector<float> latent((std::size_t)IC * gh * gw);
    for (int i = 0; i < gh; ++i) {
      for (int j = 0; j < gw; ++j) {
        const std::size_t t = (std::size_t)i * gw + j;
        for (int cc = 0; cc < IC; ++cc) {
          latent[((std::size_t)cc * gh + i) * gw + j] = pk[t * IC + cc];
        }
      }
    }
    return latent;
  };
  const auto gen_t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < S; ++i) {
    tap.arm(_preview && _preview->due(i + 1, S));
    sampler.step(i, packed,
                 prof ? genai::FlowSampler::DenoiseFn(denoise_p) : denoise);
    if (!dit_ok) { return {}; }
    if (emit_step) { emit_step(unpack(packed)); }
    if (tap.ready()) {
      // taef2 reads the DiT's latent UNPATCHIFIED -- [IC/4, 2gh, 2gw],
      // channel c*4 + ph*2 + pw landing at (2i+ph, 2j+pw), the FLUX.2
      // VAE's own order -- and NOT un-normalized: the batch-norm inverse
      // the real VAE applies first is exactly what the TAE was distilled
      // without (its diffusers wrapper swaps in an identity BN).
      const std::vector<float> z = unpack(tap.x0);
      const int L = IC / 4, h8 = 2 * gh, w8 = 2 * gw;
      std::vector<float> lat((std::size_t)L * h8 * w8);
      for (int cc = 0; cc < IC; ++cc) {
        const int c = cc / 4, ph = (cc % 4) / 2, pw = cc % 2;
        for (int y = 0; y < gh; ++y) {
          for (int x = 0; x < gw; ++x) {
            lat[((std::size_t)c * h8 + 2 * y + ph) * w8 + 2 * x + pw] =
                z[((std::size_t)cc * gh + y) * gw + x];
          }
        }
      }
      _preview->submit(std::move(lat), L, 1, h8, w8, i + 1, S);
    }
    prog.end_step(i);
  }
  const double gen_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - gen_t0).count();
  session()->info(fmt(
      "GenerateImageStage('{}'): FLUX.2 latent generated in {:.2f}s ({} steps)",
      this->id(), gen_s, S));
  if (prof) {
    session()->log_normal(fmt(
        "GenerateImageStage('{}'): FLUX.2 DiT {} forward_dit calls, {} ms total, "
        "{} ms/call (seq {}+{}={})", this->id(), dit_calls, (long)dit_ms,
        dit_calls ? (long)(dit_ms / dit_calls) : 0, n_real, img_seq,
        n_real + img_seq));
  }
  return unpack(packed);
}

// Boogu-Image: NextDiT forward + the checkpoint's own sampler.
//
// Two integrators live here because Boogu ships two kinds of checkpoint and
// they do NOT share a loop:
//   * "dmd" (Base/Edit-Turbo, the distilled default) -- at each ASCENDING sigma
//     jump straight to the x0 prediction, x <- x + (1 - s) * v, then RE-NOISE
//     down to the next sigma, x <- (1 - s') * eps + s' * x. No CFG.
//   * anything else (Base/Edit) -- Euler along the v1 logistic time-shift
//     schedule, x <- x + (s_next - s) * v, optionally with classifier-free
//     guidance from the conditioner's negative beat.
// Both run on Boogu's INVERTED sigma convention: 0 is pure noise and 1 is
// clean, so the schedule ASCENDS. That is why neither uses genai::FlowSampler
// (whose integrators all assume a descending sigma with a terminal 0).
std::vector<float>
GenerateImageStage::generate_qwen_image21_(
    const metal_compute::SharedBuffer& txt_pos, int n_real,
    const metal_compute::SharedBuffer& txt_neg, int n_real_neg, int gen_h,
    int gen_w, const std::vector<RefLatent>& refs,
    const std::function<void(const std::vector<float>&)>& emit_step) const
{
  using metal_compute::SharedBuffer;
  auto* mc = session()->services()->metal_compute();
  if (mc == nullptr || !_qi21_dit) { return {}; }
  const auto& dcfg = _qi21_dit->config();
  const int IC = dcfg.in_channels;                 // 64, latents UNPATCHED
  const int lh = gen_h / 16, lw = gen_w / 16;      // the /16 VAE's grid
  const int img_seq = lh * lw;
  // The size grid is 32, not 16: the pipeline rounds the LATENT grid
  // even, so an odd latent axis cannot be expressed. It also has to be
  // a whole number of 2x2 encoder slots, which is the same condition.
  if (img_seq <= 0 || (lh % 2) != 0 || (lw % 2) != 0) {
    session()->warn(fmt(
        "GenerateImageStage('{}'): {}x{} gives a {}x{} latent, and this model "
        "needs an EVEN latent grid (one encoder slot is 2x2 tokens) -- use "
        "multiples of 32", this->id(), gen_w, gen_h, lw, lh));
    return {};
  }

  // ---- the joint sequence -------------------------------------------
  //
  // The conditioner published which of ITS rows are image slots; the
  // target's slots are appended here, because the target geometry is
  // this stage's to know and the conditioner's to not.
  std::vector<std::uint8_t> slot = _qi21_slots;
  if ((int)slot.size() != n_real) {
    // Not a warning that can be worked around: without the mask there
    // is no way to tell a text row from an image row, and guessing
    // would lay the sequence out wrongly and still produce a picture.
    session()->warn(fmt(
        "GenerateImageStage('{}'): the conditioning beat carries a slot mask "
        "of {} for {} rows -- wire a qwen-image-21 conditioner",
        this->id(), (int)_qi21_slots.size(), n_real));
    return {};
  }
  std::vector<genai::qi21::ImgBlock> blocks;
  std::vector<const RefLatent*> used;
  for (std::size_t i = 0; i < refs.size(); ++i) {
    if (refs[i].empty()) { continue; }
    const int gh = i < _qi21_ref_gh.size() ? _qi21_ref_gh[i] : 0;
    const int gw = i < _qi21_ref_gw.size() ? _qi21_ref_gw[i] : 0;
    // THE CONTRACT, checked. One resize feeds both the tower and the
    // VAE, so a reference's latent grid must be exactly twice the
    // tower's merged grid. When it is not, the two stages resized
    // differently and the layout below would silently mean something
    // else.
    if (gh != refs[i].h || gw != refs[i].w) {
      // SAY IT IN PIXELS, because pixels are what the graph can act
      // on. Both numbers divide back: the latent grid is this model's
      // /16, and the published grid is twice the tower's merged one,
      // which is the same /16. So each side names the picture its
      // stage actually read, and the difference is a resize somebody
      // did and somebody else did not -- almost always the
      // conditioner's own `vl_pixel_budget` cap (1024^2 by default,
      // floored to a multiple of 32), which vae-encode knows nothing
      // about.
      //
      // The conditioner's size is the whole remedy, and it is worth
      // saying why it is safe to name: it RESIZES TO ITSELF. It is
      // already under the budget and already aligned to 32, so feeding
      // it to both stages leaves the cap a no-op and the two agree.
      const int vw = refs[i].w * 16, vh = refs[i].h * 16;
      const int cw = gw * 16, ch = gh * 16;
      session()->warn(fmt(
          "GenerateImageStage('{}'): reference {} disagrees -- vae-encode "
          "read a {}x{} picture ({}x{} latent) and the conditioner read "
          "{}x{} ({}x{} grid). ONE resize has to feed both. Either "
          "resample the reference to {}x{} ahead of BOTH stages "
          "(image-resample, fit 'crop', both axes multiples of 32), or "
          "set vl_pixel_budget to at least {} on "
          "qwen-image-21-model-config so the conditioner stops shrinking "
          "it",
          this->id(), (int)i, vw, vh, refs[i].w, refs[i].h, cw, ch, gw, gh,
          cw, ch, (long long)vw * (long long)vh));
      return {};
    }
    blocks.push_back(genai::qi21::ImgBlock{1, gh, gw});
    used.push_back(&refs[i]);
  }
  blocks.push_back(genai::qi21::ImgBlock{1, lh, lw});
  slot.insert(slot.end(), (std::size_t)(img_seq / 4), 1);
  genai::qi21::Layout lay;
  std::string lerr;
  if (!genai::qi21::build_layout(slot, {}, blocks, &lay, &lerr)) {
    session()->warn(fmt(
        "GenerateImageStage('{}'): the joint sequence does not add up: {}",
        this->id(), lerr));
    return {};
  }

  // ---- the NEGATIVE's joint sequence --------------------------------
  //
  // A sequence of its own, as the reference's is: its own mask (its text
  // is another length, its image slots at other rows) with the same
  // target slots appended, over the SAME image blocks -- so the latents
  // are shared and only the text rows and positions differ.
  const bool cfg = !txt_neg.empty() && n_real_neg > 0 &&
                   _guidance_scale != 1.0;
  genai::qi21::Layout lay_neg;
  if (cfg) {
    std::vector<std::uint8_t> nslot = _qi21_slots_neg;
    if ((int)nslot.size() != n_real_neg) {
      session()->warn(fmt(
          "GenerateImageStage('{}'): the NEGATIVE conditioning beat carries "
          "a slot mask of {} for {} rows -- wire a qwen-image-21 "
          "conditioner's oport1", this->id(), (int)nslot.size(),
          n_real_neg));
      return {};
    }
    nslot.insert(nslot.end(), (std::size_t)(img_seq / 4), 1);
    if (!genai::qi21::build_layout(nslot, {}, blocks, &lay_neg, &lerr)) {
      // Most often a negative encoded without the references the
      // positive was (another conditioner, or none of its pictures).
      session()->warn(fmt(
          "GenerateImageStage('{}'): the negative's joint sequence does not "
          "add up: {} -- the negative must be encoded with the positive's "
          "references", this->id(), lerr));
      return {};
    }
  }

  // ---- schedule ------------------------------------------------------
  // FlowMatchEulerDiscreteScheduler as the CHECKPOINT configures it (see
  // qi21_checkpoint_schedule): the base model shifts linspace(1, 1/S, S)
  // per image and stretches the tail; a distilled checkpoint runs its own
  // raw nodes unshifted.
  //
  // A wired scheduler contributes its STEP COUNT and nothing else.
  // scheduler-select emits a curve (simple, shift 1.15) whatever it was
  // set to, and taking it -- as this family once did -- ran every graph
  // that pinned its steps that way on a schedule neither checkpoint was
  // trained on.
  const Qi21Schedule& ck = _qi21_ck;
  genai::FlowSchedulerSpec sched;
  sched.dynamic_shift = ck.dynamic;
  // Static is diffusers' s' = shift*s / (1 + (shift-1)*s): the "linear"
  // curve here, with the static shift as its parameter.
  sched.shift_type = ck.dynamic ? ck.shift_type : "linear";
  sched.shift = ck.shift;
  sched.base_shift = ck.base_shift;
  sched.max_shift = ck.max_shift;
  sched.base_seq = ck.base_seq;
  sched.max_seq = ck.max_seq;
  sched.shift_terminal = ck.shift_terminal;
  // The reference's num_inference_steps default is 40 -- this stage's
  // own `steps` default (8) is for the turbo families.
  const bool asked = _scheduler_latched || _steps_set;
  const int want = _scheduler_latched && _scheduler_spec.steps > 0
                       ? _scheduler_spec.steps
                       : (_steps_set ? _steps : 40);
  sched.steps = want;
  if (!ck.sample_sigmas.empty()) {
    // The checkpoint's nodes ARE the schedule and the count, as in the
    // reference, which ignores num_inference_steps for them. A different
    // grid goes on qwen-image-21-model-config's `sigmas`.
    sched.base_sigmas = ck.sample_sigmas;
    sched.steps = (int)ck.sample_sigmas.size();
    if (asked && want != sched.steps && _qi21_sigmas.empty() &&
        !_qi21_sched_noted) {
      session()->warn(fmt(
          "GenerateImageStage('{}'): {} steps is IGNORED -- this "
          "checkpoint is distilled to its own {}-node schedule, which "
          "runs. Set `sigmas` on qwen-image-21-model-config to try "
          "another grid", this->id(), want, sched.steps));
    }
  }
  // A few-step adapter's own schedule, when the model config names one:
  // its raw nodes replace the grid and ARE the step count; the
  // checkpoint's shift is applied to them all the same.
  if (!_qi21_sigmas.empty()) {
    sched.base_sigmas = _qi21_sigmas;
    sched.steps = (int)_qi21_sigmas.size();
  }
  if (_qi21_shift_terminal >= 0.0) {
    sched.shift_terminal = _qi21_shift_terminal;
  }
  if (_scheduler_latched && !_qi21_sched_noted &&
      (_scheduler_spec.type != "simple" ||
       _scheduler_spec.dynamic_shift)) {
    session()->warn(fmt(
        "GenerateImageStage('{}'): the wired scheduler's '{}' curve is not "
        "used -- Qwen-Image-2.1 runs the checkpoint's own schedule and "
        "takes only the scheduler's step count", this->id(),
        _scheduler_spec.type));
  }
  _qi21_sched_noted = true;
  sched.img_seq_len = img_seq;
  genai::FlowSampler sampler(_sampler_spec, sched);
  const int S = sampler.steps();

  std::vector<float> packed((std::size_t)img_seq * IC);
  {
    std::mt19937_64 rng(_seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : packed) { v = nd(rng); }
  }

  // Every image block's latents, condition images FIRST and the target
  // LAST -- the order build_layout labelled them in.
  const std::size_t total_img = (std::size_t)lay.image_len * IC;
  SharedBuffer latbuf = mc->make_shared_buffer(total_img * 2);
  if (latbuf.empty()) { return {}; }
  {
    auto* lb = static_cast<std::uint16_t*>(latbuf.contents());
    std::size_t at = 0;
    for (const RefLatent* r : used) {
      // Channel-first [c,h,w] -> packed [h*w, c], which is the layout
      // the DiT reads and the one _pack_latents produces.
      for (int y = 0; y < r->h; ++y) {
        for (int x = 0; x < r->w; ++x) {
          for (int c = 0; c < r->c && c < IC; ++c) {
            lb[at + c] = f32_to_bf16_(
                r->chw[((std::size_t)c * r->h + y) * r->w + x]);
          }
          at += (std::size_t)IC;
        }
      }
    }
  }
  const std::size_t tgt_off = (std::size_t)(lay.image_len - img_seq) * IC;

  const float gscale = (float)_guidance_scale;
  bool dit_ok = true;
  UiProgress bar = session()->open_progress("denoise");
  DenoiseProgress prog(&bar, S, cfg ? 2 : 1);
  ScopedBlockProgress<std::remove_reference_t<decltype(*_qi21_dit)>>
      prog_guard(_qi21_dit.get(), prog);

  // ---- how much of the stack this run may KEEP ----------------------
  //
  // Without this the DiT streams and NEVER grows: BlockResidency refuses
  // every admission until a reserve has been DECLARED, because a model
  // that has not been told what else needs the box cannot decide what is
  // safe to hold. Declaring 0 is a real answer ("nothing runs beside
  // me"); declaring nothing is not, and reads as a checkpoint re-read
  // from disk on all forty steps.
  if (mc != nullptr && _qi21_dit->streaming()) {
    const std::size_t peak =
        qwen_decode_peak_(gen_w, gen_h, _vae_base > 0 ? _vae_base : 144,
                          /*residual=*/true);
    const auto mb = mc->memory_budget();
    // No budget to read -> the free bails out too, so the DiT survives
    // the decode and the reserve has to cover it.
    const bool decode_runs_beside_us =
        mb.recommended == 0 || (mb.fits(peak) && mb.fits_physical(peak));
    _qi21_dit->set_residency_reserve(decode_runs_beside_us ? peak : 0);
    session()->log_debug(fmt(
        "GenerateImageStage('{}'): Qwen-Image-2.1 residency reserve {} MB "
        "-- the {}x{} vae-decode wants {} MB and {}",
        this->id(), (decode_runs_beside_us ? peak : 0) >> 20, gen_w, gen_h,
        peak >> 20,
        decode_runs_beside_us ? "fits beside the DiT"
                              : "does not, so the DiT is freed before it"));
    // And the RATES, after the reserve because the probe is sized
    // against what is left once the reserve is taken. Both of
    // BlockResidency's defaults assume a ~30-step run; this model's own
    // default is 40, but a graph is free to ask for 8.
    _qi21_dit->set_residency_schedule(
        _scheduler_spec.steps > 0 ? _scheduler_spec.steps : 1);
  }

  // The cross-step prefix caches. TWO of them under CFG, because the
  // positive and negative prompts have different prefixes -- sharing one
  // would condition every later step on whichever ran first.
  genai::MetalQwenImage21Transformer::KvCache kv_pos, kv_neg;
  // The cache is on unless the graph's model-config turned it off. It
  // is exact rather than approximate, so there is no quality reason to
  // want it off -- the key exists for A/B and for the equivalence test.
  // Each sequence decides for itself: its prefix is its own.
  auto use_kv = [&](const genai::qi21::Layout& L) {
    return _qi21_use_kv_cache && L.prefix_len > 0;
  };
  int step_i = 0;

  auto run = [&](const SharedBuffer& txt, const genai::qi21::Layout& L,
                 genai::MetalQwenImage21Transformer::KvCache* kv, float sigma)
      -> SharedBuffer {
    genai::MetalQwenImage21Transformer::Request req;
    req.latents = &latbuf;
    req.txt = &txt;
    req.layout = &L;
    req.timestep = sigma;
    if (use_kv(L)) {
      req.kv = kv;
      req.kv_mode = step_i == 0
          ? genai::MetalQwenImage21Transformer::KvMode::kFill
          : genai::MetalQwenImage21Transformer::KvMode::kCached;
    }
    std::string e;
    SharedBuffer out = _qi21_dit->forward(req, &e);
    if (out.empty()) {
      // A cooperative stop comes back empty with NO reason, and is not
      // a failure -- the beat-level line below reports it as stopped.
      // Only a real fault has something to say here.
      if (!e.empty()) {
        session()->warn(fmt("GenerateImageStage('{}'): DiT forward failed: "
                            "{}", this->id(), e));
      }
      dit_ok = false;
    }
    return out;
  };

  X0Tap tap;   // a live preview's clean estimate (packed)
  auto denoise = [&](const std::vector<float>& cand,
                     double sigma) -> std::vector<float> {
    // The target block's rows are the TAIL of the packed latent.
    auto* lb = static_cast<std::uint16_t*>(latbuf.contents());
    for (std::size_t k = 0; k < cand.size(); ++k) {
      lb[tgt_off + k] = f32_to_bf16_(cand[k]);
    }
    SharedBuffer v = run(txt_pos, lay, &kv_pos, (float)sigma);
    if (!dit_ok || v.empty()) { return {}; }
    std::vector<float> out((std::size_t)img_seq * IC);
    const auto* vp = static_cast<const std::uint16_t*>(v.contents());
    for (std::size_t k = 0; k < out.size(); ++k) {
      out[k] = bf16_to_f32_(vp[k]);
    }
    prog.end_forward();
    if (cfg) {
      SharedBuffer nv = run(txt_neg, lay_neg, &kv_neg, (float)sigma);
      if (!dit_ok || nv.empty()) { return {}; }
      const auto* np = static_cast<const std::uint16_t*>(nv.contents());
      for (std::size_t k = 0; k < out.size(); ++k) {
        const float n = bf16_to_f32_(np[k]);
        out[k] = n + gscale * (out[k] - n);
      }
      prog.end_forward();
    }
    // THE VELOCITY IS DATA-WARD for this family's scheduler? No: the
    // reference hands the DiT output straight to
    // FlowMatchEulerDiscreteScheduler.step, which is the ordinary
    // noise-ward convention this sampler already implements.
    tap.take(cand, sigma, out);
    return out;
  };

  auto unpack = [&](const std::vector<float>& pk) {
    std::vector<float> latent((std::size_t)IC * lh * lw);
    for (int t = 0; t < img_seq; ++t) {
      const int y = t / lw, x = t % lw;
      for (int c = 0; c < IC; ++c) {
        latent[((std::size_t)c * lh + y) * lw + x] =
            pk[(std::size_t)t * IC + c];
      }
    }
    return latent;
  };

  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < S; ++i) {
    step_i = i;
    tap.arm(_preview && _preview->due(i + 1, S));
    sampler.step(i, packed, denoise);
    if (!dit_ok) { return {}; }
    if (emit_step) { emit_step(unpack(packed)); }
    if (tap.ready()) {
      // taeqi2_1 reads the DiT's latent as it stands -- [64, H/16, W/16],
      // unpatched and still whitened -- and makes RGBA of it, as the real
      // VAE does.
      _preview->submit(unpack(tap.x0), IC, 1, lh, lw, i + 1, S);
    }
    prog.end_step(i);
  }
  const double secs = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();
  session()->info(fmt(
      "GenerateImageStage('{}'): Qwen-Image-2.1 latent generated in {:.2f}s "
      "({} steps, joint {}, prefix {}{}{})", this->id(), secs, S,
      lay.joint_len, lay.prefix_len,
      cfg ? fmt("; negative joint {}, prefix {}", lay_neg.joint_len,
                lay_neg.prefix_len)()
          : std::string{},
      use_kv(lay) ? ", prefix KV cached" : ""));
  return unpack(packed);
}

// Z-Image: the NextDiT over patch-packed latents, along a STATIC
// flow-shift Euler schedule.
//
// Three things here are this family's alone and each is silent:
//
//  * THE VELOCITY IS NEGATED. The reference computes `noise_pred =
//    -model_out` before handing it to the scheduler, so a run that skips
//    the sign denoises AWAY from the data and returns noise that still
//    decodes to an image.
//  * THE TIMESTEP IS INVERTED. The DiT's own input is 1 - sigma; the
//    transformer converts, so this hands it sigma like every other
//    family here.
//  * CFG IS `pos + g*(pos - neg)`, not the usual `neg + g*(pos - neg)`.
//    The two differ by one in the scale, so a model card's "guidance 4"
//    read the ordinary way is the wrong picture rather than a broken
//    one.
std::vector<float>
GenerateImageStage::generate_z_image_(
    const metal_compute::SharedBuffer& txt_pos, int n_real,
    const metal_compute::SharedBuffer& txt_neg, int n_real_neg, int gen_h,
    int gen_w,
    const std::function<void(const std::vector<float>&)>& emit_step) const
{
  auto* mc = session()->services()->metal_compute();
  using metal_compute::SharedBuffer;
  if (mc == nullptr || !_zi_dit) { return {}; }
  const auto& dcfg = _zi_dit->config();
  const int Z = dcfg.in_channels;            // 16
  const int P = dcfg.patch;                  // 2
  const int lh = gen_h / 8, lw = gen_w / 8;  // the VAE's 8x latent grid
  const int gh = lh / P, gw = lw / P;        // the DiT's patch grid
  const int img_seq = gh * gw;
  const int PD = dcfg.patch_dim();           // 64
  if (img_seq <= 0 || (lh % P) != 0 || (lw % P) != 0) {
    session()->warn(fmt(
        "GenerateImageStage('{}'): {}x{} gives a {}x{} latent, which is not "
        "a whole number of {}x{} patches -- use multiples of {}",
        this->id(), gen_w, gen_h, lw, lh, P, P,
        genai::zimage::kPixelsPerToken));
    return {};
  }
  if (n_real <= 0) { return {}; }
  // THIS FAMILY HAS NO REFERENCE PATH. There is no vision tower, no
  // released edit checkpoint, and nothing in the sequence for a
  // reference to occupy -- so a wired reference latent is ignored, and
  // saying so beats letting someone wonder why their edit did nothing.
  if (!_ref.empty()) {
    session()->warn(fmt(
        "GenerateImageStage('{}'): a reference latent is wired but "
        "Z-Image is text-to-image only -- it is ignored. The family's "
        "editing checkpoint is not published", this->id()));
  }

  genai::zimage::Layout lay;
  std::string lerr;
  if (!genai::zimage::build_layout(gh, gw, n_real, &lay, &lerr)) {
    session()->warn(fmt(
        "GenerateImageStage('{}'): the joint sequence does not add up: {}",
        this->id(), lerr));
    return {};
  }

  // ---- schedule ------------------------------------------------------
  //
  // FlowMatchEulerDiscreteScheduler with use_dynamic_shifting FALSE, so
  // the shift is the STATIC curve s' = shift*s / (1 + (shift-1)*s) --
  // which is exactly `time_shift(..., linear)` -- over the base grid
  // linspace(1, 1/S, S), which is what `type == "simple"` builds. The
  // shift comes off the checkpoint's own scheduler_config: 3.0 for
  // Turbo and 6.0 for the undistilled base, and nothing else in either
  // repo tells the two apart.
  genai::FlowSchedulerSpec sched = _scheduler_spec;
  if (!_scheduler_latched) {
    sched.type = "simple";
    sched.dynamic_shift = false;
    sched.shift_type = "linear";
    // FROM THE CHECKPOINT, not from FlowSchedulerSpec's default --
    // which is 1.15 and would win this ternary for both variants.
    sched.shift = _zi_shift > 0.0 ? _zi_shift : 3.0;
    sched.steps = _steps > 0 ? _steps : 8;
  }
  sched.img_seq_len = img_seq;
  genai::FlowSamplerSpec samp = _sampler_spec;
  if (!_sampler_latched) { samp.method = "euler"; }
  genai::FlowSampler sampler(samp, sched);
  const int S = sampler.steps();

  std::vector<float> packed((std::size_t)img_seq * PD);
  {
    std::mt19937_64 rng(_seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : packed) { v = nd(rng); }
  }
  SharedBuffer latbuf = mc->make_shared_buffer((std::size_t)img_seq * PD * 2);
  if (latbuf.empty()) { return {}; }

  // CFG. The Turbo checkpoint is distilled to guidance 0 and answers no
  // negative prompt; the base model wants 3-5. `_zi_guidance` defaults
  // to Turbo's because the two ship identical transformer configs and
  // nothing here can tell them apart.
  const bool cfg = !txt_neg.empty() && n_real_neg > 0 && _zi_guidance > 0.0;
  const float gscale = (float)_zi_guidance;
  genai::zimage::Layout lay_neg;
  if (cfg && !genai::zimage::build_layout(gh, gw, n_real_neg, &lay_neg,
                                          &lerr)) {
    session()->warn(fmt(
        "GenerateImageStage('{}'): the negative joint sequence does not "
        "add up: {}", this->id(), lerr));
    return {};
  }
  bool dit_ok = true;
  UiProgress bar = session()->open_progress("denoise");
  DenoiseProgress prog(&bar, S, cfg ? 2 : 1);
  ScopedBlockProgress<std::remove_reference_t<decltype(*_zi_dit)>>
      prog_guard(_zi_dit.get(), prog);

  // ---- how much of the stack this run may KEEP ----------------------
  //
  // Without this the DiT streams and NEVER grows: BlockResidency refuses
  // every admission until a reserve has been DECLARED. Declaring 0 is a
  // real answer ("nothing runs beside me"); declaring nothing is not,
  // and reads as a checkpoint re-read from disk on every step.
  if (_zi_dit->streaming()) {
    const std::size_t peak =
        flux2_decode_peak_(gen_w, gen_h, _vae_base > 0 ? _vae_base : 128);
    const auto mb = mc->memory_budget();
    const bool decode_runs_beside_us =
        mb.recommended == 0 || (mb.fits(peak) && mb.fits_physical(peak));
    _zi_dit->set_residency_reserve(decode_runs_beside_us ? peak : 0);
    _zi_dit->set_residency_schedule(S);
    session()->log_debug(fmt(
        "GenerateImageStage('{}'): Z-Image residency reserve {} MB -- the "
        "{}x{} vae-decode wants {} MB and {}", this->id(),
        (decode_runs_beside_us ? peak : 0) >> 20, gen_w, gen_h, peak >> 20,
        decode_runs_beside_us ? "fits beside the DiT"
                              : "does not, so the DiT is freed before it"));
  }

  auto run = [&](const SharedBuffer& txt, const genai::zimage::Layout& L,
                 float sigma) -> SharedBuffer {
    genai::MetalZImageTransformer::Request req;
    req.latents = &latbuf;
    req.cap = &txt;
    req.layout = &L;
    req.timestep = sigma;
    std::string e;
    SharedBuffer out = _zi_dit->forward(req, &e);
    if (out.empty()) {
      // A cooperative stop comes back empty with NO reason, and is not a
      // failure -- the beat-level line reports it as stopped. Only a
      // real fault has something to say here.
      if (!e.empty()) {
        session()->warn(fmt("GenerateImageStage('{}'): DiT forward failed: "
                            "{}", this->id(), e));
      }
      dit_ok = false;
    }
    return out;
  };

  X0Tap tap;   // a live preview's clean estimate (packed)
  auto denoise = [&](const std::vector<float>& cand,
                     double sigma) -> std::vector<float> {
    auto* lb = static_cast<std::uint16_t*>(latbuf.contents());
    for (std::size_t k = 0; k < cand.size(); ++k) {
      lb[k] = f32_to_bf16_(cand[k]);
    }
    SharedBuffer v = run(txt_pos, lay, (float)sigma);
    if (!dit_ok || v.empty()) { return {}; }
    std::vector<float> out((std::size_t)img_seq * PD);
    const auto* vp = static_cast<const std::uint16_t*>(v.contents());
    for (std::size_t k = 0; k < out.size(); ++k) {
      out[k] = bf16_to_f32_(vp[k]);
    }
    prog.end_forward();
    if (cfg) {
      // CFG TRUNCATION: the reference drops guidance once (1 - sigma)
      // passes the threshold. Default 1.0, which never fires.
      const bool guide = (1.0 - sigma) <= _zi_cfg_trunc;
      SharedBuffer nv = run(txt_neg, lay_neg, (float)sigma);
      if (!dit_ok || nv.empty()) { return {}; }
      if (guide) {
        const auto* np = static_cast<const std::uint16_t*>(nv.contents());
        double pn = 0.0, dn = 0.0;
        for (std::size_t k = 0; k < out.size(); ++k) {
          const float p = out[k];
          const float n = bf16_to_f32_(np[k]);
          // `pos + g*(pos - neg)`, which is NOT the usual form.
          out[k] = p + gscale * (p - n);
          pn += (double)p * (double)p;
          dn += (double)out[k] * (double)out[k];
        }
        // CFG RENORMALIZATION: clamp the guided prediction's norm to a
        // multiple of the conditional one's. Off by default.
        if (_zi_cfg_norm > 0.0 && dn > 0.0) {
          const double maxn = std::sqrt(pn) * _zi_cfg_norm;
          const double newn = std::sqrt(dn);
          if (newn > maxn) {
            const float k = (float)(maxn / newn);
            for (auto& o : out) { o *= k; }
          }
        }
      }
      prog.end_forward();
    }
    // THE SIGN. The reference negates the DiT's output before the
    // scheduler step; this sampler's Euler update is the ordinary
    // noise-ward one, so the negation belongs here.
    for (auto& o : out) { o = -o; }
    tap.take(cand, sigma, out);
    return out;
  };

  auto unpack = [&](const std::vector<float>& pk) {
    std::vector<float> latent((std::size_t)Z * lh * lw);
    genai::zimage::unpack_latents(pk.data(), Z, lh, lw, P, latent.data());
    return latent;
  };

  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < S; ++i) {
    tap.arm(_preview && _preview->due(i + 1, S));
    sampler.step(i, packed, denoise);
    if (!dit_ok) { return {}; }
    if (emit_step) { emit_step(unpack(packed)); }
    if (tap.ready()) {
      // taef1 reads the DiT's own latent space: the FLUX.1 AE's scale and
      // shift are what the real decode undoes, and the TAE never saw them.
      _preview->submit(unpack(tap.x0), Z, 1, lh, lw, i + 1, S);
    }
    prog.end_step(i);
  }
  const double secs = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();
  session()->info(fmt(
      "GenerateImageStage('{}'): Z-Image latent generated in {:.2f}s ({} "
      "steps, joint {} = {} image + {} caption, shift {:.1f}, {})",
      this->id(), secs, S, lay.joint_len, lay.img_total, lay.cap_total,
      sched.shift,
      cfg ? "cfg " + std::to_string(_zi_guidance) : std::string("no cfg")));
  return unpack(packed);
}

// Fibo 2.0: the reference pipeline's loop (pipeline_bria_fibo2.py,
// BriaFibo2Pipeline.__call__), where a bit matters:
//   * the schedule: sigmas linspace(1, 1/S, S), shifted by
//     calculate_shift(the image's tokens) exactly as FlowMatchEuler forms
//     it -- in f32, NumPy's scalars narrowed to the array's type;
//   * the DiT handed (sigma * 1000) / 1000, the scheduler's own round trip;
//   * the noise: torch.randn((1, 32, H/8, W/8)) from a CPU generator
//     seeded with the seed, f32, CHANNEL-major, then patched;
//   * f32 Euler on the patch rows (elementwise, so the order is free);
//   * the pictures to edit: vae-encode's normalized [128, h, w] latent
//     IS this model's patch rows ((channel, dy, dx), FLUX.2's patchify),
//     token-major, in bf16 as the pipeline casts them;
//   * GUIDANCE above 1 (Fibo 2.0 Dev; the distilled Edit Turbo runs at
//     1.0): the negative's own forward at every step, on the same latent
//     rows and pictures -- its text a sequence of its own, so its own
//     gist count and layout -- and uncond + g (cond - uncond) on the two
//     velocities as f32, three roundings as torch does them.
std::vector<float>
GenerateImageStage::generate_fibo2_(
    const metal_compute::SharedBuffer& text, int n_rows,
    const metal_compute::SharedBuffer& neg, int n_neg, int gen_h,
    int gen_w, const std::vector<RefLatent>& refs,
    const std::vector<float>* init_packed,
    const std::function<void(const std::vector<float>&)>& emit_step) const
{
  auto* mc = session()->services()->metal_compute();
  using metal_compute::SharedBuffer;
  if (mc == nullptr || !_fibo2_dit) { return {}; }
  const auto& dcfg = _fibo2_dit->config();
  const int PD = dcfg.patch_dim();             // 128
  const int LC = dcfg.in_channels;             // 32
  if ((gen_h % 16) != 0 || (gen_w % 16) != 0 || gen_h <= 0 || gen_w <= 0) {
    session()->warn(fmt(
        "GenerateImageStage('{}'): Fibo 2.0 needs a size that is a multiple "
        "of 16 (got {}x{})", this->id(), gen_w, gen_h));
    return {};
  }
  const int gh = gen_h / 16, gw = gen_w / 16;
  const int img_seq = gh * gw;
  std::string terr;
  if (!_fibo2_dit->set_text(text, n_rows, &terr)) {
    session()->warn(fmt(
        "GenerateImageStage('{}'): the Fibo 2.0 DiT wants [6, rows, {}] bf16 "
        "conditioning (the diffusion-conditioner's fibo2 family): {}",
        this->id(), dcfg.cap_dim, terr));
    return {};
  }
  // ---- guidance ------------------------------------------------------------
  //
  // The reference guides when the scale is ABOVE 1
  // (do_classifier_free_guidance), so 1 and below run one pass.
  const double g = _fibo2_guidance_cfg >= 0.0 ? _fibo2_guidance_cfg
                                              : _guidance_scale;
  const bool want_cfg = g > 1.0;
  bool cfg = want_cfg && !neg.empty() && n_neg > 0;
  if (want_cfg && !cfg) {
    session()->warn(fmt(
        "GenerateImageStage('{}'): Fibo 2.0 guidance {} wants a negative "
        "conditioning (the diffusion-conditioner emits one on oport1 when it "
        "is wired -- connect it to iport1); running WITHOUT guidance",
        this->id(), g));
  }
  if (cfg && !_fibo2_dit->set_text(neg, n_neg, &terr, 1)) {
    session()->warn(fmt(
        "GenerateImageStage('{}'): the Fibo 2.0 negative conditioning: {}",
        this->id(), terr));
    return {};
  }
  // A run with no negative drops the last one's text.
  if (!cfg) { _fibo2_dit->clear_text(1); }

  // ---- the pictures to edit, as patch rows after the image's ----------
  std::vector<std::pair<int, int>> grids;
  std::vector<const RefLatent*> used;
  for (const RefLatent& r : refs) {
    if (r.empty()) { continue; }
    if (r.c != PD) {
      session()->warn(fmt(
          "GenerateImageStage('{}'): a reference latent has {} channels, the "
          "Fibo 2.0 DiT takes {} (FLUX.2's VAE through vae-encode); ignoring "
          "it", this->id(), r.c, PD));
      continue;
    }
    if ((int)used.size() >= genai::fibo2::kMaxImages) {
      session()->warn(fmt(
          "GenerateImageStage('{}'): Fibo 2.0 edits at most {} pictures; "
          "ignoring the rest", this->id(), genai::fibo2::kMaxImages));
      break;
    }
    grids.emplace_back(r.h, r.w);
    used.push_back(&r);
  }
  // The negative's sequence: the same image and pictures, its own gist.
  genai::fibo2::Layout lay, lay_neg;
  {
    std::string lerr;
    if (!genai::fibo2::build_layout(gh, gw, grids, _fibo2_dit->gist_count(),
                                    dcfg.max_gist, &lay, &lerr) ||
        (cfg && !genai::fibo2::build_layout(gh, gw, grids,
                                            _fibo2_dit->gist_count(1),
                                            dcfg.max_gist, &lay_neg,
                                            &lerr))) {
      session()->warn(fmt("GenerateImageStage('{}'): Fibo 2.0 layout: {}",
                          this->id(), lerr));
      return {};
    }
  }
  SharedBuffer latbuf =
      mc->make_shared_buffer((std::size_t)lay.refiner_len() * PD * 2);
  if (latbuf.empty()) { return {}; }
  {
    auto* d = static_cast<std::uint16_t*>(latbuf.contents()) +
              (std::size_t)img_seq * PD;
    for (const RefLatent* r : used) {
      for (int y = 0; y < r->h; ++y) {
        for (int x = 0; x < r->w; ++x) {
          const std::size_t t = (std::size_t)y * r->w + x;
          for (int c = 0; c < PD; ++c) {
            d[t * PD + c] = f32_to_bf16_(
                r->chw[((std::size_t)c * r->h + y) * r->w + x]);
          }
        }
      }
      d += (std::size_t)r->h * r->w * PD;
    }
  }
  if (!used.empty()) {
    session()->info(fmt(
        "GenerateImageStage('{}'): Fibo 2.0 editing {} picture(s)",
        this->id(), used.size()));
  }

  // ---- the schedule ------------------------------------------------------
  const int S = _scheduler_latched && _scheduler_spec.steps > 0
                    ? _scheduler_spec.steps
                : _steps_set ? _steps
                : want_cfg   ? _fibo2_guided_steps
                             : _fibo2_steps;
  if (S <= 0) { return {}; }
  std::vector<float> sig((std::size_t)S + 1, 0.0f);
  double mu = 0.0;
  {
#pragma clang fp contract(off)
    const double m = (_fibo2_max_shift - _fibo2_base_shift) /
                     (double)(_fibo2_max_seq - _fibo2_base_seq);
    const double b = _fibo2_base_shift - m * (double)_fibo2_base_seq;
    mu = (double)img_seq * m + b;
    const float e = (float)std::exp(mu);
    // np.linspace(1.0, 1 / S, S) in f64 -- start + k * step, the last
    // node the stop itself -- then .astype(f32).
    const double lo = 1.0 / (double)S;
    const double step = S > 1 ? (lo - 1.0) / (double)(S - 1) : 0.0;
    for (int k = 0; k < S; ++k) {
      const double v = (k == S - 1 && S > 1) ? lo : 1.0 + (double)k * step;
      const float t = (float)v;
      const float q = 1.0f / t - 1.0f;
      sig[(std::size_t)k] = e / (e + q);
    }
  }

  // ---- the noise ---------------------------------------------------------
  std::vector<float> x((std::size_t)img_seq * PD);
  {
    genai::TorchCpuRandn rng((std::uint64_t)_seed);
    const std::vector<float> z = rng.randn(x.size());
    if (z.size() != x.size()) { return {}; }
    genai::fibo2::pack_latents(z.data(), LC, 2 * gh, 2 * gw, dcfg.patch,
                               x.data());
  }
  if (init_packed != nullptr && init_packed->size() == x.size()) {
    x = *init_packed;                  // supplied (repro / golden)
  }

  UiProgress bar = session()->open_progress("denoise");
  DenoiseProgress prog(&bar, S, cfg ? 2 : 1);
  ScopedBlockProgress<std::remove_reference_t<decltype(*_fibo2_dit)>>
      prog_guard(_fibo2_dit.get(), prog);

  // ---- how much of the stack this run may KEEP --------------------------
  if (_fibo2_dit->streaming()) {
    const std::size_t peak =
        flux2_decode_peak_(gen_w, gen_h, _vae_base > 0 ? _vae_base : 128);
    const auto mb = mc->memory_budget();
    const bool decode_runs_beside_us =
        mb.recommended == 0 || (mb.fits(peak) && mb.fits_physical(peak));
    _fibo2_dit->set_residency_reserve(decode_runs_beside_us ? peak : 0);
    _fibo2_dit->set_residency_schedule(S);
  }

  bool ane_checked = false;
  X0Tap tap;
  // Patch rows -> [128, gh, gw]: FLUX.2's patchified latent, channel f
  // = c * 4 + dy * 2 + dx, which vae-decode takes as it comes.
  auto unpack = [&](const std::vector<float>& pk) {
    std::vector<float> latent((std::size_t)PD * gh * gw);
    for (int i = 0; i < gh; ++i) {
      for (int j = 0; j < gw; ++j) {
        const std::size_t t = (std::size_t)i * gw + j;
        for (int c = 0; c < PD; ++c) {
          latent[((std::size_t)c * gh + i) * gw + j] = pk[t * PD + c];
        }
      }
    }
    return latent;
  };

  // The forward's activations at the geometry this edit really has. An
  // edit graph takes its size from the reference, so declare_resources()
  // booked a 1024^2 edit; this is the real one (MEASURED: one forward's
  // peak allocation within 0.1% of the figure,
  // fibo2_dit.a_forward_allocates_what_the_plan_books). revise_scratch()
  // refuses to create: a no-op with no plan.
  if (auto* mgr = session()->services()->generative_model_manager()) {
    mgr->revise_scratch(
        fmt("fibo2-dit:{}", this->id())(),
        genai::MetalFibo2Transformer::forward_scratch_bytes(
            dcfg, lay.refiner_len(), n_rows, _fibo2_dit->quant_bits() > 0,
            mc->supports_matrix_cores(), cfg ? n_neg : -1));
  }

  const auto t0 = std::chrono::steady_clock::now();
  for (int k = 0; k < S; ++k) {
    tap.arm(_preview && _preview->due(k + 1, S));
    {
      auto* lb = static_cast<std::uint16_t*>(latbuf.contents());
      for (std::size_t i = 0; i < x.size(); ++i) {
        lb[i] = f32_to_bf16_(x[i]);
      }
    }
    float tstep = 0.0f;
    {
#pragma clang fp contract(off)
      // timesteps = sigmas * 1000; the pipeline hands t / 1000.
      const float tt = sig[(std::size_t)k] * 1000.0f;
      tstep = tt / 1000.0f;
    }
    // One pass: the prompt's (slot 0) or the negative's (slot 1) velocity,
    // read back as f32 (the reference's `.float()`); empty on a failure
    // or a cooperative stop.
    auto pass = [&](int slot, const genai::fibo2::Layout& l) {
      genai::MetalFibo2Transformer::Request req;
      req.latents = &latbuf;
      req.layout = &l;
      req.timestep = tstep;
      req.text_slot = slot;
      std::string e;
      SharedBuffer v = _fibo2_dit->forward(req, &e);
      prog.end_forward();
      if (!ane_checked && _fibo2_dit->ane_attempted()) {
        ane_checked = true;
        if (auto* mgr = session()->services()->generative_model_manager()) {
          mgr->revise_scratch(
              "coreml:" + ane_claim_label_(),
              _fibo2_dit->ane_armed()
                  ? genai::MetalFibo2Transformer::ane_runtime_bytes(
                        dcfg, std::max(lay.joint_len(),
                                       cfg ? lay_neg.joint_len() : 0))
                  : 0);
        }
      }
      std::vector<float> out;
      if (v.empty()) {
        // A cooperative stop comes back empty with NO reason.
        if (!e.empty()) {
          session()->warn(fmt("GenerateImageStage('{}'): Fibo 2.0 forward "
                              "failed: {}", this->id(), e));
        }
        return out;
      }
      const auto* vp = static_cast<const std::uint16_t*>(v.contents());
      out.resize(x.size());
      for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = bf16_to_f32_(vp[i]);
      }
      return out;
    };
    std::vector<float> vel = pass(0, lay);
    if (vel.empty()) { return {}; }
    if (cfg) {
      const std::vector<float> vu = pass(1, lay_neg);
      if (vu.empty()) { return {}; }
      {
#pragma clang fp contract(off)
        // torch's f32 scalar: the Python float narrowed to the tensor's
        // type, then sub, mul and add, each rounded.
        const float gs = (float)g;
        for (std::size_t i = 0; i < vel.size(); ++i) {
          vel[i] = vu[i] + gs * (vel[i] - vu[i]);
        }
      }
    }
    tap.take(x, sig[(std::size_t)k], vel);
    {
#pragma clang fp contract(off)
      const float dt = sig[(std::size_t)k + 1] - sig[(std::size_t)k];
      for (std::size_t i = 0; i < x.size(); ++i) { x[i] += dt * vel[i]; }
    }
    if (emit_step) { emit_step(unpack(x)); }
    if (tap.ready()) {
      // taef2 reads FLUX.2's latent unpatchified and NOT un-normalized --
      // the space this one is in.
      std::vector<float> lat((std::size_t)LC * 2 * gh * 2 * gw);
      genai::fibo2::unpack_latents(tap.x0.data(), LC, 2 * gh, 2 * gw,
                                   dcfg.patch, lat.data());
      _preview->submit(std::move(lat), LC, 1, 2 * gh, 2 * gw, k + 1, S);
    }
    prog.end_step(k);
  }
  const double secs = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();
  session()->info(fmt(
      "GenerateImageStage('{}'): Fibo 2.0 latent generated in {:.2f}s ({} "
      "steps, mu {:.3f}, {}, {} text rows -> {} gist, {} image + {} context "
      "rows)", this->id(), secs, S, mu,
      cfg ? fmt("guidance {} against {} negative rows -> {} gist", g, n_neg,
                lay_neg.gist)()
          : std::string("no guidance"),
      n_rows, lay.gist, img_seq, lay.ctx_tokens));
  return unpack(x);
}

// FLUX.3 Image Klein: the flow map's own loop.
//
// Everything below is the reference sampler (BFL's sampling.py, which
// diffusers' modular pipeline reproduces), to the bit where a bit
// matters:
//   * the schedule t_k = a s / (1 + (a - 1) s) over s = linspace(1, 0,
//     N + 1), in f32 exactly as torch forms it (flux3::schedule);
//   * the noise is torch.randn((h*w, 128)) from a CPU generator seeded
//     with the seed -- TOKEN-major, rounded to bf16 -- so a seed gives
//     the reference's own picture;
//   * each step x += (t_{k+1} - t_k) v(x, t_k, t_{k+1}): a flow map
//     takes the time it jumps TO, which is why this is not FlowSampler.
// The state stays f32 between bf16 forwards; the reference rounds it to
// bf16 every step, which is measurably worse against its own f32 run.
std::vector<float>
GenerateImageStage::generate_flux3_(
    const metal_compute::SharedBuffer& text, int n_rows, int gen_h,
    int gen_w, const std::vector<RefLatent>& refs,
    const std::vector<float>* init_packed,
    const std::function<void(const std::vector<float>&)>& emit_step) const
{
  auto* mc = session()->services()->metal_compute();
  using metal_compute::SharedBuffer;
  if (mc == nullptr || !_flux3_dit) { return {}; }
  const auto& dcfg = _flux3_dit->config();
  const int IC = dcfg.in_channels;          // 128
  // THE UNDISTILLED BASE IS NOT A FLOW MAP: it samples 50 steps with CFG
  // against an empty-prompt negative, and run through this loop it would
  // make a picture -- the wrong one -- without a word. Refused until that
  // loop exists and has a reference to be checked against.
  if (!dcfg.flow_map) {
    session()->error(fmt(
        "GenerateImageStage('{}'): this FLUX.3 checkpoint is not a flow map "
        "(FLUX.3 Image Klein Base?) -- it needs a CFG sampler this stage does "
        "not run for FLUX.3. Use the distilled flux-3-image-klein",
        this->id()));
    return {};
  }
  if ((gen_h % 16) != 0 || (gen_w % 16) != 0 || gen_h <= 0 || gen_w <= 0) {
    session()->warn(fmt(
        "GenerateImageStage('{}'): FLUX.3 needs a size that is a multiple "
        "of 16 (got {}x{})", this->id(), gen_w, gen_h));
    return {};
  }
  const int gh = gen_h / 16, gw = gen_w / 16;
  const int img_seq = gh * gw;
  if (n_rows <= 0 ||
      text.byte_size() < (std::size_t)n_rows * dcfg.joint_dim * 2) {
    session()->warn(fmt(
        "GenerateImageStage('{}'): the FLUX.3 DiT wants [rows, {}] bf16 "
        "conditioning (the diffusion-conditioner's flux3 family); got {} "
        "bytes for {} rows", this->id(), dcfg.joint_dim, text.byte_size(),
        n_rows));
    return {};
  }

  // ---- the references: token-major bf16, in order ----------------------
  //
  // vae-encode hands over FLUX.2's normalized, patchified latent
  // [128, h, w]; the DiT takes it token-major, row r * w + c, and gives
  // reference n the rope coordinate l = -(1 + n).
  std::vector<SharedBuffer> rbufs;
  std::vector<genai::MetalFlux3Transformer::Ref> rreq;
  rbufs.reserve(refs.size());
  for (const RefLatent& r : refs) {
    if (r.empty()) { continue; }
    if (r.c != dcfg.cond_in) {
      session()->warn(fmt(
          "GenerateImageStage('{}'): a reference latent has {} channels, the "
          "FLUX.3 DiT takes {} (FLUX.2's VAE through vae-encode); ignoring it",
          this->id(), r.c, dcfg.cond_in));
      continue;
    }
    const std::size_t n = (std::size_t)r.h * r.w;
    SharedBuffer b = mc->make_shared_buffer(n * r.c * 2);
    if (b.empty()) { return {}; }
    auto* d = static_cast<std::uint16_t*>(b.contents());
    for (int y = 0; y < r.h; ++y) {
      for (int x = 0; x < r.w; ++x) {
        const std::size_t t = (std::size_t)y * r.w + x;
        for (int c = 0; c < r.c; ++c) {
          d[t * r.c + c] =
              f32_to_bf16_(r.chw[((std::size_t)c * r.h + y) * r.w + x]);
        }
      }
    }
    rbufs.push_back(std::move(b));
  }
  {
    std::size_t bi = 0;
    for (const RefLatent& r : refs) {
      if (r.empty() || r.c != dcfg.cond_in) { continue; }
      rreq.push_back({&rbufs[bi++], r.h, r.w});
    }
  }
  if (!rreq.empty()) {
    session()->info(fmt(
        "GenerateImageStage('{}'): FLUX.3 conditioning on {} reference "
        "image(s)", this->id(), rreq.size()));
  }

  // ---- the schedule ----------------------------------------------------
  //
  // A scheduler-select beat or `steps` sets the COUNT; the curve is the
  // checkpoint's own, which no scheduler here spells.
  const int S = _scheduler_latched && _scheduler_spec.steps > 0
                    ? _scheduler_spec.steps
                    : (_steps_set ? _steps : _flux3_steps);
  const double alpha =
      _flux3_alpha_cfg > 0.0 ? _flux3_alpha_cfg : _flux3_alpha;
  const std::vector<float> sched = genai::flux3::schedule(S, (float)alpha);
  if ((int)sched.size() != S + 1) { return {}; }

  // ---- the noise -------------------------------------------------------
  std::vector<float> x((std::size_t)img_seq * IC);
  {
    genai::TorchCpuRandn rng((std::uint64_t)_seed);
    const std::vector<float> z = rng.randn(x.size());
    if (z.size() != x.size()) { return {}; }
    for (std::size_t i = 0; i < x.size(); ++i) {
      x[i] = bf16_to_f32_(f32_to_bf16_(z[i]));
    }
  }
  if (init_packed != nullptr && init_packed->size() == x.size()) {
    x = *init_packed;                  // supplied (repro / golden)
  }
  SharedBuffer latbuf = mc->make_shared_buffer(x.size() * 2);
  if (latbuf.empty()) { return {}; }

  UiProgress bar = session()->open_progress("denoise");
  DenoiseProgress prog(&bar, S, 1);
  ScopedBlockProgress<std::remove_reference_t<decltype(*_flux3_dit)>>
      prog_guard(_flux3_dit.get(), prog);

  // ---- how much of the stack this run may KEEP --------------------------
  //
  // BlockResidency refuses every admission until a reserve is DECLARED;
  // declaring nothing reads as a checkpoint re-read every step.
  if (_flux3_dit->streaming()) {
    const std::size_t peak =
        flux2_decode_peak_(gen_w, gen_h, _vae_base > 0 ? _vae_base : 128);
    const auto mb = mc->memory_budget();
    const bool decode_runs_beside_us =
        mb.recommended == 0 || (mb.fits(peak) && mb.fits_physical(peak));
    _flux3_dit->set_residency_reserve(decode_runs_beside_us ? peak : 0);
    _flux3_dit->set_residency_schedule(S);
  }

  bool ane_checked = false;
  X0Tap tap;
  auto unpack = [&](const std::vector<float>& pk) {
    std::vector<float> latent((std::size_t)IC * gh * gw);
    for (int i = 0; i < gh; ++i) {
      for (int j = 0; j < gw; ++j) {
        const std::size_t t = (std::size_t)i * gw + j;
        for (int c = 0; c < IC; ++c) {
          latent[((std::size_t)c * gh + i) * gw + j] = pk[t * IC + c];
        }
      }
    }
    return latent;
  };

  const auto t0 = std::chrono::steady_clock::now();
  for (int k = 0; k < S; ++k) {
    tap.arm(_preview && _preview->due(k + 1, S));
    {
      auto* lb = static_cast<std::uint16_t*>(latbuf.contents());
      for (std::size_t i = 0; i < x.size(); ++i) {
        lb[i] = f32_to_bf16_(x[i]);
      }
    }
    genai::MetalFlux3Transformer::Request req;
    req.latents = &latbuf;
    req.grid_h = gh;
    req.grid_w = gw;
    req.text = &text;
    req.text_rows = n_rows;
    req.refs = rreq;
    req.t = sched[(std::size_t)k];
    req.t_next = sched[(std::size_t)k + 1];
    std::string e;
    SharedBuffer v = _flux3_dit->forward(req, &e);
    prog.end_forward();
    // After the first forward, where the ANE tier arms or declines: book
    // what is HELD. revise_scratch() refuses to create, so this is a
    // no-op when the plan booked nothing.
    if (!ane_checked && _flux3_dit->ane_attempted()) {
      ane_checked = true;
      if (auto* mgr = session()->services()->generative_model_manager()) {
        mgr->revise_scratch(
            "coreml:" + ane_claim_label_(),
            _flux3_dit->ane_armed()
                ? genai::MetalFlux3Transformer::ane_runtime_bytes(
                      dcfg, n_rows + img_seq)
                : 0);
      }
    }
    if (v.empty()) {
      // A cooperative stop comes back empty with NO reason.
      if (!e.empty()) {
        session()->warn(fmt("GenerateImageStage('{}'): FLUX.3 forward "
                            "failed: {}", this->id(), e));
      }
      return {};
    }
    const auto* vp = static_cast<const std::uint16_t*>(v.contents());
    std::vector<float> vel(x.size());
    for (std::size_t i = 0; i < vel.size(); ++i) {
      vel[i] = bf16_to_f32_(vp[i]);
    }
    tap.take(x, sched[(std::size_t)k], vel);
    const float dt = sched[(std::size_t)k + 1] - sched[(std::size_t)k];
    for (std::size_t i = 0; i < x.size(); ++i) { x[i] += dt * vel[i]; }
    if (emit_step) { emit_step(unpack(x)); }
    if (tap.ready()) {
      // taef2 reads FLUX.2's latent unpatchified and NOT un-normalized --
      // the same space this one is in (see generate_flux2_).
      const std::vector<float> z = unpack(tap.x0);
      const int L = IC / 4, h8 = 2 * gh, w8 = 2 * gw;
      std::vector<float> lat((std::size_t)L * h8 * w8);
      for (int cc = 0; cc < IC; ++cc) {
        const int c = cc / 4, ph = (cc % 4) / 2, pw = cc % 2;
        for (int yy = 0; yy < gh; ++yy) {
          for (int xx = 0; xx < gw; ++xx) {
            lat[((std::size_t)c * h8 + 2 * yy + ph) * w8 + 2 * xx + pw] =
                z[((std::size_t)cc * gh + yy) * gw + xx];
          }
        }
      }
      _preview->submit(std::move(lat), L, 1, h8, w8, k + 1, S);
    }
    prog.end_step(k);
  }
  const double secs = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();
  session()->info(fmt(
      "GenerateImageStage('{}'): FLUX.3 latent generated in {:.2f}s ({} "
      "steps, alpha {:.2f}, {} text + {} image + {} reference rows)",
      this->id(), secs, S, alpha, n_rows, img_seq,
      [&] {
        int n = 0;
        for (const auto& r : rreq) { n += r.grid_h * r.grid_w; }
        return n;
      }()));
  return unpack(x);
}

// MOSS-Image: the reference pipeline's loop (pipeline_moss_image.py,
// MOSSImagePipeline.__call__), to the bit where a bit matters:
//   * the schedule: a shift-5 grid snapped to the training grid's
//     nearest nodes (moss_image::sigmas, bit-exact against the reference
//     on this box), the DiT handed sigma * 1000 in f32;
//   * the noise: torch.randn((1, 128, H/16, W/16)) from a CPU generator
//     seeded with the seed, f32 and CHANNEL-major -- so a seed gives the
//     reference's own picture;
//   * CFG in f32: uncond + g * (cond - uncond), the uncond branch the
//     negative prompt (the EMPTY one by default) run WITHOUT block 22;
//   * Euler in f32: x += (sigma_next - sigma) * v.
// The state stays f32 between bf16 forwards, as the reference keeps it.
std::vector<float>
GenerateImageStage::generate_moss_image_(
    const metal_compute::SharedBuffer& txt_pos, int n_real,
    const metal_compute::SharedBuffer& txt_neg, int n_real_neg, int gen_h,
    int gen_w, const std::vector<float>* init_packed,
    const std::function<void(const std::vector<float>&)>& emit_step) const
{
#pragma clang fp contract(off)
  auto* mc = session()->services()->metal_compute();
  using metal_compute::SharedBuffer;
  if (mc == nullptr || !_moss_dit) { return {}; }
  const auto& dcfg = _moss_dit->config();
  const int C = dcfg.latent_channels;          // 128
  const int PPT = genai::moss_image::kPixelsPerToken;
  if ((gen_h % PPT) != 0 || (gen_w % PPT) != 0 || gen_h <= 0 ||
      gen_w <= 0) {
    session()->warn(fmt(
        "GenerateImageStage('{}'): MOSS-Image needs a size that is a "
        "multiple of {} (got {}x{})", this->id(), PPT, gen_w, gen_h));
    return {};
  }
  const int gh = gen_h / PPT, gw = gen_w / PPT;
  const int IMG = gh * gw;
  if (n_real <= 0 ||
      txt_pos.byte_size() < (std::size_t)n_real * dcfg.text_dim * 2) {
    session()->warn(fmt(
        "GenerateImageStage('{}'): the MOSS-Image DiT wants [rows, {}] bf16 "
        "conditioning (the diffusion-conditioner's moss-image family); got "
        "{} bytes for {} rows", this->id(), dcfg.text_dim,
        txt_pos.byte_size(), n_real));
    return {};
  }
  if (!_ref.empty()) {
    session()->warn(fmt(
        "GenerateImageStage('{}'): a reference latent is wired but "
        "MOSS-Image is text-to-image only -- it is ignored", this->id()));
  }

  // ---- the schedule ----------------------------------------------------
  // A scheduler-select beat or `steps` sets the COUNT; the curve is the
  // checkpoint's own, which no scheduler here spells.
  const int S_req = _scheduler_latched && _scheduler_spec.steps > 0
                        ? _scheduler_spec.steps
                        : (_steps_set ? _steps : 50);
  const double shift = _moss_shift_cfg > 0.0 ? _moss_shift_cfg : _moss_shift;
  const std::vector<float> sig = genai::moss_image::sigmas(
      S_req, shift, _moss_train_shift, _moss_train_steps);
  if (sig.size() < 2) { return {}; }
  const int S = (int)sig.size() - 1;

  // ---- CFG -------------------------------------------------------------
  const double g = _moss_guidance_cfg >= 0.0 ? _moss_guidance_cfg : 3.0;
  const bool want_cfg = g != 1.0;
  const bool cfg = want_cfg && !txt_neg.empty() && n_real_neg > 0;
  if (want_cfg && !cfg) {
    session()->warn(fmt(
        "GenerateImageStage('{}'): MOSS-Image guidance {} wants a negative "
        "conditioning (the conditioner emits the empty prompt's on oport1 "
        "-- wire it to iport1); running WITHOUT guidance", this->id(), g));
  }
  std::vector<int> skip = _moss_skip_set ? _moss_skip_cfg
                                         : std::vector<int>{22};
  skip.erase(std::remove_if(skip.begin(), skip.end(),
                            [&](int L) { return L >= dcfg.n_layers; }),
             skip.end());
  const float gs = (float)g;
  // SKIP-LAYER GUIDANCE on the positive branch, a third forward, inside
  // [slg_start, slg_stop) of the denoise -- the SAME window that gates the
  // negative branch's skip (the reference's `skip_active`).
  std::vector<int> slg = _moss_slg_layers;
  slg.erase(std::remove_if(slg.begin(), slg.end(),
                           [&](int L) { return L >= dcfg.n_layers; }),
            slg.end());
  const bool slg_on = !slg.empty() && _moss_slg_scale > 0.0;
  const float ss = (float)_moss_slg_scale;

  // ---- the noise -------------------------------------------------------
  std::vector<float> x((std::size_t)IMG * C);
  {
    genai::TorchCpuRandn rng((std::uint64_t)_seed);
    const std::vector<float> z = rng.randn(x.size());
    if (z.size() != x.size()) { return {}; }
    // [1, C, gh, gw] as drawn -> token-major [IMG, C].
    for (int c = 0; c < C; ++c) {
      for (int p = 0; p < IMG; ++p) {
        x[(std::size_t)p * C + c] = z[(std::size_t)c * IMG + p];
      }
    }
  }
  if (init_packed != nullptr && init_packed->size() == x.size()) {
    x = *init_packed;                  // supplied (repro / golden)
  }
  SharedBuffer latbuf = mc->make_shared_buffer(x.size() * 2);
  if (latbuf.empty()) { return {}; }

  UiProgress bar = session()->open_progress("denoise");
  DenoiseProgress prog(&bar, S, (cfg ? 2 : 1) + (slg_on ? 1 : 0));
  ScopedBlockProgress<std::remove_reference_t<decltype(*_moss_dit)>>
      prog_guard(_moss_dit.get(), prog);

  // ---- how much of the stack this run may KEEP --------------------------
  if (_moss_dit->streaming()) {
    const std::size_t peak =
        flux2_decode_peak_(gen_w, gen_h, _vae_base > 0 ? _vae_base : 128);
    const auto mb = mc->memory_budget();
    const bool decode_runs_beside_us =
        mb.recommended == 0 || (mb.fits(peak) && mb.fits_physical(peak));
    _moss_dit->set_residency_reserve(decode_runs_beside_us ? peak : 0);
    _moss_dit->set_residency_schedule(S);
  }

  bool ane_checked = false;
  auto run = [&](const SharedBuffer& txt, int rows,
                 const std::vector<int>& sk, float t,
                 std::vector<float>* out) -> bool {
    genai::MetalMossImageTransformer::Request req;
    req.latents = &latbuf;
    req.grid_h = gh;
    req.grid_w = gw;
    req.text = &txt;
    req.text_rows = rows;
    req.timestep = t;
    req.skip_layers = sk;
    std::string e;
    SharedBuffer v = _moss_dit->forward(req, &e);
    prog.end_forward();
    // After the first forward, where the ANE tiers arm or decline: book
    // what is HELD (revise_scratch refuses to create, so a plan that
    // booked nothing stays so).
    if (!ane_checked && _moss_dit->ane_attempted()) {
      ane_checked = true;
      if (auto* mgr = session()->services()->generative_model_manager()) {
        genai::MetalMossImageTransformer::Config c = dcfg;
        c.ane_qkv = _moss_dit->ane_qkv_armed();
        mgr->revise_scratch(
            "coreml:" + ane_claim_label_(),
            _moss_dit->ane_armed()
                ? genai::MetalMossImageTransformer::ane_runtime_bytes(
                      c, rows + IMG)
                : 0);
      }
    }
    if (v.empty()) {
      // A cooperative stop comes back empty with NO reason.
      if (!e.empty()) {
        session()->warn(fmt("GenerateImageStage('{}'): MOSS-Image forward "
                            "failed: {}", this->id(), e));
      }
      return false;
    }
    const auto* vp = static_cast<const std::uint16_t*>(v.contents());
    out->resize(x.size());
    for (std::size_t i = 0; i < out->size(); ++i) {
      (*out)[i] = bf16_to_f32_(vp[i]);
    }
    return true;
  };

  X0Tap tap;
  auto unpack = [&](const std::vector<float>& pk) {
    std::vector<float> latent((std::size_t)C * IMG);
    for (int p = 0; p < IMG; ++p) {
      for (int c = 0; c < C; ++c) {
        latent[(std::size_t)c * IMG + p] = pk[(std::size_t)p * C + c];
      }
    }
    return latent;
  };

  const auto t0 = std::chrono::steady_clock::now();
  std::vector<float> vc, vn;
  for (int k = 0; k < S; ++k) {
    tap.arm(_preview && _preview->due(k + 1, S));
    {
      auto* lb = static_cast<std::uint16_t*>(latbuf.contents());
      for (std::size_t i = 0; i < x.size(); ++i) {
        lb[i] = f32_to_bf16_(x[i]);
      }
    }
    const float t = genai::moss_image::timestep(sig[(std::size_t)k]);
    // step / steps in the reference's float arithmetic.
    const double frac = (double)k / (double)S;
    const bool active = _moss_slg_start <= frac && frac < _moss_slg_stop;
    if (!run(txt_pos, n_real, {}, t, &vc)) { return {}; }
    std::vector<float> cond_v;
    if (slg_on && active) { cond_v = vc; }
    if (cfg) {
      if (!run(txt_neg, n_real_neg, active ? skip : std::vector<int>{}, t,
               &vn)) {
        return {};
      }
      for (std::size_t i = 0; i < vc.size(); ++i) {
        const float d = vc[i] - vn[i];
        vc[i] = vn[i] + gs * d;
      }
    }
    if (slg_on && active) {
      std::vector<float> vs;
      if (!run(txt_pos, n_real, slg, t, &vs)) { return {}; }
      if (!cfg) { vc = vs; }
      for (std::size_t i = 0; i < vc.size(); ++i) {
        const float d = cond_v[i] - vs[i];
        vc[i] = vc[i] + ss * d;
      }
    }
    tap.take(x, sig[(std::size_t)k], vc);
    const float dt = sig[(std::size_t)k + 1] - sig[(std::size_t)k];
    for (std::size_t i = 0; i < x.size(); ++i) {
      const float step = dt * vc[i];
      x[i] = x[i] + step;
    }
    if (emit_step) { emit_step(unpack(x)); }
    if (tap.ready()) {
      // taef2 reads FLUX.2's latent unpatchified and NOT un-normalized --
      // the same space this one is in (see generate_flux3_).
      const std::vector<float> z = unpack(tap.x0);
      const int L = C / 4, h8 = 2 * gh, w8 = 2 * gw;
      std::vector<float> lat((std::size_t)L * h8 * w8);
      for (int cc = 0; cc < C; ++cc) {
        const int c = cc / 4, ph = (cc % 4) / 2, pw = cc % 2;
        for (int yy = 0; yy < gh; ++yy) {
          for (int xx = 0; xx < gw; ++xx) {
            lat[((std::size_t)c * h8 + 2 * yy + ph) * w8 + 2 * xx + pw] =
                z[((std::size_t)cc * gh + yy) * gw + xx];
          }
        }
      }
      _preview->submit(std::move(lat), L, 1, h8, w8, k + 1, S);
    }
    prog.end_step(k);
  }
  const double secs = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();
  std::string sk_desc;
  for (int L : skip) {
    sk_desc += (sk_desc.empty() ? "" : ",") + std::to_string(L);
  }
  session()->info(fmt(
      "GenerateImageStage('{}'): MOSS-Image latent generated in {:.2f}s ({} "
      "steps, shift {:.2f}, {} image + {} text rows, {})", this->id(), secs,
      S, shift, IMG, n_real,
      (cfg ? fmt("cfg {} against {} rows, skipping [{}]", g, n_real_neg,
                 sk_desc)()
           : std::string("no cfg")) +
          (slg_on ? fmt(", slg {} on {} block(s) in [{}, {})",
                        _moss_slg_scale, slg.size(), _moss_slg_start,
                        _moss_slg_stop)()
                  : std::string())));
  return unpack(x);
}

std::vector<float>
GenerateImageStage::generate_boogu_(const metal_compute::SharedBuffer& txt_pos,
                                  int n_real,
                                  const metal_compute::SharedBuffer& txt_neg,
                                  int n_real_neg, int gen_h, int gen_w,
                                  const std::vector<float>* init_packed,
                                  const std::vector<RefLatent>& refs,
                                  const std::function<void(
                                      const std::vector<float>&)>& emit_step)
    const
{
  auto* mc = session()->services()->metal_compute();
  using metal_compute::SharedBuffer;
  const int H = gen_h, W = gen_w;
  const int Z = _boogu_dit->config().in_channels;   // 16 latent channels
  const int P = _boogu_dit->config().patch;         // 2
  const int lh = H / 8, lw = W / 8;                 // latent grid (VAE 8x)
  const int gh = lh / P, gw = lw / P;               // patch-token grid
  const int img_seq = gh * gw;
  const int IC = _boogu_dit->config().x_in();       // 64 = 16 * 2 * 2
  const int OC = _boogu_dit->config().out_channels;
  if (img_seq <= 0 || (lh % P) != 0 || (lw % P) != 0) {
    session()->warn(fmt(
        "GenerateImageStage('{}'): {}x{} gives a {}x{} latent, which is not a "
        "whole number of {}x{} patches -- use multiples of 16",
        this->id(), W, H, lw, lh, P, P));
    return {};
  }

  session()->log_debug(fmt(
      "GenerateImageStage('{}'): Boogu conditioning [{}, txt]; {}x{} latent {}x{} "
      "grid {}x{} img_seq {}", this->id(), n_real, W, H, lw, lh, gw, gh,
      img_seq));

  // Schedule. The sampler/scheduler beats win when latched; otherwise the
  // checkpoint default is the distilled 4-step DMD student.
  genai::FlowSamplerSpec samp = _sampler_spec;
  genai::FlowSchedulerSpec sched = _scheduler_spec;
  if (!_sampler_latched) { samp.method = "dmd"; samp.conditioning_sigma = 0.0; }
  if (!_scheduler_latched) {
    sched.type = "boogu_v1";
    sched.steps = _steps > 0 ? _steps : 4;
    sched.base_shift = 0.5; sched.max_shift = 1.15;
    sched.base_seq = 256; sched.max_seq = 4096;
    sched.seq_len = 4096;
  }
  sched.img_seq_len = img_seq;
  const bool dmd = (samp.method == "dmd");
  const int S = sched.steps < 1 ? 1 : sched.steps;

  // How much room the streamed DiT must leave clear to keep a block
  // resident -- and the reason this is HERE rather than only in
  // free_boogu_dit_for_decode_, which is where it used to be set.
  //
  // That free runs AFTER the denoise, so on the first prompt no reserve
  // had ever been declared while the blocks were being decided, and
  // BlockResidency keeps growth OFF until one is (deliberately: a model
  // that has not been told what its peers cost should not grow against a
  // guess). Boogu therefore streamed its whole stack for every step of
  // the generation the reserve was meant to inform.
  //
  // Same rule as FLUX.2 and Krea-2, against Boogu's own decode peak:
  // reserve the VAE decode only where it would actually run beside us.
  // Where it would not, that free drops the whole DiT first, which is
  // strictly more than a reserve could hold clear -- so reserving it
  // there protects a coexistence that never happens and costs every
  // block of every step.
  if (_vae_base > 0 && _boogu_dit) {
    const std::size_t peak = flux2_decode_peak_(W, H, _vae_base);
    const auto mb = mc->memory_budget();
    // No budget to read -> the free bails out too, so the DiT survives.
    const bool decode_runs_beside_us =
        mb.recommended == 0 || (mb.fits(peak) && mb.fits_physical(peak));
    _boogu_dit->set_residency_reserve(decode_runs_beside_us ? peak : 0);
    _boogu_dit->set_residency_schedule(S);
    session()->log_debug(fmt(
        "GenerateImageStage('{}'): Boogu residency reserve {} MB -- the "
        "{}x{} vae-decode wants {} MB and {}",
        this->id(), (decode_runs_beside_us ? peak : 0) >> 20, W, H,
        peak >> 20,
        decode_runs_beside_us ? "fits beside the DiT"
                              : "does not, so the DiT is freed before it"));
  }
  // DMD builds its own linspace(conditioning_sigma, 1, S+1)[:-1]; the Euler
  // path takes the shifted schedule (ascending, terminal 1.0).
  std::vector<double> sig;
  if (dmd) {
    sig.resize((std::size_t)S);
    const double c0 = samp.conditioning_sigma;
    for (int i = 0; i < S; ++i) {
      sig[(std::size_t)i] = c0 + (1.0 - c0) * (double)i / (double)S;
    }
  } else {
    if (sched.type != "boogu_v1") {
      session()->warn(fmt(
          "GenerateImageStage('{}'): scheduler '{}' has a DESCENDING sigma "
          "convention, which Boogu's does not share; using boogu_v1",
          this->id(), sched.type));
      sched.type = "boogu_v1";
    }
    sig = sched.sigmas();
  }

  // Packed latents [img_seq, IC]: a supplied init (repro / golden) or noise.
  std::vector<float> packed((std::size_t)img_seq * IC);
  if (init_packed != nullptr && init_packed->size() == packed.size()) {
    packed = *init_packed;
  } else {
    std::mt19937_64 rng(_seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : packed) { v = nd(rng); }
  }
  SharedBuffer latbuf = mc->make_shared_buffer((std::size_t)img_seq * IC * 2);
  if (latbuf.empty()) { return {}; }

  // Reference conditioning: vae-encode hands us [16, rlh, rlw] channel-first;
  // pack it 2x2 into token-major [rseq, IC] bf16, the same patchify the target
  // uses. Boogu-Edit takes ONE reference (the model card says so outright) but
  // the DiT carries image_index_embedding rows for up to max_ref_images, so
  // extra references are passed through rather than dropped.
  std::vector<genai::MetalBooguTransformer::RefImage> bri;
  for (const auto& r : refs) {
    if (r.empty()) { continue; }
    if (r.c != Z || (r.h % P) != 0 || (r.w % P) != 0) {
      session()->warn(fmt(
          "GenerateImageStage('{}'): reference latent [{}, {}, {}] must be "
          "{}-channel with H/W divisible by {}; ignoring", this->id(), r.c,
          r.h, r.w, Z, P));
      continue;
    }
    if ((int)bri.size() >= _boogu_dit->config().max_ref_images) {
      session()->warn(fmt(
          "GenerateImageStage('{}'): more than {} reference images; ignoring the "
          "rest", this->id(), _boogu_dit->config().max_ref_images));
      break;
    }
    const int rgh = r.h / P, rgw = r.w / P, rseq = rgh * rgw;
    SharedBuffer rb = mc->make_shared_buffer((std::size_t)rseq * IC * 2);
    if (rb.empty()) { continue; }
    auto* d = static_cast<std::uint16_t*>(rb.contents());
    std::memset(d, 0, rb.byte_size());
    for (int cc = 0; cc < Z; ++cc) {
      for (int y = 0; y < r.h; ++y) {
        for (int x = 0; x < r.w; ++x) {
          const int a = y / P, ph = y % P, bcol = x / P, pw = x % P;
          const std::size_t t = (std::size_t)a * rgw + bcol;
          // Boogu packs "c (h p1) (w p2) -> (h w) (p1 p2 c)": inside a token
          // the CHANNEL is the fastest axis and the patch offsets the slower
          // ones. (Getting this transposed still round-trips -- pack and
          // unpack cancel -- but feeds x_embedder a permuted 64-vector, which
          // shows up as a 2x2 lattice over an otherwise correct image.)
          d[t * IC + ((std::size_t)ph * P + pw) * Z + cc] =
              f32_to_bf16_(r.chw[((std::size_t)cc * r.h + y) * r.w + x]);
        }
      }
    }
    genai::MetalBooguTransformer::RefImage img;
    img.latents = std::move(rb);
    img.seq = rseq; img.grid_h = r.h; img.grid_w = r.w;
    bri.push_back(std::move(img));
  }
  if (!bri.empty()) {
    session()->info(fmt(
        "GenerateImageStage('{}'): Boogu-Image editing on {} reference image(s)",
        this->id(), bri.size()));
  }

  // Classifier-free guidance (Base/Edit only -- the Turbo students are
  // distilled to guidance 1 and the DMD path forbids it outright). Boogu's
  // reference pipeline actually runs a THREE-way guidance (text, image and
  // empty-instruction scales); this is the single-negative reduction the other
  // families here use, i.e. v = v_neg + scale * (v_pos - v_neg).
  const bool cfg = !dmd && !txt_neg.empty() && n_real_neg > 0 &&
                   _guidance_scale != 1.0;
  if (dmd && _guidance_scale != 1.0) {
    session()->warn(fmt(
        "GenerateImageStage('{}'): the DMD student is distilled to guidance 1; "
        "ignoring guidance_scale {}", this->id(), _guidance_scale));
  }
  const float gscale = (float)_guidance_scale;

  bool dit_ok = true;
  // Opened before the denoise callable so the per-block hook is
  // live for the very first forward.
  UiProgress bar = session()->open_progress("denoise");
  DenoiseProgress prog(&bar, S, cfg ? 2 : 1);
  ScopedBlockProgress<std::remove_reference_t<decltype(*_boogu_dit)>>
      prog_guard(_boogu_dit.get(), prog);
  auto velocity = [&](const std::vector<float>& cand,
                      double sigma) -> std::vector<float> {
    auto* lb = static_cast<std::uint16_t*>(latbuf.contents());
    for (std::size_t k = 0; k < cand.size(); ++k) {
      lb[k] = f32_to_bf16_(cand[k]);
    }
    SharedBuffer vel = _boogu_dit->forward_dit(txt_pos, n_real, latbuf, img_seq,
                                               lh, lw, (float)sigma, bri);
    prog.end_forward();
    if (vel.empty()) { dit_ok = false; return {}; }
    const auto* vp = static_cast<const std::uint16_t*>(vel.contents());
    std::vector<float> v((std::size_t)img_seq * OC);
    for (std::size_t k = 0; k < v.size(); ++k) { v[k] = bf16_to_f32_(vp[k]); }
    if (cfg) {
      SharedBuffer vn = _boogu_dit->forward_dit(txt_neg, n_real_neg, latbuf,
                                                img_seq, lh, lw, (float)sigma,
                                                bri);
      prog.end_forward();
      if (vn.empty()) { dit_ok = false; return {}; }
      const auto* np = static_cast<const std::uint16_t*>(vn.contents());
      for (std::size_t k = 0; k < v.size(); ++k) {
        const float neg = bf16_to_f32_(np[k]);
        v[k] = neg + gscale * (v[k] - neg);
      }
    }
    return v;
  };

  // Unpack [img_seq, IC] -> channel-first latent [Z, lh, lw], the inverse of
  // the "(h w) (p1 p2 c)" pack above (channel fastest inside a token).
  auto unpack = [&](const std::vector<float>& pk) {
    std::vector<float> latent((std::size_t)Z * lh * lw);
    for (int c = 0; c < Z; ++c) {
      for (int y = 0; y < lh; ++y) {
        for (int xx = 0; xx < lw; ++xx) {
          const int a = y / P, ph = y % P, b = xx / P, pw = xx % P;
          const std::size_t t = (std::size_t)a * gw + b;
          latent[((std::size_t)c * lh + y) * lw + xx] =
              pk[t * IC + ((std::size_t)ph * P + pw) * Z + c];
        }
      }
    }
    return latent;
  };

  const bool prof = std::getenv("VPIPE_BOOGU_PROFILE") != nullptr;
  double dit_ms = 0.0;
  int dit_calls = 0;
  std::mt19937_64 renoise_rng(samp.seed != 0 ? samp.seed : _seed + 1);
  std::normal_distribution<float> rnd(0.0f, 1.0f);
  const auto gen_t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < S; ++i) {
    const double s = sig[(std::size_t)i];
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<float> v = velocity(packed, s);
    if (prof) {
      dit_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
      ++dit_calls;
    }
    if (!dit_ok) { return {}; }
    if (dmd) {
      // Jump to the x0 prediction, then re-noise back down to the next sigma.
      for (std::size_t k = 0; k < packed.size() && k < v.size(); ++k) {
        packed[k] += (float)((1.0 - s) * (double)v[k]);
      }
      if (i + 1 < S) {
        const double s1 = sig[(std::size_t)i + 1];
        for (auto& x : packed) {
          x = (float)((1.0 - s1) * (double)rnd(renoise_rng) + s1 * (double)x);
        }
      }
    } else {
      const double dt = sig[(std::size_t)i + 1] - s;
      for (std::size_t k = 0; k < packed.size() && k < v.size(); ++k) {
        packed[k] += (float)(dt * (double)v[k]);
      }
    }
    if (emit_step) { emit_step(unpack(packed)); }
    prog.end_step(i);
  }
  const double gen_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - gen_t0).count();
  session()->info(fmt(
      "GenerateImageStage('{}'): latent generated in {:.2f}s ({} {} steps, "
      "{} ms/step)", this->id(), gen_s, S, dmd ? "DMD" : "Euler",
      S ? (long)(gen_s * 1000.0 / S) : 0));
  if (prof) {
    session()->log_normal(fmt(
        "GenerateImageStage('{}'): Boogu DiT {} forward calls, {} ms total, {} "
        "ms/call (txt {} + img {} = {})", this->id(), dit_calls, (long)dit_ms,
        dit_calls ? (long)(dit_ms / dit_calls) : 0, n_real, img_seq,
        n_real + img_seq));
  }
  return unpack(packed);
}

std::vector<float>
GenerateImageStage::generate_qie_(const metal_compute::SharedBuffer& txt_pos,
                               int n_real,
                               const metal_compute::SharedBuffer& txt_neg,
                               int n_real_neg, int gen_h, int gen_w,
                               const std::vector<float>* init_packed,
                               const std::vector<RefLatent>& refs,
                               const std::function<void(
                                   const std::vector<float>&)>& emit_step)
    const
{
  auto* mc = session()->services()->metal_compute();
  using metal_compute::SharedBuffer;
  const int H = gen_h, W = gen_w;
  const int lh = H / 8, lw = W / 8;          // latent H/W (z_dim 16)
  const int gh = H / 16, gw = W / 16;        // 2x2-patch grid
  const int img_seq = gh * gw;
  const int IC = _qie_dit->config().in_channels;   // 64 = 16 * 2 * 2

  // How much the DiT must leave clear when it decides whether to keep a
  // streamed block resident. Its growth must not eat the arena the
  // vae-decode needs right after -- that trades a disk read for an
  // allocation failure.
  //
  // Set HERE rather than at load, because the figure is a function of
  // the geometry and the geometry arrives with the beat.
  //
  // Same rule as the other three families, against this one's own decode
  // peak (free_qie_dit_for_decode_): reserve the VAE decode only where it
  // would actually run beside us. Where it would not, that free drops the
  // whole 20B DiT first, which is strictly more than a reserve could hold
  // clear -- so reserving it there protects a coexistence that never
  // happens, and pays for it by streaming every block of every step.
  //
  // The peak is the Krea-2 shape, not the FLUX.2 one it used to be: this
  // family's VAE is AutoencoderKLQwenImage, whose top-res im2col scratch
  // is [Hout*Wout, 9*base] plus ~50%. The old `base * 7` expression was
  // the FLUX.2 hw-conv activation-pool model, and it read `_vae_base`
  // that nothing in this arm had ever set -- so it ran on the 128
  // default against a checkpoint whose base_dim is 96.
  if (mc != nullptr && _vae_base > 0) {
    const std::size_t peak = qwen_decode_peak_(W, H, _vae_base);
    const auto mb = mc->memory_budget();
    // No budget to read -> the free bails out too, so the DiT survives.
    const bool decode_runs_beside_us =
        mb.recommended == 0 || (mb.fits(peak) && mb.fits_physical(peak));
    _qie_dit->set_residency_reserve(decode_runs_beside_us ? peak : 0);
    session()->log_debug(fmt(
        "GenerateImageStage('{}'): Qwen-Image-Edit residency reserve {} MB "
        "-- the {}x{} vae-decode wants {} MB and {}",
        this->id(), (decode_runs_beside_us ? peak : 0) >> 20, W, H,
        peak >> 20,
        decode_runs_beside_us ? "fits beside the DiT"
                              : "does not, so the DiT is freed before it"));
    // And the RATES for this schedule. Both of BlockResidency's defaults
    // are tuned for a ~30-step run: on a short one the probe would reach
    // full residency only near the end, where nothing is left to use it.
    // After the reserve, because the probe is sized against what is left
    // once the reserve is taken.
    _qie_dit->set_residency_schedule(
        _scheduler_spec.steps > 0 ? _scheduler_spec.steps : 1);
  }

  // `txt_pos` is the diffusion-conditioner's qwen-image-edit conditioning: the
  // image-aware last-hidden [n_real, 3584] bf16, already POST encoder final-norm
  // (the conditioner ran the vision tower + splice + drop-64 + final-RMSNorm).
  session()->log_debug(fmt(
      "GenerateImageStage('{}'): QIE conditioning [{}, txt]; {}x{} grid {}x{} "
      "img_seq {}", this->id(), n_real, W, H, gh, gw, img_seq));

  // norm-preserving true-CFG: use the negative conditioning too and, per step,
  // comb = neg + scale*(pos-neg), then rescale comb to pos's per-token norm.
  const bool cfg = !txt_neg.empty() && n_real_neg > 0 && _guidance_scale != 1.0;

  // Sampler: FlowMatchEuler with the QIE dynamic shift (mu from img_seq, terminal
  // stretch 0.02). Operator-supplied scheduler beats win (skip the override).
  genai::FlowSchedulerSpec sched = _scheduler_spec;
  if (!_scheduler_latched) {
    sched.dynamic_shift  = true;
    sched.base_shift     = 0.5;
    sched.max_shift      = 0.9;
    sched.shift_terminal = 0.02;
    sched.base_seq       = 256;
    sched.max_seq        = 8192;
    sched.num_train      = 1000;
    sched.shift_type     = "exponential";
    sched.img_seq_len    = img_seq;
  } else {
    sched.img_seq_len    = img_seq;
  }
  genai::FlowSampler sampler(_sampler_spec, sched);
  const int S = sampler.steps();

  // Packed latents [img_seq, IC]: a supplied init (repro / golden) or pure noise.
  std::vector<float> packed((std::size_t)img_seq * IC);
  if (init_packed != nullptr && init_packed->size() == packed.size()) {
    packed = *init_packed;
  } else {
    std::mt19937_64 rng(_seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : packed) { v = nd(rng); }
  }
  SharedBuffer latbuf = mc->make_shared_buffer((std::size_t)img_seq * IC * 2);
  if (latbuf.empty()) { return {}; }

  // Reference conditioning: each ref latent arrives channel-first [16, rlh, rlw]
  // (vae-encode output); pack it 2x2 into token-major [rseq, IC] bf16 (the same
  // patchify the generated latent uses) so the DiT embeds it via img_in in its
  // own RoPE frame band. Odd-dim / non-16-channel refs are skipped.
  //
  // The DiT's scale_rope centers each reference's h/w positions at 0 (matching
  // diffusers), so a reference grid SMALLER than the generated grid (gh x gw)
  // covers only the centered sub-region and leaves a visible rectangular "box"
  // at its boundary. The fix is to encode the reference at the OUTPUT resolution
  // (so ref grid == gen grid == full overlap, as the diffusers pipeline does) --
  // we keep the reference at native resolution (re-gridding the latent here just
  // blurs it) and WARN when the grids differ so the user can match them.
  std::vector<genai::MetalQwenImageTransformer::RefImage> ri;
  for (const auto& r : refs) {
    if (r.empty() || r.c != 16 || (r.h % 2) != 0 || (r.w % 2) != 0) {
      if (!r.empty()) {
        session()->warn(fmt(
            "GenerateImageStage('{}'): reference latent [{}, {}, {}] must be "
            "16-channel with even H/W; ignoring", this->id(), r.c, r.h, r.w));
      }
      continue;
    }
    const int rgh = r.h / 2, rgw = r.w / 2, rseq = rgh * rgw;
    if (rgh != gh || rgw != gw) {
      session()->warn(fmt(
          "GenerateImageStage('{}'): reference grid {}x{} != output grid {}x{} -- "
          "the centered reference will cover only part of the output and leave a "
          "rectangular artifact. Encode the reference at the output resolution "
          "(set vae-encode target to {}x{}) to avoid it.",
          this->id(), rgw, rgh, gw, gh, W, H));
    }
    SharedBuffer rb = mc->make_shared_buffer((std::size_t)rseq * IC * 2);
    if (rb.empty()) { continue; }
    auto* d = static_cast<std::uint16_t*>(rb.contents());
    std::memset(d, 0, rb.byte_size());
    for (int cc = 0; cc < 16; ++cc) {
      for (int y = 0; y < r.h; ++y) {
        for (int x = 0; x < r.w; ++x) {
          const int a = y / 2, ph = y % 2, bcol = x / 2, pw = x % 2;
          const std::size_t t = (std::size_t)a * rgw + bcol;
          d[t * IC + (std::size_t)cc * 4 + ph * 2 + pw] =
              f32_to_bf16_(r.chw[((std::size_t)cc * r.h + y) * r.w + x]);
        }
      }
    }
    genai::MetalQwenImageTransformer::RefImage img;
    img.latents = std::move(rb);
    img.seq = rseq; img.grid_h = rgh; img.grid_w = rgw;
    ri.push_back(std::move(img));
  }
  if (!ri.empty()) {
    session()->info(fmt(
        "GenerateImageStage('{}'): Qwen-Image-Edit conditioning on {} reference "
        "image(s)", this->id(), ri.size()));
  }

  // Denoise callback: upload the candidate, run the dual-stream DiT at `sigma`,
  // read the velocity back; apply norm-preserving true-CFG when enabled.
  bool dit_ok = true;
  const float gscale = (float)_guidance_scale;
  // Opened before the denoise callable so the per-block hook is
  // live for the very first forward.
  UiProgress bar = session()->open_progress("denoise");
  DenoiseProgress prog(&bar, S, cfg ? 2 : 1);
  ScopedBlockProgress<std::remove_reference_t<decltype(*_qie_dit)>>
      prog_guard(_qie_dit.get(), prog);
  auto denoise = [&](const std::vector<float>& cand,
                     double sigma) -> std::vector<float> {
    auto* lb = static_cast<std::uint16_t*>(latbuf.contents());
    for (std::size_t k = 0; k < cand.size(); ++k) {
      lb[k] = f32_to_bf16_(cand[k]);
    }
    SharedBuffer vel = _qie_dit->forward(latbuf, img_seq, txt_pos, n_real, gh,
                                         gw, (float)sigma, ri);
    prog.end_forward();
    if (vel.empty()) { dit_ok = false; return {}; }
    const auto* vp = static_cast<const std::uint16_t*>(vel.contents());
    std::vector<float> v(cand.size());
    for (std::size_t k = 0; k < v.size(); ++k) { v[k] = bf16_to_f32_(vp[k]); }
    if (cfg) {
      SharedBuffer veln = _qie_dit->forward(latbuf, img_seq, txt_neg, n_real_neg,
                                            gh, gw, (float)sigma, ri);
      prog.end_forward();
      if (veln.empty()) { dit_ok = false; return {}; }
      const auto* np = static_cast<const std::uint16_t*>(veln.contents());
      // Per-token (over IC channels): comb = neg + scale*(pos-neg), then
      // rescale comb to preserve pos's L2 norm (diffusers true-CFG).
      for (int t = 0; t < img_seq; ++t) {
        double npos = 0.0, ncomb = 0.0;
        std::vector<float> comb((std::size_t)IC);
        for (int c = 0; c < IC; ++c) {
          const std::size_t k = (std::size_t)t * IC + c;
          const float vneg = bf16_to_f32_(np[k]);
          const float cb = vneg + gscale * (v[k] - vneg);
          comb[(std::size_t)c] = cb;
          npos += (double)v[k] * v[k];
          ncomb += (double)cb * cb;
        }
        const double s = ncomb > 0.0 ? std::sqrt(npos / ncomb) : 1.0;
        for (int c = 0; c < IC; ++c) {
          v[(std::size_t)t * IC + c] = (float)(comb[(std::size_t)c] * s);
        }
      }
    }
    return v;
  };

  sampler.reset();
  // Per-step DiT timing (VPIPE_QIE_PROFILE), mirroring the Krea-2 / FLUX.2
  // paths: time each forward + log a ms/call summary. Pair with
  // VPIPE_QIE_DIT_PROFILE for the per-section breakdown inside a step.
  const bool prof = std::getenv("VPIPE_QIE_PROFILE") != nullptr;
  double dit_ms = 0.0;
  int dit_calls = 0;
  auto denoise_p = [&](const std::vector<float>& cand, double sigma) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<float> v = denoise(cand, sigma);
    dit_ms += std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t0).count();
    ++dit_calls;
    return v;
  };
  // Unpack [img_seq, IC] -> channel-first latent [16, lh, lw] (2x2). Used
  // per-step (debug step_latents) and for the final return.
  auto unpack = [&](const std::vector<float>& pk) {
    std::vector<float> latent((std::size_t)16 * lh * lw);
    for (int c = 0; c < 16; ++c) {
      for (int y = 0; y < lh; ++y) {
        for (int xx = 0; xx < lw; ++xx) {
          const int a = y / 2, ph = y % 2, b = xx / 2, pw = xx % 2;
          const std::size_t t = (std::size_t)a * gw + b;
          latent[((std::size_t)c * lh + y) * lw + xx] =
              pk[t * IC + (std::size_t)c * 4 + ph * 2 + pw];
        }
      }
    }
    return latent;
  };
  // Debug: VPIPE_QIE_DUMP_LATENT=<prefix> writes the raw PACKED latent
  // [img_seq, IC] f32 after each step as <prefix>NN.f32. Unlike the oport1
  // step stream this skips the VAE and the 8-bit PNG, so a golden comparison
  // sees the sampler state exactly -- needed because the early-step state is
  // dominated by the shared init noise, which masks the velocity error ~31x.
  const char* dump_pre = std::getenv("VPIPE_QIE_DUMP_LATENT");
  const auto gen_t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < S; ++i) {
    sampler.step(i, packed,
                 prof ? genai::FlowSampler::DenoiseFn(denoise_p) : denoise);
    if (!dit_ok) { return {}; }
    if (dump_pre != nullptr && *dump_pre != '\0') {
      char nm[32];
      std::snprintf(nm, sizeof nm, "%02d.f32", i + 1);
      std::ofstream df(std::string(dump_pre) + nm, std::ios::binary);
      if (df) {
        df.write(reinterpret_cast<const char*>(packed.data()),
                 (std::streamsize)(packed.size() * sizeof(float)));
      }
    }
    if (emit_step) { emit_step(unpack(packed)); }
    prog.end_step(i);
  }
  const double gen_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - gen_t0).count();
  session()->info(fmt(
      "GenerateImageStage('{}'): latent generated in {:.2f}s ({} denoise steps, "
      "{} ms/step)", this->id(), gen_s, S,
      S ? (long)(gen_s * 1000.0 / S) : 0));
  if (prof) {
    session()->log_normal(fmt(
        "GenerateImageStage('{}'): QIE DiT {} forward calls, {} ms total, {} "
        "ms/call (txt {} + img {} = {})", this->id(), dit_calls, (long)dit_ms,
        dit_calls ? (long)(dit_ms / dit_calls) : 0, n_real, img_seq,
        n_real + img_seq));
  }
  return unpack(packed);
}


void
GenerateImageStage::tag_output_(TensorBeat& tb) const
{
  // `_hf_dir` is the reference the user actually named (a registry key like
  // "local/Krea-2-Turbo-8bit", or a path), which is the meaningful
  // identity of the generator -- not the resolved directory.
  provenance::set_model_name(tb.sideband, _hf_dir);
  // BFL's single-reference edit generates on the reference's padded canvas
  // and cuts the result back to the picture: the box comes from
  // image-resample `pad_to_grid` through vae-encode, and vae-decode cuts.
  // Only when this output IS that canvas -- its size was taken from the
  // reference (no width/height configured), there is exactly one
  // reference, and the two latent grids agree. A configured size, or
  // several references, is a different picture with nothing to cut back
  // to (BFL's rule too).
  if (!_infer_size || _ref.size() != 1 || _ref[0].crop_box.is_null()) {
    return;
  }
  const RefLatent& r = _ref[0];
  const std::size_t n = tb.shape.size();
  if (n < 2 || tb.shape[n - 2] != r.h || tb.shape[n - 1] != r.w) { return; }
  if (!tb.sideband.is_object()) { tb.sideband = FlexData::make_object(); }
  tb.sideband.as_object().insert_or_assign(sideband::kCropBox, r.crop_box);
}

Job
GenerateImageStage::process(RuntimeContext& ctx)
{
  auto* mc = session()->services()->metal_compute();
  // ONE CALL, ONE GENERATION. Every input is read here, up front, and
  // paired by the rule in stages/generation-input.h: an input whose source
  // sends one beat and ends serves every generation of the run, one that
  // keeps sending is consumed a beat per generation, and the run ends
  // when a consumed input runs out or when nothing new arrives at all.
  //
  // ALL of them, and before anything can end the generation early -- a
  // refusal, a dropped beat. Pairing is by position in each stream, so an
  // input skipped once pairs every later generation with the wrong beat.
  // The model (iport2) is the exception: it is what LOADS, once, not an
  // input of each generation.
  using Took = GenerationInput::Took;
  GenerationRound round;

  // The model config FIRST, before anything can load: `klein_kv` is an
  // argument to the DiT's construction, not a per-step knob, so a beat
  // read after ensure_loaded_ would arrive too late to matter and would
  // do so silently.
  //
  // A generation input like the rest: a config source with no trigger
  // emits once and ends, so its beat serves the whole run, while a
  // trigger-driven one emits a beat per request. (A later beat cannot
  // change `klein_kv` on an already-built DiT -- see the note where that
  // is reported.)
  if (ctx.num_iports() > kModelCfgPort && ctx.iport_connected(kModelCfgPort) &&
      _cfg_in.wants()) {
    auto gb = co_await ctx.read(kModelCfgPort);
    const bool first = _cfg_in.beats() == 0;
    const Took took = _cfg_in.took(gb != nullptr);
    if (round.note(took)) { ctx.signal_done(); co_return; }
    if (const auto* gfd = took == Took::kFresh
            ? dynamic_cast<const FlexDataPayload*>(gb.get()) : nullptr) {
      const bool kv_before = _flux2_params.klein_kv;
      _model_cfg = gfd->data;
      // Only if the family is already known; otherwise ensure_loaded_
      // applies it the moment it is.
      if (_load_attempted) { apply_model_config_(); }
      if (!first && _family == "flux2" &&
          _flux2_params.klein_kv != kv_before) {
        session()->warn(fmt(
            "GenerateImageStage('{}'): klein_kv changed after the DiT was "
            "built; it selects the attention recipe at LOAD time, so this "
            "run keeps the old one. Restart the pipeline to change it",
            this->id()));
        _flux2_params.klein_kv = kv_before;
      }
    }
  }
  // Latch the shared model (iport2) once -- a model-select source overrides the
  // hf_dir config -- then lazily load the DiT before the first generation.
  if (!_model_latched && ctx.num_iports() > kModelPort &&
      ctx.iport_connected(kModelPort)) {
    auto mb = co_await ctx.read(kModelPort);
    _model_latched = true;
    if (const auto* mfd =
            mb ? dynamic_cast<const FlexDataPayload*>(mb.get()) : nullptr) {
      if (apply_model_select_beat(mfd->data, _hf_dir)) {
        ensure_loaded_();
      }
    }
  }
  // A config wired but no model-select source: initialize() deferred the
  // load so the config could be read first, so nothing has loaded yet.
  ensure_loaded_();
  // The conditioning (iport0) -- and with a NEW one, its negative
  // (iport1). The conditioner enqueues oport1 BEFORE oport0, so when a
  // positive arrives its paired negative is already in that FIFO: the
  // non-blocking backlog gate reads it reliably, and finds nothing for a
  // prompt that has none. No conditioning at all ends the run.
  if (_cond_in.wants()) {
    auto cb = co_await ctx.read(0);
    const Took took = _cond_in.took(cb != nullptr);
    if (round.note(took) || took == Took::kNone) {
      ctx.signal_done();
      co_return;
    }
    if (took == Took::kFresh) {
      _cond_beat = std::move(cb);
      _neg_beat.reset();
      if (ctx.num_iports() >= 2 && ctx.iport_connected(1) &&
          ctx.backlog(1) > 0) {
        _neg_beat = co_await ctx.read(1);
      }
    }
  }
  if (!_cond_beat) { ctx.signal_done(); co_return; }
  // Reference latents on iport5 / iport6, and the LIST on iport8, each a
  // generation input in `reference_mode` (by default the general rule):
  // one source picture for many prompts is sent once and held, a folder
  // to restore or edit is consumed a picture per generation. Blocking
  // reads, so a reference reliably pairs with its prompt -- a non-blocking
  // poll would race the producer. `_ref` is what the families read: the
  // two ports, then the list, packed. FLUX.2 threads them as
  // multi-reference conditioning tokens; Krea-2 uses the first as the
  // img2img init.
  // One [C,H,W] f32 latent into `dst`; false (and `dst` left empty) for
  // anything else, which `what` names in the warning.
  auto take_latent = [&](const TensorBeat& tb, RefLatent& dst,
                         const std::string& what) {
    dst = RefLatent{};
    if (tb.dtype != TensorBeat::DType::F32 || tb.shape.size() != 3 ||
        tb.shape[0] <= 0 || tb.shape[1] <= 0 || tb.shape[2] <= 0) {
      session()->warn(fmt(
          "GenerateImageStage('{}'): {} must be an f32 [C,H,W] latent; "
          "got {} [{} dims], ignoring", this->id(), what, tb.dtype_name(),
          tb.shape.size()));
      return false;
    }
    const auto bytes = tb.materialize_contiguous();
    const std::size_t n =
        (std::size_t)tb.shape[0] * tb.shape[1] * tb.shape[2];
    const float* fp = reinterpret_cast<const float*>(bytes.data());
    dst.chw.assign(fp, fp + n);
    dst.c = (int)tb.shape[0];
    dst.h = (int)tb.shape[1];
    dst.w = (int)tb.shape[2];
    if (tb.sideband.is_object()) {
      FlexData sb = tb.sideband;          // as_object() is a view: keep it
      auto so = sb.as_object();
      if (so.contains(sideband::kCropBox)) {
        dst.crop_box = so.at(sideband::kCropBox);
      }
    }
    session()->log_debug(fmt(
        "GenerateImageStage('{}'): {} = [{}, {}, {}]", this->id(), what,
        dst.c, dst.h, dst.w));
    return true;
  };
  bool refs_changed = false;
  for (int r = 0; r < 2; ++r) {
    const int port = 5 + r;
    if ((int)ctx.num_iports() <= port || !ctx.iport_connected(port) ||
        !_ref_port_in[r].wants()) {
      continue;
    }
    // COUNTED whatever arrives -- a bad beat too -- so a malformed
    // reference costs its own generation a reference, not the pairing
    // of every later one.
    auto rb = co_await ctx.read(port);
    const Took took = _ref_port_in[r].took(rb != nullptr);
    if (round.note(took)) { ctx.signal_done(); co_return; }
    if (took != Took::kFresh) { continue; }      // held, or never sent
    _ref_port[r] = RefLatent{};
    refs_changed = true;
    const auto* tb = rb ? dynamic_cast<const TensorBeatPayload*>(rb.get())
                        : nullptr;
    if (tb != nullptr) {
      take_latent(*tb, _ref_port[r], "ref_latent" + std::to_string(r));
    } else if (rb) {
      session()->warn(fmt(
          "GenerateImageStage('{}'): ref_latent{} must be an f32 [C,H,W] "
          "TensorBeat; got {}, ignoring", this->id(), r, rb->describe()));
    }
  }
  // The LIST: every reference a multi-reference graph has, in one beat.
  if ((int)ctx.num_iports() > (int)kRefListPort &&
      ctx.iport_connected(kRefListPort) && _ref_list_in.wants()) {
    auto rb = co_await ctx.read(kRefListPort);
    const Took took = _ref_list_in.took(rb != nullptr);
    if (round.note(took)) { ctx.signal_done(); co_return; }
    if (took == Took::kFresh) {
      _ref_list.clear();
      refs_changed = true;
    }
    if (const auto* lp = took == Took::kFresh
            ? dynamic_cast<const TensorListPayload*>(rb.get()) : nullptr) {
      for (std::size_t i = 0; i < lp->items.size(); ++i) {
        RefLatent one;
        if (take_latent(lp->items[i], one,
                        "ref_latents[" + std::to_string(i) + "]")) {
          _ref_list.push_back(std::move(one));
        }
      }
    } else if (took == Took::kFresh) {
      session()->warn(fmt(
          "GenerateImageStage('{}'): ref_latents must be a list of f32 "
          "[C,H,W] latents; got {}, ignoring", this->id(), rb->describe()));
    }
  }
  if (refs_changed) {
    _ref.clear();
    for (const RefLatent& r : _ref_port) {
      if (!r.empty()) { _ref.push_back(r); }
    }
    for (const RefLatent& r : _ref_list) { _ref.push_back(r); }
    if (_ref.size() > 2) {
      session()->info(fmt(
          "GenerateImageStage('{}'): {} reference latents", this->id(),
          _ref.size()));
    }
  }
  // The sampler / scheduler specs off iport3 / iport4, generation inputs
  // too: the `diffusion-sampler-select` / `scheduler-select` sources emit
  // one spec beat and end, so it serves every generation of the run.
  if (ctx.num_iports() >= 4 && ctx.iport_connected(3) && _sampler_in.wants()) {
    auto sb = co_await ctx.read(3);
    const Took took = _sampler_in.took(sb != nullptr);
    if (round.note(took)) { ctx.signal_done(); co_return; }
    const auto* sfd = took == Took::kFresh
        ? dynamic_cast<const FlexDataPayload*>(sb.get()) : nullptr;
    if (sfd != nullptr) {
      _sampler_latched = true;
      // KEPT UNPARSED as well as parsed: the built-ins take the typed
      // spec, and a registered family reads what it understands out of
      // the beat itself -- an integrator this tree does not implement is
      // still a thing a plugin may.
      _sampler_raw = sfd->data;
      std::string serr;
      _sampler_spec = genai::FlowSamplerSpec::from_flex(sfd->data, &serr);
      if (!serr.empty()) {
        session()->warn(fmt("GenerateImageStage('{}'): sampler spec: {}",
                            this->id(), serr));
      }
      session()->info(fmt(
          "GenerateImageStage('{}'): sampler = {} (eta {}, s_noise {})",
          this->id(), _sampler_spec.method, _sampler_spec.eta,
          _sampler_spec.s_noise));
    }
  }
  if (ctx.num_iports() >= 5 && ctx.iport_connected(4) &&
      _scheduler_in.wants()) {
    auto cb = co_await ctx.read(4);
    const Took took = _scheduler_in.took(cb != nullptr);
    if (round.note(took)) { ctx.signal_done(); co_return; }
    const auto* cfd = took == Took::kFresh
        ? dynamic_cast<const FlexDataPayload*>(cb.get()) : nullptr;
    if (cfd != nullptr) {
      _scheduler_latched = true;
      _scheduler_raw = cfd->data;
      std::string cerr;
      _scheduler_spec = genai::FlowSchedulerSpec::from_flex(cfd->data, &cerr);
      if (!cerr.empty()) {
        session()->warn(fmt("GenerateImageStage('{}'): scheduler spec: {}",
                            this->id(), cerr));
      }
      session()->info(fmt(
          "GenerateImageStage('{}'): scheduler = {} ({} steps, shift {} {})",
          this->id(), _scheduler_spec.type, _scheduler_spec.steps,
          _scheduler_spec.shift, _scheduler_spec.shift_type));
      // A wired scheduler OWNS the step count: its spec replaces this
      // stage's wholesale, `steps` included. When the graph also named a
      // count here and the two disagree, only one of them runs, and the
      // one in plain sight on this stage is the one that does not -- so
      // say which.
      if (_steps_set && _scheduler_spec.steps > 0 &&
          _scheduler_spec.steps != _steps) {
        session()->warn(fmt(
            "GenerateImageStage('{}'): steps {} is IGNORED -- the wired "
            "scheduler runs {} steps and owns the count. Set it on the "
            "scheduler-select stage, or make the two agree",
            this->id(), _steps, _scheduler_spec.steps));
      }
    }
  }
  // Nothing new on any input: everything is broadcasting, and this
  // generation would repeat the last one exactly.
  if (round.idle()) { ctx.signal_done(); co_return; }
  // THIS generation's seed, before anything can end it early -- see
  // `_generation`.
  _seed = generation_seed(_seed_seq, _seed_base, _generation);
  ++_generation;
  session()->info(fmt(
      "GenerateImageStage('{}'): generation {}, seed {}{}", this->id(),
      _generation, _seed,
      _seed_seq == SeedSequence::kRandomize ? " (randomized)" : ""));
  const BeatPayloadIntf* in = _cond_beat.get();
  // Did the conditioner REFUSE this prompt? Read the flag off the beat up
  // front; the refusal itself is emitted further down, once the output size
  // is known.
  bool cond_blocked = false;
  if (const auto* stb = dynamic_cast<const TensorBeatPayload*>(in)) {
    if (stb->sideband.is_object()) {
      FlexData sb = stb->sideband;        // as_object() is a view: keep it
      auto o = sb.as_object();
      cond_blocked = o.contains(sideband::kContentBlocked)
                     && o.at(sideband::kContentBlocked).as_bool(false);
      // Qwen-Image-2.1's joint-sequence bookkeeping. Read here rather
      // than in the family branch because the beat is consumed by then,
      // and read EVERY beat -- a second prompt with a different
      // reference count has a different mask, and a latched one would
      // lay the new sequence out with the old prompt's slots.
      if (_family == "qwen-image-21") {
        _qi21_slots.clear();
        _qi21_ref_gh.clear();
        _qi21_ref_gw.clear();
        auto ints = [&](std::string_view k, std::vector<int>& dst) {
          if (!o.contains(k)) { return; }
          const FlexData f = o.at(k);
          if (!f.is_array()) { return; }
          auto a = f.as_array();
          for (std::size_t i = 0; i < a.size(); ++i) {
            dst.push_back((int)a.at(i).as_int(0));
          }
        };
        std::vector<int> sl;
        ints(genai::cond_sideband::kImgSlots, sl);
        _qi21_slots.reserve(sl.size());
        for (int v : sl) { _qi21_slots.push_back(v != 0 ? 1 : 0); }
        ints(genai::cond_sideband::kRefGridH, _qi21_ref_gh);
        ints(genai::cond_sideband::kRefGridW, _qi21_ref_gw);
      }
    }
  }
  // And the NEGATIVE's own mask, off its own beat: the two prompts are
  // two sequences, and laying the negative out with the positive's mask
  // reads its rows at the wrong places (or past its end).
  _qi21_slots_neg.clear();
  if (_family == "qwen-image-21") {
    const auto* ntb =
        _neg_beat ? dynamic_cast<const TensorBeatPayload*>(_neg_beat.get())
                  : nullptr;
    if (ntb != nullptr && ntb->sideband.is_object()) {
      FlexData sb = ntb->sideband;        // as_object() is a view: keep it
      auto o = sb.as_object();
      if (o.contains(genai::cond_sideband::kImgSlots)) {
        const FlexData f = o.at(genai::cond_sideband::kImgSlots);
        if (f.is_array()) {
          auto a = f.as_array();
          _qi21_slots_neg.reserve(a.size());
          for (std::size_t i = 0; i < a.size(); ++i) {
            _qi21_slots_neg.push_back(a.at(i).as_int(0) != 0 ? 1 : 0);
          }
        }
      }
    }
  }
  // A prior generation may have freed the FLUX.2 DiT to make room for the
  // vae-decode on a memory-bounded box; reload it now a new prompt arrived.
  if (_family == "flux2" && _dit_unloaded && !_flux2_dit) {
    session()->info(fmt(
        "GenerateImageStage('{}'): reloading the FLUX.2 DiT for the next prompt",
        this->id()));
    if (load_flux2_dit_()) { _dit_unloaded = false; }
  }
  if (_family == "krea2" && _dit_unloaded && !_dit) {
    session()->info(fmt(
        "GenerateImageStage('{}'): reloading the Krea-2 DiT for the next prompt",
        this->id()));
    if (load_krea2_dit_()) { _dit_unloaded = false; }
  }
  if (_family == "qwen-image-edit" && _dit_unloaded && !_qie_dit) {
    session()->info(fmt(
        "GenerateImageStage('{}'): reloading the Qwen-Image-Edit DiT for the "
        "next prompt", this->id()));
    if (load_qie_dit_()) { _dit_unloaded = false; }
  }
  if (_family == "qwen-image-21" && _dit_unloaded && !_qi21_dit) {
    session()->info(fmt(
        "GenerateImageStage('{}'): reloading the Qwen-Image-2.1 DiT for the "
        "next prompt", this->id()));
    if (load_qwen_image21_dit_()) { _dit_unloaded = false; }
  }
  if (_family == "z-image" && _dit_unloaded && !_zi_dit) {
    session()->info(fmt(
        "GenerateImageStage('{}'): reloading the Z-Image DiT for the next "
        "prompt", this->id()));
    if (load_z_image_dit_()) { _dit_unloaded = false; }
  }
  if (_family == "fibo2" && _dit_unloaded && !_fibo2_dit) {
    session()->info(fmt(
        "GenerateImageStage('{}'): reloading the Fibo 2.0 DiT for the next "
        "prompt", this->id()));
    if (load_fibo2_dit_()) { _dit_unloaded = false; }
  }
  if (_family == "flux3" && _dit_unloaded && !_flux3_dit) {
    session()->info(fmt(
        "GenerateImageStage('{}'): reloading the FLUX.3 DiT for the next "
        "prompt", this->id()));
    if (load_flux3_dit_()) { _dit_unloaded = false; }
  }
  if (_family == "moss-image" && _dit_unloaded && !_moss_dit) {
    session()->info(fmt(
        "GenerateImageStage('{}'): reloading the MOSS-Image DiT for the "
        "next prompt", this->id()));
    if (load_moss_image_dit_()) { _dit_unloaded = false; }
  }
  if (_family == "boogu-image" && _dit_unloaded && !_boogu_dit) {
    session()->info(fmt(
        "GenerateImageStage('{}'): reloading the Boogu-Image DiT for the next "
        "prompt", this->id()));
    if (load_boogu_dit_()) { _dit_unloaded = false; }
  }
  const bool have_dit = _plugin_family != nullptr ? (bool)_plugin_gen
      : _family == "flux2" ? (bool)_flux2_dit
      : _family == "qwen-image-edit" ? (bool)_qie_dit
      : _family == "boogu-image" ? (bool)_boogu_dit
      : _family == "qwen-image-21" ? (bool)_qi21_dit
      : _family == "z-image" ? (bool)_zi_dit
      : _family == "flux3" ? (bool)_flux3_dit
      : _family == "fibo2" ? (bool)_fibo2_dit
      : _family == "moss-image" ? (bool)_moss_dit
      : _family == "vosr" ? (bool)_vosr
      : (bool)_dit;
  if (!have_dit) {
    session()->warn(fmt(
        "GenerateImageStage('{}'): models not loaded; dropping beat", this->id()));
    co_return;
  }
  // iport0: the conditioning tensor from a diffusion-conditioner stage
  // (family-shaped + typed; rows = shape[0]). Copy it into a metal buffer.
  const auto* ctb = dynamic_cast<const TensorBeatPayload*>(in);
  if (ctb == nullptr || ctb->shape.empty() || ctb->shape[0] <= 0) {
    session()->warn(fmt(
        "GenerateImageStage('{}'): expected a conditioning TensorBeat, got {}; "
        "dropping beat", this->id(), in->describe()));
    co_return;
  }
  metal_compute::SharedBuffer cond = cond_to_shared_(mc, *ctb);
  if (cond.empty() && !cond_blocked) {
    session()->warn(fmt(
        "GenerateImageStage('{}'): conditioning upload failed; dropping beat",
        this->id()));
    co_return;
  }
  const int n_real = (int)ctb->shape[0];
  session()->log_debug(fmt(
      "GenerateImageStage('{}'): conditioning beat [{} rows, {}]", this->id(),
      n_real, ctb->dtype == TensorBeat::DType::Bf16 ? "bf16" : "f16"));

  // iport1: OPTIONAL negative conditioning (the conditioner's oport1) for
  // classifier-free guidance, read with its positive at the top.
  metal_compute::SharedBuffer cond_neg;
  int n_real_neg = 0;
  {
    const auto* ntb =
        _neg_beat ? dynamic_cast<const TensorBeatPayload*>(_neg_beat.get())
                  : nullptr;
    if (ntb != nullptr && !ntb->shape.empty() && ntb->shape[0] > 0) {
      cond_neg = cond_to_shared_(mc, *ntb);
      if (!cond_neg.empty()) { n_real_neg = (int)ntb->shape[0]; }
      session()->log_debug(fmt(
          "GenerateImageStage('{}'): negative conditioning [{} rows]", this->id(),
          n_real_neg));
    }
  }

  // Optional debug/repro: packed initial latents from a raw f32 file.
  std::vector<float> init;
  const std::vector<float>* init_ptr = nullptr;
  if (!_init_latents.empty()) {
    std::ifstream f(_init_latents, std::ios::binary);
    if (f) {
      f.seekg(0, std::ios::end);
      const std::streamoff nb = f.tellg();
      f.seekg(0, std::ios::beg);
      init.resize((std::size_t)nb / 4);
      f.read(reinterpret_cast<char*>(init.data()), nb);
      init_ptr = &init;
    }
  }


  // Output size. An explicit width/height ALWAYS wins; only a completely
  // unconfigured size infers, and then from ref_latent0 -- vae-encode's output
  // for the source image -- times the family's VAE scale, so an edit comes out
  // at the source resolution with no size config at all.
  //
  // The multiple-of-16 test is the same one initialize() applies to a
  // configured size. It is vacuous for the 16x families and is the real
  // constraint for the 8x ones, whose DiTs patchify 2x2 and so need an EVEN
  // latent grid (Krea-2 and Qwen-Image-Edit both). An odd reference warns and
  // falls back rather than silently truncating the grid.
  int gen_h = _height, gen_w = _width;
  if (_infer_size) {
    const RefLatent& r0 = ref_(0);
    if (!r0.empty()) {
      const int s = latent_scale_(_family);
      const int ih = r0.h * s, iw = r0.w * s;
      if (ih % 16 == 0 && iw % 16 == 0) {
        gen_h = ih; gen_w = iw;
      } else {
        session()->warn(fmt(
            "GenerateImageStage('{}'): ref_latent0 [{}, {}, {}] implies {}x{}, "
            "not a multiple of 16 ({}x latent needs an even grid); using "
            "256x256 -- set width/height", this->id(), r0.c, r0.h, r0.w, iw,
            ih, s));
      }
    }
    // Fibo 2.0's text-to-image, as its reference defaults it: the
    // trained area, 1024^2 (default_sample_size 128 x the VAE's 8).
    if ((gen_h <= 0 || gen_w <= 0) && _family == "fibo2") {
      gen_h = 1024; gen_w = 1024;
    }
    if (gen_h <= 0 || gen_w <= 0) { gen_h = 256; gen_w = 256; }
    session()->log_debug(fmt(
        "GenerateImageStage('{}'): output size {}x{} ({})", this->id(), gen_w,
        gen_h, r0.empty() ? "no reference, default" : "from ref_latent0"));
  }
  // ---- Content-policy refusal ------------------------------------------
  // A family's content screen judges every prompt in the conditioner and
  // tags a refused one `content_blocked`. Honour it
  // HERE, once the output size is known and before any DiT work: a refusal
  // costs no denoise. The beat carries the size explicitly so vae-decode can
  // paint the blank refusal image without interpreting a latent that was
  // never generated.
  //
  // This is not family-gated: a `content_blocked` beat is refused whatever
  // produced it. Dropping the beat instead would stall a pipeline waiting on
  // this stage's oport0, so a refusal still emits.
  if (cond_blocked) {
    auto out = std::make_unique<TensorBeatPayload>();
    out->dtype = TensorBeat::DType::F32;
    out->shape = {1, 1, 1};
    out->resize_contiguous(1);
    out->as_f32()[0] = 0.0f;
    FlexData sb = FlexData::make_object();
    auto o = sb.as_object();
    o.insert_or_assign(sideband::kContentBlocked, FlexData::make_bool(true));
    o.insert_or_assign(sideband::kRefusalHeight, FlexData::make_int(gen_h));
    o.insert_or_assign(sideband::kRefusalWidth,  FlexData::make_int(gen_w));
    out->sideband = std::move(sb);
    ++_latents_emitted;
    session()->info(fmt(
        "GenerateImageStage('{}'): conditioning refused by the content policy; "
        "skipping the denoise and emitting a {}x{} refusal", this->id(),
        gen_w, gen_h));
    co_await ctx.write(0, std::move(out));
    co_return;
  }

  int lh = gen_h / 8, lw = gen_w / 8;
  std::vector<float> latent;
  const std::vector<float>* latent_ptr = nullptr;
  // Krea-2 img2img: ref latent 0 (the whitened [z_dim, H/8, W/8] vae-encode
  // output) is mixed into the noise at the strength-selected sigma. FLUX.2
  // skips this -- refs are conditioning tokens, not an init, handled in
  // generate_flux2_. When the size was inferred above, rr.h/rr.w match lh/lw
  // by construction.
  if (_family == "krea2" && _strength > 0.0 && !ref_(0).empty()) {
    const RefLatent& rr = ref_(0);
    const bool ok_type = rr.c == 16 && rr.h > 0 && rr.w > 0;
    if (ok_type && rr.h == lh && rr.w == lw) {
      latent = rr.chw;
      latent_ptr = &latent;
    } else {
      session()->warn(fmt(
          "GenerateImageStage('{}'): img2img init must be an f32 latent [16, {}, "
          "{}]; got [{}, {}, {}], ignoring", this->id(), lh, lw, rr.c, rr.h,
          rr.w));
    }
  }


  // ---- FLUX.2: text-to-image from noise (+ optional reference-image
  // conditioning from iport5/iport6) -> latent [dit_channels, H/16, W/16].
  // (No negative-prompt CFG; klein is distilled.) ----
  // step_latent (oport1): stream each denoise step's latent LIVE so a downstream
  // vae-decode -> preview updates every step. Emitted only when a consumer is
  // wired. `step_emitter(shape)` returns a per-step callback the generate_ loops
  // call right after each sampler step; it write_sync's the unpacked latent from
  // the (synchronous) denoise loop -- the runtime wakes the consumer on the
  // thread pool, so it decodes concurrently with the next DiT step. write_sync
  // never blocks the loop (drops if the ring is full) and stops on consumer
  // close (step_alive). Empty callback (no consumer) => zero overhead.
  const bool want_steps = ctx.num_oports() > 1 && ctx.has_consumers(1);
  bool step_alive = true;
  auto step_emitter = [&](std::vector<std::int64_t> shape)
      -> std::function<void(const std::vector<float>&)> {
    if (!want_steps) { return {}; }
    return [&, shape](const std::vector<float>& lat) {
      if (!step_alive) { return; }
      auto so = std::make_unique<TensorBeatPayload>();
      so->dtype = TensorBeat::DType::F32;
      so->shape = shape;
      so->resize_contiguous(lat.size());
      tag_output_(*so);
      std::memcpy(so->as_f32(), lat.data(), lat.size() * sizeof(float));
      step_alive = ctx.write_sync(1, std::move(so));
    };
  };
  // Cooperative stop: feed the active DiT a hook reporting the pipeline stop
  // flag so it abandons the forward within ~one block instead of running the
  // whole (multi-second at high res) step. Set around each generate_ call and
  // cleared after -- the callback captures ctx, valid only for this process().
  auto stopping = [&ctx]() { return ctx.stop_requested(); };

  // ---- AN OUT-OF-TREE FAMILY -----------------------------------------
  //
  // First, and instead of everything below when one claimed this
  // checkpoint. The family owns its whole denoise -- scheduler,
  // guidance, patchify, residency -- and returns the latent this stage
  // publishes. What it gets is what the built-ins get: the conditioning
  // as it ARRIVED (not the uploaded copy: the family owns its own
  // dtype and its own upload), the geometry the stage settled, the
  // reference latents, the two specs unparsed, and the acceleration bag.
  if (_plugin_gen) {
    genai::ImageGenRequest req;
    req.height = gen_h;
    req.width  = gen_w;
    req.steps  = _scheduler_spec.steps;
    req.seed   = _seed;
    req.guidance_scale = _guidance_scale;
    // MATERIALIZED, because a beat may be strided or offset and a
    // family reads a contiguous block. Held in a local for the call,
    // which is exactly the lifetime the request documents.
    const auto cond_bytes = ctb->materialize_contiguous();
    req.cond      = cond_bytes.data();
    req.cond_rows = n_real;
    req.cond_dim  = ctb->shape.size() > 1 ? (int)ctb->shape.back() : 0;
    req.cond_elem_size = ctb->dtype == TensorBeat::DType::F32 ? 4 : 2;
    req.cond_is_bf16   = ctb->dtype == TensorBeat::DType::Bf16;
    req.cond_sideband  = ctb->sideband.is_object() ? &ctb->sideband : nullptr;
    // Every wired reference, in port order, packed: a graph that wires
    // only the second port hands the family ONE reference. Held in a
    // local for the call, like the conditioning above.
    std::vector<genai::NamedTensor> refs;
    for (const RefLatent& r : _ref) {
      genai::NamedTensor t;
      t.data  = r.chw.data();
      t.shape = {r.c, r.h, r.w};
      t.set_dtype(genai::named_tensor::kF32);
      refs.push_back(std::move(t));
    }
    req.references = refs;
    req.model_config   = _model_cfg.is_object() ? &_model_cfg : nullptr;
    req.sampler_spec   = _sampler_raw.is_object() ? &_sampler_raw : nullptr;
    req.scheduler_spec = _scheduler_raw.is_object() ? &_scheduler_raw : nullptr;
    req.accel          = &_accel;
    // Cancellation and the two progress granularities. A family that
    // never calls these cannot be interrupted and reports nothing, which
    // on a 12B DiT is a Stop that takes minutes and a bar that never
    // moves.
    UiProgress bar = session()->open_progress("denoise");
    DenoiseProgress prog(&bar, req.steps, /*forwards_per_step=*/1);
    auto block_fn = prog.block_fn();
    // A plugin drives its own loop and has no command buffer to hang the
    // publish on, so the two phases collapse into one call -- staging and
    // publishing at the same instant cannot go stale between them.
    req.block_progress = [&block_fn, &ctx](int done, int total) {
      if (auto publish = block_fn(done, total)) { publish(); }
      return !ctx.stop_requested();
    };
    // `total` is the count the family's SCHEDULER settled on, not the
    // configured steps; adopting it is what makes the bar finish at
    // 100%. `step` is 1-based on entry, end_step takes the 0-based index
    // it just finished.
    req.progress = [&prog, &ctx](int step, int total) {
      prog.set_steps(total);
      prog.end_step(step - 1);
      return !ctx.stop_requested();
    };
    // Each step's latent, LIVE on oport1, when anything is listening.
    // The family calls it or does not; a step-latent nobody consumes
    // costs an empty std::function.
    if (want_steps) {
      req.step_latent = [&](int, int, const float* lat,
                            const std::vector<int>& shp) {
        if (lat == nullptr || shp.empty()) { return; }
        std::vector<std::int64_t> s64(shp.begin(), shp.end());
        std::size_t n = 1;
        for (int d : shp) { n *= (std::size_t)(d > 0 ? d : 0); }
        if (n == 0) { return; }
        auto fn = step_emitter(std::move(s64));
        if (fn) { fn(std::vector<float>(lat, lat + n)); }
      };
    }
    // ALWAYS INSTALLED, even though nothing is named yet. A family that
    // had to guard every lookup would eventually forget one, and calling
    // an empty std::function throws -- which on this path is a family
    // taking the host down for asking a question the answer to is "no".
    req.input = [](std::string_view, genai::NamedTensor*) { return false; };
    // The output seam, installed for the same reason. The one name this
    // stage takes is a live preview's clean estimate, on a step the
    // previewer renders; the family hands it back shaped like its result
    // latent, [z, h, w], and it is decoded as one latent frame.
    LatentPreviewer::Scope preview_scope(_preview.get(), ctx, kPreviewPort,
                                         0.0, 0);
    const bool previewing = preview_scope.active();
    req.output_wanted = [this, previewing](std::string_view name, int step,
                                           int total) {
      return previewing && name == genai::kOutputPreviewX0 &&
             _preview->due(step, total);
    };
    req.output = [this, previewing](std::string_view name, int step,
                                    int total, const genai::NamedTensor& t) {
      if (!previewing || name != genai::kOutputPreviewX0) { return; }
      const bool r3 = t.shape.size() == 3;
      const bool r4 = t.shape.size() == 4 && t.shape[1] == 1;
      if (t.data == nullptr || !t.is(genai::named_tensor::kF32) ||
          t.elems() == 0 || (!r3 && !r4)) {
        if (!_preview_shape_said) {
          _preview_shape_said = true;
          session()->warn(fmt(
              "GenerateImageStage('{}'): family '{}' handed back a '{}' "
              "that is not an f32 [z, h, w] latent; no previews from it",
              this->id(), _family, genai::kOutputPreviewX0));
        }
        return;
      }
      const float* p = static_cast<const float*>(t.data);
      const int h = t.shape[r3 ? 1 : 2], w = t.shape[r3 ? 2 : 3];
      _preview->submit(std::vector<float>(p, p + t.elems()), t.shape[0], 1,
                       h, w, step, total);
    };
    genai::ImageGenResult res;
    bool ok = false;
    try {
      ok = _plugin_gen->generate(req, &res);
    } catch (const std::exception& e) {
      // A family that lets an exception out would otherwise take the
      // host down. The contract says never to, and this is what makes
      // breaking it a dropped beat instead.
      session()->warn(fmt(
          "GenerateImageStage('{}'): the '{}' family threw: {}; dropping "
          "beat", this->id(), _family, e.what()));
      co_return;
    } catch (...) {
      session()->warn(fmt(
          "GenerateImageStage('{}'): the '{}' family threw a non-standard "
          "exception; dropping beat", this->id(), _family));
      co_return;
    }
    if (!ok || res.latent.empty() || res.shape.empty()) {
      session()->info(fmt(
          "GenerateImageStage('{}'): '{}' generation {}; dropping beat",
          this->id(), _family,
          ctx.stop_requested() ? "stopped" : "failed"));
      co_return;
    }
    // THE SHAPE THE FAMILY RETURNED, not the one it predicted at load:
    // a family that mispredicts its own geometry should produce a
    // confusing log, not a mislabelled beat.
    std::size_t want = 1;
    for (int d : res.shape) { want *= (std::size_t)(d > 0 ? d : 0); }
    if (want == 0 || want != res.latent.size()) {
      session()->warn(fmt(
          "GenerateImageStage('{}'): '{}' returned {} floats for a shape "
          "wanting {}; dropping beat", this->id(), _family,
          res.latent.size(), want));
      co_return;
    }
    auto out = std::make_unique<TensorBeatPayload>();
    out->dtype = TensorBeat::DType::F32;
    out->shape.assign(res.shape.begin(), res.shape.end());
    out->resize_contiguous(res.latent.size());
    std::memcpy(out->as_f32(), res.latent.data(),
                res.latent.size() * sizeof(float));
    // WHAT THE FAMILY WANTED SAID BACK. Merged before the stage's own
    // provenance so a family cannot overwrite the model name -- what it
    // is reporting about is downstream of what produced it.
    if (res.sideband.is_object()) {
      auto from = res.sideband.as_object();
      auto to = out->sideband.is_object() ? out->sideband.as_object()
                                          : (out->sideband =
                                                 FlexData::make_object(),
                                             out->sideband.as_object());
      for (const auto& kv : from) {
        to.insert_or_assign(kv.first, kv.second);
      }
    }
    tag_output_(*out);
    ++_latents_emitted;
    session()->info(fmt(
        "GenerateImageStage('{}'): '{}' latent {} floats ({} steps @ {}x{})",
        this->id(), _family, res.latent.size(), req.steps, gen_w, gen_h));
    bar.finish();
    // What the DiT held WHILE it denoised, before any release below --
    // the figure a peer sizes against. As for the built-in families.
    correct_dit_holding_("after a generation");
    // ROOM FOR THE DECODE, on the same signal the built-in families use
    // -- and before the write, so the downstream vae-decode sees the
    // freed room rather than racing it.
    //
    // This stage has no `unload_when_idle` key: the built-ins decide it
    // per generation instead, from whether the decode about to run
    // would fit beside a resident DiT. The host cannot size a plugin's
    // decode peak, so what it can offer a family is the QUESTION -- the
    // box is tight, drop what you can -- and let the family answer with
    // whatever it is willing to give up. A family with nothing to
    // release does nothing, which is the default.
    {
      const int forced = forced_dit_reclaim_();
      bool tight = forced > 0;
      if (forced < 0) {
        auto* mc2 = session()->services()->metal_compute();
        if (mc2 != nullptr) {
          const auto mb = mc2->memory_budget();
          // The latent that just came back is a floor on what the decode
          // will touch, not the peak -- but a peak this stage cannot
          // compute is better approximated by something real than by a
          // built-in's formula for a different model.
          const std::size_t hint = res.latent.size() * sizeof(float);
          tight = mb.recommended != 0 &&
                  !(mb.fits(hint) && mb.fits_physical(hint));
        }
      }
      if (tight) {
        session()->info(fmt(
            "GenerateImageStage('{}'): asking '{}' to release what it can "
            "before the decode{}", this->id(), _family,
            forced > 0 ? " -- FORCED by VPIPE_IMAGE_DIT_RECLAIM" : ""));
        _plugin_gen->release_idle();
      }
    }
    co_await ctx.write(0, std::move(out));
    co_return;
  }

  if (_family == "flux2") {
    std::vector<RefLatent> frefs;
    for (const RefLatent& r : _ref) { frefs.push_back(r); }
    const int Cdit = _flux2_dit->config().in_channels;
    const int fgh = gen_h / 16, fgw = gen_w / 16;
    _flux2_dit->set_stream_stop(stopping);
    // Previews for this image, ended when this block returns -- after the
    // latent is written.
    LatentPreviewer::Scope preview_scope(_preview.get(), ctx, kPreviewPort,
                                         0.0, 0);
    const std::vector<float> fl =
        generate_flux2_(cond, n_real, gen_h, gen_w, frefs, init_ptr,
                        step_emitter({Cdit, fgh, fgw}));
    _flux2_dit->set_stream_stop({});
    if (fl.empty()) {
      session()->info(fmt(
          "GenerateImageStage('{}'): FLUX.2 generation {}; dropping beat",
          this->id(), ctx.stop_requested() ? "stopped" : "failed"));
      co_return;
    }
    auto out = std::make_unique<TensorBeatPayload>();
    out->dtype = TensorBeat::DType::F32;
    out->shape = {Cdit, fgh, fgw};
    out->resize_contiguous(fl.size());
    std::memcpy(out->as_f32(), fl.data(), fl.size() * sizeof(float));
    tag_output_(*out);
    ++_latents_emitted;
    session()->info(fmt(
        "GenerateImageStage('{}'): FLUX.2 latent [{}, {}, {}] ({} steps @ {}x{})",
        this->id(), Cdit, fgh, fgw, _scheduler_spec.steps, gen_h, gen_w));
    // What the DiT held while it denoised, BEFORE the free (a transient).
    release_idle_slots_();
    correct_dit_holding_("after a generation");
    // Free the DiT if the downstream vae-decode won't fit alongside it (before
    // publishing the latent, so the decode stage sees the freed working set).
    if (_release_scratch && _flux2_dit) {
      _flux2_dit->release_forward_scratch();
    }
    free_flux2_dit_for_decode_(gen_w, gen_h);
    co_await ctx.write(0, std::move(out));
    co_return;
  }

  // ---- Qwen-Image-Edit: image-aware conditioning (from the conditioner) +
  // dual-stream DiT (reference latents from iport5/iport6 as DiT conditioning
  // tokens) + norm-preserving true-CFG -> whitened latent [16, H/8, W/8]. ----
  if (_family == "qwen-image-edit") {
    std::vector<RefLatent> qrefs;
    for (const RefLatent& r : _ref) { qrefs.push_back(r); }
    _qie_dit->set_stream_stop(stopping);
    const std::vector<float> ql =
        generate_qie_(cond, n_real, cond_neg, n_real_neg, gen_h, gen_w,
                      init_ptr, qrefs, step_emitter({16, lh, lw}));
    _qie_dit->set_stream_stop({});
    if (ql.empty()) {
      session()->info(fmt(
          "GenerateImageStage('{}'): Qwen-Image-Edit generation {}; dropping "
          "beat", this->id(), ctx.stop_requested() ? "stopped" : "failed"));
      co_return;
    }
    auto out = std::make_unique<TensorBeatPayload>();
    out->dtype = TensorBeat::DType::F32;
    out->shape = {16, lh, lw};
    out->resize_contiguous(ql.size());
    std::memcpy(out->as_f32(), ql.data(), ql.size() * sizeof(float));
    tag_output_(*out);
    ++_latents_emitted;
    session()->info(fmt(
        "GenerateImageStage('{}'): Qwen-Image-Edit latent [16, {}, {}] "
        "({} steps @ {}x{})", this->id(), lh, lw, _scheduler_spec.steps,
        gen_h, gen_w));
    // Before publishing: on a bounded box 20B of resident DiT would leave
    // the downstream vae-decode no working set. Freed here, reloaded on
    // the next prompt (the latent is already read back, so the DiT is
    // idle) -- and BEFORE the write, so the separate vae-decode stage
    // sees the freed room rather than racing it.
    release_idle_slots_();
    correct_dit_holding_("after a generation");
    if (_release_scratch && _qie_dit) { _qie_dit->release_forward_scratch(); }
    free_qie_dit_for_decode_(gen_w, gen_h);
    co_await ctx.write(0, std::move(out));
    co_return;
  }

  // ---- VOSR: RESTORE rather than generate ------------------------------
  //
  // The one family here whose output geometry is not a choice. The
  // low-quality latent on ref_latent0 IS the grid, so `width`/`height`
  // are read only to notice a disagreement; nothing upsamples inside the
  // model. Everything a text-to-image run configures -- guidance, the
  // negative beat, the sampler, `strength` -- has no meaning on this
  // path and is not read.
  if (_family == "vosr") {
    const RefLatent& r0 = ref_(0);
    if (r0.empty()) {
      session()->warn(fmt(
          "GenerateImageStage('{}'): VOSR restores a picture and nothing was "
          "wired to ref_latent0 -- feed it a vae-encode of the same pixels "
          "the conditioner saw; dropping beat", this->id()));
      co_return;
    }
    if (r0.c != _vosr_cfg.latent_ch) {
      session()->warn(fmt(
          "GenerateImageStage('{}'): VOSR wants a {}-channel latent on "
          "ref_latent0 and got {}; dropping beat", this->id(),
          _vosr_cfg.latent_ch, r0.c));
      co_return;
    }
    if (r0.h * 8 != gen_h || r0.w * 8 != gen_w) {
      // Not an error: the size config is simply not in charge here, and
      // a graph that set one deserves to be told which number won.
      session()->log_debug(fmt(
          "GenerateImageStage('{}'): VOSR takes its geometry from "
          "ref_latent0 ({}x{}), not from the configured {}x{}",
          this->id(), r0.w * 8, r0.h * 8, gen_w, gen_h));
      gen_h = r0.h * 8;
      gen_w = r0.w * 8;
      lh = r0.h;
      lw = r0.w;
    }
    genai::MetalVosrTransformer::RestoreRequest rr;
    rr.lq        = r0.chw.data();
    rr.lh        = r0.h;
    rr.lw        = r0.w;
    rr.cond      = ctb->as_u8();
    rr.cond_rows = n_real;
    // The distilled checkpoint is ONE step. A graph asking for more gets
    // more -- the flow loop is the same recipe at any count -- but the
    // default has to be the one the weights were distilled for, not this
    // stage's text-to-image default of 20.
    rr.steps     = _steps_set ? _scheduler_spec.steps : 1;
    rr.seed      = _seed;
    // TILING IS THE DEFAULT past the grid the weights were distilled at.
    //
    // Pixels in the config, latent cells in the request: one conversion,
    // here, so the key means the same thing it means in the reference's
    // command line. What the key does NOT do any more is default to a
    // single pass at every size -- see RestoreRequest::tile. Untiled is
    // the reference's path at the trained grid and a mode it cannot run
    // above it, and the artifact that costs is a checkerboard on the
    // content the model synthesises rather than passes through.
    const int auto_tile =
        genai::MetalVosrTransformer::default_tile(lh, lw, _vosr_cfg);
    if (_tile_size_set) {
      if (_tile_size > 0) {
        rr.tile         = _tile_size / 8;
        rr.tile_overlap = _tile_overlap / 8;
      } else if (auto_tile > 0 && _latents_emitted == 0) {
        // A graph that asked for one pass gets one. Said once, because
        // it is a deliberate setting and not a mistake -- but it is the
        // setting that produces the artifact, and nothing downstream
        // will name it.
        session()->warn(fmt(
            "GenerateImageStage('{}'): tile_size=0 runs this {}x{} latent in "
            "ONE pass, on a {}x{} token grid where the weights were "
            "distilled at {}x{}. The reference tiles instead, and off its "
            "grid the rotary positions land between the ones the model saw "
            "-- expect a periodic weave on faces. Unset tile_size to tile "
            "at {} px", this->id(), lw, lh, lw / _vosr_cfg.patch,
            lh / _vosr_cfg.patch, _vosr_cfg.train_grid, _vosr_cfg.train_grid,
            auto_tile * 8));
      }
    } else if (auto_tile > 0) {
      rr.tile         = auto_tile;
      rr.tile_overlap = _tile_overlap / 8;
      if (_latents_emitted == 0) {
        session()->info(fmt(
            "GenerateImageStage('{}'): restoring {}x{} in {} px tiles -- the "
            "resolution this checkpoint was distilled at; set tile_size to "
            "override", this->id(), gen_w, gen_h, auto_tile * 8));
      }
    }
    // The conditioning must be the bf16 DINOv2 grid the vosr
    // conditioner emits. A beat of any other element type is a graph
    // wired to the wrong conditioner, and reading bf16 as f16 gives
    // values of the right magnitude and entirely the wrong content.
    if (ctb->dtype != TensorBeat::DType::Bf16 ||
        ctb->shape.size() != 2 ||
        (int)ctb->shape.back() != _vosr_cfg.enc_dim) {
      session()->warn(fmt(
          "GenerateImageStage('{}'): VOSR needs a bf16 [tokens, {}] "
          "conditioning beat from a vosr diffusion-conditioner; got {}; "
          "dropping beat", this->id(), _vosr_cfg.enc_dim, in->describe()));
      co_return;
    }
    UiProgress vbar = session()->open_progress("restore");
    DenoiseProgress vprog(&vbar, rr.steps, /*forwards_per_step=*/1);
    rr.block_progress = vprog.block_fn();
    rr.progress = [&vprog, &ctx](int step, int total) {
      vprog.set_steps(total);
      vprog.end_step(step - 1);
      return !ctx.stop_requested();
    };
    std::vector<float> vout;
    std::string verr;
    const bool ok = _vosr->restore(rr, &vout, &verr);
    if (!ok || vout.empty()) {
      session()->info(fmt(
          "GenerateImageStage('{}'): VOSR restore {}; dropping beat",
          this->id(),
          ctx.stop_requested() ? "stopped"
                               : (verr.empty() ? "failed" : verr.c_str())));
      co_return;
    }
    auto out = std::make_unique<TensorBeatPayload>();
    out->dtype = TensorBeat::DType::F32;
    out->shape = {_vosr_cfg.latent_ch, lh, lw};
    out->resize_contiguous(vout.size());
    std::memcpy(out->as_f32(), vout.data(), vout.size() * sizeof(float));
    tag_output_(*out);
    ++_latents_emitted;
    session()->info(fmt(
        "GenerateImageStage('{}'): VOSR latent [{}, {}, {}] ({} step{} @ "
        "{}x{}, {} conditioning tokens{})", this->id(), _vosr_cfg.latent_ch,
        lh, lw, rr.steps, rr.steps == 1 ? "" : "s", gen_w, gen_h, n_real,
        rr.tile > 0 ? ", tiled" : ""));
    co_await ctx.write(0, std::move(out));
    co_return;
  }

  // ---- Boogu-Image: single last-hidden conditioning (the WHOLE templated
  // sequence) + the NextDiT, with the DMD student's ascending-sigma loop.
  // The VAE is 8x and the DiT patches 2x2 itself, so the emitted latent is
  // [16, H/8, W/8] -- the same shape Qwen-Image-Edit emits. ----
  // ---- Qwen-Image-2.1: one joint sequence, a /16 VAE, and the target
  // block's rows as the latent. The emitted shape is [64, H/16, W/16] --
  // 64 channels because the DiT consumes latents UNPATCHED. ----
  if (_family == "qwen-image-21") {
    const int qlh = gen_h / 16, qlw = gen_w / 16;
    const int QZ = _qi21_dit->config().out_channels;
    std::vector<RefLatent> qrefs;
    for (const RefLatent& r : _ref) { qrefs.push_back(r); }
    _qi21_dit->set_stream_stop(stopping);
    LatentPreviewer::Scope preview_scope(_preview.get(), ctx, kPreviewPort,
                                         0.0, 0);
    const std::vector<float> ql =
        generate_qwen_image21_(cond, n_real, cond_neg, n_real_neg, gen_h,
                               gen_w, qrefs, step_emitter({QZ, qlh, qlw}));
    _qi21_dit->set_stream_stop({});
    if (ql.empty()) {
      session()->info(fmt(
          "GenerateImageStage('{}'): Qwen-Image-2.1 generation {}; dropping "
          "beat", this->id(),
          ctx.stop_requested() ? "stopped" : "failed"));
      co_return;
    }
    auto out = std::make_unique<TensorBeatPayload>();
    out->dtype = TensorBeat::DType::F32;
    out->shape = {QZ, qlh, qlw};
    out->resize_contiguous(ql.size());
    std::memcpy(out->as_f32(), ql.data(), ql.size() * sizeof(float));
    tag_output_(*out);
    ++_latents_emitted;
    session()->info(fmt(
        "GenerateImageStage('{}'): Qwen-Image-2.1 latent [{}, {}, {}] ({} "
        "steps @ {}x{})", this->id(), QZ, qlh, qlw, _scheduler_spec.steps,
        gen_h, gen_w));
    // BEFORE the free, not after: this says what the DiT held while it
    // denoised, which is the number a peer has to size against. The
    // free that may follow is a transient (see declare_memory), so it
    // deliberately does not move the holding back down.
    release_idle_slots_();
    correct_dit_holding_("after a generation");
    if (_release_scratch && _qi21_dit) { _qi21_dit->release_forward_scratch(); }
    free_qwen_image21_dit_for_decode_(gen_w, gen_h);
    co_await ctx.write(0, std::move(out));
    co_return;
  }

  // ---- Z-Image: [image | caption] joined, an 8x VAE and a 2x patch in
  // the DiT, so the emitted latent is [16, H/8, W/8] -- the same shape
  // Boogu-Image and Qwen-Image-Edit emit, and the plain AutoencoderKL
  // decodes it. ----
  if (_family == "z-image") {
    const int ZZ = _zi_dit->config().in_channels;
    _zi_dit->set_stream_stop(stopping);
    LatentPreviewer::Scope preview_scope(_preview.get(), ctx, kPreviewPort,
                                         0.0, 0);
    const std::vector<float> zl =
        generate_z_image_(cond, n_real, cond_neg, n_real_neg, gen_h, gen_w,
                          step_emitter({ZZ, lh, lw}));
    _zi_dit->set_stream_stop({});
    if (zl.empty()) {
      session()->info(fmt(
          "GenerateImageStage('{}'): Z-Image generation {}; dropping beat",
          this->id(), ctx.stop_requested() ? "stopped" : "failed"));
      co_return;
    }
    auto out = std::make_unique<TensorBeatPayload>();
    out->dtype = TensorBeat::DType::F32;
    out->shape = {ZZ, lh, lw};
    out->resize_contiguous(zl.size());
    std::memcpy(out->as_f32(), zl.data(), zl.size() * sizeof(float));
    tag_output_(*out);
    ++_latents_emitted;
    session()->info(fmt(
        "GenerateImageStage('{}'): Z-Image latent [{}, {}, {}] ({} steps @ "
        "{}x{})", this->id(), ZZ, lh, lw, _scheduler_spec.steps, gen_h,
        gen_w));
    // BEFORE the free, not after: this says what the DiT held while it
    // denoised, which is the number a peer has to size against.
    release_idle_slots_();
    correct_dit_holding_("after a generation");
    if (_release_scratch && _zi_dit) { _zi_dit->release_forward_scratch(); }
    free_z_image_dit_for_decode_(gen_w, gen_h);
    co_await ctx.write(0, std::move(out));
    co_return;
  }

  // ---- FLUX.3: text (+ references) -> the flow map's 16 steps ->
  // [128, H/16, W/16], FLUX.2's VAE's own latent, which vae-decode takes
  // unchanged. ----
  if (_family == "fibo2") {
    // The conditioner's beat is [6, rows, 3 x hidden]: rows are its
    // second axis, not its first.
    if (ctb->shape.size() != 3 ||
        ctb->shape[0] != _fibo2_dit->config().n_bundles() ||
        ctb->shape[2] != _fibo2_dit->config().cap_dim ||
        ctb->dtype != TensorBeat::DType::Bf16) {
      session()->warn(fmt(
          "GenerateImageStage('{}'): Fibo 2.0 wants the diffusion-"
          "conditioner's fibo2 conditioning, [{}, rows, {}] bf16; dropping "
          "beat", this->id(), _fibo2_dit->config().n_bundles(),
          _fibo2_dit->config().cap_dim));
      co_return;
    }
    const int rows = (int)ctb->shape[1];
    const int PD = _fibo2_dit->config().patch_dim();
    // The negative (iport1), the same shape: rows on its second axis too.
    int rows_neg = 0;
    if (!cond_neg.empty()) {
      const auto* ntb = dynamic_cast<const TensorBeatPayload*>(
          _neg_beat.get());
      if (ntb != nullptr && ntb->shape.size() == 3 &&
          ntb->shape[0] == ctb->shape[0] && ntb->shape[2] == ctb->shape[2] &&
          ntb->dtype == TensorBeat::DType::Bf16) {
        rows_neg = (int)ntb->shape[1];
      } else {
        session()->warn(fmt(
            "GenerateImageStage('{}'): the Fibo 2.0 negative is not the "
            "conditioner's fibo2 conditioning; ignoring it", this->id()));
      }
    }
    std::vector<RefLatent> frefs;
    for (const RefLatent& r : _ref) { frefs.push_back(r); }
    _fibo2_dit->set_stream_stop(stopping);
    LatentPreviewer::Scope preview_scope(_preview.get(), ctx, kPreviewPort,
                                         0.0, 0);
    const int lh = gen_h / 16, lw = gen_w / 16;
    const metal_compute::SharedBuffer no_neg;
    const std::vector<float> fl = generate_fibo2_(
        cond, rows, rows_neg > 0 ? cond_neg : no_neg, rows_neg, gen_h, gen_w,
        frefs, init_ptr, step_emitter({PD, lh, lw}));
    _fibo2_dit->set_stream_stop({});
    if (fl.empty()) {
      session()->info(fmt(
          "GenerateImageStage('{}'): Fibo 2.0 generation {}; dropping beat",
          this->id(), ctx.stop_requested() ? "stopped" : "failed"));
      co_return;
    }
    auto out = std::make_unique<TensorBeatPayload>();
    out->dtype = TensorBeat::DType::F32;
    out->shape = {PD, lh, lw};
    out->resize_contiguous(fl.size());
    std::memcpy(out->as_f32(), fl.data(), fl.size() * sizeof(float));
    tag_output_(*out);
    ++_latents_emitted;
    session()->info(fmt(
        "GenerateImageStage('{}'): Fibo 2.0 latent [{}, {}, {}] @ {}x{}",
        this->id(), PD, lh, lw, gen_w, gen_h));
    release_idle_slots_();
    correct_dit_holding_("after a generation");
    if (_release_scratch && _fibo2_dit) {
      _fibo2_dit->release_forward_scratch();
    }
    free_fibo2_dit_for_decode_(gen_w, gen_h);
    co_await ctx.write(0, std::move(out));
    co_return;
  }

  if (_family == "flux3") {
    const int C3 = _flux3_dit->config().in_channels;
    std::vector<RefLatent> frefs;
    for (const RefLatent& r : _ref) { frefs.push_back(r); }
    _flux3_dit->set_stream_stop(stopping);
    LatentPreviewer::Scope preview_scope(_preview.get(), ctx, kPreviewPort,
                                         0.0, 0);
    const int lh3 = gen_h / 16, lw3 = gen_w / 16;
    const std::vector<float> fl =
        generate_flux3_(cond, n_real, gen_h, gen_w, frefs, init_ptr,
                        step_emitter({C3, lh3, lw3}));
    _flux3_dit->set_stream_stop({});
    if (fl.empty()) {
      session()->info(fmt(
          "GenerateImageStage('{}'): FLUX.3 generation {}; dropping beat",
          this->id(), ctx.stop_requested() ? "stopped" : "failed"));
      co_return;
    }
    auto out = std::make_unique<TensorBeatPayload>();
    out->dtype = TensorBeat::DType::F32;
    out->shape = {C3, lh3, lw3};
    out->resize_contiguous(fl.size());
    std::memcpy(out->as_f32(), fl.data(), fl.size() * sizeof(float));
    tag_output_(*out);
    ++_latents_emitted;
    session()->info(fmt(
        "GenerateImageStage('{}'): FLUX.3 latent [{}, {}, {}] @ {}x{}",
        this->id(), C3, lh3, lw3, gen_w, gen_h));
    release_idle_slots_();
    correct_dit_holding_("after a generation");
    if (_release_scratch && _flux3_dit) {
      _flux3_dit->release_forward_scratch();
    }
    free_flux3_dit_for_decode_(gen_w, gen_h);
    co_await ctx.write(0, std::move(out));
    co_return;
  }

  // ---- MOSS-Image: text (+ the negative) -> the reference's own loop ->
  // [128, H/16, W/16], FLUX.2's VAE's latent, which vae-decode takes
  // unchanged (its small decoder included). ----
  if (_family == "moss-image") {
    const int CM = _moss_dit->config().latent_channels;
    _moss_dit->set_stream_stop(stopping);
    LatentPreviewer::Scope preview_scope(_preview.get(), ctx, kPreviewPort,
                                         0.0, 0);
    const int lhm = gen_h / 16, lwm = gen_w / 16;
    const std::vector<float> ml =
        generate_moss_image_(cond, n_real, cond_neg, n_real_neg, gen_h,
                             gen_w, init_ptr, step_emitter({CM, lhm, lwm}));
    _moss_dit->set_stream_stop({});
    if (ml.empty()) {
      session()->info(fmt(
          "GenerateImageStage('{}'): MOSS-Image generation {}; dropping beat",
          this->id(), ctx.stop_requested() ? "stopped" : "failed"));
      co_return;
    }
    auto out = std::make_unique<TensorBeatPayload>();
    out->dtype = TensorBeat::DType::F32;
    out->shape = {CM, lhm, lwm};
    out->resize_contiguous(ml.size());
    std::memcpy(out->as_f32(), ml.data(), ml.size() * sizeof(float));
    tag_output_(*out);
    ++_latents_emitted;
    session()->info(fmt(
        "GenerateImageStage('{}'): MOSS-Image latent [{}, {}, {}] @ {}x{}",
        this->id(), CM, lhm, lwm, gen_w, gen_h));
    release_idle_slots_();
    correct_dit_holding_("after a generation");
    if (_release_scratch && _moss_dit) { _moss_dit->release_forward_scratch(); }
    free_moss_image_dit_for_decode_(gen_w, gen_h);
    co_await ctx.write(0, std::move(out));
    co_return;
  }

  if (_family == "boogu-image") {
    std::vector<RefLatent> brefs;
    for (const RefLatent& r : _ref) { brefs.push_back(r); }
    _boogu_dit->set_stream_stop(stopping);
    const std::vector<float> bl =
        generate_boogu_(cond, n_real, cond_neg, n_real_neg, gen_h, gen_w,
                        init_ptr, brefs, step_emitter({16, lh, lw}));
    _boogu_dit->set_stream_stop({});
    if (bl.empty()) {
      session()->info(fmt(
          "GenerateImageStage('{}'): Boogu-Image generation {}; dropping beat",
          this->id(), ctx.stop_requested() ? "stopped" : "failed"));
      co_return;
    }
    auto out = std::make_unique<TensorBeatPayload>();
    out->dtype = TensorBeat::DType::F32;
    out->shape = {16, lh, lw};
    out->resize_contiguous(bl.size());
    std::memcpy(out->as_f32(), bl.data(), bl.size() * sizeof(float));
    tag_output_(*out);
    ++_latents_emitted;
    session()->info(fmt(
        "GenerateImageStage('{}'): Boogu-Image latent [16, {}, {}] ({} steps "
        "@ {}x{})", this->id(), lh, lw, _scheduler_spec.steps, gen_h, gen_w));
    // Before publishing: on a bounded box the resident DiT would leave the
    // downstream vae-decode no working set. Freed here, reloaded on the next
    // prompt (the latent is already read back, so the DiT is idle).
    release_idle_slots_();
    correct_dit_holding_("after a generation");
    if (_release_scratch && _boogu_dit) {
      _boogu_dit->release_forward_scratch();
    }
    free_boogu_dit_for_decode_(gen_w, gen_h);
    co_await ctx.write(0, std::move(out));
    co_return;
  }


  // Krea-2 reference conditioning (Qwen-Image-Edit multi-reference): the ref
  // latents NOT consumed by img2img. strength>0 uses ref0 as the img2img init,
  // so only ref1 conditions; strength==0 (pure text-to-image) conditions on
  // both. (Reference conditioning only steers a reference-trained checkpoint.)
  std::vector<RefLatent> krefs;
  {
    const std::size_t first = (latent_ptr != nullptr) ? 1 : 0;
    for (std::size_t i = first; i < _ref.size(); ++i) {
      krefs.push_back(_ref[i]);
    }
  }
  _dit->set_stream_stop(stopping);
  // Previews for THIS image, on oport2, ended when process() returns --
  // after the latent is written, so the last step's preview renders while
  // vae-decode is already working. Declines, loading nothing, when the
  // port is unwired or no preview VAE is named.
  LatentPreviewer::Scope preview_scope(_preview.get(), ctx, kPreviewPort,
                                       0.0, 0);
  const std::vector<float> out_latent =
      generate_(cond, n_real, cond_neg, n_real_neg, gen_h, gen_w, init_ptr,
                latent_ptr, krefs, step_emitter({16, lh, lw}));
  _dit->set_stream_stop({});
  if (out_latent.empty()) {
    session()->info(fmt(
        "GenerateImageStage('{}'): generation {}; dropping beat", this->id(),
        ctx.stop_requested() ? "stopped" : "failed"));
    co_return;
  }
  // The DiT is idle now (latent read back): on a memory-bounded box first drop
  // its per-forward scratch (DitScratch activations + dequant/split-K + i8
  // accel buffers, ~1-2 GB at 1024px), which regrows on the next generation.
  release_idle_slots_();
  correct_dit_holding_("after a generation");
  if (_release_scratch && _dit) { _dit->release_forward_scratch(); }
  // The DiT's ~7 GB of quantized weights are resident (wired), not actually
  // evictable, so dropping the scratch alone leaves a 1024px vae-decode short
  // of working set. If it still won't fit, free the whole DiT here (reloaded
  // lazily on the next prompt) BEFORE publishing the latent, so the separate
  // vae-decode stage sees the freed room.
  free_krea2_dit_for_decode_(gen_w, gen_h);

  auto out = std::make_unique<TensorBeatPayload>();
  out->dtype = TensorBeat::DType::F32;
  out->shape = {16, lh, lw};
  out->resize_contiguous(out_latent.size());
  std::memcpy(out->as_f32(), out_latent.data(),
              out_latent.size() * sizeof(float));
  tag_output_(*out);
  ++_latents_emitted;
  session()->info(fmt(
      "GenerateImageStage('{}'): latent [16, {}, {}] ({}+{} {} steps @ "
      "{}x{})", this->id(), lh, lw, _sampler_spec.method,
      _scheduler_spec.type, _scheduler_spec.steps, gen_h, gen_w));
  co_await ctx.write(0, std::move(out));
}

#else   // !VPIPE_BUILD_APPLE_SILICON

Job
GenerateImageStage::initialize(RuntimeContext& ctx)
{
  (void)ctx;
  if (session()) {
    session()->error(fmt(
        "GenerateImageStage('{}'): built without VPIPE_BUILD_APPLE_SILICON; "
        "inert", this->id()));
  }
  co_return;
}

Job
GenerateImageStage::process(RuntimeContext& ctx)
{
  auto in = co_await ctx.read(0);
  (void)in;
  ctx.signal_done();
  co_return;
}

#endif  // VPIPE_BUILD_APPLE_SILICON

VPIPE_REGISTER_STAGE(GenerateImageStage)
VPIPE_REGISTER_SPEC(GenerateImageStage, kSpec)

}

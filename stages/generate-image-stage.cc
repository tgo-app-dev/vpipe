#include "stages/generate-image-stage.h"

#include "stages/model-memory.h"
#include "stages/model-provenance.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "generative-models/image-model-registry.h"
#include "generative-models/shared/dit-block-progress.h"
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
  {"choices", "auto,latch,per_beat"},
};
const ConfigKey kAttrs[] = {
  {.key = "hf_dir", .type = ConfigType::String, .required = false,
   .doc = "Krea-2-Turbo / FLUX.2 model dir (text_encoder/, transformer/, "
          "tokenizer/); an original or model-quantize'd (self-contained) "
          "pipeline. OPTIONAL: a model-select source on the model iport "
          "overrides it",
   .suggest_db = kModelRegistryDb, .suggest_db_type =
       "krea2,flux2,qwen-image,qwen-image-edit,qwen-image-21,"
       "boogu-image,boogu-image-edit,z-image,vosr",
   .model_channel = "diffusion-model"},
  {.key = "dit_dir", .type = ConfigType::String, .required = false,
   .doc = "override DiT dir (e.g. a quantized 4/8-bit DiT); else <hf_dir>/transformer",
   .suggest_db = kModelRegistryDb,
   .suggest_db_type =
       "krea2-dit,flux2-dit,qwen-image-edit-dit,qwen-image-21-dit,"
       "boogu-image-dit,z-image-dit"},
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
   .doc = "initial-noise RNG seed (default 0)"},
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
  {.key = "i8_gemm", .type = ConfigType::Bool, .required = false,
   .doc = "accelerated mode (LOSSY): dynamic-int8 GEMMs for the DiT's big "
          "block matmuls, ~2x their f16 rate at int8 quality; IGNORED "
          "without NAX matmul2d (matrix-core GPU + kernels). Default "
          "false; env VPIPE_I8_GEMM overrides"},
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
          "block count would not. HIGHER keeps fewer blocks: quality falls "
          "and speed rises, monotonically in both.\n"
          "0.7 IS THE DEFAULT HERE, against 1.0 on generate-video, and the "
          "difference is measured rather than cautious. MEASURED on "
          "Krea-2 at 1024^2 (joint 4160 tokens, 65 routing blocks, 8 "
          "routed layers), velocity rel-L2 against the dense attention: "
          "0.0529 at tau <= 0.815 and 0.0680 at tau >= 0.82 -- a 29% step "
          "inside 0.005 of a standard deviation, with flat plateaus on "
          "both sides. 1.0 sits on the far side of that step, and the "
          "image it produces is a visibly different sample rather than a "
          "slightly softer one (PSNR 15.9 dB against dense, where tau 0.7 "
          "gives 18.9 dB and two dense runs differ by 66.1 dB). 0.7 sits "
          "mid-plateau, so it is not a knife edge: anything from 0.5 to "
          "0.815 measures the same.\n"
          "IT COSTS LITTLE OF THE SPEEDUP, which is why the step is worth "
          "staying under: on the same box the denoise ran 143 s dense, "
          "132 s at tau 0.7 and 133 s at tau 1.0 -- single samples with "
          "~1.4% run-to-run drift, so read the two Sol arms as equal. "
          "generate-video keeps 1.0 because its sequences are ~5x longer, "
          "where the published profile's 1.0/1.25/1.5 was measured and "
          "where the step above has not been re-measured",
   .def_real = 0.7},
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
          "path to one. Krea-2 and FLUX.2 only. LOAD-time: it decides how "
          "the blocks are BUILT, so a beat that changes it once the DiT "
          "is up is reported and ignored. A `lora` on the family's "
          "model-config beat OVERRIDES this",
   .suggest_db = kModelRegistryDb,
   // Both families this stage can adapt. A comma list so the picker
   // offers either and refuses the rest -- a Krea-2 adapter on a FLUX.2
   // DiT binds nothing and says so, but not until it has been fetched.
   .suggest_db_type = "krea2-lora,flux2-lora"},
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
   .suggest_db_type = "krea2-lora,flux2-lora"},
  {.key = "lora2_scale", .type = ConfigType::Real, .required = false,
   .doc = "`lora2`'s strength, per FORWARD and independent of "
          "`lora_scale` -- which is the point of a second slot: one "
          "adapter stays where it was trained while the other is swept. "
          "0 skips its two GEMMs", .def_real = 1.0},
  {.key = "reference_mode", .type = ConfigType::String, .required = false,
   .doc = "how the reference latents pair with the conditioning beats. "
          "\"latch\" holds the FIRST reference for the whole run, which is "
          "what an edit graph wants: one source picture, many prompts. "
          "\"per_beat\" reads a fresh reference for every beat, which is "
          "what a RESTORATION graph wants: the reference IS the input and "
          "changes every time, and latching it would restore a whole folder "
          "from the first picture without ever saying so. \"auto\" (the "
          "default) is per_beat for a restorer and latch for everything "
          "else", .def_str = "auto", .extra = kReferenceModeChoices},
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
};

// The model iport (a model-select source) overrides hf_dir. Inserted after the
// conditioning + neg_conditioning inputs, so it is iport2; sampler / scheduler /
// ref_latent0 / ref_latent1 follow at iports 3 / 4 / 5 / 6. (Referenced only
// from the Apple-gated code; maybe_unused for the inert non-Apple build.)
[[maybe_unused]] constexpr unsigned kModelPort = 2;
[[maybe_unused]] constexpr unsigned kModelCfgPort = 7;
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
};
const StageSpec kSpec = {
  .type_name = "generate-image",
  .doc       = "Diffusion DiT denoiser: conditioning (from a diffusion-"
               "conditioner stage) -> family MMDiT -> FlowMatchEuler -> latent, "
               "on the metal-compute backend. The denoiser half of the split "
               "(feed vae-decode).",
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
  return model_memory::streaming_floor_bytes(dit, kStems);
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
    const std::string rm = attr_str("reference_mode");
    if (rm == "latch")         { _ref_mode = RefMode::kLatch; }
    else if (rm == "per_beat") { _ref_mode = RefMode::kPerBeat; }
    else if (!rm.empty() && rm != "auto") {
      // Deferred-validated config: warn and take the default.
      session()->warn(fmt(
          "GenerateImageStage('{}'): reference_mode '{}' is not "
          "auto|latch|per_beat; using auto", this->id(), rm));
    }
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
  _sage.enabled      = attr_bool("sage_attn");
  _sage.dense_layers = (int)attr_int("sage_dense_layers");
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
  genai::accel::set_flag(&_accel, genai::accel::kSageAttn, _sage.enabled);
  genai::accel::set_integer(&_accel, genai::accel::kSageDenseLayers,
                            _sage.dense_layers);
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
  _lora[0].path  = attr_str("lora");
  _lora[0].scale = attr_real("lora_scale");
  _lora[1].path  = attr_str("lora2");
  _lora[1].scale = attr_real("lora2_scale");
  _seed   = (std::uint64_t)attr_int("seed");
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

GenerateImageStage::~GenerateImageStage() = default;

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
// A checkpoint this stage RECOGNISES and no longer implements. Returns
// the plugin that does, or empty when the root is not one of these.
//
// It exists so that removing a family from the tree costs a message
// rather than a wrong model: `t2i_family_` falls through to "krea2", so
// without this a Mage-Flow root would load the Krea-2 path at full cost
// and denoise a 4B checkpoint through a 12B config.
std::string
unclaimed_family_(const std::string& transformer_dir)
{
  namespace fs = std::filesystem;
  std::ifstream in(fs::path(transformer_dir) / "config.json");
  if (!in) { return {}; }
  FlexData fd = FlexData::from_json(in);
  if (!fd.is_object()) { return {}; }
  auto obj = fd.as_object();
  if (!obj.contains("_class_name")) { return {}; }
  const std::string cls(obj.at("_class_name").as_string(""));
  if (cls == "MageFlow") { return "vpipe-mage-flow"; }
  return {};
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
         family == "qwen-image-21" || family == "z-image";
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
  if (family == "flux2" || family == "qwen-image-21") { return 16; }
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
// "MageFlow" is deliberately ABSENT and is handled by the registry
// instead: that family moved out of this tree into the vpipe-mage-flow
// plugin. It still has to be RECOGNISED, though -- see
// unclaimed_family_() -- because the fall-through here is "krea2", and
// silently loading a 12B Krea-2 config over a 4B Mage-Flow checkpoint
// is the worst failure this stage has.
std::string
t2i_family_(const std::string& transformer_dir)
{
  namespace fs = std::filesystem;
  std::ifstream in(fs::path(transformer_dir) / "config.json");
  if (in) {
    FlexData fd = FlexData::from_json(in);
    if (fd.is_object()) {
      auto obj = fd.as_object();
      if (obj.contains("_class_name")) {
        const std::string cls(obj.at("_class_name").as_string(""));
        if (cls == "Flux2Transformer2DModel") { return "flux2"; }
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
      if (o.contains("block_out_channels")) {
        FlexData bo = o.at("block_out_channels");
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
  return (std::size_t)h * (std::size_t)w * (std::size_t)base * 2 * 7;
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
qwen_decode_peak_(int w, int h, int base)
{
  if (w <= 0 || h <= 0 || base <= 0) { return 0; }
  const std::size_t hw = (std::size_t)h * (std::size_t)w;
  const std::size_t acts = hw * (std::size_t)base * 2 * 6;
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
      !_zi_dit) {
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
  _cfg_latched       = false;
  _model_cfg         = FlexData{};
  // The reference latents latch on FIRST arrival and are cached for
  // every later prompt (`_ref[]`), which is right within a run and
  // wrong across one: a non-empty cache makes the iport5/iport6 read
  // conditional, so on the next launch the stage never consumes the
  // reference beat the vae-encode upstream just produced. That beat
  // then sits on the wire forever (the edge shows a permanent backlog)
  // and the run silently re-uses the PREVIOUS image's reference.
  _ref[0] = RefLatent{};
  _ref[1] = RefLatent{};
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
  const std::string dit = planning_dit_dir_(root);
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
  }
  if (!held_by_family) {
    m.hold(dit, model_memory::dir_weights_bytes(dit),
           dit_floor_bytes_(root, dit));
  }
  m.hold(enc, model_memory::dir_weights_bytes(enc));
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
    std::vector<ResourceClaim> pout = fam->declare_resources(root);
    for (auto& c : model_memory::weight_claims({enc})) {
      pout.push_back(std::move(c));
    }
    return pout;
  }
  const std::string dit = planning_dit_dir_(root);
  std::vector<ResourceClaim> out{
      model_memory::weight_claim_streamable(dit,
                                            dit_floor_bytes_(root, dit))};
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
  std::string perr;
  // The adapter keys, read the same way by every family that has one.
  // The PATH is a LOAD-TIME argument -- it decides whether a
  // fused-activation weave could be built at all -- so a beat that
  // changes it once the DiT is up is reported rather than dropped in
  // silence. The STRENGTH is live: it rides the adapter's GEMM as a
  // constant, so it can be swept without rebuilding anything.
  //
  // Both SLOTS, read the same way. Slot 0 is the plain `lora` keys and
  // slot 1 the `lora2` ones, so a graph that names one adapter reads
  // exactly as it always did and a second is additive. Which slot holds
  // what is the caller's order, not a role the model knows.
  static const char* const kKeys[kLoraSlots][2] = {
    {"lora",  "lora_scale"},
    {"lora2", "lora2_scale"},
  };
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
      const std::string lp = o.contains(kKeys[i][0])
          ? std::string(o.at(kKeys[i][0]).as_string("")) : sl.path;
      const double ls = o.contains(kKeys[i][1])
          ? o.at(kKeys[i][1]).as_real(1.0) : sl.scale;
      if (loaded && lp != sl.path) {
        session()->warn(fmt(
            "GenerateImageStage('{}'): the {} DiT is already loaded, so the "
            "model_config beat's `{}` is IGNORED -- it is a load-time "
            "argument. `{}` still applies",
            this->id(), what, kKeys[i][0], kKeys[i][1]));
      } else {
        sl.path = lp;
      }
      if (ls != sl.scale) {
        sl.scale = ls;
        push_scale(i, (float)ls);
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
    }
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
  // A CHECKPOINT THIS STAGE RECOGNISES AND NO LONGER IMPLEMENTS. Checked
  // only once the registry has declined it, so a loaded plugin never
  // sees this -- and refused rather than warned, because the fall-through
  // below is "krea2": without it a Mage-Flow root would load a 12B
  // Krea-2 config over a 4B checkpoint, spend minutes and emit noise.
  if (_plugin_family == nullptr) {
    const std::string plug = unclaimed_family_(dit_dir);
    if (!plug.empty()) {
      session()->error(fmt(
          "GenerateImageStage('{}'): '{}' is a {} checkpoint, whose family "
          "is no longer built in -- load the {} plugin (vpipe --plugin "
          "<path>) and re-run. The stage is inert rather than guessing a "
          "family, because the guess would be a different model at full "
          "cost", this->id(), root, "Mage-Flow", plug));
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
  if (_plugin_family == nullptr && _family != "vosr") {
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
    genai::ImageModelCreateArgs args;
    args.root       = root;
    args.model_type = resolve_model(session(), _hf_dir).model_type;
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
    bool stream_blocks;
    {
      const auto plan = model_memory::plan_streaming(
          session(), dit_dir, enc_dir, model_memory::kStreamHeadroom);
      stream_blocks = plan.stream;
      if (const char* e = std::getenv("VPIPE_FLUX2_STREAM")) {
        stream_blocks = (std::atoi(e) != 0);
      }
      session()->log_debug(fmt(
          "GenerateImageStage('{}'): FLUX.2 footprint {} GB (others {} GB) "
          "+ {} GB headroom vs {} GB RAM -> {}", this->id(),
          plan.footprint >> 30, plan.others >> 30,
          model_memory::kStreamHeadroom >> 30, phys_ram() >> 30,
          stream_blocks ? "STREAM blocks" : "PRELOAD"));
    }
    genai::MetalFlux2Transformer::Config fcfg;
    fcfg.i8_gemm = _i8_gemm;
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
    _flux2_dit = genai::MetalFlux2Transformer::load(
        weight_set_(dit_dir), mc, fcfg, stream_blocks,
        lora_specs_<genai::MetalFlux2Transformer::LoraSpec>());
    if (!_flux2_dit) {
      session()->error(fmt(
          "GenerateImageStage('{}'): failed to load the FLUX.2 DiT from '{}'; "
          "inert", this->id(), dit_dir));
      return;
    }
    // When the DiT had to stream (box can't hold encoder + DiT together), drop
    // the DiT's per-forward scratch after each generation so it doesn't crowd
    // out a large downstream VAE decode.
    _release_scratch = stream_blocks;
    if (stream_blocks) { revise_dit_declaration_(dit_dir); }
    // Cache the load params + VAE base so a memory-driven free after a
    // generation (free_flux2_dit_for_decode_) can reload identically.
    _flux2_dit_dir  = dit_dir;
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
    bool   stream_blocks = plan.stream;
    if (const char* e = std::getenv("VPIPE_QIE_STREAM")) {
      stream_blocks = (std::atoi(e) != 0);
    }
    session()->log_debug(fmt(
        "GenerateImageStage('{}'): Qwen-Image-Edit footprint {} GB (others {} "
        "GB) + {} GB headroom vs {} GB RAM -> {}", this->id(),
        plan.footprint >> 30, plan.others >> 30,
        model_memory::kStreamHeadroom >> 30, phys_ram() >> 30,
        stream_blocks ? "STREAM blocks" : "PRELOAD"));
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
    _release_scratch = stream_blocks;
    if (stream_blocks) { revise_dit_declaration_(dit_dir); }
    // Cache the load params + VAE base so a memory-driven free after a
    // generation (free_qie_dit_for_decode_) can reload identically. The
    // Config is default-constructed above and everything else comes from
    // the checkpoint's own config.json, so the dir and the streaming
    // decision are the whole of it.
    _qie_dit_dir = dit_dir;
    _qie_stream  = stream_blocks;
    // AutoencoderKLQwenImage, which is the SAME VAE Krea-2 runs (see
    // vae_family_ in vae-decode-stage.cc: anything that is not Flux2 /
    // Mage / Wan / MiniMax-H3 opens as one). So it sizes from `base_dim`,
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
    bool stream_blocks = plan.stream;
    if (const char* e = std::getenv("VPIPE_QWEN_IMAGE21_STREAM")) {
      stream_blocks = (std::atoi(e) != 0);
    }
    session()->log_debug(fmt(
        "GenerateImageStage('{}'): Qwen-Image-2.1 footprint {} GB (others {} "
        "GB) + {} GB headroom vs {} GB RAM -> {}", this->id(),
        plan.footprint >> 30, plan.others >> 30,
        model_memory::kStreamHeadroom >> 30, phys_ram() >> 30,
        stream_blocks ? "STREAM blocks" : "PRELOAD"));
    _qi21_dit_dir = dit_dir;
    // The holding correct_dit_holding_ moves as residency grows.
    _dit_holding_dir = dit_dir;
    _qi21_stream = stream_blocks;
    // Its VAE is the Qwen-Image one generalized, so the same reader
    // serves it -- but the DECODER's base, which is 144 here against
    // the encoder's 96, because every peak this feeds is a decode.
    _vae_base = krea2_vae_dec_base_(root);
    _release_scratch = stream_blocks;
    if (stream_blocks) { revise_dit_declaration_(dit_dir); }
    if (stream_blocks) {
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
    bool stream_blocks = plan.stream;
    if (const char* e = std::getenv("VPIPE_Z_IMAGE_STREAM")) {
      stream_blocks = (std::atoi(e) != 0);
    }
    session()->log_debug(fmt(
        "GenerateImageStage('{}'): Z-Image footprint {} GB (others {} GB) + "
        "{} GB headroom vs {} GB RAM -> {}", this->id(),
        plan.footprint >> 30, plan.others >> 30,
        model_memory::kStreamHeadroom >> 30, phys_ram() >> 30,
        stream_blocks ? "STREAM blocks" : "PRELOAD"));
    _zi_dit_dir = dit_dir;
    // The holding correct_dit_holding_ moves as residency grows.
    _dit_holding_dir = dit_dir;
    _zi_stream = stream_blocks;
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
    _release_scratch = stream_blocks;
    if (stream_blocks) { revise_dit_declaration_(dit_dir); }
    if (stream_blocks) {
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
  } else if (_family == "boogu-image") {
    // Boogu's 10B NextDiT. At ~20 GB bf16 it streams on any box that cannot
    // hold it beside the resident Qwen3-VL mllm (~16 GB), which is every box
    // short of ~48 GB; the three refiner stacks stay resident either way.
    // VPIPE_BOOGU_STREAM / VPIPE_BOOGU_PIN_FRAC override.
    const auto plan = model_memory::plan_streaming(
        session(), dit_dir, enc_dir, model_memory::kStreamHeadroom);
    bool   stream_blocks = plan.stream;
    if (const char* e = std::getenv("VPIPE_BOOGU_STREAM")) {
      stream_blocks = (std::atoi(e) != 0);
    }
    session()->log_debug(fmt(
        "GenerateImageStage('{}'): Boogu footprint {} GB (others {} GB) + {} "
        "GB headroom vs {} GB RAM -> {}", this->id(), plan.footprint >> 30,
        plan.others >> 30, model_memory::kStreamHeadroom >> 30,
        phys_ram() >> 30,
        stream_blocks ? "STREAM blocks" : "PRELOAD"));
    // Cache the load params so the stage can free the DiT for a big decode and
    // reload it on the next prompt (free_boogu_dit_for_decode_).
    _boogu_dit_dir = dit_dir;
    _boogu_stream = stream_blocks;
    _vae_base = flux2_vae_base_(root);   // Boogu's VAE is a plain AutoencoderKL
    _release_scratch = stream_blocks;
    if (stream_blocks) { revise_dit_declaration_(dit_dir); }
    if (stream_blocks) {
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
    bool stream_blocks;
    {
      const auto plan = model_memory::plan_streaming(
          session(), dit_dir, enc_dir, model_memory::kStreamHeadroom);
      stream_blocks = plan.stream;
      if (const char* e = std::getenv("VPIPE_KREA2_STREAM")) {
        stream_blocks = (std::atoi(e) != 0);
      }
      session()->log_debug(fmt(
          "GenerateImageStage('{}'): Krea-2 footprint {} GB (others {} GB) "
          "+ {} GB headroom vs {} GB RAM -> {}", this->id(),
          plan.footprint >> 30, plan.others >> 30,
          model_memory::kStreamHeadroom >> 30, phys_ram() >> 30,
          stream_blocks ? "STREAM blocks" : "PRELOAD"));
    }
    genai::MetalKrea2Transformer::Config kcfg;
    kcfg.i8_gemm = _i8_gemm;
    kcfg.sage = _sage;
    kcfg.sol  = _sol;
    apply_ane_(kcfg);
    _dit = genai::MetalKrea2Transformer::load(
        weight_set_(dit_dir), mc, kcfg, stream_blocks,
        lora_specs_<genai::MetalKrea2Transformer::LoraSpec>());
    if (!_dit) {
      session()->error(fmt(
          "GenerateImageStage('{}'): failed to load the DiT from '{}'; inert",
          this->id(), dit_dir));
      return;
    }
    // Cache the resolved dir (reload after a decode-driven free) + the VAE base
    // channels for the decode-peak estimate (shared MetalKrea2Vae -> base_dim).
    _krea2_dit_dir = dit_dir;
    _krea2_stream  = stream_blocks;
    _vae_base = krea2_vae_base_(root);
    if (stream_blocks) { revise_dit_declaration_(dit_dir); }
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
        stream_blocks ||
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
  fcfg.sage = _sage;
  // The RELOAD path too. A tier set here and not there comes back
  // missing on the generation after a decode-driven free, which is the
  // quietest way to lose it.
  fcfg.sol  = _sol;
  _flux2_params.apply_to(fcfg);
  // The streaming flag the first load used, and the adapter with it. A
  // reload that dropped either would come back preloaded, or
  // un-adapted, on the graph that asked for both.
  _flux2_dit = genai::MetalFlux2Transformer::load(
      weight_set_(_flux2_dit_dir), mc, fcfg, _flux2_stream,
      lora_specs_<genai::MetalFlux2Transformer::LoraSpec>());
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
  cfg.sage = _sage;
  cfg.sol = _sol;
  cfg.ane_ffn = genai::accel::flag(&_accel, genai::accel::kAneFfn);
  cfg.ane_qkv = genai::accel::flag(&_accel, genai::accel::kAneQkv);
  cfg.ane_rows = (int)genai::accel::integer(&_accel, genai::accel::kAneRows,
                                            0);
  cfg.ane_layers = (int)genai::accel::integer(&_accel,
                                              genai::accel::kAneLayers, 0);
  _qi21_dit = genai::MetalQwenImage21Transformer::load(
      weight_set_(_qi21_dit_dir), mc, cfg, _qi21_stream);
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
      qwen_decode_peak_(gen_w, gen_h, _vae_base > 0 ? _vae_base : 96);
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
  cfg.sage = _sage;
  cfg.sol = _sol;
  cfg.ane_ffn = genai::accel::flag(&_accel, genai::accel::kAneFfn);
  cfg.ane_qkv = genai::accel::flag(&_accel, genai::accel::kAneQkv);
  cfg.ane_rows = (int)genai::accel::integer(&_accel, genai::accel::kAneRows,
                                            0);
  cfg.ane_layers = (int)genai::accel::integer(&_accel,
                                              genai::accel::kAneLayers, 0);
  _zi_dit = genai::MetalZImageTransformer::load(
      weight_set_(_zi_dit_dir), mc, cfg, _zi_stream,
      lora_specs_<genai::MetalZImageTransformer::LoraSpec>());
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
  const std::size_t budget = mc->memory_budget().available_physical;
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
  kcfg.sage = _sage;
  apply_ane_(kcfg);
  // The streaming flag the first load used, and the adapter with it. A
  // reload that dropped either would come back preloaded, or un-adapted,
  // on the graph that asked for both.
  _dit = genai::MetalKrea2Transformer::load(
      weight_set_(_krea2_dit_dir), mc, kcfg, _krea2_stream,
      lora_specs_<genai::MetalKrea2Transformer::LoraSpec>());
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
    sampler.step(i, packed,
                 prof ? genai::FlowSampler::DenoiseFn(denoise_p) : denoise);
    if (!dit_ok) { return {}; }
    if (emit_step) { emit_step(unpack(packed)); }
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
    sampler.step(i, packed,
                 prof ? genai::FlowSampler::DenoiseFn(denoise_p) : denoise);
    if (!dit_ok) { return {}; }
    if (emit_step) { emit_step(unpack(packed)); }
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
      session()->warn(fmt(
          "GenerateImageStage('{}'): reference {} is a {}x{} latent but the "
          "conditioner saw a {}x{} grid -- the vae-encode and the "
          "conditioner must be given the SAME resized picture",
          this->id(), (int)i, refs[i].w, refs[i].h, gw, gh));
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

  // ---- schedule ------------------------------------------------------
  genai::FlowSchedulerSpec sched = _scheduler_spec;
  if (!_scheduler_latched) {
    // FlowMatchEulerDiscreteScheduler with dynamic shifting, straight
    // off the checkpoint's scheduler_config: the base grid is
    // linspace(1, 1/S, S) and shift_terminal stretches the tail to 0.02.
    sched.dynamic_shift = true;
    sched.shift_type = "exponential";
    sched.base_shift = 0.5; sched.max_shift = 0.9;
    sched.base_seq = 256; sched.max_seq = 8192;
    sched.shift_terminal = 0.02;
    sched.steps = _steps > 0 ? _steps : 40;
  }
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

  const bool cfg = !txt_neg.empty() && n_real_neg > 0 &&
                   _guidance_scale != 1.0;
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
        qwen_decode_peak_(gen_w, gen_h, _vae_base > 0 ? _vae_base : 144);
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
  const bool use_kv = _qi21_use_kv_cache && lay.prefix_len > 0;
  int step_i = 0;

  auto run = [&](const SharedBuffer& txt, int rows,
                 genai::MetalQwenImage21Transformer::KvCache* kv, float sigma)
      -> SharedBuffer {
    (void)rows;
    genai::MetalQwenImage21Transformer::Request req;
    req.latents = &latbuf;
    req.txt = &txt;
    req.layout = &lay;
    req.timestep = sigma;
    if (use_kv) {
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

  auto denoise = [&](const std::vector<float>& cand,
                     double sigma) -> std::vector<float> {
    // The target block's rows are the TAIL of the packed latent.
    auto* lb = static_cast<std::uint16_t*>(latbuf.contents());
    for (std::size_t k = 0; k < cand.size(); ++k) {
      lb[tgt_off + k] = f32_to_bf16_(cand[k]);
    }
    SharedBuffer v = run(txt_pos, n_real, &kv_pos, (float)sigma);
    if (!dit_ok || v.empty()) { return {}; }
    std::vector<float> out((std::size_t)img_seq * IC);
    const auto* vp = static_cast<const std::uint16_t*>(v.contents());
    for (std::size_t k = 0; k < out.size(); ++k) {
      out[k] = bf16_to_f32_(vp[k]);
    }
    prog.end_forward();
    if (cfg) {
      SharedBuffer nv = run(txt_neg, n_real_neg, &kv_neg, (float)sigma);
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
    sampler.step(i, packed, denoise);
    if (!dit_ok) { return {}; }
    if (emit_step) { emit_step(unpack(packed)); }
    prog.end_step(i);
  }
  const double secs = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();
  session()->info(fmt(
      "GenerateImageStage('{}'): Qwen-Image-2.1 latent generated in {:.2f}s "
      "({} steps, joint {}, prefix {}{})", this->id(), secs, S,
      lay.joint_len, lay.prefix_len,
      use_kv ? ", prefix KV cached" : ""));
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
  if (!_ref[0].empty() || !_ref[1].empty()) {
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
    return out;
  };

  auto unpack = [&](const std::vector<float>& pk) {
    std::vector<float> latent((std::size_t)Z * lh * lw);
    genai::zimage::unpack_latents(pk.data(), Z, lh, lw, P, latent.data());
    return latent;
  };

  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < S; ++i) {
    sampler.step(i, packed, denoise);
    if (!dit_ok) { return {}; }
    if (emit_step) { emit_step(unpack(packed)); }
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
GenerateImageStage::tag_model_(TensorBeat& tb) const
{
  // `_hf_dir` is the reference the user actually named (a registry key like
  // "local/Mage-Flow-Edit-Turbo-8bit", or a path), which is the meaningful
  // identity of the generator -- not the resolved directory.
  provenance::set_model_name(tb.sideband, _hf_dir);
}

Job
GenerateImageStage::process(RuntimeContext& ctx)
{
  auto* mc = session()->services()->metal_compute();
  // The model config FIRST, before anything can load: `klein_kv` is an
  // argument to the DiT's construction, not a per-step knob, so a beat
  // read after ensure_loaded_ would arrive too late to matter and would
  // do so silently.
  //
  // Latched, but RE-READ whenever another beat is waiting: a config
  // source with no trigger emits once for the whole run, while a
  // trigger-driven one emits per request. Blocking on the first and
  // polling after serves both. (A later beat cannot change `klein_kv` on
  // an already-built DiT -- see the note where that is reported.)
  if (ctx.num_iports() > kModelCfgPort && ctx.iport_connected(kModelCfgPort) &&
      (!_cfg_latched || ctx.backlog(kModelCfgPort) > 0)) {
    auto gb = co_await ctx.read(kModelCfgPort);
    const bool first = !_cfg_latched;
    _cfg_latched = true;
    if (const auto* gfd =
            gb ? dynamic_cast<const FlexDataPayload*>(gb.get()) : nullptr) {
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
  auto in = co_await ctx.read(0);
  if (!in) { ctx.signal_done(); co_return; }
  // Did the conditioner REFUSE this prompt? Read the flag off the beat up
  // front; the refusal itself is emitted further down, once the output size
  // is known.
  bool cond_blocked = false;
  if (const auto* stb = dynamic_cast<const TensorBeatPayload*>(in.get())) {
    if (stb->sideband.is_object()) {
      FlexData sb = stb->sideband;        // as_object() is a view: keep it
      auto o = sb.as_object();
      cond_blocked = o.contains("content_blocked")
                     && o.at("content_blocked").as_bool(false);
      // Qwen-Image-2.1's joint-sequence bookkeeping. Read here rather
      // than in the family branch because the beat is consumed by then,
      // and read EVERY beat -- a second prompt with a different
      // reference count has a different mask, and a latched one would
      // lay the new sequence out with the old prompt's slots.
      if (_family == "qwen-image-21") {
        _qi21_slots.clear();
        _qi21_ref_gh.clear();
        _qi21_ref_gw.clear();
        auto ints = [&](const char* k, std::vector<int>& dst) {
          if (!o.contains(k)) { return; }
          const FlexData f = o.at(k);
          if (!f.is_array()) { return; }
          auto a = f.as_array();
          for (std::size_t i = 0; i < a.size(); ++i) {
            dst.push_back((int)a.at(i).as_int(0));
          }
        };
        std::vector<int> sl;
        ints("img_slots", sl);
        _qi21_slots.reserve(sl.size());
        for (int v : sl) { _qi21_slots.push_back(v != 0 ? 1 : 0); }
        ints("ref_grid_h", _qi21_ref_gh);
        ints("ref_grid_w", _qi21_ref_gw);
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
      : _family == "vosr" ? (bool)_vosr
      : (bool)_dit;
  if (!have_dit) {
    session()->warn(fmt(
        "GenerateImageStage('{}'): models not loaded; dropping beat", this->id()));
    co_return;
  }
  // iport0: the conditioning tensor from a diffusion-conditioner stage
  // (family-shaped + typed; rows = shape[0]). Copy it into a metal buffer.
  const auto* ctb = dynamic_cast<const TensorBeatPayload*>(in.get());
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
  // classifier-free guidance. The conditioner enqueues oport1 BEFORE oport0, so
  // when iport0 arrives its paired negative is already in this port's FIFO --
  // the non-blocking backlog gate reads it reliably (and never stalls when no
  // negative is wired).
  metal_compute::SharedBuffer cond_neg;
  int n_real_neg = 0;
  if (ctx.num_iports() >= 2 && ctx.iport_connected(1) && ctx.backlog(1) > 0) {
    auto nb = co_await ctx.read(1);
    const auto* ntb = nb ? dynamic_cast<const TensorBeatPayload*>(nb.get())
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

  // Reference latents on iport5 / iport6: latch the FIRST reference on each
  // connected port (blocking, like the sampler/scheduler specs) and cache it
  // (`_ref[]`), so the reference reliably pairs with the prompt (a non-blocking
  // poll would race the producer) and a fixed reference supplied once is reused
  // for every later prompt. FLUX.2 threads them as multi-reference conditioning
  // tokens (below); Krea-2 uses ref latent 0 as the img2img init and ignores
  // ref latent 1.
  //
  // ...UNLESS the graph said otherwise. A restoration graph hands this
  // stage a new reference on every beat and the picture IS the input, so
  // holding the first would restore a whole folder from it and never say
  // so. See the `reference_mode` config key.
  const bool ref_fresh = ref_per_beat_();
  for (int r = 0; r < 2; ++r) {
    const int port = 5 + r;
    if ((int)ctx.num_iports() > port && ctx.iport_connected(port) &&
        (_ref[r].empty() || ref_fresh)) {
      auto rb = co_await ctx.read(port);
      const auto* tb = rb ? dynamic_cast<const TensorBeatPayload*>(rb.get())
                          : nullptr;
      if (tb != nullptr && tb->dtype == TensorBeat::DType::F32 &&
          tb->shape.size() == 3 && tb->shape[0] > 0 && tb->shape[1] > 0 &&
          tb->shape[2] > 0) {
        const auto bytes = tb->materialize_contiguous();
        const std::size_t n =
            (std::size_t)tb->shape[0] * tb->shape[1] * tb->shape[2];
        const float* fp = reinterpret_cast<const float*>(bytes.data());
        _ref[r].chw.assign(fp, fp + n);
        _ref[r].c = (int)tb->shape[0];
        _ref[r].h = (int)tb->shape[1];
        _ref[r].w = (int)tb->shape[2];
        session()->log_debug(fmt(
            "GenerateImageStage('{}'): reference latent {} = [{}, {}, {}]",
            this->id(), r, _ref[r].c, _ref[r].h, _ref[r].w));
      } else if (rb) {
        session()->warn(fmt(
            "GenerateImageStage('{}'): ref_latent{} must be an f32 [C,H,W] "
            "TensorBeat; got {}, ignoring", this->id(), r, rb->describe()));
      }
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
    const RefLatent& r0 = _ref[0];
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
    if (gen_h <= 0 || gen_w <= 0) { gen_h = 256; gen_w = 256; }
    session()->log_debug(fmt(
        "GenerateImageStage('{}'): output size {}x{} ({})", this->id(), gen_w,
        gen_h, r0.empty() ? "no reference, default" : "from ref_latent0"));
  }
  // ---- Content-policy refusal ------------------------------------------
  // The Mage-Flow conditioner screens every prompt against the model's own
  // policy classifier and tags a refused one `content_blocked`. Honour it
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
    o.insert_or_assign("content_blocked", FlexData::make_bool(true));
    o.insert_or_assign("refusal_height", FlexData::make_int(gen_h));
    o.insert_or_assign("refusal_width",  FlexData::make_int(gen_w));
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
  if (_family == "krea2" && _strength > 0.0 && !_ref[0].empty()) {
    const RefLatent& rr = _ref[0];
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

  // Latch the sampler / scheduler specs off iport3 / iport4 (once each): the
  // `diffusion-sampler-select` / `scheduler-select` sources emit a single spec
  // beat, which
  // we cache and reuse for every subsequent prompt.
  if (!_sampler_latched && ctx.num_iports() >= 4 && ctx.iport_connected(3)) {
    auto sb = co_await ctx.read(3);
    _sampler_latched = true;
    const auto* sfd = dynamic_cast<const FlexDataPayload*>(sb.get());
    if (sfd != nullptr) {
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
  if (!_scheduler_latched && ctx.num_iports() >= 5 && ctx.iport_connected(4)) {
    auto cb = co_await ctx.read(4);
    _scheduler_latched = true;
    const auto* cfd = dynamic_cast<const FlexDataPayload*>(cb.get());
    if (cfd != nullptr) {
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
      tag_model_(*so);
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
    req.seed   = _seed + (std::uint64_t)_latents_emitted;
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
    if (!_ref[0].empty()) {
      req.ref_latent0 = _ref[0].chw.data();
      req.ref0_shape  = {_ref[0].c, _ref[0].h, _ref[0].w};
    }
    if (!_ref[1].empty()) {
      req.ref_latent1 = _ref[1].chw.data();
      req.ref1_shape  = {_ref[1].c, _ref[1].h, _ref[1].w};
    }
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
    tag_model_(*out);
    ++_latents_emitted;
    session()->info(fmt(
        "GenerateImageStage('{}'): '{}' latent {} floats ({} steps @ {}x{})",
        this->id(), _family, res.latent.size(), req.steps, gen_w, gen_h));
    bar.finish();
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
    if (!_ref[0].empty()) { frefs.push_back(_ref[0]); }
    if (!_ref[1].empty()) { frefs.push_back(_ref[1]); }
    const int Cdit = _flux2_dit->config().in_channels;
    const int fgh = gen_h / 16, fgw = gen_w / 16;
    _flux2_dit->set_stream_stop(stopping);
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
    tag_model_(*out);
    ++_latents_emitted;
    session()->info(fmt(
        "GenerateImageStage('{}'): FLUX.2 latent [{}, {}, {}] ({} steps @ {}x{})",
        this->id(), Cdit, fgh, fgw, _scheduler_spec.steps, gen_h, gen_w));
    // Free the DiT if the downstream vae-decode won't fit alongside it (before
    // publishing the latent, so the decode stage sees the freed working set).
    free_flux2_dit_for_decode_(gen_w, gen_h);
    co_await ctx.write(0, std::move(out));
    co_return;
  }

  // ---- Qwen-Image-Edit: image-aware conditioning (from the conditioner) +
  // dual-stream DiT (reference latents from iport5/iport6 as DiT conditioning
  // tokens) + norm-preserving true-CFG -> whitened latent [16, H/8, W/8]. ----
  if (_family == "qwen-image-edit") {
    std::vector<RefLatent> qrefs;
    if (!_ref[0].empty()) { qrefs.push_back(_ref[0]); }
    if (!_ref[1].empty()) { qrefs.push_back(_ref[1]); }
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
    tag_model_(*out);
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
    const RefLatent& r0 = _ref[0];
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
    rr.seed      = _seed + (std::uint64_t)_latents_emitted;
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
    tag_model_(*out);
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
    if (!_ref[0].empty()) { qrefs.push_back(_ref[0]); }
    if (!_ref[1].empty()) { qrefs.push_back(_ref[1]); }
    _qi21_dit->set_stream_stop(stopping);
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
    tag_model_(*out);
    ++_latents_emitted;
    session()->info(fmt(
        "GenerateImageStage('{}'): Qwen-Image-2.1 latent [{}, {}, {}] ({} "
        "steps @ {}x{})", this->id(), QZ, qlh, qlw, _scheduler_spec.steps,
        gen_h, gen_w));
    // BEFORE the free, not after: this says what the DiT held while it
    // denoised, which is the number a peer has to size against. The
    // free that may follow is a transient (see declare_memory), so it
    // deliberately does not move the holding back down.
    correct_dit_holding_("after a generation");
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
    tag_model_(*out);
    ++_latents_emitted;
    session()->info(fmt(
        "GenerateImageStage('{}'): Z-Image latent [{}, {}, {}] ({} steps @ "
        "{}x{})", this->id(), ZZ, lh, lw, _scheduler_spec.steps, gen_h,
        gen_w));
    // BEFORE the free, not after: this says what the DiT held while it
    // denoised, which is the number a peer has to size against.
    correct_dit_holding_("after a generation");
    free_z_image_dit_for_decode_(gen_w, gen_h);
    co_await ctx.write(0, std::move(out));
    co_return;
  }

  if (_family == "boogu-image") {
    std::vector<RefLatent> brefs;
    if (!_ref[0].empty()) { brefs.push_back(_ref[0]); }
    if (!_ref[1].empty()) { brefs.push_back(_ref[1]); }
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
    tag_model_(*out);
    ++_latents_emitted;
    session()->info(fmt(
        "GenerateImageStage('{}'): Boogu-Image latent [16, {}, {}] ({} steps "
        "@ {}x{})", this->id(), lh, lw, _scheduler_spec.steps, gen_h, gen_w));
    // Before publishing: on a bounded box the resident DiT would leave the
    // downstream vae-decode no working set. Freed here, reloaded on the next
    // prompt (the latent is already read back, so the DiT is idle).
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
    const int first = (latent_ptr != nullptr) ? 1 : 0;
    for (int i = first; i < 2; ++i) {
      if (!_ref[i].empty()) { krefs.push_back(_ref[i]); }
    }
  }
  _dit->set_stream_stop(stopping);
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
  tag_model_(*out);
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

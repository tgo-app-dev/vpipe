# Writing vpipe plugins

A **plugin** is a shared library (`.dylib`/`.so`) built *outside* the vpipe
source tree that the runtime `dlopen`s at startup. A plugin can contribute:

- new **stages** (pipeline nodes, incl. CoreML-backed ones),
- new **Metal shader** programs (GPU kernels),
- new **LM model** families (reusing the existing `ContextManager`),
- new **video model** families, for `generate-video`,
- **model catalogue** entries, so its models are downloadable and browsable.

The primary intent is to package separately-licensed (e.g. commercial)
functionality as standalone add-ons that drop into an unmodified vpipe.

A runnable, minimal example lives in `examples/plugin-sdk/`.

## The model in one paragraph

A plugin is a **C++** library that **links `libvpipe`** (so it sees the same
registry singletons and RTTI) and exports a tiny, stable **C** handshake.
It is not a pure-C ABI: a `Stage` is a coroutine and a `ModelExec` is a C++
vtable, so the extension *types* are necessarily C++. The consequence is
that a plugin must be **built with a matching toolchain against the same
`libvpipe`** it will load into. When `libvpipe`'s ABI changes (its
`SOVERSION`), rebuild the plugin. Backward compatibility is not promised;
the ABI-version handshake below makes a mismatch fail loudly instead of
crashing.

## Building a plugin

Install vpipe, then use `find_package(vpipe)` + the shipped
`vpipe_add_plugin` helper:

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_plugin LANGUAGES CXX)
find_package(vpipe REQUIRED)
vpipe_add_plugin(my-plugin SOURCES my-stage.cc)          # -> my-plugin.so
```

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/vpipe-install
cmake --build build
```

`vpipe_add_plugin(name SOURCES ... [METAL] [COREML])` builds a dlopen-only
`MODULE` linked to `vpipe::vpipe` with C++20. **Base plugins need no
framework flags** — every Metal/CoreML call lives inside `libvpipe` and the
SDK headers only *forward-declare* the framework types. A CoreML plugin runs
its model entirely through `CoreMLLoadedModel::predict()` (neutral structs,
no `coreml-cpp`), so it does **not** need `COREML` either. Pass `METAL` /
`COREML` only if you reach for a Metal escape hatch (`mtl_buffer()`,
`SharedBuffer::wrap`) or otherwise link the frameworks yourself.

## The three required symbols

Every plugin exports exactly three `extern "C"` symbols. The
`VPIPE_PLUGIN_DEFINE` macro emits them:

```cpp
#include "plugin/plugin-abi.h"
#include "plugin/plugin-context.h"

static const VpipePluginInfo kInfo = {
    VPIPE_PLUGIN_INFO_SCHEMA,
    "acme-codecs",     // stable id
    "1.2.0",           // your version
    "Acme Inc.",       // vendor
    "Commercial",      // license  <-- logged at load
    "Acme audio codecs for vpipe",
};

static void acme_register(vpipe::VpipePluginContext* ctx) {
    // register your extensions here (see below)
}

VPIPE_PLUGIN_DEFINE(&kInfo, acme_register)
```

The host resolves `vpipe_plugin_abi_version` / `vpipe_plugin_info` /
`vpipe_plugin_register`, checks the ABI version, logs the metadata
(including the **license**), and calls your `register` function once. A
throwing `register` is demoted to a warning — it won't take down the host.

## Extension point 1 — stages

Write a `TypedStage<T>` exactly as an in-tree stage, then register it:

```cpp
class MyStage : public vpipe::TypedStage<MyStage> {
 public:
  static constexpr const char* kTypeName = "my-stage";
  using TypedStage::TypedStage;
  vpipe::Job process(vpipe::RuntimeContext& ctx) override { /* ... */ }
};
const vpipe::StageSpec kSpec = { .type_name = "my-stage", /* ports, doc */ };

void acme_register(vpipe::VpipePluginContext* ctx) {
  ctx->register_stage<MyStage>(&kSpec);   // attaches the spec + logs
}
```

The stage then appears in `/api/stage-types` and the web-ui composer, and
is usable in any pipeline spec by its `kTypeName`. (Do **not** also use the
in-tree `VPIPE_REGISTER_STAGE` macro — `register_stage` is the plugin path.)

### Model-specific configuration

A plugin that adds a **model family** usually also has knobs that belong to
that family and to nothing else. Do not add them to the generic stage that
runs the model: `generate-image` / `generate-video` / `diffusion-conditioner`
each serve several families, and a key that applies to only one of them is
inert — *silently* inert — on all the others.

Ship a **config source** instead. Derive from `ModelConfigSourceStage<T>`
(`stages/model-config-source.h`), supply a `kSpec` in
`StageCategory::ModelSpecificConfig` with an oport tagged `model-config`, and
implement one method:

```cpp
class AcmeModelConfigStage
  : public vpipe::ModelConfigSourceStage<AcmeModelConfigStage> {
 public:
  static constexpr const char* kTypeName = "acme-model-config";
  using ModelConfigSourceStage::ModelConfigSourceStage;

  vpipe::FlexData resolved_config() const {
    vpipe::FlexData fd = vpipe::model_config::make_config("acme");
    fd.as_object().insert_or_assign("shift", vpipe::FlexData::make_real(_s));
    return fd;
  }
};
```

The base supplies the trigger contract every source shares: unwired, one beat
for the run; wired, one beat per inbound beat. The `model_family` string is
what the consuming stage matches against the resident checkpoint, so a config
wired to the wrong model is reported rather than half-applied.

Two rules make the difference between a knob that works and one that looks
like it does:

- **Emit a key only when it was set.** The consuming stage starts from the
  model layer's own defaults for the family; a source that helpfully emits
  its defaults overwrites them with something that merely looks configured.
- **Parse in the model layer, not the stage.** Give the family a
  `GenerationParams`-style struct with a `from_flex()`, and let the stage
  pass the object down unread. Then a knob added later needs no change in
  any stage — which is what lets a plugin extend a family the host does not
  know about.

## Extension point 2 — Metal shaders

Compile a self-contained `.metal` offline and embed its bytes with the
`vpipe_add_metal_library` helper:

```cmake
vpipe_add_metal_library(my_kernels_obj my_kernels SRC my_kernels.metal)
vpipe_add_plugin(my-plugin SOURCES my-stage.cc $<TARGET_OBJECTS:my_kernels_obj>)
```

It exposes two symbols (`my_kernels_metallib`, `my_kernels_metallib_len`).
Register them at load, then use the library by name from your stage/model:

```cpp
extern "C" const unsigned char my_kernels_metallib[];
extern "C" const unsigned long my_kernels_metallib_len;

void acme_register(vpipe::VpipePluginContext* ctx) {
  ctx->register_metal_library("my_kernels",
                              my_kernels_metallib, my_kernels_metallib_len);
}
// later, inside a stage/model:
//   auto lib = session()->metal_compute()->load_library("my_kernels");
//   auto fn  = lib.function("my_entry_point");   // dispatch as usual
```

Registration is first-wins: a plugin cannot shadow a built-in kernel name.

### Reuse libvpipe's kernels before writing your own

`load_library()` resolves **libvpipe's own build-embedded metallibs** by
name, not just what your plugin registered. So a plugin model gets the whole
in-tree kernel set for free:

```cpp
auto mc   = session()->services()->metal_compute();
auto gemm = mc->load_library("dense_gemm_bf16");   // built into libvpipe
auto fn   = gemm.function("dense_gemm_t_bm64_f16");
```

Available by name include `dense_gemm_bf16` / `dense_gemm_mma_bf16`,
`affine_qmm_steel_bf16`, `rms_norm_bf16`, `rope_bf16`, `sdpa_bf16`,
`attn_steel`, and `llm_elementwise_bf16` — which alone carries adaLN
modulation, gated residuals, gelu-tanh, bias-add, im2col and nearest-2x
upsample. Note the entry points keep an `_f16` suffix in **both** dtype
twins; the LIBRARY name is what selects bf16 or f16.

Write a kernel only for what is genuinely your model's own. The LTX-2.5
plugin ships three entry points and reuses everything else.

### Quantized weights from a plugin

`affine_qmm_steel_bf16` is reachable the same way, so an out-of-tree model
can read a `model-quantize` w4/w8 checkpoint. There is no exported host-side
wrapper — every in-tree family writes its own ~25-line dispatch — so plan on
copying one. Four things are **silent wrong answers** rather than load
failures, and each has bitten:

* **`scales` / `biases` are `F16` in the checkpoint and `bfloat` to the
  kernel.** Every writer emits F16; the `_bf16` twin reads buffers 1 and 2 as
  `VPIPE_ELT` = bfloat. Convert at load. The element *count* is identical, so
  nothing in the shapes catches it.
* **The qmm kernel has no bias slot.** Its buffer 2 is the quantization
  zero-point, not the linear's bias. Add the bias in a second pass.
* **`_bm64` exists only at group 64.** There is no
  `affine_qmm_steel_w4g32_bm64`. Treat the wide tile as optional.
* **Bits and group can be recovered from the shapes** —
  `group = K / scales_cols`, `bits = codes_cols * 32 / K` — which avoids
  depending on a `config.json` that a Comfy-packed checkpoint does not have.

Detection needs `WeightSet::src()->info(name + ".scales")`; both
`generative-models/weight-set.h` and `generative-models/llama3/
metal-llama-weights.h` are in the installed SDK.

To PRODUCE such a checkpoint from a Comfy single file, `model-quantize`
synthesizes the output `config.json` via `comfy_output_config`, which passes
any architecture naming itself in `_class_name` through verbatim. Give it
`target` (a tensor-name prefix) and `quant_exclude` — the wholesale rule takes
every 2D floating-point tensor whose leaf is not a norm or an embedding, which
also catches modulation tables and tiny gate projections.

## Extension point 3 — LM models

Provide a `ModelExec` (a thin adapter over a model that owns a
`ContextManager`, `owns_kv()==true`) and register a factory keyed by the
checkpoint's `config.json` `architecture` string:

> **Before writing a loader, read [MODEL-MEMORY.md](MODEL-MEMORY.md).**
> A plugin's model shares one machine with every other model in the
> graph, and the contract for that — how weights are read, how a stage
> declares them before anything loads, when blocks stream, and what
> happens to weights while they are idle — is not optional. A model that
> skips it works on the box it was written on and thrashes elsewhere. An
> exec that owns its K/V must also override `kv_bytes()`.

```cpp
#include "generative-models/model-exec-registry.h"

void acme_register(vpipe::VpipePluginContext*) {
  vpipe::genai::ModelExecRegistry::get().register_arch(
      "AcmeForCausalLM",
      [](const vpipe::genai::ModelExecCreateArgs& a) {
        return std::make_unique<AcmeModelExec>(
            a.model_dir, a.config, a.metal, a.session,
            a.page_tokens, a.max_pages, a.use_bf16);
      });
}
```

`LoadedLanguageModel` consults the registry *before* its built-in arch
dispatch, so a checkpoint whose `architecture` matches your key loads your
exec. Reuse `ContextManager` by filling its `Spec` (Paged or Contiguous) —
you do **not** create a new KV manager. v1 targets text-only models;
multimodal encoders remain a built-in concern.

## Extension point 4 — video model families

`generate-video` runs Wan and MiniMax-H3 from concrete types held side by
side, and its header explains why they are not behind one base: Wan takes a
4-D latent with one timestep, H3 a packed sequence with per-row timesteps,
and an interface wide enough for both describes neither. That argument is
about the **per-step forward**, and it still holds — so the plugin seam is
not drawn there.

It is drawn one level up, at the **generation**. Every video family answers
the same question: given conditioning, geometry, a step count and a seed,
produce a latent video and (if it has one) a latent soundtrack. A family
owns its whole denoise loop — scheduler, guidance, residency,
patchify/unpatchify. The stage owns the ports, the beats and the geometry.

```cpp
#include "generative-models/video-model-registry.h"

class AcmeVideo : public vpipe::genai::VideoModelFamily {
 public:
  std::string_view tag() const noexcept override { return "acme-video"; }

  // CHEAP -- it runs for every family on every model resolve -- and SURE.
  // Claiming a checkpoint that is not yours loads at full cost and
  // computes nonsense, which is the worst failure this stage has.
  bool claims(const std::string& root, const std::string& model_type)
      const override { /* read the config, refuse anything else */ }

  // The family's frame rule (Wan's 4k+1, H3's 17n+5). Asked of the FAMILY,
  // not the generator: the stage settles geometry when it resolves the
  // checkpoint, which is before anything is loaded.
  int align_frames(const std::string&, int frames) const override;

  // Collected BEFORE any stage initialises. A checkpoint that declares
  // nothing is invisible to every peer sizing itself -- see
  // MODEL-MEMORY.md.
  std::vector<vpipe::ResourceClaim>
  declare_resources(const std::string& root) const override;

  std::unique_ptr<vpipe::genai::VideoGenerator>
  load(const vpipe::genai::VideoModelCreateArgs&) override;
};

void acme_register(vpipe::VpipePluginContext* ctx) {
  ctx->register_video_family(std::make_unique<AcmeVideo>());
}
```

`generate-video` consults the registry **before** its built-in `wan` /
`minimax-h3` probes — mirroring how `LoadedLanguageModel` consults
`ModelExecRegistry` before its own arch dispatch. The built-in families are
unchanged and unregistered: this adds a path, it does not reroute theirs.

The `VideoGenRequest` your `generate` receives carries the conditioning
(and its sideband), any conditioning latents, the geometry, and the
**unparsed** model-config beat — so a knob your family adds later needs no
change in `generate-video`. Everything in it is borrowed from beats that
outlive the call. Call `req.progress(step, total)` between steps: a family
that never does cannot be interrupted, and on a 22B model that makes a Stop
look like a hang.

Ship the family's knobs as a `ModelConfigSourceStage` (above), not as keys
on `generate-video`.

## Extension point 4b — VAE families

A family that generates latents needs something to turn them into frames,
and `vae-decode` picks its built-in decoder from a hardcoded `_class_name`
chain (`wan`, `minimax-h3`, `flux2`, `mage`, `krea2`) that an out-of-tree
family cannot join. `register_vae_family` is the counterpart to
`register_video_family`: register **both** and your model needs a stage of
its own for neither — its graphs use the stock `vae-decode`.

```cpp
#include "generative-models/vae-model-registry.h"

class AcmeVae : public vpipe::genai::VaeModelFamily {
  std::string_view tag() const noexcept override { return "acme"; }
  bool claims(const std::string& root, const std::string& vae_dir,
              const std::string& model_type) const override;
  std::unique_ptr<vpipe::genai::VaeDecoder>
  load_decoder(const vpipe::genai::VaeModelCreateArgs&) override;
};

ctx->register_vae_family(std::make_unique<AcmeVae>());
```

The seam is drawn at **one decode**: given a latent and its sideband,
produce RGB frames. Un-whitening, tiling, colour space and residency are
yours; the stage keeps the ports, the U8 quantisation, the per-frame beat
and the idle-unload policy.

`decode` hands frames to a **sink**, one or more times, in frame order. A
decoder that can only produce a whole clip calls it once (`frame0 = 0`,
`n = frames_total`); chunking is permitted, never required, and is how a
long clip avoids existing in full. Chunks are f32 **channel-first over the
chunk**, `[channels][n][height][width]`, in **[-1, 1]** — that range is the
contract, since the host quantises `(x+1)/2*255` straight from it. Returning
`false` from the sink aborts, which is how a Stop mid-decode gets out.

Three things worth knowing before you write `claims`:

- It receives **both** `root` and `resolve_vae_dir(root)`. That resolver
  knows `vae/config.json` and `video_vae/` and returns the root unchanged
  for anything else — including every Comfy-style pack, which is the case
  this extension point exists for.
- It is asked **before** the built-in chain, so it must be cheap (one header
  read) and **sure**. Claiming a checkpoint that is not yours loads at full
  cost and decodes nonsense.
- A `tag()` colliding with a built-in name is refused outright. Dispatch is
  pointer-guarded so it would still run your code, but every log line would
  read as a built-in.

Override `idle_peers()` if your checkpoint is not laid out the diffusers
way. The stage sizes its unload decision against `root/transformer`,
`root/text_encoder` and `root/mllm`; on a pack that spells those differently
the footprint sums to zero, which reads as "the box is roomy, keep the VAE
resident" — beside a large DiT, the wrong call.

### The other two halves: `load_encoder` and `load_audio_decoder`

Both are **additive virtuals with a default of "decline"**, on the same
family object. A family that only decodes video overrides neither and
nothing changes.

`load_encoder` serves **`vae-encode`**, which has the same hardcoded
`_class_name` chain and the same reason to open it: your latent geometry is
your own, so without it a graph can decode your latents but never *produce*
one — no image reference, no img2img, no first-frame anchor.

```cpp
std::unique_ptr<vpipe::genai::VaeEncoder>
load_encoder(const vpipe::genai::VaeModelCreateArgs&) override;
```

The seam is **one image in, one latent out**. The stage hands you f32
`[3][frames][H][W]` already normalised to [-1, 1] and already
letterbox-fitted to the target size; you return a channel-first latent and
**its shape**, and the stage publishes that shape rather than one it
predicted. `frames` is 1 today for every family — the stage reads one RGB
beat and does not gather clips — so a video-shaped encoder should expect it
and say so if it wanted more.

Two behaviours worth knowing:

- A family that **claims** the checkpoint but returns null leaves the stage
  **inert**, with an error naming why. It does *not* fall through to a
  built-in: your `claims` already said this VAE is yours, and a built-in
  reading it would encode at the wrong latent geometry and emit a
  plausible, wrong beat.
- `release_idle()` is called after each beat when `unload_when_idle`
  resolves to unloading. Dropping everything and rebuilding on the next
  `encode` is a perfectly good implementation.

`load_audio_decoder` serves **`audio-vae-decode`** the same way, for a
family that generates a soundtrack, and `load_audio_encoder` serves
**`audio-vae-encode`** — a reference soundtrack for a family that
conditions on one.

The audio encoder's seam is **one call for the whole clip**, not a
stream. The stage accumulates every PCM beat and encodes once at EOS,
because an audio VAE is causal and compresses time: chunks encoded
separately and concatenated are a different tensor, wrong at every seam
and wrong in length. You are handed planar `[channels][n_samples]` in
[-1, 1] with the rate the stream claimed, and you return a latent and
its shape — `[rows, dim]` if it is destined for `generate-video`'s
`ref_audio_rows`.

Report the rate you want from `sample_rate()`; the stage compares it
against the stream and **refuses** a mismatch rather than resampling,
naming `audio-to-pcm`'s `output_sample_rate` as the fix. It owns no
resampler, and a silent rate mismatch is a reference of the right length
at the wrong pitch. `channels()` says whether you want mono or stereo; a
family that wants stereo and is handed mono should duplicate rather than
refuse.

`audio-vae-encode` has **no built-in family at all** — every audio
encoder it can reach comes from a plugin — so a checkpoint nothing claims
leaves it inert with a message, not falling back to anything.

## Extension point 5 — model catalogue entries

A family nobody can download is a family nobody can run. `model_catalog()`
is the single source of what a front end may offer — the drill-down menu,
`model-fetch`, `model-select`, the web-ui browser — so a plugin contributes
to it rather than keeping a parallel list:

```cpp
#include "stages/model-catalog.h"

void acme_register(vpipe::VpipePluginContext* ctx) {
  vpipe::ModelCatalogEntry e;
  e.family = "Acme"; e.version = "1"; e.param_class = "7B";
  e.variant = "bf16"; e.hf_path = "acme/video-7b";
  e.model_type = "acme-video";            // == your family's tag
  e.outputs = {"video"};
  e.files = { /* pin the subset that is actually needed */ };
  ctx->register_catalog_entries({e});
}
```

Entries land after the built-ins, in registration order, and behave exactly
like built-in ones. Duplicates (same registration key + same `files`) are
dropped and reported. `vpipe --list-models` loads any `--plugin` given on
the same command line before printing, so a listing never omits a model the
same invocation just registered.

**Pin `files`.** Most modern video repos publish several precisions in one
repo; fetching the whole thing pulls packings this build cannot read.

## Extension point 6 — CoreML

There is no separate CoreML plugin type: a CoreML consumer is just a
**stage** (extension point 1) that loads a model through the session and
runs it via the model manager's native `predict()` API. The plugin never
includes `coreml-cpp` and never touches a `CML::`/`NS::`/CoreVideo type —
all of that marshaling lives inside `libvpipe`:

```cpp
auto model = session()->coreml_model_manager()->load(path, /*units*/ 2);
if (!model) { /* fail_config / drop */ }

// Describe inputs/outputs with the neutral CoreMLPredict* structs. A
// tensor input is a borrowed buffer + dtype + shape; an image input is
// BGRA bytes; an output names the feature and the dtype you want back.
CoreMLPredictInput  in { .name = "waveform", .data = pcm,
                         .dtype = CoreMLDType::F32, .shape = {1, n} };
CoreMLPredictOutput out { .name = "probs", .want = CoreMLDType::F32 };
const CoreMLPredictInput ins[1]  = { std::move(in) };
CoreMLPredictOutput      outs[1] = { std::move(out) };
if (model->predict(ins, outs)) {
  const float* p = static_cast<const float*>(outs[0].data);  // + outs[0].shape
  // ... emit TensorBeat / FlexData Beats ...
}
```

`predict()` handles the whole feature dance: zero-copy tensor binding
(incl. Metal/UMA pointers), image inputs (it builds the CVPixelBuffer),
per-model serialization, `f16`/`f64` → your requested dtype decode
(non-contiguous strides included), and optional zero-copy output backings
(`CoreMLPredictOutput::backing`) for fixed-shape outputs. Model shape /
dtype / image-format introspection is available via `input_descs()` /
`output_descs()`. No `COREML` framework flag is required. Use the in-tree
stages as templates: `stages/coreml-inference-stage.cc` (generic tensor
passthrough), `stages/audio-tagging-stage.cc`,
`stages/vision/yolo-detection-stage.cc` (image input).

## Loading a plugin

Three equivalent ways, all resolved once per process (dedup by path):

- CLI: `vpipe --plugin ./my-plugin.so ...` /
  `vpipe-web-ui --plugin ./my-plugin.so` (repeatable),
- env: `VPIPE_PLUGINS=/a.so:/b.so vpipe ...` (colon-separated),
- config: `{"plugins": ["/a.so", "/b.so"], ...}` in the session config.

Plugins load before any pipeline is built, so their stages/models/shaders
are available immediately.

## Where plugins live, and how they get loaded

The convention is **`<work-dir>/plugins/`** — `plugins/` beside the directory
you start vpipe from, or `$VPIPE_PLUGINS_DIR` when set
(`common/plugins-root.h`).

Two deliberate non-behaviours:

- **Nothing there is loaded automatically.** Discovery is a listing, not an
  instruction. Loading a dylib is arbitrary code execution in this process, so
  dropping a file into a scanned folder must never be sufficient on its own.
- **The directory is not created for you.** Nothing writes there, and
  materialising a `plugins/` folder in whatever directory someone happened to
  run vpipe from is litter. The panel shows the path and says it does not
  exist yet.

Three ways to actually load one, in increasing order of who decides:

| How | Who decides | Scope |
|---|---|---|
| `--plugin PATH` on `vpipe` / `vpipe-web-ui` | whoever runs the process | any path |
| the `plugins:` config array / `VPIPE_PLUGINS` | the deployment | any path |
| the web-ui **Plugins** panel | a UI session | **only under the plugins root** |

The panel is confined to the root on purpose: it is reachable by anyone who
can reach the UI. A deployment that needs a plugin from elsewhere passes
`--plugin` at startup, which is a decision taken by the operator rather than
by an HTTP request.

### The Plugins panel, and why there is no Unload

The panel lists what is present under the root, what this process has loaded
(with version, vendor, license and the number of stage types it actually
contributed), and offers **Load** and **Disable**.

**There is no unload, and `Disable` is not a quiet one.** vpipe never
`dlclose`s a plugin — `PluginManager`'s class comment says so, and the reasons
are load-bearing: `StageRegistry` holds raw factory pointers into the dylib
and has no removal API, the `StageSpec*` handed to `/api/stage-types` points
at its static storage, live stage instances hold vtables there, and the
metal-library / video-family / VAE-family / catalogue registrations all
reference it. Unmapping any of that under a running process is a
use-after-unmap, not a reclaim.

So `Disable` means *stop offering this plugin's stages*: they leave the
composer toolbox and cannot be used for new instances. The plugin stays
loaded and mapped, the panel says "disabled, still resident", and removing it
for real needs a restart. Making unload real would mean deregistration across
those registries plus a liveness check — an architectural change, not a
method.

## Versioning

- `VPIPE_PLUGIN_ABI_VERSION` (in `plugin/plugin-abi.h`) is the plugin
  contract version, currently **1**. The host loads a plugin only when the
  plugin's reported value **equals** the host's — strict equality, no
  backward compatibility.

  There is deliberately **no ledger of superseded versions**. Strict
  equality gives the number no ordering meaning (N does not mean "N−1 and
  more"; it is an opaque cookie), so a history of retired versions tells a
  plugin author nothing they can act on — and it rots: the header's list
  had drifted to describing 1 and 2 while the value read 4. vpipe is alpha
  and no plugin has ever shipped against an earlier number, so the count
  was reset rather than carried.

- **What counts as the contract is wider than the C symbols**, and this is
  the part that catches people. Bump the version for any change to the
  three `extern "C"` entry points, to the `VpipePluginContext` facade —
  *and to any interface a plugin subclasses* (`VideoModelFamily`,
  `VaeModelFamily`, `ModelExec`, `Stage`, …). Adding a virtual to one of
  those changes no C symbol and no facade method, yet it moves the vtable:
  a plugin built against the older header passes every check and then calls
  through the wrong slot. That has already happened once, which is why it
  is written down here rather than left to judgement.
- The `libvpipe` `SOVERSION` guards the underlying C++/ABI. It moves
  independently; a plugin records a dependency on a compatible `libvpipe`.
- `VpipePluginInfo::schema_version` lets the info struct grow additively.

The practical rule: **rebuild your plugin against the vpipe you deploy
with.** A mismatch is reported and refused, not crashed.

## Where your stages show up

Stages a plugin contributes are **grouped under the plugin's name** in the
web-ui composer's toolbox, rather than mixed into the built-in category
sections. `/api/stage-types` carries a `plugin` field for this — the plugin's
`VpipePluginInfo::name`, or `""` for a built-in.

The grouping is deliberate: a plugin stage's *category* says what it does,
which the built-in sections already convey. What a reader cannot otherwise
tell is that the stage exists only because a `.dylib` was loaded and will be
absent from a deployment that does not load it. That is the property worth a
heading, so plugin stages appear once, under their plugin.

Provenance is **observed by the host, not declared by you**: `PluginManager`
records the stage-type id the registry was about to hand out before your
dylib is opened, and attributes everything registered after it. That matters
because a `TypedStage<T>` registers its own factory from a static initialiser
as the dylib maps — before `vpipe_plugin_register` runs — so attributing
inside `register_stage` would miss any stage you did not also hand to the
context. You get correct grouping whether or not you call `register_stage`.

## Licensing

`VpipePluginInfo::license` (and `vendor`) are logged when the plugin loads,
so a deployment's provenance is visible. A plugin is a separate binary with
its own license; shipping commercial functionality as a plugin keeps it out
of the vpipe tree and under its own terms.

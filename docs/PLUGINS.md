# Writing vpipe plugins

A **plugin** is a shared library (`.dylib`/`.so`) built *outside* the vpipe
source tree that the runtime `dlopen`s at startup. A plugin can contribute:

- new **stages** (pipeline nodes, incl. CoreML-backed ones),
- new **Metal shader** programs (GPU kernels),
- new **LM model** families (reusing the existing `ContextManager`).

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

## Extension point 3 — LM models

Provide a `ModelExec` (a thin adapter over a model that owns a
`ContextManager`, `owns_kv()==true`) and register a factory keyed by the
checkpoint's `config.json` `architecture` string:

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

## Extension point 4 — CoreML

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

## Versioning

- `VPIPE_PLUGIN_ABI_VERSION` (in `plugin/plugin-abi.h`) is the plugin
  contract version. The host loads a plugin only when the plugin's reported
  value **equals** the host's (strict equality — no backward compatibility
  yet). Bump it on any change to the three C symbols or the
  `VpipePluginContext` surface.
- The `libvpipe` `SOVERSION` guards the underlying C++/ABI. It moves
  independently; a plugin records a dependency on a compatible `libvpipe`.
- `VpipePluginInfo::schema_version` lets the info struct grow additively.

The practical rule: **rebuild your plugin against the vpipe you deploy
with.** A mismatch is reported and refused, not crashed.

## Licensing

`VpipePluginInfo::license` (and `vendor`) are logged when the plugin loads,
so a deployment's provenance is visible. A plugin is a separate binary with
its own license; shipping commercial functionality as a plugin keeps it out
of the vpipe tree and under its own terms.

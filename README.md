<a id="top"></a>
# VPIPE

A concurrent pipeline framework for real-time multimedia tasks, built
natively on Apple Silicon.

[**Overview**](#overview) · [**Requirements**](#requirements) ·
[**Build**](#build) · [**Run**](#run) · [**Examples**](EXAMPLES.md) ·
[**Tests**](#tests) · [**Layout**](#layout) · [**License**](#license)

---

## Overview
[back to top](#top)

VPIPE has two layers:

- **Pipeline core** — coroutine-based `Job` stages connected by buffered ports,
  driven by a runtime that launches and drains them concurrently. Stages are
  composed into a pipeline from a JSON spec; each stage registers under a
  type name (e.g. `rtsp-capture`, `video-to-rgb`, `yolo-detection`,
  `onvif-discovery`, `rest-client`). This layer is portable C++20.

- **On-device generative-model stack** *(Apple Silicon)* — a from-scratch
  LLM/VLM/ASR inference engine (text, vision, audio, video) running on a custom
  **metal-compute** backend, with no third-party tensor runtime in the forward
  pass. It powers stages such as `text-chat`, `visual-qa`, `realtime-vqa`, and
  `audio-transcribe`.

Models are loaded from local directories (sharded safetensors / GGUF) and are
not bundled with the source.

> **Platform note.** The pipeline core and the non-Apple stages build on Linux
> and Intel macOS, but the generative-model stack and the CoreML/Metal stages
> require an **Apple Silicon Mac**. On arm64 macOS these features are detected
> and enabled automatically.

---

## Requirements
[back to top](#top)

- **CMake ≥ 3.25** and a **C++20** compiler (Apple Clang or a recent Clang/GCC).
- **Git** (the build pulls a few dependencies as submodules).
- **FFmpeg development headers** — `libavformat`, `libavcodec`, `libavutil`,
  `libswresample`. VPIPE compiles against the headers and `dlopen`s the
  libraries at runtime, so FFmpeg must also be installed at runtime to decode
  media.
  - macOS: `brew install ffmpeg`
  - Debian/Ubuntu: `apt install libavformat-dev libavcodec-dev libavutil-dev libswresample-dev`
- **libcurl** — used by the `rest-client` stage. Provided by the SDK on macOS;
  on Linux install e.g. `apt install libcurl4-openssl-dev`.
- **Python 3 + development headers** — only for the optional Python extension
  (built by default). Disable with `-DVPIPE_BUILD_PYTHON=OFF` if you don't need
  it.
- **Apple Silicon Mac** — for the on-device inference stack and CoreML/Metal
  stages.
- **Full Xcode (Metal shader toolchain)** *(Apple Silicon builds only)* — the
  build compiles `.metal` kernel sources into embedded metallibs using
  `xcrun -sdk macosx metal` and `xcrun -sdk macosx metallib`. These two
  compilers ship **only with the full Xcode app** (from the App Store or
  developer.apple.com); the standalone *Command Line Tools* do **not**
  include them, even though the rest of the build never opens Xcode.

  **Common pitfall.** If you previously installed the Command Line Tools and
  *then* installed Xcode, `xcrun` usually still points at the standalone CLT,
  which lacks `metal`/`metallib` — so the build fails with an error like
  `error: cannot execute tool 'metal'` or `xcrun: error: unable to find
  utility "metal"`. Point the toolchain at Xcode with `xcode-select`:

  ```sh
  # Direct xcrun at the Xcode app (run once; needs admin)
  sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer

  # Accept the Xcode license if you haven't already
  sudo xcodebuild -license accept
  ```

  Verify the active developer dir and that both compilers resolve:

  ```sh
  xcode-select -p                 # -> /Applications/Xcode.app/Contents/Developer
  xcrun -sdk macosx -f metal      # -> a path inside Xcode.app (not /Library/Developer/CommandLineTools)
  xcrun -sdk macosx -f metallib   # -> likewise
  xcrun -sdk macosx metal --version
  ```

  To switch back to the Command Line Tools later:
  `sudo xcode-select --switch /Library/Developer/CommandLineTools`.

---

## Build
[back to top](#top)

**1. Fetch dependencies.** LMDB and pugixml are always required; nanobind is
needed for the Python bindings, and metal-cpp for the Apple Silicon features:

```sh
git submodule update --init extern/lmdb extern/pugixml extern/nanobind extern/metal-cpp
```

> `extern/mlx` is **optional and large** — it is not used by the default build
> (the metal-compute backend replaced it) and is kept only for occasional
> perf/accuracy cross-checks. Skip it unless you specifically need that path.

**2. Configure** (out-of-source build directory):

```sh
cmake -S . -B build
```

This defaults to an optimized **Release** build, so the binaries you get are
performant out of the box. Override the build type explicitly if you want a
debug build:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
```

**3. Build:**

```sh
cmake --build build -j
```

`cmake --build` is generator-agnostic; use it rather than calling `make`
directly so the build works regardless of which generator CMake selected.

Useful options (pass at configure time with `-D`):

| Option | Default | Effect |
| --- | --- | --- |
| `VPIPE_BUILD_PYTHON` | `ON` | Build the `vpipe` Python extension (needs Python dev headers). |
| `VPIPE_BUILD_APPLE_SILICON` | auto (on for arm64 macOS) | Build the CoreML/metal-compute wrappers and the inference stages. |
| `CMAKE_BUILD_TYPE` | `Release` (when unset) | Set `Debug` for an unoptimized debug build. |

**Install** (optional):

```sh
cmake --install build --prefix /path/to/install
```

---

## Run
[back to top](#top)

### Web UI

`vpipe-web-ui` serves a browser-based Pipeline Manager bound to one VPIPE
session. The web assets are embedded in the binary, so no extra files are
needed:

```sh
./build/apps/web-ui/vpipe-web-ui                    # listens on the LAN address, port 9876
./build/apps/web-ui/vpipe-web-ui --bind 127.0.0.1   # this machine only
```

Then open the printed URL (e.g. `http://localhost:9876`). By default it binds
to the machine's LAN address so other devices can connect; remote connections
must supply the 8-character access key printed at startup, while localhost
connects without one. Options: `--bind ADDR`, `--port N` (`0` = any free port),
`--config CFG` (inline JSON, a file path, or empty for defaults), `--help`.

New here? **[EXAMPLES.md](EXAMPLES.md)** walks through fetching a model and
building text-chat and speech-transcription pipelines in the web UI.

### Python

The extension lands in `build/python/`. Importing the package creates a default
session:

```sh
PYTHONPATH=build/python python3 -c "import vpipe; print(vpipe.vpipe_version())"
```

Startup configuration is resolved from `VPIPE_CONFIG` / `VPIPE_CONFIG_FILE`, an
`./init.vpipe` file, or built-in defaults. Call `vpipe.create_session(config=...)`
to make your own session.

### Log reader

`vpipe-db-log-reader` dumps (or prunes) records from a VPIPE LMDB log database:

```sh
./build/apps/db-log-reader/vpipe-db-log-reader <db-path> [--from TS] [--to TS]
```

---

## Tests
[back to top](#top)

The build produces a unit-test executable:

```sh
./build/vpipe_test                        # run everything
./build/vpipe_test --list_tests
./build/vpipe_test --filter '<pattern>'   # supports * and ? wildcards
./build/vpipe_test --color off            # for captured/non-interactive output
```

Some tests exercise real models and are gated on environment variables that
point at local model directories; when a variable is unset, the corresponding
test skips.

---

## Layout
[back to top](#top)

| Path | Contents |
| --- | --- |
| `pipeline/`, `common/`, `interfaces/`, `include/` | Pipeline core: jobs, ports, runtime, session, shared services. |
| `stages/` | Pipeline stages (capture, decode, detection, REST, the LLM/VLM stages, …). |
| `generative-models/` | On-device LLM/VLM/ASR stack (model families, tokenizers, encoders). |
| `apple-silicon/` | metal-compute backend and CoreML C++ wrappers. |
| `gpu-kernels/metal/` | Metal compute kernels (attention, GEMM, quant, …). |
| `apps/` | Executables: `web-ui`, `db-log-reader`. |
| `python/` | Python bindings (nanobind). |
| `tests/` | Unit tests. |
| `extern/`, `3rd-party/` | Vendored dependencies. |

---

## License
[back to top](#top)

VPIPE is licensed under the **Apache License, Version 2.0** — see
[`LICENSE`](LICENSE). Bundled and vendored third-party components and their
licenses are documented in
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).

Brought to you by T-Go LLC.

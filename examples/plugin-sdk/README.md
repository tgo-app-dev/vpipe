# vpipe plugin SDK example

A minimal out-of-tree vpipe plugin: one stage, `example-echo`, that
forwards each input beat unchanged. It is built entirely against an
**installed** vpipe SDK via `find_package(vpipe)` — nothing here depends on
the vpipe source tree.

## Build

First install vpipe somewhere:

```sh
cmake -S /path/to/vpipe -B /path/to/build
cmake --build /path/to/build
cmake --install /path/to/build --prefix /path/to/vpipe-install
```

Then build this plugin against that prefix:

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/vpipe-install
cmake --build build
# -> build/example-echo.so
```

## Run

Load it into a vpipe session with `--plugin` (repeatable), or via the
`VPIPE_PLUGINS` env (colon-separated), or a `plugins: [...]` array in the
session config:

```sh
vpipe --plugin build/example-echo.so --launch-stage example-echo
# or:  vpipe-web-ui --plugin build/example-echo.so
```

The stage then appears in the web-ui composer and `/api/stage-types`.

## What to look at

- `example-stage.cc` — a `TypedStage` + its `StageSpec` + the three C
  entry points emitted by `VPIPE_PLUGIN_DEFINE`. The plugin registers its
  stage in `plugin_register` via `ctx->register_stage<T>(&spec)`.
- `CMakeLists.txt` — `find_package(vpipe)` + `vpipe_add_plugin(...)`.

See `docs/PLUGINS.md` in the vpipe tree for the full contract, the other
extension points (Metal shaders, LM models, CoreML), and versioning.

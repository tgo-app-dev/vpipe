# Frame interpolation with RIFE

**RIFE** makes new frames between the ones a video has: 24 fps footage
played at 60, smooth slow motion, a clip retimed to another rate. It is the
network of Huang et al., *Real-Time Intermediate Flow Estimation for Video
Frame Interpolation* (ECCV 2022), as
[Practical-RIFE](https://github.com/hzwer/Practical-RIFE) publishes it: it
estimates how every pixel moves between two frames, coarse to fine, and
builds the frame at any moment between them from both.

vpipe runs it on-device through its **metal-compute** backend, as the
`rife` method of the **`temporal-resample`** stage — the same stage that
offers frame dropping, blending, averaging and FFmpeg's `minterpolate`.
No Python is involved; the checkpoint is read as data. Its output matches
an fp32 restatement of the network to a relative error of 0.0001–0.0008,
which is 10–30× closer than the release's own `--fp16` mode.

## What you need

| | |
|---|---|
| **Machine** | Apple Silicon Mac (M-series). |
| **Memory** | **About 0.6 GB at 1080p**: 12 MB of weights (25 MB on an M5, which keeps a second layout for its matrix cores), 0.44–0.47 GB of working memory and two frames of 51 MB in flight. 4K at the default scale (0.5) needs 1.4–1.6 GB. |
| **Disk** | **~25 MB** for one model. |
| **Build** | An Apple Silicon build of vpipe — the default on arm64 macOS. See the main [README](../README.md). |

## Step 1 — get a model

Practical-RIFE distributes each model as a zip on Google Drive, linked
from the **Trained Model** section of its
[README](https://github.com/hzwer/Practical-RIFE#trained-model). The
README recommends **4.25** for most footage; **4.26**, **4.22**,
**4.25.lite** and **4.22.lite** read too. The models are under the same MIT
license as the project.

Unzip it. You get a `train_log/` directory:

```
train_log/
  flownet.pkl      the weights
  IFNet_HDv3.py    the network, in Python
  RIFE_HDv3.py     its driver
```

Point the stage's `hf_dir` at that directory (or at `flownet.pkl`
itself). vpipe reads the network's shape off the weights, and its **scale
list** — the resolution each of its stages works at — from the line
`scale_list = [...]` in `RIFE_HDv3.py`. It never runs either Python file.
Keep the driver beside the weights: 4.25.lite's list differs from the
others, and without the file vpipe falls back to the usual one.

The model browser recognises a `train_log/` directory as a RIFE model, so
the `hf_dir` picker offers it once it is under your models directory.

Releases before 4.7 (no feature encoder) are a different network and are
refused by name.

## Step 2 — run it

**[`rife-frame-interpolation.vpipeline`](pipelines/rife-frame-interpolation.vpipeline)**
reads `input.mp4`, interpolates it to 60 fps and writes
`interpolated-60fps.mp4`:

```sh
vpipe --launch docs/pipelines/rife-frame-interpolation.vpipeline \
      --stage-cfg load-video::input_url=my-clip.mp4 \
      --stage-cfg temporal-resample::hf_dir=/path/to/train_log \
      --stage-cfg save-video::output_url=my-clip-60fps.mp4
```

or open it with **Load** in the web UI's Pipeline Manager. The graph is
`load-video → video-to-rgb → temporal-resample → rgb-to-video →
save-video`; the encoder takes the new rate from the frames.

## Settings

On `temporal-resample`, with `method: rife`:

| key | what it does |
|---|---|
| `output_fps` | The rate to produce. The output keeps the clip's length: N frames at r in become N / r seconds out. |
| `hf_dir` | The model: a `train_log/` directory, its `flownet.pkl`, or a models-DB key. |
| `rife_scale` | The resolution the motion is estimated at, relative to the frame: 0.25, 0.5, 1, 2 or 4. `0` (default) follows the release's advice — 1, or 0.5 for frames wider or taller than 2560 (4K). Lower is faster and follows large motion better; higher keeps fine motion. |
| `scene_threshold` | Two frames less similar than this (SSIM at 32 × 32) are a **scene cut**: the in-between frames repeat the first rather than morph one shot into the next. `0.2` (default) is the release's; `0` interpolates every pair. |
| `stacked` | `false` (default): one frame per beat, as `video-to-rgb` sends them. `true`: a whole clip in one beat, `[T, 3, H, W]`. |

**How the frames are timed.** Output frame k shows the moment k / `output_fps`
seconds into the clip. Where that is exactly a source frame — every other
frame of a 30 → 60 conversion — the source frame is passed through
untouched; elsewhere RIFE synthesises the frame at that moment. Times after
the last source frame (the end of the clip's last frame interval) repeat
it, so the clip keeps its length.

**Going down in rate**, RIFE takes the frame at each output moment: 60 →
24 shows the scene at 0, 2.5, 5 … source frames. It does not average what
falls between; for that — motion blur, no aliasing — use `method:
average`.

## Speed

One interpolated frame, u8 frames. On an M5 the convolutions run on the
GPU's matrix cores (the hardware convolution op); on earlier chips on its
SIMD-group matrix units.

| frame | scale | M4 Pro: prepare / interpolate | M5 (base, MacBook Air): prepare / interpolate |
|---|---|---|---|
| 960 × 540 | 1 | 2.2 / 10.0 ms | 1.4 / 8.2 ms |
| 1920 × 1080 | 1 | 6.9 / 29.0 ms | 4.4 / 25.9 ms |
| 3840 × 2160 | 0.5 | 27.0 / 46.8 ms | 17.2 / 55.1 ms |

On the M5 the matrix cores take the convolutions 1.9× faster at 1080p
than the SIMD-group route would (50.1 ms there); what is left is mostly
the work around them — warping and resizing at full resolution — which
is memory-bound.

Each source frame is prepared once — it is the end of one pair and the
start of the next — so a 2× conversion costs one preparation and one
interpolation per source frame. A 124-frame 960 × 544 clip, 24 → 60 fps
(310 frames out, 246 of them synthesised): 3.0 s for the whole graph,
decode and encode included.

## How close it is

The reference is the release's network written out again in PyTorch and run
in fp32 (it is restated from `IFNet_HDv3.py`, not imported). vpipe runs
the network in 16-bit with 32-bit accumulation, and keeps the frames, the
motion and the final blend in 32-bit. Relative L2 error of the
interpolated frame against the fp32 reference, beside the release's own
`--fp16` mode against the same:

| model | frames | vpipe | release fp16 |
|---|---|---|---|
| 4.25 | 448 × 256, t 0.5 / 0.25 | 0.00019 / 0.00017 | 0.0042 / 0.0036 |
| 4.25 | 1280 × 720 | 0.00026 | 0.0034 |
| 4.25 | 896 × 512, scale 0.5 | 0.00021 | 0.0062 |
| 4.26 | 320 × 180, t 0.5 / 0.75 | 0.00056 / 0.00077 | 0.013 / 0.023 |
| 4.22 | 448 × 256 | 0.00021 | 0.0038 |
| 4.25.lite | 448 × 256 | 0.00031 | 0.0048 |

The scene score matches the release's to 1e-6.

## Differences from the release's script

- **Rounding.** An 8-bit output is rounded to nearest; the release's script
  truncates (`(x * 255).byte()`), which reads half a level darker on
  average.
- **Static frames.** The release's video script has a special case for two
  nearly identical frames (it skips ahead and interpolates over the
  duplicate). vpipe interpolates them like any pair.
- **Edges.** Like the release, a frame is padded with black to a multiple
  of 128 / scale pixels and the result cut back, so motion near the right
  and bottom edges can be pulled toward the padding. vpipe reproduces this
  exactly rather than changing what the network sees.

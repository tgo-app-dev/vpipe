# MiniMax H3 on Apple Silicon

**MiniMax H3 (FL2VA)** is a 33-billion-parameter video model that generates a
clip **and its soundtrack together**, from a single prompt. vpipe runs it
on-device through its **metal-compute** backend — its own Metal kernels, no
Python and no third-party tensor runtime in the forward pass.

The unusual part is the *together*: video and audio are not two models glued
at the end. One denoise loop runs over one packed sequence carrying both, so
the sound is generated with knowledge of the picture rather than dubbed onto
it afterwards. Ask for rain and you get rain you can hear.

It is also **guidance-distilled**, which is what makes it practical here:
there is no second unconditional forward pass per step, and useful output
arrives in **4–8 steps** instead of 30+.

## What you need

| | |
|---|---|
| **Machine** | Apple Silicon Mac (M-series). |
| **Memory** | **16 GB minimum.** More is faster — see [Memory](#memory). |
| **Disk** | **~155 GB** to prepare, **~45 GB** to keep. See below. |
| **Build** | An Apple Silicon build of vpipe — the default on arm64 macOS. See the main [README](../README.md). |

### Disk space

The published checkpoint is bf16 and very large; vpipe quantizes it to 4-bit
once, up front. Both copies exist while that runs:

| | |
|---|---|
| Source repack (`Comfy-Org/MiniMax-H3`, bf16) | **~115 GB** |
| Peak while preparing (source + output) | **~155 GB** |
| 4-bit model, source deleted | **~45 GB** |

Once preparation finishes you can delete the source repack and keep the
~45 GB. The 115 GB is a one-time cost, not a standing one.

> **Keep the download and the output on one filesystem.** Components that
> quantization does not touch (the VAEs) are **hard-linked** into the output
> rather than copied, which is why the numbers above are smaller than they
> look. Across two volumes the link fails, vpipe falls back to a real copy,
> and you pay for those bytes twice.

## The two pipelines

- **[`prepare-minimax-h3-4bit.vpipeline`](pipelines/prepare-minimax-h3-4bit.vpipeline)**
  — download the checkpoint and quantize it. Run once.
- **[`minimax-h3-text-to-video.vpipeline`](pipelines/minimax-h3-text-to-video.vpipeline)**
  — prompt in, `.mp4` with sound out.

Follow a link and use **Raw ▸ Save as** to download it, or take them straight
from `docs/pipelines/` in your clone. Either can be run from the terminal with
`vpipe --launch <file>` or opened with **Load** in the web UI's Pipeline
Manager (or the phone UI's ⋯ menu); step 1 below uses the CLI and step 2 the
web UI, because that is what each job wants. Both are plain JSON — read them,
edit them, keep them in version control.

## Step 1 — prepare the model

### First, choose a work directory

vpipe treats **the directory you launch it from** as its workspace, and
creates its state there:

| | |
|---|---|
| `models/` | every model you download or quantize |
| `data.mdb`, `lock.mdb` | the LMDB database — the model registry, logs, stage output |
| `sandbox/` | created by **`vpipe-web-ui`** only: the directory it confines stage file I/O to |

Two things follow. Pick a directory on the volume with the **~155 GB** (see
[Disk space](#disk-space)) — that is where the download lands. And use the
**same** directory in step 2: the model you are about to prepare is recorded
in that directory's registry, so a run started somewhere else will not find
it.

### Then run the pipeline

```sh
cd ~/vpipe-work                                    # your work directory
cp ~/src/vpipe/docs/pipelines/prepare-minimax-h3-4bit.vpipeline .
~/src/vpipe/build/apps/vpipe/vpipe --launch prepare-minimax-h3-4bit.vpipeline
```

The CLI suits this job better than the web UI: it is long, unattended and
disk-bound, with nothing to click once it starts. It prints a live progress
bar per stage, and `Ctrl-C` stops it cleanly — `skip_existing_files` means
re-running picks up where it left off.

Four stages, in order:

1. **`model-fetch`** — pulls `Comfy-Org/MiniMax-H3` into `./models`.
   `skip_existing_files` is on, so re-running after an interruption skips
   every shard already on disk at the right size and re-fetches only the one
   that was cut off.
2. **`model-quantize`** (`target: dit`) — the 33B transformer to 4-bit,
   group size 64. **Leave `quant_modulation: true` alone.** H3's per-block
   AdaLN modulation is not a small side projection the way other DiTs' is —
   it is **13B of the 33B**, and left at bf16 it holds the "4-bit" checkpoint
   at ~36 GB, which is most of the reason to quantize at all. vpipe puts it
   at **8-bit** while the body goes to 4, because the modulation is what the
   residual scale rides on; the loader detects the per-tensor bit width, so
   the mix needs no configuration to stay in sync.
3. **`model-quantize`** (`target: text_encoder`) — the Qwen3-VL-32B prompt
   encoder to 4-bit. Its output, `local/MiniMax-H3-FL2VA-4bit`, is a
   **complete model**: the quantized parts plus everything untouched.
4. **`model-remove`** — deletes the intermediate from step 2, which exists
   only to feed step 3.

Everything lands under `models/` in the work directory. This takes a while and
is mostly disk-bound. When it finishes, `local/MiniMax-H3-FL2VA-4bit` is the
only thing you need; the `Comfy-Org/MiniMax-H3` download can go.

> **Gated repo.** If `model-fetch` reports an authorization failure, accept
> the model's licence on its Hugging Face page and put a token in the stage's
> `hf_token`.

## Step 2 — text to video **and audio**

```sh
cd ~/vpipe-work                                    # the SAME work directory
~/src/vpipe/build/apps/web-ui/vpipe-web-ui
```

Start the web UI from the work directory you prepared into — that is where the
registry holding your model lives. Then open the URL it prints, load
`minimax-h3-text-to-video.vpipeline`, edit the `text-prompt` stage, and press
Start. The shipped prompt asks for both halves at once:

> *A red paper boat drifting down a rain-slicked gutter at dusk, street light
> reflections rippling on the water. Sound of the rain is heard with
> occasional sound of individual water drops.*

Describe the **sound** as well as the picture. There is no separate audio
prompt — the soundtrack comes from the same text, so a prompt that mentions
only what things look like will get you whatever the model thinks that scene
sounds like.

The graph is eight stages:

```
text-prompt ──> diffusion-conditioner ──> generate-video ─┬─0─> vae-decode ──> rgb-to-video ─┐
                                                          │                                  ├─> save-video
                                                          └─1─> audio-vae-decode ────────────┘

model-select ──> diffusion-conditioner, generate-video, vae-decode, audio-vae-decode
```

`model-select` names the model once and every model-holding stage latches it,
so you point **one** stage at a checkpoint rather than four.
`generate-video` emits **two** latents — video on port 0, audio on port 1 —
which decode separately and meet again at `save-video`, muxed into one
`.mp4`.

The shipped `output_url` is `/minimax-h3-text-to-video.mp4`, and that leading
`/` is the **sandbox** root, not your filesystem root: under the web UI the
clip lands at `sandbox/minimax-h3-text-to-video.mp4` in the work directory.
Run the same graph from the CLI, which has no sandbox, and `/` means what it
usually does — give it a real path there.

### The settings worth knowing

From the `generate-video` stage:

| key | shipped | notes |
|---|---|---|
| `width` / `height` | 960 × 544 | Both must be multiples of 16. |
| `frames` | 120 | **Rounded up** to the nearest count the VAE can chunk — 5, 22, 39, 56, 73, 90, 107, **124**, … So 120 becomes 124. The stage logs the change. |
| `fps` | 24 | 124 frames ≈ 5.2 s; 56 ≈ 2.3 s. |
| `steps` | 8 | The model is distilled: **4 is enough** to see what a prompt does, 8 for a final. `guidance_scale` and a negative prompt are **inert** here — a distilled model has no unconditional pass to guide against, so vpipe skips it rather than paying 2× on a 33B model for nothing. |
| `seed` | 6 | Same seed + same settings ⇒ same clip. |
| `unload_when_idle` | `always` | Drop the weights between runs. On 16 GB this is what lets the next stage have the machine. |

Audio length follows from `frames` and `fps`; set `audio_seconds` only to
override that.

### More than text in

The checkpoint is named **FL2VA** — *first-and-last to video and audio*. Feed
`generate-video`'s port 5 a `vae-encode` of one image and generation is
anchored to it as the opening frame; add a second `vae-encode` on port 6 and
the model interpolates between two stills. Both anchors must be encoded at
the same resolution the clip is generated at.

## Memory

**16 GB is the floor, and it works** — but only because the two big models are
never fully resident. vpipe decides this per run and says what it chose in the
log.

**Weight streaming.** At 4-bit the transformer is ~24 GB and the prompt
encoder ~15 GB, on a machine with 16 GB. Both stream their layers from disk
instead of loading whole, so peak memory is set by the *working set*, not by
the checkpoint. `unload_when_idle: always` on the conditioner also means the
prompt encoder is gone before the denoise starts — the two never share the
machine.

**Adaptive residency.** Streaming everything, every step, re-reads ~4.5 GB per
forward pass. So the transformer *keeps* blocks after using them for as long
as free memory allows, growing its resident set into whatever the box has
spare and giving it back under pressure. On a 16 GB Mac it settles around
10 GB resident.

**This is where more memory pays.** The resident set is bounded by free RAM
and nothing else, so a 32 or 64 GB machine holds proportionally more of the
model between steps and re-reads proportionally less. On a large enough
machine it stops streaming altogether and simply preloads. Nothing needs
configuring for this — it is measured at load and adapts as the run proceeds.

**One heavy job at a time.** Metal buffers are wired and cannot be paged out,
so a second large model running alongside this one does not slow the machine
down gracefully — it exhausts it. Let a generation finish.

## Troubleshooting

**The model isn't in the picker.** `model-select`'s Browse list is filtered to
families the stages can actually run. If a model you prepared is missing,
check its registry `model_type` is `minimax-h3-fl2va`; you can always type the
key or an absolute path instead.

**It generated video but the audio is silent or wrong.** Check that
`save-video` has `enable_audio: true` and that `audio-vae-decode` is wired to
`generate-video` **port 1** (port 0 is video).

**`frames` isn't what you asked for.** Expected — see the table above.

## Under the hood

- One packed sequence carries video and audio rows together; the per-row AdaLN
  modulation that conditions it is 13B of the 33B.
- Two sigma schedules advance in lockstep — video shifted 12, audio 3
  (`video_shift` / `audio_shift`).
- The video VAE is 24-channel at 1/16 resolution; audio decodes through a
  separate VAE to **32 kHz stereo**.
- On **M5**, the GEMMs and attention run on the GPU's matrix cores
  (`matmul2d` / NAX flash attention).

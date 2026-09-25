# MiniMax H3 on Apple Silicon

English | [简体中文](MINIMAX-H3-zh-cn.md)

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
arrives in **8–16 steps** instead of 30+.

## Contents

- [What you need](#what-you-need)
  - [Disk space](#disk-space)
- [The pipelines](#the-pipelines)
- [Step 1 — prepare the model](#step-1--prepare-the-model)
  - [First, choose a work directory](#first-choose-a-work-directory)
  - [Then run the pipeline](#then-run-the-pipeline)
- [Step 2 — text to video and audio](#step-2--text-to-video-and-audio)
  - [The settings worth knowing](#the-settings-worth-knowing)
  - [How long it takes](#how-long-it-takes)
  - [Watching it form — live previews](#watching-it-form--live-previews)
  - [More than text in](#more-than-text-in)
  - [Conditioning on references (Ref2VA)](#conditioning-on-references-ref2va)
    - [Two ways to hand it a reference](#two-ways-to-hand-it-a-reference)
    - [No references this time — prompt only](#no-references-this-time--prompt-only)
    - [The example](#the-example)
    - [What it costs](#what-it-costs)
    - [How a reference is read](#how-a-reference-is-read)
    - [References that are not files](#references-that-are-not-files)
    - [Preparing the Ref2VA checkpoint](#preparing-the-ref2va-checkpoint)
    - [Ref2VA-like — references on the FL2VA weights](#ref2va-like--references-on-the-fl2va-weights)
  - [Longer clips — one story in four parts](#longer-clips--one-story-in-four-parts)
    - [Running the chain](#running-the-chain)
    - [How a part takes its guide](#how-a-part-takes-its-guide)
    - [Writing the prompts](#writing-the-prompts)
    - [Joining the parts](#joining-the-parts)
    - [Making it your own](#making-it-your-own)
  - [The released weights, either partition](#the-released-weights-either-partition)
  - [Fewer steps — the Turbo LoRA](#fewer-steps--the-turbo-lora)
    - [Get it](#get-it)
    - [Run it](#run-it)
    - [Runtime (recommended)](#runtime-recommended)
    - [Two at once](#two-at-once)
    - [Merging, and why it loses most of this adapter](#merging-and-why-it-loses-most-of-this-adapter)
    - [Which Turbo adapters work](#which-turbo-adapters-work)
    - [Community LoRAs — Civitai, musubi-tuner, ai-toolkit](#community-loras--civitai-musubi-tuner-ai-toolkit)
  - [Eight steps — HyperFlow](#eight-steps--hyperflow)
    - [Fetch it, then name it](#fetch-it-then-name-it)
    - [What makes it different](#what-makes-it-different)
    - [What it costs](#what-it-costs-1)
  - [Faster attention — the VDN linear branch](#faster-attention--the-vdn-linear-branch)
    - [Get it and run it](#get-it-and-run-it)
    - [What it saves](#what-it-saves)
  - [Cheaper attention — SageAttention's int8 QK](#cheaper-attention--sageattentions-int8-qk)
  - [Faster attention — Sol-Attn routing](#faster-attention--sol-attn-routing)
    - [What it saves](#what-it-saves-1)
    - [Against the VDN branch](#against-the-vdn-branch)
    - [The knobs](#the-knobs)
  - [The Neural Engine — for M4-family Macs](#the-neural-engine--for-m4-family-macs)
    - [What it saves on an M4](#what-it-saves-on-an-m4)
    - [How it behaves](#how-it-behaves)
- [Memory](#memory)
- [Troubleshooting](#troubleshooting)
- [Under the hood](#under-the-hood)
- [References and licences](#references-and-licences)

## What you need

| | |
|---|---|
| **Machine** | Apple Silicon Mac (M-series). |
| **Memory** | **16 GB minimum.** More is faster — see [Memory](#memory). |
| **Disk** | **~155 GB** to prepare, **~45 GB** to keep. See below. |
| **Build** | An Apple Silicon build of vpipe — the default on arm64 macOS. See the main [README](../README.md). |

### Disk space

The published checkpoint is bf16 and very large; vpipe quantizes it to 8-bit
once, up front. Both copies exist while that runs:

| | |
|---|---|
| Source repack (`Comfy-Org/MiniMax-H3`, bf16) | **~115 GB** |
| Peak while preparing (source + output) | **~180 GB** |
| 8-bit model, source deleted | **~65 GB** |
| Adding the Ref2VA partition | **+66 GB** source, **+60 GB** 8-bit (~**+44 GB** at 4-bit) |

Once preparation finishes you can delete the source repack and keep the
~65 GB. The 115 GB is a one-time cost, not a standing one.

The Ref2VA row is the whole partition, not just its transformer, and that is
worth reading twice. The *download* really does add only the 66 GB
transformer — the prompt encoder and both VAEs are already on disk and are
skipped. But the pipeline that prepares it **quantizes the encoder too**, into
an output of its own, so what you end up holding is a second complete model:
~33 GB of transformer plus ~27 GB of encoder at 8-bit. Only the components
quantization leaves alone are hard-linked, and a re-quantized encoder is new
bytes. The 4-bit figure is that same sum scaled by the transformer's own
8-bit-to-4-bit ratio rather than a measurement; treat it as a budget, not a
promise.

**Why 8-bit and not 4.** Quality, at the price of disk and very little else.
Both widths stream, so neither has to fit in RAM, and the transformer spends
its time on weights it is reading from storage rather than on how wide they
are — so moving to 8-bit costs about 20 GB and does not meaningfully change
how long a clip takes. The 4-bit path still works and is the one to take if
the disk matters more: set `bits` to 4 in both `model-quantize` stages and
name the output to match.

**You do not have to quantize at all.** The downloaded bf16 repack is a
complete, runnable model: point `model-select` straight at
`Comfy-Org/MiniMax-H3-FL2VA` and skip [step 1](#step-1--prepare-the-model)'s
quantize stages entirely. Nothing has to be told what width it is — the
loader reads the per-tensor bit width out of the checkpoint, and a checkpoint
that carries none is simply the dense bf16 path.

What it costs is memory, and the cost is paid per step rather than once. The
transformer is ~66 GB at bf16 against ~33 GB at 8-bit, so streaming it
re-reads about twice the bytes on every forward pass, and the resident set —
which grows into whatever RAM is spare, see [Memory](#memory) — holds
proportionally fewer blocks between steps. On a memory-bounded box that is
the whole difference: the same clip, more time in the storage path. Quantize
when you are going to generate more than once; run the repack as it comes when
you want to see the model work before spending the hours and the 115 GB.

> **Keep the download and the output on one filesystem.** Components that
> quantization does not touch (the VAEs) are **hard-linked** into the output
> rather than copied, which is why the numbers above are smaller than they
> look. Across two volumes the link fails, vpipe falls back to a real copy,
> and you pay for those bytes twice.

## The pipelines

- **[`prepare-minimax-h3-8bit.vpipeline`](pipelines/prepare-minimax-h3-8bit.vpipeline)**
  — download the checkpoint and quantize it. Run once.
- **[`prepare-minimax-h3-ref2va-8bit.vpipeline`](pipelines/prepare-minimax-h3-ref2va-8bit.vpipeline)**
  / **[`…-4bit`](pipelines/prepare-minimax-h3-ref2va-4bit.vpipeline)**
  — the same for the **Ref2VA** partition (see
  [Conditioning on references](#conditioning-on-references-ref2va)). Only if
  you want it; the two share a download.
- **[`minimax-h3-text-to-video.vpipeline`](pipelines/minimax-h3-text-to-video.vpipeline)**
  — prompt in, `.mp4` with sound out.
- **[`minimax-h3-text-to-video-preview.vpipeline`](pipelines/minimax-h3-text-to-video-preview.vpipeline)**
  — the same, with a **live preview** of the clip forming, step by step (see
  [Watching it form](#watching-it-form--live-previews)).
- **[`minimax-h3-first-last-to-video.vpipeline`](pipelines/minimax-h3-first-last-to-video.vpipeline)**
  — the same, anchored to an image at both ends (see
  [More than text in](#more-than-text-in)).
- **[`minimax-h3-reference-to-video.vpipeline`](pipelines/minimax-h3-reference-to-video.vpipeline)**
  — the **Ref2VA** partition instead: reference images, clips and soundtracks
  in, `.mp4` out, each one prepared to a size you choose (see
  [Conditioning on references](#conditioning-on-references-ref2va)).
- **[`minimax-h3-ref2va-like.vpipeline`](pipelines/minimax-h3-ref2va-like.vpipeline)**
  — reference images on the **FL2VA** weights, with the VDN branch and the
  Turbo LoRA, so no second 66 GB transformer is needed (see
  [Ref2VA-like](#ref2va-like--references-on-the-fl2va-weights)).
- **[`minimax-h3-extend-part1.vpipeline`](pipelines/minimax-h3-extend-part1.vpipeline)**
  … **[`…-part4`](pipelines/minimax-h3-extend-part4.vpipeline)**
  — a **40-second** clip in four runs: FL2VA text-to-video, then three Ref2VA
  continuations, each carrying on from the last 3.75 s of the part before it
  (see [Longer clips](#longer-clips--one-story-in-four-parts)). Runs both
  partitions of the **released** weights at 21 steps, with no Turbo adapter,
  and writes each part **lossless** — FFV1 video at 4:4:4, ALAC sound — so
  every continuation reads back exactly what the part before it made.
- **[`minimax-h3-extend-concat.vpipeline`](pipelines/minimax-h3-extend-concat.vpipeline)**
  — joins those four parts into one 40-second file through the concat
  demuxer, with no model and no hand-written ffmpeg. The one lossy encode in
  the chain.
- **[`prepare-minimax-h3-vdn.vpipeline`](pipelines/prepare-minimax-h3-vdn.vpipeline)**
  / **[`minimax-h3-vdn.vpipeline`](pipelines/minimax-h3-vdn.vpipeline)**
  — fetch the **VDN** hybrid-attention branch and run text-to-video with it.
  **FL2VA partition only** — text or first/last keyframes, not references —
  and worth more the longer the clip **and the larger the frame** (see
  [Faster attention](#faster-attention--the-vdn-linear-branch)).

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
cp ~/src/vpipe/docs/pipelines/prepare-minimax-h3-8bit.vpipeline .
~/src/vpipe/build/apps/vpipe/vpipe --launch prepare-minimax-h3-8bit.vpipeline
```

The CLI suits this job better than the web UI: it is long, unattended and
disk-bound, with nothing to click once it starts. It prints a live progress
bar per stage, and `Ctrl-C` stops it cleanly — `skip_existing_files` means
re-running picks up where it left off.

Four stages, in order:

1. **`model-fetch`** — pulls `Comfy-Org/MiniMax-H3` into `./models`.
   `skip_existing_files` is on, so re-running after an interruption skips
   every shard already on disk at the right size, and the one that was cut
   off resumes from the byte it stopped at rather than starting over — the
   partial sits next to it as `<name>.part` until it is complete and its
   checksum matches. A shard here is 30–60 GB, so that difference is hours.

   Shards over 256 MB are pulled from HuggingFace's **content store**
   rather than streamed: the repo publishes a hash the file can be
   rebuilt from, and the store holds it as deduplicated, compressed
   chunks that come down several ranges at a time. For bf16 weights that
   is 0.873× the bytes — the store separates the byte planes before
   compressing, which is what makes floats compress at all. Measured
   end to end on a 5.2 GB shard it was 1.07× faster here (94.4 vs 88.3
   MB/s median), because one stream was already close to this link's
   ceiling; on a link where a single stream is the constraint it is
   worth closer to 2×. `xet_streams: 0` turns it off.

   **`model_variant: fl2va` is required, not decorative.** That one repo
   publishes *two* models — the FL2VA and Ref2VA partitions — and they pin
   different transformer files and different tokenizers. A fetch that does
   not say which is refused, with both listed, rather than quietly taking
   the first.

   **`model_key` is what the next stage names.** The two partitions share
   a directory on disk, so what keeps them apart is the key each is
   registered under in the models DB — not the repo path, which is the
   same string for both. `model_key: Comfy-Org/MiniMax-H3-FL2VA` pins it,
   and step 2's `src_model` has to be that key. Passing the repo path
   (`Comfy-Org/MiniMax-H3`) there instead is the mistake to avoid: the
   fetch succeeds, and the quantize then fails to find a model of that
   name. Left unset, the key falls back to the catalogue entry's name,
   which happens to be the same string here — so pinning it costs
   nothing and makes the pipeline say what it depends on.
2. **`model-quantize`** (`target: dit`) — the 33B transformer to 8-bit,
   group size 64. **Leave `quant_modulation: true` alone.** H3's per-block
   AdaLN modulation is not a small side projection the way other DiTs' is —
   it is **13B of the 33B**, so leaving it at bf16 would keep 26 GB of the
   transformer unquantized inside an otherwise-8-bit pack, which is most of
   the reason to quantize at all. The loader reads the per-tensor bit width
   out of the checkpoint, so nothing downstream has to be told what was
   quantized to what.
3. **`model-quantize`** (`target: text_encoder`) — the Qwen3-VL-32B prompt
   encoder to 8-bit. Its output, `local/MiniMax-H3-FL2VA-8bit`, is a
   **complete model**: the quantized parts plus everything untouched.
4. **`model-remove`** — deletes the intermediate from step 2, which exists
   only to feed step 3.

Everything lands under `models/` in the work directory. This takes a while and
is mostly disk-bound. When it finishes, `local/MiniMax-H3-FL2VA-8bit` is the
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

> *Cinematic video of a young Asian female pianist passionately playing a
> grand piano.*

The soundtrack is in that sentence: naming the instrument is what gives the
model the music. There is no separate audio prompt — the sound comes from the
same text — so describe the **sound** as well as the picture. Either name it
outright (*"sound of the rain, with the occasional individual drop"*) or, as
here, name the thing making it. A prompt that says only what a scene looks
like gets you whatever the model thinks that scene sounds like.

The graph is nine stages:

```
text-prompt ──> diffusion-conditioner ──> generate-video ─┬─0─> vae-decode ──> rgb-to-video ─┐
                                                          │                                  ├─> save-video
                                                          └─1─> audio-vae-decode ────────────┘

model-select            ──> diffusion-conditioner, generate-video, vae-decode, audio-vae-decode
minimax-h3-model-config ──> generate-video (port 9)
```

`model-select` names the model once and every model-holding stage latches it,
so you point **one** stage at a checkpoint rather than four.
`generate-video` emits **two** latents — video on port 0, audio on port 1 —
which decode separately and meet again at `save-video`, muxed into one
`.mp4`.

The shipped `output_url` is **relative** — `minimax-h3-text-to-video.mp4` —
so the clip lands next to wherever you started vpipe, and the same file works
from the CLI and the web UI without editing.

Worth knowing if you see the other form: a **leading `/` is the sandbox
root** under the web UI, not your filesystem root, so `/clip.mp4` there means
`sandbox/clip.mp4` in the work directory. The CLI has no sandbox, and `/`
means what it usually does — which on a Mac is a read-only volume, so a graph
carrying an absolute path from the UI will generate a whole clip and then
fail to write it.

### The settings worth knowing

From the `generate-video` stage:

| key | shipped | notes |
|---|---|---|
| `width` / `height` | 960 × 544 | **Rounded up** to the nearest multiple of **32** — the video VAE's 16× spatial stride times the DiT's 2× patch. A multiple of 16 is not enough: 1360 is one, and its latent is an odd 85 that the packer cannot patch. The stage logs the change. |
| `frames` | 120 | **Rounded up** to the nearest count the VAE can chunk — 5, 22, 39, 56, 73, 90, 107, **124**, … So 120 becomes 124. The stage logs the change. |
| `fps` | 24 | 124 frames ≈ 5.2 s; 56 ≈ 2.3 s. |
| `steps` | 8 | **8 is draft quality** — enough to see what a prompt does — and **16 gives good quality**. Fewer than 8 is the [Turbo LoRA](#fewer-steps--the-turbo-lora)'s territory, not this model's. `guidance_scale` and a negative prompt are **inert** here — a distilled model has no unconditional pass to guide against, so vpipe skips it rather than paying 2× on a 33B model for nothing. |
| `seed` | 6 | Same seed + same settings ⇒ same clip. |
| `i8_gemm` | `true` | An opt-in **lossy** accelerated mode, on in every shipped pipeline here. Only matrix-core GPUs (M5 and newer) can use it, so it does nothing on an M4 — and on an M5 turning it off is slower. It changes the picture slightly, so turn it off when you are judging output rather than speed. |
| `sage_attn` | `false` | An opt-in **lossy** accelerated mode, independent of both `i8_gemm` and `sol_attn` and settable with either — it runs the attention's QK^T product in int8 with a per-block scale, where `sol_attn` decides which blocks are attended at all. 1.20× on the attention at video geometry, at the same cosine the f16 kernel scores. Matrix cores only. See [Cheaper attention — SageAttention's int8 QK](#cheaper-attention--sageattentions-int8-qk). |
| `ane_ffn` / `ane_qkv` | `false` | Opt-in **lossy** modes that run part of every block on the **Apple Neural Engine** beside the GPU. Worth it on an **M4-family** Mac, normally not on an M5. See [The Neural Engine](#the-neural-engine--for-m4-family-macs). |
| `sol_attn` | `false` | Another opt-in **lossy** accelerated mode, and an independent one — it changes how the attention between the GEMMs is computed where `i8_gemm` changes the GEMMs. 1.27× on the wall clock at 124 frames of 832 × 480, with no extra weights; see [Faster attention — Sol-Attn routing](#faster-attention--sol-attn-routing) for the knobs beside it. |
| `unload_when_idle` | `always` | Drop the weights between runs. On 16 GB this is what lets the next stage have the machine. |

And from the **`minimax-h3-model-config`** stage, wired to `generate-video`'s
`model_config` iport (port 9):

| key | shipped | notes |
|---|---|---|
| `video_shift` / `audio_shift` | 12.0 / 3.0 | The two sigma schedules. Not interchangeable — these are the released checkpoint's. |
| `condition_timestep` | 1.0 | The level the pinned keyframe rows sit at. `1.0` is **clean** in this model's `t = 1 − sigma` convention. |
| `condition_audio_timestep` | 1.0 | The same, for a Ref2VA reference **soundtrack**. |
| `audio_seconds` | 0 | Audio length follows from `frames` and `fps`; set this only to override that. |

These are H3's own knobs, so they live in an H3 stage rather than in
`generate-video`, which keeps only what every video model answers to
(geometry, length, steps, seed, residency). Leave the stage out and the
defaults above apply. Give it a **trigger** iport and it re-emits once per
inbound beat, so the settings can change per clip in a graph that generates
continuously; with no trigger it emits once for the run.

There is deliberately **no guidance scale** here: H3 is distilled, and a
distilled model has no unconditional pass to guide against. Wan's guidance
and expert boundary live in `wan2-model-config` for the same reason — each
family carries its own.

### How long it takes

Measured on the 8-bit model at 960 × 544 (0.5 MP) and 24 fps, 6 steps with
the [Turbo LoRA](#fewer-steps--the-turbo-lora) applied at run time, on the
smallest machine that runs this at all — a **fanless MacBook Air 15-inch
(M5)**, 10-core CPU / 10-core GPU, 16 GB — and on a **MacBook Pro 16-inch
(M5 Pro)**, 24 GB:

| frames | clip | M5 Air, 16 GB | M5 Pro, 24 GB |
|---|---|---|---|
| 90 | 3.75 s | **9 min 26 s** | **3 min 20 s** |
| 124 | 5.2 s | **11 min 25 s** | **5 min 0 s** |

`steps` is the setting that moves this most, and the Turbo adapter is what
buys the low count: without it, plan on 8 steps for a draft and 16 for a
final clip, at roughly proportional cost. What moves the cost of each *step*
is one of the two attention settings: [the VDN linear
branch](#faster-attention--the-vdn-linear-branch), which takes the
124-frame Pro run to **4 min 38 s** and saves more the longer the clip, or
[Sol-Attn routing](#faster-attention--sol-attn-routing), which needs no
extra weights and takes a 124-frame run at 832 × 480 from 3 min 30 s to
**2 min 44 s**.

**The two columns are not equally solid, and it is worth saying which is
which.** The M5 Pro column is repeatable: fans, one pinned clock, the same
figure run to run. The M5 Air column is not. An ice pack is placed by hand,
and where it sits changes both how long the boost window lasts and how far
the clock falls afterwards, so those figures carry a run-to-run spread that
has not been quantified here.

So the gap reads **2.8× at 90 frames and 2.3× at 124**, but the difference
between those two is not structure — it is mostly the Air moving. The same
goes for scaling: 90 → 124 frames is 1.38× the length and costs the Air
1.21× its time against the Pro's 1.50×, and only the Pro's number is a
measurement rather than one sample of a noisy quantity. Take the Air column
as the order of magnitude a well-cooled fanless M5 reaches, and the Pro
column as a figure you can reproduce.

> **The M5 Air column was measured with the chassis sitting on an ice pack**,
> and even then it is a throttled machine for most of the run. It starts at
> the full **1578 MHz** and holds it for roughly the **first two minutes**,
> then throttles and settles into a fluctuation around **1300 MHz** — 82% of
> the part — because an ice pack is a heatsink that warms up, not stable
> cooling.
>
> So the penalty is a function of how LONG the job is, not a flat tax: a
> two-minute job never leaves the boost window, while these runs spend
> **79–82% of their wall clock** past it. Read the Air column as a machine
> that was fast at the start and is not by the end.
>
> It is also **noisy**, in a way the other column is not: the ice pack is
> placed by hand and where it sits moves both the length of the boost window
> and the clock it settles to, so repeating an Air run does not repeat its
> number. On a warm desk it drops further still — the 124-frame run takes
> about 15 minutes there.
>
> **The M5 Pro column is that machine as it ships**, on its own fans with
> nothing under it, and its fans are enough that the workload pins the GPU at
> **1620 MHz — the maximum — at 100% for the whole run.** That is **1.25×**
> the Air's sustained clock before any difference in core count, so clock
> alone accounts for part of the 2.3–2.8× gap and not for most of it.

### Watching it form — live previews

A denoise of this model runs for minutes, and without a preview there is
nothing to look at until the very end. `generate-video` can show the clip
as it forms: after a step, it takes the model's current best guess at the
finished clip and decodes it with a **tiny autoencoder** (a TAE) instead
of the real video VAE. The result goes out on `generate-video`'s port 2,
and a `preview` stage wired there plays it in the web UI, looping each
clip until the next step replaces it.

**[`minimax-h3-text-to-video-preview.vpipeline`](pipelines/minimax-h3-text-to-video-preview.vpipeline)**
is the text-to-video graph with this added. It needs one small file, and
[`prepare-minimax-h3-preview.vpipeline`](pipelines/prepare-minimax-h3-preview.vpipeline)
fetches it (23 MB) and registers it as `madebyollin/taeh3`:

```sh
cd ~/vpipe-work                                    # the SAME work directory
vpipe --launch ~/src/vpipe/docs/pipelines/prepare-minimax-h3-preview.vpipeline
```

That is madebyollin's **`taeh3`**, trained for this model's latent space.
It is published on GitHub rather than Hugging Face; the catalogue knows
where. It decodes the same frame count the real VAE does and keeps the
motion. **Kijai's `MiniMax-H3-TAE`** also works (`Kijai/MiniMax-H3-TAE`,
10 MB, also in the catalogue). It is a still-image decoder, though, so it
makes one frame per latent frame and plays them slower to fill the same
time. On the same clip it also scored lower against the real VAE: 23.3 dB,
against `taeh3`'s 25.3 dB.

The knobs are on `minimax-h3-model-config`, next to H3's other settings:

| key | default | notes |
|---|---|---|
| `preview_vae` | *(empty)* | The TAE: a registered model (`madebyollin/taeh3`), a directory holding one `.safetensors`, or a path to one. Empty turns previews off. |
| `preview_every` | 1 | Render after every *N*th step; the last step always renders. |
| `preview_max_edge` | 512 | Longest edge of the preview picture. The TAE always decodes at full size; the picture is then scaled down. Shrinking the latent first instead measured blurry, with colour shifts. |
| `preview_frames` | 0 | Preview only the first *N* frames; 0 is the whole clip. This is the knob that makes a preview cheaper. |

What it costs, measured on an M4 Pro:

- A full 90-frame 960 × 576 clip takes **0.94 s** to decode on an idle GPU.
- A 39-frame 512 × 288 preview took about **0.5 s** while the DiT ran
  (the model resident, 8 s per step). That, and the 3% below, were
  measured when the full clip took 2.1 s, before the decoder's ReLUs,
  residual adds, upsamples and concats were folded into its convs, so
  read both as upper bounds.
- The decode runs off the generation thread. If a render is still running
  when the next one is due, the waiting clip is replaced rather than
  queued, so previews never make the generation wait.
- The decode does share the GPU. A preview on **every** step added about
  **3%** to the denoise: 85 s without previews against 87 and 88 s with
  them, over 7 steps with the runs alternated. Raise `preview_every` to
  spend less.
- The last step's preview matched the real VAE decode at **30–32 dB**.

Leave port 2 unwired, or `preview_vae` empty, and nothing is loaded or
decoded. Two things to know:

- **One consumer only.** The port drops a clip nobody reads rather than
  holding the generation up, and that policy takes a single consumer.
- **The `preview` stage runs until you stop the pipeline.** It is a live
  view, so a graph that includes one does not finish by itself from the
  command line. Use it in the web UI.

### More than text in

The checkpoint is named **FL2VA** — *first-and-last to video and audio*. Feed
`generate-video`'s port 5 a `vae-encode` of one image and generation is
anchored to it as the opening frame; add a second `vae-encode` on port 6 and
the model interpolates between two stills. Both anchors must be encoded at
the same resolution the clip is generated at.

[`minimax-h3-first-last-to-video.vpipeline`](pipelines/minimax-h3-first-last-to-video.vpipeline)
is that graph, worked through. Drop a `reference.jpg` beside it and run it the
way you ran the text-to-video one:

```sh
cd ~/vpipe-work                                    # the work directory again
cp ~/src/vpipe/docs/pipelines/minimax-h3-first-last-to-video.vpipeline .
cp <your picture> reference.jpg
~/src/vpipe/build/apps/vpipe/vpipe --launch minimax-h3-first-last-to-video.vpipeline
```

Run it from that directory, not from anywhere else: `models/` and the model
registry are both resolved relative to where vpipe starts, so a graph naming
`local/MiniMax-H3-FL2VA-8bit` from the wrong place reports every stage
inert.

It builds **both anchors from one picture**, which is what turns a still into
a camera move rather than a cross-fade. `load-image` fans out to two
`image-resample` stages:

| stage | `fit` | what it frames |
|---|---|---|
| `first-frame` | `crop` | the whole picture, centre-cropped to 960x544 |
| `last-frame` | `manual` | a tighter window inside it, at `scale` 1.6875 |

The model fills in the ~5 seconds between them, so the clip reads as a slow
push-in. Swap the two and it pulls out.

**Both resamplers use `algorithm: lanczos`.** The anchors are the only place
the source picture's detail enters the model — everything after them is
latents — so this is the one resize in the graph worth paying for. Bilinear
softens the fine texture (brush strokes, foliage, fabric) and the VAE then
encodes the softening as if you had meant it.

**The `manual` numbers describe a 1024x1024 source.** `src_x`/`src_y` are
absolute pixels into your own image and `scale` is the resample ratio, so the
window is `width / scale` by `height / scale` pixels starting at that corner —
960 / 1.6875 = 569 wide here. Point it at a picture of another size and you
will frame something else. Two ways out: retune the three numbers, or give the
two `image-resample` stages **two different pictures** with `fit: crop` on
both, which needs no arithmetic and is the plain reading of *first and last*.

**It applies the Turbo LoRA**, which is what lets it run at 6 steps — so
fetch the adapter first (see [Fewer steps](#fewer-steps--the-turbo-lora)).
If you would rather not, delete the `lora` and `lora_scale` keys from
`minimax-h3-model-config` and put `steps` back up to 8; everything else in
the graph is the same.

`frames` is **124**, not the 120 the text-to-video examples use, because H3
chunks video 17 frames at a time keeping 5 latents — any count is rounded up
to the next `17n + 5`. Naming one that already fits means the number in the
file is the number you get.

The two `vae-encode` stages release the VAE as soon as their keyframe is
encoded, which matters more than it sounds: H3's video VAE is 5.2 GB and the
DiT wants its scratch immediately afterwards. On a 16 GB machine holding both
at once is the difference between a clip and a refusal.

### Conditioning on references (Ref2VA)

MiniMax-H3 ships a **second checkpoint**, `ref2va`, that conditions on a
*list* of reference media instead of on keyframes: up to **9 images**, **3
video clips** and **3 soundtracks**, twelve in total, and audio can never be
the only kind. Images carry subject and style, clips carry motion and camera,
soundtracks carry a voice or a piece of music.

The two partitions are the same architecture and ship byte-identical
transformer configs, so **nothing in the weights tells them apart** — vpipe
reads it off the packaging. A Ref2VA checkpoint wired as if it were FL2VA is
refused rather than run: it would load, denoise at full 33B cost, and generate
video conditioned on nothing. Asking it for *no references on purpose* is a
different request, and one it takes — see
[prompt only](#no-references-this-time--prompt-only).

The *other* direction is a real mode rather than a mistake: FL2VA weights will
take reference images through this same sequence, which is what
[Ref2VA-like](#ref2va-like--references-on-the-fl2va-weights) below is. Read
this section first — the wiring, the limits and how a reference is read are
the same — and that one for what changes.

> **A reference is not a keyframe, and Ref2VA cannot pin one.** The two
> partitions pack different sequences — FL2VA's is
> `[text | keyframe conditions | target audio | target video]` and Ref2VA's is
> `[text | reference blocks | target audio | target video]` — so there is no
> slot that binds a reference to output frame 0. A reference image conditions
> the **whole clip**: it carries subject, outfit and style everywhere, and
> nowhere in particular.
>
> Asking for one in the prompt (*"use `<Picture 2>` as the opening frame"*)
> therefore does nothing. It is not disobedience; there is no machinery for
> that instruction to act on, and the give-away is a run where the wardrobe
> and the face transfer while the first frame does not. Anchoring an opening
> frame — continuing from the last frame of a previous clip, say — is
> [FL2VA's job](#more-than-text-in), on `generate-video` port 5, and the two
> are mutually exclusive: taking the anchor costs you the reference list.
> `generate-video` **warns** if you wire a keyframe on a Ref2VA graph rather
> than dropping it quietly.
>
> Continuing a **clip** is a different request, and one Ref2VA does take. A
> reference video described as the source of a `[video continuation]` is
> carried on from its end: its motion, its subjects and its sound. That is
> behaviour the model learned, not a pinned frame; see
> [Longer clips](#longer-clips--one-story-in-four-parts).

Wire a **`video-ref-encoder`** stage:

| | |
| --- | --- |
| port 0 in | the prompt |
| port 1 in | optional `model-select` |
| ports 2–7 in | optional `ref1`..`ref6` — a reference as a tensor |
| port 0 out | conditioning → `generate-video` port 0 |
| port 1 out | reference video rows → `generate-video` port **7** |
| port 2 out | reference audio rows → `generate-video` port **8** |

#### Two ways to hand it a reference

**A list of files**, in the stage's `references` config. They are paths to
open, so the composer's file browser fills them in — **select several at
once** and they land in the list in the order you picked them:

```json
"references": ["subject.png", "motion.mp4", "voice.wav"]
```

**Or a tensor**, on one of the six `ref` iports. That is for the references a
path cannot name — a still your graph just generated, a cropped frame, a clip
that was never written to disk — and for choosing each reference's geometry
yourself. The contract is in
[References that are not files](#references-that-are-not-files) below.

The two mix freely: port references are numbered *after* the list, and the
limits (9 images, 3 clips, 3 soundtracks, 12 total) apply to the union. Reach
for the list when the references are files you have and are happy for the
model's own rule to size; reach for the ports when a stage produces the
reference, or when you want to set its size.

#### No references this time — prompt only

An **empty list** is a request of its own: generate from the prompt alone,
through the Ref2VA checkpoint that is already loaded.

```json
"references": []
```

That is what lets one graph serve requests with and without references —
dropping the last reference no longer means swapping to the FL2VA checkpoint
(and holding both, on a box that uses Ref2VA elsewhere), and adding one later
is a change to the list rather than to the graph. A wired `ref` port that
beats an **empty tensor** says the same thing, since that is its declared way
of saying *nothing this time*.

What runs is text-to-video's own sequence, `[text | target audio | target
video]`, read by the Ref2VA weights: the conditioning is the prompt alone,
byte for byte what `diffusion-conditioner` produces for it, and no reference
rows are packed. The log says so on each side:

```
VideoRefEncoderStage('refenc'): prompt only (an explicitly empty reference
  list) -> 34 conditioning rows, no reference rows
GenerateVideoStage('gen'): prompt-only Ref2VA -- the request's reference list
  is explicitly empty, so the Ref2VA weights denoise from the prompt alone
  over the text-to-video layout
```

**Leaving the key out is not the same request.** With no `references` key and
no `ref` port wired, the encoder has been handed nothing at all — which is
what an unwired graph looks like — and it skips the request with a warning
rather than spend a 33B denoise on it. `generate-video` holds the same line:
on the Ref2VA checkpoint, a conditioning that carries no reference list (one
from a `diffusion-conditioner`, say) is refused, and only an explicitly empty
one runs.

On the **FL2VA** checkpoint an empty list is simply text-to-video, and a
keyframe wired on port 5 is honoured as usual. On Ref2VA it is not: the
prompt-only request is still Ref2VA's sequence, which has no keyframe slot.

This is the Ref2VA weights doing text-to-video, which the sequence permits
but the Ref2VA recipe does not describe. The FL2VA checkpoint is the trained
route for it; this one is for the graph that already holds Ref2VA.

#### The example

[`minimax-h3-reference-to-video.vpipeline`](pipelines/minimax-h3-reference-to-video.vpipeline)
runs on assets you already have:

| reference | carries | where it comes from |
|---|---|---|
| `minimax-h3-reference-subject.jpg` | the subject | [ships in this repo](images/minimax-h3-reference-subject.jpg) — made with vpipe's own FLUX.2 text-to-image graph, so it comes with no licence question attached |
| `minimax-h3-text-to-video.mp4` | the camera move **and** the soundtrack | whatever [step 2](#step-2--text-to-video-and-audio) wrote. A clip with audio stays ONE reference carrying both, labelled `<Video 1>` and `<Audio 1>`, however you feed it |

It takes the **ports** route, preparing each reference through ordinary stages:

```
load-image → image-resample(1024×1024)                              → ref1
load-video ┬→ video-to-rgb → image-resample(1344×768) → temporal-stack → ref2
           └→ audio-to-pcm(32000, stereo) → temporal-stack             → ref3
```

Seventeen stages where the list would need three lines, and it buys one thing:
the two `image-resample` sizes are yours. The ports default to
`short_edge: 0`, so the encoder takes what it is handed instead of re-resolving
it — change those numbers and the reference geometry changes with them. Swap
the whole chain for a `references` list if you would rather have the three
lines; the stage takes either.

Four details in that shape are load-bearing. **One `load-video` feeds both
streams**, so the clip and its soundtrack still leave one container together —
the sync argument the `references` list was built on, kept rather than traded.
**`attach_audio: [3]`** on the encoder makes **ref3**'s audio — the PCM on
iport 4, the third reference — the soundtrack of the reference before it, so
it stays one `<Video 1>` + `<Audio 1>` pair instead of becoming a third,
independent reference. The number is a **reference** number in `1..6`, not an
iport index: ref1 is iport 2. **`channels: 2`** carries true stereo, which the
`references` path also does and a mono chain would silently give up.
And **`max_mb: 384`** on the clip stacker is sized on purpose: the 124-frame
clip step 2 writes is 366 MiB at 1344 × 768, over the 256 MiB default, which
would otherwise cap the group at 86 frames and warn. Note how little headroom
384 leaves — a longer reference clip needs a larger ceiling, or a `duration_s`
on the `load-video`.

MEASURED: this produces a request identical to the `references` list at every
count that exists — 2 references, 3,115 conditioning rows, 13,120 reference
video rows, 130 audio rows, a 22,615-row packed sequence. It is not the same
*video*, at the same seed: the resize now happens in `image-resample` rather
than inside the encoder, and two implementations landing on the same dimensions
do not land on the same bytes. If you want bit-equivalence with the file list
instead of the size control, leave the resamples at the source size and put
`short_edge: 768` in the clip stacker's `sideband`.

One inefficiency worth knowing: the encoder truncates a clip to `frames`, so
this chain resizes all 124 frames of the reference and uses 39. The
`references` list truncates first and resizes only what it keeps. Bound the
source rather than raise `max_mb` if that matters.

```sh
cd ~/vpipe-work                                    # the work directory again
cp ~/src/vpipe/docs/pipelines/minimax-h3-reference-to-video.vpipeline .
cp ~/src/vpipe/docs/images/minimax-h3-reference-subject.jpg .
~/src/vpipe/build/apps/vpipe/vpipe --launch minimax-h3-reference-to-video.vpipeline
```

The `.mp4` is already beside them if you ran step 2 from this directory; any
clip with a soundtrack will do. Point `model-select` at your Ref2VA directory
if you named it something else, and note the two `frames` settings — the
encoder's and `generate-video`'s — which already agree at 39 and have to.

#### What it costs

**Ref2VA costs more than its output suggests**, and the example is sized
around that rather than around the clip it makes. A reference is packed into
the SAME sequence as the thing being generated, and a reference clip goes onto
its own canvas: the 960 × 544 file from step 2 resolves to **1344 × 768**,
more pixels than the output. MEASURED on the 16 GB M5, asking for 56 frames:

| | rows | |
|---|---|---|
| the reference clip | 18,160 | 58% |
| conditioning (text + vision) | 4,131 | 13% |
| **what is being generated** (960 × 544, 56 frames) | 8,856 | 28% |
| the reference soundtrack | 188 | <1% |

31,335 rows wants ~6.3 GB of activation scratch, and `generate-video`
**refuses** at that size on a 16 GB box rather than thrash — wired Metal
buffers cannot be paged out, so overcommitting takes the machine down instead
of failing one stage.

So `frames` is the lever, not the frame size: a reference clip is truncated to
the generated length, so shortening the output shortens the reference with it.
Dropping to 39 moves both halves and fits — 22,615 rows, of which 13,120 are
the clip. Making only `width`/`height` smaller would cut the 28% and leave the
58% exactly where it is.

The **other** lever is the reference canvas itself. That 1344 × 768 is an
*upscale* of a 960 × 544 file — 1.98× the pixels, interpolated from the same
information — and declining it is worth measuring. MEASURED back to back on an
idle 16 GB M5 at the same seed, on the two-reference request the example
builds:

| | `768` (default) | `0` (the clip's own) |
|---|---|---|
| reference clip canvas | 1344 × 768 | 960 × 544 |
| VAE tiles per chunk | 28 | 15 |
| reference rows | 13,120 | 7,144 |
| **conditioning rows** | **3,115** | **2,119** |
| packed sequence | 22,615 | 15,643 |
| wall clock | 23 m 07 s | 14 m 17 s |

**It moves three things, and the third one surprises people.** The VAE encode
and the reference rows are the obvious pair. The third is the *conditioner*:
the vision tower is handed the same normalized pixels and smart-resizes from
them, so this canvas also decides how many vision tokens a clip contributes —
996 fewer here, all of them the clip's (the still is encoded at its own short
edge and does not move). Lowering it is a fidelity decision on **both**
channels, not a free 1.6×.

And on the shipped example it is visibly not free. At `768` the generated cat
wears the reference still's navy jacket and sits in the concert hall the
prompt asks for; at `0`, at the same seed, it wears no jacket and the setting
collapses toward the reference *clip's* dark room. Different sequence shapes
give different samples, so one seed is not proof of a systematic loss — but it
is proof that `0` is a different generation and not a cheaper spelling of the
same one.

So `768` stays the default: it is the released checkpoint's rule, and a
reference the model was trained to see upscaled is what it gives. `0` is worth
*trying* where references are already smaller than 768 and the budget is
tight — and worth **comparing side by side** before you keep it. The area cap
still binds a clip larger than the canvas either way.

**Which key you turn depends on the route.** `reference_video_short_edge` is
the per-kind default, and it is only consulted for a reference whose size the
encoder has to resolve — i.e. one from the `references` LIST. A reference on a
port arrives with `short_edge: 0` already set (that is the port default, and
it is the point of the ports), so the per-kind default is never reached. On
**this** example, which is the ports route throughout, the `0` column above is
what you get by resampling the clip to 960 × 544 in `size-clip`; setting
`reference_video_short_edge` on the encoder would change nothing. The same
goes for `reference_image_short_edge` and the still.

As shipped — 960 × 544, 39 frames, 8 steps, one still and one clip — that is
**23 min 54 s** on the fanless 16 GB M5 (the `references` list, same geometry,
measured 23 min 07 s: the ports chain resizes every decoded frame and the list
resizes only the ones it keeps). Around a third of it happens before
the first denoise step: the 32B conditioner is loaded and streamed, and both
references are read twice, once by the vision tower at its own canvas and once
by the video VAE at MiniMax-H3's. Raising `frames` from here raises the
reference rows with it, so the next size up is a bigger jump than it looks.

#### How a reference is read

A single path may be written bare, without the brackets. **The order is the
request**: it numbers the references in the prompt the model reads and places
them on a shared clock, so reordering the list is a different generation — and
port references continue that numbering after the last file.

**The number in a tag counts within its KIND, not across the list.** Each of
`<Picture i>`, `<Video k>` and `<Audio j>` has a counter of its own, so the
third reference overall can perfectly well be `<Audio 1>` — it is the first
*soundtrack*, whatever sits ahead of it. Two stills and a soundtrack, in that
order, present as:

```
<Picture 1>: … <Picture 2>: … <Audio 1>: … <your prompt text>
```

and a still followed by a clip that carries sound is `<Picture 1>`,
`<Audio 1>`, `<Video 1>` — the clip is reference two and still the first video
and the first audio. Getting this backwards writes a prompt that points at a
reference which is not there, and nothing reports it: an unmatched tag is
ordinary text to the model.

A reference's own block is emitted in list order and the prompt text comes
after all of them, so where you mention a tag inside the prompt is free — the
tags refer back to blocks the model has already read.

That ordering is why the list exists at all rather than a port per reference.
A request's shape is only known when it arrives, and twelve `load-image`
chains cannot express "three clips and nine stills" without the graph being
rewritten per request. The six ports are the other half of the same argument,
not a reversal of it: a *path* cannot name a still that does not exist yet.
Six, and not more, because the numbering has to be static — a port that could
fall silent would renumber every reference behind it.

**A file does not say what it is.** vpipe opens it and reads that from the
bytes: a container with one frame is an *image* reference, an `.mp4` carrying
no video stream is an *audio* one, and an animated `.webp` is a clip. An
extension is a claim and the bytes are the fact — and a file picker hands
over whatever the user chose. Getting it wrong is not loud: a still read as
video conditions the model on a frozen clip, and a clip read as a still
quietly keeps only its first frame. A tensor has no such ambiguity, which is
one reason the ports type by rank: `[1, 3, H, W]` is a one-frame clip and
`[3, H, W]` is a still, and nothing has to be inferred.

A video reference conditions on **its own soundtrack** when it has one, and a
clip and its audio have to leave one container together to stay in sync. The
`references` list gets that by opening the file itself; the ports route gets
it from one `load-video` feeding both its streams. Either way the file's
**frame rate** has to survive the trip: MiniMax-H3 resamples every reference
onto its own 24 fps, so a rate lost on the way in is a generation conditioned
at the wrong speed with nothing to complain about. That is why a clip on a
port must state `fps` and is refused without one.

Set the stage's `frames` to the **same value** as `generate-video`'s: it is
the duration references are truncated to as well as the size of the sequence
the transformer packs. They are checked against each other, not trusted.

References never bind the generated geometry. An image is encoded at a short
edge of its own (`reference_image_short_edge`, 2048), with no area cap and
upscaling included; a clip goes onto the same canvas rule as the target
(`reference_video_short_edge` 768, under `reference_video_max_pixels`
1032192), resolved from *its* aspect ratio. Two references of different shapes
land on different canvases, which is expected.

Those defaults are the released checkpoint's, and all of them are worth knowing
about because none is visible in the output. `reference_image_short_edge` in
particular: at 2048 a single still is 121 VAE tiles, and nine are allowed; at
1024 it is 25. Lower it on any graph that feeds stills through the
`references` list.

Both keys are per-kind defaults, so they apply to the `references` list only —
a reference arriving on a port already carries its own `short_edge` and never
reaches them. The example holds its still to 1024 with an `image-resample`
stage instead, which is the ports route's equivalent and the reason it also
carries a redundant `reference_image_short_edge: 1024`.

You do not have to accept any of it silently. **Every reference the encoder
had to reshape is warned about**, naming what was kept:

```
reference 1 fitted -- rescaled 1920x1080 -> 1344x736 (48% of the pixels),
  resampled 30 -> 24 fps (18 frame(s) dropped),
  truncated 72 -> 39 frames (54% of the clip)
```

Three reductions, counted apart on purpose, because they have different
remedies: the rate resample drops whole frames whether or not the clip is also
too long, `frames` is what fixes the truncation, and the canvas keys are what
fix the rescale. A reference that needed nothing is logged at debug as *taken
as given* — which is the outcome to aim for when you have sized it yourself.

#### References that are not files

Six **tensor iports** (`ref1`..`ref6`) sit after `prompt` and `model`, for the
references a file list cannot name: a still your graph just generated, a
cropped frame, a clip that was never written to disk. They supplement the
`references` list rather than replacing it, and are numbered after it.

| what you send | rank | sideband |
|---|---|---|
| audio | `[N]` or `[channels, N]` f32 | `sr` (or `sample_rate`) — **required** |
| a picture | `[3, H, W]` u8 | — |
| a clip | `[frames, 3, H, W]` u8 | `fps` — **required** |

Rank is what types it, which also settles the one case a container cannot
state: a one-frame clip is `[1, 3, H, W]` and a still is `[3, H, W]`, and those
are different requests. The rates are **required and never defaulted** — a
soundtrack read at the wrong rate conditions on the wrong sound, and a clip at
the wrong speed generates video with nothing to complain about.

**Send audio at the audio VAE's rate: 32000 Hz.** A file reference is
decoded straight onto it, but a *beat* arrives at whatever its producer
chose, so set the producing `audio-to-pcm`'s `output_sample_rate` to
`32000`. Another rate is not silently wrong — the encoder resamples it
onto 32000 and warns — but that is a second pass of the filter over a
waveform its producer already resampled once, and the fix is one config
key. It matters because nothing downstream looks at the rate again: an
*unconformed* 44.1 kHz soundtrack would be encoded as if it were 32 kHz,
i.e. 1.38× too fast, pitched up a fourth, and 1.38× too long against the
clip it shares a rotary clock with — with every shape still valid.

One optional sideband key: `short_edge` sets *this one reference's* canvas.

When the rate is not yours to set — the PCM arrives from a stage you did not
configure, or one feed has to serve both this port and a 44.1 kHz mux — put an
**`audio-temporal-resample`** in front of it with `output_sample_rate: 32000`.
That stage also owns the *speed* and *pitch* of a soundtrack, if the reference
wants stretching before the model hears it.

##### Audio that belongs to a clip

A clip demuxed into frames and PCM arrives on **two** ports and should stay
**one** reference. `attach_audio` on the encoder names the ports whose audio is
a soundtrack rather than a reference of its own:

```json
"attach_audio": [3]
```

It is **positional**: the audio folds onto whichever reference immediately
precedes it. With one clip that is unambiguous; with two, order the ports so
each soundtrack follows its own clip. There is no way to name a target.

It is a list and not a single switch because a request may legitimately carry
both an attached soundtrack and a standalone piece of music, and those are
different references.

**It can attach to a reference from the `references` list.** Files are read
before any port, so the reference it folds onto may be one of them — which
makes `references: ["motion.mp4"]` plus a generated soundtrack on a port a
perfectly good request, and the neatest way to score a clip you already have.

An audio beat's own `attach` sideband overrides the config for that beat, so a
producer that knows better than the graph can say so — `true` to attach where
the config did not ask, `false` to decline where it did.

Attaching to a **still** is allowed and warned about. It is a real request —
a one-frame `.mp4` with an audio track comes through the `references` list the
same way — but it is far more often a port wired in the wrong order, so the run
says so rather than refusing something the other route permits.

A **wired port must beat every request**; send an empty tensor to say "nothing
this time". A port that could fall silent would renumber every reference after
it, and the numbering is the request.

The port default is `short_edge: 0` — **encode it at the size it arrived**.
That is the point of the ports: if you have resampled to 768 with an
`image-resample` stage, or cropped to a framing you chose, re-resolving it
against a per-kind default would scale it straight back up and undo the work
(and cost you a second Lanczos pass). At `0` the encoder may only **reduce**,
never enlarge — it brings a picture under the area cap, floors both axes to
the multiple of 32 the DiT patch and VAE stride demand, and otherwise leaves it
alone. Three things it still cannot skip: that grid (a 1080-tall frame becomes
1056), the 1:4 … 4:1 aspect bound, which is a refusal rather than a fit, and
for clips the 24 fps resample plus the `17n + 5` snap.

Because a short edge of `0` removes the only bound an image reference had, set
**`reference_image_max_pixels`** on a graph that feeds raw pictures. It is
uncapped by default, matching the checkpoint, and an uncapped 4K still is
~220 VAE tiles and 8,160 DiT rows — half the packed sequence of a typical
request, before its vision tokens.

This stage holds the prompt encoder, its vision tower and both VAEs while it
runs, so on a memory-bounded box leave `unload_when_idle` at `auto` — the
encoders are dropped before the denoise starts.

#### Preparing the Ref2VA checkpoint

Run
[`prepare-minimax-h3-ref2va-8bit.vpipeline`](pipelines/prepare-minimax-h3-ref2va-8bit.vpipeline)
exactly as step 1, and it produces `local/MiniMax-H3-Ref2VA-8bit`. Point the
generation pipeline's `model-select` at that instead.
[`…-4bit`](pipelines/prepare-minimax-h3-ref2va-4bit.vpipeline) is the same
job at half the disk, on the trade described under
[Why 8-bit and not 4](#disk-space).

The two partitions **share one repo and one download**. Ref2VA adds only its
own 66 GB transformer; the 51 GB prompt encoder and both VAEs are already on
disk from step 1 and are skipped. If you only ever want Ref2VA, run this
pipeline alone — it fetches what it needs.

One config key makes that sharing safe, and it is in the pipeline:

| key | why |
|---|---|
| `model_variant: ref2va` | *which* of the repo's two models to fetch. The files differ; the repo path does not. A fetch that does not say is refused, with both listed. |

Each partition also carries its own **registration key** —
`Comfy-Org/MiniMax-H3-Ref2VA` here — so the two records coexist over one
directory on disk without one overwriting the other. That is the catalogue's
own name for the entry, so `model_key` only has to be set to override it.

Everything downstream then resolves through that key rather than by
inspecting the directory — which matters because the directory holds **both**
transformers and cannot say which one you meant. Left to guess it picks
FL2VA, and a Ref2VA request would load, run at full 33B cost, and generate
video conditioned on nothing.

#### Ref2VA-like — references on the FL2VA weights

You do not always need the second checkpoint. The **FL2VA** weights will take
reference images too, through Ref2VA's own sequence rather than through
keyframes, and OpenVDN [published that
mode](https://github.com/OpenVDN/vdn-minimax-h3#supporting-ref2va-like-task)
with a rendered example on 2026-09-17. vpipe runs it: wire a
`video-ref-encoder` exactly as above and point `model-select` at an FL2VA
checkpoint instead of a Ref2VA one.

[`minimax-h3-ref2va-like.vpipeline`](pipelines/minimax-h3-ref2va-like.vpipeline)
is that graph, with the [VDN linear branch](#faster-attention--the-vdn-linear-branch)
and the [Turbo LoRA](#fewer-steps--the-turbo-lora) on it — which is the
combination upstream renders, and one this mode is what makes possible at
all: **the VDN branch is only published for FL2VA**, so before it the linear
attention and references could not be had together.

**What it actually is.** Nothing about the weights changes. The references are
packed as Ref2VA packs them — `[text | one block per reference | target audio
| target video]`, one rotary slot each, every one held just short of clean at
its noise-augmentation level — and read by the FL2VA transformer, not by
MiniMax-H3's separate `transformer_ref`. The two partitions ship
byte-identical transformer configs, so the sequence builds either way; what
differs is only which weights read it.

> **Zero-shot, and say so when you report a result.** This is an ability the
> FL2VA checkpoint turns out to have, not a task it was trained for. Ref2VA
> **is** that task, on weights trained for it, and remains the better answer
> when subject fidelity is the point. `generate-video` names the mode in its
> log every run that uses it, because the way this fails is a clip that
> quietly ignores its references and looks entirely ordinary.

**Three differences from a Ref2VA graph**, and only the first needs anything
from you:

- **Images only.** Upstream's mode is reference *images*. A reference clip or
  soundtrack packs and runs — the layout is the same one — but nothing
  published covers it, so vpipe warns. Clips and soundtracks are a trained
  input on the Ref2VA partition.
- **A 768 short edge, not 2048.** That is upstream's recipe and vpipe takes it
  automatically: left unset, `reference_image_short_edge` is **768 on FL2VA**
  and 2048 on Ref2VA. It is a quarter of the tokens per reference, and those
  rows sit in the sequence the DiT re-reads at **every step** — so the
  difference is not a fidelity knob you can leave anywhere safe. Setting the
  key is always taken as said, on either partition.
- **Still no keyframe.** References and a keyframe anchor remain mutually
  exclusive, and that was never about the partition: there is no slot for an
  anchor in the reference layout whichever weights read it.
  `generate-video` says so rather than dropping the anchor quietly.

### Longer clips — one story in four parts

A single generation gets expensive fast as it grows: every frame adds rows
to the packed sequence, and attention pays for rows squared. Four ordinary
10-second runs are the cheaper way to a **40-second** clip. Each run after
the first hands the **tail of the one before** to Ref2VA as a clip to
continue from, so the story is carried forward by the picture and the sound
rather than by the prompt alone:

- **[`minimax-h3-extend-part1.vpipeline`](pipelines/minimax-h3-extend-part1.vpipeline)**
  — text to video and audio on the **FL2VA** checkpoint, 243 frames
  (10.125 s) at 960 × 576. It writes `minimax-h3-extend-part1.mp4`.
- **[`…-part2`](pipelines/minimax-h3-extend-part2.vpipeline)** /
  **[`…-part3`](pipelines/minimax-h3-extend-part3.vpipeline)** /
  **[`…-part4`](pipelines/minimax-h3-extend-part4.vpipeline)**
  — **Ref2VA**, each conditioned on the last 3.75 s (90 frames) of the part
  before it, picture and sound. 243 frames each.
- **[`…-concat`](pipelines/minimax-h3-extend-concat.vpipeline)** — joins the
  four into one file, in vpipe rather than by hand. See
  [Joining the parts](#joining-the-parts).

Every continuation is the same graph with a different prompt and a different
file to read, so a fifth is a copy of the fourth.

**The shipped graphs run the released weights at 21 steps, with no
adapter.** Every part names the publisher's own checkpoint (see
[The released weights, either partition](#the-released-weights-either-partition)):

| | checkpoint | steps | shifts |
|---|---|---|---|
| part 1 | `MiniMaxAI/MiniMax-H3-FL2VA` | 21 | 12 / 3 |
| parts 2–4 | `MiniMaxAI/MiniMax-H3-Ref2VA` | 21 | 12 / 3 |

That is a choice for quality over time, and a chain is where it pays. A
Turbo adapter buys its step count with some of what the base model knows
about composition — a part can come back with the subject badly placed, a
hand wrong, or a beat the prompt asked for simply missing — and in a chain
every later part continues from whatever the one before it got wrong.
The graphs also leave `sol_attn` off; `i8_gemm` stays on in every part.

**To run the chain faster**, put back what the shipped graphs leave out:
the prepared 8-bit repacks (`local/MiniMax-H3-FL2VA-8bit` and
`local/MiniMax-H3-Ref2VA-8bit` on `model-select`), `sol_attn: true` on each
`generate-video`, and one Turbo adapter per partition on
`minimax-h3-model-config` — each partition takes ITS OWN, since an adapter
is distilled for one task and the catalogue's parent link refuses the other
partition's (see [Which Turbo adapters work](#which-turbo-adapters-work)):

| | `lora` | `steps` | `video_shift` |
|---|---|---|---|
| part 1 | `larryvrh/MiniMax-H3-Turbo-Lora-v4-600-ema` | 8 | 12 |
| parts 2–4 | `lightx2v/Minimax-h3-Turbo-ref2va-8step-768p` | 8 | **6.0** |

The Ref2VA adapter is the only one here that needs `video_shift: 6.0`.

When a part comes back with a composition you do not want, try a
**different `seed`** before rewriting the prompt: it is the cheaper change
and often enough. Nothing else in the graph changes, and the parts after it
still read only its tail.

The shipped story is *the clockmaker's songbird*:

| part | what happens | ends on |
|---|---|---|
| 1 | In a candlelit workshop walled with clocks, an old clockmaker winds a brass songbird, murmurs *"Just one more turn,"* and raises it on her finger. | her holding still, the bird motionless |
| 2 | The bird wakes, sings, and flies a circle as every clock chimes; she laughs *"You remembered the song!"* It settles on the frosted windowsill. | the bird still on the sill, watching the snow |
| 3 | She unlatches the window and lifts the sash; snow blows in, the bird hops onto the open frame and looks back. *"Go on, then. It's your sky."* | the bird poised on the frame, wings half-raised |
| 4 | It springs into the night, climbs past the eaves, and circles once as she watches from the lit window. *"Goodnight, little one."* | — |

#### Running the chain

You need **both** partitions of the released weights: fetch
`MiniMaxAI/MiniMax-H3` twice, with `model_variant: fl2va` and `ref2va` (see
[The released weights, either partition](#the-released-weights-either-partition)).
To run on the repacks prepared in step 1 and
[Preparing the Ref2VA checkpoint](#preparing-the-ref2va-checkpoint) instead,
change each part's `model-select` as described above. Run the parts in
order, from the same work directory — each part opens the previous part's
`.mp4` by its relative name:

```sh
cd ~/vpipe-work                                    # the work directory again
cp ~/src/vpipe/docs/pipelines/minimax-h3-extend-part*.vpipeline .
for n in 1 2 3 4; do
  ~/src/vpipe/build/apps/vpipe/vpipe \
    --launch minimax-h3-extend-part$n.vpipeline || break
done
```

Run them **one at a time**, never together: each holds a 33B transformer
(see [Memory](#memory)). The `|| break` matters — a part whose input is
missing produces nothing, and the next part would then condition on a stale
file rather than fail.

The four part files are **FFV1**, which QuickTime and most browsers do not
play; they are for the chain to read, not for watching. Watch the joined
clip, or open a part in `ffplay` or VLC.

#### How a part takes its guide

```
load-video ─> video-to-rgb(u8) ─> temporal-slice(start −90)
                                ─> temporal-stack ─────────────> ref1
load-audio(start_s 6.375, duration_s 3.75) ─> audio-to-pcm(32000, stereo)
                                           ─> temporal-stack ────────────> ref2
```

**The guide reaches the model exactly as it was decoded.** The slice feeds
the stack directly — no resample and no levels adjustment in between. What
the previous part produced is what the next part continues from.

**The prompt decides where a continuation picks up.** It is meant to carry
on from the guide's final frame, and with the shipped prompts every part
does: the parts join without a jump.

Write your own continuation prompts the way these are written — name the
subjects that carry over and say they are preserved, open the summary with
`[video continuation + audio reference]`, and give the shot an explicit
timeline of what happens and when. If a join does jump, **rewrite the
prompt** rather than correcting the picture: a resample or a brightness
lift fitted to one join will not transfer to the next.

**The parts are written lossless, because the next part reads them.** Each
part's `rgb-to-video` emits `yuv444p` at `full` range, tagged `bt709`, and
its `save-video` encodes that with `video_codec: ffv1` and
`audio_codec: alac`. So the guide a continuation reads is the decoded picture
itself — no chroma halved to 4:2:0, no lossy quantization, all 256 code
values rather than 219 — and nothing compounds down the chain however many
parts it has. The only lossy encode is the join at the end. A part is about
80–93 MB for its 10 s at 960 × 576, and the `video_bitrate` its sink carries
is inert under FFV1.

**Colour range is carried, not assumed.** `save-video` tags the stream it
writes and `video-to-rgb` reads that tag back, so a clip that is decoded,
conditioned on and re-encoded keeps its contrast. That matters here
because a continuation re-reads its own
output once per part, and an untagged file read by the wrong convention
loses 219/255 of its contrast on every hop.

**The frame cut is `temporal-slice`, not a seek.** `load-video`'s `start_s`
lands on the keyframe at or before the time asked for, which in a 10-second
file may be the first frame. A negative `start` holds exactly the last 90
decoded frames until the stream ends, so the cut is exact to the frame and
costs one decode of a short file. Audio packets decode independently, so
`load-audio` can take its window by time. That window is
`243/24 − 90/24 = 6.375 s` onward, the same 3.75 s the frames cover.

**`attach_audio: [2]`** folds that soundtrack onto the clip before it, so the
model reads one `<Video 1>` with its `<Audio 1>` rather than two unrelated
references (see [Audio that belongs to a clip](#audio-that-belongs-to-a-clip)).

**Why 90 frames.** The encoder snaps a reference clip down to a whole
`17n + 5` frames **from the start**, so a guide of any other length loses its
LAST frames — the very moment the continuation is supposed to pick up from.
Two seconds at 24 fps is 48, which it cuts to 39. 90 is `17 × 5 + 5`, so
nothing is dropped, and at 3.75 s it carries more of the motion across the
seam than the shortest clean guide, 56 frames (2.33 s). It encodes to 27
latent frames of 540 cells: **14,580** reference video rows. MiniMax
documents **2 s** as the shortest reference clip.

**`reference_video_short_edge: 0`** keeps the guide on its own 960 × 576
canvas, the output's own size, instead of upscaling it to 1344 × 768. That
keeps the reference at 14,580 rows rather than roughly twice that. It is the
trade described under [What it costs](#what-it-costs), and it has not been
compared side by side on this story.

**`max_prompt_tokens`** on `video-ref-encoder` is raised from its default of
16,384 — to **65,536** on part 2 and **131,072** on parts 3 and 4. It is the
ceiling on the conditioner's sequence: the prompt plus the vision tower's
reading of the guide, which the encoder refuses outright rather than
truncates when it runs longer. Its KV is paged and costs ~200 KB per token
actually encoded, so a high ceiling is not memory held up front. Raise it
when the encoder reports a presentation longer than its token pool.

**`unload_when_idle: destroy`** on both the reference encoder and
`generate-video` hands each model's memory back as soon as its phase of the
part is over.

#### Writing the prompts

The prompts follow MiniMax's own prompt guides, which describe the format
the model was trained on. This is not decoration: H3's encoder reads the
prompt verbatim, with no rewriting step in between.

- **Part 1** uses the three fields of the
  [base guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md):
  `integrated_multimodal_description`, `overall_soundscape` and
  `non_diegetic_music`. Shots are marked `[Shot N] At MM:SS.mmm`, the
  speaker has a stable ID, and her line is written as
  `<d>[English] Just one more turn.</d>`.
- **Parts 2 to 4** use the six sections of the
  [reference guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md).
  `subject_definitions` names the clockmaker, the bird and the workshop as
  `<Subject 1..3>` of `<Video 1>`, with `<Audio 1>` as its soundtrack. The
  `summary` opens with **`[video continuation + audio reference]`**, the task
  type that says the clip is to be continued, and its sound followed without
  being copied. `retention_analysis` and a shot-by-shot
  `detailed_description` follow.

**Every part but the last ends on a held shot, and the next one opens on
it.** Part 1 closes on a static medium close-up from 6.0 s in which nothing
moves but the candle flames and the pendulums; part 2 opens on that framing
and holds it two seconds before the bird wakes; part 2 in turn ends on the
bird motionless on the sill, and so on down the chain. Every guide ENDS
inside one of those holds, so every seam falls where the picture is still.
A 3.75 s guide does not always START inside one: parts 2 and 3 cut to their
final shot at 7.5 s and 7.0 s, so the guides parts 3 and 4 read open on the
last moments of the shot before — the cut is part of what the model sees.
Plan the story that way from the start: the beats worth generating are the
ones **between** the seams, and a hold at least as long as the guide keeps
the whole guide on it.

**Plant what the next part will act on.** Part 3 opens the window, so part
2's prompt gives the frame a small brass catch that can lift the sash, keeps
the window shut through its own shot, and lands the bird just beside the
catch. The guide part 3 reads then shows the thing it has to use; a detail
that first appears in the part that needs it is one the model has to invent
on the spot.

**Carry the subject definitions over word for word.** Parts 2 to 4 describe
the woman, the bird and the room in identical words, because the clip shows
them and the text names them — any drift in the wording is drift the model
is free to apply to the picture. Some drift arrives anyway; see below.

**Ref2VA continues AFTER the clip; it does not replay it.** Part 2 opens
on the moment after part 1's **last** frame, not on the first frame of the
guide window. That is continuation the model learned from the task type,
not a pinned frame. The
[caveat above](#conditioning-on-references-ref2va) still stands, which is
one more reason to meet on a moment that does not move.

#### Joining the parts

**The parts are CONSECUTIVE, so they are joined end to end and nothing is
trimmed.** A continuation resumes *after* its guide instead of re-rendering
it, so part 2's first frame follows part 1's **last** frame — not the frame
3.75 s earlier where the guide was cut from. There is no overlapping
material, so there is nothing to cut inside.

**In vpipe**, that is
[`minimax-h3-extend-concat.vpipeline`](pipelines/minimax-h3-extend-concat.vpipeline):
one `load-video` handed all four files, which it joins in the order given.

```json
"config": {
  "input_url": [
    "minimax-h3-extend-part1.mp4",
    "minimax-h3-extend-part2.mp4",
    "minimax-h3-extend-part3.mp4",
    "minimax-h3-extend-part4.mp4"
  ],
  "enable_video": true,
  "enable_audio": true,
  "options": {
    "safe": "0"
  }
}
```

**In the web UI this is a file picker.** Open the pipeline in the editor,
press Browse on `input_url`, and select the four parts in one dialog — they
land in the array in the order picked. There is no list file to write and no
demuxer to name.

> **Do not trim the parts as you join them.** Cutting each one where its
> guide starts looks natural — it is the moment the next part was
> conditioned on — but it **skips the rendered frames between that point
> and the part's end**, 3.75 s of story per seam with the shipped guide, and
> that skip is a visible jump.

```sh
vpipe --launch minimax-h3-extend-concat.vpipeline
```

The graph is the ordinary file chain — `load-video → video-to-rgb →
rgb-to-video → save-video`, with the soundtrack through `audio-to-pcm` — so
the join is one re-encode at whatever `video_bitrate` the sink is set to,
and picture and sound stay locked together across every boundary. It is the
chain's one lossy step: the sink's defaults, H.264 at 4:2:0 and limited
range with AAC sound, which is what every player expects of a delivered
file.
`start_s` / `duration_s` address the joined timeline, not any one part.

> **When you still want a list file.** An array joins whole clips, which is
> the case here. To trim each clip *as* it joins — `inpoint` / `outpoint`
> per entry — write the list yourself and name the demuxer with
> `format: "concat"`, which then also wants `options: {"safe": "0"}` before
> it will follow absolute paths. Setting `format` alongside an array is an
> error rather than one of them quietly winning.

The result is **40.5 s** — all four parts whole, 972 frames.

**What it costs.** Each part's held tail and the next part's held opening
are both kept, so a seam sits on its static beat for both of them rather
than for one. That longer dwell is the price of losing nothing, and it is
usually the right trade: a pause reads as deliberate, a skip reads as a
glitch. If one seam does stall, trim a *few* frames there — never the whole
guide, which puts the skip back.

**By hand**, if a seam wants a cross-fade rather than a cut — the graph has
no stage for one — fade over five frames at each junction. With whole parts
of 10.125 s, each `offset` is the timeline so far less the fade:

```sh
ffmpeg -i minimax-h3-extend-part1.mp4 -i minimax-h3-extend-part2.mp4 \
       -i minimax-h3-extend-part3.mp4 -i minimax-h3-extend-part4.mp4 \
  -filter_complex "\
[0:v]setpts=PTS-STARTPTS[v0];[1:v]setpts=PTS-STARTPTS[v1];\
[2:v]setpts=PTS-STARTPTS[v2];[3:v]setpts=PTS-STARTPTS[v3];\
[v0][v1]xfade=transition=fade:duration=0.208:offset=9.917[a01];\
[a01][v2]xfade=transition=fade:duration=0.208:offset=19.834[a02];\
[a02][v3]xfade=transition=fade:duration=0.208:offset=29.751,format=yuv420p[v];\
[0:a]asetpts=PTS-STARTPTS[t0];[1:a]asetpts=PTS-STARTPTS[t1];\
[2:a]asetpts=PTS-STARTPTS[t2];[3:a]asetpts=PTS-STARTPTS[t3];\
[t0][t1]acrossfade=d=0.208[s01];[s01][t2]acrossfade=d=0.208[s02];\
[s02][t3]acrossfade=d=0.208[a]" \
  -map "[v]" -map "[a]" -c:v libx264 -crf 18 -c:a aac -b:a 192k \
  minimax-h3-extend.mp4
```

A cross-fade consumes the frames it blends, so this lands at **39.9 s**
rather than 40.5. It is optional.

#### Making it your own

- **Keep each part's two `frames` equal**, on its `generate-video` and its
  `video-ref-encoder`; they are checked against each other. The parts do not
  have to be the same length as each other.
- **Change a part's length and the next part's audio window moves.**
  `start_s` is `previous_part_frames / 24 − 90 / 24`. Leave `duration_s` at
  3.75 as long as the slice stays at 90.
- **A guide is `17n + 5` frames**: 56 (2.33 s), 73 (3.04 s) or the shipped
  90 (3.75 s). Change `temporal-slice`'s `start`, and the audio window with
  it. A longer guide carries more motion and costs about 540 rows per extra
  latent frame at 960 × 576.
- **Keep the parts lossless.** Switch a part's sink back to H.264 and every
  continuation after it conditions on a compressed, 4:2:0 copy of the one
  before — a loss that compounds once per part.
- **Expect identity to drift along the chain**, and write against it. Each
  part sees only the 3.75 s before it, so a detail the guide does not show
  is carried by the prompt alone. Keep the descriptions identical between
  parts, keep distinguishing features in frame near the seams, and check the
  last part against the first rather than against the one before it.
- **For a fifth part**, copy part 4, point its two `load-guide*` stages at
  `…-part4.mp4`, give the new part its own `output_url`, and add it to the
  concat list.
- **Adding a Turbo adapter changes the step count and the shifts with
  it.** Each one is distilled at a recipe; the table above is what that
  pair wants, and
  [Which Turbo adapters work](#which-turbo-adapters-work) lists the rest
  with theirs. An adapter for the other partition is refused, not applied.

### The released weights, either partition

Both partitions are also catalogued from **`MiniMaxAI/MiniMax-H3`**, the
publisher's own diffusers checkout, where each lives in a complete pipeline
of its own — `FL2VA/` and `Ref2VA/`, transformer and prompt encoder and both
VAEs under each. Select one the same way, with `model_variant: fl2va` or
`ref2va`; that repo publishes two models now, so a fetch of it must say which.

It is the larger download: about **134 GB per partition**, against ~115 GB
for the whole repack. The encoder and both VAEs are *repeated* under each
partition rather than shared, so wanting both partitions from here costs two
copies of them — and its encoder ships all 64 layers where the repack's is
truncated at the tap this model actually reads. What it buys is the
reference: these are the weights the repack was converted from, and the two
group the transformer's fused qkv projection differently — a difference with no signature in the
tensor names or shapes, so having both on disk turns "is our loader right?"
into a diff.

Which partition a directory holds is read from that partition's own
`model_index.json`, and the model's registration key is what says which one
you asked for.

### Fewer steps — the Turbo LoRA

Steps are the bulk of what this model costs, and good quality from the raw
model takes 16 of them.
The community [Turbo LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)
distils that down: usable video **and** synchronized audio at **4 steps**, and
better still at 6–8. Past 8 it stops helping and starts to over-sharpen, so
4–8 is the range. Keep `scale` at **1.0** — the adapter is tuned for it.

Most of these adapt the **FL2VA** partition (text-to-video and
image-to-video), and lightx2v's line also covers **Ref2VA**. Eight adapters are
catalogued rather than one — twelve entries, because lightx2v publishes four of
them twice — since the choices between them are real rather than version
bumps: a different training resolution, a different sigma grid, a different
partition, a different decomposition of the same file.

| entry | partition | steps | shifts (v/a) | trained at | when |
|---|---|---|---|---|---|
| `Turbo few-step v4-600 EMA` (larryvrh) | FL2VA | 4–8 | 12 / 3 | — | the default. Better static and small-motion shots, better micro-detail. |
| `Turbo few-step v1-850 EMA` (larryvrh) | FL2VA | 4 | 12 / 3 | — | only for **4 steps with large, fast motion**, where v4 can trail or smear. At 6–8 steps prefer v4. |
| `Turbo 4-step v1.0 768p` (lightx2v) | FL2VA | 4 | **6** / 3 | 1344×768 | a second distillation, at 768p. **Set `video_shift: 6.0`** — see below. |
| `Turbo 4-step v1.2 768p` (lightx2v) | FL2VA | 4 | **6** / 3 | 1344×768 | the newest of that line, and the one to try first at 4 steps. Published **split** and fused; take the split. |
| `Turbo 8-step v1.0 544p` (lightx2v) | FL2VA | 8 or 4 | 12 / 3 | 544p | the 8-step of that line, on the checkpoint's own shifts. |
| `Turbo 8-step v1.0 768p` (lightx2v) | FL2VA | 8 | **6** / 3 | 1344×768 | upstream's own default — LightX2V Studio serves this one. Also published **split**, which is the copy to prefer; see below. |
| `Turbo 4-step v0.1` (lightx2v) | **Ref2VA** | 4 | 12 / 3 | 544p | the cheaper of the two for the reference partition, and the one on the checkpoint's own shifts. Also published **split**. |
| `Turbo 8-step v1.0 768p` (lightx2v) | **Ref2VA** | 8 | **6** / 3 | 1344×768 | the other one, and the only Ref2VA adapter that needs `video_shift: 6.0`. Published **split** and fused; take the split. |

> **The version numbers are upstream's and are not documented.** lightx2v's
> model card describes only the 8-step v1.0 it deploys, so what separates
> v1.0, v1.1 and v1.2 of the 4-step 768p line is not stated anywhere. Of the
> two newer ones only **v1.2** is catalogued — a version number is for taking
> the latest — but that means "newest", not "measured better here". Compare it
> against v1.0 on a seed you know before switching a habit to it.
>
> The same goes for what the catalogue *records*. For **v1.2** and for the
> **Ref2VA 8-step 768p**, every field is read off the filename (partition,
> steps, resolution, dtype) except the shift, which is inherited from the rest
> of the 768p line. That inheritance is the assumption worth re-checking if
> upstream ever documents these: a wrong shift is a wrong sigma grid, and
> nothing reports it.

**The shifts are part of the adapter, not a preference.** lightx2v's 768p
checkpoints were distilled on a video shift of **6** where this model's
default — and every adapter here not trained at 768p — is 12. A distillation is fit to the
sigma grid it was trained on, so running it on the wrong one is not a style
difference; it is a different schedule, and nothing will report it.
`video_shift` lives on the same `minimax-h3-model-config` stage as `lora`,
which is exactly why the two travel in one beat.

**`-split` is the diffusers copy, and it is the better file.** lightx2v
publishes each adapter twice — once fused for ComfyUI, once as the original
split `to_q`/`to_k`/`to_v` export — and vpipe loads both. Prefer the split
one: see [Which Turbo adapters work](#which-turbo-adapters-work) for why a
published fusion is silently wrong on one of the two weight releases and a
split file is right on both.

#### Get it

One `model-fetch` stage —
[`prepare-minimax-h3-turbo-lora.vpipeline`](pipelines/prepare-minimax-h3-turbo-lora.vpipeline):

```sh
vpipe --launch docs/pipelines/prepare-minimax-h3-turbo-lora.vpipeline
```

~744 MB, about ten seconds against the hours step 1 costs — a LoRA is used as
it ships, so there is nothing to quantize afterwards.

lightx2v's line is
[`prepare-minimax-h3-turbo-lora-lightx2v.vpipeline`](pipelines/prepare-minimax-h3-turbo-lora-lightx2v.vpipeline),
same shape, pinning `lightx2v/Minimax-h3-Turbo-4step-768p`. Both of that
repo's spellings are catalogued — it publishes each adapter twice, fused for
ComfyUI and split for diffusers, and vpipe loads either. The full set of
`model_variant` values:

| variant | what |
|---|---|
| `lightx2v/Minimax-h3-Turbo-4step-768p` | FL2VA 4-step, shift 6 |
| `lightx2v/Minimax-h3-Turbo-8step` | FL2VA 8-step, 544p, shift 12 |
| `lightx2v/Minimax-h3-Turbo-8step-768p` | FL2VA 8-step, shift 6 |
| `lightx2v/Minimax-h3-Turbo-8step-768p-split` | the same, **split** — prefer this |
| `lightx2v/Minimax-h3-Turbo-4step-768p-v1.2` | FL2VA 4-step **v1.2**, shift 6, **split** — prefer this |
| `lightx2v/Minimax-h3-Turbo-4step-768p-v1.2-comfyui` | the same, fused |
| `lightx2v/Minimax-h3-Turbo-ref2va-4step` | Ref2VA 4-step, 544p, shift 12 |
| `lightx2v/Minimax-h3-Turbo-ref2va-4step-split` | the same, **split** — prefer this |
| `lightx2v/Minimax-h3-Turbo-ref2va-8step-768p` | Ref2VA 8-step, shift 6, **split** — prefer this |
| `lightx2v/Minimax-h3-Turbo-ref2va-8step-768p-comfyui` | the same, fused |

`model_variant` is not optional here and its value is the catalogue **name**,
not a word from the title. Every Turbo checkpoint of a repo is published from
that one repo, so a bare `"turbo"` matches several and the fetch is refused
with the candidates listed rather than quietly taking the first. For larryvrh
that means swapping in `larryvrh/MiniMax-H3-Turbo-Lora-v1-850-ema` for the
other one; they share a directory on disk and register under separate keys.

The fetch **registers** the adapter under its catalogue name — an `owner/name`
key like `larryvrh/MiniMax-H3-Turbo-Lora-v4-600-ema` — and that key is what the
run pipeline names, not a path. It is also what the config stage's **Browse**
button offers, so the usual flow is: run the prepare pipeline once, then pick
the adapter from the list. `lora` takes a registered model, a
directory holding one `.safetensors`, or a direct path, in that order of
preference. Prefer the key: both Turbo checkpoints of a repo land in one
directory, so a key is the only form that says which of them you meant.

#### Run it

[`minimax-h3-text-to-video-turbo.vpipeline`](pipelines/minimax-h3-text-to-video-turbo.vpipeline)
is the step-2 graph with two edits that matter: the adapter named on the
config stage, and `steps` down from 8 to **6**.

```sh
vpipe --launch docs/pipelines/minimax-h3-text-to-video-turbo.vpipeline
```

It still needs the base model from step 1 (`local/MiniMax-H3-FL2VA-8bit`) —
the adapter replaces the step count, not the checkpoint. Everything else in
the graph is untouched, which is the point: applying a LoRA is a config edit
on one stage, not a different pipeline.

**What it costs.** 960 × 544 · 24 fps · **124 frames** (5.17 s) · 6 steps,
end to end from `vpipe --launch` to the muxed mp4. The pipeline asks for 120
and the stage rounds **up to 124** — the video VAE takes 17-frame clips and
keeps 5 latents from each, so only 17n+5 has a latent form, and it says so in
the log.

| machine | |
|---|---|
| **M4 Pro** Mac mini, 64 GB, models on an external Thunderbolt SSD | **21 min 44 s** |
| **M5** MacBook Air 15", 16 GB, fanless, on an ice pack | **11 min 25 s** |
| **M5 Pro** MacBook Pro 16", 24 GB, its own fans, no cooling aid | **5 min 0 s** |

All three rows are the same pipeline file, so they are directly comparable:
the fanless M5 finishes **1.9× faster** than the fan-cooled M4 Pro, and does
it on a quarter of the RAM. On 16 GB the DiT streams its weights, which is
what keeps that machine from going faster still. The M5 Pro is **2.3× faster
again** — 4.3× the M4 Pro — with more RAM and, measurably, a chassis that
holds its clocks.

Each row is that chassis at its best, and for the Air that still is not
very good for long. It starts at the full **1578 MHz** and holds it about
**two minutes**, then throttles to a fluctuation around **1300 MHz** — 82%
of the part — because an ice pack is a heatsink that warms up, not stable
cooling. An 11-minute run therefore spends **~82% of itself throttled**,
which is why the number to distrust on a fanless Mac is a short benchmark:
it can finish before the machine slows down. The **same run on a desk takes
about 15 minutes**, so cooling is worth roughly a **quarter** of the wall
clock here, and is the first thing to check before reading anything else
into a timing.

It is a **noisy** row for the same reason. The ice pack is placed by hand,
and where it sits moves both the length of the boost window and the clock
after it, so the Air figure is one sample rather than a repeatable number.
Treat it as the order of magnitude, and compare against it accordingly.

The M5 Pro row is the machine as it ships, with nothing under it: its fans
are enough that this workload **pins the GPU at 1620 MHz, its maximum, at
100% for the whole run** — so unlike the Air it is repeatable, and it is the
row to quote when a number has to hold up. The two M5 rows are therefore not
the same silicon running at the same rate: one is held at its ceiling, the
other spends most of its run **18% under its own**. That is about **1.25×**
of the 2.3× between them, leaving roughly **1.8×** to core count and memory
— worth separating, because only the 1.25× is something better cooling could
recover.

Like every other pipeline here this one sets `i8_gemm` (see [the settings
worth knowing](#the-settings-worth-knowing)). It does nothing on the M4 Pro,
and an M5 run without it will be slower than 11 min 25 s. It works with the
adapter, but it changes the picture by about as much as the adapter does, so
turn one at a time when you are judging output rather than speed.

There are two ways to apply it, and for this adapter they are not
equivalent.

#### Runtime (recommended)

Name it on the `minimax-h3-model-config` stage and the DiT applies it as it
runs — every adapted projection computes `W x + scale * B (A x)`:

```json
{
  "id": "h3-config", "type": "minimax-h3-model-config", "iports": [],
  "config": {
    "lora": "larryvrh/MiniMax-H3-Turbo-Lora-v4-600-ema",
    "lora_scale": 1.0
  }
}
```

Wire its output to `generate-video`'s `model_config` iport, set `steps` to 4–8,
and that is the whole change. Nothing is written to disk and the base
checkpoint is untouched, so switching adapters — or turning one off — is a
config edit rather than a 66 GB pass.

Applying it costs a few percent of a step and one small scratch buffer, so
what you save in steps you keep.

**`lora_scale` is live.** It rides a GEMM constant as a per-forward value
rather than being folded into the factors, so a trigger-driven config
stage can sweep it across beats and each change costs a setter, not a reload
of 33B of weights. `0` skips the adapter's two GEMMs entirely, so *off* is
exactly off and an A/B against the un-adapted model is one config edit. The
adapter's own `alpha/rank` (kohya-convention files carry it; these two do not)
is a property of the FILE and is folded in once at load, so the two never get
confused. Upstream tunes for `1.0`: nudge up (~1.05–1.2) for blurry ghosting,
down (~0.8–0.95) for over-sharp grain.

The `lora` path itself is a **load-time** argument, and the asymmetry with
`lora_scale` is real rather than an oversight: an adapted `mlp.fc1` changes
which kernels the blocks are built with, so it cannot be swapped under a
running DiT. A beat that changes the adapter after the DiT is built is
reported and ignored rather than silently applied to the next clip; one that
changes only the strength is applied.

#### Two at once

There are **two slots**. `lora2`, `lora2_scale` and `lora2_qkv_layout` are the
second one, and they work exactly as the first three do:

```json
{
  "id": "h3-config", "type": "minimax-h3-model-config", "iports": [],
  "config": {
    "lora":  "larryvrh/MiniMax-H3-Turbo-Lora-v4-600-ema",
    "lora_scale": 1.0,
    "lora2": "my-style-adapter",
    "lora2_scale": 0.7
  }
}
```

Every adapted projection then computes `W x + s1 B1 (A1 x) + s2 B2 (A2 x)`.
The pairing this exists for is a **few-step distillation** in the first slot —
a correction to the whole model, belonging at the strength it was trained at —
and a **style or identity adapter** in the second, which is the one you
actually dial. Nothing in the model distinguishes the slots though: either may
hold either, and `lora2` alone (no `lora`) is a perfectly ordinary request.

**The strengths are independent, and both stay live.** That is the reason the
two are separate slots rather than one merged set of factors. Merging is exact
arithmetic — stack the factors on the rank axis, `A = [s1 A1; s2 A2]` and
`B = [B1 | B2]`, and one adapter of rank `r1 + r2` computes the same sum for
free — but it folds both strengths into `A`, which turns the live knob back
into a rebuild for the adapter most likely to be swept.

What a second adapter costs is **its own pair of skinny GEMMs**, about 1.5% of
a projection at rank 64. The base weight is still read once, which is where the
time actually goes. One caveat if you compare runs: the second adapter's
delta accumulates onto the output in a separate bf16 pass, so *the same*
adapter split across both slots at half strength is not bit-identical to one
slot at full — **measured at 5.7e-3 of the output, flat across strengths**,
which is one and a half bf16 ULPs and not a scale-dependent error. Two
*different* adapters pay the same and have nothing to be compared against.

`lora2` is load-time exactly as `lora` is, and `lora2_qkv_layout` is per slot
because the fused-`qkv` row order is a property of the *file*: two adapters
from different publishers need different answers.

#### Merging, and why it loses most of this adapter

`lora-fuse` writes a new checkpoint with the delta folded in. That is the right
tool for a *stylistic* LoRA, and the wrong one here. **Measured** on the Turbo
adapter against its bf16 base:

| tensor | intended \|dW\|/\|W\| | survived the merge | elements changed |
|---|---|---|---|
| `blocks.7.mlp.fc1` | 2.26e-4 | 46% | 5.8% |
| `blocks.23.attn.qkv_proj` | 2.65e-4 | 51% | 6.4% |
| `blocks.40.adaln_proj.linear` | 3.87e-4 | 78% | 13.8% |

The update is 2–4e-4 relative to the weights; bf16's step is ~4e-3 relative.
For **94% of elements `W + dW` rounds straight back to `W`** — the correction
is an order of magnitude below the storage resolution. A quantized base —
8-bit here, 4-bit if you chose it — is coarser again. Upstream says the same thing in
passing: its ComfyUI node applies the LoRA at run time by default and calls
merging "a bit softer".

If you do want a merged checkpoint anyway:

```json
{
  "id": "fuse", "type": "lora-fuse", "iports": [],
  "config": {
    "base_model": "<models>/Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_bf16.safetensors",
    "lora": "larryvrh/MiniMax-H3-Turbo-Lora-v4-600-ema",
    "output_name": "MiniMax-H3-FL2VA-Turbo-bf16",
    "scale": 1.0
  }
}
```

`lora-fuse`'s `scale` is the same strength knob, applied once when the delta
is written. It is not live — changing it means another 66 GB pass — which is
the other reason to prefer the runtime path while you are still choosing a
number.

**`base_model` is the DiT FILE, not its directory.** A Comfy-Org repack is one
file per component and that repo's `diffusion_models/` holds *both* task
partitions at 66 GB each; naming the directory would merge two models under
one set of tensor names. Naming the file also preserves the partition — the
fused output is a directory of shards, so `fl2va` survives only because the
fuse lifts it out of the source filename and writes it into the output's
`config.json`, alongside the `qkv_per_head` flag that records Comfy-Org's flat
qkv grouping. Neither is visible in the tensors. Budget the disk: it reads
66 GB and writes 62 GB, in about nine minutes on an SSD.

#### Which Turbo adapters work

Both paths key on the model's own module names, and tolerate a
`diffusion_model.` container prefix (the ComfyUI convention) on top of them —
or kohya's flattened spelling of them, which is what community LoRAs come in
(see [Community LoRAs](#community-loras--civitai-musubi-tuner-ai-toolkit)).
Measured against the FL2VA base:

| adapter | modules | works |
|---|---|---|
| `larryvrh/MiniMax-H3-Turbo-Lora` | 259 (adds `adaln_proj`, `final_layer`) | yes, both paths |
| `lightx2v/Minimax-h3-Turbo`, the `_comfyui_` files | 208 | yes, both paths |
| `lightx2v/Minimax-h3-Turbo`, the diffusers files | 312 | yes, both paths — **and these are the ones to hold** |

The diffusers spelling is not a naming difference but a different
decomposition: separate `to_q`/`to_k`/`to_v` adapters stacked
block-diagonally into the fused `qkv_proj`, and an `ff.net.0.proj` in
diffusers' **value-first** order whose halves are swapped into this model's
gate-first `mlp.fc1`. vpipe does both transforms at load, verified
tensor-for-tensor against upstream's own ComfyUI conversion of the same
adapter: `A` concatenates on the rank axis exactly, `B` is exactly
block-diagonal, `fc1`'s halves are exactly swapped, `out_proj` and `fc2` are
byte-identical, and alpha 8 over rank 128 is the same strength as the fused
24 over 384.

**Prefer the diffusers copy**, which is why both are catalogued. A *published*
fusion is built for one qkv column grouping — flat, in every case that
exists — so it is silently wrong on the other publisher's weights. Fusing from
the split file happens against the DiT actually loaded, so it is right on
both. It is also the smaller download, 1.38 GB against 1.96, the difference
being the two thirds of a block-diagonal `B` that is zero — and the smaller
model, for the same reason: split q/k/v are fused *banded*, each output row
keeping only its own part's factors, where a published fusion arrives with
the zeros built in and holds them in RAM too.

lightx2v's `_comfyui_` `qkv_proj` is rank 384 — three rank-128 adapters
stacked — and that stacking is how you can tell which base it assumes: its
`B` is block-diagonal in the `[all q | all k | all v]` sense, so it targets
the Comfy-Org flat grouping, not the per-head release its `base_model` tag
names.

**On the released MiniMaxAI weights a fused adapter would be wrong.** It is
trained against Comfy-Org's repack, so its `attn.qkv_proj` delta assumes the
flat grouping. Applied to the per-head release it would add one head's `q`
delta onto another head's `k`, in all 50 blocks, with nothing to report. This
is exactly the case the split files do not have: their q, k and v arrive
separately and are fused here, into whichever grouping the loaded DiT has.

#### Community LoRAs — Civitai, musubi-tuner, ai-toolkit

A style, motion or character LoRA trained by the community loads exactly as
the Turbo adapters do: name the `.safetensors` in `lora` (or `lora2`) and set
its strength with `lora_scale`. In the web UI the field has two buttons — the
model picker for catalogued adapters, and a file browser for a `.safetensors`
you downloaded into the sandbox. **No conversion is needed.** The reader takes
each spelling H3's trainers are known to write:

| written by | tensor names | strength |
|---|---|---|
| musubi-tuner, kohya sd-scripts | `lora_unet_blocks_0_attn_qkv_proj.lora_down.weight` / `.lora_up.weight` | per-module `.alpha` |
| ai-toolkit, diffusion-pipe, ComfyUI conversions | `diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight` / `.lora_B.weight` | `.alpha` if present, else at full strength |
| this DiT's own names | `blocks.0.attn.qkv_proj.lora_A.weight` | the same |
| diffusers / peft | `transformer_blocks.0.attn.to_q.lora_A.default.weight` | the header's `alpha` |

kohya's names are this model's own module paths with the dots flattened to
underscores. That cannot be undone from the file's side — `qkv_proj` and
`qkv.proj` flatten alike — so vpipe goes the other way: it flattens the
model's names and looks those up, which is exact. `lora-fuse` reads kohya's
spelling the same way, so for such a file the runtime and merged paths adapt
the same projections.

The log line says which it found — here a musubi-tuner LoRA at its default
targets, four projections in each of the 50 blocks:

```
MetalMiniMaxH3Transformer: runtime LoRA 'my-style.safetensors' -- 200 modules
  at scale 1, rank <= 32, kohya lora_down/lora_up factors
```

**Do not rename the keys by hand.** A converted file works only if it keeps
two things a quick script tends to lose. The first is `.alpha`: kohya
applies each module at `alpha / rank`, so dropping it changes the strength —
an adapter saved at alpha 1 over rank 32 then lands **32× too strong**. The
second is the module each factor pair belongs to: a pair renamed onto the
wrong projection is set aside while the rest of the adapter runs — reported
as `SKIPPED (shape mismatch)` when the new name is a projection of another
shape, and not at all when it names nothing the model has. Hand the loader
the file as the trainer wrote it.

A file that still binds nothing is refused with the spellings above named in
the message. That is a naming question, not a broken download — open an
issue with a few of its tensor names.

The fused `qkv_proj` carries a row order that the names do not show. As with
the Turbo adapters, vpipe reads a fused `qkv_proj` adapter as Comfy-Org's
flat grouping and re-orders it for the per-head MiniMaxAI release. If an
adapter was trained on MiniMaxAI's own weights, set
`lora_qkv_layout: per_head` (`lora2_qkv_layout` for the second slot).

### Eight steps — HyperFlow

[HyperFlow](https://huggingface.co/videorebirth/hyperflow) (Video Rebirth) is
a second way to buy the step count down, and a different kind of adapter from
the Turbo ones above. It is an 8-step **flow-map** self-distillation: every
step is conditioned on the interval it integrates, `(t, r)` — where it starts
and where it lands — rather than on the point `t`. It covers all three
workflows (`t2va`, `fl2va`, `ref2va`) with **one file**.

#### Fetch it, then name it

[`prepare-minimax-h3-hyperflow.vpipeline`](pipelines/prepare-minimax-h3-hyperflow.vpipeline)
fetches it (2.8 GB) and registers it as `videorebirth/hyperflow`:

```sh
vpipe --launch docs/pipelines/prepare-minimax-h3-hyperflow.vpipeline
```

[`minimax-h3-text-to-video-hyperflow.vpipeline`](pipelines/minimax-h3-text-to-video-hyperflow.vpipeline)
is the step-2 graph with the adapter named on the config stage:

```json
"lora": "videorebirth/hyperflow",
"lora_scale": 1.0
```

For a **Ref2VA** graph name `videorebirth/hyperflow-ref2va` instead — the same
file, catalogued a second time so the Browse list offers it under the
reference partition. Any FL2VA graph (first frame, first-and-last) takes the
plain key.

That is the whole change. **You do not set a step count.** The adapter's file
states the sigma grid it was distilled on — nine points, eight forwards — and
vpipe runs that grid whenever the adapter is live; the log says so, and the
graph's `steps` is not used while it is:

```
GenerateVideoStage('generate-video'): HyperFlow 1.0 on lora slot 0 --
8-step flow-map grid from the adapter (`steps: 8` is not used while it is
live), two-time embedding at gate 0.25
```

Keep `video_shift` / `audio_shift` at **12 / 3**, this model's defaults and
the shifts the adapter was trained at. The grid is shifted with whatever the
graph says, as upstream does, but a mismatch is warned about: quality is only
validated at the trained values.

Keep `lora_scale` at **1.0**. At **0** the model is exactly the base model
again — both halves of the adapter switch off together, so a graph can A/B the
two from one loaded model — but values in between are a state the adapter was
never trained in. HyperFlow takes one adapter slot, so a style or identity
adapter can still ride in `lora2`. Two flow-map adapters at once are refused.
Pairing it with a **Turbo** adapter is not refused, and is not useful: both are
distillations of the same model and they would stack.

#### What makes it different

A Turbo adapter is a LoRA and nothing else: the same blocks, a fixed schedule,
fewer of them. HyperFlow is a LoRA plus exactly one architectural change, at
the timestep MLP. Beside the checkpoint's own time embedder it carries an
**endpoint** embedder — a copy of it with its own adapter — that embeds `r`,
and the two are blended before every AdaLN reads them:

```
temb = emb(t) + gate * (emb_r(r) - emb(t))        gate = 0.25
```

So each step needs two numbers per row where it used to need one. Generated
video rows step `(t_video, r_video)` and generated audio rows
`(t_audio, r_audio)`, each on its own shifted grid. Conditioning rows —
keyframes, reference frames, a reference soundtrack — have nowhere to go, so
`r == t` there. The per-step AdaLN tables are baked from `(t, r)` pairs, so
the bake still applies.

The timestep MLP runs on the host in f32, and so does this adapter's part of
it: the file ships those factors in f32 on purpose, and they never pass
through bf16. Checked against upstream's own `TwoTimeEmbedder`, the embedding
agrees to about **1e-6** relative on every `(t, r)` pair an 8-step run uses.
The adapter moves it by **9%**, so the check has five orders of magnitude of
margin. The schedule is **bit-identical** to upstream's.

The rest of the file is the familiar diffusers decomposition — split
`to_q`/`to_k`/`to_v`, value-first `ff.net.0.proj`, the refiner blocks —
fused here into whichever qkv grouping the loaded DiT has, exactly as for
lightx2v's split files. It is applied at runtime, as every adapter here is.

#### What it costs

Measured on an M4 Pro (64 GB) with the 8-bit FL2VA checkpoint preloaded, at
640 × 352 × 56 frames (3943 rows), one after the other in one sitting:

| | forwards | per forward | denoise |
|---|---:|---:|---:|
| base, `steps: 8` | 7 | 27.7 s | 194 s |
| base, `steps: 16` (good quality) | 15 | 27.7 s | ~415 s |
| **HyperFlow** | **8** | **30.2 s** | **241 s** |

The adapter makes each forward **9% slower**. What it buys is the step
count: eight forwards for a clip the base model wants fifteen or more for.
Held in RAM the adapter is about **2.8 GB**, what it is on disk. Its q/k/v
parts are three rank-256 adapters on one fused projection, and each output row
contracts only against its own part. The equivalent single rank-768 update
would be two thirds zeros — 1.1 GB more, and 1.4–2.2% slower per forward in
interleaved measurements on the M4 Pro and the M5 Pro.

Two runs of the same seed produce bit-identical video.

**With Sol-Attn.** Upstream publishes a Sol-Attn recipe for the 8-step grid:
keep the first **2 steps** and the first **2 blocks** dense, at `tau` 1.0. In
vpipe that is `sol_dense_steps: 2`, `sol_dense_layers: 2` and `sol_tau: 1.0` on
`generate-video` beside `sol_attn: true`. A dense step is verified to be the
unrouted model exactly, but the combination's picture quality has not been
measured here — judge it on a seed you know. See
[Faster attention — Sol-Attn routing](#faster-attention--sol-attn-routing).

### Faster attention — the VDN linear branch

The Turbo LoRA above cuts how many steps a clip costs. This cuts what a step
costs, and it does it by changing what attention *is* — from **dense**
attention, whose cost grows with the *square* of the sequence, to a **linear**
one whose cost grows in step with it. The sequence is every video row in the
clip, and that count rises with **both** the duration and the resolution — so
**a longer clip and a bigger frame each make this worth more**, and on a short
small one it is barely worth turning on.

[**VideoDeltaNet on MiniMax H3**](https://openvdn.github.io/#vdn-h3) (VDN-H3)
is not a new model. It is a second checkpoint that sits beside the one you
already have and replaces every main block's dense attention with a **hybrid**
of two halves:

- a **windowed softmax** over the frames near the query — a fixed span of
  whole chunks, not the whole clip;
- a **bidirectional delta rule** carrying everything that window cannot see,
  as a linear-attention recurrence over frames.

The two **partition** the keys. Every frame is read by exactly one half, which
is why this is a different attention rather than an approximation layered on
the old one — and why the branch's weights are trained for it. A frame counted
by both would be counted twice; by neither, silently dropped.

**Why the saving grows with the clip.** Dense attention compares every row
with every other row, so **doubling the clip quadruples its cost** — and the
rows here are the whole clip's video, not one frame's. Neither half of the
hybrid does that. The window is a fixed span of frames however long the clip
runs, so doubling the clip only doubles how many windows there are; the delta
rule beside it is a recurrence over frames, which is linear for the same
reason. Quadratic against linear is a gap that only opens with length, so the
saving is a few percent on a short clip and the bulk of it is still ahead of
you at 10 or 20 seconds.

Measured on one block, both halves on the matrix cores, at two sequence
lengths. Read the **growth** column: that is the quadratic and the linear
doing what they do.

| | 18,887 rows | 81,617 rows | growth |
|---|---|---|---|
| dense attention | 431 ms | 9662 ms | **22.4×** |
| the hybrid's windowed half | 222 ms | 1105 ms | **5.0×** |
| what the hybrid saves | 1.9× | **8.7×** | |

**4.3× the rows costs dense attention 22× and the window 5×.** So the gap is
not a constant to look up — it widens with every row you add.

**Resolution buys this as surely as duration does.** A frame is
`(width / 32) × (height / 32)` rows — 510 at 960 × 544, **1008** at
1344 × 768 — so doubling the canvas doubles the sequence exactly as doubling
the length does, and a quadratic does not care which one did it. Going from
544p to 768p at the same duration is therefore about **4× the dense attention
and 2× the hybrid's**, the same trade the table shows for a longer clip. And
the two compound, because they multiply the same number: a long clip at a
large size is where dense attention is worst and this is worth most. Read the
right-hand column above as about 22 s at 544p **or roughly half that at
768p** — the same rows either way.

**Text and keyframes, not references.** Two different questions sit behind
that line, and they have different answers.

*The partition.* Both released stages are built on the **FL2VA** partition's
blocks and there is no Ref2VA branch to attach, so this and the **Ref2VA
checkpoint** are a choice between, not a pair. What closes that gap is
upstream's **Ref2VA-like** mode, which feeds reference images through the
FL2VA weights — so the branch and references *can* be had together, on FL2VA.
vpipe runs it; see
[Ref2VA-like](#ref2va-like--references-on-the-fl2va-weights) for what it is
and what it is not, and
[`minimax-h3-ref2va-like.vpipeline`](pipelines/minimax-h3-ref2va-like.vpipeline)
for the branch and references in one graph. The Ref2VA *partition* with this
branch attached is still the pairing nothing published covers, and
`generate-video` warns when a graph asks for it.

*The task.* The branch was trained on **text in, video and audio out**. Since
then OpenVDN has published first-frame, last-frame and first-and-last
conditioning on the **same checkpoints**, with the weights unchanged. The
keyframes are packed as conditioning rows ahead of the generated video, held
just short of clean, and the hybrid treats them the way it treats the prompt
and the soundtrack: every generated row attends to each keyframe exactly,
outside the window, and the linear half never sees them.

vpipe gives those rows the same treatment, so the graphs in
[More than text in](#more-than-text-in) take the branch as they are: one anchor
for an opening frame, two for first and last. A last frame on its own is the one mode upstream has
that this graph does not wire — `generate-video` ignores a last-frame anchor
with no first, and says so. The log notes the keyframe count once per
geometry.

Nothing checks the partition for you, though. It is chosen on `model-select`
and the branch on `minimax-h3-model-config`, two different stages, and a
branch attached to the Ref2VA partition's blocks still returns an
ordinary-looking clip. The log warns when that happens but nothing refuses
it, so check that `model-select` names FL2VA.

#### Get it and run it

Two catalogue entries, one repo. **`stage-dmd`** is the 8-step distillation
and the one to take; `stage-b` is the 50-step model it was distilled from.
Each is about **5 GB** on top of the DiT.

```sh
vpipe --launch docs/pipelines/prepare-minimax-h3-vdn.vpipeline
```

Then [**`minimax-h3-vdn.vpipeline`**](pipelines/minimax-h3-vdn.vpipeline),
which is the text-to-video graph with two lines added to
`minimax-h3-model-config`:

```json
"linear_branch": "OpenVDN/vdn-minimax-h3-stage-dmd",
"lora": "larryvrh/MiniMax-H3-Turbo-Lora-v4-600-ema"
```

It composes with the Turbo LoRA — the branch shares the DiT's own q/k/v
projections, so one adapter on those feeds both halves of the hybrid — and
with [`i8_gemm`](#the-settings-worth-knowing), which reaches it twice over:
the branch's output projection is run by the DiT's own GEMM, and the q/k/v it
reads are quantised by then. Nothing inside the branch is affected.

**Load-time, like `lora` and for the same reason:** it changes what the blocks
*are*, not how strongly something is applied. Naming a different branch under
a DiT that is already built is refused rather than half-taken. And a branch
that is named but cannot be attached **fails the stage** instead of warning,
because a graph that silently ran without it would produce a perfectly
plausible video from weights trained for a different attention — there is no
output anyone could look at and tell.

**What it costs in memory: the weights, and at 544p nothing else.** The branch
is the ~5 GB of extra weights above, streamed block by block on the same terms
as the DiT's own. Its *working* memory is larger than that sounds — over a
gigabyte for a 5-second clip and more than two at 13 — but it is not a second
allocation. It is carved out of the attention scratch the DiT is already
holding, which sits idle for exactly the stretch the branch runs in, so at
960 × 544 and above the branch's scratch is free. The two do not scale
together, though: that scratch grows with the clip's **rows** and the branch's
with its **frames**, so a clip of many small frames — under roughly 250 rows
each, which is below 672 × 384 — crosses over and the branch starts costing
the difference. `generate-video` sizes the box for whichever it is before the
run starts.

#### What it saves

The same 124-frame clip as [How long it takes](#how-long-it-takes), on the
**M5 Pro, 24 GB**, 6 steps with the Turbo LoRA:

| | 124 frames, 5.2 s |
|---|---|
| dense attention | **5 min 0 s** |
| VDN hybrid | **4 min 38 s** |

**8% of the wall clock at this length, and that is the small end of it.** A
5-second clip is under 40 latent frames, where the window still covers a large
fraction of the sequence; the hybrid's advantage is in the part of the clip
the window *doesn't* reach, which is why the figure above is a floor. Nor is
attention the whole step — the feed-forward and the projections cost what they
cost either way — so even the 1.9× on that one stage arrives at the wall clock
diluted twice over. At four times the clip the same measurement is 8.7×, and
that is where this setting is worth reaching for.

> **This is a quality trade as well as a speed one, and it has not been
> measured here.** The hybrid is a different attention, and while the branch's
> weights are trained for it, nothing in this repository compares a VDN clip
> against a dense one for fidelity. Upstream reports quality comparable to the
> dense baseline; treat that as their claim, not a reproduction. Generate both
> at a seed you like before committing a long job to it.

### Cheaper attention — SageAttention's int8 QK

The other setting that changes attention, and the one that composes with
everything else here. [**SageAttention**](https://arxiv.org/abs/2410.02367)
runs the **QK^T product in int8** — not by dropping keys, but by computing
every one of them more cheaply. One flag:

```json
"sage_attn": true
```

**What it does.** A flash attention's matrix work is two products per key
block: `QK^T` and `P·V`. Sage quantizes the first pair of operands to int8
with **one scale per block** — per query block for Q, per key block for K,
which is exactly the granularity the kernel already tiles at, so
dequantizing a score tile is a single scalar multiply. `P·V` is left in the
tensor dtype: `P` is a probability, already well-conditioned, and
quantizing it would buy the same again for a much worse error.

**The key smoothing is what makes it work, and it is exact.** K is
quantized as `K − mean(K)` over tokens. A key channel's outlier is not
variation between tokens but a large bias shared by all of them, and
subtracting it spends the int8 range on the signal instead of the bias.
Nothing is added back afterwards: a per-channel shift moves every score in
a row by the same `⟨q, mean⟩`, and softmax does not see a per-row shift. Q
is *not* smoothed, and that asymmetry is not an oversight — only the key
side carries the shared bias, and a shift of Q would move each score by
`⟨q_shift, k_j⟩`, which varies along the row.

**Measured**, on an M5 at 8 heads × 20036 rows × head_dim 128:

| | one attention |
|---|---|
| f16 kernel | **156.6 ms** |
| int8 QK, prologue included | **130.6 ms** |

**1.20×**, of which the quantization prologue is 2.2 ms — 1.7% of the call.
The ceiling is 1.33×: int8 is 2.00× on the fragment pipe and QK is half a
flash kernel's matrix work, so this is most of what was there to take.
Accuracy is cosine **0.99992** against a double-precision reference, where
the f16 kernel is *also* 0.99992.

**Matrix cores only.** The int8 fragment MMA is an M5 instruction and there
is no ALU fallback, so an M4 says so once in the log and runs dense rather
than refusing — a graph that runs today keeps running everywhere it ran
before.

**It composes.** `sage_attn`, `sol_attn` and `i8_gemm` are three
independent choices and none of them reads the others: `i8_gemm` decides
how a block's GEMMs are computed, `sol_attn` which key blocks are attended
at all, and `sage_attn` how the attended ones are multiplied.
`sage_dense_layers` leaves a leading run of blocks in f16; it defaults to
**0**, unlike
`sol_dense_layers`' 1, because Sage computes every key and every query and
there is no published profile that needs a dense prefix.

The same setting is on `generate-image`, where FLUX.2, Krea-2 and
Qwen-Image-Edit take it.

### Faster attention — Sol-Attn routing

The other way to make a step cheaper, and it needs nothing you do not
already have. [**Sol-Attn**](https://nvlabs.github.io/Sana/Sol-Attn/) is a
training-free sparse attention from NVIDIA's Sana project. Unlike the VDN
branch above it is **not a checkpoint** — there is nothing to download and
nothing to attach, and nothing about it is partition-specific, where the
branch is trained for FL2VA's blocks. One flag on `generate-video`:

```json
"sol_attn": true
```

**What it does.** Attention is dominated by key blocks that contribute
almost nothing, and *which* ones those are depends on the clip, the head
and the layer — so it cannot be decided in advance. Sol decides it while
the softmax runs, from a proxy it computes anyway. Per block of 64 keys it
keeps two summaries — the keys' centroid and the values' mean — and per
query block a threshold. One product against the centroids scores every key
block at 1/64 of the dense cost. A block above the threshold is attended
**exactly**; one below is **folded into the same running softmax** as if all
64 of its keys carried the centroid's score. So nothing is dropped, no
routing map is ever built, and there is no second pass.

**The threshold is a distribution, not a block count.** `sol_tau` is
measured in **standard deviations** of that proxy's own spread across
blocks, which is what lets a single number serve every head, layer,
resolution and clip length — where "keep the best 20 blocks" could not.
Higher keeps fewer.

**Two things stay exact whatever `sol_tau` says**, and neither is a tuning
knob. A band of blocks either side of the query's own frame: the near
diagonal is exactly where a centroid is a poor stand-in, because
neighbouring keys are the ones a query is there to tell apart. And the
prompt and soundtrack rows — H3 packs those into one sequence with the
video, and a prompt summarised by its centroid is a prompt half-read.

#### What it saves

124 frames at **832 × 480**, 24 fps, 6 steps with the
[Turbo LoRA](#fewer-steps--the-turbo-lora) and `i8_gemm`, on a **MacBook
Pro 16-inch (M5 Pro), 24 GB**:

| | 124 frames, 5.2 s |
|---|---|
| dense attention | **3 min 30 s** |
| Sol-Attn, `sol_tau` 1.0 | **2 min 44 s** |

**1.27× on the wall clock, and 1.40× on the denoise itself** — the model
load, the prompt encode and the VAE decode are the same work either way.
The run kept **23%** of key blocks exact. That fraction is a property of
the DATA rather than of `sol_tau` alone, so it is measured rather than
predicted: the log reports it once per forward, and it is the number to
watch when tuning.

**It costs no memory.** Everything Sol needs lives for the length of one
attention call, so vpipe lends it buffers the model is not using over that
stretch instead of allocating any — **861 MB** at the size above that the
setting never asks the box for.

#### Against the VDN branch

Both replace the same attention and **only one can be on**; naming both
turns Sol off for the branch's blocks and says so in the log. They are not
the same trade:

| | VDN linear branch | Sol-Attn |
|---|---|---|
| extra weights | ~5 GB, a second checkpoint | **none** |
| partitions | FL2VA only — its weights are trained for those blocks | **not partition-specific** |
| what changes | attention itself — a window plus a linear recurrence | which blocks are attended exactly |
| how it scales | the window is **linear** in clip length, so its advantage widens without limit | a roughly constant fraction of blocks — near a quarter in the runs measured here |

So the branch is the one to reach for on a long clip at a large size, and
Sol is the one that costs nothing to try. No head-to-head at a single
geometry is offered here — the figures in each section are at the size each
was measured at.

#### The knobs

| key | shipped | notes |
|---|---|---|
| `sol_attn` | `false` | Off. It is an approximation, and which clips it is safe on is a judgement about the model rather than about the kernel. |
| `sol_tau` | `1.0` | In standard deviations. Higher keeps fewer blocks: speed rises and quality falls, monotonically in both. |
| `sol_key_block` | `64` | **32 or 64 only.** 32 does more exact work, is slower and is more faithful — a centroid over 32 keys stands in for them better, while halving the block doubles both the routing and the summary sequence. Larger blocks were measured and lose on both counts, so they are declined with a warning and a fall back to 64. |
| `sol_dense_layers` | `1` | Leading blocks left dense. The first block is where the residual stream is least redundant, and it is one of 50. |
| `sol_dense_steps` | `0` | Leading denoising steps left dense — the time-axis twin of `sol_dense_layers`. The first steps decide a clip's coarse structure. HyperFlow's published recipe uses 2 of its 8. |
| `sol_local_radius` | `1` | Blocks either side of the query's own kept exact whatever the routing says. |

It composes with the [Turbo LoRA](#fewer-steps--the-turbo-lora) and with
`i8_gemm`, and the second is worth being precise about, since both are
lossy: measured in this stack, their errors are **independent** — the pair
lands at the quadrature sum of the two apart, not at the linear one, and
switching Sol on does not amplify what `i8_gemm` costs.

> **This is a quality trade as well as a speed one.** What Sol drops is
> chosen per clip, so its effect is not the same on every prompt, and
> nothing in this repository compares a routed clip against a dense one for
> fidelity at full length. Upstream reports quality preserved; treat that as
> their claim, not a reproduction. Generate both at a seed you like before
> committing a long job to it, and raise `sol_tau` only against output you
> have looked at.

### The Neural Engine — for M4-family Macs

Every Apple Silicon Mac has a second accelerator beside the GPU, the
**Apple Neural Engine** (ANE), and a generation normally leaves it idle.
vpipe can hand it part of every block: the rows of the **feed-forward**,
and optionally of the **q|k|v projection**, are split between the two
engines and computed **at the same time**. Rows in those layers are
independent of each other, so the split is exact; there is no seam.

```json
"ane_ffn": true,
"ane_qkv": true
```

**Use it on an M4-family Mac** (M4, M4 Pro, M4 Max). There the GPU's matrix
products already run close to what its memory bandwidth allows, so the ANE
is the only spare compute on the chip, and taking a share of the rows off
the GPU shortens every block.

**On an M5, it typically does not help — leave it off.** The M5 GPU's matrix
cores run these layers several times faster than an M4 GPU's do, while the
ANE is roughly as quick as before, so there is much less for it to take back.
Both engines also read the same memory, so the GPU's own rate falls while the
ANE is predicting. What is left is a few percent, and it depends on the
geometry: MEASURED on the 24 GB M5 Pro at 960 × 544, 243 frames, Sol and
Sage on, 4 steps, the tier armed and took the denoise **203 s → 189 s
(1.07×)**.

The case where it could pay on an M5 is the one where the feed-forward and
projections dominate a block — a **long clip at high resolution** — but that
is also where the modules compete with the forward pass for memory, and on
a 24 GB box they lose. At 1344 × 768, 328 frames the plan granted the module
and the stage then declined it:

```
the 98887-row forward fits only without the ANE modules' ~2711 MB
  -- running the GPU alone for this clip
```

That run matched its GPU-only control to within a second, which is the
fallback behaving. So on an M5 treat the tier as something to **measure on
your own clip**, not as a setting to leave on.

**It does not save memory, whatever the process size suggests.** In the M5
pair above peak RSS fell from 16.0 GB to 12.1 GB, and none of that is a
saving: the modules' weights are **wired outside the process** (~1134 MB and
~374 MB, logged), and the DiT answered by keeping **9 of its 50 blocks**
resident instead of 21. The tier competes for memory with the block
residency that makes streaming cheap, so on a tight box it can cost more in
re-read weights than it wins in the feed-forward. The activation scratch is
unchanged either way — it is sized by the sequence, not by which engine
computes a row.

#### What it saves on an M4

MEASURED on a **MacBook Pro (M4 Pro), 64 GB**, 8-bit, 960 × 544, 124
frames, 4 steps, dense attention:

| | denoise | wall clock |
|---|---|---|
| GPU alone | 10 min 2 s | 14 min 17 s |
| `ane_ffn` + `ane_qkv` | **7 min 23 s** | **11 min 46 s** |

**1.36× on the denoise.** The wall clock gains less (1.21×) because it also
carries the one-time module builds and a VAE decode that the ANE does not
touch. Per block, the two tiers were measured apart at a larger geometry
(1344 × 768, 328 frames, Sol routing every block): the feed-forward alone
took a block from 23.6 s to 19.3 s, and adding q|k|v took it to 17.2 s.

The clip is not bit-identical to the GPU-only one — at the same seed the
frames land ~35 dB PSNR apart, which is what an fp16 path costs.

#### How it behaves

- **It balances itself.** `ane_rows` 0 (the default) sizes the ANE's share
  from the two engines' measured speeds. The tier also times a GPU-only block
  now and then, and **goes back to the GPU alone** when the split measures
  slower. A wrong guess costs a few percent, not the clip.
- **It is lossy, like `i8_gemm`.** The ANE computes in fp16, and the rows it
  takes differ slightly from the GPU's. MEASURED on two blocks of real
  geometry: relative L2 **0.0035** on the video output and **0.00045** on the
  audio. An activation that overflows fp16 is detected, and the chunk is
  recomputed at a smaller scale rather than passed on.
- **8-bit and 4-bit checkpoints work**, and so do runtime LoRAs: each block's
  weights are dequantized, with any adapter merged in, into buffers the ANE
  reads, one block at a time.
- **It costs memory, planned before the run.** Each tier holds one shared
  module however many blocks use it, and the resource plan books both
  together: **1922 MB** at 124 frames of 960 × 544, **2711 MB** at 328 frames
  of 1344 × 768. If the forward does not fit alongside them, the run says so
  and uses the **GPU alone** rather than refusing.
- **The first run pays a one-time build.** macOS compiles each module the
  first time: about **10 s** for the feed-forward and **5 s** for q|k|v on an
  M4 Pro, all before the first denoise step. After that it is cached and
  takes ~50 ms. The cache is kept **per program** (`vpipe` and `vpipe-web-ui`
  each build once) and **per macOS version**, so an OS update pays it once
  more. Resolution and clip length never trigger a rebuild.

| key | default | notes |
|---|---|---|
| `ane_ffn` | `false` | The feed-forward on the ANE. |
| `ane_qkv` | `false` | The q\|k\|v projection too, as a second module. **Requires `ane_ffn`**; the two are planned and granted together, so the plan may decline the pair where it would have granted the feed-forward alone. |
| `ane_rows` | `0` | The ANE's share of the rows, from 0 to 1. `0` balances automatically; a fixed share is for benchmarking. |
| `ane_layers` | `0` | Cap on how many blocks use it; `0` is all of them. Not a memory setting: every block shares one module. |

The same switch exists on **`vae-decode`**, for H3's video decoder, whose
feed-forward is most of the decode. `ane_ffn` there is independent of
`generate-video`'s, and it follows the same M4-yes, M5-no rule.

## Memory

**16 GB is the floor, and it works** — but only because the two big models are
never fully resident. vpipe decides this per run and says what it chose in the
log.

**Weight streaming.** At 8-bit the transformer is ~33 GB and the prompt
encoder ~27 GB, on a machine with 16 GB. Both stream their layers from disk
instead of loading whole, so peak memory is set by the *working set*, not by
the checkpoint. `unload_when_idle: always` on the conditioner also means the
prompt encoder is gone before the denoise starts — the two never share the
machine.

**Adaptive residency.** Streaming everything, every step, re-reads ~8.9 GB per
forward pass at 8-bit. So the transformer *keeps* blocks after using them for
as long as free memory allows, growing its resident set into whatever the box
has spare and giving it back under pressure. How much that is depends on what
else is in the machine: on a 16 GB Air with both VAEs also loaded it settles
at a handful of blocks — a few GB — and sheds when it measures its own pages
leaving RAM.

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
check its registry `model_type` is `minimax-h3-fl2va` or
`minimax-h3-ref2va`; you can always type the key or an absolute path
instead.

**It generated video but the audio is silent or wrong.** Check that
`save-video` has `enable_audio: true` and that `audio-vae-decode` is wired to
`generate-video` **port 1** (port 0 is video).

**The generated soundtrack doesn't carry the reference music.** The prompt
has to ask for it — the sound comes from the same text as the picture, so
name what is being played (see [step 2](#step-2--text-to-video-and-audio)).
Then check the encoder's line: `N reference(s) -> ... M reference audio
rows`, where `M` should be `2 × 40 × seconds` — 400 for a 5 s reference.
Far more than that means the soundtrack reached it at the wrong rate; the
encoder warns about it and resamples, and the fix is to set the producing
`audio-to-pcm`'s `output_sample_rate` to **32000**.

**`frames` isn't what you asked for.** Expected — see the table above.

**`ane_ffn` was set but nothing ran on the ANE.** Two lines say why. Either
the modules did not fit beside the forward pass (*"the N-row forward fits
only without the ANE modules"*), in which case the clip runs on the GPU
alone — lower `frames` or the frame size, or turn the tier off; or the tier
armed and its controller measured the split as slower and went back to the
GPU, which on an M5 is the expected outcome. See
[The Neural Engine](#the-neural-engine--for-m4-family-macs).

## Under the hood

- One packed sequence carries video and audio rows together; the per-row AdaLN
  modulation that conditions it is 13B of the 33B.
- Two sigma schedules advance in lockstep — video shifted 12, audio 3
  (`video_shift` / `audio_shift` on `minimax-h3-model-config`).
- The video VAE is 24-channel at 1/16 resolution; audio decodes through a
  separate VAE to **32 kHz stereo**.
- On **M5**, the GEMMs and attention run on the GPU's matrix cores
  (`matmul2d` / NAX flash attention) — including both accelerated attention
  settings, whose exact halves are that same flash kernel walking only the
  key blocks they keep.

## References and licences

**MiniMax-H3** is published by MiniMax AI under the **MiniMax H3 Community
Licence Agreement**, which is the licence the checkpoint carries and the one
you accept on its Hugging Face page before downloading:

- Weights (FL2VA + Ref2VA): <https://huggingface.co/MiniMaxAI/MiniMax-H3>
- Repack used as a second opinion on the qkv grouping:
  <https://huggingface.co/Comfy-Org/MiniMax-H3>

**VideoDeltaNet on MiniMax H3 (VDN-H3)** — the hybrid attention of
[Faster attention](#faster-attention--the-vdn-linear-branch) — is by Haocheng
Xi, Yiming Xie, Hexu Zhao, Yiwen Zhang, Michael Liu, Thomas Creavin, Kurt
Keutzer, Xiuyu Li, Zhaoyang Lv, Chenfeng Xu and Haiwen Feng, of UC Berkeley,
Impossible Inc. and UT Austin. Its Hugging Face repository declares the same
**MiniMax H3 Community Licence Agreement** as the base model it attaches to,
with the licence text in the repo's own `LICENSE`.

- Project page: <https://openvdn.github.io/#vdn-h3>
- Code: <https://github.com/OpenVDN/vdn-minimax-h3>
- Weights: <https://huggingface.co/OpenVDN/vdn-minimax-h3>

```bibtex
@misc{xi2026videodeltanet,
  title  = {VideoDeltaNet on MiniMax H3},
  author = {Haocheng Xi and Yiming Xie and Hexu Zhao and Yiwen Zhang and
            Michael Liu and Thomas Creavin and Kurt Keutzer and Xiuyu Li and
            Zhaoyang Lv and Chenfeng Xu and Haiwen Feng},
  year   = {2026},
  url    = {https://openvdn.github.io/}
}
```

**Sol-Attn** — the routed attention of
[Faster attention — Sol-Attn routing](#faster-attention--sol-attn-routing)
— is by Haopeng Li, Yitong Li, Junsong Chen, Tian Ye, Haozhe Liu, Jincheng
Yu, Duomin Wang, Ruihua Zhang, Zeke Xie, Enze Xie and Song Han, of NVIDIA.
It ships as part of NVIDIA's **Sana** repository, under that repository's
**Apache-2.0** licence. It carries no weights of its own, so nothing is
downloaded for it and nothing is redistributed here — the implementation in
this tree is written from the published method and sources.

- Project page: <https://nvlabs.github.io/Sana/Sol-Attn/>
- Paper: <https://arxiv.org/abs/2607.24027>
- Code: <https://github.com/NVlabs/Sana>

```bibtex
@misc{li2026solattn,
  title  = {Sol-Attn: Accelerating Video Generation Inference via
            On-the-Fly Attention Sparsification},
  author = {Haopeng Li and Yitong Li and Junsong Chen and Tian Ye and
            Haozhe Liu and Jincheng Yu and Duomin Wang and Ruihua Zhang and
            Zeke Xie and Enze Xie and Song Han},
  year   = {2026},
  eprint = {2607.24027},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url    = {https://arxiv.org/abs/2607.24027}
}
```

**Turbo LoRAs** are community distillations, each under its own repository's
terms: [larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)
and [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo).

None of these are redistributed here — every pipeline in this document
downloads from the publisher, and the licence you accept is theirs. Check each
before using its output commercially.

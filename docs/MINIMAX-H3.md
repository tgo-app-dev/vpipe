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
arrives in **8–16 steps** instead of 30+.

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
| Adding the Ref2VA partition (its transformer only) | **+66 GB** source, **+33 GB** 8-bit or **+24 GB** 4-bit |

Once preparation finishes you can delete the source repack and keep the
~65 GB. The 115 GB is a one-time cost, not a standing one.

**Why 8-bit and not 4.** Quality, at the price of disk and very little else.
Both widths stream, so neither has to fit in RAM, and the transformer spends
its time on weights it is reading from storage rather than on how wide they
are — so moving to 8-bit costs about 20 GB and does not meaningfully change
how long a clip takes. The 4-bit path still works and is the one to take if
the disk matters more: set `bits` to 4 in both `model-quantize` stages and
name the output to match.

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
- **[`minimax-h3-first-last-to-video.vpipeline`](pipelines/minimax-h3-first-last-to-video.vpipeline)**
  — the same, anchored to an image at both ends (see
  [More than text in](#more-than-text-in)).
- **[`minimax-h3-reference-to-video.vpipeline`](pipelines/minimax-h3-reference-to-video.vpipeline)**
  — the **Ref2VA** partition instead: reference images, clips and soundtracks
  in, `.mp4` out, each one prepared to a size you choose (see
  [Conditioning on references](#conditioning-on-references-ref2va)).

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

> *An Asian musician playing classical music on a grand piano.*

The soundtrack is in that sentence: naming what is being *played* is what
gives the model the music. There is no separate audio prompt — the sound
comes from the same text — so describe the **sound** as well as the picture.
Either name it outright (*"sound of the rain, with the occasional individual
drop"*) or, as here, name the thing making it. A prompt that says only what a
scene looks like gets you whatever the model thinks that scene sounds like.

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
final clip, at roughly proportional cost.

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
video conditioned on nothing.

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

#### The example

[`minimax-h3-reference-to-video.vpipeline`](pipelines/minimax-h3-reference-to-video.vpipeline)
runs on assets you already have:

| reference | carries | where it comes from |
|---|---|---|
| `minimax-h3-reference-subject.jpg` | the subject | [ships in this repo](images/minimax-h3-reference-subject.jpg) — made with vpipe's own FLUX.2 text-to-image graph, so it comes with no licence question attached |
| `minimax-h3-text-to-video.mp4` | the camera move **and** the soundtrack | whatever [step 2](#step-2--text-to-video-and-audio) wrote. A clip with audio stays ONE reference carrying both, labelled `<Video 2>` and `<Audio 2>`, however you feed it |

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
**`attach_audio: [3]`** on the encoder makes port 3's audio the soundtrack of
the reference before it, so it stays one `<Video 2>` + `<Audio 2>` pair instead
of becoming a third, independent reference. **`channels: 2`** carries true stereo, which the
`references` path also does and a mono chain would silently give up.
And **`max_mb: 384`** on the clip stacker is sized on purpose: 90 frames at
1344 × 768 is 279 MB, over the 256 MB default, which would otherwise cap the
group at 86 frames and warn.

MEASURED: this produces a request identical to the `references` list at every
count that exists — 2 references, 3,115 conditioning rows, 13,120 reference
video rows, 130 audio rows, a 22,615-row packed sequence. It is not the same
*video*, at the same seed: the resize now happens in `image-resample` rather
than inside the encoder, and two implementations landing on the same dimensions
do not land on the same bytes. If you want bit-equivalence with the file list
instead of the size control, leave the resamples at the source size and put
`short_edge: 768` in the clip stacker's `sideband`.

One inefficiency worth knowing: the encoder truncates a clip to `frames`, so
this chain resizes all 90 frames and uses 39. The `references` list truncates
first and resizes only what it keeps. Bound the source rather than raise
`max_mb` if that matters.

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
information — and `reference_video_short_edge: 0` declines it, encoding the
clip at its own size instead. MEASURED on the shipped example, back to back on
an idle 16 GB M5 at the same seed:

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
request**: it numbers the references in the prompt the model reads
(`<Picture 1>`, `<Audio 2>`, `<Video 1>`) and places them on a shared clock,
so reordering the list is a different generation — and port references
continue that numbering after the last file.

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
particular: at 2048 a single still is 121 VAE tiles, and nine are allowed. The
example lowers it to 1024, which is 25.

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

It adapts the **FL2VA** partition (text-to-video and image-to-video), not
Ref2VA. Two checkpoints are catalogued because the choice between them is
real rather than a version bump:

| entry | steps | shifts (v/a) | trained at | when |
|---|---|---|---|---|
| `Turbo few-step v4-600 EMA` (larryvrh) | 4–8 | 12 / 3 | — | the default. Better static and small-motion shots, better micro-detail. |
| `Turbo few-step v1-850 EMA` (larryvrh) | 4 | 12 / 3 | — | only for **4 steps with large, fast motion**, where v4 can trail or smear. At 6–8 steps prefer v4. |
| `Turbo 4-step v1.0 768p` (lightx2v) | 4 | **6** / 3 | 1344×768 | a second distillation, at 768p. **Set `video_shift: 6.0`** — see below. |
| `Turbo 8-step v1.0 544p` (lightx2v) | 8 or 4 | 12 / 3 | 544p | the 8-step of that line, on the checkpoint's own shifts. |

**The shifts are part of the adapter, not a preference.** lightx2v's 4-step
was distilled on a video shift of **6** where this model's default — and every
other adapter here — is 12. A distillation is fit to the sigma grid it was
trained on, so running it on the wrong one is not a style difference; it is a
different schedule, and nothing will report it. `video_shift` lives on the same
`minimax-h3-model-config` stage as `lora`, which is exactly why the two travel
in one beat.

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
same shape, pinning `lightx2v/Minimax-h3-Turbo-4step-768p` (swap in
`lightx2v/Minimax-h3-Turbo-8step` for the other). Only the **ComfyUI**
spellings of that repo are catalogued: it publishes each adapter twice and the
diffusers copy needs a conversion this build does not do — see *Which Turbo
adapters work* below.

`model_variant` is not optional here and its value is the catalogue **name**,
not a word from the title. Both Turbo checkpoints are published from the one
repo, so a bare `"turbo"` matches both and the fetch is refused with the
candidates listed rather than quietly taking the first. Swap in
`larryvrh/MiniMax-H3-Turbo-Lora-v1-850-ema` for the other one; they share a
directory on disk and register under separate keys.

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

Note that this pipeline sets `i8_gemm`. It is an opt-in **lossy** accelerated
mode that only matrix-core GPUs (M5 and newer) can use — so it does nothing on
the M4 Pro, and an M5 run without it will be slower than 11 min 25 s. It works
with the adapter, but it changes the picture by about as much as the adapter
does, so turn one at a time when you are judging output rather than speed.

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
`diffusion_model.` container prefix (the ComfyUI convention) on top of them.
Measured against the FL2VA base:

| adapter | modules | works |
|---|---|---|
| `larryvrh/MiniMax-H3-Turbo-Lora` | 259 (adds `adaln_proj`, `final_layer`) | yes, both paths |
| `lightx2v/Minimax-h3-Turbo`, the `_comfyui_` files | 208 | yes, both paths |
| `lightx2v/Minimax-h3-Turbo`, the diffusers files | 312 | **no** |

The diffusers spelling is not a naming difference but a different
decomposition: separate `to_q`/`to_k`/`to_v` adapters that would have to be
stacked block-diagonally into the fused `qkv_proj`, and an `ff.net.0.proj`
in diffusers' **value-first** order whose halves would need swapping. Both
transforms produce a well-shaped, wrong model if guessed at, so they are
refused rather than approximated.

lightx2v's `_comfyui_` `qkv_proj` is rank 384 — three rank-128 adapters
stacked — and that stacking is how you can tell which base it assumes: its
`B` is block-diagonal in the `[all q | all k | all v]` sense, so it targets
the Comfy-Org flat grouping, not the per-head release its `base_model` tag
names.

**On the released MiniMaxAI weights either path would be wrong.** The adapter
is trained against Comfy-Org's repack, so its `attn.qkv_proj` delta assumes the
flat grouping. Applied to the per-head release it would add one head's `q`
delta onto another head's `k`, in all 50 blocks, with nothing to report.

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

## Under the hood

- One packed sequence carries video and audio rows together; the per-row AdaLN
  modulation that conditions it is 13B of the 33B.
- Two sigma schedules advance in lockstep — video shifted 12, audio 3
  (`video_shift` / `audio_shift` on `minimax-h3-model-config`).
- The video VAE is 24-channel at 1/16 resolution; audio decodes through a
  separate VAE to **32 kHz stereo**.
- On **M5**, the GEMMs and attention run on the GPU's matrix cores
  (`matmul2d` / NAX flash attention).

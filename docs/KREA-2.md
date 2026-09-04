# Krea-2 on Apple Silicon

**Krea-2-Turbo** is a 12-billion-parameter text-to-image model — a dual-stream
MMDiT over a Qwen3-VL text encoder — and it is **CFG-distilled**, which is
what makes it pleasant to run locally: there is no second, unconditional forward
pass per step, and a finished image arrives in **8 steps** instead of 30+.

vpipe runs it on-device through its **metal-compute** backend: its own Metal
kernels, no Python and no third-party tensor runtime in the forward pass. On
an M5 the GEMMs and attention run on the GPU's matrix cores.

Nothing here is quantized. The numbers below are the **published bf16
weights**, run as they ship.

## What you need

| | |
|---|---|
| **Machine** | Apple Silicon Mac (M-series). |
| **Memory** | **16 GB** runs it; 24 GB is the faster row below. The bf16 DiT streams its blocks either way. |
| **Disk** | **~36 GB** for the model, **+0.3 GB** for the LoRA. The DiT streams, so read speed shows up in the timings — see [How long it takes](#how-long-it-takes). |
| **Build** | An Apple Silicon build of vpipe — the default on arm64 macOS. See the main [README](../README.md). |
| **Account** | A Hugging Face account, because **the model is gated**. See [step 1](#step-1--get-the-model). |

The download is ~35.6 GB rather than the repo's ~62 GB: the catalogue entry
pins the diffusers subfolders it needs and skips the redundant top-level
`turbo.safetensors`, which is a second copy of the transformer.

## The pipelines

- **[`prepare-krea-2.vpipeline`](pipelines/prepare-krea-2.vpipeline)** — fetch
  the model and the M87 LoRA. Run once.
- **[`krea-2-text-to-image.vpipeline`](pipelines/krea-2-text-to-image.vpipeline)**
  — prompt in, `.jpeg` out. Plain, and the one to run from the terminal.
- **[`krea-2-preview.vpipeline`](pipelines/krea-2-preview.vpipeline)** — the
  same image, plus a **live preview of the denoising** in the web UI.

Follow a link and use **Raw ▸ Save as** to download one, or take them straight
from `docs/pipelines/` in your clone. Either runs from the terminal with
`vpipe --launch <file>` or opens with **Load** in the web UI's Pipeline
Manager. Both are plain JSON — read them, edit them, keep them in version
control.

## Step 1 — get the model

### Krea-2 is a gated repo

`krea/Krea-2-Turbo` is **gated on Hugging Face**, so a fetch with no
credentials fails with an authorization error no matter how the pipeline is
written. Three things, once:

1. **Sign in** to [huggingface.co](https://huggingface.co) and open
   [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo). **Accept the
   licence** on the model page — the download stays refused until you do, and
   a token alone will not substitute for it.
2. **Create a token** at
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
   **Read** access is enough; this only ever downloads.
3. **Paste it** into the `fetch-krea-2` stage's **`hf_token`** in
   `prepare-krea-2.vpipeline`.

Leaving `hf_token` empty is also legitimate — the stage then falls back to
`$HF_TOKEN` / `$HUGGING_FACE_HUB_TOKEN`, and if the download turns out to be
gated it *prompts* for one. In the web UI that prompt is a masked field on the
User I/O page. Put it in the file when you want the pipeline to be
self-contained; use the environment when you would rather not have a token in
a file you might share.

> **A token is a credential.** The shipped pipeline has `hf_token` empty on
> purpose. If you paste yours in, that file now carries it — do not commit it
> or hand it round.

### From China: ModelScope, and no gate

The same model is published on **modelscope.cn**, where it is **not gated** —
no account, no licence click, no token. Set the fetch stage's **`source`**:

```json
"source": "modelscope"
```

That picks the SOURCE only. The registry key and the on-disk directory still
come from the Hugging Face path, so a model fetched from either side is one
model in one place and switching later does not produce a second copy.

For a machine that should always prefer it, set the environment variable once
instead of editing every stage:

```sh
export VPIPE_MODEL_SOURCE=modelscope
```

An empty `source` (what the pipeline ships with) reads that variable, then
falls back to `huggingface`.

### Then run it

```sh
cd ~/vpipe-work                                    # your work directory
cp ~/src/vpipe/docs/pipelines/prepare-krea-2.vpipeline .
~/src/vpipe/build/apps/vpipe/vpipe --launch prepare-krea-2.vpipeline
```

vpipe treats **the directory you launch it from** as its workspace: `models/`
and the LMDB registry are created there. Use the **same** directory in step 2,
or the run will not find what you just downloaded.

Two stages, in order:

1. **`model-fetch`** (`krea/Krea-2-Turbo`) — ~35.6 GB.
   `skip_existing_files` is on, so re-running after an interruption skips
   every shard already on disk at the right size and resumes the one that was
   cut off from the byte it stopped at.
2. **`model-fetch`** (`mgwr/M87`) — the **M87 aesthetic LoRA**, ~0.23 GB and a
   few seconds. It is wired to the first stage's output so the two run in
   order rather than fighting for the link. A LoRA is used as it ships, so
   there is nothing to quantize afterwards.

The LoRA is **not optional in the shipped pipelines** — both name it — but
it is one config key, and [Running without the LoRA](#running-without-the-lora)
says what to change.

## Step 2 — text to image

```sh
cd ~/vpipe-work                                    # the SAME directory
cp ~/src/vpipe/docs/pipelines/krea-2-text-to-image.vpipeline .
~/src/vpipe/build/apps/vpipe/vpipe --launch krea-2-text-to-image.vpipeline
```

That writes `krea-2-text-to-image.jpeg` next to wherever you started vpipe.
The path is **relative** on purpose, so the same file works from the CLI and
the web UI without editing.

The graph is eight stages:

```
text-prompt ──> diffusion-conditioner ──> generate-image ──> vae-decode ──> save-image

model-select       ──> diffusion-conditioner, generate-image, vae-decode
krea2-model-config ──> diffusion-conditioner (port 5), generate-image (port 7)
scheduler-select   ──> generate-image (port 4)
```

`model-select` names the checkpoint once and every model-holding stage latches
it, so you point **one** stage at a model rather than three.

### The settings worth knowing

| key | stage | shipped | notes |
|---|---|---|---|
| `width` / `height` | `generate-image` | 1024 × 1024 | Multiples of **16**. |
| `steps` | `generate-image` | 8 | The distillation's own count. Set it in **both** `generate-image` and `scheduler-select`, or the schedule and the loop disagree. |
| `seed` | `generate-image` | 0 | Same seed + same settings ⇒ same image. |
| `i8_gemm` | `generate-image` | `true` | An opt-in **lossy** accelerated mode. Matrix-core GPUs (M5 and newer) only — it does nothing on an M4, and the timing below assumes it is on. |
| `guidance_scale` | `generate-image` | (default 1) | **Leave it.** Turbo is CFG-distilled; above 1 it runs a second DiT pass per step and pushes toward a negative prompt it was never trained against. |
| `shift` | `scheduler-select` | **0.3** | The tuned one. See below. |
| `lora` / `lora_scale` | `krea2-model-config` | `mgwr/M87` / 1.0 | The adapter, by registry key. |

### The scheduler shift, and why 0.3

This is the setting that earned the pipeline its name, and it is worth a
paragraph because the default is not tuned for eight steps.

The flow-matching schedule warps a linear sigma ramp by `shift`
(`sigma' = e^shift / (e^shift + (1/sigma − 1))`). At the **default 1.15** an
8-step run gets these per-step sigma drops:

```
0.043  0.052  0.064  0.081  0.105  0.142  0.202  0.311
```

The last step alone covers **31% of the whole journey**, and the largest step
is **7.2×** the smallest. The schedule dawdles in the noise and then jumps to
the finish — which is fine at 30 steps and is exactly where a turbo model
loses detail at 8.

At **`shift: 0.3`** the same eight steps become:

```
0.096  0.102  0.110  0.118  0.127  0.137  0.149  0.162
```

Largest step **16%**, and only **1.7×** the smallest — nearly even. The work
moves **earlier**, which is the "front-load", and the terminal leap is roughly
halved. Same model, same eight forwards, same cost: it is a schedule change,
not extra compute.

Push it lower and the run starts spending its budget on refinements the model
has no structure to hang them on. 0.3 is a good default for 1K squares at 8
steps; treat it as a starting point rather than a constant, and re-tune if you
move far from that geometry.

### How long it takes

**1024 × 1024, 8 steps, `i8_gemm` on, bf16 weights + the M87 LoRA**, end to
end from `vpipe --launch` to the written `.jpeg`:

| machine | GPU | | |
|---|---|---|---|
| **M5 Pro** MacBook Pro 16", 24 GB, its own fans | 20 cores | **35 s** | |
| **M5** MacBook Air 15", 16 GB, fanless | 10 cores | **81 s** | 2.3× |

Both are the *published bf16 weights* — no quantization pass, no prepared
variant beyond the download.

**It scales with GPU cores, and almost exactly.** Twice the cores and a
sustained-clock ratio of 1500 / 1300 predict 2.0 × 1.15 = **2.31×**, or
**80.8 s** against the **81 s** measured — within a third of a percent. Two
points and two factors is a fit, not a proof, but it is a good deal tighter
than the inputs deserve and it says something worth knowing.

**What it says is that this run is GPU-COMPUTE bound**, and that is not
obvious, because **both machines STREAM**. At bf16 the 28 transformer blocks
do not fit beside the conditioner's text encoder even on 24 GB —
`plan_streaming` says stream there too. If the block reads were the
constraint, halving the cores would not have cost a clean 2×; the reads hide
under the compute on both boxes. That is also why 16 GB is not a cliff here:
the smaller machine is slower in proportion to its GPU, not to its memory.

**Both runs had the weights on the machine's INTERNAL SSD**, and that
qualifier carries the conclusion. A streamed forward re-reads its blocks every
step, so "the reads hide under the compute" is a statement about a particular
drive keeping up with a particular GPU — and internal storage on these
machines does, comfortably enough that doubling the GPU still bought a full
2×. It is not a property of the model.

**On external storage, check before you assume.** Thunderbolt 4 carries
40 Gbps — a 5 GB/s ceiling — so the *link* is not necessarily the limit.
What varies is the drive and the enclosure behind it, and the range is wide.
One external SSD measured for [the drive
table](MODEL-MEMORY.md#reading-a-streamed-block) in the memory notes came in
at 0.77–0.82 GB/s against 3.1–6.3 GB/s internal, but that is one device and
not a verdict on the interface.

What matters is only whether yours keeps up. If it does not, a streamed run
becomes storage-bound instead of compute-bound — and a storage-bound run does
*not* scale with GPU cores, so the table above stops predicting anything.
**Internal storage is the safe default**; on an external drive, measure your
own before reading these numbers as an estimate rather than a ceiling.

**The Air row was measured with no cooling aid of any kind**, on a desk. It
starts around **1500 MHz** and throttles to about **1300 MHz** by the end —
so the clock is worth ~**1.15×** of the 2.3× and the core count is the other
**2.0×**. An 81-second run is long enough to spend most of itself past the
boost window; a shorter one (a 512² image, or fewer steps) sits nearer the
opening clock and comes in better than this row would predict.

`steps` is the setting that moves this most, and it moves it close to
linearly: the DiT is ~all of the time, and it runs once per step.

## The live preview

[`krea-2-preview.vpipeline`](pipelines/krea-2-preview.vpipeline) is the same
graph with two stages added and two edges moved, and it shows the image
**forming** rather than appearing:

```
generate-image ─1─> vae-decode ─┬─> preview      (all 8 steps, as video)
                                └─> temporal-slice(-1) ──> save-image
```

Three things make that work, and each is worth knowing on its own.

**`generate-image` has a second oport.** Port 0 is the finished latent; port
**1** is `step_latent` — one beat per sampler step, same format — and it is
**only emitted when something is connected**. So the preview costs nothing in
the graph that does not ask for it.

**One decode serves both.** `vae-decode` is wired to port 1, so it decodes
every step, and the preview and the file both read that one stream. The
price is honest: eight VAE decodes instead of one. That is why the plain
pipeline does not do it, and why it is the better one for the terminal.

**`temporal-slice` throws the intermediates away.** `start: -1` keeps the
**last** beat of the stream and drops the rest, so `save-image` writes the
finished image and not eight files. The stage can only resolve a negative
index once the source hits EOS, which is exactly when the last step lands.

The `preview` stage carries `image_mode: false` — *always video*. The eight
frames then arrive as a short clip of the denoise rather than as eight
stills replacing one another.

Open the web UI from your work directory, load the pipeline, and the preview
appears in the Composer panel while it runs.

## The M87 LoRA

`mgwr/M87` is an early-preview aesthetic LoRA for Krea-2 Turbo — a standard
rank-32 low-rank adapter, ~0.23 GB, in the ai-toolkit / ComfyUI key
convention.

**It has a trigger, and the shipped prompts end with it:**

```
… no readable text, no watermark, no logo --preview
```

That trailing `--preview` is the trigger token the adapter was trained with,
not a command-line flag. Drop it and you keep the adapter's weights but lose
most of what you loaded it for.

It is applied **at run time**, named by registry key on
`krea2-model-config` — nothing is written to disk and the base checkpoint is
untouched, so switching adapters or turning one off is a config edit rather
than a pass over 26 GB. `lora_scale` is live: `1.0` is what upstream tunes
for, and `0` skips the adapter's GEMMs entirely, so *off* is exactly off.

There is a second slot, `lora2` / `lora2_scale`, if you want a style adapter
riding alongside.

### Running without the LoRA

Delete the `lora` and `lora_scale` keys from `krea2-model-config` and drop the
`--preview` trigger from the end of the prompt. Everything else is unchanged —
including the shift, which is a property of the schedule and not of the
adapter. The `krea2-model-config` stage can stay: with no keys set it emits
the defaults.

## Troubleshooting

**`model-fetch` reports an authorization failure.** The licence is not
accepted, or the token is missing or wrong. Both halves are needed — see
[step 1](#step-1--get-the-model). Or switch to `"source": "modelscope"`, which
is not gated.

**The model isn't in the picker.** `model-select`'s Browse list is filtered to
families the stages can run. Check the registry `model_type` is `krea2`; you
can always type the key or an absolute path instead.

**The image is soft, or over-smooth at 8 steps.** Check `shift` is on the
`scheduler-select` stage and that `steps` matches `generate-image`'s. A
default 1.15 schedule at 8 steps spends a third of the run in its last step.

**The LoRA seems to do nothing.** Check the prompt still ends with
`--preview`, and that `lora_scale` is not 0.

**It is slower than 35 s.** `i8_gemm` does nothing before M5, and 35 s is a
fan-cooled M5 Pro — see [How long it takes](#how-long-it-takes) for a fanless
16 GB row. The bf16 DiT streams its blocks on any box, so a slow drive shows
up here; a smaller box keeps fewer of them between steps.

## Under the hood

- 12B dual-stream MMDiT: 28 image blocks, head_dim 128, 48 query heads with
  GQA kv=12, 3D-RoPE, and an interleaved 12-layer text tower.
- The conditioner reads **12 selected hidden layers** of the Qwen3-VL text
  encoder, not just its last state — the choice is in the checkpoint's
  `model_index.json`.
- The VAE is 16-channel at 1/8 resolution with per-channel whitening; the
  `vae-decode` stage un-whitens.
- On **M5**, the GEMMs and attention run on the GPU's matrix cores
  (`matmul2d` / NAX flash attention).

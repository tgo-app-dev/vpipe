# Reference image editing with FLUX.2-klein-9B

English | [简体中文](KLEIN-zh-cn.md)

**FLUX.2-klein-9B** takes a photo and a sentence and gives you the photo
edited — or takes **two** pictures and a sentence and combines them into one.
vpipe runs it on-device through its **metal-compute** backend — its own Metal
kernels, no Python and no third-party tensor runtime in the forward pass —
quantized to 4-bit, in **4 steps**, on a 16 GB Mac.

## Two checkpoints: klein-9B and klein-9b-kv

Black Forest Labs publish the model twice, and vpipe runs both:

- **`FLUX.2-klein-9B`** — the model. The reference image joins the joint
  attention with the prompt and the image being generated, and is recomputed
  at every denoise step.
- **`FLUX.2-klein-9b-kv`** — a **variant** of the same model, distilled so
  that **reference tokens attend only to themselves**. They see neither the
  prompt nor the image being generated, which makes their K/V independent of
  the timestep, so vpipe computes them **once** at step 0 and reuses them for
  the rest. BFL measure **1.21–2.66×**, the larger end with several references
  at modest output sizes.

The price of the speed is that the -kv variant **sometimes gives a slightly
lower-quality result** than klein-9B on the same photo and prompt. Most edits
come out equally well; when one does not, run the same graph on klein-9B.

| | klein-9B | klein-9b-kv |
|---|---|---|
| **Reference tokens** | recomputed at every step | computed once, reused |
| **Speed with references** | baseline | faster — more so with two references |
| **Quality** | the reference point | sometimes a little lower |
| **`klein_kv`** | not set — no `flux2-model-config` stage | **`true`, required** |
| **Hugging Face repo** | [`black-forest-labs/FLUX.2-klein-9B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) | [`black-forest-labs/FLUX.2-klein-9b-kv`](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv) |

Every graph in this document ships in both versions. They differ in exactly
two places: the model `model-select` names, and one `flux2-model-config`
stage the -kv graphs add.

> **`klein_kv` is not an optimization flag.** The two checkpoints are
> indistinguishable on disk — same `config.json`, same tensor names, same
> shapes — so vpipe cannot detect which one you have. The token order and
> attention mask differ, and the weights are distilled *for* that mask, so
> running one checkpoint through the other's forward pass is not slower, it
> is **wrong**. Set `klein_kv: true` for the -kv checkpoint and leave it unset
> for klein-9B. The shipped pipelines already do.
>
> It is set on a **`flux2-model-config`** stage wired to `generate-image`'s
> `model_config` iport, not in `generate-image`'s own config. Model-specific
> settings live in per-family stages so a graph shows which checkpoint it was
> built for — and so a key that does not apply to the resident model is a
> visible wiring mistake rather than a line of config that is quietly
> ignored.

## What you need

| | |
|---|---|
| **Machine** | Apple Silicon Mac (M-series). |
| **Memory** | 16 GB. |
| **Disk** | **~32 GB** to run bf16 as published, or **~45 GB** to prepare 4-bit and **~12 GB** to keep — per checkpoint. |
| **Build** | An Apple Silicon build of vpipe — the default on arm64 macOS. See the main [README](../README.md). |
| **Hugging Face** | An account, the licence accepted, and an **access token**. Both checkpoints are gated — see [below](#the-models-are-gated). |

The published checkpoints are bf16. You can run them as they are, or have
vpipe quantize them to 4-bit once, up front — see
[Do you need to quantize?](#do-you-need-to-quantize). The two are the same
size:

| | |
|---|---|
| Download (bf16) | **~32 GB** |
| Peak while preparing (download + output) | **~45 GB** |
| 4-bit model, download deleted | **~12 GB** |

Each repo is ~49 GB, but vpipe's catalogue entries pin the diffusers
subfolders and **skip the redundant top-level copy of the transformer**
(~17 GB of the same weights) and the sample images. You download ~32 GB.

You only need one checkpoint. Preparing both needs the space twice over.

## The models are gated

Both Hugging Face repos are behind the **FLUX Non-Commercial License**. An
unauthenticated fetch does not fall back to anything — it **401s** — so all
three of these must be done before a prepare pipeline will download anything.
(The ModelScope mirrors are not gated; see
[Running bf16](#running-bf16-one-download-straight-from-the-terminal).)

1. **Accept the licence** for each checkpoint you want. Sign in to Hugging
   Face, open the model page —
   [klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) or
   [klein-9b-kv](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv)
   — and accept the terms. They are accepted per repo. Do this first: a token
   cannot grant access you have not accepted.
2. **Create an access token** at
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
   A **Read** token is enough. If you create a fine-grained token instead,
   it must carry **“Read access to contents of all public gated repos you can
   access”** — a fine-grained token without that permission looks valid and
   still 401s on these repos, which is a confusing way to lose an hour.
3. **Put the token in the pipeline.** Open the prepare pipeline and fill in
   the `model-fetch` stage's empty `hf_token`:

   ```json
   {
     "id": "fetch",
     "type": "model-fetch",
     "config": {
       "model_path": "black-forest-labs/FLUX.2-klein-9B",
       "hf_token": "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
       "base_path": "./models",
       "skip_existing_files": true,
       "overwrite_existing": false
     }
   }
   ```

   You can also set it in the web UI — Pipeline Manager ▸ the `fetch` stage ▸
   `hf_token` — which avoids putting the secret in a file at all.

> **A token is a password.** The prepare pipeline is a file you might commit
> or share; a token pasted into it goes with it. Clear the field once the
> download finishes, and revoke the token on Hugging Face if it ever leaks.

## The pipelines

| | klein-9B | klein-9b-kv |
|---|---|---|
| **Prepare** — download and quantize. Run once. | [`prepare-klein-9b-4bit`](pipelines/prepare-klein-9b-4bit.vpipeline) | [`prepare-klein-9b-kv-4bit`](pipelines/prepare-klein-9b-kv-4bit.vpipeline) |
| **Edit a photo** — photo + prompt in, edited photo out. Carries a [saved Composer view](#the-saved-composer-view). | [`klein-ref-edit`](pipelines/klein-ref-edit.vpipeline) | [`klein-kv-ref-edit`](pipelines/klein-kv-ref-edit.vpipeline) |
| **Combine two references** — two pictures + prompt in, one picture out. | [`klein-multi-ref`](pipelines/klein-multi-ref.vpipeline) | [`klein-kv-multi-ref`](pipelines/klein-kv-multi-ref.vpipeline) |

All files end in `.vpipeline`. The two-reference example reads two pictures
that ship with this repo:
[`minimax-h3-reference-subject.jpg`](images/minimax-h3-reference-subject.jpg)
(shared with the [MiniMax H3](MINIMAX-H3.md) walkthrough) and
[`klein-multi-ref-panda.jpg`](images/klein-multi-ref-panda.jpg).

Follow a link and use **Raw ▸ Save as** to download it, or take them straight
from `docs/pipelines/` and `docs/images/` in your clone.

## Do you need to quantize?

**No — not even on a 16 GB Mac.** The prepare pipelines quantize to 4-bit
because a ~12 GB model is quick to load and leaves room on a small machine,
but vpipe runs the published **bf16** checkpoint as it is.

When the bf16 model does not fit in memory, vpipe **streams** it: the
transformer's blocks and the prompt encoder's layers are read from disk as
they are needed instead of being held in RAM all at once. It decides this
per run, from the machine's memory, with nothing to configure. With the model
on the Mac's **internal SSD**, streaming often costs no extra time at all; an
external drive reads slower, and that does show.

What bf16 buys is quality. It **follows the details of a prompt more
closely**, and often gives **finer detail in the output image** than the
4-bit model.

### Running bf16: one download, straight from the terminal

With nothing to quantize, preparing the model is a single `model-fetch`
stage, so there is no pipeline file to edit: launch the stage directly with
`--launch-stage`. It downloads ~32 GB, registers the model, and exits. Run it
from your work directory (see [Step 1](#first-choose-a-work-directory)), and
keep that directory on the internal SSD.

**Where `/path/to/vpipe` comes from.** If you built vpipe yourself, it is
`build/apps/vpipe/vpipe` under your build directory. If you installed the
**Vpipe Manager** app instead, it ships the same binary: open **Settings**,
go to the **About** section at the bottom, and under **Command-Line Tools**
click **Copy vpipe Path**. The copied path contains a space, because the app
is named `Vpipe Manager` — so **wrap it in double quotes** in the shell, or
everything after `Vpipe` is taken as a separate argument:

```sh
"/Applications/Vpipe Manager.app/Contents/Helpers/vpipe" --version
```

The same row has **Copy vpipe-web-ui Path** for the web UI binary, which
Step 2 uses; it needs the same quoting.

**From Hugging Face.** The repos are gated: accept the licence and create an
access token first, as in steps 1 and 2 of
[The models are gated](#the-models-are-gated). Then hand the token to the
stage in its `hf_token` config key:

```sh
cd ~/vpipe-work                                    # your work directory
/path/to/vpipe --launch-stage model-fetch \
  --stage-cfg model_path=black-forest-labs/FLUX.2-klein-9B \
  --stage-cfg hf_token=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**From ModelScope** (modelscope.cn — for users in mainland China, where
huggingface.co is not reachable). Both checkpoints are mirrored there under
the same names —
[klein-9B](https://modelscope.cn/models/black-forest-labs/FLUX.2-klein-9B) and
[klein-9b-kv](https://modelscope.cn/models/black-forest-labs/FLUX.2-klein-9b-kv)
— and the mirrors are **not gated**, so no account or token is needed:

```sh
cd ~/vpipe-work                                    # your work directory
/path/to/vpipe --launch-stage model-fetch \
  --stage-cfg model_path=black-forest-labs/FLUX.2-klein-9B \
  --stage-cfg source=modelscope
```

For the -kv variant, use `model_path=black-forest-labs/FLUX.2-klein-9b-kv` in
either command.

`model_path` is the **Hugging Face** name even when downloading from
ModelScope: `source` changes only where the bytes come from. The model
registers under the same key and lands in the same directory either way, so
nothing downstream changes. (A machine that always fetches from ModelScope
can `export VPIPE_MODEL_SOURCE=modelscope` once instead of passing `source`.)

`Ctrl-C` stops the download cleanly, and running the same command again
resumes it.

Then **point `model-select` at the download**: set its `hf_dir` to
`black-forest-labs/FLUX.2-klein-9B` or `black-forest-labs/FLUX.2-klein-9b-kv`,
and skip straight to [Step 2](#step-2--edit-a-picture). Everything else in the
graphs stays as shipped — including `klein_kv`, which follows the checkpoint,
not the precision.

## Step 1 — prepare the model

### First, choose a work directory

vpipe treats **the directory you launch it from** as its workspace, and
creates its state there: `models/` for everything you download or quantize,
`data.mdb`/`lock.mdb` for the model registry, and — under `vpipe-web-ui`
only — a `sandbox/` it confines stage file I/O to.

Pick a directory on a volume with the space from
[What you need](#what-you-need), and use the **same**
directory in steps 2 and 3: the model you are about to prepare is recorded in
that directory's registry, so a run started elsewhere will not find it.

### Then run the pipeline

Take klein-9B's, the -kv variant's, or both — each is self-contained:

```sh
cd ~/vpipe-work                                    # your work directory
# from a clone: cp ~/src/vpipe/docs/pipelines/prepare-klein-9b-4bit.vpipeline .
curl -O https://raw.githubusercontent.com/tgo-app-dev/vpipe/main/docs/pipelines/prepare-klein-9b-4bit.vpipeline
# ... fill in hf_token (see above) ...
/path/to/vpipe --launch prepare-klein-9b-4bit.vpipeline
```

For the -kv variant, substitute `prepare-klein-9b-kv-4bit.vpipeline`.

The CLI suits this job: it is long, unattended and disk-bound, with nothing to
click once it starts. `Ctrl-C` stops it cleanly, and `skip_existing_files`
means re-running picks up where it left off.

Four stages, in order:

1. **`model-fetch`** — pulls the repo into `./models`. This is where a token
   problem shows up, as an authorization failure rather than a slow download.
2. **`model-quantize`** (`target: dit`) — the 9B transformer to 4-bit, group
   size 64, with **AWQ** activation-aware smoothing and its paired clip search
   (`awq: true`, `awq_clip: true`). AWQ searches a per-layer scale so the
   weights that matter to the activations survive rounding; it costs time
   during preparation and nothing at runtime. The -kv pipeline also sets
   `klein_kv: true` here — calibration conditions on a reference image, and
   calibrating that checkpoint under the plain joint attention would clip
   against the wrong activations. klein-9B's pipeline leaves it unset, for the
   same reason in reverse.
3. **`model-quantize`** (`target: text_encoder`) — the 8B Qwen3 prompt encoder
   to 4-bit. Its output — `local/FLUX.2-klein-9B-4bit` or
   `local/FLUX.2-klein-9b-kv-4bit` — is a **complete model**: the quantized
   parts plus everything untouched.
4. **`model-remove`** — deletes the intermediate from step 2, which exists
   only to feed step 3.

When it finishes, the 4-bit model (~12 GB) is the only thing you need; the
~32 GB download can go.

## Step 2 — edit a picture

```sh
cd ~/vpipe-work                                    # the SAME work directory
/path/to/vpipe-web-ui
```

Put your photo in the sandbox as **`sandbox/reference.jpg`** — that is the
name the shipped `load-image` stage reads, and the pipeline will fail without
it. Then open the URL the server prints, load `klein-ref-edit.vpipeline` (or
`klein-kv-ref-edit.vpipeline` for the -kv variant), edit the `text-prompt`
stage, and press Start. The shipped prompt:

> *Paint the reference picture in Claude Monet's impressionist style. Use
> low-saturation shades of blue and yellow colors. Use fine horizontal strokes
> to paint the water. Write "T-Go" as the artist name in handwriting.*

Nine stages, and a tenth on the -kv variant:

```
text-prompt ──> diffusion-conditioner ──┐ prompt
                                        ├──> generate-image ──> vae-decode ─┬─> save-image
load-image ─┬─> image-resample ─┬─> vae-encode ──┘ reference latent (port 5) │        ^
            │                   │                                            │        │
            │                   └──> compare-image <─────────────────────────┘        │
            └── metadata (EXIF) ──────────────────────────────────────────────────────┘

model-select ──> diffusion-conditioner, vae-encode, generate-image, vae-decode

-kv only:  flux2-model-config (klein_kv: true) ──> generate-image model_config (port 7)
```

Two things about this graph are worth understanding, because they are what
makes it an *edit* rather than a fresh generation:

**The reference reaches the DiT as a latent, not as pixels.** `vae-encode`
turns the resampled photo into a latent that `generate-image` takes on its
reference port. On the -kv variant, those are the tokens the KV cache is built
from. The prompt travels the other path, through `diffusion-conditioner`.

**`image-resample` sets the output size, on its own.** It crops to 512 × 512
before encoding, and because `generate-image` has **neither** `width` nor
`height` configured, it infers both from the reference latent. So this one
stage is the whole geometry control: change it to 768 × 768 and you get a
768 × 768 edit, with nothing else to keep in sync. (Set `width` and `height`
on `generate-image` and they win — but set **both**. Inference is all-or-
nothing: give it one axis and the other silently falls back to 256 rather
than being inferred, which produces a strangely shaped picture and no
error.)

Two conveniences at the end: `compare-image` receives the *original* (from
`image-resample`, so it is the same crop the model saw) and the *result*, and
pairs them for the Composer panel below; and `save-image` takes `load-image`'s
**metadata** port alongside the picture, so the source photo's EXIF is carried
into the edited file.

At 4-bit the whole model is ~12 GB — a 5.4 GB transformer, a 6 GB prompt
encoder, a small VAE — which does not leave much of a 16 GB Mac. vpipe sizes
this per run: `unload_when_idle` is unset here, so it resolves to `auto` and
each stage decides from real memory whether to drop its weights between beats.
It logs what it chose. On a larger machine everything simply stays resident.

## Step 3 — combine two references

The same model takes a **second** reference. Here it puts the cat from one
picture beside the panda from another, painted in the second picture's style.

Copy the two pictures into the sandbox under the names the pipeline reads:

```sh
cd ~/vpipe-work
# from a clone: cp ~/src/vpipe/docs/images/<name>.jpg sandbox/
curl -o sandbox/minimax-h3-reference-subject.jpg \
  https://raw.githubusercontent.com/tgo-app-dev/vpipe/main/docs/images/minimax-h3-reference-subject.jpg
curl -o sandbox/klein-multi-ref-panda.jpg \
  https://raw.githubusercontent.com/tgo-app-dev/vpipe/main/docs/images/klein-multi-ref-panda.jpg
```

Then load `klein-multi-ref.vpipeline` (or `klein-kv-multi-ref.vpipeline`) and
press Start. The shipped prompt:

> *The orange tabby cat wearing the navy velvet jacket from image 1 sits
> beside the giant panda from image 2 on a mossy rock in a misty bamboo
> forest. The panda offers the cat a bamboo stalk. Keep the cat's face, fur
> pattern and jacket exactly as in image 1. Paint the whole scene in the soft
> watercolor style of image 2.*

The graph is Step 2's with the reference chain doubled:

```
text-prompt ──> diffusion-conditioner ──────────────────────────┐ prompt
                                                                ├──> generate-image ──> vae-decode ──> save-image
load-image-1 ──> image-resample-1 ──> vae-encode-1 ─────────────┤ image 1 (port 5)
load-image-2 ──> image-resample-2 ──> vae-encode-2 ─────────────┘ image 2 (port 6)

model-select ──> diffusion-conditioner, vae-encode-1, vae-encode-2, generate-image, vae-decode

-kv only:  flux2-model-config (klein_kv: true) ──> generate-image model_config (port 7)
```

What to know about it:

**The prompt names the pictures by port.** The references reach the DiT only
as latents — klein's prompt encoder is text-only and never sees them — so the
model tells them apart by position: each reference gets its own band in the
DiT's position encoding, in port order. *Image 1* is whatever is wired to
`generate-image`'s port 5 (`ref_latent0`), *image 2* whatever is on port 6
(`ref_latent1`). Swap the wiring and you must swap the words.

**Two is the most `generate-image` takes.** It has two reference ports.

**The output size is set on `generate-image`.** With two references there is
no single source photo to take a size from, so this graph sets `width: 768`
and `height: 512` — a landscape frame for two subjects side by side — while
each reference is cropped to 512 × 512. Without them, the output would take
its size from image 1 alone. Set both axes or neither, as in Step 2.

**This is where the -kv variant pays most.** A 512 × 512 reference is 1,024
tokens, so two of them are 2,048 — more than the 1,536 tokens of the 768 × 512
picture being generated. klein-9B carries all of them through every block of
every step; the -kv variant computes them once.

There is no `compare-image` here: with two sources there is no single original
to wipe against. The result is saved as `klein-multi-ref.jpeg` (or
`klein-kv-multi-ref.jpeg`) in the sandbox.

## The settings worth knowing

| stage | key | shipped | notes |
|---|---|---|---|
| `generate-image` | `steps` | 4 | Both checkpoints are guidance-distilled. 4 is the recipe, not a corner cut. |
| `flux2-model-config` | `klein_kv` | `true` in the -kv graphs; the stage is absent from klein-9B's | **Required** for the -kv checkpoint, wrong for klein-9B. See [the note above](#two-checkpoints-klein-9b-and-klein-9b-kv). Wired to `generate-image`'s `model_config` iport. |
| `generate-image` | `i8_gemm` | `true` | Dynamic-int8 GEMMs for the DiT's big matmuls — about **2×** their f16 rate, and **lossy**. Ignored on a GPU without NAX matrix cores, so it is safe to leave on. Turn it off to compare quality. |
| `generate-image` | `width`/`height` | unset (edit), 768 × 512 (two references) | Unset, the output takes the size of the reference on port 5. Set both or neither. |
| `image-resample` | `width`/`height` | 512 × 512 | The size each reference is encoded at — and, in the edit graphs, the **output** size too. `fit: crop` fills the frame; `pad` would letterbox grey into the reference. |
| `model-select` | `hf_dir` | `local/FLUX.2-klein-9B-4bit` / `local/FLUX.2-klein-9b-kv-4bit` | Names the model once; the conditioner, VAEs and DiT all latch it. Must agree with `klein_kv`. For bf16, the download's key: `black-forest-labs/FLUX.2-klein-9B` / `black-forest-labs/FLUX.2-klein-9b-kv`. |

Paths in `load-image` and `save-image` are relative to the sandbox — under the
web UI, the edit lands at `sandbox/klein-edit.jpeg` (or
`sandbox/klein-kv-edit.jpeg`) in the work directory.

## Watching it form — live previews

Any klein graph can show the picture forming, step by step. `generate-image`'s
port **2** takes the model's current guess at the finished image after each
step and decodes it with madebyollin's **TAEF2**, a tiny autoencoder
trained for FLUX.2's latent space. A `preview` stage wired there shows each
guess as it arrives. Two edits turn it on:

1. On `flux2-model-config`, set `preview_vae` to `madebyollin/taef2`. The -kv
   graphs already have this stage; in a klein-9B graph, add one and wire it
   to `generate-image`'s `model_config` iport. Fetch the TAE once, with a
   `model-fetch` stage whose `model_path` is `madebyollin/taef2` (10 MB).
2. Add a `preview` stage whose input is `generate-image`, port 2.

`preview_every` renders every *N*th step (the last always renders), and
`preview_max_edge` (512 by default) scales the picture down before it is
sent. Leave `preview_vae` empty, or port 2 unwired, and nothing is loaded.

Measured on FLUX.2-klein-4B at 1024 × 1024, 4 steps, on an M4 Pro:

- **No measurable cost:** 21.2 s with a preview on every step, 21.0 s
  without.
- A preview arrived 1.2–1.8 s after its step, queued behind the DiT; the
  last one, with the GPU free, took 0.3 s.
- The last step's preview matched the real VAE decode at **28.2 dB**.
  TAEF2 is a coarser decoder than the full FLUX.2 VAE, so treat the
  preview as a preview. The file on port 0 is still decoded by the real
  VAE.

TAEF2 is fed the DiT's latent unpatchified, but *not* un-normalized. The
real VAE's batch-norm inverse is exactly what the TAE was distilled
without.

## The saved Composer view

`klein-ref-edit.vpipeline` and `klein-kv-ref-edit.vpipeline` carry more than a
graph. Composer arrangements are stored **in the pipeline file** — a top-level
`aux.composer` object the pipeline core ignores and the web UI reads — so a
pipeline can travel with the dashboard you want to watch it through. These
ship a **wipe comparison** of the original against the result, above a
**pipeline editor**.

### Loading it

The arrangement is restored on demand, not automatically, and it is keyed to
the pipeline — so load the pipeline first:

1. **Pipeline Manager** ▸ **Load** ▸ `klein-ref-edit.vpipeline`.
2. Switch to the **Composer** view.
3. **Load** ▸ **Load for pipeline…** ▸ pick **`klein-ref-edit`**.

(For the -kv variant, `klein-kv-ref-edit` in both places.)

It says *Layout loaded*, and the panels appear. The compare panel shows
*waiting* until you start the pipeline, then connects by itself — its
designation (`klein-ref-edit` / `compare-image`) is saved with the layout, so
it knows which stage's output it is for without being pointed at one.

Drag across the image to wipe between the original and the edit. The two views
keep their zoom and pan in sync, which is what makes a small change — a
signature, a colour shift — visible at all. The panel's ⋯ menu switches
between wipe, side-by-side and A/B-only.

### Saving your own

The reverse of the above: arrange the panels you want, then **Save** ▸ **Save
with pipeline…** ▸ pick the pipeline ▸ confirm the path. That rewrites the
`.vpipeline` file with your arrangement bundled in. **Load from file… / Save
to file…** on the same menus handle standalone layout JSON instead, for an
arrangement you want to reuse across different pipelines.

## Troubleshooting

**`model-fetch` reports an authorization failure.** All three steps under
[The models are gated](#the-models-are-gated) are required, and the one most
often missed is the fine-grained token's gated-repo permission. The licence is
accepted per repo, so accepting klein-9B's does not open klein-9b-kv. And
accepting it in a browser you are signed into does not help a token belonging
to a different account.

**The edit ignores the reference.** Check `vae-encode` is wired to
`generate-image`'s reference port and that `image-resample` runs between
`load-image` and `vae-encode`.

**The result is badly wrong after switching checkpoints.** `klein_kv` and the
model must agree: the -kv checkpoint needs the `flux2-model-config` stage with
`klein_kv: true`, and klein-9B must not have it. Changing only `model-select`'s
`hf_dir` runs one checkpoint under the other's attention.

**The -kv result is a little worse than klein-9B's.** That is the variant's
trade-off, not a fault. Load the klein-9B version of the same graph for that
picture.

**The output looks degraded on both checkpoints.** Try `i8_gemm: false` first —
it is a lossy speed mode. If that is not it, the 4-bit quantization is the
other knob: re-prepare at `bits: 8` for a larger, closer model.

**Only one reference seems to count.** Check that both `vae-encode` stages
reach `generate-image` — one on port 5, one on port 6 — and that the prompt
refers to both, as *image 1* and *image 2*.

**The picture is a strange shape.** `width` or `height` is set on
`generate-image` without the other. Set both, or neither.

**The Composer says “No saved arrangement”.** The pipeline is loaded but has
no `aux.composer` — you picked a different pipeline, or one saved without a
layout. The two-reference pipelines ship without one.

## Under the hood

- **klein-9B**: the token order is `[text, image, refs]` — references
  appended after the generated tokens — and every token attends to every
  other. Reference *i* sits at position-encoding time coordinate 10·(*i*+1),
  the generated picture at 0.
- **klein-9b-kv**: the token order is `[text, refs, image]` — references
  **before** the generated tokens.
- Its reference tokens self-attend only, and are modulated at a fixed
  timestep 0. That is what makes their K/V constant across the denoise, and
  cacheable.
- The parameter shapes are byte-identical to klein-9B, so the loader,
  quantizer and LoRA fusion needed nothing new — only the forward pass and
  the mask differ.
- On **M5**, the GEMMs and attention run on the GPU's matrix cores
  (`matmul2d` / NAX flash attention); `i8_gemm` rides on the same hardware.

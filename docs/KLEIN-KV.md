# Reference image editing with FLUX.2-klein-9b-kv

**FLUX.2-klein-9b-kv** takes a photo and a sentence and gives you the photo
edited. vpipe runs it on-device through its **metal-compute** backend — its
own Metal kernels, no Python and no third-party tensor runtime in the forward
pass — quantized to 4-bit, in **4 steps**, on a 16 GB Mac.

The `-kv` in the name is the interesting part. In plain klein-9B the reference
image joins the joint attention and is recomputed at every denoise step. This
checkpoint is distilled so that **reference tokens attend only to themselves**
— they see neither the prompt nor the image being generated — which makes
their K/V independent of the timestep, so vpipe computes them **once** at step
0 and reuses them for the rest. BFL measure **1.21–2.66×**, the larger end
with several references at modest output sizes.

> **`klein_kv` is not an optimization flag.** The two checkpoints are
> indistinguishable on disk — same `config.json`, same tensor names, same
> shapes — so vpipe cannot detect which one you have. The token order and
> attention mask differ, and the weights are distilled *for* that mask, so
> running one checkpoint through the other's forward pass is not slower, it
> is **wrong**. Both shipped pipelines set `klein_kv: true`; leave it set for
> this checkpoint and unset for any other.

## What you need

| | |
|---|---|
| **Machine** | Apple Silicon Mac (M-series). |
| **Memory** | 16 GB. |
| **Disk** | **~45 GB** to prepare, **~12 GB** to keep. |
| **Build** | An Apple Silicon build of vpipe — the default on arm64 macOS. See the main [README](../README.md). |
| **Hugging Face** | An account, the licence accepted, and an **access token**. This model is gated — see [below](#the-model-is-gated). |

The published checkpoint is bf16; vpipe quantizes it to 4-bit once, up front.

| | |
|---|---|
| Download (`black-forest-labs/FLUX.2-klein-9b-kv`, bf16) | **~32 GB** |
| Peak while preparing (download + output) | **~45 GB** |
| 4-bit model, download deleted | **~12 GB** |

The repo itself is ~49 GB, but vpipe's catalogue entry pins the diffusers
subfolders and **skips the redundant top-level copy of the transformer**
(~17 GB of the same weights) and the sample images. You download ~32 GB.

## The model is gated

`black-forest-labs/FLUX.2-klein-9b-kv` is behind the **FLUX Non-Commercial
License**. An unauthenticated fetch does not fall back to anything — it
**401s** — so all three of these must be done before the pipeline will
download anything:

1. **Accept the licence.** Sign in to Hugging Face, open
   [the model page](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv),
   and accept the terms. Do this first: a token cannot grant access you have
   not accepted.
2. **Create an access token** at
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
   A **Read** token is enough. If you create a fine-grained token instead,
   it must carry **“Read access to contents of all public gated repos you can
   access”** — a fine-grained token without that permission looks valid and
   still 401s on this repo, which is a confusing way to lose an hour.
3. **Put the token in the pipeline.** Open
   `prepare-klein-9b-kv-4bit.vpipeline` and fill in the `model-fetch` stage's
   empty `hf_token`:

   ```json
   {
     "id": "fetch",
     "type": "model-fetch",
     "config": {
       "model_path": "black-forest-labs/FLUX.2-klein-9b-kv",
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

## The two pipelines

- **[`prepare-klein-9b-kv-4bit.vpipeline`](pipelines/prepare-klein-9b-kv-4bit.vpipeline)**
  — download the checkpoint and quantize it. Run once.
- **[`klein-kv-ref-edit.vpipeline`](pipelines/klein-kv-ref-edit.vpipeline)**
  — photo + prompt in, edited photo out. **Carries a saved Composer view**
  — see [The saved Composer view](#the-saved-composer-view).

Follow a link and use **Raw ▸ Save as** to download it, or take them straight
from `docs/pipelines/` in your clone.

## Step 1 — prepare the model

### First, choose a work directory

vpipe treats **the directory you launch it from** as its workspace, and
creates its state there: `models/` for everything you download or quantize,
`data.mdb`/`lock.mdb` for the model registry, and — under `vpipe-web-ui`
only — a `sandbox/` it confines stage file I/O to.

Pick a directory on the volume with the **~45 GB**, and use the **same**
directory in step 2: the model you are about to prepare is recorded in that
directory's registry, so a run started elsewhere will not find it.

### Then run the pipeline

```sh
cd ~/vpipe-work                                    # your work directory
cp ~/src/vpipe/docs/pipelines/prepare-klein-9b-kv-4bit.vpipeline .
# ... fill in hf_token (see above) ...
~/src/vpipe/build/apps/vpipe/vpipe --launch prepare-klein-9b-kv-4bit.vpipeline
```

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
   during preparation and nothing at runtime. `klein_kv: true` matters here
   too — calibration conditions on a reference image, and calibrating this
   checkpoint under the plain joint attention would clip against the wrong
   activations.
3. **`model-quantize`** (`target: text_encoder`) — the 8B Qwen3 prompt encoder
   to 4-bit. Its output, `local/FLUX.2-klein-9b-kv-4bit`, is a **complete
   model**: the quantized parts plus everything untouched.
4. **`model-remove`** — deletes the intermediate from step 2, which exists
   only to feed step 3.

When it finishes, `local/FLUX.2-klein-9b-kv-4bit` (~12 GB) is the only thing
you need; the ~32 GB download can go.

## Step 2 — edit a picture

```sh
cd ~/vpipe-work                                    # the SAME work directory
~/src/vpipe/build/apps/web-ui/vpipe-web-ui
```

Put your photo in the sandbox as **`sandbox/reference.jpg`** — that is the
name the shipped `load-image` stage reads, and the pipeline will fail without
it. Then open the URL the server prints, load `klein-kv-ref-edit.vpipeline`,
edit the `text-prompt` stage, and press Start. The shipped prompt:

> *Paint the reference picture in Claude Monet's impressionist style. Use
> low-saturation shades of blue and yellow colors. Use fine horizontal strokes
> to paint the water. Write "T-Go" as the artist name in handwriting.*

Ten stages:

```
text-prompt ──> diffusion-conditioner ──┐ prompt
                                        ├──> generate-image ──> vae-decode ─┬─> save-image
load-image ─┬─> image-resample ─┬─> vae-encode ──┘ reference latent (port 5) │        ^
            │                   │                                            │        │
            │                   └──> compare-image <─────────────────────────┘        │
            └── metadata (EXIF) ──────────────────────────────────────────────────────┘

model-select ──> diffusion-conditioner, vae-encode, generate-image, vae-decode
```

Two things about this graph are worth understanding, because they are what
makes it an *edit* rather than a fresh generation:

**The reference reaches the DiT as a latent, not as pixels.** `vae-encode`
turns the resampled photo into a latent that `generate-image` takes on its
reference port, and those are the tokens the KV cache is built from. The
prompt travels the other path, through `diffusion-conditioner`.

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

### The settings worth knowing

| stage | key | shipped | notes |
|---|---|---|---|
| `generate-image` | `steps` | 4 | The checkpoint is guidance-distilled. 4 is the recipe, not a corner cut. |
| `generate-image` | `klein_kv` | `true` | **Required** for this checkpoint, wrong for any other. See the note at the top. |
| `generate-image` | `i8_gemm` | `true` | Dynamic-int8 GEMMs for the DiT's big matmuls — about **2×** their f16 rate, and **lossy**. Ignored on a GPU without NAX matrix cores, so it is safe to leave on. Turn it off to compare quality. |
| `image-resample` | `width`/`height` | 512 × 512 | Sets the **output** size too, since `generate-image` infers it from the reference latent. `fit: crop` fills the frame; `pad` would letterbox grey into the reference. |
| `model-select` | `hf_dir` | `local/FLUX.2-klein-9b-kv-4bit` | Names the model once; the conditioner, VAEs and DiT all latch it. |

The shipped `save-image` path is `klein-kv-edit.jpeg`, relative to the
sandbox — under the web UI it lands at `sandbox/klein-kv-edit.jpeg` in the
work directory.

## The saved Composer view

`klein-kv-ref-edit.vpipeline` carries more than a graph. Composer arrangements
are stored **in the pipeline file** — a top-level `aux.composer` object the
pipeline core ignores and the web UI reads — so a pipeline can travel with the
dashboard you want to watch it through. This one ships a **wipe comparison**
of the original against the result, above a **pipeline editor**.

### Loading it

The arrangement is restored on demand, not automatically, and it is keyed to
the pipeline — so load the pipeline first:

1. **Pipeline Manager** ▸ **Load** ▸ `klein-kv-ref-edit.vpipeline`.
2. Switch to the **Composer** view.
3. **Load** ▸ **Load for pipeline…** ▸ pick **`klein-kv-ref-edit`**.

It says *Layout loaded*, and the panels appear. The compare panel shows
*waiting* until you start the pipeline, then connects by itself — its
designation (`klein-kv-ref-edit` / `compare-image`) is saved with the layout,
so it knows which stage's output it is for without being pointed at one.

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
[The model is gated](#the-model-is-gated) are required, and the one most often
missed is the fine-grained token's gated-repo permission. Accepting the
licence in a browser you are signed into does not help a token belonging to a
different account.

**The edit ignores the reference.** Check `vae-encode` is wired to
`generate-image`'s reference port and that `image-resample` runs between
`load-image` and `vae-encode`.

**The output looks degraded.** Try `i8_gemm: false` first — it is a lossy
speed mode. If that is not it, the 4-bit quantization is the other knob:
re-prepare at `bits: 8` for a larger, closer model.

**The Composer says “No saved arrangement”.** The pipeline is loaded but has
no `aux.composer` — you picked a different pipeline, or one saved without a
layout.

## Under the hood

- Token order is `[text, refs, image]` — references **before** the generated
  tokens, where plain klein-9B appends them after.
- Reference tokens self-attend only, and are modulated at a fixed timestep 0.
  That is what makes their K/V constant across the denoise, and cacheable.
- The parameter shapes are byte-identical to klein-9B, so the loader,
  quantizer and LoRA fusion needed nothing new — only the forward pass and
  the mask differ.
- On **M5**, the GEMMs and attention run on the GPU's matrix cores
  (`matmul2d` / NAX flash attention); `i8_gemm` rides on the same hardware.

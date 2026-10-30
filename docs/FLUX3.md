# FLUX.3 Image Klein: text-to-image and editing

**FLUX.3 Image Klein** (Black Forest Labs) takes a caption — and, if you
give it some, up to ten reference pictures — and returns one picture. The
same model does text-to-image, single-reference edits and multi-reference
compositions. vpipe runs the published **bf16** checkpoint on-device through
its own Metal kernels, in the **16 steps** the distilled checkpoint is built
for, with no classifier-free guidance pass.

| | |
|---|---|
| **Denoiser** | a 5.6B flow-map DiT (11.1 GB bf16) |
| **Text encoder** | Qwen3-VL-4B-Instruct, used for text only (8.9 GB) |
| **VAE** | FLUX.2's, unchanged |
| **Steps** | 16, no CFG (the checkpoint's own default) |
| **Licence** | FLUX Kommunity License — non-commercial, with an output exception for small businesses; read it on the model page |

## Getting the model

The Hugging Face repo is **gated**: accept the licence on
[the model page](https://huggingface.co/black-forest-labs/flux-3-image-klein),
create a **Read** access token, and put it in the `model-fetch` stage of
[`prepare-flux3-klein`](pipelines/prepare-flux3-klein.vpipeline) (or in the
`hf_token` field of the stage in the web UI, which keeps the secret out of a
file). The token rules are the same as FLUX.2's — see
[The models are gated](KLEIN.md#the-models-are-gated).

The repository is 43 GB, but the catalogue entry fetches **~21 GB**: the
diffusers `transformer/`, `text_encoder/` and `vae/`, plus the root
`config.json`. It skips BFL's own copies of the same transformer and VAE at
the repo's root and the `variants/` folder (an FP8 and a Blackwell-only
NVFP4 transformer). The root `config.json` matters: it is where the
checkpoint states its sampler (`alpha` 6.93, 16 steps) and its prompt cap.

You do not need to quantize. On a machine that cannot hold the DiT beside
the encoder, vpipe **streams** the DiT's blocks from disk, deciding per run
from the machine's memory.

## The pipelines

| | |
|---|---|
| [`prepare-flux3-klein`](pipelines/prepare-flux3-klein.vpipeline) | download (fill in `hf_token`) |
| [`flux3-klein-text-to-image`](pipelines/flux3-klein-text-to-image.vpipeline) | caption in, 1024x1024 picture out |
| [`flux3-klein-edit`](pipelines/flux3-klein-edit.vpipeline) | photo + caption in, edited photo out, with a compare view |

For an edit, the reference goes through `vae-encode` and into
`generate-image`'s `ref_latent0`; more references go on `ref_latent1` or the
`ref_latents` list, in order. Leave `width`/`height` unset and the output
takes the first reference's size.

The model wants sides that are multiples of 16, and the edit graph fits the
photo the way BFL's own pipeline does, with `image-resample`'s
`fit: pad_to_grid`:

- it scales the photo down, never up, keeping its aspect, until the photo
  rounded up to multiples of 16 is at most 4 MP;
- it centres the photo on that canvas and mirror-pads the few pixels
  around it;
- the edit is generated on the canvas, and `vae-decode` cuts the result
  back to the photo's own rectangle.

A 1000x750 photo comes back as 1000x750, not as a 1024x1024 crop. Set
`width`/`height` on `generate-image` and you get that size instead, with
nothing cut. For several references, BFL fits each under 1 MP and crops it
to the grid instead: that is `fit: crop_to_grid` with
`max_pixels: 1048576`.

## Prompting

The model wants a **full scene caption** — framing, subject, background,
lighting. Short prompts give generic pictures, and a one-line edit
instruction ("make it night") tends to reframe the shot; for an edit, write
the caption of the picture you want, with the change folded in. BFL's own
CLI runs a separate 9B "prompt upsampler" to write such captions from short
requests; vpipe does not include it, so the caption is yours to write (any
chat model can do it).

Bounding boxes work as BFL documents them — they are part of the caption
text, a JSON list after it on a 0–1000 grid — because they are read by the
model, not by the pipeline.

## Settings

| Where | Key | |
|---|---|---|
| `generate-image` | `seed` | a seed reproduces BFL's own reference sampler's noise bit for bit |
| `generate-image` | `steps` | default: the checkpoint's 16 |
| `generate-image` | `ane_ffn` | the joint blocks' feed-forward on the ANE — see below |
| `generate-image` | `sol_attn` | approximate sparse attention — see below |
| `flux3-model-config` | `alpha` | the schedule's shape, `t = a s / (1 + (a - 1) s)`; default the checkpoint's 6.93 |
| `flux3-model-config` | `lora` .. `lora8` | runtime LoRA, live strengths; diffusers module names |
| `flux3-model-config` | `preview_vae` | a live per-step preview; FLUX.2's `madebyollin/taef2` previews FLUX.3 unchanged |

## Speed

MEASURED on an M4 Pro (64 GB), 1024x1024, 16 steps, every DiT block resident:

| | denoise | vs dense |
|---|---|---|
| dense (bf16) | 100.0 s | 1.00x |
| `ane_ffn` | 62.8 s | 1.59x |
| `sol_attn` | 99.1 s | 1.01x |
| `ane_ffn` + `sol_attn` | 58.9 s | 1.70x |

An edit with one 1024x1024 reference doubles the sequence: 225 s dense.

**`ane_ffn` is the lever on the M4 series**, and pays more here than on the
other families: FLUX.3's blocks are *parallel* — one norm feeds both the
attention and the feed-forward — so the ANE's share of the feed-forward runs
under the GPU's whole attention rather than only its own half. The ANE
picture is 54 dB from the GPU-only one. `sol_attn` is approximate (28 dB with
both on): attention is only 17% of a 1024x1024 step, so on its own it buys
little there. It earns its keep at larger sizes, where attention grows
quadratically: at 2048x2048 attention is 45% of a step and `sol_attn` takes
a step from 36.6 s to 29.4 s (1.24x, measured on the same M4 Pro).

Do not judge `ane_ffn` on a fanless MacBook Air: there the ANE and the GPU
share one power budget and do not truly overlap.

**On a 16 GB machine** the same graph streams. MEASURED on an M5 MacBook
Air (16 GB, fanless), 1024x1024, 16 steps, dense: the DiT's blocks and the
text encoder's layers stream from the internal SSD, the GPU's matrix cores
take the GEMMs and the attention, and the denoise takes **64.5 s** (73.7 s
for the whole graph) with the process peaking at 4.8 GB and nothing pushed
to swap. The picture is the M4 Pro's, to rounding (32.7 dB).

On the M5, `i8_gemm` is the tier worth having: 1.12x on the denoise at
1024x1024 for a picture that differs from the dense one only in fine
texture. `sage_attn` engages but buys 1.02x there -- the attention kernel
already runs at the GPU's rate, and attention is a small share at that
size.

## Up to 4K

The model generates up to 4096x4096 (any side from 64 to 4096), as BFL's
own sampler allows. It is slow -- a 4096x4096 sequence is 16 times a
1024x1024 one, and attention grows with its square -- and MEASURED at 16
steps:

| | denoise | peak memory |
|---|---|---|
| M4 Pro 64 GB, `ane_ffn` | 87 min | 20.3 GB |
| M5 MacBook Air 16 GB, streamed | 110 min (throttled from 4.7 to 6.9 min a step) | 9.7 GB |

At this size the VAE decode does not fit whole on either machine: a whole
4096x4096 decode would hold ~57 GB. So it decodes in tiles, and the tiles
reproduce the whole decode exactly.

The decoder has two global steps, and both are computed over the whole
picture:

- **the mid block's attention**, which runs at the latent's size, 64 times
  smaller than the picture;
- **every GroupNorm's statistics**, gathered by passes over the tiles that
  only sum.

Everything else is local, and a tile with enough border around it is the
whole decode's. The decode sizes its tiles to the memory that is free when
it starts. It keeps whole every intermediate plane that fits, so a later
pass recomputes only back to it, and it stays inside that memory without
pushing anything to swap. MEASURED on one 4096x4096 latent:

| | memory it used | decode | extra work |
|---|---|---|---|
| M4 Pro 64 GB | 15.3 GB | 72 s | 1.21x |
| M5 MacBook Air 16 GB, straight after the denoise | 7.3 GB | 119 s | 1.73x |

"Extra work" is the work against a whole decode, which neither machine
could run. The M5's decode shrank swap by 248 MB.

The two pictures agree to 66.6 dB (at most 2 of 255 levels apart).
Checked at 2048x2048, where a whole decode does fit: the tiled decode in
3 GB agrees with it to 68.8 dB (at most 1 level). Decoding each window on
its own instead (`VPIPE_VAE_WINDOWED=1`, kept for comparison) is 29.4 dB
from it and shifts each window's tone, which reads as a colour cast and
patches in smooth areas.

## Quantized packs

`model-quantize` packs the DiT at 8 or 4 bits (the six block projections;
embedders, modulation and output layer stay bf16). On an M4 Pro the packs
are a **footprint** option, not a speed one — the GPU has no matrix cores,
and the bf16 tiles are its fastest path:

| | DiT | denoise, 1024x1024 | picture vs bf16 |
|---|---|---|---|
| bf16 | 11.1 GB | 100.0 s | — |
| w8 | 5.7 GB | 116.5 s | 48.7 dB (indistinguishable) |
| w4 | 3.3 GB | 110.3 s | 29.4 dB (same composition, visibly different detail) |

Streaming, not quantization, is what runs the bf16 model on a small box.

## Not supported

- **BFL's own single-file layout** — the root `model.safetensors` and
  `ae.safetensors` that BFL's README downloads. vpipe reads the diffusers
  layout (`transformer/`, `vae/`) the catalogue fetches; the two hold the
  same weights. Pointed at BFL's layout, every stage says so and stops.
- **The `fp8r` and `nvfp4` variants.** FP8 would save download, not memory
  or time on Apple GPUs; NVFP4 is Blackwell-only.
- **FLUX.3 Image Klein Base** (the undistilled, CFG checkpoint) is not
  wired: its 50-step CFG loop needs an empty-prompt negative this graph
  does not produce.
- **The prompt upsampler** (`flux-3-image-klein-pup`) — see Prompting.

## How it was verified

Against BFL's reference package (`src/flux3`) and diffusers' port, each
stage held to the reference's own bf16-vs-f32 error:

| | vpipe vs the f32 reference | the reference's own bf16 |
|---|---|---|
| text conditioning (real rows) | 0.0069 | 0.0077 |
| one DiT step, teacher-forced, mean over 16 | 0.0143 | 0.0146 |
| 16 steps at 256x256 | 0.0287 | 0.0313 |
| 16-step edit at 256x256 | 0.0515 | 0.0655 |
| VAE decode / encode | 0.0018 / 0.0018 | — |

Measured on an M4 Pro; on an M5 the per-step figures are the same
(teacher-forced means 0.0142 / 0.0162 for text-to-image / edit, the
reference's 0.0146 / 0.0165). A whole 16-step trajectory is a noisier
figure than a step: kernels equally accurate per step end it anywhere from
0.8x to 1.3x the reference's own error, so the per-step test is the one to
read.

The noise for a seed, the schedule's 17 nodes and the tokenizer's ids are
bit-identical to the reference's. Tests: `flux3_dit.*`, `flux3_cond.*`,
`flux3_stage.*`, `flux3_sampler.*`; goldens from
`tools/dump_flux3_golden.py`.

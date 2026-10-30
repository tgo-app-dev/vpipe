# Fibo 2.0: image editing and text-to-image

**Fibo 2.0** (Bria) edits a picture from an instruction and generates one
from a caption. It takes one picture or several, plus a plain-language
instruction or a structured JSON caption of the result, and returns the
picture. Bria publishes it as two checkpoints of one architecture, and
vpipe runs both from the published **bf16** weights on-device through its
own Metal kernels:

| | **Edit Turbo** | **Dev** |
|---|---|---|
| Hugging Face | `briaai/Fibo-2.0-edit-turbo` | `briaai/Fibo-2.0-dev` |
| For | editing, fast | text-to-image **and** editing; the base to fine-tune |
| Steps | 4 | 50 (the model card's; 30 is the reference pipeline's default) |
| Guidance | none (1.0) | 5.0, against a negative prompt |

| | |
|---|---|
| **Denoiser** | an 8B DiT (16.3 GB bf16): 2 noise-refiner blocks and 44 main blocks; a 2-layer Perceiver squeezes the text into 256–512 *gist* tokens; five main blocks also cross-attend their own text bundle |
| **Text encoder** | Qwen3-VL-4B-Instruct (8.9 GB). It reads the instruction **and** the pictures to edit; fifteen of its hidden states, each scaled to unit RMS, become six text bundles |
| **VAE** | FLUX.2's `AutoencoderKLFlux2` |
| **Licence** | **CC BY-NC 4.0, non-commercial.** Read it on the model page |

The two checkpoints ship identical configuration files, so nothing in the
files says which one a directory holds. **The graph says it**: a Dev graph
sets `guidance_scale` on a `fibo2-model-config` stage and wires the
conditioner's negative to generate-image (see "Fibo 2.0 Dev" below). With
neither, a graph runs Edit Turbo's recipe.

## Getting the model

Both Hugging Face repos are **gated**. Accept the terms on the model page
([Edit Turbo](https://huggingface.co/briaai/Fibo-2.0-edit-turbo),
[Dev](https://huggingface.co/briaai/Fibo-2.0-dev)), create a **Read**
access token, and put it in the `model-fetch` stage of
[`prepare-fibo2-edit-turbo`](pipelines/prepare-fibo2-edit-turbo.vpipeline)
or [`prepare-fibo2-dev`](pipelines/prepare-fibo2-dev.vpipeline), or in the
`hf_token` field of the stage in the web UI, which keeps the secret out of
a file. Each download is **~26 GB**: `transformer/`, `text_encoder/`,
`processor/`, `scheduler/`, `vae/` and `fonts/`, and for Dev also
`styles/`.

You do not need to quantize. On a machine that cannot hold the DiT beside
the encoder, vpipe **streams** the DiT's blocks from disk, and it decides
this per run from the machine's memory.

## The pipeline (Edit Turbo)

[`fibo2-edit`](pipelines/fibo2-edit.vpipeline) takes a picture and an
instruction and writes the edited picture.

```
load-image -> image-resample (crop) -+-> vae-encode -----------+
                                     +-> diffusion-conditioner -+-> generate-image -> vae-decode -> save-image
text-prompt ---------------------------^
```

The **same resampled picture** has to feed both `vae-encode` and the
conditioner's `ref_image` port. The DiT sees the picture's latent, and
Qwen3-VL sees a thumbnail of the same pixels, at most 768 a side (the
conditioner makes the thumbnail itself). Leave `width` / `height` unset on
`generate-image` and the result takes the picture's size.

### Sizes

The model was trained at about one megapixel, sides multiple of 16, and its
own pipeline picks the trained size closest to the picture's aspect. Then it
scales the picture to cover that size and crops the centre. Set
`image-resample`'s `width` / `height` to the size for your picture's aspect,
with `fit: crop`:

| aspect | size |
|---|---|
| 1:1 | 1024 × 1024 |
| 4:3 / 3:4 | 1152 × 864 / 864 × 1152 |
| 3:2 / 2:3 | 1248 × 832 / 832 × 1248 |
| 16:9 / 9:16 | 1424 × 800 / 800 × 1424 |
| 21:9 | 1568 × 672 |

### The prompt

A plain instruction works as is: *"replace the jacket with a red leather
one"*. The model was trained on Bria's structured captions, and it also takes
a JSON object with an `"edit_instruction"` key and the caption of the result.
Put the caption either beside the instruction or under a
`"structured_caption"` key:

```json
{"edit_instruction": "add a straw hat",
 "structured_caption": {"short_description": "a ginger cat in a straw hat"}}
```

The conditioner builds the edit prompt exactly as the reference pipeline
does. It writes one vision marker per picture (`"image"`, or `"image_1"`,
`"image_2"` … for several), then the cleaned caption, then the instruction.
Empty values and score fields are dropped from a caption, as in training.

### Several pictures

Up to five pictures can be edited together, for example *"put the cat from
image_1 next to the panda from image_2"*. Wire them in order to the
conditioner's `ref_image`, `ref_image2` and `ref_images` ports, and the same
pictures, through `vae-encode`, to `generate-image`'s `ref_latent0`,
`ref_latent1` and `ref_latents` ports. The reference crops each picture to
the trained size for its own aspect (the table above). The result takes the
first picture's size unless `width` / `height` say otherwise.

### Masked edits

[`fibo2-masked-edit`](pipelines/fibo2-masked-edit.vpipeline) confines the
edit to a region. White on black marks the region to regenerate. The model
was trained on pictures whose masked region is greyed out: flat 128-grey
wherever the mask is at least 128, at the picture's own size, before it is
resized. `create-mask` does exactly that, headless, with these settings:

```
create-mask { interactive: false, output: "overlay",
              overlay_color: "#808080", overlay_opacity: 1.0,
              mask_threshold: 128 }
```

The picture goes on its `ref-image` port, the mask on `in-mask`. A mask PNG
can come straight from `load-image`: a picture on `in-mask` is read as its
luminance, as Pillow's `convert("L")` reads it. With `interactive: true`
you paint the mask in the web UI's mask editor instead.

### Model config

None is needed for an ordinary run. A `fibo2-model-config` stage, wired to
generate-image's `model_config` port (and the conditioner's, for the picture
bounds), carries:

- up to eight runtime LoRAs (`lora` .. `lora8`, strengths live);
- a live preview (FLUX.2's TAE, `taef2`, reads this latent space as is);
- the bounds of the pictures Qwen3-VL reads for an edit (`vl_long_edge`,
  768 by default; `vl_min_pixels` / `vl_max_pixels`).

## Fibo 2.0 Dev

Dev is the full, non-distilled model Edit Turbo is distilled from. It
generates as well as edits, and it samples with **classifier-free
guidance**: at every step the DiT runs twice, once on the prompt and once
on a negative prompt, and the step follows `v_neg + g (v - v_neg)`. Two
things in a graph turn that on, and both are needed:

- a `fibo2-model-config` stage with `guidance_scale: 5`, wired to
  generate-image's `model_config` port (generate-image's own
  `guidance_scale` works too);
- the conditioner's second output, `negative_conditioning`, wired to
  generate-image's `neg_conditioning` port. The conditioner encodes the
  negative only when that output is wired, so an Edit Turbo graph pays
  nothing for it.

Without a step count the run takes 30 steps (the reference pipeline's
default); the shipped graphs ask for the model card's 50. A guidance of 1
or below runs one forward a step whatever is wired, which is Edit Turbo's
recipe.

**The negative prompt.** Wire it to the conditioner's `negative` port.
Without one the conditioner uses the *null caption* the model was trained
with for guidance: every key of a structured caption with every value
empty. That is what the model card uses for edits. For text-to-image the
card uses a negative caption that steers away from soft, featureless,
artifact-laden pictures; [`fibo2-dev-text-to-image`](pipelines/fibo2-dev-text-to-image.vpipeline)
wires it from a second `text-prompt`. The negative is encoded exactly as
the reference encodes it: the text as written (it is not cleaned the way
a generation's caption is), and without the pictures an edit edits, even
in an edit.

**Memory.** A guided run holds two texts in the DiT, the prompt's and the
negative's: each its gist tokens and the injection blocks' keys and
values, about 5 MB for the null caption and 76 MB for a 1,024-token
negative. The activations are those of whichever pass has the longer
sequence. The memory plan books the negative when it can see the guidance
before the run starts (a `fibo2-model-config` stage's `guidance_scale`, or
generate-image's own), and corrects itself to the real lengths when the
conditioning arrives. Measured beside a 512² edit, a guided step with a
1,100-token negative allocated 387.2 MB at its peak against 387.0 MB
booked.

### Text-to-image

[`fibo2-dev-text-to-image`](pipelines/fibo2-dev-text-to-image.vpipeline):

```
text-prompt -------> diffusion-conditioner =(prompt, negative)=> generate-image -> vae-decode -> save-image
negative-prompt ----^                        fibo2-model-config --^
```

Set `width` and `height` on generate-image, multiples of 16 at about one
megapixel, the area the model was trained at: 1024 × 1024 (the default
when neither is set), 1424 × 800 (16:9), 1248 × 832 (3:2). The prompt is
best a structured JSON caption. Bria's `bria-vgl` package writes one from
a short prompt; a plain sentence works too and is used as given.

### Editing

[`fibo2-dev-edit`](pipelines/fibo2-dev-edit.vpipeline) is
[`fibo2-edit`](pipelines/fibo2-edit.vpipeline) with the negative wired,
`guidance_scale: 5` and 50 steps. Everything in "The pipeline" above
(sizes, the prompt, several pictures, masked edits) holds for Dev too.

### Styles

The Dev repo carries five photographic looks under `styles/` (Boring
Reality, Kodak Film, Hard Flash, Hazy / Dreamy, Lens Distortion). Each is a
system and a user prompt for the caption writer, which vpipe does not run,
and a `negative_prompt.json` to render with. To use a style's caption,
write it with `bria-vgl` as the model card shows, and wire the style's
negative prompt in place of the default.

## Accuracy

Checked against the reference pipeline (diffusers' `BriaFibo2Pipeline`) in
fp32. The bar for each check is the reference's own bf16 distance from its
fp32 run. The 512² edits use the reference's own crops:

| | vpipe (bf16) | the reference's bf16 |
|---|---|---|
| conditioning, one picture (six bundles) | 0.143 | 0.160 |
| conditioning, two pictures | 0.118 | 0.133 |
| DiT, one step, teacher-forced, one picture (4 steps) | 0.015 / 0.014 / 0.013 / 0.011 | 0.032 / 0.016 / 0.014 / 0.012 |
| DiT, one step, teacher-forced, two pictures | 0.014 / 0.013 / 0.013 / 0.011 | 0.028 / 0.014 / 0.013 / 0.011 |
| the whole 4-step edit through the stages, one picture (endpoint latent) | 0.022 | 0.075 |
| the same, two pictures | 0.040 | 0.055 |

A small random model of the same architecture lands at the reference's bf16
distance in three scenarios: a generation (0.0129 vs 0.0127), one picture
(0.0121 vs 0.0122), and two pictures of different sizes (0.0128 vs 0.0126).
A wider one, built to engage every fast path, lands at 0.0126 (0.0114).
Each of these holds:

- a streamed DiT is byte-identical to one with every block resident, with
  or without its slot pair;
- a second edit in the same run reproduces the first byte for byte;
- a runtime LoRA matches the same factors fused into the weights (0.0128),
  and at strength 0 it is the base model exactly;
- a w8 pack lands at 0.021 from the reference, where bf16 is at 0.011;
- every hold of a mixed w8/w4 pack is the same bytes.

Dev's guided sampler is held to the reference pipeline's own `__call__`
(6 steps at guidance 5, a small random model of the same architecture,
the negative a shorter text so its gist count differs from the
prompt's): text-to-image 0.0131 from the reference's fp32 run (its bf16
0.0134), an edit 0.0119 (0.0128). The unguided sampler lands 1.5 away, so
the comparison sees the guidance. The conditioner's null-caption negative,
on Edit Turbo's text encoder, is 0.0084 from the reference's fp32 encoding
(its bf16 0.0093), with the same 46 rows, both alone and beside an edit's
picture.

On an M5 (matrix-core GEMMs and attention, the DiT streamed on a 16 GB
machine) the same checks pass: conditioning 0.146 / 0.121, the
teacher-forced steps 0.015–0.011, and the whole edit 0.061 (one picture) and
0.035 (two). The endpoint of a 4-step edit is one draw of a sum that
amplifies rounding, so equally accurate kernels land anywhere under the bar;
the per-step figures are the tight comparison.

**Two runs of the same graph in separate processes can differ slightly**
(45–49 dB on the final picture). FLUX.2's VAE times its kernels while the
graph loads and picks the fastest. It does that twice: between two routes
for its 3×3 convolutions, and among four tiles for its middle attention.
The candidates agree to rounding, not to the bit, and a vote can flip with
whatever else the machine is doing. Within one run the picks are fixed. For
an A/B, set both pins:

```
VPIPE_VAE_CONV_NO_AUTOTUNE=1 VPIPE_VAE_ATTN_NO_AUTOTUNE=1
```

With the convolution pin alone, three runs on an idle machine gave the same
pixels; a fourth, run beside another graph, did not.

## Speed

An M4 Pro (64 GB) takes **100 s** for the 4 steps of a 1024² edit, with the
DiT streaming its blocks (99 s with every block resident); the whole run
is 1 m 50 s. The
sequence the DiT attends over is the picture **and** the picture being
edited, plus the gist: 8,512 rows at 1024². Peak memory was 18.9 GB. Of a
step, the GEMMs are 66% (the feed-forward 45%) and attention 29%. Attention
grows with the square of the rows, so its share rises with the size.

The acceleration tiers, same edit and box, denoise seconds:

| | denoise | vs dense | picture vs dense |
|---|---|---|---|
| dense | 100.0–100.4 s | | |
| `ane_ffn` | 85.3 s | 1.18× | 39.6 dB |
| `ane_ffn` + `ane_qkv` | 75.5 s | 1.33× | 38.3 dB |
| `sol_attn` | 85.0 s | 1.18× | 27.4 dB |

Every picture is clean. Sol is approximate by design: its edit is the same
but the details are drawn a little differently, so judge it by eye. The ANE
tiers are for the M4 series.

On an M5 the GEMMs and attention run on the matrix cores, and the int8 tiers
(`i8_gemm: true`, `i8_gemm_tier`) and `sage_attn` are the levers. A base M5
with 16 GB (a fanless MacBook Air, on AC) runs the same 1024² edit with the
DiT and the text encoder both streaming, so the numbers include reading
blocks from the internal SSD:

| | denoise | whole run | vs dense |
|---|---|---|---|
| dense | 171 s | 3 m 01 s | |
| `i8_gemm`, `accurate` | 127 s | 2 m 19 s | 1.35× |
| `i8_gemm`, `token` | 101 s | 1 m 51 s | 1.69× |

One step of the int8 tiers moves the DiT's output 0.022 (`accurate`) to
0.029 (`token_plain`) from bf16, against bf16's own 0.013 from the
reference. Sage moves it 0.010, and `sage_pv8` (its P·V in int8 too) a
further 0.011, staying at the reference's bar.

The VAE stages take `i8_gemm` / `i8_gemm_tier` too (Fibo 2.0's VAE is
FLUX.2's): on the same M5 a 1024² decode is 1.22–1.27× faster at `accurate`
and 1.37–1.42× at `fast`, 54–57 dB from the f16 picture, and an encode
1.28–1.36×. Like every int8 setting it is off by default; a codec is the
last step before the pixels.

## Quantized packs

`model-quantize` takes this checkpoint (`bits` 8 or 4, or `mixed`). It
quantizes the 46 blocks' projections and their modulation. It leaves
untouched everything model-level and the text path: the Perceiver and the
injection blocks' cross-attention, which run once a prompt. AWQ is refused.
You do not need a pack to run on a small machine (the DiT streams instead),
and w4 costs visible quality on models of this kind.

## Not yet supported

- **Text images.** When a caption's `text_render` entries carry text, the
  reference pipeline draws each text as a picture (with the fonts the repo
  ships) and lets Qwen3-VL read it. vpipe does not render them yet; the
  caption's text still reaches the model as text.
- **Dev on its released weights.** Dev's sampler, conditioning and graphs
  are verified against the reference on a small model of the same
  architecture and with Edit Turbo's text encoder (above). Its own
  checkpoint has not yet been run in vpipe, so no speed is quoted for it:
  at 50 steps with two forwards each, expect about 25 times Edit Turbo's
  4-step denoise.
- **The style prompts** (`styles/`). They are prompts for a caption
  writer (an LLM), which vpipe does not run; their negative prompts can be
  wired as they are.

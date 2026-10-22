# Fibo 2.0 Edit Turbo: image editing

**Fibo 2.0 Edit Turbo** (Bria) edits a picture from an instruction. It takes
one picture or several, plus a plain-language instruction or a structured
JSON caption of the result, and returns the edited picture. It is distilled
to edit in **4 steps with no classifier-free guidance**. vpipe runs the
published **bf16** checkpoint on-device through its own Metal kernels.

| | |
|---|---|
| **Denoiser** | an 8B DiT (16.3 GB bf16): 2 noise-refiner blocks and 44 main blocks; a 2-layer Perceiver squeezes the text into 256–512 *gist* tokens; five main blocks also cross-attend their own text bundle |
| **Text encoder** | Qwen3-VL-4B-Instruct (8.9 GB). It reads the instruction **and** the pictures to edit; fifteen of its hidden states, each scaled to unit RMS, become six text bundles |
| **VAE** | FLUX.2's `AutoencoderKLFlux2` |
| **Steps** | 4, no CFG (the checkpoint's own recipe) |
| **Licence** | **CC BY-NC 4.0, non-commercial.** Read it on the model page |

Turbo is for **editing**. Bria publishes text-to-image as a separate model,
Fibo 2.0 Dev. vpipe does not run Dev yet: it needs a guided sampler
(see "Not yet supported" below).

## Getting the model

The Hugging Face repo is **gated**. Accept the terms on
[the model page](https://huggingface.co/briaai/Fibo-2.0-edit-turbo),
create a **Read** access token, and put it in the `model-fetch` stage of
[`prepare-fibo2-edit-turbo`](pipelines/prepare-fibo2-edit-turbo.vpipeline),
or in the `hf_token` field of the stage in the web UI, which keeps the
secret out of a file. The download is **~26 GB**: `transformer/`,
`text_encoder/`, `processor/`, `scheduler/`, `vae/` and `fonts/`.

You do not need to quantize. On a machine that cannot hold the DiT beside
the encoder, vpipe **streams** the DiT's blocks from disk, and it decides
this per run from the machine's memory.

## The pipeline

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

- a streamed DiT is byte-identical to a preloaded one, with or without its
  slot pair;
- a second edit in the same run reproduces the first byte for byte;
- a runtime LoRA matches the same factors fused into the weights (0.0128),
  and at strength 0 it is the base model exactly;
- a w8 pack lands at 0.021 from the reference, where bf16 is at 0.011;
- every hold of a mixed w8/w4 pack is the same bytes.

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
DiT streaming its blocks (99 s preloaded); the whole run is 1 m 50 s. The
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
- **Guidance and a negative prompt** (Fibo 2.0 Dev). Turbo runs without
  them.

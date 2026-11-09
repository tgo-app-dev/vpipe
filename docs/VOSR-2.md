# VOSR 2.0 on Apple Silicon

**VOSR 2.0** restores and upscales photographs. It is a 1.4-billion-parameter
diffusion transformer that runs in **one step**, and it is **vision-only**:
there is no prompt, no tokenizer and no text encoder anywhere in it. What the
model conditions on is a **DINOv2** reading of the picture itself.

That makes it the odd one out among the image models here. Every other family
in this tree turns words into a picture; this one takes a bad picture and
returns a better one, at whatever size you upscale to.

vpipe runs it on-device through its **metal-compute** backend: its own Metal
kernels, no Python and no third-party tensor runtime in the forward pass.

Nothing here is quantized. The checkpoint ships fp32 and is read as bf16.

## What you need

| | |
|---|---|
| **Machine** | Apple Silicon Mac (M-series). |
| **Memory** | 16 GB is enough. Resident weights are ~3.3 GB: a 2.8 GB restorer plus a 0.5 GB vision tower that is parked between pictures. |
| **Disk** | **~6.5 GB** — 5.4 GB for VOSR and its autoencoder, 1.1 GB for DINOv2. |
| **Build** | An Apple Silicon build of vpipe — the default on arm64 macOS. See the main [README](../README.md). |
| **Account** | None. Both repos are public and ungated. |

## The pipelines

- **[`prepare-vosr-2.vpipeline`](pipelines/prepare-vosr-2.vpipeline)** — fetch
  the restorer and the vision tower. Run once.
- **[`vosr-2-restore.vpipeline`](pipelines/vosr-2-restore.vpipeline)** — a
  picture in, a restored `.png` out, with an A/B comparison view.

Either runs from the terminal with `vpipe --launch <file>` or opens with
**Load** in the web UI's Pipeline Manager.

## Step 1 — get the models

```sh
vpipe --launch docs/pipelines/prepare-vosr-2.vpipeline
```

Two fetches, and the second is not optional. **The vision tower is a separate
download.** VOSR's own release ships DINOv2 as a torch-hub pickle, which
nothing here can read, so the catalogue points at Meta's `facebook/dinov2-large`
instead — the same weights in safetensors.

**From ModelScope instead.** Both repos are mirrored on modelscope.cn, so a
machine that cannot reach huggingface.co sets `source` to `modelscope` on the
fetch stages, or exports `VPIPE_MODEL_SOURCE=modelscope` once. The restorer is
published there under a different name, `LULALULALU/VOSR_CKPT`, which is mapped
internally — you still name the HuggingFace path either way, and the model
still lands in the same directory and registers under the same key. The tower
is mirrored under its own name and needs no mapping.

The two copies are the same model, checked rather than assumed: a restoration
run from the ModelScope download is **bit-identical** to the same run from the
HuggingFace one. Their four pinned files match in size, and the only textual
difference anywhere is whitespace in the autoencoder's `config.json`.

The autoencoder, on the other hand, travels **with** the restorer. VOSR
publishes the Qwen-Image VAE with its temporal axis removed, because a still
image never uses it, and the fetch takes that copy. vpipe serves it with the
same code that runs the 3D original.

## Step 2 — restore a picture

Point `load-image` at your file and run:

```sh
vpipe --launch docs/pipelines/vosr-2-restore.vpipeline
```

### The shape of the graph, and why it has two branches

```
load-image ─▸ image-resample ─┬─▸ diffusion-conditioner ─┐
                              │                          ├─▸ generate-image ─▸ vae-decode
                              ├─▸ vae-encode ────────────┘                       ▲
                              └─────────────────────────────────── color_reference ┘
```

**One resample feeds every branch**, and that is the load-bearing part. The
model never enlarges anything itself: the picture is upscaled to the output
size first, and the restorer works at that size from the start. The
autoencoder and the vision tower have to see **the same pixels**, and so does
the colour fix at the end (below), so the resample happens once, upstream of
the fork.

Use `"algorithm": "bicubic"` there. The reference upscales with Pillow's
bicubic, and the pixels are the entire conditioning signal — a different filter
is a different input, not a rounding difference.

`diffusion-conditioner` on this family loads DINOv2 and nothing else. Its
prompt port stays unwired; wiring one changes nothing.

### The size is the resample's, not the config's

`generate-image` takes its geometry from the latent on `ref_latent0`. Setting
`width`/`height` there does not upscale anything — change the resample instead.
A mismatch is reported at debug level and the latent wins.

## Step 3 — the knobs that matter

| key | stage | what it does |
|---|---|---|
| `tile_size` | vosr-model-config | Tile the restorer in latent space, in output pixels. Unset tiles at the resolution the checkpoint was distilled at; `0` forces one pass. |
| `tile_overlap` | vosr-model-config | Overlap between tiles, in output pixels. Default 32, raised to an eighth of the tile when smaller. |
| `steps` | generate-image | Defaults to **1** here. The checkpoint is distilled to one step and gains nothing from more. |
| `seed` | generate-image | The noise field. One step still starts from noise. |
| `color_fix` | vosr-model-config | `wavelet` (default), `adain` or `none` — the colour correction below. |
| `encoder_dir` | diffusion-conditioner | Where DINOv2 is, if it is not `facebook/dinov2-large` in the models DB. |
| `unload_when_idle` | diffusion-conditioner | `park` hands the tower's pages back between pictures. |

### Colour, as the release corrects it

The release's script does not hand back the decode as it is: it corrects
the restored picture's colour against the upscaled picture it restored
(its `--align_method`), and by default it does so with what it calls a
**wavelet** fix — the restored picture keeps its own detail and takes its
low frequencies from the input, a Gaussian blur at sigma 5:

```
out = clamp(blur(input) + restored − blur(restored))
```

vpipe does the same, by default, in `vae-decode`: `generate-image` asks for
it on every latent, and `vae-decode` runs it against the picture on its
`color_reference` input — the same resampled pixels, which the shipped graph
wires there. `color_fix` on `vosr-model-config` picks `adain` instead (each
channel's mean and spread moved to the input's) or `none` (the script's
`nofix`).

A graph that leaves `vae-decode`'s `color_reference` unwired gets the
decode uncorrected, and the log says so once. The pictures pair with the
latents **in order, one each**, so feed that input the same stream of
pictures the restorer saw.

The blur is OpenCV's: 41 taps at sigma 5 and mirrored edges, as the
script's `cv2.GaussianBlur` uses. One difference remains: the script
corrects its 8-bit picture and truncates the result to 8 bits again;
vpipe corrects the decode before it is rounded, and rounds once. So
vpipe's output reads about half a level brighter on average.

### Tiling, and why it is the default

Past the resolution the checkpoint was distilled at — 512 px for VOSR 2.0 —
a restore **tiles at that resolution unless the graph says otherwise**, and
the two keys above live on their own `vosr-model-config` stage wired to
`generate-image`'s `model_config` iport. Leave it unwired and you get the
default.

The reason is not attention cost, though that is real: attention is quadratic
in the token count and the token count is quadratic in the output side, so a
1024×1024 output is 4096 tokens and 2048×2048 is 16384. The reason is the
rotary table. The reference's one-step sampler applies a table built once for
the trained grid, so it cannot run any other token count at all, and its
tiling exists to put every tile back on that grid. Running a larger grid
through a rescaled table puts every other token on a position the weights
never saw, which weaves a periodic pattern through whatever the model has to
synthesise rather than pass through — reported from the field as a
checkerboard on faces at a 1024 output, absent at 512 and absent tiled.
`tile_size: 0` still gives you one pass, and warns.

Tiling costs something other than time. The reference re-runs its vision tower
on each tile's own pixels; a graph whose conditioner ran once cannot, so vpipe
**crops the feature grid** per tile instead. That is the one way a tiled run
still differs from the reference.

## What it costs

Measured on an **M4 Pro, 64 GB**, a 256×256 photograph restored 4× to
1024×1024:

| | |
|---|---|
| whole pipeline, cold | 8.8 s |
| the restore itself | ~2 s |
| the same, tiled at 512 px | 10.7 s |

Against the 1024×1024 original the 256×256 came from:

| | PSNR |
|---|---|
| bicubic ×4 | 30.90 dB |
| VOSR 2.0 | 29.75 dB |

**That ordering is expected and is not a defect.** A generative restorer
invents plausible detail rather than the exact detail that was lost, which
costs a little PSNR and buys a great deal of apparent resolution. PSNR is here
as a sanity bound — a broken port lands ten decibels lower — not as a quality
score.

## Which checkpoint

Only `VOSR2/`, the folder the catalogue fetches. The same repository also
publishes the April release's checkpoints (`VOSR_0.5B_*` and
`VOSR_1.4B_*`). Each is refused by name if pointed at: the 0.5B models use
another autoencoder, `VOSR_1.4B_ms` is the multi-step model, and
`VOSR_1.4B_os` is distilled differently. The two 1.4B ones have VOSR 2.0's
tensor shapes, so their `args.json` is the only thing that tells them
apart.

## What is inside

- **The restorer** is a single-stream DiT: 36 blocks at 1536 wide, 24 heads,
  patch 2. Each block runs adaLN-modulated self-attention with 2D rotary
  positions, then **cross-attention to the DINOv2 tokens**, then a SwiGLU MLP.
  Its patch projection reads 32 channels — the low-quality latent and the noise
  side by side — which is what makes it a restorer rather than a generator.
- **Positions are rescaled, not extrapolated.** The rotary table was trained on
  a 32×32 grid; a 4× upscale's 128×128 grid is mapped back into that range. It
  is why the model works at sizes it never saw.
- **The vision tower** is DINOv2 ViT-L/14 at 448×448, and vpipe reads its
  **layer-17** output. Blocks past that are never evaluated and never loaded,
  so 18 of 24 come off disk.
- **The preprocessing is torch's, not Pillow's.** The tower's input is squashed
  to a square with torch's bicubic, and its position grid is resampled at
  DINOv2's own 0.1 offset. Both are reproduced here; either one done the
  obvious way instead gives a conditioning that looks fine and is not the
  reference's.

## Verification

Checked numerically against the reference implementation, on a 160×160
photograph upscaled 4× to 640×640. `tools/dump_vosr_golden.py` dumps the
reference tensors from a checkout of
[cswry/VOSR](https://github.com/cswry/VOSR); the `vosr_reference` tests read
them back. Relative L2 against fp32 torch:

| | rel-L2 |
|---|---|
| VAE encode | 0.0015 |
| DINOv2 tokens | 0.0232 |
| DiT velocity, from the reference's tokens | 0.0164 |
| whole stack, from vpipe's own tokens | 0.0163 |

**The last two lines are the point.** The third feeds the restorer the
reference's conditioning, so it measures the restorer alone. The fourth feeds it
the conditioning vpipe's own tower produced, and it lands in the same place —
so the tower's 2.3% costs the answer nothing. Had the DiT amplified it, the
conditioner would need a wider element type.

The tower's 2.3% is accumulated bf16, not a mistake in the preprocessing. The
dumper's `--layer` flag sweeps the depth, and the error grows with it and then
flattens: 0.0073 after one block, 0.0204 after six, 0.0269 after twelve, 0.0232
after eighteen. A wrong resize, a wrong normalisation or a wrong position grid
would already be an order of magnitude out after **one** block.

To re-run it, point three variables at the models and the goldens:

```sh
VPIPE_VOSR_TEST_MODEL_PATH=<models>/CSWRY/VOSR \
VPIPE_DINOV2_TEST_MODEL_PATH=<models>/facebook/dinov2-large \
VPIPE_VOSR_GOLDEN=<goldens>/vosr \
  vpipe_test --filter 'vosr_reference.*'
```

Unset, the tier skips.

### One measurement about the resample

The `bicubic` filter matches Pillow's to within **4 levels of 255**, mean 0.19,
on the same 160→640 upscale. The residue is Pillow's round-and-clamp to bytes
between the horizontal and vertical passes, which this stage does not do. For
scale: choosing Lanczos there instead would be 18 levels away.

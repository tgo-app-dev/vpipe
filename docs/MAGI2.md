# MAGI-2-preview

[sand-ai/MAGI-2-preview](https://huggingface.co/sand-ai/MAGI-2-preview)
(Apache-2.0) generates a 10-second clip **with its soundtrack** from a
prompt, or from a prompt and a picture. vpipe runs it in `generate-video`
as the `magi2` family.

| graph | what it does |
|---|---|
| `prepare-magi2-preview` | fetches the release (307 GB) |
| `magi2-text-to-video` | prompt to a 896 x 512 clip with sound (the release's 540p preset) |
| `magi2-text-to-video-272p` | the same at 448 x 256 (its 272p preset) |
| `magi2-image-to-video` | prompt and picture to a clip with sound |
| `magi2-text-to-video-1080p` | the 540p preview refined to 1920 x 1088 at 25 fps (its 1080p preset) |

## What runs

- **Text.** `diffusion-conditioner` on the release root reads its
  Qwen3.5-27B (`text_encoder/`), streaming the layers through two slots.
  The prompt is used raw (a JSON caption, the form the release's prompt
  enhancer writes, is turned into its compact markdown first), every CJK
  character is its own pre-token, and at most 7000 tokens are kept. The
  condition is the output of the third-to-last layer, un-normed, every
  row. The negative is the release's fixed negative prompt.
- **The denoise.** `generate-video` runs the preview DiT: 40 layers at
  3072 wide wrapped in four-stream hyper-connections; 36 of them a
  multi-head mixture of experts (12 heads of 256, each routed to 6 of
  256 experts). Video and audio latents are denoised together by UniPC
  over 100 steps at shift 7, with classifier-free guidance against the
  negative prompt (video 5, audio 7). Each step is ONE batched forward of
  the conditional and unconditional passes.
- **The refine (1080p).** With `refine_width` / `refine_height` set on
  `magi2-model-config`, `generate-video` hands the preview's video latent
  to the release's refiner (`refiner/`): a dense DiT, 30 layers at 4096
  wide, grouped-query attention (32 query / 8 key-value heads) with a
  sigmoid gate a head, and no time input. Its attention is a WINDOW: the
  video's latent cells are grouped into blocks of 8 frames x 4 x 4, and a
  block attends the blocks within two of it in each direction, plus the
  prompt; the prompt attends everything. The latent is upsampled
  trilinearly to 2T - 1 frames on the asked grid (the pixels floored to
  16-pixel cells), re-noised part way (the release's ZeroSNR schedule,
  entry 220: 0.839 of the latent, 0.544 of noise), then denoised by
  UniPC over 5 steps at shift 5 with guidance 2 against the negative
  prompt -- a conditional and a negative forward a step. The soundtrack is
  the preview's, untouched. The clip comes out at twice the preview's
  rate: 4 (2T - 2) + 1 frames at 25 fps (249 for the 10-second preview).
- **Decode.** `vae-decode` runs the release's distilled turbo decoder
  (`turbo_vae/`); `audio-vae-decode` runs Stable Audio Open's decoder
  (`stable-audio-open-1.0/`) and time-stretches its output by 441/512 to
  44.1 kHz, as the release does.
- **Image-to-video.** The picture goes to the conditioner (on its
  picture port), which addresses it from the prompt ("The first frame
  refers to <Figure 1>") and appends its special token, and to
  `vae-encode`, which encodes it with the Wan2.2 VAE's encoder
  (`vae/Wan2.2_VAE.pth`). `generate-video` takes the latent on its
  first-frame port. The picture keeps its aspect: resize it so its long
  edge is the clip's long edge (the shipped graph sets `width: 896`, for
  a landscape picture; set `height` instead for a portrait one); the
  encoder floors both sides to multiples of 32 as the release does.

## Knobs

`magi2-model-config` on `generate-video`'s model-config port:

| key | default | |
|---|---|---|
| `steps` | 100 | over generate-video's own `steps` |
| `shift` | 7 | the flow shift of the sampling grid |
| `video_guidance` | 5 | 1 runs one pass a step and no negative |
| `audio_guidance` | 7 | |
| `refine_width`, `refine_height` | unset | refine to this size (the 1080p graph: 1920 x 1088); unset, the preview is the clip |
| `refine_steps` | 5 | the refine's steps |
| `refine_shift` | 5 | the refine grid's flow shift |
| `refine_guidance` | 2 | 1 runs one forward a step and no negative |

The preview runs at **12.5 fps**: set `generate-video`'s and
`rgb-to-video`'s `fps` to 12.5 (a refined clip carries its own 25 fps;
the 1080p graph sets `rgb-to-video` to 25). Frame counts are rounded up to 4k + 1;
the model was trained on 125 frames (10 s) and the release supports no
other length, so other lengths are generated but are off its training.
The noise is drawn on the CPU (torch's generator, bit for bit), not on
CUDA as the release draws it, so a seed does not reproduce the
release's clip. The refine's re-noise is drawn from the seed + 1.

## Memory and speed

The preview DiT is 228 GB: a 10.7 GB trunk (held) and 36 MoE layers of
6 GB of routed experts each. As many MoE layers as the box has room for
are held and the rest STREAM through two slots, read once a step for
both guidance passes. On anything short of ~250 GB of RAM most of the
experts stream, so **the drive is the cost**: each step reads the
streamed experts again. Keep the checkpoint on the fastest drive you
have.

| box | geometry | per step | note |
|---|---|---|---|
| M4 Pro 64 GB, Thunderbolt 5 SSD | 448 x 256 | ~52 s | every MoE layer streamed, ~207 GB a step at ~4 GB/s |

The room for held layers is decided when the DiT loads, against
everything else the run holds at its widest: the held experts stay
through every phase of a clip -- the conditioning, the preview, the
refine (with the refiner and its forward beside them) and the decodes --
so they get what the plan's busiest phase leaves, after the trunk, two
slots and 8 GB of headroom. Each held layer is read uncached (not through
the page cache) and admitted through the session's wired pool: past the
pool it streams instead, and the model gives the pool its bytes back when
it is let go. `VPIPE_MAGI2_RESIDENT_MOE=<n>` pins the number of MoE
layers held (0: all stream). On a 64 GB box, run with a wired pool of
~30 GB (`--wired-pool-mb 30000`): the text encoder's resident layers
otherwise fill the default pool while the prompt is conditioned.

For image-to-video the picture's long edge may not pass the clip's (the
plan books the picture's tokens at that square); a larger one is refused
with a message to resize it.

The refiner holds 13 GB (bf16) and is loaded at the first refine, after
the preview DiT has let its scratch go; its forward at 1920 x 1088 works
in about 21 GB (522,240 tokens: the video's blocks padded to whole
attention tiles). A 1080p refine forward is ~5.5 PFLOP of projections and
~3.3 PFLOP of window attention (each query sees ~12,700 keys), ten of
them a refine:

| box | 1080p refine forward (30 layers) | the refine (10 forwards) |
|---|---|---|
| M5 Ultra | ~101 s (87 TFLOP/s; matmul2d + matrix-core attention) | ~17 min |
| M4 Pro | ~22 min (6.6 TFLOP/s; steel) | ~3.7 h |

The M5 routes (matmul2d projections, the matrix-core attention -- the
preview's with its sinks, the refiner's window) take over in both DiTs
where the GPU has them (`VPIPE_MAGI2_NO_MMA2=1`,
`VPIPE_MAGI2_NO_NAX_ATTN=1` turn them off). The turbo decoder decodes a
large clip in spatial tiles -- every op of it is local, so a tile with a
halo of its receptive field (15 latent cells) is the whole decode's, bit
for bit -- keeping its work buffers under 4 GB; a 1080p refined clip's
decode works in ~13.7 GB beside the clip. Run the 1080p graph on a box
with ~96 GB or more.

The achieved-FLOP/s meter (`GET /api/profiler/flops` in the web UI,
`vpipe.perf_flops()` from Python) counts the run's GPU arithmetic as it
retires: both DiTs' projections and f32 linears; the preview's attention
at every token pair, and the refiner's at the pairs its window's tiles
visit (the padding slots inside a tile included); SageAttention as the
attention it stands for; each routed expert pick as its three products;
and the turbo and Stable Audio decoders' convs, a turbo window's context
frames and a tile's halo included, since they are decoded again. The
text encoder and the host's picture encoder are not counted. On a split
DiT a rank counts its routed experts at its share of them (the GPU sorts
out which picks are its own).

The text encoder (Qwen3.5-27B, 52 GiB on disk) streams once per prompt and its
negative; what its prefill runs in (a 7000-token prompt's activations,
the K/V of its full-attention layers, its recurrent state) is booked in
the conditioning phase. The codecs are booked at what they hold, not at
their files: the turbo decoder 204 MB (its file is a 1.8 GB training
state), Stable Audio Open's decoder 149 MB (its file, 4.85 GB, is the
whole model), the picture encoder ~206 MB on the host (of the 2.8 GB
Wan2.2 VAE). A 540p clip's turbo decode works in ~4 GB; the picture
encoder encodes an 896 x 512 picture in well under a second, in at most
~0.5 GB of host memory beside its weights (booked in vae-encode's arena).

## Split across two Macs

generate-video's `tensor_parallel` runs the stage as one RANK of the DiTs
split across machines (`generative-models/tensor-parallel.h`, feature
`tensor-parallel-magi2/1`), as it does for MiniMax H3. vpipe moves no bytes
between machines: a plugin registers the transport (Thunderbolt RDMA,
sockets -- the plugin's), and the bag names it.

```json
"tensor_parallel": {"rank": 0, "ranks": 2,
                    "transport": "<the plugin's name>", "params": {}}
```

Each host runs the same graph -- conditioner, generate-video, decoders --
its own rank in the bag. `heads` and `ffn` (each `[first, end]`) give an
uneven share, for a slower Mac to take less; unsaid, an even split.

WHAT A RANK HOLDS. The preview DiT is split by its 24 query HEADS -- their
rows of every q | k | v and gate, their sinks, their columns of the
attention's output projection -- and by EXPERTS: `ffn` counts each MoE
head's 256 routed experts, and a rank holds the run `[first, end)` of
every head's, read as one range a head (so a streamed layer reads only the
share). The shared, modality-specific and mm-layer MLPs are cut by rows at
the same fraction. The routers, the split and merge projections, the
hyper-connections, norms, embedders and heads run whole on every rank:
every rank routes every token alike and computes only the picks that land
on its experts. An even split holds ~6.3 GB of the 10.7 GB trunk and
~109 GB of the 217 GB of experts a rank -- room on a 256 GB M5 Ultra for
every MoE layer, so nothing streams.

THE EXCHANGES. Each sublayer's output on a rank is a partial sum, added to
the other rank's before its hyper-connection takes it: two exchanges a
layer a sample, bf16, `tokens x 3072 x 2` bytes each. A guided step's two
samples (the prompt and its negative) run interleaved, so each sample's
exchange crosses while the GPU works on the other; the last MLP's of a
layer alone waits. `VPIPE_MAGI2_TP_LAG=0` runs them in line (the same
arithmetic, bit for bit).

A REFINE splits its refiner at the same fractions -- its 32 heads in whole
groups of four (12 of the preview's 24 is 16 of its 32), its 10,920 MLP
rows -- on the clip's one channel after the preview. Its output projection
and MLP run in bands of 16,384 rows, each band's partial crossing while the
GPU works on the next band (a 1080p layer's attention output is ~4.3 GB a
rank, too large to wait for whole).

The plan books a rank's SHARE: its trunk, its experts and the refiner's
share at the int8 tier asked, its forward's scratch at its widths with its
exchange buffers. The int8 tiers and SageAttention take a rank's widths.

Both ranks must produce the same conditioning, so run the same graph on
the same kind of GPU: the conditioner and every whole part of the DiT are
computed on each host, and only the same kernels give the same bits.

WHAT IS TESTED (one process, two ranks on two threads over an in-process
channel): the preview split against the unsplit model on the small golden
-- even and uneven shares, experts held and streamed, overlapped and in
line -- its ranks bit-identical and within the unsplit test's bar of the
fp32 reference (0.96-1.04x the reference's own bf16 distance; unsplit
0.94x), each rank holding what its plan books; the same on the release's
own weights (its layer subset, 1.03x; unsplit 0.96x); the refiner split
likewise (1.07x; unsplit 1.00x); on an M5 the int8 token tier and
SageAttention's int8 P*V on a rank (preview 2.88x, refiner 3.85x -- the
tiers' own figures). No two-host run has been made: there is no transport
in this tree.

## int8 GEMMs and SageAttention (M5)

`generate-video`'s acceleration keys reach both DiTs, the preview and
the refiner. Both tiers run on the GPU's matrix cores only: on an M4 they
are declined with a log line and the DiTs run bf16.

| key | what it takes |
|---|---|
| `i8_gemm` + `i8_gemm_tier` | the block projections -- q\|k\|v, the attention's output, the MLPs, the preview's split / merge / shared experts -- in int8. `accurate` quantizes the bf16 weight per call (from 1024 rows and 1024 deep); `fast`, `token` and `token_plain` HOLD the projections in their int8 layout, converted at load (the preview's trunk 10.7 -> 5.6 GB, the refiner 13.5 -> 8.5 GB). The refiner's MLP down projection (10920 deep) fits no held layout and runs `accurate` on every tier. |
| `sage_attn` (+ `sage_pv8`) | the attention's QK^T in int8 (and its P*V), inside the matrix-core flash kernel: the preview's full attention with its per-head sinks, the refiner's block-sparse window. Every query row. `sage_pv8_from_step` counts each stage's own steps; `sage_dense_layers` leaves leading layers bf16. |

Left in bf16 whatever the tier: the per-head gate (24 or 32 outputs a
row), the f32 embedders, routers and heads, and the preview's routed
experts, which run their own grouped kernel (on the matrix cores, below).

THE ROUTED EXPERTS RUN ON THE MATRIX CORES too (an M5; any expert width
of whole 64s; `VPIPE_MAGI2_NO_NAX_MOE=1` is the A/B): per 64 sorted rows
of one expert, gate and up, swiglu7 and down in one kernel, the hidden
width never leaving the threadgroup. They
had run on the simdgroup kernel at ~2.7 TFLOP/s and were 69-78% of a
preview forward on the M5 Ultra; now ~63 TFLOP/s, and the attention is
the larger part, which Sage takes. One mm layer and one MoE layer at the
release's widths, 896 x 512 x 125, a guided step's batch of two
(`VPIPE_MAGI2_DIT_PROFILE=1` splits a forward by section):

| arm (M5 Ultra) | preview: 2 layers, batch of 2 | refiner: 1080p forward, 30 layers |
|---|---|---|
| bf16, the simdgroup experts | 8.40 s | |
| bf16 | 2.64 s | 100.7 s |
| `i8_gemm` accurate | 2.61 s | 83.9 s (1.20x) |
| fast | 2.58 s | 79.8 s (1.26x) |
| token | 2.48 s | 71.9 s (1.40x) |
| `sage_attn` | 2.25 s | 93.0 s (1.08x) |
| `sage_attn` + `sage_pv8` | 2.00 s | 91.7 s (1.10x) |
| token + `sage_attn` + `sage_pv8` | 1.84 s (4.6x the first row) | 61.6 s (1.63x): the refine ~17 -> ~10 min |

A MoE layer alone 7.14 -> 1.45 s (1.07 s with token + Sage + int8 P*V), an
mm layer 1.28 s (0.86 s): for the release's 4 mm and 36 MoE layers, every
expert held, a preview step of ~261 s becomes ~54 s, ~40 s accelerated --
the release's 100 steps ~7 h -> ~90 min -> ~67 min on the Ultra (an
estimate from these benches; a box that streams experts adds their reads).
(Random weights at the release's widths; `magi2.refiner_bench` and
`magi2.preview_bench`, VPIPE_MAGI2_BENCH_ARMS.)

WITHOUT MATRIX CORES (an M4) the experts take a STAGED simdgroup kernel:
the expert's weights staged through threadgroup memory in coalesced loads
and shared by eight simdgroups, swiglu7 on the accumulators' registers --
the same products in the same order as the first kernel, so the same
output to the bit (`VPIPE_MAGI2_MOE_SG=1` runs the first kernel). On the
M4 Pro, one MoE layer at the release's widths over half the bench grid
(16 x 32 x 56), a batch of two: the routed experts 5.87 -> ~1.5 s (~5.5
TFLOP/s), the layer's forward 10.76 -> 6.64 s; an mm layer (no experts)
5.70 s. On a 64 GB box most experts stream, and at small geometries the
drive's reads, not the arithmetic, set the step.

ACCURACY, against the fp32 reference on synthetic DiTs 1024 wide (the
`wide` goldens, over every floor the tiers have), as a multiple of the
reference's own bf16 distance: SageAttention 0.99-1.04x, its P*V
included; the int8 tiers 2.7-4.6x (accurate 2.8 / 3.3, fast 3.8, token
3.8-3.9, token_plain 3.9-4.6 -- preview / refiner) -- on random weights,
the int8 floor of a GEMM (~1e-2 against bf16's ~4e-3). Not yet judged on
the release's weights: a clip made with them is the test of whether a
tier is good enough.

SINKS UNDER SAGE. The preview's attention carries a sink per head, which
the shared flash kernel had not met under the int8 QK: SageAttention
subtracts K's mean over tokens before quantizing it -- exact for a
softmax over keys alone, but a sink's logit must move with the scores --
and the int8 P*V quantizes P against a running max the sink had joined.
Both are fixed in the kernel (each row's sink shifted by its q.mean;
the sink kept out of the P*V's grid and added to the sum after); without
them the preview read 13x and 3.7x the bar. A host whose kernel predates
the fix cannot take an attention with sinks under Sage, so there the
preview declines it -- "attention stays dense", said in the log -- and
the refiner, which has no sinks, keeps it.

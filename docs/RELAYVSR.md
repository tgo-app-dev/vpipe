# 4x video super-resolution with RelayVSR

**RelayVSR** upscales a video to exactly four times its width and height.
It is the model of Wang et al., *RelayVSR: Large-Small Model Collaboration
for Efficient Real-World Video Super-Resolution* (arXiv 2609.37850), as
[kopperx/RelayVSR](https://github.com/kopperx/RelayVSR) publishes it, with
its weights at [kopper/RelayVSR](https://huggingface.co/kopper/RelayVSR)
(Apache-2.0).

Two models work in relay over the stream:

- A **large keyframe model**, a 5-billion-parameter diffusion transformer
  shaped like Wan 2.2-TI2V-5B, retrained to produce in **one step** a
  sharp latent of a frame. It runs on one frame in every fifteen.
- A **small decoder**, 32 million parameters, that turns **every** frame
  into its 4x frame directly in RGB. It is guided by the two keyframes
  around the frame and by the frame before it.

There is no VAE and no text encoder: the keyframe model's text context is
a constant shipped inside its checkpoint.

vpipe runs both on-device through its **metal-compute** backend, in the
**`relayvsr-upscale`** stage: frames in, 4x frames out, one for one and in
order. No Python is involved; the checkpoints are read as data.

Its sibling is **FlashVSR+** (the `flashvsr` family of `generate-video`).
The two differ mainly in this:

| | RelayVSR | FlashVSR+ |
|---|---|---|
| Output size | exactly 4x the input | 4x, on a 128-pixel grid |
| Graph | one stage | a source encoder, `generate-video`, `vae-decode` |
| Memory | larger: a 5B keyframe model and its key/value ring | smaller |
| Latency | a frame leaves 15 frames after it arrives (at most) | a chunk at a time |

## What you need

| | |
|---|---|
| **Machine** | Apple Silicon Mac (M-series). |
| **Memory** | **8.4 GB of weights** (4.3 GB with the `fast` or `token` int8 tier, on an M5), plus a working set that grows with the input frame. Most of it is the keyframe model's ring of keys and values for the last four keyframes, 30 blocks deep. See the table below. |
| **Disk** | **10.2 GB**: `wan.safetensors` (the keyframe model, 10.1 GB) and `flash.safetensors` (the decoder, 128 MB). |
| **Account** | None. The repository is public and ungated. |

What a run needs, by **input** size. This is what the stage books, and
each figure was checked against the GPU's actual peak allocation (see
"Accuracy" below):

| Input | Output | Ring | Other working set | Total with weights |
|---|---|---|---|---|
| 320 x 180 | 1280 x 720 | 1.3 GB | 0.2 GB | **9.9 GB** |
| 480 x 270 | 1920 x 1080 | 2.8 GB | 0.5 GB | **11.7 GB** |
| 640 x 360 | 2560 x 1440 | 4.9 GB | 1.0 GB | **14.3 GB** |
| 854 x 480 | 3416 x 1920 | 8.8 GB | 1.7 GB | **18.9 GB** |
| 960 x 540 | 3840 x 2160 | 11.2 GB | 2.2 GB | **21.8 GB** |
| 1280 x 720 | 5120 x 2880 | 19.8 GB | 3.9 GB | **32.0 GB** |

Beside these, the stage holds one output frame in ordinary memory at a
time: a finished window's frames are decoded and handed on one by one.

When a graph launches, the stage books the size `output_width` x
`output_height` names (a quarter of it, as input). Without them it cannot
know the frame size yet, so it books a 960 x 544 input (the 21.8 GB row),
and on a 16 GB Mac the launch then warns that the graph may not fit, even
for an input that does. Either way it revises the booking to the real size
at the first frame, and gives everything back at the end of the stream.

Nothing in RelayVSR streams, so a run that does not fit would page every
window. The stage refuses it instead, with an error naming what it wanted
and what the Mac had:

- **At launch**, when `output_width` and `output_height` are given and the
  run needs more than the GPU working set macOS gives a process, even with
  nothing else loaded. Its weights are not read.
- **At the first frame**, when the run does not fit in the working set
  left, or in the RAM that can be reclaimed, once every stage has loaded.

Either way the stage ends and gives back everything it booked, and the
pipeline ends with no frames from it.

So 4K output from 540p wants a 32 GB Mac. The keyframe model's weights are
held whole, so a **16 GB Mac runs only small inputs**: 320 x 180 and
378 x 252 ran on a 16 GB M5 MacBook Air, with about 1.6 GB of swap. Block
streaming of those weights, which would let a small Mac go further, is not
implemented yet.

## The pipelines

- **[`prepare-relayvsr.vpipeline`](pipelines/prepare-relayvsr.vpipeline)**
  fetches the model. Run it once.
- **[`relayvsr-upscale.vpipeline`](pipelines/relayvsr-upscale.vpipeline)**
  takes a video in and writes its 4x `.mp4`, soundtrack included.

Either runs from the terminal with `vpipe --launch <file>`, or opens with
**Load** in the web UI's Pipeline Manager.

## Step 1 — get the model

```sh
vpipe --launch docs/pipelines/prepare-relayvsr.vpipeline
```

This fetches `kopper/RelayVSR` into your models directory. The model
browser recognises it by the format each checkpoint states in its own
metadata (`refvsr-wan-v1`, `refvsr-flash-v1`), so a copy fetched by hand
works too: keep `wan.safetensors` and `flash.safetensors` side by side in
one directory, and point `hf_dir` at it.

## Step 2 — upscale a video

Set `input_url` (on both `load-video` and `load-audio`) and `output_url`,
then:

```sh
vpipe --launch docs/pipelines/relayvsr-upscale.vpipeline
```

```
load-video ─▸ video-to-rgb ─▸ relayvsr-upscale ─▸ rgb-to-video ─┐
                                                                ├─▸ save-video
load-audio ─▸ audio-to-pcm ────────────────────────────────────┘
```

The frame rate is preserved: each output frame carries its input's rate
and timestamp. The reference does not copy audio; the example graph does,
through its own branch. Remove `load-audio` and `audio-to-pcm` for a source
without a soundtrack.

### The stage's settings

| Key | Default | What it does |
|---|---|---|
| `hf_dir` | — | The model: `kopper/RelayVSR`, or the directory holding both checkpoints. |
| `window` | 16 | The release's `--window-size`: a keyframe every `window - 1` frames. Larger is faster (fewer runs of the keyframe model) and leans more on the small decoder. A frame leaves the stage at most `window - 1` frames after it arrived. 2 to 64. |
| `output_width`, `output_height` | 0 | Optional: the size of the 4x frames the run will make, if it is known before the first frame. Both or neither, each a multiple of 4 from 32 to 32768. Only the memory plan reads it: a small input is planned small, and a run that cannot fit the Mac even alone is refused before its weights are read (see "What you need" above). The stage still makes exactly 4x what it is given. 0 plans for 3840 x 2176. |
| `seed` | 42 | The keyframe noise: torch's CPU normal draw from one generator, seeded at the start of every run. The release uses CUDA's generator, so its draw differs. |
| `output_dtype` | `auto` | `auto` matches the input: u8 in gives u8 out, f32 in gives f32 out. `u8` writes frames as the release does: (v + 1) x 127.5, clamped and truncated. `f32` gives [0, 1], unclamped. |
| `i8_gemm` | off | Run both models' block projections on the int8 GEMMs. M5 and later only. See "Faster" below. |
| `i8_gemm_tier` | `accurate` | Which int8: `accurate`, `fast`, `token` or `token_plain`. |
| `ane_ffn` | off | Split both models' feed-forward with the Neural Engine. For an M4. See "Faster" below. |
| `ane_rows` | 0 | The Neural Engine's share of the rows; 0 lets the stage balance it. |
| `sage_attn` | off | Run the attention's QK products in int8 (SageAttention). M5 and later only. See "Faster" below. |
| `sage_pv8` | off | With `sage_attn`: the attention's P*V in int8 too. |
| `autotune` | on | Whether the `accurate` tier's kernel tuner measures. Off, a run repeats exactly. |

Any input size works. The model pads a frame to a multiple of 8 and cuts
the result back to exactly 4x. Every frame of a run must be the same size.

### How fast

The table shows model time only: the keyframe model and the decoder,
averaged over a 16-frame window (one keyframe, one priming, sixteen
frames).

| Input → output | Apple M4 Pro | Apple M5 (base, 16 GB Air) | Apple M5 Max |
|---|---|---|---|
| 320 x 180 → 1280 x 720 | 0.23 s a frame | 0.27 s a frame* | 0.030 s a frame |
| 480 x 270 → 1920 x 1080 | 0.51 s a frame | — | 0.063 s a frame |
| 624 x 360 → 2496 x 1440 | 0.91 s a frame | — | 0.109 s a frame |
| 960 x 540 → 3840 x 2160 | — | — | 0.263 s a frame |

These are without the faster settings below. *The base M5 figure was
measured before both attentions moved to the matrix cores (below), and is
not current.

At 624 x 360 on the M4 Pro:

| | |
|---|---|
| a keyframe | 5.3 s, once every 15 frames |
| priming it | 0.26 s |
| a frame | 0.49–0.54 s |

The whole example graph on 48 frames of that clip took 57 s, or 1.19 s a
frame. That includes the first keyframe, the last partial window, and the
video decode and encode around the model.

The keyframe model runs at the GPU's GEMM rate (about 6.8 TFLOP/s on the
M4 Pro). On an M5 the GEMMs run on the matrix cores, and so do both
models' attention: the windowed attention on a kernel of its own, which
stages each tile's union of windows through threadgroup memory, and the
keyframe model's cross-attention on the flash kernel. On an M5 Max that
made a 480 x 270 window 2.3x faster (0.146 s a frame against 0.063), the
same numbers as the release's bf16 within its error (see "Accuracy").

## Faster: int8 and the Neural Engine

Two settings make RelayVSR faster, each on the machine it suits. Both are
off by default. Each makes the result slightly different from the default,
as the tables below measure.

### `i8_gemm`: int8 on an M5

An M5's GPU has matrix cores with an int8 path at about twice the f16
rate. With `i8_gemm` on, the large matrix multiplies of both models run
there: the keyframe model's eight projections in each of its 30 blocks,
and the decoder's projections in each of its 10. What reads the frame or
writes the pixels stays in 16 bits.

`i8_gemm_tier` chooses how:

| Tier | What it does | Weights held |
|---|---|---|
| `accurate` | Each weight is quantized as it is used, in groups of 64. | 8.4 GB |
| `fast` | The weights are held in int8, in rotated groups of 512. | 4.3 GB |
| `token` | The weights are held in int8, one scale per output channel, rotated. The fastest. | 4.3 GB |
| `token_plain` | As `token`, without the rotation. Markedly less accurate here (see below). | 4.3 GB |

Speed on an Apple M5 Max (Mac Studio), model time, `token` against off
(a frame: one with its history, as 14 of a window's 15 are):

| Input → output | a keyframe | a frame | over a 16-frame window |
|---|---|---|---|
| 320 x 180 → 1280 x 720 | 228 → 136 ms | 15 → 13 ms | 0.030 → 0.023 s a frame |
| 480 x 270 → 1920 x 1080 | 431 → 285 ms | 33 → 31 ms | 0.063 → 0.051 s a frame |
| 624 x 360 → 2496 x 1440 | 742 → 477 ms | 58 → 54 ms | 0.109 → 0.087 s a frame |
| 960 x 540 → 3840 x 2160 | 1810 → 1170 ms | 138 → 127 ms | 0.263 → 0.208 s a frame |

The keyframe model gets about 1.55x faster, a whole window about 1.25x:
most of a frame's time is the decoder's windowed attention, which is not
a matrix multiply. `sage_attn` (below) is what reaches attention. At
480 x 270 the other tiers give a keyframe of 386 ms (`accurate`) and
321 ms (`fast`); `token_plain` runs as fast as `token`.

Accuracy, against the same fp32 reference as below. Each figure is the
worst case on the two test clips, as a multiple of the release's own bf16
error:

| Tier | a keyframe | a frame | the decoder alone |
|---|---|---|---|
| off | 1.05x | 0.95x | 0.20x |
| `accurate` | 1.31x | 1.03x | 0.32x |
| `fast` | 1.36x | 1.16x | 1.20x |
| `token` | 1.51x | 1.21x | 1.19x |
| `token_plain` | 3.91x | 3.10x | 1.62x |

So `accurate`, `fast` and `token` all stay close to the release's own
bf16 error. `token` is the one to use for speed. On a Mac without matrix
cores (an M4 or earlier) the setting does nothing, and the stage says so.

### `sage_attn` and `sage_pv8`: int8 attention on an M5

With `sage_attn` on, the attention's QK products run in int8 on the
matrix cores, as SageAttention does them: Q and K are quantized per
token, K after subtracting its mean. One softmax spans every frame a
query attends, so the decoder's keyframes and its previous and current
frame share one mean. That subtraction is exact (a softmax ignores a
per-row shift). `sage_pv8` puts the P*V product in int8 too: P against each
row's running maximum, V per channel over every attended frame, the sum
held in int32.

It reaches all three attentions: both models' windowed self-attention,
and the keyframe model's cross-attention onto its context. The
cross-attention takes it only from 4096 query rows (inputs of about
680 x 380 and up). Its 512-token context costs a fixed ~0.3 ms a call to
quantize, which smaller frames do not earn back.

Speed on an Apple M5 Max, model time, `sage_attn` + `sage_pv8` against
off, alone and on top of `i8_gemm` `token`:

| Input → output | a keyframe | over a 16-frame window | with `token` |
|---|---|---|---|
| 320 x 180 → 1280 x 720 | 228 → 219 ms | 0.030 → 0.030 s a frame | 0.023 → 0.022 |
| 480 x 270 → 1920 x 1080 | 431 → 403 ms | 0.063 → 0.061 s a frame | 0.051 → 0.049 |
| 624 x 360 → 2496 x 1440 | 742 → 689 ms | 0.109 → 0.105 s a frame | 0.087 → 0.083 |
| 960 x 540 → 3840 x 2160 | 1810 → 1680 ms | 0.263 → 0.250 s a frame | 0.208 → 0.196 |

At 960 x 540, a keyframe's windowed attention takes 162 ms against 277,
and its cross-attention 39 against 52. A frame gains only 2-4%: the
decoder's int8 kernel is faster, but quantizing its four frames' keys
and values each frame costs about what that saves. With `sage_attn`
alone (int8 QK, f16 P*V) the speed is between off and both.

Accuracy, measured as for `i8_gemm` above:

| | a keyframe | a frame | the decoder alone |
|---|---|---|---|
| off | 1.05x | 0.95x | 0.20x |
| `sage_attn` | 1.19x | 1.10x | 1.15x |
| `sage_attn` + `sage_pv8` | 1.73x | 1.31x | 1.15x |

`sage_pv8` costs about what `i8_gemm` `token` does. The two together are
the fastest setting. On a Mac without matrix cores the settings do
nothing, and the stage says so.

### `ane_ffn`: the Neural Engine on an M4

The Neural Engine runs beside the GPU. With `ane_ffn` on, each block's
feed-forward is split by rows: the Neural Engine takes some rows in fp16
while the GPU computes the rest, and the stage balances the split from
the two engines' measured times. Each model gets one Neural Engine
module, staged per block while the block's attention runs on the GPU.

The tier measures whether splitting pays on this machine, and keeps the
GPU alone where it does not:

- On an **M4 Pro**, the keyframe model's feed-forward splits about half
  and half, the Neural Engine as fast as the GPU on its share.
  At 480 x 270 a keyframe takes **2.51 s against 3.35 s**, and a window
  0.47 s a frame against 0.52.
- The decoder's feed-forward is small. On the M4 Pro the split did not pay
  (27.7 ms a block against 26.6 on the GPU alone), so the stage kept the
  GPU.
- On an **M5 Max** neither model's split paid, and both kept the GPU.

The split costs no accuracy measurable on the test clips: keyframes and
frames are as close to the fp32 reference as with the GPU alone (worst
case 1.01x and 0.96x the release's bf16 error).

The two Neural Engine modules are claimed from the memory plan at about
0.5 GB. Once the first window has run, each is booked at what it holds:
less for a small input, whose rows build a smaller module, and nothing
for a module the tier does not build. With `i8_gemm` at `fast` or above
every feed-forward is held int8, which the Neural Engine cannot read, so
the stage claims no module and the feed-forward stays on the GPU.

## How it works

The stream follows the release's `CollaborativeFrameStream` step for step:

1. Frame 0 is a **keyframe**. The large model writes its latent in one step
   from noise, conditioned on the frame upsampled 4x. The decoder **primes**
   it, running it through its own 10 blocks to get every block's keys and
   values.
2. Frames collect into a **window** of 16: frames 0 to 15. Frame 15 is the
   next keyframe.
3. The decoder turns every frame of the window into its 4x frame. Each
   block attends, through a local spatial window, to:
   - the two keyframes, 48 x 48 tokens around the position;
   - the frame before, 24 x 24 tokens;
   - the frame itself, 24 x 24 tokens.
4. The next window starts at the last keyframe. Its first frame was
   already emitted, so it is decoded again only to serve as the next
   frame's history.
5. A clip that ends mid-window is keyframed at its last frame.

The keyframe model keeps the last four keyframes' keys and values in every
block. Its self-attention reads a 22 x 40-token window, shifted inward at
the frame's edges, in each of them. That ring is what grows with the frame
size.

## Accuracy

The port was held to an fp32 restatement of the release's inference code
(`tools/dump_relayvsr_golden.py`, written from the release's code, read and
not run). The bar is the release's own default, bf16. Two real clips were
used: 30 frames with a keyframe every 3 (11 keyframes, so the ring wraps
and its rotary origin moves), and 20 frames at the default window. The
figures are relative L2 error from fp32:

| | vpipe | the release in bf16 |
|---|---|---|
| the decoder (fed the fp32 keyframes) | **0.07%** | 0.50–0.52% |
| a keyframe | 2.2–2.7% | 2.2–2.7% |
| the whole stream | **0.59–0.77%** | 0.80–0.92% |

The decoder runs in f16 (its activations stay below 20), which is why it
lands closer than bf16. The keyframe model runs in bf16, as the release
does. The keyframe noise matches torch's CPU draw bit for bit, which is
what lets the stream be compared frame by frame.

An Apple M5, where the GEMMs and both attentions run on the matrix cores,
gives the same figures: decoder 0.07%, keyframes 1.00–1.05x the release's
bf16 error, stream 0.59–0.77%.

## Limits

- **Memory:** the keyframe model's weights are held whole (8.4 GB), so a
  16 GB Mac is limited to small inputs (about 320 x 180). Streaming its
  blocks from disk, as the other large DiTs here do, is not implemented
  yet.
- On an M5 the decoder's windowed attention is still most of a frame.
  Its int8 form (`sage_attn`) gains little there, because every frame
  quantizes all four frames' keys and values again, including the two
  keyframes that stay the same for the whole window. Keeping each
  frame's int8 copy from the frame that wrote it is not implemented.
- The keyframe noise is torch's CPU generator, not CUDA's, so a frame is
  not bit-identical to the release's GPU output. It matches when the
  release is given the same noise.

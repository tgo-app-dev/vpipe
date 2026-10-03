# LoRA training

VPIPE trains LoRA adapters for its image diffusion models on the Mac that
runs them. A training run is an ordinary pipeline graph: the model's own
text encoder and VAE encode the dataset, and a `train-lora` stage trains
the adapter and saves it where generate-image's LoRA pickers offer it.

Qwen-Image-2.1 (text-to-image) is supported today. The model-specific part
is one interface, so other image families and video families plug into
the same stages.

## The graph

```
model-select --model--> training-dataset
training-dataset --captions--> diffusion-conditioner --conditioning--+
                 --images----> vae-encode -------------latents-------+
                 --manifest------------------------------------------+--> train-lora
model-select ------------------------------------------------model---+     |
optimizer-select ----------------------------------------optimizer---+     |
                                   vae-decode <-- preview latents ---------+
                                       |
                                   save-image
```

A ready graph is in
[`pipelines/qwen-image-2.1-train-lora.vpipeline`](pipelines/qwen-image-2.1-train-lora.vpipeline):
point `dataset.dir` at a folder of pictures, set `train.output_dir`, and
run it.

The same conditioner that serves generation encodes the captions, so the
adapter trains on exactly the conditioning it will later be used with.

A run has two phases:

1. **Encode.** Every caption and picture is encoded once and cached on
   disk. The conditioner and the VAE hold their models across the whole
   dataset and release them when it ends. A later run over the same
   folder encodes only what changed; when nothing did, the encoders never
   load.
2. **Train.** The model loads only now, so it never shares memory with the
   encoders. It then trains from the cache.

## Stages

### `training-dataset`

A folder of pictures (`png`, `jpg`, `jpeg`, `webp`, `bmp`, `tif`), scanned
recursively. It emits three streams:
- the pictures, fitted to aspect-ratio buckets;
- the captions, with the trigger word applied;
- a manifest that tells `train-lora` how the two pair up.

| key | default | meaning |
|---|---|---|
| `dir` | required | the folder. A subfolder named `N_name` repeats its pictures N times per epoch. |
| `captions` | `sidecar` | `sidecar` (a `.txt` beside each picture; several lines give several captions), `jsonl` (`metadata.jsonl` with `file_name` and `text`), or `none` |
| `trigger_word` | — | see below |
| `caption_template` | `{trigger}, {caption}` | how the trigger joins a caption |
| `missing_caption` | `trigger_only` | `trigger_only`, `skip` or `error` |
| `shuffle_tags`, `keep_tags`, `caption_variants` | off, 1, 4 | shuffle comma-separated tags after the first `keep_tags`; each shuffle is encoded up front |
| `resolution` | 1024 | one size or a list, e.g. `[512, 768, 1024]`: each picture is encoded at every size in the list |
| `bucket_step`, `max_aspect` | 32, 2.0 | bucket sides are multiples of `bucket_step`; wider pictures are cropped |
| `flip` | `off` | `random` also encodes a mirrored copy |
| `alpha` | `drop` | `keep` for RGBA pictures |
| `preview_prompts` | the trigger | prompts sampled at every checkpoint; `{trigger}` is expanded |
| `cache_dir` | `auto` | where encodes are kept between runs: `auto` (a folder per dataset under `~/Library/Caches/vpipe/train`), a directory, or `none` |
| `validation` | auto | pictures held out of training and scored at every checkpoint: 1% of a set of 500 or more (at most 64), none below |

### `optimizer-select`

The update rule and the learning-rate schedule, emitted as one beat.

| key | default | meaning |
|---|---|---|
| `optimizer` | `adamw` | `adamw`, `prodigy`, `lion`, `sgd` |
| `learning_rate` | per optimizer | 1e-4 AdamW, 1.0 Prodigy, 3e-5 Lion, 1e-2 SGD |
| `lr_schedule` | `auto` | `auto` (constant for a small dataset, cosine from 64 pictures), `constant`, `linear`, `cosine`, `cosine_restarts` |
| `warmup_steps` | 2% of the run | linear ramp from zero |
| `grad_clip` | 1.0 | global L2 norm; 0 disables it |
| `ema` | 0 | EMA decay of the weights; checkpoints also save the averaged adapter |
| `beta1`, `beta2`, `eps`, `weight_decay`, `min_lr_ratio`, `restarts`, `prodigy_d_coef`, `bias_correction` | | as usual |

The schedule key is `lr_schedule` because "scheduler" means the noise
schedule elsewhere in VPIPE.

### `train-lora`

Its ports:
- **In:** 0 conditioning, 1 latents, 2 manifest, 3 model, 4 optimizer.
- **Out:** 0 preview latents, 1 checkpoints, 2 metrics.

| key | default | meaning |
|---|---|---|
| `output_dir`, `name` | required | `<name>.safetensors`, `<name>-000500.safetensors` per checkpoint, `<name>.resume/` |
| `rank`, `alpha` | 16, = rank | |
| `targets` | `attn_ff` | `attn_ff` (q, k, v, out and the feed-forward) or `attn` |
| `init_from` | — | continue from an adapter of the same targets and rank |
| `steps` / `epochs` | automatic | 100 steps per picture for a small dataset (600 to 3000), two epochs for a large one |
| `batch_size` | 1 | samples accumulated per optimizer step; they run one after another, so a larger batch costs time, not memory |
| `seed` | 0 | sample order, noise, sigmas and dropout all follow from it |
| `caption_dropout` | 0.05 | how often a sample trains against the empty prompt, which keeps guidance working |
| `timestep_sampling` | `auto` | logit-normal pushed through the model's own inference shift at the picture's size; or `uniform`, `logit_normal`, `shift` |
| `save_every`, `preview_every` | automatic | every 250 steps on a small dataset; previews at every checkpoint |
| `preview_steps`, `preview_cfg`, `preview_seed` | 20, 4, 42 | the same seed every time, so previews compare |
| `recompute` | `auto` | keep every block's activations when memory allows, otherwise recompute block by block |
| `ane` | off | M4 series: split the large GEMMs with the Neural Engine |
| `i8_gemm` | off | M5 and later: int8 matrix-core GEMMs (lossy) |
| `resume` | `auto` | continue a stopped run of the same name |
| `cache` | `auto` | where the encoded dataset lives while training: `ram`, `disk` (read from the cache every step), or `auto` (RAM when it fits an eighth of the machine's memory) |
| `register` | on | add the finished adapter to the model registry |

## The trigger word

- The template applies unless a caption already contains the trigger as a
  whole word, ignoring case. A `[trigger]` placeholder in a caption is
  replaced where it stands.
- A picture with no caption is captioned with the trigger alone.
- Tag shuffling never moves the trigger, because it sits in the kept
  prefix.
- Caption dropout replaces the whole caption, trigger included, with the
  empty prompt.
- The saved adapter records the trigger in its metadata, as
  `modelspec.trigger_phrase` and `vpipe.trigger_word`. The registry entry
  records it too.

Choose a trigger that is not an ordinary word, such as `ohwx` or `sks`. A
word the model already knows is a weak key.

**Caption what should stay promptable, and leave the subject to the
trigger.** The trigger learns only what the caption does not already
explain. Measured on six pictures of one red sailboat, 300 steps each:

| prompt | captioned `ohwx, a red sailboat on the sea at sunset` | captioned `ohwx on the sea at sunset` |
|---|---|---|
| `ohwx` | an unrelated scene | the boat |
| `ohwx in a snowy harbour` | the boat | the boat |
| `ohwx on a mountain lake at dawn` | the lake, no boat | the lake, no boat |

Both adapters carry the sunset light everywhere. Neither carries the boat
into a scene that has nothing to do with the pictures after 300 steps.
The default length for six pictures is 600 steps.

## Small and large datasets

**Small (up to a few dozen pictures)** is the common case: a subject or a
style.
- The defaults train by steps: about 100 per picture, at a constant rate.
- Training at several resolutions (`resolution: [512, 768, 1024]`) helps
  an adapter learned from few pictures generalise. It costs encode time
  only.
- Save often (the default is every 250 steps) and pick the checkpoint
  whose previews look best.

**Large (thousands of pictures)** trains by epochs instead. For large
runs:
- The schedule turns cosine on its own (`lr_schedule: auto`); add an EMA
  for long runs.
- Keep several captions per picture.
- Batch sizes above 1 smooth the gradient.

The encoded dataset costs about 1–2.5 MB per picture and caption at
1024². Small sets train from RAM; a set too large for an eighth of the
machine's memory trains from the disk cache, reading 1–2.5 MB per step:
milliseconds, against steps of seconds. An adapter trained from the disk cache is identical, bit for
bit, to one trained from RAM.

**Validation.** A large set holds out 1% of its pictures (at most 64).
They are never trained on. At every checkpoint each is scored at a fixed
noise level and noise draw, so the held-out loss moves only when the
adapter does. It is logged, and reported on the checkpoint beat as
`val_loss`. Stop, or pick an earlier checkpoint, when it turns up while
the training loss keeps falling.

## The encode cache

The cache is keyed so that it is safe to reuse:

- **Captions** are keyed by their final text, after the trigger is
  applied. Changing the trigger or the template re-encodes them.
- **Pictures** are keyed by file, size and modification time, and by
  bucket, flip and alpha mode. Editing one picture re-encodes that
  picture alone.
- **The model.** Each cache sits under a folder named for the model, so
  a different checkpoint never reads another's encodes.

Delete a cache folder, or set `cache_dir: none`, to force a full encode.
Keep the cache on the internal SSD, since the encode phase writes it at
full speed. Two runs must not fill one cache at the same time.

## Memory

The two phases never coexist:

| phase | holds |
|---|---|
| encode | the text encoder (resident if it fits, otherwise streamed layer by layer and run over the captions in batches) and the VAE encoder |
| train | the DiT (resident, or block-streamed when it does not fit), the step's activations, the adapter's state, the encoded cache; previews add a VAE decode |

The encoders are released when their streams end, and the model loads
only after that: training never starts beside a leftover encoder.

**A text encoder that does not fit** (16 and 24 GB machines, for
Qwen-Image-2.1's 16.7 GB encoder) streams its layers from disk. One
caption at a time, that re-reads the whole encoder for every caption. So
the conditioner holds a training dataset's captions back and encodes them
32 at a time, layer by layer: each layer is read once and run over every
caption in the batch. The conditioning is identical, bit for bit, to
what the resident encoder produces. Measured:

| machine | captions | one at a time | one batch |
|---|---:|---:|---:|
| M5 Pro 24 GB | 5 | 13.2 s | 3.2 s |
| M5 Pro 24 GB | 10 | — | 5.7 s |
| M5 MacBook Air 16 GB | 5 | 15.4 s | 4.0 s |

Activations are the variable term. Measured for Qwen-Image-2.1, rank 16:

| | 512² | 1024² |
|---|---:|---:|
| recompute per block | 0.6 GB | 2.3 GB |
| keep every block (`recompute: never`) | 5.9 GB | 21.6 GB |

The adapter's state (fp32 weights, gradients and AdamW moments) is about
0.75 GB at rank 16 on `attn_ff`. `recompute: auto` keeps every block's
activations whenever the model, those activations, the state and the cache
fit with headroom. That saves about a quarter of every step. Everything is
judged against physical RAM, so the same graph makes the same choices on
every run.

## Speed

Qwen-Image-2.1, rank 16 on `attn_ff`, one sample per step. Each figure
alternates the arms within one sitting.

| machine and configuration | 512² | 1024² |
|---|---:|---:|
| M4 Pro 64 GB: keep every block | 5.6 s | — |
| M4 Pro 64 GB: keep every block + `ane` | — | 21.4 s |
| M5 Pro 24 GB: recompute (keeping does not fit) | 2.5 s | 9.5 s |
| M5 Pro 24 GB: recompute + `i8_gemm` | 2.5 s | 8.8–9.2 s |

On the M4, the Neural Engine pays off at 1024². At 512² there are too few
rows per GEMM for it to help.

On the M5, the attention backward and the int8 path both run on the
GPU's matrix cores. `i8_gemm` applies to the backward as well as the
forward. It gains about 6% at 1024² and nothing at 512², and a short
training run learns 98% of what bf16 does over the same steps.

## Checkpoints, previews and resuming

- **Checkpoints.** Each is a standard diffusers/PEFT adapter file
  (`transformer.<module>.lora_A.weight` / `lora_B.weight`, alpha and rank
  in the metadata). VPIPE's LoRA fields, diffusers and ComfyUI all read
  it.
- **Reloading.** An adapter saved and loaded back runs exactly the
  forward it was trained with.
- **Previews.** Each checkpoint samples every preview prompt with the live
  adapter, guided against the empty prompt. `vae-decode` and `save-image`
  turn them into pictures.
- **Stopping and resuming.** A stop saves the adapter and the full
  optimizer state. Rerunning the graph with `resume: auto` continues from
  there and gives the same result as a run that never stopped: noise,
  sigmas and sample order are functions of the seed and the sample
  number.
- **Metrics.** One beat per step on the metrics port: loss, learning rate,
  gradient norm and step time.

### While it runs

`train-lora` takes two commands between its optimizer steps:

- **`save`** writes a checkpoint now, with its resume state, and emits
  it on the checkpoints port as a scheduled one would be. The reply names
  the file, the step and the mean loss since the last checkpoint.
- **`preview`** samples the preview prompts now with the live adapter.
  Optional arguments pick one prompt (`prompt`, from 0) and override
  `steps`, `cfg` and `seed`.

In the web UI they are buttons in the stage's panel while the pipeline
runs. From Python:

```python
train = pipeline.stage("train")
train.call("save")                       # {"path": ..., "step": ...}
train.call("preview", args={"prompt": 0, "steps": 12})
```

A command sent while the dataset is still encoding is refused with the
reason. One sent during a step is answered when the step ends.

## What is not supported yet

- Edit-pair training: condition pictures plus an instruction.
- Training the text encoder.
- Adapter types other than LoRA.
- A batched encode for captions with reference pictures, or for families
  other than Qwen-Image-2.1. Those encode one caption at a time on a
  streamed encoder.

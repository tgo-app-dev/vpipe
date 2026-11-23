# LoRA training

VPIPE trains LoRA adapters for its image diffusion models on the Mac that
runs them. A training run is an ordinary pipeline graph: the model's own
text encoder and VAE encode the dataset, and a `train-lora` stage trains
the adapter and saves it where generate-image's LoRA pickers offer it.

Two families train today: Qwen-Image-2.1 (text-to-image) and MiniMax-H3
(its FL2VA and Ref2VA video-and-audio DiTs, on pictures; see
[MiniMax-H3](#minimax-h3)). The model-specific part is one interface, so
other image and video families plug into the same stages.

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

## MiniMax-H3

[`pipelines/minimax-h3-train-lora.vpipeline`](pipelines/minimax-h3-train-lora.vpipeline)
is the same graph on MiniMax-H3: the conditioner encodes the captions with
H3's Qwen3-VL-32B, vae-encode encodes each picture with H3's video VAE,
and train-lora trains the adapter that generate-video's LoRA pickers then
offer.

**A picture is a one-frame clip.** The video VAE encodes a still as one
latent frame, and each training sample is the text-to-video-and-audio
request generate-video would build for it: the caption's rows, the
picture's rows, and the two audio latents a one-frame clip carries. A
sample with no soundtrack keeps those audio rows, as inference lays them
out, but leaves them out of the loss, so the adapter is not taught to
make silence.

**The sigmas follow inference.** One draw sets both: the video's sigma
is the draw through H3's shift of 12 (what `timestep_sampling: auto`
gives), and the audio's is the same draw through the audio's shift of 3.

**The AdaLN is the memory problem.** About 13 B of the DiT's 33 B
parameters are its per-block AdaLN projections. Inference bakes them away
for each schedule, but a training step draws a fresh sigma, so they must
stay. Where the whole bf16 DiT (62 GB) fits beside a step, as on a 128 GB
or larger Mac, it is kept as released. Where it does not, each projection
is folded at load onto the time embedding's 8-dimensional curve, which
leaves 174 MB in place of 26 GB. That pruned forward lands 0.46% from the
released one, and slightly closer to the fp32 reference. The load line
says which was chosen.

**Previews** run generate-video's own sampler with the live adapter: one
frame at the first picture's size, `preview_steps` steps at H3's own
shifts. H3 is guidance-distilled, so `preview_cfg` is not used. vae-decode
decodes the one-frame preview latents as pictures.

**The adapter file** uses the diffusers spelling that H3's runtime binder
reads (`transformer_blocks.N.attn.to_q`, `to_k`, `to_v`, `to_out.0`,
`ff.net.0.proj`, `ff.net.2`, and the token refiner's two blocks). It
binds on MiniMaxAI's per-head checkpoint and on Comfy-Org's flat repack
alike. Its metadata records the partition it was trained on
(`vpipe.partition`) and the AdaLN form (`vpipe.adaln`). The registry
records it as `minimax-h3-lora` under that partition.

**Measured** on an M5 Ultra 256 GB: the bf16 FL2VA DiT resident with
its AdaLN kept, rank 32 on all 312 projections, pictures at 512², AdamW
at 2e-4. A step takes 1.2 s (0.3 s forward, 0.8 s backward), and 300
steps with six previews took 7 minutes at a peak of 70 GB. The subject
was six crops of one photo of a figurine. For "ohwx figurine", the base
model draws an unrelated toy figurine; after 300 steps it draws the
trained figurine, both in the previews and in a generate-video clip
with the adapter loaded. Six crops of one indoor photo overfit, though:
the clip kept the subject and lost the snowy setting its prompt asked
for.

### Clips and their soundtracks

[`pipelines/minimax-h3-train-lora-clips.vpipeline`](pipelines/minimax-h3-train-lora-clips.vpipeline)
trains on video. Set `media: clips` on the dataset. Each clip is resampled
to `fps` (24, H3's rate) and cut into windows of `frames` (17n + 5 pixel
frames, the counts H3's video VAE makes; 39 by default). A clip uses the
longest window it fills, and `windows` of them spread evenly across it.
Each window is cropped to its bucket and encoded as one clip latent.

With `audio: track`, each window's slice of the soundtrack goes out on
the dataset's port 3, at 32 kHz stereo (a mono track is doubled). The
graph's `audio-vae-encode` encodes it, and train-lora takes it on port 5.
A window with a soundtrack trains both losses (`audio_loss_weight`, 1 by
default). One without keeps its audio rows out of the loss, as a still's
are. Previews come out as clips, with their soundtrack on train-lora's
port 3 for `audio-vae-decode`.

**Measured** on the M5 Ultra: three fox clips of a few seconds, two
39-frame windows each at 512 × 288 with their soundtracks, rank 32. A
step takes 3.6 s (1.0 forward, 2.5 backward), and 300 steps with
previews at 150 and 300 took 19.5 minutes at a peak of 71 GB. For "ohwx
sitting on a wooden porch" the previews draw the trained fox on a porch.
Every caption said "snowy forest", though, so by step 300 the beach
prompt also came out snowy.

### Keyframes (FL2VA)

`task_mix` on the dataset trains FL2VA's keyframe tasks, for example
`{t2va: 1, i2va: 1, l2va: 0.5, fl2va: 1}`. Each window draws its task
from its own key. The task's keyframes (the window's first and/or last
frame) go out after it as stills, are encoded as generate-video encodes a
keyframe, and are pinned as the sample's conditioning rows.

Wire the dataset's port 4 to the conditioner's `ref_images` to also show
the keyframes to Qwen3-VL, as generate-video's keyframe graphs do. The
window's captions are then encoded again, presented with its frames.

### References (Ref2VA)

[`pipelines/minimax-h3-train-lora-ref2va.vpipeline`](pipelines/minimax-h3-train-lora-ref2va.vpipeline)
trains Ref2VA. Give a sample its references in a `<stem>.refs` file
beside it: one path per line, relative to the sample's folder or the
dataset's, `#` for comments. Or use a `references` list on its
`metadata.jsonl` row. They are named in the order the model should read
them, as video-ref-encoder's `references` are. A folder called
`references` is never scanned for samples, so the files can live in the
dataset.

`video-ref-encoder` replaces the conditioner in this graph. Each caption
beat carries the sample's references and its duration (the window's
frames, 1 for a picture), so each caption is that sample's own Ref2VA
request. The prompt is presented with the pictures' vision blocks, and
the references are encoded by both VAEs. Set `references_from_prompt` so
the plan books a clip's presentation. The encoder's two row ports go to
train-lora's 6 and 7. The rows are cached with their caption, and the
cache key includes the reference files, so editing one re-encodes it.

Each step packs generate-video's reference layout for that request, with
the reference rows as its conditioning. Previews are prompt-only. A
caption mentions a reference as H3's prompts do: "the fox from
`<Picture 1>`...".

A reference picture's size is the encoder's `reference_image_short_edge`.
At its default of 2048, one 16:9 still is about 7,200 rows. That is
beside the shipped graph's 1,584-row target (39 frames at 512 × 288), so
the graph uses 1024 (about 1,800). Present the references at
the same size at inference.

**Measured** on the M5 Ultra: the three fox clips, each with a still of
another fox clip as its reference (at a 512 short edge: 448 rows), 39
frames at 512 × 288, rank 32. The encode took 12.9 s for the six
captions. A step takes 4.6 s (1.3 s forward, 3.3 s backward), and 200
steps took 17 minutes. A Ref2VA clip from one of the stills, seed 6:
the base model already carries the referenced fox onto the prompt's
porch. With the adapter loaded, the clip moves toward the training
clips' snowy pine forest and their fox. Three clips of one setting teach
mostly that setting.

### The two image models

**MiniMax-H3-Image** has no audio branch and one output head per sampling
step. Each sample trains one of its release's six steps: that step's
sigma as the release runs it, through that step's head, drawn from the
sample's seed. Previews run the release's grid, one head per step, in
the velocity form. The text encoder is H3's Qwen3-VL-32B: name it in the
conditioner's `encoder_dir`.

**P-H3-Image** is FL2VA under PrunaAI's adapter. Name FL2VA's DiT in
`dit_dir` and the adapter in `base_lora`. The adapter is held frozen in
the slot below the new one, and the step's backward runs through it.
Samples are pictures, and previews run its manifest's grid. The new
adapter's metadata records the base adapter (`vpipe.base_adapter`), which
it must be stacked on at inference.

`base_lora` works on the video DiT too, to train an adapter on top of
another one (a style, say).

**Measured** on the M5 Ultra: MiniMax-H3-Image, the same six crops at
1024², rank 32. A step takes 2.4 s (0.7 forward, 1.7 backward). At step
100 the previews draw the trained figurine in both prompts' settings.

**`i8_gemm`** takes the forward's GEMMs, and the backward's recomputed
forward, wherever a GEMM clears the int8 floor of 1024 rows. The
backward's own dX GEMMs stay dense. A 512² still is 256 video rows, so
the GEMMs never clear the floor and nothing changes. A 39-frame 512 × 288
clip is 1,584 rows. On the fox clips it measured 3.0 s a step against
3.3–3.4 s (1.1×), with step 30's mean loss 0.510 against 0.505.

### Editing pairs (the two image models)

[`pipelines/minimax-h3-image-train-lora-edit.vpipeline`](pipelines/minimax-h3-image-train-lora-edit.vpipeline)
trains an edit. Each pair is the edited picture as the sample, its
instruction as the caption, and the picture it was edited from named in a
`<stem>.refs` beside it. That is the same file the Ref2VA references use;
several lines give several pictures, the first the one edited. Keep the
sources in the dataset's `references` folder, which is not scanned for
samples.

`video-ref-encoder` takes the conditioner's place, with the model's own
edit presentation ("Picture 1: ..." on MiniMax-H3-Image, "Edit
`<Picture 1>`..." on P-H3-Image) and its rules for sizing the picture.
Each step packs generate-video's edit layout: the picture's rows ahead of
the generated one. MiniMax-H3-Image's arrive clean and are noised here to
its 0.999, from the sample's own noise, as generate-video noises them.
P-H3-Image's arrive already noised by the encoder. A caption with no
picture (a preview prompt) trains text-to-image, which the batch lets
video-ref-encoder encode. Set video-ref-encoder's `seed`: it is the
posterior's, as in an edit graph.

**Measured** on the M5 Ultra, MiniMax-H3-Image. The edit was one the
base model cannot know: "apply the ohwx filter" swaps red and blue. Six
figurine crops were the sources, their swapped copies the targets, at
1024²; each source was presented at the model's 1024 short edge (1,280
rows). Rank 32, every block kept (a 46.7 GB arena). A step takes 4.9 s,
and 300 steps took 25 minutes, with the mean loss falling from 0.48 to
0.066. Asked to apply the filter to a photo it never saw (a fox in the
snow), the base model returns the fox unchanged and orange. With the
adapter it returns the same fox and scene, blue. The output's
correlation with the swapped photo rose from 0.86 to 0.95.

### Keeping every block (`recompute: never`)

By default the backward recomputes each main block from its input. With
`recompute: never`, or under `auto` where the box holds it, the forward
runs every main block through the trainer's own block code and keeps
what its backward reads, so nothing is recomputed. The token refiner's
two blocks, over the text rows only, still are. It is the same
arithmetic: against recompute the losses, velocities and gradients agree
to the bit, and an adapter trained either way is byte-identical. It
costs about 228 KB per sequence row per block, so 50 blocks of a
2,000-row clip window is about 24 GB. The load line says which was chosen.

**Measured** on the M5 Ultra, the fox clips: 2.6 s a step kept against
3.4 s recomputed (1.31×), with a 23.9 GB arena. With the engine's tuners
pinned (`VPIPE_H3_NO_QMM_AUTOTUNE=1 VPIPE_H3_NO_ATTN_TILE_AUTOTUNE=1`),
ten steps report the same losses and save byte-identical adapters.
Unpinned, two runs of either mode differ in the fourth digit, because
the tuners vote run to run.

Not yet for H3: `ane`.

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
| `media` | `auto` | `pictures`, `clips` (`mp4`, `mov`, `mkv`, `webm`, ...) or both |
| `fps`, `frames`, `windows` | 24, [39], 1 | clips: the rate they are resampled to, the window lengths (17n + 5), and how many windows per clip |
| `audio`, `audio_sample_rate` | `track`, 32000 | `track` sends each clip window's soundtrack on port 3 (when the port has a consumer) |
| `task_mix` | t2va only | FL2VA's keyframe tasks and their weights |

### `optimizer-select`

The update rule and the learning-rate schedule, emitted as one beat.

| key | default | meaning |
|---|---|---|
| `optimizer` | `adamw` | `adamw`, `adamw8bit`, `adafactor`, `prodigy`, `lion`, `sgd`. `adamw8bit` holds AdamW's two moments as one-byte codes, block-quantized per 256 (the dynamic map of Dettmers et al.'s 8-bit optimizers): 2 bytes of state per parameter instead of 8, with the update itself in f32. `adafactor` is transformers' Adafactor at a manual learning rate (`relative_step` and `scale_parameter` off, no first moment, so `beta1` is unused): each factor's second moment is kept as a row and a column vector, almost no state |
| `loraplus_lr_ratio` | 1 | LoRA+: the B factors learn this many times faster than the A factors (the paper suggests 16; 1 is off) |
| `learning_rate` | per optimizer | 1e-4 AdamW (8-bit too) and Adafactor, 1.0 Prodigy, 3e-5 Lion, 1e-2 SGD |
| `lr_schedule` | `auto` | `auto` (constant for a small dataset, cosine from 64 pictures), `constant`, `linear`, `cosine`, `cosine_restarts` |
| `warmup_steps` | 2% of the run | linear ramp from zero |
| `grad_clip` | 1.0 | global L2 norm; 0 disables it |
| `ema` | 0 | EMA decay of the weights; checkpoints also save the averaged adapter |
| `beta1`, `beta2`, `eps`, `weight_decay`, `min_lr_ratio`, `restarts`, `prodigy_d_coef`, `bias_correction` | | as usual |

The schedule key is `lr_schedule` because "scheduler" means the noise
schedule elsewhere in VPIPE.

### `train-lora`

Its ports:
- **In:** 0 conditioning, 1 latents, 2 manifest, 3 model, 4 optimizer;
  for MiniMax-H3 also 5 audio latents (clips' soundtracks), 6 and 7 the
  references' video and audio rows (Ref2VA, from video-ref-encoder).
- **Out:** 0 preview latents, 1 checkpoints, 2 metrics, 3 preview
  soundtracks (MiniMax-H3 clips).

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
| `audio_loss_weight` | 1 | MiniMax-H3: the soundtrack's loss beside the video's |
| `dit_dir` | — | the DiT that runs, when `hf_dir` names one it does not hold (P-H3-Image: FL2VA's) |
| `base_lora` | — | MiniMax-H3: frozen adapters under the trained one (P-H3-Image's), a path or a list |

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

What the plan books, beyond the DiT:
- **Everything the run holds beside the DiT** is claimed at launch: each
  frozen `base_lora` adapter at every factor the binder holds (its
  streamed set when the DiT streams), and MiniMax-H3-Image's per-step
  heads.
- **The encoded samples kept in RAM** count in the training arena with
  the activations and the adapter's state.
- **A clip's decode** holds only the source frames its windows read, not
  every frame up to the last window, and is booked before it runs. A
  42-second 1080p clip with two windows peaked at 720 MB, against 18.3 GB
  when every frame was kept, and the windows are byte-identical either
  way.
- **audio-vae-encode** books MiniMax-H3's encode at the longest soundtrack
  it has seen.
- **The optimizer's state** is booked by its own size: 8 bytes a parameter
  for AdamW, 3 for `adamw8bit`, at most 8 / rank for `adafactor`.

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
- **Metrics.** One beat per step on the metrics port: the step and the
  run's total, loss, learning rate, gradient norm and step time.
- **Progress.** The stage opens a progress report while it encodes,
  `lora encode` (the encodes taken of those the manifest wants), then one
  while it trains, `lora train` -- opened as the model loads, with no
  count until the run's total is known, then the steps done of it, the
  loss and its mean since the last checkpoint in the detail; both are
  closed when the run ends, fails or is stopped: what the console footer, the web
  UI's status bar and `SessionIntf::progress()` show.

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

- Edit-pair training for families other than the two MiniMax-H3 image
  models.
- Training the text encoder.
- Adapter types other than LoRA.
- A batched encode for captions with reference pictures, or for families
  other than Qwen-Image-2.1. Those encode one caption at a time on a
  streamed encoder.

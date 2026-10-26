# Chat with Qwen3.8-Flash-Next (text and images) on Apple Silicon

**Qwen3.8-Flash-Next** is a sparse mixture-of-experts vision-language
model: about 180 billion parameters stored, of which a token reads a small
fraction -- ten of 512 experts a layer, a shared expert, and one row of a
51-billion-parameter n-gram table. It reads pictures as well as text and
carries its own speculative-decoding head. vpipe runs all of it on-device
through its **metal-compute** backend -- its own Metal kernels, no Python
and no third-party tensor runtime in the forward pass.

The checkpoint the pipelines name is
`mlx-community/Qwen3.8-Flash-Next-Uncensored-oQ5e-mtp`, an MLX
mixed-precision pack: the experts and the n-gram table at 5 bits, the rest
at 5 to 8 bits, the vision tower in bf16. It is 128.5 GB.

## Other packs of the model

vpipe reads the model's other MLX packs too; put the repo's name where the
pipelines name the oQ5e one (`model_path` in the prepare pipeline, `hf_dir`
in the chat pipeline).

| Pack | Size | Its layout |
|---|---|---|
| `mlx-community/Qwen3.8-Flash-Next-4bit-mtp` | 113 GB | everything 4-bit at group 32, the routers 8-bit; its MTP head's attention and shared expert in bf16 |
| `mlx-community/Qwen3.8-Flash-Next-4bit` | 112 GB | the same without the MTP head |
| `mlx-community/Qwen3.8-Flash-Next-oQ6e-mtp` | 151 GB | as the oQ5e pack, the experts at 6 bits |
| `mlx-community/Qwen3.8-Flash-Next-oQ8e-mtp` | 195 GB | the experts at 8 bits |

Any other MLX affine pack of it loads as well, whatever its widths and
groups -- with two limits. Its routed experts must be 4, 5, 6 or 8-bit at
group 32 or 64, one width throughout (the 2-bit OptiQ pack and 3-bit packs
are refused, by name). And on a box too small to hold every expert, the
expert cache needs one width and group across the layers; a pack that
mixes them loads its experts resident instead.

The 4-bit packs hold their attention and gating weights at 8 bits once
loaded (the kernels' one width), about 6 GB where they take 3 on disk;
vpipe sizes the expert cache with that in mind. The load reports the layout
it found, e.g. `Qwen3.8-Flash-Next weights: projections w8g32; router w8g32,
experts w4g32, shared expert w8g32 (fused decode); MTP head: fc dense, ...`.

## What you need

| | |
|---|---|
| **Machine** | Apple Silicon Mac (M-series). |
| **Memory** | 64 GB is comfortable. The model never holds all of itself: the n-gram table is read a row at a time from disk, and the routed experts (83 GB) live in a cache sized to the box -- the most-selected kept, the rest read from disk when a token picks them. Everything else is ~6 GB. |
| **Disk** | **~130 GB**, on the internal SSD: a token that misses the expert cache reads from the drive, so the drive's speed is the decode's speed. |
| **Hugging Face** | Nothing. The repo is public and needs no account or token. |

## The pipelines

- **[`prepare-qwen38-flash-next.vpipeline`](pipelines/prepare-qwen38-flash-next.vpipeline)**
  -- download the model and register it. Run once.
- **[`qwen38-flash-next-chat.vpipeline`](pipelines/qwen38-flash-next-chat.vpipeline)**
  -- the chat, text and images.
- **[`prepare-qwen38-flash-next-dflash2.vpipeline`](pipelines/prepare-qwen38-flash-next-dflash2.vpipeline)**
  and **[`qwen38-flash-next-dflash2-chat.vpipeline`](pipelines/qwen38-flash-next-dflash2-chat.vpipeline)**
  -- the same with the DFlash 2 drafter, which decodes faster (below).

Run either with `vpipe --launch <file>` from your work directory, or open it
with **Load** in the web UI's Pipeline Manager. The steps are the same as in
the [Qwen3.5 9B chat guide](QWEN35-CHAT.md), which explains the work
directory, the five-stage chat graph and the web UI in detail.

```sh
cd ~/vpipe-work
~/src/vpipe/build/apps/vpipe/vpipe --launch prepare-qwen38-flash-next.vpipeline
~/src/vpipe/build/apps/vpipe/vpipe --launch qwen38-flash-next-chat.vpipeline
```

## Chatting about a picture

`media: true` on `text-input` lets a turn carry an attachment. In a terminal
you type the marker around a path; in the web UI the attach and
drag-and-drop controls build it for you:

```
you> <|__vpipe_fs_im_start__|>/path/to/panda.png<|__vpipe_fs_im_end__|> Describe this picture in two sentences.
This watercolor illustration depicts a charming giant panda with soft
black-and-white fur, gently holding and nibbling on a fresh green bamboo
stalk. The panda's expressive eyes and slightly open mouth give it a
friendly, endearing appearance against a clean white background.
```

A picture becomes one token per 32x32 square, at its own size within the
model's processor bounds: from 65,536 pixels (a smaller one is enlarged)
to 16.7 million (a larger one is shrunk). A 1024x1024 picture is 1024
tokens; 2048x2048 is 4096 tokens, read in about 85 s on an M4 Pro; the
largest, about 16,000 tokens, needs ~5 GB beside the model while the
vision tower runs, which the chat books when it starts.

## Settings that matter

- **Speculative decoding** (`mtp`, on by default). The model's own draft
  head guesses ahead and the model checks the guesses in one pass,
  keeping the ones it agrees with. How many it guesses (none to three) is
  measured as it decodes: what each extra checked row costs against a
  plain step, and how often a guess at each position is kept -- the
  answer moves with the machine, how much of the model is in memory, and
  the text. The output is the model's own, token for token, greedy or
  sampled; only the speed changes. The draft head is primed with the
  prompt as it is read (`mtp_prefix_seed`, on by default).
- **Context copy** (with the draft head; `VPIPE_QWEN_HC_COPY=0` turns it
  off). When the reply's last six tokens occurred earlier in the
  conversation, the next pass checks the tokens that followed them there
  -- up to seven -- instead of the head's guess: a reply that quotes,
  rewrites or re-emits what it was given goes several tokens a pass. Its
  passes are timed against the head's, and while they cost more for each
  token kept it stands aside. The output is still the model's own.
- **Sampling.** The shipped pipeline uses the vendor's instruct-mode
  settings (temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5)
  with thinking off. For thinking mode the vendor recommends temperature
  1.0, top-p 0.95, top-k 20 and no presence penalty.
- **Context.** `page_tokens` x `max_pages` is how long a conversation can
  grow. Past 2051 tokens the model's full-attention layers stop attending
  every earlier token: a small indexer ranks blocks of four and each query
  reads the best 512 blocks, as the model was trained to. vpipe does the
  same, an image's blocks ranked at the image's positions.
- **Precision.** `compute_dtype: bf16` is the model's own. `f16` measured
  closer to the full-precision reference on every check here (its values
  stay far inside f16's range).

## Faster decoding: the DFlash 2 drafter

[`incoai/Qwen3.8-Flash-Next-DFlash2`](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)
is a five-layer DFlash 2 drafter (1.2 GB) trained for this model. It reads
the model's own hidden states and drafts up to seven tokens in one forward;
the model checks the whole block at once and keeps the longest prefix it
agrees with. As with the built-in draft head, **the output is the model's
own, token for token**, greedy or sampled; only the speed changes. It
takes over from the draft head when it is set (`draft_model` on
`text-chat`, as the DFlash 2 pipelines do), and
[the Qwen3.8-27B DFlash 2 guide](QWEN38-DFLASH2.md) describes the settings
(`draft_block_size`, `draft_bits`) and how a round works.

Two things are particular to this model. The drafter conditions on five
of its layers (5, 15, 25, 35 and 45), each one's output averaged over the
model's four residual streams -- what the drafter was trained on. And
because the model hashes the token ids themselves into its n-gram table,
a round reads the drafts back before the model checks them, so the
drafter's forward and the check are two command buffers rather than one.

## What to expect

Measured on an **M4 Pro, 64 GB**, the checkpoint on the internal SSD, the
expert cache ~46 GB, 128 greedy tokens:

| | |
|---|---|
| decode right after the first prompt (the cache still filling) | ~11 tok/s (88-97 ms a token); ~13 with the draft head |
| decode, experts already cached | ~23 tok/s (44 ms a token); ~27-29 with the draft head |
| an 8,192-token prompt, the cache empty | 34-38 s |
| a 16,384-token prompt, the cache empty | 59 s |
| a 982-token image turn: prefill | 28 s |

With the DFlash 2 drafter, on a math word problem through the chat
template (greedy, the drafter held at 8 bits, which drafts as well as its
stored bf16 here):

| | decode | tokens a round |
|---|---:|---:|
| a fresh conversation (the cache still filling), `text-chat` | 9.4 tok/s | 3.4 |
| experts cached, plain decode | 20 tok/s (50 ms a token) | 1 |
| experts cached, block 4 | 32 tok/s (31 ms a token) | 3.3 |
| experts cached, block 8 (the drafter's trained block) | 21 tok/s (47 ms) | 4.8 |
| experts cached, adaptive (`draft_block_size: 0`, the default) | 31 tok/s (32 ms) | 3.2 |

A longer block keeps more tokens a round, but every row it checks routes
to ten experts in every layer, so a round's cost grows with its length --
91 ms at block 4 against 213 ms at block 8 here -- and the adaptive length
settles near 4.

The decode's cost is the drive's: a token that picks an expert the cache
lacks waits for a ~3 MB read. A machine whose memory holds every expert
decodes several times faster.

Measured on an **M5 Ultra, 256 GB**, every expert in memory (the oQ5e
checkpoint, greedy, the draft head pinned to one guess a pass -- the
count that measured fastest there):

| | plain decode | with the draft head |
|---|---:|---:|
| a 300-word story from a one-line prompt | 89.8 tok/s | 109 tok/s |
| a summary of a 7,257-token document | 82.7 tok/s | 102.5 tok/s |
| the same summary asked of that document four times over (29,000 tokens) | 81.0 tok/s | 104.2 tok/s |
| copying a 3,000-character passage it was given, one word changed | 79.7 tok/s | 184.6 tok/s (118.0 without context copy) |

A 29,000-token prompt is read at about 2,900 tok/s. The DFlash 2 drafter
decodes the story at 89.4 tok/s and the 7,257-token summary at 66.2: with
every expert in memory the draft head is the faster drafter, and the
DFlash drafter's own attention over a long context costs it most.

Correctness is checked against the reference implementation (transformers'
`qwen4_exp`, its layers run one at a time from the same checkpoint;
`tools/dump_qwen4_exp_golden.py`), with the reference's own bf16 run as the
bar: an 86-token conversation's logits as close to its full-precision run
as the reference's bf16 is, and every decisive pick; a 2422-token one
through the sparse attention, its block picks as close as the reference's
bf16 picks are; the draft head's predictions likewise. A picture turn: the
vision tower ten times closer to the reference's full-precision run than
its bf16 run is, and the language model reading it -- a short turn whose
every row is compared, and the picture before a 2400-token prompt, the
picture's blocks ranked by the sparse attention -- within the reference's
bf16 distance. Speculative decoding -- the draft head and the DFlash 2
drafter alike -- keeps the plain decode's tokens exactly, greedy and
sampled. The drafter itself is held to its reference implementation
(z-lab's `model.py`) on a synthetic context, closer to the reference's
full-precision run than the reference's own bf16 run is.

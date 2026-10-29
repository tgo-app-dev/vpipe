# Chat with Ternary-Bonsai 27B (text and images) on Apple Silicon

**Ternary-Bonsai** is Prism ML's family of *ternary* language models: every
weight of the language model -- attention, the feed-forward, the token table
and the output head -- is one of {-1, 0, +1} times a 16-bit scale shared by
128 weights. A 27-billion-parameter model is held in **7.7 GB**, so a
27B-class reasoning model with a vision tower runs on a laptop. Both models
here are Apache-2.0 and need no account to download.

vpipe runs both on its **metal-compute** backend (its own Metal kernels, no
Python and no third-party runtime in the forward pass), in `text-chat` and in
`realtime-vqa`:

| | Ternary-Bonsai 1 | Ternary-Bonsai 2 |
|---|---|---|
| Base model | Qwen3.6-27B | Qwen3.8-27B |
| Checkpoint | `prism-ml/Ternary-Bonsai-27B-mlx-2bit` (MLX) | `prism-ml/Ternary-Bonsai-2-27B-gguf` (GGUF) |
| Download | 8.5 GB (one file, tower inside) | 5.9 GB (`PTQ1_0`) or 7.2 GB (`PQ2_0`) + 0.9 GB projector |
| Held in memory | 7.7 GB + 0.9 GB tower | 7.7 GB + 0.9 GB tower |

The two GGUF packings of Bonsai 2 hold **the same values** -- every tensor
decodes identically -- and vpipe holds both the same way once loaded, so
`PTQ1_0` is simply the smaller download.

## What you need

| | |
|---|---|
| **Machine** | Apple Silicon Mac (M-series). |
| **Memory** | **24 GB or more** is comfortable (a 24 GB M5 Pro runs it with no swap). The model holds 8.7 GB with its tower, plus 64 MB of K/V per 1024 tokens of conversation (2 GB at the shipped pipelines' 32k-token context). A **16 GB** Mac runs it with little else open: on a 16 GB MacBook Air (M5) every check below passed, the process at 8.6-10.1 GB and under 1 GB pushed to swap. With other work holding 7.8 GB of that Air's memory wired, the model did not fit and decoding stalled in swap. |
| **Disk** | 8.5 GB (Bonsai 1), 6.8 GB (Bonsai 2, `PTQ1_0` + projector). |
| **Hugging Face** | Nothing: both repos are public. |

## The pipelines

- **[`prepare-ternary-bonsai-1.vpipeline`](pipelines/prepare-ternary-bonsai-1.vpipeline)** /
  **[`prepare-ternary-bonsai-2.vpipeline`](pipelines/prepare-ternary-bonsai-2.vpipeline)**
  -- download the model and register it. Run once. Bonsai 2's fetches the
  `PTQ1_0` packing and the BF16 projector (`model_variant: PTQ1_0`; name
  `PQ2_0` there for the other packing).
- **[`ternary-bonsai-1-chat.vpipeline`](pipelines/ternary-bonsai-1-chat.vpipeline)** /
  **[`ternary-bonsai-2-chat.vpipeline`](pipelines/ternary-bonsai-2-chat.vpipeline)**
  -- the chat, text and images.

Run either with `vpipe --launch <file>` from your work directory, or open it
with **Load** in the web UI's Pipeline Manager. The steps are the same as in
the [Qwen3.5 9B chat guide](QWEN35-CHAT.md), which explains the work
directory, the chat graph and the web UI in detail.

```sh
cd ~/vpipe-work
~/src/vpipe/build/apps/vpipe/vpipe --launch prepare-ternary-bonsai-2.vpipeline
~/src/vpipe/build/apps/vpipe/vpipe --launch ternary-bonsai-2-chat.vpipeline
```

```
you> What is the capital of France? Reply with the city name only.
The user is asking for the capital of France and wants only the city name
as the reply. The capital of France is Paris. ...
Paris
```

A picture rides a turn as in every vpipe chat (`media: true` on
`text-input`; the web UI's attach and drag-and-drop controls build the
marker): asked the colour of a red disc on white, both models think briefly
and answer "Red".

## Settings worth knowing

- **`compute_dtype: f16`** (as the pipelines set it). The scales in both
  checkpoints are 16-bit floats, which f16 holds exactly; against the
  reference's fp32 forward, f16 lands 0.0006-0.0007 (rel-L2 of the logits)
  where the reference's own bf16 lands 0.0047-0.0055. bf16 works, at the
  reference's own bf16 error or up to twice it.
- **Sampling.** The shipped chats use the model cards' thinking-mode
  settings: Bonsai 1 temperature 0.7, top-p 0.95, top-k 20; Bonsai 2
  temperature 1.0, top-p 0.95, top-k 20, min-p 0.05.
- **`reasoning_effort`** (Bonsai 2 only; Qwen3.8's chat-template control).
  vpipe's default is the un-instructed `medium`; `xhigh` is the reference's
  default and thinks longer. The model card notes that `low` is not
  supported by the model and behaves close to `xhigh`.
- **`realtime-vqa`** takes either model as its `hf_dir` (set
  `compute_dtype: f16`): its question branches decode batched, and a
  batched branch is token for token the one decoded alone. Its
  `disable_thinking` (on by default) suits these reasoning models there:
  with thinking off both answer directly ("Paris" in two tokens) instead
  of spending a branch's token budget on reasoning.

## How well it runs

Against the reference forward in fp32 (PyTorch, the checkpoint dequantized
exactly), a 67-token chat turn, prompt prefilled and the reply teacher-forced
through the decode step:

| | vpipe f16 | vpipe bf16 | reference bf16 |
|---|---|---|---|
| Bonsai 1, logits rel-L2 | 0.0006 | 0.0052 | 0.0047 |
| Bonsai 2, logits rel-L2 | 0.0007 | 0.0114 | 0.0055 |
| greedy picks agreeing with fp32 | 47 / 47 | 46-47 / 47 | 46-47 / 47 |

The same holds on an M5 Pro and on an M5 (f16 0.0006 / 0.0008, bf16 0.0052 /
0.0074, 47 / 47 picks on both). The vision towers, against the reference's fp32 tower on a 448 x
640 photo (rel-L2 of the image rows; Bonsai 2's reference built from its BF16
projector):

| | M4 Pro | M5 Pro and M5 | reference bf16 |
|---|---|---|---|
| Bonsai 1 | 0.0048 | 0.0039 | 0.0211 |
| Bonsai 2, BF16 projector | 0.0021 | 0.0021 | 0.0139 |
| Bonsai 2, Q8_0 projector | 0.0059 | 0.0059 | 0.0139 |

Speed, f16:

| | M4 Pro (64 GB) | M5 Pro (24 GB) | M5 MacBook Air (16 GB) | Prism ML's llama.cpp build on the M4 Pro (their card) |
|---|---|---|---|---|
| decode | 24-25 tokens/s | 29.6-31.0 tokens/s | 15.9-16.7 tokens/s | 18.0 |
| 512-token prefill | 137-139 tokens/s | 412-429 tokens/s | 169-178 tokens/s | 125 |

The Air has no fan, so its figures are for a short run from cool; a long
one slows as it heats.

## How it runs, briefly

- A ternary weight is a 2-bit code (q - 1 times the scale), so both models
  run vpipe's 2-bit, group-128 kernels -- the same kernels for a decode step,
  a batched step and a prefill, on M4 and on M5's matrix cores alike.
  `PTQ1_0`'s densely packed trits are unpacked into that layout as they load,
  which is why it holds 7.7 GB from a 5.9 GB file.
- Bonsai 2 was trained with a **Hadamard rotation** in front of every ternary
  projection, recorded in the GGUF's `prism.hadamard` metadata: each such
  projection's input is rotated first (in fp32, one rounding), and the token
  table is rotated back as it is read. vpipe checks the file's description of
  the rotation and refuses a version it does not know, by name.
- The gated-DeltaNet layers normalize their queries and keys with the
  reference's epsilon (added to the sum of squares). That matters on these
  models: their first rows' keys can be tiny, and the alternative some
  runtimes use put Bonsai 1 twelve times further from the reference.

## Not done

- `PTQ1_0` is held unpacked (2.25 bits a weight); a kernel reading its trits
  directly would hold 1.75.
- No draft head is published for either model, so speculative decoding is
  off; a drafter attached to Bonsai 2 is refused.

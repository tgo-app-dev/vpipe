# Chat and live-video questions with LFM3-VL 2B on Apple Silicon

**LFM3-VL 2B** is Liquid AI's on-device vision-language model: a
1.73-billion-parameter **LFM3** language model and a 164M **vision tower**,
1.9B parameters in all. It reads pictures and video frames at their native
resolution, thinks inside a `<think>` block before it answers, and is small
enough to answer questions about a live video stream. vpipe runs it
on-device through its **metal-compute** backend: its own Metal kernels, no
Python and no third-party tensor runtime in the forward pass.

The checkpoint is `LiquidAI/LFM3-VL-2B` as Liquid AI publishes it, in
bf16. There is one download and no quantization step.

## What makes it different

The language model is not a stack of attention layers. Of its 24 layers,
13 are **gated short convolutions**: each token is mixed with the six
before it, then gated, so the layer costs the same at any context length.
Five layers use **sliding-window** attention over the last 256 tokens, and
only six attend to the whole context. That keeps the conversation's memory
small. The K/V cache is about **22 MB per 1024 tokens**, so the model's
full 128K-token context needs about 2.8 GB.

The vision tower works the same way: 8 of its 14 layers are 3×3
convolutions over the picture's patch grid, and 6 are attention. Every 2×2
block of 16-pixel patches becomes one language token, so a 512×512 picture
is 256 tokens and a 1-megapixel photo about 1000.

## What you need

| | |
|---|---|
| **Machine** | Apple Silicon Mac (M-series). |
| **Memory** | 8 GB or more. The weights take 3.8 GB; a conversation's K/V adds ~22 MB per 1024 tokens, and the working memory of a long paste ~0.34 GB. |
| **Disk** | **~3.9 GB.** |
| **Hugging Face** | Nothing. The repo is public and needs no account or token. |
| **License** | [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM3-VL-2B/blob/main/LICENSE). Commercial use is licensed only below USD 10M annual revenue (non-profits excepted for research). |

## The two pipelines

- **[`prepare-lfm3-vl-2b.vpipeline`](pipelines/prepare-lfm3-vl-2b.vpipeline)**
  — download the model and register it. Run once.
- **[`lfm3-vl-2b-chat.vpipeline`](pipelines/lfm3-vl-2b-chat.vpipeline)**
  — the chat, text and images.

Run either with `vpipe --launch <file>` from your work directory (the
directory whose `models/` and `data.mdb` the model registers in), or open it
with **Load** in the web UI's Pipeline Manager. The steps are the same as in
the [Qwen3.5 9B chat guide](QWEN35-CHAT.md), which explains the work
directory, the five-stage chat graph and the web UI in detail.

```sh
cd ~/vpipe-work
~/src/vpipe/build/apps/vpipe/vpipe --launch prepare-lfm3-vl-2b.vpipeline
~/src/vpipe/build/apps/vpipe/vpipe --launch lfm3-vl-2b-chat.vpipeline
```

The download fetches only what vpipe reads: the weights, the tokenizer,
the chat template and the processor settings. The repo's Python files are
code for `transformers` and are not fetched.

## Chatting about a picture

`media: true` on `text-input` lets a turn carry an attachment. In a terminal
you type the marker around a path; in the web UI the attach and
drag-and-drop controls build it for you:

```
you> Describe this image in one sentence. <|__vpipe_fs_im_start__|>/path/to/panda.png<|__vpipe_fs_im_end__|>
```

How a picture becomes tokens: its sides are rounded to multiples of 32
pixels and its area kept between 65,536 pixels and 16.7 megapixels. It is
resized (bicubic) only when it falls outside that, and each 32×32 block is
one token. Large photos are allowed but long, so crop or shrink a picture
you only need the gist of.

## Thinking

The model thinks before it answers, as its chat template has it by default:
the reply opens with its reasoning, then the answer. text-chat shows the two
apart. Set `disable_thinking: true` on `text-chat` for a direct answer,
which Liquid AI also benchmarks (its "no think" column). Reasoning from
earlier turns stays in the conversation's context as it was decoded.

## Live video: `realtime-vqa`

The same model serves **`realtime-vqa`**, which collects frames from a
stream into scenes and answers a list of questions per scene. Set its
`hf_dir` to `LiquidAI/LFM3-VL-2B`, and set `disable_thinking: true`: a
scene's answers should come back in seconds, not after a page of
reasoning. A frame gets about 1000 tokens unless you set
`vlm_max_soft_tokens` (256 is a good figure for a scene of several frames).
The questions are decoded together, the weights read once for all of them
per token. The scene's prefix (the frames) is shared between the question
branches rather than copied. The **`visual-qa`** stage takes the model
too.

## Settings that matter

- **Sampling.** The shipped pipeline uses Liquid AI's recommended settings:
  temperature 0.1, top-k 50, repetition penalty 1.1. Delete the `sampler`
  stage (and its wire) for plain greedy decoding.
- **The system prompt.** None by default, as the chat template has it. A
  host driving `text-chat` with a conversation (its `messages`) can open it
  with a system message; Liquid AI's examples use `You are a helpful
  assistant trained by Liquid AI.`
- **Context.** `page_tokens` × `max_pages` is how long a conversation can
  grow: 64K tokens as shipped, up to the model's own 128K. `/clear` resets
  it. A long paste is prefilled 2048 tokens at a time.
- **Faster prefill on an M5** (`i8_prefill: true` on `text-chat`,
  `visual-qa` or `realtime-vqa`): long prompts' matrix multiplies run in
  8-bit integers on the matrix cores, about 7% faster from 2048 tokens
  (prompts under 1024 tokens and the decode are untouched). It is lossy
  -- a prompt's continuation is no longer exactly the f16 one -- though
  measured as close to the float32 reference as a bfloat16 run is. Once a
  session, a long prompt spends ~0.2 s choosing its kernels. Off by
  default; ignored on GPUs without matrix cores.
- **Precision.** The model runs in f16 whatever `compute_dtype` says: its
  residual stream stays far inside f16's range, and f16 lands closer to the
  reference than bf16 does.
- **Tools** (`enable_tools`): the chat template's tool format (a `<tools>`
  list in the system message, Qwen3-Coder-style XML calls) is implemented.
  Liquid AI notes the model is not meant for long multi-turn or agentic
  use.

## What to expect

Measured on an **M4 Pro, 64 GB**, the weights on the internal SSD:

| | |
|---|---|
| prefill | ~2,200 tokens/s (512 to 8192 tokens) |
| decode | 53 tokens/s at a 2K-token context |
| 4 question branches decoded together | 27 ms a step, 148 tokens/s in all |
| vision tower | 66 ms for 512×512 (256 tokens), 290 ms for 1344×768 (1008 tokens) |

And on a base **M5, 16 GB** (a MacBook Air on power), whose GPU has matrix
cores:

| | |
|---|---|
| prefill | ~3,800–3,900 tokens/s at 512 to 2048 tokens, ~3,300 at 8192 |
| decode | 33 tokens/s at a 2K-token context |
| 4 question branches decoded together | 36 ms a step, 111 tokens/s in all |
| vision tower | 38 ms for 512×512 (256 tokens), 172 ms for 1344×768 (1008 tokens) |

The M5's matrix cores take the prefill and the vision tower (about 3x
faster than without them); decode reads every weight once a token, so it
follows the machine's memory bandwidth.

Accuracy, against a float32 run of the checkpoint's own model code (the
reference's bf16 run in brackets, which is the error bar a port is held
to): the language model's logits are 0.07% off (0.72%); the image turn's
0.10% (0.82%); the vision tower 0.45% (2.1%). The answers are the same token
for token: "The capital of France is Paris." and "A watercolor painting
depicts a panda holding and eating bamboo."

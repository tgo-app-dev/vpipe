# Chat with Qwen3.5 9B on Apple Silicon

**Qwen3.5-9B** is a general-purpose language model — and, in this checkpoint,
one that reads pictures too. vpipe runs it on-device through its
**metal-compute** backend: its own Metal kernels, no Python and no
third-party tensor runtime in the forward pass.

This is the **smallest complete example** in these docs, and the one to start
with. There is a single download, **no quantization step**, and a five-stage
pipeline that puts a chat prompt in your terminal or in the browser. Under
8 GB on disk, comfortable on a 16 GB Mac.

The variant is **OptiQ 4-bit**, which arrives already quantized: a group-64
affine base at 4 bits with individual tensors promoted to 8. The choice is
made per *tensor*, not per layer type — in this checkpoint 134 of the 250
quantized linears sit at 8-bit and 116 at 4-bit, and `mlp.gate_proj` is 4-bit
in 30 layers and 8-bit in 2. The embeddings and the output head are 8-bit
throughout. vpipe reads each tensor's width out of the checkpoint, so none of
that is yours to configure.

## What you need

| | |
|---|---|
| **Machine** | Apple Silicon Mac (M-series). |
| **Memory** | 16 GB. |
| **Disk** | **~7.7 GB**, and that is the whole story — nothing is quantized locally, so there is never a second copy. |
| **Build** | An Apple Silicon build of vpipe — the default on arm64 macOS. See the main [README](../README.md). |
| **Hugging Face** | Nothing. The repo is public and needs no account or token. |

## The two pipelines

- **[`prepare-qwen35-9b-optiq-4bit.vpipeline`](pipelines/prepare-qwen35-9b-optiq-4bit.vpipeline)**
  — download the model and register it. Run once.
- **[`qwen35-9b-chat.vpipeline`](pipelines/qwen35-9b-chat.vpipeline)**
  — the chat itself.

Follow a link and use **Raw ▸ Save as** to download it, or take them straight
from `docs/pipelines/` in your clone. Either can be run from the terminal with
`vpipe --launch <file>` or opened with **Load** in the web UI's Pipeline
Manager (or the phone UI's ⋯ menu). Both are plain JSON — read them, edit
them, keep them in version control.

## Step 1 — get the model

### First, choose a work directory

vpipe treats **the directory you launch it from** as its workspace, and
creates its state there:

| | |
|---|---|
| `models/` | every model you download or quantize |
| `data.mdb`, `lock.mdb` | the LMDB database — the model registry, logs, stage output |
| `sandbox/` | created by **`vpipe-web-ui`** only: the directory it confines stage file I/O to |

Use the **same** directory in step 2. The model you are about to download is
recorded in that directory's registry, so a chat started somewhere else will
not find it.

### Then run the pipeline

```sh
cd ~/vpipe-work                                    # your work directory
cp ~/src/vpipe/docs/pipelines/prepare-qwen35-9b-optiq-4bit.vpipeline .
~/src/vpipe/build/apps/vpipe/vpipe --launch prepare-qwen35-9b-optiq-4bit.vpipeline
```

**One stage — `model-fetch` — and no quantize step.** That is the point of
picking a pre-quantized checkpoint for a first example: the heavier docs here
([MiniMax H3](MINIMAX-H3.md), [klein-9b-kv](KLEIN-KV.md)) spend their step 1
converting tens of gigabytes, and this one just downloads.

`skip_existing_files` is on, so `Ctrl-C` and re-run picks up where it left
off rather than starting the download again.

What lands under `models/` is ~7.7 GB: two safetensors shards, the tokenizer,
and an `optiq/` subdirectory that the recursive repo listing brings along —
`optiq_vision.safetensors` (the vision tower) and `mtp.safetensors` (a
speculative-decode draft head). Both are picked up automatically when they
are present; neither needs wiring.

The model registers under its repo path, **`mlx-community/Qwen3.5-9B-OptiQ-4bit`**,
and that string is exactly what the chat pipeline's `hf_dir` names.

## Step 2 — chat

```sh
cd ~/vpipe-work                                    # the SAME work directory
cp ~/src/vpipe/docs/pipelines/qwen35-9b-chat.vpipeline .
~/src/vpipe/build/apps/vpipe/vpipe --launch qwen35-9b-chat.vpipeline
```

```
[INFO] TextChatStage('chat'): model ready (32 layers, vocab=248320, thinking=off)
you> In one sentence, what is the Pacific Ocean?
The Pacific Ocean is the largest and deepest ocean on Earth, covering more
than 30% of the planet's surface area and separating the continents of Asia
and Australia to the east from North and South America to the west.
[INFO] TextChatStage('chat'): prefill 22 tok in 0.279 s = 78.9 tok/s, decode 45 tok in 2.308 s = 19.5 tok/s, ctx_pos 68
you>
```

Type at the `you>` prompt. `/clear` as a message resets the conversation —
the K/V context, not the process. `Ctrl-D` ends the run.

The graph is five stages:

```
text-input ──user──> text-chat ──assistant──> feedback-rx
    ▲                    ▲                         ╎
    │                    │                         ╎ paired by NAME
    │  sampler-select ───┘ (sampler)               ╎ ("from": "rx")
    └──trigger── feedback-tx ◀╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┘
```

**`feedback-tx` and `feedback-rx` are one pair, linked by name** — the tx
stage's `"from": "rx"` — and *not* by a wire. That is how a loop is expressed
in a graph the runtime otherwise requires to be acyclic. Each round runs:
you type a line → `text-chat` replies → the reply beat reaches `rx` → `tx`
re-emits it into `text-input`'s trigger → you are prompted again. The first
prompt does not wait for a beat (`present_first_without_beat`), so the loop
starts itself instead of deadlocking on its own output.

### The sampler is its own stage

Sampling is **not** a `text-chat` config key. A `sampler-select` stage emits
a sampler spec as a beat, and `text-chat` latches it on the first turn and
reuses it for every later one. Leave its `sampler` port unwired and the stage
decodes **greedily** — argmax, reproducible, and noticeably flat.

The shipped values are not invented here: they are what this checkpoint ships
in its own `generation_config.json`.

| knob | shipped | |
|---|---|---|
| `temperature` | 0.7 | |
| `top_k` | 20 | |
| `top_p` | 0.8 | |
| `min_p` | 0.0 | disabled |
| `repetition_penalty` | 1.0 | disabled |
| `presence_penalty` | 1.5 | |
| `seed` | 0 | a fresh seed every run; set a non-zero value to make a conversation reproducible |

The stage logs what it latched and `text-chat` logs what it received, so a
sampler that never arrived is visible in the first few lines of a run rather
than inferred later from dull replies:

```
[INFO] SamplerSelectStage('sampler'): temperature 0.7, top_k 20, top_p 0.8, min_p 0, rep 1, presence 1.5, seed 0
[INFO] TextChatStage('chat'): sampler = sampled (temperature 0.7, top_k 20, ...)
```

One stage covers every generative consumer: the same source feeds
`visual-qa`, `realtime-vqa`, `audio-transcribe` and `text-to-speech`.
(`diffusion-sampler-select` is an unrelated stage — it picks a *diffusion*
integrator for `generate-image`.)

### Thinking and tools are off, deliberately

Both are one key each, and both are off in the shipped pipeline so that the
example is a plain chat:

- **`disable_thinking: true`.** Qwen3.5's chat template opens a `<think>`
  block by default, and the model reasons in it before answering. The flag
  makes the template emit an empty one, so the reply starts at the answer.
  Set it to `false` for anything where you want the reasoning — it costs
  tokens and latency, which is why a demo does not want it.
- **`enable_tools: false`.** `text-chat` can run tools and feed the results
  back for another decode round. They are individually opt-in
  (`enable_python_tool`, `enable_shell_tool`, `enable_file_tools`,
  `enable_web_tools`), and the sandboxed ones **execute code the model
  wrote**. Enable them one at a time, knowing why; an introductory example
  leaves the loop off entirely.

### The settings worth knowing

From the `text-chat` stage:

| key | shipped | notes |
|---|---|---|
| `hf_dir` | `mlx-community/Qwen3.5-9B-OptiQ-4bit` | a registry key (what step 1 created) or a model directory. A key wins over a same-named path. |
| `compute_dtype` | `bf16` | the activation dtype. |
| `page_tokens` / `max_pages` | 1024 / 64 | the K/V pool: a **ceiling** of 65,536 tokens of context, not an allocation. It grows as a conversation does. |
| `max_new_tokens` | 4096 | per-turn generation budget. |
| `mtp` | `true` | use the MTP draft head from `optiq/` — speculative decode, token-exact, purely a speed feature. Ignored by checkpoints that ship no head. |

And `media: true` on `text-input`, which is what lets a turn carry an
attachment. This checkpoint has a vision tower, so a picture in the turn is
encoded by the model's own tower and spliced into the prefill:

```
you> Describe this picture in one sentence. <|__vpipe_fs_im_start__|>/path/to/shot.png<|__vpipe_fs_im_end__|>
This screenshot displays the "Settings" interface of a software application,
likely related to AI model management or pipeline execution, ...
[INFO] TextChatStage('chat'): prefill 735 tok in 1.168 s = 629.1 tok/s, decode 57 tok in 2.942 s = 19.4 tok/s
```

That marker syntax is what a **terminal** user types. In the web UI the input
row has attach and drag-and-drop controls that build the same thing for you,
so you pick a file instead of spelling a path.

## Running it in the web UI

The same file, unchanged:

```sh
cd ~/vpipe-work                                    # the SAME work directory
~/src/vpipe/build/apps/web-ui/vpipe-web-ui
```

Open the URL it prints, **Load** `qwen35-9b-chat.vpipeline`, press Start, and
open the **User I/O** panel — the `you>` prompt appears there, and replies
stream into it as they are decoded. Everything above applies identically; only
the console changes.

## What to expect

Measured on an **M5, 16 GB**, at the shipped settings:

| | |
|---|---|
| decode | **~19–20 tok/s** |
| prefill | 735 tokens — a screenshot plus a question — in **1.2 s** |

Decode is memory-bandwidth-bound at this size, so it is roughly flat with
context and scales with the machine rather than with the settings.

## Memory

The weights are ~7–8 GB and stay resident: unlike the video and image models
in these docs, a 9B chat model at this precision fits a 16 GB Mac with room to
work in, and nothing streams.

What grows during a run is the **K/V context** — the shipped page settings
cap it at 65,536 tokens, and it is allocated as the conversation reaches it,
not up front. `/clear` gives it back.

**One heavy job at a time.** Metal buffers are wired and cannot be paged out,
so a second large model running alongside this one does not slow the machine
down gracefully — it exhausts it.

## Troubleshooting

**The model isn't found.** The registry lives in the work directory. Launching
the chat from somewhere other than where step 1 ran is the usual cause — `cd`
there, or give `hf_dir` an absolute path to the model directory instead.

**Every run says the same thing.** The `sampler` port is not wired, so the
stage is decoding greedily. Check the two log lines above. With the sampler
in place, `seed: 0` draws a fresh seed each run, so the same question gets a
different answer.

**The reply begins with reasoning.** `disable_thinking` is `false`.

**It answers a question from ten minutes ago.** The context is persistent by
design. Send `/clear`.

## Under the hood

- Qwen3.5 is a **hybrid**. Of the 9B's 32 layers, **8 are full attention and
  24 are gated DeltaNet** — a linear-attention layer that carries a fixed
  recurrent state instead of a growing K/V, which is why context costs less
  here than in an all-attention model of the same size.
- **OptiQ** is per-tensor mixed precision — 4- and 8-bit affine over a shared
  group-64 base — and the loader detects each tensor's width rather than
  being told.
- The **MTP head** in `optiq/` drafts tokens the main model then verifies, so
  a run producing identical output does less work per token.
- On **M5**, the GEMMs and attention run on the GPU's matrix cores
  (`matmul2d` / NAX flash attention).

# Chat with Ministral 3 14B (text and images) on Apple Silicon

**Ministral 3 14B Instruct** is Mistral AI's largest Ministral 3 model: a
dense 14-billion-parameter language model with a 0.4B **vision encoder**, so
it reads pictures as well as text. It is Apache-2.0. vpipe runs both halves
on-device through its **metal-compute** backend — its own Metal kernels, no
Python and no third-party tensor runtime in the forward pass.

The checkpoint is `mlx-community/Ministral-3-14B-Instruct-2512-4bit`: the
language model at 4-bit (affine, group 64), the vision tower in bf16. It
arrives already quantized, so there is a single download and no
quantization step.

## What you need

| | |
|---|---|
| **Machine** | Apple Silicon Mac (M-series). |
| **Memory** | 24 GB or more. The weights take ~8.2 GB, and the conversation's K/V cache 160 MB per 1024 tokens on top -- ~5.4 GB at the shipped pipeline's 32k-token context. A 16 GB MacBook Air M5 ran 2k-token prompts but swapped heavily from 4k. |
| **Disk** | **~8.4 GB.** |
| **Hugging Face** | Nothing. The repo is public and needs no account or token. |

## The two pipelines

- **[`prepare-ministral3-14b.vpipeline`](pipelines/prepare-ministral3-14b.vpipeline)**
  — download the model and register it. Run once.
- **[`ministral3-14b-chat.vpipeline`](pipelines/ministral3-14b-chat.vpipeline)**
  — the chat, text and images.

Run either with `vpipe --launch <file>` from your work directory (the
directory whose `models/` and `data.mdb` the model registers in), or open it
with **Load** in the web UI's Pipeline Manager. The steps are the same as in
the [Qwen3.5 9B chat guide](QWEN35-CHAT.md), which explains the work
directory, the five-stage chat graph and the web UI in detail.

```sh
cd ~/vpipe-work
~/src/vpipe/build/apps/vpipe/vpipe --launch prepare-ministral3-14b.vpipeline
~/src/vpipe/build/apps/vpipe/vpipe --launch ministral3-14b-chat.vpipeline
```

## Chatting about a picture

`media: true` on `text-input` lets a turn carry an attachment. In a terminal
you type the marker around a path; in the web UI the attach and
drag-and-drop controls build it for you:

```
you> What animal is in this picture, and what is it doing? <|__vpipe_fs_im_start__|>/path/to/panda.png<|__vpipe_fs_im_end__|>
The animal in the picture is a **giant panda**. It appears to be eating or
chewing on **bamboo**, which is a primary part of its diet. ...
[INFO] TextChatStage('chat'): prefill 1167 tok in 4.329 s = 269.6 tok/s, decode 52 tok in 1.731 s = 30.0 tok/s, ctx_pos 1220
you> What colour is the background?
The background of the image is not fully visible, but the visible portion
appears to be a **light, off-white or beige color**. ...
[INFO] TextChatStage('chat'): prefill 8 tok in 0.180 s = 44.5 tok/s, decode 44 tok in 1.460 s = 30.1 tok/s, ctx_pos 1273
```

The follow-up is answered from the same context: its prefill is the eight
tokens of the new question, not the picture again.

How a picture becomes tokens. The image is resized the way Mistral's own
processor does it: shrunk (never enlarged) so its longest side fits 1540
pixels, then each side rounded up to a multiple of 28. Every 28×28 square
becomes one token, and each row of the grid ends with a row-break token.
So the 812×560 picture above is 29×20 = 580 image tokens plus 20 breaks.
A large photo approaches 55×55 ≈ 3000 tokens, and Mistral recommends an
aspect ratio near 1:1 — crop very wide or very tall pictures.

The same tower serves the **`visual-qa`** stage (images on an iport, a list
of questions) and **`realtime-vqa`** (frames from a stream). In
`realtime-vqa` a frame is capped at `vlm_max_soft_tokens` tokens, 256 when
unset, so a scene of frames stays inside the context.

## Settings that matter

- **Sampling.** Mistral recommends a temperature below 0.1 for everyday use;
  the shipped pipeline samples at 0.05. Delete the `sampler` stage (and its
  wire) for plain greedy decoding.
- **The system prompt.** With no system turn of your own, the model gets the
  one its chat template declares — Mistral's "Le Chat" assistant prompt, with
  today's and yesterday's dates filled in, about 540 tokens prefilled once
  at the start of a conversation.
- **Context.** `page_tokens` × `max_pages` is how long a conversation can
  grow: 32768 tokens as shipped, up to the model's own 262144. Each 1024
  tokens holds 160 MB of K/V once it is used (nothing is reserved up
  front), so 64k takes ~10 GB and the full 256k ~40 GB. `/clear` resets
  it. Past 16k tokens the model sharpens its attention with position (a
  Llama-4-style query temperature); vpipe applies it as the reference does.
  A long paste is prefilled 8192 tokens at a time, so its working memory
  is bounded by that piece rather than growing with the paste.
- **On an M5.** The GPU's matrix units take the prefill's matrix
  multiplies and the vision tower and, from 1536 tokens on, the prefill's
  attention as well -- about 3x the kernel an M4 runs there. Setting
  `i8_prefill: true` on `text-chat` (or `visual-qa`, `realtime-vqa`) also
  runs the prefill's matrix multiplies in int8, which is faster but not
  exact: on a 2103-token prompt the logits moved 1.3% from the plain run
  and the next token was the same. Off by default; an M4 ignores it.
- **Tools** (`enable_tools`) are not available for this family yet: Mistral
  calls tools in its own `[TOOL_CALLS]` format, which `text-chat` does not
  parse.

## What to expect

Measured on an **M4 Pro, 64 GB**, the weights on an external Thunderbolt
drive, the conversation above:

| | vpipe | mlx-vlm 0.7.6 (same weights) |
|---|---|---|
| first turn (1149–1167 tokens, image included) | 4.7 s | 4.7 s |
| decode | 30.0 tok/s | 31.1 tok/s |
| a 33,820-token document: prefill | 197 s (172 tok/s) | 198 s (171 tok/s) |
| decode at that depth | 15.3 tok/s | 15.3 tok/s |

Correctness is checked against mlx-vlm on the same 4-bit weights
(`tests/unit-tests/metal-lm/ministral3.cc`, goldens from
`tools/dump_ministral3_golden.py`): the tokenizer id-for-id, the chat
template id-for-id, the vision tower and projector closer to the reference's
fp32 run than its own bf16 run is, and every decisive greedy pick of a text
and an image conversation the same -- and of a 33,820-token one, past both
of the attention temperature's steps (16k and 32k), which finds a code
planted near its start.

# Chat with DeepSeek-V4.1-Flash (text and images) on Apple Silicon

**DeepSeek-V4.1-Flash** is a sparse mixture-of-experts vision-language
model: 552 billion backbone parameters, of which a token reads about 16
billion -- six of 384 experts a layer, a shared expert, a sliding window
of attention and a compressed view of everything before it -- plus a
196-billion-parameter n-gram memory read a few rows at a time. vpipe runs
it on-device through its **metal-compute** backend: its own Metal kernels,
no Python and no third-party tensor runtime in the forward pass.

The checkpoint the pipelines name is
[`mlx-community/DeepSeek-V4.1-Flash-1017-oQ4e-mtp`](https://huggingface.co/mlx-community/DeepSeek-V4.1-Flash-1017-oQ4e-mtp),
oMLX's oQ4e pack of
[`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017),
DeepSeek's minor update of the model (new weights; the same architecture,
code and tokenizer). The pack keeps that release's own numbers -- the
routed experts in FP4 and the attention, gating and shared expert in FP8,
the same codes and scales -- and quantizes only the n-gram tables, to 4
bits. It is **432 GB**. MIT licence.

DeepSeek's releases themselves -- the 1017 update and the first
[`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
(510 GB each, the n-gram tables in FP8) -- load as well: put the name
where the pipelines name the oQ4e pack (`model_path` in the prepare
pipeline, `hf_dir` in the chat pipeline).

### Smaller packs: 2 and 3 bits

oMLX also publishes the 1017 update at fewer bits. These packs requantize
the routed experts to 2 or 3 bits, choosing the width layer by layer from
measured sensitivity, and the attention and shared-expert projections to
4-8 bits. The draft head (DSpark) and the n-gram tables stay as in oQ4e.
vpipe runs each as stored, one width per module. Put the name where the
pipelines name the oQ4e pack:

| pack | routed experts | in memory, every expert | download |
|---|---|---|---|
| [`…-1017-oQ4e-mtp`](https://huggingface.co/mlx-community/DeepSeek-V4.1-Flash-1017-oQ4e-mtp) | the release's FP4 | 289 GB | 432 GB |
| [`…-1017-oQ3e-mtp`](https://huggingface.co/mlx-community/DeepSeek-V4.1-Flash-1017-oQ3e-mtp) | 3 bits | 238 GB | 380 GB |
| [`…-1017-oQ2.5e-mtp`](https://huggingface.co/mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp) | 2 bits, 3 in six layers, w2 at 3 | 199 GB (238 as vpipe caches them) | 343 GB |
| [`…-1017-oQ2e-mtp`](https://huggingface.co/mlx-community/DeepSeek-V4.1-Flash-1017-oQ2e-mtp) | 2 bits | 170 GB | 313 GB |

Each pack has an `-ngram-mtp` twin that keeps the n-gram tables in the
release's FP8: 80 GB more to download. Those tables are read from the drive
a row at a time either way, so the twin holds no more memory; vpipe reads
both. Fewer bits cost accuracy: oMLX quantizes these packs with
calibration, but a 2-bit expert is a coarse copy of the release's.

On a 256 GB machine every 2-bit expert fits in memory. Nothing is read
from the drive after the first tokens, and DSpark loads (see below).
Measured on an M5 Ultra with 256 GB, oQ2e answering short math questions
(ten from GSM8K): about 55 tokens a second, and 88 with DSpark; a
17,000-token prompt prefills in 6-9 s, and a 124,000-token one in ~35 s
(the spread is the drive: each prompt reads its n-gram rows off it).

vpipe gives every expert a cache slot of one size, as wide as the widest
layer's. oQ2.5e's six 3-bit layers make every slot a 3-bit one, so in
memory that pack takes as much room as oQ3e (238 GB, not 199).

## What you need

| | |
|---|---|
| **Machine** | Apple Silicon Mac (M-series). |
| **Memory** | 64 GB or more; 128 GB or more for a fluent chat. The model never holds all of itself: the two n-gram tables (61 GB each in the oQ4e pack) are read a row at a time from disk, and the routed experts (289 GB) live in a cache sized to the box -- the most-selected kept, the rest read from disk when a token picks them. Everything else is ~11 GB once loaded (the FP8 weights kept as they are stored), plus ~6 GB for a 64k-token conversation and the working memory of a prompt read 8,192 tokens at a time. |
| **Disk** | **~432 GB**. The faster the drive, the faster the model: a token that misses the expert cache waits for its read. |
| **Hugging Face** | Nothing. The repo is public and needs no account or token. |

## The pipelines

- **[`prepare-deepseek-v41-flash.vpipeline`](pipelines/prepare-deepseek-v41-flash.vpipeline)**
  -- download the model and register it. Run once.
- **[`deepseek-v41-flash-chat.vpipeline`](pipelines/deepseek-v41-flash-chat.vpipeline)**
  -- the chat, text and images.

Run either with `vpipe --launch <file>` from your work directory, or open it
with **Load** in the web UI's Pipeline Manager. The steps are the same as in
the [Qwen3.5 9B chat guide](QWEN35-CHAT.md), which explains the work
directory, the five-stage chat graph and the web UI in detail.

```sh
cd ~/vpipe-work
~/src/vpipe/build/apps/vpipe/vpipe --launch prepare-deepseek-v41-flash.vpipeline
~/src/vpipe/build/apps/vpipe/vpipe --launch deepseek-v41-flash-chat.vpipeline
```

## Chatting about a picture

`media: true` on `text-input` lets a turn carry an attachment. In a terminal
you type the marker around a path; in the web UI the attach and
drag-and-drop controls build it for you:

```
you> <|__vpipe_fs_im_start__|>/path/to/photo.png<|__vpipe_fs_im_end__|> What is in this picture?
```

A picture is padded onto a grid of 14-pixel patches and every 3x3 of them
becomes one token, a row break after each row of them: at least 544x544
pixels' worth (a smaller picture is enlarged) and at most 1024 tokens (a
larger one is shrunk). So a small picture is 13 x 13 tokens (184 with the
row breaks), a 1024x1024 one 25 x 25 (652), and any larger square one
31 x 31 (994); a 4:1 panorama tops out at 60 x 15.

## Settings that matter

- **Thinking.** The pipeline chats in DeepSeek's *chat* mode (the model
  answers directly; `disable_thinking: true`). Set `disable_thinking:
  false` for *thinking* mode: the model reasons first, and the chat shows
  the answer. The model takes a numeric reasoning effort from 1 to 100;
  text-chat's `reasoning_effort` names map onto it: `low` 50, `medium` 75
  (the model's default), `xhigh` 100.
- **Sampling.** The shipped pipeline uses DeepSeek's recommended settings,
  temperature 1.0 and top-p 0.95.
- **Context.** `page_tokens` x `max_pages` is how long a conversation can
  grow. Every query attends the last 128 positions of its layer, and --
  in all but the first two layers -- the best 512 of the compressed
  positions before them, chosen by a small indexer; the deeper layers
  choose only among the blocks the first deep layer kept. vpipe does all
  of this as the model was trained to.
- **Precision.** `compute_dtype: bf16` is the model's own and the only one
  the chat runs.

## Speculative decoding (DSpark)

The checkpoint carries DSpark, DeepSeek's own draft head: three small
layers that read the model's last layers and propose the next five
tokens at once, which the model then checks in one pass. vpipe runs it
with `mtp: true` (text-chat's default), and the reply is exactly the one
plain decoding gives -- greedy or sampled, token for token.

Two things decide whether it is used, and vpipe decides both itself:

- **Whether to load it.** Only on a machine that holds every expert in
  memory. Where the experts are cached, each checked draft picks experts
  of its own, and the ones still missing must be read from the drive
  before the check can finish (~11-13 ms a draft on an M5 Ultra, against a
  ~30 ms token), and the draft itself costs ~5 ms; on the 64-256 GB
  machines measured that is more than the ~0.6 extra tokens a round it
  keeps on prose, so there vpipe decodes token by token. How much a round
  keeps depends on the text, though. On grade-school maths problems (ten
  GSM8K questions, one after another) it keeps 3.1-3.3 tokens a round,
  and DSpark decodes 52 tokens a second against 39 on the 256 GB machine
  (128 GB: 29 against 27); on a short story at 256 GB it is 24 against
  25. `VPIPE_DSV41_DSPARK=1` loads it whatever the machine.
- **How many drafts to check.** Every checked token picks its own
  experts, and drafts further along the block are kept less often. Each
  round vpipe checks as many drafts as pay, from what it measures as it
  runs -- what a check of each length costs, and how often a draft with
  the head's own confidence score is kept -- and when drafting stops
  paying it decodes token by token.

Two alternatives can be switched on for comparison, both keeping the same
reply: drafts **chained** one at a time, as a classic MTP head drafts --
each pass of the three layers reads the drafts so far instead of DSpark's
placeholder tokens (`VPIPE_DSV41_MTP_DRAFT=chain`) -- and a draft count
chosen by **acceptance** alone, one more than the last round kept
(`VPIPE_DSV41_MTP_DEPTH=acceptance`). On the ten GSM8K questions with oQ2e
in memory they measured 70 and 85 tokens a second against the default's
88: chaining costs a pass a draft, and the head keeps fewer drafts that
way than with the placeholders it was trained on.

## What to expect

Measured on an M5 Ultra with 256 GB -- the first two rows with vpipe told
the machine has 64 and 128 GB, so it sizes everything as it would there
-- the oQ4e pack on its internal SSD, a 512-token greedy reply to a short
request, from a cold start (the first tokens read most of their experts;
the rate climbs as the cache learns the conversation's):

| memory | expert cache | decode, 512 tokens | once warm |
|---|---|---|---|
| 64 GB | 44 GB | 14.8 tokens a second | 15 (91% of picks from memory) |
| 128 GB | 109 GB | 24.3 | 31 (98%) |
| 256 GB | 180 GB | 24.8 | 35 (99%) |

Loading takes ~3 s, and a short prompt's first, cold prefill ~5 s. Once
the cache has warmed a token takes 28-33 ms at 128 GB and up: about 20 of
them the GPU's work, the rest the reads of the few experts still missed
(each ~19 MB, its six pieces read at once). While a token's layers run,
vpipe reads ahead the experts the next layer will most likely pick -- the
next layer's router, run on this layer's input -- as many as it measures
pay; most of that is won while the cache warms. The next token's experts
cannot be guessed as cheaply: the ones this token picked are already in
memory, and the ones the next token will miss are not among them.

Long conversations stay quick. With 124K tokens of context on the
256 GB machine (oQ4e, its experts cached), the prompt prefills at ~2,150
tokens a second (under a minute). On an M5 the prefill's attention and
the indexer that picks each layer's 512 compressed positions run on the
GPU's matrix units -- a query's 64 heads, which share their keys, as the
rows of one product -- and the deeper layers score only the blocks the
first deep layer kept.

A prompt is read the way the model was built to read it. DeepSeek-V4.1-
Flash is a causal encoder-decoder: its last 20 layers take their long-
range memory from the first 20 layers' output, not from their own. So
vpipe runs the first 20 layers over every token of a long prompt and the
last 20 only over its final 136 tokens -- the 128-token window the reply
goes on to read (the model card's "SWA bounded replay"), and 8 more, so
that taking back the prompt's last few tokens still finds a whole window.
That roughly halves a long prompt's work; the first half's output and the
memory the reply reads are what running every layer would give, and in
our checks the replies are the same. `VPIPE_DSV41_CED=0` runs every layer
over every token. The prompt is read in pieces of 8,192 tokens at any
length (the indexer's scores made a part of a piece at a time), with each
next piece's n-gram rows read from the drive while the GPU works on the
one before -- each distinct row once, after the piece's own rows are in.

Most of a token's GPU time on this machine is not reading weights but
waiting for small steps to finish, so the decode is built to keep the GPU
busy: the attention splits each head's keys across the GPU, the
projections spread a row across eight lanes, independent steps run side
by side, and on an M5 Ultra a layer hands its expert picks to the CPU
inside the running GPU work (~3 microseconds) instead of stopping the GPU
for each of the 40 layers (~0.3 ms each). vpipe checks at load whether the
GPU can do that and falls back to stopping where it cannot.

On an M4 Pro with 64 GB and the oQ4e pack on an external Thunderbolt
SSD (about 0.8 GB/s):

| | |
|---|---|
| Load | ~15 s. The model keeps ~11 GB, plus a 31 GB expert cache. |
| A first, cold prompt | 13 tokens in 47 s; a picture turn (210 tokens) in ~2 minutes. Each layer reads most of its experts the first time. |
| Decode | ~0.6 tokens a second once the cache has learned the conversation's experts (80% of picks served from memory). |

On this machine the drive is the whole cost: a token whose experts are
not in memory waits for ~0.9 GB of reads, and the GPU's own work is under
a fifth of a second a token. A faster drive (an internal SSD reads ~5x
faster) or more memory (a bigger cache) is what speeds the model up; a
machine that holds every expert runs it from memory.

## Tools in a conversation

A host that hands text-chat a whole conversation with its tools (an
agent's turns) has them rendered in DSML, the release's own format, as its
`encoding/encoding.py` renders them: the system message gains a "## Tools"
section -- how to call, then each function's schema on a line -- and the
model calls a tool by writing

```
<｜DSML｜ calls>
<｜DSML｜ invoke name="get_weather">
<｜DSML｜ parameter name="location" string="true">Beijing</｜DSML｜ parameter>
<｜DSML｜ parameter name="days" string="false">3</｜DSML｜ parameter>
</｜DSML｜ invoke>
</｜DSML｜ calls>
```

-- a string as it is, anything else as JSON. The turn's `tool_calls` are
read back from it, and the results go to the model as a user turn of
`<tool_result>...</tool_result>` blocks. In thinking mode an earlier turn
keeps its reasoning when the conversation has tools, as `encoding.py`
renders it then -- send each assistant message back as the turn's `text`
(its reasoning, `</think>`, then the answer and its calls); a turn sent
as its answer alone reads as one with no reasoning. Without tools only
the turns after the last user message keep theirs. One difference from
the reference: a user message sent after a turn's results opens a user
turn of its own, where `encoding.py` joins the two.

## Holding less than the box

`memory_limit_mb` on text-chat is the most the model may hold, in
megabytes: its weights as held, the expert cache and a conversation's
state. The cache is sized within it -- what the limit leaves after the
rest of the model -- and the rest of the box stays free for other models:
the graph's memory plan books the model at that size, so the picture and
video models beside it size themselves against the room it leaves. DSpark
loads only where every expert is resident within it. A model already
loaded (kept warm by another stage) is handed over at the size it loaded
at. 0, the default, sizes the cache to the box, as above.

## What is not here yet

- **The stage's own tools** (`enable_tools`: Python, files, the web) are
  not offered with this model; a conversation's own tools are (above).
- The deployment's own **FP8 / FP4 activations and KV cache**: vpipe keeps
  activations and the cache in bf16, which is closer to the model's
  full-precision arithmetic, at about 3x the cache's memory per token.

## How it is checked

Against DeepSeek's own reference implementation, the release's
`inference/model.py` and `engram.py` imported as they are, with their CUDA
kernels restated in plain PyTorch so they run on a CPU
(`tools/dump_deepseek_v41_golden.py`), in full precision -- the truth --
and in bf16, whose distance from it is the bar vpipe's bf16 is held to:

- the tokenizer, id for id with Hugging Face's on fifty awkward strings
  (scripts, numbers, punctuation runs, whitespace, emoji);
- the n-gram hashing exactly: all 129,280 entries of the normalized token
  map, the hash multipliers, the primes (whose sums are the checkpoint's
  table sizes) and every hashed row;
- a synthetic model in the release's exact layout, small enough to golden
  in seconds yet with every mechanism of the real one -- the window, both
  compression ratios, kv / index / candidate sources, a reindexing layer,
  both n-gram layers, the clamped experts, the hyper-connections, the
  vision tower -- through the prefill, decode steps and an image turn;
- the oQ4e pack (and before it the first release) on a short
  conversation and its teacher-forced decode steps, held two ways. Freely,
  every decisive next token is the reference's -- but a bf16 run flips
  some of the expert choices full precision makes by a hair (the
  reference's own bf16 flips 4 to 12 of 40 layers a token), and a flip in
  the last layers lands on the output whole, so a few steps read far from
  full precision in any bf16 run, the reference's included. Given full
  precision's expert choices, which removes that draw, vpipe's bf16 is
  closer to full precision than the reference's own bf16 (prefill 0.0099
  against 0.0107, decode 0.0058 against 0.0061);
- the DSpark head's drafts, given full precision's expert choices, closer
  to full precision than the reference's own bf16 (0.0030 against
  0.0083, every draft token full precision's), and a speculative decode
  the plain one token for token -- the model's own drafts and planted
  ones, greedy and sampled, however many drafts a round checks;
- the same checks on the M5's matrix cores, at 128 GB;
- the packs below 4 bits, two ways. First, the synthetic model in their
  layout, every check above: experts at 2 bits and at 3 by layer, w2 at 3
  (oQ2.5e's mix), projections at 4, 5, 6 and 8 bits, a requantized `wo_a`,
  the n-gram tables either way. In f16 vpipe sits a fifth of the
  reference's own bf16 distance from full precision (0.004-0.012 against
  0.049-0.052). Given full precision's expert choices, its bf16 is the
  reference's (0.035-0.040 against 0.035-0.038). A verify's rows equal
  its decode's to the bit. Second, the bit layout itself, checked against
  MLX's own `quantize` / `dequantize` at every width and group size;
- the decode's own rearrangements against the plain encoding, to the bit:
  steps run side by side, a sublayer's last step merged into the next's
  first, and the in-flight hand-over of the expert picks -- including a
  hand-over abandoned on purpose, which redoes the token the plain way and
  changes nothing else;
- the indexer's faster selections against scoring every position, to the
  pick: random scores, ties at the cut, masked and too-few positions, and
  through the model, to the bit;
- a picture turn and a question through the chat stage on the pack (a red
  picture: "Red"; the capital of France: "Paris").

# Faster Qwen3.8-27B chat with DFlash 2 speculative decoding

A 27B model on a Mac decodes as fast as it can read its weights: every
token is one full pass over ~15 GB. **Speculative decoding** lets one pass
produce several tokens. A small **drafter** guesses a block of tokens
ahead, the model checks the whole block in a single forward, and the
longest prefix it agrees with is kept.

[DFlash 2](https://inco.ai/blog/dflash2/) (Inco AI, built on z-lab's
[DFlash](https://arxiv.org/abs/2602.06036)) is a drafter of that kind. It
is a five-layer model that reads the 27B's own hidden states and drafts
seven tokens in one forward, chaining them into one coherent path with a
small candidate selector. vpipe runs it natively on the metal backend.

**The drafter changes speed, never the answer.** Greedy output is the
model's own greedy output. With a sampler, every kept token is the
model's own sample at that position; the drafts only decide how many land
per forward.

## What you need

- An Apple-silicon Mac with **32 GB** or more. The 4-bit model is 15 GB,
  and the drafter adds about 2.2 GB at its default 8 bits (1.3 GB at
  4 bits), plus the conversation's K/V.
- vpipe built from source (see the [Qwen3.5 chat guide](QWEN35-CHAT.md)
  for the build and the work-directory conventions; everything there
  applies here).

## Step 1 — get the model and the drafter

```sh
cd ~/vpipe-work                                    # your work directory
cp ~/src/vpipe/docs/pipelines/prepare-qwen38-27b-dflash2.vpipeline .
~/src/vpipe/build/apps/vpipe/vpipe --launch prepare-qwen38-27b-dflash2.vpipeline
```

Two `model-fetch` stages:

| registry key | what | size |
|---|---|---|
| `mlx-community/Qwen3.8-27B-4bit` | the model: uniform 4-bit, vision tower included | 16 GB |
| `incoai/Qwen3.8-27B-DFlash2` | the drafter (bf16 on disk; quantized at load) | 3.8 GB |

The drafter is trained for this model and **only** this model. vpipe
checks it against the model at load (layer count, width, vocabulary)
and refuses a mismatch with a warning, then serves without it.

## Step 2 — chat

```sh
cd ~/vpipe-work
cp ~/src/vpipe/docs/pipelines/qwen38-27b-dflash2-chat.vpipeline .
~/src/vpipe/build/apps/vpipe/vpipe --launch qwen38-27b-dflash2-chat.vpipeline
```

It is the [Qwen3.5 chat pipeline](QWEN35-CHAT.md) with two changes: the
`text-chat` stage names the drafter, and the sampler carries Qwen3.8's
published defaults (temperature 1.0, top-k 20, top-p 0.95). The load line
says the drafter is in:

```
[INFO] GenerativeModelManager::load('.../Qwen3.8-27B-4bit'): dflash2 drafter '.../Qwen3.8-27B-DFlash2' attached (w8)
```

and every reply reports how many tokens each forward produced:

```
[INFO] TextChatStage('chat'): dflash2 decode: 131 tokens in 59 rounds (2.22 a round), 72/73 drafts accepted
[INFO] TextChatStage('chat'): prefill 32 tok in 0.280 s = 114.3 tok/s, decode 130 tok in 5.494 s = 23.7 tok/s
```

### The settings

On `text-chat`:

| key | default | notes |
|---|---|---|
| `draft_model` | `""` | the drafter: a registry key or a directory. Set, it takes over from `mtp`. |
| `draft_block_size` | `0` | tokens a round verifies (the anchor plus drafts, 2 up to the drafter's trained 8). **0 adapts** — see below. |
| `draft_bits` | `8` | the drafter's in-memory precision: `8`, `4` (half the memory and draft time, slightly fewer accepted drafts), or `0` for its stored bf16. |

## What to expect

MEASURED on an M4 Pro (64 GB, internal fan), the same prompt and settings,
the drafter at 8 bits:

| | decode |
|---|---:|
| plain decode | 15.1 tok/s |
| DFlash 2, greedy | 37–39 tok/s (2.4–2.6×) |
| DFlash 2, Qwen3.8's sampler | 39.3 tok/s (2.6×) |

The gain depends on the content. Code and structured text draft well;
free prose drafts less well.

**What a round costs, and why "adaptive" settles near 4.** Checking a
drafted token adds a row to the model's matrix-vector work. vpipe's
verify kernels split each weight read across two SIMD groups so that up
to four rows cost about what one row does; five or six rows run one wider
tile, and eight run two. The same model and drafter on a shorter prompt,
every block size token-exact:

| block | ms per round | tokens per round | tok/s |
|---:|---:|---:|---:|
| 2 | 81 | 1.85 | 22.8 |
| 3 | 91 | 2.48 | 27.2 |
| 4 | 103 | 3.10 | 30.1 |
| 6 | 143 | 3.58 | 25.0 |
| 8 | 199 | 4.03 | 20.3 |

`draft_block_size: 0` measures this on the machine it runs on: each
length's round time against the drafts it gets accepted. It settles on the
best length within a few rounds, keeps that choice for the life of the
model, and re-checks a neighbouring length now and then. On a GPU where a
longer verify is cheaper still, it settles longer by itself; that is why
it is the default rather than a hard-coded length.

`VPIPE_QMV_KSPLIT=0` runs the verify on the row-identical kernels the
plain decode uses instead (slower; see Correctness).

## Correctness

The bar is the model's own decode. A greedy DFlash run is held
token-for-token to vpipe's plain greedy loop from the same state, at every
block size, in `tests/unit-tests/metal-lm/speculative-dflash.cc`. A
sampled run is held to the GPU sampler with the same seed, penalties
included.

One caveat comes from the arithmetic, not the drafter. In bf16, logits
carry 8 bits of mantissa, so two candidate tokens sometimes land one unit
apart, or exactly level. Two decode paths that add in a different order
can split such a tie the other way. vpipe's own two plain-decode entry
points already do on some prompts, and the verify's kernels add in a
different order from the plain decode's. A greedy divergence is therefore
accepted only where the plain decode's top two logits are within two bf16
units, and the tests assert exactly that. With a sampler the same
one-unit difference can, rarely, move a draw across a boundary; the
sampled test holds the loop exact on the plain decode's kernels and
checks the default kernels separately.

## Other drafters

The drafter is a framework piece (`generative-models/shared/
dflash-drafter.h`), configured from the drafter's own `config.json`.
z-lab's original DFlash drafters load through the same `draft_model` key;
`z-lab/Qwen3.5-4B-DFlash` for Qwen3.5-4B is in the catalogue and tested.
On a target that small the drafter costs as much as the model's own
verify, so it is no faster than plain decode on an M4. DFlash pays on big
models.

A model with an MTP head (`mtp`, the OptiQ packs) and a DFlash drafter
are two answers to the same question. When `draft_model` is set, DFlash is
used. MEASURED on the same prompt with Qwen3.8-27B's published MTP head
(`mlx-community/Qwen3.8-27B-MTP-4bit`): 24.7 tok/s drafting one token a
round, 31.8 drafting two, against DFlash 2's 37–39.

## Under the hood

- **The drafter computes in bf16** whatever the model runs in. Its
  residual stream runs to tens of thousands, past f16's range, so with a
  `compute_dtype: f16` model it converts where the two meet (the model's
  hidden states in, the embeddings in, the normed hidden out to the head).
- **The drafter borrows the model's embedding and output head.** It has
  neither of its own. It conditions on the model's residual stream at
  five layers (5, 19, 33, 47 and 61 of 64), concatenated and projected into
  each drafter layer's K/V cache.
- **Every text prefill feeds the drafter.** The model taps those layers
  over the rows the drafter's 2048-token window can see. After each round,
  the kept tokens' rows from the verify join the cache too, so the
  drafter's context follows the conversation across turns.
- **A round is one command buffer:** the drafter's block forward, its
  top-16 candidates per position, the selector's walk, the embedding of
  the drafts, and the model's verify forward. The host reads back only
  token ids.
- **Qwen3.8 is a hybrid.** 48 of its 64 layers are gated DeltaNet, whose
  recurrent state has no position to roll back to. The verify reads that
  state from one buffer and writes another. A partial accept replays the
  kept rows from the round's saved inputs at the start of the next round,
  so a rejected draft never costs a host copy.
- **Bisecting.** `VPIPE_DFLASH_PROFILE=1` prints a per-decode summary and
  the adaptive tuner's table; `=2` also splits each round into draft and
  verify time.

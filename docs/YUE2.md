# YuE2 on Apple Silicon

**YuE2** writes songs: a style prompt and a set of lyrics in, a complete
48 kHz stereo song with vocals and accompaniment out. It is a 3.6-billion-
parameter model from [m-a-p](https://huggingface.co/m-a-p/YuE2-3B), and it
works the way a musician might — it first writes a **score** (melody, and
chords if you ask for them) in ABC notation, then sings and plays to it. The
score is yours to keep, edit and hand back, which is what makes covers and
revisions possible.

vpipe runs it on-device through its **metal-compute** backend: its own Metal
kernels, no Python and no third-party tensor runtime in the forward pass.
Nothing is quantized; the checkpoint is bf16 and runs as bf16.

**The weights are CC BY-NC 4.0 — non-commercial use only.** That is the
model's license, not vpipe's; read it before you publish anything you make.

## What you need

| | |
|---|---|
| **Machine** | Apple Silicon Mac (M-series). |
| **Memory** | **11.0 GB** at its peak: 6.2 GB of weights (the song-writing half held at 8 bits — `lm_quant`, the default; 7.4 GB with `bf16`) and a 4.8 GB working set sized for a six-minute song, 2.3 GB of which stays held for the rest of the run. Measured on a 64 GB machine; a 16 GB one should hold it but has not been timed. |
| **Disk** | **~7.3 GiB** — 6.8 GiB for the model, 506 MiB for the decoder. |
| **Build** | An Apple Silicon build of vpipe — the default on arm64 macOS. See the main [README](../README.md). |
| **Account** | None. Both repos are public and ungated. |

## The pipelines

- **[`prepare-yue2.vpipeline`](pipelines/prepare-yue2.vpipeline)** — fetch the
  model and its decoder. Run once.
- **[`yue2-text-to-music.vpipeline`](pipelines/yue2-text-to-music.vpipeline)**
  — style and lyrics in, `yue2-song.m4a` and its score `yue2-song.abc` out.
- **[`yue2-text-to-music-fast-m4.vpipeline`](pipelines/yue2-text-to-music-fast-m4.vpipeline)**
  and **[`yue2-text-to-music-fast-m5.vpipeline`](pipelines/yue2-text-to-music-fast-m5.vpipeline)**
  — the same song with the acceleration that pays on each machine.
- **[`yue2-cover-from-score.vpipeline`](pipelines/yue2-cover-from-score.vpipeline)**
  — a cover: you supply the melody, the model arranges and sings it.
- **[`yue2-songs-from-lyrics.vpipeline`](pipelines/yue2-songs-from-lyrics.vpipeline)**
  — one song per lyrics file, in one run.

Any of them runs from the terminal with `vpipe --launch <file>` or opens with
**Load** in the web UI's Pipeline Manager.

## Step 1 — get the models

```sh
vpipe --launch docs/pipelines/prepare-yue2.vpipeline
```

Two fetches: `m-a-p/YuE2-3B` (the model, the tokenizer and an example
request) and `m-a-p/YuE2-Vae` (the decoder). The fetch takes only what runs
here — the repo also carries demo audio, figures and the reference's Python
packages, none of which is needed.

## Step 2 — write a song

Edit `style` and `lyrics` on the `generate-audio` stage, then:

```sh
vpipe --launch docs/pipelines/yue2-text-to-music.vpipeline
```

### The shape of the graph

```
generate-audio ─┬─▸ audio-vae-decode ─▸ save-audio      the song
                └─▸ save-text                            its score
```

`generate-audio` does everything except turn latents into sound. It writes in
three passes over one checkpoint:

1. **Plan** — an ABC score of the song (skipped when you supply one, or with
   `cot: off`). Emitted on the `score` port.
2. **Song** — one semantic token per 40 ms of audio, until the model decides
   the song is over. **The model chooses the length**; nothing here asks for
   one. `max_seconds` only caps it.
3. **Latents** — 32 flow-matching steps turn those tokens into 64-wide audio
   latents, 25 per second, on the `audio_latent` port.

`audio-vae-decode` is a separate stage on purpose: the model ships two
decoders for one latent space (see [Choosing a decoder](#choosing-a-decoder)),
and the choice belongs to the graph, not to the generator.

### Writing the prompt

**`style`** is a comma-separated list of tags: genre, mood, instruments,
vocal and feel. The example uses

> Indie pop, warm female vocal, acoustic guitar, soft drums, bright synth
> pads, uplifting, summer evening

**`lyrics`** are sectioned with headers on their own lines, a blank line
between sections:

```
[Intro]

[Verse]
Streetlights hum a quiet tune
Shadows dancing with the moon

[Chorus]
Stay awake with me tonight
...
```

`[Intro]`, `[Verse]`, `[Pre-Chorus]`, `[Chorus]`, `[Bridge]`, `[Interlude]`
and `[Outro]` are the headers the authors' examples use; an empty section is
an instrumental one. The model card lists **Chinese and English**. The
checkpoint ships one complete request, `examples/tonight-awake.json` in the
model folder — 今晚不眠, a Mandarin city-pop song, with its style, lyrics and
seed — and it is a good first test of a non-English prompt.

### Planning: `cot`

| `cot` | what the model plans first |
|---|---|
| `full` (default) | melody and chords |
| `melody` | melody only — what the authors recommend for covers |
| `off` | nothing; it goes straight to the song |

`cot: off` writes no score, so the `score` port stays silent and `save-text`
writes nothing.

## Covers and edits: bring your own score

Give `generate-audio` an `abc` and it skips planning and follows yours. That
is the whole mechanism behind both workflows:

- **Edit a song.** Run text-to-music, open the `.abc` it saved, change what
  you want — reharmonize, rewrite a phrase, add a solo — and run again with
  your edited score in `abc`, the same `seed`, and a new `style` if you like.
- **Cover a song.** Write or transcribe its melody, put it in `abc` with
  `cot: melody`, and pick a target style. The authors transcribe recordings
  with their [SheetSage2](https://huggingface.co/m-a-p/SheetSage2) model;
  `cot: melody` does not strip chord symbols for you, so leave them out of a
  melody-only score.

[`yue2-cover-from-score.vpipeline`](pipelines/yue2-cover-from-score.vpipeline)
is a worked example: *Twinkle, Twinkle, Little Star* — a public-domain tune
and public-domain words — as jazz-funk. It runs 80 seconds, and the length
comes straight from the score: 32 bars at 96 beats per minute.

### The score's dialect

The model reads and writes one particular shape of ABC, and a score you hand
it should look like the ones it writes. The simplest way to learn it is to
run text-to-music once and read the `.abc`. In outline:

```
X:1
T:
M:4/4                     meter
L:1/16                    the note grid (the model also uses 1/32)
Q:1/4=96                  tempo
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:C                       key
% intro                   a comment per section, matching the lyrics
V: Vocal
z16|z16|z16|z16|          the sung line, bar by bar
V: Ins
c4c4g4g4|a4a4g8|...       the lead instrument's line
% verse
V: Vocal
C4C4G4G4|A4A4G8|...
V: Ins
Z4|                       a four-bar rest
```

Two voices, always: the **vocal melody** and an **instrumental melody**. With
`cot: full` each bar may open with a chord symbol in quotes (`"Dm7"e3A3...`);
with `cot: melody` there are none. Sections are `%` comments in the same
order as the lyrics' headers — that is how the words find their bars.

## Many songs in one run

Wire anything that emits text into `generate-audio`'s `prompt` port and it
writes **one song per beat**; unwired, it writes one song from its config and
ends.

[`yue2-songs-from-lyrics.vpipeline`](pipelines/yue2-songs-from-lyrics.vpipeline)
does it with `load-text`: list your lyrics files in its `path`, and every
file becomes a song in the configured style, saved as `yue2-song-00.m4a`,
`yue2-song-01.m4a`, ..., with all the scores in `yue2-songs.abc`.

A plain string beat is the **lyrics**. A beat that is an **object** can carry
any of `style` (or `tags`), `lyrics`, `abc`, `cot`, `seed` and `cfg_scale`;
whatever it leaves out comes from the stage's config. That is the form to
use when a program feeds the port and every song needs its own style.

## The knobs

| key | default | what it does |
|---|---|---|
| `hf_dir` | — | The model: `m-a-p/YuE2-3B`, or a folder. |
| `lm_quant` | `w8` | How the half that writes the score and the song tokens is held: `w8` builds it at 8 bits in memory as it loads — 1.26 GB less, and that half runs 1.4× faster — `bf16` keeps it as published. The flow matching is bf16 either way. |
| `style` | | The tags. |
| `lyrics` | | The sectioned lyrics. |
| `abc` | empty | A score to follow; empty plans one. |
| `cot` | `full` | `full`, `melody` or `off`; see above. |
| `seed` | 831001 | The token draws and the flow-matching noise. |
| `cfg_scale` | protocol | Text guidance on the song tokens; the protocol uses 1.0 for `full`/`melody` and 1.01 for `off`. Above 1, the song phase decodes twice. |
| `max_seconds` | 0 | A cap on the song's length; 0 is the protocol's six minutes. |
| `ode_steps` | 32 | Flow-matching steps, two model passes each. 32 is what the model was released with. |
| `abc_temperature` / `_top_p` / `_top_k` / `_repetition_penalty` | 0.7 / 0.9 / 30 / 1.005 | Sampling of the score. |
| `song_temperature` / `_top_p` / `_top_k` / `_repetition_penalty` | 1.0 / 0.95 / 100 / 1.2 | Sampling of the song tokens. |

The sampling defaults are the release protocol's, including its
repetition-penalty windows (the last 100 score tokens, the last 50 song
tokens). A song is **reproducible for a seed**, but it is not the reference
implementation's song for that seed: the flow-matching noise is drawn
identically to the reference, the token draws are not.

## Going faster

All acceleration is on the flow-matching pass, which is most of a song's time
on every machine here. The keys are the same ones the image and video stages
use. `sage_attn` and `i8_gemm` need the GPU's matrix units and switch
themselves off, with a one-line notice, on a Mac without them. `ane_ffn` runs
wherever it is asked — and costs time on an M5:

| key | where it pays | what it changes |
|---|---|---|
| `ane_ffn` | **Macs without GPU matrix units** (measured on M4): splits the feed-forward rows between the GPU and the Apple Neural Engine, run at the same time. A loss on M5, whose matrix units already have the work. | Exact up to the ANE's fp16. Engages only for songs over ~82 s (2048 latent rows, the module's chunk). |
| `sage_attn` | **M5**: the attention scores in int8 on the matrix units. | Close: latents 0.037 from dense. |
| `i8_gemm` | **M5**: the big matrix products in int8. | Close: latents 0.05 from dense. |
| `sol_attn` | everywhere: attends only the key blocks a cheap proxy says matter, the text prompt always exact. `sol_tau` trades speed for fidelity. | **A different rendering.** The latents move 0.24–0.30 from dense, while the vocals still transcribe as the lyrics just as well (character error 0.13–0.15 against dense 0.14). Judge it by ear. |

The two `-fast` pipelines turn on what measured best on each machine:
`ane_ffn` + `sol_attn` for M4, `sage_attn` + `sol_attn` for M5.

Measured on the flow matching of a 3 min 34 s song — the bundled 今晚不眠
request — at 32 steps:

| M4 Pro, 64 GB | time | | M5 Pro | time |
|---|---:|---|---|---:|
| dense | 317 s | | dense | 80–83 s |
| `ane_ffn` | 278 s (1.14×) | | `sage_attn` | 72 s (1.14×) |
| `sol_attn`, tau 0 / 0.5 / 1 | 262 / 248 / 235 s | | `i8_gemm` | 75 s (1.09×) |
| `ane_ffn` + `sol_attn` | **225 s (1.41×)** | | `sage_attn` + `sol_attn` | **64 s (1.30×)** |

Higher `sol_tau` keeps fewer key blocks (61 / 52 / 44% at 0 / 0.5 / 1): it is
a threshold, not a budget.

## What it costs

On an **M4 Pro, 64 GB**, the same song end to end, dense:

| | |
|---|---|
| load | ~10–14 s |
| plan the score | 42 s |
| write the song | 102 s (52 tokens/s) |
| flow matching | 320 s |
| decode | ~6 s |
| **total** | **7 min 50 s** |

With `ane_ffn` and `sol_attn` the whole run is 6 min 17 s. The example
pipeline's own song, unchanged, comes out at 2 min 10 s — the model's choice
— in **4 min 21 s**. Planning and the song tokens run at 52–59 tokens/s with
that half in bf16, as measured above; held at 8 bits (`lm_quant: w8`, the
default) a 30-second song's ran at 74 and 87 tokens/s against 54 and 60. The
flow matching is what grows fastest with the song's length, because its
attention spans the whole song.

## Choosing a decoder

The model ships two decoders for the same latents:

- **`m-a-p/YuE2-Vae`** — the default, and what the authors recommend for
  listening: better perceptual quality.
- **`m-a-p/YuE2-Vae-legacy`** — the decoder the paper's benchmarks were
  measured with.

To use the legacy one, fetch it (change `model_path` in a copy of the
prepare pipeline) and set `hf_dir` on `audio-vae-decode` to
`m-a-p/YuE2-Vae-legacy`. Nothing else changes.

## What is inside

- **One checkpoint, two transformers per layer.** Each of the 28 layers holds
  an autoregressive weight set (shaped like a 1.7B Qwen3: 2048 wide, 16 query
  heads over 8 KV heads, a 6144-wide SwiGLU) and a second set for the flow
  matching. The authors call it an AR–NAR Mixture-of-Transformers.
- **One vocabulary for words and sound.** 184,704 entries: the text tokens,
  a handful of control tokens, and 32,768 codec codes, one per 40 ms of
  audio. vpipe computes only the slice of the output layer a pass can pick
  from — the text rows while planning, the codec rows while singing — rather
  than all 184,704 every step.
- **The flow matching attends the whole song at once.** Its 64-wide latent
  rows read the autoregressive pass's keys and values for the prompt, the
  score and every song token, plus each other — 5,347 rows over 13,393 keys
  for the 3:34 song. That is why it dominates the time, and why the
  attention tiers pay.
- **The decoder** is an Oobleck-style convolutional VAE: 25 latents a second
  in, 48 kHz stereo out, 1,920 samples per latent. vpipe decodes it in tiles
  with an overlap, so its memory does not grow with the song.
- **The text tokenizer** is the model's own `qwen.tiktoken`, read directly.

## Verification

Checked numerically against the reference implementation (the authors'
`yue2_infer` package, run in fp32 and bf16). The bar throughout is **the
reference's own bf16 error against its fp32 run**: matching bf16 more
closely than the reference matches itself is not possible, and not the goal.

| | vpipe | the reference's bf16 |
|---|---|---|
| tokenizer, prompt prefixes | token-exact | |
| flow-matching noise | bit-identical | |
| score logits vs fp32 | 0.0023 | 0.009 |
| song logits vs fp32 | 0.026 | 0.023 |
| flow velocity vs fp32 (two times) | 0.0125 / 0.019 | 0.0149 / 0.032 |
| decoder | 0.0017 | |

With the song-writing half held at 8 bits (`lm_quant: w8`, the default) the
same checks read: score logits 0.0022, song logits 0.0135, flow velocity
0.0153 / 0.021 — every one inside the reference's own bf16 error — and the
same 29 of 29 decisive picks, the same 48- and 32-token greedy runs.

Greedy decoding matches the reference token for token over the stretches
checked: the first 48 tokens of a **score**, and the first 32 of a `cot: off`
song with guidance. The planned song phase, free-running, does not — and
cannot: its next-token choices are full of near-ties (top-two margins of
0.01–0.07), and the reference's own bf16 and fp32 runs part ways within 40
tokens. So that phase is checked **teacher-forced**, on the steps where fp32
has a clear winner (29 of 29).

Last, the ear's proxy: a speech recognizer transcribes the generated vocals
as the requested lyrics — `yue2.sung_lyrics_transcribe` writes a 30-second
song per holding and holds the share of lyric words heard back (bf16 22 of
22, w8 20 of 22, its last two words cut off by the 30-second cap). It is
opt-in: set `VPIPE_YUE2_ASR_CHECK=1` and `VPIPE_QWEN3_ASR_TEST_MODEL_PATH`.

To re-run it, point three variables at the model, the decoder and a golden
directory written by `tools/dump_yue2_golden.py`:

```sh
VPIPE_YUE2_TEST_MODEL_PATH=<models>/m-a-p/YuE2-3B \
VPIPE_YUE2_VAE_TEST_MODEL_PATH=<models>/m-a-p/YuE2-Vae \
VPIPE_YUE2_GOLDEN=<goldens>/yue2 \
  vpipe_test --filter 'yue2.*'
```

Unset, those tiers skip; `yue2.protocol_logic` needs none of them.

# MiniMax-Music3 on Apple Silicon

**MiniMax-Music3** writes complete songs: a description of the music and a
set of lyrics in, a stereo song with vocals and accompaniment out, up to six
minutes long. It is an 11-billion-parameter model from
[MiniMax](https://huggingface.co/MiniMaxAI/MiniMax-Music3) in three parts —
an **8B language model** (initialized from Qwen3-8B) that plans the song one
25 Hz frame at a time, a **0.6B depth decoder** that fills in each frame's
acoustic detail, and a **2.4B flow-matching transformer** that turns their
hidden states into audio latents — and its own **DAC vocoder** decodes those
to 44.1 kHz stereo.

vpipe runs it on-device through its **metal-compute** backend: its own Metal
kernels, no Python and no third-party tensor runtime in the forward pass.

**The weights are under the MiniMax-Music3 Community License.** Commercial
use must display "MiniMax-Music3" in the product's interface; above USD 20
million in yearly revenue it needs MiniMax's written authorization; a hosted
service must keep safeguards against infringing use; and there is an
acceptable-use policy. That is the model's license, not vpipe's — read it
(`LICENSE` in the model folder) before you publish anything you make.

## What you need

| | |
|---|---|
| **Machine** | Apple Silicon Mac (M-series) with **16 GB or more**. On 16 GB the two halves of the model take turns (see [On a 16 GB Mac](#on-a-16-gb-mac)); from 24 GB both stay loaded. |
| **Memory** | **13.3 GB** of weights by default: the language model and the depth decoder held at 8 bits (`lm_quant: w8`, 8.4 GB; 15.5 GB with `bf16`), the flow-matching transformer at 16 bits (4.9 GB; 2.5 GB with an int8 tier that holds it, `i8_gemm_tier: fast` or `token`, on a Mac with GPU matrix cores). A song's working set comes on top, and the plan books it at the six-minute cap with a 5000-token prompt: about 5.0 GB — the two language-model contexts' K/V (the pool grows by doubling, capped at the two contexts' own 3.1 GB), the prompt's prefill (~1.1 GB), the frames' hidden states (64 KB a frame, 0.6 GB at the cap) and the flow matching's 8 s window (~130 MB). A shorter song holds less. |
| **Disk** | **~28.6 GB** — the repository's diffusers layout. The repository also carries the original layout (another ~29 GB of the same weights); `prepare-minimax-music3` fetches only what runs. |
| **Build** | An Apple Silicon build of vpipe — the default on arm64 macOS. See the main [README](../README.md). |
| **Account** | None. The repository is public and ungated. |

## The pipelines

- **[`prepare-minimax-music3.vpipeline`](pipelines/prepare-minimax-music3.vpipeline)**
  — fetch the model. Run once.
- **[`minimax-music3-text-to-music.vpipeline`](pipelines/minimax-music3-text-to-music.vpipeline)**
  — a description and lyrics in, `minimax-music3-song.m4a` out.
- **[`minimax-music3-text-to-music-fast-m5.vpipeline`](pipelines/minimax-music3-text-to-music-fast-m5.vpipeline)**
  — the same song with the flow matching on the M5's int8 matrix cores
  (`i8_gemm_tier: token`): ~1.7x faster flow matching, half the
  transformer's memory. On a Mac without matrix cores it runs as the one
  above.

## Step 1 — get the models

Run `prepare-minimax-music3.vpipeline`. It downloads the diffusers layout of
`MiniMaxAI/MiniMax-Music3` into `./models` and registers it, so the other
pipelines can name it by its key.

## Step 2 — write a song

Edit `minimax-music3-text-to-music.vpipeline`'s `style` (the description) and
`lyrics`, and run it. The song comes out as `minimax-music3-song.m4a`.

### The shape of the graph

```
generate-audio  ──audio_latent──▶  audio-vae-decode  ──pcm──▶  save-audio
(hf_dir = the model)               (hf_dir = the same model:
                                    its vocoder/)
```

`generate-audio` reads the model's `config.json` and runs MiniMax-Music3 for
a `minimax_music3` repository (and YuE2 for a YuE2 one — the two share the
stage and its keys). It emits ONE latent per song, `[1, 128, latents]` at
44100 / 512 latents a second. `audio-vae-decode`, given the same directory,
finds the vocoder in it.

### Writing the description

The description is the `style` key (or a prompt beat's `"style"` /
`"caption"`). A short sentence works; for real control, write a **structured
caption** in the three sections the model was trained on:

- **Global Metadata** — genre and subgenre, BPM, key and scale, the song's
  emotional progression, a listening scenario, the production profile.
- **Vocal Details** — the singer's gender and timbre, the performance style,
  harmonies and backing vocals, vocal effects.
- **Arrangement** — the primary and secondary instruments and how they enter
  and leave, the groove, bass and percussion, textures and spatial effects.

The shipped pipeline's description is one. MiniMax publishes a prompt
rewriter (the `music-caption-rewriter` skill in
[MiniMax-AI/MiniMax-Music3](https://github.com/MiniMax-AI/MiniMax-Music3))
that expands a short description into this form.

Markdown is fine: headings, bullets, bold and emphasis are stripped before
the model sees the text, blank lines collapse, and a tag like `<|bpm 92|>`
becomes "bpm is 92" — the model's own input contract, reproduced exactly.

### Writing the lyrics

Lyrics take section tags — `[Intro]`, `[Verse]`, `[Pre-Chorus]`, `[Chorus]`,
`[Post-Chorus]`, `[Bridge]`, `[Instrumental]`, `[Solo]`, `[Outro]` —
**each on its own line**. Text after a tag on the same line is dropped (the
model's contract), tags are lower-cased, and a `" ^ "` is a line break.

Both the description and the lyrics are required, and the two together must
come to at most 5000 tokens.

## The knobs

| key | default | what it does |
|---|---|---|
| `style` | — | The description. Required. |
| `lyrics` | — | The lyrics. Required. |
| `max_seconds` | 0 | A cap on the song's length; 0 is the model's own (9000 frames, six minutes, within its 10240-token context). The model ends the song itself, usually before the cap. |
| `seed` | 831001 | The draws of the song's codes and the flow matching's noise. A song is reproducible for a seed on one machine; it is not the reference implementation's song for that seed (see Verification). |
| `lm_quant` | `w8` | How the language model and the depth decoder are held: `w8` quantizes their projections to 8 bits as they load; `bf16` keeps them as published. See Verification for what it costs. |
| `ode_steps` | 0 | The flow matching's Euler steps per 8 s window; 0 is the model's 30. |
| `flow_cfg_scale` | −1 | The flow matching's guidance; below 0 is the model's 1.7. |
| `cfg_scale` | −1 | The guidance of the codes' draws; below 0 is the model's 1.5. |
| `i8_gemm` | false | Runs the flow-matching transformer's block projections as int8 GEMMs on the GPU's matrix cores (the M5 family). LOSSY. Ignored, with a note in the log, where there are no matrix cores. |
| `i8_gemm_tier` | `accurate` | Which int8 GEMM: `accurate` keeps the transformer 16-bit and quantizes its weights per call; `fast`, `token` and `token_plain` HOLD its block weights as int8 (2.5 GB instead of 4.9), converted once as they load. `token` is the fastest that stays close (see What it costs and Verification); `token_plain`, unrotated, is no faster and several times further from the reference. |

The keys `abc`, `cot` and the `abc_*` / `song_*` sampling keys are YuE2's and
do nothing here; `ane_ffn`, `sage_attn` and `sol_attn` are YuE2's
flow-matching tiers and are not wired for MiniMax-Music3 (the stage says so
when they are set).

**Sample rate.** The vocoder decodes at 44.1 kHz, and so does this graph.
MiniMax's own serving stack resamples that to 32 kHz before delivering it;
if you want its exact format, add an `audio-temporal-resample` stage at
`output_sample_rate: 32000`.

## On a 16 GB Mac

The model's two halves — the language model with its depth decoder (8.4 GB),
and the flow-matching transformer with its condition encoder (4.9 GB, or 2.5
with an int8 tier) — are only needed one after the other: a song's codes are
written first, then turned into audio. When the Mac's memory cannot hold both
halves with a song's working set and a 6 GB cushion, `generate-audio` holds
them IN TURN: the language model loads for the song's codes and is let go
before the transformer loads for the flow matching, and the other way round
for the next song. The memory plan books each half in its own phase, so the
rest of the graph sizes against one half, not both. Nothing about the song
changes — it is the same audio, sample for sample, as with both held — and
the cost is a load of each half per song (a few seconds each from an
internal SSD, against minutes of generation).

The decision is the engine's, taken from the Mac's total memory before
anything loads; `VPIPE_MUSIC3_SWAP=0` or `1` pins it for a test. MEASURED on
the M4 Pro with its memory capped to 16 GB (`VPIPE_RAM_LIMIT_MB=16384`), a
10 s song: the process's peak footprint 8.5 GB taking turns against 13.5 GB
with both halves held, the PCM identical. A long song adds its K/V to the
language model's half — the pool grows by doubling, so a two-minute song
with a long caption holds ~2.4 GB of it, and the six-minute cap 3.1 GB (the
two contexts' own), which with a 5000-token prompt's prefill books that half
at about 13.3 GB — so on 16 GB a song of a few minutes is comfortable and
the six-minute cap is tight.

## What it costs

MEASURED on an M4 Pro (64 GB), `lm_quant: w8`, the default steps:

| | |
|---|---|
| **The song's codes** | 57-58 ms a frame over a 250-frame song (language model 36, depth decoder 21-22), in the latest measurements; earlier sittings read 72-75 with the same arithmetic, and the difference was not traced. A frame costs more as the song's context grows. |
| **The flow matching** | 0.94-1.07 s an evaluation of both guidance rows; 30 evaluations per 8 s window, windows on a 4 s hop — about 8 s of compute per second of song. |
| **The vocoder** | ~1 s for 10 s. |
| **A whole song** | 40 s in 6 m 50 s (codes 81 s, flow matching 322 s); 90 s in 15 m 40 s (225 s, 709 s). |

The M4 Pro's flow matching runs at its GEMM rate (~7 TFLOP/s): it is the
whole cost of a long song. A Mac with GPU matrix cores (the M5 family) runs
those GEMMs and the attention on them, and can run the GEMMs in int8
(`i8_gemm`). MEASURED on a base M5 (a fanless 16 GB MacBook Air, on AC), one
evaluation of both guidance rows at a full 8 s window, from a BF16 copy of
the transformer (the release's F32 does not fit that Mac's disk); the fanless
Mac drifts run to run, so read the ratios:

| `i8_gemm_tier` | transformer held | ms an evaluation | vs 16-bit |
|---|---|---|---|
| (16-bit) | 4.9 GB | 531-566 | 1.0x |
| `accurate` | 4.9 GB | 477-531 | ~1.1x |
| `fast` | 2.5 GB | 369-409 | ~1.4x |
| `token` | 2.5 GB | 309-335 | ~1.7x |
| `token_plain` | 2.5 GB | 313-332 | ~1.7x |

A song's short last window (under 1024 rows, e.g. 696) runs `token` in 174
ms against 16-bit's 310. The language model was not run on the M5 (that Mac
has no disk room for it); it reads every weight of both halves of the
autoregressive model once a frame, so its rate follows the Mac's memory
bandwidth.

## What is inside

1. **The codes.** The prompt — the description and lyrics wrapped in the
   checkpoint's special tokens — is prefilled twice, as is and with its text
   replaced by `<|audio_cfg|>` (the guidance branch). Per frame, the language
   model's head over the 16384 semantic codes and the end token picks `c0`
   under guidance 1.5, restricted to the conditional branch's top 50; the
   depth decoder then picks `c1`..`c7` (1024 codes each) the same way, each
   from its own head. The frame is fed back as the sum of its eight
   embeddings. The first frame only moves past `<|audio_start|>`.
2. **The condition.** Each frame's eight hidden states (the language
   model's, then the depth decoder's at each code) are mixed with learned
   weights and convolved to a 2048-wide condition, resampled from 25 to
   44100 / 512 a second.
3. **The latents.** 200-frame windows on a 100-frame hop; each is solved
   from noise by 30 Euler steps of the transformer under guidance 1.7
   against an all-zero condition, its first 172 latents pinned to the
   previous window's by a blend with the noise.
4. **The audio.** The vocoder decodes the song's latents, its 128 channels
   as two 64-wide streams, left and right. The reference decodes each window
   and crops the overlaps; the crop seam lies 86 latents inside a span both
   windows share exactly and the decoder sees about 11 latents either side,
   so decoding the stitched latents is the same waveform (measured: 1.5e-7
   apart in the reference itself) — one decode a song.

## Verification

Against the reference — the diffusers modular pipeline the checkpoint
ships for, run in fp32 on the CPU, with its own bf16 run as the bar
(`tools/dump_minimax_music3_golden.py`; tests `minimax_music3.*`):

| | vpipe | reference bf16 |
|---|---|---|
| prompt text and ids (4 cases: markdown, special tags, Unicode line separators, Cyrillic and Chinese tags) | identical | — |
| codes, teacher-forced over 251 frames: `c0` logits / depth logits / frame states, `bf16` | 0.0094 / 0.0099 / 0.0112 | 0.0099 / 0.0104 / 0.0115 |
| the same, `w8` | 0.0121 / 0.0122 / 0.0147 | |
| decisive `c0` picks (fp32 margin ≥ 0.5) | 157 / 157 | |
| condition | 0.0003 | 0.0035 |
| one transformer evaluation at the fp32 trajectory's states | 0.0021–0.0024 | 0.0165 |
| a 2-window song's stitched latents (30 steps from the same noise) | 0.0022 | 0.0214 |
| vocoder, the reference's per-window crops | 6.0e-4 | — |

The transformer runs **f16**, not bf16: its residual stream peaks near 10^4
on the time token (the song's own rows stay under ~1.5e3), well inside
f16's range, and f16's mantissa is why it lands ten times closer to fp32
than the reference's bf16 does.

**The int8 tiers** (`i8_gemm`, base M5, from a BF16 copy of the
transformer — whose own rounding puts the 16-bit run at 0.0070 per
evaluation and 0.0077 over the song, against 0.0022 from the release's
F32):

| `i8_gemm_tier` | one evaluation | the 2-window song | reference bf16 |
|---|---|---|---|
| `accurate` | 0.0082 | 0.0098 | 0.0165 / 0.0214 |
| `fast` | 0.0089 | 0.0115 | |
| `token` | 0.0099 | 0.0146 | |
| `token_plain` | 0.0438 | 0.0899 | |

So the three rotated tiers keep the flow matching closer to the reference's
fp32 than the reference's own bf16 run is; `token_plain` does not, because
without the rotation an outlier channel sets a token's scale over its whole
row.

**Heard back.** Qwen3-ASR, given the vocals BS-RoFormer separates from
the song, transcribes vpipe's render of MiniMax's own example request (its
caption and lyrics, 90 s) as the lyrics in order — "I'm learning how to
fill up every space I used to leave, teaching my own heart the patience
that my family needs from me…" — as it does MiniMax's published render of
the same request; a 40 s song from a structured caption and a verse and
pre-chorus came back 52 words of 52 (test `sung_lyrics_transcribe`).
Give it a real description: from a one-line one and a four-line lyric
the model opens with ~30 s of wordless singing, and the reference's own
render of the golden's codes does the same.

The codes are **sampled**, and the reference draws them with
`torch.multinomial`; vpipe draws from the same distributions with a
generator of its own. So a song is reproducible for a seed, but it is not
the reference's song for that seed — the teacher-forced test above is where
the two are compared.

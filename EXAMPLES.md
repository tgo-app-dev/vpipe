<a id="top"></a>
# VPIPE Examples

Hands-on walkthroughs that take you from an empty session to working
text-chat and speech-transcription pipelines. They use the **web UI** as the
front end and the same model **catalogue** the project ships with.

[1. Fetch a model](#1-fetch-a-model-interactively) ·
[2. Interactive text chat](#2-interactive-text-chat) ·
[3. Microphone transcription](#3-microphone-transcription) ·
[4. Better transcription with VAD](#4-better-transcription-with-vad-segmentation)

> These examples exercise the on-device LLM/ASR stages, so they require an
> **Apple Silicon** build (the default on arm64 macOS — see the main
> [README](README.md)).

---

## Before you start
[back to top](#top)

Launch the web UI and open the URL it prints:

```sh
./build/apps/web-ui/vpipe-web-ui
# -> open http://localhost:9876  (or the LAN URL it prints)
```

Two views do all the work in these walkthroughs:

- **Pipeline Manager** — create a pipeline, drag in stages, set their config,
  wire output ports to input ports, and start/stop. **Editing requires the
  pipeline to be stopped**; start/connect/config actions are disabled while a
  pipeline runs.
- **User I/O** — where interactive stages show their prompts and you type
  answers. **Open this view before launching any interactive pipeline.** An
  interactive stage (model fetch, text input) *blocks* until its prompt is
  answered, and the prompt only appears in the User I/O view — if it isn't
  open, the pipeline just stalls.

**Building a pipeline in the Pipeline Manager:**

1. Click **Create** and give the pipeline an id.
2. From the **Toolbox** palette, **drag a stage onto the canvas** (or
   double-click it). Give it a stage id when prompted.
3. Click a stage to select it; set its config fields in the right-hand panel
   and click **Apply**.
4. **Wire ports**: click a stage's **output port** to arm it, then click the
   target stage's **input port**. (Click an output again to disarm; press
   `Esc` to cancel.)
5. Use the per-pipeline **Start / Pause / Stop** controls to run it.

**Build these pipelines interactively in the Pipeline Manager** — that's the
intended way to learn the stages and see how they connect. Each example below
walks you through it click by click.

The JSON shown after each example is the **outcome** of that interactive build,
not a substitute for it: it's what the web UI writes when you click **Save**
(a `.vpipeline` file). Keep it for reference, to version-control a pipeline, or
to reopen later with **Load** — but the walkthroughs assume you're building in
the UI. The spec format is:

```jsonc
{
  "id": "my-pipeline",
  "stages": [
    {
      "id": "stage-id",            // unique within the pipeline
      "type": "stage-type-name",   // a registered stage type
      "iports": [                  // positional: entry i feeds this stage's input i
        { "src": "upstream-stage-id", "oport": 0 }
      ],
      "config": { /* stage-specific keys */ }
    }
  ]
}
```

A stage's `iports` array is **positional** — the first entry connects to the
stage's input port 0, the second to input port 1, and so on. A `src` must name
a stage **declared earlier** in the array (forward references are rejected), so
order matters in the JSON. `oport` defaults to `0`.

**Referencing models.** Once you fetch a model (Example 1), downstream stages
refer to it by its **registry key** (the catalogue's HuggingFace path, e.g.
`mlx-community/Qwen3.5-4B-MLX-4bit`). A model field also accepts a plain
**filesystem path** to a model directory, which always works as a fallback.

**Prefer the terminal?** The web UI isn't the only front end — VPIPE also ships
a command-line entrance, **`vpipe`** (`./build/apps/vpipe/vpipe`), that launches
pipelines straight from the shell. Two uses fit these walkthroughs:

- **One-off single-stage pipelines** — wrap a single stage in a throwaway
  pipeline with `--launch-stage <type>`, setting its config with
  `--stage-cfg key=value`. Ideal for utility stages such as `model-fetch`
  (Example 1, shown next) or `onvif-discovery`.
- **Replaying a saved pipeline** — once you've tuned a pipeline in the web UI
  and hit **Save**, run that `.vpipeline` spec in the terminal with
  `--launch <file>`, overriding any stage's config with
  `--stage-cfg stage-id::key=value`.

Under the CLI an interactive stage prompts **directly on the terminal**
(stdin/stdout) — there's no User I/O view to open. The web-UI walkthroughs
below are still the best way to *learn* the stages; reach for the CLI once you
know what you want to run. `vpipe --help` lists every option.

---

## 1. Fetch a model interactively
[back to top](#top)

**Goal:** download and register the MLX-community **Qwen3.5 4B (4-bit)** model
using the **Model Fetch** stage. We'll reuse it in Example 2.

The `model-fetch` stage has no input or output ports — it's a one-shot stage
that downloads a model and records it in the local model registry.

1. **Create** a pipeline, e.g. `fetch-qwen`.
2. Drag in one **Model Fetch** (`model-fetch`) stage. Leave its config empty
   to browse the catalogue interactively (or set `model_path` directly — see
   below).
3. Open the **User I/O** view.
4. **Start** the pipeline and answer the prompts:

   | Prompt | What to enter |
   | --- | --- |
   | `HuggingFace path (owner/repo or URL), or Enter to browse the catalogue:` | press **Enter** to browse |
   | `Model family` | the index next to **Qwen** |
   | `Version` | **3.5** |
   | `Parameter class` | **4B** |
   | `Variant` | **MLX 4-bit (mlx-community)** |
   | `Base download path [default ./models]` | press **Enter** for the default, or type a directory |

   The catalogue menus are **numbered** — type the **index** of the option
   whose label matches (the numbers depend on the catalogue and may shift, so
   match on the label, not a fixed number). A level with only one choice is
   auto-selected.

The stage downloads the files, registers the model under the key
**`mlx-community/Qwen3.5-4B-MLX-4bit`**, and finishes; the pipeline then
auto-stops. Models land under `<base_path>/<owner>/<repo>` (default
`./models/...`).

**Non-interactive shortcut.** Set the stage's `model_path` config to
`mlx-community/Qwen3.5-4B-MLX-4bit` and it fetches without prompting. Useful
keys: `base_path` (download root, default `./models`), `hf_token` (for
gated/private repos; falls back to `$HF_TOKEN`), `overwrite_existing`
(re-download if already registered, default `true`).

**As a spec.** The same fetch in spec form — here with the non-interactive
`model_path` set so it runs without prompts (an interactively-built fetch with
an empty config saves the same stage with an empty `config`):

```json
{
  "id": "fetch-qwen",
  "stages": [
    {
      "id": "fetch",
      "type": "model-fetch",
      "config": { "model_path": "mlx-community/Qwen3.5-4B-MLX-4bit" }
    }
  ]
}
```

### Fetch from the terminal (vpipe CLI)

Same fetch, no browser. `model-fetch` is a single-stage, no-port stage, so it's
a natural fit for the CLI's `--launch-stage`: `vpipe` wraps it in a one-shot
pipeline, runs it, and exits when the download finishes.

**Interactive — browse the catalogue:**

```sh
./build/apps/vpipe/vpipe --launch-stage model-fetch
```

With an empty config the stage browses the catalogue interactively — the same
numbered menus as the web UI, except the prompts appear **right in your
terminal** (the CLI routes the session's user I/O to stdin/stdout). Answer them
the same way: press **Enter** to browse, then pick **Qwen → 3.5 → 4B → MLX
4-bit**.

**Non-interactive — one command:** point `model_path` at the registry key and
it downloads without prompting:

```sh
./build/apps/vpipe/vpipe --launch-stage model-fetch \
  --stage-cfg model_path=mlx-community/Qwen3.5-4B-MLX-4bit
```

Each `--stage-cfg key=value` sets one config key on the stage; repeat it for
more (e.g. add `--stage-cfg base_path=./models`). Values that look like JSON are
read as such (numbers, `true`/`false`); anything else — like a HuggingFace path
— is taken verbatim as a string. The model registers under
**`mlx-community/Qwen3.5-4B-MLX-4bit`** exactly as the web-UI flow does, so
Example 2 can use it right away.

---

## 2. Interactive text chat
[back to top](#top)

**Goal:** chat with the Qwen3.5 4B model from Example 1, one turn at a time.
We build it in two stages: first the core, then make it turn-by-turn
interactive with a feedback pair.

### 2a. The core: text input → chat

1. **Create** a pipeline, e.g. `chat`.
2. Drag in a **Text Input** (`text-input`) stage. Config: `prompt` = `you> `,
   `count` = `0` (read until end of input).
3. Drag in a **Text Chat** (`text-chat`) stage. Config: `hf_dir` =
   `mlx-community/Qwen3.5-4B-MLX-4bit` (the key you fetched), `compute_dtype` =
   `f16`.
4. **Wire** `text-input` output `text` → `text-chat` input `user`.
5. Open **User I/O**, **Start**, and type a message at the `you>` prompt. The
   assistant's reply streams back into the User I/O console.

The text-input stage loops, reading a line and emitting it; text-chat replies
to each. Type `/clear` as a message to reset the conversation context.

### 2b. Make it turn-by-turn with a feedback pair

In 2a, text-input prompts for the next line on its own schedule. Add a
**feedback-rx / feedback-tx** pair so it waits for each reply before prompting
again — clean one-turn-at-a-time pacing. The pair is a one-beat-delay
"register" wired **by name**: `feedback-tx` relays whatever `feedback-rx` last
received, and the two are linked by the tx stage's `from` config (the rx
stage's id), **not** by a port wire.

Stop the pipeline, then:

1. Drag in a **feedback-rx** (`feedback-rx`) stage, id `rx`.
2. Drag in a **feedback-tx** (`feedback-tx`) stage, id `tx`. Config: `from` =
   `rx`.
3. **Wire** `text-chat` output `assistant` → `feedback-rx` input `in`.
4. **Wire** `feedback-tx` output `out` → `text-input` input `trigger`.
5. **Start** again.

Now each round is: you type a line → chat replies → the reply is relayed back
to text-input's `trigger`, which prompts you for the next turn. The first
prompt fires immediately (text-input's `present_first_without_beat` defaults to
`true`), so the loop self-starts without deadlocking.

**Saved spec.** Once wired up and saved, the pipeline looks like this. (The
stage order reflects the dependency rule the UI enforces when it serializes:
`tx` appears before `input` because `input` reads from `tx`; `rx` comes after
`chat`.)

```json
{
  "id": "chat-interactive",
  "stages": [
    { "id": "tx", "type": "feedback-tx", "config": { "from": "rx" } },
    {
      "id": "input",
      "type": "text-input",
      "iports": [ { "src": "tx", "oport": 0 } ],
      "config": { "prompt": "you> ", "count": 0 }
    },
    {
      "id": "chat",
      "type": "text-chat",
      "iports": [ { "src": "input", "oport": 0 } ],
      "config": {
        "hf_dir": "mlx-community/Qwen3.5-4B-MLX-4bit",
        "compute_dtype": "f16"
      }
    },
    {
      "id": "rx",
      "type": "feedback-rx",
      "iports": [ { "src": "chat", "oport": 0 } ]
    }
  ]
}
```

Useful `text-chat` knobs: `max_new_tokens` (per-turn budget, default `1024`),
`disable_thinking` (`true`/`false` for thinking-capable models), and the
`sampler_*` keys (`sampler_temperature`, `sampler_top_p`, `sampler_top_k`,
`sampler_seed`, …); all samplers at their defaults means greedy decoding.

---

## 3. Microphone transcription
[back to top](#top)

**Goal:** transcribe your microphone live with Qwen3-ASR. The chain is
**audio capture → audio to PCM → transcribe**.

### Microphone prerequisites (macOS)

- A **microphone device must exist** and be visible to AVFoundation. Capture
  goes through FFmpeg's `avfoundation` input, so a Homebrew FFmpeg
  (with `libavdevice`) must be installed (see the README).
- **Microphone permission** must be granted to the app that launched the web
  UI (your terminal, or the IDE/app running `vpipe-web-ui`): **System Settings
  → Privacy & Security → Microphone**. The web UI's startup checks report the
  current status; if permission is missing, capture logs a warning and keeps
  retrying rather than producing audio.
- To find device names/indices: `ffmpeg -f avfoundation -list_devices true -i ""`.

### Steps

1. **Fetch the ASR model** (Example 1's flow): browse **Qwen → 3-ASR → 0.6B →
   ASR MLX 8-bit (mlx-community)**, which registers
   `mlx-community/Qwen3-ASR-0.6B-8bit`. (Model Fetch also synthesizes the
   tokenizer this model needs.)
2. **Create** a pipeline, e.g. `mic-transcribe`.
3. Drag in **Audio Capture** (`audio-capture`). Set **exactly one** of
   `device_name` (a case-insensitive substring, e.g. `MacBook Pro Microphone`)
   or `device_id` (e.g. `0`).
4. Drag in **Audio → PCM** (`audio-to-pcm`). Default `output_sample_rate` is
   `16000`, which is what the ASR model expects — leave it.
5. Drag in **Transcribe** (`audio-transcribe`). Config: `hf_dir` =
   `mlx-community/Qwen3-ASR-0.6B-8bit`.
6. **Wire**: `audio-capture` `audio` → `audio-to-pcm` `audio`; then
   `audio-to-pcm` `pcm` → `audio-transcribe` `audio`.
7. **Start** and speak. Transcripts are surfaced to the session log / User I/O
   (the transcribe stage is a sink — it has no output port).

**Saved spec.** The pipeline you just built, as written by **Save**:

```json
{
  "id": "mic-transcribe",
  "stages": [
    {
      "id": "mic",
      "type": "audio-capture",
      "config": { "device_name": "MacBook Pro Microphone" }
    },
    {
      "id": "pcm",
      "type": "audio-to-pcm",
      "iports": [ { "src": "mic", "oport": 0 } ],
      "config": { "output_sample_rate": 16000 }
    },
    {
      "id": "asr",
      "type": "audio-transcribe",
      "iports": [ { "src": "pcm", "oport": 0 } ],
      "config": { "hf_dir": "mlx-community/Qwen3-ASR-0.6B-8bit" }
    }
  ]
}
```

By default the PCM stage emits ~10-second chunks, so transcripts appear a few
seconds apart. Example 4 makes this utterance-aware.

---

## 4. Better transcription with VAD segmentation
[back to top](#top)

**Goal:** improve transcription quality by feeding the transcriber clean,
utterance-bounded clips instead of fixed time chunks. We add an **Audio
Segment** (Silero VAD) stage that detects speech boundaries.

The segment stage doesn't sit *in series* in the audio path — it's a parallel
branch. The PCM stream fans out to **both** the segmenter and the transcriber;
the segmenter emits speech-interval markers that feed a **second input** on the
transcriber. Wiring that second input switches the transcriber into
**streaming mode**: it keeps a rolling PCM buffer and, on each marker, slices
out exactly `[start, end)` of speech and transcribes that utterance.

### Steps

1. **Fetch the VAD model** (Example 1's flow): browse **Silero → VAD v6 →
   unified → CoreML (vpipe-supplement)**. This is a packaged archive, so it
   registers under the key **`silero_vad_unified_v6`**.
2. Start from the Example 3 pipeline (stopped). Lower the PCM chunk size so
   markers arrive promptly: set `audio-to-pcm` `chunk_duration_s` to a small
   value, e.g. `0.5`.
3. Drag in **Audio Segment (VAD)** (`audio-segment`). Config: `model_path` =
   `silero_vad_unified_v6`.
4. **Wire** `audio-to-pcm` `pcm` → `audio-segment` `audio`. (The `pcm` output
   now feeds two stages — the segmenter and the transcriber; that fan-out is
   fine.)
5. **Wire** `audio-segment` `segments` → `audio-transcribe` `segments` (the
   transcriber's second input). This is what enables streaming mode.
6. **Start** and speak in phrases. Each detected utterance is transcribed as a
   unit.

Keep the sample rates consistent (all default to `16000`); mismatched-rate
beats are dropped. The transcriber's rolling buffer (`pcm_buffer_s`, default
`30`) comfortably exceeds the segmenter's max segment length (`max_segment_s`,
default `12`).

**Saved spec.** The finished pipeline (note `seg` is serialized before `asr`,
and both read from `pcm`):

```json
{
  "id": "mic-transcribe-vad",
  "stages": [
    {
      "id": "mic",
      "type": "audio-capture",
      "config": { "device_name": "MacBook Pro Microphone" }
    },
    {
      "id": "pcm",
      "type": "audio-to-pcm",
      "iports": [ { "src": "mic", "oport": 0 } ],
      "config": { "output_sample_rate": 16000, "chunk_duration_s": 0.5 }
    },
    {
      "id": "seg",
      "type": "audio-segment",
      "iports": [ { "src": "pcm", "oport": 0 } ],
      "config": { "model_path": "silero_vad_unified_v6", "sample_rate": 16000 }
    },
    {
      "id": "asr",
      "type": "audio-transcribe",
      "iports": [
        { "src": "pcm", "oport": 0 },
        { "src": "seg", "oport": 0 }
      ],
      "config": { "hf_dir": "mlx-community/Qwen3-ASR-0.6B-8bit" }
    }
  ]
}
```

The transcriber's `iports` has **two** entries: input 0 (`audio`) from the PCM
stage and input 1 (`segments`) from the segmenter. That second wire is the
switch that turns on utterance-streaming — without it the transcriber stays in
fixed-chunk mode.

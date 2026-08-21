# Model memory in VPIPE

A VPIPE graph routinely asks one machine to hold several large models at
once: a diffusion transformer, a text encoder, a VAE, sometimes a language
model beside them. On a 16 GB machine that does not fit, and on a 64 GB
machine it usually does — but the same pipeline description has to work on
both, and it has to work the *same way* on two runs of the same machine.

## Who this is for

Anyone adding a model, a loader, or a model-holding stage, **including
plug-ins built out of tree**. If your plug-in loads weights, allocates a
large working buffer, hands a large buffer downstream, or decides whether
to keep something resident, this document is the contract you are
implementing against.

The reason it is a contract rather than advice: memory decisions in a
graph are not independent. A stage choosing to keep a text encoder
resident is deciding, on behalf of a transformer it has never heard of,
whether that transformer can hold its weights. There is no arbiter that
can work this out for itself — it works only if every participant
declares what it intends to take. A plug-in that skips its half is not
merely unoptimised; it makes the *other* stages' decisions wrong, and the
symptom appears in them rather than in you.

Everything here is reachable from the public plug-in surface. Nothing in
this document requires modifying the host.

## Start here

The minimum for a plug-in that holds weights, in the order you will need
it:

1. **Take a `std::shared_ptr<WeightSet>`** in your loader and keep it for
   the model's lifetime. Never open a checkpoint yourself.
2. **Read `Copied`** unless your loader converts almost nothing
   (mechanism 1), and use the accessor that matches what you do with the
   tensor.
3. **Override `declare_memory()`** on your stage to say what you hold,
   what you hold *at least*, whether you let go of it, what you allocate
   while running, and how big your outputs are (mechanism 4). Name each
   holding by its checkpoint directory so the plan can tell one model
   held twice from two models.
4. **Override `declare_resources()`** as well — or
   `declare_resources(root)` on your model family — to name every
   checkpoint you are about to open (mechanism 2). The two ledgers run
   side by side today and answer different questions; see mechanism 4.
5. **If your model can arrive on an input port**, implement
   `apply_constant()`, or both declarations above will be empty in
   exactly the graphs that name their model once, in a source.
6. **Declare large working buffers and large outputs** (mechanism 3) if
   they are comparable to your weights. For a VAE decode they are usually
   much larger.

Steps 1–5 are mandatory. Everything after that applies when the model is
too big for the machine it lands on, and can be added later without
reworking the above.

If you are implementing a model **family** rather than a stage, read
"The plug-in surface" near the end first: it lists the methods these
mechanisms actually reach your code through, and what each default
silently asserts about your model.

## The two rules

> **1. Nothing goes to swap.** That is the single condition the whole
> subsystem is built to hold. A model that runs 3× slower because it
> streams is working; a model that pushes the box into the compressor and
> swap makes no progress at all and takes the rest of the machine with
> it. Every trade below resolves in favour of not swapping — which is
> also why the *wired* pool exists (mechanism 7): wired pages cannot be
> compressed or swapped, so what is in the pool is what this process
> genuinely keeps.

> **2. Measure, do not predict.** Free memory moves underfoot, and
> arithmetic over it is self-fulfilling on a streaming model: a model
> that has evicted its own weights sees free RAM and concludes it has
> room. Where a decision can be re-checked against what actually
> happened, re-check it — and prefer a number the machine reported to one
> you derived.

A corollary of the second worth stating separately, because it is the
failure mode this document exists to prevent: **an under-estimate is not
a conservative error.** Declaring less than you will hold reads to every
peer as room that is not there, and the peer that believed you may
already have taken a decision it cannot reverse. Where you cannot compute
a figure, say so (there are mechanisms below for exactly that) rather
than supplying one that is merely small.

---

## The mechanisms

| # | Mechanism | Question it answers | Granularity | When decided |
|---|-----------|--------------------|-------------|--------------|
| 1 | Tensor residency | How do these bytes get into RAM? | per tensor | in the loader |
| 2 | Declarations and phases | How much will everyone else hold, and when? | per checkpoint | before any model loads |
| 3 | Scratch and payloads | How much will everyone else *allocate*, and hand on? | per buffer | declared before, corrected per beat |
| 4 | The memory plan | What does this whole graph cost, in running order? | per stage | derived from the topology, revised at run time |
| 5 | Block streaming | Do I hold this checkpoint at all? | per model | at construction, irreversible |
| 6 | Block residency | May I keep *this* block resident? | per block | re-measured every forward |
| 7 | The wired pool | Which of my bytes may the OS never take? | per buffer | at allocation |
| 8 | Parking | May somebody else have my idle bytes? | per weight set | at an idle point |
| 9 | The removable pool | May the *next launch* have what I finished with? | per weight set | at release |
| 10 | The memory cap | What does the process insist on holding? | process-wide | from config |

Mechanisms 1–4 are mandatory for every model. 5–7 apply to models too big
for the machine. 8–10 are how idle weights stop being dead weight.

They are layers, not independent knobs: each answers a different question,
and a model that wires one and skips another tends to behave correctly on
the machine it was written on and nowhere else.

---

## 1. Tensor residency — `Copied` vs `Mapped`

Every loader takes a `std::shared_ptr<WeightSet>` and reads tensors through
it. Two residencies exist:

- **`Copied`** — allocate a `SharedBuffer` and fill it from the file with
  an **uncached `pread`**. Anonymous memory: reclaimable only through the
  compressor and swap, wirable into the pool, parkable.
- **`Mapped`** — a zero-copy view into the mapped shard. Clean file-backed
  pages, which the kernel can drop for free and re-fault on demand. *Not*
  parkable and *not* wirable (there is nothing to hand over that the OS
  does not already manage better).

`Mapped` sounds strictly better and usually is not. Mapping one tensor
wraps the **whole shard**, so touching a single tensor makes the entire
checkpoint resident. That only pays for conversion-light loaders. A loader
that allocates a converted copy on any dtype mismatch pays both bills:
measured on one language model, switching its reads to `Mapped` took peak
footprint from 3.62 GB to 5.00 GB (+38%).

**The tree is moving away from mapping, and a new loader should not
reach for it.** Two reasons that have nothing to do with the copy. The
first is that asking the OS to manage 4 KB pages over a 10+ GB
checkpoint costs more than owning the bytes does — the page table and
the fault traffic scale with the file, not with what the model touches.
The second is the one this document is otherwise about: a mapped weight
can be neither wired, parked nor pooled, so it sits outside every
mechanism below except the accounting. On a box with the wired pool on,
that is the difference between weights the process keeps and weights
the compressor may take.

> Read that number with the alignment trap below in mind. `Mapped` on a
> shard the file cannot satisfy is the overhead with none of the benefit,
> and most published checkpoints cannot satisfy it — so a measurement
> that did not first confirm the file was mappable measured the cost
> alone. On a checkpoint that *is* mappable the sign can flip: measured
> on a 4-bit language model used as a text encoder, whose dominant bytes
> are packed codes read as they sit with nothing converted beside them,
> the same switch took the load's physical footprint from 10.3 GB to
> **7.3 GB**, with identical output.

**Default to `Copied`,** and treat it as a statement about *conversion*
rather than about model kind. Language models, vision and audio towers
read `Copied` because they narrow dtypes on the way in; what they gain
from the weight set is ownership, dedup and a single open, not reduced
residency. A loader that keeps its bytes exactly as they sit on disk —
quantized codes, or a checkpoint already in the compute dtype — is the
case worth measuring both ways.

Which accessor to use:

| You are… | Use | Cached? |
|---|---|---|
| keeping the tensor as-is | `tensor(name, mc, res, part)` | yes, parkable |
| keeping a *transform* of it | `derived(key, build, part)` | yes, not parkable |
| consuming it and dropping it | `read(name, mc, res)` | no |
| re-reading it per forward | `stream_tensor` / `stream_derived` | no, counted |
| re-reading it into a buffer you already own | `stream_into` | no, counted |

The rule is **cache what the model keeps; read uncached what it consumes.**
Pieces row-concatenated into a fused matrix, bytes copied into a host
vector, and streamed blocks are all consumed — caching them keeps a
redundant copy alive next to the product.

Weights obtained from a weight set are **shared and immutable**. Never
write through one; transform into a new buffer. `VPIPE_WEIGHT_INTEGRITY=1`
hashes `Copied` entries at load and reports any that changed.

> **Trap.** The accessor you choose decides whether mechanism 8 can ever
> see your weights. See "Parking cannot reach uncached weights" below.

### Reads do not grow the file cache — as long as the file is aligned

A `Copied` read is not a memcpy out of the mapping. It is a `pread` with
`F_NOCACHE` set, into the buffer the model will keep. Two reasons, and the
second is the one that decides whether a small box survives:

- It is **faster** where the disk allows. MEASURED over a 206 MB
  transformer block, arms interleaved and their order rotated: a fresh
  allocation plus a memcpy out of the mapping runs at 0.86–1.48 GB/s, and
  a `pread` into a buffer that already exists at 6.7–6.9 GB/s. The spread
  matters as much as the mean — the mapped rate moves by 2× between
  rounds where `pread`'s does not.
- It leaves **no file cache behind**. For a model that re-reads its whole
  checkpoint every forward, that cache is pure pressure, and it is
  exactly the pressure that squeezes out the blocks a resident set is
  trying to keep.

> **Trap: `F_NOCACHE` only bypasses the cache on a PAGE-ALIGNED file
> offset, and a read that misses the alignment does not fail — it
> silently falls back to the buffered path.** MEASURED with a clean
> control, 4096 MB read from one file with `F_NOCACHE` set throughout:
>
> | file offset | file-backed pages gained |
> |---|---|
> | page-aligned | +0 MB |
> | + 8 bytes | **+1813 MB** |
> | + 4096 bytes | +0 MB |
>
> A safetensors data section starts at `8 + header_len`, so its alignment
> is whatever the header length happened to be, and every tensor in the
> shard inherits it. MEASURED through the real loader on a misaligned
> checkpoint: 6233 MB read grew the file cache by **6212 MB** — one for
> one, on a path that sets `F_NOCACHE` on every read.
>
> The read now stages through page boundaries and copies out, so the
> same 6233 MB grows the cache by **1 MB**. You do not have to do
> anything to get this; it is in the accessor. What you *do* have to do
> is not build a second read path that skips it.

### Alignment, and the two costs of a misaligned pack

The same misalignment has a second, older consequence: a Metal buffer
offset must be **16-byte aligned**, so `Mapped` is a *request* the file
can refuse. Two things have to hold for a tensor to get a zero-copy view,
and they fail independently:

1. **The data section starts aligned.** Get `8 + header_len` wrong and
   *every* tensor in the shard falls back to a copy. MEASURED on a 22B
   DiT whose section began at 677624 (≡ 8 mod 16): **4349 of 4349
   tensors, 39.1 GB, copied** — anonymous memory, where the model asked
   for file-backed pages.
2. **Each tensor's SIZE is a multiple of 16.** Tensors are packed
   contiguously, so one that is not shifts every tensor after it off the
   boundary — and the offenders are small and easy to miss. MEASURED on
   a quantized 12B: one 2-byte per-layer scalar per block, 48 of them
   interleaved with the weights, plus four JSON asset blobs — 53 tensors
   that between them cost **1181 of 1344**, with the section start a
   clean 0 mod 16 the whole time.

**Reported once per checkpoint, at `open()`, as a warning** — not only
where it bites. The models that suffer most are the ones reading
`Copied`, which never reach the mapping code at all, so a misaligned
pack cost them a silently buffered read with nothing in the log
connecting the cost to the cause.

That warning names the **read** cost and not the mapping one, on
purpose. The mapping penalty is real and is reported separately by
`load_mapped()`, where a caller actually asked for it — but leading with
it would point an operator at a repair for the path this tree is
leaving. Every loader pays the read.

If the warning names a checkpoint **you** produced with `model-quantize`
or `lora-fuse`, re-run it: the writer pads the data section to 16 **and**
writes odd-sized tensors last, so anything it produces today is aligned,
and re-running lays the same weights down at no cost but the time.
Ordering rather than padding, because the byte ranges have to keep tiling
the blob — the reference safetensors reader validates that they are
contiguous, so a writer that bought alignment with gaps would produce
files nothing else can read.

Assume a downloaded checkpoint is not aligned until the log says
otherwise. Many land on `%16 == 8`.

---

## 2. Declarations and phases — sizing before anything loads

Every driver runs `initialize()` concurrently, and that is where
model-holding stages both size the box and load. Without declarations a
stage sees whichever peers happened to have loaded first, the same graph
decides differently run to run, and a model held by an unrelated stage is
invisible.

So: **every stage that loads a model overrides `declare_resources()`.**

```cpp
std::vector<ResourceClaim>
MyStage::declare_resources() const
{
  return model_memory::weight_claims({_model_dir, _encoder_dir});
}
```

`PipelineRuntime` collects all claims before starting any driver. A claim
is an opaque `{kind, key}` pair routed to a registered `ResourcePlanner`;
model weights are one kind, activation scratch another. A claim whose
kind has no planner is warned about, never dropped silently.

A component that **can be reduced** rather than dropped says so with a
floor:

```cpp
return { model_memory::weight_claim_streamable(_dit_dir, floor_bytes) };
```

The floor is what a block-streaming DiT holds when it streams — its trunk
plus the in-flight slots, not the checkpoint. The planner reports both
columns (`N MB preloaded / M MB if every streamable component streams`),
and a floor above the checkpoint's own size is clamped. Declaring one
matters more than it looks: the refusal check in mechanism 7 runs against
the floor column, so a streamable component that declares none is counted
at full size and reads as a graph that cannot run.

**If the model arrives on an iport, say so with `constant_output()`.** A
graph commonly names its model once, in a `model-select` source, and wires
that one beat into every model-holding stage — so those stages have an
empty `hf_dir` when they are asked to declare, and the guard every one of
them opens with (`if (_hf_dir.empty()) { return {}; }`) makes the whole
phase declare nothing. The fix is on the *source*: a stage whose output is
fixed by configuration returns it from `Stage::constant_output()`, and the
runtime delivers it to each consumer's `Stage::apply_constant()` before
planning begins. Consumers latch it there exactly as they latch the beat
at run time — the source still emits normally, since folding is an
analysis and not a rewrite.

Both sides are needed. A consumer that takes its model from an iport and
does not implement `apply_constant()` declares nothing, and is invisible
to every peer sizing the box against it.

### Phases — weights that never coexist

A generation graph runs in stages that do not overlap: a text encoder
produces one conditioning and is done, *then* a DiT denoises for minutes,
*then* the VAEs turn the latent into audio and pixels. Summing all of them
sizes the box for a moment that never happens.

A claim may therefore name the phase it belongs to:

```cpp
return model_memory::weight_claims_in_phase(
    {_encoder_dir}, model_memory::kPhaseCondition);
```

The vocabulary is fixed **and ordered**, and the order is the order the
phases run in:

```
kPhaseCondition → kPhaseDenoise → kPhaseDecodeAudio → kPhaseDecode
```

An unrecognised phase is warned about and counted as resident for the
whole run, because a typo that became its own phase would be counted
apart from the claims it really coexists with, and that *under*-counts.

**A claim spans an INTERVAL of phases, not one.** This is the part that
changed, and it changed because a single phase per claim can only say
"resident while my own stage runs" — while the term that actually decides
a constrained box is usually something *produced* by one phase and
*consumed* by a later one. The output latent is alive from the denoise
that writes it until the last decode that reads it. The conditioning is
alive from the encoder that makes it until the denoise that consumes it.
Neither belongs to one phase, and counting them as persistent (the old
answer for anything not single-phase) puts them in every phase including
the ones they are absent from.

So what a memory-constrained box must hold is

```
max over phases p of ( sum of every claim alive during p )
```

which for a generation graph reads as four terms, no two of which are
ever live together:

```
condition      encoder floor + encoder scratch + conditioning
denoise        conditioning  + DiT floor       + DiT scratch  + latent
decode-audio   latent        + audio VAE floor + its scratch  + PCM
decode         latent        + video VAE floor + its scratch  + frames
```

`GenerativeModelManager::phase_peak()` computes exactly that, and fills a
per-phase breakdown so a report can name *which* moment is the tight one.
The two decodes are separate phases because on a constrained box they do
not overlap either — an audio VAE and a video VAE are loaded and dropped
independently, and summing them sizes a moment that does not happen. Both
read the same latent, which is why the latent's interval has to *reach*
both rather than belong to one.

MEASURED on a bf16 MiniMax-H3 graph — the difference between an
accounting that summed everything and one that walks intervals in order:

```
declared, summed  117873 MB
phase peak           5527 MB   (condition 4479, denoise 3116,
                                decode-audio 585, decode 5527)
```

On an **abundant** box the same graph may keep all of it resident at once
so a second launch pays no reload, which is the sum rather than the max.
Both are reported: the max is what decides whether a graph can run, the
sum is what it costs to make relaunches free.

### Declare, then decide — two passes

`declare_resources()` describes and must not read the manager;
`decide_resources()` decides, and runs only after every stage has
declared:

```cpp
std::vector<ResourceClaim>
MyStage::declare_resources() const     // describe: both, unconditionally
{
  return model_memory::weight_claims({_encoder_dir, _dit_dir});
}

std::vector<ResourceClaim>
MyStage::decide_resources() const      // decide: is the encoder phased?
{
  if (!will_certainly_release()) { return {}; }
  return model_memory::weight_claims_in_phase(
      {_encoder_dir}, model_memory::kPhaseCondition);
}
```

The split is what makes the answer independent of visiting order. In one
pass, a stage that sizes the box sees only the peers declared before it —
the first sees an empty graph and the last sees all of it, so the same
graph decides differently depending on where the flattener emitted each
stage. That is the race the planning phase exists to remove, reintroduced
inside it. Pass 1 writes and does not read; pass 2 reads and does not
write. Refinements are buffered and applied together at the end, so one
stage's decision can never move the box another stage is deciding against.

**Two conditions on a phase claim, and neither is checkable by the
planner.**

1. **The release must be certain from config, before anything loads.**
   `unload_when_idle: destroy` qualifies. **`park` does not** —
   `park_weights()` walks a weight set's *cached* entries, and every text
   encoder reads uncached into the model's own members, so a parked
   encoder parks 0 bytes and is still entirely resident. A peer that
   subtracted it is short by its whole size.
2. **The release decision must not read a phased figure.** `auto` resolves
   against `bounded()`, which sums every phase on purpose. Reading the
   phased number there would let the claim justify itself: the encoder is
   subtracted, the box looks roomy, `auto` resolves to *keep*, and the
   subtraction has already been handed to a DiT that cannot take it back.

`resident_weight_bytes()` stays the unphased, no-release worst case for
exactly that reason. `phase_footprint(p)` is persistent + phase *p*;
`phase_footprint({})` is persistent + the widest single phase;
`phase_peak()` is the interval-aware peak above, and is what the refusal
check uses.

Report the release with `note_phase_released(dir)` when the weights
actually go. A phase claim whose release never arrives is warned about at
the end of the run — otherwise the promise is unfalsifiable, and a broken
one shows up only as thrash with nothing in the log connecting it to the
claim that caused it.

**A phase says "not at the same time", which in a concurrent pipeline is
a statement about the graph, not just the stage.** These claims are sound
for a run whose beats do not overlap — one prompt in, one video out,
which is how the generation pipelines are written. A graph that pipelined
beats would have the encoder loading beat *N+1* while the DiT denoises
beat *N*, and the two phases would then be resident together. The
end-of-run audit is what catches that after the fact.

Declarations **persist for the run** and count at `max(held, estimate)`.
Do not clear them once loading is done: a weight set only accounts for
what it *cached*, so an uncached-reading model reports a fraction of its
real bytes, and clearing erases it exactly when peers start sizing against
it. A model that genuinely keeps less calls `revise_declaration()`.

Sizing helpers (`stages/model-memory.h`):

- `weight_footprint(session, dirs, phase)` — a **union** over checkpoints,
  not a sum over directory names. A shared checkpoint loads once; an open
  one contributes what it *holds*.
- `streaming_floor_bytes(dir, stems, exclude)` — the floor above, computed
  from the checkpoint on disk: everything outside the repeating unit, plus
  two in-flight slots.
- `plan_streaming(session, dit_dir, enc_dir, headroom, dit_retires)` — the
  single streaming rule for every DiT family. Do not copy-paste another.
- `kHeadroom` (6 GB) for **revisable** decisions; `kStreamHeadroom` (8 GB)
  for **irreversible** ones. Failing to stream means thrash or an OOM
  kill; streaming needlessly costs ~2–3× per step, so the decision that
  cannot be taken back gets the bigger cushion.

These compare against **total RAM**, not free memory, on purpose: every
stage must answer identically and reproducibly. `VPIPE_RAM_LIMIT_MB`
simulates a small box on a big one.

---

## 3. Scratch and payloads — the other things that have to fit

Weights are what a model *holds*; scratch is what it *allocates to run*;
a payload is what it *hands downstream*. For a VAE decode the second
dwarfs the first and the third dwarfs both: FLUX.2's VAE weighs 160 MB on
disk and its decode peaks near 2.8 GB at 1024², 14 GB at 2048². A graph
accounted purely by weights reads as roomy right up to the allocation
that does not fit.

```cpp
const std::size_t arena =
    model_memory::vae_decode_scratch_bytes(root, _width, _height);
return model_memory::scratch_claims("vae-decode", arena,
                                    model_memory::kPhaseDecode);
```

`vae_decode_scratch_bytes()` estimates from `<root>/vae/config.json`
alone — no model load, so it is answerable during planning. The formula is
the VAE's own, chosen by `_class_name`, mirroring that class's
`decode_peak_bytes()`. An **unrecognised** VAE gets the larger of the known
estimates: under-declaring an arena reads as room that is not there, and
the stage that believed it has already decided something it cannot undo.
A missing config returns 0, and consumers fall back to their old behaviour
rather than to a number nobody computed.

`video_decode_scratch_bytes(w, h, frames)` is the video form, where the
dominant transient is not the convolution arena but the **output**:
`[3, frames, H, W]` at bf16 plus the planar-U8 clip the stage buffers
behind it — 9 bytes per output pixel, growing linearly with length.

**A payload is declared separately, because it outlives its producer.**

```cpp
return model_memory::payload_claims("video-latent", bytes,
                                    model_memory::kPhaseDenoise,
                                    model_memory::kPhaseDecode);
```

Same machinery as an arena — a label, a size, a lifetime — and a separate
name because the two are different facts and a reader sizing a graph needs
to tell them apart. A scratch arena is torn down when its stage finishes;
a payload is handed on, so it is alive from the phase that writes it
through the last phase that reads it. These are the terms a
weights-and-scratch accounting cannot see, and on a constrained box they
are often the largest thing in the moment they exist.

**Declared by whoever has the geometry, not by whoever allocates.**
`vae-decode` sizes itself from whatever latent arrives, so at planning time
it cannot name a number — which is why it had been using peer *weights* as
a proxy and `kHeadroom` as a stand-in for the arena. `generate-image` and
`generate-video` have the geometry in config, so they declare. A graph
whose geometry comes from an input image (an edit pipeline) declares the
marker below, honestly.

**An arena's size is a function of the BEAT, so the plan can only bound
it.** A video decode's transient scales with the *pixel* frame count,
which is the VAE's expansion of the latent — a property of the loaded
model, not of config. MEASURED on MiniMax-H3 at 256²: config says 9
frames, the VAE decodes 22, and an unrounded 5 MB bound is 2.4× short of
the 12 MB truth.

An image **edit** graph is the harder half: its output geometry comes from
the reference image, so there is no `height`/`width` in any config to
estimate from at all.

The plan still declares — `model_memory::kUnknownArena`, a 4 KB presence
marker. That is what keeps the plan authoritative about **what** exists
while runtime supplies **how much**, which is the contract weights already
have (`declare_weights` / `revise_declaration`, where revise also refuses
to create). Without a marker a stage would have to introduce an arena the
plan never saw, and then nothing distinguishes a legitimate late truth
from a mistyped label. The marker is negligible by construction — it
cannot move a sizing decision, and the first beat replaces it outright
before the decision it feeds is acted on.

`generate-image` revises from the same expression its DiT-reclaim check
uses, so the ledger and the decision cannot drift; `vae-decode` revises
from the geometry of the latent in hand:

```
resource-plan:  5 MB of activation scratch declared
VaeDecodeStage: decode arena 0 MB -> 12 MB (this beat's real geometry;
                the plan could only bound it)
```

MEASURED on an image-edit graph, which has no config geometry: the plan
declares one arena at the marker (`1 activation arena(s) declared, 0 MB`)
and the runtime figure replaces it with **224 MB**. Without a declared
arena the stage falls back to a flat `kHeadroom` of 6144 MB — a 27×
over-estimate on that graph, and wrong in the other direction at 2K.

Both the **ledger** and the **policy** follow the beat in both
directions. `model_memory::resolve_idle_unload()` is the rule, re-asked
per beat.

Loosening matters as much as tightening, because this mechanism is shared
between image and video: a run of image edits at mixed sizes has one large
frame and several small ones, and a rule that could only tighten makes
every small frame after the large one pay a reload it did not need.
Nothing about the decision is on the critical path — it is taken after a
decode completes and decides only whether to hold the weights until the
next beat.

What it does carry is a **band**: unload when the arena genuinely does not
fit, keep when it fits with room to spare (`arena/8`), and hold the
current answer in between. A flip is a real reload — MiniMax-H3's video
VAE is 10.4 GB — so an arena sitting on the threshold must not flip every
beat. The band is proportional rather than a constant, so it scales with
whatever is being decoded.

Both reset per launch. A label that was never declared cannot be revised
into existence, so a stage cannot invent an arena the plan never accounted
for.

Coverage, and what each family's plan-time bound is actually worth:

| family | declares from | revised by | bound vs truth |
|---|---|---|---|
| FLUX.2 | `AutoencoderKLFlux2`, `block_out` ×14 | generate-image, vae-decode | exact |
| Boogu | `AutoencoderKL`, `block_out` ×14 | generate-image, vae-decode | exact |
| Krea-2 | `AutoencoderKLQwenImage`, `base_dim` ×27 | generate-image, vae-decode | exact |
| Qwen-Image-Edit | `AutoencoderKLQwenImage`, `base_dim` ×27 | vae-decode | exact |
| LTX-2.5 | rounded config geometry | vae-decode (family path) | exact |
| MiniMax-H3 | rounded config geometry | vae-decode (video path) | exact |
| wan | rounded config geometry | vae-decode (video path) | exact |

> **The Qwen-Image VAE spells its width `base_dim` and ships no
> `block_out_channels` at all** — every quantized Krea-2 and
> Qwen-Image-Edit pack has the key absent, so an estimator reading
> `block_out` fell to the 128 default against a real 96 and
> over-declared 1.33×. Two families, one config key apart. MEASURED on
> `Qwen-Image-Edit-2511-4bit` at 1024×1024: 2592 MB, which is
> `1024*1024*96*27` exactly.

**Size the arena from the geometry the model will PRODUCE, not the one
config asked for.** Every video model constrains its latent shape and
rounds up, so an estimate over the requested numbers describes a clip
nobody will make — and it under-counts, which reads as room that is not
there. MEASURED on MiniMax-H3: config 9 frames becomes 22 (the VAE takes
17-frame chunks keeping 5 latents each), and the unrounded bound was 5 MB
against a 12 MB truth. Rounded, the bound is 12 MB — exact.

`GenerateVideoStage::planned_geometry_()` applies the same rules
`resolve_config_` will: a registered family answers `align_frames()` and
`size_grid()` without loading anything.

> **Detect by PROBING the checkpoint, not by asking the model record.**
> The record commonly says nothing about the family — a `model_type` of
> `-` is normal, and the stage then probes the checkpoint to find out
> what it is holding. A family test keyed on `model_type` therefore never
> fires, the rounding never happens, and nothing says so: the code
> compiles, runs, and leaves the bound at its unrounded value. This is
> the most easily missed step on this page, because every symptom of
> getting it wrong appears in another stage.
>
> The probe is `config_from_json`, which reads a diffusers `config.json`
> **or** a Comfy-Org repack's safetensors `__metadata__` envelope — so
> both shipped weight sources are covered by one call, and a repack needs
> no separate arm.

**A source with no recognised rounding rule declares the marker, not an
unrounded estimate.** A GGUF DiT carries neither a `config.json` nor a
metadata envelope, so every probe fails — and falling through with the
config numbers would produce not a conservative bound but an optimistic
one, which is the failure this whole section exists to fix, returning
silently. `planned_geometry_()` returns false instead, the stage declares
`kUnknownArena`, and says so:

```
GenerateVideoStage: no rounding rule for '<dir>', so the decode arena is
declared as present-but-unsized and corrected on the first clip
```

MEASURED, FLUX.2 klein-9B-4bit at a simulated 16 GB (peers 10059 MB):

| geometry | arena | old (flat 6 GB headroom) | declared |
|---|---|---|---|
| 1024² | 3584 MB | keep | keep |
| 2048² | 14336 MB | **keep** — wrong | **unload** |

---

## 4. The memory plan — derived from the graph, not from phase names

Mechanisms 2 and 3 ask every stage to name the phase it belongs to, from
a fixed vocabulary, and take the maximum across those names. That works
for the graph shape the vocabulary was written for and degrades quietly
everywhere else: two DiTs both name `denoise` and are **summed**, a graph
that is not conditioner-DiT-VAE has no phase to name so everything is
counted as persistent, and a decode that feeds another denoise cannot say
so because the running order is a constant list.

The deeper problem is that a phase is **asserted** by a stage about a
global property it cannot see. A stage does not know whether it is the
only DiT or the second of three, and `weight_claims_in_phase` documents
its own preconditions as ones the planner cannot check.
`note_phase_released` exists because of that: an audit, one launch late,
for a promise nothing can verify up front.

So the plan derives the same answer from the **topology** instead. The
runtime already builds the post-inlining logical edge list for
clock-domain analysis; ordering the stages by it gives every claim a
position, and a buffer's lifetime falls out of *who consumes it* rather
than being named. Two DiTs land at different positions and are maxed. A
graph nobody anticipated works, because nothing had to anticipate it.

```cpp
StageMemory
MyStage::declare_memory() const
{
  StageMemory m;
  // NAMED by its directory, so two stages over one checkpoint are one
  // set of weights in the plan rather than two.
  m.hold(_dit_dir,
         model_memory::dir_weights_bytes(_dit_dir),   // preload
         MyTransformer::streaming_floor_bytes(_dit_dir),  // floor
         _unload_cfg == UnloadPolicy::kDestroy,       // releases
         _unload_cfg == UnloadPolicy::kAuto);         // reclaimable
  m.scratch = arena_bytes_();
  m.outputs = { latent_bytes_() };                    // by oport
  return m;
}
```

Four things a stage says, and one it does not:

- **`preload`** — what this holding costs when the box has room.
- **`floor`** — the least it can be held at while still being held: a
  streaming DiT's trunk plus its slots. Zero reads as `preload`.
- **`releases`** — gone when this stage finishes, rather than held for the
  run. A **lifetime that ends**.
- **`reclaimable`** — can be given back entirely under pressure and
  reloaded (or found still there) when next used, so its floor is zero
  however large it is. This is the `auto` policy said in the plan's own
  terms, and it is *not* the same as `releases`: those bytes are still
  wanted, they are merely surrenderable.
- It does **not** say when any of that is live relative to anyone else.
  The runtime works that out from the edges — which is precisely the part
  a stage cannot know.

**`source` is what makes the arithmetic right when a graph holds a
checkpoint twice.** Two `generate-image` stages over one model are two
stages and **one** set of weights; counting the bytes per stage bills the
graph for a model it loaded once. The plan cannot know that from the bytes
alone — they are just numbers — so the stage names what it holds and the
plan merges by that name, taking the larger size and the union of the two
lifetimes. Empty means "unique to this stage", never merged: the right
answer for anything a stage allocates for itself, and the safe one when a
stage cannot identify what it holds, since over-counting refuses a run
rather than thrashing one.

**Position is not time, and the model does not pretend otherwise.** Stages
in a pipeline are coroutines that overlap; a topological position is a
data dependency, not a schedule. What makes the arithmetic right anyway is
that overlap is expressed through `releases`: a stage that runs the whole
time does not release, so it is counted at every position. A generation
graph is a chain and its positions read as time; a continuous capture
graph has everything live at once and every stage says so.

The plan reports two columns and names the position each peak occurs at:

```
memory-plan (declared): peak 5527 MB at 'vae-decode' streaming,
  11841 MB at 'generate-video' preloaded, over 7 stages in running order
  (diffusion-conditioner 4479 MB, generate-video 3116 MB, ...)
```

`peak_floor` is what the box must hold for the graph to run at all;
`peak_preload` is what it costs to keep everything resident so a second
launch pays no reload.

### Revising, once a stage knows

The plan is a **snapshot** taken before anything loads, from numbers a
stage can know from its configuration. Several of the largest terms are
not knowable then — an activation arena sized by the first beat's
geometry, a streaming model's real resident set — so the snapshot has to
be correctable, or its unknowns are permanent.

```cpp
StageMemory m = declare_memory();
for (StageHolding& h : m.holdings) {
  if (h.source != dit_dir) { continue; }
  h.preload = held;                 // what streaming actually kept
  h.floor   = held;
}
revise_memory(m);                   // safe from process()
```

Three properties of the revise channel:

- **It goes to the runtime, not to your stage.** What a revision has to
  reach is the *other* stages' numbers, since the peak is a property of
  the graph. A stage that could only correct its own entry would be
  writing to something nothing reads.
- **Report the whole `StageMemory`, not a delta.** A revision adjusting
  one field would have to be merged against a snapshot the caller cannot
  see, and two revisions racing would interleave into a figure neither
  stage stated.
- **It is reported when the peak moves**, in megabytes — so a stage that
  corrects itself without changing the number the box has to survive
  produces no line, and a graph whose stages each correct themselves once
  does not produce a line per stage saying nothing changed.

A revision that arrives after the plan closes is warned about and dropped;
a revision naming a stage that is not in this plan likewise. A stage
constructed outside a launch — which every unit test does — has no sink
and simply drops the revision.

### Which ledger do I use?

Both, today. They answer different questions and they are computed
differently:

| | Phase claims (2, 3) | The plan (4) |
|---|---|---|
| lifetime comes from | a name the stage asserts | the graph's edges |
| holds one checkpoint twice | deduped by directory | merged by `source` |
| what it feeds | the pool refusal check, `plan_streaming` | the reported per-stage peak |
| corrected at run time by | `revise_declaration`, `revise_scratch` | `revise_memory` |

A stage that implements only one of them is visible to half the
accounting. Implement both; they are a few lines each, and the numbers
they produce should agree — when they do not, that disagreement is the
most useful diagnostic on the page.

---

## 5. Block streaming — the irreversible one

A model that cannot fit reads its blocks from disk per forward, keeping
only a pinned prefix. This is an argument to the transformer's `load()`,
so it must be right at construction — changing it later means destroying
and rebuilding the model.

Take the decision from `plan_streaming()`, never by hand. It also records
the fact that streaming was chosen, which peers depend on (below).

`dit_retires` is what the model will **release** after load and before the
denoise — weights it reads once and then drops, so they are never part of
the set that has to coexist. An AdaLN bake is the case it exists for: it
retires every `adaln_proj` projection, 24.3 GB of a 61.7 GB bf16
checkpoint, and without this the irreversible decision is taken against a
model 39% of which is about to stop existing. Pass it **only** when the
release is certain; zero is always safe.

A streaming model must then call `revise_declaration()` down to what it
actually pins, so peers do not size against weights that are never there.

> **Trap: the revision erases the evidence.** A streaming model correctly
> reports a small number, and that small number reads as *roominess* to
> whoever sizes next. MEASURED on a 16 GB box: a DiT concluded "11 GB +
> 6 GB vs 16 GB RAM → stream blocks", revised itself down to 3.3 GB, and
> seconds later the text-encoder stage sized against that 3.3 GB, decided
> there was room, and kept a 1.2 GB encoder resident. The run then spent
> its whole denoise at 9.3 GB compressed with 3.8 GB of swap. Teaching
> the encoder to see the streaming peer took it to 5.5 GB compressed /
> 0.7 GB swap, and the denoise from 66.4 s to 60.2 s.
>
> That is what `model_memory::peer_streams(session)` is for: it reports
> whether *anyone* in this run decided to stream — i.e. whether somebody
> is already paying per-step disk reads for want of RAM. Weigh it before
> keeping anything resident. It is a record of a decision that was taken,
> not a prediction.

`StreamPlan::pin_frac` is **language-model layer pinning only.** Every DiT
that used it is retired: the block-streaming families grow a resident set
by measuring instead (mechanism 6), which is what a fraction of total RAM
decided before the run could never do. The LMs keep it because they have
nothing to grow into — they pin a prefix of layers with no residency
policy behind it, so removing the fraction would leave the text encoders
streaming everything with no way back. Give them a `BlockResidency` and
the field goes too.

### Reading a streamed block

Having decided to stream, *how* the bytes arrive is the rest of the cost.
The obvious shape — allocate a destination per tensor, fill it from the
mapping — pays twice: for the allocation, and for demand-faulting the
shard's mapping a cluster at a time. The rates are in mechanism 1:
0.86–1.48 GB/s against 6.7–6.9 GB/s.

So keep the destinations. Two blocks' worth, alternating, refilled in
place — which is also what makes a prefetch free, since the buffer the
reader fills is already allocated and no longer a growth question.
`generative-models/shared/streamed-refill.h` is the read:

```cpp
switch (refill_streamed_tensor(ws, name, dst, RefillDst::kRaw)) {
  case Refill::kFilled:     break;                  // ready
  case Refill::kUnservable: build_it_the_old_way(); break;
  case Refill::kFailed:     return false;           // dst is unusable
}
```

**`kUnservable` is about one tensor, not the block.** It means "build this
one however you did before" — not "give up" — and the decision is
therefore **per tensor**, not sticky for the block. Blocks are rarely
uniform: a released video DiT's block is 140 tensors of which 78 are
servable raw (387 MB) and 62 are not (24 MB of f16 scales and f32
modulation tables). A rule that abandoned the block for the awkward 24 MB
would buy nothing, and one that latched on the first refusal would fall
back for the whole 411 MB. `kFailed` is the separate case where the
checkpoint and the destination disagree and the buffer is now partly
written.

**Ask `refill_serves()` before allocating**, if you still allocate per
tensor. Otherwise the tensors that fall back each cost a buffer that is
immediately dropped.

> **Trap: the destination dtype is yours to state, and there is no
> default.** `kRaw` places the checkpoint's own bytes; `kBf16` also
> converts f16, which is possible only because f16 and bf16 are the same
> width. The two differ *only* for f16 — and a wrong answer there is
> silent, because the buffer is the right size and full of plausible
> numbers that are off by an exponent bias. If your loader converts f16 by
> some other route, ask for `kRaw`, or it will be converted twice.

A model that is not ready to restructure who owns its buffers can still
change only the *fill* and keep the per-block allocation. That is most of
the win — of the mapped path's cost above, the allocation was roughly a
fifth and the faulting the rest.

---

## 6. Block residency — growing back into free RAM

Streaming every block on every step is the safe answer and an expensive
one. `BlockResidency` (`generative-models/shared/block-residency.h`) lets a
streaming model keep blocks it can afford, and take them back when it
cannot. It is deliberately **not a cache**: a forward is a cyclic scan, so
recency predicts nothing and a fixed resident subset gives exactly its
share of hits.

Six properties keep it from thrashing:

1. **Spend only free headroom.** `fits_growth(need)` is
   `!paging() && need <= available_physical * 0.9`, where `paging()` means
   the box holds more than a quarter of RAM compressed or more than an
   eighth in swap.
2. **Keep the reserve clear** — room for whatever runs *after* the forward.
3. **Hysteresis** — admission demands the block, the reserve, and one more
   block, so the set stops short of the limit rather than at it.
4. **A decaying ratchet** — a level that had to be given back is not
   retried immediately, and is not forbidden forever either.
5. **Measure that kept blocks are still in RAM** — the only signal that is
   not arithmetic. `mincore()` over the resident blocks; pages that have
   left mean the box overruled you.
6. **Probe, then double.** The per-forward admission cap starts small and
   **doubles on every healthy forward**; any shed puts it straight back to
   the probe.

### Probe-and-double, and why not a schedule

What the per-forward cap bounds is the commitment made *before the first
evidence arrives* — which is a question about how much room there is, not
about how many steps are left. Sizing it from the schedule is the obvious
idea and the wrong one: a 5-step run would reach full residency only on
its last forward, where nothing can use it, and a long run would commit
the whole checkpoint on evidence it has not gathered yet.

Doubling is what makes "as much as fits, as early as possible" compatible
with "learn before committing": the probe is one forward's worth of risk,
and two or three healthy forwards reach the whole stack. Retreat stays
multiplicative in the other direction — a shed drops straight back to the
probe rather than decaying — so the ramp cannot walk back up into pressure
it has just been told about.

### Wiring it into a model

```cpp
// Per generation, from the stage: what must stay clear for peers that
// have NOT allocated yet.
dit->set_residency_reserve(bytes);

// In the forward, BEFORE reading the budget:
_resid.note_reserve_allocated(scratch_bytes(...));   // see below
const auto mb = _mc->memory_budget();
_resid.begin_forward(mb, [this] { return evict_tail_block_(); });

// The measurement, once per forward, gated on our own compressed
// footprint having moved (the page walk is not free -- MEASURED at 57 ms
// per 4.3 GB examined):
if (_resid.count() > 0 && _resid.self_compression_grew(mb.self_compressed)) {
  std::size_t examined = 0, incore = 0;
  resident_pages_(&examined, &incore);
  if (examined > 0 && incore < examined) {
    shortfall = true;
    _resid.note_weight_residency(examined, incore, evict);
  }
}
if (!shortfall) { _resid.note_healthy_forward(); }

// Per block:
if (_resid.admit(_mc, block_bytes)) { keep(); _resid.note_admitted(block_bytes); }
```

Four things routinely go wrong here:

- **Double-counting the reserve.** If the model allocates its activations
  *before* it reads the budget — most do — those bytes are already out of
  `available_physical`, and reserving them again asks for the same room
  twice. MEASURED: a 5086 MB reserve against 6021 MB available refused a
  206 MB block by 27 MB, while the 4 GB it was protecting was already in
  the model's own hands. Call `note_reserve_allocated()` with the part you
  have already allocated; `reserve_now()` is what is left to find.

- **Reserving for a peer that will not coexist with you.** If the stage
  frees this model before the next peer runs, that peer's peak is not your
  constraint, and reserving it costs the whole denoise for nothing. Ask
  the same question the free asks, and reserve accordingly.

- **A reserve that is never set.** Growth stays off until a caller sets a
  reserve *at least once*. Setting it to **zero is a real answer** — "the
  question was asked, and nothing runs after me that I do not free first"
  — and is different from never setting one, which leaves the model in
  pure-streaming mode. A model whose residency machinery is wired but
  whose stage never calls `set_residency_reserve()` will stream forever
  and look, in the logs, exactly like a model that chose to.

- **A ratchet that latches.** Giving back to zero must not mean "zero
  forever". The ratchet is floored at one block and decays back upward
  after several clean forwards; a known one-time transient (e.g. a large
  bake that retires per-step weights) should call
  `note_landscape_changed()` explicitly rather than waiting it out.

### Residency alone is not enough — the blocks have to be wired

A block that is written once and then read only by the GPU carries no CPU
reference bits, so the compressor treats it as cold and harvests it within
a single forward. MEASURED on a 64 GB box running a bf16 33B DiT: the
model grew to 7 of 50 blocks and stopped, with `paged_out 5888/5888` and
9939 MB of its own pages compressed — a resident set that was resident
only in the accounting.

The fix is mechanism 7. With the same blocks wired into the pool, the same
run reached **49 of 50**.

Be honest about what that buys. On the same run the GPU spends ~4146 ms
per block against ~22 ms of exposed read, with the prefetch hiding 49 of
49 — so a perfect resident set hides at most ~0.5% of that pass. The gains
are **memory health and disk traffic** (163 GB → 90 GB read over the run),
not wall clock. A residency change that claims a speedup on a
GPU-dominated model should be measured twice.

### Reading the refusal

At debug level, a refusal prints why:

```
block residency: stopped growing at 0 blocks (0 MB) -- 3116 MB available
(553 MB of it idle), 9316 MB compressed of ours, 3828 MB swap, against a
0 MB reserve still to find (of 0 MB)
```

Read it in this order: **is the box already paging?** (compressed and swap
against total RAM). If so, nothing about your reserve matters — property 1
short-circuits and the fix is upstream, in what else the graph is holding.
Only if the box is *not* paging is the reserve the thing to look at.

---

## 7. The wired pool

Rule 1 is that nothing goes to swap. Wired (`mlock`'d) memory is the only
mechanism that *enforces* it: wired pages cannot be compressed or swapped,
so what is in the pool is what this process genuinely keeps, and what is
outside it is what the OS may take back at any moment.

One pool, on the manager, for everything vpipe plans to hold: streamed
blocks, the trunk, activation scratch, the VAE.

```cpp
if (mgr->wired_pool_can_take(bytes)) { /* allocate it */ }
const std::size_t got = mgr->wire_into_pool(buffer);   // 0 = refused
...
mgr->unwire_from_pool(buffer);   // ONLY on the way to destroying it
```

**Why a pool and not a budget per model.** A per-model ceiling is a guess
about how the box divides, made by each model separately and therefore
never adding up to the box. MEASURED: a DiT wired 32 GB against its own
half-of-RAM ceiling while the scratch, the trunk and the VAE stayed
reclaimable beside it — which is the wrong half to protect. Those are the
allocations a forward cannot proceed without.

**Order matters, and it is the opposite of intuition.** Wire the trunk and
the scratch *first*, the blocks after. A resident block is an optimisation
the model can shed and stream instead; the scratch is what a forward
cannot proceed without and the trunk is read on every block of every
forward. Protecting the optional half first is how a run ends up with
32 GB of wired blocks beside an activation buffer the compressor is free
to take.

**`wired_pool_pct` is an UP-TO, not a reservation.** Default **75**
(config key `wired_pool_pct`, `VPIPE_WIRED_POOL_PCT` overrides). When
`mlock` refuses — another process holds wired memory, or the system limit
is nearer than the fraction implies — the pool collapses to what was
actually granted and callers stop asking for more. So the figure is a
ceiling on ambition, never a promise. `0` turns wiring off entirely, which
is the right answer on a shared machine.

**A refusal must not be rolled back.** `SharedBuffer::set_wired(false)`
also flips the buffer to purgeable *volatile*, which is the one state a
block still in use must never be in. When the pool refuses mid-block,
**stop and keep what is already wired**: a partly wired block is merely
partly protected, where a rolled-back one can be reclaimed under the
model's feet.

### The refusal check — and why it is off by default

The plan compares `phase_peak()` — every streamable component at its
floor, weights counted once, the interval-aware peak from mechanism 2 —
against the pool, and reports:

```
resource-plan: peak 5527 MB in phase 'decode' (condition 4479,
  denoise 3116, decode-audio 585, decode 5527)
```

When the peak does not fit, it says so, names the phase, and computes the
percentage that *would* hold it:

```
resource-plan: this graph needs at least 59392 MB resident at its peak --
  phase 'denoise', with everything streamable at its floor -- and the
  wired pool is 49010 MB at wired_pool_pct=75. Set wired_pool_pct to 91
  or more.
```

Above 95 there is no such percentage, and it says *that* instead — "raise
it to 723%" reads like a setting where "it does not fit this machine at
any setting" is the fact.

**`wired_pool_enforce` is FALSE by default**, so this is reported and not
refused. A refusal is only as good as the accounting behind it, and the
accounting is not yet complete: a component that can stream but does not
declare its floor is counted at full size, so a graph that fits reads as
one that cannot. MEASURED on a bf16 MiniMax-H3 graph that runs fine on a
64 GB box — the DiT declares a 3 GB floor against 63 GB, the 48 GB text
encoder declares none, and the plan concludes 58 GB against a 49 GB pool.
That veto would have blocked a working run.

So the numbers ship first and the veto follows, per component, as each
streamable thing states its floor. Turn it on with `wired_pool_enforce`
(or `VPIPE_WIRED_POOL_ENFORCE=1`) on a box where the graph is known to be
declared completely — and note what it buys: an **early** refusal, before
minutes of loading, instead of a box discovering the problem by thrashing.

---

## 8. Parking — the reclaimable middle

Weights are the largest, longest-lived, most idle allocations in the
process, and they are immutable once loaded. Historically there were two
ways to deal with an idle model: destroy it (paying a full reload) or keep
it (starving everything else). Neither is the answer usually wanted, which
is *let go of it if something else needs the RAM, otherwise keep it.*

That is what parking is. `WeightRegistry::park()` walks an owner's buffers
and marks them purgeable; the pages survive unless the kernel needs them.
The next access reactivates, reloading from disk **only if they were
actually reclaimed** — and reporting whether they were, so a purge is never
silent.

Consequences worth internalising:

- Parking is **never worse than destroying**: same reclaim value, and free
  when the pages survive.
- Parked bytes count as *available* to a peer's growth check
  (`available_physical` includes purgeable), so parking hands room to a
  streaming peer immediately in the accounting, without the reload cost.
- But parked bytes are **still held bytes** for `weight_footprint()` and
  the manager's totals. When a peer needs room *deterministically and now*
  — a DiT deciding this step whether to keep a block — destroying is the
  only answer that delivers.
- **Wired bytes are not parkable**, and that is the point of both: the
  pool is what this process refuses to give up, parking is what it offers.

### Parking cannot reach uncached weights

`for_each_weight()` enumerates a weight set's **cached `Copied` tensors
with a known source name** — the only ones that can be put back. A loader
that reads its weights with `read()` holds them in its own members, where
the registry cannot see them, and parking such a set reports **0 bytes**.

This is not a bug in parking; it is the direct consequence of mechanism 1.
Language-model loaders read uncached by design, so today they park
nothing, and `park_weights()` logs the zero rather than hiding it. **If
your model holds large weights outside the weight set and you want them
parkable, make the model a `WeightOwner` and implement `for_each_weight()`**
— enumerating weights only. Never list scratch, activation or K/V buffers:
those are written during a forward, so parking them loses live state
instead of reclaiming dead bytes.

The same enumeration is what mechanism 7 wires the trunk through, so
implementing it buys both.

### Who parks

- `GenerativeModelManager::park_weights(dir)` — a stage saying "I am done
  with this one for now".
- `enforce_memory_cap()` — the process-wide policy (mechanism 10).

---

## 9. The removable pool — weights that outlive a launch

A checkpoint a stage is finished with can be kept alive and purgeable
instead of dropped:

```cpp
// BEFORE the reset, always -- see the trap below.
mgr->pool_weights(_dit_dir);
_dit.reset();
```

Two things fall out of that:

- A **relaunch** over the same model pays no reload. This is why the pool
  lives on the manager and not on the runtime — a launch-scoped pool could
  not do the one thing it exists for.
- Under pressure the pages go, because pooling marks them purgeable.
  Nothing pooled is holding the box hostage; it is spare capacity that
  happens to still be useful. `pooled_bytes()` is therefore **capacity,
  not occupancy** — report it that way. `pool_evict(want)` drops pooled
  sets until `want` bytes have been given back.

In plan terms (mechanism 4) this is what `reclaimable` means: floor zero,
preload full. On a tight box it costs nothing; on a roomy one it costs
`preload` and saves the reload.

> **Trap: pool BEFORE you drop the last reference.** `pool_weights()`
> finds the set through a **weak** reference, so calling it after
> `.reset()` finds an expired pointer and does nothing — silently, and
> with exactly the log line you expected to see. This was wrong at all
> seven call sites when the pool was introduced.

### Not everything may be recycled

A set specialised to a run's parameters would be silently wrong for a
launch that does not share them. Weights say so themselves:

```cpp
_ws->set_not_recyclable(
    fmt("AdaLN baked for a {}-step schedule", steps));
```

An AdaLN bake is the case this exists for: it retires the per-step
projections into a schedule-specific form, so it is reusable only by a
launch with the same step count. `pool_weights()` checks `recyclable()`
and drops the set instead of pooling it, naming the reason. The default is
`true` — a set nobody specialised is reusable — so this is opt-out, and
the opt-out belongs wherever the specialisation happens, not at the pool.

---

## 10. The memory cap

`set_memory_cap(bytes)` — config key `memory_cap_mb`, or `--memory-cap-mb N`
on `vpipe` / `vpipe-web-ui`. Over the cap the manager **parks** the
least-recently-used weights rather than refusing a load. So the cap bounds
what the process *insists* on holding, and overshooting costs throughput,
never correctness. Parked bytes stop counting toward it — otherwise the
policy would chase a number it can never reach.

**The cap is off by default (0 = uncapped), and when it is off,
`enforce_memory_cap()` parks nothing at all.** A model that relies on
parking for its memory story must not assume the cap is set.

Mapped weights and K/V cannot be parked, so the cap is a target on the
dominant term, not a process-wide guarantee.

### K/V grows during the run

K/V is the one large allocation that grows *while running*, so a
weights-only accounting reads as healthy right until a long context
exhausts the box. `ContextManager::resident_bytes()` → `ModelExec::kv_bytes()`
→ `GenerativeModelManager::resident_kv_bytes()` reports what the pools grew
to. **A new exec that owns its K/V must override `kv_bytes()`.**

---

## Idle policy: `destroy` / `park` / `keep` / `auto`

Model-holding stages accept `unload_when_idle`:

| Value | Effect | In the plan |
|---|---|---|
| `destroy` | Free the bytes. The accounting shrinks; the next beat reloads from disk. The only one that gives a peer room immediately. | `releases = true` |
| `park` | Purgeable. Reclaimed only under pressure, reused with no reload otherwise. | held |
| `keep` | Hold pinned. Nothing can reclaim it, so under pressure the OS compresses and swaps it instead — paid in both directions, for read-only bytes that could have been re-read from a file. | held, full `preload` |
| `auto` | Resolve from the box (default), and pool what is released. | `reclaimable = true` |

`always` and `never` are accepted as legacy spellings of `destroy` and
`keep`.

`auto` resolves **after the init barrier**, at the first `process()`, when
every peer has loaded and real bytes are authoritative — not at
construction, where a stage would size against a half-loaded graph. A text
encoder resolves it roughly as:

- a peer decided to stream, **or** the box cannot hold the footprint plus
  `kHeadroom` → `destroy`;
- otherwise → `park`.

Note what this implies for the common case: on a constrained box the text
encoder goes **before** the DiT does. The asymmetry justifies it — the
DiT's alternative is streaming every block of every step, while the
encoder's is one reload per prompt.

`auto` is also the case that reaches the removable pool: the weights are
handed to the manager rather than dropped, so a second launch over the
same model finds them. Which is why the plan calls it `reclaimable`
rather than `released` — the lifetime has not ended, the bytes are merely
surrenderable.

Decisions that *cannot* be revised — block streaming above all — must
still be taken at construction, with the wider `kStreamHeadroom` cushion.

Two release paths that must not be confused:

- **A stage that hands off and is done** (a `vae-encode` producing one
  latent) releases at its own idle point, and pools.
- **A stage that runs two models in sequence** (a `model-eval` comparing
  two models' logits) frees the first before the second loads, so its peak
  is **one** model — the larger — not their sum. Declaring the sum there
  refuses runs that work.

---

## What a run reports

All of the accounting goes to the UI at **info** level, so a user who is
about to be told a graph does not fit can see the arithmetic that said so:

```
resource-plan: 11841 MB of model weights declared before init (117873 MB
               of claims, before shared checkpoints are counted once)
resource-plan: 11841 MB preloaded / 5527 MB if every streamable
               component streams
resource-plan: 5464 MB of that is phase-limited; peak across phases 5527 MB
resource-plan: 3 activation arena(s) declared, 617 MB; peak across
               phases 585 MB
resource-plan: peak 5527 MB in phase 'decode' (condition 4479,
               denoise 3116, decode-audio 585, decode 5527)
memory-plan (declared): peak 5527 MB at 'vae-decode' streaming, 11841 MB
               at 'generate-video' preloaded, over 7 stages in running
               order (...)
```

At **warning** level: a misaligned checkpoint (mechanism 1), a phase claim
whose release never arrived (mechanism 2), a claim whose kind has no
planner, and a graph that does not fit the pool while `wired_pool_enforce`
is off (mechanism 7). At **error**, only the last of those with the veto
turned on.

---

## The plug-in surface

A plug-in registers a **family**, not a stage, so these are the methods
through which everything above reaches your code. Defaults are given for
every one — and each default is a specific claim about your model that is
wrong for most models.

### `VideoModelFamily` / `VaeModelFamily`

| Method | Default | What the default asserts |
|---|---|---|
| `declare_resources(root)` | `{}` | "This family loads nothing." Override it. |
| `declare_holdings(root)` | `{}` | "…and nothing in the plan either." Override it. |
| `latent_bytes(root, w, h, frames)` *(video)* | `0` | "I cannot size my own latent." |
| `audio_cost(root, frames, fps, …)` *(video)* | `false` | "I generate no soundtrack." |
| `vae_path(root, role)` *(VAE)* | `{}` | "The host can find my weights." |
| `align_frames(root, frames)` | identity | "Any frame count is legal." |
| `size_grid(root, &gh, &gw)` | `(0, 0)` | "Any frame size is legal." |
| `idle_peers(root)` *(VAE)* | built-in guess | "My peers use the standard directory names." |

`declare_resources(root)` is the one that must not be skipped. It is
answered **before any weights load**, so it has to be derivable from the
checkpoint on disk — a directory listing, a config file, a metadata
envelope. If a component is optional and absent, declare it anyway: a
directory that resolves to nothing contributes zero, which is harmless,
whereas a component you did not name is invisible to every peer.

`declare_holdings(root)` is the same checkpoints in **mechanism 4's**
terms — `source`, `preload`, `floor`. Answer both: they are two ledgers,
computed differently, and a family that fills one is visible to half the
accounting. Leave `releases` and `reclaimable` alone; they are the
stage's `unload_when_idle`, which a family cannot see, and the stage
stamps them on.

`latent_bytes()` and `audio_cost()` are the **beat-shaped** terms — the
payloads and arenas mechanism 3 covers — and they exist because that
geometry is yours. Channel count, spatial and temporal compression all
differ per family, so a host substituting a built-in's formula produces
a confident number for the wrong model. Both decline by default, and the
host then declares *nothing* rather than something plausible.

`vae_path(root, role)` is how a single-file pack tells the host where
its weights actually are. The host resolves a VAE by looking for
`vae/config.json` and falls back to `root` when there is none — and
`root` is then the name it releases, pools and reports a phase release
for, which on a repack is the entire repository. A family that ships a
video VAE and an audio VAE in two files must answer for both roles: they
are loaded and dropped by different stages, in different phases.

`align_frames()` and `size_grid()` look like validation helpers and are
also **memory inputs**. The scratch estimate is computed from the
geometry your model will *produce*, and a family that rounds up without
saying so is asking peers to size a clip nobody will make. Both are asked
of the family precisely so they can be answered without loading anything.

### `VideoGenerator` / `VaeDecoder` / `VaeEncoder`

| Method | Default | What the default asserts |
|---|---|---|
| `resident_bytes()` | `0` | "I hold nothing the manager cannot see." |
| `release_idle()` | no-op | "I have nothing to drop between requests." |
| `decoded_frames(latent_frames)` *(VAE)* | identity | "I do not expand time." |

**`resident_bytes()` is not optional if you read uncached.** A weight set
accounts for what it *cached*; weights read with `read()` into your own
members are invisible to it. Answer with what you are actually holding,
or the graph sizes itself against a checkpoint it cannot see. This is the
single most consequential default on this page.

**`release_idle()` is a request, not a contract.** It means "drop what
you can", and a family with nothing droppable legitimately does nothing.
Because of that, the host will not subtract your weights from a peer's
budget on the strength of it — if you want your model excluded from a
peer's sizing, the mechanism is a phase claim, which carries a promise
the host audits.

**`req.progress` is not optional either, and not only about progress.**
The same callback carries the counts a progress bar shows *and* the
host's answer to "should I still be running?". A `decode()` that never
calls it publishes nothing — and, more to the point, **cannot be
cancelled**: a stop pressed during a minute-long decode does nothing
until the decode has finished anyway. Call it wherever your decode
already synchronises — per tile, per block, per chunk — and treat `false`
as "return now". Nothing else can stop you, because nothing else is
looking.

### What the host will not do for you

- It will not discover a checkpoint you did not declare.
- It will not infer your rounding rules from your output.
- It will not notice that `resident_bytes()` returning `0` is a lie.
- It will not merge two stages' holdings you did not name with a `source`.

Each of these fails silently and in another stage. That asymmetry is why
the defaults are conservative-looking but are, in practice, the wrong
answer for any family that holds real weights.

---

## Porting checklist

For a new model:

- [ ] The loader takes `std::shared_ptr<WeightSet>` and **keeps it** for
      the model's lifetime (a `_ws` member). Mapped tensors point into the
      set's mmap; cached tensors are refcounted aliases of its buffers.
- [ ] Obtain it with `open_weight_set(dir, session())`, so the manager owns
      the checkpoint and two models over one checkpoint share it. A
      loader that opens a checkpoint privately is invisible to every
      mechanism on this page.
- [ ] Reads are `Copied` unless the loader converts almost nothing. If you
      chose `Mapped`, the log says the shard was mappable — otherwise you
      have the copies anyway, plus the mapping.
- [ ] The right accessor per tensor: cache what you keep, read what you
      consume. `derived()` keys name the transform *and* every config input
      that changes the bytes.
- [ ] Nothing ever writes into a buffer that came from the weight set.
- [ ] If the model specialises its weights to a run's parameters, it calls
      `set_not_recyclable()` with the reason.

For a plug-in family (see "The plug-in surface" above):

- [ ] `declare_resources(root)` names every checkpoint the family opens,
      answerable from disk before anything loads.
- [ ] `declare_holdings(root)` says the same thing in the plan's terms —
      `source`, `preload`, `floor` — leaving the policy fields alone.
- [ ] `vae_path(root, role)` if the weights are not where the host's
      `vae/config.json` walk would find them, answered for BOTH roles
      when the family ships a video VAE and an audio VAE.
- [ ] `latent_bytes()` / `audio_cost()` size the payloads and arenas the
      beat implies, or decline so the host declares nothing rather than a
      built-in's formula over your geometry.
- [ ] `resident_bytes()` overridden if any weights are read uncached into
      your own members — the default `0` is a claim, and a false one. It
      is also what corrects the plan after a streaming load, so a family
      that leaves it at 0 goes on being counted at its on-disk size.
- [ ] `align_frames()` / `size_grid()` report the rounding you actually
      apply, because the scratch estimate is computed from them.
- [ ] `decoded_frames()` reports time expansion, if your VAE expands time.
- [ ] A large working buffer is stated with a scratch claim, and restated
      per beat once its real size is known.

For a stage that holds one:

- [ ] `declare_memory()` names each holding by its **source**, with
      `preload`, `floor`, `releases` and `reclaimable`, plus `scratch` and
      per-oport `outputs`.
- [ ] `declare_resources()` returns `model_memory::weight_claims({dirs...})`,
      or `weight_claim_streamable(dir, floor)` for anything that can be
      reduced rather than dropped.
- [ ] If the model can arrive on an iport, `apply_constant()` latches it —
      otherwise both declarations are empty in exactly the graphs that
      name their model once, in a source.
- [ ] Both declare methods read nothing — no `bounded()`, no footprint,
      no manager. Anything conditional is in `decide_resources()`.
- [ ] Weights released before the peers that size against them run are
      claimed with `weight_claims_in_phase()` **from `decide_resources()`**,
      gated on `destroy` (never `park`), and report `note_phase_released()`
      when they go.
- [ ] Buffers handed downstream are declared with `payload_claims()`,
      spanning from the phase that writes them to the last that reads them.
- [ ] Streaming decided by `plan_streaming()`, with `kStreamHeadroom`.
- [ ] `revise_declaration()` and `revise_memory()` after load if it keeps
      less than it declared; `revise_scratch()` per beat once the geometry
      is known.
- [ ] Revisable idle policy resolved at the first `process()`, not at load.
- [ ] `pool_weights(dir)` **before** dropping the last reference, on every
      release path.
- [ ] Weigh `peer_streams(session)` before choosing to keep anything.

If it streams blocks:

- [ ] Blocks are read into destinations the model keeps, refilled in place
      — not allocated and mapped-copied per block. State the destination
      dtype (`kRaw` / `kBf16`) rather than taking a default; there is none.
- [ ] The refill fallback is decided **per tensor**, not latched for the
      block.
- [ ] `set_residency_reserve()` is actually called — with a real figure, or
      with an explicit `0`.
- [ ] `note_reserve_allocated()` if activations are allocated before the
      budget read.
- [ ] The per-forward residency measurement and `note_healthy_forward()`.
- [ ] `note_landscape_changed()` after any one-time event that frees a
      large amount.
- [ ] Trunk and scratch are wired into the pool **before** any block is.
- [ ] A pool refusal stops the loop and keeps what is wired; it never
      rolls back.

If it owns K/V:

- [ ] `kv_bytes()` overridden.

If it wants to be parkable or wirable:

- [ ] It implements `WeightOwner::for_each_weight()`, listing weights only —
      or it accepts that parking reports 0 for it, and says so.

---

## Diagnosing a box that is thrashing

1. **Is anything in swap?** That is rule 1, and it is the only symptom
   that is never acceptable. Everything below is about finding which
   mechanism let it happen.
2. **What did the plan say?** The `resource-plan:` and `memory-plan:`
   lines are info-level and name the tight phase and the tight stage. If
   the reported peak is far below what the box actually did, something is
   holding bytes it did not declare — the usual culprit is a loader that
   reads uncached and does not override `resident_bytes()`.
3. **Is the checkpoint aligned?** A misaligned pack costs a full extra
   copy per tensor per read **and** grows the file cache one-for-one with
   what it reads, which is the pressure that evicts everything else. The
   warning is at `open()`. If it names a pack you produced, re-quantize.
4. **Which mechanism refused?** Run at debug level and read the
   `block residency: stopped growing …` line. Compressed and swap against
   total RAM tell you whether the box was already paging, in which case the
   reserve is irrelevant.
5. **What is actually resident?** `mincore()`-based residency is the only
   trustworthy answer; free-memory arithmetic is self-generated on a
   streaming model. On one 64 GB box, `available_physical` read ~18.5 GB
   while the machine held 33 GB compressed and 28.7 GB of swap — almost
   all of that "available" was file cache.
6. **Is the right half wired?** A run with wired blocks beside a
   reclaimable scratch arena has the pool inverted. Trunk and scratch
   first.
7. **Who is holding what?** A stage keeping something resident that a
   streaming peer needs is the usual cause, and the idle policy above is
   the usual fix.
8. **Did the mapping happen?** A model that asked for `Mapped` and got
   copies looks exactly like one that asked for `Copied`, and the only
   places it shows are the warning line and the memory meter. Check
   before attributing the footprint to anything else.
9. **Are you reading the right meter?** Physical footprint
   (`vmmap -summary`) is the number to compare; max-RSS counts shared
   file pages and reads far higher on a mapped model — on one run, 47 GB
   of RSS against 27 GB of footprint. Two residency changes measured as
   *regressions* on RSS that were flat or wins on footprint.
10. **Did a "win" survive a re-run?** Thermal and residency effects both
    invert A/B results. Interleave the arms.

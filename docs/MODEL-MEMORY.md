# Model memory in VPIPE

A VPIPE graph routinely asks one machine to hold several large models at
once: a diffusion transformer, a text encoder, a VAE, sometimes a language
model beside them. On a 16 GB machine that does not fit, and on a 64 GB
machine it usually does — but the same pipeline description has to work on
both, and it has to work the *same way* on two runs of the same machine.

## Who this is for

Anyone adding a model, a loader, or a model-holding stage, **including
plug-ins built out of tree**. If your plug-in loads weights, allocates a
large working buffer, or decides whether to keep something resident, this
document is the contract you are implementing against.

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
3. **Override `declare_resources()`** on your stage — or
   `declare_resources(root)` on your model family — to name every
   checkpoint you are about to open (mechanism 2). This is the single
   most common omission, and the one whose absence is invisible in your
   own stage.
4. **If your model can arrive on an input port**, implement
   `apply_constant()`, or the declaration above will be empty in exactly
   the graphs that name their model once, in a source.
5. **Declare large working buffers** with a scratch claim (mechanism 3)
   if you allocate one that is comparable to your weights. For a VAE
   decode it is usually much larger than them.

Steps 1–3 are mandatory. Everything after that applies when the model is
too big for the machine it lands on, and can be added later without
reworking the above.

If you are implementing a model **family** rather than a stage, read
"The plug-in surface" near the end first: it lists the methods these
mechanisms actually reach your code through, and what each default
silently asserts about your model.

## The rule the rest follows from

> **Measure, do not predict.** Free memory moves underfoot, and
> arithmetic over it is self-fulfilling on a streaming model: a model
> that has evicted its own weights sees free RAM and concludes it has
> room. Where a decision can be re-checked against what actually
> happened, re-check it — and prefer a number the machine reported to one
> you derived.

A corollary worth stating separately, because it is the failure mode this
document exists to prevent: **an under-estimate is not a conservative
error.** Declaring less than you will hold reads to every peer as room
that is not there, and the peer that believed you may already have taken
a decision it cannot reverse. Where you cannot compute a figure, say so
(there are mechanisms below for exactly that) rather than supplying one
that is merely small.

---

## The seven mechanisms

| # | Mechanism | Question it answers | Granularity | When decided |
|---|-----------|--------------------|-------------|--------------|
| 1 | Tensor residency | How do these bytes get into RAM? | per tensor | in the loader |
| 2 | Declarations | How much will everyone else hold? | per checkpoint | before any model loads |
| 3 | Activation scratch | How much will everyone else *allocate*? | per buffer | declared before, corrected per beat |
| 4 | Block streaming | Do I hold this checkpoint at all? | per model | at construction, irreversible |
| 5 | Block residency | May I keep *this* block resident? | per block | re-measured every forward |
| 6 | Parking | May somebody else have my idle bytes? | per weight set | at an idle point |
| 7 | Memory cap | What does the process insist on holding? | process-wide | from config |

Mechanisms 1–3 are mandatory for every model. 4–5 apply to models too big
for the machine. 6–7 are how idle weights stop being dead weight.

They are layers, not independent knobs: each answers a different question,
and a model that wires one and skips another tends to behave correctly on
the machine it was written on and nowhere else.

---

## 1. Tensor residency — `Copied` vs `Mapped`

Every loader takes a `std::shared_ptr<WeightSet>` and reads tensors through
it. Two residencies exist:

- **`Copied`** — allocate a `SharedBuffer` and memcpy into it. Anonymous
  memory: reclaimable only through the compressor and swap. Parkable.
- **`Mapped`** — a zero-copy view into the mapped shard. Clean file-backed
  pages, which the kernel can drop for free and re-fault on demand. *Not*
  parkable (there is nothing to hand over that the OS does not already
  manage better).

`Mapped` sounds strictly better and usually is not. Mapping one tensor
wraps the **whole shard**, so touching a single tensor makes the entire
checkpoint resident. That only pays for conversion-light loaders. A loader
that allocates a converted copy on any dtype mismatch pays both bills:
measured on one language model, switching its reads to `Mapped` took peak
footprint from 3.62 GB to 5.00 GB (+38%).

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

> **Trap.** The accessor you choose decides whether mechanism 6 can ever
> see your weights. See "Parking cannot reach uncached weights" below.

> **Bigger trap: `Mapped` is a REQUEST, and the file can refuse it.**
> Zero-copy wrapping hands Metal a subview of the mapped shard, and a
> Metal buffer offset must be **16-byte aligned**. Two things have to hold
> for a tensor to get one, and they fail independently:
>
> 1. **The data section starts aligned.** A safetensors file is
>    `[u64 length][JSON header][data]`, so `8 + header_len` decides where
>    the blob begins. Get it wrong and *every* tensor in the shard falls
>    back to a copy. MEASURED on a 22B DiT whose section began at 677624
>    (≡ 8 mod 16): **4349 of 4349 tensors, 39.1 GB, copied** — anonymous
>    memory, where the model asked for file-backed pages. It surfaced as
>    a box thrashing with 36 GB in the compressor, a long way from the
>    cause.
> 2. **Each tensor's SIZE is a multiple of 16.** Tensors are packed
>    contiguously, so one that is not shifts every tensor after it off
>    the boundary — and the offenders are small and easy to miss. MEASURED
>    on a quantized 12B: one 2-byte per-layer scalar per block, 48 of them
>    interleaved with the weights, plus four JSON asset blobs — 53 tensors
>    that between them cost **1181 of 1344**, with the section start a
>    clean 0 mod 16 the whole time.
>
> Both now warn, once per shard, and the two warnings are different
> sentences on purpose: one names the section offset, the other counts the
> tensors. **Check the log** before believing a model is mapped — a model
> that asked for `Mapped` and got copies is indistinguishable from one
> that asked for `Copied`, except in the log and on the memory meter.
>
> Assume a downloaded checkpoint is not mappable until the log says
> otherwise. Many land on `%16 == 8` (the safetensors header is padded to
> 8, and the 8-byte length prefix then lands the data on an odd
> 16-boundary), and it is common for none of a shard's tensors to be
> mappable at all. vpipe's own `SafetensorsWriter` pads the section to 16
> **and** writes odd-sized tensors last, so in anything `model-quantize`
> or `lora-fuse` produces, every tensor whose size is a multiple of 16 is
> mappable; the few that are not are the odd-sized ones themselves, and
> they are small by construction.
>
> Ordering rather than padding, because the byte ranges have to keep
> tiling the blob: the reference safetensors reader validates that they
> are contiguous, so a writer that bought alignment with gaps would
> produce files nothing else can read.

---

## 2. Declarations — sizing before anything loads

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
model weights are one kind. A claim whose kind has no planner is warned
about, never dropped silently.

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
*then* a VAE turns the latent into pixels. Summing all three sizes the box
for a moment that never happens.

A claim may therefore name the phase it belongs to:

```cpp
return model_memory::weight_claims_in_phase(
    {_encoder_dir}, model_memory::kPhaseCondition);
```

Peers in a *different* phase then stop counting those bytes
(`GenerativeModelManager::phase_footprint`), and `plan_streaming()` asks
in `kPhaseDenoise` — so a DiT sizes itself against a box the encoder has
already left. The vocabulary is fixed (`kPhaseCondition`, `kPhaseDenoise`,
`kPhaseDecode`); an unrecognised phase is warned about and counted as
resident for the whole run, because a typo that became its own phase would
be counted apart from the claims it really coexists with, and that
*under*-counts.

**Phases are decided in a second pass.** `declare_resources()` describes
and must not read the manager; `decide_resources()` decides, and runs only
after every stage has declared:

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

**Two conditions, and neither is checkable by the planner.**

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
`phase_footprint({})` is persistent + the widest single phase, which is
what the machine must survive.

Report the release with `note_phase_released(dir)` when the weights
actually go. A phase claim whose release never arrives is warned about at
the end of the run — otherwise the promise is unfalsifiable, and a broken
one shows up only as thrash with nothing in the log connecting it to the
claim that caused it.

---

## 3. Activation scratch — the other thing that has to fit

Weights are what a model *holds*; scratch is what it *allocates to run*.
For a VAE decode the second dwarfs the first: FLUX.2's VAE weighs 160 MB
on disk and its decode peaks near 2.8 GB at 1024², 14 GB at 2048². A graph
accounted purely by weights reads as roomy right up to the allocation that
does not fit.

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

**Declared by whoever has the geometry, not by whoever allocates.**
`vae-decode` sizes itself from whatever latent arrives, so at planning time
it cannot name a number — which is why it had been using peer *weights* as
a proxy and `kHeadroom` as a stand-in for the arena. `generate-image` has
the geometry in config, so it declares. A graph whose geometry comes from
an input image (an edit pipeline) declares 0, honestly.

Scratch is **phased like weights** and kept in its own ledger, so
`resident_weight_bytes()` keeps meaning what it says. `kPhaseDecode` keeps
the arena out of the DiT's sizing — it does not exist during the denoise —
while leaving it in the box-level peak, which is what says whether the DiT
will have to be freed for the decode.

**An arena's size is a function of the BEAT, so the plan can only bound
it.** A video decode's transient scales with the *pixel* frame count,
which is the VAE's expansion of the latent — a property of the loaded
model, not of config. MEASURED on MiniMax-H3 at 256²: config says 9
frames, the VAE decodes 22, and the declared 5 MB bound is 2.4× short of
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
arena the stage falls back to a flat `kHeadroom` of 6144 MB — a 27x
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
| FLUX.2 | `AutoencoderKLFlux2`, `block_out` x14 | generate-image, vae-decode | exact |
| Boogu | `AutoencoderKL`, `block_out` x14 | generate-image, vae-decode | exact |
| Krea-2 | `AutoencoderKLQwenImage`, `base_dim` x27 | generate-image, vae-decode | exact |
| Qwen-Image-Edit | `AutoencoderKLQwenImage`, `base_dim` x27 | vae-decode | exact |
| LTX-2.5 | rounded config geometry | vae-decode (family path) | exact |
| MiniMax-H3 | rounded config geometry | vae-decode (video path) | exact |
| wan | rounded config geometry | vae-decode (video path) | exact |

> **The Qwen-Image VAE spells its width `base_dim` and ships no
> `block_out_channels` at all** — every quantized Krea-2 and
> Qwen-Image-Edit pack has the key absent, so an estimator reading
> `block_out` fell to the 128 default against a real 96 and
> over-declared 1.33x. Two families, one config key apart. MEASURED on
> `Qwen-Image-Edit-2511-4bit` at 1024x1024: 2592 MB, which is
> `1024*1024*96*27` exactly.
>
> **`vae-decode` publishes on every branch,** not just the video ones.
> Qwen-Image-Edit has no `free_*_dit_for_decode_` in generate-image, so
> before that nothing upstream stated its arena at all — and being an
> edit model it has no config geometry to declare from either, which
> left it on the marker for the whole run.

An image **edit** graph declares `kUnknownArena` instead, since it has no
geometry at all.

**Size the arena from the geometry the model will PRODUCE, not the one
config asked for.** Every video model constrains its latent shape and
rounds up, so an estimate over the requested numbers describes a clip
nobody will make — and it under-counts, which reads as room that is not
there. MEASURED on MiniMax-H3: config 9 frames becomes 22 (the VAE takes
17-frame chunks keeping 5 latents each), and the unrounded bound was 5 MB
against a 12 MB truth. Rounded, the bound is 12 MB — exact.

`GenerateVideoStage::planned_geometry_()` applies the same rules
`resolve_config_` will: a registered family answers `align_frames()` and
`size_grid()` without loading anything, and H3's is a free function over
its config numbers.

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
unrounded estimate.** A future GGUF DiT carries neither a `config.json`
nor a metadata envelope, so every probe fails — and falling through with
the config numbers would produce not a conservative bound but an
optimistic one, which is the failure this whole section exists to fix,
returning silently. `planned_geometry_()` returns false instead, the
stage declares `kUnknownArena`, and says so:

```
GenerateVideoStage: no rounding rule for '<dir>', so the decode arena is
declared as present-but-unsized and corrected on the first clip
```

MEASURED, FLUX.2 klein-9B-4bit at a simulated 16 GB (peers 10059 MB):

| geometry | arena | old (flat 6 GB headroom) | declared |
|---|---|---|---|
| 1024² | 3584 MB | keep | keep |
| 2048² | 14336 MB | **keep** — wrong | **unload** |

**A phase says "not at the same time", which in a concurrent pipeline is a
statement about the graph, not just the stage.** These claims are sound
for a run whose beats do not overlap — one prompt in, one video out, which
is how the generation pipelines are written. A graph that pipelines beats
would have the encoder loading beat *N+1* while the DiT denoises beat *N*,
and the two phases would then be resident together. Nothing detects that
in advance; the end-of-run audit is what catches it after the fact.

Declarations **persist for the run** and count at `max(held, estimate)`.
Do not clear them once loading is done: a weight set only accounts for
what it *cached*, so an uncached-reading model reports a fraction of its
real bytes, and clearing erases it exactly when peers start sizing against
it. A model that genuinely keeps less calls `revise_declaration()`.

Sizing helpers (`stages/model-memory.h`):

- `weight_footprint(session, dirs)` — a **union** over checkpoints, not a
  sum over directory names. A shared checkpoint loads once; an open one
  contributes what it *holds*.
- `plan_streaming(session, dit_dir, enc_dir, headroom)` — the single
  streaming rule for every DiT family. Do not copy-paste another.
- `kHeadroom` (6 GB) for **revisable** decisions; `kStreamHeadroom` (8 GB)
  for **irreversible** ones. Failing to stream means thrash or an OOM
  kill; streaming needlessly costs ~2–3× per step, so the decision that
  cannot be taken back gets the bigger cushion.

These compare against **total RAM**, not free memory, on purpose: every
stage must answer identically and reproducibly. `VPIPE_RAM_LIMIT_MB`
simulates a small box on a big one.

---

## 4. Block streaming — the irreversible one

A model that cannot fit reads its blocks from disk per forward, keeping
only a pinned prefix. This is an argument to the transformer's `load()`,
so it must be right at construction — changing it later means destroying
and rebuilding the model.

Take the decision from `plan_streaming()`, never by hand. It also records
the fact that streaming was chosen, which peers depend on (below).

A streaming model must then call `revise_declaration()` down to what it
actually pins, so peers do not size against weights that are never there.

> **Trap: the revision erases the evidence.** A streaming model correctly
> reports a small number, and that small number reads as *roominess* to
> whoever sizes next. Measured on a 16 GB box: a DiT concluded "11 GB +
> 6 GB vs 16 GB RAM → stream blocks", revised itself down to 3.3 GB, and
> seconds later the text-encoder stage sized against that 3.3 GB, decided
> there was room, and kept a 1.2 GB encoder resident. The run then spent
> its whole denoise at 9.3 GB compressed with 3.8 GB of swap.
>
> That is what `model_memory::peer_streams(session)` is for: it reports
> whether *anyone* in this run decided to stream — i.e. whether somebody
> is already paying per-step disk reads for want of RAM. Weigh it before
> keeping anything resident. It is a record of a decision that was taken,
> not a prediction.

### Reading a streamed block

Having decided to stream, *how* the bytes arrive is the rest of the cost.
The obvious shape — allocate a destination per tensor, fill it with
`stream_tensor` — pays twice: for the allocation, and for demand-faulting
the shard's mapping a cluster at a time.

MEASURED on an Apple Silicon laptop over a 206 MB block, arms interleaved
and their **order rotated** (rotating only the file region flatters
whichever arm runs first):

| Read shape | Rate |
|---|---|
| fresh allocation + `memcpy` from the mapping | 0.86–1.48 GB/s |
| `pread` into a destination that already exists | 6.7–6.9 GB/s |

The spread matters as much as the mean: the mapped rate moves by 2× between
rounds where `pread`'s does not, and a streamed forward turns that into GPU
occupancy that will not sit still.

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
one however you did before" — not "give up". Blocks are rarely uniform:
a released video DiT's block is 140 tensors of which 78 are servable raw
(387 MB) and 62 are not (24 MB of f16 scales and f32 modulation tables).
A rule that abandoned the block for the awkward 24 MB would buy nothing.
`kFailed` is the separate case where the checkpoint and the destination
disagree and the buffer is now partly written.

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

## 5. Block residency — growing back into free RAM

Streaming every block on every step is the safe answer and an expensive
one. `BlockResidency` (`generative-models/shared/block-residency.h`) lets a
streaming model keep blocks it can afford, and take them back when it
cannot. It is deliberately **not a cache**: a forward is a cyclic scan, so
recency predicts nothing and a fixed resident subset gives exactly its
share of hits.

Five properties keep it from thrashing:

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
// footprint having moved (the page walk is not free):
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
  twice. Measured: a 5086 MB reserve against 6021 MB available refused a
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

## 6. Parking — the reclaimable middle

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

### Parking cannot reach uncached weights

`for_each_weight()` enumerates a weight set's **cached `Copied` tensors
with a known source name** — the only ones that can be put back. A loader
that reads its weights with `read()` holds them in its own members, where
the registry cannot see them, and parking such a set reports **0 bytes**.

This is not a bug in parking; it is the direct consequence of mechanism 1.
Language-model loaders read uncached by design, so today they park
nothing. **If your model holds large weights outside the weight set and
you want them parkable, make the model a `WeightOwner` and implement
`for_each_weight()`** — enumerating weights only. Never list scratch,
activation or K/V buffers: those are written during a forward, so parking
them loses live state instead of reclaiming dead bytes.

### Who parks

- `GenerativeModelManager::park_weights(dir)` — a stage saying "I am done
  with this one for now".
- `enforce_memory_cap()` — the process-wide policy (mechanism 7).

---

## 7. The memory cap

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

## Idle policy: `destroy` / `park` / `keep`

Model-holding stages accept `unload_when_idle`:

| Value | Effect |
|---|---|
| `destroy` | Free the bytes. The accounting shrinks; the next beat reloads from disk. The only one that gives a peer room immediately. |
| `park` | Purgeable. Reclaimed only under pressure, reused with no reload otherwise. |
| `keep` | Hold pinned. Nothing can reclaim it, so under pressure the OS compresses and swaps it instead — paid in both directions, for read-only bytes that could have been re-read from a file. |
| `auto` | Resolve from the box (default). |

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

Decisions that *cannot* be revised — block streaming above all — must
still be taken at construction, with the wider `kStreamHeadroom` cushion.

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
| `align_frames(root, frames)` | identity | "Any frame count is legal." |
| `size_grid(root, &gh, &gw)` | `(0, 0)` | "Any frame size is legal." |
| `idle_peers(root)` *(VAE)* | built-in guess | "My peers use the standard directory names." |

`declare_resources(root)` is the one that must not be skipped. It is
answered **before any weights load**, so it has to be derivable from the
checkpoint on disk — a directory listing, a config file, a metadata
envelope. If a component is optional and absent, declare it anyway: a
directory that resolves to nothing contributes zero, which is harmless,
whereas a component you did not name is invisible to every peer.

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
      the checkpoint and two models over one checkpoint share it.
- [ ] Reads are `Copied` unless the loader converts almost nothing. If you
      chose `Mapped`, the log says the shard was mappable — otherwise you
      have the copies anyway, plus the mapping.
- [ ] The right accessor per tensor: cache what you keep, read what you
      consume. `derived()` keys name the transform *and* every config input
      that changes the bytes.
- [ ] Nothing ever writes into a buffer that came from the weight set.

For a plug-in family (see "The plug-in surface" above):

- [ ] `declare_resources(root)` names every checkpoint the family opens,
      answerable from disk before anything loads.
- [ ] `resident_bytes()` overridden if any weights are read uncached into
      your own members — the default `0` is a claim, and a false one.
- [ ] `align_frames()` / `size_grid()` report the rounding you actually
      apply, because the scratch estimate is computed from them.
- [ ] `decoded_frames()` reports time expansion, if your VAE expands time.
- [ ] A large working buffer is stated with a scratch claim, and restated
      per beat once its real size is known.

For a stage that holds one:

- [ ] `declare_resources()` returns `model_memory::weight_claims({dirs...})`.
- [ ] If the model can arrive on an iport, `apply_constant()` latches it —
      otherwise the declaration above is empty in exactly the graphs that
      name their model once, in a source.
- [ ] `declare_resources()` reads nothing — no `bounded()`, no footprint,
      no manager. Anything conditional is in `decide_resources()`.
- [ ] Weights released before the peers that size against them run are
      claimed with `weight_claims_in_phase()` **from `decide_resources()`**,
      gated on `destroy` (never `park`), and report `note_phase_released()`
      when they go.
- [ ] Streaming decided by `plan_streaming()`, with `kStreamHeadroom`.
- [ ] `revise_declaration()` after load if it keeps less than it declared.
- [ ] Revisable idle policy resolved at the first `process()`, not at load.
- [ ] Weigh `peer_streams(session)` before choosing to keep anything.

If it streams blocks:

- [ ] Blocks are read into destinations the model keeps, refilled in place
      — not allocated and mapped-copied per block. State the destination
      dtype (`kRaw` / `kBf16`) rather than taking a default; there is none.
- [ ] `set_residency_reserve()` is actually called — with a real figure, or
      with an explicit `0`.
- [ ] `note_reserve_allocated()` if activations are allocated before the
      budget read.
- [ ] The per-forward residency measurement and `note_healthy_forward()`.
- [ ] `note_landscape_changed()` after any one-time event that frees a
      large amount.

If it owns K/V:

- [ ] `kv_bytes()` overridden.

If it wants to be parkable:

- [ ] It implements `WeightOwner::for_each_weight()`, listing weights only —
      or it accepts that parking reports 0 for it, and says so.

---

## Diagnosing a box that is thrashing

1. **Which mechanism refused?** Run at debug level and read the
   `block residency: stopped growing …` line. Compressed and swap against
   total RAM tell you whether the box was already paging, in which case the
   reserve is irrelevant.
2. **What is actually resident?** `mincore()`-based residency is the only
   trustworthy answer; free-memory arithmetic is self-generated on a
   streaming model. On one 64 GB box, `available_physical` read ~18.5 GB
   while the machine held 33 GB compressed and 28.7 GB of swap — almost
   all of that "available" was file cache.
3. **Who is holding what?** A stage keeping something resident that a
   streaming peer needs is the usual cause, and the idle policy above is
   the usual fix.
4. **Did the mapping happen?** A model that asked for `Mapped` and got
   copies looks exactly like one that asked for `Copied`, and the only
   places it shows are the warning line and the memory meter. Check
   before attributing the footprint to anything else.
5. **Are you reading the right meter?** Physical footprint
   (`vmmap -summary`) is the number to compare; max-RSS counts shared
   file pages and reads far higher on a mapped model — on one run, 47 GB
   of RSS against 27 GB of footprint. Two residency changes measured as
   *regressions* on RSS that were flat or wins on footprint.
6. **Did a "win" survive a re-run?** Thermal and residency effects both
   invert A/B results. Interleave the arms.

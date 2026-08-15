# Model memory in VPIPE

A VPIPE graph routinely asks one machine to hold several large models at
once: a diffusion transformer, a text encoder, a VAE, sometimes a language
model beside them. On a 16 GB box that does not fit, and on a 64 GB box it
usually does — but the same pipeline description has to work on both, and
it has to work the *same way* on two runs of the same box.

This document is the contract for anyone adding a model, a loader, or a
model-holding stage — including plug-in stages built out of tree. The
mechanisms below are not independent knobs: they are layers, each of which
answers a different question, and a model that wires one and skips another
tends to behave correctly on the machine it was written on and nowhere
else.

The single most important rule, from which most of the rest follows:

> **Measure, do not predict.** Free memory moves underfoot, and arithmetic
> over it is self-fulfilling on a streaming model. Where a decision can be
> re-checked against what actually happened, re-check it.

---

## The six mechanisms

| # | Mechanism | Question it answers | Granularity | When decided |
|---|-----------|--------------------|-------------|--------------|
| 1 | Tensor residency | How do these bytes get into RAM? | per tensor | in the loader |
| 2 | Declarations | How much will everyone else hold? | per checkpoint | before any model loads |
| 3 | Block streaming | Do I hold this checkpoint at all? | per model | at construction, irreversible |
| 4 | Block residency | May I keep *this* block resident? | per block | re-measured every forward |
| 5 | Parking | May somebody else have my idle bytes? | per weight set | at an idle point |
| 6 | Memory cap | What does the process insist on holding? | process-wide | from config |

Mechanisms 1–2 are mandatory for every model. 3–4 apply to models too big
for the box. 5–6 are how idle weights stop being dead weight.

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

**Default to `Copied`.** Language models, vision and audio towers all read
`Copied`. What they gain from the weight set is ownership, dedup and a
single open — not reduced residency.

Which accessor to use:

| You are… | Use | Cached? |
|---|---|---|
| keeping the tensor as-is | `tensor(name, mc, res, part)` | yes, parkable |
| keeping a *transform* of it | `derived(key, build, part)` | yes, not parkable |
| consuming it and dropping it | `read(name, mc, res)` | no |
| re-reading it per forward | `stream_tensor` / `stream_derived` | no, counted |

The rule is **cache what the model keeps; read uncached what it consumes.**
Pieces row-concatenated into a fused matrix, bytes copied into a host
vector, and streamed blocks are all consumed — caching them keeps a
redundant copy alive next to the product.

Weights obtained from a weight set are **shared and immutable**. Never
write through one; transform into a new buffer. `VPIPE_WEIGHT_INTEGRITY=1`
hashes `Copied` entries at load and reports any that changed.

> **Trap.** The accessor you choose decides whether mechanism 5 can ever
> see your weights. See "Parking cannot reach uncached weights" below.

> **Bigger trap: `Mapped` is a REQUEST, and the file can refuse it.**
> Zero-copy wrapping hands Metal a subview of the mapped shard, and a
> Metal buffer offset must be **16-byte aligned**. A safetensors file is
> `[u64 length][JSON header][data]`, so `8 + header_len` fixes the
> alignment of *every tensor in the shard at once* — get it wrong and the
> loader falls back to a copy for all of them.
>
> This is not hypothetical and it is not rare. MEASURED on a 22B DiT
> whose data section began at 677624 (≡ 8 mod 16): **4349 of 4349
> tensors, 39.1 GB, copied** — anonymous memory, where the model asked
> for file-backed pages. It surfaced as a box thrashing with 36 GB in the
> compressor, a long way from the cause, because the fallback used to be
> silent. It now warns once per shard.
>
> Two consequences: **check the log** before believing a model is mapped,
> and note that many published checkpoints land on `%16 == 8` (the
> safetensors header is padded to 8, and the 8-byte length prefix then
> lands the data on an odd 16-boundary). vpipe's own `SafetensorsWriter`
> pads to 16, so anything `model-quantize` or `lora-fuse` writes is
> mappable; what you downloaded may not be.

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

## 3. Block streaming — the irreversible one

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

---

## 4. Block residency — growing back into free RAM

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

## 5. Parking — the reclaimable middle

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
- `enforce_memory_cap()` — the process-wide policy (mechanism 6).

---

## 6. The memory cap

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

## Porting checklist

For a new model:

- [ ] The loader takes `std::shared_ptr<WeightSet>` and **keeps it** for
      the model's lifetime (a `_ws` member). Mapped tensors point into the
      set's mmap; cached tensors are refcounted aliases of its buffers.
- [ ] Obtain it with `open_weight_set(dir, session())`, so the manager owns
      the checkpoint and two models over one checkpoint share it.
- [ ] Reads are `Copied` unless the loader converts almost nothing.
- [ ] The right accessor per tensor: cache what you keep, read what you
      consume. `derived()` keys name the transform *and* every config input
      that changes the bytes.
- [ ] Nothing ever writes into a buffer that came from the weight set.

For a stage that holds one:

- [ ] `declare_resources()` returns `model_memory::weight_claims({dirs...})`.
- [ ] Streaming decided by `plan_streaming()`, with `kStreamHeadroom`.
- [ ] `revise_declaration()` after load if it keeps less than it declared.
- [ ] Revisable idle policy resolved at the first `process()`, not at load.
- [ ] Weigh `peer_streams(session)` before choosing to keep anything.

If it streams blocks:

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
4. **Did a "win" survive a re-run?** Thermal and residency effects both
   invert A/B results. Interleave the arms.

# Stage commands

A running stage can be sent **commands**: named requests from outside the
pipeline — the C++ API, Python, a GUI view — that carry JSON arguments and,
optionally, **buffers** in either direction. Data moves by reference both
ways, so a command costs no copy however large its buffers are.

- **In.** The caller hands the stage buffers it owns. The stage reads them
  in place — copies them into a beat, decodes them — and may keep a
  reference past its reply.
- **Out.** The stage replies with views of *its own* memory. The caller
  reads (or, where the channel allows, writes) the stage's bytes in place
  until it **closes** the command. A stage whose exposed bytes would
  otherwise move on **holds** its output while the command is open, so
  everything downstream of it waits exactly that long.

A stage declares its commands in its spec. The host validates every command
against that declaration before the stage sees it, queues commands first in
first out, and cancels them when the pipeline stops.

## Contents

1. [Declaring a channel](#declaring-a-channel)
2. [The layout grammar](#the-layout-grammar)
3. [Lifecycle](#lifecycle)
4. [Ownership and lifetime](#ownership-and-lifetime)
5. [Ordering and delivery](#ordering-and-delivery)
6. [Serving commands in a stage](#serving-commands-in-a-stage)
7. [Calling from C++](#calling-from-c)
8. [Calling from Python](#calling-from-python)
9. [Stages that accept commands](#stages-that-accept-commands)
10. [Versioning](#versioning)

## Declaring a channel

`StageSpec::commands` lists what a stage accepts (`pipeline/stage-spec.h`).
Each entry is a `CommandSpec`:

| field     | meaning |
|-----------|---------|
| `name`    | what the caller sends |
| `doc`     | one line for discovery |
| `args`    | the JSON arguments, as `ConfigKey`s. Type and `required` are enforced exactly as for configuration; an argument **not** listed is refused, so a typo is an error rather than a silent default. Absent optional arguments are filled with their declared defaults before the stage sees them. |
| `results` | the keys of the JSON result, for discovery (not enforced) |
| `in`      | `BufferSpec`s the caller sends |
| `out`     | `BufferSpec`s the stage replies with |
| `holds`   | the stage holds its output while the reply is exposed |
| `extra`   | `{key, value}` pairs for anything the struct has no field for |

Each buffer is a `BufferSpec`:

| field        | meaning |
|--------------|---------|
| `name`, `doc`| identity and description |
| `type`       | accepted element types — see the grammar |
| `shape`      | accepted shapes — see the grammar |
| `format`     | for `bytes`: the media type, e.g. `image/png` |
| `optional`   | IN: the caller may omit it. OUT: the stage may omit it |
| `writable`   | OUT only: the caller may write the stage's bytes before closing |
| `contiguous` | the buffer must be C-contiguous |
| `extra`      | as above |

A command with no buffers at all is ordinary: `finish`, `release`.

## The layout grammar

A buffer carries a concrete `BufferLayout`: an element type, a shape, byte
strides (empty meaning C-contiguous) and, for byte strings, a format. A
`BufferSpec` is a *pattern* over it.

```
type     := "" | type-name ("," type-name)*
type-name:= "bytes" | "u8" | "i8" | "u16" | "i16" | "u32" | "i32"
          | "u64" | "i64" | "f16" | "bf16" | "f32" | "f64"

shape    := "" | alt ("|" alt)*
alt      := "" | dim ("," dim)*
dim      := INTEGER | NAME | "?" | "..."
```

- **type.** Empty accepts any type. `bytes` is an opaque byte string: any
  one-byte element type, rank 1, contiguous. The stage receives it as type
  `Bytes` with the spec's `format`, so a plain Python `bytes` object or a
  `uint8` array both arrive as the same thing. A buffer that names a
  *different* format is refused.
- **shape.** Empty accepts any shape. Alternatives are tried left to right;
  an empty alternative matches rank 0.
  - `INTEGER`: exactly that size.
  - `NAME`: any size, bound on first use. The name must have the same size
    wherever it appears in that command's buffers **of the same direction**.
  - `?`: any size, unbound.
  - `...`: any number of dimensions, including none. At most once per
    alternative.

  | pattern      | accepts |
  |--------------|---------|
  | `3,H,W`      | planar RGB of any size |
  | `1,H,W\|H,W` | a mask with or without its channel axis |
  | `H,W,3`      | packed RGB |
  | `B,...`      | anything with a leading batch axis |

- **Strides** are in bytes and never negative. A unit dimension's stride is
  ignored when judging contiguity, as numpy does. The layout's extent — first
  element to one past the last element reached — must fit in the buffer's
  `size`.

The host checks inbound buffers when the command is **sent** and outbound
buffers when the stage **replies**. A stage replying with a buffer its own
spec does not describe fails the command, naming the buffer, instead of
handing the caller a view it cannot trust. Writability is the spec's call,
not the buffer's: inbound buffers are never writable for the stage, and
outbound ones are writable exactly when `writable` says so.

## Lifecycle

```
 refused by the host ─────────────────────────────────────▶ Failed

          take               reply
 Pending ──────▶ Active ────────────────────▶ Replied
    │              │   fail
    │              └────────────────────────▶ Failed
    │              │
    └──────────────┴── stop / end of stream ─▶ Cancelled

 close() by the caller, from any state ─────▶ Closed
```

| state       | meaning |
|-------------|---------|
| `Pending`   | queued for the stage |
| `Active`    | the stage took it and has not answered |
| `Replied`   | answered: the result and buffers are readable |
| `Failed`    | refused by the host (unknown command, bad arguments, a buffer that does not fit, a stage not running) or by the stage; `error()` says which |
| `Cancelled` | the pipeline stopped, or the stage ended its stream, before the command was **answered** |
| `Closed`    | the caller closed it |

An **answer is final**. A stop while the caller still reads a reply leaves
the command `Replied`, with its buffers still readable, and only releases
the stage from holding it. A refused command is still a handle: the caller
gets a `Failed` command with the reason, never an exception or a null.

## Ownership and lifetime

Every buffer is a `DataBuffer`: a view (`data`, `size`, `layout`,
`writable`) plus an `owner`, a `shared_ptr<void>` that keeps `data` alive.
Copying a `DataBuffer` copies the view and shares the owner.

- **Inbound.** The owner is the caller's: set it to whatever owns the
  memory. The stage may keep the buffer — and therefore the memory — past
  its reply, and the memory is released when the last reference goes, on
  whichever side that is. Python arrays are released through the
  interpreter even when the last reference drops on one of vpipe's threads.
- **Outbound.** The owner is the stage's. The caller reads in place until
  it closes the command. Its views keep the memory **allocated** past the
  close — never its contents unchanged. After the close the stage may reuse
  or release the bytes, and a view the caller kept sees whatever the stage
  does with them.
- **A stage that wants its memory back** checks after the close whether
  the caller let go (`use_count() == 1` on its holder). If so it takes the
  memory back whole. If not, it copies, so the memory the caller kept is
  never shared with the pipeline again. `external-input`'s `lease` and
  `external-tap`'s `read` both do this. Their zero-copy path is the common
  case: a caller that closes and drops its views loses nothing.
- **Closing drops the command's own references** to the stage's buffers,
  so `buffers()` is empty afterwards. The C++ `CommandHandle` and the
  Python object close on destruction, and Python's `with` closes on exit.

### Holding the pipeline

A stage that exposes a buffer another stage would otherwise see must not
let it move on while the caller reads it. It **holds**: it suspends in
`co_await cmd->until_closed()` and emits nothing. Everything downstream
starves and everything upstream backs up, so the pipeline pauses at that
stage and nowhere else. This suspends a coroutine rather than blocking a
worker thread.

A hold always ends: a close by the caller, destruction of the handle, or a
stop (which cancels the command) all release it. Declare `holds = true` on
such a command so callers can discover the promise.

## Ordering and delivery

- Commands to one stage are served **first in first out**, one at a time.
  N commands are N requests, never merged, however close together they
  land.
- A stage's inbox exists from **launch to stop**, and only for a stage whose
  spec declares commands. A command sent at any other time is refused
  (`not running`).
- An inbox holds at most 64 unanswered commands. Beyond that a new one is
  refused rather than queued behind a stage that is not taking them.
- A command closed while still queued is skipped when its turn comes.
- When a stage's driver ends (end of stream, or `finish`), its inbox shuts:
  queued and unanswered commands are cancelled, and later ones are refused
  with the reason.

## Serving commands in a stage

The stage side is `pipeline/stage-command.h`, reached through the stage's
`RuntimeContext`:

| call | does |
|------|------|
| `ctx.try_command()` | the next command, or null — never suspends |
| `co_await ctx.next_command()` | the next command, suspending until one arrives; null once the inbox shuts |
| `co_await ctx.read_any(ports, true)` | wake on a beat **or** a command, whichever comes first; `ports` may be empty |
| `ctx.cancel_open_commands(why)` | cancel what the stage took and still holds, e.g. at end of stream |

A command taken is `Active` and the stage's to answer:

| call | does |
|------|------|
| `cmd->name()`, `args()`, `input(name)` | what was asked; arguments with defaults filled, buffers validated |
| `cmd->reply(result, buffers)` | answer once. `false` means not delivered: already answered, refused for a buffer mismatch, or closed/cancelled first. In every such case nothing is exposed and there is nothing to hold for. |
| `cmd->fail(message)` | refuse it, with the reason |
| `co_await cmd->until_closed()` | hold until the caller closes (or the runtime cancels) |
| `cmd->closed()`, `cancelled()` | closed by either side; cancelled by the runtime, i.e. **drop** rather than emit what the command was about |

A stage that answers commands while idle sleeps on both at once. Wait only
on ports still open: a port at end of stream always reads as ready.

```cpp
vector<unsigned> open;
for (unsigned p : ports) {
  if (!ctx.eos(p)) { open.push_back(p); }
}
co_await ctx.read_any(std::move(open), true);
// ... drain the inputs ...
while (auto cmd = ctx.try_command()) {
  co_await serve_(ctx, std::move(cmd));
}
```

For a tensor, `stages/tensor-buffer.h` converts in both directions:
`tensor_view()` exposes a `TensorBeat` in place, and
`tensor_from_buffer()` gathers a buffer into a new beat.

## Calling from C++

```cpp
#include "vpipe/vpipe.h"

vpipe::PipelineHandle p = session->load_pipeline("graph.vpipeline");
session->launch_pipeline(p);

// IN: the caller's memory, read in place.
auto frame = std::make_shared<std::vector<float>>(3 * 480 * 640);
vpipe::CommandHandle push = p.stage("in").command(
    "push", R"({"sideband": {"t": 0}})",
    {vpipe::DataBuffer::view(
        "data", frame->data(),
        vpipe::BufferLayout::contiguous(vpipe::ElementType::F32,
                                        {3, 480, 640}),
        frame)});
push.wait();                       // replied once the beat is out

// OUT: the stage's memory, held until close().
{
  vpipe::CommandHandle rd = p.stage("tap").command("read");
  if (rd.wait(1000) == vpipe::CommandState::Replied) {
    vpipe::DataBuffer b;
    rd.buffer("data", &b);        // b.data, b.layout, b.writable
  }
}                                  // destruction closes: the tap moves on
```

`StageHandle::commands_json()` describes what a stage accepts. Every call
is thread-safe, and none throws.

## Calling from Python

Buffers go in as anything array-like: numpy, torch or jax arrays, `bytes`,
`bytearray`, `memoryview`, or any DLPack or buffer-protocol object. They
come back as objects with both the buffer protocol and `__dlpack__`, so
`np.asarray(x)`, `memoryview(x)` and `torch.from_dlpack(x)` all view the
stage's bytes without a copy.

```python
import numpy as np, vpipe

p = vpipe.session.load_pipeline("graph.vpipeline")
vpipe.session.launch_pipeline(p)
src, tap = p.stage("in"), p.stage("tap")

src.call("push", buffers={"data": np.zeros((3, 480, 640), np.float32)})

with tap.command("read") as rd:       # closes on exit
    rd.wait()
    frame = np.asarray(rd.buffers["data"])   # the beat, in place
    frame[0] *= 0.5                           # the edit moves downstream

with src.command("lease", args={"type": "u8", "shape": [480, 640]}) as ls:
    ls.wait()
    np.asarray(ls.buffers["data"])[:] = 255  # fill the beat itself
# the lease's beat is emitted here, with no copy
```

`call()` sends, waits, closes and returns the result dict. It raises
`RuntimeError` with the stage's reason on a refusal and `TimeoutError` on a
timeout. `command()` returns a `CommandHandle` with `wait()`, `state`,
`result`, `error`, `buffers` and `close()`. Waits release the GIL and
still answer Ctrl-C.

## Stages that accept commands

**`external-input`** — a source that emits what the caller sends.

- `push`: a copy of the caller's tensor, emitted as one beat. The reply
  comes once the oport has taken it, so a caller that waits on each push is
  paced by the pipeline.
- `lease`: a writable beat of a given type and shape. The caller fills it
  in place and closes, and the beat is emitted — with no copy anywhere when
  the caller let go of its view.
- `finish`: end of stream.

**`external-tap`** — a passthrough that lets the caller read and edit beats
in place.

- `read`: exposes the next `TensorBeat`, writable, and holds it until the
  close; edits travel downstream with it. In `pass` mode beats flow freely
  and a queued read takes the next one. In `gate` mode every beat waits for
  a read, so the pipeline runs at the caller's pace.

**`create-mask`** — the mask editor stage.

- `commit`: a mask as samples (`mask`) or as the PNG the editor paints
  (`png`), emitted as one beat.
- `release`: re-emit the current mask.
- `read`: the current mask and reference image in place; the stage does
  nothing else until the close. The web UI's editor sends its commits
  through this same channel.

## Versioning

Command channels are the feature `stage-commands/1` of the plugin ABI (see
[PLUGINS.md](PLUGINS.md), "Versioning"). What it freezes: the layouts of
`CommandSpec`, `BufferSpec`, `DataBuffer`, `BufferLayout` and the two
awaiters, which live in a plugin's coroutine frame. The ABI snapshot
(`abi/`) holds them to that.

What grows without an ABI change:

- `StageCommand` and `CommandInbox`, which are opaque (`VPIPE_ABI_OPAQUE`).
  The host creates them, and a plugin reaches them only through
  out-of-line methods.
- A new command, a new buffer, or a new `extra` key on any spec.

# Vpipe Manager — the macOS app

A drag-to-Applications app wrapping the two command-line binaries. The
GUI is a **launcher**, not a second implementation: it drives
`Contents/Helpers/vpipe` and `Contents/Helpers/vpipe-web-ui` by argv,
exactly as you would from a terminal. Nothing a pipeline can do is
reachable only from the app, and the CLI cannot drift out from under it.

## Why a bundle at all

Convenience is the smaller half. The larger half is **permissions**.

Every check in `vpipe-web-ui`'s startup report — Local Network, Camera,
Microphone, Full Disk Access — is a macOS TCC permission attributed to
the *controlling process*. Run from a terminal, that is the terminal:
the report describes Terminal.app, granting the permission means finding
Terminal in System Settings, and under `tmux` or `ssh` it is some other
process again. Inside a signed bundle the app is its own TCC subject. It
prompts under its own name, appears under its own name in System
Settings, and the helpers it spawns inherit that identity.

## Building

```sh
cmake -S . -B <build>
cmake --build <build> --target macos-app
```

That produces `<build>/Vpipe Manager.app`, **ad-hoc signed** — it runs locally
and needs no Apple account. Developer ID signing, notarization and the
`.dmg` are a separate step (below).

| Option | Default | Meaning |
|---|---|---|
| `VPIPE_BUILD_MACOS_APP` | `ON` | Build the bundle at all |
| `VPIPE_APP_BUNDLE_ID` | `com.tgous.vpipe` | Bundle identifier |
| `VPIPE_APP_SIGN_IDENTITY` | `-` | codesign identity; `-` is ad-hoc |
| `VPIPE_BUNDLE_FFMPEG` | `OFF` | Copy FFmpeg into the bundle |
| `VPIPE_FFMPEG_BUNDLE_DIR` | — | Where to copy it from |
| `VPIPE_SPARKLE_DIR` | — | `Sparkle.framework` for auto-update |

The Swift sources are compiled by `swiftc` directly, so the whole tree
stays one `cmake --build`; there is no Xcode project.

## What gets bundled

`libssl.3` / `libcrypto.3` are **not optional**: `vpipe-web-ui` links
them directly, so without them the app will not launch on a Mac that has
no Homebrew. They are copied in and relocated automatically.

`bundle-app.sh` chases dependencies to a fixpoint, rewrites every
non-system reference to `@rpath`, and **fails the build** if any
absolute reference survives. That check matters because the failure it
catches is invisible on the machine that produced the bundle — a
leftover `/opt/homebrew/...` path still resolves there.

For the same reason the bundler *deletes* the build tree's absolute
`LC_RPATH` rather than appending to it. Left in place it resolves on the
developer's machine, so the app would load the build tree's `libvpipe`
and appear to work right up until someone else opened it.

### FFmpeg: bundled or not

FFmpeg is `dlopen`'d, never linked, so bundling it is genuinely
optional. The loader tries, in order: `$VPIPE_FFMPEG_DIR`, then
`@loader_path` (the bundled copy), then the system locations.

**Do not bundle Homebrew's FFmpeg.** It is configured `--enable-gpl`
(x264, x265, xvid), which would put this Apache-2.0 project's
distributed binary under the GPL, and its `libavcodec` drags in 31
further Homebrew dylibs. Build a minimal LGPL one instead:

```sh
apps/macos-app/build-lgpl-ffmpeg.sh --prefix ~/dump/ffmpeg-lgpl
cmake -S . -B <build> -DVPIPE_BUNDLE_FFMPEG=ON \
      -DVPIPE_FFMPEG_BUNDLE_DIR=~/dump/ffmpeg-lgpl/lib
```

That build keeps VideoToolbox and AudioToolbox, so H.264/HEVC/ProRes run
on the hardware blocks. LGPL is satisfied by construction: vpipe
`dlopen`s the libraries and never links them, so the relink right is
inherent.

It produces seven dylibs — the six vpipe names plus `libavfilter`, which
`libavdevice` requires. If LAME is installed it is detected and MP3
output is enabled, adding `libmp3lame` to the bundle (LAME is LGPL, so
that is fine); without it, `save-audio`'s `mp3` format has no encoder,
since FFmpeg's native MP3 support is decode-only. `wav`, `aac` and `m4a`
are unaffected either way.

The script ends with a **`dlopen` self-check**, and that check is not
ceremony. vpipe *probes* each candidate library with `dlopen` and treats
a failure as "not present", moving on to the next one — so a library
that links correctly but cannot be loaded produces no error anywhere. It
produces a silent fallback to whatever FFmpeg is next in the search
order. That happened during bring-up: the first build had `@rpath`
dependencies but no `LC_RPATH` of its own, and a whole test run
"passed" against Homebrew's FFmpeg instead. The libraries are now built
with `-Wl,-rpath,@loader_path` so they can find their siblings wherever
they sit.

The corollary for testing: when pointing `VPIPE_FFMPEG_DIR` at a build,
confirm from the log which library actually loaded
(`libavutil: using '…'`) before trusting the results.

Related: `save-video` defaults to `h264_videotoolbox` (matching
`hls-broadcast`), falling back to `libx264` only where VideoToolbox is
absent, such as a Linux build. libx264 is GPL and cannot be in an FFmpeg
build this project redistributes.

## Auto-update (Sparkle)

Optional and off unless a framework is supplied — Sparkle is a 15 MB
binary dependency that only a published build needs.

```sh
apps/macos-app/fetch-sparkle.sh          # verifies a pinned SHA-256
cmake -S . -B <build> -DVPIPE_SPARKLE_DIR=~/dump/sparkle/Sparkle-2.9.5
```

The Swift sources guard every reference with `#if VPIPE_SPARKLE`, so the
same sources build both ways; without it, the "Check for Updates…" menu
item is simply absent.

The feed is a **release asset**, not a file in the repo:
`https://github.com/tgo-app-dev/vpipe/releases/latest/download/appcast.xml`.
GitHub serves the newest release's assets from that fixed path, so the
URL never changes while always resolving to the current appcast.

> **Back up the EdDSA private key.** `generate_keys` puts it in the
> release machine's keychain. It is what proves an update is genuinely
> yours, and there is no recovery: lose it and no already-installed copy
> will ever accept another update — every user has to re-download by
> hand. Export it once with `generate_keys -x <file>` and store it
> somewhere durable.

## Releasing

`tools/release-macos-app.sh` (internal; stripped from the public
snapshot) does the whole path: build → sign with Developer ID →
`.dmg` → notarize → staple → verify → appcast.

```sh
tools/release-macos-app.sh                  # full release
tools/release-macos-app.sh --no-notarize    # local check, no upload
```

It verifies the signing identity and the notary profile **before**
building, since both failures need a human and finding out after a long
compile wastes it.

Publishing needs both files on the same GitHub release — the script
prints the exact `gh release create` command:

- the `.dmg`, which Sparkle downloads from the path baked into the
  appcast;
- `appcast.xml`, which must be an asset of the **latest** release,
  because that is what the feed URL resolves to.

### The GPU check that this Mac cannot perform

**A release must be smoke-tested on an M5 before it is published.**
This is not a nice-to-have, and it is the one release step that cannot
be delegated to the build machine.

The matrix-core (NAX) kernels — `dense_gemm_mma`, `conv2d_mma`, and the
`matmul2d` attention paths — require `MTLGPUFamilyApple10`, which means
M5 or newer. On an M4 they are *never dispatched*: every call site gates
on `MetalCompute::supports_matrix_cores()` and takes the steel/scalar
path instead. So no amount of testing on an M4 exercises a single
instruction of that code. A green 1570-test suite here says nothing
about them.

That gap shipped once. The first published `.dmg` launched on an M5 and
wedged the GPU at 100% utilisation — surviving the app being killed, and
recoverable only by rebooting.

**Root cause: a stale Metal compiler on the build machine.** The build
box was on Xcode 26.2 (Metal 32023.864) while the M5 ran Xcode 26.6
(32023.883). Rebuilding with the matching compiler and re-testing on the
M5 was clean. The details below are what that conclusion rests on, and
what to check when it recurs.

- **The build machine's GPU is not baked in.** `xcrun metal` emits AIR:
  device-independent LLVM bitcode tagged `air64_v28-apple-macosx<ver>`,
  which the driver JITs to the target device's ISA at load time. The
  `matmul2d` operations arrive as `air.externally_defined` and are
  supplied by whatever GPU loads the metallib. So "an M4 cannot compile
  for an M5 GPU" is not the mechanism.
- **The build machine's Metal *compiler* was older.** Xcode 26.2
  (17C52) ships Metal 32023.864; the M5, on Xcode 26.6 (17F113), had
  32023.883. The toolchain is a separately versioned cryptex asset, so
  `xcodebuild -downloadComponent MetalToolchain` reports "installed"
  while still being behind — it only fetches what the *installed Xcode*
  asks for, and is a no-op until Xcode itself moves. **Compare
  `xcrun metal --version` across the build box and the M5 before every
  release**; they must match.

  Updating Xcode has two consequences worth knowing in advance: the old
  SDK is deleted, so any build tree with `MacOSX<old>.sdk` baked into
  its cache must be recreated rather than reconfigured; and the licence
  must be re-accepted (`sudo xcodebuild -license accept`) before *any*
  `xcrun` tool runs — including ones like `strings` that are quietly
  `xcrun` shims, which makes the failure look unrelated.

- **Nothing used to rebuild when the toolchain changed.** The metallib
  rules depended on the `.metal` source and its headers only, so a
  compiler update rebuilt nothing: every `.air` was newer than its
  unchanged source. `add_vpipe_metal_kernel` now adds a configure-time
  stamp holding `metal --version` to each `.air`'s `DEPENDS`, printing
  `toolchain changed -- rebuilding all metallibs`. Verify the result in
  the artifact itself, which records its own producer:

  ```sh
  LC_ALL=C grep -ao 'metalfe-[0-9.]*' \
    <build>/apple-silicon/metal-compute/dense_gemm_mma.metallib | sort -u
  ```
- **The OS-version gate was a separate real bug**, but not this one.
  The NAX kernels are built with `-mmacosx-version-min=26.2` because
  below that they miscompile (mlx#3622 — measured 400% error), so they
  record `…-apple-macosx26.2.0` while every other kernel records
  `26.0.0`. The app advertises `LSMinimumSystemVersion` 26.0, so an M5
  on 26.0/26.1 would have selected NAX and run 26.2-targeted AIR on an
  older runtime. `supports_matrix_cores()` now requires Apple10 **and**
  `kern.osproductversion >= 26.2`, which puts such a machine on the M4
  path — slower, but the path the whole suite covers.
  `VPIPE_FORCE_MATRIX_CORES=1` overrides the OS half for deliberate
  testing; it cannot conjure hardware that is not there.

**`VPIPE_NO_MATRIX_CORES=1` takes the whole matrix-core path out**, ahead
of every other check. The per-family switches (`VPIPE_QWEN_NO_MMA`,
`VPIPE_GEMMA_NO_MMA`, `VPIPE_KREA2_NO_MMA2`, …) each disable one
consumer, and a wedged GPU does not say which kernel caused it — so
bisecting with them costs a reboot per guess. It is also the workaround
to hand anyone who hits this: it puts the machine on the pre-M5 path.

`vpipe --gpu-info` reports the gate's individual inputs
(`device_family_apple10`, `macos_26_2_or_newer`, and both env vars), so
a report from a machine you cannot log into still says *which* condition
decided. A matrix-core fault is only ever visible as a hang on someone
else's hardware, where text is all anyone can send back.

Confirm what a machine will do before trusting a result on it:

```sh
vpipe --gpu-info
# {"cpu":"Apple M4","macos":"26.5.1","metal_valid":true,"matrix_cores":false}
```

`matrix_cores: true` is the only configuration in which the NAX kernels
run at all. If the M5 reports `false`, raise its macOS to 26.2+ before
treating a green run as evidence — otherwise the smoke test passes
precisely because it tested nothing.

### Release checklist

Build from the **public** tree, so the version hash the app reports
matches a commit an outside reader can actually fetch.

```sh
cd /Users/wfang/projects/vpipe-release/github/vpipe
git rev-parse --short=8 HEAD          # this is what the app will show
```

1. **Snapshot and commit the public tree.** `tools/make_release.sh`
   from the private repo, then review `git -C <dest> diff` and commit
   there. Push it before building: the hash must exist upstream.
2. **Check out the submodules in the public tree.** The snapshot strips
   submodule *contents* and re-registers only the gitlinks, so a fresh
   clone or a freshly synced tree has empty `extern/*` directories and
   the configure fails inside `nanobind_add_module` — which reads as a
   nanobind problem rather than a missing checkout.
   ```sh
   git -C /Users/wfang/projects/vpipe-release/github/vpipe \
       submodule update --init --recursive
   ```
3. **Build both variants** (bundled FFmpeg and not) from that tree.
   Each needs its own build tree AND its own output directory:
   `generate_appcast` signs every `.dmg` in the directory it is given,
   so two variants sharing one would both land in the feed.
   ```sh
   PUB=/Users/wfang/projects/vpipe-release/github/vpipe
   cmake -S "$PUB" -B ~/dump/vpipe-pub-full -DCMAKE_BUILD_TYPE=Release \
         -DVPIPE_BUNDLE_FFMPEG=ON \
         -DVPIPE_FFMPEG_BUNDLE_DIR=~/dump/ffmpeg-lgpl/lib \
         -DVPIPE_SPARKLE_DIR=~/dump/sparkle/Sparkle-2.9.5
   cmake -S "$PUB" -B ~/dump/vpipe-pub-slim -DCMAKE_BUILD_TYPE=Release \
         -DVPIPE_BUNDLE_FFMPEG=OFF \
         -DVPIPE_SPARKLE_DIR=~/dump/sparkle/Sparkle-2.9.5

   tools/release-macos-app.sh --src-dir "$PUB" \
       --build-dir ~/dump/vpipe-pub-full                 # appcast here
   tools/release-macos-app.sh --src-dir "$PUB" \
       --build-dir ~/dump/vpipe-pub-slim --no-appcast
   ```
4. **Verify on this Mac** — `--no-notarize` first if iterating; check
   the dmg mounts, the app launches from `/Applications`, and
   `spctl -a -vv` accepts it.
5. **Verify on the M5** — the step above cannot substitute for this.
   Expect to reboot; a failure here wedges the GPU.
   - `xcrun metal --version` on both machines. If the M5's is newer,
     the release should be built on the M5 (step 3 there instead).
   - Install the `.dmg` and run `vpipe --gpu-info` from inside the
     bundle; record it. `matrix_cores: false` means the run proves
     nothing about NAX, so fix that before continuing.
   - Run the matrix-core suites, cheapest first — they are far easier
     to attribute than a whole pipeline:
     ```sh
     for s in gemm_mma sdpa_mma conv2d_mma qmm_mma gdn_mma gemm_i8; do
       ./vpipe_test --filter "$s.*" --color off | tail -3
     done
     ```
   - Then one model end to end (`realtime-vqa` or `text-chat`; both
     hit the NAX GEMM), watching GPU utilisation return to idle after
     the process exits.
   - If anything wedges: reboot, re-run the identical command with
     `VPIPE_NO_MATRIX_CORES=1` to confirm the pre-M5 path is clean,
     and do not publish. That one switch isolates the fault to the
     matrix-core kernels rather than anything else in the app.
6. **Publish** — `gh release create` with both dmgs and `appcast.xml`,
   using the command the script prints.

Steps 1–4 can happen on any Mac. Step 5 requires the M5 and is the
gate: it is the only point in the process where the matrix-core code is
executed by anything at all. Building on the M5 is *not* required — the
one time it appeared to be, the real difference was the compiler
version, which is now checked in step 5 and enforced by the rebuild
stamp. If it ever is convenient, `release-macos-app.sh --src-dir` takes
the public tree from wherever it is checked out; the only extra
requirements are the signing identity and notary profile in that
machine's keychain (see One-time setup).

### One-time setup

```sh
# Notary credentials (app-specific password from appleid.apple.com)
xcrun notarytool store-credentials vpipe-notary \
  --apple-id <apple-id> --team-id 72K6DYBTKN --password <app-specific>

# Stop codesign prompting for the signing key on every release
security set-key-partition-list -S apple-tool:,apple:,codesign: \
  -s -k <login-password> ~/Library/Keychains/login.keychain-db
```

A Developer ID `.cer` on its own is not enough — the matching **private
key** must be in the keychain. A `.cer` downloaded from
developer.apple.com contains only the public half; if
`security find-identity -v -p codesigning` reports no identity, import
the `.p12` from the Mac where the CSR was generated.

## Entitlements

App Sandbox is deliberately absent: it is not required outside the Mac
App Store, and it would break the `run_python` chat tool, which
sandboxes each call itself and cannot be nested. The hardened runtime
*is* applied, because notarization requires it.

| Entitlement | Why |
|---|---|
| `cs.disable-library-validation` | Required: `dlopen` of host FFmpeg and of plugin dylibs signed by other teams |
| `device.camera`, `device.audio-input` | Capture stages, paired with the Info.plist usage strings |

`com.apple.developer.networking.multicast` is **not** requested. ONVIF
discovery falls back to a subnet scan when multicast is unavailable, so
asking for it would add an Apple review cycle for no functional gain.

> When editing `vpipe.entitlements`, never write `--` inside an XML
> comment. XML forbids it, `codesign` rejects the file with
> `AMFIUnserializeXML: syntax error near line N` at the very end of a
> build — and `plutil -lint` reports OK regardless. `bundle-app.sh`
> runs `xmllint` up front to catch it early.

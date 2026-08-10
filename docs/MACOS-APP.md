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

#!/bin/bash
#
# bundle-app.sh -- assemble the app bundle from already-built binaries.
#
# Produces a RELOCATABLE bundle: every non-system dylib the binaries
# reference is copied into Contents/Frameworks and every reference to it
# rewritten to @rpath, so the app runs on a Mac that has neither Homebrew
# nor a vpipe build tree.
#
# Two things make that more than a copy loop:
#
#   * Homebrew records some dependencies by their VERSION-PINNED Cellar
#     path (libssl asks for .../Cellar/openssl@3/3.6.2/lib/libcrypto.3.dylib,
#     not the opt/ symlink). A relocator that only knew the tidy path
#     would ship a library that still points into a directory the next
#     `brew upgrade` deletes.
#   * The build-tree binaries carry an ABSOLUTE LC_RPATH pointing at the
#     build directory. Left in place it is not merely dead weight -- on
#     the developer's own machine it still RESOLVES, so the bundle would
#     load the build tree's libvpipe and appear to work right up until it
#     is opened on a different Mac. It is deleted, not appended to.
#
# Dependencies are chased to a fixpoint, so a newly copied library's own
# dependencies are relocated too, however deep.
#
# Usage: see print_usage_ below. Invoked by apps/macos-app/CMakeLists.txt.

set -euo pipefail

APP=""; SWIFT_EXE=""; CLI=""; WEBUI=""; LIBVPIPE=""; PLIST=""
ENTITLEMENTS=""; RESOURCES=""; FFMPEG_DIR=""; SIGN_ID="-"
SPARKLE_FRAMEWORK=""

print_usage_() {
  cat <<'EOF'
Usage: bundle-app.sh --app OUT.app --swift-exe PATH --cli PATH --webui PATH
                     --libvpipe PATH --plist PATH [--entitlements PATH]
                     [--resources DIR] [--ffmpeg-dir DIR]
                     [--sign-identity ID]

  --app            bundle to create (removed and rebuilt)
  --swift-exe      the GUI executable -> Contents/MacOS/vpipe
  --cli            vpipe            -> Contents/Helpers/vpipe
  --webui          vpipe-web-ui     -> Contents/Helpers/vpipe-web-ui
  --libvpipe       libvpipe.N.dylib -> Contents/Frameworks/
  --plist          a configured Info.plist
  --entitlements   entitlements plist (required for a real identity)
  --resources      directory copied into Contents/Resources/
  --ffmpeg-dir     directory of libav*/libsw* dylibs to bundle. Omit to
                   leave FFmpeg unbundled (dlopen'd from the host).
  --sparkle-framework  path to Sparkle.framework to bundle. Omit for a
                   build with no auto-update.
  --sign-identity  codesign identity; "-" (default) is ad-hoc, which is
                   fine for local runs but cannot be notarized.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --app)           APP="$2"; shift 2 ;;
    --swift-exe)     SWIFT_EXE="$2"; shift 2 ;;
    --cli)           CLI="$2"; shift 2 ;;
    --webui)         WEBUI="$2"; shift 2 ;;
    --libvpipe)      LIBVPIPE="$2"; shift 2 ;;
    --plist)         PLIST="$2"; shift 2 ;;
    --entitlements)  ENTITLEMENTS="$2"; shift 2 ;;
    --resources)     RESOURCES="$2"; shift 2 ;;
    --ffmpeg-dir)    FFMPEG_DIR="$2"; shift 2 ;;
    --sparkle-framework) SPARKLE_FRAMEWORK="$2"; shift 2 ;;
    --sign-identity) SIGN_ID="$2"; shift 2 ;;
    -h|--help)       print_usage_; exit 0 ;;
    *) echo "bundle-app.sh: unknown argument '$1'" >&2
       print_usage_ >&2; exit 2 ;;
  esac
done

# Pairs rather than ${name,,}: /bin/bash on macOS is 3.2, which has no
# case-conversion expansion. Everything else here stays 3.2-clean too.
require_() {
  if [ -z "$2" ]; then
    echo "bundle-app.sh: $1 is required" >&2; exit 2
  fi
}
require_ --app       "$APP"
require_ --swift-exe "$SWIFT_EXE"
require_ --cli       "$CLI"
require_ --webui     "$WEBUI"
require_ --libvpipe  "$LIBVPIPE"
require_ --plist     "$PLIST"

FRAMEWORKS="$APP/Contents/Frameworks"

# Validate the entitlements as XML before doing any work.
#
# codesign hands the file to AMFI, whose parser is stricter than the one
# everybody reaches for to check: `plutil -lint` calls a file with a
# double hyphen inside a comment OK (illegal in XML, and easy to write
# when using -- as a dash), while AMFI rejects it with
# "AMFIUnserializeXML: syntax error near line N" at the very END of the
# build. Fail here instead, with the reason.
if [ -n "$ENTITLEMENTS" ] && command -v xmllint >/dev/null 2>&1; then
  if ! xmllint --noout "$ENTITLEMENTS" 2>/tmp/vpipe-ent-err.$$; then
    echo "bundle-app.sh: $ENTITLEMENTS is not valid XML:" >&2
    sed 's/^/  /' /tmp/vpipe-ent-err.$$ >&2
    echo "  (a '--' inside an XML comment is the usual cause; note that" \
         "plutil -lint does NOT catch it)" >&2
    rm -f /tmp/vpipe-ent-err.$$
    exit 1
  fi
  rm -f /tmp/vpipe-ent-err.$$
fi

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

# otool's output is PER ARCHITECTURE on a universal binary:
#
#   /path/to/bin (architecture x86_64):
#   <TAB>/usr/lib/foo.dylib (compatibility version ...)
#   /path/to/bin (architecture arm64):
#   <TAB>/usr/lib/foo.dylib (compatibility version ...)
#
# so "skip the first line, take field 1" -- the obvious reading, and the
# one this script started with -- turns every architecture HEADER after
# the first into a phantom dependency. Worse, that phantom is an
# absolute path to the file itself, which exists, so a relocator copies
# the binary over itself under its own basename and then finds that copy
# on the next pass. Sparkle ships universal, and that is exactly what it
# did: five bogus "dependencies" and a loop that never settled.
#
# Real dependencies are TAB-indented; headers never are. Filter on that.
deps_of_() {
  otool -L "$1" 2>/dev/null | grep '^	' | awk '{print $1}'
}

# The install-name a Mach-O records for ITSELF, which must never be
# rewritten as though it were a dependency. Executables have none, and
# on a universal binary the answer repeats per architecture, so drop the
# headers and take the first real line (all architectures agree).
own_id_() {
  otool -D "$1" 2>/dev/null | grep -v '(architecture ' | grep -v ':$' \
    | head -1 || true
}

# A path dyld resolves without our help: the OS's own libraries, and
# anything already expressed relative to the loader.
is_system_dep_() {
  case "$1" in
    /usr/lib/*|/System/*) return 0 ;;
    @rpath/*|@loader_path/*|@executable_path/*) return 0 ;;
    *) return 1 ;;
  esac
}

# Every Mach-O currently in the bundle that we are responsible for.
# Recomputed each pass, because the whole point of the loop is that it
# grows.
#
# .framework subtrees are EXCLUDED. A framework arrives already
# self-contained and internally linked through @rpath; there is nothing
# to relocate, and descending into one only risks flattening its
# internal binaries into Frameworks/ as though they were loose
# libraries. Frameworks still get signed -- see the signing section,
# which walks them deliberately and inside-out.
bundle_machos_() {
  find "$APP/Contents/MacOS" "$APP/Contents/Helpers" "$FRAMEWORKS" \
       -type f 2>/dev/null | grep -v '\.framework/' | while read -r f; do
    if file -b "$f" | grep -q "Mach-O"; then echo "$f"; fi
  done
}

# ---------------------------------------------------------------------
# 1. lay out the bundle
# ---------------------------------------------------------------------
echo "bundle-app.sh: assembling $APP"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Helpers" \
         "$FRAMEWORKS" "$APP/Contents/Resources"

cp "$PLIST"     "$APP/Contents/Info.plist"
cp "$SWIFT_EXE" "$APP/Contents/MacOS/vpipe"
cp "$CLI"       "$APP/Contents/Helpers/vpipe"
cp "$WEBUI"     "$APP/Contents/Helpers/vpipe-web-ui"
cp "$LIBVPIPE"  "$FRAMEWORKS/"

if [ -n "$RESOURCES" ] && [ -d "$RESOURCES" ]; then
  cp -R "$RESOURCES/." "$APP/Contents/Resources/"
fi

# FFmpeg, when asked for. Copied BEFORE the relocation pass so its own
# dependencies are chased like everything else -- a minimal LGPL build
# has none outside /usr/lib, but a richer one would.
if [ -n "$FFMPEG_DIR" ]; then
  if [ ! -d "$FFMPEG_DIR" ]; then
    echo "bundle-app.sh: --ffmpeg-dir '$FFMPEG_DIR' is not a directory" >&2
    exit 1
  fi
  # Copy the SYMLINKS TOO (cp -a, which copies a link as a link).
  #
  # FFmpeg installs libavutil.59.39.100.dylib as the real file, with
  # libavutil.59.dylib and libavutil.dylib as symlinks beside it -- and
  # the name recorded in every dependent is the MIDDLE one,
  # @rpath/libavutil.59.dylib. Copying only real files therefore
  # produces a Frameworks directory in which nothing can resolve
  # anything, and the unversioned name the dlopen candidate list asks
  # for is missing as well. The links cost nothing: they are links, not
  # second copies.
  #
  # The glob is libav*/libsw* rather than a hand-listed set, because
  # libavdevice pulls in libavfilter, which is not one of the six vpipe
  # names against and would otherwise arrive only via the relocator --
  # without its symlinks.
  n=0
  for lib in "$FFMPEG_DIR"/libav*.dylib "$FFMPEG_DIR"/libsw*.dylib; do
    [ -e "$lib" ] || continue
    cp -a "$lib" "$FRAMEWORKS/"
    if [ ! -L "$lib" ]; then n=$((n + 1)); fi
  done
  if [ "$n" -eq 0 ]; then
    echo "bundle-app.sh: no libav*/libsw* dylibs in '$FFMPEG_DIR'" >&2
    exit 1
  fi
  echo "bundle-app.sh: bundled $n FFmpeg libraries"
else
  echo "bundle-app.sh: FFmpeg NOT bundled (dlopen'd from the host)"
fi

# Sparkle, when asked for. Copied WITH symlinks preserved (cp -R, not
# -RL): a macOS framework is a symlink farm -- Versions/Current and the
# top-level Sparkle/Resources/Updater.app all point into Versions/B --
# and flattening them produces a bundle that codesign rejects as
# malformed, several steps later and with a much less obvious message.
if [ -n "$SPARKLE_FRAMEWORK" ]; then
  if [ ! -d "$SPARKLE_FRAMEWORK" ]; then
    echo "bundle-app.sh: --sparkle-framework '$SPARKLE_FRAMEWORK' is not" \
         "a directory" >&2
    exit 1
  fi
  cp -R "$SPARKLE_FRAMEWORK" "$FRAMEWORKS/"
  echo "bundle-app.sh: bundled $(basename "$SPARKLE_FRAMEWORK")"
fi

# ---------------------------------------------------------------------
# 2. rpaths
# ---------------------------------------------------------------------
# Delete every absolute build-tree rpath (see the header: on this
# machine it still resolves, which is what makes it dangerous rather
# than merely stale), then point at Frameworks. Both MacOS/ and
# Helpers/ are one level under Contents, so the same relative path
# serves both.
for exe in "$APP/Contents/MacOS/vpipe" \
           "$APP/Contents/Helpers/vpipe" \
           "$APP/Contents/Helpers/vpipe-web-ui"; do
  while read -r rp; do
    [ -n "$rp" ] || continue
    install_name_tool -delete_rpath "$rp" "$exe" 2>/dev/null || true
  done < <(otool -l "$exe" | awk '/LC_RPATH/{f=1} f&&/path /{print $2; f=0}')
  install_name_tool -add_rpath "@executable_path/../Frameworks" "$exe"
done

# A dylib in Frameworks resolving a sibling: @loader_path IS the
# Frameworks directory, so this covers libssl -> libcrypto without
# depending on which executable pulled it in.
#
# Symlinks are skipped throughout. `[ -f ]` is TRUE for a symlink to a
# regular file, so without the -L guard each of FFmpeg's two links per
# library would rewrite the SAME target again, appending duplicate
# rpaths to it.
for lib in "$FRAMEWORKS"/*.dylib; do
  [ -f "$lib" ] || continue
  [ -L "$lib" ] && continue
  install_name_tool -add_rpath "@loader_path" "$lib" 2>/dev/null || true
done

# ---------------------------------------------------------------------
# 3. relocate dependencies to a fixpoint
# ---------------------------------------------------------------------
pass=0
while : ; do
  pass=$((pass + 1))
  changed=0
  while read -r macho; do
    [ -n "$macho" ] || continue
    id="$(own_id_ "$macho")"
    while read -r dep; do
      [ -n "$dep" ] || continue
      [ "$dep" = "$id" ] && continue
      if is_system_dep_ "$dep"; then continue; fi

      base="$(basename "$dep")"
      if [ ! -f "$FRAMEWORKS/$base" ]; then
        if [ ! -f "$dep" ]; then
          echo "bundle-app.sh: WARNING: $macho needs '$dep', which does" \
               "not exist -- leaving the reference alone" >&2
          continue
        fi
        cp "$dep" "$FRAMEWORKS/$base"
        install_name_tool -id "@rpath/$base" "$FRAMEWORKS/$base"
        install_name_tool -add_rpath "@loader_path" \
                          "$FRAMEWORKS/$base" 2>/dev/null || true
        echo "bundle-app.sh: + $base  (from $dep)"
        changed=1
      fi
      install_name_tool -change "$dep" "@rpath/$base" "$macho"
      changed=1
    done < <(deps_of_ "$macho")
  done < <(bundle_machos_)

  [ "$changed" -eq 0 ] && break
  if [ "$pass" -ge 16 ]; then
    echo "bundle-app.sh: dependency relocation did not settle after" \
         "$pass passes -- aborting rather than shipping a half-rewritten" \
         "bundle" >&2
    exit 1
  fi
done
echo "bundle-app.sh: dependencies settled after $pass pass(es)"

# Each Frameworks dylib should call itself @rpath/<name>, including the
# ones copied in step 1 rather than discovered as a dependency.
#
# But ONLY if it does not already use an @rpath id. FFmpeg is built with
# --install-name-dir=@rpath, so libavutil.59.39.100.dylib already calls
# itself @rpath/libavutil.59.dylib -- the SONAME, deliberately not the
# filename. Rewriting that to @rpath/libavutil.59.39.100.dylib would
# rename the library out from under every dependent that asks for it by
# soname.
for lib in "$FRAMEWORKS"/*.dylib; do
  [ -f "$lib" ] || continue
  [ -L "$lib" ] && continue
  cur_id="$(own_id_ "$lib")"
  case "$cur_id" in
    @rpath/*) continue ;;
  esac
  install_name_tool -id "@rpath/$(basename "$lib")" "$lib"
done

# ---------------------------------------------------------------------
# 4. verify before signing
# ---------------------------------------------------------------------
# A dangling absolute reference is the failure this script exists to
# prevent, and it is INVISIBLE on the build machine -- the path still
# resolves here. Catch it now rather than in a bug report from someone
# who does not have Homebrew.
leaks=0
while read -r macho; do
  [ -n "$macho" ] || continue
  id="$(own_id_ "$macho")"
  while read -r dep; do
    [ -n "$dep" ] || continue
    [ "$dep" = "$id" ] && continue
    if is_system_dep_ "$dep"; then continue; fi
    echo "bundle-app.sh: LEAK: $(basename "$macho") -> $dep" >&2
    leaks=$((leaks + 1))
  done < <(deps_of_ "$macho")
done < <(bundle_machos_)

if [ "$leaks" -gt 0 ]; then
  echo "bundle-app.sh: $leaks unrelocated reference(s); refusing to sign" >&2
  exit 1
fi
echo "bundle-app.sh: no unrelocated references"

# Every @rpath dependency must actually EXIST in Frameworks.
#
# The leak check above deliberately treats @rpath as fine -- it is the
# spelling we rewrite everything TO. But "correctly spelled" is not
# "present", and the two came apart badly: the GUI links
# @rpath/Sparkle.framework/Versions/B/Sparkle, and when a build produced
# the executable but not the framework, the bundle passed every check
# here, signed cleanly, notarized, and then failed to launch on someone
# else's machine with a dyld error naming a path that simply was not
# there. It survived this far because THIS machine happened to have a
# copy the loader could reach.
#
# Both rpaths we install (@executable_path/../Frameworks for the
# executables, @loader_path for the dylibs) resolve to Frameworks, so
# one existence test covers every case.
missing=0
while read -r macho; do
  [ -n "$macho" ] || continue
  id="$(own_id_ "$macho")"
  while read -r dep; do
    [ -n "$dep" ] || continue
    [ "$dep" = "$id" ] && continue
    case "$dep" in
      @rpath/*) ;;
      *) continue ;;
    esac
    rel="${dep#@rpath/}"
    if [ ! -e "$FRAMEWORKS/$rel" ]; then
      echo "bundle-app.sh: MISSING: $(basename "$macho") needs '$dep'," \
           "which is not in Frameworks" >&2
      missing=$((missing + 1))
    fi
  done < <(deps_of_ "$macho")
done < <(bundle_machos_)

if [ "$missing" -gt 0 ]; then
  echo "bundle-app.sh: $missing @rpath dependency(ies) absent from the" \
       "bundle; refusing to sign. The app would launch on this machine" \
       "and fail on a clean one." >&2
  exit 1
fi
echo "bundle-app.sh: every @rpath dependency present"

# ---------------------------------------------------------------------
# 5. sign, inside out
# ---------------------------------------------------------------------
# Nested code must be signed before its container: signing the .app
# seals a hash of everything inside it, so a later signature on a helper
# invalidates the outer one.
#
# --options runtime is the hardened runtime, which notarization
# requires. Frameworks take no entitlements; executables do.
sign_args=(--force --timestamp --options runtime --sign "$SIGN_ID")
if [ "$SIGN_ID" = "-" ]; then
  # Ad-hoc cannot carry a trusted timestamp, and a hardened runtime
  # without a real identity buys nothing. Keep it simple so a local
  # build just runs.
  sign_args=(--force --sign "-")
fi

for lib in "$FRAMEWORKS"/*.dylib; do
  [ -f "$lib" ] || continue
  # Signing through a symlink would re-sign the same target once per
  # link. Harmless but slow, and it muddies the log.
  [ -L "$lib" ] && continue
  codesign "${sign_args[@]}" "$lib"
done

# Sparkle before the executables that link it, and its own nested code
# before the framework that contains it. Sparkle ships an updater app
# and two XPC services inside the framework, each of which is separately
# sealed; signing the framework first would seal their OLD signatures
# and then invalidate the lot.
#
# None of these get vpipe's entitlements. They are Sparkle's own
# helpers, and handing an updater the camera and library-validation
# opt-outs would widen its privileges for no reason.
if [ -d "$FRAMEWORKS/Sparkle.framework" ]; then
  SPK="$FRAMEWORKS/Sparkle.framework"
  for v in "$SPK"/Versions/*/; do
    [ -d "$v" ] || continue
    case "$(basename "$v")" in Current) continue ;; esac
    for xpc in "$v"XPCServices/*.xpc; do
      [ -d "$xpc" ] && codesign "${sign_args[@]}" "$xpc"
    done
    [ -d "${v}Updater.app" ] && codesign "${sign_args[@]}" "${v}Updater.app"
    [ -f "${v}Autoupdate" ] && codesign "${sign_args[@]}" "${v}Autoupdate"
    # Sign the VERSIONED directory, not the top-level symlink farm --
    # that is where a framework's seal actually lives.
    codesign "${sign_args[@]}" "$v"
  done
fi

# Expanding an EMPTY array as "${a[@]}" is an unbound-variable error
# under `set -u` in bash 3.2, so fold the entitlements flag into the
# argument list instead of keeping a second array that may be empty.
if [ -n "$ENTITLEMENTS" ]; then
  sign_exe_args=("${sign_args[@]}" --entitlements "$ENTITLEMENTS")
else
  sign_exe_args=("${sign_args[@]}")
fi

for exe in "$APP/Contents/Helpers/vpipe" \
           "$APP/Contents/Helpers/vpipe-web-ui"; do
  codesign "${sign_exe_args[@]}" "$exe"
done

codesign "${sign_exe_args[@]}" "$APP"

codesign --verify --deep --strict --verbose=1 "$APP"
echo "bundle-app.sh: $APP signed with identity '$SIGN_ID'"

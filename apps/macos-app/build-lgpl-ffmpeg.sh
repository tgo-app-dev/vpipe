#!/bin/bash
#
# build-lgpl-ffmpeg.sh -- build a minimal, redistributable FFmpeg for
# bundling inside Vpipe Manager.app.
#
# WHY NOT JUST COPY HOMEBREW'S:
#
#   * LICENSING. Homebrew configures FFmpeg with --enable-gpl (x264,
#     x265, xvid). Shipping that inside this Apache-2.0 project would put
#     the whole distributed application under the GPL. The build here
#     omits every GPL component, so the result is LGPL-2.1, which may be
#     shipped alongside a differently-licensed application as long as it
#     stays a separate, replaceable library -- which it does: vpipe
#     dlopens FFmpeg and never links it, so the relink right users are
#     entitled to is satisfied by construction.
#
#   * SIZE AND FRAGILITY. Homebrew's libavcodec pulls in 31 further
#     Homebrew dylibs (aom, dav1d, rav1e, svtav1, harfbuzz, tesseract,
#     ...). Bundling that transitive closure means shipping tens of
#     megabytes of decoders vpipe never calls, each one more thing to
#     relocate and re-sign.
#
# WHAT IS KEPT: VideoToolbox and AudioToolbox, so H.264/HEVC/ProRes
# decode and encode run on the hardware blocks, plus FFmpeg's own native
# decoders, the muxers/demuxers, and the RTSP/HTTP protocols the capture
# stages use. That covers every codec path vpipe exercises. libmp3lame
# is included because LAME is LGPL, and save-audio's mp3 output wants it.
#
# Output: <prefix>/lib holding six dylibs with no non-system
# dependencies. Point the build at it with:
#
#   cmake -S . -B <build> -DVPIPE_BUNDLE_FFMPEG=ON \
#         -DVPIPE_FFMPEG_BUNDLE_DIR=<prefix>/lib
#
# Usage: build-lgpl-ffmpeg.sh [--prefix DIR] [--version N.N] [--jobs N]

set -euo pipefail

PREFIX="${HOME}/dump/ffmpeg-lgpl"
VERSION="8.1.2"
JOBS="$(sysctl -n hw.ncpu)"

while [ $# -gt 0 ]; do
  case "$1" in
    --prefix)  PREFIX="$2"; shift 2 ;;
    --version) VERSION="$2"; shift 2 ;;
    --jobs)    JOBS="$2"; shift 2 ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "build-lgpl-ffmpeg.sh: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

WORK="${PREFIX}/src"
mkdir -p "$WORK"
cd "$WORK"

TARBALL="ffmpeg-${VERSION}.tar.xz"
if [ ! -f "$TARBALL" ]; then
  echo "==> downloading FFmpeg ${VERSION}"
  curl -fLO "https://ffmpeg.org/releases/${TARBALL}"
fi

# Keep the source next to the build. The LGPL asks a distributor to be
# able to supply the library's source; having it here is the cheapest
# way to always be able to.
if [ ! -d "ffmpeg-${VERSION}" ]; then
  echo "==> unpacking"
  tar xf "$TARBALL"
fi

cd "ffmpeg-${VERSION}"

# libmp3lame is the only MP3 ENCODER FFmpeg can use (its native MP3
# support is decode-only), so without it save-audio's mp3 format has
# nothing to encode with -- wav / aac / m4a are unaffected.
#
# LAME is LGPL, so bundling it is fine; it simply becomes a seventh
# dylib that bundle-app.sh relocates. It is detected rather than assumed
# because FFmpeg's configure does not search Homebrew's prefix, and a
# hard --enable-libmp3lame turns a missing package into a failed build
# twenty minutes in.
#
# The include/lib flags name the LAME keg specifically rather than all
# of /opt/homebrew: opening the whole prefix invites configure to
# discover other libraries and quietly link them in, which is how a
# build meant to be minimal stops being minimal.

# Every library gets an LC_RPATH of @loader_path, so it can find its
# siblings in whatever directory it happens to be sitting in.
#
# This is not optional. --install-name-dir=@rpath below makes each
# library ask for its siblings as @rpath/libavutil.59.dylib, but @rpath
# is resolved from the LC_RPATHs of the loading image -- and a library
# with no LC_RPATH of its own has nothing to resolve against. The result
# is a set of libraries that cannot be dlopen'd from their own directory
# ("no LC_RPATH's found"), which is invisible here, because vpipe's
# loader treats a failed probe as "not present" and quietly moves on to
# the next candidate -- i.e. to Homebrew's build. MEASURED: without
# this, VPIPE_FFMPEG_DIR pointed at this prefix silently loaded
# /opt/homebrew/lib instead, and a whole test run "passed" against the
# wrong FFmpeg.
#
# Inside the app bundle it would happen to work anyway, since the exe
# supplies @executable_path/../Frameworks -- which is exactly what makes
# the standalone case worth fixing rather than relying on.
EXTRA_LDFLAGS="-Wl,-rpath,@loader_path"
EXTRA_CFLAGS=""

LAME_ARGS=()
LAME_PREFIX=""
for p in /opt/homebrew/opt/lame /usr/local/opt/lame; do
  if [ -f "$p/include/lame/lame.h" ]; then LAME_PREFIX="$p"; break; fi
done
if [ -n "$LAME_PREFIX" ]; then
  echo "==> libmp3lame from $LAME_PREFIX (MP3 output enabled)"
  LAME_ARGS=(--enable-libmp3lame)
  EXTRA_CFLAGS="-I${LAME_PREFIX}/include"
  EXTRA_LDFLAGS="$EXTRA_LDFLAGS -L${LAME_PREFIX}/lib"
else
  echo "==> libmp3lame NOT found; building without it."
  echo "    MP3 output will be unavailable (wav/aac/m4a still work)."
  echo "    'brew install lame' and re-run to include it."
fi

echo "==> configuring (LGPL, minimal)"
./configure \
  --prefix="$PREFIX" \
  --enable-shared \
  --disable-static \
  --disable-gpl \
  --disable-nonfree \
  --disable-version3 \
  --enable-videotoolbox \
  --enable-audiotoolbox \
  ${LAME_ARGS[@]+"${LAME_ARGS[@]}"} \
  --extra-cflags="$EXTRA_CFLAGS" \
  --extra-ldflags="$EXTRA_LDFLAGS" \
  --disable-programs \
  --disable-doc \
  --disable-debug \
  --disable-sdl2 \
  --disable-libxcb \
  --disable-xlib \
  --disable-vulkan \
  --disable-libx264 \
  --disable-libx265 \
  --disable-libvpx \
  --disable-libaom \
  --disable-libdav1d \
  --disable-libsvtav1 \
  --disable-librav1e \
  --disable-libopus \
  --disable-libvorbis \
  --disable-libwebp \
  --disable-libfreetype \
  --disable-libfontconfig \
  --disable-libass \
  --disable-libbluray \
  --disable-libssh \
  --disable-libxml2 \
  --disable-libsrt \
  --disable-librist \
  --disable-libzmq \
  --disable-libsnappy \
  --disable-libtheora \
  --disable-libspeex \
  --disable-libopenjpeg \
  --disable-libtesseract \
  --disable-lzma \
  --disable-libjxl \
  --enable-network \
  --enable-protocol=file,http,https,tcp,udp,rtp,tls \
  --enable-demuxer=rtsp \
  --enable-muxer=mp4,mov,matroska,wav,mp3,adts \
  --install-name-dir='@rpath'

echo "==> building"
make -j"$JOBS"
make install

echo
echo "==> result"
ls -1 "$PREFIX"/lib/*.dylib 2>/dev/null | sed 's/^/  /'

# Report anything the bundler will have to drag in alongside these.
# Not a failure: libmp3lame is LGPL and legitimately bundled, and
# bundle-app.sh relocates whatever it finds. It is listed so the set of
# things being shipped is a decision rather than a surprise.
#
# Tab-filtered, not `tail -n +2`: otool prints a header line per
# ARCHITECTURE, and on a universal build the extra headers read as
# phantom dependencies pointing at the file itself.
echo
echo "==> non-system dependencies (these get bundled too)"
found=0
for lib in "$PREFIX"/lib/*.dylib; do
  [ -f "$lib" ] || continue
  [ -L "$lib" ] && continue
  while read -r dep; do
    case "$dep" in
      /usr/lib/*|/System/*|@rpath/*|@loader_path/*) ;;
      *) echo "  $(basename "$lib") -> $dep"; found=1 ;;
    esac
  done < <(otool -L "$lib" | grep '^	' | awk '{print $1}')
done
# `if`, not `[ ... ] && echo`: under `set -e` a failing test as the
# whole command aborts the script.
if [ "$found" -eq 0 ]; then
  echo "  (none -- self-contained)"
fi

# Prove the libraries can actually be dlopen'd from where they sit.
#
# This is the check that matters, and it has to be an actual dlopen:
# vpipe PROBES each candidate with dlopen and treats a failure as
# "absent", so a library that links fine but cannot be loaded does not
# produce an error anywhere -- it produces a silent fallback to whatever
# FFmpeg is next in the search order. That failure mode already bit
# once; it is not hypothetical.
echo
echo "==> dlopen check"
probe_src="$(mktemp -t vpipe_dlprobe).c"
probe_bin="${probe_src%.c}"
cat > "$probe_src" <<'EOF'
#include <dlfcn.h>
#include <stdio.h>
int main(int argc, char** argv) {
  void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (h) { return 0; }
  fprintf(stderr, "%s\n", dlerror());
  return 1;
}
EOF
dl_fail=0
if clang -o "$probe_bin" "$probe_src" 2>/dev/null; then
  for n in libavutil libavcodec libavformat libavdevice \
           libswresample libswscale; do
    lib="$PREFIX/lib/$n.dylib"
    [ -e "$lib" ] || continue
    if "$probe_bin" "$lib" 2>/tmp/vpipe_dlprobe_err.$$; then
      echo "  $n: ok"
    else
      echo "  $n: FAILED"
      sed 's/^/      /' /tmp/vpipe_dlprobe_err.$$
      dl_fail=1
    fi
  done
  rm -f /tmp/vpipe_dlprobe_err.$$
else
  echo "  (could not build the probe; skipping)"
fi
rm -f "$probe_src" "$probe_bin"

if [ "$dl_fail" -ne 0 ]; then
  echo >&2
  echo "build-lgpl-ffmpeg.sh: these libraries cannot be loaded from" >&2
  echo "  their own directory, so vpipe would silently fall back to a" >&2
  echo "  different FFmpeg rather than report an error. Refusing to" >&2
  echo "  call this build usable." >&2
  exit 1
fi

echo
echo "Bundle it with:"
echo "  cmake -S <src> -B <build> -DVPIPE_BUNDLE_FFMPEG=ON \\"
echo "        -DVPIPE_FFMPEG_BUNDLE_DIR=$PREFIX/lib"

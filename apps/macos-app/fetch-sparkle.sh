#!/bin/bash
#
# fetch-sparkle.sh -- download the Sparkle auto-update framework.
#
# Sparkle is NOT vendored in this repository: it is a 15 MB binary
# release, and a binary that large in git is a cost paid by everyone who
# clones, for a feature only the people producing a signed .dmg need.
# Auto-update is optional -- without it the app builds and runs, it just
# never offers an update.
#
# The download is verified against a pinned SHA-256. An auto-updater is
# the one component whose compromise is indistinguishable from a working
# product, so its provenance is checked rather than assumed.
#
# Usage: fetch-sparkle.sh [--prefix DIR] [--version N.N.N]
#
# Then configure with:
#   cmake -S . -B <build> -DVPIPE_SPARKLE_DIR=<prefix>/Sparkle-<version>

set -euo pipefail

PREFIX="${HOME}/dump/sparkle"
VERSION="2.9.5"
# sha256 of Sparkle-2.9.5.tar.xz from the sparkle-project GitHub release.
SHA256="015336b601493e05c237964954bff6191370003d94edefe663724c88840d73cc"
VERIFY=1

while [ $# -gt 0 ]; do
  case "$1" in
    --prefix)    PREFIX="$2"; shift 2 ;;
    --version)   VERSION="$2"; shift 2 ;;
    --sha256)    SHA256="$2"; shift 2 ;;
    --no-verify) VERIFY=0; shift ;;
    -h|--help)   sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "fetch-sparkle.sh: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

DEST="${PREFIX}/Sparkle-${VERSION}"
if [ -d "${DEST}/Sparkle.framework" ]; then
  echo "Sparkle ${VERSION} already at ${DEST}"
  echo "VPIPE_SPARKLE_DIR=${DEST}"
  exit 0
fi

mkdir -p "$DEST"
cd "$PREFIX"

TARBALL="Sparkle-${VERSION}.tar.xz"
URL="https://github.com/sparkle-project/Sparkle/releases/download/${VERSION}/${TARBALL}"

if [ ! -f "$TARBALL" ]; then
  echo "==> downloading $URL"
  curl -fL -o "$TARBALL" "$URL"
fi

if [ "$VERIFY" -eq 1 ]; then
  actual="$(shasum -a 256 "$TARBALL" | awk '{print $1}')"
  if [ "$actual" != "$SHA256" ]; then
    echo "fetch-sparkle.sh: SHA-256 MISMATCH for $TARBALL" >&2
    echo "  expected: $SHA256" >&2
    echo "  actual:   $actual" >&2
    echo >&2
    echo "  Refusing to unpack. If you are deliberately moving to a new" >&2
    echo "  Sparkle version, check the hash against the release page and" >&2
    echo "  update SHA256 at the top of this script (or pass --sha256)." >&2
    echo "  Do NOT pass --no-verify to make this go away." >&2
    exit 1
  fi
  echo "==> sha256 verified"
fi

echo "==> unpacking into $DEST"
tar xf "$TARBALL" -C "$DEST"

if [ ! -d "${DEST}/Sparkle.framework" ]; then
  # The archive layout has moved between releases; find it rather than
  # assuming, and say so if it genuinely is not there.
  found="$(find "$DEST" -maxdepth 3 -name Sparkle.framework -type d \
           | head -1)"
  if [ -z "$found" ]; then
    echo "fetch-sparkle.sh: no Sparkle.framework in the archive" >&2
    exit 1
  fi
  mv "$found" "$DEST/"
fi

echo
echo "Sparkle ${VERSION} ready."
echo "  framework: ${DEST}/Sparkle.framework"
echo "  tools:     ${DEST}/bin (generate_keys, sign_update, generate_appcast)"
echo
echo "Configure with:"
echo "  cmake -S <src> -B <build> -DVPIPE_SPARKLE_DIR=${DEST}"

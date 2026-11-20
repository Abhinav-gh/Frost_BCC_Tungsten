#!/usr/bin/env bash
# Compile the design notes to PDF with a pinned tectonic, so no TeX install is needed.
#
#   scripts/build_docs.sh                         # docs/design/idea_extension_frost.tex
#   scripts/build_docs.sh path/to/other.tex ...
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
VERSION=0.15.0
TOOL=$ROOT/.tools/tectonic-$VERSION/tectonic

if [ ! -x "$TOOL" ]; then
  mkdir -p "$(dirname "$TOOL")"
  curl -fsSL "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%40$VERSION/tectonic-$VERSION-x86_64-unknown-linux-musl.tar.gz" \
    | tar xz -C "$(dirname "$TOOL")"
fi

if [ "$#" -eq 0 ]; then
  set -- docs/design/idea_extension_frost.tex
fi
for tex in "$@"; do
  nice -n 19 "$TOOL" -X compile --outdir "$(dirname "$tex")" "$tex"
done

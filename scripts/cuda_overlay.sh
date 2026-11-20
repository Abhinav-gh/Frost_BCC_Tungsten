#!/usr/bin/env bash
# Build a CUDA_HOME for the pip CUDA toolkit without touching site-packages.
#
# The nvidia cu13 wheels ship versioned libraries only (libcudart.so.13) and no
# lib64 directory. torch's C++/CUDA extension builds (FlashTP, OpenEquivariance's
# JIT) link against unversioned names under $CUDA_HOME/lib64. Adding symlinks
# inside the installed wheel would modify an installed package. Instead, this
# builds an overlay in the gitignored .tools/ that mirrors the toolkit and adds
# those names; CUDA_HOME points at the overlay.
#
#   scripts/cuda_overlay.sh mace     # -> .tools/cuda13-mace
#   scripts/cuda_overlay.sh nequip   # -> .tools/cuda13-nequip
set -eu
cd "$(dirname "$0")/.."
VENV="${1:-mace}"
SRC="$PWD/.venvs/frost-$VENV/lib/python3.12/site-packages/nvidia/cu13"
DST="$PWD/.tools/cuda13-$VENV"
[ -d "$SRC" ] || { echo "no CUDA toolkit wheel at $SRC" >&2; exit 1; }

rm -rf "$DST"
mkdir -p "$DST/lib"
for entry in "$SRC"/*; do
  name=$(basename "$entry")
  case "$name" in lib | lib64) ;; *) ln -s "$entry" "$DST/$name" ;; esac
done
for f in "$SRC"/lib/*; do
  [ -L "$f" ] && continue   # mirror the wheel's own files only
  ln -s "$f" "$DST/lib/$(basename "$f")"
done
# unversioned names, each pointing at the shortest versioned file (the soname)
for f in $(find "$SRC/lib" -maxdepth 1 -type f -name '*.so.[0-9]*' -printf '%f\n' \
           | awk '{ print length, $0 }' | sort -n | cut -d' ' -f2-); do
  base="${f%%.so.*}.so"
  [ -e "$DST/lib/$base" ] || ln -s "$SRC/lib/$f" "$DST/lib/$base"
done
ln -s lib "$DST/lib64"
echo "CUDA_HOME overlay for frost-$VENV: $DST ($(ls "$DST/lib" | wc -l) library entries)"

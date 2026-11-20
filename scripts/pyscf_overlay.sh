#!/usr/bin/env bash
# The library names GPU4PySCF looks for, without touching site-packages (frost-pyscf env, the pyscf: teacher kind).
#
# gpu4pyscf.lib.cusolver loads 'libcusolver' from '<sys.path entry>/nvidia/cusolver/lib' through numpy.ctypeslib, which
# wants an unversioned libcusolver.so; the nvidia-cusolver-cu12 wheel ships only libcusolver.so.11. As for the CUDA
# overlay (scripts/cuda_overlay.sh), the unversioned name lives in the gitignored .tools/: .tools/pyscf-overlay/nvidia/
# cusolver/lib/libcusolver.so -> the wheel's file. The overlay directory goes on the server's PYTHONPATH (its nvidia/
# has no __init__.py, so `import nvidia` still resolves to the wheel's regular package; only the directory search sees it).
set -eu
cd "$(dirname "$0")/.."
SITE=$(ls -d "$PWD"/.venvs/frost-pyscf/lib/python3*/site-packages)
DST="$PWD/.tools/pyscf-overlay"
rm -rf "$DST"
mkdir -p "$DST/nvidia/cusolver/lib"
ln -s "$SITE/nvidia/cusolver/lib/libcusolver.so.11" "$DST/nvidia/cusolver/lib/libcusolver.so"
echo "$DST"

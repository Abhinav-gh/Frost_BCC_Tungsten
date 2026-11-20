#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
VENV="${1:-mace}"
export VIRTUAL_ENV="$PWD/.venvs/frost-$VENV"
export CUDA_HOME="$PWD/.tools/cuda13-$VENV"   # overlay; never links inside site-packages
[ -d "$CUDA_HOME" ] || scripts/cuda_overlay.sh "$VENV"
export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export CUDA_ARCH_LIST="86"          # A6000 = sm_86
export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS=16
cd third_party/flashTP
"$VIRTUAL_ENV/bin/python" -m pip install . --no-build-isolation

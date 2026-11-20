#!/usr/bin/env bash
# Frost environment. Source this, do not execute it.
#   source scripts/env.sh mace      # MACE / TorchSim / OpenEquivariance path
#   source scripts/env.sh nequip    # NequIP / Allegro / SevenNet path
set -u
FROST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FROST_VENV="${1:-mace}"
export FROST_ROOT
export VIRTUAL_ENV="$FROST_ROOT/.venvs/frost-$FROST_VENV"
# CUDA toolkit ships as a pip wheel (no system nvcc on this box). CUDA_HOME is an
# overlay in .tools that adds lib64 and unversioned .so names without modifying
# the installed wheel; scripts/cuda_overlay.sh builds it.
export CUDA_HOME="$FROST_ROOT/.tools/cuda13-$FROST_VENV"
[ -d "$CUDA_HOME" ] || bash "$FROST_ROOT/scripts/cuda_overlay.sh" "$FROST_VENV" >/dev/null
export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"

# A6000 is sm_86. Pin so JIT builds do not probe or fan out.
export TORCH_CUDA_ARCH_LIST="8.6"

# MANDATORY for the mace venv: e3nn 0.4.4 calls torch.load() without
# weights_only=False, which torch >= 2.6 rejects. Harmless in the nequip venv.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# Determinism knobs for the epsilon=0 differential test.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Local secrets (MP_API_KEY, GH_TOKEN, ...). Gitignored; never commit.
[ -f "$FROST_ROOT/.env.local" ] && . "$FROST_ROOT/.env.local"

export FROST_DATA="$FROST_ROOT/data"
echo "frost env: $FROST_VENV | python $($VIRTUAL_ENV/bin/python -V 2>&1 | cut -d' ' -f2) | CUDA_HOME=$CUDA_HOME"

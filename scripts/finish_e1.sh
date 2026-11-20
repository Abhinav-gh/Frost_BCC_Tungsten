#!/usr/bin/env bash
# Wait for the E1 MD matrix, then calibrate and sweep, ending in a Gate 1
# verdict. Calibration waits for the cards because on a shared GPU it competes
# with the MD streams for the last gigabyte.
set -u
cd "$(dirname "$0")/.."
ROOT=$PWD
PY=$ROOT/.venvs/frost-mace/bin/python
export CUDA_HOME=$ROOT/.tools/cuda13-mace   # scripts/cuda_overlay.sh
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONUNBUFFERED=1

running() { ps -eo args= | grep -c "[r]un_trajectory.py --system" || true; }
while [ "$(running)" -gt 0 ]; do sleep 60; done
echo "=== MD matrix finished at $(date -u +%H:%M:%S) UTC ==="
ls -lh data/trajectories/*.h5 2>/dev/null | awk '{print "  "$NF" "$5}'

echo
echo "=== calibrating S_hat ==="
"$PY" experiments/e1_redundancy/simulate_cache.py \
    --calibrate --calib-system si512 --calib-frames 32 2>&1 \
  | grep -viE "futurewarning|warnings.warn|precompiled|for these|from openequi|_Jd,|torch.load|Using Materials|Using float|UserWarning|^ *$|WARNING:root"

echo
echo "=== sweeping tolerances ==="
"$PY" experiments/e1_redundancy/simulate_cache.py --all 2>&1 \
  | grep -viE "futurewarning|warnings.warn|UserWarning|^ *$"
echo "=== E1 complete at $(date -u +%H:%M:%S) UTC ==="

#!/usr/bin/env bash
# Launch the E1 trajectory matrix: 5 systems x 2 temperatures.
#
# Jobs are spread across both GPUs, SLOTS at a time in total, because a single
# MACE MD stream leaves the device mostly idle on ASE overhead (plan 1.3).
#
#   ./scripts/run_e1.sh                 # full matrix, 10k steps, 20 ps equil
#   STEPS=2000 EQUIL=2000 ./scripts/run_e1.sh   # short pilot
set -u
cd "$(dirname "$0")/.."
ROOT=$PWD
PY=$ROOT/.venvs/frost-mace/bin/python
export CUDA_HOME=$ROOT/.tools/cuda13-mace   # scripts/cuda_overlay.sh
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONUNBUFFERED=1

STEPS=${STEPS:-10000}
EQUIL=${EQUIL:-20000}
SLOTS=${SLOTS:-4}
mkdir -p data/logs/e1

# pgrep exits 1 and prints nothing when there is no match, so count lines
# rather than relying on -c, which combined with `|| echo 0` yields "0\n0".
running() { ps -eo args= | grep -c "[r]un_trajectory.py --system" || true; }

i=0
for sys in ${SYSTEMS:-si512 silica1152 li3po4 cu500 water216}; do
  for T in 300 1000; do
    while [ "$(running)" -ge "$SLOTS" ]; do sleep 20; done
    gpu=$(( i % 2 ))
    log="data/logs/e1/${sys}_T${T}.log"
    echo "[launch $(date -u +%H:%M:%S)] $sys @ ${T}K on cuda:$gpu -> $log"
    CUDA_VISIBLE_DEVICES=$gpu nohup "$PY" experiments/e1_redundancy/run_trajectory.py \
        --system "$sys" --temp "$T" --steps "$STEPS" --equil "$EQUIL" \
        > "$log" 2>&1 &
    i=$(( i + 1 ))
    sleep 10
  done
done

while [ "$(running)" -gt 0 ]; do sleep 30; done

# Calibration needs a GPU to itself: on a shared card it competes with the MD
# streams for the last gigabyte and can OOM a run that is most of the way done.
echo "[calibrate $(date -u +%H:%M:%S)] fitting S_hat on si512"
"$PY" experiments/e1_redundancy/simulate_cache.py \
    --calibrate --calib-system si512 --calib-frames 32 \
    > data/logs/calibrate_si512.log 2>&1 || echo "!! calibration failed"

echo "=== E1 trajectory matrix complete at $(date -u +%H:%M:%S) UTC ==="
ls -lh data/trajectories/*.h5 2>/dev/null | awk '{print "  "$NF, $5}'

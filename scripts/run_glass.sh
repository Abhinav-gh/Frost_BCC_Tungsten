#!/usr/bin/env bash
# Silica glass through the whole E1 pipeline, unattended:
# wait for the quench -> trajectories at 300 K and 1000 K -> phi -> force error -> gate.
#
# GPUs on this host are shared with workloads outside the container, so every
# launch waits until the target card has NEED_MIB free rather than starting and
# dying on an out-of-memory error halfway through equilibration.
set -u
cd "$(dirname "$0")/.."
ROOT=$PWD
PY=$ROOT/.venvs/frost-mace/bin/python
export CUDA_HOME=$ROOT/.tools/cuda13-mace   # scripts/cuda_overlay.sh
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONUNBUFFERED=1
NEED_MIB=${NEED_MIB:-8000}

free_on() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1"; }
# first GPU with NEED_MIB free; blocks until one appears
wait_for_gpu() {
  local warned=0
  while :; do
    for g in 0 1; do
      if [ "$(free_on $g)" -ge "$NEED_MIB" ]; then echo "$g"; return; fi
    done
    if [ $warned -eq 0 ]; then
      echo "[$(date -u +%H:%M)] waiting for ${NEED_MIB} MiB free (now: $(free_on 0) / $(free_on 1))" >&2
      warned=1
    fi
    sleep 120
  done
}
launch_traj() {  # temp equil
  local g; g=$(wait_for_gpu)
  echo "[$(date -u +%H:%M)] silica_glass1152 @ $1 K on cuda:$g"
  CUDA_VISIBLE_DEVICES=$g "$PY" experiments/e1_redundancy/run_trajectory.py \
    --system silica_glass1152 --temp "$1" --equil "$2" \
    > "data/logs/e1/silica_glass1152_T$1.log" 2>&1 &
  sleep 90   # let the allocation land before the next free-memory check
}

while ps -eo args= | grep -q "[q]uench_silica.py"; do sleep 60; done
[ -s data/systems/silica_glass1152.xyz ] || { echo "!! quench produced no glass"; exit 1; }
echo "[$(date -u +%H:%M)] glass ready"; grep -a "min distance" data/logs/e1/quench_silica.log

# The quench already annealed 5 ps at 300 K, so 300 K gets a 5 ps top-up.
launch_traj 300 5000
launch_traj 1000 20000
wait
for T in 300 1000; do
  grep -qa "done " "data/logs/e1/silica_glass1152_T$T.log" \
    || { echo "!! glass trajectory at $T K failed"; tail -3 "data/logs/e1/silica_glass1152_T$T.log"; exit 1; }
done
echo "[$(date -u +%H:%M)] glass trajectories done"

"$PY" experiments/e1_redundancy/analyse.py --phi --only "silica_glass1152_T*" --max-steps 2000 \
  > data/logs/e1_phi_glass.log 2>&1
g=$(wait_for_gpu)
CUDA_VISIBLE_DEVICES=$g "$PY" experiments/e1_redundancy/analyse.py --force-error \
  --system silica_glass1152 --temp 300 --frames 5 > data/logs/e1_ferr_glass.log 2>&1

[ -s data/results/e1/ferr_silica_glass1152.json ] \
  || { echo "!! glass force-error step failed"; tail -3 data/logs/e1_ferr_glass.log; exit 1; }
for c in 5 20; do "$PY" experiments/e1_redundancy/analyse.py --gate --ceiling $c; done
echo "=== GLASS E1 COMPLETE $(date -u +%H:%M) ==="

#!/usr/bin/env bash
# Run one heavy Frost job without taking the shared server down.
#
# Earlier runs pushed the 15-minute load average to 658 and made the machine
# unresponsive. The causes were ours: several jobs at once, each PyTorch or
# NumPy process spawning a thread per core (32), large models evaluated on CPU
# in float64, and nothing stopping a job from starting when RAM was already
# short. This wrapper removes each of those.
#
#   scripts/safe_run.sh <name> <cmd...>        e.g.
#   scripts/safe_run.sh ferr_glass .venvs/frost-mace/bin/python experiments/...
#
# Guarantees
#   * one heavy job at a time (a lock); a second launch waits, it does not stack
#   * THREADS threads per job (default 4) for torch, BLAS, OpenMP and numexpr
#   * lowest CPU and IO priority, so interactive use stays responsive
#   * refuses to start below MIN_RAM_GB of available RAM (default 24)
#   * hard wall-clock limit MAX_HOURS (default 6)
#   * output only to data/logs/jobs/<name>.log, never to the terminal
#
# Lanes. LANE=heavy (default) is for compute; LANE=gpu1 is compute on the second GPU. LANE=download is for network and
# disk jobs: they take their own lock, so a multi-hour download does not block
# GPU experiments, and they keep the same priority, timeout and logging.
# The body is one compound command ending in exit: bash parses all of it before running any of it, so editing this
# file while jobs wait or run cannot change what they execute (an in-place edit on 2026-10-20 made two waiting
# wrappers re-read shifted lines and rerun their jobs). Still replace it atomically (write a copy, then mv).
{
set -u
cd "$(dirname "$0")/.."
ROOT=$PWD
NAME=${1:?usage: safe_run.sh <name> <cmd...>}; shift
THREADS=${THREADS:-4}
MIN_RAM_GB=${MIN_RAM_GB:-24}
MAX_HOURS=${MAX_HOURS:-6}
LOG=$ROOT/data/logs/jobs/$NAME.log
LANE=${LANE:-heavy}
LOCK=/tmp/frost-$LANE-job.lock
# LANE=gpu1: a second compute lane pinned to the second GPU, so both GPUs work at once (heavy keeps GPU 0 by
# default). A job that needs both GPUs (two-GPU modes) runs in heavy and must not be queued while gpu1 is busy.
# LANE=heavy is pinned to GPU 0 (a JAX job would otherwise shard onto GPU 1's work); LANE=both takes both GPU locks
# for a job that needs both GPUs (two-GPU modes, a sharded DNS).
LOCK2=""
# flock -o: the lock stays with flock itself for the command's lifetime; children the command spawns (background CPU
# scoring) do not inherit it, so a GPU lane frees when its job ends, not when its scorer does (2026-11-12).
if [ "$LANE" = gpu1 ]; then
  export CUDA_VISIBLE_DEVICES=1
elif [ "$LANE" = both ]; then
  LOCK=/tmp/frost-heavy-job.lock
  LOCK2=/tmp/frost-gpu1-job.lock
  export CUDA_VISIBLE_DEVICES=0,1
elif [ "$LANE" = heavy ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES=0
fi
mkdir -p "$(dirname "$LOG")"

avail_gb=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
if [ "$avail_gb" -lt "$MIN_RAM_GB" ]; then
  echo "refusing $NAME: only ${avail_gb} GB RAM available (need ${MIN_RAM_GB})" | tee "$LOG"
  exit 3
fi

export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
export NUMEXPR_NUM_THREADS=$THREADS TORCH_NUM_THREADS=$THREADS
export CUDA_HOME=$ROOT/.tools/cuda13-mace   # scripts/cuda_overlay.sh; site-packages stays untouched
# as scripts/env.sh does: extensions such as cuEquivariance's link libnvrtc from here
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

{
  echo "[$(date -u +%F\ %H:%M)] $NAME waiting for lock | RAM ${avail_gb} GB | threads $THREADS"
  if [ -n "$LOCK2" ]; then
    flock -o "$LOCK" flock -o "$LOCK2" nice -n 19 ionice -c 3 timeout "${MAX_HOURS}h" "$@"
  else
    flock -o "$LOCK" nice -n 19 ionice -c 3 timeout "${MAX_HOURS}h" "$@"
  fi
  rc=$?   # capture before any command substitution overwrites $?
  echo "[$(date -u +%F\ %H:%M)] $NAME exit=$rc"
} >"$LOG" 2>&1
exit $rc
}

#!/bin/bash
# Steps of a structure-family or defect run for dispatcher jobs (scripts/gpu_dispatcher.py). Runs INSIDE a job that
# already holds a GPU lane (so no safe_run.sh on GPU lanes here); scoring goes to the CPU lane in the background.
#   depth_pipeline.sh family <label> <temp> <system> <tag> <compile|none> <replay args...>
#   depth_pipeline.sh defect <label> <temp> <system>
# Each step is skipped when its output exists, so a rerun resumes. Pre-registration: docs/design/gaps.md E.8.
set -u
cd /home/shashank/Frost
source scripts/env.sh mace >/dev/null 2>&1
P=.venvs/frost-mace/bin/python; RT=experiments/e13_runtime/md_runtime.py; M=data/models/students; R=data/results/e13
NVT5="--ensemble nvt --friction 0.001 --nvt-drift --nvt-bias-clause 0.005 --resonance-fraction 0.45 --never-lose --edge-forces"
score() { ( LANE=check scripts/safe_run.sh "score_$1" env PYTHONPATH=src $P experiments/e13_runtime/score_frames.py "$R/frames_$1.npz" ) & }
prep() {  # label temp system tag compile replay-args -> sets RUN, ST, PER, KC
  local label=$1 temp=$2 sys=$3 tag=$4 comp=$5; shift 5
  RUN="${label}_T${temp}"; ST="$M/${RUN}_${tag}.pt"; COMP=""; [ "$comp" = none ] || COMP="$comp"
  [ -f "data/trajectories/$RUN.h5" ] || $P experiments/e1_redundancy/fast_trajectory.py --system "$sys" --temp "$temp" --label "$label" || return 1
  [ -f "$ST" ] || $P experiments/e5_student/replay.py --only "$RUN" "$@" --tag "$tag" || return 1
  if [ ! -f "$R/${RUN}_period.json" ]; then
    [ -f "$R/frames_${RUN}_pernve.npz" ] || $P $RT --run "$RUN" --student "$ST" $COMP --fixed-k 2 --fixed-uncertified --k-max 2 \
      --fs 10000 --frames --frames-v-fs 2 --edge-forces --tag pernve || return 1
    env PYTHONPATH=src $P experiments/e13_runtime/period.py "$R/frames_${RUN}_pernve.npz" || return 1
  fi
  PER=$(python3 -c "import json;print(round(json.load(open('$R/${RUN}_period.json'))['fastest_period_fs'],2))")
  KC=$(python3 -c "print(max(2, 2*int(0.22*$PER/2)))")
  # the largest interval allowed: never below twice the prior's cap, so it cannot bind under the prior (trap 15; K at
  # 200 K has a cap of 66 and was held to 64 by a fixed limit of 64, 2% behind the fixed interval, 2026-11-13)
  KMAX=$(( 2 * KC > 64 ? 2 * KC : 64 ))
  echo "$RUN: period $PER fs, NVE cap k $KC, k max $KMAX"
}
case "$1" in
  family)
    shift; FSNVT=${FSNVT:-20000}
    prep "$@" || exit 1
    # NVE (amended 2026-11-13 05:40 UTC, gaps.md E.8, before any family after Mo ran): 50 ps, the length the energy criterion
    # needs (trap 16), with the version 3 drift check (W and Mo ran 20 ps under the frozen one, which cannot decide in 20 ps)
    $P $RT --run "$RUN" --student "$ST" $COMP --period-fs "$PER" --k-max "$KMAX" --fixed-k "$KC" --fs 50000 --frames --frames-v-fs 4 \
      --edge-forces --never-lose --drift-sequential ${DRIFT_REWIND:+--drift-rewind} --tag depth_nve50s && score "${RUN}_depth_nve50s"
    $P $RT --run "$RUN" --student "$ST" $COMP --period-fs "$PER" --k-max "$KMAX" --fixed-k "$KC" --fs "$FSNVT" $NVT5 --frames \
      --frames-v-fs 4 --tag depth_nvt5 && score "${RUN}_depth_nvt5"
    ;;
  defect)
    shift
    prep "$1" "$2" "$3" f45c16 "--compile default" --student frost --cutoff 4.5 --channels 16 --iters 5000 || exit 1
    for s in 0 1; do
      $P $RT --run "$RUN" --student "$ST" $COMP --period-fs "$PER" --k-max "$KMAX" --fixed-k "$KC" --fs 100000 --seed $s $NVT5 \
        --frames --frames-v-fs 10 --tag dfx_s$s && \
      { ( LANE=check scripts/safe_run.sh "dfxscore_${RUN}_s$s" bash -c "env PYTHONPATH=src $P experiments/e13_runtime/score_frames.py \
          $R/frames_${RUN}_dfx_s$s.npz; env PYTHONPATH=src $P experiments/e13_runtime/defects.py $R/frames_${RUN}_dfx_s$s.npz" ) & }
    done
    ;;
esac
# no wait: background scorers run on the CPU lane after this job (and its GPU lane) has ended

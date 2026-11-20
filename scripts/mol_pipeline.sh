#!/bin/bash
# A molecule-in-water run (further_expansions_beyond.md 6.2: DMTS-NC's class, MACE-OFF teacher) for dispatcher jobs. Runs
# INSIDE a job that already holds a GPU lane (no safe_run.sh on GPU lanes here); scoring goes to the CPU lane in the
# background. Each step is skipped when its output exists, so a rerun resumes.
#   mol_pipeline.sh <label> <temp> <system> <teacher>        e.g. mol_pipeline.sh ala2off 300 ala2_water off-small
#   PREP_ONLY=1 mol_pipeline.sh ...   the untimed preparation only (trajectory, student, periods)
# Arms (each against its own paired teacher): NVT and NVE with hydrogen mass repartitioning (3 Da, DMTS's protocol, the
# water rematch's flags), and NVT without it (the risk in 6.2: report both). Periods from the teacher's own NVE spectrum,
# with and without HMR (trap 15).
set -u
cd /home/shashank/Frost
source scripts/env.sh mace >/dev/null 2>&1
P=.venvs/frost-mace/bin/python; RT=experiments/e13_runtime/md_runtime.py; M=data/models/students; R=data/results/e13
label=$1; temp=$2; sys=$3; teacher=$4
# STUDENT_TAG: m45c16L (MACE, smooth, ~43% of a MACE-OFF small call per step) or f45c16 (descriptor, compiled, ~4.6x
# cheaper per call; on water it certified 2.01x against m45c16L's 1.54x at equal fidelity, gaps.md 2026-11-14)
TAG=${STUDENT_TAG:-m45c16L}
RUN="${label}_T${temp}"; ST="$M/${RUN}_${TAG}.pt"
case "$TAG" in f*) TRAIN="--student frost --cutoff 4.5 --channels 16 --iters 5000"; COMPILE="--compile default";;
  *) TRAIN="--student mace --cutoff 4.5 --channels 16 --student-backend cueq --iters 15000 --lr 0.005"; COMPILE="";; esac
score() {  # frames tag: structure and spectrum, plus the dipeptide's dihedrals when it is one
  ( LANE=check scripts/safe_run.sh "score_${RUN}_$1" bash -c "env PYTHONPATH=src $P experiments/e13_runtime/score_frames.py \
      $R/frames_${RUN}_$1.npz; case $sys in ala2*) env PYTHONPATH=src $P experiments/e13_runtime/dihedrals.py \
      $R/frames_${RUN}_$1.npz;; esac" ) &
}
period() {  # tag hmr -> prints the period (fs) from an uncertified fixed-2 NVE run's teacher arm
  local tag=$1 hmr=$2 out="$R/${RUN}_${1}_period.json"
  if [ ! -f "$out" ]; then
    [ -f "$R/frames_${RUN}_$tag.npz" ] || $P $RT --run "$RUN" --student "$ST" --fixed-k 2 --fixed-uncertified --k-max 2 \
      --fs 10000 --dt 1 --hmr "$hmr" --frames --frames-v-fs 1 --edge-forces --no-derived-lists --tag "$tag" >&2 || return 1
    env PYTHONPATH=src $P experiments/e13_runtime/period.py "$R/frames_${RUN}_$tag.npz" >&2 || return 1
    mv "$R/${RUN}_period.json" "$out"
  fi
  python3 -c "import json;print(round(json.load(open('$out'))['fastest_period_fs'],2))"
}
[ -f "data/trajectories/$RUN.h5" ] || $P experiments/e1_redundancy/fast_trajectory.py --system "$sys" --temp "$temp" \
  --teacher "$teacher" --label "$label" || exit 1
[ -f "$ST" ] || $P experiments/e5_student/replay.py --only "$RUN" $TRAIN --tag "$TAG" || exit 1
PH=$(period pernve_hmr3 3) || exit 1
P0=$(period pernve_hmr0 0) || exit 1
echo "$RUN: fastest period $PH fs with HMR 3, $P0 fs without"
# PREP_ONLY=1: stop here (the untimed part: trajectory, student, periods), so a timed job runs only the comparisons
[ -n "${PREP_ONLY:-}" ] && exit 0
COMMON="--student $ST $COMPILE --start 9000 --dt 1 --edge-forces --never-lose --no-derived-lists --neighbours auto --epsilon-rel 0.043 --frames"
NVT="--ensemble nvt --friction 0.001 --nvt-drift --nvt-bias-clause 0.005 --resonance-fraction 0.45"
$P $RT --run "$RUN" $COMMON $NVT --fs 50000 --hmr 3 --period-fs "$PH" --k0 2 --k-max 16 --fixed-k 3,4,5 \
  --fixed-uncertified --fine-lattice 4 --tag mol_nvt_hmr3 && score mol_nvt_hmr3
$P $RT --run "$RUN" $COMMON --fs 50000 --hmr 3 --period-fs "$PH" --k-max 16 --fixed-k 3 --fine-lattice 4 \
  --drift-sequential --drift-rewind --tag mol_nve50s_hmr3 && score mol_nve50s_hmr3
$P $RT --run "$RUN" $COMMON $NVT --fs 50000 --period-fs "$P0" --k0 2 --k-max 16 --fixed-k 2,3 --fixed-uncertified \
  --fine-lattice 4 --tag mol_nvt_hmr0 && score mol_nvt_hmr0
# no wait: background scorers run on the CPU lane after this job (and its GPU lane) has ended

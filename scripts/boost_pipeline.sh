#!/bin/bash
# Frost against BoostMD (the validated rebuild) on BoostMD's own protocol: alanine dipeptide in the gas phase with
# MACE-OFF23-M, 1 fs, real masses, Langevin 300 K (here 64 copies, frost.systems.ala2_gas, for a GPU-sized call and pooled
# dihedrals). Stages for dispatcher jobs; runs INSIDE a job that already holds a GPU lane; every step skips what exists.
#   boost_pipeline.sh prep          trajectory (teacher), the descriptor student, the fastest period           untimed
#   boost_pipeline.sh frost         md_runtime: teacher, certified and fixed 2 / 3 / 10 arms, frames, 50 ps   TIMED
#   boost_pipeline.sh boost N       BoostMD with the reference every N steps from the same frame, 50 ps        TIMED
#   boost_pipeline.sh score         score_frames.py and dihedrals.py --copies 64 on every frames file          untimed
#   boost_pipeline.sh student_mace  the smoother MACE student (m45c16L, as on water) on the same trajectory     untimed
# STUDENT_TAG (default f45c16) picks Frost's student for the frost stage (m45c16L runs without --compile); TAG names
# the run (default vsboost). Frost at its best (2026-11-14): the descriptor student's residual sits at the magnitude
# bound at k 2 on this system (staleness 19.39 mean against epsilon 19.31), so the smoother student is run as well.
# Spec, decisions and the pre-registered predictions: experiments/e14_baselines/boostmd_spec.md ("The comparison").
set -u
cd /home/shashank/Frost
source scripts/env.sh mace >/dev/null 2>&1
P=.venvs/frost-mace/bin/python; RT=experiments/e13_runtime/md_runtime.py; M=data/models/students; R=data/results/e13
RUN=ala2gas_T300; STUDENT_TAG=${STUDENT_TAG:-f45c16}; ST="$M/${RUN}_${STUDENT_TAG}.pt"; TAG=${TAG:-vsboost}
FS=${FS:-50000}; case "$STUDENT_TAG" in f*) COMPILE="--compile default";; *) COMPILE="";; esac
# FRICTION (per fs, default 0.001 = 1/ps): the comparison's thermostat; 0.01 (10/ps) is the rebuild's own data and
# stability setting, the competitor-favourable bound (at 1/ps every BoostMD copy blew up within 2 ps, 2026-11-14)
FRICTION=${FRICTION:-0.001}
case "$1" in
  prep)
    [ -f "data/trajectories/$RUN.h5" ] || $P experiments/e1_redundancy/fast_trajectory.py --system ala2_gas64 --temp 300 \
      --teacher off-medium --label ala2gas --equil 5000 --steps 10000 || exit 1
    [ -f "$ST" ] || $P experiments/e5_student/replay.py --only "$RUN" --student frost --cutoff 4.5 --channels 16 \
      --iters 5000 --tag f45c16 || exit 1
    if [ ! -f "$R/${RUN}_period.json" ]; then
      [ -f "$R/frames_${RUN}_pernve.npz" ] || $P $RT --run "$RUN" --student "$ST" --compile default --fixed-k 2 \
        --fixed-uncertified --k-max 2 --fs 10000 --dt 1 --frames --frames-v-fs 1 --edge-forces --no-derived-lists \
        --neighbours auto --tag pernve || exit 1
      env PYTHONPATH=src $P experiments/e13_runtime/period.py "$R/frames_${RUN}_pernve.npz" || exit 1
    fi
    ;;
  frost)
    PER=$(python3 -c "import json;print(round(json.load(open('$R/${RUN}_period.json'))['fastest_period_fs'],2))")
    echo "$RUN: fastest period $PER fs"
    $P $RT --run "$RUN" --student "$ST" $COMPILE --start 9000 --dt 1 --edge-forces --never-lose \
      --no-derived-lists --neighbours auto --epsilon-rel 0.043 --frames --ensemble nvt --friction "$FRICTION" --nvt-drift \
      --nvt-bias-clause 0.005 --resonance-fraction 0.45 --fs "$FS" --period-fs "$PER" --k0 2 --k-max 16 \
      --fixed-k 2,3,10 --fixed-uncertified --fine-lattice 4 --tag "$TAG"
    ;;
  boost)
    N=$2
    $P experiments/e14_baselines/boostmd_md.py --run "$RUN" --start 9000 --fs "$FS" --ref-every "$N" --friction "$FRICTION" \
      --frames-like "$R/frames_${RUN}_${TAG}.npz" --tag "$TAG"
    ;;
  score)
    for f in "$R/frames_${RUN}_${TAG}.npz" "$R"/frames_${RUN}_boostmd*_${TAG}.npz; do
      [ -f "$f" ] || continue
      env PYTHONPATH=src $P experiments/e13_runtime/score_frames.py "$f"
      env PYTHONPATH=src $P experiments/e13_runtime/dihedrals.py --copies 64 "$f"
    done
    ;;
  student_mace)
    S2="$M/${RUN}_m45c16L.pt"
    [ -f "$S2" ] || $P experiments/e5_student/replay.py --only "$RUN" --student mace --cutoff 4.5 --channels 16 \
      --student-backend cueq --iters 15000 --lr 0.005 --tag m45c16L || exit 1
    ;;
  *) echo "unknown stage $1"; exit 2 ;;
esac

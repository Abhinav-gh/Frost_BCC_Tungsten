#!/bin/bash
# Melting points by interface pinning (further_expansions_beyond.md 7.2), stages for dispatcher jobs. Runs INSIDE a job
# that already holds a GPU lane (no safe_run.sh on GPU lanes here). Each stage skips what already exists.
#   melt_pipeline.sh prep <T>                          the pinned two-phase slab at T (prep_slab.py)          untimed
#   melt_pipeline.sh student <T>                       the descriptor student on that slab's trajectory       untimed
#   melt_pipeline.sh period <T>                        the slab's fastest period, NVE teacher spectrum        untimed
#   melt_pipeline.sh run <T> <seed> <tag> <fs> <ST_T>  teacher and Frost arms, NP_zAT, pinned                 TIMED
# LABEL (cuslab) and CELLS ("6 6 20") select the slab; ST_T names the temperature whose student and period serve the run;
# FIXED_K overrides the fixed arms (default the cap), e.g. FIXED_K=10,20 for the melting point's k diagnostic.
set -u
cd /home/shashank/Frost
source scripts/env.sh mace >/dev/null 2>&1
P=.venvs/frost-mace/bin/python; RT=experiments/e13_runtime/md_runtime.py; M=data/models/students; R=data/results/e13
LABEL=${LABEL:-cuslab}; CELLS=${CELLS:-"6 6 20"}; NXY=$(echo "$CELLS" | awk '{print $1","$2}')
stage=$1; T=$2; RUN="${LABEL}_T$T"
pinarg() {  # KAPPA:A of the slab prepared at temperature $1
  python3 -c "import json;d=json.load(open('data/results/e17/${LABEL}_T$1_prep.json'));print(f\"{d['kappa_eV']}:{d['a_pin']}\")"
}
case "$stage" in
  prep)
    [ -f "data/trajectories/$RUN.h5" ] || $P experiments/e17_melting/prep_slab.py --temp "$T" --cells $CELLS --label "$LABEL"
    ;;
  student)
    [ -f "$M/${RUN}_f45c16.pt" ] || $P experiments/e5_student/replay.py --only "$RUN" --student frost --cutoff 4.5 \
      --channels 16 --iters 5000 --tag f45c16
    ;;
  period)
    [ -f "$R/${RUN}_period.json" ] && exit 0
    [ -f "$R/frames_${RUN}_pernve.npz" ] || $P $RT --run "$RUN" --start 9999 --student "$M/${RUN}_f45c16.pt" \
      --compile default --fixed-k 2 --fixed-uncertified --k-max 2 --fs 10000 --frames --frames-v-fs 2 --edge-forces \
      --pin "$(pinarg "$T")" --pin-cells "$NXY" --tag pernve || exit 1
    env PYTHONPATH=src $P experiments/e13_runtime/period.py "$R/frames_${RUN}_pernve.npz"
    ;;
  run)
    seed=$3; tag=$4; fs=$5; ST_T=$6; ST="$M/${LABEL}_T${ST_T}_f45c16.pt"
    PER=$(python3 -c "import json;print(round(json.load(open('$R/${LABEL}_T${ST_T}_period.json'))['fastest_period_fs'],2))")
    KC=$(python3 -c "print(max(2, 2*int(0.22*$PER/2)))"); KMAX=$(( 2 * KC > 64 ? 2 * KC : 64 ))
    echo "$RUN seed $seed: period $PER fs (from T $ST_T), cap k $KC, k max $KMAX, pin $(pinarg "$T")"
    $P $RT --run "$RUN" --start 9999 --student "$ST" --compile default --ensemble npt --npt-axes z \
      --pin "$(pinarg "$T")" --pin-cells "$NXY" --edge-forces --friction 0.001 --nvt-drift --nvt-bias-clause 0.005 \
      --resonance-fraction 0.45 --never-lose --period-fs "$PER" --k-max "$KMAX" --fixed-k "${FIXED_K:-$KC}" --fs "$fs" \
      --seed "$seed" --tag "${tag}_s$seed"
    ;;
  *) echo "unknown stage $stage"; exit 2 ;;
esac

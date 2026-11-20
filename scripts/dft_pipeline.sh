#!/bin/bash
# Certified ab initio MD (further_expansions_beyond.md 7.1): Frost with a DFT teacher (GPU4PySCF through the pyscf:
# kind of frost.external_teacher) on a molecule in vacuum, for dispatcher jobs. Runs INSIDE a job that already holds a GPU
# lane. Each step is skipped when its output exists, so a rerun resumes.
#   dft_pipeline.sh <label> <system> <xc/basis spec> <temp>     e.g. dft_pipeline.sh ethanolpbe ethanol_box pbe/def2-svp/df 300
# The figure of merit is DFT calls saved at matched observables: a call costs ~0.7-1 s on the A6000 against a
# millisecond student, so wall time follows the call count, and the count does not depend on GPU contention (these jobs
# may share a GPU).
set -u
cd /home/shashank/Frost
source scripts/env.sh mace >/dev/null 2>&1
P=.venvs/frost-mace/bin/python; RT=experiments/e13_runtime/md_runtime.py; M=data/models/students; R=data/results/e13
label=$1; sys=$2; spec=$3; temp=$4
RUN="${label}_T${temp}"; T="pyscf:$spec"; ST="$M/${RUN}_f45c16.pt"
# 1. an AIMD trajectory to train on (1 fs, Langevin; 1 ps settling, 4 ps kept)
[ -f "data/trajectories/$RUN.h5" ] || $P experiments/e1_redundancy/fast_trajectory.py --system "$sys" --temp "$temp" \
  --teacher-ext "$T" --label "$label" --equil 1000 --steps 4000 || exit 1
# 2. the cheap descriptor student on DFT labels: the first 3.5 ps trained (every 4th frame labelled), the last 0.5 ps scored
[ -f "$ST" ] || $P experiments/e5_student/replay.py --only "$RUN" --teacher-ext "$T" --student frost --cutoff 4.5 \
  --channels 16 --iters 5000 --train-steps 3500 --label-stride 4 --starts 3600,3750,3900 --tag f45c16 || exit 1
# 3. the fastest period with hydrogens at 4 amu, from the teacher's own NVE spectrum (trap 15), 2 ps
OUT="$R/${RUN}_pernve_hmr3_period.json"
if [ ! -f "$OUT" ]; then
  [ -f "$R/frames_${RUN}_pernve_hmr3.npz" ] || $P $RT --run "$RUN" --teacher-ext "$T" --student "$ST" --compile default \
    --start 3999 --fixed-k 2 --fixed-uncertified --k-max 2 --fs 2000 --dt 1 --hmr 3 --frames --frames-v-fs 1 \
    --neighbours auto --tag pernve_hmr3 || exit 1
  env PYTHONPATH=src $P experiments/e13_runtime/period.py "$R/frames_${RUN}_pernve_hmr3.npz" || exit 1
  mv "$R/${RUN}_period.json" "$OUT"
fi
PER=$(python3 -c "import json;print(round(json.load(open('$OUT'))['fastest_period_fs'],2))")
echo "$RUN: fastest period $PER fs (HMR 3)"
# 4. AIMD against Frost, NVT (DMTS's protocol: 1 fs, hydrogens at 4 amu, friction 1/ps), 8 ps from the trajectory's end
$P $RT --run "$RUN" --teacher-ext "$T" --student "$ST" --compile default --start 3999 --fs 8000 --dt 1 --hmr 3 \
  --ensemble nvt --friction 0.001 --nvt-drift --nvt-bias-clause 0.005 --resonance-fraction 0.45 --never-lose \
  --period-fs "$PER" --k0 2 --k-max 16 --fixed-k 3 --fixed-uncertified --fine-lattice 4 --epsilon-rel 0.043 \
  --neighbours auto --frames --tag aimd_nvt && \
  { ( LANE=check scripts/safe_run.sh "score_${RUN}_aimd_nvt" env PYTHONPATH=src $P experiments/e13_runtime/score_frames.py \
      "$R/frames_${RUN}_aimd_nvt.npz" ) & }

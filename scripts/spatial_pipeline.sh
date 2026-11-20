#!/bin/bash
# Spatial multirate on water/Cu(111) (further_expansions_beyond.md 7.3) for dispatcher jobs; runs INSIDE a job that already
# holds a GPU lane. Steps are skipped when their outputs exist, so a rerun resumes.
#   spatial_pipeline.sh [FS] [TAG]        default 20000 fs, tag sp_s0
set -u
cd /home/shashank/Frost
source scripts/env.sh mace >/dev/null 2>&1
P=.venvs/frost-mace/bin/python; M=data/models/students; R19=data/results/e19
FS=${1:-20000}; TAG=${2:-sp_s0}
RUN=cuwater_T300; ST="$M/${RUN}_f45c16.pt"
# a teacher trajectory to start from and train on (MACE-MP-0 medium, 1 fs, Langevin; 10 ps settling, 10 ps kept)
[ -f "data/trajectories/$RUN.h5" ] || $P experiments/e1_redundancy/fast_trajectory.py --system cu111_water --temp 300 \
  --label cuwater --equil 10000 --steps 10000 || exit 1
[ -f "$ST" ] || $P experiments/e5_student/replay.py --only "$RUN" --student frost --cutoff 4.5 --channels 16 --iters 5000 \
  --train-steps 8000 --label-stride 8 --tag f45c16 || exit 1
$P experiments/e19_spatial/spatial_runtime.py --run "$RUN" --student "$ST" --fs "$FS" --tag "$TAG" && \
  { ( LANE=check scripts/safe_run.sh "score_${RUN}_$TAG" env PYTHONPATH=src $P experiments/e13_runtime/score_frames.py \
      "$R19/frames_${RUN}_$TAG.npz" ) & }

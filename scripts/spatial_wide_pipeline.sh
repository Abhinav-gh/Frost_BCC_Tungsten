#!/bin/bash
# Spatial multirate on a thick-electrode interface (further_expansions_beyond.md 7.3, the wide-interface test, 2026-11-15):
# water on 72 layers of Cu(111) (~150 A) under 30 A of vacuum (frost.systems "cu111_water72", 2,898 atoms), so the exact
# cut-out (the water and the top two layers plus MACE-MP-0's 24 A buffer) is ~28% of the cell; in the 24-layer cell it was
# 97% (it reached the slab's bottom through the periodic image across 12 A of vacuum) and the region calls cost as much as
# full ones. Runs INSIDE a job that already holds a GPU lane; every finished step is skipped.
#   spatial_wide_pipeline.sh prep            teacher trajectory (10 ps settling, 10 ps kept) and the descriptor student  untimed
#   spatial_wide_pipeline.sh run [FS] [TAG] [ARGS...]  teacher, global and spatial arms (24:12 exact, 8:4 thin), graphed
#                                            regions; ARGS go to spatial_runtime.py (e.g. --edge-region)       TIMED
set -u
cd /home/shashank/Frost
source scripts/env.sh mace >/dev/null 2>&1
P=.venvs/frost-mace/bin/python; M=data/models/students; R19=data/results/e19
RUN=cuwaterw_T300; ST="$M/${RUN}_f45c16.pt"
case "$1" in
  prep)
    [ -f "data/trajectories/$RUN.h5" ] || $P experiments/e1_redundancy/fast_trajectory.py --system cu111_water72 --temp 300 \
      --label cuwaterw --equil 10000 --steps 10000 || exit 1
    [ -f "$ST" ] || $P experiments/e5_student/replay.py --only "$RUN" --student frost --cutoff 4.5 --channels 16 \
      --iters 5000 --train-steps 8000 --label-stride 8 --tag f45c16 || exit 1
    ;;
  run)
    FS=${2:-20000}; TAG=${3:-spg_s0}; shift; shift; shift
    $P experiments/e19_spatial/spatial_runtime.py --run "$RUN" --student "$ST" --fs "$FS" --buffers 24:12,8:4 \
      --graphed-region --tag "$TAG" "$@" || exit 1
    # the whole cell, then the water film alone (2,557 of 2,898 atoms are Cu: the whole-cell gate barely sees the film)
    ( LANE=check scripts/safe_run.sh "score_${RUN}_$TAG" env PYTHONPATH=src $P experiments/e13_runtime/score_frames.py \
        "$R19/frames_${RUN}_$TAG.npz" && LANE=check scripts/safe_run.sh "score_${RUN}_${TAG}_water" env PYTHONPATH=src \
        $P experiments/e13_runtime/score_frames.py "$R19/frames_${RUN}_$TAG.npz" --species 1,8 ) &
    ;;
  *) echo "unknown stage $1"; exit 2 ;;
esac

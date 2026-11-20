#!/bin/bash
# Frost on Kanda et al.'s own system: their 12,000-atom anatase TiO2 at their NPT-equilibrated volume (the last frame of
# our run of their code, data/baselines/kanda_t1/val_6), with their 6 A model as Frost's teacher (the kanda: kind, their
# built libdeepmd_c.so through the C API). Pre-registered: docs/design/gaps.md E.10. Runs INSIDE a dispatcher job that
# holds a GPU lane (no safe_run.sh on GPU lanes here); scoring goes to the CPU lane in the background. Each step is
# skipped when its output exists, so a rerun resumes.
set -u
cd /home/shashank/Frost
source scripts/env.sh mace >/dev/null 2>&1
P=.venvs/frost-mace/bin/python; RT=experiments/e13_runtime/md_runtime.py; M=data/models/students; R=data/results/e13
T="kanda:$PWD/third_party/deepmd-mts/examples/scale_mts_tio2/models/6ang.pb"
RUN=tio2kanda12000_T300; TAG=s35c16e1; ST="$M/${RUN}_$TAG.pt"
EXT=(--teacher-ext "$T" --no-derived-lists)
NVT5="--ensemble nvt --friction 0.001 --nvt-drift --nvt-bias-clause 0.005 --resonance-fraction 0.45 --never-lose --edge-forces"
score() { ( LANE=check scripts/safe_run.sh "score_$1" env PYTHONPATH=src $P experiments/e13_runtime/score_frames.py "$R/frames_$1.npz" ) & }

# 1. a 300 K trajectory on their teacher from their equilibrated frame (Langevin 10/ps as every E1 run, 2 ps settling)
[ -f "data/trajectories/$RUN.h5" ] || $P experiments/e1_redundancy/fast_trajectory.py \
  --from-dump data/baselines/kanda_t1/val_6/tio2_anatase.dump --type-map Ti O --teacher-ext "$T" --temp 300 \
  --label tio2kanda12000 --equil 2000 --steps 10000 || exit 1
# 2. the oxides' student (MACE 3.5 A x 16, l 1, 2 layers), labelled by their teacher
[ -f "$ST" ] || $P experiments/e5_student/replay.py --only "$RUN" --student mace --cutoff 3.5 --channels 16 --max-ell 1 \
  --layers 2 --correlation 3 --iters 5000 --teacher-ext "$T" --tag "$TAG" || exit 1
# 3. the fastest period of their teacher's own NVE spectrum (trap 15)
if [ ! -f "$R/${RUN}_period.json" ]; then
  [ -f "$R/frames_${RUN}_pernve.npz" ] || $P $RT --run "$RUN" --student "$ST" "${EXT[@]}" --fixed-k 2 --fixed-uncertified \
    --k-max 2 --fs 10000 --frames --frames-v-fs 2 --edge-forces --tag pernve || exit 1
  env PYTHONPATH=src $P experiments/e13_runtime/period.py "$R/frames_${RUN}_pernve.npz" || exit 1
fi
PER=$(python3 -c "import json;print(round(json.load(open('$R/${RUN}_period.json'))['fastest_period_fs'],2))")
KC=$(python3 -c "print(max(2, 2*int(0.22*$PER/2)))")
echo "$RUN: period $PER fs, NVE cap k $KC"
# 4. certified NVT (fifth version) and NVE, 20 ps, against the paired teacher served the same way, and the cap's fixed k
$P $RT --run "$RUN" --student "$ST" "${EXT[@]}" --period-fs "$PER" --k-max 64 --fixed-k "$KC" --fs 20000 $NVT5 --frames \
  --frames-v-fs 4 --tag kanda_nvt5 && score "${RUN}_kanda_nvt5"
$P $RT --run "$RUN" --student "$ST" "${EXT[@]}" --period-fs "$PER" --k-max 64 --fixed-k "$KC" --fs 20000 --frames \
  --frames-v-fs 4 --edge-forces --never-lose --drift-sequential --tag kanda_nve && score "${RUN}_kanda_nve"
# no wait: background scorers run on the CPU lane after this job (and its GPU lane) has ended

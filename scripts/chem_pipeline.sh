#!/bin/bash
# Reactive chemistry with ANI-1xnr as the teacher (further_expansions_beyond.md 7.4, tier 3: bonds that break) for
# dispatcher jobs, in stages: the teacher costs ~127 ms per call in float64 through the server, so one 10 ps teacher arm at
# 0.25 fs is ~1.4 h and the seeds run as separate jobs on both GPUs (2026-11-14). Runs INSIDE a job that already holds a
# GPU lane; every step is skipped when its output exists.
#   chem_pipeline.sh prep [label] [temp]       trajectory, the descriptor student on its labels, the fastest period
#   chem_pipeline.sh seed S [label] [temp]     teacher, certified and fixed arms from the same frame, seed S, 10 ps NVT
#   chem_pipeline.sh loc S [label] [temp]      the certified arm with the local clause as well (no teacher arm), seed S
#   chem_pipeline.sh unc S [label] [temp]      the cap's fixed interval UNCERTIFIED (no clause), no teacher arm, seed S:
#                                              what holding does to the chemistry (the seed stage's fixed arm keeps its
#                                              clauses and rolled back to k 1 on every interval, 2026-11-15)
#   chem_pipeline.sh species [label] [temp]    the local and uncertified arms merged onto the teacher arms; species pooled
#   chem_pipeline.sh [label] [temp]            all of the above in order
# Default ch4o2ani 3000: 30 CH4 + 60 O2 at 0.4 g/cm^3 (methane combustion, one of ANI-1xnr's own applications). The
# observable is chemistry, not structure: species against time (methane consumed; water, CO, CO2 formed) and bond events
# per ps, teacher against the certified and fixed arms over four seeds (species.py). The local clause (2 x 2 x 2 regions
# of ~34 atoms, frost.locality): a reaction is a local event that the whole-cell RMS dilutes.
set -u
cd /home/shashank/Frost
source scripts/env.sh mace >/dev/null 2>&1
P=.venvs/frost-mace/bin/python; RT=experiments/e13_runtime/md_runtime.py; M=data/models/students; R=data/results/e13
case "${1:-}" in
  prep|species) stage=$1; shift ;;
  seed|loc|unc) stage=$1; S=$2; shift 2 ;;
  *) stage=all ;;
esac
label=${1:-ch4o2ani}; temp=${2:-3000}; RUN="${label}_T${temp}"; DT=0.25
T="ani:third_party/ani-1xnr/model/ani-1xnr"; ST="$M/${RUN}_f45c16.pt"; OUT="$R/${RUN}_period.json"
NVT=(--run "$RUN" --teacher-ext "$T" --student "$ST" --compile default --start 0 --dt "$DT" --ensemble nvt
     --friction 0.001 --nvt-drift --nvt-bias-clause 0.005 --resonance-fraction 0.45 --never-lose --k-max 64 --fs 10000
     --frames --frames-v-fs 50)

do_prep() {
  # a reactive trajectory: 0.5 ps settling, 5 ps kept, every 0.25 fs step written
  [ -f "data/trajectories/$RUN.h5" ] || $P experiments/e1_redundancy/fast_trajectory.py --system ch4_o2_270 \
    --temp "$temp" --teacher-ext "$T" --label "$label" --dt "$DT" --equil 2000 --steps 20000 || return 1
  # the descriptor student on ANI-1xnr's labels (the first 4 ps trained, the last 1 ps scored)
  [ -f "$ST" ] || $P experiments/e5_student/replay.py --only "$RUN" --teacher-ext "$T" --student frost --cutoff 4.5 \
    --channels 16 --iters 5000 --train-steps 16000 --label-stride 8 --starts 17000,18000,19000 --tag f45c16 || return 1
  # the fastest period from the teacher's own NVE spectrum (1 ps, velocities every 0.5 fs; trap 15)
  if [ ! -f "$OUT" ]; then
    [ -f "$R/frames_${RUN}_pernve.npz" ] || $P $RT --run "$RUN" --teacher-ext "$T" --student "$ST" --compile default \
      --start 0 --dt "$DT" --fixed-k 2 --fixed-uncertified --k-max 2 --fs 1000 --frames --frames-v-fs 0.5 \
      --tag pernve || return 1
    env PYTHONPATH=src $P experiments/e13_runtime/period.py "$R/frames_${RUN}_pernve.npz" || return 1
  fi
}

period() { python3 -c "import json;print(round(json.load(open('$OUT'))['fastest_period_fs'],2))"; }

do_seed() {  # teacher against Frost from the same early frame (most fuel still intact)
  local PER KC
  PER=$(period) || return 1
  KC=$(python3 -c "print(max(2, 2*int(0.22*$PER/$DT/2)))")
  echo "$RUN seed $1: fastest period $PER fs, NVE cap k $KC at $DT fs"
  [ -f "$R/frames_${RUN}_rx_s$1.npz" ] || $P $RT "${NVT[@]}" --period-fs "$PER" --fixed-k "$KC" --seed "$1" \
    --tag "rx_s$1"
}

do_loc() {
  local PER
  PER=$(period) || return 1
  [ -f "$R/frames_${RUN}_rxloc_s$1.npz" ] || $P $RT "${NVT[@]}" --period-fs "$PER" --fixed-k "" --local-clause 2 \
    --no-teacher --seed "$1" --tag "rxloc_s$1"
}

do_unc() {
  local PER KC
  PER=$(period) || return 1
  KC=$(python3 -c "print(max(2, 2*int(0.22*$PER/$DT/2)))")
  [ -f "$R/frames_${RUN}_rxunc_s$1.npz" ] || $P $RT "${NVT[@]}" --period-fs "$PER" --fixed-k "$KC" --fixed-uncertified \
    --no-teacher --seed "$1" --tag "rxunc_s$1"
}

do_species() {
  local s
  for s in 0 1 2 3; do
    [ -f "$R/frames_${RUN}_rxm_s$s.npz" ] || $P experiments/e13_runtime/merge_frames.py "$R/frames_${RUN}_rx_s$s.npz" \
      "$R/frames_${RUN}_rxloc_s$s.npz" loc "$R/frames_${RUN}_rxm_s$s.npz" || return 1
    if [ -f "$R/frames_${RUN}_rxunc_s$s.npz" ] && [ ! -f "$R/frames_${RUN}_rxu_s$s.npz" ]; then
      $P experiments/e13_runtime/merge_frames.py "$R/frames_${RUN}_rx_s$s.npz" "$R/frames_${RUN}_rxunc_s$s.npz" unc \
        "$R/frames_${RUN}_rxu_s$s.npz" || return 1
    fi
  done
  env PYTHONPATH=src $P experiments/e21_chemistry/species.py "$R/frames_${RUN}_rx_s0.npz" "$R/frames_${RUN}_rx_s1.npz" \
    "$R/frames_${RUN}_rx_s2.npz" "$R/frames_${RUN}_rx_s3.npz" || return 1
  env PYTHONPATH=src $P experiments/e21_chemistry/species.py "$R/frames_${RUN}_rxm_s0.npz" "$R/frames_${RUN}_rxm_s1.npz" \
    "$R/frames_${RUN}_rxm_s2.npz" "$R/frames_${RUN}_rxm_s3.npz" || return 1
  if [ -f "$R/frames_${RUN}_rxu_s0.npz" ]; then
    env PYTHONPATH=src $P experiments/e21_chemistry/species.py "$R"/frames_${RUN}_rxu_s*.npz
  fi
}

case "$stage" in
  prep) do_prep ;;
  seed) do_seed "$S" ;;
  loc) do_loc "$S" ;;
  unc) do_unc "$S" ;;
  species) do_species ;;
  all) do_prep || exit 1
       for s in 0 1 2 3; do do_seed "$s" || exit 1; done
       for s in 0 1 2 3; do do_loc "$s" || exit 1; done
       do_species ;;
esac

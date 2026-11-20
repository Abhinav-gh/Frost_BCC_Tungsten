#!/bin/bash
# PIMD on the strongest baseline, second version (2026-11-17): the teacher AND the student on bead lists filtered to their
# cutoffs every call, with skins large enough that lists rarely rebuild (the loop profile: at P x T the student's 0.5 A
# skin rebuilt 4.9 of 16 bead lists per step, the teacher's 1.0 A skin 1.6). First the baseline's own skin check (the full
# arm alone at 1.0, 1.5 and 2.0 A, 300 fs each), then the production arms at 1.5 A. Runs inside a job holding a GPU lane.
set -u
cd /home/shashank/Frost
source scripts/env.sh mace >/dev/null 2>&1
P=.venvs/frost-mace/bin/python
BASE="experiments/e15_pimd/pimd_runtime.py --run water216off_T300 --student data/models/students/water216off_T300_m45c16L.pt --beads 16 --contract 2 --dt 0.5 --every 20 --period-fs 9.0 --epsilon-rel 0.043 --tight-teacher"
for sk in 1.0 1.5 2.0; do
  $P $BASE --arms full --fs 300 --equil 50 --teacher-skin $sk --tag tskin_${sk/./p} || exit 1
done
$P $BASE --fs 3500 --equil 500 --arms full,rpc,frost --fixed-k 2,3,4 --teacher-skin 1.5 --tight-student --student-skin 1.5 \
  --tag P16c2_s0_tightv2 || exit 1

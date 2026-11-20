#!/bin/bash
# Does arm order move a PIMD timing? (2026-11-17) The certified arm (k 3.0, zero refusals, the same calls as fixed 3) read
# 2% slower than fixed 3 on both seeds of a clean run, and it always ran right after the heavy full arm. Order A puts the
# certified arm right after the full arm, order B last; each job runs both orders on one GPU (A then B, or B then A, by its
# argument), each attempt behind clean_timing.py, and every arm records its SM clock and temperature (ContentionWatch).
# Usage (inside a job lane): pimd_order_test.sh AB|BA TAG
set -u
cd /home/shashank/Frost
source scripts/env.sh mace >/dev/null 2>&1
P=.venvs/frost-mace/bin/python
SEQ=$1; TAG=$2
BASE="experiments/e15_pimd/pimd_runtime.py --run water216off_T300 --student data/models/students/water216offpimd_T300_f45c16L.pt --beads 16 --contract 1 --dt 0.5 --every 20 --period-fs 8.47 --epsilon-rel 0.043 --tight-teacher --teacher-skin 2.0 --tight-student --student-skin 1.5 --rollback-policy halves --fs 800 --equil 100 --seed 0"
for o in $(echo "$SEQ" | grep -o .); do
  if [ "$o" = A ]; then ARMS=full,frost,fixed3,fixed4; else ARMS=full,fixed3,fixed4,frost; fi
  $P scripts/clean_timing.py "data/results/e15/pimd_water216off_T300_P16c1_order${o}_${TAG}.json" 8 -- \
    $P $BASE --arms "$ARMS" --tag "P16c1_order${o}_${TAG}" || exit 1
done

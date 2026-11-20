#!/bin/bash
# PIMD, third version (2026-11-17): the unified runtime's rollback policy (a refused interval redone as two halves, the
# controller kept on its rung) against the prior's cap (fixed 3) and fixed 4, on the v2 lists (both models filtered; the
# teacher at its best skin, 2.0 A, from the v2 skin check). A 100 fs smoke run of the new policy first. Inside a job lane.
set -u
cd /home/shashank/Frost
source scripts/env.sh mace >/dev/null 2>&1
P=.venvs/frost-mace/bin/python
BASE="experiments/e15_pimd/pimd_runtime.py --run water216off_T300 --student data/models/students/water216off_T300_m45c16L.pt --beads 16 --contract 2 --dt 0.5 --every 20 --period-fs 9.0 --epsilon-rel 0.043 --tight-teacher --teacher-skin 2.0 --tight-student --student-skin 1.5 --rollback-policy halves"
$P $BASE --arms frost --fs 100 --equil 20 --tag v3smoke || exit 1
$P $BASE --arms full,frost --fixed-k 3,4 --fs 3500 --equil 500 --tag P16c2_s0_v3 || exit 1

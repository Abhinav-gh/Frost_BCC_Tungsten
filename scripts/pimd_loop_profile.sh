#!/bin/bash
# The PIMD loop's list overhead, measured in the real loop (2026-11-17): the contraction arm (k 1) on the tight teacher, with
# the student's lists as before (skin 0.5 A), filtered (tight) at skin 0.5, 1.0 and 1.5 A; the full arm once. Runs inside a
# job that holds a GPU lane. Each variant writes data/results/e15/pimd_water216off_T300_loop_<v>.json.
set -u
cd /home/shashank/Frost
source scripts/env.sh mace >/dev/null 2>&1
P=.venvs/frost-mace/bin/python
BASE="experiments/e15_pimd/pimd_runtime.py --run water216off_T300 --student data/models/students/water216off_T300_m45c16L.pt --beads 16 --contract 2 --dt 0.5 --fs 300 --equil 50 --every 20 --period-fs 9.0 --epsilon-rel 0.043 --tight-teacher"
$P $BASE --arms full,rpc --tag loop_v0 || exit 1
$P $BASE --arms rpc --tight-student --student-skin 0.5 --tag loop_v1 || exit 1
$P $BASE --arms rpc --tight-student --student-skin 1.0 --tag loop_v2 || exit 1
$P $BASE --arms rpc --tight-student --student-skin 1.5 --tag loop_v3 || exit 1

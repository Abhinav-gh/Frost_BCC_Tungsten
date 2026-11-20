#!/bin/bash
# Water's fastest period at physical masses from its own NVE teacher spectrum (trap 15; the PIMD runs used 9.0 fs set by
# hand), as ala2off_T300_pernve_hmr0 was measured. Inside a job lane.
set -u
cd /home/shashank/Frost
source scripts/env.sh mace >/dev/null 2>&1
[ -f data/results/e13/frames_water216off_T300_pernve_hmr0.npz ] || .venvs/frost-mace/bin/python experiments/e13_runtime/md_runtime.py --run water216off_T300 --student data/models/students/water216off_T300_m45c16L.pt --compile '' --start 9000 --fs 10000 --dt 0.5 --ensemble nve --pressure-bar 1.0 --tau-p 1000.0 --compressibility 1e-05 --baro-every 10 --npt-equil-fs 0.0 --npt-axes xyz --pin '' --pin-cells '' --pin-hkl '2,0,0;0,2,0' --pin-skip-fs 5000.0 --friction 0.01 --nvt-bias-clause 0.0 --epsilon-rel 0.0 --nvt-kinetic-clause 0.0 --fine-lattice 0.0 --thermostat langevin --nhc-tau 100.0 --ramp '' --mode sync --teacher-device 0 --teacher-path graph --resonance-fraction 0.22 --k0 2 --k-max 2 --fixed-k 2 --monitor '' --fixed-uncertified --teacher-fs 0 --seed 0 --tile 1 --edge-forces --hmr 0.0 --period-fs 0.0 --frames --frames-v-fs 0.5 --teacher-ext '' --neighbours alchemi --no-derived-lists --tag pernve_hmr0 --tight-teacher || exit 1
PYTHONPATH=src .venvs/frost-mace/bin/python experiments/e13_runtime/period.py data/results/e13/frames_water216off_T300_pernve_hmr0.npz

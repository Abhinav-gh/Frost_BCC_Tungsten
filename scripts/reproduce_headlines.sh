#!/usr/bin/env bash
# One command per headline claim in docs/design/CLAIMS.md (reviewer item C22). Every run goes through scripts/safe_run.sh
# (GPU lanes: heavy = GPU 0, gpu1 = GPU 1), writes the result file CLAIMS.md cites, and is the exact command that produced
# it (2026-10-20 / 21).
#
#   scripts/reproduce_headlines.sh --list          # the headlines and their result files
#   scripts/reproduce_headlines.sh <name> [...]    # queue those headlines (LANE=gpu1 to use the second GPU)
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh mace >/dev/null 2>&1
P=.venvs/frost-mace/bin/python
RT=experiments/e13_runtime/md_runtime.py
V=experiments/e6_dynamics/validate.py
M=data/models/students
JX="env -u LD_LIBRARY_PATH -u CUDA_HOME PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false .venvs/frost-mhd/bin/python"
GCM="env -u LD_LIBRARY_PATH -u CUDA_HOME PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false .venvs/frost-gcm/bin/python"
RTC="--fs 50000 --frames --edge-forces --never-lose"
run() { scripts/safe_run.sh "repro_$1" "${@:2}"; }

declare -A WHAT=(
  [md_cu]="Cu 500 certified adaptive, 50 ps NVE through the runtime -> data/results/e13/md_cu500_T300_rt50.json (+ frames, score_frames.py)"
  [md_cu4000]="Cu 4,000 with fidelity -> md_cu4000_T300_rt50.json"
  [md_si]="Si at 2 fs -> md_si512_T300_rt50_dt2.json"
  [md_mgo]="MgO held out -> md_mgo512_T300_rt50.json"
  [md_li]="Li3PO4 -> md_li3po4_T300_rt50.json"
  [md_silica]="silica -> md_silica1152_T300_rt50.json"
  [md_water]="water, DMTS's protocol, never lose -> md_water216off_T300_rt_hmr3_dt1.json"
  [md_energy]="GPU energy per ps, own GPU only (5 ps each) -> md_*_nrg5.json"
  [md_floor]="decorrelated teacher floor (Cu, 1e-2 replicas) -> fidelity_cu500_T300_nve_ref50_v3_q{3,4,5}.json"
  [md_midpoint]="midpoint residual, Cu k = 32 -> fidelity_cu500_T300_nve_mid_k32_p{1,2}.json"
  [cfd_hit_heldout]="forced 3D turbulence on a fresh 384^3 DNS -> data/results/e10/stats3d_w384fresh.json"
  [mhd_heldout]="MHD on held-out reference C, with the classical frontier -> stats3d_w5all_C.json, pooled_w5all_C.json"
  [mhd_canary]="canary width 6 / 24 / 96 members -> data/results/e13/canary_mhd_m{6,24,96}.json"
  [cfd_tgv]="Taylor-Green vortex Re 1600 -> data/results/e10/tgv_re1600.json"
  [gcm_14d]="NeuralGCM deterministic 1.4 deg, 3 and 2 substeps, paired skill -> data/results (forecast_skill tag d14_skill)"
)

case "${1:-}" in
  ""|--list)
    for k in "${!WHAT[@]}"; do printf '  %-16s %s\n' "$k" "${WHAT[$k]}"; done | sort
    exit 0 ;;
esac

for name in "$@"; do
  case "$name" in
    md_cu) run md_cu $P $RT --run cu500_T300 --student $M/cu500_T300_f45c16.pt --compile default --k-max 24 --fixed-k 24 $RTC --tag rt50 ;;
    md_cu4000) run md_cu4000 $P $RT --run cu4000_T300 --student $M/cu500_T300_f45c16.pt --compile default --period-fs 113.7 \
      --k-max 24 --fixed-k 24 $RTC --tag rt50 ;;
    md_si) run md_si $P $RT --run si512_T300 --student $M/si512_T300_f45c16.pt --compile default --dt 2 --k-max 8 --fixed-k 5 $RTC --tag rt50_dt2 ;;
    md_mgo) run md_mgo $P $RT --run mgo512_T300 --student $M/mgo512_T300_s35c16e1.pt --period-fs 52.49 --k-max 10 --fixed-k 10 $RTC --tag rt50 ;;
    md_li) run md_li $P $RT --run li3po4_T300 --student $M/li3po4_T300_s35c16e1.pt --k-max 6 --fixed-k 6 $RTC --tag rt50 ;;
    md_silica) run md_silica $P $RT --run silica1152_T300 --student $M/silica1152_T300_s35c16e1.pt --k-max 6 --fixed-k 6 $RTC --tag rt50 ;;
    md_water) run md_water $P $RT --run water216off_T300 --student $M/water216off_T300_f45c16.pt --compile default --ensemble nvt \
      --friction 0.001 --hmr 3 --dt 1 --period-fs 16.78 --k-max 16 --fixed-k 5 --fixed-uncertified --no-derived-lists $RTC --tag rt_hmr3_dt1 ;;
    md_energy) for r in "cu500_T300 cu500_T300_f45c16 --compile default --k-max 24" "li3po4_T300 li3po4_T300_s35c16e1 --k-max 6"; do
        set -- $r; run md_energy_${1%%_*} $P $RT --run $1 --student $M/$2.pt "${@:3}" --fs 5000 --edge-forces --never-lose --tag nrg5; done ;;
    md_floor) for s in 3 4 5; do run md_floor_q$s $P $V --run cu500_T300 --student $M/cu500_T300_f45c16.pt --arms teacher --fs 50000 \
        --edge-forces --perturb-velocities 1e-2 --seed $s --tag ref50_v3_q$s; done ;;
    md_midpoint) for s in 1 2; do run md_mid_p$s $P $V --run cu500_T300 --student $M/cu500_T300_f45c16.pt --student-compile default \
        --fs 50000 --edge-forces --lagged-lists --device-samples --derived-lists --arms frost \
        --teacher-ref data/results/e6/fidelity_cu500_T300_nve_ref50_v3.json --respa midpoint --k 32 --k-max 32 \
        --perturb-velocities 1e-6 --seed $s --tag mid_k32_p$s; done ;;
    cfd_hit_heldout) run hit_stats $JX experiments/e10_cfd/stats3d.py --data data/cfd/hit384_nu1_fresh.h5 --shells 12 --ref-from 0.25 \
      --classical "56:2:none:0:t" --learned "hit384_32k8sk:8:h,hit384_32k8sk_s1:8:h,hit384_32k8sk_s2:8:h" --tag w384fresh ;;
    mhd_heldout) run mhd_C $JX experiments/e10_cfd/stats3d.py --data data/cfd/mhd256f_s4.h5 --shells 12 \
      --classical "48:2:none:0:t,48:2:cl:1:t,64:3:none:0:t,64:3:cl:1:t,96:5:none:0:t" \
      --learned "mhd_ab_32k8_sw100:8:h,mhd_ab_32k8_sw100_s1:8:h,mhd_ab_32k8_sw100:1:h" --tag w5all_C && \
      LANE=check run mhd_C_pooled $JX experiments/e10_cfd/pooled_stats.py data/results/e10/stats3d_w5all_C.json ;;
    mhd_canary) for m in "6 eval" "24 train:12" "96 train:48"; do set -- $m
        x=""; [ "$2" != eval ] && x="--extra-data data/cfd/mhd256f_s3.h5 --starts $2"
        run mhd_canary_m$1 $JX experiments/e13_runtime/canary_cfd.py --data data/cfd/mhd256f.h5 $x --trusted mhd_ab_32k8_sw100_s1:8 \
          --candidates mhd_ab_32k8_sw100_s1:16,mhd_ab_32k8_sw100_s1:1 --tag mhd_m$1; done ;;
    cfd_tgv) run tgv $JX experiments/e10_cfd/tgv.py --n-dns 384 --sub 8 --tag re1600 ;;
    gcm_14d) run gcm_14d $GCM experiments/e11_gcm/forecast_skill.py --model data/gcm/models/deterministic_1_4_deg.pkl --members 0 \
      --data-tag jf2020 --days 5 --fourier fft --variants "released;sub3+damp;sub2+damp" --tag d14_skill ;;
    *) echo "unknown headline: $name (see --list)"; exit 2 ;;
  esac
done

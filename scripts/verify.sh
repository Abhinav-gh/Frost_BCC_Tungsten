#!/usr/bin/env bash
# End-to-end verification that the Frost setup is complete and working.
# Exits non-zero if anything required is missing or broken.
#
#   ./scripts/verify.sh          full run, including the GPU probes
#   ./scripts/verify.sh --fast   skip the GPU probes
cd "$(dirname "$0")/.."
ROOT=$PWD
FAST=${1:-}
PASS=0; FAIL=0; WARN=0

ok()   { printf "  \033[32mok  \033[0m %s\n" "$1"; PASS=$((PASS+1)); }
bad()  { printf "  \033[31mFAIL\033[0m %s\n" "$1"; FAIL=$((FAIL+1)); }
warn() { printf "  \033[33mwarn\033[0m %s\n" "$1"; WARN=$((WARN+1)); }
sec()  { printf "\n== %s ==\n" "$1"; }

sec "hardware"
n_gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
[ "$n_gpu" -eq 2 ] && ok "2 GPUs present" || bad "expected 2 GPUs, found $n_gpu"
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "A6000" \
  && ok "RTX A6000" || bad "not an A6000"
nvidia-smi topo -m 2>/dev/null | grep -q "PHB" \
  && ok "PHB interconnect (no NVLink, as the proposal assumes)" \
  || warn "interconnect is not PHB; re-read plan section 9.5"
free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$free_mib" -gt 40000 ]; then ok "GPU0 free: ${free_mib} MiB"
else warn "GPU0 free only ${free_mib} MiB; host processes still hold memory (plan M1)"; fi

sec "environments"
for v in mace nequip; do
  P="$ROOT/.venvs/frost-$v/bin/python"
  if [ ! -x "$P" ]; then bad "venv frost-$v missing"; continue; fi
  t=$("$P" -c "import torch;print(torch.__version__)" 2>/dev/null)
  c=$("$P" -c "import torch;print(torch.cuda.is_available())" 2>/dev/null)
  [ "$c" = "True" ] && ok "frost-$v: torch $t, CUDA available" \
                    || bad "frost-$v: torch $t, CUDA NOT available"
  CU="$ROOT/.tools/cuda13-$v"
  [ -e "$CU/lib64/libcudart.so" ] && ok "frost-$v: CUDA_HOME overlay has lib64 and .so names" \
    || bad "frost-$v: no CUDA_HOME overlay; run scripts/cuda_overlay.sh $v (plan 3.3)"
  [ -z "$(find "$ROOT/.venvs/frost-$v/lib/python3.12/site-packages/nvidia" -type l 2>/dev/null | head -1)" ] \
    && ok "frost-$v: CUDA wheels unmodified (no links added in site-packages)" \
    || bad "frost-$v: symlinks inside site-packages/nvidia; use the overlay (plan 3.3)"
  nv=$("$CU/bin/nvcc" --version 2>/dev/null | grep -o "V13\.[0-9.]*")
  px=$("$CU/bin/ptxas" --version 2>/dev/null | grep -o "V13\.[0-9.]*")
  [ -n "$nv" ] && [ "$nv" = "$px" ] && ok "frost-$v: nvcc and ptxas agree ($nv)" \
    || bad "frost-$v: nvcc/ptxas version skew ($nv vs $px) -- plan section 3.5"
done

sec "kernel backends"
export CUDA_HOME="$ROOT/.tools/cuda13-mace"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 TORCH_CUDA_ARCH_LIST=8.6
"$ROOT/.venvs/frost-mace/bin/python" - <<'PY' 2>/dev/null
import sys, torch
bad = []
try:
    import cuequivariance_torch, cuequivariance as cue
    print("  ok   cuEquivariance importable")
except Exception as e: print(f"  FAIL cuEquivariance: {e}"); bad.append(1)
try:
    import openequivariance as oeq
    assert oeq.TensorProductConv is not oeq.TensorProduct
    print("  ok   OpenEquivariance: unfused TensorProduct is separate (Gate 0)")
except Exception as e: print(f"  FAIL OpenEquivariance: {e}"); bad.append(1)
try:
    import flashTP_e3nn
    print("  ok   FlashTP importable (fused only; split-scatter per plan 5.0)")
except Exception as e: print(f"  FAIL FlashTP: {e}"); bad.append(1)
sys.exit(len(bad))
PY
[ $? -eq 0 ] && PASS=$((PASS+3)) || FAIL=$((FAIL+1))

sec "no patches (CLAUDE.md: public APIs only)"
for v in mace nequip mhd gcm; do
  P="$ROOT/.venvs/frost-$v/bin/python"
  [ -x "$P" ] || continue
  line=$("$P" "$ROOT/scripts/check_unpatched.py" installed 2>/dev/null | tail -1)
  case "$line" in ok*) ok "frost-$v: ${line#ok }" ;; *) bad "frost-$v: ${line:-check did not run}" ;; esac
done
line=$("$ROOT/.venvs/frost-mace/bin/python" "$ROOT/scripts/check_unpatched.py" code 2>/dev/null | tail -1)
case "$line" in ok*) ok "${line#ok }" ;; *) bad "${line:-code check did not run}" ;; esac

sec "models"
count_ge() { [ "$(ls -1 $1 2>/dev/null | wc -l)" -ge "$2" ]; }
count_ge "$HOME/.cache/mace/*model*" 5 \
  && ok "MACE checkpoints: $(ls -1 $HOME/.cache/mace | wc -l) files" \
  || bad "MACE checkpoints missing"
count_ge "$ROOT/data/models/nequip/*.nequip.zip" 4 \
  && ok "Allegro/NequIP packages: $(ls -1 $ROOT/data/models/nequip/*.zip 2>/dev/null | wc -l)" \
  || bad "Allegro/NequIP packages missing"
[ -f "$ROOT/data/models/nequip/compiled/Allegro-MP-L-0.1_ase.nequip.pt2" ] \
  && ok "Allegro AOTInductor artifact" || bad "Allegro not compiled"
[ -f "$ROOT/data/models/mgn/vortex_shedding_mgn/model.pt" ] \
  && ok "MeshGraphNet negative-control checkpoint" || bad "MGN checkpoint missing"
"$ROOT/.venvs/frost-nequip/bin/python" -c "
import sevenn, os, glob
d=os.path.join(os.path.dirname(sevenn.__file__),'pretrained_potentials')
n=len(glob.glob(d+'/*/*.pth')); print(n)" 2>/dev/null | grep -q "[1-9]" \
  && ok "SevenNet pretrained potentials in-wheel" || warn "SevenNet potentials not found"

sec "datasets"
d_gb() { du -sm "$1" 2>/dev/null | awk '{printf "%.1f", $1/1024}'; }
for set in rmd17 md17 md22 3bpa mptrj; do  # gns and mgn deleted 2026-11-12 (user request)
  g=$(d_gb "$ROOT/data/datasets/$set")
  if [ -z "$g" ] || [ "$g" = "0.0" ]; then bad "$set missing"
  else ok "$set: ${g} GB"; fi
done
[ -d "$ROOT/data/datasets/spice2" ] && [ "$(du -sm $ROOT/data/datasets/spice2 2>/dev/null | cut -f1)" -gt 100 ] \
  && ok "spice2 present" || warn "spice2 not downloaded (needs GH_TOKEN; not needed before week 13)"

sec "simulation cells"
"$ROOT/.venvs/frost-mace/bin/python" -m frost.systems 2>/dev/null \
  | grep -v SKIPPED | sed 's/^/  ok   /'
nsys=$("$ROOT/.venvs/frost-mace/bin/python" -m frost.systems 2>/dev/null | grep -vc SKIPPED)
# compare against the registry, not a literal: the silica glass made it six
nexp=$("$ROOT/.venvs/frost-mace/bin/python" -c "from frost.systems import E1_SYSTEMS; print(len(E1_SYSTEMS))" 2>/dev/null)
[ "$nsys" -eq "${nexp:-0}" ] && PASS=$((PASS+1)) || { bad "only $nsys/${nexp:-?} cells build"; }

sec "tests"
"$ROOT/.venvs/frost-mace/bin/python" -m pytest "$ROOT/tests" -q 2>&1 | tail -1 | sed 's/^/  /'
"$ROOT/.venvs/frost-mace/bin/python" -m pytest "$ROOT/tests" -q >/dev/null 2>&1 \
  && { ok "pytest suite green"; } || bad "pytest suite red"
# the JAX CFD code lives in frost-mhd, which has no pytest: plain checks, CPU, tiny grids
line=$(env -u LD_LIBRARY_PATH -u CUDA_HOME JAX_PLATFORMS=cpu PYTHONPATH="$ROOT/src" \
  "$ROOT/.venvs/frost-mhd/bin/python" "$ROOT/tests/jax/check_cfd.py" 2>/dev/null | tail -1)
case "$line" in ok*) ok "frost.cfd checks: ${line#ok }" ;; *) bad "frost.cfd checks: ${line:-did not run}" ;; esac
# NeuralGCM (G1) lives in frost-gcm: the same plain CPU checks, tiny models only
if [ -x "$ROOT/.venvs/frost-gcm/bin/python" ]; then
  line=$(env -u LD_LIBRARY_PATH -u CUDA_HOME JAX_PLATFORMS=cpu PYTHONPATH="$ROOT/src" \
    "$ROOT/.venvs/frost-gcm/bin/python" "$ROOT/tests/jax/check_gcm.py" 2>/dev/null | tail -1)
  case "$line" in ok*) ok "frost.gcm checks: ${line#ok }" ;; *) bad "frost.gcm checks: ${line:-did not run}" ;; esac
fi

if [ "$FAST" != "--fast" ]; then
sec "theorem probes (GPU)"
  for probe in probe_zero_closure probe_force_error; do
    if timeout 1800 "$ROOT/.venvs/frost-mace/bin/python" "$ROOT/experiments/probes/$probe.py" \
         >"/tmp/frost_$probe.log" 2>&1; then
      ok "$probe"
    else
      bad "$probe (see /tmp/frost_$probe.log)"
    fi
  done
  grep -h "CONFIRMED" /tmp/frost_probe_force_error.log 2>/dev/null | sed 's/^/       /'
fi

sec "summary"
printf "  %d passed, %d warnings, %d failures\n" "$PASS" "$WARN" "$FAIL"
df -h / | tail -1 | awk '{print "  disk free: "$4}'
[ "$FAIL" -eq 0 ] && echo "  SETUP COMPLETE" || echo "  SETUP INCOMPLETE"
exit "$FAIL"

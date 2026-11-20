#!/usr/bin/env bash
# Build Athena++ reference executables for the MHD workloads (note section 6.4).
#
# The PyTorch finite-volume solver used in W1-W3 is validated against Athena++
# on standard problems. Athena++ compiles one problem generator per executable,
# so each problem gets its own binary under .tools/athena/bin/, and the commit
# is recorded next to them.
#
#   LANE=build scripts/safe_run.sh build_athena scripts/build_athena.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
SRC=$ROOT/.tools/athena/src
BIN=$ROOT/.tools/athena/bin
JOBS=${JOBS:-4}
mkdir -p "$BIN"
command -v g++ >/dev/null || { echo "g++ not found"; exit 2; }
command -v make >/dev/null || { echo "make not found"; exit 2; }

if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 https://github.com/PrincetonUniversity/athena.git "$SRC"
fi
cd "$SRC"

build() {  # name, then configure.py arguments
  local name=$1; shift
  if [ -x "$BIN/athena_$name" ]; then
    echo "= athena_$name (present)"
    return
  fi
  python3 configure.py "$@" > "$BIN/configure_$name.log"
  make clean > /dev/null
  make -j"$JOBS" > "$BIN/make_$name.log" 2>&1
  cp bin/athena "$BIN/athena_$name"
  echo "+ athena_$name"
}

build orszag_tang     --prob orszag_tang -b --flux hlld
build blast_mhd       --prob blast       -b --flux hlld
build field_loop      --prob field_loop  -b --flux hlld
build linear_wave_mhd --prob linear_wave -b --flux hlld

git rev-parse HEAD > "$BIN/ATHENA_COMMIT"
echo "done: $(ls "$BIN" | grep athena_ | tr '\n' ' ')"

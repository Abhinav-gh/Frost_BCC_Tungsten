#!/usr/bin/env bash
# Flag E1 runs that are alive but not progressing.
#
# A wedged MD run and a slow one look identical from the process table: both
# are alive, both hold GPU memory. Two signals separate them, and it takes both
# to be confident:
#
#   * the log stops growing, and
#   * GPU utilisation falls to zero.
#
# A quiet log on a busy GPU is just a run that does not print often, which is
# what the old silent equilibration looked like. A quiet log on an idle GPU is
# a run doing no work, which is what a degenerate starting geometry produces.
#
#   ./scripts/watchdog.sh            # one report
#   ./scripts/watchdog.sh --watch    # every 10 minutes until all runs finish
set -u
cd "$(dirname "$0")/.."
STALL_MIN=${STALL_MIN:-15}
IDLE_PCT=${IDLE_PCT:-5}

report() {
  local now busy alive
  now=$(date -u +%H:%M:%S)
  # Busy if any GPU is doing real work; a wedged run leaves both idle.
  busy=0
  while read -r u; do
    [ "${u:-0}" -gt "$IDLE_PCT" ] && busy=1
  done < <(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)

  echo "[$now] gpu: $(nvidia-smi --query-gpu=index,utilization.gpu,memory.free \
        --format=csv,noheader | tr '\n' '|' )"
  alive=0
  for log in data/logs/e1/*.log; do
    [ -e "$log" ] || continue
    name=$(basename "$log" .log)
    sys=${name%_T*}; temp=${name##*_T}
    if ! ps -eo args= | grep -q "[r]un_trajectory.py --system $sys --temp $temp"; then
      if grep -qa "done " "$log"; then echo "  $name: finished"
      else echo "  $name: not running"; fi
      continue
    fi
    alive=$(( alive + 1 ))
    age_min=$(( ( $(date +%s) - $(stat -c %Y "$log") ) / 60 ))
    last=$(grep -aE "equilibrating|step [0-9]+/|done " "$log" | tail -1 \
           | sed 's/^ *//' | cut -c1-52)
    if [ "$age_min" -ge "$STALL_MIN" ] && [ "$busy" -eq 0 ]; then
      echo "  $name: WEDGED -- ${age_min} min silent and the GPUs are idle." \
           "Check the cell with tests/unit/test_systems.py"
    elif [ "$age_min" -ge "$STALL_MIN" ]; then
      echo "  $name: quiet for ${age_min}m but the GPUs are busy | ${last:-starting up}"
    else
      echo "  $name: ${age_min}m since output | ${last:-starting up}"
    fi
  done
  echo "  ($alive running)"
  [ "$alive" -gt 0 ]
}

if [ "${1:-}" = "--watch" ]; then
  while report; do sleep 600; done
  echo "all E1 runs finished"
  exit 0
fi
report >/dev/null 2>&1 || true
report
exit 0

#!/usr/bin/env bash
# What is running, what is waiting, and what finished — across everyone's jobs.
#
# safe_run.sh serialises every heavy job behind one lock per lane, so two people
# (or a person and an agent) queueing work see each other's jobs only as delay.
# This prints the lane's state from the job logs, which is the one place every
# launch records itself.
#
#   scripts/queue.sh            # running, waiting, and the last 10 finished
#   scripts/queue.sh 30         # ... and the last 30 finished
set -u
cd "$(dirname "$0")/.."
LOGS=data/logs/jobs
RECENT=${1:-10}

stamp() { grep -m1 -oE '^\[[^]]+\]' "$1" | tr -d '[]'; }
finished() { grep -oE "exit=[0-9]+" "$1" | tail -1; }

# /proc/locks names the pid that actually holds each flock; everything else with
# the same name in the process table is a wrapper (the caller's shell, safe_run).
holder_of() {
  local lock=$1 inode
  [ -e "$lock" ] || return 0
  inode=$(stat -c %i "$lock" 2>/dev/null) || return 0
  awk -v ino=":$inode" '$2 == "FLOCK" && $6 ~ ino {print $5; exit}' /proc/locks
}

name_of() {  # walk up from the lock holder to the safe_run that names the job
  local pid=$1 depth=0
  while [ -n "$pid" ] && [ "$pid" != "1" ] && [ "$depth" -lt 6 ]; do
    local args name
    args=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
    name=$(sed -nE 's#.*safe_run\.sh +([A-Za-z0-9_.:-]+).*#\1#p' <<<"$args")
    if [ -n "$name" ]; then printf '%s' "$name"; return 0; fi
    pid=$(awk '{print $4}' "/proc/$pid/stat" 2>/dev/null)
    depth=$((depth + 1))
  done
}

printf '== lanes ==\n'
for lane in heavy gpu1 download build check; do
  lock=/tmp/frost-$lane-job.lock
  pid=$(holder_of "$lock")
  if [ -n "${pid:-}" ]; then
    name=$(name_of "$pid")
    log=$LOGS/${name:-unknown}.log
    since=$([ -f "$log" ] && stamp "$log" || echo '?')
    printf '  %-9s RUNNING %-32s pid %-7s since %s\n' "$lane" "${name:-?}" "$pid" "$since"
  else
    printf '  %-9s idle\n' "$lane"
  fi
done

# anything launched but not holding a lock is queued behind the job above
printf '\n== waiting ==\n'
waiting=$(pgrep -af 'safe_run\.sh ' | grep -v 'queue\.sh' |
  sed -nE 's#.*safe_run\.sh +([A-Za-z0-9_.:-]+).*#\1#p' | grep -v '[$]' | sort -u)
for lane in heavy gpu1 download build check; do
  pid=$(holder_of "/tmp/frost-$lane-job.lock")
  [ -n "${pid:-}" ] && waiting=$(grep -vx "$(name_of "$pid")" <<<"$waiting" || true)
done
shown=0
for name in $waiting; do
  # a real launch writes its log at once; a name without one is a caller's shell
  # whose command line still holds an unexpanded loop variable
  [ -f "$LOGS/$name.log" ] || continue
  printf '  %s\n' "$name"; shown=$((shown + 1))
done
[ "$shown" -gt 0 ] || printf '  (nothing)\n'

printf '\n== last %s touched ==\n' "$RECENT"
for log in $(ls -t $LOGS/*.log 2>/dev/null | head -"$RECENT"); do
  name=$(basename "$log" .log)
  code=$(finished "$log")
  case "$code" in
    exit=0) state='ok     ' ;;
    exit=*) state="FAILED " ;;
    *)      state='running' ;;
  esac
  printf '  %s %-34s %s\n' "$state" "$name" "$(stamp "$log")"
done

printf '\nfailures keep their traceback: less %s/<name>.log\n' "$LOGS"

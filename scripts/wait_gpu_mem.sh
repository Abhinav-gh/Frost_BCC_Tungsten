#!/bin/bash
# wait_gpu_mem.sh GB MAX_MIN CMD...: wait until this job's GPU (CUDA_VISIBLE_DEVICES, a physical index, as the dispatcher
# sets it) has GB GiB free, then run CMD (exec). The server is shared: on 2026-11-15 another tenant's process held 26 GB of
# GPU 0 and 42 GB of GPU 1 for hours, and three of our jobs started into the remaining memory and died of OOM within a
# minute. Gives up with exit 3 after MAX_MIN minutes.
need=$1; maxmin=$2; shift 2
gpu=${CUDA_VISIBLE_DEVICES:-0}
gpu=${gpu%%,*}
for ((i = 0; i < maxmin * 2; i++)); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' ')
  if [ -n "$free" ] && [ "$free" -ge $((need * 1024)) ]; then
    [ "$i" -gt 0 ] && echo "wait_gpu_mem: GPU $gpu has $free MiB free after $((i / 2)) min"
    exec "$@"
  fi
  [ "$i" -eq 0 ] && echo "wait_gpu_mem: GPU $gpu has $free MiB free, waiting for ${need} GiB"
  sleep 30
done
echo "wait_gpu_mem: GPU $gpu never had ${need} GiB free in ${maxmin} min"
exit 3

#!/usr/bin/env bash
# Large optional sets, sequential, in order of experimental value.
set -u
cd "$(dirname "$0")/.."
PY=.venvs/frost-mace/bin/python
for item in gns_train mgn_train_small mgn_train_rest gns_3d; do
  echo "##### $item  $(date -u +%H:%M:%S) #####"
  $PY scripts/fetch_data.py --tier C --only "$item" || echo "!! $item failed"
  df -h / | tail -1
done
echo "##### big fetch complete #####"

#!/usr/bin/env bash
# One-glance status of the Frost provisioning state.
cd "$(dirname "$0")/.."
echo "== GPUs =="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo
echo "== venvs =="
for v in .venvs/*/; do
  n=$(basename "$v")
  t=$("$v/bin/python" -c "import torch;print(torch.__version__)" 2>/dev/null || echo "broken")
  echo "  $n  torch=$t"
done
echo
echo "== models =="
du -sh ~/.cache/mace 2>/dev/null | sed 's/^/  mace  /'
du -sh data/models/* 2>/dev/null | sed 's/^/  /'
echo
echo "== datasets =="
du -sh data/datasets/* 2>/dev/null | sed 's/^/  /'
echo "  ---"; du -sh data/datasets 2>/dev/null | sed 's/^/  total /'
df -h / | tail -1 | awk '{print "  disk free: "$4}'
echo
echo "== running jobs =="
pgrep -fa "fetch_data|nequip-compile|ninja|setup.py" 2>/dev/null \
  | grep -v "bash -c" | cut -c1-110 | sed 's/^/  /' || echo "  none"

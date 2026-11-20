#!/usr/bin/env bash
# Sequential fetch so we never saturate the link or the disk.
set -u
cd "$(dirname "$0")/.."
PY=.venvs/frost-mace/bin/python
while pgrep -f "fetch_data.py --tier A" >/dev/null; do sleep 20; done
$PY scripts/fetch_data.py --tier B --only gns_rollout,mgn_rollout,mptrj
echo "=== tier B rollout sets complete ==="
df -h / | tail -1

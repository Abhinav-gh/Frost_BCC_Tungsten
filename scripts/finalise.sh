#!/usr/bin/env bash
# Wait for the large fetch, then run the full verification once.
cd "$(dirname "$0")/.."
while pgrep -f "fetch_data.py --tier C" >/dev/null || pgrep -f "fetch_big.sh" >/dev/null; do
  sleep 60
done
echo "downloads finished at $(date -u +%H:%M:%S) UTC"
tr '\r' '\n' < data/logs/fetch_big.log | grep -E "^#####|!! .* failed|REFUSING" | tail -8
echo
./scripts/verify.sh 2>&1

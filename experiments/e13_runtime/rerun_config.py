"""The md_runtime.py command that reproduces a result file's run, with overrides (2026-11-16).

Every md_*.json keeps its run's arguments (``config`` = vars(args)). Re-timing a claim on a changed stack (the tight
teacher list, teacher_list_cost.py) must change nothing else, so the command is rebuilt from that record: each key the
current parser still knows becomes its flag (True -> the bare flag, False / None -> omitted, anything else -> its value);
keys the parser no longer knows are reported, not passed.

    python experiments/e13_runtime/rerun_config.py data/results/e13/md_cu500_T300_rtfix.json --set tight_teacher=1 \
        --set tag=rtfix_tight
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "e13_runtime" / "md_runtime.py"


def known_flags() -> dict[str, bool]:
    """dest -> is a store_true flag, from the runner's add_argument calls."""
    src = RUNNER.read_text()
    out = {}
    for m in re.finditer(r'add_argument\(\s*"--([a-z0-9-]+)"([^)]*)\)', src, flags=re.S):
        dest = m.group(1).replace("-", "_")
        out[dest] = "store_true" in m.group(2)
    return out


def command(config: dict, overrides: dict) -> tuple[str, list[str]]:
    flags = known_flags()
    cfg = {**config, **overrides}
    parts, dropped = [".venvs/frost-mace/bin/python", "experiments/e13_runtime/md_runtime.py"], []
    for key, value in cfg.items():
        if key not in flags:
            dropped.append(key)
            continue
        flag = "--" + key.replace("_", "-")
        if flags[key]:
            if value in (True, 1, "1", "true", "True"):
                parts.append(flag)
            continue
        if value is None or value is False:
            continue
        parts += [flag, str(value)]
    return " ".join(shlex.quote(p) for p in parts), dropped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result")
    ap.add_argument("--set", action="append", default=[], help="key=value overrides (keys as in the config)")
    args = ap.parse_args()
    config = json.loads(Path(args.result).read_text())["config"]
    overrides = dict(kv.split("=", 1) for kv in args.set)
    cmd, dropped = command(config, overrides)
    if dropped:
        print(f"# not passed (unknown to the current runner): {', '.join(dropped)}")
    print(cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

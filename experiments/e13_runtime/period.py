"""The fastest period of a system from its teacher's own NVE spectrum (``md_runtime.py --frames``, the teacher arm), by
``frost.observables.fastest_period_fs`` (the frequency below which 99% of the vibrational density of states lies).

The certificate's resonance prior needs this period for every new system (CLAUDE.md trap 15: pass ``--period-fs``
or the cap is off). A Langevin-thermostatted spectrum is flattened (trap 28), so only NVE teacher frames are used.

Usage: python experiments/e13_runtime/period.py data/results/e13/frames_<run>_<tag>.npz [ARM]
       -> prints the period and writes data/results/e13/<run>_period.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

from frost.observables import fastest_period_fs, vdos


def main(argv) -> int:
    path = Path(argv[0])
    arm = argv[1] if len(argv) > 1 else "teacher"
    d = np.load(path)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    v = torch.as_tensor(d[f"{arm}__v"], dtype=torch.float64, device=dev)  # unweighted, as score_frames' spectrum
    dt = float(d["v_fs"])
    freq, spec, _ = vdos(v, dt, max_lag=min(2000, v.shape[0] // 2))
    period = float(fastest_period_fs(freq, spec, quantile=0.99))
    stem = path.stem.removeprefix("frames_")
    m = re.match(r"(.*?_T\d+)(_|$)", stem)
    run = m.group(1) if m else stem
    out = path.parent / f"{run}_period.json"
    out.write_text(json.dumps({"fastest_period_fs": period, "quantile": 0.99, "frames": str(path), "arm": arm,
                               "cap_0p22_fs": 0.22 * period, "cap_0p45_fs": 0.45 * period}, indent=1))
    print(f"{run}: fastest period {period:.2f} fs (0.22 T = {0.22 * period:.2f} fs) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

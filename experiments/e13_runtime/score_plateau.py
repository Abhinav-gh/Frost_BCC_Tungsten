"""Score every arm of a runtime run on a time window, paired with the teacher arm's same window.

``score_frames.py``'s skip mode is for a Frost run longer than its teacher (steady state, unpaired). A heating ramp
(``md_runtime.py --ramp``) has every arm the same length and the question is the plateau after the ramp: this cuts
every arm, the teacher included, to [START_PS, end) and scores the cut file with ``score_frames.py`` unchanged.

Usage: python experiments/e13_runtime/score_plateau.py frames_<run>_<tag>.npz START_PS
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def main(argv) -> int:
    src, start = Path(argv[0]), float(argv[1])
    d = np.load(src)
    out = {k: d[k] for k in ("numbers", "cell", "masses", "v_fs", "x_fs")}
    for k in d.files:
        if "__" in k:
            step = float(d["v_fs"]) if k.endswith("__v") else float(d["x_fs"])
            out[k] = d[k][int(round(start / (step / 1000.0))):]
    dst = src.with_name(src.stem + f"_from{int(start)}ps.npz")
    np.savez(dst, **out)
    import score_frames  # same directory

    return score_frames.main([str(dst)])


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main(sys.argv[1:]))

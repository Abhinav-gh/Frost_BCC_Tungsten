"""Score FlashMD's arms against its own reference arm (``flashmd_run.py`` frames), with our estimators (frost env).

FlashMD returns velocities once per stride, so its spectrum can only be compared below that stride's Nyquist
frequency. To compare like with like, the reference's velocities are decimated to each arm's stride and both spectra
are computed by the same estimator (``frost.observables.vdos``) on the same band; the reference's own half-to-half
spread at that sampling is the error bar. Structure (total RDF) uses positions at the common ``x_fs`` spacing, which
``flashmd_run.py`` keeps equal across arms; temperature uses the run's masses.

The same scoring serves ``kanda_deepmd.py``'s arms (prefix ``kanda``), whose reference is their accurate model run
every step (``accurate``), ``flare_otf.py``'s (prefix ``flare``), and Frost's own runtime runs (prefix ``frost``: the
``md_runtime.py --frames`` file of a tag, against its teacher arm), so every method in a comparison is scored by one
estimator.

An optional fourth argument skips the first that many ps of every arm, for runs that are stationary only after a
transient (a heating ramp's plateau): the error bar is then the plateau's own half-to-half spread.

Usage
  PYTHONPATH=src .venvs/frost-mace/bin/python experiments/e14_baselines/flashmd_score.py cu500_T300 nve20
  PYTHONPATH=src .venvs/frost-mace/bin/python experiments/e14_baselines/flashmd_score.py cu500_T300 nve20 kanda
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from frost.observables import rdf, vdos

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "data" / "results" / "e14"
KB = 8.617333262e-5


def spectrum(v, dt_fs):
    freq, s, _ = vdos(torch.as_tensor(v, dtype=torch.float64), dt_fs, max_lag=min(1024, len(v) // 2))
    return freq, s / s.max()


def l1(a, b):
    return float(np.abs(b - a).sum() / np.abs(a).sum())


def g_total(x, cell, numbers, every=1):
    species, types = np.unique(numbers, return_inverse=True)
    return np.asarray(rdf(torch.as_tensor(x[::every], dtype=torch.float64), cell, types, len(species))["g_total"])


def temperature(v, masses):
    ke = 0.5 * (masses[None, :, None] * v.astype(np.float64) ** 2).sum((1, 2))
    return float(np.mean(2.0 * ke / (3 * v.shape[1] * KB)))


def arms_of(prefix: str, run: str, tag: str):
    """(report, reference arm name, arm name -> arrays) for one comparison's files."""
    if prefix == "frost":
        f = np.load(ROOT / "data" / "results" / "e13" / f"frames_{run}_{tag}.npz")
        md = json.loads((ROOT / "data" / "results" / "e13" / f"md_{run}_{tag}.json").read_text())
        names = sorted({k.split("__")[0] for k in f.files if "__" in k})
        meta = {"arms": {a: {"ms_per_fs": md["arms"][a]["ms_per_fs"]} for a in names}}
        common = {k: f[k] for k in ("cell", "numbers", "masses")}

        def load(a):
            return {"v": f[f"{a}__v"], "x": f[f"{a}__x"], "v_fs": float(f["v_fs"]), "x_fs": float(f["x_fs"]), **common}

        return meta, "teacher", load
    ref_arm = {"flashmd": "reference", "kanda": "accurate", "flare": "reference"}[prefix]
    meta = json.loads((RES / f"{prefix}_{run}_{tag}.json").read_text())
    return meta, ref_arm, lambda a: np.load(RES / f"{prefix}_{run}_{tag}_{a}.npz")


def main(argv) -> int:
    run, tag = argv[0], argv[1]
    prefix = argv[2] if len(argv) > 2 else "flashmd"
    skip_ps = float(argv[3]) if len(argv) > 3 else 0.0
    meta, ref_arm, load_all = arms_of(prefix, run, tag)

    def load(a):
        arm = dict(load_all(a))
        if skip_ps:
            v_fs = float(arm["v_fs"])
            x_fs = float(arm["x_fs"]) if "x_fs" in arm else float(meta.get("x_fs", v_fs))
            arm["v"], arm["x"] = arm["v"][round(1e3 * skip_ps / v_fs):], arm["x"][round(1e3 * skip_ps / x_fs):]
        return arm

    ref = load(ref_arm)
    v_ref, x_ref, dt_ref = ref["v"], ref["x"], float(ref["v_fs"])
    cell, numbers, masses = ref["cell"], ref["numbers"], ref["masses"]
    g_ref = g_total(x_ref, cell, numbers)
    half = len(x_ref) // 2
    rdf_own = l1(g_total(x_ref[:half], cell, numbers), g_total(x_ref[half:], cell, numbers))
    out = {"run": run, "tag": tag, "skip_ps": skip_ps, "reference": {"T": temperature(v_ref, masses), "rdf_own_spread": rdf_own,
                                                 **meta["arms"][ref_arm]}, "arms": {}}
    print(f"{run} {tag}: reference T {out['reference']['T']:.1f} K, RDF own spread {rdf_own:.4f}, "
          f"{meta['arms'][ref_arm]['ms_per_fs']:.2f} ms/fs", flush=True)
    for name, row in meta["arms"].items():
        if name == ref_arm:
            continue
        arm = load(name)
        stride = float(arm["v_fs"])
        step = round(stride / dt_ref)
        v_dec = v_ref[step - 1::step]  # the reference's velocities at the arm's sampling
        f_r, s_r = spectrum(v_dec, stride)
        f_a, s_a = spectrum(arm["v"], stride)
        h = len(v_dec) // 2
        _, s_r1 = spectrum(v_dec[:h], stride)
        _, s_r2 = spectrum(v_dec[h:], stride)
        own_v = l1(s_r1, np.interp(np.linspace(0, 1, len(s_r1)), np.linspace(0, 1, len(s_r2)), s_r2))
        n_x = min(len(arm["x"]), len(x_ref))
        g_a = g_total(arm["x"][:n_x], cell, numbers)
        res = {"stride_fs": stride, "nyquist_THz": float(1e3 / (2 * stride)),
               "vdos_l1": l1(s_r, np.interp(f_r, f_a, s_a)), "vdos_own_spread": own_v,
               "rdf_l1": l1(g_ref, g_a), "rdf_own_spread": rdf_own,
               "T_shift_K": temperature(arm["v"], masses) - out["reference"]["T"],
               "ms_per_fs": row["ms_per_fs"],
               "speedup_vs_reference": meta["arms"][ref_arm]["ms_per_fs"] / row["ms_per_fs"],
               "energy_change_meV_per_atom": row.get("energy_change_meV_per_atom"), "finite": row.get("finite")}
        res["pass"] = res["vdos_l1"] <= own_v and res["rdf_l1"] <= rdf_own
        out["arms"][name] = res
        print(f"  {name}: {res['ms_per_fs']:.2f} ms/fs ({res['speedup_vs_reference']:.1f}x) | VDOS below "
              f"{res['nyquist_THz']:.1f} THz {res['vdos_l1']:.4f} (own {own_v:.4f}) | RDF {res['rdf_l1']:.4f} "
              f"(own {rdf_own:.4f}) | T {res['T_shift_K']:+.1f} K -> {'pass' if res['pass'] else 'FAIL'}", flush=True)
    suffix = f"_from{skip_ps:g}ps" if skip_ps else ""
    (RES / f"{prefix}_score_{run}_{tag}{suffix}.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

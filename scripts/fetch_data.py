#!/usr/bin/env python3
"""Tiered, resumable downloader for every Frost dataset and checkpoint.

Tiers
  A  everything needed to reach Gate 1 and Gate 2, plus the small-molecule
     fidelity sets.  ~16 GB.
  B  rollout inputs for the cross-domain study (test+valid only -- Frost never
     trains, so train splits are only needed if we must train GNS/MGN
     ourselves) and the bulk provenance sets.  ~55 GB.
  C  opt-in giants: OMat24, Alexandria, GNS/MGN 3-D sets, MGN train splits.
     Hundreds of GB.  Never downloaded without --tier C.

Usage
  python scripts/fetch_data.py --tier A
  python scripts/fetch_data.py --tier B --only gns_rollout,mgn_rollout
  python scripts/fetch_data.py --list
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "datasets"

GNS = "https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets"
MGN = "https://storage.googleapis.com/dm-meshgraphnets"
QM = "http://www.quantum-machine.org/gdml"
FIG = "https://ndownloader.figshare.com/files"


@dataclass
class Item:
    key: str
    tier: str
    dest: str  # relative to data/datasets
    urls: list[str]
    approx_gb: float
    note: str = ""
    unpack: str | None = None  # "tar.bz2" | "tar.gz" | None
    extra: dict = field(default_factory=dict)


ITEMS: list[Item] = [
    # ---------------------------------------------------------------- tier A
    Item(
        "rmd17",
        "A",
        "rmd17",
        [f"{FIG}/23950376"],
        1.07,
        "Revised MD17 (figshare 12672038). Tighter DFT convergence than MD17.",
        unpack="tar.bz2",
    ),
    Item(
        "md17",
        "A",
        "md17",
        [
            f"{QM}/data/npz/md17_{m}.npz"
            for m in (
                "aspirin",
                "benzene2017",
                "ethanol",
                "malonaldehyde",
                "naphthalene",
                "salicylic",
                "toluene",
                "uracil",
            )
        ],
        1.3,
        "Original MD17 trajectories, for continuity with published numbers.",
    ),
    Item(
        "md22",
        "A",
        "md22",
        [
            f"{QM}/repo/datasets/md22_{m}.npz"
            for m in (
                "Ac-Ala3-NHMe",
                "DHA",
                "stachyose",
                "AT-AT",
                "AT-AT-CG-CG",
                "buckyball-catcher",
                "double-walled_nanotube",
            )
        ],
        0.8,
        "42-370 atom biomolecules; bridges MD17 and condensed phase.",
    ),
    Item(
        "3bpa",
        "A",
        "3bpa",
        ["https://github.com/davkovacs/BOTNet-datasets/raw/main/dataset_3BPA.tar.gz"],
        0.02,
        "Flexible-molecule torsional benchmark.",
        unpack="tar.gz",
    ),
    Item(
        "spice2",
        "A",
        "spice2",
        ["__github_release__:openmm/spice-dataset"],
        12.0,
        "SPICE2 HDF5; organic/drug-like provenance for MACE-OFF.",
        extra={"pattern": ".hdf5"},
    ),
    # ---------------------------------------------------------------- tier B
    Item(
        "gns_rollout",
        "B",
        "gns",
        [
            f"{GNS}/{d}/{f}"
            for d in ("WaterDrop", "Sand", "Goop")
            for f in ("metadata.json", "valid.tfrecord", "test.tfrecord")
        ],
        0.7,
        "GNS rollout inputs (E19-E21). No train split needed for inference.",
    ),
    Item(
        "mgn_rollout",
        "B",
        "mgn",
        [
            f"{MGN}/{d}/{f}"
            for d in (
                "flag_simple",
                "deforming_plate",
                "sphere_dynamic",
                "cylinder_flow",
            )
            for f in ("meta.json", "valid.tfrecord", "test.tfrecord")
        ],
        12.0,
        "MeshGraphNets rollout inputs. cylinder_flow is the NEGATIVE CONTROL.",
    ),
    Item(
        "mptrj",
        "B",
        "mptrj",
        ["__hf__:nimashoghi/mptrj"],
        4.0,
        "MPtrj provenance for MACE-MP-0 / CHGNet; used to calibrate S-hat.",
    ),
    Item(
        "gns_train",
        "B",
        "gns",
        [f"{GNS}/{d}/train.tfrecord" for d in ("WaterDrop", "Sand", "Goop")],
        10.3,
        "ONLY needed if no public GNS checkpoint exists and we must train.",
    ),
    Item(
        "mgn_train_small",
        "B",
        "mgn",
        [f"{MGN}/{d}/train.tfrecord" for d in ("deforming_plate", "flag_simple")],
        18.8,
        "ONLY needed if we must train MGN ourselves.",
    ),
    # ---------------------------------------------------------------- tier C
    Item(
        "mgn_train_rest",
        "C",
        "mgn",
        [f"{MGN}/{d}/train.tfrecord" for d in ("sphere_dynamic", "cylinder_flow")],
        42.8,
        "Remaining MGN train splits.",
    ),
    Item(
        "gns_3d",
        "C",
        "gns",
        [
            f"{GNS}/{d}/{f}"
            for d in ("Water-3D", "Sand-3D")
            for f in ("metadata.json", "valid.tfrecord", "test.tfrecord")
        ],
        21.7,
        "3-D GNS sets. Large; only if the 2-D result is favourable.",
    ),
    Item(
        "omat24",
        "C",
        "omat24",
        ["__hf__:fairchem/OMAT24"],
        400.0,
        "OMat24. Provenance only -- Frost trains nothing. Opt in deliberately.",
    ),
    # ------------------------------------------------------- plasma / MHD (note section 6)
    Item(
        "well_mhd64",
        "B",
        "the_well/MHD_64",
        ["__hf__:polymathic-ai/MHD_64"],
        71.6,
        "The Well MHD_64: isothermal compressible MHD turbulence, 100 trajectories "
        "x 100 steps, M_s 0.5-7, M_A 0.7-2 (W4 and the M_f sweep).",
    ),
    Item(
        "walrus",
        "C",
        "../models/walrus",
        ["__hf_model__:polymathic-ai/walrus"],
        6.0,
        "Walrus 1.3B continuum-dynamics foundation model (MIT). Optional W4 surrogate.",
    ),
    Item(
        "well_mhd256",
        "C",
        "the_well/MHD_256",
        ["__hf__:polymathic-ai/MHD_256"],
        4580.0,
        "MHD_64 at 256^3 (~64x larger). Do not pull; stream single files if ever needed.",
    ),
]

# Not fetchable by script. Recorded here so the gap is visible:
#   JHTDB forced MHD 1024^3  needs a SciServer token (pyJHTDB); ask before week 9
#   TURB-MHD (CC BY 4.0)     portal download from http://smart-turb.roma2.infn.it

BY_KEY = {i.key: i for i in ITEMS}


def sh(cmd: list[str], **kw) -> int:
    return subprocess.call(cmd, **kw)


def curl(url: str, out: Path) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    # -C - resumes a partial file; --retry survives GCS hiccups.
    rc = sh(
        [
            "curl",
            "-fL",
            "--retry",
            "5",
            "--retry-delay",
            "5",
            "-C",
            "-",
            "-o",
            str(out),
            url,
        ]
    )
    return rc == 0


def github_release_assets(repo: str, pattern: str) -> list[tuple[str, str]]:
    """Anonymous GitHub API allows 60 requests/hour; set GH_TOKEN to lift it."""
    import urllib.request

    url = f"https://api.github.com/repos/{repo}/releases/latest"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as fh:
        rel = json.load(fh)
    return [
        (a["name"], a["browser_download_url"])
        for a in rel.get("assets", [])
        if pattern in a["name"]
    ]


def fetch_item(it: Item, dry: bool) -> None:
    dest = DATA / it.dest
    dest.mkdir(parents=True, exist_ok=True)
    for url in it.urls:
        if url.startswith("__github_release__:"):
            repo = url.split(":", 1)[1]
            pat = it.extra.get("pattern", "")
            try:
                assets = github_release_assets(repo, pat)
            except Exception as exc:
                print(f"  ! github api failed for {repo}: {exc}")
                continue
            for name, aurl in assets:
                target = dest / name
                if target.exists() and target.stat().st_size > 0:
                    print(f"  = {name} (present)")
                    continue
                print(f"  + {name}")
                if not dry:
                    curl(aurl, target)
            continue

        if url.startswith(("__hf__:", "__hf_model__:")):
            repo = url.split(":", 1)[1]
            repo_type = "model" if url.startswith("__hf_model__:") else "dataset"
            print(f"  + huggingface {repo_type} snapshot {repo} -> {dest}")
            if not dry:
                try:
                    from huggingface_hub import snapshot_download

                    snapshot_download(
                        repo_id=repo,
                        repo_type=repo_type,
                        local_dir=str(dest),
                        max_workers=8,
                    )
                except ImportError:
                    print("  ! pip install huggingface_hub")
                except Exception as exc:
                    print(
                        f"  ! hf download failed for {repo}: "
                        f"{type(exc).__name__}: {str(exc)[:200]}"
                    )
                    print(
                        "  ! if this says 'gated', accept the terms on the dataset "
                        "page and set HF_TOKEN"
                    )
            continue

        name = url.rstrip("/").split("/")[-1]
        # keep the split directory structure for tfrecord sets
        parts = url.rstrip("/").split("/")
        if name.endswith((".tfrecord", ".json")) and len(parts) >= 2:
            target = dest / parts[-2] / name
        else:
            target = dest / name
        if target.exists() and target.stat().st_size > 0:
            print(f"  = {target.relative_to(DATA)} (present)")
            continue
        print(f"  + {target.relative_to(DATA)}")
        if not dry and not curl(url, target):
            print(f"  ! FAILED {url}")

    if it.unpack and not dry:
        for arc in dest.glob("*"):
            if arc.suffix in (".bz2", ".gz") and arc.is_file():
                marker = dest / (arc.name + ".unpacked")
                if marker.exists():
                    continue
                print(f"  ~ unpacking {arc.name}")
                flag = "xjf" if it.unpack == "tar.bz2" else "xzf"
                if sh(["tar", flag, str(arc), "-C", str(dest)]) == 0:
                    marker.touch()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="A", choices=["A", "B", "C"])
    ap.add_argument("--only", default="", help="comma-separated item keys")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.list:
        tot = {}
        for it in ITEMS:
            tot[it.tier] = tot.get(it.tier, 0) + it.approx_gb
            print(f"[{it.tier}] {it.key:<18} ~{it.approx_gb:>7.1f} GB  {it.note}")
        print("\ntotals:", ", ".join(f"{k}={v:.0f} GB" for k, v in sorted(tot.items())))
        return 0

    if args.only:
        wanted = [BY_KEY[k] for k in args.only.split(",") if k in BY_KEY]
    else:
        wanted = [i for i in ITEMS if i.tier == args.tier]

    need = sum(i.approx_gb for i in wanted)
    free = os.statvfs(DATA if DATA.exists() else ROOT)
    free_gb = free.f_bavail * free.f_frsize / 2**30
    print(f"requested ~{need:.1f} GB; {free_gb:.0f} GB free on {ROOT}")
    if need > free_gb * 0.8:
        print("REFUSING: would use more than 80% of free space.")
        return 2

    for it in wanted:
        print(f"\n=== {it.key} (tier {it.tier}, ~{it.approx_gb} GB) ===")
        print(f"    {it.note}")
        t0 = time.time()
        fetch_item(it, args.dry_run)
        print(f"    done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

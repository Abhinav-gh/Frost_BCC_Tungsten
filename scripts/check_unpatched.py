"""Frost patches nothing: installed libraries match their wheels, and our code patches nothing at runtime.

  python scripts/check_unpatched.py installed   # run with each venv's python
  python scripts/check_unpatched.py code

``installed`` re-hashes every file of the libraries Frost drives against the
sha256 that its wheel recorded (RECORD). It also requires site-packages/nvidia
to hold no added symlinks; the CUDA_HOME overlay in .tools replaces them.
``code`` scans src, experiments, tests and scripts for runtime-patching
constructs. Each mode prints one line starting with "ok" or "FAIL".
"""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata as md
import pathlib
import re
import sys
import sysconfig

ROOT = pathlib.Path(__file__).resolve().parents[1]
WATCHED = (
    "mace-torch", "e3nn", "cuequivariance", "cuequivariance-torch",
    "cuequivariance-ops-torch-cu13", "openequivariance", "flashTP_e3nn", "vesin",
    "torch-sim-atomistic", "nvalchemi-toolkit-ops", "ase", "matscipy",
    "nequip", "nequip-allegro", "sevenn", "astronomix", "the_well",
    "neuralgcm", "dinosaur", "terrax",
)
PATCH = re.compile(
    r"monkeypatch|mock\.patch|unittest\.mock|\bsetattr\(|\.forward\s*=[^=]|types\.MethodType"
    r"|sys\.modules\[[^]]*\]\s*=[^=]|__class__\s*=[^=]|importlib\.reload"
)


def installed() -> str:
    problems: list[str] = []
    packages = files = 0
    for name in WATCHED:
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            continue
        packages += 1
        for record in dist.files or []:
            if not record.hash or record.hash.mode != "sha256":
                continue
            path = pathlib.Path(record.locate())
            if not path.exists():
                problems.append(f"{name}: missing {record}")
                continue
            files += 1
            digest = hashlib.sha256(path.read_bytes()).digest()
            if base64.urlsafe_b64encode(digest).rstrip(b"=").decode() != record.hash.value:
                problems.append(f"{name}: modified {record}")
    nvidia = pathlib.Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    if nvidia.is_dir():
        problems += [f"added symlink {p.relative_to(nvidia.parent)}"
                     for p in nvidia.rglob("*") if p.is_symlink()]
    if problems:
        return f"FAIL {len(problems)} changes to installed packages, e.g. {problems[0]}"
    return f"ok installed libraries match their wheels ({packages} packages, {files} files)"


def code() -> str:
    hits = []
    for folder in ("src", "experiments", "tests", "scripts"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            if path.resolve() == pathlib.Path(__file__).resolve():
                continue
            for n, line in enumerate(path.read_text().splitlines(), 1):
                if PATCH.search(line.split("#")[0]):
                    hits.append(f"{path.relative_to(ROOT)}:{n}")
    if hits:
        return f"FAIL runtime-patching constructs in our code: {', '.join(hits[:3])}"
    return "ok no runtime-patching constructs in src, experiments, tests or scripts"


def main() -> int:
    line = installed() if sys.argv[1:] == ["installed"] else code()
    print(line)
    return 0 if line.startswith("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

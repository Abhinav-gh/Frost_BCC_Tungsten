"""A teacher served from its own Python environment, for MLIPs whose stacks cannot share ours.

FlashMD's reference model (PET-OMATPES, metatomic) and UMA (fairchem) pin their own torch and dependencies, so they run
in ``.venvs/frost-flashmd`` and ``.venvs/frost-uma``. ``experiments/e14_baselines/teacher_server.py`` runs inside that
environment, builds the model's own ASE calculator, and answers position frames with energy and forces over pipes.
Nothing of theirs is imported here: this module only starts the process and exchanges arrays.

Spec strings: ``pet:<mlip>`` (``flashmd.get_pretrained(mlip)``'s energy model, e.g. ``pet:pet-omatpes-v2``) and
``uma:<checkpoint>:<task>`` (``fairchem`` ``pretrained_mlip.get_predict_unit``, e.g. ``uma:uma-m-1p1:omat``) and
``orb:<pretrained function>`` (``orb_models.forcefield.pretrained``, e.g. ``orb:orb_v3_conservative_inf_omat``) and
``mace:<teacher>`` (our MACE teacher, ``frost.teachers.load_calc``, e.g. ``mace:mp-medium``), which serves tools from
other environments (FLARE) with our oracle, and ``nequip:<compiled model>`` (an Allegro or NequIP model compiled for
ASE by ``nequip-compile``, e.g. Allegro-MP-L), and ``deepmd:<frozen model>`` (DeePMD-kit's ``DP`` calculator, e.g. the
Kanda et al. accurate model), and ``kanda:<frozen .pb>`` (the same calculator from Kanda et al.'s released TensorFlow
fork, for their own TiO2 models). torch is imported only by ``__call__``, so ``numpy`` works in an environment
without it.

Every call pays a device-to-host copy, two pipe transfers and the server's ASE overhead. A baseline arm that calls the
same teacher every step pays exactly the same, so ratios against it are fair; the absolute overhead is reported by
``ExternalTeacher.overhead_ms``.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / "experiments" / "e14_baselines" / "teacher_server.py"
VENVS = {"pet": ROOT / ".venvs" / "frost-flashmd", "uma": ROOT / ".venvs" / "frost-uma",
         "orb": ROOT / ".venvs" / "frost-lsd", "mace": ROOT / ".venvs" / "frost-mace",
         "deepmd": ROOT / ".venvs" / "frost-deepmd", "nequip": ROOT / ".venvs" / "frost-nequip",
         "mtsallegro": ROOT / ".venvs" / "frost-nequip", "kanda": ROOT / ".venvs" / "frost-kanda",
         "pyscf": ROOT / ".venvs" / "frost-pyscf",  # GPU4PySCF DFT (further_expansions_beyond.md 7.1)
         "ani": ROOT / ".venvs" / "frost-ani"}  # ANI-1xnr through TorchANI (7.4)


def _read_exact(stream, n: int) -> bytes:
    out = bytearray()
    while len(out) < n:
        chunk = stream.read(n - len(out))
        if not chunk:
            raise RuntimeError("external teacher closed its pipe")
        out.extend(chunk)
    return bytes(out)


class ExternalTeacher:
    """Energy and forces from a model served by another environment's process (see the module docstring)."""

    def __init__(self, spec: str, numbers, cell, device: str = "cuda", stress: bool = False):
        kind = spec.split(":", 1)[0]
        if kind not in VENVS:
            raise ValueError(f"unknown external teacher {spec!r}; kinds: {sorted(VENVS)}")
        venv = VENVS[kind]
        if kind == "mace":  # our own environment: keep env.sh's settings (CUDA overlay, torch load flag)
            env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        else:
            env = {k: v for k, v in os.environ.items() if k not in ("LD_LIBRARY_PATH", "CUDA_HOME", "PYTHONPATH")}
        env["PATH"] = f"{venv / 'bin'}:{env.get('PATH', '/usr/bin:/bin')}"
        if kind == "deepmd":  # torch's JIT in that env needs its own nvrtc builtins, which pip puts off the loader path
            env["LD_LIBRARY_PATH"] = ":".join(str(p) for p in venv.glob("lib/python3*/site-packages/nvidia/cu13/lib"))
        if kind == "pyscf":  # unversioned CUDA library names gpu4pyscf searches for (scripts/pyscf_overlay.sh)
            env["PYTHONPATH"] = str(ROOT / ".tools" / "pyscf-overlay")
        if kind == "ani":  # TorchANI warns at import about optional compiled extensions; the pure-PyTorch path is used
            env["TORCHANI_NO_WARN_EXTENSIONS"] = "1"
        if kind == "kanda":
            # their built library, TensorFlow's own .so files and the pip CUDA 12 libraries, as their README sets for
            # LAMMPS; TensorFlow allocates GPU memory on demand, so it can share the card with the student
            site = next(venv.glob("lib/python3*/site-packages"))
            libs = [str(ROOT / "third_party" / "deepmd-mts" / "install-gpu" / "lib"), str(site / "tensorflow")]
            libs += [str(p) for p in sorted(site.glob("nvidia/*/lib"))]
            env.update(LD_LIBRARY_PATH=":".join(libs), TF_FORCE_GPU_ALLOW_GROWTH="true", TF_CPP_MIN_LOG_LEVEL="2")
        self.spec, self.device, self.n, self.stress = spec, device, len(numbers), bool(stress)
        self.last_stress = None
        self.proc = subprocess.Popen([str(venv / "bin" / "python"), str(SERVER), spec], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, env=env, cwd=str(ROOT))
        header = json.dumps({"numbers": [int(z) for z in numbers],
                             "cell": np.asarray(cell, dtype=np.float64).tolist(), "stress": self.stress}).encode()
        self.proc.stdin.write(struct.pack("<Q", len(header)) + header)
        self.proc.stdin.flush()
        if _read_exact(self.proc.stdout, 2) != b"OK":
            raise RuntimeError(f"external teacher {spec} did not start")
        self.calls, self.seconds = 0, 0.0

    def numpy(self, positions: np.ndarray) -> tuple[float, np.ndarray]:
        """(energy eV, forces eV/A) at ``positions`` (n, 3) in A; with ``stress``, ``last_stress`` holds its stress."""
        t0 = time.perf_counter()
        self.proc.stdin.write(np.ascontiguousarray(positions, dtype=np.float64).tobytes())
        self.proc.stdin.flush()
        nf = 8 + 24 * self.n
        buf = _read_exact(self.proc.stdout, nf + (48 if self.stress else 0))
        self.calls += 1
        self.seconds += time.perf_counter() - t0
        if self.stress:
            self.last_stress = np.frombuffer(buf[nf:], dtype=np.float64).copy()
        return struct.unpack("<d", buf[:8])[0], np.frombuffer(buf[8:nf], dtype=np.float64).reshape(self.n, 3)

    def __call__(self, x: torch.Tensor):
        """The runtime's teacher signature: (energy, forces as float64 on the device, x)."""
        import torch

        e, f = self.numpy(x.detach().to("cpu", torch.float64).numpy())
        return torch.tensor(e, dtype=torch.float64, device=x.device), torch.as_tensor(f, device=x.device), x

    @property
    def ms_per_call(self) -> float:
        return 1e3 * self.seconds / max(self.calls, 1)

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()

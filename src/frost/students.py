"""One constructor for both student families, so every harness builds them the same way."""

from __future__ import annotations

from collections.abc import Sequence


def make_student(kind: str, species: Sequence[int], mean_neighbours: float, cutoff: float = 4.0,
                 channels: int = 16, mix: int | None = None, hidden: int = 64, lmax: int = 2,
                 max_ell: int = 2, layers: int = 2, correlation: int = 3,
                 vector_channels: int | None = None, backend: str = "e3nn",
                 num_bessel: int = 8, body4: bool = False):
    """A ``FrostStudent`` ("frost") or ``MaceStudent`` ("mace") normalised for this system.

    ``backend`` picks the MACE student's kernels: "e3nn", "cueq" or "oeq".
    """
    if kind == "mace":
        from frost.student_mace import MaceStudent

        return MaceStudent(species, cutoff=cutoff, channels=channels, max_ell=max_ell,
                           layers=layers, correlation=correlation, mean_neighbours=mean_neighbours,
                           vector_channels=vector_channels, backend=backend,
                           num_bessel=num_bessel)
    if kind != "frost":
        raise ValueError(f"unknown student kind {kind!r}")
    if backend != "e3nn":
        raise ValueError("the FrostStudent has a single implementation; backend must be 'e3nn'")
    from frost.student import FrostStudent

    student = FrostStudent(species, cutoff=cutoff, channels=channels, mix=mix or channels,
                           hidden=hidden, lmax=lmax, body4=body4)
    student.set_neighbour_normalisation(mean_neighbours)
    return student


def load_student(path, device="cpu"):
    """Rebuild a saved student (``{"state_dict", "config"}``) on ``device``."""
    import torch

    checkpoint = torch.load(path, map_location=device)
    cfg = dict(checkpoint["config"])
    kind = cfg.get("kind", "frost")
    if kind == "mace":
        student = make_student("mace", cfg["species"], cfg["mean_neighbours"], cfg["cutoff"],
                               cfg["channels"], max_ell=cfg["max_ell"], layers=cfg["layers"],
                               correlation=cfg["correlation"],
                               vector_channels=cfg.get("vector_channels"),
                               backend=cfg.get("backend", "e3nn"),
                               num_bessel=cfg.get("num_bessel", 8))
    elif kind == "nc":
        from frost.student_nc import ForceStudent

        student = ForceStudent(cfg["species"], cutoff=cfg["cutoff"], n_rbf=cfg["n_rbf"], channels=cfg["channels"],
                               mix=cfg["mix"], hidden=cfg["hidden"], envelope_p=cfg["envelope_p"],
                               lmax=cfg.get("lmax", 2), body4=cfg.get("body4", False), node_dim=cfg["node_dim"],
                               edge_hidden=cfg["edge_hidden"])
    else:
        from frost.student import FrostStudent

        student = FrostStudent(cfg["species"], cutoff=cfg["cutoff"], n_rbf=cfg["n_rbf"],
                               channels=cfg["channels"], mix=cfg["mix"], hidden=cfg["hidden"],
                               envelope_p=cfg["envelope_p"], lmax=cfg.get("lmax", 2),
                               body4=cfg.get("body4", False))
    # constant buffers added after a checkpoint was written are rebuilt, not loaded
    missing, unexpected = student.load_state_dict(checkpoint["state_dict"], strict=False)
    constants = {"c3", "metric", "freq"}
    mismatched = [k for k in (*missing, *unexpected) if k not in constants]
    if mismatched:
        raise RuntimeError(f"checkpoint {path} does not match its student: {mismatched[:5]}")
    return student.to(device)

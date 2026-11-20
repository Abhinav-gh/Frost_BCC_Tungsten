"""Offline tests of C3 and C5 on dense teacher labels (new_directions.md sections 4 and 6).

Both ideas act on what the impulse kick has to carry, so both are scored by the
impulse error of the residual over an outer step of k inner steps:

    I = sum_j c_j R_j  -  (k/2) (R_0 + R_k),     R = F_T - F_S,  c = (1/2, 1, ..., 1, 1/2),

in acceleration units (divided by each atom's mass) times fs, RMS over every atom of
every interval in held-out windows. The held staleness mean_j |R_j - R_0| is reported
alongside, as the certificate measures it.

C3, integrator-aware distillation. One trained student is fine-tuned twice on the
same dense labels, for the same number of updates:
  A  the current loss: force + force change (any two calls <= k apart) + energy change;
  B  force + the impulse loss |I|^2 over windows of k + 1 frames + energy change.
The impulse weight is set so that both extra terms start at the same size. The
hypothesis holds if B lowers the impulse error at k more than A does, at a
comparable force error.

C5, a fast online residual head. A linear energy head on the student's own
invariant node features, E_R = sum_i w . h_i(x), has forces G(x) w with
G = -d(sum_i h_i)/dx: conservative, and cheap in production (one extra term in the
student's energy). w is fitted causally by recursive least squares, with forgetting,
from the teacher calls at every k-th frame. The head joins the fast force, so the
kick carries R' = R - G w. The hypothesis holds if R' has a clearly smaller impulse
error and staleness than R.

Labels: the graphed cuEquivariance teacher on every frame of a 2000-frame training
window and two 600-frame held-out windows of the E1 trajectory.

Output: data/results/e9/offline_students_<run>.json

Usage (guarded launcher; GPU)
  scripts/safe_run.sh e9_offline .venvs/frost-mace/bin/python \\
      experiments/e9_direction/offline_students.py --run li3po4_T300 \\
      --student data/models/students/li3po4_T300_s35c16e1.pt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TRAJ = ROOT / "data" / "trajectories"
OUT = ROOT / "data" / "results" / "e9"
KS = (4, 6, 8, 12)


def impulse_weights(k: int, device) -> torch.Tensor:
    c = torch.ones(k + 1, device=device, dtype=torch.float64)
    c[0] = c[-1] = 0.5
    return c


def score(residuals: list[torch.Tensor], inv_mass: torch.Tensor) -> dict:
    """Impulse error and held staleness of a residual sequence, per k (acceleration units)."""
    out = {}
    for window in residuals:
        a = window.double() * inv_mass * 1e3  # (frames, atoms, 3), meV/(A amu)
        for k in KS:
            c = impulse_weights(k, a.device)
            imp, held, count = 0.0, 0.0, 0
            for t0 in range(0, a.shape[0] - k, k):
                seg = a[t0:t0 + k + 1]
                integral = torch.einsum("j,jad->ad", c, seg)
                trap = 0.5 * k * (seg[0] + seg[-1])
                imp += float((integral - trap).pow(2).sum(-1).mean())
                held += float((seg[1:] - seg[0]).pow(2).sum(-1).mean())
                count += 1
            slot = out.setdefault(k, [0.0, 0.0, 0])
            slot[0] += imp
            slot[1] += held
            slot[2] += count
    return {k: {"impulse_rms": math.sqrt(v[0] / v[2]), "held_rms": math.sqrt(v[1] / v[2])}
            for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="li3po4_T300")
    ap.add_argument("--student", required=True)
    ap.add_argument("--train", type=int, nargs=2, default=(1000, 3000))
    ap.add_argument("--eval", type=int, nargs="+", default=(7000, 8400))
    ap.add_argument("--eval-len", type=int, default=600)
    ap.add_argument("--k-train", type=int, default=6)
    ap.add_argument("--updates", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--forget", type=float, nargs="+", default=(1.0, 0.99, 0.95))
    ap.add_argument("--ridge", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    from mace.calculators import mace_mp

    from frost.insitu import Frame, InSituTrainer, LabelBuffer, collate
    from frost.neighbours import VerletList
    from frost.students import load_student
    from frost.teacher import frame_atoms
    from frost.teacher_fast import GraphedTeacher

    dev = "cuda"
    with h5py.File(TRAJ / f"{args.run}.h5", "r") as fh:
        numbers = np.asarray(fh.attrs["numbers"], dtype=np.int64)
        cell = np.asarray(fh.attrs["cell"], dtype=np.float64)
        wanted = list(range(*args.train))
        for start in args.eval:
            wanted += list(range(start, start + args.eval_len))
        positions = {f: np.asarray(fh["positions"][f], dtype=np.float64) for f in wanted}
    atoms = frame_atoms(numbers, positions[wanted[0]], cell)
    n = len(atoms)
    masses = torch.as_tensor(atoms.get_masses(), dtype=torch.float64, device=dev).unsqueeze(1)
    inv_mass = 1.0 / masses

    # ------------------------------------------------ dense teacher labels
    calc = mace_mp(model="medium", device=dev, default_dtype="float32", enable_cueq=True)
    teacher = GraphedTeacher(calc, atoms, dev)
    t_list = VerletList(cell, teacher.r_max, 1.0, dev, wrap_positions=False, builder="alchemi")
    student = load_student(args.student, dev)
    types = student.types(torch.as_tensor(numbers, device=dev))
    s_list = VerletList(cell, float(student.config["cutoff"]), 0.0, dev, wrap_positions=False,
                        builder="alchemi")
    frames: dict[int, Frame] = {}
    for f in wanted:
        x = torch.as_tensor(positions[f], device=dev)
        if t_list.update(x, force=t_list.i is None)[1]:
            teacher.set_edges(t_list.i, t_list.j, t_list.shifts, t_list.unit_shifts)
        energy, forces = teacher(x)
        s_list.update(x, force=True)
        frames[f] = Frame(step=f, x=x.float(), forces=forces.detach().clone().float(),
                          energy=float(energy), edge_i=s_list.i.clone(), edge_j=s_list.j.clone(),
                          shifts=s_list.shifts.float().clone())
    print(f"labelled {len(frames)} frames", flush=True)

    def student_forces(model, fr: Frame, create_graph=False):
        x = fr.x.detach().requires_grad_(True)
        e = model(x, types, fr.edge_i, fr.edge_j, fr.shifts)
        (g,) = torch.autograd.grad(e.sum(), x, create_graph=create_graph)
        return e, -g

    def residual_windows(model):
        out = []
        for start in args.eval:
            seq = [(frames[f].forces - student_forces(model, frames[f])[1]).detach()
                   for f in range(start, start + args.eval_len)]
            out.append(torch.stack(seq))
        return out

    def force_rmse(windows):
        return 1e3 * math.sqrt(float(torch.cat(windows).double().pow(2).sum(-1).mean()))

    report: dict = {"run": args.run, "student": args.student, "k_train": args.k_train,
                    "updates": args.updates}
    base_windows = residual_windows(student)
    report["base"] = {"force_rmse_meV_per_A": force_rmse(base_windows),
                      "score": score(base_windows, inv_mass)}

    # ------------------------------------------------ C3: two fine-tunes
    train_frames = [frames[f] for f in range(*args.train)]
    k = args.k_train
    # fresh loads, never deep copies of a model that has already run: cuEquivariance's
    # naive training path then found a tensor left on the CPU
    arm_a = load_student(args.student, dev)
    buffer = LabelBuffer(capacity=len(train_frames) + 1)
    for fr in train_frames:
        buffer.add(fr)
    trainer_a = InSituTrainer(arm_a, types, lr=args.lr, max_gap=k, spread=True, seed=args.seed)

    arm_b = load_student(args.student, dev)
    opt_b = torch.optim.Adam(arm_b.parameters(), lr=args.lr)
    gen = torch.Generator().manual_seed(args.seed)
    c = impulse_weights(k, dev).float()
    mev = 1e3

    def loss_b(model, window: list[Frame]):
        x, t, ei, ej, sh, batch = collate(window, types)
        x = x.detach().requires_grad_(True)
        energy = model(x, t, ei, ej, sh, batch, len(window))
        (grad,) = torch.autograd.grad(energy.sum(), x, create_graph=True)
        f_s = -grad.view(len(window), n, 3)
        f_t = torch.stack([fr.forces for fr in window])
        r = (f_t - f_s) * mev
        l_force = r.pow(2).sum(-1).mean()
        imp = torch.einsum("j,jad->ad", c, r) - 0.5 * k * (r[0] + r[-1])
        l_imp = imp.pow(2).sum(-1).mean()
        e_t = torch.tensor([window[0].energy, window[-1].energy], device=dev, dtype=energy.dtype)
        l_de = (((energy[-1] - energy[0]) - (e_t[1] - e_t[0])) * mev).pow(2) / n
        return l_force, l_imp, l_de

    starts = list(range(len(train_frames) - k))

    def draw_window():
        s = starts[int(torch.randint(len(starts), (1,), generator=gen))]
        return train_frames[s:s + k + 1]

    # balance: the impulse term starts at the size of A's force-change term
    probe = trainer_a.step(buffer, batch_pairs=4, report=True)  # one update of A, reported
    lf, li, _ = loss_b(arm_b, draw_window())
    lam = (probe["dforce_rmse_meV_per_A"] ** 2) / max(float(li.detach()), 1e-12)
    report["c3_lambda_impulse"] = lam
    for _ in range(args.updates - 1):
        trainer_a.step(buffer, batch_pairs=4)
    for _ in range(args.updates):
        lf, li, lde = loss_b(arm_b, draw_window())
        loss = lf + lam * li + 10.0 * lde
        opt_b.zero_grad(set_to_none=True)
        loss.backward()
        opt_b.step()
    for name, model in (("c3_arm_A_force_change", arm_a), ("c3_arm_B_impulse", arm_b)):
        w = residual_windows(model)
        report[name] = {"force_rmse_meV_per_A": force_rmse(w), "score": score(w, inv_mass)}
    del arm_a, arm_b, trainer_a, opt_b
    print("C3 done", flush=True)

    # ------------------------------------------------ C5: online RLS head on node features
    def head_basis(fr: Frame):
        """G = -d(sum_i h_i)/dx, one column per invariant channel: (atoms*3, C)."""
        x = fr.x.detach().requires_grad_(True)
        h = student.node_features(x, types, fr.edge_i, fr.edge_j, fr.shifts).sum(0)
        cols = []
        for ch in range(h.shape[0]):
            (g,) = torch.autograd.grad(h[ch], x, retain_graph=ch + 1 < h.shape[0])
            cols.append(-g.reshape(-1).double())
        return torch.stack(cols, dim=1)

    call_frames = list(range(args.train[0], args.train[1], k))
    base_res = {f: (frames[f].forces - student_forces(student, frames[f])[1]).detach().double()
                for f in list(frames)}
    basis = {}
    for f in call_frames:
        basis[f] = head_basis(frames[f])
    for start in args.eval:
        for f in range(start, start + args.eval_len):
            basis[f] = head_basis(frames[f])
    channels = next(iter(basis.values())).shape[1]
    report["c5_channels"] = channels
    print(f"C5 basis on {len(basis)} frames, {channels} channels", flush=True)

    report["c5"] = {}
    for forget in args.forget:
        a_mat = torch.zeros(channels, channels, dtype=torch.float64, device=dev)
        b_vec = torch.zeros(channels, dtype=torch.float64, device=dev)

        def observe(f, a_mat=a_mat, b_vec=b_vec, forget=forget):
            g = basis[f]
            a_mat.mul_(forget).add_(g.T @ g)
            b_vec.mul_(forget).add_(g.T @ base_res[f].reshape(-1))

        def solve(a_mat=a_mat, b_vec=b_vec):
            scale = float(a_mat.diagonal().mean()) or 1.0
            eye = torch.eye(channels, dtype=torch.float64, device=dev)
            return torch.linalg.solve(a_mat + args.ridge * scale * eye, b_vec)

        for f in call_frames:  # warm-up on the training stretch, causally
            observe(f)
        windows = []
        for start in args.eval:
            seq = []
            w = solve()
            for f in range(start, start + args.eval_len):
                if (f - start) % k == 0:
                    observe(f)  # the teacher call at this frame is now known
                    w = solve()
                head = (basis[f] @ w).reshape(n, 3)
                seq.append((base_res[f] - head).float())
            windows.append(torch.stack(seq))
        report["c5"][str(forget)] = {"force_rmse_meV_per_A": force_rmse(windows),
                                     "score": score(windows, inv_mass)}

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"offline_students_{args.run}.json"
    path.write_text(json.dumps(report, indent=1, default=float))

    def line(name, block):
        s = block["score"]
        return (f"{name:26s} force {block['force_rmse_meV_per_A']:7.2f} | " + " ".join(
            f"k={kk}: imp {s[kk]['impulse_rms']:.3f} held {s[kk]['held_rms']:.3f}" for kk in KS))

    print(line("base student", report["base"]))
    print(line("C3 A (force change)", report["c3_arm_A_force_change"]))
    print(line("C3 B (impulse loss)", report["c3_arm_B_impulse"]))
    for forget, block in report["c5"].items():
        print(line(f"C5 head, forget {forget}", block))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

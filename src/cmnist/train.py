"""Training loop shared by all methods (device-agnostic, T4-safe)."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .losses import compute_objective


def set_seed(seed: int):
    import numpy as np
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def env_accuracy(model, loader: DataLoader, device: str) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        pred = (model(x) > 0).float()
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def train_one_run(
    model,
    train_loaders: list[DataLoader],
    val_loaders: dict[str, DataLoader],
    method: str = "erm",
    steps: int = 2000,
    lr: float = 1e-3,
    penalty_weight: float = 1e4,
    anneal_steps: int = 190,
    rex_beta: float = 1e3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    log_every: int = 200,
    seed: int = 0,
) -> dict:
    set_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    iters = [iter(l) for l in train_loaders]

    def _next(i):
        try:
            return next(iters[i])
        except StopIteration:
            iters[i] = iter(train_loaders[i])
            return next(iters[i])

    def _snapshot(m):
        return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}

    hist: list[dict] = []
    best_id_acc, best_ood_acc = -1.0, -1.0
    best_id_state, best_ood_state = None, None
    for step in range(1, steps + 1):
        model.train()
        logits_e, y_e = [], []
        for i in range(len(train_loaders)):
            x, y, _ = _next(i)
            x, y = x.to(device), y.to(device)
            logits_e.append(model(x))
            y_e.append(y)
        obj, info = compute_objective(
            logits_e, y_e, method, step, penalty_weight, anneal_steps, rex_beta
        )
        opt.zero_grad()
        obj.backward()
        opt.step()
        if step % log_every == 0 or step == steps:
            rec = {"step": step, "obj": obj.item(),
                   **{k: (v.item() if torch.is_tensor(v) else v) for k, v in info.items()}}
            for name, loader in val_loaders.items():
                rec[f"acc_{name}"] = env_accuracy(model, loader, device)
            # Honest model selection (DomainBed): train-domain validation =
            # mean accuracy over training envs; OOD oracle = test accuracy.
            id_avg = float(sum(rec.get(f"acc_{k}", float("nan"))
                               for k in ("train_e1", "train_e2") if f"acc_{k}" in rec) / 2)
            ood = float(rec.get("acc_test", float("nan")))
            rec["id_avg"] = id_avg
            if id_avg == id_avg and id_avg > best_id_acc:
                best_id_acc = id_avg
                best_id_state = _snapshot(model)
            if ood == ood and ood > best_ood_acc:
                best_ood_acc = ood
                best_ood_state = _snapshot(model)
            hist.append(rec)
    return {"model": model, "history": hist,
            "best_id_state": best_id_state, "best_id_acc": best_id_acc,
            "best_ood_state": best_ood_state, "best_ood_acc": best_ood_acc}

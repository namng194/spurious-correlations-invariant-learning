"""Training loop shared by all methods (device-agnostic, T4-safe)."""

from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader

from .losses import compute_objective


def set_seed(seed: int):
    import random

    import numpy as np

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
    """Pass a model factory to seed initialization as well as training.

    Preconstructed modules remain supported, but their caller owns initialization.
    ID checkpoint selection exclusively uses loaders named val_*.
    """
    set_seed(seed)
    if steps < 1 or log_every < 1 or not train_loaders:
        raise ValueError("steps, log_every and number of training loaders must be positive")
    if not isinstance(model, torch.nn.Module):
        model = model()
    id_keys = [k for k in val_loaders if k.startswith("val_")]
    if not id_keys:
        raise ValueError("held-out val_* loaders are required for checkpoint selection")
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
    best_id_step, best_ood_step = None, None
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
            rec = {
                "step": step,
                "obj": obj.item(),
                **{k: (v.item() if torch.is_tensor(v) else v) for k, v in info.items()},
            }
            for name, loader in val_loaders.items():
                rec[f"acc_{name}"] = env_accuracy(model, loader, device)
            # Held-out ID validation; test selection is a separately labeled oracle.
            id_avg = sum(rec[f"acc_{k}"] for k in id_keys) / len(id_keys)
            ood = float(rec.get("acc_test", float("nan")))
            rec["id_avg"] = id_avg
            if math.isfinite(id_avg) and id_avg > best_id_acc:
                best_id_acc = id_avg
                best_id_state = _snapshot(model)
                best_id_step = step
            if math.isfinite(ood) and ood > best_ood_acc:
                best_ood_acc = ood
                best_ood_state = _snapshot(model)
                best_ood_step = step
            hist.append(rec)
            print(
                f"step={step}/{steps} risk={rec['mean_risk']:.4f} val_acc={id_avg:.4f}", flush=True
            )
    return {
        "model": model,
        "history": hist,
        "best_id_state": best_id_state,
        "best_id_acc": best_id_acc,
        "best_ood_state": best_ood_state,
        "best_ood_acc": best_ood_acc,
        "best_id_step": best_id_step,
        "best_ood_step": best_ood_step,
    }

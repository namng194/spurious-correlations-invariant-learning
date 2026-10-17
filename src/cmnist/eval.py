"""Eval: per-env acc, worst-group acc, ID/OOD gap, ECE."""

from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def collect(model, loader: DataLoader, device: str):
    model.eval()
    ps, ys, cs = [], [], []
    for x, y, c in loader:
        x = x.to(device)
        p = torch.sigmoid(model(x)).cpu()
        ps.append(p)
        ys.append(y.cpu())
        cs.append(c.cpu())
    return torch.cat(ps), torch.cat(ys), torch.cat(cs)


def accuracy(p: torch.Tensor, y: torch.Tensor) -> float:
    return ((p > 0.5).float() == y).float().mean().item()


def worst_group_acc(p: torch.Tensor, y: torch.Tensor, color: torch.Tensor) -> dict:
    out = {}
    for yy in (0, 1):
        for cc in (0, 1):
            m = (y == yy) & (color == cc)
            out[f"g_y{yy}c{cc}"] = (
                float(((p[m] > 0.5).float() == y[m]).float().mean()) if m.any() else float("nan")
            )
    vals = [v for v in out.values() if math.isfinite(v)]
    out["worst"] = min(vals) if vals else float("nan")
    for yy in (0, 1):
        for cc in (0, 1):
            out[f"n_y{yy}c{cc}"] = int(((y == yy) & (color == cc)).sum())
    return out


def expected_calibration_error(p: torch.Tensor, y: torch.Tensor, n_bins: int = 15) -> float:
    conf = torch.maximum(p, 1 - p).numpy()
    pred = (p > 0.5).float().numpy()
    true = y.numpy()
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        m = (conf > lo) & (conf <= hi) if b > 0 else (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == true[m]).mean() - conf[m].mean())
    return float(ece)


def summarize(model, loaders: dict[str, DataLoader], device: str) -> dict:
    rep = {}
    for name, loader in loaders.items():
        p, y, c = collect(model, loader, device)
        rep[name] = {
            "acc": accuracy(p, y),
            "ece": expected_calibration_error(p, y),
            **worst_group_acc(p, y, c),
        }
    train_keys = [k for k in rep if k.startswith("train_")]
    if train_keys and "test" in rep:
        id_acc = sum(rep[k]["acc"] for k in train_keys) / len(train_keys)
        rep["ood_gap"] = id_acc - rep["test"]["acc"]
    return rep


@torch.no_grad()
def recoloring_sensitivity(model, loader: DataLoader, device: str) -> dict:
    """Swap red/green channels, preserving shape and any white patch."""
    model.eval()
    changes, flips = [], []
    for x, _, _ in loader:
        x = x.to(device)
        original = torch.sigmoid(model(x))
        swapped = torch.sigmoid(model(x[:, [1, 0, 2]]))
        changes.append((original - swapped).abs().cpu())
        flips.append(((original > 0.5) != (swapped > 0.5)).float().cpu())
    return {
        "mean_probability_change": float(torch.cat(changes).mean()),
        "prediction_flip_rate": float(torch.cat(flips).mean()),
        "n": int(sum(v.numel() for v in changes)),
    }

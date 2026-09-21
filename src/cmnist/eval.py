"""Eval: per-env acc, worst-group acc, ID/OOD gap, ECE."""
from __future__ import annotations

import numpy as np
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
        ys.append(y)
        cs.append(c)
    return torch.cat(ps), torch.cat(ys), torch.cat(cs)


def accuracy(p: torch.Tensor, y: torch.Tensor) -> float:
    return (((p > 0.5).float() == y).float().mean().item())


def worst_group_acc(p: torch.Tensor, y: torch.Tensor, color: torch.Tensor) -> dict:
    out = {}
    for yy in (0, 1):
        for cc in (0, 1):
            m = (y == yy) & (color == cc)
            out[f"g_y{yy}c{cc}"] = float(((p[m] > 0.5).float() == y[m]).float().mean()) if m.any() else float("nan")
    vals = [v for v in out.values() if v == v]
    out["worst"] = min(vals) if vals else float("nan")
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
            ece += m.mean() * abs(pred[m].mean() - true[m].mean())
    return float(ece)


def summarize(model, loaders: dict[str, DataLoader], device: str) -> dict:
    rep = {}
    for name, loader in loaders.items():
        p, y, c = collect(model, loader, device)
        rep[name] = {"acc": accuracy(p, y), "ece": expected_calibration_error(p, y),
                     **worst_group_acc(p, y, c)}
    if "train_e1" in rep and "train_e2" in rep and "test" in rep:
        id_acc = (rep["train_e1"]["acc"] + rep["train_e2"]["acc"]) / 2
        rep["ood_gap"] = id_acc - rep["test"]["acc"]
    return rep

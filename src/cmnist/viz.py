"""Viz helpers: reliability diagram, sample grid, saliency maps."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_samples(env: dict, n: int = 8, path: str | None = None):
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2))
    for i in range(n):
        img = env["x"][i].permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].set_title(f"y={int(env['y'][i])} c={int(env['color'][i])}")
        axes[i].axis("off")
    fig.suptitle(f"{env['name']} (p_e={env['p_e']})")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def reliability_diagram(p, y, n_bins: int = 15, path: str | None = None):
    p = np.asarray(p)
    y = np.asarray(y)
    conf = np.maximum(p, 1 - p)
    pred = (p > 0.5).astype(float)
    xs, accs, confs = [], [], []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        m = (conf > lo) & (conf <= hi) if b > 0 else (conf <= hi)
        if m.any():
            xs.append((lo + hi) / 2)
            accs.append(pred[m].mean())
            confs.append(conf[m].mean())
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "--", label="perfect")
    ax.plot(confs, accs, "o-", label="model")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.legend()
    ax.set_title("Reliability diagram")
    if path:
        fig.savefig(path, dpi=150)
    return fig


@torch.enable_grad()
def saliency(model, x: torch.Tensor, device: str) -> np.ndarray:
    """Vanilla gradient magnitude over input, summed over channels -> (28,28)."""
    model.eval()
    x = x.unsqueeze(0).to(device).requires_grad_(True)
    s = model(x).sum()
    g = torch.autograd.grad(s, x)[0].detach().cpu().squeeze(0)
    return g.abs().sum(0).numpy()

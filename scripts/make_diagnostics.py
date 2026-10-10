"""Diagnostics nối các pillar: calibration + XAI + probe (chạy nhanh, dùng fast config).

- Reliability diagram ERM vs IRM trên OOD test (Pillar Reliability §4.2.1)
- Saliency grid: ERM nhìn nền màu, IRM nhìn nét số (Pillar Explainability §5.3.2)
- Linear probe: representation còn encode màu không (đo decodability spurious)

Usage: uv run python scripts/make_diagnostics.py --config configs/fast.yaml --out outputs/figs
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cmnist.data import build_colored_mnist, to_loader  # noqa: E402
from cmnist.eval import collect, expected_calibration_error  # noqa: E402
from cmnist.models import make_model  # noqa: E402
from cmnist.train import train_one_run  # noqa: E402
from cmnist.viz import integrated_gradients, reliability_diagram, saliency  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fast.yaml")
    ap.add_argument("--out", default="outputs/figs")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    envs = build_colored_mnist(
        data_dir=cfg["data"]["data_dir"], label_noise=cfg["data"]["label_noise"],
        seed=cfg["data"]["seed"], dataset=cfg["data"].get("dataset", "mnist"),
    )
    loaders = {k: to_loader(v, batch_size=cfg["data"]["batch_size"]) for k, v in envs.items()}
    models = {}
    for method in ("erm", "irm"):
        kw = {"penalty_weight": 1e4, "rex_beta": 1e3} if method == "irm" else {}
        res = train_one_run(
            make_model(cfg["model"]),
            [loaders["train_e1"], loaders["train_e2"]], loaders,
            method=method, steps=cfg["train"]["steps"], lr=cfg["train"]["lr"],
            anneal_steps=cfg["train"]["anneal_steps"],
            device=device, log_every=cfg["train"]["steps"], seed=0, **kw,
        )
        models[method] = res["model"]

    # 1) reliability diagrams on OOD test + ECE values
    import json

    ece = {}
    for method, model in models.items():
        p, y, _ = collect(model, loaders["test"], device)
        reliability_diagram(p.numpy(), y.numpy(),
                            path=os.path.join(args.out, f"reliability_{method}.png"))
        plt.close("all")
        ece[method] = expected_calibration_error(p, y)
    print(f"saved reliability_*.png ECE={ece}")

    # 2) attribution grid: vanilla saliency + Integrated Gradients, ERM vs IRM
    x4 = envs["test"]["x"][:4]
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for j in range(4):
        for i, m in enumerate(("erm", "irm")):
            s = saliency(models[m], x4[j], device)
            axes[i, j].imshow(x4[j].permute(1, 2, 0).numpy(), alpha=0.4)
            axes[i, j].imshow(s, cmap="hot", alpha=0.6)
            axes[i, j].set_title(f"{m} sal y={int(envs['test']['y'][j])}")
            axes[i, j].axis("off")
    fig.suptitle("Vanilla-gradient saliency (baseline; sanity-check caveat)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "saliency_erm_vs_irm.png"), dpi=150)
    plt.close("all")
    print("saved saliency_erm_vs_irm.png")

    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for j in range(4):
        for i, m in enumerate(("erm", "irm")):
            s = integrated_gradients(models[m], x4[j], device, steps=32)
            axes[i, j].imshow(x4[j].permute(1, 2, 0).numpy(), alpha=0.4)
            axes[i, j].imshow(s, cmap="hot", alpha=0.6)
            axes[i, j].set_title(f"{m} IG y={int(envs['test']['y'][j])}")
            axes[i, j].axis("off")
    fig.suptitle("Integrated Gradients (primary XAI evidence)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "ig_erm_vs_irm.png"), dpi=150)
    plt.close("all")
    print("saved ig_erm_vs_irm.png")

    # 3) linear probe: is color decodable from features? (DFR head-only hypothesis)
    from sklearn.linear_model import LogisticRegression

    probe = {"ece": ece}
    for method, model in models.items():
        feats, _, colors = [], [], []
        model.eval()
        with torch.no_grad():
            for x, _, c in loaders["test"]:
                _, h = model(x.to(device), return_feat=True)
                feats.append(h.cpu())
                colors.append(c)
        X = torch.cat(feats).numpy()
        Cc = torch.cat(colors).numpy()
        idx = np.random.default_rng(0).choice(len(X), size=min(4000, len(X)), replace=False)
        cut = len(idx) // 2
        tr, te = idx[:cut], idx[cut:]
        clf = LogisticRegression(max_iter=200).fit(X[tr], Cc[tr])
        probe[method] = {"train": float(clf.score(X[tr], Cc[tr])),
                         "heldout": float(clf.score(X[te], Cc[te]))}
        print(f"probe color-decodability {method}: train={probe[method]['train']:.3f} "
              f"heldout={probe[method]['heldout']:.3f} "
              f"(high = color retained in representation, consistent with DFR)")
    json.dump(probe, open(os.path.join(args.out, "diagnostics.json"), "w"), indent=2)


if __name__ == "__main__":
    main()

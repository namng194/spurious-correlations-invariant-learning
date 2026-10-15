"""Diagnostics nối các pillar: calibration + XAI + probe (chạy nhanh, dùng fast config).

- Reliability diagram ERM vs IRM trên OOD test (Pillar Reliability §4.2.1)
- Attribution maps plus a counterfactual recoloring test for color dependence
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
from cmnist.data import build_colored_mnist, experiment_loaders, split_environments
from cmnist.eval import collect, expected_calibration_error, recoloring_sensitivity
from cmnist.experiment import save_run, write_json
from cmnist.models import make_model
from cmnist.train import train_one_run
from cmnist.viz import integrated_gradients, reliability_diagram, saliency


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fast.yaml")
    ap.add_argument("--out", default="outputs/figs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--checkpoint-root", help="Reuse sweep trainval checkpoints instead of retraining"
    )
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as config_file:
        cfg = yaml.safe_load(config_file)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    envs = build_colored_mnist(
        data_dir=cfg["data"]["data_dir"],
        label_noise=cfg["data"]["label_noise"],
        seed=cfg["data"]["seed"],
        dataset=cfg["data"].get("dataset", "mnist"),
    )
    envs = split_environments(
        envs, cfg["data"].get("val_fraction", 0.2), cfg["data"].get("split_seed", 0)
    )
    models, selected_steps = {}, {}
    for method in ("erm", "irm", "rex"):
        training, validation, loaders = experiment_loaders(
            envs, cfg["data"]["batch_size"], args.seed, storage_device=device
        )
        if args.checkpoint_root:
            path = os.path.join(args.checkpoint_root, f"{method}_s{args.seed}", "trainval.pt")
            checkpoint = torch.load(path, map_location=device, weights_only=True)
            metadata = checkpoint["metadata"]
            if (
                metadata["config"] != cfg
                or metadata["seed"] != args.seed
                or checkpoint["protocol"] != "trainval"
            ):
                raise ValueError(f"Checkpoint configuration does not match diagnostics: {path}")
            model = make_model(cfg["model"]).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            models[method] = model
            selected_steps[method] = checkpoint["step"]
            continue
        mcfg = cfg["methods"].get(method, {})
        res = train_one_run(
            lambda: make_model(cfg["model"]),
            training,
            validation,
            method=method,
            steps=cfg["train"]["steps"],
            lr=cfg["train"]["lr"],
            anneal_steps=cfg["train"]["anneal_steps"],
            penalty_weight=float(mcfg.get("penalty_weight", 1e4)),
            rex_beta=float(mcfg.get("rex_beta", 1e3)),
            device=device,
            log_every=cfg["train"].get("log_every", 200),
            seed=args.seed,
        )
        save_run(
            res,
            loaders,
            envs,
            device,
            os.path.join(args.out, "runs", f"{method}_s{args.seed}"),
            {
                "config": cfg,
                "method": method,
                "seed": args.seed,
                "steps": cfg["train"]["steps"],
                "model": cfg["model"],
            },
        )
        res["model"].load_state_dict(res["best_id_state"])
        models[method] = res["model"]
        selected_steps[method] = res["best_id_step"]

    # 1) reliability diagrams on OOD test + ECE values
    ece = {}
    for method, model in models.items():
        p, y, _ = collect(model, loaders["test"], device)
        reliability_diagram(
            p.numpy(), y.numpy(), path=os.path.join(args.out, f"reliability_{method}.png")
        )
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
    fig.suptitle("Integrated Gradients (spatial attribution; not a color-dependence test)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "ig_erm_vs_irm.png"), dpi=150)
    plt.close("all")
    print("saved ig_erm_vs_irm.png")

    # 3) linear probe: is color decodable from features? (DFR head-only hypothesis)
    from sklearn.linear_model import LogisticRegression

    probe = {
        "ece": ece,
        "seed": args.seed,
        "protocol": "trainval",
        "checkpoint_steps": selected_steps,
        "recoloring": {
            m: recoloring_sensitivity(model, loaders["test"], device) for m, model in models.items()
        },
    }
    for method, model in models.items():
        feats, _, colors = [], [], []
        model.eval()
        with torch.no_grad():
            for x, _, c in loaders["test"]:
                _, h = model(x.to(device), return_feat=True)
                feats.append(h.cpu())
                colors.append(c.cpu())
        X = torch.cat(feats).numpy()
        Cc = torch.cat(colors).numpy()
        idx = np.random.default_rng(0).choice(len(X), size=min(4000, len(X)), replace=False)
        cut = len(idx) // 2
        tr, te = idx[:cut], idx[cut:]
        clf = LogisticRegression(max_iter=200).fit(X[tr], Cc[tr])
        probe[method] = {
            "train": float(clf.score(X[tr], Cc[tr])),
            "heldout": float(clf.score(X[te], Cc[te])),
        }
        print(
            f"probe color-decodability {method}: train={probe[method]['train']:.3f} "
            f"heldout={probe[method]['heldout']:.3f} "
            f"(high = color retained in representation, consistent with DFR)"
        )
    write_json(os.path.join(args.out, "diagnostics.json"), probe)


if __name__ == "__main__":
    main()

"""Full sweep: methods x seeds (+ optional lambda sweep). Saves results.json/csv.

Usage (uv-only):
  uv run python scripts/run_sweep.py --config configs/base.yaml --out outputs/sweep
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch

from cmnist.data import build_colored_mnist, experiment_loaders, split_environments
from cmnist.experiment import completed_run, save_run, write_json
from cmnist.models import make_model
from cmnist.train import train_one_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out", default="outputs/sweep")
    ap.add_argument("--quick", action="store_true", help="tiny smoke run (CPU, 200 steps)")
    ap.add_argument("--resume", action="store_true", help="Reuse complete matching runs")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as config_file:
        cfg = yaml.safe_load(config_file)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    os.makedirs(args.out, exist_ok=True)
    steps = 200 if args.quick else cfg["train"]["steps"]
    seeds = [0] if args.quick else cfg["train"]["seeds"]

    envs = build_colored_mnist(
        data_dir=cfg["data"]["data_dir"],
        label_noise=cfg["data"]["label_noise"],
        seed=cfg["data"]["seed"],
        dataset=cfg["data"].get("dataset", "mnist"),
    )
    bs = 512 if args.quick else cfg["data"]["batch_size"]
    envs = split_environments(
        envs, cfg["data"].get("val_fraction", 0.2), cfg["data"].get("split_seed", 0)
    )

    rows = []
    for method, mcfg in cfg["methods"].items():
        for seed in seeds:
            print(f"[{method} seed={seed}] ...", flush=True)
            run_dir = os.path.join(args.out, f"{method}_s{seed}")
            metadata = {
                "config": cfg,
                "method": method,
                "seed": seed,
                "steps": steps,
                "batch_size": bs,
                "model": cfg["model"],
            }
            evaluations = completed_run(run_dir, metadata) if args.resume else None
            if evaluations is not None:
                print("  reusing completed run", flush=True)
            else:
                training, validation, loaders = experiment_loaders(
                    envs, bs, seed, storage_device=device
                )
                res = train_one_run(
                    lambda: make_model(cfg["model"]),
                    training,
                    validation,
                    method=method,
                    steps=steps,
                    lr=cfg["train"]["lr"],
                    penalty_weight=float(mcfg.get("penalty_weight", 1e4)),
                    anneal_steps=cfg["train"]["anneal_steps"],
                    rex_beta=float(mcfg.get("rex_beta", 1e3)),
                    device=device,
                    log_every=cfg["train"].get("log_every", 200),
                    seed=seed,
                )
                evaluations = save_run(res, loaders, envs, device, run_dir, metadata)
                del res, training, validation, loaders
            for proto, evaluation in evaluations.items():
                summ = evaluation["metrics"]
                row = {
                    "method": method,
                    "seed": seed,
                    "protocol": proto,
                    "checkpoint_step": evaluation["step"],
                    "train_acc": (summ["train_e1"]["acc"] + summ["train_e2"]["acc"]) / 2,
                    "test_acc": summ["test"]["acc"],
                    "ood_gap": summ.get("ood_gap", float("nan")),
                    "test_worst": summ["test"]["worst"],
                    "test_ece": summ["test"]["ece"],
                }
                rows.append(row)
                print(f"  -> [{proto}] {row}", flush=True)
            write_json(os.path.join(args.out, "results.json"), rows)

    write_json(os.path.join(args.out, "results.json"), rows)
    with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {len(rows)} runs -> {args.out}")


if __name__ == "__main__":
    main()

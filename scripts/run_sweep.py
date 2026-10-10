"""Full sweep: methods x seeds (+ optional lambda sweep). Saves results.json/csv.

Usage (uv-only):
  uv run python scripts/run_sweep.py --config configs/base.yaml --out outputs/sweep
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch  # noqa: E402

from cmnist.data import EnvSpec, build_colored_mnist, to_loader  # noqa: E402
from cmnist.eval import summarize  # noqa: E402
from cmnist.models import make_model  # noqa: E402
from cmnist.train import train_one_run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out", default="outputs/sweep")
    ap.add_argument("--quick", action="store_true", help="tiny smoke run (CPU, 200 steps)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
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
    loaders = {k: to_loader(v, batch_size=bs, shuffle=(k != "test")) for k, v in envs.items()}

    rows = []
    for method, mcfg in cfg["methods"].items():
        for seed in seeds:
            print(f"[{method} seed={seed}] ...", flush=True)
            res = train_one_run(
                make_model(cfg["model"]),
                [loaders["train_e1"], loaders["train_e2"]],
                {"train_e1": loaders["train_e1"], "train_e2": loaders["train_e2"],
                 "test": loaders["test"]},
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
            # Two model-selection protocols (DomainBed): last checkpoint vs
            # train-domain-validation (best ID avg) vs OOD oracle (best test).
            import copy

            base_model = make_model(cfg["model"])
            evals = {}
            evals["last"] = summarize(res["model"], loaders, device)
            if res["best_id_state"] is not None:
                base_model.load_state_dict(
                    {k: v for k, v in res["best_id_state"].items()})
                base_model.to(device)
                evals["trainval"] = summarize(base_model, loaders, device)
            if res["best_ood_state"] is not None:
                base_model.load_state_dict(
                    {k: v for k, v in res["best_ood_state"].items()})
                base_model.to(device)
                evals["oracle"] = summarize(base_model, loaders, device)
            for proto, summ in evals.items():
                row = {"method": method, "seed": seed, "protocol": proto,
                       "train_acc": (summ["train_e1"]["acc"] + summ["train_e2"]["acc"]) / 2,
                       "test_acc": summ["test"]["acc"],
                       "ood_gap": summ.get("ood_gap", float("nan")),
                       "test_worst": summ["test"]["worst"],
                       "test_ece": summ["test"]["ece"]}
                rows.append(row)
                print(f"  -> [{proto}] {row}", flush=True)

    json.dump(rows, open(os.path.join(args.out, "results.json"), "w"), indent=2)
    with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {len(rows)} runs -> {args.out}")


if __name__ == "__main__":
    main()

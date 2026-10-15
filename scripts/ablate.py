"""Ablations tạo đóng góp (đọc heavy.yaml -> mục `ablate`).

Cover:
  A1 lambda-sweep IRM (chữ U ngược: ERM-collapse ↔ sweet spot ↔ constant-collapse)
  A2 backbone mlp vs cnn (overparam demo — Zhou et al. SparseIRM)
  A3 failure demos: single-env IRM | patch spurious thứ 2 | small-n per env
  A4 cross-dataset: Colored-FashionMNIST quick check

Usage:
  uv run python scripts/ablate.py --config configs/heavy.yaml --out outputs/ablate --steps 200
  (bỏ --steps để chạy full 2000 steps x seeds x configs ~4-6h T4)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch

from cmnist.data import EnvSpec, build_colored_mnist, experiment_loaders, split_environments
from cmnist.experiment import completed_run, save_run, selected_row, training_signature, write_json
from cmnist.models import make_model
from cmnist.train import train_one_run


def run_case(
    envs,
    method,
    backbone,
    seed,
    steps,
    device,
    bs,
    log_every=200,
    val_fraction=0.2,
    split_seed=0,
    out_dir=None,
    config=None,
    resume=False,
    **kw,
) -> dict:
    metadata = {
        "config": config,
        "method": method,
        "model": backbone,
        "seed": seed,
        "steps": steps,
        "batch_size": bs,
        "val_fraction": val_fraction,
        "split_seed": split_seed,
        **kw,
    }
    evaluations = completed_run(out_dir, metadata) if resume else None
    if evaluations is None:
        envs = split_environments(envs, val_fraction, split_seed)
        training, validation, loaders = experiment_loaders(envs, bs, seed, storage_device=device)
        res = train_one_run(
            lambda: make_model(backbone),
            training,
            validation,
            method=method,
            steps=steps,
            device=device,
            log_every=log_every,
            seed=seed,
            **kw,
        )
        evaluations = save_run(res, loaders, envs, device, out_dir, metadata)
    return selected_row(evaluations["trainval"], seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/heavy.yaml")
    ap.add_argument("--out", default="outputs/ablate")
    ap.add_argument("--steps", type=int, default=0, help="0 = dùng steps trong config")
    ap.add_argument("--seed", type=int, default=-1, help="-1 = use all seeds in config")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--reuse-sweep", help="Reuse matching baseline sweep checkpoints")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as config_file:
        cfg = yaml.safe_load(config_file)
    ab = cfg.get("ablate", {})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    steps = args.steps or cfg["train"]["steps"]
    bs = cfg["data"]["batch_size"]
    dd = cfg["data"]["data_dir"]
    ln = cfg["data"]["label_noise"]
    log_every = cfg["train"].get("log_every", 200)
    seeds = cfg["train"]["seeds"] if args.seed < 0 else [args.seed]
    rows: list[dict] = []

    def add(name, **kw):
        short = {k: v for k, v in kw.items() if k != "envs"}
        print(f"[{name}] {short}", flush=True)
        r = None
        if args.reuse_sweep and kw["envs"] is base_envs and kw["backbone"] == "mlp":
            method, seed = kw["method"], kw["seed"]
            source = Path(args.reuse_sweep) / f"{method}_s{seed}" / "run.json"
            if source.exists():
                record = json.loads(source.read_text(encoding="utf-8"))
                desired = training_signature(
                    cfg,
                    method,
                    model=kw["backbone"],
                    steps=steps,
                    **{
                        k: v
                        for k, v in kw.items()
                        if k in ("penalty_weight", "rex_beta", "anneal_steps")
                    },
                )
                metadata = record["metadata"]
                actual = training_signature(
                    metadata["config"], method, model=metadata["model"], steps=metadata["steps"]
                )
                actual["batch_size"] = metadata["batch_size"]
                complete = completed_run(source.parent, {"seed": seed, "method": method})
                if desired == actual and complete is not None:
                    r = selected_row(record["evaluations"]["trainval"], seed)
                    r["reused_from"] = str(source)
                    write_json(Path(args.out) / name / "reused.json", r)
                    print(f"  reused {source}", flush=True)
        if r is None:
            r = run_case(
                log_every=log_every,
                lr=cfg["train"]["lr"],
                val_fraction=cfg["data"].get("val_fraction", 0.2),
                split_seed=cfg["data"].get("split_seed", 0),
                out_dir=os.path.join(args.out, name),
                config=cfg,
                resume=args.resume,
                **kw,
            )
        rows.append({"case": name, **r})
        write_json(os.path.join(args.out, "ablate.json"), rows)
        print(f"  -> test={r['test_acc']:.3f} gap={r['ood_gap']:.3f}", flush=True)

    base_envs = build_colored_mnist(data_dir=dd, label_noise=ln, seed=cfg["data"]["seed"])

    # A1: lambda sweep (IRM stability curve), all seeds
    for lam in ab.get("lambda_sweep", [1e2, 1e3, 1e4, 3e4]):
        for seed in seeds:
            add(
                f"irm_lam{lam:g}_s{seed}",
                envs=base_envs,
                method="irm",
                backbone="mlp",
                seed=seed,
                steps=steps,
                device=device,
                bs=bs,
                penalty_weight=float(lam),
                anneal_steps=cfg["train"]["anneal_steps"],
            )

    # A1b: REx beta sweep (stability comparison)
    for beta in ab.get("rex_beta_sweep", [1e2, 1e3, 1e4]):
        for seed in seeds:
            add(
                f"rex_beta{beta:g}_s{seed}",
                envs=base_envs,
                method="rex",
                backbone="mlp",
                seed=seed,
                steps=steps,
                device=device,
                bs=bs,
                rex_beta=float(beta),
                anneal_steps=cfg["train"]["anneal_steps"],
            )

    # A2: backbone compare (erm + irm)
    for bb in ab.get("backbones", ["mlp", "cnn"]):
        for m in ("erm", "irm"):
            for seed in seeds:
                add(
                    f"{m}_{bb}_s{seed}",
                    envs=base_envs,
                    method=m,
                    backbone=bb,
                    seed=seed,
                    steps=steps,
                    device=device,
                    bs=bs,
                    penalty_weight=1e4,
                    anneal_steps=cfg["train"]["anneal_steps"],
                )

    # A3a: single-env IRM (phải fail — Byrne/Rosenfeld: E=1 không đủ)
    single = build_colored_mnist(
        data_dir=dd,
        label_noise=ln,
        seed=cfg["data"]["seed"],
        env_specs=[EnvSpec("train_e1", p_e=0.1, train=True), EnvSpec("test", p_e=0.9, train=False)],
    )
    for seed in seeds:
        add(
            f"irm_single_env_s{seed}",
            envs=single,
            method="irm",
            backbone="mlp",
            seed=seed,
            steps=steps,
            device=device,
            bs=bs,
            penalty_weight=1e4,
            anneal_steps=cfg["train"]["anneal_steps"],
        )

    # A3b: spurious thứ 2 (patch) — IRM 2 env phải gãy theo ERM
    patched = build_colored_mnist(
        data_dir=dd,
        label_noise=ln,
        seed=cfg["data"]["seed"],
        env_specs=[
            EnvSpec("train_e1", p_e=0.1, train=True, p_patch_e=ab.get("patch_p_e", 0.1)),
            EnvSpec("train_e2", p_e=0.2, train=True, p_patch_e=ab.get("patch_p_e", 0.1)),
            EnvSpec("test", p_e=0.9, train=False, p_patch_e=0.9),
        ],
    )
    for m in ("erm", "irm"):
        for seed in seeds:
            add(
                f"{m}_patch2spur_s{seed}",
                envs=patched,
                method=m,
                backbone="mlp",
                seed=seed,
                steps=steps,
                device=device,
                bs=bs,
                penalty_weight=1e4,
                anneal_steps=cfg["train"]["anneal_steps"],
            )

    # A3c: small-n per env
    small = build_colored_mnist(
        data_dir=dd,
        label_noise=ln,
        seed=cfg["data"]["seed"],
        env_specs=[
            EnvSpec("train_e1", p_e=0.1, train=True, n=ab.get("small_n", 6250)),
            EnvSpec("train_e2", p_e=0.2, train=True, n=ab.get("small_n", 6250)),
            EnvSpec("test", p_e=0.9, train=False),
        ],
    )
    for m in ("erm", "irm"):
        for seed in seeds:
            add(
                f"{m}_smalln_s{seed}",
                envs=small,
                method=m,
                backbone="mlp",
                seed=seed,
                steps=steps,
                device=device,
                bs=bs,
                penalty_weight=1e4,
                anneal_steps=cfg["train"]["anneal_steps"],
            )

    # A4: cross-dataset FashionMNIST (erm vs irm)
    if ab.get("second_dataset") == "fashion":
        fenvs = build_colored_mnist(
            data_dir=dd, label_noise=ln, seed=cfg["data"]["seed"], dataset="fashion"
        )
        for m in ("erm", "irm"):
            for seed in seeds:
                add(
                    f"{m}_fashion_s{seed}",
                    envs=fenvs,
                    method=m,
                    backbone="mlp",
                    seed=seed,
                    steps=steps,
                    device=device,
                    bs=bs,
                    penalty_weight=1e4,
                    anneal_steps=cfg["train"]["anneal_steps"],
                )

    os.makedirs(args.out, exist_ok=True)
    write_json(os.path.join(args.out, "ablate.json"), rows)
    print(f"saved {len(rows)} cases -> {args.out}/ablate.json")


if __name__ == "__main__":
    main()

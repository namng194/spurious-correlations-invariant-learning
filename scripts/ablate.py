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

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch  # noqa: E402

from cmnist.data import EnvSpec, build_colored_mnist, to_loader  # noqa: E402
from cmnist.eval import summarize  # noqa: E402
from cmnist.models import make_model  # noqa: E402
from cmnist.train import train_one_run  # noqa: E402


def run_case(envs, method, backbone, seed, steps, device, bs, log_every=200, **kw) -> dict:
    loaders = {k: to_loader(v, batch_size=bs, shuffle=(k != "test")) for k, v in envs.items()}
    train_keys = [k for k in ("train_e1", "train_e2") if k in loaders]
    res = train_one_run(
        make_model(backbone), [loaders[k] for k in train_keys], loaders,
        method=method, steps=steps, device=device, log_every=log_every, seed=seed, **kw,
    )
    # Honest reporting: train-domain-validation checkpoint (best ID avg).
    if res.get("best_id_state") is not None:
        model = make_model(backbone)
        model.load_state_dict({k: v for k, v in res["best_id_state"].items()})
        model.to(device)
    else:
        model = res["model"]
    s = summarize(model, loaders, device)
    return {
        "train_acc": sum(s[k]["acc"] for k in train_keys) / len(train_keys),
        "test_acc": s["test"]["acc"],
        "ood_gap": s.get("ood_gap", float("nan")),
        "test_worst": s["test"]["worst"],
        "test_ece": s["test"]["ece"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/heavy.yaml")
    ap.add_argument("--out", default="outputs/ablate")
    ap.add_argument("--steps", type=int, default=0, help="0 = dùng steps trong config")
    ap.add_argument("--seed", type=int, default=-1, help="-1 = use all seeds in config")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
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
        r = run_case(log_every=log_every, **kw)
        rows.append({"case": name, **r})
        print(f"  -> test={r['test_acc']:.3f} gap={r['ood_gap']:.3f}", flush=True)

    base_envs = build_colored_mnist(data_dir=dd, label_noise=ln, seed=cfg["data"]["seed"])

    # A1: lambda sweep (IRM stability curve), all seeds
    for lam in ab.get("lambda_sweep", [1e2, 1e3, 1e4, 3e4]):
        for seed in seeds:
            add(f"irm_lam{lam:g}_s{seed}", envs=base_envs, method="irm", backbone="mlp",
                seed=seed, steps=steps, device=device, bs=bs,
                penalty_weight=float(lam), anneal_steps=cfg["train"]["anneal_steps"])

    # A1b: REx beta sweep (stability comparison)
    for beta in ab.get("rex_beta_sweep", [1e2, 1e3, 1e4]):
        for seed in seeds:
            add(f"rex_beta{beta:g}_s{seed}", envs=base_envs, method="rex", backbone="mlp",
                seed=seed, steps=steps, device=device, bs=bs,
                rex_beta=float(beta), anneal_steps=cfg["train"]["anneal_steps"])

    # A2: backbone compare (erm + irm)
    for bb in ab.get("backbones", ["mlp", "cnn"]):
        for m in ("erm", "irm"):
            for seed in seeds:
                add(f"{m}_{bb}_s{seed}", envs=base_envs, method=m, backbone=bb,
                    seed=seed, steps=steps, device=device, bs=bs,
                    penalty_weight=1e4, anneal_steps=cfg["train"]["anneal_steps"])

    # A3a: single-env IRM (phải fail — Byrne/Rosenfeld: E=1 không đủ)
    single = build_colored_mnist(
        data_dir=dd, label_noise=ln, seed=cfg["data"]["seed"],
        env_specs=[EnvSpec("train_e1", p_e=0.1, train=True),
                   EnvSpec("test", p_e=0.9, train=False)],
    )
    for seed in seeds:
        add(f"irm_single_env_s{seed}", envs=single, method="irm", backbone="mlp",
            seed=seed, steps=steps, device=device, bs=bs,
            penalty_weight=1e4, anneal_steps=cfg["train"]["anneal_steps"])

    # A3b: spurious thứ 2 (patch) — IRM 2 env phải gãy theo ERM
    patched = build_colored_mnist(
        data_dir=dd, label_noise=ln, seed=cfg["data"]["seed"],
        env_specs=[EnvSpec("train_e1", p_e=0.1, train=True,
                           p_patch_e=ab.get("patch_p_e", 0.1)),
                   EnvSpec("train_e2", p_e=0.2, train=True,
                           p_patch_e=ab.get("patch_p_e", 0.1)),
                   EnvSpec("test", p_e=0.9, train=False, p_patch_e=0.9)],
    )
    for m in ("erm", "irm"):
        for seed in seeds:
            add(f"{m}_patch2spur_s{seed}", envs=patched, method=m, backbone="mlp",
                seed=seed, steps=steps, device=device, bs=bs,
                penalty_weight=1e4, anneal_steps=cfg["train"]["anneal_steps"])

    # A3c: small-n per env
    small = build_colored_mnist(
        data_dir=dd, label_noise=ln, seed=cfg["data"]["seed"],
        env_specs=[EnvSpec("train_e1", p_e=0.1, train=True, n=ab.get("small_n", 6250)),
                   EnvSpec("train_e2", p_e=0.2, train=True, n=ab.get("small_n", 6250)),
                   EnvSpec("test", p_e=0.9, train=False)],
    )
    for m in ("erm", "irm"):
        for seed in seeds:
            add(f"{m}_smalln_s{seed}", envs=small, method=m, backbone="mlp",
                seed=seed, steps=steps, device=device, bs=bs,
                penalty_weight=1e4, anneal_steps=cfg["train"]["anneal_steps"])

    # A4: cross-dataset FashionMNIST (erm vs irm)
    if ab.get("second_dataset") == "fashion":
        fenvs = build_colored_mnist(data_dir=dd, label_noise=ln,
                                    seed=cfg["data"]["seed"], dataset="fashion")
        for m in ("erm", "irm"):
            for seed in seeds:
                add(f"{m}_fashion_s{seed}", envs=fenvs, method=m, backbone="mlp",
                    seed=seed, steps=steps, device=device, bs=bs,
                    penalty_weight=1e4, anneal_steps=cfg["train"]["anneal_steps"])

    os.makedirs(args.out, exist_ok=True)
    json.dump(rows, open(os.path.join(args.out, "ablate.json"), "w"), indent=2)
    print(f"saved {len(rows)} cases -> {args.out}/ablate.json")


if __name__ == "__main__":
    main()

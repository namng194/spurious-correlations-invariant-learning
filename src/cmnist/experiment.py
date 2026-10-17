"""Persist replayable run metadata, checkpoints, and complete evaluation metrics."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import torch

from .eval import collect, summarize


def completed_run(out_dir, expected_metadata):
    """Resume only complete, matching runs; never silently reuse a different protocol."""
    out = Path(out_dir)
    if not (out / "run.json").exists():
        return None
    record = json.loads((out / "run.json").read_text(encoding="utf-8"))
    if any(record["metadata"].get(k) != v for k, v in expected_metadata.items()):
        raise ValueError(f"Existing run configuration differs: {out}")
    if not all((out / f"{p}.pt").exists() for p in ("last", "trainval", "oracle")):
        return None
    return record["evaluations"]


def selected_row(evaluation, seed):
    metrics = evaluation["metrics"]
    train = [v["acc"] for k, v in metrics.items() if k.startswith("train_")]
    return {
        "seed": seed,
        "protocol": "trainval",
        "checkpoint_step": evaluation["step"],
        "train_acc": sum(train) / len(train),
        "test_acc": metrics["test"]["acc"],
        "ood_gap": metrics["ood_gap"],
        "test_worst": metrics["test"]["worst"],
        "test_ece": metrics["test"]["ece"],
    }


def training_signature(cfg, method, model=None, steps=None, **overrides):
    """Compare executed settings, ignoring unrelated ablation-grid configuration."""
    data = cfg["data"]
    train = cfg["train"]
    mcfg = cfg["methods"].get(method, {})
    signature = {
        "model": model or cfg["model"],
        "steps": steps or train["steps"],
        "lr": train["lr"],
        "log_every": train.get("log_every", 200),
        "data_dir": data["data_dir"],
        "dataset": data.get("dataset", "mnist"),
        "data_seed": data["seed"],
        "label_noise": data["label_noise"],
        "batch_size": data["batch_size"],
        "val_fraction": data.get("val_fraction", 0.2),
        "split_seed": data.get("split_seed", 0),
    }
    if method == "irm":
        signature.update(
            anneal_steps=overrides.get("anneal_steps", train["anneal_steps"]),
            penalty_weight=float(overrides.get("penalty_weight", mcfg.get("penalty_weight", 1e4))),
        )
    elif method == "rex":
        signature["rex_beta"] = float(overrides.get("rex_beta", mcfg.get("rex_beta", 1e3)))
    return signature


def write_json(path, value):
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return x

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")


def save_run(result, loaders, envs, device, out_dir, metadata):
    """Save all selection protocols without constructing another random model."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = {
        "last": {k: v.detach().cpu().clone() for k, v in result["model"].state_dict().items()},
        "trainval": result["best_id_state"],
        "oracle": result["best_ood_state"],
    }
    steps = {
        "last": metadata["steps"],
        "trainval": result["best_id_step"],
        "oracle": result["best_ood_step"],
    }
    model = copy.deepcopy(result["model"])
    evaluations = {}
    for protocol, state in checkpoints.items():
        if state is None:
            continue
        model.load_state_dict(state)
        metrics = summarize(model, loaders, device)
        # Sequential evaluation loaders preserve environment metadata alignment.
        p, y, c = collect(model, loaders["test"], device)
        patch = envs["test"].get("patch")
        if patch is not None:
            groups = {}
            for yy in (0, 1):
                for cc in (0, 1):
                    for pp in (0, 1):
                        mask = (y == yy) & (c == cc) & (patch == pp)
                        n = int(mask.sum())
                        groups[f"y{yy}c{cc}p{pp}"] = {
                            "n": n,
                            "acc": float(((p[mask] > 0.5) == y[mask]).float().mean())
                            if n
                            else None,
                        }
            valid = [g["acc"] for g in groups.values() if g["n"]]
            metrics["test"]["patch_groups"] = groups
            metrics["test"]["patch_worst"] = min(valid) if valid else None
        evaluations[protocol] = {"step": steps[protocol], "metrics": metrics}
        torch.save(
            {
                "state_dict": state,
                "metadata": metadata,
                "protocol": protocol,
                "step": steps[protocol],
            },
            out / f"{protocol}.pt",
        )
    write_json(
        out / "run.json",
        {
            "metadata": {
                **metadata,
                "torch_version": str(torch.__version__),
                "device": device,
                "rex_warmup": False,
            },
            "splits": {
                k: v["source_indices"].tolist() for k, v in envs.items() if "source_indices" in v
            },
            "history": result["history"],
            "evaluations": evaluations,
        },
    )
    return evaluations

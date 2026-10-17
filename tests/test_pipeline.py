"""Offline regression tests for metrics, selection, seeds and artifacts."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from cmnist.data import EnvSpec, build_env, experiment_loaders, split_environments
from cmnist.eval import expected_calibration_error, recoloring_sensitivity, worst_group_acc
from cmnist.experiment import save_run
from cmnist.losses import compute_objective
from cmnist.models import MLP
from cmnist.train import train_one_run
from cmnist.viz import reliability_diagram


@pytest.fixture
def envs():
    rng = np.random.default_rng(8)
    imgs = rng.random((40, 28, 28), dtype=np.float32)
    digits = np.arange(40) % 10
    return {
        name: build_env(imgs, digits, EnvSpec(name, p, p_patch_e=0.2), 0.25, rng)
        for name, p in (("train_e1", 0.1), ("train_e2", 0.2), ("test", 0.9))
    }


@pytest.mark.parametrize(
    "p,y,expected",
    [
        ([0.9, 0.1], [0, 1], 0.9),
        ([0.9, 0.1], [1, 0], 0.1),
        ([1.0, 0.0], [1, 0], 0.0),
        ([0.5, 0.5], [0, 1], 0.0),
    ],
)
def test_ece_known_answers(p, y, expected):
    assert expected_calibration_error(torch.tensor(p), torch.tensor(y)) == pytest.approx(expected)


def test_calibration_weighting_and_diagram():
    p = torch.tensor([0.6, 0.6, 0.6, 1.0])
    y = torch.tensor([1, 0, 0, 1])
    assert expected_calibration_error(p, y, 5) == pytest.approx(0.2)
    fig = reliability_diagram([0.1, 0.9], [0, 1])
    assert list(fig.axes[0].lines[1].get_ydata()) == [1.0]
    plt.close(fig)


def test_split_disjoint_repeatable_and_metadata(envs):
    first = split_environments(envs, 0.2, 7)
    again = split_environments(envs, 0.2, 7)
    train, val = (first[k]["source_indices"] for k in ("train_e1", "val_e1"))
    assert not set(train.tolist()) & set(val.tolist())
    assert sorted(torch.cat([train, val]).tolist()) == list(range(40))
    assert torch.equal(val, again["val_e1"]["source_indices"])
    assert torch.equal(first["val_e1"]["patch"], envs["train_e1"]["patch"][val])
    assert first["test"] is envs["test"]


def test_group_counts():
    report = worst_group_acc(
        torch.tensor([0.1, 0.9, 0.1, 0.9]), torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1])
    )
    assert report["worst"] == 0
    assert sum(v for k, v in report.items() if k.startswith("n_")) == 4


def test_irm_penalty_and_gradient_activate():
    logits = [torch.tensor([0.8, -0.2], requires_grad=True)]
    labels = [torch.tensor([1.0, 0.0])]
    warm, _ = compute_objective(logits, labels, "irm", 1, 10, 2)
    active, info = compute_objective(logits, labels, "irm", 2, 10, 2)
    assert active > warm
    assert info["lam"] == 10
    warm_grad = torch.autograd.grad(warm, logits, retain_graph=True)[0]
    active_grad = torch.autograd.grad(active, logits)[0]
    assert not torch.allclose(warm_grad, active_grad)


def test_rex_has_no_warmup():
    logits = [torch.tensor([1.0]), torch.tensor([-1.0])]
    labels = [torch.ones(1), torch.ones(1)]
    obj, info = compute_objective(logits, labels, "rex", 1, 10, 190, 100)
    assert obj > info["mean_risk"]


@pytest.mark.parametrize("method", ["erm", "irm", "rex"])
def test_offline_pipeline_and_artifacts(envs, tmp_path, method):
    torch.set_num_threads(1)
    parts = split_environments(envs)
    training, validation, evaluation = experiment_loaders(parts, 8, seed=3)
    res = train_one_run(
        lambda: MLP(hidden=8),
        training,
        validation,
        method=method,
        steps=4,
        anneal_steps=2,
        penalty_weight=10,
        rex_beta=10,
        device="cpu",
        log_every=1,
        seed=3,
    )
    if method == "irm":
        assert res["history"][-1]["lam"] == 10
    summaries = save_run(
        res, evaluation, parts, "cpu", tmp_path, {"steps": 4, "method": method, "seed": 3}
    )
    assert set(summaries) == {"last", "trainval", "oracle"}
    assert 1 <= summaries["trainval"]["step"] <= 4
    saved = json.loads((tmp_path / "run.json").read_text())
    metrics = saved["evaluations"]["trainval"]["metrics"]["test"]
    assert 0 <= metrics["ece"] <= 1
    assert sum(g["n"] for g in metrics["patch_groups"].values()) == 40
    assert (tmp_path / "trainval.pt").is_file()
    for rec in res["history"]:
        assert rec["id_avg"] == pytest.approx((rec["acc_val_e1"] + rec["acc_val_e2"]) / 2)


def test_seed_and_evaluation_frequency_independence(envs):
    torch.set_num_threads(1)
    parts = split_environments(envs)
    states = []
    for interval in (1, 4):
        torch.rand(17)
        training, validation, _ = experiment_loaders(parts, 8, seed=3)
        res = train_one_run(
            lambda: MLP(hidden=8),
            training,
            validation,
            steps=4,
            device="cpu",
            log_every=interval,
            seed=3,
        )
        states.append(res["model"].state_dict())
    assert all(torch.equal(states[0][k], states[1][k]) for k in states[0])


def test_single_env_validation_average(envs):
    parts = split_environments({k: v for k, v in envs.items() if k != "train_e2"})
    training, validation, _ = experiment_loaders(parts, 8)
    res = train_one_run(
        lambda: MLP(hidden=8), training, validation, steps=1, device="cpu", log_every=1
    )
    assert res["best_id_acc"] == res["history"][0]["acc_val_e1"]


def test_training_only_selection_rejected(envs):
    training, _, evaluation = experiment_loaders(envs, 8)
    with pytest.raises(ValueError, match="held-out"):
        train_one_run(lambda: MLP(hidden=8), training, evaluation, steps=1, device="cpu")


def test_recoloring_detects_color_rule(envs):
    class ColorRule(torch.nn.Module):
        def forward(self, x):
            return x[:, 0].sum((1, 2)) - x[:, 1].sum((1, 2))

    _, _, evaluation = experiment_loaders(envs, 8)
    result = recoloring_sensitivity(ColorRule(), evaluation["test"], "cpu")
    assert result["prediction_flip_rate"] == 1
    assert result["n"] == 40


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_active_irm_cuda(envs):
    parts = split_environments(envs)
    training, validation, _ = experiment_loaders(parts, 8)
    res = train_one_run(
        lambda: MLP(hidden=8),
        training,
        validation,
        method="irm",
        steps=3,
        anneal_steps=2,
        penalty_weight=10,
        device="cuda",
        log_every=1,
    )
    assert res["history"][-1]["lam"] == 10
    assert all(torch.isfinite(p).all() for p in res["model"].parameters())


@pytest.mark.parametrize("script", ["run_sweep", "ablate", "make_diagnostics"])
def test_script_entrypoints(envs, tmp_path, monkeypatch, script):
    import importlib.util

    import yaml

    spec = importlib.util.spec_from_file_location(
        script, Path(__file__).resolve().parents[1] / "scripts" / f"{script}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = {
        "data": {
            "data_dir": "unused",
            "label_noise": 0.25,
            "seed": 0,
            "batch_size": 8,
            "val_fraction": 0.2,
            "split_seed": 0,
        },
        "model": "mlp",
        "train": {"steps": 3, "lr": 0.001, "anneal_steps": 2, "log_every": 1, "seeds": [0]},
        "methods": {"erm": {}, "irm": {"penalty_weight": 10}, "rex": {"rex_beta": 10}},
        "ablate": {"lambda_sweep": [], "rex_beta_sweep": [], "backbones": []},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "outputs"
    monkeypatch.setattr(sys, "argv", [script, "--config", str(config_path), "--out", str(out)])

    def build(**kw):
        keys = [s.name for s in kw["env_specs"]] if "env_specs" in kw else list(envs)
        return {k: envs[k] for k in keys}

    monkeypatch.setattr(module, "build_colored_mnist", build)
    monkeypatch.setattr(module, "make_model", lambda _: MLP(hidden=8))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    module.main()
    if script == "run_sweep":
        rows = json.loads((out / "results.json").read_text())
        assert len(rows) == 9
        assert {r["protocol"] for r in rows} == {"last", "trainval", "oracle"}
        assert (out / "results.csv").is_file()

        def no_retraining(*args, **kwargs):
            raise AssertionError("completed sweep must not retrain")

        monkeypatch.setattr(module, "train_one_run", no_retraining)
        monkeypatch.setattr(
            sys, "argv", [script, "--config", str(config_path), "--out", str(out), "--resume"]
        )
        module.main()
        assert json.loads((out / "results.json").read_text()) == rows
        # Three settings match the sweep; the fixed IRM baseline has a different penalty.
        cfg["ablate"] = {"lambda_sweep": [10], "rex_beta_sweep": [10], "backbones": ["mlp"]}
        config_path.write_text(yaml.safe_dump(cfg))
        ablate_spec = importlib.util.spec_from_file_location(
            "reuse_ablate", Path(__file__).resolve().parents[1] / "scripts" / "ablate.py"
        )
        ablate = importlib.util.module_from_spec(ablate_spec)
        ablate_spec.loader.exec_module(ablate)
        monkeypatch.setattr(ablate, "build_colored_mnist", build)
        monkeypatch.setattr(ablate, "make_model", lambda _: MLP(hidden=8))
        ablation_out = tmp_path / "ablation_reuse"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "ablate",
                "--config",
                str(config_path),
                "--out",
                str(ablation_out),
                "--reuse-sweep",
                str(out),
                "--resume",
            ],
        )
        ablate.main()
        reused_rows = json.loads((ablation_out / "ablate.json").read_text())
        assert len(reused_rows) == 9
        assert sum("reused_from" in row for row in reused_rows) == 3
        monkeypatch.setattr(ablate, "train_one_run", no_retraining)
        ablate.main()
    elif script == "ablate":
        rows = json.loads((out / "ablate.json").read_text())
        assert len(rows) == 5
        assert all(r["protocol"] == "trainval" for r in rows)
    else:
        data = json.loads((out / "diagnostics.json").read_text())
        assert set(data["recoloring"]) == {"erm", "irm", "rex"}
        assert (out / "reliability_rex.png").is_file()

        # Reusing the exact selected checkpoints must reproduce calibration without training.
        def no_retraining(*args, **kwargs):
            raise AssertionError("checkpoint diagnostics must not retrain")

        monkeypatch.setattr(module, "train_one_run", no_retraining)
        reused = tmp_path / "reused"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                script,
                "--config",
                str(config_path),
                "--out",
                str(reused),
                "--checkpoint-root",
                str(out / "runs"),
            ],
        )
        module.main()
        assert json.loads((reused / "diagnostics.json").read_text())["ece"] == data["ece"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_batched_fetch_preserves_sampling_and_training(envs, device):
    from torch.utils.data import DataLoader, TensorDataset

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    parts = split_environments(envs)
    fast_train, validation, _ = experiment_loaders(parts, 8, seed=3, storage_device=device)
    legacy_train = []
    for i, key in enumerate(sorted(k for k in parts if k.startswith("train_"))):
        env = parts[key]
        legacy_train.append(
            DataLoader(
                TensorDataset(env["x"], env["y"], env["color"]),
                batch_size=8,
                shuffle=True,
                generator=torch.Generator().manual_seed(3 + i),
            )
        )

    def factory():
        return MLP(hidden=8)

    legacy = train_one_run(
        factory, legacy_train, validation, steps=9, log_every=9, device=device, seed=3
    )
    fast = train_one_run(
        factory, fast_train, validation, steps=9, log_every=9, device=device, seed=3
    )
    assert all(
        torch.equal(v, fast["model"].state_dict()[k])
        for k, v in legacy["model"].state_dict().items()
    )

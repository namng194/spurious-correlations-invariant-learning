import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_smoke_pipeline_cpu():
    """Tiny end-to-end run on CPU (fast, no download cache assumption issues)."""
    import torch

    from cmnist.data import EnvSpec, build_colored_mnist, to_loader
    from cmnist.eval import summarize
    from cmnist.models import make_model
    from cmnist.train import train_one_run

    envs = build_colored_mnist(
        data_dir="data",
        label_noise=0.25,
        env_specs=[
            EnvSpec("train_e1", p_e=0.1, train=True, n=1024),
            EnvSpec("train_e2", p_e=0.2, train=True, n=1024),
            EnvSpec("test", p_e=0.9, train=False, n=1024),
        ],
        seed=0,
    )
    assert envs["train_e1"]["x"].shape[1:] == (3, 28, 28)
    loaders = {k: to_loader(v, batch_size=256) for k, v in envs.items()}
    for method in ("erm", "irm", "rex"):
        res = train_one_run(
            make_model("mlp"),
            [loaders["train_e1"], loaders["train_e2"]],
            loaders,
            method=method,
            steps=20,
            device="cpu",
            log_every=20,
            seed=0,
        )
        summ = summarize(res["model"], loaders, "cpu")
        assert 0.0 <= summ["test"]["acc"] <= 1.0
        assert summ["test"]["ece"] >= 0.0

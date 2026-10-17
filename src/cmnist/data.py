"""Colored-MNIST builder following Arjovsky et al. (2019), IRM.

Protocol:
  1. y_tilde = 0 if digit in 0-4 else 1
  2. y = flip(y_tilde) with prob `label_noise` (default 0.25)
  3. color id z = y XOR Bernoulli(p_e); red if z==0 else green.
     Image is RGB: R = img*(z==0), G = img*(z==1), B = 0.
  4. Optional 2nd spurious: 3x3 white patch top-left correlated with y
     with prob (1 - p_patch_e) (i.e. patch=y unless flipped w.p. p_patch_e).

Envs default: train e1 p=0.1, e2 p=0.2, test p=0.9.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets


@dataclass
class EnvSpec:
    name: str
    p_e: float
    n: int | None = None  # None = full split
    train: bool = True
    p_patch_e: float | None = None  # None = no patch spurious


def _load_raw(data_dir: str, train: bool, dataset: str = "mnist"):
    if dataset == "mnist":
        ds = datasets.MNIST(data_dir, train=train, download=True)
    elif dataset == "fashion":
        ds = datasets.FashionMNIST(data_dir, train=train, download=True)
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    imgs = ds.data.numpy().astype(np.float32) / 255.0  # (N,28,28)
    digits = ds.targets.numpy()
    return imgs, digits


def _load_mnist_raw(data_dir: str, train: bool):
    return _load_raw(data_dir, train, "mnist")


def colorize(imgs: np.ndarray, z: np.ndarray) -> np.ndarray:
    """imgs: (N,28,28) float32 in [0,1]; z: (N,) binary color id. -> (N,3,28,28)."""
    n = imgs.shape[0]
    out = np.zeros((n, 3, 28, 28), dtype=np.float32)
    red = z == 0
    green = ~red
    out[red, 0] = imgs[red]
    out[green, 1] = imgs[green]
    return out


def build_env(
    imgs: np.ndarray,
    digits: np.ndarray,
    spec: EnvSpec,
    label_noise: float,
    rng: np.random.Generator,
) -> dict:
    y_tilde = (digits >= 5).astype(np.int64)
    flip_y = rng.random(len(imgs)) < label_noise
    y = np.where(flip_y, 1 - y_tilde, y_tilde).astype(np.int64)
    flip_c = rng.random(len(imgs)) < spec.p_e
    z = np.where(flip_c, 1 - y, y).astype(np.int64)  # color id
    x = colorize(imgs, z)
    if spec.p_patch_e is not None:
        flip_p = rng.random(len(imgs)) < spec.p_patch_e
        has_patch = np.where(flip_p, 1 - y, y).astype(bool)
        x[has_patch, :, 0:3, 0:3] = 1.0  # white 3x3 patch, all channels
    else:
        has_patch = np.zeros(len(imgs), dtype=bool)
    if spec.n is not None:
        idx = rng.choice(len(imgs), size=min(spec.n, len(imgs)), replace=False)
        x, y, z = x[idx], y[idx], z[idx]
        digits, y_tilde = digits[idx], y_tilde[idx]
        has_patch = has_patch[idx]
    return {
        "x": torch.from_numpy(x),
        "y": torch.from_numpy(y).float(),
        "color": torch.from_numpy(z).long(),
        "digit": torch.from_numpy(digits).long(),
        "y_tilde": torch.from_numpy(y_tilde).long(),
        "patch": torch.from_numpy(has_patch),
        "name": spec.name,
        "p_e": spec.p_e,
    }


def build_colored_mnist(
    data_dir: str = "data",
    label_noise: float = 0.25,
    env_specs: list[EnvSpec] | None = None,
    seed: int = 0,
    dataset: str = "mnist",
) -> dict[str, dict]:
    """Returns {env_name: env_dict}. Train envs split MNIST-train in halves (disjoint)."""
    rng = np.random.default_rng(seed)
    if env_specs is None:
        env_specs = [
            EnvSpec("train_e1", p_e=0.1, train=True),
            EnvSpec("train_e2", p_e=0.2, train=True),
            EnvSpec("test", p_e=0.9, train=False),
        ]
    train_imgs, train_digits = _load_raw(data_dir, train=True, dataset=dataset)
    test_imgs, test_digits = _load_raw(data_dir, train=False, dataset=dataset)
    # disjoint halves for the two train envs (Arjovsky protocol uses split in thirds;
    # halves give more samples per env with same correlation structure)
    perm = rng.permutation(len(train_imgs))
    half = len(train_imgs) // 2
    pools = {
        "train": (
            train_imgs[perm[:half]],
            train_digits[perm[:half]],
            train_imgs[perm[half:]],
            train_digits[perm[half:]],
        )
    }
    envs: dict[str, dict] = {}
    train_pool_idx = 0
    for spec in env_specs:
        if spec.train:
            pi, di = pools["train"][train_pool_idx * 2], pools["train"][train_pool_idx * 2 + 1]
            train_pool_idx = (train_pool_idx + 1) % 2
            envs[spec.name] = build_env(pi, di, spec, label_noise, rng)
        else:
            envs[spec.name] = build_env(test_imgs, test_digits, spec, label_noise, rng)
    return envs


class BatchTensorDataset(TensorDataset):
    """Fetch whole batches instead of thousands of Python sample lookups."""

    def __getitems__(self, indices):
        index = torch.as_tensor(indices, device=self.tensors[0].device)
        return tuple(t.index_select(0, index) for t in self.tensors)


def _already_batched(batch):
    return batch


def to_loader(
    env: dict, batch_size: int = 2048, shuffle: bool = True, storage_device: str = "cpu"
) -> DataLoader:
    ds = BatchTensorDataset(*(env[k].to(storage_device) for k in ("x", "y", "color")))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(0),
        collate_fn=_already_batched,
    )


def save_envs(envs: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for name, env in envs.items():
        torch.save(
            {
                "x": env["x"],
                "y": env["y"],
                "color": env["color"],
                "digit": env["digit"],
                "p_e": env["p_e"],
            },
            os.path.join(out_dir, f"{name}.pt"),
        )


def split_environments(envs: dict, val_fraction: float = 0.2, split_seed: int = 0):
    """Split training environments once; retain indices for auditing and replay.

    The test environment is never split or used to construct validation sets.
    Tensor metadata (including patch membership) follows the same partition.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one")
    result = {}
    generator = torch.Generator().manual_seed(split_seed)
    for name, env in sorted(envs.items()):
        if not name.startswith("train_"):
            result[name] = env
            continue
        n = len(env["y"])
        if n < 2:
            raise ValueError("each training environment needs at least two examples")
        order = torch.randperm(n, generator=generator)
        n_val = min(n - 1, max(1, round(n * val_fraction)))
        for key, indices in (
            (name, order[n_val:]),
            (name.replace("train_", "val_", 1), order[:n_val]),
        ):
            part = {
                k: v[indices] if torch.is_tensor(v) and v.ndim and len(v) == n else v
                for k, v in env.items()
            }
            part.update(name=key, source_indices=indices)
            result[key] = part
    return result


def experiment_loaders(envs: dict, batch_size: int, seed: int = 0, storage_device: str = "cpu"):
    """Independent training RNGs keep evaluation frequency out of batch sampling."""
    evaluation = {
        k: to_loader(v, batch_size, shuffle=False, storage_device=storage_device)
        for k, v in envs.items()
    }
    training = []
    for i, name in enumerate(sorted(k for k in envs if k.startswith("train_"))):
        ds = evaluation[name].dataset
        training.append(
            DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(seed + i),
                collate_fn=_already_batched,
            )
        )
    validation = {k: v for k, v in evaluation.items() if k.startswith("val_") or k == "test"}
    return training, validation, evaluation

"""Backbones: MLP (default, IRM-friendly) + small CNN (overparam demo)."""
from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int = 3 * 28 * 28, hidden: int = 256, n_layers: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        self.phi = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 1)
        self._feat_dim = hidden

    def forward(self, x: torch.Tensor, return_feat: bool = False):
        h = self.phi(x.flatten(1))
        logit = self.head(h).squeeze(1)
        return (logit, h) if return_feat else logit


class SmallCNN(nn.Module):
    """2-conv CNN, deliberately higher capacity than MLP for the overparam demo."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.phi = nn.Sequential(nn.Linear(64 * 7 * 7, 256), nn.ReLU())
        self.head = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor, return_feat: bool = False):
        h = self.phi(self.conv(x).flatten(1))
        logit = self.head(h).squeeze(1)
        return (logit, h) if return_feat else logit


def make_model(name: str) -> nn.Module:
    if name == "mlp":
        return MLP()
    if name == "cnn":
        return SmallCNN()
    raise ValueError(f"unknown model {name!r}")

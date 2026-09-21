"""Losses: ERM / IRMv1 / V-REx / GroupDRO(max-env).

All methods share the same per-env BCE risk. The IRMv1 penalty uses the
dummy-classifier trick: R^e(w * Phi) with w=1.0, penalty = ||grad_w R^e||^2.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def env_risk(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, y)


def irmv1_penalty(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
    loss = F.binary_cross_entropy_with_logits(logits * scale, y)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return grad.pow(2)


def compute_objective(
    logits_per_env: list[torch.Tensor],
    y_per_env: list[torch.Tensor],
    method: str,
    step: int,
    penalty_weight: float,
    anneal_steps: int,
    rex_beta: float = 1e3,
) -> tuple[torch.Tensor, dict]:
    risks = torch.stack([env_risk(l, y) for l, y in zip(logits_per_env, y_per_env)])
    mean_risk = risks.mean()
    lam = 0.0 if step < anneal_steps else penalty_weight
    info = {"mean_risk": mean_risk.detach(), "lam": lam}
    if method == "erm":
        return mean_risk, info
    if method == "groupdro":
        # lite version: minimize worst-env risk (Sagawa et al. 2019, static form)
        return risks.max(), {**info, "worst_risk": risks.max().detach()}
    if method == "rex":
        obj = mean_risk + rex_beta * risks.var(unbiased=False)
        return obj, {**info, "risk_var": risks.var(unbiased=False).detach()}
    if method == "irm":
        pens = torch.stack(
            [irmv1_penalty(l, y) for l, y in zip(logits_per_env, y_per_env)]
        )
        obj = mean_risk + lam * pens.sum()
        return obj, {**info, "penalty": pens.sum().detach()}
    raise ValueError(f"unknown method {method!r}")

"""Straight-through (piecewise) Hungarian scoring for Phase75E training."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def hungarian_assignment(similarity: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Choose exact assignment indices from detached CPU similarities.

    The indices are intentionally non-differentiable.  Callers gather the
    selected entries from the original torch matrix, preserving gradients
    through the selected similarities (piecewise/straight-through scoring).
    """
    if similarity.ndim != 2 or similarity.numel() == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    arr = similarity.detach().float().cpu().numpy()
    if arr.shape[0] <= arr.shape[1]:
        ri, ci = linear_sum_assignment(-arr)
        return ri.astype(np.int64), ci.astype(np.int64)
    ci, ri = linear_sum_assignment(-arr.T)
    return ri.astype(np.int64), ci.astype(np.int64)


def pairwise_torch_score(query: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    """Mean matched cosine for normalized ``[T,768]`` sequences."""
    if query.ndim != 2 or candidate.ndim != 2 or query.shape[-1] != candidate.shape[-1]:
        raise ValueError(f"expected [T,D] tensors, got {tuple(query.shape)} and {tuple(candidate.shape)}")
    q = torch.nn.functional.normalize(query.float(), dim=-1)
    c = torch.nn.functional.normalize(candidate.float(), dim=-1)
    sim = q @ c.transpose(0, 1)
    ri, ci = hungarian_assignment(sim)
    if len(ri) == 0:
        return sim.new_tensor(float("-inf"))
    rows = torch.as_tensor(ri, device=sim.device, dtype=torch.long)
    cols = torch.as_tensor(ci, device=sim.device, dtype=torch.long)
    return sim[rows, cols].mean()


def raw_mean_score(query: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    """The frozen raw comparator used by the registered protocol."""
    q = torch.nn.functional.normalize(query.float(), dim=-1).mean(dim=0)
    c = torch.nn.functional.normalize(candidate.float(), dim=-1).mean(dim=0)
    return torch.nn.functional.cosine_similarity(q.unsqueeze(0), c.unsqueeze(0)).squeeze(0)


def adapter_drift(adapted: torch.Tensor, raw: torch.Tensor) -> dict[str, float]:
    """Return cosine and relative residual diagnostics for report/checkpoint logs."""
    a = torch.nn.functional.normalize(adapted.float(), dim=-1)
    r = torch.nn.functional.normalize(raw.float(), dim=-1)
    cos = (a * r).sum(dim=-1)
    diff = (adapted.float() - raw.float()).norm(dim=-1)
    denom = raw.float().norm(dim=-1).clamp_min(1e-8)
    rel = diff / denom
    q = torch.quantile(cos, torch.tensor([0.05, 0.50, 0.95], device=cos.device))
    return {
        "mean_cosine": float(cos.mean().detach().cpu()),
        "p05_cosine": float(q[0].detach().cpu()),
        "p50_cosine": float(q[1].detach().cpu()),
        "p95_cosine": float(q[2].detach().cpu()),
        "mean_delta_norm_over_raw": float(rel.mean().detach().cpu()),
    }


def finite_tensor_sequence(items: Iterable[np.ndarray], device: torch.device) -> list[torch.Tensor]:
    """Convert a small sequence list without permitting metadata tensors."""
    out = []
    for item in items:
        arr = np.asarray(item, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 768 or not np.isfinite(arr).all():
            raise ValueError(f"invalid feature sequence shape/content: {arr.shape}")
        out.append(torch.as_tensor(arr, device=device))
    return out

#!/usr/bin/env python3
"""Dependency-light Phase76A contract smoke for the pinned OVTR environment."""
from __future__ import annotations

import numpy as np
import torch

from src.iclr27_phase76a.correspondence import hungarian_match, pair_relation_features, relation_summary
from src.iclr27_phase76a.raw_anchor import raw_mean_cosine
from src.iclr27_phase76a.relation_model import AnchoredRelationReranker


def main() -> None:
    q = np.zeros((2, 768), dtype=np.float32); c = np.zeros((3, 768), dtype=np.float32)
    q[0, 0] = 1.0; q[1, 1] = 1.0; c[0, 0] = 1.0; c[1, 1] = 1.0; c[2, 2] = 1.0
    m = hungarian_match(q, c); tokens = pair_relation_features(q, c, m); summary = relation_summary(q, c, m, raw_mean_cosine(q, c))
    model = AnchoredRelationReranker().eval()
    with torch.no_grad(): out = model(torch.as_tensor(tokens), torch.as_tensor(summary), torch.tensor(float(summary[0])))
    assert tokens.shape == (2, 1536) and summary.shape == (13,)
    assert torch.isfinite(out["final"]).item() and abs(float(out["delta"])) <= 1e-7
    assert abs(float(out["confidence"]) - 0.5) <= 1e-7 and abs(float(out["final"]) - float(summary[0])) <= 1e-7
    print({"pair_tokens": list(tokens.shape), "summary": list(summary.shape), "delta": float(out["delta"]), "confidence": float(out["confidence"]), "final": float(out["final"]), "status": "PASS"})


if __name__ == "__main__": main()


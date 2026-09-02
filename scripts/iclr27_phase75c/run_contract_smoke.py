#!/usr/bin/env python3
"""Bounded contract smoke for the frozen Phase75C representation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.iclr27_phase75c.grounded_correspondence import GroundedCorrespondence

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    rng = np.random.default_rng(7503)
    x = rng.normal(size=(5, 768)).astype(np.float32)
    model = GroundedCorrespondence()
    z1 = model.encode_prefix(x, 1); z5 = model.encode_prefix(x, 5)
    assert z1.shape == (768,) and z5.shape == (768,)
    assert np.isfinite(z1).all() and np.isfinite(z5).all()
    assert np.isclose(np.linalg.norm(z5), 1.0, atol=1e-5)
    # Causality: changing an unseen suffix cannot alter prefix-1 output.
    y = x.copy(); y[1:] = rng.normal(size=(4, 768)).astype(np.float32)
    assert np.array_equal(z1, model.encode_prefix(y, 1))
    meta = model.metadata()
    forbidden = {"category", "category_text", "physical_id", "semantic_id", "future_frame", "future_track", "held_gt"}
    assert not forbidden.intersection(meta["input"])
    payload = {"status": "PASS", "shape": list(z5.shape), "causal_prefix_immutable": True, "metadata": meta}
    path = ROOT / "outputs/iclr27_phase75c/audit/contract_smoke.json"; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (ROOT / "outputs/iclr27_phase75c/completion/contract_smoke.done").parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs/iclr27_phase75c/completion/contract_smoke.done").write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()


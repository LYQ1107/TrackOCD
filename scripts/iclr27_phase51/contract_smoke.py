#!/usr/bin/env python3
"""Bounded Phase51 graph contract smoke (no TRAIN labels in inputs)."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from src.iclr27_phase51.unified_model import UnifiedTrackOCD, metadata

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase51/audit"


def main() -> None:
    torch.manual_seed(510051)
    model = UnifiedTrackOCD().eval()
    q = F.normalize(torch.randn(2, 4, 768), dim=-1)
    g = torch.randn(2, 4, 15)
    mask = torch.ones(2, 4, dtype=torch.bool)
    no = model.encode_sequence(q, g, mask, support=None)
    # Explicit invalid-support mask: entries exist but none is valid.
    support = F.normalize(torch.randn(2, 3, 768), dim=-1)
    invalid = model.encode_sequence(q, g, mask, support=support, support_mask=torch.zeros(2, 3, dtype=torch.bool))
    valid = model.encode_sequence(q, g, mask, support=support, support_mask=torch.ones(2, 3, dtype=torch.bool))
    expected = F.normalize(q[:, -1], dim=-1)
    assoc = model.association(q[:, 0], g[:, 0], q[:, 1], g[:, 1])
    proposal = model.proposal(q[:, 0], g[:, 0])
    result = {
        "phase": 52, "finite_no_support": bool(torch.isfinite(no["semantic"]).all()),
        "finite_invalid_support": bool(torch.isfinite(invalid["semantic"]).all()),
        "finite_valid_support": bool(torch.isfinite(valid["semantic"]).all()),
        "no_support_exact_raw": bool(torch.equal(no["semantic"], expected)),
        "invalid_support_exact_raw": bool(torch.equal(invalid["semantic"], expected)),
        "valid_shape": list(valid["semantic"].shape), "valid_norm": valid["semantic"].norm(dim=-1).tolist(),
        "association_shape": list(assoc.shape),
        "proposal_shapes": {k: list(v.shape) for k, v in proposal.items() if k != "proposal_hidden"},
        "actions": ["COMMIT", "DEFER", "RESET_REJECT"],
        "forbidden_inputs": metadata(model)["forbidden_inputs"],
        "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels", "future rows/tracks", "held GT", "category/text/ID features"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "contract_smoke.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ROOT / "outputs/iclr27_phase51/completion/contract_smoke.done").write_text(json.dumps({"phase": 52, "status": "PASS"}) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

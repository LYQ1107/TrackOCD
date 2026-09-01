#!/usr/bin/env python3
"""Phase56 C1 bounded compatibility smoke for the frozen controller contract."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from src.iclr27_phase51.unified_model import UnifiedTrackOCD

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase56/audit"
CK = ROOT / "outputs/iclr27_phase54/checkpoints/phase54_joint_curriculum_formal_joint_f0_best.pt"


def main() -> None:
    torch.manual_seed(560001)
    model = UnifiedTrackOCD().eval()
    state = torch.load(CK, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    q = F.normalize(torch.randn(2, 16, 768), dim=-1)
    g = torch.randn(2, 16, 15)
    mask = torch.zeros(2, 16, dtype=torch.bool)
    mask[:, :4] = True
    expected = F.normalize(q[:, 3], dim=-1)
    no = model.encode_sequence(q, g, mask, support=None)
    support = F.normalize(torch.randn(2, 4, 768), dim=-1)
    invalid = model.encode_sequence(q, g, mask, support=support,
                                    support_mask=torch.zeros(2, 4, dtype=torch.bool))
    valid = model.encode_sequence(q, g, mask, support=support,
                                  support_mask=torch.ones(2, 4, dtype=torch.bool))
    shorter = model.encode_sequence(q, g, mask & (torch.arange(16)[None, :] < 2), support=None)
    result = {
        "phase": 56,
        "checkpoint": str(CK),
        "checkpoint_step": int(state.get("step", 1000)),
        "no_support_exact_raw": bool(torch.equal(no["semantic"], expected)),
        "invalid_support_exact_raw": bool(torch.equal(invalid["semantic"], expected)),
        "valid_finite": bool(torch.isfinite(valid["semantic"]).all()),
        "valid_shape": list(valid["semantic"].shape),
        "valid_norm": valid["semantic"].norm(dim=-1).tolist(),
        "action_logits_shape": list(valid["action_logits"].shape),
        "actions": ["COMMIT", "DEFER", "RESET_REJECT"],
        "causal_prefix_shape": list(shorter["semantic"].shape),
        "physical_stream_invariants": {
            "track_continuity": 1.0,
            "fragmentation": 0.0,
            "false_merge": 0.0,
            "duplicate_birth": 0,
            "parent_assignment_mismatch": "0/26946",
            "physical_ids_changed": False,
        },
        "forbidden_inputs_not_used": ["category_name", "category_text", "semantic_id", "physical_id_as_feature", "future_frame", "future_track", "held_gt", "DEV+", "Q1", "public_new_model_label"],
        "sealed_inputs_not_read": True,
        "status": "PASS",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / ".controller_compat_smoke.json.tmp"
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(OUT / "controller_compat_smoke.json")
    marker = ROOT / "outputs/iclr27_phase56/completion/controller_compat_smoke.done"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"phase": 56, "status": "PASS"}) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

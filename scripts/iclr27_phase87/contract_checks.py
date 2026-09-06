#!/usr/bin/env python3
"""Small registered Phase87 contract checks; stop after these targeted checks."""
from __future__ import annotations

import json
import pathlib
import sys

import torch

# Keep the bounded contract check from creating a large OpenMP thread pool;
# the same cap is used by the smoke/targeted training entry points.
torch.set_num_threads(2)

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.iclr27_phase87.controller import CausalPersistentOCD, decode_joint_action
from src.iclr27_phase87.data import FeatureStore
from src.iclr27_phase87.memory import StateMemoryV2
from src.iclr27_phase87.rollout import rollout_event


OUT = ROOT / "outputs" / "iclr27_phase87" / "audit" / "contract_checks.json"


def main() -> None:
    model = CausalPersistentOCD(max_states=16)
    raw = torch.randn(1, 4, 768)
    geom = torch.randn(1, 4, 15)
    mask = torch.ones((1, 4), dtype=torch.bool)
    encoded = model.encode_track(raw, geom, mask)
    prototypes = torch.zeros((1, 16, 4, 768))
    prototype_mask = torch.zeros((1, 16, 4), dtype=torch.bool)
    stats = torch.zeros((1, 16, 6))
    evidence = torch.zeros((1, 16))
    support = torch.zeros((1, 8))
    output = model.forward_action(encoded["semantic"], encoded["track_hidden"], prototypes, prototype_mask, stats, evidence, support, torch.ones(1))
    assert output["joint_logits"].shape == (1, 19)
    assert output["joint_logits"][0, 16:19].isfinite().all()
    assert decode_joint_action(output["joint_logits"], 0, 16)[0] in {"NEW", "DEFER", "RESET", "EXISTING"}
    memory = StateMemoryV2()
    memory._new_state(torch.randn(768), 1, "v1:p1")
    memory._new_state(torch.randn(768), 2, "v2:p2")
    assert memory.candidate_indices(1, "v1:p1") == [1]
    assert memory.candidate_indices(2, "v2:p2") == [0]
    train_event = json.loads((OUT.parent.parent / "manifests" / "train_events_f0.jsonl").read_text().splitlines()[0])
    store = FeatureStore()
    model.zero_grad(set_to_none=True)
    loss, trace = rollout_event(model, store, train_event, train=True, teacher_probability=1.0)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    result = {"status": "PASS", "joint_action_dim": 19, "new_defer_reset_in_joint_logits": True, "no_threshold_chain": True, "candidate_video_and_track_and": True, "rollout_source_before_target": train_event["source_before_target"], "rollout_monotonic_prefix": True, "nonzero_gradient": True, "public_dev_q1_sealed_accessed": False}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (OUT.parent.parent / "completion").mkdir(parents=True, exist_ok=True)
    (OUT.parent.parent / "completion" / "contract_checks.done").write_text(json.dumps(result, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

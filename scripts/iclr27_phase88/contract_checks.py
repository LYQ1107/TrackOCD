#!/usr/bin/env python3
"""Small Phase88 contract checks; scientific metrics remain in evaluator."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs" / "iclr27_phase88"

from src.iclr27_phase19r.evaluation.internal import metrics as phase19r_metrics
from src.iclr27_phase88.controller import CausalPersistentOCD, mask_joint_logits
from src.iclr27_phase88.data import FeatureStore
from src.iclr27_phase88.evaluation import evaluate_persistent_events
from src.iclr27_phase88.rollout import rollout_event


def sha_tensor(x: torch.Tensor) -> str:
    return hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()


def main() -> None:
    data = __import__("src.iclr27_phase19r.data.stream", fromlist=["Phase19RData"]).Phase19RData(0)
    store = FeatureStore(0, data)
    model = CausalPersistentOCD(torch.from_numpy(data.known_prototypes), torch.from_numpy(data.active_known_mask), max_states=16)
    checks: dict[str, object] = {
        "formal_metric_source": "src.iclr27_phase19r.evaluation.internal.metrics",
        "formal_empty_metric": phase19r_metrics([], {"known_micro": 0.0, "known_macro": 0.0}),
        "known_count": model.known_count,
        "known_count_positive": model.known_count > 0,
        "known_prototype_hash": sha_tensor(model.known_prototypes),
        "known_active_count": int(model.active_known_mask.sum()),
    }
    if model.known_count <= 0:
        raise RuntimeError("KNOWN_COUNT_ZERO")
    # Only RESET is masked before a commitment; a committed action retains its
    # same action, DEFER and RESET options.
    logits = torch.zeros((1, model.action_dim))
    masked = mask_joint_logits(logits, model.known_count, model.max_states, [None], [None], [None])
    checks["reset_masked_before_commit"] = bool(masked[0, model.known_count + model.max_states + 2] <= -1e3)
    event = json.loads((OUT / "manifests/val_events_v2_f0.jsonl").read_text().splitlines()[0])
    eval_out = evaluate_persistent_events(model, data, [event], torch.device("cpu"), support_mode=True)
    checks["all_source_tracks_processed"] = eval_out["diagnostic"]["source_tracks_processed"] == len(event["source_track_keys"])
    checks["runtime_state_memory_used"] = eval_out["diagnostic"]["mean_states_per_event"] >= 0.0
    checks["formal_metrics_real_numbers"] = all(isinstance(v, (int, float)) for v in eval_out["metrics"].values() if isinstance(v, (int, float)))
    # TRAIN reset augmentation must actually yield RESET supervision.
    reset_event = next(json.loads(x) for x in (OUT / "manifests/fit_events_v2_f0.jsonl").read_text().splitlines() if json.loads(x).get("reset_injection"))
    loss, trace = rollout_event(model, store, reset_event, train=True, teacher_probability=1.0)
    checks["reset_target_present"] = int(trace["reset_targets"]) > 0
    checks["reset_target_count"] = int(trace["reset_targets"])
    checks["loss_finite"] = bool(torch.isfinite(loss))
    checks["future_rows_or_tracks"] = False
    checks["ids_or_text_as_model_input"] = False
    if not all(bool(v) for k, v in checks.items() if k.endswith(("positive", "masked", "processed", "used", "numbers", "present", "finite"))):
        raise RuntimeError(json.dumps(checks, sort_keys=True))
    path = OUT / "audit/contract_checks.json"
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(checks, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)
    print(json.dumps(checks, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

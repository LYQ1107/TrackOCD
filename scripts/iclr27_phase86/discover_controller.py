#!/usr/bin/env python3
"""Discover and freeze the historically formal Phase19R OCD controller."""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "iclr27_phase86"


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_blob(path: pathlib.Path) -> str:
    return subprocess.run(["git", "hash-object", str(path)], cwd=ROOT, text=True,
                          capture_output=True, check=True).stdout.strip()


def main() -> None:
    controller = ROOT / "src/iclr27_phase19r/models/controller.py"
    state = ROOT / "src/iclr27_phase19r/runtime/state.py"
    runner = ROOT / "src/iclr27_phase19r/runtime/runner.py"
    evaluator = ROOT / "src/iclr27_phase19r/evaluation/internal.py"
    trainer = ROOT / "src/iclr27_phase19r/training/train_controller.py"
    config = ROOT / "configs/iclr27_phase19r/preregistered_main.json"
    ck = []
    for fold in range(4):
        p = ROOT / "outputs/iclr27_phase19r/checkpoints" / f"fold{fold}_best_internal.pt"
        ck.append({"fold": fold, "path": str(p), "sha256": sha(p), "bytes": p.stat().st_size})
    candidates = [
        {
            "path": str(controller),
            "commit": "working-tree blob at Phase86 start; source history is Phase19R",
            "code_sha256": sha(controller), "git_blob": git_blob(controller),
            "artifact": [x["path"] for x in ck],
            "config": str(config), "threshold": {"tau_known": 0.20, "tau_assign": 0.52, "tau_ready": 0.45},
            "memory_policy": {"class": "StateMemory", "max_states": 16, "max_anchors": 8, "sid_start": 100000,
                              "update": "quality >= .35 and confidence >= .55; causal EMA"},
            "input_contract": ["normalized 768-D raw visual vector", "causal geometry/quality", "known mask", "causal StateMemory candidate bundle"],
            "output_contract": ["KNOWN", "EXISTING", "NEW", "DEFER", "causal state trace"],
            "metric_evaluator": str(evaluator), "historical_phase": "Phase19R",
            "historical_status": "formally frozen and used for legal persistent Commit-CT replay; later Phase46 C2 comparator",
            "physical_semantic_separation": True, "category_text_in_forward": False,
        },
        {
            "path": "src/iclr27_phase56/unified_model.py",
            "commit": "Phase56 local candidate",
            "code_sha256": sha(ROOT / "src/iclr27_phase51/unified_model.py") if (ROOT / "src/iclr27_phase51/unified_model.py").exists() else None,
            "artifact": [], "config": "outputs/iclr27_phase56/final_decision.json",
            "threshold": "Phase56 candidate thresholds; not selected for Phase86",
            "memory_policy": "new unified semantic controller",
            "input_contract": ["Phase56 unified feature-row path"],
            "output_contract": ["COMMIT", "DEFER", "RESET_REJECT"],
            "metric_evaluator": "scripts/iclr27_phase56/evaluate_end_to_end.py",
            "historical_phase": "Phase56", "historical_status": "Gate C56 FAIL; retained as negative evidence, not Phase86 frozen controller",
            "physical_semantic_separation": True, "category_text_in_forward": False,
        },
    ]
    discovery = {
        "schema_version": "trackocd.phase86.controller_discovery.v1",
        "phase": 86,
        "selection_rule": "last historically formal frozen controller used for legal Commit-CT, not selected by Phase86 events",
        "selected": "Phase19R RC-MS-OCD",
        "selected_paths": {"controller": str(controller), "state_memory": str(state), "runner": str(runner),
                           "evaluator": str(evaluator), "trainer": str(trainer), "config": str(config)},
        "selected_hashes": {p.name: sha(p) for p in (controller, state, runner, evaluator, trainer, config)},
        "candidate_checkpoints": ck,
        "candidates": candidates,
        "sealed_or_public_accessed": False,
    }
    out = OUT / "audit" / "controller_discovery.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(discovery, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(out)
    manifest = {
        "schema_version": "trackocd.phase86.frozen_controller_manifest.v1",
        "phase": 86, "controller_family": "Phase19R RC-MS-OCD",
        "controller_code_commit": "Phase19R source in current main; byte-identical source hash recorded",
        "controller_code_sha256": sha(controller), "state_memory_sha256": sha(state),
        "controller_checkpoint_sha256": {str(x["fold"]): x["sha256"] for x in ck},
        "thresholds": {"tau_known": 0.20, "tau_assign": 0.52, "tau_ready": 0.45, "known_scale": 12.0,
                       "birth_bias_from_checkpoint": True, "defer_bias_from_checkpoint": True},
        "state_memory_parameters": {"max_states": 16, "max_anchors": 8, "sid_start": 100000,
                                     "quality_gate": 0.35, "confidence_gate": 0.55},
        "commit_defer_reset_rules": "decode action logits through Phase19R runner; no Phase86 modification",
        "prefix_policy": [1, 2, 4, 8, 16],
        "metric_evaluator": str(evaluator),
        "positive_manifest": str(ROOT / "outputs/iclr27_phase19r/manifests/held_known_positive_events.jsonl"),
        "negative_manifest": str(ROOT / "outputs/iclr27_phase19r/manifests/held_known_negative_events.jsonl"),
        "positive_manifest_sha256": sha(ROOT / "outputs/iclr27_phase19r/manifests/held_known_positive_events.jsonl"),
        "negative_manifest_sha256": sha(ROOT / "outputs/iclr27_phase19r/manifests/held_known_negative_events.jsonl"),
        "selection_basis": "chronology and formal-use history only; Phase86 diagnostic results cannot alter this manifest",
        "weights_frozen": True, "thresholds_frozen": True, "memory_frozen": True,
        "public_dev_q1_sealed_accessed": False,
    }
    m = OUT / "manifests" / "frozen_controller_manifest.json"
    m.parent.mkdir(parents=True, exist_ok=True)
    tmp = m.with_name(m.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(m)
    print(json.dumps({"selected": discovery["selected"], "checkpoints": len(ck), "controller_sha256": sha(controller)}, indent=2))


if __name__ == "__main__":
    main()

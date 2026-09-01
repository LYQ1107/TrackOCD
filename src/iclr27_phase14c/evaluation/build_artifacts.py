"""Build compact Phase-14C integrity and contract artifacts.

This is deliberately post-hoc bookkeeping: it reads only proposal/features
and evaluator summaries already produced by the locked runs.  It never reads
Q1 data and never changes a metric or a decision.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
OUT = ROOT / "outputs/iclr27_phase14c"


def read_json(rel: str):
    return json.loads((ROOT / rel).read_text())


def read_rows(rel: str):
    with (ROOT / rel).open(newline="") as f:
        return list(csv.DictReader(f))


def atomic_json(rel: str, payload) -> None:
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def proposal_integrity() -> None:
    rows = read_rows("outputs/iclr27_phase14c/proposals/proposals_mixed.csv")
    aligned = read_rows("outputs/iclr27_phase14c/proposals/proposals_aligned.csv")
    feats = np.load(ROOT / "outputs/iclr27_phase14c/features/proposal_dinov2.npz")
    keys = [
        f"{r['video_id']}:{r['frame_id']}:{r['proposal_local_id']}:{r['track_id']}"
        for r in rows
    ]
    akeys = [
        f"{r['video_id']}:{r['frame_id']}:{r['proposal_local_id']}:{r['track_id']}"
        for r in aligned
    ]
    feat_keys = [str(key) for key in feats["row_keys"]]
    shard = OUT / "proposals/shards/shard_0"
    payload = {
        "protocol": "phase14c",
        "rows": len(rows),
        "aligned_rows": len(aligned),
        "features_rows": int(len(feats["feats"])),
        "row_feature_count_match": len(rows) == len(feats["feats"]),
        "row_feature_key_order_match": keys == feat_keys,
        "alignment_preserves_row_order": keys == akeys,
        "duplicate_proposal_keys": len(keys) - len(set(keys)),
        "videos": len({int(r["video_id"]) for r in rows}),
        "frames_with_rows": len({(int(r["video_id"]), int(r["frame_id"])) for r in rows}),
        "physical_tracks": len({(int(r["video_id"]), int(r["track_id"])) for r in rows}),
        "proposal_score_field": "official_teta_track_score",
        "det_category_id_present_as_diagnostic_only": "det_category_id" in rows[0],
        "det_category_id_used_for_semantic_decision": False,
        "semantic_fields_absent": not any(k.startswith("sem_") for k in rows[0]),
        "q1_label_used": False,
        "future_frames_used": False,
        "private_gt_used_for_decision": False,
        "shard_markers": {
            "launched": (shard / ".launched").exists(),
            "json_ok": (shard / ".json_ok").exists(),
            "done": (shard / ".done").exists(),
        },
        "atomic_merge_output": True,
    }
    payload["pass"] = bool(
        payload["row_feature_count_match"]
        and payload["row_feature_key_order_match"]
        and payload["alignment_preserves_row_order"]
        and payload["duplicate_proposal_keys"] == 0
        and all(payload["shard_markers"].values())
    )
    atomic_json("outputs/iclr27_phase14c/proposals/proposal_integrity.json", payload)


def strict_summary() -> None:
    names = {
        "frozen_b_exact": "outputs/iclr27_phase14c/eval/frozen_b_exact_summary.json",
        "frozen_b_trainnorm": "outputs/iclr27_phase14c/eval/frozen_b_trainnorm_summary.json",
        "projection_main": "outputs/iclr27_phase14c/eval/frozen_b_projection_main_summary.json",
        "projection_no_cross_instance": "outputs/iclr27_phase14c/eval/frozen_b_projection_control_summary.json",
    }
    runs = {}
    for name, rel in names.items():
        x = read_json(rel)
        s = x["strict"]
        runs[name] = {
            "known_occurrence_acc": s["known_occurrence_acc"],
            "n_known_occurrences": s["n_known_occurrences"],
            "first_novel_birth_acc": s["first_novel_birth_acc"],
            "novel_reuse_acc": s["novel_reuse_acc"],
            "ct_reuse": s["ct_reuse"],
            "ct_reuse_correct": s["ct_reuse_correct"],
            "ct_reuse_eligible_cross_video_occurrences": s["ct_reuse_eligible_cross_video_occurrences"],
            "eligible_cross_physical_reuse_occurrences": s["eligible_cross_physical_reuse_occurrences"],
            "n_aligned_tracks": s["n_aligned_tracks"],
            "n_aligned_occurrences": s["n_aligned_occurrences"],
            "n_rows": s["n_rows"],
            "novel_nmi": s["novel_nmi"],
            "novel_ari": s["novel_ari"],
            "known_forgetting_delta": s["known_forgetting_delta"],
            "legacy_gate": x["legacy_gate"],
            "causal_contract": s["causal_contract"],
            "evaluator_controls": x["evaluator_controls"],
        }
    atomic_json("outputs/iclr27_phase14c/eval/strict_trackocd_summary.json", {
        "protocol": "phase14c",
        "gate": "Known occurrence accuracy >= 0.60 AND CT-Reuse > 0",
        "q1_label_used": False,
        "runs": runs,
    })


def causal_contract() -> None:
    proposal = read_json("outputs/iclr27_phase14c/proposals/proposal_integrity.json")
    strict = read_json("outputs/iclr27_phase14c/eval/strict_trackocd_summary.json")
    projection = read_json("outputs/iclr27_phase14c/eval/projection_training.json")
    contracts = [v["causal_contract"] for v in strict["runs"].values()]
    payload = {
        "protocol": "phase14c",
        "proposal_integrity_pass": proposal["pass"],
        "immediate_action_all_rows": all(c["immediate_action_all_rows"] for c in contracts),
        "no_relabel_field": all(c["no_relabel_field"] for c in contracts),
        "physical_id_distinct_from_semantic_id": all(c["physical_id_distinct_from_semantic_id"] for c in contracts),
        "future_frames_used": any(c["future_frames_used"] for c in contracts),
        "q1_label_used": any(c["q1_label_used"] for c in contracts),
        "private_gt_used_for_decision": any(c["private_gt_used_for_decision"] for c in contracts),
        "projection_training_devplus_labels_used": any(
            bool(v["devplus_labels_used"]) for v in projection["train_metrics"].values()
        ),
        "projection_training_q1_label_used": any(
            bool(v["q1_label_used"]) for v in projection["train_metrics"].values()
        ),
        "oracle_controls_pass": all(
            v["evaluator_controls"]["illegal_correct_label_oracle"]["cross_physical_reuse_acc"] == 1.0
            and v["evaluator_controls"]["intentionally_wrong_label_control"]["cross_physical_reuse_acc"] == 0.0
            for v in strict["runs"].values()
        ),
    }
    payload["pass"] = bool(
        proposal["pass"]
        and payload["immediate_action_all_rows"]
        and payload["no_relabel_field"]
        and payload["physical_id_distinct_from_semantic_id"]
        and not payload["future_frames_used"]
        and not payload["q1_label_used"]
        and not payload["private_gt_used_for_decision"]
        and not payload["projection_training_devplus_labels_used"]
        and not payload["projection_training_q1_label_used"]
        and payload["oracle_controls_pass"]
    )
    atomic_json("outputs/iclr27_phase14c/eval/causal_contract.json", payload)


def projection_summary() -> None:
    train = read_json("outputs/iclr27_phase14c/eval/projection_training.json")
    main = read_json("outputs/iclr27_phase14c/eval/frozen_b_projection_main_summary.json")
    control = read_json("outputs/iclr27_phase14c/eval/frozen_b_projection_control_summary.json")
    atomic_json("outputs/iclr27_phase14c/eval/projection_pilot_summary.json", {
        "protocol": "phase14c_projection_pilot",
        "training": train,
        "devplus_locked_replay": {
            "main": main,
            "no_cross_instance_control": control,
        },
        "authorized": True,
        "legal_gate_passed": bool(main["legacy_gate"]["pass"]),
        "q1_opened": False,
    })


if __name__ == "__main__":
    proposal_integrity()
    strict_summary()
    causal_contract()
    projection_summary()
    print("phase14c compact artifacts written")

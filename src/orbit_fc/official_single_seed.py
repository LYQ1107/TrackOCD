"""Official validation single-seed run for frozen ORBIT-FC (F1)."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import torch

from src.orbit_fc.evaluate import load_fc_model, evaluate_official
from src.orbit.protocol import load_gt

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
CHECKPOINT = ROOT / "runs/orbit_fc/fc_F1/model.pth"
GATE_THR = 0.5
REUSE_THR = 0.45


def decompose(logs, assignment, gt_by_sid):
    known = [l for l in logs if l["true_role"] in ("supported_known", "zero_shot_known")]
    novel = [l for l in logs if l["true_role"] == "novel"]
    routed = [l for l in novel if l["predicted_action"] != "KNOWN"]
    known_to_novel = sum(1 for l in known if l["predicted_action"] != "KNOWN")
    repeated = [l for l in novel if not l["first_occurrence"]]
    first = [l for l in novel if l["first_occurrence"]]
    repeated_fb = sum(1 for l in repeated if l["predicted_action"] == "NEW_NOVEL")
    first_merge = sum(1 for l in first if l["predicted_action"] == "EXISTING_NOVEL")
    wrong_existing = 0
    for l in routed:
        vid = int(l["predicted_virtual_novel_id"])
        mapped = assignment.get(vid)
        if l["predicted_action"] == "EXISTING_NOVEL" and mapped != int(l["true_class"]):
            wrong_existing += 1
    return {
        "num_known": len(known),
        "num_novel": len(novel),
        "num_routed": len(routed),
        "known_to_novel_error": known_to_novel / max(len(known), 1),
        "repeated_false_birth": repeated_fb / max(len(repeated), 1),
        "first_occurrence_merge": first_merge / max(len(first), 1),
        "wrong_existing_assignment": wrong_existing / max(len(routed), 1),
    }


def main():
    device = "cuda"
    model, ck = load_fc_model(CHECKPOINT, device=device)
    res, preds, logs = evaluate_official(model, ck, "pure", "full", "main_seed1027",
                                         device, gate_thr=GATE_THR,
                                         reuse_thr=REUSE_THR)
    gt = load_gt("pure")
    gt_by_sid = {g["sample_id"]: g for g in gt}
    seen = set()
    for l in logs:
        g = gt_by_sid.get(l["sample_id"], {})
        l["true_role"] = g.get("protocol_role", "?")
        l["true_class"] = g.get("ground_truth_category_id", "?")
        first = l["true_class"] not in seen
        l["first_occurrence"] = first
        if first:
            seen.add(l["true_class"])
    dec = decompose(logs, res["hungarian_assignment"], gt_by_sid)
    row = {
        "method": "ORBIT-FC",
        "seed": 1027,
        "gate_threshold": GATE_THR,
        "reuse_threshold": REUSE_THR,
        "all_acc": res["all_track_acc"],
        "known_acc": res["overall_known_acc"],
        "rn_acc": res["route_aware_novel_acc"],
        "cond_novel_acc": res["conditional_novel_acc"],
        "routing_recall": res["novel_routing_recall"],
        "nmi": res["novel_only_nmi"],
        "ari": res["novel_only_ari"],
        "count_error": res["novel_count_abs_error"],
        "predicted_novel_count": res["predicted_novel_count"],
        "known_to_novel_error": dec["known_to_novel_error"],
        "repeated_false_birth": dec["repeated_false_birth"],
        "wrong_existing_assignment": dec["wrong_existing_assignment"],
        "first_occurrence_merge": dec["first_occurrence_merge"],
    }
    out_dir = ROOT / "outputs" / "orbit_fc" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "orbit_fc_seed1027.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    (ROOT / "runs/orbit_fc/orbit_fc_seed1027.json").write_text(
        json.dumps({**res, "decomposition": dec, "prediction_log": preds},
                   indent=1, default=str))
    print(json.dumps(row, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evidence-only taxonomy of p16 pool-good but unreliable events."""
from __future__ import annotations
import collections, datetime, hashlib, json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "outputs/iclr27_phase80c/audit/proposal_quality_event_records.json"
OUT = ROOT / "outputs/iclr27_phase81p/audit/assignment_failure_taxonomy.json"

def classify(side):
    frag = int(side.get("fragmentation_transitions", 0)); ti = float(side.get("max_event_track_temporal_iou", 0.0)); ri = float(side.get("max_iou", 0.0)); rows = int(side.get("rows_used", 0))
    if side.get("pool_class") == "pool_present_but_no_reliable_candidate" or ri < 0.5:
        return "F_LOW_SCORE_SUPPRESSION" if float(side.get("max_q0_score", 0.0)) < 0.25 else "H_MOTION_DRIFT"
    if frag >= 3 and ti < 0.5:
        return "C_FRAGMENTATION"
    if frag >= 1 and ti < 0.5:
        return "B_MISSED_CONTINUATION"
    if rows <= 1 and ti < 0.5:
        return "D_PREMATURE_TERMINATION"
    if ti < 0.5:
        return "A_ASSOCIATION_SWAP"
    return "J_UNKNOWN_OTHER"

def main():
    data = json.loads(IN.read_text(encoding="utf-8"))
    # The registered Stage-2 taxonomy is specifically the 36 events for which
    # both sides have a pool candidate but the frozen physical assignment or
    # temporal aggregation is unreliable.  The other 15 events are proposal
    # coverage failures and remain separate evidence.
    records = [x for x in data if x.get("prefix") == 16 and x.get("polarity") == "positive" and x.get("joint_class") == "pool_has_candidates_assignment_or_temporal_gap"]
    rows = []
    for event in records:
        sides = []
        for name in ("source", "target"):
            side = event[name]
            sides.append({"side": name, "taxonomy": classify(side), "pool_class": side.get("pool_class"), "candidate_count": side.get("candidate_count"), "rows_used": side.get("rows_used"), "max_iou": side.get("max_iou"), "max_q0_score": side.get("max_q0_score"), "temporal_iou": side.get("max_event_track_temporal_iou"), "fragmentation_transitions": side.get("fragmentation_transitions"), "event_reliable_rows": side.get("event_reliable_rows"), "tracklet_key": event.get(name + "_tracklet_key"), "video_id": event.get(name + "_video")})
        dominant = collections.Counter(s["taxonomy"] for s in sides if s["taxonomy"] != "J_UNKNOWN_OTHER").most_common(1)
        rows.append({"event_key": event["event_key"], "model_event_uid": event["model_event_uid"], "fold": event["fold"], "source_reliable": event["source_reliable"], "target_reliable": event["target_reliable"], "sides": sides, "dominant_failure": dominant[0][0] if dominant else "J_UNKNOWN_OTHER", "evidence_only": True, "training_or_inference_labels_used": False})
    summary = collections.Counter(); byfold = collections.defaultdict(collections.Counter)
    for row in rows:
        summary[row["dominant_failure"]] += 1; byfold[str(row["fold"])][row["dominant_failure"]] += 1
    result = {"schema_version": "phase81p.assignment_failure_taxonomy.v1", "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "input": str(IN), "input_sha256": hashlib.sha256(IN.read_bytes()).hexdigest(), "denominator": 76, "assignment_gap_denominator": 36, "unreliable_event_count": len(rows), "taxonomy_counts": dict(summary), "by_fold": {k: dict(v) for k, v in byfold.items()}, "categories": ["A_ASSOCIATION_SWAP", "B_MISSED_CONTINUATION", "C_FRAGMENTATION", "D_PREMATURE_TERMINATION", "E_DUPLICATE_BIRTH", "F_LOW_SCORE_SUPPRESSION", "G_ONE_TO_ONE_CONFLICT", "H_MOTION_DRIFT", "I_APPEARANCE_DRIFT", "J_UNKNOWN_OTHER"], "records": rows, "causal_contract": {"evaluator_only": True, "held_labels_used_for_training": False, "future_rows_or_tracks": False, "physical_ids_as_model_input": False}}
    OUT.parent.mkdir(parents=True, exist_ok=True); tmp = OUT.with_name("." + OUT.name + ".tmp"); tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); os.replace(tmp, OUT)
    print(json.dumps({"unreliable_event_count": len(rows), "taxonomy_counts": dict(summary)}, indent=2))

if __name__ == "__main__":
    main()

"""B0: replay the immutable Phase17R M1 action CSV under Phase18 metrics.

Only rows present in the historical 1,903-row audit CSV have an action.  The
registered mapping for undefined historical actions is DEFER; no M1 output is
retrained or relabelled.  This is a coverage/compatibility control, not a
Phase18-designed DEFER baseline.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.iclr27_phase18.evaluation.baseline_b1 import atomic_json, event_metrics, load_data, load_jsonl


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/iclr27_phase18"


def load_historical() -> dict[str, dict[str, Any]]:
    paths = sorted((ROOT / "outputs/iclr27_phase17r/csv").glob("public_final_audit_decisions_*.csv"))
    path = paths[0]
    with path.open(newline="") as f:
        rows = {r["row_key"]: r for r in csv.DictReader(f)}
    return {"path": str(path), "rows": rows}


def run() -> dict[str, Any]:
    data = load_data(); hist = load_historical(); mapped = hist["rows"]
    tracklets = data["tracklets"]
    events = data["positives"] + data["negatives"]
    records = []

    for event in events:
        memory: dict[int, dict[str, Any]] = {}
        local: dict[str, int] = {}
        errors = []; decisions_by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)

        def process(track_key: str, evaluator_category: int, phase: str) -> list[dict[str, Any]]:
            t = tracklets[track_key]; out = []
            for pos, idx in enumerate(t["row_indices"]):
                row = data["rows"][int(idx)]; h = mapped.get(row["row_key"])
                action = "DEFER"; sid = None; evidence = "historical_action_undefined_mapped_to_defer"
                if h is not None:
                    raw = h.get("action", "").lower(); raw_sid = h.get("semantic_id", "")
                    try: parsed_sid = int(raw_sid)
                    except (TypeError, ValueError): parsed_sid = None
                    if raw == "known" and parsed_sid is not None:
                        action, sid, evidence = "KNOWN", parsed_sid, "phase17r_m1_replay"
                    elif raw == "new" and parsed_sid is not None:
                        if parsed_sid in memory:
                            errors.append(f"duplicate_historical_birth:{parsed_sid}")
                        else:
                            memory[parsed_sid] = {"birth_tracklet": track_key, "birth_video": int(t["video_id"]), "eval_category_not_model_input": evaluator_category}
                            local[track_key] = parsed_sid; action, sid, evidence = "NEW_NOVEL", parsed_sid, "phase17r_m1_replay"
                    elif raw == "existing" and parsed_sid is not None and parsed_sid in memory:
                        state = memory[parsed_sid]
                        if state["birth_tracklet"] != track_key and state["birth_video"] != int(t["video_id"]):
                            action, sid, evidence = "EXISTING_NOVEL", parsed_sid, "phase17r_m1_replay"
                            local[track_key] = parsed_sid
                        else:
                            errors.append(f"illegal_historical_existing:{parsed_sid}")
                out.append({"row_key": row["row_key"], "tracklet_position": pos, "phase": phase, "action": action,
                            "semantic_id": sid, "readiness_score": float(h.get("observability_score", 0.0)) if h else 0.0,
                            "predicted_ready": bool(h and h.get("predicted_observable", "False") == "True"), "evidence": evidence})
            decisions_by_track[track_key].extend(out)
            return out

        source_cat = int(event.get("category_gt_denominator_only", event.get("distractor_category_gt_denominator_only")))
        source_decisions = []
        for key in event["source_tracklet_keys"]: source_decisions.extend(process(key, source_cat, "source"))
        target_cat = int(event.get("category_gt_denominator_only", event.get("target_category_gt_denominator_only")))
        target = process(event["target_tracklet_key"], target_cat, "target")
        prefix = int(event["target_first_reliable_prefix_index_gt_only"]); post = target[prefix:]

        def correct_existing(d: dict[str, Any]) -> bool:
            if d["action"] != "EXISTING_NOVEL" or d["semantic_id"] is None: return False
            m = memory.get(int(d["semantic_id"]))
            return bool(m and m["eval_category_not_model_input"] == target_cat and m["birth_video"] != int(event["target_video"]) and m["birth_tracklet"] != event["target_tracklet_key"])

        first_offset = next((i for i, d in enumerate(post) if d["action"] != "DEFER"), None)
        first = post[first_offset] if first_offset is not None else None
        correct_offsets = [i for i, d in enumerate(post) if correct_existing(d)]
        premature = [d for d in target[:prefix] if d["action"] != "DEFER"]
        records.append({"event_key": event["event_key"], "kind": event["kind"], "fold": event["fold"],
                        "target_category_gt_evaluator_only": target_cat, "source_decisions": source_decisions, "target_decisions": target,
                        "first_commit_after_prefix": first, "first_commit_offset": first_offset,
                        "first_commit_correct_existing": bool(first and correct_existing(first)),
                        "post_prefix_correct_existing_rows": len(correct_offsets), "post_prefix_rows": len(post),
                        "time_to_correct_commit": min(correct_offsets) if correct_offsets else None,
                        "pre_prefix_rows": prefix, "pre_prefix_defer_rows": prefix - len(premature),
                        "premature_commit": bool(premature), "unresolved_after_prefix": first is None,
                        "state_count": len(memory), "merge_count": 0,
                        "duplicate_target_births": sum(m["eval_category_not_model_input"] == target_cat and m["birth_video"] == int(event["target_video"]) for m in memory.values()),
                        "transition_errors": errors})

    known_rows = []; by_cat = defaultdict(list)
    for row in data["rows"]:
        if row["role17"] != "known_audit" or row["gt_role_common"] != "supported_known": continue
        h = mapped.get(row["row_key"]); pred = int(h["semantic_id"]) if h and h.get("action") == "known" and h.get("semantic_id", "").isdigit() else -1
        ok = int(pred == int(row["cat_i"])); known_rows.append(ok); by_cat[int(row["cat_i"])].append(ok)
    target_rows = []; target_scores = []; target_labels = []
    for e in data["positives"]:
        for idx in tracklets[e["target_tracklet_key"]]["row_indices"]:
            row = data["rows"][int(idx)]; h = mapped.get(row["row_key"])
            target_rows.append(int(h is not None)); target_scores.append(float(h.get("observability_score", 0.0)) if h else 0.0); target_labels.append(bool(row["reliable"]))
    s=np.asarray(target_scores,np.float32); y=np.asarray(target_labels,bool); pred=s>=.5
    result = {"protocol":"trackocd_iclr27_phase18_B0_phase17r_m1_historical_controller_replay",
              "historical_csv":hist["path"], "historical_rows":len(mapped), "phase18_target_rows":len(target_rows),
              "mapped_target_rows":sum(target_rows), "mapped_target_row_fraction":float(np.mean(target_rows)),
              "undefined_action_mapping":"DEFER", "gt_or_future_used_as_inference_input":False,
              "metrics":event_metrics(records),
              "known":{"rows":len(known_rows),"micro_accuracy":float(np.mean(known_rows)) if known_rows else 0.0,"category_macro_accuracy":float(np.mean([np.mean(v) for v in by_cat.values()])) if by_cat else 0.0,"by_category":{str(c):{"correct":sum(v),"rows":len(v)} for c,v in by_cat.items()}},
              "reliability":{"rows":len(y),"positive_rows":int(y.sum()),"auroc":float(roc_auc_score(y,s)) if len(set(y.tolist()))==2 else None,"auprc":float(average_precision_score(y,s)) if y.sum() else None,"precision_at_0.5":int((pred&y).sum())/max(int(pred.sum()),1),"recall_at_0.5":int((pred&y).sum())/max(int(y.sum()),1)},
              "event_records":records}
    atomic_json(OUT/"eval/b0_historical_replay.json", result)
    print(json.dumps({"coverage":result["mapped_target_row_fraction"],"metrics":result["metrics"],"known":result["known"],"reliability":result["reliability"]},indent=2,sort_keys=True))
    return result


if __name__ == "__main__": run()

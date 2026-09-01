"""Join public truth only after the Phase19R prediction freeze marker."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.iclr27_phase19r.data.stream import Phase19RData


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase19r"
MARKER = OUT / "completion/public_predictions.frozen"


def sha(path: Path) -> str:
    h = hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()


def load_truth() -> list[dict[str, Any]]:
    src = ROOT / "data/iclr27_phase19r/sources"
    out: list[dict[str, Any]] = []
    for name in ("positive_events.jsonl", "negative_events.jsonl"):
        out.extend(json.loads(x) for x in (src / name).read_text().splitlines() if x.strip())
    return sorted(out, key=lambda x: x["event_key"])


def track_categories(data: Phase19RData) -> dict[str, int]:
    # This map is constructed after freeze from evaluator-only rows; it is never
    # serialized into the frozen predictions.
    return {k: int(data.track_category[k]) for k in data.track_category}


def state_map(rec: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(s["sid"]): s for s in rec.get("final_states", []) if s.get("sid") is not None}


def score_record(pred: dict[str, Any], truth: dict[str, Any], cats: dict[str, int]) -> dict[str, Any]:
    target_cat = int(truth.get("target_category_gt_denominator_only", truth.get("category_gt_denominator_only")))
    kind = "positive_existing" if truth.get("kind") == "positive_existing" else "negative_new"
    source_keys = set(truth["source_tracklet_keys"])
    target_key = truth["target_tracklet_key"]
    prefix = int(truth["target_first_reliable_prefix_index_gt_only"])
    decisions = pred.get("target_decisions", [])
    states = state_map(pred)

    def state_category(sid: int | None) -> int | None:
        if sid is None or sid not in states:
            return None
        return cats.get(str(states[sid].get("birth_track")))

    def correct_existing(d: dict[str, Any]) -> bool:
        if d.get("action") != "EXISTING":
            return False
        sid = d.get("semantic_id")
        if sid is None:
            return False
        birth = states.get(int(sid), {}).get("birth_track")
        return bool(birth in source_keys and state_category(int(sid)) == target_cat and birth != target_key)

    post = decisions[prefix:]
    first = next((d for d in post if d.get("action") != "DEFER"), None)
    existing = [d for d in post if d.get("action") == "EXISTING"]
    first_correct = bool(first and (correct_existing(first) if kind == "positive_existing" else first.get("action") == "NEW"))
    false_merge = bool(kind == "negative_new" and first and first.get("action") == "EXISTING")
    target_births = sum(1 for s in states.values() if s.get("birth_track") == target_key and state_category(int(s["sid"])) == target_cat)
    existing_correct = sum(correct_existing(d) for d in existing)
    return {"event_key": pred["event_key"], "kind": kind, "fold": truth.get("fold"),
            "target_category": target_cat, "target_video": int(truth["target_video"]),
            "first_commit": first, "first_commit_correct": first_correct,
            "post_prefix_rows": len(post), "post_prefix_correct_rows": int(existing_correct),
            "existing_rows": len(existing), "existing_correct_rows": int(existing_correct),
            "negative_false_merge": false_merge, "premature": any(d.get("action") != "DEFER" for d in decisions[:prefix]),
            "unresolved": first is None, "duplicate_target_births": int(target_births),
            "state_count": len(states)}


def metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    pos = [r for r in records if r["kind"] == "positive_existing"]
    neg = [r for r in records if r["kind"] == "negative_new"]
    by_cat: dict[int, list[int]] = defaultdict(list); by_video: dict[int, list[int]] = defaultdict(list)
    for r in pos:
        by_cat[int(r["target_category"])].append(int(r["first_commit_correct"])); by_video[int(r["target_video"])].append(int(r["first_commit_correct"]))
    ex_total = sum(r["existing_rows"] for r in records); ex_good = sum(r["existing_correct_rows"] for r in records)
    post_total = sum(r["post_prefix_rows"] for r in pos); post_good = sum(r["post_prefix_correct_rows"] for r in pos)
    new_rows = sum(1 for r in records for d in (r.get("first_commit"),) if d and d.get("action") == "NEW")
    new_good = sum(1 for r in neg if r.get("first_commit") and r["first_commit"].get("action") == "NEW")
    ep = ex_good / max(ex_total, 1); er = ex_good / max(post_total, 1)
    return {"positive_events": len(pos), "negative_events": len(neg),
            "commit_ct": {"correct": int(sum(r["first_commit_correct"] for r in pos)), "eligible": len(pos),
                          "recall": float(np.mean([r["first_commit_correct"] for r in pos])) if pos else 0.0},
            "post_prefix_ct": {"correct_rows": int(post_good), "rows": int(post_total), "recall": post_good / max(post_total, 1)},
            "existing_precision": ep, "existing_recall": er, "existing_f1": 2 * ep * er / max(ep + er, 1e-9),
            "new_precision": new_good / max(new_rows, 1), "new_recall": new_good / max(len(neg), 1),
            "negative_false_merge_rate": float(np.mean([r["negative_false_merge"] for r in neg])) if neg else 0.0,
            "duplicate_births": int(sum(r["duplicate_target_births"] for r in records)),
            "premature_rate": float(np.mean([r["premature"] for r in records])) if records else 0.0,
            "unresolved_rate": float(np.mean([r["unresolved"] for r in records])) if records else 0.0,
            "category_macro_reuse": float(np.mean([np.mean(v) for v in by_cat.values()])) if by_cat else 0.0,
            "category_coverage": int(sum(any(v) for v in by_cat.values())),
            "video_coverage": int(sum(any(v) for v in by_video.values()))}


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True); a = p.parse_args()
    if not MARKER.exists():
        raise SystemExit("prediction freeze marker is required before public truth join")
    freeze = json.loads((OUT / "manifests/prediction_freeze.json").read_text())
    truth = load_truth(); data = Phase19RData(final=True); cats = track_categories(data)
    results: dict[str, Any] = {}
    for name, info in freeze["candidates"].items():
        path = Path(info["path"]); payload = json.loads(path.read_text())
        if sha(path) != info["sha256"]:
            raise SystemExit(f"frozen prediction hash mismatch for {name}")
        by_key = {r["event_key"]: r for r in payload["records"]}
        records = [score_record(by_key[e["event_key"]], e, cats) for e in truth]
        results[name] = {"candidate": name, "prediction_sha256": info["sha256"], "metrics": metrics(records), "records": records}
    result = {"protocol": "trackocd_iclr27_phase19r_public_measurement_after_freeze", "freeze_timestamp_utc": freeze["freeze_timestamp_utc"], "truth_join_after_freeze": True, "event_count": len(truth), "candidates": results}
    a.out.parent.mkdir(parents=True, exist_ok=True); tmp = a.out.with_name(a.out.name + ".tmp"); tmp.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"); os.replace(tmp, a.out)
    print(json.dumps({k: v["metrics"] for k, v in results.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

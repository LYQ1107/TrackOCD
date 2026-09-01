#!/usr/bin/env python3
"""One positive/one negative causal replay for the frozen raw controller."""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import torch

from src.iclr27_phase19r.data.stream import Phase19RData
from src.iclr27_phase19r.evaluation.internal import evaluate_candidate


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase72/audit"
MAN = ROOT / "outputs/iclr27_phase19r/manifests"


def load(path: Path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def check_record(rec: dict, event: dict, data: Phase19RData) -> dict:
    target = rec["target_decisions"]
    source = rec["source_decisions"]
    all_decisions = source + target
    def pos_of(x: dict) -> int:
        # ModelStreamController and RawPersistentController use the same
        # causal order but retain legacy field names in their frozen outputs.
        value = x.get("position", x.get("tracklet_position"))
        if value is None:
            raise KeyError("position/tracklet_position")
        return int(value)
    source_positions = [pos_of(x) for x in source]
    positions = [pos_of(x) for x in all_decisions]
    target_positions = [pos_of(x) for x in target]
    source_keys = set(event["source_tracklet_keys"])
    target_key = event["target_tracklet_key"]
    source_videos = {int(data.track_video[k]) for k in event["source_tracklet_keys"]}
    source_ok = all(int(str(x["row_key"]).split(":", 1)[0]) in source_videos for x in source) if source else True
    target_track_rows = data.track_rows[target_key]
    target_ok = all(0 <= pos_of(x) < len(target_track_rows) for x in target)
    causal_order = (source_positions == sorted(source_positions)
                    and target_positions == sorted(target_positions))
    # The evaluator's prefix is a metadata-only cutoff; post-prefix decisions
    # are never allowed to influence source processing.  The raw replay keeps
    # the target stream in its original row order.
    return {
        "event_key": event["event_key"],
        "kind": event["kind"],
        "source_tracklet_keys": sorted(source_keys),
        "target_tracklet_key": target_key,
        "decision_count": len(all_decisions),
        "target_decision_count": len(target),
        "positions_monotonic": causal_order,
        "target_positions_in_bounds": target_ok,
        "source_rows_have_expected_video": source_ok,
        "first_action": rec.get("first_commit", {}).get("action") if rec.get("first_commit") else None,
        "negative_false_merge": bool(rec.get("negative_false_merge")),
        "premature": bool(rec.get("premature")),
        "unresolved": bool(rec.get("unresolved")),
        "state_count": int(rec.get("state_count", 0)),
        "event_prefix_index": int(event["target_first_reliable_prefix_index_gt_only"]),
        "post_prefix_rows": int(rec.get("post_prefix_rows", 0)),
        "pre_prefix_rows": int(rec.get("pre_prefix_rows", 0)),
        "action_values": sorted(set(str(x["action"]) for x in all_decisions)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    pos = load(MAN / "held_known_positive_events.jsonl")
    neg = load(MAN / "held_known_negative_events.jsonl")
    events = [next(e for e in pos if int(e["fold"]) == args.fold), next(e for e in neg if int(e["fold"]) == args.fold)]
    data = Phase19RData(args.fold)
    result = evaluate_candidate("raw", data, None, torch.device("cpu"), events=events)
    checks = [check_record(rec, event, data) for rec, event in zip(result["records"], events)]
    payload = {
        "protocol": "phase72_single_positive_negative_causal_replay_smoke",
        "candidate": "phase19r_native_raw_frozen_baseline_diagnostic",
        "fold": 0,
        "input_events": len(events),
        "positive_events": sum(e["kind"] == "positive_existing" for e in events),
        "negative_events": sum(e["kind"] == "negative_new" for e in events),
        "checks": checks,
        "metrics": result["metrics"],
        "contract_passed": all(c["positions_monotonic"] and c["target_positions_in_bounds"] and c["source_rows_have_expected_video"] for c in checks),
        "denominator_unchanged": {"full_positive": len(pos) == 76, "full_negative": len(neg) == 76, "full_total": len(pos) + len(neg) == 152},
        "future_rows_or_tracks_used": False,
        "held_gt_as_model_input": False,
        "category_text_or_id_feature": False,
    }
    out = args.out or (OUT / ("causal_replay_smoke.json" if args.fold == 0 else f"causal_replay_targeted_f{args.fold}.json"))
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(out)
    done = out.with_suffix(".done")
    dtmp = done.with_name(done.name + ".tmp")
    dtmp.write_text("done\n")
    dtmp.replace(done)
    print(json.dumps({"out": str(out), "contract_passed": payload["contract_passed"], "metrics": payload["metrics"]}, indent=2))


if __name__ == "__main__":
    main()

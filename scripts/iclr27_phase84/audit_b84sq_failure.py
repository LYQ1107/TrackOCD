#!/usr/bin/env python3
"""Audit the frozen B84S-Q event replay without changing any decisions.

The replay is deliberately post-hoc: event GT is used only to score the
already-frozen choices.  This script classifies observable versus selector
failures and records the exact 76-positive/76-negative denominator.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPLAY = ROOT / "outputs/iclr27_phase84/metrics/b84s_event_replay_b84sq_v3.json"
DEFAULT_FORMAL = ROOT / "outputs/iclr27_phase84/metrics/b84s_formal_aggregate_b84sq_v3.json"
DEFAULT_MANIFEST = ROOT / "outputs/iclr27_phase84/manifests/b84sq_balanced_v3_manifest.json"
DEFAULT_OUT = ROOT / "outputs/iclr27_phase84/audit/b84sq_failure_audit.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def stats(values: list[int | float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "n": len(values),
        "min": float(min(values)),
        "median": float(statistics.median(values)),
        "mean": float(statistics.fmean(values)),
        "max": float(max(values)),
    }


def classify(record: dict[str, Any]) -> str:
    """Assign one mutually exclusive causal diagnosis to a frozen event."""
    if not record["source_reliable_frozen"]:
        return "source_observability_unreliable"
    if not record["target_reliable_frozen"]:
        return "target_observability_unreliable"
    if record["candidate_count_total"] <= 0:
        return "native_candidate_pool_empty"
    if record["selected_reliable"]:
        return "learned_selection_reliable"
    if record["selected_candidate"]:
        return "candidate_selected_but_iou_unreliable"
    return "defer_with_reliable_target_available"


def fold_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for prefix in (1, 2, 4, 8, 16):
        for fold in range(4):
            for polarity in ("positive", "negative"):
                rs = [r for r in records if r["prefix"] == prefix and r["fold"] == fold and r["polarity"] == polarity]
                if not rs:
                    continue
                out.append({
                    "prefix": prefix,
                    "fold": fold,
                    "polarity": polarity,
                    "events": len(rs),
                    "source_reliable": sum(int(r["source_reliable_frozen"]) for r in rs),
                    "target_reliable": sum(int(r["target_reliable_frozen"]) for r in rs),
                    "both_reliable": sum(int(r["both_reliable_frozen"]) for r in rs),
                    "candidate_available": sum(int(r["candidate_count_total"] > 0) for r in rs),
                    "selected_candidate": sum(int(r["selected_candidate"]) for r in rs),
                    "selected_reliable": sum(int(r["selected_reliable"]) for r in rs),
                    "raw_source_mean_reliable": sum(int(r["raw_selected_reliable"]) for r in rs),
                    "candidate_count": stats([int(r["candidate_count_total"]) for r in rs]),
                    "taxonomy": dict(sorted(Counter(classify(r) for r in rs).items())),
                })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", default=str(DEFAULT_REPLAY))
    ap.add_argument("--formal", default=str(DEFAULT_FORMAL))
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--decision", default="B84S_Q_FAIL_SELECTION_DID_NOT_IMPROVE_FROZEN_Q0")
    ap.add_argument("--route", default="B84S-Q")
    args = ap.parse_args()
    replay_path, formal_path, manifest_path, out_path = map(Path, (args.replay, args.formal, args.manifest, args.out))
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    formal = json.loads(formal_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = list(replay["records"])
    p16 = [r for r in records if r["prefix"] == 16]
    by_pol: dict[str, dict[str, Any]] = {}
    for polarity in ("positive", "negative"):
        rs = [r for r in p16 if r["polarity"] == polarity]
        by_pol[polarity] = {
            "events": len(rs),
            "source_reliable": sum(int(r["source_reliable_frozen"]) for r in rs),
            "target_reliable": sum(int(r["target_reliable_frozen"]) for r in rs),
            "both_reliable": sum(int(r["both_reliable_frozen"]) for r in rs),
            "candidate_available": sum(int(r["candidate_count_total"] > 0) for r in rs),
            "selected_candidate": sum(int(r["selected_candidate"]) for r in rs),
            "selected_reliable": sum(int(r["selected_reliable"]) for r in rs),
            "raw_source_mean_reliable": sum(int(r["raw_selected_reliable"]) for r in rs),
            "taxonomy": dict(sorted(Counter(classify(r) for r in rs).items())),
            "candidate_count": stats([int(r["candidate_count_total"]) for r in rs]),
        }
    taxonomy_events = []
    for r in p16:
        taxonomy_events.append({
            "event_key": r["event_key"],
            "model_event_uid": r["model_event_uid"],
            "fold": r["fold"],
            "model_fold": r.get("model_fold"),
            "polarity": r["polarity"],
            "source_tracklet_key": r["source_tracklet_key"],
            "target_tracklet_key": r["target_tracklet_key"],
            "source_reliable_frozen": r["source_reliable_frozen"],
            "target_reliable_frozen": r["target_reliable_frozen"],
            "both_reliable_frozen": r["both_reliable_frozen"],
            "candidate_count_total": r["candidate_count_total"],
            "selected_candidate": r["selected_candidate"],
            "selected_reliable": r["selected_reliable"],
            "raw_selected_reliable": r["raw_selected_reliable"],
            "taxonomy": classify(r),
        })
    # The B84S-Q manifest intentionally fell back to 3 disjoint folds because
    # a four-way split did not retain enough fit/validation groups.  Event fold
    # 3 is therefore deterministically evaluated with model fold 0; this is
    # reported explicitly rather than hidden in the aggregate.
    decision = str(args.decision)
    result = {
        "schema_version": "trackocd.phase84.b84sq.failure_audit.v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "decision": decision,
        "route": str(args.route),
        "inputs": {
            "replay": str(replay_path.resolve()),
            "replay_sha256": sha(replay_path),
            "formal_aggregate": str(formal_path.resolve()),
            "formal_aggregate_sha256": sha(formal_path),
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": sha(manifest_path),
        },
        "protocol": {
            "positive_events": 76,
            "negative_events": 76,
            "prefixes": [1, 2, 4, 8, 16],
            "reliable_rule": "assigned == 1 and posthoc IoU >= 0.5",
            "event_labels_posthoc_only": True,
            "controller_run": False,
            "public_dev_q1_sealed_accessed": False,
            "future_rows_or_tracks": False,
            "ids_as_model_input": False,
        },
        "formal_validation": formal.get("validation_weighted", {}),
        "manifest_contract": {
            "groups": manifest.get("groups"),
            "candidate_rows": manifest.get("candidate_rows"),
            "fold_count": manifest.get("fold_count"),
            "fold_assignment": manifest.get("fold_assignment"),
            "event_videos_excluded": len(manifest.get("event_videos_excluded", [])),
            "source_support_may_cross_fold": {k: v.get("source_support_may_cross_fold") for k, v in manifest.get("folds", {}).items()},
        },
        "p16": by_pol,
        "fold_prefix_summary": fold_summary(records),
        "p16_event_taxonomy": taxonomy_events,
        "root_cause_evidence": {
            "candidate_pool_is_not_empty": all(r["candidate_count_total"] > 0 for r in p16),
            "positive_observable_target_events": by_pol["positive"]["target_reliable"],
            "positive_learned_reliable_events": by_pol["positive"]["selected_reliable"],
            "positive_raw_source_mean_reliable_events": by_pol["positive"]["raw_source_mean_reliable"],
            "negative_learned_reliable_events_false_support": by_pol["negative"]["selected_reliable"],
            "interpretation": "The native candidate pool is nonempty, but this frozen source-conditioned selection route must be judged against the same-space raw source-mean diagnostic and frozen-Q0 reliability. The event-level counts above separate source/target observability from learned ranking and DEFER behavior; they are post-hoc evidence, not controller or sealed results.",
        },
    }
    atomic_json(out_path, result)
    print(json.dumps({"out": str(out_path.resolve()), "decision": decision, "p16": by_pol}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""GT-only post-replay analysis for Phase-9A Stage-1 contamination.

This script never feeds labels into a replay.  It joins the already-emitted
baseline/gated CSVs and gated event log to the locked Q1 DEV labels solely to
describe where births and blocked attachments came from.  It also reports
the same strict metrics used by ``strict_eval_any`` so the comparison is
auditable from one JSON artifact.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.iclr27_phase4s.frontend import align_pred_to_gt, gt_track_boxes
from src.iclr27_phase4s.protocol import group_tracks, load_proposals
from src.iclr27_phase7a.evaluation.strict_eval_any import load_gt_videos

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
Q1_VIDEOS = [88, 90, 122, 291, 334, 888, 931, 1159, 1232, 1276,
             1572, 1865, 2254, 2347, 2564, 2675, 2690, 2759, 2802, 2888]


def read_csv(path: str) -> list[dict]:
    with open(ROOT / path, newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def role_map(rows: list[dict]) -> tuple[dict, dict]:
    # ``csv.DictReader`` leaves identifiers as strings, while the canonical
    # protocol helpers operate on numeric proposal fields.  Normalize only
    # the physical-key fields used for alignment; semantic columns remain
    # untouched for the source/action accounting below.
    physical_rows = []
    for row in rows:
        r = dict(row)
        for name in ("video_id", "frame_id", "proposal_local_id", "track_id"):
            if r.get(name) not in (None, ""):
                r[name] = int(r[name])
        physical_rows.append(r)
    tracks = group_tracks(physical_rows)
    stream, labels = load_gt_videos(Q1_VIDEOS)
    mapping = align_pred_to_gt(tracks, gt_track_boxes(stream))
    roles = {}
    for key, sid in mapping.items():
        role = labels[sid]["protocol_role"]
        if role == "novel":
            source = "true_novel"
        elif role in ("supported_known", "zero_shot_known"):
            source = "known_confusion"
        else:
            source = "other_aligned"
        roles[tuple(key)] = {
            "source": source,
            "role": role,
            "sample_id": sid,
            "category_id": int(labels[sid]["ground_truth_category_id"]),
        }
    return roles, mapping


def source_for(row: dict, roles: dict) -> str:
    return roles.get((int(row["video_id"]), int(row["track_id"])),
                    {"source": "fp_noisy"})["source"]


def source_counts(rows: list[dict], roles: dict, action: str = "new") -> dict:
    vals = [source_for(r, roles) for r in rows if r.get("sem_action") == action]
    c = Counter(vals)
    return {k: int(c.get(k, 0)) for k in
            ("true_novel", "known_confusion", "fp_noisy", "other_aligned")}


def strict_metrics(path: str) -> dict:
    return json.loads((ROOT / path / "summary.json").read_text())["strict"]


def requested_metrics(strict: dict, rows: list[dict], roles: dict) -> dict:
    novel = [r for r in rows if source_for(r, roles) == "true_novel"]
    return {
        "Known": float(strict["known_occurrence_acc"]),
        "First birth": float(strict["first_novel_birth_acc"]),
        "Novel reuse": float(strict["novel_reuse_acc"]),
        "CT-Reuse": float(strict["cross_physical_reuse_acc"]),
        "Known->Existing error": float(strict["known_to_existing_rate"]),
        "Novel->Known error": float(
            sum(r.get("sem_action") == "known" for r in novel)
            / max(len(novel), 1)),
    }


def load_events(path: str) -> list[dict]:
    events = []
    with open(ROOT / path) as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events


def contamination(events: list[dict], gate_rows: list[dict], roles: dict) -> dict:
    by_idx = {int(e["row_index"]): e for e in events}
    blocked = [e for e in events if e.get("gate_blocked")]
    blocked_sources = Counter()
    for e in blocked:
        key = (int(e["video_id"]), int(e["track_id"]))
        blocked_sources[roles.get(key, {"source": "fp_noisy"})["source"]] += 1

    # A gate state is considered trusted if an emitted causal event reports
    # reuse_allowed=True.  This is derived from prefixes only; GT is joined
    # afterwards to label the birth source.
    trusted_slots = set()
    for e in events:
        rel = e.get("reliability") or {}
        if rel.get("reuse_allowed"):
            trusted_slots.add(int(e["selected_slot"]))
    births = {}
    for i, r in enumerate(gate_rows):
        if r.get("sem_action") == "new":
            births[int(r["sem_slot"])] = {
                "row_index": i,
                "source": roles.get(
                    (int(r["video_id"]), int(r["track_id"])),
                    {"source": "fp_noisy"})["source"],
            }
    trusted_birth_sources = Counter()
    candidate_not_trusted = Counter()
    for slot, b in births.items():
        (trusted_birth_sources if slot in trusted_slots
         else candidate_not_trusted)[b["source"]] += 1

    return {
        "baseline_contamination_proxy": {
            "definition": "all baseline NEW rows are immediately reusable",
            "birth_sources": source_counts([], roles),  # filled by caller
        },
        "gated_candidate_birth_sources": {
            k: int(v) for k, v in Counter(
                b["source"] for b in births.values()).items()
        },
        "gated_trusted_birth_sources": {
            k: int(v) for k, v in trusted_birth_sources.items()
        },
        "gated_candidate_not_trusted_sources": {
            k: int(v) for k, v in candidate_not_trusted.items()
        },
        "blocked_untrusted_attachment_attempts": int(len(blocked)),
        "blocked_attachment_sources": {
            k: int(v) for k, v in blocked_sources.items()
        },
        "blocked_wrong_attachment_attempts": int(
            sum(v for k, v in blocked_sources.items()
                if k in ("known_confusion", "fp_noisy", "other_aligned"))),
        "n_events": len(events),
        "n_event_rows_without_gate_log": int(
            len(set(range(len(gate_rows))) - set(by_idx))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-csv", required=True)
    ap.add_argument("--baseline-summary", required=True)
    ap.add_argument("--gate-csv", required=True)
    ap.add_argument("--gate-summary", required=True)
    ap.add_argument("--gate-events", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    baseline = read_csv(args.baseline_csv)
    gated = read_csv(args.gate_csv)
    if len(baseline) != len(gated):
        raise ValueError(f"baseline/gate row mismatch: {len(baseline)} vs {len(gated)}")
    roles, mapping = role_map(baseline)
    events = load_events(args.gate_events)

    base_births = source_counts(baseline, roles)
    gate_births = source_counts(gated, roles)
    cont = contamination(events, gated, roles)
    cont["baseline_contamination_proxy"]["birth_sources"] = base_births
    cont["baseline_contamination_proxy"]["wrong_births_immediately_reusable"] = int(
        base_births.get("known_confusion", 0)
        + base_births.get("fp_noisy", 0)
        + base_births.get("other_aligned", 0))
    cont["gated_candidate_birth_sources"] = gate_births
    cont["gated_trusted_wrong_births"] = int(sum(
        cont["gated_trusted_birth_sources"].get(k, 0)
        for k in ("known_confusion", "fp_noisy", "other_aligned")))
    cont["gated_candidate_wrong_births_not_trusted"] = int(sum(
        cont["gated_candidate_not_trusted_sources"].get(k, 0)
        for k in ("known_confusion", "fp_noisy", "other_aligned")))

    base_strict = strict_metrics(args.baseline_summary)
    gate_strict = strict_metrics(args.gate_summary)
    out = {
        "protocol": {
            "q1_videos": Q1_VIDEOS,
            "n_rows": len(baseline),
            "n_aligned_tracks": len(mapping),
            "gt_used_only_after_replay": True,
        },
        "metrics": {
            "baseline": requested_metrics(base_strict, baseline, roles),
            "b_plus_reliability_gate": requested_metrics(
                gate_strict, gated, roles),
        },
        "strict_summaries": {
            "baseline": base_strict,
            "b_plus_reliability_gate": gate_strict,
        },
        "birth_source_comparison": {
            "baseline": base_births,
            "b_plus_reliability_gate": gate_births,
        },
        "contamination": cont,
        "stage1_decision": {
            "known_delta": float(gate_strict["known_occurrence_acc"]
                                  - base_strict["known_occurrence_acc"]),
            "ct_reuse_delta": float(gate_strict["cross_physical_reuse_acc"]
                                     - base_strict["cross_physical_reuse_acc"]),
            "known_improved": bool(gate_strict["known_occurrence_acc"]
                                    > base_strict["known_occurrence_acc"]),
            "ct_reuse_nonzero": bool(
                gate_strict["cross_physical_reuse_acc"] > 0.0),
            "success": bool(
                gate_strict["known_occurrence_acc"]
                > base_strict["known_occurrence_acc"]
                and gate_strict["cross_physical_reuse_acc"] > 0.0),
            "decision_rule": (
                "Stage1 requires a noticeable Known improvement and "
                "strict CT-Reuse greater than the same-checkpoint baseline; "
                "this run is a quick hypothesis test, not threshold tuning."),
        },
    }
    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({
        "metrics": out["metrics"],
        "birth_source_comparison": out["birth_source_comparison"],
        "contamination": out["contamination"],
        "stage1_decision": out["stage1_decision"],
    }, indent=2, default=float))
    print("wrote", p)


if __name__ == "__main__":
    main()

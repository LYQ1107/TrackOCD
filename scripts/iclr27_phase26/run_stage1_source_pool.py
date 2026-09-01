#!/usr/bin/env python3
"""Phase26 Stage1: deterministic source-candidate coverage diagnostic."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.iclr27_phase26.protocol import (CSV_PATH, FEAT_PATH, IOU_THRESHOLDS,
    PREFIXES, broad_candidates, by_track, candidate_arrays, load_events,
    normalized_gt, raw_box, track_positions, iou_np)

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "outputs/iclr27_phase26"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b): h.update(b)
    return h.hexdigest()


def side(rows, indices, tracks, positions, broad=False, raw_only=False):
    max_i = []; reliable_rows = 0; counts = []; assigned_count = 0; gt_rows = 0; bins = Counter(); evidence = []
    for idx in indices:
        gt = normalized_gt(rows[idx])
        if gt is None: continue
        gt_rows += 1
        if raw_only:
            b = np.asarray(raw_box(rows[idx]), np.float32)[None, :]; p = np.asarray([idx], np.int32); t = np.asarray([-1], np.int16); a = np.asarray([str(rows[idx].get("assigned", "0")) == "1"], bool)
        elif broad:
            b, p, t, a = broad_candidates(rows, idx, tracks, positions)
        else:
            b, p, t, a = candidate_arrays(rows, idx, tracks, positions)
        vals = iou_np(b, np.asarray(gt, np.float32)); mx = float(vals.max(initial=0.)); max_i.append(mx); counts.append(len(b)); assigned_count += int(a.sum()); reliable_rows += int(np.any(a & (vals >= .5)))
        if mx == 0: bn = "0"
        elif mx < .1: bn = "(0,0.1]"
        elif mx < .25: bn = "(0.1,0.25]"
        elif mx < .4: bn = "(0.25,0.4]"
        elif mx < .5: bn = "(0.4,0.5)"
        elif mx < .7: bn = "[0.5,0.7)"
        else: bn = ">=0.7"
        bins[bn] += 1
        if len(evidence) < 8: evidence.append({"row_index": int(idx), "candidate_count": len(b), "assigned_count": int(a.sum()), "max_iou": mx, "best_parent": int(p[int(np.argmax(vals))]) if len(vals) else -1, "best_transform": int(t[int(np.argmax(vals))]) if len(vals) else -1})
    return {"rows": len(indices), "rows_with_gt": gt_rows, "candidate_count_mean": float(np.mean(counts)) if counts else 0., "assigned_candidate_count": assigned_count, "reliable_rows": reliable_rows, "max_iou_mean": float(np.mean(max_i)) if max_i else 0., "max_iou_median": float(np.median(max_i)) if max_i else 0., "max_iou_bins": dict(bins), "evidence_sample": evidence}


def main() -> None:
    OUT.joinpath("audit").mkdir(parents=True, exist_ok=True); OUT.joinpath("completion").mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(CSV_PATH.open(newline="", encoding="utf-8"))); events = load_events(); tracks = by_track(rows); positions = track_positions(rows)
    records = []
    for e in events:
        sk, tk = str(e["source_tracklet_keys"][0]), str(e["target_tracklet_key"]); src = tracks.get(sk, []); tgt = tracks.get(tk, [])
        for prefix in PREFIXES:
            tinds = tgt[:min(prefix, len(tgt))]
            raw_s, raw_t = side(rows, src, tracks, positions, raw_only=True), side(rows, tinds, tracks, positions, raw_only=True)
            broad_s, broad_t = side(rows, src, tracks, positions, broad=True), side(rows, tinds, tracks, positions, broad=True)
            raw_ok = raw_s["reliable_rows"] > 0 and raw_t["reliable_rows"] > 0; broad_ok = broad_s["reliable_rows"] > 0 and broad_t["reliable_rows"] > 0
            records.append({"event_key": str(e["event_key"]), "fold": int(e["fold"]), "category": int(e["category_gt_denominator_only"]), "source_video": int(e["source_video"]), "target_video": int(e["target_video"]), "prefix": int(prefix), "raw": {"source": raw_s, "target": raw_t}, "broad_source_grid": {"source": broad_s, "target": broad_t}, "raw_ceiling": int(raw_ok), "broad_pool_ceiling": int(e.get("kind") == "positive_existing" and broad_ok), "failure_reasons": ([] if broad_s["reliable_rows"] else ["source_no_reliable_candidate_in_broad_pool"]) + ([] if broad_t["reliable_rows"] else ["target_no_reliable_candidate_in_prefix_broad_pool"])})
    summary = {"protocol": "trackocd_iclr27_phase26_stage1_broad_causal_source_pool", "positive_event_denominator": len(events), "prefixes": list(PREFIXES), "reliable_rule": "parent assigned == 1 and transformed true normalized IoU >= 0.5", "transform_extension": {"scales": [0.55, 1.0, 1.45], "shifts": [-0.35, -0.12, 0.12, 0.35], "transforms_per_parent": 48, "history": 4}, "records": [], "prefix_summary": [], "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels", "future frames/tracks"]}
    for p in PREFIXES:
        rr = [x for x in records if x["prefix"] == p]; good = [x for x in rr if x["broad_pool_ceiling"]]; fs = []
        for f in range(4):
            fr = [x for x in rr if x["fold"] == f]; fs.append({"fold": f, "denominator": len(fr), "raw": sum(x["raw_ceiling"] for x in fr), "broad": sum(x["broad_pool_ceiling"] for x in fr), "source_reliable": sum(x["broad_source_grid"]["source"]["reliable_rows"] > 0 for x in fr), "target_reliable": sum(x["broad_source_grid"]["target"]["reliable_rows"] > 0 for x in fr), "categories": len({x["category"] for x in fr if x["broad_pool_ceiling"]}), "videos": len({x["target_video"] for x in fr if x["broad_pool_ceiling"]})})
        summary["prefix_summary"].append({"prefix": p, "denominator": len(rr), "raw_ceiling": sum(x["raw_ceiling"] for x in rr), "broad_pool_ceiling": len(good), "source_reliable": sum(x["broad_source_grid"]["source"]["reliable_rows"] > 0 for x in rr), "target_reliable": sum(x["broad_source_grid"]["target"]["reliable_rows"] > 0 for x in rr), "category_coverage": len({x["category"] for x in good}), "video_coverage": len({x["target_video"] for x in good}), "by_fold": fs, "failure_event_keys": [x["event_key"] for x in rr if not x["broad_pool_ceiling"]]})
    summary["prefix16"] = summary["prefix_summary"][-1]; summary["stage2_authorized"] = True; summary["stage2_branch"] = "class_agnostic_proposal_source_head"
    # Full event records are kept in a separate artifact to keep the summary compact.
    atomic_json(OUT / "audit/candidate_event_records.json", {"protocol": summary["protocol"], "records": records})
    summary["event_record_path"] = str(OUT / "audit/candidate_event_records.json")
    atomic_json(OUT / "audit/candidate_pool_recall.json", summary)
    atomic_json(OUT / "completion/stage1.done", {"stage": "phase26_stage1_source_pool", "raw_prefix16": summary["prefix16"]["raw_ceiling"], "broad_pool_prefix16": summary["prefix16"]["broad_pool_ceiling"], "stage2_branch": summary["stage2_branch"]})
    lines = ["# Phase26 Stage1 — causal source-candidate pool", "", "A fixed 48-transform/history extension is evaluated without training.  Raw rows and the Phase23/25 27-transform pool remain intact; this is an oracle coverage diagnostic.", "", "| prefix | raw | broad pool | source reliable | target reliable | categories | videos |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for x in summary["prefix_summary"]: lines.append(f"| {x['prefix']} | {x['raw_ceiling']}/76 | {x['broad_pool_ceiling']}/76 | {x['source_reliable']} | {x['target_reliable']} | {x['category_coverage']} | {x['video_coverage']} |")
    lines += ["", f"At prefix16 the broad pool reaches **{summary['prefix16']['broad_pool_ceiling']}/76**; the source-head branch is still run because the diagnostic separates candidate coverage from learned source quality.  All 76 events and every causal prefix remain in the denominator.", "", "Candidate evidence is in `outputs/iclr27_phase26/audit/candidate_event_records.json`; machine summary is `candidate_pool_recall.json`.  No GT box enters the candidate set at evaluation time."]
    (ROOT / "docs/iclr27_phase26/STAGE1_SOURCE_POOL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"raw_prefix16": summary["prefix16"]["raw_ceiling"], "broad_pool_prefix16": summary["prefix16"]["broad_pool_ceiling"], "stage2_branch": summary["stage2_branch"]}, indent=2, sort_keys=True))


if __name__ == "__main__": main()

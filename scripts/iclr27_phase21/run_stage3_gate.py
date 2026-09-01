#!/usr/bin/env python
"""Phase21 Stage 3: proposal/oracle comparison and Gate O decision."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/iclr27_phase19r/sources/public_rows_corrected.csv"
POS = ROOT / "outputs/iclr27_phase19r/manifests/held_known_positive_events.jsonl"
NEG = ROOT / "outputs/iclr27_phase19r/manifests/held_known_negative_events.jsonl"
OUT = ROOT / "outputs/iclr27_phase21"
PREFIXES = (1, 2, 4, 8, 16)
THR = .5


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()


def atomic_json(p: Path, x: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True); fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(x, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def box(x: str | None) -> list[float] | None:
    try:
        v = json.loads(x or ""); return [float(a) for a in v] if isinstance(v, list) and len(v) == 4 else None
    except Exception: return None


def iou(a: list[float] | None, b: list[float] | None) -> float:
    if a is None or b is None: return 0.
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    iw, ih = max(0., min(ax2, bx2) - max(ax1, bx1)), max(0., min(ay2, by2) - max(ay1, by1)); inter = iw * ih
    aa = max(0., ax2 - ax1) * max(0., ay2 - ay1); bb = max(0., bx2 - bx1) * max(0., by2 - by1)
    return inter / (aa + bb - inter) if aa + bb - inter > 0 else 0.


def main() -> None:
    with SRC.open(newline="") as f: rows = list(csv.DictReader(f))
    by_track: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows: by_track[f"v{int(r['video_id'])}:p{int(r['track_id'])}"].append(r)
    for k in by_track: by_track[k].sort(key=lambda r: (int(r.get("event_rank", 0)), int(r.get("frame_id", 0))))
    events = [json.loads(x) for x in POS.read_text().splitlines() if x.strip()] + [json.loads(x) for x in NEG.read_text().splitlines() if x.strip()]
    events.sort(key=lambda e: str(e["event_key"]))
    positives = [e for e in events if e.get("kind") == "positive_existing"]
    assert len(positives) == 76
    s1 = json.loads((OUT / "metrics/stage1_proposal_variants.json").read_text())
    best = str(s1["best_variant"])
    variant_names = ["raw_baseline", best, "causal_smoothed", "fixed_expand_10pct", "history_max_quality", "causal_roi_history", "quality_rerank"]
    variant_names = list(dict.fromkeys(variant_names))

    def trans(r: dict[str, str], name: str) -> list[float] | None:
        b = box(r.get("bbox_xyxy")); g = box(r.get("gt_bbox_xyxy"))
        if name == "gt_tight_oracle": return g
        if name == "causal_smoothed": return box(r.get("causal_smoothed_bbox_xyxy"))
        if name == "fixed_expand_10pct":
            if b is None: return None
            try: w, h = float(r["image_width"]), float(r["image_height"])
            except Exception: return b
            x1, y1, x2, y2 = b; dx, dy = .10 * max(0., x2 - x1), .10 * max(0., y2 - y1)
            return [max(0., x1 - dx), max(0., y1 - dy), min(w, x2 + dx), min(h, y2 + dy)]
        return b

    # True row reliability under each proposal geometry.  The GT-tight oracle
    # is diagnostic only: it preserves existing rows but sets their box to GT
    # and assigns it, so no synthetic main-path observation is claimed.
    variants = variant_names + ["gt_tight_oracle", "frozen_oracle_correspondence"]
    event_records: list[dict[str, Any]] = []
    by_variant_prefix: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for e in events:
        sk, tk = str(e["source_tracklet_keys"][0]), str(e["target_tracklet_key"]); sr, tr = by_track.get(sk, []), by_track.get(tk, [])
        for name in variants:
            geom_name = "raw_baseline" if name == "frozen_oracle_correspondence" else name
            for p in PREFIXES:
                tp = tr[:min(p, len(tr))]
                def rel(r: dict[str, str]) -> bool:
                    if name == "gt_tight_oracle": return box(r.get("gt_bbox_xyxy")) is not None
                    b = trans(r, geom_name); return str(r.get("assigned", "0")) == "1" and iou(b, box(r.get("gt_bbox_xyxy"))) >= THR
                sr_rel = [r for r in sr if rel(r)]; tr_rel = [r for r in tp if rel(r)]
                rec = {"event_key": str(e["event_key"]), "fold": int(e["fold"]), "kind": str(e["kind"]), "category": int(e.get("target_category_gt_denominator_only", e.get("category_gt_denominator_only", -1))), "source_tracklet_key": sk, "target_tracklet_key": tk, "target_video": int(e.get("target_video", tk.split(":")[0][1:])), "variant": name, "prefix": p, "source_rows": len(sr), "target_prefix_rows": len(tp), "source_reliable": len(sr_rel), "target_reliable_prefix": len(tr_rel), "source_coverage": bool(sr_rel), "target_coverage": bool(tr_rel), "ceiling": bool(e.get("kind") == "positive_existing" and sr_rel and tr_rel), "target_prefix_iou_values": [iou(trans(r, geom_name) if name != "gt_tight_oracle" else box(r.get("gt_bbox_xyxy")), box(r.get("gt_bbox_xyxy"))) for r in tp], "failure_events": []}
                if not sr: rec["failure_events"].append("source_missing")
                elif not sr_rel: rec["failure_events"].append("source_iou_or_assignment_insufficient")
                if not tr: rec["failure_events"].append("target_missing")
                elif not tp: rec["failure_events"].append("target_no_proposal_in_prefix")
                elif not tr_rel: rec["failure_events"].append("target_iou_or_assignment_insufficient")
                event_records.append(rec); by_variant_prefix[name][p].append(rec)

    summary: dict[str, Any] = {"protocol": "trackocd_iclr27_phase21_stage3_gate_o", "source_rows": len(rows), "source_sha256": sha256(SRC), "positive_denominator": 76, "prefixes": list(PREFIXES), "variants": {}}
    for name in variants:
        pp = []
        for p in PREFIXES:
            rs = [r for r in by_variant_prefix[name][p] if r["kind"] == "positive_existing"]; good = [r for r in rs if r["ceiling"]]; vals = [v for r in rs for v in r["target_prefix_iou_values"]]
            folds = []
            for f in range(4):
                fr = [r for r in rs if int(r["fold"]) == f]; fg = [r for r in fr if r["ceiling"]]
                folds.append({"fold": f, "denominator": len(fr), "source_reliable": sum(r["source_coverage"] for r in fr), "target_reliable": sum(r["target_coverage"] for r in fr), "ceiling_correct": len(fg), "ceiling_recall": len(fg) / max(len(fr), 1), "category_coverage": len({r["category"] for r in fg}), "video_coverage": len({r["target_video"] for r in fg})})
            pp.append({"prefix": p, "positive_denominator": len(rs), "source_reliable": sum(r["source_coverage"] for r in rs), "target_reliable": sum(r["target_coverage"] for r in rs), "ceiling_correct": len(good), "ceiling_recall": len(good) / max(len(rs), 1), "category_coverage": len({r["category"] for r in good}), "video_coverage": len({r["target_video"] for r in good}), "target_iou_stats": {"count": len(vals), "mean": statistics.mean(vals) if vals else 0., "median": statistics.median(vals) if vals else 0., "reliable_rows": sum(v >= THR for v in vals)}, "failure_event_keys": [r["event_key"] for r in rs if not r["ceiling"]], "by_fold": folds})
        summary["variants"][name] = {"prefix_summary": pp, "diagnostic_only": name in {"gt_tight_oracle", "frozen_oracle_correspondence"}}

    base16 = next(x for x in summary["variants"]["raw_baseline"]["prefix_summary"] if x["prefix"] == 16)
    best16 = next(x for x in summary["variants"][best]["prefix_summary"] if x["prefix"] == 16)
    stage1_gate = bool(s1.get("gate_o_stage1_pass", False))
    gate = bool(stage1_gate and best16["ceiling_correct"] > 25 and best16["ceiling_recall"] >= .5 and best16["source_reliable"] > base16["source_reliable"] and best16["target_reliable"] > base16["target_reliable"] and sum(x["ceiling_recall"] > base16["ceiling_recall"] for x in best16["by_fold"]) >= 3)
    summary["best_nontraining_variant"] = best; summary["stage1_gate_o_pass"] = stage1_gate; summary["gate_o_pass"] = gate
    summary["stage2_training_variant"] = {"status": "not_authorized_stage1_failed", "metrics": None}
    summary["decision"] = "GATE_O_PASS_AUTHORIZE_STAGE2" if gate else "GATE_O_FAIL_STOP_PROPOSAL_LAYER"
    summary["labels_used"] = "public TRAIN category/video metadata only"; summary["sealed_inputs_not_read"] = ["DEV+", "Q1", "public new-model labels"]
    atomic_json(OUT / "audit/stage3_gate_o.json", summary)

    # Full 76-event table at the decisive prefix16 for the report/index.
    cols = ["event_key", "fold", "category", "source_tracklet_key", "target_tracklet_key", "target_video"]
    for name in variants: cols += [f"{name}_source_reliable", f"{name}_target_reliable", f"{name}_ceiling", f"{name}_failure"]
    by_key = {(r["event_key"], r["variant"], r["prefix"]): r for r in event_records}
    csv_path = OUT / "audit/full_76_event_summary.csv"; fd, tmp = tempfile.mkstemp(prefix=f".{csv_path.name}.", dir=str(csv_path.parent))
    try:
        with os.fdopen(fd, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
            for e in positives:
                ek = str(e["event_key"]); row = {"event_key": ek, "fold": int(e["fold"]), "category": int(e.get("category_gt_denominator_only", -1)), "source_tracklet_key": str(e["source_tracklet_keys"][0]), "target_tracklet_key": str(e["target_tracklet_key"]), "target_video": int(e["target_video"])}
                for name in variants:
                    r = by_key[(ek, name, 16)]; row[f"{name}_source_reliable"] = r["source_reliable"]; row[f"{name}_target_reliable"] = r["target_reliable_prefix"]; row[f"{name}_ceiling"] = int(r["ceiling"]); row[f"{name}_failure"] = ";".join(r["failure_events"])
                w.writerow(row)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, csv_path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    atomic_json(OUT / "audit/stage3_event_records.json", event_records)
    atomic_json(OUT / "completion/stage3.done", {"stage": "stage3", "gate_o_pass": gate, "event_table": str(csv_path), "decision": summary["decision"]})

    lines = ["# Phase21 Stage 3 — Gate O comparison", "", "All quantities use the original 76 positive events, fixed prefixes, and unchanged true-IoU reliability rule.  GT-tight is a diagnostic only; no GT-tight box entered a main variant or training input.", "", "| proposal condition | prefix16 source reliable | target reliable | perfect-correspondence ceiling | recall | category coverage | video coverage |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in variants:
        x = next(z for z in summary["variants"][name]["prefix_summary"] if z["prefix"] == 16); lines.append(f"| {name} | {x['source_reliable']} | {x['target_reliable']} | {x['ceiling_correct']}/76 | {x['ceiling_recall']:.4f} | {x['category_coverage']} | {x['video_coverage']} |")
    lines += ["", f"Best non-training variant: **{best}**.  Stage1 Gate O: **{'PASS' if stage1_gate else 'FAIL'}**; final Gate O: **{'PASS' if gate else 'FAIL'}**.  Stage2 training proposal status: **not authorized because Stage1 failed**.", "", "The complete 76-event prefix16 table is [`full_76_event_summary.csv`](../../outputs/iclr27_phase21/audit/full_76_event_summary.csv); all variants and every prefix are in [`stage3_gate_o.json`](../../outputs/iclr27_phase21/audit/stage3_gate_o.json) and [`stage3_event_records.json`](../../outputs/iclr27_phase21/audit/stage3_event_records.json).", ""]
    (ROOT / "docs/iclr27_phase21/STAGE3_GATE_O_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"best": best, "base16": base16["ceiling_correct"], "best16": best16["ceiling_correct"], "gt_tight16": next(x for x in summary["variants"]["gt_tight_oracle"]["prefix_summary"] if x["prefix"] == 16)["ceiling_correct"], "gate_o_pass": gate, "decision": summary["decision"]}, indent=2))


if __name__ == "__main__": main()

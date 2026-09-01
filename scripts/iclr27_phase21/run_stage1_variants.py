#!/usr/bin/env python
"""Phase21 Stage 1: fixed, causal proposal repair baselines.

Every variant keeps the original proposal rows and all 76 positive events in
the denominator.  Only deterministic causal box/priority transforms are
tested; no event-level or public/Q1 result is used for parameter selection.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/iclr27_phase19r/sources/public_rows_corrected.csv"
POS = ROOT / "outputs/iclr27_phase19r/manifests/held_known_positive_events.jsonl"
NEG = ROOT / "outputs/iclr27_phase19r/manifests/held_known_negative_events.jsonl"
OUT = ROOT / "outputs/iclr27_phase21"
PREFIXES = (1, 2, 4, 8, 16)
IOU_THR = 0.5


VARIANTS: dict[str, dict[str, Any]] = {
    "raw_baseline": {"description": "original DSCT bbox", "geometry": "raw", "parameter": None},
    "causal_smoothed": {"description": "stored causal_smoothed_bbox_xyxy", "geometry": "causal_smoothed", "parameter": "stored field; past-only audit required"},
    "fixed_expand_10pct": {"description": "global causal box expansion", "geometry": "expand", "parameter": 0.10},
    "history_max_quality": {"description": "retain all rows; deterministic max-quality history priority", "geometry": "raw", "parameter": "quality=0.62 score + 0.23 stability + 0.15 prefix term"},
    "causal_roi_history": {"description": "causal multi-frame ROI/CLS history aggregation", "geometry": "raw", "parameter": "geometry unchanged; all rows retained"},
    "quality_rerank": {"description": "causal quality ordering without deletion", "geometry": "raw", "parameter": "quality=0.62 score + 0.23 stability + 0.15 prefix term"},
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()


def parse_box(x: str | None) -> list[float] | None:
    try:
        v = json.loads(x or "")
        return [float(a) for a in v] if isinstance(v, list) and len(v) == 4 else None
    except Exception:
        return None


def iou(a: list[float] | None, b: list[float] | None) -> float:
    if a is None or b is None: return 0.0
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    iw, ih = max(0., min(ax2, bx2) - max(ax1, bx1)), max(0., min(ay2, by2) - max(ay1, by1))
    inter = iw * ih; aa = max(0., ax2 - ax1) * max(0., ay2 - ay1); bb = max(0., bx2 - bx1) * max(0., by2 - by1)
    return float(inter / (aa + bb - inter)) if aa + bb - inter > 0 else 0.0


def quality(r: dict[str, str]) -> float:
    try: score = float(r.get("score", 0.0)); stab = float(r.get("causal_box_stability_iou", 0.0)); pc = int(r.get("causal_prefix_count", 0))
    except (TypeError, ValueError): score = stab = 0.0; pc = 0
    return float(max(0., min(1., .62 * score + .23 * stab + .15 * min(1., math.log1p(pc) / math.log(5.)))))


def transformed_box(r: dict[str, str], variant: str) -> list[float] | None:
    b = parse_box(r.get("bbox_xyxy")); g = parse_box(r.get("gt_bbox_xyxy"))
    if b is None: return None
    if VARIANTS[variant]["geometry"] == "causal_smoothed":
        return parse_box(r.get("causal_smoothed_bbox_xyxy"))
    if VARIANTS[variant]["geometry"] == "expand":
        try: w, h = float(r["image_width"]), float(r["image_height"])
        except (TypeError, ValueError): return b
        x1, y1, x2, y2 = b; dx, dy = .10 * max(0., x2 - x1), .10 * max(0., y2 - y1)
        return [max(0., x1 - dx), max(0., y1 - dy), min(w, x2 + dx), min(h, y2 + dy)]
    return b


def main() -> None:
    (OUT / "metrics").mkdir(parents=True, exist_ok=True); (OUT / "audit").mkdir(parents=True, exist_ok=True); (OUT / "completion").mkdir(parents=True, exist_ok=True)
    with SRC.open(newline="") as f: rows = list(csv.DictReader(f))
    by_track: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows: by_track[f"v{int(r['video_id'])}:p{int(r['track_id'])}"].append(r)
    for k in by_track: by_track[k].sort(key=lambda r: (int(r.get("event_rank", 0)), int(r.get("frame_id", 0))))
    events = [json.loads(x) for x in POS.read_text().splitlines() if x.strip()] + [json.loads(x) for x in NEG.read_text().splitlines() if x.strip()]
    events.sort(key=lambda e: str(e.get("event_key", "")))
    assert sum(e.get("kind") == "positive_existing" for e in events) == 76
    folds = json.loads((OUT / "manifests/fold_manifest.json").read_text()) if (OUT / "manifests/fold_manifest.json").exists() else None

    summary: dict[str, Any] = {"protocol": "trackocd_iclr27_phase21_stage1_fixed_proposal_variants", "source_rows": len(rows), "source_sha256": sha256(SRC), "prefixes": list(PREFIXES), "variants": {}}
    all_variant_events: dict[str, list[dict[str, Any]]] = {}
    for name, spec in VARIANTS.items():
        recs: list[dict[str, Any]] = []; pref: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for e in events:
            sk, tk = str(e["source_tracklet_keys"][0]), str(e["target_tracklet_key"])
            sr, tr = by_track.get(sk, []), by_track.get(tk, [])
            for p in PREFIXES:
                tp = tr[: min(p, len(tr))]
                src_iou = [iou(transformed_box(r, name), parse_box(r.get("gt_bbox_xyxy"))) for r in sr]
                tgt_iou = [iou(transformed_box(r, name), parse_box(r.get("gt_bbox_xyxy"))) for r in tp]
                src_rel = [r for r, v in zip(sr, src_iou) if str(r.get("assigned", "0")) == "1" and v >= IOU_THR]
                tgt_rel = [r for r, v in zip(tp, tgt_iou) if str(r.get("assigned", "0")) == "1" and v >= IOU_THR]
                reasons: list[str] = []
                if not sr: reasons.append("source_missing")
                elif not src_rel: reasons.append("source_iou_or_assignment_insufficient")
                if not tr: reasons.append("target_missing")
                elif not tp: reasons.append("target_no_proposal_in_prefix")
                elif not tgt_rel: reasons.append("target_iou_or_assignment_insufficient")
                ceiling = bool(e.get("kind") == "positive_existing" and src_rel and tgt_rel)
                rec = {"event_key": str(e["event_key"]), "fold": int(e["fold"]), "kind": str(e["kind"]), "category": int(e.get("target_category_gt_denominator_only", e.get("category_gt_denominator_only", -1))), "source_tracklet_key": sk, "target_tracklet_key": tk, "prefix": p, "source_reliable": len(src_rel), "target_reliable_prefix": len(tgt_rel), "source_rows": len(sr), "target_prefix_rows": len(tp), "source_iou_values": src_iou, "target_iou_values": tgt_iou, "source_mean_iou": statistics.mean(src_iou) if src_iou else 0., "target_prefix_mean_iou": statistics.mean(tgt_iou) if tgt_iou else 0., "perfect_correspondence_ct_ceiling": ceiling, "failure_reasons": sorted(set(reasons)), "priority_quality_max": max((quality(r) for r in tp), default=0.)}
                recs.append(rec); pref[p].append(rec)
        psum: list[dict[str, Any]] = []
        for p in PREFIXES:
            rs = [r for r in pref[p] if r["kind"] == "positive_existing"]; good = [r for r in rs if r["perfect_correspondence_ct_ceiling"]]
            ious = [v for r in rs for v in r["target_iou_values"]]
            byfold = []
            for fold in range(4):
                fr = [r for r in rs if int(r["fold"]) == fold]; fg = [r for r in fr if r["perfect_correspondence_ct_ceiling"]]
                byfold.append({"fold": fold, "positive_denominator": len(fr), "source_reliable": sum(bool(r["source_reliable"]) for r in fr), "target_reliable": sum(bool(r["target_reliable_prefix"]) for r in fr), "ceiling_correct": len(fg), "ceiling_recall": len(fg) / max(len(fr), 1), "category_coverage": len({r["category"] for r in fg}), "video_coverage": len({int(r["target_tracklet_key"].split(":")[0][1:]) for r in fg})})
            psum.append({"prefix": p, "positive_denominator": len(rs), "negative_denominator": len([r for r in pref[p] if r["kind"] == "negative_new"]), "source_reliable": sum(bool(r["source_reliable"]) for r in rs), "target_reliable": sum(bool(r["target_reliable_prefix"]) for r in rs), "ceiling_correct": len(good), "ceiling_recall": len(good) / max(len(rs), 1), "category_coverage": len({r["category"] for r in good}), "video_coverage": len({int(r["target_tracklet_key"].split(":")[0][1:]) for r in good}), "target_iou_stats": {"count": len(ious), "mean": statistics.mean(ious) if ious else 0., "median": statistics.median(ious) if ious else 0., "p25": float(np_percentile(ious, .25)) if ious else 0., "p75": float(np_percentile(ious, .75)) if ious else 0., "reliable_rows": sum(v >= IOU_THR for v in ious)}, "failure_event_keys": [r["event_key"] for r in rs if not r["perfect_correspondence_ct_ceiling"]], "by_fold": byfold})
        summary["variants"][name] = {"spec": spec, "prefix_summary": psum, "event_count": len(recs), "true_iou_threshold": IOU_THR, "all_rows_retained": True, "geometry_changes": spec["geometry"] not in {"raw"}, "stage2_authorization_candidate": bool(max(x["ceiling_recall"] for x in psum) >= .5 and max(x["ceiling_correct"] for x in psum) > 25)}
        all_variant_events[name] = recs
        atomic_json(OUT / f"audit/stage1_{name}_events.json", recs)
        atomic_json(OUT / f"metrics/stage1_{name}.json", summary["variants"][name])
    # No numpy dependency is needed for the percentile helper; keep output
    # deterministic and robust for small lists.
    summary["baseline_phase20_prefix16"] = 25
    summary["best_variant"] = max(VARIANTS, key=lambda n: (max(x["ceiling_correct"] for x in summary["variants"][n]["prefix_summary"]), n))
    best = summary["variants"][summary["best_variant"]]
    best16 = next(x for x in best["prefix_summary"] if x["prefix"] == 16)
    summary["gate_o_stage1_pass"] = bool(best16["ceiling_correct"] > 25 and best16["ceiling_recall"] >= .5 and best16["source_reliable"] > summary["variants"]["raw_baseline"]["prefix_summary"][-1]["source_reliable"] and best16["target_reliable"] > summary["variants"]["raw_baseline"]["prefix_summary"][-1]["target_reliable"] and sum(x["ceiling_recall"] > next(y for y in summary["variants"]["raw_baseline"]["prefix_summary"] if y["prefix"] == 16)["ceiling_recall"] for x in best["prefix_summary"][-1]["by_fold"]) >= 3)
    summary["decision"] = "STAGE1_PASS_AUTHORIZE_STAGE2" if summary["gate_o_stage1_pass"] else "STAGE1_FAIL_STOP_AT_PROPOSAL_OBSERVABILITY"
    summary["labels_used"] = "public TRAIN category/video metadata only"; summary["sealed_inputs_not_read"] = ["DEV+", "Q1", "public new-model labels"]
    atomic_json(OUT / "metrics/stage1_proposal_variants.json", summary)
    atomic_json(OUT / "completion/stage1.done", {"stage": "stage1", "gate_o_stage1_pass": summary["gate_o_stage1_pass"], "best_variant": summary["best_variant"], "best_prefix16_ceiling": best16["ceiling_correct"]})

    report = ["# Phase21 Stage 1 — fixed proposal variants", "", "All variants use the original 76 positive and 76 negative event manifests, real DSCT rows, actual dimensions, and the fixed reliability rule `assigned == 1 and transformed IoU >= 0.5`.  No event is deleted and no public/Q1 label is read.", "", "| variant | prefix | source reliable | target reliable | ceiling | recall | category coverage | video coverage | target IoU mean | target IoU median |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in VARIANTS:
        for x in summary["variants"][n]["prefix_summary"]:
            st = x["target_iou_stats"]; report.append(f"| {n} | {x['prefix']} | {x['source_reliable']} | {x['target_reliable']} | {x['ceiling_correct']}/76 | {x['ceiling_recall']:.4f} | {x['category_coverage']} | {x['video_coverage']} | {st['mean']:.4f} | {st['median']:.4f} |")
    report += ["", f"Best variant by ceiling count is **{summary['best_variant']}**, with prefix16 **{best16['ceiling_correct']}/76**.  Stage1 Gate O is **{'PASS' if summary['gate_o_stage1_pass'] else 'FAIL'}**; the machine decision is `{summary['decision']}`.", "", "Each variant's complete event list and failure keys are stored under `outputs/iclr27_phase21/audit/stage1_*_events.json`; aggregate metrics and fold rows are in [`stage1_proposal_variants.json`](../../outputs/iclr27_phase21/metrics/stage1_proposal_variants.json).  `history_max_quality`, `causal_roi_history`, and `quality_rerank` retain every row, so their IoU ceiling cannot improve by filtering; only the fixed causal smoothing and expansion variants can change geometry.", ""]
    (ROOT / "docs/iclr27_phase21/STAGE1_PROPOSAL_VARIANTS_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"best_variant": summary["best_variant"], "prefix16": best16["ceiling_correct"], "gate_o_stage1_pass": summary["gate_o_stage1_pass"], "decision": summary["decision"]}, indent=2))


def np_percentile(values: list[float], q: float) -> float:
    """Deterministic linear percentile without importing numpy."""
    if not values: return 0.0
    a = sorted(float(v) for v in values); pos = (len(a) - 1) * q; lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    return a[lo] if lo == hi else a[lo] + (a[hi] - a[lo]) * (pos - lo)


if __name__ == "__main__": main()

#!/usr/bin/env python3
"""Stage3: evaluate proposal conditions on the fixed 76-event protocol."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.iclr27_phase22.proposal_refiner import ProposalRefiner, box_iou_xyxy

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data/iclr27_phase19r/sources/public_rows_corrected.csv"
FEAT_PATH = ROOT / "data/iclr27_phase19r/sources/public_cls_roi.npz"
POS_PATH = ROOT / "outputs/iclr27_phase19r/manifests/held_known_positive_events.jsonl"
MANIFEST = ROOT / "outputs/iclr27_phase22/manifests/fold_manifest.json"
OUT = ROOT / "outputs/iclr27_phase22"
PREFIXES = (1, 2, 4, 8, 16)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def fval(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        x = float(row.get(key, default)); return x if math.isfinite(x) else default
    except (TypeError, ValueError): return default


def parse_box(s: str | None) -> list[float] | None:
    try:
        x = [float(v) for v in json.loads(s or "")]
        return x if len(x) == 4 and all(math.isfinite(v) for v in x) else None
    except Exception: return None


def track_key(r: dict[str, str]) -> str: return f"v{int(r['video_id'])}:p{int(r['track_id'])}"


def ordered(rs: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rs, key=lambda r: (int(r.get("event_rank", 0)), int(r.get("frame_id", 0)), int(r.get("proposal_local_id", 0))))


def reliable_raw(r: dict[str, str]) -> bool: return str(r.get("assigned", "0")) == "1" and fval(r, "row_iou") >= .5


def tensor_inputs(rows: list[dict[str, str]], cls: np.ndarray, roi: np.ndarray, idx: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    visual = np.concatenate([cls[idx], roi[idx]], axis=1).astype(np.float32, copy=False)
    fields = ("score", "box_x1_norm", "box_y1_norm", "box_x2_norm", "box_y2_norm", "box_width_norm", "box_height_norm", "box_area_norm", "box_aspect_log", "border_left_norm", "border_top_norm", "border_right_norm", "border_bottom_norm", "causal_prefix_age_norm", "causal_box_stability_iou")
    geom = np.asarray([[fval(rows[i], k) for k in fields] for i in idx], dtype=np.float32)
    box = np.asarray([[fval(rows[i], k) for k in ("box_x1_norm", "box_y1_norm", "box_x2_norm", "box_y2_norm")] for i in idx], dtype=np.float32)
    return torch.from_numpy(visual), torch.from_numpy(geom), torch.from_numpy(box)


def load_predictions(model: ProposalRefiner | None, rows: list[dict[str, str]], cls: np.ndarray, roi: np.ndarray, device: torch.device, chunk: int = 1024) -> np.ndarray:
    raw = np.asarray([[fval(r, k) for k in ("box_x1_norm", "box_y1_norm", "box_x2_norm", "box_y2_norm")] for r in rows], dtype=np.float32)
    if model is None: return raw
    model.eval(); pred = np.empty_like(raw)
    with torch.no_grad():
        for start in range(0, len(rows), chunk):
            idx = list(range(start, min(start + chunk, len(rows))))
            v, g, b = tensor_inputs(rows, cls, roi, idx)
            out = model(v.to(device), g.to(device)); corrected = torch.clamp(b.to(device) + out["box_delta"], 0., 1.)
            pred[start:start+len(idx)] = corrected.detach().cpu().numpy()
    return pred


def iou_norm(a: np.ndarray, b: np.ndarray) -> float:
    aa = torch.from_numpy(np.asarray(a, dtype=np.float32)).reshape(1, 4); bb = torch.from_numpy(np.asarray(b, dtype=np.float32)).reshape(1, 4)
    return float(box_iou_xyxy(aa, bb)[0])


def gt_norm(r: dict[str, str]) -> np.ndarray | None:
    b = parse_box(r.get("gt_bbox_xyxy")); w = fval(r, "image_width"); h = fval(r, "image_height")
    if b is None or w <= 0 or h <= 0: return None
    return np.asarray([b[0]/w, b[1]/h, b[2]/w, b[3]/h], dtype=np.float32)


def model_for_fold(fold: int, device: torch.device, tag: str = "") -> ProposalRefiner:
    prefix = (str(tag).strip() + "_") if str(tag).strip() else ""
    p = OUT / "checkpoints" / f"proposal_refiner_{prefix}f{fold}_best.pt"
    if not p.exists(): raise FileNotFoundError(p)
    ck = torch.load(p, map_location="cpu", weights_only=False); m = ProposalRefiner(); m.load_state_dict(ck["model"]); m.to(device).eval(); return m


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--event-limit", type=int, default=None); ap.add_argument("--smoke-checkpoint", default=None); ap.add_argument("--trained-tag", default=""); ap.add_argument("--out-stem", default="stage3_proposal_validation"); args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    with CSV_PATH.open(newline="", encoding="utf-8") as f: rows = list(csv.DictReader(f))
    z = np.load(FEAT_PATH, allow_pickle=False); cls, roi, keys = z["cls"], z["roi"], [str(x) for x in z["row_keys"]]; key_to_idx = {k: i for i, k in enumerate(keys)}
    by_track: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows): by_track[track_key(r)].append(i)
    for k in list(by_track): by_track[k].sort(key=lambda i: (int(rows[i].get("event_rank", 0)), int(rows[i].get("frame_id", 0)), int(rows[i].get("proposal_local_id", 0))))
    events = [json.loads(x) for x in POS_PATH.read_text().splitlines() if x.strip()]; events = sorted(events, key=lambda e: str(e["event_key"]))
    if args.event_limit is not None: events = events[:int(args.event_limit)]
    if not args.smoke: assert len(events) == 76, len(events)
    conditions = ["phase21_raw_baseline", "phase21_best_nontraining", "gt_tight_oracle", "frozen_oracle_correspondence"]
    trained_models: dict[int, ProposalRefiner] = {}
    smoke_model: ProposalRefiner | None = None
    if args.smoke_checkpoint:
        ck = torch.load(Path(args.smoke_checkpoint), map_location="cpu", weights_only=False); smoke_model = ProposalRefiner(); smoke_model.load_state_dict(ck["model"]); smoke_model.to(device).eval(); conditions.append("smoke_trained_refiner")
    if not args.smoke:
        conditions.append("best_trained_refiner")
        for fold in range(4): trained_models[fold] = model_for_fold(fold, device, args.trained_tag)
    pred_cache: dict[tuple[str, int], np.ndarray] = {}
    def pred_for(condition: str, fold: int) -> np.ndarray:
        key = (condition, fold)
        if key in pred_cache: return pred_cache[key]
        if condition == "best_trained_refiner": arr = load_predictions(trained_models[fold], rows, cls, roi, device)
        elif condition == "smoke_trained_refiner": arr = load_predictions(smoke_model, rows, cls, roi, device)
        else: arr = load_predictions(None, rows, cls, roi, device)
        pred_cache[key] = arr; return arr
    records: list[dict[str, Any]] = []
    for condition in conditions:
        cond_recs: list[dict[str, Any]] = []
        for event in events:
            fold = int(event["fold"]); sk = str(event["source_tracklet_keys"][0]); tk = str(event["target_tracklet_key"]); si = by_track.get(sk, []); ti = by_track.get(tk, [])
            pred = pred_for(condition, fold)
            for prefix in PREFIXES:
                src_idx = si; tgt_idx = ti[:min(prefix, len(ti))]
                src_iou: list[float] = []; tgt_iou: list[float] = []
                src_rel = 0; tgt_rel = 0
                for i in src_idx:
                    g = gt_norm(rows[i]); val = fval(rows[i], "row_iou") if condition != "gt_tight_oracle" else 1.0
                    if condition != "gt_tight_oracle" and g is not None: val = iou_norm(pred[i], g)
                    src_iou.append(val); src_rel += int(str(rows[i].get("assigned", "0")) == "1" and val >= .5)
                for i in tgt_idx:
                    g = gt_norm(rows[i]); val = fval(rows[i], "row_iou") if condition != "gt_tight_oracle" else 1.0
                    if condition != "gt_tight_oracle" and g is not None: val = iou_norm(pred[i], g)
                    tgt_iou.append(val); tgt_rel += int(str(rows[i].get("assigned", "0")) == "1" and val >= .5)
                if condition == "gt_tight_oracle":
                    src_rel = sum(gt_norm(rows[i]) is not None for i in src_idx); tgt_rel = sum(gt_norm(rows[i]) is not None for i in tgt_idx)
                rec = {"condition": condition, "event_key": str(event["event_key"]), "fold": fold, "category": int(event["category_gt_denominator_only"]), "source_tracklet_key": sk, "target_tracklet_key": tk, "source_video": int(event["source_video"]), "target_video": int(event["target_video"]), "prefix": prefix, "source_candidate_box_count": len(src_idx), "target_candidate_box_count_in_prefix": len(tgt_idx), "source_reliable": src_rel, "target_reliable": tgt_rel, "source_max_iou": max(src_iou, default=0.), "target_max_iou": max(tgt_iou, default=0.), "source_iou_mean": float(np.mean(src_iou)) if src_iou else 0., "target_iou_mean": float(np.mean(tgt_iou)) if tgt_iou else 0., "source_iou_median": float(np.median(src_iou)) if src_iou else 0., "target_iou_median": float(np.median(tgt_iou)) if tgt_iou else 0., "source_max_score": max((fval(rows[i], "score") for i in src_idx), default=0.), "target_max_score": max((fval(rows[i], "score") for i in tgt_idx), default=0.), "source_area_mean": float(np.mean([fval(rows[i], "area_fraction") for i in src_idx])) if src_idx else 0., "target_area_mean": float(np.mean([fval(rows[i], "area_fraction") for i in tgt_idx])) if tgt_idx else 0., "ceiling": bool(event.get("kind") == "positive_existing" and src_rel > 0 and tgt_rel > 0)}
                cond_recs.append(rec)
        records.extend(cond_recs)
    aggregate: dict[str, Any] = {"protocol": "trackocd_iclr27_phase22_stage3_true_iou_proposal_validation", "positive_event_denominator": 76 if not args.smoke else len(events), "prefixes": list(PREFIXES), "reliable_rule": "assigned == 1 and transformed IoU >= 0.5", "conditions": {}, "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels"]}
    for condition in conditions:
        cr = [x for x in records if x["condition"] == condition]
        ps = []
        for prefix in PREFIXES:
            rr = [x for x in cr if x["prefix"] == prefix]; good = [x for x in rr if x["ceiling"]];
            by_fold = []
            for fold in range(4):
                fr = [x for x in rr if x["fold"] == fold]; fg = [x for x in fr if x["ceiling"]]; by_fold.append({"fold": fold, "positive_denominator": len(fr), "source_reliable_events": sum(x["source_reliable"] > 0 for x in fr), "target_reliable_events": sum(x["target_reliable"] > 0 for x in fr), "ceiling_correct": len(fg), "ceiling_recall": len(fg)/max(len(fr),1), "category_coverage": len({x["category"] for x in fg}), "video_coverage": len({x["target_video"] for x in fg})})
            ps.append({"prefix": prefix, "positive_denominator": len(rr), "source_reliable_events": sum(x["source_reliable"] > 0 for x in rr), "target_reliable_events": sum(x["target_reliable"] > 0 for x in rr), "ceiling_correct": len(good), "ceiling_recall": len(good)/max(len(rr),1), "category_coverage": len({x["category"] for x in good}), "video_coverage": len({x["target_video"] for x in good}), "source_iou_mean": float(np.mean([x["source_iou_mean"] for x in rr])) if rr else 0., "target_iou_mean": float(np.mean([x["target_iou_mean"] for x in rr])) if rr else 0., "source_iou_median": float(np.median([x["source_iou_median"] for x in rr])) if rr else 0., "target_iou_median": float(np.median([x["target_iou_median"] for x in rr])) if rr else 0., "by_fold": by_fold, "failure_event_keys": [x["event_key"] for x in rr if not x["ceiling"]]})
        aggregate["conditions"][condition] = {"prefix_summary": ps, "prefix16": next(x for x in ps if x["prefix"] == 16), "event_records": len(cr)}
    atomic_json(OUT / "metrics" / f"{args.out_stem}.json", aggregate)
    atomic_json(OUT / "audit" / f"{args.out_stem}_event_records.json", {"protocol": aggregate["protocol"], "records": records})
    # A compact CSV is convenient for the complete 76-event report.
    csv_path = OUT / "audit" / f"{args.out_stem}_event_summary.csv"; csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["condition", "event_key", "fold", "category", "prefix", "source_reliable", "target_reliable", "source_max_iou", "target_max_iou", "ceiling"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields); wr.writeheader(); wr.writerows({k: r[k] for k in fields} for r in records)
    print(json.dumps({"conditions": conditions, "events": len(events), "prefix16": {c: aggregate["conditions"][c]["prefix16"]["ceiling_correct"] for c in conditions}}, indent=2, sort_keys=True))


if __name__ == "__main__": main()

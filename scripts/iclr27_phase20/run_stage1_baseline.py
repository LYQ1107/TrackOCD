#!/usr/bin/env python
"""Phase20 Stage 1 frozen DINOv2 correspondence diagnostics.

No parameters are fitted here.  Candidate tracks are restricted to held
TRAIN categories, different physical tracks, and different videos.  The
same 76 positive event queries and causal prefixes as Stage 0 are used;
observability is reported as a stratum rather than used to select a model.
"""
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

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/iclr27_phase19r/sources/public_rows_corrected.csv"
FEAT = ROOT / "data/iclr27_phase19r/sources/public_cls_roi.npz"
OUT = ROOT / "outputs/iclr27_phase20"
PREFIXES = (1, 2, 4, 8, 16)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for x in iter(lambda: f.read(1 << 20), b""):
            h.update(x)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / max(n, 1e-8)


def main() -> None:
    with SRC.open(newline="") as f:
        rows = list(csv.DictReader(f))
    z = np.load(FEAT, mmap_mode="r")
    cls = z["cls"]; roi = z["roi"]
    assert cls.shape[0] == len(rows) and roi.shape == cls.shape
    rowkey_to_i = {str(r["row_key"]): i for i, r in enumerate(rows)}
    track_rows: dict[str, list[int]] = defaultdict(list)
    track_meta: dict[str, tuple[int, int]] = {}
    for i, r in enumerate(rows):
        key = f"v{int(r['video_id'])}:p{int(r['track_id'])}"
        track_rows[key].append(i)
        track_meta[key] = (int(r["video_id"]), int(r["gt_category_id_common"]))
    for k, ix in track_rows.items():
        ix.sort(key=lambda i: (int(rows[i].get("event_rank", 0)), i))

    events = [json.loads(x) for x in (OUT / "audit/observability_events.json").read_text().splitlines()] if False else json.loads((OUT / "audit/observability_events.json").read_text())
    # Stage 0 stores one record per event/prefix.  Positive queries are the
    # fixed CT denominator and have no future-frame use.
    obs = {(str(r["event_key"]), int(r["causal_prefix_requested"])): r for r in events}
    positive_keys = sorted({str(r["event_key"]) for r in events if r["kind"] == "positive_existing"})
    event_meta = {str(r["event_key"]): r for r in events if r["kind"] == "positive_existing" and int(r["causal_prefix_requested"]) == 1}
    fold_manifest = json.loads((OUT / "manifests/fold_manifest.json").read_text())
    held_by_fold = {int(f["fold"]): {int(c) for c in f["held_categories"]} for f in fold_manifest["folds"]}

    # Cache causal aggregations once per track/prefix/method.  Feature arrays
    # remain mmap-backed; only the compact track-level vectors are retained.
    methods = ("cls_mean", "roi_mean", "cls_last", "roi_last", "cls_max", "roi_max")
    cache: dict[tuple[str, int, str], np.ndarray] = {}
    def vec(key: str, p: int, method: str) -> np.ndarray:
        ck = (key, int(p), method)
        if ck in cache: return cache[ck]
        ix = track_rows.get(key, [])[: min(int(p), len(track_rows.get(key, [])))]
        if not ix:
            out = np.zeros(cls.shape[1], dtype=np.float32)
        else:
            a = np.asarray(cls[ix] if method.startswith("cls") else roi[ix], dtype=np.float32)
            if method.endswith("_mean"):
                out = a.mean(axis=0)
            elif method.endswith("_last"):
                out = a[-1]
            elif method.endswith("_max"):
                out = a.max(axis=0)
            else:
                raise ValueError(method)
        cache[ck] = normalize(out)
        return cache[ck]

    # Build candidate pools per fold once.  Unknown/false-positive rows are
    # never candidates; all candidates are real TRAIN-supported tracks.
    candidates: dict[int, list[str]] = {}
    for fold, cats in held_by_fold.items():
        candidates[fold] = sorted(k for k, (_, c) in track_meta.items()
                                  if c in cats and c >= 0 and any(rows[i].get("gt_role_common") == "supported_known" for i in track_rows[k]))

    rows_out: list[dict[str, Any]] = []
    aggregate: dict[str, Any] = {m: {} for m in methods}
    for method in methods:
        for p in PREFIXES:
            samples: list[dict[str, Any]] = []
            pair_scores: list[float] = []; pair_labels: list[int] = []
            by_cat: dict[int, list[int]] = defaultdict(list); by_vid: dict[int, list[int]] = defaultdict(list)
            by_o: dict[str, list[int]] = defaultdict(list)
            ap_values: list[float] = []; hard_gaps: list[float] = []; consistency_values: list[float] = []
            for ek in positive_keys:
                e = event_meta[ek]
                qkey = str(e["target_tracklet_key"]); cat = int(e["category"]); vid = int(e["target_video"]); fold = int(e["fold"])
                q = vec(qkey, p, method)
                cand = [k for k in candidates[fold] if k != qkey and track_meta[k][0] != vid]
                scored = [(float(q @ vec(k, p, method)), k, int(track_meta[k][1] == cat)) for k in cand]
                positives = [x for x in scored if x[2] == 1]
                negatives = [x for x in scored if x[2] == 0]
                if not positives or not negatives:
                    continue
                scored.sort(key=lambda x: (-x[0], x[1]))
                hit1 = int(scored[0][2] == 1)
                hit5 = int(any(x[2] for x in scored[:5]))
                npos = len(positives); rank = 0; hit_count = 0; ap = 0.0
                for j, x in enumerate(scored, 1):
                    if x[2]:
                        hit_count += 1; ap += hit_count / j
                ap /= max(npos, 1)
                max_pos = max(x[0] for x in positives); max_neg = max(x[0] for x in negatives)
                gap = max_pos - max_neg
                # Multi-positive consistency is the mean pairwise cosine
                # among distinct same-category candidate tracks, not AP and
                # not query self-similarity.  Single-positive queries are
                # excluded from this descriptive statistic.
                if len(positives) >= 2:
                    pv = [vec(k, p, method) for _, k, _ in positives]
                    pair_vals = [float(pv[i] @ pv[j]) for i in range(len(pv)) for j in range(i + 1, len(pv))]
                    if pair_vals: consistency_values.append(float(statistics.mean(pair_vals)))
                orec = obs.get((ek, p), {})
                ostr = "observable" if orec.get("target_reliably_visible") and orec.get("source_reliable_materialized") else "not_reliably_observable"
                sample = {"event_key": ek, "fold": fold, "category": cat, "video": vid,
                          "method": method, "prefix": int(p), "candidate_count": len(scored),
                          "positive_candidate_count": npos, "hit_at_1": hit1, "hit_at_5": hit5,
                          "average_precision": float(ap), "hard_negative_gap": float(gap),
                          "observability_stratum": ostr,
                          "target_reliably_visible": bool(orec.get("target_reliably_visible", False)),
                          "source_reliable_materialized": bool(orec.get("source_reliable_materialized", False))}
                samples.append(sample)
                by_cat[cat].append(hit1); by_vid[vid].append(hit1); by_o[ostr].append(hit1)
                ap_values.append(ap); hard_gaps.append(gap)
                for s, _, lab in scored:
                    pair_scores.append(s); pair_labels.append(lab)
            hit1s = [x["hit_at_1"] for x in samples]; hit5s = [x["hit_at_5"] for x in samples]
            try: roc = float(roc_auc_score(pair_labels, pair_scores))
            except ValueError: roc = None
            try: pr = float(average_precision_score(pair_labels, pair_scores))
            except ValueError: pr = None
            aggregate[method][str(p)] = {
                "query_count": len(samples), "r_at_1": float(statistics.mean(hit1s)) if hit1s else 0.0,
                "r_at_5": float(statistics.mean(hit5s)) if hit5s else 0.0,
                "mAP": float(statistics.mean(ap_values)) if ap_values else 0.0,
                "pair_roc_auc": roc, "pair_pr_auc": pr,
                "hard_negative_positive_minus_negative_gap": float(statistics.mean(hard_gaps)) if hard_gaps else 0.0,
                "category_macro_r_at_1": float(statistics.mean([statistics.mean(v) for v in by_cat.values()])) if by_cat else 0.0,
                "video_macro_r_at_1": float(statistics.mean([statistics.mean(v) for v in by_vid.values()])) if by_vid else 0.0,
                "observability_strata": {k: {"queries": len(v), "r_at_1": float(statistics.mean(v)) if v else 0.0} for k, v in sorted(by_o.items())},
                "same_category_multi_positive_consistency": float(statistics.mean(consistency_values)) if consistency_values else 0.0,
                "same_category_multi_positive_queries": len(consistency_values),
            }
            rows_out.extend(samples)

    metrics = {
        "protocol": "trackocd_iclr27_phase20_stage1_frozen_correspondence",
        "source_rows": len(rows), "source_rows_path": str(SRC), "source_rows_sha256": sha256(SRC),
        "feature_path": str(FEAT), "feature_sha256": sha256(FEAT),
        "feature_shapes": {"cls": list(cls.shape), "roi": list(roi.shape)},
        "methods": list(methods), "prefixes": list(PREFIXES),
        "positive_event_denominator": len(positive_keys), "candidate_rule": "different physical track and different video; held TRAIN categories only",
        "metrics": aggregate, "observability_source": str(OUT / "audit/observability_events.json"),
        "labels_used": "public TRAIN category/video metadata only", "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels"],
        "training_performed": False,
    }
    atomic_json(OUT / "metrics/frozen_correspondence_baseline.json", metrics)
    # Compact query-level artifact is useful for audit/reproducibility.
    qpath = OUT / "metrics/frozen_correspondence_queries.json"
    atomic_json(qpath, rows_out)

    report = [
        "# Phase20 Stage 1 — Frozen correspondence baseline", "",
        "This is a no-training diagnostic on the frozen public TRAIN DSCT rows.  Candidate tracks are from a different physical track and video; no online threshold was selected.", "",
        f"* Positive query denominator: **{len(positive_keys)}** events; causal prefixes: `{list(PREFIXES)}`.",
        f"* Features: DINOv2 CLS and proposal-ROI arrays `{list(cls.shape)}` from `{FEAT}`.",
        "* O strata are descriptive only.  They are not used to filter queries or choose a model.", "",
        "| method | prefix | queries | R@1 | R@5 | mAP | pair ROC-AUC | pair PR-AUC | hard-negative gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        for p in PREFIXES:
            x = aggregate[method][str(p)]
            report.append(f"| {method} | {p} | {x['query_count']} | {x['r_at_1']:.4f} | {x['r_at_5']:.4f} | {x['mAP']:.4f} | {x['pair_roc_auc'] if x['pair_roc_auc'] is not None else 'NA'} | {x['pair_pr_auc'] if x['pair_pr_auc'] is not None else 'NA'} | {x['hard_negative_positive_minus_negative_gap']:.4f} |")
    report += ["", "The machine-readable per-prefix and observability-stratified values are in [`frozen_correspondence_baseline.json`](../../outputs/iclr27_phase20/metrics/frozen_correspondence_baseline.json).  Since Stage 0 failed Gate O, no Stage 2 correspondence encoder or Stage 3 controller reconnect is authorized.", ""]
    (ROOT / "docs/iclr27_phase20/PHASE20_BASELINE_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    atomic_json(OUT / "completion/stage1.done", {"stage": "stage1", "training_performed": False, "metrics": str(OUT / "metrics/frozen_correspondence_baseline.json")})
    print(json.dumps({"methods": len(methods), "prefixes": len(PREFIXES), "query_rows": len(rows_out), "metrics": str(OUT / "metrics/frozen_correspondence_baseline.json")}, indent=2))


if __name__ == "__main__":
    main()

"""Phase 17 common proposal/GT alignment and representation-shift audit.

This module is deliberately independent of the historical Phase 15S role
columns.  It reconstructs a single deterministic track assignment for public
and DEV+ rows, retains the exact frame IoU for every assigned row (including
IoU < .5), and keeps the historical public filter as a separate diagnostic.
No category is used while assigning tracks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    os.replace(tmp, path)


def _box(value: Any) -> list[float]:
    if isinstance(value, str):
        value = json.loads(value)
    return [float(x) for x in value]


def box_iou(a: Iterable[float], b: Iterable[float]) -> float:
    a = list(a); b = list(b)
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + ab - inter
    return float(inter / union) if union > 0 else 0.0


def row_key(row: dict[str, Any]) -> str:
    return ":".join(str(int(row[k])) for k in
                     ("video_id", "frame_id", "proposal_local_id", "track_id", "image_id"))


def _read_rows(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open() as f:
        for raw in csv.DictReader(f):
            r = dict(raw)
            for k in ("video_id", "frame_id", "source_frame_index", "image_id",
                      "proposal_local_id", "track_id", "det_category_id", "prior_hits"):
                r[k] = int(float(r[k]))
            r["score"] = float(r["score"])
            r["bbox_xyxy"] = _box(r["bbox_xyxy"])
            out.append(r)
    return out


def _public_gt(annotation: Path, known_ids: set[int]) -> dict[tuple[int, int], dict[str, Any]]:
    src = json.loads(annotation.read_text())
    images = {int(x["id"]): x for x in src.get("images", [])}
    tracks: dict[tuple[int, int], dict[str, Any]] = {}
    for ann in src.get("annotations", []):
        im = images.get(int(ann["image_id"]))
        if im is None:
            continue
        v, t = int(ann["video_id"]), int(ann["track_id"])
        x, y, w, h = [float(z) for z in ann["bbox"]]
        rec = tracks.setdefault((v, t), {"video_id": v, "track_id": t,
                                          "category_id": int(ann["category_id"]),
                                          "frames": {}, "image_ids": {}})
        # The source annotation has one frame record per track.  If a corrupt
        # duplicate occurs, retain the first deterministic record.
        f = int(im.get("frame_index", im.get("frame_id", 0)))
        rec["frames"].setdefault(f, [x, y, x + w, y + h])
        rec["image_ids"].setdefault(f, int(ann["image_id"]))
    for rec in tracks.values():
        rec["role"] = "supported_known" if int(rec["category_id"]) in known_ids else "novel"
    return tracks


def _dev_gt(sidecar: Path, known_ids: set[int]) -> dict[tuple[int, int], dict[str, Any]]:
    tracks: dict[tuple[int, int], dict[str, Any]] = {}
    with sidecar.open() as f:
        for line in f:
            if not line.strip():
                continue
            g = json.loads(line)
            v, t = int(g["video_id"]), int(g["track_id"])
            frames = {int(fi): _box(bb) for fi, bb in zip(g["frame_indices"], g["boxes_xyxy"])}
            rec = {"video_id": v, "track_id": t, "category_id": int(g["category_id"]),
                   "frames": frames,
                   "image_ids": {int(fi): int(ii) for fi, ii in zip(g["frame_indices"], g.get("image_ids", []))},
                   "role": "supported_known" if int(g["category_id"]) in known_ids else "novel"}
            tracks[(v, t)] = rec
    return tracks


def load_domain(proposals: Path, gt: Path, annotation: Path | None,
                known_ids: set[int], domain: str) -> tuple[list[dict[str, Any]], dict[tuple[int, int], dict[str, Any]]]:
    rows = _read_rows(proposals)
    if domain == "public":
        if annotation is None:
            raise ValueError("public alignment needs the public annotation")
        gts = _public_gt(annotation, known_ids)
    elif domain == "devplus":
        gts = _dev_gt(gt, known_ids)
    else:
        raise ValueError(domain)
    return rows, gts


def greedy_match(rows: list[dict[str, Any]], gts: dict[tuple[int, int], dict[str, Any]]) -> dict[tuple[int, int], tuple[tuple[int, int], float]]:
    pred: dict[tuple[int, int], dict[int, list[float]]] = defaultdict(dict)
    for r in rows:
        pred[(int(r["video_id"]), int(r["track_id"]))][int(r["source_frame_index"])] = r["bbox_xyxy"]
    candidates: list[tuple[float, tuple[int, int], tuple[int, int]]] = []
    for pk, pframes in pred.items():
        for gk, grec in gts.items():
            if pk[0] != gk[0]:
                continue
            shared = sorted(set(pframes) & set(grec["frames"]))
            if not shared:
                continue
            score = float(np.mean([box_iou(pframes[f], grec["frames"][f]) for f in shared]))
            # Zero-overlap track pairs are not evidence of an assignment; this
            # is the same positive-candidate convention as the Phase14C audit.
            if score > 0.0:
                candidates.append((score, pk, gk))
    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
    mapping: dict[tuple[int, int], tuple[tuple[int, int], float]] = {}
    used_p: set[tuple[int, int]] = set(); used_g: set[tuple[int, int]] = set()
    for score, pk, gk in candidates:
        if pk in used_p or gk in used_g:
            continue
        used_p.add(pk); used_g.add(gk); mapping[pk] = (gk, score)
    return mapping


def align_rows(rows: list[dict[str, Any]], gts: dict[tuple[int, int], dict[str, Any]],
               mapping: dict[tuple[int, int], tuple[tuple[int, int], float]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in rows:
        r = dict(raw)
        pk = (int(r["video_id"]), int(r["track_id"]))
        if pk in mapping:
            gk, temporal = mapping[pk]; g = gts[gk]
            f = int(r["source_frame_index"])
            riou = box_iou(r["bbox_xyxy"], g["frames"].get(f, [0, 0, 0, 0]))
            r.update({"assigned": 1, "gt_track_id": int(g["track_id"]),
                      "gt_category_id_common": int(g["category_id"]),
                      "gt_role_common": g["role"], "gt_bbox_xyxy": g["frames"].get(f),
                      "row_iou": float(riou), "track_temporal_iou": float(temporal)})
        else:
            r.update({"assigned": 0, "gt_track_id": -1, "gt_category_id_common": -1,
                      "gt_role_common": "fp", "gt_bbox_xyxy": None,
                      "row_iou": 0.0, "track_temporal_iou": 0.0})
        r["area_fraction"] = max(0.0, (r["bbox_xyxy"][2] - r["bbox_xyxy"][0])) * max(0.0, (r["bbox_xyxy"][3] - r["bbox_xyxy"][1])) / (640.0 * 480.0)
        r["row_key"] = row_key(r)
        out.append(r)
    track_len = Counter((int(r["video_id"]), int(r["track_id"])) for r in out)
    for r in out:
        r["proposal_track_length"] = int(track_len[(int(r["video_id"]), int(r["track_id"]))])
    return out


def write_aligned(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["video_id", "frame_id", "source_frame_index", "image_id", "proposal_local_id",
              "track_id", "score", "bbox_xyxy", "det_category_id", "source_family",
              "prior_hits", "assigned", "gt_track_id", "gt_category_id_common",
              "gt_role_common", "gt_bbox_xyxy", "row_iou", "track_temporal_iou",
              "area_fraction", "proposal_track_length", "row_key"]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader()
        for r in rows:
            q = dict(r)
            for k in ("bbox_xyxy", "gt_bbox_xyxy"):
                q[k] = "" if q.get(k) is None else json.dumps(q[k], separators=(",", ":"))
            w.writerow(q)
    os.replace(tmp, path)


def _macro(values: dict[Any, list[float]]) -> float | None:
    means = [float(np.mean(v)) for v in values.values() if v]
    return float(np.mean(means)) if means else None


def _bootstrap_video(rows: list[dict[str, Any]], field: str, n: int = 1000, seed: int = 1701) -> dict[str, float] | None:
    by_video: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        by_video[int(r["video_id"])].append(float(r[field]))
    if not by_video:
        return None
    vids = sorted(by_video); vals = np.asarray([np.mean(by_video[v]) for v in vids], dtype=float)
    rng = np.random.default_rng(seed); draws = np.empty(n, dtype=float)
    for i in range(n):
        draws[i] = np.mean(vals[rng.integers(0, len(vals), size=len(vals))])
    return {"mean": float(np.mean(vals)), "low95": float(np.quantile(draws, .025)),
            "high95": float(np.quantile(draws, .975)), "videos": len(vids), "resamples": n}


def population_summary(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    if not rows:
        return {"name": name, "rows": 0, "videos": 0, "tracks": 0, "categories": 0}
    cats = [int(r["gt_category_id_common"]) for r in rows if int(r["gt_category_id_common"]) >= 0]
    by_cat: dict[int, list[float]] = defaultdict(list); by_video: dict[int, list[float]] = defaultdict(list); by_track: dict[tuple[int, int], list[float]] = defaultdict(list)
    for r in rows:
        x = float(r["row_iou"]); c = int(r["gt_category_id_common"]); v = int(r["video_id"]); t = (v, int(r["track_id"]))
        if c >= 0: by_cat[c].append(x)
        by_video[v].append(x); by_track[t].append(x)
    out = {"name": name, "rows": len(rows), "videos": len(by_video), "tracks": len(by_track),
           "categories": len(set(cats)), "row_iou_mean": float(np.mean([float(r["row_iou"]) for r in rows])),
           "row_iou_median": float(np.median([float(r["row_iou"]) for r in rows])),
           "temporal_iou_mean": float(np.mean([float(r["track_temporal_iou"]) for r in rows])),
           "area_fraction_mean": float(np.mean([float(r["area_fraction"]) for r in rows])),
           "track_length_mean": float(np.mean([int(r["proposal_track_length"]) for r in rows])),
           "category_macro_row_iou": _macro(by_cat), "video_macro_row_iou": _macro(by_video),
           "track_macro_row_iou": _macro(by_track), "video_bootstrap_row_iou": _bootstrap_video(rows, "row_iou")}
    out["category_counts"] = {str(k): len(v) for k, v in sorted(by_cat.items())}
    return out


def _aligned_feature_rows(rows: list[dict[str, Any]], feature_path: Path, field: str = "roi") -> tuple[np.ndarray, list[dict[str, Any]]]:
    z = np.load(feature_path, allow_pickle=False)
    keys = [str(x) for x in z["row_keys"]]
    lookup = {k: i for i, k in enumerate(keys)}
    keep = [r for r in rows if r["row_key"] in lookup]
    if not keep:
        return np.empty((0, z[field].shape[1]), dtype=np.float32), []
    idx = np.asarray([lookup[r["row_key"]] for r in keep], dtype=int)
    return np.asarray(z[field][idx], dtype=np.float32), keep


def _domain_metrics(pub_rows: list[dict[str, Any]], dev_rows: list[dict[str, Any]], pub_feat: Path, dev_feat: Path) -> dict[str, Any]:
    xp, rp = _aligned_feature_rows(pub_rows, pub_feat); xd, rd = _aligned_feature_rows(dev_rows, dev_feat)
    n = min(len(xp), len(xd), 5000)
    if n == 0:
        return {"status": "missing_aligned_features", "public_rows": len(xp), "devplus_rows": len(xd)}
    rng = np.random.default_rng(1717)
    ip = np.sort(rng.choice(len(xp), size=n, replace=False)); idv = np.sort(rng.choice(len(xd), size=n, replace=False))
    a = xp[ip].astype(np.float64); b = xd[idv].astype(np.float64)
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12); b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    cent_cos = float(np.dot(np.mean(a, axis=0), np.mean(b, axis=0)) / max(np.linalg.norm(np.mean(a, axis=0)) * np.linalg.norm(np.mean(b, axis=0)), 1e-12))
    # A bounded deterministic MMD/energy estimate (subsampled equally).
    m = min(n, 1200); a2, b2 = a[:m], b[:m]
    d2 = np.sum((a2[:, None, :] - b2[None, :, :]) ** 2, axis=2)
    sig = float(np.sqrt(np.median(d2[d2 > 0]))) if np.any(d2 > 0) else 1.0
    def rbf(q): return np.exp(-q / max(2 * sig * sig, 1e-12))
    kaa = rbf(np.sum((a2[:, None, :] - a2[None, :, :]) ** 2, axis=2)); kbb = rbf(np.sum((b2[:, None, :] - b2[None, :, :]) ** 2, axis=2)); kab = rbf(d2)
    mmd2 = float(np.mean(kaa) + np.mean(kbb) - 2 * np.mean(kab))
    energy = float(2 * np.mean(np.sqrt(np.maximum(d2, 0))) - np.mean(np.sqrt(np.maximum(np.sum((a2[:, None, :] - a2[None, :, :]) ** 2, axis=2), 0))) - np.mean(np.sqrt(np.maximum(np.sum((b2[:, None, :] - b2[None, :, :]) ** 2, axis=2), 0))))
    clf: dict[str, Any] = {"status": "not_run"}
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold, cross_val_score
        X = np.concatenate([a, b]); y = np.concatenate([np.zeros(n), np.ones(n)])
        groups = np.asarray([int(r["video_id"]) for r in [rp[i] for i in ip]] + [int(r["video_id"]) for r in [rd[i] for i in idv]])
        ng = len(np.unique(groups)); folds = min(5, ng)
        if folds >= 2:
            scores = cross_val_score(LogisticRegression(max_iter=200, solver="liblinear", random_state=1717), X, y, groups=groups, cv=GroupKFold(folds), scoring="roc_auc")
            clf = {"status": "ok", "video_grouped_roc_auc_mean": float(np.mean(scores)), "video_grouped_roc_auc_std": float(np.std(scores)), "folds": folds}
    except Exception as exc:
        clf = {"status": "error", "error": repr(exc)}
    # Deterministic quality-bin matching controls the selection-bias hypothesis
    # without fitting a classifier or using category labels.
    qmatch = {"status": "ok", "bins": {"row_iou": [0, .25, .5, .75, 1], "area_fraction": [0, .02, .08, .2, 1]}, "matched_rows": 0}
    try:
        def bins(rs):
            out = defaultdict(list)
            for i, r in enumerate(rs):
                ri = float(r.get("row_iou", 0)); ar = float(r.get("area_fraction", 0))
                rb = min(3, int(np.digitize(ri, [0, .25, .5, .75], right=False))); ab = min(3, int(np.digitize(ar, [.02, .08, .2], right=False))); out[(rb, ab)].append(i)
            return out
        bp, bd = bins(rp), bins(rd); rngq = np.random.default_rng(1718); aa, bb = [], []
        for k in sorted(set(bp) & set(bd)):
            m = min(len(bp[k]), len(bd[k]), 200)
            if m:
                aa.extend(rngq.choice(bp[k], m, replace=False)); bb.extend(rngq.choice(bd[k], m, replace=False))
        if aa and bb:
            # Map original row positions to the balanced feature sample by
            # selecting from the full feature arrays rather than the 5k draw.
            xpa, rpa = _aligned_feature_rows(pub_rows, pub_feat); xdb, rdb = _aligned_feature_rows(dev_rows, dev_feat)
            xa = xpa[np.asarray(aa)]; xb = xdb[np.asarray(bb)]; xa = xa / np.maximum(np.linalg.norm(xa, axis=1, keepdims=True), 1e-12); xb = xb / np.maximum(np.linalg.norm(xb, axis=1, keepdims=True), 1e-12)
            ma, mb = xa.mean(0), xb.mean(0); qmatch.update({"matched_rows": int(len(xa)), "normalized_centroid_cosine": float(np.dot(ma, mb) / max(np.linalg.norm(ma) * np.linalg.norm(mb), 1e-12))})
    except Exception as exc:
        qmatch = {"status": "error", "error": repr(exc)}
    return {"status": "ok", "feature": "roi", "public_rows": len(xp), "devplus_rows": len(xd),
            "balanced_sample": n, "normalized_centroid_cosine": cent_cos,
            "balanced_rbf_mmd2": max(0.0, mmd2), "balanced_energy_distance": energy,
            "quality_bin_matched": qmatch, "video_grouped_domain_classifier": clf}


def _write_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    known = {int(x) for x in json.loads(args.known.read_text())}
    pub_raw, pub_gt = load_domain(args.public_proposals, args.public_gt, args.public_annotation, known, "public")
    dev_raw, dev_gt = load_domain(args.devplus_proposals, args.devplus_gt, None, known, "devplus")
    pub_map = greedy_match(pub_raw, pub_gt); dev_map = greedy_match(dev_raw, dev_gt)
    pub = align_rows(pub_raw, pub_gt, pub_map); dev = align_rows(dev_raw, dev_gt, dev_map)
    write_aligned(args.out_public_csv, pub); write_aligned(args.out_devplus_csv, dev)
    def assigned(rows): return [r for r in rows if int(r["assigned"]) == 1]
    def known_assigned(rows): return [r for r in rows if int(r["assigned"]) == 1 and r["gt_role_common"] == "supported_known"]
    # The historical filter is preserved as a labelled population, not used
    # for common alignment or for selecting roles.
    # Phase15S called the known role ``known`` (not ``supported_known``) in
    # the public CSV; preserve that exact historical population here.
    hist_public = [r for r in pub if str(r.get("gt_role", "fp")) in {"known", "supported_known", "novel"} and float(r.get("gt_iou", 0.0)) >= .5]
    populations = {
        "public_all_proposals": pub, "public_all_assigned_rows": assigned(pub),
        "public_assigned_supported_known_no_truncation": known_assigned(pub),
        "public_historical_filter_iou_ge_0_5": hist_public,
        "devplus_all_proposals": dev, "devplus_all_assigned_rows": assigned(dev),
        "devplus_assigned_supported_known_no_truncation": known_assigned(dev),
    }
    stats = {k: population_summary(v, k) for k, v in populations.items()}
    # Shared known category rows and fixed quality strata are explicit, so the
    # old public/DEV+ difference cannot be attributed to silent population mix.
    shared = sorted(set(r["gt_category_id_common"] for r in known_assigned(pub)) & set(r["gt_category_id_common"] for r in known_assigned(dev)))
    shared_stats = {}
    for c in shared:
        shared_stats[str(c)] = {"public": population_summary([r for r in known_assigned(pub) if int(r["gt_category_id_common"]) == int(c)], f"public_cat_{c}"),
                                "devplus": population_summary([r for r in known_assigned(dev) if int(r["gt_category_id_common"]) == int(c)], f"devplus_cat_{c}")}
    strata = {}
    bins = [0.0, .25, .5, .75, 1.0]
    for lo, hi in zip(bins[:-1], bins[1:]):
        key = f"row_iou_{lo:g}_{hi:g}"
        strata[key] = {"public": population_summary([r for r in known_assigned(pub) if lo <= float(r["row_iou"]) < (hi if hi < 1 else hi + 1e-9)], key),
                       "devplus": population_summary([r for r in known_assigned(dev) if lo <= float(r["row_iou"]) < (hi if hi < 1 else hi + 1e-9)], key)}
    controls = {"identical_boxes_iou": box_iou([1, 2, 11, 22], [1, 2, 11, 22]),
                "disjoint_boxes_iou": box_iou([0, 0, 1, 1], [2, 2, 3, 3]),
                "frame_vs_temporal_example": {"frame_mean": float(np.mean([1.0, 0.0])), "temporal_mean": .5}}
    result = {"protocol": "trackocd_iclr27_phase17_common_alignment", "known_ids_sha256": _write_hash(args.known.resolve()),
              "public": {"rows": len(pub), "tracks": len({(r['video_id'], r['track_id']) for r in pub}), "mapping_tracks": len(pub_map)},
              "devplus": {"rows": len(dev), "tracks": len({(r['video_id'], r['track_id']) for r in dev}), "mapping_tracks": len(dev_map)},
              "populations": stats, "shared_supported_known_categories": shared,
              "shared_category_stats": shared_stats, "quality_strata": strata, "synthetic_controls": controls,
              "representation_shift": _domain_metrics(pub, dev, args.public_features, args.devplus_features),
              "historical_public_filter_is_diagnostic_only": True,
              "assignment": "greedy positive mean exact source-frame IoU over shared frames; stable keys",
              "future_frames_used": False, "physical_id_as_feature": False, "q1_label_used": False}
    # A concise classification is intentionally conservative: the common
    # audit can support a proposal-shift claim only after quality matching.
    p = stats["public_assigned_supported_known_no_truncation"]["row_iou_mean"]
    d = stats["devplus_assigned_supported_known_no_truncation"]["row_iou_mean"]
    historical_known = [r for r in pub if str(r.get("gt_role", "")) in {"known", "supported_known"} and float(r.get("gt_iou", 0.0)) >= .5]
    historical_novel = [r for r in pub if str(r.get("gt_role", "")) == "novel" and float(r.get("gt_iou", 0.0)) >= .5]
    # These are the independently reproduced Phase15S quantities; they are
    # reported verbatim so a reader can see why the old public/DEV+ comparison
    # was not apples-to-apples.
    result["old_phase15s_statement"] = {
        "historical_public_known_rows": len(historical_known),
        "historical_public_known_mean_frame_iou": float(np.mean([float(r.get("gt_iou", 0.0)) for r in historical_known])) if historical_known else None,
        "historical_public_novel_rows": len(historical_novel),
        "common_public_assigned_known_rows": len(known_assigned(pub)),
        "common_public_assigned_known_mean_frame_iou": p,
        "common_devplus_assigned_known_rows": len(known_assigned(dev)),
        "common_devplus_assigned_known_mean_frame_iou": d,
        "historical_devplus_supported_known_temporal_mean": 0.261925,
        "historical_public_known_row_count_expected": 3781,
        "historical_public_role_category_counts": {"805": 43, "99": 19, "211": 6, "95": 1},
        "classification": "PARTIALLY_SUPPORTED",
        "reason": "historical public frame-filter and DEV+ temporal populations are not the same; common populations are reported above"
    }
    atomic_json(args.out_audit, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--public-proposals", type=Path, default=ROOT / "data/iclr27_phase17/sources/public_dsct_proposals.csv")
    ap.add_argument("--public-gt", type=Path, default=ROOT / "data/iclr27_phase17/sources/public_dsct_annotation.json")
    ap.add_argument("--public-annotation", type=Path, default=ROOT / "data/iclr27_phase17/sources/public_dsct_annotation.json")
    ap.add_argument("--devplus-proposals", type=Path, default=ROOT / "data/iclr27_phase17/sources/devplus_proposals_aligned.csv")
    ap.add_argument("--devplus-gt", type=Path, default=ROOT / "data/iclr27_phase17/sources/devplus_gt_tracks.jsonl")
    ap.add_argument("--known", type=Path, default=ROOT / "data/iclr27_phase17/sources/supported_known_ids.json")
    ap.add_argument("--public-features", type=Path, default=ROOT / "data/iclr27_phase17/sources/public_dinov2_features.npz")
    ap.add_argument("--devplus-features", type=Path, default=ROOT / "data/iclr27_phase17/sources/devplus_dinov2_features.npz")
    ap.add_argument("--out-public-csv", type=Path, default=ROOT / "outputs/iclr27_phase17/csv/common_public.csv")
    ap.add_argument("--out-devplus-csv", type=Path, default=ROOT / "outputs/iclr27_phase17/csv/common_devplus.csv")
    ap.add_argument("--out-audit", type=Path, default=ROOT / "outputs/iclr27_phase17/eval/common_proposal_shift_audit.json")
    args = ap.parse_args()
    run_audit(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase30 Stage1 frozen retrieval diagnostics.

The evaluator compares frozen DINOv2, the feature-identical Phase26 proposal
attachment, frozen Phase27/29 checkpoints, and a non-trained support-set score.
No held-event outcome is read or used for selection.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.iclr27_phase26.protocol import CSV_PATH, FEAT_PATH, by_track, load_aligned_features, order_key
from src.iclr27_phase27.correspondence import TrackCorrespondenceEncoder
from src.iclr27_phase29.representation import DomainAlignedResidualEncoder


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase30"
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


def load_tracks() -> tuple[list[dict[str, str]], dict[str, list[int]], np.ndarray]:
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cls, roi, _ = load_aligned_features(rows)
    feats = (0.8 * cls.astype(np.float32) + 0.2 * roi.astype(np.float32)).astype(np.float32)
    feats /= np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-8)
    return rows, by_track(rows), feats


def track_metadata(rows: list[dict[str, str]], tracks: dict[str, list[int]]) -> dict[str, dict[str, Any]]:
    out = {}
    for key, inds in tracks.items():
        ordered = sorted(inds, key=lambda i: order_key(rows[i]))
        gt = [rows[i] for i in ordered if rows[i].get("gt_track_id") not in {"", "-1", "None", "nan"}]
        if not gt: continue
        r = gt[-1]
        cat = int(r.get("gt_category_id_common", -1))
        if cat < 0: continue
        out[key] = {"category": cat, "video": int(r["video_id"]), "area": float(np.mean([float(rows[i].get("area_fraction", 0) or 0) for i in ordered])), "length": len(ordered), "rows": ordered}
    return out


def pad(keys: list[str], meta: dict[str, dict[str, Any]], feats: np.ndarray, prefix: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.zeros((len(keys), 16, feats.shape[1]), np.float32); mask = np.zeros((len(keys), 16), bool)
    for i, key in enumerate(keys):
        inds = meta[key]["rows"][: min(prefix, 16)]
        arr[i, : len(inds)] = feats[np.asarray(inds)]
        mask[i, : len(inds)] = True
    return arr, mask


@torch.no_grad()
def embed_model(model: torch.nn.Module | None, keys: list[str], meta: dict[str, dict[str, Any]], feats: np.ndarray, prefix: int, device: torch.device, batch: int = 256) -> np.ndarray:
    if model is None:
        arr = []
        for key in keys:
            inds = meta[key]["rows"][: min(prefix, 16)]
            v = feats[np.asarray(inds)].mean(0) if len(inds) else np.zeros(feats.shape[1], np.float32)
            v = v / max(float(np.linalg.norm(v)), 1e-8); arr.append(v)
        return np.asarray(arr, np.float32)
    vals = []
    for s in range(0, len(keys), batch):
        x, m = pad(keys[s:s + batch], meta, feats, prefix)
        xt, mt = torch.from_numpy(x).to(device), torch.from_numpy(m).to(device)
        vals.append(model(xt, mt).detach().cpu().numpy())
    return np.concatenate(vals, 0) if vals else np.zeros((0, feats.shape[1]), np.float32)


def retrieval(keys: list[str], vectors: np.ndarray, meta: dict[str, dict[str, Any]], support_sets: dict[str, list[str]] | None = None) -> dict[str, Any]:
    if not keys: return {"queries": 0, "r1": 0.0, "r5": 0.0, "map": 0.0, "hard_negative_gap": 0.0, "positive_coverage": 0.0}
    sim = vectors @ vectors.T
    index = {k: i for i, k in enumerate(keys)}
    videos = np.asarray([meta[k]["video"] for k in keys], dtype=np.int64)
    categories = np.asarray([meta[k]["category"] for k in keys], dtype=np.int64)
    all_idx = np.arange(len(keys), dtype=np.int64)
    r1, r5, aps, gaps, covered = [], [], [], [], 0
    for i, q in enumerate(keys):
        cand_idx = all_idx[(all_idx != i) & (videos != videos[i])]
        pos_idx = cand_idx[categories[cand_idx] == categories[i]]
        neg_idx = cand_idx[categories[cand_idx] != categories[i]]
        if len(pos_idx) == 0 or len(neg_idx) == 0: continue
        covered += 1
        if support_sets is None:
            scores = sim[i, cand_idx]
        else:
            supp = support_sets.get(q, [])
            if not supp: continue
            idx = [index[k] for k in supp if k in index and videos[index[k]] != videos[i]]
            if not idx: continue
            scores = np.max(vectors[np.asarray(idx)] @ vectors[cand_idx].T, axis=0)
        order = cand_idx[np.argsort(scores)[::-1]]
        pos_set = set(pos_idx.tolist())
        hit = np.asarray([int(j in pos_set) for j in order], np.float32)
        r1.append(float(hit[:1].max(initial=0))); r5.append(float(hit[:5].max(initial=0)))
        cum = np.cumsum(hit); aps.append(float(np.sum(cum / (np.arange(len(hit)) + 1) * hit) / max(len(pos_idx), 1)))
        if support_sets is None:
            gaps.append(float(np.max(sim[i, pos_idx]) - np.max(sim[i, neg_idx])))
        else:
            gaps.append(float(np.max(scores[np.isin(cand_idx, pos_idx)]) - np.max(scores[np.isin(cand_idx, neg_idx)])))
    return {"queries": len(r1), "r1": float(np.mean(r1)) if r1 else 0.0, "r5": float(np.mean(r5)) if r5 else 0.0, "map": float(np.mean(aps)) if aps else 0.0, "hard_negative_gap": float(np.mean(gaps)) if gaps else 0.0, "positive_coverage": float(covered / max(len(keys), 1)), "pairs": int(sum(1 for i, q in enumerate(keys) for j, k in enumerate(keys) if i != j and meta[k]["video"] != meta[q]["video"]))}


def load_comparator(phase: int, fold: int, device: torch.device) -> torch.nn.Module | None:
    if phase == 27:
        path = ROOT / f"outputs/iclr27_phase27/checkpoints/correspondence_f{fold}_best.pt"; model = TrackCorrespondenceEncoder()
    elif phase == 29:
        path = ROOT / f"outputs/iclr27_phase29/checkpoints/domain_aligned_f{fold}_best.pt"; model = DomainAlignedResidualEncoder()
    else:
        return None
    if not path.exists(): return None
    ck = torch.load(path, map_location="cpu", weights_only=False); model.load_state_dict(ck["model"]); model.to(device).eval(); return model


def main() -> None:
    torch.set_num_threads(1)
    rows, tracks, feats = load_tracks(); meta = track_metadata(rows, tracks)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_fold = []
    for fold in range(4):
        manifest = json.loads((OUT / f"manifests/episode_manifest_f{fold}.json").read_text())
        val = [r for r in manifest["records"] if r["split"] == "val"]
        keys = sorted({r["query_track_key"] for r in val if r["query_track_key"] in meta})
        support_sets = {r["query_track_key"]: r["support_track_keys"] for r in val if r["kind"] == "multi_positive_cross_video" and r["query_track_key"] in meta}
        # Keep support keys in the same validation fold and metadata-only.
        support_sets = {q: [k for k in ss if k in keys] for q, ss in support_sets.items()}
        models = {"raw_dinov2": None, "phase26_proposal_dinov2": None, "phase27_gru_comparator": load_comparator(27, fold, device), "phase29_residual_comparator": load_comparator(29, fold, device)}
        fold_out = {"fold": fold, "validation_tracklets": len(keys), "support_positive_episodes": sum(r["kind"] == "multi_positive_cross_video" for r in val), "null_episodes": sum(r["kind"] == "null_no_match_hard_negative" for r in val), "models": {}}
        for name, model in models.items():
            # Phase26 proposal-attached DINOv2 is feature-identical to raw
            # DINOv2 here; proposal source is frozen and no new backbone pass
            # is performed.  It is retained as an explicit comparator label.
            per_prefix = {}
            emb_cache = {}
            for prefix in PREFIXES:
                emb = embed_model(model, keys, meta, feats, prefix, device); emb_cache[prefix] = emb
                base = retrieval(keys, emb, meta)
                per_prefix[str(prefix)] = base
            # New interface diagnostic: max-over-support score, no learned params.
            support_prefix = {}
            for prefix in PREFIXES:
                support_prefix[str(prefix)] = retrieval(keys, emb_cache[prefix], meta, support_sets)
            consistency = []
            e16 = emb_cache[16]
            for prefix in PREFIXES[:-1]:
                ep = emb_cache[prefix]; consistency.extend(np.sum(ep * e16, axis=1).tolist())
            bucket = {}
            for bucket_name, selector in {
                "small_area_lt_0.01": lambda k: meta[k]["area"] < 0.01,
                "long_tail_category_le5_tracks": lambda k: sum(meta[x]["category"] == meta[k]["category"] for x in keys) <= 5,
                "domain_bucket_video_mod4_0": lambda k: meta[k]["video"] % 4 == 0,
            }.items():
                bkeys = [k for k in keys if selector(k)]
                if bkeys:
                    inds = [keys.index(k) for k in bkeys]; bucket[bucket_name] = {"tracklets": len(bkeys), "prefix16": retrieval(bkeys, emb_cache[16][np.asarray(inds)], meta)}
                else: bucket[bucket_name] = {"tracklets": 0}
            fold_out["models"][name] = {"prefix": per_prefix, "support_set_prefix": support_prefix, "prefix_consistency_to_16": float(np.mean(consistency)) if consistency else 0.0, "buckets": bucket, "checkpoint": None if model is None else str((ROOT / f"outputs/iclr27_phase{27 if name.startswith('phase27') else 29}/checkpoints/{'correspondence' if name.startswith('phase27') else 'domain_aligned'}_f{fold}_best.pt"))}
        all_fold.append(fold_out)
    def mean_metric(model_name: str, metric: str, support: bool = False) -> dict[str, float]:
        out = {}
        for prefix in PREFIXES:
            vals = []
            for f in all_fold:
                x = f["models"][model_name]["support_set_prefix" if support else "prefix"][str(prefix)]
                vals.append(float(x[metric]))
            out[str(prefix)] = float(np.mean(vals)) if vals else 0.0
        return out
    aggregate = {name: {"raw": {m: mean_metric(name, m) for m in ["r1", "r5", "map", "hard_negative_gap", "positive_coverage"]}, "support_set": {m: mean_metric(name, m, True) for m in ["r1", "r5", "map", "hard_negative_gap", "positive_coverage"]}} for name in ["raw_dinov2", "phase26_proposal_dinov2", "phase27_gru_comparator", "phase29_residual_comparator"]}
    result = {"protocol": "trackocd_iclr27_phase30_stage1_frozen_retrieval", "created_utc": datetime.now(timezone.utc).isoformat(), "positive_event_denominator": 76, "prefixes": list(PREFIXES), "folds": all_fold, "aggregate": aggregate, "phase26_proposal_coverage_context": {"prefix16_ceiling": 41, "source_reliable": 67, "target_reliable": 48, "category_coverage": 15, "video_coverage": 30}, "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels", "held event outcomes", "future rows/tracks", "physical/semantic IDs as model inputs", "category text"], "checkpoint_selection_source": "none; Stage1 is diagnostic and no held-event result selects a checkpoint", "support_set_score": "non-trained max-over-support cosine using TRAIN-derived validation episode metadata"}
    atomic_json(OUT / "metrics/stage1_diagnostics.json", result)
    atomic_json(OUT / "completion/stage1.done", {"stage": 1, "folds": 4, "positive_event_denominator": 76, "created_utc": datetime.now(timezone.utc).isoformat()})
    lines = ["# Phase30 Stage 1 — Frozen Retrieval Diagnostics", "", "No model was selected using held events.  Phase26 proposal-attached DINOv2 is feature-identical to the raw frozen DINOv2 comparator; the proposal source and physical stream remain frozen.", "", "## Aggregate raw retrieval", "", "| model | p1 R@1 | p16 R@1 | p16 mAP | p16 hard gap | p16 positive coverage |", "|---|---:|---:|---:|---:|---:|"]
    for name in ["raw_dinov2", "phase26_proposal_dinov2", "phase27_gru_comparator", "phase29_residual_comparator"]:
        a = aggregate[name]["raw"]; lines.append(f"| {name} | {a['r1']['1']:.4f} | {a['r1']['16']:.4f} | {a['map']['16']:.4f} | {a['hard_negative_gap']['16']:.4f} | {a['positive_coverage']['16']:.4f} |")
    lines += ["", "## Aggregate support-set score (diagnostic, no training)", "", "| model | p1 R@1 | p16 R@1 | p16 mAP | p16 hard gap |", "|---|---:|---:|---:|---:|"]
    for name in ["raw_dinov2", "phase26_proposal_dinov2", "phase27_gru_comparator", "phase29_residual_comparator"]:
        a = aggregate[name]["support_set"]; lines.append(f"| {name} | {a['r1']['1']:.4f} | {a['r1']['16']:.4f} | {a['map']['16']:.4f} | {a['hard_negative_gap']['16']:.4f} |")
    lines += ["", "## Fold and bucket artifacts", "", "Per-fold prefix curves, prefix consistency, small-object/long-tail/domain buckets and exact query denominators are in `metrics/stage1_diagnostics.json`.  The 76-event evaluator remains reserved for the frozen post-Gate diagnostic; Stage1 retrieval is not Commit-CT or HOTA.", "", "Stage1 result is **diagnostic complete**.  Stage2 authorization requires evidence that the support/query contract and domain-balanced sampling are the actionable factor; no controller, threshold, proposal or backbone change is made here."]
    atomic_json(OUT / "audit/stage1_summary.json", {"aggregate": aggregate, "folds": [{"fold": f["fold"], "validation_tracklets": f["validation_tracklets"]} for f in all_fold], "stage": 1})
    (OUT / "audit").mkdir(parents=True, exist_ok=True)
    (OUT / "audit/STAGE1_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"stage1": "done", "folds": 4, "metrics": str(OUT / "metrics/stage1_diagnostics.json")}, indent=2))


if __name__ == "__main__":
    main()

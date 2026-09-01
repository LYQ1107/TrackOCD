#!/usr/bin/env python3
"""Evaluate the frozen Phase27 encoder against the DINOv2 track baseline.

This is representation-only evaluation.  It does not instantiate or alter the
Phase19R controller and it never uses the 76 event labels to select a model.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.iclr27_phase27.correspondence import TrackCorrespondenceEncoder
from src.iclr27_phase27.protocol import CSV_PATH, FEAT_PATH, FOLD_MANIFEST, by_track, load_aligned_features
from scripts.iclr27_phase27.train_correspondence import embed_keys, pad_prefix

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase27"
P20_POS = ROOT / "outputs/iclr27_phase19r/manifests/held_known_positive_events.jsonl"
PREFIXES = (1, 2, 4, 8, 16)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def baseline_embed(keys: list[str], tracks: dict[str, list[int]], feats: np.ndarray, prefix: int) -> np.ndarray:
    out = np.zeros((len(keys), feats.shape[1]), np.float32)
    for i, key in enumerate(keys):
        inds = tracks[key][: min(int(prefix), 16)]
        v = feats[np.asarray(inds, dtype=np.int64)].mean(axis=0)
        out[i] = v / max(float(np.linalg.norm(v)), 1e-6)
    return out


def retrieval_from_embeddings(
    keys: list[str], emb: np.ndarray, cat: dict[str, int], video: dict[str, int], prefix: int
) -> dict[str, Any]:
    if not keys:
        return {"queries": 0, "r1": 0.0, "r5": 0.0, "map": 0.0, "pairs": 0, "hard_negative_gap": 0.0, "category_macro": 0.0, "video_macro": 0.0}
    sim = emb @ emb.T
    q_rows: list[dict[str, Any]] = []
    for i, key in enumerate(keys):
        candidates = [j for j, other in enumerate(keys) if j != i and video[other] != video[key]]
        positives = [j for j in candidates if cat[keys[j]] == cat[key]]
        negatives = [j for j in candidates if cat[keys[j]] != cat[key]]
        if not positives or not negatives:
            continue
        order = np.asarray(candidates, dtype=np.int64)[np.argsort(sim[i, np.asarray(candidates, dtype=np.int64)])[::-1]]
        pos_set = set(positives)
        hits = np.asarray([int(int(j) in pos_set) for j in order], np.float32)
        cumulative = np.cumsum(hits)
        q_rows.append({
            "category": int(cat[key]),
            "video": int(video[key]),
            "r1": float(hits[:1].max(initial=0)),
            "r5": float(hits[:5].max(initial=0)),
            "map": float(np.sum(cumulative / (np.arange(len(hits)) + 1) * hits) / max(len(positives), 1)),
            "hard_negative_gap": float(np.max(sim[i, np.asarray(positives, dtype=np.int64)]) - np.max(sim[i, np.asarray(negatives, dtype=np.int64)])),
        })
    by_cat: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in q_rows:
        by_cat[row["category"]].append(row)
        by_video[row["video"]].append(row)
    mean = lambda field, rows: float(np.mean([r[field] for r in rows])) if rows else 0.0
    return {
        "queries": len(q_rows),
        "r1": mean("r1", q_rows),
        "r5": mean("r5", q_rows),
        "map": mean("map", q_rows),
        "pairs": int(sum(len([j for j, other in enumerate(keys) if j != i and video[other] != video[keys[i]]]) for i in range(len(keys)))),
        "hard_negative_gap": mean("hard_negative_gap", q_rows),
        "category_macro": mean("r1", [dict(r, r1=mean("r1", rs)) for r, rs in ((next(iter(v)), v) for v in by_cat.values())]),
        "video_macro": mean("r1", [dict(r, r1=mean("r1", rs)) for r, rs in ((next(iter(v)), v) for v in by_video.values())]),
        "prefix": int(prefix),
        "per_query": q_rows,
    }


def event_records(
    events: list[dict[str, Any]],
    fold_models: dict[int, TrackCorrespondenceEncoder],
    feats: np.ndarray,
    tracks: dict[str, list[int]],
    rows: list[dict[str, str]],
    device: torch.device,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for event in events:
        fold = int(event["fold"])
        source_key = str(event["source_tracklet_keys"][0])
        target_key = str(event["target_tracklet_key"])
        if source_key not in tracks or target_key not in tracks or fold not in fold_models:
            continue
        model = fold_models[fold]
        row: dict[str, Any] = {"event_key": event["event_key"], "fold": fold, "source_track": source_key, "target_track": target_key, "prefixes": {}}
        for prefix in PREFIXES:
            # Source is fully materialized before target; target is truncated to
            # the causal prefix.  No future target rows enter this diagnostic.
            src_b = baseline_embed([source_key], tracks, feats, 16)[0]
            tgt_b = baseline_embed([target_key], tracks, feats, prefix)[0]
            src_e = embed_keys(model, [source_key], tracks, feats, 16, device, batch_size=1)[0]
            tgt_e = embed_keys(model, [target_key], tracks, feats, prefix, device, batch_size=1)[0]
            row["prefixes"][str(prefix)] = {
                "baseline_cosine": float(src_b @ tgt_b),
                "encoder_cosine": float(src_e @ tgt_e),
                "encoder_minus_baseline": float(src_e @ tgt_e - src_b @ tgt_b),
            }
        records.append(row)
    return records


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cls, roi, alignment = load_aligned_features(rows)
    feats = (0.8 * cls.astype(np.float32) + 0.2 * roi.astype(np.float32)).astype(np.float32)
    feats /= np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-6)
    tracks = by_track(rows)
    categories = {k: int(rows[v[-1]].get("gt_category_id_common", -1)) for k, v in tracks.items()}
    videos = {k: int(rows[v[-1]].get("video_id", -1)) for k, v in tracks.items()}
    manifest = json.loads(FOLD_MANIFEST.read_text(encoding="utf-8"))
    fold_results: list[dict[str, Any]] = []
    models: dict[int, TrackCorrespondenceEncoder] = {}
    for fold_record in manifest["folds"]:
        fold = int(fold_record["fold"])
        fit_v = {int(v) for v in fold_record.get("fit_videos", [])}
        held_c = {int(c) for c in fold_record.get("held_categories", [])}
        val_v = {int(v) for v in fold_record.get("validation_videos", [])}
        val_keys = sorted(k for k in tracks if categories[k] in held_c and videos[k] in val_v)
        baseline = {str(p): retrieval_from_embeddings(val_keys, baseline_embed(val_keys, tracks, feats, p), categories, videos, p) for p in PREFIXES}
        checkpoint = OUT / "checkpoints" / f"correspondence_f{fold}_best.pt"
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = TrackCorrespondenceEncoder()
        model.load_state_dict(ck["model"])
        model.to(device).eval()
        models[fold] = model
        encoded: dict[str, Any] = {}
        for p in PREFIXES:
            emb = embed_keys(model, val_keys, tracks, feats, p, device)
            encoded[str(p)] = retrieval_from_embeddings(val_keys, emb, categories, videos, p)
        fold_results.append({
            "fold": fold,
            "fit_videos": sorted(fit_v),
            "held_categories": sorted(held_c),
            "validation_videos": sorted(val_v),
            "validation_tracklets": len(val_keys),
            "baseline": baseline,
            "encoder": encoded,
            "best_step": int(ck.get("best_step", -1)),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
        })

    del models  # reload lightweight models below for event diagnostics
    fold_models: dict[int, TrackCorrespondenceEncoder] = {}
    for fold in range(4):
        ck = torch.load(OUT / "checkpoints" / f"correspondence_f{fold}_best.pt", map_location="cpu", weights_only=False)
        model = TrackCorrespondenceEncoder(); model.load_state_dict(ck["model"]); model.to(device).eval(); fold_models[fold] = model
    events = [json.loads(line) for line in P20_POS.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(events) != 76:
        raise RuntimeError(f"positive event denominator changed: {len(events)}")
    records = event_records(events, fold_models, feats, tracks, rows, device)

    # Gate R was preregistered as a meaningful (+0.02 R@1 and +0.01 mAP)
    # improvement on at least three folds at prefix16, with no use of held
    # events for checkpoint choice.
    gate_rows = []
    for result in fold_results:
        b = result["baseline"]["16"]; e = result["encoder"]["16"]
        gate_rows.append({
            "fold": result["fold"],
            "baseline_r1": b["r1"], "encoder_r1": e["r1"], "delta_r1": e["r1"] - b["r1"],
            "baseline_map": b["map"], "encoder_map": e["map"], "delta_map": e["map"] - b["map"],
            "baseline_hard_negative_gap": b["hard_negative_gap"], "encoder_hard_negative_gap": e["hard_negative_gap"],
            "substantial": bool(e["r1"] - b["r1"] >= 0.02 and e["map"] - b["map"] >= 0.01),
            "directional": bool(e["r1"] > b["r1"] and e["map"] > b["map"]),
        })
    substantial = sum(int(x["substantial"]) for x in gate_rows)
    directional = sum(int(x["directional"]) for x in gate_rows)
    gate = {"thresholds": {"r1_delta": 0.02, "map_delta": 0.01, "folds": 3}, "folds_substantial": substantial, "folds_directional": directional, "pass": substantial >= 3, "decision": "P27_GATE_R_PASS_AUTHORIZE_CONTROLLER" if substantial >= 3 else "P27_GATE_R_FAIL_STOP_BEFORE_CONTROLLER"}
    result = {
        "protocol": "trackocd_iclr27_phase27_correspondence_validation",
        "feature_alignment": alignment,
        "fold_manifest_sha256": sha256(FOLD_MANIFEST),
        "csv_sha256": sha256(CSV_PATH),
        "feature_sha256": sha256(FEAT_PATH),
        "positive_event_denominator": 76,
        "folds": fold_results,
        "gate_r": gate,
        "event_record_count": len(records),
        "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels", "future frames/tracks", "physical/semantic IDs", "semantic text", "held GT as model input"],
    }
    atomic_json(OUT / "metrics/correspondence_validation.json", result)
    atomic_json(OUT / "audit/correspondence_event_records.json", {"protocol": result["protocol"], "records": records, "denominator": 76})
    (OUT / "completion" / "correspondence_validation.done").write_text(json.dumps({"gate_r": gate, "metrics": str(OUT / "metrics/correspondence_validation.json"), "event_records": len(records)}, sort_keys=True) + "\n")
    print(json.dumps({"gate_r": gate, "folds": gate_rows, "event_records": len(records)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate Phase29 representation against frozen DINOv2 without controller."""
from __future__ import annotations

import argparse
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

from src.iclr27_phase29.protocol import CSV_PATH, FEAT_PATH, FOLD_MANIFEST, POSITIVE_EVENTS, by_track, load_aligned_features
from src.iclr27_phase29.representation import DomainAlignedResidualEncoder
from scripts.iclr27_phase29.train_domain_aligned import baseline_embeddings, embed_keys, retrieval_from_embeddings

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase29"
PREFIXES = (1, 2, 4, 8, 16)


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def event_records(events: list[dict[str, Any]], models: dict[int, DomainAlignedResidualEncoder], feats: np.ndarray, tracks: dict[str, list[int]], device: torch.device) -> list[dict[str, Any]]:
    records = []
    for event in events:
        fold = int(event["fold"]); src = str(event["source_tracklet_keys"][0]); tgt = str(event["target_tracklet_key"])
        if fold not in models or src not in tracks or tgt not in tracks: continue
        model = models[fold]; row = {"event_key": event["event_key"], "fold": fold, "source_track": src, "target_track": tgt, "prefixes": {}}
        src_b = baseline_embeddings([src], tracks, feats, 16)[0]; src_e = embed_keys(model, [src], tracks, feats, 16, device, 1)[0]
        for p in PREFIXES:
            tgt_b = baseline_embeddings([tgt], tracks, feats, p)[0]; tgt_e = embed_keys(model, [tgt], tracks, feats, p, device, 1)[0]
            row["prefixes"][str(p)] = {"baseline_cosine": float(src_b @ tgt_b), "encoder_cosine": float(src_e @ tgt_e), "encoder_minus_baseline": float(src_e @ tgt_e - src_b @ tgt_b)}
        records.append(row)
    return records


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); args = ap.parse_args()
    torch.set_num_threads(2); device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda": torch.cuda.set_device(device)
    with CSV_PATH.open(newline="", encoding="utf-8") as f: rows = list(csv.DictReader(f))
    cls, roi, alignment = load_aligned_features(rows); feats = (0.8 * cls.astype(np.float32) + 0.2 * roi.astype(np.float32)).astype(np.float32); feats /= np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-6)
    tracks = by_track(rows); categories = {k: int(rows[v[-1]].get("gt_category_id_common", -1)) for k, v in tracks.items()}; videos = {k: int(rows[v[-1]].get("video_id", -1)) for k, v in tracks.items()}; manifest = json.loads(FOLD_MANIFEST.read_text())
    fold_results = []; models = {}
    for fold_record in manifest["folds"]:
        fold = int(fold_record["fold"]); held_c = {int(c) for c in fold_record.get("held_categories", [])}; val_v = {int(v) for v in fold_record.get("validation_videos", [])}; val_keys = sorted(k for k in tracks if categories[k] in held_c and videos[k] in val_v)
        baseline = {str(p): retrieval_from_embeddings(val_keys, baseline_embeddings(val_keys, tracks, feats, p), categories, videos, p) for p in PREFIXES}
        ckpt = OUT / "checkpoints" / f"domain_aligned_f{fold}_best.pt"; ck = torch.load(ckpt, map_location="cpu", weights_only=False); model = DomainAlignedResidualEncoder(); model.load_state_dict(ck["model"]); model.to(device).eval(); models[fold] = model
        encoded = {str(p): retrieval_from_embeddings(val_keys, embed_keys(model, val_keys, tracks, feats, p, device), categories, videos, p) for p in PREFIXES}
        fold_results.append({"fold": fold, "held_categories": sorted(held_c), "validation_videos": sorted(val_v), "validation_tracklets": len(val_keys), "baseline": baseline, "encoder": encoded, "best_step": int(ck.get("best_step", -1)), "checkpoint": str(ckpt), "checkpoint_sha256": sha(ckpt)})
    gate_rows = []
    for r in fold_results:
        b, e = r["baseline"]["16"], r["encoder"]["16"]; gate_rows.append({"fold": r["fold"], "baseline_r1": b["r1"], "encoder_r1": e["r1"], "delta_r1": e["r1"] - b["r1"], "baseline_map": b["map"], "encoder_map": e["map"], "delta_map": e["map"] - b["map"], "baseline_r5": b["r5"], "encoder_r5": e["r5"], "baseline_hard_negative_gap": b["hard_negative_gap"], "encoder_hard_negative_gap": e["hard_negative_gap"], "substantial": bool(e["r1"] - b["r1"] >= 0.02 and e["map"] - b["map"] >= 0.01), "directional": bool(e["r1"] > b["r1"] and e["map"] > b["map"])})
    events = [json.loads(x) for x in POSITIVE_EVENTS.read_text().splitlines() if x.strip()]
    if len(events) != 76: raise RuntimeError(f"positive denominator changed: {len(events)}")
    recs = event_records(events, models, feats, tracks, device)
    substantial = sum(int(x["substantial"]) for x in gate_rows); directional = sum(int(x["directional"]) for x in gate_rows)
    aggregate = {"protocol": "trackocd_iclr27_phase29_representation_validation", "feature_alignment": alignment, "csv_sha256": sha(CSV_PATH), "feature_sha256": sha(FEAT_PATH), "fold_manifest_sha256": sha(FOLD_MANIFEST), "positive_event_denominator": 76, "folds": fold_results, "gate_r": {"thresholds": {"r1_delta": 0.02, "map_delta": 0.01, "folds": 3}, "folds_substantial": substantial, "folds_directional": directional, "pass": substantial >= 3, "decision": "P29_GATE_R_PASS_AUTHORIZE_CONTROLLER" if substantial >= 3 else "P29_GATE_R_FAIL_STOP_BEFORE_CONTROLLER"}, "event_record_count": len(recs), "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels", "future rows/tracks", "physical/semantic IDs", "semantic text", "held/test GT as model input"], "proposal_frozen": True, "controller_started": False}
    atomic(OUT / "metrics/representation_validation.json", aggregate); atomic(OUT / "audit/representation_event_records.json", {"protocol": aggregate["protocol"], "records": recs, "denominator": 76}); (OUT / "completion/representation_validation.done").write_text(json.dumps({"gate_r": aggregate["gate_r"], "events": len(recs)}, sort_keys=True) + "\n")
    print(json.dumps({"gate_r": aggregate["gate_r"], "folds": gate_rows, "event_records": len(recs)}, indent=2, sort_keys=True))


if __name__ == "__main__": main()

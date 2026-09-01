#!/usr/bin/env python3
"""Build lightweight Phase26 TRAIN-only source-head manifests."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from src.iclr27_phase26.protocol import CSV_PATH, P22_MANIFEST, normalized_gt

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "outputs/iclr27_phase26/manifests"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f: json.dump(value, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent)); os.close(fd)
    try:
        with open(tmp, "wb") as f: np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def sha256(path: Path) -> str:
    h = hashlib.sha256();
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def build(rows, indices, path, seed):
    pos = [i for i in indices if normalized_gt(rows[i]) is not None]; neg = [i for i in indices if normalized_gt(rows[i]) is None]
    rng = np.random.default_rng(seed); rng.shuffle(neg); neg = neg[:min(len(neg), max(2 * len(pos), 256))]
    chosen = np.asarray(pos + neg, np.int32); positive = np.asarray([1] * len(pos) + [0] * len(neg), np.float32); gt = np.zeros((len(chosen), 4), np.float32)
    for j, i in enumerate(pos): gt[j] = np.asarray(normalized_gt(rows[i]), np.float32)
    atomic_npz(path, row_idx=chosen, positive=positive, gt_box=gt)
    return {"path": str(path), "rows": int(len(chosen)), "positive_rows": int(len(pos)), "negative_rows": int(len(neg)), "sha256": sha256(path)}


def main() -> None:
    rows = list(csv.DictReader(CSV_PATH.open(newline="", encoding="utf-8"))); manifest = json.loads(P22_MANIFEST.read_text(encoding="utf-8")); OUT.mkdir(parents=True, exist_ok=True); folds = []
    for f in manifest["folds"]:
        fold = int(f["fold"]); fit_v, val_v = set(map(int, f["fit_videos"])), set(map(int, f["validation_videos"])); fit_c, held_c = set(map(int, f["fit_categories"])), set(map(int, f["held_categories"]))
        fit_idx = [i for i, r in enumerate(rows) if int(r["video_id"]) in fit_v and int(r.get("gt_category_id_common", -1)) in fit_c]
        val_idx = [i for i, r in enumerate(rows) if int(r["video_id"]) in val_v and int(r.get("gt_category_id_common", -1)) in held_c]
        fit = build(rows, fit_idx, OUT / f"source_fit_f{fold}.npz", 20260829 + fold); val = build(rows, val_idx, OUT / f"source_val_f{fold}.npz", 20260929 + fold)
        folds.append({"fold": fold, "fit_videos": sorted(fit_v), "validation_videos": sorted(val_v), "fit_categories": sorted(fit_c), "held_categories": sorted(held_c), "fit": fit, "validation": val})
    result = {"protocol": "trackocd_iclr27_phase26_source_head_train_manifest", "source_csv": str(CSV_PATH), "positive_labels": "TRAIN GT only", "feature_artifact_not_copied": True, "folds": folds, "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels", "future frames/tracks", "physical/semantic IDs"]}; atomic_json(OUT / "source_manifest.json", result); atomic_json(ROOT / "outputs/iclr27_phase26/completion/manifests.done", {"stage": "phase26_source_manifests", "folds": 4, "feature_artifact_not_copied": True})
    print(json.dumps({"folds": [{"fold": x["fold"], "fit": x["fit"]["rows"], "fit_pos": x["fit"]["positive_rows"], "fit_neg": x["fit"]["negative_rows"], "val": x["validation"]["rows"]} for x in folds]}, indent=2))


if __name__ == "__main__": main()

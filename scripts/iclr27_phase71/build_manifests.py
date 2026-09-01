#!/usr/bin/env python3
"""Materialise lightweight Phase71 TRAIN fold policies.

The Phase69 policy files are read-only lineage.  This script copies only the
video/category allow-lists and records a source hash; it never copies the LVIS
annotation or image store and never includes held/DEV/Q1 rows.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import tempfile
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "outputs/iclr27_phase69/manifests"
OUT = ROOT / "outputs/iclr27_phase71/manifests"
ANN = ROOT / "third_party/research_refs_phase4n/OVTR/data/lvis_clear_75_60.json"


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: pathlib.Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main() -> None:
    if not ANN.exists():
        raise FileNotFoundError(ANN)
    source_hash = sha(ANN)
    manifests = []
    for fold in range(4):
        src = SRC_ROOT / f"fold_{fold}_train.json"
        val_src = SRC_ROOT / f"fold_{fold}_val.json"
        meta_src = SRC_ROOT / f"fold_{fold}.json"
        if not src.exists() or not val_src.exists() or not meta_src.exists():
            raise FileNotFoundError(src)
        policy = json.loads(src.read_text())
        val_policy = json.loads(val_src.read_text())
        meta = json.loads(meta_src.read_text())
        # Explicitly preserve the original policy lists; sorting makes the
        # digest reproducible while the loader uses sets at runtime.
        train = {
            "phase": 71,
            "fold": fold,
            "split": "train_fit",
            "seed": 575700 + fold,
            "source_annotation": str(ANN.resolve()),
            "source_annotation_sha256": source_hash,
            "source_policy": str(src.resolve()),
            "source_policy_sha256": sha(src),
            "allowed_videos": sorted(int(x) for x in policy["allowed_videos"]),
            "allowed_categories": sorted(int(x) for x in policy["allowed_categories"]),
            "video_disjoint": True,
            "category_disjoint": True,
            "image_source": "OVTR/data/lvis_filtered_train_images.h5 (read-only)",
            "supervision_boundary": "TRAIN GT labels are loss metadata only; no held/DEV+/Q1/public labels",
            "parent_assignment": "frozen Q0 RuntimeTrackerBase bookkeeping",
            "row_key": ["video_id", "frame_id", "image_id", "proposal_local_id", "track_id"],
        }
        val = {
            "phase": 71,
            "fold": fold,
            "split": "train_disjoint_validation",
            "seed": 575700 + fold,
            "source_annotation": str(ANN.resolve()),
            "source_annotation_sha256": source_hash,
            "source_policy": str(val_src.resolve()),
            "source_policy_sha256": sha(val_src),
            "allowed_videos": sorted(int(x) for x in val_policy["allowed_videos"]),
            "allowed_categories": sorted(int(x) for x in val_policy["allowed_categories"]),
            "video_disjoint": True,
            "category_disjoint": True,
            "supervision_boundary": "TRAIN-only validation; no held/DEV+/Q1/public labels",
            "row_key": ["video_id", "frame_id", "image_id", "proposal_local_id", "track_id"],
        }
        atomic_json(OUT / f"fold_{fold}_train.json", train)
        atomic_json(OUT / f"fold_{fold}_val.json", val)
        # Keep a compact fold index for supervisors and audits.  It does not
        # replicate the annotation content.
        index = {
            "phase": 71,
            "fold": fold,
            "seed": 575700 + fold,
            "train": str((OUT / f"fold_{fold}_train.json").resolve()),
            "val": str((OUT / f"fold_{fold}_val.json").resolve()),
            "train_sha256": sha(OUT / f"fold_{fold}_train.json"),
            "val_sha256": sha(OUT / f"fold_{fold}_val.json"),
            "phase69_metadata_source": str(meta_src.resolve()),
            "phase69_metadata_sha256": sha(meta_src),
            "fit_categories": meta.get("fit_categories"),
            "held_categories": meta.get("held_categories"),
            "fit_videos": meta.get("fit_videos"),
            "held_videos": meta.get("held_videos"),
            "source_annotation_sha256": source_hash,
            "sealed_public_q1_accessed": False,
        }
        atomic_json(OUT / f"fold_{fold}.json", index)
        manifests.append(index)
    atomic_json(OUT / "inventory.json", {
        "phase": 71,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_annotation": str(ANN.resolve()),
        "source_annotation_sha256": source_hash,
        "folds": manifests,
        "protocol": "Q0-preserving TRAIN video/category-disjoint policy; metadata only",
        "sealed_public_q1_accessed": False,
    })
    print(json.dumps({"out": str(OUT), "source_annotation_sha256": source_hash, "folds": 4}, indent=2))


if __name__ == "__main__":
    main()

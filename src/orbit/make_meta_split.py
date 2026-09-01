#!/usr/bin/env python3
"""Deterministic meta-train/meta-dev class split for ORBIT development."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
OUT = ROOT / "outputs" / "orbit" / "splits"
TRAIN_FILE = ROOT / "data" / "trackocd_v1" / "pure" / "public" / "train_known_tracks.jsonl"


def main() -> None:
    cats = set()
    with open(TRAIN_FILE) as f:
        for line in f:
            if line.strip():
                cats.add(int(json.loads(line)["category_id"]))
    cats = sorted(cats)
    assert len(cats) == 48, len(cats)
    hashed = sorted(
        (hashlib.sha256(f"{c}orbit_meta_split_v1".encode()).hexdigest(), c)
        for c in cats
    )
    meta_dev = {c for _, c in hashed[:10]}
    meta_train = set(cats) - meta_dev
    OUT.mkdir(parents=True, exist_ok=True)
    for name, ids in (("meta_train_classes", meta_train), ("meta_dev_classes", meta_dev)):
        with open(OUT / f"{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class_id"])
            for c in sorted(ids):
                w.writerow([c])
    print("meta_train", len(meta_train), sorted(meta_train))
    print("meta_dev", len(meta_dev), sorted(meta_dev))


if __name__ == "__main__":
    main()

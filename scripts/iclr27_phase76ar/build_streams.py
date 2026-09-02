#!/usr/bin/env python3
"""Build Phase76AR streams and per-match caches from frozen inputs."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import tempfile
from pathlib import Path

from src.iclr27_phase75d.protocol import load_frozen_tracks
from src.iclr27_phase76ar.data import bank_payload, build_legal_fit, build_memory_mimic
from src.iclr27_phase76ar.pair_cache import build_pair_cache, cache_hash

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase76ar"
MANIFEST_ROOT = ROOT / "outputs/iclr27_phase30/manifests"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, default=None); args = ap.parse_args()
    folds = range(4) if args.fold is None else [int(args.fold)]
    table = load_frozen_tracks()
    summaries = []
    for fold in folds:
        manifest = MANIFEST_ROOT / f"episode_manifest_f{fold}.json"
        memory_fit = build_memory_mimic(manifest, fold, "fit", table)
        legal_fit = build_legal_fit(manifest, fold, "fit", table)
        memory_val = build_memory_mimic(manifest, fold, "val", table)
        legal_val = build_legal_fit(manifest, fold, "val", table)
        payload = {
            "phase": "Phase76AR", "fold": fold, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "manifest": str(manifest), "manifest_sha256": sha(manifest),
            "fit": bank_payload(memory_fit, legal_fit, manifest_sha256=sha(manifest), fold=fold, split="fit"),
            "val": bank_payload(memory_val, legal_val, manifest_sha256=sha(manifest), fold=fold, split="val"),
            "table_hashes": {"csv": table.csv_sha256, "features": table.feature_sha256},
        }
        stream_path = OUT / "banks" / f"streams_f{fold}.json"
        atomic_json(stream_path, payload)
        all_banks = memory_fit + memory_val + legal_fit + legal_val
        cache_path = OUT / "banks" / f"pair_cache_f{fold}.json"
        build_pair_cache(all_banks, table, cache_path)
        summaries.append({
            "fold": fold, "manifest_sha256": sha(manifest),
            "fit_counts": {"memory_mimic": len(memory_fit), "legal_fit": len(legal_fit)},
            "val_counts": {"memory_mimic": len(memory_val), "legal_fit": len(legal_val)},
            "stream_path": str(stream_path), "stream_sha256": sha(stream_path),
            "pair_cache": str(cache_path), "pair_cache_sha256": cache_hash(cache_path),
            "pair_cache_size_bytes": cache_path.stat().st_size,
        })
    atomic_json(OUT / "audit/build_summary.json", {"phase": "Phase76AR", "folds": summaries, "source": "Phase30 manifests + frozen Phase75D table", "sealed_accessed": False})
    atomic_json(OUT / "completion/build_streams.done", {"phase": "Phase76AR", "folds": [x["fold"] for x in summaries], "created_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    print(json.dumps({"phase": "Phase76AR", "folds": summaries}, sort_keys=True))


if __name__ == "__main__": main()

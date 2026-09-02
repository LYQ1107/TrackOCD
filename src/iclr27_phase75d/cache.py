"""Atomic cache-key and matrix helpers for bounded Phase75D runs."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def cache_key(*, feature_sha256: str, csv_sha256: str, fold_manifest_sha256: str, prefix: int, method_version: str, query_key: str, candidate_bank_hash: str) -> str:
    payload = {
        "feature_sha256": feature_sha256, "csv_sha256": csv_sha256,
        "fold_manifest_sha256": fold_manifest_sha256, "prefix": int(prefix),
        "method_version": method_version, "query_key": query_key,
        "candidate_bank_hash": candidate_bank_hash,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    os.close(fd)
    try:
        np.save(tmp, np.asarray(value, dtype=np.float32), allow_pickle=False)
        os.replace(tmp + ".npy", path)
    finally:
        for p in (tmp, tmp + ".npy"):
            if os.path.exists(p): os.unlink(p)

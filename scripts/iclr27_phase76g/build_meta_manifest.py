#!/usr/bin/env python3
"""Build deterministic meta-holdout groups within each TRAIN fit fold."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/iclr27_phase76s/examples"
OUT = ROOT / "outputs/iclr27_phase76g"


def atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def main() -> None:
    summaries = []
    for fold in range(4):
        source = SOURCE / f"examples_f{fold}.json"
        payload = json.loads(source.read_text())
        categories = sorted({int(row["category"]) for row in payload["fit"]})
        cat_group = {str(cat): int(i % 4) for i, cat in enumerate(categories)}
        groups = {str(i): [] for i in range(4)}
        for index, row in enumerate(payload["fit"]):
            groups[str(cat_group[str(int(row["category"]))])].append(index)
        val_groups = {str(i): [] for i in range(4)}
        for index, row in enumerate(payload["val"]):
            # Validation rows are never used for fitting; this is only a
            # deterministic diagnostic partition preserving the full val set.
            val_groups[str(index % 4)].append(index)
        manifest = {
            "phase": "Phase76G", "fold": fold, "source_examples": str(source.resolve()),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "fit_count": len(payload["fit"]), "val_count": len(payload["val"]),
            "fit_categories": categories, "category_to_group": cat_group,
            "fit_indices_by_group": groups, "val_indices_by_diagnostic_group": val_groups,
            "group_rule": "sorted fit categories; category ordinal modulo 4",
            "fit_only": True, "sealed_accessed": False, "public_or_dev_accessed": False,
            "forbidden_inference_inputs": ["category", "semantic_id", "physical_id", "text", "future", "held/DEV+/Q1/public-new/sealed labels"]
        }
        path = OUT / "manifests" / f"meta_manifest_f{fold}.json"; atomic(path, manifest)
        summaries.append({"fold": fold, "fit_count": len(payload["fit"]), "val_count": len(payload["val"]), "fit_categories": categories, "group_counts": {key: len(value) for key, value in groups.items()}, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "path": str(path)})
    summary = {"phase": "Phase76G", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "folds": summaries, "sealed_accessed": False, "public_or_dev_accessed": False}
    atomic(OUT / "audit/meta_manifest_summary.json", summary); atomic(OUT / "completion/meta_manifest.done", summary); print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__": main()

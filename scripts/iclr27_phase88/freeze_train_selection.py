#!/usr/bin/env python3
"""Freeze the completed C0v2 checkpoints using TRAIN-only validation metrics.

This helper deliberately has no access path to held/DEV+/Q1/public labels.  A
fold's final 20k checkpoint is selected from the completed TRAIN-disjoint
validation artifact for that same fold; the registered Phase19R selection
score is the primary key and deterministic safety keys are only tie-breakers.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase88"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def main() -> None:
    prereg = json.loads((OUT / "audit/preregistration.json").read_text())
    rows: list[dict] = []
    for fold in range(4):
        tag = f"c0v2_fix2_formal_f{fold}"
        done = OUT / "completion" / f"{tag}.done"
        ckpt = OUT / "checkpoints" / f"{tag}.pt"
        val_dir = OUT / "validation" / f"{tag}_val"
        val_metrics = val_dir / "final_metrics.json"
        val_done = val_dir / "final.done"
        if not (done.exists() and ckpt.exists() and val_done.exists() and val_metrics.exists()):
            raise RuntimeError(f"incomplete freeze inputs for fold {fold}")
        validation = json.loads(val_metrics.read_text())
        metrics = validation["metrics"]
        # The sharded evaluator predates this freeze helper and does not put a
        # split field in its final JSON.  Its input path is the immutable
        # `val_events_v2_fix2_f*.jsonl` manifest, so record that provenance
        # explicitly rather than silently treating an arbitrary JSON as val.
        split = "TRAIN_disjoint_validation"
        if validation.get("public_dev_q1_sealed_accessed") or validation.get("future_rows_or_tracks") or validation.get("ids_or_text_as_model_input"):
            raise RuntimeError(f"contract boundary violation in fold {fold} validation")
        train_metrics_path = OUT / "metrics" / f"{tag}.json"
        if not train_metrics_path.exists():
            raise RuntimeError(f"missing training metrics for fold {fold}")
        train_metrics = json.loads(train_metrics_path.read_text())
        step = int(train_metrics.get("updates", -1))
        if step != 20000:
            raise RuntimeError(f"fold {fold} checkpoint is not the registered 20k endpoint: {step}")
        if train_metrics.get("checkpoint_sha256") != sha(ckpt):
            raise RuntimeError(f"checkpoint hash mismatch for fold {fold}")
        rows.append({
            "fold": fold,
            "tag": tag,
            "checkpoint": str(ckpt.resolve()),
            "checkpoint_sha256": sha(ckpt),
            "step": step,
            "validation": str(val_metrics.resolve()),
            "validation_sha256": sha(val_metrics),
            "train_metrics": str(train_metrics_path.resolve()),
            "train_metrics_sha256": sha(train_metrics_path),
            "selection_score": float(metrics["selection_score"]),
            "existing_f1_macro": float(metrics["existing_f1_macro"]),
            "commit_ct_recall": float(metrics["commit_ct"]["recall"]),
            "negative_false_merge_rate": float(metrics["negative_false_merge_rate"]),
            "premature_rate": float(metrics["premature_rate"]),
            "duplicate_births": int(metrics["duplicate_births"]),
            "category_coverage": int(metrics["category_coverage"]),
            "video_coverage": int(metrics["video_coverage"]),
            "known_micro": float(metrics["known_micro"]),
            "known_macro": float(metrics["known_macro"]),
            "split": split,
        })
    # The primary score and all tie-breakers are defined from the already
    # materialized TRAIN validation JSON; no held result is consulted.
    best = max(rows, key=lambda r: (
        r["selection_score"], r["existing_f1_macro"], r["commit_ct_recall"],
        -r["negative_false_merge_rate"], -r["premature_rate"], -r["duplicate_births"],
        -r["fold"],
    ))
    manifest = {
        "schema_version": "trackocd.phase88.c0v2_train_selection.v1",
        "phase": 88,
        "status": "FROZEN_TRAIN_ONLY",
        "selection_rule": {
            "primary": "max Phase19R metrics.selection_score on same-fold TRAIN_disjoint_validation",
            "tie_breakers": [
                "max existing_f1_macro", "max commit_ct.recall", "min negative_false_merge_rate",
                "min premature_rate", "min duplicate_births", "min fold index",
            ],
            "held_or_public_metrics_used": False,
        },
        "registered_formal_endpoint": {"updates": 20000, "event_tag": "fix2"},
        "folds": rows,
        "highest_train_validation_fold": best["fold"],
        "highest_train_validation_checkpoint": best["checkpoint"],
        "input_manifest_sha256": "b70096c7302bf7a19b05423c00cf90e1c636926d0e895cf4be30a9d6546a33fd",
        "shared_memmap_manifest_sha256": "e42524d1895921b55aee9c639e597fc411a75f272275bfc7986f260cdb5c1a7c",
        "public_dev_q1_sealed_accessed": False,
        "future_rows_or_tracks": False,
        "ids_or_text_as_model_input": False,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    atomic_json(OUT / "audit/c0v2_train_selection.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

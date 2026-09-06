#!/usr/bin/env python3
"""Recompute frozen Phase86 U2 validation with dropout disabled."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "iclr27_phase87"
CK = Path("/data2/usr_for_deadline/trackocd_phase86/project_outputs/checkpoints")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.iclr27_phase86.train_u2 import load, metrics
from src.iclr27_phase86.set_aware_relation import SetAwareResidualReranker


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atom(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def main() -> None:
    manifest, features, offsets, targets, metadata, contexts = load()
    rows = []
    tags = [(f"u2_cv_f{fold}", fold, False) for fold in range(3)] + [(f"u2_balanced_cv_f{fold}", fold, True) for fold in range(3)]
    for tag, fold, balanced in tags:
        checkpoint = CK / f"{tag}.pt"
        payload = torch.load(checkpoint, map_location="cpu")
        model = SetAwareResidualReranker()
        model.load_state_dict(payload["model"])
        model.eval()
        fit = [int(v) for v in manifest["folds"][str(fold)]["fit_groups"]]
        validation = [int(v) for v in manifest["folds"][str(fold)]["validation_groups"]]
        train_metrics = metrics(model, fit, features, offsets, targets, contexts, payload["mean"], payload["std"])
        validation_metrics = metrics(model, validation, features, offsets, targets, contexts, payload["mean"], payload["std"])
        rows.append({"tag": tag, "fold": fold, "balanced_group_sampling": balanced, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha(checkpoint), "model_eval": True, "fit_metrics": train_metrics, "validation_metrics": validation_metrics})
    p16 = {"rows": rows, "all_validation_zero_rescue": all(row["validation_metrics"]["rescue"] == 0 for row in rows), "all_validation_zero_harm": all(row["validation_metrics"]["harm"] == 0 for row in rows), "decision": "U2_COMPRESSED_FEATURE_COLLAPSE_CONFIRMED" if all(row["validation_metrics"]["rescue"] == 0 and row["validation_metrics"]["harm"] == 0 for row in rows) else "U2_EVAL_MODE_CHANGES_RESULT", "model_eval": True, "dropout_disabled": True, "phase86_checkpoints_retrained": False, "public_dev_q1_sealed_accessed": False}
    atom(OUT / "audit" / "phase86_u2_evalmode_correction.json", p16)
    atom(OUT / "completion" / "phase86_u2_evalmode_correction.done", {"status": "DONE", "decision": p16["decision"]})
    print(json.dumps(p16, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

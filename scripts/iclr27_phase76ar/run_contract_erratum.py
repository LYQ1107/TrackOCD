#!/usr/bin/env python3
"""Write the Phase76A implementation erratum without mutating history."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase76ar/audit"


def atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    code = ROOT / "src/iclr27_phase76a/relation_model.py"
    erratum = {
        "phase": "Phase76AR", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "historical_phase76a_status_unchanged": True,
        "dual_stream_not_implemented": True,
        "per_match_quality_not_implemented": True,
        "fold0_checkpoint_subset_bias": True,
        "confidence_not_bank_aware": True,
        "delta_unbounded": True,
        "safety_negative_coverage_partial": True,
        "hard_negative_prefix_mismatch": True,
        "evidence": {
            "source": str(code), "source_sha256": sha(code),
            "single_stream_expression": "fit = load_banks(...); idx = order[(step-1)%len(order)]; bank=fit[idx]",
            "quality_expression": "quality(summary[5 scalar fields]) -> one scalar pooled weight",
            "validation_expression": "evaluate_banks(... limit=validation_limit) uses banks[:limit]",
            "safety_expression": "_margin selects top8 raw negatives",
        },
        "repair_namespace": "src/iclr27_phase76ar, scripts/iclr27_phase76ar, outputs/iclr27_phase76ar",
        "forbidden_actions": ["modify historical Phase76A status/report", "StateMemory/controller/sealed evaluation", "DEV+/Q1/public-new access"],
    }
    atomic(OUT / "phase76a_contract_erratum.json", erratum)
    atomic(OUT / "preregistration.json", json.loads((ROOT / "configs/iclr27_phase76ar/preregistration.json").read_text()))
    print(json.dumps({"erratum": str(OUT / "phase76a_contract_erratum.json"), "source_sha256": erratum["evidence"]["source_sha256"]}, sort_keys=True))


if __name__ == "__main__": main()

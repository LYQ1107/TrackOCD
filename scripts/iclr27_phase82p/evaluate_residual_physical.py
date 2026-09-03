#!/usr/bin/env python3
"""Cheap, post-inference physical safety proxy for a Phase82P lineage."""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "scripts/iclr27_phase81p/evaluate_physical_replay.py"
NATIVE = Path("/data2/usr_for_deadline/trackocd_phase75b/event_full_sequence_repair2/native_lineage.jsonl")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp"); tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"); os.replace(tmp, path)


def load_old() -> Any:
    spec = importlib.util.spec_from_file_location("phase81_physical_readonly", OLD)
    if spec is None or spec.loader is None: raise RuntimeError(f"cannot import {OLD}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--replay", type=Path, required=True); ap.add_argument("--tag", default="residual"); args = ap.parse_args()
    old = load_old(); gt = old.load_gt(); learned = read_jsonl(args.replay); native = old.load_native()
    result = {"schema_version": "trackocd.phase82p.physical_proxy.v1", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "replay": str(args.replay), "replay_sha256": __import__("hashlib").sha256(args.replay.read_bytes()).hexdigest(), "native": str(NATIVE), "native_sha256": __import__("hashlib").sha256(NATIVE.read_bytes()).hexdigest(), "learned": old.physical_summary(learned, gt), "q0_native": old.physical_summary(native, gt), "definition": "Phase81 post-inference highest-IoU GT join; cheap proxy, not TrackEval", "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_as_model_input": False}
    atomic_json(ROOT / "outputs/iclr27_phase82p/metrics" / f"physical_{args.tag}.json", result); print(json.dumps({"learned": result["learned"], "q0_native": result["q0_native"]}, indent=2, sort_keys=True))


if __name__ == "__main__": main()

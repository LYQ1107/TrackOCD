#!/usr/bin/env python3
"""Post-inference physical proxy in the Phase82R output namespace."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "scripts/iclr27_phase81p/evaluate_physical_replay.py"
NATIVE = Path("/data2/usr_for_deadline/trackocd_phase75b/event_full_sequence_repair2/native_lineage.jsonl")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp"); tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"); os.replace(tmp, path)


def load_old() -> Any:
    spec = importlib.util.spec_from_file_location("phase82r_physical_readonly", OLD)
    if spec is None or spec.loader is None: raise RuntimeError(f"cannot import {OLD}")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--replay", type=Path, required=True); ap.add_argument("--tag", default="balanced"); args = ap.parse_args()
    old = load_old(); gt = old.load_gt(); learned = [json.loads(line) for line in args.replay.read_text(encoding="utf-8").splitlines() if line.strip()]; native = old.load_native()
    result = {"schema_version": "trackocd.phase82r.physical_proxy.v1", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "replay": str(args.replay), "replay_sha256": hashlib.sha256(args.replay.read_bytes()).hexdigest(), "native": str(NATIVE), "native_sha256": hashlib.sha256(NATIVE.read_bytes()).hexdigest(), "learned": old.physical_summary(learned, gt), "q0_native": old.physical_summary(native, gt), "definition": "post-inference highest-IoU GT join; cheap diagnostic proxy, not TrackEval", "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_as_model_input": False}
    atomic_json(ROOT / "outputs/iclr27_phase82r/metrics" / f"physical_{args.tag}.json", result); print(json.dumps({"learned": result["learned"], "q0_native": result["q0_native"]}, indent=2, sort_keys=True))


if __name__ == "__main__": main()

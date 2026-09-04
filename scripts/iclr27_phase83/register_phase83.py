#!/usr/bin/env python3
"""Write a reproducible Phase83 registration/status snapshot.

This script is deliberately read-only with respect to earlier phases.  Large
artifacts remain on /data2 through the Phase83 output symlink.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase83"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def snapshot() -> dict[str, object]:
    free = subprocess.run(["free", "-b"], capture_output=True, text=True, check=False)
    disk = subprocess.run(["df", "-B1", "/data1", "/data2"], capture_output=True, text=True, check=False)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    return {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cwd": str(Path.cwd()), "pid": os.getpid(),
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip(),
        "free_h": free.stdout, "disk": disk.stdout, "gpu": gpu.stdout,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    refs = {
        "corrected_csv": ROOT / "outputs/iclr27_phase17r/csv/public_rows_corrected.csv",
        "phase15s_features": ROOT / "outputs/iclr27_phase15s/features/public_cls_roi.npz",
        "native_lineage": Path("/data2/usr_for_deadline/trackocd_phase75b/event_full_sequence_repair2/native_lineage.jsonl"),
        "native_dino": ROOT / "outputs/iclr27_phase82r/features/native_dinov2_corrected_r1.npz",
        "q0_dino": ROOT / "outputs/iclr27_phase82r/features/q0_dinov2_corrected_r1.npz",
        "positive_events": ROOT / "outputs/iclr27_phase19r/manifests/held_known_positive_events.jsonl",
        "negative_events": ROOT / "outputs/iclr27_phase19r/manifests/held_known_negative_events.jsonl",
        "evaluator_join": ROOT / "outputs/iclr27_phase74s/manifests/evaluator_join_v2.jsonl",
        "phase30_manifests": ROOT / "outputs/iclr27_phase30/manifests",
    }
    inputs = {}
    for name, path in refs.items():
        if path.is_file():
            inputs[name] = {"path": str(path.resolve()), "sha256": sha(path), "bytes": path.stat().st_size}
        elif path.exists():
            inputs[name] = {"path": str(path.resolve()), "type": "directory"}
        else:
            inputs[name] = {"path": str(path), "missing": True}
    reg = {
        "schema_version": "trackocd.phase83.registration.v1",
        "phase": "Phase83", "route_a": "physical_to_raw_R_then_unchanged_C",
        "route_b": "O_support_v1_train_only_router",
        "window_start_utc": "2026-09-04T07:43:07Z", "window_deadline_utc": "2026-09-04T17:43:07Z",
        "immutable_protocol": {"positive_events": 76, "negative_events": 76, "prefixes": [1,2,4,8,16], "row_rule": "assigned == 1 AND transformed/event row IoU >= 0.5", "same_candidate_universe": True},
        "boundaries": {"train_only_supervision": True, "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_as_model_input": False, "category_text_as_input": False, "old_evaluator_modified": False},
        "inputs": inputs,
        "resource_snapshot": snapshot(),
        "gates": {"R83": "raw temporal-mean R improvement with safety", "O83": "versioned support assignment vs frozen O", "C83": "NOT_RUN until R/O evidence", "sealed": "NOT_RUN"},
        "status": "REGISTERED_STAGE0_AUDIT",
    }
    atomic_json(OUT / "audit/registration.json", reg)
    atomic_json(OUT / "status.json", {"phase": "Phase83", "status": "REGISTERED_STAGE0_AUDIT", "next_action": "run support assignment callgraph and 76-event taxonomy", "inputs": inputs, "resource_event": reg["resource_snapshot"], "public_dev_q1_sealed_accessed": False})


if __name__ == "__main__":
    main()

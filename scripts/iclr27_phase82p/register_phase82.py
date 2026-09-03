#!/usr/bin/env python3
"""Register the Phase82P+ run and its frozen, causal input contract.

The registration is deliberately written once at run start.  It records the
actual host clock, resource preflight, lineage hashes and the ten-hour
deadline without touching any previous phase's artifacts.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase82p"
AUDIT = OUT / "audit"
COMP = OUT / "completion"

FROZEN = {
    "q0_native_lineage": Path("/data2/usr_for_deadline/trackocd_phase75b/event_full_sequence_repair2/native_lineage.jsonl"),
    "q0_native_frames": Path("/data2/usr_for_deadline/trackocd_phase75b/event_full_sequence_repair2/native_lineage.frames.jsonl"),
    "positive_events": ROOT / "outputs/iclr27_phase19r/manifests/held_known_positive_events.jsonl",
    "negative_events": ROOT / "outputs/iclr27_phase19r/manifests/held_known_negative_events.jsonl",
    "evaluator_join": ROOT / "outputs/iclr27_phase74s/manifests/evaluator_join_v2.jsonl",
    "corrected_rows": ROOT / "outputs/iclr27_phase17r/csv/public_rows_corrected.csv",
    "q0_train_stream": ROOT / "outputs/iclr27_phase4t/train_stream/teta/tao_track.json",
    "q0_checkpoint": ROOT / "checkpoints/ovtr/ovtr_5_frame.pth",
}


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.STDOUT, timeout=30).strip()
    except Exception as exc:  # resource audit must remain machine-readable
        return f"ERROR: {type(exc).__name__}: {exc}"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    start = dt.datetime.now(dt.timezone.utc)
    deadline = start + dt.timedelta(hours=10)
    output_link = OUT.resolve()
    payload = {
        "schema_version": "trackocd.phase82p.registration.v1",
        "phase": "Phase82P+",
        "status": "REGISTERED",
        "start_time_utc": start.isoformat(),
        "deadline_utc": deadline.isoformat(),
        "duration_hours": 10,
        "cwd": str(ROOT),
        "git_head": run(["git", "rev-parse", "HEAD"]),
        "git_origin_main": run(["git", "ls-remote", "origin", "refs/heads/main"]),
        "branch": run(["git", "branch", "--show-current"]),
        "output_dir": str(output_link),
        "output_link_target": os.path.realpath(OUT),
        "resource_preflight": {
            "free_h": run(["free", "-h"]),
            "nvidia_smi": run(["nvidia-smi"]),
            "process_count": run(["bash", "-lc", "ps -e --no-headers | wc -l"]),
            "disk_data1": run(["df", "-h", "/data1"]),
            "disk_data2": run(["df", "-h", "/data2"]),
            "gpu_policy": "use only CUDA devices 4,5,6,7; one bounded worker per fold; preserve >=25% RAM",
        },
        "frozen_inputs": {str(k): {"path": str(v), "exists": v.is_file(), "sha256": sha256(v)} for k, v in FROZEN.items()},
        "data_boundary": {
            "inference_forbidden": ["category_name", "category_text", "semantic_id", "physical_id_feature", "future_frame", "future_track", "held_gt", "DEV+", "Q1", "public_new_model_labels"],
            "train_supervision_only": ["public TRAIN GT track/category labels for target construction; never serialized into inference tensors"],
            "q0_anchor": "proposal boxes, base score, frame order and non-birth continuation rows remain frozen",
            "candidate_policy": "causal dormant fragments <=16 frames, K=8 history, max 16 deterministic candidates",
        },
        "next_action": "run strict Phase75B O parity wrapper and build per-video residual manifest before any training",
    }
    atomic_json(AUDIT / "phase82_registration.json", payload)
    atomic_json(AUDIT / "status.json", payload)
    COMP.mkdir(parents=True, exist_ok=True)
    (COMP / "phase82_registered.done").write_text(start.isoformat() + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

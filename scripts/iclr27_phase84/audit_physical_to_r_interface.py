#!/usr/bin/env python3
"""Emit a read-only physical-lineage -> frozen Phase75D-R callgraph audit."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase84/audit/physical_to_r_interface_callgraph.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name): os.unlink(tmp_name)


def main() -> None:
    csv_path = ROOT / "data/iclr27_phase19r/sources/public_rows_corrected.csv"
    feat_path = ROOT / "data/iclr27_phase19r/sources/public_cls_roi.npz"
    manifest_dir = ROOT / "outputs/iclr27_phase30/manifests"
    native = Path("/data2/usr_for_deadline/trackocd_phase83/a2_full/native_lineage.jsonl")
    native_feat = Path("/data2/usr_for_deadline/trackocd_phase83/a2_dino_full_r1/merged/native_dinov2.npz")
    phase75d = ROOT / "src/iclr27_phase75d/protocol.py"
    phase75d_runner = ROOT / "scripts/iclr27_phase75d/run_pairwise_r.py"
    value = {
        "phase": "Phase84",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "read_only": True,
        "frozen_r_contract": {
            "query_count": 984,
            "prefixes": [1, 2, 4, 8, 16],
            "candidate_rule": "all validation tracks except self and same video",
            "candidate_order": "sorted frozen track keys / manifest order",
            "same_video_exclusion": True,
            "row_key": "video_id:image_id:proposal_local_id:track_id:image_id (CSV row_key source retained)",
            "raw_feature": "L2-normalized 0.8*DINOv2 CLS + 0.2*ROI, 768-D",
            "semantic_inputs_forbidden": ["category", "text", "semantic_id", "physical_id", "future", "held labels"],
        },
        "physical_stream_callgraph": [
            {"stage": "native_q0", "script": "Phase83 run_a2_full_q0.py", "output": str(native), "role": "raw class-agnostic proposal/physical rows"},
            {"stage": "causal_reassociation", "new_phase84": True, "script": "scripts/iclr27_phase84/run_full_temporal_physical.py", "output": "outputs/iclr27_phase84/physical/full_temporal_lineage.jsonl", "role": "canonical-root unions over dormant native fragments"},
            {"stage": "row_adapter", "new_phase84": True, "script": "scripts/iclr27_phase84/build_physical_r_adapter.py", "output": "outputs/iclr27_phase84/manifests/physical_r_adapter.json", "role": "map canonical root membership to frozen public R rows without GT"},
            {"stage": "r_evaluator", "new_phase84": True, "script": "scripts/iclr27_phase84/evaluate_physical_r.py", "output": "outputs/iclr27_phase84/metrics/physical_r_q0_adapter.json", "role": "same denominator/candidate order and R metrics"},
        ],
        "inputs": {
            "public_csv": {"path": str(csv_path.resolve()), "sha256": sha256(csv_path)},
            "public_feature_npz": {"path": str(feat_path.resolve()), "sha256": sha256(feat_path)},
            "native_lineage": {"path": str(native), "sha256": sha256(native)},
            "native_dinov2": {"path": str(native_feat), "sha256": sha256(native_feat)},
            "phase75d_protocol": {"path": str(phase75d.resolve()), "sha256": sha256(phase75d)},
            "phase75d_runner": {"path": str(phase75d_runner.resolve()), "sha256": sha256(phase75d_runner)},
            "fold_manifests": sorted({str(p.resolve()): sha256(p) for p in manifest_dir.glob("episode_manifest_f*.json")}.items()),
        },
        "required_membership_change": "canonical physical roots must merge native fragments before R aggregation; merely replacing vectors inside an unchanged public track is not sufficient",
        "causal_rules": {"dormant_only": True, "max_gap": 16, "observed_step_timing": True, "no_retroactive_future_merge": True, "same_frame_collision_safe": True},
        "public_dev_q1_sealed_accessed": False,
        "future_rows_or_tracks": False,
        "ids_as_model_input": False,
    }
    atomic_json(OUT, value)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__": main()

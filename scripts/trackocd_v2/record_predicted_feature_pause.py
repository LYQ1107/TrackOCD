#!/usr/bin/env python3
"""Record the protocol-driven pause of the large predicted feature build.

This is an audit record for an already completed intervention.  It does not
inspect or terminate processes; the worker identity and graceful termination
details are supplied by the execution log so the record remains stable.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, ensure_output_layout, sha256_file  # noqa: E402


def main() -> int:
    out = ensure_output_layout()
    source = ROOT / "data/tao_ow_ocd_v1/public/pred_track_stream.jsonl"
    relocation = OUTPUT_TARGET / "audit/predicted_cache_relocation.json"
    result = {
        "schema_version": "trackocd.v2.predicted_feature_pause.v1",
        "status": "PAUSED_BY_PROTOCOL_REPRIORITIZATION",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stage": "PREDICTED_STREAM_BUILD",
        "reason": (
            "Physical frontend selection is not frozen; full SimOWT/Q0 "
            "p1/p2/p4/p8/p16 DINOv2 precomputation is premature."
        ),
        "failure": False,
        "builder_identity": {
            "verified_as_this_trackocd_v2_task": True,
            "wrapper_pid": 30578,
            "builder_pid": 30579,
            "resource_tracker_pid": 30736,
            "worker_pid": 30737,
            "process_group_id": 30579,
            "command": [
                "/home/lwr/anaconda3/envs/ovtr/bin/python",
                str((ROOT / "scripts/trackocd_v2/build_common_features.py").resolve()),
                "--split",
                "pred",
                "--workers",
                "1",
            ],
        },
        "termination": {
            "method": "explicit_pid_SIGTERM",
            "signal": "SIGTERM",
            "worker_then_builder": True,
            "builder_returncode": 143,
            "graceful": True,
            "pkill_used": False,
            "killall_used": False,
            "external_user_processes_touched": False,
        },
        "artifacts": {
            "completed_atomic_artifacts_retained": True,
            "partial_cache_is_not_formal_ready": True,
            "partial_cache_logical_path": str(OUTPUT_TARGET / "features/pred_tracks/pred"),
            "partial_cache_canonical_path": "/home/lwr/trackocd_v2_cache/pred",
            "relocation_audit": str(relocation.resolve()),
            "relocation_status": "COMPLETE" if relocation.exists() else "NOT_FOUND",
        },
        "predicted_source": {
            "path": str(source.resolve()),
            "sha256": sha256_file(source) if source.exists() else None,
            "tracks": 649378,
            "observations": 1853369,
        },
        "feature_route": {
            "semantic_representation_required": True,
            "full_dino_precompute_required_now": False,
            "frontend_selection_precedes_formal_cache": True,
            "test_semantic_accessed": False,
        },
    }
    atomic_json(out / "audit/predicted_feature_pause.json", result)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Record the implemented support-selection/alignment call graph.

This is a read-only contract audit.  It does not run the controller, alter
candidate sets, or create transformed support boxes.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase84/audit/support_alignment_callgraph.json"
CODE = ROOT / "scripts/iclr27_phase84/evaluate_b84s_event_replay.py"
SOURCE_CACHE = Path("/data2/usr_for_deadline/trackocd_phase84/project_outputs/manifests/source_track_native_vectors.npz")
NATIVE_FEATURES = Path("/data2/usr_for_deadline/trackocd_phase83/a2_dino_full_r1/merged/native_dinov2.npz")
OBS = Path("/data2/usr_for_deadline/trackocd_phase75b/observability_repair2/event_observability.jsonl")
RA_REPLAY = ROOT / "outputs/iclr27_phase84/metrics/b84s_event_replay_b84sra_v1.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def ref(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "exists": path.exists(), "sha256": sha(path) if path.is_file() else None}


def main() -> None:
    ra = json.loads(RA_REPLAY.read_text(encoding="utf-8"))
    ra_p16 = {r["polarity"]: r for r in ra["summary"] if r["prefix"] == 16}
    result = {
        "schema_version": "trackocd.phase84.support_alignment_callgraph.v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "AUDITED_NO_ALIGNMENT_IMPLEMENTED",
        "selection_stage": {
            "source": "source cache prefix-16 mean plus fixed M=3 contiguous causal prototypes",
            "target": "native Q0 per-image candidate set grouped by (video_id, image_id), corrected native DINOv2 row feature",
            "candidate_order": "native candidate_rank, physical_track_id, native row index",
            "causal_cutoff": "event target rows supplied by frozen observability manifest; target current/prefix rows only",
            "decision": "B84S/B84S-Q/B84S-RA select a native candidate or use explicit/raw fallback; no support box transformation is performed",
        },
        "posthoc_scoring": {
            "row_iou_source": "evaluate_b84s_event_replay.iou(native bbox, GT bbox looked up by event row_key)",
            "transformed_iou": "NOT_IMPLEMENTED",
            "support_transformation": "NOT_IMPLEMENTED",
            "gt_used_at_inference": False,
            "event_labels_posthoc_only": True,
        },
        "inputs": {
            "selection_code": ref(CODE),
            "source_cache": ref(SOURCE_CACHE),
            "target_native_features": ref(NATIVE_FEATURES),
            "causal_observability": ref(OBS),
        },
        "boundaries": {
            "future_rows_or_tracks": False,
            "ids_as_model_input": False,
            "category_text_as_input": False,
            "public_dev_q1_sealed_accessed": False,
            "controller_run": False,
            "proposal_source_modified": False,
        },
        "routing": {
            "selection_threshold_for_alignment": "registered meaningful selection >30/76 positive reliable events",
            "b84s_ra_prefix16_positive_reliable": ra_p16["positive"]["selected_reliable_events"],
            "b84s_ra_prefix16_negative_reliable": ra_p16["negative"]["selected_reliable_events"],
            "alignment_authorized": False,
            "reason": f"B84S-RA improves the raw source-mean diagnostic {ra_p16['positive']['raw_source_mean_reliable_events']}->{ra_p16['positive']['selected_reliable_events']} but remains below the registered >30/76 meaningful route criterion; no alignment training or controller replay is authorized.",
        },
    }
    atomic_json(OUT, result)
    print(json.dumps({"out": str(OUT.resolve()), "status": result["status"], "alignment_authorized": False}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

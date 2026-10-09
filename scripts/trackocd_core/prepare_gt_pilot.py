#!/usr/bin/env python3
"""Prepare a bounded Train Known GT plan; no encoder, training or Val/Test."""
from __future__ import annotations

import datetime as dt
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.trackocd_core.audit_assets import DATASET, file_record, read_annotation
from src.trackocd_core.pilot import select_pilot
from src.trackocd_v2.io import atomic_json, sha256_file


def main() -> int:
    started = time.monotonic()
    config_path = ROOT / "configs/trackocd_core/gt_feasibility_pilot.json"
    config = json.loads(config_path.read_text())
    roles_path = ROOT / "configs/trackocd_core/roles.json"
    known = set(json.loads(roles_path.read_text())["known_ids"])
    annotation_path = DATASET / "annotations/train.json"
    selected, summary = select_pilot(read_annotation(annotation_path, "train"), known, config)
    rows = []
    for row in selected:
        observations = []
        for ob in row["observations"]:
            ann, image = ob["annotation"], ob["image"]
            path = DATASET / "frames" / image["file_name"]
            if not path.is_file():
                raise ValueError("Missing registered Train image")
            x, y, w, h = map(float, ann["bbox"])
            if not all(__import__('math').isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
                raise ValueError("Nonfinite/degenerate Train GT box")
            observations.append({"frame_id": int(image["frame_index"]), "image_id": int(image["id"]),
                                 "image_path": image["file_name"], "bbox_xyxy": [x, y, x + w, y + h]})
        rows.append({k: row[k] for k in ("video_id", "gt_track_id", "category", "partition", "purpose", "simulation_role")}
                    | {"observations": observations})
    plan_path = ROOT / "outputs/trackocd_core/pilots/gt_train_known/selection_plan.json"
    if plan_path.exists():
        raise RuntimeError("Pilot plan exists: preserve its preregistration, do not overwrite")
    records = {"config": file_record(config_path), "annotation": file_record(annotation_path), "roles": file_record(roles_path),
               "visual_config": file_record(ROOT / config["visual_protocol"])}
    plan = {"schema_version": "trackocd.core.gt-pilot-plan.v1", "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            **records, "summary": summary, "rows": rows}
    atomic_json(plan_path, plan)
    public = {"schema_version": "trackocd.core.gt-pilot-preregistration.v1",
              "status": config["status"], "generated_utc": plan["generated_utc"], **records, "selection": summary,
              "private_plan": {"bytes": plan_path.stat().st_size, "sha256": sha256_file(plan_path)},
              "source_sha256": {name: sha256_file(ROOT / name) for name in
                                ("scripts/trackocd_core/prepare_gt_pilot.py", "src/trackocd_core/pilot.py")},
              "labels_for_sampling_only_not_model_input": True, "inference_started": False, "training_started": False,
              "val_or_test_access": False, "formal_predicted_result": False,
              "resources": {"wall_seconds": time.monotonic() - started,
                            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "workers": 1}}
    atomic_json(ROOT / "outputs/trackocd_core/audit/gt_pilot_preregistration.json", public)
    print(json.dumps({"status": public["status"], "selection": summary, "resources": public["resources"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Freeze Phase19R semantic predictions from an explicit label-free model manifest.

Unlike ``scripts/iclr27_phase19r/freeze_predictions.py``, this entry point has
no fallback path.  The caller must provide the versioned Phase74S model
manifest, and evaluator labels are never opened by this process.  Scoring is a
separate post-freeze operation over ``evaluator_join_v2.jsonl``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.iclr27_phase19r.data.stream import Phase19RData  # noqa: E402
from src.iclr27_phase19r.models.controller import RCMSOCD  # noqa: E402
from src.iclr27_phase19r.runtime.runner import ModelStreamController  # noqa: E402
from src.iclr27_phase74s.io import atomic_json, sha256  # noqa: E402


MODEL_FIELDS = {"model_event_uid", "source_tracklet_keys", "target_tracklet_key", "source_video", "target_video"}
FORBIDDEN = {"event_key", "category", "kind", "role", "polarity", "fold", "semantic_id", "physical_id", "target_first_reliable_prefix_index_gt_only"}


def load_model_events(path: Path, expected_count: int) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"explicit --model-event-manifest is missing: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(value)
    if len(rows) != expected_count:
        raise ValueError(f"model manifest count {len(rows)} != expected {expected_count}")
    if len({row.get("model_event_uid") for row in rows}) != len(rows):
        raise ValueError("model manifest has duplicate opaque UIDs")
    for index, row in enumerate(rows):
        if set(row) != MODEL_FIELDS:
            raise ValueError(f"row {index} fields {sorted(row)} are not exactly {sorted(MODEL_FIELDS)}")
        if any(key in row for key in FORBIDDEN):
            raise ValueError(f"row {index} contains forbidden evaluator/ID field")
        if not isinstance(row["source_tracklet_keys"], list) or not row["source_tracklet_keys"]:
            raise ValueError(f"row {index} has no source tracklet")
        if any(str(key).startswith("v") is False for key in row["source_tracklet_keys"]):
            raise ValueError(f"row {index} has malformed source tracklet")
    return rows


def scrub_decision(rec: dict[str, Any], position: int, row_key: str) -> dict[str, Any]:
    states = []
    for state in rec.get("states", []):
        states.append({key: state.get(key) for key in ("sid", "birth_video", "birth_track", "count", "dispersion", "age", "impurity_count", "anchor_count")})
    return {
        "row_key": row_key,
        "position": int(position),
        "action": rec.get("action"),
        "semantic_id": rec.get("semantic_id"),
        "known_index": rec.get("known_index"),
        "quality": float(rec.get("quality", 0.0)),
        "confidence": float(rec.get("confidence", rec.get("selected_confidence", 0.0))),
        "state_count": int(rec.get("state_count", len(states))),
        "candidate_sids": [int(x) for x in rec.get("candidate_sids", [])],
        "states": states,
    }


def model_controller(data: Phase19RData, checkpoint: Path, device: torch.device) -> ModelStreamController:
    checkpoint_obj = torch.load(checkpoint, map_location="cpu")
    model = RCMSOCD(torch.from_numpy(data.known_prototypes), torch.from_numpy(data.active_known_mask), max_states=16, known_bias=torch.from_numpy(data.known_bias))
    model.load_state_dict(checkpoint_obj["model_state"])
    model.to(device).eval()
    return ModelStreamController(model, max_states=16, allow_defer=True, tau_ready=model.tau_ready, tau_known=model.tau_known, tau_assign=model.tau_assign)


def run_event(controller: ModelStreamController, data: Phase19RData, event: dict[str, Any], device: torch.device) -> dict[str, Any]:
    controller.reset_stream()
    known_mask = torch.from_numpy(data.active_known_mask).to(device)
    source_decisions: list[dict[str, Any]] = []
    for key in event["source_tracklet_keys"]:
        for position in range(len(data.track_rows[key])):
            raw, geom, quality, _ = data.prefix(key, position)
            row = data.rows[data.track_rows[key][position]]
            got = controller.process_item(torch.from_numpy(raw).to(device), torch.from_numpy(geom).to(device), quality, int(row["video_id"]), key, known_mask, oracle_category=None)
            source_decisions.append(scrub_decision(got, position, row["row_key"]))
    target_decisions: list[dict[str, Any]] = []
    key = event["target_tracklet_key"]
    for position in range(len(data.track_rows[key])):
        raw, geom, quality, _ = data.prefix(key, position)
        row = data.rows[data.track_rows[key][position]]
        got = controller.process_item(torch.from_numpy(raw).to(device), torch.from_numpy(geom).to(device), quality, int(row["video_id"]), key, known_mask, oracle_category=None)
        target_decisions.append(scrub_decision(got, position, row["row_key"]))
    return {
        "model_event_uid": event["model_event_uid"],
        "source_decisions": source_decisions,
        "target_decisions": target_decisions,
        "source_tracklet_keys": event["source_tracklet_keys"],
        "target_tracklet_key": event["target_tracklet_key"],
        "source_video": int(event["source_video"]),
        "target_video": int(event["target_video"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-event-manifest", type=Path, required=True, help="required label-free model manifest; no fallback is permitted")
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/iclr27_phase74s/replay/predictions_v2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expected-events", type=int, default=152)
    args = parser.parse_args()
    events = load_model_events(args.model_event_manifest, args.expected_events)
    if not args.final_checkpoint.is_file():
        raise FileNotFoundError(args.final_checkpoint)
    device = torch.device(args.device)
    data = Phase19RData(final=True)
    controller = model_controller(data, args.final_checkpoint, device)
    records = [run_event(controller, data, event, device) for event in events]
    payload = {
        "protocol": "trackocd_phase74s_prediction_freeze_v2",
        "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_event_manifest": str(args.model_event_manifest.resolve()),
        "model_event_manifest_sha256": sha256(args.model_event_manifest),
        "event_count": len(records),
        "checkpoint": str(args.final_checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.final_checkpoint),
        "labels_joined_before_freeze": False,
        "evaluator_join_read": False,
        "records": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "main_v2.json", payload)
    atomic_json(args.output_dir / "freeze_v2.json", {key: value for key, value in payload.items() if key != "records"} | {"prediction_sha256": sha256(args.output_dir / "main_v2.json")})
    marker = args.output_dir / "predictions_v2.frozen"
    marker.write_text("frozen\n", encoding="utf-8")
    print(json.dumps({"frozen": True, "event_count": len(records), "prediction": str(args.output_dir / "main_v2.json")}, sort_keys=True))


if __name__ == "__main__":
    main()

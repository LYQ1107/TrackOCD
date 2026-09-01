"""Freeze Phase19R public predictions without joining evaluator labels.

The event manifest used here contains only track keys, prefix positions and video
ids.  Category/kind fields stay in the evaluator-only Phase18 files and are not
read by this script.  The frozen marker is written atomically after prediction
files and their hashes have been committed; scoring is a separate post-freeze
operation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.iclr27_phase19r.data.stream import Phase19RData
from src.iclr27_phase19r.models.controller import RCMSOCD
from src.iclr27_phase19r.models.known_osr import GaussianController, RawPersistentController, TALONStyleController
from src.iclr27_phase19r.runtime.runner import ModelStreamController


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase19r"
MANIFEST = OUT / "manifests/public_model_events.jsonl"
MARKER = OUT / "completion/public_predictions.frozen"


def sha(path: Path) -> str:
    h = hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()


def sha_many(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(str(path).encode()); h.update(path.read_bytes())
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def public_events() -> list[dict[str, Any]]:
    if MANIFEST.exists():
        text = MANIFEST.read_text()
        if text.lstrip().startswith("["):
            return list(json.loads(text))
        return [json.loads(x) for x in text.splitlines() if x.strip()]
    # Build a label-free view once from the fixed Phase18 event membership.  No
    # category, event kind or correctness field is retained in this artifact.
    src = ROOT / "data/iclr27_phase19r/sources"
    rows: list[dict[str, Any]] = []
    for name in ("positive_events.jsonl", "negative_events.jsonl"):
        for line in (src / name).read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            rows.append({"event_key": e["event_key"],
                         "source_tracklet_keys": e["source_tracklet_keys"],
                         "target_tracklet_key": e["target_tracklet_key"],
                         "source_video": int(e["source_video"]),
                         "target_video": int(e["target_video"]),
                         "target_first_reliable_prefix_index_gt_only": int(e["target_first_reliable_prefix_index_gt_only"])})
    rows.sort(key=lambda x: x["event_key"])
    tmp = MANIFEST.with_name(MANIFEST.name + ".tmp")
    tmp.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))
    os.replace(tmp, MANIFEST)
    return rows


def scrub_decision(rec: dict[str, Any], position: int, row_key: str) -> dict[str, Any]:
    states = []
    for s in rec.get("states", []):
        states.append({k: s.get(k) for k in
                       ("sid", "birth_video", "birth_track", "count", "dispersion", "age", "impurity_count", "anchor_count")})
    return {"row_key": row_key, "position": int(position), "action": rec.get("action"),
            "semantic_id": rec.get("semantic_id"), "known_index": rec.get("known_index"), "quality": float(rec.get("quality", 0.0)),
            "confidence": float(rec.get("confidence", rec.get("selected_confidence", 0.0))),
            "state_count": int(rec.get("state_count", len(states))),
            "candidate_sids": [int(x) for x in rec.get("candidate_sids", [])], "states": states}


def model_controller(data: Phase19RData, checkpoint: Path, device: torch.device) -> ModelStreamController:
    ck = torch.load(checkpoint, map_location="cpu")
    model = RCMSOCD(torch.from_numpy(data.known_prototypes), torch.from_numpy(data.active_known_mask),
                   max_states=16, known_bias=torch.from_numpy(data.known_bias))
    model.load_state_dict(ck["model_state"]); model.to(device).eval()
    return ModelStreamController(model, max_states=16, allow_defer=True,
                                 tau_ready=model.tau_ready, tau_known=model.tau_known,
                                 tau_assign=model.tau_assign)


def run_event(controller: Any, data: Phase19RData, event: dict[str, Any], device: torch.device,
              model_mode: bool) -> dict[str, Any]:
    if hasattr(controller, "reset_stream"):
        controller.reset_stream()
    known_mask = torch.from_numpy(data.active_known_mask).to(device)
    source_out: list[dict[str, Any]] = []
    for key in event["source_tracklet_keys"]:
        for pos in range(len(data.track_rows[key])):
            raw, geom, quality, _ = data.prefix(key, pos)
            row = data.rows[data.track_rows[key][pos]]
            if model_mode:
                got = controller.process_item(torch.from_numpy(raw).to(device), torch.from_numpy(geom).to(device),
                                              quality, int(row["video_id"]), key, known_mask, oracle_category=None)
            else:
                got = controller.process_track(key, phase="source", eval_category=None)
                source_out.extend(got)
                break
            source_out.append(scrub_decision(got, pos, row["row_key"]))
    target_key = event["target_tracklet_key"]
    target_out: list[dict[str, Any]] = []
    if model_mode:
        for pos in range(len(data.track_rows[target_key])):
            raw, geom, quality, _ = data.prefix(target_key, pos)
            row = data.rows[data.track_rows[target_key][pos]]
            got = controller.process_item(torch.from_numpy(raw).to(device), torch.from_numpy(geom).to(device),
                                          quality, int(row["video_id"]), target_key, known_mask, oracle_category=None)
            target_out.append(scrub_decision(got, pos, row["row_key"]))
    else:
        target_out = controller.process_track(target_key, phase="target", eval_category=None)
    final_states = []
    for s in getattr(controller, "memory", None).states if getattr(controller, "memory", None) is not None else []:
        final_states.append({k: s.get(k) for k in
                             ("sid", "birth_video", "birth_track", "count", "dispersion", "age", "impurity_count", "anchor_count")}
                            if isinstance(s, dict) else {"sid": int(s.sid), "birth_video": int(s.birth_video),
                            "birth_track": str(s.birth_track), "count": int(s.count),
                            "dispersion": float(s.dispersion), "age": int(s.age),
                            "impurity_count": int(s.impurity_count), "anchor_count": len(s.anchors)})
    return {"event_key": event["event_key"], "source_decisions": source_out,
            "target_decisions": target_out, "target_first_reliable_prefix_index_gt_only": int(event["target_first_reliable_prefix_index_gt_only"]),
            "source_tracklet_keys": event["source_tracklet_keys"], "target_tracklet_key": target_key,
            "source_video": int(event["source_video"]), "target_video": int(event["target_video"]),
            "final_states": final_states}


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--device", default="cpu"); p.add_argument("--final-checkpoint", type=Path, required=True)
    args = p.parse_args(); device = torch.device(args.device); data = Phase19RData(final=True); events = public_events()
    candidates: dict[str, Any] = {"raw": None, "age": None, "talon": None,
                                  "main": args.final_checkpoint, "fallback_f_a": None}
    prediction_hashes: dict[str, str] = {}; metadata: dict[str, Any] = {}
    for name, ck in candidates.items():
        if name == "raw": controller: Any = RawPersistentController(data, deferred=True); is_model = False
        elif name == "age": controller = GaussianController(data, deferred=True); is_model = False
        elif name == "talon": controller = TALONStyleController(data, deferred=True); is_model = False
        elif name == "fallback_f_a": controller = GaussianController(data, deferred=True); is_model = False
        else: controller = model_controller(data, ck, device); is_model = True
        payload = {"protocol": "trackocd_iclr27_phase19r_public_raw_prediction",
                   "candidate": name, "event_count": len(events),
                   "records": [run_event(controller, data, e, device, is_model) for e in events]}
        path = OUT / "public_predictions" / f"{name}_raw.json"; atomic_json(path, payload); prediction_hashes[name] = sha(path)
        metadata[name] = {"path": str(path), "sha256": prediction_hashes[name],
                          "checkpoint": str(ck) if ck else None,
                          "checkpoint_sha256": sha(ck) if ck else None}
    freeze = {"protocol": "trackocd_iclr27_phase19r_prediction_freeze",
              "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "event_manifest": str(MANIFEST), "event_manifest_sha256": sha(MANIFEST),
              "event_count": len(events), "candidates": metadata,
              "config": "configs/iclr27_phase19r/preregistered_main.json",
              "config_sha256": sha(ROOT / "configs/iclr27_phase19r/preregistered_main.json"),
              "code_sha256": sha_many([ROOT / "src/iclr27_phase19r/models/controller.py", ROOT / "src/iclr27_phase19r/runtime/state.py", ROOT / "src/iclr27_phase19r/runtime/runner.py", ROOT / "scripts/iclr27_phase19r/freeze_predictions.py"]),
              "known_stage": "outputs/iclr27_phase19r/checkpoints/known_stage_final.npz",
              "known_stage_sha256": sha(OUT / "checkpoints/known_stage_final.npz") if (OUT / "checkpoints/known_stage_final.npz").exists() else None,
              "feature": "frozen Phase19R DINOv2 0.8 CLS + 0.2 ROI",
              "labels_joined_before_freeze": False,
              "public_scoring_allowed_after_marker": str(MARKER)}
    atomic_json(OUT / "manifests/prediction_freeze.json", freeze)
    MARKER.parent.mkdir(parents=True, exist_ok=True); tmp = MARKER.with_name(MARKER.name + ".tmp"); tmp.write_text("frozen\n"); os.replace(tmp, MARKER)
    print(json.dumps({"frozen": True, "event_count": len(events), "prediction_hashes": prediction_hashes}, sort_keys=True))


if __name__ == "__main__":
    main()

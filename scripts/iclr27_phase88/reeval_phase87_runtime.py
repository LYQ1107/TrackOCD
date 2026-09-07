#!/usr/bin/env python3
"""Replay Phase87 checkpoints through the corrected Phase88 runtime only."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import torch
torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs" / "iclr27_phase88"
OLD = Path("/data2/usr_for_deadline/trackocd_phase87/project_outputs/checkpoints")

from src.iclr27_phase19r.data.stream import Phase19RData
from src.iclr27_phase88.controller import CausalPersistentOCD
from src.iclr27_phase88.evaluation import evaluate_persistent_events, normalize_event


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    output = {"schema_version": "trackocd.phase88.phase87_full_runtime_replay.v1", "phase": 88, "source_phase": 87, "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_or_text_as_model_input": False, "routes": []}
    pos = read(ROOT / "outputs/iclr27_phase19r/manifests/held_known_positive_events.jsonl")
    neg = read(ROOT / "outputs/iclr27_phase19r/manifests/held_known_negative_events.jsonl")
    for route in ("c0_formal", "c1_formal"):
        aggregate = []
        for fold in range(4):
            ckpt = OLD / f"{route}_f{fold}.pt"
            data = Phase19RData(fold)
            model = CausalPersistentOCD(torch.from_numpy(data.known_prototypes), torch.from_numpy(data.active_known_mask), max_states=16)
            payload = torch.load(ckpt, map_location="cpu")
            state = dict(payload.get("model", payload.get("model_state", {})))
            state.pop("known_prototypes", None)
            state.pop("active_known_mask", None)
            model.load_state_dict(state, strict=False)
            model.to(device).eval()
            events = [normalize_event(e) for e in pos + neg if int(e.get("fold", -1)) == fold]
            result = evaluate_persistent_events(model, data, events, device, support_mode=True)
            aggregate.append({"fold": fold, "checkpoint": str(ckpt), "checkpoint_sha256": sha(ckpt), "events": len(events), "metrics": result["metrics"], "known_metrics": result["known_metrics"], "diagnostic": result["diagnostic"]})
        output["routes"].append({"route": route, "folds": aggregate})
    path = OUT / "audit/phase87_full_runtime_replay.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

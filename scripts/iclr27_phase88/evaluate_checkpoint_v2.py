#!/usr/bin/env python3
"""Bounded TRAIN-disjoint validation for an intermediate Phase88 checkpoint."""
from __future__ import annotations

import argparse
import datetime as dt
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
OUT = ROOT / "outputs/iclr27_phase88"

from src.iclr27_phase19r.data.stream import Phase19RData
from src.iclr27_phase88.controller import CausalPersistentOCD
from src.iclr27_phase88.evaluation import evaluate_persistent_events
from src.iclr27_phase88.evaluation import normalize_event


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cuda:5")
    ap.add_argument("--max-events", type=int, default=0)
    args = ap.parse_args()
    ckpt = Path(args.checkpoint).resolve()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    data = Phase19RData(args.fold)
    payload = torch.load(ckpt, map_location=device)
    model = CausalPersistentOCD(torch.from_numpy(data.known_prototypes), torch.from_numpy(data.active_known_mask), max_states=16).to(device)
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    events = [normalize_event(e) for e in read_jsonl(OUT / "manifests" / f"val_events_v2_f{args.fold}.jsonl")]
    if args.max_events > 0:
        events = events[:args.max_events]
    with torch.no_grad():
        result = evaluate_persistent_events(model, data, events, device, support_mode=True)
    step = int(payload.get("step", 0))
    out = {
        "schema_version": "trackocd.phase88.intermediate_validation.v1",
        "phase": 88, "tag": args.tag, "fold": args.fold, "step": step,
        "checkpoint": str(ckpt), "checkpoint_sha256": sha(ckpt),
        "split": "TRAIN_disjoint_validation", "events": len(events),
        "device": str(device), "metrics": result["metrics"],
        "known_metrics": result["known_metrics"], "diagnostic": result["diagnostic"],
        "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False,
        "ids_or_text_as_model_input": False,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    root = OUT / "validation" / args.tag
    atomic_json(root / f"step_{step:06d}_metrics.json", out)
    atomic_json(root / f"step_{step:06d}.done", {"status": "DONE", "metrics": str((root / f"step_{step:06d}_metrics.json").resolve())})
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
